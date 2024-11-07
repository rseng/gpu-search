# https://github.com/PyFstat/PyFstat

```console
setup.py:    "pycuda": ["pycuda"],
setup.py:            "pyCUDAkernels/cudaTransientFstatExpWindow.cu",
setup.py:            "pyCUDAkernels/cudaTransientFstatRectWindow.cu",
.zenodo.json:    "pycuda"
joss-paper/paper.bib:   title = "{PyCUDA and PyOpenCL: A Scripting-Based Approach to GPU Run-Time Code Generation}",
joss-paper/paper.bib:    title = "{Faster search for long gravitational-wave transients: GPU implementation of the transient $\mathcal F$-statistic}",
joss-paper/arxiv/paper-arxiv-v2.tex:\texttt{PyCUDA} (Klöckner et al. 2012) for speedup, is discussed in
joss-paper/arxiv/paper-arxiv-v2.tex:with modern technologies like GPUs or machine learning. Hence,
joss-paper/arxiv/paper-arxiv-v2.tex:  speeding up long-transient searches with GPUs (Keitel and Ashton
joss-paper/arxiv/paper-arxiv-v2.tex:gravitational-wave transients: GPU implementation of the transient
joss-paper/arxiv/paper-arxiv-v2.tex:and Ahmed Fasih. 2012. ``PyCUDA and PyOpenCL: A Scripting-Based Approach
joss-paper/arxiv/paper-arxiv-v2.tex:to GPU Run-Time Code Generation.'' \emph{Parallel Computing} 38 (3):
joss-paper/arxiv/paper-arxiv-v1.tex:\texttt{PyCUDA} (Klöckner et al. 2012) for speedup, is discussed in
joss-paper/arxiv/paper-arxiv-v1.tex:with modern technologies like GPUs or machine learning. Hence,
joss-paper/arxiv/paper-arxiv-v1.tex:  speeding up long-transient searches with GPUs (Keitel and Ashton
joss-paper/arxiv/paper-arxiv-v1.tex:gravitational-wave transients: GPU implementation of the transient
joss-paper/arxiv/paper-arxiv-v1.tex:and Ahmed Fasih. 2012. ``PyCUDA and PyOpenCL: A Scripting-Based Approach
joss-paper/arxiv/paper-arxiv-v1.tex:to GPU Run-Time Code Generation.'' \emph{Parallel Computing} 38 (3):
joss-paper/paper.md:The extension to transient signals, which uses `PyCUDA` [@Kloeckner:2012pyc] for speedup,
joss-paper/paper.md:like GPUs or machine learning.
joss-paper/paper.md:- speeding up long-transient searches with GPUs [@Keitel:2018pxz];
CHANGELOG.md: - Transient F-stat GPU implementation:
CHANGELOG.md:   - Improved CUDA device info logging.
CHANGELOG.md: - fix CUDA context detaching at garbage collection time for
CHANGELOG.md: - transient-on-GPU output file writing fix
CHANGELOG.md:- pycuda as optional dependency
tests/test_tcw_fstat_map_funcs.py:@pytest.mark.parametrize("tCWFstatMapVersion", ["lal", "pycuda"])
tests/test_tcw_fstat_map_funcs.py:    if tCWFstatMapVersion == "pycuda" and not features[tCWFstatMapVersion]:
tests/test_tcw_fstat_map_funcs.py:        gpu_context,
tests/test_tcw_fstat_map_funcs.py:    if gpu_context:
tests/test_tcw_fstat_map_funcs.py:        logging.info("Detaching GPU context...")
tests/test_tcw_fstat_map_funcs.py:        gpu_context.detach()
tests/test_core.py:@pytest.mark.parametrize("tCWFstatMapVersion", ["lal", "pycuda"])
tests/test_core.py:    if cleanup == "manual" and not tCWFstatMapVersion == "pycuda":
tests/test_core.py:        pytest.skip("Manual cleanup won't work in non-pycuda case.")
tests/test_core.py:    # if GPU available, try the real thing;
tests/test_core.py:    # but without actually trying to run on GPU
tests/test_core.py:    if tCWFstatMapVersion == "pycuda":
tests/test_core.py:        have_pycuda = pyfstat.tcw_fstat_map_funcs._optional_imports_pycuda()
tests/test_core.py:        if not have_pycuda:
tests/test_core.py:            pytest.skip("Optional imports failed, skipping actual pycuda test.")
tests/test_core.py:            if tCWFstatMapVersion == "pycuda":
tests/test_core.py:        if tCWFstatMapVersion == "pycuda":
README.md:* `pycuda`: Required for the `tCWFstatMapVersion=pycuda`
README.md:  (Note: Installing the `pycuda` package,
README.md:  see e.g. on [PyPI](https://pypi.org/project/pycuda/),
README.md:For example, installing PyFstat including `chainconsumer`, `pycuda` and `style` dependencies would look like
README.md:pip install pyfstat[chainconsumer,pycuda,style]
examples/transient_examples/PyFstat_example_short_transient_grid_search.py:This is also ready to use on a GPU,
examples/transient_examples/PyFstat_example_short_transient_grid_search.py:if you have one available and `pycuda` installed.
examples/transient_examples/PyFstat_example_short_transient_grid_search.py:Just change to `tCWFstatMapVersion = "pycuda"`.
pyfstat/tcw_fstat_map_funcs.py:for a detailed discussion of the GPU implementation.
pyfstat/tcw_fstat_map_funcs.py:    when CUDA_DEVICE is set to too high a number.
pyfstat/tcw_fstat_map_funcs.py:    "pycuda": lambda multiFstatAtoms, windowRange, BtSG: pycuda_compute_transient_fstat_map(
pyfstat/tcw_fstat_map_funcs.py:def _optional_imports_pycuda():
pyfstat/tcw_fstat_map_funcs.py:    have_pycuda = _optional_import("pycuda")
pyfstat/tcw_fstat_map_funcs.py:    have_pycuda_drv = _optional_import("pycuda.driver", "drv")
pyfstat/tcw_fstat_map_funcs.py:    have_pycuda_gpuarray = _optional_import("pycuda.gpuarray", "gpuarray")
pyfstat/tcw_fstat_map_funcs.py:    have_pycuda_tools = _optional_import("pycuda.tools", "cudatools")
pyfstat/tcw_fstat_map_funcs.py:    have_pycuda_compiler = _optional_import("pycuda.compiler", "cudacomp")
pyfstat/tcw_fstat_map_funcs.py:        have_pycuda
pyfstat/tcw_fstat_map_funcs.py:        and have_pycuda_drv
pyfstat/tcw_fstat_map_funcs.py:        and have_pycuda_gpuarray
pyfstat/tcw_fstat_map_funcs.py:        and have_pycuda_tools
pyfstat/tcw_fstat_map_funcs.py:        and have_pycuda_compiler
pyfstat/tcw_fstat_map_funcs.py:    features["pycuda"] = _optional_imports_pycuda()
pyfstat/tcw_fstat_map_funcs.py:def init_transient_fstat_map_features(feature="lal", cudaDeviceName=None):
pyfstat/tcw_fstat_map_funcs.py:    2. `pycuda`: requires the `pycuda` package to be importable
pyfstat/tcw_fstat_map_funcs.py:    `driver`, `gpuarray`, `tools` and `compiler`.
pyfstat/tcw_fstat_map_funcs.py:    cudaDeviceName: str or None
pyfstat/tcw_fstat_map_funcs.py:        Request a CUDA device with this name.
pyfstat/tcw_fstat_map_funcs.py:    gpu_context: pycuda.driver.Context or None
pyfstat/tcw_fstat_map_funcs.py:        A CUDA device context object, if assigned.
pyfstat/tcw_fstat_map_funcs.py:    if feature == "pycuda":
pyfstat/tcw_fstat_map_funcs.py:        if not features["pycuda"]:
pyfstat/tcw_fstat_map_funcs.py:            raise RuntimeError("pycuda use was requested, but imports failed.")
pyfstat/tcw_fstat_map_funcs.py:        logger.info("CUDA version: " + ".".join(map(str, drv.get_version())))
pyfstat/tcw_fstat_map_funcs.py:            "Starting with default pyCUDA context,"
pyfstat/tcw_fstat_map_funcs.py:            context0 = pycuda.tools.make_default_context()
pyfstat/tcw_fstat_map_funcs.py:        except pycuda._driver.LogicError as e:
pyfstat/tcw_fstat_map_funcs.py:                devn = int(os.environ["CUDA_DEVICE"])
pyfstat/tcw_fstat_map_funcs.py:                    "Requested CUDA device number {} exceeds"
pyfstat/tcw_fstat_map_funcs.py:                    " variable $CUDA_DEVICE.".format(devn)
pyfstat/tcw_fstat_map_funcs.py:                raise pycuda._driver.LogicError(e.message)
pyfstat/tcw_fstat_map_funcs.py:        num_gpus = drv.Device.count()
pyfstat/tcw_fstat_map_funcs.py:        logger.info("Found {} CUDA device(s).".format(num_gpus))
pyfstat/tcw_fstat_map_funcs.py:        devnames = np.empty(num_gpus, dtype="S32")
pyfstat/tcw_fstat_map_funcs.py:        for n in range(num_gpus):
pyfstat/tcw_fstat_map_funcs.py:        if "CUDA_DEVICE" in os.environ:
pyfstat/tcw_fstat_map_funcs.py:            devnum0 = int(os.environ["CUDA_DEVICE"])
pyfstat/tcw_fstat_map_funcs.py:        if cudaDeviceName:
pyfstat/tcw_fstat_map_funcs.py:                if cudaDeviceName in devname
pyfstat/tcw_fstat_map_funcs.py:                    'Requested CUDA device "{}" not found.'
pyfstat/tcw_fstat_map_funcs.py:                        cudaDeviceName, ",".join(devnames)
pyfstat/tcw_fstat_map_funcs.py:                        'Found {} CUDA devices matching name "{}".'
pyfstat/tcw_fstat_map_funcs.py:                            len(devmatches), cudaDeviceName, devnum
pyfstat/tcw_fstat_map_funcs.py:            os.environ["CUDA_DEVICE"] = str(devnum)
pyfstat/tcw_fstat_map_funcs.py:            matchbit = '(matched to user request "{}")'.format(cudaDeviceName)
pyfstat/tcw_fstat_map_funcs.py:        elif "CUDA_DEVICE" in os.environ:
pyfstat/tcw_fstat_map_funcs.py:            devnum = int(os.environ["CUDA_DEVICE"])
pyfstat/tcw_fstat_map_funcs.py:            "Choosing CUDA device {},"
pyfstat/tcw_fstat_map_funcs.py:                devnum, num_gpus, devn.name(), matchbit
pyfstat/tcw_fstat_map_funcs.py:            gpu_context = context0
pyfstat/tcw_fstat_map_funcs.py:            gpu_context = pycuda.tools.make_default_context()
pyfstat/tcw_fstat_map_funcs.py:            gpu_context.push()
pyfstat/tcw_fstat_map_funcs.py:        _print_GPU_memory_MB("Available")
pyfstat/tcw_fstat_map_funcs.py:        gpu_context = None
pyfstat/tcw_fstat_map_funcs.py:    return features, gpu_context
pyfstat/tcw_fstat_map_funcs.py:        (currently supported: 'lal' or 'pycuda').
pyfstat/tcw_fstat_map_funcs.py:    return os.path.join(pyfstatdir, "pyCUDAkernels", kernelfile)
pyfstat/tcw_fstat_map_funcs.py:def _print_GPU_memory_MB(key):
pyfstat/tcw_fstat_map_funcs.py:        "{} GPU memory: {:.4f} / {:.4f} MB free".format(key, mem_used_MB, mem_total_MB)
pyfstat/tcw_fstat_map_funcs.py:def pycuda_compute_transient_fstat_map(multiFstatAtoms, windowRange, BtSG=False):
pyfstat/tcw_fstat_map_funcs.py:    """GPU version of computing a transient F-statistic map.
pyfstat/tcw_fstat_map_funcs.py:    the actual CUDA computations are performed in one of the functions
pyfstat/tcw_fstat_map_funcs.py:    `pycuda_compute_transient_fstat_map_rect()`
pyfstat/tcw_fstat_map_funcs.py:    or `pycuda_compute_transient_fstat_map_exp()`,
pyfstat/tcw_fstat_map_funcs.py:    # make a combined input matrix of all atoms vectors, for transfer to GPU
pyfstat/tcw_fstat_map_funcs.py:        FstatMap.F_mn = pycuda_compute_transient_fstat_map_rect(
pyfstat/tcw_fstat_map_funcs.py:        FstatMap.F_mn = pycuda_compute_transient_fstat_map_exp(
pyfstat/tcw_fstat_map_funcs.py:        # so far seems there is no need to move this onto the GPU
pyfstat/tcw_fstat_map_funcs.py:def pycuda_compute_transient_fstat_map_rect(atomsInputMatrix, windowRange, tCWparams):
pyfstat/tcw_fstat_map_funcs.py:    """GPU computation of the transient F-stat map for rectangular windows.
pyfstat/tcw_fstat_map_funcs.py:    this version only does GPU parallelization for the outer loop,
pyfstat/tcw_fstat_map_funcs.py:    # gpu data setup and transfer
pyfstat/tcw_fstat_map_funcs.py:    _print_GPU_memory_MB("Initial")
pyfstat/tcw_fstat_map_funcs.py:    input_gpu = gpuarray.to_gpu(atomsInputMatrix)
pyfstat/tcw_fstat_map_funcs.py:    Fmn_gpu = gpuarray.GPUArray(
pyfstat/tcw_fstat_map_funcs.py:    _print_GPU_memory_MB("After input+output allocation:")
pyfstat/tcw_fstat_map_funcs.py:    # GPU kernel
pyfstat/tcw_fstat_map_funcs.py:    kernel = "cudaTransientFstatRectWindow"
pyfstat/tcw_fstat_map_funcs.py:    partial_Fstat_cuda_code = cudacomp.SourceModule(open(kernelfile, "r").read())
pyfstat/tcw_fstat_map_funcs.py:    partial_Fstat_cuda = partial_Fstat_cuda_code.get_function(kernel)
pyfstat/tcw_fstat_map_funcs.py:    partial_Fstat_cuda.prepare("PIIIIIIIIP")
pyfstat/tcw_fstat_map_funcs.py:    # GPU grid setup
pyfstat/tcw_fstat_map_funcs.py:        "Calling pyCUDA kernel with a grid of {}*{}={} blocks"
pyfstat/tcw_fstat_map_funcs.py:    partial_Fstat_cuda.prepared_call(
pyfstat/tcw_fstat_map_funcs.py:        input_gpu.gpudata,
pyfstat/tcw_fstat_map_funcs.py:        Fmn_gpu.gpudata,
pyfstat/tcw_fstat_map_funcs.py:    F_mn = Fmn_gpu.get()
pyfstat/tcw_fstat_map_funcs.py:    _print_GPU_memory_MB("Final")
pyfstat/tcw_fstat_map_funcs.py:def pycuda_compute_transient_fstat_map_exp(atomsInputMatrix, windowRange, tCWparams):
pyfstat/tcw_fstat_map_funcs.py:    """GPU computation of the transient F-stat map for exponential windows.
pyfstat/tcw_fstat_map_funcs.py:    this version does full GPU parallelization
pyfstat/tcw_fstat_map_funcs.py:    # gpu data setup and transfer
pyfstat/tcw_fstat_map_funcs.py:    _print_GPU_memory_MB("Initial")
pyfstat/tcw_fstat_map_funcs.py:    input_gpu = gpuarray.to_gpu(atomsInputMatrix)
pyfstat/tcw_fstat_map_funcs.py:    Fmn_gpu = gpuarray.GPUArray(
pyfstat/tcw_fstat_map_funcs.py:    _print_GPU_memory_MB("After input+output allocation:")
pyfstat/tcw_fstat_map_funcs.py:    # GPU kernel
pyfstat/tcw_fstat_map_funcs.py:    kernel = "cudaTransientFstatExpWindow"
pyfstat/tcw_fstat_map_funcs.py:    partial_Fstat_cuda_code = cudacomp.SourceModule(open(kernelfile, "r").read())
pyfstat/tcw_fstat_map_funcs.py:    partial_Fstat_cuda = partial_Fstat_cuda_code.get_function(kernel)
pyfstat/tcw_fstat_map_funcs.py:    partial_Fstat_cuda.prepare("PIIIIIIIIIP")
pyfstat/tcw_fstat_map_funcs.py:    # GPU grid setup
pyfstat/tcw_fstat_map_funcs.py:    partial_Fstat_cuda.prepared_call(
pyfstat/tcw_fstat_map_funcs.py:        input_gpu.gpudata,
pyfstat/tcw_fstat_map_funcs.py:        Fmn_gpu.gpudata,
pyfstat/tcw_fstat_map_funcs.py:    F_mn = Fmn_gpu.get()
pyfstat/tcw_fstat_map_funcs.py:    _print_GPU_memory_MB("Final")
pyfstat/pyCUDAkernels/cudaTransientFstatRectWindow.cu:__global__ void cudaTransientFstatRectWindow ( float *input,
pyfstat/pyCUDAkernels/cudaTransientFstatRectWindow.cu:  /* match CUDA thread indexing and high-level (t0,tau) indexing */
pyfstat/pyCUDAkernels/cudaTransientFstatRectWindow.cu:   * (empirically seems to be faster than 2D CUDA version)
pyfstat/pyCUDAkernels/cudaTransientFstatRectWindow.cu:} // cudaTransientFstatRectWindow()
pyfstat/pyCUDAkernels/cudaTransientFstatExpWindow.cu:__global__ void cudaTransientFstatExpWindow ( float *input,
pyfstat/pyCUDAkernels/cudaTransientFstatExpWindow.cu:  /* match CUDA thread indexing and high-level (t0,tau) indexing */
pyfstat/pyCUDAkernels/cudaTransientFstatExpWindow.cu:} // cudaTransientFstatExpWindow()
pyfstat/grid_based_searches.py:    for a detailed discussion of the GPU implementation.
pyfstat/grid_based_searches.py:    NOTE for GPU users (`tCWFstatMapVersion="pycuda"`):
pyfstat/grid_based_searches.py:    conveniently deal with GPU context management behind the scenes.
pyfstat/grid_based_searches.py:    that is because the GPU is still blocked from the first instance when
pyfstat/grid_based_searches.py:            tCWFstatMapVersion="pycuda",
pyfstat/grid_based_searches.py:        cudaDeviceName=None,
pyfstat/grid_based_searches.py:            `pycuda` for GPU version,
pyfstat/grid_based_searches.py:        cudaDeviceName: str
pyfstat/grid_based_searches.py:            GPU name to be matched against drv.Device output,
pyfstat/grid_based_searches.py:            only for `tCWFstatMapVersion=pycuda`.
pyfstat/grid_based_searches.py:            cudaDeviceName=self.cudaDeviceName,
pyfstat/mcmc_based_searches.py:            'pycuda' for gpu, and some others for devel/debug.
pyfstat/core.py:    NOTE for GPU users (`tCWFstatMapVersion="pycuda"`):
pyfstat/core.py:    This class tries to conveniently deal with GPU context management behind the scenes.
pyfstat/core.py:    that is because the GPU is still blocked from the first instance when
pyfstat/core.py:            tCWFstatMapVersion="pycuda",
pyfstat/core.py:        cudaDeviceName=None,
pyfstat/core.py:            `pycuda` for GPU version,
pyfstat/core.py:        cudaDeviceName: str
pyfstat/core.py:            GPU name to be matched against drv.Device output,
pyfstat/core.py:            only for `tCWFstatMapVersion=pycuda`.
pyfstat/core.py:        Setup for proper cleanup at end of context in pycuda case.
pyfstat/core.py:        if "cuda" in self.tCWFstatMapVersion:
pyfstat/core.py:                f"Setting up GPU context finalizer for {self.tCWFstatMapVersion} transient maps."
pyfstat/core.py:            self._finalizer = finalize(self, self._finalize_gpu_context)
pyfstat/core.py:    def _finalize_gpu_context(self):
pyfstat/core.py:        if hasattr(self, "gpu_context") and self.gpu_context:
pyfstat/core.py:            logger.debug("Detaching GPU context...")
pyfstat/core.py:            # this is needed because we use pyCuda without autoinit
pyfstat/core.py:            self.gpu_context.detach()
pyfstat/core.py:        if "cuda" in self.tCWFstatMapVersion:
pyfstat/core.py:                self.gpu_context,
pyfstat/core.py:                self.tCWFstatMapVersion, self.cudaDeviceName
pyfstat/core.py:        self.cudaDeviceName = None
pyfstat/core.py:        self.cudaDeviceName = None

```
