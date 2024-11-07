# https://github.com/radis/radis

```console
setup.py:        # packages=find_packages(), #this misses the gpu folder, see https://github.com/radis/radis/issues/681
setup.py:            'nvidia-cufft-cu11; sys_platform != "darwin" ',
README.rst:GPU Acceleration
README.rst:RADIS supports GPU acceleration for super-fast computation of spectra. Refer to `GPU Spectrum Calculation on RADIS <https://radis.readthedocs.io/en/latest/lbl/lbl.html#calculating-spectrum-using-gpu>`__ for more details on GPU acceleration.::
setup.cfg:    needs_cuda: this requires CUDA Installed (deselect with '-m "not needs_cuda"')
setup.cfg:addopts = -m "not needs_cuda and not download_large_databases and not needs_db_CDSD_HITEMP and not needs_db_CDSD_HITEMP_PCN and not needs_db_CDSD_HITEMP_PC and not needs_db_HITEMP_CO2_DUNHAM and not needs_db_HITEMP_CO_DUNHAM"
docs/lbl/lbl.rst:Calculating spectrum using GPU
docs/lbl/lbl.rst:RADIS also supports CUDA-native parallel computation, specifically
docs/lbl/lbl.rst:for lineshape calculation and broadening. To use these GPU-accelerated methods to compute the spectra, use either :py:func:`~radis.lbl.calc.calc_spectrum`
docs/lbl/lbl.rst:function with parameter `mode` set to `gpu`, or :py:meth:`~radis.lbl.factory.SpectrumFactory.eq_spectrum_gpu`. In order to use these methods,
docs/lbl/lbl.rst:ensure that your system has an Nvidia GPU with compute capability of at least 3.0 and CUDA Toolkit 8.0 or above. Refer to
docs/lbl/lbl.rst::ref:`GPU Spectrum Calculation on RADIS <label_radis_gpu>` to see how to setup your system to run GPU accelerated spectrum
docs/lbl/lbl.rst:Currently, GPU-powered spectra calculations are supported only at thermal equilibrium
docs/lbl/lbl.rst:and therefore, the method to calculate the spectra has been named :py:meth:`~radis.lbl.factory.SpectrumFactory.eq_spectrum_gpu`.
docs/lbl/lbl.rst:function set the parameter `mode` to `gpu`, or use :py:meth:`~radis.lbl.factory.SpectrumFactory.eq_spectrum_gpu`
docs/lbl/lbl.rst:One could compute the spectra with the assistance of GPU using the
docs/lbl/lbl.rst:        	mode='gpu'
docs/lbl/lbl.rst:Refer to :ref:`GPU Spectrum Calculation on RADIS <label_radis_gpu>` for more details.
docs/lbl/gpu.rst:.. _label_radis_gpu:
docs/lbl/gpu.rst:RADIS-GPU Spectrum Calculation
docs/lbl/gpu.rst:RADIS provides GPU acceleration to massively speedup spectral computations.
docs/lbl/gpu.rst:Currently only Nvidia GPU's are supported, but this will likely change in the future
docs/lbl/gpu.rst:Generally GPU computations are memory bandwidth limited, meaning the computation time of
docs/lbl/gpu.rst:(=CPU) to device (=GPU) memory. Because of this, GPU computations take place in two steps:
docs/lbl/gpu.rst:An initialization step :py:func:`~radis.gpu.gpu.gpu_init` where, among other things, the database is
docs/lbl/gpu.rst:uploaded to the GPU, and an iteration step :py:func:`~radis.gpu.gpu.gpu_iterate`, where a new spectrum
docs/lbl/gpu.rst:RADIS implements two functions that expose GPU functionality:
docs/lbl/gpu.rst:- :py:func:`~radis.lbl.factory.SpectrumFactory.eq_spectrum_gpu` computes a single spectrum and returns.
docs/lbl/gpu.rst:- :py:func:`~radis.lbl.factory.SpectrumFactory.eq_spectrum_gpu_interactive` computes a single
docs/lbl/gpu.rst:By default both functions will be ran on a GPU if available. The CUDA code can also be compiled as pure
docs/lbl/gpu.rst:C++, which means it can be compiled for CPU in addition to GPU.
docs/lbl/gpu.rst:As a result, it ispossible to use the same GPU functions without an actual GPU by passing the
docs/lbl/gpu.rst:keyword ``backend='cpu-cuda'``, which forces use of the CPU targeted compiled code. This feature is
docs/lbl/gpu.rst:mostly for developers to check for errors in the CUDA code, but it can also be used for interactive
docs/lbl/gpu.rst:GPU computation is currently only supported for equilibrium spectra. It is likely that
docs/lbl/gpu.rst:As mentioned above, the function :py:func:`~radis.lbl.factory.SpectrumFactory.eq_spectrum_gpu`
docs/lbl/gpu.rst:produces a single equilibrium spectrum using GPU acceleration. Below is a usage example::
docs/lbl/gpu.rst:    s = sf.eq_spectrum_gpu(
docs/lbl/gpu.rst:.. minigallery:: radis.lbl.SpectrumFactory.eq_spectrum_gpu
docs/lbl/gpu.rst:As mentioned before, computing the first GPU spectrum in a session takes a comparatively long time because the
docs/lbl/gpu.rst:entire database must be transferred to the GPU. The real power of GPU acceleration
docs/lbl/gpu.rst::py:func:`~radis.lbl.factory.eq_spectrum_gpu_interactive()`. A usage example is shown below::
docs/lbl/gpu.rst:    s = sf.eq_spectrum_gpu_interactive(
docs/lbl/gpu.rst:        emulate=False,  # runs on GPU
docs/lbl/gpu.rst:.. minigallery:: radis.lbl.SpectrumFactory.eq_spectrum_gpu_interactive
docs/lbl/gpu.rst:Note that `eq_spectrum_gpu_interactive()` replaces all of `eq_spectrum_gpu()`,
docs/lbl/gpu.rst:`eq_spectrum_gpu_interactive()` to specify which spectrum should be plotted, and keyword arguments to `s.plot()`
docs/lbl/gpu.rst: provided the GPU didn't run out of memory.
docs/lbl/gpu.rst:will also move to the GPU at some point in the future.
docs/lbl/gpu.rst:Did you miss any feature implemented on GPU? or support for your particular system? The GPU code is heavily under development, so drop us a visit on [our Githup](https://github.com/radis/radis/issues/616) and let us know what you're looking for!
docs/dev/_architecture.rst:  GPU calculation can be done with :py:meth:`~radis.lbl.factory.SpectrumFactory.eq_spectrum_gpu`
.gitattributes:radis/gpu/build_kernels binary
.gitattributes:radis/gpu/*.so binary
radis/spectrum/spectrum.py:from radis.misc.warning import GPUInitWarning
radis/spectrum/spectrum.py:    def recalc_gpu(
radis/spectrum/spectrum.py:        for spectrum objects produced by :py:meth:`~radis.lbl.factory.SpectrumFctory.eq_spectrum_gpu`.
radis/spectrum/spectrum.py:        This method is used internally by :py:meth:`~radis.lbl.factory.SpectrumFctory.eq_spectrum_gpu_interactive`.
radis/spectrum/spectrum.py:        after which spectrum.recalc_gpu() may be called without passing arguments.
radis/spectrum/spectrum.py:            s = sf.eq_spectrum_gpu(
radis/spectrum/spectrum.py:                I.append(s.recalc_gpu('radiance', Tgas=T)
radis/spectrum/spectrum.py:            self.conditions["gpu_backend"]
radis/spectrum/spectrum.py:                "GPU not initialized, spectrum.recalc_gpu() can only be called on spectrum objects produced by sf.eq_spectrum_gpu()!",
radis/spectrum/spectrum.py:                GPUInitWarning,
radis/spectrum/spectrum.py:        from radis.gpu.gpu import gpu_iterate
radis/spectrum/spectrum.py:        abscoeff, iter_params, times = gpu_iterate(
radis/spectrum/spectrum.py:            # TODO: GPU apply_slit not supported yet
radis/test/gpu/test_gpu.py:from radis.misc.utils import NotInstalled, not_installed_nvidia_args
radis/test/gpu/test_gpu.py:from radis.misc.warning import NoGPUWarning
radis/test/gpu/test_gpu.py:    from nvidia.cufft import __path__ as cufft_path
radis/test/gpu/test_gpu.py:    cufft_path = NotInstalled(*not_installed_nvidia_args)
radis/test/gpu/test_gpu.py:    reason="nvidia package not installed. Probably because on MAC OS",
radis/test/gpu/test_gpu.py:def test_eq_spectrum_emulated_gpu(
radis/test/gpu/test_gpu.py:    backend="cpu-cuda", verbose=False, plot=False, *args, **kwargs
radis/test/gpu/test_gpu.py:    """Compare Spectrum calculated in the emulated-GPU code
radis/test/gpu/test_gpu.py:    :py:func:`radis.lbl.factory.SpectrumFactory.eq_spectrum_gpu` to Spectrum
radis/test/gpu/test_gpu.py:    # Compare GPU & CPU without sparse-LDM (not implemented in GPU yet)
radis/test/gpu/test_gpu.py:    s_gpu = sf.eq_spectrum_gpu(
radis/test/gpu/test_gpu.py:        name="GPU (emulate)" if backend == "cpu-cuda" else "GPU",
radis/test/gpu/test_gpu.py:    s_gpu.name += f"[{s_gpu.c['calculation_time']:.2f}s]"
radis/test/gpu/test_gpu.py:    s_gpu.crop(wmin=2284.2, wmax=2284.8)
radis/test/gpu/test_gpu.py:        s_cpu.compare_with(s_gpu, spectra_only=True, plot=plot)
radis/test/gpu/test_gpu.py:    assert get_residual(s_cpu, s_gpu, "abscoeff") < 1.4e-5
radis/test/gpu/test_gpu.py:    assert get_residual(s_cpu, s_gpu, "radiance_noslit") < 7.3e-6
radis/test/gpu/test_gpu.py:    assert get_residual(s_cpu, s_gpu, "transmittance_noslit") < 1.4e-5
radis/test/gpu/test_gpu.py:        print(s_gpu)
radis/test/gpu/test_gpu.py:        s_gpu.print_perf_profile()
radis/test/gpu/test_gpu.py:    reason="nvidia package not installed. Probably because on MAC OS",
radis/test/gpu/test_gpu.py:@pytest.mark.needs_cuda
radis/test/gpu/test_gpu.py:def test_eq_spectrum_gpu(plot=False, *args, **kwargs):
radis/test/gpu/test_gpu.py:    """Compare Spectrum calculated in the GPU code
radis/test/gpu/test_gpu.py:    :py:func:`radis.lbl.factory.SpectrumFactory.eq_spectrum_gpu` to Spectrum
radis/test/gpu/test_gpu.py:    # Ensure that GPU is not deactivated (which triggers a NoGPUWarning)
radis/test/gpu/test_gpu.py:        warnings.simplefilter("error", category=NoGPUWarning)
radis/test/gpu/test_gpu.py:        test_eq_spectrum_emulated_gpu(backend="gpu-cuda", plot=plot, *args, **kwargs)
radis/test/gpu/test_gpu.py:    reason="nvidia package not installed. Probably because on MAC OS",
radis/test/gpu/test_gpu.py:def test_multiple_gpu_calls():
radis/test/gpu/test_gpu.py:    s1_gpu = sf.eq_spectrum_gpu(
radis/test/gpu/test_gpu.py:        Tgas=300, backend="gpu-cuda", diluent={"air": 0.99}  # K  # runs on GPU
radis/test/gpu/test_gpu.py:    s2_gpu = sf.eq_spectrum_gpu(
radis/test/gpu/test_gpu.py:        Tgas=300, backend="gpu-cuda", diluent={"air": 0.99}  # K  # runs on GPU
radis/test/gpu/test_gpu.py:    assert abs(s1_gpu.get_power() - s2_gpu.get_power()) / s1_gpu.get_power() < 1e-5
radis/test/gpu/test_gpu.py:    assert s1_gpu.get_power() > 0
radis/test/gpu/test_gpu.py:    # test_eq_spectrum_gpu(plot=True)
radis/test/gpu/test_gpu.py:    test_eq_spectrum_emulated_gpu(plot=True, verbose=2)
radis/test/gpu/test_gpu.py:    printm("Testing GPU spectrum calculation:", pytest.main(["test_gpu.py"]))
radis/tools/new_fitting.py:    # s_model = sf.eq_spectrum_gpu(**kwargs)
radis/tools/plot_tools.py:        r"""Used in :py:func:`radis.lbl.factory.SpectrumFactory.eq_spectrum_gpu_interactive`"""
radis/lbl/factory.py:        gpu_backend=None,
radis/lbl/factory.py:    def eq_spectrum_gpu(
radis/lbl/factory.py:        backend="gpu-cuda",
radis/lbl/factory.py:        exit_gpu=True,
radis/lbl/factory.py:        and broadening done on the GPU.
radis/lbl/factory.py:            This method requires CUDA compatible hardware to execute.
radis/lbl/factory.py:            For more information on how to setup your system to run GPU-accelerated methods
radis/lbl/factory.py:            using CUDA and Cython, check :ref:`GPU Spectrum Calculation on RADIS <label_radis_gpu>`
radis/lbl/factory.py:            if ``'gpu-cuda'``, set CUDA as backend to run code on Nvidia GPU.
radis/lbl/factory.py:            if ``'cpu-cuda'``, execute the GPU code on the CPU (useful for development)
radis/lbl/factory.py:        .. minigallery:: radis.lbl.SpectrumFactory.eq_spectrum_gpu
radis/lbl/factory.py:        :meth:`~radis.lbl.factory.SpectrumFactory.eq_spectrum_gpu_interactive`
radis/lbl/factory.py:                "eq_spectrum_gpu hasn't been implemented for atomic spectra"
radis/lbl/factory.py:                ].unique()  # get all the molecules in the dataframe, should ideally be 1 element for GPU
radis/lbl/factory.py:        from radis.gpu.gpu import gpu_exit, gpu_init, gpu_iterate
radis/lbl/factory.py:        gpu_init(
radis/lbl/factory.py:        abscoeff_calc, iter_params, times = gpu_iterate(
radis/lbl/factory.py:        # If sf.eq_spectrum_gpu() was called directly by the user, this is the time to
radis/lbl/factory.py:        # destroy the CUDA context since we're done with all GPU calculations.
radis/lbl/factory.py:        # When called from within sf.eq_spectrum_gpu_interactive(), the context must remain active
radis/lbl/factory.py:        # because more calls to gpu_iterate() will follow. This is controlled by the exit_gpu keyword.
radis/lbl/factory.py:        if exit_gpu:
radis/lbl/factory.py:            gpu_exit()
radis/lbl/factory.py:        # TODO: this should is inconistent with eq_spectrum_gpu_interactive, where all quantities
radis/lbl/factory.py:        #      are calculated on CPU. (here the transmittance comes from GPU)
radis/lbl/factory.py:                "gpu_backend": backend,
radis/lbl/factory.py:                "add_at_used": "gpu-backend",
radis/lbl/factory.py:    def eq_spectrum_gpu_interactive(
radis/lbl/factory.py:            arguments forwarded to :py:meth:`~radis.lbl.factory.SpectrumFactory.eq_spectrum_gpu`
radis/lbl/factory.py:            arguments forwarded to :py:meth:`~radis.lbl.factory.SpectrumFactory.eq_spectrum_gpu`
radis/lbl/factory.py:            s = sf.eq_spectrum_gpu_interactive(Tgas=ParamRange(300.0,2000.0,1200.0), #K
radis/lbl/factory.py:        .. minigallery:: radis.lbl.factory.SpectrumFactory.eq_spectrum_gpu_interactive
radis/lbl/factory.py:                "eq_spectrum_gpu hasn't been implemented for atomic spectra"
radis/lbl/factory.py:        from radis.gpu.gpu import gpu_exit
radis/lbl/factory.py:        kwargs["exit_gpu"] = False
radis/lbl/factory.py:        s = self.eq_spectrum_gpu(*vargs, **kwargs)
radis/lbl/factory.py:            # be passed to s.recalc_gpu() anymore
radis/lbl/factory.py:            new_y = s.recalc_gpu(
radis/lbl/factory.py:        fig.canvas.mpl_connect("close_event", gpu_exit)
radis/lbl/calc.py:    using either CPU or GPU.
radis/lbl/calc.py:    mode: ``'cpu'``, ``'gpu'``, ``'emulated_gpu'``
radis/lbl/calc.py:        if set to ``'cpu'``, computes the spectra purely on the CPU. if set to ``'gpu'``,
radis/lbl/calc.py:        offloads the calculations of lineshape and broadening steps to the GPU
radis/lbl/calc.py:        Note that ``mode='gpu'`` requires CUDA compatible hardware to execute.
radis/lbl/calc.py:        For more information on how to setup your system to run GPU-accelerated
radis/lbl/calc.py:        methods using CUDA and Cython, check `GPU Spectrum Calculation on RADIS <https://radis.readthedocs.io/en/latest/lbl/gpu.html>`__
radis/lbl/calc.py:        To try the GPU code without an actual GPU, you can use ``mode='emulated_gpu'``.
radis/lbl/calc.py:        This will run the GPU equivalent code on the CPU.
radis/lbl/calc.py:    ​.. [2] RADIS GPU support: `GPU Calculations on RADIS <https://radis.readthedocs.io/en/latest/lbl/gpu.html>`__
radis/lbl/calc.py:                          mode='gpu'
radis/lbl/calc.py:    This example uses the :py:meth:`~radis.lbl.factory.SpectrumFactory.eq_spectrum_gpu` method to calculate
radis/lbl/calc.py:    the spectrum on the GPU. The databank points to the CDSD-4000 databank that has been
radis/lbl/calc.py:    For more details on how to use the GPU method and process the database, refer to the examples
radis/lbl/calc.py:    linked above and the documentation on :ref:`GPU support for RADIS <label_radis_gpu>`.
radis/lbl/calc.py:        elif mode in ("gpu", "emulated_gpu"):
radis/lbl/calc.py:            s = sf.eq_spectrum_gpu(
radis/lbl/calc.py:                backend=("cpu-cuda" if mode == "emulated_gpu" else "gpu-cuda")
radis/lbl/calc.py:                # emulate=(True if mode == "emulated_gpu" else False),
radis/lbl/calc.py:                f"mode= should be one of 'cpu', 'gpu', 'emulated_gpu' (GPU code running on CPU). Got {mode}"
radis/lbl/base.py:        Currently this value is used in GPU calculations.
radis/lbl/base.py:        It is one of the columns that is transferred to the GPU
radis/misc/utils.py:not_installed_nvidia_args = (
radis/misc/utils.py:    "nvidia-cufft",
radis/misc/utils.py:    "Nvidia was not installed on your computer. `nvidia-cufft` is"
radis/misc/warning.py:class NoGPUWarning(PerformanceWarning):
radis/misc/warning.py:    """Triggered when GPU doesn't work"""
radis/misc/warning.py:class GPUInitWarning(PerformanceWarning):
radis/misc/warning.py:    """Triggered when GPU isn't initialized"""
radis/misc/warning.py:    "NoGPUWarning": NoGPUWarning,
radis/misc/warning.py:    "GPUInitWarning": GPUInitWarning,
radis/misc/warning.py:    "NoGPUWarning": "warn",
radis/misc/warning.py:    "GPUInitWarning": "warn",
radis/gpu/gpu.py:from radis.gpu.params import (
radis/gpu/gpu.py:from radis.gpu.structs import initData_t, iterData_t
radis/gpu/gpu.py:from radis.misc.warning import NoGPUWarning
radis/gpu/gpu.py:gpu_mod = None
radis/gpu/gpu.py:def gpu_init(
radis/gpu/gpu.py:    backend="gpu-cuda",
radis/gpu/gpu.py:    Initialize GPU-based calculation for emission and absorption spectra in spectroscopy.
radis/gpu/gpu.py:    backend :  ``'gpu-cuda'``, ``'cpu-cuda'``, optional
radis/gpu/gpu.py:        Which backend to use; currently only CUDA backends (Nvidia) are supported. ``'cpu-cuda'`` runs the kernel on CPU. Default is ``'gpu-cuda'``.
radis/gpu/gpu.py:    init_h : radis.gpu.structs.initData_t
radis/gpu/gpu.py:        structue with parameters used for GPU computation that are constant
radis/gpu/gpu.py:    global gpu_mod
radis/gpu/gpu.py:    if gpu_mod is not None:
radis/gpu/gpu.py:        warn("Only a single GPU context allowed; please call gpu_exit() first.")
radis/gpu/gpu.py:    ## First a GPU context is created, then the .ptx file is read
radis/gpu/gpu.py:    ## and made available as the GPUModule object gpu_mod
radis/gpu/gpu.py:    if backend == "cpu-cuda":
radis/gpu/gpu.py:        from radis.gpu.cuda.emulate import CuContext as GPUContext
radis/gpu/gpu.py:        ctx = GPUContext.Open(verbose=verbose)
radis/gpu/gpu.py:        import radis.gpu.cuda.emulate as backend_module
radis/gpu/gpu.py:        # Try to load GPU
radis/gpu/gpu.py:        from radis.gpu.cuda.driver import CuContext as GPUContext
radis/gpu/gpu.py:        ctx = GPUContext.Open(verbose=verbose)  # Set verbose to >=2 for comments
radis/gpu/gpu.py:                NoGPUWarning(
radis/gpu/gpu.py:                    "Failed to load CUDA context, this happened either because"
radis/gpu/gpu.py:                    + "CUDA is not installed properly, or you have no NVIDIA GPU. "
radis/gpu/gpu.py:                    + "Continuing with emulated GPU on CPU..."
radis/gpu/gpu.py:                    + "This means *NO* GPU acceleration!"
radis/gpu/gpu.py:            # failed to init CUDA context, continue with CPU:
radis/gpu/gpu.py:            from radis.gpu.cuda.emulate import CuContext as GPUContext
radis/gpu/gpu.py:            ctx = GPUContext.Open(verbose=verbose)
radis/gpu/gpu.py:            import radis.gpu.cuda.emulate as backend_module
radis/gpu/gpu.py:            # successfully initialized CUDA context, continue with GPU:
radis/gpu/gpu.py:            import radis.gpu.cuda.driver as backend_module
radis/gpu/gpu.py:    GPUContext, GPUModule, GPUArray, GPUFFT, GPUTimer = backend_module.getClasses()
radis/gpu/gpu.py:    ptx_path = os.path.join(getProjectRoot(), "gpu", "cuda", "build", "kernels.ptx")
radis/gpu/gpu.py:    gpu_mod = GPUModule(ctx, ptx_path)  # gpu
radis/gpu/gpu.py:        print("mode:", gpu_mod.getMode())
radis/gpu/gpu.py:    ## Next, the GPU is made aware of a number of parameters.
radis/gpu/gpu.py:    ## in init_h. They are copied to the GPU through gpu_mod.setConstant()
radis/gpu/gpu.py:    gpu_mod.setConstant("init_d", init_h)
radis/gpu/gpu.py:    ## Next the block- and thread size of the GPU kernels are set.
radis/gpu/gpu.py:    ## This determines how the GPU internally divides up the work.
radis/gpu/gpu.py:    gpu_mod.fillLDM.setGrid((Nli // Ntpb + 1, 1, 1), threads)
radis/gpu/gpu.py:    gpu_mod.applyLineshapes.setGrid((NxFT // Ntpb + 1, 1, 1), threads)
radis/gpu/gpu.py:    gpu_mod.calcTransmittanceNoslit.setGrid((NvFT // Ntpb + 1, 1, 1), threads)
radis/gpu/gpu.py:    gpu_mod.applyGaussianSlit.setGrid((NxFT // Ntpb + 1, 1, 1), threads)
radis/gpu/gpu.py:    ## Next the variables are initialized on the GPU. Constant variables
radis/gpu/gpu.py:    ## copied to the GPU through GPUArray.fromArray().
radis/gpu/gpu.py:    S_klm_d = GPUArray(0, dtype=np.float32, grow_only=True)
radis/gpu/gpu.py:    S_klm_FT_d = GPUArray(0, dtype=np.complex64, grow_only=True)
radis/gpu/gpu.py:    spectrum_in_d = GPUArray(NxFT, dtype=np.complex64)
radis/gpu/gpu.py:    spectrum_out_d = GPUArray(NvFT, dtype=np.float32)
radis/gpu/gpu.py:    transmittance_noslit_d = GPUArray(NvFT, dtype=np.float32)
radis/gpu/gpu.py:    transmittance_noslit_FT_d = GPUArray(NxFT, dtype=np.complex64)
radis/gpu/gpu.py:    transmittance_FT_d = GPUArray(NxFT, dtype=np.complex64)
radis/gpu/gpu.py:    transmittance_d = GPUArray(NvFT, dtype=np.float32)
radis/gpu/gpu.py:    gpu_mod.fillLDM.setArgs(
radis/gpu/gpu.py:        GPUArray.fromArray(iso),
radis/gpu/gpu.py:        GPUArray.fromArray(v0),
radis/gpu/gpu.py:        GPUArray.fromArray(da),
radis/gpu/gpu.py:        GPUArray.fromArray(S0),
radis/gpu/gpu.py:        GPUArray.fromArray(El),
radis/gpu/gpu.py:        GPUArray.fromArray(gamma_arr),
radis/gpu/gpu.py:        GPUArray.fromArray(na),
radis/gpu/gpu.py:    gpu_mod.applyLineshapes.setArgs(S_klm_FT_d, spectrum_in_d)
radis/gpu/gpu.py:    gpu_mod.calcTransmittanceNoslit.setArgs(spectrum_out_d, transmittance_noslit_d)
radis/gpu/gpu.py:    gpu_mod.applyGaussianSlit.setArgs(transmittance_noslit_FT_d, transmittance_FT_d)
radis/gpu/gpu.py:    ## FFT's are performed through the GPUFFT object. The required functions are internally
radis/gpu/gpu.py:    ## reuse the work area, we make a GPUArray at this scope that is passed to the
radis/gpu/gpu.py:    ## GPUFFT objects. The work area will be scaled according to needs by the GPUFFT objects,
radis/gpu/gpu.py:    workarea_d = GPUArray(0, dtype=np.byte, grow_only=True)
radis/gpu/gpu.py:    gpu_mod.fft_fwd = GPUFFT(S_klm_d, S_klm_FT_d, workarea=workarea_d, direction="fwd")
radis/gpu/gpu.py:    gpu_mod.fft_rev = GPUFFT(
radis/gpu/gpu.py:    gpu_mod.fft_fwd2 = GPUFFT(
radis/gpu/gpu.py:    gpu_mod.fft_rev2 = GPUFFT(
radis/gpu/gpu.py:    gpu_mod.timer = GPUTimer()
radis/gpu/gpu.py:def gpu_iterate(
radis/gpu/gpu.py:    # for GPU instrument functions (not currently supported):
radis/gpu/gpu.py:    iter_h : radis.gpu.structs.iterData_t
radis/gpu/gpu.py:        different stages of the GPU computation. The ``'total'`` key
radis/gpu/gpu.py:    if gpu_mod is None:
radis/gpu/gpu.py:        warn("Must have an open GPU context; please call gpu_init() first.")
radis/gpu/gpu.py:    ## are computed and copied to the GPU.
radis/gpu/gpu.py:    gpu_mod.timer.reset()
radis/gpu/gpu.py:    gpu_mod.setConstant("iter_d", iter_h)
radis/gpu/gpu.py:    gpu_mod.timer.lap("iter_params")
radis/gpu/gpu.py:    gpu_mod.fillLDM.args[-1].resize(S_klm_shape, init="zeros")
radis/gpu/gpu.py:    gpu_mod.fillLDM()
radis/gpu/gpu.py:    gpu_mod.timer.lap("fillLDM")
radis/gpu/gpu.py:    gpu_mod.fft_fwd.arr_out.resize(S_klm_FT_shape)
radis/gpu/gpu.py:    gpu_mod.fft_fwd()
radis/gpu/gpu.py:    gpu_mod.timer.lap("fft_fwd")
radis/gpu/gpu.py:    gpu_mod.applyLineshapes()
radis/gpu/gpu.py:    gpu_mod.timer.lap("applyLineshapes")
radis/gpu/gpu.py:    gpu_mod.fft_rev()
radis/gpu/gpu.py:    gpu_mod.timer.lap("fft_rev")
radis/gpu/gpu.py:    abscoeff_h = gpu_mod.fft_rev.arr_out.getArray()[: init_h.N_v]
radis/gpu/gpu.py:    ##    ##The code below is to process slits on the GPU, which is currently unsupported.
radis/gpu/gpu.py:    ##    gpu_mod.calcTransmittanceNoslit()
radis/gpu/gpu.py:    ##    gpu_mod.timer.lap("calcTransmittanceNoslit")
radis/gpu/gpu.py:    ##    gpu_mod.fft_fwd2()
radis/gpu/gpu.py:    ##    gpu_mod.timer.lap("fft_fwd2")
radis/gpu/gpu.py:    ##    gpu_mod.applyGaussianSlit()
radis/gpu/gpu.py:    ##    gpu_mod.timer.lap("applyGaussianSlit")
radis/gpu/gpu.py:    ##    gpu_mod.fft_rev2()
radis/gpu/gpu.py:    ##    gpu_mod.timer.lap("fft_rev2")
radis/gpu/gpu.py:    ##    transmittance_h = gpu_mod.fft_rev2.arr_out.getArray()[: init_h.N_v]
radis/gpu/gpu.py:    gpu_mod.timer.lap("total")
radis/gpu/gpu.py:    times = gpu_mod.timer.getTimes()
radis/gpu/gpu.py:    ##    diffs = gpu_mod.timer.getDiffs()
radis/gpu/gpu.py:def gpu_exit(event=None):
radis/gpu/gpu.py:    global gpu_mod
radis/gpu/gpu.py:    gpu_mod.context.destroy()
radis/gpu/gpu.py:    gpu_mod = None
radis/gpu/cuda/emulate.py:from radis.gpu.structs import blockDim_t, gridDim_t
radis/gpu/cuda/emulate.py:# so we just default to a typical number for GPU.
radis/gpu/cuda/emulate.py:        # Verbose output not implemented for GPU emulation
radis/gpu/cuda/emulate.py:        print("> GPU emulated by CPU")
radis/gpu/cuda/emulate.py:            os.path.join(radis_path, "gpu", "cuda", "build", self.module_name)
radis/gpu/cuda/kernels.cu:#include "gpu_cpu_agnostic.h"
radis/gpu/cuda/build/kernels.ptx:// Generated by NVIDIA NVVM Compiler
radis/gpu/cuda/build/kernels.ptx:// Cuda compilation tools, release 11.8, V11.8.89
radis/gpu/cuda/driver.py:from radis.misc.utils import NotInstalled, not_installed_nvidia_args
radis/gpu/cuda/driver.py:    from nvidia.cufft import __path__ as cufft_path
radis/gpu/cuda/driver.py:    cufft_path = NotInstalled(*not_installed_nvidia_args)
radis/gpu/cuda/driver.py:def getCUDAVersion(ptx_file):
radis/gpu/cuda/driver.py:    # Reads the version of the CUDA Toolkit that was used to compile PTX file
radis/gpu/cuda/driver.py:    # Returns the minimal required CUDA driver version
radis/gpu/cuda/driver.py:    major, minor, patch = getCUDAVersion(ptx_file)
radis/gpu/cuda/driver.py:CUDA_SUCCESS = 0
radis/gpu/cuda/driver.py:        cuda_name = "nvcuda.dll" if os_name == "nt" else "libcuda.so"
radis/gpu/cuda/driver.py:            lib = dllobj.LoadLibrary(cuda_name)
radis/gpu/cuda/driver.py:                print("Can't find {:s}...".format(cuda_name))
radis/gpu/cuda/driver.py:                cuda_name = "libcuda.so.1"
radis/gpu/cuda/driver.py:                    lib = dllobj.LoadLibrary(cuda_name)
radis/gpu/cuda/driver.py:                    print("Can't find {:s}...".format(cuda_name))
radis/gpu/cuda/driver.py:            print("Error: no devices supporting CUDA\n")
radis/gpu/cuda/driver.py:        if err != CUDA_SUCCESS:
radis/gpu/cuda/driver.py:            print("Error initializing the CUDA context.")
radis/gpu/cuda/driver.py:            "> GPU Device has SM {:d}.{:d} compute capability".format(
radis/gpu/cuda/driver.py:        self.mode = "GPU"
radis/gpu/cuda/driver.py:        if err != CUDA_SUCCESS:
radis/gpu/cuda/driver.py:                    "\n\n*** CUDA Driver too old!***\nMinimally required driver version is {:s}\n".format(
radis/gpu/cuda/driver.py:                    + "Please update driver by downloading it at:\n-> www.nvidia.com/download/index.aspx \n"
radis/gpu/cuda/driver.py:            if err != CUDA_SUCCESS:
radis/gpu/cuda/gpu_cpu_agnostic.h:#ifdef __CUDACC__
radis/gpu/cuda/gpu_cpu_agnostic.h:#include <cuda/std/complex>
radis/gpu/cuda/gpu_cpu_agnostic.h:using namespace cuda::std;
radis/gpu/README:This folder contains all files related to GPU calculations of RADIS spectra.
radis/gpu/README:At the moment, only Nvidia cards are supported.
radis/gpu/README:Nvidia uses CUDA for general purpose GPU (GPGPU) programming.
radis/gpu/README:Most applications require the entire CUDA Toolkit installed in order to run.
radis/gpu/README:Instead, RADIS interfaces directly with the CUDA GPU driver, and can therefore run without any additional installations.
radis/gpu/README:The code that runs on the GPU is written in CUDA C and resides in the kernels.cu file.
radis/gpu/README:The CUDA driver loads the compiled .ptx file at runtime to launch the GPU kernels through driver.py.
requirements.txt:nvidia-cufft-cu11; sys_platform != "darwin"
MANIFEST.in:include radis/gpu/cuda/*.cu
MANIFEST.in:include radis/gpu/cuda/*.h
MANIFEST.in:include radis/gpu/cuda/build/*.*
examples/4_GPU/plot_gpu.py:GPU Accelerated Spectra
examples/4_GPU/plot_gpu.py:Example using GPU calculation with :py:meth:`~radis.lbl.SpectrumFactory.eq_spectrum_gpu`
examples/4_GPU/plot_gpu.py:This method requires a GPU - Currently, only Nvidia GPU's are supported.
examples/4_GPU/plot_gpu.py:For more information on how to setup your system to run GPU-accelerated methods
examples/4_GPU/plot_gpu.py:using CUDA, check :ref:`GPU Spectrum Calculation on RADIS <label_radis_gpu>`
examples/4_GPU/plot_gpu.py:    in the example below, the code runs on the GPU by default. In case no Nvidia GPU is
examples/4_GPU/plot_gpu.py:    the ``backend`` keyword either to ``'gpu-cuda'`` or ``'cpu-cuda'``.
examples/4_GPU/plot_gpu.py:s_gpu = sf.eq_spectrum_gpu(
examples/4_GPU/plot_gpu.py:    name="GPU",
examples/4_GPU/plot_gpu.py:    backend="gpu-cuda",
examples/4_GPU/plot_gpu.py:s_gpu.apply_slit(w_slit, unit="cm-1")
examples/4_GPU/plot_gpu.py:plot_diff(s_cpu, s_gpu, var="radiance", wunit="nm", method="diff")
examples/4_GPU/README.rst:GPU calculations
examples/4_GPU/README.rst:Unleash the power of your GPU to speed up your calculations.
examples/4_GPU/plot_gpu_recalc.py:GPU Accelerated Spectra (recalc_gpu() demo)
examples/4_GPU/plot_gpu_recalc.py:Example using GPU calculation with :py:meth:`~radis.spectrum.spectrum.Spectrum.recalc_gpu`
examples/4_GPU/plot_gpu_recalc.py:After producing a spectrum object with sf.eq_spectrum_gpu(), new spectra can be produced quickly
examples/4_GPU/plot_gpu_recalc.py:with spectrum.recalc_gpu().
examples/4_GPU/plot_gpu_recalc.py:    make sure you pass ``exit_gpu=False`` when producing the spectrum object, otherwise
examples/4_GPU/plot_gpu_recalc.py:    it will destroy the GPU context which is needed for spectrum.recalc_gpu().
examples/4_GPU/plot_gpu_recalc.py:    Also be sure to call gpu_exit() at the end.
examples/4_GPU/plot_gpu_recalc.py:from radis.gpu.gpu import gpu_exit
examples/4_GPU/plot_gpu_recalc.py:s = sf.eq_spectrum_gpu(
examples/4_GPU/plot_gpu_recalc.py:    exit_gpu=False,
examples/4_GPU/plot_gpu_recalc.py:s.recalc_gpu(Tgas=T_list[1])
examples/4_GPU/plot_gpu_recalc.py:s.recalc_gpu(Tgas=T_list[2])
examples/4_GPU/plot_gpu_recalc.py:gpu_exit()
examples/4_GPU/plot_gpu_widgets.py:.. _example_real_time_gpu_spectra:
examples/4_GPU/plot_gpu_widgets.py:Real-time GPU Accelerated Spectra (Interactive)
examples/4_GPU/plot_gpu_widgets.py:Example using GPU sliders and GPU calculation with :py:meth:`~radis.lbl.SpectrumFactory.eq_spectrum_gpu_intereactive`
examples/4_GPU/plot_gpu_widgets.py:This method requires a GPU - Currently, only Nvidia GPU's are supported.
examples/4_GPU/plot_gpu_widgets.py:For more information on how to setup your system to run GPU-accelerated methods
examples/4_GPU/plot_gpu_widgets.py:using CUDA, check :ref:`GPU Spectrum Calculation on RADIS <label_radis_gpu>`
examples/4_GPU/plot_gpu_widgets.py:    by running python in interactive mode as follows: ``python -i plot_gpu_widgets.py``.
examples/4_GPU/plot_gpu_widgets.py:    in the example below, the code runs on the GPU by default. In case no Nvidia GPU is
examples/4_GPU/plot_gpu_widgets.py:    the ``backend`` keyword either to ``'gpu-cuda'`` or ``'cpu-cuda'``.
examples/4_GPU/plot_gpu_widgets.py:s = sf.eq_spectrum_gpu_interactive(
examples/3_Fitting/plot4_legacyFit_Tgas.py:Finally, the :ref:`GPU-accelerated example<example_real_time_gpu_spectra>` shows
examples/3_Fitting/plot5_legacyFit_Trot-Tvib.py:Finally, the :ref:`GPU-accelerated example<example_real_time_gpu_spectra>` shows
.travis.yml:  - xvfb-run -a pytest -m "not fast and not needs_cuda and not download_large_databases and not needs_db_CDSD_HITEMP and not needs_db_CDSD_HITEMP_PCN and not needs_db_CDSD_HITEMP_PC and not needs_db_HITEMP_CO2_DUNHAM and not needs_db_HITEMP_CO_DUNHAM" --durations=10
.travis.yml:        # - pytest -m "not fast and not needs_cuda and not download_large_databases and not needs_db_CDSD_HITEMP and not needs_db_CDSD_HITEMP_PCN and not needs_db_CDSD_HITEMP_PC and not needs_db_HITEMP_CO2_DUNHAM and not needs_db_HITEMP_CO_DUNHAM" --durations=10
.travis.yml:      - xvfb-run -a pytest -m "fast and not needs_cuda and not download_large_databases and not needs_db_CDSD_HITEMP and not needs_db_CDSD_HITEMP_PCN and not needs_db_CDSD_HITEMP_PC and not needs_db_HITEMP_CO2_DUNHAM and not needs_db_HITEMP_CO_DUNHAM" --durations=10
.gitignore:!radis/gpu/cuda/build/*.so
.gitignore:!radis/gpu/cuda/build/

```
