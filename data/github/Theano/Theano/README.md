# https://github.com/Theano/Theano

```console
setup.py:                       'expressions on CPUs and GPUs.')
setup.py:              'numpy', 'gpu', 'autodiff', 'differentiation'
theano/gradient.py:            # Tensor, Sparse and GpuArray have the ndim attribute
theano/ifelse.py:    def __init__(self, n_outs, as_view=False, gpu=False, name=None):
theano/ifelse.py:        self.gpu = gpu
theano/ifelse.py:        if not self.gpu == other.gpu:
theano/ifelse.py:                hash(self.gpu) ^
theano/ifelse.py:        if self.gpu:
theano/ifelse.py:            args.append('gpu')
theano/ifelse.py:                gpu=False,
theano/ifelse.py:        if not self.gpu:
theano/ifelse.py:            # When gpu is true, we are given only gpuarrays, and we want
theano/ifelse.py:            # to keep them as gpuarrays
theano/ifelse.py:                            gpu=self.gpu,
theano/ifelse.py:                             gpu=self.gpu,
theano/ifelse.py:            # This case happens when one of the elements has a GPU type,
theano/ifelse.py:            # for instance a shared variable that was silently moved to GPU.
theano/ifelse.py:                        gpu=False,
theano/ifelse.py:                      gpu=op.gpu,
theano/ifelse.py:                    gpu=False,
theano/ifelse.py:                        gpu=op.gpu,
theano/ifelse.py:                gpu=False,
theano/scalar/basic.py:            # If we are in a gpuarray kernel, %(fail)s exits the kernel,
theano/scalar/basic.py:            # cuda function, which return a binary pattern of all 1s.
theano/scalar/basic.py:            # If we are in a gpuarray kernel, %(fail)s exits the kernel,
theano/scalar/basic.py:            # cuda function, returning a binary pattern depending on dtype
theano/scalar/tests/test_basic.py:        # local_gpu_elemwise_0 optimization and printed an
theano/scalar/c_code/gamma.c:            2013.11.13 modification to make it work with CUDA
theano/scalar/c_code/gamma.c://For GPU support
theano/scalar/c_code/gamma.c:#ifdef __CUDACC__
theano/scalar/basic_scipy.py:    This op can still be executed on GPU, despite not having c_code. When
theano/scalar/basic_scipy.py:    running on GPU an optimization will replace it with a gpu version.
theano/scalar/basic_scipy.py:    This op can still be executed on GPU, despite not having c_code. When
theano/scalar/basic_scipy.py:    running on GPU, an optimization will replace it with a GPU version.
theano/scalar/basic_scipy.py:        # For some reason, on the GPU, uint64 inputs don't get casted
theano/scalar/basic_scipy.py:            // For GPU support
theano/scalar/basic_scipy.py:            // For GPU support
theano/compile/sharedvalue.py:        Set_value will work in-place on the GPU, if
theano/compile/sharedvalue.py:            * The destination on the GPU must be c_contiguous.
theano/compile/sharedvalue.py:        It is also worth mentioning that, for efficient transfer to the GPU,
theano/compile/sharedvalue.py:        The inplace on gpu memory work when borrow is either True or False.
theano/compile/ops.py:    Works inplace and works for CudaNdarrayType.
theano/compile/nanguardmode.py:    from theano.gpuarray.type import GpuArrayType, _name_for_ctx
theano/compile/nanguardmode.py:    from pygpu.gpuarray import GpuArray
theano/compile/nanguardmode.py:    pygpu_available = True
theano/compile/nanguardmode.py:    pygpu_available = False
theano/compile/nanguardmode.py:    elif pygpu_available and isinstance(arr, GpuArray):
theano/compile/nanguardmode.py:        return np.isnan(f_gpua_min(arr.reshape(arr.size)))
theano/compile/nanguardmode.py:    elif pygpu_available and isinstance(arr, GpuArray):
theano/compile/nanguardmode.py:        return (np.isinf(f_gpua_min(arr.reshape(arr.size))) or
theano/compile/nanguardmode.py:                np.isinf(f_gpua_max(arr.reshape(arr.size))))
theano/compile/nanguardmode.py:            guard_in = GpuArrayType(str(dtype), (False,), context_name=ctx_name)()
theano/compile/nanguardmode.py:            mode = get_mode('FAST_RUN').including('gpuarray')
theano/compile/nanguardmode.py:f_gpua_min = f_compute(T.min)
theano/compile/nanguardmode.py:f_gpua_max = f_compute(T.max)
theano/compile/nanguardmode.py:f_gpua_absmax = f_compute(lambda x: T.max(T.abs_(x)))
theano/compile/nanguardmode.py:                elif pygpu_available and isinstance(value, GpuArray):
theano/compile/nanguardmode.py:                    err = (f_gpua_absmax(value.reshape(value.size)) > 1e10)
theano/compile/tests/test_profiling.py:            p = theano.ProfileStats(False, gpu_checks=False)
theano/compile/tests/test_profiling.py:                assert "GPU: 8204KB (8204KB)" in the_string, (lines1, lines2)
theano/compile/tests/test_profiling.py:                assert "GPU: 12300KB (12300KB)" in the_string, (lines1, lines2)
theano/compile/tests/test_profiling.py:                assert "GPU: 8212KB" in the_string, (lines1, lines2)
theano/compile/tests/test_profiling.py:            p = theano.ProfileStats(False, gpu_checks=False)
theano/compile/tests/test_function_module.py:    # there is a GPU.  To test if we really sync, we compare a case we
theano/compile/tests/test_function_module.py:    # can run in parallel GPU and CPU computation. Then we sync to
theano/compile/tests/test_function_module.py:    import theano.gpuarray.tests.config
theano/compile/tests/test_function_module.py:    if theano.gpuarray.pygpu_activated:
theano/compile/tests/test_function_module.py:        w = theano.gpuarray.gpuarray_shared_constructor(
theano/compile/tests/test_function_module.py:            target=theano.gpuarray.tests.config.test_ctx_name)
theano/compile/tests/test_function_module.py:        x = theano.gpuarray.gpuarray_shared_constructor(
theano/compile/tests/test_function_module.py:            target=theano.gpuarray.tests.config.test_ctx_name)
theano/compile/tests/test_function_module.py:                            mode=theano.gpuarray.tests.config.mode_with_gpu)
theano/compile/tests/test_function_module.py:        assert any(isinstance(n.op, theano.gpuarray.blas.GpuGemm)
theano/compile/tests/test_function_module.py:        # Make sure libgpuarray have compile all kernels
theano/compile/tests/test_function_module.py:        # This is to make the test more stable across different GPUs.
theano/compile/tests/test_function_module.py:        raise SkipTest("Sync is only available when pygpu is activated.")
theano/compile/tests/test_pfunc.py:    # on a GPU device(s), or even on a remote machine.
theano/compile/builders.py:        - Add support for the GPU? Probably just need an opt to remove transfer
theano/compile/profiling.py:                 gpu_checks=True, **kwargs):
theano/compile/profiling.py:        if (gpu_checks and
theano/compile/profiling.py:            (hasattr(theano, 'gpuarray') and
theano/compile/profiling.py:             theano.gpuarray.pygpu_activated) and
theano/compile/profiling.py:                os.environ.get('CUDA_LAUNCH_BLOCKING', '0') != '1'):
theano/compile/profiling.py:                "You are running the Theano profiler with CUDA enabled."
theano/compile/profiling.py:                " Theano GPU ops execution is asynchronous by default."
theano/compile/profiling.py:                " CUDA_LAUNCH_BLOCKING to 1 to tell the CUDA driver to"
theano/compile/profiling.py:        if (config.profile and gpu_checks and
theano/compile/profiling.py:                hasattr(theano, 'gpuarray') and
theano/compile/profiling.py:                theano.gpuarray.pygpu_activated and
theano/compile/profiling.py:                "This cause bad profiling result in the gpu "
theano/compile/profiling.py:            # "<class 'theano.gpuarray.blas.GpuDot22'>" ->
theano/compile/profiling.py:            #  "theano.gpuarray.blas.GpuDot22"
theano/compile/profiling.py:        print('    Theano Linker time (includes C, CUDA code '
theano/compile/profiling.py:            from theano.gpuarray import GpuArrayType
theano/compile/profiling.py:            # Initial Mem info values [CPU, GPU]
theano/compile/profiling.py:                    if isinstance(out.type, GpuArrayType):
theano/compile/profiling.py:                    if isinstance(ins.type, GpuArrayType):
theano/compile/profiling.py:                # Separate CPU and GPU
theano/compile/profiling.py:            print("        GPU: %dKB (%dKB)" % ((int(round(
theano/compile/profiling.py:            print("        CPU + GPU: %dKB (%dKB)" % (int(round(
theano/compile/profiling.py:        print("        GPU: %dKB" % int(round(
theano/compile/profiling.py:        print("        CPU + GPU: %dKB" % int(round(
theano/compile/profiling.py:                if config.device.startswith("gpu"):
theano/compile/profiling.py:                          " generator supported on the GPU.", file=file)
theano/compile/profiling.py:        import theano.gpuarray
theano/compile/profiling.py:                if not theano.gpuarray.dnn.dnn_present():
theano/compile/profiling.py:                          "this allows the operation to run on GPU")
theano/compile/profiling.py:                if not theano.gpuarray.dnn.dnn_present():
theano/compile/profiling.py:                          "this allows the operation to run on GPU")
theano/compile/mode.py:# We need fast_compile_gpu here.  As on the GPU, we don't have all
theano/compile/mode.py:# fast_compile+gpu. We can't tag them just as 'gpu', as this would
theano/compile/mode.py:# exclude them if we exclude 'gpu'.
theano/compile/mode.py:OPT_FAST_COMPILE = gof.Query(include=['fast_compile', 'fast_compile_gpu'],
theano/compile/mode.py:               2, 'fast_run', 'fast_compile_gpu')
theano/compile/mode.py:               48.6, 'fast_compile', 'fast_run')  # must be after gpu stuff at 48.5
theano/compile/debugmode.py:    from theano.gpuarray import GpuArrayType
theano/compile/debugmode.py:        import pygpu
theano/compile/debugmode.py:            if isinstance(r.type, (TensorType, GpuArrayType)):
theano/compile/debugmode.py:            if isinstance(r.type, (TensorType, GpuArrayType)):
theano/compile/debugmode.py:            if isinstance(r.type, (TensorType, GpuArrayType)):
theano/compile/debugmode.py:                if isinstance(r.type, GpuArrayType):
theano/compile/debugmode.py:                    new_buf = pygpu.array(new_buf)
theano/compile/debugmode.py:            if isinstance(r.type, (TensorType, GpuArrayType)):
theano/compile/debugmode.py:            if isinstance(r.type, (TensorType, GpuArrayType)):
theano/compile/debugmode.py:                    if isinstance(r.type, (TensorType, GpuArrayType)):
theano/compile/function_module.py:        if (hasattr(theano, "gpuarray") and
theano/compile/function_module.py:                theano.gpuarray.pygpu_activated):
theano/compile/function_module.py:            import pygpu
theano/compile/function_module.py:                if isinstance(inp.data, pygpu.gpuarray.GpuArray):
theano/compile/pfunc.py:        # filter_variable ensure smooth conversion of cpu/gpu Types
theano/pathparse.py:    at runtime. Currently used in ``theano.gpuarray.dnn`` module
theano/tests/test_flake8.py:    "gpuarray/__init__.py",
theano/tests/test_flake8.py:    "gpuarray/tests/__init__.py",
theano/tests/test_flake8.py:    "sandbox/cuda/__init__.py",
theano/tests/test_flake8.py:    "sandbox/gpuarray/__init__.py",
theano/tests/test_ifelse.py:        # There is only 2 of the 3 ifelse that are moved on the GPU.
theano/tests/test_ifelse.py:        # Tests the gradient when both inputs are on the GPU.
theano/tests/test_printing.py:    prof = theano.compile.ProfileStats(atexit_print=False, gpu_checks=False)
theano/tests/main.py:        # error if device==gpu.
theano/tests/main.py:                                     " This will also run GPU tests when possible.\n"
theano/tests/main.py:                                     " If you want GPU-related tests to run on a"
theano/tests/main.py:                                     " specific GPU device, and not the default one,"
theano/tests/main.py:                                     " you should use the init_gpu_device theano flag.")
theano/gof/opt.py:class GraphToGPULocalOptGroup(LocalOptGroup):
theano/gof/opt.py:    """This is the equivalent of LocalOptGroup for GraphToGPU.
theano/gof/opt.py:    optimizer that use the GraphToGPU signature and not the normal
theano/gof/opt.py:        super(GraphToGPULocalOptGroup, self).__init__(*optimizers, **kwargs)
theano/gof/params_type.py:types wrapped into a ParamsType must provide a C interface (e.g. TensorType, Scalar, GpuArrayType,
theano/gof/params_type.py:    This class can create a struct of Theano types (like TensorType, GpuArrayType, etc.)
theano/gof/cc.py:        # broadcast check on the old GPU back-end. This check isn't
theano/gof/cc.py:        # done in the new GPU back-end or on the CPU.
theano/gof/cc.py:    # broadcast check on the old GPU back-end. This check isn't
theano/gof/cc.py:    # done in the new GPU back-end or on the CPU.
theano/gof/cmodule.py:            # Don't delete the gpuarray kernel cache
theano/gof/cmodule.py:            if root == config.gpuarray.cache_path:
theano/gof/cmodule.py:            #    When device=gpu, we compile during Theano
theano/gof/type.py:         - For ``GpuArrayType(dtype='int32', ...)``: should return ``"ga_int"``.
theano/gof/type.py:    # shared variable on the gpu.
theano/gof/type.py:        TensorType and GpuArrayType, provided they have the same
theano/gof/graph.py:            GpuArray of the same dtype and broadcastable patterns,
theano/gof/graph.py:    - `GpuArrayVariable` subclass of Variable that represents our object on
theano/gof/graph.py:      the GPU that is a subset of numpy.ndarray.
theano/gof/sandbox/typeattr.txt:  - On GPU, shape and strides have a big impact on the choice of algorithm for many ops.
theano/gof/sandbox/typeattr.txt:  - knowing the shape and stride we can generate faster GPU code that uses fewer registers
theano/configdefaults.py:             " created. They can't be run on the GPU with the current(old)"
theano/configdefaults.py:             " gpu back-end and are slow with gamer GPUs.",
theano/configdefaults.py:             "are more deterministic, but slower. In particular, on the GPU, "
theano/configdefaults.py:             "non-deterministic implementaion, e.g. when we do not have a GPU "
theano/configdefaults.py:# gpu means let the driver select the gpu. Needed in case of gpu in
theano/configdefaults.py:# gpuX mean use the gpu number X.
theano/configdefaults.py:                val.startswith('opencl') or
theano/configdefaults.py:                    val.startswith('cuda')):
theano/configdefaults.py:            elif val.startswith('gpu'):
theano/configdefaults.py:                    'You are tring to use the old GPU back-end. '
theano/configdefaults.py:                    'It was removed from Theano. Use device=cuda* now. '
theano/configdefaults.py:                    'See https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29 '
theano/configdefaults.py:                                  'one of "cpu", "opencl" or "cuda".'
theano/configdefaults.py:        return '%s (%s, opencl*, cuda*) ' % (self.fullname, self.default)
theano/configdefaults.py:    ("Default device for computations. If cuda* or opencl*, change the"
theano/configdefaults.py:     "default to try to move computation to the GPU. Do not use upper case"
theano/configdefaults.py:     "letters, only lower case even if NVIDIA uses capital letters."),
theano/configdefaults.py:    'init_gpu_device',
theano/configdefaults.py:    ("Initialize the gpu device to use, works only if device=cpu. "
theano/configdefaults.py:     "nor shared variables, to the specified GPU. "
theano/configdefaults.py:     "It can be used to run GPU-specific tests on a particular GPU."),
theano/configdefaults.py:                if (s[0] == 'cpu' or s[0].startswith('cuda') or
theano/configdefaults.py:                        s[0].startswith('opencl')):
theano/configdefaults.py:    Context map for multi-gpu operation. Format is a
theano/configdefaults.py:    device 'cuda0' and name 'test2' to device 'opencl0:0' follows:
theano/configdefaults.py:    "test->cuda0;test2->opencl0:0".
theano/configdefaults.py:    Invalid context names are 'cpu', 'cuda*' and 'opencl*'
theano/configdefaults.py:    "Print active device at when the GPU device is initialized.",
theano/configdefaults.py:def deprecated_gpuarray_sync(val):
theano/configdefaults.py:        raise RuntimeError("Flag gpuarray.sync is deprecated and will be removed in next Theano release.")
theano/configdefaults.py:AddConfigVar('gpuarray.sync',
theano/configdefaults.py:             ConfigParam(False, allow_override=False, filter=deprecated_gpuarray_sync),
theano/configdefaults.py:AddConfigVar('gpuarray.preallocate',
theano/configdefaults.py:             preallocates that fraction of the total GPU memory.  If 1
theano/configdefaults.py:AddConfigVar('gpuarray.sched',
theano/configdefaults.py:             """The sched parameter passed for context creation to pygpu.
theano/configdefaults.py:                With CUDA, using "multi" is equivalent to using the parameter
theano/configdefaults.py:                cudaDeviceScheduleBlockingSync. This is useful to lower the
theano/configdefaults.py:                CPU overhead when waiting for GPU. One user found that it
theano/configdefaults.py:AddConfigVar('gpuarray.single_stream',
theano/configdefaults.py:             multi-stream is enabled in libgpuarray, this may change.
theano/configdefaults.py:def get_cuda_root():
theano/configdefaults.py:    # We look for the cuda path since we need headers from there
theano/configdefaults.py:    v = os.getenv('CUDA_ROOT', "")
theano/configdefaults.py:    v = os.getenv('CUDA_PATH', "")
theano/configdefaults.py:AddConfigVar('cuda.root',
theano/configdefaults.py:             "Location of the cuda installation",
theano/configdefaults.py:             StrParam(get_cuda_root),
theano/configdefaults.py:def default_cuda_include():
theano/configdefaults.py:    if theano.config.cuda.root:
theano/configdefaults.py:        return os.path.join(theano.config.cuda.root, 'include')
theano/configdefaults.py:AddConfigVar('cuda.include_path',
theano/configdefaults.py:             "Location of the cuda includes",
theano/configdefaults.py:             StrParam(default_cuda_include),
theano/configdefaults.py:# We want to default to the cuda root if cudnn is installed there
theano/configdefaults.py:    root = theano.config.cuda.root
theano/configdefaults.py:            'optimized C-implementations (for both CPU and GPU) and will '
theano/configdefaults.py:    'gpu.local_elemwise_fusion',
theano/configdefaults.py:    ("Enable or not in fast_run mode(fast_run optimization) the gpu "
theano/configdefaults.py:    'gpuelemwise.sync',
theano/configdefaults.py:    "when true, wait that the gpu fct finished and check it error code.",
theano/configdefaults.py:AddConfigVar('experimental.unpickle_gpu_on_cpu',
theano/configdefaults.py:             "Allow unpickling of pickled GpuArrays as numpy.ndarrays."
theano/configdefaults.py:             "This is useful, if you want to open a GpuArray without "
theano/configdefaults.py:             "having cuda installed."
theano/configdefaults.py:             "If you have cuda installed, this will force unpickling to"
theano/configdefaults.py:             "however, trying to unpicke gpu functions will not succeed."
theano/configdefaults.py:             "gpu<>cpu transparency is solved.",
theano/configdefaults.py:AddConfigVar('warn.gpusum_01_011_0111_bug',
theano/configdefaults.py:              "silent bug with GpuSum pattern 01,011 and 0111 when the first "
theano/configdefaults.py:    'warn.gpu_set_subtensor1',
theano/configdefaults.py:    "incorrect results when moving to the gpu "
theano/configdefaults.py:    'gpuarray.cache_path',
theano/configdefaults.py:    'Directory to cache pre-compiled kernels for the gpuarray backend.',
theano/configdefaults.py:        lambda: os.path.join(config.compiledir, 'gpuarray_kernels'),
theano/tensor/opt.py:    We parametrise it to make it work for Elemwise and GpuElemwise op.
theano/tensor/opt.py:            # gpuarray GpuElemwise inherit from Elemwise
theano/tensor/opt.py:                    # with the InputToGpuOptimizer optimizer.
theano/tensor/opt.py:# 3. After it goes to 48.5 that move to the gpu. So 10 seem resonable.
theano/tensor/opt.py:    the GPU version of this optimizations.
theano/tensor/opt.py:                       # After move to gpu and merge2, before inplace.
theano/tensor/opt.py:@register_canonicalize('fast_compile_gpu')
theano/tensor/opt.py:    Also work for GpuIncSubtensor.
theano/tensor/opt.py:    Also work for GpuAdvancedIncSubtensor1.
theano/tensor/opt.py:                #    which simplifies the codegen for sum, especially on GPU
theano/tensor/opt.py:    This is faster on the GPU when memory fetching is a big part of
theano/tensor/opt.py:            # 512 is too small for the cpu and too big for some gpu!
theano/tensor/opt.py:    We parametrize it to make it work for Elemwise and GpuElemwise op.
theano/tensor/opt.py:        GpuElemwise or Elemwise class (the one that we want to fuse)
theano/tensor/opt.py:        that this elemwise can take (useful for GpuElemwise).
theano/tensor/opt.py:        GPU kernel currently has a limit of 256 bytes for
theano/tensor/opt.py:        # GPU kernel function.
theano/tensor/opt.py:    # Must be after gpu(48.5) and before AddDestroyHandler(49.5)
theano/tensor/extra_ops.py:    Wraps numpy.unique. This op is not implemented on the GPU.
theano/tensor/slinalg.py:    For on CPU and GPU.
theano/tensor/basic.py:    # useful to test the GPU as they don't use extended precision and
theano/tensor/basic.py:            # If the clients is a transfer to the GPU, we don't want to
theano/tensor/basic.py:            # fold. We let the Alloc being moved to the GPU, then we
theano/tensor/basic.py:            # let the GPU algo decide if it need to fold it or not.
theano/tensor/basic.py:            elif client[0].op.__class__.__name__.lower().startswith("gpu"):
theano/tensor/basic.py:    For gpu, if you specify dtype=float32, everything will be done on the gpu.
theano/tensor/basic.py:    the gpu optimization
theano/tensor/basic.py:    the gpu optimization
theano/tensor/basic.py:    We apply the opt here not to pollute the graph especially during the gpu
theano/tensor/nnet/opt.py:            'both "conv_dnn" and "conv_gemm" from the optimizer? If on GPU, '
theano/tensor/nnet/opt.py:            'is cuDNN available and does the GPU support it? If on CPU, '
theano/tensor/nnet/neighbours.py:        # GpuImages2Neibs should not run this perform in DebugMode
theano/tensor/nnet/bn.py:    Also works on GPUs, but is not optimized using cuDNN.
theano/tensor/nnet/abstract_conv.py:        GPU. Otherwise, it is the *Corr3dMM* convolution that will be used
theano/tensor/nnet/abstract_conv.py:        GPU. Otherwise, it is the *CorrMM* convolution that will be used
theano/tensor/nnet/abstract_conv.py:        GPU. Otherwise, it is the *Corr3dMM* convolution that will be used
theano/tensor/nnet/abstract_conv.py:        GPU. Otherwise, it is the *CorrMM* convolution that will be used
theano/tensor/nnet/abstract_conv.py:        GPU. Otherwise, it is the *Corr3dMM* convolution that will be used
theano/tensor/nnet/conv3d2d.py:    Work on the GPU.
theano/tensor/nnet/conv3d2d.py:    For the GPU, use nnet.conv3d.
theano/tensor/nnet/tests/test_neighbours.py:mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')
theano/tensor/nnet/tests/test_neighbours.py:    mode = mode_without_gpu
theano/tensor/nnet/tests/test_corr.py:    mode = theano.compile.get_mode("FAST_RUN").excluding('gpuarray')
theano/tensor/nnet/tests/test_corr.py:        mode = theano.compile.get_mode("FAST_RUN").excluding('gpuarray')
theano/tensor/nnet/tests/test_corr.py:        mode = theano.compile.get_mode("FAST_RUN").excluding('gpuarray')
theano/tensor/nnet/tests/test_corr.py:        mode = theano.compile.get_mode("FAST_RUN").excluding('gpuarray')
theano/tensor/nnet/c_code/corr_gemm.c:// Unlike the Caffe and Theano GPU verions, the data_im array is set to zero
theano/tensor/nnet/c_code/corr_gemm.c:// GPU version authors: Arjun Jain, Frederic Bastien, Jan Schlueter
theano/tensor/nnet/c_code/corr_gemm.c:// CPU version adapted from GPU version
theano/tensor/nnet/c_code/corr_gemm.c:        // Note that this code was translated from the Theano GPU code,
theano/tensor/nnet/c_code/corr_gemm.c:          im2col_gpu(bottom_data + bottom[i]->offset(n), channels_, height_,
theano/tensor/nnet/c_code/corr_gemm.c:            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
theano/tensor/nnet/c_code/corr_gemm.c:        // Note that this code was translated from the Theano GPU code,
theano/tensor/nnet/c_code/corr_gemm.c:          im2col_gpu(bottom_data + (*bottom)[i]->offset(n), channels_, height_,
theano/tensor/nnet/c_code/corr_gemm.c:            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
theano/tensor/nnet/c_code/corr_gemm.c:        // Note that this code was translated from the Theano GPU code,
theano/tensor/nnet/c_code/corr_gemm.c:              caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
theano/tensor/nnet/c_code/corr_gemm.c:            col2im_gpu(col_diff, channels_, height_, width_,
theano/tensor/nnet/c_code/corr3d_gemm.c:// Unlike the Caffe and Theano GPU verions, the data_im array is set to zero
theano/tensor/nnet/c_code/corr3d_gemm.c:// GPU version authors: Arjun Jain, Frederic Bastien, Jan Schlueter
theano/tensor/nnet/c_code/corr3d_gemm.c:// CPU version adapted from GPU version
theano/tensor/nnet/conv.py:        Passed to GpuConv.
theano/tensor/nnet/conv.py:        Passed to GpuConv, if version='no_fft', fft
theano/tensor/nnet/conv.py:        Passed to GpuConv, used by graph optimizers to aid algorithm choice.
theano/tensor/nnet/nnet.py:We register all optimization with the gpu tag as we don't
theano/tensor/nnet/nnet.py:implement all the intermediate case on the GPU (in particular
theano/tensor/nnet/nnet.py:AdvancedSubtensor). So to make sure it run well on the gpu with
theano/tensor/nnet/nnet.py:fast_compile, we register them as needed for the GPU. This can be
theano/tensor/nnet/nnet.py:revisited later when all the intermediate part are on the GPU.
theano/tensor/nnet/nnet.py:@opt.register_specialize('fast_compile_gpu')
theano/tensor/nnet/nnet.py:@opt.register_stabilize('fast_compile_gpu')
theano/tensor/nnet/nnet.py:@opt.register_specialize('fast_compile_gpu')
theano/tensor/nnet/nnet.py:               'fast_run', 'xent', 'fast_compile_gpu')
theano/tensor/nnet/nnet.py:    'fast_compile_gpu',
theano/tensor/nnet/nnet.py:@opt.register_specialize('fast_compile_gpu')
theano/tensor/nnet/nnet.py:@opt.register_specialize('fast_compile_gpu')
theano/tensor/nnet/nnet.py:@opt.register_specialize('fast_compile_gpu')
theano/tensor/nnet/nnet.py:@opt.register_specialize('fast_compile_gpu')
theano/tensor/nnet/nnet.py:    # This is probably the fastest implementation for GPUs. Both the forward
theano/tensor/nnet/nnet.py:    # pass and the gradient get compiled into a single GpuElemwise call.
theano/tensor/nnet/__init__.py:        GPU. Otherwise, it is the *CorrMM* convolution that will be used
theano/tensor/nnet/__init__.py:        GPU. Otherwise, it is the *CorrMM* convolution that will be used
theano/tensor/signal/pool.py:            " On the GPU, using ignore_border=True is needed to use cuDNN."
theano/tensor/signal/pool.py:            " GPU combination supported is when"
theano/tensor/signal/pool.py:            " On the GPU, using ignore_border=True is needed to use cuDNN."
theano/tensor/signal/pool.py:            " GPU combination supported is when"
theano/tensor/signal/pool.py:# NB: This enum type is currently used in gpuarray/pool.py.
theano/tensor/signal/pool.py:# (cf. `theano/gpuarray/cudnn_defs.py`).
theano/tensor/nlinalg.py:    Works on GPU since 0.6rc4.
theano/tensor/nlinalg.py:    Does not run on GPU;
theano/tensor/nlinalg.py:    Theano utilization of numpy.linalg.tensorsolve. Does not run on GPU!
theano/tensor/tests/test_mpi.py:    keep_flags = ','.join((f for f in flags.split(',') if not f.startswith('init_gpu_device')))
theano/tensor/tests/test_opt.py:    mode = mode.excluding('fusion').excluding('gpu')
theano/tensor/tests/test_opt.py:            'canonicalize', 'fast_run').excluding('gpu', 'fusion')
theano/tensor/tests/test_opt.py:            'canonicalize', 'fast_run').excluding('gpu', 'fusion')
theano/tensor/tests/test_opt.py:            'canonicalize').including('fast_run').excluding('gpu')
theano/tensor/tests/test_elemwise.py:                        # GpuCAReduce don't implement all cases when size is 0
theano/tensor/tests/test_elemwise.py:                        # GpuCAReduce don't implement all cases when size is 0
theano/tensor/tests/test_elemwise.py:                    # GpuCAReduce don't implement all cases when size is 0
theano/tensor/tests/test_elemwise.py:                        # or the overflow will be different between CPU and GPU,
theano/tensor/tests/test_sort.py:        ((17, 15), (2, 3, 5, 7, 11), (500, 5, 3)),  # NB: Test may fail with bigger sizes (e.g. (2017, 5, 3)) due to "too many resources requested" kernel error on some GPUs.
theano/tensor/tests/test_blas_scipy.py:    mode = mode.including('fast_run').excluding('gpu', 'c_blas')
theano/tensor/tests/test_subtensor.py:    This is build in a way that allow to reuse it to test the equivalent gpu op.
theano/tensor/tests/test_subtensor.py:                          # Test 4 dims as gpu code use another algo
theano/tensor/tests/test_subtensor.py:        orig_warn = theano.config.warn.gpu_set_subtensor1
theano/tensor/tests/test_subtensor.py:            theano.config.warn.gpu_set_subtensor1 = False
theano/tensor/tests/test_subtensor.py:            theano.config.warn.gpu_set_subtensor1 = orig_warn
theano/tensor/tests/test_subtensor.py:                          # Test 4 dims as gpu code use another algo
theano/tensor/tests/test_blas.py:        # We put this test in this class to test it on the gpu too.
theano/tensor/tests/test_blas.py:    mode = mode.including('fast_run').excluding('gpu', 'c_blas', 'scipy_blas')
theano/tensor/tests/test_sharedvar.py:            # test optimized get set value on the gpu(don't pass data to the cpu)
theano/tensor/tests/test_sharedvar.py:                # not updated for GpuArray, but it is for ndarray
theano/tensor/tests/test_sharedvar.py:            # specificaly useful for gpu data
theano/tensor/tests/test_sharedvar.py:                assert sum([node.op.__class__.__name__ in ["Gemm", "GpuGemm", "StructuredDot"] for node in topo]) == 1
theano/tensor/tests/test_sharedvar.py:                assert all(node.op.inplace for node in topo if node.op.__class__.__name__ == "GpuGemm")
theano/tensor/tests/test_sharedvar.py:                assert sum([node.op.__class__.__name__ in ["Gemm", "GpuGemm", "StructuredDot"] for node in topo]) == 1
theano/tensor/tests/test_sharedvar.py:                assert all(node.op.inplace for node in topo if node.op.__class__.__name__ == "GpuGemm")
theano/tensor/tests/test_sharedvar.py:                assert sum([node.op.__class__.__name__ in ["Gemm", "GpuGemm", "StructuredDot"] for node in topo]) == 1
theano/tensor/tests/test_sharedvar.py:                assert all(node.op.inplace for node in topo if node.op.__class__.__name__ == "GpuGemm")
theano/tensor/tests/mlp_test.py:        Theano to copy it into the GPU memory (when code is run on GPU).
theano/tensor/tests/mlp_test.py:        Since copying data into the GPU is slow, copying a minibatch every time
theano/tensor/tests/mlp_test.py:        # When storing data on the GPU it has to be stored as floats
theano/tensor/tests/mlp_test.py:        # theano.config.floatX so that the code is runable on GPU
theano/tensor/tests/test_basic.py:        # tested only on cpu as gpu support only float32
theano/tensor/tests/test_basic.py:        # In the past it was broken on the GPU.
theano/tensor/type.py:        For the moment, only a TensorType and GpuArrayType will be
theano/tensor/subtensor.py:        function on PyArray and GpuArray object.
theano/tensor/subtensor.py:        # to change the particulars, e.g. GpuIncSubtensor
theano/tensor/subtensor.py:        # into the same operations on gpu arrays.
theano/tensor/fourier.py:    transfers to GPU ops.
theano/tensor/blas.py:    - GPU-based (theano.gpuarray)
theano/tensor/blas.py:# fast_compile is needed to have GpuDot22 created.
theano/tensor/sort.py:    - CPU and GPU ops don't produce same output order. This is expected.
theano/tensor/sort.py:      elements are on the correct side. On the GPU, they
theano/tensor/sort.py:        # however, we add "idx_dtype" param as memory is more precious on gpu
theano/printing.py:default_colorCodes = {'GpuFromHost': 'red',
theano/printing.py:                      'HostFromGpu': 'red',
theano/printing.py:        - Red ellipses are transfers from/to the gpu
theano/updates.py:            # GPU SharedVariable is customarily associated with a TensorType
theano/updates.py:            # value. Should it be cast to a GPU value right away?  Should
theano/scan_module/scan_perform.pyx:        # 4.5. Keep a reference to the variables (ndarrays, GpuArrays,
theano/scan_module/scan_perform.pyx:                old_output_data[idx] = var.gpudata
theano/scan_module/scan_perform.pyx:        # 4.6. Keep a reference to the variables (ndarrays, GpuArrays,
theano/scan_module/scan_perform.pyx:                old_mitmot_input_data[idx] = var.gpudata
theano/scan_module/scan_perform.pyx:                            same_data = (new_var.gpudata == old_data)
theano/scan_module/scan_perform.pyx:                        output_reused = (new_var.gpudata == old_data)
theano/scan_module/scan_perform.pyx:                        output_reused = (new_var.gpudata == old_data)
theano/scan_module/scan_op.py:        of different types of arguments, name, mode, if it should run on GPU or
theano/scan_module/scan_op.py:    Theano deals with the GPU. If it runs on the GPU, scan needs
theano/scan_module/scan_op.py:    to construct certain outputs (those who reside in the GPU
theano/scan_module/scan_op.py:    memory) as the GPU-specific type.  However we can not import
theano/scan_module/scan_op.py:    gpu code in this file (as it is in sandbox, and not available
theano/scan_module/scan_op.py:    on each machine) so the workaround is that the GPU
theano/scan_module/scan_op.py:    function that is able to construct a GPU type. This way the
theano/scan_module/scan_op.py:    GPU, it just constructs any tensor using this function (which
theano/scan_module/scan_op.py:        if self.info['gpua']:
theano/scan_module/scan_op.py:            self._hash_inner_graph = self.info['gpu_hash']
theano/scan_module/scan_op.py:        # If scan has the flag 'gpua' set to false (meaning that is shouldn't
theano/scan_module/scan_op.py:        # use the gpuarray gpu backend ), ensure that is has no input and no
theano/scan_module/scan_op.py:        # output with type GpuArrayType
theano/scan_module/scan_op.py:        from theano.gpuarray import GpuArrayType
theano/scan_module/scan_op.py:        if not self.info.get("gpua", False):
theano/scan_module/scan_op.py:                if isinstance(inp.type, GpuArrayType):
theano/scan_module/scan_op.py:                                    "inner graph is of type GpuArrayType but "
theano/scan_module/scan_op.py:                if isinstance(out.type, GpuArrayType):
theano/scan_module/scan_op.py:                                    "inner graph is of type GpuArrayType but "
theano/scan_module/scan_op.py:            they are both TensorType or GpuArrayType. It internally
theano/scan_module/scan_op.py:                         'n_sit_sot', 'gpua', 'n_mit_mot_outs',
theano/scan_module/scan_op.py:        if self.gpua:
theano/scan_module/scan_op.py:            gpu_str = 'gpu'
theano/scan_module/scan_op.py:            gpu_str = 'cpu'
theano/scan_module/scan_op.py:        aux_txt = aux_txt % (name, gpu_str, str(self.name))
theano/scan_module/scan_op.py:        # outputs are on the gpu and speed up some checks during the execution
theano/scan_module/scan_op.py:            # 4.5. Keep a reference to the variables (ndarrays, GpuArrays,
theano/scan_module/scan_op.py:                    old_output_data[idx] = var.gpudata
theano/scan_module/scan_op.py:            # 4.6. Keep a reference to the variables (ndarrays, GpuArrays,
theano/scan_module/scan_op.py:                    old_mitmot_input_data[idx] = var.gpudata
theano/scan_module/scan_op.py:                                same_data = (new_var.gpudata == old_data)
theano/scan_module/scan_op.py:                            output_reused = (new_var.gpudata == old_data)
theano/scan_module/scan_op.py:                            output_reused = (new_var.gpudata == old_data)
theano/scan_module/scan_op.py:        info['gpua'] = False
theano/scan_module/scan_op.py:        info['gpua'] = False
theano/scan_module/scan_opt.py:    def __init__(self, typeInfer=None, gpua_flag=False):
theano/scan_module/scan_opt.py:        self.gpua_flag = gpua_flag
theano/scan_module/scan_opt.py:        # Depending on the value of gpua_flag, get the list of memory
theano/scan_module/scan_opt.py:        if self.gpua_flag:
theano/scan_module/scan_opt.py:            # gpuarray might be imported but not its GpuAlloc and
theano/scan_module/scan_opt.py:            # GpuAllopEmpty ops.
theano/scan_module/scan_opt.py:                alloc_ops += (theano.gpuarray.GpuAlloc,
theano/scan_module/scan_opt.py:                              theano.gpuarray.GpuAllocEmpty)
theano/scan_module/scan_opt.py:                          x.op.info['gpua'] == self.gpua_flag)]
theano/scan_module/scan_opt.py:        info['gpua'] = False
theano/scan_module/tests/test_scan.py:        # = gpu_from_host(orig)  # <-- this doesn't work
theano/scan_module/tests/test_scan.py:        # so it will be slower and won't get transferred to the gpu.
theano/scan_module/tests/test_scan.py:class ScanGpuTests:
theano/scan_module/tests/test_scan.py:    This class defines a number of tests for Scan on GPU as well as a few
theano/scan_module/tests/test_scan.py:    helper functions for these tests. The GPU tests defined in this class are
theano/scan_module/tests/test_scan.py:    independent of the GPU backend used. Because of this, a class inheriting
theano/scan_module/tests/test_scan.py:    from ScanGpuTests should define the following attributes and methods to
theano/scan_module/tests/test_scan.py:    - self.gpu_backend : Reference to the backend module
theano/scan_module/tests/test_scan.py:    - self.mode_with_opt : Compilation mode to force usage of the gpu backend
theano/scan_module/tests/test_scan.py:    - self.is_scan_on_gpu(node) : Method to determine is a scan node has been
theano/scan_module/tests/test_scan.py:                                  moved to run on a gpu under the specific
theano/scan_module/tests/test_scan.py:    def test_one_sequence_one_output_weights_gpu1(self):
theano/scan_module/tests/test_scan.py:        mode = self.mode_with_gpu.excluding('InputToGpuOptimizer')
theano/scan_module/tests/test_scan.py:        output = self.gpu_backend.gpu_from_host(output)
theano/scan_module/tests/test_scan.py:                             mode=self.mode_with_gpu)
theano/scan_module/tests/test_scan.py:        assert sum([isinstance(node.op, self.gpu_backend.HostFromGpu)
theano/scan_module/tests/test_scan.py:        assert sum([isinstance(node.op, self.gpu_backend.GpuFromHost)
theano/scan_module/tests/test_scan.py:        # check that there is no gpu transfer in the inner loop.
theano/scan_module/tests/test_scan.py:        assert any([isinstance(node.op, self.gpu_backend.GpuElemwise)
theano/scan_module/tests/test_scan.py:        assert not any([isinstance(node.op, self.gpu_backend.HostFromGpu)
theano/scan_module/tests/test_scan.py:        assert not any([isinstance(node.op, self.gpu_backend.GpuFromHost)
theano/scan_module/tests/test_scan.py:    # This second version test the second case in the optimizer to the gpu.
theano/scan_module/tests/test_scan.py:    def test_one_sequence_one_output_weights_gpu2(self):
theano/scan_module/tests/test_scan.py:                                      mode=self.mode_with_gpu)
theano/scan_module/tests/test_scan.py:                             mode=self.mode_with_gpu)
theano/scan_module/tests/test_scan.py:        assert sum([isinstance(node.op, self.gpu_backend.HostFromGpu)
theano/scan_module/tests/test_scan.py:        assert sum([isinstance(node.op, self.gpu_backend.GpuFromHost)
theano/scan_module/tests/test_scan.py:        # check that there is no gpu transfer in the inner loop.
theano/scan_module/tests/test_scan.py:        assert any([isinstance(node.op, self.gpu_backend.GpuElemwise)
theano/scan_module/tests/test_scan.py:        assert not any([isinstance(node.op, self.gpu_backend.HostFromGpu)
theano/scan_module/tests/test_scan.py:        assert not any([isinstance(node.op, self.gpu_backend.GpuFromHost)
theano/scan_module/tests/test_scan.py:    # outputs when is running on GPU
theano/scan_module/tests/test_scan.py:    def test_gpu3_mixture_dtype_outputs(self):
theano/scan_module/tests/test_scan.py:                                      mode=self.mode_with_gpu)
theano/scan_module/tests/test_scan.py:                             mode=self.mode_with_gpu)
theano/scan_module/tests/test_scan.py:        assert self.is_scan_on_gpu(scan_node)
theano/scan_module/tests/test_scan.py:                                               mode=self.mode_with_gpu)
theano/scan_module/tests/test_scan.py:                               mode=self.mode_with_gpu)
theano/scan_module/tests/test_scan.py:    def test_gpu_memory_usage(self):
theano/scan_module/tests/test_scan.py:        # function is reasonnable when executed on the GPU. It checks for
theano/scan_module/tests/test_scan.py:        # brought memory usage on the GPU to ~12G.
theano/scan_module/tests/test_scan.py:                                mode=self.mode_with_gpu_nodebug)
theano/scan_module/tests/test_scan.py:                                         mode=self.mode_with_gpu_nodebug)
theano/scan_module/tests/test_scan.py:    def test_memory_reuse_gpudimshuffle(self):
theano/scan_module/tests/test_scan.py:        # the result of a GpuDimshuffle (because an optimization in
theano/scan_module/tests/test_scan.py:        # GpuDimshuffle can cause issues with the memory pre-allocation
theano/scan_module/tests/test_scan.py:                             mode=self.mode_with_gpu)
theano/scan_module/tests/test_scan.py:                              mode=self.mode_with_gpu)
theano/scan_module/tests/test_scan.py:class T_Scan_Gpuarray(unittest.TestCase, ScanGpuTests):
theano/scan_module/tests/test_scan.py:    This class takes the gpu tests for scan that are defined in
theano/scan_module/tests/test_scan.py:    class ScanGpuTests and runs them using the gpuarray backend.
theano/scan_module/tests/test_scan.py:        from theano import gpuarray
theano/scan_module/tests/test_scan.py:        self.gpu_backend = gpuarray
theano/scan_module/tests/test_scan.py:        def gpu_from_host(v):
theano/scan_module/tests/test_scan.py:            return gpuarray.GpuFromHost(None)(v)
theano/scan_module/tests/test_scan.py:        self.gpu_backend.gpu_from_host = gpu_from_host
theano/scan_module/tests/test_scan.py:        self.mode_with_gpu = mode_with_opt.including('gpuarray', 'scan')
theano/scan_module/tests/test_scan.py:        self.mode_with_gpu_nodebug = mode_nodebug.including('gpuarray', 'scan')
theano/scan_module/tests/test_scan.py:        super(T_Scan_Gpuarray, self).__init__(*args, **kwargs)
theano/scan_module/tests/test_scan.py:        import theano.gpuarray.tests.config
theano/scan_module/tests/test_scan.py:        # Skip the test if pygpu is not available
theano/scan_module/tests/test_scan.py:        if not self.gpu_backend.pygpu_activated:
theano/scan_module/tests/test_scan.py:            raise SkipTest('Optional package pygpu disabled')
theano/scan_module/tests/test_scan.py:        super(T_Scan_Gpuarray, self).setUp()
theano/scan_module/tests/test_scan.py:    def is_scan_on_gpu(self, node):
theano/scan_module/tests/test_scan.py:        return node.op.info.get('gpua', False)
theano/scan_module/tests/test_scan_checkpoints.py:    from pygpu.gpuarray import GpuArrayException
theano/scan_module/tests/test_scan_checkpoints.py:    PYGPU_AVAILABLE = True
theano/scan_module/tests/test_scan_checkpoints.py:    PYGPU_AVAILABLE = False
theano/scan_module/tests/test_scan_checkpoints.py:    @unittest.skipUnless(PYGPU_AVAILABLE, 'Requires pygpu.')
theano/scan_module/tests/test_scan_checkpoints.py:        if None not in theano.gpuarray.type.list_contexts():
theano/scan_module/tests/test_scan_checkpoints.py:            return unittest.SkipTest('Requires gpuarray backend.')
theano/scan_module/tests/test_scan_checkpoints.py:        from theano.gpuarray.tests.config import mode_with_gpu  # noqa
theano/scan_module/tests/test_scan_checkpoints.py:                            outputs=self.grad_A, mode=mode_with_gpu)
theano/scan_module/tests/test_scan_checkpoints.py:                                  mode=mode_with_gpu)
theano/scan_module/tests/test_scan_checkpoints.py:        free_gmem = theano.gpuarray.type._context_reg[None].free_gmem
theano/scan_module/tests/test_scan_checkpoints.py:        if isinstance(mode_with_gpu, theano.compile.DebugMode):
theano/scan_module/tests/test_scan_checkpoints.py:        if not isinstance(mode_with_gpu, theano.compile.DebugMode):
theano/scan_module/tests/test_scan_checkpoints.py:            self.assertRaises(GpuArrayException, f, data, 1000)
theano/scan_module/c_code/scan_perform.c:static const char __pyx_k_gpudata[] = "gpudata";
theano/scan_module/c_code/scan_perform.c:static PyObject *__pyx_n_s_gpudata;
theano/scan_module/c_code/scan_perform.c: *         # 4.5. Keep a reference to the variables (ndarrays, GpuArrays,
theano/scan_module/c_code/scan_perform.c: *                 old_output_data[idx] = var.gpudata
theano/scan_module/c_code/scan_perform.c: *                 old_output_data[idx] = var.gpudata             # <<<<<<<<<<<<<<
theano/scan_module/c_code/scan_perform.c: *         # 4.6. Keep a reference to the variables (ndarrays, GpuArrays,
theano/scan_module/c_code/scan_perform.c:        __pyx_t_3 = __Pyx_PyObject_GetAttrStr(__pyx_v_var, __pyx_n_s_gpudata); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 374, __pyx_L1_error)
theano/scan_module/c_code/scan_perform.c: *                 old_mitmot_input_data[idx] = var.gpudata
theano/scan_module/c_code/scan_perform.c: *                 old_mitmot_input_data[idx] = var.gpudata             # <<<<<<<<<<<<<<
theano/scan_module/c_code/scan_perform.c:        __pyx_t_3 = __Pyx_PyObject_GetAttrStr(__pyx_v_var, __pyx_n_s_gpudata); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 391, __pyx_L1_error)
theano/scan_module/c_code/scan_perform.c: *                             same_data = (new_var.gpudata == old_data)
theano/scan_module/c_code/scan_perform.c: *                             same_data = (new_var.gpudata == old_data)             # <<<<<<<<<<<<<<
theano/scan_module/c_code/scan_perform.c:              __pyx_t_2 = __Pyx_PyObject_GetAttrStr(__pyx_v_new_var, __pyx_n_s_gpudata); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 456, __pyx_L1_error)
theano/scan_module/c_code/scan_perform.c: *                             same_data = (new_var.gpudata == old_data)
theano/scan_module/c_code/scan_perform.c: *                         output_reused = (new_var.gpudata == old_data)
theano/scan_module/c_code/scan_perform.c: *                         output_reused = (new_var.gpudata == old_data)             # <<<<<<<<<<<<<<
theano/scan_module/c_code/scan_perform.c:            __pyx_t_10 = __Pyx_PyObject_GetAttrStr(__pyx_v_new_var, __pyx_n_s_gpudata); if (unlikely(!__pyx_t_10)) __PYX_ERR(0, 500, __pyx_L1_error)
theano/scan_module/c_code/scan_perform.c: *                         output_reused = (new_var.gpudata == old_data)
theano/scan_module/c_code/scan_perform.c: *                         output_reused = (new_var.gpudata == old_data)
theano/scan_module/c_code/scan_perform.c: *                         output_reused = (new_var.gpudata == old_data)             # <<<<<<<<<<<<<<
theano/scan_module/c_code/scan_perform.c:            __pyx_t_2 = __Pyx_PyObject_GetAttrStr(__pyx_v_new_var, __pyx_n_s_gpudata); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 540, __pyx_L1_error)
theano/scan_module/c_code/scan_perform.c: *                         output_reused = (new_var.gpudata == old_data)
theano/scan_module/c_code/scan_perform.c:  {&__pyx_n_s_gpudata, __pyx_k_gpudata, sizeof(__pyx_k_gpudata), 0, 0, 1, 1},
theano/scan_module/scan.py:* it ensures that data is not copied from host to gpu and gpu to
theano/scan_module/scan.py:        If you use preallocate and this scan is on GPU, the speed up
theano/scan_module/scan.py:    # gpuarray is imported here, instead of being imported on top of
theano/scan_module/scan.py:    from theano import gpuarray
theano/scan_module/scan.py:    if gpuarray.pygpu_activated:
theano/scan_module/scan.py:        # replace w with w_copy, where w is a GPU variable
theano/scan_module/scan.py:        # variables are put on GPU right away >:| ,
theano/scan_module/scan.py:            if (isinstance(w.type, gpuarray.GpuArrayType) and
theano/scan_module/scan.py:    info['gpua'] = False
theano/scan_module/scan_utils.py:        2) x is on gpu, x_copy on host, then you need to replace
theano/scan_module/scan_utils.py:        host_from_gpu(x) with x_copy
theano/scan_module/scan_utils.py:    This happens because initially shared variables are on GPU... which is
theano/scan_module/scan_utils.py:    from theano.gpuarray.basic_ops import GpuFromHost, host_from_gpu
theano/scan_module/scan_utils.py:    from theano.gpuarray import pygpu_activated
theano/scan_module/scan_utils.py:    from theano.gpuarray.type import GpuArrayType
theano/scan_module/scan_utils.py:        assert isinstance(x.type, GpuArrayType)
theano/scan_module/scan_utils.py:        d[out] = GpuFromHost(x.type.context_name)(x_copy)
theano/scan_module/scan_utils.py:    elif (pygpu_activated and
theano/scan_module/scan_utils.py:          out.owner.op == host_from_gpu and
theano/scan_module/scan_utils.py:    info['gpua'] = op.info['gpua']
theano/scan_module/scan_utils.py:                  'gpua', 'as_while', 'profile', 'allow_gc'):
theano/misc/do_nightly_build_send:    gpu_time = None
theano/misc/do_nightly_build_send:        if gpu_time is None and line.startswith("gpu % expected/get"):
theano/misc/do_nightly_build_send:                if start.startswith("gpu % expected/get"):
theano/misc/do_nightly_build_send:                    gpu_time = start
theano/misc/may_share_memory.py:Function to detect memory sharing for ndarray AND sparse type AND GpuArray.
theano/misc/may_share_memory.py:    # scipy not imported, their can be only ndarray and gpuarray
theano/misc/may_share_memory.py:from theano import gpuarray
theano/misc/may_share_memory.py:if gpuarray.pygpu:
theano/misc/may_share_memory.py:    def _is_gpua(a):
theano/misc/may_share_memory.py:        return isinstance(a, gpuarray.pygpu.gpuarray.GpuArray)
theano/misc/may_share_memory.py:    def _is_gpua(a):
theano/misc/may_share_memory.py:    a_gpua = _is_gpua(a)
theano/misc/may_share_memory.py:    b_gpua = _is_gpua(b)
theano/misc/may_share_memory.py:    if a_gpua and b_gpua:
theano/misc/may_share_memory.py:        return gpuarray.pygpu.gpuarray.may_share_memory(a, b)
theano/misc/may_share_memory.py:    if (not(a_ndarray or a_sparse or a_gpua) or
theano/misc/may_share_memory.py:            not(b_ndarray or b_sparse or b_gpua)):
theano/misc/may_share_memory.py:                            " and scipy.sparse or GpuArray type")
theano/misc/may_share_memory.py:    if a_gpua or b_gpua:
theano/misc/tests/test_may_share_memory.py:test the tensor and sparse type. (gpuarray is tested in the gpuarray folder).
theano/misc/pkl_utils.py:class PersistentGpuArrayID(PersistentNdarrayID):
theano/misc/pkl_utils.py:        from theano.gpuarray.type import _name_for_ctx
theano/misc/pkl_utils.py:            import pygpu
theano/misc/pkl_utils.py:            pygpu = None
theano/misc/pkl_utils.py:        if (pygpu and
theano/misc/pkl_utils.py:                isinstance(obj, pygpu.gpuarray.GpuArray)):
theano/misc/pkl_utils.py:                self.seen[id(obj)] = 'gpuarray.{0}'.format(name)
theano/misc/pkl_utils.py:        return super(PersistentGpuArrayID, self).__call__(obj)
theano/misc/pkl_utils.py:class PersistentSharedVariableID(PersistentGpuArrayID):
theano/misc/pkl_utils.py:        from theano.gpuarray.type import get_context
theano/misc/pkl_utils.py:        from theano.gpuarray import pygpu
theano/misc/pkl_utils.py:        if array_type == 'gpuarray':
theano/misc/pkl_utils.py:            if config.experimental.unpickle_gpu_on_cpu:
theano/misc/pkl_utils.py:                warnings.warn("config.experimental.unpickle_gpu_on_cpu is set "
theano/misc/pkl_utils.py:                              "to True. Unpickling GpuArray as numpy.ndarray")
theano/misc/pkl_utils.py:            elif pygpu:
theano/misc/pkl_utils.py:                ret = pygpu.array(array, context=get_context(ctx_name))
theano/misc/pkl_utils.py:                raise ImportError("pygpu not found. Cannot unpickle GpuArray")
theano/misc/check_blas.py:    elif any([x.op.__class__.__name__ == 'GpuGemm' for x in
theano/misc/check_blas.py:        impl = 'GPU'
theano/misc/check_blas.py:        impl = 'ERROR, unable to tell if Theano used the cpu or the gpu:\n'
theano/misc/check_blas.py:        sync = (hasattr(theano, "gpuarray") and
theano/misc/check_blas.py:                isinstance(c, theano.gpuarray.GpuArraySharedVariable))
theano/misc/check_blas.py:        cuda version      8.0    7.5    7.0
theano/misc/check_blas.py:        gpu
theano/misc/burn_gpu.py:GPU power consumption then gemm call.
theano/misc/burn_gpu.py:from theano.gpuarray import dnn
theano/misc/burn_gpu.py:    # replaced by a GpuAllocEmpty
theano/misc/check_multi_gpu.py:and two GPU to measure the speedup.
theano/misc/check_multi_gpu.py:This should be 2x if the GPUs are equivalent.
theano/misc/check_multi_gpu.py:from theano.gpuarray import init_dev
theano/misc/check_multi_gpu.py:from theano.gpuarray.blas import gpu_dot22
theano/misc/check_multi_gpu.py:    f1 = theano.function([], [gpu_dot22(val1a, val1b),
theano/misc/check_multi_gpu.py:                              gpu_dot22(val1c, val1d)])
theano/misc/check_multi_gpu.py:    f2 = theano.function([], [gpu_dot22(val1a, val1b),
theano/misc/check_multi_gpu.py:                              gpu_dot22(val2a, val2b)])
theano/misc/check_multi_gpu.py:    f3 = theano.function([], [gpu_dot22(val1a, val1b)])
theano/misc/check_multi_gpu.py:    f4 = theano.function([], [gpu_dot22(val2a, val2b)])
theano/misc/check_multi_gpu.py:    f5 = theano.function([], [gpu_dot22(val1a, val1b)[0, 0].transfer('cpu')])
theano/misc/check_multi_gpu.py:    f6 = theano.function([], [gpu_dot22(val2a, val2b)[0, 0].transfer('cpu')])
theano/misc/check_multi_gpu.py:    # pre-execute to load code to GPU.
theano/misc/do_nightly_build:FLAGS=${THEANO_FLAGS},warn.argmax_pushdown_bug=False,warn.gpusum_01_011_0111_bug=False,warn.sum_sum_bug=False,warn.sum_div_dimshuffle_bug=False,warn.subtensor_merge_bug=False,$FLAGS
theano/d3viz/d3viz.py:    ellipses are transfers from/to the GPU (ops with names GpuFromHost,
theano/d3viz/d3viz.py:    HostFromGpu).
theano/d3viz/formatting.py:        self.apply_colors = {'GpuFromHost': 'red',
theano/d3viz/formatting.py:                             'HostFromGpu': 'red',
theano/gpuarray/nerv.py:    "You are importing theano.gpuarray.nerv. "
theano/gpuarray/nerv.py:    "This module was removed as it was based on nervanagpu that is now deprecated. "
theano/gpuarray/nerv.py:    "More info about nervanagpu here: https://github.com/NervanaSystems/nervanagpu "
theano/gpuarray/elemwise.py:    import pygpu
theano/gpuarray/elemwise.py:    from pygpu import gpuarray
theano/gpuarray/elemwise.py:    from pygpu.tools import ArrayArg
theano/gpuarray/elemwise.py:    from pygpu.reduction import ReductionKernel
theano/gpuarray/elemwise.py:    from pygpu.gpuarray import dtype_to_typecode
theano/gpuarray/elemwise.py:from .basic_ops import (as_gpuarray_variable, HideC, GpuKernelBase, Kernel,
theano/gpuarray/elemwise.py:from .type import GpuArrayType, gpu_context_type
theano/gpuarray/elemwise.py:def max_inputs_to_GpuElemwise(node_or_outputs):
theano/gpuarray/elemwise.py:    # we take the limit from CUDA for now
theano/gpuarray/elemwise.py:class GpuElemwise(HideC, Elemwise):
theano/gpuarray/elemwise.py:    Elemwise on the GPU.
theano/gpuarray/elemwise.py:    params_type = gpu_context_type
theano/gpuarray/elemwise.py:        return "GpuElemwise{%s}%s<gpuarray>" % (self.scalar_op, items)
theano/gpuarray/elemwise.py:        return max_inputs_to_GpuElemwise(node_or_outputs)
theano/gpuarray/elemwise.py:        inputs = [as_gpuarray_variable(i, ctx_name) for i in inputs]
theano/gpuarray/elemwise.py:        out_info = Elemwise.get_output_info(self, GpuDimShuffle, *inputs)
theano/gpuarray/elemwise.py:        outputs = [GpuArrayType(broadcastable=br,
theano/gpuarray/elemwise.py:        if len(inputs) > max_inputs_to_GpuElemwise(outputs):
theano/gpuarray/elemwise.py:                "Can not make this GpuElemwise with that much inputs")
theano/gpuarray/elemwise.py:                    "struct aren't supported in GpuElemwise support_code" +
theano/gpuarray/elemwise.py:        # As float16 isn't a c type and most GPU don't compute on it,
theano/gpuarray/elemwise.py:        # We convert the computation to float32, and let libgpuarray
theano/gpuarray/elemwise.py:                "No c code for this scalar. Can not make a GpuElemwise")
theano/gpuarray/elemwise.py:        return ['<numpy_compat.h>', '<gpuarray/types.h>',
theano/gpuarray/elemwise.py:                '<gpuarray/elemwise.h>']
theano/gpuarray/elemwise.py:        return "\nGpuElemwise *ge;\n"
theano/gpuarray/elemwise.py:        gpuelemwise_arg args[%(nargs)s] = {{0}};
theano/gpuarray/elemwise.py:        ge = GpuElemwise_new(%(ctx)s->ctx, %(support)s, %(kop)s, %(nargs)s, args, %(nd)s, GE_CONVERT_F16);
theano/gpuarray/elemwise.py:        GpuElemwise_free(ge);
theano/gpuarray/elemwise.py:        if (%(nd)s != PyGpuArray_NDIM(%(iname)s))
theano/gpuarray/elemwise.py:                         PyGpuArray_NDIM(%(iname)s));
theano/gpuarray/elemwise.py:            dims[i] = (dims[i] == 1) ? PyGpuArray_DIMS(%(iname)s)[i] : dims[i];
theano/gpuarray/elemwise.py:                 PyGpuArray_DIMS(%(iname)s)[i] == 1)) &&
theano/gpuarray/elemwise.py:                (dims[i] != PyGpuArray_DIMS(%(iname)s)[i]))
theano/gpuarray/elemwise.py:                             "GpuElemwise. Input dimension mis-match. Input"
theano/gpuarray/elemwise.py:                             (unsigned long long)PyGpuArray_DIMS(%(iname)s)[i],
theano/gpuarray/elemwise.py:            if (dims[i] != PyGpuArray_DIMS(%(oname)s)[i])
theano/gpuarray/elemwise.py:        if (%(oname)s && !GpuArray_CHKFLAGS(&(%(oname)s->ga), GA_C_CONTIGUOUS))
theano/gpuarray/elemwise.py:            %(oname)s = pygpu_empty(%(nd)d, dims,
theano/gpuarray/elemwise.py:            if (dims[i] != PyGpuArray_DIMS(%(oname)s)[i])
theano/gpuarray/elemwise.py:                             "GpuElemwise. Output dimension mis-match. Output"
theano/gpuarray/elemwise.py:                             (unsigned long long)PyGpuArray_DIMS(%(oname)s)[i],
theano/gpuarray/elemwise.py:        if (GpuElemwise_call(ge, rargs, GE_BROADCAST) != GA_NO_ERROR) {
theano/gpuarray/elemwise.py:class GpuDimShuffle(DimShuffle):
theano/gpuarray/elemwise.py:    DimShuffle on the GPU.
theano/gpuarray/elemwise.py:    c_func_name = 'APPLY_SPECIFIC(gpu_dimshuffle)'
theano/gpuarray/elemwise.py:        otype = GpuArrayType(dtype=res.outputs[0].type.dtype,
theano/gpuarray/elemwise.py:        input = as_gpuarray_variable(input, ctx_name)
theano/gpuarray/elemwise.py:            s = "InplaceGpuDimShuffle{%s}"
theano/gpuarray/elemwise.py:            s = "GpuDimShuffle{%s}"
theano/gpuarray/elemwise.py:class GpuCAReduceCuda(GpuKernelBase, HideC, CAReduceDtype):
theano/gpuarray/elemwise.py:    GpuCAReduceCuda is a Reduction along some dimensions by a scalar op.
theano/gpuarray/elemwise.py:    This op was recently upgraded from just GpuSum a general CAReduce. Not
theano/gpuarray/elemwise.py:    GPUs are not especially well-suited to reduction operations so it is
theano/gpuarray/elemwise.py:    quite possible that the GPU might be slower for some cases.
theano/gpuarray/elemwise.py:        return "GpuCAReduceCuda{%s%s}%s" % (pre, str(self.scalar_op), ax)
theano/gpuarray/elemwise.py:        x = as_gpuarray_variable(x, infer_context_name(x))
theano/gpuarray/elemwise.py:        if x.type.context.kind != b'cuda':
theano/gpuarray/elemwise.py:            raise TypeError("GpuCAReduceCuda doesn't work for non-cuda devices")
theano/gpuarray/elemwise.py:        ret = super(GpuCAReduceCuda, self).make_node(x)
theano/gpuarray/elemwise.py:            raise NotImplementedError("We don't support complex in gpu reduction")
theano/gpuarray/elemwise.py:        return Apply(self, [x], [GpuArrayType(ret.outputs[0].dtype,
theano/gpuarray/elemwise.py:        # local_gpu_sum)
theano/gpuarray/elemwise.py:            if not self.gpu_kernels(node, name):
theano/gpuarray/elemwise.py:        return ['<numpy_compat.h>', '<gpuarray/types.h>']
theano/gpuarray/elemwise.py:        # have it run. But libgpuarray don't understand it.
theano/gpuarray/elemwise.py:        if (PyGpuArray_NDIM(%(x)s) != %(nd_in)s)
theano/gpuarray/elemwise.py:                         "required nd=%(nd_in)s, got nd=%%u", PyGpuArray_NDIM(%(x)s));
theano/gpuarray/elemwise.py:            conds = ["(PyGpuArray_DIMS(%s)[%d] == 0)" % (x, i)
theano/gpuarray/elemwise.py:           || (PyGpuArray_NDIM(%(z)s) != %(nd_out)s)
theano/gpuarray/elemwise.py:                print(" || (PyGpuArray_DIMS(%(z)s)[%(j)s] != PyGpuArray_DIMS(%(x)s)[%(i)d]) " % locals(), file=sio)
theano/gpuarray/elemwise.py:                print('new_dims[%(j)s] = PyGpuArray_DIMS(%(x)s)[%(i)s];' % locals(), file=sio)
theano/gpuarray/elemwise.py:            %(z)s = pygpu_empty(%(nd_out)s, new_dims,
theano/gpuarray/elemwise.py:            zero_shp = "GpuArray_memset(&%(z)s->ga, 0)" % locals()
theano/gpuarray/elemwise.py:                         "GpuCAReduceCuda not implemented when input shape is 0"
theano/gpuarray/elemwise.py:        if (PyGpuArray_SIZE(%(z)s) && ! PyGpuArray_SIZE(%(x)s)){
theano/gpuarray/elemwise.py:        else if (PyGpuArray_SIZE(%(z)s))
theano/gpuarray/elemwise.py:                ssize_t stride_A0 = PyGpuArray_STRIDES(%(x)s)[0]/sizeof(%(in_dtype)s);
theano/gpuarray/elemwise.py:                ssize_t stride_A1 = PyGpuArray_STRIDES(%(x)s)[1]/sizeof(%(in_dtype)s);
theano/gpuarray/elemwise.py:                ssize_t stride_Z0 = PyGpuArray_STRIDES(%(z)s)[0]/sizeof(%(out_dtype)s);
theano/gpuarray/elemwise.py:                        (void *)&PyGpuArray_DIMS(%(x)s)[0],
theano/gpuarray/elemwise.py:                        (void *)&PyGpuArray_DIMS(%(x)s)[1],
theano/gpuarray/elemwise.py:                int err = GpuKernel_call(&%(k_var)s, 3, n_blocks, n_threads, n_shared, kernel_params);
theano/gpuarray/elemwise.py:        shapes_data = ",".join(["(size_t) PyGpuArray_DIMS(%s)[%d]" % (x, i)
theano/gpuarray/elemwise.py:            params.append("(void *)&PyGpuArray_DIMS(%(x)s)[%(i)s]" % locals())
theano/gpuarray/elemwise.py:            ssize_t stride_A%(i)d = PyGpuArray_STRIDES(%(x)s)[%(i)s]/sizeof(%(in_dtype)s);
theano/gpuarray/elemwise.py:            ssize_t stride_Z%(i)d = PyGpuArray_STRIDES(%(z)s)[%(i)s]/sizeof(%(out_dtype)s);
theano/gpuarray/elemwise.py:                             "gpuarray error: %(k_var)s: %%s.",
theano/gpuarray/elemwise.py:                             GpuKernel_error(&%(k_var)s, err));
theano/gpuarray/elemwise.py:            int err = GpuKernel_call(&%(k_var)s, 3, n_blocks, n_threads, n_shared, kernel_params);
theano/gpuarray/elemwise.py:        in_type = gpuarray.dtype_to_ctype(in_dtype)
theano/gpuarray/elemwise.py:        out_type = gpuarray.dtype_to_ctype(out_dtype)
theano/gpuarray/elemwise.py:        params.append(gpuarray.GpuArray)
theano/gpuarray/elemwise.py:        params.append(gpuarray.GpuArray)
theano/gpuarray/elemwise.py:        in_type = gpuarray.dtype_to_ctype(in_dtype)
theano/gpuarray/elemwise.py:        out_type = gpuarray.dtype_to_ctype(out_dtype)
theano/gpuarray/elemwise.py:        acc_type = gpuarray.dtype_to_ctype(acc_dtype)
theano/gpuarray/elemwise.py:            zero_shp = "GpuArray_memset(&%(z)s->ga, 0)" % locals()
theano/gpuarray/elemwise.py:                         "GpuCAReduceCuda not implemented when input shape is 0 for this scalar_op");
theano/gpuarray/elemwise.py:                             "gpuarray error: %(k_var)s: %%s.",
theano/gpuarray/elemwise.py:                             GpuKernel_error(&%(k_var)s, err));
theano/gpuarray/elemwise.py:          if(PyGpuArray_SIZE(%(x)s)==0){
theano/gpuarray/elemwise.py:            size_t numEls = PyGpuArray_SIZE(%(x)s);
theano/gpuarray/elemwise.py:                                PyGpuArray_NDIM(%(x)s));
theano/gpuarray/elemwise.py:            int err = GpuKernel_call(&%(k_var)s, 1, &n_blocks, &n_threads, n_shared, kernel_params);
theano/gpuarray/elemwise.py:            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t) 256), 1, 1};
theano/gpuarray/elemwise.py:            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[1], (size_t) 256), 1, 1};
theano/gpuarray/elemwise.py:            if (n_threads[1] > PyGpuArray_DIMS(%(x)s)[0])
theano/gpuarray/elemwise.py:                n_threads[1] = PyGpuArray_DIMS(%(x)s)[0];
theano/gpuarray/elemwise.py:        param_dim = ",".join(["PyGpuArray_DIMS(%s)[%d]" % (x, i)
theano/gpuarray/elemwise.py:        strides_dim = ",".join(["PyGpuArray_STRIDES(%s)[%d]/sizeof(%s)"
theano/gpuarray/elemwise.py:                if (n_threads[1] < PyGpuArray_DIMS(%(x)s)[%(N)s-1])
theano/gpuarray/elemwise.py:                if (n_threads[2] < PyGpuArray_DIMS(%(x)s)[%(N)s-2])
theano/gpuarray/elemwise.py:            //Maximum for Fermi GPU on that dimensions.
theano/gpuarray/elemwise.py:            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[%(N)s], (size_t) 256), 1, 1};
theano/gpuarray/elemwise.py:            size_t n_blocks[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t) 4096), 1, 1};
theano/gpuarray/elemwise.py:                             "gpuarray error: %(k_var)s: %%s.",
theano/gpuarray/elemwise.py:                             GpuKernel_error(%(k_var)s, err));
theano/gpuarray/elemwise.py:        if(PyGpuArray_STRIDES(%(x)s)[0]>
theano/gpuarray/elemwise.py:           PyGpuArray_STRIDES(%(x)s)[1]){
theano/gpuarray/elemwise.py:                GpuKernel *%(k_var)s = &kernel_reduce_010_AD_%(name)s;
theano/gpuarray/elemwise.py:                size_t B = PyGpuArray_DIMS(%(x)s)[0];
theano/gpuarray/elemwise.py:                size_t C = PyGpuArray_DIMS(%(x)s)[1];
theano/gpuarray/elemwise.py:                ssize_t stride_A1 = PyGpuArray_STRIDES(%(x)s)[0]/sizeof(%(in_dtype)s);
theano/gpuarray/elemwise.py:                ssize_t stride_A2 = PyGpuArray_STRIDES(%(x)s)[1]/sizeof(%(in_dtype)s);
theano/gpuarray/elemwise.py:                ssize_t stride_Z1 = PyGpuArray_STRIDES(%(z)s)[0]/sizeof(%(out_dtype)s);
theano/gpuarray/elemwise.py:                int err = GpuKernel_call(%(k_var)s, 3, n_blocks, n_threads, 0, kernel_params);
theano/gpuarray/elemwise.py:            GpuKernel *%(k_var)s = &kernel_reduce_010_%(name)s;
theano/gpuarray/elemwise.py:            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t) 256), 1, 1};
theano/gpuarray/elemwise.py:            size_t n_blocks[3] = {1, std::min(PyGpuArray_DIMS(%(x)s)[1], (size_t) 4096), 1};
theano/gpuarray/elemwise.py:            assert(PyGpuArray_DIMS(%(x)s)[1] == PyGpuArray_DIMS(%(z)s)[0]);
theano/gpuarray/elemwise.py:            ssize_t stride_A1 = PyGpuArray_STRIDES(%(x)s)[0]/sizeof(%(in_dtype)s);
theano/gpuarray/elemwise.py:            ssize_t stride_A2 = PyGpuArray_STRIDES(%(x)s)[1]/sizeof(%(in_dtype)s);
theano/gpuarray/elemwise.py:            ssize_t stride_Z1 = PyGpuArray_STRIDES(%(z)s)[0]/sizeof(%(out_dtype)s);
theano/gpuarray/elemwise.py:                    (void *)&PyGpuArray_DIMS(%(x)s)[0],
theano/gpuarray/elemwise.py:                    (void *)&PyGpuArray_DIMS(%(x)s)[1],
theano/gpuarray/elemwise.py:            int err = GpuKernel_call(%(k_var)s, 3, n_blocks, n_threads, n_shared, kernel_params);
theano/gpuarray/elemwise.py:                             "gpuarray error: %(k_var)s: %%s.",
theano/gpuarray/elemwise.py:                             GpuKernel_error(&%(k_var)s, err));
theano/gpuarray/elemwise.py:            //int n_summations = PyGpuArray_DIMS(%(x)s)[0] * PyGpuArray_DIMS(%(x)s)[2];
theano/gpuarray/elemwise.py:            //if ((n_summations >= 15 * 32) && (PyGpuArray_DIMS(%(x)s)[2]>=16))
theano/gpuarray/elemwise.py:                size_t A = PyGpuArray_DIMS(%(x)s)[0];
theano/gpuarray/elemwise.py:                size_t B = PyGpuArray_DIMS(%(x)s)[1];
theano/gpuarray/elemwise.py:                size_t C = PyGpuArray_DIMS(%(x)s)[2];
theano/gpuarray/elemwise.py:                ssize_t stride_A0 = PyGpuArray_STRIDES(%(x)s)[0]/sizeof(%(in_dtype)s);
theano/gpuarray/elemwise.py:                ssize_t stride_A1 = PyGpuArray_STRIDES(%(x)s)[1]/sizeof(%(in_dtype)s);
theano/gpuarray/elemwise.py:                ssize_t stride_A2 = PyGpuArray_STRIDES(%(x)s)[2]/sizeof(%(in_dtype)s);
theano/gpuarray/elemwise.py:                ssize_t stride_Z0 = PyGpuArray_STRIDES(%(z)s)[0]/sizeof(%(out_dtype)s);
theano/gpuarray/elemwise.py:                ssize_t stride_Z1 = PyGpuArray_STRIDES(%(z)s)[1]/sizeof(%(out_dtype)s);
theano/gpuarray/elemwise.py:                int err = GpuKernel_call(&%(k_var)s, 3, n_blocks, n_threads, 0, kernel_params);
theano/gpuarray/elemwise.py:                  size_t n_threads[3] = {std::min((size_t) 32, PyGpuArray_DIMS(%(x)s)[2]), 1, 1};
theano/gpuarray/elemwise.py:                         && (n_threads[1]<PyGpuArray_DIMS(%(x)s)[1])){
theano/gpuarray/elemwise.py:                  size_t n_blocks[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t)4096), 1, 1};
theano/gpuarray/elemwise.py:                      ceil_intdiv(PyGpuArray_DIMS(%(x)s)[2],
theano/gpuarray/elemwise.py:                if(std::min(std::min(PyGpuArray_STRIDES(%(x)s)[0]/sizeof(%(in_dtype)s),
theano/gpuarray/elemwise.py:                                     PyGpuArray_STRIDES(%(x)s)[1]/sizeof(%(in_dtype)s)),
theano/gpuarray/elemwise.py:                            PyGpuArray_STRIDES(%(x)s)[2]/sizeof(%(in_dtype)s))
theano/gpuarray/elemwise.py:                   ==PyGpuArray_STRIDES(%(x)s)[2]/sizeof(%(in_dtype)s)
theano/gpuarray/elemwise.py:                  && n_blocks[1]==ceil_intdiv(PyGpuArray_DIMS(%(x)s)[2],
theano/gpuarray/elemwise.py:                           PyGpuArray_DIMS(%(x)s)[0],4096,
theano/gpuarray/elemwise.py:                           ceil_intdiv(PyGpuArray_DIMS(%(x)s)[2],(size_t)n_threads[0]),
theano/gpuarray/elemwise.py:                  n_threads[0] = std::min(PyGpuArray_DIMS(%(x)s)[1],
theano/gpuarray/elemwise.py:                  n_blocks[0] = std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t)4096);
theano/gpuarray/elemwise.py:                      PyGpuArray_DIMS(%(x)s)[2],
theano/gpuarray/elemwise.py:            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[3], (size_t) 256), 1, 1};
theano/gpuarray/elemwise.py:                if (n_threads[1] > PyGpuArray_DIMS(%(x)s)[1]) break;
theano/gpuarray/elemwise.py:            size_t n_blocks[3] = {PyGpuArray_DIMS(%(x)s)[0], PyGpuArray_DIMS(%(x)s)[2], 1};
theano/gpuarray/elemwise.py:                             "gpuarray error: %(k_var)s: %%s.",
theano/gpuarray/elemwise.py:                             GpuKernel_error(&%(k_var)s, err));
theano/gpuarray/elemwise.py:            if (PyGpuArray_STRIDES(%(x)s)[2] != sizeof(%(in_dtype)s)){
theano/gpuarray/elemwise.py:                size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t) 256), 1, 1};
theano/gpuarray/elemwise.py:                size_t n_blocks[3] = {std::min(PyGpuArray_DIMS(%(x)s)[1], (size_t)4096), 1, 1};
theano/gpuarray/elemwise.py:                       n_blocks[1] <= PyGpuArray_DIMS(%(x)s)[2])
theano/gpuarray/elemwise.py:                size_t A = PyGpuArray_DIMS(%(x)s)[1];
theano/gpuarray/elemwise.py:                size_t B = PyGpuArray_DIMS(%(x)s)[0];
theano/gpuarray/elemwise.py:                size_t C = PyGpuArray_DIMS(%(x)s)[2];
theano/gpuarray/elemwise.py:                ssize_t stride_A0 = PyGpuArray_STRIDES(%(x)s)[1]/sizeof(%(in_dtype)s);
theano/gpuarray/elemwise.py:                ssize_t stride_A1 = PyGpuArray_STRIDES(%(x)s)[0]/sizeof(%(in_dtype)s);
theano/gpuarray/elemwise.py:                ssize_t stride_A2 = PyGpuArray_STRIDES(%(x)s)[2]/sizeof(%(in_dtype)s);
theano/gpuarray/elemwise.py:                ssize_t stride_Z0 = PyGpuArray_STRIDES(%(z)s)[0]/sizeof(%(out_dtype)s);
theano/gpuarray/elemwise.py:                ssize_t stride_Z1 = PyGpuArray_STRIDES(%(z)s)[1]/sizeof(%(out_dtype)s);
theano/gpuarray/elemwise.py:                int err = GpuKernel_call(&%(k_var)s, 3, n_blocks, n_threads, 0, kernel_params);
theano/gpuarray/elemwise.py:            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[1], (size_t) 256), 1, 1};
theano/gpuarray/elemwise.py:                if (n_threads[1] > PyGpuArray_DIMS(%(x)s)[0])
theano/gpuarray/elemwise.py:            size_t n_blocks[3] = {PyGpuArray_DIMS(%(x)s)[2], 1, 1};
theano/gpuarray/elemwise.py:            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[2], (size_t) 256), 1, 1};
theano/gpuarray/elemwise.py:            size_t n_blocks[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t) 4096), 1, 1};
theano/gpuarray/elemwise.py:                if (n_blocks[1] > PyGpuArray_DIMS(%(x)s)[1])
theano/gpuarray/elemwise.py://            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[3],
theano/gpuarray/elemwise.py:            if (n_threads[1] > PyGpuArray_DIMS(%(x)s)[2])
theano/gpuarray/elemwise.py:                n_threads[1] = PyGpuArray_DIMS(%(x)s)[2];
theano/gpuarray/elemwise.py:            if (n_threads[2] > PyGpuArray_DIMS(%(x)s)[0])
theano/gpuarray/elemwise.py:                n_threads[2] = PyGpuArray_DIMS(%(x)s)[0];
theano/gpuarray/elemwise.py:            size_t n_blocks[3] = {PyGpuArray_DIMS(%(x)s)[1], 1, 1};
theano/gpuarray/elemwise.py:            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[2], (size_t) 256), 1, 1};
theano/gpuarray/elemwise.py:                if (n_threads[1] > PyGpuArray_DIMS(%(x)s)[1])
theano/gpuarray/elemwise.py:                if (n_threads[2] > PyGpuArray_DIMS(%(x)s)[0])
theano/gpuarray/elemwise.py:            //Maximum for Fermi GPU on that dimensions.
theano/gpuarray/elemwise.py:            size_t n_blocks[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t) 4096), 1, 1};
theano/gpuarray/elemwise.py:                   n_blocks[1] < PyGpuArray_DIMS(%(x)s)[1])
theano/gpuarray/elemwise.py:            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[3], (size_t) 256), 1, 1};
theano/gpuarray/elemwise.py:                   && n_threads[1] < PyGpuArray_DIMS(%(x)s)[2]
theano/gpuarray/elemwise.py:            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[2], (size_t) 256), 1, 1};
theano/gpuarray/elemwise.py:                if (n_threads[1] > PyGpuArray_DIMS(%(x)s)[1])
theano/gpuarray/elemwise.py:                if (n_threads[2] > PyGpuArray_DIMS(%(x)s)[0])
theano/gpuarray/elemwise.py:            //Maximum for Fermi GPU on that dimensions.
theano/gpuarray/elemwise.py:            size_t n_threads[3] = {std::min(PyGpuArray_DIMS(%(x)s)[3], (size_t) 256), 1, 1};
theano/gpuarray/elemwise.py:            if (n_threads[1] > PyGpuArray_DIMS(%(x)s)[2])
theano/gpuarray/elemwise.py:                n_threads[1] = PyGpuArray_DIMS(%(x)s)[2];
theano/gpuarray/elemwise.py:            if (n_threads[2] > PyGpuArray_DIMS(%(x)s)[0])
theano/gpuarray/elemwise.py:                n_threads[2] = PyGpuArray_DIMS(%(x)s)[0];
theano/gpuarray/elemwise.py:            size_t n_blocks[3] = {PyGpuArray_DIMS(%(x)s)[1], 1, 1};
theano/gpuarray/elemwise.py:    def gpu_kernels(self, node, nodename):
theano/gpuarray/elemwise.py:        in_type = gpuarray.dtype_to_ctype(in_dtype)
theano/gpuarray/elemwise.py:        out_type = gpuarray.dtype_to_ctype(out_dtype)
theano/gpuarray/elemwise.py:        acc_type = gpuarray.dtype_to_ctype(acc_dtype)
theano/gpuarray/elemwise.py:                gpuarray.GpuArray, 'uintp',
theano/gpuarray/elemwise.py:                gpuarray.GpuArray, 'uintp'
theano/gpuarray/elemwise.py:                gpuarray.GpuArray, 'uintp',
theano/gpuarray/elemwise.py:                gpuarray.GpuArray, 'uintp'
theano/gpuarray/elemwise.py:                gpuarray.GpuArray, 'uintp',
theano/gpuarray/elemwise.py:                gpuarray.GpuArray, 'uintp'
theano/gpuarray/elemwise.py:                gpuarray.GpuArray, 'uintp',
theano/gpuarray/elemwise.py:                gpuarray.GpuArray, 'uintp',
theano/gpuarray/elemwise.py:                gpuarray.GpuArray, 'uintp',
theano/gpuarray/elemwise.py:                gpuarray.GpuArray, 'uintp',
theano/gpuarray/elemwise.py:                gpuarray.GpuArray, 'uintp',
theano/gpuarray/elemwise.py:                gpuarray.GpuArray, 'uintp',
theano/gpuarray/elemwise.py:                gpuarray.GpuArray, 'uintp',
theano/gpuarray/elemwise.py:                gpuarray.GpuArray, 'uintp',
theano/gpuarray/elemwise.py:                gpuarray.GpuArray, 'uintp',
theano/gpuarray/elemwise.py:                gpuarray.GpuArray, 'uintp',
theano/gpuarray/elemwise.py:class GpuErfinv(Erfinv):
theano/gpuarray/elemwise.py:    Inverse error function for GPU.
theano/gpuarray/elemwise.py:        # NB: CUDA erfinv function (GPU op) returns NaN if x not in [-1;1],
theano/gpuarray/elemwise.py:        # For consistency of CPU and GPU ops, we wrap the CUDA erfinv in the following conditions
theano/gpuarray/elemwise.py:        # to ensure that GPU op returns the same values as CPU op.
theano/gpuarray/elemwise.py:gpu_erfinv = GpuErfinv(upgrade_to_float_no_complex, name='gpu_erfinv')
theano/gpuarray/elemwise.py:class GpuErfcinv(Erfcinv):
theano/gpuarray/elemwise.py:    Inverse complementary error function for GPU.
theano/gpuarray/elemwise.py:        # NB: CUDA erfcinv function (GPU op) returns NaN if x not in [0;2],
theano/gpuarray/elemwise.py:        # For consistency of CPU and GPU ops, we wrap the CUDA erfcinv in the following conditions
theano/gpuarray/elemwise.py:        # to ensure that GPU op returns the same values as CPU op.
theano/gpuarray/elemwise.py:gpu_erfcinv = GpuErfcinv(upgrade_to_float_no_complex, name='gpu_erfcinv')
theano/gpuarray/elemwise.py:# Caching GpuCAReduceCuda
theano/gpuarray/elemwise.py:def gpu_ca_reduce_cuda(scalar_op, axis=None, reduce_mask=None, dtype=None, acc_dtype=None,
theano/gpuarray/elemwise.py:    if key not in gpu_ca_reduce_cuda.cache:
theano/gpuarray/elemwise.py:        gpu_ca_reduce_cuda.cache[key] = GpuCAReduceCuda(scalar_op, axis, reduce_mask, dtype,
theano/gpuarray/elemwise.py:    return gpu_ca_reduce_cuda.cache[key]
theano/gpuarray/elemwise.py:gpu_ca_reduce_cuda.cache = {}
theano/gpuarray/elemwise.py:class GpuCAReduceCPY(GpuKernelBase, HideC, CAReduceDtype):
theano/gpuarray/elemwise.py:    CAReduce that reuse the python code from gpuarray.
theano/gpuarray/elemwise.py:        return "GpuReduce{%s}%s" % (self.scalar_op, ax)
theano/gpuarray/elemwise.py:        input = as_gpuarray_variable(input, ctx_name)
theano/gpuarray/elemwise.py:        otype = GpuArrayType(dtype=res.outputs[0].dtype,
theano/gpuarray/elemwise.py:    def gpu_kernels(self, node, name):
theano/gpuarray/elemwise.py:            # Some OpenCL compilers do not accept no-arguments empty kernels
theano/gpuarray/elemwise.py:            params = ['uint32', gpuarray.GpuArray, 'uint32']
theano/gpuarray/elemwise.py:            params.append(gpuarray.GpuArray)
theano/gpuarray/elemwise.py:            # We special case the no-reduction case since the gpu
theano/gpuarray/elemwise.py:        %(out)s = pygpu_copy(%(inp)s, GA_ANY_ORDER);
theano/gpuarray/elemwise.py:        PyGpuArrayObject *tmp;
theano/gpuarray/elemwise.py:             %(output)s = pygpu_empty(%(nd_out)s, out_dims, %(out_type)s, GA_C_ORDER, %(ctx)s, Py_None);
theano/gpuarray/elemwise.py:            %(output)s = pygpu_empty(0, NULL, %(out_type)s, GA_C_ORDER,
theano/gpuarray/elemwise.py:        tmp = pygpu_empty(%(output)s->ga.nd, %(output)s->ga.dimensions,
theano/gpuarray/elemwise.py:        err = GpuKernel_call(&%(k_var)s, 1, &gs, &ls, 0, args);
theano/gpuarray/elemwise.py:                         "gpuarray error: GpuCAReduceCPY: %%s.",
theano/gpuarray/elemwise.py:                         GpuKernel_error(&%(k_var)s, err));
theano/gpuarray/elemwise.py:            err = GpuArray_move(&%(output)s->ga, &tmp->ga);
theano/gpuarray/elemwise.py:                             "gpuarray error: GpuCAReduceCPY [cast]: %%s.",
theano/gpuarray/elemwise.py:                             GpuArray_error(&tmp->ga, err));
theano/gpuarray/elemwise.py:            output[0] = pygpu.gpuarray.array(input, copy=True,
theano/gpuarray/elemwise.py:GpuCAReduce = GpuCAReduceCPY
theano/gpuarray/opt.py:from .type import (GpuArrayType, GpuArrayConstant, get_context,
theano/gpuarray/opt.py:                   ContextNotDefined, move_to_gpu)
theano/gpuarray/opt.py:from .basic_ops import (as_gpuarray_variable, infer_context_name,
theano/gpuarray/opt.py:                        host_from_gpu, GpuToGpu,
theano/gpuarray/opt.py:                        HostFromGpu, GpuFromHost,
theano/gpuarray/opt.py:                        GpuSplit, GpuContiguous, gpu_contiguous,
theano/gpuarray/opt.py:                        GpuAlloc, GpuAllocEmpty, GpuReshape,
theano/gpuarray/opt.py:                        GpuEye, GpuTri, gpu_join, GpuJoin)
theano/gpuarray/opt.py:from .blas import (gpu_dot22, GpuGemm, GpuGer, GpuGemmBatch,
theano/gpuarray/opt.py:                   gpugemm_no_inplace, gpugemm_inplace,
theano/gpuarray/opt.py:                   gpugemmbatch_no_inplace,
theano/gpuarray/opt.py:                   gpugemv_no_inplace, gpugemv_inplace,
theano/gpuarray/opt.py:                   GpuCorrMM, GpuCorrMM_gradInputs, GpuCorrMM_gradWeights,
theano/gpuarray/opt.py:                   GpuCorr3dMM, GpuCorr3dMM_gradInputs, GpuCorr3dMM_gradWeights)
theano/gpuarray/opt.py:from .pool import (GpuPool, GpuMaxPoolGrad, GpuAveragePoolGrad, GpuMaxPoolRop,
theano/gpuarray/opt.py:                   GpuDownsampleFactorMaxGradGrad)
theano/gpuarray/opt.py:from .blocksparse import (GpuSparseBlockGemv, GpuSparseBlockOuter,
theano/gpuarray/opt.py:                          gpu_sparse_block_outer,
theano/gpuarray/opt.py:                          gpu_sparse_block_outer_inplace,
theano/gpuarray/opt.py:                          gpu_sparse_block_gemv, gpu_sparse_block_gemv_inplace)
theano/gpuarray/opt.py:from .nnet import (gpu_crossentropy_softmax_1hot_with_bias_dx,
theano/gpuarray/opt.py:                   gpu_crossentropy_softmax_argmax_1hot_with_bias,
theano/gpuarray/opt.py:                   gpu_softmax_with_bias, gpu_softmax)
theano/gpuarray/opt.py:from .elemwise import (GpuElemwise, GpuDimShuffle, GpuCAReduceCuda,
theano/gpuarray/opt.py:                       GpuCAReduceCPY, gpu_erfinv, gpu_erfcinv,
theano/gpuarray/opt.py:                       max_inputs_to_GpuElemwise)
theano/gpuarray/opt.py:from .subtensor import (GpuIncSubtensor, GpuSubtensor,
theano/gpuarray/opt.py:                        GpuAdvancedSubtensor,
theano/gpuarray/opt.py:                        GpuAdvancedSubtensor1,
theano/gpuarray/opt.py:                        GpuAdvancedBooleanSubtensor,
theano/gpuarray/opt.py:                        GpuAdvancedIncSubtensor,
theano/gpuarray/opt.py:                        GpuAdvancedIncSubtensor1,
theano/gpuarray/opt.py:                        GpuAdvancedIncSubtensor1_dev20,
theano/gpuarray/opt.py:                        GpuAdvancedBooleanIncSubtensor,
theano/gpuarray/opt.py:                        GpuAllocDiag, GpuExtractDiag)
theano/gpuarray/opt.py:from .reduction import GpuMaxAndArgmax
theano/gpuarray/opt.py:from .linalg import (GpuCusolverSolve, MATRIX_STRUCTURES_SOLVE, GpuCholesky,
theano/gpuarray/opt.py:                     cusolver_available, GpuMagmaMatrixInverse, gpu_svd,
theano/gpuarray/opt.py:                     GpuMagmaCholesky, gpu_qr, GpuMagmaEigh,
theano/gpuarray/opt.py:                     GpuCublasTriangularSolve, cublas_available)
theano/gpuarray/opt.py:from .neighbours import GpuImages2Neibs
theano/gpuarray/opt.py:from .ctc import GpuConnectionistTemporalClassification
theano/gpuarray/opt.py:_logger = logging.getLogger("theano.gpuarray.opt")
theano/gpuarray/opt.py:gpu_optimizer = EquilibriumDB()
theano/gpuarray/opt.py:gpu_cut_copies = EquilibriumDB()
theano/gpuarray/opt.py:# Not used for an EquilibriumOptimizer. It has the "tracks" that we need for GraphToGPUDB.
theano/gpuarray/opt.py:gpu_optimizer2 = EquilibriumDB()
theano/gpuarray/opt.py:class GraphToGPUDB(DB):
theano/gpuarray/opt.py:        opt = gpu_optimizer2.query(*tags, **kwtags)
theano/gpuarray/opt.py:        return GraphToGPU(opt.local_optimizers_all, opt.local_optimizers_map)
theano/gpuarray/opt.py:gpu_seqopt = SequenceDB()
theano/gpuarray/opt.py:gpu_seqopt.register('gpuarray_graph_optimization', GraphToGPUDB(), -0.5,
theano/gpuarray/opt.py:                    'fast_compile', 'fast_run', 'gpuarray')
theano/gpuarray/opt.py:gpu_seqopt.register('gpuarray_local_optimizations', gpu_optimizer, 1,
theano/gpuarray/opt.py:                    'fast_compile', 'fast_run', 'gpuarray', 'gpuarray_local_optimiziations')
theano/gpuarray/opt.py:gpu_seqopt.register('gpuarray_cut_transfers', gpu_cut_copies, 2,
theano/gpuarray/opt.py:                    'fast_compile', 'fast_run', 'gpuarray')
theano/gpuarray/opt.py:# do not add 'fast_run' to these two as this would always enable gpuarray mode
theano/gpuarray/opt.py:optdb.register('gpuarray_opt', gpu_seqopt,
theano/gpuarray/opt.py:               'gpuarray')
theano/gpuarray/opt.py:        gpu_optimizer.register(name, local_opt, 'fast_run', 'gpuarray', *tags)
theano/gpuarray/opt.py:    Decorator for the new GraphToGPU optimizer.
theano/gpuarray/opt.py:        gpu_optimizer2.register(name, opt, 'fast_run', 'gpuarray', *tags)
theano/gpuarray/opt.py:            60, 'fast_run', 'inplace', 'gpuarray', *tags)
theano/gpuarray/opt.py:register_opt(final_opt=True, name='gpua_constant_folding')(
theano/gpuarray/opt.py:gpu_optimizer.register('local_remove_all_assert',
theano/gpuarray/opt.py:def safe_to_gpu(x, ctx_name):
theano/gpuarray/opt.py:        return GpuFromHost(ctx_name)(x)
theano/gpuarray/opt.py:    if isinstance(x.type, GpuArrayType):
theano/gpuarray/opt.py:gpu_log = GpuElemwise(log)
theano/gpuarray/opt.py:gpu_neg = GpuElemwise(neg)
theano/gpuarray/opt.py:gpu_true_div = GpuElemwise(true_div)
theano/gpuarray/opt.py:def op_lifter(OP, cuda_only=False):
theano/gpuarray/opt.py:    OP(..., host_from_gpu(), ...) -> host_from_gpu(GpuOP(...))
theano/gpuarray/opt.py:    gpu_from_host(OP(inp0, ...)) -> GpuOP(inp0, ...)
theano/gpuarray/opt.py:                # Either one of our inputs is on the gpu or
theano/gpuarray/opt.py:                # all of our clients are on the gpu
theano/gpuarray/opt.py:                # We replace if any input is a host_from_gpu
theano/gpuarray/opt.py:                    if (i.owner and i.owner.op == host_from_gpu and
theano/gpuarray/opt.py:                            move_to_gpu(i)):
theano/gpuarray/opt.py:                    # We replace if *all* clients are on the GPU
theano/gpuarray/opt.py:                                not isinstance(c.op, GpuFromHost)):
theano/gpuarray/opt.py:                        # All clients are GpuFromHost and we have at least one
theano/gpuarray/opt.py:                        (cuda_only and
theano/gpuarray/opt.py:                         get_context(context_name).kind != b'cuda') or
theano/gpuarray/opt.py:                    else:  # suppose it is a variable on the GPU
theano/gpuarray/opt.py:                    # copy stack traces onto gpu outputs
theano/gpuarray/opt.py:                    # also copy the stack traces onto HostFromGpu outputs
theano/gpuarray/opt.py:class InputToGpuOptimizer(Optimizer):
theano/gpuarray/opt.py:    Transfer the input to the gpu to start the rolling wave.
theano/gpuarray/opt.py:            if isinstance(input.type, GpuArrayType):
theano/gpuarray/opt.py:            if (all(cl[0] == 'output' or isinstance(cl[0].op, GpuFromHost)
theano/gpuarray/opt.py:                    not move_to_gpu(input)):
theano/gpuarray/opt.py:                new_input = GpuFromHost(target)(input).transfer('cpu')
theano/gpuarray/opt.py:                                        "InputToGpuOptimizer")
theano/gpuarray/opt.py:gpu_seqopt.register('InputToGpuArrayOptimizer', InputToGpuOptimizer(),
theano/gpuarray/opt.py:class GraphToGPU(Optimizer):
theano/gpuarray/opt.py:    Transfer the graph as a whole to GPU instead of transferring node by node.
theano/gpuarray/opt.py:            if isinstance(i.type, tensor.TensorType) and move_to_gpu(i):
theano/gpuarray/opt.py:            if isinstance(node.op, HostFromGpu):
theano/gpuarray/opt.py:            # Move only if any of the inputs are on the GPU.
theano/gpuarray/opt.py:            move_to_GPU = False
theano/gpuarray/opt.py:                if isinstance(i.type, GpuArrayType):
theano/gpuarray/opt.py:                    move_to_GPU = True
theano/gpuarray/opt.py:            if (not move_to_GPU and
theano/gpuarray/opt.py:                # to the GPU, we should move the Alloc* on the GPU.
theano/gpuarray/opt.py:                # move the client to the GPU.
theano/gpuarray/opt.py:                        move_to_GPU = True
theano/gpuarray/opt.py:            if move_to_GPU and any(["complex" in getattr(i, 'dtype', "")
theano/gpuarray/opt.py:                move_to_GPU = False
theano/gpuarray/opt.py:            if move_to_GPU:
theano/gpuarray/opt.py:                assert isinstance(new_o.type, GpuArrayType)
theano/gpuarray/opt.py:                # gpu.
theano/gpuarray/opt.py:                        isinstance(new_o.owner.op, GpuFromHost) and
theano/gpuarray/opt.py:        print(blanc, "GraphToGPUOptimizer", end=' ', file=stream)
theano/gpuarray/opt.py:        new_opt = GraphToGPU(local_optimizers, local_optimizers_map)
theano/gpuarray/opt.py:@local_optimizer([GpuFromHost, GpuToGpu, HostFromGpu])
theano/gpuarray/opt.py:def local_cut_gpu_transfers(node):
theano/gpuarray/opt.py:    # gpu[ab] -> host -> gpub
theano/gpuarray/opt.py:    if (isinstance(node.op, GpuFromHost) and
theano/gpuarray/opt.py:            isinstance(node.inputs[0].owner.op, HostFromGpu)):
theano/gpuarray/opt.py:            return [GpuToGpu(node.op.context_name)(other)]
theano/gpuarray/opt.py:    # ? -> gpua -> host
theano/gpuarray/opt.py:    elif (isinstance(node.op, HostFromGpu) and
theano/gpuarray/opt.py:        if isinstance(n2.op, GpuFromHost):
theano/gpuarray/opt.py:        # gpub ->
theano/gpuarray/opt.py:        if isinstance(n2.op, GpuToGpu):
theano/gpuarray/opt.py:    # ? -> gpua -> gpub
theano/gpuarray/opt.py:    elif isinstance(node.op, GpuToGpu):
theano/gpuarray/opt.py:            if isinstance(n2.op, GpuFromHost):
theano/gpuarray/opt.py:                return [as_gpuarray_variable(n2.inputs[0],
theano/gpuarray/opt.py:            # gpuc ->
theano/gpuarray/opt.py:            if isinstance(n2.op, GpuToGpu):
theano/gpuarray/opt.py:gpu_cut_copies.register('cut_gpua_host_transfers', local_cut_gpu_transfers,
theano/gpuarray/opt.py:                        'fast_compile', 'fast_run', 'gpuarray')
theano/gpuarray/opt.py:gpu_cut_copies.register('cut_gpua_constant_transfers',
theano/gpuarray/opt.py:                        'fast_compile', 'fast_run', 'gpuarray')
theano/gpuarray/opt.py:optdb['canonicalize'].register('local_cut_gpua_host_gpua',
theano/gpuarray/opt.py:                               local_cut_gpu_transfers,
theano/gpuarray/opt.py:                               'fast_compile', 'fast_run', 'gpuarray')
theano/gpuarray/opt.py:def local_gpua_alloc2(node):
theano/gpuarray/opt.py:    Join(axis, {Alloc or HostFromGPU}, ...) -> Join(axis, GpuAlloc, Alloc, ...)
theano/gpuarray/opt.py:    Moves an alloc that is an input to join to the gpu.
theano/gpuarray/opt.py:                i.owner.op in [host_from_gpu, tensor.alloc]
theano/gpuarray/opt.py:        return [GpuAlloc(None)(*node.inputs).transfer('cpu')]
theano/gpuarray/opt.py:def local_gpuaalloc(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return GpuAlloc(context_name)(*inputs)
theano/gpuarray/opt.py:def local_gpua_alloc_empty(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    # We use _props_dict() to make sure that the GPU op know all the
theano/gpuarray/opt.py:    return GpuAllocEmpty(context_name=context_name, **op._props_dict())(*inputs)
theano/gpuarray/opt.py:@local_optimizer([GpuAlloc])
theano/gpuarray/opt.py:def local_gpualloc_memset_0(node):
theano/gpuarray/opt.py:    if isinstance(node.op, GpuAlloc) and not node.op.memset_0:
theano/gpuarray/opt.py:        if (isinstance(inp, GpuArrayConstant) and
theano/gpuarray/opt.py:            new_op = GpuAlloc(node.op.context_name, memset_0=True)
theano/gpuarray/opt.py:@gof.local_optimizer([GpuAllocEmpty])
theano/gpuarray/opt.py:def local_gpua_alloc_empty_to_zeros(node):
theano/gpuarray/opt.py:    if isinstance(node.op, GpuAllocEmpty):
theano/gpuarray/opt.py:            return [GpuAlloc(context_name)(
theano/gpuarray/opt.py:                as_gpuarray_variable(z, context_name), *node.inputs)]
theano/gpuarray/opt.py:optdb.register('local_gpua_alloc_empty_to_zeros',
theano/gpuarray/opt.py:               theano.tensor.opt.in2out(local_gpua_alloc_empty_to_zeros),
theano/gpuarray/opt.py:               # After move to gpu and merge2, before inplace.
theano/gpuarray/opt.py:@local_optimizer([GpuContiguous])
theano/gpuarray/opt.py:def local_gpu_contiguous_gpu_contiguous(node):
theano/gpuarray/opt.py:    gpu_contiguous(gpu_contiguous(x)) -> gpu_contiguous(x)
theano/gpuarray/opt.py:    if isinstance(node.op, GpuContiguous):
theano/gpuarray/opt.py:        if inp.owner and isinstance(inp.owner.op, GpuContiguous):
theano/gpuarray/opt.py:def local_gpua_contiguous(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return gpu_contiguous
theano/gpuarray/opt.py:def local_gpua_reshape(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    res = GpuReshape(op.ndim)
theano/gpuarray/opt.py:def local_gpua_rebroadcast(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return op(as_gpuarray_variable(inputs[0], context_name))
theano/gpuarray/opt.py:def local_gpua_flatten(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    res = GpuReshape(op.outdim)
theano/gpuarray/opt.py:def local_gpua_elemwise(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:        name = 'Gpu' + name
theano/gpuarray/opt.py:    have_cuda = False
theano/gpuarray/opt.py:    have_opencl = False
theano/gpuarray/opt.py:    if inputs and isinstance(inputs[0].type, GpuArrayType):
theano/gpuarray/opt.py:        if kind.startswith(b'opencl'):
theano/gpuarray/opt.py:            have_opencl = True
theano/gpuarray/opt.py:        elif kind.startswith(b'cuda'):
theano/gpuarray/opt.py:            have_cuda = True
theano/gpuarray/opt.py:    convert = {Erfinv: gpu_erfinv,
theano/gpuarray/opt.py:               Erfcinv: gpu_erfcinv}
theano/gpuarray/opt.py:        if have_opencl:
theano/gpuarray/opt.py:                'Function "%s" is not supported with OpenCL. Use "device=cuda" instead.' %
theano/gpuarray/opt.py:        if not have_cuda:
theano/gpuarray/opt.py:    res = GpuElemwise(scal_op, name=name,
theano/gpuarray/opt.py:        # Only transfer the computation on the gpu if the output dtype is
theano/gpuarray/opt.py:        # floating point. Else, give up on the transfer to the gpu.
theano/gpuarray/opt.py:        # Transfer the inputs on the GPU and cast them to the right dtype.
theano/gpuarray/opt.py:                gpu_cast_op = GpuElemwise(Cast(Scalar(out_dtype)))
theano/gpuarray/opt.py:                new_inputs.append(gpu_cast_op(as_gpuarray_variable(inp, context_name)))
theano/gpuarray/opt.py:                new_inputs.append(as_gpuarray_variable(inp, context_name))
theano/gpuarray/opt.py:        # Perform the exponent on the gpu and transfer the output back to the
theano/gpuarray/opt.py:        gpu_output = res(*new_inputs)
theano/gpuarray/opt.py:        return [gpu_output]
theano/gpuarray/opt.py:            return [split_inputs(inputs, max_inputs_to_GpuElemwise(outputs), res)]
theano/gpuarray/opt.py:    This should not happen for other GpuElemwise as their is only the fusion
theano/gpuarray/opt.py:gpu_local_elemwise_fusion = tensor.opt.local_elemwise_fusion_op(
theano/gpuarray/opt.py:    GpuElemwise,
theano/gpuarray/opt.py:    max_inputs_to_GpuElemwise)
theano/gpuarray/opt.py:optdb.register('gpua_elemwise_fusion',
theano/gpuarray/opt.py:               # 48.5 move to gpu
theano/gpuarray/opt.py:               tensor.opt.FusionOptimizer(gpu_local_elemwise_fusion), 49,
theano/gpuarray/opt.py:               'fast_run', 'fusion', 'local_elemwise_fusion', 'gpuarray')
theano/gpuarray/opt.py:inplace_gpu_elemwise_opt = tensor.opt.InplaceElemwiseOptimizer(
theano/gpuarray/opt.py:    GpuElemwise)
theano/gpuarray/opt.py:optdb.register('gpua_inplace_opt', inplace_gpu_elemwise_opt, 75,
theano/gpuarray/opt.py:               'inplace_elemwise_optimizer', 'fast_run', 'inplace', 'gpuarray')
theano/gpuarray/opt.py:def local_gpua_dimshuffle(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return GpuDimShuffle(op.input_broadcastable,
theano/gpuarray/opt.py:def local_gpua_specifyShape(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    if isinstance(inputs[0].type, GpuArrayType):
theano/gpuarray/opt.py:    return local_gpua_specifyShape_graph(op, context_name, inputs, outputs)
theano/gpuarray/opt.py:def local_gpua_specifyShape_graph(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    inp = [as_gpuarray_variable(inputs[0], context_name)]
theano/gpuarray/opt.py:def local_gpua_shape(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    if isinstance(inputs[0].type, GpuArrayType):
theano/gpuarray/opt.py:    return local_gpua_shape_graph(op, context_name, inputs, outputs)
theano/gpuarray/opt.py:def local_gpua_shape_graph(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return [as_gpuarray_variable(inputs[0], context_name).shape]
theano/gpuarray/opt.py:def gpu_print_wrapper(op, cnda):
theano/gpuarray/opt.py:def local_gpua_print_op(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:        gpu_x = as_gpuarray_variable(x, context_name=context_name)
theano/gpuarray/opt.py:        new_op = op.__class__(global_fn=gpu_print_wrapper)
theano/gpuarray/opt.py:        return new_op(gpu_x)
theano/gpuarray/opt.py:def local_gpu_pdbbreakpoint_op(node):
theano/gpuarray/opt.py:        # Go through the monitored variables, only transferring on GPU those
theano/gpuarray/opt.py:        # for which the input comes from the GPU or the output will be
theano/gpuarray/opt.py:        # transferred on the GPU.
theano/gpuarray/opt.py:            input_is_from_gpu = (inp.owner and
theano/gpuarray/opt.py:                                 isinstance(inp.owner.op, HostFromGpu))
theano/gpuarray/opt.py:            output_goes_to_gpu = False
theano/gpuarray/opt.py:                if isinstance(c[0].op, GpuFromHost):
theano/gpuarray/opt.py:                    output_goes_to_gpu = True
theano/gpuarray/opt.py:            if input_is_from_gpu:
theano/gpuarray/opt.py:                # The op should be applied on the GPU version of the input
theano/gpuarray/opt.py:            elif output_goes_to_gpu:
theano/gpuarray/opt.py:                # The input should be transferred to the gpu
theano/gpuarray/opt.py:                new_inputs.append(as_gpuarray_variable(inp, context_name))
theano/gpuarray/opt.py:        # transferred to the gpu
theano/gpuarray/opt.py:            # Propagate the transfer to the gpu through the outputs that require
theano/gpuarray/opt.py:def local_gpua_lazy_ifelse(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    if op.gpu:
theano/gpuarray/opt.py:    # But we can't rely on inputs to respect that, as GraphToGPU don't enforce that.
theano/gpuarray/opt.py:        if ((isinstance(v1.type, tensor.TensorType) and move_to_gpu(v1)) or
theano/gpuarray/opt.py:                isinstance(v1.type, GpuArrayType) or
theano/gpuarray/opt.py:                isinstance(v2.type, GpuArrayType)):
theano/gpuarray/opt.py:            inps.append(as_gpuarray_variable(v1, context_name))
theano/gpuarray/opt.py:            falses.append(as_gpuarray_variable(v2, context_name))
theano/gpuarray/opt.py:    return IfElse(op.n_outs, gpu=True)(c, *inps, return_list=True)
theano/gpuarray/opt.py:def local_gpua_join(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return gpu_join
theano/gpuarray/opt.py:@local_optimizer([GpuJoin])
theano/gpuarray/opt.py:def local_gpua_join_1(node):
theano/gpuarray/opt.py:    if (isinstance(node.op, GpuJoin) and
theano/gpuarray/opt.py:def local_gpua_split(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return GpuSplit(op.len_splits)
theano/gpuarray/opt.py:def local_gpua_subtensor(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    if (x.owner and isinstance(x.owner.op, HostFromGpu)):
theano/gpuarray/opt.py:        gpu_x = x.owner.inputs[0]
theano/gpuarray/opt.py:        if (gpu_x.owner and
theano/gpuarray/opt.py:                isinstance(gpu_x.owner.op, GpuFromHost) and
theano/gpuarray/opt.py:                not gpu_x.owner.inputs[0].owner):
theano/gpuarray/opt.py:                if any([n == 'output' or any([isinstance(v.type, GpuArrayType)
theano/gpuarray/opt.py:                    return [gpu_x.owner.op(outputs[0]).transfer('cpu')]
theano/gpuarray/opt.py:    return GpuSubtensor(op.idx_list)
theano/gpuarray/opt.py:def local_gpua_subtensor_graph(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    # We don't want to move the subtensor to the GPU if the inputs is
theano/gpuarray/opt.py:    if (x.owner and isinstance(x.owner.op, GpuFromHost)):
theano/gpuarray/opt.py:    return GpuSubtensor(op.idx_list)
theano/gpuarray/opt.py:def local_gpua_inc_subtensor(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    op = GpuIncSubtensor(op.idx_list, op.inplace,
theano/gpuarray/opt.py:def local_gpua_advanced_subtensor1(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return GpuAdvancedSubtensor1()
theano/gpuarray/opt.py:def local_gpua_advanced_subtensor(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return GpuAdvancedSubtensor()
theano/gpuarray/opt.py:def local_gpua_advanced_boolean_subtensor(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return GpuAdvancedBooleanSubtensor()
theano/gpuarray/opt.py:def local_gpua_advanced_incsubtensor1(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:        ret = GpuAdvancedIncSubtensor1_dev20(
theano/gpuarray/opt.py:        ret = GpuDimShuffle(ret.type.broadcastable, [0])(ret)
theano/gpuarray/opt.py:        return GpuAdvancedIncSubtensor1(
theano/gpuarray/opt.py:        return GpuAdvancedIncSubtensor1_dev20(
theano/gpuarray/opt.py:def local_gpua_advanced_incsubtensor(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:        return GpuAdvancedIncSubtensor()
theano/gpuarray/opt.py:def local_gpua_advanced_boolean_incsubtensor(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    # GpuAdvancedIncSubtensor only works with a single boolean mask,
theano/gpuarray/opt.py:        return GpuAdvancedBooleanIncSubtensor()
theano/gpuarray/opt.py:@local_optimizer([GpuAdvancedIncSubtensor1, GpuAdvancedIncSubtensor1_dev20])
theano/gpuarray/opt.py:def local_advincsub1_gpua_inplace(node):
theano/gpuarray/opt.py:    if isinstance(node.op, (GpuAdvancedIncSubtensor1,
theano/gpuarray/opt.py:                            GpuAdvancedIncSubtensor1_dev20)):
theano/gpuarray/opt.py:def local_gpu_alloc_diag(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return GpuAllocDiag(offset=op.offset)
theano/gpuarray/opt.py:def local_gpu_extract_diag(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return GpuExtractDiag(offset=op.offset, axis1=op.axis1, axis2=op.axis2, view=op.view)
theano/gpuarray/opt.py:def local_gpua_careduce(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:        if ctx.kind == b'opencl':
theano/gpuarray/opt.py:            op2 = GpuCAReduceCPY
theano/gpuarray/opt.py:        elif ctx.kind == b'cuda':
theano/gpuarray/opt.py:            op2 = GpuCAReduceCuda
theano/gpuarray/opt.py:        # accumulation and float64 is much slower on GPU.
theano/gpuarray/opt.py:        if (op2 is GpuCAReduceCPY or
theano/gpuarray/opt.py:                    as_gpuarray_variable(x, context_name)])):
theano/gpuarray/opt.py:                gpu_reshaped_x = as_gpuarray_variable(reshaped_x, context_name)
theano/gpuarray/opt.py:                gvar = greduce(gpu_reshaped_x)
theano/gpuarray/opt.py:                reshaped_gpu_inputs = [gpu_reshaped_x]
theano/gpuarray/opt.py:                if greduce.supports_c_code(reshaped_gpu_inputs):
theano/gpuarray/opt.py:                    reduce_reshaped_x = greduce(gpu_reshaped_x)
theano/gpuarray/opt.py:                        unreshaped_reduce = GpuReshape(len(out_shp))(
theano/gpuarray/opt.py:def local_gpua_gemv(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:        return gpugemm_no_inplace(inputs[0][:, None],
theano/gpuarray/opt.py:        return gpugemv_inplace
theano/gpuarray/opt.py:        return gpugemv_no_inplace
theano/gpuarray/opt.py:def local_gpua_gemm(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:        return gpugemm_inplace
theano/gpuarray/opt.py:        return gpugemm_no_inplace
theano/gpuarray/opt.py:def local_gpua_gemmbatch(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:        # Since GpuGemmBatch only supports 3D inputs and output,
theano/gpuarray/opt.py:            a = GpuDimShuffle(a.broadcastable, (0, 'x', 1))(a)
theano/gpuarray/opt.py:            b = GpuDimShuffle(b.broadcastable, (0, 1, 'x'))(b)
theano/gpuarray/opt.py:            gpu_cast_op = GpuElemwise(Cast(Scalar(out_dtype)))
theano/gpuarray/opt.py:                a = gpu_cast_op(a)
theano/gpuarray/opt.py:                b = gpu_cast_op(b)
theano/gpuarray/opt.py:        c = GpuAllocEmpty(out_dtype, context_name)(
theano/gpuarray/opt.py:        out = gpugemmbatch_no_inplace(c, np.asarray(1.0, dtype=out_dtype),
theano/gpuarray/opt.py:            out = GpuDimShuffle(out.broadcastable, output_dims)(out)
theano/gpuarray/opt.py:@alpha_merge(GpuGemm, alpha_in=1, beta_in=4)
theano/gpuarray/opt.py:def local_gpua_gemm_alpha_merge(node, *inputs):
theano/gpuarray/opt.py:    return [gpugemm_no_inplace(*inputs)]
theano/gpuarray/opt.py:@output_merge(GpuGemm, alpha_in=1, beta_in=4, out_in=0)
theano/gpuarray/opt.py:def local_gpua_gemm_output_merge(node, *inputs):
theano/gpuarray/opt.py:    return [gpugemm_no_inplace(*inputs)]
theano/gpuarray/opt.py:@alpha_merge(GpuGemmBatch, alpha_in=1, beta_in=4)
theano/gpuarray/opt.py:def local_gpua_gemmbatch_alpha_merge(node, *inputs):
theano/gpuarray/opt.py:    return [gpugemmbatch_no_inplace(*inputs)]
theano/gpuarray/opt.py:@output_merge(GpuGemmBatch, alpha_in=1, beta_in=4, out_in=0)
theano/gpuarray/opt.py:def local_gpua_gemmbatch_output_merge(node, *inputs):
theano/gpuarray/opt.py:    return [gpugemmbatch_no_inplace(*inputs)]
theano/gpuarray/opt.py:def local_gpua_ger(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return GpuGer(inplace=op.destructive)
theano/gpuarray/opt.py:def local_gpua_dot22(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return gpu_dot22
theano/gpuarray/opt.py:def local_gpua_dot22scalar(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:        x = as_gpuarray_variable(x, context_name)
theano/gpuarray/opt.py:        y = as_gpuarray_variable(y, context_name)
theano/gpuarray/opt.py:        z = GpuAllocEmpty(x.dtype, context_name)(x.shape[0], y.shape[1])
theano/gpuarray/opt.py:        return [gpugemm_no_inplace(z, a, x, y, 0)]
theano/gpuarray/opt.py:def local_gpua_eye(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return GpuEye(dtype=op.dtype, context_name=context_name)
theano/gpuarray/opt.py:def local_gpua_tri(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return GpuTri(dtype=op.dtype, context_name=context_name)
theano/gpuarray/opt.py:def local_gpua_crossentropysoftmaxargmax1hotwithbias(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return gpu_crossentropy_softmax_argmax_1hot_with_bias
theano/gpuarray/opt.py:def local_gpua_crossentropysoftmax1hotwithbiasdx(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return gpu_crossentropy_softmax_1hot_with_bias_dx
theano/gpuarray/opt.py:def local_gpua_softmax(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return gpu_softmax
theano/gpuarray/opt.py:def local_gpua_softmaxwithbias(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return gpu_softmax_with_bias
theano/gpuarray/opt.py:def local_gpu_crossentropycategorical1hot(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    # There is no corresponding GPU Op, but we can express it as:
theano/gpuarray/opt.py:    return [gpu_neg(gpu_log(coding[idx0, one_of_n]))]
theano/gpuarray/opt.py:def local_gpu_crossentropycategorical1hotgrad(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    # There is no corresponding GPU Op, but we can express it as:
theano/gpuarray/opt.py:    z = GpuAlloc(context_name, memset_0=True)(
theano/gpuarray/opt.py:        as_gpuarray_variable(np.zeros((), dtype=coding.dtype), context_name),
theano/gpuarray/opt.py:        gpu_neg(gpu_true_div(gy, coding[idx0, one_of_n])))
theano/gpuarray/opt.py:def local_gpua_assert(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    if isinstance(inputs[0].type, GpuArrayType):
theano/gpuarray/opt.py:    return local_gpua_assert_graph(op, context_name, inputs, outputs)
theano/gpuarray/opt.py:def local_gpua_assert_graph(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    return [op(as_gpuarray_variable(inputs[0], context_name),
theano/gpuarray/opt.py:def local_gpua_error_convop(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:ConvOp does not work with the gpuarray backend.
theano/gpuarray/opt.py:Use the new convolution interface to have GPU convolution working:
theano/gpuarray/opt.py:def local_gpua_sparseblockgemv(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:        return gpu_sparse_block_gemv_inplace
theano/gpuarray/opt.py:        return gpu_sparse_block_gemv
theano/gpuarray/opt.py:def local_gpua_sparseblockouter(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:        return gpu_sparse_block_outer_inplace
theano/gpuarray/opt.py:        return gpu_sparse_block_outer
theano/gpuarray/opt.py:@local_optimizer([GpuSparseBlockGemv], inplace=True)
theano/gpuarray/opt.py:    if isinstance(node.op, GpuSparseBlockGemv) and not node.op.inplace:
theano/gpuarray/opt.py:        return [gpu_sparse_block_gemv_inplace(*node.inputs)]
theano/gpuarray/opt.py:@local_optimizer([GpuSparseBlockOuter], inplace=True)
theano/gpuarray/opt.py:    if isinstance(node.op, GpuSparseBlockOuter) and not node.op.inplace:
theano/gpuarray/opt.py:        return [GpuSparseBlockOuter(inplace=True)(*node.inputs)]
theano/gpuarray/opt.py:# Move to Gpu optimization
theano/gpuarray/opt.py:@local_optimizer([GpuFromHost,
theano/gpuarray/opt.py:def local_conv_gpu_conv(node):
theano/gpuarray/opt.py:    gpu_from_host(AbstractConv) -> AbstractConv(gpu_from_host)
theano/gpuarray/opt.py:    AbstractConv(host_from_gpu) -> host_from_gpu(AbstractConv)
theano/gpuarray/opt.py:    if isinstance(node.op, GpuFromHost):
theano/gpuarray/opt.py:            inps[0] = as_gpuarray_variable(inps[0], context_name=ctx)
theano/gpuarray/opt.py:            inps[1] = as_gpuarray_variable(inps[1], context_name=ctx)
theano/gpuarray/opt.py:            # out is on the GPU because both inputs are.
theano/gpuarray/opt.py:        # conv(host_from_gpu) -> host_from_gpu(gpu_conv)
theano/gpuarray/opt.py:        if ((isinstance(inp1.type, GpuArrayType) and
theano/gpuarray/opt.py:             isinstance(inp2.type, GpuArrayType))):
theano/gpuarray/opt.py:            # Both inputs are already directly on the GPU, nothing to do
theano/gpuarray/opt.py:        inp1_on_gpu = (isinstance(inp1.type, GpuArrayType) or
theano/gpuarray/opt.py:                       (inp1.owner and isinstance(inp1.owner.op, HostFromGpu)))
theano/gpuarray/opt.py:        inp2_on_gpu = (isinstance(inp2.type, GpuArrayType) or
theano/gpuarray/opt.py:                       (inp2.owner and isinstance(inp2.owner.op, HostFromGpu)))
theano/gpuarray/opt.py:        if inp1_on_gpu or inp2_on_gpu:
theano/gpuarray/opt.py:            inps[0] = as_gpuarray_variable(inps[0], context_name=ctx)
theano/gpuarray/opt.py:            inps[1] = as_gpuarray_variable(inps[1], context_name=ctx)
theano/gpuarray/opt.py:            # out is on the GPU because both inputs are.
theano/gpuarray/opt.py:register_opt()(local_conv_gpu_conv)
theano/gpuarray/opt.py:    if (not isinstance(img.type, GpuArrayType) or
theano/gpuarray/opt.py:            not isinstance(kern.type, GpuArrayType)):
theano/gpuarray/opt.py:        # call GpuCorrMM_gradInputs
theano/gpuarray/opt.py:        rval = GpuCorrMM_gradInputs('valid',
theano/gpuarray/opt.py:            gpu_contiguous(kern), gpu_contiguous(img))
theano/gpuarray/opt.py:        # By default use GpuCorrMM
theano/gpuarray/opt.py:        rval = GpuCorrMM(border_mode,
theano/gpuarray/opt.py:                         unshared)(gpu_contiguous(img),
theano/gpuarray/opt.py:                                   gpu_contiguous(kern))
theano/gpuarray/opt.py:        # call GpuCorrMM_gradWeights if good
theano/gpuarray/opt.py:        # GpuConv does not always store information on the batchsize and
theano/gpuarray/opt.py:                rval = GpuCorrMM_gradWeights(border_mode,
theano/gpuarray/opt.py:                    gpu_contiguous(img.dimshuffle(1, 0, 2, 3)),
theano/gpuarray/opt.py:                    gpu_contiguous(kern.dimshuffle(1, 0, 2, 3)))
theano/gpuarray/opt.py:                # (we need to wrap the result in as_gpuarray_variable,
theano/gpuarray/opt.py:                # because we are not allowed to replace a GpuArray with
theano/gpuarray/opt.py:                rval = as_gpuarray_variable(
theano/gpuarray/opt.py:    if (not isinstance(img.type, GpuArrayType) or
theano/gpuarray/opt.py:            not isinstance(kern.type, GpuArrayType)):
theano/gpuarray/opt.py:    rval = GpuCorrMM(border_mode,
theano/gpuarray/opt.py:                     unshared)(gpu_contiguous(img),
theano/gpuarray/opt.py:                               gpu_contiguous(kern))
theano/gpuarray/opt.py:    if (not isinstance(img.type, GpuArrayType) or
theano/gpuarray/opt.py:            not isinstance(kern.type, GpuArrayType)):
theano/gpuarray/opt.py:        rval = GpuCorrMM_gradInputs('valid',
theano/gpuarray/opt.py:            gpu_contiguous(kern), gpu_contiguous(img))
theano/gpuarray/opt.py:        rval = GpuCorrMM_gradWeights(border_mode,
theano/gpuarray/opt.py:            gpu_contiguous(img.dimshuffle(1, 0, 2, 3)),
theano/gpuarray/opt.py:            gpu_contiguous(kern.dimshuffle(1, 0, 2, 3)))
theano/gpuarray/opt.py:        rval = as_gpuarray_variable(rval.dimshuffle(1, 0, 2, 3),
theano/gpuarray/opt.py:    if (not isinstance(img.type, GpuArrayType) or
theano/gpuarray/opt.py:            not isinstance(kern.type, GpuArrayType)):
theano/gpuarray/opt.py:        # call GpuCorr3dMM_gradInputs
theano/gpuarray/opt.py:        rval = GpuCorr3dMM_gradInputs('valid',
theano/gpuarray/opt.py:            gpu_contiguous(kern), gpu_contiguous(img))
theano/gpuarray/opt.py:        # By default use GpuCorr3dMM
theano/gpuarray/opt.py:        rval = GpuCorr3dMM(border_mode,
theano/gpuarray/opt.py:                           num_groups)(gpu_contiguous(img),
theano/gpuarray/opt.py:                                       gpu_contiguous(kern))
theano/gpuarray/opt.py:        # call GpuCorr3dMM_gradWeights if good
theano/gpuarray/opt.py:        # GpuConv does not always store information on the batchsize and
theano/gpuarray/opt.py:                rval = GpuCorr3dMM_gradWeights(border_mode,
theano/gpuarray/opt.py:                    gpu_contiguous(img.dimshuffle(1, 0, 2, 3, 4)),
theano/gpuarray/opt.py:                    gpu_contiguous(kern.dimshuffle(1, 0, 2, 3, 4)))
theano/gpuarray/opt.py:                # (we need to wrap the result in as_gpuarray_variable,
theano/gpuarray/opt.py:                # because we are not allowed to replace a GpuArray with
theano/gpuarray/opt.py:                rval = as_gpuarray_variable(
theano/gpuarray/opt.py:    if (not isinstance(img.type, GpuArrayType) or
theano/gpuarray/opt.py:            not isinstance(kern.type, GpuArrayType)):
theano/gpuarray/opt.py:    # By default use GpuCorr3dMM
theano/gpuarray/opt.py:    rval = GpuCorr3dMM(border_mode,
theano/gpuarray/opt.py:                       node.op.num_groups)(gpu_contiguous(img),
theano/gpuarray/opt.py:                                           gpu_contiguous(kern))
theano/gpuarray/opt.py:    if (not isinstance(img.type, GpuArrayType) or
theano/gpuarray/opt.py:            not isinstance(kern.type, GpuArrayType)):
theano/gpuarray/opt.py:        rval = GpuCorr3dMM_gradInputs('valid',
theano/gpuarray/opt.py:            gpu_contiguous(kern), gpu_contiguous(img))
theano/gpuarray/opt.py:        rval = GpuCorr3dMM_gradWeights(border_mode,
theano/gpuarray/opt.py:            gpu_contiguous(img.dimshuffle(1, 0, 2, 3, 4)),
theano/gpuarray/opt.py:            gpu_contiguous(kern.dimshuffle(1, 0, 2, 3, 4)))
theano/gpuarray/opt.py:        rval = as_gpuarray_variable(rval.dimshuffle(1, 0, 2, 3, 4),
theano/gpuarray/opt.py:    if (not isinstance(img.type, GpuArrayType) or
theano/gpuarray/opt.py:            not isinstance(kern.type, GpuArrayType)):
theano/gpuarray/opt.py:        rval = conv3d2d.conv3d(gpu_contiguous(img.dimshuffle(*reorder_array)),
theano/gpuarray/opt.py:                               gpu_contiguous(kern.dimshuffle(*reorder_array)),
theano/gpuarray/opt.py:        rval = as_gpuarray_variable(rval.dimshuffle(*reorder_array),
theano/gpuarray/opt.py:    if not isinstance(img.type, GpuArrayType) or \
theano/gpuarray/opt.py:            not isinstance(topgrad.type, GpuArrayType):
theano/gpuarray/opt.py:    rval = GpuCorrMM_gradWeights(border_mode=node.op.border_mode,
theano/gpuarray/opt.py:        gpu_contiguous(img), gpu_contiguous(topgrad), shape)
theano/gpuarray/opt.py:    rval = as_gpuarray_variable(rval, context_name=ctx)
theano/gpuarray/opt.py:    if not isinstance(img.type, GpuArrayType) or \
theano/gpuarray/opt.py:            not isinstance(topgrad.type, GpuArrayType):
theano/gpuarray/opt.py:        rval = GpuCorrMM(border_mode,
theano/gpuarray/opt.py:            gpu_contiguous(img.dimshuffle(1, 0, 2, 3)),
theano/gpuarray/opt.py:            gpu_contiguous(topgrad.dimshuffle(1, 0, 2, 3)))
theano/gpuarray/opt.py:        rval = as_gpuarray_variable(rval, context_name=ctx)
theano/gpuarray/opt.py:    if not isinstance(img.type, GpuArrayType) or \
theano/gpuarray/opt.py:            not isinstance(topgrad.type, GpuArrayType):
theano/gpuarray/opt.py:        rval = GpuCorr3dMM(border_mode,
theano/gpuarray/opt.py:            gpu_contiguous(img.dimshuffle(1, 0, 2, 3, 4)),
theano/gpuarray/opt.py:            gpu_contiguous(topgrad.dimshuffle(1, 0, 2, 3, 4)))
theano/gpuarray/opt.py:        rval = as_gpuarray_variable(rval, context_name=ctx)
theano/gpuarray/opt.py:    if not isinstance(img.type, GpuArrayType) or \
theano/gpuarray/opt.py:            not isinstance(topgrad.type, GpuArrayType):
theano/gpuarray/opt.py:    rval = GpuCorr3dMM_gradWeights(border_mode=node.op.border_mode,
theano/gpuarray/opt.py:        gpu_contiguous(img), gpu_contiguous(topgrad), shape)
theano/gpuarray/opt.py:    rval = as_gpuarray_variable(rval, context_name=ctx)
theano/gpuarray/opt.py:    if not isinstance(kern.type, GpuArrayType) or \
theano/gpuarray/opt.py:            not isinstance(topgrad.type, GpuArrayType):
theano/gpuarray/opt.py:    rval = GpuCorrMM_gradInputs(border_mode=node.op.border_mode,
theano/gpuarray/opt.py:        gpu_contiguous(kern), gpu_contiguous(topgrad), shape)
theano/gpuarray/opt.py:    if not isinstance(kern.type, GpuArrayType) or \
theano/gpuarray/opt.py:            not isinstance(topgrad.type, GpuArrayType):
theano/gpuarray/opt.py:        rval = GpuCorrMM(border_mode='full',
theano/gpuarray/opt.py:            gpu_contiguous(topgrad),
theano/gpuarray/opt.py:            gpu_contiguous(kern.dimshuffle(1, 0, 2, 3)))
theano/gpuarray/opt.py:    if not isinstance(kern.type, GpuArrayType) or \
theano/gpuarray/opt.py:            not isinstance(topgrad.type, GpuArrayType):
theano/gpuarray/opt.py:    rval = GpuCorr3dMM_gradInputs(border_mode=node.op.border_mode,
theano/gpuarray/opt.py:        gpu_contiguous(kern), gpu_contiguous(topgrad), shape)
theano/gpuarray/opt.py:    if not isinstance(kern.type, GpuArrayType) or \
theano/gpuarray/opt.py:            not isinstance(topgrad.type, GpuArrayType):
theano/gpuarray/opt.py:        rval = GpuCorr3dMM(border_mode='full',
theano/gpuarray/opt.py:            gpu_contiguous(topgrad),
theano/gpuarray/opt.py:            gpu_contiguous(kern.dimshuffle(1, 0, 2, 3, 4)))
theano/gpuarray/opt.py:def local_gpua_abstractconv(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    if isinstance(outputs[0].type, GpuArrayType):
theano/gpuarray/opt.py:        # Don't handle this node here, it's already on the GPU.
theano/gpuarray/opt.py:    return local_gpua_lift_abstractconv_graph(op, context_name, inputs, outputs)
theano/gpuarray/opt.py:def local_gpua_lift_abstractconv_graph(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    inps[0] = as_gpuarray_variable(inputs[0],
theano/gpuarray/opt.py:    inps[1] = as_gpuarray_variable(inputs[1],
theano/gpuarray/opt.py:def local_gpu_pool(op, ctx_name, inputs, outputs):
theano/gpuarray/opt.py:    inp = gpu_contiguous(as_gpuarray_variable(inp, ctx_name))
theano/gpuarray/opt.py:    op = GpuPool(op.ignore_border, op.mode, op.ndim)
theano/gpuarray/opt.py:pool_db2 = LocalGroupDB(local_opt=theano.gof.opt.GraphToGPULocalOptGroup)
theano/gpuarray/opt.py:lifter = op_lifter([pool.Pool])(local_gpu_pool)
theano/gpuarray/opt.py:pool_db.register("local_gpu_pool", lifter,
theano/gpuarray/opt.py:                 'gpuarray', 'fast_compile', 'fast_run',
theano/gpuarray/opt.py:pool_db2.register("local_gpu_pool",
theano/gpuarray/opt.py:                  local_optimizer([pool.Pool])(local_gpu_pool),
theano/gpuarray/opt.py:                  'gpuarray', 'fast_compile', 'fast_run',
theano/gpuarray/opt.py:def local_gpu_max_pool_grad(op, ctx_name, inputs, outputs):
theano/gpuarray/opt.py:    inp = gpu_contiguous(as_gpuarray_variable(inp, ctx_name))
theano/gpuarray/opt.py:    out = gpu_contiguous(as_gpuarray_variable(out, ctx_name))
theano/gpuarray/opt.py:    out_grad = gpu_contiguous(as_gpuarray_variable(out_grad, ctx_name))
theano/gpuarray/opt.py:    op = GpuMaxPoolGrad(op.ignore_border, op.mode, op.ndim)
theano/gpuarray/opt.py:lifter = op_lifter([pool.MaxPoolGrad])(local_gpu_max_pool_grad)
theano/gpuarray/opt.py:pool_db.register("local_gpu_max_pool_grad", lifter,
theano/gpuarray/opt.py:                 'gpuarray', 'fast_compile', 'fast_run',
theano/gpuarray/opt.py:pool_db2.register("local_gpu_max_pool_grad",
theano/gpuarray/opt.py:                  local_optimizer([pool.MaxPoolGrad])(local_gpu_max_pool_grad),
theano/gpuarray/opt.py:                  'gpuarray', 'fast_compile', 'fast_run',
theano/gpuarray/opt.py:def local_gpu_average_pool_grad(op, ctx_name, inputs, outputs):
theano/gpuarray/opt.py:    inp = gpu_contiguous(as_gpuarray_variable(inp, ctx_name))
theano/gpuarray/opt.py:    out_grad = gpu_contiguous(as_gpuarray_variable(out_grad, ctx_name))
theano/gpuarray/opt.py:    op = GpuAveragePoolGrad(op.ignore_border, op.mode, op.ndim)
theano/gpuarray/opt.py:lifter = op_lifter([pool.AveragePoolGrad])(local_gpu_average_pool_grad)
theano/gpuarray/opt.py:pool_db.register("local_gpu_average_pool_grad", lifter,
theano/gpuarray/opt.py:                 'gpuarray', 'fast_compile', 'fast_run',
theano/gpuarray/opt.py:pool_db2.register("local_gpu_average_pool_grad",
theano/gpuarray/opt.py:                  local_optimizer([pool.AveragePoolGrad])(local_gpu_average_pool_grad),
theano/gpuarray/opt.py:                  'gpuarray', 'fast_compile', 'fast_run',
theano/gpuarray/opt.py:def local_gpu_downsample_factor_max_grad_grad(op, ctx_name, inputs, outputs):
theano/gpuarray/opt.py:    inp = gpu_contiguous(as_gpuarray_variable(inp, ctx_name))
theano/gpuarray/opt.py:    out = gpu_contiguous(as_gpuarray_variable(out, ctx_name))
theano/gpuarray/opt.py:    out_grad = gpu_contiguous(as_gpuarray_variable(out_grad, ctx_name))
theano/gpuarray/opt.py:    op = GpuDownsampleFactorMaxGradGrad(op.ignore_border, op.mode, op.ndim)
theano/gpuarray/opt.py:def local_gpu_max_pool_rop(op, ctx_name, inputs, outputs):
theano/gpuarray/opt.py:    inp = gpu_contiguous(as_gpuarray_variable(inp, ctx_name))
theano/gpuarray/opt.py:    eval_inp = gpu_contiguous(as_gpuarray_variable(eval_inp, ctx_name))
theano/gpuarray/opt.py:    op = GpuMaxPoolRop(op.ignore_border, op.mode, op.ndim)
theano/gpuarray/opt.py:@local_optimizer([GpuCAReduceCuda])
theano/gpuarray/opt.py:def local_gpu_elemwise_careduce(node):
theano/gpuarray/opt.py:    Merge some GpuCAReduceCuda and GPUElemwise.
theano/gpuarray/opt.py:    if (isinstance(node.op, GpuCAReduceCuda) and
theano/gpuarray/opt.py:            isinstance(node.inputs[0].owner.op, GpuElemwise) and
theano/gpuarray/opt.py:            out = GpuCAReduceCuda(**props)(inp)
theano/gpuarray/opt.py:    if (all([var.owner and isinstance(var.owner.op, HostFromGpu)
theano/gpuarray/opt.py:        any([[c for c in var.clients if isinstance(c[0].op, GpuFromHost)]
theano/gpuarray/opt.py:optdb.register('gpua_assert_no_cpu_op', assert_no_cpu_op, 49.2,
theano/gpuarray/opt.py:def tensor_to_gpu(x, context_name):
theano/gpuarray/opt.py:        y = GpuArrayType(broadcastable=x.type.broadcastable,
theano/gpuarray/opt.py:            y.name = x.name + '[Gpua]'
theano/gpuarray/opt.py:def gpu_safe_new(x, tag=''):
theano/gpuarray/opt.py:def gpu_reconstruct_graph(inputs, outputs, tag=None):
theano/gpuarray/opt.py:    nw_inputs = [gpu_safe_new(x, tag) for x in inputs]
theano/gpuarray/opt.py:def local_gpua_scan_to_gpua(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    if info.get('gpua', False):
theano/gpuarray/opt.py:    info['gpua'] = True
theano/gpuarray/opt.py:    nw_ins += [safe_to_gpu(x, context_name) for x in inputs[1:e]]
theano/gpuarray/opt.py:    nw_ins += [safe_to_gpu(x, context_name) for x in inputs[e:]]
theano/gpuarray/opt.py:    scan_ins = [tensor_to_gpu(x, context_name) for x in op.inputs]
theano/gpuarray/opt.py:    # moved to the gpu
theano/gpuarray/opt.py:        scan_outs = [safe_to_gpu(x, context_name) for x in op.outputs[:-1]]
theano/gpuarray/opt.py:        scan_outs = [safe_to_gpu(x, context_name) for x in op.outputs]
theano/gpuarray/opt.py:    # __init__ does not know about the gpu and can not
theano/gpuarray/opt.py:    # handle graphs with inputs being on the gpu
theano/gpuarray/opt.py:    tmp_in, tmp_out = gpu_reconstruct_graph(scan_ins, scan_outs)
theano/gpuarray/opt.py:    info['gpu_hash'] = hash(_cmodule_key)
theano/gpuarray/opt.py:        return GpuArrayType(dtype=dtype, broadcastable=broadcastable,
theano/gpuarray/opt.py:        return GpuArrayType(dtype=dtype, broadcastable=broadcastable,
theano/gpuarray/opt.py:# Add optimization : maxandargmax (CPU -> GPU)
theano/gpuarray/opt.py:def local_gpu_maxandargmax(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    op = GpuMaxAndArgmax(op.get_params(None))
theano/gpuarray/opt.py:        # For now it is better to copy/cast on the GPU then transfer to the CPU
theano/gpuarray/opt.py:def local_gpua_images2neibs(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:        return GpuImages2Neibs(op.mode)
theano/gpuarray/opt.py:def local_gpu_solve(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:        op = GpuCublasTriangularSolve(lower)
theano/gpuarray/opt.py:        op = GpuCusolverSolve(A_structure=op.A_structure)
theano/gpuarray/opt.py:@local_optimizer([GpuCusolverSolve], inplace=True)
theano/gpuarray/opt.py:def local_inplace_gpu_solve(node):
theano/gpuarray/opt.py:    if isinstance(node.op, GpuCusolverSolve) and not node.op.inplace:
theano/gpuarray/opt.py:            return [GpuCusolverSolve(A_structure=node.op.A_structure,
theano/gpuarray/opt.py:def local_gpu_cholesky(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    op = GpuCholesky(lower=op.lower, inplace=op.destructive)
theano/gpuarray/opt.py:matrix_ops_db2 = LocalGroupDB(local_opt=theano.gof.opt.GraphToGPULocalOptGroup)
theano/gpuarray/opt.py:lifter = op_lifter([slinalg.Cholesky])(local_gpu_cholesky)
theano/gpuarray/opt.py:matrix_ops_db.register("local_gpu_cholesky", lifter,
theano/gpuarray/opt.py:                       'gpuarray', 'fast_compile', 'fast_run', 'cusolver',
theano/gpuarray/opt.py:matrix_ops_db2.register("local_gpu_cholesky",
theano/gpuarray/opt.py:                        local_optimizer([slinalg.Cholesky])(local_gpu_cholesky),
theano/gpuarray/opt.py:                        'gpuarray', 'fast_compile', 'fast_run', 'cusolver',
theano/gpuarray/opt.py:@local_optimizer([GpuCholesky], inplace=True)
theano/gpuarray/opt.py:def local_inplace_gpu_cholesky(node):
theano/gpuarray/opt.py:    if isinstance(node.op, GpuCholesky) and not node.op.inplace:
theano/gpuarray/opt.py:def local_gpu_magma_cholesky(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    op = GpuMagmaCholesky(lower=op.lower, inplace=op.destructive)
theano/gpuarray/opt.py:lifter = op_lifter([slinalg.Cholesky])(local_gpu_magma_cholesky)
theano/gpuarray/opt.py:matrix_ops_db.register("local_gpu_magma_cholesky", lifter,
theano/gpuarray/opt.py:                       'gpuarray', 'fast_compile', 'fast_run', 'magma',
theano/gpuarray/opt.py:matrix_ops_db2.register("local_gpu_magma_cholesky",
theano/gpuarray/opt.py:                        local_optimizer([slinalg.Cholesky])(local_gpu_magma_cholesky),
theano/gpuarray/opt.py:                        'gpuarray', 'fast_compile', 'fast_run', 'magma',
theano/gpuarray/opt.py:@local_optimizer([GpuMagmaCholesky], inplace=True)
theano/gpuarray/opt.py:def local_inplace_gpu_magma_cholesky(node):
theano/gpuarray/opt.py:    if isinstance(node.op, GpuMagmaCholesky) and not node.op.inplace:
theano/gpuarray/opt.py:def local_gpu_magma_qr(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    out = gpu_qr(x, complete=True)
theano/gpuarray/opt.py:def local_gpu_magma_qr_incomplete(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    out = gpu_qr(x, complete=False)
theano/gpuarray/opt.py:def local_gpu_magma_matrix_inverse(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    op = GpuMagmaMatrixInverse()
theano/gpuarray/opt.py:@local_optimizer([GpuMagmaMatrixInverse])
theano/gpuarray/opt.py:def local_inplace_gpu_magma_matrix_inverse(node):
theano/gpuarray/opt.py:    if isinstance(node.op, GpuMagmaMatrixInverse) and not node.op.inplace:
theano/gpuarray/opt.py:def local_gpu_magma_eigh(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    op = GpuMagmaEigh(UPLO=op.UPLO, compute_v=True)
theano/gpuarray/opt.py:def local_gpu_magma_svd(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    out = gpu_svd(x, compute_uv=op.compute_uv, full_matrices=op.full_matrices)
theano/gpuarray/opt.py:def local_gpu_ctc(op, context_name, inputs, outputs):
theano/gpuarray/opt.py:    op = GpuConnectionistTemporalClassification(compute_grad=op.compute_grad)
theano/gpuarray/opt.py:# It will be added to fast_run if the GPU is enabled.
theano/gpuarray/opt.py:optdb.register('gpua_scanOp_make_inplace',
theano/gpuarray/opt.py:                                             gpua_flag=True),
theano/gpuarray/opt.py:               'gpuarray',
theano/gpuarray/opt.py:# Register GPU convolution implementation
theano/gpuarray/opt.py:abstractconv_groupopt.__name__ = "gpuarray_abstractconv_opts"
theano/gpuarray/opt.py:                               'gpuarray', 'fast_compile', 'fast_run', 'cudnn')
theano/gpuarray/opt.py:                               'gpuarray', 'fast_compile', 'fast_run', 'cudnn')
theano/gpuarray/opt.py:                               'gpuarray', 'fast_compile', 'fast_run', 'cudnn')
theano/gpuarray/opt.py:                               'gpuarray', 'fast_compile', 'fast_run')
theano/gpuarray/opt.py:                               'gpuarray', 'fast_compile', 'fast_run')
theano/gpuarray/opt.py:                               'gpuarray', 'fast_compile', 'fast_run')
theano/gpuarray/opt.py:                               'gpuarray', 'fast_compile', 'fast_run')
theano/gpuarray/opt.py:                               'gpuarray', 'fast_compile', 'fast_run')
theano/gpuarray/opt.py:                               'gpuarray', 'fast_compile', 'fast_run')
theano/gpuarray/opt.py:abstract_batch_norm_groupopt.__name__ = "gpuarray_batchnorm_opts"
theano/gpuarray/opt.py:    local_opt=theano.gof.opt.GraphToGPULocalOptGroup)
theano/gpuarray/opt.py:                                    'gpuarray', 'fast_compile', 'fast_run',
theano/gpuarray/opt.py:                                     'gpuarray', 'fast_compile', 'fast_run',
theano/gpuarray/opt.py:    # GraphToGPU.  So for now, only add it to the slower EQ phase.  If
theano/gpuarray/opt.py:    # there is no cuDNN, we still want to move it to the GPU now with
theano/gpuarray/opt.py:    # a Theano graph so to have this graph on the GPU.
theano/gpuarray/opt.py:                                    'gpuarray', 'fast_compile', 'fast_run',
theano/gpuarray/extra_ops.py:    from pygpu import gpuarray
theano/gpuarray/extra_ops.py:from .basic_ops import (as_gpuarray_variable, GpuKernelBase, Kernel, GpuReshape, infer_context_name, gpuarray_helper_inc_dir)
theano/gpuarray/extra_ops.py:from .type import gpu_context_type
theano/gpuarray/extra_ops.py:class GpuCumOp(GpuKernelBase, Op):
theano/gpuarray/extra_ops.py:                             context=gpu_context_type)
theano/gpuarray/extra_ops.py:        return ['<numpy_compat.h>', '<gpuarray/types.h>', '<gpuarray_helper.h>']
theano/gpuarray/extra_ops.py:        return [gpuarray_helper_inc_dir()]
theano/gpuarray/extra_ops.py:        assert x.type.dtype == 'float32', "Only float32 supported for GpuCumOp"
theano/gpuarray/extra_ops.py:        x = as_gpuarray_variable(x, context_name)
theano/gpuarray/extra_ops.py:        if x.ndim > GpuCumOp.SUPPORTED_NDIMS:
theano/gpuarray/extra_ops.py:    def gpu_kernels(self, node, nodename):
theano/gpuarray/extra_ops.py:        params = [gpuarray.GpuArray, gpuarray.SIZE,
theano/gpuarray/extra_ops.py:                  gpuarray.GpuArray, gpuarray.SIZE,
theano/gpuarray/extra_ops.py:                  gpuarray.SSIZE, gpuarray.SSIZE,
theano/gpuarray/extra_ops.py:                  gpuarray.SSIZE, gpuarray.SSIZE,
theano/gpuarray/extra_ops.py:                  gpuarray.SSIZE, gpuarray.SSIZE,
theano/gpuarray/extra_ops.py:        params = [gpuarray.GpuArray, gpuarray.SIZE,
theano/gpuarray/extra_ops.py:                  gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SIZE,
theano/gpuarray/extra_ops.py:                  gpuarray.SSIZE, gpuarray.SSIZE, gpuarray.SSIZE,
theano/gpuarray/extra_ops.py:                  gpuarray.SSIZE, gpuarray.SSIZE, gpuarray.SSIZE,
theano/gpuarray/extra_ops.py:                  'int32', 'int32', gpuarray.GpuArray, gpuarray.SIZE]
theano/gpuarray/extra_ops.py:            // Similar to http://www.umiacs.umd.edu/~ramani/cmsc828e_gpusci/ScanTalk.pdf
theano/gpuarray/extra_ops.py:        params = [gpuarray.GpuArray, gpuarray.SIZE,
theano/gpuarray/extra_ops.py:                  gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SIZE,
theano/gpuarray/extra_ops.py:                  gpuarray.SSIZE, gpuarray.SSIZE, gpuarray.SSIZE,
theano/gpuarray/extra_ops.py:        if node.inputs[0].type.context.kind != b'cuda':
theano/gpuarray/extra_ops.py:            raise NotImplementedError("cuda only")
theano/gpuarray/extra_ops.py:            const size_t* shape = PyGpuArray_DIMS(%(x)s);
theano/gpuarray/extra_ops.py:            bool needAllocation = !%(z)s || PyGpuArray_NDIM(%(x)s) != PyGpuArray_NDIM(%(z)s);
theano/gpuarray/extra_ops.py:                axis += PyGpuArray_NDIM(%(x)s);
theano/gpuarray/extra_ops.py:            if (theano_prep_output(&%(z)s, PyGpuArray_NDIM(%(x)s), PyGpuArray_DIMS(%(x)s),
theano/gpuarray/extra_ops.py:                err = gpucontext_property(%(params)s->context->ctx, GA_CTX_PROP_MAXLSIZE0, &max_threads_dim0);
theano/gpuarray/extra_ops.py:                err = gpucontext_property(%(params)s->context->ctx, GA_CTX_PROP_MAXGSIZE1, &max_grid_size1);
theano/gpuarray/extra_ops.py:                err = gpucontext_property(%(params)s->context->ctx, GA_CTX_PROP_MAXGSIZE2, &max_grid_size2);
theano/gpuarray/extra_ops.py:        int cumOp_%(nodename)s(PyGpuArrayObject* input, PyGpuArrayObject* output, int axis, size_t maxThreads, size_t maxGridY, size_t maxGridZ) {
theano/gpuarray/extra_ops.py:            switch (PyGpuArray_NDIM(input))
theano/gpuarray/extra_ops.py:                shape[0] = PyGpuArray_DIMS(input)[0];
theano/gpuarray/extra_ops.py:                inputStrides_x = PyGpuArray_STRIDES(input)[0] / sizeof(float);
theano/gpuarray/extra_ops.py:                outputStrides_x = PyGpuArray_STRIDES(output)[0] / sizeof(float);
theano/gpuarray/extra_ops.py:                shape[0] = PyGpuArray_DIMS(input)[0];
theano/gpuarray/extra_ops.py:                shape[1] = PyGpuArray_DIMS(input)[1];
theano/gpuarray/extra_ops.py:                inputStrides_x = PyGpuArray_STRIDES(input)[0] / sizeof(float);
theano/gpuarray/extra_ops.py:                inputStrides_y = PyGpuArray_STRIDES(input)[1] / sizeof(float);
theano/gpuarray/extra_ops.py:                outputStrides_x = PyGpuArray_STRIDES(output)[0] / sizeof(float);
theano/gpuarray/extra_ops.py:                outputStrides_y = PyGpuArray_STRIDES(output)[1] / sizeof(float);
theano/gpuarray/extra_ops.py:                shape[0] = PyGpuArray_DIMS(input)[0];
theano/gpuarray/extra_ops.py:                shape[1] = PyGpuArray_DIMS(input)[1];
theano/gpuarray/extra_ops.py:                shape[2] = PyGpuArray_DIMS(input)[2];
theano/gpuarray/extra_ops.py:                inputStrides_x = PyGpuArray_STRIDES(input)[0] / sizeof(float);
theano/gpuarray/extra_ops.py:                inputStrides_y = PyGpuArray_STRIDES(input)[1] / sizeof(float);
theano/gpuarray/extra_ops.py:                inputStrides_z = PyGpuArray_STRIDES(input)[2] / sizeof(float);
theano/gpuarray/extra_ops.py:                outputStrides_x = PyGpuArray_STRIDES(output)[0] / sizeof(float);
theano/gpuarray/extra_ops.py:                outputStrides_y = PyGpuArray_STRIDES(output)[1] / sizeof(float);
theano/gpuarray/extra_ops.py:                outputStrides_z = PyGpuArray_STRIDES(output)[2] / sizeof(float);
theano/gpuarray/extra_ops.py:                int err = pygpu_move(output, input);
theano/gpuarray/extra_ops.py:            PyGpuArrayObject* deviceBlockSum = pygpu_empty(2, shapeBlockSum, output->ga.typecode,
theano/gpuarray/extra_ops.py:        return super(GpuCumOp, self).c_support_code_struct(node, nodename) + code
theano/gpuarray/extra_ops.py:# GpuCumsumOp exists only to serve backward compatibility.
theano/gpuarray/extra_ops.py:class GpuCumsumOp(GpuKernelBase, Op):
theano/gpuarray/extra_ops.py:        obj = object.__new__(GpuCumOp, *args, **kwargs)
theano/gpuarray/extra_ops.py:def local_gpua_cumop(op, ctx_name, inputs, outputs):
theano/gpuarray/extra_ops.py:    if axis is not None and x.ndim > GpuCumOp.SUPPORTED_NDIMS:
theano/gpuarray/extra_ops.py:    x = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/extra_ops.py:        x = GpuReshape(1)(x, (-1,))
theano/gpuarray/extra_ops.py:    # ``gpu_cumop`` assume array has been flattened if needed.
theano/gpuarray/extra_ops.py:    return GpuCumOp(axis, op.mode)(x)
theano/gpuarray/opt_util.py:from .basic_ops import GpuFromHost, HostFromGpu, GpuAllocEmpty, GpuReshape
theano/gpuarray/opt_util.py:from .elemwise import GpuDimShuffle, GpuElemwise
theano/gpuarray/opt_util.py:        if (isinstance(n.op, (GpuDimShuffle, DimShuffle)) and
theano/gpuarray/opt_util.py:        elif isinstance(n.op, (GpuFromHost, HostFromGpu)):
theano/gpuarray/opt_util.py:        elif (isinstance(v.owner.op, GpuFromHost) and
theano/gpuarray/opt_util.py:              isinstance(v.owner.inputs[0].owner.op, HostFromGpu)):
theano/gpuarray/opt_util.py:        @local_optimizer([GpuElemwise])
theano/gpuarray/opt_util.py:            if (isinstance(node.op, GpuElemwise) and
theano/gpuarray/opt_util.py:        @local_optimizer([GpuElemwise])
theano/gpuarray/opt_util.py:            if (isinstance(node.op, GpuElemwise) and
theano/gpuarray/opt_util.py:                    isinstance(alloc.owner.op, GpuAllocEmpty) and
theano/gpuarray/opt_util.py:                alloc_op = GpuAllocEmpty(alloc.owner.op.dtype, alloc.owner.op.context_name)
theano/gpuarray/opt_util.py:    GPU pooling ops.
theano/gpuarray/opt_util.py:    input_ND = GpuReshape(leftdims + rightdims)(input, new_shape)
theano/gpuarray/opt_util.py:    return GpuReshape(input.ndim)(output, outshp)
theano/gpuarray/reduction.py:from .basic_ops import (infer_context_name, as_gpuarray_variable, gpuarray_helper_inc_dir)
theano/gpuarray/reduction.py:from .type import GpuArrayType
theano/gpuarray/reduction.py:    import pygpu
theano/gpuarray/reduction.py:class GpuMaxAndArgmax(Op):
theano/gpuarray/reduction.py:    GPU version of MaxAndArgmax
theano/gpuarray/reduction.py:        inputs = [as_gpuarray_variable(X, context_name)]
theano/gpuarray/reduction.py:        outputs = [GpuArrayType(X.type.dtype, broadcastable, context_name=context_name)(),
theano/gpuarray/reduction.py:                   GpuArrayType(self.argmax_dtype, broadcastable, context_name=context_name)()]
theano/gpuarray/reduction.py:        return ['<numpy_compat.h>', '<gpuarray_helper.h>']
theano/gpuarray/reduction.py:        return [pygpu.get_include(), gpuarray_helper_inc_dir()]
theano/gpuarray/reduction.py:        max_typecode = pygpu.gpuarray.dtype_to_typecode(node.inputs[0].dtype)
theano/gpuarray/reduction.py:        argmax_typecode = pygpu.gpuarray.dtype_to_typecode(self.argmax_dtype)
theano/gpuarray/reduction.py:        size_t  %(name)s_input_ndim = PyGpuArray_NDIM(%(X)s);
theano/gpuarray/reduction.py:                %(name)s_output_dims[i] = PyGpuArray_DIM(%(X)s, i);
theano/gpuarray/reduction.py:                %(name)s_output_dims[i-1] = PyGpuArray_DIM(%(X)s, i);
theano/gpuarray/reduction.py:                    %(name)s_output_dims[++current_output_pos] = PyGpuArray_DIM(%(X)s, current_input_pos);
theano/gpuarray/reduction.py:                %(name)s_output_dims[++current_output_pos] = PyGpuArray_DIM(%(X)s, current_input_pos);
theano/gpuarray/reduction.py:            PyErr_SetString(PyExc_RuntimeError, "GpuMaxAndArgmax: unable to prepare max output.");
theano/gpuarray/reduction.py:            PyErr_SetString(PyExc_RuntimeError, "GpuMaxAndArgmax: unable to prepare argmax output.");
theano/gpuarray/reduction.py:            /* GpuArray_maxandargmax can't handle a 0-d array
theano/gpuarray/reduction.py:            if (GA_NO_ERROR != GpuArray_setarray(&%(max)s->ga, &%(X)s->ga)) {
theano/gpuarray/reduction.py:                PyErr_SetString(PyExc_RuntimeError, "GpuMaxAndArgmax: unable to copy input to max when input is a scalar.");
theano/gpuarray/reduction.py:            if (GA_NO_ERROR != GpuArray_memset(&%(argmax)s->ga, 0)) {
theano/gpuarray/reduction.py:                PyErr_SetString(PyExc_RuntimeError, "GpuMaxAndArgmax: unable to set argmax to 0 when input is a scalar.");
theano/gpuarray/reduction.py:            GpuArray_maxandargmax(&%(max)s->ga, &%(argmax)s->ga, &%(X)s->ga, %(name)s_redux_len, %(name)s_axes_to_reduce)
theano/gpuarray/reduction.py:                "GpuMaxAndArgmax: unable to compute gpuarray maxandargmax: error %%d: %%s (%%s).",
theano/gpuarray/reduction.py:                err, gpuarray_error_str(err), GpuArray_error(&%(X)s->ga, err));
theano/gpuarray/kernel_codegen.py:Helper routines for generating gpu kernels for nvcc.
theano/gpuarray/kernel_codegen.py:    from pygpu import gpuarray
theano/gpuarray/kernel_codegen.py:    `buf` should be in gpu shared memory, we access it many times.
theano/gpuarray/kernel_codegen.py:    `buf` and `buf2` should be in gpu shared memory, we access it many
theano/gpuarray/kernel_codegen.py:    ctype = gpuarray.dtype_to_ctype(dtype)
theano/gpuarray/kernel_codegen.py:    `buf` should be in gpu shared memory, we access it many times.
theano/gpuarray/kernel_codegen.py:    ctype = gpuarray.dtype_to_ctype(dtype)
theano/gpuarray/kernel_codegen.py:        A ptr to the gpu memory where the row is stored.
theano/gpuarray/kernel_codegen.py:        A ptr to the gpu memory to store the result.
theano/gpuarray/kernel_codegen.py:    `buf` should be in gpu shared memory, we access it many times.
theano/gpuarray/kernel_codegen.py:    ctype = gpuarray.dtype_to_ctype(dtype)
theano/gpuarray/neighbours.py:    from pygpu import gpuarray
theano/gpuarray/neighbours.py:from .basic_ops import (as_gpuarray_variable, GpuKernelBase, Kernel,
theano/gpuarray/neighbours.py:from .type import GpuArrayType, gpu_context_type
theano/gpuarray/neighbours.py:class GpuImages2Neibs(GpuKernelBase, Images2Neibs, Op):
theano/gpuarray/neighbours.py:    Images2Neibs for the GPU.
theano/gpuarray/neighbours.py:    params_type = ParamsType(mode=Images2Neibs.BORDER_MODE, context=gpu_context_type)
theano/gpuarray/neighbours.py:        ten4 = as_gpuarray_variable(ten4, infer_context_name(ten4))
theano/gpuarray/neighbours.py:                     [GpuArrayType(broadcastable=(False, False),
theano/gpuarray/neighbours.py:        return ['<numpy_compat.h>', '<gpuarray/types.h>']
theano/gpuarray/neighbours.py:    def gpu_kernels(self, node, nodename):
theano/gpuarray/neighbours.py:        type_ten4 = gpuarray.dtype_to_ctype(dtype_ten4)
theano/gpuarray/neighbours.py:        type_z = gpuarray.dtype_to_ctype(dtype_z)
theano/gpuarray/neighbours.py:            gpuarray.GpuArray, 'uintp',
theano/gpuarray/neighbours.py:            gpuarray.GpuArray, 'uintp',
theano/gpuarray/neighbours.py:            gpuarray.GpuArray, 'uintp',
theano/gpuarray/neighbours.py:            gpuarray.GpuArray, 'uintp',
theano/gpuarray/neighbours.py:                             "gpuarray error: *fptr: %%s.",
theano/gpuarray/neighbours.py:                             GpuKernel_error(fptr, err));
theano/gpuarray/neighbours.py:        # For itemsize_ten4, I use GpuArray_ITEMSIZE(&ten4->ga) instead of np.dtype(node.inputs[0].dtype).itemsize
theano/gpuarray/neighbours.py:        size_t itemsize_ten4 = GpuArray_ITEMSIZE(&%(ten4)s->ga);
theano/gpuarray/neighbours.py:            if (PyGpuArray_NDIM(%(ten4)s) != 4)
theano/gpuarray/neighbours.py:                             "GpuImages2Neibs: pvals wrong rank");
theano/gpuarray/neighbours.py:                             "GpuImages2Neibs: unis wrong rank");
theano/gpuarray/neighbours.py:                             "GpuImages2Neibs: neib_shape has to contain two"
theano/gpuarray/neighbours.py:                                 "GpuImages2Neibs: in mode wrap_centered need patch with odd shapes");
theano/gpuarray/neighbours.py:                if ( PyGpuArray_DIMS(%(ten4)s)[2] < c ||
theano/gpuarray/neighbours.py:                     PyGpuArray_DIMS(%(ten4)s)[3] < d)
theano/gpuarray/neighbours.py:                                 "GpuImages2Neibs: in wrap_centered mode,"
theano/gpuarray/neighbours.py:                                 c, d, PyGpuArray_DIMS(%(ten4)s)[2],
theano/gpuarray/neighbours.py:                                 PyGpuArray_DIMS(%(ten4)s)[3]);
theano/gpuarray/neighbours.py:                grid_c = ceil_intdiv(((PyGpuArray_DIMS(%(ten4)s))[2]),
theano/gpuarray/neighbours.py:                grid_d = ceil_intdiv(((PyGpuArray_DIMS(%(ten4)s))[3]),
theano/gpuarray/neighbours.py:                if ( ((PyGpuArray_DIMS(%(ten4)s))[2] < c) ||
theano/gpuarray/neighbours.py:                     ((((PyGpuArray_DIMS(%(ten4)s))[2]-c) %% step_x)!=0))
theano/gpuarray/neighbours.py:                    PyErr_Format(PyExc_TypeError, "GpuImages2Neibs:"
theano/gpuarray/neighbours.py:                                 PyGpuArray_DIMS(%(ten4)s)[2]);
theano/gpuarray/neighbours.py:                if ( ((PyGpuArray_DIMS(%(ten4)s))[3] < d) ||
theano/gpuarray/neighbours.py:                     ((((PyGpuArray_DIMS(%(ten4)s))[3]-d) %% step_y)!=0))
theano/gpuarray/neighbours.py:                    PyErr_Format(PyExc_TypeError, "GpuImages2Neibs:"
theano/gpuarray/neighbours.py:                                 PyGpuArray_DIMS(%(ten4)s)[3]);
theano/gpuarray/neighbours.py:                grid_c = 1+(((PyGpuArray_DIMS(%(ten4)s))[2]-c)/step_x);
theano/gpuarray/neighbours.py:                grid_d = 1+(((PyGpuArray_DIMS(%(ten4)s))[3]-d)/step_y);
theano/gpuarray/neighbours.py:                grid_c = 1+(((PyGpuArray_DIMS(%(ten4)s))[2]-c)/step_x);
theano/gpuarray/neighbours.py:                grid_d = 1+(((PyGpuArray_DIMS(%(ten4)s))[3]-d)/step_y);
theano/gpuarray/neighbours.py:                if ( ((PyGpuArray_DIMS(%(ten4)s))[2] < c) ||
theano/gpuarray/neighbours.py:                     ((((PyGpuArray_DIMS(%(ten4)s))[2]-(c%%2)) %% step_x)!=0))
theano/gpuarray/neighbours.py:                    PyErr_Format(PyExc_TypeError, "GpuImages2Neibs:"
theano/gpuarray/neighbours.py:                                 PyGpuArray_DIMS(%(ten4)s)[2]);
theano/gpuarray/neighbours.py:                if ( ((PyGpuArray_DIMS(%(ten4)s))[3] < d) ||
theano/gpuarray/neighbours.py:                     ((((PyGpuArray_DIMS(%(ten4)s))[3]-(d%%2)) %% step_y)!=0))
theano/gpuarray/neighbours.py:                    PyErr_Format(PyExc_TypeError, "GpuImages2Neibs:"
theano/gpuarray/neighbours.py:                                 PyGpuArray_DIMS(%(ten4)s)[3]);
theano/gpuarray/neighbours.py:                grid_c = 1+(((PyGpuArray_DIMS(%(ten4)s))[2]-(c%%2))/step_x);
theano/gpuarray/neighbours.py:                grid_d = 1+(((PyGpuArray_DIMS(%(ten4)s))[3]-(d%%2))/step_y);
theano/gpuarray/neighbours.py:                if ( ((PyGpuArray_DIMS(%(ten4)s))[2] < c) ||
theano/gpuarray/neighbours.py:                 ( (((PyGpuArray_DIMS(%(ten4)s))[2]+c-2) %% step_x)!=0))
theano/gpuarray/neighbours.py:                                 (long int)(PyGpuArray_DIMS(%(ten4)s)[2]));
theano/gpuarray/neighbours.py:                if ( ((PyGpuArray_DIMS(%(ten4)s))[3] < d) ||
theano/gpuarray/neighbours.py:                     ( (((PyGpuArray_DIMS(%(ten4)s))[3]+d-2) %% step_y)!=0))
theano/gpuarray/neighbours.py:                                 (long int)(PyGpuArray_DIMS(%(ten4)s)[3]));
theano/gpuarray/neighbours.py:                grid_c = 1+(((PyGpuArray_DIMS(%(ten4)s))[2]+c-2)/step_x);
theano/gpuarray/neighbours.py:                grid_d = 1+(((PyGpuArray_DIMS(%(ten4)s))[3]+d-2)/step_y);
theano/gpuarray/neighbours.py:                             "GpuImages2Neibs:: unknown mode %%d", %(params)s->mode);
theano/gpuarray/neighbours.py:                                * PyGpuArray_DIMS(%(ten4)s)[1]
theano/gpuarray/neighbours.py:                                * PyGpuArray_DIMS(%(ten4)s)[0];
theano/gpuarray/neighbours.py:                || (PyGpuArray_DIMS(%(z)s)[0] != z_dim0)
theano/gpuarray/neighbours.py:                || (PyGpuArray_DIMS(%(z)s)[1] != z_dim1))
theano/gpuarray/neighbours.py:                %(z)s = pygpu_empty(2, dims, typecode_z,
theano/gpuarray/neighbours.py:                    PyErr_SetString(PyExc_MemoryError, "GpuImages2Neibs:"
theano/gpuarray/neighbours.py:            const int nb_batch = PyGpuArray_DIMS(%(ten4)s)[0];
theano/gpuarray/neighbours.py:            const int nb_stack = PyGpuArray_DIMS(%(ten4)s)[1];
theano/gpuarray/neighbours.py:            const int height = PyGpuArray_DIMS(%(ten4)s)[2];
theano/gpuarray/neighbours.py:            const int width = PyGpuArray_DIMS(%(ten4)s)[3];
theano/gpuarray/neighbours.py:            int err = gpucontext_property(%(params)s->context->ctx, GA_CTX_PROP_MAXLSIZE0, &max_threads_dim);
theano/gpuarray/neighbours.py:                  threads_per_block[2]<PyGpuArray_DIMS(%(z)s)[0]){
theano/gpuarray/neighbours.py:            if (PyGpuArray_DIMS(%(z)s)[0] %% threads_per_block[2] == 0)
theano/gpuarray/neighbours.py:                nb_block = PyGpuArray_DIMS(%(z)s)[0] / threads_per_block[2];
theano/gpuarray/neighbours.py:                nb_block = (PyGpuArray_DIMS(%(z)s)[0] / threads_per_block[2]) + 1;
theano/gpuarray/neighbours.py:            GpuKernel *fptr;
theano/gpuarray/neighbours.py:            size_t stride_A0 = PyGpuArray_STRIDES(%(ten4)s)[0] / itemsize_ten4;
theano/gpuarray/neighbours.py:            size_t stride_A1 = PyGpuArray_STRIDES(%(ten4)s)[1] / itemsize_ten4;
theano/gpuarray/neighbours.py:            size_t stride_A2 = PyGpuArray_STRIDES(%(ten4)s)[2] / itemsize_ten4;
theano/gpuarray/neighbours.py:            size_t stride_A3 = PyGpuArray_STRIDES(%(ten4)s)[3] / itemsize_ten4;
theano/gpuarray/neighbours.py:            size_t stride_Z0 = PyGpuArray_STRIDES(%(z)s)[0] / itemsize_z;
theano/gpuarray/neighbours.py:            size_t stride_Z1 = PyGpuArray_STRIDES(%(z)s)[1] / itemsize_z;
theano/gpuarray/neighbours.py:            err = GpuKernel_call(fptr, 3, n_blocks, threads_per_block, 0, kernel_params);
theano/gpuarray/pool.py:from .type import gpu_context_type
theano/gpuarray/pool.py:from .basic_ops import (CGpuKernelBase, infer_context_name, gpuarray_helper_inc_dir,
theano/gpuarray/pool.py:                        as_gpuarray_variable, gpu_contiguous)
theano/gpuarray/pool.py:    import pygpu
theano/gpuarray/pool.py:class GpuPool(CGpuKernelBase):
theano/gpuarray/pool.py:    Implement the max and average pooling on the gpu.
theano/gpuarray/pool.py:                             context=gpu_context_type)
theano/gpuarray/pool.py:        CGpuKernelBase.__init__(self, ['c_code/pool.c'],
theano/gpuarray/pool.py:        return ['gpuarray_api.h', 'gpuarray_helper.h', 'numpy_compat.h']
theano/gpuarray/pool.py:        return [gpuarray_helper_inc_dir(), pygpu.get_include()]
theano/gpuarray/pool.py:        inp = as_gpuarray_variable(inp, ctx_name)
theano/gpuarray/pool.py:        grad = gpu_contiguous(grad)
theano/gpuarray/pool.py:            g_out = GpuMaxPoolGrad(ndim=self.ndim,
theano/gpuarray/pool.py:            g_out = GpuAveragePoolGrad(ndim=self.ndim,
theano/gpuarray/pool.py:            GpuDownsampleFactorMaxGradGrad(self.ignore_border, self.mode,
theano/gpuarray/pool.py:class GpuMaxPoolGrad(CGpuKernelBase):
theano/gpuarray/pool.py:    Implement the grad of max pooling on the gpu.
theano/gpuarray/pool.py:        CGpuKernelBase.__init__(self, ['c_code/pool_max_grad.c'],
theano/gpuarray/pool.py:        return ['gpuarray_api.h', 'gpuarray_helper.h', 'numpy_compat.h']
theano/gpuarray/pool.py:        return [gpuarray_helper_inc_dir(), pygpu.get_include()]
theano/gpuarray/pool.py:        inp = as_gpuarray_variable(inp, ctx_name)
theano/gpuarray/pool.py:        out = as_gpuarray_variable(out, ctx_name)
theano/gpuarray/pool.py:        out_grad = as_gpuarray_variable(out_grad, ctx_name)
theano/gpuarray/pool.py:                 GpuDownsampleFactorMaxGradGrad(ndim=self.ndim,
theano/gpuarray/pool.py:class GpuAveragePoolGrad(CGpuKernelBase):
theano/gpuarray/pool.py:    Implement the grad of average pooling on the gpu.
theano/gpuarray/pool.py:    params_type = ParamsType(mode=PoolingMode_t, context=gpu_context_type)
theano/gpuarray/pool.py:        CGpuKernelBase.__init__(self, ['c_code/pool_ave_grad.c'],
theano/gpuarray/pool.py:        return ['gpuarray_api.h', 'gpuarray_helper.h', 'numpy_compat.h']
theano/gpuarray/pool.py:        return [gpuarray_helper_inc_dir(), pygpu.get_include()]
theano/gpuarray/pool.py:        inp = as_gpuarray_variable(inp, ctx_name)
theano/gpuarray/pool.py:        out_grad = as_gpuarray_variable(out_grad, ctx_name)
theano/gpuarray/pool.py:                 GpuPool(ignore_border=self.ignore_border,
theano/gpuarray/pool.py:class GpuDownsampleFactorMaxGradGrad(CGpuKernelBase):
theano/gpuarray/pool.py:    Implement the grad of downsample with max on the gpu.
theano/gpuarray/pool.py:        CGpuKernelBase.__init__(self, ['c_code/pool_grad_grad.c'],
theano/gpuarray/pool.py:        return ['gpuarray_api.h', 'gpuarray_helper.h', 'numpy_compat.h']
theano/gpuarray/pool.py:        return [gpuarray_helper_inc_dir(), pygpu.get_include()]
theano/gpuarray/pool.py:        inp = as_gpuarray_variable(inp, ctx_name)
theano/gpuarray/pool.py:        out = as_gpuarray_variable(out, ctx_name)
theano/gpuarray/pool.py:        out_grad = as_gpuarray_variable(out_grad, ctx_name)
theano/gpuarray/pool.py:                GpuMaxPoolGrad(ignore_border=self.ignore_border,
theano/gpuarray/pool.py:class GpuMaxPoolRop(CGpuKernelBase):
theano/gpuarray/pool.py:    params_type = ParamsType(ignore_border=bool_t, context=gpu_context_type)
theano/gpuarray/pool.py:        CGpuKernelBase.__init__(self, ['c_code/pool_max_rop.c'],
theano/gpuarray/pool.py:        return ['gpuarray_api.h', 'gpuarray_helper.h', 'numpy_compat.h']
theano/gpuarray/pool.py:        return [gpuarray_helper_inc_dir(), pygpu.get_include()]
theano/gpuarray/pool.py:        inp = as_gpuarray_variable(inp, ctx_name)
theano/gpuarray/pool.py:        eval_point = as_gpuarray_variable(eval_point, ctx_name)
theano/gpuarray/tests/test_extra_ops.py:from .config import mode_with_gpu, test_ctx_name
theano/gpuarray/tests/test_extra_ops.py:from ..extra_ops import GpuCumOp
theano/gpuarray/tests/test_extra_ops.py:class TestGpuCumOp(theano.tensor.tests.test_extra_ops.TestCumOp):
theano/gpuarray/tests/test_extra_ops.py:    mode = mode_with_gpu
theano/gpuarray/tests/test_extra_ops.py:        super(TestGpuCumOp, self).setUp()
theano/gpuarray/tests/test_extra_ops.py:        if test_ctx.kind != b'cuda':
theano/gpuarray/tests/test_extra_ops.py:            raise SkipTest("Cuda specific tests")
theano/gpuarray/tests/test_extra_ops.py:        # GpuFromHost seems overkill, we just relax the rtol for these tests
theano/gpuarray/tests/test_extra_ops.py:        super(TestGpuCumOp, self).tearDown()
theano/gpuarray/tests/test_extra_ops.py:        # GpuCumOp is only defined for float32 for now, so we skip it
theano/gpuarray/tests/test_extra_ops.py:        gpucumop_supported_dtypes = ('float32',)
theano/gpuarray/tests/test_extra_ops.py:        if theano.config.floatX not in gpucumop_supported_dtypes:
theano/gpuarray/tests/test_extra_ops.py:            raise SkipTest('Gpucumop not implemented for dtype %s'
theano/gpuarray/tests/test_extra_ops.py:                                    GpuCumOp)
theano/gpuarray/tests/test_extra_ops.py:        # no grad for GpuCumOp
theano/gpuarray/tests/test_extra_ops.py:                        if isinstance(n.op, GpuCumOp)]
theano/gpuarray/tests/test_extra_ops.py:                        if isinstance(n.op, GpuCumOp)]
theano/gpuarray/tests/test_extra_ops.py:                        if isinstance(n.op, GpuCumOp)]
theano/gpuarray/tests/test_extra_ops.py:    def test_GpuCumOp1D(self, mode):
theano/gpuarray/tests/test_extra_ops.py:                if isinstance(n.op, GpuCumOp)]
theano/gpuarray/tests/test_extra_ops.py:        # Use multiple GPU threadblocks
theano/gpuarray/tests/test_extra_ops.py:    def test_GpuCumOp2D(self, mode):
theano/gpuarray/tests/test_extra_ops.py:                    if isinstance(n.op, GpuCumOp)]
theano/gpuarray/tests/test_extra_ops.py:            # Use multiple GPU threadblocks
theano/gpuarray/tests/test_extra_ops.py:            # Use multiple GPU gridblocks
theano/gpuarray/tests/test_extra_ops.py:    def test_GpuCumOp3D(self, mode):
theano/gpuarray/tests/test_extra_ops.py:                    if isinstance(n.op, GpuCumOp)]
theano/gpuarray/tests/test_extra_ops.py:            # Use multiple GPU threadblocks (along accumulation axis)
theano/gpuarray/tests/test_extra_ops.py:            # Use multiple GPU gridblocks (not along accumulation axis)
theano/gpuarray/tests/test_extra_ops.py:    def test_GpuCumOp4D(self, mode):
theano/gpuarray/tests/test_extra_ops.py:        # Should not use the GPU version.
theano/gpuarray/tests/test_gemmcorr3d.py:from ..type import gpuarray_shared_constructor
theano/gpuarray/tests/test_gemmcorr3d.py:from ..blas import GpuCorr3dMM, GpuCorr3dMM_gradWeights, GpuCorr3dMM_gradInputs
theano/gpuarray/tests/test_gemmcorr3d.py:from .config import mode_with_gpu, mode_without_gpu, ref_cast
theano/gpuarray/tests/test_gemmcorr3d.py:        inputs = gpuarray_shared_constructor(inputs_val)
theano/gpuarray/tests/test_gemmcorr3d.py:        filters = gpuarray_shared_constructor(filters_val)
theano/gpuarray/tests/test_gemmcorr3d.py:        f_ref = theano.function([], conv_ref, mode=mode_without_gpu)
theano/gpuarray/tests/test_gemmcorr3d.py:        conv = GpuCorr3dMM(border_mode=border_mode,
theano/gpuarray/tests/test_gemmcorr3d.py:        f = theano.function([], conv, mode=mode_with_gpu)
theano/gpuarray/tests/test_gemmcorr3d.py:            utt.verify_grad(GpuCorr3dMM(border_mode=border_mode,
theano/gpuarray/tests/test_gemmcorr3d.py:                            [inputs_val, filters_val], mode=mode_with_gpu)
theano/gpuarray/tests/test_gemmcorr3d.py:        inputs = gpuarray_shared_constructor(inputs_val)
theano/gpuarray/tests/test_gemmcorr3d.py:        dCdH = gpuarray_shared_constructor(dCdH_val)
theano/gpuarray/tests/test_gemmcorr3d.py:        shape = gpuarray_shared_constructor(np.array(filters_shape[2:]))
theano/gpuarray/tests/test_gemmcorr3d.py:            conv_gemm = GpuCorr3dMM_gradWeights(subsample=subsample)(
theano/gpuarray/tests/test_gemmcorr3d.py:            conv_gemm = GpuCorr3dMM_gradWeights(subsample=subsample)(
theano/gpuarray/tests/test_gemmcorr3d.py:        f_ref = theano.function([], conv_ref, mode=mode_without_gpu)
theano/gpuarray/tests/test_gemmcorr3d.py:        f = theano.function([], conv_gemm, mode=mode_with_gpu)
theano/gpuarray/tests/test_gemmcorr3d.py:        inputs = gpuarray_shared_constructor(inputs_val)
theano/gpuarray/tests/test_gemmcorr3d.py:        filters = gpuarray_shared_constructor(filters_val)
theano/gpuarray/tests/test_gemmcorr3d.py:        bottom_shape = gpuarray_shared_constructor(np.array([bottom_height, bottom_width, bottom_depth]))
theano/gpuarray/tests/test_gemmcorr3d.py:            conv_gemm = GpuCorr3dMM_gradInputs(subsample=subsample)(
theano/gpuarray/tests/test_gemmcorr3d.py:            conv_gemm = GpuCorr3dMM_gradInputs(subsample=subsample)(
theano/gpuarray/tests/test_gemmcorr3d.py:        f_ref = theano.function([], conv_ref, mode=mode_without_gpu)
theano/gpuarray/tests/test_gemmcorr3d.py:        f = theano.function([], conv_gemm, mode=mode_with_gpu)
theano/gpuarray/tests/test_gemmcorr3d.py:class TestGroupGpuCorr3d(Grouped_conv3d_noOptim):
theano/gpuarray/tests/test_gemmcorr3d.py:    mode = mode_with_gpu.excluding('cudnn')
theano/gpuarray/tests/test_gemmcorr3d.py:    conv_op = GpuCorr3dMM
theano/gpuarray/tests/test_gemmcorr3d.py:    conv_gradw_op = GpuCorr3dMM_gradWeights
theano/gpuarray/tests/test_gemmcorr3d.py:    conv_gradi_op = GpuCorr3dMM_gradInputs
theano/gpuarray/tests/test_fft.py:import theano.gpuarray.fft
theano/gpuarray/tests/test_fft.py:from .config import mode_with_gpu
theano/gpuarray/tests/test_fft.py:# Skip tests if pygpu is not available.
theano/gpuarray/tests/test_fft.py:from theano.gpuarray.fft import pygpu_available, skcuda_available, pycuda_available
theano/gpuarray/tests/test_fft.py:if not pygpu_available:  # noqa
theano/gpuarray/tests/test_fft.py:    raise SkipTest('Optional package pygpu not available')
theano/gpuarray/tests/test_fft.py:if not skcuda_available:  # noqa
theano/gpuarray/tests/test_fft.py:    raise SkipTest('Optional package scikit-cuda not available')
theano/gpuarray/tests/test_fft.py:if not pycuda_available:  # noqa
theano/gpuarray/tests/test_fft.py:    raise SkipTest('Optional package pycuda not available')
theano/gpuarray/tests/test_fft.py:        rfft = theano.gpuarray.fft.curfft(x)
theano/gpuarray/tests/test_fft.py:        f_rfft = theano.function([x], rfft, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:        irfft = theano.gpuarray.fft.cuirfft(m)
theano/gpuarray/tests/test_fft.py:        f_irfft = theano.function([m], irfft, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:            return theano.gpuarray.fft.curfft(inp)
theano/gpuarray/tests/test_fft.py:        utt.verify_grad(f_rfft, [inputs_val], eps=eps, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:            return theano.gpuarray.fft.cuirfft(inp)
theano/gpuarray/tests/test_fft.py:        utt.verify_grad(f_irfft, [inputs_val], eps=eps, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:        rfft = theano.gpuarray.fft.curfft(inputs)
theano/gpuarray/tests/test_fft.py:        f_rfft = theano.function([], rfft, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:        fft = theano.gpuarray.fft.curfft(inputs)
theano/gpuarray/tests/test_fft.py:        f_fft = theano.function([], fft, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:        ifft = theano.gpuarray.fft.cuirfft(m)
theano/gpuarray/tests/test_fft.py:        f_ifft = theano.function([m], ifft, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:        irfft = theano.gpuarray.fft.cuirfft(inputs)
theano/gpuarray/tests/test_fft.py:        f_irfft = theano.function([], irfft, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:            theano.gpuarray.fft.curfft(inputs)
theano/gpuarray/tests/test_fft.py:            theano.gpuarray.fft.cuirfft(inputs)
theano/gpuarray/tests/test_fft.py:        rfft = theano.gpuarray.fft.curfft(inputs, norm='ortho')
theano/gpuarray/tests/test_fft.py:        f_rfft = theano.function([], rfft, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:        rfft = theano.gpuarray.fft.curfft(inputs, norm='no_norm')
theano/gpuarray/tests/test_fft.py:        f_rfft = theano.function([], rfft, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:        irfft = theano.gpuarray.fft.cuirfft(inputs, norm='ortho')
theano/gpuarray/tests/test_fft.py:        f_irfft = theano.function([], irfft, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:        irfft = theano.gpuarray.fft.cuirfft(inputs, norm='no_norm')
theano/gpuarray/tests/test_fft.py:        f_irfft = theano.function([], irfft, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:            return theano.gpuarray.fft.curfft(inp)
theano/gpuarray/tests/test_fft.py:        utt.verify_grad(f_rfft, [inputs_val], eps=eps, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:            return theano.gpuarray.fft.cuirfft(inp)
theano/gpuarray/tests/test_fft.py:        utt.verify_grad(f_irfft, [inputs_val], eps=eps, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:            return theano.gpuarray.fft.curfft(inp, norm='ortho')
theano/gpuarray/tests/test_fft.py:        utt.verify_grad(f_rfft, [inputs_val], eps=eps, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:            return theano.gpuarray.fft.cuirfft(inp, norm='no_norm')
theano/gpuarray/tests/test_fft.py:        utt.verify_grad(f_irfft, [inputs_val], eps=eps, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:        rfft = theano.gpuarray.fft.curfft(inputs)
theano/gpuarray/tests/test_fft.py:        f_rfft = theano.function([], rfft, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:        ifft = theano.gpuarray.fft.cuirfft(m, is_odd=True)
theano/gpuarray/tests/test_fft.py:        f_ifft = theano.function([m], ifft, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:        irfft = theano.gpuarray.fft.cuirfft(inputs, norm='ortho', is_odd=True)
theano/gpuarray/tests/test_fft.py:        f_irfft = theano.function([], irfft, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:            return theano.gpuarray.fft.curfft(inp)
theano/gpuarray/tests/test_fft.py:        utt.verify_grad(f_rfft, [inputs_val], eps=eps, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:            return theano.gpuarray.fft.cuirfft(inp, is_odd=True)
theano/gpuarray/tests/test_fft.py:        utt.verify_grad(f_irfft, [inputs_val], eps=eps, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:            return theano.gpuarray.fft.curfft(inp, norm='ortho')
theano/gpuarray/tests/test_fft.py:        utt.verify_grad(f_rfft, [inputs_val], eps=eps, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:            return theano.gpuarray.fft.cuirfft(inp, norm='no_norm', is_odd=True)
theano/gpuarray/tests/test_fft.py:        utt.verify_grad(f_irfft, [inputs_val], eps=eps, mode=mode_with_gpu)
theano/gpuarray/tests/test_fft.py:        self.assertRaises(ValueError, theano.gpuarray.fft.curfft, inputs, norm=123)
theano/gpuarray/tests/test_fft.py:        self.assertRaises(ValueError, theano.gpuarray.fft.cuirfft, inputs, norm=123)
theano/gpuarray/tests/test_fft.py:        self.assertRaises(ValueError, theano.gpuarray.fft.cuirfft, inputs, is_odd=123)
theano/gpuarray/tests/test_reduction.py:from .config import mode_with_gpu, mode_without_gpu
theano/gpuarray/tests/test_reduction.py:from .test_basic_ops import rand_gpuarray
theano/gpuarray/tests/test_reduction.py:from .. import GpuArrayType
theano/gpuarray/tests/test_reduction.py:from ..reduction import GpuMaxAndArgmax
theano/gpuarray/tests/test_reduction.py:from ..dnn import GpuDnnReduction
theano/gpuarray/tests/test_reduction.py:def check_if_gpu_reduce_in_graph(theano_function):
theano/gpuarray/tests/test_reduction.py:    assert any(isinstance(node.op, (GpuMaxAndArgmax, GpuDnnReduction))
theano/gpuarray/tests/test_reduction.py:def check_if_gpu_reduce_not_in_graph(theano_function):
theano/gpuarray/tests/test_reduction.py:    assert all(not isinstance(node.op, (GpuMaxAndArgmax, GpuDnnReduction))
theano/gpuarray/tests/test_reduction.py:    def get_gpu_tensor(self):
theano/gpuarray/tests/test_reduction.py:        return GpuArrayType(self.dtype, broadcastable)()
theano/gpuarray/tests/test_reduction.py:    def get_gpu_value(self):
theano/gpuarray/tests/test_reduction.py:        return rand_gpuarray(*self.shape)
theano/gpuarray/tests/test_reduction.py:    # NB: In compute_host() and compute_gpu(),
theano/gpuarray/tests/test_reduction.py:                            name='shape:' + str(test_tensor.shape) + '/axis:' + str(axis) + '/HOST', mode=mode_without_gpu)
theano/gpuarray/tests/test_reduction.py:        check_if_gpu_reduce_not_in_graph(f)
theano/gpuarray/tests/test_reduction.py:    def compute_gpu(self, test_gpu_tensor, test_host_tensor, axis):
theano/gpuarray/tests/test_reduction.py:        M = self.get_gpu_tensor()
theano/gpuarray/tests/test_reduction.py:                            name='shape:' + str(test_gpu_tensor.shape) + '/axis:' + str(axis) + '/GPU', mode=mode_with_gpu)
theano/gpuarray/tests/test_reduction.py:        check_if_gpu_reduce_in_graph(f)
theano/gpuarray/tests/test_reduction.py:        f(test_gpu_tensor)
theano/gpuarray/tests/test_reduction.py:        theano_max, theano_argmax = f(test_gpu_tensor)
theano/gpuarray/tests/test_reduction.py:        # We want to run CPU op and GPU op on the same tensor randomly generated.
theano/gpuarray/tests/test_reduction.py:        test_gpu_tensor = self.get_gpu_value()
theano/gpuarray/tests/test_reduction.py:        test_host_tensor = np.asarray(test_gpu_tensor)
theano/gpuarray/tests/test_reduction.py:        self.compute_gpu(test_gpu_tensor, test_host_tensor, axis)
theano/gpuarray/tests/test_abstractconv.py:from ..type import GpuArrayType, gpuarray_shared_constructor, get_context
theano/gpuarray/tests/test_abstractconv.py:from ..dnn import dnn_available, GpuDnnConv, GpuDnnConvGradW, GpuDnnConvGradI
theano/gpuarray/tests/test_abstractconv.py:    GpuCorrMM, GpuCorrMM_gradWeights, GpuCorrMM_gradInputs,
theano/gpuarray/tests/test_abstractconv.py:    GpuCorr3dMM, GpuCorr3dMM_gradWeights, GpuCorr3dMM_gradInputs)
theano/gpuarray/tests/test_abstractconv.py:from .config import mode_with_gpu, test_ctx_name
theano/gpuarray/tests/test_abstractconv.py:from pygpu import gpuarray
theano/gpuarray/tests/test_abstractconv.py:gpu_ftensor4 = GpuArrayType(dtype='float32', broadcastable=(False,) * 4)
theano/gpuarray/tests/test_abstractconv.py:        cls.shared = staticmethod(gpuarray_shared_constructor)
theano/gpuarray/tests/test_abstractconv.py:        mode = mode_with_gpu
theano/gpuarray/tests/test_abstractconv.py:                     filter_flip=flip, target_op=GpuDnnConv)
theano/gpuarray/tests/test_abstractconv.py:                            filter_flip=flip, target_op=GpuDnnConvGradW)
theano/gpuarray/tests/test_abstractconv.py:                           filter_flip=flip, target_op=GpuDnnConvGradI)
theano/gpuarray/tests/test_abstractconv.py:        mode = mode_with_gpu
theano/gpuarray/tests/test_abstractconv.py:                               filter_flip=flip, target_op=GpuDnnConvGradI,
theano/gpuarray/tests/test_abstractconv.py:                          filter_flip=flip, target_op=GpuDnnConvGradI,
theano/gpuarray/tests/test_abstractconv.py:        cls.shared = staticmethod(gpuarray_shared_constructor)
theano/gpuarray/tests/test_abstractconv.py:        mode = mode_with_gpu
theano/gpuarray/tests/test_abstractconv.py:                     filter_flip=flip, target_op=GpuDnnConv)
theano/gpuarray/tests/test_abstractconv.py:                            filter_flip=flip, target_op=GpuDnnConvGradW)
theano/gpuarray/tests/test_abstractconv.py:                           filter_flip=flip, target_op=GpuDnnConvGradI)
theano/gpuarray/tests/test_abstractconv.py:        mode = mode_with_gpu
theano/gpuarray/tests/test_abstractconv.py:                               filter_flip=flip, target_op=GpuDnnConvGradI,
theano/gpuarray/tests/test_abstractconv.py:                          filter_flip=flip, target_op=GpuDnnConvGradI,
theano/gpuarray/tests/test_abstractconv.py:        cls.shared = staticmethod(gpuarray_shared_constructor)
theano/gpuarray/tests/test_abstractconv.py:        cls.mode = mode_with_gpu.excluding('cudnn')
theano/gpuarray/tests/test_abstractconv.py:                     filter_flip=flip, target_op=(GpuCorrMM,
theano/gpuarray/tests/test_abstractconv.py:                                                  GpuCorrMM_gradWeights,
theano/gpuarray/tests/test_abstractconv.py:                                                  GpuCorrMM_gradInputs),
theano/gpuarray/tests/test_abstractconv.py:                            target_op=GpuCorrMM_gradWeights,
theano/gpuarray/tests/test_abstractconv.py:                           target_op=GpuCorrMM_gradInputs,
theano/gpuarray/tests/test_abstractconv.py:                               target_op=GpuCorrMM_gradInputs,
theano/gpuarray/tests/test_abstractconv.py:                          target_op=GpuCorrMM_gradInputs,
theano/gpuarray/tests/test_abstractconv.py:        cls.shared = staticmethod(gpuarray_shared_constructor)
theano/gpuarray/tests/test_abstractconv.py:        cls.mode = mode_with_gpu.excluding('cudnn')
theano/gpuarray/tests/test_abstractconv.py:                     filter_flip=flip, target_op=(GpuCorr3dMM,
theano/gpuarray/tests/test_abstractconv.py:                                                  GpuCorr3dMM_gradWeights,
theano/gpuarray/tests/test_abstractconv.py:                                                  GpuCorr3dMM_gradInputs),
theano/gpuarray/tests/test_abstractconv.py:                            target_op=GpuCorr3dMM_gradWeights,
theano/gpuarray/tests/test_abstractconv.py:                           target_op=GpuCorr3dMM_gradInputs,
theano/gpuarray/tests/test_abstractconv.py:                               target_op=GpuCorr3dMM_gradInputs,
theano/gpuarray/tests/test_abstractconv.py:                          target_op=GpuCorr3dMM_gradInputs,
theano/gpuarray/tests/test_abstractconv.py:        self.input = gpu_ftensor4()
theano/gpuarray/tests/test_abstractconv.py:        self.filters = gpu_ftensor4()
theano/gpuarray/tests/test_abstractconv.py:        self.topgrad = gpu_ftensor4()
theano/gpuarray/tests/test_abstractconv.py:        self.constant_tensor = gpuarray.array(
theano/gpuarray/tests/test_abstractconv.py:    mode = mode_with_gpu
theano/gpuarray/tests/test_others.py:from .config import test_ctx_name, mode_with_gpu
theano/gpuarray/tests/test_others.py:from ..basic_ops import (HostFromGpu, GpuFromHost)
theano/gpuarray/tests/test_others.py:from ..type import (get_context, GpuArrayType, GpuArraySharedVariable,
theano/gpuarray/tests/test_others.py:                    gpuarray_shared_constructor)
theano/gpuarray/tests/test_others.py:import pygpu
theano/gpuarray/tests/test_others.py:    mode = mode_with_gpu.excluding('local_dnn_reduction')
theano/gpuarray/tests/test_others.py:    _shared = staticmethod(gpuarray_shared_constructor)
theano/gpuarray/tests/test_others.py:    topo_exclude = (GpuFromHost, HostFromGpu)
theano/gpuarray/tests/test_others.py:    a = pygpu.empty((5, 4), context=ctx)
theano/gpuarray/tests/test_others.py:    b = pygpu.empty((5, 4), context=ctx)
theano/gpuarray/tests/test_others.py:    x = GpuArraySharedVariable('x',
theano/gpuarray/tests/test_others.py:                               GpuArrayType('float32', (1, 1), name='x',
theano/gpuarray/tests/check_dnn_conv.py:# supported algorithms and data type configurations for current GPU and cuDNN version.
theano/gpuarray/tests/check_dnn_conv.py:from theano.gpuarray import cudnn_defs
theano/gpuarray/tests/check_dnn_conv.py:from theano.gpuarray.dnn import (GpuDnnConv, GpuDnnConvGradW, GpuDnnConvGradI, version,
theano/gpuarray/tests/check_dnn_conv.py:from theano.gpuarray.tests.config import mode_with_gpu, ref_cast
theano/gpuarray/tests/check_dnn_conv.py:    f = theano.function([], conv, mode=mode_with_gpu)
theano/gpuarray/tests/check_dnn_conv.py:        f = theano.function([], conv, mode=mode_with_gpu)
theano/gpuarray/tests/check_dnn_conv.py:        f = theano.function([], grad_i, mode=mode_with_gpu)
theano/gpuarray/tests/check_dnn_conv.py:        f = theano.function([], grad_w, mode=mode_with_gpu)
theano/gpuarray/tests/check_dnn_conv.py:            f = theano.function([inputs, filters], conv, mode=mode_with_gpu)
theano/gpuarray/tests/check_dnn_conv.py:                    gpu_res = np.asarray(f(inputs_val, filters_val))
theano/gpuarray/tests/check_dnn_conv.py:                    self.scale_numpy_arrays_inplace(cpu_res, gpu_res, 1)
theano/gpuarray/tests/check_dnn_conv.py:                    utt.assert_allclose(cpu_res, gpu_res)
theano/gpuarray/tests/check_dnn_conv.py:            f = theano.function([inputs, filters], grad_i, mode=mode_with_gpu)
theano/gpuarray/tests/check_dnn_conv.py:            assert 1 == len([node for node in f.maker.fgraph.apply_nodes if isinstance(node.op, GpuDnnConvGradI)])
theano/gpuarray/tests/check_dnn_conv.py:            assert not any(isinstance(node.op, GpuDnnConv) for node in f.maker.fgraph.apply_nodes)
theano/gpuarray/tests/check_dnn_conv.py:            assert not any(isinstance(node.op, GpuDnnConvGradW) for node in f.maker.fgraph.apply_nodes)
theano/gpuarray/tests/check_dnn_conv.py:                    gpu_res = f(inputs_val, filters_val)
theano/gpuarray/tests/check_dnn_conv.py:                    utt.assert_allclose(cpu_res, np.asarray(gpu_res))
theano/gpuarray/tests/check_dnn_conv.py:            f = theano.function([inputs, filters], grad_w, mode=mode_with_gpu)
theano/gpuarray/tests/check_dnn_conv.py:            assert 1 == len([node for node in f.maker.fgraph.apply_nodes if isinstance(node.op, GpuDnnConvGradW)])
theano/gpuarray/tests/check_dnn_conv.py:            assert not any(isinstance(node.op, GpuDnnConv) for node in f.maker.fgraph.apply_nodes)
theano/gpuarray/tests/check_dnn_conv.py:            assert not any(isinstance(node.op, GpuDnnConvGradI) for node in f.maker.fgraph.apply_nodes)
theano/gpuarray/tests/check_dnn_conv.py:                    gpu_res = f(inputs_val, filters_val)
theano/gpuarray/tests/check_dnn_conv.py:                    utt.assert_allclose(cpu_res, np.asarray(gpu_res))
theano/gpuarray/tests/check_dnn_conv.py:        raise SkipTest('FWD: TRUE_HALF_CONFIG not supported on this GPU.')
theano/gpuarray/tests/test_linalg.py:from theano.gpuarray.linalg import (GpuCusolverSolve, GpuCublasTriangularSolve,
theano/gpuarray/tests/test_linalg.py:                                    GpuCholesky, GpuMagmaCholesky,
theano/gpuarray/tests/test_linalg.py:                                    GpuMagmaEigh, GpuMagmaMatrixInverse,
theano/gpuarray/tests/test_linalg.py:                                    GpuMagmaQR, GpuMagmaSVD,
theano/gpuarray/tests/test_linalg.py:                                    cusolver_available, gpu_matrix_inverse,
theano/gpuarray/tests/test_linalg.py:                                    gpu_cholesky,
theano/gpuarray/tests/test_linalg.py:                                    gpu_solve, gpu_solve_lower_triangular,
theano/gpuarray/tests/test_linalg.py:                                    gpu_svd, gpu_qr)
theano/gpuarray/tests/test_linalg.py:from .. import gpuarray_shared_constructor
theano/gpuarray/tests/test_linalg.py:from .config import mode_with_gpu, mode_without_gpu
theano/gpuarray/tests/test_linalg.py:            self.skipTest('Optional package scikits.cuda.cusolver not available')
theano/gpuarray/tests/test_linalg.py:    def run_gpu_solve(self, A_val, x_val, A_struct=None):
theano/gpuarray/tests/test_linalg.py:            solver = gpu_solve(A, b)
theano/gpuarray/tests/test_linalg.py:            solver_trans = gpu_solve(A, b_trans, trans='T')
theano/gpuarray/tests/test_linalg.py:            solver = gpu_solve(A, b, A_struct)
theano/gpuarray/tests/test_linalg.py:            solver_trans = gpu_solve(A, b_trans, A_struct, trans='T')
theano/gpuarray/tests/test_linalg.py:        fn = theano.function([A, b, b_trans], [solver, solver_trans], mode=mode_with_gpu)
theano/gpuarray/tests/test_linalg.py:        self.run_gpu_solve(A_val, x_val)
theano/gpuarray/tests/test_linalg.py:        self.run_gpu_solve(A_val, x_val)
theano/gpuarray/tests/test_linalg.py:        self.run_gpu_solve(A_sym, x_val, 'symmetric')
theano/gpuarray/tests/test_linalg.py:        self.run_gpu_solve(A_orth, x_val)
theano/gpuarray/tests/test_linalg.py:        self.run_gpu_solve(A_val, x_val)
theano/gpuarray/tests/test_linalg.py:        solver = gpu_solve(A, b, 'symmetric')
theano/gpuarray/tests/test_linalg.py:        fn = theano.function([A, b], [solver], mode=mode_with_gpu)
theano/gpuarray/tests/test_linalg.py:        solver = gpu_solve(A, b, trans='T')
theano/gpuarray/tests/test_linalg.py:        fn = theano.function([A, b], [solver], mode=mode_with_gpu)
theano/gpuarray/tests/test_linalg.py:            solve_op = GpuCublasTriangularSolve(lower=lower)
theano/gpuarray/tests/test_linalg.py:            solve_op = GpuCusolverSolve(A_structure="general")
theano/gpuarray/tests/test_linalg.py:class TestGpuCholesky(unittest.TestCase):
theano/gpuarray/tests/test_linalg.py:            self.skipTest('Optional package scikits.cuda.cusolver not available')
theano/gpuarray/tests/test_linalg.py:    def get_gpu_cholesky_func(self, lower=True, inplace=False):
theano/gpuarray/tests/test_linalg.py:        # Helper function to compile function from GPU Cholesky op.
theano/gpuarray/tests/test_linalg.py:        cholesky_op = GpuCholesky(lower=lower, inplace=inplace)
theano/gpuarray/tests/test_linalg.py:                               mode=mode_with_gpu)
theano/gpuarray/tests/test_linalg.py:    def compare_gpu_cholesky_to_np(self, A_val, lower=True, inplace=False):
theano/gpuarray/tests/test_linalg.py:        fn = self.get_gpu_cholesky_func(lower, inplace)
theano/gpuarray/tests/test_linalg.py:    def test_gpu_cholesky_opt(self):
theano/gpuarray/tests/test_linalg.py:        fn = theano.function([A], cholesky(A), mode=mode_with_gpu)
theano/gpuarray/tests/test_linalg.py:        assert any([isinstance(node.op, GpuCholesky)
theano/gpuarray/tests/test_linalg.py:        fn = self.get_gpu_cholesky_func(True, False)
theano/gpuarray/tests/test_linalg.py:            GpuCholesky(lower=True, inplace=False)(A)
theano/gpuarray/tests/test_linalg.py:            GpuCholesky(lower=True, inplace=False)(A)
theano/gpuarray/tests/test_linalg.py:                self.compare_gpu_cholesky_to_np(A_val, lower=lower, inplace=inplace)
theano/gpuarray/tests/test_linalg.py:                self.compare_gpu_cholesky_to_np(A_val, lower=lower, inplace=inplace)
theano/gpuarray/tests/test_linalg.py:        fn = self.get_gpu_cholesky_func(True, False)
theano/gpuarray/tests/test_linalg.py:        fn = self.get_gpu_cholesky_func(True, False)
theano/gpuarray/tests/test_linalg.py:class TestGpuCholesky64(unittest.TestCase):
theano/gpuarray/tests/test_linalg.py:            self.skipTest('Optional package scikits.cuda.cusolver not available')
theano/gpuarray/tests/test_linalg.py:    def get_gpu_cholesky_func(self, lower=True, inplace=False):
theano/gpuarray/tests/test_linalg.py:        # Helper function to compile function from GPU Cholesky op.
theano/gpuarray/tests/test_linalg.py:        cholesky_op = GpuCholesky(lower=lower, inplace=inplace)
theano/gpuarray/tests/test_linalg.py:                               mode=mode_with_gpu)
theano/gpuarray/tests/test_linalg.py:    def compare_gpu_cholesky_to_np(self, A_val, lower=True, inplace=False):
theano/gpuarray/tests/test_linalg.py:        fn = self.get_gpu_cholesky_func(lower, inplace)
theano/gpuarray/tests/test_linalg.py:    def test_gpu_cholesky_opt(self):
theano/gpuarray/tests/test_linalg.py:        fn = theano.function([A], cholesky(A), mode=mode_with_gpu)
theano/gpuarray/tests/test_linalg.py:        assert any([isinstance(node.op, GpuCholesky)
theano/gpuarray/tests/test_linalg.py:        fn = self.get_gpu_cholesky_func(True, False)
theano/gpuarray/tests/test_linalg.py:            GpuCholesky(lower=True, inplace=False)(A)
theano/gpuarray/tests/test_linalg.py:            GpuCholesky(lower=True, inplace=False)(A)
theano/gpuarray/tests/test_linalg.py:                self.compare_gpu_cholesky_to_np(A_val, lower=lower, inplace=inplace)
theano/gpuarray/tests/test_linalg.py:                self.compare_gpu_cholesky_to_np(A_val, lower=lower, inplace=inplace)
theano/gpuarray/tests/test_linalg.py:        fn = self.get_gpu_cholesky_func(True, False)
theano/gpuarray/tests/test_linalg.py:        fn = self.get_gpu_cholesky_func(True, False)
theano/gpuarray/tests/test_linalg.py:        ops_to_gpu = [(MatrixInverse(), GpuMagmaMatrixInverse),
theano/gpuarray/tests/test_linalg.py:                      (SVD(), GpuMagmaSVD),
theano/gpuarray/tests/test_linalg.py:                      (QRFull(mode='reduced'), GpuMagmaQR),
theano/gpuarray/tests/test_linalg.py:                      (QRIncomplete(mode='r'), GpuMagmaQR),
theano/gpuarray/tests/test_linalg.py:                      # (Eigh(), GpuMagmaEigh),
theano/gpuarray/tests/test_linalg.py:                      (Cholesky(), GpuMagmaCholesky)]
theano/gpuarray/tests/test_linalg.py:        for op, gpu_op in ops_to_gpu:
theano/gpuarray/tests/test_linalg.py:            fn = theano.function([A], op(A), mode=mode_with_gpu.excluding('cusolver'))
theano/gpuarray/tests/test_linalg.py:            assert any([isinstance(node.op, gpu_op)
theano/gpuarray/tests/test_linalg.py:    def test_gpu_matrix_inverse(self):
theano/gpuarray/tests/test_linalg.py:        fn = theano.function([A], gpu_matrix_inverse(A), mode=mode_with_gpu)
theano/gpuarray/tests/test_linalg.py:    def test_gpu_matrix_inverse_inplace(self):
theano/gpuarray/tests/test_linalg.py:        A_val_gpu = gpuarray_shared_constructor(test_rng.rand(N, N).astype('float32') * 2 - 1)
theano/gpuarray/tests/test_linalg.py:        A_val_copy = A_val_gpu.get_value()
theano/gpuarray/tests/test_linalg.py:        A_val_gpu_inv = GpuMagmaMatrixInverse()(A_val_gpu)
theano/gpuarray/tests/test_linalg.py:        fn = theano.function([], A_val_gpu_inv, mode=mode_with_gpu, updates=[(A_val_gpu, A_val_gpu_inv)])
theano/gpuarray/tests/test_linalg.py:            isinstance(node.op, GpuMagmaMatrixInverse)
theano/gpuarray/tests/test_linalg.py:        utt.assert_allclose(np.eye(N), np.dot(A_val_gpu.get_value(), A_val_copy), atol=5e-3)
theano/gpuarray/tests/test_linalg.py:    def test_gpu_matrix_inverse_inplace_opt(self):
theano/gpuarray/tests/test_linalg.py:        fn = theano.function([A], matrix_inverse(A), mode=mode_with_gpu)
theano/gpuarray/tests/test_linalg.py:            isinstance(node.op, GpuMagmaMatrixInverse)
theano/gpuarray/tests/test_linalg.py:    def run_gpu_svd(self, A_val, full_matrices=True, compute_uv=True):
theano/gpuarray/tests/test_linalg.py:            [A], gpu_svd(A, full_matrices=full_matrices, compute_uv=compute_uv),
theano/gpuarray/tests/test_linalg.py:            mode=mode_with_gpu)
theano/gpuarray/tests/test_linalg.py:    def test_gpu_svd_wide(self):
theano/gpuarray/tests/test_linalg.py:        U, S, VT = self.run_gpu_svd(A)
theano/gpuarray/tests/test_linalg.py:        U, S, VT = self.run_gpu_svd(A, full_matrices=False)
theano/gpuarray/tests/test_linalg.py:    def test_gpu_svd_tall(self):
theano/gpuarray/tests/test_linalg.py:        U, S, VT = self.run_gpu_svd(A)
theano/gpuarray/tests/test_linalg.py:        U, S, VT = self.run_gpu_svd(A, full_matrices=False)
theano/gpuarray/tests/test_linalg.py:    def test_gpu_singular_values(self):
theano/gpuarray/tests/test_linalg.py:            mode=mode_without_gpu)
theano/gpuarray/tests/test_linalg.py:        f_gpu = theano.function(
theano/gpuarray/tests/test_linalg.py:            [A], gpu_svd(A, compute_uv=False), mode=mode_with_gpu)
theano/gpuarray/tests/test_linalg.py:        utt.assert_allclose(f_cpu(A_val), f_gpu(A_val))
theano/gpuarray/tests/test_linalg.py:        utt.assert_allclose(f_cpu(A_val), f_gpu(A_val))
theano/gpuarray/tests/test_linalg.py:    def run_gpu_cholesky(self, A_val, lower=True):
theano/gpuarray/tests/test_linalg.py:        f = theano.function([A], GpuMagmaCholesky(lower=lower)(A),
theano/gpuarray/tests/test_linalg.py:                            mode=mode_with_gpu.excluding('cusolver'))
theano/gpuarray/tests/test_linalg.py:        # magma cholesky failure due to gpu limited numerical precision
theano/gpuarray/tests/test_linalg.py:        L = self.run_gpu_cholesky(A, lower=lower)
theano/gpuarray/tests/test_linalg.py:    def test_gpu_cholesky(self):
theano/gpuarray/tests/test_linalg.py:    def test_gpu_cholesky_opt(self):
theano/gpuarray/tests/test_linalg.py:        fn = theano.function([A], cholesky(A), mode=mode_with_gpu.excluding('cusolver'))
theano/gpuarray/tests/test_linalg.py:        assert any([isinstance(node.op, GpuMagmaCholesky)
theano/gpuarray/tests/test_linalg.py:    def test_gpu_cholesky_inplace(self):
theano/gpuarray/tests/test_linalg.py:        A_gpu = gpuarray_shared_constructor(A)
theano/gpuarray/tests/test_linalg.py:        A_copy = A_gpu.get_value()
theano/gpuarray/tests/test_linalg.py:        C = GpuMagmaCholesky()(A_gpu)
theano/gpuarray/tests/test_linalg.py:        fn = theano.function([], C, mode=mode_with_gpu, updates=[(A_gpu, C)])
theano/gpuarray/tests/test_linalg.py:            isinstance(node.op, GpuMagmaCholesky)
theano/gpuarray/tests/test_linalg.py:        L = A_gpu.get_value()
theano/gpuarray/tests/test_linalg.py:    def test_gpu_cholesky_inplace_opt(self):
theano/gpuarray/tests/test_linalg.py:        fn = theano.function([A], GpuMagmaCholesky()(A), mode=mode_with_gpu)
theano/gpuarray/tests/test_linalg.py:            isinstance(node.op, GpuMagmaCholesky)
theano/gpuarray/tests/test_linalg.py:    def run_gpu_qr(self, A_val, complete=True):
theano/gpuarray/tests/test_linalg.py:        fn = theano.function([A], gpu_qr(A, complete=complete),
theano/gpuarray/tests/test_linalg.py:                             mode=mode_with_gpu)
theano/gpuarray/tests/test_linalg.py:    def check_gpu_qr(self, M, N, complete=True, rtol=None, atol=None):
theano/gpuarray/tests/test_linalg.py:            Q_gpu, R_gpu = self.run_gpu_qr(A, complete=complete)
theano/gpuarray/tests/test_linalg.py:            R_gpu = self.run_gpu_qr(A, complete=complete)
theano/gpuarray/tests/test_linalg.py:        utt.assert_allclose(R_np, R_gpu, rtol=rtol, atol=atol)
theano/gpuarray/tests/test_linalg.py:            utt.assert_allclose(Q_np, Q_gpu, rtol=rtol, atol=atol)
theano/gpuarray/tests/test_linalg.py:    def test_gpu_qr(self):
theano/gpuarray/tests/test_linalg.py:        self.check_gpu_qr(1000, 500, atol=1e-3)
theano/gpuarray/tests/test_linalg.py:        self.check_gpu_qr(1000, 500, complete=False, atol=1e-3)
theano/gpuarray/tests/test_linalg.py:        self.check_gpu_qr(500, 1000, atol=1e-3)
theano/gpuarray/tests/test_linalg.py:        self.check_gpu_qr(500, 1000, complete=False, atol=1e-3)
theano/gpuarray/tests/test_linalg.py:    def test_gpu_qr_opt(self):
theano/gpuarray/tests/test_linalg.py:        fn = theano.function([A], qr(A), mode=mode_with_gpu)
theano/gpuarray/tests/test_linalg.py:            isinstance(node.op, GpuMagmaQR) and node.op.complete
theano/gpuarray/tests/test_linalg.py:    def test_gpu_qr_incomplete_opt(self):
theano/gpuarray/tests/test_linalg.py:        fn = theano.function([A], qr(A, mode='r'), mode=mode_with_gpu)
theano/gpuarray/tests/test_linalg.py:            isinstance(node.op, GpuMagmaQR) and not node.op.complete
theano/gpuarray/tests/test_linalg.py:    def run_gpu_eigh(self, A_val, UPLO='L', compute_v=True):
theano/gpuarray/tests/test_linalg.py:        fn = theano.function([A], GpuMagmaEigh(UPLO=UPLO, compute_v=compute_v)(A),
theano/gpuarray/tests/test_linalg.py:                             mode=mode_with_gpu)
theano/gpuarray/tests/test_linalg.py:    def check_gpu_eigh(self, N, UPLO='L', compute_v=True, rtol=None, atol=None):
theano/gpuarray/tests/test_linalg.py:            d_gpu, v_gpu = self.run_gpu_eigh(A, UPLO=UPLO, compute_v=compute_v)
theano/gpuarray/tests/test_linalg.py:            d_gpu = self.run_gpu_eigh(A, UPLO=UPLO, compute_v=False)
theano/gpuarray/tests/test_linalg.py:        utt.assert_allclose(d_np, d_gpu, rtol=rtol, atol=atol)
theano/gpuarray/tests/test_linalg.py:                np.eye(N), np.dot(v_gpu, v_gpu.T), rtol=rtol, atol=atol)
theano/gpuarray/tests/test_linalg.py:            np.fill_diagonal(D_m, d_gpu)
theano/gpuarray/tests/test_linalg.py:                A, np.dot(np.dot(v_gpu, D_m), v_gpu.T), rtol=rtol, atol=atol)
theano/gpuarray/tests/test_linalg.py:    def test_gpu_eigh(self):
theano/gpuarray/tests/test_linalg.py:        self.check_gpu_eigh(1000, UPLO='L', atol=1e-3)
theano/gpuarray/tests/test_linalg.py:        self.check_gpu_eigh(1000, UPLO='U', atol=1e-3)
theano/gpuarray/tests/test_linalg.py:        self.check_gpu_eigh(1000, UPLO='L', compute_v=False, atol=1e-3)
theano/gpuarray/tests/test_linalg.py:        self.check_gpu_eigh(1000, UPLO='U', compute_v=False, atol=1e-3)
theano/gpuarray/tests/test_linalg.py:    def test_gpu_eigh_opt(self):
theano/gpuarray/tests/test_linalg.py:        fn = theano.function([A], eigh(A), mode=mode_with_gpu)
theano/gpuarray/tests/test_linalg.py:            isinstance(node.op, GpuMagmaEigh)
theano/gpuarray/tests/test_linalg.py:    yield (lambda: utt.verify_grad(lambda r: gpu_cholesky(r.dot(r.T)),
theano/gpuarray/tests/test_linalg.py:    yield (lambda: utt.verify_grad(lambda r: GpuCholesky(lower=True)(r.dot(r.T)),
theano/gpuarray/tests/test_linalg.py:    yield (lambda: utt.verify_grad(lambda r: GpuCholesky(lower=False)(r.dot(r.T)),
theano/gpuarray/tests/test_linalg.py:    cholesky = GpuCholesky(lower=True)
theano/gpuarray/tests/test_linalg.py:    # cholesky = GpuCholesky(lower=True, on_error='nan')
theano/gpuarray/tests/test_linalg.py:    # chol_f = function([x], grad(gpu_cholesky(x).sum(), [x]))
theano/gpuarray/tests/test_linalg.py:        L = gpu_cholesky(PD)
theano/gpuarray/tests/test_linalg.py:        A = gpu_solve_lower_triangular(L, y)
theano/gpuarray/tests/test_linalg.py:        LB = gpu_cholesky(B)
theano/gpuarray/tests/test_basic_ops.py:from ..type import (GpuArrayType, get_context,
theano/gpuarray/tests/test_basic_ops.py:                    gpuarray_shared_constructor)
theano/gpuarray/tests/test_basic_ops.py:    host_from_gpu, HostFromGpu, GpuFromHost, GpuReshape, GpuToGpu,
theano/gpuarray/tests/test_basic_ops.py:    GpuAlloc, GpuAllocEmpty, GpuContiguous,
theano/gpuarray/tests/test_basic_ops.py:    gpu_join, GpuJoin, GpuSplit, GpuEye, GpuTri,
theano/gpuarray/tests/test_basic_ops.py:    gpu_contiguous)
theano/gpuarray/tests/test_basic_ops.py:from ..elemwise import GpuDimShuffle, GpuElemwise
theano/gpuarray/tests/test_basic_ops.py:from ..subtensor import GpuSubtensor
theano/gpuarray/tests/test_basic_ops.py:from .config import mode_with_gpu, mode_without_gpu, test_ctx_name
theano/gpuarray/tests/test_basic_ops.py:from pygpu import gpuarray
theano/gpuarray/tests/test_basic_ops.py:        mode = mode_with_gpu
theano/gpuarray/tests/test_basic_ops.py:    for c in (gpuarray_shared_constructor, tensor_constructor,
theano/gpuarray/tests/test_basic_ops.py:def rand_gpuarray(*shape, **kwargs):
theano/gpuarray/tests/test_basic_ops.py:    return gpuarray.array(r, dtype=dtype, cls=cls,
theano/gpuarray/tests/test_basic_ops.py:def makeTester(name, op, gpu_op, cases, checks=None, mode_gpu=mode_with_gpu,
theano/gpuarray/tests/test_basic_ops.py:               mode_nogpu=mode_without_gpu, skip=False, eps=1e-10):
theano/gpuarray/tests/test_basic_ops.py:    _gpu_op = gpu_op
theano/gpuarray/tests/test_basic_ops.py:        gpu_op = staticmethod(_gpu_op)
theano/gpuarray/tests/test_basic_ops.py:                           "a node with inputs %s") % (self.gpu_op, testname,
theano/gpuarray/tests/test_basic_ops.py:                f_ref = inplace_func([], node_ref.outputs, mode=mode_nogpu)
theano/gpuarray/tests/test_basic_ops.py:                f_tst = inplace_func([], node_tst.outputs, mode=mode_gpu)
theano/gpuarray/tests/test_basic_ops.py:                           "make a Function") % (self.gpu_op, testname)
theano/gpuarray/tests/test_basic_ops.py:            self.assertFunctionContains1(f_tst, self.gpu_op)
theano/gpuarray/tests/test_basic_ops.py:                               "Function") % (self.gpu_op, testname)
theano/gpuarray/tests/test_basic_ops.py:                                   (self.gpu_op, testname, type(exc),
theano/gpuarray/tests/test_basic_ops.py:def test_transfer_cpu_gpu():
theano/gpuarray/tests/test_basic_ops.py:    g = GpuArrayType(dtype='float32', broadcastable=(False, False))('g')
theano/gpuarray/tests/test_basic_ops.py:    gv = gpuarray.array(av, context=get_context(test_ctx_name))
theano/gpuarray/tests/test_basic_ops.py:    f = theano.function([a], GpuFromHost(test_ctx_name)(a))
theano/gpuarray/tests/test_basic_ops.py:    assert GpuArrayType.values_eq(fv, gv)
theano/gpuarray/tests/test_basic_ops.py:    f = theano.function([g], host_from_gpu(g))
theano/gpuarray/tests/test_basic_ops.py:def test_transfer_gpu_gpu():
theano/gpuarray/tests/test_basic_ops.py:    g = GpuArrayType(dtype='float32', broadcastable=(False, False),
theano/gpuarray/tests/test_basic_ops.py:    gv = gpuarray.array(av, context=get_context(test_ctx_name))
theano/gpuarray/tests/test_basic_ops.py:    mode = mode_with_gpu.excluding('cut_gpua_host_transfers', 'local_cut_gpua_host_gpua')
theano/gpuarray/tests/test_basic_ops.py:    f = theano.function([g], GpuToGpu(test_ctx_name)(g), mode=mode)
theano/gpuarray/tests/test_basic_ops.py:    assert isinstance(topo[0].op, GpuToGpu)
theano/gpuarray/tests/test_basic_ops.py:    assert GpuArrayType.values_eq(fv, gv)
theano/gpuarray/tests/test_basic_ops.py:    # libgpuarray has a much more comprehensive suit of tests to
theano/gpuarray/tests/test_basic_ops.py:    g = GpuArrayType(dtype='float32', broadcastable=(False, False))('g')
theano/gpuarray/tests/test_basic_ops.py:    gv = gpuarray.array(av, context=get_context(test_ctx_name))
theano/gpuarray/tests/test_basic_ops.py:    f = theano.function([a], GpuFromHost(test_ctx_name)(a))
theano/gpuarray/tests/test_basic_ops.py:    assert GpuArrayType.values_eq(fv, gv)
theano/gpuarray/tests/test_basic_ops.py:    f = theano.function([g], host_from_gpu(g))
theano/gpuarray/tests/test_basic_ops.py:def gpu_alloc_expected(x, *shp):
theano/gpuarray/tests/test_basic_ops.py:    g = gpuarray.empty(shp, dtype=x.dtype, context=get_context(test_ctx_name))
theano/gpuarray/tests/test_basic_ops.py:GpuAllocTester = makeTester(
theano/gpuarray/tests/test_basic_ops.py:    name="GpuAllocTester",
theano/gpuarray/tests/test_basic_ops.py:    # The +1 is there to allow the lift to the GPU.
theano/gpuarray/tests/test_basic_ops.py:    gpu_op=GpuAlloc(test_ctx_name),
theano/gpuarray/tests/test_basic_ops.py:    mode = mode_with_gpu
theano/gpuarray/tests/test_basic_ops.py:    shared = staticmethod(gpuarray_shared_constructor)
theano/gpuarray/tests/test_basic_ops.py:    allocs = [GpuAlloc(test_ctx_name), GpuAlloc(test_ctx_name), T.Alloc()]
theano/gpuarray/tests/test_basic_ops.py:        f = theano.function([], GpuAllocEmpty(dt, context_name=test_ctx_name)(2, 3))
theano/gpuarray/tests/test_basic_ops.py:    f = theano.function([], [GpuAllocEmpty('uint64', test_ctx_name)(3, 2),
theano/gpuarray/tests/test_basic_ops.py:                             GpuAllocEmpty('uint64', test_ctx_name)(3, 2)])
theano/gpuarray/tests/test_basic_ops.py:                if isinstance(node.op, GpuAllocEmpty)]) == 1
theano/gpuarray/tests/test_basic_ops.py:    x = GpuArrayType(dtype='float32', broadcastable=[False, False, False])()
theano/gpuarray/tests/test_basic_ops.py:    v = gpuarray.zeros((3, 4, 5), dtype='float32', context=get_context(test_ctx_name))
theano/gpuarray/tests/test_basic_ops.py:    mode = mode_with_gpu.excluding("local_shape_to_shape_i")
theano/gpuarray/tests/test_basic_ops.py:def test_gpu_contiguous():
theano/gpuarray/tests/test_basic_ops.py:    f = theano.function([a, i], gpu_contiguous(a.reshape((5, 4))[::i]),
theano/gpuarray/tests/test_basic_ops.py:                        mode=mode_with_gpu)
theano/gpuarray/tests/test_basic_ops.py:    assert any([isinstance(node.op, GpuSubtensor) for node in topo])
theano/gpuarray/tests/test_basic_ops.py:    assert any([isinstance(node.op, GpuContiguous) for node in topo])
theano/gpuarray/tests/test_basic_ops.py:            shared=gpuarray_shared_constructor,
theano/gpuarray/tests/test_basic_ops.py:            op=GpuReshape,
theano/gpuarray/tests/test_basic_ops.py:            mode=mode_with_gpu,
theano/gpuarray/tests/test_basic_ops.py:            ignore_topo=(HostFromGpu, GpuFromHost,
theano/gpuarray/tests/test_basic_ops.py:                         GpuDimShuffle, GpuElemwise,
theano/gpuarray/tests/test_basic_ops.py:        assert self.op == GpuReshape
theano/gpuarray/tests/test_basic_ops.py:        self.mode = mode_with_gpu
theano/gpuarray/tests/test_basic_ops.py:        self.shared = gpuarray_shared_constructor
theano/gpuarray/tests/test_basic_ops.py:        self.mode = mode_with_gpu.excluding('constant_folding')
theano/gpuarray/tests/test_basic_ops.py:        self.join_op = GpuJoin()
theano/gpuarray/tests/test_basic_ops.py:        self.split_op_class = GpuSplit
theano/gpuarray/tests/test_basic_ops.py:        # Use join instead of MakeVector since there is no MakeVector on GPU
theano/gpuarray/tests/test_basic_ops.py:        self.make_vector_op = GpuJoin()
theano/gpuarray/tests/test_basic_ops.py:            return gpuarray_shared_constructor(x, target=test_ctx_name,
theano/gpuarray/tests/test_basic_ops.py:    def test_gpusplit_opt(self):
theano/gpuarray/tests/test_basic_ops.py:        # Test that we move the node to the GPU
theano/gpuarray/tests/test_basic_ops.py:def test_gpujoin_gpualloc():
theano/gpuarray/tests/test_basic_ops.py:                        mode=mode_without_gpu)
theano/gpuarray/tests/test_basic_ops.py:    f_gpu = theano.function([a, b], T.join(0, T.zeros_like(a), T.ones_like(b)),
theano/gpuarray/tests/test_basic_ops.py:                            mode=mode_with_gpu)
theano/gpuarray/tests/test_basic_ops.py:    f_gpu2 = theano.function([a, b], T.join(0, T.zeros_like(a),
theano/gpuarray/tests/test_basic_ops.py:                             mode=mode_with_gpu)
theano/gpuarray/tests/test_basic_ops.py:    assert sum([isinstance(node.op, GpuAlloc)
theano/gpuarray/tests/test_basic_ops.py:                for node in f_gpu.maker.fgraph.toposort()]) == 2
theano/gpuarray/tests/test_basic_ops.py:    assert sum([node.op == gpu_join
theano/gpuarray/tests/test_basic_ops.py:                for node in f_gpu.maker.fgraph.toposort()]) == 1
theano/gpuarray/tests/test_basic_ops.py:    assert sum([isinstance(node.op, GpuAlloc)
theano/gpuarray/tests/test_basic_ops.py:                for node in f_gpu2.maker.fgraph.toposort()]) == 2
theano/gpuarray/tests/test_basic_ops.py:    assert sum([node.op == gpu_join
theano/gpuarray/tests/test_basic_ops.py:                for node in f_gpu2.maker.fgraph.toposort()]) == 1
theano/gpuarray/tests/test_basic_ops.py:    assert np.allclose(f(a_val, b_val), f_gpu2(a_val, b_val))
theano/gpuarray/tests/test_basic_ops.py:def test_gpueye():
theano/gpuarray/tests/test_basic_ops.py:                            mode=mode_with_gpu)
theano/gpuarray/tests/test_basic_ops.py:        assert any([isinstance(node.op, GpuEye)
theano/gpuarray/tests/test_basic_ops.py:def test_hostfromgpu_shape_i():
theano/gpuarray/tests/test_basic_ops.py:    # Test that the shape is lifted over hostfromgpu
theano/gpuarray/tests/test_basic_ops.py:    m = mode_with_gpu.including('local_dot_to_dot22',
theano/gpuarray/tests/test_basic_ops.py:    ca = theano.gpuarray.type.GpuArrayType('float32', (False, False))()
theano/gpuarray/tests/test_basic_ops.py:    cv = gpuarray.asarray(np.random.rand(5, 4),
theano/gpuarray/tests/test_basic_ops.py:    f = theano.function([a], GpuFromHost(test_ctx_name)(a), mode=m)
theano/gpuarray/tests/test_basic_ops.py:    assert any(isinstance(x.op, GpuFromHost)
theano/gpuarray/tests/test_basic_ops.py:    f = theano.function([a], GpuFromHost(test_ctx_name)(a).shape, mode=m)
theano/gpuarray/tests/test_basic_ops.py:    f = theano.function([ca], host_from_gpu(ca), mode=m)
theano/gpuarray/tests/test_basic_ops.py:    assert host_from_gpu in [x.op
theano/gpuarray/tests/test_basic_ops.py:    f = theano.function([ca], host_from_gpu(ca).shape, mode=m)
theano/gpuarray/tests/test_basic_ops.py:def test_Gpujoin_inplace():
theano/gpuarray/tests/test_basic_ops.py:    # Test Gpujoin to work inplace.
theano/gpuarray/tests/test_basic_ops.py:    # Gpujoin function but all except one of them are empty. In this case
theano/gpuarray/tests/test_basic_ops.py:    # Gpujoin should work inplace and the output should be the view of the
theano/gpuarray/tests/test_basic_ops.py:    x = gpuarray_shared_constructor(data, borrow=True)
theano/gpuarray/tests/test_basic_ops.py:    join = GpuJoin(view=0)
theano/gpuarray/tests/test_basic_ops.py:    if not isinstance(mode_with_gpu, theano.compile.DebugMode):
theano/gpuarray/tests/test_basic_ops.py:def test_gpu_tril_triu():
theano/gpuarray/tests/test_basic_ops.py:                            mode=mode_with_gpu)
theano/gpuarray/tests/test_basic_ops.py:        assert any([isinstance(node.op, GpuTri)
theano/gpuarray/tests/test_basic_ops.py:                            mode=mode_with_gpu)
theano/gpuarray/tests/test_basic_ops.py:        assert any([isinstance(node.op, GpuTri)
theano/gpuarray/tests/test_basic_ops.py:def test_gputri():
theano/gpuarray/tests/test_basic_ops.py:                            mode=mode_with_gpu)
theano/gpuarray/tests/test_basic_ops.py:        assert any([isinstance(node.op, GpuTri)
theano/gpuarray/tests/test_cgpukernelbase.py:from ..basic_ops import CGpuKernelBase
theano/gpuarray/tests/test_cgpukernelbase.py:from ..type import GpuArrayType, get_context, gpu_context_type
theano/gpuarray/tests/test_cgpukernelbase.py:# This is an implementation to test that CGpuKernelBase works and also
theano/gpuarray/tests/test_cgpukernelbase.py:class GpuEye(CGpuKernelBase, Op):
theano/gpuarray/tests/test_cgpukernelbase.py:    Eye for GPU.
theano/gpuarray/tests/test_cgpukernelbase.py:    params_type = ParamsType(typecode=int_t, context=gpu_context_type)
theano/gpuarray/tests/test_cgpukernelbase.py:        CGpuKernelBase.__init__(self, ['c_code/tstgpueye.c'],
theano/gpuarray/tests/test_cgpukernelbase.py:                                'APPLY_SPECIFIC(tstgpueye)')
theano/gpuarray/tests/test_cgpukernelbase.py:        from pygpu.gpuarray import dtype_to_typecode
theano/gpuarray/tests/test_cgpukernelbase.py:        return ['<gpuarray/types.h>', '<gpuarray/kernel.h>']
theano/gpuarray/tests/test_cgpukernelbase.py:        otype = GpuArrayType(dtype=self.dtype,
theano/gpuarray/tests/test_cgpukernelbase.py:def test_cgpukernelbase():
theano/gpuarray/tests/test_cgpukernelbase.py:    # initialized when reloading the GpuEye object from cache.
theano/gpuarray/tests/test_cgpukernelbase.py:    from .config import mode_with_gpu, test_ctx_name
theano/gpuarray/tests/test_cgpukernelbase.py:    op = GpuEye(dtype='int32', context_name=test_ctx_name)
theano/gpuarray/tests/test_cgpukernelbase.py:    f = theano.function([], op(4, 5), mode=mode_with_gpu)
theano/gpuarray/tests/test_misc.py:# Test that normaly could be outside gpuarray, to have all gpuarray
theano/gpuarray/tests/test_misc.py:from .config import mode_with_gpu
theano/gpuarray/tests/test_misc.py:                            optimizer=mode_with_gpu.optimizer)
theano/gpuarray/tests/c_code/tstgpueye.c:int APPLY_SPECIFIC(tstgpueye)(PyArrayObject *n, PyArrayObject *m,
theano/gpuarray/tests/c_code/tstgpueye.c:                              PyGpuArrayObject **z, PARAMS_TYPE* params) {
theano/gpuarray/tests/c_code/tstgpueye.c:  *z = pygpu_zeros(2, dims,
theano/gpuarray/tests/c_code/tstgpueye.c:                 "gpuarray error: kEye: %s. n%lu, m=%lu.",
theano/gpuarray/tests/c_code/tstgpueye.c:                 GpuKernel_error(&k_eye, err),
theano/gpuarray/tests/test_dnn.py:from ..basic_ops import GpuAllocEmpty
theano/gpuarray/tests/test_dnn.py:from ..type import gpuarray_shared_constructor, GpuArrayType
theano/gpuarray/tests/test_dnn.py:from .config import mode_with_gpu, mode_without_gpu, test_ctx_name, ref_cast
theano/gpuarray/tests/test_dnn.py:    import pygpu
theano/gpuarray/tests/test_dnn.py:mode_with_gpu = mode_with_gpu.including()
theano/gpuarray/tests/test_dnn.py:# Globally disabled for mode_without_gpu
theano/gpuarray/tests/test_dnn.py:mode_with_gpu.check_py_code = False
theano/gpuarray/tests/test_dnn.py:    desc1 = dnn.GpuDnnConvDesc(border_mode='valid', subsample=(2, 2), dilation=(1, 1),
theano/gpuarray/tests/test_dnn.py:    desc2 = dnn.GpuDnnConvDesc(border_mode='full', subsample=(1, 1), dilation=(1, 1),
theano/gpuarray/tests/test_dnn.py:                        mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    desc = dnn.GpuDnnConvDesc(border_mode='valid')(kern.shape)
theano/gpuarray/tests/test_dnn.py:    f = theano.function([img, kern], [o1, o2], mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    assert len([n for n in topo if isinstance(n.op, dnn.GpuDnnConv)]) == 1
theano/gpuarray/tests/test_dnn.py:    o1 = dnn.GpuDnnConvGradW()(img, kern, out, desc)
theano/gpuarray/tests/test_dnn.py:    o2 = dnn.GpuDnnConvGradW()(img, kern, out, desc)
theano/gpuarray/tests/test_dnn.py:    f = theano.function([img, kern, out], [o1, o2], mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    assert len([n for n in topo if isinstance(n.op, dnn.GpuDnnConvGradW)]) == 1
theano/gpuarray/tests/test_dnn.py:    o1 = dnn.GpuDnnConvGradI()(img, kern, out, desc)
theano/gpuarray/tests/test_dnn.py:    o2 = dnn.GpuDnnConvGradI()(img, kern, out, desc)
theano/gpuarray/tests/test_dnn.py:    f = theano.function([img, kern, out], [o1, o2], mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    assert len([n for n in topo if isinstance(n.op, dnn.GpuDnnConvGradI)]) == 1
theano/gpuarray/tests/test_dnn.py:    # GpuAllocEmpty get merged together.
theano/gpuarray/tests/test_dnn.py:    desc1 = dnn.GpuDnnConvDesc(border_mode='valid', conv_mode='conv')(
theano/gpuarray/tests/test_dnn.py:    desc2 = dnn.GpuDnnConvDesc(
theano/gpuarray/tests/test_dnn.py:    f = theano.function([img, kern], [o1, o2], mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    convs = [n for n in topo if isinstance(n.op, dnn.GpuDnnConv)]
theano/gpuarray/tests/test_dnn.py:    assert len([n for n in topo if isinstance(n.op, GpuAllocEmpty)]) == 2
theano/gpuarray/tests/test_dnn.py:    out = GpuAllocEmpty(kern.dtype, test_ctx_name)(*kern.shape)
theano/gpuarray/tests/test_dnn.py:    o1 = dnn.GpuDnnConvGradW()(img, kern, out, desc1)
theano/gpuarray/tests/test_dnn.py:    o2 = dnn.GpuDnnConvGradW()(img, kern, out, desc2)
theano/gpuarray/tests/test_dnn.py:    f = theano.function([img, kern], [o1, o2], mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    convs = [n for n in topo if isinstance(n.op, dnn.GpuDnnConvGradW)]
theano/gpuarray/tests/test_dnn.py:    assert len([n for n in topo if isinstance(n.op, GpuAllocEmpty)]) == 2
theano/gpuarray/tests/test_dnn.py:    out = GpuAllocEmpty(img.dtype, test_ctx_name)(*img.shape)
theano/gpuarray/tests/test_dnn.py:    o1 = dnn.GpuDnnConvGradI()(img, kern, out, desc1)
theano/gpuarray/tests/test_dnn.py:    o2 = dnn.GpuDnnConvGradI()(img, kern, out, desc2)
theano/gpuarray/tests/test_dnn.py:    f = theano.function([img, kern], [o1, o2], mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    convs = [n for n in topo if isinstance(n.op, dnn.GpuDnnConvGradI)]
theano/gpuarray/tests/test_dnn.py:    assert len([n for n in topo if isinstance(n.op, GpuAllocEmpty)]) == 2
theano/gpuarray/tests/test_dnn.py:                mode_without_gpu2 = mode_without_gpu.including()
theano/gpuarray/tests/test_dnn.py:                mode_without_gpu2.check_isfinite = False
theano/gpuarray/tests/test_dnn.py:                # GPU implementation
theano/gpuarray/tests/test_dnn.py:                f_gpu = theano.function([x], out, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                assert any([isinstance(node.op, dnn.GpuDnnPool)
theano/gpuarray/tests/test_dnn.py:                            for node in f_gpu.maker.fgraph.apply_nodes])
theano/gpuarray/tests/test_dnn.py:                f_cpu = theano.function([x], out, mode=mode_without_gpu2)
theano/gpuarray/tests/test_dnn.py:                assert not any([isinstance(node.op, dnn.GpuDnnPool)
theano/gpuarray/tests/test_dnn.py:                    b = f_gpu(data).__array__()
theano/gpuarray/tests/test_dnn.py:            # This tests the CPU grad + opt + GPU implementation
theano/gpuarray/tests/test_dnn.py:            utt.verify_grad(fn, [data], mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                                 mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:            assert any([isinstance(node.op, dnn.GpuDnnPoolGrad)
theano/gpuarray/tests/test_dnn.py:            # Test the GPU grad + GPU implementation
theano/gpuarray/tests/test_dnn.py:            utt.verify_grad(fn, [data], mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                                 mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:            assert any([isinstance(node.op, dnn.GpuDnnPoolGrad)
theano/gpuarray/tests/test_dnn.py:            fn, [data], mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    mode_without_gpu2 = mode_without_gpu.including()
theano/gpuarray/tests/test_dnn.py:    mode_without_gpu2.check_isfinite = False
theano/gpuarray/tests/test_dnn.py:    # GPU implementation
theano/gpuarray/tests/test_dnn.py:    f_gpu = theano.function([x], fn(x), mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    assert any([isinstance(node.op, dnn.GpuDnnPool)
theano/gpuarray/tests/test_dnn.py:                for node in f_gpu.maker.fgraph.apply_nodes])
theano/gpuarray/tests/test_dnn.py:    f_cpu = theano.function([x], out_cpu, mode=mode_without_gpu2)
theano/gpuarray/tests/test_dnn.py:    assert not any([isinstance(node.op, dnn.GpuDnnPool)
theano/gpuarray/tests/test_dnn.py:        a = f_gpu(data).__array__()
theano/gpuarray/tests/test_dnn.py:    mode_without_gpu_ref = theano.compile.mode.get_mode(
theano/gpuarray/tests/test_dnn.py:        'FAST_RUN').excluding('gpuarray')
theano/gpuarray/tests/test_dnn.py:                # GPU implementation
theano/gpuarray/tests/test_dnn.py:                f_gpu = theano.function([x], out, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                assert any([isinstance(node.op, dnn.GpuDnnPool)
theano/gpuarray/tests/test_dnn.py:                            for node in f_gpu.maker.fgraph.apply_nodes])
theano/gpuarray/tests/test_dnn.py:                f_cpu = theano.function([x], out, mode=mode_without_gpu_ref)
theano/gpuarray/tests/test_dnn.py:                assert not any([isinstance(node.op, dnn.GpuDnnPool)
theano/gpuarray/tests/test_dnn.py:                    b = f_gpu(data).__array__()
theano/gpuarray/tests/test_dnn.py:            # Test the GPU grad + GPU implementation
theano/gpuarray/tests/test_dnn.py:            utt.verify_grad(fn, [data], mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                                 mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:            assert any([isinstance(node.op, dnn.GpuDnnPoolGrad)
theano/gpuarray/tests/test_dnn.py:        mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    assert any([isinstance(n.op, dnn.GpuDnnPool)
theano/gpuarray/tests/test_dnn.py:        mode=mode_with_gpu.including("cudnn"))
theano/gpuarray/tests/test_dnn.py:    assert any([isinstance(n.op, dnn.GpuDnnPoolGrad)
theano/gpuarray/tests/test_dnn.py:        mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    assert any([isinstance(n.op, dnn.GpuDnnPool)
theano/gpuarray/tests/test_dnn.py:        mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    assert any([isinstance(n.op, dnn.GpuDnnPool)
theano/gpuarray/tests/test_dnn.py:        mode=mode_with_gpu.including("cudnn"))
theano/gpuarray/tests/test_dnn.py:    assert any([isinstance(n.op, dnn.GpuDnnPoolGrad)
theano/gpuarray/tests/test_dnn.py:    # is correctly reshaped to run on the GPU
theano/gpuarray/tests/test_dnn.py:            input = gpuarray_shared_constructor(data)
theano/gpuarray/tests/test_dnn.py:                # run on GPU
theano/gpuarray/tests/test_dnn.py:                fg = theano.function([], out, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                assert any([isinstance(node.op, dnn.GpuDnnPool)
theano/gpuarray/tests/test_dnn.py:                assert any([isinstance(node.op, dnn.GpuDnnPoolGrad)
theano/gpuarray/tests/test_dnn.py:                res_gpu = fg()
theano/gpuarray/tests/test_dnn.py:                fc = theano.function([], out, mode=mode_without_gpu)
theano/gpuarray/tests/test_dnn.py:                utt.assert_allclose(res_gpu[0], res_cpu[0])
theano/gpuarray/tests/test_dnn.py:                utt.assert_allclose(res_gpu[1], res_cpu[1])
theano/gpuarray/tests/test_dnn.py:    f = theano.function([img], o, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    f = theano.function([img], g, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:            mode=mode_with_gpu.including("cudnn"))
theano/gpuarray/tests/test_dnn.py:        assert any([isinstance(n.op, dnn.GpuDnnPool)
theano/gpuarray/tests/test_dnn.py:        self.mode = mode_with_gpu
theano/gpuarray/tests/test_dnn.py:            [dnn.GpuDnnSoftmax('accurate', 'channel')(t)],
theano/gpuarray/tests/test_dnn.py:            dnn.GpuDnnSoftmax
theano/gpuarray/tests/test_dnn.py:                    dnn.GpuDnnSoftmax(
theano/gpuarray/tests/test_dnn.py:            dnn.GpuDnnSoftmaxGrad
theano/gpuarray/tests/test_dnn.py:                    dnn.GpuDnnConv.get_out_shape(img_val.shape, kern_vals.shape,
theano/gpuarray/tests/test_dnn.py:                desc = dnn.GpuDnnConvDesc(
theano/gpuarray/tests/test_dnn.py:                conv = dnn.GpuDnnConv(algo=algo)(img, kerns, out, desc)
theano/gpuarray/tests/test_dnn.py:                    dnn.GpuDnnConv
theano/gpuarray/tests/test_dnn.py:                desc = dnn.GpuDnnConvDesc(
theano/gpuarray/tests/test_dnn.py:                conv_grad_w = dnn.GpuDnnConvGradW()(
theano/gpuarray/tests/test_dnn.py:                    dnn.GpuDnnConvGradW
theano/gpuarray/tests/test_dnn.py:            desc = dnn.GpuDnnConvDesc(
theano/gpuarray/tests/test_dnn.py:            conv_grad_i = dnn.GpuDnnConvGradI()(
theano/gpuarray/tests/test_dnn.py:                dnn.GpuDnnConvGradI
theano/gpuarray/tests/test_dnn.py:                [dnn.GpuDnnPool(mode=params[2])(img, params[0], params[1], (0, 0))],
theano/gpuarray/tests/test_dnn.py:                dnn.GpuDnnPool
theano/gpuarray/tests/test_dnn.py:                [dnn.GpuDnnPool(mode=params[2])(img, params[0], params[1], (0, 0, 0))],
theano/gpuarray/tests/test_dnn.py:                dnn.GpuDnnPool
theano/gpuarray/tests/test_dnn.py:            pool_grad = dnn.GpuDnnPoolGrad(mode=params[2])(
theano/gpuarray/tests/test_dnn.py:                dnn.GpuDnnPoolGrad
theano/gpuarray/tests/test_dnn.py:            pool_grad = dnn.GpuDnnPoolGrad(mode=params[2])(
theano/gpuarray/tests/test_dnn.py:                dnn.GpuDnnPoolGrad
theano/gpuarray/tests/test_dnn.py:    f1 = theano.function([img, kern, out], [fr, wr, ir], mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                      dnn.GpuDnnConv)
theano/gpuarray/tests/test_dnn.py:                      dnn.GpuDnnConvGradW)
theano/gpuarray/tests/test_dnn.py:                      dnn.GpuDnnConvGradI)
theano/gpuarray/tests/test_dnn.py:    mode = mode_with_gpu
theano/gpuarray/tests/test_dnn.py:                          dnn.GpuDnnConv)
theano/gpuarray/tests/test_dnn.py:                          dnn.GpuDnnConvGradW)
theano/gpuarray/tests/test_dnn.py:                          dnn.GpuDnnConvGradI)
theano/gpuarray/tests/test_dnn.py:        desc = dnn.GpuDnnConvDesc(border_mode='valid', subsample=(1, 1), dilation=(1, 1),
theano/gpuarray/tests/test_dnn.py:        return dnn.GpuDnnConv()(img, kern, out, desc, alpha=0.5, beta=0.75)
theano/gpuarray/tests/test_dnn.py:        desc = dnn.GpuDnnConvDesc(border_mode='valid', subsample=(1, 1), dilation=(1, 1),
theano/gpuarray/tests/test_dnn.py:        return dnn.GpuDnnConvGradI()(kern, out, img, desc, alpha=-1.0,
theano/gpuarray/tests/test_dnn.py:        desc = dnn.GpuDnnConvDesc(border_mode='valid', subsample=(1, 1), dilation=(1, 1),
theano/gpuarray/tests/test_dnn.py:        return dnn.GpuDnnConvGradW()(img, out, kern, desc, alpha=0.75,
theano/gpuarray/tests/test_dnn.py:    utt.verify_grad(dconv, [img_val, kern_val, out_val], eps=1e-3, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    utt.verify_grad(dconvi, [img_val, kern_val, out_val], eps=1e-3, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    utt.verify_grad(dconvw, [img_val, kern_val, out_val], eps=1e-3, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    f = theano.function([], [conv, sub_conv_top, sub_conv_bottom], mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:        f = theano.function([], conv, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:        f = theano.function([], [grad_i, grad_w], mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    gpu_op = dnn.GpuDnnSoftmax
theano/gpuarray/tests/test_dnn.py:    gpu_grad_op = dnn.GpuDnnSoftmaxGrad
theano/gpuarray/tests/test_dnn.py:    mode = mode_with_gpu
theano/gpuarray/tests/test_dnn.py:        x_gpu = T.tensor4('x_gpu')
theano/gpuarray/tests/test_dnn.py:        f_gpu = dnn.GpuDnnSoftmax('accurate', 'channel')(x_gpu)
theano/gpuarray/tests/test_dnn.py:        f_gpu = theano.function([x_gpu], f_gpu, mode=self.mode)
theano/gpuarray/tests/test_dnn.py:        assert f_gpu(data).shape == dims
theano/gpuarray/tests/test_dnn.py:        dy_gpu = T.tensor4('dy_gpu')
theano/gpuarray/tests/test_dnn.py:        sm_gpu = T.tensor4('sm_gpu')
theano/gpuarray/tests/test_dnn.py:        f_grad_gpu = dnn.GpuDnnSoftmaxGrad('accurate', 'channel')(dy_gpu, sm_gpu)
theano/gpuarray/tests/test_dnn.py:        f_grad_gpu = theano.function([dy_gpu, sm_gpu], f_grad_gpu, mode=self.mode)
theano/gpuarray/tests/test_dnn.py:        assert f_grad_gpu(data, data).shape == dims
theano/gpuarray/tests/test_dnn.py:        x_gpu = T.tensor4('x_gpu', 'float16')
theano/gpuarray/tests/test_dnn.py:        f_gpu = dnn.GpuDnnSoftmax(
theano/gpuarray/tests/test_dnn.py:        def cmp(n, m, f, f_gpu):
theano/gpuarray/tests/test_dnn.py:            gout = np.asarray(f_gpu(gdata))[:, :, 0, 0]
theano/gpuarray/tests/test_dnn.py:        self._test_softmax(x, x_gpu, f_z, f_gpu, cmp)
theano/gpuarray/tests/test_dnn.py:        def cmp(n, m, f, f_gpu):
theano/gpuarray/tests/test_dnn.py:            gout = np.asarray(f_gpu(gdata))[:, :, 0, 0]
theano/gpuarray/tests/test_dnn.py:        x_gpu = T.tensor4('x_gpu')
theano/gpuarray/tests/test_dnn.py:        f_gpu = dnn.GpuDnnSoftmax(
theano/gpuarray/tests/test_dnn.py:        T.verify_grad(f_gpu, [gdata], rng=np.random,
theano/gpuarray/tests/test_dnn.py:                      mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:        # Verify that the CPU and GPU implementations return the same results
theano/gpuarray/tests/test_dnn.py:            x_gpu,
theano/gpuarray/tests/test_dnn.py:            f_gpu,
theano/gpuarray/tests/test_dnn.py:        # Verify that the SoftmaxGrad -> Gpu[Dnn]SoftmaxGrad
theano/gpuarray/tests/test_dnn.py:            mode=mode_with_gpu
theano/gpuarray/tests/test_dnn.py:                        self.gpu_grad_op)
theano/gpuarray/tests/test_dnn.py:        # Verify that the SoftmaxGrad -> Gpu[Dnn]SoftmaxGrad
theano/gpuarray/tests/test_dnn.py:        mode_wo_cudnn = mode_with_gpu.excluding("cudnn")
theano/gpuarray/tests/test_dnn.py:                        self.gpu_grad_op)
theano/gpuarray/tests/test_dnn.py:        # Verify that the SoftmaxGrad -> GpuDnnSoftmaxGrad do not
theano/gpuarray/tests/test_dnn.py:        f = theano.function([y], o, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                        self.gpu_grad_op)
theano/gpuarray/tests/test_dnn.py:        softmax_out = dnn.GpuDnnSoftmax('accurate', 'channel')(x)
theano/gpuarray/tests/test_dnn.py:        f = theano.function([x], log_out, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                             isinstance(n.op, dnn.GpuDnnSoftmax)]
theano/gpuarray/tests/test_dnn.py:        f = theano.function([x], log_softmax_out, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                             isinstance(n.op, dnn.GpuDnnSoftmax)]
theano/gpuarray/tests/test_dnn.py:        f = theano.function([x], log_softmax_out, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                             isinstance(n.op, dnn.GpuDnnSoftmax)]
theano/gpuarray/tests/test_dnn.py:    f = theano.function([inp], res, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    assert any(isinstance(n.op, dnn.GpuDnnReduction)
theano/gpuarray/tests/test_dnn.py:        f = theano.function([M], out, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:        assert any(isinstance(node.op, dnn.GpuDnnReduction) and node.op.red_op == 'norm2'
theano/gpuarray/tests/test_dnn.py:        f = theano.function([M], out, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:        assert any(isinstance(node.op, dnn.GpuDnnReduction) and node.op.red_op == 'norm1'
theano/gpuarray/tests/test_dnn.py:        f = theano.function([M], out, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:        assert any(isinstance(node.op, dnn.GpuDnnReduction) and node.op.red_op == 'absmax'
theano/gpuarray/tests/test_dnn.py:            cpu_f = theano.function([x], [sum, sum_squares, sum_abs, absmax], mode=mode_without_gpu)
theano/gpuarray/tests/test_dnn.py:            f1 = theano.function([x], sum, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:            f2 = theano.function([x], sum_squares, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:            f3 = theano.function([x], sum_abs, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:            f4 = theano.function([x], absmax, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                assert any(isinstance(node.op, dnn.GpuDnnReduction) and node.op.red_op == red_op
theano/gpuarray/tests/test_dnn.py:    inp = GpuArrayType('float32', (False,) * len(shp),
theano/gpuarray/tests/test_dnn.py:    f = theano.function([inp], res, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    assert any(isinstance(n.op, dnn.GpuDnnReduction)
theano/gpuarray/tests/test_dnn.py:    gdata = pygpu.array(data, context=inp.type.context)
theano/gpuarray/tests/test_dnn.py:    f = theano.function([inp], res, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    assert any(isinstance(n.op, dnn.GpuDnnReduction)
theano/gpuarray/tests/test_dnn.py:            out_gpu, x_mean_gpu, x_invstd_gpu, \
theano/gpuarray/tests/test_dnn.py:                out_running_mean_gpu, out_running_var_gpu = \
theano/gpuarray/tests/test_dnn.py:            grads_gpu = T.grad(None, wrt=[x, scale, bias], known_grads={out_gpu: dy})
theano/gpuarray/tests/test_dnn.py:            f_gpu = theano.function([x, scale, bias, running_mean, running_var, dy],
theano/gpuarray/tests/test_dnn.py:                                    [out_gpu, x_mean_gpu, x_invstd_gpu,
theano/gpuarray/tests/test_dnn.py:                                     out_running_mean_gpu, out_running_var_gpu] + grads_gpu,
theano/gpuarray/tests/test_dnn.py:                                    mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                                         mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                                    mode=mode_without_gpu)
theano/gpuarray/tests/test_dnn.py:            assert any([isinstance(n.op, dnn.GpuDnnBatchNorm) for n
theano/gpuarray/tests/test_dnn.py:            assert any([isinstance(n.op, dnn.GpuDnnBatchNormGrad) for n
theano/gpuarray/tests/test_dnn.py:                outputs_gpu = f_gpu(X, Scale, Bias, Running_mean, Running_var, Dy)
theano/gpuarray/tests/test_dnn.py:                utt.assert_allclose(outputs_gpu[0], outputs_ref[0])  # out
theano/gpuarray/tests/test_dnn.py:                utt.assert_allclose(outputs_gpu[1], outputs_ref[1])  # mean
theano/gpuarray/tests/test_dnn.py:                utt.assert_allclose(outputs_gpu[2], outputs_ref[2])  # invstd
theano/gpuarray/tests/test_dnn.py:                utt.assert_allclose(outputs_gpu[3], outputs_ref[3])  # running_mean
theano/gpuarray/tests/test_dnn.py:                utt.assert_allclose(np.nan_to_num(outputs_gpu[4]),
theano/gpuarray/tests/test_dnn.py:                utt.assert_allclose(outputs_gpu[5], outputs_ref[5], atol=2e-4)  # dx
theano/gpuarray/tests/test_dnn.py:                utt.assert_allclose(outputs_gpu[6], outputs_ref[6], rtol=4e-4, atol=1e-4)  # dscale
theano/gpuarray/tests/test_dnn.py:                utt.assert_allclose(outputs_gpu[7], outputs_ref[7])  # dbias
theano/gpuarray/tests/test_dnn.py:    out_gpu, x_mean_gpu, x_invstd_gpu = \
theano/gpuarray/tests/test_dnn.py:    grads_gpu = T.grad(None, wrt=[x, scale, bias], known_grads={out_gpu: dy})
theano/gpuarray/tests/test_dnn.py:    f_gpu = theano.function([x, scale, bias, dy],
theano/gpuarray/tests/test_dnn.py:                            [out_gpu, x_mean_gpu, x_invstd_gpu] +
theano/gpuarray/tests/test_dnn.py:                            grads_gpu,
theano/gpuarray/tests/test_dnn.py:                            mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                                 mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    assert any([isinstance(n.op, dnn.GpuDnnBatchNorm)
theano/gpuarray/tests/test_dnn.py:    assert any([isinstance(n.op, dnn.GpuDnnBatchNormGrad)
theano/gpuarray/tests/test_dnn.py:    f_gpu(X, Scale, Bias, Dy)
theano/gpuarray/tests/test_dnn.py:    # But disable cudnn and make sure it run on the GPU.
theano/gpuarray/tests/test_dnn.py:                                 mode=mode_with_gpu.excluding('cudnn'))
theano/gpuarray/tests/test_dnn.py:    assert not any([isinstance(n.op, dnn.GpuDnnBatchNorm)
theano/gpuarray/tests/test_dnn.py:    assert not any([isinstance(n.op, dnn.GpuDnnBatchNormGrad)
theano/gpuarray/tests/test_dnn.py:    assert any([isinstance(n.op, dnn.GpuElemwise)
theano/gpuarray/tests/test_dnn.py:    running_mean = gpuarray_shared_constructor(
theano/gpuarray/tests/test_dnn.py:    running_var = gpuarray_shared_constructor(
theano/gpuarray/tests/test_dnn.py:                        mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:             if isinstance(n.op, dnn.GpuDnnBatchNorm)]
theano/gpuarray/tests/test_dnn.py:            out_gpu = dnn.dnn_batch_normalization_test(x, scale, bias, mean,
theano/gpuarray/tests/test_dnn.py:            grads_gpu = T.grad(None, wrt=[x, scale, bias, mean, var], known_grads={out_gpu: dy})
theano/gpuarray/tests/test_dnn.py:            f_gpu = theano.function([x, scale, bias, mean, var, dy],
theano/gpuarray/tests/test_dnn.py:                                    [out_gpu] + grads_gpu, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                                         [out_abstract] + grads_abstract, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:            assert any([isinstance(n.op, dnn.GpuDnnBatchNormInference) for n
theano/gpuarray/tests/test_dnn.py:                outputs_gpu = f_gpu(X, Scale, Bias, Mean, Var, Dy)
theano/gpuarray/tests/test_dnn.py:                utt.assert_allclose(outputs_gpu[0], outputs_ref[0])  # out
theano/gpuarray/tests/test_dnn.py:                utt.assert_allclose(outputs_gpu[1], outputs_ref[1], atol=4e-5)  # dx
theano/gpuarray/tests/test_dnn.py:                utt.assert_allclose(outputs_gpu[2], outputs_ref[2], atol=4e-5)  # dscale
theano/gpuarray/tests/test_dnn.py:                utt.assert_allclose(outputs_gpu[3], outputs_ref[3])  # dbias
theano/gpuarray/tests/test_dnn.py:                utt.assert_allclose(outputs_gpu[4], outputs_ref[4])  # dmean
theano/gpuarray/tests/test_dnn.py:                utt.assert_allclose(outputs_gpu[5], outputs_ref[5], rtol=2e-3, atol=4e-5)  # dvar
theano/gpuarray/tests/test_dnn.py:    f = theano.function([x, scale, bias, mean, var], [out], mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:             if isinstance(n.op, dnn.GpuDnnBatchNormInference)]
theano/gpuarray/tests/test_dnn.py:                                mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                assert any([isinstance(n.op, dnn.GpuDnnBatchNorm) for n
theano/gpuarray/tests/test_dnn.py:                assert any([isinstance(n.op, dnn.GpuDnnBatchNormGrad) for n
theano/gpuarray/tests/test_dnn.py:                assert any([isinstance(n.op, dnn.GpuDnnBatchNormInference) for n
theano/gpuarray/tests/test_dnn.py:                assert not any([isinstance(n.op, (dnn.GpuDnnBatchNorm,
theano/gpuarray/tests/test_dnn.py:                                                  dnn.GpuDnnBatchNormGrad,
theano/gpuarray/tests/test_dnn.py:    params_cudnn = gpuarray_shared_constructor(
theano/gpuarray/tests/test_dnn.py:        grad_fn = theano.function([X, Y, h0], grad, mode=mode_with_gpu,
theano/gpuarray/tests/test_dnn.py:    ref_fn = theano.function([X, h0], ref_y, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    cudnn_fn = theano.function([X, h0], y, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:        cudnn_grad_params = gpuarray_shared_constructor(cudnn_grads[2])
theano/gpuarray/tests/test_dnn.py:    params_cudnn = gpuarray_shared_constructor(
theano/gpuarray/tests/test_dnn.py:        grad_fn = theano.function([X, Y, h0], grad, mode=mode_with_gpu,
theano/gpuarray/tests/test_dnn.py:    cudnn_fn = theano.function([X, h0], y, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    params_cudnn = gpuarray_shared_constructor(
theano/gpuarray/tests/test_dnn.py:        fn = theano.function([X, h0, c0], out, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:        grad_fn = theano.function([X, Y, h0, c0], grad, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    cudnn_grads_params = gpuarray_shared_constructor(cudnn_grads[3])
theano/gpuarray/tests/test_dnn.py:    params_cudnn = gpuarray_shared_constructor(
theano/gpuarray/tests/test_dnn.py:        grad_fn = theano.function([X, CY, h0, c0], grad, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    cudnn_grads_params = gpuarray_shared_constructor(cudnn_grads[3])
theano/gpuarray/tests/test_dnn.py:    mode = mode_with_gpu.excluding('conv_gemm')
theano/gpuarray/tests/test_dnn.py:    conv_op = dnn.GpuDnnConv
theano/gpuarray/tests/test_dnn.py:    conv_gradw_op = dnn.GpuDnnConvGradW
theano/gpuarray/tests/test_dnn.py:    conv_gradi_op = dnn.GpuDnnConvGradI
theano/gpuarray/tests/test_dnn.py:    mode = mode_with_gpu.excluding('conv_gemm')
theano/gpuarray/tests/test_dnn.py:    conv_op = dnn.GpuDnnConv
theano/gpuarray/tests/test_dnn.py:    conv_gradw_op = dnn.GpuDnnConvGradW
theano/gpuarray/tests/test_dnn.py:    conv_gradi_op = dnn.GpuDnnConvGradI
theano/gpuarray/tests/test_dnn.py:    st_dnn_func = theano.function([t_img, t_theta], st_dnn, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    assert any([isinstance(node.op, dnn.GpuDnnTransformerGrid) for node in apply_nodes])
theano/gpuarray/tests/test_dnn.py:    assert any([isinstance(node.op, dnn.GpuDnnTransformerSampler) for node in apply_nodes])
theano/gpuarray/tests/test_dnn.py:    img_out_gpu = st_dnn_func(img, theta)
theano/gpuarray/tests/test_dnn.py:    img_out_gpu = np.asarray(img_out_gpu)
theano/gpuarray/tests/test_dnn.py:    st_cpu_func = theano.function([t_img, t_theta], st_cpu, mode=mode_without_gpu)
theano/gpuarray/tests/test_dnn.py:    utt.assert_allclose(img_out_cpu, img_out_gpu, atol=atol, rtol=rtol)
theano/gpuarray/tests/test_dnn.py:    st_dnn_func = theano.function([inputs, theta], st_dnn, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    f_gi = theano.function([inputs, theta], mean_gi, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    assert any([isinstance(node.op, dnn.GpuDnnTransformerGradI)
theano/gpuarray/tests/test_dnn.py:    f_gt = theano.function([inputs, theta], mean_gt, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:    assert any([isinstance(node.op, dnn.GpuDnnTransformerGradT)
theano/gpuarray/tests/test_dnn.py:    utt.verify_grad(grad_functor, [inputs_val, theta_val], mode=mode_with_gpu,
theano/gpuarray/tests/test_dnn.py:            f = theano.function([inputs, filters], conv, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:                    gpu_res = f(inputs_val, filters_val)
theano/gpuarray/tests/test_dnn.py:                    utt.assert_allclose(cpu_res, np.asarray(gpu_res),
theano/gpuarray/tests/test_dnn.py:            f = theano.function([inputs, filters], grad_i, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:            assert 1 == len([node for node in f.maker.fgraph.apply_nodes if isinstance(node.op, dnn.GpuDnnConvGradI)])
theano/gpuarray/tests/test_dnn.py:            assert not any(isinstance(node.op, dnn.GpuDnnConv) for node in f.maker.fgraph.apply_nodes)
theano/gpuarray/tests/test_dnn.py:            assert not any(isinstance(node.op, dnn.GpuDnnConvGradW) for node in f.maker.fgraph.apply_nodes)
theano/gpuarray/tests/test_dnn.py:                    gpu_res = f(inputs_val, filters_val)
theano/gpuarray/tests/test_dnn.py:                    utt.assert_allclose(cpu_res, np.asarray(gpu_res))
theano/gpuarray/tests/test_dnn.py:            f = theano.function([inputs, filters], grad_w, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:            assert 1 == len([node for node in f.maker.fgraph.apply_nodes if isinstance(node.op, dnn.GpuDnnConvGradW)])
theano/gpuarray/tests/test_dnn.py:            assert not any(isinstance(node.op, dnn.GpuDnnConv) for node in f.maker.fgraph.apply_nodes)
theano/gpuarray/tests/test_dnn.py:            assert not any(isinstance(node.op, dnn.GpuDnnConvGradI) for node in f.maker.fgraph.apply_nodes)
theano/gpuarray/tests/test_dnn.py:                    gpu_res = f(inputs_val, filters_val)
theano/gpuarray/tests/test_dnn.py:                    utt.assert_allclose(cpu_res, np.asarray(gpu_res))
theano/gpuarray/tests/test_dnn.py:        return theano.function([], conv, mode=mode_with_gpu)
theano/gpuarray/tests/test_dnn.py:        # float16 precision is not supported on all GPU cards.
theano/gpuarray/tests/test_dnn.py:    theano.function([inputs, filters], [conv, gfilt], mode=mode_with_gpu)
theano/gpuarray/tests/test_nnet.py:from .config import mode_with_gpu, mode_without_gpu
theano/gpuarray/tests/test_nnet.py:    GpuCrossentropySoftmaxArgmax1HotWithBias,
theano/gpuarray/tests/test_nnet.py:    GpuCrossentropySoftmax1HotWithBiasDx,
theano/gpuarray/tests/test_nnet.py:    GpuSoftmaxWithBias, GpuSoftmax)
theano/gpuarray/tests/test_nnet.py:mode_wo_cudnn = mode_with_gpu.excluding("cudnn")
theano/gpuarray/tests/test_nnet.py:def test_GpuCrossentropySoftmaxArgmax1HotWithBias():
theano/gpuarray/tests/test_nnet.py:    # This is basic test for GpuCrossentropySoftmaxArgmax1HotWithBias
theano/gpuarray/tests/test_nnet.py:    if not isinstance(mode_with_gpu, theano.compile.DebugMode):
theano/gpuarray/tests/test_nnet.py:    # GpuCrossentropySoftmax1HotWithBiasDx to don't fail with the error
theano/gpuarray/tests/test_nnet.py:    # (the launch timed out and was terminated) on GPU card not
theano/gpuarray/tests/test_nnet.py:                               mode=mode_without_gpu)
theano/gpuarray/tests/test_nnet.py:    classify_gpu = theano.function(inputs=[y, b, dot_result],
theano/gpuarray/tests/test_nnet.py:                                   mode=mode_with_gpu)
theano/gpuarray/tests/test_nnet.py:                           GpuCrossentropySoftmaxArgmax1HotWithBias)
theano/gpuarray/tests/test_nnet.py:                for node in classify_gpu.maker.fgraph.toposort()])
theano/gpuarray/tests/test_nnet.py:    gout = classify_gpu(yy, b_values, dot_value)
theano/gpuarray/tests/test_nnet.py:def test_GpuCrossentropySoftmax1HotWithBiasDx():
theano/gpuarray/tests/test_nnet.py:    # This is basic test for GpuCrossentropySoftmax1HotWithBiasDx
theano/gpuarray/tests/test_nnet.py:    if not isinstance(mode_with_gpu, theano.compile.DebugMode):
theano/gpuarray/tests/test_nnet.py:    cpu_f = theano.function([softmax_output], op, mode=mode_without_gpu)
theano/gpuarray/tests/test_nnet.py:    gpu_f = theano.function([softmax_output], op, mode=mode_with_gpu)
theano/gpuarray/tests/test_nnet.py:    # theano.printing.debugprint(gpu_f)
theano/gpuarray/tests/test_nnet.py:                           GpuCrossentropySoftmax1HotWithBiasDx)
theano/gpuarray/tests/test_nnet.py:                for node in gpu_f.maker.fgraph.toposort()])
theano/gpuarray/tests/test_nnet.py:    gpu_out = gpu_f(softmax_output_value)
theano/gpuarray/tests/test_nnet.py:    utt.assert_allclose(cpu_out, gpu_out, rtol=rtol, atol=atol)
theano/gpuarray/tests/test_nnet.py:    # This is a basic test for GpuSoftmaxWithBias.
theano/gpuarray/tests/test_nnet.py:    f = theano.function([x, b], z, mode=mode_without_gpu)
theano/gpuarray/tests/test_nnet.py:    f_gpu = theano.function([x, b], z, mode=mode_with_gpu)
theano/gpuarray/tests/test_nnet.py:    assert isinstance(f_gpu.maker.fgraph.toposort()[-2].op,
theano/gpuarray/tests/test_nnet.py:                      GpuSoftmaxWithBias)
theano/gpuarray/tests/test_nnet.py:        gout = f_gpu(data, b_data)
theano/gpuarray/tests/test_nnet.py:    # This is basic test for GpuSoftmax.
theano/gpuarray/tests/test_nnet.py:    f = theano.function([x], z, mode=mode_without_gpu)
theano/gpuarray/tests/test_nnet.py:    f_gpu = theano.function([x], z, mode=mode_wo_cudnn)
theano/gpuarray/tests/test_nnet.py:    assert isinstance(f_gpu.maker.fgraph.toposort()[-2].op,
theano/gpuarray/tests/test_nnet.py:                      GpuSoftmax)
theano/gpuarray/tests/test_nnet.py:        gout = f_gpu(data)
theano/gpuarray/tests/test_nnet.py:    gpu_op = GpuSoftmax
theano/gpuarray/tests/test_nnet.py:    def _test_softmax(self, x, x_gpu, f_z, f_gpu_z, cmp):
theano/gpuarray/tests/test_nnet.py:        # This is basic test for GpuSoftmax and GpuDnnSoftmax
theano/gpuarray/tests/test_nnet.py:        f_gpu_z_out = f_gpu_z(x_gpu)
theano/gpuarray/tests/test_nnet.py:        f = theano.function([x], f_z_out, mode=mode_without_gpu)
theano/gpuarray/tests/test_nnet.py:        f_gpu = theano.function([x_gpu], f_gpu_z_out, mode=self.mode)
theano/gpuarray/tests/test_nnet.py:        self._check_types(f, f_gpu, T.nnet.Softmax, self.gpu_op)
theano/gpuarray/tests/test_nnet.py:        cmp(1, 5, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        cmp(2, 5, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        cmp(10, 5, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        cmp(100, 5, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        cmp(1000, 5, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        cmp(10000, 5, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        cmp(4074, 400, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        cmp(784, 784, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        cmp(4, 1000, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        cmp(4, 1024, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        cmp(4, 2000, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        cmp(4, 2024, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        cmp(4, 4074, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        cmp(2, 10000, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        cmp(128, 16 * 1024, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        cmp(128, 64 * 1024, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        cmp((2 << 15) - 1, 5, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        cmp(5, 2 << 15, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        return f, f_gpu
theano/gpuarray/tests/test_nnet.py:    def _cmp(self, n, m, f, f_gpu):
theano/gpuarray/tests/test_nnet.py:        gout = f_gpu(data)
theano/gpuarray/tests/test_nnet.py:    def _check_types(self, graph, graph_gpu, f_type, f_gpu_type):
theano/gpuarray/tests/test_nnet.py:        assert len([node for node in graph_gpu.maker.fgraph.toposort()
theano/gpuarray/tests/test_nnet.py:                    if isinstance(node.op, f_gpu_type)]) == 1
theano/gpuarray/tests/test_nnet.py:        f, f_gpu = self._test_softmax(
theano/gpuarray/tests/test_nnet.py:        self._cmp(2 << 15, 5, f, f_gpu)
theano/gpuarray/tests/test_nnet.py:        f, f_gpu = self._test_softmax(x, x, z, z, self._cmp)
theano/gpuarray/tests/test_nnet.py:        self._cmp(0, 10, f, f_gpu)
theano/gpuarray/tests/test_pickle.py:Some pickle test when pygpu isn't there. The test when pygpu is
theano/gpuarray/tests/test_pickle.py:This is needed as we skip all the test file when pygpu isn't there in
theano/gpuarray/tests/test_pickle.py:    have_pygpu = True
theano/gpuarray/tests/test_pickle.py:    have_pygpu = False
theano/gpuarray/tests/test_pickle.py:def test_unpickle_gpuarray_as_numpy_ndarray_flag1():
theano/gpuarray/tests/test_pickle.py:    # Only test when pygpu isn't
theano/gpuarray/tests/test_pickle.py:    # available. test_unpickle_gpuarray_as_numpy_ndarray_flag0 in
theano/gpuarray/tests/test_pickle.py:    # test_type.py test it when pygpu is there.
theano/gpuarray/tests/test_pickle.py:    if have_pygpu:
theano/gpuarray/tests/test_pickle.py:        raise SkipTest("pygpu active")
theano/gpuarray/tests/test_pickle.py:    oldflag = config.experimental.unpickle_gpu_on_cpu
theano/gpuarray/tests/test_pickle.py:    config.experimental.unpickle_gpu_on_cpu = False
theano/gpuarray/tests/test_pickle.py:        fname = 'GpuArray.pkl'
theano/gpuarray/tests/test_pickle.py:        config.experimental.unpickle_gpu_on_cpu = oldflag
theano/gpuarray/tests/test_pickle.py:def test_unpickle_gpuarray_as_numpy_ndarray_flag2():
theano/gpuarray/tests/test_pickle.py:    oldflag = config.experimental.unpickle_gpu_on_cpu
theano/gpuarray/tests/test_pickle.py:    config.experimental.unpickle_gpu_on_cpu = True
theano/gpuarray/tests/test_pickle.py:        fname = 'GpuArray.pkl'
theano/gpuarray/tests/test_pickle.py:        config.experimental.unpickle_gpu_on_cpu = oldflag
theano/gpuarray/tests/test_gpuarray_multinomial_wo_replacement.pkl:(ctheano.gpuarray.multinomial
theano/gpuarray/tests/test_gpuarray_multinomial_wo_replacement.pkl:GPUAMultinomialWOReplacementFromUniform
theano/gpuarray/tests/test_opt.py:import theano.gpuarray
theano/gpuarray/tests/test_opt.py:from ..type import GpuArrayType, gpuarray_shared_constructor, get_context
theano/gpuarray/tests/test_opt.py:    GpuAlloc, GpuAllocEmpty, GpuReshape, GpuFromHost, HostFromGpu, host_from_gpu)
theano/gpuarray/tests/test_opt.py:from ..blas import GpuGemm
theano/gpuarray/tests/test_opt.py:    GpuCAReduceCuda, GpuCAReduceCPY, GpuElemwise, Elemwise, max_inputs_to_GpuElemwise)
theano/gpuarray/tests/test_opt.py:from ..dnn import GpuDnnReduction
theano/gpuarray/tests/test_opt.py:from ..subtensor import GpuSubtensor
theano/gpuarray/tests/test_opt.py:from ..linalg import GpuCusolverSolve, cusolver_available, GpuCholesky
theano/gpuarray/tests/test_opt.py:from .config import mode_with_gpu, mode_without_gpu, test_ctx_name, SkipTest
theano/gpuarray/tests/test_opt.py:from theano.gpuarray import dnn, blas, opt
theano/gpuarray/tests/test_opt.py:                                   GpuFromHost, HostFromGpu,
theano/gpuarray/tests/test_opt.py:    f = theano.function([x], a, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert isinstance(a_op[0].inputs[0].type, GpuArrayType)
theano/gpuarray/tests/test_opt.py:    f = theano.function([x], a, mode=mode_with_gpu.excluding('unsafe'))
theano/gpuarray/tests/test_opt.py:    f = theano.function([x], a, mode=mode_with_gpu.including('unsafe'))
theano/gpuarray/tests/test_opt.py:    f = theano.function([x], a, mode=mode_with_gpu.excluding('unsafe'))
theano/gpuarray/tests/test_opt.py:def test_local_gpu_contiguous_gpu_contiguous():
theano/gpuarray/tests/test_opt.py:    o1 = basic_ops.gpu_contiguous(a)
theano/gpuarray/tests/test_opt.py:    o2 = basic_ops.gpu_contiguous(o1)
theano/gpuarray/tests/test_opt.py:    f1 = theano.function([a], o1, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    f2 = theano.function([a], o2, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:                     if isinstance(node.op, basic_ops.GpuContiguous)])
theano/gpuarray/tests/test_opt.py:                     if isinstance(node.op, basic_ops.GpuContiguous)])
theano/gpuarray/tests/test_opt.py:def test_local_gpu_contiguous():
theano/gpuarray/tests/test_opt.py:    f = theano.function([a], o, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:                     if isinstance(node.op, basic_ops.GpuContiguous)])
theano/gpuarray/tests/test_opt.py:    f = theano.function([m], m.flatten(), mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert GpuReshape in [type(node.op)
theano/gpuarray/tests/test_opt.py:    assert GpuReshape in [type(node.op)
theano/gpuarray/tests/test_opt.py:                        mode=mode_with_gpu.excluding("local_useless_reshape"))
theano/gpuarray/tests/test_opt.py:    assert GpuReshape in [type(node.op)
theano/gpuarray/tests/test_opt.py:    f = theano.function([m], m.flatten(ndim=2), mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert GpuReshape in [type(node.op)
theano/gpuarray/tests/test_opt.py:                            mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:        # it is using GpuCAReduceCuda that has an empty stack
theano/gpuarray/tests/test_opt.py:        if kind == b'opencl' and method in ["max", "min"]:
theano/gpuarray/tests/test_opt.py:            assert not(GpuCAReduceCuda in ops or
theano/gpuarray/tests/test_opt.py:                       GpuCAReduceCPY in ops or
theano/gpuarray/tests/test_opt.py:                       GpuDnnReduction in ops)
theano/gpuarray/tests/test_opt.py:            assert (GpuCAReduceCuda in ops or
theano/gpuarray/tests/test_opt.py:                    GpuCAReduceCPY in ops or
theano/gpuarray/tests/test_opt.py:                    GpuDnnReduction in ops)
theano/gpuarray/tests/test_opt.py:def test_local_gpualloc_memset_0():
theano/gpuarray/tests/test_opt.py:    f = theano.function([i], a, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    f = theano.function([i], a.cumsum(), mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert isinstance(topo[0].op, GpuAlloc)
theano/gpuarray/tests/test_opt.py:    a = GpuAlloc(test_ctx_name)(z, i)
theano/gpuarray/tests/test_opt.py:    f = theano.function([i], a, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert isinstance(topo[0].op, GpuAlloc) and topo[0].op.memset_0
theano/gpuarray/tests/test_opt.py:    a = GpuAlloc(test_ctx_name)(o, i)
theano/gpuarray/tests/test_opt.py:    f = theano.function([i], a, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert isinstance(topo[0].op, GpuAlloc)
theano/gpuarray/tests/test_opt.py:    a = GpuAlloc(test_ctx_name)(ones, i)
theano/gpuarray/tests/test_opt.py:    f = theano.function([i], a, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert isinstance(topo[0].op, GpuAlloc)
theano/gpuarray/tests/test_opt.py:def test_local_gpualloc_empty():
theano/gpuarray/tests/test_opt.py:    f = theano.function([i], a, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    f = theano.function([i], a.cumsum(), mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert isinstance(topo[0].op, GpuAllocEmpty)
theano/gpuarray/tests/test_opt.py:    f = theano.function([i, ii], a.cumsum(axis=0), mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert isinstance(topo[0].op, GpuAllocEmpty)
theano/gpuarray/tests/test_opt.py:    f = theano.function([v], [up], mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert isinstance(rebr.inputs[0].type, GpuArrayType)
theano/gpuarray/tests/test_opt.py:    assert isinstance(rebr.outputs[0].type, GpuArrayType)
theano/gpuarray/tests/test_opt.py:    mode = mode_with_gpu
theano/gpuarray/tests/test_opt.py:    input_type = GpuArrayType
theano/gpuarray/tests/test_opt.py:class test_gpu_ifelse(test_ifelse.test_ifelse):
theano/gpuarray/tests/test_opt.py:    mode = mode_with_gpu
theano/gpuarray/tests/test_opt.py:        return basic_ops.as_gpuarray_variable(v, test_ctx_name)
theano/gpuarray/tests/test_opt.py:    shared = staticmethod(gpuarray_shared_constructor)
theano/gpuarray/tests/test_opt.py:        return theano.ifelse.IfElse(n, gpu=True, as_view=True)
theano/gpuarray/tests/test_opt.py:                            mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:                            mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:        y = gpuarray_shared_constructor(np.asarray(1, dtype='float32'),
theano/gpuarray/tests/test_opt.py:            theano.function([x], [a], mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    # Test that print ops don't block gpu optimization
theano/gpuarray/tests/test_opt.py:                        mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert isinstance(topo[0].op, GpuFromHost)
theano/gpuarray/tests/test_opt.py:    assert isinstance(topo[2].op, GpuElemwise)
theano/gpuarray/tests/test_opt.py:    assert topo[3].op == host_from_gpu
theano/gpuarray/tests/test_opt.py:    # Test that PdbBreakpoint ops don't block gpu optimization
theano/gpuarray/tests/test_opt.py:    f = theano.function([b], output, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    # breakpoint has been moved to the gpu.
theano/gpuarray/tests/test_opt.py:    assert isinstance(topo[-2].op, GpuElemwise)
theano/gpuarray/tests/test_opt.py:    assert topo[-1].op == host_from_gpu
theano/gpuarray/tests/test_opt.py:def test_local_gpu_elemwise_careduce():
theano/gpuarray/tests/test_opt.py:    mode_with_gpu_no_cudnn = mode_with_gpu.excluding('cudnn')
theano/gpuarray/tests/test_opt.py:            f = theano.function([x], o, mode=mode_with_gpu_no_cudnn)
theano/gpuarray/tests/test_opt.py:            assert isinstance(topo[1].op, GpuCAReduceCuda)
theano/gpuarray/tests/test_opt.py:    f_gpu = theano.function([x, y, a], o, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:                   for n in f_gpu.maker.fgraph.apply_nodes)
theano/gpuarray/tests/test_opt.py:    assert any(isinstance(n.op, GpuGemm)
theano/gpuarray/tests/test_opt.py:               for n in f_gpu.maker.fgraph.apply_nodes)
theano/gpuarray/tests/test_opt.py:    utt.assert_allclose(f_cpu(x_val, y_val, a_val), f_gpu(x_val, y_val, a_val))
theano/gpuarray/tests/test_opt.py:    assert _check_stack_trace(f_gpu)
theano/gpuarray/tests/test_opt.py:def test_local_gpu_subtensor():
theano/gpuarray/tests/test_opt.py:    f = theano.function([], t[3:4], mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert not any([isinstance(node.op, GpuSubtensor) for node in topo])
theano/gpuarray/tests/test_opt.py:    f = theano.function([t], t[3:4], mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert not any([isinstance(node.op, GpuSubtensor) for node in topo])
theano/gpuarray/tests/test_opt.py:    # We want the subtensor to be on the GPU to prevent multiple transfer.
theano/gpuarray/tests/test_opt.py:    f = theano.function([t], [t[3:4], t + 1], mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert any([isinstance(node.op, GpuSubtensor) for node in topo])
theano/gpuarray/tests/test_opt.py:    # We want the subtensor to be on the GPU to prevent multiple transfer.
theano/gpuarray/tests/test_opt.py:    f = theano.function([t], [t[3:4], t + 1, t], mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert any([isinstance(node.op, GpuSubtensor) for node in topo])
theano/gpuarray/tests/test_opt.py:    f = theano.function([], t[3:4] + 1, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert not any([isinstance(node.op, GpuSubtensor) for node in topo])
theano/gpuarray/tests/test_opt.py:    # Our optimizer isn't smart enough to move to the GPU Elemwise.
theano/gpuarray/tests/test_opt.py:    # If it where just a little bit smarter, it could wrongly move it to the GPU.
theano/gpuarray/tests/test_opt.py:    # If it where super smart, it would know it should not move it to the GPU.
theano/gpuarray/tests/test_opt.py:def test_local_gpu_elemwise():
theano/gpuarray/tests/test_opt.py:    # Test local_gpu_elemwise when there is a dtype upcastable to float32
theano/gpuarray/tests/test_opt.py:    # the op are on the gpu.
theano/gpuarray/tests/test_opt.py:    f = theano.function([a, b, c], a + b + c, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert sum(isinstance(node.op, GpuElemwise) for node in topo) == 1
theano/gpuarray/tests/test_opt.py:    # to the gpu
theano/gpuarray/tests/test_opt.py:    f = theano.function([a, b, c], out_op(a, b, c), mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert sum(isinstance(node.op, GpuElemwise) for node in topo) == 1
theano/gpuarray/tests/test_opt.py:    f = theano.function([a, b, c], outs_op(a, b, c), mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert sum(isinstance(node.op, GpuElemwise) for node in topo) == 1
theano/gpuarray/tests/test_opt.py:    f = theano.function([a, b, c], outs_op(a, b, c), mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    assert sum(isinstance(node.op, GpuElemwise) for node in topo) == 1
theano/gpuarray/tests/test_opt.py:    c = gpuarray_shared_constructor(np.asarray(c_v, dtype='float32'))
theano/gpuarray/tests/test_opt.py:                        mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    # extremely large numbers of arguments on gpu.
theano/gpuarray/tests/test_opt.py:                for mode in [mode_with_gpu, mode_without_gpu]:
theano/gpuarray/tests/test_opt.py:                    # test the optimization local_gpua_elemwise
theano/gpuarray/tests/test_opt.py:                    # assert that the test was done on the gpu.
theano/gpuarray/tests/test_opt.py:                    if mode is mode_with_gpu:
theano/gpuarray/tests/test_opt.py:                            max_inputs_to_GpuElemwise(output.owner) - num_args)
theano/gpuarray/tests/test_opt.py:                        assert any(isinstance(node.op, GpuElemwise)
theano/gpuarray/tests/test_opt.py:                                       if not isinstance(node.op, GpuElemwise))
theano/gpuarray/tests/test_opt.py:                results_gpu, results_cpu = outputs
theano/gpuarray/tests/test_opt.py:                utt.assert_allclose(results_gpu, results_cpu)
theano/gpuarray/tests/test_opt.py:def test_not_useless_scalar_gpuelemwise():
theano/gpuarray/tests/test_opt.py:    # We don't want to move elemwise on scalar on the GPU when the
theano/gpuarray/tests/test_opt.py:    # result will not be used on the GPU!
theano/gpuarray/tests/test_opt.py:                                mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:        gemms = [app for app in topo if isinstance(app.op, GpuGemm)]
theano/gpuarray/tests/test_opt.py:def test_local_lift_abstractconv_gpu_shape():
theano/gpuarray/tests/test_opt.py:        f = theano.function([s, a, b], c, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    ms = gpuarray_shared_constructor(m, name="m_shared")
theano/gpuarray/tests/test_opt.py:    mode_local_assert = mode_with_gpu.including("assert_no_cpu_op")
theano/gpuarray/tests/test_opt.py:    mode_local_assert = mode_local_assert.excluding("local_gpua_elemwise")
theano/gpuarray/tests/test_opt.py:                        mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    f_cpu = theano.function([A, b], o, mode_without_gpu)
theano/gpuarray/tests/test_opt.py:    f_gpu = theano.function([A, b], o, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:                   for n in f_gpu.maker.fgraph.apply_nodes)
theano/gpuarray/tests/test_opt.py:    assert any(isinstance(n.op, GpuCusolverSolve) and n.op.inplace
theano/gpuarray/tests/test_opt.py:               for n in f_gpu.maker.fgraph.apply_nodes)
theano/gpuarray/tests/test_opt.py:    utt.assert_allclose(f_cpu(A_val, b_val), f_gpu(A_val, b_val))
theano/gpuarray/tests/test_opt.py:    assert _check_stack_trace(f_gpu)
theano/gpuarray/tests/test_opt.py:def test_gpu_solve_not_inplace():
theano/gpuarray/tests/test_opt.py:    f_cpu = theano.function([A, b], o, mode_without_gpu)
theano/gpuarray/tests/test_opt.py:    f_gpu = theano.function([A, b], o, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    count_not_inplace = len([n.op for n in f_gpu.maker.fgraph.apply_nodes
theano/gpuarray/tests/test_opt.py:                             if isinstance(n.op, GpuCusolverSolve) and not n.op.inplace])
theano/gpuarray/tests/test_opt.py:    utt.assert_allclose(f_cpu(A_val, b_val), f_gpu(A_val, b_val))
theano/gpuarray/tests/test_opt.py:    f_cpu = theano.function([A], o, mode=mode_without_gpu)
theano/gpuarray/tests/test_opt.py:    f_gpu = theano.function([A], o, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:                   for n in f_gpu.maker.fgraph.apply_nodes)
theano/gpuarray/tests/test_opt.py:    # GpuCholesky op in this graph should be inplace (as his input is not reused by other op).
theano/gpuarray/tests/test_opt.py:    assert any(isinstance(n.op, GpuCholesky) and n.op.inplace
theano/gpuarray/tests/test_opt.py:               for n in f_gpu.maker.fgraph.apply_nodes)
theano/gpuarray/tests/test_opt.py:    utt.assert_allclose(f_cpu(A_val), f_gpu(A_val))
theano/gpuarray/tests/test_opt.py:def test_gpu_cholesky_not_inplace():
theano/gpuarray/tests/test_opt.py:    f_cpu = theano.function([A], D, mode=mode_without_gpu)
theano/gpuarray/tests/test_opt.py:    f_gpu = theano.function([A], D, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    # GpuCholesky op in this graph should NOT be inplace (as his input is reused in another op)
theano/gpuarray/tests/test_opt.py:    count_cholesky_not_inplace = len([n.op for n in f_gpu.maker.fgraph.apply_nodes
theano/gpuarray/tests/test_opt.py:                                      if isinstance(n.op, GpuCholesky) and not n.op.inplace])
theano/gpuarray/tests/test_opt.py:    utt.assert_allclose(f_cpu(A_val), f_gpu(A_val))
theano/gpuarray/tests/test_opt.py:def test_local_gpua_advanced_incsubtensor():
theano/gpuarray/tests/test_opt.py:        f = theano.function([x, y], z, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:    f = theano.function([x, y], [z, gx], mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:            inp1 = basic_ops.as_gpuarray_variable(inp1, test_ctx_name)
theano/gpuarray/tests/test_opt.py:            inp2 = basic_ops.as_gpuarray_variable(inp2, test_ctx_name)
theano/gpuarray/tests/test_opt.py:        mode = mode_with_gpu.including('conv_meta').excluding('conv_dnn').excluding('conv_gemm')
theano/gpuarray/tests/test_opt.py:            ref_func = theano.function([], conv_op, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:            inp1 = basic_ops.as_gpuarray_variable(inp1, None)
theano/gpuarray/tests/test_opt.py:            inp2 = basic_ops.as_gpuarray_variable(inp2, None)
theano/gpuarray/tests/test_opt.py:        mode = mode_with_gpu.including('conv_meta').excluding('conv_dnn').excluding('conv_gemm')
theano/gpuarray/tests/test_opt.py:                    mode=mode_with_gpu.including('conv_meta'))
theano/gpuarray/tests/test_opt.py:        ref_func = theano.function([], conv_op, mode=mode_with_gpu)
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorrMM)
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorrMM_gradWeights)
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConv)
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConvGradW)
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorrMM_gradWeights)
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorrMM)
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConvGradW)
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConv)
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorrMM_gradInputs)
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorrMM)
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConvGradI)
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConv)
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorr3dMM)
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorr3dMM_gradWeights)
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConvGradW)
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConv)
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorr3dMM_gradWeights)
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorr3dMM)
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConv)
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConvGradW)
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorr3dMM_gradInputs)
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorr3dMM)
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConv)
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConvGradI)
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorrMM,
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorrMM_gradInputs,
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConv,
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConvGradI,
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorr3dMM,
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorr3dMM_gradInputs,
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConv,
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConvGradI,
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorrMM,
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConv,
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorrMM_gradWeights,
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConvGradW,
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorrMM_gradInputs,
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConvGradI,
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorrMM,
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorrMM_gradWeights,
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorrMM_gradInputs,
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorr3dMM,
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConv,
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorr3dMM_gradWeights,
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConvGradW,
theano/gpuarray/tests/test_opt.py:                              blas.GpuCorr3dMM_gradInputs,
theano/gpuarray/tests/test_opt.py:                              dnn.GpuDnnConvGradI,
theano/gpuarray/tests/test_elemwise.py:from .config import mode_with_gpu, mode_without_gpu, test_ctx_name
theano/gpuarray/tests/test_elemwise.py:from .test_basic_ops import rand_gpuarray
theano/gpuarray/tests/test_elemwise.py:from ..elemwise import (GpuElemwise, GpuDimShuffle,
theano/gpuarray/tests/test_elemwise.py:                        GpuCAReduceCuda, GpuCAReduceCPY, GpuErfinv, GpuErfcinv)
theano/gpuarray/tests/test_elemwise.py:from ..dnn import GpuDnnReduction
theano/gpuarray/tests/test_elemwise.py:from ..type import GpuArrayType, get_context, gpuarray_shared_constructor
theano/gpuarray/tests/test_elemwise.py:from pygpu import ndgpuarray as gpuarray
theano/gpuarray/tests/test_elemwise.py:# This is actually a test for GpuElemwise
theano/gpuarray/tests/test_elemwise.py:class test_gpu_Broadcast(test_elemwise.test_Broadcast):
theano/gpuarray/tests/test_elemwise.py:    cop = GpuElemwise
theano/gpuarray/tests/test_elemwise.py:    ctype = GpuArrayType
theano/gpuarray/tests/test_elemwise.py:        return rand_gpuarray(*shp, **dict(cls=gpuarray))
theano/gpuarray/tests/test_elemwise.py:    # Test that GpuElemwise(pow) can compile with any combination of integer
theano/gpuarray/tests/test_elemwise.py:            # Compile a gpu function with the specified dtypes
theano/gpuarray/tests/test_elemwise.py:            exp = gpuarray_shared_constructor(exp_val)
theano/gpuarray/tests/test_elemwise.py:            f = theano.function([base], output, mode=mode_with_gpu)
theano/gpuarray/tests/test_elemwise.py:            # We don't transfer to the GPU when the output dtype is int*
theano/gpuarray/tests/test_elemwise.py:                     if isinstance(n.op, GpuElemwise)])
theano/gpuarray/tests/test_elemwise.py:        # to have the GPU ops run on large data.
theano/gpuarray/tests/test_elemwise.py:        if isinstance(mode_with_gpu, DebugMode):
theano/gpuarray/tests/test_elemwise.py:            cls.mode_with_gpu = copy(mode_with_gpu)
theano/gpuarray/tests/test_elemwise.py:            cls.mode_with_gpu.check_isfinite = False
theano/gpuarray/tests/test_elemwise.py:            cls.mode_with_gpu = mode_with_gpu
theano/gpuarray/tests/test_elemwise.py:        if isinstance(mode_without_gpu, DebugMode):
theano/gpuarray/tests/test_elemwise.py:            cls.mode_without_gpu = copy(mode_without_gpu)
theano/gpuarray/tests/test_elemwise.py:            cls.mode_without_gpu.check_isfinite = False
theano/gpuarray/tests/test_elemwise.py:            cls.mode_without_gpu = mode_without_gpu
theano/gpuarray/tests/test_elemwise.py:    def check_gpu_scalar_op(self, theano_function, scalar_optype):
theano/gpuarray/tests/test_elemwise.py:            if isinstance(node.op, GpuElemwise) and isinstance(node.op.scalar_op, scalar_optype):
theano/gpuarray/tests/test_elemwise.py:            f_host = theano.function([vector], output, name='HOST/erfinv/' + dtype, mode=self.mode_without_gpu)
theano/gpuarray/tests/test_elemwise.py:            f_gpu = theano.function([vector], output, name='GPU/erfinv/' + dtype, mode=self.mode_with_gpu)
theano/gpuarray/tests/test_elemwise.py:            assert len([n for n in f_host.maker.fgraph.apply_nodes if isinstance(n.op, GpuElemwise)]) == 0
theano/gpuarray/tests/test_elemwise.py:            if not theano.config.device.startswith('opencl'):
theano/gpuarray/tests/test_elemwise.py:                assert self.check_gpu_scalar_op(f_gpu, GpuErfinv), \
theano/gpuarray/tests/test_elemwise.py:                    'Function graph does not contains scalar op "GpuErfinv".'
theano/gpuarray/tests/test_elemwise.py:            f_gpu(vector_val)
theano/gpuarray/tests/test_elemwise.py:            out_gpu = f_gpu(vector_val)
theano/gpuarray/tests/test_elemwise.py:            assert_allclose(out_host, out_gpu)
theano/gpuarray/tests/test_elemwise.py:            assert_allclose(self.expected_erfinv_outputs[dtype], out_gpu)
theano/gpuarray/tests/test_elemwise.py:            f_host = theano.function([vector], output, name='HOST/erfcinv/' + dtype, mode=self.mode_without_gpu)
theano/gpuarray/tests/test_elemwise.py:            f_gpu = theano.function([vector], output, name='GPU/erfcinv/' + dtype, mode=self.mode_with_gpu)
theano/gpuarray/tests/test_elemwise.py:            assert len([n for n in f_host.maker.fgraph.apply_nodes if isinstance(n.op, GpuElemwise)]) == 0
theano/gpuarray/tests/test_elemwise.py:            if not theano.config.device.startswith('opencl'):
theano/gpuarray/tests/test_elemwise.py:                assert self.check_gpu_scalar_op(f_gpu, GpuErfcinv), \
theano/gpuarray/tests/test_elemwise.py:                    'Function graph does not contains scalar op "GpuErfcinv".'
theano/gpuarray/tests/test_elemwise.py:            f_gpu(vector_val)
theano/gpuarray/tests/test_elemwise.py:            out_gpu = f_gpu(vector_val)
theano/gpuarray/tests/test_elemwise.py:            assert_allclose(out_host, out_gpu)
theano/gpuarray/tests/test_elemwise.py:            assert_allclose(self.expected_erfcinv_outputs[dtype], out_gpu)
theano/gpuarray/tests/test_elemwise.py:        theano.function([w, x, y], o, mode=mode_with_gpu)
theano/gpuarray/tests/test_elemwise.py:        theano.function([v, w, x, y, z], o, mode=mode_with_gpu)
theano/gpuarray/tests/test_elemwise.py:                            mode=mode_with_gpu)
theano/gpuarray/tests/test_elemwise.py:class test_GpuDimShuffle(test_elemwise.test_DimShuffle):
theano/gpuarray/tests/test_elemwise.py:    op = GpuDimShuffle
theano/gpuarray/tests/test_elemwise.py:class test_GpuCAReduceCPY(test_elemwise.test_CAReduce):
theano/gpuarray/tests/test_elemwise.py:    op = GpuCAReduceCPY
theano/gpuarray/tests/test_elemwise.py:    mode = mode_with_gpu
theano/gpuarray/tests/test_elemwise.py:                                    optimizer=mode_with_gpu.optimizer),
theano/gpuarray/tests/test_elemwise.py:                                    optimizer=mode_with_gpu.optimizer),
theano/gpuarray/tests/test_elemwise.py:                                    optimizer=mode_with_gpu.optimizer),
theano/gpuarray/tests/test_elemwise.py:                                    optimizer=mode_with_gpu.optimizer),
theano/gpuarray/tests/test_elemwise.py:            super(test_GpuCAReduceCPY, self).test_infer_shape(dtype)
theano/gpuarray/tests/test_elemwise.py:class test_GpuCAReduceCuda(test_GpuCAReduceCPY):
theano/gpuarray/tests/test_elemwise.py:             # Test all GPU cases implemented
theano/gpuarray/tests/test_elemwise.py:    op = GpuCAReduceCuda
theano/gpuarray/tests/test_elemwise.py:        super(test_GpuCAReduceCuda, self).setUp()
theano/gpuarray/tests/test_elemwise.py:        if get_context(test_ctx_name).kind != b'cuda':
theano/gpuarray/tests/test_elemwise.py:            raise SkipTest("Cuda specific tests")
theano/gpuarray/tests/test_elemwise.py:class T_gpureduce_dtype(test_elemwise.T_reduce_dtype):
theano/gpuarray/tests/test_elemwise.py:    mode = mode_with_gpu.excluding('local_cut_useless_reduce')
theano/gpuarray/tests/test_elemwise.py:    # GpuDnnReduction doesn't cover all cases, but should cover some
theano/gpuarray/tests/test_elemwise.py:    op = (GpuCAReduceCuda, GpuDnnReduction)
theano/gpuarray/tests/test_elemwise.py:        if get_context(test_ctx_name).kind != b'cuda':
theano/gpuarray/tests/test_elemwise.py:            raise SkipTest("Cuda specific tests")
theano/gpuarray/tests/test_elemwise.py:                        mode=mode_with_gpu)
theano/gpuarray/tests/test_sort.py:from .config import mode_with_gpu
theano/gpuarray/tests/test_sort.py:from ..sort import GpuTopKOp
theano/gpuarray/tests/test_sort.py:class Test_GpuTopK(theano.tensor.tests.test_sort.Test_TopK):
theano/gpuarray/tests/test_sort.py:    mode = mode_with_gpu
theano/gpuarray/tests/test_sort.py:    op_class = GpuTopKOp
theano/gpuarray/tests/config.py:import theano.gpuarray
theano/gpuarray/tests/config.py:if theano.gpuarray.pygpu is None:
theano/gpuarray/tests/config.py:    raise SkipTest("pygpu not installed")
theano/gpuarray/tests/config.py:if (not theano.gpuarray.pygpu_activated and
theano/gpuarray/tests/config.py:        theano.gpuarray.init_dev('cuda')
theano/gpuarray/tests/config.py:if not theano.gpuarray.pygpu_activated:
theano/gpuarray/tests/config.py:        raise SkipTest("pygpu disabled")
theano/gpuarray/tests/config.py:    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpuarray').excluding('gpu')
theano/gpuarray/tests/config.py:    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpuarray')
theano/gpuarray/tests/config.py:    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpuarray').excluding('gpu')
theano/gpuarray/tests/config.py:    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpuarray')
theano/gpuarray/tests/config.py:    mode_without_gpu.check_py_code = False
theano/gpuarray/tests/test_neighbours.py:from .config import mode_with_gpu
theano/gpuarray/tests/test_neighbours.py:from ..neighbours import GpuImages2Neibs
theano/gpuarray/tests/test_neighbours.py:class T_GpuImages2Neibs(test_neighbours.T_Images2Neibs):
theano/gpuarray/tests/test_neighbours.py:    mode = mode_with_gpu
theano/gpuarray/tests/test_neighbours.py:    op = GpuImages2Neibs
theano/gpuarray/tests/test_subtensor.py:from ..basic_ops import HostFromGpu, GpuFromHost, GpuContiguous
theano/gpuarray/tests/test_subtensor.py:from ..elemwise import GpuDimShuffle
theano/gpuarray/tests/test_subtensor.py:from ..subtensor import (GpuIncSubtensor, GpuSubtensor,
theano/gpuarray/tests/test_subtensor.py:                         GpuAdvancedSubtensor1,
theano/gpuarray/tests/test_subtensor.py:                         GpuAdvancedSubtensor,
theano/gpuarray/tests/test_subtensor.py:                         GpuAdvancedBooleanSubtensor,
theano/gpuarray/tests/test_subtensor.py:                         GpuAdvancedIncSubtensor,
theano/gpuarray/tests/test_subtensor.py:                         GpuAdvancedIncSubtensor1,
theano/gpuarray/tests/test_subtensor.py:                         GpuAdvancedIncSubtensor1_dev20,
theano/gpuarray/tests/test_subtensor.py:                         GpuExtractDiag,
theano/gpuarray/tests/test_subtensor.py:                         GpuAllocDiag)
theano/gpuarray/tests/test_subtensor.py:from ..type import gpuarray_shared_constructor
theano/gpuarray/tests/test_subtensor.py:from .config import mode_with_gpu, test_ctx_name
theano/gpuarray/tests/test_subtensor.py:            return gpuarray_shared_constructor(x, target=test_ctx_name,
theano/gpuarray/tests/test_subtensor.py:            sub=GpuSubtensor,
theano/gpuarray/tests/test_subtensor.py:            inc_sub=GpuIncSubtensor,
theano/gpuarray/tests/test_subtensor.py:            adv_sub1=GpuAdvancedSubtensor1,
theano/gpuarray/tests/test_subtensor.py:            adv_incsub1=GpuAdvancedIncSubtensor1,
theano/gpuarray/tests/test_subtensor.py:            adv_sub=GpuAdvancedSubtensor,
theano/gpuarray/tests/test_subtensor.py:            adv_bool_sub=GpuAdvancedBooleanSubtensor,
theano/gpuarray/tests/test_subtensor.py:            dimshuffle=GpuDimShuffle,
theano/gpuarray/tests/test_subtensor.py:            mode=mode_with_gpu,
theano/gpuarray/tests/test_subtensor.py:            ignore_topo=(HostFromGpu, GpuFromHost,
theano/gpuarray/tests/test_subtensor.py:                         DeepCopyOp, GpuContiguous))
theano/gpuarray/tests/test_subtensor.py:        # GPU opt can't run in fast_compile only.
theano/gpuarray/tests/test_subtensor.py:        assert self.sub == GpuSubtensor
theano/gpuarray/tests/test_subtensor.py:            return gpuarray_shared_constructor(x, target=test_ctx_name,
theano/gpuarray/tests/test_subtensor.py:            sub=GpuSubtensor,
theano/gpuarray/tests/test_subtensor.py:            inc_sub=GpuIncSubtensor,
theano/gpuarray/tests/test_subtensor.py:            adv_sub1=GpuAdvancedSubtensor1,
theano/gpuarray/tests/test_subtensor.py:            adv_incsub1=GpuAdvancedIncSubtensor1,
theano/gpuarray/tests/test_subtensor.py:            adv_sub=GpuAdvancedSubtensor,
theano/gpuarray/tests/test_subtensor.py:            adv_bool_sub=GpuAdvancedBooleanSubtensor,
theano/gpuarray/tests/test_subtensor.py:            dimshuffle=GpuDimShuffle,
theano/gpuarray/tests/test_subtensor.py:            mode=mode_with_gpu,
theano/gpuarray/tests/test_subtensor.py:            ignore_topo=(HostFromGpu, GpuFromHost,
theano/gpuarray/tests/test_subtensor.py:                         DeepCopyOp, GpuContiguous))
theano/gpuarray/tests/test_subtensor.py:        # GPU opt can't run in fast_compile only.
theano/gpuarray/tests/test_subtensor.py:        assert self.sub == GpuSubtensor
theano/gpuarray/tests/test_subtensor.py:    # Test the second case in the opt local_gpu_advanced_incsubtensor1
theano/gpuarray/tests/test_subtensor.py:        shared = gpuarray_shared_constructor
theano/gpuarray/tests/test_subtensor.py:        f = theano.function([y], expr, mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:        assert sum([isinstance(node.op, GpuAdvancedIncSubtensor1)
theano/gpuarray/tests/test_subtensor.py:        shared = gpuarray_shared_constructor
theano/gpuarray/tests/test_subtensor.py:        f = theano.function([y], expr, mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:        assert sum([isinstance(node.op, GpuAdvancedIncSubtensor1_dev20)
theano/gpuarray/tests/test_subtensor.py:        shared = gpuarray_shared_constructor
theano/gpuarray/tests/test_subtensor.py:        f = theano.function([y], expr, mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:        assert sum([isinstance(node.op, GpuAdvancedIncSubtensor1)
theano/gpuarray/tests/test_subtensor.py:        shared = gpuarray_shared_constructor
theano/gpuarray/tests/test_subtensor.py:        f = theano.function([y], expr, mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:        assert sum([isinstance(node.op, (GpuAdvancedIncSubtensor1_dev20,
theano/gpuarray/tests/test_subtensor.py:                                         GpuAdvancedIncSubtensor1))
theano/gpuarray/tests/test_subtensor.py:    shared = gpuarray_shared_constructor
theano/gpuarray/tests/test_subtensor.py:    f = theano.function([y], expr, mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:    assert sum([isinstance(node.op, GpuAdvancedIncSubtensor1)
theano/gpuarray/tests/test_subtensor.py:    f = theano.function([y], expr, mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:    assert sum([isinstance(node.op, GpuIncSubtensor)
theano/gpuarray/tests/test_subtensor.py:    # Build a GPU variable which value will have an offset (x1)
theano/gpuarray/tests/test_subtensor.py:    x = gpuarray_shared_constructor(np.zeros(5, dtype=theano.config.floatX))
theano/gpuarray/tests/test_subtensor.py:    f = theano.function([y], z, updates={x: z}, mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:            shared=gpuarray_shared_constructor,
theano/gpuarray/tests/test_subtensor.py:            sub=GpuAdvancedSubtensor,
theano/gpuarray/tests/test_subtensor.py:            inc_sub=GpuAdvancedIncSubtensor,
theano/gpuarray/tests/test_subtensor.py:            mode=mode_with_gpu,
theano/gpuarray/tests/test_subtensor.py:            ignore_topo=(HostFromGpu, GpuFromHost,
theano/gpuarray/tests/test_subtensor.py:        # GPU opt can't run in fast_compile only.
theano/gpuarray/tests/test_subtensor.py:        assert self.sub == GpuAdvancedSubtensor
theano/gpuarray/tests/test_subtensor.py:            shared=gpuarray_shared_constructor,
theano/gpuarray/tests/test_subtensor.py:            sub=GpuAdvancedSubtensor,
theano/gpuarray/tests/test_subtensor.py:            mode=mode_with_gpu,
theano/gpuarray/tests/test_subtensor.py:            ignore_topo=(HostFromGpu, GpuFromHost,
theano/gpuarray/tests/test_subtensor.py:        # GPU opt can't run in fast_compile only.
theano/gpuarray/tests/test_subtensor.py:        assert self.sub == GpuAdvancedSubtensor
theano/gpuarray/tests/test_subtensor.py:    # Test the advancedsubtensor on gpu.
theano/gpuarray/tests/test_subtensor.py:    shared = gpuarray_shared_constructor
theano/gpuarray/tests/test_subtensor.py:    f = theano.function([idx1, idx2], expr, mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:    assert sum([isinstance(node.op, GpuAdvancedSubtensor)
theano/gpuarray/tests/test_subtensor.py:class test_gpuextractdiag(unittest.TestCase):
theano/gpuarray/tests/test_subtensor.py:        fn = theano.function([x], tensor.ExtractDiag()(x), mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:        assert any([isinstance(node.op, GpuExtractDiag)
theano/gpuarray/tests/test_subtensor.py:        fn = theano.function([x], GpuExtractDiag()(x), mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:        fn = theano.function([x], GpuExtractDiag(2)(x), mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:        fn = theano.function([x], GpuExtractDiag(-3)(x), mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:                GpuExtractDiag(offset, axis1, axis2)(x).eval({x: np_x}),
theano/gpuarray/tests/test_subtensor.py:                GpuExtractDiag(offset, axis1, axis2)(x).eval({x: np_x}),
theano/gpuarray/tests/test_subtensor.py:class TestGpuAllocDiag(test_basic.TestAllocDiag):
theano/gpuarray/tests/test_subtensor.py:            alloc_diag=GpuAllocDiag,
theano/gpuarray/tests/test_subtensor.py:            mode=mode_with_gpu
theano/gpuarray/tests/test_subtensor.py:class test_gpuallocdiag(unittest.TestCase):
theano/gpuarray/tests/test_subtensor.py:        fn = theano.function([x], tensor.AllocDiag()(x), mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:        assert any([isinstance(node.op, GpuAllocDiag)
theano/gpuarray/tests/test_subtensor.py:        fn = theano.function([x], GpuAllocDiag()(x), mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:        fn = theano.function([x], GpuAllocDiag(2)(x), mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:        fn = theano.function([x], GpuAllocDiag(-3)(x), mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:        mtx_x = GpuAllocDiag()(x)
theano/gpuarray/tests/test_subtensor.py:        fn_grad_x = theano.function([x], grad_x, mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:        fn_grad_mtx_x = theano.function([x], grad_mtx_x, mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:        mtx_x = GpuAllocDiag(2)(x)
theano/gpuarray/tests/test_subtensor.py:        fn_grad_x = theano.function([x], grad_x, mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:        fn_grad_mtx_x = theano.function([x], grad_mtx_x, mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:        mtx_x = GpuAllocDiag(-3)(x)
theano/gpuarray/tests/test_subtensor.py:        fn_grad_x = theano.function([x], grad_x, mode=mode_with_gpu)
theano/gpuarray/tests/test_subtensor.py:        fn_grad_mtx_x = theano.function([x], grad_mtx_x, mode=mode_with_gpu)
theano/gpuarray/tests/test_blas.py:from .. import gpuarray_shared_constructor
theano/gpuarray/tests/test_blas.py:from .config import mode_with_gpu, test_ctx_name
theano/gpuarray/tests/test_blas.py:from ..blas import (gpugemv_inplace, gpugemv_no_inplace,
theano/gpuarray/tests/test_blas.py:                    gpugemm_inplace, gpugemm_no_inplace,
theano/gpuarray/tests/test_blas.py:                    gpugemmbatch_inplace,
theano/gpuarray/tests/test_blas.py:                    gpuger_inplace, gpuger_no_inplace,
theano/gpuarray/tests/test_blas.py:                    GpuGer, GpuGemm, gpu_dot22)
theano/gpuarray/tests/test_blas.py:GpuGemvTester = makeTester(
theano/gpuarray/tests/test_blas.py:    'GpuGemvTester',
theano/gpuarray/tests/test_blas.py:    op=gemv_inplace, gpu_op=gpugemv_inplace,
theano/gpuarray/tests/test_blas.py:    float16_shared = [gpuarray_shared_constructor(val, target=test_ctx_name)
theano/gpuarray/tests/test_blas.py:    f = theano.function([], o, mode=mode_with_gpu)
theano/gpuarray/tests/test_blas.py:    assert any([isinstance(n.op, GpuGemm) for n in topo])
theano/gpuarray/tests/test_blas.py:    float16_shared = [gpuarray_shared_constructor(val, target=test_ctx_name)
theano/gpuarray/tests/test_blas.py:    o = gpugemm_no_inplace(*float16_shared)
theano/gpuarray/tests/test_blas.py:    float16_shared = [gpuarray_shared_constructor(val)
theano/gpuarray/tests/test_blas.py:    o = gpu_dot22(*float16_shared)
theano/gpuarray/tests/test_blas.py:class TestGpuSgemv(TestCase, BaseGemv, utt.TestOptimizationMixin):
theano/gpuarray/tests/test_blas.py:    mode = mode_with_gpu
theano/gpuarray/tests/test_blas.py:    gemv = gpugemv_no_inplace
theano/gpuarray/tests/test_blas.py:    gemv_inplace = gpugemv_inplace
theano/gpuarray/tests/test_blas.py:            return gpuarray_shared_constructor(val)
theano/gpuarray/tests/test_blas.py:GpuGemmTester = makeTester(
theano/gpuarray/tests/test_blas.py:    'GpuGemmTester',
theano/gpuarray/tests/test_blas.py:    op=gemm_inplace, gpu_op=gpugemm_inplace,
theano/gpuarray/tests/test_blas.py:GpuGemmBatchTester = makeTester(
theano/gpuarray/tests/test_blas.py:    'GpuGemmBatchTester',
theano/gpuarray/tests/test_blas.py:    gpu_op=gpugemmbatch_inplace,
theano/gpuarray/tests/test_blas.py:class TestGpuGemmBatchStrided(TestCase):
theano/gpuarray/tests/test_blas.py:        f = theano.function([x, y], z, mode=mode_with_gpu)
theano/gpuarray/tests/test_blas.py:class TestGpuSger(TestGer):
theano/gpuarray/tests/test_blas.py:        self.mode = mode_with_gpu
theano/gpuarray/tests/test_blas.py:        self.ger_destructive = gpuger_inplace
theano/gpuarray/tests/test_blas.py:        # data on the gpu make the op always inplace
theano/gpuarray/tests/test_blas.py:        self.ger = gpuger_inplace
theano/gpuarray/tests/test_blas.py:        self.gemm = gpugemm_inplace
theano/gpuarray/tests/test_blas.py:class TestGpuSgerNoTransfer(TestGpuSger):
theano/gpuarray/tests/test_blas.py:    shared = staticmethod(gpuarray_shared_constructor)
theano/gpuarray/tests/test_blas.py:class TestGpuGer_OpContract(TestCase, utt.T_OpContractMixin):
theano/gpuarray/tests/test_blas.py:        self.ops = [gpuger_no_inplace, gpuger_inplace]
theano/gpuarray/tests/test_blas.py:        return GpuGer(inplace=op.inplace)
theano/gpuarray/tests/test_blas.py:GpuDot22Tester = makeTester(
theano/gpuarray/tests/test_blas.py:    'GpuDot22Tester',
theano/gpuarray/tests/test_blas.py:    op=_dot22, gpu_op=gpu_dot22,
theano/gpuarray/tests/test_blas.py:    f = theano.function([W, v], W.dot(v), mode=mode_with_gpu)
theano/gpuarray/tests/test_blas.py:    x = gpuarray_shared_constructor(xv)
theano/gpuarray/tests/test_blas.py:    y = gpuarray_shared_constructor(yv, broadcastable=(False, True))
theano/gpuarray/tests/test_blas.py:    f = theano.function([], tensor.dot(x, y[::-1]), mode=mode_with_gpu)
theano/gpuarray/tests/test_type.py:from .test_basic_ops import rand_gpuarray
theano/gpuarray/tests/test_type.py:from ..type import GpuArrayType, gpuarray_shared_constructor
theano/gpuarray/tests/test_type.py:import pygpu
theano/gpuarray/tests/test_type.py:        a = rand_gpuarray(20, dtype=dtype)
theano/gpuarray/tests/test_type.py:        g = GpuArrayType(dtype=dtype, broadcastable=(False,))('g')
theano/gpuarray/tests/test_type.py:        assert GpuArrayType.values_eq(res, a)
theano/gpuarray/tests/test_type.py:        a = rand_gpuarray(20, dtype=dtype)
theano/gpuarray/tests/test_type.py:        g = GpuArrayType(dtype=dtype, broadcastable=(False,))('g')
theano/gpuarray/tests/test_type.py:        assert GpuArrayType.values_eq(res, a)
theano/gpuarray/tests/test_type.py:        a = rand_gpuarray(1, dtype=dtype)
theano/gpuarray/tests/test_type.py:        g = GpuArrayType(dtype=dtype, broadcastable=(False,))('g')
theano/gpuarray/tests/test_type.py:        assert GpuArrayType.values_eq(res, a)
theano/gpuarray/tests/test_type.py:    a = rand_gpuarray(20, dtype='float32')
theano/gpuarray/tests/test_type.py:    assert GpuArrayType.values_eq_approx(a, a)
theano/gpuarray/tests/test_type.py:    assert not GpuArrayType.values_eq_approx(a, b)
theano/gpuarray/tests/test_type.py:    assert not GpuArrayType.values_eq_approx(a, b)
theano/gpuarray/tests/test_type.py:        a = rand_gpuarray(20, dtype=dtype)
theano/gpuarray/tests/test_type.py:        g = GpuArrayType(dtype=dtype, broadcastable=(False,))('g')
theano/gpuarray/tests/test_type.py:    theano.compile.shared_constructor(gpuarray_shared_constructor)
theano/gpuarray/tests/test_type.py:    gpu_row = GpuArrayType(dtype=theano.config.floatX,
theano/gpuarray/tests/test_type.py:    gpu_matrix = GpuArrayType(dtype=theano.config.floatX,
theano/gpuarray/tests/test_type.py:    r = gpu_row()
theano/gpuarray/tests/test_type.py:    m = gpu_matrix.filter_variable(r)
theano/gpuarray/tests/test_type.py:    assert m.type == gpu_matrix
theano/gpuarray/tests/test_type.py:    m = gpu_matrix.filter_variable(r)
theano/gpuarray/tests/test_type.py:    assert m.type == gpu_matrix
theano/gpuarray/tests/test_type.py:def test_gpuarray_shared_scalar():
theano/gpuarray/tests/test_type.py:    # By default, we don't put scalar as shared variable on the GPU
theano/gpuarray/tests/test_type.py:        TypeError, gpuarray_shared_constructor, np.asarray(1, dtype='float32'))
theano/gpuarray/tests/test_type.py:    gpuarray_shared_constructor(np.asarray(1, dtype='float32'),
theano/gpuarray/tests/test_type.py:def test_unpickle_gpuarray_as_numpy_ndarray_flag0():
theano/gpuarray/tests/test_type.py:    # Test when pygpu isn't there for unpickle are in test_pickle.py
theano/gpuarray/tests/test_type.py:    oldflag = config.experimental.unpickle_gpu_on_cpu
theano/gpuarray/tests/test_type.py:    config.experimental.unpickle_gpu_on_cpu = False
theano/gpuarray/tests/test_type.py:        fname = 'GpuArray.pkl'
theano/gpuarray/tests/test_type.py:            assert isinstance(mat, pygpu.gpuarray.GpuArray)
theano/gpuarray/tests/test_type.py:        config.experimental.unpickle_gpu_on_cpu = oldflag
theano/gpuarray/tests/test_type.py:    shared_constructor_=gpuarray_shared_constructor,
theano/gpuarray/tests/test_type.py:    internal_type_=lambda v: pygpu.array(v, context=get_context(test_ctx_name),
theano/gpuarray/tests/test_type.py:                                         cls=pygpu._array.ndgpuarray),
theano/gpuarray/tests/test_type.py:    test_internal_type_=lambda a: isinstance(a, pygpu.gpuarray.GpuArray),
theano/gpuarray/tests/test_type.py:    cast_value_=lambda v: pygpu.array(v, context=get_context(test_ctx_name),
theano/gpuarray/tests/test_type.py:                                      cls=pygpu._array.ndgpuarray))
theano/gpuarray/tests/test_type.py:    shared_constructor_=gpuarray_shared_constructor,
theano/gpuarray/tests/test_type.py:    internal_type_=lambda v: pygpu.array(v, context=get_context(test_ctx_name),
theano/gpuarray/tests/test_type.py:                                         cls=pygpu._array.ndgpuarray),
theano/gpuarray/tests/test_type.py:    test_internal_type_=lambda a: isinstance(a, pygpu.gpuarray.GpuArray),
theano/gpuarray/tests/test_type.py:    cast_value_=lambda v: pygpu.array(v, context=get_context(test_ctx_name),
theano/gpuarray/tests/test_type.py:                                      cls=pygpu._array.ndgpuarray))
theano/gpuarray/tests/test_type.py:    s = gpuarray_shared_constructor(
theano/gpuarray/tests/test_scan.py:from ..basic_ops import GpuFromHost, HostFromGpu
theano/gpuarray/tests/test_scan.py:from ..elemwise import GpuElemwise
theano/gpuarray/tests/test_scan.py:from .config import mode_with_gpu, test_ctx_name
theano/gpuarray/tests/test_scan.py:    def test_one_sequence_one_output_weights_gpu1(self):
theano/gpuarray/tests/test_scan.py:        mode = mode_with_gpu.excluding('InputToGpuOptimizer')
theano/gpuarray/tests/test_scan.py:        output = GpuFromHost(test_ctx_name)(output)
theano/gpuarray/tests/test_scan.py:        assert sum([isinstance(node.op, HostFromGpu)
theano/gpuarray/tests/test_scan.py:        assert sum([isinstance(node.op, GpuFromHost)
theano/gpuarray/tests/test_scan.py:        # check that there is no gpu transfer in the inner loop.
theano/gpuarray/tests/test_scan.py:        assert any([isinstance(node.op, GpuElemwise)
theano/gpuarray/tests/test_scan.py:        assert not any([isinstance(node.op, HostFromGpu)
theano/gpuarray/tests/test_scan.py:        assert not any([isinstance(node.op, GpuFromHost)
theano/gpuarray/tests/test_scan.py:    # This second version test the second case in the optimizer to the gpu.
theano/gpuarray/tests/test_scan.py:    def test_one_sequence_one_output_weights_gpu2(self):
theano/gpuarray/tests/test_scan.py:                                      mode=mode_with_gpu)
theano/gpuarray/tests/test_scan.py:                             mode=mode_with_gpu)
theano/gpuarray/tests/test_scan.py:        assert sum([isinstance(node.op, HostFromGpu)
theano/gpuarray/tests/test_scan.py:        assert sum([isinstance(node.op, GpuFromHost)
theano/gpuarray/tests/test_scan.py:        # check that there is no gpu transfer in the inner loop.
theano/gpuarray/tests/test_scan.py:        assert any([isinstance(node.op, GpuElemwise)
theano/gpuarray/tests/test_scan.py:        assert not any([isinstance(node.op, HostFromGpu)
theano/gpuarray/tests/test_scan.py:        assert not any([isinstance(node.op, GpuFromHost)
theano/gpuarray/tests/test_scan.py:    # outputs when is running on GPU
theano/gpuarray/tests/test_scan.py:    def test_gpu3_mixture_dtype_outputs(self):
theano/gpuarray/tests/test_scan.py:                                      mode=mode_with_gpu)
theano/gpuarray/tests/test_scan.py:                             mode=mode_with_gpu)
theano/gpuarray/tests/test_scan.py:        assert scan_node.op.gpua
theano/gpuarray/tests/test_scan.py:        # check that there is no gpu transfer in the inner loop.
theano/gpuarray/tests/test_scan.py:        assert not any([isinstance(node.op, HostFromGpu)
theano/gpuarray/tests/test_scan.py:        assert not any([isinstance(node.op, GpuFromHost)
theano/gpuarray/tests/test_scan.py:    def test_gpu4_gibbs_chain(self):
theano/gpuarray/tests/test_scan.py:                                               mode=mode_with_gpu)
theano/gpuarray/tests/test_scan.py:                               mode=mode_with_gpu)
theano/gpuarray/tests/test_ctc.py:import theano.gpuarray
theano/gpuarray/tests/test_ctc.py:from theano.gpuarray.ctc import (gpu_ctc, GpuConnectionistTemporalClassification)
theano/gpuarray/tests/test_ctc.py:from .config import (mode_with_gpu, mode_without_gpu)
theano/gpuarray/tests/test_ctc.py:        self.compare_gpu_and_cpu_values(*inputs)
theano/gpuarray/tests/test_ctc.py:        self.run_gpu_optimization_with_grad(*inputs)
theano/gpuarray/tests/test_ctc.py:        self.run_gpu_optimization_no_grad(*inputs)
theano/gpuarray/tests/test_ctc.py:    def setup_cpu_op(self, activations, labels, input_length, compute_grad=True, mode=mode_without_gpu):
theano/gpuarray/tests/test_ctc.py:    def setup_gpu_op(self, activations, labels, input_length, compute_grad=True):
theano/gpuarray/tests/test_ctc.py:        gpu_ctc_cost = gpu_ctc(activations, labels, input_length)
theano/gpuarray/tests/test_ctc.py:        outputs = [gpu_ctc_cost]
theano/gpuarray/tests/test_ctc.py:            gpu_ctc_grad = T.grad(T.mean(gpu_ctc_cost), activations)
theano/gpuarray/tests/test_ctc.py:            outputs += [gpu_ctc_grad]
theano/gpuarray/tests/test_ctc.py:        return theano.function([], outputs, mode=mode_with_gpu)
theano/gpuarray/tests/test_ctc.py:        gpu_train = self.setup_gpu_op(activations, labels, input_length)
theano/gpuarray/tests/test_ctc.py:        gpu_cost, gpu_grad = gpu_train()
theano/gpuarray/tests/test_ctc.py:        # Transfer costs from GPU memory to host
theano/gpuarray/tests/test_ctc.py:        cost_from_gpu = np.asarray(gpu_cost)
theano/gpuarray/tests/test_ctc.py:        # Transfer gradients from GPU memory to host
theano/gpuarray/tests/test_ctc.py:        grad_from_gpu = np.asarray(gpu_grad)
theano/gpuarray/tests/test_ctc.py:        utt.assert_allclose(expected_grads / cost_from_gpu.shape[0], grad_from_gpu)
theano/gpuarray/tests/test_ctc.py:        utt.assert_allclose(expected_costs, cost_from_gpu)
theano/gpuarray/tests/test_ctc.py:    def compare_gpu_and_cpu_values(self, activations, labels, input_length):
theano/gpuarray/tests/test_ctc.py:        gpu_train = self.setup_gpu_op(activations, labels, input_length)
theano/gpuarray/tests/test_ctc.py:        gpu_cost, gpu_grad = gpu_train()
theano/gpuarray/tests/test_ctc.py:        # Transfer costs from GPU memory to host
theano/gpuarray/tests/test_ctc.py:        cost_from_gpu = np.asarray(gpu_cost)
theano/gpuarray/tests/test_ctc.py:        # Transfer gradients from GPU memory to host
theano/gpuarray/tests/test_ctc.py:        grad_from_gpu = np.asarray(gpu_grad)
theano/gpuarray/tests/test_ctc.py:        utt.assert_allclose(cpu_grad, grad_from_gpu)
theano/gpuarray/tests/test_ctc.py:        utt.assert_allclose(cpu_cost, cost_from_gpu)
theano/gpuarray/tests/test_ctc.py:        gpu_ctc_cost = gpu_ctc(activations, labels, input_length)
theano/gpuarray/tests/test_ctc.py:        gpu_ctc_function = theano.function([], [gpu_ctc_cost])
theano/gpuarray/tests/test_ctc.py:        for node in gpu_ctc_function.maker.fgraph.apply_nodes:
theano/gpuarray/tests/test_ctc.py:            if isinstance(node.op, GpuConnectionistTemporalClassification):
theano/gpuarray/tests/test_ctc.py:    def run_gpu_optimization_with_grad(self, activations, labels, input_length):
theano/gpuarray/tests/test_ctc.py:        cpu_lifted_train = self.setup_cpu_op(activations, labels, input_length, mode=mode_with_gpu)
theano/gpuarray/tests/test_ctc.py:        # Check whether Op is lifted to the GPU
theano/gpuarray/tests/test_ctc.py:        assert self.has_only_gpu_op(cpu_lifted_train)
theano/gpuarray/tests/test_ctc.py:    def run_gpu_optimization_no_grad(self, activations, labels, input_length):
theano/gpuarray/tests/test_ctc.py:        cpu_lifted_test = self.setup_cpu_op(activations, labels, input_length, compute_grad=False, mode=mode_with_gpu)
theano/gpuarray/tests/test_ctc.py:        # Check whether Op is lifted to the GPU
theano/gpuarray/tests/test_ctc.py:        assert self.has_only_gpu_op(cpu_lifted_test)
theano/gpuarray/tests/test_ctc.py:        gpu_cost = cpu_lifted_test()
theano/gpuarray/tests/test_ctc.py:        # Transfer costs from GPU memory to host
theano/gpuarray/tests/test_ctc.py:        cost_from_gpu = np.asarray(gpu_cost)
theano/gpuarray/tests/test_ctc.py:        # Compare values from CPU and GPU Ops
theano/gpuarray/tests/test_ctc.py:        utt.assert_allclose(cpu_cost, cost_from_gpu)
theano/gpuarray/tests/test_ctc.py:    def has_only_gpu_op(self, function):
theano/gpuarray/tests/test_ctc.py:        has_gpu_instance = False
theano/gpuarray/tests/test_ctc.py:            if isinstance(node.op, GpuConnectionistTemporalClassification):
theano/gpuarray/tests/test_ctc.py:                has_gpu_instance = True
theano/gpuarray/tests/test_ctc.py:        return has_gpu_instance and (not has_cpu_instance)
theano/gpuarray/tests/test_ctc.py:                return gpu_ctc(acts, t_labels, t_activation_times)
theano/gpuarray/tests/test_ctc.py:        utt.verify_grad(ctc_op, [activations], mode=mode_with_gpu)
theano/gpuarray/tests/test_multinomial.py:from .config import mode_with_gpu
theano/gpuarray/tests/test_multinomial.py:from ..multinomial import (GPUAMultinomialFromUniform,
theano/gpuarray/tests/test_multinomial.py:                           GPUAChoiceFromUniform)
theano/gpuarray/tests/test_multinomial.py:    # multinomial() call in GPU random generation.
theano/gpuarray/tests/test_multinomial.py:        f = function([p, u], m * 2, allow_input_downcast=True, mode=mode_with_gpu)
theano/gpuarray/tests/test_multinomial.py:        assert any([type(node.op) is GPUAMultinomialFromUniform
theano/gpuarray/tests/test_multinomial.py:        # and also make sure that the GPU version doesn't screw up the
theano/gpuarray/tests/test_multinomial.py:    # multinomial() call in GPU random generation.
theano/gpuarray/tests/test_multinomial.py:            f = function([p, u], m * 2, allow_input_downcast=True, mode=mode_with_gpu)
theano/gpuarray/tests/test_multinomial.py:            assert any([type(node.op) is GPUAMultinomialFromUniform
theano/gpuarray/tests/test_multinomial.py:            # and also make sure that the GPU version doesn't screw up the
theano/gpuarray/tests/test_multinomial.py:# TODO: check a bigger example (make sure blocking on GPU is handled correctly)
theano/gpuarray/tests/test_multinomial.py:    # DEBUG_MODE will test this on GPU
theano/gpuarray/tests/test_multinomial.py:    f = function([p, u], m * 2, allow_input_downcast=True, mode=mode_with_gpu)
theano/gpuarray/tests/test_multinomial.py:    assert any([type(node.op) is GPUAMultinomialFromUniform
theano/gpuarray/tests/test_multinomial.py:def test_gpu_opt_dtypes():
theano/gpuarray/tests/test_multinomial.py:        f = function([p, u], m, allow_input_downcast=True, mode=mode_with_gpu)
theano/gpuarray/tests/test_multinomial.py:        assert any([type(node.op) is GPUAMultinomialFromUniform
theano/gpuarray/tests/test_multinomial.py:def test_gpu_opt():
theano/gpuarray/tests/test_multinomial.py:    # We test the case where we put the op on the gpu when the output
theano/gpuarray/tests/test_multinomial.py:    # is moved to the gpu.
theano/gpuarray/tests/test_multinomial.py:    f = function([p, u], m, allow_input_downcast=True, mode=mode_with_gpu)
theano/gpuarray/tests/test_multinomial.py:    assert any([type(node.op) is GPUAMultinomialFromUniform
theano/gpuarray/tests/test_multinomial.py:    f = function([r, u], m, allow_input_downcast=True, mode=mode_with_gpu)
theano/gpuarray/tests/test_multinomial.py:    assert any([type(node.op) is GPUAMultinomialFromUniform
theano/gpuarray/tests/test_multinomial.py:def test_gpu_opt_wor():
theano/gpuarray/tests/test_multinomial.py:    # We test the case where we put the op on the gpu when the output
theano/gpuarray/tests/test_multinomial.py:    # is moved to the gpu.
theano/gpuarray/tests/test_multinomial.py:                     mode=mode_with_gpu)
theano/gpuarray/tests/test_multinomial.py:        assert any([type(node.op) is GPUAChoiceFromUniform
theano/gpuarray/tests/test_multinomial.py:                     mode=mode_with_gpu)
theano/gpuarray/tests/test_multinomial.py:        assert any([type(node.op) is GPUAChoiceFromUniform
theano/gpuarray/tests/test_multinomial.py:    fname = 'test_gpuarray_multinomial_wo_replacement.pkl'
theano/gpuarray/tests/test_multinomial.py:            assert isinstance(m, GPUAChoiceFromUniform)
theano/gpuarray/tests/test_rng_mrg.py:from .config import mode_with_gpu as mode
theano/gpuarray/tests/test_rng_mrg.py:from ..type import gpuarray_shared_constructor
theano/gpuarray/tests/test_rng_mrg.py:from ..rng_mrg import GPUA_mrg_uniform
theano/gpuarray/tests/test_rng_mrg.py:def test_consistency_GPUA_serial():
theano/gpuarray/tests/test_rng_mrg.py:    # Verify that the random numbers generated by GPUA_mrg_uniform, serially,
theano/gpuarray/tests/test_rng_mrg.py:            rstate = gpuarray_shared_constructor(substream_rstate)
theano/gpuarray/tests/test_rng_mrg.py:            new_rstate, sample = GPUA_mrg_uniform.new(rstate,
theano/gpuarray/tests/test_rng_mrg.py:def test_consistency_GPUA_parallel():
theano/gpuarray/tests/test_rng_mrg.py:    # Verify that the random numbers generated by GPUA_mrg_uniform, in
theano/gpuarray/tests/test_rng_mrg.py:        rstate = gpuarray_shared_constructor(rstate)
theano/gpuarray/tests/test_rng_mrg.py:        new_rstate, sample = GPUA_mrg_uniform.new(rstate, ndim=None,
theano/gpuarray/tests/test_rng_mrg.py:def test_GPUA_full_fill():
theano/gpuarray/tests/test_rng_mrg.py:    # This needs to be large to trigger the problem on GPU
theano/gpuarray/tests/test_rng_mrg.py:    rstate_gpu = gpuarray_shared_constructor(R.state_updates[-1][0].get_value())
theano/gpuarray/tests/test_rng_mrg.py:    new_rstate, sample = GPUA_mrg_uniform.new(rstate_gpu, ndim=None,
theano/gpuarray/tests/test_rng_mrg.py:    rstate_gpu.default_update = new_rstate
theano/gpuarray/tests/test_rng_mrg.py:    f_gpu = theano.function([], sample, mode=mode)
theano/gpuarray/tests/test_rng_mrg.py:    utt.assert_allclose(f_cpu(), f_gpu())
theano/gpuarray/tests/test_rng_mrg.py:def test_overflow_gpu_new_backend():
theano/gpuarray/tests/test_rng_mrg.py:    rstate = gpuarray_shared_constructor(rstate)
theano/gpuarray/tests/test_rng_mrg.py:    fct = functools.partial(GPUA_mrg_uniform.new, rstate,
theano/gpuarray/tests/test_rng_mrg.py:def test_validate_input_types_gpuarray_backend():
theano/gpuarray/tests/test_rng_mrg.py:        rstate = gpuarray_shared_constructor(rstate)
theano/gpuarray/tests/test_rng_mrg.py:        # To have theano.shared(x) try to move on the GPU
theano/gpuarray/tests/test_rng_mrg.py:        theano.compile.shared_constructor(gpuarray_shared_constructor)
theano/gpuarray/tests/test_rng_mrg.py:        cpu_f16_nonzero(mode=mode, op_to_check=GPUA_mrg_uniform)
theano/gpuarray/tests/test_rng_mrg.py:        theano.compile.shared_constructor(gpuarray_shared_constructor,
theano/gpuarray/tests/test_rng_mrg.py:    x = gpuarray_shared_constructor(s, name='x')
theano/gpuarray/tests/test_rng_mrg.py:        # To have theano.shared(x) try to move on the GPU
theano/gpuarray/tests/test_rng_mrg.py:        theano.compile.shared_constructor(gpuarray_shared_constructor)
theano/gpuarray/tests/test_rng_mrg.py:        assert not any([isinstance(node.op, GPUA_mrg_uniform) for node in nodes])
theano/gpuarray/tests/test_rng_mrg.py:        theano.compile.shared_constructor(gpuarray_shared_constructor,
theano/gpuarray/tests/run_dnn_conv.py:from theano.gpuarray.cudnn_defs import (HALF, FLOAT, DOUBLE,
theano/gpuarray/tests/run_dnn_conv.py:from theano.gpuarray.tests.check_dnn_conv import (cudnn, TestDnnConv2D, TestDnnConv3D, CheckDnn)
theano/gpuarray/tests/test_gemmcorr.py:from ..type import gpuarray_shared_constructor
theano/gpuarray/tests/test_gemmcorr.py:from ..blas import GpuCorrMM, GpuCorrMM_gradWeights, GpuCorrMM_gradInputs
theano/gpuarray/tests/test_gemmcorr.py:from .config import mode_with_gpu, mode_without_gpu, ref_cast
theano/gpuarray/tests/test_gemmcorr.py:        inputs = gpuarray_shared_constructor(inputs_val)
theano/gpuarray/tests/test_gemmcorr.py:        filters = gpuarray_shared_constructor(filters_val)
theano/gpuarray/tests/test_gemmcorr.py:        f_ref = theano.function([], conv_ref, mode=mode_without_gpu)
theano/gpuarray/tests/test_gemmcorr.py:        conv = GpuCorrMM(border_mode=border_mode,
theano/gpuarray/tests/test_gemmcorr.py:        f = theano.function([], conv, mode=mode_with_gpu)
theano/gpuarray/tests/test_gemmcorr.py:            utt.verify_grad(GpuCorrMM(border_mode=border_mode,
theano/gpuarray/tests/test_gemmcorr.py:                            [inputs_val, filters_val], mode=mode_with_gpu)
theano/gpuarray/tests/test_gemmcorr.py:        inputs = gpuarray_shared_constructor(inputs_val)
theano/gpuarray/tests/test_gemmcorr.py:        dCdH = gpuarray_shared_constructor(dCdH_val)
theano/gpuarray/tests/test_gemmcorr.py:        shape = gpuarray_shared_constructor(np.array(filters_shape[2:]))
theano/gpuarray/tests/test_gemmcorr.py:            conv_gemm = GpuCorrMM_gradWeights(subsample=subsample)(
theano/gpuarray/tests/test_gemmcorr.py:            conv_gemm = GpuCorrMM_gradWeights(subsample=subsample)(
theano/gpuarray/tests/test_gemmcorr.py:        f_ref = theano.function([], conv_ref, mode=mode_without_gpu)
theano/gpuarray/tests/test_gemmcorr.py:        f = theano.function([], conv_gemm, mode=mode_with_gpu)
theano/gpuarray/tests/test_gemmcorr.py:        inputs = gpuarray_shared_constructor(inputs_val)
theano/gpuarray/tests/test_gemmcorr.py:        filters = gpuarray_shared_constructor(filters_val)
theano/gpuarray/tests/test_gemmcorr.py:        bottom_shape = gpuarray_shared_constructor(np.array([bottom_height, bottom_width]))
theano/gpuarray/tests/test_gemmcorr.py:            conv_gemm = GpuCorrMM_gradInputs(subsample=subsample)(
theano/gpuarray/tests/test_gemmcorr.py:            conv_gemm = GpuCorrMM_gradInputs(subsample=subsample)(
theano/gpuarray/tests/test_gemmcorr.py:        f_ref = theano.function([], conv_ref, mode=mode_without_gpu)
theano/gpuarray/tests/test_gemmcorr.py:        f = theano.function([], conv_gemm, mode=mode_with_gpu)
theano/gpuarray/tests/test_gemmcorr.py:class TestGroupGpuCorr2d(Grouped_conv_noOptim):
theano/gpuarray/tests/test_gemmcorr.py:    mode = mode_with_gpu.excluding('cudnn')
theano/gpuarray/tests/test_gemmcorr.py:    conv_op = GpuCorrMM
theano/gpuarray/tests/test_gemmcorr.py:    conv_gradw_op = GpuCorrMM_gradWeights
theano/gpuarray/tests/test_gemmcorr.py:    conv_gradi_op = GpuCorrMM_gradInputs
theano/gpuarray/tests/test_gemmcorr.py:class TestUnsharedGpuCorr2d(TestUnsharedConv):
theano/gpuarray/tests/test_gemmcorr.py:    mode = mode_with_gpu
theano/gpuarray/tests/test_gemmcorr.py:    conv2d_op = GpuCorrMM
theano/gpuarray/tests/test_gemmcorr.py:    conv2d_gradw_op = GpuCorrMM_gradWeights
theano/gpuarray/tests/test_gemmcorr.py:    conv2d_gradi_op = GpuCorrMM_gradInputs
theano/gpuarray/tests/test_gemmcorr.py:class TestAsymmetricGpu(TestAsymmetricPadding):
theano/gpuarray/tests/test_gemmcorr.py:    mode = mode_with_gpu
theano/gpuarray/tests/test_gemmcorr.py:    conv2d_op = GpuCorrMM
theano/gpuarray/tests/test_gemmcorr.py:    conv2d_gradw_op = GpuCorrMM_gradWeights
theano/gpuarray/tests/test_gemmcorr.py:    conv2d_gradi_op = GpuCorrMM_gradInputs
theano/gpuarray/tests/test_gemmcorr.py:class TestCausalGpuCorr(TestCausalConv):
theano/gpuarray/tests/test_gemmcorr.py:    mode = mode_with_gpu
theano/gpuarray/tests/test_blocksparse.py:from .config import mode_with_gpu, test_ctx_name
theano/gpuarray/tests/test_blocksparse.py:from ..type import gpuarray_shared_constructor
theano/gpuarray/tests/test_blocksparse.py:from ..blocksparse import (GpuSparseBlockGemv,
theano/gpuarray/tests/test_blocksparse.py:                           GpuSparseBlockOuter,
theano/gpuarray/tests/test_blocksparse.py:                           gpu_sparse_block_gemv,
theano/gpuarray/tests/test_blocksparse.py:                           gpu_sparse_block_outer)
theano/gpuarray/tests/test_blocksparse.py:        self.mode = mode_with_gpu.excluding('constant_folding')
theano/gpuarray/tests/test_blocksparse.py:        self.gemv_op = gpu_sparse_block_gemv
theano/gpuarray/tests/test_blocksparse.py:        self.outer_op = gpu_sparse_block_outer
theano/gpuarray/tests/test_blocksparse.py:        self.gemv_class = GpuSparseBlockGemv
theano/gpuarray/tests/test_blocksparse.py:        self.outer_class = GpuSparseBlockOuter
theano/gpuarray/tests/test_blocksparse.py:        W = gpuarray_shared_constructor(W_val, context=test_ctx_name)
theano/gpuarray/tests/test_blocksparse.py:        o = gpu_sparse_block_gemv(b.take(oIdx, axis=0), W, h, iIdx, oIdx)
theano/gpuarray/tests/test_blocksparse.py:                             mode=mode_with_gpu)
theano/gpuarray/tests/test_blocksparse.py:                          GpuSparseBlockOuter)
theano/gpuarray/tests/test_blocksparse.py:        mode = mode_with_gpu.excluding('local_merge_blocksparse_alpha')
theano/gpuarray/tests/test_blocksparse.py:                              GpuSparseBlockOuter)
theano/gpuarray/tests/test_pool.py:from .config import mode_with_gpu, mode_without_gpu
theano/gpuarray/tests/test_pool.py:from ..pool import (GpuPool, GpuMaxPoolGrad, GpuAveragePoolGrad,
theano/gpuarray/tests/test_pool.py:                    GpuDownsampleFactorMaxGradGrad)
theano/gpuarray/tests/test_pool.py:            ds_op = GpuPool(ignore_border=True, ndim=2)
theano/gpuarray/tests/test_pool.py:            ds_op = GpuPool(ignore_border=False, ndim=2)
theano/gpuarray/tests/test_pool.py:        gpu_mode = mode_with_gpu.excluding("cudnn")
theano/gpuarray/tests/test_pool.py:        gpu_mode.check_py_code = False
theano/gpuarray/tests/test_pool.py:            ds_op = GpuPool(ignore_border=False, ndim=2)
theano/gpuarray/tests/test_pool.py:            f = theano.function([], ds_op(inp, [2, 2], pad=pad), mode=gpu_mode)
theano/gpuarray/tests/test_pool.py:        gpu_mode = mode_with_gpu.excluding("cudnn")
theano/gpuarray/tests/test_pool.py:        gpu_mode.check_py_code = False
theano/gpuarray/tests/test_pool.py:        ds_op = GpuPool(ignore_border=False, mode='average_exc_pad', ndim=2)
theano/gpuarray/tests/test_pool.py:                            mode=gpu_mode)
theano/gpuarray/tests/test_pool.py:    ref_mode = copy.copy(mode_without_gpu)
theano/gpuarray/tests/test_pool.py:    gpu_mode = mode_with_gpu.excluding("cudnn")
theano/gpuarray/tests/test_pool.py:    gpu_mode.check_py_code = False
theano/gpuarray/tests/test_pool.py:                f = theano.function([], a_pooled, mode=gpu_mode)
theano/gpuarray/tests/test_pool.py:                assert any([isinstance(node.op, GpuPool)
theano/gpuarray/tests/test_pool.py:                g = theano.function([], a_pooled_grad, mode=gpu_mode)
theano/gpuarray/tests/test_pool.py:                    gop = GpuMaxPoolGrad
theano/gpuarray/tests/test_pool.py:                    gop = GpuAveragePoolGrad
theano/gpuarray/tests/test_pool.py:                gr = theano.function([], tensor.Rop(a_pooled, a, ea), mode=gpu_mode)
theano/gpuarray/tests/test_pool.py:                    isinstance(node.op, GpuDownsampleFactorMaxGradGrad)
theano/gpuarray/tests/test_pool.py:                gg = theano.function([], ggf, mode=gpu_mode)
theano/gpuarray/tests/test_pool.py:                    isinstance(node.op, GpuDownsampleFactorMaxGradGrad)
theano/gpuarray/tests/test_pool.py:    ref_mode = copy.copy(mode_without_gpu)
theano/gpuarray/tests/test_pool.py:    gpu_mode = mode_with_gpu.excluding("cudnn")
theano/gpuarray/tests/test_pool.py:    gpu_mode.check_py_code = False
theano/gpuarray/tests/test_pool.py:                f = theano.function([], a_pooled, mode=gpu_mode)
theano/gpuarray/tests/test_pool.py:                assert any([isinstance(node.op, GpuPool)
theano/gpuarray/tests/test_pool.py:                g = theano.function([], a_pooled_grad, mode=gpu_mode)
theano/gpuarray/tests/test_pool.py:                    gop = GpuMaxPoolGrad
theano/gpuarray/tests/test_pool.py:                    gop = GpuAveragePoolGrad
theano/gpuarray/tests/test_pool.py:                gr = theano.function([], tensor.Rop(a_pooled, a, ea), mode=gpu_mode)
theano/gpuarray/tests/test_pool.py:                    isinstance(node.op, GpuDownsampleFactorMaxGradGrad)
theano/gpuarray/tests/test_pool.py:                gg = theano.function([], ggf, mode=gpu_mode)
theano/gpuarray/tests/test_pool.py:                    isinstance(node.op, GpuDownsampleFactorMaxGradGrad)
theano/gpuarray/c_code/topk_common.cuh:#if __CUDA_ARCH__ < 350
theano/gpuarray/c_code/topk_common.cuh: * TODO: Maybe should gpuarray provide abstract functions to manipulate ga_half internal structure? e.g:
theano/gpuarray/c_code/topk_common.cuh:// we cannot use templated kernel because gpuarray API does not support it
theano/gpuarray/c_code/dnn_rnn_fwd.c:                PyGpuArrayObject *w, PyGpuArrayObject *x,
theano/gpuarray/c_code/dnn_rnn_fwd.c:                PyGpuArrayObject *hx, PyGpuArrayObject *cx,
theano/gpuarray/c_code/dnn_rnn_fwd.c:                gpudata **reserve, PyGpuArrayObject **y,
theano/gpuarray/c_code/dnn_rnn_fwd.c:                PyGpuArrayObject **hy, PyGpuArrayObject **cy,
theano/gpuarray/c_code/dnn_rnn_fwd.c:  PyGpuContextObject *c = x->context;
theano/gpuarray/c_code/dnn_rnn_fwd.c:  gpudata *workspace = NULL;
theano/gpuarray/c_code/dnn_rnn_fwd.c:  size_t seqLength = PyGpuArray_DIM(x, 0);
theano/gpuarray/c_code/dnn_rnn_fwd.c:  size_t miniBatch = PyGpuArray_DIM(x, 1);
theano/gpuarray/c_code/dnn_rnn_fwd.c:  size_t inputSize = PyGpuArray_DIM(x, 2);
theano/gpuarray/c_code/dnn_rnn_fwd.c:  size_t hiddenSize = PyGpuArray_DIM(hx, 2);
theano/gpuarray/c_code/dnn_rnn_fwd.c:  cuda_enter(c->ctx);
theano/gpuarray/c_code/dnn_rnn_fwd.c:  dims[0] = PyGpuArray_DIM(x, 1);
theano/gpuarray/c_code/dnn_rnn_fwd.c:  dims[1] = PyGpuArray_DIM(x, 2);
theano/gpuarray/c_code/dnn_rnn_fwd.c:  if (theano_prep_output(hy, 3, PyGpuArray_DIMS(hx),
theano/gpuarray/c_code/dnn_rnn_fwd.c:    if (theano_prep_output(cy, 3, PyGpuArray_DIMS(cx),
theano/gpuarray/c_code/dnn_rnn_fwd.c:  workspace = gpudata_alloc(c->ctx, worksize, NULL, 0, NULL);
theano/gpuarray/c_code/dnn_rnn_fwd.c:  *reserve = gpudata_alloc(c->ctx, ressize, NULL, 0, NULL);
theano/gpuarray/c_code/dnn_rnn_fwd.c:                                xl, PyGpuArray_DEV_DATA(x),
theano/gpuarray/c_code/dnn_rnn_fwd.c:                                hxdesc, PyGpuArray_DEV_DATA(hx),
theano/gpuarray/c_code/dnn_rnn_fwd.c:                                cxdesc, cx ? PyGpuArray_DEV_DATA(cx) : NULL,
theano/gpuarray/c_code/dnn_rnn_fwd.c:                                wdesc, PyGpuArray_DEV_DATA(w),
theano/gpuarray/c_code/dnn_rnn_fwd.c:                                yl, PyGpuArray_DEV_DATA(*y),
theano/gpuarray/c_code/dnn_rnn_fwd.c:                                hydesc, PyGpuArray_DEV_DATA(*hy),
theano/gpuarray/c_code/dnn_rnn_fwd.c:                                cydesc, cy ? PyGpuArray_DEV_DATA(*cy) : NULL,
theano/gpuarray/c_code/dnn_rnn_fwd.c:    gpudata_release(workspace);
theano/gpuarray/c_code/dnn_rnn_fwd.c:  cuda_exit(c->ctx);
theano/gpuarray/c_code/dimshuffle.c:int APPLY_SPECIFIC(gpu_dimshuffle)(PyGpuArrayObject* input, PyGpuArrayObject** out, PARAMS_TYPE* params) {
theano/gpuarray/c_code/dimshuffle.c:    PyGpuArrayObject *tmp = NULL;
theano/gpuarray/c_code/dimshuffle.c:        PyErr_SetString(PyExc_RuntimeError, "GpuDimShuffle: param transposition must be C-contiguous.");
theano/gpuarray/c_code/dimshuffle.c:    tmp = pygpu_transpose(input, transposition);
theano/gpuarray/c_code/dimshuffle.c:    *out = pygpu_reshape(tmp, nd_out, sh, GA_ANY_ORDER, 1, -1);
theano/gpuarray/c_code/dimshuffle.c:        tmp = pygpu_copy(*out, GA_ANY_ORDER);
theano/gpuarray/c_code/dnn_dropout_desc.c:                     PyGpuContextObject *c,
theano/gpuarray/c_code/dnn_dropout_desc.c:                     PyGpuArrayObject **ostates,
theano/gpuarray/c_code/dnn_dropout_desc.c:  PyGpuArrayObject *states;
theano/gpuarray/c_code/dnn_dropout_desc.c:  cuda_enter(c->ctx);
theano/gpuarray/c_code/dnn_dropout_desc.c:    cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_dropout_desc.c:  states = pygpu_empty(1, &states_sz, GA_UBYTE, GA_C_ORDER, c, Py_None);
theano/gpuarray/c_code/dnn_dropout_desc.c:    cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_dropout_desc.c:                                  PyGpuArray_DEV_DATA(states),
theano/gpuarray/c_code/dnn_dropout_desc.c:    cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_dropout_desc.c:  cuda_exit(c->ctx);
theano/gpuarray/c_code/magma_inv.c:setup_ext_cuda();
theano/gpuarray/c_code/magma_inv.c:int APPLY_SPECIFIC(magma_inv)(PyGpuArrayObject *A, PyGpuArrayObject **A_inv,
theano/gpuarray/c_code/magma_inv.c:  gpudata *dwork = NULL;
theano/gpuarray/c_code/magma_inv.c:                    "GpuMagmaMatrixInverse: Unsupported data type");
theano/gpuarray/c_code/magma_inv.c:  cuda_enter(params->context->ctx);
theano/gpuarray/c_code/magma_inv.c:  if (!GpuArray_IS_C_CONTIGUOUS(&A->ga)) {
theano/gpuarray/c_code/magma_inv.c:                    "GpuMagmaMatrixInverse: requires data to be C-contiguous");
theano/gpuarray/c_code/magma_inv.c:  if (PyGpuArray_NDIM(A) != 2) {
theano/gpuarray/c_code/magma_inv.c:                    "GpuMagmaMatrixInverse: matrix rank error");
theano/gpuarray/c_code/magma_inv.c:  dims = PyGpuArray_DIMS(A);
theano/gpuarray/c_code/magma_inv.c:                    "GpuMagmaMatrixInverse: matrix is not square");
theano/gpuarray/c_code/magma_inv.c:          "GpuMagmaMatrixInverse: failed to allocate memory for the output");
theano/gpuarray/c_code/magma_inv.c:  dwork = gpudata_alloc(params->context->ctx, ldwork * sizeof(float), NULL, 0, NULL);
theano/gpuarray/c_code/magma_inv.c:                    "GpuMagmaMatrixInverse: failed to allocate working memory");
theano/gpuarray/c_code/magma_inv.c:        "GpuMagmaMatrixInverse: failed to allocate memory for the pivot array");
theano/gpuarray/c_code/magma_inv.c:  magma_sgetrf_gpu(N, N, (float *)PyGpuArray_DEV_DATA(*A_inv), N, piv, &info);
theano/gpuarray/c_code/magma_inv.c:        "GpuMagmaMatrixInverse: magma_sgetrf_gpu returned error %d: %s.", info,
theano/gpuarray/c_code/magma_inv.c:  magma_sgetri_gpu(N, (float *)PyGpuArray_DEV_DATA(*A_inv), N, piv,
theano/gpuarray/c_code/magma_inv.c:        "GpuMagmaMatrixInverse: magma_sgetri_gpu returned error %d: %s.", info,
theano/gpuarray/c_code/magma_inv.c:    gpudata_release(dwork);
theano/gpuarray/c_code/magma_inv.c:  cuda_exit(params->context->ctx);
theano/gpuarray/c_code/dnn_sptf_gi.c:            "GpuDnnTransformerGradI: could not allocate spatial transformer descriptor (sptf): %s",
theano/gpuarray/c_code/dnn_sptf_gi.c:            "GpuDnnTransformerGradI: failed to allocate cuDNN tensor descriptor xdesc: %s",
theano/gpuarray/c_code/dnn_sptf_gi.c:            "GpuDnnTransformerGradI: failed to allocate cuDNN tensor descriptor dxdesc: %s",
theano/gpuarray/c_code/dnn_sptf_gi.c:            "GpuDnnTransformerGradI: failed to allocate cuDNN tensor descriptor dydesc: %s",
theano/gpuarray/c_code/dnn_sptf_gi.c:APPLY_SPECIFIC(dnn_sptf_gi)(PyGpuArrayObject * input,
theano/gpuarray/c_code/dnn_sptf_gi.c:                            PyGpuArrayObject * grid,
theano/gpuarray/c_code/dnn_sptf_gi.c:                            PyGpuArrayObject * dy,
theano/gpuarray/c_code/dnn_sptf_gi.c:                            PyGpuArrayObject ** input_grad,
theano/gpuarray/c_code/dnn_sptf_gi.c:                            PyGpuArrayObject ** grid_grad,
theano/gpuarray/c_code/dnn_sptf_gi.c:    PyGpuContextObject * gpu_ctx = input->context;
theano/gpuarray/c_code/dnn_sptf_gi.c:            "GpuDnnTransformerGradI: unsupported type for input in spatial transformer gradients" );
theano/gpuarray/c_code/dnn_sptf_gi.c:            "GpuDnnTransformerGradI: unsupported data type for grid in spatial transformer gradients." );
theano/gpuarray/c_code/dnn_sptf_gi.c:    if ( theano_prep_output( input_grad, PyGpuArray_NDIM( input ),
theano/gpuarray/c_code/dnn_sptf_gi.c:                             PyGpuArray_DIMS( input ), input->ga.typecode,
theano/gpuarray/c_code/dnn_sptf_gi.c:                             GA_C_ORDER, gpu_ctx ) != 0 )
theano/gpuarray/c_code/dnn_sptf_gi.c:    if ( theano_prep_output( grid_grad, PyGpuArray_NDIM( grid ),
theano/gpuarray/c_code/dnn_sptf_gi.c:                             PyGpuArray_DIMS( grid ), grid->ga.typecode,
theano/gpuarray/c_code/dnn_sptf_gi.c:                             GA_C_ORDER, gpu_ctx ) != 0 )
theano/gpuarray/c_code/dnn_sptf_gi.c:    out_dims[0] = (int) PyGpuArray_DIM(input, 0); // num_images
theano/gpuarray/c_code/dnn_sptf_gi.c:    out_dims[1] = (int) PyGpuArray_DIM(input, 1); // num_channels
theano/gpuarray/c_code/dnn_sptf_gi.c:    out_dims[2] = (int) PyGpuArray_DIM(grid, 1); // grid height
theano/gpuarray/c_code/dnn_sptf_gi.c:    out_dims[3] = (int) PyGpuArray_DIM(grid, 2); // grid width
theano/gpuarray/c_code/dnn_sptf_gi.c:            "GpuDnnTransformerGradI: could not initialize descriptor (sptf): %s",
theano/gpuarray/c_code/dnn_sptf_gi.c:    if ( PyGpuArray_SIZE( *input_grad ) == 0 || PyGpuArray_SIZE( *grid_grad ) == 0 )
theano/gpuarray/c_code/dnn_sptf_gi.c:    cuda_enter( gpu_ctx->ctx );
theano/gpuarray/c_code/dnn_sptf_gi.c:    cuda_wait( input->ga.data, GPUARRAY_CUDA_WAIT_READ );
theano/gpuarray/c_code/dnn_sptf_gi.c:    cuda_wait( grid->ga.data, GPUARRAY_CUDA_WAIT_READ );
theano/gpuarray/c_code/dnn_sptf_gi.c:    cuda_wait( dy->ga.data, GPUARRAY_CUDA_WAIT_READ );
theano/gpuarray/c_code/dnn_sptf_gi.c:    cuda_wait( (*input_grad)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
theano/gpuarray/c_code/dnn_sptf_gi.c:    cuda_wait( (*grid_grad)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
theano/gpuarray/c_code/dnn_sptf_gi.c:        APPLY_SPECIFIC(xdesc), PyGpuArray_DEV_DATA( input ), beta_p,
theano/gpuarray/c_code/dnn_sptf_gi.c:        APPLY_SPECIFIC(dxdesc), PyGpuArray_DEV_DATA( *input_grad ), alpha_p,
theano/gpuarray/c_code/dnn_sptf_gi.c:        APPLY_SPECIFIC(dydesc), PyGpuArray_DEV_DATA( dy ), PyGpuArray_DEV_DATA( grid ),
theano/gpuarray/c_code/dnn_sptf_gi.c:        beta_p, PyGpuArray_DEV_DATA( *grid_grad ) );
theano/gpuarray/c_code/dnn_sptf_gi.c:    cuda_record( input->ga.data, GPUARRAY_CUDA_WAIT_READ );
theano/gpuarray/c_code/dnn_sptf_gi.c:    cuda_record( grid->ga.data, GPUARRAY_CUDA_WAIT_READ );
theano/gpuarray/c_code/dnn_sptf_gi.c:    cuda_record( dy->ga.data, GPUARRAY_CUDA_WAIT_READ );
theano/gpuarray/c_code/dnn_sptf_gi.c:    cuda_record( (*input_grad)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
theano/gpuarray/c_code/dnn_sptf_gi.c:    cuda_record( (*grid_grad)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
theano/gpuarray/c_code/dnn_sptf_gi.c:    cuda_exit( gpu_ctx->ctx );
theano/gpuarray/c_code/dnn_sptf_gi.c:            "GpuDnnTransformerGradI: failed to compute gradients of the inputs: %s",
theano/gpuarray/c_code/corr_gemm.c:// GPU kernel for the case of dilation
theano/gpuarray/c_code/corr_gemm.c:// GPU kernel for the case of dilation
theano/gpuarray/c_code/corr_gemm.c:          GpuArray *A, size_t offA, size_t lda,
theano/gpuarray/c_code/corr_gemm.c:          GpuArray *B, size_t offB, size_t ldb,
theano/gpuarray/c_code/corr_gemm.c:          double beta, GpuArray *C, size_t offC, size_t ldc) {
theano/gpuarray/c_code/corr_gemm.c:    return gpublas_sgemm(o, tA, tB,
theano/gpuarray/c_code/corr_gemm.c:    return gpublas_dgemm(o, tA, tB,
theano/gpuarray/c_code/corr_gemm.c:    return gpublas_hgemm(o, tA, tB,
theano/gpuarray/c_code/corr_gemm.c:int im2col(GpuArray *data_im, const size_t data_im_offset, const size_t channels,
theano/gpuarray/c_code/corr_gemm.c:    GpuArray *data_col) {
theano/gpuarray/c_code/corr_gemm.c:                     "gpuarray error: dilated_im2col_kernel: %s.",
theano/gpuarray/c_code/corr_gemm.c:                     GpuKernel_error(&k_dilated_im2col_kernel, err));
theano/gpuarray/c_code/corr_gemm.c:                     "gpuarray error: im2col_kernel: %s.",
theano/gpuarray/c_code/corr_gemm.c:                     GpuKernel_error(&k_im2col_kernel, err));
theano/gpuarray/c_code/corr_gemm.c:int col2im(GpuArray *data_col, const size_t channels,
theano/gpuarray/c_code/corr_gemm.c:    const size_t stride_h, const size_t stride_w, GpuArray *data_im, const size_t data_im_offset) {
theano/gpuarray/c_code/corr_gemm.c:                     "gpuarray error: dilated_col2im_kernel: %s.",
theano/gpuarray/c_code/corr_gemm.c:                     GpuKernel_error(&k_dilated_col2im_kernel, err));
theano/gpuarray/c_code/corr_gemm.c:                     "gpuarray error: col2im_kernel: %s.",
theano/gpuarray/c_code/corr_gemm.c:                     GpuKernel_error(&k_col2im_kernel, err));
theano/gpuarray/c_code/corr_gemm.c:PyGpuArrayObject* corrMM(PyGpuArrayObject *const bottom,
theano/gpuarray/c_code/corr_gemm.c:                         PyGpuArrayObject *const weight,
theano/gpuarray/c_code/corr_gemm.c:                         PyGpuArrayObject *const top,
theano/gpuarray/c_code/corr_gemm.c:    if (PyGpuArray_NDIM(bottom) != 4)
theano/gpuarray/c_code/corr_gemm.c:        PyErr_SetString(PyExc_ValueError, "GpuCorrMM requires bottom of 4D");
theano/gpuarray/c_code/corr_gemm.c:    if (!GpuArray_IS_C_CONTIGUOUS(&bottom->ga))
theano/gpuarray/c_code/corr_gemm.c:                "GpuCorrMM requires bottom to be C-contiguous, "
theano/gpuarray/c_code/corr_gemm.c:                PyGpuArray_STRIDES(bottom)[0],
theano/gpuarray/c_code/corr_gemm.c:                PyGpuArray_STRIDES(bottom)[1],
theano/gpuarray/c_code/corr_gemm.c:                PyGpuArray_STRIDES(bottom)[2],
theano/gpuarray/c_code/corr_gemm.c:                PyGpuArray_STRIDES(bottom)[3]);
theano/gpuarray/c_code/corr_gemm.c:    if (PyGpuArray_NDIM(weight) != (unshared ? 6 : 4))
theano/gpuarray/c_code/corr_gemm.c:        PyErr_Format(PyExc_ValueError, "GpuCorrMM requires weight of %dD", unshared ? 6 : 4);
theano/gpuarray/c_code/corr_gemm.c:    if (!GpuArray_IS_C_CONTIGUOUS(&weight->ga))
theano/gpuarray/c_code/corr_gemm.c:                    "GpuCorrMM requires weight to be C-contiguous, "
theano/gpuarray/c_code/corr_gemm.c:                    PyGpuArray_STRIDES(weight)[0],
theano/gpuarray/c_code/corr_gemm.c:                    PyGpuArray_STRIDES(weight)[1],
theano/gpuarray/c_code/corr_gemm.c:                    PyGpuArray_STRIDES(weight)[2],
theano/gpuarray/c_code/corr_gemm.c:                    PyGpuArray_STRIDES(weight)[3],
theano/gpuarray/c_code/corr_gemm.c:                    PyGpuArray_STRIDES(weight)[4],
theano/gpuarray/c_code/corr_gemm.c:                    PyGpuArray_STRIDES(weight)[5]);
theano/gpuarray/c_code/corr_gemm.c:                    "GpuCorrMM requires weight to be C-contiguous, "
theano/gpuarray/c_code/corr_gemm.c:                    PyGpuArray_STRIDES(weight)[0],
theano/gpuarray/c_code/corr_gemm.c:                    PyGpuArray_STRIDES(weight)[1],
theano/gpuarray/c_code/corr_gemm.c:                    PyGpuArray_STRIDES(weight)[2],
theano/gpuarray/c_code/corr_gemm.c:                    PyGpuArray_STRIDES(weight)[3]);
theano/gpuarray/c_code/corr_gemm.c:    if (PyGpuArray_NDIM(top) != 4)
theano/gpuarray/c_code/corr_gemm.c:        PyErr_SetString(PyExc_ValueError, "GpuCorrMM requires top of 4D");
theano/gpuarray/c_code/corr_gemm.c:    if (!GpuArray_IS_C_CONTIGUOUS(&top->ga))
theano/gpuarray/c_code/corr_gemm.c:                "GpuCorrMM requires top to be C-contiguous, "
theano/gpuarray/c_code/corr_gemm.c:                PyGpuArray_STRIDES(top)[0],
theano/gpuarray/c_code/corr_gemm.c:                PyGpuArray_STRIDES(top)[1],
theano/gpuarray/c_code/corr_gemm.c:                PyGpuArray_STRIDES(top)[2],
theano/gpuarray/c_code/corr_gemm.c:                PyGpuArray_STRIDES(top)[3]);
theano/gpuarray/c_code/corr_gemm.c:    const size_t batchSize = PyGpuArray_DIMS(bottom)[0];
theano/gpuarray/c_code/corr_gemm.c:    const size_t nChannels = PyGpuArray_DIMS(bottom)[1];
theano/gpuarray/c_code/corr_gemm.c:    const size_t bottomHeight = PyGpuArray_DIMS(bottom)[2];
theano/gpuarray/c_code/corr_gemm.c:    const size_t bottomWidth = PyGpuArray_DIMS(bottom)[3];
theano/gpuarray/c_code/corr_gemm.c:    const size_t nFilters = PyGpuArray_DIMS(weight)[0];
theano/gpuarray/c_code/corr_gemm.c:    const size_t kH = PyGpuArray_DIMS(weight)[unshared ? 4 : 2];
theano/gpuarray/c_code/corr_gemm.c:    const size_t kW = PyGpuArray_DIMS(weight)[unshared ? 5 : 3];
theano/gpuarray/c_code/corr_gemm.c:    if (nChannels != PyGpuArray_DIMS(weight)[unshared ? 3 : 1] * numgroups) {
theano/gpuarray/c_code/corr_gemm.c:                "GpuCorrMM images and kernel must have the same stack size\n");
theano/gpuarray/c_code/corr_gemm.c:                "GPUCorrMM the number of filters must be divisible by the number of groups\n");
theano/gpuarray/c_code/corr_gemm.c:        if (topHeight != PyGpuArray_DIMS(weight)[1] ||
theano/gpuarray/c_code/corr_gemm.c:                topWidth != PyGpuArray_DIMS(weight)[2]) {
theano/gpuarray/c_code/corr_gemm.c:                    "GpuCorrMM regions in kernel must match output regions:\n"
theano/gpuarray/c_code/corr_gemm.c:                    nFilters, PyGpuArray_DIMS(weight)[1],
theano/gpuarray/c_code/corr_gemm.c:                    PyGpuArray_DIMS(weight)[2], nChannels / numgroups, kH, kW,
theano/gpuarray/c_code/corr_gemm.c:        if (batchSize != PyGpuArray_DIMS(top)[0] ||
theano/gpuarray/c_code/corr_gemm.c:                nFilters != PyGpuArray_DIMS(top)[1] ||
theano/gpuarray/c_code/corr_gemm.c:                topHeight != PyGpuArray_DIMS(top)[2] ||
theano/gpuarray/c_code/corr_gemm.c:                topWidth != PyGpuArray_DIMS(top)[3]) {
theano/gpuarray/c_code/corr_gemm.c:                    "GpuCorrMM shape inconsistency:\n"
theano/gpuarray/c_code/corr_gemm.c:                    PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
theano/gpuarray/c_code/corr_gemm.c:                    PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3],
theano/gpuarray/c_code/corr_gemm.c:        if (batchSize != PyGpuArray_DIMS(top)[0] ||
theano/gpuarray/c_code/corr_gemm.c:                nFilters != PyGpuArray_DIMS(top)[1] ||
theano/gpuarray/c_code/corr_gemm.c:                topHeight != PyGpuArray_DIMS(top)[2] ||
theano/gpuarray/c_code/corr_gemm.c:                topWidth != PyGpuArray_DIMS(top)[3]) {
theano/gpuarray/c_code/corr_gemm.c:                    "GpuCorrMM shape inconsistency:\n"
theano/gpuarray/c_code/corr_gemm.c:                    PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
theano/gpuarray/c_code/corr_gemm.c:                    PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3],
theano/gpuarray/c_code/corr_gemm.c:    int err = gpublas_setup(bottom->context->ctx);
theano/gpuarray/c_code/corr_gemm.c:    PyGpuArrayObject* col = (PyGpuArrayObject*)pygpu_empty(2, col_dim,
theano/gpuarray/c_code/corr_gemm.c:                "GpuCorrMM failed to allocate working memory of %ld x %ld\n",
theano/gpuarray/c_code/corr_gemm.c:    const size_t batch_bottom_stride = PyGpuArray_STRIDES(bottom)[0] / gpuarray_get_elsize(bottom->ga.typecode);
theano/gpuarray/c_code/corr_gemm.c:    const size_t batch_top_stride = PyGpuArray_STRIDES(top)[0] / gpuarray_get_elsize(top->ga.typecode);
theano/gpuarray/c_code/corr_gemm.c:    const size_t group_bottom_stride = (PyGpuArray_STRIDES(bottom)[1] * nChannels / numgroups) / gpuarray_get_elsize(bottom->ga.typecode);
theano/gpuarray/c_code/corr_gemm.c:    const size_t group_top_stride = (PyGpuArray_STRIDES(top)[1] * nFilters / numgroups) / gpuarray_get_elsize(top->ga.typecode);
theano/gpuarray/c_code/corr_gemm.c:    const size_t group_weight_stride = (PyGpuArray_STRIDES(weight)[0] * nFilters / numgroups) / gpuarray_get_elsize(weight->ga.typecode);
theano/gpuarray/c_code/corr_gemm.c:    PyGpuArrayObject *output;
theano/gpuarray/c_code/corr_gemm.c:            err = GpuArray_memset(&output->ga, 0);
theano/gpuarray/c_code/corr_gemm.c:                             "GpuCorrMM could not fill the output with zeros: %d", err);
theano/gpuarray/c_code/corr_gemm.c:                      PyErr_Format(PyExc_RuntimeError, "GpuCorrMM forward encountered an error running gemm: %d", err);
theano/gpuarray/c_code/corr_gemm.c:                    PyErr_Format(PyExc_RuntimeError, "GpuCorrMM forward encountered an error running gemm: %d", err);
theano/gpuarray/c_code/corr_gemm.c:            err = GpuArray_memset(&output->ga, 0);
theano/gpuarray/c_code/corr_gemm.c:                             "GpuCorrMM grad wrt. weights could not fill the output with zeros: %d", err);
theano/gpuarray/c_code/corr_gemm.c:                      PyErr_Format(PyExc_RuntimeError, "GpuCorrMM grad weights encountered an error running gemm: %d", err);
theano/gpuarray/c_code/corr_gemm.c:                    PyErr_Format(PyExc_RuntimeError, "GpuCorrMM grad weights encountered an error running gemm: %d", err);
theano/gpuarray/c_code/corr_gemm.c:            err = GpuArray_memset(&output->ga, 0);
theano/gpuarray/c_code/corr_gemm.c:                             "GpuCorrMM grad wrt. inputs could not fill the output with zeros: %d", err);
theano/gpuarray/c_code/corr_gemm.c:                      PyErr_Format(PyExc_RuntimeError, "GpuCorrMM grad inputs encountered an error running gemm: %d", err);
theano/gpuarray/c_code/corr_gemm.c:                    PyErr_Format(PyExc_RuntimeError, "GpuCorrMM grad inputs encountered an error running gemm: %d", err);
theano/gpuarray/c_code/corr_gemm.c:    // (re)allocation and refcounting is done in BaseGpuCorrMM.c_code_helper();
theano/gpuarray/c_code/dnn_base.c:c_set_tensor_for_conv(PyGpuArrayObject *var, cudnnTensorDescriptor_t desc, size_t groups) {
theano/gpuarray/c_code/dnn_base.c:  ds = gpuarray_get_elsize(var->ga.typecode);
theano/gpuarray/c_code/dnn_base.c:  unsigned int nd = PyGpuArray_NDIM(var);
theano/gpuarray/c_code/dnn_base.c:    strs[i] = (PyGpuArray_DIM(var, i) != 1 && PyGpuArray_STRIDE(var, i)) ?
theano/gpuarray/c_code/dnn_base.c:      PyGpuArray_STRIDE(var, i)/ds : default_stride;
theano/gpuarray/c_code/dnn_base.c:    default_stride *= PyGpuArray_DIM(var, i);
theano/gpuarray/c_code/dnn_base.c:    dims[i] = PyGpuArray_DIM(var, i);
theano/gpuarray/c_code/dnn_base.c:c_set_tensorNd(PyGpuArrayObject *var, cudnnTensorDescriptor_t desc) {
theano/gpuarray/c_code/dnn_base.c:static int c_make_tensorNd(PyGpuArrayObject *var, cudnnTensorDescriptor_t *desc) {
theano/gpuarray/c_code/dnn_base.c:c_set_filter(PyGpuArrayObject *var, cudnnFilterDescriptor_t desc, size_t groups) {
theano/gpuarray/c_code/dnn_base.c:  if (!GpuArray_IS_C_CONTIGUOUS(&var->ga)) {
theano/gpuarray/c_code/dnn_base.c:  unsigned int nd = PyGpuArray_NDIM(var);
theano/gpuarray/c_code/dnn_base.c:    dims[i] = PyGpuArray_DIM(var, i);
theano/gpuarray/c_code/dnn_base.c:static int c_make_filter(PyGpuArrayObject *var, cudnnFilterDescriptor_t *desc) {
theano/gpuarray/c_code/dnn_base.c:setup_ext_cuda();
theano/gpuarray/c_code/dnn_pool_grad.c:int APPLY_SPECIFIC(dnn_pool_grad)(PyGpuArrayObject *inp,
theano/gpuarray/c_code/dnn_pool_grad.c:                                  PyGpuArrayObject *out,
theano/gpuarray/c_code/dnn_pool_grad.c:                                  PyGpuArrayObject *out_grad,
theano/gpuarray/c_code/dnn_pool_grad.c:                                  PyGpuArrayObject **inp_grad,
theano/gpuarray/c_code/dnn_pool_grad.c:  PyGpuContextObject *c = inp->context;
theano/gpuarray/c_code/dnn_pool_grad.c:  if (!GpuArray_IS_C_CONTIGUOUS(&inp->ga)) {
theano/gpuarray/c_code/dnn_pool_grad.c:  if (!GpuArray_IS_C_CONTIGUOUS(&out_grad->ga)) {
theano/gpuarray/c_code/dnn_pool_grad.c:  if (!GpuArray_IS_C_CONTIGUOUS(&out->ga)) {
theano/gpuarray/c_code/dnn_pool_grad.c:  if (theano_prep_output(inp_grad, PyGpuArray_NDIM(inp),
theano/gpuarray/c_code/dnn_pool_grad.c:                         PyGpuArray_DIMS(inp), inp->ga.typecode,
theano/gpuarray/c_code/dnn_pool_grad.c:  if (PyGpuArray_DIM(*inp_grad, 0) == 0)
theano/gpuarray/c_code/dnn_pool_grad.c:  int ndims = PyArray_DIM(ws, 0);//PyGpuArray_NDIM(img) - 2;
theano/gpuarray/c_code/dnn_pool_grad.c:    cuda_enter(c->ctx);
theano/gpuarray/c_code/dnn_pool_grad.c:    cuda_wait(out->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_pool_grad.c:    cuda_wait(out_grad->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_pool_grad.c:    cuda_wait(inp->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_pool_grad.c:    cuda_wait((*inp_grad)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_pool_grad.c:      APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(out),
theano/gpuarray/c_code/dnn_pool_grad.c:      APPLY_SPECIFIC(output_grad), PyGpuArray_DEV_DATA(out_grad),
theano/gpuarray/c_code/dnn_pool_grad.c:      APPLY_SPECIFIC(input), PyGpuArray_DEV_DATA(inp),
theano/gpuarray/c_code/dnn_pool_grad.c:      APPLY_SPECIFIC(input_grad), PyGpuArray_DEV_DATA(*inp_grad)
theano/gpuarray/c_code/dnn_pool_grad.c:    cuda_record(out->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_pool_grad.c:    cuda_record(out_grad->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_pool_grad.c:    cuda_record(inp->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_pool_grad.c:    cuda_record((*inp_grad)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_pool_grad.c:    cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_sptf_grid.c:            "GpuDnnTransformerGrid: could not allocate spatial transformer descriptor (sptf): %s",
theano/gpuarray/c_code/dnn_sptf_grid.c:APPLY_SPECIFIC(dnn_sptf_grid)(PyGpuArrayObject * theta,
theano/gpuarray/c_code/dnn_sptf_grid.c:                              PyGpuArrayObject ** grid,
theano/gpuarray/c_code/dnn_sptf_grid.c:    PyGpuContextObject * gpu_ctx = theta->context;
theano/gpuarray/c_code/dnn_sptf_grid.c:            "GpuDnnTransformerGrid: unsupported data type for theta in spatial transformer." );
theano/gpuarray/c_code/dnn_sptf_grid.c:            "GpuDnnTransformerGrid: out_dims must have 4 elements." );
theano/gpuarray/c_code/dnn_sptf_grid.c:    if ( PyGpuArray_DIM( theta, 0 ) != num_images ||
theano/gpuarray/c_code/dnn_sptf_grid.c:         PyGpuArray_DIM( theta, 1 ) != 2 || PyGpuArray_DIM( theta, 2 ) != 3 )
theano/gpuarray/c_code/dnn_sptf_grid.c:            "GpuDnnTransformerGrid: incorrect dimensions for theta, expected (%d, %d, %d), got (%d, %d, %d)",
theano/gpuarray/c_code/dnn_sptf_grid.c:            num_images, 2, 3, PyGpuArray_DIMS( theta )[0],
theano/gpuarray/c_code/dnn_sptf_grid.c:            PyGpuArray_DIMS( theta )[1], PyGpuArray_DIMS( theta )[2] );
theano/gpuarray/c_code/dnn_sptf_grid.c:            "GpuDnnTransformerGrid: could not initialize descriptor (sptf): %s",
theano/gpuarray/c_code/dnn_sptf_grid.c:                             GA_C_ORDER, gpu_ctx ) != 0 )
theano/gpuarray/c_code/dnn_sptf_grid.c:            "GpuDnnTransformerGrid: could not allocate memory for grid of coordinates" );
theano/gpuarray/c_code/dnn_sptf_grid.c:    cuda_enter( gpu_ctx->ctx );
theano/gpuarray/c_code/dnn_sptf_grid.c:    cuda_wait( theta->ga.data, GPUARRAY_CUDA_WAIT_READ );
theano/gpuarray/c_code/dnn_sptf_grid.c:    cuda_wait( (*grid)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
theano/gpuarray/c_code/dnn_sptf_grid.c:        PyGpuArray_DEV_DATA( theta ), PyGpuArray_DEV_DATA( *grid ) );
theano/gpuarray/c_code/dnn_sptf_grid.c:    cuda_record( theta->ga.data, GPUARRAY_CUDA_WAIT_READ );
theano/gpuarray/c_code/dnn_sptf_grid.c:    cuda_record( (*grid)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
theano/gpuarray/c_code/dnn_sptf_grid.c:    cuda_exit( gpu_ctx->ctx );
theano/gpuarray/c_code/dnn_sptf_grid.c:            "GpuDnnTransformerGrid: could not create grid of coordinates: %s",
theano/gpuarray/c_code/gpuarray_helper.h:#ifndef THEANO_GPUARRAY_HELPER
theano/gpuarray/c_code/gpuarray_helper.h:#define THEANO_GPUARRAY_HELPER
theano/gpuarray/c_code/gpuarray_helper.h:#include <gpuarray_api.h>
theano/gpuarray/c_code/gpuarray_helper.h:#include <gpuarray/util.h>
theano/gpuarray/c_code/gpuarray_helper.h:static int theano_size_check(PyGpuArrayObject *a, unsigned int nd,
theano/gpuarray/c_code/gpuarray_helper.h:static int theano_prep_output(PyGpuArrayObject **out, unsigned int nd,
theano/gpuarray/c_code/gpuarray_helper.h:                             PyGpuContextObject *c) {
theano/gpuarray/c_code/gpuarray_helper.h:  *out = pygpu_empty(nd, dims, typecode, ord, c, Py_None);
theano/gpuarray/c_code/gpuarray_helper.h:static PyGpuArrayObject *theano_try_copy(PyGpuArrayObject *out,
theano/gpuarray/c_code/gpuarray_helper.h:                                         PyGpuArrayObject *V) {
theano/gpuarray/c_code/gpuarray_helper.h:      GpuArray_CHKFLAGS(&out->ga, GA_CARRAY) &&
theano/gpuarray/c_code/gpuarray_helper.h:      theano_size_check(out, PyGpuArray_NDIM(V),
theano/gpuarray/c_code/gpuarray_helper.h:                        PyGpuArray_DIMS(V),
theano/gpuarray/c_code/gpuarray_helper.h:    if (pygpu_move(out, V)) {
theano/gpuarray/c_code/gpuarray_helper.h:    out = pygpu_copy(V, GA_C_ORDER);
theano/gpuarray/c_code/gpuarray_helper.h:static inline void *PyGpuArray_DEV_DATA(PyGpuArrayObject *a) {
theano/gpuarray/c_code/gpuarray_helper.h:  /* This is guaranteed to work and return the raw CUDA/OpenCL object on
theano/gpuarray/c_code/gpuarray_helper.h:   * all recent (as of June 2015) version of libgpuarray. This is also
theano/gpuarray/c_code/gpuarray_helper.h:  /* This only works on cuda since we have a real pointer. */
theano/gpuarray/c_code/dnn_sptf_sampler.c:            "GpuDnnTransformerSampler: could not allocate spatial transformer descriptor (sptf): %s",
theano/gpuarray/c_code/dnn_sptf_sampler.c:            "GpuDnnTransformerSampler: failed to allocate cuDNN tensor descriptor xdesc: %s",
theano/gpuarray/c_code/dnn_sptf_sampler.c:            "GpuDnnTransformerSampler: failed to allocate cuDNN tensor descriptor ydesc: %s",
theano/gpuarray/c_code/dnn_sptf_sampler.c:APPLY_SPECIFIC(dnn_sptf_sampler)(PyGpuArrayObject * input,
theano/gpuarray/c_code/dnn_sptf_sampler.c:                                 PyGpuArrayObject * grid,
theano/gpuarray/c_code/dnn_sptf_sampler.c:                                 PyGpuArrayObject ** output,
theano/gpuarray/c_code/dnn_sptf_sampler.c:    PyGpuContextObject * gpu_ctx = input->context;
theano/gpuarray/c_code/dnn_sptf_sampler.c:            "GpuDnnTransformer: unsupported type for input in spatial transformer." );
theano/gpuarray/c_code/dnn_sptf_sampler.c:    out_dims[0] = (size_t) PyGpuArray_DIM(input, 0); // num_images
theano/gpuarray/c_code/dnn_sptf_sampler.c:    out_dims[1] = (size_t) PyGpuArray_DIM(input, 1); // num_channels
theano/gpuarray/c_code/dnn_sptf_sampler.c:    out_dims[2] = (size_t) PyGpuArray_DIM(grid, 1); // grid height
theano/gpuarray/c_code/dnn_sptf_sampler.c:    out_dims[3] = (size_t) PyGpuArray_DIM(grid, 2); // grid width
theano/gpuarray/c_code/dnn_sptf_sampler.c:            "GpuDnnTransformerSampler: one of the sampler dimensions is zero" );
theano/gpuarray/c_code/dnn_sptf_sampler.c:                             GA_C_ORDER, gpu_ctx ) != 0 )
theano/gpuarray/c_code/dnn_sptf_sampler.c:            "GpuDnnTransformerSampler: could not allocate memory for grid sampler" );
theano/gpuarray/c_code/dnn_sptf_sampler.c:            "GpuDnnTransformerSampler: could not initialize descriptor: %s",
theano/gpuarray/c_code/dnn_sptf_sampler.c:    cuda_enter( gpu_ctx->ctx );
theano/gpuarray/c_code/dnn_sptf_sampler.c:    cuda_wait( input->ga.data, GPUARRAY_CUDA_WAIT_READ );
theano/gpuarray/c_code/dnn_sptf_sampler.c:    cuda_wait( grid->ga.data, GPUARRAY_CUDA_WAIT_READ );
theano/gpuarray/c_code/dnn_sptf_sampler.c:    cuda_wait( (*output)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
theano/gpuarray/c_code/dnn_sptf_sampler.c:        APPLY_SPECIFIC(xdesc), PyGpuArray_DEV_DATA( input ), PyGpuArray_DEV_DATA( grid ),
theano/gpuarray/c_code/dnn_sptf_sampler.c:        beta_p, APPLY_SPECIFIC(ydesc), PyGpuArray_DEV_DATA( *output ) );
theano/gpuarray/c_code/dnn_sptf_sampler.c:    cuda_record( input->ga.data, GPUARRAY_CUDA_WAIT_READ );
theano/gpuarray/c_code/dnn_sptf_sampler.c:    cuda_record( grid->ga.data, GPUARRAY_CUDA_WAIT_READ );
theano/gpuarray/c_code/dnn_sptf_sampler.c:    cuda_record( (*output)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
theano/gpuarray/c_code/dnn_sptf_sampler.c:    cuda_exit( gpu_ctx->ctx );
theano/gpuarray/c_code/dnn_sptf_sampler.c:            "GpuDnnTransformerSampler: could not create grid sampler: %s",
theano/gpuarray/c_code/blockgemv.c:int APPLY_SPECIFIC(blockgemv)(PyGpuArrayObject *o, PyGpuArrayObject *W,
theano/gpuarray/c_code/blockgemv.c:                              PyGpuArrayObject *h, PyArrayObject *inputIdx,
theano/gpuarray/c_code/blockgemv.c:                              PyGpuArrayObject **_out,
theano/gpuarray/c_code/blockgemv.c:  PyGpuArrayObject *out = *_out;
theano/gpuarray/c_code/blockgemv.c:  gpudata **W_list = NULL;
theano/gpuarray/c_code/blockgemv.c:  gpudata **inp_list = NULL;
theano/gpuarray/c_code/blockgemv.c:  gpudata **out_list = NULL;
theano/gpuarray/c_code/blockgemv.c:  err = gpublas_setup(params->context->ctx);
theano/gpuarray/c_code/blockgemv.c:  size_t maxi = PyGpuArray_DIMS(h)[1];
theano/gpuarray/c_code/blockgemv.c:  size_t maxj = PyGpuArray_DIMS(out)[1];
theano/gpuarray/c_code/blockgemv.c:  size_t maxb = PyGpuArray_DIMS(out)[0];
theano/gpuarray/c_code/blockgemv.c:  ssize_t h_str_0 = PyGpuArray_STRIDES(h)[0];
theano/gpuarray/c_code/blockgemv.c:  ssize_t h_str_1 = PyGpuArray_STRIDES(h)[1];
theano/gpuarray/c_code/blockgemv.c:  ssize_t o_str_0 = PyGpuArray_STRIDES(out)[0];
theano/gpuarray/c_code/blockgemv.c:  ssize_t o_str_1 = PyGpuArray_STRIDES(out)[1];
theano/gpuarray/c_code/blockgemv.c:  ssize_t W_str_0 = PyGpuArray_STRIDES(W)[0];
theano/gpuarray/c_code/blockgemv.c:  ssize_t W_str_1 = PyGpuArray_STRIDES(W)[1];
theano/gpuarray/c_code/blockgemv.c:  W_list = (gpudata **)calloc(sizeof(gpudata *), maxi * maxj * maxb);
theano/gpuarray/c_code/blockgemv.c:  inp_list = (gpudata **)calloc(sizeof(gpudata *), maxi * maxj * maxb);
theano/gpuarray/c_code/blockgemv.c:  out_list = (gpudata **)calloc(sizeof(gpudata *), maxi * maxj * maxb);
theano/gpuarray/c_code/blockgemv.c:  size_t lda = PyGpuArray_STRIDES(W)[2] / gpuarray_get_elsize(W->ga.typecode);
theano/gpuarray/c_code/blockgemv.c:    lda = PyGpuArray_STRIDES(W)[3] / gpuarray_get_elsize(W->ga.typecode);
theano/gpuarray/c_code/blockgemv.c:    err = gpublas_sgemvBatch(cb_fortran, transA,
theano/gpuarray/c_code/blockgemv.c:                             PyGpuArray_DIMS(out)[2],
theano/gpuarray/c_code/blockgemv.c:                             PyGpuArray_DIMS(h)[2], 1,
theano/gpuarray/c_code/blockgemv.c:                             inp_list, offInp, PyGpuArray_STRIDES(h)[2] / gpuarray_get_elsize(h->ga.typecode),
theano/gpuarray/c_code/blockgemv.c:                             1, out_list, offOut, PyGpuArray_STRIDES(out)[2] / gpuarray_get_elsize(out->ga.typecode),
theano/gpuarray/c_code/blockgemv.c:                             PyGpuArray_DIMS(out)[1] * PyGpuArray_DIMS(h)[1] * PyGpuArray_DIMS(out)[0], 0);
theano/gpuarray/c_code/blockgemv.c:    err = gpublas_dgemvBatch(cb_fortran, transA,
theano/gpuarray/c_code/blockgemv.c:                             PyGpuArray_DIMS(out)[2],
theano/gpuarray/c_code/blockgemv.c:                             PyGpuArray_DIMS(h)[2], 1,
theano/gpuarray/c_code/blockgemv.c:                             inp_list, offInp, PyGpuArray_STRIDES(h)[2] / gpuarray_get_elsize(h->ga.typecode),
theano/gpuarray/c_code/blockgemv.c:                             1, out_list, offOut, PyGpuArray_STRIDES(out)[2] / gpuarray_get_elsize(out->ga.typecode),
theano/gpuarray/c_code/blockgemv.c:                             PyGpuArray_DIMS(out)[1] * PyGpuArray_DIMS(h)[1] * PyGpuArray_DIMS(out)[0], 0);
theano/gpuarray/c_code/blockgemv.c:    err = gpublas_sgemvBatch(cb_fortran, transA,
theano/gpuarray/c_code/blockgemv.c:                             PyGpuArray_DIMS(out)[2],
theano/gpuarray/c_code/blockgemv.c:                             PyGpuArray_DIMS(h)[2], 1,
theano/gpuarray/c_code/blockgemv.c:                             inp_list, offInp, PyGpuArray_STRIDES(h)[2] / gpuarray_get_elsize(h->ga.typecode),
theano/gpuarray/c_code/blockgemv.c:                             1, out_list, offOut, PyGpuArray_STRIDES(out)[2] / gpuarray_get_elsize(out->ga.typecode),
theano/gpuarray/c_code/blockgemv.c:                             PyGpuArray_DIMS(out)[1] * PyGpuArray_DIMS(h)[1] * PyGpuArray_DIMS(out)[0], 0);
theano/gpuarray/c_code/dnn_pool.c:int APPLY_SPECIFIC(dnn_pool)(PyGpuArrayObject *img,
theano/gpuarray/c_code/dnn_pool.c:                             PyGpuArrayObject **out,
theano/gpuarray/c_code/dnn_pool.c:  PyGpuContextObject *c = img->context;
theano/gpuarray/c_code/dnn_pool.c:  if (!GpuArray_IS_C_CONTIGUOUS(&img->ga)) {
theano/gpuarray/c_code/dnn_pool.c:  int ndims = PyArray_DIM(ws, 0);//PyGpuArray_NDIM(img) - 2;
theano/gpuarray/c_code/dnn_pool.c:  dims[0] = PyGpuArray_DIM(img, 0);
theano/gpuarray/c_code/dnn_pool.c:  dims[1] = PyGpuArray_DIM(img, 1);
theano/gpuarray/c_code/dnn_pool.c:  dims[2] = (PyGpuArray_DIM(img, 2) + (p[0]*2) - w[0]) / s[0] + 1;
theano/gpuarray/c_code/dnn_pool.c:  dims[3] = (PyGpuArray_DIM(img, 3) + (p[1]*2) - w[1]) / s[1] + 1;
theano/gpuarray/c_code/dnn_pool.c:    dims[4] = (PyGpuArray_DIM(img, 4) + (p[2]*2) - w[2]) / s[2] + 1;
theano/gpuarray/c_code/dnn_pool.c:  if (PyGpuArray_DIM(*out, 0) == 0)
theano/gpuarray/c_code/dnn_pool.c:    cuda_enter(c->ctx);
theano/gpuarray/c_code/dnn_pool.c:    cuda_wait(img->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_pool.c:    cuda_wait((*out)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_pool.c:      APPLY_SPECIFIC(input), PyGpuArray_DEV_DATA(img),
theano/gpuarray/c_code/dnn_pool.c:      APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(*out));
theano/gpuarray/c_code/dnn_pool.c:    cuda_record(img->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_pool.c:    cuda_record((*out)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_pool.c:    cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_pool.c:                 "GpuDnnPool: error doing cudnnPoolingForward operation: %s",
theano/gpuarray/c_code/pool_ave_grad.c:int APPLY_SPECIFIC(ave_pool_grad)(PyGpuArrayObject *x,
theano/gpuarray/c_code/pool_ave_grad.c:                                  PyGpuArrayObject *gz,
theano/gpuarray/c_code/pool_ave_grad.c:                                  PyGpuArrayObject **gx,
theano/gpuarray/c_code/pool_ave_grad.c:  if (!GpuArray_IS_C_CONTIGUOUS(&x->ga)
theano/gpuarray/c_code/pool_ave_grad.c:      || !GpuArray_IS_C_CONTIGUOUS(&gz->ga))
theano/gpuarray/c_code/pool_ave_grad.c:                   "GpuMaxPoolGrad: requires data to be C-contiguous");
theano/gpuarray/c_code/pool_ave_grad.c:  if (PyGpuArray_NDIM(x) != ndims + 2
theano/gpuarray/c_code/pool_ave_grad.c:      || PyGpuArray_NDIM(gz) != ndims + 2)
theano/gpuarray/c_code/pool_ave_grad.c:      PyErr_SetString(PyExc_ValueError, "GpuMaxPoolGrad: rank error");
theano/gpuarray/c_code/pool_ave_grad.c:  if (theano_prep_output(gx, PyGpuArray_NDIM(x), PyGpuArray_DIMS(x),
theano/gpuarray/c_code/pool_ave_grad.c:                      "GpuMaxPoolGrad: failed to allocate memory");
theano/gpuarray/c_code/pool_ave_grad.c:    const size_t* z_dims = PyGpuArray_DIMS(gz);
theano/gpuarray/c_code/pool_ave_grad.c:    const size_t* x_dims = PyGpuArray_DIMS(x);
theano/gpuarray/c_code/pool_ave_grad.c:                     "GpuAveragePoolGrad: ave_pool2d_grad_kernel %s.",
theano/gpuarray/c_code/pool_ave_grad.c:                     GpuKernel_error(&k_ave_pool2d_grad_kernel, err));
theano/gpuarray/c_code/pool_ave_grad.c:                     "GpuAveragePoolGrad: ave_pool3d_grad_kernel %s.",
theano/gpuarray/c_code/pool_ave_grad.c:                     GpuKernel_error(&k_ave_pool3d_grad_kernel, err));
theano/gpuarray/c_code/blockger.c:int APPLY_SPECIFIC(blockger)(PyGpuArrayObject *o, PyGpuArrayObject *x,
theano/gpuarray/c_code/blockger.c:                             PyGpuArrayObject *y, PyArrayObject *xIdx,
theano/gpuarray/c_code/blockger.c:                             PyGpuArrayObject **_out,
theano/gpuarray/c_code/blockger.c:  PyGpuArrayObject *out = *_out;
theano/gpuarray/c_code/blockger.c:  gpudata **o_list = NULL;
theano/gpuarray/c_code/blockger.c:  gpudata **x_list = NULL;
theano/gpuarray/c_code/blockger.c:  gpudata **y_list = NULL;
theano/gpuarray/c_code/blockger.c:  err = gpublas_setup(params->context->ctx);
theano/gpuarray/c_code/blockger.c:  size_t maxi = PyGpuArray_DIMS(x)[1];
theano/gpuarray/c_code/blockger.c:  size_t maxj = PyGpuArray_DIMS(y)[1];
theano/gpuarray/c_code/blockger.c:  size_t maxb = PyGpuArray_DIMS(x)[0];
theano/gpuarray/c_code/blockger.c:  ssize_t x_str_0 = PyGpuArray_STRIDES(x)[0];
theano/gpuarray/c_code/blockger.c:  ssize_t x_str_1 = PyGpuArray_STRIDES(x)[1];
theano/gpuarray/c_code/blockger.c:  ssize_t y_str_0 = PyGpuArray_STRIDES(y)[0];
theano/gpuarray/c_code/blockger.c:  ssize_t y_str_1 = PyGpuArray_STRIDES(y)[1];
theano/gpuarray/c_code/blockger.c:  ssize_t o_str_0 = PyGpuArray_STRIDES(out)[0];
theano/gpuarray/c_code/blockger.c:  ssize_t o_str_1 = PyGpuArray_STRIDES(out)[1];
theano/gpuarray/c_code/blockger.c:  o_list = (gpudata **)calloc(sizeof(gpudata *), maxi * maxj * maxb);
theano/gpuarray/c_code/blockger.c:  x_list = (gpudata **)calloc(sizeof(gpudata *), maxi * maxj * maxb);
theano/gpuarray/c_code/blockger.c:  y_list = (gpudata **)calloc(sizeof(gpudata *), maxi * maxj * maxb);
theano/gpuarray/c_code/blockger.c:  ssize_t str_y = PyGpuArray_STRIDES(y)[2] / gpuarray_get_elsize(y->ga.typecode);
theano/gpuarray/c_code/blockger.c:  ssize_t str_x = PyGpuArray_STRIDES(x)[2] / gpuarray_get_elsize(x->ga.typecode);
theano/gpuarray/c_code/blockger.c:  ssize_t str_out = PyGpuArray_STRIDES(out)[2] / gpuarray_get_elsize(out->ga.typecode);
theano/gpuarray/c_code/blockger.c:    err = gpublas_sgerBatch(cb_fortran,
theano/gpuarray/c_code/blockger.c:                            PyGpuArray_DIMS(y)[2], PyGpuArray_DIMS(x)[2],
theano/gpuarray/c_code/blockger.c:                            PyGpuArray_DIMS(x)[0] * PyGpuArray_DIMS(x)[1] * PyGpuArray_DIMS(y)[1], 0);
theano/gpuarray/c_code/blockger.c:    err = gpublas_dgerBatch(cb_fortran,
theano/gpuarray/c_code/blockger.c:                            PyGpuArray_DIMS(y)[2], PyGpuArray_DIMS(x)[2],
theano/gpuarray/c_code/blockger.c:                            PyGpuArray_DIMS(x)[0] * PyGpuArray_DIMS(x)[1] * PyGpuArray_DIMS(y)[1], 0);
theano/gpuarray/c_code/blockger.c:    err = gpublas_hgerBatch(cb_fortran,
theano/gpuarray/c_code/blockger.c:                            PyGpuArray_DIMS(y)[2], PyGpuArray_DIMS(x)[2],
theano/gpuarray/c_code/blockger.c:                            PyGpuArray_DIMS(x)[0] * PyGpuArray_DIMS(x)[1] * PyGpuArray_DIMS(y)[1], 0);
theano/gpuarray/c_code/dnn_rnn_gi.c:               PyGpuArrayObject *y, PyGpuArrayObject *dy,
theano/gpuarray/c_code/dnn_rnn_gi.c:               PyGpuArrayObject *w, PyGpuArrayObject *hx,
theano/gpuarray/c_code/dnn_rnn_gi.c:               gpudata *reserve, PyGpuArrayObject *cx,
theano/gpuarray/c_code/dnn_rnn_gi.c:               PyGpuArrayObject *dhy, PyGpuArrayObject *dcy,
theano/gpuarray/c_code/dnn_rnn_gi.c:               gpudata **oreserve, PyGpuArrayObject **dx,
theano/gpuarray/c_code/dnn_rnn_gi.c:               PyGpuArrayObject **dhx, PyGpuArrayObject **dcx,
theano/gpuarray/c_code/dnn_rnn_gi.c:  PyGpuContextObject *c = y->context;
theano/gpuarray/c_code/dnn_rnn_gi.c:  gpudata *workspace = NULL;
theano/gpuarray/c_code/dnn_rnn_gi.c:  size_t seqLength = PyGpuArray_DIM(y, 0);
theano/gpuarray/c_code/dnn_rnn_gi.c:  size_t miniBatch = PyGpuArray_DIM(y, 1);
theano/gpuarray/c_code/dnn_rnn_gi.c:  cuda_enter(c->ctx);
theano/gpuarray/c_code/dnn_rnn_gi.c:  dims[0] = PyGpuArray_DIM(y, 1);
theano/gpuarray/c_code/dnn_rnn_gi.c:  dims[1] = PyGpuArray_DIM(y, 2);
theano/gpuarray/c_code/dnn_rnn_gi.c:  if (theano_prep_output(dhx, 3, PyGpuArray_DIMS(hx), hx->ga.typecode,
theano/gpuarray/c_code/dnn_rnn_gi.c:    if (theano_prep_output(dcx, 3, PyGpuArray_DIMS(cx), cx->ga.typecode,
theano/gpuarray/c_code/dnn_rnn_gi.c:  workspace = gpudata_alloc(c->ctx, worksize, NULL, 0, NULL);
theano/gpuarray/c_code/dnn_rnn_gi.c:  *oreserve = gpudata_alloc(c->ctx, ressize, NULL, 0, NULL);
theano/gpuarray/c_code/dnn_rnn_gi.c:  if (gpudata_move(*oreserve, 0, reserve, 0, ressize) != GA_NO_ERROR) {
theano/gpuarray/c_code/dnn_rnn_gi.c:                             yl, PyGpuArray_DEV_DATA(y),
theano/gpuarray/c_code/dnn_rnn_gi.c:                             yl, PyGpuArray_DEV_DATA(dy),
theano/gpuarray/c_code/dnn_rnn_gi.c:                             dhydesc, dhy ? PyGpuArray_DEV_DATA(dhy) : NULL,
theano/gpuarray/c_code/dnn_rnn_gi.c:                             dcydesc, dcy ? PyGpuArray_DEV_DATA(dcy) : NULL,
theano/gpuarray/c_code/dnn_rnn_gi.c:                             wdesc, PyGpuArray_DEV_DATA(w),
theano/gpuarray/c_code/dnn_rnn_gi.c:                             hxdesc, PyGpuArray_DEV_DATA(hx),
theano/gpuarray/c_code/dnn_rnn_gi.c:                             cxdesc, cx ? PyGpuArray_DEV_DATA(cx) : NULL,
theano/gpuarray/c_code/dnn_rnn_gi.c:                             dxl, PyGpuArray_DEV_DATA(*dx),
theano/gpuarray/c_code/dnn_rnn_gi.c:                             dhxdesc, PyGpuArray_DEV_DATA(*dhx),
theano/gpuarray/c_code/dnn_rnn_gi.c:                             dcxdesc, dcx ? PyGpuArray_DEV_DATA(*dcx) : NULL,
theano/gpuarray/c_code/dnn_rnn_gi.c:    gpudata_release(workspace);
theano/gpuarray/c_code/dnn_rnn_gi.c:  cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gi.c:hash_prefix = std::string("GI|GPU#");
theano/gpuarray/c_code/dnn_gi.c:                         const PyGpuArrayObject* input,
theano/gpuarray/c_code/dnn_gi.c:                         const PyGpuArrayObject* kerns,
theano/gpuarray/c_code/dnn_gi.c:       algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT) && PyGpuArray_NDIM(kerns) == 4) {
theano/gpuarray/c_code/dnn_gi.c:          PyGpuArray_DIM(input, 2) > 1024 || PyGpuArray_DIM(input, 3) > 1024 ||
theano/gpuarray/c_code/dnn_gi.c:          (PyGpuArray_DIM(kerns, 2) == 1 && PyGpuArray_DIM(kerns, 3) == 1))
theano/gpuarray/c_code/dnn_gi.c:APPLY_SPECIFIC(conv_gi)(PyGpuArrayObject *kerns, PyGpuArrayObject *output,
theano/gpuarray/c_code/dnn_gi.c:                        PyGpuArrayObject *im,
theano/gpuarray/c_code/dnn_gi.c:                        double alpha, double beta, PyGpuArrayObject **input,
theano/gpuarray/c_code/dnn_gi.c:  PyGpuContextObject *c = kerns->context;
theano/gpuarray/c_code/dnn_gi.c:  if (PyGpuArray_DIMS(im)[1] != PyGpuArray_DIMS(kerns)[1] * params->num_groups) {
theano/gpuarray/c_code/dnn_gi.c:  if ((PyGpuArray_DIMS(kerns)[0] % params->num_groups) != 0) {
theano/gpuarray/c_code/dnn_gi.c:    if (theano_prep_output(input, PyGpuArray_NDIM(im), PyGpuArray_DIMS(im),
theano/gpuarray/c_code/dnn_gi.c:    if (beta != 0.0 && pygpu_move(*input, im))
theano/gpuarray/c_code/dnn_gi.c:  if (PyGpuArray_DIMS(im)[0] == 0 || PyGpuArray_DIMS(kerns)[0] == 0 || PyGpuArray_DIMS(kerns)[1] == 0) {
theano/gpuarray/c_code/dnn_gi.c:    int err2 = GpuArray_memset(&(*input)->ga, 0);
theano/gpuarray/c_code/dnn_gi.c:                     "GpuDnnConv grad wrt. inputs could not fill the output with zeros: %d", err2);
theano/gpuarray/c_code/dnn_gi.c:                                        PyGpuArray_NDIM(kerns), output, groups))
theano/gpuarray/c_code/dnn_gi.c:  size_t input_offset = PyGpuArray_STRIDE(*input, 0) / groups;
theano/gpuarray/c_code/dnn_gi.c:  size_t kern_offset = PyGpuArray_STRIDE(kerns, 0) * PyGpuArray_DIM(kerns, 0) / groups;
theano/gpuarray/c_code/dnn_gi.c:  size_t output_offset = PyGpuArray_STRIDE(output, 0) / groups;
theano/gpuarray/c_code/dnn_gi.c:  cuda_enter(c->ctx);
theano/gpuarray/c_code/dnn_gi.c:    cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gi.c:      gpucontext_property(c->ctx, GA_CTX_PROP_UNIQUE_ID, pci_id);
theano/gpuarray/c_code/dnn_gi.c:        cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gi.c:        gpudata *tmpmem;
theano/gpuarray/c_code/dnn_gi.c:        tmpmem = gpudata_alloc(c->ctx, maxfree, NULL, 0, NULL);
theano/gpuarray/c_code/dnn_gi.c:          PyErr_SetString(PyExc_MemoryError, "Could not allocate working GPU memory");
theano/gpuarray/c_code/dnn_gi.c:          cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gi.c:        PyGpuArrayObject* ip = *input;
theano/gpuarray/c_code/dnn_gi.c:            ip = pygpu_empty(PyGpuArray_NDIM(*input), PyGpuArray_DIMS(*input), (*input)->ga.typecode, GA_C_ORDER, c, Py_None);
theano/gpuarray/c_code/dnn_gi.c:          params->handle, APPLY_SPECIFIC(kerns), PyGpuArray_DEV_DATA(kerns),
theano/gpuarray/c_code/dnn_gi.c:          APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(output), desc,
theano/gpuarray/c_code/dnn_gi.c:          APPLY_SPECIFIC(input), PyGpuArray_DEV_DATA(ip),
theano/gpuarray/c_code/dnn_gi.c:        gpudata_release(tmpmem);
theano/gpuarray/c_code/dnn_gi.c:          cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gi.c:            cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gi.c:            cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gi.c:          cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gi.c:    cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gi.c:        cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gi.c:      cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gi.c:        cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gi.c:  gpudata *workspace = 0;
theano/gpuarray/c_code/dnn_gi.c:    workspace = gpudata_alloc(c->ctx, worksize, NULL, 0, NULL);
theano/gpuarray/c_code/dnn_gi.c:      cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gi.c:    cuda_wait(workspace, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_gi.c:  cuda_wait(kerns->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_gi.c:  cuda_wait(output->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_gi.c:  cuda_wait((*input)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_gi.c:  GpuArray_sync(&(*input)->ga);
theano/gpuarray/c_code/dnn_gi.c:      APPLY_SPECIFIC(kerns), ((char *)PyGpuArray_DEV_DATA(kerns)) + kern_offset * g,
theano/gpuarray/c_code/dnn_gi.c:      APPLY_SPECIFIC(output), ((char *)PyGpuArray_DEV_DATA(output)) + output_offset * g,
theano/gpuarray/c_code/dnn_gi.c:      APPLY_SPECIFIC(input), ((char *)PyGpuArray_DEV_DATA(*input)) + input_offset * g);
theano/gpuarray/c_code/dnn_gi.c:    cuda_record(workspace, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_gi.c:    gpudata_release(workspace);
theano/gpuarray/c_code/dnn_gi.c:  cuda_record(kerns->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_gi.c:  cuda_record(output->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_gi.c:  cuda_record((*input)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_gi.c:  GpuArray_sync(&(*input)->ga);
theano/gpuarray/c_code/dnn_gi.c:  cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_softmax.c:int APPLY_SPECIFIC(softmax)(PyGpuArrayObject *x,
theano/gpuarray/c_code/dnn_softmax.c:                            PyGpuArrayObject **out,
theano/gpuarray/c_code/dnn_softmax.c:  PyGpuContextObject *c = x->context;
theano/gpuarray/c_code/dnn_softmax.c:  if (theano_prep_output(out, PyGpuArray_NDIM(x),
theano/gpuarray/c_code/dnn_softmax.c:                         PyGpuArray_DIMS(x), x->ga.typecode,
theano/gpuarray/c_code/dnn_softmax.c:  if (PyGpuArray_SIZE(*out) == 0)
theano/gpuarray/c_code/dnn_softmax.c:    cuda_enter(c->ctx);
theano/gpuarray/c_code/dnn_softmax.c:    cuda_wait(x->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_softmax.c:    cuda_wait((*out)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_softmax.c:      PyGpuArray_DEV_DATA(x),
theano/gpuarray/c_code/dnn_softmax.c:      PyGpuArray_DEV_DATA(*out)
theano/gpuarray/c_code/dnn_softmax.c:    cuda_record(x->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_softmax.c:    cuda_record((*out)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_softmax.c:    cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_fwd.c:hash_prefix = std::string("FWD|GPU#");
theano/gpuarray/c_code/dnn_fwd.c:                          const PyGpuArrayObject* input,
theano/gpuarray/c_code/dnn_fwd.c:                          const PyGpuArrayObject* kerns,
theano/gpuarray/c_code/dnn_fwd.c:  if (PyGpuArray_NDIM(input) == 5 &&
theano/gpuarray/c_code/dnn_fwd.c:  if ((cudnnGetVersion() < 6100 || PyGpuArray_NDIM(input) == 5) &&
theano/gpuarray/c_code/dnn_fwd.c:      PyGpuArray_DIM(input, 0) > 65536)
theano/gpuarray/c_code/dnn_fwd.c:       algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING) && PyGpuArray_NDIM(input) == 4) {
theano/gpuarray/c_code/dnn_fwd.c:          PyGpuArray_DIM(input, 2) > 1024 || PyGpuArray_DIM(input, 3) > 1024 ||
theano/gpuarray/c_code/dnn_fwd.c:          (PyGpuArray_DIM(kerns, 2) == 1 && PyGpuArray_DIM(kerns, 3) == 1))
theano/gpuarray/c_code/dnn_fwd.c:APPLY_SPECIFIC(conv_fwd)(PyGpuArrayObject *input, PyGpuArrayObject *kerns,
theano/gpuarray/c_code/dnn_fwd.c:                         PyGpuArrayObject *om,
theano/gpuarray/c_code/dnn_fwd.c:                         PyGpuArrayObject **output,
theano/gpuarray/c_code/dnn_fwd.c:  PyGpuContextObject *c = input->context;
theano/gpuarray/c_code/dnn_fwd.c:  if (PyGpuArray_DIMS(input)[1] != PyGpuArray_DIMS(kerns)[1] * params->num_groups) {
theano/gpuarray/c_code/dnn_fwd.c:  if ((PyGpuArray_DIMS(kerns)[0] % params->num_groups) != 0) {
theano/gpuarray/c_code/dnn_fwd.c:    if (theano_prep_output(output, PyGpuArray_NDIM(om), PyGpuArray_DIMS(om),
theano/gpuarray/c_code/dnn_fwd.c:    if (beta != 0.0 && pygpu_move(*output, om))
theano/gpuarray/c_code/dnn_fwd.c:  if (PyGpuArray_DIMS(input)[0] == 0 || PyGpuArray_DIMS(kerns)[0] == 0 || PyGpuArray_DIMS(kerns)[1] == 0) {
theano/gpuarray/c_code/dnn_fwd.c:    int err2 = GpuArray_memset(&(*output)->ga, 0);
theano/gpuarray/c_code/dnn_fwd.c:                     "GpuDnnConv could not fill the output with zeros: %d", err2);
theano/gpuarray/c_code/dnn_fwd.c:  size_t input_offset = PyGpuArray_STRIDE(input, 0) / groups;
theano/gpuarray/c_code/dnn_fwd.c:  size_t kern_offset = PyGpuArray_STRIDE(kerns, 0) * PyGpuArray_DIM(kerns, 0) / groups;
theano/gpuarray/c_code/dnn_fwd.c:  size_t output_offset = PyGpuArray_STRIDE(*output, 0) / groups;
theano/gpuarray/c_code/dnn_fwd.c:  cuda_enter(c->ctx);
theano/gpuarray/c_code/dnn_fwd.c:    cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_fwd.c:      gpucontext_property(c->ctx, GA_CTX_PROP_UNIQUE_ID, pci_id);
theano/gpuarray/c_code/dnn_fwd.c:        cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_fwd.c:        gpudata *tmpmem;
theano/gpuarray/c_code/dnn_fwd.c:        tmpmem = gpudata_alloc(c->ctx, maxfree, NULL, 0, NULL);
theano/gpuarray/c_code/dnn_fwd.c:          PyErr_SetString(PyExc_MemoryError, "Could not allocate GPU memory for FindEx");
theano/gpuarray/c_code/dnn_fwd.c:          cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_fwd.c:        PyGpuArrayObject* o = *output;
theano/gpuarray/c_code/dnn_fwd.c:            o = pygpu_empty(PyGpuArray_NDIM(*output), PyGpuArray_DIMS(*output), (*output)->ga.typecode, GA_C_ORDER, c, Py_None);
theano/gpuarray/c_code/dnn_fwd.c:          params->handle, APPLY_SPECIFIC(input), PyGpuArray_DEV_DATA(input),
theano/gpuarray/c_code/dnn_fwd.c:          APPLY_SPECIFIC(kerns), PyGpuArray_DEV_DATA(kerns),
theano/gpuarray/c_code/dnn_fwd.c:          desc, APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(o),
theano/gpuarray/c_code/dnn_fwd.c:        gpudata_release(tmpmem);
theano/gpuarray/c_code/dnn_fwd.c:          cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_fwd.c:            cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_fwd.c:            cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_fwd.c:          cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_fwd.c:    cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_fwd.c:        cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_fwd.c:      cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_fwd.c:      cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_fwd.c:    gpudata *workspace = 0;
theano/gpuarray/c_code/dnn_fwd.c:      workspace = gpudata_alloc(c->ctx, worksize, NULL, 0, NULL);
theano/gpuarray/c_code/dnn_fwd.c:        cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_fwd.c:      cuda_wait(workspace, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_fwd.c:    cuda_wait(input->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_fwd.c:    cuda_wait(kerns->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_fwd.c:    cuda_wait((*output)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_fwd.c:    GpuArray_sync(&(*output)->ga);
theano/gpuarray/c_code/dnn_fwd.c:        APPLY_SPECIFIC(input), ((char *)PyGpuArray_DEV_DATA(input)) + input_offset * g,
theano/gpuarray/c_code/dnn_fwd.c:        APPLY_SPECIFIC(kerns), ((char *)PyGpuArray_DEV_DATA(kerns)) + kern_offset * g,
theano/gpuarray/c_code/dnn_fwd.c:        APPLY_SPECIFIC(output), ((char *)PyGpuArray_DEV_DATA(*output)) + output_offset * g);
theano/gpuarray/c_code/dnn_fwd.c:      cuda_record(workspace, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_fwd.c:      gpudata_release(workspace);
theano/gpuarray/c_code/dnn_fwd.c:    cuda_record(input->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_fwd.c:    cuda_record(kerns->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_fwd.c:    cuda_record((*output)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_fwd.c:  GpuArray_sync(&(*output)->ga);
theano/gpuarray/c_code/dnn_fwd.c:  cuda_exit(c->ctx);
theano/gpuarray/c_code/magma_cholesky.c:setup_ext_cuda();
theano/gpuarray/c_code/magma_cholesky.c:int APPLY_SPECIFIC(magma_cholesky)(PyGpuArrayObject *A, PyGpuArrayObject **L,
theano/gpuarray/c_code/magma_cholesky.c:                    "GpuMagmaCholesky: unsupported data type");
theano/gpuarray/c_code/magma_cholesky.c:  cuda_enter(params->context->ctx);
theano/gpuarray/c_code/magma_cholesky.c:  if (!GpuArray_IS_C_CONTIGUOUS(&A->ga)) {
theano/gpuarray/c_code/magma_cholesky.c:                    "GpuMagmaCholesky: requires data to be C-contiguous");
theano/gpuarray/c_code/magma_cholesky.c:  if (PyGpuArray_NDIM(A) != 2) {
theano/gpuarray/c_code/magma_cholesky.c:    PyErr_SetString(PyExc_ValueError, "GpuMagmaCholesky: matrix rank error");
theano/gpuarray/c_code/magma_cholesky.c:  dims = PyGpuArray_DIMS(A);
theano/gpuarray/c_code/magma_cholesky.c:    PyErr_SetString(PyExc_ValueError, "GpuMagmaCholesky: matrix is not square");
theano/gpuarray/c_code/magma_cholesky.c:          "GpuMagmaCholesky: failed to allocate memory for the output");
theano/gpuarray/c_code/magma_cholesky.c:  magma_spotrf_gpu(ul, N, (float *)PyGpuArray_DEV_DATA(*L), N, &info);
theano/gpuarray/c_code/magma_cholesky.c:    PyErr_Format(PyExc_RuntimeError, "GpuMagmaCholesky: the leading minor of "
theano/gpuarray/c_code/magma_cholesky.c:        "GpuMagmaCholesky: magma_spotrf_gpu argument %d has an illegal value",
theano/gpuarray/c_code/magma_cholesky.c:      PyErr_Format(PyExc_RuntimeError, "GpuMagmaCholesky: tril_kernel %s.",
theano/gpuarray/c_code/magma_cholesky.c:                   GpuKernel_error(&k_tril_kernel, res));
theano/gpuarray/c_code/magma_cholesky.c:      PyErr_Format(PyExc_RuntimeError, "GpuMagmaCholesky: triu_kernel %s.",
theano/gpuarray/c_code/magma_cholesky.c:                   GpuKernel_error(&k_triu_kernel, res));
theano/gpuarray/c_code/magma_cholesky.c:  cuda_exit(params->context->ctx);
theano/gpuarray/c_code/pool_grad_grad.c:int APPLY_SPECIFIC(pool_grad_grad)(PyGpuArrayObject *x,
theano/gpuarray/c_code/pool_grad_grad.c:                                   PyGpuArrayObject *z,
theano/gpuarray/c_code/pool_grad_grad.c:                                   PyGpuArrayObject *gx,
theano/gpuarray/c_code/pool_grad_grad.c:                                   PyGpuArrayObject **gz,
theano/gpuarray/c_code/pool_grad_grad.c:                                   PyGpuContextObject *ctx) {
theano/gpuarray/c_code/pool_grad_grad.c:  if (!GpuArray_IS_C_CONTIGUOUS(&x->ga)
theano/gpuarray/c_code/pool_grad_grad.c:      || !GpuArray_IS_C_CONTIGUOUS(&z->ga)
theano/gpuarray/c_code/pool_grad_grad.c:      || !GpuArray_IS_C_CONTIGUOUS(&gx->ga))
theano/gpuarray/c_code/pool_grad_grad.c:                   "GpuPoolingGradGrad: requires data to be C-contiguous");
theano/gpuarray/c_code/pool_grad_grad.c:  if (PyGpuArray_NDIM(x) != ndims + 2
theano/gpuarray/c_code/pool_grad_grad.c:      || PyGpuArray_NDIM(z) != ndims + 2
theano/gpuarray/c_code/pool_grad_grad.c:      || PyGpuArray_NDIM(gx) != ndims + 2)
theano/gpuarray/c_code/pool_grad_grad.c:      PyErr_SetString(PyExc_ValueError, "GpuPoolingGradGrad: rank error");
theano/gpuarray/c_code/pool_grad_grad.c:  if (theano_prep_output(gz, PyGpuArray_NDIM(z), PyGpuArray_DIMS(z),
theano/gpuarray/c_code/pool_grad_grad.c:                      "GpuPoolingGradGrad: failed to allocate memory");
theano/gpuarray/c_code/pool_grad_grad.c:    const size_t* z_dims = PyGpuArray_DIMS(z);
theano/gpuarray/c_code/pool_grad_grad.c:    const size_t* x_dims = PyGpuArray_DIMS(x);
theano/gpuarray/c_code/pool_grad_grad.c:                     "GpuPoolingGradGrad: max_pool2d_grad_grad_kernel %s.",
theano/gpuarray/c_code/pool_grad_grad.c:                     GpuKernel_error(&k_max_pool2d_grad_grad_kernel, err));
theano/gpuarray/c_code/pool_grad_grad.c:                     "GpuPoolingGradGrad: max_pool3d_grad_grad_kernel %s.",
theano/gpuarray/c_code/pool_grad_grad.c:                     GpuKernel_error(&k_max_pool3d_grad_grad_kernel, err));
theano/gpuarray/c_code/dnn_batchnorm_inf.c:int dnn_batchnorm_op(PyGpuArrayObject *inp, PyGpuArrayObject *scale,
theano/gpuarray/c_code/dnn_batchnorm_inf.c:                     PyGpuArrayObject *bias, PyGpuArrayObject *est_mean,
theano/gpuarray/c_code/dnn_batchnorm_inf.c:                     PyGpuArrayObject *est_var, npy_float64 epsilon, 
theano/gpuarray/c_code/dnn_batchnorm_inf.c:                     PyGpuArrayObject **outp, PARAMS_TYPE* params) {
theano/gpuarray/c_code/dnn_batchnorm_inf.c:  PyGpuContextObject *c = inp->context;
theano/gpuarray/c_code/dnn_batchnorm_inf.c:      PyGpuArray_DEV_DATA(inp),
theano/gpuarray/c_code/dnn_batchnorm_inf.c:      PyGpuArray_DEV_DATA(*outp),
theano/gpuarray/c_code/dnn_batchnorm_inf.c:      PyGpuArray_DEV_DATA(scale),
theano/gpuarray/c_code/dnn_batchnorm_inf.c:      PyGpuArray_DEV_DATA(bias),
theano/gpuarray/c_code/dnn_batchnorm_inf.c:      PyGpuArray_DEV_DATA(est_mean),
theano/gpuarray/c_code/dnn_batchnorm_inf.c:      PyGpuArray_DEV_DATA(est_var),
theano/gpuarray/c_code/magma_eigh.c:setup_ext_cuda();
theano/gpuarray/c_code/magma_eigh.c:int APPLY_SPECIFIC(magma_eigh)(PyGpuArrayObject *A_,
theano/gpuarray/c_code/magma_eigh.c:                               PyGpuArrayObject **D,
theano/gpuarray/c_code/magma_eigh.c:                               PyGpuArrayObject **V, // may be NULL
theano/gpuarray/c_code/magma_eigh.c:  PyGpuArrayObject *A = NULL;
theano/gpuarray/c_code/magma_eigh.c:                    "GpuMagmaEigh: Unsupported data type");
theano/gpuarray/c_code/magma_eigh.c:  cuda_enter(params->context->ctx);
theano/gpuarray/c_code/magma_eigh.c:  if (!GpuArray_IS_C_CONTIGUOUS(&A_->ga)) {
theano/gpuarray/c_code/magma_eigh.c:                    "GpuMagmaEigh: requires data to be C-contiguous");
theano/gpuarray/c_code/magma_eigh.c:  if (PyGpuArray_NDIM(A_) != 2) {
theano/gpuarray/c_code/magma_eigh.c:                    "GpuMagmaEigh: matrix rank error");
theano/gpuarray/c_code/magma_eigh.c:  if (PyGpuArray_DIM(A_, 0) != PyGpuArray_DIM(A_, 1)) {
theano/gpuarray/c_code/magma_eigh.c:                    "GpuMagmaEigh: matrix is not square");
theano/gpuarray/c_code/magma_eigh.c:  A = pygpu_copy(A_, GA_F_ORDER);
theano/gpuarray/c_code/magma_eigh.c:                    "GpuMagmaEigh: failed to change to column-major order");
theano/gpuarray/c_code/magma_eigh.c:  N = PyGpuArray_DIM(A, 0);
theano/gpuarray/c_code/magma_eigh.c:                    "GpuMagmaEigh: failed to allocate working memory");
theano/gpuarray/c_code/magma_eigh.c:                    "GpuMagmaEigh: failed to allocate working memory");
theano/gpuarray/c_code/magma_eigh.c:  magma_ssyevd_gpu(jobz, uplo, N, NULL, N, NULL, NULL, N, &lwork,
theano/gpuarray/c_code/magma_eigh.c:                    "GpuMagmaEigh: failed to allocate working memory");
theano/gpuarray/c_code/magma_eigh.c:                    "GpuMagmaEigh: failed to allocate working memory");
theano/gpuarray/c_code/magma_eigh.c:  magma_ssyevd_gpu(jobz, uplo, N, (float *)PyGpuArray_DEV_DATA(A), N, w_data,
theano/gpuarray/c_code/magma_eigh.c:        "GpuMagmaEigh:  %d off-diagonal elements of an didn't converge to zero",
theano/gpuarray/c_code/magma_eigh.c:        "GpuMagmaEigh: magma_ssyevd_gpu argument %d has an illegal value", -info);
theano/gpuarray/c_code/magma_eigh.c:                    "GpuMagmaEigh: failed to allocate memory for the output");
theano/gpuarray/c_code/magma_eigh.c:  cudaMemcpy(PyGpuArray_DEV_DATA(*D), w_data, N * sizeof(float),
theano/gpuarray/c_code/magma_eigh.c:             cudaMemcpyDeviceToDevice);
theano/gpuarray/c_code/magma_eigh.c:                      "GpuMagmaEigh: failed to allocate memory for the output");
theano/gpuarray/c_code/magma_eigh.c:  cuda_exit(params->context->ctx);
theano/gpuarray/c_code/pool_max_grad.c:int APPLY_SPECIFIC(max_pool_grad)(PyGpuArrayObject *x,
theano/gpuarray/c_code/pool_max_grad.c:                                  PyGpuArrayObject *z,
theano/gpuarray/c_code/pool_max_grad.c:                                  PyGpuArrayObject *gz,
theano/gpuarray/c_code/pool_max_grad.c:                                  PyGpuArrayObject **gx,
theano/gpuarray/c_code/pool_max_grad.c:                                  PyGpuContextObject *ctx) {
theano/gpuarray/c_code/pool_max_grad.c:  if (!GpuArray_IS_C_CONTIGUOUS(&x->ga)
theano/gpuarray/c_code/pool_max_grad.c:      || !GpuArray_IS_C_CONTIGUOUS(&z->ga)
theano/gpuarray/c_code/pool_max_grad.c:      || !GpuArray_IS_C_CONTIGUOUS(&gz->ga))
theano/gpuarray/c_code/pool_max_grad.c:                   "GpuMaxPoolGrad: requires data to be C-contiguous");
theano/gpuarray/c_code/pool_max_grad.c:  if (PyGpuArray_NDIM(x) != ndims + 2
theano/gpuarray/c_code/pool_max_grad.c:      || PyGpuArray_NDIM(z) != ndims + 2
theano/gpuarray/c_code/pool_max_grad.c:      || PyGpuArray_NDIM(gz) != ndims + 2)
theano/gpuarray/c_code/pool_max_grad.c:      PyErr_SetString(PyExc_ValueError, "GpuMaxPoolGrad: rank error");
theano/gpuarray/c_code/pool_max_grad.c:  if (theano_prep_output(gx, PyGpuArray_NDIM(x), PyGpuArray_DIMS(x),
theano/gpuarray/c_code/pool_max_grad.c:                      "GpuMaxPoolGrad: failed to allocate memory");
theano/gpuarray/c_code/pool_max_grad.c:    const size_t* z_dims = PyGpuArray_DIMS(z);
theano/gpuarray/c_code/pool_max_grad.c:    const size_t* x_dims = PyGpuArray_DIMS(x);
theano/gpuarray/c_code/pool_max_grad.c:                     "GpuMaxPoolGrad: max_pool2d_grad_kernel %s.",
theano/gpuarray/c_code/pool_max_grad.c:                     GpuKernel_error(&k_max_pool2d_grad_kernel, err));
theano/gpuarray/c_code/pool_max_grad.c:                     "GpuMaxPoolGrad: max_pool3d_grad_kernel %s.",
theano/gpuarray/c_code/pool_max_grad.c:                     GpuKernel_error(&k_max_pool3d_grad_kernel, err));
theano/gpuarray/c_code/magma_qr.c:setup_ext_cuda();
theano/gpuarray/c_code/magma_qr.c:static PyGpuArrayObject *pygpu_narrow(PyGpuArrayObject *src, size_t dim,
theano/gpuarray/c_code/magma_qr.c:  PyGpuArrayObject *src_view = pygpu_view(src, Py_None);
theano/gpuarray/c_code/magma_qr.c:  GpuArray_fix_flags(&src_view->ga);
theano/gpuarray/c_code/magma_qr.c:  return pygpu_copy(src_view, GA_C_ORDER);
theano/gpuarray/c_code/magma_qr.c:int APPLY_SPECIFIC(magma_qr)(PyGpuArrayObject *A_,
theano/gpuarray/c_code/magma_qr.c:                             PyGpuArrayObject **R,
theano/gpuarray/c_code/magma_qr.c:                             PyGpuArrayObject **Q, // may be NULL
theano/gpuarray/c_code/magma_qr.c:  PyGpuArrayObject *A = NULL;
theano/gpuarray/c_code/magma_qr.c:  gpudata *work_data = NULL;
theano/gpuarray/c_code/magma_qr.c:                    "GpuMagmaQR: Unsupported data type");
theano/gpuarray/c_code/magma_qr.c:  cuda_enter(params->context->ctx);
theano/gpuarray/c_code/magma_qr.c:  if (!GpuArray_IS_C_CONTIGUOUS(&A->ga)) {
theano/gpuarray/c_code/magma_qr.c:                    "GpuMagmaQR: requires data to be C-contiguous");
theano/gpuarray/c_code/magma_qr.c:  if (PyGpuArray_NDIM(A) != 2) {
theano/gpuarray/c_code/magma_qr.c:    PyErr_SetString(PyExc_ValueError, "GpuMagmaQR: matrix rank error");
theano/gpuarray/c_code/magma_qr.c:  A = pygpu_copy(A_, GA_F_ORDER);
theano/gpuarray/c_code/magma_qr.c:                    "GpuMagmaQR: failed to change to column-major order");
theano/gpuarray/c_code/magma_qr.c:  M = PyGpuArray_DIM(A, 0);
theano/gpuarray/c_code/magma_qr.c:  N = PyGpuArray_DIM(A, 1);
theano/gpuarray/c_code/magma_qr.c:                    "GpuMagmaQR: failed to allocate working memory");
theano/gpuarray/c_code/magma_qr.c:  work_data = gpudata_alloc(params->context->ctx, ldwork * sizeof(float), NULL, 0, NULL);
theano/gpuarray/c_code/magma_qr.c:                    "GpuMagmaQR: failed to allocate working memory");
theano/gpuarray/c_code/magma_qr.c:  magma_sgeqrf2_gpu(M, N, (float *)PyGpuArray_DEV_DATA(A), M, tau_data, &info);
theano/gpuarray/c_code/magma_qr.c:        "GpuMagmaQR: magma_sgeqrf2_gpu argument %d has an illegal value", -info);
theano/gpuarray/c_code/magma_qr.c:  *R = pygpu_narrow(A, 0, K);
theano/gpuarray/c_code/magma_qr.c:    PyErr_SetString(PyExc_RuntimeError, "GpuMagmaQR: failed to narrow array");
theano/gpuarray/c_code/magma_qr.c:    PyErr_Format(PyExc_RuntimeError, "GpuMagmaQR: triu_kernel %s.",
theano/gpuarray/c_code/magma_qr.c:                 GpuKernel_error(&k_triu_kernel, res));
theano/gpuarray/c_code/magma_qr.c:    A = pygpu_copy(A_, GA_F_ORDER);
theano/gpuarray/c_code/magma_qr.c:                      "GpuMagmaQR: failed to change to column-major order");
theano/gpuarray/c_code/magma_qr.c:    magma_sgeqrf_gpu(M, N, (float *)PyGpuArray_DEV_DATA(A), M, tau_data,
theano/gpuarray/c_code/magma_qr.c:                   "GpuMagmaQR: magma_sgeqrf_gpu argument %d has an illegal value", -info);
theano/gpuarray/c_code/magma_qr.c:    magma_sorgqr_gpu(M, K, K, (float *)PyGpuArray_DEV_DATA(A), M, tau_data,
theano/gpuarray/c_code/magma_qr.c:                   "GpuMagmaQR: magma_sorgqr_gpu argument %d has an illegal value", -info);
theano/gpuarray/c_code/magma_qr.c:    *Q = pygpu_narrow(A, 1, K);
theano/gpuarray/c_code/magma_qr.c:      PyErr_SetString(PyExc_RuntimeError, "GpuMagmaQR: failed to narrow array");
theano/gpuarray/c_code/magma_qr.c:    gpudata_release(work_data);
theano/gpuarray/c_code/magma_qr.c:  cuda_exit(params->context->ctx);
theano/gpuarray/c_code/dnn_gw.c:hash_prefix = std::string("GW|GPU#");
theano/gpuarray/c_code/dnn_gw.c:                         const PyGpuArrayObject* input,
theano/gpuarray/c_code/dnn_gw.c:                         const PyGpuArrayObject* kerns,
theano/gpuarray/c_code/dnn_gw.c:      PyGpuArray_NDIM(input) == 4) {
theano/gpuarray/c_code/dnn_gw.c:        PyGpuArray_DIM(input, 2) > 1024 || PyGpuArray_DIM(input, 3) > 1024 ||
theano/gpuarray/c_code/dnn_gw.c:        (PyGpuArray_DIM(kerns, 2) == 1 && PyGpuArray_DIM(kerns, 3) == 1)) {
theano/gpuarray/c_code/dnn_gw.c:APPLY_SPECIFIC(conv_gw)(PyGpuArrayObject *input, PyGpuArrayObject *output,
theano/gpuarray/c_code/dnn_gw.c:                        PyGpuArrayObject *km,
theano/gpuarray/c_code/dnn_gw.c:                        double alpha, double beta, PyGpuArrayObject **kerns,
theano/gpuarray/c_code/dnn_gw.c:  PyGpuContextObject *c = input->context;
theano/gpuarray/c_code/dnn_gw.c:  if (PyGpuArray_DIMS(input)[1] != PyGpuArray_DIMS(km)[1] * params->num_groups) {
theano/gpuarray/c_code/dnn_gw.c:                    "GpuDnnConv images and kernel must have the same stack size");
theano/gpuarray/c_code/dnn_gw.c:  if ((PyGpuArray_DIMS(output)[1] % params->num_groups) != 0) {
theano/gpuarray/c_code/dnn_gw.c:    if (theano_prep_output(kerns, PyGpuArray_NDIM(km), PyGpuArray_DIMS(km),
theano/gpuarray/c_code/dnn_gw.c:    if (beta != 0.0 && pygpu_move(*kerns, km))
theano/gpuarray/c_code/dnn_gw.c:  if (PyGpuArray_DIMS(input)[0] == 0 || PyGpuArray_DIMS(km)[0] == 0 || PyGpuArray_DIMS(km)[1] == 0) {
theano/gpuarray/c_code/dnn_gw.c:    int err2 = GpuArray_memset(&(*kerns)->ga, 0);
theano/gpuarray/c_code/dnn_gw.c:                     "GpuDnnConv grad wrt. weights could not fill the output with zeros: %d", err2);
theano/gpuarray/c_code/dnn_gw.c:                                        PyGpuArray_NDIM(*kerns), output, groups))
theano/gpuarray/c_code/dnn_gw.c:  size_t input_offset = PyGpuArray_STRIDE(input, 0) / groups;
theano/gpuarray/c_code/dnn_gw.c:  size_t kern_offset = PyGpuArray_STRIDE(*kerns, 0) * PyGpuArray_DIM(*kerns, 0) / groups;
theano/gpuarray/c_code/dnn_gw.c:  size_t output_offset = PyGpuArray_STRIDE(output, 0) / groups;
theano/gpuarray/c_code/dnn_gw.c:  cuda_enter(c->ctx);
theano/gpuarray/c_code/dnn_gw.c:    cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gw.c:      gpucontext_property(c->ctx, GA_CTX_PROP_UNIQUE_ID, pci_id);
theano/gpuarray/c_code/dnn_gw.c:        cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gw.c:        gpudata *tmpmem;
theano/gpuarray/c_code/dnn_gw.c:        tmpmem = gpudata_alloc(c->ctx, maxfree, NULL, 0, NULL);
theano/gpuarray/c_code/dnn_gw.c:          PyErr_SetString(PyExc_MemoryError, "Could not allocate working GPU memory");
theano/gpuarray/c_code/dnn_gw.c:          cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gw.c:        PyGpuArrayObject* k = *kerns;
theano/gpuarray/c_code/dnn_gw.c:            k = pygpu_empty(PyGpuArray_NDIM(*kerns), PyGpuArray_DIMS(*kerns), (*kerns)->ga.typecode, GA_C_ORDER, c, Py_None);
theano/gpuarray/c_code/dnn_gw.c:          params->handle, APPLY_SPECIFIC(input), PyGpuArray_DEV_DATA(input),
theano/gpuarray/c_code/dnn_gw.c:          APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(output), desc,
theano/gpuarray/c_code/dnn_gw.c:          APPLY_SPECIFIC(kerns), PyGpuArray_DEV_DATA(k),
theano/gpuarray/c_code/dnn_gw.c:        gpudata_release(tmpmem);
theano/gpuarray/c_code/dnn_gw.c:          cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gw.c:            cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gw.c:            cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gw.c:          cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gw.c:    cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gw.c:        cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gw.c:      cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gw.c:      cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gw.c:  gpudata *workspace = 0;
theano/gpuarray/c_code/dnn_gw.c:    workspace = gpudata_alloc(c->ctx, worksize, NULL, 0, NULL);
theano/gpuarray/c_code/dnn_gw.c:      cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_gw.c:    cuda_wait(workspace, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_gw.c:  cuda_wait(input->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_gw.c:  cuda_wait(output->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_gw.c:  cuda_wait((*kerns)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_gw.c:  GpuArray_sync(&(*kerns)->ga);
theano/gpuarray/c_code/dnn_gw.c:      APPLY_SPECIFIC(input), ((char *)PyGpuArray_DEV_DATA(input)) + input_offset * g ,
theano/gpuarray/c_code/dnn_gw.c:      APPLY_SPECIFIC(output), ((char *)PyGpuArray_DEV_DATA(output)) + output_offset * g,
theano/gpuarray/c_code/dnn_gw.c:      APPLY_SPECIFIC(kerns), ((char *)PyGpuArray_DEV_DATA(*kerns)) + kern_offset * g);
theano/gpuarray/c_code/dnn_gw.c:    cuda_record(workspace, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_gw.c:    gpudata_release(workspace);
theano/gpuarray/c_code/dnn_gw.c:  cuda_record(input->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_gw.c:  cuda_record(output->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_gw.c:  cuda_record((*kerns)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_gw.c:  GpuArray_sync(&(*kerns)->ga);
theano/gpuarray/c_code/dnn_gw.c:  cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_batchnorm_grad.c:int dnn_batchnorm_grad(PyGpuArrayObject *inp, PyGpuArrayObject *doutp,
theano/gpuarray/c_code/dnn_batchnorm_grad.c:                       PyGpuArrayObject *scale, PyGpuArrayObject *x_mean,
theano/gpuarray/c_code/dnn_batchnorm_grad.c:                       PyGpuArrayObject *x_invstd, npy_float64 epsilon,
theano/gpuarray/c_code/dnn_batchnorm_grad.c:                       PyGpuArrayObject **dinp, PyGpuArrayObject **dscale,
theano/gpuarray/c_code/dnn_batchnorm_grad.c:                       PyGpuArrayObject **dbias, PARAMS_TYPE* params) {
theano/gpuarray/c_code/dnn_batchnorm_grad.c:  PyGpuContextObject *c = inp->context;
theano/gpuarray/c_code/dnn_batchnorm_grad.c:      PyGpuArray_DEV_DATA(inp),
theano/gpuarray/c_code/dnn_batchnorm_grad.c:      PyGpuArray_DEV_DATA(doutp),
theano/gpuarray/c_code/dnn_batchnorm_grad.c:      PyGpuArray_DEV_DATA(*dinp),
theano/gpuarray/c_code/dnn_batchnorm_grad.c:      PyGpuArray_DEV_DATA(scale),
theano/gpuarray/c_code/dnn_batchnorm_grad.c:      PyGpuArray_DEV_DATA(*dscale),
theano/gpuarray/c_code/dnn_batchnorm_grad.c:      PyGpuArray_DEV_DATA(*dbias),
theano/gpuarray/c_code/dnn_batchnorm_grad.c:      PyGpuArray_DEV_DATA(x_mean),
theano/gpuarray/c_code/dnn_batchnorm_grad.c:      PyGpuArray_DEV_DATA(x_invstd)
theano/gpuarray/c_code/pool_max_rop.c:int APPLY_SPECIFIC(max_pool_rop)(PyGpuArrayObject *x,
theano/gpuarray/c_code/pool_max_rop.c:                                 PyGpuArrayObject *ex,
theano/gpuarray/c_code/pool_max_rop.c:                                 PyGpuArrayObject **z,
theano/gpuarray/c_code/pool_max_rop.c:  if (!GpuArray_IS_C_CONTIGUOUS(&x->ga) || !GpuArray_IS_C_CONTIGUOUS(&ex->ga))
theano/gpuarray/c_code/pool_max_rop.c:                   "GpuMaxPoolRop: requires data to be C-contiguous");
theano/gpuarray/c_code/pool_max_rop.c:  if (PyGpuArray_NDIM(x) != ndims + 2 || PyGpuArray_NDIM(ex) != ndims + 2)
theano/gpuarray/c_code/pool_max_rop.c:      PyErr_SetString(PyExc_ValueError, "GpuMaxPoolRop: rank error");
theano/gpuarray/c_code/pool_max_rop.c:  const size_t* x_dims = PyGpuArray_DIMS(x);
theano/gpuarray/c_code/pool_max_rop.c:                    "GpuMaxPoolRop: padding works only with ignore_border=True");
theano/gpuarray/c_code/pool_max_rop.c:  if (theano_prep_output(z, PyGpuArray_NDIM(ex), z_dims,
theano/gpuarray/c_code/pool_max_rop.c:                      "GpuMaxPoolRop: failed to allocate memory");
theano/gpuarray/c_code/pool_max_rop.c:                     "GpuMaxPoolRop: max_pool2d_rop_kernel %s.",
theano/gpuarray/c_code/pool_max_rop.c:                     GpuKernel_error(&k_max_pool2d_rop_kernel, err));
theano/gpuarray/c_code/pool_max_rop.c:                     "GpuMaxPoolRop: max_pool3d_rop_kernel %s.",
theano/gpuarray/c_code/pool_max_rop.c:                     GpuKernel_error(&k_max_pool2d_rop_kernel, err));
theano/gpuarray/c_code/dnn_redux.c:GpuElemwise* elemwise;
theano/gpuarray/c_code/dnn_redux.c:gpuelemwise_arg arg;
theano/gpuarray/c_code/dnn_redux.c:    GpuElemwise_free(elemwise);
theano/gpuarray/c_code/dnn_redux.c:int APPLY_SPECIFIC(dnn_redux)(PyGpuArrayObject *input,
theano/gpuarray/c_code/dnn_redux.c:                              PyGpuArrayObject **output,
theano/gpuarray/c_code/dnn_redux.c:                              PyGpuArrayObject **indices,
theano/gpuarray/c_code/dnn_redux.c:  PyGpuContextObject *c = input->context;
theano/gpuarray/c_code/dnn_redux.c:  gpudata *workspace = NULL;
theano/gpuarray/c_code/dnn_redux.c:  if (!GpuArray_IS_C_CONTIGUOUS(&input->ga)) {
theano/gpuarray/c_code/dnn_redux.c:  for (unsigned int i = 0; i < PyGpuArray_NDIM(input); i++) {
theano/gpuarray/c_code/dnn_redux.c:      dims[p] = PyGpuArray_DIM(input, i);
theano/gpuarray/c_code/dnn_redux.c:      rsz *= PyGpuArray_DIM(input, i);
theano/gpuarray/c_code/dnn_redux.c:    indsize = PyGpuArray_SIZE(*indices) * 4;
theano/gpuarray/c_code/dnn_redux.c:    *output = pygpu_copy(input, GA_C_ORDER);
theano/gpuarray/c_code/dnn_redux.c:    err = GpuArray_reshape_inplace(&(*output)->ga, p, dims, GA_C_ORDER);
theano/gpuarray/c_code/dnn_redux.c:      PyErr_Format(PyExc_RuntimeError, "GpuArray_reshape_inplace: %s", GpuArray_error(&(*output)->ga, err));
theano/gpuarray/c_code/dnn_redux.c:       * cuDNN (up to 7004) does not support this case. Let's use GpuElemwise. */
theano/gpuarray/c_code/dnn_redux.c:              elemwise = GpuElemwise_new(c->ctx, "", "out = (out < 0 ? -out : out)", 1, &arg, p, GE_CONVERT_F16);
theano/gpuarray/c_code/dnn_redux.c:                  PyErr_SetString(PyExc_RuntimeError, "Unable to create GpuElemwise for output.");
theano/gpuarray/c_code/dnn_redux.c:            int err = GpuElemwise_call(elemwise, args, 0);
theano/gpuarray/c_code/dnn_redux.c:                PyErr_SetString(PyExc_RuntimeError, "Unable to call GpuElemwise on output.");
theano/gpuarray/c_code/dnn_redux.c:      err = GpuArray_memset(&(*indices)->ga, 0);
theano/gpuarray/c_code/dnn_redux.c:        PyErr_Format(PyExc_RuntimeError, "GpuArray_memset: %s", GpuArray_error(&(*indices)->ga, err));
theano/gpuarray/c_code/dnn_redux.c:  for (unsigned int i = 0; i < PyGpuArray_NDIM(input); i++) {
theano/gpuarray/c_code/dnn_redux.c:      dims[i] = PyGpuArray_DIM(input, i);
theano/gpuarray/c_code/dnn_redux.c:      strs[i] = PyGpuArray_STRIDE(*output, p);
theano/gpuarray/c_code/dnn_redux.c:    workspace = gpudata_alloc(c->ctx, worksize, NULL, 0, &e);
theano/gpuarray/c_code/dnn_redux.c:      PyErr_Format(PyExc_RuntimeError, "gpudata_alloc: %s",
theano/gpuarray/c_code/dnn_redux.c:                   gpucontext_error(c->ctx, e));
theano/gpuarray/c_code/dnn_redux.c:                          indices ? PyGpuArray_DEV_DATA(*indices) : NULL, indsize,
theano/gpuarray/c_code/dnn_redux.c:                          APPLY_SPECIFIC(input), PyGpuArray_DEV_DATA(input),
theano/gpuarray/c_code/dnn_redux.c:                          APPLY_SPECIFIC(output), PyGpuArray_DEV_DATA(*output));
theano/gpuarray/c_code/dnn_redux.c:    gpudata_release(workspace);
theano/gpuarray/c_code/magma_svd.c:setup_ext_cuda();
theano/gpuarray/c_code/magma_svd.c:int APPLY_SPECIFIC(magma_svd)(PyGpuArrayObject *A,
theano/gpuarray/c_code/magma_svd.c:                              PyGpuArrayObject **S,
theano/gpuarray/c_code/magma_svd.c:                              PyGpuArrayObject **U, // may be NULL
theano/gpuarray/c_code/magma_svd.c:                              PyGpuArrayObject **VT, // may be NULL
theano/gpuarray/c_code/magma_svd.c:                    "GpuMagmaSVD: Unsupported data type");
theano/gpuarray/c_code/magma_svd.c:  cuda_enter(params->context->ctx);
theano/gpuarray/c_code/magma_svd.c:  if (!GpuArray_IS_C_CONTIGUOUS(&A->ga)) {
theano/gpuarray/c_code/magma_svd.c:                    "GpuMagmaSVD: requires data to be C-contiguous");
theano/gpuarray/c_code/magma_svd.c:  if (PyGpuArray_NDIM(A) != 2) {
theano/gpuarray/c_code/magma_svd.c:                    "GpuMagmaSVD: matrix rank error");
theano/gpuarray/c_code/magma_svd.c:  M = PyGpuArray_DIM(A, 1);
theano/gpuarray/c_code/magma_svd.c:  N = PyGpuArray_DIM(A, 0);
theano/gpuarray/c_code/magma_svd.c:                    "GpuMagmaSVD: failed to allocate memory");
theano/gpuarray/c_code/magma_svd.c:  cudaMemcpy(a_data, PyGpuArray_DEV_DATA(A), M * N * sizeof(float),
theano/gpuarray/c_code/magma_svd.c:             cudaMemcpyDeviceToDevice);
theano/gpuarray/c_code/magma_svd.c:                    "GpuMagmaSVD: failed to allocate memory");
theano/gpuarray/c_code/magma_svd.c:                      "GpuMagmaSVD: failed to allocate memory");
theano/gpuarray/c_code/magma_svd.c:                      "GpuMagmaSVD: failed to allocate memory");
theano/gpuarray/c_code/magma_svd.c:                    "GpuMagmaSVD: failed to allocate working memory");
theano/gpuarray/c_code/magma_svd.c:                    "GpuMagmaSVD: failed to allocate working memory");
theano/gpuarray/c_code/magma_svd.c:        "GpuMagmaSVD: the updating process of SBDSDC did not converge (error: %d)",
theano/gpuarray/c_code/magma_svd.c:        "GpuMagmaSVD: magma_sgesdd_gpu argument %d has an illegal value", -info);
theano/gpuarray/c_code/magma_svd.c:                    "GpuMagmaSVD: failed to allocate memory");
theano/gpuarray/c_code/magma_svd.c:  cudaMemcpy(PyGpuArray_DEV_DATA(*S), s_data, K * sizeof(float),
theano/gpuarray/c_code/magma_svd.c:             cudaMemcpyDeviceToDevice);
theano/gpuarray/c_code/magma_svd.c:                      "GpuMagmaSVD: failed to allocate memory");
theano/gpuarray/c_code/magma_svd.c:    cudaMemcpy(PyGpuArray_DEV_DATA(*U), vt_data, N * N_VT * sizeof(float),
theano/gpuarray/c_code/magma_svd.c:               cudaMemcpyDeviceToDevice);
theano/gpuarray/c_code/magma_svd.c:                      "GpuMagmaSVD: failed to allocate memory");
theano/gpuarray/c_code/magma_svd.c:    cudaMemcpy(PyGpuArray_DEV_DATA(*VT), u_data, M_U * M * sizeof(float),
theano/gpuarray/c_code/magma_svd.c:               cudaMemcpyDeviceToDevice);
theano/gpuarray/c_code/magma_svd.c:  cuda_exit(params->context->ctx);
theano/gpuarray/c_code/dnn_rnn_gw.c:               PyGpuArrayObject *x, PyGpuArrayObject *hx,
theano/gpuarray/c_code/dnn_rnn_gw.c:               PyGpuArrayObject *y, gpudata *reserve,
theano/gpuarray/c_code/dnn_rnn_gw.c:               PyGpuArrayObject **dw, cudnnHandle_t _handle) {
theano/gpuarray/c_code/dnn_rnn_gw.c:  PyGpuContextObject *c = x->context;
theano/gpuarray/c_code/dnn_rnn_gw.c:  gpudata *workspace = NULL;
theano/gpuarray/c_code/dnn_rnn_gw.c:  size_t iters = PyGpuArray_DIM(x, 0);
theano/gpuarray/c_code/dnn_rnn_gw.c:  cuda_enter(c->ctx);
theano/gpuarray/c_code/dnn_rnn_gw.c:  dims[0] = PyGpuArray_DIM(x, 1);
theano/gpuarray/c_code/dnn_rnn_gw.c:  dims[1] = PyGpuArray_DIM(x, 2);
theano/gpuarray/c_code/dnn_rnn_gw.c:  dims[0] = PyGpuArray_DIM(y, 1);
theano/gpuarray/c_code/dnn_rnn_gw.c:  dims[1] = PyGpuArray_DIM(y, 2);
theano/gpuarray/c_code/dnn_rnn_gw.c:  GpuArray_memset(&(*dw)->ga, 0);
theano/gpuarray/c_code/dnn_rnn_gw.c:  workspace = gpudata_alloc(c->ctx, worksize, NULL, 0, NULL);
theano/gpuarray/c_code/dnn_rnn_gw.c:                                xl, PyGpuArray_DEV_DATA(x),
theano/gpuarray/c_code/dnn_rnn_gw.c:                                hxdesc, PyGpuArray_DEV_DATA(hx),
theano/gpuarray/c_code/dnn_rnn_gw.c:                                yl, PyGpuArray_DEV_DATA(y),
theano/gpuarray/c_code/dnn_rnn_gw.c:                                dwdesc, PyGpuArray_DEV_DATA(*dw),
theano/gpuarray/c_code/dnn_rnn_gw.c:    gpudata_release(workspace);
theano/gpuarray/c_code/dnn_rnn_gw.c:  cuda_exit(c->ctx);
theano/gpuarray/c_code/pool.c:int APPLY_SPECIFIC(pool)(PyGpuArrayObject *x,
theano/gpuarray/c_code/pool.c:                         PyGpuArrayObject **z,
theano/gpuarray/c_code/pool.c:  if (!GpuArray_IS_C_CONTIGUOUS(&x->ga))
theano/gpuarray/c_code/pool.c:                   "GpuPool: requires data to be C-contiguous");
theano/gpuarray/c_code/pool.c:  if (PyGpuArray_NDIM(x) != ndims + 2)
theano/gpuarray/c_code/pool.c:      PyErr_SetString(PyExc_ValueError, "GpuPool: rank error");
theano/gpuarray/c_code/pool.c:  const size_t* x_dims = PyGpuArray_DIMS(x);
theano/gpuarray/c_code/pool.c:                    "GpuPool: padding works only with ignore_border=True");
theano/gpuarray/c_code/pool.c:  if (theano_prep_output(z, PyGpuArray_NDIM(x), z_dims,
theano/gpuarray/c_code/pool.c:                      "GpuPool: failed to allocate memory");
theano/gpuarray/c_code/pool.c:                       "GpuPool: max_pool2d_kernel %s.",
theano/gpuarray/c_code/pool.c:                       GpuKernel_error(&k_max_pool2d_kernel, err));
theano/gpuarray/c_code/pool.c:                       "GpuPool: ave_pool2d_kernel %s.",
theano/gpuarray/c_code/pool.c:                       GpuKernel_error(&k_ave_pool2d_kernel, err));
theano/gpuarray/c_code/pool.c:                       "GpuPool: max_pool3d_kernel %s.",
theano/gpuarray/c_code/pool.c:                       GpuKernel_error(&k_max_pool2d_kernel, err));
theano/gpuarray/c_code/pool.c:                       "GpuPool: ave_pool3d_kernel %s.",
theano/gpuarray/c_code/pool.c:                       GpuKernel_error(&k_ave_pool3d_kernel, err));
theano/gpuarray/c_code/ctc_wrapper.c:setup_ext_cuda();
theano/gpuarray/c_code/ctc_wrapper.c:    gpudata * workspace;
theano/gpuarray/c_code/ctc_wrapper.c:void ctc_context_init(ctc_context_t * context, PyGpuContextObject * gpu_context)
theano/gpuarray/c_code/ctc_wrapper.c:    context->options.loc = CTC_GPU;
theano/gpuarray/c_code/ctc_wrapper.c:    // Get CUDA function pointer to obtain stream
theano/gpuarray/c_code/ctc_wrapper.c:    CUstream (*getstream_func_ptr)(void *) = (CUstream (*)(void *)) gpuarray_get_extension( "cuda_get_stream" );
theano/gpuarray/c_code/ctc_wrapper.c:    context->options.stream = getstream_func_ptr(gpu_context->ctx);
theano/gpuarray/c_code/ctc_wrapper.c:    gpudata_release( context->workspace );
theano/gpuarray/c_code/ctc_wrapper.c:                      "GpuConnectionistTemporalClassification: %s CTC error: %s",
theano/gpuarray/c_code/ctc_wrapper.c:int APPLY_SPECIFIC(ctc_cost_gpu)(PyGpuArrayObject   *  in_activations,
theano/gpuarray/c_code/ctc_wrapper.c:                                 PyGpuArrayObject   ** out_costs,
theano/gpuarray/c_code/ctc_wrapper.c:                                 PyGpuArrayObject   ** out_gradients,
theano/gpuarray/c_code/ctc_wrapper.c:                                 PyGpuContextObject *  gpu_context)
theano/gpuarray/c_code/ctc_wrapper.c:    size_t gpu_workspace_size;
theano/gpuarray/c_code/ctc_wrapper.c:    const size_t num_activations = PyGpuArray_DIMS( in_activations )[0];
theano/gpuarray/c_code/ctc_wrapper.c:    const size_t minibatch_size = PyGpuArray_DIMS( in_activations )[1];
theano/gpuarray/c_code/ctc_wrapper.c:    const size_t alphabet_size = PyGpuArray_DIMS( in_activations )[2];
theano/gpuarray/c_code/ctc_wrapper.c:    cuda_enter( gpu_context->ctx );
theano/gpuarray/c_code/ctc_wrapper.c:    ctc_context_init( context, gpu_context );
theano/gpuarray/c_code/ctc_wrapper.c:        activations = (float *) PyGpuArray_DEV_DATA( in_activations );
theano/gpuarray/c_code/ctc_wrapper.c:        cuda_exit( gpu_context->ctx );
theano/gpuarray/c_code/ctc_wrapper.c:            "GpuConnectionistTemporalClassification: Unsupported type for activations." );
theano/gpuarray/c_code/ctc_wrapper.c:        cuda_exit( gpu_context->ctx );
theano/gpuarray/c_code/ctc_wrapper.c:            "GpuConnectionistTemporalClassification: Could not allocate memory for input lengths." );
theano/gpuarray/c_code/ctc_wrapper.c:        cuda_exit( gpu_context->ctx );
theano/gpuarray/c_code/ctc_wrapper.c:            "GpuConnectionistTemporalClassification: Could not allocate memory for labels and their lengths." );
theano/gpuarray/c_code/ctc_wrapper.c:                             GA_C_ORDER, gpu_context ) != 0 )
theano/gpuarray/c_code/ctc_wrapper.c:        cuda_exit( gpu_context->ctx );
theano/gpuarray/c_code/ctc_wrapper.c:    GpuArray_memset( &((*out_costs)->ga), 0 );
theano/gpuarray/c_code/ctc_wrapper.c:    costs = (float *) PyGpuArray_DEV_DATA( *out_costs );
theano/gpuarray/c_code/ctc_wrapper.c:                                 GA_C_ORDER, gpu_context ) != 0 )
theano/gpuarray/c_code/ctc_wrapper.c:            cuda_exit( gpu_context->ctx );
theano/gpuarray/c_code/ctc_wrapper.c:        GpuArray_memset( &((*out_gradients)->ga), 0 );
theano/gpuarray/c_code/ctc_wrapper.c:        gradients = (float *) PyGpuArray_DEV_DATA( *out_gradients );
theano/gpuarray/c_code/ctc_wrapper.c:        &gpu_workspace_size ),
theano/gpuarray/c_code/ctc_wrapper.c:        cuda_exit( gpu_context->ctx );
theano/gpuarray/c_code/ctc_wrapper.c:    context->workspace = gpudata_alloc( gpu_context->ctx, gpu_workspace_size, NULL, 0, NULL );
theano/gpuarray/c_code/ctc_wrapper.c:        cuda_exit( gpu_context->ctx );
theano/gpuarray/c_code/ctc_wrapper.c:            "GpuConnectionistTemporalClassification: Failed to allocate memory for CTC workspace." );
theano/gpuarray/c_code/ctc_wrapper.c:    cuda_wait( in_activations->ga.data, GPUARRAY_CUDA_WAIT_READ );
theano/gpuarray/c_code/ctc_wrapper.c:    cuda_wait( (*out_costs)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
theano/gpuarray/c_code/ctc_wrapper.c:        cuda_wait( (*out_gradients)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
theano/gpuarray/c_code/ctc_wrapper.c:    cuda_record( in_activations->ga.data, GPUARRAY_CUDA_WAIT_READ );
theano/gpuarray/c_code/ctc_wrapper.c:    cuda_record( (*out_costs)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
theano/gpuarray/c_code/ctc_wrapper.c:        cuda_record( (*out_gradients)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
theano/gpuarray/c_code/ctc_wrapper.c:        cuda_exit( gpu_context->ctx );
theano/gpuarray/c_code/ctc_wrapper.c:    cuda_exit( gpu_context->ctx );
theano/gpuarray/c_code/dnn_batchnorm.c:int dnn_batchnorm_op(PyGpuArrayObject *inp, PyGpuArrayObject *scale,
theano/gpuarray/c_code/dnn_batchnorm.c:                     PyGpuArrayObject *bias, npy_float64 epsilon,
theano/gpuarray/c_code/dnn_batchnorm.c:                     PyGpuArrayObject *in_running_mean, // may be NULL
theano/gpuarray/c_code/dnn_batchnorm.c:                     PyGpuArrayObject *in_running_var, // may be NULL
theano/gpuarray/c_code/dnn_batchnorm.c:                     PyGpuArrayObject **outp,
theano/gpuarray/c_code/dnn_batchnorm.c:                     PyGpuArrayObject **x_mean,
theano/gpuarray/c_code/dnn_batchnorm.c:                     PyGpuArrayObject **x_invstd,
theano/gpuarray/c_code/dnn_batchnorm.c:                     PyGpuArrayObject **out_running_mean, // may be NULL
theano/gpuarray/c_code/dnn_batchnorm.c:                     PyGpuArrayObject **out_running_var, // may be NULL
theano/gpuarray/c_code/dnn_batchnorm.c:  PyGpuContextObject *c = inp->context;
theano/gpuarray/c_code/dnn_batchnorm.c:  PyGpuArrayObject *running_mean = NULL;
theano/gpuarray/c_code/dnn_batchnorm.c:  PyGpuArrayObject *running_var = NULL;
theano/gpuarray/c_code/dnn_batchnorm.c:      PyGpuArray_DEV_DATA(inp),
theano/gpuarray/c_code/dnn_batchnorm.c:      PyGpuArray_DEV_DATA(*outp),
theano/gpuarray/c_code/dnn_batchnorm.c:      PyGpuArray_DEV_DATA(scale),
theano/gpuarray/c_code/dnn_batchnorm.c:      PyGpuArray_DEV_DATA(bias),
theano/gpuarray/c_code/dnn_batchnorm.c:      running_averages ? PyGpuArray_DEV_DATA(running_mean) : NULL,
theano/gpuarray/c_code/dnn_batchnorm.c:      running_averages ? PyGpuArray_DEV_DATA(running_var): NULL,
theano/gpuarray/c_code/dnn_batchnorm.c:      PyGpuArray_DEV_DATA(*x_mean),
theano/gpuarray/c_code/dnn_batchnorm.c:      PyGpuArray_DEV_DATA(*x_invstd)
theano/gpuarray/c_code/dnn_sptf_gt.c:            "GpuDnnTransformerGradT: could not allocate spatial transformer descriptor (sptf): %s",
theano/gpuarray/c_code/dnn_sptf_gt.c:APPLY_SPECIFIC(dnn_sptf_gt)(PyGpuArrayObject * dgrid,
theano/gpuarray/c_code/dnn_sptf_gt.c:                            PyGpuArrayObject ** dtheta,
theano/gpuarray/c_code/dnn_sptf_gt.c:    PyGpuContextObject * gpu_ctx = dgrid->context;
theano/gpuarray/c_code/dnn_sptf_gt.c:            "GpuDnnTransformerGradT: unsupported data type for dgrid in spatial transformer." );
theano/gpuarray/c_code/dnn_sptf_gt.c:    num_images = (int) PyGpuArray_DIM( dgrid, 0 );
theano/gpuarray/c_code/dnn_sptf_gt.c:    height = (int) PyGpuArray_DIM( dgrid, 1 );
theano/gpuarray/c_code/dnn_sptf_gt.c:    width = (int) PyGpuArray_DIM( dgrid, 2 );
theano/gpuarray/c_code/dnn_sptf_gt.c:                             GA_C_ORDER, gpu_ctx ) != 0 )
theano/gpuarray/c_code/dnn_sptf_gt.c:            "GpuDnnTransformerGrid: could not initialize descriptor (sptf): %s",
theano/gpuarray/c_code/dnn_sptf_gt.c:    cuda_enter( gpu_ctx->ctx );
theano/gpuarray/c_code/dnn_sptf_gt.c:    cuda_wait( dgrid->ga.data, GPUARRAY_CUDA_WAIT_READ );
theano/gpuarray/c_code/dnn_sptf_gt.c:    cuda_wait( (*dtheta)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
theano/gpuarray/c_code/dnn_sptf_gt.c:        PyGpuArray_DEV_DATA( dgrid ), PyGpuArray_DEV_DATA( *dtheta ) );
theano/gpuarray/c_code/dnn_sptf_gt.c:    cuda_record( dgrid->ga.data, GPUARRAY_CUDA_WAIT_READ );
theano/gpuarray/c_code/dnn_sptf_gt.c:    cuda_record( (*dtheta)->ga.data, GPUARRAY_CUDA_WAIT_WRITE );
theano/gpuarray/c_code/dnn_sptf_gt.c:    cuda_exit( gpu_ctx->ctx );
theano/gpuarray/c_code/dnn_sptf_gt.c:            "GpuDnnTransformerGradT: could not compute gradients of the affine transformation: %s",
theano/gpuarray/c_code/corr3d_gemm.c:// GPU kernel for the case of dilation
theano/gpuarray/c_code/corr3d_gemm.c:// GPU kernel for the case of dilation
theano/gpuarray/c_code/corr3d_gemm.c:          GpuArray *A, size_t offA, size_t lda,
theano/gpuarray/c_code/corr3d_gemm.c:          GpuArray *B, size_t offB, size_t ldb,
theano/gpuarray/c_code/corr3d_gemm.c:          double beta, GpuArray *C, size_t offC, size_t ldc) {
theano/gpuarray/c_code/corr3d_gemm.c:    return gpublas_sgemm(o, tA, tB,
theano/gpuarray/c_code/corr3d_gemm.c:    return gpublas_dgemm(o, tA, tB,
theano/gpuarray/c_code/corr3d_gemm.c:    return gpublas_hgemm(o, tA, tB,
theano/gpuarray/c_code/corr3d_gemm.c:    GpuArray *data_im, const size_t data_im_offset, const size_t channels,
theano/gpuarray/c_code/corr3d_gemm.c:    GpuArray *data_col) {
theano/gpuarray/c_code/corr3d_gemm.c:                     "gpuarray error: dilated_im3d2col_kernel: %s.",
theano/gpuarray/c_code/corr3d_gemm.c:                     GpuKernel_error(&k_dilated_im3d2col_kernel, err));
theano/gpuarray/c_code/corr3d_gemm.c:                     "gpuarray error: im3d2col_kernel: %s.",
theano/gpuarray/c_code/corr3d_gemm.c:                     GpuKernel_error(&k_im3d2col_kernel, err));
theano/gpuarray/c_code/corr3d_gemm.c:int col2im3d(GpuArray *data_col, const size_t channels,
theano/gpuarray/c_code/corr3d_gemm.c:    GpuArray *data_im, const size_t data_im_offset) {
theano/gpuarray/c_code/corr3d_gemm.c:                     "gpuarray error: dilated_col2im3d_kernel: %s.",
theano/gpuarray/c_code/corr3d_gemm.c:                     GpuKernel_error(&k_dilated_col2im3d_kernel, err));
theano/gpuarray/c_code/corr3d_gemm.c:                     "gpuarray error: col2im3d_kernel: %s.",
theano/gpuarray/c_code/corr3d_gemm.c:                     GpuKernel_error(&k_col2im3d_kernel, err));
theano/gpuarray/c_code/corr3d_gemm.c:PyGpuArrayObject* corr3dMM(PyGpuArrayObject *const bottom,
theano/gpuarray/c_code/corr3d_gemm.c:                           PyGpuArrayObject *const weight,
theano/gpuarray/c_code/corr3d_gemm.c:                           PyGpuArrayObject *const top,
theano/gpuarray/c_code/corr3d_gemm.c:    if (PyGpuArray_NDIM(bottom) != 5)
theano/gpuarray/c_code/corr3d_gemm.c:        PyErr_SetString(PyExc_ValueError, "GpuCorr3dMM requires bottom of 5D");
theano/gpuarray/c_code/corr3d_gemm.c:    if (!GpuArray_IS_C_CONTIGUOUS(&bottom->ga))
theano/gpuarray/c_code/corr3d_gemm.c:                "GpuCorr3dMM requires bottom to be C-contiguous, "
theano/gpuarray/c_code/corr3d_gemm.c:                PyGpuArray_STRIDES(bottom)[0],
theano/gpuarray/c_code/corr3d_gemm.c:                PyGpuArray_STRIDES(bottom)[1],
theano/gpuarray/c_code/corr3d_gemm.c:                PyGpuArray_STRIDES(bottom)[2],
theano/gpuarray/c_code/corr3d_gemm.c:                PyGpuArray_STRIDES(bottom)[3],
theano/gpuarray/c_code/corr3d_gemm.c:                PyGpuArray_STRIDES(bottom)[4]);
theano/gpuarray/c_code/corr3d_gemm.c:    if (PyGpuArray_NDIM(weight) != 5)
theano/gpuarray/c_code/corr3d_gemm.c:        PyErr_SetString(PyExc_ValueError, "GpuCorr3dMM requires weight of 5D");
theano/gpuarray/c_code/corr3d_gemm.c:    if (!GpuArray_IS_C_CONTIGUOUS(&weight->ga))
theano/gpuarray/c_code/corr3d_gemm.c:                "GpuCorr3dMM requires weight to be C-contiguous, "
theano/gpuarray/c_code/corr3d_gemm.c:                PyGpuArray_STRIDES(weight)[0],
theano/gpuarray/c_code/corr3d_gemm.c:                PyGpuArray_STRIDES(weight)[1],
theano/gpuarray/c_code/corr3d_gemm.c:                PyGpuArray_STRIDES(weight)[2],
theano/gpuarray/c_code/corr3d_gemm.c:                PyGpuArray_STRIDES(weight)[3],
theano/gpuarray/c_code/corr3d_gemm.c:                PyGpuArray_STRIDES(weight)[4]);
theano/gpuarray/c_code/corr3d_gemm.c:    if (PyGpuArray_NDIM(top) != 5)
theano/gpuarray/c_code/corr3d_gemm.c:        PyErr_SetString(PyExc_ValueError, "GpuCorr3dMM requires top of 5D");
theano/gpuarray/c_code/corr3d_gemm.c:    if (!GpuArray_IS_C_CONTIGUOUS(&top->ga))
theano/gpuarray/c_code/corr3d_gemm.c:                "GpuCorr3dMM requires top to be C-contiguous, "
theano/gpuarray/c_code/corr3d_gemm.c:                PyGpuArray_STRIDES(top)[0],
theano/gpuarray/c_code/corr3d_gemm.c:                PyGpuArray_STRIDES(top)[1],
theano/gpuarray/c_code/corr3d_gemm.c:                PyGpuArray_STRIDES(top)[2],
theano/gpuarray/c_code/corr3d_gemm.c:                PyGpuArray_STRIDES(top)[3],
theano/gpuarray/c_code/corr3d_gemm.c:                PyGpuArray_STRIDES(top)[4]);
theano/gpuarray/c_code/corr3d_gemm.c:    const size_t batchSize = PyGpuArray_DIMS(bottom)[0];
theano/gpuarray/c_code/corr3d_gemm.c:    const size_t nChannels = PyGpuArray_DIMS(bottom)[1];
theano/gpuarray/c_code/corr3d_gemm.c:    const size_t bottomHeight = PyGpuArray_DIMS(bottom)[2];
theano/gpuarray/c_code/corr3d_gemm.c:    const size_t bottomWidth = PyGpuArray_DIMS(bottom)[3];
theano/gpuarray/c_code/corr3d_gemm.c:    const size_t bottomDepth = PyGpuArray_DIMS(bottom)[4];
theano/gpuarray/c_code/corr3d_gemm.c:    const size_t nFilters = PyGpuArray_DIMS(weight)[0];
theano/gpuarray/c_code/corr3d_gemm.c:    const size_t kH = PyGpuArray_DIMS(weight)[2];
theano/gpuarray/c_code/corr3d_gemm.c:    const size_t kW = PyGpuArray_DIMS(weight)[3];
theano/gpuarray/c_code/corr3d_gemm.c:    const size_t kD = PyGpuArray_DIMS(weight)[4];
theano/gpuarray/c_code/corr3d_gemm.c:    if (nChannels != PyGpuArray_DIMS(weight)[1] * numgroups) {
theano/gpuarray/c_code/corr3d_gemm.c:                "GpuCorr3dMM images and kernel must have the same stack size\n");
theano/gpuarray/c_code/corr3d_gemm.c:    if (batchSize != PyGpuArray_DIMS(top)[0] ||
theano/gpuarray/c_code/corr3d_gemm.c:            nFilters != PyGpuArray_DIMS(top)[1] ||
theano/gpuarray/c_code/corr3d_gemm.c:            topHeight != PyGpuArray_DIMS(top)[2] ||
theano/gpuarray/c_code/corr3d_gemm.c:            topWidth != PyGpuArray_DIMS(top)[3] ||
theano/gpuarray/c_code/corr3d_gemm.c:            topDepth != PyGpuArray_DIMS(top)[4]) {
theano/gpuarray/c_code/corr3d_gemm.c:                "GpuCorr3dMM shape inconsistency:\n"
theano/gpuarray/c_code/corr3d_gemm.c:                PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
theano/gpuarray/c_code/corr3d_gemm.c:                PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3], PyGpuArray_DIMS(top)[4],
theano/gpuarray/c_code/corr3d_gemm.c:    int err = gpublas_setup(bottom->context->ctx);
theano/gpuarray/c_code/corr3d_gemm.c:    PyGpuArrayObject* col = (PyGpuArrayObject*)pygpu_empty(2, col_dim,
theano/gpuarray/c_code/corr3d_gemm.c:                "GpuCorr3dMM failed to allocate working memory of %ld x %ld\n",
theano/gpuarray/c_code/corr3d_gemm.c:    const size_t batch_bottom_stride = PyGpuArray_STRIDES(bottom)[0] / gpuarray_get_elsize(bottom->ga.typecode);
theano/gpuarray/c_code/corr3d_gemm.c:    const size_t batch_top_stride = PyGpuArray_STRIDES(top)[0] / gpuarray_get_elsize(top->ga.typecode);
theano/gpuarray/c_code/corr3d_gemm.c:    const size_t group_bottom_stride = (PyGpuArray_STRIDES(bottom)[1] * nChannels / numgroups) / gpuarray_get_elsize(bottom->ga.typecode);
theano/gpuarray/c_code/corr3d_gemm.c:    const size_t group_top_stride = (PyGpuArray_STRIDES(top)[1] * nFilters / numgroups) / gpuarray_get_elsize(top->ga.typecode);
theano/gpuarray/c_code/corr3d_gemm.c:    const size_t group_weight_stride = (PyGpuArray_STRIDES(weight)[0] * nFilters / numgroups) / gpuarray_get_elsize(weight->ga.typecode);
theano/gpuarray/c_code/corr3d_gemm.c:    PyGpuArrayObject *output;
theano/gpuarray/c_code/corr3d_gemm.c:            err = GpuArray_memset(&output->ga, 0);
theano/gpuarray/c_code/corr3d_gemm.c:                             "GpuCorr3dMM could not fill the output with zeros: %d", err);
theano/gpuarray/c_code/corr3d_gemm.c:                             "GpuCorr3dMM forward encountered an error running gemm.");
theano/gpuarray/c_code/corr3d_gemm.c:            err = GpuArray_memset(&output->ga, 0);
theano/gpuarray/c_code/corr3d_gemm.c:                             "GpuCorr3dMM grad wrt. weights could not fill the output with zeros: %d", err);
theano/gpuarray/c_code/corr3d_gemm.c:                             "GpuCorr3dMM grad weights encountered an error running gemm.");
theano/gpuarray/c_code/corr3d_gemm.c:            err = GpuArray_memset(&weight->ga, 0);
theano/gpuarray/c_code/corr3d_gemm.c:                             "GpuCorr3dMM grad weights could not fill the output with zeros: %d", err);
theano/gpuarray/c_code/corr3d_gemm.c:            err = GpuArray_memset(&output->ga, 0);
theano/gpuarray/c_code/corr3d_gemm.c:                             "GpuCorr3dMM grad wrt. inputs could not fill the output with zeros: %d", err);
theano/gpuarray/c_code/corr3d_gemm.c:                         "GpuCorr3dMM grad inputs encountered an error running gemm.");
theano/gpuarray/c_code/corr3d_gemm.c:    // (re)allocation and refcounting is done in BaseGpuCorr3dMM.c_code_helper();
theano/gpuarray/c_code/dnn_softmax_grad.c:int APPLY_SPECIFIC(softmax_grad)(PyGpuArrayObject *dy,
theano/gpuarray/c_code/dnn_softmax_grad.c:                                 PyGpuArrayObject *sm,
theano/gpuarray/c_code/dnn_softmax_grad.c:                                 PyGpuArrayObject **dx,
theano/gpuarray/c_code/dnn_softmax_grad.c:  PyGpuContextObject *c = dy->context;
theano/gpuarray/c_code/dnn_softmax_grad.c:  if (theano_prep_output(dx, PyGpuArray_NDIM(dy),
theano/gpuarray/c_code/dnn_softmax_grad.c:                         PyGpuArray_DIMS(dy), dy->ga.typecode,
theano/gpuarray/c_code/dnn_softmax_grad.c:  if (PyGpuArray_SIZE(*dx) == 0)
theano/gpuarray/c_code/dnn_softmax_grad.c:    cuda_enter(c->ctx);
theano/gpuarray/c_code/dnn_softmax_grad.c:    cuda_wait(sm->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_softmax_grad.c:    cuda_wait(dy->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_softmax_grad.c:    cuda_wait((*dx)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_softmax_grad.c:      PyGpuArray_DEV_DATA(sm),
theano/gpuarray/c_code/dnn_softmax_grad.c:      PyGpuArray_DEV_DATA(dy),
theano/gpuarray/c_code/dnn_softmax_grad.c:      PyGpuArray_DEV_DATA(*dx)
theano/gpuarray/c_code/dnn_softmax_grad.c:    cuda_record(sm->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_softmax_grad.c:    cuda_record(dy->ga.data, GPUARRAY_CUDA_WAIT_READ);
theano/gpuarray/c_code/dnn_softmax_grad.c:    cuda_record((*dx)->ga.data, GPUARRAY_CUDA_WAIT_WRITE);
theano/gpuarray/c_code/dnn_softmax_grad.c:    cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_dropout_fwd.c:int dnn_dropout_fwd(PyGpuArrayObject *x,
theano/gpuarray/c_code/dnn_dropout_fwd.c:                    PyGpuArrayObject *state,
theano/gpuarray/c_code/dnn_dropout_fwd.c:                    PyGpuArrayObject **y,
theano/gpuarray/c_code/dnn_dropout_fwd.c:                    PyGpuArrayObject **ostate,
theano/gpuarray/c_code/dnn_dropout_fwd.c:                    gpudata **reserve,
theano/gpuarray/c_code/dnn_dropout_fwd.c:  PyGpuArrayContext *c = x->context;
theano/gpuarray/c_code/dnn_dropout_fwd.c:  gpudata *res;
theano/gpuarray/c_code/dnn_dropout_fwd.c:  res = gpudata_alloc(c->ctx, res_zs, NULL, 0, NULL);
theano/gpuarray/c_code/dnn_dropout_fwd.c:  cuda_enter(c->ctx);
theano/gpuarray/c_code/dnn_dropout_fwd.c:  err = cudnnDropoutForward(_handle, desc, xdesc, PyGpuArray_DEV_DATA(x),
theano/gpuarray/c_code/dnn_dropout_fwd.c:                            ydesc, PyGpuArray_DEV_DATA(y), *(void **)res,
theano/gpuarray/c_code/dnn_dropout_fwd.c:    cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_dropout_fwd.c:  cuda_exit(c->ctx);
theano/gpuarray/c_code/dnn_conv_base.c:c_get_largest_free_block_size(PyGpuContextObject *c)
theano/gpuarray/c_code/dnn_conv_base.c:  int err2 = gpucontext_property(c->ctx, GA_CTX_PROP_LARGEST_MEMBLOCK, &maxfree);
theano/gpuarray/c_code/dnn_conv_base.c:                 "memory information on the GPU");
theano/gpuarray/c_code/dnn_conv_base.c:                                        PyGpuArrayObject* output,
theano/gpuarray/c_code/dnn_conv_base.c:      if ((PyGpuArray_DIMS(output)[0] != expected_output_dims[0]) ||
theano/gpuarray/c_code/dnn_conv_base.c:          (PyGpuArray_DIMS(output)[1] / groups != expected_output_dims[1]) ||
theano/gpuarray/c_code/dnn_conv_base.c:          (PyGpuArray_DIMS(output)[2] != expected_output_dims[2]) ||
theano/gpuarray/c_code/dnn_conv_base.c:          (PyGpuArray_DIMS(output)[3] != expected_output_dims[3])) {
theano/gpuarray/c_code/dnn_conv_base.c:                     PyGpuArray_DIMS(output)[0], PyGpuArray_DIMS(output)[1],
theano/gpuarray/c_code/dnn_conv_base.c:                     PyGpuArray_DIMS(output)[2], PyGpuArray_DIMS(output)[3]);
theano/gpuarray/c_code/dnn_conv_base.c:      if ((PyGpuArray_DIMS(output)[0] != expected_output_dims[0]) ||
theano/gpuarray/c_code/dnn_conv_base.c:          (PyGpuArray_DIMS(output)[1] / groups != expected_output_dims[1]) ||
theano/gpuarray/c_code/dnn_conv_base.c:          (PyGpuArray_DIMS(output)[2] != expected_output_dims[2]) ||
theano/gpuarray/c_code/dnn_conv_base.c:          (PyGpuArray_DIMS(output)[3] != expected_output_dims[3]) ||
theano/gpuarray/c_code/dnn_conv_base.c:          (PyGpuArray_DIMS(output)[4] != expected_output_dims[4])) {
theano/gpuarray/c_code/dnn_conv_base.c:                     PyGpuArray_DIMS(output)[0], PyGpuArray_DIMS(output)[1],
theano/gpuarray/c_code/dnn_conv_base.c:                     PyGpuArray_DIMS(output)[2], PyGpuArray_DIMS(output)[3],
theano/gpuarray/c_code/dnn_conv_base.c:                     PyGpuArray_DIMS(output)[4]);
theano/gpuarray/c_code/dnn_conv_base.c:static std::string dnn_conv_shape(cudnnTensorDescriptor_t inputDesc, PyGpuArrayObject* input,
theano/gpuarray/c_code/dnn_conv_base.c:				  cudnnFilterDescriptor_t filterDesc, PyGpuArrayObject* filter,
theano/gpuarray/c_code/dnn_conv_base.c:				  PyGpuArrayObject* output, int groups)
theano/gpuarray/c_code/dnn_conv_base.c:    if (dnn_check_convolution_output(convDesc, inputDesc, filterDesc, PyGpuArray_NDIM(filter), output, groups) != 0)
theano/gpuarray/c_code/dnn_conv_base.c:    if (!all_aligned(dType, PyGpuArray_DEV_DATA(input), PyGpuArray_DEV_DATA(output), PyGpuArray_DEV_DATA(filter)))
theano/gpuarray/cudnn_defs.py:Declarations of cuDNN types and constants used in Theano gpuarray DNN module.
theano/gpuarray/cudnn_defs.py:    (typically the version returned by :func:`theano.gpuarray.dnn.version`).
theano/gpuarray/ctc.py:from .basic_ops import (gpu_contiguous, as_gpuarray_variable, infer_context_name, gpuarray_helper_inc_dir)
theano/gpuarray/ctc.py:from .type import (GpuArrayType, gpu_context_type)
theano/gpuarray/ctc.py:from .elemwise import GpuDimShuffle
theano/gpuarray/ctc.py:from . import pygpu
theano/gpuarray/ctc.py:class GpuConnectionistTemporalClassification(gof.COp):
theano/gpuarray/ctc.py:    GPU wrapper for Baidu CTC loss function.
theano/gpuarray/ctc.py:    func_name = "APPLY_SPECIFIC(ctc_cost_gpu)"
theano/gpuarray/ctc.py:    params_type = gpu_context_type
theano/gpuarray/ctc.py:                               'GpuConnectionistTemporalClassification Op '
theano/gpuarray/ctc.py:        return ["warpctc", "gpuarray"]
theano/gpuarray/ctc.py:        dirs = [gpuarray_helper_inc_dir(), pygpu.get_include(),
theano/gpuarray/ctc.py:                config.cuda.include_path]
theano/gpuarray/ctc.py:        return ['ctc.h', 'numpy_compat.h', 'gpuarray/ext_cuda.h',
theano/gpuarray/ctc.py:                'gpuarray_helper.h', 'gpuarray/types.h', 'gpuarray_api.h',
theano/gpuarray/ctc.py:                'gpuarray/array.h', 'gpuarray/util.h', 'gpuarray/extension.h']
theano/gpuarray/ctc.py:        t_activations = as_gpuarray_variable(activations,
theano/gpuarray/ctc.py:        t_activations = gpu_contiguous(t_activations)
theano/gpuarray/ctc.py:        costs = GpuArrayType(dtype='float32',
theano/gpuarray/ctc.py:            gradients = GpuArrayType(dtype='float32',
theano/gpuarray/ctc.py:        grad_shuffle = GpuDimShuffle(input_broadcastable=(False, False, False,),
theano/gpuarray/ctc.py:        grad_shuffle_reverse = GpuDimShuffle(input_broadcastable=(False, False, False,),
theano/gpuarray/ctc.py:def gpu_ctc(activations, labels, input_lengths):
theano/gpuarray/ctc.py:    Compute CTC loss function on the GPU.
theano/gpuarray/ctc.py:    return GpuConnectionistTemporalClassification()(activations, labels, input_lengths)
theano/gpuarray/ctc.py:@local_optimizer([GpuConnectionistTemporalClassification])
theano/gpuarray/ctc.py:def local_gpu_ctc_no_grad(node):
theano/gpuarray/ctc.py:    if isinstance(node.op, GpuConnectionistTemporalClassification):
theano/gpuarray/ctc.py:                return [GpuConnectionistTemporalClassification(compute_grad=False)(*node.inputs), None]
theano/gpuarray/basic_ops.py:    import pygpu
theano/gpuarray/basic_ops.py:    from pygpu import gpuarray
theano/gpuarray/basic_ops.py:from .type import (GpuArrayType, GpuArrayConstant, gpu_context_type,
theano/gpuarray/basic_ops.py:def as_gpuarray_variable(x, context_name):
theano/gpuarray/basic_ops.py:    This will attempt to convert `x` into a variable on the GPU.
theano/gpuarray/basic_ops.py:            # If we are already a GpuArrayVariable in the right context
theano/gpuarray/basic_ops.py:            if (isinstance(x.type, GpuArrayType) and
theano/gpuarray/basic_ops.py:                if isinstance(x.owner.op, HostFromGpu):
theano/gpuarray/basic_ops.py:                if isinstance(x.owner.op, GpuFromHost):
theano/gpuarray/basic_ops.py:                if isinstance(x.owner.op, GpuToGpu):
theano/gpuarray/basic_ops.py:            return copy_stack_trace(x, GpuFromHost(context_name)(x))
theano/gpuarray/basic_ops.py:    # Try _as_GpuArrayVariable if possible
theano/gpuarray/basic_ops.py:    if hasattr(x, '_as_GpuArrayVariable'):
theano/gpuarray/basic_ops.py:        return copy_stack_trace(x, x._as_GpuArrayVariable(context_name))
theano/gpuarray/basic_ops.py:    if isinstance(x, gpuarray.GpuArray):
theano/gpuarray/basic_ops.py:    x = gpuarray.asarray(x, context=ctx)
theano/gpuarray/basic_ops.py:    return GpuArrayConstant(GpuArrayType(dtype=x.dtype,
theano/gpuarray/basic_ops.py:        if isinstance(v.type, GpuArrayType):
theano/gpuarray/basic_ops.py:            if isinstance(v.owner.op, HostFromGpu):
theano/gpuarray/basic_ops.py:def gpuarray_helper_inc_dir():
theano/gpuarray/basic_ops.py:    This class groups together all the attributes of a gpu kernel.
theano/gpuarray/basic_ops.py:    arguments should use the GpuArray class as the data type and
theano/gpuarray/basic_ops.py:    ga_ssize, use gpuarray.SIZE and gpuarray.SSIZE.
theano/gpuarray/basic_ops.py:            if t == gpuarray.GpuArray:
theano/gpuarray/basic_ops.py:                return str(gpuarray.dtype_to_typecode(t))
theano/gpuarray/basic_ops.py:    if dtype is gpuarray.GpuArray:
theano/gpuarray/basic_ops.py:        return "gpudata *"
theano/gpuarray/basic_ops.py:    elif dtype == gpuarray.SIZE:
theano/gpuarray/basic_ops.py:    elif dtype == gpuarray.SSIZE:
theano/gpuarray/basic_ops.py:class GpuKernelBase(object):
theano/gpuarray/basic_ops.py:    params_type = gpu_context_type
theano/gpuarray/basic_ops.py:        assert (self.params_type is gpu_context_type and
theano/gpuarray/basic_ops.py:                isinstance(node.inputs[0].type, GpuArrayType))
theano/gpuarray/basic_ops.py:    def get_gpu_context(self, node):
theano/gpuarray/basic_ops.py:        # Private method used to retrieve GPU context, instead of
theano/gpuarray/basic_ops.py:        if isinstance(self.params_type, ParamsType) and self.params_type.has_type(gpu_context_type):
theano/gpuarray/basic_ops.py:            # Get field name of gpu_context_type into ParamsType object.
theano/gpuarray/basic_ops.py:            gpu_context_field = self.params_type.get_field(gpu_context_type)
theano/gpuarray/basic_ops.py:            # Get GPU context from Params object.
theano/gpuarray/basic_ops.py:            return getattr(wrap, gpu_context_field)
theano/gpuarray/basic_ops.py:        assert self.params_type is gpu_context_type
theano/gpuarray/basic_ops.py:    def get_gpu_context_c_name(self, params_c_name):
theano/gpuarray/basic_ops.py:        # Private method used to retrieve C name of GPU context variable,
theano/gpuarray/basic_ops.py:        # instead of directly using sub['params'], as params may not be a GPU context
theano/gpuarray/basic_ops.py:        if isinstance(self.params_type, ParamsType) and self.params_type.has_type(gpu_context_type):
theano/gpuarray/basic_ops.py:            return "(%s->%s)" % (params_c_name, self.params_type.get_field(gpu_context_type))
theano/gpuarray/basic_ops.py:        assert self.params_type is gpu_context_type
theano/gpuarray/basic_ops.py:    def gpu_kernels(self, node, name):
theano/gpuarray/basic_ops.py:        raise MethodNotDefined('gpu_kernels')
theano/gpuarray/basic_ops.py:            o = super(GpuKernelBase, self).c_headers()
theano/gpuarray/basic_ops.py:        return o + ['gpuarray/types.h', 'numpy/npy_common.h']
theano/gpuarray/basic_ops.py:            o = super(GpuKernelBase, self).c_header_dirs()
theano/gpuarray/basic_ops.py:        # We rely on the input types for the directory to gpuarray includes
theano/gpuarray/basic_ops.py:        return """GpuKernel %(kname)s;""" % dict(kname=k.objvar)
theano/gpuarray/basic_ops.py:            if p is gpuarray.GpuArray:
theano/gpuarray/basic_ops.py:                setarg = "GpuKernel_setarg(&{0}, {1}, arg{1});"
theano/gpuarray/basic_ops.py:                setarg = "GpuKernel_setarg(&{0}, {1}, &arg{1});"
theano/gpuarray/basic_ops.py:  return GpuKernel_call(&{kname}, _nd, _gdim, _ldim, _shared, NULL);
theano/gpuarray/basic_ops.py:  _err = GpuKernel_sched(&{kname}, _n[0], &_gs, &_ls);
theano/gpuarray/basic_ops.py:  return GpuKernel_call(&{kname}, 1, &_gs, &_ls, _shared, NULL);
theano/gpuarray/basic_ops.py:        kernels = self.gpu_kernels(node, name)
theano/gpuarray/basic_ops.py:        kernels = self.gpu_kernels(node, name)
theano/gpuarray/basic_ops.py:  if ((err = GpuKernel_init(&%(ovar)s, %(ctx)s->ctx, 1,
theano/gpuarray/basic_ops.py:    PyErr_Format(PyExc_RuntimeError, "GpuKernel_init error %%d: %%s",
theano/gpuarray/basic_ops.py:                 err, gpucontext_error(%(ctx)s->ctx, err));
theano/gpuarray/basic_ops.py:        ctx = self.get_gpu_context_c_name(sub['params'])
theano/gpuarray/basic_ops.py:        kernels = self.gpu_kernels(node, name)
theano/gpuarray/basic_ops.py:        return "GpuKernel_clear(&%(ovar)s);" % dict(ovar=k.objvar)
theano/gpuarray/basic_ops.py:        kernels = self.gpu_kernels(node, name)
theano/gpuarray/basic_ops.py:        res = getattr(GpuKernelBase, name)(*args)
theano/gpuarray/basic_ops.py:        return gpuarray.GpuArray
theano/gpuarray/basic_ops.py:        return gpuarray.SIZE
theano/gpuarray/basic_ops.py:        return gpuarray.SSIZE
theano/gpuarray/basic_ops.py:class CGpuKernelBase(COp, GpuKernelBase):
theano/gpuarray/basic_ops.py:    Class to combine GpuKernelBase and COp.
theano/gpuarray/basic_ops.py:    get_params = GpuKernelBase.get_params
theano/gpuarray/basic_ops.py:        return GpuKernelBase.c_code_cache_version_apply(self, node)
theano/gpuarray/basic_ops.py:            if isinstance(v.type, GpuArrayType):
theano/gpuarray/basic_ops.py:                macro_value = pygpu.gpuarray.dtype_to_ctype(v.dtype)
theano/gpuarray/basic_ops.py:            if isinstance(v.type, GpuArrayType):
theano/gpuarray/basic_ops.py:                macro_value = pygpu.gpuarray.dtype_to_ctype(v.dtype)
theano/gpuarray/basic_ops.py:    def gpu_kernels(self, node, name):
theano/gpuarray/basic_ops.py:            return GpuKernelBase.gpu_kernels(self, node, name)
theano/gpuarray/basic_ops.py:class HostFromGpu(Op):
theano/gpuarray/basic_ops.py:        return 'HostFromGpu(gpuarray)'
theano/gpuarray/basic_ops.py:        if not isinstance(x.type, GpuArrayType):
theano/gpuarray/basic_ops.py:        GpuArray %(name)s_ga_s;
theano/gpuarray/basic_ops.py:        GpuArray *%(name)s_ga = NULL;
theano/gpuarray/basic_ops.py:        if (!GpuArray_ISONESEGMENT(&%(inp)s->ga)) {
theano/gpuarray/basic_ops.py:            if (GpuArray_copy(&%(name)s_ga_s, &%(inp)s->ga, GA_C_ORDER) != GA_NO_ERROR) {
theano/gpuarray/basic_ops.py:            if (%(name)s_ga == &%(name)s_ga_s) GpuArray_clear(%(name)s_ga);
theano/gpuarray/basic_ops.py:        %(name)serr = GpuArray_read(PyArray_DATA(%(out)s),
theano/gpuarray/basic_ops.py:        if (%(name)s_ga == &%(name)s_ga_s) GpuArray_clear(%(name)s_ga);
theano/gpuarray/basic_ops.py:        return [GpuFromHost(inputs[0].type.context_name)(gz)]
theano/gpuarray/basic_ops.py:host_from_gpu = HostFromGpu()
theano/gpuarray/basic_ops.py:class GpuFromHost(Op):
theano/gpuarray/basic_ops.py:    Transfer data to GPU.
theano/gpuarray/basic_ops.py:    params_type = gpu_context_type
theano/gpuarray/basic_ops.py:        return 'GpuFromHost<%s>' % (self.context_name,)
theano/gpuarray/basic_ops.py:            raise TypeError("complex not supported in the new gpuarray back-end.", x)
theano/gpuarray/basic_ops.py:        out_var = GpuArrayType(broadcastable=x.broadcastable,
theano/gpuarray/basic_ops.py:        z[0] = gpuarray.array(x, context=ctx)
theano/gpuarray/basic_ops.py:        return [as_gpuarray_variable(
theano/gpuarray/basic_ops.py:        return ["gpuarray_helper.h"]
theano/gpuarray/basic_ops.py:        return [gpuarray_helper_inc_dir()]
theano/gpuarray/basic_ops.py:        if (%(out)s == NULL || !GpuArray_IS_C_CONTIGUOUS(&%(out)s->ga) ||
theano/gpuarray/basic_ops.py:          %(out)s = pygpu_empty(PyArray_NDIM(%(name)s_tmp),
theano/gpuarray/basic_ops.py:        err = GpuArray_write(&%(out)s->ga, PyArray_DATA(%(name)s_tmp),
theano/gpuarray/basic_ops.py:          PyErr_Format(PyExc_RuntimeError, "Could not write data to gpu");
theano/gpuarray/basic_ops.py:class GpuToGpu(Op):
theano/gpuarray/basic_ops.py:    Transfer data between GPUs.
theano/gpuarray/basic_ops.py:    params_type = gpu_context_type
theano/gpuarray/basic_ops.py:        return 'GpuToGpu<%s>' % (self.context_name,)
theano/gpuarray/basic_ops.py:        if not isinstance(x.type, GpuArrayType):
theano/gpuarray/basic_ops.py:        return Apply(self, [x], [GpuArrayType(broadcastable=x.broadcastable,
theano/gpuarray/basic_ops.py:        return [GpuToGpu(inputs[0].type.context_name)(gz)]
theano/gpuarray/basic_ops.py:        %(out)s = pygpu_empty(%(inp)s->ga.nd,
theano/gpuarray/basic_ops.py:                              GpuArray_IS_C_CONTIGUOUS(&(%(inp)s->ga)) ? GA_C_ORDER:GA_F_ORDER,
theano/gpuarray/basic_ops.py:        if (pygpu_transfer(%(out)s, %(inp)s)) {
theano/gpuarray/basic_ops.py:class GpuAlloc(HideC, Alloc):
theano/gpuarray/basic_ops.py:    Allocate initialized memory on the GPU.
theano/gpuarray/basic_ops.py:    params_type = ParamsType(context=gpu_context_type, memset_0=bool_t)
theano/gpuarray/basic_ops.py:        value = as_gpuarray_variable(value, context_name=self.context_name)
theano/gpuarray/basic_ops.py:            TypeError("The GpuAlloc value to use has more dimensions "
theano/gpuarray/basic_ops.py:                out[0] = gpuarray.zeros(sh, dtype=v.dtype, context=params.context)
theano/gpuarray/basic_ops.py:                out[0] = gpuarray.empty(sh, dtype=v.dtype, context=params.context)
theano/gpuarray/basic_ops.py:            //pygpu_zeros can be faster then empty followed by memset.
theano/gpuarray/basic_ops.py:            %(zz)s = pygpu_zeros(%(ndim)s, %(name)s_shape,
theano/gpuarray/basic_ops.py:                %(zz)s = pygpu_empty(%(ndim)s, %(name)s_shape,
theano/gpuarray/basic_ops.py:            if (%(params)s->memset_0 && GpuArray_ISONESEGMENT(&%(zz)s->ga))
theano/gpuarray/basic_ops.py:                int err = GpuArray_memset(&%(zz)s->ga, 0);
theano/gpuarray/basic_ops.py:                                 "GpuAlloc: Error memsetting %%llu"
theano/gpuarray/basic_ops.py:                                 (unsigned long long)PyGpuArray_SIZE(%(zz)s));
theano/gpuarray/basic_ops.py:            else if (GpuArray_setarray(&%(zz)s->ga, &%(vv)s->ga) !=
theano/gpuarray/basic_ops.py:                             (subtensor.GpuIncSubtensor,
theano/gpuarray/basic_ops.py:                              subtensor.GpuAdvancedIncSubtensor1,
theano/gpuarray/basic_ops.py:                              subtensor.GpuAdvancedIncSubtensor1_dev20,
theano/gpuarray/basic_ops.py:                              subtensor.GpuAdvancedIncSubtensor,
theano/gpuarray/basic_ops.py:                              blas.GpuGemm, blas.GpuGemv,
theano/gpuarray/basic_ops.py:                              blas.GpuGer)
theano/gpuarray/basic_ops.py:            elif isinstance(client[0].op, HostFromGpu):
theano/gpuarray/basic_ops.py:class GpuAllocEmpty(HideC, AllocEmpty):
theano/gpuarray/basic_ops.py:    Allocate uninitialized memory on the GPU.
theano/gpuarray/basic_ops.py:    params_type = ParamsType(context=gpu_context_type,
theano/gpuarray/basic_ops.py:        return gpuarray.dtype_to_typecode(self.dtype)
theano/gpuarray/basic_ops.py:        output = GpuArrayType(dtype=self.dtype, broadcastable=bcast,
theano/gpuarray/basic_ops.py:            out[0] = pygpu.empty(sh, dtype=self.dtype, context=params.context)
theano/gpuarray/basic_ops.py:        return ['<gpuarray_helper.h>']
theano/gpuarray/basic_ops.py:        return [gpuarray_helper_inc_dir()]
theano/gpuarray/basic_ops.py:    return GpuAllocEmpty(var.type.dtype, var.type.context_name)(*var.shape)
theano/gpuarray/basic_ops.py:class GpuContiguous(Op):
theano/gpuarray/basic_ops.py:        dout = as_gpuarray_variable(dout, context_name=infer_context_name(x))
theano/gpuarray/basic_ops.py:        input = as_gpuarray_variable(input,
theano/gpuarray/basic_ops.py:        return [gpuarray_helper_inc_dir()]
theano/gpuarray/basic_ops.py:        return ['<gpuarray_helper.h>']
theano/gpuarray/basic_ops.py:            if (GpuArray_IS_C_CONTIGUOUS(&(%(input)s->ga))) {
theano/gpuarray/basic_ops.py:                || !theano_size_check(%(z)s, PyGpuArray_NDIM(%(input)s), PyGpuArray_DIMS(%(input)s),
theano/gpuarray/basic_ops.py:                || !GpuArray_IS_C_CONTIGUOUS(&(%(z)s->ga)))
theano/gpuarray/basic_ops.py:                %(z)s = pygpu_copy(%(input)s, GA_C_ORDER);
theano/gpuarray/basic_ops.py:            } else if(pygpu_move(%(z)s, %(input)s) == -1) {
theano/gpuarray/basic_ops.py:        out[0] = pygpu.ascontiguousarray(x)
theano/gpuarray/basic_ops.py:gpu_contiguous = GpuContiguous()
theano/gpuarray/basic_ops.py:class GpuReshape(HideC, tensor.Reshape):
theano/gpuarray/basic_ops.py:    Reshape for GPU variables.
theano/gpuarray/basic_ops.py:        x = as_gpuarray_variable(x, context_name=ctx_name)
theano/gpuarray/basic_ops.py:        otype = GpuArrayType(dtype=res.dtype,
theano/gpuarray/basic_ops.py:            raise ValueError('shape argument to GpuReshape.perform'
theano/gpuarray/basic_ops.py:            # We should make pygpu do the same.
theano/gpuarray/basic_ops.py:                         "GpuReshape: given shape is of incorrect "
theano/gpuarray/basic_ops.py:                                 "GpuReshape: only one -1 is accepted "
theano/gpuarray/basic_ops.py:                         "GpuReshape: trying to reshape an array of "
theano/gpuarray/basic_ops.py:                         "GpuReshape: -1 axis found at index %%d in "
theano/gpuarray/basic_ops.py:        %(output)s = pygpu_reshape(%(x)s, %(params)s->ndim, new_dims,
theano/gpuarray/basic_ops.py:class GpuJoin(HideC, Join):
theano/gpuarray/basic_ops.py:    Join for GPU.
theano/gpuarray/basic_ops.py:    params_type = gpu_context_type
theano/gpuarray/basic_ops.py:            return as_gpuarray_variable(v, context_name=ctx_name)
theano/gpuarray/basic_ops.py:                     [GpuArrayType(broadcastable=node.outputs[0].broadcastable,
theano/gpuarray/basic_ops.py:            out[0] = pygpu.concatenate(tensors, axis=axis, context=ctx).astype(
theano/gpuarray/basic_ops.py:        restype = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
theano/gpuarray/basic_ops.py:        const GpuArray **als = (const GpuArray **)PyMem_Malloc(sizeof(GpuArray *) *
theano/gpuarray/basic_ops.py:                tensors_lens_sum -= PyGpuArray_DIM(%(non_empty_tensor)s, axis);
theano/gpuarray/basic_ops.py:                %(out)s = pygpu_concatenate(als, %(n)s, axis,
theano/gpuarray/basic_ops.py:                                            %(restype)s, (PyObject *)&PyGpuArrayType,
theano/gpuarray/basic_ops.py:gpu_join = GpuJoin()
theano/gpuarray/basic_ops.py:class GpuSplit(HideC, Split):
theano/gpuarray/basic_ops.py:    Split for GPU.
theano/gpuarray/basic_ops.py:        super(GpuSplit, self).__init__(len_splits)
theano/gpuarray/basic_ops.py:        # The GPU version of Split returns splits as views of the input.
theano/gpuarray/basic_ops.py:        x = as_gpuarray_variable(x, infer_context_name(x))
theano/gpuarray/basic_ops.py:        outs = [GpuArrayType(dtype=o.dtype, broadcastable=o.broadcastable,
theano/gpuarray/basic_ops.py:        return ['<numpy_compat.h>', '<gpuarray_helper.h>']
theano/gpuarray/basic_ops.py:        return [pygpu.get_include(), gpuarray_helper_inc_dir()]
theano/gpuarray/basic_ops.py:        int ndim = PyGpuArray_NDIM(%(x)s);
theano/gpuarray/basic_ops.py:        GpuArray* split_views = NULL;
theano/gpuarray/basic_ops.py:        GpuArray** split_views_pointers = NULL;
theano/gpuarray/basic_ops.py:        PyGpuArrayObject** outputs[] = {%(outputs_pointers)s};
theano/gpuarray/basic_ops.py:                "GpuSplit: splits count (%%d) != expected count (%%d).", splits_count, %(expected_splits_count)s);
theano/gpuarray/basic_ops.py:            PyErr_Format(PyExc_IndexError, "GpuSplit: invalid axis %%d for a %%d-D array.", axis, ndim);
theano/gpuarray/basic_ops.py:        len_along_axis = PyGpuArray_DIM(%(x)s, axis);
theano/gpuarray/basic_ops.py:                    "GpuSplit: you try to take a negative number (%%ld) of elements.", current_split_length);
theano/gpuarray/basic_ops.py:            PyErr_Format(PyExc_ValueError, "GpuSplit: the splits sums to %%ld, expected %%ld.", sum_of_splits, len_along_axis);
theano/gpuarray/basic_ops.py:        split_views = (GpuArray*) malloc(splits_count * sizeof(GpuArray));
theano/gpuarray/basic_ops.py:        split_views_pointers = (GpuArray**) malloc(splits_count * sizeof(GpuArray*));
theano/gpuarray/basic_ops.py:        if (GpuArray_split(split_views_pointers, &%(x)s->ga, splits_count - 1, split_points, axis) != GA_NO_ERROR) {
theano/gpuarray/basic_ops.py:            PyErr_SetString(PyExc_RuntimeError, "GpuSplit: unable to compute split.");
theano/gpuarray/basic_ops.py:                GpuArray_clear(split_views_pointers[i]);
theano/gpuarray/basic_ops.py:            PyGpuArrayObject** output = outputs[i];
theano/gpuarray/basic_ops.py:            *output = pygpu_fromgpudata(
theano/gpuarray/basic_ops.py:                PyErr_SetString(PyExc_RuntimeError, "GpuSplit: unable to update an output from a split view.");
theano/gpuarray/basic_ops.py:                    GpuArray_clear(split_views_pointers[j]);
theano/gpuarray/basic_ops.py:           GpuArray_clear(split_views_pointers[i]);
theano/gpuarray/basic_ops.py:    if any([x.op.__class__.__name__.lower().startswith("gpu")
theano/gpuarray/basic_ops.py:        print('Some info useful for gpu:', file=file)
theano/gpuarray/basic_ops.py:        gpu = 0
theano/gpuarray/basic_ops.py:            if isinstance(node.op, (HostFromGpu, GpuFromHost)):
theano/gpuarray/basic_ops.py:            elif node.op.__class__.__name__.lower().startswith("gpu"):
theano/gpuarray/basic_ops.py:                gpu += t
theano/gpuarray/basic_ops.py:        print("    Spent %.3fs(%.2f%%) in cpu Op, %.3fs(%.2f%%) in gpu Op and %.3fs(%.2f%%) transfert Op" % (
theano/gpuarray/basic_ops.py:            cpu, cpu / local_time * 100, gpu, gpu / local_time * 100,
theano/gpuarray/basic_ops.py:        print("    (Useful to know if we forgot some cast when using floatX=float32 or gpu code)", file=file)
theano/gpuarray/basic_ops.py:class GpuEye(GpuKernelBase, Op):
theano/gpuarray/basic_ops.py:    Eye for GPU.
theano/gpuarray/basic_ops.py:        otype = GpuArrayType(dtype=self.dtype,
theano/gpuarray/basic_ops.py:    def gpu_kernels(self, node, name):
theano/gpuarray/basic_ops.py:}""" % dict(ctype=pygpu.gpuarray.dtype_to_ctype(self.dtype),
theano/gpuarray/basic_ops.py:                params=[gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SIZE,
theano/gpuarray/basic_ops.py:                        gpuarray.SIZE, gpuarray.SSIZE],
theano/gpuarray/basic_ops.py:        typecode = pygpu.gpuarray.dtype_to_typecode(self.dtype)
theano/gpuarray/basic_ops.py:        kname = self.gpu_kernels(node, name)[0].objvar
theano/gpuarray/basic_ops.py:        %(z)s = pygpu_zeros(2, dims,
theano/gpuarray/basic_ops.py:                             "gpuarray error: kEye: %%s. n%%lu, m=%%lu.",
theano/gpuarray/basic_ops.py:                             GpuKernel_error(&%(kname)s, err),
theano/gpuarray/basic_ops.py:class GpuTri(GpuKernelBase, Op):
theano/gpuarray/basic_ops.py:    Tri for GPU.
theano/gpuarray/basic_ops.py:        otype = GpuArrayType(dtype=self.dtype,
theano/gpuarray/basic_ops.py:    def gpu_kernels(self, node, name):
theano/gpuarray/basic_ops.py:}""" % dict(ctype=pygpu.gpuarray.dtype_to_ctype(self.dtype),
theano/gpuarray/basic_ops.py:                params=[gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SIZE,
theano/gpuarray/basic_ops.py:                        gpuarray.SIZE, gpuarray.SSIZE],
theano/gpuarray/basic_ops.py:        typecode = pygpu.gpuarray.dtype_to_typecode(self.dtype)
theano/gpuarray/basic_ops.py:        kname = self.gpu_kernels(node, name)[0].objvar
theano/gpuarray/basic_ops.py:        %(z)s = pygpu_zeros(2, dims,
theano/gpuarray/basic_ops.py:                         "gpuarray error: kTri: %%s. n%%lu, m=%%lu.",
theano/gpuarray/basic_ops.py:                         GpuKernel_error(&%(kname)s, err),
theano/gpuarray/rng_mrg.py:GPU implementation of MRG31k3p random number generator for Theano.
theano/gpuarray/rng_mrg.py:from .basic_ops import (GpuKernelBase, Kernel, infer_context_name,
theano/gpuarray/rng_mrg.py:                        GpuFromHost, host_from_gpu, as_gpuarray_variable)
theano/gpuarray/rng_mrg.py:from .type import GpuArrayType, gpu_context_type
theano/gpuarray/rng_mrg.py:class GPUA_mrg_uniform(GpuKernelBase, mrg_uniform_base):
theano/gpuarray/rng_mrg.py:    # GpuArray version
theano/gpuarray/rng_mrg.py:    params_type = mrg_uniform_base.params_type.extended(otypecode=int_t, context=gpu_context_type)
theano/gpuarray/rng_mrg.py:        rstate = as_gpuarray_variable(rstate, infer_context_name(rstate))
theano/gpuarray/rng_mrg.py:        op = cls(GpuArrayType(dtype, (False,) * ndim))
theano/gpuarray/rng_mrg.py:        return super(GPUA_mrg_uniform, self).c_headers() + ['numpy_compat.h']
theano/gpuarray/rng_mrg.py:    def gpu_kernels(self, node, name):
theano/gpuarray/rng_mrg.py:        from pygpu import gpuarray
theano/gpuarray/rng_mrg.py:                       params=[gpuarray.GpuArray, gpuarray.SIZE,
theano/gpuarray/rng_mrg.py:                               gpuarray.GpuArray, gpuarray.SIZE,
theano/gpuarray/rng_mrg.py:                || !pygpu_GpuArray_Check((PyObject*)%(o_sample)s)
theano/gpuarray/rng_mrg.py:                || (PyGpuArray_NDIM(%(o_sample)s) != %(params)s->ndim));
theano/gpuarray/rng_mrg.py:                    || PyGpuArray_DIMS(%(o_sample)s)[i] != odims[i]);
theano/gpuarray/rng_mrg.py:                "rng_mrg gpu implementation does not support more than (2**31 -1) samples");
theano/gpuarray/rng_mrg.py:            %(o_sample)s = pygpu_empty(%(params)s->ndim, odims, %(params)s->otypecode, GA_C_ORDER,
theano/gpuarray/rng_mrg.py:        if (!pygpu_GpuArray_Check((PyObject*)%(rstate)s))
theano/gpuarray/rng_mrg.py:            PyErr_Format(PyExc_ValueError, "rstate must be gpuarray");
theano/gpuarray/rng_mrg.py:            %(o_rstate)s = pygpu_copy(%(rstate)s, GA_ANY_ORDER);
theano/gpuarray/rng_mrg.py:        if (PyGpuArray_NDIM(%(o_rstate)s) != 2)
theano/gpuarray/rng_mrg.py:        if (PyGpuArray_DIMS(%(o_rstate)s)[1] != 6)
theano/gpuarray/rng_mrg.py:        if (!GpuArray_CHKFLAGS(&%(o_rstate)s->ga, GA_C_CONTIGUOUS)) {
theano/gpuarray/rng_mrg.py:        n_streams = PyGpuArray_DIMS(%(o_rstate)s)[0];
theano/gpuarray/rng_mrg.py:          int err = GpuKernel_sched(&%(kname)s, n_streams, &ls, &gs);
theano/gpuarray/rng_mrg.py:              PyErr_Format(PyExc_RuntimeError, "GpuKernel_sched: %%s\\n",
theano/gpuarray/rng_mrg.py:                           GpuKernel_error(&%(kname)s, err));
theano/gpuarray/rng_mrg.py:                           GpuKernel_error(&%(kname)s, err));
theano/gpuarray/rng_mrg.py:                   kname=self.gpu_kernels(node, nodename)[0].objvar,
theano/gpuarray/rng_mrg.py:def local_gpua_mrg_graph(op, context_name, inputs, outputs):
theano/gpuarray/rng_mrg.py:            isinstance(inputs[0].type, GpuArrayType) and
theano/gpuarray/rng_mrg.py:                            GpuFromHost))):
theano/gpuarray/rng_mrg.py:        outs = GPUA_mrg_uniform.new(inputs[0],
theano/gpuarray/rng_mrg.py:        return [outs[0], host_from_gpu(outs[1])]
theano/gpuarray/rng_mrg.py:def local_gpua_mrg(node):
theano/gpuarray/rng_mrg.py:    return local_gpua_mrg_graph(node.op, context_name, node.inputs, node.outputs)
theano/gpuarray/multinomial.py:    import pygpu
theano/gpuarray/multinomial.py:from .basic_ops import as_gpuarray_variable, infer_context_name, GpuKernelBase, Kernel, gpuarray_helper_inc_dir
theano/gpuarray/multinomial.py:from .type import GpuArrayType
theano/gpuarray/multinomial.py:from .elemwise import GpuDimShuffle
theano/gpuarray/multinomial.py:class GPUAMultinomialFromUniform(GpuKernelBase, Op):
theano/gpuarray/multinomial.py:        return ['<numpy_compat.h>', 'gpuarray_helper.h']
theano/gpuarray/multinomial.py:        return [gpuarray_helper_inc_dir()]
theano/gpuarray/multinomial.py:        pvals = as_gpuarray_variable(pvals, ctx_name)
theano/gpuarray/multinomial.py:        unis = as_gpuarray_variable(unis, ctx_name)
theano/gpuarray/multinomial.py:        out = GpuArrayType(broadcastable=br,
theano/gpuarray/multinomial.py:    def gpu_kernels(self, node, name):
theano/gpuarray/multinomial.py:        out_ctype = pygpu.gpuarray.dtype_to_ctype(node.outputs[0].dtype)
theano/gpuarray/multinomial.py:        pvals_ctype = pygpu.gpuarray.dtype_to_ctype(node.inputs[0].dtype)
theano/gpuarray/multinomial.py:        unis_ctype = pygpu.gpuarray.dtype_to_ctype(node.inputs[1].dtype)
theano/gpuarray/multinomial.py:        work_ctype = pygpu.gpuarray.dtype_to_ctype(work_dtype(node.inputs[0].dtype))
theano/gpuarray/multinomial.py:            params=[pygpu.gpuarray.SIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.GpuArray,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SSIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SSIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.GpuArray,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SSIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.GpuArray,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SSIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SSIZE],
theano/gpuarray/multinomial.py:        kname = self.gpu_kernels(node, name)[0].objvar
theano/gpuarray/multinomial.py:        out_typecode = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
theano/gpuarray/multinomial.py:        pvals_typecode = pygpu.gpuarray.dtype_to_typecode(node.inputs[0].dtype)
theano/gpuarray/multinomial.py:        unis_typecode = pygpu.gpuarray.dtype_to_typecode(node.inputs[1].dtype)
theano/gpuarray/multinomial.py:        PyGpuArrayObject * pvals = %(pvals)s;
theano/gpuarray/multinomial.py:        PyGpuArrayObject * unis = %(unis)s;
theano/gpuarray/multinomial.py:        PyGpuArrayObject * out = %(out)s;
theano/gpuarray/multinomial.py:    if (PyGpuArray_NDIM(pvals) != 2)
theano/gpuarray/multinomial.py:    if (PyGpuArray_NDIM(unis) != 1)
theano/gpuarray/multinomial.py:    if (PyGpuArray_DIMS(unis)[0] != PyGpuArray_DIMS(pvals)[0])
theano/gpuarray/multinomial.py:    dims[0] = PyGpuArray_DIMS(pvals)[1];
theano/gpuarray/multinomial.py:    dims[1] = PyGpuArray_DIMS(pvals)[0];
theano/gpuarray/multinomial.py:    GpuArray_memset(&(out->ga), 0);
theano/gpuarray/multinomial.py:        int nb_multi = PyGpuArray_DIMS(pvals)[0];
theano/gpuarray/multinomial.py:        int nb_outcomes = PyGpuArray_DIMS(pvals)[1];
theano/gpuarray/multinomial.py:          PyGpuArray_DIMS(out)[1], PyGpuArray_DIMS(out)[0], pvals->ga.data, pvals->ga.offset,
theano/gpuarray/multinomial.py:          PyGpuArray_STRIDES(pvals)[0]/gpuarray_get_elsize(%(pvals_typecode)s),
theano/gpuarray/multinomial.py:          PyGpuArray_STRIDES(pvals)[1]/gpuarray_get_elsize(%(pvals_typecode)s),
theano/gpuarray/multinomial.py:          PyGpuArray_STRIDES(unis)[0]/gpuarray_get_elsize(%(unis_typecode)s), out->ga.data,
theano/gpuarray/multinomial.py:          out->ga.offset, PyGpuArray_STRIDES(out)[0]/gpuarray_get_elsize(%(out_typecode)s),
theano/gpuarray/multinomial.py:          PyGpuArray_STRIDES(out)[1]/gpuarray_get_elsize(%(out_typecode)s));
theano/gpuarray/multinomial.py:                "gpuarray error: %%s: %%s.\\n",
theano/gpuarray/multinomial.py:                GpuKernel_error(&%(kname)s, err));
theano/gpuarray/multinomial.py:class GPUAChoiceFromUniform(GpuKernelBase, Op):
theano/gpuarray/multinomial.py:    The optimization that moves it to the gpu does it.
theano/gpuarray/multinomial.py:        return ['<numpy_compat.h>', 'gpuarray_helper.h']
theano/gpuarray/multinomial.py:        return [gpuarray_helper_inc_dir()]
theano/gpuarray/multinomial.py:        pvals = as_gpuarray_variable(pvals, ctx_name)
theano/gpuarray/multinomial.py:        unis = as_gpuarray_variable(unis, ctx_name)
theano/gpuarray/multinomial.py:        out = GpuArrayType(broadcastable=br,
theano/gpuarray/multinomial.py:    def gpu_kernels(self, node, name):
theano/gpuarray/multinomial.py:            params=[pygpu.gpuarray.SIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.GpuArray,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SSIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SSIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.GpuArray,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SSIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.GpuArray,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SSIZE,
theano/gpuarray/multinomial.py:                    pygpu.gpuarray.SSIZE
theano/gpuarray/multinomial.py:        kname = self.gpu_kernels(node, name)[0].objvar
theano/gpuarray/multinomial.py:    PyGpuArrayObject * pvals = %(pvals)s;
theano/gpuarray/multinomial.py:    PyGpuArrayObject * unis = %(unis)s;
theano/gpuarray/multinomial.py:    PyGpuArrayObject * out = %(out)s;
theano/gpuarray/multinomial.py:    PyGpuArrayObject * pvals_copy = NULL;
theano/gpuarray/multinomial.py:    if (PyGpuArray_NDIM(pvals) != 2)
theano/gpuarray/multinomial.py:    if (PyGpuArray_NDIM(unis) != 1)
theano/gpuarray/multinomial.py:    if ( n_samples > (PyGpuArray_DIMS(pvals)[1]) )
theano/gpuarray/multinomial.py:    if (PyGpuArray_DIMS(unis)[0] != PyGpuArray_DIMS(pvals)[0] * n_samples)
theano/gpuarray/multinomial.py:        pvals_copy = pygpu_copy(pvals, GA_C_ORDER);
theano/gpuarray/multinomial.py:    dims[1] = PyGpuArray_DIMS(pvals)[0];
theano/gpuarray/multinomial.py:        int nb_multi = PyGpuArray_DIMS(pvals)[0];
theano/gpuarray/multinomial.py:        int nb_outcomes = PyGpuArray_DIMS(pvals)[1];
theano/gpuarray/multinomial.py:        int err = k_multi_warp_multinomial_wor_call(1, &nb_blocks, &nb_threads, 0, PyGpuArray_DIMS(pvals)[0], PyGpuArray_DIMS(pvals)[1], n_samples, pvals_copy->ga.data, pvals_copy->ga.offset, PyGpuArray_STRIDES(pvals)[0]/sizeof(float), PyGpuArray_STRIDES(pvals)[1]/sizeof(float), unis->ga.data, unis->ga.offset, PyGpuArray_STRIDES(unis)[0]/sizeof(float), out->ga.data, out->ga.offset, PyGpuArray_STRIDES(out)[0]/8, PyGpuArray_STRIDES(out)[1]/8);
theano/gpuarray/multinomial.py:                "gpuarray error: %%s: %%s.\\n",
theano/gpuarray/multinomial.py:                GpuKernel_error(&%(kname)s, err));
theano/gpuarray/multinomial.py:def local_gpua_multinomial(op, context_name, inputs, outputs):
theano/gpuarray/multinomial.py:    gpu_op = GPUAMultinomialFromUniform(op.odtype)
theano/gpuarray/multinomial.py:    return GpuDimShuffle([False, False], [1, 0])(
theano/gpuarray/multinomial.py:        gpu_op(p, u))
theano/gpuarray/multinomial.py:def local_gpua_multinomial_wor(op, context_name, inputs, outputs):
theano/gpuarray/multinomial.py:        gpu_op = GPUAChoiceFromUniform(**op._props_dict())
theano/gpuarray/multinomial.py:        return GpuDimShuffle([False, False], [1, 0])(
theano/gpuarray/multinomial.py:            gpu_op(p, u, n))
theano/gpuarray/multinomial.py:class GPUAMultinomialWOReplacementFromUniform(GPUAChoiceFromUniform):
theano/gpuarray/multinomial.py:        warnings.warn("GPUAMultinomialWOReplacementFromUniform is deprecated, "
theano/gpuarray/multinomial.py:                      "use GPUAChoiceFromUniform instead.",
theano/gpuarray/multinomial.py:        super(GPUAMultinomialWOReplacementFromUniform, self).__init__(*args, **kwargs)
theano/gpuarray/nnet.py:    import pygpu
theano/gpuarray/nnet.py:    from pygpu import gpuarray
theano/gpuarray/nnet.py:from .basic_ops import (as_gpuarray_variable, GpuKernelBase, Kernel, gpuarray_helper_inc_dir,
theano/gpuarray/nnet.py:from .type import GpuArrayType
theano/gpuarray/nnet.py:class GpuCrossentropySoftmaxArgmax1HotWithBias(GpuKernelBase, Op):
theano/gpuarray/nnet.py:    Implement CrossentropySoftmaxArgmax1HotWithBias on the gpu.
theano/gpuarray/nnet.py:        x = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/nnet.py:        b = as_gpuarray_variable(b, ctx_name)
theano/gpuarray/nnet.py:        y_idx = as_gpuarray_variable(y_idx, ctx_name)
theano/gpuarray/nnet.py:        nll = GpuArrayType(x.type.dtype,
theano/gpuarray/nnet.py:        return ['<numpy_compat.h>', '<gpuarray/types.h>', 'gpuarray_helper.h']
theano/gpuarray/nnet.py:        return [gpuarray_helper_inc_dir()]
theano/gpuarray/nnet.py:    def gpu_kernels(self, node, nodename):
theano/gpuarray/nnet.py:        type_x = gpuarray.dtype_to_ctype(dtype_x)
theano/gpuarray/nnet.py:        type_b = gpuarray.dtype_to_ctype(dtype_b)
theano/gpuarray/nnet.py:        work_x = gpuarray.dtype_to_ctype(work_x)
theano/gpuarray/nnet.py:        type_y_idx = gpuarray.dtype_to_ctype(dtype_y_idx)
theano/gpuarray/nnet.py:        if node.inputs[0].type.context.kind != b'cuda':
theano/gpuarray/nnet.py:            gpuarray.SIZE, gpuarray.SIZE,
theano/gpuarray/nnet.py:            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SSIZE,
theano/gpuarray/nnet.py:            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE,
theano/gpuarray/nnet.py:            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE,
theano/gpuarray/nnet.py:            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE,
theano/gpuarray/nnet.py:            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SSIZE,
theano/gpuarray/nnet.py:            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE
theano/gpuarray/nnet.py:                             "gpuarray error: %(k_var)s: %%s.",
theano/gpuarray/nnet.py:                             GpuKernel_error(&%(k_var)s, err));
theano/gpuarray/nnet.py:        if (PyGpuArray_DIMS(%(x)s)[0] !=
theano/gpuarray/nnet.py:            PyGpuArray_DIMS(%(y_idx)s)[0])
theano/gpuarray/nnet.py:        if (PyGpuArray_DIMS(%(x)s)[1] != PyGpuArray_DIMS(%(b)s)[0])
theano/gpuarray/nnet.py:        if (theano_prep_output(&%(nll)s, 1, PyGpuArray_DIMS(%(y_idx)s), %(x)s->ga.typecode, GA_C_ORDER, %(ctx)s)) %(fail)s
theano/gpuarray/nnet.py:        if (theano_prep_output(&%(sm)s, 2, PyGpuArray_DIMS(%(x)s), %(x)s->ga.typecode, GA_C_ORDER, %(ctx)s)) %(fail)s
theano/gpuarray/nnet.py:        if (theano_prep_output(&%(am)s, 1, PyGpuArray_DIMS(%(y_idx)s), %(y_idx)s->ga.typecode, GA_C_ORDER, %(ctx)s)) %(fail)s
theano/gpuarray/nnet.py:            size_t n_blocks = std::min(PyGpuArray_DIM(%(x)s, 0), (size_t)4096);
theano/gpuarray/nnet.py:            size_t n_threads = std::min(PyGpuArray_DIM(%(x)s, 1), (size_t)256);
theano/gpuarray/nnet.py:                PyGpuArray_DIMS(%(x)s)[0],
theano/gpuarray/nnet.py:                PyGpuArray_DIMS(%(x)s)[1],
theano/gpuarray/nnet.py:                PyGpuArray_STRIDE(%(x)s, 0) / %(itemsize_x)s,
theano/gpuarray/nnet.py:                PyGpuArray_STRIDE(%(x)s, 1) / %(itemsize_x)s,
theano/gpuarray/nnet.py:                PyGpuArray_STRIDE(%(b)s, 0) / %(itemsize_b)s,
theano/gpuarray/nnet.py:                PyGpuArray_STRIDE(%(y_idx)s, 0) / %(itemsize_y_idx)s,
theano/gpuarray/nnet.py:                PyGpuArray_STRIDE(%(nll)s, 0) / %(itemsize_nll)s,
theano/gpuarray/nnet.py:                PyGpuArray_STRIDE(%(sm)s, 0) / %(itemsize_sm)s,
theano/gpuarray/nnet.py:                PyGpuArray_STRIDE(%(sm)s, 1) / %(itemsize_sm)s,
theano/gpuarray/nnet.py:                PyGpuArray_STRIDE(%(am)s, 0) / %(itemsize_am)s);
theano/gpuarray/nnet.py:gpu_crossentropy_softmax_argmax_1hot_with_bias = GpuCrossentropySoftmaxArgmax1HotWithBias()
theano/gpuarray/nnet.py:class GpuCrossentropySoftmax1HotWithBiasDx(GpuKernelBase, Op):
theano/gpuarray/nnet.py:    Implement CrossentropySoftmax1HotWithBiasDx on the gpu.
theano/gpuarray/nnet.py:        dnll = as_gpuarray_variable(dnll, ctx_name)
theano/gpuarray/nnet.py:        sm = as_gpuarray_variable(sm, ctx_name)
theano/gpuarray/nnet.py:        y_idx = as_gpuarray_variable(y_idx, ctx_name)
theano/gpuarray/nnet.py:        return ['<numpy_compat.h>', '<gpuarray/types.h>']
theano/gpuarray/nnet.py:        typecode_dx = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
theano/gpuarray/nnet.py:        type_intp = gpuarray.dtype_to_ctype(np.intp)
theano/gpuarray/nnet.py:                             "gpuarray error: %(k_var)s: %%s.",
theano/gpuarray/nnet.py:                             GpuKernel_error(&%(k_var)s, err));
theano/gpuarray/nnet.py:        const ssize_t %(dnll)s_dims0 = (PyGpuArray_NDIM(%(dnll)s) > 0 ?
theano/gpuarray/nnet.py:                                        PyGpuArray_DIMS(%(dnll)s)[0] :
theano/gpuarray/nnet.py:                                           PyGpuArray_STRIDES(%(dnll)s)[0] :
theano/gpuarray/nnet.py:        if ((PyGpuArray_NDIM(%(dnll)s) > 1)
theano/gpuarray/nnet.py:            || (PyGpuArray_NDIM(%(sm)s) != 2)
theano/gpuarray/nnet.py:            || (PyGpuArray_NDIM(%(y_idx)s) != 1))
theano/gpuarray/nnet.py:            PyGpuArray_DIMS(%(sm)s)[0] && %(dnll)s_dims0 > 1)
theano/gpuarray/nnet.py:                         PyGpuArray_DIMS(%(sm)s)[0]);
theano/gpuarray/nnet.py:            PyGpuArray_DIMS(%(y_idx)s)[0] && %(dnll)s_dims0 > 1)
theano/gpuarray/nnet.py:        if (PyGpuArray_DIMS(%(sm)s)[0] !=
theano/gpuarray/nnet.py:            PyGpuArray_DIMS(%(y_idx)s)[0])
theano/gpuarray/nnet.py:            || (PyGpuArray_DIMS(%(dx)s)[0] !=
theano/gpuarray/nnet.py:                PyGpuArray_DIMS(%(sm)s)[0])
theano/gpuarray/nnet.py:            || (PyGpuArray_DIMS(%(dx)s)[1] !=
theano/gpuarray/nnet.py:                PyGpuArray_DIMS(%(sm)s)[1]))
theano/gpuarray/nnet.py:            %(dx)s = pygpu_empty(2, PyGpuArray_DIMS(%(sm)s),
theano/gpuarray/nnet.py:            size_t n_blocks[3] = {std::min(PyGpuArray_DIMS(%(dx)s)[0], (size_t)256), 1, 1};
theano/gpuarray/nnet.py:            size_t threads_per_block[3] = {std::min(PyGpuArray_DIMS(%(dx)s)[1], (size_t)256), 1, 1};
theano/gpuarray/nnet.py:            ssize_t stride_SM0 = PyGpuArray_STRIDES(%(sm)s)[0] / %(itemsize_sm)s;
theano/gpuarray/nnet.py:            ssize_t stride_SM1 = PyGpuArray_STRIDES(%(sm)s)[1] / %(itemsize_sm)s;
theano/gpuarray/nnet.py:            ssize_t stride_YIDX0 = PyGpuArray_STRIDES(%(y_idx)s)[0] / %(itemsize_y_idx)s;
theano/gpuarray/nnet.py:            ssize_t stride_DX0 = PyGpuArray_STRIDES(%(dx)s)[0] / %(itemsize_dx)s;
theano/gpuarray/nnet.py:            ssize_t stride_DX1 = PyGpuArray_STRIDES(%(dx)s)[1] / %(itemsize_dx)s;
theano/gpuarray/nnet.py:                (void *)&PyGpuArray_DIMS(%(dx)s)[0],
theano/gpuarray/nnet.py:                (void *)&PyGpuArray_DIMS(%(dx)s)[1],
theano/gpuarray/nnet.py:            int err = GpuKernel_call(&%(k_var)s, 3, n_blocks, threads_per_block, 0, kernel_params);
theano/gpuarray/nnet.py:    def gpu_kernels(self, node, nodename):
theano/gpuarray/nnet.py:        wtype_dnll = gpuarray.dtype_to_ctype(work_dnll)
theano/gpuarray/nnet.py:        type_dnll = gpuarray.dtype_to_ctype(dtype_dnll)
theano/gpuarray/nnet.py:        type_sm = gpuarray.dtype_to_ctype(dtype_sm)
theano/gpuarray/nnet.py:        type_y_idx = gpuarray.dtype_to_ctype(dtype_y_idx)
theano/gpuarray/nnet.py:        type_dx = gpuarray.dtype_to_ctype(dtype_dx)
theano/gpuarray/nnet.py:            gpuarray.SIZE, gpuarray.SIZE,
theano/gpuarray/nnet.py:            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE,
theano/gpuarray/nnet.py:            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SSIZE,
theano/gpuarray/nnet.py:            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE,
theano/gpuarray/nnet.py:            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SSIZE,
theano/gpuarray/nnet.py:gpu_crossentropy_softmax_1hot_with_bias_dx = GpuCrossentropySoftmax1HotWithBiasDx()
theano/gpuarray/nnet.py:class GpuSoftmax(GpuKernelBase, Op):
theano/gpuarray/nnet.py:    Implement Softmax on the gpu.
theano/gpuarray/nnet.py:        x = as_gpuarray_variable(x, infer_context_name(x))
theano/gpuarray/nnet.py:        return ['<numpy_compat.h>', '<gpuarray/types.h>']
theano/gpuarray/nnet.py:        typecode = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
theano/gpuarray/nnet.py:        if (PyGpuArray_NDIM(%(x)s) != 2)
theano/gpuarray/nnet.py:            (PyGpuArray_DIMS(%(z)s)[0] !=
theano/gpuarray/nnet.py:             PyGpuArray_DIMS(%(x)s)[0]) ||
theano/gpuarray/nnet.py:            (PyGpuArray_DIMS(%(z)s)[1] !=
theano/gpuarray/nnet.py:             PyGpuArray_DIMS(%(x)s)[1]))
theano/gpuarray/nnet.py:            %(z)s = pygpu_empty(2, PyGpuArray_DIMS(%(x)s),
theano/gpuarray/nnet.py:            size_t n_blocks[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t)(32 * 1024)), 1, 1};
theano/gpuarray/nnet.py:            size_t threads_per_block[3] = {std::min(PyGpuArray_DIMS(%(x)s)[1], (size_t)256), 1, 1}; // TODO: Read GA_CTX_PROP_MAXLSIZE0
theano/gpuarray/nnet.py:            size_t shmem_sz = PyGpuArray_DIMS(%(x)s)[1] *
theano/gpuarray/nnet.py:            ssize_t stride_X0 = PyGpuArray_STRIDES(%(x)s)[0] / %(itemsize_x)s;
theano/gpuarray/nnet.py:            ssize_t stride_X1 = PyGpuArray_STRIDES(%(x)s)[1] / %(itemsize_x)s;
theano/gpuarray/nnet.py:            ssize_t stride_Z0 = PyGpuArray_STRIDES(%(z)s)[0] / %(itemsize_z)s;
theano/gpuarray/nnet.py:            ssize_t stride_Z1 = PyGpuArray_STRIDES(%(z)s)[1] / %(itemsize_z)s;
theano/gpuarray/nnet.py:                (void *)&PyGpuArray_DIMS(%(x)s)[0],
theano/gpuarray/nnet.py:                (void *)&PyGpuArray_DIMS(%(x)s)[1],
theano/gpuarray/nnet.py:            if (PyGpuArray_DIMS(%(x)s)[0] > 0)
theano/gpuarray/nnet.py:              //Those numbers are based on not too recent GPU
theano/gpuarray/nnet.py:              //to make them compatible with more GPU.
theano/gpuarray/nnet.py:                err = GpuKernel_call(&kSoftmax_%(nodename)s, 3,
theano/gpuarray/nnet.py:                fmt_str = "gpuarray error: kSoftmax_%(nodename)s: %%s";
theano/gpuarray/nnet.py:                msg = GpuKernel_error(&kSoftmax_%(nodename)s, err);
theano/gpuarray/nnet.py:                err = GpuKernel_call(&kSoftmax_fixed_shared%(nodename)s, 3,
theano/gpuarray/nnet.py:                fmt_str = "gpuarray error: kSoftmax_fixed_shared%(nodename)s: %%s";
theano/gpuarray/nnet.py:                msg = GpuKernel_error(&kSoftmax_fixed_shared%(nodename)s, err);
theano/gpuarray/nnet.py:    def gpu_kernels(self, node, nodename):
theano/gpuarray/nnet.py:        type_x = gpuarray.dtype_to_ctype(dtype_x)
theano/gpuarray/nnet.py:        type_sm = gpuarray.dtype_to_ctype(dtype_sm)
theano/gpuarray/nnet.py:        type_acc = gpuarray.dtype_to_ctype(work_sm)
theano/gpuarray/nnet.py:        ctype = gpuarray.dtype_to_ctype(work_sm)
theano/gpuarray/nnet.py:            gpuarray.SIZE, gpuarray.SIZE,
theano/gpuarray/nnet.py:            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SSIZE,
theano/gpuarray/nnet.py:            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SSIZE
theano/gpuarray/nnet.py:gpu_softmax = GpuSoftmax()
theano/gpuarray/nnet.py:class GpuSoftmaxWithBias(GpuKernelBase, Op):
theano/gpuarray/nnet.py:    Implement SoftmaxWithBias on the gpu.
theano/gpuarray/nnet.py:        x = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/nnet.py:        b = as_gpuarray_variable(b, ctx_name)
theano/gpuarray/nnet.py:        return ['<numpy_compat.h>', '<gpuarray/types.h>']
theano/gpuarray/nnet.py:        typecode = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
theano/gpuarray/nnet.py:        if (PyGpuArray_NDIM(%(x)s) != 2)
theano/gpuarray/nnet.py:        if (PyGpuArray_NDIM(%(b)s) != 1)
theano/gpuarray/nnet.py:        if ((PyGpuArray_DIMS(%(x)s)[1] !=
theano/gpuarray/nnet.py:            PyGpuArray_DIMS(%(b)s)[0]))
theano/gpuarray/nnet.py:                         (long int)PyGpuArray_DIMS(%(x)s)[1],
theano/gpuarray/nnet.py:                         (long int)PyGpuArray_DIMS(%(b)s)[0]);
theano/gpuarray/nnet.py:            || (PyGpuArray_DIMS(%(z)s)[0] !=
theano/gpuarray/nnet.py:                PyGpuArray_DIMS(%(x)s)[0])
theano/gpuarray/nnet.py:            || (PyGpuArray_DIMS(%(z)s)[1] !=
theano/gpuarray/nnet.py:                PyGpuArray_DIMS(%(x)s)[1]))
theano/gpuarray/nnet.py:            %(z)s = pygpu_empty(2, PyGpuArray_DIMS(%(x)s),
theano/gpuarray/nnet.py:            size_t n_blocks[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t)(32*1024)), 1, 1};
theano/gpuarray/nnet.py:            size_t threads_per_block[3] = {std::min(PyGpuArray_DIMS(%(x)s)[1], (size_t)256), 1, 1}; // TODO: Read GA_CTX_PROP_MAXLSIZE0
theano/gpuarray/nnet.py:            size_t shmem_sz = PyGpuArray_DIMS(%(x)s)[1] *
theano/gpuarray/nnet.py:            ssize_t stride_X0 = PyGpuArray_STRIDES(%(x)s)[0] / %(itemsize_x)s;
theano/gpuarray/nnet.py:            ssize_t stride_X1 = PyGpuArray_STRIDES(%(x)s)[1] / %(itemsize_x)s;
theano/gpuarray/nnet.py:            ssize_t stride_B0 = PyGpuArray_STRIDES(%(b)s)[0] / %(itemsize_b)s;
theano/gpuarray/nnet.py:            ssize_t stride_Z0 = PyGpuArray_STRIDES(%(z)s)[0] / %(itemsize_z)s;
theano/gpuarray/nnet.py:            ssize_t stride_Z1 = PyGpuArray_STRIDES(%(z)s)[1] / %(itemsize_z)s;
theano/gpuarray/nnet.py:                (void *)&PyGpuArray_DIMS(%(x)s)[0],
theano/gpuarray/nnet.py:                (void *)&PyGpuArray_DIMS(%(x)s)[1],
theano/gpuarray/nnet.py:            if (PyGpuArray_DIMS(%(x)s)[0] > 0)
theano/gpuarray/nnet.py:                err = GpuKernel_call(&kSoftmaxWithBias_%(nodename)s, 3,
theano/gpuarray/nnet.py:                fmt_str = "gpuarray error: kSoftmaxWithBias_%(nodename)s: %%s";
theano/gpuarray/nnet.py:                msg = GpuKernel_error(&kSoftmaxWithBias_%(nodename)s, err);
theano/gpuarray/nnet.py:                err = GpuKernel_call(&kSoftmaxWithBias_fixed_shared%(nodename)s,
theano/gpuarray/nnet.py:                fmt_str = "gpuarray error: kSoftmaxWithBias_fixed_shared%(nodename)s: %%s";
theano/gpuarray/nnet.py:                msg = GpuKernel_error(&kSoftmaxWithBias_fixed_shared%(nodename)s, err);
theano/gpuarray/nnet.py:    def gpu_kernels(self, node, nodename):
theano/gpuarray/nnet.py:        type_x = gpuarray.dtype_to_ctype(dtype_x)
theano/gpuarray/nnet.py:        type_b = gpuarray.dtype_to_ctype(dtype_b)
theano/gpuarray/nnet.py:        type_sm = gpuarray.dtype_to_ctype(dtype_sm)
theano/gpuarray/nnet.py:        type_acc = gpuarray.dtype_to_ctype(work_sm)
theano/gpuarray/nnet.py:        ctype = gpuarray.dtype_to_ctype(work_sm)
theano/gpuarray/nnet.py:            gpuarray.SIZE, gpuarray.SIZE,
theano/gpuarray/nnet.py:            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SSIZE,
theano/gpuarray/nnet.py:            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE,
theano/gpuarray/nnet.py:            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SSIZE,
theano/gpuarray/nnet.py:gpu_softmax_with_bias = GpuSoftmaxWithBias()
theano/gpuarray/type.py:# Make sure this is importable even if pygpu is absent
theano/gpuarray/type.py:    import pygpu
theano/gpuarray/type.py:    from pygpu import gpuarray
theano/gpuarray/type.py:    from pygpu.elemwise import compare, elemwise2
theano/gpuarray/type.py:    pygpu = None
theano/gpuarray/type.py:def gpu_supported(data):
theano/gpuarray/type.py:    Is the following data supported on the GPU?
theano/gpuarray/type.py:def move_to_gpu(data):
theano/gpuarray/type.py:    Do we want to move this computation to the GPU?
theano/gpuarray/type.py:    # We don't support complex on the GPU
theano/gpuarray/type.py:    if not gpu_supported(data):
theano/gpuarray/type.py:    # We don't want scalars on the GPU.
theano/gpuarray/type.py:    The context must be of type `GpuContext` and the name can be
theano/gpuarray/type.py:    ctx : GpuContext
theano/gpuarray/type.py:    if not isinstance(ctx, gpuarray.GpuContext):
theano/gpuarray/type.py:        raise TypeError("context is not GpuContext")
theano/gpuarray/type.py:class GpuArrayType(Type):
theano/gpuarray/type.py:    The type that represents an array on a gpu.
theano/gpuarray/type.py:        The name of a gpu context on which variables will have their values.
theano/gpuarray/type.py:        The gpuarray typecode for `dtype`
theano/gpuarray/type.py:            self.typecode = gpuarray.dtype_to_typecode(self.dtype)
theano/gpuarray/type.py:        except gpuarray.GpuArrayException:
theano/gpuarray/type.py:            return "GpuArrayType<%s>(%s, %s)" % (self.context_name, self.dtype, bcast)
theano/gpuarray/type.py:        if (isinstance(data, gpuarray.GpuArray) and
theano/gpuarray/type.py:            if not isinstance(data, gpuarray.GpuArray):
theano/gpuarray/type.py:                raise TypeError("%s expected a GpuArray object." % self,
theano/gpuarray/type.py:            if not isinstance(data, gpuarray.GpuArray):
theano/gpuarray/type.py:                data = gpuarray.array(data, dtype=self.typecode, copy=False,
theano/gpuarray/type.py:                if not isinstance(data, gpuarray.GpuArray):
theano/gpuarray/type.py:                    data = gpuarray.array(data, dtype=self.dtype, copy=False)
theano/gpuarray/type.py:        if not isinstance(data, gpuarray.GpuArray):
theano/gpuarray/type.py:                data = pygpu.array(data, context=self.context)
theano/gpuarray/type.py:        if hasattr(other, '_as_GpuArrayVariable'):
theano/gpuarray/type.py:            other = other._as_GpuArrayVariable(self.context_name)
theano/gpuarray/type.py:        if not isinstance(other.type, (TensorType, GpuArrayType)):
theano/gpuarray/type.py:        if (not isinstance(a, gpuarray.GpuArray) or
theano/gpuarray/type.py:                not isinstance(b, gpuarray.GpuArray)):
theano/gpuarray/type.py:        return pygpu.gpuarray.may_share_memory(a, b)
theano/gpuarray/type.py:        return pygpu.gpuarray.zeros(shape, dtype=self.typecode,
theano/gpuarray/type.py:        return pygpu.gpuarray.dtype_to_ctype(self.dtype)
theano/gpuarray/type.py:        PyGpuArrayObject *%(name)s;
theano/gpuarray/type.py:            PyErr_SetString(PyExc_ValueError, "expected a GpuArray, not None");
theano/gpuarray/type.py:        if (py_%(name)s->ob_type != &PyGpuArrayType &&
theano/gpuarray/type.py:            !PyObject_TypeCheck(py_%(name)s, &PyGpuArrayType)) {
theano/gpuarray/type.py:            PyErr_SetString(PyExc_ValueError, "expected a GpuArray");
theano/gpuarray/type.py:        %(name)s = (PyGpuArrayObject *)py_%(name)s;
theano/gpuarray/type.py:        # HostFromGpu and GpuFromHost and those case will be covered
theano/gpuarray/type.py:        return ['import_pygpu__gpuarray();']
theano/gpuarray/type.py:        return ['<gpuarray/array.h>', '<gpuarray/kernel.h>',
theano/gpuarray/type.py:                '<gpuarray/error.h>', '<gpuarray/buffer.h>',
theano/gpuarray/type.py:                '<gpuarray/buffer_blas.h>', '<numpy/arrayobject.h>',
theano/gpuarray/type.py:                '<gpuarray_api.h>']
theano/gpuarray/type.py:        return [pygpu.get_include(), np.get_include()] + other_dirs
theano/gpuarray/type.py:        return ['gpuarray']
theano/gpuarray/type.py:        ver = pygpu.gpuarray.abi_version()
theano/gpuarray/type.py:        return GpuArrayType.values_eq(a, b)
theano/gpuarray/type.py:# This is to map ndarray-specific versions of these functions to the GPU.
theano/gpuarray/type.py:        from .basic_ops import host_from_gpu
theano/gpuarray/type.py:        return host_from_gpu(self)
theano/gpuarray/type.py:    def _as_GpuArrayVariable(self, context_name):
theano/gpuarray/type.py:            from .basic_ops import GpuToGpu
theano/gpuarray/type.py:            return GpuToGpu(context_name)(self)
theano/gpuarray/type.py:class GpuArrayVariable(_operators, Variable):
theano/gpuarray/type.py:    A variable representing a computation on a certain GPU.
theano/gpuarray/type.py:GpuArrayType.Variable = GpuArrayVariable
theano/gpuarray/type.py:class GpuArraySignature(tensor.TensorConstantSignature):
theano/gpuarray/type.py:    # might do something better if we can run the sum on the GPU, but
theano/gpuarray/type.py:class GpuArrayConstant(_operators, Constant):
theano/gpuarray/type.py:    A constant representing a value on a certain GPU.
theano/gpuarray/type.py:        return GpuArraySignature((self.type, np.asarray(self.data)))
theano/gpuarray/type.py:        except gpuarray.GpuArrayException:
theano/gpuarray/type.py:        return "GpuArrayConstant{%s}" % np_data
theano/gpuarray/type.py:GpuArrayType.Constant = GpuArrayConstant
theano/gpuarray/type.py:class GpuArraySharedVariable(_operators, SharedVariable):
theano/gpuarray/type.py:    A variable representing a shared value on a certain GPU.
theano/gpuarray/type.py:        if isinstance(value, pygpu.gpuarray.GpuArray):
theano/gpuarray/type.py:            value = pygpu.gpuarray.array(value, copy=(not borrow),
theano/gpuarray/type.py:GpuArrayType.SharedVariable = GpuArraySharedVariable
theano/gpuarray/type.py:def gpuarray_shared_constructor(value, name=None, strict=False,
theano/gpuarray/type.py:    SharedVariable constructor for GpuArrayType.
theano/gpuarray/type.py:    if not isinstance(value, (np.ndarray, pygpu.gpuarray.GpuArray)):
theano/gpuarray/type.py:        raise TypeError('ndarray or GpuArray required')
theano/gpuarray/type.py:        if not gpu_supported(value):
theano/gpuarray/type.py:            raise TypeError('The GPU do not support that value.')
theano/gpuarray/type.py:        if not move_to_gpu(value):
theano/gpuarray/type.py:            raise TypeError('We do not move that data by default to the GPU')
theano/gpuarray/type.py:    type = GpuArrayType(value.dtype, broadcastable, context_name=target)
theano/gpuarray/type.py:    deviceval = pygpu.gpuarray.array(value, copy=(not borrow),
theano/gpuarray/type.py:    return GpuArraySharedVariable(type=type, value=deviceval, name=name,
theano/gpuarray/type.py:theano.compile.register_view_op_c_code(GpuArrayType, """
theano/gpuarray/type.py:# Register GpuArrayType C code for Shape Op.
theano/gpuarray/type.py:    GpuArrayType,
theano/gpuarray/type.py:    GpuArrayType,
theano/gpuarray/type.py:theano.compile.register_deep_copy_op_c_code(GpuArrayType, """
theano/gpuarray/type.py:    %(oname)s = pygpu_copy(%(iname)s, GA_ANY_ORDER);
theano/gpuarray/type.py:    GpuArrayType,
theano/gpuarray/type.py:    GpuArrayType,
theano/gpuarray/type.py:        if (PyGpuArray_NDIM(%(iname)s) != PyArray_DIMS(%(shape)s)[0]) {
theano/gpuarray/type.py:                         PyGpuArray_NDIM(%(iname)s),
theano/gpuarray/type.py:        for(int i = 0; i < PyGpuArray_NDIM(%(iname)s); i++){
theano/gpuarray/type.py:            if (PyGpuArray_DIMS(%(iname)s)[i] != shp) {
theano/gpuarray/type.py:                             i, PyGpuArray_DIMS(%(iname)s)[i],
theano/gpuarray/type.py:class GpuContextType(Type):
theano/gpuarray/type.py:        if not isinstance(data, gpuarray.GpuContext):
theano/gpuarray/type.py:            raise TypeError('context is not a GpuContext')
theano/gpuarray/type.py:        return "PyGpuContextObject *%s;" % (name,)
theano/gpuarray/type.py:if (!PyObject_TypeCheck(py_%(name)s, &PyGpuContextType)) {
theano/gpuarray/type.py:  PyErr_SetString(PyExc_TypeError, "expected a GpuContext");
theano/gpuarray/type.py:%(name)s = (PyGpuContextObject *)py_%(name)s;
theano/gpuarray/type.py:        return ['import_pygpu__gpuarray();']
theano/gpuarray/type.py:        return ['<gpuarray_api.h>']
theano/gpuarray/type.py:        return [pygpu.get_include()]
theano/gpuarray/type.py:        ver = pygpu.gpuarray.api_version()
theano/gpuarray/type.py:Instance of :class:`GpuContextType` to use for the context_type
theano/gpuarray/type.py:gpu_context_type = GpuContextType()
theano/gpuarray/type.py:# THIS WORKS But GpuArray instances don't compare equal to one
theano/gpuarray/type.py:def GpuArray_unpickler(npa, ctx_name):
theano/gpuarray/type.py:    if config.experimental.unpickle_gpu_on_cpu:
theano/gpuarray/type.py:            "config.experimental.unpickle_gpu_on_cpu is set to True. "
theano/gpuarray/type.py:            "Unpickling GpuArray as numpy.ndarray")
theano/gpuarray/type.py:    elif pygpu:
theano/gpuarray/type.py:        return pygpu.gpuarray.array(npa, copy=True, context=ctx)
theano/gpuarray/type.py:        raise ImportError("pygpu not found. Cannot unpickle GpuArray")
theano/gpuarray/type.py:copyreg.constructor(GpuArray_unpickler)
theano/gpuarray/type.py:def GpuArray_pickler(cnda):
theano/gpuarray/type.py:    return (GpuArray_unpickler, (np.asarray(cnda), ctx_name))
theano/gpuarray/type.py:# In case pygpu is not imported.
theano/gpuarray/type.py:if pygpu is not None:
theano/gpuarray/type.py:    copyreg.pickle(pygpu.gpuarray.GpuArray,
theano/gpuarray/type.py:                   GpuArray_pickler,
theano/gpuarray/type.py:                   GpuArray_unpickler)
theano/gpuarray/fft.py:from .basic_ops import (gpu_contiguous, as_gpuarray_variable,
theano/gpuarray/fft.py:from .type import GpuArrayType
theano/gpuarray/fft.py:    import pygpu
theano/gpuarray/fft.py:    pygpu_available = True
theano/gpuarray/fft.py:    pygpu_available = False
theano/gpuarray/fft.py:    import pycuda.driver
theano/gpuarray/fft.py:    pycuda_available = True
theano/gpuarray/fft.py:    pycuda_available = False
theano/gpuarray/fft.py:    import skcuda
theano/gpuarray/fft.py:    from skcuda import fft
theano/gpuarray/fft.py:    skcuda_available = True
theano/gpuarray/fft.py:    skcuda_available = False
theano/gpuarray/fft.py:        return GpuArrayType(inp.dtype,
theano/gpuarray/fft.py:        if not skcuda_available:
theano/gpuarray/fft.py:            raise RuntimeError("skcuda is needed for CuFFTOp")
theano/gpuarray/fft.py:        if not pygpu_available:
theano/gpuarray/fft.py:            raise RuntimeError("pygpu is needed for CuFFTOp")
theano/gpuarray/fft.py:        if not pycuda_available:
theano/gpuarray/fft.py:            raise RuntimeError("pycuda is needed for CuFFTOp")
theano/gpuarray/fft.py:        inp = gpu_contiguous(as_gpuarray_variable(inp,
theano/gpuarray/fft.py:        # Initiliaze cuda context to the input's.
theano/gpuarray/fft.py:            skcuda.misc.init()
theano/gpuarray/fft.py:                z[0] = pygpu.zeros(output_shape, context=inputs[0][0].context,
theano/gpuarray/fft.py:            input_pycuda = inputs[0][0]
theano/gpuarray/fft.py:            # I thought we'd need to change the type on output_pycuda
theano/gpuarray/fft.py:            # so it is complex64, but as it turns out skcuda.fft
theano/gpuarray/fft.py:            output_pycuda = z[0]
theano/gpuarray/fft.py:            with input_pycuda.context:
theano/gpuarray/fft.py:                # Sync GPU variables before computation
theano/gpuarray/fft.py:                input_pycuda.sync()
theano/gpuarray/fft.py:                output_pycuda.sync()
theano/gpuarray/fft.py:                fft.fft(input_pycuda, output_pycuda, plan[0])
theano/gpuarray/fft.py:                pycuda.driver.Context.synchronize()
theano/gpuarray/fft.py:        return GpuArrayType(inp.dtype,
theano/gpuarray/fft.py:        if not skcuda_available:
theano/gpuarray/fft.py:            raise RuntimeError("skcuda is needed for CuIFFTOp")
theano/gpuarray/fft.py:        if not pygpu_available:
theano/gpuarray/fft.py:            raise RuntimeError("pygpu is needed for CuIFFTOp")
theano/gpuarray/fft.py:        if not pycuda_available:
theano/gpuarray/fft.py:            raise RuntimeError("pycuda is needed for CuIFFTOp")
theano/gpuarray/fft.py:        inp = gpu_contiguous(as_gpuarray_variable(inp,
theano/gpuarray/fft.py:        # Initiliaze cuda context to the input's.
theano/gpuarray/fft.py:            skcuda.misc.init()
theano/gpuarray/fft.py:                z[0] = pygpu.zeros(output_shape, context=inputs[0][0].context,
theano/gpuarray/fft.py:            input_pycuda = inputs[0][0]
theano/gpuarray/fft.py:            # input_pycuda is a float32 array with an extra dimension,
theano/gpuarray/fft.py:            # but will be interpreted by skcuda as a complex64
theano/gpuarray/fft.py:            output_pycuda = z[0]
theano/gpuarray/fft.py:            with input_pycuda.context:
theano/gpuarray/fft.py:                # Sync GPU variables before computation
theano/gpuarray/fft.py:                input_pycuda.sync()
theano/gpuarray/fft.py:                output_pycuda.sync()
theano/gpuarray/fft.py:                fft.ifft(input_pycuda, output_pycuda, plan[0])
theano/gpuarray/fft.py:                pycuda.driver.Context.synchronize()
theano/gpuarray/fft.py:    Performs the fast Fourier transform of a real-valued input on the GPU.
theano/gpuarray/fft.py:    The output is a GpuArray of dimensions (m, ..., n//2+1, 2). The second to
theano/gpuarray/fft.py:    Performs the inverse fast Fourier Transform with real-valued output on the GPU.
theano/gpuarray/fft.py:if skcuda_available:
theano/gpuarray/fft.py:    def local_gpua_curfft_op(op, ctx_name, inputs, outputs):
theano/gpuarray/fft.py:    def local_gpua_cuirfft_op(op, ctx_name, inputs, outputs):
theano/gpuarray/blocksparse.py:from .type import gpu_context_type
theano/gpuarray/blocksparse.py:from .basic_ops import as_gpuarray_variable, infer_context_name, gpuarray_helper_inc_dir
theano/gpuarray/blocksparse.py:_logger = logging.getLogger('theano.gpuarray.blocksparse')
theano/gpuarray/blocksparse.py:class GpuSparseBlockGemv(COp):
theano/gpuarray/blocksparse.py:    GPU version of SparseBlockGemv. Check SparseBlockGemv's docstring for more
theano/gpuarray/blocksparse.py:    params_type = ParamsType(inplace=bool_t, context=gpu_context_type)
theano/gpuarray/blocksparse.py:        return [gpuarray_helper_inc_dir()]
theano/gpuarray/blocksparse.py:        return ['<gpuarray/buffer_blas.h>', '<gpuarray/buffer.h>',
theano/gpuarray/blocksparse.py:                '<gpuarray_helper.h>']
theano/gpuarray/blocksparse.py:        o = as_gpuarray_variable(o, ctx)
theano/gpuarray/blocksparse.py:        W = as_gpuarray_variable(W, ctx)
theano/gpuarray/blocksparse.py:        h = as_gpuarray_variable(h, ctx)
theano/gpuarray/blocksparse.py:        Wgrad = gpu_sparse_block_outer(W.zeros_like(),
theano/gpuarray/blocksparse.py:        hgrad = gpu_sparse_block_gemv(h.zeros_like(),
theano/gpuarray/blocksparse.py:gpu_sparse_block_gemv = GpuSparseBlockGemv(False)
theano/gpuarray/blocksparse.py:gpu_sparse_block_gemv_inplace = GpuSparseBlockGemv(True)
theano/gpuarray/blocksparse.py:class GpuSparseBlockOuter(COp):
theano/gpuarray/blocksparse.py:    GPU version of SparseBlockOuter. See SparseBlockOuter's docstring for more
theano/gpuarray/blocksparse.py:    of GpuSparseBlockGemv. The gradient is not implemented.
theano/gpuarray/blocksparse.py:    params_type = ParamsType(inplace=bool_t, context=gpu_context_type)
theano/gpuarray/blocksparse.py:        o = as_gpuarray_variable(o, ctx)
theano/gpuarray/blocksparse.py:        x = as_gpuarray_variable(x, ctx)
theano/gpuarray/blocksparse.py:        y = as_gpuarray_variable(y, ctx)
theano/gpuarray/blocksparse.py:        return [gpuarray_helper_inc_dir()]
theano/gpuarray/blocksparse.py:        return ['<gpuarray/buffer_blas.h>', '<gpuarray/buffer.h>',
theano/gpuarray/blocksparse.py:                '<gpuarray_helper.h>']
theano/gpuarray/blocksparse.py:gpu_sparse_block_outer = GpuSparseBlockOuter(False)
theano/gpuarray/blocksparse.py:gpu_sparse_block_outer_inplace = GpuSparseBlockOuter(True)
theano/gpuarray/subtensor.py:    import pygpu
theano/gpuarray/subtensor.py:    from pygpu import gpuarray
theano/gpuarray/subtensor.py:from .type import GpuArrayType, gpu_context_type
theano/gpuarray/subtensor.py:from .basic_ops import (as_gpuarray_variable, HideC, GpuKernelBase, Kernel, gpuarray_helper_inc_dir,
theano/gpuarray/subtensor.py:                        infer_context_name, gpu_contiguous)
theano/gpuarray/subtensor.py:        a_arg = pygpu.elemwise.arg('a', a.type.dtype, read=True, write=True)
theano/gpuarray/subtensor.py:        b_arg = pygpu.elemwise.arg('b', b.type.dtype, read=True)
theano/gpuarray/subtensor.py:        res = pygpu.elemwise.GpuElemwise(a.type.context, "a = a + b", [a_arg, b_arg], convert_f16=True)
theano/gpuarray/subtensor.py:class GpuSubtensor(HideC, Subtensor):
theano/gpuarray/subtensor.py:    Subtensor on the GPU.
theano/gpuarray/subtensor.py:        otype = GpuArrayType(dtype=rval.outputs[0].type.dtype,
theano/gpuarray/subtensor.py:        x = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/subtensor.py:        # This case fails when we use pygpu_index(), so here is some
theano/gpuarray/subtensor.py:        %(out)s = pygpu_copy(%(inp)s, GA_ANY_ORDER);
theano/gpuarray/subtensor.py:        %(out)s = pygpu_index(%(inp)s, starts, stops, steps);
theano/gpuarray/subtensor.py:class GpuIncSubtensor(IncSubtensor):
theano/gpuarray/subtensor.py:    Implement IncSubtensor on the gpu.
theano/gpuarray/subtensor.py:    The same optimization handles IncSubtensor and GpuIncSubtensor.
theano/gpuarray/subtensor.py:    params_type = gpu_context_type
theano/gpuarray/subtensor.py:        x = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/subtensor.py:        y = as_gpuarray_variable(y, ctx_name)
theano/gpuarray/subtensor.py:                tmp = pygpu.elemwise.elemwise2(sub_x, '+', y, sub_x,
theano/gpuarray/subtensor.py:        if not isinstance(node.inputs[0].type, GpuArrayType):
theano/gpuarray/subtensor.py:        return """pygpu_copy(%(x)s, GA_ANY_ORDER)""" % locals()
theano/gpuarray/subtensor.py:        return "PyGpuArrayObject* zview = NULL;"
theano/gpuarray/subtensor.py:        zview = pygpu_fromgpudata(%(x)s->ga.data,
theano/gpuarray/subtensor.py:                                  (PyObject *)&PyGpuArrayType);
theano/gpuarray/subtensor.py:        return {'c_prefix': 'PyGpuArray',
theano/gpuarray/subtensor.py:        return ['<numpy_compat.h>', '<gpuarray/error.h>', '<gpuarray/array.h>',
theano/gpuarray/subtensor.py:                '<gpuarray/elemwise.h>']
theano/gpuarray/subtensor.py:int sub_setarray(GpuArray *dst, GpuArray *src) {
theano/gpuarray/subtensor.py:  err = GpuArray_setarray(dst, src);
theano/gpuarray/subtensor.py:    PyErr_SetString(PyExc_RuntimeError, GpuArray_error(src, err));
theano/gpuarray/subtensor.py:        return "\nGpuElemwise *iadd;\n"
theano/gpuarray/subtensor.py:        gpuelemwise_arg args[2] = {{0}};
theano/gpuarray/subtensor.py:        iadd = GpuElemwise_new(%(ctx)s->ctx, "", "a += b",
theano/gpuarray/subtensor.py:          if (GpuElemwise_call(iadd, args, GE_BROADCAST | GE_PADSHAPE) != GA_NO_ERROR) {
theano/gpuarray/subtensor.py:        parent_version = super(GpuIncSubtensor, self).c_code_cache_version()
theano/gpuarray/subtensor.py:class GpuAdvancedSubtensor1(HideC, tensor.AdvancedSubtensor1):
theano/gpuarray/subtensor.py:    AdvancedSubrensor1 on the GPU.
theano/gpuarray/subtensor.py:        x_ = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/subtensor.py:        ilist_ = gpu_contiguous(as_gpuarray_variable(ilist__, ctx_name))
theano/gpuarray/subtensor.py:                         [GpuArrayType(dtype=x.dtype,
theano/gpuarray/subtensor.py:int take1_match_dims(GpuArray *a, GpuArray *v) {
theano/gpuarray/subtensor.py:if (%(out)s == NULL || !GpuArray_IS_C_CONTIGUOUS(&%(out)s->ga) ||
theano/gpuarray/subtensor.py:  %(out)s = pygpu_empty(%(v)s->ga.nd, %(v)s->ga.dimensions, %(v)s->ga.typecode,
theano/gpuarray/subtensor.py:err = GpuArray_take1(&%(out)s->ga, &%(v)s->ga, &%(idx)s->ga, 1);
theano/gpuarray/subtensor.py:    PyErr_SetString(PyExc_RuntimeError, GpuArray_error(&%(v)s->ga, err));
theano/gpuarray/subtensor.py:class BaseGpuAdvancedSubtensor(object):
theano/gpuarray/subtensor.py:        out_flat = input_flat.take1(pygpu.asarray(take_idx.flatten(),
theano/gpuarray/subtensor.py:class GpuAdvancedSubtensor(HideC, BaseGpuAdvancedSubtensor, tensor.AdvancedSubtensor):
theano/gpuarray/subtensor.py:    AdvancedSubtensor on the GPU.
theano/gpuarray/subtensor.py:        otype = GpuArrayType(dtype=rval.outputs[0].type.dtype,
theano/gpuarray/subtensor.py:        x = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/subtensor.py:class GpuAdvancedBooleanSubtensor(HideC, BaseGpuAdvancedSubtensor, tensor.AdvancedBooleanSubtensor):
theano/gpuarray/subtensor.py:    AdvancedBooleanSubtensor on the GPU.
theano/gpuarray/subtensor.py:        otype = GpuArrayType(dtype=rval.outputs[0].type.dtype,
theano/gpuarray/subtensor.py:        x = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/subtensor.py:class BaseGpuAdvancedIncSubtensor(object):
theano/gpuarray/subtensor.py:        # Get a handle to the GpuElemwise object that will be called.
theano/gpuarray/subtensor.py:            if isinstance(idx[i], gpuarray.GpuArray):
theano/gpuarray/subtensor.py:class GpuAdvancedIncSubtensor(HideC, BaseGpuAdvancedIncSubtensor, tensor.AdvancedIncSubtensor):
theano/gpuarray/subtensor.py:    Implement AdvancedIncSubtensor on the gpu.
theano/gpuarray/subtensor.py:        otype = GpuArrayType(dtype=rval.outputs[0].type.dtype,
theano/gpuarray/subtensor.py:        x = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/subtensor.py:        y = as_gpuarray_variable(y, ctx_name)
theano/gpuarray/subtensor.py:class GpuAdvancedBooleanIncSubtensor(HideC, BaseGpuAdvancedIncSubtensor, tensor.AdvancedBooleanIncSubtensor):
theano/gpuarray/subtensor.py:    Implement AdvancedBooleanIncSubtensor on the gpu.
theano/gpuarray/subtensor.py:        otype = GpuArrayType(dtype=rval.outputs[0].type.dtype,
theano/gpuarray/subtensor.py:        x = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/subtensor.py:        y = as_gpuarray_variable(y, ctx_name)
theano/gpuarray/subtensor.py:class GpuAdvancedIncSubtensor1(Op):
theano/gpuarray/subtensor.py:    Implement AdvancedIncSubtensor1 on the gpu.
theano/gpuarray/subtensor.py:                             context=gpu_context_type,
theano/gpuarray/subtensor.py:        x_ = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/subtensor.py:        y_ = as_gpuarray_variable(y, ctx_name)
theano/gpuarray/subtensor.py:        # Make sure idx is not a GpuArray otherwise we cannot use its
theano/gpuarray/subtensor.py:        if isinstance(idx, gpuarray.GpuArray):
theano/gpuarray/subtensor.py:        return ['<numpy_compat.h>', '<gpuarray/error.h>', '<gpuarray/array.h>',
theano/gpuarray/subtensor.py:                '<gpuarray/elemwise.h>', 'gpuarray_helper.h']
theano/gpuarray/subtensor.py:        return [gpuarray_helper_inc_dir()]
theano/gpuarray/subtensor.py:        return "\nGpuElemwise *iadd;\n"
theano/gpuarray/subtensor.py:        gpuelemwise_arg args[2] = {{0}};
theano/gpuarray/subtensor.py:        iadd = GpuElemwise_new(%(params)s->context->ctx, "", "a += b",
theano/gpuarray/subtensor.py:        PyGpuArrayObject *row_x, *row_y;
theano/gpuarray/subtensor.py:          broadcast_y = PyGpuArray_DIM(%(y)s, 0) == 1;
theano/gpuarray/subtensor.py:              start[0] += PyGpuArray_DIM(%(out)s, 0);
theano/gpuarray/subtensor.py:            if (start[0] < 0 || start[0] >= PyGpuArray_DIM(%(out)s, 0)) {
theano/gpuarray/subtensor.py:            row_x = pygpu_index(%(out)s, start, (ssize_t *)PyGpuArray_DIMS(%(out)s), step);
theano/gpuarray/subtensor.py:            row_y = pygpu_index(%(y)s, start, (ssize_t *)PyGpuArray_DIMS(%(y)s), step);
theano/gpuarray/subtensor.py:              ret = GpuArray_setarray(&row_x->ga, &row_y->ga);
theano/gpuarray/subtensor.py:              ret = GpuElemwise_call(iadd, args, GE_BROADCAST | GE_PADSHAPE);
theano/gpuarray/subtensor.py:class GpuAdvancedIncSubtensor1_dev20(GpuKernelBase, HideC,
theano/gpuarray/subtensor.py:                                     GpuAdvancedIncSubtensor1):
theano/gpuarray/subtensor.py:    Implement AdvancedIncSubtensor1 on the gpu with atomics
theano/gpuarray/subtensor.py:    params_type = GpuAdvancedIncSubtensor1.params_type
theano/gpuarray/subtensor.py:    get_params = GpuAdvancedIncSubtensor1.get_params
theano/gpuarray/subtensor.py:        It differs from GpuAdvancedIncSubtensor1 in that it makes sure
theano/gpuarray/subtensor.py:        x_ = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/subtensor.py:        y_ = as_gpuarray_variable(y.astype(x.dtype), ctx_name)
theano/gpuarray/subtensor.py:        ilist_ = as_gpuarray_variable(ilist, ctx_name)
theano/gpuarray/subtensor.py:        return super(GpuAdvancedIncSubtensor1_dev20, self).perform(node, inp, out)
theano/gpuarray/subtensor.py:        return ['<numpy_compat.h>', '<gpuarray_helper.h>',
theano/gpuarray/subtensor.py:                '<gpuarray/types.h>']
theano/gpuarray/subtensor.py:        return [gpuarray_helper_inc_dir()]
theano/gpuarray/subtensor.py:if (GpuArray_vector_add_fast(%(out)s, %(y)s, %(ind)s, %(params)s->set_instead_of_inc)) {
theano/gpuarray/subtensor.py:    def gpu_kernels(self, node, nodename):
theano/gpuarray/subtensor.py:        type_x = gpuarray.dtype_to_ctype(dtype_x)
theano/gpuarray/subtensor.py:        type_y = gpuarray.dtype_to_ctype(dtype_y)
theano/gpuarray/subtensor.py:        type_ind = gpuarray.dtype_to_ctype(dtype_ind)
theano/gpuarray/subtensor.py:        from pygpu.gpuarray import SIZE, SSIZE
theano/gpuarray/subtensor.py:            SIZE, SIZE, SSIZE, SSIZE, gpuarray.GpuArray, SIZE,
theano/gpuarray/subtensor.py:            SIZE, SIZE, SSIZE, SSIZE, gpuarray.GpuArray, SIZE,
theano/gpuarray/subtensor.py:            SIZE, SSIZE, gpuarray.GpuArray, SIZE, 'int32',
theano/gpuarray/subtensor.py:            gpuarray.GpuArray]
theano/gpuarray/subtensor.py:        return super(GpuAdvancedIncSubtensor1_dev20, self).c_support_code_struct(node, nodename) + """
theano/gpuarray/subtensor.py:        int GpuArray_vector_add_fast(PyGpuArrayObject* py_self,
theano/gpuarray/subtensor.py:                                     PyGpuArrayObject* py_other,
theano/gpuarray/subtensor.py:                                     PyGpuArrayObject* indices_arr,
theano/gpuarray/subtensor.py:            size_t threads_per_block = std::min(PyGpuArray_DIMS(py_self)[1], (size_t)256);
theano/gpuarray/subtensor.py:            size_t n_blocks = std::min(PyGpuArray_SIZE(indices_arr), (size_t)4096);
theano/gpuarray/subtensor.py:            gpudata *errbuf;
theano/gpuarray/subtensor.py:            size_t itemsize_x = GpuArray_ITEMSIZE(&py_self->ga);
theano/gpuarray/subtensor.py:            size_t itemsize_y = GpuArray_ITEMSIZE(&py_other->ga);
theano/gpuarray/subtensor.py:            size_t itemsize_ind = GpuArray_ITEMSIZE(&indices_arr->ga);
theano/gpuarray/subtensor.py:              err = gpudata_property(py_self->ga.data,
theano/gpuarray/subtensor.py:        PyGpuArray_DIMS(py_self)[0],
theano/gpuarray/subtensor.py:        PyGpuArray_DIMS(py_self)[1],
theano/gpuarray/subtensor.py:        PyGpuArray_STRIDES(py_self)[0] / itemsize_x,
theano/gpuarray/subtensor.py:        PyGpuArray_STRIDES(py_self)[1] / itemsize_x,
theano/gpuarray/subtensor.py:        PyGpuArray_DIMS(py_other)[0],
theano/gpuarray/subtensor.py:        PyGpuArray_DIMS(py_other)[1],
theano/gpuarray/subtensor.py:        PyGpuArray_DIMS(py_other)[0] == 1 ? 0 : PyGpuArray_STRIDES(py_other)[0] / itemsize_y,
theano/gpuarray/subtensor.py:        PyGpuArray_DIMS(py_other)[1] == 1 ? 0 : PyGpuArray_STRIDES(py_other)[1] / itemsize_y,
theano/gpuarray/subtensor.py:        PyGpuArray_DIMS(indices_arr)[0],
theano/gpuarray/subtensor.py:        PyGpuArray_STRIDES(indices_arr)[0] / itemsize_ind,
theano/gpuarray/subtensor.py:                             "gpuarray error: %(k_var)s: %%s.",
theano/gpuarray/subtensor.py:                             GpuKernel_error(&%(k_var)s, err));
theano/gpuarray/subtensor.py:              err = gpudata_read(&kerr, errbuf, 0, sizeof(int));
theano/gpuarray/subtensor.py:                gpudata_write(errbuf, 0, &kerr, sizeof(int));
theano/gpuarray/subtensor.py:class GpuExtractDiag(Op):
theano/gpuarray/subtensor.py:        x = as_gpuarray_variable(_x, ctx_name)
theano/gpuarray/subtensor.py:class GpuAllocDiag(AllocDiag):
theano/gpuarray/subtensor.py:        diag = as_gpuarray_variable(diag, ctx_name)
theano/gpuarray/subtensor.py:        result_buffer = gpuarray.zeros(result_buffer_shape,
theano/gpuarray/subtensor.py:        return [GpuExtractDiag(offset=self.offset, axis1=self.axis1, axis2=self.axis2)(gz)]
theano/gpuarray/__init__.py:_logger_name = 'theano.gpuarray'
theano/gpuarray/__init__.py:pygpu_activated = False
theano/gpuarray/__init__.py:theano_gpu_is_already_active = False
theano/gpuarray/__init__.py:    import pygpu
theano/gpuarray/__init__.py:    import pygpu.gpuarray
theano/gpuarray/__init__.py:    pygpu = None
theano/gpuarray/__init__.py:# This is for documentation not to depend on the availability of pygpu
theano/gpuarray/__init__.py:from .type import (GpuArrayType, GpuArrayVariable, GpuArrayConstant,
theano/gpuarray/__init__.py:                   GpuArraySharedVariable, gpuarray_shared_constructor,
theano/gpuarray/__init__.py:from .basic_ops import as_gpuarray_variable
theano/gpuarray/__init__.py:        return as_gpuarray_variable(x, target)
theano/gpuarray/__init__.py:def pygpu_parse_version(version_string):
theano/gpuarray/__init__.py:    global pygpu_activated
theano/gpuarray/__init__.py:    global theano_gpu_is_already_active
theano/gpuarray/__init__.py:    if not theano_gpu_is_already_active and os.environ.get('THEANO_GPU_IS_ALREADY_ACTIVE', '') == 'Yes':
theano/gpuarray/__init__.py:        raise RuntimeError("You can't initialize the GPU in a subprocess if the parent process already did it")
theano/gpuarray/__init__.py:        raise RuntimeError("The new gpu-backend need a c++ compiler.")
theano/gpuarray/__init__.py:    pygpu_version = pygpu_parse_version(pygpu.__version__)
theano/gpuarray/__init__.py:    if (pygpu_version.major != 0 or pygpu_version.minor != 7 or
theano/gpuarray/__init__.py:            pygpu_version.patch < 0):
theano/gpuarray/__init__.py:            "Your installed version of pygpu(%s) is too old, please upgrade to 0.7.0 or later (but below 0.8.0)" %
theano/gpuarray/__init__.py:            pygpu_version.fullversion)
theano/gpuarray/__init__.py:    gpuarray_version_major_supported = 2
theano/gpuarray/__init__.py:    gpuarray_version_major_detected = pygpu.gpuarray.api_version()[0]
theano/gpuarray/__init__.py:    if gpuarray_version_major_detected != gpuarray_version_major_supported:
theano/gpuarray/__init__.py:            "Your installed version of libgpuarray is not in sync with the current Theano"
theano/gpuarray/__init__.py:            " version. The installed libgpuarray version supports API version %d,"
theano/gpuarray/__init__.py:            " libgpuarray or Theano to fix this problem.",
theano/gpuarray/__init__.py:            gpuarray_version_major_detected,
theano/gpuarray/__init__.py:            gpuarray_version_major_supported)
theano/gpuarray/__init__.py:        if config.gpuarray.cache_path != '':
theano/gpuarray/__init__.py:            args['kernel_cache_path'] = config.gpuarray.cache_path
theano/gpuarray/__init__.py:            preallocate = config.gpuarray.preallocate
theano/gpuarray/__init__.py:        context = pygpu.init(
theano/gpuarray/__init__.py:            sched=config.gpuarray.sched,
theano/gpuarray/__init__.py:            single_stream=config.gpuarray.single_stream,
theano/gpuarray/__init__.py:        os.environ['THEANO_GPU_IS_ALREADY_ACTIVE'] = 'Yes'
theano/gpuarray/__init__.py:        theano_gpu_is_already_active = True
theano/gpuarray/__init__.py:        if dev.startswith('cuda'):
theano/gpuarray/__init__.py:            # If we try to enable cudnn and there isn't enough GPU
theano/gpuarray/__init__.py:                    "Can not enable cuDNN as there is only %d MB of free GPU memory." %
theano/gpuarray/__init__.py:                    "Trying to preallocate %d MB of GPU memory while only"
theano/gpuarray/__init__.py:            # which will reserve that amount of memory on the GPU.
theano/gpuarray/__init__.py:            pygpu.empty((gmem,), dtype='int8', context=context)
theano/gpuarray/__init__.py:        tmp = pygpu.empty((2, 2), dtype='float32', context=context)
theano/gpuarray/__init__.py:        if dev.startswith('cuda'):
theano/gpuarray/__init__.py:            # In OpenCL, BLAS isn't always available
theano/gpuarray/__init__.py:            pygpu.blas.gemm(0, tmp, tmp, 0, tmp, overwrite_c=True)
theano/gpuarray/__init__.py:        except pygpu.gpuarray.UnsupportedException:
theano/gpuarray/__init__.py:    pygpu_activated = True
theano/gpuarray/__init__.py:# This maps things like 'cuda0' to the context object on that device.
theano/gpuarray/__init__.py:        default_to_move_computation_to_gpu=True,
theano/gpuarray/__init__.py:        move_shared_to_gpu=True,
theano/gpuarray/__init__.py:    Error and warning about CUDA should be displayed only when this
theano/gpuarray/__init__.py:        "cuda", "cuda0", "cudaN", "" (N is the device number to use).
theano/gpuarray/__init__.py:        Will always raise an exception if we can't use the gpu.
theano/gpuarray/__init__.py:    default_to_move_computation_to_gpu
theano/gpuarray/__init__.py:        If gpu init succeeded, enable by default optimizations to move
theano/gpuarray/__init__.py:        computations to the gpu.
theano/gpuarray/__init__.py:    move_shared_to_gpu
theano/gpuarray/__init__.py:        If gpu init succeeded, put new shared variables on the gpu.
theano/gpuarray/__init__.py:        gpuarray.preallocate.
theano/gpuarray/__init__.py:        if not (device.startswith('cuda') or device.startswith('opencl')):
theano/gpuarray/__init__.py:    if default_to_move_computation_to_gpu:
theano/gpuarray/__init__.py:        optdb.add_tags('gpuarray_opt', 'fast_run', 'fast_compile')
theano/gpuarray/__init__.py:        optdb.add_tags('gpua_scanOp_make_inplace', 'fast_run')
theano/gpuarray/__init__.py:    if move_shared_to_gpu:
theano/gpuarray/__init__.py:        theano.compile.shared_constructor(gpuarray_shared_constructor)
theano/gpuarray/__init__.py:if pygpu:
theano/gpuarray/__init__.py:        if (config.device.startswith('cuda') or
theano/gpuarray/__init__.py:                config.device.startswith('opencl')):
theano/gpuarray/__init__.py:        elif (config.init_gpu_device.startswith('cuda') or
theano/gpuarray/__init__.py:              config.init_gpu_device.startswith('opencl')):
theano/gpuarray/__init__.py:                    'you must set device=cpu to use init_gpu_device.')
theano/gpuarray/__init__.py:                print("Using contexts will make init_gpu_device act like device and move all computations by default, which might not be what you want.")
theano/gpuarray/__init__.py:            init_dev(config.init_gpu_device)
theano/gpuarray/__init__.py:            # To have shared var default on the GPU and opt to move to the GPU.
theano/gpuarray/__init__.py:        error("Could not initialize pygpu, support disabled", exc_info=True)
theano/gpuarray/__init__.py:    from .basic_ops import (GpuAlloc, GpuAllocEmpty, GpuContiguous, GpuEye,
theano/gpuarray/__init__.py:                            GpuFromHost, GpuJoin, GpuReshape, GpuSplit,
theano/gpuarray/__init__.py:                            HostFromGpu, host_from_gpu)
theano/gpuarray/__init__.py:    from .elemwise import GpuElemwise
theano/gpuarray/__init__.py:    from .subtensor import (GpuSubtensor, GpuIncSubtensor,
theano/gpuarray/__init__.py:                            GpuAdvancedIncSubtensor1)
theano/gpuarray/__init__.py:    if (config.init_gpu_device.startswith('cuda') or
theano/gpuarray/__init__.py:            config.init_gpu_device.startswith('opencl') or
theano/gpuarray/__init__.py:            config.device.startswith('opencl') or
theano/gpuarray/__init__.py:            config.device.startswith('cuda') or
theano/gpuarray/__init__.py:        error("pygpu was configured but could not be imported or is too old (version 0.7 or higher required)",
theano/gpuarray/blas.py:from .basic_ops import (GpuArrayType, CGpuKernelBase,
theano/gpuarray/blas.py:                        as_gpuarray_variable, gpu_contiguous, infer_context_name, gpuarray_helper_inc_dir)
theano/gpuarray/blas.py:    import pygpu
theano/gpuarray/blas.py:    from pygpu import blas
theano/gpuarray/blas.py:        return ['<blas_api.h>', '<numpy_compat.h>', '<gpuarray_helper.h>']
theano/gpuarray/blas.py:        return [pygpu.get_include(), gpuarray_helper_inc_dir()]
theano/gpuarray/blas.py:        return ['import_pygpu__blas();']
theano/gpuarray/blas.py:class GpuGemv(BlasOp):
theano/gpuarray/blas.py:    Gemv on the GPU.
theano/gpuarray/blas.py:        A = as_gpuarray_variable(A, ctx_name)
theano/gpuarray/blas.py:        x = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/blas.py:        y = as_gpuarray_variable(y, ctx_name)
theano/gpuarray/blas.py:            out_storage[0][0] = pygpu.zeros(y.shape, dtype=y.dtype,
theano/gpuarray/blas.py:        if (PyGpuArray_DIM(%(A)s, 1) == 0) {
theano/gpuarray/blas.py:          code = GpuArray_memset(&%(out)s->ga, 0);
theano/gpuarray/blas.py:        } else if ( PyGpuArray_DIM(%(A)s, 0) == 1
theano/gpuarray/blas.py:            if (pygpu_blas_rdot(%(x)s, %(A)s, %(out)s, 0) == -1) {
theano/gpuarray/blas.py:            pygpu_blas_rgemv(cb_no_trans,
theano/gpuarray/blas.py:gpugemv_no_inplace = GpuGemv(inplace=False)
theano/gpuarray/blas.py:gpugemv_inplace = GpuGemv(inplace=True)
theano/gpuarray/blas.py:class GpuGemm(BlasOp):
theano/gpuarray/blas.py:    Gemm on the GPU.
theano/gpuarray/blas.py:        A = as_gpuarray_variable(A, ctx_name)
theano/gpuarray/blas.py:        B = as_gpuarray_variable(B, ctx_name)
theano/gpuarray/blas.py:        C = as_gpuarray_variable(C, ctx_name)
theano/gpuarray/blas.py:               if (!%(params)s->inplace || !GpuArray_ISONESEGMENT(&%(C)s->ga)) {
theano/gpuarray/blas.py:               if (pygpu_blas_rgemm(cb_no_trans, cb_no_trans,
theano/gpuarray/blas.py:gpugemm_no_inplace = GpuGemm(inplace=False)
theano/gpuarray/blas.py:gpugemm_inplace = GpuGemm(inplace=True)
theano/gpuarray/blas.py:class GpuGer(BlasOp):
theano/gpuarray/blas.py:    Ger on the GPU.
theano/gpuarray/blas.py:        A = as_gpuarray_variable(A, ctx_name)
theano/gpuarray/blas.py:        x = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/blas.py:        y = as_gpuarray_variable(y, ctx_name)
theano/gpuarray/blas.py:               if (!%(params)s->inplace || !GpuArray_ISONESEGMENT(&%(A)s->ga)) {
theano/gpuarray/blas.py:               if (pygpu_blas_rger(((dtype_%(alpha)s *)PyArray_DATA(%(alpha)s))[0],
theano/gpuarray/blas.py:gpuger_no_inplace = GpuGer(inplace=False)
theano/gpuarray/blas.py:gpuger_inplace = GpuGer(inplace=True)
theano/gpuarray/blas.py:class GpuDot22(BlasOp):
theano/gpuarray/blas.py:    Dot22 on the GPU.
theano/gpuarray/blas.py:        x = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/blas.py:        y = as_gpuarray_variable(y, ctx_name)
theano/gpuarray/blas.py:        out = pygpu.empty((x.shape[0], y.shape[1]), dtype=x.dtype,
theano/gpuarray/blas.py:        typecode = pygpu.gpuarray.dtype_to_typecode(dtype)
theano/gpuarray/blas.py:        dims[0] = PyGpuArray_DIMS(%(A)s)[0];
theano/gpuarray/blas.py:        dims[1] = PyGpuArray_DIMS(%(B)s)[1];
theano/gpuarray/blas.py:        if (pygpu_blas_rgemm(cb_no_trans, cb_no_trans,
theano/gpuarray/blas.py:gpu_dot22 = GpuDot22()
theano/gpuarray/blas.py:class GpuGemmBatch(BlasOp):
theano/gpuarray/blas.py:        A = as_gpuarray_variable(A, ctx_name)
theano/gpuarray/blas.py:        B = as_gpuarray_variable(B, ctx_name)
theano/gpuarray/blas.py:        C = as_gpuarray_variable(C, ctx_name)
theano/gpuarray/blas.py:        return super(GpuGemmBatch, self).c_headers() + ['<gpuarray/blas.h>']
theano/gpuarray/blas.py:                   if (!GpuArray_ISONESEGMENT(&%(C)s->ga)) {
theano/gpuarray/blas.py:        err = GpuArray_rgemmBatch_3d(
theano/gpuarray/blas.py:                         "%%s", GpuArray_error(&%(A)s->ga, err));
theano/gpuarray/blas.py:gpugemmbatch_no_inplace = GpuGemmBatch(inplace=False)
theano/gpuarray/blas.py:gpugemmbatch_inplace = GpuGemmBatch(inplace=True)
theano/gpuarray/blas.py:class BaseGpuCorrMM(CGpuKernelBase):
theano/gpuarray/blas.py:    Base class for `GpuCorrMM`, `GpuCorrMM_gradWeights` and
theano/gpuarray/blas.py:    `GpuCorrMM_gradInputs`. Cannot be used directly.
theano/gpuarray/blas.py:        CGpuKernelBase.__init__(self, ['c_code/corr_gemm.c'])
theano/gpuarray/blas.py:        return ["<gpuarray/array.h>", "<gpuarray/blas.h>", "gpuarray_helper.h"]
theano/gpuarray/blas.py:        return [gpuarray_helper_inc_dir()]
theano/gpuarray/blas.py:        This generates the C code for GpuCorrMM (direction="forward"),
theano/gpuarray/blas.py:        GpuCorrMM_gradWeights (direction="backprop weights"), and
theano/gpuarray/blas.py:        GpuCorrMM_gradInputs (direction="backprop inputs").
theano/gpuarray/blas.py:    PyGpuArrayObject * bottom = %(bottom)s;
theano/gpuarray/blas.py:    PyGpuArrayObject * weights = %(weights)s;
theano/gpuarray/blas.py:    PyGpuArrayObject * top = %(top)s;
theano/gpuarray/blas.py:    PyGpuArrayObject * out2 = NULL;
theano/gpuarray/blas.py:        kH = PyGpuArray_DIMS(weights)[wdim-2];
theano/gpuarray/blas.py:        kW = PyGpuArray_DIMS(weights)[wdim-1];
theano/gpuarray/blas.py:            kH = (2 - PyGpuArray_DIMS(bottom)[2] + (PyGpuArray_DIMS(top)[2] - 1) * dH - 1) / dilH + 1;
theano/gpuarray/blas.py:            kH = (PyGpuArray_DIMS(bottom)[2] + padH_l + padH_r - (PyGpuArray_DIMS(top)[2] - 1) * dH - 1) / dilH + 1 ;
theano/gpuarray/blas.py:            kW = (2 - PyGpuArray_DIMS(bottom)[3] + (PyGpuArray_DIMS(top)[3] - 1) * dW - 1) / dilW + 1;
theano/gpuarray/blas.py:            kW = (PyGpuArray_DIMS(bottom)[3] + padW_l + padW_r - (PyGpuArray_DIMS(top)[3] - 1) * dW - 1) / dilW + 1;
theano/gpuarray/blas.py:        PyErr_SetString(PyExc_ValueError, "BaseGpuCorrMM: padH must be >= -2");
theano/gpuarray/blas.py:        PyErr_SetString(PyExc_ValueError, "BaseGpuCorrMM: padW must be >= -2");
theano/gpuarray/blas.py:    PyGpuContextObject *out_context;
theano/gpuarray/blas.py:        out_dim[0] = PyGpuArray_DIMS(bottom)[0];
theano/gpuarray/blas.py:        out_dim[1] = PyGpuArray_DIMS(weights)[0];
theano/gpuarray/blas.py:        out_dim[2] = (PyGpuArray_DIMS(bottom)[2] + padH_l + padH_r - ((PyGpuArray_DIMS(weights)[wdim-2]-1)*dilH + 1)) / dH + 1;
theano/gpuarray/blas.py:        out_dim[3] = (PyGpuArray_DIMS(bottom)[3] + padW_l + padW_r - ((PyGpuArray_DIMS(weights)[wdim-1]-1)*dilW + 1)) / dW + 1;
theano/gpuarray/blas.py:                             "GpuCorrMM: impossible output shape\\n"
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(bottom)[0], PyGpuArray_DIMS(bottom)[1],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(bottom)[2], PyGpuArray_DIMS(bottom)[3],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(weights)[0], PyGpuArray_DIMS(weights)[1],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(weights)[2], PyGpuArray_DIMS(weights)[3],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(weights)[4], PyGpuArray_DIMS(weights)[5],
theano/gpuarray/blas.py:                             "GpuCorrMM: impossible output shape\\n"
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(bottom)[0], PyGpuArray_DIMS(bottom)[1],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(bottom)[2], PyGpuArray_DIMS(bottom)[3],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(weights)[0], PyGpuArray_DIMS(weights)[1],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(weights)[2], PyGpuArray_DIMS(weights)[3],
theano/gpuarray/blas.py:        out_dim[0] = PyGpuArray_DIMS(top)[1];
theano/gpuarray/blas.py:            out_dim[1] = PyGpuArray_DIMS(top)[2];
theano/gpuarray/blas.py:            out_dim[2] = PyGpuArray_DIMS(top)[3];
theano/gpuarray/blas.py:        out_dim[wdim-3] = PyGpuArray_DIMS(bottom)[1] / numgroups;
theano/gpuarray/blas.py:                             "GpuCorrMM backprop wrt. weights: impossible output shape\\n"
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(bottom)[0], PyGpuArray_DIMS(bottom)[1],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(bottom)[2], PyGpuArray_DIMS(bottom)[3],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3]);
theano/gpuarray/blas.py:                             "GpuCorrMM backprop wrt. weights: impossible output shape\\n"
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(bottom)[0], PyGpuArray_DIMS(bottom)[1],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(bottom)[2], PyGpuArray_DIMS(bottom)[3],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3]);
theano/gpuarray/blas.py:        out_dim[0] = PyGpuArray_DIMS(top)[0];
theano/gpuarray/blas.py:        out_dim[1] = PyGpuArray_DIMS(weights)[wdim-3] * numgroups;
theano/gpuarray/blas.py:        out_dim[2] = (%(height)s != -1) ? %(height)s : (PyGpuArray_DIMS(top)[2] - 1) * dH + (PyGpuArray_DIMS(weights)[wdim-2]-1)*dilH + 1 - padH_l - padH_r;
theano/gpuarray/blas.py:        out_dim[3] = (%(width)s != -1) ? %(width)s : (PyGpuArray_DIMS(top)[3] - 1) * dW + (PyGpuArray_DIMS(weights)[wdim-1]-1)*dilW + 1 - padW_l - padW_r;
theano/gpuarray/blas.py:                             "GpuCorrMM backprop wrt. inputs: impossible output shape\\n"
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(weights)[0], PyGpuArray_DIMS(weights)[1],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(weights)[2], PyGpuArray_DIMS(weights)[3],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(weights)[4], PyGpuArray_DIMS(weights)[5],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3]);
theano/gpuarray/blas.py:                             "GpuCorrMM backprop wrt. inputs: impossible output shape\\n"
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(weights)[0], PyGpuArray_DIMS(weights)[1],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(weights)[2], PyGpuArray_DIMS(weights)[3],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
theano/gpuarray/blas.py:                             PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3]);
theano/gpuarray/blas.py:        PyErr_SetString(PyExc_ValueError, "BaseGpuCorrMM: direction must be 0, 1, or 2\\n");
theano/gpuarray/blas.py:                    "BaseGpuCorrMM: Failed to allocate output of %%lld x %%lld x %%lld x %%lld",
theano/gpuarray/blas.py:                    "BaseGpuCorrMM: Failed to allocate output of %%lld x %%lld x %%lld x %%lld %%lld %%lld",
theano/gpuarray/blas.py:    if (!GpuArray_IS_C_CONTIGUOUS(&%(out)s->ga)) {
theano/gpuarray/blas.py:    // Call GPU code
theano/gpuarray/blas.py:class GpuCorrMM(BaseGpuCorrMM):
theano/gpuarray/blas.py:    GPU correlation implementation using Matrix Multiplication.
theano/gpuarray/blas.py:        `(sv, sh)` is equivalent to `GpuCorrMM(...)(...)[:,:,::sv, ::sh]`,
theano/gpuarray/blas.py:    C-contiguous. Use :func:`gpu_contiguous
theano/gpuarray/blas.py:    <theano.gpuarray.basic_ops.gpu_contiguous>` on these arguments
theano/gpuarray/blas.py:    to automatically replace all convolution operations with `GpuCorrMM`
theano/gpuarray/blas.py:    `GpuCorrMM(subsample=...)(image, filters)`. The latter is currently
theano/gpuarray/blas.py:        super(GpuCorrMM, self).__init__(border_mode, subsample,
theano/gpuarray/blas.py:        img = as_gpuarray_variable(img, ctx_name)
theano/gpuarray/blas.py:        kern = as_gpuarray_variable(kern, ctx_name)
theano/gpuarray/blas.py:        return Apply(self, [img, kern], [GpuArrayType(dtype=img.dtype,
theano/gpuarray/blas.py:        return super(GpuCorrMM, self).c_code_helper(bottom, weights, top, direction, sub)
theano/gpuarray/blas.py:        top = gpu_contiguous(top)
theano/gpuarray/blas.py:        d_bottom = GpuCorrMM_gradInputs(self.border_mode,
theano/gpuarray/blas.py:        d_weights = GpuCorrMM_gradWeights(self.border_mode,
theano/gpuarray/blas.py:class GpuCorrMM_gradWeights(BaseGpuCorrMM):
theano/gpuarray/blas.py:    Gradient wrt. filters for `GpuCorrMM`.
theano/gpuarray/blas.py:        super(GpuCorrMM_gradWeights, self).__init__(border_mode,
theano/gpuarray/blas.py:        img = as_gpuarray_variable(img, ctx_name)
theano/gpuarray/blas.py:        topgrad = as_gpuarray_variable(topgrad, ctx_name)
theano/gpuarray/blas.py:        return Apply(self, [img, topgrad] + height_width, [GpuArrayType(dtype=img.dtype,
theano/gpuarray/blas.py:        return super(GpuCorrMM_gradWeights, self).c_code_helper(bottom, weights, top, direction, sub, height, width)
theano/gpuarray/blas.py:        weights = gpu_contiguous(weights)
theano/gpuarray/blas.py:        d_bottom = GpuCorrMM_gradInputs(self.border_mode,
theano/gpuarray/blas.py:        d_top = GpuCorrMM(
theano/gpuarray/blas.py:class GpuCorrMM_gradInputs(BaseGpuCorrMM):
theano/gpuarray/blas.py:    Gradient wrt. inputs for `GpuCorrMM`.
theano/gpuarray/blas.py:        super(GpuCorrMM_gradInputs, self).__init__(border_mode, subsample,
theano/gpuarray/blas.py:        kern = as_gpuarray_variable(kern, ctx_name)
theano/gpuarray/blas.py:        topgrad = as_gpuarray_variable(topgrad, ctx_name)
theano/gpuarray/blas.py:        return Apply(self, [kern, topgrad] + height_width, [GpuArrayType(dtype=topgrad.dtype,
theano/gpuarray/blas.py:        return super(GpuCorrMM_gradInputs, self).c_code_helper(bottom, weights, top, direction, sub, height, width)
theano/gpuarray/blas.py:        bottom = gpu_contiguous(bottom)
theano/gpuarray/blas.py:        d_weights = GpuCorrMM_gradWeights(self.border_mode,
theano/gpuarray/blas.py:        d_top = GpuCorrMM(self.border_mode,
theano/gpuarray/blas.py:class BaseGpuCorr3dMM(CGpuKernelBase):
theano/gpuarray/blas.py:    Base class for `GpuCorr3dMM`, `GpuCorr3dMM_gradWeights` and
theano/gpuarray/blas.py:    `GpuCorr3dMM_gradInputs`. Cannot be used directly.
theano/gpuarray/blas.py:        CGpuKernelBase.__init__(self, ['c_code/corr3d_gemm.c'])
theano/gpuarray/blas.py:        return ["<gpuarray/array.h>", "<gpuarray/blas.h>", "gpuarray_helper.h"]
theano/gpuarray/blas.py:        return [gpuarray_helper_inc_dir()]
theano/gpuarray/blas.py:        This generates the C code for GpuCorr3dMM (direction="forward"),
theano/gpuarray/blas.py:        GpuCorr3dMM_gradWeights (direction="backprop weights"), and
theano/gpuarray/blas.py:        GpuCorr3dMM_gradInputs (direction="backprop inputs").
theano/gpuarray/blas.py:    PyGpuArrayObject * bottom = %(bottom)s;
theano/gpuarray/blas.py:    PyGpuArrayObject * weights = %(weights)s;
theano/gpuarray/blas.py:    PyGpuArrayObject * top = %(top)s;
theano/gpuarray/blas.py:    PyGpuArrayObject * out2 = NULL;
theano/gpuarray/blas.py:        kH = PyGpuArray_DIMS(weights)[2];
theano/gpuarray/blas.py:        kW = PyGpuArray_DIMS(weights)[3];
theano/gpuarray/blas.py:        kD = PyGpuArray_DIMS(weights)[4];
theano/gpuarray/blas.py:            kH = (2 - PyGpuArray_DIMS(bottom)[2] + (PyGpuArray_DIMS(top)[2] - 1) * dH - 1) / dilH + 1;
theano/gpuarray/blas.py:            kH = (PyGpuArray_DIMS(bottom)[2] + 2*padH - (PyGpuArray_DIMS(top)[2] - 1) * dH - 1) / dilH + 1 ;
theano/gpuarray/blas.py:            kW = (2 - PyGpuArray_DIMS(bottom)[3] + (PyGpuArray_DIMS(top)[3] - 1) * dW - 1) / dilW + 1;
theano/gpuarray/blas.py:            kW = (PyGpuArray_DIMS(bottom)[3] + 2*padW - (PyGpuArray_DIMS(top)[3] - 1) * dW - 1) / dilW + 1;
theano/gpuarray/blas.py:            kD = (2 - PyGpuArray_DIMS(bottom)[4] + (PyGpuArray_DIMS(top)[4] - 1) * dD - 1) / dilD + 1;
theano/gpuarray/blas.py:            kD = (PyGpuArray_DIMS(bottom)[4] + 2*padD - (PyGpuArray_DIMS(top)[4] - 1) * dD - 1) / dilD + 1;
theano/gpuarray/blas.py:        PyErr_SetString(PyExc_ValueError, "BaseGpuCorr3dMM: padH must be >= -2");
theano/gpuarray/blas.py:        PyErr_SetString(PyExc_ValueError, "BaseGpuCorr3dMM: padW must be >= -2");
theano/gpuarray/blas.py:        PyErr_SetString(PyExc_ValueError, "BaseGpuCorr3dMM: padD must be >= -2");
theano/gpuarray/blas.py:    PyGpuContextObject *out_context;
theano/gpuarray/blas.py:        out_dim[0] = PyGpuArray_DIMS(bottom)[0];
theano/gpuarray/blas.py:        out_dim[1] = PyGpuArray_DIMS(weights)[0];
theano/gpuarray/blas.py:        out_dim[2] = (PyGpuArray_DIMS(bottom)[2] + 2*padH - ((PyGpuArray_DIMS(weights)[2]-1)*dilH + 1)) / dH + 1;
theano/gpuarray/blas.py:        out_dim[3] = (PyGpuArray_DIMS(bottom)[3] + 2*padW - ((PyGpuArray_DIMS(weights)[3]-1)*dilW + 1)) / dW + 1;
theano/gpuarray/blas.py:        out_dim[4] = (PyGpuArray_DIMS(bottom)[4] + 2*padD - ((PyGpuArray_DIMS(weights)[4]-1)*dilD + 1)) / dD + 1;
theano/gpuarray/blas.py:                         "GpuCorr3dMM: impossible output shape\\n"
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(bottom)[0], PyGpuArray_DIMS(bottom)[1],
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(bottom)[2], PyGpuArray_DIMS(bottom)[3],
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(bottom)[4],
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(weights)[0], PyGpuArray_DIMS(weights)[1],
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(weights)[2], PyGpuArray_DIMS(weights)[3],
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(weights)[4],
theano/gpuarray/blas.py:        out_dim[0] = PyGpuArray_DIMS(top)[1];
theano/gpuarray/blas.py:        out_dim[1] = PyGpuArray_DIMS(bottom)[1] / numgroups;
theano/gpuarray/blas.py:                         "GpuCorr3dMM backprop wrt. weights: impossible output shape\\n"
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(bottom)[0], PyGpuArray_DIMS(bottom)[1],
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(bottom)[2], PyGpuArray_DIMS(bottom)[3],
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(bottom)[4],
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3],
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(top)[4]);
theano/gpuarray/blas.py:        out_dim[0] = PyGpuArray_DIMS(top)[0];
theano/gpuarray/blas.py:        out_dim[1] = PyGpuArray_DIMS(weights)[1] * numgroups;
theano/gpuarray/blas.py:        out_dim[2] = (%(height)s != -1) ? %(height)s : (PyGpuArray_DIMS(top)[2] - 1) * dH + (PyGpuArray_DIMS(weights)[2]-1)*dilH + 1 - 2*padH;
theano/gpuarray/blas.py:        out_dim[3] = (%(width)s != -1) ? %(width)s : (PyGpuArray_DIMS(top)[3] - 1) * dW + (PyGpuArray_DIMS(weights)[3]-1)*dilW + 1 - 2*padW;
theano/gpuarray/blas.py:        out_dim[4] = (%(depth)s != -1) ? %(depth)s : (PyGpuArray_DIMS(top)[4] - 1) * dD + (PyGpuArray_DIMS(weights)[4]-1)*dilD + 1 - 2*padD;
theano/gpuarray/blas.py:                         "GpuCorr3dMM backprop wrt. inputs: impossible output shape\\n"
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(weights)[0], PyGpuArray_DIMS(weights)[1],
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(weights)[2], PyGpuArray_DIMS(weights)[3],
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(weights)[4],
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(top)[0], PyGpuArray_DIMS(top)[1],
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(top)[2], PyGpuArray_DIMS(top)[3],
theano/gpuarray/blas.py:                         PyGpuArray_DIMS(top)[4]);
theano/gpuarray/blas.py:        PyErr_SetString(PyExc_ValueError, "BaseGpuCorr3dMM: direction must be 0, 1, or 2\\n");
theano/gpuarray/blas.py:                "BaseGpuCorrMM: Failed to allocate output of %%lld x %%lld x %%lld x %%lld x %%lld",
theano/gpuarray/blas.py:    if (!GpuArray_IS_C_CONTIGUOUS(&%(out)s->ga)) {
theano/gpuarray/blas.py:    // Call GPU code
theano/gpuarray/blas.py:class GpuCorr3dMM(BaseGpuCorr3dMM):
theano/gpuarray/blas.py:    GPU correlation implementation using Matrix Multiplication.
theano/gpuarray/blas.py:        `GpuCorrMM(...)(...)[:,:,::sv, ::sh, ::sl]`, but faster.
theano/gpuarray/blas.py:    C-contiguous. Use :func:`gpu_contiguous
theano/gpuarray/blas.py:    <theano.gpuarray.basic_ops.gpu_contiguous>` on these arguments
theano/gpuarray/blas.py:    to automatically replace all convolution operations with `GpuCorr3dMM`
theano/gpuarray/blas.py:    `GpuCorr3dMM(subsample=...)(image, filters)`. The latter is currently
theano/gpuarray/blas.py:        super(GpuCorr3dMM, self).__init__(border_mode, subsample,
theano/gpuarray/blas.py:        img = as_gpuarray_variable(img, ctx_name)
theano/gpuarray/blas.py:        kern = as_gpuarray_variable(kern, ctx_name)
theano/gpuarray/blas.py:        return Apply(self, [img, kern], [GpuArrayType(dtype=img.dtype,
theano/gpuarray/blas.py:        return super(GpuCorr3dMM, self).c_code_helper(bottom, weights, top, direction, sub)
theano/gpuarray/blas.py:        top = gpu_contiguous(top)
theano/gpuarray/blas.py:        d_bottom = GpuCorr3dMM_gradInputs(self.border_mode,
theano/gpuarray/blas.py:        d_weights = GpuCorr3dMM_gradWeights(self.border_mode,
theano/gpuarray/blas.py:class GpuCorr3dMM_gradWeights(BaseGpuCorr3dMM):
theano/gpuarray/blas.py:    Gradient wrt. filters for `GpuCorr3dMM`.
theano/gpuarray/blas.py:        super(GpuCorr3dMM_gradWeights, self).__init__(border_mode,
theano/gpuarray/blas.py:        img = as_gpuarray_variable(img, ctx_name)
theano/gpuarray/blas.py:        topgrad = as_gpuarray_variable(topgrad, ctx_name)
theano/gpuarray/blas.py:                     [GpuArrayType(dtype=img.dtype,
theano/gpuarray/blas.py:        return super(GpuCorr3dMM_gradWeights, self).c_code_helper(bottom, weights, top, direction, sub, height, width, depth)
theano/gpuarray/blas.py:        weights = gpu_contiguous(weights)
theano/gpuarray/blas.py:        d_bottom = GpuCorr3dMM_gradInputs(self.border_mode,
theano/gpuarray/blas.py:        d_top = GpuCorr3dMM(
theano/gpuarray/blas.py:class GpuCorr3dMM_gradInputs(BaseGpuCorr3dMM):
theano/gpuarray/blas.py:    Gradient wrt. inputs for `GpuCorr3dMM`.
theano/gpuarray/blas.py:        super(GpuCorr3dMM_gradInputs, self).__init__(border_mode, subsample,
theano/gpuarray/blas.py:        kern = as_gpuarray_variable(kern, ctx_name)
theano/gpuarray/blas.py:        topgrad = as_gpuarray_variable(topgrad, ctx_name)
theano/gpuarray/blas.py:                     [GpuArrayType(dtype=topgrad.dtype,
theano/gpuarray/blas.py:        return super(GpuCorr3dMM_gradInputs, self).c_code_helper(bottom, weights, top, direction, sub, height, width, depth)
theano/gpuarray/blas.py:        bottom = gpu_contiguous(bottom)
theano/gpuarray/blas.py:        d_weights = GpuCorr3dMM_gradWeights(self.border_mode,
theano/gpuarray/blas.py:        d_top = GpuCorr3dMM(self.border_mode,
theano/gpuarray/blas.py:@inplace_allocempty(GpuGemv, 0)
theano/gpuarray/blas.py:def local_inplace_gpuagemv(node, inputs):
theano/gpuarray/blas.py:    return [gpugemv_inplace(*inputs)]
theano/gpuarray/blas.py:@inplace_allocempty(GpuGemm, 0)
theano/gpuarray/blas.py:def local_inplace_gpuagemm(node, inputs):
theano/gpuarray/blas.py:    return [gpugemm_inplace(*inputs)]
theano/gpuarray/blas.py:@inplace_allocempty(GpuGer, 0)
theano/gpuarray/blas.py:def local_inplace_gpuager(node, inputs):
theano/gpuarray/blas.py:    return [gpuger_inplace(*inputs)]
theano/gpuarray/blas.py:@inplace_allocempty(GpuGemmBatch, 0)
theano/gpuarray/blas.py:def local_inplace_gpuagemmbatch(node, inputs):
theano/gpuarray/blas.py:    return [gpugemmbatch_inplace(*inputs)]
theano/gpuarray/blas.py:gpuablas_opt_inplace = in2out(LocalOptGroup(local_inplace_gpuagemv,
theano/gpuarray/blas.py:                                            local_inplace_gpuagemm,
theano/gpuarray/blas.py:                                            local_inplace_gpuager,
theano/gpuarray/blas.py:                                            local_inplace_gpuagemmbatch),
theano/gpuarray/blas.py:                              name='gpuablas_opt_inplace')
theano/gpuarray/blas.py:optdb.register('InplaceGpuaBlasOpt',
theano/gpuarray/blas.py:               gpuablas_opt_inplace,
theano/gpuarray/blas.py:               70.0, 'fast_run', 'inplace', 'gpuarray')
theano/gpuarray/dnn.py:from . import pygpu, cudnn_defs
theano/gpuarray/dnn.py:from .type import (get_context, gpu_context_type, list_contexts,
theano/gpuarray/dnn.py:                   GpuArraySharedVariable)
theano/gpuarray/dnn.py:from .basic_ops import (as_gpuarray_variable, infer_context_name, gpuarray_helper_inc_dir,
theano/gpuarray/dnn.py:                        gpu_contiguous, GpuAllocEmpty,
theano/gpuarray/dnn.py:                        empty_like, GpuArrayType, HostFromGpu)
theano/gpuarray/dnn.py:from .elemwise import GpuElemwise, GpuCAReduceCuda
theano/gpuarray/dnn.py:from .reduction import GpuMaxAndArgmax
theano/gpuarray/dnn.py:# These don't exist in gpuarray
theano/gpuarray/dnn.py:# GpuDownsampleFactorMax, GpuDownsampleFactorMaxGrad
theano/gpuarray/dnn.py:from .nnet import GpuSoftmax
theano/gpuarray/dnn.py:from .opt import (gpu_seqopt, register_opt, pool_db, pool_db2,
theano/gpuarray/dnn.py:    from pygpu import gpuarray
theano/gpuarray/dnn.py:    params.extend(['-I%s%s%s' % (path_wrapper, gpuarray_helper_inc_dir(), path_wrapper)])
theano/gpuarray/dnn.py:    if config.cuda.include_path:
theano/gpuarray/dnn.py:        params.extend(['-I%s%s%s' % (path_wrapper, config.cuda.include_path, path_wrapper)])
theano/gpuarray/dnn.py:    # default gpu, not the one selected by the user. If mixed
theano/gpuarray/dnn.py:    # GPU are installed or if the GPUs are configured in
theano/gpuarray/dnn.py:    if pygpu is None:
theano/gpuarray/dnn.py:        dnn_present.msg = "PyGPU not available"
theano/gpuarray/dnn.py:    if not ctx.kind == b'cuda':
theano/gpuarray/dnn.py:        dnn_available.msg = "Not on a CUDA device."
theano/gpuarray/dnn.py:    # "<something>_<major><minor>" for cuda devices.
theano/gpuarray/dnn.py:                                  config.cuda.include_path],
theano/gpuarray/dnn.py:        return [config.dnn.include_path, config.cuda.include_path]
theano/gpuarray/dnn.py:        return ['gpuarray/types.h', 'gpuarray/array.h', 'gpuarray/kernel.h',
theano/gpuarray/dnn.py:                'gpuarray/util.h', 'gpuarray/ext_cuda.h', 'gpuarray_api.h',
theano/gpuarray/dnn.py:                'gpuarray_helper.h']
theano/gpuarray/dnn.py:        return [gpuarray_helper_inc_dir(), pygpu.get_include(),
theano/gpuarray/dnn.py:                config.dnn.include_path, config.cuda.include_path]
theano/gpuarray/dnn.py:        return ['cudnn', 'gpuarray']
theano/gpuarray/dnn.py:class GpuDnnConvDesc(COp):
theano/gpuarray/dnn.py:        return [gpuarray_helper_inc_dir(), config.dnn.include_path,
theano/gpuarray/dnn.py:                config.cuda.include_path]
theano/gpuarray/dnn.py:        return (super(GpuDnnConvDesc, self).c_code_cache_version(), version())
theano/gpuarray/dnn.py:class GpuDnnConv(DnnBase):
theano/gpuarray/dnn.py:        img = as_gpuarray_variable(img, ctx_name)
theano/gpuarray/dnn.py:        kern = as_gpuarray_variable(kern, ctx_name)
theano/gpuarray/dnn.py:        output = as_gpuarray_variable(output, ctx_name)
theano/gpuarray/dnn.py:        top = gpu_contiguous(top)
theano/gpuarray/dnn.py:        d_img = GpuDnnConvGradI(num_groups=self.num_groups)(kerns, top, empty_like(img), desc)
theano/gpuarray/dnn.py:        d_kerns = GpuDnnConvGradW(num_groups=self.num_groups)(img, top, empty_like(kerns), desc)
theano/gpuarray/dnn.py:class GpuDnnConvGradW(DnnBase):
theano/gpuarray/dnn.py:        kerns = gpu_contiguous(kerns)
theano/gpuarray/dnn.py:        d_img = GpuDnnConvGradI(num_groups=self.num_groups)(kerns, top, empty_like(img), desc)
theano/gpuarray/dnn.py:        d_top = GpuDnnConv(num_groups=self.num_groups)(img, kerns, empty_like(top), desc)
theano/gpuarray/dnn.py:                          'with certain cuDNN algorithms depending on the compute capability of your GPU '
theano/gpuarray/dnn.py:                          'with certain cuDNN algorithms depending on the compute capability of your GPU '
theano/gpuarray/dnn.py:        img = as_gpuarray_variable(img, ctx_name)
theano/gpuarray/dnn.py:        topgrad = as_gpuarray_variable(topgrad, ctx_name)
theano/gpuarray/dnn.py:        output = as_gpuarray_variable(output, ctx_name)
theano/gpuarray/dnn.py:class GpuDnnConvGradI(DnnBase):
theano/gpuarray/dnn.py:        img = gpu_contiguous(img)
theano/gpuarray/dnn.py:        d_kerns = GpuDnnConvGradW(num_groups=self.num_groups)(img, top, empty_like(kerns), desc)
theano/gpuarray/dnn.py:        d_top = GpuDnnConv(num_groups=self.num_groups)(img, kerns, empty_like(top), desc)
theano/gpuarray/dnn.py:        kern = as_gpuarray_variable(kern, ctx_name)
theano/gpuarray/dnn.py:        topgrad = as_gpuarray_variable(topgrad, ctx_name)
theano/gpuarray/dnn.py:        output = as_gpuarray_variable(output, ctx_name)
theano/gpuarray/dnn.py:    img = as_gpuarray_variable(img, ctx_name)
theano/gpuarray/dnn.py:    kerns = as_gpuarray_variable(kerns, ctx_name)
theano/gpuarray/dnn.py:    img = gpu_contiguous(img.astype(dt))
theano/gpuarray/dnn.py:    kerns = gpu_contiguous(kerns.astype(dt))
theano/gpuarray/dnn.py:    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, dilation=dilation,
theano/gpuarray/dnn.py:        real_out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*out_shp)
theano/gpuarray/dnn.py:        out = gpu_contiguous(as_gpuarray_variable(out, ctx_name))
theano/gpuarray/dnn.py:        check = Assert('GpuDnnConv: given output (for beta not null) does not have expected shape')
theano/gpuarray/dnn.py:    return GpuDnnConv(algo=algo, num_groups=num_groups)(img, kerns, real_out, desc, alpha, beta)
theano/gpuarray/dnn.py:    img = as_gpuarray_variable(img, ctx_name)
theano/gpuarray/dnn.py:    topgrad = as_gpuarray_variable(topgrad, ctx_name)
theano/gpuarray/dnn.py:    img = gpu_contiguous(img.astype(dt))
theano/gpuarray/dnn.py:    topgrad = gpu_contiguous(topgrad.astype(dt))
theano/gpuarray/dnn.py:    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, dilation=dilation,
theano/gpuarray/dnn.py:        real_out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*kerns_shp)
theano/gpuarray/dnn.py:        out = gpu_contiguous(as_gpuarray_variable(out, ctx_name))
theano/gpuarray/dnn.py:        check = Assert('GpuDnnConvGradW: given output (for beta not null) does not have expected shape')
theano/gpuarray/dnn.py:    return GpuDnnConvGradW(algo=algo, num_groups=num_groups)(img, topgrad, real_out, desc, alpha, beta)
theano/gpuarray/dnn.py:    kerns = as_gpuarray_variable(kerns, ctx_name)
theano/gpuarray/dnn.py:    topgrad = as_gpuarray_variable(topgrad, ctx_name)
theano/gpuarray/dnn.py:    kerns = gpu_contiguous(kerns.astype(dt))
theano/gpuarray/dnn.py:    topgrad = gpu_contiguous(topgrad.astype(dt))
theano/gpuarray/dnn.py:    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, dilation=dilation,
theano/gpuarray/dnn.py:        real_out = GpuAllocEmpty(dtype=kerns.dtype, context_name=ctx_name)(*img_shp)
theano/gpuarray/dnn.py:        out = gpu_contiguous(as_gpuarray_variable(out, ctx_name))
theano/gpuarray/dnn.py:        check = Assert('GpuDnnConvGradI: given output (for beta not null) does not have expected shape')
theano/gpuarray/dnn.py:    return GpuDnnConvGradI(algo=algo, num_groups=num_groups)(kerns, topgrad, real_out, desc, alpha, beta)
theano/gpuarray/dnn.py:    GPU convolution using cuDNN from NVIDIA.
theano/gpuarray/dnn.py:        By default, GpuDnnConv will be used to carry out the convolution.
theano/gpuarray/dnn.py:        direction_hint is 'bprop weights', it will use GpuDnnConvGradW.
theano/gpuarray/dnn.py:        direction_hint is *not* 'forward!', it will use GpuDnnConvGradI.
theano/gpuarray/dnn.py:    .. warning:: The cuDNN library only works with GPUs that have a compute
theano/gpuarray/dnn.py:        capability of 3.0 or higher. This means that older GPUs will not
theano/gpuarray/dnn.py:        # Special case: We are asked to use GpuDnnConvGradW. We need to set
theano/gpuarray/dnn.py:        img = gpu_contiguous(img.dimshuffle(1, 0, 2, 3))
theano/gpuarray/dnn.py:            # that would be flipped by conv_mode='conv' in GpuDnnConvGradW.
theano/gpuarray/dnn.py:        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3))
theano/gpuarray/dnn.py:        out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*out_shp)
theano/gpuarray/dnn.py:        desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1), dilation=(1, 1),
theano/gpuarray/dnn.py:        conv = GpuDnnConvGradW(num_groups=num_groups)(img, kerns, out, desc)
theano/gpuarray/dnn.py:        return as_gpuarray_variable(conv.dimshuffle(1, 0, 2, 3), ctx_name)
theano/gpuarray/dnn.py:        # Special case: We can be faster by using GpuDnnConvGradI to compute
theano/gpuarray/dnn.py:        img = gpu_contiguous(img)  # cudnn v2 rc3 need contiguous data
theano/gpuarray/dnn.py:        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3))
theano/gpuarray/dnn.py:        out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*out_shp)
theano/gpuarray/dnn.py:        desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1), dilation=dilation,
theano/gpuarray/dnn.py:        return GpuDnnConvGradI(num_groups=num_groups)(kerns, img, out, desc)
theano/gpuarray/dnn.py:    # Standard case: We use GpuDnnConv with suitable padding.
theano/gpuarray/dnn.py:    GPU convolution using cuDNN from NVIDIA.
theano/gpuarray/dnn.py:        By default, GpuDnnConv will be used to carry out the convolution.
theano/gpuarray/dnn.py:        GpuDnnConvGradW.
theano/gpuarray/dnn.py:        GpuDnnConvGradI.
theano/gpuarray/dnn.py:    .. warning:: The cuDNN library only works with GPUs that have a compute
theano/gpuarray/dnn.py:        capability of 3.0 or higher. This means that older GPUs will not
theano/gpuarray/dnn.py:        # Special case: We are asked to use GpuDnnConvGradW. We need to set
theano/gpuarray/dnn.py:        img = gpu_contiguous(img.dimshuffle(1, 0, 2, 3, 4))
theano/gpuarray/dnn.py:            # that would be flipped by conv_mode='conv' in GpuDnnConvGradW.
theano/gpuarray/dnn.py:        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3, 4))
theano/gpuarray/dnn.py:        out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*out_shp)
theano/gpuarray/dnn.py:        desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1, 1), dilation=(1, 1, 1),
theano/gpuarray/dnn.py:        conv = GpuDnnConvGradW(num_groups=num_groups)(img, kerns, out, desc)
theano/gpuarray/dnn.py:        return as_gpuarray_variable(conv.dimshuffle(1, 0, 2, 3, 4), ctx_name)
theano/gpuarray/dnn.py:        # Special case: We can be faster by using GpuDnnConvGradI to compute
theano/gpuarray/dnn.py:        img = gpu_contiguous(img)  # cudnn v2 rc3 need contiguous data
theano/gpuarray/dnn.py:        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3, 4))
theano/gpuarray/dnn.py:        out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*out_shp)
theano/gpuarray/dnn.py:        desc = GpuDnnConvDesc(border_mode='valid', subsample=(1, 1, 1), dilation=dilation,
theano/gpuarray/dnn.py:        return GpuDnnConvGradI(num_groups=num_groups)(kerns, img, out, desc)
theano/gpuarray/dnn.py:    # Standard case: We use GpuDnnConv with suitable padding.
theano/gpuarray/dnn.py:class GpuDnnPoolDesc(Op):
theano/gpuarray/dnn.py:        return [gpuarray_helper_inc_dir(), config.dnn.include_path]
theano/gpuarray/dnn.py:class GpuDnnPoolBase(DnnBase):
theano/gpuarray/dnn.py:    Abstract base class for GpuDnnPool and GpuDnnPoolGrad.
theano/gpuarray/dnn.py:class GpuDnnPool(GpuDnnPoolBase):
theano/gpuarray/dnn.py:        img = as_gpuarray_variable(img, ctx_name)
theano/gpuarray/dnn.py:        grad = gpu_contiguous(grad)
theano/gpuarray/dnn.py:        g_out = GpuDnnPoolGrad(mode=self.mode)(img, out, grad, ws, stride, pad)
theano/gpuarray/dnn.py:class GpuDnnPoolGrad(GpuDnnPoolBase):
theano/gpuarray/dnn.py:        inp = as_gpuarray_variable(inp, ctx_name)
theano/gpuarray/dnn.py:        out_grad = as_gpuarray_variable(out_grad, ctx_name)
theano/gpuarray/dnn.py:        out = as_gpuarray_variable(out, ctx_name)
theano/gpuarray/dnn.py:    GPU pooling using cuDNN from NVIDIA.
theano/gpuarray/dnn.py:    .. warning:: The cuDNN library only works with GPU that have a compute
theano/gpuarray/dnn.py:        capability of 3.0 or higher.  This means that older GPU will not
theano/gpuarray/dnn.py:    img = gpu_contiguous(img)
theano/gpuarray/dnn.py:        ret = GpuDnnPool(mode="average_inc_pad")(img, ws, stride, pad)
theano/gpuarray/dnn.py:        return as_gpuarray_variable(ret * window_elem, context_name)
theano/gpuarray/dnn.py:    return GpuDnnPool(mode=mode)(img, ws, stride, pad)
theano/gpuarray/dnn.py:class GpuDnnSoftmaxBase(DnnBase):
theano/gpuarray/dnn.py:class GpuDnnSoftmax(GpuDnnSoftmaxBase):
theano/gpuarray/dnn.py:        x = as_gpuarray_variable(x, infer_context_name(x))
theano/gpuarray/dnn.py:        return [GpuDnnSoftmaxGrad(
theano/gpuarray/dnn.py:class GpuDnnSoftmaxGrad(GpuDnnSoftmaxBase):
theano/gpuarray/dnn.py:        dy = as_gpuarray_variable(dy, ctx_name)
theano/gpuarray/dnn.py:        sm = as_gpuarray_variable(sm, ctx_name)
theano/gpuarray/dnn.py:class GpuDnnReduction(DnnBase):
theano/gpuarray/dnn.py:        inp = as_gpuarray_variable(inp, ctx_name)
theano/gpuarray/dnn.py:        inp = gpu_contiguous(inp)
theano/gpuarray/dnn.py:            outs.append(GpuArrayType(dtype='uint32', broadcastable=bcast,
theano/gpuarray/dnn.py:class GpuDnnBatchNorm(DnnBase):
theano/gpuarray/dnn.py:        x = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/dnn.py:        scale = as_gpuarray_variable(scale, ctx_name)
theano/gpuarray/dnn.py:        bias = as_gpuarray_variable(bias, ctx_name)
theano/gpuarray/dnn.py:            inputs.append(as_gpuarray_variable(running_mean, ctx_name))
theano/gpuarray/dnn.py:            inputs.append(as_gpuarray_variable(running_var, ctx_name))
theano/gpuarray/dnn.py:        return GpuDnnBatchNormGrad(self.mode)(
theano/gpuarray/dnn.py:class GpuDnnBatchNormInference(DnnBase):
theano/gpuarray/dnn.py:        x = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/dnn.py:        scale = as_gpuarray_variable(scale, ctx_name)
theano/gpuarray/dnn.py:        bias = as_gpuarray_variable(bias, ctx_name)
theano/gpuarray/dnn.py:        estimated_mean = as_gpuarray_variable(estimated_mean, ctx_name)
theano/gpuarray/dnn.py:        estimated_variance = as_gpuarray_variable(estimated_variance, ctx_name)
theano/gpuarray/dnn.py:class GpuDnnBatchNormGrad(DnnBase):
theano/gpuarray/dnn.py:        x = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/dnn.py:        dy = as_gpuarray_variable(dy, ctx_name)
theano/gpuarray/dnn.py:        scale = as_gpuarray_variable(scale, ctx_name)
theano/gpuarray/dnn.py:        x_mean = as_gpuarray_variable(x_mean, ctx_name)
theano/gpuarray/dnn.py:        x_invstd = as_gpuarray_variable(x_invstd, ctx_name)
theano/gpuarray/dnn.py:gpudata_type = CDataType('gpudata *', 'gpudata_release')
theano/gpuarray/dnn.py:class GpuDnnDropoutOp(DnnBase):
theano/gpuarray/dnn.py:        inp = as_gpuarray_variable(inp, ctx_name)
theano/gpuarray/dnn.py:                     [inp.type(), state.type(), gpudata_type()])
theano/gpuarray/dnn.py:        assert self.inplace, "GpuDnnDropoutOp not inplace"
theano/gpuarray/dnn.py:        context = gpu_context_type.make_constant(get_context(context_name))
theano/gpuarray/dnn.py:                      GpuArrayType('uint8', (False,),
theano/gpuarray/dnn.py:    y, odesc = GpuDnnDropoutOp()(x, desc)
theano/gpuarray/dnn.py:        dtype = as_i32(gpuarray.dtype_to_typecode(dtype))
theano/gpuarray/dnn.py:    typecode = gpuarray.dtype_to_typecode(dtype)
theano/gpuarray/dnn.py:        w = as_gpuarray_variable(w, infer_context_name(w))
theano/gpuarray/dnn.py:        _1d = GpuArrayType(w.type.dtype, [False],
theano/gpuarray/dnn.py:        _2d = GpuArrayType(w.type.dtype, [False, False],
theano/gpuarray/dnn.py:  w = PyGpuArray_DEV_DATA(%(w)s);
theano/gpuarray/dnn.py:  nshp[0] = PyGpuArray_DIM(%(w)s, 0);
theano/gpuarray/dnn.py:  %(b)s = pygpu_view(%(w)s, Py_None);
theano/gpuarray/dnn.py:  %(b)s = pygpu_view(%(w)s, Py_None);
theano/gpuarray/dnn.py:  GpuArray_fix_flags(&%(b)s->ga);
theano/gpuarray/dnn.py:  %(m)s = pygpu_reshape(%(w)s, 2, nshp, GA_F_ORDER, 1, -1);
theano/gpuarray/dnn.py:  %(m)s = pygpu_reshape(%(w)s, 2, nshp, GA_F_ORDER, 1, -1);
theano/gpuarray/dnn.py:  %(m)s->ga.strides[1] = %(m)s->ga.dimensions[0] * gpuarray_get_elsize(%(m)s->ga.typecode);
theano/gpuarray/dnn.py:  GpuArray_fix_flags(&%(m)s->ga);
theano/gpuarray/dnn.py:    typecode = gpuarray.dtype_to_typecode(dtype)
theano/gpuarray/dnn.py:class GpuDnnRNNOp(DnnBase):
theano/gpuarray/dnn.py:        w = as_gpuarray_variable(w, context_name)
theano/gpuarray/dnn.py:        x = as_gpuarray_variable(x, context_name)
theano/gpuarray/dnn.py:        hx = as_gpuarray_variable(hx, context_name)
theano/gpuarray/dnn.py:            cx = as_gpuarray_variable(cx, context_name)
theano/gpuarray/dnn.py:        _3d = GpuArrayType(dtype=x.dtype, broadcastable=(False, False, False),
theano/gpuarray/dnn.py:        reserve = gpudata_type()
theano/gpuarray/dnn.py:            dy = as_gpuarray_variable(y.zeros_like(),
theano/gpuarray/dnn.py:        dinputs = GpuDnnRNNGradInputs(rnn_mode=self.rnn_mode,
theano/gpuarray/dnn.py:        dw = GpuDnnRNNGradWeights()(
theano/gpuarray/dnn.py:class GpuDnnRNNGradInputs(DnnBase):
theano/gpuarray/dnn.py:class GpuDnnRNNGradWeights(DnnBase):
theano/gpuarray/dnn.py:    theano/gpuarray/tests/test_dnn.py for now.
theano/gpuarray/dnn.py:        w: GpuArraySharedVariable
theano/gpuarray/dnn.py:        if not isinstance(w, GpuArraySharedVariable):
theano/gpuarray/dnn.py:            raise TypeError("split_params only works on gpuarray shared variables")
theano/gpuarray/dnn.py:        return GpuDnnRNNOp(self.rnn_mode, self.direction_mode)(
theano/gpuarray/dnn.py:    batchnorm_op = GpuDnnBatchNorm(mode=mode, running_averages=running_averages)
theano/gpuarray/dnn.py:            gpu_contiguous(inputs), gpu_contiguous(gamma),
theano/gpuarray/dnn.py:            gpu_contiguous(beta), epsilon=epsilon,
theano/gpuarray/dnn.py:            running_mean=gpu_contiguous(running_mean),
theano/gpuarray/dnn.py:            running_var=gpu_contiguous(running_var))
theano/gpuarray/dnn.py:        result = batchnorm_op(gpu_contiguous(inputs), gpu_contiguous(gamma),
theano/gpuarray/dnn.py:                              gpu_contiguous(beta), epsilon=epsilon)
theano/gpuarray/dnn.py:    batchnorm_op = GpuDnnBatchNormInference(mode=mode)
theano/gpuarray/dnn.py:    result = batchnorm_op(gpu_contiguous(inputs), gpu_contiguous(gamma),
theano/gpuarray/dnn.py:                          gpu_contiguous(beta), gpu_contiguous(mean),
theano/gpuarray/dnn.py:                          gpu_contiguous(var), epsilon=epsilon)
theano/gpuarray/dnn.py:class GpuDnnTransformerGrid(DnnBase):
theano/gpuarray/dnn.py:        theta = gpu_contiguous(as_gpuarray_variable(theta, context_name))
theano/gpuarray/dnn.py:        grid = GpuArrayType(dtype=theta.dtype,
theano/gpuarray/dnn.py:        dtheta = GpuDnnTransformerGradT()(dgrid)
theano/gpuarray/dnn.py:class GpuDnnTransformerSampler(DnnBase):
theano/gpuarray/dnn.py:        grid : GpuDnnTransformerGrid
theano/gpuarray/dnn.py:        img = gpu_contiguous(as_gpuarray_variable(img, context_name))
theano/gpuarray/dnn.py:        grid = gpu_contiguous(as_gpuarray_variable(grid, context_name))
theano/gpuarray/dnn.py:        out = GpuArrayType(dtype=img.dtype,
theano/gpuarray/dnn.py:        dimg, dgrid = GpuDnnTransformerGradI()(img, grid, dy)
theano/gpuarray/dnn.py:class GpuDnnTransformerGradI(DnnBase):
theano/gpuarray/dnn.py:        img = as_gpuarray_variable(gpu_contiguous(img), context_name)
theano/gpuarray/dnn.py:        grid = as_gpuarray_variable(gpu_contiguous(grid), context_name)
theano/gpuarray/dnn.py:        dy = as_gpuarray_variable(dy, context_name)
theano/gpuarray/dnn.py:class GpuDnnTransformerGradT(DnnBase):
theano/gpuarray/dnn.py:        dgrid = as_gpuarray_variable(dgrid, context_name)
theano/gpuarray/dnn.py:        dtheta = GpuArrayType(dtype=dgrid.dtype,
theano/gpuarray/dnn.py:    GPU spatial transformer using cuDNN from NVIDIA.
theano/gpuarray/dnn.py:    grid = GpuDnnTransformerGrid()(theta, out_dims)
theano/gpuarray/dnn.py:    sampler = GpuDnnTransformerSampler()(img, grid)
theano/gpuarray/dnn.py:    if not isinstance(node.inputs[0].type, GpuArrayType):
theano/gpuarray/dnn.py:            img = gpu_contiguous(inp1)
theano/gpuarray/dnn.py:            topgrad = gpu_contiguous(inp2)
theano/gpuarray/dnn.py:            img = gpu_contiguous(img.dimshuffle(1, 0, 2, 3))
theano/gpuarray/dnn.py:            topgrad = gpu_contiguous(topgrad.dimshuffle(1, 0, 2, 3))
theano/gpuarray/dnn.py:            out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*out_shp)
theano/gpuarray/dnn.py:            desc = GpuDnnConvDesc(border_mode=border_mode,
theano/gpuarray/dnn.py:            conv = GpuDnnConv(algo=None, num_groups=num_groups)(img, topgrad, out, desc)
theano/gpuarray/dnn.py:            rval = as_gpuarray_variable(conv.dimshuffle(1, 0, 2, 3), ctx_name)
theano/gpuarray/dnn.py:            kerns = gpu_contiguous(inp1.dimshuffle(1, 0, 2, 3))
theano/gpuarray/dnn.py:            topgrad = gpu_contiguous(inp2)
theano/gpuarray/dnn.py:            desc = GpuDnnConvDesc(border_mode='full',
theano/gpuarray/dnn.py:            out = GpuAllocEmpty(dtype=topgrad.dtype, context_name=ctx_name)(*shape)
theano/gpuarray/dnn.py:            rval = GpuDnnConv(algo=None, num_groups=num_groups)(topgrad, kerns, out, desc)
theano/gpuarray/dnn.py:            img = gpu_contiguous(inp1)
theano/gpuarray/dnn.py:            topgrad = gpu_contiguous(inp2)
theano/gpuarray/dnn.py:            img = gpu_contiguous(img.dimshuffle(1, 0, 2, 3, 4))
theano/gpuarray/dnn.py:            topgrad = gpu_contiguous(topgrad.dimshuffle(1, 0, 2, 3, 4))
theano/gpuarray/dnn.py:            out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*out_shp)
theano/gpuarray/dnn.py:            desc = GpuDnnConvDesc(border_mode=border_mode,
theano/gpuarray/dnn.py:            conv = GpuDnnConv(algo=None, num_groups=num_groups)(
theano/gpuarray/dnn.py:            rval = as_gpuarray_variable(conv.dimshuffle(1, 0, 2, 3, 4), ctx_name)
theano/gpuarray/dnn.py:            kerns = gpu_contiguous(inp1.dimshuffle(1, 0, 2, 3, 4))
theano/gpuarray/dnn.py:            topgrad = gpu_contiguous(inp2)
theano/gpuarray/dnn.py:            desc = GpuDnnConvDesc(border_mode='full',
theano/gpuarray/dnn.py:            out = GpuAllocEmpty(dtype=topgrad.dtype, context_name=ctx_name)(*shape)
theano/gpuarray/dnn.py:            rval = GpuDnnConv(algo=None, num_groups=num_groups)(
theano/gpuarray/dnn.py:    if not isinstance(node.inputs[0].type, GpuArrayType):
theano/gpuarray/dnn.py:    if not isinstance(node.inputs[0].type, GpuArrayType):
theano/gpuarray/dnn.py:@inplace_allocempty(GpuDnnConv, 2)
theano/gpuarray/dnn.py:    return [GpuDnnConv(algo=node.op.algo, inplace=True, num_groups=node.op.num_groups)(*inputs)]
theano/gpuarray/dnn.py:@inplace_allocempty(GpuDnnConvGradW, 2)
theano/gpuarray/dnn.py:    return [GpuDnnConvGradW(algo=node.op.algo, inplace=True, num_groups=node.op.num_groups)(*inputs)]
theano/gpuarray/dnn.py:@inplace_allocempty(GpuDnnConvGradI, 2)
theano/gpuarray/dnn.py:    return [GpuDnnConvGradI(algo=node.op.algo, inplace=True, num_groups=node.op.num_groups)(*inputs)]
theano/gpuarray/dnn.py:               70.0, 'fast_run', 'inplace', 'gpuarray', 'cudnn')
theano/gpuarray/dnn.py:@alpha_merge(GpuDnnConv, alpha_in=4, beta_in=5)
theano/gpuarray/dnn.py:    return [GpuDnnConv(algo=node.op.algo, num_groups=node.op.num_groups)(*inputs)]
theano/gpuarray/dnn.py:@alpha_merge(GpuDnnConvGradW, alpha_in=4, beta_in=5)
theano/gpuarray/dnn.py:    return [GpuDnnConvGradW(algo=node.op.algo, num_groups=node.op.num_groups)(*inputs)]
theano/gpuarray/dnn.py:@alpha_merge(GpuDnnConvGradI, alpha_in=4, beta_in=5)
theano/gpuarray/dnn.py:    return [GpuDnnConvGradI(algo=node.op.algo, num_groups=node.op.num_groups)(*inputs)]
theano/gpuarray/dnn.py:@output_merge(GpuDnnConv, alpha_in=4, beta_in=5, out_in=2)
theano/gpuarray/dnn.py:    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
theano/gpuarray/dnn.py:    return [GpuDnnConv(algo=node.op.algo, num_groups=node.op.num_groups)(*inputs)]
theano/gpuarray/dnn.py:@output_merge(GpuDnnConvGradW, alpha_in=4, beta_in=5, out_in=2)
theano/gpuarray/dnn.py:    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
theano/gpuarray/dnn.py:    return [GpuDnnConvGradW(algo=node.op.algo, num_groups=node.op.num_groups)(*inputs)]
theano/gpuarray/dnn.py:@output_merge(GpuDnnConvGradI, alpha_in=4, beta_in=5, out_in=2)
theano/gpuarray/dnn.py:    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
theano/gpuarray/dnn.py:    return [GpuDnnConvGradI(algo=node.op.algo, num_groups=node.op.num_groups)(*inputs)]
theano/gpuarray/dnn.py:def local_gpua_pool_dnn_alternative(op, ctx_name, inputs, outputs):
theano/gpuarray/dnn.py:    img = gpu_contiguous(as_gpuarray_variable(img, ctx_name))
theano/gpuarray/dnn.py:pool_db.register("local_gpua_pool_dnn_alternative",
theano/gpuarray/dnn.py:                 op_lifter([Pool])(local_gpua_pool_dnn_alternative),
theano/gpuarray/dnn.py:                 'gpuarray', 'fast_compile', 'fast_run', 'cudnn',
theano/gpuarray/dnn.py:pool_db2.register("local_gpua_pool_dnn_alternative",
theano/gpuarray/dnn.py:                  local_optimizer([Pool])(local_gpua_pool_dnn_alternative),
theano/gpuarray/dnn.py:                  'gpuarray', 'fast_compile', 'fast_run', 'cudnn',
theano/gpuarray/dnn.py:def local_gpua_pool_dnn_grad_stride(op, ctx_name, inputs, outputs):
theano/gpuarray/dnn.py:    inp = gpu_contiguous(as_gpuarray_variable(inp, ctx_name))
theano/gpuarray/dnn.py:    out = gpu_contiguous(as_gpuarray_variable(out, ctx_name))
theano/gpuarray/dnn.py:    out_grad = gpu_contiguous(as_gpuarray_variable(out_grad, ctx_name))
theano/gpuarray/dnn.py:    # the GPU ops expect exactly 2 non-pooling dimensions
theano/gpuarray/dnn.py:        return GpuDnnPoolGrad(mode=mode)(inp,
theano/gpuarray/dnn.py:        ret_padded = GpuDnnPoolGrad(mode=mode)(inp_padded,
theano/gpuarray/dnn.py:pool_db.register("local_gpua_pool_dnn_grad_stride",
theano/gpuarray/dnn.py:                 op_lifter([MaxPoolGrad])(local_gpua_pool_dnn_grad_stride),
theano/gpuarray/dnn.py:                 'gpuarray', 'fast_compile', 'fast_run', 'cudnn',
theano/gpuarray/dnn.py:pool_db2.register("local_gpua_pool_dnn_grad_stride",
theano/gpuarray/dnn.py:                  local_optimizer([MaxPoolGrad])(local_gpua_pool_dnn_grad_stride),
theano/gpuarray/dnn.py:                  'gpuarray', 'fast_compile', 'fast_run', 'cudnn',
theano/gpuarray/dnn.py:def local_gpua_avg_pool_dnn_grad_stride(op, ctx_name, inputs, outputs):
theano/gpuarray/dnn.py:    inp = gpu_contiguous(as_gpuarray_variable(inp, ctx_name))
theano/gpuarray/dnn.py:    out_grad = gpu_contiguous(as_gpuarray_variable(out_grad, ctx_name))
theano/gpuarray/dnn.py:    # the GPU ops expect exactly 2 non-pooling dimensions
theano/gpuarray/dnn.py:        return GpuDnnPoolGrad(mode=mode)(inp, out_grad, out_grad, ws, stride, pad)
theano/gpuarray/dnn.py:        ret_padded = GpuDnnPoolGrad(mode=mode)(inp_padded,
theano/gpuarray/dnn.py:pool_db.register("local_gpua_avg_pool_dnn_grad_stride",
theano/gpuarray/dnn.py:                 op_lifter([AveragePoolGrad])(local_gpua_avg_pool_dnn_grad_stride),
theano/gpuarray/dnn.py:                 'gpuarray', 'fast_compile', 'fast_run', 'cudnn',
theano/gpuarray/dnn.py:pool_db2.register("local_gpua_avg_pool_dnn_grad_stride",
theano/gpuarray/dnn.py:                  local_optimizer([AveragePoolGrad])(local_gpua_avg_pool_dnn_grad_stride),
theano/gpuarray/dnn.py:                  'gpuarray', 'fast_compile', 'fast_run', 'cudnn',
theano/gpuarray/dnn.py:@local_optimizer([GpuSoftmax])
theano/gpuarray/dnn.py:    if isinstance(node.op, GpuSoftmax):
theano/gpuarray/dnn.py:        ins = gpu_contiguous(ins)
theano/gpuarray/dnn.py:        out = GpuDnnSoftmax('accurate', 'channel')(ins)
theano/gpuarray/dnn.py:        out = as_gpuarray_variable(out.dimshuffle(0, 1), out.type.context_name)
theano/gpuarray/dnn.py:@local_optimizer([GpuElemwise])
theano/gpuarray/dnn.py:    # This looks for GpuDnnSoftmax so we know that we have cudnn.
theano/gpuarray/dnn.py:    if (isinstance(node.op, GpuElemwise) and
theano/gpuarray/dnn.py:            isinstance(node.inputs[0].owner.op, GpuDnnSoftmax) and
theano/gpuarray/dnn.py:        new_softmax = GpuDnnSoftmax('log', softmax_node.op.mode)
theano/gpuarray/dnn.py:def local_gpua_logsoftmax_to_dnn(op, ctx_name, inputs, outputs):
theano/gpuarray/dnn.py:    # Transform the input in the format expected by GpuDnnSoftmax
theano/gpuarray/dnn.py:    # Apply GpuDnnSoftmax and return the result
theano/gpuarray/dnn.py:    out = GpuDnnSoftmax('log', 'channel')(gpu_contiguous(inp))
theano/gpuarray/dnn.py:def local_gpua_softmax_dnn_grad(op, ctx_name, inputs, outputs):
theano/gpuarray/dnn.py:        n = as_gpuarray_variable(n, ctx_name)
theano/gpuarray/dnn.py:    out = GpuDnnSoftmaxGrad('accurate', 'instance')(
theano/gpuarray/dnn.py:        gpu_contiguous(ins[0]), gpu_contiguous(ins[1]))
theano/gpuarray/dnn.py:@local_optimizer([GpuCAReduceCuda])
theano/gpuarray/dnn.py:    if not isinstance(node.op, GpuCAReduceCuda):
theano/gpuarray/dnn.py:        return GpuElemwise(theano.scalar.basic.sqr)(a)
theano/gpuarray/dnn.py:        ret = GpuDnnReduction(scal,
theano/gpuarray/dnn.py:@local_optimizer([GpuMaxAndArgmax])
theano/gpuarray/dnn.py:    if not isinstance(node.op, GpuMaxAndArgmax):
theano/gpuarray/dnn.py:    max, arg = GpuDnnReduction('maximum', node.op.axis, node.outputs[0].dtype,
theano/gpuarray/dnn.py:    return (max, as_gpuarray_variable(arg.astype('int64'),
theano/gpuarray/dnn.py:    max, arg = GpuDnnReduction('maximum', op.axis, inputs[0].dtype,
theano/gpuarray/dnn.py:    return [as_gpuarray_variable(arg.astype('int64'), ctx_name)]
theano/gpuarray/dnn.py:gpu_seqopt.register("NoCuDNNRaise", NoCuDNNRaise(), 0, 'cudnn')
theano/gpuarray/dnn.py:    x = as_gpuarray_variable(x, context_name=ctx)
theano/gpuarray/dnn.py:    scale = as_gpuarray_variable(scale, context_name=ctx)
theano/gpuarray/dnn.py:    bias = as_gpuarray_variable(bias, context_name=ctx)
theano/gpuarray/dnn.py:@local_optimizer([GpuDnnBatchNorm], inplace=True)
theano/gpuarray/dnn.py:    if isinstance(node.op, GpuDnnBatchNorm) and not node.op.inplace_output:
theano/gpuarray/dnn.py:        return GpuDnnBatchNorm(mode=node.op.mode,
theano/gpuarray/dnn.py:@local_optimizer([GpuDnnBatchNorm], inplace=True)
theano/gpuarray/dnn.py:    if isinstance(node.op, GpuDnnBatchNorm) and node.op.running_averages and not node.op.inplace_running_mean:
theano/gpuarray/dnn.py:        return GpuDnnBatchNorm(mode=node.op.mode,
theano/gpuarray/dnn.py:@local_optimizer([GpuDnnBatchNorm], inplace=True)
theano/gpuarray/dnn.py:    if isinstance(node.op, GpuDnnBatchNorm) and node.op.running_averages and not node.op.inplace_running_var:
theano/gpuarray/dnn.py:        return GpuDnnBatchNorm(mode=node.op.mode,
theano/gpuarray/dnn.py:@local_optimizer([GpuDnnBatchNormInference], inplace=True)
theano/gpuarray/dnn.py:    if isinstance(node.op, GpuDnnBatchNormInference) and not node.op.inplace:
theano/gpuarray/dnn.py:        return [GpuDnnBatchNormInference(mode=node.op.mode, inplace=True)(*node.inputs)]
theano/gpuarray/dnn.py:    # input on gpu?  TODO what about the output?
theano/gpuarray/dnn.py:    x_on_gpu = (isinstance(x.type, GpuArrayType) or
theano/gpuarray/dnn.py:                (x.owner and isinstance(x.owner.op, HostFromGpu)))
theano/gpuarray/dnn.py:    dy_on_gpu = (isinstance(dy.type, GpuArrayType) or
theano/gpuarray/dnn.py:                 (dy.owner and isinstance(dy.owner.op, HostFromGpu)))
theano/gpuarray/dnn.py:    if not (x_on_gpu or dy_on_gpu):
theano/gpuarray/dnn.py:    x = as_gpuarray_variable(x, context_name=ctx)
theano/gpuarray/dnn.py:    dy = as_gpuarray_variable(dy, context_name=ctx)
theano/gpuarray/dnn.py:    scale = as_gpuarray_variable(scale, context_name=ctx)
theano/gpuarray/dnn.py:    x_mean = as_gpuarray_variable(x_mean, context_name=ctx)
theano/gpuarray/dnn.py:    x_invstd = as_gpuarray_variable(x_invstd, context_name=ctx)
theano/gpuarray/dnn.py:        GpuDnnBatchNormGrad(mode)(x, dy, scale, x_mean, x_invstd, eps)
theano/gpuarray/dnn.py:    x = as_gpuarray_variable(x, context_name=ctx)
theano/gpuarray/dnn.py:    scale = as_gpuarray_variable(scale, context_name=ctx)
theano/gpuarray/dnn.py:    bias = as_gpuarray_variable(bias, context_name=ctx)
theano/gpuarray/dnn.py:    estimated_mean = as_gpuarray_variable(estimated_mean, context_name=ctx)
theano/gpuarray/dnn.py:    estimated_variance = as_gpuarray_variable(estimated_variance, context_name=ctx)
theano/gpuarray/sort.py:from .basic_ops import (GpuKernelBase, Kernel, infer_context_name,
theano/gpuarray/sort.py:                        as_gpuarray_variable, gpuarray_helper_inc_dir)
theano/gpuarray/sort.py:from .type import GpuArrayType
theano/gpuarray/sort.py:    import pygpu
theano/gpuarray/sort.py:    import pygpu.gpuarray as ga
theano/gpuarray/sort.py:# TODO GPU sort / argsort
theano/gpuarray/sort.py:class GpuTopKOp(GpuKernelBase, TopKOp):
theano/gpuarray/sort.py:    '''Implements TopKOp on gpu
theano/gpuarray/sort.py:                "GpuTopK currently is not sure to give sorted output even if they look sorted..")
theano/gpuarray/sort.py:        GpuKernelBase.__init__(self)
theano/gpuarray/sort.py:        return ['gpuarray_api.h', 'gpuarray_helper.h', 'numpy_compat.h']
theano/gpuarray/sort.py:            gpuarray_helper_inc_dir(),
theano/gpuarray/sort.py:            pygpu.get_include()]
theano/gpuarray/sort.py:    def gpu_kernels(self, node, nodename):
theano/gpuarray/sort.py:        kernel_ext = {b'cuda': '.cu', b'opencl': '.cl'}[device_type]
theano/gpuarray/sort.py:        common_ext = {b'cuda': '.cuh', b'opencl': '.h'}[device_type]
theano/gpuarray/sort.py:        if device_type == b'cuda':
theano/gpuarray/sort.py:        elif device_type == b'opencl':
theano/gpuarray/sort.py:            param_types.append(ga.GpuArray)  # dst*
theano/gpuarray/sort.py:        param_types.append(ga.GpuArray)  # src
theano/gpuarray/sort.py:        if context.kind != b'cuda':
theano/gpuarray/sort.py:                '%s: We only have CUDA '
theano/gpuarray/sort.py:            def_dvstrides = 'const ssize_t *dvstrides = PyGpuArray_STRIDES(%s)' % yv
theano/gpuarray/sort.py:            def_distrides = 'const ssize_t *distrides = PyGpuArray_STRIDES(%s)' % yi
theano/gpuarray/sort.py:    const size_t *dims = PyGpuArray_DIMS(%(x)s);
theano/gpuarray/sort.py:    const ssize_t *sstrides = PyGpuArray_STRIDES(%(x)s);
theano/gpuarray/sort.py:            "topk: gpu kernel failed to execute");
theano/gpuarray/sort.py:        inp = as_gpuarray_variable(inp, ctx_name)
theano/gpuarray/sort.py:            outs.append(GpuArrayType(
theano/gpuarray/sort.py:@op_lifter([TopKOp], cuda_only=True)
theano/gpuarray/sort.py:def local_gpua_topkop(op, ctx_name, inputs, outputs):
theano/gpuarray/sort.py:    x = as_gpuarray_variable(x, ctx_name)
theano/gpuarray/sort.py:    gpu_op = GpuTopKOp(
theano/gpuarray/sort.py:    rets = gpu_op(x, k, return_list=True)
theano/gpuarray/linalg.py:from theano.gpuarray import GpuArrayType
theano/gpuarray/linalg.py:from .basic_ops import (CGpuKernelBase, as_gpuarray_variable, gpu_contiguous, gpuarray_helper_inc_dir,
theano/gpuarray/linalg.py:from .type import gpu_context_type
theano/gpuarray/linalg.py:    import pygpu
theano/gpuarray/linalg.py:    from pygpu.basic import triu, tril
theano/gpuarray/linalg.py:    pygpu_available = True
theano/gpuarray/linalg.py:    pygpu_available = False
theano/gpuarray/linalg.py:    import skcuda
theano/gpuarray/linalg.py:    from skcuda import cusolver
theano/gpuarray/linalg.py:    from skcuda import cublas
theano/gpuarray/linalg.py:    # Add cusolver call as it is missing in skcuda
theano/gpuarray/linalg.py:        `cusolverDn<t>potrs <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrs>`_
theano/gpuarray/linalg.py:class GpuCusolverSolve(Op):
theano/gpuarray/linalg.py:    CUSOLVER GPU solver OP.
theano/gpuarray/linalg.py:        super(GpuCusolverSolve, self).__init__()
theano/gpuarray/linalg.py:                               'GpuCusolverSolve Op can not be constructed.')
theano/gpuarray/linalg.py:        if skcuda.__version__ <= '0.5.1':
theano/gpuarray/linalg.py:            warnings.warn('The GpuSolve op requires scikit-cuda > 0.5.1 to work with CUDA 8')
theano/gpuarray/linalg.py:        inp1 = as_gpuarray_variable(inp1, context_name)
theano/gpuarray/linalg.py:        inp2 = as_gpuarray_variable(inp2, context_name)
theano/gpuarray/linalg.py:        inp1 = gpu_contiguous(inp1)
theano/gpuarray/linalg.py:        inp2 = gpu_contiguous(inp2)
theano/gpuarray/linalg.py:            [GpuArrayType(inp1.dtype,
theano/gpuarray/linalg.py:        b = pygpu.array(b, copy=True, order='F')
theano/gpuarray/linalg.py:            A = pygpu.array(A, copy=True)
theano/gpuarray/linalg.py:        A_ptr = A.gpudata
theano/gpuarray/linalg.py:        b_ptr = b.gpudata
theano/gpuarray/linalg.py:            workspace = pygpu.zeros(workspace_size, dtype=A.dtype,
theano/gpuarray/linalg.py:            dev_info = pygpu.zeros((1,), dtype='int32', context=context)
theano/gpuarray/linalg.py:            workspace_ptr = workspace.gpudata
theano/gpuarray/linalg.py:            dev_info_ptr = dev_info.gpudata
theano/gpuarray/linalg.py:            workspace = pygpu.zeros(workspace_size, dtype=A.dtype,
theano/gpuarray/linalg.py:            pivots = pygpu.zeros(n, dtype='int32', context=context)
theano/gpuarray/linalg.py:            dev_info = pygpu.zeros((1,), dtype='int32', context=context)
theano/gpuarray/linalg.py:            workspace_ptr = workspace.gpudata
theano/gpuarray/linalg.py:            pivots_ptr = pivots.gpudata
theano/gpuarray/linalg.py:            dev_info_ptr = dev_info.gpudata
theano/gpuarray/linalg.py:        # FIXME: triangular structure would use GpuCublasTriangularsolve?
theano/gpuarray/linalg.py:        trans_solve_op = GpuCusolverSolve('general')
theano/gpuarray/linalg.py:class GpuCublasTriangularSolve(Op):
theano/gpuarray/linalg.py:    CUBLAS GPU Triangular Solve Op.
theano/gpuarray/linalg.py:        super(GpuCublasTriangularSolve, self).__init__()
theano/gpuarray/linalg.py:                               'GpuCublasTriangularSolve Op '
theano/gpuarray/linalg.py:        inp1 = as_gpuarray_variable(inp1, context_name)
theano/gpuarray/linalg.py:        inp2 = as_gpuarray_variable(inp2, context_name)
theano/gpuarray/linalg.py:        inp1 = gpu_contiguous(inp1)
theano/gpuarray/linalg.py:        inp2 = gpu_contiguous(inp2)
theano/gpuarray/linalg.py:                            [GpuArrayType(inp1.dtype,
theano/gpuarray/linalg.py:        b = pygpu.array(b, copy=True, order='F')
theano/gpuarray/linalg.py:        A_ptr = A.gpudata
theano/gpuarray/linalg.py:        b_ptr = b.gpudata
theano/gpuarray/linalg.py:        trans_solve_op = GpuCublasTriangularSolve(not self.lower)
theano/gpuarray/linalg.py:def gpu_solve(A, b, A_structure='general', trans='N'):
theano/gpuarray/linalg.py:        return GpuCublasTriangularSolve(True, trans)(A, b)
theano/gpuarray/linalg.py:        return GpuCublasTriangularSolve(False, trans)(A, b)
theano/gpuarray/linalg.py:    return GpuCusolverSolve(A_structure, trans)(A, b)
theano/gpuarray/linalg.py:def gpu_solve_lower_triangular(A, b, trans='N'):
theano/gpuarray/linalg.py:    return GpuCublasTriangularSolve(True, trans)(A, b)
theano/gpuarray/linalg.py:def gpu_solve_upper_triangular(A, b, trans='N'):
theano/gpuarray/linalg.py:    return GpuCublasTriangularSolve(False, trans)(A, b)
theano/gpuarray/linalg.py:class GpuCholesky(Op):
theano/gpuarray/linalg.py:    CUSOLVER GPU Cholesky Op.
theano/gpuarray/linalg.py:        super(GpuCholesky, self).__init__()
theano/gpuarray/linalg.py:                               'GpuCholesky Op can not be constructed.')
theano/gpuarray/linalg.py:        if skcuda.__version__ <= '0.5.1':
theano/gpuarray/linalg.py:            warnings.warn('The GpuCholesky op requires scikit-cuda > '
theano/gpuarray/linalg.py:                          '0.5.1 to work with CUDA 8')
theano/gpuarray/linalg.py:        if not pygpu_available:
theano/gpuarray/linalg.py:            raise RuntimeError('Missing pygpu or triu/tril functions.'
theano/gpuarray/linalg.py:                               'Install or update libgpuarray.')
theano/gpuarray/linalg.py:        inp = as_gpuarray_variable(inp, context_name)
theano/gpuarray/linalg.py:        inp = gpu_contiguous(inp)
theano/gpuarray/linalg.py:            L = pygpu.array(A, copy=True)
theano/gpuarray/linalg.py:        L_ptr = L.gpudata
theano/gpuarray/linalg.py:            workspace = pygpu.zeros(workspace_size, dtype=A.dtype,
theano/gpuarray/linalg.py:            dev_info = pygpu.zeros((1,), dtype='int32', context=context)
theano/gpuarray/linalg.py:            workspace_ptr = workspace.gpudata
theano/gpuarray/linalg.py:            dev_info_ptr = dev_info.gpudata
theano/gpuarray/linalg.py:            return gpu_solve_upper_triangular(
theano/gpuarray/linalg.py:                outer.T, gpu_solve_upper_triangular(outer.T, inner.T).T)
theano/gpuarray/linalg.py:def gpu_cholesky(A, lower=True):
theano/gpuarray/linalg.py:    return GpuCholesky(lower)(A)
theano/gpuarray/linalg.py:class GpuMagmaBase(COp):
theano/gpuarray/linalg.py:        return ['gpuarray/types.h', 'gpuarray/array.h', 'gpuarray/ext_cuda.h',
theano/gpuarray/linalg.py:                'gpuarray_helper.h', 'magma.h']
theano/gpuarray/linalg.py:        dirs = [gpuarray_helper_inc_dir(), pygpu.get_include(),
theano/gpuarray/linalg.py:                config.cuda.include_path]
theano/gpuarray/linalg.py:        from skcuda.magma import magma_init
theano/gpuarray/linalg.py:class GpuMagmaSVD(GpuMagmaBase):
theano/gpuarray/linalg.py:        in order ``S, U, VT``. Use :func:`theano.gpuarray.linalg.gpu_svd`
theano/gpuarray/linalg.py:    params_type = ParamsType(full_matrices=bool_t, context=gpu_context_type)
theano/gpuarray/linalg.py:        A = as_gpuarray_variable(A, ctx_name)
theano/gpuarray/linalg.py:        A = gpu_contiguous(A)
theano/gpuarray/linalg.py:                                [GpuArrayType(A.dtype, broadcastable=[False],
theano/gpuarray/linalg.py:                                [GpuArrayType(A.dtype, broadcastable=[False],
theano/gpuarray/linalg.py:        super(GpuMagmaSVD, self).prepare_node(node, storage_map, compute_map, impl)
theano/gpuarray/linalg.py:                "Due to implementation constraints, GpuMagmaSVD interface has changed and now returns (S, U, VT) " \
theano/gpuarray/linalg.py:                "instead of (U, S, VT). Either update your code, or use gpu_svd() to get the expected (U, S, VT) order."
theano/gpuarray/linalg.py:def gpu_svd(a, full_matrices=1, compute_uv=1):
theano/gpuarray/linalg.py:    This function performs the SVD on GPU.
theano/gpuarray/linalg.py:    out = GpuMagmaSVD(full_matrices, compute_uv)(a)
theano/gpuarray/linalg.py:class GpuMagmaMatrixInverse(GpuMagmaBase):
theano/gpuarray/linalg.py:    params_type = ParamsType(inplace=bool_t, context=gpu_context_type)
theano/gpuarray/linalg.py:        A = as_gpuarray_variable(A, ctx_name)
theano/gpuarray/linalg.py:        A = gpu_contiguous(A)
theano/gpuarray/linalg.py:def gpu_matrix_inverse(a):
theano/gpuarray/linalg.py:    This function performs the matrix inverse on GPU.
theano/gpuarray/linalg.py:    return GpuMagmaMatrixInverse()(a)
theano/gpuarray/linalg.py:class GpuMagmaCholesky(GpuMagmaBase, CGpuKernelBase):
theano/gpuarray/linalg.py:    params_type = ParamsType(lower=bool_t, inplace=bool_t, context=gpu_context_type)
theano/gpuarray/linalg.py:        A = as_gpuarray_variable(A, ctx_name)
theano/gpuarray/linalg.py:        A = gpu_contiguous(A)
theano/gpuarray/linalg.py:class GpuMagmaQR(GpuMagmaBase, CGpuKernelBase):
theano/gpuarray/linalg.py:        in order ``R, Q``. Use :func:`theano.gpuarray.linalg.gpu_qr`
theano/gpuarray/linalg.py:    params_type = ParamsType(complete=bool_t, context=gpu_context_type)
theano/gpuarray/linalg.py:        A = as_gpuarray_variable(A, ctx_name)
theano/gpuarray/linalg.py:        A = gpu_contiguous(A)
theano/gpuarray/linalg.py:def gpu_qr(a, complete=True):
theano/gpuarray/linalg.py:    This function performs the QR on GPU.
theano/gpuarray/linalg.py:    out = GpuMagmaQR(complete)(a)
theano/gpuarray/linalg.py:class GpuMagmaEigh(GpuMagmaBase):
theano/gpuarray/linalg.py:                             context=gpu_context_type)
theano/gpuarray/linalg.py:        A = as_gpuarray_variable(A, ctx_name)
theano/gpuarray/linalg.py:        A = gpu_contiguous(A)
theano/gpuarray/linalg.py:                                [GpuArrayType(A.dtype, broadcastable=[False],
theano/gpuarray/linalg.py:                                [GpuArrayType(A.dtype, broadcastable=[False],
theano/__init__.py:# This need to be before the init of GPU, as it add config variable
theano/__init__.py:if (config.device.startswith('cuda') or
theano/__init__.py:        config.device.startswith('opencl') or
theano/__init__.py:        config.init_gpu_device.startswith('cuda') or
theano/__init__.py:        config.init_gpu_device.startswith('opencl') or
theano/__init__.py:    import theano.gpuarray
theano/sandbox/tests/test_multinomial.py:    # multinomial() call in GPU random generation.
theano/sandbox/tests/test_multinomial.py:    # and also make sure that the GPU version doesn't screw up the
theano/sandbox/tests/test_multinomial.py:# TODO: check a bigger example (make sure blocking on GPU is handled correctly)
theano/sandbox/rng_mrg.py:            # The limit is on the C and GPU code. This perform don't
theano/sandbox/rng_mrg.py:        rstate = np.asarray(rstate)  # bring state from GPU if necessary
theano/sandbox/rng_mrg.py:        # send to GPU if necessary
theano/sandbox/rng_mrg.py:        # TensorType, something is wrong (likely one of the GPU ops
theano/sandbox/rng_mrg.py:        # the GPU to its full capacity. It just wastes RAM and
theano/sandbox/rng_mrg.py:        # for the GPU.
theano/sandbox/rng_mrg.py:        #      Better would be to use pycuda to query the number of
theano/sandbox/rng_mrg.py:        #      processors on the GPU device,
theano/sandbox/rng_mrg.py:        # op might be gpu version
theano/sandbox/cuda/__init__.py:    "You are importing theano.sandbox.cuda. This is the old GPU back-end and "
theano/sandbox/cuda/__init__.py:    "transition to the new GPU back-end! See "
theano/sandbox/cuda/__init__.py:    "https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29")
theano/sandbox/fourier.py:           "automatic optimization transfers to the GPU ops.")
NEWS_DEV.txt:GPU:
Theano.pyproj:    <Compile Include="theano\misc\cudamat_utils.py" />
Theano.pyproj:    <Compile Include="theano\misc\latence_gpu_transfert.py" />
Theano.pyproj:    <Compile Include="theano\misc\pycuda_example.py" />
Theano.pyproj:    <Compile Include="theano\misc\pycuda_init.py" />
Theano.pyproj:    <Compile Include="theano\misc\pycuda_utils.py" />
Theano.pyproj:    <Compile Include="theano\misc\tests\test_cudamat_utils.py" />
Theano.pyproj:    <Compile Include="theano\misc\tests\test_pycuda_example.py" />
Theano.pyproj:    <Compile Include="theano\misc\tests\test_pycuda_theano_simple.py" />
Theano.pyproj:    <Compile Include="theano\misc\tests\test_pycuda_utils.py" />
Theano.pyproj:    <Compile Include="theano\sandbox\cuda\basic_ops.py" />
Theano.pyproj:    <Compile Include="theano\sandbox\cuda\blas.py" />
Theano.pyproj:    <Compile Include="theano\sandbox\cuda\elemwise.py" />
Theano.pyproj:    <Compile Include="theano\sandbox\cuda\GpuConv3D.py" />
Theano.pyproj:    <Compile Include="theano\sandbox\cuda\GpuConvGrad3D.py" />
Theano.pyproj:    <Compile Include="theano\sandbox\cuda\GpuConvTransp3D.py" />
Theano.pyproj:    <Compile Include="theano\sandbox\cuda\kernel_codegen.py" />
Theano.pyproj:    <Compile Include="theano\sandbox\cuda\nnet.py" />
Theano.pyproj:    <Compile Include="theano\sandbox\cuda\nvcc_compiler.py" />
Theano.pyproj:    <Compile Include="theano\sandbox\cuda\opt.py" />
Theano.pyproj:    <Compile Include="theano\sandbox\cuda\rng_curand.py" />
Theano.pyproj:    <Compile Include="theano\sandbox\cuda\type.py" />
Theano.pyproj:    <Compile Include="theano\sandbox\cuda\var.py" />
Theano.pyproj:    <Compile Include="theano\sandbox\cuda\__init__.py" />
Theano.pyproj:    <Folder Include="theano\sandbox\cuda\" />
Theano.pyproj:    <Content Include="theano\sandbox\cuda\conv.cu" />
Theano.pyproj:    <Content Include="theano\sandbox\cuda\conv_full_kernel.cu" />
Theano.pyproj:    <Content Include="theano\sandbox\cuda\conv_kernel.cu" />
Theano.pyproj:    <Content Include="theano\sandbox\cuda\cuda_ndarray.cu" />
Theano.pyproj:    <Content Include="theano\sandbox\cuda\cuda_ndarray.cuh" />
doc/install_ubuntu.txt:.. _gpu_linux:
doc/install_ubuntu.txt:For Ubuntu 16.04 with cuda 7.5
doc/install_ubuntu.txt:    # cuda 7.5 don't support the default g++ version. Install an supported version and make it the default.
doc/install_windows.txt:    * ``git`` package installs git source control through conda, which is required for the development versions of Theano and libgpuarray
doc/install_windows.txt:.. _gpu_windows:
doc/install_windows.txt:Install and configure the GPU drivers (recommended)
doc/install_windows.txt:    OpenCL support is still minimal for now.
doc/install_windows.txt:Install CUDA drivers
doc/install_windows.txt:Follow `this link <https://developer.nvidia.com/cuda-downloads>`__
doc/install_windows.txt:to install the CUDA driver and the CUDA Toolkit.
doc/install_windows.txt:.. Installation of Theano and libgpuarray.
doc/install_windows.txt:    * Install CUDA with the same instructions as above.
doc/install_windows.txt:    * Install the latest, development version of libgpuarray following the
doc/install_windows.txt:      `Step-by-step instructions <http://deeplearning.net/software/libgpuarray/installation.html#step-by-step-install>`__.
doc/install.txt:Once your setup is complete and if you installed the GPU libraries, head to :ref:`testing_the_gpu` to find how to verify
doc/cifarSC2011/gpundarray.txt:.. _cifar2013_gpundarray:
doc/cifarSC2011/gpundarray.txt:GpuNdArray
doc/cifarSC2011/gpundarray.txt:Why a common GPU ndarray?
doc/cifarSC2011/gpundarray.txt:- Currently there are at least 4 different GPU array data structures in use by Python packages
doc/cifarSC2011/gpundarray.txt:  - CudaNdarray (Theano), GPUArray (PyCUDA), CUDAMatrix (cudamat), GPUArray (PyOpenCL), ...
doc/cifarSC2011/gpundarray.txt:- All of them are a subset of the functionality of ``numpy.ndarray`` on the GPU
doc/cifarSC2011/gpundarray.txt:  - GPU code is harder/slower to do {\bf correctly} and {\bf fast} than on the CPU/Python
doc/cifarSC2011/gpundarray.txt:- Be compatible with both CUDA and OpenCL
doc/cifarSC2011/gpundarray.txt:  - We want people from C, C++, Ruby, R, ... all use the same base GPU N-dimensional array
doc/cifarSC2011/gpundarray.txt:Final GpuNdArray Note
doc/cifarSC2011/gpundarray.txt:- Will be the next GPU array container for Theano (this summer!)
doc/cifarSC2011/gpundarray.txt:- Probably also for PyCUDA, PyOpenCL
doc/cifarSC2011/gpundarray.txt:- Mailing list: http://lists.tiker.net/listinfo/gpundarray
doc/cifarSC2011/boot_camp_overview.txt:     gpu
doc/cifarSC2011/boot_camp_overview.txt:  * How to use pycuda code in Theano
doc/cifarSC2011/theano.txt:* Dynamic C/CUDA code generation
doc/cifarSC2011/theano.txt:* Transparent use of a GPU
doc/cifarSC2011/theano.txt:  * On GPU data-intensive calculations are typically between 6.5x and 44x faster. We've seen speedups up to 140x
doc/cifarSC2011/theano.txt:* GPU-ready
doc/cifarSC2011/theano.txt:    elif any( [x.op.__class__.__name__=='GpuGemm' for x in
doc/cifarSC2011/theano.txt:        print 'Used the gpu'
doc/cifarSC2011/theano.txt:        print 'ERROR, not able to tell if theano used the cpu or the gpu'
doc/cifarSC2011/theano.txt:GPU
doc/cifarSC2011/theano.txt:* Only 1 GPU per process
doc/cifarSC2011/theano.txt:* Use the Theano flag ``device=gpu`` to tell to use the GPU device
doc/cifarSC2011/theano.txt: * Use ``device=gpu{0, 1, ...}`` to specify which GPU if you have more than one
doc/cifarSC2011/theano.txt: * Shared variables with float32 dtype are by default moved to the GPU memory space
doc/cifarSC2011/theano.txt:* Modify and execute the example of `Exercise 2`_ to run with floatX=float32 on GPU
doc/cifarSC2011/introduction.txt:* Who has programmed a GPU before?
doc/cifarSC2011/introduction.txt: * Using CUDA (runtime? / driver?)
doc/cifarSC2011/introduction.txt: * Using PyCUDA ?
doc/cifarSC2011/introduction.txt: * Using OpenCL / PyOpenCL ?
doc/cifarSC2011/introduction.txt: * Using cudamat / gnumpy ?
doc/cifarSC2011/introduction.txt:you have GPU (I'm skipping some dtype-details which we'll come back to).
doc/cifarSC2011/introduction.txt:* Compiles most common expressions to C for CPU and GPU.
doc/cifarSC2011/introduction.txt: * FFTW, MKL, ATLAS, SciPy, Cython, CUDA
doc/cifarSC2011/introduction.txt:Why scripting for GPUs?
doc/cifarSC2011/introduction.txt:* GPUs are everything that scripting/high level languages are not
doc/cifarSC2011/introduction.txt:Best of both: scripted CPU invokes JIT-compiled kernels on GPU.
doc/cifarSC2011/introduction.txt:How Fast are GPUs?
doc/cifarSC2011/introduction.txt: * NVIDIA C2050 (515 Gf/s float64, 1Tf/s float32) 480 cores
doc/cifarSC2011/introduction.txt: * NVIDIA GTX580 (1.5Tf/s float32) 512 cores
doc/cifarSC2011/introduction.txt: * GPUs are faster, cheaper, more power-efficient
doc/cifarSC2011/introduction.txt:  * How much time was spent optimizing CPU vs GPU code?
doc/cifarSC2011/introduction.txt: * Theano goes up to 100x faster on GPU because it uses only one CPU core
doc/cifarSC2011/introduction.txt:Software for Directly Programming a GPU
doc/cifarSC2011/introduction.txt:* CUDA: C extension by NVIDIA 
doc/cifarSC2011/introduction.txt:* OpenCL: multi-vendor version of CUDA
doc/cifarSC2011/introduction.txt:* PyCUDA: python bindings to CUDA driver interface
doc/cifarSC2011/introduction.txt: * Python interface to CUDA
doc/cifarSC2011/introduction.txt: * Memory management of GPU objects
doc/cifarSC2011/introduction.txt: * Makes it easy to do GPU meta-programming from within Python
doc/cifarSC2011/introduction.txt:* PyOpenCL: PyCUDA for PyOpenCL
doc/cifarSC2011/index.txt: * automatic GPU use, and
doc/cifarSC2011/index.txt:    pyCUDA
doc/cifarSC2011/index.txt:    gpundarray
doc/cifarSC2011/advanced_theano.txt:  - Minimizes GPU transfers if GPU is involved
doc/cifarSC2011/advanced_theano.txt:        Theano Linker time (includes C, CUDA code generation/compiling): 3.185582e-02s
doc/cifarSC2011/advanced_theano.txt:      27.2%    77.8%       0.001s       5.74e-04s     C        2       2   theano.sandbox.cuda.basic_ops.HostFromGpu
doc/cifarSC2011/advanced_theano.txt:      18.1%    95.9%       0.001s       3.81e-04s     C        2       2   theano.sandbox.cuda.basic_ops.GpuFromHost
doc/cifarSC2011/advanced_theano.txt:       0.8%    99.3%       0.000s       3.29e-05s     C        1       1   theano.sandbox.cuda.basic_ops.GpuElemwise
doc/cifarSC2011/advanced_theano.txt:       0.2%    99.8%       0.000s       6.91e-06s     C        1       1   theano.sandbox.cuda.basic_ops.GpuDimShuffle
doc/cifarSC2011/advanced_theano.txt:      27.2%    77.8%       0.001s       5.74e-04s     C        2        2   HostFromGpu
doc/cifarSC2011/advanced_theano.txt:      18.1%    95.9%       0.001s       3.81e-04s     C        2        2   GpuFromHost
doc/cifarSC2011/advanced_theano.txt:       0.8%    97.7%       0.000s       3.29e-05s     C        1        1   GpuElemwise{Sub}[(0, 1)]
doc/cifarSC2011/advanced_theano.txt:       0.2%    99.3%       0.000s       6.91e-06s     C        1        1   GpuDimShuffle{1,0}
doc/cifarSC2011/advanced_theano.txt:      26.5%    53.4%       0.001s       1.12e-03s      1    10                     HostFromGpu(GpuDimShuffle{1,0}.0)
doc/cifarSC2011/advanced_theano.txt:       9.6%    86.7%       0.000s       4.04e-04s      1     3                     GpuFromHost(y)
doc/cifarSC2011/advanced_theano.txt:       8.5%    95.2%       0.000s       3.58e-04s      1     2                     GpuFromHost(x)
doc/cifarSC2011/advanced_theano.txt:       1.0%    96.3%       0.000s       4.39e-05s      1    13                     Elemwise{Composite{((i0 * scalar_softplus(i1)) - (i2 * i3 * scalar_softplus(i4)))}}(y, Elemwise{Composite{((-i0) - i1)}}[(0, 0)].0, TensorConstant{(1,) of -1.0}, HostFromGpu.0, Elemwise{neg,no_inplace}.0)
doc/cifarSC2011/advanced_theano.txt:       0.8%    97.1%       0.000s       3.29e-05s      1     7                     GpuElemwise{Sub}[(0, 1)](CudaNdarrayConstant{[ 1.]}, GpuFromHost.0)
doc/cifarSC2011/advanced_theano.txt:       0.7%    97.7%       0.000s       2.91e-05s      1    11                     HostFromGpu(GpuElemwise{Sub}[(0, 1)].0)
doc/cifarSC2011/advanced_theano.txt:       0.4%    98.1%       0.000s       1.50e-05s      1    15                     Elemwise{Composite{(((scalar_sigmoid(i0) * i1 * i2) / i3) - ((i4 * i1 * i5) / i3))}}[(0, 0)](Elemwise{Composite{((-i0) - i1)}}[(0, 0)].0, TensorConstant{(1,) of -1.0}, y, Elemwise{Cast{float64}}.0, Elemwise{ScalarSigmoid}[(0, 0)].0, HostFromGpu.0)
doc/cifarSC2011/advanced_theano.txt:       0.2%    99.4%       0.000s       6.91e-06s      1     6                     GpuDimShuffle{1,0}(GpuFromHost.0)
doc/cifarSC2011/advanced_theano.txt:        GPU: 1227KB (1227KB)
doc/cifarSC2011/advanced_theano.txt:        GPU: 1225KB (1227KB)
doc/cifarSC2011/advanced_theano.txt:           1254400B  [(400, 784)] c GpuFromHost(x)
doc/cifarSC2011/advanced_theano.txt:           1254400B  [(784, 400)] v GpuDimShuffle{1,0}(GpuFromHost.0)
doc/cifarSC2011/advanced_theano.txt:           1254400B  [(784, 400)] c HostFromGpu(GpuDimShuffle{1,0}.0)
doc/cifarSC2011/advanced_theano.txt:              3200B  [(400,)] i Elemwise{Composite{(((scalar_sigmoid(i0) * i1 * i2) / i3) - ((i4 * i1 * i5) / i3))}}[(0, 0)](Elemwise{Composite{((-i0) - i1)}}[(0, 0)].0, TensorConstant{(1,) of -1.0}, y, Elemwise{Cast{float64}}.0, Elemwise{ScalarSigmoid}[(0, 0)].0, HostFromGpu.0)
doc/cifarSC2011/advanced_theano.txt:              3200B  [(400,)] c Elemwise{Composite{((i0 * scalar_softplus(i1)) - (i2 * i3 * scalar_softplus(i4)))}}(y, Elemwise{Composite{((-i0) - i1)}}[(0, 0)].0, TensorConstant{(1,) of -1.0}, HostFromGpu.0, Elemwise{neg,no_inplace}.0)
doc/cifarSC2011/advanced_theano.txt:              1600B  [(400,)] i GpuElemwise{Sub}[(0, 1)](CudaNdarrayConstant{[ 1.]}, GpuFromHost.0)
doc/cifarSC2011/advanced_theano.txt:              1600B  [(400,)] c HostFromGpu(GpuElemwise{Sub}[(0, 1)].0)
doc/cifarSC2011/advanced_theano.txt:              1600B  [(400,)] c GpuFromHost(y)
doc/cifarSC2011/advanced_theano.txt:- In the last exercises, do you see a speed up with the GPU?
doc/cifarSC2011/advanced_theano.txt:- Is there something we can do to speed up the GPU version?
doc/cifarSC2011/pyCUDA.txt:.. _pyCUDA:
doc/cifarSC2011/pyCUDA.txt:PyCUDA
doc/cifarSC2011/pyCUDA.txt:- PyCUDA can access Nvidia's CUDA parallel computation API from Python
doc/cifarSC2011/pyCUDA.txt:  - PyCUDA knows about dependencies (e.g.. it won't detach from a context before all memory allocated in it is also freed)
doc/cifarSC2011/pyCUDA.txt:  - Abstractions to compile CUDA code from Python: ``pycuda.driver.SourceModule``
doc/cifarSC2011/pyCUDA.txt:  - A GPU memory buffer: \texttt{pycuda.gpuarray.GPUArray}
doc/cifarSC2011/pyCUDA.txt:  - Binding to all of CUDA's driver API
doc/cifarSC2011/pyCUDA.txt:  - All CUDA errors are automatically translated into Python exceptions
doc/cifarSC2011/pyCUDA.txt:  - PyCUDA's base layer is written in C++
doc/cifarSC2011/pyCUDA.txt:  import pycuda.autoinit
doc/cifarSC2011/pyCUDA.txt:  import pycuda.driver as drv
doc/cifarSC2011/pyCUDA.txt:  from pycuda.compiler import SourceModule
doc/cifarSC2011/pyCUDA.txt:.. _cifar2011_pyCUDA_theano:
doc/cifarSC2011/pyCUDA.txt:Theano + PyCUDA
doc/cifarSC2011/pyCUDA.txt:    import theano.misc.pycuda_init
doc/cifarSC2011/pyCUDA.txt:    from pycuda.compiler import SourceModule
doc/cifarSC2011/pyCUDA.txt:    import theano.sandbox.cuda as cuda
doc/cifarSC2011/pyCUDA.txt:    class PyCUDADoubleOp(theano.Op):
doc/cifarSC2011/pyCUDA.txt:            inp = cuda.basic_ops.gpu_contiguous(
doc/cifarSC2011/pyCUDA.txt:               cuda.basic_ops.as_cuda_ndarray_variable(inp))
doc/cifarSC2011/pyCUDA.txt:            pycuda_fct = mod.get_function("my_fct")
doc/cifarSC2011/pyCUDA.txt:                    z[0] = cuda.CudaNdarray.zeros(inputs[0][0].shape)
doc/cifarSC2011/pyCUDA.txt:                pycuda_fct(inputs[0][0], z[0], numpy.intc(inputs[0][0].size),
doc/cifarSC2011/pyCUDA.txt:   This contains GPU code so skip it
doc/cifarSC2011/pyCUDA.txt:>>> f = theano.function([x], PyCUDADoubleOp()(x)) # doctest: +SKIP
doc/internal/metadocumentation.txt:We use the Jenkins software to run daily buildbots for Theano, libgpuarray and
doc/internal/metadocumentation.txt:* `gpuarray buildbot <http://darjeeling.iro.umontreal.ca:8080/job/Buildbot_gpuarray/>`__
doc/acknowledgement.txt:* The GPU implementation of tensordot is based on code from Tijmen
doc/acknowledgement.txt:* Our random number generator implementation on CPU and GPU uses the MRG31k3p algorithm that is described in:
doc/acknowledgement.txt:* A better GPU memory allocator :attr:`CNMeM <config.lib.cnmem>` was included in Theano in the previous GPU back-end. It is still in the history, but not in the current version. It has the same license.
doc/proposals/conditional.txt:At the same time, it appears that as we try to integrate PyCUDA Ops another
doc/proposals/conditional.txt:problem arises.  We would like to use Op.perform() to drive the GPU, but it is
doc/proposals/conditional.txt:natural to move compilation of the CUDA kernel to a point after make_node() and a
doc/dev_start_guide.txt:Also, if you are changing GPU code, Travis doesn't test that, because
doc/dev_start_guide.txt:there are no GPUs on the test nodes.
doc/faq.txt:Theano flags, Theano versions, CPU and GPU or with other software like
doc/faq.txt:optimizations and disables the generation of any c/cuda code. This is useful
doc/faq.txt:If c/cuda code is necessary, as when using a GPU, the flag
doc/faq.txt:skip time consuming optimizations but still generate c/cuda code.
doc/faq.txt:   With :attr:`preallocate <config.gpuarray.preallocate>`, this isn't
doc/faq.txt:   very useful with GPU anymore.
doc/faq.txt:New GPU backend using libgpuarray
doc/faq.txt:The new theano GPU backend (:ref:`gpuarray`) uses ``config.gpuarray.preallocate`` for GPU memory allocation. 
doc/install_others.txt:NVIDIA Jetson TX1 embedded platform
doc/install_others.txt:AWS EC2 AMI pre-installed with Nvidia drivers, CUDA, cuDNN, Theano, Keras, Lasagne, Python 2, Python 3, PyCuda, Scikit-Learn, Pandas, Enum34, iPython, and Jupyter. Note, as always there is no charge for Theano and other open software, however there is a charge for AWS hosting + Bitfusion.
doc/install_others.txt:`Theano Docker (CUDA) <https://hub.docker.com/r/kaixhin/cuda-theano/>`_. These
doc/install_others.txt:    sudo nvidia-docker run -it kaixhin/cuda-theano:7.0
doc/install_others.txt:CUDA support requires `NVIDIA Docker <https://github.com/NVIDIA/nvidia-docker>`_.
doc/troubleshooting.txt:GPUs do not have virtual memory and as such all allocations must be assigned to
doc/troubleshooting.txt:support for virtual memory. Multiple allocations on a GPU can result in memory
doc/troubleshooting.txt:Since the GPU can't compute this kind of output, it would be
doc/troubleshooting.txt:    Theano's test should **NOT** be run with ``device=cuda``
doc/troubleshooting.txt:    or they will fail. The tests automatically use the gpu, if any, when
doc/troubleshooting.txt:    needed. If you don't want Theano to ever use the gpu when running tests,
doc/troubleshooting.txt:CPU and GPU memory usage.
doc/troubleshooting.txt:- :ref:`cuDNN <libdoc_gpuarray_dnn>` default cuDNN convolution use less
doc/troubleshooting.txt:   memory. GPU only.
doc/troubleshooting.txt:- :attr:`config.gpuarray.preallocate` = 1  # Preallocates the GPU memory
doc/troubleshooting.txt:  usage, but if you are at the limit of GPU memory available you might
doc/troubleshooting.txt:  need to specify a lower value. GPU only.
doc/troubleshooting.txt:- :attr:`config.optimizer_excluding` =low_memory , GPU only for now.
doc/troubleshooting.txt:- :attr:`config.scan.allow_gc` = True # Probably not significant slowdown on the GPU if memory cache is not disabled
doc/troubleshooting.txt:threads on multiple CPUs and GPUs. It will also print some Theano/NumPy
doc/troubleshooting.txt:alternate installation steps, GPU instructions, as well as tests that fail on
doc/crei2013/gpundarray.txt:.. _crei2013_gpundarray:
doc/crei2013/gpundarray.txt:GpuNdArray
doc/crei2013/gpundarray.txt:Why a common GPU ndarray?
doc/crei2013/gpundarray.txt:- Currently there are at least 4 different GPU array data structures in use by Python packages
doc/crei2013/gpundarray.txt:  - CudaNdarray (Theano), GPUArray (PyCUDA), CUDAMatrix (cudamat), GPUArray (PyOpenCL), ...
doc/crei2013/gpundarray.txt:- All of them are a subset of the functionality of ``numpy.ndarray`` on the GPU
doc/crei2013/gpundarray.txt:  - GPU code is harder/slower to do {\bf correctly} and {\bf fast} than on the CPU/Python
doc/crei2013/gpundarray.txt:- Be compatible with both CUDA and OpenCL
doc/crei2013/gpundarray.txt:  - We want people from C, C++, Ruby, R, ... all use the same base GPU N-dimensional array
doc/crei2013/gpundarray.txt:- Will be the next GPU array container for Theano (*this summer!*)
doc/crei2013/gpundarray.txt:- Probably also for PyCUDA, PyOpenCL
doc/crei2013/gpundarray.txt:- Mailing list: http://lists.tiker.net/listinfo/gpundarray
doc/crei2013/theano.txt:* Dynamic C/CUDA code generation
doc/crei2013/theano.txt:* Transparent use of a GPU
doc/crei2013/theano.txt:  * On GPU data-intensive calculations are typically between 6.5x and 44x faster. We've seen speedups up to 140x
doc/crei2013/theano.txt:* GPU-ready
doc/crei2013/theano.txt:    elif any([x.op.__class__.__name__=='GpuGemm' for x in
doc/crei2013/theano.txt:        print 'Used the gpu'
doc/crei2013/theano.txt:        print 'ERROR, not able to tell if theano used the cpu or the gpu'
doc/crei2013/theano.txt:GPU
doc/crei2013/theano.txt:* Only 1 GPU per process. Wiki page on using multiple process for multiple GPU
doc/crei2013/theano.txt:* Use the Theano flag ``device=gpu`` to tell to use the GPU device
doc/crei2013/theano.txt: * Use ``device=gpu{0, 1, ...}`` to specify which GPU if you have more than one
doc/crei2013/theano.txt: * Shared variables with float32 dtype are by default moved to the GPU memory space
doc/crei2013/theano.txt:* Use the Theano flag ``force_device=True``, to exit if Theano isn't able to use a GPU.
doc/crei2013/theano.txt:    and ``device=cpu`` disable the GPU.
doc/crei2013/theano.txt:* Modify and execute the example of `Exercise 2`_ to run with floatX=float32 on GPU
doc/crei2013/introduction.txt:* Who has programmed a GPU before?
doc/crei2013/introduction.txt: * Using CUDA (runtime? / driver?)
doc/crei2013/introduction.txt: * Using PyCUDA ?
doc/crei2013/introduction.txt: * Using OpenCL / PyOpenCL ?
doc/crei2013/introduction.txt: * Using cudamat / gnumpy ?
doc/crei2013/introduction.txt:you have GPU (I'm skipping some dtype-details which we'll come back to).
doc/crei2013/introduction.txt:* Compiles most common expressions to C for CPU and GPU.
doc/crei2013/introduction.txt: * FFTW, MKL, ATLAS, SciPy, Cython, CUDA
doc/crei2013/introduction.txt:Why scripting for GPUs?
doc/crei2013/introduction.txt:* GPUs are everything that scripting/high level languages are not
doc/crei2013/introduction.txt:Best of both: scripted CPU invokes JIT-compiled kernels on GPU.
doc/crei2013/introduction.txt:How Fast are GPUs?
doc/crei2013/introduction.txt: * NVIDIA C2050 (515 Gf/s float64, 1Tf/s float32) 480 cores
doc/crei2013/introduction.txt: * NVIDIA GTX580 (1.5Tf/s float32) 512 cores
doc/crei2013/introduction.txt: * GPUs are faster, cheaper, more power-efficient
doc/crei2013/introduction.txt:  * How much time was spent optimizing CPU vs GPU code?
doc/crei2013/introduction.txt: * Theano goes up to 100x faster on GPU because it uses only one CPU core
doc/crei2013/index.txt: * automatic GPU use,
doc/crei2013/index.txt:    gpundarray
doc/crei2013/advanced_theano.txt:  - Minimizes GPU transfers if GPU is involved
doc/crei2013/advanced_theano.txt:- In the last exercises, do you see a speed up with the GPU?
doc/crei2013/advanced_theano.txt:- Is there something we can do to speed up the GPU version?
doc/crei2013/logreg_profile.prof:    Theano Linker time (includes C, CUDA code generation/compiling): 2.649593e-02s
doc/crei2013/logreg_profile.prof:    Theano Linker time (includes C, CUDA code generation/compiling): 4.319906e-03s
doc/crei2013/logreg_profile.prof:    Theano Linker time (includes C, CUDA code generation/compiling): 3.081584e-02s
doc/install_macos.txt:.. _gpu_macos:
doc/install_macos.txt:    setup CUDA, but be aware of the following caveats:
doc/install_macos.txt:       * If you want to compile the CUDA SDK code, you may need to temporarily
doc/install_macos.txt:       * If CUDA seems unable to find a CUDA-capable GPU, you may need to manually
doc/install_macos.txt:         toggle your GPU on, which can be done with
doc/tutorial/shape_info.txt:- To generate faster C code for the 2d convolution on the CPU and the GPU,
doc/tutorial/loop.txt:  - Minimizes GPU transfers (if GPU is involved).
doc/tutorial/nan_tutorial.txt:CUDA Specific Option
doc/tutorial/modes_solution_1.py:elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
doc/tutorial/modes_solution_1.py:    print('Used the gpu')
doc/tutorial/modes_solution_1.py:    print('ERROR, not able to tell if theano used the cpu or the gpu')
doc/tutorial/using_gpu.txt:.. _using_gpu:
doc/tutorial/using_gpu.txt:Using the GPU
doc/tutorial/using_gpu.txt:For an introductory discussion of *Graphical Processing Units* (GPU)
doc/tutorial/using_gpu.txt:and their use for intensive parallel computation purposes, see `GPGPU
doc/tutorial/using_gpu.txt:<http://en.wikipedia.org/wiki/GPGPU>`_.
doc/tutorial/using_gpu.txt:Using the GPU in Theano is as simple as setting the ``device``
doc/tutorial/using_gpu.txt:configuration flag to ``device=cuda``. You can optionally target a
doc/tutorial/using_gpu.txt:specific gpu by specifying the number of the gpu as in
doc/tutorial/using_gpu.txt:e.g. ``device=cuda2``.  It is also encouraged to set the floating
doc/tutorial/using_gpu.txt:point precision to float32 when working on the GPU as that is usually
doc/tutorial/using_gpu.txt:``THEANO_FLAGS='device=cuda,floatX=float32'``.  You can also set these
doc/tutorial/using_gpu.txt:        device = cuda
doc/tutorial/using_gpu.txt:    * If your computer has multiple GPUs and you use ``device=cuda``,
doc/tutorial/using_gpu.txt:      the driver selects the one to use (usually cuda0).
doc/tutorial/using_gpu.txt:    * You can use the program ``nvidia-smi`` to change this policy.
doc/tutorial/using_gpu.txt:    * By default, when ``device`` indicates preference for GPU computations,
doc/tutorial/using_gpu.txt:      Theano will fall back to the CPU if there is a problem with the GPU.
doc/tutorial/using_gpu.txt:      Theano cannot use the GPU.
doc/tutorial/using_gpu.txt:.. _gpuarray:
doc/tutorial/using_gpu.txt:GpuArray Backend
doc/tutorial/using_gpu.txt:If you have not done so already, you will need to install libgpuarray
doc/tutorial/using_gpu.txt:as well as at least one computing toolkit (CUDA or OpenCL). Detailed
doc/tutorial/using_gpu.txt:`libgpuarray <http://deeplearning.net/software/libgpuarray/installation.html>`_.
doc/tutorial/using_gpu.txt:To install Nvidia's GPU-programming toolchain (CUDA) and configure
doc/tutorial/using_gpu.txt::ref:`Linux <gpu_linux>`, :ref:`MacOS <gpu_macos>` and :ref:`Windows <gpu_windows>`.
doc/tutorial/using_gpu.txt:While all types of devices are supported if using OpenCL, for the
doc/tutorial/using_gpu.txt:be referred to as GPU.
doc/tutorial/using_gpu.txt:  GpuArray backend uses ``config.gpuarray.preallocate`` for GPU memory
doc/tutorial/using_gpu.txt:  The backend was designed to support OpenCL, however current support is
doc/tutorial/using_gpu.txt:  .. _testing_the_gpu:
doc/tutorial/using_gpu.txt:Testing Theano with GPU
doc/tutorial/using_gpu.txt:To see if your GPU is being used, cut and paste the following program
doc/tutorial/using_gpu.txt:Use the Theano flag ``device=cuda`` to require the use of the GPU. Use the flag
doc/tutorial/using_gpu.txt:``device=cuda{0,1,...}`` to specify which GPU to use.
doc/tutorial/using_gpu.txt:                ('Gpu' not in type(x.op).__name__)
doc/tutorial/using_gpu.txt:      print('Used the gpu')
doc/tutorial/using_gpu.txt:input *x* is stored on the GPU.
doc/tutorial/using_gpu.txt:  $ THEANO_FLAGS=device=cpu python gpu_tutorial1.py
doc/tutorial/using_gpu.txt:  $ THEANO_FLAGS=device=cuda0 python gpu_tutorial1.py
doc/tutorial/using_gpu.txt:  Mapped name None to device cuda0: GeForce GTX 750 Ti (0000:07:00.0)
doc/tutorial/using_gpu.txt:  [GpuElemwise{exp,no_inplace}(<GpuArrayType<None>(float64, (False,))>), HostFromGpu(gpuarray)(GpuElemwise{exp,no_inplace}.0)]
doc/tutorial/using_gpu.txt:  Used the gpu
doc/tutorial/using_gpu.txt:By default functions that execute on the GPU still return a standard
doc/tutorial/using_gpu.txt:the GPU object directly.  The following code is modified to do just that.
doc/tutorial/using_gpu.txt:                ('Gpu' not in type(x.op).__name__)
doc/tutorial/using_gpu.txt:      print('Used the gpu')
doc/tutorial/using_gpu.txt:Here ``tensor.exp(x).transfer(None)`` means "copy ``exp(x)`` to the GPU",
doc/tutorial/using_gpu.txt:with ``None`` the default GPU context when not explicitly given.
doc/tutorial/using_gpu.txt:For information on how to set GPU contexts, see :ref:`tut_using_multi_gpu`.
doc/tutorial/using_gpu.txt:   $ THEANO_FLAGS=device=cuda0 python gpu_tutorial2.py
doc/tutorial/using_gpu.txt:   Mapped name None to device cuda0: GeForce GTX 750 Ti (0000:07:00.0)
doc/tutorial/using_gpu.txt:   [GpuElemwise{exp,no_inplace}(<GpuArrayType<None>(float64, (False,))>)]
doc/tutorial/using_gpu.txt:   Used the gpu
doc/tutorial/using_gpu.txt:  $ THEANO_FLAGS=device=cuda0 python gpu_tutorial2.py
doc/tutorial/using_gpu.txt:  Mapped name None to device cuda0: GeForce GTX 750 Ti (0000:07:00.0)
doc/tutorial/using_gpu.txt:  [GpuElemwise{exp,no_inplace}(<GpuArrayType<None>(float64, (False,))>)]
doc/tutorial/using_gpu.txt:  Used the gpu
doc/tutorial/using_gpu.txt:of execution on GPUs, meaning that the work isn't completed yet, just
doc/tutorial/using_gpu.txt:The object returned is a GpuArray from pygpu.  It mostly acts as a
doc/tutorial/using_gpu.txt:numpy ndarray with some exceptions due to its data being on the GPU.
doc/tutorial/using_gpu.txt:What Can be Accelerated on the GPU
doc/tutorial/using_gpu.txt:  equally fast on GPU as on CPU.
doc/tutorial/using_gpu.txt:  GPU than on the CPU.
doc/tutorial/using_gpu.txt:  on that data. Getting GPU performance largely hinges on making data transfer
doc/tutorial/using_gpu.txt:int, ...), however GPU support varies and some units can't deal with
doc/tutorial/using_gpu.txt:By default all inputs will get transferred to GPU. You can prevent an
doc/tutorial/using_gpu.txt:Tips for Improving Performance on GPU
doc/tutorial/using_gpu.txt:  ``.theanorc`` file if you plan to do a lot of GPU work.
doc/tutorial/using_gpu.txt:* The GPU backend supports *float64* variables, but they are still slower
doc/tutorial/using_gpu.txt:  to compute than *float32*. The more *float32*, the better GPU performance
doc/tutorial/using_gpu.txt:  machines), which slows down GPU computations on current hardware.
doc/tutorial/using_gpu.txt:* Minimize transfers to the GPU device by using ``shared`` variables
doc/tutorial/using_gpu.txt:  When using the GPU, tensor ``shared`` variables are stored on
doc/tutorial/using_gpu.txt:  the GPU by default to eliminate transfer time for GPU ops using those
doc/tutorial/using_gpu.txt:  something about GPU programming, have a look at how it's implemented
doc/tutorial/using_gpu.txt:  in theano.gpuarray.  Check the line similar to *Spent Xs(X%) in cpu
doc/tutorial/using_gpu.txt:  op, Xs(X%) in gpu op and Xs(X%) in transfer op*. This can tell you
doc/tutorial/using_gpu.txt:  if not enough of your graph is on the GPU or if there is too much
doc/tutorial/using_gpu.txt:  running on GPU, it is possible to debug or check your code by providing
doc/tutorial/using_gpu.txt:  .. _gpu_async:
doc/tutorial/using_gpu.txt:GPU Async Capabilities
doc/tutorial/using_gpu.txt:By default, all operations on the GPU are run asynchronously.  This
doc/tutorial/using_gpu.txt:This is made somewhat transparently by the underlying libgpuarray.
doc/tutorial/using_gpu.txt:It is possible to force synchronization for a particular GpuArray by
doc/tutorial/using_gpu.txt:    elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
doc/tutorial/using_gpu.txt:        print('Used the gpu')
doc/tutorial/using_gpu.txt:        print('ERROR, not able to tell if theano used the cpu or the gpu')
doc/tutorial/using_gpu.txt:Modify and execute this example to run on GPU with ``floatX=float32``
doc/tutorial/using_gpu.txt:Is there an increase in speed from CPU to GPU?
doc/tutorial/using_gpu.txt:What can be done to further increase the speed of the GPU version? Put
doc/tutorial/using_gpu.txt::download:`Solution<using_gpu_solution_1.py>`
doc/tutorial/using_gpu.txt:Software for Directly Programming a GPU
doc/tutorial/using_gpu.txt:* **CUDA**: GPU programming API by NVIDIA based on extension to C (CUDA C)
doc/tutorial/using_gpu.txt:* **OpenCL**: multi-vendor version of CUDA
doc/tutorial/using_gpu.txt:* **PyCUDA**: Python bindings to CUDA driver interface allow to access Nvidia's CUDA parallel
doc/tutorial/using_gpu.txt:    Makes it easy to do GPU meta-programming from within Python.
doc/tutorial/using_gpu.txt:    Abstractions to compile low-level CUDA code from Python (``pycuda.driver.SourceModule``).
doc/tutorial/using_gpu.txt:    GPU memory buffer (``pycuda.gpuarray.GPUArray``).
doc/tutorial/using_gpu.txt:  * Completeness: Binding to all of CUDA's driver API.
doc/tutorial/using_gpu.txt:  * Automatic error checking: All CUDA errors are automatically translated into Python exceptions.
doc/tutorial/using_gpu.txt:  * Speed: PyCUDA's base layer is written in C++.
doc/tutorial/using_gpu.txt:  * Good memory management of GPU objects:
doc/tutorial/using_gpu.txt:    PyCUDA knows about dependencies (e.g. it won't detach from a context before all memory
doc/tutorial/using_gpu.txt:  (This is adapted from PyCUDA's `documentation <http://documen.tician.de/pycuda/index.html>`_
doc/tutorial/using_gpu.txt:  and Andreas Kloeckner's `website <http://mathema.tician.de/software/pycuda>`_ on PyCUDA.)
doc/tutorial/using_gpu.txt:* **PyOpenCL**: PyCUDA for OpenCL
doc/tutorial/using_gpu.txt:Learning to Program with PyCUDA
doc/tutorial/using_gpu.txt:may easily leverage your knowledge by learning, first, to program a GPU with the
doc/tutorial/using_gpu.txt:CUDA extension to C (CUDA C) and, second, to use PyCUDA to access the CUDA
doc/tutorial/using_gpu.txt:* **CUDA API and CUDA C: Introductory**
doc/tutorial/using_gpu.txt:  * `NVIDIA's slides <http://www.sdsc.edu/us/training/assets/docs/NVIDIA-02-BasicsOfCUDA.pdf>`_
doc/tutorial/using_gpu.txt:  * `Stein's (NYU) slides <http://www.cs.nyu.edu/manycores/cuda_many_cores.pdf>`_
doc/tutorial/using_gpu.txt:* **CUDA API and CUDA C: Advanced**
doc/tutorial/using_gpu.txt:  * `MIT IAP2009 CUDA <https://sites.google.com/site/cudaiap2009/home>`_
doc/tutorial/using_gpu.txt:  * `NVIDIA's knowledge base <http://www.nvidia.com/content/cuda/cuda-developer-resources.html>`_
doc/tutorial/using_gpu.txt:  * `practical issues <http://stackoverflow.com/questions/2392250/understanding-cuda-grid-dimensions-block-dimensions-and-threads-organization-s>`_
doc/tutorial/using_gpu.txt:  * `CUDA optimization <http://www.gris.informatik.tu-darmstadt.de/cuda-workshop/slides.html>`_
doc/tutorial/using_gpu.txt:* **PyCUDA: Introductory**
doc/tutorial/using_gpu.txt:  * `Kloeckner's slides <http://www.gputechconf.com/gtcnew/on-demand-gtc.php?sessionTopic=&searchByKeyword=kloeckner&submit=&select=+&sessionEvent=2&sessionYear=2010&sessionFormat=3>`_
doc/tutorial/using_gpu.txt:  * `Kloeckner' website <http://mathema.tician.de/software/pycuda>`_
doc/tutorial/using_gpu.txt:* **PYCUDA: Advanced**
doc/tutorial/using_gpu.txt:  * `PyCUDA documentation website <http://documen.tician.de/pycuda/>`_
doc/tutorial/using_gpu.txt:The following examples give a foretaste of programming a GPU with PyCUDA. Once
doc/tutorial/using_gpu.txt:**Example: PyCUDA**
doc/tutorial/using_gpu.txt:  # (from PyCUDA's documentation)
doc/tutorial/using_gpu.txt:  import pycuda.autoinit
doc/tutorial/using_gpu.txt:  import pycuda.driver as drv
doc/tutorial/using_gpu.txt:  from pycuda.compiler import SourceModule
doc/tutorial/using_gpu.txt:.. _pyCUDA_theano:
doc/tutorial/using_gpu.txt:**Example: Theano + PyCUDA**
doc/tutorial/using_gpu.txt:    import theano.misc.pycuda_init
doc/tutorial/using_gpu.txt:    from pycuda.compiler import SourceModule
doc/tutorial/using_gpu.txt:    import theano.sandbox.cuda as cuda
doc/tutorial/using_gpu.txt:    class PyCUDADoubleOp(theano.Op):
doc/tutorial/using_gpu.txt:            inp = cuda.basic_ops.gpu_contiguous(
doc/tutorial/using_gpu.txt:               cuda.basic_ops.as_cuda_ndarray_variable(inp))
doc/tutorial/using_gpu.txt:            pycuda_fct = mod.get_function("my_fct")
doc/tutorial/using_gpu.txt:                    z[0] = cuda.CudaNdarray.zeros(inputs[0][0].shape)
doc/tutorial/using_gpu.txt:                pycuda_fct(inputs[0][0], z[0], numpy.intc(inputs[0][0].size),
doc/tutorial/using_gpu.txt:>>> f = theano.function([x], PyCUDADoubleOp()(x))  # doctest: +SKIP
doc/tutorial/using_gpu.txt:  on the GPU.
doc/tutorial/using_gpu.txt:* The mode `FAST_COMPILE` disables C code, so also disables the GPU. You
doc/tutorial/using_gpu.txt:  compilation and keep the GPU.
doc/tutorial/aliasing.txt:* Physically, Theano's memory space may be spread across the host, a GPU
doc/tutorial/aliasing.txt:It is not guaranteed to occur because if Theano is using a GPU device, then the
doc/tutorial/aliasing.txt:through side-effect, because with some devices (e.g. GPU devices) this technique will
doc/tutorial/aliasing.txt:must return a NumPy array too.  That's how Theano can make the GPU use
doc/tutorial/aliasing.txt:transparent.  But when you are using a GPU (or in the future perhaps a remote machine),
doc/tutorial/aliasing.txt:the gpu and create a ``shared`` variable on the gpu with this data, ``get_value`` will always
doc/tutorial/aliasing.txt:return gpu data even when ``return_internal_type=False``.
doc/tutorial/aliasing.txt:Modification of GPU variables through this sort of side-effect is impossible.
doc/tutorial/aliasing.txt:When ``shared`` variables are allocated on the GPU, the transfers to and from the GPU device memory can
doc/tutorial/aliasing.txt:be costly.  Here are a few tips to ensure fast and efficient use of GPU memory and bandwidth:
doc/tutorial/aliasing.txt:* Prior to Theano 0.3.1, ``set_value`` did not work in-place on the GPU. This meant that, sometimes,
doc/tutorial/aliasing.txt:  GPU memory for the new value would be allocated before the old memory was released. If you're
doc/tutorial/aliasing.txt:  running near the limits of GPU memory, this could cause you to run out of GPU memory
doc/tutorial/aliasing.txt:* It is also worth mentioning that, current GPU copying routines
doc/tutorial/aliasing.txt:  you assign to a GpuArraySharedVariable is *already*  *C-contiguous*.
doc/tutorial/aliasing.txt:(Further information on the current implementation of the GPU version
doc/tutorial/aliasing.txt:of ``set_value()`` can be found here: :ref:`libdoc_gpuarray_type`)
doc/tutorial/modes.txt:    elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
doc/tutorial/modes.txt:        print('Used the gpu')
doc/tutorial/modes.txt:        print('ERROR, not able to tell if theano used the cpu or the gpu')
doc/tutorial/modes.txt:- ``'FAST_COMPILE'``: Apply just a few graph optimizations and only use Python implementations. So GPU is disabled.
doc/tutorial/using_gpu_solution_1.py:# Solution to Exercise in section 'Using the GPU'
doc/tutorial/using_gpu_solution_1.py:elif any([n.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for n in
doc/tutorial/using_gpu_solution_1.py:    print('Used the gpu')
doc/tutorial/using_gpu_solution_1.py:    print('ERROR, not able to tell if theano used the cpu or the gpu')
doc/tutorial/using_gpu_solution_1.py:$ THEANO_FLAGS=profile=True,device=cpu python using_gpu_solution_1.py
doc/tutorial/using_gpu_solution_1.py:    Theano Linker time (includes C, CUDA code generation/compiling): 2.949309e-02s
doc/tutorial/using_gpu_solution_1.py:# 2.2 Profiling for GPU computations
doc/tutorial/using_gpu_solution_1.py:$ CUDA_LAUNCH_BLOCKING=1 THEANO_FLAGS=profile=True,device=cuda python using_gpu_solution_1.py
doc/tutorial/using_gpu_solution_1.py:Used the gpu
doc/tutorial/using_gpu_solution_1.py:    Theano Linker time (includes C, CUDA code generation/compiling): 8.239602e+00s
doc/tutorial/using_gpu_solution_1.py:  59.5%    59.5%       2.329s       1.16e-04s     C    20001       3   theano.sandbox.gpuarray.blas.GpuGemv
doc/tutorial/using_gpu_solution_1.py:  29.8%    89.3%       1.166s       1.30e-05s     C    90001      10   theano.sandbox.gpuarray.elemwise.GpuElemwise
doc/tutorial/using_gpu_solution_1.py:   4.1%    93.4%       0.162s       8.10e-06s     C    20001       3   theano.sandbox.gpuarray.basic_ops.HostFromGpu
doc/tutorial/using_gpu_solution_1.py:   3.3%    96.7%       0.131s       1.31e-05s     C    10000       1   theano.sandbox.gpuarray.elemwise.GpuCAReduceCuda
doc/tutorial/using_gpu_solution_1.py:   1.6%    98.3%       0.061s       6.10e-06s     C    10000       1   theano.sandbox.gpuarray.basic_ops.GpuFromHost
doc/tutorial/using_gpu_solution_1.py:   0.8%    99.1%       0.033s       1.09e-06s     C    30001       4   theano.sandbox.gpuarray.elemwise.GpuDimShuffle
doc/tutorial/using_gpu_solution_1.py:   0.7%    99.8%       0.026s       2.59e-06s     C    10001       2   theano.sandbox.gpuarray.basic_ops.GpuAllocEmpty
doc/tutorial/using_gpu_solution_1.py:  59.5%    59.5%       2.329s       1.16e-04s     C     20001        3   GpuGemv{inplace=True}
doc/tutorial/using_gpu_solution_1.py:   4.1%    63.6%       0.162s       8.10e-06s     C     20001        3   HostFromGpu(gpuarray)
doc/tutorial/using_gpu_solution_1.py:   4.0%    67.6%       0.157s       1.57e-05s     C     10000        1   GpuElemwise{Composite{((i0 * scalar_softplus(i1)) - (i2 * i3 * scalar_softplus(i4)))}}[]<gpuarray>
doc/tutorial/using_gpu_solution_1.py:   3.8%    71.4%       0.149s       1.49e-05s     C     10000        1   GpuElemwise{Composite{(((scalar_sigmoid(i0) * i1 * i2) / i3) - ((i4 * i1 * i5) / i3))}}[(0, 0)]<gpuarray>
doc/tutorial/using_gpu_solution_1.py:   3.7%    75.1%       0.144s       1.44e-05s     C     10000        1   GpuElemwise{sub,no_inplace}
doc/tutorial/using_gpu_solution_1.py:   3.6%    78.7%       0.141s       1.41e-05s     C     10000        1   GpuElemwise{gt,no_inplace}
doc/tutorial/using_gpu_solution_1.py:   3.4%    82.1%       0.133s       1.33e-05s     C     10000        1   GpuElemwise{Cast{float32}}[]<gpuarray>
doc/tutorial/using_gpu_solution_1.py:   3.4%    85.5%       0.133s       1.33e-05s     C     10000        1   GpuElemwise{Composite{((-i0) - i1)}}[(0, 0)]<gpuarray>
doc/tutorial/using_gpu_solution_1.py:   3.3%    88.8%       0.131s       1.31e-05s     C     10000        1   GpuCAReduceCuda{add}
doc/tutorial/using_gpu_solution_1.py:   2.9%    91.7%       0.112s       1.12e-05s     C     10000        1   GpuElemwise{neg,no_inplace}
doc/tutorial/using_gpu_solution_1.py:   2.6%    94.3%       0.102s       1.02e-05s     C     10000        1   GpuElemwise{Composite{(i0 - (i1 * i2))}}[(0, 0)]<gpuarray>
doc/tutorial/using_gpu_solution_1.py:   2.5%    96.7%       0.096s       9.63e-06s     C     10000        1   GpuElemwise{ScalarSigmoid}[(0, 0)]<gpuarray>
doc/tutorial/using_gpu_solution_1.py:   1.6%    98.3%       0.061s       6.10e-06s     C     10000        1   GpuFromHost<None>
doc/tutorial/using_gpu_solution_1.py:   0.7%    99.0%       0.026s       2.59e-06s     C     10001        2   GpuAllocEmpty{dtype='float32', context_name=None}
doc/tutorial/using_gpu_solution_1.py:   0.5%    99.5%       0.021s       1.06e-06s     C     20001        3   InplaceGpuDimShuffle{x}
doc/tutorial/using_gpu_solution_1.py:   0.3%    99.8%       0.011s       1.14e-06s     C     10000        1   InplaceGpuDimShuffle{1,0}
doc/tutorial/using_gpu_solution_1.py:   0.0%   100.0%       0.000s       2.00e-05s     C        1        1   GpuElemwise{Composite{GT(scalar_sigmoid((-((-i0) - i1))), i2)}}[]<gpuarray>
doc/tutorial/using_gpu_solution_1.py:  55.0%    55.0%       2.154s       2.15e-04s   10000     7   GpuGemv{inplace=True}(GpuAllocEmpty{dtype='float32', context_name=None}.0, TensorConstant{1.0}, x, w, TensorConstant{0.0})
doc/tutorial/using_gpu_solution_1.py:   4.5%    59.5%       0.176s       1.76e-05s   10000    18   GpuGemv{inplace=True}(w, TensorConstant{-0.00999999977648}, InplaceGpuDimShuffle{1,0}.0, GpuElemwise{Composite{(((scalar_sigmoid(i0) * i1 * i2) / i3) - ((i4 * i1 * i5) / i3))}}[(0, 0)]<gpuarray>.0, TensorConstant{0.999800026417})
doc/tutorial/using_gpu_solution_1.py:   4.0%    63.5%       0.157s       1.57e-05s   10000    12   GpuElemwise{Composite{((i0 * scalar_softplus(i1)) - (i2 * i3 * scalar_softplus(i4)))}}[]<gpuarray>(y, GpuElemwise{Composite{((-i0) - i1)}}[(0, 0)]<gpuarray>.0, GpuArrayConstant{[-1.]}, GpuElemwise{sub,no_inplace}.0, GpuElemwise{neg,no_inplace}.0)
doc/tutorial/using_gpu_solution_1.py:   3.8%    67.3%       0.149s       1.49e-05s   10000    15   GpuElemwise{Composite{(((scalar_sigmoid(i0) * i1 * i2) / i3) - ((i4 * i1 * i5) / i3))}}[(0, 0)]<gpuarray>(GpuElemwise{Composite{((-i0) - i1)}}[(0, 0)]<gpuarray>.0, GpuArrayConstant{[-1.]}, y, GpuElemwise{Cast{float32}}[]<gpuarray>.0, GpuElemwise{ScalarSigmoid}[(0, 0)]<gpuarray>.0, GpuElemwise{sub,no_inplace}.0)
doc/tutorial/using_gpu_solution_1.py:   3.7%    71.0%       0.144s       1.44e-05s   10000     4   GpuElemwise{sub,no_inplace}(GpuArrayConstant{[ 1.]}, y)
doc/tutorial/using_gpu_solution_1.py:   3.6%    74.6%       0.141s       1.41e-05s   10000    16   GpuElemwise{gt,no_inplace}(GpuElemwise{ScalarSigmoid}[(0, 0)]<gpuarray>.0, GpuArrayConstant{[ 0.5]})
doc/tutorial/using_gpu_solution_1.py:   3.4%    78.0%       0.133s       1.33e-05s   10000    10   GpuElemwise{Cast{float32}}[]<gpuarray>(InplaceGpuDimShuffle{x}.0)
doc/tutorial/using_gpu_solution_1.py:   3.4%    81.4%       0.133s       1.33e-05s   10000     9   GpuElemwise{Composite{((-i0) - i1)}}[(0, 0)]<gpuarray>(GpuGemv{inplace=True}.0, InplaceGpuDimShuffle{x}.0)
doc/tutorial/using_gpu_solution_1.py:   3.3%    84.7%       0.131s       1.31e-05s   10000    17   GpuCAReduceCuda{add}(GpuElemwise{Composite{(((scalar_sigmoid(i0) * i1 * i2) / i3) - ((i4 * i1 * i5) / i3))}}[(0, 0)]<gpuarray>.0)
doc/tutorial/using_gpu_solution_1.py:   2.9%    87.5%       0.112s       1.12e-05s   10000    11   GpuElemwise{neg,no_inplace}(GpuElemwise{Composite{((-i0) - i1)}}[(0, 0)]<gpuarray>.0)
doc/tutorial/using_gpu_solution_1.py:   2.6%    90.1%       0.102s       1.02e-05s   10000    20   GpuElemwise{Composite{(i0 - (i1 * i2))}}[(0, 0)]<gpuarray>(b, GpuArrayConstant{0.00999999977648}, GpuCAReduceCuda{add}.0)
doc/tutorial/using_gpu_solution_1.py:   2.5%    92.6%       0.096s       9.63e-06s   10000    13   GpuElemwise{ScalarSigmoid}[(0, 0)]<gpuarray>(GpuElemwise{neg,no_inplace}.0)
doc/tutorial/using_gpu_solution_1.py:   2.3%    94.9%       0.090s       9.04e-06s   10000    19   HostFromGpu(gpuarray)(GpuElemwise{gt,no_inplace}.0)
doc/tutorial/using_gpu_solution_1.py:   1.8%    96.7%       0.072s       7.16e-06s   10000    14   HostFromGpu(gpuarray)(GpuElemwise{Composite{((i0 * scalar_softplus(i1)) - (i2 * i3 * scalar_softplus(i4)))}}[]<gpuarray>.0)
doc/tutorial/using_gpu_solution_1.py:   1.6%    98.3%       0.061s       6.10e-06s   10000     6   GpuFromHost<None>(Shape_i{0}.0)
doc/tutorial/using_gpu_solution_1.py:   0.7%    99.0%       0.026s       2.59e-06s   10000     5   GpuAllocEmpty{dtype='float32', context_name=None}(Shape_i{0}.0)
doc/tutorial/using_gpu_solution_1.py:   0.3%    99.3%       0.013s       1.33e-06s   10000     0   InplaceGpuDimShuffle{x}(b)
doc/tutorial/using_gpu_solution_1.py:   0.3%    99.6%       0.011s       1.14e-06s   10000     2   InplaceGpuDimShuffle{1,0}(x)
doc/tutorial/using_gpu_solution_1.py:   0.2%    99.8%       0.008s       7.94e-07s   10000     8   InplaceGpuDimShuffle{x}(GpuFromHost<None>.0)
doc/tutorial/using_gpu_solution_1.py:Examine and compare 'Ops' summaries for CPU and GPU. Usually GPU ops 'GpuFromHost' and 'HostFromGpu' by themselves
doc/tutorial/using_gpu_solution_1.py:consume a large amount of extra time, but by making as few as possible data transfers between GPU and CPU, you can minimize their overhead.
doc/tutorial/using_gpu_solution_1.py:Notice that each of the GPU ops consumes more time than its CPU counterpart. This is because the ops operate on small inputs;
doc/tutorial/using_gpu_solution_1.py:if you increase the input data size (e.g. set N = 4000), you will see a gain from using the GPU.
doc/tutorial/index.txt:    using_gpu
doc/tutorial/index.txt:    using_multi_gpu
doc/tutorial/debug_faq.txt:"Why does my GPU function seem to be slow?"
doc/tutorial/debug_faq.txt:on CPU instead GPU. If that is the case, you can use assert_no_cpu_op to check if there
doc/tutorial/debug_faq.txt:``THEANO_FLAGS="float32,device=gpu,assert_no_cpu_op='raise'" python test.py``
doc/tutorial/examples.txt:on the :ref:`GPU<using_gpu>`.
doc/tutorial/examples.txt:See `Other Implementations`_ for GPU version.
doc/tutorial/examples.txt:The RandomStream only work on the CPU, MRG31k3p work on the CPU and GPU.
doc/tutorial/loading_and_saving.txt:``CudaNdarray`` objects contained within the object are saved separately as NPY
doc/tutorial/using_multi_gpu.txt:.. _tut_using_multi_gpu:
doc/tutorial/using_multi_gpu.txt:Using multiple GPUs
doc/tutorial/using_multi_gpu.txt:Theano has a feature to allow the use of multiple GPUs at the same
doc/tutorial/using_multi_gpu.txt:time in one function.  The multiple gpu feature requires the use of
doc/tutorial/using_multi_gpu.txt:the :ref:`gpuarray` backend, so make sure that works correctly.
doc/tutorial/using_multi_gpu.txt:refer to device names directly for multiple-gpu use.  You instead
doc/tutorial/using_multi_gpu.txt:    dev0->cuda0;dev1->cuda1
doc/tutorial/using_multi_gpu.txt:`dev0->cuda0` and `dev1->cuda1`.
doc/tutorial/using_multi_gpu.txt:';'.  To avoid confusion context names that begin with 'cuda' or
doc/tutorial/using_multi_gpu.txt:'opencl' are disallowed.  The device name is a device in the form that
doc/tutorial/using_multi_gpu.txt:gpuarray expects like 'cuda0' or 'opencl0:0'.
doc/tutorial/using_multi_gpu.txt:       $ THEANO_FLAGS="contexts=dev0->cuda0"
doc/tutorial/using_multi_gpu.txt:   $ THEANO_FLAGS="contexts=dev0->cuda0;dev1->cuda1" python -c 'import theano'
doc/tutorial/using_multi_gpu.txt:   Mapped name dev0 to device cuda0: GeForce GTX TITAN X (0000:09:00.0)
doc/tutorial/using_multi_gpu.txt:   Mapped name dev1 to device cuda1: GeForce GTX TITAN X (0000:06:00.0)
doc/tutorial/using_multi_gpu.txt:If you don't have enough GPUs for a certain model, you can assign the
doc/tutorial/using_multi_gpu.txt:   It is often the case that multi-gpu operation requires or assumes
doc/tutorial/using_multi_gpu.txt:   that all the GPUs involved are equivalent.  This is not the case
doc/tutorial/using_multi_gpu.txt:   built on the assumption that one of the GPU is slower or has
doc/tutorial/using_multi_gpu.txt:A simple graph on two GPUs
doc/tutorial/using_multi_gpu.txt:The following simple program works on two GPUs.  It builds a function
doc/tutorial/using_multi_gpu.txt:which perform two dot products on two different GPUs.
doc/tutorial/using_multi_gpu.txt:moving data between GPUs and also between the host and the GPUs.  Here
doc/tutorial/using_multi_gpu.txt:   # Move to the device associated with 'gpudev'
doc/tutorial/using_multi_gpu.txt:   gv = v.transfer('gpudev')
doc/tutorial/profiling_example_out.prof:    Theano Linker time (includes C, CUDA code generation/compiling): 9.584920e-01s
doc/omlw2014/presentation.tex:\title{Theano, Pylearn2, libgpuarray Presentation}
doc/omlw2014/presentation.tex:  Python <- \{NumPy/SciPy/libgpuarray\} <- Theano <- Pylearn2
doc/omlw2014/presentation.tex:  \item libgpuarray: GPU $n$-dimensional array object in C for CUDA and OpenCL
doc/omlw2014/presentation.tex:%% \begin{frame}{Why scripting for GPUs?}
doc/omlw2014/presentation.tex:%%   GPUs are everything that high level languages are not
doc/omlw2014/presentation.tex:%%   \begin{bf}Best of both worlds:\end{bf} easily scripted code which invokes high-performance GPU kernels.
doc/omlw2014/presentation.tex:    \item Compiles most common expressions to C for CPU and/or GPU
doc/omlw2014/presentation.tex:      \item BLAS, SciPy, Cython, Numba, PyCUDA, CUDA
doc/omlw2014/presentation.tex:    \item Built on top of Theano, for fast execution and use of GPU
doc/omlw2014/presentation.tex:\begin{frame}{libgpuarray}
doc/omlw2014/presentation.tex:  Goal: A common GPU $n$-dimensional array that can be reused by all projects, support for both CUDA and OpenCL.
doc/omlw2014/presentation.tex:  \item Currently there are at least 6 different GPU arrays in Python
doc/omlw2014/presentation.tex:    \item CudaNdarray (Theano), GPUArray (pycuda), CUDAMatrix (cudamat), GPUArray (pyopencl), Clyther, Copperhead, ...
doc/omlw2014/presentation.tex:  \item This is the new GPU backend on Theano
doc/omlw2014/presentation.tex:    \item Dynamic C/CUDA code generation
doc/omlw2014/presentation.tex:      \item C/C++, CUDA, OpenCL, PyCUDA, Cython, Numba, \ldots
doc/omlw2014/presentation.tex:    \item Transparent use of a GPU
doc/omlw2014/presentation.tex:      \item {\tt float32} only for now (libgpuarray provides much more)
doc/omlw2014/presentation.tex:\section{libgpuarray}
doc/omlw2014/presentation.tex:\begin{frame}{libgpuarray: Design Goals}
doc/omlw2014/presentation.tex:    \item We want people from C, C++, ruby, R, \ldots all use the same base GPU ndarray.
doc/omlw2014/presentation.tex:  \item Be compatible with CUDA and OpenCL.
doc/omlw2014/presentation.tex:  \item Multiple GPUs works.
doc/omlw2014/presentation.tex:  \item Is the next GPU array container for Theano and is working.
doc/omlw2014/presentation.tex:    \item OpenCL misses more implementations.
doc/omlw2014/presentation.tex:    \item Multiple GPUs on the way.
doc/omlw2014/presentation.tex:  \item Web site: \url{http://deeplearning.net/software/libgpuarray/}
doc/omlw2014/presentation.tex:Theano/Pylearn2/libgpuarry provide an environment for machine learning that is:
doc/omlw2014/Makefile:	rm -f pygpu_ndarray.so core.* *.o *~
doc/omlw2014/sharing.tex:\title{Theano, Pylearn2, libgpuarray: Sharing and Future}
doc/omlw2014/sharing.tex:\item Multi-GPU
doc/omlw2014/sharing.tex:\item Allow checkpoint with GPU to reload without GPU
doc/omlw2014/sharing.tex:\begin{frame}{libgpuarray}
doc/omlw2014/sharing.tex:\item Optimize the kernel selection and parametrization based on the GPU
doc/omlw2014/sharing.tex:  \item<2-> Common base object! \begin{bf}libgpuarray\end{bf}
doc/scripts/docgen.py:    # Make sure we don't use gpu to compile documentation
doc/introduction.txt:of magnitude by taking advantage of recent GPUs.
doc/introduction.txt:* use of GPU for computations
doc/introduction.txt:  parts your expression graph into CPU or GPU instructions, which run
doc/introduction.txt:* Can use many compiled languages, instructions sets: C/C++, CUDA, OpenCL, PTX, CAL, AVX, ...
doc/introduction.txt:    GPU or not depending on the input size.
doc/introduction.txt:* We have a new CUDA backend for tensors with many dtype support.
doc/index.txt:* **transparent use of a GPU** -- Perform data-intensive computations much faster than on a CPU.
doc/index.txt:* Removed support for the old (device=gpu) backend.  Use the new
doc/index.txt:  backend (device=cuda) for gpu computing.  See `Converting to the new
doc/index.txt:  gpu back end(gpuarray)
doc/index.txt:  <https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29>`_
doc/index.txt:* Multi-GPU.
doc/index.txt:* We added support for CNMeM to speed up the GPU memory allocation.
doc/index.txt:  `Transparent GPU Computing With Theano`_.
doc/index.txt:.. _Transparent GPU Computing With Theano: http://www.archive.org/details/Scipy2010-JamesBergstra-TransparentGpuComputingWithTheano
doc/index.txt:* Whether you're using a CPU or GPU device
doc/citation.txt:  Bengio. `"Theano: A CPU and GPU Math Expression Compiler"
doc/nextml2015/presentation.tex:  Python <- \{NumPy/SciPy/libgpuarray\} <- Theano <- Pylearn2
doc/nextml2015/presentation.tex:  \item libgpuarray: GPU $n$-dimensional array object in C for CUDA and OpenCL
doc/nextml2015/presentation.tex:    \item Compiles most common expressions to C for CPU and/or GPU
doc/nextml2015/presentation.tex:      \item BLAS, SciPy, Cython, Numba, PyCUDA, CUDA, ...
doc/nextml2015/presentation.tex:%% \begin{frame}{Why scripting for GPUs?}
doc/nextml2015/presentation.tex:%%   GPUs are everything that high level languages are not
doc/nextml2015/presentation.tex:%%   \begin{bf}Best of both worlds:\end{bf} easily scripted code which invokes high-performance GPU kernels.
doc/nextml2015/presentation.tex:# return views, supported on GPU
doc/nextml2015/presentation.tex:a_tensor[an_index_vector] # Supported on GPU
doc/nextml2015/presentation.tex:  \item compilation for GPU
doc/nextml2015/presentation.tex:\begin{frame}{Compilation for GPU}
doc/nextml2015/presentation.tex:  \item Theano current back-end only supports 32 bit on GPU
doc/nextml2015/presentation.tex:  \item libgpuarray (new-backend) support all dtype
doc/nextml2015/presentation.tex:  \item CUDA supports 64 bit, but is slow on gamer GPUs
doc/nextml2015/presentation.tex:  \item Set device flag to gpu (or a specific gpu, like gpu0)
doc/nextml2015/presentation.tex:%%     \item Built on top of Theano, for fast execution and use of GPU
doc/nextml2015/presentation.tex:%% \begin{frame}{libgpuarray}
doc/nextml2015/presentation.tex:%%   Goal: A common GPU $n$-dimensional array that can be reused by all projects, support for both CUDA and OpenCL.
doc/nextml2015/presentation.tex:%%   \item Currently there are at least 6 different GPU arrays in Python
doc/nextml2015/presentation.tex:%%     \item CudaNdarray (Theano), GPUArray (pycuda), CUDAMatrix (cudamat), GPUArray (pyopencl), Clyther, Copperhead, ...
doc/nextml2015/presentation.tex:%%   \item This is the new GPU backend on Theano
doc/nextml2015/presentation.tex:%%   \item Multiple GPUs works.
doc/nextml2015/presentation.tex:%%   \item Is the next GPU array container for Theano and is working.
doc/nextml2015/presentation.tex:%%     \item OpenCL misses more implementations.
doc/nextml2015/presentation.tex:%%     \item Multiple GPUs: supported in libgpuarray
doc/nextml2015/presentation.tex:%%     \item Multiple GPUs: close to get integrated in Theano.
doc/nextml2015/presentation.tex:%%   \item Web site: \url{http://deeplearning.net/software/libgpuarray/}
doc/nextml2015/presentation.tex:%% \section{libgpuarray}
doc/nextml2015/presentation.tex:%% \begin{frame}{libgpuarray: Design Goals}
doc/nextml2015/presentation.tex:%%     \item We want people from C, C++, ruby, R, \ldots all use the same base GPU ndarray.
doc/nextml2015/presentation.tex:%%   \item Be compatible with CUDA and OpenCL.
doc/install_generic.inc:If you use conda, you can directly install both theano and pygpu. Libgpuarray
doc/install_generic.inc:will be automatically installed as a dependency of pygpu.
doc/install_generic.inc:    conda install theano pygpu
doc/install_generic.inc:    Latest conda packages for theano (``>= 0.9``) and pygpu (``>= 0.6*``) currently don't support
doc/install_generic.inc:If you use pip, you have to install Theano and libgpuarray separately.
doc/install_generic.inc:libgpuarray
doc/install_generic.inc:    git clone https://github.com/Theano/libgpuarray.git
doc/install_generic.inc:    cd libgpuarray
doc/install_generic.inc:and then follow the `Step-by-step instructions <http://deeplearning.net/software/libgpuarray/installation.html#step-by-step-install>`__.
doc/install_generic.inc:libgpuarray
doc/install_generic.inc:Install the latest, development version of libgpuarray following the
doc/install_generic.inc:`Step-by-step instructions <http://deeplearning.net/software/libgpuarray/installation.html#step-by-step-install>`__.
doc/install_generic.inc:    Currently, you need ``libgpuarray`` version ``0.7.X`` that is not in conda default channel.
doc/install_generic.inc:        conda install -c mila-udem pygpu
doc/install_generic.inc:libgpuarray
doc/install_generic.inc:See instructions for bleeding-edge installation about ``libgpuarray``.
doc/library/compile/shared.txt:       the data is on the GPU, it will return a copy, but if the data
doc/library/compile/shared.txt:       GPU object.
doc/library/tensor/nnet/ctc.txt:   automatically to the GPU.
doc/library/tensor/nnet/bn.txt:.. seealso:: cuDNN batch normalization: :class:`theano.gpuarray.dnn.dnn_batch_normalization_train`, :class:`theano.gpuarray.dnn.dnn_batch_normalization_test>`.
doc/library/tensor/nnet/conv.txt:based one. On the GPU, there is a GEMM based and :ref:`cuDNN
doc/library/tensor/nnet/conv.txt:<libdoc_gpuarray_dnn>` version.
doc/library/tensor/nnet/conv.txt:By default on the GPU, if cuDNN is available, it will be used,
doc/library/tensor/nnet/conv.txt:Also, a meta-optimizer has been implemented for the gpu convolution
doc/library/tensor/nnet/conv.txt:    and GPU computation. They also support less type of convolution.
doc/library/tensor/nnet/conv.txt:    - :func:`GpuCorrMM <theano.gpuarray.blas.GpuCorrMM>`
doc/library/tensor/nnet/conv.txt:      This is a GPU-only 2d correlation implementation taken from
doc/library/tensor/nnet/conv.txt:      `caffe's CUDA implementation <https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu>`_. It does not flip the kernel.
doc/library/tensor/nnet/conv.txt:      `Toeplitz <http://en.wikipedia.org/wiki/Toeplitz_matrix>`_ matrix in a CUDA kernel.
doc/library/tensor/nnet/conv.txt:    - :func:`dnn_conv <theano.gpuarray.dnn.dnn_conv>` GPU-only
doc/library/tensor/nnet/conv.txt:      convolution using NVIDIA's cuDNN library.
doc/library/tensor/nnet/conv.txt:    - :func:`GpuCorr3dMM <theano.gpuarray.blas.GpuCorr3dMM>`
doc/library/tensor/nnet/conv.txt:      This is a GPU-only 3d correlation relying on a Toeplitz matrix
doc/library/tensor/nnet/conv.txt:      and gemm implementation (see :func:`GpuCorrMM <theano.sandbox.cuda.blas.GpuCorrMM>`)
doc/library/tensor/nnet/conv.txt:    - :func:`dnn_conv3d <theano.gpuarray.dnn.dnn_conv3d>` GPU-only
doc/library/tensor/nnet/conv.txt:      3D convolution using NVIDIA's cuDNN library (as :func:`dnn_conv <theano.gpuarray.dnn.dnn_conv>` but for 3d).
doc/library/sparse/index.txt:do not store data in a contiguous array. Note that there are no GPU
doc/library/index.txt:   gpuarray/index
doc/library/config.txt:        THEANO_FLAGS='floatX=float32,device=cuda0,gpuarray.preallocate=1'  python <myscript>.py
doc/library/config.txt:    ``THEANO_FLAGS='device=cpu,device=cuda0'``, then cuda0 will be used.
doc/library/config.txt:        device = cuda0
doc/library/config.txt:        [gpuarray]
doc/library/config.txt:    Attributes from a subsection of ``config`` (e.g. ``config.gpuarray.preallocate``,
doc/library/config.txt:    section (e.g. ``[gpuarray]``, ``[dnn.conv]``).
doc/library/config.txt:    String value: either ``'cpu'``, ``'cuda'``, ``'cuda0'``, ``'cuda1'``,
doc/library/config.txt:    ``'opencl0:0'``, ``'opencl0:1'``, ...
doc/library/config.txt:    Default device for computations. If ``'cuda*``, change the default to try
doc/library/config.txt:    to move computation to the GPU using CUDA libraries. If ``'opencl*'``,
doc/library/config.txt:    the OpenCL libraries will be used. To let the driver select the device,
doc/library/config.txt:    use ``'cuda'`` or ``'opencl'``. If we are not able to use the GPU,
doc/library/config.txt:    Do not use upper case letters, only lower case even if NVIDIA uses
doc/library/config.txt:    If ``True`` and ``device=gpu*``, we raise an error if we cannot
doc/library/config.txt:    we disable the GPU.  If ``False`` and ``device=gpu*``, and if the
doc/library/config.txt:    This is useful to run Theano's tests on a computer with a GPU, but
doc/library/config.txt:    without running the GPU tests.
doc/library/config.txt:.. attribute:: init_gpu_device
doc/library/config.txt:    String value: either ``''``, ``'cuda'``, ``'cuda0'``, ``'cuda1'``,
doc/library/config.txt:    ``'opencl0:0'``, ``'opencl0:1'``, ...
doc/library/config.txt:    Initialize the gpu device to use.
doc/library/config.txt:    When its value is ``'cuda*'`` or ``'opencl*'``, the theano
doc/library/config.txt:    Unlike :attr:`device`, setting this flag to a specific GPU will not
doc/library/config.txt:    computations, nor shared variables, to the specified GPU.
doc/library/config.txt:    This flag is useful to run GPU-specific tests on a particular GPU, instead
doc/library/config.txt:    Print active device at when the GPU device is initialized.
doc/library/config.txt:    are more deterministic, but slower. In particular, on the GPU,
doc/library/config.txt:    non-deterministic implementaion, e.g. when we do not have a GPU
doc/library/config.txt:.. note:: if :attr:`config.gpuarray.preallocate` is the default value
doc/library/config.txt:    or not disabled (-1), this is not useful anymore on the GPU.
doc/library/config.txt:.. attribute:: config.gpuarray.preallocate
doc/library/config.txt:    Controls the preallocation of memory with the gpuarray backend.
doc/library/config.txt:    of total GPU memory) of the memory pool. If more memory is needed,
doc/library/config.txt:        * 0 <= N <= 1: use this fraction of the total GPU memory (clipped to .95 for driver memory).
doc/library/config.txt:.. attribute:: config.gpuarray.sched
doc/library/config.txt:    The sched parameter passed for context creation to pygpu.  With
doc/library/config.txt:    CUDA, using "multi" mean using the parameter
doc/library/config.txt:    cudaDeviceScheduleBlockingSync. This is useful to lower the CPU overhead
doc/library/config.txt:    when waiting for GPU. One user found that it speeds up his other
doc/library/config.txt:.. attribute:: config.gpuarray.single_stream
doc/library/config.txt:    In the future when true multi-stream is enabled in libgpuarray,
doc/library/config.txt:.. attribute:: config.gpuarray.cache_path
doc/library/config.txt:   Default: ``config.compiledir``/gpuarray_kernels
doc/library/config.txt:   Directory to cache pre-compiled kernels for the gpuarray backend.
doc/library/config.txt:    `cuDNN <https://developer.nvidia.com/cudnn>`_ if it is available.
doc/library/config.txt:    Default: ``include`` sub-folder in CUDA root directory, or headers paths defined for the compiler.
doc/library/config.txt:    Default: Library sub-folder (``lib64`` on Linux) in CUDA root directory, or libraries paths defined for the compiler.
doc/library/gpuarray/linalg.txt:.. _libdoc_gpuarray_linalg:
doc/library/gpuarray/linalg.txt::mod:`theano.gpuarray.linalg` -- Linear algebra operation
doc/library/gpuarray/linalg.txt:.. automodule:: theano.gpuarray.linalg
doc/library/gpuarray/extra.txt:.. _libdoc_gpuarray_extra:
doc/library/gpuarray/extra.txt:.. automodule:: theano.gpuarray.opt_util
doc/library/gpuarray/extra.txt:.. automodule:: theano.gpuarray.kernel_codegen
doc/library/gpuarray/extra.txt:.. automodule:: theano.gpuarray.fp16_help
doc/library/gpuarray/ctc.txt:.. _libdoc_gpuarray_ctc:
doc/library/gpuarray/ctc.txt::mod:`theano.gpuarray.ctc` -- Connectionist Temporal Classification (CTC) loss
doc/library/gpuarray/ctc.txt:    automatically to the GPU.
doc/library/gpuarray/ctc.txt:.. module:: theano.gpuarray.ctc
doc/library/gpuarray/ctc.txt:.. autofunction:: theano.gpuarray.ctc.gpu_ctc
doc/library/gpuarray/ctc.txt:.. autoclass:: theano.gpuarray.ctc.GpuConnectionistTemporalClassification
doc/library/gpuarray/op.txt:.. _libdoc_gpuarray_op:
doc/library/gpuarray/op.txt:List of gpuarray Ops implemented
doc/library/gpuarray/op.txt:automatically transform CPU ops to their GPU equivalent. So this list
doc/library/gpuarray/op.txt:is just useful to let people know what is implemented on the GPU.
doc/library/gpuarray/op.txt:.. automodule:: theano.gpuarray.basic_ops
doc/library/gpuarray/op.txt:.. automodule:: theano.gpuarray.blas
doc/library/gpuarray/op.txt:.. automodule:: theano.gpuarray.elemwise
doc/library/gpuarray/op.txt:.. automodule:: theano.gpuarray.subtensor
doc/library/gpuarray/op.txt:.. automodule:: theano.gpuarray.nnet
doc/library/gpuarray/op.txt:.. automodule:: theano.gpuarray.neighbours
doc/library/gpuarray/fft.txt:.. _libdoc_gpuarray_fft:
doc/library/gpuarray/fft.txt::mod:`theano.gpuarray.fft` -- Fast Fourier Transforms
doc/library/gpuarray/fft.txt:Performs Fast Fourier Transforms (FFT) on the GPU.
doc/library/gpuarray/fft.txt:    You must install `scikit-cuda <http://scikit-cuda.readthedocs.io/en/latest>`_
doc/library/gpuarray/fft.txt:    to compute Fourier transforms on the GPU.
doc/library/gpuarray/fft.txt:.. automodule:: theano.gpuarray.fft
doc/library/gpuarray/fft.txt:shifted to the middle of the array. The Theano flag ``device=cuda{0,1...}`` must be used.
doc/library/gpuarray/fft.txt:    from theano.gpuarray import fft
doc/library/gpuarray/dnn.txt:.. _libdoc_gpuarray_dnn:
doc/library/gpuarray/dnn.txt::mod:`theano.gpuarray.dnn` -- cuDNN
doc/library/gpuarray/dnn.txt:`cuDNN <https://developer.nvidia.com/cuDNN>`_ is an NVIDIA library
doc/library/gpuarray/dnn.txt:currently installed with CUDA. You must download and install it
doc/library/gpuarray/dnn.txt:- The easiest is to include them in your CUDA installation. Copy the
doc/library/gpuarray/dnn.txt:  ``*.h`` files to ``CUDA_ROOT/include`` and the ``*.so*`` files to
doc/library/gpuarray/dnn.txt:  ``CUDA_ROOT/lib64`` (by default, ``CUDA_ROOT`` is ``/usr/local/cuda``
doc/library/gpuarray/dnn.txt:    Normally you should not call GPU Ops directly, but the CPU interface
doc/library/gpuarray/dnn.txt:    - :func:`theano.gpuarray.dnn.dnn_conv`, :func:`theano.gpuarray.dnn.dnn_conv3d`.
doc/library/gpuarray/dnn.txt:    - :func:`theano.gpuarray.dnn.dnn_gradweight`, :func:`theano.gpuarray.dnn.dnn_gradweight3d`.
doc/library/gpuarray/dnn.txt:    - :func:`theano.gpuarray.dnn.dnn_gradinput`, :func:`theano.gpuarray.dnn.dnn_gradinput3d`.
doc/library/gpuarray/dnn.txt:    - :func:`theano.gpuarray.dnn.dnn_pool`.
doc/library/gpuarray/dnn.txt:    - :func:`theano.gpuarray.dnn.dnn_batch_normalization_train`
doc/library/gpuarray/dnn.txt:    - :func:`theano.gpuarray.dnn.dnn_batch_normalization_test`.
doc/library/gpuarray/dnn.txt:    - :class:`theano.gpuarray.dnn.RNNBlock`
doc/library/gpuarray/dnn.txt:    - You can manually use the op :class:`GpuDnnSoftmax
doc/library/gpuarray/dnn.txt:      <theano.gpuarray.dnn.GpuDnnSoftmax>` to use its extra feature.
doc/library/gpuarray/dnn.txt:    - :func:`theano.gpuarray.dnn.dnn_spatialtf`.
doc/library/gpuarray/dnn.txt:    params_cudnn = gpuarray_shared_constructor(
doc/library/gpuarray/dnn.txt:.. automodule:: theano.gpuarray.dnn
doc/library/gpuarray/index.txt:.. _libdoc_gpuarray:
doc/library/gpuarray/index.txt::mod:`gpuarray` -- The (new) GPU backend
doc/library/gpuarray/index.txt:.. module:: theano.gpuarray
doc/library/gpuarray/index.txt:   :synopsis: Code for GPU programming (new)
doc/library/gpuarray/type.txt:.. _libdoc_gpuarray_type:
doc/library/gpuarray/type.txt::mod:`theano.gpuarray.type` -- Type classes
doc/library/gpuarray/type.txt:.. automodule:: theano.gpuarray.type
doc/hpcs2011_tutorial/presentation.tex:\title[GPU Programming made Easy]{GPU Programming made Easy}
doc/hpcs2011_tutorial/presentation.tex:\textcolor{red}{\huge{GPU Programming made Easy}}\\
doc/hpcs2011_tutorial/presentation.tex:  \frametitle{Faster on CPU and GPU}
doc/hpcs2011_tutorial/presentation.tex:%Why GPU
doc/hpcs2011_tutorial/presentation.tex:    \item Why Scripting for GPUs?
doc/hpcs2011_tutorial/presentation.tex:    \item Theano vs. PyCUDA vs. PyOpenCL vs. CUDA
doc/hpcs2011_tutorial/presentation.tex:% gpu for exercices
doc/hpcs2011_tutorial/presentation.tex:    \item GPU
doc/hpcs2011_tutorial/presentation.tex:% Exercises 3: logreg\_example.py on the gpu
doc/hpcs2011_tutorial/presentation.tex:%exercises 4: ProfileMode on logreg\_example, CPU vs GPU
doc/hpcs2011_tutorial/presentation.tex:    %\imagetop{\includegraphics[width=.6in]{pics/pycuda-logo-crop.pdf}}
doc/hpcs2011_tutorial/presentation.tex:  \item PyCUDA
doc/hpcs2011_tutorial/presentation.tex:% Exercices 6: pycuda_simple.py
doc/hpcs2011_tutorial/presentation.tex:  \item CUDA Overview
doc/hpcs2011_tutorial/presentation.tex:    \item Theano + PyCUDA
doc/hpcs2011_tutorial/presentation.tex:% Exercises 8: pycuda_double_op.py
doc/hpcs2011_tutorial/presentation.tex:  \item GpuNdArray
doc/hpcs2011_tutorial/presentation.tex:    \imagetop{\includegraphics[width=.6in]{pics/pycuda-logo-crop.pdf}}
doc/hpcs2011_tutorial/presentation.tex:  \item Only high level overview of CUDA
doc/hpcs2011_tutorial/presentation.tex:  \item Won't talk about how to optimize GPU code
doc/hpcs2011_tutorial/presentation.tex:  \frametitle{Why GPU}
doc/hpcs2011_tutorial/presentation.tex:      \item How much time was spent optimizing CPU vs GPU code
doc/hpcs2011_tutorial/presentation.tex:      \item NVIDIA C2050 (515 Gf/s float64, 1Tf/s float32) 480 cores
doc/hpcs2011_tutorial/presentation.tex:      \item NVIDIA GTX580 (1.5Tf/s float32) 512 cores
doc/hpcs2011_tutorial/presentation.tex:  \item Theano goes up to 100x faster on th GPU because we don't use multiple core on CPU
doc/hpcs2011_tutorial/presentation.tex:  \frametitle{Why Scripting for GPUs}
doc/hpcs2011_tutorial/presentation.tex:  \item GPUs are everything that scripting/high level languages are not
doc/hpcs2011_tutorial/presentation.tex:  \frametitle{Theano vs PyCUDA vs PyOpenCL vs CUDA}
doc/hpcs2011_tutorial/presentation.tex:      \item Generates costum C and CUDA code
doc/hpcs2011_tutorial/presentation.tex:    \item CUDA
doc/hpcs2011_tutorial/presentation.tex:      \item C extension by NVIDA that allow to code and use GPU
doc/hpcs2011_tutorial/presentation.tex:    \item PyCUDA (Python + CUDA)
doc/hpcs2011_tutorial/presentation.tex:        \item Python interface to CUDA
doc/hpcs2011_tutorial/presentation.tex:        \item Memory management of GPU objects
doc/hpcs2011_tutorial/presentation.tex:    \item PyOpenCL (Python + OpenCL)
doc/hpcs2011_tutorial/presentation.tex:      \item PyCUDA for OpenCL
doc/hpcs2011_tutorial/presentation.tex:  \item GPU programming / CUDA / OpenCL
doc/hpcs2011_tutorial/presentation.tex:%  There are some competitors for easy computing on gpu.
doc/hpcs2011_tutorial/presentation.tex:%  \item Jacket(GPU for matlab): http://www.accelereyes.com/
doc/hpcs2011_tutorial/presentation.tex:%  \item GPUmat(GPU for matlab, free): http://gp-you.org/
doc/hpcs2011_tutorial/presentation.tex:  \item Dynamic C/CUDA code generation
doc/hpcs2011_tutorial/presentation.tex:  \item Transparent use of a GPU
doc/hpcs2011_tutorial/presentation.tex:    \item On GPU data-intensive calculations are typically between 6.5x and 44x faster. We've seen speedups up to 140x
doc/hpcs2011_tutorial/presentation.tex:  \item Uses a variety of backend technologies (GPU,...)
doc/hpcs2011_tutorial/presentation.tex:source /groups/h/hpc2011/bin/GPU.csh
doc/hpcs2011_tutorial/presentation.tex:  \item GPU-ready
doc/hpcs2011_tutorial/presentation.tex:\subsection{GPU}
doc/hpcs2011_tutorial/presentation.tex:\frametitle{GPU}
doc/hpcs2011_tutorial/presentation.tex:\item Only 1 GPU per process
doc/hpcs2011_tutorial/presentation.tex:\item Use the Theano flag \texttt{device=gpu} to tell to use the GPU device
doc/hpcs2011_tutorial/presentation.tex:  \item Use \texttt{device=gpu{0, 1, ...}} to specify which GPU if you have more than one
doc/hpcs2011_tutorial/presentation.tex:  \item Shared variables with float32 dtype are by default moved to the GPU memory space
doc/hpcs2011_tutorial/presentation.tex:\frametitle{GPU for Exercises}
doc/hpcs2011_tutorial/presentation.tex:\item NVIDIA C2050 (515 Gf/s float64, 1Tf/s float32, 2400\$, 480 cores), compute capability 2.0
doc/hpcs2011_tutorial/presentation.tex:\item NVIDIA GTX580 (1.5Tf/s float32, 500\$, 512 cores), compute capability 2.0
doc/hpcs2011_tutorial/presentation.tex:\item NVIDIA Quadro FX 580 (71GF/s single, 140\$, 32 cores), compute capability 1.1, 'profesionnal card'
doc/hpcs2011_tutorial/presentation.tex:% BLAS on the cpu took 48s, 4s on the GPU
doc/hpcs2011_tutorial/presentation.tex:\item Modify and execute the code to run with floatX=float32 on GPU
doc/hpcs2011_tutorial/presentation.tex:\item In the last exercises, do you see a speed up with the GPU?
doc/hpcs2011_tutorial/presentation.tex:\item Is there something we can do to speed up the GPU version?
doc/hpcs2011_tutorial/presentation.tex:  \item Minimizes GPU transfers if GPU is involved
doc/hpcs2011_tutorial/presentation.tex:\section{PyCUDA}
doc/hpcs2011_tutorial/presentation.tex:\subsection{PyCUDA}
doc/hpcs2011_tutorial/presentation.tex:\frametitle{PyCUDA}
doc/hpcs2011_tutorial/presentation.tex:\includegraphics[width=2.5in]{pics/pycuda-logo-crop.pdf}
doc/hpcs2011_tutorial/presentation.tex:\item PyCUDA can access Nvidia's CUDA parallel computation API from Python
doc/hpcs2011_tutorial/presentation.tex:  \item PyCUDA knows about dependencies (e.g.. it won't detach from a context before all memory allocated in it is also freed)
doc/hpcs2011_tutorial/presentation.tex:  \item Abstractions to compile CUDA code from Python: \texttt{pycuda.driver.SourceModule}
doc/hpcs2011_tutorial/presentation.tex:  \item A GPU memory buffer: \texttt{pycuda.gpuarray.GPUArray}
doc/hpcs2011_tutorial/presentation.tex:  \item Binding to all of CUDA's driver API
doc/hpcs2011_tutorial/presentation.tex:  \item All CUDA errors are automatically translated into Python exceptions
doc/hpcs2011_tutorial/presentation.tex:  \item PyCUDA's base layer is written in C++
doc/hpcs2011_tutorial/presentation.tex:import pycuda.autoinit
doc/hpcs2011_tutorial/presentation.tex:import pycuda.driver as drv
doc/hpcs2011_tutorial/presentation.tex:from pycuda.compiler import SourceModule
doc/hpcs2011_tutorial/presentation.tex:%\frametitle{GpuArray}
doc/hpcs2011_tutorial/presentation.tex:\section{CUDA}
doc/hpcs2011_tutorial/presentation.tex:\subsection{CUDA Overview}
doc/hpcs2011_tutorial/presentation.tex:\frametitle{GPU Programming: Gains and Losses}
doc/hpcs2011_tutorial/presentation.tex:\item Data size influence more the implementation code on GPU
doc/hpcs2011_tutorial/presentation.tex:* Less problematic with new hardware (NVIDIA Fermi)
doc/hpcs2011_tutorial/presentation.tex:\frametitle{CPU vs GPU Architecture}
doc/hpcs2011_tutorial/presentation.tex:\includegraphics[width=4.7in]{pics/CPU_VS_GPU.png}
doc/hpcs2011_tutorial/presentation.tex:\small{\color{gray}Source NVIDIA CUDA\_C\_Programming\_Guide.pdf document}
doc/hpcs2011_tutorial/presentation.tex:\frametitle{Different GPU Block Repartition}
doc/hpcs2011_tutorial/presentation.tex:\small{\color{gray}Source NVIDIA CUDA\_C\_Programming\_Guide.pdf document}
doc/hpcs2011_tutorial/presentation.tex:\frametitle{GPU thread structure}
doc/hpcs2011_tutorial/presentation.tex:\small{\color{gray}Source NVIDIA CUDA\_C\_Programming\_Guide.pdf document}
doc/hpcs2011_tutorial/presentation.tex:\item Run the example in the file pycuda\_simple.py
doc/hpcs2011_tutorial/presentation.tex:%\frametitle{PyCUDA Exercises:TODO MOVE?!?!?}
doc/hpcs2011_tutorial/presentation.tex:{\color{gray}# others implementation (pycuda, ...):}
doc/hpcs2011_tutorial/presentation.tex:\subsection{Theano+PyCUDA}
doc/hpcs2011_tutorial/presentation.tex:\frametitle{Theano+PyCUDA Op Example}
doc/hpcs2011_tutorial/presentation.tex:import theano.misc.pycuda_init
doc/hpcs2011_tutorial/presentation.tex:from pycuda.compiler import SourceModule
doc/hpcs2011_tutorial/presentation.tex:import theano.sandbox.cuda as cuda
doc/hpcs2011_tutorial/presentation.tex:class PyCUDADoubleOp(theano.Op):
doc/hpcs2011_tutorial/presentation.tex:        inp = cuda.basic_ops.gpu_contiguous(
doc/hpcs2011_tutorial/presentation.tex:           cuda.basic_ops.as_cuda_ndarray_variable(inp))
doc/hpcs2011_tutorial/presentation.tex:\frametitle{Theano + PyCUDA Op Example: make\_thunk}
doc/hpcs2011_tutorial/presentation.tex:        pycuda_fct = mod.get_function("my_fct")
doc/hpcs2011_tutorial/presentation.tex:                z[0] = cuda.CudaNdarray.zeros(inputs[0][0].shape)
doc/hpcs2011_tutorial/presentation.tex:            pycuda_fct(inputs[0][0], z[0], numpy.intc(inputs[0][0].size),
doc/hpcs2011_tutorial/presentation.tex:\frametitle{Theano + PyCUDA Op Example: GPU Code}
doc/hpcs2011_tutorial/presentation.tex:\frametitle{Theano + PyCUDA Op Example: Test it!}
doc/hpcs2011_tutorial/presentation.tex:f = theano.function([x], PyCUDADoubleOp()(x))
doc/hpcs2011_tutorial/presentation.tex:\item Run the example in the file pycuda\_double\_op.py
doc/hpcs2011_tutorial/presentation.tex:\section{GpuNdArray}
doc/hpcs2011_tutorial/presentation.tex:\subsection{GpuNdArray}
doc/hpcs2011_tutorial/presentation.tex:\frametitle{Why a common GPU ndarray?}
doc/hpcs2011_tutorial/presentation.tex:\item Currently there are at least 4 different GPU array data structures in use by Python packages
doc/hpcs2011_tutorial/presentation.tex:  \item CudaNdarray (Theano), GPUArray (PyCUDA), CUDAMatrix (cudamat), GPUArray (PyOpenCL), ...
doc/hpcs2011_tutorial/presentation.tex:\item All of them are a subset of the functionality of \texttt{numpy.ndarray} on the GPU
doc/hpcs2011_tutorial/presentation.tex:  \item GPU code is harder/slower to do {\bf correctly} and {\bf fast} than on the CPU/Python
doc/hpcs2011_tutorial/presentation.tex:\item Be compatible with both CUDA and OpenCL
doc/hpcs2011_tutorial/presentation.tex:  \item We want people from C, C++, Ruby, R, ... all use the same base GPU N-dimensional array
doc/hpcs2011_tutorial/presentation.tex:\frametitle{Final GpuNdArray Note}
doc/hpcs2011_tutorial/presentation.tex:\item Will be the next GPU array container for Theano (this summer!)
doc/hpcs2011_tutorial/presentation.tex:\item Probably also for PyCUDA, PyOpenCL
doc/hpcs2011_tutorial/presentation.tex:\item Mailing list: http://lists.tiker.net/listinfo/gpundarray
doc/hpcs2011_tutorial/presentation.tex:  \item Generates fast, custom CPU code \textit{and} GPU code
doc/hpcs2011_tutorial/presentation.tex:  \item You can easily wrap existing CPU/GPU code with Theano
doc/hpcs2011_tutorial/pycuda_double_op.py:import theano.misc.pycuda_init
doc/hpcs2011_tutorial/pycuda_double_op.py:from pycuda.compiler import SourceModule
doc/hpcs2011_tutorial/pycuda_double_op.py:import theano.sandbox.cuda as cuda
doc/hpcs2011_tutorial/pycuda_double_op.py:class PyCUDADoubleOp(theano.Op):
doc/hpcs2011_tutorial/pycuda_double_op.py:        inp = cuda.basic_ops.gpu_contiguous(
doc/hpcs2011_tutorial/pycuda_double_op.py:           cuda.basic_ops.as_cuda_ndarray_variable(inp))
doc/hpcs2011_tutorial/pycuda_double_op.py:        pycuda_fct = mod.get_function("my_fct")
doc/hpcs2011_tutorial/pycuda_double_op.py:                z[0] = cuda.CudaNdarray.zeros(inputs[0][0].shape)
doc/hpcs2011_tutorial/pycuda_double_op.py:            pycuda_fct(inputs[0][0], z[0], numpy.intc(inputs[0][0].size),
doc/hpcs2011_tutorial/pycuda_double_op.py:f = theano.function([x], PyCUDADoubleOp()(x))
doc/hpcs2011_tutorial/pycuda_simple.py:import pycuda.autoinit
doc/hpcs2011_tutorial/pycuda_simple.py:import pycuda.driver as drv
doc/hpcs2011_tutorial/pycuda_simple.py:from pycuda.compiler import SourceModule
doc/hpcs2011_tutorial/pics/UdeM_NoirBleu_logo_Marie_crop.pdf:zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz('"=7<@%VFNCj0N]PIQJg4oY.hi7[:gPu76foQ16
doc/hpcs2011_tutorial/pics/UdeM_NoirBleu_logo_Marie_crop.pdf:)F9g6)F9g6)F9foQ16i043AjH]cGgPu76]Q4&QLe7XI;]u,@)ZTj<zzzzzzzzzzzzzzzzzzzz
doc/hpcs2011_tutorial/pics/UdeM_NoirBleu_logo_Marie_crop.pdf:zzzzzzzzzzzz$ig8-0*M=U;B>f<E]=%!PuI_`Y\=(;^2j8S`cqI`c@>lpf8'D,gPu76h3%a<h
doc/hpcs2011_tutorial/pics/UdeM_NoirBleu_logo_Marie_crop.pdf:zzzzzzzzzzzz!<<*".KTSM;&oW:FZBF%R8X(c^N0ATd=VE!gPu76iKaHDk*uDOjdH/LiKjNEg
doc/hpcs2011_tutorial/pics/UdeM_NoirBleu_logo_Marie_crop.pdf:zzzzzzzzzzzzzzz*WQ0?9H!m2F>s7#R8X(c^iKJUdY%T#gPu76hin*@jI#uJjI#uJhj+6Bg6)
doc/hpcs2011_tutorial/pics/UdeM_NoirBleu_logo_Marie_crop.pdf:P.?Gb^'6heV!o&gPu76i0F?Cig9]Gi0OEDh3@s?gQV[<gQMU;g6)F9foQ16foQ16foQ16foQ1
doc/hpcs2011_tutorial/pics/UdeM_NoirBleu_logo_Marie_crop.pdf:7gQDO:gl_X;h3.g=hNS!?i0=9BiKaHDig'QEhie$?gPu76fS]_0d=_K"bC9Kl`d.Ub_05kZ]l
doc/hpcs2011_tutorial/pics/UdeM_NoirBleu_logo_Marie_crop.pdf:zzzzzzzzzzzzzzzzzzzz$ig8-,QIfE72#Y$@jh<WHoqB/Pu7S^Zt]R@`d.Ubd=_K"gPu76ig'
doc/hpcs2011_tutorial/pics/UdeM_NoirBleu_logo_Marie_crop.pdf:Q16foQ16foQ16foQ16foQ16foQ16foQ16g5l:7g5l:7g5l:7gQDO:glh^<gQV[<foH+5gPu76
doc/hpcs2011_tutorial/pics/UdeM_NoirBleu_logo_Marie_crop.pdf:+dTmM+dTmM+dTmM+dU3h:gUj[gpUj[gpUjI[nWI9L$Y(VZ4X*o^&N*Krp<%'L%"98E%zzzzzz
doc/hpcs2011_tutorial/logreg_example.py:elif any( [x.op.__class__.__name__=='GpuGemm' for x in train.maker.fgraph.toposort()]):
doc/hpcs2011_tutorial/logreg_example.py:    print('Used the gpu')
doc/hpcs2011_tutorial/logreg_example.py:    print('ERROR, not able to tell if theano used the cpu or the gpu')
doc/extending/extending_theano.txt:the old GPU back-end, we use a float32 CudaNdarray to store the MRG
doc/extending/extending_theano.txt:[Gpu]AllocEmpty or some computation on it (like done by Scan).
doc/extending/ctype.txt:        This is used for the GPU code.
doc/extending/ctype.txt:respectively. See an example for the type ``GpuArrayType`` (GPU
doc/extending/ctype.txt:array) in the file `theano/gpuarray/type.py`. The version
doc/extending/ctype.txt:respectively. See an example for the type ``GpuArrayType`` (GPU
doc/extending/ctype.txt:array) in the file `thean/gpuarray/type.py`. The version
doc/extending/graphstructures.txt:twice or reformulate parts of the graph to a GPU specific version.
doc/extending/extending_theano_c.txt:implementation. It does not cover ops that run on a GPU but it does introduce
doc/extending/extending_theano_c.txt:many elements and concepts which are relevant for GPU ops. This tutorial is
doc/extending/extending_theano_c.txt:Tensors, and equivalent types on GPU), the following macros will be
doc/extending/extending_theano_c.txt:For the GPU, you must add in this second flag `nvcc.flags=-g` (it slow
doc/extending/extending_theano_c.txt:down computation on the GPU, but it is enabled by default on the CPU).
doc/extending/other_ops.txt:operation. It can also reuse the GPU elemwise code. It is similar for
doc/extending/other_ops.txt:The fastest, but less developed, is CURAND. It works only on CUDA-enabled
doc/extending/other_ops.txt:GPUs. It does not work on the CPU and it has fewer random distributions
doc/extending/other_ops.txt:The recommended and 2nd faster is MRG. It works on the GPU and CPU and
doc/extending/cop.txt:         You cannot specify CUDA kernels in the code returned by this
doc/extending/cop.txt:         since that isn't supported by CUDA.  You should place your
doc/extending/optimization.txt:        Theano Linker time (includes C, CUDA code generation/compiling): 7.893991e-02s
doc/extending/optimization.txt:* ``Theano Optimizer time: 1.152431e+00s`` means that we spend 1.15s in the ``theano.function`` phase where we optimize (modify) the graph to make it faster / more stable numerically / work on GPU /...
doc/extending/optimization.txt:* ``Theano Linker time (includes C, CUDA code generation/compiling): 7.893991e-02s`` means that we spent 7.9e-2s in *linker* phase of ``theano.function``.
doc/extending/extending_theano_gpu.txt:.. _extending_theano_gpu:
doc/extending/extending_theano_gpu.txt:Extending Theano with a GPU Op
doc/extending/extending_theano_gpu.txt:    This covers the :ref:`gpuarray <gpuarray>` back-end for the GPU.
doc/extending/extending_theano_gpu.txt:This tutorial covers how to extend Theano with an op that offers a GPU
doc/extending/extending_theano_gpu.txt:Writing a new GPU op can be done in Python for some simple tasks, but
doc/extending/extending_theano_gpu.txt:One of the major differences with GPU ops is that they require a
doc/extending/extending_theano_gpu.txt:interface, :func:`theano.gpuarray.basic_ops.infer_context_name` was
doc/extending/extending_theano_gpu.txt:        a = as_gpuarray_variable(a, ctx)
doc/extending/extending_theano_gpu.txt:        b = as_gpuarray_variable(b, ctx)
doc/extending/extending_theano_gpu.txt:        c = as_gpuarray_variable(c, ctx)
doc/extending/extending_theano_gpu.txt:In this example the Op takes three inputs, all on the GPU.  In case
doc/extending/extending_theano_gpu.txt:one or more of your inputs is not supposed to be on the GPU, you
doc/extending/extending_theano_gpu.txt::func:`as_gpuarray_variable` on it.
doc/extending/extending_theano_gpu.txt:Also note that :func:`theano.gpuarray.basic_ops.as_gpuarray_variable`
doc/extending/extending_theano_gpu.txt:not enough to know you want the value to be on the GPU, you also want
doc/extending/extending_theano_gpu.txt:to know which GPU to put it on.  In almost all cases, you can pass in
doc/extending/extending_theano_gpu.txt:        C[0] = pygpu.empty([A.shape[0], B.shape[1]], dtype=A.dtype, A.context)
doc/extending/extending_theano_gpu.txt:        pygpu.blas.gemm(1, A, B, 0, C, overwrite_c=True)
doc/extending/extending_theano_gpu.txt:Note that ``GpuArrayType`` objects also have a ``context_name``
doc/extending/extending_theano_gpu.txt:be used for calls to pygpu or libgpuarray, but it should be used for
doc/extending/extending_theano_gpu.txt::data:`theano.gpuarray.type.gpu_context_type` and the params object
doc/extending/extending_theano_gpu.txt:If you don't have any input variables on the GPU you can follow the
doc/extending/extending_theano_gpu.txt:the example of :class:`GpuFromHost
doc/extending/extending_theano_gpu.txt:<theano.gpuarray.basic_ops.GpuFromHost>` or :class:`GpuEye
doc/extending/extending_theano_gpu.txt:<theano.gpuarray.basic_ops.GpuEye>`.  This is not a case that you
doc/extending/extending_theano_gpu.txt:to leverage :class:`GpuKernelBase
doc/extending/extending_theano_gpu.txt:<theano.gpuarray.basic_ops.GpuKernelBase>` (or :class:`CGpuKernelBase
doc/extending/extending_theano_gpu.txt:<theano.gpuarray.basic_ops.CGpuKernelBase>` if you want to use the
doc/extending/extending_theano_gpu.txt:For plain :class:`GpuKernelBase
doc/extending/extending_theano_gpu.txt:<theano.gpuarray.basic_ops.GpuKernelBase>`, you have to define a
doc/extending/extending_theano_gpu.txt:method called ``gpu_kernels`` which returns a list of :class:`Kernel
doc/extending/extending_theano_gpu.txt:<theano.gpuarray.basic_ops.Kernel>` objects.  You can define as many
doc/extending/extending_theano_gpu.txt:    def gpu_kernels(self, node, name):
doc/extending/extending_theano_gpu.txt:                params=[gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SIZE],
doc/extending/extending_theano_gpu.txt:If you want to use ``COp``, then you should use ``CGpuKernelBase``
doc/extending/extending_theano_gpu.txt:``CGpuKernelBase``::
doc/extending/extending_theano_gpu.txt:right, which GpuKernelBase handles for you.  But if you really want to
doc/extending/extending_theano_gpu.txt:libgpuarray.
doc/extending/extending_theano_gpu.txt:`libgpuarray documentation
doc/extending/extending_theano_gpu.txt:<http://deeplearning.net/software/libgpuarray/>`_.
doc/extending/extending_theano_gpu.txt:declared in :mod:`theano.gpuarray.fp16_help`.
doc/extending/extending_theano_gpu.txt:by :func:`load_w() <theano.gpuarray.fp16_help.load_w>`. Similarly writes
doc/extending/extending_theano_gpu.txt:<theano.gpuarray.fp16_help.write_w>`.  Finally working data should
doc/extending/extending_theano_gpu.txt:<theano.gpuarray.fp16_help.work_dtype>`.
doc/extending/extending_theano_gpu.txt:GpuKernelBase
doc/extending/extending_theano_gpu.txt:.. literalinclude:: ../../theano/gpuarray/basic_ops.py
doc/extending/extending_theano_gpu.txt:    :pyobject: GpuEye
doc/extending/extending_theano_gpu.txt:CGpuKernelBase
doc/extending/extending_theano_gpu.txt:.. literalinclude:: ../../theano/gpuarray/tests/test_cgpukernelbase.py
doc/extending/extending_theano_gpu.txt:    :pyobject: GpuEye
doc/extending/extending_theano_gpu.txt:``tstgpueye.c``
doc/extending/extending_theano_gpu.txt:.. literalinclude:: ../../theano/gpuarray/tests/c_code/tstgpueye.c
doc/extending/extending_theano_gpu.txt:PyCUDA
doc/extending/extending_theano_gpu.txt:For things in PyCUDA (or things wrapped with PyCUDA), we usually need
doc/extending/extending_theano_gpu.txt:to create a PyCUDA context.  This can be done with the following
doc/extending/extending_theano_gpu.txt:    with gpuarray_cuda_context:
doc/extending/extending_theano_gpu.txt:        pycuda_context = pycuda.driver.Context.attach()
doc/extending/extending_theano_gpu.txt:require it, you can also just use the pygpu context and a `with`
doc/extending/extending_theano_gpu.txt:current context on the cuda stack.
doc/extending/extending_theano_gpu.txt:GpuArray objects are compatible with PyCUDA and will expose the
doc/extending/extending_theano_gpu.txt:notable exception is PyCUDA kernels which require native objects.  If
doc/extending/extending_theano_gpu.txt:you need to convert a pygpu GpuArray to a PyCUDA GPUArray, this code
doc/extending/extending_theano_gpu.txt:    assert pygpu_array.flags['IS_C_CONTIGUOUS']
doc/extending/extending_theano_gpu.txt:    pycuda_array = pycuda.gpuarray.GPUArray(pygpu_array.shape,
doc/extending/extending_theano_gpu.txt:                                            pygpu_array.dtype,
doc/extending/extending_theano_gpu.txt:                                            base=pygpu_array,
doc/extending/extending_theano_gpu.txt:                                            gpudata=(pygpu_array.gpudata +
doc/extending/extending_theano_gpu.txt:                                                     pygpu_array.offset))
doc/extending/extending_theano_gpu.txt:Otherwise, you will have to make sure that you synchronize the pygpu
doc/extending/index.txt:    extending_theano_gpu
doc/extending/type.txt:      shared variable on the gpu.
doc/extending/type.txt:      CudaNdarray, ...
doc/optimizations.txt::term:`GPU transfer`                                      x              x   x    x                              x   
doc/optimizations.txt:        GPU) is a bottleneck.
doc/optimizations.txt:    GPU transfer
doc/optimizations.txt:        CPU and which to evaluate on the GPU is a greedy one.  There are a
doc/optimizations.txt:        number of Ops ***TODO*** with GPU implementations and whenever we find
doc/optimizations.txt:        a graph copying data from GPU to CPU in order to evaluate an
doc/optimizations.txt:        expression that could have been evaluated on the GPU, we substitute
doc/optimizations.txt:        the GPU version of that Op for the CPU version.  Likewise if we are
doc/optimizations.txt:        copying the output of a Op with a GPU implementation to the GPU, 
doc/optimizations.txt:        then we substitute the GPU version for the CPU version.  In this way, if all goes well,
doc/optimizations.txt:        1. copy non-shared inputs to GPU
doc/optimizations.txt:        2. carry out most/all computations on the GPU
doc/optimizations.txt:        When using a GPU, :func:`shared()` will default to GPU storage for
doc/optimizations.txt:        See :func:`theano.sandbox.cuda.opt.*`.
doc/requirements.inc:.. _NVIDIA CUDA drivers and SDK: http://developer.nvidia.com/object/gpucomputing.html
doc/requirements.inc:.. _libgpuarray: http://deeplearning.net/software/libgpuarray/installation.html
doc/requirements.inc:.. _pycuda: https://mathema.tician.de/software/pycuda/
doc/requirements.inc:.. _skcuda: http://scikit-cuda.readthedocs.io/en/latest/
doc/requirements.inc:    `NVIDIA CUDA drivers and SDK`_
doc/requirements.inc:        **Highly recommended** Required for GPU code generation/execution on NVIDIA gpus. See instruction below.
doc/requirements.inc:    `libgpuarray`_
doc/requirements.inc:        Required for GPU/CPU code generation on CUDA and OpenCL devices (see: :ref:`gpuarray`).
doc/requirements.inc:    `pycuda`_ and `skcuda`_
doc/requirements.inc:        Required for some extra operations on the GPU like fft and
doc/requirements.inc:        ``pip install pycuda scikit-cuda``. For cuda 8, the dev
doc/requirements.inc:        version of skcuda (will be released as 0.5.2) is needed for
doc/requirements.inc:        cusolver: ``pip install pycuda; pip install
doc/requirements.inc:        git+https://github.com/lebedov/scikit-cuda.git#egg=scikit-cuda``.
doc/requirements.inc:Install and configure the GPU drivers (recommended)
doc/requirements.inc:    OpenCL support is still minimal for now.
doc/requirements.inc:1. Install CUDA drivers
doc/requirements.inc:    * Follow `this link <https://developer.nvidia.com/cuda-downloads>`__
doc/requirements.inc:      to install the CUDA driver and the CUDA Toolkit.
doc/requirements.inc:      command `nvidia-smi` from the command line.
doc/requirements.inc:        program. This folder is called the *cuda root* directory.
doc/requirements.inc:    * Add the CUDA 'lib' subdirectory (and/or 'lib64' subdirectory if you have a
doc/requirements.inc:      variable. Example: ``/usr/local/cuda/lib64``
conda/meta.yaml:    - {{ pin_compatible('pygpu', '0.7', max_pin='0.8') }}   # [not osx]
conda/meta.yaml:    - theano.gpuarray
conda/meta.yaml:    - theano.gpuarray.tests
conda/meta.yaml:  summary: Optimizing compiler for evaluating mathematical expressions on CPUs and GPUs.
conda/meta.yaml:    of a GPU, efficient symbolic differentiation, speed and stability
HISTORY.txt: - Handle reductions with non-default accumulator dtype better on the GPU.
HISTORY.txt: - Fixed compilation and improved float16 support for topK on GPU
HISTORY.txt:   - **NB**: topK support on GPU is experimental and may not work for
HISTORY.txt:             large input sizes on certain GPUs
HISTORY.txt: - Attempted to prevent re-initialization of the GPU in a child process
HISTORY.txt:   To install it: ``conda install -c mila-udem theano pygpu``
HISTORY.txt: - Support pygpu ``0.7``
HISTORY.txt: - Removed old GPU backend ``theano.sandbox.cuda``. New backend ``theano.gpuarray`` is now the official GPU backend
HISTORY.txt:   - Fixed memory leaks related to elemwise ops on GPU
HISTORY.txt:   - ``cuda.enabled``
HISTORY.txt:   - ``gpuarray.sync``
HISTORY.txt:   - ``pycuda.init``
HISTORY.txt:GPU:
HISTORY.txt: - Added a meta-optimizer to select the fastest GPU implementations for convolutions
HISTORY.txt: - Prevent GPU initialization when not required
HISTORY.txt: - Added method ``my_theano_function.sync_shared()`` to help synchronize GPU Theano functions
HISTORY.txt: - Added useful stats for GPU in profile mode
HISTORY.txt: - Added GPU ops based on `magma library <http://icl.cs.utk.edu/magma/software/>`_:
HISTORY.txt: - Added ``GpuCublasTriangularSolve``
HISTORY.txt: - Added atomic addition and exchange for ``long long`` values in ``GpuAdvancedIncSubtensor1_dev20``
HISTORY.txt: - Support GPU SoftMax in both OpenCL and CUDA
HISTORY.txt: - Support offset parameter ``k`` for ``GpuEye``
HISTORY.txt: - ``CrossentropyCategorical1Hot`` and its gradient are now lifted to GPU
HISTORY.txt:   - Added new Theano flags ``cuda.include_path``, ``dnn.base_path`` and ``dnn.bin_path``
HISTORY.txt:     to help configure Theano when CUDA and cuDNN can not be found automatically
HISTORY.txt:   - Added documentation for GPU float16 ops
HISTORY.txt:   - Support ``float16`` for ``GpuGemmBatch``
HISTORY.txt:   - Started to use ``float32`` precision for computations that don't support ``float16`` on GPU
HISTORY.txt: - Implemented ``topk`` and ``argtopk`` on CPU and GPU
HISTORY.txt: - Kept stack trace for optimizations in new GPU backend
HISTORY.txt: - Macro names provided for array properties are now standardized in both CPU and GPU C codes
HISTORY.txt:cuDNN (GPU):
HISTORY.txt: - Improved stack trace follow-up for GPU optimizations
HISTORY.txt:   To install it: ``conda install -c mila-udem -c mila-udem/label/pre theano pygpu``
HISTORY.txt:   - Fixed memory leak related to elemwise ops on GPU
HISTORY.txt: - Fixed pygpu detection
HISTORY.txt: - Implemented ``topk`` and ``argtopk`` on CPU and GPU
HISTORY.txt: - Support pygpu ``0.7``
HISTORY.txt:GPU:
HISTORY.txt: - Added a meta-optimizer to select the fastest GPU implementations for convolutions
HISTORY.txt: - Kept stack trace for optimizations in new GPU backend
HISTORY.txt: - Removed old GPU backend ``theano.sandbox.cuda``. New backend ``theano.gpuarray`` is now the official GPU backend
HISTORY.txt:   - ``cuda.enabled``
HISTORY.txt:   - ``gpuarray.sync``
HISTORY.txt:   - ``pycuda.init``
HISTORY.txt:GPU:
HISTORY.txt: - Prevent GPU initialization when not required
HISTORY.txt: - Added method ``my_theano_function.sync_shared()`` to help synchronize GPU Theano functions
HISTORY.txt: - Added useful stats for GPU in profile mode
HISTORY.txt: - Added GPU ops based on `magma library <http://icl.cs.utk.edu/magma/software/>`_:
HISTORY.txt: - Added ``GpuCublasTriangularSolve``
HISTORY.txt: - Added atomic addition and exchange for ``long long`` values in ``GpuAdvancedIncSubtensor1_dev20``
HISTORY.txt: - Support GPU SoftMax in both OpenCL and CUDA
HISTORY.txt: - Support offset parameter ``k`` for ``GpuEye``
HISTORY.txt: - ``CrossentropyCategorical1Hot`` and its gradient are now lifted to GPU
HISTORY.txt:   - Added new Theano flags ``cuda.include_path``, ``dnn.base_path`` and ``dnn.bin_path``
HISTORY.txt:     to help configure Theano when CUDA and cuDNN can not be found automatically.
HISTORY.txt:   - Added documentation for GPU float16 ops
HISTORY.txt:   - Support ``float16`` for ``GpuGemmBatch``
HISTORY.txt:   - Started to use ``float32`` precision for computations that don't support ``float16`` on GPU
HISTORY.txt: - Macro names provided for array properties are now standardized in both CPU and GPU C codes
HISTORY.txt: - New GPU back-end:
HISTORY.txt:   - Removed warp-synchronous programming to get good results with newer CUDA drivers
HISTORY.txt:   - More pooling support on GPU when cuDNN isn't available
HISTORY.txt:   - Using PCI bus ID of graphic cards for a better mapping between theano device number and nvidia-smi number
HISTORY.txt:   - Fixed offset error in ``GpuIncSubtensor``
HISTORY.txt:GPU:
HISTORY.txt: - Multiple-GPU, synchrone update (via platoon, use NCCL)
HISTORY.txt: - ``GPUMultinomialFromUniform`` op now supports multiple dtypes
HISTORY.txt: - Implemented ``GpuAdvancedSubtensor``
HISTORY.txt: - Added Abstract Ops for batch normalization that use cuDNN when available and pure Theano CPU/GPU alternatives otherwise
HISTORY.txt: - Added new Theano flag ``cuda.enabled``
HISTORY.txt: - Added new Theano flag ``nvcc.cudafe`` to enable faster compilation and import with old CUDA back-end
HISTORY.txt: - Added new Theano flag ``profiling.ignore_first_call``, useful to profile the new gpu back-end
HISTORY.txt: - Split op now has C code for CPU and GPU
HISTORY.txt: - Speed up argmax only on GPU (without also needing the max)
HISTORY.txt: - Added Jenkins (gpu tests run on pull requests in addition to daily buildbot)
HISTORY.txt: - New GPU back-end:
HISTORY.txt:   - Fixed offset error in GpuIncSubtensor
HISTORY.txt:   - Fixed indexing error in GpuAdvancedSubtensor for more than 2 dimensions
HISTORY.txt: - New GPU back-end:
HISTORY.txt:   - Removed warp-synchronous programming, to get good results with newer CUDA drivers
HISTORY.txt: - Better integration of Theano+libgpuarray packages into conda distribution
HISTORY.txt: - New GPU back-end:
HISTORY.txt:GPU:
HISTORY.txt: - GPUMultinomialFromUniform op now supports multiple dtypes
HISTORY.txt: - Added Abstract Ops for batch normalization that use cuDNN when available and pure Theano CPU/GPU alternatives otherwise
HISTORY.txt: - Added new Theano flag cuda.enabled
HISTORY.txt: - Split op now has C code for CPU and GPU
HISTORY.txt: - Jenkins (gpu tests run on PR in addition to daily buildbot)
HISTORY.txt: - New GPU back-end:
HISTORY.txt:   - better mapping between theano device number and nvidia-smi number, using the PCI bus ID of graphic cards
HISTORY.txt:   - More pooling support on GPU when cuDNN isn't there
HISTORY.txt:GPU:
HISTORY.txt: - Multiple-GPU, synchrone update (via platoon, use NCCL)
HISTORY.txt: - GpuAdvancedSubtensor in new back-end
HISTORY.txt: - Speed up argmax only on gpu (without also needing the max)
HISTORY.txt: - Add flag profiling.ignore_first_call, useful to profile the new gpu back-end
HISTORY.txt: - Integration of cuDNN for better GPU performance
HISTORY.txt: - optimizer=fast_compile moves computation to the GPU.
HISTORY.txt: - Better convolution on CPU and GPU. (CorrMM, cudnn, 3d conv, more parameter)
HISTORY.txt: - cnmem (better memory management on GPU)
HISTORY.txt: - Multi-GPU for data parallism via Platoon (https://github.com/mila-udem/platoon/)
HISTORY.txt: - New GPU back-end:
HISTORY.txt:   * Float16 new back-end (need cuda 7.5)
HISTORY.txt:   * Multi-GPU support in the same process
HISTORY.txt: - GpuJoin now supports negative axis
HISTORY.txt: - Fix GpuCumsum for negative axis
HISTORY.txt: - optimizer=fast_compile moves computation to the GPU
HISTORY.txt: - Make bincount work on GPU
HISTORY.txt: - SolveOp on GPU
HISTORY.txt: - theano.tensor.repeat works on GPU
HISTORY.txt: - BatchedDot on the GPU and faster on the CPU.
HISTORY.txt: - Faster batched_tensordot and make it work on GPU.
HISTORY.txt: - 3d conv via CorrMM on the GPU
HISTORY.txt: - theano.tensor.tile update (accept symbolic reps, work on GPU)
HISTORY.txt: - Faster SetSubtensor on the GPU.
HISTORY.txt: - Support more reduction pattern on the GPU.
HISTORY.txt: - GpuCrossentropySoftmaxArgmax1HotWithBias
HISTORY.txt: * Integration of cuDNN for 2D convolutions and pooling on supported GPUs
HISTORY.txt: * Better support for GPU on Windows
HISTORY.txt: * Mac: Fix wrong result of GpuDownsampleFactorMaxGrad on Mac OSX. (Pascal L.)
HISTORY.txt: * First version of the new GPU back-end available (Arnaud Bergeron, Frederic B.)
HISTORY.txt:     To use, use Theano flag device=cudaN or device=openclN, where N is a integer.
HISTORY.txt:   Now we raise an error if we try to profile when the gpu is enabled if we didn't set
HISTORY.txt: * When device=cpu and force_device=True, force that we disable the gpu. (Frederic B.)
HISTORY.txt: * GpuSoftmax[WithBias] for bigger row. (Frederic B.)
HISTORY.txt: * Make Erfinv work on the GPU (Guillaume Desjardin, Pascal L.)
HISTORY.txt: * Add cuda.unuse() to help tests that need to enable/disable the GPU (Frederic B.)
HISTORY.txt: * Allow a_cuda_ndarray += another_cuda_ndarray for 6d tensor. (Frederic B.)
HISTORY.txt: * Make the op ExtractDiag work on the GPU. (Frederic B.)
HISTORY.txt: * Add support for ScalarOp.c_support_code in GpuElemwise. (Frederic B.)
HISTORY.txt: * Also make the Psi function run on GPU. (Frederic B.)
HISTORY.txt: * Kron op: Speed up/generalize/GPU friendly. (Frederic B.)
HISTORY.txt: * Add gpu max for pattern (0, 1) and added all gpu max pattern for gpu min. (Frederic B.)
HISTORY.txt: * Add GpuEye (Frederic B.)
HISTORY.txt: * Make GpuCrossentropySoftmaxArgmax1HotWithBias and GpuCrossentropySoftmax1HotWithBiasDx work for bigger inputs (Frederic B., reported by Ryan Price)
HISTORY.txt: * Implement the mode ignore_borders for GpuImages2Neibs (Frederic B.)
HISTORY.txt: * Allow numpy.asarray(cuda_ndarray, dtype=...) (Frederic B.)
HISTORY.txt: * Speed up GpuJoin with c code (Ludwig Schmidt-Hackenberg, Frederic B.)
HISTORY.txt: * Faster GpuAdvancedIncSubtensor1 on Fermi GPU (and up) on matrix. (Vivek Kulkarni)
HISTORY.txt: * Faster GPUAdvancedIncSubtensor1 in some cases on all GPU (Vivek Kulkarni)
HISTORY.txt: * Fix infinite loop related to Scan on the GPU. (Pascal L.)
HISTORY.txt: * Fix some GPU compilation issue on Mac (John Yani, Frederic B.)
HISTORY.txt: * Fix local_gpu_multinomial optimization handling of broadcast information. (Frederic B., reported by Caglar)
HISTORY.txt: * Gpu reduction on all dimensions of a 4d tensor. (Frederic B., reported by Arjun Jain)
HISTORY.txt: * Fix compile and import errors on Windows including for the GPU. (Bogdan Budescu)
HISTORY.txt: * Fix GPU compilation on Windows (XterNalz)
HISTORY.txt: * Crash fix in the grad of GPU op in corner case (Pascal L.)
HISTORY.txt: * theano.misc.gnumpy_utils.garray_to_cudandarray() set strides correctly for dimensions of 1. (Frederic B., reported by Justin Bayer)
HISTORY.txt: * Fix opt crash/disabled with ifelse on the gpu (Frederic B, reported by Ryan Price)
HISTORY.txt: * Fix Error reporting with GpuConv (Frederic B., reported by Heng Luo and Nicolas Pinto)
HISTORY.txt: * Export some functions that work on CudaNdarray for windows (Frederic B.)
HISTORY.txt: * If the user specifies a -arch=sm_* value in the Theano flags for the gpu, don't add one (Frederic B., Pascal L.)
HISTORY.txt: * GPU memory leak fix.
HISTORY.txt: * tensor.{dot,tensordot} more complete/faster/GPU friendly.
HISTORY.txt: * Fix memory leak on the GPU in some corner cases with the Theano flags `allow_gc=False`. (Frederic B., reported by Jonas Gehring)
HISTORY.txt:   Work on the GPU too.
HISTORY.txt: * Implemented GpuContiguous.grad. (Ian G.)
HISTORY.txt: * DebugMode will now complain when the strides of CudaNdarray of dimensions of 1 are not 0. (Frederic B.)
HISTORY.txt: * GpuSum work with bigger shape when summing on the first dim on 3d tensor. (Frederic B., reported Chris Currivan)
HISTORY.txt: * Fix GpuSoftmax and GpuSoftmaxWithBias crash on GTX285. (Frederic B.)
HISTORY.txt: * Fix compilation problems on GPU on Windows. (Frederic B.)
HISTORY.txt: * Fix copy on the GPU with big shape for 4d tensor (Pascal L.)
HISTORY.txt: * GpuSubtensor didn't set the stride to 0 for dimensions of 1. This could lead to check failing later that caused a crash. (Frederic B., reported by vmichals)
HISTORY.txt: * GpuContiguous, GpuAlloc, GpuDownSampleGrad, Conv2d now check the preallocated outputs strides before using it. (Pascal L.)
HISTORY.txt: * GpuDownSample, GpuDownSampleGrad didn't work correctly with negative strides in their output due to problem with nvcc (Pascal L, reported by abalkin?)
HISTORY.txt: * The current GPU back-end have a new function CudaNdarray_prep_output(CudaNdarray ** arr, int nd, const int * dims) (Ian G)
HISTORY.txt: * Faster GpuIncSubtensor (Ian G.)
HISTORY.txt: * Faster copy on the GPU for 4d tensor. (Ian G.)
HISTORY.txt: * Enable inc_subtensor on the GPU when updating it with a float64 dtype. (Ian G.)
HISTORY.txt: * Move the convolution to the GPU when the image shape and logical image shape differ. (Frederic Bastien)
HISTORY.txt: * Fix crash when mixing shared variable on the GPU and sparse dot. (Pascal L.)
HISTORY.txt: * Bug fixes, crash fixes, CPU and GPU speed up.
HISTORY.txt: * Use GPU asynchronous functionality (Frederic B.)
HISTORY.txt: * set_subtensor(x[int vector], new_value) when moved to the GPU
HISTORY.txt:   was transformed into inc_subtensor on the GPU. Now we have a correct
HISTORY.txt:   (but slow) GPU implementation.
HISTORY.txt: * Correctly record the GPU device number used when we let the driver select it.
HISTORY.txt: * We warn when a user tries to use an old GPU with which Theano is untested.
HISTORY.txt: * GPU scan now works (does not crash) when there is a mixture of float32 and other dtypes.
HISTORY.txt: * "CudaNdarray[*] = ndarray" works in more cases (Frederic B.)
HISTORY.txt: * "CudaNdarray[*] += ndarray" works in more cases (Frederic B.)
HISTORY.txt: * We add dimensions to CudaNdarray to automatically broadcast more frequently.
HISTORY.txt: * Theano GPU variables, shared variables and constants now support <, <=,
HISTORY.txt:   > and >= similar to those not on the GPU.
HISTORY.txt: * Enable ifelse on the GPU. (Frederic B.)
HISTORY.txt: * Remove GPU transfer around specify_shape op. (Frederic B.)
HISTORY.txt:Speed up GPU:
HISTORY.txt: * Convolution on the GPU now checks the generation of the card to make
HISTORY.txt: * Faster GpuAdvancedSubtensor1, GpuSubtensor, GpuAlloc (Frederic B.)
HISTORY.txt: * We now pass the GPU architecture to nvcc when compiling (Frederic B.)
HISTORY.txt: * Now we use the GPU function async feature by default. (Frederic B.)
HISTORY.txt:   Set the environment variable `CUDA_LAUNCH_BLOCKING` to `1` to disable this
HISTORY.txt: * Faster creation of CudaNdarray objects (Frederic B.)
HISTORY.txt: * Now some Max reductions are implemented on the GPU. (Ian G.)
HISTORY.txt: * Installation documentation for Ubuntu (with GPU) (Frederic B., Matthias Zoehrer)
HISTORY.txt: * Fix how exception are raised in GPU code (James B.)
HISTORY.txt: * TensorType and CudaNdarrayType now have a value_zeros method that call CudaNdarray.zeros or
HISTORY.txt: * When importing theano on a computer without GPU with the Theano
HISTORY.txt:   flags 'device' or 'init_gpu_device' set to gpu* (Frederic B., reported by  Luo Heng)
HISTORY.txt: * GPU conv crash/slowdown on newer hardware (James B.)
HISTORY.txt: * Better error handling in GPU conv (Frederic B.)
HISTORY.txt: * GPU optimization that moves element-wise Ops to the GPU. Crash happened in
HISTORY.txt:   float32 (to compute them on the GPU).
HISTORY.txt: * GpuReshape in some particular case when the input is not contiguous
HISTORY.txt: * GpuSoftmaxWithBias with shape (0, N) with N > 1.
HISTORY.txt: * Fix crash on GPU when the GpuSubtensor didn't put the right stride
HISTORY.txt: * Fix scan crash that made it not run on the GPU in one case. (Guillaume D.)
HISTORY.txt: * GpuDownsampleFactorMax and its grad with inputs dimensions 0 and 1 bigger then 65535.
HISTORY.txt: * Potential crash due to parallel compilation when importing theano.sandbox.cuda
HISTORY.txt: * Fix crash of GpuSum when some dimensions shape was 0. (Frederic B.)
HISTORY.txt: * Remove GpuOuter as it is a subset of the new GpuGer (Frederic B.)
HISTORY.txt: * Better PyCUDA sharing of the GPU context.(fix crash at exit) (Frederic B.)
HISTORY.txt: * Theano with GPU works in some cases on Windows now. Still experimental. (Sebastian Urban)
HISTORY.txt: * Faster dot() call: New/Better direct call to cpu and gpu ger, gemv, gemm
HISTORY.txt: * When using a GPU, detect faulty nvidia drivers. This was detected
HISTORY.txt: * Theoretical bug: in some case we could have GPUSum return bad value.
HISTORY.txt: * Theano with GPU works in some cases on Windows now. Still experimental. (Sebastian Urban)
HISTORY.txt: * GpuAdvancedSubtensor1 supports broadcasted dimensions. (Frederic)
HISTORY.txt: * theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.device_properties() (Frederic)
HISTORY.txt: * theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info() return free and total gpu memory (Frederic)
HISTORY.txt:    * It makes use of gpu shared variable more transparent with theano.function updates and givens parameter.
HISTORY.txt: * a_CudaNdarray_object[*] = int, now works (Frederic)
HISTORY.txt: * tensor.tensordot can now be moved to GPU (Sander Dieleman,
HISTORY.txt: * The Theano flag "nvcc.fastmath" is now also used for the cuda_ndarray.cu file.
HISTORY.txt:   currently used only by cuda, but if we use libraries that are only headers,
HISTORY.txt: * Alloc, GpuAlloc are not always pre-computed (constant_folding optimization)
HISTORY.txt: * GPU crash with elemwise. (Frederic, some reported by Chris Currivan)
HISTORY.txt: * Compilation crash with amdlibm and the GPU. (Frederic)
HISTORY.txt: * GPU compilation crash on MacOS X. (Olivier)
HISTORY.txt: * Crashes of blas functions (Gemv on CPU; Ger, Gemv and Gemm on GPU)
HISTORY.txt:   when matrices had non-unit stride in both dimensions (CPU and GPU),
HISTORY.txt:   or when matrices had negative strides (GPU only). In those cases,
HISTORY.txt: * CURAND_RandomStreams for uniform and normal (not picklable, GPU only) (James)
HISTORY.txt: * Refactored GPU installation of Theano. (Olivier)
HISTORY.txt: * More tests for join on the GPU and CPU. (Frederic)
HISTORY.txt: * Do not request to load the GPU module by default in scan module. (Razvan)
HISTORY.txt: * better pycuda tests (Frederic)
HISTORY.txt:GPU:
HISTORY.txt: * PyCUDA/CUDAMat/Gnumpy/Theano bridge and `documentation <http://deeplearning.net/software/theano/tutorial/gpu_data_convert.html>`_.
HISTORY.txt:   * New function to easily convert pycuda GPUArray object to and from CudaNdarray object
HISTORY.txt:   * Fixed a bug if you crated a view of a manually created CudaNdarray that are view of GPUArray.
HISTORY.txt: * renamed config option cuda.nvccflags -> nvcc.flags
HISTORY.txt: * Allow GpuSoftmax and GpuSoftmaxWithBias to work with bigger input.
HISTORY.txt: * In one case an AdvancedSubtensor1 could be converted to a GpuAdvancedIncSubtensor1 insted of GpuAdvancedSubtensor1.
HISTORY.txt: * Compilation crash with CUDA 4
HISTORY.txt: * GPU:
HISTORY.txt:   * Compilation crash fixed with CUDA 4.0
HISTORY.txt: * CudaNdarray_new_null is deprecated in favour of CudaNdarray_New
HISTORY.txt:    a numpy.ndarray, for a GPU variable, a CudaNdarray, for instance)
HISTORY.txt: * CudaNdarray_new_null is deprecated in favour of CudaNdarray_New
HISTORY.txt: * In CudaNdarray.__{iadd,idiv}__, when it is not implemented, return the error.
HISTORY.txt: * Fixed memory leak in error handling on GPU-to-host copy
HISTORY.txt:GPU:
HISTORY.txt: * Move to the gpu fused elemwise that have other dtype then float32 in them
HISTORY.txt:   * This allow to move elemwise comparisons to the GPU if we cast it to
HISTORY.txt: * Implemented CudaNdarray.ndim to have the same interface in ndarray.
HISTORY.txt: * Fixed slowdown caused by multiple chained views on CudaNdarray objects
HISTORY.txt: * CudaNdarray_alloc_contiguous changed so as to never try to free
HISTORY.txt: * Safer decref behaviour in CudaNdarray in case of failed allocations
HISTORY.txt: * New GPU implementation of tensor.basic.outer
HISTORY.txt: * Multinomial random variates now available on GPU
HISTORY.txt:   work inplace, gpu work)
HISTORY.txt: * cuda.root inferred if nvcc is on the path, otherwise defaults to
HISTORY.txt:   /usr/local/cuda
HISTORY.txt: * CUDA devices 4 - 16 should now be available if present.
HISTORY.txt: * Better commenting of cuda_ndarray.cu
HISTORY.txt: * Reuse test for subtensor of tensor for gpu tensor(more gpu test)
HISTORY.txt: * Better test of copies in CudaNdarray
HISTORY.txt: * Some tests are now run whenever cuda is available and not just when it has
HISTORY.txt: * The random number generator in theano/sandbox/rng_mrg.py did not always return the same sequence of number on the CPU and GPU.
HISTORY.txt: * In GpuConv, errors in conv_patch_stack_reduce when the entire kernel doesn't fit into shared memory.
HISTORY.txt: * Compilation crash for GpuElemwise with tensor with high number of dimensions (~6 or more).
HISTORY.txt: * Output shape is now computed correctly for matrix-vector multiplication on GPU.
HISTORY.txt: * In GpuSum, bug in calculation of n_blocks for the 10 pattern. (Sum on the row of a matrix)
HISTORY.txt: * Some segfault at exit with GPU code.
HISTORY.txt:GPU:
HISTORY.txt: * cuda_shared.value = X now works inplace!
HISTORY.txt:     * cuda_shared_var.set_value(new_ndarray) will overwrite the old value inplace in the most common case.
HISTORY.txt: * Allow to create a CudaNdarraySharedVariable from a CudaNdarray.
HISTORY.txt: * New init_gpu_device theano flags.
HISTORY.txt: * Fuse GpuElemwise more often (in the case where there are so many inputs that fusing them all would bust the 256 bytes limit of parameter to gpu function).
HISTORY.txt: * CPU join of only 1 element that was not moved to the GPU.
HISTORY.txt: * New 3D convolution ops, with CPU and GPU implementations.
HISTORY.txt: * Documented lib.amdlibm and (new) init_gpu_device config variables.
HISTORY.txt: * The cuda documentation is now generated on the web server.
HISTORY.txt: * Better testing of GPU convolution nets.
HISTORY.txt: * GPU code using NVIDIA's CUDA framework is now generated for many Ops.
DESCRIPTION.txt: * **transparent use of a GPU:** perform data-intensive computations up to 140x faster than on a CPU (support for float32 only).
EMAIL.txt: * transparent use of a GPU: perform data-intensive computations much faster than on a CPU.
bin/theano_nose.py:    # error if device==gpu.
bin/theano_nose.py:                                 " This will also run GPU tests when possible.\n"
bin/theano_nose.py:                                 " If you want GPU-related tests to run on a"
bin/theano_nose.py:                                 " specific GPU device, and not the default one,"
bin/theano_nose.py:                                 " you should use the init_gpu_device theano flag.")

```
