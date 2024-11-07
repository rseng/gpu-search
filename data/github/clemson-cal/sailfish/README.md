# https://github.com/clemson-cal/sailfish

```console
dist/setup.cfg:description = GPU accelerated astrophysical gasdynamics code
ideas/solver.py:                int nccl = (i + 0) * si + (j + 0) * sj + (k - 1) * sk;
ideas/solver.py:                        double yl = y[nccl + q * sq];
ideas/solver.py:def make_stream(hardware: str, gpu_streams: str):
ideas/solver.py:    if hardware == "gpu":
ideas/solver.py:        from cupy.cuda import Stream
ideas/solver.py:        if gpu_streams == "per-thread":
ideas/solver.py:        if gpu_streams == "per-patch":
ideas/solver.py:    if hardware == "gpu":
ideas/solver.py:    gpu_streams = strategy.gpu_streams
ideas/solver.py:        stream = make_stream(hardware, gpu_streams)
ideas/kernels.py:Enables interaction with embedded C or CUDA code.
ideas/kernels.py:KERNEL_DISABLE_GPU_MODE = False
ideas/kernels.py:KERNEL_DEFINE_MACROS_GPU = R"""
ideas/kernels.py:#define GPU_MODE
ideas/kernels.py:            if mode == "gpu":
ideas/kernels.py:                from cupy.cuda.runtime import deviceSynchronize
ideas/kernels.py:    disable_gpu_mode=None,
ideas/kernels.py:    global KERNEL_DISABLE_GPU_MODE
ideas/kernels.py:    if disable_gpu_mode:
ideas/kernels.py:        KERNEL_DISABLE_GPU_MODE = True
ideas/kernels.py:        if default_exec_mode not in ("cpu", "gpu"):
ideas/kernels.py:            raise ValueError("execution mode must be cpu or gpu")
ideas/kernels.py:    logger.debug(f"KERNEL_DISABLE_GPU_MODE={KERNEL_DISABLE_GPU_MODE}")
ideas/kernels.py:def to_gpu_args(args):
ideas/kernels.py:    Return a generator that yields arguments suitable for args to a GPU kernel
ideas/kernels.py:    installed, so compilation of the respective CPU or GPU extension was not
ideas/kernels.py:    A CPU or GPU extension module that could not be created for some reason.
ideas/kernels.py:        # It should not be fatal if cffi is not available, since GPU kernels
ideas/kernels.py:def gpu_extension(code, name, define_macros=list()):
ideas/kernels.py:    if KERNEL_DISABLE_GPU_MODE:
ideas/kernels.py:        logger.debug(f"KERNEL_DISABLE_GPU_MODE=True; skip GPU extension")
ideas/kernels.py:        return MissingModule(RuntimeError("invoke skipped GPU extension"))
ideas/kernels.py:        from cupy.cuda.compiler import CompileException
ideas/kernels.py:        code = KERNEL_DEFINE_MACROS_GPU + code
ideas/kernels.py:        logger.info(f"compile GPU module {name}[{define_str}]")
ideas/kernels.py:        logger.debug(f"{e}; skip GPU extension")
ideas/kernels.py:        logger.warning(f"{e}; skip GPU extension")
ideas/kernels.py:        return MissingModule(RuntimeError(f"invoke failed GPU extension"))
ideas/kernels.py:def gpu_extension_function(module, stub):
ideas/kernels.py:    gpu_func = module.get_function(stub.__name__)
ideas/kernels.py:            ValueError(f"GPU kernel {stub.__name__} may not return a value")
ideas/kernels.py:        gpu_func(nb, bs, tuple(to_gpu_args(pyargs)))
ideas/kernels.py:    wrapper.__gpu_func__ = gpu_func
ideas/kernels.py:def extension_function(cpu_module, gpu_module, stub):
ideas/kernels.py:    gpu_func = gpu_extension_function(gpu_module, stub)
ideas/kernels.py:        if exec_mode == "gpu":
ideas/kernels.py:            return gpu_func(*args)
ideas/kernels.py:    wrapper.__gpu_func__ = getattr(gpu_func, "__gpu_func__", None)
ideas/kernels.py:            gpu_module = gpu_extension(code, name, self._define_macros)
ideas/kernels.py:            self.inject_modules(cpu_module, gpu_module)
ideas/kernels.py:    def inject_modules(self, cpu_module, gpu_module):
ideas/kernels.py:        self._func = extension_function(cpu_module, gpu_module, self._stub)
ideas/kernels.py:    some way, and is compiled to either CPU or GPU code. The wrapper function
ideas/kernels.py:    returned is a proxy to the respective CPU and GPU compiled extension
ideas/kernels.py:    argument `exec_mode='cpu'|'gpu'` to the wrapper function. It defaults to
ideas/kernels.py:    set with `configure_kernel_module(default_exec_mode='gpu')`.
ideas/kernels.py:        wrapper.__gpu_func__ = kernel_data._func.__gpu_func__
ideas/kernels.py:        gpu_module = gpu_extension(code, name, define_macros)
ideas/kernels.py:            k.inject_modules(cpu_module, gpu_module)
ideas/kernels.py:    gpu_func = kernel.__gpu_func__
ideas/kernels.py:        metadata["num_regs"] = gpu_func.num_regs
ideas/kernels.py:        choices=["cpu", "gpu"],
ideas/kernels.py:    if args.exec_mode == "gpu":
ideas/kernels.py:    # functions that return a value cannot be GPU kernels, but they can be CPU
ideas/kernels.py:    if args.exec_mode != "gpu":
ideas/kernels.py:    if args.exec_mode != "gpu":
ideas/sailfish:        table.add_row("[blue]cupy", have("cupy"), "GPU acceleration", "cupy-cuda116")
ideas/sailfish:        - [ ] multi-GPU support
ideas/sailfish:        description="sailfish is a GPU-accelerated astrophysical gasdynamics code",
ideas/config.py:    hardware:    compute device [cpu|gpu]
ideas/config.py:    num_patches: decompose domain to enable threads, streams, or multiple GPU's
ideas/config.py:    gpu_streams: use the per-thread-default-stream, or one stream per grid patch
ideas/config.py:    hardware: Literal["cpu", "gpu"] = "cpu"
ideas/config.py:    gpu_streams: Literal["per-thread", "per-patch"] = "per-thread"
ideas/config.py:        choices=Strategy.type_args("gpu_streams"),
ideas/config.py:        help=Strategy.describe("gpu_streams"),
ideas/config.py:        dest="strategy.gpu_streams",
ideas/system.py:        from cupy.cuda.runtime import getDeviceProperties, getDeviceCount
ideas/system.py:        gpu_info = list(
ideas/system.py:        gpu_info = None
ideas/system.py:        gpu_info=gpu_info,
doc/source/index.rst:Sailfish is a GPU-accelerated astrophysical gasdynamics code.
doc/source/kernels.rst:Sailfish has a module dedicated to generating CPU-GPU agnostic compute
doc/source/kernels.rst:floating point data, and may be parallelized using OpenMP, CUDA, or ROCm.
doc/source/kernels.rst:indicates sequential processing, as opposed to parallelized OpenMP or GPU
doc/source/kernels.rst:- :code:`gpu` kernel body is executed once per GPU thread; compiled with `cupy`
doc/source/kernels.rst:`__pycache__` directory. No caching is done for GPU builds.
sailfish/kernel/library.py:Defines a `Library` utility class to encapulsate CPU/GPU compiled kernels.
sailfish/kernel/library.py:reuse based on the SHA value of the source code and #define macros. GPU
sailfish/kernel/library.py:modules are JIT-compiled with cupy. No caching is presently done for the GPU
sailfish/kernel/library.py:#define EXEC_GPU 2
sailfish/kernel/library.py:#if (EXEC_MODE != EXEC_GPU)
sailfish/kernel/library.py:#elif (EXEC_MODE == EXEC_GPU)
sailfish/kernel/library.py:    Builds and maintains (in memory) a CPU or GPU dynamically compiled module.
sailfish/kernel/library.py:            self.cpu_mode = mode != "gpu"
sailfish/kernel/library.py:                self.load_gpu_module(code, define_macros)
sailfish/kernel/library.py:    def load_gpu_module(self, code, define_macros):
sailfish/kernel/system.py:    if mode is "gpu" then `cupy` is returned. The `cupy` documentation
sailfish/kernel/system.py:    CPU-GPU agnostic code.
sailfish/kernel/system.py:    elif mode == "gpu":
sailfish/kernel/system.py:        raise ValueError(f"unknown execution mode {mode}, must be [cpu|omp|gpu]")
sailfish/kernel/system.py:    If `mode` is "gpu", then a specific device id may be provided to specify
sailfish/kernel/system.py:    the GPU onto which kernel launches should be spawned.
sailfish/kernel/system.py:    elif mode == "gpu":
sailfish/kernel/system.py:        from cupy.cuda import Device
sailfish/kernel/system.py:    elif mode == "gpu":
sailfish/kernel/system.py:        from cupy.cuda.runtime import getDeviceCount
sailfish/kernel/system.py:    if mode == "gpu":
sailfish/kernel/system.py:        from cupy.cuda.runtime import getDeviceCount, getDeviceProperties
sailfish/kernel/system.py:        gpu_devices = ":".join(
sailfish/kernel/system.py:        logger.info(f"gpu devices: {num_devices}x {gpu_devices}")
sailfish/kernel/system.py:        if mode == "gpu":
sailfish/kernel/system.py:            from cupy.cuda.runtime import deviceSynchronize
sailfish/kernel/__init__.py:A Python module to facilitate JIT-compiled CPU-GPU agnostic compute kernels.
sailfish/kernel/__init__.py:for GPU execution using a CUDA or ROCm compiler via cupy.
sailfish/driver.py:        description="sailfish is a GPU-accelerated astrophysical gasdynamics code",
sailfish/driver.py:        choices=["cpu", "omp", "gpu"],
sailfish/driver.py:        "--use-gpu",
sailfish/driver.py:        const="gpu",
sailfish/driver.py:        help="gpu acceleration",
sailfish/solvers/srhd_2d.c:    #if (EXEC_MODE != EXEC_GPU)
sailfish/solvers/srhd_1d.c:    #if (EXEC_MODE != EXEC_GPU)
sailfish/solvers/scdg_1d.py:            of its time derivative (optionally) which could instead be recomputed e.g. on a GPU
sailfish/solvers/scdg_1d.py:            of its time derivative (optionally) which could instead be recomputed e.g. on a GPU
sailfish/solvers/scdg_1d.py:            of its time derivative (optionally) which could instead be recomputed e.g. on a GPU
sailfish/solvers/scdg_1d.py:            of its time derivative (optionally) which could instead be recomputed e.g. on a GPU
scripts/test_kernel.py:    parser.add_argument("--mode", default="cpu", choices=["cpu", "omp", "gpu"])
scripts/test_kernel.py:    if args.mode == "gpu":
scripts/test_kernel.py:    if args.mode == "gpu":
README:Sailfish is a GPU-accelerated astrophysical gasdynamics code

```
