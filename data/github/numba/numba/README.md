# https://github.com/numba/numba

```console
setup.py:    ext_cuda_extras = Extension(name='numba.cuda.cudadrv._extras',
setup.py:                                sources=['numba/cuda/cudadrv/_extras.c'],
setup.py:                   ext_cuda_extras, ext_devicearray]
setup.py:        "numba.cuda.tests.data": ["*.ptx", "*.cu"],
setup.py:        "numba.cuda.tests.doc_examples.ffi": ["*.cu"],
setup.py:        "numba.cuda" : ["cpp_function_wrappers.cu", "cuda_fp16.h",
setup.py:                        "cuda_fp16.hpp"]
README.rst:parallelization of loops, generation of GPU-accelerated code, and creation of
numba/_typeof.cpp:/* CUDA device array API */
numba/_typeof.cpp:    /* Subtype of CUDA device array */
numba/core/typed_passes.py:                    # e.g. Calls to CUDA Intrinsic have no mapped type
numba/core/datamodel/packer.py:    OpenCL, CUDA), flattening composite argument types simplifes the call
numba/core/typing/templates.py:        # in which to register lowering implementations, the CUDA target
numba/core/typing/templates.py:        # modules, _AND_ CUDA also makes use of the same impl, then it's
numba/core/typing/templates.py:        # required that the registry in use is one that CUDA borrows from. This
numba/core/typing/templates.py:        # In case the target has swapped, e.g. cuda borrowing cpu, refresh to
numba/core/typing/context.py:            if numba.cuda.is_cuda_array(val):
numba/core/typing/context.py:                return typeof(numba.cuda.as_cuda_array(val, sync=False),
numba/core/callconv.py:    A minimal calling convention, suitable for e.g. GPU targets.
numba/core/config.py:    Parse CUDA compute capability version string.
numba/core/config.py:        global CUDA_USE_NVIDIA_BINDING
numba/core/config.py:        if CUDA_USE_NVIDIA_BINDING:  # noqa: F821
numba/core/config.py:                import cuda  # noqa: F401
numba/core/config.py:                msg = ("CUDA Python bindings requested (the environment "
numba/core/config.py:                       "variable NUMBA_CUDA_USE_NVIDIA_BINDING is set), "
numba/core/config.py:                CUDA_USE_NVIDIA_BINDING = False
numba/core/config.py:            if CUDA_PER_THREAD_DEFAULT_STREAM:  # noqa: F821
numba/core/config.py:                warnings.warn("PTDS support is handled by CUDA Python when "
numba/core/config.py:                              "using the NVIDIA binding. Please set the "
numba/core/config.py:                              "CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM to 1 "
numba/core/config.py:        # under utilize the GPU due to low occupancy. On by default.
numba/core/config.py:        CUDA_LOW_OCCUPANCY_WARNINGS = _readenv(
numba/core/config.py:            "NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS", int, 1)
numba/core/config.py:        # Whether to use the official CUDA Python API Bindings
numba/core/config.py:        CUDA_USE_NVIDIA_BINDING = _readenv(
numba/core/config.py:            "NUMBA_CUDA_USE_NVIDIA_BINDING", int, 0)
numba/core/config.py:        # CUDA Configs
numba/core/config.py:        CUDA_WARN_ON_IMPLICIT_COPY = _readenv(
numba/core/config.py:            "NUMBA_CUDA_WARN_ON_IMPLICIT_COPY", int, 1)
numba/core/config.py:        # Force CUDA compute capability to a specific version
numba/core/config.py:        FORCE_CUDA_CC = _readenv("NUMBA_FORCE_CUDA_CC", _parse_cc, None)
numba/core/config.py:        CUDA_DEFAULT_PTX_CC = _readenv("NUMBA_CUDA_DEFAULT_PTX_CC", _parse_cc,
numba/core/config.py:        # Disable CUDA support
numba/core/config.py:        DISABLE_CUDA = _readenv("NUMBA_DISABLE_CUDA",
numba/core/config.py:        # Enable CUDA simulator
numba/core/config.py:        ENABLE_CUDASIM = _readenv("NUMBA_ENABLE_CUDASIM", int, 0)
numba/core/config.py:        # CUDA logging level
numba/core/config.py:        CUDA_LOG_LEVEL = _readenv("NUMBA_CUDA_LOG_LEVEL", str, '')
numba/core/config.py:        # Include argument values in the CUDA Driver API logs
numba/core/config.py:        CUDA_LOG_API_ARGS = _readenv("NUMBA_CUDA_LOG_API_ARGS", int, 0)
numba/core/config.py:        # Maximum number of pending CUDA deallocations (default: 10)
numba/core/config.py:        CUDA_DEALLOCS_COUNT = _readenv("NUMBA_CUDA_MAX_PENDING_DEALLOCS_COUNT",
numba/core/config.py:        # Maximum ratio of pending CUDA deallocations to capacity (default: 0.2)
numba/core/config.py:        CUDA_DEALLOCS_RATIO = _readenv("NUMBA_CUDA_MAX_PENDING_DEALLOCS_RATIO",
numba/core/config.py:        CUDA_ARRAY_INTERFACE_SYNC = _readenv("NUMBA_CUDA_ARRAY_INTERFACE_SYNC",
numba/core/config.py:        # Path of the directory that the CUDA driver libraries are located
numba/core/config.py:        CUDA_DRIVER = _readenv("NUMBA_CUDA_DRIVER", str, '')
numba/core/config.py:        # Buffer size for logs produced by CUDA driver operations (e.g.
numba/core/config.py:        CUDA_LOG_SIZE = _readenv("NUMBA_CUDA_LOG_SIZE", int, 1024)
numba/core/config.py:        CUDA_VERBOSE_JIT_LOG = _readenv("NUMBA_CUDA_VERBOSE_JIT_LOG", int, 1)
numba/core/config.py:        CUDA_PER_THREAD_DEFAULT_STREAM = _readenv(
numba/core/config.py:            "NUMBA_CUDA_PER_THREAD_DEFAULT_STREAM", int, 0)
numba/core/config.py:        CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = _readenv(
numba/core/config.py:            "NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY", int, 0)
numba/core/config.py:        # Location of the CUDA include files
numba/core/config.py:            cuda_path = os.environ.get('CUDA_PATH')
numba/core/config.py:            if cuda_path:
numba/core/config.py:                default_cuda_include_path = os.path.join(cuda_path, "include")
numba/core/config.py:                default_cuda_include_path = "cuda_include_not_found"
numba/core/config.py:            default_cuda_include_path = os.path.join(os.sep, 'usr', 'local',
numba/core/config.py:                                                     'cuda', 'include')
numba/core/config.py:        CUDA_INCLUDE_PATH = _readenv("NUMBA_CUDA_INCLUDE_PATH", str,
numba/core/config.py:                                     default_cuda_include_path)
numba/core/config.py:        CUDA_DEBUGINFO_DEFAULT = _readenv("NUMBA_CUDA_DEBUGINFO", int, 0)
numba/core/config.py:        # CUDA Memory management
numba/core/config.py:        CUDA_MEMORY_MANAGER = _readenv("NUMBA_CUDA_MEMORY_MANAGER", str,
numba/core/decorators.py:        if config.ENABLE_CUDASIM and target == 'cuda':
numba/core/decorators.py:            from numba import cuda
numba/core/decorators.py:            return cuda.jit(func)
numba/core/target_extension.py:class GPU(Generic):
numba/core/target_extension.py:    """Mark the target as GPU, i.e. suitable for compilation on a GPU
numba/core/target_extension.py:class CUDA(GPU):
numba/core/target_extension.py:    """Mark the target as CUDA.
numba/core/target_extension.py:target_registry['GPU'] = GPU
numba/core/target_extension.py:target_registry['gpu'] = GPU
numba/core/target_extension.py:target_registry['CUDA'] = CUDA
numba/core/target_extension.py:target_registry['cuda'] = CUDA
numba/_dispatcher.cpp:   CUDA targets. Its main responsibilities are:
numba/_dispatcher.cpp:    /* The cuda kwarg is a temporary addition until CUDA overloads are compiled
numba/_dispatcher.cpp:        (char*)"cuda",
numba/_dispatcher.cpp:    int cuda = 0;
numba/_dispatcher.cpp:                                     &cfunc, &objectmode, &cuda)) {
numba/_dispatcher.cpp:    if (!cuda && !PyObject_TypeCheck(cfunc, &PyCFunction_Type) ) {
numba/_dispatcher.cpp:/* A copy of compile_and_invoke, that only compiles. This is needed for CUDA
numba/_dispatcher.cpp: * rather than compiled functions. Once CUDA overloads are compiled functions,
numba/_dispatcher.cpp: * cuda_compile_only can be removed. */
numba/_dispatcher.cpp:cuda_compile_only(Dispatcher *self, PyObject *args, PyObject *kws, PyObject *locals)
numba/_dispatcher.cpp:   This is because CUDA functions are, at present, _Kernel objects rather than
numba/_dispatcher.cpp:Dispatcher_cuda_call(Dispatcher *self, PyObject *args, PyObject *kws)
numba/_dispatcher.cpp:            retval = cuda_compile_only(self, args, kws, locals);
numba/_dispatcher.cpp:        retval = cuda_compile_only(self, args, kws, locals);
numba/_dispatcher.cpp:    { "_cuda_call", (PyCFunction)Dispatcher_cuda_call,
numba/_dispatcher.cpp:      METH_VARARGS | METH_KEYWORDS, "CUDA call resolution" },
numba/tests/test_runtests.py:from numba import cuda
numba/tests/test_runtests.py:        # CUDA should be included by default
numba/tests/test_runtests.py:        self.assertTrue(any('numba.cuda.tests.' in line for line in lines))
numba/tests/test_runtests.py:    def test_cuda(self):
numba/tests/test_runtests.py:        # Even without CUDA enabled, there is at least one test
numba/tests/test_runtests.py:        # (in numba.cuda.tests.nocuda)
numba/tests/test_runtests.py:        minsize = 100 if cuda.is_available() else 1
numba/tests/test_runtests.py:        self.check_testsuite_size(['numba.cuda.tests'], minsize)
numba/tests/test_runtests.py:    @unittest.skipIf(not cuda.is_available(), "NO CUDA")
numba/tests/test_runtests.py:    def test_cuda_submodules(self):
numba/tests/test_runtests.py:        self.check_listing_prefix('numba.cuda.tests.cudadrv')
numba/tests/test_runtests.py:        self.check_listing_prefix('numba.cuda.tests.cudapy')
numba/tests/test_runtests.py:        self.check_listing_prefix('numba.cuda.tests.nocuda')
numba/tests/test_runtests.py:        self.check_listing_prefix('numba.cuda.tests.cudasim')
numba/tests/test_boundscheck.py:from numba.cuda.testing import SerialMixin
numba/tests/test_boundscheck.py:from numba import typeof, cuda, njit
numba/tests/test_boundscheck.py:class TestNoCudaBoundsCheck(SerialMixin, TestCase):
numba/tests/test_boundscheck.py:    @unittest.skipIf(not cuda.is_available(), "NO CUDA")
numba/tests/test_boundscheck.py:    def test_no_cuda_boundscheck(self):
numba/tests/test_boundscheck.py:            @cuda.jit(boundscheck=True)
numba/tests/test_boundscheck.py:        @cuda.jit(boundscheck=False)
numba/tests/test_boundscheck.py:        @cuda.jit
numba/tests/test_boundscheck.py:        if not config.ENABLE_CUDASIM:
numba/tests/test_alignment.py:# See also numba.cuda.tests.test_alignment
numba/tests/test_alignment.py:        # Unlike the CUDA target, this will not generate an error
numba/tests/test_record_dtype.py:        # the following is the definition of int4 vector type from pyopencl
numba/tests/test_target_extension.py:the CPU but is part of the GPU class of target. The DPU target has deliberately
numba/tests/test_target_extension.py:    GPU,
numba/tests/test_target_extension.py:# Define a new target, this target extends GPU, this places the DPU in the
numba/tests/test_target_extension.py:# target hierarchy as a type of GPU.
numba/tests/test_target_extension.py:class DPU(GPU):
numba/tests/test_target_extension.py:    def test_specialise_gpu(self):
numba/tests/test_target_extension.py:        @overload(my_func, target="gpu")
numba/tests/test_target_extension.py:        @overload(my_func, target="gpu")
numba/tests/test_target_extension.py:        # only create a cuda specialisation
numba/tests/test_target_extension.py:        @overload(my_func, target='cuda')
numba/tests/test_target_extension.py:        def ol_my_func_cuda(x):
numba/tests/test_target_extension.py:        def cuda_target_attr_use(res, dummy):
numba/tests/npyufunc/test_update_inplace.py:        # writable_args are not supported for target='cuda'
numba/tests/npyufunc/test_update_inplace.py:                        target='cuda')(py_replace_2nd)
numba/tests/npyufunc/test_ufunc.py:    # CudaVectorize,
numba/tests/__init__.py:    # Numba CUDA tests are located in a separate directory:
numba/tests/__init__.py:    cuda_dir = join(dirname(dirname(__file__)), 'cuda/tests')
numba/tests/__init__.py:    suite.addTests(loader.discover(cuda_dir))
numba/tests/test_import.py:                   'numba.cuda',
numba/testing/main.py:def cuda_sensitive_mtime(x):
numba/testing/main.py:    Return a key for sorting tests bases on mtime and test name. For CUDA
numba/testing/main.py:    tests, interleaving tests from different classes is dangerous as the CUDA
numba/testing/main.py:    CUDA tests the key prioritises the test module and class ahead of the
numba/testing/main.py:    from numba.cuda.testing import CUDATestCase
numba/testing/main.py:    if CUDATestCase in cls.mro():
numba/testing/main.py:        self._test_list.sort(key=cuda_sensitive_mtime)
numba/testing/main.py:        run.sort(key=cuda_sensitive_mtime)
numba/testing/main.py:    "numba.cuda.tests.cudapy.test_libdevice.TestLibdeviceCompilation",
numba/testing/main.py:        tests.sort(key=cuda_sensitive_mtime)
numba/misc/findlib.py:        # on windows, historically `DLLs` has been used for CUDA libraries,
numba/misc/findlib.py:        # since approximately CUDA 9.2, `Library\bin` has been used.
numba/misc/numba_sysinfo.py:from numba import cuda as cu, __version__ as version_number
numba/misc/numba_sysinfo.py:from numba.cuda import cudadrv
numba/misc/numba_sysinfo.py:from numba.cuda.cudadrv.driver import driver as cudriver
numba/misc/numba_sysinfo.py:from numba.cuda.cudadrv.runtime import runtime as curuntime
numba/misc/numba_sysinfo.py:# CUDA info
numba/misc/numba_sysinfo.py:_cu_target_impl = 'CUDA Target Impl'
numba/misc/numba_sysinfo.py:_cu_dev_init = 'CUDA Device Init'
numba/misc/numba_sysinfo.py:_cu_drv_ver = 'CUDA Driver Version'
numba/misc/numba_sysinfo.py:_cu_rt_ver = 'CUDA Runtime Version'
numba/misc/numba_sysinfo.py:_cu_nvidia_bindings = 'NVIDIA CUDA Bindings'
numba/misc/numba_sysinfo.py:_cu_nvidia_bindings_used = 'NVIDIA CUDA Bindings In Use'
numba/misc/numba_sysinfo.py:_cu_detect_out, _cu_lib_test = 'CUDA Detect Output', 'CUDA Lib Test'
numba/misc/numba_sysinfo.py:_cu_mvc_available = 'NVIDIA CUDA Minor Version Compatibility Available'
numba/misc/numba_sysinfo.py:_cu_mvc_needed = 'NVIDIA CUDA Minor Version Compatibility Needed'
numba/misc/numba_sysinfo.py:_cu_mvc_in_use = 'NVIDIA CUDA Minor Version Compatibility In Use'
numba/misc/numba_sysinfo.py:    # CUDA information
numba/misc/numba_sysinfo.py:        msg_not_found = "CUDA driver library cannot be found"
numba/misc/numba_sysinfo.py:        msg_disabled_by_user = "CUDA is disabled"
numba/misc/numba_sysinfo.py:        msg_end = " or no CUDA enabled devices are present."
numba/misc/numba_sysinfo.py:        msg_generic_problem = "CUDA device initialisation problem."
numba/misc/numba_sysinfo.py:        _warning_log.append("Warning (cuda): %s\nException class: %s" %
numba/misc/numba_sysinfo.py:                cudadrv.libs.test()
numba/misc/numba_sysinfo.py:                from cuda import cuda  # noqa: F401
numba/misc/numba_sysinfo.py:                nvidia_bindings_available = True
numba/misc/numba_sysinfo.py:                nvidia_bindings_available = False
numba/misc/numba_sysinfo.py:            sys_info[_cu_nvidia_bindings] = nvidia_bindings_available
numba/misc/numba_sysinfo.py:            nv_binding_used = bool(cudadrv.driver.USE_NV_BINDING)
numba/misc/numba_sysinfo.py:            sys_info[_cu_nvidia_bindings_used] = nv_binding_used
numba/misc/numba_sysinfo.py:                config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY)
numba/misc/numba_sysinfo.py:                "Warning (cuda): Probing CUDA failed "
numba/misc/numba_sysinfo.py:                f"(cuda) {type(e)}: {e}")
numba/misc/numba_sysinfo.py:        ("__CUDA Information__",),
numba/misc/numba_sysinfo.py:        ("CUDA Target Implementation", info.get(_cu_target_impl, '?')),
numba/misc/numba_sysinfo.py:        ("CUDA Device Initialized", info.get(_cu_dev_init, '?')),
numba/misc/numba_sysinfo.py:        ("CUDA Driver Version", info.get(_cu_drv_ver, '?')),
numba/misc/numba_sysinfo.py:        ("CUDA Runtime Version", info.get(_cu_rt_ver, '?')),
numba/misc/numba_sysinfo.py:        ("CUDA NVIDIA Bindings Available", info.get(_cu_nvidia_bindings, '?')),
numba/misc/numba_sysinfo.py:        ("CUDA NVIDIA Bindings In Use",
numba/misc/numba_sysinfo.py:         info.get(_cu_nvidia_bindings_used, '?')),
numba/misc/numba_sysinfo.py:        ("CUDA Minor Version Compatibility Available",
numba/misc/numba_sysinfo.py:        ("CUDA Minor Version Compatibility Needed",
numba/misc/numba_sysinfo.py:        ("CUDA Minor Version Compatibility In Use",
numba/misc/numba_sysinfo.py:        ("CUDA Detect Output:",),
numba/misc/numba_sysinfo.py:        ("CUDA Libraries Test Output:",),
numba/np/ufunc/_internal.c:} PyUFuncCleaner;
numba/np/ufunc/_internal.c:PyTypeObject PyUFuncCleaner_Type;
numba/np/ufunc/_internal.c:    PyUFuncCleaner *obj = PyObject_New(PyUFuncCleaner, &PyUFuncCleaner_Type);
numba/np/ufunc/_internal.c:cleaner_dealloc(PyUFuncCleaner *self)
numba/np/ufunc/_internal.c:PyTypeObject PyUFuncCleaner_Type = {
numba/np/ufunc/_internal.c:    "numba._UFuncCleaner",                      /* tp_name*/
numba/np/ufunc/_internal.c:    sizeof(PyUFuncCleaner),                     /* tp_basicsize*/
numba/np/ufunc/_internal.c:    if (PyType_Ready(&PyUFuncCleaner_Type) < 0)
numba/np/ufunc/__init__.py:    def init_cuda_vectorize():
numba/np/ufunc/__init__.py:        from numba.cuda.vectorizers import CUDAVectorize
numba/np/ufunc/__init__.py:        return CUDAVectorize
numba/np/ufunc/__init__.py:    def init_cuda_guvectorize():
numba/np/ufunc/__init__.py:        from numba.cuda.vectorizers import CUDAGUFuncVectorize
numba/np/ufunc/__init__.py:        return CUDAGUFuncVectorize
numba/np/ufunc/__init__.py:    Vectorize.target_registry.ondemand['cuda'] = init_cuda_vectorize
numba/np/ufunc/__init__.py:    GUVectorize.target_registry.ondemand['cuda'] = init_cuda_guvectorize
numba/_devicearray.cpp:/* CUDA device array C API */
numba/cuda/testing.py:from numba.cuda.cuda_paths import get_conda_ctk
numba/cuda/testing.py:from numba.cuda.cudadrv import driver, devices, libs
numba/cuda/testing.py:numba_cuda_dir = Path(__file__).parent
numba/cuda/testing.py:test_data_dir = numba_cuda_dir / 'tests' / 'data'
numba/cuda/testing.py:class CUDATestCase(SerialMixin, TestCase):
numba/cuda/testing.py:    For tests that use a CUDA device. Test methods in a CUDATestCase must not
numba/cuda/testing.py:    the context and destroy resources used by a normal CUDATestCase if any of
numba/cuda/testing.py:    its tests are run between tests from a CUDATestCase.
numba/cuda/testing.py:        self._low_occupancy_warnings = config.CUDA_LOW_OCCUPANCY_WARNINGS
numba/cuda/testing.py:        self._warn_on_implicit_copy = config.CUDA_WARN_ON_IMPLICIT_COPY
numba/cuda/testing.py:        # Disable warnings about low gpu utilization in the test suite
numba/cuda/testing.py:        config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
numba/cuda/testing.py:        config.CUDA_WARN_ON_IMPLICIT_COPY = 0
numba/cuda/testing.py:        config.CUDA_LOW_OCCUPANCY_WARNINGS = self._low_occupancy_warnings
numba/cuda/testing.py:        config.CUDA_WARN_ON_IMPLICIT_COPY = self._warn_on_implicit_copy
numba/cuda/testing.py:class ContextResettingTestCase(CUDATestCase):
numba/cuda/testing.py:        from numba.cuda.cudadrv.devices import reset
numba/cuda/testing.py:    from numba.cuda import is_available as cuda_is_available
numba/cuda/testing.py:    from numba.cuda.cudadrv import nvvm
numba/cuda/testing.py:    if cuda_is_available():
numba/cuda/testing.py:        # Ensure that cudart.so is loaded and the list of supported compute
numba/cuda/testing.py:        # needed because some compilation tests don't require a CUDA context,
numba/cuda/testing.py:        # but do use NVVM, and it is required that libcudart.so should be
numba/cuda/testing.py:def skip_on_cudasim(reason):
numba/cuda/testing.py:    """Skip this test if running on the CUDA simulator"""
numba/cuda/testing.py:    return unittest.skipIf(config.ENABLE_CUDASIM, reason)
numba/cuda/testing.py:def skip_unless_cudasim(reason):
numba/cuda/testing.py:    """Skip this test if running on CUDA hardware"""
numba/cuda/testing.py:    return unittest.skipUnless(config.ENABLE_CUDASIM, reason)
numba/cuda/testing.py:def skip_unless_conda_cudatoolkit(reason):
numba/cuda/testing.py:    """Skip test if the CUDA toolkit was not installed by Conda"""
numba/cuda/testing.py:    return unittest.skipIf(config.CUDA_MEMORY_MANAGER != 'default', reason)
numba/cuda/testing.py:def skip_under_cuda_memcheck(reason):
numba/cuda/testing.py:    return unittest.skipIf(os.environ.get('CUDA_MEMCHECK') is not None, reason)
numba/cuda/testing.py:def skip_if_cuda_includes_missing(fn):
numba/cuda/testing.py:    # Skip when cuda.h is not available - generally this should indicate
numba/cuda/testing.py:    # whether the CUDA includes are available or not
numba/cuda/testing.py:    cuda_h = os.path.join(config.CUDA_INCLUDE_PATH, 'cuda.h')
numba/cuda/testing.py:    cuda_h_file = (os.path.exists(cuda_h) and os.path.isfile(cuda_h))
numba/cuda/testing.py:    reason = 'CUDA include dir not available on this system'
numba/cuda/testing.py:    return unittest.skipUnless(cuda_h_file, reason)(fn)
numba/cuda/testing.py:    return unittest.skipIf(config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY,
numba/cuda/testing.py:    if not config.ENABLE_CUDASIM:
numba/cuda/testing.py:def xfail_unless_cudasim(fn):
numba/cuda/testing.py:    if config.ENABLE_CUDASIM:
numba/cuda/testing.py:def skip_with_cuda_python(reason):
numba/cuda/testing.py:def cudadevrt_missing():
numba/cuda/testing.py:    if config.ENABLE_CUDASIM:
numba/cuda/testing.py:        path = libs.get_cudalib('cudadevrt', static=True)
numba/cuda/testing.py:def skip_if_cudadevrt_missing(fn):
numba/cuda/testing.py:    return unittest.skipIf(cudadevrt_missing(), 'cudadevrt missing')(fn)
numba/cuda/testing.py:    Class for emulating an array coming from another library through the CUDA
numba/cuda/testing.py:        self.__cuda_array_interface__ = arr.__cuda_array_interface__
numba/cuda/stubs.py:    outside the context of a CUDA kernel
numba/cuda/stubs.py:    outside the context of a CUDA kernel
numba/cuda/stubs.py:    attribute in :attr:`numba.cuda.blockDim` exclusive.
numba/cuda/stubs.py:    attribute in :attr:`numba.cuda.gridDim` exclusive.
numba/cuda/stubs.py:    :attr:`numba.cuda.warpsize` - 1.
numba/cuda/stubs.py:    docs.nvidia.com/cuda/nvvm-ir-spec/index.html#nvvm-intrin-warp-level-datamove
numba/cuda/stubs.py:    docs.nvidia.com/cuda/nvvm-ir-spec/index.html#nvvm-intrin-warp-level-vote
numba/cuda/stubs.py:                    "CUDA kernels."
numba/cuda/cpp_function_wrappers.cu:#include "cuda_fp16.h"
numba/cuda/extending.py:intrinsic = _intrinsic(target='cuda')
numba/cuda/cg.py:from numba.cuda import nvvmutils
numba/cuda/cg.py:from numba.cuda.extending import intrinsic
numba/cuda/cg.py:from numba.cuda.types import grid_group, GridGroup as GridGroupClass
numba/cuda/cg.py:            nvvmutils.declare_cudaCGGetIntrinsicHandle(mod),
numba/cuda/cg.py:@overload(this_grid, target='cuda')
numba/cuda/cg.py:            nvvmutils.declare_cudaCGSynchronize(mod),
numba/cuda/cg.py:@overload_method(GridGroupClass, 'sync', target='cuda')
numba/cuda/target.py:from .cudadrv import nvvm
numba/cuda/target.py:from numba.cuda import codegen, nvvmutils, ufuncs
numba/cuda/target.py:from numba.cuda.models import cuda_data_manager
numba/cuda/target.py:class CUDATypingContext(typing.BaseContext):
numba/cuda/target.py:        from . import cudadecl, cudamath, libdevicedecl, vector_types
numba/cuda/target.py:        self.install_registry(cudadecl.registry)
numba/cuda/target.py:        self.install_registry(cudamath.registry)
numba/cuda/target.py:        from numba.cuda.dispatcher import CUDADispatcher
numba/cuda/target.py:                isinstance(val, CUDADispatcher)):
numba/cuda/target.py:                disp = CUDADispatcher(val.py_func, targetoptions)
numba/cuda/target.py:        return super(CUDATypingContext, self).resolve_value_type(val)
numba/cuda/target.py:class CUDATargetContext(BaseContext):
numba/cuda/target.py:    def __init__(self, typingctx, target='cuda'):
numba/cuda/target.py:        self.data_model_manager = cuda_data_manager.chain(
numba/cuda/target.py:        self._internal_codegen = codegen.JITCUDACodegen("numba.cuda.jit")
numba/cuda/target.py:            cudaimpl, printimpl, libdeviceimpl, mathimpl, vector_types
numba/cuda/target.py:        self.install_registry(cudaimpl.registry)
numba/cuda/target.py:        Some CUDA intrinsics are at the module level, but cannot be treated as
numba/cuda/target.py:        from numba import cuda
numba/cuda/target.py:        nonconsts_with_mod = tuple([(types.Module(cuda), nc)
numba/cuda/target.py:        return CUDACallConv(self)
numba/cuda/target.py:    def prepare_cuda_kernel(self, codelib, fndesc, debug, lineinfo,
numba/cuda/target.py:        Adapt a code library ``codelib`` with the numba compiled CUDA kernel
numba/cuda/target.py:            fndesc.llvm_func_name, ns='cudapy',
numba/cuda/target.py:        wrapper_module = self.create_module("cuda.kernel.wrapper")
numba/cuda/target.py:        prefixed = itanium_mangler.prepend_namespace(func.name, ns='cudapy')
numba/cuda/target.py:        nvvm.set_cuda_kernel(wrapfn)
numba/cuda/target.py:        gv = cgutils.add_global_variable(lmod, constary.type, "_cudapy_cmem",
numba/cuda/target.py:class CUDACallConv(MinimalCallConv):
numba/cuda/target.py:class CUDACABICallConv(BaseCallConv):
numba/cuda/target.py:    Calling convention aimed at matching the CUDA C/C++ ABI. The implemented
numba/cuda/target.py:        msg = "Python exceptions are unsupported in the CUDA C/C++ ABI"
numba/cuda/target.py:        msg = "Return status is unsupported in the CUDA C/C++ ABI"
numba/cuda/errors.py:class CudaLoweringError(LoweringError):
numba/cuda/errors.py:_launch_help_url = ("https://numba.readthedocs.io/en/stable/cuda/"
numba/cuda/descriptor.py:from .target import CUDATargetContext, CUDATypingContext
numba/cuda/descriptor.py:class CUDATargetOptions(TargetOptions):
numba/cuda/descriptor.py:class CUDATarget(TargetDescriptor):
numba/cuda/descriptor.py:        self.options = CUDATargetOptions
numba/cuda/descriptor.py:        # this prevents an attempt to load CUDA libraries at import time on
numba/cuda/descriptor.py:            self._typingctx = CUDATypingContext()
numba/cuda/descriptor.py:            self._targetctx = CUDATargetContext(self._typingctx)
numba/cuda/descriptor.py:cuda_target = CUDATarget('cuda')
numba/cuda/ufuncs.py:"""Contains information on how to translate different ufuncs for the CUDA
numba/cuda/ufuncs.py:    from numba.cuda.mathimpl import (get_unary_impl_for_fn_and_ty,
numba/cuda/deviceufunc.py:            raise TypeError("No matching version.  GPU ufunc requires array "
numba/cuda/deviceufunc.py:                warnings.warn("nopython kwarg for cuda target is redundant",
numba/cuda/deviceufunc.py:                fmt += "cuda vectorize target does not support option: '%s'"
numba/cuda/deviceufunc.py:        # { arg_dtype: (return_dtype), cudakernel }
numba/cuda/deviceufunc.py:        # example, any output passed in that supports the CUDA Array Interface
numba/cuda/deviceufunc.py:        # is converted to a Numba CUDA device array; others are left untouched.
numba/cuda/random.py:from numba import (config, cuda, float32, float64, uint32, int64, uint64,
numba/cuda/random.py:# turn integers into floats when using these functions in the CUDA simulator.
numba/cuda/random.py:# both CPU and CUDA device functions.
numba/cuda/random.py:# When cudasim is enabled, Fake CUDA arrays are passed to some of the
numba/cuda/random.py:_forceobj = _looplift = config.ENABLE_CUDASIM
numba/cuda/random.py:_nopython = not config.ENABLE_CUDASIM
numba/cuda/random.py:    '''Initialize RNG states on the GPU for parallel generators.
numba/cuda/random.py:    sequence.  Therefore, as long no CUDA thread requests more than 2**64
numba/cuda/random.py:    # Initialization on CPU is much faster than the GPU
numba/cuda/random.py:    sequence.  Therefore, as long no CUDA thread requests more than 2**64
numba/cuda/random.py:    :type stream: CUDA stream
numba/cuda/random.py:    states = cuda.device_array(n, dtype=xoroshiro128p_dtype, stream=stream)
numba/cuda/models.py:from numba.cuda.types import Dim3, GridGroup, CUDADispatcher
numba/cuda/models.py:cuda_data_manager = DataModelManager()
numba/cuda/models.py:register_model = functools.partial(register, cuda_data_manager)
numba/cuda/models.py:register_model(CUDADispatcher)(models.OpaqueModel)
numba/cuda/cuda_fp16.hpp:* Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
numba/cuda/cuda_fp16.hpp:* subject to NVIDIA intellectual property rights under U.S. and
numba/cuda/cuda_fp16.hpp:* CONFIDENTIAL to NVIDIA and is being provided under the terms and
numba/cuda/cuda_fp16.hpp:* conditions of a form of NVIDIA software license agreement by and
numba/cuda/cuda_fp16.hpp:* between NVIDIA and Licensee ("License Agreement") or electronically
numba/cuda/cuda_fp16.hpp:* written consent of NVIDIA is prohibited.
numba/cuda/cuda_fp16.hpp:* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
numba/cuda/cuda_fp16.hpp:* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
numba/cuda/cuda_fp16.hpp:* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
numba/cuda/cuda_fp16.hpp:#if !defined(__CUDA_FP16_HPP__)
numba/cuda/cuda_fp16.hpp:#define __CUDA_FP16_HPP__
numba/cuda/cuda_fp16.hpp:#if !defined(__CUDA_FP16_H__)
numba/cuda/cuda_fp16.hpp:#error "Do not include this file directly. Instead, include cuda_fp16.h."
numba/cuda/cuda_fp16.hpp:#if defined(__CPP_VERSION_AT_LEAST_11_FP16) && !defined(__CUDACC_RTC__)
numba/cuda/cuda_fp16.hpp:#endif /* __cplusplus >= 201103L && !defined(__CUDACC_RTC__) */
numba/cuda/cuda_fp16.hpp: * When compiling as a CUDA source file memcpy is provided implicitly.
numba/cuda/cuda_fp16.hpp: * !defined(__CUDACC__) implies !defined(__CUDACC_RTC__).
numba/cuda/cuda_fp16.hpp:#if defined(__cplusplus) && !defined(__CUDACC__)
numba/cuda/cuda_fp16.hpp:#endif /* defined(__cplusplus) && !defined(__CUDACC__) */
numba/cuda/cuda_fp16.hpp:#if defined(__CUDACC__)
numba/cuda/cuda_fp16.hpp:#define __CUDA_FP16_DECL__ static __device__ __inline__
numba/cuda/cuda_fp16.hpp:#define __CUDA_HOSTDEVICE_FP16_DECL__ static __host__ __device__ __inline__
numba/cuda/cuda_fp16.hpp:#define __CUDA_HOSTDEVICE__ __host__ __device__
numba/cuda/cuda_fp16.hpp:#else /* !defined(__CUDACC__) */
numba/cuda/cuda_fp16.hpp:#define __CUDA_HOSTDEVICE_FP16_DECL__ static __attribute__ ((unused))
numba/cuda/cuda_fp16.hpp:#define __CUDA_HOSTDEVICE_FP16_DECL__ static
numba/cuda/cuda_fp16.hpp:#define __CUDA_HOSTDEVICE__
numba/cuda/cuda_fp16.hpp:#endif /* defined(__CUDACC_) */
numba/cuda/cuda_fp16.hpp:#if defined(__CUDACC__)
numba/cuda/cuda_fp16.hpp:#define __CUDA_ALIGN__(align) __align__(align)
numba/cuda/cuda_fp16.hpp:#define __CUDA_ALIGN__(n) alignas(n)    /* C++11 kindly gives us a keyword for this */
numba/cuda/cuda_fp16.hpp:#define __CUDA_ALIGN__(n) __attribute__ ((aligned(n)))
numba/cuda/cuda_fp16.hpp:#define __CUDA_ALIGN__(n) __declspec(align(n))
numba/cuda/cuda_fp16.hpp:#define __CUDA_ALIGN__(n)
numba/cuda/cuda_fp16.hpp:#endif /* defined(__CUDACC__) */
numba/cuda/cuda_fp16.hpp:* Such a representation will be deprecated in a future version of CUDA. 
numba/cuda/cuda_fp16.hpp:typedef struct __CUDA_ALIGN__(2) {
numba/cuda/cuda_fp16.hpp:typedef struct __CUDA_ALIGN__(4) {
numba/cuda/cuda_fp16.hpp:struct __CUDA_ALIGN__(2) __half {
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half() { }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half(const __half_raw &hr) : __x(hr.x) { }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half &operator=(const __half_raw &hr) { __x = hr.x; return *this; }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ volatile __half &operator=(const __half_raw &hr) volatile { __x = hr.x; return *this; }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ volatile __half &operator=(const volatile __half_raw &hr) volatile { __x = hr.x; return *this; }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ operator __half_raw() const { __half_raw ret; ret.x = __x; return ret; }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ operator __half_raw() const volatile { __half_raw ret; ret.x = __x; return ret; }
numba/cuda/cuda_fp16.hpp:#if !defined(__CUDA_NO_HALF_CONVERSIONS__)
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half(const float f) { __x = __float2half(f).__x;  }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half(const double f) { __x = __double2half(f).__x;  }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ operator float() const { return __half2float(*this); }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half &operator=(const float f) { __x = __float2half(f).__x; return *this; }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half &operator=(const double f) { __x = __double2half(f).__x; return *this; }
numba/cuda/cuda_fp16.hpp:#if defined(__CUDACC__)
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half(const short val) { __x = __short2half_rn(val).__x;  }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half(const unsigned short val) { __x = __ushort2half_rn(val).__x;  }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half(const int val) { __x = __int2half_rn(val).__x;  }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half(const unsigned int val) { __x = __uint2half_rn(val).__x;  }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half(const long long val) { __x = __ll2half_rn(val).__x;  }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half(const unsigned long long val) { __x = __ull2half_rn(val).__x; }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ operator short() const { return __half2short_rz(*this); }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half &operator=(const short val) { __x = __short2half_rn(val).__x; return *this; }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ operator unsigned short() const { return __half2ushort_rz(*this); }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half &operator=(const unsigned short val) { __x = __ushort2half_rn(val).__x; return *this; }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ operator int() const { return __half2int_rz(*this); }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half &operator=(const int val) { __x = __int2half_rn(val).__x; return *this; }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ operator unsigned int() const { return __half2uint_rz(*this); }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half &operator=(const unsigned int val) { __x = __uint2half_rn(val).__x; return *this; }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ operator long long() const { return __half2ll_rz(*this); }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half &operator=(const long long val) { __x = __ll2half_rn(val).__x; return *this; }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ operator unsigned long long() const { return __half2ull_rz(*this); }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half &operator=(const unsigned long long val) { __x = __ull2half_rn(val).__x; return *this; }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ operator bool() const { return (__x & 0x7FFFU) != 0U; }
numba/cuda/cuda_fp16.hpp:#endif /* defined(__CUDACC__) */
numba/cuda/cuda_fp16.hpp:#endif /* !defined(__CUDA_NO_HALF_CONVERSIONS__) */
numba/cuda/cuda_fp16.hpp:#if defined(__CUDACC__)
numba/cuda/cuda_fp16.hpp:#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:#if !defined(__CUDA_NO_HALF_OPERATORS__)
numba/cuda/cuda_fp16.hpp:#endif /* !defined(__CUDA_NO_HALF_OPERATORS__) */
numba/cuda/cuda_fp16.hpp:#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
numba/cuda/cuda_fp16.hpp:#endif /* defined(__CUDACC__) */
numba/cuda/cuda_fp16.hpp:struct __CUDA_ALIGN__(4) __half2 {
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half2(const __half2 &&src) { __HALF2_TO_UI(*this) = std::move(__HALF2_TO_CUI(src)); }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half2 &operator=(const __half2 &&src) { __HALF2_TO_UI(*this) = std::move(__HALF2_TO_CUI(src)); return *this; }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half2() { }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half2(const __half &a, const __half &b) : x(a), y(b) { }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half2(const __half2 &src) { __HALF2_TO_UI(*this) = __HALF2_TO_CUI(src); }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half2 &operator=(const __half2 &src) { __HALF2_TO_UI(*this) = __HALF2_TO_CUI(src); return *this; }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half2(const __half2_raw &h2r ) { __HALF2_TO_UI(*this) = __HALF2_TO_CUI(h2r); }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ __half2 &operator=(const __half2_raw &h2r) { __HALF2_TO_UI(*this) = __HALF2_TO_CUI(h2r); return *this; }
numba/cuda/cuda_fp16.hpp:    __CUDA_HOSTDEVICE__ operator __half2_raw() const { __half2_raw ret; ret.x = 0U; ret.y = 0U; __HALF2_TO_UI(ret) = __HALF2_TO_CUI(*this); return ret; }
numba/cuda/cuda_fp16.hpp:#if defined(__CUDACC__)
numba/cuda/cuda_fp16.hpp:#if (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)) && !defined(__CUDA_NO_HALF2_OPERATORS__)
numba/cuda/cuda_fp16.hpp:#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
numba/cuda/cuda_fp16.hpp:#endif /* defined(__CUDACC__) */
numba/cuda/cuda_fp16.hpp:#undef __CUDA_HOSTDEVICE__
numba/cuda/cuda_fp16.hpp:#undef __CUDA_ALIGN__
numba/cuda/cuda_fp16.hpp:#ifndef __CUDACC_RTC__  /* no host functions in NVRTC mode */
numba/cuda/cuda_fp16.hpp:#if defined(__CUDACC__)
numba/cuda/cuda_fp16.hpp:#endif  /* #if !defined(__CUDACC_RTC__) */
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ __half __double2half(const double a)
numba/cuda/cuda_fp16.hpp:#if defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:    #if defined(__CUDACC__)
numba/cuda/cuda_fp16.hpp:        #if defined(__CUDACC__)
numba/cuda/cuda_fp16.hpp:        #if defined(__CUDACC__)
numba/cuda/cuda_fp16.hpp:        #if defined(__CUDACC__)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half(const float a)
numba/cuda/cuda_fp16.hpp:#if defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_rn(const float a)
numba/cuda/cuda_fp16.hpp:#if defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_rz(const float a)
numba/cuda/cuda_fp16.hpp:#if defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_rd(const float a)
numba/cuda/cuda_fp16.hpp:#if defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_ru(const float a)
numba/cuda/cuda_fp16.hpp:#if defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __float2half2_rn(const float a)
numba/cuda/cuda_fp16.hpp:#if defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __floats2half2_rn(const float a, const float b)
numba/cuda/cuda_fp16.hpp:#if defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:#ifndef __CUDACC_RTC__  /* no host functions in NVRTC mode */
numba/cuda/cuda_fp16.hpp:#if defined(__CUDACC__)
numba/cuda/cuda_fp16.hpp:#endif  /* !defined(__CUDACC_RTC__) */
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ float __half2float(const __half a)
numba/cuda/cuda_fp16.hpp:#if defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ float __low2float(const __half2 a)
numba/cuda/cuda_fp16.hpp:#if defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ float __high2float(const __half2 a)
numba/cuda/cuda_fp16.hpp:#if defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:#if defined(__CUDACC__)
numba/cuda/cuda_fp16.hpp:/* CUDA vector-types compatible vector creation function (note returns __half2, not half2) */
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __float22half2_rn(const float2 a)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ float2 __half22float2(const __half2 a)
numba/cuda/cuda_fp16.hpp:#if defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ int __half2int_rn(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ int __half2int_rz(const __half h)
numba/cuda/cuda_fp16.hpp:#if defined __CUDA_ARCH__
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ int __half2int_rd(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ int __half2int_ru(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ __half __int2half_rn(const int i)
numba/cuda/cuda_fp16.hpp:#if defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __int2half_rz(const int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __int2half_rd(const int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __int2half_ru(const int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ short int __half2short_rn(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ short int __half2short_rz(const __half h)
numba/cuda/cuda_fp16.hpp:#if defined __CUDA_ARCH__
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ short int __half2short_rd(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ short int __half2short_ru(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ __half __short2half_rn(const short int i)
numba/cuda/cuda_fp16.hpp:#if defined __CUDA_ARCH__
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __short2half_rz(const short int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __short2half_rd(const short int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __short2half_ru(const short int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ unsigned int __half2uint_rn(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __half2uint_rz(const __half h)
numba/cuda/cuda_fp16.hpp:#if defined __CUDA_ARCH__
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ unsigned int __half2uint_rd(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ unsigned int __half2uint_ru(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ __half __uint2half_rn(const unsigned int i)
numba/cuda/cuda_fp16.hpp:#if defined __CUDA_ARCH__
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __uint2half_rz(const unsigned int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __uint2half_rd(const unsigned int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __uint2half_ru(const unsigned int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ unsigned short int __half2ushort_rn(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ unsigned short int __half2ushort_rz(const __half h)
numba/cuda/cuda_fp16.hpp:#if defined __CUDA_ARCH__
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ unsigned short int __half2ushort_rd(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ unsigned short int __half2ushort_ru(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ __half __ushort2half_rn(const unsigned short int i)
numba/cuda/cuda_fp16.hpp:#if defined __CUDA_ARCH__
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __ushort2half_rz(const unsigned short int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __ushort2half_rd(const unsigned short int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __ushort2half_ru(const unsigned short int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ unsigned long long int __half2ull_rn(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ unsigned long long int __half2ull_rz(const __half h)
numba/cuda/cuda_fp16.hpp:#if defined __CUDA_ARCH__
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ unsigned long long int __half2ull_rd(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ unsigned long long int __half2ull_ru(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ __half __ull2half_rn(const unsigned long long int i)
numba/cuda/cuda_fp16.hpp:#if defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __ull2half_rz(const unsigned long long int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __ull2half_rd(const unsigned long long int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __ull2half_ru(const unsigned long long int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ long long int __half2ll_rn(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ long long int __half2ll_rz(const __half h)
numba/cuda/cuda_fp16.hpp:#if defined __CUDA_ARCH__
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ long long int __half2ll_rd(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ long long int __half2ll_ru(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_HOSTDEVICE_FP16_DECL__ __half __ll2half_rn(const long long int i)
numba/cuda/cuda_fp16.hpp:#if defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __ll2half_rz(const long long int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __ll2half_rd(const long long int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __ll2half_ru(const long long int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half htrunc(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half hceil(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half hfloor(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half hrint(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 h2trunc(const __half2 h)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 h2ceil(const __half2 h)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 h2floor(const __half2 h)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 h2rint(const __half2 h)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __lows2half2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __highs2half2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __low2half(const __half2 a)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ int __hisinf(const __half a)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __low2half2(const __half2 a)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __high2half2(const __half2 a)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __high2half(const __half2 a)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __halves2half2(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __half2half2(const __half a)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __lowhigh2highlow(const __half2 a)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ short int __half_as_short(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ unsigned short int __half_as_ushort(const __half h)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __short_as_half(const short int i)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __ushort_as_half(const unsigned short int i)
numba/cuda/cuda_fp16.hpp:#if __CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 700
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __shfl(const __half2 var, const int delta, const int width)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __shfl_up(const __half2 var, const unsigned int delta, const int width)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __shfl_down(const __half2 var, const unsigned int delta, const int width)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __shfl_xor(const __half2 var, const int delta, const int width)
numba/cuda/cuda_fp16.hpp:#endif /* !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 700 */
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __shfl_sync(const unsigned mask, const __half2 var, const int delta, const int width)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __shfl_up_sync(const unsigned mask, const __half2 var, const unsigned int delta, const int width)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __shfl_down_sync(const unsigned mask, const __half2 var, const unsigned int delta, const int width)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __shfl_xor_sync(const unsigned mask, const __half2 var, const int delta, const int width)
numba/cuda/cuda_fp16.hpp:#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 700
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __shfl(const __half var, const int delta, const int width)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __shfl_up(const __half var, const unsigned int delta, const int width)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __shfl_down(const __half var, const unsigned int delta, const int width)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __shfl_xor(const __half var, const int delta, const int width)
numba/cuda/cuda_fp16.hpp:#endif /* !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 700 */
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __shfl_sync(const unsigned mask, const __half var, const int delta, const int width)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __shfl_up_sync(const unsigned mask, const __half var, const unsigned int delta, const int width)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __shfl_down_sync(const unsigned mask, const __half var, const unsigned int delta, const int width)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __shfl_xor_sync(const unsigned mask, const __half var, const int delta, const int width)
numba/cuda/cuda_fp16.hpp:#endif /*__CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__)*/
numba/cuda/cuda_fp16.hpp:#if defined(__cplusplus) && (__CUDA_ARCH__ >= 320 || !defined(__CUDA_ARCH__))
numba/cuda/cuda_fp16.hpp:#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
numba/cuda/cuda_fp16.hpp:#endif /*(defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)*/
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __ldg(const  __half2 *const ptr)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __ldg(const __half *const ptr)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __ldcg(const  __half2 *const ptr)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __ldcg(const __half *const ptr)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __ldca(const  __half2 *const ptr)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __ldca(const __half *const ptr)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __ldcs(const  __half2 *const ptr)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __ldcs(const __half *const ptr)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __ldlu(const  __half2 *const ptr)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __ldlu(const __half *const ptr)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __ldcv(const  __half2 *const ptr)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __ldcv(const __half *const ptr)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ void __stwb(__half2 *const ptr, const __half2 value)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ void __stwb(__half *const ptr, const __half value)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ void __stcg(__half2 *const ptr, const __half2 value)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ void __stcg(__half *const ptr, const __half value)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ void __stcs(__half2 *const ptr, const __half2 value)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ void __stcs(__half *const ptr, const __half value)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ void __stwt(__half2 *const ptr, const __half2 value)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ void __stwt(__half *const ptr, const __half value)
numba/cuda/cuda_fp16.hpp:#endif /*defined(__cplusplus) && (__CUDA_ARCH__ >= 320 || !defined(__CUDA_ARCH__))*/
numba/cuda/cuda_fp16.hpp:#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __heq2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hne2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hle2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hge2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hlt2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hgt2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hequ2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hneu2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hleu2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hgeu2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hltu2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hgtu2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hbeq2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hbne2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hble2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hbge2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hblt2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hbgt2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hbequ2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hbneu2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hbleu2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hbgeu2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hbltu2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hbgtu2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __heq(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hne(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hle(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hge(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hlt(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hgt(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hequ(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hneu(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hleu(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hgeu(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hltu(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hgtu(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hadd2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hsub2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hmul2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hadd2_sat(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hsub2_sat(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hmul2_sat(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __h2div(const __half2 a, const __half2 b) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __hadd(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __hsub(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __hmul(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __hadd_sat(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __hsub_sat(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __hmul_sat(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __hfma(const __half a, const __half b, const __half c)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __hfma_sat(const __half a, const __half b, const __half c)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __hdiv(const __half a, const __half b) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __hsin_internal(const __half a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half hsin(const __half a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 h2sin(const __half2 a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __hcos_internal(const __half a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half hcos(const __half a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 h2cos(const __half2 a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half hexp(const __half a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 h2exp(const __half2 a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half hexp2(const __half a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 h2exp2(const __half2 a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half hexp10(const __half a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 h2exp10(const __half2 a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half hlog2(const __half a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 h2log2(const __half2 a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half hlog(const __half a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 h2log(const __half2 a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half hlog10(const __half a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 h2log10(const __half2 a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 h2rcp(const __half2 a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half hrcp(const __half a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 h2rsqrt(const __half2 a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half hrsqrt(const __half a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 h2sqrt(const __half2 a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half hsqrt(const __half a) {
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hisnan2(const __half2 a)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ bool __hisnan(const __half a)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hneg2(const __half2 a)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __hneg(const __half a)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __habs2(const __half2 a)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __habs(const __half a)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hcmadd(const __half2 a, const __half2 b, const __half2 c)
numba/cuda/cuda_fp16.hpp:#endif /*__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
numba/cuda/cuda_fp16.hpp:#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __hmax(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __hmin(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __hmax_nan(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __hmin_nan(const __half a, const __half b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half __hfma_relu(const __half a, const __half b, const __half c)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hmax2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hmin2(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hmax2_nan(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hmin2_nan(const __half2 a, const __half2 b)
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__ __half2 __hfma2_relu(const __half2 a, const __half2 b, const __half2 c)
numba/cuda/cuda_fp16.hpp:#endif /*__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)*/
numba/cuda/cuda_fp16.hpp:#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
numba/cuda/cuda_fp16.hpp:#endif /*(defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)*/
numba/cuda/cuda_fp16.hpp:#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__  __half2 atomicAdd(__half2 *const address, const __half2 val) {
numba/cuda/cuda_fp16.hpp:#endif /*!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600*/
numba/cuda/cuda_fp16.hpp:#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
numba/cuda/cuda_fp16.hpp:__CUDA_FP16_DECL__  __half atomicAdd(__half *const address, const __half val) {
numba/cuda/cuda_fp16.hpp:#endif /*!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700*/
numba/cuda/cuda_fp16.hpp:#undef __CUDA_FP16_DECL__
numba/cuda/cuda_fp16.hpp:#endif /* defined(__CUDACC__) */
numba/cuda/cuda_fp16.hpp:#undef __CUDA_HOSTDEVICE_FP16_DECL__
numba/cuda/cuda_fp16.hpp:#undef __CUDA_FP16_DECL__
numba/cuda/cuda_fp16.hpp:/* Define first-class types "half" and "half2", unless user specifies otherwise via "#define CUDA_NO_HALF" */
numba/cuda/cuda_fp16.hpp:#if defined(__cplusplus) && !defined(CUDA_NO_HALF)
numba/cuda/cuda_fp16.hpp:#endif /* defined(__cplusplus) && !defined(CUDA_NO_HALF) */
numba/cuda/cuda_fp16.hpp:#endif /* end of include guard: __CUDA_FP16_HPP__ */
numba/cuda/types.py:class CUDADispatcher(types.Dispatcher):
numba/cuda/types.py:    """The type of CUDA dispatchers"""
numba/cuda/types.py:    # This type exists (instead of using types.Dispatcher as the type of CUDA
numba/cuda/types.py:    # generally valid to use the address of CUDA kernels and functions.
numba/cuda/types.py:    # is still probably a good idea to have a separate type for CUDA
numba/cuda/initialize.py:    import numba.cuda.models  # noqa: F401
numba/cuda/initialize.py:    from numba.cuda.decorators import jit
numba/cuda/initialize.py:    from numba.cuda.dispatcher import CUDADispatcher
numba/cuda/initialize.py:    cuda_target = target_registry["cuda"]
numba/cuda/initialize.py:    jit_registry[cuda_target] = jit
numba/cuda/initialize.py:    dispatcher_registry[cuda_target] = CUDADispatcher
numba/cuda/mathimpl.py:from numba.cuda import libdevice
numba/cuda/mathimpl.py:from numba import cuda
numba/cuda/mathimpl.py:        return cuda.fp16.hsin(x)
numba/cuda/mathimpl.py:        return cuda.fp16.hcos(x)
numba/cuda/mathimpl.py:        return cuda.fp16.hlog(x)
numba/cuda/mathimpl.py:        return cuda.fp16.hlog10(x)
numba/cuda/mathimpl.py:        return cuda.fp16.hlog2(x)
numba/cuda/mathimpl.py:        return cuda.fp16.hexp(x)
numba/cuda/mathimpl.py:        return cuda.fp16.hfloor(x)
numba/cuda/mathimpl.py:        return cuda.fp16.hceil(x)
numba/cuda/mathimpl.py:        return cuda.fp16.hsqrt(x)
numba/cuda/mathimpl.py:        return cuda.fp16.habs(x)
numba/cuda/mathimpl.py:        return cuda.fp16.htrunc(x)
numba/cuda/tests/cudapy/test_extending.py:from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
numba/cuda/tests/cudapy/test_extending.py:from numba import config, cuda, njit, types
numba/cuda/tests/cudapy/test_extending.py:if not config.ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_extending.py:    from numba.cuda.cudadecl import registry as cuda_registry
numba/cuda/tests/cudapy/test_extending.py:    from numba.cuda.cudaimpl import lower_attr as cuda_lower_attr
numba/cuda/tests/cudapy/test_extending.py:    @cuda_registry.register_attr
numba/cuda/tests/cudapy/test_extending.py:    @cuda_lower_attr(IntervalType, 'width')
numba/cuda/tests/cudapy/test_extending.py:    def cuda_Interval_width(context, builder, sig, arg):
numba/cuda/tests/cudapy/test_extending.py:@skip_on_cudasim('Extensions not supported in the simulator')
numba/cuda/tests/cudapy/test_extending.py:class TestExtending(CUDATestCase):
numba/cuda/tests/cudapy/test_extending.py:        @cuda.jit
numba/cuda/tests/cudapy/test_extending.py:        @cuda.jit
numba/cuda/tests/cudapy/test_extending.py:        @cuda.jit
numba/cuda/tests/cudapy/test_extending.py:        @cuda.jit
numba/cuda/tests/cudapy/test_compiler.py:from numba import cuda, float32, int16, int32, int64, uint32, void
numba/cuda/tests/cudapy/test_compiler.py:from numba.cuda import (compile, compile_for_current_device, compile_ptx,
numba/cuda/tests/cudapy/test_compiler.py:from numba.cuda.cudadrv import runtime
numba/cuda/tests/cudapy/test_compiler.py:from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
numba/cuda/tests/cudapy/test_compiler.py:@skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_compiler.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_compiler.py:        # between CUDA toolkit versions.
numba/cuda/tests/cudapy/test_compiler.py:        # with CUDA 11.2 / NVVM 7.0 onwards. Previously it failed because NVVM
numba/cuda/tests/cudapy/test_compiler.py:@skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_compiler.py:class TestCompileForCurrentDevice(CUDATestCase):
numba/cuda/tests/cudapy/test_compiler.py:        device_cc = cuda.get_current_device().compute_capability
numba/cuda/tests/cudapy/test_compiler.py:        cc = cuda.cudadrv.nvvm.find_closest_arch(device_cc)
numba/cuda/tests/cudapy/test_compiler.py:@skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_compiler.py:            cuda.nanosleep(32)
numba/cuda/tests/cudapy/test_compiler.py:            cuda.nanosleep(x)
numba/cuda/tests/cudapy/test_lineinfo.py:from numba import cuda, float32, int32
numba/cuda/tests/cudapy/test_lineinfo.py:from numba.cuda.testing import CUDATestCase, skip_on_cudasim
numba/cuda/tests/cudapy/test_lineinfo.py:@skip_on_cudasim('Simulator does not produce lineinfo')
numba/cuda/tests/cudapy/test_lineinfo.py:class TestCudaLineInfo(CUDATestCase):
numba/cuda/tests/cudapy/test_lineinfo.py:        @cuda.jit(lineinfo=False)
numba/cuda/tests/cudapy/test_lineinfo.py:        @cuda.jit(lineinfo=True)
numba/cuda/tests/cudapy/test_lineinfo.py:        @cuda.jit(sig, lineinfo=True)
numba/cuda/tests/cudapy/test_lineinfo.py:        @cuda.jit
numba/cuda/tests/cudapy/test_lineinfo.py:        @cuda.jit
numba/cuda/tests/cudapy/test_lineinfo.py:        @cuda.jit(lineinfo=True)
numba/cuda/tests/cudapy/test_lineinfo.py:        @cuda.jit(lineinfo=True)
numba/cuda/tests/cudapy/test_lineinfo.py:            @cuda.jit(debug=True, lineinfo=True, opt=False)
numba/cuda/tests/cudapy/test_nondet.py:from numba import cuda, float32, void
numba/cuda/tests/cudapy/test_nondet.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_nondet.py:class TestCudaNonDet(CUDATestCase):
numba/cuda/tests/cudapy/test_nondet.py:        @cuda.jit(void(float32[:, :], float32[:, :], float32[:]))
numba/cuda/tests/cudapy/test_nondet.py:            startX, startY = cuda.grid(2)
numba/cuda/tests/cudapy/test_nondet.py:            gridX = cuda.gridDim.x * cuda.blockDim.x
numba/cuda/tests/cudapy/test_nondet.py:            gridY = cuda.gridDim.y * cuda.blockDim.y
numba/cuda/tests/cudapy/test_nondet.py:        dA = cuda.to_device(A)
numba/cuda/tests/cudapy/test_nondet.py:        dB = cuda.to_device(B)
numba/cuda/tests/cudapy/test_nondet.py:        dF = cuda.to_device(F, copy=False)
numba/cuda/tests/cudapy/test_localmem.py:from numba import cuda, int32, complex128, void
numba/cuda/tests/cudapy/test_localmem.py:from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
numba/cuda/tests/cudapy/test_localmem.py:    C = cuda.local.array(1000, dtype=int32)
numba/cuda/tests/cudapy/test_localmem.py:    C = cuda.local.array(100, dtype=complex128)
numba/cuda/tests/cudapy/test_localmem.py:    C = cuda.local.array((5,), dtype=int32)
numba/cuda/tests/cudapy/test_localmem.py:@skip_on_cudasim('PTX inspection not available in cudasim')
numba/cuda/tests/cudapy/test_localmem.py:class TestCudaLocalMem(CUDATestCase):
numba/cuda/tests/cudapy/test_localmem.py:        jculocal = cuda.jit(sig)(culocal)
numba/cuda/tests/cudapy/test_localmem.py:        jculocal = cuda.jit('void(int32[:], int32[:])')(culocal1tuple)
numba/cuda/tests/cudapy/test_localmem.py:        jculocalcomplex = cuda.jit(sig)(culocalcomplex)
numba/cuda/tests/cudapy/test_localmem.py:        # Find the typing of the dtype argument to cuda.local.array
numba/cuda/tests/cudapy/test_localmem.py:    @skip_on_cudasim("Can't check typing in simulator")
numba/cuda/tests/cudapy/test_localmem.py:        @cuda.jit(void(int32[::1]))
numba/cuda/tests/cudapy/test_localmem.py:            l = cuda.local.array(10, dtype=int32)
numba/cuda/tests/cudapy/test_localmem.py:    @skip_on_cudasim("Can't check typing in simulator")
numba/cuda/tests/cudapy/test_localmem.py:        @cuda.jit(void(int32[::1]))
numba/cuda/tests/cudapy/test_localmem.py:            l = cuda.local.array(10, dtype=np.int32)
numba/cuda/tests/cudapy/test_localmem.py:    @skip_on_cudasim("Can't check typing in simulator")
numba/cuda/tests/cudapy/test_localmem.py:        @cuda.jit(void(int32[::1]))
numba/cuda/tests/cudapy/test_localmem.py:            l = cuda.local.array(10, dtype='int32')
numba/cuda/tests/cudapy/test_localmem.py:    @skip_on_cudasim("Can't check typing in simulator")
numba/cuda/tests/cudapy/test_localmem.py:            @cuda.jit(void(int32[::1]))
numba/cuda/tests/cudapy/test_localmem.py:                l = cuda.local.array(10, dtype='int33')
numba/cuda/tests/cudapy/test_localmem.py:        @cuda.jit(void(test_struct_model_type[::1]))
numba/cuda/tests/cudapy/test_localmem.py:            l = cuda.local.array(10, dtype=test_struct_model_type)
numba/cuda/tests/cudapy/test_localmem.py:        @cuda.jit(void(int32[::1], int32[::1]))
numba/cuda/tests/cudapy/test_localmem.py:            arr = cuda.local.array(10, dtype=test_struct_model_type)
numba/cuda/tests/cudapy/test_localmem.py:        @cuda.jit
numba/cuda/tests/cudapy/test_localmem.py:            arr = cuda.local.array(shape, dtype=ty)
numba/cuda/tests/cudapy/test_optimization.py:from numba.cuda.testing import skip_on_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_optimization.py:from numba import cuda, float64
numba/cuda/tests/cudapy/test_optimization.py:@skip_on_cudasim('Simulator does not optimize code')
numba/cuda/tests/cudapy/test_optimization.py:class TestOptimization(CUDATestCase):
numba/cuda/tests/cudapy/test_optimization.py:        kernel = cuda.jit(sig)(kernel_func)
numba/cuda/tests/cudapy/test_optimization.py:        kernel = cuda.jit(sig, opt=False)(kernel_func)
numba/cuda/tests/cudapy/test_optimization.py:        kernel = cuda.jit(kernel_func)
numba/cuda/tests/cudapy/test_optimization.py:        kernel = cuda.jit(opt=False)(kernel_func)
numba/cuda/tests/cudapy/test_optimization.py:        device = cuda.jit(sig, device=True)(device_func)
numba/cuda/tests/cudapy/test_optimization.py:        device = cuda.jit(sig, device=True, opt=False)(device_func)
numba/cuda/tests/cudapy/cache_usecases.py:from numba import cuda
numba/cuda/tests/cudapy/cache_usecases.py:from numba.cuda.testing import CUDATestCase
numba/cuda/tests/cudapy/cache_usecases.py:    This allows the CUDA cache tests to closely match the CPU cache tests, and
numba/cuda/tests/cudapy/cache_usecases.py:class CUDAUseCase(UseCase):
numba/cuda/tests/cudapy/cache_usecases.py:@cuda.jit(cache=True)
numba/cuda/tests/cudapy/cache_usecases.py:@cuda.jit(cache=False)
numba/cuda/tests/cudapy/cache_usecases.py:add_usecase = CUDAUseCase(add_usecase_kernel)
numba/cuda/tests/cudapy/cache_usecases.py:add_nocache_usecase = CUDAUseCase(add_nocache_usecase_kernel)
numba/cuda/tests/cudapy/cache_usecases.py:@cuda.jit(cache=True)
numba/cuda/tests/cudapy/cache_usecases.py:@cuda.jit(cache=True)
numba/cuda/tests/cudapy/cache_usecases.py:@cuda.jit(cache=False)
numba/cuda/tests/cudapy/cache_usecases.py:outer = CUDAUseCase(outer_kernel)
numba/cuda/tests/cudapy/cache_usecases.py:outer_uncached = CUDAUseCase(outer_uncached_kernel)
numba/cuda/tests/cudapy/cache_usecases.py:@cuda.jit(cache=True)
numba/cuda/tests/cudapy/cache_usecases.py:record_return_packed = CUDAUseCase(record_return, retty=packed_record_type)
numba/cuda/tests/cudapy/cache_usecases.py:record_return_aligned = CUDAUseCase(record_return, retty=aligned_record_type)
numba/cuda/tests/cudapy/cache_usecases.py:    @cuda.jit(cache=True)
numba/cuda/tests/cudapy/cache_usecases.py:    return CUDAUseCase(closure)
numba/cuda/tests/cudapy/cache_usecases.py:@cuda.jit(cache=True)
numba/cuda/tests/cudapy/cache_usecases.py:renamed_function1 = CUDAUseCase(ambiguous_function)
numba/cuda/tests/cudapy/cache_usecases.py:@cuda.jit(cache=True)
numba/cuda/tests/cudapy/cache_usecases.py:renamed_function2 = CUDAUseCase(ambiguous_function)
numba/cuda/tests/cudapy/cache_usecases.py:@cuda.jit(cache=True)
numba/cuda/tests/cudapy/cache_usecases.py:    aa = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    ab = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    ac = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    ad = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    ae = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    af = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    ag = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    ah = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    ai = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    aj = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    ak = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    al = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    am = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    an = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    ao = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    ap = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    ar = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    at = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    au = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    av = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    aw = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    ax = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    ay = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:    az = cuda.local.array((1, 1), np.float64)
numba/cuda/tests/cudapy/cache_usecases.py:@cuda.jit(cache=True)
numba/cuda/tests/cudapy/cache_usecases.py:simple_usecase_caller = CUDAUseCase(simple_usecase_kernel)
numba/cuda/tests/cudapy/cache_usecases.py:@cuda.jit(cache=True)
numba/cuda/tests/cudapy/cache_usecases.py:    grid = cuda.cg.this_grid()
numba/cuda/tests/cudapy/cache_usecases.py:cg_usecase = CUDAUseCase(cg_usecase_kernel)
numba/cuda/tests/cudapy/cache_usecases.py:class _TestModule(CUDATestCase):
numba/cuda/tests/cudapy/test_device_func.py:from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_device_func.py:from numba import cuda, jit, float32, int32
numba/cuda/tests/cudapy/test_device_func.py:class TestDeviceFunc(CUDATestCase):
numba/cuda/tests/cudapy/test_device_func.py:        @cuda.jit("float32(float32, float32)", device=True)
numba/cuda/tests/cudapy/test_device_func.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_device_func.py:        compiled = cuda.jit("void(float32[:])")(use_add2f)
numba/cuda/tests/cudapy/test_device_func.py:        @cuda.jit("float32(float32, float32)", device=True)
numba/cuda/tests/cudapy/test_device_func.py:        @cuda.jit("float32(float32, float32)", device=True)
numba/cuda/tests/cudapy/test_device_func.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_device_func.py:        compiled = cuda.jit("void(float32[:])")(indirect_add2f)
numba/cuda/tests/cudapy/test_device_func.py:        @cuda.jit
numba/cuda/tests/cudapy/test_device_func.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_device_func.py:    @skip_on_cudasim('not supported in cudasim')
numba/cuda/tests/cudapy/test_device_func.py:        # compiling on CUDA.
numba/cuda/tests/cudapy/test_device_func.py:        @cuda.jit
numba/cuda/tests/cudapy/test_device_func.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_device_func.py:    @skip_on_cudasim('not supported in cudasim')
numba/cuda/tests/cudapy/test_device_func.py:        @cuda.jit(device=True)
numba/cuda/tests/cudapy/test_device_func.py:    @skip_on_cudasim('not supported in cudasim')
numba/cuda/tests/cudapy/test_device_func.py:        @cuda.jit(device=True)
numba/cuda/tests/cudapy/test_device_func.py:    @skip_on_cudasim('not supported in cudasim')
numba/cuda/tests/cudapy/test_device_func.py:        @cuda.jit(device=True)
numba/cuda/tests/cudapy/test_device_func.py:    @skip_on_cudasim('cudasim will allow calling any function')
numba/cuda/tests/cudapy/test_device_func.py:        @cuda.jit(device=True)
numba/cuda/tests/cudapy/test_device_func.py:    @skip_on_cudasim('cudasim ignores casting by jit decorator signature')
numba/cuda/tests/cudapy/test_device_func.py:        @cuda.jit('int32(int32, int32, int32, int32)', device=True)
numba/cuda/tests/cudapy/test_device_func.py:        @cuda.jit
numba/cuda/tests/cudapy/test_device_func.py:        x = cuda.device_array(1, dtype=np.int32)
numba/cuda/tests/cudapy/test_device_func.py:        channels = cuda.to_device(np.asarray([1.0, 2.0, 3.0, 4.0],
numba/cuda/tests/cudapy/test_device_func.py:    @skip_on_cudasim('cudasim does not check signatures')
numba/cuda/tests/cudapy/test_device_func.py:        f1 = cuda.declare_device('f1', int32(float32[:]))
numba/cuda/tests/cudapy/test_device_func.py:    @skip_on_cudasim('cudasim does not check signatures')
numba/cuda/tests/cudapy/test_device_func.py:        f1 = cuda.declare_device('f1', 'int32(float32[:])')
numba/cuda/tests/cudapy/test_device_func.py:    @skip_on_cudasim('cudasim does not check signatures')
numba/cuda/tests/cudapy/test_device_func.py:            cuda.declare_device('f1', (float32[:],))
numba/cuda/tests/cudapy/test_device_func.py:    @skip_on_cudasim('cudasim does not check signatures')
numba/cuda/tests/cudapy/test_device_func.py:            cuda.declare_device('f1', '(float32[:],)')
numba/cuda/tests/cudapy/test_cooperative_groups.py:from numba import config, cuda, int32
numba/cuda/tests/cudapy/test_cooperative_groups.py:from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
numba/cuda/tests/cudapy/test_cooperative_groups.py:                                skip_unless_cc_60, skip_if_cudadevrt_missing,
numba/cuda/tests/cudapy/test_cooperative_groups.py:@cuda.jit
numba/cuda/tests/cudapy/test_cooperative_groups.py:    cuda.cg.this_grid()
numba/cuda/tests/cudapy/test_cooperative_groups.py:@cuda.jit
numba/cuda/tests/cudapy/test_cooperative_groups.py:    g = cuda.cg.this_grid()
numba/cuda/tests/cudapy/test_cooperative_groups.py:@cuda.jit
numba/cuda/tests/cudapy/test_cooperative_groups.py:    A[0] = cuda.grid(1)
numba/cuda/tests/cudapy/test_cooperative_groups.py:    col = cuda.grid(1)
numba/cuda/tests/cudapy/test_cooperative_groups.py:    g = cuda.cg.this_grid()
numba/cuda/tests/cudapy/test_cooperative_groups.py:@skip_if_cudadevrt_missing
numba/cuda/tests/cudapy/test_cooperative_groups.py:class TestCudaCooperativeGroups(CUDATestCase):
numba/cuda/tests/cudapy/test_cooperative_groups.py:        # Ensure the kernel executed beyond the call to cuda.this_grid()
numba/cuda/tests/cudapy/test_cooperative_groups.py:    @skip_on_cudasim("Simulator doesn't differentiate between normal and "
numba/cuda/tests/cudapy/test_cooperative_groups.py:        # Ensure the kernel executed beyond the call to cuda.sync_group()
numba/cuda/tests/cudapy/test_cooperative_groups.py:    @skip_on_cudasim("Simulator doesn't differentiate between normal and "
numba/cuda/tests/cudapy/test_cooperative_groups.py:    @skip_on_cudasim("Simulator does not implement linking")
numba/cuda/tests/cudapy/test_cooperative_groups.py:    def test_false_cooperative_doesnt_link_cudadevrt(self):
numba/cuda/tests/cudapy/test_cooperative_groups.py:        We should only mark a kernel as cooperative and link cudadevrt if the
numba/cuda/tests/cudapy/test_cooperative_groups.py:                self.assertNotIn('cudadevrt', link)
numba/cuda/tests/cudapy/test_cooperative_groups.py:        if config.ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_cooperative_groups.py:        c_sequential_rows = cuda.jit(sig)(sequential_rows)
numba/cuda/tests/cudapy/test_cooperative_groups.py:            unittest.skip("GPU cannot support enough cooperative grid blocks")
numba/cuda/tests/cudapy/test_cooperative_groups.py:        c_sequential_rows = cuda.jit(sig)(sequential_rows)
numba/cuda/tests/cudapy/test_laplace.py:from numba import cuda, float64, void
numba/cuda/tests/cudapy/test_laplace.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_laplace.py:# NOTE: CUDA kernel does not return any value
numba/cuda/tests/cudapy/test_laplace.py:if config.ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_laplace.py:class TestCudaLaplace(CUDATestCase):
numba/cuda/tests/cudapy/test_laplace.py:        @cuda.jit(float64(float64, float64), device=True, inline=True)
numba/cuda/tests/cudapy/test_laplace.py:        @cuda.jit(void(float64[:, :], float64[:, :], float64[:, :]))
numba/cuda/tests/cudapy/test_laplace.py:            err_sm = cuda.shared.array(SM_SIZE, dtype=float64)
numba/cuda/tests/cudapy/test_laplace.py:            ty = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_laplace.py:            tx = cuda.threadIdx.y
numba/cuda/tests/cudapy/test_laplace.py:            bx = cuda.blockIdx.x
numba/cuda/tests/cudapy/test_laplace.py:            by = cuda.blockIdx.y
numba/cuda/tests/cudapy/test_laplace.py:            i, j = cuda.grid(2)
numba/cuda/tests/cudapy/test_laplace.py:            cuda.syncthreads()
numba/cuda/tests/cudapy/test_laplace.py:                cuda.syncthreads()
numba/cuda/tests/cudapy/test_laplace.py:                cuda.syncthreads()
numba/cuda/tests/cudapy/test_laplace.py:        if config.ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_laplace.py:        stream = cuda.stream()
numba/cuda/tests/cudapy/test_laplace.py:        dA = cuda.to_device(A, stream)          # to device and don't come back
numba/cuda/tests/cudapy/test_laplace.py:        dAnew = cuda.to_device(Anew, stream)    # to device and don't come back
numba/cuda/tests/cudapy/test_laplace.py:        derror_grid = cuda.to_device(error_grid, stream)
numba/cuda/tests/cudapy/test_casting.py:from numba.cuda import compile_ptx
numba/cuda/tests/cudapy/test_casting.py:from numba import cuda
numba/cuda/tests/cudapy/test_casting.py:from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
numba/cuda/tests/cudapy/test_casting.py:# float16s yet, and the host does not implement cuda.fp16.*, we need two
numba/cuda/tests/cudapy/test_casting.py:# - The device version uses cuda.fp16.hmul
numba/cuda/tests/cudapy/test_casting.py:def cuda_int_literal_to_float16(x):
numba/cuda/tests/cudapy/test_casting.py:    return cuda.fp16.hmul(np.float16(x), 2)
numba/cuda/tests/cudapy/test_casting.py:def cuda_float_literal_to_float16(x):
numba/cuda/tests/cudapy/test_casting.py:    return cuda.fp16.hmul(np.float16(x), 2.5)
numba/cuda/tests/cudapy/test_casting.py:class TestCasting(CUDATestCase):
numba/cuda/tests/cudapy/test_casting.py:        wrapped_func = cuda.jit(device=True)(pyfunc)
numba/cuda/tests/cudapy/test_casting.py:        @cuda.jit
numba/cuda/tests/cudapy/test_casting.py:        def cuda_wrapper_fn(arg, res):
numba/cuda/tests/cudapy/test_casting.py:            cuda_wrapper_fn[1, 1](argarray, resarray)
numba/cuda/tests/cudapy/test_casting.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_casting.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_casting.py:        cudafuncs = (cuda_int_literal_to_float16,
numba/cuda/tests/cudapy/test_casting.py:                     cuda_float_literal_to_float16)
numba/cuda/tests/cudapy/test_casting.py:        for cudafunc, hostfunc in zip(cudafuncs, hostfuncs):
numba/cuda/tests/cudapy/test_casting.py:            with self.subTest(func=cudafunc):
numba/cuda/tests/cudapy/test_casting.py:                cfunc = self._create_wrapped(cudafunc, np.float16, np.float16)
numba/cuda/tests/cudapy/test_casting.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_casting.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_casting.py:                # the CUDA target doesn't yet implement division (or operators)
numba/cuda/tests/cudapy/test_casting.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_casting.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_casting.py:        float32_ptx, _ = cuda.compile_ptx(native_cast, (float32,), device=True)
numba/cuda/tests/cudapy/test_casting.py:        float16_ptx, _ = cuda.compile_ptx(native_cast, (float16,), device=True)
numba/cuda/tests/cudapy/test_minmax.py:from numba import cuda, float64
numba/cuda/tests/cudapy/test_minmax.py:from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
numba/cuda/tests/cudapy/test_minmax.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_minmax.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_minmax.py:@skip_on_cudasim('Tests PTX emission')
numba/cuda/tests/cudapy/test_minmax.py:class TestCudaMinMax(CUDATestCase):
numba/cuda/tests/cudapy/test_minmax.py:        kernel = cuda.jit(kernel)
numba/cuda/tests/cudapy/test_vectorize_scalar_arg.py:from numba import cuda, float64
numba/cuda/tests/cudapy/test_vectorize_scalar_arg.py:from numba.cuda.testing import skip_on_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_vectorize_scalar_arg.py:@skip_on_cudasim('ufunc API unsupported in the simulator')
numba/cuda/tests/cudapy/test_vectorize_scalar_arg.py:class TestCUDAVectorizeScalarArg(CUDATestCase):
numba/cuda/tests/cudapy/test_vectorize_scalar_arg.py:        @vectorize(sig, target='cuda')
numba/cuda/tests/cudapy/test_vectorize_scalar_arg.py:        dA = cuda.to_device(A)
numba/cuda/tests/cudapy/test_vectorize_scalar_arg.py:        @vectorize(sig, target='cuda')
numba/cuda/tests/cudapy/test_debug.py:from numba.cuda.testing import skip_on_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_debug.py:from numba import cuda, float64
numba/cuda/tests/cudapy/test_debug.py:def simple_cuda(A, B):
numba/cuda/tests/cudapy/test_debug.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_debug.py:@skip_on_cudasim('Simulator does not produce debug dumps')
numba/cuda/tests/cudapy/test_debug.py:class TestDebugOutput(CUDATestCase):
numba/cuda/tests/cudapy/test_debug.py:    def compile_simple_cuda(self):
numba/cuda/tests/cudapy/test_debug.py:                cfunc = cuda.jit((float64[:], float64[:]))(simple_cuda)
numba/cuda/tests/cudapy/test_debug.py:        self.assertIn('--IR DUMP: simple_cuda--', out)
numba/cuda/tests/cudapy/test_debug.py:        self.assertIn('--ASSEMBLY simple_cuda', out)
numba/cuda/tests/cudapy/test_debug.py:        self.assertIn('Generated by NVIDIA NVVM Compiler', out)
numba/cuda/tests/cudapy/test_debug.py:            out = self.compile_simple_cuda()
numba/cuda/tests/cudapy/test_debug.py:            out = self.compile_simple_cuda()
numba/cuda/tests/cudapy/test_debug.py:            out = self.compile_simple_cuda()
numba/cuda/tests/cudapy/test_debug.py:            out = self.compile_simple_cuda()
numba/cuda/tests/cudapy/test_debug.py:            out = self.compile_simple_cuda()
numba/cuda/tests/cudapy/test_exception.py:from numba import cuda
numba/cuda/tests/cudapy/test_exception.py:from numba.cuda.testing import unittest, xfail_unless_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_exception.py:class TestException(CUDATestCase):
numba/cuda/tests/cudapy/test_exception.py:            x = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_exception.py:        unsafe_foo = cuda.jit(foo)
numba/cuda/tests/cudapy/test_exception.py:        safe_foo = cuda.jit(debug=True, opt=False)(foo)
numba/cuda/tests/cudapy/test_exception.py:        if not config.ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_exception.py:        @cuda.jit(debug=True, opt=False)
numba/cuda/tests/cudapy/test_exception.py:        @cuda.jit(debug=with_debug_mode, opt=with_opt_mode)
numba/cuda/tests/cudapy/test_exception.py:            tid = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_exception.py:            ntid = cuda.blockDim.x
numba/cuda/tests/cudapy/test_exception.py:            cuda.syncthreads()
numba/cuda/tests/cudapy/test_exception.py:        @cuda.jit
numba/cuda/tests/cudapy/test_exception.py:            tid = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_exception.py:            ntid = cuda.blockDim.x
numba/cuda/tests/cudapy/test_exception.py:            cuda.syncthreads()
numba/cuda/tests/cudapy/test_exception.py:        @cuda.jit
numba/cuda/tests/cudapy/test_exception.py:        @cuda.jit(debug=True, opt=False)
numba/cuda/tests/cudapy/test_exception.py:        # raised - in debug mode, the CUDA target uses the Python error model,
numba/cuda/tests/cudapy/test_exception.py:        if config.ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_exception.py:    @xfail_unless_cudasim
numba/cuda/tests/cudapy/test_exception.py:        @cuda.jit(device=True)
numba/cuda/tests/cudapy/test_exception.py:        @cuda.jit(debug=True)
numba/cuda/tests/cudapy/test_reduction.py:from numba import cuda
numba/cuda/tests/cudapy/test_reduction.py:from numba.core.config import ENABLE_CUDASIM
numba/cuda/tests/cudapy/test_reduction.py:from numba.cuda.testing import CUDATestCase
numba/cuda/tests/cudapy/test_reduction.py:sum_reduce = cuda.Reduce(lambda a, b: a + b)
numba/cuda/tests/cudapy/test_reduction.py:class TestReduction(CUDATestCase):
numba/cuda/tests/cudapy/test_reduction.py:        if ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_reduction.py:        dA = cuda.to_device(A)
numba/cuda/tests/cudapy/test_reduction.py:        prod_reduce = cuda.reduce(lambda a, b: a * b)
numba/cuda/tests/cudapy/test_reduction.py:        max_reduce = cuda.Reduce(lambda a, b: max(a, b))
numba/cuda/tests/cudapy/test_reduction.py:        got = cuda.to_device(np.zeros(1, dtype=np.float64))
numba/cuda/tests/cudapy/test_matmul.py:from numba import cuda, float32, void
numba/cuda/tests/cudapy/test_matmul.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_matmul.py:if config.ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_matmul.py:class TestCudaMatMul(CUDATestCase):
numba/cuda/tests/cudapy/test_matmul.py:        @cuda.jit(void(float32[:, ::1], float32[:, ::1], float32[:, ::1]))
numba/cuda/tests/cudapy/test_matmul.py:            sA = cuda.shared.array(shape=SM_SIZE, dtype=float32)
numba/cuda/tests/cudapy/test_matmul.py:            sB = cuda.shared.array(shape=(tpb, tpb), dtype=float32)
numba/cuda/tests/cudapy/test_matmul.py:            tx = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_matmul.py:            ty = cuda.threadIdx.y
numba/cuda/tests/cudapy/test_matmul.py:            bx = cuda.blockIdx.x
numba/cuda/tests/cudapy/test_matmul.py:            by = cuda.blockIdx.y
numba/cuda/tests/cudapy/test_matmul.py:            bw = cuda.blockDim.x
numba/cuda/tests/cudapy/test_matmul.py:            bh = cuda.blockDim.y
numba/cuda/tests/cudapy/test_matmul.py:                cuda.syncthreads()
numba/cuda/tests/cudapy/test_matmul.py:                cuda.syncthreads()
numba/cuda/tests/cudapy/test_matmul.py:        stream = cuda.stream()
numba/cuda/tests/cudapy/test_matmul.py:            dA = cuda.to_device(A, stream)
numba/cuda/tests/cudapy/test_matmul.py:            dB = cuda.to_device(B, stream)
numba/cuda/tests/cudapy/test_matmul.py:            dC = cuda.to_device(C, stream)
numba/cuda/tests/cudapy/test_multiprocessing.py:from numba import cuda
numba/cuda/tests/cudapy/test_multiprocessing.py:from numba.cuda.testing import skip_on_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_multiprocessing.py:    from numba.cuda.cudadrv.error import CudaDriverError
numba/cuda/tests/cudapy/test_multiprocessing.py:        cuda.to_device(np.arange(1))
numba/cuda/tests/cudapy/test_multiprocessing.py:    except CudaDriverError as e:
numba/cuda/tests/cudapy/test_multiprocessing.py:@skip_on_cudasim('disabled for cudasim')
numba/cuda/tests/cudapy/test_multiprocessing.py:class TestMultiprocessing(CUDATestCase):
numba/cuda/tests/cudapy/test_multiprocessing.py:        cuda.current_context()  # force cuda initialize
numba/cuda/tests/cudapy/test_multiprocessing.py:        # fork in process that also uses CUDA
numba/cuda/tests/cudapy/test_multiprocessing.py:        self.assertIn('CUDA initialized before forking', str(exc))
numba/cuda/tests/cudapy/test_gufunc.py:from numba import cuda
numba/cuda/tests/cudapy/test_gufunc.py:from numba.cuda.testing import skip_on_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_gufunc.py:                 target='cuda')
numba/cuda/tests/cudapy/test_gufunc.py:@skip_on_cudasim('ufunc API unsupported in the simulator')
numba/cuda/tests/cudapy/test_gufunc.py:class TestCUDAGufunc(CUDATestCase):
numba/cuda/tests/cudapy/test_gufunc.py:        dB = cuda.to_device(B)
numba/cuda/tests/cudapy/test_gufunc.py:        matrix_ct = 1001 # an odd number to test thread/block division in CUDA
numba/cuda/tests/cudapy/test_gufunc.py:        matrix_ct = 100 # an odd number to test thread/block division in CUDA
numba/cuda/tests/cudapy/test_gufunc.py:        #cuda.driver.flush_pending_free()
numba/cuda/tests/cudapy/test_gufunc.py:        matrix_ct = 1001 # an odd number to test thread/block division in CUDA
numba/cuda/tests/cudapy/test_gufunc.py:        stream = cuda.stream()
numba/cuda/tests/cudapy/test_gufunc.py:        dA = cuda.to_device(A, stream)
numba/cuda/tests/cudapy/test_gufunc.py:        dB = cuda.to_device(B, stream)
numba/cuda/tests/cudapy/test_gufunc.py:        dC = cuda.device_array(shape=(1001, 2, 5), dtype=A.dtype, stream=stream)
numba/cuda/tests/cudapy/test_gufunc.py:                     target='cuda')
numba/cuda/tests/cudapy/test_gufunc.py:                     target='cuda')
numba/cuda/tests/cudapy/test_gufunc.py:                     target='cuda')
numba/cuda/tests/cudapy/test_gufunc.py:                     target='cuda')
numba/cuda/tests/cudapy/test_gufunc.py:                     '(n)->(n)', target='cuda')
numba/cuda/tests/cudapy/test_gufunc.py:        @cuda.jit
numba/cuda/tests/cudapy/test_gufunc.py:        def cuda_jit(A, b):
numba/cuda/tests/cudapy/test_gufunc.py:            cuda_jit[1, 1](A, b)
numba/cuda/tests/cudapy/test_gufunc.py:    # Test inefficient use of the GPU where the inputs are all mapped onto a
numba/cuda/tests/cudapy/test_gufunc.py:                     '(n),(n)->(n)', target='cuda')
numba/cuda/tests/cudapy/test_gufunc.py:        def numba_dist_cuda(a, b, dist):
numba/cuda/tests/cudapy/test_gufunc.py:        with override_config('CUDA_LOW_OCCUPANCY_WARNINGS', 1):
numba/cuda/tests/cudapy/test_gufunc.py:                numba_dist_cuda(a, b, dist)
numba/cuda/tests/cudapy/test_gufunc.py:                     '(n),(n)->(n)', nopython=True, target='cuda')
numba/cuda/tests/cudapy/test_gufunc.py:        def numba_dist_cuda2(a, b, dist):
numba/cuda/tests/cudapy/test_gufunc.py:        with override_config('CUDA_LOW_OCCUPANCY_WARNINGS', 1):
numba/cuda/tests/cudapy/test_gufunc.py:                numba_dist_cuda2(a, b, dist)
numba/cuda/tests/cudapy/test_gufunc.py:        guvectorize([void(float32[:], float32[:])], '(x)->(x)', target='cuda',
numba/cuda/tests/cudapy/test_gufunc.py:                        target='cuda', nopython=False)(foo)
numba/cuda/tests/cudapy/test_gufunc.py:                        target='cuda', what1=True, ever2=False)(foo)
numba/cuda/tests/cudapy/test_gufunc.py:        @guvectorize([void(float32[:], float32[:])], '(x)->(x)', target='cuda')
numba/cuda/tests/cudapy/test_gufunc.py:                     target='cuda')
numba/cuda/tests/cudapy/test_gufunc.py:            @guvectorize([int32(int32[:], int32[:])], '(m)->(m)', target='cuda')
numba/cuda/tests/cudapy/test_gufunc.py:                     '(m),(m)->(m)', target='cuda')
numba/cuda/tests/cudapy/test_gufunc.py:@skip_on_cudasim('ufunc API unsupported in the simulator')
numba/cuda/tests/cudapy/test_gufunc.py:class TestMultipleOutputs(CUDATestCase):
numba/cuda/tests/cudapy/test_gufunc.py:                     target='cuda')
numba/cuda/tests/cudapy/test_gufunc.py:                     target='cuda')
numba/cuda/tests/cudapy/test_gufunc.py:                     target='cuda')
numba/cuda/tests/cudapy/test_gufunc.py:                     target='cuda')
numba/cuda/tests/cudapy/test_gufunc.py:                     '(m),(m)->(m),(m)', target='cuda')
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:from numba import cuda
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:from numba.cuda.testing import CUDATestCase
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:class TestCudaJitNoTypes(CUDATestCase):
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:        @cuda.jit
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:        dx = cuda.to_device(x)
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:        dy = cuda.to_device(y)
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:        @cuda.jit(device=True)
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:        @cuda.jit(device=True)
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:        @cuda.jit
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:        @cuda.jit(device=True)
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:        @cuda.jit
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:        stream = cuda.stream()
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:        d_a = cuda.to_device(a, stream)
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:        d_b = cuda.to_device(b, stream)
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:        with override_config('ENABLE_CUDASIM', 1):
numba/cuda/tests/cudapy/test_cuda_jit_no_types.py:            @cuda.jit(debug=True)
numba/cuda/tests/cudapy/test_inspect.py:from numba import cuda, float32, float64, int32, intp
numba/cuda/tests/cudapy/test_inspect.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_inspect.py:from numba.cuda.testing import (skip_on_cudasim, skip_with_nvdisasm,
numba/cuda/tests/cudapy/test_inspect.py:@skip_on_cudasim('Simulator does not generate code to be inspected')
numba/cuda/tests/cudapy/test_inspect.py:class TestInspect(CUDATestCase):
numba/cuda/tests/cudapy/test_inspect.py:        return cuda.current_context().device.compute_capability
numba/cuda/tests/cudapy/test_inspect.py:        @cuda.jit(sig)
numba/cuda/tests/cudapy/test_inspect.py:        self.assertIn('cuda.kernel.wrapper', llvm)
numba/cuda/tests/cudapy/test_inspect.py:        self.assertIn("Generated by NVIDIA NVVM Compiler", asm)
numba/cuda/tests/cudapy/test_inspect.py:        @cuda.jit
numba/cuda/tests/cudapy/test_inspect.py:        self.assertIn('cuda.kernel.wrapper', llvmirs[intp, intp])
numba/cuda/tests/cudapy/test_inspect.py:        self.assertIn('cuda.kernel.wrapper', llvmirs[float64, float64])
numba/cuda/tests/cudapy/test_inspect.py:        @cuda.jit(sig, lineinfo=True)
numba/cuda/tests/cudapy/test_inspect.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_inspect.py:        @cuda.jit(lineinfo=True)
numba/cuda/tests/cudapy/test_inspect.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_inspect.py:        @cuda.jit((float32[::1],))
numba/cuda/tests/cudapy/test_inspect.py:        @cuda.jit(sig)
numba/cuda/tests/cudapy/test_inspect.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_print.py:from numba.cuda.testing import CUDATestCase, skip_on_cudasim
numba/cuda/tests/cudapy/test_print.py:from numba import cuda
numba/cuda/tests/cudapy/test_print.py:@cuda.jit
numba/cuda/tests/cudapy/test_print.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_print.py:cuda.synchronize()
numba/cuda/tests/cudapy/test_print.py:from numba import cuda
numba/cuda/tests/cudapy/test_print.py:@cuda.jit
numba/cuda/tests/cudapy/test_print.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_print.py:cuda.synchronize()
numba/cuda/tests/cudapy/test_print.py:from numba import cuda
numba/cuda/tests/cudapy/test_print.py:@cuda.jit
numba/cuda/tests/cudapy/test_print.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_print.py:cuda.synchronize()
numba/cuda/tests/cudapy/test_print.py:from numba import cuda
numba/cuda/tests/cudapy/test_print.py:@cuda.jit
numba/cuda/tests/cudapy/test_print.py:cuda.synchronize()
numba/cuda/tests/cudapy/test_print.py:from numba import cuda
numba/cuda/tests/cudapy/test_print.py:@cuda.jit
numba/cuda/tests/cudapy/test_print.py:cuda.synchronize()
numba/cuda/tests/cudapy/test_print.py:class TestPrint(CUDATestCase):
numba/cuda/tests/cudapy/test_print.py:        # The output of GPU threads is intermingled, but each print()
numba/cuda/tests/cudapy/test_print.py:        # CUDA and the simulator use different formats for float formatting
numba/cuda/tests/cudapy/test_print.py:    @skip_on_cudasim('cudasim can print unlimited output')
numba/cuda/tests/cudapy/test_print.py:        # than 32 arguments, in common with CUDA C/C++ printf - this is due to
numba/cuda/tests/cudapy/test_print.py:        # a limitation in CUDA vprintf, see:
numba/cuda/tests/cudapy/test_print.py:        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#limitations
numba/cuda/tests/cudapy/test_print.py:        warn_msg = ('CUDA print() cannot print more than 32 items. The raw '
numba/cuda/tests/cudapy/test_powi.py:from numba import cuda, float64, int8, int32, void
numba/cuda/tests/cudapy/test_powi.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_powi.py:    y, x = cuda.grid(2)
numba/cuda/tests/cudapy/test_powi.py:    y, x = cuda.grid(2)
numba/cuda/tests/cudapy/test_powi.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_powi.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_powi.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_powi.py:class TestCudaPowi(CUDATestCase):
numba/cuda/tests/cudapy/test_powi.py:        dec = cuda.jit(void(float64[:, :], int8, float64[:, :]))
numba/cuda/tests/cudapy/test_powi.py:        dec = cuda.jit(void(float64[:, :], int8, float64[:, :]))
numba/cuda/tests/cudapy/test_powi.py:        cfunc = cuda.jit(func)
numba/cuda/tests/cudapy/test_powi.py:        cfunc = cuda.jit(vec_pow_inplace_binop)
numba/cuda/tests/cudapy/test_array_methods.py:from numba import cuda
numba/cuda/tests/cudapy/test_array_methods.py:from numba.cuda.testing import CUDATestCase
numba/cuda/tests/cudapy/test_array_methods.py:class TestCudaArrayMethods(CUDATestCase):
numba/cuda/tests/cudapy/test_array_methods.py:        Reinterpret byte array as int32 in the GPU.
numba/cuda/tests/cudapy/test_array_methods.py:        kernel = cuda.jit(pyfunc)
numba/cuda/tests/cudapy/test_complex.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_complex.py:from numba import cuda
numba/cuda/tests/cudapy/test_complex.py:    device_func = cuda.jit(restype(*argtypes), device=True)(pyfunc)
numba/cuda/tests/cudapy/test_complex.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_complex.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_complex.py:    kernel = cuda.jit(tuple(kernel_types))(kernel_func)
numba/cuda/tests/cudapy/test_complex.py:class BaseComplexTest(CUDATestCase):
numba/cuda/tests/cudapy/test_complex.py:            cudafunc = compile_scalar_func(pyfunc, sig.args, sig.return_type)
numba/cuda/tests/cudapy/test_complex.py:            got_list = cudafunc(ok_values)
numba/cuda/tests/cudapy/test_complex.py:class TestAtomicOnComplexComponents(CUDATestCase):
numba/cuda/tests/cudapy/test_complex.py:        @cuda.jit
numba/cuda/tests/cudapy/test_complex.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_complex.py:            cuda.atomic.add(values.real, i, 1)
numba/cuda/tests/cudapy/test_complex.py:        @cuda.jit
numba/cuda/tests/cudapy/test_complex.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_complex.py:            cuda.atomic.add(values.imag, i, 1)
numba/cuda/tests/cudapy/test_libdevice.py:from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
numba/cuda/tests/cudapy/test_libdevice.py:from numba import cuda
numba/cuda/tests/cudapy/test_libdevice.py:from numba.cuda import libdevice, compile_ptx
numba/cuda/tests/cudapy/test_libdevice.py:from numba.cuda.libdevicefuncs import functions, create_signature
numba/cuda/tests/cudapy/test_libdevice.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_libdevice.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_libdevice.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_libdevice.py:@skip_on_cudasim('Libdevice functions are not supported on cudasim')
numba/cuda/tests/cudapy/test_libdevice.py:class TestLibdevice(CUDATestCase):
numba/cuda/tests/cudapy/test_libdevice.py:        cufunc = cuda.jit(use_sincos)
numba/cuda/tests/cudapy/test_libdevice.py:        cufunc = cuda.jit(use_frexp)
numba/cuda/tests/cudapy/test_libdevice.py:        cufunc = cuda.jit(use_sad)
numba/cuda/tests/cudapy/test_libdevice.py:from numba.cuda import libdevice
numba/cuda/tests/cudapy/test_libdevice.py:@skip_on_cudasim('Compilation to PTX is not supported on cudasim')
numba/cuda/tests/cudapy/test_vector_type.py:CUDA vector type tests. Note that this test file imports
numba/cuda/tests/cudapy/test_vector_type.py:`cuda.vector_type` module to programmatically test all the
numba/cuda/tests/cudapy/test_vector_type.py:corresponding vector type from `cuda` module in kernel to use them.
numba/cuda/tests/cudapy/test_vector_type.py:from numba.cuda.testing import CUDATestCase
numba/cuda/tests/cudapy/test_vector_type.py:from numba import cuda
numba/cuda/tests/cudapy/test_vector_type.py:if config.ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_vector_type.py:    from numba.cuda.simulator.vector_types import vector_types
numba/cuda/tests/cudapy/test_vector_type.py:    from numba.cuda.vector_types import vector_types
numba/cuda/tests/cudapy/test_vector_type.py:    return cuda.jit(host_function)
numba/cuda/tests/cudapy/test_vector_type.py:    v1 = getattr(cuda, f"{vtype.name[:-1]}1")
numba/cuda/tests/cudapy/test_vector_type.py:    v2 = getattr(cuda, f"{vtype.name[:-1]}2")
numba/cuda/tests/cudapy/test_vector_type.py:    v3 = getattr(cuda, f"{vtype.name[:-1]}3")
numba/cuda/tests/cudapy/test_vector_type.py:    v4 = getattr(cuda, f"{vtype.name[:-1]}4")
numba/cuda/tests/cudapy/test_vector_type.py:    return cuda.jit(kernel)
numba/cuda/tests/cudapy/test_vector_type.py:class TestCudaVectorType(CUDATestCase):
numba/cuda/tests/cudapy/test_vector_type.py:        are available within the cuda module from both device and
numba/cuda/tests/cudapy/test_vector_type.py:        @cuda.jit("void(float64[:])")
numba/cuda/tests/cudapy/test_vector_type.py:            v1 = cuda.float64x4(1.0, 3.0, 5.0, 7.0)
numba/cuda/tests/cudapy/test_vector_type.py:            v2 = cuda.short2(10, 11)
numba/cuda/tests/cudapy/test_vector_type.py:        """Tests that `cuda.<vector_type.alias>` are importable and
numba/cuda/tests/cudapy/test_vector_type.py:        that is the same as `cuda.<vector_type.name>`.
numba/cuda/tests/cudapy/test_vector_type.py:                        id(getattr(cuda, vty.name)), id(getattr(cuda, alias))
numba/cuda/tests/cudapy/cache_with_cpu_usecases.py:from numba import cuda, njit
numba/cuda/tests/cudapy/cache_with_cpu_usecases.py:from numba.cuda.testing import CUDATestCase
numba/cuda/tests/cudapy/cache_with_cpu_usecases.py:from numba.cuda.tests.cudapy.cache_usecases import CUDAUseCase, UseCase
numba/cuda/tests/cudapy/cache_with_cpu_usecases.py:# Using the same function as a cached CPU and CUDA-jitted function
numba/cuda/tests/cudapy/cache_with_cpu_usecases.py:assign_cuda_kernel = cuda.jit(cache=True)(target_shared_assign)
numba/cuda/tests/cudapy/cache_with_cpu_usecases.py:assign_cuda = CUDAUseCase(assign_cuda_kernel)
numba/cuda/tests/cudapy/cache_with_cpu_usecases.py:class _TestModule(CUDATestCase):
numba/cuda/tests/cudapy/cache_with_cpu_usecases.py:        self.assertPreciseEqual(mod.assign_cuda(5), 5)
numba/cuda/tests/cudapy/cache_with_cpu_usecases.py:        self.assertPreciseEqual(mod.assign_cuda(5.5), 5.5)
numba/cuda/tests/cudapy/test_errors.py:from numba import cuda
numba/cuda/tests/cudapy/test_errors.py:from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
numba/cuda/tests/cudapy/test_errors.py:class TestJitErrors(CUDATestCase):
numba/cuda/tests/cudapy/test_errors.py:        kernfunc = cuda.jit(noop)
numba/cuda/tests/cudapy/test_errors.py:        kernfunc = cuda.jit(noop)
numba/cuda/tests/cudapy/test_errors.py:    def test_unconfigured_typed_cudakernel(self):
numba/cuda/tests/cudapy/test_errors.py:        kernfunc = cuda.jit("void(int32)")(noop)
numba/cuda/tests/cudapy/test_errors.py:    def test_unconfigured_untyped_cudakernel(self):
numba/cuda/tests/cudapy/test_errors.py:        kernfunc = cuda.jit(noop)
numba/cuda/tests/cudapy/test_errors.py:    @skip_on_cudasim('TypingError does not occur on simulator')
numba/cuda/tests/cudapy/test_errors.py:        # accidentally breaking the CUDA target
numba/cuda/tests/cudapy/test_errors.py:        @cuda.jit(device=True)
numba/cuda/tests/cudapy/test_errors.py:        @cuda.jit
numba/cuda/tests/cudapy/test_errors.py:        self.assertIn("resolving callee type: type(CUDADispatcher", excstr)
numba/cuda/tests/cudapy/test_random.py:from numba import cuda
numba/cuda/tests/cudapy/test_random.py:from numba.cuda.testing import unittest
numba/cuda/tests/cudapy/test_random.py:from numba.cuda.testing import skip_on_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_random.py:from numba.cuda.random import \
numba/cuda/tests/cudapy/test_random.py:@cuda.jit
numba/cuda/tests/cudapy/test_random.py:    thread_id = cuda.grid(1)
numba/cuda/tests/cudapy/test_random.py:@cuda.jit
numba/cuda/tests/cudapy/test_random.py:    thread_id = cuda.grid(1)
numba/cuda/tests/cudapy/test_random.py:class TestCudaRandomXoroshiro128p(CUDATestCase):
numba/cuda/tests/cudapy/test_random.py:        states = cuda.random.create_xoroshiro128p_states(10, seed=1)
numba/cuda/tests/cudapy/test_random.py:        states = cuda.random.create_xoroshiro128p_states(10, seed=1)
numba/cuda/tests/cudapy/test_random.py:        states = cuda.random.create_xoroshiro128p_states(10, seed=1,
numba/cuda/tests/cudapy/test_random.py:        stream = cuda.stream()
numba/cuda/tests/cudapy/test_random.py:        states = cuda.random.create_xoroshiro128p_states(10, seed=1,
numba/cuda/tests/cudapy/test_random.py:        states = cuda.random.create_xoroshiro128p_states(32 * 2, seed=1)
numba/cuda/tests/cudapy/test_random.py:    @skip_on_cudasim('skip test for speed under cudasim')
numba/cuda/tests/cudapy/test_random.py:        states = cuda.random.create_xoroshiro128p_states(32 * 2, seed=1)
numba/cuda/tests/cudapy/test_random.py:    @skip_on_cudasim('skip test for speed under cudasim')
numba/cuda/tests/cudapy/test_const_string.py:from numba import cuda
numba/cuda/tests/cudapy/test_const_string.py:from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_const_string.py:@skip_on_cudasim("This is testing CUDA backend code generation")
numba/cuda/tests/cudapy/test_const_string.py:        # These imports are incompatible with CUDASIM
numba/cuda/tests/cudapy/test_const_string.py:        from numba.cuda.descriptor import cuda_target
numba/cuda/tests/cudapy/test_const_string.py:        from numba.cuda.cudadrv.nvvm import compile_ir
numba/cuda/tests/cudapy/test_const_string.py:        targetctx = cuda_target.target_context
numba/cuda/tests/cudapy/test_const_string.py:class TestConstString(CUDATestCase):
numba/cuda/tests/cudapy/test_const_string.py:        @cuda.jit
numba/cuda/tests/cudapy/test_const_string.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_const_string.py:        @cuda.jit
numba/cuda/tests/cudapy/test_const_string.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_const_string.py:        @cuda.jit
numba/cuda/tests/cudapy/test_const_string.py:        @cuda.jit
numba/cuda/tests/cudapy/test_alignment.py:from numba import from_dtype, cuda
numba/cuda/tests/cudapy/test_alignment.py:from numba.cuda.testing import skip_on_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_alignment.py:class TestAlignment(CUDATestCase):
numba/cuda/tests/cudapy/test_alignment.py:        @cuda.jit((rec[:],))
numba/cuda/tests/cudapy/test_alignment.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_alignment.py:    @skip_on_cudasim('Simulator does not check alignment')
numba/cuda/tests/cudapy/test_alignment.py:            @cuda.jit((rec[:],))
numba/cuda/tests/cudapy/test_alignment.py:                i = cuda.grid(1)
numba/cuda/tests/cudapy/test_gufunc_scalar.py:from numba import guvectorize, cuda
numba/cuda/tests/cudapy/test_gufunc_scalar.py:from numba.cuda.testing import skip_on_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_gufunc_scalar.py:@skip_on_cudasim('ufunc API unsupported in the simulator')
numba/cuda/tests/cudapy/test_gufunc_scalar.py:class TestGUFuncScalar(CUDATestCase):
numba/cuda/tests/cudapy/test_gufunc_scalar.py:        @guvectorize(['void(int32[:], int32[:])'], '(n)->()', target='cuda')
numba/cuda/tests/cudapy/test_gufunc_scalar.py:        # is performed. But, broadcasting on CUDA arrays is not supported.
numba/cuda/tests/cudapy/test_gufunc_scalar.py:        # invoke on CUDA with manually managed memory
numba/cuda/tests/cudapy/test_gufunc_scalar.py:        dev_inp = cuda.to_device(
numba/cuda/tests/cudapy/test_gufunc_scalar.py:        dev_out1 = cuda.to_device(out1, copy=False)   # alloc only
numba/cuda/tests/cudapy/test_gufunc_scalar.py:        @guvectorize(['void(int32, int32[:])'], '()->()', target='cuda')
numba/cuda/tests/cudapy/test_gufunc_scalar.py:                     '(),(t),(t)->(t)', target='cuda')
numba/cuda/tests/cudapy/test_gufunc_scalar.py:                     target='cuda')
numba/cuda/tests/cudapy/test_gufunc_scalar.py:        da = cuda.to_device(a)
numba/cuda/tests/cudapy/test_gufunc_scalar.py:                     target='cuda')
numba/cuda/tests/cudapy/test_slicing.py:from numba import cuda
numba/cuda/tests/cudapy/test_slicing.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_slicing.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_slicing.py:class TestCudaSlicing(CUDATestCase):
numba/cuda/tests/cudapy/test_slicing.py:        cufoo = cuda.jit("void(int32[:], int32[:])", device=True)(foo)
numba/cuda/tests/cudapy/test_slicing.py:        cucopy = cuda.jit("void(int32[:,:], int32[:,:])")(copy)
numba/cuda/tests/cudapy/test_slicing.py:        # CudaAPIError.
numba/cuda/tests/cudapy/test_slicing.py:        arr = cuda.device_array(len(a))
numba/cuda/tests/cudapy/test_slicing.py:        arr[:] = cuda.to_device(a)
numba/cuda/tests/cudapy/test_enums.py:from numba import cuda, vectorize, njit
numba/cuda/tests/cudapy/test_enums.py:from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
numba/cuda/tests/cudapy/test_enums.py:class EnumTest(CUDATestCase):
numba/cuda/tests/cudapy/test_enums.py:        cuda_f = cuda.jit(f)
numba/cuda/tests/cudapy/test_enums.py:            cuda_f[1, 1](a, b, got)
numba/cuda/tests/cudapy/test_enums.py:        cuda_f = cuda.jit(f)
numba/cuda/tests/cudapy/test_enums.py:        cuda_f[1, 1](got)
numba/cuda/tests/cudapy/test_enums.py:        cuda_f = cuda.jit(f)
numba/cuda/tests/cudapy/test_enums.py:        cuda_f[1, 1](True, got)
numba/cuda/tests/cudapy/test_enums.py:        cuda_f = cuda.jit(f)
numba/cuda/tests/cudapy/test_enums.py:            cuda_f[1, 1](x, got)
numba/cuda/tests/cudapy/test_enums.py:        cuda_f = cuda.jit(f)
numba/cuda/tests/cudapy/test_enums.py:            cuda_f[1, 1](x, got)
numba/cuda/tests/cudapy/test_enums.py:    @skip_on_cudasim("ufuncs are unsupported on simulator.")
numba/cuda/tests/cudapy/test_enums.py:        cuda_func = vectorize("int64(int64)", target='cuda')(f)
numba/cuda/tests/cudapy/test_enums.py:        got = cuda_func(arr)
numba/cuda/tests/cudapy/recursion_usecases.py:Usecases of recursive functions in the CUDA target, many derived from
numba/cuda/tests/cudapy/recursion_usecases.py:from numba import cuda
numba/cuda/tests/cudapy/recursion_usecases.py:@cuda.jit("i8(i8)", device=True)
numba/cuda/tests/cudapy/recursion_usecases.py:    @cuda.jit("i8(i8)", device=True)
numba/cuda/tests/cudapy/recursion_usecases.py:@cuda.jit
numba/cuda/tests/cudapy/recursion_usecases.py:@cuda.jit(device=True)
numba/cuda/tests/cudapy/recursion_usecases.py:@cuda.jit(device=True)
numba/cuda/tests/cudapy/recursion_usecases.py:@cuda.jit(device=True)
numba/cuda/tests/cudapy/recursion_usecases.py:@cuda.jit(debug=True, opt=False)
numba/cuda/tests/cudapy/test_math.py:from numba.cuda.testing import (skip_unless_cc_53,
numba/cuda/tests/cudapy/test_math.py:                                CUDATestCase,
numba/cuda/tests/cudapy/test_math.py:                                skip_on_cudasim)
numba/cuda/tests/cudapy/test_math.py:from numba import cuda, float32, float64, int32, vectorize, void, int64
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_math.py:class TestCudaMath(CUDATestCase):
numba/cuda/tests/cudapy/test_math.py:        cfunc = cuda.jit((arytype, restype))(func)
numba/cuda/tests/cudapy/test_math.py:        cfunc = cuda.jit((npmtype[::1], int32[::1]))(func)
numba/cuda/tests/cudapy/test_math.py:        cfunc = cuda.jit((iarytype, oarytype))(func)
numba/cuda/tests/cudapy/test_math.py:        cfunc = cuda.jit((arytype, arytype, restype))(func)
numba/cuda/tests/cudapy/test_math.py:    @skip_on_cudasim("numpy does not support trunc for float16")
numba/cuda/tests/cudapy/test_math.py:    @skip_on_cudasim('math.remainder(0, 0) raises a ValueError on CUDASim')
numba/cuda/tests/cudapy/test_math.py:        @cuda.jit(void(float64[::1], int64, int64))
numba/cuda/tests/cudapy/test_math.py:        cfunc = cuda.jit((arytype, int32[::1], arytype))(math_pow)
numba/cuda/tests/cudapy/test_math.py:    @skip_on_cudasim('trunc only supported on NumPy float64')
numba/cuda/tests/cudapy/test_math.py:            cfunc = cuda.jit((arytype, arytype, arytype))(math_modf)
numba/cuda/tests/cudapy/test_math.py:            cfunc = cuda.jit((arytype, arytype, arytype))(math_modf)
numba/cuda/tests/cudapy/extensions_usecases.py:if not config.ENABLE_CUDASIM:
numba/cuda/tests/cudapy/extensions_usecases.py:    from numba.cuda.cudaimpl import lower
numba/cuda/tests/cudapy/jitlink.ptx:// Generated by NVIDIA NVVM Compiler
numba/cuda/tests/cudapy/jitlink.ptx:// Cuda compilation tools, release 6.0, V6.0.1
numba/cuda/tests/cudapy/test_boolean.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_boolean.py:from numba import cuda
numba/cuda/tests/cudapy/test_boolean.py:class TestCudaBoolean(CUDATestCase):
numba/cuda/tests/cudapy/test_boolean.py:        func = cuda.jit('void(float64[:], bool_)')(boolean_func)
numba/cuda/tests/cudapy/test_serialize.py:from numba import cuda, vectorize
numba/cuda/tests/cudapy/test_serialize.py:from numba.cuda.testing import skip_on_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_serialize.py:@skip_on_cudasim('pickling not supported in CUDASIM')
numba/cuda/tests/cudapy/test_serialize.py:class TestPickle(CUDATestCase):
numba/cuda/tests/cudapy/test_serialize.py:        @cuda.jit(device=True)
numba/cuda/tests/cudapy/test_serialize.py:        @cuda.jit('void(intp[:])')
numba/cuda/tests/cudapy/test_serialize.py:        @cuda.jit(device=True)
numba/cuda/tests/cudapy/test_serialize.py:        @cuda.jit
numba/cuda/tests/cudapy/test_serialize.py:        @vectorize(['intp(intp)', 'float64(float64)'], target='cuda')
numba/cuda/tests/cudapy/test_serialize.py:        def cuda_vect(x):
numba/cuda/tests/cudapy/test_serialize.py:        expected = cuda_vect(ary)
numba/cuda/tests/cudapy/test_serialize.py:        foo1 = pickle.loads(pickle.dumps(cuda_vect))
numba/cuda/tests/cudapy/test_serialize.py:        del cuda_vect
numba/cuda/tests/cudapy/test_idiv.py:from numba import cuda, float32, float64, int32, void
numba/cuda/tests/cudapy/test_idiv.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_idiv.py:class TestCudaIDiv(CUDATestCase):
numba/cuda/tests/cudapy/test_idiv.py:        @cuda.jit(void(float32[:, :], int32, int32))
numba/cuda/tests/cudapy/test_idiv.py:        grid = cuda.to_device(x)
numba/cuda/tests/cudapy/test_idiv.py:        @cuda.jit(void(float64[:, :], int32, int32))
numba/cuda/tests/cudapy/test_idiv.py:        grid = cuda.to_device(x)
numba/cuda/tests/cudapy/test_debuginfo.py:from numba.cuda.testing import skip_on_cudasim
numba/cuda/tests/cudapy/test_debuginfo.py:from numba import cuda
numba/cuda/tests/cudapy/test_debuginfo.py:from numba.cuda.testing import CUDATestCase
numba/cuda/tests/cudapy/test_debuginfo.py:@skip_on_cudasim('Simulator does not produce debug dumps')
numba/cuda/tests/cudapy/test_debuginfo.py:class TestCudaDebugInfo(CUDATestCase):
numba/cuda/tests/cudapy/test_debuginfo.py:        @cuda.jit(debug=False)
numba/cuda/tests/cudapy/test_debuginfo.py:        @cuda.jit(debug=True, opt=False)
numba/cuda/tests/cudapy/test_debuginfo.py:        with override_config('CUDA_DEBUGINFO_DEFAULT', 1):
numba/cuda/tests/cudapy/test_debuginfo.py:            @cuda.jit(opt=False)
numba/cuda/tests/cudapy/test_debuginfo.py:            @cuda.jit(debug=False)
numba/cuda/tests/cudapy/test_debuginfo.py:        @cuda.jit((types.int32[::1],), debug=True, opt=False)
numba/cuda/tests/cudapy/test_debuginfo.py:        @cuda.jit(sig, debug=True, opt=0)
numba/cuda/tests/cudapy/test_debuginfo.py:                   if 'define void @"_ZN6cudapy' in line]
numba/cuda/tests/cudapy/test_debuginfo.py:        @cuda.jit((types.int32[:], types.int32[:]), debug=True, opt=False)
numba/cuda/tests/cudapy/test_debuginfo.py:        @cuda.jit(device=True, debug=True, opt=0)
numba/cuda/tests/cudapy/test_debuginfo.py:            return cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
numba/cuda/tests/cudapy/test_debuginfo.py:        @cuda.jit((types.int32[:],), debug=True, opt=0)
numba/cuda/tests/cudapy/test_debuginfo.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_debuginfo.py:        @cuda.jit(device=True, debug=f2_debug, opt=False)
numba/cuda/tests/cudapy/test_debuginfo.py:        @cuda.jit(device=True, debug=f1_debug, opt=False)
numba/cuda/tests/cudapy/test_debuginfo.py:        @cuda.jit((types.int32, types.int32), debug=kernel_debug, opt=False)
numba/cuda/tests/cudapy/test_debuginfo.py:        @cuda.jit(device=True, debug=f2_debug, opt=False)
numba/cuda/tests/cudapy/test_debuginfo.py:        @cuda.jit(device=True, debug=f1_debug, opt=False)
numba/cuda/tests/cudapy/test_debuginfo.py:        @cuda.jit(debug=kernel_debug, opt=False)
numba/cuda/tests/cudapy/test_debuginfo.py:            @cuda.jit(device=True, debug=leaf_debug, opt=False)
numba/cuda/tests/cudapy/test_debuginfo.py:            @cuda.jit(device=True)
numba/cuda/tests/cudapy/test_debuginfo.py:            @cuda.jit(device=True)
numba/cuda/tests/cudapy/test_debuginfo.py:            @cuda.jit(debug=kernel_debug, opt=False)
numba/cuda/tests/cudapy/test_overload.py:from numba import cuda, njit, types
numba/cuda/tests/cudapy/test_overload.py:from numba.cuda.testing import CUDATestCase, skip_on_cudasim, unittest
numba/cuda/tests/cudapy/test_overload.py:def cuda_func_1():
numba/cuda/tests/cudapy/test_overload.py:def cuda_func_2():
numba/cuda/tests/cudapy/test_overload.py:def generic_calls_cuda():
numba/cuda/tests/cudapy/test_overload.py:def cuda_calls_generic():
numba/cuda/tests/cudapy/test_overload.py:def cuda_calls_cuda():
numba/cuda/tests/cudapy/test_overload.py:def cuda_calls_target_overloaded():
numba/cuda/tests/cudapy/test_overload.py:CUDA_FUNCTION_1 = 3
numba/cuda/tests/cudapy/test_overload.py:CUDA_FUNCTION_2 = 7
numba/cuda/tests/cudapy/test_overload.py:GENERIC_CALLS_CUDA = 13
numba/cuda/tests/cudapy/test_overload.py:CUDA_CALLS_GENERIC = 17
numba/cuda/tests/cudapy/test_overload.py:CUDA_CALLS_CUDA = 19
numba/cuda/tests/cudapy/test_overload.py:CUDA_TARGET_OL = 29
numba/cuda/tests/cudapy/test_overload.py:CUDA_CALLS_TARGET_OL = 37
numba/cuda/tests/cudapy/test_overload.py:CUDA_TARGET_OL_CALLS_TARGET_OL = 43
numba/cuda/tests/cudapy/test_overload.py:@overload(cuda_func_1, target='cuda')
numba/cuda/tests/cudapy/test_overload.py:def ol_cuda_func_1(x):
numba/cuda/tests/cudapy/test_overload.py:        x[0] *= CUDA_FUNCTION_1
numba/cuda/tests/cudapy/test_overload.py:@overload(cuda_func_2, target='cuda')
numba/cuda/tests/cudapy/test_overload.py:def ol_cuda_func(x):
numba/cuda/tests/cudapy/test_overload.py:        x[0] *= CUDA_FUNCTION_2
numba/cuda/tests/cudapy/test_overload.py:@overload(generic_calls_cuda, target='generic')
numba/cuda/tests/cudapy/test_overload.py:def ol_generic_calls_cuda(x):
numba/cuda/tests/cudapy/test_overload.py:        x[0] *= GENERIC_CALLS_CUDA
numba/cuda/tests/cudapy/test_overload.py:        cuda_func_1(x)
numba/cuda/tests/cudapy/test_overload.py:@overload(cuda_calls_generic, target='cuda')
numba/cuda/tests/cudapy/test_overload.py:def ol_cuda_calls_generic(x):
numba/cuda/tests/cudapy/test_overload.py:        x[0] *= CUDA_CALLS_GENERIC
numba/cuda/tests/cudapy/test_overload.py:@overload(cuda_calls_cuda, target='cuda')
numba/cuda/tests/cudapy/test_overload.py:def ol_cuda_calls_cuda(x):
numba/cuda/tests/cudapy/test_overload.py:        x[0] *= CUDA_CALLS_CUDA
numba/cuda/tests/cudapy/test_overload.py:        cuda_func_1(x)
numba/cuda/tests/cudapy/test_overload.py:@overload(target_overloaded, target='cuda')
numba/cuda/tests/cudapy/test_overload.py:def ol_target_overloaded_cuda(x):
numba/cuda/tests/cudapy/test_overload.py:        x[0] *= CUDA_TARGET_OL
numba/cuda/tests/cudapy/test_overload.py:@overload(cuda_calls_target_overloaded, target='cuda')
numba/cuda/tests/cudapy/test_overload.py:def ol_cuda_calls_target_overloaded(x):
numba/cuda/tests/cudapy/test_overload.py:        x[0] *= CUDA_CALLS_TARGET_OL
numba/cuda/tests/cudapy/test_overload.py:@overload(target_overloaded_calls_target_overloaded, target='cuda')
numba/cuda/tests/cudapy/test_overload.py:def ol_generic_calls_target_overloaded_cuda(x):
numba/cuda/tests/cudapy/test_overload.py:        x[0] *= CUDA_TARGET_OL_CALLS_TARGET_OL
numba/cuda/tests/cudapy/test_overload.py:@skip_on_cudasim('Overloading not supported in cudasim')
numba/cuda/tests/cudapy/test_overload.py:class TestOverload(CUDATestCase):
numba/cuda/tests/cudapy/test_overload.py:        cuda.jit(kernel)[1, 1](x)
numba/cuda/tests/cudapy/test_overload.py:    def test_cuda(self):
numba/cuda/tests/cudapy/test_overload.py:            cuda_func_1(x)
numba/cuda/tests/cudapy/test_overload.py:        expected = CUDA_FUNCTION_1
numba/cuda/tests/cudapy/test_overload.py:    def test_generic_and_cuda(self):
numba/cuda/tests/cudapy/test_overload.py:            cuda_func_1(x)
numba/cuda/tests/cudapy/test_overload.py:        expected = GENERIC_FUNCTION_1 * CUDA_FUNCTION_1
numba/cuda/tests/cudapy/test_overload.py:    def test_call_two_cuda_calls(self):
numba/cuda/tests/cudapy/test_overload.py:            cuda_func_1(x)
numba/cuda/tests/cudapy/test_overload.py:            cuda_func_2(x)
numba/cuda/tests/cudapy/test_overload.py:        expected = CUDA_FUNCTION_1 * CUDA_FUNCTION_2
numba/cuda/tests/cudapy/test_overload.py:    def test_generic_calls_cuda(self):
numba/cuda/tests/cudapy/test_overload.py:            generic_calls_cuda(x)
numba/cuda/tests/cudapy/test_overload.py:        expected = GENERIC_CALLS_CUDA * CUDA_FUNCTION_1
numba/cuda/tests/cudapy/test_overload.py:    def test_cuda_calls_generic(self):
numba/cuda/tests/cudapy/test_overload.py:            cuda_calls_generic(x)
numba/cuda/tests/cudapy/test_overload.py:        expected = CUDA_CALLS_GENERIC * GENERIC_FUNCTION_1
numba/cuda/tests/cudapy/test_overload.py:    def test_cuda_calls_cuda(self):
numba/cuda/tests/cudapy/test_overload.py:            cuda_calls_cuda(x)
numba/cuda/tests/cudapy/test_overload.py:        expected = CUDA_CALLS_CUDA * CUDA_FUNCTION_1
numba/cuda/tests/cudapy/test_overload.py:        expected = CUDA_TARGET_OL
numba/cuda/tests/cudapy/test_overload.py:        expected = GENERIC_CALLS_TARGET_OL * CUDA_TARGET_OL
numba/cuda/tests/cudapy/test_overload.py:    def test_cuda_calls_target_overloaded(self):
numba/cuda/tests/cudapy/test_overload.py:            cuda_calls_target_overloaded(x)
numba/cuda/tests/cudapy/test_overload.py:        expected = CUDA_CALLS_TARGET_OL * CUDA_TARGET_OL
numba/cuda/tests/cudapy/test_overload.py:        # Check the CUDA overloads are used on CUDA
numba/cuda/tests/cudapy/test_overload.py:        expected = CUDA_TARGET_OL_CALLS_TARGET_OL * CUDA_TARGET_OL
numba/cuda/tests/cudapy/test_overload.py:        @overload_attribute(MyDummyType, 'cuda_only', target='cuda')
numba/cuda/tests/cudapy/test_overload.py:        def ov_dummy_cuda_attr(obj):
numba/cuda/tests/cudapy/test_overload.py:        # Ensure that we cannot use the CUDA target-specific attribute on the
numba/cuda/tests/cudapy/test_overload.py:                                    "Unknown attribute 'cuda_only'"):
numba/cuda/tests/cudapy/test_overload.py:                return x.cuda_only
numba/cuda/tests/cudapy/test_overload.py:        # Ensure that the CUDA target-specific attribute is usable and works
numba/cuda/tests/cudapy/test_overload.py:        # correctly when the target is CUDA - note eager compilation via
numba/cuda/tests/cudapy/test_overload.py:        @cuda.jit(types.void(types.int64[::1], mydummy_type))
numba/cuda/tests/cudapy/test_overload.py:        def cuda_target_attr_use(res, dummy):
numba/cuda/tests/cudapy/test_overload.py:            res[0] = dummy.cuda_only
numba/cuda/tests/cudapy/test_freevar.py:from numba import cuda
numba/cuda/tests/cudapy/test_freevar.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_freevar.py:class TestFreeVar(CUDATestCase):
numba/cuda/tests/cudapy/test_freevar.py:        @cuda.jit("(float32[::1], intp)")
numba/cuda/tests/cudapy/test_freevar.py:            sdata = cuda.shared.array(size,   # size is freevar
numba/cuda/tests/cudapy/test_sm_creation.py:from numba import cuda, float32, int32, void
numba/cuda/tests/cudapy/test_sm_creation.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_sm_creation.py:from numba.cuda.testing import skip_on_cudasim
numba/cuda/tests/cudapy/test_sm_creation.py:    sa = cuda.shared.array(shape=GLOBAL_CONSTANT, dtype=float32)
numba/cuda/tests/cudapy/test_sm_creation.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_sm_creation.py:    sa = cuda.shared.array(shape=(GLOBAL_CONSTANT, GLOBAL_CONSTANT_2),
numba/cuda/tests/cudapy/test_sm_creation.py:    i, j = cuda.grid(2)
numba/cuda/tests/cudapy/test_sm_creation.py:    sa = cuda.shared.array(shape=[GLOBAL_CONSTANT, GLOBAL_CONSTANT_2],
numba/cuda/tests/cudapy/test_sm_creation.py:    i, j = cuda.grid(2)
numba/cuda/tests/cudapy/test_sm_creation.py:    sa = cuda.shared.array(shape=GLOBAL_CONSTANT_TUPLE, dtype=float32)
numba/cuda/tests/cudapy/test_sm_creation.py:    i, j = cuda.grid(2)
numba/cuda/tests/cudapy/test_sm_creation.py:    sa = cuda.shared.array(shape=A[0], dtype=float32)
numba/cuda/tests/cudapy/test_sm_creation.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_sm_creation.py:    sa = cuda.shared.array(shape=(1, A[0]), dtype=float32)
numba/cuda/tests/cudapy/test_sm_creation.py:    i, j = cuda.grid(2)
numba/cuda/tests/cudapy/test_sm_creation.py:    sa = cuda.shared.array(shape=(1, A[0]), dtype=float32)
numba/cuda/tests/cudapy/test_sm_creation.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_sm_creation.py:class TestSharedMemoryCreation(CUDATestCase):
numba/cuda/tests/cudapy/test_sm_creation.py:        udt = cuda.jit((float32[:],))(udt_global_constants)
numba/cuda/tests/cudapy/test_sm_creation.py:        udt = cuda.jit((float32[:, :],))(udt_global_build_tuple)
numba/cuda/tests/cudapy/test_sm_creation.py:    @skip_on_cudasim('Simulator does not prohibit lists for shared array shape')
numba/cuda/tests/cudapy/test_sm_creation.py:            cuda.jit((float32[:, :],))(udt_global_build_list)
numba/cuda/tests/cudapy/test_sm_creation.py:        udt = cuda.jit((float32[:, :],))(udt_global_constant_tuple)
numba/cuda/tests/cudapy/test_sm_creation.py:    @skip_on_cudasim("Can't check for constants in simulator")
numba/cuda/tests/cudapy/test_sm_creation.py:            cuda.jit((float32[:],))(udt_invalid_1)
numba/cuda/tests/cudapy/test_sm_creation.py:    @skip_on_cudasim("Can't check for constants in simulator")
numba/cuda/tests/cudapy/test_sm_creation.py:            cuda.jit((float32[:, :],))(udt_invalid_2)
numba/cuda/tests/cudapy/test_sm_creation.py:    @skip_on_cudasim("Can't check for constants in simulator")
numba/cuda/tests/cudapy/test_sm_creation.py:            cuda.jit((int32[:],))(udt_invalid_1)
numba/cuda/tests/cudapy/test_sm_creation.py:    @skip_on_cudasim("Can't check for constants in simulator")
numba/cuda/tests/cudapy/test_sm_creation.py:            cuda.jit((int32[:],))(udt_invalid_3)
numba/cuda/tests/cudapy/test_sm_creation.py:        # Find the typing of the dtype argument to cuda.shared.array
numba/cuda/tests/cudapy/test_sm_creation.py:    @skip_on_cudasim("Can't check typing in simulator")
numba/cuda/tests/cudapy/test_sm_creation.py:        @cuda.jit(void(int32[::1]))
numba/cuda/tests/cudapy/test_sm_creation.py:            s = cuda.shared.array(10, dtype=int32)
numba/cuda/tests/cudapy/test_sm_creation.py:    @skip_on_cudasim("Can't check typing in simulator")
numba/cuda/tests/cudapy/test_sm_creation.py:        @cuda.jit(void(int32[::1]))
numba/cuda/tests/cudapy/test_sm_creation.py:            s = cuda.shared.array(10, dtype=np.int32)
numba/cuda/tests/cudapy/test_sm_creation.py:    @skip_on_cudasim("Can't check typing in simulator")
numba/cuda/tests/cudapy/test_sm_creation.py:        @cuda.jit(void(int32[::1]))
numba/cuda/tests/cudapy/test_sm_creation.py:            s = cuda.shared.array(10, dtype='int32')
numba/cuda/tests/cudapy/test_sm_creation.py:    @skip_on_cudasim("Can't check typing in simulator")
numba/cuda/tests/cudapy/test_sm_creation.py:            @cuda.jit(void(int32[::1]))
numba/cuda/tests/cudapy/test_sm_creation.py:                s = cuda.shared.array(10, dtype='int33')
numba/cuda/tests/cudapy/test_sm_creation.py:    @skip_on_cudasim("Can't check typing in simulator")
numba/cuda/tests/cudapy/test_sm_creation.py:        @cuda.jit(void(test_struct_model_type[::1]))
numba/cuda/tests/cudapy/test_sm_creation.py:            s = cuda.shared.array(10, dtype=test_struct_model_type)
numba/cuda/tests/cudapy/test_intrinsics.py:from numba import cuda, int64
numba/cuda/tests/cudapy/test_intrinsics.py:from numba.cuda import compile_ptx
numba/cuda/tests/cudapy/test_intrinsics.py:from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_intrinsics.py:    j = cuda.threadIdx.y
numba/cuda/tests/cudapy/test_intrinsics.py:    k = cuda.threadIdx.z
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:    i, j = cuda.grid(2)
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:    x = cuda.gridsize(1)
numba/cuda/tests/cudapy/test_intrinsics.py:    i, j = cuda.grid(2)
numba/cuda/tests/cudapy/test_intrinsics.py:    x, y = cuda.gridsize(2)
numba/cuda/tests/cudapy/test_intrinsics.py:    startX, startY = cuda.grid(2)
numba/cuda/tests/cudapy/test_intrinsics.py:    gridX = cuda.gridDim.x * cuda.blockDim.x
numba/cuda/tests/cudapy/test_intrinsics.py:    gridY = cuda.gridDim.y * cuda.blockDim.y
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.popc(c)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fma(a, b, c)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hadd(a[0], b[0])
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hadd(a, b)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hfma(a[0], b[0], c[0])
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hfma(a, b, c)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hsub(a[0], b[0])
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hsub(a, b)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hmul(a[0], b[0])
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hmul(a, b)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hdiv(a, b)
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:        ary[i] = cuda.fp16.hdiv(a, b)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hneg(a[0])
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hneg(a)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.habs(a[0])
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.habs(a)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.heq(a, b)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hne(a, b)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hge(a, b)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hgt(a, b)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hle(a, b)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hlt(a, b)
numba/cuda/tests/cudapy/test_intrinsics.py:@cuda.jit(device=True)
numba/cuda/tests/cudapy/test_intrinsics.py:    return cuda.fp16.hlt(x, y)
numba/cuda/tests/cudapy/test_intrinsics.py:@cuda.jit(device=True)
numba/cuda/tests/cudapy/test_intrinsics.py:    return cuda.fp16.hlt(x, y)
numba/cuda/tests/cudapy/test_intrinsics.py:    r[0] = hlt_func_1(a, b) and cuda.fp16.hlt(b, c)
numba/cuda/tests/cudapy/test_intrinsics.py:    r[0] = hlt_func_1(a, b) and cuda.fp16.hge(c, b)
numba/cuda/tests/cudapy/test_intrinsics.py:    r[0] = cuda.fp16.hlt(a, b) and cuda.fp16.hlt(b, c)
numba/cuda/tests/cudapy/test_intrinsics.py:    r[0] = cuda.fp16.hlt(a, b) and cuda.fp16.hge(c, b)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hmax(a, b)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.fp16.hmin(a, b)
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:        r[i] = cuda.fp16.hsin(x[i])
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:        r[i] = cuda.fp16.hcos(x[i])
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:        r[i] = cuda.fp16.hlog(x[i])
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:        r[i] = cuda.fp16.hlog2(x[i])
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:        r[i] = cuda.fp16.hlog10(x[i])
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:        r[i] = cuda.fp16.hexp(x[i])
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:        r[i] = cuda.fp16.hexp2(x[i])
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:        r[i] = cuda.fp16.hsqrt(x[i])
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:        r[i] = cuda.fp16.hrsqrt(x[i])
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:        r[i] = cuda.fp16.hceil(x[i])
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:        r[i] = cuda.fp16.hfloor(x[i])
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:        r[i] = cuda.fp16.hrcp(x[i])
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:        r[i] = cuda.fp16.htrunc(x[i])
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:        r[i] = cuda.fp16.hrint(x[i])
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.cbrt(a)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.brev(c)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.clz(c)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.ffs(c)
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:    inner = cuda.selp(b % 2 == 0, c[i], 13)
numba/cuda/tests/cudapy/test_intrinsics.py:    a[i] = cuda.selp(a[i] > 4, inner, 3)
numba/cuda/tests/cudapy/test_intrinsics.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[i] = cuda.laneid
numba/cuda/tests/cudapy/test_intrinsics.py:    ary[0] = cuda.warpsize
numba/cuda/tests/cudapy/test_intrinsics.py:    cuda.grid(x)
numba/cuda/tests/cudapy/test_intrinsics.py:    cuda.gridsize(x)
numba/cuda/tests/cudapy/test_intrinsics.py:class TestCudaIntrinsic(CUDATestCase):
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:])")(simple_threadidx)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:])")(fill_threadidx)
numba/cuda/tests/cudapy/test_intrinsics.py:            compiled = cuda.jit("void(int32[:,:,::1])")(fill3d_threadidx)
numba/cuda/tests/cudapy/test_intrinsics.py:            compiled = cuda.jit("void(int32[::1,:,:])")(fill3d_threadidx)
numba/cuda/tests/cudapy/test_intrinsics.py:    @skip_on_cudasim('Cudasim does not check types')
numba/cuda/tests/cudapy/test_intrinsics.py:            cuda.jit('void(int32)')(nonliteral_grid)
numba/cuda/tests/cudapy/test_intrinsics.py:    @skip_on_cudasim('Cudasim does not check types')
numba/cuda/tests/cudapy/test_intrinsics.py:            cuda.jit('void(int32)')(nonliteral_gridsize)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[::1])")(simple_grid1d)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:,::1])")(simple_grid2d)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[::1])")(simple_gridsize1d)
numba/cuda/tests/cudapy/test_intrinsics.py:    @skip_on_cudasim('Requires too many threads')
numba/cuda/tests/cudapy/test_intrinsics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_intrinsics.py:            i1 = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:            i2 = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
numba/cuda/tests/cudapy/test_intrinsics.py:            gs1 = cuda.gridsize(1)
numba/cuda/tests/cudapy/test_intrinsics.py:            gs2 = cuda.blockDim.x * cuda.gridDim.x
numba/cuda/tests/cudapy/test_intrinsics.py:    @skip_on_cudasim('Tests PTX emission')
numba/cuda/tests/cudapy/test_intrinsics.py:        cu_branching_with_ifs = cuda.jit(sig)(branching_with_ifs)
numba/cuda/tests/cudapy/test_intrinsics.py:        cu_branching_with_selps = cuda.jit(sig)(branching_with_selps)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[::1])")(simple_gridsize2d)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:,::1])")(intrinsic_forloop_step)
numba/cuda/tests/cudapy/test_intrinsics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_intrinsics.py:            x, y, z = cuda.grid(3)
numba/cuda/tests/cudapy/test_intrinsics.py:            a, b, c = cuda.gridsize(3)
numba/cuda/tests/cudapy/test_intrinsics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_intrinsics.py:            x, y, z = cuda.grid(3)
numba/cuda/tests/cudapy/test_intrinsics.py:            a, b, c = cuda.gridsize(3)
numba/cuda/tests/cudapy/test_intrinsics.py:                x == cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x and
numba/cuda/tests/cudapy/test_intrinsics.py:                y == cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y and
numba/cuda/tests/cudapy/test_intrinsics.py:                z == cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
numba/cuda/tests/cudapy/test_intrinsics.py:            gridsize_is_right = (a == cuda.blockDim.x * cuda.gridDim.x and
numba/cuda/tests/cudapy/test_intrinsics.py:                                 b == cuda.blockDim.y * cuda.gridDim.y and
numba/cuda/tests/cudapy/test_intrinsics.py:                                 c == cuda.blockDim.z * cuda.gridDim.z)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:], uint32)")(simple_popc)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:], uint64)")(simple_popc)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(f4[:], f4, f4, f4)")(simple_fma)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(f8[:], f8, f8, f8)")(simple_fma)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(f2[:], f2[:], f2[:])")(simple_hadd)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(f2[:], f2, f2)")(simple_hadd_scalar)
numba/cuda/tests/cudapy/test_intrinsics.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(f2[:], f2[:], f2[:], f2[:])")(simple_hfma)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(f2[:], f2, f2, f2)")(simple_hfma_scalar)
numba/cuda/tests/cudapy/test_intrinsics.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(f2[:], f2[:], f2[:])")(simple_hsub)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(f2[:], f2, f2)")(simple_hsub_scalar)
numba/cuda/tests/cudapy/test_intrinsics.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit()(simple_hmul)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(f2[:], f2, f2)")(simple_hmul_scalar)
numba/cuda/tests/cudapy/test_intrinsics.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(f2[:], f2, f2)")(simple_hdiv_scalar)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(f2[:], f2[:], f2[:])")(simple_hdiv_kernel)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(f2[:], f2[:])")(simple_hneg)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(f2[:], f2)")(simple_hneg_scalar)
numba/cuda/tests/cudapy/test_intrinsics.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit()(simple_habs)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(f2[:], f2)")(simple_habs_scalar)
numba/cuda/tests/cudapy/test_intrinsics.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_intrinsics.py:                kernel = cuda.jit("void(f2[:], f2[:])")(kernel)
numba/cuda/tests/cudapy/test_intrinsics.py:                kernel = cuda.jit("void(f2[:], f2[:])")(kernel)
numba/cuda/tests/cudapy/test_intrinsics.py:        @cuda.jit()
numba/cuda/tests/cudapy/test_intrinsics.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_intrinsics.py:                r[i] = cuda.fp16.hexp10(x[i])
numba/cuda/tests/cudapy/test_intrinsics.py:                kernel = cuda.jit("void(b1[:], f2, f2)")(fn)
numba/cuda/tests/cudapy/test_intrinsics.py:                compiled = cuda.jit("void(b1[:], f2, f2, f2)")(fn)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(f2[:], f2, f2)")(simple_hmax_scalar)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(f2[:], f2, f2)")(simple_hmin_scalar)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(float32[:], float32)")(simple_cbrt)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(float64[:], float64)")(simple_cbrt)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(uint32[:], uint32)")(simple_brev)
numba/cuda/tests/cudapy/test_intrinsics.py:    @skip_on_cudasim('only get given a Python "int", assumes 32 bits')
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(uint64[:], uint64)")(simple_brev)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:], int32)")(simple_clz)
numba/cuda/tests/cudapy/test_intrinsics.py:        Although the CUDA Math API
numba/cuda/tests/cudapy/test_intrinsics.py:        (http://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html)
numba/cuda/tests/cudapy/test_intrinsics.py:        http://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#bit-manipulations-intrinics
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:], uint32)")(simple_clz)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:], int32)")(simple_clz)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:], int32)")(simple_clz)
numba/cuda/tests/cudapy/test_intrinsics.py:        self.assertEqual(ary[0], 32, "CUDA semantics")
numba/cuda/tests/cudapy/test_intrinsics.py:    @skip_on_cudasim('only get given a Python "int", assumes 32 bits')
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:], int64)")(simple_clz)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:], int32)")(simple_ffs)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:], uint32)")(simple_ffs)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:], int32)")(simple_ffs)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:], int32)")(simple_ffs)
numba/cuda/tests/cudapy/test_intrinsics.py:    @skip_on_cudasim('only get given a Python "int", assumes 32 bits')
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:], int64)")(simple_ffs)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:])")(simple_laneid)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int32[:])")(simple_warpsize)
numba/cuda/tests/cudapy/test_intrinsics.py:        self.assertEqual(ary[0], 32, "CUDA semantics")
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int64[:], float32)")(simple_round)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(int64[:], float64)")(simple_round)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(float32[:], float32, int32)")(simple_round_to)
numba/cuda/tests/cudapy/test_intrinsics.py:    # CPython on most platforms uses rounding based on dtoa.c, whereas the CUDA
numba/cuda/tests/cudapy/test_intrinsics.py:    # slightly different behavior at the edges of the domain. Since the CUDA
numba/cuda/tests/cudapy/test_intrinsics.py:    @skip_on_cudasim('Overflow behavior differs on CPython')
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(float32[:], float32, int32)")(simple_round_to)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(float32[:], float32, int32)")(simple_round_to)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(float64[:], float64, int32)")(simple_round_to)
numba/cuda/tests/cudapy/test_intrinsics.py:    # Skipped on cudasim for the same reasons as test_round_to_f4 above.
numba/cuda/tests/cudapy/test_intrinsics.py:    @skip_on_cudasim('Overflow behavior differs on CPython')
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(float64[:], float64, int32)")(simple_round_to)
numba/cuda/tests/cudapy/test_intrinsics.py:        compiled = cuda.jit("void(float64[:], float64, int32)")(simple_round_to)
numba/cuda/tests/cudapy/test_record_dtype.py:from numba import cuda
numba/cuda/tests/cudapy/test_record_dtype.py:from numba.cuda.testing import skip_on_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_record_dtype.py:class TestRecordDtype(CUDATestCase):
numba/cuda/tests/cudapy/test_record_dtype.py:        return cuda.jit()(pyfunc)
numba/cuda/tests/cudapy/test_record_dtype.py:@skip_on_cudasim('Structured array attr access not supported in simulator')
numba/cuda/tests/cudapy/test_record_dtype.py:class TestNestedArrays(CUDATestCase):
numba/cuda/tests/cudapy/test_record_dtype.py:    # as the CUDA test implementations need to be launched (and in some cases
numba/cuda/tests/cudapy/test_record_dtype.py:        # Create a host-callable function for testing CUDA device functions
numba/cuda/tests/cudapy/test_record_dtype.py:        inner = cuda.jit(device=True)(pyfunc)
numba/cuda/tests/cudapy/test_record_dtype.py:        @cuda.jit
numba/cuda/tests/cudapy/test_record_dtype.py:        cfunc = cuda.jit(pyfunc)
numba/cuda/tests/cudapy/test_record_dtype.py:    @skip_on_cudasim('Structured array attr access not supported in simulator')
numba/cuda/tests/cudapy/test_record_dtype.py:        kernel = cuda.jit(pyfunc)
numba/cuda/tests/cudapy/test_record_dtype.py:        kernel = cuda.jit(pyfunc)
numba/cuda/tests/cudapy/test_record_dtype.py:        kernel = cuda.jit(pyfunc)
numba/cuda/tests/cudapy/test_record_dtype.py:        @cuda.jit
numba/cuda/tests/cudapy/test_record_dtype.py:    # all xfailed because CUDA cannot handle returning arrays from device
numba/cuda/tests/cudapy/test_record_dtype.py:    @skip_on_cudasim('Will unexpectedly pass on cudasim')
numba/cuda/tests/cudapy/test_record_dtype.py:            kernel = cuda.jit(pyfunc)
numba/cuda/tests/cudapy/test_vectorize_device.py:from numba import cuda, float32
numba/cuda/tests/cudapy/test_vectorize_device.py:from numba.cuda.testing import skip_on_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_vectorize_device.py:@skip_on_cudasim('ufunc API unsupported in the simulator')
numba/cuda/tests/cudapy/test_vectorize_device.py:class TestCudaVectorizeDeviceCall(CUDATestCase):
numba/cuda/tests/cudapy/test_vectorize_device.py:    def test_cuda_vectorize_device_call(self):
numba/cuda/tests/cudapy/test_vectorize_device.py:        @cuda.jit(float32(float32, float32, float32), device=True)
numba/cuda/tests/cudapy/test_vectorize_device.py:        ufunc = vectorize([float32(float32, float32, float32)], target='cuda')(
numba/cuda/tests/cudapy/test_constmem.py:from numba import cuda, complex64, int32, float64
numba/cuda/tests/cudapy/test_constmem.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_constmem.py:from numba.core.config import ENABLE_CUDASIM
numba/cuda/tests/cudapy/test_constmem.py:    C = cuda.const.array_like(CONST_EMPTY)
numba/cuda/tests/cudapy/test_constmem.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_constmem.py:    C = cuda.const.array_like(CONST1D)
numba/cuda/tests/cudapy/test_constmem.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_constmem.py:    C = cuda.const.array_like(CONST2D)
numba/cuda/tests/cudapy/test_constmem.py:    i, j = cuda.grid(2)
numba/cuda/tests/cudapy/test_constmem.py:    C = cuda.const.array_like(CONST3D)
numba/cuda/tests/cudapy/test_constmem.py:    i = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_constmem.py:    j = cuda.threadIdx.y
numba/cuda/tests/cudapy/test_constmem.py:    k = cuda.threadIdx.z
numba/cuda/tests/cudapy/test_constmem.py:    C = cuda.const.array_like(CONST_RECORD_EMPTY)
numba/cuda/tests/cudapy/test_constmem.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_constmem.py:    C = cuda.const.array_like(CONST_RECORD)
numba/cuda/tests/cudapy/test_constmem.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_constmem.py:    Z = cuda.const.array_like(CONST_RECORD_ALIGN)
numba/cuda/tests/cudapy/test_constmem.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_constmem.py:    a = cuda.const.array_like(CONST3BYTES)
numba/cuda/tests/cudapy/test_constmem.py:    b = cuda.const.array_like(CONST1D)
numba/cuda/tests/cudapy/test_constmem.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_constmem.py:class TestCudaConstantMemory(CUDATestCase):
numba/cuda/tests/cudapy/test_constmem.py:        jcuconst = cuda.jit(sig)(cuconst)
numba/cuda/tests/cudapy/test_constmem.py:        if not ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_constmem.py:        jcuconstEmpty = cuda.jit('void(int64[:])')(cuconstEmpty)
numba/cuda/tests/cudapy/test_constmem.py:        jcuconstAlign = cuda.jit('void(float64[:])')(cuconstAlign)
numba/cuda/tests/cudapy/test_constmem.py:        jcuconst2d = cuda.jit(sig)(cuconst2d)
numba/cuda/tests/cudapy/test_constmem.py:        if not ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_constmem.py:        jcuconst3d = cuda.jit(sig)(cuconst3d)
numba/cuda/tests/cudapy/test_constmem.py:        if not ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_constmem.py:        jcuconstRecEmpty = cuda.jit('void(int64[:])')(cuconstRecEmpty)
numba/cuda/tests/cudapy/test_constmem.py:        jcuconst = cuda.jit(cuconstRec).specialize(A, B)
numba/cuda/tests/cudapy/test_constmem.py:        jcuconst = cuda.jit(cuconstRecAlign).specialize(A, B, C, D, E)
numba/cuda/tests/cudapy/test_ipc.py:from numba import cuda
numba/cuda/tests/cudapy/test_ipc.py:from numba.cuda.cudadrv import driver
numba/cuda/tests/cudapy/test_ipc.py:from numba.cuda.testing import (skip_on_arm, skip_on_cudasim,
numba/cuda/tests/cudapy/test_ipc.py:                                skip_under_cuda_memcheck,
numba/cuda/tests/cudapy/test_ipc.py:        with cuda.open_ipc_array(handle, shape=size // dtype.itemsize,
numba/cuda/tests/cudapy/test_ipc.py:        darr = handle.open_array(cuda.current_context(),
numba/cuda/tests/cudapy/test_ipc.py:@skip_under_cuda_memcheck('Hangs cuda-memcheck')
numba/cuda/tests/cudapy/test_ipc.py:@skip_on_cudasim('Ipc not available in CUDASIM')
numba/cuda/tests/cudapy/test_ipc.py:@skip_on_arm('CUDA IPC not supported on ARM in Numba')
numba/cuda/tests/cudapy/test_ipc.py:        devarr = cuda.to_device(arr)
numba/cuda/tests/cudapy/test_ipc.py:        ctx = cuda.current_context()
numba/cuda/tests/cudapy/test_ipc.py:        ipch = ctx.get_ipc_handle(devarr.gpu_data)
numba/cuda/tests/cudapy/test_ipc.py:        # the CUDA Array Interface
numba/cuda/tests/cudapy/test_ipc.py:        devarr = cuda.to_device(arr)
numba/cuda/tests/cudapy/test_ipc.py:            devarr = cuda.as_cuda_array(ForeignArray(devarr))
numba/cuda/tests/cudapy/test_ipc.py:        ctx = cuda.current_context()
numba/cuda/tests/cudapy/test_ipc.py:        ipch = ctx.get_ipc_handle(devarr.gpu_data)
numba/cuda/tests/cudapy/test_ipc.py:        devarr = cuda.to_device(arr)
numba/cuda/tests/cudapy/test_ipc.py:            devarr = cuda.as_cuda_array(ForeignArray(devarr))
numba/cuda/tests/cudapy/test_ipc.py:        with cuda.gpus[device_num]:
numba/cuda/tests/cudapy/test_ipc.py:            this_ctx = cuda.devices.get_context()
numba/cuda/tests/cudapy/test_ipc.py:            cuda.driver.device_to_host(
numba/cuda/tests/cudapy/test_ipc.py:        with cuda.gpus[device_num]:
numba/cuda/tests/cudapy/test_ipc.py:@skip_under_cuda_memcheck('Hangs cuda-memcheck')
numba/cuda/tests/cudapy/test_ipc.py:@skip_on_cudasim('Ipc not available in CUDASIM')
numba/cuda/tests/cudapy/test_ipc.py:@skip_on_arm('CUDA IPC not supported on ARM in Numba')
numba/cuda/tests/cudapy/test_ipc.py:        devarr = cuda.to_device(arr)
numba/cuda/tests/cudapy/test_ipc.py:        ctx = cuda.current_context()
numba/cuda/tests/cudapy/test_ipc.py:        ipch = ctx.get_ipc_handle(devarr.gpu_data)
numba/cuda/tests/cudapy/test_ipc.py:        # Test on every CUDA devices
numba/cuda/tests/cudapy/test_ipc.py:        for device_num in range(len(cuda.gpus)):
numba/cuda/tests/cudapy/test_ipc.py:        for device_num in range(len(cuda.gpus)):
numba/cuda/tests/cudapy/test_ipc.py:            devarr = cuda.to_device(arr)
numba/cuda/tests/cudapy/test_ipc.py:@skip_on_cudasim('Ipc not available in CUDASIM')
numba/cuda/tests/cudapy/test_ipc.py:        devarr = cuda.to_device(arr)
numba/cuda/tests/cudapy/test_ipc.py:        self.assertIn('OS does not support CUDA IPC', errmsg)
numba/cuda/tests/cudapy/test_multithreads.py:from numba import cuda
numba/cuda/tests/cudapy/test_multithreads.py:from numba.cuda.testing import (skip_on_cudasim, skip_under_cuda_memcheck,
numba/cuda/tests/cudapy/test_multithreads.py:                                CUDATestCase)
numba/cuda/tests/cudapy/test_multithreads.py:    @cuda.jit
numba/cuda/tests/cudapy/test_multithreads.py:    arrays = [cuda.to_device(np.arange(10)) for i in range(10)]
numba/cuda/tests/cudapy/test_multithreads.py:@skip_under_cuda_memcheck('Hangs cuda-memcheck')
numba/cuda/tests/cudapy/test_multithreads.py:@skip_on_cudasim('disabled for cudasim')
numba/cuda/tests/cudapy/test_multithreads.py:class TestMultiThreadCompiling(CUDATestCase):
numba/cuda/tests/cudapy/test_multithreads.py:        # force CUDA context init
numba/cuda/tests/cudapy/test_multithreads.py:        cuda.get_current_device()
numba/cuda/tests/cudapy/test_multithreads.py:        # use "spawn" to avoid inheriting the CUDA context
numba/cuda/tests/cudapy/test_multithreads.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudapy/test_multithreads.py:        common = cuda.to_device(arr)
numba/cuda/tests/cudapy/test_multithreads.py:        darr = cuda.to_device(np.zeros(common.shape, dtype=common.dtype))
numba/cuda/tests/cudapy/test_complex_kernel.py:from numba import cuda
numba/cuda/tests/cudapy/test_complex_kernel.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_complex_kernel.py:class TestCudaComplex(CUDATestCase):
numba/cuda/tests/cudapy/test_complex_kernel.py:    def test_cuda_complex_arg(self):
numba/cuda/tests/cudapy/test_complex_kernel.py:        @cuda.jit('void(complex128[:], complex128)')
numba/cuda/tests/cudapy/test_complex_kernel.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_dispatcher.py:from numba import boolean, config, cuda, float32, float64, int32, int64, void
numba/cuda/tests/cudapy/test_dispatcher.py:from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
numba/cuda/tests/cudapy/test_dispatcher.py:@skip_on_cudasim('Specialization not implemented in the simulator')
numba/cuda/tests/cudapy/test_dispatcher.py:class TestDispatcherSpecialization(CUDATestCase):
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit('void(float32[::1])')
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit('void(int32[::1])')
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit
numba/cuda/tests/cudapy/test_dispatcher.py:class TestDispatcher(CUDATestCase):
numba/cuda/tests/cudapy/test_dispatcher.py:        c_add = cuda.jit(add_kernel)
numba/cuda/tests/cudapy/test_dispatcher.py:        c_add = cuda.jit('(i4[::1], i4, i4)')(add_kernel)
numba/cuda/tests/cudapy/test_dispatcher.py:    @skip_on_cudasim('Simulator ignores signature')
numba/cuda/tests/cudapy/test_dispatcher.py:        # test_coerce_input_types. This test presently fails with the CUDA
numba/cuda/tests/cudapy/test_dispatcher.py:        c_add = cuda.jit('(i4[::1], i4, i4)')(add_kernel)
numba/cuda/tests/cudapy/test_dispatcher.py:    @skip_on_cudasim('Simulator ignores signature')
numba/cuda/tests/cudapy/test_dispatcher.py:        c_add = cuda.jit('(i4[::1], i4, i4)')(add_kernel)
numba/cuda/tests/cudapy/test_dispatcher.py:    @skip_on_cudasim('Simulator does not track overloads')
numba/cuda/tests/cudapy/test_dispatcher.py:        c_add = cuda.jit(add_kernel)
numba/cuda/tests/cudapy/test_dispatcher.py:    @skip_on_cudasim("Simulator doesn't support concurrent kernels")
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit
numba/cuda/tests/cudapy/test_dispatcher.py:        f = cuda.jit(sigs)(add_kernel)
numba/cuda/tests/cudapy/test_dispatcher.py:        if config.ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_dispatcher.py:        f = cuda.jit(sigs)(add_kernel)
numba/cuda/tests/cudapy/test_dispatcher.py:    @skip_on_cudasim('No overload resolution in the simulator')
numba/cuda/tests/cudapy/test_dispatcher.py:        f = cuda.jit(["(float64[::1], float32, float64)",
numba/cuda/tests/cudapy/test_dispatcher.py:    @skip_on_cudasim('Simulator does not use _prepare_args')
numba/cuda/tests/cudapy/test_dispatcher.py:        # at present because _prepare_args in the CUDA target cannot handle
numba/cuda/tests/cudapy/test_dispatcher.py:        f = cuda.jit("(int64[::1], int64, int64)")(add_kernel)
numba/cuda/tests/cudapy/test_dispatcher.py:        f = cuda.jit(sigs)(add_kernel)
numba/cuda/tests/cudapy/test_dispatcher.py:        add_device = cuda.jit(sigs, device=True)(add)
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit
numba/cuda/tests/cudapy/test_dispatcher.py:        if config.ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_dispatcher.py:    @skip_on_cudasim('CUDA Simulator does not force casting')
numba/cuda/tests/cudapy/test_dispatcher.py:        # variant of these tests can succeed on CUDA because the compilation
numba/cuda/tests/cudapy/test_dispatcher.py:        # Ensure that CUDA-jitting a function preserves its docstring. See
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit(device=True)
numba/cuda/tests/cudapy/test_dispatcher.py:@skip_on_cudasim("CUDA simulator doesn't implement kernel properties")
numba/cuda/tests/cudapy/test_dispatcher.py:class TestDispatcherKernelProperties(CUDATestCase):
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit
numba/cuda/tests/cudapy/test_dispatcher.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_dispatcher.py:            cuda.detect()
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit(void(float32[::1], int64))
numba/cuda/tests/cudapy/test_dispatcher.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit(sig)
numba/cuda/tests/cudapy/test_dispatcher.py:            C = cuda.const.array_like(arr)
numba/cuda/tests/cudapy/test_dispatcher.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit
numba/cuda/tests/cudapy/test_dispatcher.py:            sm = cuda.shared.array(N, dtype=ary.dtype)
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit(void(float32[::1]))
numba/cuda/tests/cudapy/test_dispatcher.py:            sm = cuda.shared.array(100, dtype=float32)
numba/cuda/tests/cudapy/test_dispatcher.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_dispatcher.py:            cuda.syncthreads()
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit
numba/cuda/tests/cudapy/test_dispatcher.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_dispatcher.py:        # cuda.local.array and use local registers instead
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit
numba/cuda/tests/cudapy/test_dispatcher.py:            lm = cuda.local.array(N, dtype=ary.dtype)
numba/cuda/tests/cudapy/test_dispatcher.py:        # cuda.local.array and use local registers instead
numba/cuda/tests/cudapy/test_dispatcher.py:        @cuda.jit(void(float32[::1]))
numba/cuda/tests/cudapy/test_dispatcher.py:            lm = cuda.local.array(N, dtype=ary.dtype)
numba/cuda/tests/cudapy/test_warp_ops.py:from numba import cuda, int32, int64, float32, float64
numba/cuda/tests/cudapy/test_warp_ops.py:from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
numba/cuda/tests/cudapy/test_warp_ops.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_warp_ops.py:    cuda.syncwarp(0xffffffff)
numba/cuda/tests/cudapy/test_warp_ops.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_warp_ops.py:    val = cuda.shfl_sync(0xffffffff, i, idx)
numba/cuda/tests/cudapy/test_warp_ops.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_warp_ops.py:    val = cuda.shfl_up_sync(0xffffffff, i, delta)
numba/cuda/tests/cudapy/test_warp_ops.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_warp_ops.py:    val = cuda.shfl_down_sync(0xffffffff, i, delta)
numba/cuda/tests/cudapy/test_warp_ops.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_warp_ops.py:    val = cuda.shfl_xor_sync(0xffffffff, i, xor)
numba/cuda/tests/cudapy/test_warp_ops.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_warp_ops.py:    val = cuda.shfl_sync(0xffffffff, into, 0)
numba/cuda/tests/cudapy/test_warp_ops.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_warp_ops.py:    pred = cuda.all_sync(0xffffffff, ary_in[i])
numba/cuda/tests/cudapy/test_warp_ops.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_warp_ops.py:    pred = cuda.any_sync(0xffffffff, ary_in[i])
numba/cuda/tests/cudapy/test_warp_ops.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_warp_ops.py:    pred = cuda.eq_sync(0xffffffff, ary_in[i])
numba/cuda/tests/cudapy/test_warp_ops.py:    i = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_warp_ops.py:    ballot = cuda.ballot_sync(0xffffffff, True)
numba/cuda/tests/cudapy/test_warp_ops.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_warp_ops.py:    ballot = cuda.match_any_sync(0xffffffff, ary_in[i])
numba/cuda/tests/cudapy/test_warp_ops.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_warp_ops.py:    ballot, pred = cuda.match_all_sync(0xffffffff, ary_in[i])
numba/cuda/tests/cudapy/test_warp_ops.py:    i = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_warp_ops.py:        ballot = cuda.ballot_sync(0x11111111, True)
numba/cuda/tests/cudapy/test_warp_ops.py:        ballot = cuda.ballot_sync(0x22222222, True)
numba/cuda/tests/cudapy/test_warp_ops.py:        ballot = cuda.ballot_sync(0x44444444, True)
numba/cuda/tests/cudapy/test_warp_ops.py:        ballot = cuda.ballot_sync(0x88888888, True)
numba/cuda/tests/cudapy/test_warp_ops.py:    if config.ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_warp_ops.py:        return cuda.get_current_device().compute_capability >= cc
numba/cuda/tests/cudapy/test_warp_ops.py:@skip_on_cudasim("Warp Operations are not yet implemented on cudasim")
numba/cuda/tests/cudapy/test_warp_ops.py:class TestCudaWarpOperations(CUDATestCase):
numba/cuda/tests/cudapy/test_warp_ops.py:        compiled = cuda.jit("void(int32[:])")(useful_syncwarp)
numba/cuda/tests/cudapy/test_warp_ops.py:        compiled = cuda.jit("void(int32[:], int32)")(use_shfl_sync_idx)
numba/cuda/tests/cudapy/test_warp_ops.py:        compiled = cuda.jit("void(int32[:], int32)")(use_shfl_sync_up)
numba/cuda/tests/cudapy/test_warp_ops.py:        compiled = cuda.jit("void(int32[:], int32)")(use_shfl_sync_down)
numba/cuda/tests/cudapy/test_warp_ops.py:        compiled = cuda.jit("void(int32[:], int32)")(use_shfl_sync_xor)
numba/cuda/tests/cudapy/test_warp_ops.py:            compiled = cuda.jit((typ[:], typ))(use_shfl_sync_with_val)
numba/cuda/tests/cudapy/test_warp_ops.py:        compiled = cuda.jit("void(int32[:], int32[:])")(use_vote_sync_all)
numba/cuda/tests/cudapy/test_warp_ops.py:        compiled = cuda.jit("void(int32[:], int32[:])")(use_vote_sync_any)
numba/cuda/tests/cudapy/test_warp_ops.py:        compiled = cuda.jit("void(int32[:], int32[:])")(use_vote_sync_eq)
numba/cuda/tests/cudapy/test_warp_ops.py:        compiled = cuda.jit("void(uint32[:])")(use_vote_sync_ballot)
numba/cuda/tests/cudapy/test_warp_ops.py:        compiled = cuda.jit("void(int32[:], int32[:])")(use_match_any_sync)
numba/cuda/tests/cudapy/test_warp_ops.py:        compiled = cuda.jit("void(int32[:], int32[:])")(use_match_all_sync)
numba/cuda/tests/cudapy/test_warp_ops.py:        compiled = cuda.jit("void(uint32[:])")(use_independent_scheduling)
numba/cuda/tests/cudapy/test_warp_ops.py:        @cuda.jit
numba/cuda/tests/cudapy/test_warp_ops.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_warp_ops.py:                x[i] = cuda.activemask()
numba/cuda/tests/cudapy/test_warp_ops.py:                x[i] = cuda.activemask()
numba/cuda/tests/cudapy/test_warp_ops.py:        @cuda.jit
numba/cuda/tests/cudapy/test_warp_ops.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_warp_ops.py:            x[i] = cuda.lanemask_lt()
numba/cuda/tests/cudapy/test_multigpu.py:from numba import cuda
numba/cuda/tests/cudapy/test_multigpu.py:from numba.cuda.testing import skip_on_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_multigpu.py:class TestMultiGPUContext(CUDATestCase):
numba/cuda/tests/cudapy/test_multigpu.py:    @unittest.skipIf(len(cuda.gpus) < 2, "need more than 1 gpus")
numba/cuda/tests/cudapy/test_multigpu.py:    def test_multigpu_context(self):
numba/cuda/tests/cudapy/test_multigpu.py:        @cuda.jit("void(float64[:], float64[:])")
numba/cuda/tests/cudapy/test_multigpu.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_multigpu.py:        with cuda.gpus[0]:
numba/cuda/tests/cudapy/test_multigpu.py:        with cuda.gpus[0]:
numba/cuda/tests/cudapy/test_multigpu.py:            with cuda.gpus[1]:
numba/cuda/tests/cudapy/test_multigpu.py:    @skip_on_cudasim('Simulator does not support multiple threads')
numba/cuda/tests/cudapy/test_multigpu.py:        def work(gpu, dA, results, ridx):
numba/cuda/tests/cudapy/test_multigpu.py:                with gpu:
numba/cuda/tests/cudapy/test_multigpu.py:        dA = cuda.to_device(np.arange(10))
numba/cuda/tests/cudapy/test_multigpu.py:        threads = [threading.Thread(target=work, args=(cuda.gpus.current,
numba/cuda/tests/cudapy/test_multigpu.py:    @unittest.skipIf(len(cuda.gpus) < 2, "need more than 1 gpus")
numba/cuda/tests/cudapy/test_multigpu.py:        @cuda.jit
numba/cuda/tests/cudapy/test_multigpu.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_multigpu.py:        with cuda.gpus[0]:
numba/cuda/tests/cudapy/test_multigpu.py:            arr1 = cuda.to_device(hostarr)
numba/cuda/tests/cudapy/test_multigpu.py:        with cuda.gpus[1]:
numba/cuda/tests/cudapy/test_multigpu.py:            arr2 = cuda.to_device(hostarr)
numba/cuda/tests/cudapy/test_multigpu.py:        with cuda.gpus[0]:
numba/cuda/tests/cudapy/test_multigpu.py:        with cuda.gpus[1]:
numba/cuda/tests/cudapy/test_multigpu.py:        with cuda.gpus[0]:
numba/cuda/tests/cudapy/test_multigpu.py:        with cuda.gpus[1]:
numba/cuda/tests/cudapy/test_multigpu.py:    @unittest.skipIf(len(cuda.gpus) < 2, "need more than 1 gpus")
numba/cuda/tests/cudapy/test_multigpu.py:        # Peer access is not always possible - for example, with one GPU in TCC
numba/cuda/tests/cudapy/test_multigpu.py:        with cuda.gpus[0]:
numba/cuda/tests/cudapy/test_multigpu.py:            ctx = cuda.current_context()
numba/cuda/tests/cudapy/test_multigpu.py:                self.skipTest('Peer access between GPUs disabled')
numba/cuda/tests/cudapy/test_multigpu.py:        # 2. Copy range array from host -> GPU 0
numba/cuda/tests/cudapy/test_multigpu.py:        with cuda.gpus[0]:
numba/cuda/tests/cudapy/test_multigpu.py:            arr1 = cuda.to_device(hostarr)
numba/cuda/tests/cudapy/test_multigpu.py:        # 3. Initialize a zero-filled array on GPU 1
numba/cuda/tests/cudapy/test_multigpu.py:        with cuda.gpus[1]:
numba/cuda/tests/cudapy/test_multigpu.py:            arr2 = cuda.to_device(np.zeros_like(hostarr))
numba/cuda/tests/cudapy/test_multigpu.py:        with cuda.gpus[0]:
numba/cuda/tests/cudapy/test_multigpu.py:            # 4. Copy range from GPU 0 -> GPU 1
numba/cuda/tests/cudapy/test_multigpu.py:            # 5. Copy range from GPU 1 -> host and check contents
numba/cuda/tests/cudapy/test_globals.py:from numba import cuda, int32, float32
numba/cuda/tests/cudapy/test_globals.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_globals.py:    sm = cuda.shared.array(N, int32)
numba/cuda/tests/cudapy/test_globals.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_globals.py:    cuda.syncthreads()
numba/cuda/tests/cudapy/test_globals.py:    i, j = cuda.grid(2)
numba/cuda/tests/cudapy/test_globals.py:    sm = cuda.shared.array((S0, S1), float32)
numba/cuda/tests/cudapy/test_globals.py:    cuda.syncthreads()
numba/cuda/tests/cudapy/test_globals.py:class TestCudaTestGlobal(CUDATestCase):
numba/cuda/tests/cudapy/test_globals.py:        compiled = cuda.jit("void(int32[:])")(simple_smem)
numba/cuda/tests/cudapy/test_globals.py:        compiled = cuda.jit("void(float32[:,:])")(coop_smem2d)
numba/cuda/tests/cudapy/test_mandel.py:from numba.cuda.compiler import compile_ptx
numba/cuda/tests/cudapy/test_mandel.py:from numba.cuda.testing import skip_on_cudasim, unittest
numba/cuda/tests/cudapy/test_mandel.py:@skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_mandel.py:class TestCudaMandel(unittest.TestCase):
numba/cuda/tests/cudapy/test_vectorize_complex.py:from numba.cuda.testing import skip_on_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_vectorize_complex.py:@skip_on_cudasim('ufunc API unsupported in the simulator')
numba/cuda/tests/cudapy/test_vectorize_complex.py:class TestVectorizeComplex(CUDATestCase):
numba/cuda/tests/cudapy/test_vectorize_complex.py:        @vectorize(['complex128(complex128)'], target='cuda')
numba/cuda/tests/cudapy/test_datetime.py:from numba import cuda, vectorize, guvectorize
numba/cuda/tests/cudapy/test_datetime.py:from numba.cuda.testing import CUDATestCase, skip_on_cudasim
numba/cuda/tests/cudapy/test_datetime.py:class TestCudaDateTime(CUDATestCase):
numba/cuda/tests/cudapy/test_datetime.py:        @cuda.jit
numba/cuda/tests/cudapy/test_datetime.py:            for i in range(cuda.grid(1), delta.size, cuda.gridsize(1)):
numba/cuda/tests/cudapy/test_datetime.py:        @cuda.jit
numba/cuda/tests/cudapy/test_datetime.py:            for i in range(cuda.grid(1), matches.size, cuda.gridsize(1)):
numba/cuda/tests/cudapy/test_datetime.py:    @skip_on_cudasim('ufunc API unsupported in the simulator')
numba/cuda/tests/cudapy/test_datetime.py:        @vectorize([(datetime_t, datetime_t)], target='cuda')
numba/cuda/tests/cudapy/test_datetime.py:    @skip_on_cudasim('ufunc API unsupported in the simulator')
numba/cuda/tests/cudapy/test_datetime.py:                     target='cuda')
numba/cuda/tests/cudapy/test_datetime.py:    @skip_on_cudasim('no .copy_to_host() in the simulator')
numba/cuda/tests/cudapy/test_datetime.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudapy/test_datetime.py:        self.assertEqual(viewed.gpu_data, darr.gpu_data)
numba/cuda/tests/cudapy/test_datetime.py:    @skip_on_cudasim('no .copy_to_host() in the simulator')
numba/cuda/tests/cudapy/test_datetime.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudapy/test_datetime.py:        self.assertEqual(viewed.gpu_data, darr.gpu_data)
numba/cuda/tests/cudapy/test_vectorize.py:from numba import cuda, int32, float32, float64
numba/cuda/tests/cudapy/test_vectorize.py:from numba.cuda.cudadrv.driver import CudaAPIError, driver
numba/cuda/tests/cudapy/test_vectorize.py:from numba.cuda.testing import skip_on_cudasim
numba/cuda/tests/cudapy/test_vectorize.py:from numba.cuda.testing import CUDATestCase
numba/cuda/tests/cudapy/test_vectorize.py:# casted to a previously-used dtype. This is unlikely to be an issue for CUDA,
numba/cuda/tests/cudapy/test_vectorize.py:@skip_on_cudasim('ufunc API unsupported in the simulator')
numba/cuda/tests/cudapy/test_vectorize.py:class TestCUDAVectorize(CUDATestCase):
numba/cuda/tests/cudapy/test_vectorize.py:        @vectorize(signatures, target='cuda')
numba/cuda/tests/cudapy/test_vectorize.py:        @vectorize(signatures, target='cuda')
numba/cuda/tests/cudapy/test_vectorize.py:        @vectorize(signatures, target='cuda')
numba/cuda/tests/cudapy/test_vectorize.py:        stream = cuda.stream()
numba/cuda/tests/cudapy/test_vectorize.py:            device_data = cuda.to_device(data, stream)
numba/cuda/tests/cudapy/test_vectorize.py:        @vectorize(signatures, target='cuda')
numba/cuda/tests/cudapy/test_vectorize.py:        @vectorize(signatures, target='cuda')
numba/cuda/tests/cudapy/test_vectorize.py:        @vectorize(signatures, target='cuda')
numba/cuda/tests/cudapy/test_vectorize.py:        @vectorize(signatures, target='cuda')
numba/cuda/tests/cudapy/test_vectorize.py:        stream = cuda.stream()
numba/cuda/tests/cudapy/test_vectorize.py:            dx = cuda.to_device(x, stream)
numba/cuda/tests/cudapy/test_vectorize.py:        @vectorize(signatures, target='cuda')
numba/cuda/tests/cudapy/test_vectorize.py:        dx = cuda.to_device(x)
numba/cuda/tests/cudapy/test_vectorize.py:        @vectorize(signatures, target='cuda')
numba/cuda/tests/cudapy/test_vectorize.py:        dx = cuda.to_device(x)
numba/cuda/tests/cudapy/test_vectorize.py:        @vectorize(signatures, target='cuda')
numba/cuda/tests/cudapy/test_vectorize.py:        @vectorize('f8(f8)', target='cuda')
numba/cuda/tests/cudapy/test_vectorize.py:        noise = cuda.to_device(noise)
numba/cuda/tests/cudapy/test_vectorize.py:        # A mock of a CUDA function that always raises a CudaAPIError
numba/cuda/tests/cudapy/test_vectorize.py:            raise CudaAPIError(999, 'Transfer not allowed')
numba/cuda/tests/cudapy/test_vectorize.py:        with self.assertRaisesRegex(CudaAPIError, "Transfer not allowed"):
numba/cuda/tests/cudapy/test_vectorize.py:        with self.assertRaisesRegex(CudaAPIError, "Transfer not allowed"):
numba/cuda/tests/cudapy/test_vectorize.py:            cuda.to_device([1])
numba/cuda/tests/cudapy/test_vectorize.py:            @vectorize(['float32(float32)'], target='cuda')
numba/cuda/tests/cudapy/test_array_args.py:from numba import cuda
numba/cuda/tests/cudapy/test_array_args.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_array_args.py:class TestCudaArrayArg(CUDATestCase):
numba/cuda/tests/cudapy/test_array_args.py:        @cuda.jit('double(double[:],int64)', device=True, inline=True)
numba/cuda/tests/cudapy/test_array_args.py:        @cuda.jit('void(double[:],double[:])')
numba/cuda/tests/cudapy/test_array_args.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_array_args.py:        @cuda.jit
numba/cuda/tests/cudapy/test_array_args.py:        @cuda.jit
numba/cuda/tests/cudapy/test_array_args.py:        @cuda.jit
numba/cuda/tests/cudapy/test_array_args.py:        @cuda.jit
numba/cuda/tests/cudapy/test_array_args.py:        @cuda.jit
numba/cuda/tests/cudapy/test_array_args.py:        @cuda.jit
numba/cuda/tests/cudapy/test_array_args.py:        @cuda.jit
numba/cuda/tests/cudapy/test_array_args.py:        @cuda.jit
numba/cuda/tests/cudapy/test_array_args.py:        @cuda.jit
numba/cuda/tests/cudapy/test_array_args.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_array_args.py:        @cuda.jit
numba/cuda/tests/cudapy/test_vectorize_decor.py:from numba import vectorize, cuda
numba/cuda/tests/cudapy/test_vectorize_decor.py:from numba.cuda.testing import skip_on_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_vectorize_decor.py:@skip_on_cudasim('ufunc API unsupported in the simulator')
numba/cuda/tests/cudapy/test_vectorize_decor.py:class TestVectorizeDecor(CUDATestCase, BaseVectorizeDecor):
numba/cuda/tests/cudapy/test_vectorize_decor.py:    Runs the tests from BaseVectorizeDecor with the CUDA target.
numba/cuda/tests/cudapy/test_vectorize_decor.py:    target = 'cuda'
numba/cuda/tests/cudapy/test_vectorize_decor.py:@skip_on_cudasim('ufunc API unsupported in the simulator')
numba/cuda/tests/cudapy/test_vectorize_decor.py:class TestGPUVectorizeBroadcast(CUDATestCase):
numba/cuda/tests/cudapy/test_vectorize_decor.py:        @vectorize(['float64(float64,float64)'], target='cuda')
numba/cuda/tests/cudapy/test_vectorize_decor.py:        def fngpu(a, b):
numba/cuda/tests/cudapy/test_vectorize_decor.py:        got = fngpu(a, b)
numba/cuda/tests/cudapy/test_vectorize_decor.py:        @vectorize(['float64(float64,float64)'], target='cuda')
numba/cuda/tests/cudapy/test_vectorize_decor.py:        def fngpu(a, b):
numba/cuda/tests/cudapy/test_vectorize_decor.py:        got = fngpu(cuda.to_device(a), cuda.to_device(b))
numba/cuda/tests/cudapy/test_vectorize_decor.py:@skip_on_cudasim('ufunc API unsupported in the simulator')
numba/cuda/tests/cudapy/test_vectorize_decor.py:class TestVectorizeNopythonArg(BaseVectorizeNopythonArg, CUDATestCase):
numba/cuda/tests/cudapy/test_vectorize_decor.py:    def test_target_cuda_nopython(self):
numba/cuda/tests/cudapy/test_vectorize_decor.py:        warnings = ["nopython kwarg for cuda target is redundant"]
numba/cuda/tests/cudapy/test_vectorize_decor.py:        self._test_target_nopython('cuda', warnings)
numba/cuda/tests/cudapy/test_vectorize_decor.py:@skip_on_cudasim('ufunc API unsupported in the simulator')
numba/cuda/tests/cudapy/test_vectorize_decor.py:class TestVectorizeUnrecognizedArg(BaseVectorizeUnrecognizedArg, CUDATestCase):
numba/cuda/tests/cudapy/test_vectorize_decor.py:    def test_target_cuda_unrecognized_arg(self):
numba/cuda/tests/cudapy/test_vectorize_decor.py:        self._test_target_unrecognized_arg('cuda')
numba/cuda/tests/cudapy/test_sm.py:from numba import cuda, int32, float64, void
numba/cuda/tests/cudapy/test_sm.py:from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
numba/cuda/tests/cudapy/test_sm.py:class TestSharedMemoryIssue(CUDATestCase):
numba/cuda/tests/cudapy/test_sm.py:        @cuda.jit(device=True)
numba/cuda/tests/cudapy/test_sm.py:            inner_arr = cuda.shared.array(1, dtype=int32)  # noqa: F841
numba/cuda/tests/cudapy/test_sm.py:        @cuda.jit
numba/cuda/tests/cudapy/test_sm.py:            outer_arr = cuda.shared.array(1, dtype=int32)  # noqa: F841
numba/cuda/tests/cudapy/test_sm.py:        @cuda.jit
numba/cuda/tests/cudapy/test_sm.py:            arr = cuda.shared.array(shape, dtype=int32)
numba/cuda/tests/cudapy/test_sm.py:        @cuda.jit
numba/cuda/tests/cudapy/test_sm.py:            arr = cuda.shared.array(shape, dtype=ty)
numba/cuda/tests/cudapy/test_sm.py:        @cuda.jit
numba/cuda/tests/cudapy/test_sm.py:            s_features = cuda.shared.array((examples_per_block, num_weights),
numba/cuda/tests/cudapy/test_sm.py:            s_initialcost = cuda.shared.array(7, float64)  # Bug
numba/cuda/tests/cudapy/test_sm.py:            threadIdx = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_sm.py:        d_block_costs = cuda.to_device(block_costs)
numba/cuda/tests/cudapy/test_sm.py:        cuda.synchronize()
numba/cuda/tests/cudapy/test_sm.py:class TestSharedMemory(CUDATestCase):
numba/cuda/tests/cudapy/test_sm.py:        @cuda.jit
numba/cuda/tests/cudapy/test_sm.py:            sm = cuda.shared.array(nthreads, dtype=dt)
numba/cuda/tests/cudapy/test_sm.py:            tx = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_sm.py:            bx = cuda.blockIdx.x
numba/cuda/tests/cudapy/test_sm.py:            bd = cuda.blockDim.x
numba/cuda/tests/cudapy/test_sm.py:            cuda.syncthreads()
numba/cuda/tests/cudapy/test_sm.py:        d_result = cuda.device_array_like(arr)
numba/cuda/tests/cudapy/test_sm.py:        @cuda.jit
numba/cuda/tests/cudapy/test_sm.py:            dynsmem = cuda.shared.array(0, dtype=int32)
numba/cuda/tests/cudapy/test_sm.py:        @cuda.jit
numba/cuda/tests/cudapy/test_sm.py:            dynsmem = cuda.shared.array(0, dtype=int32)
numba/cuda/tests/cudapy/test_sm.py:        @cuda.jit
numba/cuda/tests/cudapy/test_sm.py:            dynsmem = cuda.shared.array(0, dtype=int32)
numba/cuda/tests/cudapy/test_sm.py:        @cuda.jit
numba/cuda/tests/cudapy/test_sm.py:            dynsmem = cuda.shared.array(0, dtype=int32)
numba/cuda/tests/cudapy/test_sm.py:        @cuda.jit
numba/cuda/tests/cudapy/test_sm.py:            dynsmem = cuda.shared.array(0, dtype=int32)
numba/cuda/tests/cudapy/test_sm.py:        @cuda.jit
numba/cuda/tests/cudapy/test_sm.py:            dynsmem = cuda.shared.array(0, dtype=int32)
numba/cuda/tests/cudapy/test_sm.py:        @cuda.jit
numba/cuda/tests/cudapy/test_sm.py:            dynsmem = cuda.shared.array(0, dtype=int32)
numba/cuda/tests/cudapy/test_sm.py:        @cuda.jit
numba/cuda/tests/cudapy/test_sm.py:            dynsmem = cuda.shared.array(0, dtype=int32)
numba/cuda/tests/cudapy/test_sm.py:        @cuda.jit
numba/cuda/tests/cudapy/test_sm.py:            dynsmem = cuda.shared.array(0, dtype=dt)
numba/cuda/tests/cudapy/test_sm.py:            tx = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_sm.py:            bx = cuda.blockIdx.x
numba/cuda/tests/cudapy/test_sm.py:            bd = cuda.blockDim.x
numba/cuda/tests/cudapy/test_sm.py:            cuda.syncthreads()
numba/cuda/tests/cudapy/test_sm.py:        d_result = cuda.device_array_like(arr)
numba/cuda/tests/cudapy/test_sm.py:    @skip_on_cudasim("Can't check typing in simulator")
numba/cuda/tests/cudapy/test_sm.py:            arr = cuda.shared.array(10, dtype=np.dtype('O')) # noqa: F841
numba/cuda/tests/cudapy/test_sm.py:            cuda.jit(void())(unsupported_type)
numba/cuda/tests/cudapy/test_sm.py:            arr = cuda.shared.array(10, dtype='int33') # noqa: F841
numba/cuda/tests/cudapy/test_sm.py:            cuda.jit(void())(invalid_string_type)
numba/cuda/tests/cudapy/test_sm.py:    @skip_on_cudasim("Struct model array unsupported in simulator")
numba/cuda/tests/cudapy/test_sm.py:        @cuda.jit(void(int32[::1], int32[::1]))
numba/cuda/tests/cudapy/test_sm.py:            arr = cuda.shared.array(nthreads, dtype=test_struct_model_type)
numba/cuda/tests/cudapy/test_sm.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_sm.py:                cuda.syncthreads()
numba/cuda/tests/cudapy/test_atomics.py:from numba import cuda, uint32, uint64, float32, float64
numba/cuda/tests/cudapy/test_atomics.py:from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
numba/cuda/tests/cudapy/test_atomics.py:@cuda.jit(device=True)
numba/cuda/tests/cudapy/test_atomics.py:@cuda.jit(device=True)
numba/cuda/tests/cudapy/test_atomics.py:@cuda.jit(device=True)
numba/cuda/tests/cudapy/test_atomics.py:@cuda.jit(device=True)
numba/cuda/tests/cudapy/test_atomics.py:    tid = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_atomics.py:    sm = cuda.shared.array(ary_nelements, ary_dtype)
numba/cuda/tests/cudapy/test_atomics.py:    cuda.syncthreads()
numba/cuda/tests/cudapy/test_atomics.py:    cuda.syncthreads()
numba/cuda/tests/cudapy/test_atomics.py:@cuda.jit(device=True)
numba/cuda/tests/cudapy/test_atomics.py:    tid = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_atomics.py:    sm = cuda.shared.array(ary_nelements, ary_dtype)
numba/cuda/tests/cudapy/test_atomics.py:    cuda.syncthreads()
numba/cuda/tests/cudapy/test_atomics.py:    cuda.syncthreads()
numba/cuda/tests/cudapy/test_atomics.py:@cuda.jit(device=True)
numba/cuda/tests/cudapy/test_atomics.py:    tx = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_atomics.py:    ty = cuda.threadIdx.y
numba/cuda/tests/cudapy/test_atomics.py:    sm = cuda.shared.array(ary_shape, ary_dtype)
numba/cuda/tests/cudapy/test_atomics.py:    cuda.syncthreads()
numba/cuda/tests/cudapy/test_atomics.py:    cuda.syncthreads()
numba/cuda/tests/cudapy/test_atomics.py:@cuda.jit(device=True)
numba/cuda/tests/cudapy/test_atomics.py:    tx = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_atomics.py:    ty = cuda.threadIdx.y
numba/cuda/tests/cudapy/test_atomics.py:@cuda.jit(device=True)
numba/cuda/tests/cudapy/test_atomics.py:    tid = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.add, atomic_cast_none, 0, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.add, atomic_cast_none, 0, True)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.add, atomic_cast_none, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.add, atomic_cast_none, True)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.add, atomic_cast_to_uint64, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.add, atomic_cast_to_int, 0.0, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.add, atomic_cast_to_int, 0.0, True)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.add, atomic_cast_none, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.add, atomic_cast_none, True)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.add, atomic_cast_to_uint64, False)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_1dim_global(ary, idx, 32, 1.0, cuda.atomic.add, False)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_1dim_global(ary, idx, 32, 1.0, cuda.atomic.add, True)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_2dim_global(ary, 1, cuda.atomic.add, atomic_cast_none, False)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_2dim_global(ary, 1, cuda.atomic.add, atomic_cast_none, True)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_2dim_global(ary, 1, cuda.atomic.add, atomic_cast_to_uint64,
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.add, atomic_cast_none, 0.0, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.add, atomic_cast_none, 0.0, True)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.add, atomic_cast_none, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.add, atomic_cast_none, True)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.add, atomic_cast_to_uint64, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.sub, atomic_cast_none, 0, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.sub, atomic_cast_none, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.sub, atomic_cast_to_uint64, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.sub, atomic_cast_to_int, 0.0, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.sub, atomic_cast_none, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.sub, atomic_cast_to_uint64, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.sub, atomic_cast_none, 0.0, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.sub, atomic_cast_none, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.sub, atomic_cast_to_uint64, False)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_1dim_global(ary, idx, 32, 1.0, cuda.atomic.sub, False)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_2dim_global(ary, 1.0, cuda.atomic.sub, atomic_cast_none,
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.sub, atomic_cast_to_uint64, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.and_, atomic_cast_none, 1, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.and_, atomic_cast_none, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.and_, atomic_cast_to_uint64, False)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_1dim_global(ary, idx, 32, op2, cuda.atomic.and_, False)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_2dim_global(ary, op2, cuda.atomic.and_,
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.or_, atomic_cast_none, 0, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.or_, atomic_cast_none, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.or_, atomic_cast_to_uint64, False)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_1dim_global(ary, idx, 32, op2, cuda.atomic.or_, False)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_2dim_global(ary, op2, cuda.atomic.or_,
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.xor, atomic_cast_none, 0, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.xor, atomic_cast_none, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.xor, atomic_cast_to_uint64, False)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_1dim_global(ary, idx, 32, op2, cuda.atomic.xor, False)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_2dim_global(ary, op2, cuda.atomic.xor,
numba/cuda/tests/cudapy/test_atomics.py:                               cuda.atomic.inc, atomic_cast_none)
numba/cuda/tests/cudapy/test_atomics.py:                               cuda.atomic.inc, atomic_cast_to_int)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.inc, atomic_cast_none, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.inc, atomic_cast_none, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.inc, atomic_cast_to_uint64, False)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_1dim_global(ary, idx, 32, op2, cuda.atomic.inc, False)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_2dim_global(ary, op2, cuda.atomic.inc,
numba/cuda/tests/cudapy/test_atomics.py:                               cuda.atomic.dec, atomic_cast_none)
numba/cuda/tests/cudapy/test_atomics.py:                               cuda.atomic.dec, atomic_cast_to_int)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.dec, atomic_cast_none, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.dec, atomic_cast_none, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.dec, atomic_cast_to_uint64, False)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_1dim_global(ary, idx, 32, op2, cuda.atomic.dec, False)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_2dim_global(ary, op2, cuda.atomic.dec,
numba/cuda/tests/cudapy/test_atomics.py:                               cuda.atomic.exch, atomic_cast_none)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.exch, atomic_cast_none, False)
numba/cuda/tests/cudapy/test_atomics.py:                              cuda.atomic.exch, atomic_cast_none, False)
numba/cuda/tests/cudapy/test_atomics.py:    atomic_binary_1dim_global(ary, idx, 32, op2, cuda.atomic.exch, False)
numba/cuda/tests/cudapy/test_atomics.py:        tx = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_atomics.py:        bx = cuda.blockIdx.x
numba/cuda/tests/cudapy/test_atomics.py:        tx = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_atomics.py:        bx = cuda.blockIdx.x
numba/cuda/tests/cudapy/test_atomics.py:        tx = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_atomics.py:        tid = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_atomics.py:        smary = cuda.shared.array(32, float64)
numba/cuda/tests/cudapy/test_atomics.py:        smres = cuda.shared.array(1, float64)
numba/cuda/tests/cudapy/test_atomics.py:        cuda.syncthreads()
numba/cuda/tests/cudapy/test_atomics.py:        cuda.syncthreads()
numba/cuda/tests/cudapy/test_atomics.py:    exec(fns, {'cuda': cuda, 'float64': float64, 'uint64': uint64}, ld)
numba/cuda/tests/cudapy/test_atomics.py: atomic_max_double_shared) = gen_atomic_extreme_funcs('cuda.atomic.max')
numba/cuda/tests/cudapy/test_atomics.py: atomic_min_double_shared) = gen_atomic_extreme_funcs('cuda.atomic.min')
numba/cuda/tests/cudapy/test_atomics.py:    gen_atomic_extreme_funcs('cuda.atomic.nanmax')
numba/cuda/tests/cudapy/test_atomics.py:    gen_atomic_extreme_funcs('cuda.atomic.nanmin')
numba/cuda/tests/cudapy/test_atomics.py:    gid = cuda.grid(1)
numba/cuda/tests/cudapy/test_atomics.py:        old[gid] = cuda.atomic.compare_and_swap(res[gid:], fill_val, ary[gid])
numba/cuda/tests/cudapy/test_atomics.py:    gid = cuda.grid(1)
numba/cuda/tests/cudapy/test_atomics.py:        old[gid] = cuda.atomic.cas(res, gid, fill_val, ary[gid])
numba/cuda/tests/cudapy/test_atomics.py:    gid = cuda.grid(2)
numba/cuda/tests/cudapy/test_atomics.py:        old[gid] = cuda.atomic.cas(res, gid, fill_val, ary[gid])
numba/cuda/tests/cudapy/test_atomics.py:class TestCudaAtomics(CUDATestCase):
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_add = cuda.jit('void(uint32[:])')(atomic_add)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_add[1, 32](ary)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_add_wrap = cuda.jit('void(uint32[:])')(atomic_add_wrap)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_add_wrap[1, 32](ary_wrap)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_add2 = cuda.jit('void(uint32[:,:])')(atomic_add2)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_add2[1, (4, 8)](ary)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_add2_wrap = cuda.jit('void(uint32[:,:])')(atomic_add2_wrap)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_add2_wrap[1, (4, 8)](ary_wrap)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_add3 = cuda.jit('void(uint32[:,:])')(atomic_add3)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_add3[1, (4, 8)](ary)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_add_float = cuda.jit('void(float32[:])')(atomic_add_float)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_add_float[1, 32](ary)
numba/cuda/tests/cudapy/test_atomics.py:        add_float_wrap = cuda.jit('void(float32[:])')(atomic_add_float_wrap)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_add2 = cuda.jit('void(float32[:,:])')(atomic_add_float_2)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_add2[1, (4, 8)](ary)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func_wrap = cuda.jit('void(float32[:,:])')(atomic_add_float_2_wrap)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func_wrap[1, (4, 8)](ary_wrap)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_add3 = cuda.jit('void(float32[:,:])')(atomic_add_float_3)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_add3[1, (4, 8)](ary)
numba/cuda/tests/cudapy/test_atomics.py:        if config.ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_atomics.py:            if cuda.runtime.get_version() > (12, 1):
numba/cuda/tests/cudapy/test_atomics.py:                # CUDA 12.2 and above generate a more optimized reduction
numba/cuda/tests/cudapy/test_atomics.py:        cuda_fn = cuda.jit('void(int64[:], float64[:])')(atomic_add_double)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_fn[1, 32](idx, ary)
numba/cuda/tests/cudapy/test_atomics.py:        wrap_fn = cuda.jit('void(int64[:], float64[:])')(atomic_add_double_wrap)
numba/cuda/tests/cudapy/test_atomics.py:        self.assertCorrectFloat64Atomics(cuda_fn)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_fn = cuda.jit('void(float64[:,:])')(atomic_add_double_2)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_fn[1, (4, 8)](ary)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_fn_wrap = cuda.jit('void(float64[:,:])')(atomic_add_double_2_wrap)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_fn_wrap[1, (4, 8)](ary_wrap)
numba/cuda/tests/cudapy/test_atomics.py:        self.assertCorrectFloat64Atomics(cuda_fn)
numba/cuda/tests/cudapy/test_atomics.py:        self.assertCorrectFloat64Atomics(cuda_fn_wrap)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(float64[:,:])')(atomic_add_double_3)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, (4, 8)](ary)
numba/cuda/tests/cudapy/test_atomics.py:        self.assertCorrectFloat64Atomics(cuda_func)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(sig)(atomic_add_double_global)
numba/cuda/tests/cudapy/test_atomics.py:        wrap_cuda_func = cuda.jit(sig)(atomic_add_double_global_wrap)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](idx, ary)
numba/cuda/tests/cudapy/test_atomics.py:        wrap_cuda_func[1, 32](idx, ary_wrap)
numba/cuda/tests/cudapy/test_atomics.py:        self.assertCorrectFloat64Atomics(cuda_func, shared=False)
numba/cuda/tests/cudapy/test_atomics.py:        self.assertCorrectFloat64Atomics(wrap_cuda_func, shared=False)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(sig)(atomic_add_double_global_2)
numba/cuda/tests/cudapy/test_atomics.py:        wrap_cuda_func = cuda.jit(sig)(atomic_add_double_global_2_wrap)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, (4, 8)](ary)
numba/cuda/tests/cudapy/test_atomics.py:        wrap_cuda_func[1, (4, 8)](ary_wrap)
numba/cuda/tests/cudapy/test_atomics.py:        self.assertCorrectFloat64Atomics(cuda_func, shared=False)
numba/cuda/tests/cudapy/test_atomics.py:        self.assertCorrectFloat64Atomics(wrap_cuda_func, shared=False)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(float64[:,:])')(atomic_add_double_global_3)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, (4, 8)](ary)
numba/cuda/tests/cudapy/test_atomics.py:        self.assertCorrectFloat64Atomics(cuda_func, shared=False)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_sub = cuda.jit('void(uint32[:])')(atomic_sub)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_sub[1, 32](ary)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_sub2 = cuda.jit('void(uint32[:,:])')(atomic_sub2)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_sub2[1, (4, 8)](ary)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_sub3 = cuda.jit('void(uint32[:,:])')(atomic_sub3)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_sub3[1, (4, 8)](ary)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_sub_float = cuda.jit('void(float32[:])')(atomic_sub_float)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_sub_float[1, 32](ary)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_sub2 = cuda.jit('void(float32[:,:])')(atomic_sub_float_2)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_sub2[1, (4, 8)](ary)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_sub3 = cuda.jit('void(float32[:,:])')(atomic_sub_float_3)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_sub3[1, (4, 8)](ary)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(int64[:], float64[:])')(atomic_sub_double)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](idx, ary)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(float64[:,:])')(atomic_sub_double_2)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, (4, 8)](ary)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(float64[:,:])')(atomic_sub_double_3)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, (4, 8)](ary)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(sig)(atomic_sub_double_global)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](idx, ary)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(float64[:,:])')(atomic_sub_double_global_2)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, (4, 8)](ary)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(float64[:,:])')(atomic_sub_double_global_3)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, (4, 8)](ary)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(uint32[:], uint32)')(atomic_and)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_and2 = cuda.jit('void(uint32[:,:], uint32)')(atomic_and2)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_and2[1, (4, 8)](ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_and3 = cuda.jit('void(uint32[:,:], uint32)')(atomic_and3)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_and3[1, (4, 8)](ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(sig)(atomic_and_global)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](idx, ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(uint32[:,:], uint32)')(atomic_and_global_2)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, (4, 8)](ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(uint32[:], uint32)')(atomic_or)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_and2 = cuda.jit('void(uint32[:,:], uint32)')(atomic_or2)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_and2[1, (4, 8)](ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_and3 = cuda.jit('void(uint32[:,:], uint32)')(atomic_or3)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_and3[1, (4, 8)](ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(sig)(atomic_or_global)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](idx, ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(uint32[:,:], uint32)')(atomic_or_global_2)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, (4, 8)](ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(uint32[:], uint32)')(atomic_xor)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_xor2 = cuda.jit('void(uint32[:,:], uint32)')(atomic_xor2)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_xor2[1, (4, 8)](ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_xor3 = cuda.jit('void(uint32[:,:], uint32)')(atomic_xor3)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_atomic_xor3[1, (4, 8)](ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(sig)(atomic_xor_global)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](idx, ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(uint32[:,:], uint32)')(atomic_xor_global_2)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, (4, 8)](ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(sig)(func)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[nblocks, blksize](ary, idx, rconst)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(sig)(func)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[nblocks, blksize](idx, ary, rconst)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(sig)(func)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[nblocks, blksize](ary, rconst)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(sig)(func)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[nblocks, blksize](ary, idx, rconst)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(sig)(func)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[nblocks, blksize](idx, ary, rconst)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(sig)(func)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[nblocks, blksize](ary, rconst)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(uint32[:], uint32[:], uint32)')(atomic_exch)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](ary, idx, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(uint32[:,:], uint32)')(atomic_exch2)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, (4, 8)](ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(uint64[:,:], uint64)')(atomic_exch3)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, (4, 8)](ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(sig)(atomic_exch_global)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](idx, ary, rand_const)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(atomic_max)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[32, 32](res, vals)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(float64[:], float64[:,:])')(
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[32, 32](res, vals)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(float64[:], float64[:])')(
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](res, vals)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(atomic_min)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[32, 32](res, vals)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(float64[:], float64[:,:])')(
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[32, 32](res, vals)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(float64[:], float64[:])')(
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](res, vals)
numba/cuda/tests/cudapy/test_atomics.py:    #    cuda.atomic.{min,max}(ary, idx, val)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(float64[:], float64[:,:])')(func)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 1](res, vals)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(float64[:], float64[:,:])')(func)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 1](res, vals)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(sig)(atomic_max_double_shared)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](res, vals)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(sig)(atomic_min_double_shared)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](res, vals)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(cas_func)
numba/cuda/tests/cudapy/test_atomics.py:            cuda_func[10, 10](res, out, ary, fill)
numba/cuda/tests/cudapy/test_atomics.py:            cuda_func[(10, 10), (10, 10)](res, out, ary, fill)
numba/cuda/tests/cudapy/test_atomics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_atomics.py:            x[1] = cuda.atomic.add(x, 0, 1)
numba/cuda/tests/cudapy/test_atomics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_atomics.py:            x[1] = cuda.atomic.max(x, 0, 1)
numba/cuda/tests/cudapy/test_atomics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_atomics.py:            x[1] = cuda.atomic.max(x, 0, 10)
numba/cuda/tests/cudapy/test_atomics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_atomics.py:            x[1] = cuda.atomic.max(x, 0, 1)
numba/cuda/tests/cudapy/test_atomics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_atomics.py:            x[1] = cuda.atomic.max(x, 0, np.nan)
numba/cuda/tests/cudapy/test_atomics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_atomics.py:            x[1] = cuda.atomic.min(x, 0, 11)
numba/cuda/tests/cudapy/test_atomics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_atomics.py:            x[1] = cuda.atomic.min(x, 0, 10)
numba/cuda/tests/cudapy/test_atomics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_atomics.py:            x[1] = cuda.atomic.min(x, 0, 11)
numba/cuda/tests/cudapy/test_atomics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_atomics.py:            x[1] = cuda.atomic.min(x, 0, np.nan)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(atomic_nanmax)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[32, 32](res, vals)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(sig)(atomic_nanmax_double_shared)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](res, vals)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(float64[:], float64[:])')(
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](res, vals)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(atomic_nanmin)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[32, 32](res, vals)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit(sig)(atomic_nanmin_double_shared)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](res, vals)
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func = cuda.jit('void(float64[:], float64[:])')(
numba/cuda/tests/cudapy/test_atomics.py:        cuda_func[1, 32](res, vals)
numba/cuda/tests/cudapy/test_atomics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_atomics.py:            x[1] = cuda.atomic.nanmax(x, 0, 1)
numba/cuda/tests/cudapy/test_atomics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_atomics.py:            x[1] = cuda.atomic.nanmax(x, 0, 10)
numba/cuda/tests/cudapy/test_atomics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_atomics.py:            x[1] = cuda.atomic.nanmax(x, 0, 1)
numba/cuda/tests/cudapy/test_atomics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_atomics.py:            x[1] = cuda.atomic.nanmax(x, 0, np.nan)
numba/cuda/tests/cudapy/test_atomics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_atomics.py:            x[1] = cuda.atomic.nanmin(x, 0, 11)
numba/cuda/tests/cudapy/test_atomics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_atomics.py:            x[1] = cuda.atomic.nanmin(x, 0, 10)
numba/cuda/tests/cudapy/test_atomics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_atomics.py:            x[1] = cuda.atomic.nanmin(x, 0, 11)
numba/cuda/tests/cudapy/test_atomics.py:        @cuda.jit
numba/cuda/tests/cudapy/test_atomics.py:            x[1] = cuda.atomic.nanmin(x, 0, np.nan)
numba/cuda/tests/cudapy/test_caching.py:from numba import cuda
numba/cuda/tests/cudapy/test_caching.py:from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
numba/cuda/tests/cudapy/test_caching.py:                                skip_unless_cc_60, skip_if_cudadevrt_missing,
numba/cuda/tests/cudapy/test_caching.py:@skip_on_cudasim('Simulator does not implement caching')
numba/cuda/tests/cudapy/test_caching.py:class CUDACachingTest(SerialMixin, DispatcherCacheUsecasesTest):
numba/cuda/tests/cudapy/test_caching.py:    modname = "cuda_caching_test_fodder"
numba/cuda/tests/cudapy/test_caching.py:        CUDATestCase.setUp(self)
numba/cuda/tests/cudapy/test_caching.py:        CUDATestCase.tearDown(self)
numba/cuda/tests/cudapy/test_caching.py:    @skip_if_cudadevrt_missing
numba/cuda/tests/cudapy/test_caching.py:    @skip_if_cudadevrt_missing
numba/cuda/tests/cudapy/test_caching.py:        msg = 'Cannot pickle CUDACodeLibrary with linking files'
numba/cuda/tests/cudapy/test_caching.py:            @cuda.jit('void()', cache=True, link=[link])
numba/cuda/tests/cudapy/test_caching.py:@skip_on_cudasim('Simulator does not implement caching')
numba/cuda/tests/cudapy/test_caching.py:class CUDAAndCPUCachingTest(SerialMixin, DispatcherCacheUsecasesTest):
numba/cuda/tests/cudapy/test_caching.py:    modname = "cuda_and_cpu_caching_test_fodder"
numba/cuda/tests/cudapy/test_caching.py:        CUDATestCase.setUp(self)
numba/cuda/tests/cudapy/test_caching.py:        CUDATestCase.tearDown(self)
numba/cuda/tests/cudapy/test_caching.py:    def test_cpu_and_cuda_targets(self):
numba/cuda/tests/cudapy/test_caching.py:        # The same function jitted for CPU and CUDA targets should maintain
numba/cuda/tests/cudapy/test_caching.py:        f_cuda = mod.assign_cuda
numba/cuda/tests/cudapy/test_caching.py:        self.assertPreciseEqual(f_cuda(5), 5)
numba/cuda/tests/cudapy/test_caching.py:        self.check_hits(f_cuda.func, 0, 1)
numba/cuda/tests/cudapy/test_caching.py:        self.assertPreciseEqual(f_cuda(5.5), 5.5)
numba/cuda/tests/cudapy/test_caching.py:        self.check_hits(f_cuda.func, 0, 2)
numba/cuda/tests/cudapy/test_caching.py:    def test_cpu_and_cuda_reuse(self):
numba/cuda/tests/cudapy/test_caching.py:        # Existing cache files for the CPU and CUDA targets are reused.
numba/cuda/tests/cudapy/test_caching.py:        mod.assign_cuda(5)
numba/cuda/tests/cudapy/test_caching.py:        mod.assign_cuda(5.5)
numba/cuda/tests/cudapy/test_caching.py:        self.check_hits(mod.assign_cuda.func, 0, 2)
numba/cuda/tests/cudapy/test_caching.py:        f_cuda = mod2.assign_cuda
numba/cuda/tests/cudapy/test_caching.py:        f_cuda(2)
numba/cuda/tests/cudapy/test_caching.py:        self.check_hits(f_cuda.func, 1, 0)
numba/cuda/tests/cudapy/test_caching.py:        f_cuda(2.5)
numba/cuda/tests/cudapy/test_caching.py:        self.check_hits(f_cuda.func, 2, 0)
numba/cuda/tests/cudapy/test_caching.py:def get_different_cc_gpus():
numba/cuda/tests/cudapy/test_caching.py:    # Find two GPUs with different Compute Capabilities and return them as a
numba/cuda/tests/cudapy/test_caching.py:    # tuple. If two GPUs with distinct Compute Capabilities cannot be found,
numba/cuda/tests/cudapy/test_caching.py:    first_gpu = cuda.gpus[0]
numba/cuda/tests/cudapy/test_caching.py:    with first_gpu:
numba/cuda/tests/cudapy/test_caching.py:        first_cc = cuda.current_context().device.compute_capability
numba/cuda/tests/cudapy/test_caching.py:    for gpu in cuda.gpus[1:]:
numba/cuda/tests/cudapy/test_caching.py:        with gpu:
numba/cuda/tests/cudapy/test_caching.py:            cc = cuda.current_context().device.compute_capability
numba/cuda/tests/cudapy/test_caching.py:                return (first_gpu, gpu)
numba/cuda/tests/cudapy/test_caching.py:@skip_on_cudasim('Simulator does not implement caching')
numba/cuda/tests/cudapy/test_caching.py:    modname = "cuda_multi_cc_caching_test_fodder"
numba/cuda/tests/cudapy/test_caching.py:        CUDATestCase.setUp(self)
numba/cuda/tests/cudapy/test_caching.py:        CUDATestCase.tearDown(self)
numba/cuda/tests/cudapy/test_caching.py:        gpus = get_different_cc_gpus()
numba/cuda/tests/cudapy/test_caching.py:        if not gpus:
numba/cuda/tests/cudapy/test_caching.py:        # Step 1. Populate the cache with the first GPU
numba/cuda/tests/cudapy/test_caching.py:        with gpus[0]:
numba/cuda/tests/cudapy/test_caching.py:        # Step 2. Run with the second GPU - under present behaviour this
numba/cuda/tests/cudapy/test_caching.py:        with gpus[1]:
numba/cuda/tests/cudapy/test_caching.py:        # Step 3. Run in a separate module with the second GPU - this populates
numba/cuda/tests/cudapy/test_caching.py:        with gpus[1]:
numba/cuda/tests/cudapy/test_caching.py:        # the cached version containing a cubin for GPU 1. There will be no
numba/cuda/tests/cudapy/test_caching.py:        # cubin for GPU 0, so when we try to use it the PTX must be generated.
numba/cuda/tests/cudapy/test_caching.py:        # Step 4. Run with GPU 1 and get a cache hit, loading the cache created
numba/cuda/tests/cudapy/test_caching.py:        with gpus[1]:
numba/cuda/tests/cudapy/test_caching.py:        # Step 5. Run with GPU 0 using the module from Step 4, to force PTX
numba/cuda/tests/cudapy/test_caching.py:        with gpus[0]:
numba/cuda/tests/cudapy/test_caching.py:    config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
numba/cuda/tests/cudapy/test_caching.py:    config.CUDA_WARN_ON_IMPLICIT_COPY = 0
numba/cuda/tests/cudapy/test_caching.py:@skip_on_cudasim('Simulator does not implement caching')
numba/cuda/tests/cudapy/test_caching.py:    modname = "cuda_mp_caching_test_fodder"
numba/cuda/tests/cudapy/test_caching.py:        CUDATestCase.setUp(self)
numba/cuda/tests/cudapy/test_caching.py:        CUDATestCase.tearDown(self)
numba/cuda/tests/cudapy/test_caching.py:@skip_on_cudasim('Simulator does not implement the CUDACodeLibrary')
numba/cuda/tests/cudapy/test_caching.py:class TestCUDACodeLibrary(CUDATestCase):
numba/cuda/tests/cudapy/test_caching.py:    # For tests of miscellaneous CUDACodeLibrary behaviour that we wish to
numba/cuda/tests/cudapy/test_caching.py:        # The CUDA codegen failes to import under the simulator, so we cannot
numba/cuda/tests/cudapy/test_caching.py:        from numba.cuda.codegen import CUDACodeLibrary
numba/cuda/tests/cudapy/test_caching.py:        cl = CUDACodeLibrary(codegen, name)
numba/cuda/tests/cudapy/test_montecarlo.py:from numba import cuda
numba/cuda/tests/cudapy/test_montecarlo.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_montecarlo.py:class TestCudaMonteCarlo(CUDATestCase):
numba/cuda/tests/cudapy/test_montecarlo.py:        @cuda.jit(
numba/cuda/tests/cudapy/test_montecarlo.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_retrieve_autoconverted_arrays.py:from numba import cuda
numba/cuda/tests/cudapy/test_retrieve_autoconverted_arrays.py:from numba.cuda.args import wrap_arg
numba/cuda/tests/cudapy/test_retrieve_autoconverted_arrays.py:from numba.cuda.testing import CUDATestCase
numba/cuda/tests/cudapy/test_retrieve_autoconverted_arrays.py:        return ty, wrap_arg(val, default=cuda.In)
numba/cuda/tests/cudapy/test_retrieve_autoconverted_arrays.py:class TestRetrieveAutoconvertedArrays(CUDATestCase):
numba/cuda/tests/cudapy/test_retrieve_autoconverted_arrays.py:        self.set_array_to_three = cuda.jit(set_array_to_three)
numba/cuda/tests/cudapy/test_retrieve_autoconverted_arrays.py:        self.set_array_to_three_nocopy = nocopy(cuda.jit(set_array_to_three))
numba/cuda/tests/cudapy/test_retrieve_autoconverted_arrays.py:        self.set_record_to_three = cuda.jit(set_record_to_three)
numba/cuda/tests/cudapy/test_retrieve_autoconverted_arrays.py:        self.set_record_to_three_nocopy = nocopy(cuda.jit(set_record_to_three))
numba/cuda/tests/cudapy/test_retrieve_autoconverted_arrays.py:        self.set_array_to_three[1, 1](cuda.InOut(host_arr))
numba/cuda/tests/cudapy/test_retrieve_autoconverted_arrays.py:        self.set_array_to_three[1, 1](cuda.In(host_arr))
numba/cuda/tests/cudapy/test_retrieve_autoconverted_arrays.py:        self.set_record_to_three[1, 1](cuda.In(host_rec))
numba/cuda/tests/cudapy/test_retrieve_autoconverted_arrays.py:        self.set_record_to_three[1, 1](cuda.InOut(host_rec))
numba/cuda/tests/cudapy/test_lang.py:from numba import cuda, float64
numba/cuda/tests/cudapy/test_lang.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_lang.py:class TestLang(CUDATestCase):
numba/cuda/tests/cudapy/test_lang.py:        @cuda.jit("void(float64[:])")
numba/cuda/tests/cudapy/test_lang.py:        @cuda.jit("void(float64[:])")
numba/cuda/tests/cudapy/test_lang.py:        Ensure that typing and lowering of CUDA kernel API primitives works in
numba/cuda/tests/cudapy/test_lang.py:        @cuda.jit("void(float64[:,:])")
numba/cuda/tests/cudapy/test_lang.py:        def cuda_kernel_api_in_multiple_blocks(ary):
numba/cuda/tests/cudapy/test_lang.py:                tx = cuda.threadIdx.x
numba/cuda/tests/cudapy/test_lang.py:                ty = cuda.threadIdx.y
numba/cuda/tests/cudapy/test_lang.py:            sm = cuda.shared.array((2, 3), float64)
numba/cuda/tests/cudapy/test_lang.py:        cuda_kernel_api_in_multiple_blocks[1, (2, 3)](a)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:from numba import cuda
numba/cuda/tests/cudapy/test_cuda_array_interface.py:from numba.cuda.cudadrv import driver
numba/cuda/tests/cudapy/test_cuda_array_interface.py:from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
numba/cuda/tests/cudapy/test_cuda_array_interface.py:from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
numba/cuda/tests/cudapy/test_cuda_array_interface.py:@skip_on_cudasim('CUDA Array Interface is not supported in the simulator')
numba/cuda/tests/cudapy/test_cuda_array_interface.py:class TestCudaArrayInterface(ContextResettingTestCase):
numba/cuda/tests/cudapy/test_cuda_array_interface.py:    def test_as_cuda_array(self):
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        self.assertFalse(cuda.is_cuda_array(h_arr))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        d_arr = cuda.to_device(h_arr)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        self.assertTrue(cuda.is_cuda_array(d_arr))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        self.assertTrue(cuda.is_cuda_array(my_arr))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        wrapped = cuda.as_cuda_array(my_arr)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        self.assertTrue(cuda.is_cuda_array(wrapped))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        ctx = cuda.current_context()
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        d_arr = cuda.to_device(np.arange(100))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        cvted = cuda.as_cuda_array(d_arr)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        d_arr = cuda.to_device(h_arr)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        wrapped = cuda.as_cuda_array(my_arr)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        @cuda.jit
numba/cuda/tests/cudapy/test_cuda_array_interface.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        @vectorize(['f8(f8, f8)'], target='cuda')
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        arr = ForeignArray(cuda.to_device(h_arr))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        out = ForeignArray(cuda.device_array(h_arr.shape))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        @guvectorize(['(f8, f8, f8[:])'], '(),()->()', target='cuda')
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        arr = ForeignArray(cuda.to_device(h_arr))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        out = ForeignArray(cuda.device_array(h_arr.shape))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        c_arr = cuda.to_device(h_arr)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        arr = cuda.as_cuda_array(c_arr)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        arr_strided = cuda.as_cuda_array(c_arr[::2])
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        arr[:] = cuda.to_device(h_arr)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        c_arr = cuda.to_device(h_arr)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        c_ai = c_arr.__cuda_array_interface__
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        c_ai_sliced = c_arr[::-1].__cuda_array_interface__
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        c_arr = cuda.to_device(h_arr)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        c_arr = cuda.to_device(h_arr)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        c_mask = cuda.to_device(h_mask)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        # Manually create a masked CUDA Array Interface dictionary
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        masked_cuda_array_interface = c_arr.__cuda_array_interface__.copy()
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        masked_cuda_array_interface['mask'] = c_mask
numba/cuda/tests/cudapy/test_cuda_array_interface.py:            cuda.from_cuda_array_interface(masked_cuda_array_interface)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        c_arr = cuda.device_array(0)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        self.assertEqual(c_arr.__cuda_array_interface__['data'][0], 0)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        @cuda.jit
numba/cuda/tests/cudapy/test_cuda_array_interface.py:            x = cuda.grid(1)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        c_arr = cuda.device_array((2, 3, 4))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        self.assertEqual(c_arr.__cuda_array_interface__['strides'], None)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        self.assertNotEqual(c_arr.__cuda_array_interface__['strides'], None)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        devarray = cuda.to_device(hostarray)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        face = devarray.__cuda_array_interface__
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        got = cuda.from_cuda_array_interface(face).copy_to_host()
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        got = cuda.from_cuda_array_interface(face).copy_to_host()
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        c_arr = cuda.device_array(10)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        self.assertIsNone(c_arr.__cuda_array_interface__['stream'])
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        mapped_arr = cuda.mapped_array(10)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        self.assertIsNone(mapped_arr.__cuda_array_interface__['stream'])
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        managed_arr = cuda.managed_array(10)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        self.assertIsNone(managed_arr.__cuda_array_interface__['stream'])
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        s = cuda.stream()
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        c_arr = cuda.device_array(10, stream=s)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        cai_stream = c_arr.__cuda_array_interface__['stream']
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        s = cuda.stream()
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        mapped_arr = cuda.mapped_array(10, stream=s)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        cai_stream = mapped_arr.__cuda_array_interface__['stream']
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        s = cuda.stream()
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        managed_arr = cuda.managed_array(10, stream=s)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        cai_stream = managed_arr.__cuda_array_interface__['stream']
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        f_arr = ForeignArray(cuda.device_array(10))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        c_arr = cuda.as_cuda_array(f_arr)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        s = cuda.stream()
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        f_arr = ForeignArray(cuda.device_array(10, stream=s))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        c_arr = cuda.as_cuda_array(f_arr)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        f_arr = ForeignArray(cuda.device_array(10))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        with patch.object(cuda.cudadrv.driver.Stream, 'synchronize',
numba/cuda/tests/cudapy/test_cuda_array_interface.py:            cuda.as_cuda_array(f_arr)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        s = cuda.stream()
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        f_arr = ForeignArray(cuda.device_array(10, stream=s))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        with patch.object(cuda.cudadrv.driver.Stream, 'synchronize',
numba/cuda/tests/cudapy/test_cuda_array_interface.py:            cuda.as_cuda_array(f_arr)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        s = cuda.stream()
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        f_arr = ForeignArray(cuda.device_array(10, stream=s))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        with override_config('CUDA_ARRAY_INTERFACE_SYNC', False):
numba/cuda/tests/cudapy/test_cuda_array_interface.py:            with patch.object(cuda.cudadrv.driver.Stream, 'synchronize',
numba/cuda/tests/cudapy/test_cuda_array_interface.py:                cuda.as_cuda_array(f_arr)
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        f_arr = ForeignArray(cuda.device_array(10))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        @cuda.jit
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        with patch.object(cuda.cudadrv.driver.Stream, 'synchronize',
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        s = cuda.stream()
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        f_arr = ForeignArray(cuda.device_array(10, stream=s))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        @cuda.jit
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        with patch.object(cuda.cudadrv.driver.Stream, 'synchronize',
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        s1 = cuda.stream()
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        s2 = cuda.stream()
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        f_arr1 = ForeignArray(cuda.device_array(10, stream=s1))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        f_arr2 = ForeignArray(cuda.device_array(10, stream=s2))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        @cuda.jit
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        with patch.object(cuda.cudadrv.driver.Stream, 'synchronize',
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        s1 = cuda.stream()
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        s2 = cuda.stream()
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        f_arr1 = ForeignArray(cuda.device_array(10, stream=s1))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        f_arr2 = ForeignArray(cuda.device_array(10, stream=s2))
numba/cuda/tests/cudapy/test_cuda_array_interface.py:        with override_config('CUDA_ARRAY_INTERFACE_SYNC', False):
numba/cuda/tests/cudapy/test_cuda_array_interface.py:            @cuda.jit
numba/cuda/tests/cudapy/test_cuda_array_interface.py:            with patch.object(cuda.cudadrv.driver.Stream, 'synchronize',
numba/cuda/tests/cudapy/test_operator.py:from numba.cuda.testing import (unittest, CUDATestCase, skip_unless_cc_53,
numba/cuda/tests/cudapy/test_operator.py:                                skip_on_cudasim)
numba/cuda/tests/cudapy/test_operator.py:from numba import cuda
numba/cuda/tests/cudapy/test_operator.py:from numba.cuda import compile_ptx
numba/cuda/tests/cudapy/test_operator.py:@cuda.jit('b1(f2, f2)', device=True)
numba/cuda/tests/cudapy/test_operator.py:@cuda.jit('b1(f2, f2)', device=True)
numba/cuda/tests/cudapy/test_operator.py:class TestOperatorModule(CUDATestCase):
numba/cuda/tests/cudapy/test_operator.py:    Test if operator module is supported by the CUDA target.
numba/cuda/tests/cudapy/test_operator.py:        @cuda.jit
numba/cuda/tests/cudapy/test_operator.py:                kernel = cuda.jit("void(f2[:], f2, f2)")(fn)
numba/cuda/tests/cudapy/test_operator.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_operator.py:                kernel = cuda.jit(fn)
numba/cuda/tests/cudapy/test_operator.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_operator.py:                kernel = cuda.jit("void(f2[:], f2)")(fn)
numba/cuda/tests/cudapy/test_operator.py:                kernel = cuda.jit("void(f2[:], f2)")(fn)
numba/cuda/tests/cudapy/test_operator.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_operator.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_operator.py:                kernel = cuda.jit("void(b1[:], f2, f2)")(fn)
numba/cuda/tests/cudapy/test_operator.py:                kernel = cuda.jit(fn)
numba/cuda/tests/cudapy/test_operator.py:                compiled = cuda.jit("void(b1[:], f2, f2, f2)")(fn)
numba/cuda/tests/cudapy/test_operator.py:                compiled = cuda.jit("void(b1[:], f2, f2, f2)")(fn)
numba/cuda/tests/cudapy/test_operator.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_operator.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_operator.py:    @skip_on_cudasim('Compilation unsupported in the simulator')
numba/cuda/tests/cudapy/test_sync.py:from numba import cuda, int32, float32
numba/cuda/tests/cudapy/test_sync.py:from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
numba/cuda/tests/cudapy/test_sync.py:from numba.core.config import ENABLE_CUDASIM
numba/cuda/tests/cudapy/test_sync.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_sync.py:    cuda.syncthreads()
numba/cuda/tests/cudapy/test_sync.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_sync.py:    cuda.syncwarp()
numba/cuda/tests/cudapy/test_sync.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_sync.py:    cuda.syncwarp(0xFFFF)
numba/cuda/tests/cudapy/test_sync.py:    sm = cuda.shared.array(32, int32)
numba/cuda/tests/cudapy/test_sync.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_sync.py:    cuda.syncwarp()
numba/cuda/tests/cudapy/test_sync.py:        cuda.syncwarp(0xFFFF)
numba/cuda/tests/cudapy/test_sync.py:        cuda.syncwarp(0xFF)
numba/cuda/tests/cudapy/test_sync.py:        cuda.syncwarp(0xF)
numba/cuda/tests/cudapy/test_sync.py:        cuda.syncwarp(0x3)
numba/cuda/tests/cudapy/test_sync.py:    sm = cuda.shared.array(N, int32)
numba/cuda/tests/cudapy/test_sync.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_sync.py:    cuda.syncthreads()
numba/cuda/tests/cudapy/test_sync.py:    i, j = cuda.grid(2)
numba/cuda/tests/cudapy/test_sync.py:    sm = cuda.shared.array((10, 20), float32)
numba/cuda/tests/cudapy/test_sync.py:    cuda.syncthreads()
numba/cuda/tests/cudapy/test_sync.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_sync.py:    sm = cuda.shared.array(0, float32)
numba/cuda/tests/cudapy/test_sync.py:    cuda.syncthreads()
numba/cuda/tests/cudapy/test_sync.py:    cuda.threadfence()
numba/cuda/tests/cudapy/test_sync.py:    cuda.threadfence_block()
numba/cuda/tests/cudapy/test_sync.py:    cuda.threadfence_system()
numba/cuda/tests/cudapy/test_sync.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_sync.py:    ary_out[i] = cuda.syncthreads_count(ary_in[i])
numba/cuda/tests/cudapy/test_sync.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_sync.py:    ary_out[i] = cuda.syncthreads_and(ary_in[i])
numba/cuda/tests/cudapy/test_sync.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_sync.py:    ary_out[i] = cuda.syncthreads_or(ary_in[i])
numba/cuda/tests/cudapy/test_sync.py:    if ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_sync.py:        return cuda.get_current_device().compute_capability >= cc
numba/cuda/tests/cudapy/test_sync.py:class TestCudaSync(CUDATestCase):
numba/cuda/tests/cudapy/test_sync.py:        compiled = cuda.jit("void(int32[::1])")(kernel)
numba/cuda/tests/cudapy/test_sync.py:    @skip_on_cudasim("syncwarp not implemented on cudasim")
numba/cuda/tests/cudapy/test_sync.py:    @skip_on_cudasim("syncwarp not implemented on cudasim")
numba/cuda/tests/cudapy/test_sync.py:    @skip_on_cudasim("syncwarp not implemented on cudasim")
numba/cuda/tests/cudapy/test_sync.py:        compiled = cuda.jit("void(int32[::1])")(coop_syncwarp)
numba/cuda/tests/cudapy/test_sync.py:        compiled = cuda.jit("void(int32[::1])")(simple_smem)
numba/cuda/tests/cudapy/test_sync.py:        compiled = cuda.jit("void(float32[:,::1])")(coop_smem2d)
numba/cuda/tests/cudapy/test_sync.py:        compiled = cuda.jit("void(float32[::1])")(dyn_shared_memory)
numba/cuda/tests/cudapy/test_sync.py:        compiled = cuda.jit(sig)(use_threadfence)
numba/cuda/tests/cudapy/test_sync.py:        if not ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_sync.py:        compiled = cuda.jit(sig)(use_threadfence_block)
numba/cuda/tests/cudapy/test_sync.py:        if not ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_sync.py:        compiled = cuda.jit(sig)(use_threadfence_system)
numba/cuda/tests/cudapy/test_sync.py:        if not ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_sync.py:        compiled = cuda.jit(use_syncthreads_count)
numba/cuda/tests/cudapy/test_sync.py:        compiled = cuda.jit(use_syncthreads_and)
numba/cuda/tests/cudapy/test_sync.py:        compiled = cuda.jit(use_syncthreads_or)
numba/cuda/tests/cudapy/test_iterators.py:from numba import cuda
numba/cuda/tests/cudapy/test_iterators.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_iterators.py:class TestIterators(CUDATestCase):
numba/cuda/tests/cudapy/test_iterators.py:        @cuda.jit
numba/cuda/tests/cudapy/test_iterators.py:        @cuda.jit
numba/cuda/tests/cudapy/test_iterators.py:        @cuda.jit
numba/cuda/tests/cudapy/test_iterators.py:        @cuda.jit
numba/cuda/tests/cudapy/test_recursion.py:from numba import cuda
numba/cuda/tests/cudapy/test_recursion.py:from numba.cuda.testing import CUDATestCase, skip_on_cudasim
numba/cuda/tests/cudapy/test_recursion.py:class TestSelfRecursion(CUDATestCase):
numba/cuda/tests/cudapy/test_recursion.py:        from numba.cuda.tests.cudapy import recursion_usecases
numba/cuda/tests/cudapy/test_recursion.py:        @cuda.jit
numba/cuda/tests/cudapy/test_recursion.py:    @skip_on_cudasim('Simulator does not compile')
numba/cuda/tests/cudapy/test_recursion.py:            @cuda.jit('void()')
numba/cuda/tests/cudapy/test_recursion.py:        @cuda.jit
numba/cuda/tests/cudapy/test_recursion.py:        cfunc = self.mod.make_optional_return_case(cuda.jit)
numba/cuda/tests/cudapy/test_recursion.py:        @cuda.jit
numba/cuda/tests/cudapy/test_recursion.py:    @skip_on_cudasim('Recursion handled because simulator does not compile')
numba/cuda/tests/cudapy/test_recursion.py:        cfunc = self.mod.make_growing_tuple_case(cuda.jit)
numba/cuda/tests/cudapy/test_recursion.py:            @cuda.jit('void()')
numba/cuda/tests/cudapy/test_cffi.py:from numba import cuda, types
numba/cuda/tests/cudapy/test_cffi.py:from numba.cuda.testing import (skip_on_cudasim, test_data_dir, unittest,
numba/cuda/tests/cudapy/test_cffi.py:                                CUDATestCase)
numba/cuda/tests/cudapy/test_cffi.py:@skip_on_cudasim('Simulator does not support linking')
numba/cuda/tests/cudapy/test_cffi.py:class TestCFFI(CUDATestCase):
numba/cuda/tests/cudapy/test_cffi.py:        array_mutator = cuda.declare_device('array_mutator', sig)
numba/cuda/tests/cudapy/test_cffi.py:        @cuda.jit(link=[link])
numba/cuda/tests/cudapy/test_py2_div_issue.py:from numba import cuda, float32, int32, void
numba/cuda/tests/cudapy/test_py2_div_issue.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_py2_div_issue.py:class TestCudaPy2Div(CUDATestCase):
numba/cuda/tests/cudapy/test_py2_div_issue.py:        @cuda.jit(void(float32[:], float32[:], float32[:], int32))
numba/cuda/tests/cudapy/test_py2_div_issue.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_array.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_array.py:from numba.cuda.testing import skip_on_cudasim, skip_unless_cudasim
numba/cuda/tests/cudapy/test_array.py:from numba import config, cuda
numba/cuda/tests/cudapy/test_array.py:if config.ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_array.py:    ARRAY_LIKE_FUNCTIONS = (cuda.device_array_like, cuda.pinned_array_like)
numba/cuda/tests/cudapy/test_array.py:    ARRAY_LIKE_FUNCTIONS = (cuda.device_array_like, cuda.mapped_array_like,
numba/cuda/tests/cudapy/test_array.py:                            cuda.pinned_array_like)
numba/cuda/tests/cudapy/test_array.py:class TestCudaArray(CUDATestCase):
numba/cuda/tests/cudapy/test_array.py:    def test_gpu_array_zero_length(self):
numba/cuda/tests/cudapy/test_array.py:        dx = cuda.to_device(x)
numba/cuda/tests/cudapy/test_array.py:        shape1 = cuda.device_array(()).shape
numba/cuda/tests/cudapy/test_array.py:        shape2 = cuda.device_array_like(np.ndarray(())).shape
numba/cuda/tests/cudapy/test_array.py:    def test_gpu_array_strided(self):
numba/cuda/tests/cudapy/test_array.py:        @cuda.jit('void(double[:])')
numba/cuda/tests/cudapy/test_array.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_array.py:    def test_gpu_array_interleaved(self):
numba/cuda/tests/cudapy/test_array.py:        @cuda.jit('void(double[:], double[:])')
numba/cuda/tests/cudapy/test_array.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_array.py:            cuda.devicearray.auto_device(y)
numba/cuda/tests/cudapy/test_array.py:        d, _ = cuda.devicearray.auto_device(2)
numba/cuda/tests/cudapy/test_array.py:        d_a = cuda.device_array(10, order='C')
numba/cuda/tests/cudapy/test_array.py:        d_a = cuda.device_array((10, 12), order='C')
numba/cuda/tests/cudapy/test_array.py:        d_a = cuda.device_array((10, 12), order='C')
numba/cuda/tests/cudapy/test_array.py:        d_a = cuda.device_array((10, 12, 14), order='C')
numba/cuda/tests/cudapy/test_array.py:        d_a = cuda.device_array(10, order='F')
numba/cuda/tests/cudapy/test_array.py:        d_a = cuda.device_array((10, 12), order='F')
numba/cuda/tests/cudapy/test_array.py:        d_a = cuda.device_array((10, 12), order='F')
numba/cuda/tests/cudapy/test_array.py:        d_a = cuda.device_array((10, 12, 14), order='F')
numba/cuda/tests/cudapy/test_array.py:        d_view = cuda.device_array(shape)[::2]
numba/cuda/tests/cudapy/test_array.py:        d_view = cuda.device_array(shape, order='F')[::2]
numba/cuda/tests/cudapy/test_array.py:        d_view = cuda.device_array(shape)[::2, ::2]
numba/cuda/tests/cudapy/test_array.py:        d_view = cuda.device_array(shape, order='F')[::2, ::2]
numba/cuda/tests/cudapy/test_array.py:    @skip_on_cudasim('Numba and NumPy stride semantics differ for transpose')
numba/cuda/tests/cudapy/test_array.py:        d_view = cuda.device_array(shape)[::2, ::2].T
numba/cuda/tests/cudapy/test_array.py:    @skip_unless_cudasim('Numba and NumPy stride semantics differ for '
numba/cuda/tests/cudapy/test_array.py:        d_view = cuda.device_array(shape)[::2, ::2].T
numba/cuda/tests/cudapy/test_array.py:                # CUDA device (See issue #4974). Here we can compare strides
numba/cuda/tests/cudapy/test_array.py:        d_view = cuda.device_array(shape, order='F')[::2, ::2].T
numba/cuda/tests/cudapy/test_array.py:    @skip_on_cudasim('Kernel overloads not created in the simulator')
numba/cuda/tests/cudapy/test_array.py:        # CUDA Device arrays were reported as always being typed with 'A' order
numba/cuda/tests/cudapy/test_array.py:        @cuda.jit
numba/cuda/tests/cudapy/test_array.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_array.py:        d_a = cuda.to_device(a)
numba/cuda/tests/cudapy/test_gufunc_scheduling.py:from numba.cuda.deviceufunc import GUFuncEngine
numba/cuda/tests/cudapy/test_blackscholes.py:from numba import cuda, double, void
numba/cuda/tests/cudapy/test_blackscholes.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_blackscholes.py:class TestBlackScholes(CUDATestCase):
numba/cuda/tests/cudapy/test_blackscholes.py:        @cuda.jit(double(double), device=True, inline=True)
numba/cuda/tests/cudapy/test_blackscholes.py:        def cnd_cuda(d):
numba/cuda/tests/cudapy/test_blackscholes.py:        @cuda.jit(void(double[:], double[:], double[:], double[:], double[:],
numba/cuda/tests/cudapy/test_blackscholes.py:        def black_scholes_cuda(callResult, putResult, S, X, T, R, V):
numba/cuda/tests/cudapy/test_blackscholes.py:            i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
numba/cuda/tests/cudapy/test_blackscholes.py:            cndd1 = cnd_cuda(d1)
numba/cuda/tests/cudapy/test_blackscholes.py:            cndd2 = cnd_cuda(d2)
numba/cuda/tests/cudapy/test_blackscholes.py:        stream = cuda.stream()
numba/cuda/tests/cudapy/test_blackscholes.py:        d_callResult = cuda.to_device(callResultNumba, stream)
numba/cuda/tests/cudapy/test_blackscholes.py:        d_putResult = cuda.to_device(putResultNumba, stream)
numba/cuda/tests/cudapy/test_blackscholes.py:        d_stockPrice = cuda.to_device(stockPrice, stream)
numba/cuda/tests/cudapy/test_blackscholes.py:        d_optionStrike = cuda.to_device(optionStrike, stream)
numba/cuda/tests/cudapy/test_blackscholes.py:        d_optionYears = cuda.to_device(optionYears, stream)
numba/cuda/tests/cudapy/test_blackscholes.py:            black_scholes_cuda[griddim, blockdim, stream](
numba/cuda/tests/cudapy/test_warning.py:from numba import cuda
numba/cuda/tests/cudapy/test_warning.py:from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
numba/cuda/tests/cudapy/test_warning.py:@skip_on_cudasim('cudasim does not raise performance warnings')
numba/cuda/tests/cudapy/test_warning.py:class TestWarnings(CUDATestCase):
numba/cuda/tests/cudapy/test_warning.py:        @cuda.jit
numba/cuda/tests/cudapy/test_warning.py:        with override_config('CUDA_LOW_OCCUPANCY_WARNINGS', 1):
numba/cuda/tests/cudapy/test_warning.py:        @cuda.jit
numba/cuda/tests/cudapy/test_warning.py:        with override_config('CUDA_LOW_OCCUPANCY_WARNINGS', 1):
numba/cuda/tests/cudapy/test_warning.py:        @cuda.jit
numba/cuda/tests/cudapy/test_warning.py:        with override_config('CUDA_WARN_ON_IMPLICIT_COPY', 1):
numba/cuda/tests/cudapy/test_warning.py:        self.assertIn('Host array used in CUDA kernel will incur',
numba/cuda/tests/cudapy/test_warning.py:        @cuda.jit
numba/cuda/tests/cudapy/test_warning.py:        ary = cuda.pinned_array(N, dtype=np.float32)
numba/cuda/tests/cudapy/test_warning.py:        with override_config('CUDA_WARN_ON_IMPLICIT_COPY', 1):
numba/cuda/tests/cudapy/test_warning.py:        self.assertIn('Host array used in CUDA kernel will incur',
numba/cuda/tests/cudapy/test_warning.py:        @cuda.jit
numba/cuda/tests/cudapy/test_warning.py:        ary = cuda.mapped_array(N, dtype=np.float32)
numba/cuda/tests/cudapy/test_warning.py:        with override_config('CUDA_WARN_ON_IMPLICIT_COPY', 1):
numba/cuda/tests/cudapy/test_warning.py:        @cuda.jit
numba/cuda/tests/cudapy/test_warning.py:        ary = cuda.managed_array(N, dtype=np.float32)
numba/cuda/tests/cudapy/test_warning.py:        with override_config('CUDA_WARN_ON_IMPLICIT_COPY', 1):
numba/cuda/tests/cudapy/test_warning.py:        @cuda.jit
numba/cuda/tests/cudapy/test_warning.py:        ary = cuda.device_array(N, dtype=np.float32)
numba/cuda/tests/cudapy/test_warning.py:        with override_config('CUDA_WARN_ON_IMPLICIT_COPY', 1):
numba/cuda/tests/cudapy/test_warning.py:            cuda.jit(debug=True, opt=True)
numba/cuda/tests/cudapy/test_warning.py:        self.assertIn('not supported by CUDA', str(w[0].message))
numba/cuda/tests/cudapy/test_warning.py:            cuda.jit(debug=True)
numba/cuda/tests/cudapy/test_warning.py:        self.assertIn('not supported by CUDA', str(w[0].message))
numba/cuda/tests/cudapy/test_warning.py:            cuda.jit(debug=True, opt=False)
numba/cuda/tests/cudapy/test_warning.py:            cuda.jit()
numba/cuda/tests/cudapy/test_userexc.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_userexc.py:from numba import cuda
numba/cuda/tests/cudapy/test_userexc.py:class TestUserExc(CUDATestCase):
numba/cuda/tests/cudapy/test_userexc.py:        @cuda.jit("void(int32)", debug=True)
numba/cuda/tests/cudapy/test_userexc.py:        if not config.ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_userexc.py:        if not config.ENABLE_CUDASIM:
numba/cuda/tests/cudapy/test_transpose.py:from numba import cuda
numba/cuda/tests/cudapy/test_transpose.py:from numba.cuda.kernels.transpose import transpose
numba/cuda/tests/cudapy/test_transpose.py:from numba.cuda.testing import unittest
numba/cuda/tests/cudapy/test_transpose.py:from numba.cuda.testing import skip_on_cudasim, CUDATestCase
numba/cuda/tests/cudapy/test_transpose.py:@skip_on_cudasim('Device Array API unsupported in the simulator')
numba/cuda/tests/cudapy/test_transpose.py:class TestTranspose(CUDATestCase):
numba/cuda/tests/cudapy/test_transpose.py:                dx = cuda.to_device(x)
numba/cuda/tests/cudapy/test_transpose.py:                dy = cuda.cudadrv.devicearray.from_array_like(y)
numba/cuda/tests/cudapy/test_transpose.py:                d_arr = cuda.to_device(arr)
numba/cuda/tests/cudapy/test_transpose.py:                d_transposed = cuda.device_array_like(transposed)
numba/cuda/tests/cudapy/test_transpose.py:                d_arr = cuda.to_device(arr)
numba/cuda/tests/cudapy/test_transpose.py:                d_transposed = cuda.device_array_like(transposed)
numba/cuda/tests/cudapy/test_transpose.py:        d_a = cuda.to_device(a)
numba/cuda/tests/cudapy/test_forall.py:from numba import cuda
numba/cuda/tests/cudapy/test_forall.py:from numba.cuda.testing import CUDATestCase
numba/cuda/tests/cudapy/test_forall.py:@cuda.jit
numba/cuda/tests/cudapy/test_forall.py:    i = cuda.grid(1)
numba/cuda/tests/cudapy/test_forall.py:class TestForAll(CUDATestCase):
numba/cuda/tests/cudapy/test_forall.py:        @cuda.jit("void(float32, float32[:], float32[:])")
numba/cuda/tests/cudapy/test_forall.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_frexp_ldexp.py:from numba import cuda
numba/cuda/tests/cudapy/test_frexp_ldexp.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudapy/test_frexp_ldexp.py:class TestCudaFrexpLdexp(CUDATestCase):
numba/cuda/tests/cudapy/test_frexp_ldexp.py:        compiled = cuda.jit(void(nbtype[:], int32[:], nbtype))(simple_frexp)
numba/cuda/tests/cudapy/test_frexp_ldexp.py:        compiled = cuda.jit(void(nbtype[:], nbtype, int32))(simple_ldexp)
numba/cuda/tests/cudapy/__init__.py:from numba.cuda.testing import ensure_supported_ccs_initialized
numba/cuda/tests/cudapy/test_ufuncs.py:from numba import config, cuda, types
numba/cuda/tests/cudapy/test_ufuncs.py:# This test would also be a CUDATestCase, but to avoid a confusing and
numba/cuda/tests/cudapy/test_ufuncs.py:# global state, we implement the necessary parts of CUDATestCase within this
numba/cuda/tests/cudapy/test_ufuncs.py:# - Disabling CUDA performance warnings for the duration of tests.
numba/cuda/tests/cudapy/test_ufuncs.py:        # some here for testing with CUDA.
numba/cuda/tests/cudapy/test_ufuncs.py:        self._low_occupancy_warnings = config.CUDA_LOW_OCCUPANCY_WARNINGS
numba/cuda/tests/cudapy/test_ufuncs.py:        self._warn_on_implicit_copy = config.CUDA_WARN_ON_IMPLICIT_COPY
numba/cuda/tests/cudapy/test_ufuncs.py:        # Disable warnings about low gpu utilization in the test suite
numba/cuda/tests/cudapy/test_ufuncs.py:        config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
numba/cuda/tests/cudapy/test_ufuncs.py:        config.CUDA_WARN_ON_IMPLICIT_COPY = 0
numba/cuda/tests/cudapy/test_ufuncs.py:        config.CUDA_LOW_OCCUPANCY_WARNINGS = self._low_occupancy_warnings
numba/cuda/tests/cudapy/test_ufuncs.py:        config.CUDA_WARN_ON_IMPLICIT_COPY = self._warn_on_implicit_copy
numba/cuda/tests/cudapy/test_ufuncs.py:        return cuda.jit(args)(pyfunc)[1, 1]
numba/cuda/tests/cudapy/test_fastmath.py:from numba import cuda, float32
numba/cuda/tests/cudapy/test_fastmath.py:from numba.cuda.compiler import compile_ptx_for_current_device, compile_ptx
numba/cuda/tests/cudapy/test_fastmath.py:from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
numba/cuda/tests/cudapy/test_fastmath.py:    def check(self, test: CUDATestCase, fast: str, prec: str):
numba/cuda/tests/cudapy/test_fastmath.py:@skip_on_cudasim('Fastmath and PTX inspection not available on cudasim')
numba/cuda/tests/cudapy/test_fastmath.py:class TestFastMathOption(CUDATestCase):
numba/cuda/tests/cudapy/test_fastmath.py:        fastver = cuda.jit(sig, device=device, fastmath=True)(pyfunc)
numba/cuda/tests/cudapy/test_fastmath.py:        precver = cuda.jit(sig, device=device)(pyfunc)
numba/cuda/tests/cudapy/test_fastmath.py:        fastver = cuda.jit(sig, fastmath=True, debug=True)(f10)
numba/cuda/tests/cudapy/test_fastmath.py:        precver = cuda.jit(sig, debug=True)(f10)
numba/cuda/tests/cudapy/test_fastmath.py:        @cuda.jit("float32(float32, float32)", device=True)
numba/cuda/tests/cudapy/test_fastmath.py:            i = cuda.grid(1)
numba/cuda/tests/cudapy/test_fastmath.py:        fastver = cuda.jit(sig, fastmath=True)(bar)
numba/cuda/tests/cudapy/test_fastmath.py:        precver = cuda.jit(sig)(bar)
numba/cuda/tests/cudapy/test_fastmath.py:        # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-div
numba/cuda/tests/data/jitlink.ptx:// Generated by NVIDIA NVVM Compiler
numba/cuda/tests/data/jitlink.ptx:// Cuda compilation tools, release 10.2, V10.2.89
numba/cuda/tests/data/cuda_include.cu:// Not all CUDA includes are safe to include in device code compiled by NVRTC,
numba/cuda/tests/data/cuda_include.cu:// such as cuda_device_runtime_api.h are safe to use in NVRTC without adding
numba/cuda/tests/data/cuda_include.cu:#include <cuda_device_runtime_api.h>
numba/cuda/tests/data/jitlink.cu:// The out argument is necessary due to Numba's CUDA calling convention, which
numba/cuda/tests/nocuda/test_library_lookup.py:from numba.cuda.cudadrv import nvvm
numba/cuda/tests/nocuda/test_library_lookup.py:from numba.cuda.testing import (
numba/cuda/tests/nocuda/test_library_lookup.py:    skip_on_cudasim,
numba/cuda/tests/nocuda/test_library_lookup.py:    skip_unless_conda_cudatoolkit,
numba/cuda/tests/nocuda/test_library_lookup.py:from numba.cuda.cuda_paths import (
numba/cuda/tests/nocuda/test_library_lookup.py:    _get_cudalib_dir_path_decision,
numba/cuda/tests/nocuda/test_library_lookup.py:has_cuda = nvvm.is_available()
numba/cuda/tests/nocuda/test_library_lookup.py:@skip_on_cudasim('Library detection unsupported in the simulator')
numba/cuda/tests/nocuda/test_library_lookup.py:@skip_unless_conda_cudatoolkit('test assumes conda installed cudatoolkit')
numba/cuda/tests/nocuda/test_library_lookup.py:        if has_cuda:
numba/cuda/tests/nocuda/test_library_lookup.py:        # Check that CUDA_HOME works by removing conda-env
numba/cuda/tests/nocuda/test_library_lookup.py:        by, info, warns = self.remote_do(self.do_set_cuda_home)
numba/cuda/tests/nocuda/test_library_lookup.py:        self.assertEqual(by, 'CUDA_HOME')
numba/cuda/tests/nocuda/test_library_lookup.py:        self.assertEqual(info, os.path.join('mycudahome', 'nvvm', 'libdevice'))
numba/cuda/tests/nocuda/test_library_lookup.py:            # Fake remove conda environment so no cudatoolkit is available
numba/cuda/tests/nocuda/test_library_lookup.py:            # Use system available cudatoolkit
numba/cuda/tests/nocuda/test_library_lookup.py:        remove_env('CUDA_HOME')
numba/cuda/tests/nocuda/test_library_lookup.py:        remove_env('CUDA_PATH')
numba/cuda/tests/nocuda/test_library_lookup.py:    def do_set_cuda_home():
numba/cuda/tests/nocuda/test_library_lookup.py:        os.environ['CUDA_HOME'] = os.path.join('mycudahome')
numba/cuda/tests/nocuda/test_library_lookup.py:@skip_on_cudasim('Library detection unsupported in the simulator')
numba/cuda/tests/nocuda/test_library_lookup.py:@skip_unless_conda_cudatoolkit('test assumes conda installed cudatoolkit')
numba/cuda/tests/nocuda/test_library_lookup.py:        if has_cuda:
numba/cuda/tests/nocuda/test_library_lookup.py:        # Check that CUDA_HOME works by removing conda-env
numba/cuda/tests/nocuda/test_library_lookup.py:        by, info, warns = self.remote_do(self.do_set_cuda_home)
numba/cuda/tests/nocuda/test_library_lookup.py:        self.assertEqual(by, 'CUDA_HOME')
numba/cuda/tests/nocuda/test_library_lookup.py:            self.assertEqual(info, os.path.join('mycudahome', 'nvvm', 'bin'))
numba/cuda/tests/nocuda/test_library_lookup.py:            self.assertEqual(info, os.path.join('mycudahome', 'nvvm', 'lib'))
numba/cuda/tests/nocuda/test_library_lookup.py:            self.assertEqual(info, os.path.join('mycudahome', 'nvvm', 'lib64'))
numba/cuda/tests/nocuda/test_library_lookup.py:            # Fake remove conda environment so no cudatoolkit is available
numba/cuda/tests/nocuda/test_library_lookup.py:            # Use system available cudatoolkit
numba/cuda/tests/nocuda/test_library_lookup.py:        remove_env('CUDA_HOME')
numba/cuda/tests/nocuda/test_library_lookup.py:        remove_env('CUDA_PATH')
numba/cuda/tests/nocuda/test_library_lookup.py:    def do_set_cuda_home():
numba/cuda/tests/nocuda/test_library_lookup.py:        os.environ['CUDA_HOME'] = os.path.join('mycudahome')
numba/cuda/tests/nocuda/test_library_lookup.py:@skip_on_cudasim('Library detection unsupported in the simulator')
numba/cuda/tests/nocuda/test_library_lookup.py:@skip_unless_conda_cudatoolkit('test assumes conda installed cudatoolkit')
numba/cuda/tests/nocuda/test_library_lookup.py:class TestCudaLibLookUp(LibraryLookupBase):
numba/cuda/tests/nocuda/test_library_lookup.py:    def test_cudalib_path_decision(self):
numba/cuda/tests/nocuda/test_library_lookup.py:        if has_cuda:
numba/cuda/tests/nocuda/test_library_lookup.py:        # Check that CUDA_HOME works by removing conda-env
numba/cuda/tests/nocuda/test_library_lookup.py:        by, info, warns = self.remote_do(self.do_set_cuda_home)
numba/cuda/tests/nocuda/test_library_lookup.py:        self.assertEqual(by, 'CUDA_HOME')
numba/cuda/tests/nocuda/test_library_lookup.py:            self.assertEqual(info, os.path.join('mycudahome', 'bin'))
numba/cuda/tests/nocuda/test_library_lookup.py:            self.assertEqual(info, os.path.join('mycudahome', 'lib'))
numba/cuda/tests/nocuda/test_library_lookup.py:            self.assertEqual(info, os.path.join('mycudahome', 'lib64'))
numba/cuda/tests/nocuda/test_library_lookup.py:            # Fake remove conda environment so no cudatoolkit is available
numba/cuda/tests/nocuda/test_library_lookup.py:            # Use system available cudatoolkit
numba/cuda/tests/nocuda/test_library_lookup.py:        remove_env('CUDA_HOME')
numba/cuda/tests/nocuda/test_library_lookup.py:        remove_env('CUDA_PATH')
numba/cuda/tests/nocuda/test_library_lookup.py:        return True, _get_cudalib_dir_path_decision()
numba/cuda/tests/nocuda/test_library_lookup.py:    def do_set_cuda_home():
numba/cuda/tests/nocuda/test_library_lookup.py:        os.environ['CUDA_HOME'] = os.path.join('mycudahome')
numba/cuda/tests/nocuda/test_library_lookup.py:        return True, _get_cudalib_dir_path_decision()
numba/cuda/tests/nocuda/test_dummyarray.py:from numba.cuda.cudadrv.dummyarray import Array
numba/cuda/tests/nocuda/test_dummyarray.py:from numba.cuda.testing import skip_on_cudasim
numba/cuda/tests/nocuda/test_dummyarray.py:@skip_on_cudasim("Tests internals of the CUDA driver device array")
numba/cuda/tests/nocuda/test_dummyarray.py:@skip_on_cudasim("Tests internals of the CUDA driver device array")
numba/cuda/tests/nocuda/test_dummyarray.py:@skip_on_cudasim("Tests internals of the CUDA driver device array")
numba/cuda/tests/nocuda/test_dummyarray.py:@skip_on_cudasim("Tests internals of the CUDA driver device array")
numba/cuda/tests/nocuda/test_dummyarray.py:@skip_on_cudasim("Tests internals of the CUDA driver device array")
numba/cuda/tests/nocuda/test_function_resolution.py:from numba.cuda.testing import unittest, skip_on_cudasim
numba/cuda/tests/nocuda/test_function_resolution.py:from numba.cuda.cudadrv import nvvm
numba/cuda/tests/nocuda/test_function_resolution.py:@skip_on_cudasim("Skip on simulator due to use of cuda_target")
numba/cuda/tests/nocuda/test_function_resolution.py:        from numba.cuda.descriptor import cuda_target
numba/cuda/tests/nocuda/test_function_resolution.py:            typingctx = cuda_target.typing_context
numba/cuda/tests/nocuda/test_function_resolution.py:        from numba.cuda.descriptor import cuda_target
numba/cuda/tests/nocuda/test_function_resolution.py:            typingctx = cuda_target.typing_context
numba/cuda/tests/nocuda/test_nvvm.py:from numba.cuda.cudadrv import nvvm
numba/cuda/tests/nocuda/test_nvvm.py:from numba.cuda.testing import skip_on_cudasim
numba/cuda/tests/nocuda/test_nvvm.py:@skip_on_cudasim('libNVVM not supported in simulator')
numba/cuda/tests/nocuda/test_nvvm.py:@unittest.skipIf(utils.MACHINE_BITS == 32, "CUDA not support for 32-bit")
numba/cuda/tests/nocuda/test_nvvm.py:class TestNvvmWithoutCuda(unittest.TestCase):
numba/cuda/tests/nocuda/test_nvvm.py:        m.triple = 'nvptx64-nvidia-cuda'
numba/cuda/tests/nocuda/__init__.py:from numba.cuda.testing import ensure_supported_ccs_initialized
numba/cuda/tests/nocuda/test_import.py:        Tests that importing cuda doesn't trigger the import of modules
numba/cuda/tests/nocuda/test_import.py:        code = "import sys; from numba import cuda; print(list(sys.modules))"
numba/cuda/tests/doc_examples/test_laplace.py:from numba.cuda.testing import (CUDATestCase, skip_if_cudadevrt_missing,
numba/cuda/tests/doc_examples/test_laplace.py:                                skip_on_cudasim, skip_unless_cc_60,
numba/cuda/tests/doc_examples/test_laplace.py:@skip_if_cudadevrt_missing
numba/cuda/tests/doc_examples/test_laplace.py:@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
numba/cuda/tests/doc_examples/test_laplace.py:class TestLaplace(CUDATestCase):
numba/cuda/tests/doc_examples/test_laplace.py:        from numba import cuda
numba/cuda/tests/doc_examples/test_laplace.py:        buf_0 = cuda.to_device(data)
numba/cuda/tests/doc_examples/test_laplace.py:        buf_1 = cuda.device_array_like(buf_0)
numba/cuda/tests/doc_examples/test_laplace.py:        @cuda.jit
numba/cuda/tests/doc_examples/test_laplace.py:            i = cuda.grid(1)
numba/cuda/tests/doc_examples/test_laplace.py:            grid = cuda.cg.this_grid()
numba/cuda/tests/doc_examples/test_sessionize.py:from numba.cuda.testing import (CUDATestCase, skip_if_cudadevrt_missing,
numba/cuda/tests/doc_examples/test_sessionize.py:                                skip_on_cudasim, skip_unless_cc_60,
numba/cuda/tests/doc_examples/test_sessionize.py:@skip_if_cudadevrt_missing
numba/cuda/tests/doc_examples/test_sessionize.py:@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
numba/cuda/tests/doc_examples/test_sessionize.py:class TestSessionization(CUDATestCase):
numba/cuda/tests/doc_examples/test_sessionize.py:        from numba import cuda
numba/cuda/tests/doc_examples/test_sessionize.py:        ids = cuda.to_device(
numba/cuda/tests/doc_examples/test_sessionize.py:        sec = cuda.to_device(
numba/cuda/tests/doc_examples/test_sessionize.py:        results = cuda.to_device(np.zeros(len(ids)))
numba/cuda/tests/doc_examples/test_sessionize.py:        @cuda.jit
numba/cuda/tests/doc_examples/test_sessionize.py:            gid = cuda.grid(1)
numba/cuda/tests/doc_examples/test_sessionize.py:                grid = cuda.cg.this_grid()
numba/cuda/tests/doc_examples/test_reduction.py:from numba.cuda.testing import CUDATestCase, skip_on_cudasim
numba/cuda/tests/doc_examples/test_reduction.py:@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
numba/cuda/tests/doc_examples/test_reduction.py:class TestReduction(CUDATestCase):
numba/cuda/tests/doc_examples/test_reduction.py:        from numba import cuda
numba/cuda/tests/doc_examples/test_reduction.py:        a = cuda.to_device(np.arange(1024))
numba/cuda/tests/doc_examples/test_reduction.py:        @cuda.jit
numba/cuda/tests/doc_examples/test_reduction.py:            tid = cuda.threadIdx.x
numba/cuda/tests/doc_examples/test_reduction.py:                i = cuda.grid(1)
numba/cuda/tests/doc_examples/test_reduction.py:                shr = cuda.shared.array(nelem, int32)
numba/cuda/tests/doc_examples/test_reduction.py:                cuda.syncthreads()
numba/cuda/tests/doc_examples/test_reduction.py:                while s < cuda.blockDim.x:
numba/cuda/tests/doc_examples/test_reduction.py:                    cuda.syncthreads()
numba/cuda/tests/doc_examples/test_matmul.py:Matrix multiplication example via `cuda.jit`.
numba/cuda/tests/doc_examples/test_matmul.py:from numba.cuda.testing import CUDATestCase, skip_on_cudasim
numba/cuda/tests/doc_examples/test_matmul.py:@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
numba/cuda/tests/doc_examples/test_matmul.py:class TestMatMul(CUDATestCase):
numba/cuda/tests/doc_examples/test_matmul.py:        from numba import cuda, float32
numba/cuda/tests/doc_examples/test_matmul.py:        @cuda.jit
numba/cuda/tests/doc_examples/test_matmul.py:            i, j = cuda.grid(2)
numba/cuda/tests/doc_examples/test_matmul.py:        x_d = cuda.to_device(x_h)
numba/cuda/tests/doc_examples/test_matmul.py:        y_d = cuda.to_device(y_h)
numba/cuda/tests/doc_examples/test_matmul.py:        z_d = cuda.to_device(z_h)
numba/cuda/tests/doc_examples/test_matmul.py:        @cuda.jit
numba/cuda/tests/doc_examples/test_matmul.py:            Perform matrix multiplication of C = A * B using CUDA shared memory.
numba/cuda/tests/doc_examples/test_matmul.py:            sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
numba/cuda/tests/doc_examples/test_matmul.py:            sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
numba/cuda/tests/doc_examples/test_matmul.py:            x, y = cuda.grid(2)
numba/cuda/tests/doc_examples/test_matmul.py:            tx = cuda.threadIdx.x
numba/cuda/tests/doc_examples/test_matmul.py:            ty = cuda.threadIdx.y
numba/cuda/tests/doc_examples/test_matmul.py:            bpg = cuda.gridDim.x    # blocks per grid
numba/cuda/tests/doc_examples/test_matmul.py:                cuda.syncthreads()
numba/cuda/tests/doc_examples/test_matmul.py:                cuda.syncthreads()
numba/cuda/tests/doc_examples/test_matmul.py:        x_d = cuda.to_device(x_h)
numba/cuda/tests/doc_examples/test_matmul.py:        y_d = cuda.to_device(y_h)
numba/cuda/tests/doc_examples/test_matmul.py:        z_d = cuda.to_device(z_h)
numba/cuda/tests/doc_examples/test_matmul.py:        x_d = cuda.to_device(x_h)
numba/cuda/tests/doc_examples/test_matmul.py:        y_d = cuda.to_device(y_h)
numba/cuda/tests/doc_examples/test_matmul.py:        z_d = cuda.to_device(z_h)
numba/cuda/tests/doc_examples/test_random.py:from numba.cuda.testing import CUDATestCase, skip_on_cudasim
numba/cuda/tests/doc_examples/test_random.py:@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
numba/cuda/tests/doc_examples/test_random.py:class TestRandom(CUDATestCase):
numba/cuda/tests/doc_examples/test_random.py:        from numba import cuda
numba/cuda/tests/doc_examples/test_random.py:        from numba.cuda.random import (create_xoroshiro128p_states,
numba/cuda/tests/doc_examples/test_random.py:        @cuda.jit
numba/cuda/tests/doc_examples/test_random.py:            startx, starty, startz = cuda.grid(3)
numba/cuda/tests/doc_examples/test_random.py:            stridex, stridey, stridez = cuda.gridsize(3)
numba/cuda/tests/doc_examples/test_random.py:        arr = cuda.device_array((X, Y, Z), dtype=np.float32)
numba/cuda/tests/doc_examples/test_cg.py:from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
numba/cuda/tests/doc_examples/test_cg.py:                                skip_if_cudadevrt_missing, skip_unless_cc_60,
numba/cuda/tests/doc_examples/test_cg.py:@skip_if_cudadevrt_missing
numba/cuda/tests/doc_examples/test_cg.py:@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
numba/cuda/tests/doc_examples/test_cg.py:class TestCooperativeGroups(CUDATestCase):
numba/cuda/tests/doc_examples/test_cg.py:        from numba import cuda, int32
numba/cuda/tests/doc_examples/test_cg.py:        @cuda.jit(sig)
numba/cuda/tests/doc_examples/test_cg.py:            col = cuda.grid(1)
numba/cuda/tests/doc_examples/test_cg.py:            g = cuda.cg.this_grid()
numba/cuda/tests/doc_examples/test_cg.py:        # a cooperative launch on the current GPU
numba/cuda/tests/doc_examples/test_ffi.py:from numba.cuda.testing import (CUDATestCase, skip_on_cudasim)
numba/cuda/tests/doc_examples/test_ffi.py:@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
numba/cuda/tests/doc_examples/test_ffi.py:class TestFFI(CUDATestCase):
numba/cuda/tests/doc_examples/test_ffi.py:        from numba import cuda
numba/cuda/tests/doc_examples/test_ffi.py:        mul = cuda.declare_device('mul_f32_f32', 'float32(float32, float32)')
numba/cuda/tests/doc_examples/test_ffi.py:        @cuda.jit(link=[functions_cu])
numba/cuda/tests/doc_examples/test_ffi.py:            i = cuda.grid(1)
numba/cuda/tests/doc_examples/test_ffi.py:        from numba import cuda
numba/cuda/tests/doc_examples/test_ffi.py:        sum_reduce = cuda.declare_device('sum_reduce', signature)
numba/cuda/tests/doc_examples/test_ffi.py:        @cuda.jit(link=[functions_cu])
numba/cuda/tests/doc_examples/test_vecadd.py:from numba.cuda.testing import CUDATestCase, skip_on_cudasim
numba/cuda/tests/doc_examples/test_vecadd.py:@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
numba/cuda/tests/doc_examples/test_vecadd.py:class TestVecAdd(CUDATestCase):
numba/cuda/tests/doc_examples/test_vecadd.py:        from numba import cuda
numba/cuda/tests/doc_examples/test_vecadd.py:        @cuda.jit
numba/cuda/tests/doc_examples/test_vecadd.py:            tid = cuda.grid(1)
numba/cuda/tests/doc_examples/test_vecadd.py:        a = cuda.to_device(np.random.random(N))
numba/cuda/tests/doc_examples/test_vecadd.py:        b = cuda.to_device(np.random.random(N))
numba/cuda/tests/doc_examples/test_vecadd.py:        c = cuda.device_array_like(a)
numba/cuda/tests/doc_examples/test_ufunc.py:from numba.cuda.testing import CUDATestCase, skip_on_cudasim
numba/cuda/tests/doc_examples/test_ufunc.py:@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
numba/cuda/tests/doc_examples/test_ufunc.py:class TestUFunc(CUDATestCase):
numba/cuda/tests/doc_examples/test_ufunc.py:    def test_ex_cuda_ufunc_call(self):
numba/cuda/tests/doc_examples/test_ufunc.py:        # ex_cuda_ufunc.begin
numba/cuda/tests/doc_examples/test_ufunc.py:        from numba import cuda
numba/cuda/tests/doc_examples/test_ufunc.py:        @cuda.jit
numba/cuda/tests/doc_examples/test_ufunc.py:        # ex_cuda_ufunc.end
numba/cuda/tests/doc_examples/test_montecarlo.py:from numba.cuda.testing import CUDATestCase, skip_on_cudasim
numba/cuda/tests/doc_examples/test_montecarlo.py:@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
numba/cuda/tests/doc_examples/test_montecarlo.py:class TestMonteCarlo(CUDATestCase):
numba/cuda/tests/doc_examples/test_montecarlo.py:        from numba import cuda
numba/cuda/tests/doc_examples/test_montecarlo.py:        from numba.cuda.random import (
numba/cuda/tests/doc_examples/test_montecarlo.py:        @cuda.jit
numba/cuda/tests/doc_examples/test_montecarlo.py:            gid = cuda.grid(1)
numba/cuda/tests/doc_examples/test_montecarlo.py:        @cuda.reduce
numba/cuda/tests/doc_examples/test_montecarlo.py:            out = cuda.to_device(np.zeros(nsamps, dtype="float32"))
numba/cuda/tests/doc_examples/test_montecarlo.py:            # jit the function for use in CUDA kernels
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:from numba.cuda.testing import CUDATestCase, skip_on_cudasim
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:class TestCpuGpuCompat(CUDATestCase):
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:    Test compatibility of CPU and GPU functions
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:    def test_ex_cpu_gpu_compat(self):
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        # ex_cpu_gpu_compat.import.begin
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        from numba import cuda
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        # ex_cpu_gpu_compat.import.end
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        # ex_cpu_gpu_compat.allocate.begin
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        X = cuda.to_device([1, 10, 234])
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        Y = cuda.to_device([2, 2, 4014])
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        Z = cuda.to_device([3, 14, 2211])
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        results = cuda.to_device([0.0, 0.0, 0.0])
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        # ex_cpu_gpu_compat.allocate.end
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        # ex_cpu_gpu_compat.define.begin
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        # ex_cpu_gpu_compat.define.end
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        # ex_cpu_gpu_compat.cpurun.begin
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        # ex_cpu_gpu_compat.cpurun.end
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        # ex_cpu_gpu_compat.usegpu.begin
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        @cuda.jit
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:            tid = cuda.grid(1)
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        # ex_cpu_gpu_compat.usegpu.end
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        # ex_cpu_gpu_compat.launch.begin
numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py:        # ex_cpu_gpu_compat.launch.end
numba/cuda/tests/cudadrv/test_init.py:from numba import cuda
numba/cuda/tests/cudadrv/test_init.py:from numba.cuda.cudadrv.driver import CudaAPIError, driver
numba/cuda/tests/cudadrv/test_init.py:from numba.cuda.cudadrv.error import CudaSupportError
numba/cuda/tests/cudadrv/test_init.py:from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
numba/cuda/tests/cudadrv/test_init.py:# A mock of cuInit that always raises a CudaAPIError
numba/cuda/tests/cudadrv/test_init.py:    raise CudaAPIError(999, 'CUDA_ERROR_UNKNOWN')
numba/cuda/tests/cudadrv/test_init.py:        # A CUDA operation that forces initialization of the device
numba/cuda/tests/cudadrv/test_init.py:        cuda.device_array(1)
numba/cuda/tests/cudadrv/test_init.py:    except CudaSupportError as e:
numba/cuda/tests/cudadrv/test_init.py:# returned by cuda_error() is as expected.
numba/cuda/tests/cudadrv/test_init.py:        # A CUDA operation that forces initialization of the device
numba/cuda/tests/cudadrv/test_init.py:        cuda.device_array(1)
numba/cuda/tests/cudadrv/test_init.py:    except CudaSupportError:
numba/cuda/tests/cudadrv/test_init.py:    msg = cuda.cuda_error()
numba/cuda/tests/cudadrv/test_init.py:# For testing the path where Driver.__init__() catches a CudaSupportError
numba/cuda/tests/cudadrv/test_init.py:def cuda_disabled_test(result_queue):
numba/cuda/tests/cudadrv/test_init.py:        # A CUDA operation that forces initialization of the device
numba/cuda/tests/cudadrv/test_init.py:        cuda.device_array(1)
numba/cuda/tests/cudadrv/test_init.py:    except CudaSupportError as e:
numba/cuda/tests/cudadrv/test_init.py:# Similar to cuda_disabled_test, but checks cuda.cuda_error() instead of the
numba/cuda/tests/cudadrv/test_init.py:def cuda_disabled_error_test(result_queue):
numba/cuda/tests/cudadrv/test_init.py:        # A CUDA operation that forces initialization of the device
numba/cuda/tests/cudadrv/test_init.py:        cuda.device_array(1)
numba/cuda/tests/cudadrv/test_init.py:    except CudaSupportError:
numba/cuda/tests/cudadrv/test_init.py:    msg = cuda.cuda_error()
numba/cuda/tests/cudadrv/test_init.py:@skip_on_cudasim('CUDA Simulator does not initialize driver')
numba/cuda/tests/cudadrv/test_init.py:class TestInit(CUDATestCase):
numba/cuda/tests/cudadrv/test_init.py:            self.fail('CudaSupportError not raised')
numba/cuda/tests/cudadrv/test_init.py:        expected = 'Error at driver init: CUDA_ERROR_UNKNOWN (999)'
numba/cuda/tests/cudadrv/test_init.py:        expected = 'CUDA_ERROR_UNKNOWN (999)'
numba/cuda/tests/cudadrv/test_init.py:    def _test_cuda_disabled(self, target):
numba/cuda/tests/cudadrv/test_init.py:        # with CUDA disabled.
numba/cuda/tests/cudadrv/test_init.py:        cuda_disabled = os.environ.get('NUMBA_DISABLE_CUDA')
numba/cuda/tests/cudadrv/test_init.py:        os.environ['NUMBA_DISABLE_CUDA'] = "1"
numba/cuda/tests/cudadrv/test_init.py:            expected = 'CUDA is disabled due to setting NUMBA_DISABLE_CUDA=1'
numba/cuda/tests/cudadrv/test_init.py:            self._test_init_failure(cuda_disabled_test, expected)
numba/cuda/tests/cudadrv/test_init.py:            if cuda_disabled is not None:
numba/cuda/tests/cudadrv/test_init.py:                os.environ['NUMBA_DISABLE_CUDA'] = cuda_disabled
numba/cuda/tests/cudadrv/test_init.py:                os.environ.pop('NUMBA_DISABLE_CUDA')
numba/cuda/tests/cudadrv/test_init.py:    def test_cuda_disabled_raising(self):
numba/cuda/tests/cudadrv/test_init.py:        self._test_cuda_disabled(cuda_disabled_test)
numba/cuda/tests/cudadrv/test_init.py:    def test_cuda_disabled_error(self):
numba/cuda/tests/cudadrv/test_init.py:        self._test_cuda_disabled(cuda_disabled_error_test)
numba/cuda/tests/cudadrv/test_init.py:        self.assertIsNone(cuda.cuda_error())
numba/cuda/tests/cudadrv/test_streams.py:from numba import cuda
numba/cuda/tests/cudadrv/test_streams.py:from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
numba/cuda/tests/cudadrv/test_streams.py:@skip_on_cudasim('CUDA Driver API unsupported in the simulator')
numba/cuda/tests/cudadrv/test_streams.py:class TestCudaStream(CUDATestCase):
numba/cuda/tests/cudadrv/test_streams.py:        stream = cuda.stream()
numba/cuda/tests/cudadrv/test_streams.py:        stream = cuda.stream()
numba/cuda/tests/cudadrv/test_streams.py:        stream = cuda.stream()
numba/cuda/tests/cudadrv/test_streams.py:        async def async_cuda_fn(value_in: float) -> float:
numba/cuda/tests/cudadrv/test_streams.py:            stream = cuda.stream()
numba/cuda/tests/cudadrv/test_streams.py:            h_src, h_dst = cuda.pinned_array(8), cuda.pinned_array(8)
numba/cuda/tests/cudadrv/test_streams.py:            d_ary = cuda.to_device(h_src, stream=stream)
numba/cuda/tests/cudadrv/test_streams.py:        tasks = [asyncio.create_task(async_cuda_fn(v)) for v in values_in]
numba/cuda/tests/cudadrv/test_streams.py:        stream = cuda.stream()
numba/cuda/tests/cudadrv/test_streams.py:        streams = [cuda.stream() for _ in range(4)]
numba/cuda/tests/cudadrv/test_streams.py:        stream = cuda.stream()
numba/cuda/tests/cudadrv/test_streams.py:@skip_on_cudasim('CUDA Driver API unsupported in the simulator')
numba/cuda/tests/cudadrv/test_streams.py:class TestFailingStream(CUDATestCase):
numba/cuda/tests/cudadrv/test_streams.py:    # This test can only be run in isolation because it corrupts the CUDA
numba/cuda/tests/cudadrv/test_streams.py:    # CUDA will have been initialized before the fork, so it cannot be used in
numba/cuda/tests/cudadrv/test_streams.py:        ctx = cuda.current_context()
numba/cuda/tests/cudadrv/test_streams.py:        stream = cuda.stream()
numba/cuda/tests/cudadrv/test_deallocations.py:from numba import cuda
numba/cuda/tests/cudadrv/test_deallocations.py:from numba.cuda.testing import (unittest, skip_on_cudasim,
numba/cuda/tests/cudadrv/test_deallocations.py:                                skip_if_external_memmgr, CUDATestCase)
numba/cuda/tests/cudadrv/test_deallocations.py:@skip_on_cudasim('not supported on CUDASIM')
numba/cuda/tests/cudadrv/test_deallocations.py:class TestDeallocation(CUDATestCase):
numba/cuda/tests/cudadrv/test_deallocations.py:        deallocs = cuda.current_context().memory_manager.deallocations
numba/cuda/tests/cudadrv/test_deallocations.py:        for i in range(config.CUDA_DEALLOCS_COUNT):
numba/cuda/tests/cudadrv/test_deallocations.py:            cuda.to_device(np.arange(1))
numba/cuda/tests/cudadrv/test_deallocations.py:        cuda.to_device(np.arange(1))
numba/cuda/tests/cudadrv/test_deallocations.py:        ctx = cuda.current_context()
numba/cuda/tests/cudadrv/test_deallocations.py:        old_ratio = config.CUDA_DEALLOCS_RATIO
numba/cuda/tests/cudadrv/test_deallocations.py:            config.CUDA_DEALLOCS_RATIO = max_pending / mi.total
numba/cuda/tests/cudadrv/test_deallocations.py:            cuda.to_device(np.ones(max_pending // 2, dtype=np.int8))
numba/cuda/tests/cudadrv/test_deallocations.py:            cuda.to_device(np.ones(deallocs._max_pending_bytes -
numba/cuda/tests/cudadrv/test_deallocations.py:            cuda.to_device(np.ones(1, dtype=np.int8))
numba/cuda/tests/cudadrv/test_deallocations.py:            config.CUDA_DEALLOCS_RATIO = old_ratio
numba/cuda/tests/cudadrv/test_deallocations.py:@skip_on_cudasim("defer_cleanup has no effect in CUDASIM")
numba/cuda/tests/cudadrv/test_deallocations.py:class TestDeferCleanup(CUDATestCase):
numba/cuda/tests/cudadrv/test_deallocations.py:        darr1 = cuda.to_device(harr)
numba/cuda/tests/cudadrv/test_deallocations.py:        deallocs = cuda.current_context().memory_manager.deallocations
numba/cuda/tests/cudadrv/test_deallocations.py:        with cuda.defer_cleanup():
numba/cuda/tests/cudadrv/test_deallocations.py:            darr2 = cuda.to_device(harr)
numba/cuda/tests/cudadrv/test_deallocations.py:        darr1 = cuda.to_device(harr)
numba/cuda/tests/cudadrv/test_deallocations.py:        deallocs = cuda.current_context().memory_manager.deallocations
numba/cuda/tests/cudadrv/test_deallocations.py:        with cuda.defer_cleanup():
numba/cuda/tests/cudadrv/test_deallocations.py:            with cuda.defer_cleanup():
numba/cuda/tests/cudadrv/test_deallocations.py:                darr2 = cuda.to_device(harr)
numba/cuda/tests/cudadrv/test_deallocations.py:        darr1 = cuda.to_device(harr)
numba/cuda/tests/cudadrv/test_deallocations.py:        deallocs = cuda.current_context().memory_manager.deallocations
numba/cuda/tests/cudadrv/test_deallocations.py:            with cuda.defer_cleanup():
numba/cuda/tests/cudadrv/test_deallocations.py:                darr2 = cuda.to_device(harr)
numba/cuda/tests/cudadrv/test_deallocations.py:class TestDeferCleanupAvail(CUDATestCase):
numba/cuda/tests/cudadrv/test_deallocations.py:        with cuda.defer_cleanup():
numba/cuda/tests/cudadrv/test_deallocations.py:@skip_on_cudasim('not supported on CUDASIM')
numba/cuda/tests/cudadrv/test_deallocations.py:class TestDel(CUDATestCase):
numba/cuda/tests/cudadrv/test_deallocations.py:        ctx = cuda.current_context()
numba/cuda/tests/cudadrv/test_deallocations.py:        ctx = cuda.current_context()
numba/cuda/tests/cudadrv/test_deallocations.py:        ctx = cuda.current_context()
numba/cuda/tests/cudadrv/test_deallocations.py:        ctx = cuda.current_context()
numba/cuda/tests/cudadrv/test_deallocations.py:        ctx = cuda.current_context()
numba/cuda/tests/cudadrv/test_deallocations.py:        ctx = cuda.current_context()
numba/cuda/tests/cudadrv/test_deallocations.py:        ctx = cuda.current_context()
numba/cuda/tests/cudadrv/test_deallocations.py:            with cuda.pinned(arr):
numba/cuda/tests/cudadrv/test_deallocations.py:            with cuda.pinned(arr):
numba/cuda/tests/cudadrv/test_deallocations.py:            with cuda.defer_cleanup():
numba/cuda/tests/cudadrv/test_deallocations.py:                with cuda.pinned(arr):
numba/cuda/tests/cudadrv/test_deallocations.py:                with cuda.pinned(arr):
numba/cuda/tests/cudadrv/test_deallocations.py:                with cuda.pinned(arr):
numba/cuda/tests/cudadrv/test_deallocations.py:                with cuda.pinned(arr):
numba/cuda/tests/cudadrv/test_deallocations.py:        ctx = cuda.current_context()
numba/cuda/tests/cudadrv/test_deallocations.py:            with cuda.mapped(arr):
numba/cuda/tests/cudadrv/test_deallocations.py:            with cuda.mapped(arr):
numba/cuda/tests/cudadrv/test_deallocations.py:            with cuda.defer_cleanup():
numba/cuda/tests/cudadrv/test_deallocations.py:                with cuda.mapped(arr):
numba/cuda/tests/cudadrv/test_deallocations.py:                with cuda.mapped(arr):
numba/cuda/tests/cudadrv/test_deallocations.py:                with cuda.mapped(arr):
numba/cuda/tests/cudadrv/test_deallocations.py:                with cuda.mapped(arr):
numba/cuda/tests/cudadrv/test_cuda_auto_context.py:from numba import cuda
numba/cuda/tests/cudadrv/test_cuda_auto_context.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudadrv/test_cuda_auto_context.py:class TestCudaAutoContext(CUDATestCase):
numba/cuda/tests/cudadrv/test_cuda_auto_context.py:        """A problem was revealed by a customer that the use cuda.to_device
numba/cuda/tests/cudadrv/test_cuda_auto_context.py:        does not create a CUDA context.
numba/cuda/tests/cudadrv/test_cuda_auto_context.py:        dA = cuda.to_device(A)
numba/cuda/tests/cudadrv/test_runtime.py:from numba.cuda.cudadrv.runtime import runtime
numba/cuda/tests/cudadrv/test_runtime.py:from numba.cuda.testing import unittest, SerialMixin, skip_on_cudasim
numba/cuda/tests/cudadrv/test_runtime.py:        from numba import cuda
numba/cuda/tests/cudadrv/test_runtime.py:        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
numba/cuda/tests/cudadrv/test_runtime.py:        q.put(len(cuda.gpus.lst))
numba/cuda/tests/cudadrv/test_runtime.py:if config.ENABLE_CUDASIM:
numba/cuda/tests/cudadrv/test_runtime.py:    @skip_on_cudasim('The simulator always simulates a supported runtime')
numba/cuda/tests/cudadrv/test_runtime.py:        # CUDA_VISIBLE_DEVICES after importing Numba and have the value
numba/cuda/tests/cudadrv/test_runtime.py:        # reflected in the available list of GPUs. Prior to the fix for this
numba/cuda/tests/cudadrv/test_runtime.py:        # CUDA_VISIBLE_DEVICES could be set by the user.
numba/cuda/tests/cudadrv/test_runtime.py:        # Avoid importing cuda at the top level so that
numba/cuda/tests/cudadrv/test_runtime.py:        from numba import cuda
numba/cuda/tests/cudadrv/test_runtime.py:        if len(cuda.gpus.lst) in (0, 1):
numba/cuda/tests/cudadrv/test_runtime.py:            self.skipTest('This test requires multiple GPUs')
numba/cuda/tests/cudadrv/test_runtime.py:        if os.environ.get('CUDA_VISIBLE_DEVICES'):
numba/cuda/tests/cudadrv/test_runtime.py:            msg = 'Cannot test when CUDA_VISIBLE_DEVICES already set'
numba/cuda/tests/cudadrv/test_runtime.py:            visible_gpu_count = q.get()
numba/cuda/tests/cudadrv/test_runtime.py:        # and an incorrect number of GPUs in the list
numba/cuda/tests/cudadrv/test_runtime.py:        self.assertNotEqual(visible_gpu_count, -1, msg=msg)
numba/cuda/tests/cudadrv/test_runtime.py:        # The actual check that we see only one GPU
numba/cuda/tests/cudadrv/test_runtime.py:        self.assertEqual(visible_gpu_count, 1)
numba/cuda/tests/cudadrv/test_is_fp16.py:from numba import cuda
numba/cuda/tests/cudadrv/test_is_fp16.py:from numba.cuda.testing import CUDATestCase, skip_on_cudasim, skip_unless_cc_53
numba/cuda/tests/cudadrv/test_is_fp16.py:class TestIsFP16Supported(CUDATestCase):
numba/cuda/tests/cudadrv/test_is_fp16.py:        self.assertTrue(cuda.is_float16_supported())
numba/cuda/tests/cudadrv/test_is_fp16.py:    @skip_on_cudasim
numba/cuda/tests/cudadrv/test_is_fp16.py:        self.assertTrue(cuda.get_current_device().supports_float16)
numba/cuda/tests/cudadrv/test_linker.py:from numba.cuda.testing import unittest
numba/cuda/tests/cudadrv/test_linker.py:from numba.cuda.testing import (skip_on_cudasim, skip_if_cuda_includes_missing)
numba/cuda/tests/cudadrv/test_linker.py:from numba.cuda.testing import CUDATestCase, test_data_dir
numba/cuda/tests/cudadrv/test_linker.py:from numba.cuda.cudadrv.driver import (CudaAPIError, Linker,
numba/cuda/tests/cudadrv/test_linker.py:from numba.cuda.cudadrv.error import NvrtcError
numba/cuda/tests/cudadrv/test_linker.py:from numba.cuda import require_context
numba/cuda/tests/cudadrv/test_linker.py:from numba import cuda, void, float64, int64, int32, typeof, float32
numba/cuda/tests/cudadrv/test_linker.py:    C = cuda.const.array_like(CONST1D)
numba/cuda/tests/cudadrv/test_linker.py:    i = cuda.grid(1)
numba/cuda/tests/cudadrv/test_linker.py:    x[cuda.grid(1)] = a1 + a2 + a3 + a4 + a5
numba/cuda/tests/cudadrv/test_linker.py:    x[cuda.grid(1)] += b1 + b2 + b3 + b4 + b5
numba/cuda/tests/cudadrv/test_linker.py:    x[cuda.grid(1)] += c1 + c2 + c3 + c4 + c5
numba/cuda/tests/cudadrv/test_linker.py:    x[cuda.grid(1)] += d1 + d2 + d3 + d4 + d5
numba/cuda/tests/cudadrv/test_linker.py:    sm = cuda.shared.array(100, dty)
numba/cuda/tests/cudadrv/test_linker.py:    i = cuda.grid(1)
numba/cuda/tests/cudadrv/test_linker.py:    cuda.syncthreads()
numba/cuda/tests/cudadrv/test_linker.py:    i, j = cuda.grid(2)
numba/cuda/tests/cudadrv/test_linker.py:    sm = cuda.shared.array((10, 20), float32)
numba/cuda/tests/cudadrv/test_linker.py:    cuda.syncthreads()
numba/cuda/tests/cudadrv/test_linker.py:    i = cuda.grid(1)
numba/cuda/tests/cudadrv/test_linker.py:    C = cuda.local.array(LMEM_SIZE, dty)
numba/cuda/tests/cudadrv/test_linker.py:@skip_on_cudasim('Linking unsupported in the simulator')
numba/cuda/tests/cudadrv/test_linker.py:class TestLinker(CUDATestCase):
numba/cuda/tests/cudadrv/test_linker.py:    _NUMBA_NVIDIA_BINDING_0_ENV = {'NUMBA_CUDA_USE_NVIDIA_BINDING': '0'}
numba/cuda/tests/cudadrv/test_linker.py:        bar = cuda.declare_device('bar', 'int32(int32)')
numba/cuda/tests/cudadrv/test_linker.py:        @cuda.jit(*args, link=[link])
numba/cuda/tests/cudadrv/test_linker.py:            i = cuda.grid(1)
numba/cuda/tests/cudadrv/test_linker.py:        bar = cuda.declare_device('bar', 'int32(int32)')
numba/cuda/tests/cudadrv/test_linker.py:        @cuda.jit(link=[link])
numba/cuda/tests/cudadrv/test_linker.py:            i = cuda.grid(1)
numba/cuda/tests/cudadrv/test_linker.py:        bar = cuda.declare_device('bar', 'int32(int32)')
numba/cuda/tests/cudadrv/test_linker.py:            @cuda.jit('void(int32)', link=[link])
numba/cuda/tests/cudadrv/test_linker.py:        bar = cuda.declare_device('bar', 'int32(int32)')
numba/cuda/tests/cudadrv/test_linker.py:            @cuda.jit('void(int32)', link=[link])
numba/cuda/tests/cudadrv/test_linker.py:        # Check the expected error in the CUDA source is reported
numba/cuda/tests/cudadrv/test_linker.py:            @cuda.jit('void()', link=['header.cuh'])
numba/cuda/tests/cudadrv/test_linker.py:            @cuda.jit('void()', link=['data'])
numba/cuda/tests/cudadrv/test_linker.py:    @skip_if_cuda_includes_missing
numba/cuda/tests/cudadrv/test_linker.py:    def test_linking_cu_cuda_include(self):
numba/cuda/tests/cudadrv/test_linker.py:        link = str(test_data_dir / 'cuda_include.cu')
numba/cuda/tests/cudadrv/test_linker.py:        # compile failure if CUDA includes cannot be found by Nvrtc.
numba/cuda/tests/cudadrv/test_linker.py:        @cuda.jit('void()', link=[link])
numba/cuda/tests/cudadrv/test_linker.py:            @cuda.jit('void(int32[::1])', link=['nonexistent.a'])
numba/cuda/tests/cudadrv/test_linker.py:        compiled = cuda.jit(func_with_lots_of_registers)
numba/cuda/tests/cudadrv/test_linker.py:        compiled = cuda.jit(max_registers=57)(func_with_lots_of_registers)
numba/cuda/tests/cudadrv/test_linker.py:        compiled = cuda.jit(max_registers=38)(func_with_lots_of_registers)
numba/cuda/tests/cudadrv/test_linker.py:        compiled = cuda.jit(sig, max_registers=38)(func_with_lots_of_registers)
numba/cuda/tests/cudadrv/test_linker.py:        compiled = cuda.jit(sig)(simple_const_mem)
numba/cuda/tests/cudadrv/test_linker.py:        compiled = cuda.jit(func_with_lots_of_registers)
numba/cuda/tests/cudadrv/test_linker.py:        compiled = cuda.jit(sig)(simple_smem)
numba/cuda/tests/cudadrv/test_linker.py:        compiled = cuda.jit(simple_smem)
numba/cuda/tests/cudadrv/test_linker.py:        compiled = cuda.jit("void(float32[:,::1])")(coop_smem2d)
numba/cuda/tests/cudadrv/test_linker.py:        compiled = cuda.jit("void(int32[::1])")(simple_maxthreads)
numba/cuda/tests/cudadrv/test_linker.py:        except CudaAPIError as e:
numba/cuda/tests/cudadrv/test_linker.py:        compiled = cuda.jit(sig)(simple_lmem)
numba/cuda/tests/cudadrv/test_linker.py:        compiled = cuda.jit(simple_lmem)
numba/cuda/tests/cudadrv/test_profiler.py:from numba.cuda.testing import ContextResettingTestCase
numba/cuda/tests/cudadrv/test_profiler.py:from numba import cuda
numba/cuda/tests/cudadrv/test_profiler.py:from numba.cuda.testing import skip_on_cudasim
numba/cuda/tests/cudadrv/test_profiler.py:@skip_on_cudasim('CUDA Profiler unsupported in the simulator')
numba/cuda/tests/cudadrv/test_profiler.py:        with cuda.profiling():
numba/cuda/tests/cudadrv/test_profiler.py:            a = cuda.device_array(10)
numba/cuda/tests/cudadrv/test_profiler.py:        with cuda.profiling():
numba/cuda/tests/cudadrv/test_profiler.py:            a = cuda.device_array(100)
numba/cuda/tests/cudadrv/test_select_device.py:from numba import cuda
numba/cuda/tests/cudadrv/test_select_device.py:from numba.cuda.testing import unittest, ContextResettingTestCase
numba/cuda/tests/cudadrv/test_select_device.py:        cuda.select_device(0)
numba/cuda/tests/cudadrv/test_select_device.py:        stream = cuda.stream()
numba/cuda/tests/cudadrv/test_select_device.py:        dA = cuda.to_device(A, stream=stream)
numba/cuda/tests/cudadrv/test_select_device.py:        cuda.close()
numba/cuda/tests/cudadrv/test_host_alloc.py:from numba.cuda.cudadrv import driver
numba/cuda/tests/cudadrv/test_host_alloc.py:from numba import cuda
numba/cuda/tests/cudadrv/test_host_alloc.py:from numba.cuda.testing import unittest, ContextResettingTestCase
numba/cuda/tests/cudadrv/test_host_alloc.py:        mem = cuda.current_context().memhostalloc(n, mapped=True)
numba/cuda/tests/cudadrv/test_host_alloc.py:        ary = cuda.pinned_array(10, dtype=np.uint32)
numba/cuda/tests/cudadrv/test_host_alloc.py:        devary = cuda.to_device(ary)
numba/cuda/tests/cudadrv/test_host_alloc.py:        ary = cuda.mapped_array(10, dtype=np.uint32)
numba/cuda/tests/cudadrv/test_host_alloc.py:        for ary in [cuda.mapped_array(10, dtype=np.uint32),
numba/cuda/tests/cudadrv/test_host_alloc.py:                    cuda.pinned_array(10, dtype=np.uint32)]:
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:from numba.cuda.cudadrv.devicearray import (DeviceRecord, from_record_like,
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:from numba.cuda.testing import skip_on_cudasim
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:from numba import cuda
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:@skip_on_cudasim('Device Record API unsupported in the simulator')
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:class TestCudaDeviceRecord(CUDATestCase):
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:        self.assertIsNotNone(rec.gpu_data)
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:        # Copy non-zero values to GPU and back and check values
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:        self.assertNotEqual(devrec.gpu_data, devrec2.gpu_data)
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:        devrec, new_gpu_obj = auto_device(hostrec)
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:        self.assertTrue(new_gpu_obj)
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:class TestCudaDeviceRecordWithRecord(TestCudaDeviceRecord):
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:        CUDATestCase.setUp(self)
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:@skip_on_cudasim('Structured array attr access not supported in simulator')
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:class TestRecordDtypeWithStructArrays(CUDATestCase):
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:        self.sample1d = cuda.device_array(3, dtype=recordtype)
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:        self.samplerec1darr = cuda.device_array(1, dtype=recordwitharray)[0]
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:        self.samplerecmat = cuda.device_array(1,dtype=recwithmat)[0]
numba/cuda/tests/cudadrv/test_cuda_devicerecord.py:        d_arr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_events.py:from numba import cuda
numba/cuda/tests/cudadrv/test_events.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudadrv/test_events.py:class TestCudaEvent(CUDATestCase):
numba/cuda/tests/cudadrv/test_events.py:        dary = cuda.device_array(N, dtype=np.double)
numba/cuda/tests/cudadrv/test_events.py:        evtstart = cuda.event()
numba/cuda/tests/cudadrv/test_events.py:        evtend = cuda.event()
numba/cuda/tests/cudadrv/test_events.py:        cuda.to_device(np.arange(N, dtype=np.double), to=dary)
numba/cuda/tests/cudadrv/test_events.py:        stream = cuda.stream()
numba/cuda/tests/cudadrv/test_events.py:        dary = cuda.device_array(N, dtype=np.double)
numba/cuda/tests/cudadrv/test_events.py:        evtstart = cuda.event()
numba/cuda/tests/cudadrv/test_events.py:        evtend = cuda.event()
numba/cuda/tests/cudadrv/test_events.py:        cuda.to_device(np.arange(N, dtype=np.double), to=dary, stream=stream)
numba/cuda/tests/cudadrv/test_cuda_driver.py:from numba.cuda.cudadrv.driver import (host_to_device, device_to_host, driver,
numba/cuda/tests/cudadrv/test_cuda_driver.py:from numba.cuda.cudadrv import devices, drvapi, driver as _driver
numba/cuda/tests/cudadrv/test_cuda_driver.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudadrv/test_cuda_driver.py:from numba.cuda.testing import skip_on_cudasim
numba/cuda/tests/cudadrv/test_cuda_driver.py:    .param .u64 __cudaparm__Z10helloworldPi_A)
numba/cuda/tests/cudadrv/test_cuda_driver.py:    ld.param.u64 	%rd1, [__cudaparm__Z10helloworldPi_A];
numba/cuda/tests/cudadrv/test_cuda_driver.py:    .file	1 "/tmp/tmpxft_000012c7_00000000-9_testcuda.cpp3.i"
numba/cuda/tests/cudadrv/test_cuda_driver.py:    .file	2 "testcuda.cu"
numba/cuda/tests/cudadrv/test_cuda_driver.py:@skip_on_cudasim('CUDA Driver API unsupported in the simulator')
numba/cuda/tests/cudadrv/test_cuda_driver.py:class TestCudaDriver(CUDATestCase):
numba/cuda/tests/cudadrv/test_cuda_driver.py:        self.assertTrue(len(devices.gpus) > 0)
numba/cuda/tests/cudadrv/test_cuda_driver.py:    def test_cuda_driver_basic(self):
numba/cuda/tests/cudadrv/test_cuda_driver.py:    def test_cuda_driver_stream_operations(self):
numba/cuda/tests/cudadrv/test_cuda_driver.py:    def test_cuda_driver_default_stream(self):
numba/cuda/tests/cudadrv/test_cuda_driver.py:        self.assertIn("Default CUDA stream", repr(ds))
numba/cuda/tests/cudadrv/test_cuda_driver.py:    def test_cuda_driver_legacy_default_stream(self):
numba/cuda/tests/cudadrv/test_cuda_driver.py:        self.assertIn("Legacy default CUDA stream", repr(ds))
numba/cuda/tests/cudadrv/test_cuda_driver.py:    def test_cuda_driver_per_thread_default_stream(self):
numba/cuda/tests/cudadrv/test_cuda_driver.py:        self.assertIn("Per-thread default CUDA stream", repr(ds))
numba/cuda/tests/cudadrv/test_cuda_driver.py:    def test_cuda_driver_stream(self):
numba/cuda/tests/cudadrv/test_cuda_driver.py:        self.assertIn("CUDA stream", repr(s))
numba/cuda/tests/cudadrv/test_cuda_driver.py:    def test_cuda_driver_external_stream(self):
numba/cuda/tests/cudadrv/test_cuda_driver.py:        self.assertIn("External CUDA stream", repr(s))
numba/cuda/tests/cudadrv/test_cuda_driver.py:    def test_cuda_driver_occupancy(self):
numba/cuda/tests/cudadrv/test_cuda_driver.py:class TestDevice(CUDATestCase):
numba/cuda/tests/cudadrv/test_cuda_driver.py:        #     GPU-e6489c45-5b68-3b03-bab7-0e7c8e809643
numba/cuda/tests/cudadrv/test_cuda_driver.py:        uuid_format = f'^GPU-{h8}-{h4}-{h4}-{h4}-{h12}$'
numba/cuda/tests/cudadrv/test_cuda_libraries.py:from numba.cuda.testing import unittest
numba/cuda/tests/cudadrv/test_cuda_libraries.py:from numba.cuda.testing import skip_on_cudasim, skip_unless_conda_cudatoolkit
numba/cuda/tests/cudadrv/test_cuda_libraries.py:@skip_on_cudasim('Library detection unsupported in the simulator')
numba/cuda/tests/cudadrv/test_cuda_libraries.py:@skip_unless_conda_cudatoolkit
numba/cuda/tests/cudadrv/test_cuda_libraries.py:        This test is solely present to ensure that shipped cudatoolkits have
numba/cuda/tests/cudadrv/test_detect.py:from numba import cuda
numba/cuda/tests/cudadrv/test_detect.py:from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
numba/cuda/tests/cudadrv/test_detect.py:                                skip_under_cuda_memcheck)
numba/cuda/tests/cudadrv/test_detect.py:class TestCudaDetect(CUDATestCase):
numba/cuda/tests/cudadrv/test_detect.py:    def test_cuda_detect(self):
numba/cuda/tests/cudadrv/test_detect.py:            cuda.detect()
numba/cuda/tests/cudadrv/test_detect.py:        self.assertIn('CUDA devices', output)
numba/cuda/tests/cudadrv/test_detect.py:@skip_under_cuda_memcheck('Hangs cuda-memcheck')
numba/cuda/tests/cudadrv/test_detect.py:class TestCUDAFindLibs(CUDATestCase):
numba/cuda/tests/cudadrv/test_detect.py:            from numba import cuda
numba/cuda/tests/cudadrv/test_detect.py:            @cuda.jit('(int64,)')
numba/cuda/tests/cudadrv/test_detect.py:    @skip_on_cudasim('Simulator does not hit device library search code path')
numba/cuda/tests/cudadrv/test_detect.py:    def test_cuda_find_lib_errors(self):
numba/cuda/tests/cudadrv/test_detect.py:            out, err = self.run_test_in_separate_process("NUMBA_CUDA_DRIVER",
numba/cuda/tests/cudadrv/test_context_stack.py:from numba import cuda
numba/cuda/tests/cudadrv/test_context_stack.py:from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
numba/cuda/tests/cudadrv/test_context_stack.py:from numba.cuda.cudadrv import driver
numba/cuda/tests/cudadrv/test_context_stack.py:class TestContextStack(CUDATestCase):
numba/cuda/tests/cudadrv/test_context_stack.py:        cuda.close()
numba/cuda/tests/cudadrv/test_context_stack.py:    def test_gpus_current(self):
numba/cuda/tests/cudadrv/test_context_stack.py:        self.assertIs(cuda.gpus.current, None)
numba/cuda/tests/cudadrv/test_context_stack.py:        with cuda.gpus[0]:
numba/cuda/tests/cudadrv/test_context_stack.py:            self.assertEqual(int(cuda.gpus.current.id), 0)
numba/cuda/tests/cudadrv/test_context_stack.py:    def test_gpus_len(self):
numba/cuda/tests/cudadrv/test_context_stack.py:        self.assertGreater(len(cuda.gpus), 0)
numba/cuda/tests/cudadrv/test_context_stack.py:    def test_gpus_iter(self):
numba/cuda/tests/cudadrv/test_context_stack.py:        gpulist = list(cuda.gpus)
numba/cuda/tests/cudadrv/test_context_stack.py:        self.assertGreater(len(gpulist), 0)
numba/cuda/tests/cudadrv/test_context_stack.py:class TestContextAPI(CUDATestCase):
numba/cuda/tests/cudadrv/test_context_stack.py:        cuda.close()
numba/cuda/tests/cudadrv/test_context_stack.py:            mem = cuda.current_context().get_memory_info()
numba/cuda/tests/cudadrv/test_context_stack.py:    @unittest.skipIf(len(cuda.gpus) < 2, "need more than 1 gpus")
numba/cuda/tests/cudadrv/test_context_stack.py:    @skip_on_cudasim('CUDA HW required')
numba/cuda/tests/cudadrv/test_context_stack.py:        # Cannot switch context inside a `cuda.require_context`
numba/cuda/tests/cudadrv/test_context_stack.py:        @cuda.require_context
numba/cuda/tests/cudadrv/test_context_stack.py:        def switch_gpu():
numba/cuda/tests/cudadrv/test_context_stack.py:            with cuda.gpus[1]:
numba/cuda/tests/cudadrv/test_context_stack.py:        with cuda.gpus[0]:
numba/cuda/tests/cudadrv/test_context_stack.py:                switch_gpu()
numba/cuda/tests/cudadrv/test_context_stack.py:            self.assertIn("Cannot switch CUDA-context.", str(raises.exception))
numba/cuda/tests/cudadrv/test_context_stack.py:    @unittest.skipIf(len(cuda.gpus) < 2, "need more than 1 gpus")
numba/cuda/tests/cudadrv/test_context_stack.py:        def switch_gpu():
numba/cuda/tests/cudadrv/test_context_stack.py:            with cuda.gpus[1]:
numba/cuda/tests/cudadrv/test_context_stack.py:                return cuda.current_context().device.id
numba/cuda/tests/cudadrv/test_context_stack.py:        with cuda.gpus[0]:
numba/cuda/tests/cudadrv/test_context_stack.py:            devid = switch_gpu()
numba/cuda/tests/cudadrv/test_context_stack.py:@skip_on_cudasim('CUDA HW required')
numba/cuda/tests/cudadrv/test_context_stack.py:class Test3rdPartyContext(CUDATestCase):
numba/cuda/tests/cudadrv/test_context_stack.py:        cuda.close()
numba/cuda/tests/cudadrv/test_context_stack.py:            my_ctx = cuda.current_context()
numba/cuda/tests/cudadrv/test_context_stack.py:            cuda.current_context()
numba/cuda/tests/cudadrv/test_context_stack.py:            # Expecting an error about non-primary CUDA context
numba/cuda/tests/cudadrv/test_context_stack.py:            self.assertIn("Numba cannot operate on non-primary CUDA context ",
numba/cuda/tests/cudadrv/test_context_stack.py:    def test_cudajit_in_attached_primary_context(self):
numba/cuda/tests/cudadrv/test_context_stack.py:            from numba import cuda
numba/cuda/tests/cudadrv/test_context_stack.py:            @cuda.jit
numba/cuda/tests/cudadrv/test_context_stack.py:            a = cuda.device_array(10)
numba/cuda/tests/cudadrv/test_ptds.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudadrv/test_ptds.py:from numba.cuda.testing import (skip_on_cudasim, skip_with_cuda_python,
numba/cuda/tests/cudadrv/test_ptds.py:                                skip_under_cuda_memcheck)
numba/cuda/tests/cudadrv/test_ptds.py:    from numba import cuda, int32, void
numba/cuda/tests/cudadrv/test_ptds.py:    # Enable PTDS before we make any CUDA driver calls.  Enabling it first
numba/cuda/tests/cudadrv/test_ptds.py:    # ensures that PTDS APIs are used because the CUDA driver looks up API
numba/cuda/tests/cudadrv/test_ptds.py:    config.CUDA_PER_THREAD_DEFAULT_STREAM = 1
numba/cuda/tests/cudadrv/test_ptds.py:    cudadrv_logger = logging.getLogger('numba.cuda.cudadrv.driver')
numba/cuda/tests/cudadrv/test_ptds.py:    cudadrv_logger.addHandler(handler)
numba/cuda/tests/cudadrv/test_ptds.py:    cudadrv_logger.setLevel(logging.DEBUG)
numba/cuda/tests/cudadrv/test_ptds.py:    xs = [cuda.to_device(x) for _ in range(N_THREADS)]
numba/cuda/tests/cudadrv/test_ptds.py:    rs = [cuda.to_device(r) for _ in range(N_THREADS)]
numba/cuda/tests/cudadrv/test_ptds.py:    stream = cuda.default_stream()
numba/cuda/tests/cudadrv/test_ptds.py:    @cuda.jit(void(int32[::1], int32[::1]))
numba/cuda/tests/cudadrv/test_ptds.py:        i = cuda.grid(1)
numba/cuda/tests/cudadrv/test_ptds.py:    cuda.synchronize()
numba/cuda/tests/cudadrv/test_ptds.py:@skip_under_cuda_memcheck('Hangs cuda-memcheck')
numba/cuda/tests/cudadrv/test_ptds.py:@skip_on_cudasim('Streams not supported on the simulator')
numba/cuda/tests/cudadrv/test_ptds.py:class TestPTDS(CUDATestCase):
numba/cuda/tests/cudadrv/test_ptds.py:    @skip_with_cuda_python('Function names unchanged for PTDS with NV Binding')
numba/cuda/tests/cudadrv/test_reset_device.py:from numba import cuda
numba/cuda/tests/cudadrv/test_reset_device.py:from numba.cuda.cudadrv.driver import driver
numba/cuda/tests/cudadrv/test_reset_device.py:from numba.cuda.testing import unittest, ContextResettingTestCase
numba/cuda/tests/cudadrv/test_reset_device.py:                        cuda.select_device(d)
numba/cuda/tests/cudadrv/test_reset_device.py:                        cuda.close()
numba/cuda/tests/cudadrv/test_cuda_memory.py:from numba.cuda.cudadrv import driver, drvapi, devices
numba/cuda/tests/cudadrv/test_cuda_memory.py:from numba.cuda.testing import unittest, ContextResettingTestCase
numba/cuda/tests/cudadrv/test_cuda_memory.py:from numba.cuda.testing import skip_on_cudasim
numba/cuda/tests/cudadrv/test_cuda_memory.py:@skip_on_cudasim('CUDA Memory API unsupported in the simulator')
numba/cuda/tests/cudadrv/test_cuda_memory.py:class TestCudaMemory(ContextResettingTestCase):
numba/cuda/tests/cudadrv/test_cuda_memory.py:        super(TestCudaMemory, self).tearDown()
numba/cuda/tests/cudadrv/test_cuda_memory.py:class TestCudaMemoryFunctions(ContextResettingTestCase):
numba/cuda/tests/cudadrv/test_cuda_memory.py:        super(TestCudaMemoryFunctions, self).tearDown()
numba/cuda/tests/cudadrv/test_cuda_memory.py:@skip_on_cudasim('CUDA Memory API unsupported in the simulator')
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:from numba import cuda
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:class CudaArrayIndexing(CUDATestCase):
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:class CudaArrayStridedSlice(CUDATestCase):
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:class CudaArraySlicing(CUDATestCase):
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        da = cuda.to_device(a)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        da = cuda.to_device(a)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:class CudaArraySetting(CUDATestCase):
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(np.arange(5 * 7))
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(np.arange(5))
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:    @skip_on_cudasim('cudasim does not use streams and operates synchronously')
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        darr = cuda.to_device(np.arange(5))
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        with patch.object(cuda.cudadrv.driver.Stream, 'synchronize',
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:    @skip_on_cudasim('cudasim does not use streams and operates synchronously')
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        streams = (cuda.stream(), cuda.default_stream(),
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:                   cuda.legacy_default_stream(),
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:                   cuda.per_thread_default_stream())
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:            darr = cuda.to_device(np.arange(5), stream=stream)
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:            with patch.object(cuda.cudadrv.driver.Stream, 'synchronize',
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:    @skip_on_cudasim('cudasim does not use streams and operates synchronously')
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        streams = (cuda.stream(), cuda.default_stream(),
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:                   cuda.legacy_default_stream(),
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:                   cuda.per_thread_default_stream())
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:            darr = cuda.to_device(np.arange(5))
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:            with patch.object(cuda.cudadrv.driver.Stream, 'synchronize',
numba/cuda/tests/cudadrv/test_cuda_array_slicing.py:        ary = cuda.mapped_array(2, dtype=np.int32)
numba/cuda/tests/cudadrv/test_inline_ptx.py:from numba.cuda.cudadrv import nvvm
numba/cuda/tests/cudadrv/test_inline_ptx.py:from numba.cuda.testing import unittest, ContextResettingTestCase
numba/cuda/tests/cudadrv/test_inline_ptx.py:from numba.cuda.testing import skip_on_cudasim
numba/cuda/tests/cudadrv/test_inline_ptx.py:@skip_on_cudasim('Inline PTX cannot be used in the simulator')
numba/cuda/tests/cudadrv/test_inline_ptx.py:class TestCudaInlineAsm(ContextResettingTestCase):
numba/cuda/tests/cudadrv/test_inline_ptx.py:        mod.triple = 'nvptx64-nvidia-cuda'
numba/cuda/tests/cudadrv/test_inline_ptx.py:        nvvm.set_cuda_kernel(fn)
numba/cuda/tests/cudadrv/test_managed_alloc.py:from numba.cuda.cudadrv.driver import device_memset, driver, USE_NV_BINDING
numba/cuda/tests/cudadrv/test_managed_alloc.py:from numba import cuda
numba/cuda/tests/cudadrv/test_managed_alloc.py:from numba.cuda.testing import unittest, ContextResettingTestCase
numba/cuda/tests/cudadrv/test_managed_alloc.py:from numba.cuda.testing import skip_on_cudasim, skip_on_arm
numba/cuda/tests/cudadrv/test_managed_alloc.py:@skip_on_cudasim('CUDA Driver API unsupported in the simulator')
numba/cuda/tests/cudadrv/test_managed_alloc.py:    def get_total_gpu_memory(self):
numba/cuda/tests/cudadrv/test_managed_alloc.py:        # We use a driver function to directly get the total GPU memory because
numba/cuda/tests/cudadrv/test_managed_alloc.py:        ctx = cuda.current_context()
numba/cuda/tests/cudadrv/test_managed_alloc.py:    # CUDA Unified Memory comes in two flavors. For GPUs in the Kepler and
numba/cuda/tests/cudadrv/test_managed_alloc.py:    # GPUs in the Pascal or later generations, managed memory operates on a
numba/cuda/tests/cudadrv/test_managed_alloc.py:    # per-page basis, so we can have arrays larger than GPU memory, where only
numba/cuda/tests/cudadrv/test_managed_alloc.py:    # test works correctly on all supported GPUs, we'll select the size of our
numba/cuda/tests/cudadrv/test_managed_alloc.py:    # memory such that we only oversubscribe the GPU memory if we're on a
numba/cuda/tests/cudadrv/test_managed_alloc.py:    # Pascal or newer GPU (compute capability at least 6.0).
numba/cuda/tests/cudadrv/test_managed_alloc.py:    # for a very long time or get OOM killed if the GPU memory size is >50% of
numba/cuda/tests/cudadrv/test_managed_alloc.py:    # of the GPU, this test runs for a very long time (in comparison to the
numba/cuda/tests/cudadrv/test_managed_alloc.py:        # memory through the CUDA driver interface.
numba/cuda/tests/cudadrv/test_managed_alloc.py:        total_mem_size = self.get_total_gpu_memory()
numba/cuda/tests/cudadrv/test_managed_alloc.py:        ctx = cuda.current_context()
numba/cuda/tests/cudadrv/test_managed_alloc.py:        # test effectively drives both the CPU and the GPU on
numba/cuda/tests/cudadrv/test_managed_alloc.py:        ary = cuda.managed_array(100, dtype=np.double)
numba/cuda/tests/cudadrv/test_managed_alloc.py:        @cuda.jit('void(double[:])')
numba/cuda/tests/cudadrv/test_managed_alloc.py:            i = cuda.grid(1)
numba/cuda/tests/cudadrv/test_managed_alloc.py:        cuda.current_context().synchronize()
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:from numba.cuda.cudadrv import devicearray
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:from numba import cuda
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:from numba.cuda.testing import skip_on_cudasim
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:class TestCudaNDArray(CUDATestCase):
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        dary = cuda.device_array(shape=100)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        devicearray.verify_cuda_ndarray_interface(dary)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        dary = cuda.to_device(ary)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        devicearray.verify_cuda_ndarray_interface(dary)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        dary = cuda.to_device(ary)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        devicearray.verify_cuda_ndarray_interface(dary)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        dary = cuda.to_device(ary)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        dary = cuda.device_array(shape=(100,), dtype="f4")
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        cuda.to_device(array, copy=False)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        dary = cuda.to_device(ary)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        gpumem = cuda.to_device(array)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        gpumem.copy_to_host(array)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        stream = cuda.stream()
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:            arr = cuda.device_array(
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        dary = cuda.device_array(3)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        dary = cuda.device_array((3, 5))
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        dary = cuda.device_array((3, 5, 7))
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        gpumem = cuda.to_device(array)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        left, right = gpumem.split(N // 2)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        gpumem = cuda.to_device(array)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        cuda.to_device(array * 2, to=gpumem)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        gpumem.copy_to_host(array)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:    @skip_on_cudasim('This works in the simulator')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        gpumem = cuda.to_device(np.array(np.arange(12)).reshape(3, 4, 1))
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:            np.transpose(gpumem)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        array = np.transpose(cuda.to_device(original),
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        gpumem = cuda.to_device(np.array(np.arange(12)).reshape(3, 4))
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:            np.transpose(gpumem, axes=(0, 0))
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:                'invalid axes list (0, 0)',  # GPU
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        gpumem = cuda.to_device(np.array(np.arange(12)).reshape(3, 4))
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:            np.transpose(gpumem, axes=(0, 2))
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:                'invalid axes list (0, 2)',  # GPU
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        array = cuda.to_device(original)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        array = cuda.to_device(original)[:, ::2]
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        array = cuda.to_device(original)[:, ::2]
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        array = cuda.to_device(original)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        array = np.transpose(cuda.to_device(original)).copy_to_host()
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        array = cuda.to_device(original).T.copy_to_host()
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d = cuda.to_device(a)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:            d = cuda.to_device(original)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d = cuda.to_device(a_c)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:            d.copy_to_device(cuda.to_device(a_f))
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d.copy_to_device(cuda.to_device(a_c))
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d = cuda.to_device(a_f)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:            d.copy_to_device(cuda.to_device(a_c))
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d.copy_to_device(cuda.to_device(a_f))
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:            dbroad_c = cuda.to_device(broad_c)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:            dbroad_f = cuda.to_device(broad_f)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d = cuda.to_device(a_c)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d = cuda.to_device(np.arange(20))
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:            d.copy_to_device(cuda.to_device(arr)[::2])
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:    @skip_on_cudasim('DeviceNDArray class not present in simulator')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:            d_arr = cuda.to_device(arr)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:    @skip_on_cudasim('Typing not done in the simulator')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d = cuda.to_device(a)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:    @skip_on_cudasim('Typing not done in the simulator')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d = cuda.to_device(a)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:    @skip_on_cudasim('Typing not done in the simulator')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d = cuda.to_device(a)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:    @skip_on_cudasim('Typing not done in the simulator')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d = cuda.to_device(a)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:    @skip_on_cudasim('Typing not done in the simulator')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d = cuda.to_device(a)[:,2]
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:    @skip_on_cudasim('Typing not done in the simulator')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d = cuda.to_device(a)[2,:]
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:    @skip_on_cudasim('Typing not done in the simulator')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d = cuda.to_device(a)[2,:]
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:    @skip_on_cudasim('Typing not done in the simulator')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d = cuda.to_device(a)[:,2]
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:    @skip_on_cudasim('Typing not done in the simulator')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d = cuda.to_device(a)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        dary = cuda.to_device(ary)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:    @skip_on_cudasim('DeviceNDArray class not present in simulator')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        dev_array_from_host = cuda.to_device(host_array)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:class TestRecarray(CUDATestCase):
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:            i = cuda.grid(1)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        cuda.jit(test)[1, a.size](a, got1, got2)
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:class TestCoreContiguous(CUDATestCase):
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d_a = cuda.device_array(10, order='C')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d_a = cuda.device_array((10, 12), order='C')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d_a = cuda.device_array((10, 12), order='C')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d_a = cuda.device_array((10, 12, 14), order='C')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d_a = cuda.device_array(10, order='F')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d_a = cuda.device_array((10, 12), order='F')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d_a = cuda.device_array((10, 12), order='F')
numba/cuda/tests/cudadrv/test_cuda_ndarray.py:        d_a = cuda.device_array((10, 12, 14), order='F')
numba/cuda/tests/cudadrv/test_pinned.py:from numba import cuda
numba/cuda/tests/cudadrv/test_pinned.py:from numba.cuda.testing import unittest, ContextResettingTestCase
numba/cuda/tests/cudadrv/test_pinned.py:        stream = cuda.stream()
numba/cuda/tests/cudadrv/test_pinned.py:        ptr = cuda.to_device(A, copy=False, stream=stream)
numba/cuda/tests/cudadrv/test_pinned.py:        with cuda.pinned(A):
numba/cuda/tests/cudadrv/test_mvc.py:from numba.cuda.testing import unittest, CUDATestCase
numba/cuda/tests/cudadrv/test_mvc.py:from numba.cuda.testing import (skip_on_cudasim, skip_under_cuda_memcheck,
numba/cuda/tests/cudadrv/test_mvc.py:    from numba import config, cuda
numba/cuda/tests/cudadrv/test_mvc.py:    # Change the MVC config after importing numba.cuda
numba/cuda/tests/cudadrv/test_mvc.py:    config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = 1
numba/cuda/tests/cudadrv/test_mvc.py:    @cuda.jit
numba/cuda/tests/cudadrv/test_mvc.py:@skip_under_cuda_memcheck('May hang CUDA memcheck')
numba/cuda/tests/cudadrv/test_mvc.py:@skip_on_cudasim('Simulator does not require or implement MVC')
numba/cuda/tests/cudadrv/test_mvc.py:class TestMinorVersionCompatibility(CUDATestCase):
numba/cuda/tests/cudadrv/test_array_attr.py:from numba import cuda
numba/cuda/tests/cudadrv/test_array_attr.py:from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
numba/cuda/tests/cudadrv/test_array_attr.py:class TestArrayAttr(CUDATestCase):
numba/cuda/tests/cudadrv/test_array_attr.py:        dcary = cuda.to_device(cary)
numba/cuda/tests/cudadrv/test_array_attr.py:        dfary = cuda.to_device(fary)
numba/cuda/tests/cudadrv/test_array_attr.py:        dcary = cuda.to_device(cary)
numba/cuda/tests/cudadrv/test_array_attr.py:        dfary = cuda.to_device(fary)
numba/cuda/tests/cudadrv/test_array_attr.py:        dcary = cuda.to_device(cary)
numba/cuda/tests/cudadrv/test_array_attr.py:        dfary = cuda.to_device(fary)
numba/cuda/tests/cudadrv/test_array_attr.py:        dary = cuda.to_device(ary)
numba/cuda/tests/cudadrv/test_array_attr.py:    @skip_on_cudasim('CUDA Array Interface is not supported in the simulator')
numba/cuda/tests/cudadrv/test_array_attr.py:        dary = cuda.to_device(ary)
numba/cuda/tests/cudadrv/test_array_attr.py:        dary_data = dary.__cuda_array_interface__['data'][0]
numba/cuda/tests/cudadrv/test_array_attr.py:        ddarystride_data = darystride.__cuda_array_interface__['data'][0]
numba/cuda/tests/cudadrv/test_array_attr.py:        dary = cuda.to_device(reshaped)
numba/cuda/tests/cudadrv/test_array_attr.py:            dary = cuda.to_device(reshaped)
numba/cuda/tests/cudadrv/test_array_attr.py:    @skip_on_cudasim('CUDA Array Interface is not supported in the simulator')
numba/cuda/tests/cudadrv/test_array_attr.py:        dary = cuda.to_device(reshaped)
numba/cuda/tests/cudadrv/test_array_attr.py:        dary_data = dary.__cuda_array_interface__['data'][0]
numba/cuda/tests/cudadrv/test_array_attr.py:        ddarystride_data = darystride.__cuda_array_interface__['data'][0]
numba/cuda/tests/cudadrv/test_array_attr.py:            dary = cuda.to_device(reshaped)
numba/cuda/tests/cudadrv/test_array_attr.py:    @skip_on_cudasim('CUDA Array Interface is not supported in the simulator')
numba/cuda/tests/cudadrv/test_array_attr.py:        dary = cuda.to_device(reshaped)
numba/cuda/tests/cudadrv/test_array_attr.py:        dary_data = dary.__cuda_array_interface__['data'][0]
numba/cuda/tests/cudadrv/test_array_attr.py:        ddarystride_data = darystride.__cuda_array_interface__['data'][0]
numba/cuda/tests/cudadrv/test_array_attr.py:        dary = cuda.to_device(ary)
numba/cuda/tests/cudadrv/test_array_attr.py:        dary = cuda.to_device(ary)
numba/cuda/tests/cudadrv/test_nvvm_driver.py:from numba.cuda.cudadrv import nvvm, runtime
numba/cuda/tests/cudadrv/test_nvvm_driver.py:from numba.cuda.testing import unittest
numba/cuda/tests/cudadrv/test_nvvm_driver.py:from numba.cuda.cudadrv.nvvm import LibDevice, NvvmError, NVVM
numba/cuda/tests/cudadrv/test_nvvm_driver.py:from numba.cuda.testing import skip_on_cudasim
numba/cuda/tests/cudadrv/test_nvvm_driver.py:@skip_on_cudasim('NVVM Driver unsupported in the simulator')
numba/cuda/tests/cudadrv/test_nvvm_driver.py:        # -gen-lto is not available prior to CUDA 11.5
numba/cuda/tests/cudadrv/test_nvvm_driver.py:        m.triple = 'nvptx64-nvidia-cuda'
numba/cuda/tests/cudadrv/test_nvvm_driver.py:        kernel = ir.Function(m, fty, name='mycudakernel')
numba/cuda/tests/cudadrv/test_nvvm_driver.py:        nvvm.set_cuda_kernel(kernel)
numba/cuda/tests/cudadrv/test_nvvm_driver.py:        self.assertTrue('mycudakernel' in ptx)
numba/cuda/tests/cudadrv/test_nvvm_driver.py:        m.triple = 'nvptx64-nvidia-cuda'
numba/cuda/tests/cudadrv/test_nvvm_driver.py:        kernel = ir.Function(m, fty, name='mycudakernel')
numba/cuda/tests/cudadrv/test_nvvm_driver.py:        nvvm.set_cuda_kernel(kernel)
numba/cuda/tests/cudadrv/test_nvvm_driver.py:        self.assertIn("mycudakernel", used_line)
numba/cuda/tests/cudadrv/test_nvvm_driver.py:        m.triple = 'nvptx64-nvidia-cuda'
numba/cuda/tests/cudadrv/test_nvvm_driver.py:        nvvm.set_cuda_kernel(kernel)
numba/cuda/tests/cudadrv/test_nvvm_driver.py:@skip_on_cudasim('NVVM Driver unsupported in the simulator')
numba/cuda/tests/cudadrv/test_nvvm_driver.py:@skip_on_cudasim('NVVM Driver unsupported in the simulator')
numba/cuda/tests/cudadrv/test_nvvm_driver.py:target triple="nvptx64-nvidia-cuda"
numba/cuda/tests/cudadrv/__init__.py:from numba.cuda.testing import ensure_supported_ccs_initialized
numba/cuda/tests/cudadrv/test_emm_plugins.py:from numba import cuda
numba/cuda/tests/cudadrv/test_emm_plugins.py:from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
numba/cuda/tests/cudadrv/test_emm_plugins.py:if not config.ENABLE_CUDASIM:
numba/cuda/tests/cudadrv/test_emm_plugins.py:    class DeviceOnlyEMMPlugin(cuda.HostOnlyCUDAMemoryManager):
numba/cuda/tests/cudadrv/test_emm_plugins.py:            return cuda.cudadrv.driver.AutoFreePointer(ctx, ptr, size,
numba/cuda/tests/cudadrv/test_emm_plugins.py:            return cuda.MemoryInfo(free=32, total=64)
numba/cuda/tests/cudadrv/test_emm_plugins.py:@skip_on_cudasim('EMM Plugins not supported on CUDA simulator')
numba/cuda/tests/cudadrv/test_emm_plugins.py:class TestDeviceOnlyEMMPlugin(CUDATestCase):
numba/cuda/tests/cudadrv/test_emm_plugins.py:        cuda.close()
numba/cuda/tests/cudadrv/test_emm_plugins.py:        cuda.set_memory_manager(DeviceOnlyEMMPlugin)
numba/cuda/tests/cudadrv/test_emm_plugins.py:        cuda.close()
numba/cuda/tests/cudadrv/test_emm_plugins.py:        cuda.cudadrv.driver._memory_manager = None
numba/cuda/tests/cudadrv/test_emm_plugins.py:        mgr = cuda.current_context().memory_manager
numba/cuda/tests/cudadrv/test_emm_plugins.py:        d_arr_1 = cuda.device_array_like(arr_1)
numba/cuda/tests/cudadrv/test_emm_plugins.py:        d_arr_2 = cuda.device_array_like(arr_2)
numba/cuda/tests/cudadrv/test_emm_plugins.py:        # If we have a CUDA context, it should already have initialized its
numba/cuda/tests/cudadrv/test_emm_plugins.py:        self.assertTrue(cuda.current_context().memory_manager.initialized)
numba/cuda/tests/cudadrv/test_emm_plugins.py:        ctx = cuda.current_context()
numba/cuda/tests/cudadrv/test_emm_plugins.py:        ctx = cuda.current_context()
numba/cuda/tests/cudadrv/test_emm_plugins.py:        d_arr = cuda.device_array_like(arr)
numba/cuda/tests/cudadrv/test_emm_plugins.py:        ctx = cuda.current_context()
numba/cuda/tests/cudadrv/test_emm_plugins.py:@skip_on_cudasim('EMM Plugins not supported on CUDA simulator')
numba/cuda/tests/cudadrv/test_emm_plugins.py:class TestBadEMMPluginVersion(CUDATestCase):
numba/cuda/tests/cudadrv/test_emm_plugins.py:            cuda.set_memory_manager(BadVersionEMMPlugin)
numba/cuda/tests/cudasim/test_cudasim_issues.py:from numba import cuda
numba/cuda/tests/cudasim/test_cudasim_issues.py:from numba.cuda.testing import CUDATestCase, skip_unless_cudasim
numba/cuda/tests/cudasim/test_cudasim_issues.py:import numba.cuda.simulator as simulator
numba/cuda/tests/cudasim/test_cudasim_issues.py:class TestCudaSimIssues(CUDATestCase):
numba/cuda/tests/cudasim/test_cudasim_issues.py:        @cuda.jit
numba/cuda/tests/cudasim/test_cudasim_issues.py:        @cuda.jit
numba/cuda/tests/cudasim/test_cudasim_issues.py:    def test_cuda_module_in_device_function(self):
numba/cuda/tests/cudasim/test_cudasim_issues.py:        When the `cuda` module is referenced in a device function,
numba/cuda/tests/cudasim/test_cudasim_issues.py:        it does not have the kernel API (e.g. cuda.threadIdx, cuda.shared)
numba/cuda/tests/cudasim/test_cudasim_issues.py:        from numba.cuda.tests.cudasim import support
numba/cuda/tests/cudasim/test_cudasim_issues.py:        inner = support.cuda_module_in_device_function
numba/cuda/tests/cudasim/test_cudasim_issues.py:        @cuda.jit
numba/cuda/tests/cudasim/test_cudasim_issues.py:    @skip_unless_cudasim('Only works on CUDASIM')
numba/cuda/tests/cudasim/test_cudasim_issues.py:            i = cuda.grid(1)
numba/cuda/tests/cudasim/test_cudasim_issues.py:            cuda.syncthreads()
numba/cuda/tests/cudasim/test_cudasim_issues.py:            cuda.syncthreads()
numba/cuda/tests/cudasim/support.py:from numba import cuda
numba/cuda/tests/cudasim/support.py:@cuda.jit(device=True)
numba/cuda/tests/cudasim/support.py:def cuda_module_in_device_function():
numba/cuda/tests/cudasim/support.py:    return cuda.threadIdx.x
numba/cuda/tests/__init__.py:from numba.cuda.testing import ensure_supported_ccs_initialized
numba/cuda/tests/__init__.py:from numba import cuda
numba/cuda/tests/__init__.py:    suite.addTests(load_testsuite(loader, join(this_dir, 'nocuda')))
numba/cuda/tests/__init__.py:    if cuda.is_available():
numba/cuda/tests/__init__.py:        suite.addTests(load_testsuite(loader, join(this_dir, 'cudasim')))
numba/cuda/tests/__init__.py:        gpus = cuda.list_devices()
numba/cuda/tests/__init__.py:        if gpus and gpus[0].compute_capability >= (2, 0):
numba/cuda/tests/__init__.py:            suite.addTests(load_testsuite(loader, join(this_dir, 'cudadrv')))
numba/cuda/tests/__init__.py:            suite.addTests(load_testsuite(loader, join(this_dir, 'cudapy')))
numba/cuda/tests/__init__.py:            print("skipped CUDA tests because GPU CC < 2.0")
numba/cuda/tests/__init__.py:        print("skipped CUDA tests")
numba/cuda/device_init.py:from numba.cuda import cg
numba/cuda/device_init.py:from .cudadrv.error import CudaSupportError
numba/cuda/device_init.py:from numba.cuda.cudadrv.driver import (BaseCUDAMemoryManager,
numba/cuda/device_init.py:                                       HostOnlyCUDAMemoryManager,
numba/cuda/device_init.py:from numba.cuda.cudadrv.runtime import runtime
numba/cuda/device_init.py:from .cudadrv import nvvm
numba/cuda/device_init.py:from numba.cuda import initialize
numba/cuda/device_init.py:    """Returns a boolean to indicate the availability of a CUDA GPU.
numba/cuda/device_init.py:    # test discovery/orchestration as `cuda.is_available` is often
numba/cuda/device_init.py:    # used as a guard for whether to run a CUDA test, the try/except
numba/cuda/device_init.py:    except CudaSupportError:
numba/cuda/device_init.py:    """Returns True if the CUDA Runtime is a supported version.
numba/cuda/device_init.py:    - Generating an error or otherwise preventing the use of CUDA.
numba/cuda/device_init.py:def cuda_error():
numba/cuda/device_init.py:    """Returns None if there was no error initializing the CUDA driver.
numba/cuda/libdeviceimpl.py:from numba.cuda import libdevice, libdevicefuncs
numba/cuda/api.py:API that are reported to numba.cuda
numba/cuda/api.py:from .cudadrv import devicearray, devices, driver
numba/cuda/api.py:from numba.cuda.api_util import prepare_shape_strides_dtype
numba/cuda/api.py:gpus = devices.gpus
numba/cuda/api.py:def from_cuda_array_interface(desc, owner=None, sync=True):
numba/cuda/api.py:    """Create a DeviceNDArray from a cuda-array-interface description.
numba/cuda/api.py:        if sync and config.CUDA_ARRAY_INTERFACE_SYNC:
numba/cuda/api.py:        stream = 0 # No "Numba default stream", not the CUDA default stream
numba/cuda/api.py:                                   dtype=dtype, gpu_data=data,
numba/cuda/api.py:def as_cuda_array(obj, sync=True):
numba/cuda/api.py:    the :ref:`cuda array interface <cuda-array-interface>`.
numba/cuda/api.py:    A view of the underlying GPU buffer is created.  No copying of the data
numba/cuda/api.py:    if not is_cuda_array(obj):
numba/cuda/api.py:        raise TypeError("*obj* doesn't implement the cuda array interface.")
numba/cuda/api.py:        return from_cuda_array_interface(obj.__cuda_array_interface__,
numba/cuda/api.py:def is_cuda_array(obj):
numba/cuda/api.py:    """Test if the object has defined the `__cuda_array_interface__` attribute.
numba/cuda/api.py:    return hasattr(obj, '__cuda_array_interface__')
numba/cuda/api.py:        d_ary = cuda.to_device(ary)
numba/cuda/api.py:        stream = cuda.stream()
numba/cuda/api.py:        d_ary = cuda.to_device(ary, stream=stream)
numba/cuda/api.py:    Call :func:`device_array() <numba.cuda.device_array>` with information from
numba/cuda/api.py:    Call :func:`mapped_array() <numba.cuda.mapped_array>` with the information
numba/cuda/api.py:    Call :func:`pinned_array() <numba.cuda.pinned_array>` with the information
numba/cuda/api.py:    Create a CUDA stream that represents a command queue for the device.
numba/cuda/api.py:    Get the default CUDA stream. CUDA semantics in general are that the default
numba/cuda/api.py:    depending on which CUDA APIs are in use. In Numba, the APIs for the legacy
numba/cuda/api.py:    Get the legacy default CUDA stream.
numba/cuda/api.py:    Get the per-thread default CUDA stream.
numba/cuda/api.py:        devary = devicearray.from_array_like(ary, gpu_data=pm, stream=stream)
numba/cuda/api.py:        # When exiting from `with cuda.mapped(*arrs) as mapped_arrs:`, the name
numba/cuda/api.py:    Create a CUDA event. Timing data is only recorded by the event if it is
numba/cuda/api.py:    return devices.gpus
numba/cuda/api.py:    Detect supported CUDA hardware and print a summary of the detected hardware.
numba/cuda/api.py:    print('Found %d CUDA devices' % len(devlist))
numba/cuda/cudadrv/_extras.c: * Helper binding to call some CUDA Runtime API that cannot be directly
numba/cuda/cudadrv/_extras.c:#define CUDA_IPC_HANDLE_SIZE 64
numba/cuda/cudadrv/_extras.c:    char reserved[CUDA_IPC_HANDLE_SIZE];
numba/cuda/cudadrv/_extras.c:    PyModule_AddIntConstant(m, "CUDA_IPC_HANDLE_SIZE", CUDA_IPC_HANDLE_SIZE);
numba/cuda/cudadrv/drvapi.py:from numba.cuda.cudadrv import _extras
numba/cuda/cudadrv/drvapi.py:cu_ipc_mem_handle = (c_byte * _extras.CUDA_IPC_HANDLE_SIZE)   # 64 bytes wide
numba/cuda/cudadrv/drvapi.py:# See https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html
numba/cuda/cudadrv/drvapi.py:    # CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction hfunc,
numba/cuda/cudadrv/drvapi.py:    #    CUresult CUDAAPI
numba/cuda/cudadrv/drvapi.py:    #    CUresult CUDAAPI
numba/cuda/cudadrv/drvapi.py:    # CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(
numba/cuda/cudadrv/drvapi.py:    # CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
numba/cuda/cudadrv/drvapi.py:    # CUresult CUDAAPI cuOccupancyMaxPotentialBlockSize(
numba/cuda/cudadrv/drvapi.py:    # CUresult CUDAAPI cuOccupancyMaxPotentialBlockSizeWithFlags(
numba/cuda/cudadrv/ndarray.py:from numba.cuda.cudadrv import devices, driver
numba/cuda/cudadrv/ndarray.py:    Allocate gpu data buffer
numba/cuda/cudadrv/ndarray.py:    gpu_data = devices.get_context().memalloc(datasize)
numba/cuda/cudadrv/ndarray.py:    return gpu_data
numba/cuda/cudadrv/devices.py:Expose each GPU devices directly.
numba/cuda/cudadrv/devices.py:This module implements a API that is like the "CUDA runtime" context manager
numba/cuda/cudadrv/devices.py:for managing CUDA context stack and clean up.  It relies on thread-local globals
numba/cuda/cudadrv/devices.py:            # Query all CUDA devices.
numba/cuda/cudadrv/devices.py:            gpus = [_DeviceContextManager(driver.get_device(devid))
numba/cuda/cudadrv/devices.py:            self.lst = gpus
numba/cuda/cudadrv/devices.py:            return gpus
numba/cuda/cudadrv/devices.py:    ``numba.cuda.gpus``. For example, to execute on device 2::
numba/cuda/cudadrv/devices.py:       with numba.cuda.gpus[2]:
numba/cuda/cudadrv/devices.py:           d_a = numba.cuda.to_device(a)
numba/cuda/cudadrv/devices.py:    """Emulate the CUDA runtime context management.
numba/cuda/cudadrv/devices.py:        self.gpus = _DeviceList()
numba/cuda/cudadrv/devices.py:        # For caching the attached CUDA Context
numba/cuda/cudadrv/devices.py:        """Ensure a CUDA context is available inside the context.
numba/cuda/cudadrv/devices.py:        On entrance, queries the CUDA driver for an active CUDA context and
numba/cuda/cudadrv/devices.py:        the CUDA driver again.  On exit, detach the CUDA context from the TLS.
numba/cuda/cudadrv/devices.py:        This will allow us to pickup thirdparty activated CUDA context in
numba/cuda/cudadrv/devices.py:        any top-level Numba CUDA API.
numba/cuda/cudadrv/devices.py:        for *devnum*.  If *devnum* is None, use the active CUDA context (must
numba/cuda/cudadrv/devices.py:            # Try to get the active context in the CUDA stack or
numba/cuda/cudadrv/devices.py:            # activate GPU-0 with the primary context
numba/cuda/cudadrv/devices.py:                    ctx = self.gpus[ac.devnum].get_primary_context()
numba/cuda/cudadrv/devices.py:                               ' CUDA context {:x}')
numba/cuda/cudadrv/devices.py:            gpu = self.gpus[devnum]
numba/cuda/cudadrv/devices.py:            newctx = gpu.get_primary_context()
numba/cuda/cudadrv/devices.py:                raise RuntimeError('Cannot switch CUDA-context.')
numba/cuda/cudadrv/devices.py:        for gpu in self.gpus:
numba/cuda/cudadrv/devices.py:            gpu.reset()
numba/cuda/cudadrv/devices.py:gpus = _runtime.gpus
numba/cuda/cudadrv/devices.py:    return the CUDA context.
numba/cuda/cudadrv/devices.py:    A decorator that ensures a CUDA context is available when *fn* is executed.
numba/cuda/cudadrv/devices.py:    Note: The function *fn* cannot switch CUDA-context.
numba/cuda/cudadrv/devices.py:    def _require_cuda_context(*args, **kws):
numba/cuda/cudadrv/devices.py:    return _require_cuda_context
numba/cuda/cudadrv/devices.py:    """Reset the CUDA subsystem for the current thread.
numba/cuda/cudadrv/devices.py:    This removes all CUDA contexts.  Only use this at shutdown or for
numba/cuda/cudadrv/devices.py:    This clear the CUDA context stack only.
numba/cuda/cudadrv/driver.py:CUDA driver bridge implementation
numba/cuda/cudadrv/driver.py:crashing the system (particularly OSX) when the CUDA context is corrupted at
numba/cuda/cudadrv/driver.py:into the object destructor; thus, at corruption of the CUDA context,
numba/cuda/cudadrv/driver.py:subsequent deallocation could further corrupt the CUDA context and causes the
numba/cuda/cudadrv/driver.py:from .error import CudaSupportError, CudaDriverError
numba/cuda/cudadrv/driver.py:from numba.cuda.cudadrv import enums, drvapi, nvrtc, _extras
numba/cuda/cudadrv/driver.py:USE_NV_BINDING = config.CUDA_USE_NVIDIA_BINDING
numba/cuda/cudadrv/driver.py:    from cuda import cuda as binding
numba/cuda/cudadrv/driver.py:    # There is no definition of the default stream in the Nvidia bindings (nor
numba/cuda/cudadrv/driver.py:        lvl = str(config.CUDA_LOG_LEVEL).upper()
numba/cuda/cudadrv/driver.py:        if config.CUDA_LOG_LEVEL:
numba/cuda/cudadrv/driver.py:            fmt = '== CUDA [%(relativeCreated)d] %(levelname)5s -- %(message)s'
numba/cuda/cudadrv/driver.py:class CudaAPIError(CudaDriverError):
numba/cuda/cudadrv/driver.py:        super(CudaAPIError, self).__init__(code, msg)
numba/cuda/cudadrv/driver.py:    envpath = config.CUDA_DRIVER
numba/cuda/cudadrv/driver.py:        dlnames = ['nvcuda.dll']
numba/cuda/cudadrv/driver.py:        dldir = ['/usr/local/cuda/lib']
numba/cuda/cudadrv/driver.py:        dlnames = ['libcuda.dylib']
numba/cuda/cudadrv/driver.py:        dlnames = ['libcuda.so', 'libcuda.so.1']
numba/cuda/cudadrv/driver.py:            raise ValueError("NUMBA_CUDA_DRIVER %s is not a valid path" %
numba/cuda/cudadrv/driver.py:            raise ValueError("NUMBA_CUDA_DRIVER %s is not a valid file "
numba/cuda/cudadrv/driver.py:CUDA driver library cannot be found.
numba/cuda/cudadrv/driver.py:If you are sure that a CUDA driver is installed,
numba/cuda/cudadrv/driver.py:try setting environment variable NUMBA_CUDA_DRIVER
numba/cuda/cudadrv/driver.py:with the file path of the CUDA driver shared library.
numba/cuda/cudadrv/driver.py:Possible CUDA driver libraries are found but error occurred during load:
numba/cuda/cudadrv/driver.py:    raise CudaSupportError(DRIVER_NOT_FOUND_MSG)
numba/cuda/cudadrv/driver.py:    raise CudaSupportError(DRIVER_LOAD_ERROR_MSG % e)
numba/cuda/cudadrv/driver.py:    prefix = 'CUDA_ERROR'
numba/cuda/cudadrv/driver.py:            if config.DISABLE_CUDA:
numba/cuda/cudadrv/driver.py:                msg = ("CUDA is disabled due to setting NUMBA_DISABLE_CUDA=1 "
numba/cuda/cudadrv/driver.py:                       "in the environment, or because CUDA is unsupported on "
numba/cuda/cudadrv/driver.py:                raise CudaSupportError(msg)
numba/cuda/cudadrv/driver.py:        except CudaSupportError as e:
numba/cuda/cudadrv/driver.py:        except CudaAPIError as e:
numba/cuda/cudadrv/driver.py:            raise CudaSupportError(f"Error at driver init: {description}")
numba/cuda/cudadrv/driver.py:            raise CudaSupportError("Error at driver init: \n%s:" %
numba/cuda/cudadrv/driver.py:            return self._cuda_python_wrap_fn(fname)
numba/cuda/cudadrv/driver.py:        # Wrap a CUDA driver function by default
numba/cuda/cudadrv/driver.py:        def verbose_cuda_api_call(*args):
numba/cuda/cudadrv/driver.py:        def safe_cuda_api_call(*args):
numba/cuda/cudadrv/driver.py:        if config.CUDA_LOG_API_ARGS:
numba/cuda/cudadrv/driver.py:            wrapper = verbose_cuda_api_call
numba/cuda/cudadrv/driver.py:            wrapper = safe_cuda_api_call
numba/cuda/cudadrv/driver.py:    def _cuda_python_wrap_fn(self, fname):
numba/cuda/cudadrv/driver.py:        def verbose_cuda_api_call(*args):
numba/cuda/cudadrv/driver.py:            return self._check_cuda_python_error(fname, libfn(*args))
numba/cuda/cudadrv/driver.py:        def safe_cuda_api_call(*args):
numba/cuda/cudadrv/driver.py:            return self._check_cuda_python_error(fname, libfn(*args))
numba/cuda/cudadrv/driver.py:        if config.CUDA_LOG_API_ARGS:
numba/cuda/cudadrv/driver.py:            wrapper = verbose_cuda_api_call
numba/cuda/cudadrv/driver.py:            wrapper = safe_cuda_api_call
numba/cuda/cudadrv/driver.py:        # binding. For the NVidia binding, it handles linking to the correct
numba/cuda/cudadrv/driver.py:        if config.CUDA_PER_THREAD_DEFAULT_STREAM and not USE_NV_BINDING:
numba/cuda/cudadrv/driver.py:            raise CudaDriverError(f'Driver missing function: {fname}')
numba/cuda/cudadrv/driver.py:            msg = 'pid %s forked from pid %s after CUDA driver init'
numba/cuda/cudadrv/driver.py:            raise CudaDriverError("CUDA initialized before forking")
numba/cuda/cudadrv/driver.py:        if retcode != enums.CUDA_SUCCESS:
numba/cuda/cudadrv/driver.py:            errname = ERROR_MAP.get(retcode, "UNKNOWN_CUDA_ERROR")
numba/cuda/cudadrv/driver.py:            if retcode == enums.CUDA_ERROR_NOT_INITIALIZED:
numba/cuda/cudadrv/driver.py:            raise CudaAPIError(retcode, msg)
numba/cuda/cudadrv/driver.py:    def _check_cuda_python_error(self, fname, returned):
numba/cuda/cudadrv/driver.py:        if retcode != binding.CUresult.CUDA_SUCCESS:
numba/cuda/cudadrv/driver.py:            if retcode == binding.CUresult.CUDA_ERROR_NOT_INITIALIZED:
numba/cuda/cudadrv/driver.py:            raise CudaAPIError(retcode, msg)
numba/cuda/cudadrv/driver.py:        """Pop the active CUDA context and return the handle.
numba/cuda/cudadrv/driver.py:        If no CUDA context is active, return None.
numba/cuda/cudadrv/driver.py:        Returns the CUDA Runtime version as a tuple (major, minor).
numba/cuda/cudadrv/driver.py:    on querying the CUDA driver API.
numba/cuda/cudadrv/driver.py:    Once entering the context, it is assumed that the active CUDA context is
numba/cuda/cudadrv/driver.py:        """Returns True is there's a valid and active CUDA context.
numba/cuda/cudadrv/driver.py:    The device object owns the CUDA contexts.  This is owned by the driver
numba/cuda/cudadrv/driver.py:        fmt = f'GPU-{b4}-{b2}-{b2}-{b2}-{b6}'
numba/cuda/cudadrv/driver.py:        return "<CUDA device %d '%s'>" % (self.id, self.name)
numba/cuda/cudadrv/driver.py:        raise CudaSupportError("%s has compute capability < %s" %
numba/cuda/cudadrv/driver.py:class BaseCUDAMemoryManager(object, metaclass=ABCMeta):
numba/cuda/cudadrv/driver.py:                       CUDA address space.
numba/cuda/cudadrv/driver.py:        Return an IPC handle from a GPU allocation.
numba/cuda/cudadrv/driver.py:class HostOnlyCUDAMemoryManager(BaseCUDAMemoryManager):
numba/cuda/cudadrv/driver.py:    :class:`numba.cuda.BaseCUDAMemoryManager`) for its own internal state
numba/cuda/cudadrv/driver.py:    ``super()`` to give ``HostOnlyCUDAMemoryManager`` an opportunity to do the
numba/cuda/cudadrv/driver.py:        except CudaAPIError as e:
numba/cuda/cudadrv/driver.py:                oom_code = binding.CUresult.CUDA_ERROR_OUT_OF_MEMORY
numba/cuda/cudadrv/driver.py:                oom_code = enums.CUDA_ERROR_OUT_OF_MEMORY
numba/cuda/cudadrv/driver.py:        implementations - instead, use the :class:`BaseCUDAMemoryManager`.
numba/cuda/cudadrv/driver.py:        implementations - instead, use the :class:`BaseCUDAMemoryManager`.
numba/cuda/cudadrv/driver.py:        ``cuIpcGetMemHandle``. A :class:`numba.cuda.IpcHandle` is returned,
numba/cuda/cudadrv/driver.py:class NumbaCUDAMemoryManager(GetIpcHandleMixin, HostOnlyCUDAMemoryManager):
numba/cuda/cudadrv/driver.py:    if config.CUDA_MEMORY_MANAGER == 'default':
numba/cuda/cudadrv/driver.py:        _memory_manager = NumbaCUDAMemoryManager
numba/cuda/cudadrv/driver.py:        mgr_module = importlib.import_module(config.CUDA_MEMORY_MANAGER)
numba/cuda/cudadrv/driver.py:                           config.CUDA_MEMORY_MANAGER)
numba/cuda/cudadrv/driver.py:    :type mm_plugin: BaseCUDAMemoryManager
numba/cuda/cudadrv/driver.py:        return int(self.memory_capacity * config.CUDA_DEALLOCS_RATIO)
numba/cuda/cudadrv/driver.py:        if (len(self._cons) > config.CUDA_DEALLOCS_COUNT or
numba/cuda/cudadrv/driver.py:    This object wraps a CUDA Context resource.
numba/cuda/cudadrv/driver.py:            return self._cuda_python_active_blocks_per_multiprocessor(*args)
numba/cuda/cudadrv/driver.py:    def _cuda_python_active_blocks_per_multiprocessor(self, func, blocksize,
numba/cuda/cudadrv/driver.py:                         Use `0` to pass `NULL` to the underlying CUDA API.
numba/cuda/cudadrv/driver.py:            return self._cuda_python_max_potential_block_size(*args)
numba/cuda/cudadrv/driver.py:    def _cuda_python_max_potential_block_size(self, func, b2d_func, memsize,
numba/cuda/cudadrv/driver.py:            raise CudaDriverError("%s cannot map host memory" % self.device)
numba/cuda/cudadrv/driver.py:        Returns an *IpcHandle* from a GPU allocation.
numba/cuda/cudadrv/driver.py:            raise OSError('OS does not support CUDA IPC')
numba/cuda/cudadrv/driver.py:        return "<CUDA context %s of device %d>" % (self.handle, self.device.id)
numba/cuda/cudadrv/driver.py:        return load_module_image_cuda_python(context, image)
numba/cuda/cudadrv/driver.py:    logsz = config.CUDA_LOG_SIZE
numba/cuda/cudadrv/driver.py:        enums.CU_JIT_LOG_VERBOSE: c_void_p(config.CUDA_VERBOSE_JIT_LOG),
numba/cuda/cudadrv/driver.py:    except CudaAPIError as e:
numba/cuda/cudadrv/driver.py:        raise CudaAPIError(e.code, msg)
numba/cuda/cudadrv/driver.py:def load_module_image_cuda_python(context, image):
numba/cuda/cudadrv/driver.py:    logsz = config.CUDA_LOG_SIZE
numba/cuda/cudadrv/driver.py:        jit_option.CU_JIT_LOG_VERBOSE: config.CUDA_VERBOSE_JIT_LOG,
numba/cuda/cudadrv/driver.py:    except CudaAPIError as e:
numba/cuda/cudadrv/driver.py:        raise CudaAPIError(e.code, msg)
numba/cuda/cudadrv/driver.py:    return CudaPythonModule(weakref.proxy(context), handle, info_log,
numba/cuda/cudadrv/driver.py:    This memory is managed by CUDA, and finalization entails deallocation. The
numba/cuda/cudadrv/driver.py:    This applies to memory not otherwise managed by CUDA. Page-locking can
numba/cuda/cudadrv/driver.py:    `mempin` may fail with `CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED`, leading
numba/cuda/cudadrv/driver.py:    to unexpected behavior for the context managers `cuda.{pinned,mapped}`.
numba/cuda/cudadrv/driver.py:class _CudaIpcImpl(object):
numba/cuda/cudadrv/driver.py:    """Implementation of GPU IPC using CUDA driver API.
numba/cuda/cudadrv/driver.py:        Import the IPC memory and returns a raw CUDA memory pointer object
numba/cuda/cudadrv/driver.py:    """Implementation of GPU IPC using custom staging logic to workaround
numba/cuda/cudadrv/driver.py:    CUDA IPC limitation on peer accessibility between devices.
numba/cuda/cudadrv/driver.py:        from numba import cuda
numba/cuda/cudadrv/driver.py:        impl = _CudaIpcImpl(parent=self.parent)
numba/cuda/cudadrv/driver.py:        with cuda.gpus[srcdev_id]:
numba/cuda/cudadrv/driver.py:            source_ptr = impl.open(cuda.devices.get_context())
numba/cuda/cudadrv/driver.py:        # Allocate GPU buffer.
numba/cuda/cudadrv/driver.py:        with cuda.gpus[srcdev_id]:
numba/cuda/cudadrv/driver.py:    CUDA IPC handle. Serialization of the CUDA IPC handle object is implemented
numba/cuda/cudadrv/driver.py:    :param handle: The CUDA IPC handle, as a ctypes array of bytes.
numba/cuda/cudadrv/driver.py:        Import the IPC memory and returns a raw CUDA memory pointer object
numba/cuda/cudadrv/driver.py:        self._impl = _CudaIpcImpl(self)
numba/cuda/cudadrv/driver.py:        context.  Returns a raw CUDA memory pointer object.
numba/cuda/cudadrv/driver.py:        This is enhanced over CUDA IPC that it will work regardless of whether
numba/cuda/cudadrv/driver.py:                                         dtype=dtype, gpu_data=dptr)
numba/cuda/cudadrv/driver.py:    __cuda_memory__ = True
numba/cuda/cudadrv/driver.py:        self._cuda_memsize_ = size
numba/cuda/cudadrv/driver.py:    __cuda_memory__ = True
numba/cuda/cudadrv/driver.py:    __cuda_memory__ = True
numba/cuda/cudadrv/driver.py:                CU_STREAM_DEFAULT: "<Default CUDA stream on %s>",
numba/cuda/cudadrv/driver.py:                    "<Legacy default CUDA stream on %s>",
numba/cuda/cudadrv/driver.py:                    "<Per-thread default CUDA stream on %s>",
numba/cuda/cudadrv/driver.py:                drvapi.CU_STREAM_DEFAULT: "<Default CUDA stream on %s>",
numba/cuda/cudadrv/driver.py:                drvapi.CU_STREAM_LEGACY: "<Legacy default CUDA stream on %s>",
numba/cuda/cudadrv/driver.py:                    "<Per-thread default CUDA stream on %s>",
numba/cuda/cudadrv/driver.py:            return "<External CUDA stream %d on %s>" % (ptr, self.context)
numba/cuda/cudadrv/driver.py:            return "<CUDA stream %d on %s>" % (ptr, self.context)
numba/cuda/cudadrv/driver.py:        Callback functions are called from a CUDA driver thread, not from
numba/cuda/cudadrv/driver.py:        the thread that invoked `add_callback`. No CUDA API functions may
numba/cuda/cudadrv/driver.py:        eventual deprecation and may be replaced in a future CUDA release.
numba/cuda/cudadrv/driver.py:        except CudaAPIError as e:
numba/cuda/cudadrv/driver.py:            if e.code == enums.CUDA_ERROR_NOT_READY:
numba/cuda/cudadrv/driver.py:class CudaPythonModule(Module):
numba/cuda/cudadrv/driver.py:        return CudaPythonFunction(weakref.proxy(self), handle, name)
numba/cuda/cudadrv/driver.py:        return "<CUDA function %s>" % self.name
numba/cuda/cudadrv/driver.py:class CudaPythonFunction(Function):
numba/cuda/cudadrv/driver.py:        if config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY:
numba/cuda/cudadrv/driver.py:            return CudaPythonLinker(max_registers, lineinfo, cc)
numba/cuda/cudadrv/driver.py:        """Add CUDA source in a string to the link. The name of the source
numba/cuda/cudadrv/driver.py:        ptx_compile_opts = ['--gpu-name', arch, '-c']
numba/cuda/cudadrv/driver.py:    def add_ptx(self, ptx, name='<cudapy-ptx>'):
numba/cuda/cudadrv/driver.py:        logsz = config.CUDA_LOG_SIZE
numba/cuda/cudadrv/driver.py:    def add_ptx(self, ptx, name='<cudapy-ptx>'):
numba/cuda/cudadrv/driver.py:        except CudaAPIError as e:
numba/cuda/cudadrv/driver.py:        except CudaAPIError as e:
numba/cuda/cudadrv/driver.py:            if e.code == enums.CUDA_ERROR_FILE_NOT_FOUND:
numba/cuda/cudadrv/driver.py:        except CudaAPIError as e:
numba/cuda/cudadrv/driver.py:class CudaPythonLinker(Linker):
numba/cuda/cudadrv/driver.py:        logsz = config.CUDA_LOG_SIZE
numba/cuda/cudadrv/driver.py:    def add_ptx(self, ptx, name='<cudapy-ptx>'):
numba/cuda/cudadrv/driver.py:        except CudaAPIError as e:
numba/cuda/cudadrv/driver.py:        except CudaAPIError as e:
numba/cuda/cudadrv/driver.py:            if e.code == binding.CUresult.CUDA_ERROR_FILE_NOT_FOUND:
numba/cuda/cudadrv/driver.py:        except CudaAPIError as e:
numba/cuda/cudadrv/driver.py:    sz = getattr(devmem, '_cuda_memsize_', None)
numba/cuda/cudadrv/driver.py:        devmem._cuda_memsize_ = sz
numba/cuda/cudadrv/driver.py:    """All CUDA memory object is recognized as an instance with the attribute
numba/cuda/cudadrv/driver.py:    "__cuda_memory__" defined and its value evaluated to True.
numba/cuda/cudadrv/driver.py:    All CUDA memory object should also define an attribute named
numba/cuda/cudadrv/driver.py:    return getattr(obj, '__cuda_memory__', False)
numba/cuda/cudadrv/driver.py:    """A sentry for methods that accept CUDA memory object.
numba/cuda/cudadrv/driver.py:        raise Exception("Not a CUDA memory object.")
numba/cuda/cudadrv/driver.py:    stream: a CUDA stream
numba/cuda/cudadrv/error.py:class CudaDriverError(Exception):
numba/cuda/cudadrv/error.py:class CudaRuntimeError(Exception):
numba/cuda/cudadrv/error.py:class CudaSupportError(ImportError):
numba/cuda/cudadrv/nvrtc.py:from numba.cuda.cudadrv.error import (NvrtcError, NvrtcCompilationError,
numba/cuda/cudadrv/nvrtc.py:    (for Numba) open_cudalib function to load the NVRTC library.
numba/cuda/cudadrv/nvrtc.py:                from numba.cuda.cudadrv.libs import open_cudalib
numba/cuda/cudadrv/nvrtc.py:                    lib = open_cudalib('nvrtc')
numba/cuda/cudadrv/nvrtc.py:    Compile a CUDA C/C++ source to PTX for a given compute capability.
numba/cuda/cudadrv/nvrtc.py:    # - The CUDA include path is added.
numba/cuda/cudadrv/nvrtc.py:    arch = f'--gpu-architecture=compute_{major}{minor}'
numba/cuda/cudadrv/nvrtc.py:    include = f'-I{config.CUDA_INCLUDE_PATH}'
numba/cuda/cudadrv/nvrtc.py:    cudadrv_path = os.path.dirname(os.path.abspath(__file__))
numba/cuda/cudadrv/nvrtc.py:    numba_cuda_path = os.path.dirname(cudadrv_path)
numba/cuda/cudadrv/nvrtc.py:    numba_include = f'-I{numba_cuda_path}'
numba/cuda/cudadrv/runtime.py:CUDA Runtime wrapper.
numba/cuda/cudadrv/runtime.py:from numba.cuda.cudadrv.driver import ERROR_MAP, make_logger
numba/cuda/cudadrv/runtime.py:from numba.cuda.cudadrv.error import CudaSupportError, CudaRuntimeError
numba/cuda/cudadrv/runtime.py:from numba.cuda.cudadrv.libs import open_cudalib
numba/cuda/cudadrv/runtime.py:from numba.cuda.cudadrv.rtapi import API_PROTOTYPES
numba/cuda/cudadrv/runtime.py:from numba.cuda.cudadrv import enums
numba/cuda/cudadrv/runtime.py:class CudaRuntimeAPIError(CudaRuntimeError):
numba/cuda/cudadrv/runtime.py:    Raised when there is an error accessing a C API from the CUDA Runtime.
numba/cuda/cudadrv/runtime.py:        if config.DISABLE_CUDA:
numba/cuda/cudadrv/runtime.py:            msg = ("CUDA is disabled due to setting NUMBA_DISABLE_CUDA=1 "
numba/cuda/cudadrv/runtime.py:                   "in the environment, or because CUDA is unsupported on "
numba/cuda/cudadrv/runtime.py:            raise CudaSupportError(msg)
numba/cuda/cudadrv/runtime.py:        self.lib = open_cudalib('cudart')
numba/cuda/cudadrv/runtime.py:        def safe_cuda_api_call(*args):
numba/cuda/cudadrv/runtime.py:        return safe_cuda_api_call
numba/cuda/cudadrv/runtime.py:        if retcode != enums.CUDA_SUCCESS:
numba/cuda/cudadrv/runtime.py:            errname = ERROR_MAP.get(retcode, "cudaErrorUnknown")
numba/cuda/cudadrv/runtime.py:            raise CudaRuntimeAPIError(retcode, msg)
numba/cuda/cudadrv/runtime.py:            raise CudaRuntimeError(msg % fname)
numba/cuda/cudadrv/runtime.py:        Returns the CUDA Runtime version as a tuple (major, minor).
numba/cuda/cudadrv/runtime.py:        self.cudaRuntimeGetVersion(ctypes.byref(rtver))
numba/cuda/cudadrv/runtime.py:        Returns True if the CUDA Runtime is a supported version.
numba/cuda/cudadrv/runtime.py:        """A tuple of all supported CUDA toolkit versions. Versions are given in
numba/cuda/cudadrv/devicearray.py:A CUDA ND Array is recognized by checking the __cuda_memory__ attribute
numba/cuda/cudadrv/devicearray.py:from numba.cuda.cudadrv import devices, dummyarray
numba/cuda/cudadrv/devicearray.py:from numba.cuda.cudadrv import driver as _driver
numba/cuda/cudadrv/devicearray.py:from numba.cuda.api_util import prepare_shape_strides_dtype
numba/cuda/cudadrv/devicearray.py:def is_cuda_ndarray(obj):
numba/cuda/cudadrv/devicearray.py:    "Check if an object is a CUDA ndarray"
numba/cuda/cudadrv/devicearray.py:    return getattr(obj, '__cuda_ndarray__', False)
numba/cuda/cudadrv/devicearray.py:def verify_cuda_ndarray_interface(obj):
numba/cuda/cudadrv/devicearray.py:    "Verify the CUDA ndarray interface for an obj"
numba/cuda/cudadrv/devicearray.py:    require_cuda_ndarray(obj)
numba/cuda/cudadrv/devicearray.py:def require_cuda_ndarray(obj):
numba/cuda/cudadrv/devicearray.py:    "Raises ValueError is is_cuda_ndarray(obj) evaluates False"
numba/cuda/cudadrv/devicearray.py:    if not is_cuda_ndarray(obj):
numba/cuda/cudadrv/devicearray.py:        raise ValueError('require an cuda ndarray object')
numba/cuda/cudadrv/devicearray.py:    """A on GPU NDArray representation
numba/cuda/cudadrv/devicearray.py:    __cuda_memory__ = True
numba/cuda/cudadrv/devicearray.py:    __cuda_ndarray__ = True     # There must be gpu_data attribute
numba/cuda/cudadrv/devicearray.py:    def __init__(self, shape, strides, dtype, stream=0, gpu_data=None):
numba/cuda/cudadrv/devicearray.py:            cuda stream.
numba/cuda/cudadrv/devicearray.py:        gpu_data
numba/cuda/cudadrv/devicearray.py:        # prepare gpu memory
numba/cuda/cudadrv/devicearray.py:            if gpu_data is None:
numba/cuda/cudadrv/devicearray.py:                gpu_data = devices.get_context().memalloc(self.alloc_size)
numba/cuda/cudadrv/devicearray.py:                self.alloc_size = _driver.device_memory_size(gpu_data)
numba/cuda/cudadrv/devicearray.py:            gpu_data = _driver.MemoryPointer(context=devices.get_context(),
numba/cuda/cudadrv/devicearray.py:        self.gpu_data = gpu_data
numba/cuda/cudadrv/devicearray.py:    def __cuda_array_interface__(self):
numba/cuda/cudadrv/devicearray.py:        """Bind a CUDA stream to this object so that all subsequent operation
numba/cuda/cudadrv/devicearray.py:            from numba.cuda.kernels.transpose import transpose
numba/cuda/cudadrv/devicearray.py:        """Returns the ctypes pointer to the GPU data buffer
numba/cuda/cudadrv/devicearray.py:        if self.gpu_data is None:
numba/cuda/cudadrv/devicearray.py:            return self.gpu_data.device_ctypes_pointer
numba/cuda/cudadrv/devicearray.py:        If `ary` is a CUDA memory, perform a device-to-device transfer.
numba/cuda/cudadrv/devicearray.py:        If a CUDA ``stream`` is given, then the transfer will be made
numba/cuda/cudadrv/devicearray.py:            from numba import cuda
numba/cuda/cudadrv/devicearray.py:            d_arr = cuda.to_device(arr)
numba/cuda/cudadrv/devicearray.py:            gpu_data = self.gpu_data.view(begin * itemsize, end * itemsize)
numba/cuda/cudadrv/devicearray.py:                                gpu_data=gpu_data)
numba/cuda/cudadrv/devicearray.py:    def as_cuda_arg(self):
numba/cuda/cudadrv/devicearray.py:        return self.gpu_data
numba/cuda/cudadrv/devicearray.py:        ipch = devices.get_context().get_ipc_handle(self.gpu_data)
numba/cuda/cudadrv/devicearray.py:        stream : cuda stream or 0, optional
numba/cuda/cudadrv/devicearray.py:            gpu_data=self.gpu_data,
numba/cuda/cudadrv/devicearray.py:            gpu_data=self.gpu_data,
numba/cuda/cudadrv/devicearray.py:    An on-GPU record type
numba/cuda/cudadrv/devicearray.py:    def __init__(self, dtype, stream=0, gpu_data=None):
numba/cuda/cudadrv/devicearray.py:                                           gpu_data)
numba/cuda/cudadrv/devicearray.py:        """Do `__getitem__(item)` with CUDA stream
numba/cuda/cudadrv/devicearray.py:        newdata = self.gpu_data.view(offset)
numba/cuda/cudadrv/devicearray.py:                                    gpu_data=newdata)
numba/cuda/cudadrv/devicearray.py:                                 dtype=dtype, gpu_data=newdata,
numba/cuda/cudadrv/devicearray.py:        """Do `__setitem__(key, value)` with CUDA stream
numba/cuda/cudadrv/devicearray.py:        newdata = self.gpu_data.view(offset)
numba/cuda/cudadrv/devicearray.py:        lhs = type(self)(dtype=typ, stream=stream, gpu_data=newdata)
numba/cuda/cudadrv/devicearray.py:    :param ndim: We need to have static array sizes for cuda.local.array, so
numba/cuda/cudadrv/devicearray.py:    from numba import cuda  # circular!
numba/cuda/cudadrv/devicearray.py:        @cuda.jit
numba/cuda/cudadrv/devicearray.py:    @cuda.jit
numba/cuda/cudadrv/devicearray.py:        location = cuda.grid(1)
numba/cuda/cudadrv/devicearray.py:        idx = cuda.local.array(
numba/cuda/cudadrv/devicearray.py:    An on-GPU array type
numba/cuda/cudadrv/devicearray.py:                       dtype=self.dtype, gpu_data=self.gpu_data)
numba/cuda/cudadrv/devicearray.py:                       dtype=self.dtype, gpu_data=self.gpu_data)
numba/cuda/cudadrv/devicearray.py:                       dtype=self.dtype, gpu_data=self.gpu_data,
numba/cuda/cudadrv/devicearray.py:        """Do `__getitem__(item)` with CUDA stream
numba/cuda/cudadrv/devicearray.py:            newdata = self.gpu_data.view(*extents[0])
numba/cuda/cudadrv/devicearray.py:                                        gpu_data=newdata)
numba/cuda/cudadrv/devicearray.py:                           dtype=self.dtype, gpu_data=newdata, stream=stream)
numba/cuda/cudadrv/devicearray.py:            newdata = self.gpu_data.view(*arr.extent)
numba/cuda/cudadrv/devicearray.py:                       dtype=self.dtype, gpu_data=newdata, stream=stream)
numba/cuda/cudadrv/devicearray.py:        """Do `__setitem__(key, value)` with CUDA stream
numba/cuda/cudadrv/devicearray.py:        newdata = self.gpu_data.view(*arr.extent)
numba/cuda/cudadrv/devicearray.py:            gpu_data=newdata,
numba/cuda/cudadrv/devicearray.py:    in the same machine for share a GPU allocation.
numba/cuda/cudadrv/devicearray.py:            # use ipc_array here as a normal gpu array object
numba/cuda/cudadrv/devicearray.py:        return DeviceNDArray(gpu_data=dptr, **self._array_desc)
numba/cuda/cudadrv/devicearray.py:    A host array that uses CUDA mapped memory.
numba/cuda/cudadrv/devicearray.py:    def device_setup(self, gpu_data, stream=0):
numba/cuda/cudadrv/devicearray.py:        self.gpu_data = gpu_data
numba/cuda/cudadrv/devicearray.py:    A host array that uses CUDA managed memory.
numba/cuda/cudadrv/devicearray.py:    def device_setup(self, gpu_data, stream=0):
numba/cuda/cudadrv/devicearray.py:        self.gpu_data = gpu_data
numba/cuda/cudadrv/devicearray.py:def from_array_like(ary, stream=0, gpu_data=None):
numba/cuda/cudadrv/devicearray.py:                         gpu_data=gpu_data)
numba/cuda/cudadrv/devicearray.py:def from_record_like(rec, stream=0, gpu_data=None):
numba/cuda/cudadrv/devicearray.py:    return DeviceRecord(rec.dtype, stream=stream, gpu_data=gpu_data)
numba/cuda/cudadrv/devicearray.py:    elif hasattr(obj, '__cuda_array_interface__'):
numba/cuda/cudadrv/devicearray.py:        return numba.cuda.as_cuda_array(obj), False
numba/cuda/cudadrv/devicearray.py:            if config.CUDA_WARN_ON_IMPLICIT_COPY:
numba/cuda/cudadrv/devicearray.py:                    msg = ("Host array used in CUDA kernel will incur "
numba/cuda/cudadrv/enums.py:Enum values for CUDA driver. Information about the values
numba/cuda/cudadrv/enums.py:can be found on the official NVIDIA documentation website.
numba/cuda/cudadrv/enums.py:ref: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html
numba/cuda/cudadrv/enums.py:anchor: #group__CUDA__TYPES
numba/cuda/cudadrv/enums.py:CUDA_SUCCESS = 0
numba/cuda/cudadrv/enums.py:CUDA_ERROR_INVALID_VALUE = 1
numba/cuda/cudadrv/enums.py:CUDA_ERROR_OUT_OF_MEMORY = 2
numba/cuda/cudadrv/enums.py:CUDA_ERROR_NOT_INITIALIZED = 3
numba/cuda/cudadrv/enums.py:CUDA_ERROR_DEINITIALIZED = 4
numba/cuda/cudadrv/enums.py:CUDA_ERROR_PROFILER_DISABLED = 5
numba/cuda/cudadrv/enums.py:CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6
numba/cuda/cudadrv/enums.py:CUDA_ERROR_PROFILER_ALREADY_STARTED = 7
numba/cuda/cudadrv/enums.py:CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8
numba/cuda/cudadrv/enums.py:CUDA_ERROR_STUB_LIBRARY = 34
numba/cuda/cudadrv/enums.py:CUDA_ERROR_DEVICE_UNAVAILABLE = 46
numba/cuda/cudadrv/enums.py:CUDA_ERROR_NO_DEVICE = 100
numba/cuda/cudadrv/enums.py:CUDA_ERROR_INVALID_DEVICE = 101
numba/cuda/cudadrv/enums.py:CUDA_ERROR_DEVICE_NOT_LICENSED = 102
numba/cuda/cudadrv/enums.py:CUDA_ERROR_INVALID_IMAGE = 200
numba/cuda/cudadrv/enums.py:CUDA_ERROR_INVALID_CONTEXT = 201
numba/cuda/cudadrv/enums.py:CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202
numba/cuda/cudadrv/enums.py:CUDA_ERROR_MAP_FAILED = 205
numba/cuda/cudadrv/enums.py:CUDA_ERROR_UNMAP_FAILED = 206
numba/cuda/cudadrv/enums.py:CUDA_ERROR_ARRAY_IS_MAPPED = 207
numba/cuda/cudadrv/enums.py:CUDA_ERROR_ALREADY_MAPPED = 208
numba/cuda/cudadrv/enums.py:CUDA_ERROR_NO_BINARY_FOR_GPU = 209
numba/cuda/cudadrv/enums.py:CUDA_ERROR_ALREADY_ACQUIRED = 210
numba/cuda/cudadrv/enums.py:CUDA_ERROR_NOT_MAPPED = 211
numba/cuda/cudadrv/enums.py:CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212
numba/cuda/cudadrv/enums.py:CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213
numba/cuda/cudadrv/enums.py:CUDA_ERROR_ECC_UNCORRECTABLE = 214
numba/cuda/cudadrv/enums.py:CUDA_ERROR_UNSUPPORTED_LIMIT = 215
numba/cuda/cudadrv/enums.py:CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216
numba/cuda/cudadrv/enums.py:CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217
numba/cuda/cudadrv/enums.py:CUDA_ERROR_INVALID_PTX = 218
numba/cuda/cudadrv/enums.py:CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219
numba/cuda/cudadrv/enums.py:CUDA_ERROR_NVLINK_UNCORRECTABLE = 220
numba/cuda/cudadrv/enums.py:CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221
numba/cuda/cudadrv/enums.py:CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222
numba/cuda/cudadrv/enums.py:CUDA_ERROR_JIT_COMPILATION_DISABLED = 223
numba/cuda/cudadrv/enums.py:CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224
numba/cuda/cudadrv/enums.py:CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC = 225
numba/cuda/cudadrv/enums.py:CUDA_ERROR_INVALID_SOURCE = 300
numba/cuda/cudadrv/enums.py:CUDA_ERROR_FILE_NOT_FOUND = 301
numba/cuda/cudadrv/enums.py:CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302
numba/cuda/cudadrv/enums.py:CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303
numba/cuda/cudadrv/enums.py:CUDA_ERROR_OPERATING_SYSTEM = 304
numba/cuda/cudadrv/enums.py:CUDA_ERROR_INVALID_HANDLE = 400
numba/cuda/cudadrv/enums.py:CUDA_ERROR_ILLEGAL_STATE = 401
numba/cuda/cudadrv/enums.py:CUDA_ERROR_NOT_FOUND = 500
numba/cuda/cudadrv/enums.py:CUDA_ERROR_NOT_READY = 600
numba/cuda/cudadrv/enums.py:CUDA_ERROR_LAUNCH_FAILED = 700
numba/cuda/cudadrv/enums.py:CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
numba/cuda/cudadrv/enums.py:CUDA_ERROR_LAUNCH_TIMEOUT = 702
numba/cuda/cudadrv/enums.py:CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703
numba/cuda/cudadrv/enums.py:CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
numba/cuda/cudadrv/enums.py:CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705
numba/cuda/cudadrv/enums.py:CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708
numba/cuda/cudadrv/enums.py:CUDA_ERROR_CONTEXT_IS_DESTROYED = 709
numba/cuda/cudadrv/enums.py:CUDA_ERROR_ASSERT = 710
numba/cuda/cudadrv/enums.py:CUDA_ERROR_TOO_MANY_PEERS = 711
numba/cuda/cudadrv/enums.py:CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
numba/cuda/cudadrv/enums.py:CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713
numba/cuda/cudadrv/enums.py:CUDA_ERROR_HARDWARE_STACK_ERROR = 714
numba/cuda/cudadrv/enums.py:CUDA_ERROR_ILLEGAL_INSTRUCTION = 715
numba/cuda/cudadrv/enums.py:CUDA_ERROR_MISALIGNED_ADDRESS = 716
numba/cuda/cudadrv/enums.py:CUDA_ERROR_INVALID_ADDRESS_SPACE = 717
numba/cuda/cudadrv/enums.py:CUDA_ERROR_INVALID_PC = 718
numba/cuda/cudadrv/enums.py:CUDA_ERROR_LAUNCH_FAILED = 719
numba/cuda/cudadrv/enums.py:CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720
numba/cuda/cudadrv/enums.py:CUDA_ERROR_NOT_PERMITTED = 800
numba/cuda/cudadrv/enums.py:CUDA_ERROR_NOT_SUPPORTED = 801
numba/cuda/cudadrv/enums.py:CUDA_ERROR_SYSTEM_NOT_READY = 802
numba/cuda/cudadrv/enums.py:CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803
numba/cuda/cudadrv/enums.py:CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804
numba/cuda/cudadrv/enums.py:CUDA_ERROR_MPS_CONNECTION_FAILED = 805
numba/cuda/cudadrv/enums.py:CUDA_ERROR_MPS_RPC_FAILURE = 806
numba/cuda/cudadrv/enums.py:CUDA_ERROR_MPS_SERVER_NOT_READY = 807
numba/cuda/cudadrv/enums.py:CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808
numba/cuda/cudadrv/enums.py:CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809
numba/cuda/cudadrv/enums.py:CUDA_ERROR_MPS_CLIENT_TERMINATED = 810
numba/cuda/cudadrv/enums.py:CUDA_ERROR_CDP_NOT_SUPPORTED = 811
numba/cuda/cudadrv/enums.py:CUDA_ERROR_CDP_VERSION_MISMATCH = 812
numba/cuda/cudadrv/enums.py:CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900
numba/cuda/cudadrv/enums.py:CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901
numba/cuda/cudadrv/enums.py:CUDA_ERROR_STREAM_CAPTURE_MERGE = 902
numba/cuda/cudadrv/enums.py:CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903
numba/cuda/cudadrv/enums.py:CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904
numba/cuda/cudadrv/enums.py:CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905
numba/cuda/cudadrv/enums.py:CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906
numba/cuda/cudadrv/enums.py:CUDA_ERROR_CAPTURED_EVENT = 907
numba/cuda/cudadrv/enums.py:CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908
numba/cuda/cudadrv/enums.py:CUDA_ERROR_TIMEOUT = 909
numba/cuda/cudadrv/enums.py:CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910
numba/cuda/cudadrv/enums.py:CUDA_ERROR_EXTERNAL_DEVICE = 911
numba/cuda/cudadrv/enums.py:CUDA_ERROR_INVALID_CLUSTER_SIZE = 912
numba/cuda/cudadrv/enums.py:CUDA_ERROR_UNKNOWN = 999
numba/cuda/cudadrv/enums.py:#   This flag was deprecated as of CUDA 11.0 and it no longer has effect.
numba/cuda/cudadrv/enums.py:#   All contexts as of CUDA 3.2 behave as though the flag is enabled.
numba/cuda/cudadrv/enums.py:# Force synchronous blocking on cudaMemcpy/cudaMemset
numba/cuda/cudadrv/enums.py:# If set, host memory is portable between CUDA contexts.
numba/cuda/cudadrv/enums.py:# If set, host memory is mapped into CUDA address space and
numba/cuda/cudadrv/enums.py:# If set, host memory is portable between CUDA contexts.
numba/cuda/cudadrv/enums.py:# If set, host memory is mapped into CUDA address space and
numba/cuda/cudadrv/enums.py:# as non cache-coherent for the GPU and is expected
numba/cuda/cudadrv/enums.py:# to be physically contiguous. It may return CUDA_ERROR_NOT_PERMITTED
numba/cuda/cudadrv/enums.py:# if run as an unprivileged user, CUDA_ERROR_NOT_SUPPORTED on older
numba/cuda/cudadrv/enums.py:# and CUDA_ERROR_NOT_SUPPORTED is returned.
numba/cuda/cudadrv/enums.py:# to error with CUDA_ERROR_NOT_SUPPORTED.
numba/cuda/cudadrv/enums.py:# CUDA Mem Attach Flags
numba/cuda/cudadrv/enums.py:# cudaDevAttrConcurrentManagedAccess is zero, then managed memory is
numba/cuda/cudadrv/enums.py:# with cudaStreamAttachMemAsync, in which case it can be used in kernels
numba/cuda/cudadrv/enums.py:# cudaDevAttrConcurrentManagedAccess is zero, then managed memory accesses
numba/cuda/cudadrv/enums.py:# that is suitable for cudaIpcGetMemHandle, 0 otherwise
numba/cuda/cudadrv/enums.py:CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10
numba/cuda/cudadrv/enums.py:# can be used with the GPUDirect RDMA API
numba/cuda/cudadrv/enums.py:CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15
numba/cuda/cudadrv/enums.py:# compiled prior to CUDA 3.0.
numba/cuda/cudadrv/enums.py:# Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.
numba/cuda/cudadrv/enums.py:# Setting it at runtime should return CUDA_ERROR_NOT_PERMITTED.
numba/cuda/cudadrv/enums.py:# Setting it at runtime should return CUDA_ERROR_NOT_PERMITTED.
numba/cuda/cudadrv/enums.py:# https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES
numba/cuda/cudadrv/enums.py:# The value type is CUclusterSchedulingPolicy / cudaClusterSchedulingPolicy.
numba/cuda/cudadrv/enums.py:CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15
numba/cuda/cudadrv/enums.py:CU_DEVICE_ATTRIBUTE_IS_MULTI_GPU_BOARD = 84
numba/cuda/cudadrv/enums.py:CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85
numba/cuda/cudadrv/rtapi.py:    # cudaError_t cudaRuntimeGetVersion ( int* runtimeVersion )
numba/cuda/cudadrv/rtapi.py:    'cudaRuntimeGetVersion': (c_int, POINTER(c_int)),
numba/cuda/cudadrv/libs.py:"""CUDA Toolkit libraries lookup utilities.
numba/cuda/cudadrv/libs.py:CUDA Toolkit libraries can be available via either:
numba/cuda/cudadrv/libs.py:- the `cuda-nvcc` and `cuda-nvrtc` conda packages for CUDA 12,
numba/cuda/cudadrv/libs.py:- the `cudatoolkit` conda package for CUDA 11,
numba/cuda/cudadrv/libs.py:- a user supplied location from CUDA_HOME,
numba/cuda/cudadrv/libs.py:- package-specific locations (e.g. the Debian NVIDIA packages),
numba/cuda/cudadrv/libs.py:from numba.cuda.cuda_paths import get_cuda_paths
numba/cuda/cudadrv/libs.py:from numba.cuda.cudadrv.driver import locate_driver_and_loader, load_driver
numba/cuda/cudadrv/libs.py:from numba.cuda.cudadrv.error import CudaSupportError
numba/cuda/cudadrv/libs.py:    d = get_cuda_paths()
numba/cuda/cudadrv/libs.py:def get_cudalib(lib, static=False):
numba/cuda/cudadrv/libs.py:    Find the path of a CUDA library based on a search of known locations. If
numba/cuda/cudadrv/libs.py:        return get_cuda_paths()['nvvm'].info or _dllnamepattern % 'nvvm'
numba/cuda/cudadrv/libs.py:        dir_type = 'static_cudalib_dir' if static else 'cudalib_dir'
numba/cuda/cudadrv/libs.py:        libdir = get_cuda_paths()[dir_type].info
numba/cuda/cudadrv/libs.py:def open_cudalib(lib):
numba/cuda/cudadrv/libs.py:    path = get_cudalib(lib)
numba/cuda/cudadrv/libs.py:        return get_cuda_paths()['nvvm'].by
numba/cuda/cudadrv/libs.py:        return get_cuda_paths()['libdevice'].by
numba/cuda/cudadrv/libs.py:        dir_type = 'static_cudalib_dir' if static else 'cudalib_dir'
numba/cuda/cudadrv/libs.py:        return get_cuda_paths()[dir_type].by
numba/cuda/cudadrv/libs.py:    except CudaSupportError as e:
numba/cuda/cudadrv/libs.py:    # number in the soname (e.g. "libcuda.so.530.30.02"), which can be used to
numba/cuda/cudadrv/libs.py:            # to actual CUDA functionality.
numba/cuda/cudadrv/libs.py:                  'path to libcuda.so')
numba/cuda/cudadrv/libs.py:            locations = set(s for s in maps.split() if 'libcuda.so' in s)
numba/cuda/cudadrv/libs.py:            print('\tMapped libcuda.so paths:')
numba/cuda/cudadrv/libs.py:    libs = 'nvvm nvrtc cudart'.split()
numba/cuda/cudadrv/libs.py:        path = get_cudalib(lib)
numba/cuda/cudadrv/libs.py:            open_cudalib(lib)
numba/cuda/cudadrv/libs.py:    # Check for cudadevrt (the only static library)
numba/cuda/cudadrv/libs.py:    lib = 'cudadevrt'
numba/cuda/cudadrv/libs.py:    path = get_cudalib(lib, static=True)
numba/cuda/cudadrv/__init__.py:"""CUDA Driver
numba/cuda/cudadrv/__init__.py:assert not config.ENABLE_CUDASIM, 'Cannot use real driver API with simulator'
numba/cuda/cudadrv/nvvm.py:from .libs import get_libdevice, open_libdevice, open_cudalib
numba/cuda/cudadrv/nvvm.py:# Data layouts. NVVM IR 1.8 (CUDA 11.6) introduced 128-bit integer support.
numba/cuda/cudadrv/nvvm.py:                    inst.driver = open_cudalib('nvvm')
numba/cuda/cudadrv/nvvm.py:                              "cudatoolkit`:\n%s")
numba/cuda/cudadrv/nvvm.py:        For documentation on NVVM compilation options, see the CUDA Toolkit
numba/cuda/cudadrv/nvvm.py:        https://docs.nvidia.com/cuda/libnvvm-api/index.html#_CPPv418nvvmCompileProgram11nvvmProgramiPPKc
numba/cuda/cudadrv/nvvm.py:        # For unsupported CUDA toolkit versions, all we can do is assume all
numba/cuda/cudadrv/nvvm.py:                      if cc >= config.CUDA_DEFAULT_PTX_CC])
numba/cuda/cudadrv/nvvm.py:        from numba.cuda.cudadrv.runtime import runtime
numba/cuda/cudadrv/nvvm.py:        cudart_version = runtime.get_version()
numba/cuda/cudadrv/nvvm.py:    min_cudart = min(CTK_SUPPORTED)
numba/cuda/cudadrv/nvvm.py:    if cudart_version < min_cudart:
numba/cuda/cudadrv/nvvm.py:        ctk_ver = f"{cudart_version[0]}.{cudart_version[1]}"
numba/cuda/cudadrv/nvvm.py:        unsupported_ver = (f"CUDA Toolkit {ctk_ver} is unsupported by Numba - "
numba/cuda/cudadrv/nvvm.py:                           f"{min_cudart[0]}.{min_cudart[1]} is the minimum "
numba/cuda/cudadrv/nvvm.py:    _supported_cc = ccs_supported_by_ctk(cudart_version)
numba/cuda/cudadrv/nvvm.py:    by the CUDA toolkit.
numba/cuda/cudadrv/nvvm.py:        msg = "No supported GPU compute capabilities found. " \
numba/cuda/cudadrv/nvvm.py:              "Please check your cudatoolkit version matches your CUDA version."
numba/cuda/cudadrv/nvvm.py:                msg = "GPU compute capability %d.%d is not supported" \
numba/cuda/cudadrv/nvvm.py:    if config.FORCE_CUDA_CC:
numba/cuda/cudadrv/nvvm.py:        arch = config.FORCE_CUDA_CC
numba/cuda/cudadrv/nvvm.py:Please ensure you have a CUDA Toolkit 11.2 or higher.
numba/cuda/cudadrv/nvvm.py:For CUDA 12, ``cuda-nvcc`` and ``cuda-nvrtc`` are required:
numba/cuda/cudadrv/nvvm.py:    $ conda install -c conda-forge cuda-nvcc cuda-nvrtc "cuda-version>=12.0"
numba/cuda/cudadrv/nvvm.py:For CUDA 11, ``cudatoolkit`` is required:
numba/cuda/cudadrv/nvvm.py:    $ conda install -c conda-forge cudatoolkit "cuda-version>=11.2,<12.0"
numba/cuda/cudadrv/nvvm.py:# Translation of code from CUDA Programming Guide v6.5, section B.12
numba/cuda/cudadrv/nvvm.py:def set_cuda_kernel(function):
numba/cuda/cudadrv/nvvm.py:    Mark a function as a CUDA kernel. Kernels have the following requirements:
numba/cuda/libdevicedecl.py:from numba.cuda import libdevice, libdevicefuncs
numba/cuda/decorators.py:from numba.cuda.compiler import declare_device_function
numba/cuda/decorators.py:from numba.cuda.dispatcher import CUDADispatcher
numba/cuda/decorators.py:from numba.cuda.simulator.kernel import FakeCUDAKernel
numba/cuda/decorators.py:    JIT compile a Python function for CUDA GPUs.
numba/cuda/decorators.py:       :class:`Dispatcher <numba.cuda.dispatcher.CUDADispatcher>` is returned.
numba/cuda/decorators.py:       <numba.cuda.dispatcher.CUDADispatcher>`. See :ref:`jit-decorator` for
numba/cuda/decorators.py:    :param link: A list of files containing PTX or CUDA C/C++ source to link
numba/cuda/decorators.py:       environment variable ``NUMBA_CUDA_DEBUGINFO=1``.)
numba/cuda/decorators.py:       the :ref:`CUDA Fast Math documentation <cuda-fast-math>`.
numba/cuda/decorators.py:       assembly code. This enables inspection of the source code in NVIDIA
numba/cuda/decorators.py:    if link and config.ENABLE_CUDASIM:
numba/cuda/decorators.py:        raise NotImplementedError("bounds checking is not supported for CUDA")
numba/cuda/decorators.py:    debug = config.CUDA_DEBUGINFO_DEFAULT if debug is None else debug
numba/cuda/decorators.py:               "is not supported by CUDA. This may result in a crash"
numba/cuda/decorators.py:        if config.ENABLE_CUDASIM:
numba/cuda/decorators.py:                return FakeCUDAKernel(func, device=device, fastmath=fastmath)
numba/cuda/decorators.py:            disp = CUDADispatcher(func, targetoptions=targetoptions)
numba/cuda/decorators.py:                    raise TypeError("CUDA kernel must have void return type.")
numba/cuda/decorators.py:            if config.ENABLE_CUDASIM:
numba/cuda/decorators.py:                    return FakeCUDAKernel(func, device=device,
numba/cuda/decorators.py:            if config.ENABLE_CUDASIM:
numba/cuda/decorators.py:                return FakeCUDAKernel(func_or_sig, device=device,
numba/cuda/decorators.py:                disp = CUDADispatcher(func_or_sig, targetoptions=targetoptions)
numba/cuda/intrinsic_wrapper.py:    return numba.cuda.vote_sync_intrinsic(mask, 0, predicate)[1]
numba/cuda/intrinsic_wrapper.py:    return numba.cuda.vote_sync_intrinsic(mask, 1, predicate)[1]
numba/cuda/intrinsic_wrapper.py:    return numba.cuda.vote_sync_intrinsic(mask, 2, predicate)[1]
numba/cuda/intrinsic_wrapper.py:    return numba.cuda.vote_sync_intrinsic(mask, 3, predicate)[0]
numba/cuda/intrinsic_wrapper.py:    return numba.cuda.shfl_sync_intrinsic(mask, 0, value, src_lane, 0x1f)[0]
numba/cuda/intrinsic_wrapper.py:    return numba.cuda.shfl_sync_intrinsic(mask, 1, value, delta, 0)[0]
numba/cuda/intrinsic_wrapper.py:    return numba.cuda.shfl_sync_intrinsic(mask, 2, value, delta, 0x1f)[0]
numba/cuda/intrinsic_wrapper.py:    return numba.cuda.shfl_sync_intrinsic(mask, 3, value, lane_mask, 0x1f)[0]
numba/cuda/simulator/kernelapi.py:Implements the cuda module as called from within an executing kernel
numba/cuda/simulator/kernelapi.py:(@cuda.jit-decorated function).
numba/cuda/simulator/kernelapi.py:class FakeCUDACg:
numba/cuda/simulator/kernelapi.py:    CUDA Cooperative Groups
numba/cuda/simulator/kernelapi.py:class FakeCUDALocal(object):
numba/cuda/simulator/kernelapi.py:    CUDA Local arrays
numba/cuda/simulator/kernelapi.py:class FakeCUDAConst(object):
numba/cuda/simulator/kernelapi.py:    CUDA Const arrays
numba/cuda/simulator/kernelapi.py:class FakeCUDAShared(object):
numba/cuda/simulator/kernelapi.py:    CUDA Shared arrays.
numba/cuda/simulator/kernelapi.py:    Limitations: assumes that only one call to cuda.shared.array is on a line,
numba/cuda/simulator/kernelapi.py:        a = cuda.shared.array(...); b = cuda.shared.array(...)
numba/cuda/simulator/kernelapi.py:            sharedarrs[i] = cuda.shared.array(...)
numba/cuda/simulator/kernelapi.py:class FakeCUDAAtomic(object):
numba/cuda/simulator/kernelapi.py:class FakeCUDAFp16(object):
numba/cuda/simulator/kernelapi.py:class FakeCUDAModule(object):
numba/cuda/simulator/kernelapi.py:    executing function in order to implement calls to cuda.*. This will fail to
numba/cuda/simulator/kernelapi.py:        from numba import cuda as something_else
numba/cuda/simulator/kernelapi.py:    In other words, the CUDA module must be called cuda.
numba/cuda/simulator/kernelapi.py:        self._cg = FakeCUDACg()
numba/cuda/simulator/kernelapi.py:        self._local = FakeCUDALocal()
numba/cuda/simulator/kernelapi.py:        self._shared = FakeCUDAShared(dynshared_size)
numba/cuda/simulator/kernelapi.py:        self._const = FakeCUDAConst()
numba/cuda/simulator/kernelapi.py:        self._atomic = FakeCUDAAtomic()
numba/cuda/simulator/kernelapi.py:        self._fp16 = FakeCUDAFp16()
numba/cuda/simulator/kernelapi.py:        # to access the actual cuda module as well as the fake cuda module
numba/cuda/simulator/kernelapi.py:def swapped_cuda_module(fn, fake_cuda_module):
numba/cuda/simulator/kernelapi.py:    from numba import cuda
numba/cuda/simulator/kernelapi.py:    # get all globals that is the "cuda" module
numba/cuda/simulator/kernelapi.py:    orig = dict((k, v) for k, v in fn_globs.items() if v is cuda)
numba/cuda/simulator/kernelapi.py:    repl = dict((k, fake_cuda_module) for k, v in orig.items())
numba/cuda/simulator/kernel.py:from .cudadrv.devicearray import FakeCUDAArray, FakeWithinKernelCUDAArray
numba/cuda/simulator/kernel.py:from .kernelapi import Dim3, FakeCUDAModule, swapped_cuda_module
numba/cuda/simulator/kernel.py:FakeCUDAModule.  We only support one kernel launch at a time.
numba/cuda/simulator/kernel.py:class FakeCUDAKernel(object):
numba/cuda/simulator/kernel.py:    Wraps a @cuda.jit-ed function.
numba/cuda/simulator/kernel.py:            with swapped_cuda_module(self.fn, _get_kernel_context()):
numba/cuda/simulator/kernel.py:        fake_cuda_module = FakeCUDAModule(grid_dim, block_dim,
numba/cuda/simulator/kernel.py:        with _push_kernel_context(fake_cuda_module):
numba/cuda/simulator/kernel.py:            # fake_args substitutes all numpy arrays for FakeCUDAArrays
numba/cuda/simulator/kernel.py:                    ret = FakeCUDAArray(arg)  # In case a np record comes in.
numba/cuda/simulator/kernel.py:                if isinstance(ret, FakeCUDAArray):
numba/cuda/simulator/kernel.py:                    return FakeWithinKernelCUDAArray(ret)
numba/cuda/simulator/kernel.py:            with swapped_cuda_module(self.fn, fake_cuda_module):
numba/cuda/simulator/kernel.py:    Manages the execution of a function for a single CUDA thread.
numba/cuda/simulator/api.py:Contains CUDA API functions
numba/cuda/simulator/api.py:from .cudadrv.devices import require_context, reset, gpus  # noqa: F401
numba/cuda/simulator/api.py:from .kernel import FakeCUDAKernel
numba/cuda/simulator/api.py:    gpus.closed = True
numba/cuda/simulator/api.py:    print('Found 1 CUDA devices')
numba/cuda/simulator/api.py:    return gpus
numba/cuda/simulator/api.py:        raise NotImplementedError("bounds checking is not supported for CUDA")
numba/cuda/simulator/api.py:            return FakeCUDAKernel(fn,
numba/cuda/simulator/api.py:    return FakeCUDAKernel(func_or_sig, device=device, debug=debug)
numba/cuda/simulator/cudadrv/devices.py:class FakeCUDADevice:
numba/cuda/simulator/cudadrv/devices.py:        self.uuid = 'GPU-00000000-0000-0000-0000-000000000000'
numba/cuda/simulator/cudadrv/devices.py:class FakeCUDAContext:
numba/cuda/simulator/cudadrv/devices.py:    This stub implements functionality only for simulating a single GPU
numba/cuda/simulator/cudadrv/devices.py:        self._device = FakeCUDADevice()
numba/cuda/simulator/cudadrv/devices.py:    This stub implements a device list containing a single GPU. It also
numba/cuda/simulator/cudadrv/devices.py:    keeps track of the GPU status, i.e. whether the context is closed or not,
numba/cuda/simulator/cudadrv/devices.py:        self.lst = (FakeCUDAContext(0),)
numba/cuda/simulator/cudadrv/devices.py:gpus = FakeDeviceList()
numba/cuda/simulator/cudadrv/devices.py:    gpus[0].closed = True
numba/cuda/simulator/cudadrv/devices.py:    return FakeCUDAContext(devnum)
numba/cuda/simulator/cudadrv/driver.py:class CudaAPIError(RuntimeError):
numba/cuda/simulator/cudadrv/error.py:class CudaSupportError(RuntimeError):
numba/cuda/simulator/cudadrv/devicearray.py:    indexing, similar to the shape in CUDA Python. (Numpy shape arrays allow
numba/cuda/simulator/cudadrv/devicearray.py:class FakeWithinKernelCUDAArray(object):
numba/cuda/simulator/cudadrv/devicearray.py:        assert isinstance(item, FakeCUDAArray)
numba/cuda/simulator/cudadrv/devicearray.py:        if isinstance(item, FakeCUDAArray):
numba/cuda/simulator/cudadrv/devicearray.py:            return FakeWithinKernelCUDAArray(item)
numba/cuda/simulator/cudadrv/devicearray.py:        # things that implement its interfaces, like the FakeCUDAArray or
numba/cuda/simulator/cudadrv/devicearray.py:        # FakeWithinKernelCUDAArray). For other objects, __array_ufunc__ is
numba/cuda/simulator/cudadrv/devicearray.py:        # to somehow implement the ufunc. Since the FakeWithinKernelCUDAArray
numba/cuda/simulator/cudadrv/devicearray.py:            if isinstance(obj, FakeWithinKernelCUDAArray):
numba/cuda/simulator/cudadrv/devicearray.py:class FakeCUDAArray(object):
numba/cuda/simulator/cudadrv/devicearray.py:    __cuda_ndarray__ = True  # There must be gpu_data attribute
numba/cuda/simulator/cudadrv/devicearray.py:        # return nbytes -- FakeCUDAArray is a wrapper around NumPy
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(self._ary, stream)
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(np.transpose(self._ary, axes=axes))
numba/cuda/simulator/cudadrv/devicearray.py:            return FakeCUDAArray(ret, stream=self.stream)
numba/cuda/simulator/cudadrv/devicearray.py:        This may be less forgiving than the CUDA Python implementation, which
numba/cuda/simulator/cudadrv/devicearray.py:        if isinstance(ary, FakeCUDAArray):
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(self._ary.ravel(*args, **kwargs))
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(self._ary.reshape(*args, **kwargs))
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(self._ary.view(*args, **kwargs))
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(self._ary == other)
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(self._ary != other)
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(self._ary < other)
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(self._ary <= other)
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(self._ary > other)
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(self._ary >= other)
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(self._ary + other)
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(self._ary - other)
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(self._ary * other)
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(self._ary // other)
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(self._ary / other)
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(self._ary % other)
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(self._ary ** other)
numba/cuda/simulator/cudadrv/devicearray.py:            FakeCUDAArray(a)
numba/cuda/simulator/cudadrv/devicearray.py:        return FakeCUDAArray(
numba/cuda/simulator/cudadrv/devicearray.py:    return FakeCUDAArray(np.ndarray(*args, **kwargs), stream=stream)
numba/cuda/simulator/cudadrv/devicearray.py:    if isinstance(ary, FakeCUDAArray):
numba/cuda/simulator/cudadrv/devicearray.py:def is_cuda_ndarray(obj):
numba/cuda/simulator/cudadrv/devicearray.py:    "Check if an object is a CUDA ndarray"
numba/cuda/simulator/cudadrv/devicearray.py:    return getattr(obj, '__cuda_ndarray__', False)
numba/cuda/simulator/cudadrv/devicearray.py:def verify_cuda_ndarray_interface(obj):
numba/cuda/simulator/cudadrv/devicearray.py:    "Verify the CUDA ndarray interface for an obj"
numba/cuda/simulator/cudadrv/devicearray.py:    require_cuda_ndarray(obj)
numba/cuda/simulator/cudadrv/devicearray.py:def require_cuda_ndarray(obj):
numba/cuda/simulator/cudadrv/devicearray.py:    "Raises ValueError is is_cuda_ndarray(obj) evaluates False"
numba/cuda/simulator/cudadrv/devicearray.py:    if not is_cuda_ndarray(obj):
numba/cuda/simulator/cudadrv/devicearray.py:        raise ValueError('require an cuda ndarray object')
numba/cuda/simulator/cudadrv/libs.py:    raise FileNotFoundError('Linking libraries not supported by cudasim')
numba/cuda/simulator/cudadrv/__init__.py:from numba.cuda.simulator.cudadrv import (devicearray, devices, driver, drvapi,
numba/cuda/simulator/cudadrv/nvvm.py:set_cuda_kernel = None
numba/cuda/simulator/vector_types.py:from numba.cuda.stubs import _vector_type_stubs
numba/cuda/simulator/__init__.py:from .cudadrv.devicearray import (device_array, device_array_like, pinned,
numba/cuda/simulator/__init__.py:from .cudadrv import devicearray
numba/cuda/simulator/__init__.py:from .cudadrv.devices import require_context, gpus
numba/cuda/simulator/__init__.py:from .cudadrv.devices import get_context as current_context
numba/cuda/simulator/__init__.py:from .cudadrv.runtime import runtime
numba/cuda/simulator/__init__.py:# Ensure that any user code attempting to import cudadrv etc. gets the
numba/cuda/simulator/__init__.py:if config.ENABLE_CUDASIM:
numba/cuda/simulator/__init__.py:    from numba.cuda.simulator import cudadrv
numba/cuda/simulator/__init__.py:    sys.modules['numba.cuda.cudadrv'] = cudadrv
numba/cuda/simulator/__init__.py:    sys.modules['numba.cuda.cudadrv.devicearray'] = cudadrv.devicearray
numba/cuda/simulator/__init__.py:    sys.modules['numba.cuda.cudadrv.devices'] = cudadrv.devices
numba/cuda/simulator/__init__.py:    sys.modules['numba.cuda.cudadrv.driver'] = cudadrv.driver
numba/cuda/simulator/__init__.py:    sys.modules['numba.cuda.cudadrv.runtime'] = cudadrv.runtime
numba/cuda/simulator/__init__.py:    sys.modules['numba.cuda.cudadrv.drvapi'] = cudadrv.drvapi
numba/cuda/simulator/__init__.py:    sys.modules['numba.cuda.cudadrv.error'] = cudadrv.error
numba/cuda/simulator/__init__.py:    sys.modules['numba.cuda.cudadrv.nvvm'] = cudadrv.nvvm
numba/cuda/simulator/__init__.py:    sys.modules['numba.cuda.compiler'] = compiler
numba/cuda/printimpl.py:from numba.cuda import nvvmutils
numba/cuda/printimpl.py:        msg = ('CUDA print() cannot print more than 32 items. '
numba/cuda/cudadecl.py:from numba.cuda.types import dim3
numba/cuda/cudadecl.py:from numba import cuda
numba/cuda/cudadecl.py:from numba.cuda.compiler import declare_device_function_template
numba/cuda/cudadecl.py:class Cuda_array_decl(CallableTemplate):
numba/cuda/cudadecl.py:class Cuda_shared_array(Cuda_array_decl):
numba/cuda/cudadecl.py:    key = cuda.shared.array
numba/cuda/cudadecl.py:class Cuda_local_array(Cuda_array_decl):
numba/cuda/cudadecl.py:    key = cuda.local.array
numba/cuda/cudadecl.py:class Cuda_const_array_like(CallableTemplate):
numba/cuda/cudadecl.py:    key = cuda.const.array_like
numba/cuda/cudadecl.py:class Cuda_threadfence_device(ConcreteTemplate):
numba/cuda/cudadecl.py:    key = cuda.threadfence
numba/cuda/cudadecl.py:class Cuda_threadfence_block(ConcreteTemplate):
numba/cuda/cudadecl.py:    key = cuda.threadfence_block
numba/cuda/cudadecl.py:class Cuda_threadfence_system(ConcreteTemplate):
numba/cuda/cudadecl.py:    key = cuda.threadfence_system
numba/cuda/cudadecl.py:class Cuda_syncwarp(ConcreteTemplate):
numba/cuda/cudadecl.py:    key = cuda.syncwarp
numba/cuda/cudadecl.py:class Cuda_shfl_sync_intrinsic(ConcreteTemplate):
numba/cuda/cudadecl.py:    key = cuda.shfl_sync_intrinsic
numba/cuda/cudadecl.py:class Cuda_vote_sync_intrinsic(ConcreteTemplate):
numba/cuda/cudadecl.py:    key = cuda.vote_sync_intrinsic
numba/cuda/cudadecl.py:class Cuda_match_any_sync(ConcreteTemplate):
numba/cuda/cudadecl.py:    key = cuda.match_any_sync
numba/cuda/cudadecl.py:class Cuda_match_all_sync(ConcreteTemplate):
numba/cuda/cudadecl.py:    key = cuda.match_all_sync
numba/cuda/cudadecl.py:class Cuda_activemask(ConcreteTemplate):
numba/cuda/cudadecl.py:    key = cuda.activemask
numba/cuda/cudadecl.py:class Cuda_lanemask_lt(ConcreteTemplate):
numba/cuda/cudadecl.py:    key = cuda.lanemask_lt
numba/cuda/cudadecl.py:class Cuda_popc(ConcreteTemplate):
numba/cuda/cudadecl.py:    [here](http://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#bit-manipulations-intrinics)
numba/cuda/cudadecl.py:    key = cuda.popc
numba/cuda/cudadecl.py:class Cuda_fma(ConcreteTemplate):
numba/cuda/cudadecl.py:    [here](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#standard-c-library-intrinics)
numba/cuda/cudadecl.py:    key = cuda.fma
numba/cuda/cudadecl.py:class Cuda_hfma(ConcreteTemplate):
numba/cuda/cudadecl.py:    key = cuda.fp16.hfma
numba/cuda/cudadecl.py:class Cuda_cbrt(ConcreteTemplate):
numba/cuda/cudadecl.py:    key = cuda.cbrt
numba/cuda/cudadecl.py:class Cuda_brev(ConcreteTemplate):
numba/cuda/cudadecl.py:    key = cuda.brev
numba/cuda/cudadecl.py:class Cuda_clz(ConcreteTemplate):
numba/cuda/cudadecl.py:    [here](http://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#bit-manipulations-intrinics)
numba/cuda/cudadecl.py:    key = cuda.clz
numba/cuda/cudadecl.py:class Cuda_ffs(ConcreteTemplate):
numba/cuda/cudadecl.py:    [here](http://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#bit-manipulations-intrinics)
numba/cuda/cudadecl.py:    key = cuda.ffs
numba/cuda/cudadecl.py:class Cuda_selp(AbstractTemplate):
numba/cuda/cudadecl.py:    key = cuda.selp
numba/cuda/cudadecl.py:        # http://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp
numba/cuda/cudadecl.py:    class Cuda_fp16_unary(ConcreteTemplate):
numba/cuda/cudadecl.py:    return Cuda_fp16_unary
numba/cuda/cudadecl.py:    class Cuda_fp16_unary(AbstractTemplate):
numba/cuda/cudadecl.py:    return Cuda_fp16_unary
numba/cuda/cudadecl.py:    class Cuda_fp16_binary(ConcreteTemplate):
numba/cuda/cudadecl.py:    return Cuda_fp16_binary
numba/cuda/cudadecl.py:    class Cuda_fp16_cmp(ConcreteTemplate):
numba/cuda/cudadecl.py:    return Cuda_fp16_cmp
numba/cuda/cudadecl.py:    class Cuda_fp16_operator(AbstractTemplate):
numba/cuda/cudadecl.py:    return Cuda_fp16_operator
numba/cuda/cudadecl.py:Cuda_hadd = _genfp16_binary(cuda.fp16.hadd)
numba/cuda/cudadecl.py:Cuda_add = _genfp16_binary_operator(operator.add)
numba/cuda/cudadecl.py:Cuda_iadd = _genfp16_binary_operator(operator.iadd)
numba/cuda/cudadecl.py:Cuda_hsub = _genfp16_binary(cuda.fp16.hsub)
numba/cuda/cudadecl.py:Cuda_sub = _genfp16_binary_operator(operator.sub)
numba/cuda/cudadecl.py:Cuda_isub = _genfp16_binary_operator(operator.isub)
numba/cuda/cudadecl.py:Cuda_hmul = _genfp16_binary(cuda.fp16.hmul)
numba/cuda/cudadecl.py:Cuda_mul = _genfp16_binary_operator(operator.mul)
numba/cuda/cudadecl.py:Cuda_imul = _genfp16_binary_operator(operator.imul)
numba/cuda/cudadecl.py:Cuda_hmax = _genfp16_binary(cuda.fp16.hmax)
numba/cuda/cudadecl.py:Cuda_hmin = _genfp16_binary(cuda.fp16.hmin)
numba/cuda/cudadecl.py:Cuda_hneg = _genfp16_unary(cuda.fp16.hneg)
numba/cuda/cudadecl.py:Cuda_neg = _genfp16_unary_operator(operator.neg)
numba/cuda/cudadecl.py:Cuda_habs = _genfp16_unary(cuda.fp16.habs)
numba/cuda/cudadecl.py:Cuda_abs = _genfp16_unary_operator(abs)
numba/cuda/cudadecl.py:Cuda_heq = _genfp16_binary_comparison(cuda.fp16.heq)
numba/cuda/cudadecl.py:Cuda_hne = _genfp16_binary_comparison(cuda.fp16.hne)
numba/cuda/cudadecl.py:Cuda_hge = _genfp16_binary_comparison(cuda.fp16.hge)
numba/cuda/cudadecl.py:Cuda_hgt = _genfp16_binary_comparison(cuda.fp16.hgt)
numba/cuda/cudadecl.py:Cuda_hle = _genfp16_binary_comparison(cuda.fp16.hle)
numba/cuda/cudadecl.py:Cuda_hlt = _genfp16_binary_comparison(cuda.fp16.hlt)
numba/cuda/cudadecl.py:    class Cuda_atomic(AbstractTemplate):
numba/cuda/cudadecl.py:    return Cuda_atomic
numba/cuda/cudadecl.py:Cuda_atomic_add = _gen(cuda.atomic.add, all_numba_types)
numba/cuda/cudadecl.py:Cuda_atomic_sub = _gen(cuda.atomic.sub, all_numba_types)
numba/cuda/cudadecl.py:Cuda_atomic_max = _gen(cuda.atomic.max, all_numba_types)
numba/cuda/cudadecl.py:Cuda_atomic_min = _gen(cuda.atomic.min, all_numba_types)
numba/cuda/cudadecl.py:Cuda_atomic_nanmax = _gen(cuda.atomic.nanmax, all_numba_types)
numba/cuda/cudadecl.py:Cuda_atomic_nanmin = _gen(cuda.atomic.nanmin, all_numba_types)
numba/cuda/cudadecl.py:Cuda_atomic_and = _gen(cuda.atomic.and_, integer_numba_types)
numba/cuda/cudadecl.py:Cuda_atomic_or = _gen(cuda.atomic.or_, integer_numba_types)
numba/cuda/cudadecl.py:Cuda_atomic_xor = _gen(cuda.atomic.xor, integer_numba_types)
numba/cuda/cudadecl.py:Cuda_atomic_inc = _gen(cuda.atomic.inc, unsigned_int_numba_types)
numba/cuda/cudadecl.py:Cuda_atomic_dec = _gen(cuda.atomic.dec, unsigned_int_numba_types)
numba/cuda/cudadecl.py:Cuda_atomic_exch = _gen(cuda.atomic.exch, integer_numba_types)
numba/cuda/cudadecl.py:class Cuda_atomic_compare_and_swap(AbstractTemplate):
numba/cuda/cudadecl.py:    key = cuda.atomic.compare_and_swap
numba/cuda/cudadecl.py:class Cuda_atomic_cas(AbstractTemplate):
numba/cuda/cudadecl.py:    key = cuda.atomic.cas
numba/cuda/cudadecl.py:class Cuda_nanosleep(ConcreteTemplate):
numba/cuda/cudadecl.py:    key = cuda.nanosleep
numba/cuda/cudadecl.py:class CudaSharedModuleTemplate(AttributeTemplate):
numba/cuda/cudadecl.py:    key = types.Module(cuda.shared)
numba/cuda/cudadecl.py:        return types.Function(Cuda_shared_array)
numba/cuda/cudadecl.py:class CudaConstModuleTemplate(AttributeTemplate):
numba/cuda/cudadecl.py:    key = types.Module(cuda.const)
numba/cuda/cudadecl.py:        return types.Function(Cuda_const_array_like)
numba/cuda/cudadecl.py:class CudaLocalModuleTemplate(AttributeTemplate):
numba/cuda/cudadecl.py:    key = types.Module(cuda.local)
numba/cuda/cudadecl.py:        return types.Function(Cuda_local_array)
numba/cuda/cudadecl.py:class CudaAtomicTemplate(AttributeTemplate):
numba/cuda/cudadecl.py:    key = types.Module(cuda.atomic)
numba/cuda/cudadecl.py:        return types.Function(Cuda_atomic_add)
numba/cuda/cudadecl.py:        return types.Function(Cuda_atomic_sub)
numba/cuda/cudadecl.py:        return types.Function(Cuda_atomic_and)
numba/cuda/cudadecl.py:        return types.Function(Cuda_atomic_or)
numba/cuda/cudadecl.py:        return types.Function(Cuda_atomic_xor)
numba/cuda/cudadecl.py:        return types.Function(Cuda_atomic_inc)
numba/cuda/cudadecl.py:        return types.Function(Cuda_atomic_dec)
numba/cuda/cudadecl.py:        return types.Function(Cuda_atomic_exch)
numba/cuda/cudadecl.py:        return types.Function(Cuda_atomic_max)
numba/cuda/cudadecl.py:        return types.Function(Cuda_atomic_min)
numba/cuda/cudadecl.py:        return types.Function(Cuda_atomic_nanmin)
numba/cuda/cudadecl.py:        return types.Function(Cuda_atomic_nanmax)
numba/cuda/cudadecl.py:        return types.Function(Cuda_atomic_compare_and_swap)
numba/cuda/cudadecl.py:        return types.Function(Cuda_atomic_cas)
numba/cuda/cudadecl.py:class CudaFp16Template(AttributeTemplate):
numba/cuda/cudadecl.py:    key = types.Module(cuda.fp16)
numba/cuda/cudadecl.py:        return types.Function(Cuda_hadd)
numba/cuda/cudadecl.py:        return types.Function(Cuda_hsub)
numba/cuda/cudadecl.py:        return types.Function(Cuda_hmul)
numba/cuda/cudadecl.py:        return types.Function(Cuda_hneg)
numba/cuda/cudadecl.py:        return types.Function(Cuda_habs)
numba/cuda/cudadecl.py:        return types.Function(Cuda_hfma)
numba/cuda/cudadecl.py:        return types.Function(Cuda_heq)
numba/cuda/cudadecl.py:        return types.Function(Cuda_hne)
numba/cuda/cudadecl.py:        return types.Function(Cuda_hge)
numba/cuda/cudadecl.py:        return types.Function(Cuda_hgt)
numba/cuda/cudadecl.py:        return types.Function(Cuda_hle)
numba/cuda/cudadecl.py:        return types.Function(Cuda_hlt)
numba/cuda/cudadecl.py:        return types.Function(Cuda_hmax)
numba/cuda/cudadecl.py:        return types.Function(Cuda_hmin)
numba/cuda/cudadecl.py:class CudaModuleTemplate(AttributeTemplate):
numba/cuda/cudadecl.py:    key = types.Module(cuda)
numba/cuda/cudadecl.py:        return types.Module(cuda.cg)
numba/cuda/cudadecl.py:        return types.Module(cuda.shared)
numba/cuda/cudadecl.py:        return types.Function(Cuda_popc)
numba/cuda/cudadecl.py:        return types.Function(Cuda_brev)
numba/cuda/cudadecl.py:        return types.Function(Cuda_clz)
numba/cuda/cudadecl.py:        return types.Function(Cuda_ffs)
numba/cuda/cudadecl.py:        return types.Function(Cuda_fma)
numba/cuda/cudadecl.py:        return types.Function(Cuda_cbrt)
numba/cuda/cudadecl.py:        return types.Function(Cuda_threadfence_device)
numba/cuda/cudadecl.py:        return types.Function(Cuda_threadfence_block)
numba/cuda/cudadecl.py:        return types.Function(Cuda_threadfence_system)
numba/cuda/cudadecl.py:        return types.Function(Cuda_syncwarp)
numba/cuda/cudadecl.py:        return types.Function(Cuda_shfl_sync_intrinsic)
numba/cuda/cudadecl.py:        return types.Function(Cuda_vote_sync_intrinsic)
numba/cuda/cudadecl.py:        return types.Function(Cuda_match_any_sync)
numba/cuda/cudadecl.py:        return types.Function(Cuda_match_all_sync)
numba/cuda/cudadecl.py:        return types.Function(Cuda_activemask)
numba/cuda/cudadecl.py:        return types.Function(Cuda_lanemask_lt)
numba/cuda/cudadecl.py:        return types.Function(Cuda_selp)
numba/cuda/cudadecl.py:        return types.Function(Cuda_nanosleep)
numba/cuda/cudadecl.py:        return types.Module(cuda.atomic)
numba/cuda/cudadecl.py:        return types.Module(cuda.fp16)
numba/cuda/cudadecl.py:        return types.Module(cuda.const)
numba/cuda/cudadecl.py:        return types.Module(cuda.local)
numba/cuda/cudadecl.py:register_global(cuda, types.Module(cuda))
numba/cuda/cuda_fp16.h:* Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
numba/cuda/cuda_fp16.h:* subject to NVIDIA intellectual property rights under U.S. and
numba/cuda/cuda_fp16.h:* CONFIDENTIAL to NVIDIA and is being provided under the terms and
numba/cuda/cuda_fp16.h:* conditions of a form of NVIDIA software license agreement by and
numba/cuda/cuda_fp16.h:* between NVIDIA and Licensee ("License Agreement") or electronically
numba/cuda/cuda_fp16.h:* written consent of NVIDIA is prohibited.
numba/cuda/cuda_fp16.h:* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
numba/cuda/cuda_fp16.h:* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
numba/cuda/cuda_fp16.h:* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
numba/cuda/cuda_fp16.h:* \defgroup CUDA_MATH_INTRINSIC_HALF Half Precision Intrinsics
numba/cuda/cuda_fp16.h:* To use these functions, include the header file \p cuda_fp16.h in your program.
numba/cuda/cuda_fp16.h:* \defgroup CUDA_MATH__HALF_ARITHMETIC Half Arithmetic Functions
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH_INTRINSIC_HALF
numba/cuda/cuda_fp16.h:* To use these functions, include the header file \p cuda_fp16.h in your program.
numba/cuda/cuda_fp16.h:* \defgroup CUDA_MATH__HALF2_ARITHMETIC Half2 Arithmetic Functions
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH_INTRINSIC_HALF
numba/cuda/cuda_fp16.h:* To use these functions, include the header file \p cuda_fp16.h in your program.
numba/cuda/cuda_fp16.h:* \defgroup CUDA_MATH__HALF_COMPARISON Half Comparison Functions
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH_INTRINSIC_HALF
numba/cuda/cuda_fp16.h:* To use these functions, include the header file \p cuda_fp16.h in your program.
numba/cuda/cuda_fp16.h:* \defgroup CUDA_MATH__HALF2_COMPARISON Half2 Comparison Functions
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH_INTRINSIC_HALF
numba/cuda/cuda_fp16.h:* To use these functions, include the header file \p cuda_fp16.h in your program.
numba/cuda/cuda_fp16.h:* \defgroup CUDA_MATH__HALF_MISC Half Precision Conversion and Data Movement
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH_INTRINSIC_HALF
numba/cuda/cuda_fp16.h:* To use these functions, include the header file \p cuda_fp16.h in your program.
numba/cuda/cuda_fp16.h:* \defgroup CUDA_MATH__HALF_FUNCTIONS Half Math Functions
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH_INTRINSIC_HALF
numba/cuda/cuda_fp16.h:* To use these functions, include the header file \p cuda_fp16.h in your program.
numba/cuda/cuda_fp16.h:* \defgroup CUDA_MATH__HALF2_FUNCTIONS Half2 Math Functions
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH_INTRINSIC_HALF
numba/cuda/cuda_fp16.h:* To use these functions, include the header file \p cuda_fp16.h in your program.
numba/cuda/cuda_fp16.h:#ifndef __CUDA_FP16_H__
numba/cuda/cuda_fp16.h:#define __CUDA_FP16_H__
numba/cuda/cuda_fp16.h:#if defined(__CUDACC__)
numba/cuda/cuda_fp16.h:#define __CUDA_FP16_DECL__ static __device__ __inline__
numba/cuda/cuda_fp16.h:#define __CUDA_HOSTDEVICE_FP16_DECL__ static __host__ __device__ __inline__
numba/cuda/cuda_fp16.h:#define __CUDA_HOSTDEVICE_FP16_DECL__ static
numba/cuda/cuda_fp16.h:#endif /* defined(__CUDACC__) */
numba/cuda/cuda_fp16.h:#define __CUDA_FP16_TYPES_EXIST__
numba/cuda/cuda_fp16.h:/* Forward-declaration of structures defined in "cuda_fp16.hpp" */
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ __half __double2half(const double a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half(const float a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_rn(const float a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_rz(const float a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_rd(const float a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ __half __float2half_ru(const float a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ float __half2float(const __half a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __float2half2_rn(const float a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __floats2half2_rn(const float a, const float b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ float __low2float(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ float __high2float(const __half2 a);
numba/cuda/cuda_fp16.h:#if defined(__CUDACC__)
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ __half2 __float22half2_rn(const float2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ float2 __half22float2(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ int __half2int_rn(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ int __half2int_rz(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ int __half2int_rd(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ int __half2int_ru(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ __half __int2half_rn(const int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __int2half_rz(const int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __int2half_rd(const int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __int2half_ru(const int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ short int __half2short_rn(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ short int __half2short_rz(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ short int __half2short_rd(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ short int __half2short_ru(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ __half __short2half_rn(const short int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __short2half_rz(const short int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __short2half_rd(const short int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __short2half_ru(const short int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ unsigned int __half2uint_rn(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ unsigned int __half2uint_rz(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ unsigned int __half2uint_rd(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ unsigned int __half2uint_ru(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ __half __uint2half_rn(const unsigned int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __uint2half_rz(const unsigned int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __uint2half_rd(const unsigned int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __uint2half_ru(const unsigned int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ unsigned short int __half2ushort_rn(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ unsigned short int __half2ushort_rz(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ unsigned short int __half2ushort_rd(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ unsigned short int __half2ushort_ru(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ __half __ushort2half_rn(const unsigned short int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __ushort2half_rz(const unsigned short int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __ushort2half_rd(const unsigned short int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __ushort2half_ru(const unsigned short int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ unsigned long long int __half2ull_rn(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ unsigned long long int __half2ull_rz(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ unsigned long long int __half2ull_rd(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ unsigned long long int __half2ull_ru(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ __half __ull2half_rn(const unsigned long long int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __ull2half_rz(const unsigned long long int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __ull2half_rd(const unsigned long long int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __ull2half_ru(const unsigned long long int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ long long int __half2ll_rn(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ long long int __half2ll_rz(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ long long int __half2ll_rd(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ long long int __half2ll_ru(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_HOSTDEVICE_FP16_DECL__ __half __ll2half_rn(const long long int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __ll2half_rz(const long long int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __ll2half_rd(const long long int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __ll2half_ru(const long long int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half htrunc(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half hceil(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half hfloor(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half hrint(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 h2trunc(const __half2 h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 h2ceil(const __half2 h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 h2floor(const __half2 h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 h2rint(const __half2 h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __half2half2(const __half a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __lowhigh2highlow(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __lows2half2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __highs2half2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __high2half(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __low2half(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ int __hisinf(const __half a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __halves2half2(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __low2half2(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __high2half2(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ short int __half_as_short(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ unsigned short int __half_as_ushort(const __half h);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __short_as_half(const short int i);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __ushort_as_half(const unsigned short int i);
numba/cuda/cuda_fp16.h:#if __CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.h:#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 700
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl)) __half2 __shfl(const __half2 var, const int delta, const int width = warpSize);
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_up)) __half2 __shfl_up(const __half2 var, const unsigned int delta, const int width = warpSize);
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_down))__half2 __shfl_down(const __half2 var, const unsigned int delta, const int width = warpSize);
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_xor)) __half2 __shfl_xor(const __half2 var, const int delta, const int width = warpSize);
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl)) __half __shfl(const __half var, const int delta, const int width = warpSize);
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_up)) __half __shfl_up(const __half var, const unsigned int delta, const int width = warpSize);
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_down)) __half __shfl_down(const __half var, const unsigned int delta, const int width = warpSize);
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__shfl_xor)) __half __shfl_xor(const __half var, const int delta, const int width = warpSize);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __shfl_sync(const unsigned mask, const __half2 var, const int delta, const int width = warpSize);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __shfl_up_sync(const unsigned mask, const __half2 var, const unsigned int delta, const int width = warpSize);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __shfl_down_sync(const unsigned mask, const __half2 var, const unsigned int delta, const int width = warpSize);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __shfl_xor_sync(const unsigned mask, const __half2 var, const int delta, const int width = warpSize);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __shfl_sync(const unsigned mask, const __half var, const int delta, const int width = warpSize);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __shfl_up_sync(const unsigned mask, const __half var, const unsigned int delta, const int width = warpSize);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __shfl_down_sync(const unsigned mask, const __half var, const unsigned int delta, const int width = warpSize);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __shfl_xor_sync(const unsigned mask, const __half var, const int delta, const int width = warpSize);
numba/cuda/cuda_fp16.h:#endif /*__CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__) */
numba/cuda/cuda_fp16.h:#if defined(__cplusplus) && ( __CUDA_ARCH__ >=320 || !defined(__CUDA_ARCH__) )
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __ldg(const  __half2 *const ptr);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __ldg(const __half *const ptr);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __ldcg(const  __half2 *const ptr);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __ldcg(const __half *const ptr);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __ldca(const  __half2 *const ptr);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __ldca(const __half *const ptr);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __ldcs(const  __half2 *const ptr);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __ldcs(const __half *const ptr);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __ldlu(const  __half2 *const ptr);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __ldlu(const __half *const ptr);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __ldcv(const  __half2 *const ptr);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __ldcv(const __half *const ptr);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ void __stwb(__half2 *const ptr, const __half2 value);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ void __stwb(__half *const ptr, const __half value);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ void __stcg(__half2 *const ptr, const __half2 value);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ void __stcg(__half *const ptr, const __half value);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ void __stcs(__half2 *const ptr, const __half2 value);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ void __stcs(__half *const ptr, const __half value);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ void __stwt(__half2 *const ptr, const __half2 value);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_MISC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ void __stwt(__half *const ptr, const __half value);
numba/cuda/cuda_fp16.h:#endif /*defined(__cplusplus) && ( __CUDA_ARCH__ >=320 || !defined(__CUDA_ARCH__) )*/
numba/cuda/cuda_fp16.h:#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __heq2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hne2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hle2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hge2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hlt2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hgt2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hequ2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hneu2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hleu2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hgeu2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hltu2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hgtu2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hisnan2(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hadd2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hsub2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hmul2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __h2div(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __habs2(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hadd2_sat(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hsub2_sat(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hmul2_sat(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hneg2(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __habs(const __half a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __hadd(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __hsub(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __hmul(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__  __half __hdiv(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __hadd_sat(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __hsub_sat(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __hmul_sat(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __hfma(const __half a, const __half b, const __half c);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __hfma_sat(const __half a, const __half b, const __half c);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __hneg(const __half a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hbeq2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hbne2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hble2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hbge2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hblt2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hbgt2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hbequ2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hbneu2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hbleu2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hbgeu2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hbltu2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hbgtu2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __heq(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hne(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hle(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hge(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hlt(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hgt(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hequ(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hneu(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hleu(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hgeu(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hltu(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hgtu(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ bool __hisnan(const __half a);
numba/cuda/cuda_fp16.h:#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __hmax(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __hmin(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __hmax_nan(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __hmin_nan(const __half a, const __half b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half __hfma_relu(const __half a, const __half b, const __half c);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hmax2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hmin2(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hmax2_nan(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_COMPARISON
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hmin2_nan(const __half2 a, const __half2 b);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hfma2_relu(const __half2 a, const __half2 b, const __half2 c);
numba/cuda/cuda_fp16.h:#endif /*__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)*/
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_ARITHMETIC
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 __hcmadd(const __half2 a, const __half2 b, const __half2 c);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half hsqrt(const __half a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half hrsqrt(const __half a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half hrcp(const __half a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half hlog(const __half a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half hlog2(const __half a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half hlog10(const __half a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half hexp(const __half a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half hexp2(const __half a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half hexp10(const __half a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half hcos(const __half a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half hsin(const __half a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 h2sqrt(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 h2rsqrt(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 h2rcp(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 h2log(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 h2log2(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 h2log10(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 h2exp(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 h2exp2(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 h2exp10(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 h2cos(const __half2 a);
numba/cuda/cuda_fp16.h:* \ingroup CUDA_MATH__HALF2_FUNCTIONS
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 h2sin(const __half2 a);
numba/cuda/cuda_fp16.h:#endif /*if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
numba/cuda/cuda_fp16.h:#if __CUDA_ARCH__ >= 600 || !defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half2 atomicAdd(__half2 *const address, const __half2 val);
numba/cuda/cuda_fp16.h:#endif /*if __CUDA_ARCH__ >= 600 || !defined(__CUDA_ARCH__)*/
numba/cuda/cuda_fp16.h:#if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)
numba/cuda/cuda_fp16.h:__CUDA_FP16_DECL__ __half atomicAdd(__half *const address, const __half val);
numba/cuda/cuda_fp16.h:#endif /*if __CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__)*/
numba/cuda/cuda_fp16.h:#endif /* defined(__CUDACC__) */
numba/cuda/cuda_fp16.h:#undef __CUDA_FP16_DECL__
numba/cuda/cuda_fp16.h:#undef __CUDA_HOSTDEVICE_FP16_DECL__
numba/cuda/cuda_fp16.h:#include "cuda_fp16.hpp"
numba/cuda/cuda_fp16.h:#endif /* end of include guard: __CUDA_FP16_H__ */
numba/cuda/simulator_init.py:# numba.cuda.__init__.
numba/cuda/simulator_init.py:    """Returns a boolean to indicate the availability of a CUDA GPU.
numba/cuda/simulator_init.py:def cuda_error():
numba/cuda/simulator_init.py:    """Returns None or an exception if the CUDA driver fails to initialize.
numba/cuda/args.py:        from .cudadrv.devicearray import auto_device
numba/cuda/args.py:        from .cudadrv.devicearray import auto_device
numba/cuda/args.py:        from .cudadrv.devicearray import auto_device
numba/cuda/kernels/reduction.py:A library written in CUDA Python for generating reduction kernels
numba/cuda/kernels/reduction.py:def _gpu_reduce_factory(fn, nbtype):
numba/cuda/kernels/reduction.py:    from numba import cuda
numba/cuda/kernels/reduction.py:    reduce_op = cuda.jit(device=True)(fn)
numba/cuda/kernels/reduction.py:    @cuda.jit(device=True)
numba/cuda/kernels/reduction.py:        tid = cuda.threadIdx.x
numba/cuda/kernels/reduction.py:        cuda.syncwarp()
numba/cuda/kernels/reduction.py:            cuda.syncwarp()
numba/cuda/kernels/reduction.py:    @cuda.jit(device=True)
numba/cuda/kernels/reduction.py:        tid = cuda.threadIdx.x
numba/cuda/kernels/reduction.py:        blkid = cuda.blockIdx.x
numba/cuda/kernels/reduction.py:        blksz = cuda.blockDim.x
numba/cuda/kernels/reduction.py:        gridsz = cuda.gridDim.x
numba/cuda/kernels/reduction.py:        cuda.syncthreads()
numba/cuda/kernels/reduction.py:        cuda.syncthreads()
numba/cuda/kernels/reduction.py:            cuda.syncwarp()
numba/cuda/kernels/reduction.py:    @cuda.jit(device=True)
numba/cuda/kernels/reduction.py:        tid = cuda.threadIdx.x
numba/cuda/kernels/reduction.py:        blkid = cuda.blockIdx.x
numba/cuda/kernels/reduction.py:        blksz = cuda.blockDim.x
numba/cuda/kernels/reduction.py:        tid = cuda.threadIdx.x
numba/cuda/kernels/reduction.py:        cuda.syncthreads()
numba/cuda/kernels/reduction.py:        cuda.syncthreads()
numba/cuda/kernels/reduction.py:    def gpu_reduce_block_strided(arr, partials, init, use_init):
numba/cuda/kernels/reduction.py:        tid = cuda.threadIdx.x
numba/cuda/kernels/reduction.py:        sm_partials = cuda.shared.array((_NUMWARPS, inner_sm_size),
numba/cuda/kernels/reduction.py:        if cuda.blockDim.x == max_blocksize:
numba/cuda/kernels/reduction.py:        if use_init and tid == 0 and cuda.blockIdx.x == 0:
numba/cuda/kernels/reduction.py:    return cuda.jit(gpu_reduce_block_strided)
numba/cuda/kernels/reduction.py:                        reduction. It will be compiled as a CUDA device
numba/cuda/kernels/reduction.py:                        function using ``cuda.jit(device=True)``.
numba/cuda/kernels/reduction.py:            kernel = _gpu_reduce_factory(self._functor, from_dtype(dtype))
numba/cuda/kernels/reduction.py:        :param stream: Optional CUDA stream in which to perform the reduction.
numba/cuda/kernels/reduction.py:        from numba import cuda
numba/cuda/kernels/reduction.py:        # Perform the reduction on the GPU
numba/cuda/kernels/reduction.py:        partials = cuda.device_array(shape=partials_size, dtype=arr.dtype)
numba/cuda/kernels/transpose.py:from numba import cuda
numba/cuda/kernels/transpose.py:from numba.cuda.cudadrv.driver import driver
numba/cuda/kernels/transpose.py:    http://devblogs.nvidia.com/parallelforall/efficient-matrix-transpose-cuda-cc/
numba/cuda/kernels/transpose.py:        b = cuda.cudadrv.devicearray.DeviceNDArray(
numba/cuda/kernels/transpose.py:    @cuda.jit
numba/cuda/kernels/transpose.py:        tile = cuda.shared.array(shape=tile_shape, dtype=dt)
numba/cuda/kernels/transpose.py:        tx = cuda.threadIdx.x
numba/cuda/kernels/transpose.py:        ty = cuda.threadIdx.y
numba/cuda/kernels/transpose.py:        bx = cuda.blockIdx.x * cuda.blockDim.x
numba/cuda/kernels/transpose.py:        by = cuda.blockIdx.y * cuda.blockDim.y
numba/cuda/kernels/transpose.py:        cuda.syncthreads()
numba/cuda/codegen.py:from .cudadrv import devices, driver, nvvm, runtime
numba/cuda/codegen.py:from numba.cuda.cudadrv.libs import get_cudalib
numba/cuda/codegen.py:CUDA_TRIPLE = 'nvptx64-nvidia-cuda'
numba/cuda/codegen.py:                   "to install the CUDA toolkit and ensure that "
numba/cuda/codegen.py:class CUDACodeLibrary(serialize.ReduceMixin, CodeLibrary):
numba/cuda/codegen.py:    The CUDACodeLibrary generates PTX, SASS, cubins for multiple different
numba/cuda/codegen.py:        # Should we link libcudadevrt?
numba/cuda/codegen.py:        self.needs_cudadevrt = False
numba/cuda/codegen.py:        if self.needs_cudadevrt:
numba/cuda/codegen.py:            linker.add_file_guess_ext(get_cudalib('cudadevrt', static=True))
numba/cuda/codegen.py:            raise RuntimeError('CUDACodeLibrary only supports one module')
numba/cuda/codegen.py:            msg = 'Cannot pickle CUDACodeLibrary with linking files'
numba/cuda/codegen.py:            raise RuntimeError('Cannot pickle unfinalized CUDACodeLibrary')
numba/cuda/codegen.py:            needs_cudadevrt=self.needs_cudadevrt
numba/cuda/codegen.py:                 needs_cudadevrt):
numba/cuda/codegen.py:        instance.needs_cudadevrt = needs_cudadevrt
numba/cuda/codegen.py:class JITCUDACodegen(Codegen):
numba/cuda/codegen.py:    This codegen implementation for CUDA only generates optimized LLVM IR.
numba/cuda/codegen.py:    Generation of PTX code is done separately (see numba.cuda.compiler).
numba/cuda/codegen.py:    _library_class = CUDACodeLibrary
numba/cuda/codegen.py:        ir_module.triple = CUDA_TRIPLE
numba/cuda/nvvmutils.py:from .cudadrv import nvvm
numba/cuda/nvvmutils.py:def declare_cudaCGGetIntrinsicHandle(lmod):
numba/cuda/nvvmutils.py:    fname = 'cudaCGGetIntrinsicHandle'
numba/cuda/nvvmutils.py:def declare_cudaCGSynchronize(lmod):
numba/cuda/nvvmutils.py:    fname = 'cudaCGSynchronize'
numba/cuda/vector_types.py:# CUDA built-in Vector Types
numba/cuda/vector_types.py:# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types
numba/cuda/vector_types.py:from numba.cuda import stubs
numba/cuda/vector_types.py:from numba.cuda.errors import CudaLoweringError
numba/cuda/vector_types.py:        The handle to be used in cuda kernel.
numba/cuda/vector_types.py:                raise CudaLoweringError(
numba/cuda/cuda_paths.py:        ('Conda environment (NVIDIA package)', get_nvidia_libdevice_ctk()),
numba/cuda/cuda_paths.py:        ('CUDA_HOME', get_cuda_home('nvvm', 'libdevice')),
numba/cuda/cuda_paths.py:        ('Conda environment (NVIDIA package)', get_nvidia_nvvm_ctk()),
numba/cuda/cuda_paths.py:        ('CUDA_HOME', get_cuda_home(*_nvvm_lib_dir())),
numba/cuda/cuda_paths.py:def _cudalib_path():
numba/cuda/cuda_paths.py:def _cuda_home_static_cudalib_path():
numba/cuda/cuda_paths.py:def _get_cudalib_dir_path_decision():
numba/cuda/cuda_paths.py:        ('Conda environment (NVIDIA package)', get_nvidia_cudalib_ctk()),
numba/cuda/cuda_paths.py:        ('CUDA_HOME', get_cuda_home(_cudalib_path())),
numba/cuda/cuda_paths.py:        ('System', get_system_ctk(_cudalib_path())),
numba/cuda/cuda_paths.py:def _get_static_cudalib_dir_path_decision():
numba/cuda/cuda_paths.py:        ('Conda environment (NVIDIA package)', get_nvidia_static_cudalib_ctk()),
numba/cuda/cuda_paths.py:        ('CUDA_HOME', get_cuda_home(*_cuda_home_static_cudalib_path())),
numba/cuda/cuda_paths.py:        ('System', get_system_ctk(_cudalib_path())),
numba/cuda/cuda_paths.py:def _get_cudalib_dir():
numba/cuda/cuda_paths.py:    by, libdir = _get_cudalib_dir_path_decision()
numba/cuda/cuda_paths.py:def _get_static_cudalib_dir():
numba/cuda/cuda_paths.py:    by, libdir = _get_static_cudalib_dir_path_decision()
numba/cuda/cuda_paths.py:    """Return path to system-wide cudatoolkit; or, None if it doesn't exist.
numba/cuda/cuda_paths.py:        # Is cuda alias to /usr/local/cuda?
numba/cuda/cuda_paths.py:        # We are intentionally not getting versioned cuda installation.
numba/cuda/cuda_paths.py:        base = '/usr/local/cuda'
numba/cuda/cuda_paths.py:    """Return path to directory containing the shared libraries of cudatoolkit.
numba/cuda/cuda_paths.py:    # Assume the existence of NVVM to imply cudatoolkit installed
numba/cuda/cuda_paths.py:def get_nvidia_nvvm_ctk():
numba/cuda/cuda_paths.py:    # Assume the existence of NVVM in the conda env implies that a CUDA toolkit
numba/cuda/cuda_paths.py:    libdir = os.path.join(sys.prefix, 'nvvm', _cudalib_path())
numba/cuda/cuda_paths.py:        libdir = os.path.join(sys.prefix, 'Library', 'nvvm', _cudalib_path())
numba/cuda/cuda_paths.py:            # If that doesn't exist either, assume we don't have the NVIDIA
numba/cuda/cuda_paths.py:def get_nvidia_libdevice_ctk():
numba/cuda/cuda_paths.py:    nvvm_ctk = get_nvidia_nvvm_ctk()
numba/cuda/cuda_paths.py:def get_nvidia_cudalib_ctk():
numba/cuda/cuda_paths.py:    """Return path to directory containing the shared libraries of cudatoolkit.
numba/cuda/cuda_paths.py:    nvvm_ctk = get_nvidia_nvvm_ctk()
numba/cuda/cuda_paths.py:def get_nvidia_static_cudalib_ctk():
numba/cuda/cuda_paths.py:    """Return path to directory containing the static libraries of cudatoolkit.
numba/cuda/cuda_paths.py:    nvvm_ctk = get_nvidia_nvvm_ctk()
numba/cuda/cuda_paths.py:        # Location specific to CUDA 11.x packages on Windows
numba/cuda/cuda_paths.py:        # Linux, or Windows with CUDA 12.x packages
numba/cuda/cuda_paths.py:def get_cuda_home(*subdirs):
numba/cuda/cuda_paths.py:    """Get paths of CUDA_HOME.
numba/cuda/cuda_paths.py:    cuda_home = os.environ.get('CUDA_HOME')
numba/cuda/cuda_paths.py:    if cuda_home is None:
numba/cuda/cuda_paths.py:        # Try Windows CUDA installation without Anaconda
numba/cuda/cuda_paths.py:        cuda_home = os.environ.get('CUDA_PATH')
numba/cuda/cuda_paths.py:    if cuda_home is not None:
numba/cuda/cuda_paths.py:        return os.path.join(cuda_home, *subdirs)
numba/cuda/cuda_paths.py:def get_cuda_paths():
numba/cuda/cuda_paths.py:    - "cudalib_dir": directory_path
numba/cuda/cuda_paths.py:    if hasattr(get_cuda_paths, '_cached_result'):
numba/cuda/cuda_paths.py:        return get_cuda_paths._cached_result
numba/cuda/cuda_paths.py:            'cudalib_dir': _get_cudalib_dir(),
numba/cuda/cuda_paths.py:            'static_cudalib_dir': _get_static_cudalib_dir(),
numba/cuda/cuda_paths.py:        get_cuda_paths._cached_result = d
numba/cuda/cuda_paths.py:    Return the Debian NVIDIA Maintainers-packaged libdevice location, if it
numba/cuda/cuda_paths.py:    pkg_libdevice_location = '/usr/lib/nvidia-cuda-toolkit/libdevice'
numba/cuda/libdevicefuncs.py:    # argument. If a NaN is required, one can be obtained in CUDA Python by
numba/cuda/libdevicefuncs.py:    signature of the stub function used to call it from CUDA Python.
numba/cuda/libdevicefuncs.py:# python -c "from numba.cuda.libdevicefuncs import generate_stubs; \
numba/cuda/libdevicefuncs.py:#            generate_stubs()" > numba/cuda/libdevice.py
numba/cuda/libdevicefuncs.py:See https://docs.nvidia.com/cuda/libdevice-users-guide/{func}.html
numba/cuda/dispatcher.py:from numba.cuda.api import get_current_device
numba/cuda/dispatcher.py:from numba.cuda.args import wrap_arg
numba/cuda/dispatcher.py:from numba.cuda.compiler import compile_cuda, CUDACompiler
numba/cuda/dispatcher.py:from numba.cuda.cudadrv import driver
numba/cuda/dispatcher.py:from numba.cuda.cudadrv.devices import get_context
numba/cuda/dispatcher.py:from numba.cuda.descriptor import cuda_target
numba/cuda/dispatcher.py:from numba.cuda.errors import (missing_launch_config_msg,
numba/cuda/dispatcher.py:from numba.cuda import types as cuda_types
numba/cuda/dispatcher.py:from numba import cuda
numba/cuda/dispatcher.py:cuda_fp16_math_funcs = ['hsin', 'hcos',
numba/cuda/dispatcher.py:    CUDA Kernel specialized for a given set of argument types. When called, this
numba/cuda/dispatcher.py:        # CUDA target, _Kernel instances are stored instead, so we provide this
numba/cuda/dispatcher.py:        # attribute here to avoid duplicating nopython_signatures() in the CUDA
numba/cuda/dispatcher.py:        cres = compile_cuda(self.py_func, types.void, self.argtypes,
numba/cuda/dispatcher.py:        lib, kernel = tgt_ctx.prepare_cuda_kernel(cres.library, cres.fndesc,
numba/cuda/dispatcher.py:        self.cooperative = 'cudaCGGetIntrinsicHandle' in lib.get_asm_str()
numba/cuda/dispatcher.py:        # We need to link against cudadevrt if grid sync is being used.
numba/cuda/dispatcher.py:            lib.needs_cudadevrt = True
numba/cuda/dispatcher.py:        res = [fn for fn in cuda_fp16_math_funcs
numba/cuda/dispatcher.py:        # - There are no referenced environments in CUDA.
numba/cuda/dispatcher.py:        Force binding to current CUDA context
numba/cuda/dispatcher.py:        if config.CUDA_LOW_OCCUPANCY_WARNINGS:
numba/cuda/dispatcher.py:            # some very small GPUs might only have 4 SMs, but an H100-SXM5 has
numba/cuda/dispatcher.py:                msg = (f"Grid size {grid_size} will likely result in GPU "
numba/cuda/dispatcher.py:class CUDACacheImpl(CacheImpl):
numba/cuda/dispatcher.py:        # CUDA Kernels are always cachable - the reasons for an entity not to
numba/cuda/dispatcher.py:        # neither of which apply to CUDA kernels.
numba/cuda/dispatcher.py:class CUDACache(Cache):
numba/cuda/dispatcher.py:    Implements a cache that saves and loads CUDA kernels and compile results.
numba/cuda/dispatcher.py:    _impl_class = CUDACacheImpl
numba/cuda/dispatcher.py:        # initialized. To initialize the correct (i.e. CUDA) target, we need to
numba/cuda/dispatcher.py:        # enforce that the current target is the CUDA target.
numba/cuda/dispatcher.py:        with target_override('cuda'):
numba/cuda/dispatcher.py:class CUDADispatcher(Dispatcher, serialize.ReduceMixin):
numba/cuda/dispatcher.py:    CUDA Dispatcher object. When configured and called, the dispatcher will
numba/cuda/dispatcher.py:    created using the :func:`numba.cuda.jit` decorator.
numba/cuda/dispatcher.py:    # presently unsupported on CUDA, so we can leave this as False in all
numba/cuda/dispatcher.py:    targetdescr = cuda_target
numba/cuda/dispatcher.py:    def __init__(self, py_func, targetoptions, pipeline_class=CUDACompiler):
numba/cuda/dispatcher.py:        # The following properties are for specialization of CUDADispatchers. A
numba/cuda/dispatcher.py:        # specialized CUDADispatcher is one that is compiled for exactly one
numba/cuda/dispatcher.py:        return cuda_types.CUDADispatcher(self)
numba/cuda/dispatcher.py:        self._cache = CUDACache(self.py_func)
numba/cuda/dispatcher.py:        - the kernel maps the Global Thread ID ``cuda.grid(1)`` to tasks on a
numba/cuda/dispatcher.py:        - `stream` the CUDA stream used for the current call to the kernel
numba/cuda/dispatcher.py:            kernel = _dispatcher.Dispatcher._cuda_call(self, *args)
numba/cuda/dispatcher.py:        # the CUDA Array Interface.
numba/cuda/dispatcher.py:            if cuda.is_cuda_array(val):
numba/cuda/dispatcher.py:                return typeof(cuda.as_cuda_array(val, sync=False),
numba/cuda/dispatcher.py:        specialization = CUDADispatcher(self.py_func,
numba/cuda/dispatcher.py:                cres = compile_cuda(self.py_func, return_type, args,
numba/cuda/dispatcher.py:        self._insert(c_sig, kernel, cuda=True)
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_abs.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_acos.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_acosf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_acosh.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_acoshf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_asin.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_asinf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_asinh.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_asinhf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atan.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atan2.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atan2f.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atanf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atanh.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atanhf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_brev.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_brevll.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_byte_perm.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cbrt.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cbrtf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ceil.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ceilf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_clz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_clzll.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_copysign.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_copysignf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cos.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cosf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cosh.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_coshf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cospi.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cospif.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dadd_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dadd_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dadd_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dadd_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ddiv_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ddiv_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ddiv_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ddiv_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dmul_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dmul_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dmul_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dmul_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2float_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2float_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2float_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2float_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2hiint.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2int_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2int_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2int_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2int_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ll_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ll_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ll_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ll_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2loint.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2uint_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2uint_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2uint_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2uint_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ull_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ull_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ull_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ull_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double_as_longlong.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_drcp_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_drcp_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_drcp_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_drcp_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dsqrt_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dsqrt_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dsqrt_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dsqrt_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfc.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfcf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfcinv.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfcinvf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfcx.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfcxf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erff.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfinv.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfinvf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_exp.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_exp10.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_exp10f.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_exp2.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_exp2f.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_expf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_expm1.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_expm1f.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fabs.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fabsf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fadd_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fadd_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fadd_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fadd_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_cosf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_exp10f.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_expf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_fdividef.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_log10f.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_log2f.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_logf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_powf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_sincosf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_sinf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_tanf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdim.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdimf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdiv_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdiv_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdiv_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdiv_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ffs.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ffsll.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_finitef.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2half_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2int_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2int_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2int_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2int_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ll_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ll_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ll_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ll_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2uint_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2uint_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2uint_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2uint_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ull_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ull_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ull_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ull_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float_as_int.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_floor.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_floorf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fma.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fma_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fma_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fma_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fma_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaf_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaf_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaf_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaf_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmax.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaxf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmin.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fminf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmod.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmodf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmul_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmul_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmul_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmul_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frcp_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frcp_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frcp_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frcp_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frexp.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frexpf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frsqrt_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsqrt_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsqrt_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsqrt_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsqrt_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsub_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsub_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsub_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsub_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_hadd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_half2float.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_hiloint2double.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_hypot.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_hypotf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ilogb.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ilogbf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int2double_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int2float_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int2float_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int2float_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int2float_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int_as_float.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_isfinited.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_isinfd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_isinff.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_isnand.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_isnanf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_j0.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_j0f.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_j1.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_j1f.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_jn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_jnf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ldexp.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ldexpf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_lgamma.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_lgammaf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2double_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2double_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2double_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2double_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2float_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2float_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2float_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2float_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llabs.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llmax.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llmin.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llrint.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llrintf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llround.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llroundf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log10.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log10f.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log1p.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log1pf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log2.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log2f.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_logb.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_logbf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_logf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_longlong_as_double.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_max.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_min.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_modf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_modff.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_mul24.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_mul64hi.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_mulhi.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_nearbyint.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_nearbyintf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_nextafter.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_nextafterf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_normcdf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_normcdff.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_normcdfinv.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_normcdfinvf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_popc.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_popcll.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_pow.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_powf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_powi.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_powif.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rcbrt.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rcbrtf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_remainder.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_remainderf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_remquo.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_remquof.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rhadd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rint.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rintf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_round.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_roundf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rsqrt.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rsqrtf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sad.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_saturatef.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_scalbn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_scalbnf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_signbitd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_signbitf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sin.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sincos.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sincosf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sincospi.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sincospif.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sinf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sinh.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sinhf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sinpi.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sinpif.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sqrt.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sqrtf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tan.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tanf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tanh.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tanhf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tgamma.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tgammaf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_trunc.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_truncf.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uhadd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uint2double_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uint2float_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uint2float_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uint2float_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uint2float_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2double_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2double_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2double_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2double_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2float_rd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2float_rn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2float_ru.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2float_rz.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ullmax.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ullmin.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_umax.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_umin.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_umul24.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_umul64hi.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_umulhi.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_urhadd.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_usad.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_y0.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_y0f.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_y1.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_y1f.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_yn.html
numba/cuda/libdevice.py:    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ynf.html
numba/cuda/intrinsics.py:from numba import cuda, types
numba/cuda/intrinsics.py:from numba.cuda import nvvmutils
numba/cuda/intrinsics.py:from numba.cuda.extending import intrinsic
numba/cuda/intrinsics.py:        cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
numba/cuda/intrinsics.py:        cuda.blockDim.x * cuda.gridDim.x
numba/cuda/intrinsics.py:@overload_attribute(types.Module(cuda), 'warpsize', target='cuda')
numba/cuda/intrinsics.py:def cuda_warpsize(mod):
numba/cuda/intrinsics.py:    An extension to numba.cuda.syncthreads where the return value is a count
numba/cuda/intrinsics.py:    An extension to numba.cuda.syncthreads where 1 is returned if predicate is
numba/cuda/intrinsics.py:    An extension to numba.cuda.syncthreads where 1 is returned if predicate is
numba/cuda/vectorizers.py:from numba import cuda
numba/cuda/vectorizers.py:from numba.cuda import deviceufunc
numba/cuda/vectorizers.py:from numba.cuda.deviceufunc import (UFuncMechanism, GeneralizedUFunc,
numba/cuda/vectorizers.py:class CUDAUFuncDispatcher(object):
numba/cuda/vectorizers.py:    Invoke the CUDA ufunc specialization for the given inputs.
numba/cuda/vectorizers.py:        *args: numpy arrays or DeviceArrayBase (created by cuda.to_device).
numba/cuda/vectorizers.py:            stream -- cuda stream; when defined, asynchronous mode is used.
numba/cuda/vectorizers.py:        return CUDAUFuncMechanism.call(self.functions, args, kws)
numba/cuda/vectorizers.py:        gpu_mems = []
numba/cuda/vectorizers.py:        stream = stream or cuda.stream()
numba/cuda/vectorizers.py:            if cuda.cudadrv.devicearray.is_cuda_ndarray(arg):
numba/cuda/vectorizers.py:                mem = cuda.to_device(arg, stream)
numba/cuda/vectorizers.py:            out = self.__reduce(mem, gpu_mems, stream)
numba/cuda/vectorizers.py:    def __reduce(self, mem, gpu_mems, stream):
numba/cuda/vectorizers.py:            gpu_mems.append(fatcut)
numba/cuda/vectorizers.py:            gpu_mems.append(thincut)
numba/cuda/vectorizers.py:            out = self.__reduce(fatcut, gpu_mems, stream)
numba/cuda/vectorizers.py:            gpu_mems.append(out)
numba/cuda/vectorizers.py:            gpu_mems.append(left)
numba/cuda/vectorizers.py:            gpu_mems.append(right)
numba/cuda/vectorizers.py:                return self.__reduce(left, gpu_mems, stream)
numba/cuda/vectorizers.py:class _CUDAGUFuncCallSteps(GUFuncCallSteps):
numba/cuda/vectorizers.py:        return cuda.is_cuda_array(obj)
numba/cuda/vectorizers.py:        # We don't want to call as_cuda_array on objects that are already Numba
numba/cuda/vectorizers.py:        if cuda.cudadrv.devicearray.is_cuda_ndarray(obj):
numba/cuda/vectorizers.py:        return cuda.as_cuda_array(obj)
numba/cuda/vectorizers.py:        return cuda.to_device(hostary, stream=self._stream)
numba/cuda/vectorizers.py:        return cuda.device_array(shape=shape, dtype=dtype, stream=self._stream)
numba/cuda/vectorizers.py:class CUDAGeneralizedUFunc(GeneralizedUFunc):
numba/cuda/vectorizers.py:        return _CUDAGUFuncCallSteps
numba/cuda/vectorizers.py:        return cuda.cudadrv.devicearray.DeviceNDArray(shape=shape,
numba/cuda/vectorizers.py:                                                      gpu_data=ary.gpu_data)
numba/cuda/vectorizers.py:        return cuda.cudadrv.devicearray.DeviceNDArray(shape=newshape,
numba/cuda/vectorizers.py:                                                      gpu_data=ary.gpu_data)
numba/cuda/vectorizers.py:class CUDAUFuncMechanism(UFuncMechanism):
numba/cuda/vectorizers.py:    Provide CUDA specialization
numba/cuda/vectorizers.py:        return cuda.is_cuda_array(obj)
numba/cuda/vectorizers.py:        # We don't want to call as_cuda_array on objects that are already Numba
numba/cuda/vectorizers.py:        if cuda.cudadrv.devicearray.is_cuda_ndarray(obj):
numba/cuda/vectorizers.py:        return cuda.as_cuda_array(obj)
numba/cuda/vectorizers.py:        return cuda.to_device(hostary, stream=stream)
numba/cuda/vectorizers.py:        return cuda.device_array(shape=shape, dtype=dtype, stream=stream)
numba/cuda/vectorizers.py:        return cuda.cudadrv.devicearray.DeviceNDArray(shape=shape,
numba/cuda/vectorizers.py:                                                      gpu_data=ary.gpu_data)
numba/cuda/vectorizers.py:    __tid__ = __cuda__.grid(1)
numba/cuda/vectorizers.py:class CUDAVectorize(deviceufunc.DeviceVectorize):
numba/cuda/vectorizers.py:        cudevfn = cuda.jit(sig, device=True, inline=True)(self.pyfunc)
numba/cuda/vectorizers.py:        glbl.update({'__cuda__': cuda,
numba/cuda/vectorizers.py:        return cuda.jit(fnobj)
numba/cuda/vectorizers.py:        return CUDAUFuncDispatcher(self.kernelmap, self.pyfunc)
numba/cuda/vectorizers.py:# Generalized CUDA ufuncs
numba/cuda/vectorizers.py:    __tid__ = __cuda__.grid(1)
numba/cuda/vectorizers.py:class CUDAGUFuncVectorize(deviceufunc.DeviceGUFuncVectorize):
numba/cuda/vectorizers.py:        return CUDAGeneralizedUFunc(kernelmap=self.kernelmap,
numba/cuda/vectorizers.py:        return cuda.jit(sig)(fnobj)
numba/cuda/vectorizers.py:        corefn = cuda.jit(sig, device=True)(self.pyfunc)
numba/cuda/vectorizers.py:        glbls.update({'__cuda__': cuda,
numba/cuda/__init__.py:if config.ENABLE_CUDASIM:
numba/cuda/__init__.py:from numba.cuda.compiler import (compile, compile_for_current_device,
numba/cuda/__init__.py:# Are we the numba.cuda built in to upstream Numba, or the out-of-tree
numba/cuda/__init__.py:# NVIDIA-maintained target?
numba/cuda/__init__.py:        raise cuda_error()
numba/cuda/__init__.py:    return runtests.main("numba.cuda.tests", *args, **kwargs)
numba/cuda/cudaimpl.py:from .cudadrv import nvvm
numba/cuda/cudaimpl.py:from numba import cuda
numba/cuda/cudaimpl.py:from numba.cuda import nvvmutils, stubs, errors
numba/cuda/cudaimpl.py:from numba.cuda.types import dim3, CUDADispatcher
numba/cuda/cudaimpl.py:@lower_attr(types.Module(cuda), 'threadIdx')
numba/cuda/cudaimpl.py:def cuda_threadIdx(context, builder, sig, args):
numba/cuda/cudaimpl.py:@lower_attr(types.Module(cuda), 'blockDim')
numba/cuda/cudaimpl.py:def cuda_blockDim(context, builder, sig, args):
numba/cuda/cudaimpl.py:@lower_attr(types.Module(cuda), 'blockIdx')
numba/cuda/cudaimpl.py:def cuda_blockIdx(context, builder, sig, args):
numba/cuda/cudaimpl.py:@lower_attr(types.Module(cuda), 'gridDim')
numba/cuda/cudaimpl.py:def cuda_gridDim(context, builder, sig, args):
numba/cuda/cudaimpl.py:@lower_attr(types.Module(cuda), 'laneid')
numba/cuda/cudaimpl.py:def cuda_laneid(context, builder, sig, args):
numba/cuda/cudaimpl.py:@lower(cuda.const.array_like, types.Array)
numba/cuda/cudaimpl.py:def cuda_const_array_like(context, builder, sig, args):
numba/cuda/cudaimpl.py:    # This is a no-op because CUDATargetContext.make_constant_array already
numba/cuda/cudaimpl.py:@lower(cuda.shared.array, types.IntegerLiteral, types.Any)
numba/cuda/cudaimpl.py:def cuda_shared_array_integer(context, builder, sig, args):
numba/cuda/cudaimpl.py:                          symbol_name=_get_unique_smem_id('_cudapy_smem'),
numba/cuda/cudaimpl.py:@lower(cuda.shared.array, types.Tuple, types.Any)
numba/cuda/cudaimpl.py:@lower(cuda.shared.array, types.UniTuple, types.Any)
numba/cuda/cudaimpl.py:def cuda_shared_array_tuple(context, builder, sig, args):
numba/cuda/cudaimpl.py:                          symbol_name=_get_unique_smem_id('_cudapy_smem'),
numba/cuda/cudaimpl.py:@lower(cuda.local.array, types.IntegerLiteral, types.Any)
numba/cuda/cudaimpl.py:def cuda_local_array_integer(context, builder, sig, args):
numba/cuda/cudaimpl.py:                          symbol_name='_cudapy_lmem',
numba/cuda/cudaimpl.py:@lower(cuda.local.array, types.Tuple, types.Any)
numba/cuda/cudaimpl.py:@lower(cuda.local.array, types.UniTuple, types.Any)
numba/cuda/cudaimpl.py:                          symbol_name='_cudapy_lmem',
numba/cuda/cudaimpl.py:    The NVVM intrinsic for shfl only supports i32, but the cuda intrinsic
numba/cuda/cudaimpl.py:        raise errors.CudaLoweringError(msg)
numba/cuda/cudaimpl.py:        raise errors.CudaLoweringError(msg)
numba/cuda/cudaimpl.py:        return cuda.fp16.hdiv(x, y)
numba/cuda/cudaimpl.py:# https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cbrt.html#__nv_cbrt
numba/cuda/cudaimpl.py:# https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cbrtf.html#__nv_cbrtf
numba/cuda/cudaimpl.py:    if dtype in cuda.cudadecl.unsigned_int_numba_types:
numba/cuda/cudaimpl.py:    if dtype in cuda.cudadecl.unsigned_int_numba_types:
numba/cuda/cudaimpl.py:        if dtype in (cuda.cudadecl.integer_numba_types):
numba/cuda/cudaimpl.py:    if dtype in (cuda.cudadecl.integer_numba_types):
numba/cuda/cudaimpl.py:    if aryty.dtype in (cuda.cudadecl.integer_numba_types):
numba/cuda/cudaimpl.py:@lower_constant(CUDADispatcher)
numba/cuda/cudaimpl.py:def cuda_dispatcher_const(context, builder, ty, pyval):
numba/cuda/compiler.py:from numba.cuda.api import get_current_device
numba/cuda/compiler.py:from numba.cuda.target import CUDACABICallConv
numba/cuda/compiler.py:class CUDAFlags(Flags):
numba/cuda/compiler.py:# The CUDACompileResult (CCR) has a specially-defined entry point equal to its
numba/cuda/compiler.py:# unique property of a CompileResult in the CUDA target (cf. the CPU target,
numba/cuda/compiler.py:class CUDACompileResult(CompileResult):
numba/cuda/compiler.py:def cuda_compile_result(**entries):
numba/cuda/compiler.py:    return CUDACompileResult(**entries)
numba/cuda/compiler.py:class CUDABackend(LoweringPass):
numba/cuda/compiler.py:    _name = "cuda_backend"
numba/cuda/compiler.py:        state.cr = cuda_compile_result(
numba/cuda/compiler.py:    Create a CUDACodeLibrary for the NativeLowering pass to populate. The
numba/cuda/compiler.py:class CUDACompiler(CompilerBase):
numba/cuda/compiler.py:        pm = PassManager('cuda')
numba/cuda/compiler.py:        lowering_passes = self.define_cuda_lowering_pipeline(self.state)
numba/cuda/compiler.py:    def define_cuda_lowering_pipeline(self, state):
numba/cuda/compiler.py:        pm = PassManager('cuda_lowering')
numba/cuda/compiler.py:        pm.add_pass(CUDABackend, "cuda backend")
numba/cuda/compiler.py:def compile_cuda(pyfunc, return_type, args, debug=False, lineinfo=False,
numba/cuda/compiler.py:    from .descriptor import cuda_target
numba/cuda/compiler.py:    typingctx = cuda_target.typing_context
numba/cuda/compiler.py:    targetctx = cuda_target.target_context
numba/cuda/compiler.py:    flags = CUDAFlags()
numba/cuda/compiler.py:    with target_override('cuda'):
numba/cuda/compiler.py:                                      pipeline_class=CUDACompiler)
numba/cuda/compiler.py:    c_call_conv = CUDACABICallConv(context)
numba/cuda/compiler.py:    wrapper_module = context.create_module("cuda.cabi.wrapper")
numba/cuda/compiler.py:               "is not supported by CUDA. This may result in a crash"
numba/cuda/compiler.py:    cc = cc or config.CUDA_DEFAULT_PTX_CC
numba/cuda/compiler.py:    cres = compile_cuda(pyfunc, return_type, args, debug=debug,
numba/cuda/compiler.py:        raise TypeError("CUDA kernel must have void return type.")
numba/cuda/compiler.py:        lib, kernel = tgt.prepare_cuda_kernel(cres.library, cres.fndesc, debug,
numba/cuda/compiler.py:    from .descriptor import cuda_target
numba/cuda/compiler.py:    typingctx = cuda_target.typing_context
numba/cuda/compiler.py:    targetctx = cuda_target.target_context
towncrier.toml:    directory = "cuda"
towncrier.toml:    name = "CUDA API Changes"
buildscripts/condarecipe.local/meta.yaml:    # CUDA 11.2 or later is required for CUDA support
buildscripts/condarecipe.local/meta.yaml:    - cuda-version >=11.2
buildscripts/condarecipe.local/meta.yaml:    - cudatoolkit >=11.2
buildscripts/condarecipe.local/meta.yaml:    # CUDA Python 11.6 or later
buildscripts/condarecipe.local/meta.yaml:    - cuda-python >=11.6
buildscripts/incremental/test.sh:NUMBA_ENABLE_CUDASIM=1 $SEGVCATCH python -m numba.runtests -b -v -g -m $TEST_NPROCS -- numba.tests
buildscripts/incremental/test.sh:    NUMBA_USE_TYPEGUARD=1 NUMBA_ENABLE_CUDASIM=1 PYTHONWARNINGS="ignore:::typeguard" $SEGVCATCH python runtests.py -b -j "$TEST_START_INDEX:$TEST_COUNT" --exclude-tags='long_running' -m $TEST_NPROCS -- numba.tests
buildscripts/incremental/test.sh:    NUMBA_ENABLE_CUDASIM=1 $SEGVCATCH python -m numba.runtests -b -j "$TEST_START_INDEX:$TEST_COUNT" --exclude-tags='long_running' -m $TEST_NPROCS -- numba.tests
buildscripts/gpuci/axis.yaml:CUDA_VER:
buildscripts/gpuci/axis.yaml:CUDA_TOOLKIT_VER:
buildscripts/gpuci/build.sh:# Numba GPU build and test script for CI     #
buildscripts/gpuci/build.sh:export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
buildscripts/gpuci/build.sh:# Determine CUDA release version
buildscripts/gpuci/build.sh:export CUDA_REL=${CUDA_VERSION%.*}
buildscripts/gpuci/build.sh:# Test with NVIDIA Bindings on CUDA 11.5
buildscripts/gpuci/build.sh:if [ $CUDA_TOOLKIT_VER == "11.5" ]
buildscripts/gpuci/build.sh:  export NUMBA_CUDA_USE_NVIDIA_BINDING=1;
buildscripts/gpuci/build.sh:  export NUMBA_CUDA_USE_NVIDIA_BINDING=0;
buildscripts/gpuci/build.sh:# Test with Minor Version Compatibility on CUDA 11.8
buildscripts/gpuci/build.sh:if [ $CUDA_TOOLKIT_VER == "11.8" ]
buildscripts/gpuci/build.sh:  export NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1;
buildscripts/gpuci/build.sh:  export NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=0;
buildscripts/gpuci/build.sh:# with different CUDA, NumPy, and Python versions).
buildscripts/gpuci/build.sh:NUMPY_VER="${CTK_NUMPY_VMAP[$CUDA_TOOLKIT_VER]}"
buildscripts/gpuci/build.sh:PYTHON_VER="${CTK_PYTHON_VMAP[$CUDA_TOOLKIT_VER]}"
buildscripts/gpuci/build.sh:gpuci_logger "Check environment variables"
buildscripts/gpuci/build.sh:gpuci_logger "Check GPU usage"
buildscripts/gpuci/build.sh:nvidia-smi
buildscripts/gpuci/build.sh:gpuci_logger "Create testing env"
buildscripts/gpuci/build.sh:gpuci_mamba_retry create -n numba_ci -y \
buildscripts/gpuci/build.sh:                  "cudatoolkit=${CUDA_TOOLKIT_VER}" \
buildscripts/gpuci/build.sh:if [ $NUMBA_CUDA_USE_NVIDIA_BINDING == "1" ]
buildscripts/gpuci/build.sh:  gpuci_logger "Install NVIDIA CUDA Python bindings";
buildscripts/gpuci/build.sh:  gpuci_mamba_retry install cuda-python=11.8 cuda-cudart=11.5 cuda-nvrtc=11.5;
buildscripts/gpuci/build.sh:gpuci_logger "Install numba"
buildscripts/gpuci/build.sh:gpuci_logger "Check Compiler versions"
buildscripts/gpuci/build.sh:gpuci_logger "Check conda environment"
buildscripts/gpuci/build.sh:gpuci_logger "Dump system information from Numba"
buildscripts/gpuci/build.sh:gpuci_logger "Run tests in numba.cuda.tests"
buildscripts/gpuci/build.sh:python -m numba.runtests numba.cuda.tests -v -m
buildscripts/azure/azure-windows.yml:        set NUMBA_ENABLE_CUDASIM=1
buildscripts/azure/azure-windows.yml:        set NUMBA_ENABLE_CUDASIM=1
buildscripts/azure/azure-windows.yml:        set NUMBA_ENABLE_CUDASIM=1
docs/source/cuda-reference/types.rst:CUDA-Specific Types
docs/source/cuda-reference/types.rst:    This page is about types specific to CUDA targets. Many other types are also
docs/source/cuda-reference/types.rst:    available in the CUDA target - see :ref:`cuda-built-in-types`.
docs/source/cuda-reference/types.rst:`CUDA Vector Types <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types>`_
docs/source/cuda-reference/types.rst:are usable in kernels. There are two important distinctions from vector types in CUDA C/C++:
docs/source/cuda-reference/types.rst:First, the recommended names for vector types in Numba CUDA is formatted as ``<base_type>x<N>``,
docs/source/cuda-reference/types.rst:Examples include ``int64x3``, ``uint16x4``, ``float32x4``, etc. For new Numba CUDA kernels,
docs/source/cuda-reference/types.rst:For convenience, users adapting existing kernels from CUDA C/C++ to Python may use
docs/source/cuda-reference/types.rst:Second, unlike CUDA C/C++ where factory functions are used, vector types are constructed directly
docs/source/cuda-reference/types.rst:    from numba.cuda import float32x3
docs/source/cuda-reference/host.rst:CUDA Host API
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.is_available
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.detect
docs/source/cuda-reference/host.rst:CUDA Python functions execute within a CUDA context. Each CUDA device in a
docs/source/cuda-reference/host.rst:system has an associated CUDA context, and Numba presently allows only one context
docs/source/cuda-reference/host.rst:per thread. For further details on CUDA Contexts, refer to the `CUDA Driver API
docs/source/cuda-reference/host.rst:<http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html>`_ and the
docs/source/cuda-reference/host.rst:`CUDA C Programming Guide Context Documentation
docs/source/cuda-reference/host.rst:<http://docs.nvidia.com/cuda/cuda-c-programming-guide/#context>`_. CUDA Contexts
docs/source/cuda-reference/host.rst:are instances of the :class:`~numba.cuda.cudadrv.driver.Context` class:
docs/source/cuda-reference/host.rst:.. autoclass:: numba.cuda.cudadrv.driver.Context
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.current_context
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.require_context
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.synchronize
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.close
docs/source/cuda-reference/host.rst:Numba maintains a list of supported CUDA-capable devices:
docs/source/cuda-reference/host.rst:.. attribute:: numba.cuda.gpus
docs/source/cuda-reference/host.rst:   An indexable list of supported CUDA devices. This list is indexed by integer
docs/source/cuda-reference/host.rst:.. attribute:: numba.cuda.gpus.current
docs/source/cuda-reference/host.rst:Getting a device through :attr:`numba.cuda.gpus` always provides an instance of
docs/source/cuda-reference/host.rst::class:`numba.cuda.cudadrv.devices._DeviceContextManager`, which acts as a
docs/source/cuda-reference/host.rst:.. autoclass:: numba.cuda.cudadrv.devices._DeviceContextManager
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.select_device
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.get_current_device
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.list_devices
docs/source/cuda-reference/host.rst:The :class:`numba.cuda.cudadrv.driver.Device` class can be used to enquire about
docs/source/cuda-reference/host.rst:.. class:: numba.cuda.cudadrv.driver.Device
docs/source/cuda-reference/host.rst:      The UUID of the device (e.g. "GPU-e6489c45-5b68-3b03-bab7-0e7c8e809643").
docs/source/cuda-reference/host.rst:- Generating code prior to a fork without initializing CUDA.
docs/source/cuda-reference/host.rst:   :ref:`cuda-using-the-c-abi`.
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.compile
docs/source/cuda-reference/host.rst:The environment variable ``NUMBA_CUDA_DEFAULT_PTX_CC`` can be set to control
docs/source/cuda-reference/host.rst::ref:`numba-envvars-gpu-support`. If code for the compute capability of the
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.compile_for_current_device
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.compile_ptx
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.compile_ptx_for_current_device
docs/source/cuda-reference/host.rst:.. _cuda-profiling:
docs/source/cuda-reference/host.rst:The NVidia Visual Profiler can be used directly on executing CUDA Python code -
docs/source/cuda-reference/host.rst:profiling, see the `NVidia Profiler User's Guide
docs/source/cuda-reference/host.rst:<https://docs.nvidia.com/cuda/profiler-users-guide/>`_.
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.profile_start
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.profile_stop
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.profiling
docs/source/cuda-reference/host.rst:further information, see the `CUDA C Programming Guide Events section
docs/source/cuda-reference/host.rst:<http://docs.nvidia.com/cuda/cuda-c-programming-guide/#events>`_.
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.event
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.event_elapsed_time
docs/source/cuda-reference/host.rst:Events are instances of the :class:`numba.cuda.cudadrv.driver.Event` class:
docs/source/cuda-reference/host.rst:.. autoclass:: numba.cuda.cudadrv.driver.Event
docs/source/cuda-reference/host.rst:CUDA device can be performed asynchronously using streams, including data
docs/source/cuda-reference/host.rst:transfers and kernel execution. For further details on streams, see the `CUDA C
docs/source/cuda-reference/host.rst:<http://docs.nvidia.com/cuda/cuda-c-programming-guide/#streams>`_.
docs/source/cuda-reference/host.rst:environment variable ``NUMBA_CUDA_PER_THREAD_DEFAULT_STREAM`` to ``1`` (see the
docs/source/cuda-reference/host.rst::ref:`CUDA Environment Variables section <numba-envvars-gpu-support>`).
docs/source/cuda-reference/host.rst:Streams are instances of :class:`numba.cuda.cudadrv.driver.Stream`:
docs/source/cuda-reference/host.rst:.. autoclass:: numba.cuda.cudadrv.driver.Stream
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.stream
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.default_stream
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.legacy_default_stream
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.per_thread_default_stream
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.external_stream
docs/source/cuda-reference/host.rst:accessed through ``cuda.runtime``, which is an instance of the
docs/source/cuda-reference/host.rst::class:`numba.cuda.cudadrv.runtime.Runtime` class:
docs/source/cuda-reference/host.rst:.. autoclass:: numba.cuda.cudadrv.runtime.Runtime
docs/source/cuda-reference/host.rst:.. autofunction:: numba.cuda.is_supported_version
docs/source/cuda-reference/memory.rst:.. autofunction:: numba.cuda.to_device
docs/source/cuda-reference/memory.rst:.. autofunction:: numba.cuda.device_array
docs/source/cuda-reference/memory.rst:.. autofunction:: numba.cuda.device_array_like
docs/source/cuda-reference/memory.rst:.. autofunction:: numba.cuda.pinned_array
docs/source/cuda-reference/memory.rst:.. autofunction:: numba.cuda.pinned_array_like
docs/source/cuda-reference/memory.rst:.. autofunction:: numba.cuda.mapped_array
docs/source/cuda-reference/memory.rst:.. autofunction:: numba.cuda.mapped_array_like
docs/source/cuda-reference/memory.rst:.. autofunction:: numba.cuda.managed_array
docs/source/cuda-reference/memory.rst:.. autofunction:: numba.cuda.pinned
docs/source/cuda-reference/memory.rst:.. autofunction:: numba.cuda.mapped
docs/source/cuda-reference/memory.rst:.. autoclass:: numba.cuda.cudadrv.devicearray.DeviceNDArray
docs/source/cuda-reference/memory.rst:.. autoclass:: numba.cuda.cudadrv.devicearray.DeviceRecord
docs/source/cuda-reference/memory.rst:.. autoclass:: numba.cuda.cudadrv.devicearray.MappedNDArray
docs/source/cuda-reference/index.rst:CUDA Python Reference
docs/source/cuda-reference/kernel.rst:CUDA Kernel API
docs/source/cuda-reference/kernel.rst:The ``@cuda.jit`` decorator is used to create a CUDA dispatcher object that can
docs/source/cuda-reference/kernel.rst:.. autofunction:: numba.cuda.jit
docs/source/cuda-reference/kernel.rst:   # func is some function decorated with @cuda.jit
docs/source/cuda-reference/kernel.rst:This is similar to launch configuration in CUDA C/C++:
docs/source/cuda-reference/kernel.rst:.. code-block:: cuda
docs/source/cuda-reference/kernel.rst:   compared to in CUDA C/C++.
docs/source/cuda-reference/kernel.rst:.. autoclass:: numba.cuda.dispatcher.CUDADispatcher
docs/source/cuda-reference/kernel.rst:from within a CUDA Kernel.
docs/source/cuda-reference/kernel.rst:.. attribute:: numba.cuda.threadIdx
docs/source/cuda-reference/kernel.rst:    :attr:`numba.cuda.blockDim` exclusive.
docs/source/cuda-reference/kernel.rst:.. attribute:: numba.cuda.blockIdx
docs/source/cuda-reference/kernel.rst:    :attr:`numba.cuda.gridDim` exclusive.
docs/source/cuda-reference/kernel.rst:.. attribute:: numba.cuda.blockDim
docs/source/cuda-reference/kernel.rst:.. attribute:: numba.cuda.gridDim
docs/source/cuda-reference/kernel.rst:.. attribute:: numba.cuda.laneid
docs/source/cuda-reference/kernel.rst:    from 0 inclusive to the :attr:`numba.cuda.warpsize` exclusive.
docs/source/cuda-reference/kernel.rst:.. attribute:: numba.cuda.warpsize
docs/source/cuda-reference/kernel.rst:    The size in threads of a warp on the GPU. Currently this is always 32.
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.grid(ndim)
docs/source/cuda-reference/kernel.rst:      cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.gridsize(ndim)
docs/source/cuda-reference/kernel.rst:       cuda.blockDim.x * cuda.gridDim.x
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.shared.array(shape, dtype)
docs/source/cuda-reference/kernel.rst:   Creates an array in the local memory space of the CUDA kernel with
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.local.array(shape, dtype)
docs/source/cuda-reference/kernel.rst:   Creates an array in the local memory space of the CUDA kernel with the
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.const.array_like(ary)
docs/source/cuda-reference/kernel.rst:   Copies the ``ary`` into constant memory space on the CUDA kernel at compile
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.atomic.add(array, idx, value)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.atomic.sub(array, idx, value)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.atomic.and_(array, idx, value)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.atomic.or_(array, idx, value)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.atomic.xor(array, idx, value)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.atomic.exch(array, idx, value)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.atomic.inc(array, idx, value)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.atomic.dec(array, idx, value)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.atomic.max(array, idx, value)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.atomic.cas(array, idx, old, value)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.syncthreads
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.syncthreads_count(predicate)
docs/source/cuda-reference/kernel.rst:    An extension to :attr:`numba.cuda.syncthreads` where the return value is a count
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.syncthreads_and(predicate)
docs/source/cuda-reference/kernel.rst:    An extension to :attr:`numba.cuda.syncthreads` where 1 is returned if ``predicate`` is
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.syncthreads_or(predicate)
docs/source/cuda-reference/kernel.rst:    An extension to :attr:`numba.cuda.syncthreads` where 1 is returned if ``predicate`` is
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.cg.this_grid()
docs/source/cuda-reference/kernel.rst:   :rtype: numba.cuda.cg.GridGroup
docs/source/cuda-reference/kernel.rst:.. class:: numba.cuda.cg.GridGroup
docs/source/cuda-reference/kernel.rst:   the current grid group using :func:`cg.this_grid() <numba.cuda.cg.this_grid>`.
docs/source/cuda-reference/kernel.rst:are visible by other threads within the same thread-block, the same GPU device,
docs/source/cuda-reference/kernel.rst:and the same system (across GPUs on global memory). Memory loads and stores
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.threadfence
docs/source/cuda-reference/kernel.rst:   A memory fence at device level (within the GPU).
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.threadfence_block
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.threadfence_system
docs/source/cuda-reference/kernel.rst:   A memory fence at system level (across GPUs).
docs/source/cuda-reference/kernel.rst:the GPU compute capability is below 7.x.
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.syncwarp(membermask)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.all_sync(membermask, predicate)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.any_sync(membermask, predicate)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.eq_sync(membermask, predicate)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.ballot_sync(membermask, predicate)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.shfl_sync(membermask, value, src_lane)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.shfl_up_sync(membermask, value, delta)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.shfl_down_sync(membermask, value, delta)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.shfl_xor_sync(membermask, value, lane_mask)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.match_any_sync(membermask, value, lane_mask)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.match_all_sync(membermask, value, lane_mask)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.activemask()
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.lanemask_lt()
docs/source/cuda-reference/kernel.rst:A subset of the CUDA Math API's integer intrinsics are available. For further
docs/source/cuda-reference/kernel.rst:documentation, including semantics, please refer to the `CUDA Toolkit
docs/source/cuda-reference/kernel.rst:<https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html>`_.
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.popc(x)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.brev(x)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.clz(x)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.ffs(x)
docs/source/cuda-reference/kernel.rst:A subset of the CUDA Math API's floating point intrinsics are available. For further
docs/source/cuda-reference/kernel.rst:<https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html>`_ and
docs/source/cuda-reference/kernel.rst:`double <https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html>`_
docs/source/cuda-reference/kernel.rst:precision parts of the CUDA Toolkit documentation.
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fma
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.cbrt (x)
docs/source/cuda-reference/kernel.rst:The functions in the ``cuda.fp16`` module are used to operate on 16-bit
docs/source/cuda-reference/kernel.rst:   .. function:: numba.cuda.is_float16_supported ()
docs/source/cuda-reference/kernel.rst::attr:`supports_float16 <numba.cuda.cudadrv.driver.Device.supports_float16>`
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hfma (a, b, c)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hadd (a, b)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hsub (a, b)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hmul (a, b)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hdiv (a, b)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hneg (a)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.habs (a)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hsin (a)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hcos (a)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hlog (a)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hlog10 (a)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hlog2 (a)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hexp (a)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hexp10 (a)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hexp2 (a)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hfloor (a)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hceil (a)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hsqrt (a)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hrsqrt (a)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hrcp (a)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hrint (a)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.htrunc (a)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.heq (a, b)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hne (a, b)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hgt (a, b)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hge (a, b)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hlt (a, b)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hle (a, b)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hmax (a, b)
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.fp16.hmin (a, b)
docs/source/cuda-reference/kernel.rst:A subset of the CUDA's control flow instructions are directly available as
docs/source/cuda-reference/kernel.rst:intrinsics. Avoiding branches is a key way to improve CUDA performance, and
docs/source/cuda-reference/kernel.rst:semantics, please refer to the `relevant CUDA Toolkit documentation
docs/source/cuda-reference/kernel.rst:<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions>`_.
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.selp
docs/source/cuda-reference/kernel.rst:.. function:: numba.cuda.nanosleep(ns)
docs/source/cuda-reference/libdevice.rst:CUDA Python by other means, e.g. ``math.nan``.
docs/source/cuda-reference/libdevice.rst:.. automodule:: numba.cuda.libdevice
docs/source/user/jitclass.rst:  (Note: Support for GPU devices is planned for a future release.)
docs/source/user/vectorize.rst:cuda                    CUDA GPU
docs/source/user/vectorize.rst:			  See `documentation for CUDA ufunc <../cuda/ufunc.html>`_ for detail.
docs/source/user/vectorize.rst:The "cuda" target works well for big data sizes (approx. greater than 1MB) and
docs/source/user/vectorize.rst:high compute intensity algorithms.  Transferring memory to and from the GPU adds
docs/source/user/faq.rst:can also target parallel execution on GPU architectures using its CUDA and HSA
docs/source/user/faq.rst:GPU Programming
docs/source/user/faq.rst:How do I work around the ``CUDA initialized before forking`` error?
docs/source/user/faq.rst:processes, CUDA will not work correctly in the child process if the CUDA
docs/source/user/faq.rst:``CudaDriverError`` with the message ``CUDA initialized before forking``.
docs/source/user/faq.rst:One approach to avoid this error is to make all calls to ``numba.cuda``
docs/source/user/faq.rst:available GPUs before starting the process pool.  In Python 3, you can change
docs/source/user/faq.rst:Switching from ``fork`` to ``spawn`` or ``forkserver`` will avoid the CUDA
docs/source/user/installing.rst:* NVIDIA GPUs of compute capability 5.0 and later
docs/source/user/installing.rst:* ARMv8 (64-bit little-endian, such as the NVIDIA Jetson)
docs/source/user/installing.rst:To enable CUDA GPU support for Numba, install the latest `graphics drivers from
docs/source/user/installing.rst:NVIDIA <https://www.nvidia.com/Download/index.aspx>`_ for your platform.
docs/source/user/installing.rst:distributions do not support CUDA.)  Then install the CUDA Toolkit package.
docs/source/user/installing.rst:For CUDA 12, ``cuda-nvcc`` and ``cuda-nvrtc`` are required::
docs/source/user/installing.rst:    $ conda install -c conda-forge cuda-nvcc cuda-nvrtc "cuda-version>=12.0"
docs/source/user/installing.rst:For CUDA 11, ``cudatoolkit`` is required::
docs/source/user/installing.rst:    $ conda install -c conda-forge cudatoolkit "cuda-version>=11.2,<12.0"
docs/source/user/installing.rst:You do not need to install the CUDA SDK from NVIDIA.
docs/source/user/installing.rst:To use CUDA with Numba installed by ``pip``, you need to install the `CUDA SDK
docs/source/user/installing.rst:<https://developer.nvidia.com/cuda-downloads>`_ from NVIDIA.  Please refer to
docs/source/user/installing.rst::ref:`cudatoolkit-lookup` for details. Numba can also detect CUDA libraries
docs/source/user/installing.rst:We build and test conda packages on the `NVIDIA Jetson TX2
docs/source/user/installing.rst:<https://www.nvidia.com/en-us/autonomous-machines/embedded-systems-dev-kits-modules/>`_,
docs/source/user/installing.rst:On CUDA-enabled systems, like the Jetson, the CUDA toolkit should be
docs/source/user/installing.rst:  * ``cuda-python`` - The NVIDIA CUDA Python bindings. See :ref:`cuda-bindings`.
docs/source/user/installing.rst:    __CUDA Information__
docs/source/user/installing.rst:    Found 1 CUDA devices
docs/source/user/installing.rst:                                        UUID: GPU-e6489c45-5b68-3b03-bab7-0e7c8e809643
docs/source/user/5minguide.rst:* GPUs: Nvidia CUDA.
docs/source/user/5minguide.rst:GPU targets:
docs/source/user/5minguide.rst:Numba can target `Nvidia CUDA <https://developer.nvidia.com/cuda-zone>`_ GPUs.
docs/source/user/5minguide.rst::ref:`CUDA <cuda-index>`.
docs/source/user/talks.rst:* GPU Technology Conference 2016 - Accelerating a Spectral Algorithm for Plasma Physics with Python/Numba on GPU - Manuel Kirchen & Rémi Lehe (`Slides <http://on-demand.gputechconf.com/gtc/2016/presentation/s6353-manuel-kirchen-spectral-algorithm-plasma-physics.pdf>`__)
docs/source/user/talks.rst:* GPU Technology Conference 2018 - GPU Computing in Python with Numba - Stan Seibert (`Notebooks <https://github.com/ContinuumIO/gtc2018-numba>`__)
docs/source/user/talks.rst:* PyData Amsterdam 2019 - Create CUDA kernels from Python using Numba and CuPy - Valentin Haenel (`Video <https://www.youtube.com/watch?v=CQDsT81GyS8>`__)
docs/source/user/cli.rst:    __CUDA Information__
docs/source/user/cli.rst:    CUDA Device Initialized                       : False
docs/source/user/cli.rst:    CUDA Driver Version                           : ?
docs/source/user/cli.rst:    CUDA Runtime Version                          : ?
docs/source/user/cli.rst:    CUDA NVIDIA Bindings Available                : ?
docs/source/user/cli.rst:    CUDA NVIDIA Bindings In Use                   : ?
docs/source/user/cli.rst:    CUDA Detect Output:
docs/source/user/cli.rst:    CUDA Libraries Test Output:
docs/source/user/troubleshoot.rst:.. _debugging-cuda-python-code:
docs/source/user/troubleshoot.rst:Debugging CUDA Python code
docs/source/user/troubleshoot.rst:CUDA Python code can be run in the Python interpreter using the CUDA Simulator,
docs/source/user/troubleshoot.rst:enable the CUDA simulator, set the environment variable
docs/source/user/troubleshoot.rst::envvar:`NUMBA_ENABLE_CUDASIM` to 1. For more information on the CUDA Simulator,
docs/source/user/troubleshoot.rst:see :ref:`the CUDA Simulator documentation <simulator>`.
docs/source/user/troubleshoot.rst:By setting the ``debug`` argument to ``cuda.jit`` to ``True``
docs/source/user/troubleshoot.rst:(``@cuda.jit(debug=True)``), Numba will emit source location in the compiled
docs/source/user/troubleshoot.rst:CUDA code.  Unlike the CPU target, only filename and line information are
docs/source/user/troubleshoot.rst:`cuda-memcheck <http://docs.nvidia.com/cuda/cuda-memcheck/index.html>`_.
docs/source/user/troubleshoot.rst:For example, given the following cuda python code:
docs/source/user/troubleshoot.rst:  from numba import cuda
docs/source/user/troubleshoot.rst:  @cuda.jit(debug=True)
docs/source/user/troubleshoot.rst:      arr[cuda.threadIdx.x] = 1
docs/source/user/troubleshoot.rst:We can use ``cuda-memcheck`` to find the memory error:
docs/source/user/troubleshoot.rst:  $ cuda-memcheck python chk_cuda_debug.py
docs/source/user/troubleshoot.rst:  ========= CUDA-MEMCHECK
docs/source/user/troubleshoot.rst:  =========     at 0x00000148 in /home/user/chk_cuda_debug.py:6:cudapy::__main__::foo$241(Array<__int64, int=1, C, mutable, aligned>)
docs/source/user/troubleshoot.rst:  =========     at 0x00000148 in /home/user/chk_cuda_debug.py:6:cudapy::__main__::foo$241(Array<__int64, int=1, C, mutable, aligned>)
docs/source/user/overview.rst:  :doc:`GPU hardware <../cuda/index>`
docs/source/proposals/jit-classes.rst:GPUs (e.g. CUDA and HSA) targets are supported via an immutable version of the
docs/source/proposals/extension-points.rst:   This document doesn't cover CUDA or any other non-CPU backend.
docs/source/proposals/external-memory-management.rst:NBEP 7: CUDA External Memory Management Plugins
docs/source/proposals/external-memory-management.rst::Author: Graham Markall, NVIDIA
docs/source/proposals/external-memory-management.rst:The :ref:`CUDA Array Interface <cuda-array-interface>` enables sharing of data
docs/source/proposals/external-memory-management.rst:between different Python libraries that access CUDA devices. However, each
docs/source/proposals/external-memory-management.rst:However, not all CUDA memory management libraries also support managing host
docs/source/proposals/external-memory-management.rst:the :func:`~numba.cuda.defer_cleanup` context manager.
docs/source/proposals/external-memory-management.rst:compiled object, which is generated from ``@cuda.jit``\ -ted functions). The
docs/source/proposals/external-memory-management.rst:free to take a CUDA stream and execute asynchronously. For freeing, this is
docs/source/proposals/external-memory-management.rst:* Any changes to the ``__cuda_array_interface__`` to further define its semantics,
docs/source/proposals/external-memory-management.rst:New classes and functions will be added to ``numba.cuda.cudadrv.driver``:
docs/source/proposals/external-memory-management.rst:* ``BaseCUDAMemoryManager`` and ``HostOnlyCUDAMemoryManager``\ : base classes for
docs/source/proposals/external-memory-management.rst:These will be exposed through the public API, in the ``numba.cuda`` module.
docs/source/proposals/external-memory-management.rst:   export NUMBA_CUDA_MEMORY_MANAGER="<module>"
docs/source/proposals/external-memory-management.rst:An EMM plugin is implemented by inheriting from the ``BaseCUDAMemoryManager``
docs/source/proposals/external-memory-management.rst:   class BaseCUDAMemoryManager(object, metaclass=ABCMeta):
docs/source/proposals/external-memory-management.rst:           - `mapped`: Whether the allocated memory should be mapped into the CUDA
docs/source/proposals/external-memory-management.rst:           Return an `IpcHandle` from a GPU allocation. Arguments:
docs/source/proposals/external-memory-management.rst:``defer_cleanup`` is called when the ``numba.cuda.defer_cleanup`` context manager
docs/source/proposals/external-memory-management.rst:Memory mapped into the CUDA address space (which is created when the
docs/source/proposals/external-memory-management.rst:   class HostOnlyCUDAMemoryManager(BaseCUDAMemoryManager):
docs/source/proposals/external-memory-management.rst:A class can subclass the ``HostOnlyCUDAMemoryManager`` and then it only needs to
docs/source/proposals/external-memory-management.rst:  ``HostOnlyCUDAMemoryManager.__init__``\ , as this is used to initialize some of
docs/source/proposals/external-memory-management.rst:  used by the ``HostOnlyCUDAMemoryManager``.
docs/source/proposals/external-memory-management.rst:    methods from ``HostOnlyCUDAMemoryManager`` in its own implementations.
docs/source/proposals/external-memory-management.rst:  provided by ``HostOnlyCUDAManager.defer_cleanup()`` prior to ``yield``\ ing (or in
docs/source/proposals/external-memory-management.rst:   from numba import cuda
docs/source/proposals/external-memory-management.rst:   from numba import cuda
docs/source/proposals/external-memory-management.rst:are equivalent - this is because Numba does not initialize CUDA or allocate any
docs/source/proposals/external-memory-management.rst:memory until the first call to a CUDA function - neither instantiating and
docs/source/proposals/external-memory-management.rst:registering an EMM plugin, nor importing ``numba.cuda`` causes a call to a CUDA
docs/source/proposals/external-memory-management.rst:       numba.cuda.cudadrv.driver.set_memory_manager(MyNumbaMemoryManager)
docs/source/proposals/external-memory-management.rst:   from numba.cuda import (HostOnlyCUDAMemoryManager, MemoryPointer, IpcHandle,
docs/source/proposals/external-memory-management.rst:   class RMMNumbaManager(HostOnlyCUDAMemoryManager):
docs/source/proposals/external-memory-management.rst:           ctx = cuda.current_context()
docs/source/proposals/external-memory-management.rst:           cuda.cudadrv.memory.driver_funcs.cuIpcGetMemHandle(
docs/source/proposals/external-memory-management.rst:           source_info = cuda.current_context().device.get_device_identity()
docs/source/proposals/external-memory-management.rst:   # To support `NUMBA_CUDA_MEMORY_MANAGER=rmm`:
docs/source/proposals/external-memory-management.rst:   from numba import cuda
docs/source/proposals/external-memory-management.rst:   d_a = cuda.to_device(a)
docs/source/proposals/external-memory-management.rst:   Alloc,0,0x7fae06600000,0,80,0,0,1,1.10549,1.1074,0.00191666,<path>/numba/numba/cuda/cudadrv/driver.py:683
docs/source/proposals/external-memory-management.rst:   NUMBA_CUDA_MEMORY_MANAGER="rmm.RMMNumbaManager" python example.py
docs/source/proposals/external-memory-management.rst::class:`~numba.cuda.cudadrv.driver.Context` class. It maintains lists of
docs/source/proposals/external-memory-management.rst:The ``numba.cuda.cudadrv.driver.Context`` class will no longer directly allocate
docs/source/proposals/external-memory-management.rst:               raise CudaDriverError("%s cannot map host memory" % self.device)
docs/source/proposals/external-memory-management.rst:* ``BaseCUDAMemoryManager``\ : An abstract class, as defined in the plugin interface
docs/source/proposals/external-memory-management.rst:* ``HostOnlyCUDAMemoryManager``\ : A subclass of ``BaseCUDAMemoryManager``\ , with the
docs/source/proposals/external-memory-management.rst:* ``NumbaCUDAMemoryManager``\ : A subclass of ``HostOnlyCUDAMemoryManager``\ , which
docs/source/proposals/external-memory-management.rst:    parent class ``HostOnlyCUDAMemoryManager``\ , and it uses these for the
docs/source/proposals/external-memory-management.rst:  manager class. This global initially holds ``NumbaCUDAMemoryManager`` (the
docs/source/proposals/external-memory-management.rst:tested to have no effect on the CUDA test suite:
docs/source/proposals/external-memory-management.rst:   diff --git a/numba/cuda/cudadrv/driver.py b/numba/cuda/cudadrv/driver.py
docs/source/proposals/external-memory-management.rst:   --- a/numba/cuda/cudadrv/driver.py
docs/source/proposals/external-memory-management.rst:   +++ b/numba/cuda/cudadrv/driver.py
docs/source/proposals/external-memory-management.rst:            with cuda.gpus[srcdev.id]:
docs/source/proposals/external-memory-management.rst:   from numba import cuda
docs/source/proposals/external-memory-management.rst:   d_a = cuda.to_device(a)
docs/source/proposals/external-memory-management.rst:   Alloc,0,0x7f96c7400000,0,80,0,0,1,1.13396,1.13576,0.00180059,<path>/numba/numba/cuda/cudadrv/driver.py:686
docs/source/proposals/external-memory-management.rst:   from numba import cuda
docs/source/proposals/external-memory-management.rst:   d_a = cuda.to_device(a)
docs/source/proposals/external-memory-management.rst:Numba CUDA Unit tests
docs/source/proposals/external-memory-management.rst:CUDA unit tests also pass with the prototype branch, for both the internal memory
docs/source/proposals/external-memory-management.rst:   NUMBA_CUDA_MEMORY_MANAGER=rmm python -m numba.runtests numba.cuda.tests
docs/source/proposals/external-memory-management.rst:* ``TestCudaArrayInterface.test_ownership``\ : skipped as Numba does not own memory
docs/source/proposals/external-memory-management.rst:   NUMBA_CUDA_MEMORY_MANAGER=nbep7.cupy_mempool python -m numba.runtests numba.cuda.tests
docs/source/developer/repomap.rst:CPU unit tests (GPU target unit tests listed in later sections
docs/source/developer/repomap.rst:CUDA GPU Target
docs/source/developer/repomap.rst:Note that the CUDA target does reuse some parts of the CPU target.
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/` - The implementation of the CUDA (NVIDIA GPU) target
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/decorators.py` - Compiler decorators for CUDA kernels
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/deviceufunc.py` - Custom ufunc dispatch for CUDA
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/dispatcher.py` - Dispatcher for CUDA JIT functions
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/printimpl.py` - Special implementation of device printing
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/libdevice.py` - Registers libdevice functions
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/kernels/` - Custom kernels for reduction and transpose
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/device_init.py` - Initializes the CUDA target when
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/compiler.py` - Compiler pipeline for CUDA target
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/intrinsic_wrapper.py` - CUDA device intrinsics
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/initialize.py` - Deferred initialization of the CUDA
docs/source/developer/repomap.rst:  device and subsystem.  Called only when user imports ``numba.cuda``
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/simulator_init.py` - Initializes the CUDA simulator
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/random.py` - Implementation of random number generator
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/api.py` - User facing APIs imported into ``numba.cuda.*``
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/stubs.py` - Python placeholders for functions that
docs/source/developer/repomap.rst:  only can be used in GPU device code
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/simulator/` - Simulate execution of CUDA kernels in
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/vectorizers.py` - Subclasses of ufunc/gufunc compilers
docs/source/developer/repomap.rst:  for CUDA
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/args.py` - Management of kernel arguments, including
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/target.py` - Typing and target contexts for GPU
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/cudamath.py` - Type signatures for math functions in
docs/source/developer/repomap.rst:  CUDA Python
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/errors.py` - Validation of kernel launch configuration
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/nvvmutils.py` - Helper functions for generating
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/testing.py` - Support code for creating CUDA unit
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/cudadecl.py` - Type signatures of CUDA API (threadIdx,
docs/source/developer/repomap.rst:  blockIdx, atomics) in Python on GPU
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/cudaimpl.py` - Implementations of CUDA API functions
docs/source/developer/repomap.rst:  on GPU
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/codegen.py` - Code generator object for CUDA target
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/cudadrv/` - Wrapper around CUDA driver API
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/cudadrv/dummyarray.py` - Used to hold array information
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/tests/` - CUDA unit tests, skipped when CUDA is not
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/tests/cudasim/` - Tests of CUDA simulator
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/tests/nocuda/` - Tests for NVVM functionality when
docs/source/developer/repomap.rst:  CUDA not present
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/tests/cudapy/` - Tests of compiling Python functions
docs/source/developer/repomap.rst:  for GPU
docs/source/developer/repomap.rst:- :ghfile:`numba/cuda/tests/cudadrv/` - Tests of Python wrapper around CUDA
docs/source/developer/architecture.rst:architectures like CPUs and GPUs.  In order to support these different
docs/source/developer/architecture.rst:and a "cuda" context for those two kinds of architecture, and a "parallel"
docs/source/developer/debugging.rst:       _context=<CUDATargetContext(address_size=64,
docs/source/developer/debugging.rst:       typing_context=<CUDATypingContext(_registries={<Registry(functions=[<type
docs/source/developer/debugging.rst:       '_ZN08NumbaEnv5numba4cuda5tests6cudapy13test_constmem19cuconstRecAlign$247E5ArrayIdLi1E1C7mutable7alignedE5ArrayIdLi1E1C7mutable7alignedE5ArrayIdLi1E1C7mutable7alignedE5ArrayIdLi1E1C7mutable7alignedE5ArrayIdLi1E1C7mutable7alignedE',
docs/source/developer/debugging.rst:       '_ZN5numba4cuda5tests6cudapy13test_constmem19cuconstRecAlign$247E5ArrayIdLi1E1C7mutable7alignedE5ArrayIdLi1E1C7mutable7alignedE5ArrayIdLi1E1C7mutable7alignedE5ArrayIdLi1E1C7mutable7alignedE5ArrayIdLi1E1C7mutable7alignedE'},
docs/source/developer/debugging.rst:       _basenamemap={}) at remote 0x1d27bf10>, triple='nvptx64-nvidia-cuda',
docs/source/developer/debugging.rst:       globals={'_ZN08NumbaEnv5numba4cuda5tests6cudapy13test_constmem19cuconstRecAlign$247E5ArrayIdLi1E1C7mutable7ali...(truncated),
docs/source/developer/debugging.rst:   _ZN5numba4cuda5tests6cudapy13test_constmem19cuconstRecAlign$247E5ArrayIdLi1E1C7mutable7alignedE5ArrayIdLi1E1C7mutable7alignedE5ArrayIdLi1E1C7mutable7alignedE5ArrayIdLi1E1C7mutable7alignedE5ArrayIdLi1E1C7mutable7alignedE
docs/source/developer/debugging.rst:   numba::cuda::tests::cudapy::test_constmem::cuconstRecAlign$247(
docs/source/developer/contributing.rst:platforms Numba supports. This includes ARMv8, and NVIDIA GPUs.
docs/source/developer/contributing.rst:   (e.g. OpenMP, TBB)) and then there's CUDA too and all its version
docs/source/developer/contributing.rst:If a Pull Request (PR) changes CUDA code or will affect the CUDA target, it
docs/source/developer/contributing.rst:needs to be run on `gpuCI <https://gpuci.gpuopenanalytics.com/job/numba/>`_.
docs/source/developer/contributing.rst:This can be triggered by one of the Numba maintainers commenting ``run gpuCI
docs/source/developer/contributing.rst:tests`` on the PR discussion. This runs the CUDA testsuite with various CUDA
docs/source/developer/contributing.rst:correctness of the changes with respect to CUDA. Following approval, the PR
docs/source/developer/contributing.rst:will also be run on Numba's build farm to test other configurations with CUDA
docs/source/developer/contributing.rst:(including Windows, which is not tested by gpuCI).
docs/source/developer/contributing.rst:If the PR is not CUDA-related but makes changes to something that the core
docs/source/index.rst:   :caption: For CUDA users
docs/source/index.rst:   cuda/index.rst
docs/source/index.rst:   cuda-reference/index.rst
docs/source/reference/jit-compilation.rst:   "parallel", and "cuda".  To use a multithreaded version, change the
docs/source/reference/jit-compilation.rst:   For the CUDA target, use "cuda"::
docs/source/reference/jit-compilation.rst:      @vectorize(["float64(float64)", "float32(float32)"], target='cuda')
docs/source/reference/envvars.rst:.. _numba-envvars-gpu-support:
docs/source/reference/envvars.rst:GPU support
docs/source/reference/envvars.rst:.. envvar:: NUMBA_DISABLE_CUDA
docs/source/reference/envvars.rst:   If set to non-zero, disable CUDA support.
docs/source/reference/envvars.rst:.. envvar:: NUMBA_FORCE_CUDA_CC
docs/source/reference/envvars.rst:   If set, force the CUDA compute capability to the given version (a
docs/source/reference/envvars.rst:.. envvar:: NUMBA_CUDA_DEFAULT_PTX_CC
docs/source/reference/envvars.rst:   target when compiling to PTX using ``cuda.compile_ptx``. The default is
docs/source/reference/envvars.rst:   recent version of the CUDA toolkit supported (12.4 at present).
docs/source/reference/envvars.rst:.. envvar:: NUMBA_ENABLE_CUDASIM
docs/source/reference/envvars.rst:   If set, don't compile and execute code for the GPU, but use the CUDA
docs/source/reference/envvars.rst:.. envvar:: NUMBA_CUDA_ARRAY_INTERFACE_SYNC
docs/source/reference/envvars.rst:   Whether to synchronize on streams provided by objects imported using the CUDA
docs/source/reference/envvars.rst:   takes place, and the user of Numba (and other CUDA libraries) is responsible
docs/source/reference/envvars.rst:.. envvar:: NUMBA_CUDA_LOG_LEVEL
docs/source/reference/envvars.rst:   variable is the logging level for CUDA API calls. The default value is
docs/source/reference/envvars.rst:.. envvar:: NUMBA_CUDA_LOG_API_ARGS
docs/source/reference/envvars.rst:   By default the CUDA API call logs only give the names of functions called.
docs/source/reference/envvars.rst:.. envvar:: NUMBA_CUDA_DRIVER
docs/source/reference/envvars.rst:   Path of the directory in which the CUDA driver libraries are to be found.
docs/source/reference/envvars.rst:.. envvar:: NUMBA_CUDA_LOG_SIZE
docs/source/reference/envvars.rst:   Buffer size for logs produced by CUDA driver API operations. This defaults
docs/source/reference/envvars.rst:.. envvar:: NUMBA_CUDA_VERBOSE_JIT_LOG
docs/source/reference/envvars.rst:   Whether the CUDA driver should produce verbose log messages. Defaults to 1,
docs/source/reference/envvars.rst:.. envvar:: NUMBA_CUDA_PER_THREAD_DEFAULT_STREAM
docs/source/reference/envvars.rst:   <https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html>`_
docs/source/reference/envvars.rst:   This variable only takes effect when using Numba's internal CUDA bindings;
docs/source/reference/envvars.rst:   when using the NVIDIA bindings, use the environment variable
docs/source/reference/envvars.rst:   ``CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM`` instead.
docs/source/reference/envvars.rst:      <https://nvidia.github.io/cuda-python/release/11.6.0-notes.html#default-stream>`_
docs/source/reference/envvars.rst:      in the NVIDIA Bindings documentation.
docs/source/reference/envvars.rst:.. envvar:: NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS
docs/source/reference/envvars.rst:   this warning will reduce the number of CUDA API calls (during JIT compilation), as the
docs/source/reference/envvars.rst:.. envvar:: NUMBA_CUDA_WARN_ON_IMPLICIT_COPY
docs/source/reference/envvars.rst:.. envvar:: NUMBA_CUDA_USE_NVIDIA_BINDING
docs/source/reference/envvars.rst:   When set to 1, Numba will attempt to use the `NVIDIA CUDA Python binding
docs/source/reference/envvars.rst:   <https://nvidia.github.io/cuda-python/>`_ to make calls to the driver API
docs/source/reference/envvars.rst:   NVIDIA binding is currently missing support for Per-Thread Default
docs/source/reference/envvars.rst:.. envvar:: NUMBA_CUDA_INCLUDE_PATH
docs/source/reference/envvars.rst:   The location of the CUDA include files. This is used when linking CUDA C/C++
docs/source/reference/envvars.rst:   sources to Python kernels, and needs to be correctly set for CUDA includes to
docs/source/reference/envvars.rst:   ``/usr/local/cuda/include``. On Windows, the default is
docs/source/reference/envvars.rst:   ``$env:CUDA_PATH\include``.
docs/source/reference/deprecation.rst:they support far more options and both the CPU and CUDA targets. Essentially a
docs/source/reference/deprecation.rst:Deprecation and removal of CUDA Toolkits < 11.2 and devices with CC < 5.0
docs/source/reference/deprecation.rst:- Support for CUDA toolkits less than 11.2 has been removed.
docs/source/reference/deprecation.rst:- CUDA toolkit 11.2 or later should be installed.
docs/source/reference/deprecation.rst:- In Numba 0.55.1: support for CC < 5.0 and CUDA toolkits < 10.2 was deprecated.
docs/source/reference/deprecation.rst:- In Numba 0.56: support for CC < 3.5 and CUDA toolkits < 10.2 was removed.
docs/source/reference/deprecation.rst:- In Numba 0.57: Support for CUDA toolkit 10.2 was removed.
docs/source/reference/deprecation.rst:- In Numba 0.58: Support CUDA toolkits 11.0 and 11.1 was removed.
docs/source/release-notes.rst:* PR `#8964 <https://github.com/numba/numba/pull/8964>`_: fix missing nopython keyword in cuda random module (`esc <https://github.com/esc>`_)
docs/source/release-notes.rst:* PR `#8895 <https://github.com/numba/numba/pull/8895>`_: CUDA: Enable caching functions that use CG (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#9005 <https://github.com/numba/numba/pull/9005>`_: Fix: Issue #8923 - avoid spurious device-to-host transfers in CUDA ufuncs (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:  ``@intrinsic`` functions will by default be accepted by both the CPU and CUDA
docs/source/release-notes.rst:CUDA:
docs/source/release-notes.rst:* New NVIDIA hardware and software compatibility / support:
docs/source/release-notes.rst:  * Toolkits: CUDA 11.8 and 12, with Minor Version Compatibility for 11.x.
docs/source/release-notes.rst:  * Packaging: NVIDIA-packaged CUDA toolkit conda packages.
docs/source/release-notes.rst:  * The high-level extension API is now fully-supported in the CUDA target.
docs/source/release-notes.rst:* PR `#7255 <https://github.com/numba/numba/pull/7255>`_: CUDA: Support CUDA Toolkit conda packages from NVIDIA (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7885 <https://github.com/numba/numba/pull/7885>`_: Adds CUDA FP16 arithmetic operators (`testhound <https://github.com/testhound>`_)
docs/source/release-notes.rst:* PR `#8001 <https://github.com/numba/numba/pull/8001>`_: CUDA fp16 math functions (`testhound <https://github.com/testhound>`_ `gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8167 <https://github.com/numba/numba/pull/8167>`_: CUDA: Facilitate and document passing arrays / pointers to foreign functions (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8180 <https://github.com/numba/numba/pull/8180>`_: CUDA: Initial support for Minor Version Compatibility (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8221 <https://github.com/numba/numba/pull/8221>`_: CUDA stubs docstring: Replace illegal escape sequence (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8253 <https://github.com/numba/numba/pull/8253>`_: CUDA: Verify NVVM IR prior to compilation (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8255 <https://github.com/numba/numba/pull/8255>`_: CUDA: Make numba.cuda.tests.doc_examples.ffi a module to fix #8252 (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8271 <https://github.com/numba/numba/pull/8271>`_: Implement some CUDA intrinsics with ``@overload``, ``@overload_attribute``, and ``@intrinsic`` (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8287 <https://github.com/numba/numba/pull/8287>`_: Drop CUDA 10.2 (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8290 <https://github.com/numba/numba/pull/8290>`_: CUDA: Replace use of deprecated NVVM IR features, questionable constructs (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8294 <https://github.com/numba/numba/pull/8294>`_: CUDA: Add trig ufunc support (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8297 <https://github.com/numba/numba/pull/8297>`_: Add __name__ attribute to CUDAUFuncDispatcher and test case (`testhound <https://github.com/testhound>`_)
docs/source/release-notes.rst:* PR `#8302 <https://github.com/numba/numba/pull/8302>`_: CUDA: Revert numba_nvvm intrinsic name workaround (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8308 <https://github.com/numba/numba/pull/8308>`_: CUDA: Support for multiple signatures (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8341 <https://github.com/numba/numba/pull/8341>`_: CUDA: Support multiple outputs for Generalized Ufuncs (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8401 <https://github.com/numba/numba/pull/8401>`_: Remove Cuda toolkit version check (`testhound <https://github.com/testhound>`_)
docs/source/release-notes.rst:* PR `#8525 <https://github.com/numba/numba/pull/8525>`_: Making CUDA specific datamodel manager (`sklam <https://github.com/sklam>`_)
docs/source/release-notes.rst:* PR `#8532 <https://github.com/numba/numba/pull/8532>`_: Vary NumPy version on gpuCI (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8542 <https://github.com/numba/numba/pull/8542>`_: CUDA: Make arg optional for Stream.add_callback() (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8572 <https://github.com/numba/numba/pull/8572>`_: CUDA: Reduce memory pressure from local memory tests (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8579 <https://github.com/numba/numba/pull/8579>`_: CUDA: Add CUDA 11.8 / Hopper support and required fixes (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8594 <https://github.com/numba/numba/pull/8594>`_: Fix various CUDA lineinfo issues (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8605 <https://github.com/numba/numba/pull/8605>`_: Support for CUDA fp16 math functions (part 1) (`testhound <https://github.com/testhound>`_)
docs/source/release-notes.rst:* PR `#8636 <https://github.com/numba/numba/pull/8636>`_: CUDA: Skip ``test_ptds`` on Windows (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8673 <https://github.com/numba/numba/pull/8673>`_: Enable the CUDA simulator tests on Windows builds in Azure CI. (`stuartarchibald <https://github.com/stuartarchibald>`_)
docs/source/release-notes.rst:* PR `#8723 <https://github.com/numba/numba/pull/8723>`_: Check for void return type in ``cuda.compile_ptx`` (`brandonwillard <https://github.com/brandonwillard>`_)
docs/source/release-notes.rst:* PR `#8764 <https://github.com/numba/numba/pull/8764>`_: CUDA tidy-up: remove some unneeded methods (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8788 <https://github.com/numba/numba/pull/8788>`_: CUDA: Fix returned dtype of vectorized functions (Issue #8400) (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8790 <https://github.com/numba/numba/pull/8790>`_: CUDA compare and swap with index (`ianthomas23 <https://github.com/ianthomas23>`_)
docs/source/release-notes.rst:* PR `#8826 <https://github.com/numba/numba/pull/8826>`_: CUDA CFFI test: conditionally require cffi module (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8866 <https://github.com/numba/numba/pull/8866>`_: Revise CUDA deprecation notices (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8870 <https://github.com/numba/numba/pull/8870>`_: Fix opcode "spelling" change since Python 3.11 in CUDA debug test. (`stuartarchibald <https://github.com/stuartarchibald>`_)
docs/source/release-notes.rst:This is a bugfix release to fix a regression in the CUDA target in relation to
docs/source/release-notes.rst:the ``.view()`` method on CUDA device arrays that is present when using NumPy
docs/source/release-notes.rst:* PR `#8570 <https://github.com/numba/numba/pull/8570>`_: Release 0.56 branch: Fix overloads with ``target="generic"`` for CUDA (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:``setuptools`` package and to fix a bug in the CUDA target in relation to
docs/source/release-notes.rst:This is a bugfix release that supports NumPy 1.23 and fixes CUDA function
docs/source/release-notes.rst:* PR `#8310 <https://github.com/numba/numba/pull/8310>`_: CUDA: Fix Issue #8309 - atomics don't work on complex components (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8374 <https://github.com/numba/numba/pull/8374>`_: Don't pickle LLVM IR for CUDA code libraries (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:CUDA enhancements:
docs/source/release-notes.rst:  * Support for linking CUDA C / C++ device functions into Python kernels.
docs/source/release-notes.rst:  * On-disk caching of CUDA kernels is now supported.
docs/source/release-notes.rst:* PR `#7363 <https://github.com/numba/numba/pull/7363>`_: Update cuda.local.array to clarify "simple constant expression" (e.g. no NumPy ints) (`Sterling Baird <https://github.com/sgbaird>`_)
docs/source/release-notes.rst:* PR `#7619 <https://github.com/numba/numba/pull/7619>`_: CUDA: Fix linking with PTX when compiling lazily (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7621 <https://github.com/numba/numba/pull/7621>`_: Add support for linking CUDA C / C++ with `@cuda.jit` kernels (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7680 <https://github.com/numba/numba/pull/7680>`_: CUDA Docs: include example calling slow matmul (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7699 <https://github.com/numba/numba/pull/7699>`_: CUDA: Provide helpful error if the return type is missing for `declare_device` (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7709 <https://github.com/numba/numba/pull/7709>`_: CUDA: Fixes missing type annotation pass following #7704 (`stuartarchibald <https://github.com/stuartarchibald>`_)
docs/source/release-notes.rst:* PR `#7740 <https://github.com/numba/numba/pull/7740>`_: CUDA Python 11.6 support (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7755 <https://github.com/numba/numba/pull/7755>`_: CUDA: Deprecate support for CC < 5.3 and CTK < 10.2 (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7772 <https://github.com/numba/numba/pull/7772>`_: CUDA: Add Support to Creating `StructModel` Array (`Michael Wang <https://github.com/isVoid>`_)
docs/source/release-notes.rst:* PR `#7814 <https://github.com/numba/numba/pull/7814>`_: CUDA Dispatcher refactor (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7815 <https://github.com/numba/numba/pull/7815>`_: CUDA Dispatcher refactor 2: inherit from `dispatcher.Dispatcher` (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7829 <https://github.com/numba/numba/pull/7829>`_: CUDA: Support `Enum/IntEnum` in Kernel (`Michael Wang <https://github.com/isVoid>`_)
docs/source/release-notes.rst:* PR `#7846 <https://github.com/numba/numba/pull/7846>`_: Fix CUDA enum vectorize test on Windows (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7850 <https://github.com/numba/numba/pull/7850>`_: CUDA: Pass `fastmath` compiler flag down to `compile_ptx` and `compile_device`; Improve `fastmath` tests (`Michael Wang <https://github.com/isVoid>`_)
docs/source/release-notes.rst:* PR `#7858 <https://github.com/numba/numba/pull/7858>`_: CUDA: Deprecate `ptx` Attribute and Update Tests (`Graham Markall <https://github.com/gmarkall>`_ `Michael Wang <https://github.com/isVoid>`_)
docs/source/release-notes.rst:* PR `#7878 <https://github.com/numba/numba/pull/7878>`_: CUDA: Remove some deprecated support, add CC 8.6 and 8.7 (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7898 <https://github.com/numba/numba/pull/7898>`_: Skip test_ptds under cuda-memcheck (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7915 <https://github.com/numba/numba/pull/7915>`_: CUDA: Fix test checking debug info rendering. (`stuartarchibald <https://github.com/stuartarchibald>`_)
docs/source/release-notes.rst:* PR `#7918 <https://github.com/numba/numba/pull/7918>`_: Add JIT examples to CUDA docs (`brandon-b-miller <https://github.com/brandon-b-miller>`_ `Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7949 <https://github.com/numba/numba/pull/7949>`_: Add Cuda Vector Types (`Michael Wang <https://github.com/isVoid>`_)
docs/source/release-notes.rst:* PR `#7972 <https://github.com/numba/numba/pull/7972>`_: Fix fp16 support for cuda shared array (`Michael Collison <https://github.com/testhound>`_ `Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8004 <https://github.com/numba/numba/pull/8004>`_: CUDA fixes for Windows (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8037 <https://github.com/numba/numba/pull/8037>`_: CUDA self-recursion tests (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8089 <https://github.com/numba/numba/pull/8089>`_: Support on-disk caching in the CUDA target (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8123 <https://github.com/numba/numba/pull/8123>`_: Fix CUDA print tests on Windows (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8137 <https://github.com/numba/numba/pull/8137>`_: CUDA: Fix #7806, Division by zero stops the kernel (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8142 <https://github.com/numba/numba/pull/8142>`_: CUDA: Fix some missed changes from dropping 9.2 (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8163 <https://github.com/numba/numba/pull/8163>`_: CUDA: Remove context query in launch config (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8170 <https://github.com/numba/numba/pull/8170>`_: CUDA: Fix missing space in low occupancy warning (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8191 <https://github.com/numba/numba/pull/8191>`_: CUDA: Update deprecation notes for 0.56. (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#8255 <https://github.com/numba/numba/pull/8255>`_: CUDA: Make numba.cuda.tests.doc_examples.ffi a module to fix #8252 (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:CUDA target deprecation notices:
docs/source/release-notes.rst:* Support for CUDA toolkits < 10.2 is deprecated and will be removed in Numba
docs/source/release-notes.rst:* PR `#7755 <https://github.com/numba/numba/pull/7755>`_: CUDA: Deprecate support for CC < 5.3 and CTK < 10.2 (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7740 <https://github.com/numba/numba/pull/7740>`_: CUDA Python 11.6 support (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7709 <https://github.com/numba/numba/pull/7709>`_: CUDA: Fixes missing type annotation pass following #7704 (`stuartarchibald <https://github.com/stuartarchibald>`_)
docs/source/release-notes.rst:* PR `#7619 <https://github.com/numba/numba/pull/7619>`_: CUDA: Fix linking with PTX when compiling lazily (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:Highlights of changes for the CUDA target:
docs/source/release-notes.rst:* Support for NVIDIA's CUDA Python bindings.
docs/source/release-notes.rst:* Much underlying refactoring to align the CUDA target more closely with the
docs/source/release-notes.rst:  API in CUDA in future releases.
docs/source/release-notes.rst:CUDA target deprecation notices:
docs/source/release-notes.rst:* There are no new CUDA target deprecations.
docs/source/release-notes.rst:* PR `#7057 <https://github.com/numba/numba/pull/7057>`_: Fix #7041: Add charseq registry to CUDA target (`Graham Markall <https://github.com/gmarkall>`_ `stuartarchibald <https://github.com/stuartarchibald>`_)
docs/source/release-notes.rst:* PR `#7189 <https://github.com/numba/numba/pull/7189>`_: CUDA: Skip IPC tests on ARM (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7190 <https://github.com/numba/numba/pull/7190>`_: CUDA: Fix test_pinned on Jetson (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7221 <https://github.com/numba/numba/pull/7221>`_: Show GPU UUIDs in cuda.detect() output (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7222 <https://github.com/numba/numba/pull/7222>`_: CUDA: Warn when debug=True and opt=True (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7236 <https://github.com/numba/numba/pull/7236>`_: CUDA: Skip managed alloc tests on ARM (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7266 <https://github.com/numba/numba/pull/7266>`_: CUDA: Skip multi-GPU copy test with peer access disabled (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7285 <https://github.com/numba/numba/pull/7285>`_: CUDA: Fix OOB in test_kernel_arg (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7329 <https://github.com/numba/numba/pull/7329>`_: Improve documentation in reference to CUDA local memory (`Sterling Baird <https://github.com/sgbaird>`_)
docs/source/release-notes.rst:* PR `#7330 <https://github.com/numba/numba/pull/7330>`_: Cuda matmul docs (`Sterling Baird <https://github.com/sgbaird>`_)
docs/source/release-notes.rst:* PR `#7375 <https://github.com/numba/numba/pull/7375>`_: CUDA: Run doctests as part of numba.cuda.tests and fix test_cg (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7453 <https://github.com/numba/numba/pull/7453>`_: CUDA: Provide stream in async_done result (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7460 <https://github.com/numba/numba/pull/7460>`_: Add FP16 support for CUDA (`Michael Collison <https://github.com/testhound>`_ `Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7461 <https://github.com/numba/numba/pull/7461>`_: Support NVIDIA's CUDA Python bindings (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7479 <https://github.com/numba/numba/pull/7479>`_: CUDA: Print format string and warn for > 32 print() args (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7484 <https://github.com/numba/numba/pull/7484>`_: Fixed outgoing link to nvidia documentation. (`Dhruv Patel <https://github.com/DhruvPatel01>`_)
docs/source/release-notes.rst:* PR `#7496 <https://github.com/numba/numba/pull/7496>`_: CUDA: Use a single dispatcher class for all kinds of functions (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7499 <https://github.com/numba/numba/pull/7499>`_: Add build scripts for CUDA testing on gpuCI  (`Charles Blackmon-Luca <https://github.com/charlesbluca>`_ `Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7514 <https://github.com/numba/numba/pull/7514>`_: Fixup cuda debuginfo emission for 7177 (`Siu Kwan Lam <https://github.com/sklam>`_)
docs/source/release-notes.rst:* PR `#7632 <https://github.com/numba/numba/pull/7632>`_: Capture output in CUDA matmul doctest (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:type handling, a potential leak on initialization failure in the CUDA target, a
docs/source/release-notes.rst:* PR `#7360 <https://github.com/numba/numba/pull/7360>`_: CUDA: Fix potential leaks when initialization fails (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:CUDA target changes:
docs/source/release-notes.rst:* Support for passing tuples to CUDA ufuncs
docs/source/release-notes.rst:  * Using support for more CUDA functions: ``activemask()``, ``lanemask_lt()``
docs/source/release-notes.rst:* Support for ``@overload`` in the CUDA target
docs/source/release-notes.rst:* The ``ROCm`` target (for AMD ROC GPUs) has been moved to an "unmaintained"
docs/source/release-notes.rst:  https://github.com/numba/numba-rocm
docs/source/release-notes.rst:CUDA target deprecations and breaking changes:
docs/source/release-notes.rst:* PR `#6695 <https://github.com/numba/numba/pull/6695>`_: Enable negative indexing for cuda atomic operations (`Ashutosh Varma <https://github.com/ashutoshvarma>`_)
docs/source/release-notes.rst:* PR `#6700 <https://github.com/numba/numba/pull/6700>`_: Add UUID to CUDA devices (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6731 <https://github.com/numba/numba/pull/6731>`_: Add CUDA-specific pipeline (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6735 <https://github.com/numba/numba/pull/6735>`_: CUDA: Don't parse IR for modules with llvmlite (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6769 <https://github.com/numba/numba/pull/6769>`_: CUDA: Replace ``CachedPTX`` and ``CachedCUFunction`` with ``CUDACodeLibrary`` functionality (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6795 <https://github.com/numba/numba/pull/6795>`_: CUDA: Lazily add libdevice to compilation units  (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6798 <https://github.com/numba/numba/pull/6798>`_: CUDA: Add optional Driver API argument logging (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6809 <https://github.com/numba/numba/pull/6809>`_: #3203 additional info in cuda detect (`Kalyan <https://github.com/rawwar>`_)
docs/source/release-notes.rst:* PR `#6811 <https://github.com/numba/numba/pull/6811>`_: CUDA: Remove test of runtime being a supported version (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6813 <https://github.com/numba/numba/pull/6813>`_: Mostly CUDA: Replace llvmpy API usage with llvmlite APIs (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6818 <https://github.com/numba/numba/pull/6818>`_: CUDA: Support IPC on Windows (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6830 <https://github.com/numba/numba/pull/6830>`_: CUDA: Use relaxed strides checking to compute contiguity (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6836 <https://github.com/numba/numba/pull/6836>`_: CUDA: Documentation updates (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6840 <https://github.com/numba/numba/pull/6840>`_: CUDA: Remove items deprecated in 0.53 + simulator test fixes (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6841 <https://github.com/numba/numba/pull/6841>`_: CUDA: Fix source location on kernel entry and enable breakpoints to be set on kernels by mangled name (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6844 <https://github.com/numba/numba/pull/6844>`_: CUDA: Remove NUMBAPRO env var warnings, envvars.py + other small tidy-ups (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6878 <https://github.com/numba/numba/pull/6878>`_: CUDA: Support passing tuples to ufuncs (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6879 <https://github.com/numba/numba/pull/6879>`_: CUDA: NumPy and string dtypes for local and shared arrays (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6885 <https://github.com/numba/numba/pull/6885>`_: CUDA: Explicitly specify objmode + looplifting for jit functions in cuda.random (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6886 <https://github.com/numba/numba/pull/6886>`_: CUDA: Fix parallel testing for all testsuite submodules (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6889 <https://github.com/numba/numba/pull/6889>`_: Address guvectorize too slow for cuda target (`Michael Collison <https://github.com/testhound>`_)
docs/source/release-notes.rst:* PR `#6911 <https://github.com/numba/numba/pull/6911>`_: CUDA: Add support for activemask(), lanemask_lt(), and nanosleep() (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6924 <https://github.com/numba/numba/pull/6924>`_: CUDA: Fix ``ffs`` (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6936 <https://github.com/numba/numba/pull/6936>`_: CUDA: Implement support for PTDS globally (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6944 <https://github.com/numba/numba/pull/6944>`_: CUDA: Support for ``@overload`` (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6953 <https://github.com/numba/numba/pull/6953>`_: CUDA: Fix and deprecate ``inspect_ptx()``, fix NVVM option setup for device functions (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#6958 <https://github.com/numba/numba/pull/6958>`_: Inconsistent behavior of reshape between numpy and numba/cuda device array (`Lauren Arnett <https://github.com/laurenarnett>`_)
docs/source/release-notes.rst:* PR `#6971 <https://github.com/numba/numba/pull/6971>`_: Fix CUDA ``@intrinsic`` use (`stuartarchibald <https://github.com/stuartarchibald>`_)
docs/source/release-notes.rst:* PR `#6991 <https://github.com/numba/numba/pull/6991>`_: Move ROCm target status to "unmaintained". (`stuartarchibald <https://github.com/stuartarchibald>`_)
docs/source/release-notes.rst:* PR `#6997 <https://github.com/numba/numba/pull/6997>`_: CUDA: Remove catch of NotImplementedError in target.py (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7004 <https://github.com/numba/numba/pull/7004>`_: Test extending the CUDA target (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7052 <https://github.com/numba/numba/pull/7052>`_: Fix string support in CUDA target (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7114 <https://github.com/numba/numba/pull/7114>`_: CUDA: Deprecate eager compilation of device functions (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7152 <https://github.com/numba/numba/pull/7152>`_: Fix iterators in CUDA (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7162 <https://github.com/numba/numba/pull/7162>`_: CUDA: Fix linkage of device functions when compiling for debug (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7189 <https://github.com/numba/numba/pull/7189>`_: CUDA: Skip IPC tests on ARM (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7190 <https://github.com/numba/numba/pull/7190>`_: CUDA: Fix test_pinned on Jetson (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7236 <https://github.com/numba/numba/pull/7236>`_: CUDA: Skip managed alloc tests on ARM (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR `#7285 <https://github.com/numba/numba/pull/7285>`_: CUDA: Fix OOB in test_kernel_arg (`Graham Markall <https://github.com/gmarkall>`_)
docs/source/release-notes.rst:* PR #6828 Fix regression in CUDA: Set stream in mapped and managed array
docs/source/release-notes.rst:Highlights of changes for the CUDA target:
docs/source/release-notes.rst:* CUDA 11.2 onwards (versions of the toolkit using NVVM IR 1.6 / LLVM IR 7.0.1)
docs/source/release-notes.rst:* Addition of ``cuda.is_supported_version()`` to check if the CUDA runtime
docs/source/release-notes.rst:* The CUDA dispatcher now shares infrastructure with the CPU dispatcher,
docs/source/release-notes.rst:* The CUDA Array Interface is updated to version 3, with support for streams
docs/source/release-notes.rst:CUDA target deprecation notices:
docs/source/release-notes.rst:* CUDA support on macOS is deprecated with this release (it still works, it is
docs/source/release-notes.rst:  ``cuda.jit`` decorator, deprecated since 0.51.0, are removed
docs/source/release-notes.rst:CUDA Enhancements/Fixes:
docs/source/release-notes.rst:* PR #5162: Specify synchronization semantics of CUDA Array Interface (Graham
docs/source/release-notes.rst:* PR #6245: CUDA Cooperative grid groups (Graham Markall and Nick White)
docs/source/release-notes.rst:* PR #6343: CUDA: Add support for passing tuples and namedtuples to kernels
docs/source/release-notes.rst:* PR #6358: Add log2 and remainder implementations for cuda (Guilherme Leobas)
docs/source/release-notes.rst:* PR #6377: CUDA: Fix various issues in test suite (Graham Markall)
docs/source/release-notes.rst:* PR #6409: Implement cuda atomic xor (Michael Collison)
docs/source/release-notes.rst:* PR #6422: CUDA: Remove deprecated items, expect CUDA 11.1 (Graham Markall)
docs/source/release-notes.rst:* PR #6432: CUDA: Use ``_dispatcher.Dispatcher`` as base Dispatcher class
docs/source/release-notes.rst:* PR #6447: CUDA: Add get_regs_per_thread method to Dispatcher (Graham Markall)
docs/source/release-notes.rst:* PR #6499: CUDA atomic increment, decrement, exchange and compare and swap
docs/source/release-notes.rst:* PR #6510: CUDA: Make device array assignment synchronous where necessary
docs/source/release-notes.rst:* PR #6517: CUDA: Add NVVM test of all 8-bit characters (Graham Markall)
docs/source/release-notes.rst:* PR #6642: Testhound/cuda cuberoot (Michael Collison)
docs/source/release-notes.rst:* PR #6661: CUDA: Support NVVM70 / CUDA 11.2 (Graham Markall)
docs/source/release-notes.rst:* PR #6666: CUDA: Add a function to query whether the runtime version is
docs/source/release-notes.rst:* PR #6725: CUDA: Fix compile to PTX with debug for CUDA 11.2 (Graham Markall)
docs/source/release-notes.rst:* PR #6430: CUDA docs: Add RNG example with 3D grid and strided loops (Graham
docs/source/release-notes.rst:* On the CUDA target:
docs/source/release-notes.rst:  * CUDA 9.0 is now the minimum supported version (Graham Markall).
docs/source/release-notes.rst:  * Cudasim support for mapped array, memcopies and memset has been added (Mike
docs/source/release-notes.rst:  * Additional CUDA atomic operations have been added (Michael Collison).
docs/source/release-notes.rst:CUDA Enhancements/Fixes:
docs/source/release-notes.rst:* PR #5741: CUDA: Add two-argument implementation of round() (Graham Markall)
docs/source/release-notes.rst:* PR #5900: Enable CUDA Unified Memory (Max Katz)
docs/source/release-notes.rst:* PR #6042: CUDA: Lower launch overhead by launching kernel directly (Graham
docs/source/release-notes.rst:* PR #6064: Lower math.frexp and math.ldexp in numba.cuda (Zhihao Yuan)
docs/source/release-notes.rst:* PR #6066: Lower math.isfinite in numba.cuda (Zhihao Yuan)
docs/source/release-notes.rst:* PR #6092: CUDA: Add mapped_array_like and pinned_array_like (Graham Markall)
docs/source/release-notes.rst:* PR #6127: Fix race in reduction kernels on Volta, require CUDA 9, add syncwarp
docs/source/release-notes.rst:* PR #6129: Extend Cudasim to support most of the memory functionality. (Mike
docs/source/release-notes.rst:* PR #6150: CUDA: Turn on flake8 for cudadrv and fix errors (Graham Markall)
docs/source/release-notes.rst:* PR #6152: CUDA: Provide wrappers for all libdevice functions, and fix typing
docs/source/release-notes.rst:* PR #6244: CUDA Docs: Make workflow using simulator more explicit (Graham
docs/source/release-notes.rst:* PR #6248: Add support for CUDA atomic subtract operations (Michael Collison)
docs/source/release-notes.rst:* PR #6290: CUDA: Add support for complex power (Graham Markall)
docs/source/release-notes.rst:* PR #6296: Fix flake8 violations in numba.cuda module (Graham Markall)
docs/source/release-notes.rst:* PR #6297: Fix flake8 violations in numba.cuda.tests.cudapy module (Graham
docs/source/release-notes.rst:* PR #6298: Fix flake8 violations in numba.cuda.tests.cudadrv (Graham Markall)
docs/source/release-notes.rst:* PR #6299: Fix flake8 violations in numba.cuda.simulator (Graham Markall)
docs/source/release-notes.rst:* PR #6306: Fix flake8 in cuda atomic test from merge. (Stuart Archibald)
docs/source/release-notes.rst:* PR #6329: Flake8 fix for a CUDA test (Stuart Archibald)
docs/source/release-notes.rst:* PR #6331: Explicitly state that NUMBA_ENABLE_CUDASIM needs to be set before
docs/source/release-notes.rst:* PR #6340: CUDA: Fix #6339, performance regression launching specialized
docs/source/release-notes.rst:* PR #6128: CUDA Docs: Restore Dispatcher.forall() docs (Graham Markall)
docs/source/release-notes.rst:critical bug in the CUDA target initialisation sequence and also fixes some
docs/source/release-notes.rst:* PR #6147: CUDA: Don't make a runtime call on import
docs/source/release-notes.rst:* PR #6168: Fix Issue #6167: Failure in test_cuda_submodules
docs/source/release-notes.rst:* On the CUDA target: Support for CUDA Toolkit 11, Ampere, and Compute
docs/source/release-notes.rst:  functions can be inserted into CUDA streams, and streams are async awaitable;
docs/source/release-notes.rst:``@cuda.jit`` are now deprecated, the ``bind`` kwarg is also deprecated.
docs/source/release-notes.rst:CUDA Enhancements/Fixes:
docs/source/release-notes.rst:* PR #5709: CUDA: Refactoring of cuda.jit and kernel / dispatcher abstractions
docs/source/release-notes.rst:* PR #5732: CUDA Docs: document ``forall`` method of kernels
docs/source/release-notes.rst:* PR #5745: CUDA stream callbacks and async awaitable streams
docs/source/release-notes.rst:* PR #5761: Add implmentation for int types for isnan and isinf for CUDA
docs/source/release-notes.rst:* PR #5819: Add support for CUDA 11 and Ampere / CC 8.0
docs/source/release-notes.rst:* PR #5826: CUDA: Add function to get SASS for kernels
docs/source/release-notes.rst:* PR #5846: CUDA: Allow disabling NVVM optimizations, and fix debug issues
docs/source/release-notes.rst:* PR #5851: CUDA EMM enhancements - add default get_ipc_handle implementation,
docs/source/release-notes.rst:* PR #5852: CUDA: Fix ``cuda.test()``
docs/source/release-notes.rst:* PR #5857: CUDA docs: Add notes on resetting the EMM plugin
docs/source/release-notes.rst:* PR #5859: CUDA: Fix reduce docs and style improvements
docs/source/release-notes.rst:* PR #6016: Fixes change of list spelling in a cuda test.
docs/source/release-notes.rst:* PR #6020: CUDA: Fix #5820, adding atomic nanmin / nanmax
docs/source/release-notes.rst:* PR #6030: CUDA: Don't optimize IR before sending it to NVVM
docs/source/release-notes.rst:* PR #6080: CUDA: Prevent auto-upgrade of atomic intrinsics
docs/source/release-notes.rst:* PR #6013: emphasize cuda kernel functions are asynchronous
docs/source/release-notes.rst:* Graham Markall contributed many patches to the CUDA target, as follows:
docs/source/release-notes.rst:  * #6030: CUDA: Don't optimize IR before sending it to NVVM
docs/source/release-notes.rst:  * #5846: CUDA: Allow disabling NVVM optimizations, and fix debug issues
docs/source/release-notes.rst:  * #5826: CUDA: Add function to get SASS for kernels
docs/source/release-notes.rst:  * #5851: CUDA EMM enhancements - add default get_ipc_handle implementation,
docs/source/release-notes.rst:  * #5709: CUDA: Refactoring of cuda.jit and kernel / dispatcher abstractions
docs/source/release-notes.rst:  * #5819: Add support for CUDA 11 and Ampere / CC 8.0
docs/source/release-notes.rst:  * #6020: CUDA: Fix #5820, adding atomic nanmin / nanmax
docs/source/release-notes.rst:  * #5857: CUDA docs: Add notes on resetting the EMM plugin
docs/source/release-notes.rst:  * #5859: CUDA: Fix reduce docs and style improvements
docs/source/release-notes.rst:  * #5852: CUDA: Fix ``cuda.test()``
docs/source/release-notes.rst:  * #5732: CUDA Docs: document ``forall`` method of kernels
docs/source/release-notes.rst:* Kayran Schmidt emphasized that CUDA kernel functions are asynchronous in the
docs/source/release-notes.rst:  CUDA target in #5761 and implemented ``np.positive`` in #5796.
docs/source/release-notes.rst:* Peter Würtz added CUDA stream callbacks and async awaitable streams in #5745.
docs/source/release-notes.rst:* PR #5918: Fix cuda test due to #5876
docs/source/release-notes.rst:* The CUDA target has more stream constructors available and a new function for
docs/source/release-notes.rst:  the macro-based system for describing CUDA threads and blocks has been
docs/source/release-notes.rst:CUDA Enhancements/Fixes:
docs/source/release-notes.rst:* PR #5347: CUDA: Provide more stream constructors
docs/source/release-notes.rst:* PR #5388: CUDA: Fix OOB write in test_round{f4,f8}
docs/source/release-notes.rst:  ``as_cuda_array(...)``
docs/source/release-notes.rst:* PR #5481: CUDA: Replace macros with typing and lowering implementations
docs/source/release-notes.rst:* PR #5556: CUDA: Make atomic semantics match Python / NumPy, and fix #5458
docs/source/release-notes.rst:* PR #5558: CUDA: Only release primary ctx if retained
docs/source/release-notes.rst:* PR #5561: CUDA: Add function for compiling to PTX (+ other small fixes)
docs/source/release-notes.rst:* PR #5573: CUDA: Skip tests under cuda-memcheck that hang it
docs/source/release-notes.rst:* PR #5578: Implement math.modf for CUDA target
docs/source/release-notes.rst:* PR #5704: CUDA Eager compilation: Fix max_registers kwarg
docs/source/release-notes.rst:* PR #5718: CUDA lib path tests: unset CUDA_PATH when CUDA_HOME unset
docs/source/release-notes.rst:* PR #5803: CUDA Update expected error messages to fix #5797
docs/source/release-notes.rst:* Gabriele Gemmi implemented ``math.modf`` for the CUDA target in #5578
docs/source/release-notes.rst:* Graham Markall contributed many patches, largely to the CUDA target, as
docs/source/release-notes.rst:  * #5347: CUDA: Provide more stream constructors
docs/source/release-notes.rst:  * #5388: CUDA: Fix OOB write in test_round{f4,f8}
docs/source/release-notes.rst:    ``as_cuda_array(...)``
docs/source/release-notes.rst:  * #5481: CUDA: Replace macros with typing and lowering implementations
docs/source/release-notes.rst:  * #5556: CUDA: Make atomic semantics match Python / NumPy, and fix #5458
docs/source/release-notes.rst:  * #5558: CUDA: Only release primary ctx if retained
docs/source/release-notes.rst:  * #5561: CUDA: Add function for compiling to PTX (+ other small fixes)
docs/source/release-notes.rst:  * #5573: CUDA: Skip tests under cuda-memcheck that hang it
docs/source/release-notes.rst:  * #5704: CUDA Eager compilation: Fix max_registers kwarg
docs/source/release-notes.rst:  * #5718: CUDA lib path tests: unset CUDA_PATH when CUDA_HOME unset
docs/source/release-notes.rst:  * #5803: CUDA Update expected error messages to fix #5797
docs/source/release-notes.rst:* For the CUDA target, all kernel launches now require a configuration, this
docs/source/release-notes.rst:  tuning is deferred to CUDA API calls that provide the same functionality
docs/source/release-notes.rst:* The CUDA target also gained an External Memory Management plugin interface to
docs/source/release-notes.rst:  allow Numba to use another CUDA-aware library for all memory allocations and
docs/source/release-notes.rst:  of long standing issues in #5346. Also contributed were a large number of CUDA
docs/source/release-notes.rst:  * #5519: CUDA: Silence the test suite - Fix #4809, remove autojit, delete
docs/source/release-notes.rst:  * #5443: Fix #5196: Docs: assert in CUDA only enabled for debug
docs/source/release-notes.rst:  * #5423: Fix #5421: Add notes on printing in CUDA kernels
docs/source/release-notes.rst:  * #5400: Fix #4954, and some other small CUDA testsuite fixes
docs/source/release-notes.rst:  * #5323: Document lifetime semantics of CUDA Array Interface
docs/source/release-notes.rst:  * #5136: CUDA: Enable asynchronous operations on the default stream
docs/source/release-notes.rst:  * #5059: Docs: Explain how to use Memcheck with Numba, fixups in CUDA
docs/source/release-notes.rst:* John Kirkham added ``numpy.dtype`` coercion for the ``dtype`` argument to CUDA
docs/source/release-notes.rst:* Leo Fang added a list of libraries that support ``__cuda_array_interface__``
docs/source/release-notes.rst:* Mads R. B. Kristensen fixed an issue with ``__cuda_array_interface__`` not
docs/source/release-notes.rst:* Mike Williams fixed some issues with NumPy records and ``getitem`` in the CUDA
docs/source/release-notes.rst:* hdf fixed an issue with the ``boundscheck`` flag in the CUDA jit target in
docs/source/release-notes.rst:CUDA Enhancements/Fixes:
docs/source/release-notes.rst:* PR #5104: Add a list of libraries that support __cuda_array_interface__
docs/source/release-notes.rst:* PR #5136: CUDA: Enable asynchronous operations on the default stream
docs/source/release-notes.rst:* PR #5189: __cuda_array_interface__ not requiring context
docs/source/release-notes.rst:* PR #5323: Document lifetime semantics of CUDA Array Interface
docs/source/release-notes.rst:* PR #5343: Fix cuda spoof
docs/source/release-notes.rst:* PR #5400: Fix #4954, and some other small CUDA testsuite fixes
docs/source/release-notes.rst:* PR #5519: CUDA: Silence the test suite - Fix #4809, remove autojit, delete
docs/source/release-notes.rst:* PR #5059: Docs: Explain how to use Memcheck with Numba, fixups in CUDA
docs/source/release-notes.rst:* PR #5423: Fix #5421: Add notes on printing in CUDA kernels
docs/source/release-notes.rst:* PR #5443: Fix #5196: Docs: assert in CUDA only enabled for debug
docs/source/release-notes.rst:needed for the end of Python 2.7 support, improvements to the CUDA target and
docs/source/release-notes.rst:* Graham Markall contributed a large number of CUDA enhancements and fixes,
docs/source/release-notes.rst:  * #5016: Fix various issues in CUDA library search (Fixes #4979)
docs/source/release-notes.rst:  * #4964: Fix #4628: Add more appropriate typing for CUDA device arrays
docs/source/release-notes.rst:  * #4997: State that CUDA Toolkit 8.0 required in docs
docs/source/release-notes.rst:* John Kirkham added a clarification to the ``__cuda_array_interface__``
docs/source/release-notes.rst:CUDA Enhancements/Fixes:
docs/source/release-notes.rst:* PR #4964: Fix #4628: Add more appropriate typing for CUDA device arrays
docs/source/release-notes.rst:* PR #4997: State that CUDA Toolkit 8.0 required in docs
docs/source/release-notes.rst:* PR #5016: Fix various issues in CUDA library search (Fixes #4979)
docs/source/release-notes.rst:* Graham Markall fixed some issues with the CUDA target, namely:
docs/source/release-notes.rst:  * #4931: Added physical limits for CC 7.0 / 7.5 to CUDA autotune
docs/source/release-notes.rst:  * #4934: Fixed bugs in TestCudaWarpOperations
docs/source/release-notes.rst:  * #4938: Improved errors / warnings for the CUDA vectorize decorator
docs/source/release-notes.rst:CUDA Enhancements/Fixes:
docs/source/release-notes.rst:* PR #4675: Bump cuda array interface to version 2
docs/source/release-notes.rst:* PR #4741: Update choosing the "CUDA_PATH" for windows
docs/source/release-notes.rst:* PR #4838: Permit ravel('A') for contig device arrays in CUDA target
docs/source/release-notes.rst:* PR #4934: Fix fails in TestCudaWarpOperations
docs/source/release-notes.rst:* PR #4938: Improve errors / warnings for cuda vectorize decorator
docs/source/release-notes.rst:This release has updated the CUDA Array Interface specification to version 2,
docs/source/release-notes.rst:* Ashwin Srinath fixed a CUDA performance bug via #4576.
docs/source/release-notes.rst:* Leo Fang updated the CUDA Array Interface contract in #4609.
docs/source/release-notes.rst:* Peter Andreas Entschev fixed a CUDA concurrency bug in #4581.
docs/source/release-notes.rst:CUDA Enhancements/Fixes:
docs/source/release-notes.rst:* PR #4410: Fix #4111. cudasim mishandling recarray
docs/source/release-notes.rst:* PR #4609: Update CUDA Array Interface & Enforce Numba compliance
docs/source/release-notes.rst:* PR #4619: Implement math.{degrees, radians} for the CUDA target.
docs/source/release-notes.rst:* PR #4675: Bump cuda array interface to version 2
docs/source/release-notes.rst:* PR #4493: Fix Overload Inliner wrt CUDA Intrinsics
docs/source/release-notes.rst:* Nick White fixed the issue with ``round`` in the CUDA target in #4137.
docs/source/release-notes.rst:* Keith Kraus extended the ``__cuda_array_interface__`` with an optional mask
docs/source/release-notes.rst:CUDA Enhancements/Fixes:
docs/source/release-notes.rst:* PR #4199: Extend `__cuda_array_interface__` with optional mask attribute,
docs/source/release-notes.rst:* PR #4137: CUDA - Fix round Builtin
docs/source/release-notes.rst:* PR #4114: Support 3rd party activated CUDA context
docs/source/release-notes.rst:- Nick White enhanced the CUDA backend to use min/max PTX instructions where
docs/source/release-notes.rst:CUDA Enhancements/Fixes:
docs/source/release-notes.rst:* PR #3933: Adds `.nbytes` property to CUDA device array objects.
docs/source/release-notes.rst:* PR #4011: Add .inspect_ptx() to cuda device function
docs/source/release-notes.rst:* PR #4054: CUDA: Use min/max PTX Instructions
docs/source/release-notes.rst:* PR #4096: Update env-vars for CUDA libraries lookup
docs/source/release-notes.rst:* PR #4105: Remove info about CUDA ENVVAR potential replacement
docs/source/release-notes.rst:CUDA Enhancements/Fixes:
docs/source/release-notes.rst:* PR #3755: Make cuda.to_device accept readonly host array
docs/source/release-notes.rst:- #3532. Daniel Wennberg improved the ``cuda.{pinned, mapped}`` API so that
docs/source/release-notes.rst:CUDA Enhancements:
docs/source/release-notes.rst:* PR #3578: Align cuda and cuda simulator kwarg names.
docs/source/release-notes.rst:* PR #3570: Minor documentation fixes for numba.cuda
docs/source/release-notes.rst:CUDA Enhancements:
docs/source/release-notes.rst:* PR #3399: Add max_registers Option to cuda.jit
docs/source/release-notes.rst:* PR #3419: Fix cuda tests and error reporting in test discovery
docs/source/release-notes.rst:* PR #3493: Fix CUDA test that used negative indexing behaviour that's fixed.
docs/source/release-notes.rst:* PR #3335: Fix memory management of __cuda_array_interface__ views.
docs/source/release-notes.rst:* PR #3382: CUDA_ERROR_MISALIGNED_ADDRESS Using Multiple Const Arrays
docs/source/release-notes.rst:* PR #3322: Add FAQ on CUDA + fork issue. Fixes #3315.
docs/source/release-notes.rst:* A new GPU backend: kernels for AMD GPUs can now be compiled using the ROCm
docs/source/release-notes.rst:* CUDA improvements: FMA, faster float64 atomics on supporting hardware,
docs/source/release-notes.rst:CUDA Enhancements:
docs/source/release-notes.rst:* PR #3152: Use cuda driver api to get best blocksize for best occupancy
docs/source/release-notes.rst:* PR #3186: Support Records in CUDA Const Memory
docs/source/release-notes.rst:* PR #3191: CUDA: fix log size
docs/source/release-notes.rst:* PR #3198: Fix GPU datetime timedelta types usage
docs/source/release-notes.rst:* PR #3221: Support datetime/timedelta scalar argument to a CUDA kernel.
docs/source/release-notes.rst:* PR #3310: Fix IPC handling of sliced cuda array.
docs/source/release-notes.rst:ROCm Enhancements:
docs/source/release-notes.rst:* PR #3023: Support for AMDGCN/ROCm.
docs/source/release-notes.rst:* PR #3125: Skip cudasim only tests
docs/source/release-notes.rst:* PR #3216: Fix libcuda.so loading in some container setup
docs/source/release-notes.rst:* PR #3266: Fix cuda pointer ownership problem with user/externally allocated pointer
docs/source/release-notes.rst:* For CUDA GPU support, we added a `__cuda_array_interface__` following the
docs/source/release-notes.rst:  test out the concept and be able to use a CuPy GPU array.
docs/source/release-notes.rst:CUDA Enhancements:
docs/source/release-notes.rst:* PR #2860: __cuda_array_interface__
docs/source/release-notes.rst:* PR #2910: More CUDA intrinsics
docs/source/release-notes.rst:* PR #3037: Add CUDA IPC support on non-peer-accessible devices
docs/source/release-notes.rst:* PR #3048: [WIP] Fix cuda tests failure on buildfarm
docs/source/release-notes.rst:* PR #3062: Fix cuda.In freeing devary before the kernel launch
docs/source/release-notes.rst:* PR #2967: Fix typo in CUDA kernel layout example.
docs/source/release-notes.rst: * CUDA 8.0 is now the minimum supported CUDA version.
docs/source/release-notes.rst: * The CUDA functionality has gained support for a larger selection of bit
docs/source/release-notes.rst:* PR #2895: Move to CUDA 8.0
docs/source/release-notes.rst:* PR #2842: Fix issue with test discovery and broken CUDA drivers.
docs/source/release-notes.rst:CUDA enhancements:
docs/source/release-notes.rst:* PR #2734: More Constants From cuda.h
docs/source/release-notes.rst:* PR #2778: Add More Device Array API Functions to CUDA Simulator
docs/source/release-notes.rst:* PR #2824: Add CUDA Primitives for Population Count
docs/source/release-notes.rst:* PR #2867: Full support for CUDA device attributes
docs/source/release-notes.rst:CUDA fixes:
docs/source/release-notes.rst:enhancements went into the CUDA implementation and ParallelAccelerator gained
docs/source/release-notes.rst:* PR #2722: Add docs on numpy support in cuda
docs/source/release-notes.rst:CUDA enhancements:
docs/source/release-notes.rst:* PR #2665: CUDA DeviceNDArray: Support numpy tranpose API
docs/source/release-notes.rst:CUDA fixes:
docs/source/release-notes.rst:* PR #2667: Fix CUDA DeviceNDArray slicing
docs/source/release-notes.rst:* PR #2686: Fix #2663: incorrect offset when indexing cuda array.
docs/source/release-notes.rst:* PR #2707: Fix regression: cuda test submodules not loading properly in
docs/source/release-notes.rst:* PR #2720: A quick testsuite fix to not run the new cuda testcase in the
docs/source/release-notes.rst:* PR #2652: Add support for CUDA 9.
docs/source/release-notes.rst:CUDA support fixes:
docs/source/release-notes.rst:* PR #2523: Fix invalid cuda context in memory transfer calls in another thread
docs/source/release-notes.rst:* PR #2575: Use CPU to initialize xoroshiro states for GPU RNG. Fixes #2573
docs/source/release-notes.rst:* PR #2581: Fix cuda gufunc mishandling of scalar arg as array and out argument
docs/source/release-notes.rst:CUDA support fixes:
docs/source/release-notes.rst:* PR #2504: Enable CUDA toolkit version testing
docs/source/release-notes.rst:* PR #2511: Fix Windows 64 bit CUDA tests.
docs/source/release-notes.rst:and closure support, support for Numpy 1.13 and a new, faster, CUDA reduction
docs/source/release-notes.rst:CUDA support enhancements:
docs/source/release-notes.rst:* PR #2377: New GPU reduction algorithm
docs/source/release-notes.rst:CUDA support fixes:
docs/source/release-notes.rst:* PR #2397: Fix #2393, always set alignment of cuda static memory regions
docs/source/release-notes.rst:There are also several enhancements to the CUDA GPU support:
docs/source/release-notes.rst:* A GPU random number generator based on `xoroshiro128+ algorithm <http://xoroshiro.di.unimi.it/>`_ is added.
docs/source/release-notes.rst:  See details and examples in :ref:`documentation <cuda-random>`.
docs/source/release-notes.rst:* ``@cuda.jit`` CUDA kernels can now call ``@jit`` and ``@njit``
docs/source/release-notes.rst:  CPU functions and they will automatically be compiled as CUDA device
docs/source/release-notes.rst:* CUDA IPC memory API is exposed for sharing memory between proceses.
docs/source/release-notes.rst:  See usage details in :ref:`documentation <cuda-ipc-memory>`.
docs/source/release-notes.rst:CUDA support enhancements:
docs/source/release-notes.rst:* PR #2023: Supports CUDA IPC for device array
docs/source/release-notes.rst:* PR #2343, Issue #2335: Allow CPU jit decorated function to be used as cuda device function
docs/source/release-notes.rst:* PR #2347: Add random number generator support for CUDA device code
docs/source/release-notes.rst:* PR #2308: Add details to error message on why cuda support is disabled.
docs/source/release-notes.rst:* PR #2331: Fix a bug in the GPU array indexing
docs/source/release-notes.rst:The CUDA backend also gained limited debugging support so that source locations
docs/source/release-notes.rst:* PR #2278: Add CUDA atomic.{max, min, compare_and_swap}
docs/source/release-notes.rst:* PR #2271: Adopt itanium C++-style mangling for CPU and CUDA targets
docs/source/release-notes.rst:* PR #2272: Fix breakage to cuda7.5
docs/source/release-notes.rst:* PR #2269: Fix caching of copy_strides kernel in cuda.reduce
docs/source/release-notes.rst:* PR #2156, Issue #2155: Fix divmod, floordiv segfault on CUDA.
docs/source/release-notes.rst:This release depends on llvmlite 0.14.0 and supports CUDA 8 but it is not
docs/source/release-notes.rst:* PR #2052: Add logging to the CUDA driver.
docs/source/release-notes.rst:* PR #2046: Improving CUDA memory management by deferring deallocations
docs/source/release-notes.rst:* PR #2040: Switch the CUDA driver implementation to use CUDA's
docs/source/release-notes.rst:* PR #2039: Reduce fork() detection overhead in CUDA.
docs/source/release-notes.rst:  to a CUDA kernel.
docs/source/release-notes.rst:* PR #1823: Support ``compute_50`` in CUDA backend.
docs/source/release-notes.rst:* PR #1963: Make CUDA print() atomic.
docs/source/release-notes.rst:* Issue #1837: Fix CUDA simulator issues with device function.
docs/source/release-notes.rst:* Issue #1800: Add erf(), erfc(), gamma() and lgamma() to CUDA targets.
docs/source/release-notes.rst:* PR #1752: Make CUDA features work in dask, distributed and Spark.
docs/source/release-notes.rst:data between the CPU and the GPU.
docs/source/release-notes.rst:  multiple address spaces (host & GPU).
docs/source/release-notes.rst:* PR #1732: Fix tuple getitem regression for CUDA target.
docs/source/release-notes.rst:* Issue #1645: CUDA ufuncs were broken in 0.23.0.
docs/source/release-notes.rst:* Issue #1587: Make CudaAPIError picklable
docs/source/release-notes.rst:* Issue #1538: Fix array broadcasting in CUDA gufuncs
docs/source/release-notes.rst:* PR #1521: Fix cuda.test()
docs/source/release-notes.rst:* PR #1409: Support explicit CUDA memory fences
docs/source/release-notes.rst:* PR #1416: Add support for vectorize() and guvectorize() with CUDA,
docs/source/release-notes.rst:* PR #1415: Add functions to estimate the occupancy of a CUDA kernel
docs/source/release-notes.rst:* PR #1400: Add the cuda.reduce() decorator originally provided in NumbaPro
docs/source/release-notes.rst:which allows memory to be shared directly between the CPU and the GPU.
docs/source/release-notes.rst:* PR #1391: Implement print() for CUDA code
docs/source/release-notes.rst:* PR #1371: Support array.view() in CUDA mode
docs/source/release-notes.rst:* PR #1321: Document features supported with CUDA
docs/source/release-notes.rst:* Issue #1385: Allow CUDA local arrays to be declared anywhere in a function
docs/source/release-notes.rst:This release updates Numba to use LLVM 3.6 and CUDA 7 for CUDA support.
docs/source/release-notes.rst:Following the platform deprecation in CUDA 7, Numba's CUDA feature is no
docs/source/release-notes.rst:* PR #1252: Support cmath module in CUDA
docs/source/release-notes.rst:* Issue #1164: Avoid warnings from CUDA context at shutdown
docs/source/release-notes.rst:   for 32-bit platforms (Win/Mac/Linux) with the CUDA compiler target are
docs/source/release-notes.rst:* Issue #1127: Add a CUDA simulator running on the CPU, enabled with the
docs/source/release-notes.rst:  NUMBA_ENABLE_CUDASIM environment variable.
docs/source/release-notes.rst:* Issue #1074: Fixes CUDA support on Windows machine due to NVVM API mismatch
docs/source/release-notes.rst:* Issue #979: Add cuda.atomic.max().
docs/source/release-notes.rst:* Issue #1010: Simpler and faster CUDA argument marshalling thanks to a
docs/source/release-notes.rst:  methods for CUDA kernels.
docs/source/release-notes.rst:* Issue #1029: Support Numpy structured arrays with CUDA as well.
docs/source/release-notes.rst:* Issue #1048: Allow calling Numpy scalar constructors from CUDA functions.
docs/source/release-notes.rst:* Issue #1017: Update instructions for CUDA in the README.
docs/source/release-notes.rst:* Issue #1008: Generate shorter LLVM type names to avoid segfaults with CUDA.
docs/source/release-notes.rst:* Issue #1053: Fix the size attribute of CUDA shared arrays.
docs/source/release-notes.rst:* Issue #863: CUDA kernels can now infer the types of their arguments
docs/source/release-notes.rst:* Issue #955: Add support for 3D CUDA grids and thread blocks.
docs/source/release-notes.rst:* Issue #889: Fix ``NUMBA_DUMP_ASSEMBLY`` for the CUDA backend.
docs/source/release-notes.rst:* Issue #431: Allow overloading of cuda device function.
docs/source/release-notes.rst:* CUDA JIT functions can be returned by factory functions with variables in
docs/source/release-notes.rst:* Allow the shape of a 1D ``cuda.shared.array`` and ``cuda.local.array`` to be
docs/source/release-notes.rst:  used when transferring CUDA arrays.
docs/source/release-notes.rst:* Support for Numpy record arrays on the GPU. (Note: Improper alignment of dtype
docs/source/release-notes.rst:* Slices on GPU device arrays.
docs/source/release-notes.rst:* GPU objects can be used as Python context managers to select the active
docs/source/release-notes.rst:* GPU device arrays can be bound to a CUDA stream.  All subsequent operations
docs/source/release-notes.rst:* Fixed a problem with selecting CUDA devices in multithreaded programs on
docs/source/release-notes.rst:* Support for NVIDIA compute capability 5.0 devices (such as the GTX 750)
docs/source/release-notes.rst:* Importing Numba will no longer throw an exception if the CUDA driver is
docs/source/release-notes.rst:* CUDA driver is lazily initialized
docs/source/release-notes.rst:* Added cuda.gridsize
docs/source/release-notes.rst:* Initial support for CUDA array slicing
docs/source/release-notes.rst:* Indirectly fixes numbapro when the system has a incompatible CUDA driver
docs/source/release-notes.rst:* Fix numba.cuda.detect
docs/source/release-notes.rst:* Opensourcing NumbaPro CUDA python support in `numba.cuda`
docs/source/cuda/cooperative_groups.rst:<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tesla-compute-cluster-mode-for-windows>`_.
docs/source/cuda/cooperative_groups.rst:<numba.cuda.cg.this_grid>` function:
docs/source/cuda/cooperative_groups.rst:   g = cuda.cg.this_grid()
docs/source/cuda/cooperative_groups.rst:<numba.cuda.cg.GridGroup.sync>` method of the grid group:
docs/source/cuda/cooperative_groups.rst:Unlike the CUDA C/C++ API, a cooperative launch is invoked using the same syntax
docs/source/cuda/cooperative_groups.rst:.. automethod:: numba.cuda.dispatcher._Kernel.max_cooperative_grid_blocks
docs/source/cuda/cooperative_groups.rst:cooperative launch will result in a ``CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE``
docs/source/cuda/cooperative_groups.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_cg.py
docs/source/cuda/cooperative_groups.rst:   :caption: from ``test_grid_sync`` of ``numba/cuda/tests/doc_example/test_cg.py``
docs/source/cuda/cooperative_groups.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_cg.py
docs/source/cuda/cooperative_groups.rst:   :caption: from ``test_grid_sync`` of ``numba/cuda/tests/doc_example/test_cg.py``
docs/source/cuda/cooperative_groups.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_cg.py
docs/source/cuda/cooperative_groups.rst:   :caption: from ``test_grid_sync`` of ``numba/cuda/tests/doc_example/test_cg.py``
docs/source/cuda/cooperative_groups.rst:   # 1152 (e.g. on Quadro RTX 8000 with Numba 0.52.1 and CUDA 11.0)
docs/source/cuda/examples.rst:.. _cuda-vecadd:
docs/source/cuda/examples.rst:it is a warmup for learning how to write GPU kernels using Numba. We'll begin
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_vecadd.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_vecadd`` in ``numba/cuda/tests/doc_examples/test_vecadd.py``
docs/source/cuda/examples.rst:CUDA kernel specialized for them.
docs/source/cuda/examples.rst:arrays passed in as parameters (this is similar to the requirement that CUDA
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_vecadd.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_vecadd`` in ``numba/cuda/tests/doc_examples/test_vecadd.py``
docs/source/cuda/examples.rst::func:`cuda.to_device() <numba.cuda.to_device>` can be used create device-side
docs/source/cuda/examples.rst:copies of arrays.  :func:`cuda.device_array_like()
docs/source/cuda/examples.rst:<numba.cuda.device_array_like>` creates an uninitialized array of the same shape
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_vecadd.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_vecadd`` in ``numba/cuda/tests/doc_examples/test_vecadd.py``
docs/source/cuda/examples.rst:A call to :meth:`forall() <numba.cuda.dispatcher.Dispatcher.forall>` generates
docs/source/cuda/examples.rst::ref:`cuda-kernel-invocation`) for a given data size and is often the simplest
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_vecadd.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_vecadd`` in ``numba/cuda/tests/doc_examples/test_vecadd.py``
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_vecadd.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_vecadd`` in ``numba/cuda/tests/doc_examples/test_vecadd.py``
docs/source/cuda/examples.rst:.. _cuda-laplace:
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_laplace.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_laplace`` in ``numba/cuda/tests/doc_examples/test_laplace.py``
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_laplace.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_laplace`` in ``numba/cuda/tests/doc_examples/test_laplace.py``
docs/source/cuda/examples.rst::func:`numba.cuda.cg.this_grid() <numba.cuda.cg.this_grid>` for details.
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_laplace.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_laplace`` in ``numba/cuda/tests/doc_examples/test_laplace.py``
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_laplace.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_laplace`` in ``numba/cuda/tests/doc_examples/test_laplace.py``
docs/source/cuda/examples.rst:.. _cuda_reduction_shared:
docs/source/cuda/examples.rst:Numba exposes many CUDA features, including :ref:`shared memory
docs/source/cuda/examples.rst:<cuda-shared-memory>`. To demonstrate shared memory, let's reimplement a
docs/source/cuda/examples.rst:famous CUDA solution for summing a vector which works by "folding" the data up
docs/source/cuda/examples.rst:using Numba - see :ref:`cuda_montecarlo` for an example.
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_reduction.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_reduction`` in ``numba/cuda/tests/doc_examples/test_reduction.py``
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_reduction.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_reduction`` in ``numba/cuda/tests/doc_examples/test_reduction.py``
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_reduction.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_reduction`` in ``numba/cuda/tests/doc_examples/test_reduction.py``
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_reduction.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_reduction`` in ``numba/cuda/tests/doc_examples/test_reduction.py``
docs/source/cuda/examples.rst:.. _cuda_sessionization:
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_sessionize.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_sessionize`` in ``numba/cuda/tests/doc_examples/test_sessionize.py``
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_sessionize.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_sessionize`` in ``numba/cuda/tests/doc_examples/test_sessionize.py``
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_sessionize.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_sessionize`` in ``numba/cuda/tests/doc_examples/test_sessionize.py``
docs/source/cuda/examples.rst:.. _cuda_reuse_function:
docs/source/cuda/examples.rst:JIT Function CPU-GPU Compatibility
docs/source/cuda/examples.rst:it available for use inside CUDA kernels. This can be very useful for users that are migrating workflows from CPU to GPU as 
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_cpu_gpu_compat`` in ``numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py``
docs/source/cuda/examples.rst:   :start-after: ex_cpu_gpu_compat.define.begin
docs/source/cuda/examples.rst:   :end-before: ex_cpu_gpu_compat.define.end
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_cpu_gpu_compat`` in ``numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py``
docs/source/cuda/examples.rst:   :start-after: ex_cpu_gpu_compat.cpurun.begin
docs/source/cuda/examples.rst:   :end-before: ex_cpu_gpu_compat.cpurun.end
docs/source/cuda/examples.rst:It can also be directly reused threadwise inside a GPU kernel. For example one may 
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_cpu_gpu_compat`` in ``numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py``
docs/source/cuda/examples.rst:   :start-after: ex_cpu_gpu_compat.allocate.begin
docs/source/cuda/examples.rst:   :end-before: ex_cpu_gpu_compat.allocate.end
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_cpu_gpu_compat`` in ``numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py``
docs/source/cuda/examples.rst:   :start-after: ex_cpu_gpu_compat.usegpu.begin
docs/source/cuda/examples.rst:   :end-before: ex_cpu_gpu_compat.usegpu.end
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_cpu_gpu_compat`` in ``numba/cuda/tests/doc_examples/test_cpu_gpu_compat.py``
docs/source/cuda/examples.rst:   :start-after: ex_cpu_gpu_compat.launch.begin
docs/source/cuda/examples.rst:   :end-before: ex_cpu_gpu_compat.launch.end
docs/source/cuda/examples.rst:.. _cuda_montecarlo:
docs/source/cuda/examples.rst:random numbers on the GPU. A detailed description of the mathematical mechanics of Monte Carlo integration
docs/source/cuda/examples.rst::func:`cuda.reduce() <numba.cuda.Reduce>` API.
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_montecarlo.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_montecarlo`` in ``numba/cuda/tests/doc_examples/test_montecarlo.py``
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_montecarlo.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_montecarlo`` in ``numba/cuda/tests/doc_examples/test_montecarlo.py``
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_montecarlo.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_montecarlo`` in ``numba/cuda/tests/doc_examples/test_montecarlo.py``
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_montecarlo.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_montecarlo`` in ``numba/cuda/tests/doc_examples/test_montecarlo.py``
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_montecarlo.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_montecarlo`` in ``numba/cuda/tests/doc_examples/test_montecarlo.py``
docs/source/cuda/examples.rst:.. _cuda-matmul:
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_matmul.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_matmul`` in ``numba/cuda/tests/doc_examples/test_matmul.py``
docs/source/cuda/examples.rst:Here is a naïve implementation of matrix multiplication using a CUDA kernel:
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_matmul.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_matmul`` in ``numba/cuda/tests/doc_examples/test_matmul.py``
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_matmul.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_matmul`` in ``numba/cuda/tests/doc_examples/test_matmul.py``
docs/source/cuda/examples.rst:device memory.  CUDA provides a fast :ref:`shared memory <cuda-shared-memory>`
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_matmul.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_matmul`` in ``numba/cuda/tests/doc_examples/test_matmul.py``
docs/source/cuda/examples.rst::func:`~numba.cuda.syncthreads` to wait until all threads have finished
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_matmul.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_matmul`` in ``numba/cuda/tests/doc_examples/test_matmul.py``
docs/source/cuda/examples.rst:This passes a :ref:`CUDA memory check test <debugging-cuda-python-code>`, which
docs/source/cuda/examples.rst:.. note:: For high performance matrix multiplication in CUDA, see also the `CuPy implementation <https://docs.cupy.dev/en/stable/reference/generated/cupy.matmul.html>`_.
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_matmul.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_matmul`` in ``numba/cuda/tests/doc_examples/test_matmul.py``
docs/source/cuda/examples.rst:.. _cuda_ufunc_call_example:
docs/source/cuda/examples.rst:UFuncs supported in the CUDA target (see :ref:`cuda_numpy_support`) can be
docs/source/cuda/examples.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_ufunc.py
docs/source/cuda/examples.rst:   :caption: from ``test_ex_cuda_ufunc_call`` in ``numba/cuda/tests/doc_examples/test_ufunc.py``
docs/source/cuda/examples.rst:   :start-after: ex_cuda_ufunc.begin
docs/source/cuda/examples.rst:   :end-before: ex_cuda_ufunc.end
docs/source/cuda/minor_version_compatibility.rst:CUDA Minor Version Compatibility
docs/source/cuda/minor_version_compatibility.rst:CUDA `Minor Version Compatibility
docs/source/cuda/minor_version_compatibility.rst:<https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility>`_
docs/source/cuda/minor_version_compatibility.rst:(MVC) enables the use of a newer CUDA Toolkit version than the CUDA version
docs/source/cuda/minor_version_compatibility.rst:major version. For example, use of CUDA Toolkit 11.5 with CUDA driver 450 (CUDA
docs/source/cuda/minor_version_compatibility.rst:Numba supports MVC for CUDA 12 on Linux using the external ``pynvjitlink``
docs/source/cuda/minor_version_compatibility.rst:Numba supports MVC for CUDA 11 on Linux using the external ``cubinlinker`` and
docs/source/cuda/minor_version_compatibility.rst:CUDA 12
docs/source/cuda/minor_version_compatibility.rst:To install with pip, use the NVIDIA package index:
docs/source/cuda/minor_version_compatibility.rst:   pip install --extra-index-url https://pypi.nvidia.com pynvjitlink-cu12
docs/source/cuda/minor_version_compatibility.rst:CUDA 11
docs/source/cuda/minor_version_compatibility.rst:To install with pip, use the NVIDIA package index:
docs/source/cuda/minor_version_compatibility.rst:   pip install --extra-index-url https://pypi.nvidia.com ptxcompiler-cu11 cubinlinker-cu11
docs/source/cuda/minor_version_compatibility.rst:   export NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY=1
docs/source/cuda/minor_version_compatibility.rst:or by setting a configuration variable prior to using any CUDA functionality in
docs/source/cuda/minor_version_compatibility.rst:   config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = True
docs/source/cuda/minor_version_compatibility.rst:- The `CUDA Compatibility Guide
docs/source/cuda/minor_version_compatibility.rst:  <https://docs.nvidia.com/deploy/cuda-compatibility/index.html>`_.
docs/source/cuda/cudapysupported.rst:Supported Python features in CUDA Python
docs/source/cuda/cudapysupported.rst:This page lists the Python features supported in the CUDA Python.  This includes
docs/source/cuda/cudapysupported.rst:all kernel and device functions compiled with ``@cuda.jit`` and other higher
docs/source/cuda/cudapysupported.rst:level Numba decorators that targets the CUDA GPU.
docs/source/cuda/cudapysupported.rst:CUDA Python maps directly to the *single-instruction multiple-thread*
docs/source/cuda/cudapysupported.rst:execution (SIMT) model of CUDA.  Each instruction is implicitly
docs/source/cuda/cudapysupported.rst:`CUDA Programming Guide
docs/source/cuda/cudapysupported.rst:<http://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model>`_.
docs/source/cuda/cudapysupported.rst:By default, CUDA Python kernels execute with the NumPy error model. In this
docs/source/cuda/cudapysupported.rst::func:`@cuda.jit <numba.cuda.jit>` decorator), the Python error model is used.
docs/source/cuda/cudapysupported.rst:  :func:`@cuda.jit <numba.cuda.jit>` decorator. This is similar to the behavior
docs/source/cuda/cudapysupported.rst:  of the ``assert`` keyword in CUDA C/C++, which is ignored unless compiling
docs/source/cuda/cudapysupported.rst:kernel launch, it is necessary to call :func:`numba.cuda.synchronize`. Eliding
docs/source/cuda/cudapysupported.rst:This is due to a general limitation in CUDA printing, as outlined in the
docs/source/cuda/cudapysupported.rst:<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#limitations>`_
docs/source/cuda/cudapysupported.rst:in the CUDA C++ Programming Guide.
docs/source/cuda/cudapysupported.rst:   @cuda.jit("int64(int64)", device=True)
docs/source/cuda/cudapysupported.rst:   @cuda.jit
docs/source/cuda/cudapysupported.rst:   The call stack in CUDA is typically quite limited in size, so it is easier
docs/source/cuda/cudapysupported.rst:   to overflow it with recursive calls on CUDA devices than it is on CPUs.
docs/source/cuda/cudapysupported.rst:   <https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html>`_,
docs/source/cuda/cudapysupported.rst:.. _cuda-built-in-types:
docs/source/cuda/cudapysupported.rst:with CUDA 11.2 onwards.
docs/source/cuda/cudapysupported.rst:.. _cuda_numpy_support:
docs/source/cuda/cudapysupported.rst:Due to the CUDA programming model, dynamic memory allocation inside a kernel is
docs/source/cuda/cudapysupported.rst:  positional argument (see :ref:`cuda_ufunc_call_example`). Note that ufuncs
docs/source/cuda/cudapysupported.rst:functions (see the :ref:`CUDA FFI documentation <cuda_ffi>`).
docs/source/cuda/faq.rst:.. _cudafaq:
docs/source/cuda/faq.rst:CUDA Frequently Asked Questions
docs/source/cuda/faq.rst:When using the ``nvprof`` tool to profile Numba jitted code for the CUDA
docs/source/cuda/faq.rst:exit, see the `NVIDIA CUDA documentation
docs/source/cuda/faq.rst:<http://docs.nvidia.com/cuda/profiler-users-guide/#flush-profile-data>`_ for
docs/source/cuda/faq.rst:details. To fix this simply add a call to ``numba.cuda.profile_stop()`` prior
docs/source/cuda/faq.rst:For more on CUDA profiling support in Numba, see :ref:`cuda-profiling`.
docs/source/cuda/random.rst:.. _cuda-random:
docs/source/cuda/random.rst:the GPU.  Due to technical issues with how NVIDIA implemented cuRAND, however,
docs/source/cuda/random.rst:Numba's GPU random number generator is not based on cuRAND.  Instead, Numba's
docs/source/cuda/random.rst:GPU RNG is an implementation of the `xoroshiro128+ algorithm
docs/source/cuda/random.rst:When using any RNG on the GPU, it is important to make sure that each thread
docs/source/cuda/random.rst:sequences.  The  numba.cuda.random module provides a host function to do this,
docs/source/cuda/random.rst:as well as CUDA device functions to obtain uniformly or normally distributed
docs/source/cuda/random.rst:.. automodule:: numba.cuda.random
docs/source/cuda/random.rst:    from numba import cuda
docs/source/cuda/random.rst:    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
docs/source/cuda/random.rst:    @cuda.jit
docs/source/cuda/random.rst:        thread_id = cuda.grid(1)
docs/source/cuda/random.rst:states.  This would take a long time to initialize and poorly utilize the GPU.
docs/source/cuda/random.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_random.py
docs/source/cuda/random.rst:   :caption: from ``test_ex_3d_grid of ``numba/cuda/tests/doc_example/test_random.py``
docs/source/cuda/ipc.rst:Sharing CUDA Memory
docs/source/cuda/ipc.rst:.. _cuda-ipc-memory:
docs/source/cuda/ipc.rst:Sharing between processes is implemented using the Legacy CUDA IPC API
docs/source/cuda/ipc.rst:the CUDA IPC API.  To do so, use the ``.get_ipc_handle()`` method on the device
docs/source/cuda/ipc.rst:.. automethod:: numba.cuda.cudadrv.devicearray.DeviceNDArray.get_ipc_handle
docs/source/cuda/ipc.rst:.. autoclass:: numba.cuda.cudadrv.devicearray.IpcArrayHandle
docs/source/cuda/ipc.rst:.. automethod:: numba.cuda.open_ipc_array
docs/source/cuda/cuda_compilation.rst:.. _cuda_compilation:
docs/source/cuda/cuda_compilation.rst:incorporated into CUDA code written in other languages (e.g. C/C++).  It is
docs/source/cuda/cuda_compilation.rst:The compilation API can be used without a GPU present, as it uses no driver
docs/source/cuda/cuda_compilation.rst:functions and avoids initializing CUDA in the process. It is invoked through
docs/source/cuda/cuda_compilation.rst:.. autofunction:: numba.cuda.compile
docs/source/cuda/cuda_compilation.rst:.. autofunction:: numba.cuda.compile_for_current_device
docs/source/cuda/cuda_compilation.rst:.. autofunction:: numba.cuda.compile_ptx
docs/source/cuda/cuda_compilation.rst:.. autofunction:: numba.cuda.compile_ptx_for_current_device
docs/source/cuda/cuda_compilation.rst:.. _cuda-using-the-c-abi:
docs/source/cuda/cuda_compilation.rst:    ptx, resty = cuda.compile_ptx(add, int32(int32, int32), device=True)
docs/source/cuda/cuda_compilation.rst:   ptx, resty = cuda.compile_ptx(add, int32(int32, int32), device=True, abi="c")
docs/source/cuda/cuda_compilation.rst:   ptx, resty = cuda.compile_ptx(add, float32(float32, float32), device=True,
docs/source/cuda/device-functions.rst:CUDA device functions can only be invoked from within the device (by a kernel
docs/source/cuda/device-functions.rst:    from numba import cuda
docs/source/cuda/device-functions.rst:    @cuda.jit(device=True)
docs/source/cuda/memory.rst:.. _cuda-device-memory:
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.device_array
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.device_array_like
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.to_device
docs/source/cuda/memory.rst::ref:`cuda array interface <cuda-array-interface>`.  These objects also can be
docs/source/cuda/memory.rst:manually converted into a Numba device array by creating a view of the GPU
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.as_cuda_array
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.is_cuda_array
docs/source/cuda/memory.rst:called in host code, not within CUDA-jitted functions.
docs/source/cuda/memory.rst:.. autoclass:: numba.cuda.cudadrv.devicearray.DeviceNDArray
docs/source/cuda/memory.rst:.. note:: DeviceNDArray defines the :ref:`cuda array interface <cuda-array-interface>`.
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.pinned
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.pinned_array
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.pinned_array_like
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.mapped
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.mapped_array
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.mapped_array_like
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.managed_array
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.stream
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.default_stream
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.legacy_default_stream
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.per_thread_default_stream
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.external_stream
docs/source/cuda/memory.rst:CUDA streams have the following methods:
docs/source/cuda/memory.rst:.. autoclass:: numba.cuda.cudadrv.driver.Stream
docs/source/cuda/memory.rst:.. _cuda-shared-memory:
docs/source/cuda/memory.rst:.. function:: numba.cuda.shared.array(shape, type)
docs/source/cuda/memory.rst:.. function:: numba.cuda.syncthreads()
docs/source/cuda/memory.rst:   :ref:`Matrix multiplication example <cuda-matmul>`.
docs/source/cuda/memory.rst:   @cuda.jit
docs/source/cuda/memory.rst:      dyn_arr = cuda.shared.array(0, dtype=np.float32)
docs/source/cuda/memory.rst:   from numba import cuda
docs/source/cuda/memory.rst:   @cuda.jit
docs/source/cuda/memory.rst:      f32_arr = cuda.shared.array(0, dtype=np.float32)
docs/source/cuda/memory.rst:      i32_arr = cuda.shared.array(0, dtype=np.int32)
docs/source/cuda/memory.rst:   cuda.synchronize()
docs/source/cuda/memory.rst:   from numba import cuda
docs/source/cuda/memory.rst:   @cuda.jit
docs/source/cuda/memory.rst:      f32_arr = cuda.shared.array(0, dtype=np.float32)
docs/source/cuda/memory.rst:      i32_arr = cuda.shared.array(0, dtype=np.int32)[1:] # 1 int32 = 4 bytes
docs/source/cuda/memory.rst:   cuda.synchronize()
docs/source/cuda/memory.rst:.. _cuda-local-memory:
docs/source/cuda/memory.rst:.. function:: numba.cuda.local.array(shape, type)
docs/source/cuda/memory.rst:      <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses>`_
docs/source/cuda/memory.rst:      in the CUDA programming guide.
docs/source/cuda/memory.rst:.. function:: numba.cuda.const.array_like(arr)
docs/source/cuda/memory.rst::ref:`cuda-emm-plugin`), then deallocation behaviour may differ; you may refer to the
docs/source/cuda/memory.rst:Deallocation of all CUDA resources are tracked on a per-context basis.
docs/source/cuda/memory.rst:   Continued deallocation errors can cause critical errors at the CUDA driver
docs/source/cuda/memory.rst:   level.  In some cases, this could mean a segmentation fault in the CUDA
docs/source/cuda/memory.rst:   CUDA driver is able to release all allocated resources by the terminated
docs/source/cuda/memory.rst:  `NUMBA_CUDA_MAX_PENDING_DEALLOCS_COUNT`.  For example,
docs/source/cuda/memory.rst:  `NUMBA_CUDA_MAX_PENDING_DEALLOCS_COUNT=20`, increases the limit to 20.
docs/source/cuda/memory.rst:  `NUMBA_CUDA_MAX_PENDING_DEALLOCS_RATIO`. For example,
docs/source/cuda/memory.rst:  `NUMBA_CUDA_MAX_PENDING_DEALLOCS_RATIO=0.5` sets the limit to 50% of the
docs/source/cuda/memory.rst:.. autofunction:: numba.cuda.defer_cleanup
docs/source/cuda/index.rst:.. _cuda-index:
docs/source/cuda/index.rst:Numba for CUDA GPUs
docs/source/cuda/index.rst:   cudapysupported.rst
docs/source/cuda/index.rst:   cuda_array_interface.rst
docs/source/cuda/index.rst:   cuda_ffi.rst
docs/source/cuda/index.rst:   cuda_compilation.rst
docs/source/cuda/cuda_array_interface.rst:.. _cuda-array-interface:
docs/source/cuda/cuda_array_interface.rst:CUDA Array Interface (Version 3)
docs/source/cuda/cuda_array_interface.rst:The *CUDA Array Interface* (or CAI) is created for interoperability between
docs/source/cuda/cuda_array_interface.rst:different implementations of CUDA array-like objects in various projects. The
docs/source/cuda/cuda_array_interface.rst:The ``__cuda_array_interface__`` attribute returns a dictionary (``dict``)
docs/source/cuda/cuda_array_interface.rst:  ``CU_POINTER_ATTRIBUTE_DEVICE_POINTER`` in the CUDA driver API (or the
docs/source/cuda/cuda_array_interface.rst:  equivalent CUDA Runtime API) to retrieve a device pointer that
docs/source/cuda/cuda_array_interface.rst:- **mask**: ``None`` or object exposing the ``__cuda_array_interface__``
docs/source/cuda/cuda_array_interface.rst:  .. note:: Numba does not currently support working with masked CUDA arrays
docs/source/cuda/cuda_array_interface.rst:            to a GPU function.
docs/source/cuda/cuda_array_interface.rst:  - Any other integer: a ``cudaStream_t`` represented as a Python integer.
docs/source/cuda/cuda_array_interface.rst:  :ref:`cuda-array-interface-synchronization` section below for further details.
docs/source/cuda/cuda_array_interface.rst:.. _cuda-array-interface-synchronization:
docs/source/cuda/cuda_array_interface.rst:- *Producer*: The library / object on which ``__cuda_array_interface__`` is
docs/source/cuda/cuda_array_interface.rst:  ``__cuda_array_interface__`` of the Producer.
docs/source/cuda/cuda_array_interface.rst:   from numba import cuda
docs/source/cuda/cuda_array_interface.rst:   @cuda.jit
docs/source/cuda/cuda_array_interface.rst:       start = cuda.grid(1)
docs/source/cuda/cuda_array_interface.rst:       stride = cuda.gridsize(1)
docs/source/cuda/cuda_array_interface.rst:     are required to understand the details of the CUDA Array Interface, and
docs/source/cuda/cuda_array_interface.rst:     third-party libraries oblivious to the CUDA Array Interface.
docs/source/cuda/cuda_array_interface.rst:  <numba.cuda.cudadrv.devicearray.DeviceNDArray>` created from an array-like
docs/source/cuda/cuda_array_interface.rst:- When Numba acts as a Producer (when the ``__cuda_array_interface__`` property
docs/source/cuda/cuda_array_interface.rst:  of a Numba CUDA Array is accessed): If the exported CUDA Array has a
docs/source/cuda/cuda_array_interface.rst:          <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#default-stream>`_
docs/source/cuda/cuda_array_interface.rst:          in normal CUDA terminology.
docs/source/cuda/cuda_array_interface.rst:environment variable ``NUMBA_CUDA_ARRAY_INTERFACE_SYNC`` or the config variable
docs/source/cuda/cuda_array_interface.rst:``CUDA_ARRAY_INTERFACE_SYNC`` to ``0`` (see :ref:`GPU Support Environment
docs/source/cuda/cuda_array_interface.rst:Variables <numba-envvars-gpu-support>`).  When set, Numba will not synchronize
docs/source/cuda/cuda_array_interface.rst:creating a Numba CUDA Array from an object exporting the CUDA Array Interface
docs/source/cuda/cuda_array_interface.rst:may also be elided by passing ``sync=False`` when creating the Numba CUDA
docs/source/cuda/cuda_array_interface.rst:Array with :func:`numba.cuda.as_cuda_array` or
docs/source/cuda/cuda_array_interface.rst::func:`numba.cuda.from_cuda_array_interface`.
docs/source/cuda/cuda_array_interface.rst:   from numba import cuda, int32, void
docs/source/cuda/cuda_array_interface.rst:   @cuda.jit(void, int32[::1])
docs/source/cuda/cuda_array_interface.rst:       i = cuda.grid(1)
docs/source/cuda/cuda_array_interface.rst:   array_stream = cuda.stream()
docs/source/cuda/cuda_array_interface.rst:   kernel_stream = cuda.stream()
docs/source/cuda/cuda_array_interface.rst:   x = cuda.device_array(N, stream=array_stream)
docs/source/cuda/cuda_array_interface.rst:   evt = cuda.event()
docs/source/cuda/cuda_array_interface.rst:Obtaining the value of the ``__cuda_array_interface__`` property of any object
docs/source/cuda/cuda_array_interface.rst:Like data, CUDA streams also have a finite lifetime. It is therefore required
docs/source/cuda/cuda_array_interface.rst:          ``cuda.default_stream()``, ``cuda.legacy_default_stream()``, or
docs/source/cuda/cuda_array_interface.rst:          ``cuda.per_thread_default_stream()``. Streams not managed by Numba
docs/source/cuda/cuda_array_interface.rst:          are created from an external stream with ``cuda.external_stream()``.
docs/source/cuda/cuda_array_interface.rst:the CUDA Array Interface. Which to use depends on whether the created device
docs/source/cuda/cuda_array_interface.rst:- ``as_cuda_array``: This creates a device array that holds a reference to the
docs/source/cuda/cuda_array_interface.rst:- ``from_cuda_array_interface``: This creates a device array with no reference
docs/source/cuda/cuda_array_interface.rst:.. automethod:: numba.cuda.as_cuda_array
docs/source/cuda/cuda_array_interface.rst:.. automethod:: numba.cuda.from_cuda_array_interface
docs/source/cuda/cuda_array_interface.rst:``cuPointerGetAttribute`` or ``cudaPointerGetAttributes``.  Such information
docs/source/cuda/cuda_array_interface.rst:- the CUDA context that owns the pointer;
docs/source/cuda/cuda_array_interface.rst:Differences with CUDA Array Interface (Version 0)
docs/source/cuda/cuda_array_interface.rst:Version 0 of the CUDA Array Interface did not have the optional **mask**
docs/source/cuda/cuda_array_interface.rst:Differences with CUDA Array Interface (Version 1)
docs/source/cuda/cuda_array_interface.rst:Versions 0 and 1 of the CUDA Array Interface neither clarified the
docs/source/cuda/cuda_array_interface.rst:Differences with CUDA Array Interface (Version 2)
docs/source/cuda/cuda_array_interface.rst:Prior versions of the CUDA Array Interface made no statement about
docs/source/cuda/cuda_array_interface.rst:The following Python libraries have adopted the CUDA Array Interface:
docs/source/cuda/cuda_array_interface.rst:- `PyArrow <https://arrow.apache.org/docs/python/generated/pyarrow.cuda.Context.html#pyarrow.cuda.Context.buffer_from_object>`_
docs/source/cuda/cuda_array_interface.rst:- `mpi4py <https://mpi4py.readthedocs.io/en/latest/overview.html#support-for-cuda-aware-mpi>`_
docs/source/cuda/cuda_array_interface.rst:- `PyCUDA <https://documen.tician.de/pycuda/tutorial.html#interoperability-with-other-libraries-using-the-cuda-array-interface>`_
docs/source/cuda/cuda_array_interface.rst:- `DALI: the NVIDIA Data Loading Library <https://github.com/NVIDIA/DALI>`_ :
docs/source/cuda/cuda_array_interface.rst:    - `TensorGPU objects
docs/source/cuda/cuda_array_interface.rst:      <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/data_types.html#nvidia.dali.backend.TensorGPU>`_
docs/source/cuda/cuda_array_interface.rst:      expose the CUDA Array Interface.
docs/source/cuda/cuda_array_interface.rst:      <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html#nvidia.dali.fn.external_source>`_
docs/source/cuda/cuda_array_interface.rst:      consumes objects exporting the CUDA Array Interface.
docs/source/cuda/fastmath.rst:.. _cuda-fast-math:
docs/source/cuda/fastmath.rst:CUDA Fast Math
docs/source/cuda/fastmath.rst:The CUDA target implements :ref:`fast-math` behavior with two differences.
docs/source/cuda/fastmath.rst:  <numba.cuda.jit>` is limited to the values ``True`` and ``False``.
docs/source/cuda/fastmath.rst:  See the `documentation for nvvmCompileProgram <https://docs.nvidia.com/cuda/libnvvm-api/group__compilation.html#group__compilation_1g76ac1e23f5d0e2240e78be0e63450346>`_ for more details of these optimizations.
docs/source/cuda/fastmath.rst:  - :func:`math.cos`: Implemented using `__nv_fast_cosf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_cosf.html>`_.
docs/source/cuda/fastmath.rst:  - :func:`math.sin`: Implemented using `__nv_fast_sinf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_sinf.html>`_.
docs/source/cuda/fastmath.rst:  - :func:`math.tan`: Implemented using `__nv_fast_tanf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_tanf.html>`_.
docs/source/cuda/fastmath.rst:  - :func:`math.exp`: Implemented using `__nv_fast_expf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_expf.html>`_.
docs/source/cuda/fastmath.rst:  - :func:`math.log2`: Implemented using `__nv_fast_log2f <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_log2f.html>`_.
docs/source/cuda/fastmath.rst:  - :func:`math.log10`: Implemented using `__nv_fast_log10f <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_log10f.html>`_.
docs/source/cuda/fastmath.rst:  - :func:`math.log`: Implemented using `__nv_fast_logf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_logf.html>`_.
docs/source/cuda/fastmath.rst:  - :func:`math.pow`: Implemented using `__nv_fast_powf <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_powf.html>`_.
docs/source/cuda/external-memory.rst:.. _cuda-emm-plugin:
docs/source/cuda/external-memory.rst:The :ref:`CUDA Array Interface <cuda-array-interface>` enables sharing of data
docs/source/cuda/external-memory.rst:between different Python libraries that access CUDA devices. However, each
docs/source/cuda/external-memory.rst:- By default, Numba allocates memory on CUDA devices by interacting with the
docs/source/cuda/external-memory.rst:  CUDA driver API to call functions such as ``cuMemAlloc`` and ``cuMemFree``,
docs/source/cuda/external-memory.rst:When multiple CUDA-aware libraries are used together, it may be preferable for
docs/source/cuda/external-memory.rst:interface facilitates this, by enabling Numba to use another CUDA-aware library
docs/source/cuda/external-memory.rst:However, not all CUDA-aware libraries also support managing host memory, so a
docs/source/cuda/external-memory.rst::ref:`host-only-cuda-memory-manager`).
docs/source/cuda/external-memory.rst:sections, using the :func:`~numba.cuda.defer_cleanup` context manager.
docs/source/cuda/external-memory.rst:compiled object, which is generated from ``@cuda.jit``\ -ted functions). The
docs/source/cuda/external-memory.rst::class:`~numba.cuda.BaseCUDAMemoryManager`. A summary of considerations for the
docs/source/cuda/external-memory.rst:  called when the current CUDA context is the context that owns the EMM Plugin
docs/source/cuda/external-memory.rst:- To support inter-GPU communication, the ``get_ipc_handle`` method should
docs/source/cuda/external-memory.rst:  provide an :class:`~numba.cuda.IpcHandle` for a given
docs/source/cuda/external-memory.rst:  :class:`~numba.cuda.MemoryPointer` instance. This method is part of the EMM
docs/source/cuda/external-memory.rst:.. autoclass:: numba.cuda.BaseCUDAMemoryManager
docs/source/cuda/external-memory.rst:.. _host-only-cuda-memory-manager:
docs/source/cuda/external-memory.rst:The Host-Only CUDA Memory Manager
docs/source/cuda/external-memory.rst::class:`~numba.cuda.HostOnlyCUDAMemoryManager` instead of
docs/source/cuda/external-memory.rst::class:`~numba.cuda.BaseCUDAMemoryManager`. Guidelines for using this class
docs/source/cuda/external-memory.rst:Documentation for the methods of :class:`~numba.cuda.HostOnlyCUDAMemoryManager`
docs/source/cuda/external-memory.rst:.. autoclass:: numba.cuda.HostOnlyCUDAMemoryManager
docs/source/cuda/external-memory.rst:.. autoclass:: numba.cuda.GetIpcHandleMixin
docs/source/cuda/external-memory.rst:- :class:`~numba.cuda.MemoryPointer`: returned from ``memalloc``
docs/source/cuda/external-memory.rst:- :class:`~numba.cuda.MappedMemory`: returned from ``memhostalloc`` or
docs/source/cuda/external-memory.rst:- :class:`~numba.cuda.PinnedMemory`: return from ``memhostalloc`` or ``mempin``
docs/source/cuda/external-memory.rst:.. autoclass:: numba.cuda.MemoryPointer
docs/source/cuda/external-memory.rst:as it is subclassed by :class:`numba.cuda.MappedMemory`:
docs/source/cuda/external-memory.rst:.. autoclass:: numba.cuda.cudadrv.driver.AutoFreePointer
docs/source/cuda/external-memory.rst:.. autoclass:: numba.cuda.MappedMemory
docs/source/cuda/external-memory.rst:.. autoclass:: numba.cuda.PinnedMemory
docs/source/cuda/external-memory.rst::meth:`~numba.cuda.BaseCUDAMemoryManager.get_memory_info` is to provide a
docs/source/cuda/external-memory.rst:.. autoclass:: numba.cuda.MemoryInfo
docs/source/cuda/external-memory.rst:of :meth:`~numba.cuda.BaseCUDAMemoryManager.get_ipc_handle`:
docs/source/cuda/external-memory.rst:.. autoclass:: numba.cuda.IpcHandle
docs/source/cuda/external-memory.rst:  CUDA IPC handle appropriate to the underlying library.
docs/source/cuda/external-memory.rst:  by the CUDA driver or runtime API (e.g. if a pool allocator is in use) then
docs/source/cuda/external-memory.rst:``NUMBA_CUDA_MEMORY_MANAGER``. If this environment variable is set, Numba will
docs/source/cuda/external-memory.rst:   $ NUMBA_CUDA_MEMORY_MANAGER=rmm python -m numba.runtests numba.cuda.tests
docs/source/cuda/external-memory.rst:The :func:`~numba.cuda.set_memory_manager` function can be used to set the
docs/source/cuda/external-memory.rst:.. autofunction:: numba.cuda.set_memory_manager
docs/source/cuda/external-memory.rst:It is recommended that the memory manager is set once prior to using any CUDA
docs/source/cuda/external-memory.rst:* :func:`numba.cuda.close` can be used to destroy contexts after setting the
docs/source/cuda/external-memory.rst:    an exception being raised due to a ``CUDA_ERROR_INVALID_CONTEXT`` or
docs/source/cuda/external-memory.rst:    ``CUDA_ERROR_CONTEXT_IS_DESTROYED`` return code from a Driver API function.
docs/source/cuda/external-memory.rst:          ``@cuda.jit`` prior to context destruction will need to be
docs/source/cuda/external-memory.rst:          from the GPU.
docs/source/cuda/caching.rst:When the ``cache`` keyword argument of the :func:`@cuda.jit <numba.cuda.jit>`
docs/source/cuda/caching.rst:Therefore, on systems that have multiple GPUs with differing compute
docs/source/cuda/caching.rst:For example: if a system has two GPUs, one of compute capability 7.5 and one of
docs/source/cuda/caching.rst:as multi-GPU production systems tend to have identical GPUs within each node.
docs/source/cuda/device-management.rst:For multi-GPU machines, users may want to select which GPU to use.
docs/source/cuda/device-management.rst:By default the CUDA driver selects the fastest GPU as the device 0,
docs/source/cuda/device-management.rst:unless working with systems hosting/offering more than one CUDA-capable GPU.
docs/source/cuda/device-management.rst:If at all required, device selection must be done before any CUDA feature is
docs/source/cuda/device-management.rst:    from numba import cuda
docs/source/cuda/device-management.rst:    cuda.select_device(0)
docs/source/cuda/device-management.rst:    cuda.close()
docs/source/cuda/device-management.rst:    cuda.select_device(1)  # assuming we have 2 GPUs
docs/source/cuda/device-management.rst:.. function:: numba.cuda.select_device(device_id)
docs/source/cuda/device-management.rst:   Create a new CUDA context for the selected *device_id*.  *device_id*
docs/source/cuda/device-management.rst:   is determined by the CUDA libraries).  The context is associated with
docs/source/cuda/device-management.rst:.. function:: numba.cuda.close
docs/source/cuda/device-management.rst:      Compiled functions are associated with the CUDA context.
docs/source/cuda/device-management.rst:      has multiple GPUs.
docs/source/cuda/device-management.rst:The Device List is a list of all the GPUs in the system, and can be indexed to
docs/source/cuda/device-management.rst:obtain a context manager that ensures execution on the selected GPU.
docs/source/cuda/device-management.rst:.. attribute:: numba.cuda.gpus
docs/source/cuda/device-management.rst:.. attribute:: numba.cuda.cudadrv.devices.gpus
docs/source/cuda/device-management.rst::py:data:`numba.cuda.gpus` is an instance of the ``_DeviceList`` class, from
docs/source/cuda/device-management.rst:which the current GPU context can also be retrieved:
docs/source/cuda/device-management.rst:.. autoclass:: numba.cuda.cudadrv.devices._DeviceList
docs/source/cuda/device-management.rst:The UUID of a device (equal to that returned by ``nvidia-smi -L``) is available
docs/source/cuda/device-management.rst:in the :attr:`uuid <numba.cuda.cudadrv.driver.Device.uuid>` attribute of a CUDA
docs/source/cuda/device-management.rst:   dev = cuda.current_context().device
docs/source/cuda/device-management.rst:   # prints e.g. "GPU-e6489c45-5b68-3b03-bab7-0e7c8e809643"
docs/source/cuda/simulator.rst:Debugging CUDA Python with the the CUDA Simulator
docs/source/cuda/simulator.rst:Numba includes a CUDA Simulator that implements most of the semantics in CUDA
docs/source/cuda/simulator.rst:be used to debug CUDA Python code, either by adding print statements to your
docs/source/cuda/simulator.rst:The simulator deliberately allows running non-CUDA code like starting a debugger 
docs/source/cuda/simulator.rst:best to start from code that compiles for the CUDA target, and then move over to
docs/source/cuda/simulator.rst::envvar:`NUMBA_ENABLE_CUDASIM` to 1 prior to importing Numba. CUDA Python code 
docs/source/cuda/simulator.rst:    @cuda.jit
docs/source/cuda/simulator.rst:        x = cuda.threadIdx.x
docs/source/cuda/simulator.rst:        bx = cuda.blockIdx.x
docs/source/cuda/simulator.rst:        bdx = cuda.blockDim.x
docs/source/cuda/simulator.rst:GPU as possible - in particular, the following are supported:
docs/source/cuda/simulator.rst:* Data transfer to and from the GPU - in particular, creating array objects with
docs/source/cuda/simulator.rst:* The driver API implementation of the list of GPU contexts (``cuda.gpus`` and
docs/source/cuda/simulator.rst:  ``cuda.cudadrv.devices.gpus``) is supported, and reports a single GPU context.
docs/source/cuda/simulator.rst:  :meth:`~numba.cuda.dispatcher._Kernel.max_cooperative_grid_blocks` method.
docs/source/cuda/simulator.rst:* Only one GPU is simulated.
docs/source/cuda/simulator.rst:* Multithreaded accesses to a single GPU are not supported, and will result in
docs/source/cuda/simulator.rst:* It is not possible to link PTX code with CUDA Python functions.
docs/source/cuda/simulator.rst:* The :func:`ffs() <numba.cuda.ffs>` function only works correctly for values
docs/source/cuda/simulator.rst:CUDA grid in order to make debugging with the simulator tractable.
docs/source/cuda/intrinsics.rst:Numba provides access to some of the atomic operations supported in CUDA. Those
docs/source/cuda/intrinsics.rst:.. automodule:: numba.cuda
docs/source/cuda/intrinsics.rst:The following code demonstrates the use of :class:`numba.cuda.atomic.max` to
docs/source/cuda/intrinsics.rst:    from numba import cuda
docs/source/cuda/intrinsics.rst:    @cuda.jit
docs/source/cuda/intrinsics.rst:        tid = cuda.threadIdx.x
docs/source/cuda/intrinsics.rst:        bid = cuda.blockIdx.x
docs/source/cuda/intrinsics.rst:        bdim = cuda.blockDim.x
docs/source/cuda/intrinsics.rst:        cuda.atomic.max(result, 0, values[i])
docs/source/cuda/intrinsics.rst:    print(result[0]) # Found using cuda.atomic.max
docs/source/cuda/intrinsics.rst:    @cuda.jit
docs/source/cuda/intrinsics.rst:        i, j, k = cuda.grid(3)
docs/source/cuda/intrinsics.rst:        cuda.atomic.max(result, (0, 1, 2), values[i, j, k])
docs/source/cuda/cuda_ffi.rst:.. _cuda_ffi:
docs/source/cuda/cuda_ffi.rst:Python kernels can call device functions written in other languages. CUDA C/C++,
docs/source/cuda/cuda_ffi.rst:- The device function implementation in a foreign language (e.g. CUDA C).
docs/source/cuda/cuda_ffi.rst:<numba.cuda.declare_device>`:
docs/source/cuda/cuda_ffi.rst:.. autofunction:: numba.cuda.declare_device
docs/source/cuda/cuda_ffi.rst:   mul = cuda.declare_device('mul_f32_f32', 'float32(float32, float32)')
docs/source/cuda/cuda_ffi.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/ffi/functions.cu
docs/source/cuda/cuda_ffi.rst:   :caption: ``numba/cuda/tests/doc_examples/ffi/functions.cu``
docs/source/cuda/cuda_ffi.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_ffi.py
docs/source/cuda/cuda_ffi.rst:   :caption: from ``test_ex_from_buffer`` in ``numba/cuda/tests/doc_examples/test_ffi.py``
docs/source/cuda/cuda_ffi.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_ffi.py
docs/source/cuda/cuda_ffi.rst:   :caption: from ``test_ex_from_buffer`` in ``numba/cuda/tests/doc_examples/test_ffi.py``
docs/source/cuda/cuda_ffi.rst:The ``link`` keyword argument of the :func:`@cuda.jit <numba.cuda.jit>`
docs/source/cuda/cuda_ffi.rst:will be compiled with the `NVIDIA Runtime Compiler (NVRTC)
docs/source/cuda/cuda_ffi.rst:<https://docs.nvidia.com/cuda/nvrtc/index.html>`_ and linked into the kernel as
docs/source/cuda/cuda_ffi.rst:PTX; other files will be passed directly to the CUDA Linker.
docs/source/cuda/cuda_ffi.rst:   @cuda.jit(link=['functions.cu'])
docs/source/cuda/cuda_ffi.rst:       i = cuda.grid(1)
docs/source/cuda/cuda_ffi.rst:Support for compiling and linking of CUDA C/C++ code is provided through the use
docs/source/cuda/cuda_ffi.rst:- It is only available when using the NVIDIA Bindings. See
docs/source/cuda/cuda_ffi.rst:  :envvar:`NUMBA_CUDA_USE_NVIDIA_BINDING`.
docs/source/cuda/cuda_ffi.rst:  NVIDIA CUDA Bindings must be available.
docs/source/cuda/cuda_ffi.rst:- The CUDA include path is assumed by default to be ``/usr/local/cuda/include``
docs/source/cuda/cuda_ffi.rst:  on Linux and ``$env:CUDA_PATH\include`` on Windows. It can be modified using
docs/source/cuda/cuda_ffi.rst:  the environment variable :envvar:`NUMBA_CUDA_INCLUDE_PATH`.
docs/source/cuda/cuda_ffi.rst:- The CUDA include directory will be made available to NVRTC on the include
docs/source/cuda/cuda_ffi.rst:This example demonstrates calling a foreign function written in CUDA C to
docs/source/cuda/cuda_ffi.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/ffi/functions.cu
docs/source/cuda/cuda_ffi.rst:   :caption: ``numba/cuda/tests/doc_examples/ffi/functions.cu``
docs/source/cuda/cuda_ffi.rst:.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_ffi.py
docs/source/cuda/cuda_ffi.rst:   :caption: from ``test_ex_linking_cu`` in ``numba/cuda/tests/doc_examples/test_ffi.py``
docs/source/cuda/bindings.rst:CUDA Bindings
docs/source/cuda/bindings.rst:Numba supports two bindings to the CUDA Driver APIs: its own internal bindings
docs/source/cuda/bindings.rst:based on ctypes, and the official `NVIDIA CUDA Python bindings
docs/source/cuda/bindings.rst:<https://nvidia.github.io/cuda-python/>`_. Functionality is equivalent between
docs/source/cuda/bindings.rst:The internal bindings are used by default. If the NVIDIA bindings are installed,
docs/source/cuda/bindings.rst:``NUMBA_CUDA_USE_NVIDIA_BINDING`` to ``1`` prior to the import of Numba. Once
docs/source/cuda/bindings.rst:the NVIDIA bindings when they are in use. To use PTDS with the NVIDIA bindings,
docs/source/cuda/bindings.rst:set the environment variable ``CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM`` to
docs/source/cuda/bindings.rst::envvar:`NUMBA_CUDA_PER_THREAD_DEFAULT_STREAM`.
docs/source/cuda/bindings.rst:   <https://nvidia.github.io/cuda-python/release/11.6.0-notes.html#default-stream>`_
docs/source/cuda/bindings.rst:   in the NVIDIA Bindings documentation.
docs/source/cuda/bindings.rst:In Numba 0.56, the NVIDIA Bindings will be used by default, if they are
docs/source/cuda/reduction.rst:GPU Reduction
docs/source/cuda/reduction.rst:Writing a reduction algorithm for CUDA GPU can be tricky.  Numba provides a
docs/source/cuda/reduction.rst:    from numba import cuda
docs/source/cuda/reduction.rst:    @cuda.reduce
docs/source/cuda/reduction.rst:    got = sum_reduce(A)   # cuda sum reduction
docs/source/cuda/reduction.rst:    sum_reduce = cuda.reduce(lambda a, b: a + b)
docs/source/cuda/reduction.rst:.. autoclass:: numba.cuda.Reduce
docs/source/cuda/ufunc.rst:CUDA Ufuncs and Generalized Ufuncs
docs/source/cuda/ufunc.rst:This page describes the CUDA ufunc-like object.
docs/source/cuda/ufunc.rst:To support the programming pattern of CUDA programs, CUDA Vectorize and
docs/source/cuda/ufunc.rst:compatible with a regular NumPy ufunc.  The CUDA ufunc adds support for
docs/source/cuda/ufunc.rst:passing intra-device arrays (already on the GPU device) to reduce
docs/source/cuda/ufunc.rst:    from numba import vectorize, cuda
docs/source/cuda/ufunc.rst:               target='cuda')
docs/source/cuda/ufunc.rst:All CUDA ufunc kernels have the ability to call other CUDA device functions::
docs/source/cuda/ufunc.rst:    from numba import vectorize, cuda
docs/source/cuda/ufunc.rst:    @cuda.jit('float32(float32, float32, float32)', device=True, inline=True)
docs/source/cuda/ufunc.rst:    @vectorize(['float32(float32, float32, float32)'], target='cuda')
docs/source/cuda/ufunc.rst:Generalized CUDA ufuncs
docs/source/cuda/ufunc.rst:Generalized ufuncs may be executed on the GPU using CUDA, analogous to
docs/source/cuda/ufunc.rst:the CUDA ufunc functionality.  This may be accomplished as follows::
docs/source/cuda/ufunc.rst:                 '(m,n),(n,p)->(m,p)', target='cuda')
docs/source/cuda/ufunc.rst:    capacity of your GPU.  For example:
docs/source/cuda/ufunc.rst:        from numba import vectorize, cuda
docs/source/cuda/ufunc.rst:                                    target='cuda')(discriminant)
docs/source/cuda/ufunc.rst:        # create a CUDA stream
docs/source/cuda/ufunc.rst:        stream = cuda.stream()
docs/source/cuda/ufunc.rst:            # by using the CUDA stream
docs/source/cuda/ufunc.rst:                dA = cuda.to_device(a, stream)
docs/source/cuda/ufunc.rst:                dB = cuda.to_device(b, stream)
docs/source/cuda/ufunc.rst:                dC = cuda.to_device(c, stream)
docs/source/cuda/ufunc.rst:                dD = cuda.to_device(d, stream, copy=False) # no copying
docs/source/cuda/overview.rst:Numba supports CUDA GPU programming by directly compiling a restricted subset
docs/source/cuda/overview.rst:of Python code into CUDA kernels and device functions following the CUDA
docs/source/cuda/overview.rst:GPU automatically.
docs/source/cuda/overview.rst:Several important terms in the topic of CUDA programming are listed here:
docs/source/cuda/overview.rst:- *device*: the GPU
docs/source/cuda/overview.rst:- *device memory*: onboard memory on a GPU card
docs/source/cuda/overview.rst:- *kernels*: a GPU function launched by the host and executed on the device
docs/source/cuda/overview.rst:- *device function*: a GPU function executed on the device which can only be
docs/source/cuda/overview.rst:Most CUDA programming facilities exposed by Numba map directly to the CUDA
docs/source/cuda/overview.rst:C language offered by NVidia.  Therefore, it is recommended you read the
docs/source/cuda/overview.rst:official `CUDA C programming guide <http://docs.nvidia.com/cuda/cuda-c-programming-guide>`_.
docs/source/cuda/overview.rst:Supported GPUs
docs/source/cuda/overview.rst:Numba supports CUDA-enabled GPUs with Compute Capability 3.5 or greater.
docs/source/cuda/overview.rst:- Embedded platforms: NVIDIA Jetson Nano, Jetson Orin Nano, TX1, TX2, Xavier
docs/source/cuda/overview.rst:- Desktop / Server GPUs: All GPUs with Maxwell microarchitecture or later. E.g.
docs/source/cuda/overview.rst:- Laptop GPUs: All GPUs with Maxwell microarchitecture or later. E.g. MX series,
docs/source/cuda/overview.rst:Numba aims to support CUDA Toolkit versions released within the last 3 years.
docs/source/cuda/overview.rst:Presently 11.2 is the minimum required toolkit version. An NVIDIA driver
docs/source/cuda/overview.rst:Conda users can install the CUDA Toolkit into a conda environment.
docs/source/cuda/overview.rst:For CUDA 12, ``cuda-nvcc`` and ``cuda-nvrtc`` are required::
docs/source/cuda/overview.rst:    $ conda install -c conda-forge cuda-nvcc cuda-nvrtc "cuda-version>=12.0"
docs/source/cuda/overview.rst:For CUDA 11, ``cudatoolkit`` is required::
docs/source/cuda/overview.rst:    $ conda install -c conda-forge cudatoolkit "cuda-version>=11.2,<12.0"
docs/source/cuda/overview.rst:If you are not using Conda or if you want to use a different version of CUDA
docs/source/cuda/overview.rst:toolkit, the following describes how Numba searches for a CUDA toolkit
docs/source/cuda/overview.rst:.. _cuda-bindings:
docs/source/cuda/overview.rst:CUDA Bindings
docs/source/cuda/overview.rst:Numba supports interacting with the CUDA Driver API via the `NVIDIA CUDA Python
docs/source/cuda/overview.rst:bindings <https://nvidia.github.io/cuda-python/>`_ and its own ctypes-based
docs/source/cuda/overview.rst:ctypes-based bindings are presently the default, but the NVIDIA bindings will
docs/source/cuda/overview.rst:You can install the NVIDIA bindings with::
docs/source/cuda/overview.rst:   $ conda install -c conda-forge cuda-python
docs/source/cuda/overview.rst:   $ pip install cuda-python
docs/source/cuda/overview.rst:The use of the NVIDIA bindings is enabled by setting the environment variable
docs/source/cuda/overview.rst::envvar:`NUMBA_CUDA_USE_NVIDIA_BINDING` to ``"1"``.
docs/source/cuda/overview.rst:.. _cudatoolkit-lookup:
docs/source/cuda/overview.rst:Setting CUDA Installation Path
docs/source/cuda/overview.rst:Numba searches for a CUDA toolkit installation in the following order:
docs/source/cuda/overview.rst:1. Conda installed CUDA Toolkit packages
docs/source/cuda/overview.rst:2. Environment variable ``CUDA_HOME``, which points to the directory of the
docs/source/cuda/overview.rst:   installed CUDA toolkit (i.e. ``/home/user/cuda-12``)
docs/source/cuda/overview.rst:3. System-wide installation at exactly ``/usr/local/cuda`` on Linux platforms.
docs/source/cuda/overview.rst:   Versioned installation paths (i.e. ``/usr/local/cuda-12.0``) are intentionally
docs/source/cuda/overview.rst:   ignored.  Users can use ``CUDA_HOME`` to select specific versions.
docs/source/cuda/overview.rst:In addition to the CUDA toolkit libraries, which can be installed by conda into
docs/source/cuda/overview.rst:an environment or installed system-wide by the `CUDA SDK installer
docs/source/cuda/overview.rst:<https://developer.nvidia.com/cuda-downloads>`_, the CUDA target in Numba
docs/source/cuda/overview.rst:also requires an up-to-date NVIDIA graphics driver.  Updated graphics drivers
docs/source/cuda/overview.rst:are also installed by the CUDA SDK installer, so there is no need to do both.
docs/source/cuda/overview.rst:If the ``libcuda`` library is in a non-standard location, users can set
docs/source/cuda/overview.rst:environment variable ``NUMBA_CUDA_DRIVER`` to the file path (not the directory
docs/source/cuda/overview.rst:Missing CUDA Features
docs/source/cuda/overview.rst:Numba does not implement all features of CUDA, yet.  Some missing features
docs/source/cuda/kernels.rst:Writing CUDA Kernels
docs/source/cuda/kernels.rst:CUDA has an execution model unlike the traditional sequential model used
docs/source/cuda/kernels.rst:for programming CPUs.  In CUDA, the code you write will be executed by
docs/source/cuda/kernels.rst:Numba's CUDA support exposes facilities to declare and manage this
docs/source/cuda/kernels.rst:exposed by NVidia's CUDA C language.
docs/source/cuda/kernels.rst:Numba also exposes three kinds of GPU memory: global :ref:`device memory
docs/source/cuda/kernels.rst:<cuda-device-memory>` (the large, relatively slow
docs/source/cuda/kernels.rst:off-chip memory that's connected to the GPU itself), on-chip
docs/source/cuda/kernels.rst::ref:`shared memory <cuda-shared-memory>` and :ref:`local memory <cuda-local-memory>`.
docs/source/cuda/kernels.rst:A *kernel function* is a GPU function that is meant to be called from CPU
docs/source/cuda/kernels.rst:At first sight, writing a CUDA kernel with Numba looks very much like
docs/source/cuda/kernels.rst:    @cuda.jit
docs/source/cuda/kernels.rst:(*) Note: newer CUDA devices support device-side kernel launching; this feature
docs/source/cuda/kernels.rst:.. _cuda-kernel-invocation:
docs/source/cuda/kernels.rst:  :func:`cuda.synchronize() <numba.cuda.synchronize>` to wait for all previous
docs/source/cuda/kernels.rst:  share a given area of :ref:`shared memory <cuda-shared-memory>`.
docs/source/cuda/kernels.rst:  `CUDA C Programming Guide`_.
docs/source/cuda/kernels.rst:To help deal with multi-dimensional arrays, CUDA allows you to specify
docs/source/cuda/kernels.rst:    @cuda.jit
docs/source/cuda/kernels.rst:        tx = cuda.threadIdx.x
docs/source/cuda/kernels.rst:        ty = cuda.blockIdx.x
docs/source/cuda/kernels.rst:        bw = cuda.blockDim.x
docs/source/cuda/kernels.rst:are special objects provided by the CUDA backend for the sole purpose of
docs/source/cuda/kernels.rst::ref:`invoked <cuda-kernel-invocation>`.  To access the value at each
docs/source/cuda/kernels.rst:.. attribute:: numba.cuda.threadIdx
docs/source/cuda/kernels.rst:   inclusive to :attr:`numba.cuda.blockDim` exclusive.  A similar rule
docs/source/cuda/kernels.rst:.. attribute:: numba.cuda.blockDim
docs/source/cuda/kernels.rst:.. attribute:: numba.cuda.blockIdx
docs/source/cuda/kernels.rst:   from 0 inclusive to :attr:`numba.cuda.gridDim` exclusive.  A similar rule
docs/source/cuda/kernels.rst:.. attribute:: numba.cuda.gridDim
docs/source/cuda/kernels.rst:.. function:: numba.cuda.grid(ndim)
docs/source/cuda/kernels.rst:.. function:: numba.cuda.gridsize(ndim)
docs/source/cuda/kernels.rst:    @cuda.jit
docs/source/cuda/kernels.rst:        pos = cuda.grid(1)
docs/source/cuda/kernels.rst:    @cuda.jit
docs/source/cuda/kernels.rst:        x, y = cuda.grid(2)
docs/source/cuda/kernels.rst:Please refer to the the `CUDA C Programming Guide`_ for a detailed discussion
docs/source/cuda/kernels.rst:of CUDA programming.
docs/source/cuda/kernels.rst:.. _CUDA C Programming Guide: http://docs.nvidia.com/cuda/cuda-c-programming-guide
docs/source/release/0.59.1-notes.rst:CUDA API Changes
docs/source/release/0.59.1-notes.rst:* PR `#9450 <https://github.com/numba/numba/pull/9450>`_: Fix gpuci versions (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release/0.59.0-notes.rst:CUDA API Changes
docs/source/release/0.59.0-notes.rst::func:`compile_ptx() <numba.cuda.compile_ptx>` API, for easier interoperability
docs/source/release/0.59.0-notes.rst:with CUDA C/C++ and other languages.
docs/source/release/0.59.0-notes.rst:``cuda.grid()`` and ``cuda.gridsize()`` now use 64-bit integers, so they no longer
docs/source/release/0.59.0-notes.rst:Support for Windows CUDA 12.0 toolkit conda packages
docs/source/release/0.59.0-notes.rst:The library paths used in CUDA toolkit 12.0 conda packages on Windows are
docs/source/release/0.59.0-notes.rst:added to the search paths used when detecting CUDA libraries.
docs/source/release/0.59.0-notes.rst:* PR `#9223 <https://github.com/numba/numba/pull/9223>`_: CUDA: Add support for compiling device functions with C ABI (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release/0.59.0-notes.rst:* PR `#9235 <https://github.com/numba/numba/pull/9235>`_: CUDA: Make `grid()` and `gridsize()` use 64-bit integers (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release/0.59.0-notes.rst:* PR `#9267 <https://github.com/numba/numba/pull/9267>`_: CUDA: Fix dropping of kernels by nvjitlink, by implementing the used list (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release/0.59.0-notes.rst:* PR `#9279 <https://github.com/numba/numba/pull/9279>`_: CUDA: Add support for CUDA 12.0 Windows conda packages (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release/0.59.0-notes.rst:* PR `#9292 <https://github.com/numba/numba/pull/9292>`_: CUDA: Switch cooperative groups to use overloads (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release/0.59.0-notes.rst:* PR `#9318 <https://github.com/numba/numba/pull/9318>`_: GPU CI: Test with Python 3.9-3.12 (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release/0.59.0-notes.rst:* PR `#9325 <https://github.com/numba/numba/pull/9325>`_: Fix GPUCI (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release/0.58.0-notes.rst:Support for CUDA toolkits < 11.2 is removed.
docs/source/release/0.58.0-notes.rst:CUDA Changes
docs/source/release/0.58.0-notes.rst:Bitwise operation ``ufunc`` support for the CUDA target.
docs/source/release/0.58.0-notes.rst:CUDA target. Namely:
docs/source/release/0.58.0-notes.rst:Add support for the latest CUDA driver codes.
docs/source/release/0.58.0-notes.rst:Support is added for the latest set of CUDA driver codes.
docs/source/release/0.58.0-notes.rst:Add NumPy comparison ufunc in CUDA
docs/source/release/0.58.0-notes.rst:this PR adds support for comparison ufuncs for the CUDA target
docs/source/release/0.58.0-notes.rst:Report absolute path of ``libcuda.so`` on Linux
docs/source/release/0.58.0-notes.rst:``numba -s`` now reports the absolute path to ``libcuda.so`` on Linux, to aid
docs/source/release/0.58.0-notes.rst:functions that make calls through ``nvdisasm``. For example the CUDA dispatcher
docs/source/release/0.58.0-notes.rst:Add CUDA SASS CFG Support
docs/source/release/0.58.0-notes.rst:It adds an ``inspect_sass_cfg()`` method to CUDADispatcher and the ``-cfg``
docs/source/release/0.58.0-notes.rst:linking CUDA C / C++ sources without needing the NVIDIA CUDA Python bindings.
docs/source/release/0.58.0-notes.rst:Fix CUDA atomics tests with toolkit 12.2
docs/source/release/0.58.0-notes.rst:CUDA 12.2 generates slightly different PTX for some atomics, so the relevant
docs/source/release/0.58.0-notes.rst:* PR `#8861 <https://github.com/numba/numba/pull/8861>`_: CUDA: Don't add device kwarg for jit registry (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release/0.58.0-notes.rst:* PR `#8895 <https://github.com/numba/numba/pull/8895>`_: CUDA: Enable caching functions that use CG (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release/0.58.0-notes.rst:* PR `#8974 <https://github.com/numba/numba/pull/8974>`_: CUDA: Add binary ufunc support (`Matt711 <https://github.com/Matt711>`_)
docs/source/release/0.58.0-notes.rst:* PR `#8988 <https://github.com/numba/numba/pull/8988>`_: support for latest CUDA driver codes #8363 (`s1Sharp <https://github.com/s1Sharp>`_)
docs/source/release/0.58.0-notes.rst:* PR `#9007 <https://github.com/numba/numba/pull/9007>`_: CUDA: Add comparison ufunc support (`Matt711 <https://github.com/Matt711>`_)
docs/source/release/0.58.0-notes.rst:* PR `#9034 <https://github.com/numba/numba/pull/9034>`_: CUDA libs test: Report the absolute path of the loaded libcuda.so on Linux, + other improvements (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release/0.58.0-notes.rst:* PR `#9035 <https://github.com/numba/numba/pull/9035>`_: CUDA: Allow for debuginfo in nvdisasm output (`Matt711 <https://github.com/Matt711>`_)
docs/source/release/0.58.0-notes.rst:* PR `#9051 <https://github.com/numba/numba/pull/9051>`_: Add CUDA CFG support (`Matt711 <https://github.com/Matt711>`_)
docs/source/release/0.58.0-notes.rst:* PR `#9088 <https://github.com/numba/numba/pull/9088>`_: Fix: Issue 9063 - CUDA atomics tests failing with CUDA 12.2 (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release/0.60.0-notes.rst:CUDA API Changes
docs/source/release/0.60.0-notes.rst:Support math.log, math.log2 and math.log10 in CUDA
docs/source/release/0.60.0-notes.rst:CUDA target now supports ``np.log``, ``np.log2`` and ``np.log10``.
docs/source/release/0.60.0-notes.rst:``numba.cuda.gpus.current`` documentation correction
docs/source/release/0.60.0-notes.rst:``numba.cuda.gpus.current`` was erroneously described
docs/source/release/0.60.0-notes.rst:CUDA 12 conda installation documentation
docs/source/release/0.60.0-notes.rst:Installation instructions have been added for CUDA 12 conda users.
docs/source/release/0.60.0-notes.rst:* PR `#9274 <https://github.com/numba/numba/pull/9274>`_: CUDA: Add support for compilation to LTO-IR (`gmarkall <https://github.com/gmarkall>`_)
docs/source/release/0.60.0-notes.rst:* PR `#9417 <https://github.com/numba/numba/pull/9417>`_: Add np.log* bindings for CUDA (`guilhermeleobas <https://github.com/guilhermeleobas>`_ `gmarkall <https://github.com/gmarkall>`_)
docs/source/release/0.60.0-notes.rst:* PR `#9487 <https://github.com/numba/numba/pull/9487>`_: Add CUDA 12 conda installation docs (`bdice <https://github.com/bdice>`_ `gmarkall <https://github.com/gmarkall>`_)
docs/upcoming_changes/README.rst:* ``cuda``: Changes in the CUDA target implementation.
.flake8:    # the public API to be star-imported in numba.cuda.__init__
.flake8:    numba/cuda/device_init.py:F401,F403,F405
.flake8:    numba/cuda/libdevice.py:E501
.flake8:    numba/cuda/tests/doc_examples/test_random.py:E501
.flake8:    numba/cuda/tests/doc_examples/test_cg.py:E501
.flake8:    numba/cuda/tests/doc_examples/test_matmul.py:E501
codecov.yml:        - "numba/cuda/.*"
maint/towncrier_rst_validator.py:                    "cuda",
maint/towncrier_rst_validator.py:    ", cuda, new_feature, improvement, performance, change, doc" + \
LICENSES.third-party:CUDA Half Precision Headers
LICENSES.third-party:The files numba/cuda/cuda_fp16.h and numba/cuda/cuda_fp16.hpp are vendored from
LICENSES.third-party:the CUDA Toolkit version 11.2.2 under the terms of the NVIDIA Software License
LICENSES.third-party:Agreement and CUDA Supplement to Software License Agreement, available at:
LICENSES.third-party:https://docs.nvidia.com/cuda/archive/11.2.2/eula/index.html
LICENSES.third-party:https://docs.nvidia.com/cuda/archive/11.2.2/eula/index.html#attachment-a

```
