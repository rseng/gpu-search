# https://github.com/benvanwerkhoven/kernel_tuner

```console
.zenodo.json:        "GPU Computing",
.zenodo.json:        "OpenCL",
.zenodo.json:        "CUDA"
noxfile.py:    # check if optional dependencies have been disabled by user arguments (e.g. `nox -- skip-gpu`, `nox -- skip-cuda`)
noxfile.py:    install_cuda = True
noxfile.py:    install_opencl = True
noxfile.py:            if arg.lower() == "skip-gpu":
noxfile.py:                install_cuda = False
noxfile.py:                install_opencl = False
noxfile.py:            elif arg.lower() == "skip-cuda":
noxfile.py:                install_cuda = False
noxfile.py:            elif arg.lower() == "skip-opencl":
noxfile.py:                install_opencl = False
noxfile.py:    full_install = install_cuda and install_hip and install_opencl and install_additional_tests
noxfile.py:    if install_cuda:
noxfile.py:        extras_args.extend(["-E", "cuda"])
noxfile.py:    if install_opencl:
noxfile.py:        extras_args.extend(["-E", "opencl"])
noxfile.py:                  Run with `-- skip-gpu` or one of the more specific options (e.g. `-- skip-cuda`) to avoid this."""
noxfile.py:    if install_cuda:
noxfile.py:        # use NVCC to get the CUDA version
noxfile.py:        cuda_version = re.match(r"^.*release ([0-9]+.[0-9]+).*$", nvcc_output, flags=re.IGNORECASE).group(1).strip()
noxfile.py:        session.warn(f"Detected CUDA version: {cuda_version}")
noxfile.py:        # if we need to install the CUDA extras, first install pycuda seperately, reason:
noxfile.py:        if " not found: " in session.run("pip", "show", "pycuda", external=True, silent=True, success_codes=[0,1]):
noxfile.py:            # if PyCUDA is not installed, install it
noxfile.py:            session.warn("PyCUDA not installed")
noxfile.py:                session.install("pycuda", "--no-cache-dir", "--force-reinstall") # Attention: if changed, check `pycuda` in pyproject.toml as well
noxfile.py:            session.warn("PyCUDA installed")
noxfile.py:            # if PyCUDA is already installed, check whether the CUDA version PyCUDA was installed with matches the current CUDA version
noxfile.py:            session.install("numpy")    # required by pycuda.driver
noxfile.py:            pycuda_version = session.run("python", "-c", "import pycuda.driver as drv; drv.init(); print('.'.join(list(str(d) for d in drv.get_version())))", silent=True)
noxfile.py:            shortest_string, longest_string = (pycuda_version, cuda_version) if len(pycuda_version) < len(cuda_version) else (cuda_version, pycuda_version)
noxfile.py:                session.warn(f"PyCUDA was compiled with a version of CUDA ({pycuda_version}) that does not match the current version ({cuda_version}). Re-installing.")
noxfile.py:                    session.install("pycuda", "--no-cache-dir", "--force-reinstall")  # Attention: if changed, check `pycuda` in pyproject.toml as well
noxfile.py:    if install_additional_tests and install_cuda:
noxfile.py:        # install cuda-python
noxfile.py:            session.install("cuda-python")
noxfile.py:                # based on the CUDA version, try installing the exact prebuilt cupy version
noxfile.py:                cuda_cupy_version = f"cupy-cuda{''.join(cuda_version.split('.'))}"
noxfile.py:                session.install(cuda_cupy_version)
noxfile.py:                cuda_cupy_version_x = f"cupy-cuda{cuda_version.split('.')[0]}x"
noxfile.py:                session.warn(f"CuPy exact prebuilt not available for {cuda_version}, trying {cuda_cupy_version_x}")
noxfile.py:                session.install(cuda_cupy_version_x)
noxfile.py:            session.warn(f"No prebuilt CuPy found for CUDA {cuda_version}, building from source...")
noxfile.py:                     Run with 'additional-tests' and without 'skip-gpu', 'skip-cuda' etc. to avoid this.
test/test_integration.py:        "device_name": "My GPU"
test/test_integration.py:        assert len([d for d in stored_data if d["device_name"] == "My_GPU" and d["problem_size"] == "100"]) == 3
test/test_integration.py:        assert len([d for d in stored_data if d["device_name"] == "My_GPU" and d["problem_size"] == "100"]) == 3
test/test_integration.py:        assert len([d for d in stored_data if d["device_name"] == "My_GPU" and d["problem_size"] == "1000"]) == 3
test/test_integration.py:        #test if results for a different GPU can be added
test/test_integration.py:        integration.store_results(filename, kernel_name, kernel_string, tune_params, problem_size, results, { "device_name": "Another GPU"}, top=3)
test/test_integration.py:        my_gpu_100_data = [d for d in stored_data if d["device_name"] == "My_GPU" and d["problem_size"] == "100"]
test/test_integration.py:        assert len(my_gpu_100_data) == 1
test/test_integration.py:        assert my_gpu_100_data[0]["time"] < 100
test/test_integration.py:        #{'My_GPU': {'100': [{'a': 1, 'b': 4, 'time': 100.0}, {'a': 1, 'b': 5, 'time': 101.0}, {'a': 1, 'b': 6, 'time': 102.0}]}}
test/test_integration.py:        assert "#ifdef TARGET_My_GPU" in output_str
test/test_integration.py:        expected = "\n".join(["TARGET_My_GPU", "#define a 1", "#define b 5"])
test/test_integration.py:        expected = "\n".join(["TARGET_My_GPU", "#define a 1", "#define b 6"])
test/test_integration.py:        #test output when more then one GPU is used
test/test_integration.py:        env['device_name'] = "My_GPU2"
test/test_integration.py:        expected = "\n".join(["TARGET_My_GPU", "#define a 1", "#define b 6"])
test/test_integration.py:        expected = "\n".join(["TARGET_My_GPU2", "#define a 1", "#define b 5"])
test/test_integration.py:        assert "TARGET_My_GPU" in output_str
test/test_integration.py:        expected = "\n".join(["TARGET_My_GPU", "#define a 1", "#define b 5"])
test/test_searchspace.py:tune_params["gpu1"] = list(range(num_layers))
test/test_searchspace.py:tune_params["gpu2"] = list(range(num_layers))
test/test_searchspace.py:tune_params["gpu3"] = list(range(num_layers))
test/test_searchspace.py:tune_params["gpu4"] = list(range(num_layers))
test/test_searchspace.py:# each GPU must have at least one layer and the sum of all layers must not exceed the total number of layers
test/test_searchspace.py:def _min_func(gpu1, gpu2, gpu3, gpu4):
test/test_searchspace.py:    return min([gpu1, gpu2, gpu3, gpu4]) >= 1
test/test_searchspace.py:sort_tune_params["gpu1"] = list(range(num_layers))
test/test_searchspace.py:sort_tune_params["gpu2"] = list(range(num_layers))
test/test_searchspace.py:sort_tune_params["gpu3"] = list(range(num_layers))
test/test_searchspace.py:        'device_name': 'NVIDIA A40',
test/test_pycuda_functions.py:from .context import skip_if_no_pycuda
test/test_pycuda_functions.py:from kernel_tuner.backends import pycuda as kt_pycuda
test/test_pycuda_functions.py:    import pycuda.driver
test/test_pycuda_functions.py:@skip_if_no_pycuda
test/test_pycuda_functions.py:    dev = kt_pycuda.PyCudaFunctions(0)
test/test_pycuda_functions.py:    gpu_args = dev.ready_argument_list(arguments)
test/test_pycuda_functions.py:    assert isinstance(gpu_args[0], pycuda.driver.DeviceAllocation)
test/test_pycuda_functions.py:    assert isinstance(gpu_args[1], np.int32)
test/test_pycuda_functions.py:    assert isinstance(gpu_args[2], pycuda.driver.DeviceAllocation)
test/test_pycuda_functions.py:    assert isinstance(gpu_args[3], np.uint8)
test/test_pycuda_functions.py:@skip_if_no_pycuda
test/test_pycuda_functions.py:    kernel_sources = KernelSource(kernel_name, kernel_string, "cuda")
test/test_pycuda_functions.py:    dev = kt_pycuda.PyCudaFunctions(0)
test/context.py:    import pycuda.driver as drv
test/context.py:    pycuda_present = True
test/context.py:    pycuda_present = False
test/context.py:    import pyopencl
test/context.py:    opencl_present = True
test/context.py:    if "namespace" in str(sys.modules["pyopencl"]):
test/context.py:        opencl_present = False
test/context.py:    if len(pyopencl.get_platforms()) == 0:
test/context.py:        opencl_present = False
test/context.py:    opencl_present = False
test/context.py:openacc_present = shutil.which("nvc++") is not None
test/context.py:    cupy.cuda.Device(
test/context.py:    ).attributes  # triggers exception if there are no CUDA-capable devices
test/context.py:    import cuda
test/context.py:    cuda_present = True
test/context.py:    cuda_present = False
test/context.py:skip_if_no_pycuda = pytest.mark.skipif(
test/context.py:    not pycuda_present, reason="PyCuda not installed or no CUDA device detected"
test/context.py:    not cupy_present, reason="CuPy not installed or no CUDA device detected"
test/context.py:skip_if_no_cuda = pytest.mark.skipif(
test/context.py:    not cuda_present, reason="NVIDIA CUDA not installed"
test/context.py:skip_if_no_opencl = pytest.mark.skipif(
test/context.py:    not opencl_present, reason="PyOpenCL not installed or no OpenCL device detected"
test/context.py:skip_if_no_openacc = pytest.mark.skipif(not openacc_present, reason="No nvc++ on PATH")
test/context.py:    if backend.upper() == "CUDA" and not pycuda_present:
test/context.py:        pytest.skip("PyCuda not installed or no CUDA device detected")
test/context.py:        pytest.skip("CuPy not installed or no CUDA device detected")
test/context.py:    elif backend.upper() == "NVCUDA" and not cuda_present:
test/context.py:        pytest.skip("NVIDIA CUDA not installed")
test/context.py:    elif backend.upper() == "OPENCL" and not opencl_present:
test/context.py:        pytest.skip("PyOpenCL not installed or no OpenCL device detected")
test/context.py:    elif backend.upper() == "OPENACC" and not openacc_present:
test/test_cuda_functions.py:from kernel_tuner.backends import nvcuda
test/test_cuda_functions.py:from .context import skip_if_no_cuda
test/test_cuda_functions.py:    from cuda import cuda
test/test_cuda_functions.py:@skip_if_no_cuda
test/test_cuda_functions.py:    dev = nvcuda.CudaFunctions(0)
test/test_cuda_functions.py:    gpu_args = dev.ready_argument_list(arguments)
test/test_cuda_functions.py:    assert isinstance(gpu_args[0], cuda.CUdeviceptr)
test/test_cuda_functions.py:    assert isinstance(gpu_args[1], np.int32)
test/test_cuda_functions.py:    assert isinstance(gpu_args[2], cuda.CUdeviceptr)
test/test_cuda_functions.py:@skip_if_no_cuda
test/test_cuda_functions.py:    kernel_sources = KernelSource(kernel_name, kernel_string, "cuda")
test/test_cuda_functions.py:    dev = nvcuda.CudaFunctions(0)
test/test_cuda_functions.py:@skip_if_no_cuda
test/test_cuda_functions.py:    result, _ = tune_kernel(*env, lang="nvcuda", verbose=True)
test/test_hip_functions.py:    gpu_args = dev.ready_argument_list(arguments)
test/test_hip_functions.py:    assert(isinstance(gpu_args[1], ctypes.c_int))
test/test_hip_functions.py:    assert(isinstance(gpu_args[3], ctypes.c_bool))
test/test_hip_functions.py:    assert(gpu_args[1] == a)
test/test_hip_functions.py:    assert(gpu_args[3] == c)
test/test_hip_functions.py:    gpu_args = dev.ready_argument_list([output])
test/test_hip_functions.py:    dev.run_kernel(kernel, gpu_args, threads, grid)
test/test_hip_functions.py:    dev.memcpy_dtoh(output, gpu_args[0])
test/test_core.py:from .context import skip_if_no_pycuda
test/test_core.py:    lang = "CUDA"
test/test_core.py:@skip_if_no_pycuda
test/test_core.py:    # gpu_args = dev.ready_argument_list(args)
test/test_core.py:        # dev.check_kernel_output(func, gpu_args, instance, answer, 1e-6, None, verbose)
test/test_core.py:@patch('kernel_tuner.core.PyCudaFunctions')
test/test_core.py:    dev = core.DeviceInterface(core.KernelSource("name", "", lang="CUDA"))
test/test_core.py:def test_preprocess_gpu_arguments():
test/test_core.py:    assert core._preprocess_gpu_arguments(arguments, params) == expected
test/test_compiler_functions.py:from .test_runners import env as cuda_env  # noqa: F401
test/test_compiler_functions.py:    mem = cp.cuda.UnownedMemory(
test/test_compiler_functions.py:    ptr = cp.cuda.MemoryPointer(mem, 0)
test/test_cache_file.json:    "device_name": "NVIDIA RTX A4000",
test/test_kernelbuilder.py:backends = ["cuda", "cupy"]
test/test_kernelbuilder.py:    env = {"device_name": "bogus GPU"}
test/test_energy.py:from .context import skip_if_no_pycuda, skip_if_no_pynvml
test/test_energy.py:cache_filename = os.path.dirname(os.path.realpath(__file__)) + "/synthetic_fp32_cache_NVIDIA_RTX_A4000.json"
test/test_energy.py:@skip_if_no_pycuda
test/test_util_functions.py:import kernel_tuner.backends.nvcuda as nvcuda
test/test_util_functions.py:import kernel_tuner.backends.opencl as opencl
test/test_util_functions.py:import kernel_tuner.backends.pycuda as pycuda
test/test_util_functions.py:from .context import skip_if_no_cuda, skip_if_no_opencl, skip_if_no_pycuda
test/test_util_functions.py:def test_to_valid_nvrtc_gpu_arch_cc():
test/test_util_functions.py:    assert to_valid_nvrtc_gpu_arch_cc("89") == "89"
test/test_util_functions.py:    assert to_valid_nvrtc_gpu_arch_cc("88") == "87"
test/test_util_functions.py:    assert to_valid_nvrtc_gpu_arch_cc("86") == "80"
test/test_util_functions.py:    assert to_valid_nvrtc_gpu_arch_cc("40") == "52"
test/test_util_functions.py:    assert to_valid_nvrtc_gpu_arch_cc("90b") == "90a"
test/test_util_functions.py:    assert to_valid_nvrtc_gpu_arch_cc("91c") == "90a"
test/test_util_functions.py:    assert to_valid_nvrtc_gpu_arch_cc("1234") == "52"
test/test_util_functions.py:        "this", kernel, params, grid, threads, block_size_names, "CUDA", None
test/test_util_functions.py:    _, output = prepare_kernel_string("this", kernel, params, grid, threads, block_size_names, "CUDA", None)
test/test_util_functions.py:    assert lang == "CUDA"
test/test_util_functions.py:    assert lang == "OpenCL"
test/test_util_functions.py:@skip_if_no_pycuda
test/test_util_functions.py:    lang = "CUDA"
test/test_util_functions.py:    assert isinstance(dev.dev, pycuda.PyCudaFunctions)
test/test_util_functions.py:@skip_if_no_cuda
test/test_util_functions.py:    lang = "NVCUDA"
test/test_util_functions.py:    assert isinstance(dev.dev, nvcuda.CudaFunctions)
test/test_util_functions.py:@skip_if_no_opencl
test/test_util_functions.py:    lang = "OpenCL"
test/test_util_functions.py:    assert isinstance(dev.dev, opencl.OpenCLFunctions)
test/test_backend.py:    skip_if_no_cuda,
test/test_backend.py:    skip_if_no_opencl,
test/test_backend.py:    skip_if_no_pycuda,
test/test_backend.py:from kernel_tuner.backends import backend, compiler, cupy, nvcuda, opencl, pycuda
test/test_backend.py:@skip_if_no_cuda
test/test_backend.py:def test_cuda_backend():
test/test_backend.py:    dev = nvcuda.CudaFunctions()
test/test_backend.py:@skip_if_no_opencl
test/test_backend.py:def test_opencl_backend():
test/test_backend.py:    dev = opencl.OpenCLFunctions()
test/test_backend.py:@skip_if_no_pycuda
test/test_backend.py:def test_pycuda_backend():
test/test_backend.py:    dev = pycuda.PyCudaFunctions()
test/test_runners.py:from .context import skip_if_no_pycuda
test/test_runners.py:@skip_if_no_pycuda
test/test_runners.py:@skip_if_no_pycuda
test/test_runners.py:@skip_if_no_pycuda
test/test_runners.py:@skip_if_no_pycuda
test/test_runners.py:@skip_if_no_pycuda
test/test_runners.py:@skip_if_no_pycuda
test/test_observers.py:    skip_if_no_cuda,
test/test_observers.py:    skip_if_no_opencl,
test/test_observers.py:    skip_if_no_pycuda,
test/test_observers.py:from .test_opencl_functions import env as env_opencl  # noqa: F401
test/test_observers.py:@skip_if_no_pycuda
test/test_observers.py:@skip_if_no_pycuda
test/test_observers.py:@skip_if_no_pycuda
test/test_observers.py:def test_register_observer_pycuda(env):
test/test_observers.py:    result, _ = kernel_tuner.tune_kernel(*env, observers=[RegisterObserver()], lang='CUDA')
test/test_observers.py:@skip_if_no_cuda
test/test_observers.py:def test_register_observer_nvcuda(env):
test/test_observers.py:    result, _ = kernel_tuner.tune_kernel(*env, observers=[RegisterObserver()], lang='NVCUDA')
test/test_observers.py:@skip_if_no_opencl
test/test_observers.py:def test_register_observer_opencl(env_opencl):
test/test_observers.py:        kernel_tuner.tune_kernel(*env_opencl, observers=[RegisterObserver()], lang='OpenCL')
test/test_observers.py:    assert "OpenCL" in str(err.value)
test/test_opencl_functions.py:from kernel_tuner.backends import opencl
test/test_opencl_functions.py:from .context import skip_if_no_opencl
test/test_opencl_functions.py:    import pyopencl
test/test_opencl_functions.py:@skip_if_no_opencl
test/test_opencl_functions.py:    dev = opencl.OpenCLFunctions(0)
test/test_opencl_functions.py:    gpu_args = dev.ready_argument_list(arguments)
test/test_opencl_functions.py:    assert isinstance(gpu_args[0], pyopencl.Buffer)
test/test_opencl_functions.py:    assert isinstance(gpu_args[1], np.int32)
test/test_opencl_functions.py:    assert isinstance(gpu_args[2], pyopencl.Buffer)
test/test_opencl_functions.py:    gpu_args[0].release()
test/test_opencl_functions.py:    gpu_args[2].release()
test/test_opencl_functions.py:@skip_if_no_opencl
test/test_opencl_functions.py:    kernel_sources = KernelSource("sum", original_kernel, "opencl")
test/test_opencl_functions.py:    dev = opencl.OpenCLFunctions(0)
test/test_opencl_functions.py:    assert isinstance(func, pyopencl.Kernel)
test/test_opencl_functions.py:@skip_if_no_opencl
test/test_opencl_functions.py:    dev = opencl.OpenCLFunctions(0)
test/test_opencl_functions.py:@skip_if_no_opencl
test/utils/test_directives.py:def test_is_openacc():
test/utils/test_directives.py:    assert is_openacc(OpenACC())
test/utils/test_directives.py:    assert not is_openacc(None)
test/utils/test_directives.py:def test_line_contains_openacc_directive():
test/utils/test_directives.py:    assert line_contains_openacc_directive(cxx_code, Cxx())
test/utils/test_directives.py:    assert not line_contains_openacc_directive(f90_code, Cxx())
test/utils/test_directives.py:    assert line_contains_openacc_directive(f90_code, Fortran())
test/utils/test_directives.py:    assert not line_contains_openacc_directive(cxx_code, Fortran())
test/utils/test_directives.py:    assert not line_contains_openacc_directive(cxx_code, None)
test/utils/test_directives.py:def test_line_contains_openacc_parallel_directive():
test/utils/test_directives.py:    assert line_contains_openacc_parallel_directive("#pragma acc parallel wait", Cxx())
test/utils/test_directives.py:    assert line_contains_openacc_parallel_directive("!$acc parallel", Fortran())
test/utils/test_directives.py:    assert not line_contains_openacc_parallel_directive("#pragma acc loop", Cxx())
test/utils/test_directives.py:    assert not line_contains_openacc_parallel_directive("!$acc loop", Fortran())
test/utils/test_directives.py:    assert not line_contains_openacc_parallel_directive("!$acc parallel", None)
test/utils/test_directives.py:def test_openacc_directive_contains_data_clause():
test/utils/test_directives.py:    assert openacc_directive_contains_data_clause("#pragma acc parallel present(A[:1089])")
test/utils/test_directives.py:    assert not openacc_directive_contains_data_clause("#pragma acc parallel for")
test/utils/test_directives.py:        create_data_directive_openacc("array", size, Cxx())
test/utils/test_directives.py:        create_data_directive_openacc("matrix", size, Fortran())
test/utils/test_directives.py:    assert create_data_directive_openacc("array", size, None) == ""
test/utils/test_directives.py:    assert exit_data_directive_openacc("array", size, Cxx()) == "#pragma acc exit data copyout(array[:1024])\n"
test/utils/test_directives.py:    assert exit_data_directive_openacc("matrix", size, Fortran()) == "!$acc exit data copyout(matrix(:35,:16))\n"
test/utils/test_directives.py:    assert exit_data_directive_openacc("matrix", size, None) == ""
test/utils/test_directives.py:    acc_cxx = Code(OpenACC(), Cxx())
test/utils/test_directives.py:    acc_f90 = Code(OpenACC(), Fortran())
test/utils/test_directives.py:    acc_cxx = Code(OpenACC(), Cxx())
test/utils/test_directives.py:    returns = extract_directive_code(code, Code(OpenACC(), Fortran()), "vector_add")
test/utils/test_directives.py:    acc_cxx = Code(OpenACC(), Cxx())
test/utils/test_directives.py:    signatures = extract_directive_signature(code, Code(OpenACC(), Fortran()))
test/utils/test_directives.py:    acc_cxx = Code(OpenACC(), Cxx())
test/utils/test_directives.py:    acc_f90 = Code(OpenACC(), Fortran())
test/utils/test_directives.py:    data = extract_directive_data(code, Code(OpenACC(), Cxx()))
test/utils/test_directives.py:    data = extract_directive_data(code, Code(OpenACC(), Fortran()))
test/utils/test_directives.py:    assert extract_initialization_code(code_cpp, Code(OpenACC(), Cxx())) == "const int value = 42;\n"
test/utils/test_directives.py:    assert extract_initialization_code(code_f90, Code(OpenACC(), Fortran())) == "integer :: value\n"
test/utils/test_directives.py:    assert extract_deinitialization_code(code_cpp, Code(OpenACC(), Cxx())) == "const int value = 42;\n"
test/utils/test_directives.py:    assert extract_deinitialization_code(code_f90, Code(OpenACC(), Fortran())) == "integer :: value\n"
test/utils/test_directives.py:def test_add_present_openacc():
test/utils/test_directives.py:    acc_cxx = Code(OpenACC(), Cxx())
test/utils/test_directives.py:    acc_f90 = Code(OpenACC(), Fortran())
test/utils/test_directives.py:    assert add_present_openacc(code_cxx, acc_cxx, data, preprocessor, None) == expected_cxx
test/utils/test_directives.py:    assert add_present_openacc(code_f90, acc_f90, data, preprocessor, None) == expected_f90
test/utils/test_directives.py:    assert add_present_openacc(code_f90, acc_f90, data, preprocessor, None) == code_f90
test/utils/test_directives.py:    assert add_present_openacc(code_cxx, acc_cxx, data, preprocessor, None) == expected_cxx
test/utils/test_directives.py:    assert add_present_openacc(code_f90, acc_f90, data, preprocessor, None) == expected_f90
test/utils/test_directives.py:    assert add_present_openacc(code_f90, acc_f90, data, user_dimensions=dimensions) == expected_f90
test/utils/test_directives.py:    assert add_present_openacc(code_f90, acc_f90, data, preprocessor=[], user_dimensions=dimensions) == expected_f90
test/test_pycuda_mocked.py:from kernel_tuner.backends import pycuda
test/test_pycuda_mocked.py:@patch('kernel_tuner.backends.pycuda.nvml')
test/test_pycuda_mocked.py:@patch('kernel_tuner.backends.pycuda.DynamicSourceModule')
test/test_pycuda_mocked.py:@patch('kernel_tuner.backends.pycuda.drv')
test/test_pycuda_mocked.py:    dev = pycuda.PyCudaFunctions(0)
test/test_pycuda_mocked.py:    gpu_args = dev.ready_argument_list(arguments)
test/test_pycuda_mocked.py:    print(gpu_args)
test/test_pycuda_mocked.py:    assert isinstance(gpu_args[0], np.int32)
test/test_pycuda_mocked.py:@patch('kernel_tuner.backends.pycuda.nvml')
test/test_pycuda_mocked.py:@patch('kernel_tuner.backends.pycuda.DynamicSourceModule')
test/test_pycuda_mocked.py:@patch('kernel_tuner.backends.pycuda.drv')
test/test_pycuda_mocked.py:    dev = pycuda.PyCudaFunctions(0)
test/test_pycuda_mocked.py:    kernel_sources = KernelSource(kernel_name, kernel_string, "cuda")
test/test_pycuda_mocked.py:@patch('kernel_tuner.backends.pycuda.nvml')
test/test_pycuda_mocked.py:@patch('kernel_tuner.backends.pycuda.DynamicSourceModule')
test/test_pycuda_mocked.py:@patch('kernel_tuner.backends.pycuda.drv')
test/test_pycuda_mocked.py:    dev = pycuda.PyCudaFunctions(0)
test/test_pycuda_mocked.py:@patch('kernel_tuner.backends.pycuda.nvml')
test/test_pycuda_mocked.py:@patch('kernel_tuner.backends.pycuda.DynamicSourceModule')
test/test_pycuda_mocked.py:@patch('kernel_tuner.backends.pycuda.drv')
test/test_pycuda_mocked.py:    dev = pycuda.PyCudaFunctions(0)
test/synthetic_fp32_cache_NVIDIA_RTX_A4000.json:"device_name": "NVIDIA RTX A4000",
CHANGELOG.md:- HIP backend to support tuning HIP kernels on AMD GPUs
CHANGELOG.md:- Experimental features for OpenACC tuning
CHANGELOG.md:- No longer inserting partial loop unrolling factor of 0 in CUDA
CHANGELOG.md:- A new backend that uses Nvidia cuda-python
CHANGELOG.md:- Support for boolean scalar arguments in PyCUDA backend
CHANGELOG.md:- Cupy backend to support C++ templated CUDA kernels
CHANGELOG.md:- support for templated CUDA kernels using PyCUDA backend
CHANGELOG.md:- smem_args option for dynamically allocated shared memory in CUDA kernels
CHANGELOG.md:- bugfix for Nvidia devices without internal current sensor
CHANGELOG.md:- support for kernels that use texture memory in CUDA
CHANGELOG.md:- support for measuring energy consumption of CUDA kernels
CHANGELOG.md:- the install procedure now uses extras, e.g. [cuda,opencl]
CHANGELOG.md:- printing units for measured time with CUDA and OpenCL
CHANGELOG.md:- bugfix for GPU cleanup when using Noodles runner
CHANGELOG.md:- actively freeing GPU memory after tuning
CHANGELOG.md:- bugfix for 3D grids when using OpenCL
CHANGELOG.md:- support for dynamic parallelism when using PyCUDA
CHANGELOG.md:- fixed a bug in memset for OpenCL backend
CHANGELOG.md:- CUDA backend prints device in use, similar to OpenCL backend
CHANGELOG.md:- example showing GPU code unit testing with the Kernel Tuner
CHANGELOG.md:- support for OpenCL platform selection
CHANGELOG.md:- Support for constant memory arguments to CUDA kernels
CHANGELOG.md:- OpenCL support
CITATION.cff:  - "GPU Computing"
CITATION.cff:  - "OpenCL"
CITATION.cff:  - "CUDA"
CITATION.cff:    title: "Kernel Tuner: A Search-Optimizing GPU Code Auto-Tuner"
doc/source/templates.rst:It is quite common in CUDA programming to write kernels that use C++ templates. This can be very useful when writing code that can work for several types, for example floats and doubles. However, the use of C++ templates makes it slightly more difficult to directly 
doc/source/templates.rst:integrate the CUDA kernel into applications that are not written in C++, for example Matlab, Fortran, or Python. And since Kernel Tuner is written in Python, we needed to take a few extra steps to provide support for templated CUDA kernels. Let's first look at an 
doc/source/templates.rst:Say we have a templated CUDA kernel in a file called vector_add.cu:
doc/source/templates.rst:.. code-block:: cuda
doc/source/templates.rst:Kernel Tuner supports multiple backends, for CUDA these are based on PyCUDA and Cupy. The following explains how to enable tuning of templated kernels with either backend.
doc/source/templates.rst:The PyCuda backend is the default backend in Kernel Tuner and is selected if the user does not supply the 'lang' option and CUDA code is detected in the kernel source, or when lang is set to "CUDA" by the user. PyCuda requires CUDA kernels to have extern C linkage, 
doc/source/templates.rst:which means that C++ templated kernels are not supported. To support templated kernels regardless of this limitation Kernel Tuner attempts to wrap the templated CUDA kernel by inserting a compile-time template instantiation statement and a wrapper kernel that calls 
doc/source/templates.rst:the templated CUDA kernel, which is actually demoted to a __device__ function in the process. These automatic code rewrites have a real risk of breaking the code. To minimize the chance of errors due to Kernel Tuner's automatic code rewrites, it's best to isolate the 
doc/source/templates.rst:The Cupy backend provides much more advanced support for C++ templated kernels, because it internally uses NVRTC, the Nvidia runtime compiler. NVRTC does come with some restrictions however, for example NVRTC does not allow any host code to be inside code that
doc/source/templates.rst:is passed. So, like with the PyCuda backend it helps to separate the source code of device and host functions into seperate files. You can force Kernel Tuner to use the Cupy backend by passing the lang="cupy" option to tune_kernel. 
doc/source/observers.rst:function, the state of GPU memory, or any other information in the GPU runtime.
doc/source/observers.rst:The PyOpenCL, PyCUDA, Cupy, and cuda-python backends support observers. Each backend also implements their own observer to
doc/source/observers.rst:main advantage of using PowerSensor2 over the GPU's built-in power sensor is that PowerSensor2 reports
doc/source/observers.rst:benchmarking as reported by the NVIDIA Management Library (NVML). To facilitate the interaction with
doc/source/observers.rst:almost all Nvidia GPUs, so this method is much more accessible to end-users compared to solutions that require
doc/source/observers.rst:GPU systems. Recently, power-capping, setting application-specific power limits, is also becoming more popular
doc/source/observers.rst:approach to optimize energy consumption of applications. To enable energy tuning of GPU applications,
doc/source/observers.rst:setting does require root privileges. As such, these features may not be available to all users on all systems. The optional argument ``nvidia_smi_fallback`` to NVMLObserver may be set to 
doc/source/observers.rst:the path where you are allowed to run nvidia-smi with privileges. This allows your Kernel Tuner application to run without privileges, and configurating the clock frequencies or power 
doc/source/observers.rst:limits will be done through nvidia-smi.
doc/source/observers.rst:The PMTObserver can be used to measure power and energy on various platforms including Nvidia Jetson, Nvidia NVML,
doc/source/observers.rst:the RAPL interface, AMD ROCM, and Xilinx. It requires PMT to be installed, as well as the PMT's Python interface. 
doc/source/observers.rst:The NCUObserver can be used to automatically extract performance counters during tuning using Nvidia's NsightCompute profiler.
doc/source/matmul/matmul.cu: * Optimized CUDA kernel for matrix multiplication
doc/source/matmul/matmul.cu: * GPU Technology Conference, GTC 2010.
doc/source/matmul/matmul.cu: * tuned towards each GPU. This kernel assumes that
doc/source/matmul/matmul.py:results = kernel_tuner.run_kernel("matmul_kernel", "../examples/cuda/matmul.cu",
doc/source/index.rst:Kernel Tuner is a software development tool for the creation of highly-optimized and tuned GPU applications.
doc/source/index.rst:To tune CUDA kernels:
doc/source/index.rst:- First, make sure you have the `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ installed
doc/source/index.rst:- Then type: ``pip install kernel_tuner[cuda]``
doc/source/index.rst:To tune OpenCL kernels:
doc/source/index.rst:- First, make sure you have an OpenCL compiler for your intended OpenCL platform
doc/source/index.rst:- Then type: ``pip install kernel_tuner[opencl]``
doc/source/index.rst:- ``pip install kernel_tuner[cuda,opencl,hip]``
doc/source/index.rst:The following shows a simple example for tuning a CUDA kernel:
doc/source/index.rst:      title   = {Kernel Tuner: A search-optimizing GPU code auto-tuner},
doc/source/index.rst:      title = {Bayesian Optimization for auto-tuning GPU kernels},
doc/source/index.rst:For a performance comparison of different optimization algorithms for auto-tuning and an analysis of tuning difficulty for different GPUs:
doc/source/index.rst:      title={Benchmarking optimization algorithms for auto-tuning GPU kernels},
doc/source/index.rst:For referencing to Kernel Tuner's capabilities in measuring and optimizing energy consumption of GPU kernels, please cite the following:
doc/source/index.rst:      title = {Going green: optimizing GPUs for energy efficiency through model-steered auto-tuning},
doc/source/correctness.rst:The example in ``examples/cuda/convolution_correct.py`` demonstrates how
doc/source/correctness.rst:interface as ``tune_kernel()``. In this example we run a naive CUDA
doc/source/correctness.rst:This function should accept three parameters: ``cpu_result``, ``gpu_result``, and ``atol``.
doc/source/correctness.rst:The example in ``examples/cuda/reduction.py`` demonstrates how to use the ``verify`` option of ``tune_kernel()``;
doc/source/correctness.rst:    # gpu_result
doc/source/correctness.rst:    def verify_partial_reduce(cpu_result, gpu_result, atol=None):
doc/source/correctness.rst:        return numpy.isclose(cpu_result, numpy.sum(gpu_result), atol=atol)
doc/source/correctness.rst:The second argument, ``gpu_result``, is mapped to the NumPy array provided to the ``arguments`` option of
doc/source/correctness.rst:In the example, the user-defined ``verify`` function is used to compare the partial results, computed on the GPU,
doc/source/hostcode.rst:With the Kernel Tuner it is also possible to tune the host code of your GPU programs, or even just any C function for that matter.
doc/source/hostcode.rst:Tuning host code can be useful when it contains parameters that have impact on the performance of kernel on the GPU, such as the number of
doc/source/hostcode.rst:There are few differences with tuning just a single CUDA or OpenCL kernel, to list them:  
doc/source/hostcode.rst:without having PyCuda installed, because the C functions interface calls the CUDA compiler directly.
doc/source/hostcode.rst:of the parameter space the returned floats will be averaged for the multiple runs in the same way as with direct CUDA or OpenCL kernel tuning.
doc/source/hostcode.rst:By itself the C language does not provide any very precise timing functions. If you are tuning the host code of a CUDA program you can use
doc/source/hostcode.rst:CUDA Events to do the timing for you. However, if you are using plain C then you have to supply your own timing function.
doc/source/hostcode.rst:The following describes the example in ``examples/cuda/convolution_streams.py``.
doc/source/hostcode.rst:What is different is that we also supply the host code, which you can find in ``examples/cuda/convolution_streams.cu``. It is a bit
doc/source/hostcode.rst:is spread across a number of CUDA streams. In this way, it is possible to overlap the data transfers from host to device with kernel execution, and with
doc/source/hostcode.rst:in streams `n` and `n-1` have to be finished. To ensure the latter, we use CUDA Events and in particular cudaStreamWaitEvent(), which halts stream `n` until the 
doc/source/hostcode.rst:The way you use the Kernel Tuner to tune this CUDA program is very similar to when you are tuning a CUDA kernel directly, as you can see below:
doc/source/hostcode.rst:The function that we are tuning is a C function that launches the CUDA kernel by itself, yet we supply the grid_div_x and 
doc/source/hostcode.rst:The filter is not passed separately as a constant memory argument, because the CudaMemcpyToSymbol operation is now performed by the C host function. Also, 
doc/source/structs.rst:One of the issues with calling GPU kernels from Python is the use of custom data types in kernel arguments. In general, it is recommended for portability of your GPU code, which may be
doc/source/structs.rst:For performance reasons, it is also recommended to not use arrays of structs for kernel arguments, as this is very likely to lead to inefficient memory accesses on the GPU.
doc/source/structs.rst:However, there are situations, in particular in scientific applications, where the GPU code needs a lot of input parameters where it makes sense to collect these in a struct that 
doc/source/structs.rst:Numpy, and Kernel Tuner to call a CUDA kernel that uses a struct as kernel argument.
doc/source/structs.rst:The most difficult part of this code is ensuring the struct.pack format string is correct and keeping it in sync with the GPU code. Note the ``0l`` at the end of string. This enables 
doc/source/design.rst:functions such as ``ready_argument_list`` which allocates GPU memory and
doc/source/design.rst:moves data to the GPU, and functions like ``compile``, ``benchmark``, or
doc/source/design.rst:PyCUDA, CuPy, cuda-python, PyOpenCL and PyHIP are for tuning either CUDA, OpenCL, or HIP kernels.
doc/source/design.rst:functions, but in particular to tune C functions that in turn launch GPU kernels.
doc/source/design.rst:kernel_tuner.backends.pycuda.PyCudaFunctions
doc/source/design.rst:.. autoclass:: kernel_tuner.backends.pycuda.PyCudaFunctions
doc/source/design.rst:kernel_tuner.backends.nvcuda.CudaFunctions
doc/source/design.rst:.. autoclass:: kernel_tuner.backends.nvcuda.CudaFunctions
doc/source/design.rst:kernel_tuner.backends.opencl.OpenCLFunctions
doc/source/design.rst:.. autoclass:: kernel_tuner.backends.opencl.OpenCLFunctions
doc/source/conf.py:        "A simple CUDA/OpenCL Auto-Tuner in Python",
doc/source/vocabulary.rst:This document specifies which parameters are special and what there uses are when auto-tuning GPU kernels.
doc/source/backends.rst:Kernel Tuner implements multiple backends for CUDA, one for OpenCL, one for HIP, and a generic 
doc/source/backends.rst:CUDA Backends
doc/source/backends.rst:PyCUDA is default CUDA backend in Kernel Tuner. It is comparable in feature completeness with CuPy.
doc/source/backends.rst:Because the HIP kernel language is identical to the CUDA kernel language, HIP is included here as well.
doc/source/backends.rst:To use HIP on nvidia GPUs, see https://github.com/jatinx/hip-on-nv.
doc/source/backends.rst:While the PyCUDA backend expects all inputs and outputs to be Numpy arrays, the CuPy backend also 
doc/source/backends.rst:entirely on the GPU when using only cupy arrays.
doc/source/backends.rst:Texture memory is only supported by the PyCUDA backend, while the CuPy backend is the only one that 
doc/source/backends.rst:limited support is implemented by Kernel Tuner to support templated kernels for the PyCUDA and 
doc/source/backends.rst:CUDA-Python backends.
doc/source/backends.rst:  :header: Feature, PyCUDA, CuPy, CUDA-Python, HIP
doc/source/backends.rst:  :header: Feature, PyCUDA, CuPy, CUDA-Python, HIP
doc/source/backends.rst:  Python package,      "pycuda", "cupy", "cuda-python", "pyhip-interface"
doc/source/backends.rst:  Selected with lang=, "CUDA", "CUPY", "NVCUDA", "HIP"
doc/source/quickstart.rst:So you have installed Kernel Tuner! That's great! But now you'd like to get started tuning some GPU code.
doc/source/quickstart.rst:Let's say we have a simple CUDA kernel stored in a file called ``vector_add_kernel.cu``:
doc/source/quickstart.rst:.. code-block:: cuda
doc/source/quickstart.rst:our CUDA kernel's argument list with the same order and types.
doc/source/quickstart.rst:What happens how, is that Kernel Tuner will copy the our kernel's input and output data to the GPU, iteratively compile and 
doc/source/dev-environment.rst:#. Make sure that non-Python dependencies are installed if applicable, such as CUDA, OpenCL or HIP. This is described in :ref:`Installation <installation>`.
doc/source/dev-environment.rst:#. Install the project, dependencies and extras: :bash:`poetry install --with test,docs -E cuda -E opencl -E hip`, leaving out :bash:`-E cuda`, :bash:`-E opencl` or :bash:`-E hip` if this does not apply on your system. To go all-out, use :bash:`--all-extras`
doc/source/dev-environment.rst:    * Depending on the environment, it may be necessary or convenient to install extra packages such as :bash:`cupy-cuda11x` / :bash:`cupy-cuda12x`, and :bash:`cuda-python`. These are currently not defined as dependencies for kernel-tuner, but can be part of tests.
doc/source/dev-environment.rst:    * Do not forget to make sure the paths are set correctly. If you're using CUDA, the desired CUDA version should be in :bash:`$PATH`, :bash:`$LD_LIBARY_PATH` and :bash:`$CPATH`.
doc/source/dev-environment.rst:    * [Note]: sometimes, changing the NVIDIA driver privileges is required to read program counters and energy measurements. Check if :bash:`cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly` is set to 1. If so, `follow these steps <https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters>`__
doc/source/dev-environment.rst:#. Make sure that non-Python dependencies are loaded if applicable, such as CUDA, OpenCL or HIP. On most clusters it is possible to load (or unload) modules (e.g. CUDA, OpenCL / ROCM). For more information, see :ref:`Installation <installation>`.
doc/source/dev-environment.rst:    * Do not forget to make sure the paths are set correctly. If you're using CUDA, the desired CUDA version should be in :bash:`$PATH`, :bash:`$LD_LIBARY_PATH` and :bash:`$CPATH`.
doc/source/dev-environment.rst:#. Install the project, dependencies and extras: :bash:`poetry install --with test,docs -E cuda -E opencl -E hip`, leaving out :bash:`-E cuda`, :bash:`-E opencl` or :bash:`-E hip` if this does not apply on your system. To go all-out, use :bash:`--all-extras`.
doc/source/dev-environment.rst:    * Depending on the environment, it may be necessary or convenient to install extra packages such as :bash:`cupy-cuda11x` / :bash:`cupy-cuda12x`, and :bash:`cuda-python`. These are currently not defined as dependencies for kernel-tuner, but can be part of tests.
doc/source/dev-environment.rst:#. Check if the environment is setup correctly by running :bash:`pytest`. All tests should pass, except if you're not on a GPU node, or one or more extras has been left out in the previous step, then these tests will skip gracefully.
doc/source/dev-environment.rst:For full coverage, make Nox use the additional tests (such as cupy and cuda-python) with :bash:`nox -- additional-tests`.
doc/source/dev-environment.rst:* :bash:`nox -- skip-cuda` to skip tests involving CUDA.
doc/source/dev-environment.rst:* :bash:`nox -- skip-opencl` to skip tests involving OpenCL.
doc/source/dev-environment.rst:* :bash:`nox -- skip-gpu` to skip all tests on the GPU (the same as :bash:`nox -- skip-cuda skip-hip skip-opencl`), especially helpful if you don't have a GPU locally. 
doc/source/dev-environment.rst:In this case, tests that require PyCuda and/or a CUDA capable GPU will be skipped automatically if these are not installed/present. 
doc/source/dev-environment.rst:The same holds for tests that require PyOpenCL, Cupy, and CUDA.
README.md:Create optimized GPU applications in any mainstream GPU 
README.md:programming language (CUDA, HIP, OpenCL, OpenACC).
README.md:- Works as an external tool to benchmark and optimize GPU kernels in isolation
README.md:- First, make sure you have your [CUDA](https://kerneltuner.github.io/kernel_tuner/stable/install.html#cuda-and-pycuda), [OpenCL](https://kerneltuner.github.io/kernel_tuner/stable/install.html#opencl-and-pyopencl), or [HIP](https://kerneltuner.github.io/kernel_tuner/stable/install.html#hip-and-pyhipl) compiler installed
README.md:- Then type: `pip install kernel_tuner[cuda]`, `pip install kernel_tuner[opencl]`, or `pip install kernel_tuner[hip]`
README.md:- or why not all of them: `pip install kernel_tuner[cuda,opencl,hip]`
README.md:  - [Test GPU code from Python](https://github.com/KernelTuner/kernel_tuner/blob/master/examples/cuda/test_vector_add.py)
README.md:  - [Mixed-precision & Accuracy tuning](https://github.com/KernelTuner/kernel_tuner/blob/master/examples/cuda/accuracy.py)
README.md:  - Vector add example [[.ipynb](https://github.com/KernelTuner/kernel_tuner_tutorial/blob/master/hands-on/cuda/00_Kernel_Tuner_Introduction.ipynb)] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelTuner/kernel_tuner_tutorial/blob/master/hands-on/cuda/00_Kernel_Tuner_Introduction.ipynb)
README.md:  - Tuning thread block dimensions [[.ipynb](https://github.com/KernelTuner/kernel_tuner_tutorial/blob/master/hands-on/cuda/01_Kernel_Tuner_Getting_Started.ipynb)] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelTuner/kernel_tuner_tutorial/blob/master/hands-on/cuda/01_Kernel_Tuner_Getting_Started.ipynb)
README.md:  - Search space restrictions & output verification [[.ipynb](https://github.com/KernelTuner/kernel_tuner_tutorial/blob/master/hands-on/cuda/02_Kernel_Tuner_Intermediate.ipynb)] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelTuner/kernel_tuner_tutorial/blob/master/hands-on/cuda/02_Kernel_Tuner_Intermediate.ipynb)
README.md:  - Visualization & search space optimization [[.ipynb](https://github.com/KernelTuner/kernel_tuner_tutorial/blob/master/hands-on/cuda/03_Kernel_Tuner_Advanced.ipynb)] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelTuner/kernel_tuner_tutorial/blob/master/hands-on/cuda/03_Kernel_Tuner_Advanced.ipynb)
README.md:- **Energy Efficient GPU Computing** tutorial slides [[PDF]](https://github.com/KernelTuner/kernel_tuner_tutorial/blob/master/slides/2023_Supercomputing/SC23.pdf), hands-on:
README.md:  - Kernel Tuner for GPU energy measurements [[.ipynb](https://github.com/KernelTuner/kernel_tuner_tutorial/blob/master/energy/00_Kernel_Tuner_Introduction.ipynb)] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelTuner/kernel_tuner_tutorial/blob/master/energy/00_Kernel_Tuner_Introduction.ipynb)
README.md:<img width="250px" src="https://raw.githubusercontent.com/KernelTuner/kernel_tuner/master/doc/images/kernel_float.png"/><br />C++ data types for mixed-precision CUDA kernel programming
README.md:  title   = {Kernel Tuner: A search-optimizing GPU code auto-tuner},
kernel_tuner/file_utils.py:    """Get the information about GPUs in the current system, target is any of ['nvidia', 'amd']."""
kernel_tuner/file_utils.py:    if target == "nvidia":
kernel_tuner/file_utils.py:        nvidia_smi_out = subprocess.run(["nvidia-smi", "--query", "-x"], capture_output=True)
kernel_tuner/file_utils.py:        nvidia_smi = xmltodict.parse(nvidia_smi_out.stdout)
kernel_tuner/file_utils.py:        gpu_info = nvidia_smi["nvidia_smi_log"]["gpu"]
kernel_tuner/file_utils.py:        # on multi-GPU systems gpu_info is a list
kernel_tuner/file_utils.py:        if isinstance(gpu_info, list):
kernel_tuner/file_utils.py:            for gpu in gpu_info:
kernel_tuner/file_utils.py:                del gpu[del_key]
kernel_tuner/file_utils.py:        elif isinstance(gpu_info, dict) and del_key in gpu_info:
kernel_tuner/file_utils.py:            del gpu_info[del_key]
kernel_tuner/file_utils.py:        return nvidia_smi
kernel_tuner/file_utils.py:        rocm_smi_out = subprocess.run(["rocm-smi", "--showallinfo", "--json"], capture_output=True)
kernel_tuner/file_utils.py:        return json.loads(rocm_smi_out.stdout)
kernel_tuner/file_utils.py:    # attempts to use nvidia-smi or rocm-smi if present
kernel_tuner/file_utils.py:        device_query["nvidia-smi"] = get_device_query("nvidia")
kernel_tuner/file_utils.py:        # ignore if nvidia-smi is not found, or parse error occurs
kernel_tuner/file_utils.py:        device_query["rocm-smi"] = get_device_query("amd")
kernel_tuner/file_utils.py:        # ignore if rocm-smi is not found, or parse error occurs
kernel_tuner/integration.py:    def get_best_config(self, gpu_name="default", problem_size=None):
kernel_tuner/integration.py:            based on the tuning results for a given gpu_name and problem_size.
kernel_tuner/integration.py:            based on the tuning results for all problem_sizes and the given gpu_name.
kernel_tuner/integration.py:            If gpu_name is not given this function will select a default configuration
kernel_tuner/integration.py:            :param gpu_name: Name of the GPU for which the best configuration
kernel_tuner/integration.py:            :type gpu_name: string
kernel_tuner/integration.py:                on the given gpu_name needs to be retrieved.
kernel_tuner/integration.py:        gpu_name = gpu_name.replace("-", "_").replace(" ", "_")
kernel_tuner/integration.py:        gpu_match = [result for result in self.data if result["device_name"] == gpu_name]
kernel_tuner/integration.py:        if gpu_match:
kernel_tuner/integration.py:            gpu_ps_match = [result for result in gpu_match if problem_size and result["problem_size"] == problem_size_str]
kernel_tuner/integration.py:            if gpu_ps_match:
kernel_tuner/integration.py:                return _get_best_config_from_list(gpu_ps_match, self.objective, self.objective_higher_is_better)
kernel_tuner/integration.py:            return _select_best_common_config(gpu_match, self.objective, self.objective_higher_is_better)
kernel_tuner/integration.py:        #gpu is not among the results, so return a good default
kernel_tuner/integration.py:    #remove existing entries for this GPU and problem_size combination from the results if any
kernel_tuner/integration.py:        ``-DTARGET_GPU="name_of_gpu"``
kernel_tuner/integration.py:    gpu_targets = list({r["device_name"] for r in data})
kernel_tuner/integration.py:    for gpu_name in gpu_targets:
kernel_tuner/integration.py:        targets[gpu_name] = results.get_best_config(gpu_name)
kernel_tuner/integration.py:    for gpu_name, params in targets.items():
kernel_tuner/integration.py:            if_block += f"\n#ifdef TARGET_{gpu_name}\n"
kernel_tuner/integration.py:            if_block += f"\n#elif TARGET_{gpu_name}\n"
kernel_tuner/integration.py:#endif /* GPU TARGETS */
kernel_tuner/schema/T4/1.0.0/metadata-schema.json:              "description": "The output from tools such as nvidia-smi as JSON"
kernel_tuner/backends/nvcuda.py:"""This module contains all NVIDIA cuda-python specific kernel_tuner functions."""
kernel_tuner/backends/nvcuda.py:from kernel_tuner.backends.backend import GPUBackend
kernel_tuner/backends/nvcuda.py:from kernel_tuner.observers.nvcuda import CudaRuntimeObserver
kernel_tuner/backends/nvcuda.py:from kernel_tuner.util import SkippableFailure, cuda_error_check, to_valid_nvrtc_gpu_arch_cc
kernel_tuner/backends/nvcuda.py:# and run tests without cuda-python installed
kernel_tuner/backends/nvcuda.py:    from cuda import cuda, cudart, nvrtc
kernel_tuner/backends/nvcuda.py:    cuda = None
kernel_tuner/backends/nvcuda.py:class CudaFunctions(GPUBackend):
kernel_tuner/backends/nvcuda.py:    """Class that groups the Cuda functions on maintains state about the device."""
kernel_tuner/backends/nvcuda.py:        """Instantiate CudaFunctions object used for interacting with the CUDA device.
kernel_tuner/backends/nvcuda.py:        :param device: Number of CUDA device to use for this context
kernel_tuner/backends/nvcuda.py:        :param compiler_options: Compiler options for the CUDA runtime compiler
kernel_tuner/backends/nvcuda.py:        if not cuda:
kernel_tuner/backends/nvcuda.py:                "cuda-python not installed, install using 'pip install cuda-python', or check https://kerneltuner.github.io/kernel_tuner/stable/install.html#cuda-and-pycuda."
kernel_tuner/backends/nvcuda.py:        err = cuda.cuInit(0)
kernel_tuner/backends/nvcuda.py:        cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:        err, self.device = cuda.cuDeviceGet(device)
kernel_tuner/backends/nvcuda.py:        cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:        err, self.context = cuda.cuDevicePrimaryCtxRetain(device)
kernel_tuner/backends/nvcuda.py:        cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:        if CudaFunctions.last_selected_device != device:
kernel_tuner/backends/nvcuda.py:            err = cuda.cuCtxSetCurrent(self.context)
kernel_tuner/backends/nvcuda.py:            cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:            CudaFunctions.last_selected_device = device
kernel_tuner/backends/nvcuda.py:        err, major = cudart.cudaDeviceGetAttribute(
kernel_tuner/backends/nvcuda.py:            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device
kernel_tuner/backends/nvcuda.py:        cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:        err, minor = cudart.cudaDeviceGetAttribute(
kernel_tuner/backends/nvcuda.py:            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device
kernel_tuner/backends/nvcuda.py:        cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:        err, self.max_threads = cudart.cudaDeviceGetAttribute(
kernel_tuner/backends/nvcuda.py:            cudart.cudaDeviceAttr.cudaDevAttrMaxThreadsPerBlock, device
kernel_tuner/backends/nvcuda.py:        cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:        err, self.stream = cuda.cuStreamCreate(0)
kernel_tuner/backends/nvcuda.py:        cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:        err, self.start = cuda.cuEventCreate(0)
kernel_tuner/backends/nvcuda.py:        cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:        err, self.end = cuda.cuEventCreate(0)
kernel_tuner/backends/nvcuda.py:        cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:        self.observers.append(CudaRuntimeObserver(self))
kernel_tuner/backends/nvcuda.py:        err, device_properties = cudart.cudaGetDeviceProperties(device)
kernel_tuner/backends/nvcuda.py:        cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:        env["cuda_version"] = cuda.CUDA_VERSION
kernel_tuner/backends/nvcuda.py:            if isinstance(device_memory, cuda.CUdeviceptr):
kernel_tuner/backends/nvcuda.py:                err = cuda.cuMemFree(device_memory)
kernel_tuner/backends/nvcuda.py:                cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:        """Ready argument list to be passed to the kernel, allocates gpu mem.
kernel_tuner/backends/nvcuda.py:            The order should match the argument list on the CUDA kernel.
kernel_tuner/backends/nvcuda.py:        :returns: A list of arguments that can be passed to an CUDA kernel.
kernel_tuner/backends/nvcuda.py:        :rtype: list( pycuda.driver.DeviceAllocation, numpy.int32, ... )
kernel_tuner/backends/nvcuda.py:        gpu_args = []
kernel_tuner/backends/nvcuda.py:                err, device_memory = cuda.cuMemAlloc(arg.nbytes)
kernel_tuner/backends/nvcuda.py:                cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:                gpu_args.append(device_memory)
kernel_tuner/backends/nvcuda.py:                gpu_args.append(arg)
kernel_tuner/backends/nvcuda.py:        return gpu_args
kernel_tuner/backends/nvcuda.py:        """Call the CUDA compiler to compile the kernel, return the device function.
kernel_tuner/backends/nvcuda.py:        :param kernel_string: The CUDA kernel code that contains the function `kernel_name`
kernel_tuner/backends/nvcuda.py:        :returns: A kernel that can be launched by the CUDA runtime
kernel_tuner/backends/nvcuda.py:        # mimic pycuda behavior to wrap kernel_string in extern "C" if not in kernel_string already
kernel_tuner/backends/nvcuda.py:        if not any([b"--gpu-architecture=" in opt or b"-arch" in opt for opt in compiler_options]):
kernel_tuner/backends/nvcuda.py:                f"--gpu-architecture=compute_{to_valid_nvrtc_gpu_arch_cc(self.cc)}".encode("UTF-8")
kernel_tuner/backends/nvcuda.py:        if not any(["--gpu-architecture=" in opt or "-arch" in opt for opt in self.compiler_options]):
kernel_tuner/backends/nvcuda.py:            self.compiler_options.append(f"--gpu-architecture=compute_{to_valid_nvrtc_gpu_arch_cc(self.cc)}")
kernel_tuner/backends/nvcuda.py:            str.encode(kernel_string), b"CUDAProgram", 0, [], []
kernel_tuner/backends/nvcuda.py:            cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:            cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:            cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:            cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:            err, self.current_module = cuda.cuModuleLoadData(np.char.array(buff))
kernel_tuner/backends/nvcuda.py:            if err == cuda.CUresult.CUDA_ERROR_INVALID_PTX:
kernel_tuner/backends/nvcuda.py:                cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:            err, self.func = cuda.cuModuleGetFunction(
kernel_tuner/backends/nvcuda.py:            cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:            num_regs = cuda.cuFuncGetAttribute(cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, self.func)
kernel_tuner/backends/nvcuda.py:        err = cudart.cudaEventRecord(self.start, self.stream)
kernel_tuner/backends/nvcuda.py:        cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:        err = cudart.cudaEventRecord(self.end, self.stream)
kernel_tuner/backends/nvcuda.py:        cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:        err = cudart.cudaEventQuery(self.end)
kernel_tuner/backends/nvcuda.py:        if err[0] == cudart.cudaError_t.cudaSuccess:
kernel_tuner/backends/nvcuda.py:        err = cudart.cudaDeviceSynchronize()
kernel_tuner/backends/nvcuda.py:        cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:            err, symbol, _ = cuda.cuModuleGetGlobal(self.current_module, str.encode(k))
kernel_tuner/backends/nvcuda.py:            cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:            err = cuda.cuMemcpyHtoD(symbol, v, v.nbytes)
kernel_tuner/backends/nvcuda.py:            cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:        raise NotImplementedError("NVIDIA CUDA backend does not support texture memory")
kernel_tuner/backends/nvcuda.py:    def run_kernel(self, func, gpu_args, threads, grid, stream=None):
kernel_tuner/backends/nvcuda.py:        """Runs the CUDA kernel passed as 'func'.
kernel_tuner/backends/nvcuda.py:        :param func: A CUDA kernel compiled for this specific kernel configuration
kernel_tuner/backends/nvcuda.py:        :type func: cuda.CUfunction
kernel_tuner/backends/nvcuda.py:        :param gpu_args: A list of arguments to the kernel, order should match the
kernel_tuner/backends/nvcuda.py:        :type gpu_args: list( cupy.ndarray, numpy.int32, ...)
kernel_tuner/backends/nvcuda.py:        for arg in gpu_args:
kernel_tuner/backends/nvcuda.py:            if isinstance(arg, cuda.CUdeviceptr):
kernel_tuner/backends/nvcuda.py:        kernel_args = (tuple(gpu_args), tuple(arg_types))
kernel_tuner/backends/nvcuda.py:        err = cuda.cuLaunchKernel(
kernel_tuner/backends/nvcuda.py:        cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:        :param allocation: A GPU memory allocation unit
kernel_tuner/backends/nvcuda.py:        err = cudart.cudaMemset(allocation, value, size)
kernel_tuner/backends/nvcuda.py:        cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:        :param src: A GPU memory allocation unit
kernel_tuner/backends/nvcuda.py:        :type src: cuda.CUdeviceptr
kernel_tuner/backends/nvcuda.py:        err = cuda.cuMemcpyDtoH(dest, src, dest.nbytes)
kernel_tuner/backends/nvcuda.py:        cuda_error_check(err)
kernel_tuner/backends/nvcuda.py:        :param dest: A GPU memory allocation unit
kernel_tuner/backends/nvcuda.py:        :type dest: cuda.CUdeviceptr
kernel_tuner/backends/nvcuda.py:        err = cuda.cuMemcpyHtoD(dest, src, src.nbytes)
kernel_tuner/backends/nvcuda.py:        cuda_error_check(err)
kernel_tuner/backends/pycuda.py:"""This module contains all CUDA specific kernel_tuner functions."""
kernel_tuner/backends/pycuda.py:from kernel_tuner.backends.backend import GPUBackend
kernel_tuner/backends/pycuda.py:from kernel_tuner.observers.pycuda import PyCudaRuntimeObserver
kernel_tuner/backends/pycuda.py:# and run tests without pycuda installed
kernel_tuner/backends/pycuda.py:    import pycuda.driver as drv
kernel_tuner/backends/pycuda.py:    pycuda_available = True
kernel_tuner/backends/pycuda.py:    class PyCudaPlaceHolder:
kernel_tuner/backends/pycuda.py:    drv = PyCudaPlaceHolder()
kernel_tuner/backends/pycuda.py:    pycuda_available = False
kernel_tuner/backends/pycuda.py:    from pycuda.compiler import SourceModule
kernel_tuner/backends/pycuda.py:    from pycuda.compiler import DynamicSourceModule
kernel_tuner/backends/pycuda.py:    """class to interoperate torch device memory allocations with PyCUDA."""
kernel_tuner/backends/pycuda.py:        self.gpudata = tensor.data_ptr()
kernel_tuner/backends/pycuda.py:class PyCudaFunctions(GPUBackend):
kernel_tuner/backends/pycuda.py:    """Class that groups the CUDA functions on maintains state about the device."""
kernel_tuner/backends/pycuda.py:        """Instantiate PyCudaFunctions object used for interacting with the CUDA device.
kernel_tuner/backends/pycuda.py:        :param device: Number of CUDA device to use for this context
kernel_tuner/backends/pycuda.py:        # if not PyCuda available, check if mocking before raising exception
kernel_tuner/backends/pycuda.py:        if not pycuda_available and isinstance(drv, PyCudaPlaceHolder):
kernel_tuner/backends/pycuda.py:                "pycuda not installed, install using 'pip install pycuda', or check https://kerneltuner.github.io/kernel_tuner/stable/install.html#cuda-and-pycuda."
kernel_tuner/backends/pycuda.py:        if PyCudaFunctions.last_selected_device != device:
kernel_tuner/backends/pycuda.py:            # pycuda does not wrap cuCtxSetCurrent.
kernel_tuner/backends/pycuda.py:            if PyCudaFunctions.last_selected_context is not None:
kernel_tuner/backends/pycuda.py:                PyCudaFunctions.last_selected_context.pop()
kernel_tuner/backends/pycuda.py:                    PyCudaFunctions.last_selected_context.pop()
kernel_tuner/backends/pycuda.py:            PyCudaFunctions.last_selected_device = device
kernel_tuner/backends/pycuda.py:            PyCudaFunctions.last_selected_context = self.context
kernel_tuner/backends/pycuda.py:        # select PyCUDA source module
kernel_tuner/backends/pycuda.py:                "Error: pycuda not correctly installed, please ensure pycuda is installed on the same CUDA installation as you're using right now"
kernel_tuner/backends/pycuda.py:        self.observers.append(PyCudaRuntimeObserver(self))
kernel_tuner/backends/pycuda.py:        env["cuda_version"] = ".".join([str(i) for i in drv.get_version()])
kernel_tuner/backends/pycuda.py:        for gpu_mem in self.allocations:
kernel_tuner/backends/pycuda.py:            if hasattr(gpu_mem, "free"):
kernel_tuner/backends/pycuda.py:                gpu_mem.free()
kernel_tuner/backends/pycuda.py:        """Ready argument list to be passed to the kernel, allocates gpu mem.
kernel_tuner/backends/pycuda.py:            The order should match the argument list on the CUDA kernel.
kernel_tuner/backends/pycuda.py:        :returns: A list of arguments that can be passed to an CUDA kernel.
kernel_tuner/backends/pycuda.py:        :rtype: list( pycuda.driver.DeviceAllocation, numpy.int32, ... )
kernel_tuner/backends/pycuda.py:        gpu_args = []
kernel_tuner/backends/pycuda.py:                gpu_args.append(alloc)
kernel_tuner/backends/pycuda.py:                drv.memcpy_htod(gpu_args[-1], arg)
kernel_tuner/backends/pycuda.py:                if arg.is_cuda:
kernel_tuner/backends/pycuda.py:                    gpu_args.append(Holder(arg))
kernel_tuner/backends/pycuda.py:                    gpu_args.append(Holder(arg.cuda()))
kernel_tuner/backends/pycuda.py:            # pycuda does not support bool, convert to uint8 instead
kernel_tuner/backends/pycuda.py:                gpu_args.append(arg.astype(np.uint8))
kernel_tuner/backends/pycuda.py:                gpu_args.append(arg)
kernel_tuner/backends/pycuda.py:        return gpu_args
kernel_tuner/backends/pycuda.py:        """Call the CUDA compiler to compile the kernel, return the device function.
kernel_tuner/backends/pycuda.py:        :param kernel_string: The CUDA kernel code that contains the function `kernel_name`
kernel_tuner/backends/pycuda.py:        :returns: An CUDA kernel that can be called directly.
kernel_tuner/backends/pycuda.py:        :rtype: pycuda.driver.Function
kernel_tuner/backends/pycuda.py:    def run_kernel(self, func, gpu_args, threads, grid, stream=None):
kernel_tuner/backends/pycuda.py:        """Runs the CUDA kernel passed as 'func'.
kernel_tuner/backends/pycuda.py:        :param func: A PyCuda kernel compiled for this specific kernel configuration
kernel_tuner/backends/pycuda.py:        :type func: pycuda.driver.Function
kernel_tuner/backends/pycuda.py:        :param gpu_args: A list of arguments to the kernel, order should match the
kernel_tuner/backends/pycuda.py:        :type gpu_args: list( pycuda.driver.DeviceAllocation, numpy.int32, ...)
kernel_tuner/backends/pycuda.py:            *gpu_args,
kernel_tuner/backends/pycuda.py:        :param allocation: A GPU memory allocation unit
kernel_tuner/backends/pycuda.py:        :type allocation: pycuda.driver.DeviceAllocation
kernel_tuner/backends/pycuda.py:        :param src: A GPU memory allocation unit
kernel_tuner/backends/pycuda.py:        :type src: pycuda.driver.DeviceAllocation
kernel_tuner/backends/pycuda.py:        :param dest: A GPU memory allocation unit
kernel_tuner/backends/pycuda.py:        :type dest: pycuda.driver.DeviceAllocation
kernel_tuner/backends/hip.py:from kernel_tuner.backends.backend import GPUBackend
kernel_tuner/backends/hip.py:class HipFunctions(GPUBackend):
kernel_tuner/backends/hip.py:    def run_kernel(self, func, gpu_args, threads, grid, stream=None):
kernel_tuner/backends/hip.py:        :param gpu_args: A ctypes structure of arguments to the kernel, order should match the
kernel_tuner/backends/hip.py:        :type gpu_args: ctypes structure
kernel_tuner/backends/hip.py:        field_types = [type(x) for x in gpu_args]
kernel_tuner/backends/hip.py:        ctype_args = ArgListStructure(*gpu_args)
kernel_tuner/backends/hip.py:        :param allocation: A GPU memory allocation unit
kernel_tuner/backends/hip.py:        :param src: A GPU memory allocation unit
kernel_tuner/backends/hip.py:        :param dest: A GPU memory allocation unit
kernel_tuner/backends/cupy.py:from kernel_tuner.backends.backend import GPUBackend
kernel_tuner/backends/cupy.py:class CupyFunctions(GPUBackend):
kernel_tuner/backends/cupy.py:        """Instantiate CupyFunctions object used for interacting with the CUDA device.
kernel_tuner/backends/cupy.py:        :param device: Number of CUDA device to use for this context
kernel_tuner/backends/cupy.py:                "cupy not installed, install using 'pip install cupy', or check https://kerneltuner.github.io/kernel_tuner/stable/install.html#cuda-and-pycuda."
kernel_tuner/backends/cupy.py:        self.dev = dev = cp.cuda.Device(device)
kernel_tuner/backends/cupy.py:        self.stream = cp.cuda.Stream()
kernel_tuner/backends/cupy.py:        self.start = cp.cuda.Event()
kernel_tuner/backends/cupy.py:        self.end = cp.cuda.Event()
kernel_tuner/backends/cupy.py:        env["cuda_version"] = cp.cuda.runtime.driverGetVersion()
kernel_tuner/backends/cupy.py:        """Ready argument list to be passed to the kernel, allocates gpu mem.
kernel_tuner/backends/cupy.py:            The order should match the argument list on the CUDA kernel.
kernel_tuner/backends/cupy.py:        :returns: A list of arguments that can be passed to an CUDA kernel.
kernel_tuner/backends/cupy.py:        gpu_args = []
kernel_tuner/backends/cupy.py:                gpu_args.append(alloc)
kernel_tuner/backends/cupy.py:                gpu_args.append(arg)
kernel_tuner/backends/cupy.py:        return gpu_args
kernel_tuner/backends/cupy.py:        """Call the CUDA compiler to compile the kernel, return the device function.
kernel_tuner/backends/cupy.py:        :param kernel_string: The CUDA kernel code that contains the function `kernel_name`
kernel_tuner/backends/cupy.py:        :returns: An CUDA kernel that can be called directly.
kernel_tuner/backends/cupy.py:        # CuPy already sets the --gpu-architecture by itself, as per https://github.com/cupy/cupy/blob/main/cupy/cuda/compiler.py#L145
kernel_tuner/backends/cupy.py:    def run_kernel(self, func, gpu_args, threads, grid, stream=None):
kernel_tuner/backends/cupy.py:        """Runs the CUDA kernel passed as 'func'.
kernel_tuner/backends/cupy.py:        :param gpu_args: A list of arguments to the kernel, order should match the
kernel_tuner/backends/cupy.py:        :type gpu_args: list( cupy.ndarray, numpy.int32, ...)
kernel_tuner/backends/cupy.py:        func(grid, threads, gpu_args, stream=stream, shared_mem=self.smem_size)
kernel_tuner/backends/cupy.py:        :param allocation: A GPU memory allocation unit
kernel_tuner/backends/cupy.py:        :param src: A GPU memory allocation unit
kernel_tuner/backends/cupy.py:        :param dest: A GPU memory allocation unit
kernel_tuner/backends/backend.py:    def run_kernel(self, func, gpu_args, threads, grid, stream):
kernel_tuner/backends/backend.py:class GPUBackend(Backend):
kernel_tuner/backends/backend.py:    """Base class for GPU backends"""
kernel_tuner/backends/backend.py:        """This method must implement the allocation and copy of constant memory to the GPU."""
kernel_tuner/backends/backend.py:        """This method must implement the dynamic allocation of shared memory on the GPU."""
kernel_tuner/backends/backend.py:        """This method must implement the allocation and copy of texture memory to the GPU."""
kernel_tuner/backends/compiler.py:    This function is used to implement CPU/GPU generic code. If the cupy module can be imported
kernel_tuner/backends/compiler.py:        self.using_openacc = False
kernel_tuner/backends/compiler.py:        # detect openacc
kernel_tuner/backends/compiler.py:            self.using_openacc = True
kernel_tuner/backends/compiler.py:            ((suffix == ".cu") or ("#include <cuda" in kernel_string) or ("cudaMemcpy" in kernel_string))
kernel_tuner/backends/compiler.py:        # at the moment any C, C++, or CUDA code is assumed to use extern "C" linkage
kernel_tuner/backends/compiler.py:            lib_args = ["-lOpenCL"]
kernel_tuner/backends/compiler.py:            cp.cuda.runtime.memset(allocation.numpy.data.ptr, value, size)
kernel_tuner/backends/compiler.py:        if not self.using_openmp and not self.using_openacc:
kernel_tuner/backends/opencl.py:"""This module contains all OpenCL specific kernel_tuner functions."""
kernel_tuner/backends/opencl.py:from kernel_tuner.backends.backend import GPUBackend
kernel_tuner/backends/opencl.py:from kernel_tuner.observers.opencl import OpenCLObserver
kernel_tuner/backends/opencl.py:    import pyopencl as cl
kernel_tuner/backends/opencl.py:class OpenCLFunctions(GPUBackend):
kernel_tuner/backends/opencl.py:    """Class that groups the OpenCL functions on maintains some state about the device."""
kernel_tuner/backends/opencl.py:        """Creates OpenCL device context and reads device properties.
kernel_tuner/backends/opencl.py:        :param device: The ID of the OpenCL device to use for benchmarking
kernel_tuner/backends/opencl.py:                "pyopencl not installed, install using 'pip install pyopencl', or check https://kerneltuner.github.io/kernel_tuner/stable/install.html#opencl-and-pyopencl."
kernel_tuner/backends/opencl.py:        self.observers.append(OpenCLObserver(self))
kernel_tuner/backends/opencl.py:        env["opencl_c_version"] = dev.opencl_c_version
kernel_tuner/backends/opencl.py:        """Ready argument list to be passed to the kernel, allocates gpu mem.
kernel_tuner/backends/opencl.py:            The order should match the argument list on the OpenCL kernel.
kernel_tuner/backends/opencl.py:        :returns: A list of arguments that can be passed to an OpenCL kernel.
kernel_tuner/backends/opencl.py:        :rtype: list( pyopencl.Buffer, numpy.int32, ... )
kernel_tuner/backends/opencl.py:        gpu_args = []
kernel_tuner/backends/opencl.py:                gpu_args.append(
kernel_tuner/backends/opencl.py:                gpu_args.append(arg)
kernel_tuner/backends/opencl.py:        return gpu_args
kernel_tuner/backends/opencl.py:        """Call the OpenCL compiler to compile the kernel, return the device function.
kernel_tuner/backends/opencl.py:        :param kernel_string: The OpenCL kernel code that contains the function `kernel_name`
kernel_tuner/backends/opencl.py:        :returns: An OpenCL kernel that can be called directly.
kernel_tuner/backends/opencl.py:        :rtype: pyopencl.Kernel
kernel_tuner/backends/opencl.py:        In OpenCL the event is created when the kernel is launched
kernel_tuner/backends/opencl.py:        In OpenCL the event is created when the kernel is launched
kernel_tuner/backends/opencl.py:    def run_kernel(self, func, gpu_args, threads, grid):
kernel_tuner/backends/opencl.py:        """Runs the OpenCL kernel passed as 'func'.
kernel_tuner/backends/opencl.py:        :param func: An OpenCL Kernel
kernel_tuner/backends/opencl.py:        :type func: pyopencl.Kernel
kernel_tuner/backends/opencl.py:        :param gpu_args: A list of arguments to the kernel, order should match the
kernel_tuner/backends/opencl.py:        :type gpu_args: list( pyopencl.Buffer, numpy.int32, ...)
kernel_tuner/backends/opencl.py:        self.event = func(self.queue, global_size, local_size, *gpu_args)
kernel_tuner/backends/opencl.py:        :param allocation: An OpenCL Buffer to fill
kernel_tuner/backends/opencl.py:        :type allocation: pyopencl.Buffer
kernel_tuner/backends/opencl.py:        :param src: An OpenCL Buffer to copy data from
kernel_tuner/backends/opencl.py:        :type src: pyopencl.Buffer
kernel_tuner/backends/opencl.py:        :param dest: An OpenCL Buffer to copy data from
kernel_tuner/backends/opencl.py:        :type dest: pyopencl.Buffer
kernel_tuner/backends/opencl.py:        raise NotImplementedError("PyOpenCL backend does not support constant memory")
kernel_tuner/backends/opencl.py:        raise NotImplementedError("PyOpenCL backend does not support shared memory")
kernel_tuner/backends/opencl.py:        raise NotImplementedError("PyOpenCL backend does not support texture memory")
kernel_tuner/interface.py:                """The CUDA, OpenCL, HIP, or C kernel code.
kernel_tuner/interface.py:                """Specifies the language used for GPU kernels. The kernel_tuner
kernel_tuner/interface.py:        the language using this argument, currently supported: "CUDA", "Cupy",
kernel_tuner/interface.py:        "OpenCL", "HIP", or "C".""",
kernel_tuner/interface.py:            See the reduction CUDA example for an example use of this feature.""",
kernel_tuner/interface.py:                """CUDA-specific feature for specifying shared memory options
kernel_tuner/interface.py:            shared memory configuration on Kepler GPUs for example could be added
kernel_tuner/interface.py:                """CUDA-specific feature for specifying constant memory
kernel_tuner/interface.py:            arguments to the kernel. In OpenCL these are handled as normal
kernel_tuner/interface.py:            kernel arguments, but in CUDA you can copy to a symbol. The way you
kernel_tuner/interface.py:                """CUDA-specific feature for specifying texture memory
kernel_tuner/interface.py:            may use the built-in variables blockDim.xyz in CUDA or the
kernel_tuner/interface.py:            built-in function get_local_size() in OpenCL instead.""",
kernel_tuner/interface.py:                """CUDA/OpenCL device to use, in case you have multiple
kernel_tuner/interface.py:        CUDA-capable GPUs or OpenCL devices you may use this to select one,
kernel_tuner/interface.py:                """OpenCL platform to use, in case you have multiple
kernel_tuner/interface.py:        OpenCL platforms you may use this to select one,
kernel_tuner/interface.py:        0 by default. Ignored if not using OpenCL. """,
kernel_tuner/interface.py:    """ Tune a CUDA kernel given a set of tunable parameters
kernel_tuner/interface.py:    after execution on the GPU.
kernel_tuner/interface.py:     * Allocate GPU memory to hold all kernel arguments
kernel_tuner/interface.py:     * Move the all data to the GPU
kernel_tuner/interface.py:     * Execute the kernel on the GPU
kernel_tuner/interface.py:     * Copy all data from the GPU back to the host and return it as a list of Numpy arrays
kernel_tuner/interface.py:    # Preprocess GPU arguments. Require for handling `Tunable` arguments
kernel_tuner/interface.py:    arguments = dev.preprocess_gpu_arguments(arguments, params)
kernel_tuner/interface.py:    # move data to the GPU
kernel_tuner/interface.py:    gpu_args = dev.ready_argument_list(arguments)
kernel_tuner/interface.py:    if not dev.run_kernel(func, gpu_args, instance):
kernel_tuner/interface.py:    # copy data in GPU memory back to the host
kernel_tuner/interface.py:            dev.memcpy_dtoh(results[-1], gpu_args[i])
kernel_tuner/observers/nvcuda.py:    from cuda import cudart
kernel_tuner/observers/nvcuda.py:    cuda = None
kernel_tuner/observers/nvcuda.py:from kernel_tuner.util import cuda_error_check
kernel_tuner/observers/nvcuda.py:class CudaRuntimeObserver(BenchmarkObserver):
kernel_tuner/observers/nvcuda.py:    """Observer that measures time using CUDA events during benchmarking"""
kernel_tuner/observers/nvcuda.py:        err, time = cudart.cudaEventElapsedTime(self.start, self.end)
kernel_tuner/observers/nvcuda.py:        cuda_error_check(err)
kernel_tuner/observers/pycuda.py:class PyCudaRuntimeObserver(BenchmarkObserver):
kernel_tuner/observers/pycuda.py:    """Observer that measures time using CUDA events during benchmarking"""
kernel_tuner/observers/tegra.py:        """Create object to control GPU core clock on a Tegra device."""
kernel_tuner/observers/tegra.py:            self.gpu_temp_path = self.get_temp_path()
kernel_tuner/observers/tegra.py:            self.gpu_temp_path = temp_path
kernel_tuner/observers/tegra.py:            self.gpu_power_path = self.get_power_path()
kernel_tuner/observers/tegra.py:            self.gpu_power_path = power_path
kernel_tuner/observers/tegra.py:        self.gpu_channel = self.get_gpu_channel()
kernel_tuner/observers/tegra.py:        # loop to find GPU device name based on jetson_clocks
kernel_tuner/observers/tegra.py:            if name in ("gv11b", "gp10b", "ga10b", "gpu"):
kernel_tuner/observers/tegra.py:            raise FileNotFoundError("No internal tegra GPU found")
kernel_tuner/observers/tegra.py:        """Find the file which holds the GPU temperature"""
kernel_tuner/observers/tegra.py:            if name == "GPU-therm":
kernel_tuner/observers/tegra.py:                gpu_temp_path = str(zone)
kernel_tuner/observers/tegra.py:        if gpu_temp_path is None:
kernel_tuner/observers/tegra.py:            raise FileNotFoundError("No GPU sensor for temperature found")
kernel_tuner/observers/tegra.py:        return gpu_temp_path
kernel_tuner/observers/tegra.py:    def get_gpu_channel(self):
kernel_tuner/observers/tegra.py:        """Get the channel number of the sensor which measures the GPU power"""
kernel_tuner/observers/tegra.py:        # find the channel which holds GPU power information
kernel_tuner/observers/tegra.py:        for channel_dir in Path(self.gpu_power_path + "/of_node/").iterdir():
kernel_tuner/observers/tegra.py:                if "GPU" in channel_label:
kernel_tuner/observers/tegra.py:        # If this statement is reached, no channel for the GPU was found
kernel_tuner/observers/tegra.py:        raise FileNotFoundError("No channel found with GPU power readings")
kernel_tuner/observers/tegra.py:    def read_gpu_temp(self):
kernel_tuner/observers/tegra.py:        """Read GPU temperature"""
kernel_tuner/observers/tegra.py:        with open(self.gpu_temp_path + "/temp") as fp:
kernel_tuner/observers/tegra.py:    def read_gpu_power(self):
kernel_tuner/observers/tegra.py:        result_cur = subprocess.run(["sudo", "cat", f"{self.gpu_power_path}/curr{self.gpu_channel}_input"], capture_output=True, text=True)
kernel_tuner/observers/tegra.py:        result_vol = subprocess.run(["sudo", "cat", f"{self.gpu_power_path}/in{self.gpu_channel}_input"], capture_output=True, text=True)
kernel_tuner/observers/tegra.py:        return self.tegra.read_gpu_power()
kernel_tuner/observers/tegra.py:        if "gpu_temp" in self.observables:
kernel_tuner/observers/tegra.py:            self.iteration["gpu_temp"].append(self.tegra.read_gpu_temp())
kernel_tuner/observers/tegra.py:        if "gpu_temp" in self.observables:
kernel_tuner/observers/tegra.py:            self.results["gpu_temps"].append(np.average(self.iteration["gpu_temp"]))
kernel_tuner/observers/observer.py:        # only store the result if we get a new measurement from the GPU
kernel_tuner/observers/hip.py:    """Observer that measures time using CUDA events during benchmarking."""
kernel_tuner/observers/pmt.py:          instance "/dev/ttyACM0". For nvml, it should correspond to the GPU
kernel_tuner/observers/pmt.py:        accuracy when using internal power sensors, such as NVML or ROCM,
kernel_tuner/observers/pmt.py:        supported = ["powersensor2", "powersensor3", "nvidia", "likwid", "rapl", "rocm", "xilinx"]
kernel_tuner/observers/nvml.py:    def __init__(self, device_id=0, nvidia_smi_fallback="nvidia-smi", use_locked_clocks=False):
kernel_tuner/observers/nvml.py:        self.nvidia_smi = nvidia_smi_fallback
kernel_tuner/observers/nvml.py:            if self.nvidia_smi:
kernel_tuner/observers/nvml.py:                # nvidia-smi expects Watts rather than milliwatts
kernel_tuner/observers/nvml.py:                    self.nvidia_smi,
kernel_tuner/observers/nvml.py:                self.nvidia_smi,
kernel_tuner/observers/nvml.py:                pynvml.nvmlDeviceSetGpuLockedClocks(self.dev, gr_clock, gr_clock)
kernel_tuner/observers/nvml.py:                if self.nvidia_smi:
kernel_tuner/observers/nvml.py:                    args = ["sudo", self.nvidia_smi, "-i", str(self.id)]
kernel_tuner/observers/nvml.py:                    command_set_gpu_clocks = f"--lock-gpu-clocks={str(gr_clock)},{str(gr_clock)}"
kernel_tuner/observers/nvml.py:                    subprocess.run(args + [command_set_gpu_clocks], check=True)
kernel_tuner/observers/nvml.py:                if self.nvidia_smi:
kernel_tuner/observers/nvml.py:                    args = ["sudo", self.nvidia_smi, "-i", str(self.id)]
kernel_tuner/observers/nvml.py:                pynvml.nvmlDeviceResetGpuLockedClocks(self.dev)
kernel_tuner/observers/nvml.py:                if self.nvidia_smi:
kernel_tuner/observers/nvml.py:                        self.nvidia_smi,
kernel_tuner/observers/nvml.py:                        "--reset-gpu-clocks",
kernel_tuner/observers/nvml.py:                        self.nvidia_smi,
kernel_tuner/observers/nvml.py:        """Get the GPU temperature."""
kernel_tuner/observers/nvml.py:        return pynvml.nvmlDeviceGetTemperature(self.dev, pynvml.NVML_TEMPERATURE_GPU)
kernel_tuner/observers/nvml.py:        args = ["nvidia-smi", "-i", str(self.id), "-q", "-d", "VOLTAGE"]
kernel_tuner/observers/nvml.py:        consumption of a GPU kernel executing on the GPU use "nvml_power". The "power_readings" are the individual power readings
kernel_tuner/observers/nvml.py:    :param device: Device ordinal used by Nvidia to identify your device, same as reported by nvidia-smi.
kernel_tuner/observers/nvml.py:    :param nvidia_smi_fallback: String with the location of your nvidia-smi executable to use when Python cannot execute with root privileges, default None.
kernel_tuner/observers/nvml.py:    :type nvidia_smi_fallback: string
kernel_tuner/observers/nvml.py:    :param use_locked_clocks: Boolean to opt in to using the locked clocks feature on Ampere or newer GPUs.
kernel_tuner/observers/nvml.py:        will set the GPU clocks using the application clocks feature.
kernel_tuner/observers/nvml.py:        nvidia_smi_fallback=None,
kernel_tuner/observers/nvml.py:        if nvidia_smi_fallback:
kernel_tuner/observers/nvml.py:                nvidia_smi_fallback=nvidia_smi_fallback,
kernel_tuner/observers/cupy.py:    """Observer that measures time using CUDA events during benchmarking in the CuPy backend"""
kernel_tuner/observers/cupy.py:        self.times.append(cp.cuda.get_elapsed_time(self.start, self.end))
kernel_tuner/observers/ncu.py:        The exact performance counters supported differ per GPU, some examples:
kernel_tuner/observers/opencl.py:class OpenCLObserver(BenchmarkObserver):
kernel_tuner/observers/opencl.py:    """Observer that measures time using CUDA events during benchmarking"""
kernel_tuner/energy/energy.py:    import pycuda.driver as drv
kernel_tuner/energy/energy.py:def get_frequency_power_relation_fp32(device, n_samples=10, nvidia_smi_fallback=None, use_locked_clocks=False, cache=None, simulation_mode=None):
kernel_tuner/energy/energy.py:    """Use NVML and PyCUDA with a synthetic kernel to obtain samples of frequency-power pairs."""
kernel_tuner/energy/energy.py:            raise ImportError("get_ridge_point_gr_frequency requires PyCUDA")
kernel_tuner/energy/energy.py:        ["core_freq", "nvml_power"], device=device, nvidia_smi_fallback=nvidia_smi_fallback, use_locked_clocks=use_locked_clocks)
kernel_tuner/energy/energy.py:def create_power_frequency_model(device=0, n_samples=10, verbose=False, nvidia_smi_fallback=None, use_locked_clocks=False, cache=None, simulation_mode=None):
kernel_tuner/energy/energy.py:     * Going green: optimizing GPUs for energy efficiency through model-steered auto-tuning
kernel_tuner/energy/energy.py:    Requires NVML and PyCUDA.
kernel_tuner/energy/energy.py:    :param nvidia_smi_fallback: Path to nvidia-smi when insufficient permissions to use NVML directly
kernel_tuner/energy/energy.py:    :type nvidia_smi_fallback: string
kernel_tuner/energy/energy.py:    freqs, nvml_power = get_frequency_power_relation_fp32(device, n_samples, nvidia_smi_fallback, use_locked_clocks, cache=cache, simulation_mode=simulation_mode)
kernel_tuner/kernelbuilder.py:            This object compiles a GPU kernel parameterized using the parameters in params.
kernel_tuner/kernelbuilder.py:            GPU memory is allocated for each argument using its size and type as listed in arguments.
kernel_tuner/kernelbuilder.py:            Kernel arguments marked as inputs will be copied to the GPU on every kernel launch.
kernel_tuner/kernelbuilder.py:        #setup GPU memory
kernel_tuner/kernelbuilder.py:        self.gpu_args = self.dev.ready_argument_list(arguments)
kernel_tuner/kernelbuilder.py:    def update_gpu_args(self, args):
kernel_tuner/kernelbuilder.py:                    self.dev.dev.memcpy_htod(self.gpu_args[i], arg)
kernel_tuner/kernelbuilder.py:                    self.gpu_args[i] = arg
kernel_tuner/kernelbuilder.py:        return self.gpu_args
kernel_tuner/kernelbuilder.py:    def get_gpu_result(self, args):
kernel_tuner/kernelbuilder.py:        for i, _ in enumerate(self.gpu_args):
kernel_tuner/kernelbuilder.py:                self.dev.memcpy_dtoh(res, self.gpu_args[i])
kernel_tuner/kernelbuilder.py:        """Run the GPU kernel
kernel_tuner/kernelbuilder.py:        Copy the arguments marked as inputs to the GPU
kernel_tuner/kernelbuilder.py:        Call the GPU kernel
kernel_tuner/kernelbuilder.py:        Copy the arguments marked as outputs from the GPU
kernel_tuner/kernelbuilder.py:        self.update_gpu_args(args)
kernel_tuner/kernelbuilder.py:        self.dev.run_kernel(self.func, self.gpu_args, self.kernel_instance)
kernel_tuner/kernelbuilder.py:        return self.get_gpu_result(args)
kernel_tuner/kernelbuilder.py:        """Run the GPU kernel
kernel_tuner/kernelbuilder.py:        Copy the arguments marked as inputs to the GPU
kernel_tuner/kernelbuilder.py:        Call the GPU kernel
kernel_tuner/kernelbuilder.py:        Copy the arguments marked as outputs from the GPU
kernel_tuner/util.py:    from cuda import cuda, cudart, nvrtc
kernel_tuner/util.py:    cuda = None
kernel_tuner/util.py:        "int32": ["int", "int32_t"],  # discrepancy between OpenCL and C here, long may be 32bits in C
kernel_tuner/util.py:        lang = "CUDA"
kernel_tuner/util.py:        lang = "OpenCL"
kernel_tuner/util.py:def to_valid_nvrtc_gpu_arch_cc(compute_capability: str) -> str:
kernel_tuner/util.py:    """Returns a valid Compute Capability for NVRTC `--gpu-architecture=`, as per https://docs.nvidia.com/cuda/nvrtc/index.html#group__options."""
kernel_tuner/util.py:        # string must contain substring ".c", ".opencl", or ".F"
kernel_tuner/util.py:        result = result and any([s in kernel_source for s in (".c", ".opencl", ".F")])
kernel_tuner/util.py:        if "loop_unroll_factor" in k and lang in ("CUDA", "HIP"):
kernel_tuner/util.py:            # this handles the special case that in CUDA/HIP
kernel_tuner/util.py:            # in OpenCL this isn't the case and we can just insert "#define loop_unroll_factor N"
kernel_tuner/util.py:def cuda_error_check(error):
kernel_tuner/util.py:    """Checking the status of CUDA calls using the NVIDIA cuda-python backend."""
kernel_tuner/util.py:    if isinstance(error, cuda.CUresult):
kernel_tuner/util.py:        if error != cuda.CUresult.CUDA_SUCCESS:
kernel_tuner/util.py:            _, name = cuda.cuGetErrorName(error)
kernel_tuner/util.py:            raise RuntimeError(f"CUDA error: {name.decode()}")
kernel_tuner/util.py:    elif isinstance(error, cudart.cudaError_t):
kernel_tuner/util.py:        if error != cudart.cudaError_t.cudaSuccess:
kernel_tuner/util.py:            _, name = cudart.getErrorName(error)
kernel_tuner/util.py:            raise RuntimeError(f"CUDART error: {name.decode()}")
kernel_tuner/utils/directives.py:class OpenACC(Directive):
kernel_tuner/utils/directives.py:    """Class to represent OpenACC"""
kernel_tuner/utils/directives.py:        return "openacc"
kernel_tuner/utils/directives.py:def is_openacc(directive: Directive) -> bool:
kernel_tuner/utils/directives.py:    """Check if a directive is OpenACC"""
kernel_tuner/utils/directives.py:    return isinstance(directive, OpenACC)
kernel_tuner/utils/directives.py:def line_contains_openacc_directive(line: str, lang: Language) -> bool:
kernel_tuner/utils/directives.py:    """Check if line contains an OpenACC directive or not"""
kernel_tuner/utils/directives.py:        return line_contains_openacc_directive_cxx(line)
kernel_tuner/utils/directives.py:        return line_contains_openacc_directive_fortran(line)
kernel_tuner/utils/directives.py:def line_contains_openacc_directive_cxx(line: str) -> bool:
kernel_tuner/utils/directives.py:    """Check if a line of code contains a C++ OpenACC directive or not"""
kernel_tuner/utils/directives.py:def line_contains_openacc_directive_fortran(line: str) -> bool:
kernel_tuner/utils/directives.py:    """Check if a line of code contains a Fortran OpenACC directive or not"""
kernel_tuner/utils/directives.py:def line_contains_openacc_parallel_directive(line: str, lang: Language) -> bool:
kernel_tuner/utils/directives.py:    """Check if line contains an OpenACC parallel directive or not"""
kernel_tuner/utils/directives.py:        return line_contains_openacc_parallel_directive_cxx(line)
kernel_tuner/utils/directives.py:        return line_contains_openacc_parallel_directive_fortran(line)
kernel_tuner/utils/directives.py:def line_contains_openacc_parallel_directive_cxx(line: str) -> bool:
kernel_tuner/utils/directives.py:    """Check if a line of code contains a C++ OpenACC parallel directive or not"""
kernel_tuner/utils/directives.py:def line_contains_openacc_parallel_directive_fortran(line: str) -> bool:
kernel_tuner/utils/directives.py:    """Check if a line of code contains a Fortran OpenACC parallel directive or not"""
kernel_tuner/utils/directives.py:def openacc_directive_contains_clause(line: str, clauses: list) -> bool:
kernel_tuner/utils/directives.py:    """Check if an OpenACC directive contains one clause from a list"""
kernel_tuner/utils/directives.py:def openacc_directive_contains_data_clause(line: str) -> bool:
kernel_tuner/utils/directives.py:    """Check if an OpenACC directive contains one data clause"""
kernel_tuner/utils/directives.py:    return openacc_directive_contains_clause(line, data_clauses)
kernel_tuner/utils/directives.py:def create_data_directive_openacc(name: str, size: ArraySize, lang: Language) -> str:
kernel_tuner/utils/directives.py:        return create_data_directive_openacc_cxx(name, size)
kernel_tuner/utils/directives.py:        return create_data_directive_openacc_fortran(name, size)
kernel_tuner/utils/directives.py:def create_data_directive_openacc_cxx(name: str, size: ArraySize) -> str:
kernel_tuner/utils/directives.py:    """Create C++ OpenACC code to allocate and copy data"""
kernel_tuner/utils/directives.py:def create_data_directive_openacc_fortran(name: str, size: ArraySize) -> str:
kernel_tuner/utils/directives.py:    """Create Fortran OpenACC code to allocate and copy data"""
kernel_tuner/utils/directives.py:def exit_data_directive_openacc(name: str, size: ArraySize, lang: Language) -> str:
kernel_tuner/utils/directives.py:        return exit_data_directive_openacc_cxx(name, size)
kernel_tuner/utils/directives.py:        return exit_data_directive_openacc_fortran(name, size)
kernel_tuner/utils/directives.py:def exit_data_directive_openacc_cxx(name: str, size: ArraySize) -> str:
kernel_tuner/utils/directives.py:    """Create C++ OpenACC code to copy back data"""
kernel_tuner/utils/directives.py:def exit_data_directive_openacc_fortran(name: str, size: ArraySize) -> str:
kernel_tuner/utils/directives.py:    """Create Fortran OpenACC code to copy back data"""
kernel_tuner/utils/directives.py:            if is_openacc(langs.directive) and is_cxx(langs.language):
kernel_tuner/utils/directives.py:                intro += create_data_directive_openacc_cxx(name, size)
kernel_tuner/utils/directives.py:                outro += exit_data_directive_openacc_cxx(name, size)
kernel_tuner/utils/directives.py:            elif is_openacc(langs.directive) and is_fortran(langs.language):
kernel_tuner/utils/directives.py:                intro += create_data_directive_openacc_fortran(name, size)
kernel_tuner/utils/directives.py:                outro += exit_data_directive_openacc_fortran(name, size)
kernel_tuner/utils/directives.py:        body = add_present_openacc(body, langs, data, preprocessor, user_dimensions)
kernel_tuner/utils/directives.py:def add_present_openacc(
kernel_tuner/utils/directives.py:    """Add the present clause to OpenACC directive"""
kernel_tuner/utils/directives.py:        if not line_contains_openacc_parallel_directive(line, langs.language):
kernel_tuner/utils/directives.py:            # The line contains an OpenACC directive
kernel_tuner/utils/directives.py:            if openacc_directive_contains_data_clause(line):
kernel_tuner/utils/directives.py:                # The OpenACC directive manages memory, do not interfere
kernel_tuner/utils/directives.py:                            present_clause += add_present_openacc_cxx(name, size)
kernel_tuner/utils/directives.py:                            present_clause += add_present_openacc_fortran(name, size)
kernel_tuner/utils/directives.py:def add_present_openacc_cxx(name: str, size: ArraySize) -> str:
kernel_tuner/utils/directives.py:    """Create present clause for C++ OpenACC directive"""
kernel_tuner/utils/directives.py:def add_present_openacc_fortran(name: str, size: ArraySize) -> str:
kernel_tuner/utils/directives.py:    """Create present clause for Fortran OpenACC directive"""
kernel_tuner/runners/sequential.py:        #move data to the GPU
kernel_tuner/runners/sequential.py:        self.gpu_args = self.dev.ready_argument_list(kernel_options.arguments)
kernel_tuner/runners/sequential.py:                # attempt to warmup the GPU by running the first config in the parameter space and ignoring the result
kernel_tuner/runners/sequential.py:                    self.dev.compile_and_benchmark(self.kernel_source, self.gpu_args, params, self.kernel_options, tuning_options)
kernel_tuner/runners/sequential.py:                result = self.dev.compile_and_benchmark(self.kernel_source, self.gpu_args, params, self.kernel_options, tuning_options)
kernel_tuner/core.py:from kernel_tuner.backends.pycuda import PyCudaFunctions
kernel_tuner/core.py:from kernel_tuner.backends.nvcuda import CudaFunctions
kernel_tuner/core.py:from kernel_tuner.backends.opencl import OpenCLFunctions
kernel_tuner/core.py:        _suffixes = {"CUDA": ".cu", "OpenCL": ".cl", "C": ".c"}
kernel_tuner/core.py:        :param device: CUDA/OpenCL device to use, in case you have multiple
kernel_tuner/core.py:            CUDA-capable GPUs or OpenCL devices you may use this to select one,
kernel_tuner/core.py:        :param platform: OpenCL platform to use, in case you have multiple
kernel_tuner/core.py:            OpenCL platforms you may use this to select one,
kernel_tuner/core.py:            0 by default. Ignored if not using OpenCL.
kernel_tuner/core.py:        :param lang: Specifies the language used for GPU kernels.
kernel_tuner/core.py:            Currently supported: "CUDA", "OpenCL", "HIP" or "C"
kernel_tuner/core.py:        if lang.upper() == "CUDA":
kernel_tuner/core.py:            dev = PyCudaFunctions(
kernel_tuner/core.py:        elif lang.upper() == "NVCUDA":
kernel_tuner/core.py:            dev = CudaFunctions(
kernel_tuner/core.py:        elif lang.upper() == "OPENCL":
kernel_tuner/core.py:            dev = OpenCLFunctions(
kernel_tuner/core.py:            raise ValueError("Sorry, support for languages other than CUDA, OpenCL, HIP, C, and Fortran is not implemented yet")
kernel_tuner/core.py:    def benchmark_prologue(self, func, gpu_args, threads, grid, result):
kernel_tuner/core.py:            self.dev.run_kernel(func, gpu_args, threads, grid)
kernel_tuner/core.py:    def benchmark_default(self, func, gpu_args, threads, grid, result):
kernel_tuner/core.py:            self.dev.run_kernel(func, gpu_args, threads, grid)
kernel_tuner/core.py:    def benchmark_continuous(self, func, gpu_args, threads, grid, result, duration):
kernel_tuner/core.py:            self.dev.run_kernel(func, gpu_args, threads, grid)
kernel_tuner/core.py:    def benchmark(self, func, gpu_args, instance, verbose, objective, skip_nvml_setting=False):
kernel_tuner/core.py:            self.benchmark_prologue(func, gpu_args, instance.threads, instance.grid, result)
kernel_tuner/core.py:            self.benchmark_default(func, gpu_args, instance.threads, instance.grid, result)
kernel_tuner/core.py:                    func, gpu_args, instance.threads, instance.grid, result, duration
kernel_tuner/core.py:        self, func, gpu_args, instance, answer, atol, verify, verbose
kernel_tuner/core.py:        # re-copy original contents of output arguments to GPU memory, to overwrite any changes
kernel_tuner/core.py:                self.dev.memcpy_htod(gpu_args[i], arg)
kernel_tuner/core.py:        check = self.run_kernel(func, gpu_args, instance)
kernel_tuner/core.py:        # retrieve gpu results to host memory
kernel_tuner/core.py:                    self.dev.memcpy_dtoh(result_host[-1], gpu_args[i])
kernel_tuner/core.py:                    if not answer[i].is_cuda:
kernel_tuner/core.py:                        #if the answer is on the host, copy gpu output to host as well
kernel_tuner/core.py:                        self.dev.memcpy_dtoh(result_host[-1], gpu_args[i].tensor)
kernel_tuner/core.py:                        result_host.append(gpu_args[i].tensor)
kernel_tuner/core.py:    def compile_and_benchmark(self, kernel_source, gpu_args, params, kernel_options, to):
kernel_tuner/core.py:            gpu_args = _preprocess_gpu_arguments(gpu_args, params)
kernel_tuner/core.py:                        func, gpu_args, instance, to.answer, to.atol, to.verify, verbose
kernel_tuner/core.py:                        self.benchmark(func, gpu_args, instance, verbose, to.objective, skip_nvml_setting=False)
kernel_tuner/core.py:    def preprocess_gpu_arguments(old_arguments, params):
kernel_tuner/core.py:        return _preprocess_gpu_arguments(old_arguments, params)
kernel_tuner/core.py:        if kernel_source.lang in ["CUDA", "NVCUDA"] and "<" in name and ">" in name:
kernel_tuner/core.py:        # Preprocess GPU arguments. Require for handling `Tunable` arguments
kernel_tuner/core.py:        arguments = _preprocess_gpu_arguments(kernel_options.arguments, params)
kernel_tuner/core.py:        """ready argument list to be passed to the kernel, allocates gpu mem if necessary"""
kernel_tuner/core.py:        flat_gpu_args = iter(self.dev.ready_argument_list(flat_args))
kernel_tuner/core.py:        gpu_args = []
kernel_tuner/core.py:                    arrays[key] = next(flat_gpu_args)
kernel_tuner/core.py:                gpu_args.append(Tunable(argument.param_key, arrays))
kernel_tuner/core.py:                gpu_args.append(next(flat_gpu_args))
kernel_tuner/core.py:        return gpu_args
kernel_tuner/core.py:    def run_kernel(self, func, gpu_args, instance):
kernel_tuner/core.py:            self.dev.run_kernel(func, gpu_args, instance.threads, instance.grid)
kernel_tuner/core.py:def _preprocess_gpu_arguments(old_arguments, params):
kernel_tuner/core.py:# these functions facilitate compiling templated kernels with PyCuda
pyproject.toml:description = "An easy to use CUDA/OpenCL kernel tuner in Python"
pyproject.toml:    "gpu",
pyproject.toml:    "pycuda",
pyproject.toml:    "cuda",
pyproject.toml:    "pyopencl",
pyproject.toml:    "opencl",
pyproject.toml:    "Environment :: GPU",
pyproject.toml:] # this ensures that people won't have to clone the whole repo to include notebooks, they can just do `pip install kernel_tuner[tutorial,cuda]`
pyproject.toml:# List of optional dependencies for user installation, e.g. `pip install kernel_tuner[cuda]`, used in the below `extras`.
pyproject.toml:# CUDA
pyproject.toml:pycuda = { version = "^2024.1", optional = true }           # Attention: if pycuda is changed here, also change `session.install("pycuda")` in the Noxfile
pyproject.toml:nvidia-ml-py = { version = "^12.535.108", optional = true }
pyproject.toml:# cupy-cuda11x = { version = "*", optional = true }    # Note: these are completely optional dependencies as described in CONTRIBUTING.rst
pyproject.toml:# cupy-cuda12x = { version = "*", optional = true }
pyproject.toml:# cuda-python = { version = "*", optional = true }
pyproject.toml:# OpenCL
pyproject.toml:pyopencl = { version = "*", optional = true } # Attention: if pyopencl is changed here, also change `session.install("pyopencl")` in the Noxfile
pyproject.toml:cuda = ["pycuda", "nvidia-ml-py", "pynvml"]
pyproject.toml:opencl = ["pyopencl"]
pyproject.toml:cuda_opencl = ["pycuda", "pyopencl"]
pyproject.toml:tutorial = ["jupyter", "matplotlib", "nvidia-ml-py"]
CONTRIBUTING.rst:* List the version of Python, CUDA or OpenCL, and C compiler, if applicable.
CONTRIBUTING.rst:* You have written unit tests to test your additions and all unit tests pass (run :bash:`nox`). If you do not have the required hardware, you can run :bash:`nox -- skip-gpu`, or :bash:`skip-cuda`, :bash:`skip-hip`, :bash:`skip-opencl`.
examples/README.rst:CUDA, OpenCL, or C kernel, while demonstrating a particular usecase of Kernel Tuner.
examples/README.rst:Except for `test\_vector\_add.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/test_vector_add.py>`__  and 
examples/README.rst:`test\_vector\_add_parameterized.py <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/test_vector_add_parameterized.py>`__,
examples/README.rst:which show how to write tests for GPU kernels with Kernel Tuner.
examples/README.rst:[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/vector_add.py>`__] [`CUDA-C++ <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda-c++/vector_add.py>`__] [`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/vector_add.py>`__] [`C <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/c/vector_add.py>`__] [`Fortran <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/fortran/vector_add.py>`__] [`OpenACC-C++ <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/directives/vector_add_c_openacc.py>`__] [`OpenACC-Fortran <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/directives/vector_add_fortran_openacc.py>`__]
examples/README.rst:[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/stencil.py>`__] [`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/stencil.py>`__]
examples/README.rst:[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/matmul.py>`__] [`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/matmul.py>`__]
examples/README.rst:kernel [`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/convolution.cu>`__]
examples/README.rst:[`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/convolution.cl>`__]
examples/README.rst:[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/convolution.py>`__] [`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/convolution.py>`__]
examples/README.rst:[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/sepconv.py>`__] [`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/sepconv.py>`__]
examples/README.rst:[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/convolution_correct.py>`__] [`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/convolution_correct.py>`__]
examples/README.rst:[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/convolution_streams.py>`__]
examples/README.rst: - overlap transfers to and from the GPU with computation
examples/README.rst:[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/reduction.py>`__] [`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/reduction.py>`__]
examples/README.rst: - use vector types and shuffle instructions (shuffle is only available in CUDA)
examples/README.rst:[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/spmv.py>`__]
examples/README.rst:[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/pnpoly.py>`__]
examples/README.rst:[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/expdist.py>`__]
examples/README.rst: -  C++ in CUDA kernel code
examples/README.rst:[`CUDA <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/cuda/vector_add_codegen.py>`__] [`OpenCL <https://github.com/kerneltuner/kernel_tuner/blob/master/examples/opencl/vector_add_codegen.py>`__]
examples/fortran/vector_add_acc.py:        compiler_options=["-fast", "-acc=gpu"],
examples/fortran/vector_add_acc.F90:    !$acc parallel loop device_type(nvidia) vector_length(block_size_x)
examples/opencl/reduction.py:    def verify_partial_reduce(cpu_result, gpu_result, atol=None):
examples/opencl/reduction.py:        return numpy.isclose(cpu_result[0], numpy.sum(gpu_result[0]), atol=atol)
examples/opencl/vector_add_codegen.py:        tune_params, lang="OpenCL")
examples/opencl/matmul.cl: * Optimized CUDA kernel for matrix multiplication
examples/opencl/matmul.cl: * GPU Technology Conference, GTC 2010.
examples/opencl/matmul.cl: * tuned towards each GPU. This kernel assumes that
examples/cuda/python_kernel_cupy.py:    # Note that the type and order should match our GPU code
examples/cuda/python_kernel_cupy.py:    # we can use it to conveniently use the GPU kernel in Python
examples/cuda/python_kernel_cupy.py:    # applications that want to frequently call the GPU kernel
examples/cuda/vector_add_jinja2.py:    result = tune_kernel("vector_add", template.render, size, arguments, tuning_parameters, lang="CUDA", grid_div_x=["block_size_x * vector_size * tiling_x"], answer=control)
examples/cuda/pnpoly_cupy.py:This program is used for auto-tuning the host and device code of a CUDA program
examples/cuda/pnpoly_cupy.py:execution on the GPU. Because each input is read only once and each output
examples/cuda/pnpoly_cupy.py:reuse those results on the GPU, instead of recomputing them on the GPU all
examples/cuda/pnpoly_cupy.py:def allocator(size: int) -> cp.cuda.PinnedMemoryPointer:
examples/cuda/pnpoly_cupy.py:    flags = cp.cuda.runtime.hostAllocPortable | cp.cuda.runtime.hostAllocMapped
examples/cuda/pnpoly_cupy.py:    mem = cp.cuda.PinnedMemory(size, flags=flags)
examples/cuda/pnpoly_cupy.py:    return cp.cuda.PinnedMemoryPointer(mem, offset=0)
examples/cuda/pnpoly_cupy.py:    cp.cuda.set_pinned_memory_allocator(allocator)
examples/cuda/reduction.py:    def verify_partial_reduce(cpu_result, gpu_result, atol=None):
examples/cuda/reduction.py:        return numpy.isclose(cpu_result[0], numpy.sum(gpu_result[0]), atol=atol)
examples/cuda/going_green_performance_model.py:  * Going green: optimizing GPUs for energy efficiency through model-steered auto-tuning
examples/cuda/going_green_performance_model.py:to reduce the number of frequencies for GPU energy tuning.
examples/cuda/going_green_performance_model.py:This example requires CUDA and NVML as well as PyCuda and a CUDA-capable
examples/cuda/going_green_performance_model.py:GPU with the ability (and permissions) to set applications clocks. GPUs
examples/cuda/going_green_performance_model.py:    from pycuda import driver as drv
examples/cuda/going_green_performance_model.py:                        default=0, help="GPU ID to use")
examples/cuda/going_green_performance_model.py:    parser.add_argument("-nsf", dest="nvidia_smi_fallback", nargs="?", default=None,
examples/cuda/going_green_performance_model.py:                        help="Path to nvidia-smi as fallback when missing NVML permissions")
examples/cuda/going_green_performance_model.py:                                                                                               nvidia_smi_fallback=args.nvidia_smi_fallback,
examples/cuda/going_green_performance_model.py:    plt.title('GPU modelled power consumption', size=18)
examples/cuda/going_green_performance_model.py:    plt.savefig("GPU_power_consumption_model.pdf")
examples/cuda/vector_add_observers_pmt.py:    pmtobserver = PMTObserver([("nvidia", 0), "rapl"])
examples/cuda/vector_add_observers_pmt.py:    metrics["GPU W"] = lambda p: p["nvidia_power"]
examples/cuda/convolution_streams.py:import pycuda.driver as drv
examples/cuda/matmul.cu: * Optimized CUDA kernel for matrix multiplication
examples/cuda/matmul.cu: * GPU Technology Conference, GTC 2010.
examples/cuda/matmul.cu: * tuned towards each GPU. This kernel assumes that
examples/cuda/texture.py:// This kernel is adapted from the PyCuda "Rotate" example.
examples/cuda/zeromeanfilter.cu: * This file contains CUDA kernels for applying a zero-mean total
examples/cuda/vector_add_jinja.py:    result = tune_kernel("vector_add", kernel_template.render, size, args, tune_params, lang="CUDA")
examples/cuda/python_kernel.py:    args = [c, a, b, n] #note that the type and order should match our GPU code
examples/cuda/python_kernel.py:    #we can use it to conveniently use the GPU kernel in Python
examples/cuda/python_kernel.py:    #applications that want to frequently call the GPU kernel
examples/cuda/test_vector_add.py:"""Minimal example for a CUDA Kernel unit test with the Kernel Tuner"""
examples/cuda/test_vector_add.py:    #Check pycuda is installed and if a CUDA capable device is present, if not skip the test
examples/cuda/test_vector_add.py:        import pycuda.driver as drv
examples/cuda/test_vector_add.py:        pytest.skip("PyCuda not installed or no CUDA device detected")
examples/cuda/convolution_streams.cu: * parameters in the host code of GPU programs, such as the number of 
examples/cuda/convolution_streams.cu:#include <cuda.h>
examples/cuda/convolution_streams.cu:    cudaError_t err;
examples/cuda/convolution_streams.cu:    err = cudaMalloc((void **)&d_output, image_width*image_height*sizeof(float));
examples/cuda/convolution_streams.cu:    if (err != cudaSuccess) {
examples/cuda/convolution_streams.cu:        fprintf(stderr, "Error in cudaMalloc: %s\n", cudaGetErrorString(err));
examples/cuda/convolution_streams.cu:    err = cudaMemset(d_output, 0, image_width*image_height*sizeof(float));
examples/cuda/convolution_streams.cu:    if (err != cudaSuccess) {
examples/cuda/convolution_streams.cu:        fprintf(stderr, "Error in cudaMemset: %s\n", cudaGetErrorString(err));
examples/cuda/convolution_streams.cu:    err = cudaMalloc((void **)&d_input, input_width*input_height*sizeof(float));
examples/cuda/convolution_streams.cu:    if (err != cudaSuccess) {
examples/cuda/convolution_streams.cu:        fprintf(stderr, "Error in cudaMalloc: %s\n", cudaGetErrorString(err));
examples/cuda/convolution_streams.cu:    cudaStream_t stream[num_streams];
examples/cuda/convolution_streams.cu:    cudaEvent_t event_htod[num_streams];
examples/cuda/convolution_streams.cu:        err = cudaStreamCreate(&stream[i]);
examples/cuda/convolution_streams.cu:        if (err != cudaSuccess) {
examples/cuda/convolution_streams.cu:            fprintf(stderr, "Error in cudaStreamCreate: %s\n", cudaGetErrorString(err));
examples/cuda/convolution_streams.cu:        err = cudaEventCreate(&event_htod[i]);
examples/cuda/convolution_streams.cu:        if (err != cudaSuccess) {
examples/cuda/convolution_streams.cu:            fprintf(stderr, "Error in cudaEventCreate: %s\n", cudaGetErrorString(err));
examples/cuda/convolution_streams.cu:    cudaEvent_t start;
examples/cuda/convolution_streams.cu:    err = cudaEventCreate(&start);
examples/cuda/convolution_streams.cu:    if (err != cudaSuccess) {
examples/cuda/convolution_streams.cu:        fprintf(stderr, "Error in cudaEventCreate: %s\n", cudaGetErrorString(err));
examples/cuda/convolution_streams.cu:    cudaEvent_t stop;
examples/cuda/convolution_streams.cu:    err = cudaEventCreate(&stop);
examples/cuda/convolution_streams.cu:    if (err != cudaSuccess) {
examples/cuda/convolution_streams.cu:        fprintf(stderr, "Error in cudaEventCreate: %s\n", cudaGetErrorString(err));
examples/cuda/convolution_streams.cu:    cudaDeviceSynchronize();
examples/cuda/convolution_streams.cu:    err = cudaGetLastError();
examples/cuda/convolution_streams.cu:    if (err != cudaSuccess) {
examples/cuda/convolution_streams.cu:        fprintf(stderr, "Error after memory setup in convolution_streams: %s\n", cudaGetErrorString(err));
examples/cuda/convolution_streams.cu:    cudaDeviceSynchronize();
examples/cuda/convolution_streams.cu:    cudaEventRecord(start, 0);
examples/cuda/convolution_streams.cu:    err = cudaMemcpyToSymbolAsync(d_filter, h_filter, filter_width*filter_height*sizeof(float), 0, cudaMemcpyHostToDevice, stream[0]);
examples/cuda/convolution_streams.cu:    if (err != cudaSuccess) {
examples/cuda/convolution_streams.cu:        fprintf(stderr, "Error in cudaMemcpyToSymbolAsync: %s\n", cudaGetErrorString(err));
examples/cuda/convolution_streams.cu:            err = cudaMemcpyAsync(d_input, h_input, (border + dps)*sizeof(float), cudaMemcpyHostToDevice, stream[k]);
examples/cuda/convolution_streams.cu:            err = cudaStreamWaitEvent(stream[k], event_htod[k-1], 0);
examples/cuda/convolution_streams.cu:            if (err != cudaSuccess) {
examples/cuda/convolution_streams.cu:                fprintf(stderr, "Error in cudaStreamWaitEvent htod k-1: %s\n", cudaGetErrorString(err));
examples/cuda/convolution_streams.cu:            err = cudaMemcpyAsync(d_input +border+k*dps, h_input +border+k*dps, dps*sizeof(float), cudaMemcpyHostToDevice, stream[k]);
examples/cuda/convolution_streams.cu:        if (err != cudaSuccess) {
examples/cuda/convolution_streams.cu:            fprintf(stderr, "Error in cudaMemcpyHostToDevice: %s\n", cudaGetErrorString(err));
examples/cuda/convolution_streams.cu:        err = cudaEventRecord(event_htod[k], stream[k]);
examples/cuda/convolution_streams.cu:        if (err != cudaSuccess) {
examples/cuda/convolution_streams.cu:            fprintf(stderr, "Error in cudaEventRecord htod: %s\n", cudaGetErrorString(err));
examples/cuda/convolution_streams.cu:        err = cudaMemcpyAsync(h_output + k*lps*image_width, d_output + k*lps*image_width, lps*image_width*sizeof(float), cudaMemcpyDeviceToHost, stream[k]);
examples/cuda/convolution_streams.cu:        if (err != cudaSuccess) {
examples/cuda/convolution_streams.cu:            fprintf(stderr, "Error in cudaMemcpyDeviceToHost: %s\n", cudaGetErrorString(err));
examples/cuda/convolution_streams.cu:    cudaEventRecord(stop, 0);
examples/cuda/convolution_streams.cu:    cudaDeviceSynchronize();
examples/cuda/convolution_streams.cu:    cudaEventElapsedTime(&time, start, stop);
examples/cuda/convolution_streams.cu:    cudaFree(d_output);
examples/cuda/convolution_streams.cu:    cudaFree(d_input);
examples/cuda/convolution_streams.cu:        cudaStreamDestroy(stream[k]);
examples/cuda/convolution_streams.cu:        cudaEventDestroy(event_htod[k]);
examples/cuda/convolution_streams.cu:    cudaEventDestroy(start);
examples/cuda/convolution_streams.cu:    cudaEventDestroy(stop);
examples/cuda/convolution_streams.cu:    cudaDeviceSynchronize();
examples/cuda/convolution_streams.cu:    err = cudaGetLastError();
examples/cuda/convolution_streams.cu:    if (err != cudaSuccess) {
examples/cuda/convolution_streams.cu:        const char *error_string = cudaGetErrorString(err);
examples/cuda/pnpoly.py:This program is used for auto-tuning the host and device code of a CUDA program
examples/cuda/pnpoly.py:execution on the GPU. Because each input is read only once and each output
examples/cuda/pnpoly.py:reuse those results on the GPU, instead of recomputing them on the GPU all
examples/cuda/pnpoly.py:import pycuda.driver as drv
examples/cuda/expdist.py:    print("best GPU configuration, total time=", best_config1['time'] + best_config2['time'])
examples/cuda/pnpoly.cu: * This file contains the implementation of a CUDA Kernel for the
examples/cuda/vector_add_codegen.py:        tune_params, lang="CUDA")
examples/cuda/test_vector_add_parameterized.py:"""Minimal example for a parameterized test of a CUDA kernel using Kernel Tuner"""
examples/cuda/pnpoly_host.cu:#include <cuda.h>
examples/cuda/pnpoly_host.cu: * This function contains the host code for benchmarking the cn_pnpoly CUDA kernel
examples/cuda/pnpoly_host.cu: * between host and device with kernel execution on the GPU. Because each input
examples/cuda/pnpoly_host.cu: * reuse those results on the GPU, instead of recomputing them on the GPU all
examples/cuda/pnpoly_host.cu:    cudaError_t err;
examples/cuda/pnpoly_host.cu:    err = cudaHostAlloc((void **)&h_slopes, VERTICES*sizeof(float), cudaHostAllocMapped);
examples/cuda/pnpoly_host.cu:    if (err != cudaSuccess) {
examples/cuda/pnpoly_host.cu:        fprintf(stderr, "Error in cudaHostAlloc: %s\n", cudaGetErrorString(err));
examples/cuda/pnpoly_host.cu:    //create CUDA streams and events
examples/cuda/pnpoly_host.cu:    cudaStream_t stream[1];
examples/cuda/pnpoly_host.cu:    err = cudaStreamCreate(&stream[0]);
examples/cuda/pnpoly_host.cu:    if (err != cudaSuccess) {
examples/cuda/pnpoly_host.cu:        fprintf(stderr, "Error in cudaStreamCreate: %s\n", cudaGetErrorString(err));
examples/cuda/pnpoly_host.cu:    cudaEvent_t start;
examples/cuda/pnpoly_host.cu:    err = cudaEventCreate(&start);
examples/cuda/pnpoly_host.cu:    if (err != cudaSuccess) {
examples/cuda/pnpoly_host.cu:        fprintf(stderr, "Error in cudaEventCreate: %s\n", cudaGetErrorString(err));
examples/cuda/pnpoly_host.cu:    cudaEvent_t stop;
examples/cuda/pnpoly_host.cu:    err = cudaEventCreate(&stop);
examples/cuda/pnpoly_host.cu:    if (err != cudaSuccess) {
examples/cuda/pnpoly_host.cu:        fprintf(stderr, "Error in cudaEventCreate: %s\n", cudaGetErrorString(err));
examples/cuda/pnpoly_host.cu:    cudaDeviceSynchronize();
examples/cuda/pnpoly_host.cu:    cudaEventRecord(start, stream[0]);
examples/cuda/pnpoly_host.cu:    err = cudaMemcpyToSymbolAsync(d_vertices, vertices, VERTICES*sizeof(float2), 0, cudaMemcpyHostToDevice, stream[0]);
examples/cuda/pnpoly_host.cu:    if (err != cudaSuccess) {
examples/cuda/pnpoly_host.cu:        fprintf(stderr, "Error in cudaMemcpyToSymbolAsync: %s\n", cudaGetErrorString(err));
examples/cuda/pnpoly_host.cu:    err = cudaMemcpyToSymbolAsync(d_slopes, h_slopes, VERTICES*sizeof(float), 0, cudaMemcpyHostToDevice, stream[0]);
examples/cuda/pnpoly_host.cu:    if (err != cudaSuccess) {
examples/cuda/pnpoly_host.cu:        fprintf(stderr, "Error in cudaMemcpyToSymbolAsync: %s\n", cudaGetErrorString(err));
examples/cuda/pnpoly_host.cu:    cudaEventRecord(stop, stream[0]);
examples/cuda/pnpoly_host.cu:    cudaDeviceSynchronize();
examples/cuda/pnpoly_host.cu:    cudaEventElapsedTime(&time, start, stop);
examples/cuda/pnpoly_host.cu:    cudaFreeHost(h_slopes);
examples/cuda/pnpoly_host.cu:    cudaStreamDestroy(stream[0]);
examples/cuda/pnpoly_host.cu:    cudaEventDestroy(start);
examples/cuda/pnpoly_host.cu:    cudaEventDestroy(stop);
examples/cuda/pnpoly_host.cu:    cudaDeviceSynchronize();
examples/cuda/pnpoly_host.cu:    err = cudaGetLastError();
examples/cuda/pnpoly_host.cu:    if (err != cudaSuccess) {
examples/cuda/pnpoly_host.cu:        const char *error_string = cudaGetErrorString(err);
examples/cuda/pnpoly_host.cu:            fprintf(stderr, "Error after CUDA kernel: %s\n", error_string);
examples/cuda/vector_add_nvcuda.py:"""This is the minimal example from the README, but using the Nvidia CUDA Python bindings
examples/cuda/vector_add_nvcuda.py:You can install the Nvidia CUDA bindings using 'pip install cuda-python'. The backend
examples/cuda/vector_add_nvcuda.py:can be selected using the lang='nvcuda' option of tune_kernel.
examples/cuda/vector_add_nvcuda.py:    result = tune_kernel("vector_add", kernel_string, size, args, tune_params, lang="nvcuda", answer=answer, verbose=True)
examples/cuda/accuracy.py:    #include <cuda_fp16.h>
examples/cuda/accuracy.py:        lang="CUDA",
examples/directives/vector_add_c_openacc.py:"""This is a simple example for tuning C++ OpenACC code with the kernel tuner"""
examples/directives/vector_add_c_openacc.py:    OpenACC,
examples/directives/vector_add_c_openacc.py:app = Code(OpenACC(), Cxx())
examples/directives/vector_add_c_openacc.py:    compiler_options=["-fast", "-acc=gpu"],
examples/directives/vector_add_fortran_openacc.py:"""This is a simple example for tuning Fortran OpenACC code with the kernel tuner"""
examples/directives/vector_add_fortran_openacc.py:    OpenACC,
examples/directives/vector_add_fortran_openacc.py:app = Code(OpenACC(), Fortran())
examples/directives/vector_add_fortran_openacc.py:    compiler_options=["-fast", "-acc=gpu"],
examples/directives/matrix_multiply_c_openacc.py:    OpenACC,
examples/directives/matrix_multiply_c_openacc.py:app = Code(OpenACC(), Cxx())
examples/directives/matrix_multiply_c_openacc.py:    compiler_options=["-fast", "-acc=gpu"],
INSTALL.rst:working Python version, several Python packages, and optionally CUDA and/or OpenCL
INSTALL.rst:.. _installing cuda:
INSTALL.rst:CUDA and PyCUDA
INSTALL.rst:Installing CUDA and PyCUDA is optional, because you may want to only use Kernel
INSTALL.rst:Tuner for tuning OpenCL or C kernels.
INSTALL.rst:CUDA kernels you will first need to install the CUDA toolkit
INSTALL.rst:(https://developer.nvidia.com/cuda-toolkit). A recent version of the
INSTALL.rst:CUDA toolkit (and the PyCUDA Python bindings for CUDA) are
INSTALL.rst:It's very important that you install the CUDA toolkit before trying to install PyCuda.
INSTALL.rst:You can install PyCuda manually using:
INSTALL.rst:    pip install pycuda
INSTALL.rst:Or you could install Kernel Tuner and PyCUDA together if you haven't done so already:
INSTALL.rst:    pip install kernel_tuner[cuda]
INSTALL.rst:If you run into trouble with installing PyCuda, make sure you have CUDA installed first.
INSTALL.rst:If you retry the ``pip install pycuda`` command, you may need to use the
INSTALL.rst:``--no-cache-dir`` option to ensure the pycuda installation really starts over and not continues
INSTALL.rst:If this fails, I recommend to see the PyCuda installation guide (https://wiki.tiker.net/PyCuda/Installation)
INSTALL.rst:Other CUDA Backends
INSTALL.rst:Kernel Tuner can also be used with CuPy (https://cupy.dev/) or Nvidia's CUDA Python bindings (https://nvidia.github.io/cuda-python/). Please see the installation instructions of those projects for how the required Python packages.
INSTALL.rst:OpenCL and PyOpenCL
INSTALL.rst:Before we can install PyOpenCL you'll need an OpenCL compiler. There are several
INSTALL.rst:OpenCL compilers available depending on the OpenCL platform you want to your
INSTALL.rst:* `AMD APP SDK <https://rocmdocs.amd.com/en/latest/Programming_Guides/Opencl-programming-guide.html>`__
INSTALL.rst:* `Intel OpenCL <https://software.intel.com/en-us/iocl_rt_ref>`__
INSTALL.rst:* `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`__
INSTALL.rst:* `Apple OpenCL <https://developer.apple.com/opencl/>`__
INSTALL.rst:You can also look at this `OpenCL Installation Guide <https://wiki.tiker.net/OpenCLHowTo>`__ for PyOpenCL.
INSTALL.rst:As with the CUDA toolkit, recent versions of one or more of the above OpenCL SDK's and
INSTALL.rst:PyOpenCL are recommended to support all features of the Kernel Tuner.
INSTALL.rst:After you've installed your OpenCL compiler of choice you can install PyOpenCL using:
INSTALL.rst:    pip install pyopencl
INSTALL.rst:Or you could install Kernel Tuner and PyOpenCL together if you haven't done so already:
INSTALL.rst:    pip install kernel_tuner[opencl]
INSTALL.rst:If this fails, please see the PyOpenCL installation guide (https://wiki.tiker.net/PyOpenCL/Installation)
INSTALL.rst:The HIP compiler is included as part of the ROCm software stack. Here is AMD's installation guide:
INSTALL.rst:* `ROCm Documentation: HIP Installation Guide <https://docs.amd.com/bundle/HIP-Installation-Guide-v5.3/page/Introduction_to_HIP_Installation_Guide.html>`__
INSTALL.rst:- `cuda`: install pycuda along with kernel_tuner
INSTALL.rst:- `opencl`: install pycuda along with kernel_tuner
INSTALL.rst:These can be installed by appending e.g. ``-E cuda -E opencl -E hip``.
INSTALL.rst:    poetry install --with test,docs -E cuda -E opencl
INSTALL.rst:    pip install kernel_tuner[tutorial,cuda]
INSTALL.rst:Or if you have already installed Kernel Tuner and PyCUDA, just use:
.gitignore:examples/cuda/output

```
