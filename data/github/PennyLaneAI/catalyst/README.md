# https://github.com/PennyLaneAI/catalyst

```console
setup.py:        "softwareq.qpp = catalyst.third_party.cuda:SoftwareQQPP",
setup.py:        "nvidia.custatevec = catalyst.third_party.cuda:NvidiaCuStateVec",
setup.py:        "nvidia.cutensornet = catalyst.third_party.cuda:NvidiaCuTensorNet",
setup.py:        "cuda_quantum.context = catalyst.tracing.contexts:EvaluationContext",
setup.py:        "cuda_quantum.ops = catalyst.api_extensions",
setup.py:        "cuda_quantum.qjit = catalyst.third_party.cuda:cudaqjit",
runtime/README.rst:     - **PennyLane-Lightning-Kokkos** and **PennyLane-Lightning-GPU**
runtime/.clang-tidy:  - key:             cppcoreguidelines-non-private-member-variables-in-classes.IgnoreClassesWithAllMemberVariablesBeingPublic
doc/dev/callbacks.rst:Callbacks and GPUs
doc/dev/callbacks.rst:- executing classical subroutines on accelerators such as GPUs or TPUs, or
doc/dev/callbacks.rst:  these callbacks directly on classical accelerators such as GPUs and TPUs.
doc/dev/callbacks.rst:Accelerator (GPU and TPU) support
doc/dev/callbacks.rst:classical accelerators such as GPUs and TPUs:
doc/dev/callbacks.rst:    @accelerate(dev=jax.devices("gpu")[0])
doc/dev/callbacks.rst:        y = classical_fn(jnp.sqrt(x)) # will be executed on a GPU
doc/dev/architecture.rst:  provided by the runtime as an interface layer to backend devices (such as CPU simulators, GPU
doc/dev/devices.rst:    - A fast state-vector qubit simulator utilizing the Kokkos library for CPU and GPU accelerated
doc/dev/devices.rst:  * - ``lightning.gpu``
doc/dev/devices.rst:    - A fast state-vector qubit simulator based on the `NVIDIA cuQuantum SDK <https://developer.nvidia.com/cuquantum-sdk>`__
doc/dev/devices.rst:      for the GPU-accelerated quantum simulation. See the
doc/dev/devices.rst:      `Lightning-GPU documentation <https://docs.pennylane.ai/projects/lightning/en/stable/lightning_gpu/device.html>`__
doc/dev/devices.rst:      `Catalyst configuration file <https://github.com/PennyLaneAI/pennylane-lightning/blob/master/pennylane_lightning/lightning_gpu/lightning_gpu.toml>`__
doc/dev/devices.rst:    - `Qrack <https://github.com/unitaryfund/qrack>`__ is a GPU-accelerated quantum computer
doc/dev/quick_start.rst:    ``lightning.qubit``, ``lightning.kokkos``, ``lightning.gpu``, and ``braket.aws.qubit``. For
doc/dev/quick_start.rst:   1. For devices without native controlled gates support (e.g., ``lightning.kokkos`` and ``lightning.gpu``), all :class:`qml.Controlled <pennylane.ops.op_math.Controlled>` operations will decompose to :class:`qml.QubitUnitary <pennylane.QubitUnitary>` operations.
doc/dev/arch/runtime.puml:  Container_Ext(_pl_lightning_kokkos_, "Lightning-Kokkos", "CPU/GPU")
doc/dev/arch/runtime.puml:  Container_Ext(_pl_lightning_gpu_, "Lightning-GPU", "GPU")
doc/dev/arch/runtime.puml:_pl_lightning_kokkos_ -[hidden]> _pl_lightning_gpu_
doc/dev/arch/runtime.puml:Rel_U(_pl_lightning_gpu_, _quantumdevice_, "", $tags="implementation")
doc/dev/arch/runtime.puml:_pl_lightning_gpu_ -[hidden]> _openqasm_
doc/code/__init__.rst:Module: catalyst.third_party.cuda
doc/code/__init__.rst:.. automodapi:: catalyst.third_party.cuda
doc/conf.py:    "cudaq",
doc/releases/changelog-0.9.0.md:* Add Lightning-GPU support to Catalyst docs and update tests.
doc/releases/changelog-0.9.0.md:  Similar to MLIR's `gpu.launch_kernel` function, Catalyst, now supports a `call_function_in_module`.
doc/releases/changelog-0.6.0.md:* Raises an exception if the user has an incompatible CUDA Quantum version installed.
doc/releases/changelog-0.7.0.md:  on GPUs or other accelerators with `catalyst.accelerate`, right inside of QJIT-compiled functions.
doc/releases/changelog-0.7.0.md:  @accelerate(dev=jax.devices("gpu")[0])
doc/releases/changelog-0.7.0.md:      y = classical_fn(jnp.sqrt(x)) # will be executed on a GPU
doc/releases/changelog-0.5.0.md:  CUDA Quantum compiler toolchain.
doc/releases/changelog-0.5.0.md:  Simply import the CUDA Quantum `@cudaqjit` decorator to use this functionality:
doc/releases/changelog-0.5.0.md:  from catalyst.cuda import cudaqjit
doc/releases/changelog-0.5.0.md:  Or, if using Catalyst from PennyLane, simply specify `@qml.qjit(compiler="cuda_quantum")`.
doc/releases/changelog-0.5.0.md:  The following devices are available when compiling with CUDA Quantum:
doc/releases/changelog-0.5.0.md:  * `nvidia.custatevec`: The NVIDIA CuStateVec GPU simulator (with support for multi-gpu)
doc/releases/changelog-0.5.0.md:  * `nvidia.cutensornet`: The NVIDIA CuTensorNet GPU simulator (with support for matrix product state)
doc/releases/changelog-0.5.0.md:  @cudaqjit
doc/releases/changelog-0.5.0.md:  Note that CUDA Quantum compilation currently does not have feature parity with Catalyst
doc/releases/changelog-0.8.0.md:* JAX-compatible functions that run on classical accelerators, such as GPUs, via `catalyst.accelerate` now support autodifferentiation.
MANIFEST.in:recursive-include frontend/catalyst/third_party/cuda/ *.toml
frontend/test/pytest/test_seeded_qjit.py:    if backend not in ["lightning.qubit", "lightning.kokkos", "lightning.gpu"]:
frontend/test/pytest/test_seeded_qjit.py:            "Sample seeding is only supported on lightning.qubit, lightning.kokkos and lightning.gpu"
frontend/test/pytest/test_global_phase.py:    if backend in ("lightning.kokkos", "lightning.gpu"):
frontend/test/pytest/test_mid_circuit_measurement.py:        if backend in ("lightning.kokkos", "lightning.gpu"):
frontend/test/pytest/test_cuda_integration.py:"""CUDA Integration testing."""
frontend/test/pytest/test_cuda_integration.py:# This import is here on purpose. We shouldn't ever import CUDA
frontend/test/pytest/test_cuda_integration.py:# when we are running kokkos. Importing CUDA before running any kokkos
frontend/test/pytest/test_cuda_integration.py:@pytest.mark.cuda
frontend/test/pytest/test_cuda_integration.py:class TestCudaQ:
frontend/test/pytest/test_cuda_integration.py:    """CUDA Quantum integration tests. Skip if kokkos."""
frontend/test/pytest/test_cuda_integration.py:        from catalyst.third_party.cuda import cudaqjit as cjit
frontend/test/pytest/test_cuda_integration.py:        from catalyst.third_party.cuda import cudaqjit as cjit
frontend/test/pytest/test_cuda_integration.py:        from catalyst.third_party.cuda import cudaqjit as cjit
frontend/test/pytest/test_cuda_integration.py:        cuda_compiled = catalyst.third_party.cuda.cudaqjit()(circuit_a)
frontend/test/pytest/test_cuda_integration.py:        observed = cuda_compiled(3.14)
frontend/test/pytest/test_cuda_integration.py:    def test_cuda_device(self):
frontend/test/pytest/test_cuda_integration.py:        cuda_compiled = catalyst.third_party.cuda.cudaqjit()(circuit)
frontend/test/pytest/test_cuda_integration.py:        observed = cuda_compiled(3.14)
frontend/test/pytest/test_cuda_integration.py:        cuda_compiled = catalyst.third_party.cuda.cudaqjit()(circuit)
frontend/test/pytest/test_cuda_integration.py:        observed = cuda_compiled(3.14)
frontend/test/pytest/test_cuda_integration.py:        cuda_compiled = catalyst.third_party.cuda.cudaqjit()(circuit)
frontend/test/pytest/test_cuda_integration.py:        observed = cuda_compiled(3.14)
frontend/test/pytest/test_cuda_integration.py:    def test_qjit_cuda_device(self):
frontend/test/pytest/test_cuda_integration.py:        cuda_compiled = catalyst.third_party.cuda.cudaqjit(fn=circuit)
frontend/test/pytest/test_cuda_integration.py:        observed = cuda_compiled(3.14)
frontend/test/pytest/test_cuda_integration.py:        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
frontend/test/pytest/test_cuda_integration.py:        observed = cuda_compiled(3.14)
frontend/test/pytest/test_cuda_integration.py:        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
frontend/test/pytest/test_cuda_integration.py:        observed = cuda_compiled(3.14)
frontend/test/pytest/test_cuda_integration.py:        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
frontend/test/pytest/test_cuda_integration.py:        observed = cuda_compiled(jax.numpy.array([3.14, 0.0]))
frontend/test/pytest/test_cuda_integration.py:    def test_cuda_device_entry_point(self):
frontend/test/pytest/test_cuda_integration.py:        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
frontend/test/pytest/test_cuda_integration.py:        observed = cuda_compiled(3.14)
frontend/test/pytest/test_cuda_integration.py:    def test_cuda_device_entry_point_compiler(self):
frontend/test/pytest/test_cuda_integration.py:        """Test the entry point for cudaq.qjit"""
frontend/test/pytest/test_cuda_integration.py:        @qml.qjit(compiler="cuda_quantum")
frontend/test/pytest/test_cuda_integration.py:        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
frontend/test/pytest/test_cuda_integration.py:        observed = cuda_compiled()
frontend/test/pytest/test_cuda_integration.py:        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
frontend/test/pytest/test_cuda_integration.py:        observed = cuda_compiled()
frontend/test/pytest/test_cuda_integration.py:        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
frontend/test/pytest/test_cuda_integration.py:        observed = cuda_compiled()
frontend/test/pytest/test_cuda_integration.py:        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
frontend/test/pytest/test_cuda_integration.py:        observed = cuda_compiled()
frontend/test/pytest/test_cuda_integration.py:        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
frontend/test/pytest/test_cuda_integration.py:        observed = cuda_compiled()
frontend/test/pytest/test_cuda_integration.py:        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
frontend/test/pytest/test_cuda_integration.py:        observed = cuda_compiled()
frontend/test/pytest/test_cuda_integration.py:        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
frontend/test/pytest/test_cuda_integration.py:        observed = cuda_compiled()
frontend/test/pytest/test_cuda_integration.py:        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
frontend/test/pytest/test_cuda_integration.py:        observed = cuda_compiled()
frontend/test/pytest/test_cuda_integration.py:            catalyst.third_party.cuda.cudaqjit(wrapper)(1.0)
frontend/test/pytest/test_cuda_integration.py:        from catalyst.third_party.cuda import cudaqjit as cjit
frontend/test/conftest.py:def is_cuda_available():
frontend/test/conftest.py:    """Checks if cuda is available by trying an import.
frontend/test/conftest.py:    We do not want to import cudaq unless we absolutely need to.
frontend/test/conftest.py:    This is because cudaq prevents kokkos kernels from executing properly.
frontend/test/conftest.py:        import cudaq
frontend/test/conftest.py:        cudaq_available = False
frontend/test/conftest.py:        cudaq_available = True
frontend/test/conftest.py:    return cudaq_available
frontend/test/conftest.py:        "cuda: run cuda tests",
frontend/test/conftest.py:def skip_cuda_tests(config, items):
frontend/test/conftest.py:    """Skip cuda tests according to the following logic:
frontend/test/conftest.py:      except: if lightning.kokkos or lightning.gpu
frontend/test/conftest.py:      except: is cuda-quantum not installed
frontend/test/conftest.py:    Important! We should only check if cuda-quantum is installed
frontend/test/conftest.py:    as a last resort. We don't want to check if cuda-quantum is
frontend/test/conftest.py:    is_kokkos_or_gpu = config.getoption("backend") in ("lightning.kokkos", "lightning.gpu")
frontend/test/conftest.py:    # CUDA quantum is not supported in apple silicon.
frontend/test/conftest.py:    # CUDA quantum cannot run with kokkos
frontend/test/conftest.py:    skip_cuda_tests_val = is_kokkos_or_gpu or is_apple
frontend/test/conftest.py:    if not skip_cuda_tests_val and not is_cuda_available():
frontend/test/conftest.py:        # Only check this conditionally as it imports cudaq.
frontend/test/conftest.py:        skip_cuda_tests_val = True
frontend/test/conftest.py:        is_cuda_test = "cuda" in item.keywords
frontend/test/conftest.py:        skip_cuda = is_cuda_test and skip_cuda_tests_val
frontend/test/conftest.py:        if skip_cuda:
frontend/test/conftest.py:    skip_cuda_tests(config, items)
frontend/catalyst/api_extensions/callbacks.py:    accelerators such as GPUs from within a qjit-compiled function.
frontend/catalyst/api_extensions/callbacks.py:        @accelerate(dev=jax.devices("gpu")[0])
frontend/catalyst/api_extensions/callbacks.py:            y = classical_fn(jnp.sqrt(x)) # will be executed on a GPU
frontend/catalyst/api_extensions/callbacks.py:            x = jax.device_put(x, jax.local_devices("gpu")[0])
frontend/catalyst/api_extensions/callbacks.py:            y = accelerate(classical_fn)(x) # will be executed on a GPU
frontend/catalyst/device/qjit_device.py:from catalyst.third_party.cuda import SoftwareQQPP
frontend/catalyst/third_party/cuda/primitives/__init__.py:This module implements JAXPR primitives for CUDA-quantum.
frontend/catalyst/third_party/cuda/primitives/__init__.py:import cudaq
frontend/catalyst/third_party/cuda/primitives/__init__.py:# We disable protected access in particular to avoid warnings with cudaq._pycuda.
frontend/catalyst/third_party/cuda/primitives/__init__.py:class AbsCudaQState(jax.core.AbstractValue):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    "Abstract CUDA-quantum State."
frontend/catalyst/third_party/cuda/primitives/__init__.py:    hash_value = hash("AbsCudaQState")
frontend/catalyst/third_party/cuda/primitives/__init__.py:        return isinstance(other, AbsCudaQState)  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:class CudaQState(cudaq.State):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    "Concrete CUDA-quantum state."
frontend/catalyst/third_party/cuda/primitives/__init__.py:    aval = AbsCudaQState
frontend/catalyst/third_party/cuda/primitives/__init__.py:class AbsCudaQbit(jax.core.AbstractValue):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    "Abstract CUDA-quantum qbit."
frontend/catalyst/third_party/cuda/primitives/__init__.py:    hash_value = hash("AbsCudaQbit")
frontend/catalyst/third_party/cuda/primitives/__init__.py:        return isinstance(other, AbsCudaQbit)  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:class CudaQbit(cudaq._pycudaq.QuakeValue):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    "Concrete CUDA-quantum qbit."
frontend/catalyst/third_party/cuda/primitives/__init__.py:    aval = AbsCudaQbit
frontend/catalyst/third_party/cuda/primitives/__init__.py:class AbsCudaQReg(jax.core.AbstractValue):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    "Abstract CUDA-quantum quantum register."
frontend/catalyst/third_party/cuda/primitives/__init__.py:    hash_value = hash("AbsCudaQReg")
frontend/catalyst/third_party/cuda/primitives/__init__.py:        return isinstance(other, AbsCudaQReg)  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:class CudaQReg(cudaq._pycudaq.QuakeValue):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    "Concrete CUDA-quantum quantum register."
frontend/catalyst/third_party/cuda/primitives/__init__.py:    aval = AbsCudaQReg
frontend/catalyst/third_party/cuda/primitives/__init__.py:class AbsCudaValue(jax.core.AbstractValue):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    "Abstract CUDA-quantum value."
frontend/catalyst/third_party/cuda/primitives/__init__.py:    hash_value = hash("AbsCudaValue")
frontend/catalyst/third_party/cuda/primitives/__init__.py:        return isinstance(other, AbsCudaValue)  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:class CudaValue(cudaq._pycudaq.QuakeValue):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    "Concrete CUDA-quantum value."
frontend/catalyst/third_party/cuda/primitives/__init__.py:    aval = AbsCudaValue
frontend/catalyst/third_party/cuda/primitives/__init__.py:class AbsCudaKernel(jax.core.AbstractValue):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    "Abstract CUDA-quantum kernel."
frontend/catalyst/third_party/cuda/primitives/__init__.py:    hash_value = hash("AbsCudaKernel")
frontend/catalyst/third_party/cuda/primitives/__init__.py:        return isinstance(other, AbsCudaKernel)  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:class CudaKernel(cudaq._pycudaq.QuakeValue):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    "Concrete CUDA-quantum kernel."
frontend/catalyst/third_party/cuda/primitives/__init__.py:    aval = AbsCudaKernel
frontend/catalyst/third_party/cuda/primitives/__init__.py:class AbsCudaSampleResult(jax.core.AbstractValue):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    "Abstract CUDA-quantum kernel."
frontend/catalyst/third_party/cuda/primitives/__init__.py:    hash_value = hash("AbsCudaSampleResult")
frontend/catalyst/third_party/cuda/primitives/__init__.py:        return isinstance(other, AbsCudaSampleResult)  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:class CudaSampleResult(cudaq.SampleResult):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    "Concrete CUDA-quantum kernel."
frontend/catalyst/third_party/cuda/primitives/__init__.py:    aval = AbsCudaSampleResult
frontend/catalyst/third_party/cuda/primitives/__init__.py:class AbsCudaSpinOperator(jax.core.AbstractValue):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    "Abstract CUDA-quantum spin operator."
frontend/catalyst/third_party/cuda/primitives/__init__.py:    hash_value = hash("AbsCudaSpinOperator")
frontend/catalyst/third_party/cuda/primitives/__init__.py:        return isinstance(other, AbsCudaSpinOperator)  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:class CudaSpinOperator(cudaq.SpinOperator):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    "Concrete CUDA-quantum spin operator."
frontend/catalyst/third_party/cuda/primitives/__init__.py:    aval = AbsCudaSpinOperator
frontend/catalyst/third_party/cuda/primitives/__init__.py:class AbsCudaQObserveResult(jax.core.AbstractValue):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    "Abstract CUDA-quantum observe result."
frontend/catalyst/third_party/cuda/primitives/__init__.py:    hash_value = hash("AbsCudaQObserveResult")
frontend/catalyst/third_party/cuda/primitives/__init__.py:        return isinstance(other, AbsCudaQObserveResult)  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:class CudaQObserveResult(cudaq.ObserveResult):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    "Concrete CUDA-quantum observe result."
frontend/catalyst/third_party/cuda/primitives/__init__.py:    aval = AbsCudaQObserveResult
frontend/catalyst/third_party/cuda/primitives/__init__.py:# https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html
frontend/catalyst/third_party/cuda/primitives/__init__.py:# cudaq.make_kernel() -> cudaq.Kernel
frontend/catalyst/third_party/cuda/primitives/__init__.py:cudaq_make_kernel_p = jax.core.Primitive("cudaq_make_kernel")
frontend/catalyst/third_party/cuda/primitives/__init__.py:cudaq_make_kernel_p.multiple_results = True
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_make_kernel(*args):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    """Just a convenience function to bind the cudaq make kernel primitive.
frontend/catalyst/third_party/cuda/primitives/__init__.py:    From the documentation: https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html#cudaq.make_kernel
frontend/catalyst/third_party/cuda/primitives/__init__.py:    The following types are supported as kernel arguments: int, float, list/List, cudaq.qubit,
frontend/catalyst/third_party/cuda/primitives/__init__.py:    or cudaq.qreg.
frontend/catalyst/third_party/cuda/primitives/__init__.py:    return cudaq_make_kernel_p.bind(*args)
frontend/catalyst/third_party/cuda/primitives/__init__.py:@cudaq_make_kernel_p.def_impl
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_make_kernel_primitive_impl(*args):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    """Concrete implementation of cudaq.make_kernel is just a call."""
frontend/catalyst/third_party/cuda/primitives/__init__.py:        return (cudaq.make_kernel(),)
frontend/catalyst/third_party/cuda/primitives/__init__.py:    return cudaq.make_kernel(*args)
frontend/catalyst/third_party/cuda/primitives/__init__.py:@cudaq_make_kernel_p.def_abstract_eval
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_make_kernel_primitive_abs(*args):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    """Abstract implementation of cudaq.make_kernel."""
frontend/catalyst/third_party/cuda/primitives/__init__.py:    retvals.append(AbsCudaKernel())
frontend/catalyst/third_party/cuda/primitives/__init__.py:# cudaq.make_kernel(*args) -> tuple
frontend/catalyst/third_party/cuda/primitives/__init__.py:# cudaq.from_state(kernel: cudaq.Kernel, qubits: cudaq.QuakeValue, state: numpy.ndarray[]) -> None
frontend/catalyst/third_party/cuda/primitives/__init__.py:# cudaq.from_state(state: numpy.ndarray[]) -> cudaq.Kernel
frontend/catalyst/third_party/cuda/primitives/__init__.py:# qalloc(self: cudaq.Kernel)                   -> cudaq.QuakeValue
frontend/catalyst/third_party/cuda/primitives/__init__.py:# qalloc(self: cudaq.Kernel, qubit_count: int) -> cudaq.QuakeValue
frontend/catalyst/third_party/cuda/primitives/__init__.py:    return AbsCudaQReg()
frontend/catalyst/third_party/cuda/primitives/__init__.py:    return AbsCudaQbit()
frontend/catalyst/third_party/cuda/primitives/__init__.py:cudaq_getstate_p = jax.core.Primitive("cudaq_getstate")
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_getstate(kernel):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    return cudaq_getstate_p.bind(kernel)
frontend/catalyst/third_party/cuda/primitives/__init__.py:@cudaq_getstate_p.def_impl
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_getstate_primitive_impl(kernel):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    return jax.numpy.array(cudaq.get_state(kernel))
frontend/catalyst/third_party/cuda/primitives/__init__.py:@cudaq_getstate_p.def_abstract_eval
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_getstate_primitive_abs(_kernel):  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:    return AbsCudaQState()
frontend/catalyst/third_party/cuda/primitives/__init__.py:# __call__(self: cudaq.Kernel, *args) -> None
frontend/catalyst/third_party/cuda/primitives/__init__.py:# x(self: cudaq.Kernel, target: cudaq.QuakeValue) -> None
frontend/catalyst/third_party/cuda/primitives/__init__.py:        Quantum operations in CUDA-quantum return no values. But JAXPR expects return values.
frontend/catalyst/third_party/cuda/primitives/__init__.py:        method = getattr(cudaq.Kernel, inst)
frontend/catalyst/third_party/cuda/primitives/__init__.py:cuda_inst, _ = make_primitive_for_gate()
frontend/catalyst/third_party/cuda/primitives/__init__.py:    method = getattr(cudaq.Kernel, gate)
frontend/catalyst/third_party/cuda/primitives/__init__.py:        return AbsCudaValue()
frontend/catalyst/third_party/cuda/primitives/__init__.py:cudaq_sample_p = jax.core.Primitive("cudaq_sample")
frontend/catalyst/third_party/cuda/primitives/__init__.py:cudaq_counts_p = jax.core.Primitive("cudaq_counts")
frontend/catalyst/third_party/cuda/primitives/__init__.py:cudaq_counts_p.multiple_results = True
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_sample(kernel, *args, shots_count=1000):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    return cudaq_sample_p.bind(kernel, *args, shots_count=shots_count)
frontend/catalyst/third_party/cuda/primitives/__init__.py:@cudaq_sample_p.def_impl
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_sample_impl(kernel, *args, shots_count=1000):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    """Concrete implementation of cudaq.sample.
frontend/catalyst/third_party/cuda/primitives/__init__.py:    `cudaq.sample` returns an object which is a compressed version of what
frontend/catalyst/third_party/cuda/primitives/__init__.py:    population, `cudaq.sample` returns a dictionary where the keys are bitstrings and
frontend/catalyst/third_party/cuda/primitives/__init__.py:    In a way `qml.count` is more similar to `cudaq.sample` than `qml.sample`.
frontend/catalyst/third_party/cuda/primitives/__init__.py:    a_dict = cudaq.sample(kernel, *args, shots_count=shots_count)
frontend/catalyst/third_party/cuda/primitives/__init__.py:@cudaq_sample_p.def_abstract_eval
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_sample_abs(_kernel, *_args, shots_count=1000):  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:    return AbsCudaSampleResult()
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_counts(kernel, *args, shape, shots_count=1000):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    return cudaq_counts_p.bind(kernel, *args, shape=shape, shots_count=shots_count)
frontend/catalyst/third_party/cuda/primitives/__init__.py:@cudaq_counts_p.def_impl
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_counts_impl(kernel, *args, shape=None, shots_count=1000):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    `cudaq.sample` returns an object which is a compressed version of what
frontend/catalyst/third_party/cuda/primitives/__init__.py:    population, `cudaq.sample` returns a dictionary where the keys are bitstrings and
frontend/catalyst/third_party/cuda/primitives/__init__.py:    CUDA-quantum does not implement another function similar to `qml.counts`.
frontend/catalyst/third_party/cuda/primitives/__init__.py:    The closest function is `cudaq.sample`.
frontend/catalyst/third_party/cuda/primitives/__init__.py:    a_dict = cudaq.sample(kernel, *args, shots_count=shots_count)
frontend/catalyst/third_party/cuda/primitives/__init__.py:    # It looks like cuda uses a different endianness than catalyst.
frontend/catalyst/third_party/cuda/primitives/__init__.py:@cudaq_counts_p.def_abstract_eval
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_counts_abs(kernel, shape, shots_count=1000):  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:cudaq_spin_p = jax.core.Primitive("spin")
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_spin(target, kind: str):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    return cudaq_spin_p.bind(target, kind)
frontend/catalyst/third_party/cuda/primitives/__init__.py:@cudaq_spin_p.def_impl
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_spin_impl(target, kind: str):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    method = getattr(cudaq.spin, kind)
frontend/catalyst/third_party/cuda/primitives/__init__.py:@cudaq_spin_p.def_abstract_eval
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_spin_abs(target, kind):  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:    return AbsCudaSpinOperator()
frontend/catalyst/third_party/cuda/primitives/__init__.py:cudaq_observe_p = jax.core.Primitive("observe")
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_observe(kernel, spin_operator, shots_count=-1, noise_model=None):
frontend/catalyst/third_party/cuda/primitives/__init__.py:        https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html#cudaq.observe
frontend/catalyst/third_party/cuda/primitives/__init__.py:    return cudaq_observe_p.bind(
frontend/catalyst/third_party/cuda/primitives/__init__.py:@cudaq_observe_p.def_abstract_eval
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_observe_abs(kernel, spin_operator, shots_count=-1, noise_model=None):  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:    return AbsCudaQObserveResult()
frontend/catalyst/third_party/cuda/primitives/__init__.py:@cudaq_observe_p.def_impl
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_observe_impl(kernel, spin_operator, shots_count=-1, noise_model=None):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    return cudaq.observe(kernel, spin_operator, shots_count=shots_count, noise_model=noise_model)
frontend/catalyst/third_party/cuda/primitives/__init__.py:cudaq_expectation_p = jax.core.Primitive("expectation")
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_expectation(observe_result):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    return cudaq_expectation_p.bind(observe_result)
frontend/catalyst/third_party/cuda/primitives/__init__.py:@cudaq_expectation_p.def_abstract_eval
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_expectation_abs(observe_result):  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:@cudaq_expectation_p.def_impl
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_expectation_impl(observe_result):
frontend/catalyst/third_party/cuda/primitives/__init__.py:cudaq_adjoint_p = jax.core.Primitive("cudaq_adjoint")
frontend/catalyst/third_party/cuda/primitives/__init__.py:cudaq_adjoint_p.multiple_results = True
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_adjoint(kernel, target, *args):
frontend/catalyst/third_party/cuda/primitives/__init__.py:    cudaq_adjoint_p.bind(kernel, target, *args)
frontend/catalyst/third_party/cuda/primitives/__init__.py:@cudaq_adjoint_p.def_abstract_eval
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_adjoint_abs(kernel, target, *args):  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:@cudaq_adjoint_p.def_impl
frontend/catalyst/third_party/cuda/primitives/__init__.py:def cudaq_adjoint_impl(kernel, target, *args):
frontend/catalyst/third_party/cuda/primitives/__init__.py:jax.core.pytype_aval_mappings[CudaValue] = lambda x: x.aval  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:jax.core.pytype_aval_mappings[CudaQReg] = lambda x: x.aval  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:jax.core.pytype_aval_mappings[CudaQbit] = lambda x: x.aval  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:jax.core.pytype_aval_mappings[CudaSampleResult] = lambda x: x.aval  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:jax.core.pytype_aval_mappings[CudaQState] = lambda x: x.aval  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:jax.core.pytype_aval_mappings[CudaSpinOperator] = lambda x: x.aval  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:jax.core.pytype_aval_mappings[CudaQObserveResult] = lambda x: x.aval  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:jax.core.raise_to_shaped_mappings[AbsCudaValue] = lambda aval, _: aval
frontend/catalyst/third_party/cuda/primitives/__init__.py:jax.core.raise_to_shaped_mappings[AbsCudaQReg] = lambda aval, _: aval
frontend/catalyst/third_party/cuda/primitives/__init__.py:jax.core.raise_to_shaped_mappings[AbsCudaKernel] = lambda aval, _: aval
frontend/catalyst/third_party/cuda/primitives/__init__.py:jax.core.raise_to_shaped_mappings[AbsCudaQbit] = lambda aval, _: aval
frontend/catalyst/third_party/cuda/primitives/__init__.py:jax.core.raise_to_shaped_mappings[AbsCudaSampleResult] = lambda aval, _: aval  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:jax.core.raise_to_shaped_mappings[AbsCudaQState] = lambda aval, _: aval
frontend/catalyst/third_party/cuda/primitives/__init__.py:jax.core.raise_to_shaped_mappings[AbsCudaQObserveResult] = lambda aval, _: aval  # pragma: nocover
frontend/catalyst/third_party/cuda/primitives/__init__.py:jax.core.raise_to_shaped_mappings[AbsCudaSpinOperator] = lambda aval, _: aval  # pragma: nocover
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:Catalyst operations, it will issue calls to cuda-quantum operations.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:This effectively transforms a catalyst program into something that can generate cuda
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:This module also uses the CUDA-quantum API. Here is the reference:
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:  https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:import cudaq
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    cuda_inst,
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    cudaq_adjoint,
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    cudaq_counts,
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    cudaq_expectation,
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    cudaq_getstate,
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    cudaq_make_kernel,
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    cudaq_observe,
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    cudaq_sample,
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    cudaq_spin,
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:# cudaq._pycuda.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    This is useful for example in the proof-of-concept for cuda integration.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    it in CUDA-quantum primitives.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:            self.kernel, _shots = change_device_to_cuda_device(self)
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:            change_alloc_to_cuda_alloc(self, self.kernel)
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:def change_device_to_cuda_device(ctx):
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    """Map Catalyst's qdevice_p primitive to its equivalent CUDA-quantum primitive
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    # Shots are specified in PL at the very beginning, but in cuda
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    if not cudaq.has_target(device_name):
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    cudaq_target = cudaq.get_target(device_name)
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    cudaq.set_target(cudaq_target)
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    # cudaq_make_kernel returns a multiple values depending on the arguments.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    kernel = cudaq_make_kernel()
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:def change_alloc_to_cuda_alloc(ctx, kernel):
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    """Change Catalyst's qalloc_p primitive to a CUDA-quantum primitive.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    CUDA-quantum does require a kernel as an operand.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    """Change catalyst's qextract_p primitive to a CUDA-quantum primitive."""
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    # invals[0] should point to a correct cuda register.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    cuda_qubit = qreg_getitem(register, idx)
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    ctx.set_qubit_to_wire(idx, cuda_qubit)
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    outvals = [cuda_qubit]
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    """Set the correct post-conditions for CUDA-quantum when interpreting qinsert_p primitive
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    This method is interesting because CUDA-quantum does not use value semantics for their qubits.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    CUDA-quantum does not use value semantics for their qubits nor their quantum registers,
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    # variables, invals[0] now holds a reference to a cuda register
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    """Change the instruction to one supported in CUDA-quantum."""
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    from_catalyst_to_cuda = {
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    cuda_inst_name = from_catalyst_to_cuda[op]
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    cuda_inst(ctx.kernel, *qubits_or_params, inst=cuda_inst_name, qubits_len=qubits_len)
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    """Change Catalyst's state_p to CUDA-quantum's state primitive."""
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    # To get a state in cuda we need a kernel
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    cuda_state = cudaq_getstate(ctx.kernel)
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    outvals = [cuda_state]
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    """Change Catalyst's sample_p or counts_p primitive to respective CUDA-quantum primitives."""
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:        shots_result = cudaq_sample(ctx.kernel, shots_count=shots)
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:        outvals = cudaq_counts(ctx.kernel, shape=shape, shots_count=shots)
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    """Change Catalyst's qmeasure_p to CUDA-quantum measure."""
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    # this qubit refers to one in the cuda program.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    # Cuda can measure in multiple basis.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    # So we map this measurement op to mz in cuda.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    """Change Catalyst's expval to CUDA-quantum equivalent."""
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    observe_results = cudaq_observe(ctx.kernel, obs, shots)
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    result = cudaq_expectation(observe_results)
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    """Change named observable to CUDA-quantum equivalent."""
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    # Since CUDA doesn't use SSA for qubits.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    # This will be the target to cudaq_spin....
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    catalyst_cuda_map = {"PauliZ": "z", "PauliX": "x", "PauliY": "y"}
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    assert kind in catalyst_cuda_map
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    cuda_name = catalyst_cuda_map[kind]
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    outvals = [cudaq_spin(idx, cuda_name)]
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    """Change catalyst hamiltonian to an equivalent expression in CUDA."""
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    """Change Catalyst adjoint to an equivalent expression in CUDA.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    Notice that cudaq_make_kernel's API can take types as inputs.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    E.g., (int, float, List, cudaq.qreg)
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    We are currently not passing these inputs to cudaq_make_kernel.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    Instead we are passing only cudaq.qreg and all these inputs are concretely
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    in cudaq.make_kernel.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    # cudaq.qreg is essentially an abstract type in cudaq
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    kernel_to_adjoint, abstract_qreg = cudaq_make_kernel(cudaq.qreg)
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    cudaq_adjoint(ctx.kernel, kernel_to_adjoint, register)
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    msg = f"{eqn.primitive} is not yet implemented in Catalyst's CUDA-Quantum support."
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    # not necessary in the CUDA-quantum API.
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:    CUDA-quantum equivalent instructions. As these operations are
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:            m = "You cannot return measurements directly from a tape when compiling for cuda quantum."
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:class QJIT_CUDAQ:
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:        ``QJIT_CUDAQ`` objects are created by the :func:`~.qjit` decorator. Please see
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:        def cudaq_backend_info(device, _capabilities) -> BackendInfo:
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:            """The extract_backend_info should not be run by the cuda compiler as it is
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:            (QJITDevice, "extract_backend_info", cudaq_backend_info),
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:            raise NotImplementedError("CUDA tapes do not yet have kwargs.")  # pragma: no cover
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:        # QJIT_CUDAQ(fun).get_jaxpr
frontend/catalyst/third_party/cuda/catalyst_to_cuda_interpreter.py:        catalyst_jaxpr_with_host, out_tree = QJIT_CUDAQ(fun).get_jaxpr(*args)
frontend/catalyst/third_party/cuda/__init__.py:This module contains a CudaQDevice and the qjit
frontend/catalyst/third_party/cuda/__init__.py:    installed_version = version("cuda_quantum")
frontend/catalyst/third_party/cuda/__init__.py:        msg = f"Compiling with incompatible version cuda_quantum=={installed_version}. "
frontend/catalyst/third_party/cuda/__init__.py:        msg += f"Please install compatible version cuda_quantum=={compatible_version}."
frontend/catalyst/third_party/cuda/__init__.py:def cudaqjit(fn=None, **kwargs):
frontend/catalyst/third_party/cuda/__init__.py:    """A decorator for compiling PennyLane and JAX programs using CUDA Quantum.
frontend/catalyst/third_party/cuda/__init__.py:        This feature currently only supports CUDA Quantum version 0.6.
frontend/catalyst/third_party/cuda/__init__.py:        * :class:`nvidia.statevec <NvidiaCuStateVec>`: The NVIDIA CuStateVec GPU simulator
frontend/catalyst/third_party/cuda/__init__.py:                                                       (with support for multi-gpu)
frontend/catalyst/third_party/cuda/__init__.py:        * :class:`nvidia.tensornet <NvidiaCuTensorNet>`: The NVIDIA CuTensorNet GPU simulator
frontend/catalyst/third_party/cuda/__init__.py:        @cudaqjit
frontend/catalyst/third_party/cuda/__init__.py:    >>> @qml.qjit(compiler="cuda_quantum")
frontend/catalyst/third_party/cuda/__init__.py:    Note that CUDA Quantum compilation currently does not have feature parity with Catalyst
frontend/catalyst/third_party/cuda/__init__.py:    from catalyst.third_party.cuda.catalyst_to_cuda_interpreter import interpret
frontend/catalyst/third_party/cuda/__init__.py:class BaseCudaInstructionSet(qml.devices.QubitDevice):
frontend/catalyst/third_party/cuda/__init__.py:    """Base instruction set for CUDA-Quantum devices"""
frontend/catalyst/third_party/cuda/__init__.py:    config = Path(__file__).parent / "cuda_quantum.toml"
frontend/catalyst/third_party/cuda/__init__.py:class SoftwareQQPP(BaseCudaInstructionSet):
frontend/catalyst/third_party/cuda/__init__.py:        This device currently only supports QNodes compiled with CUDA Quantum. For a
frontend/catalyst/third_party/cuda/__init__.py:        @catalyst.third_party.cuda.cudaqjit
frontend/catalyst/third_party/cuda/__init__.py:class NvidiaCuStateVec(BaseCudaInstructionSet):
frontend/catalyst/third_party/cuda/__init__.py:    """The NVIDIA CuStateVec GPU simulator (with support for multi-gpu).
frontend/catalyst/third_party/cuda/__init__.py:        This device currently only supports QNodes compiled with CUDA Quantum. For a multi-GPU
frontend/catalyst/third_party/cuda/__init__.py:        device with support with other compilers, please use ``lightning.gpu``.
frontend/catalyst/third_party/cuda/__init__.py:        multi_gpu (bool): Whether to utilize multiple GPUs.
frontend/catalyst/third_party/cuda/__init__.py:        dev = qml.device("nvidia.custatevec", wires=2)
frontend/catalyst/third_party/cuda/__init__.py:        @catalyst.third_party.cuda.cudaqjit
frontend/catalyst/third_party/cuda/__init__.py:    short_name = "nvidia.custatevec"
frontend/catalyst/third_party/cuda/__init__.py:    def __init__(self, shots=None, wires=None, multi_gpu=False):  # pragma: no cover
frontend/catalyst/third_party/cuda/__init__.py:        self.multi_gpu = multi_gpu
frontend/catalyst/third_party/cuda/__init__.py:        option = "-mgpu" if self.multi_gpu else ""
frontend/catalyst/third_party/cuda/__init__.py:        return f"nvidia{option}"
frontend/catalyst/third_party/cuda/__init__.py:class NvidiaCuTensorNet(BaseCudaInstructionSet):
frontend/catalyst/third_party/cuda/__init__.py:    """The NVIDIA CuTensorNet GPU simulator (with support for matrix product state)
frontend/catalyst/third_party/cuda/__init__.py:        This device currently only supports QNodes compiled with CUDA Quantum.
frontend/catalyst/third_party/cuda/__init__.py:        dev = qml.device("nvidia.cutensornet", wires=2)
frontend/catalyst/third_party/cuda/__init__.py:        @catalyst.third_party.cuda.cudaqjit
frontend/catalyst/third_party/cuda/__init__.py:    short_name = "nvidia.cutensornet"
frontend/catalyst/third_party/cuda/__init__.py:    "cudaqjit",
frontend/catalyst/third_party/cuda/__init__.py:    "BaseCudaInstructionSet",
frontend/catalyst/third_party/cuda/__init__.py:    "NvidiaCuStateVec",
frontend/catalyst/third_party/cuda/__init__.py:    "NvidiaCuTensorNet",
frontend/catalyst/jit.py:        ``lightning.qubit``, ``lightning.kokkos``, ``lightning.gpu``, and ``braket.aws.qubit``. For
frontend/catalyst/jit.py:            ``lightning.gpu``. The default value is None, which means no seeding is performed,

```
