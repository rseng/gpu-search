# https://github.com/bempp/bempp-cl

```console
Dockerfile:        # OpenCL
Dockerfile:RUN python3 -m pip install --no-cache-dir matplotlib pyopencl numpy scipy numba meshio && \
Dockerfile:        # OpenCL
Dockerfile:    python3 -m pip install --no-cache-dir matplotlib pyopencl numpy scipy numba meshio pyopencl && \
INSTALLATION.md:numpy, scipy, numba, pytest, pyopencl, meshio, plotly, jupyter.
INSTALLATION.md:By default bempp-cl uses the pocl (portablecl.org) OpenCL CPU library.
INSTALLATION.md:Other OpenCL capable devices can be used if a corresponding ICD file is
INSTALLATION.md:~/.conda/envs/bempp/conda/etc/OpenCL/vendors directory.
joss/paper.bib:@article{pyopencl,
joss/paper.bib:   title = {{PyCUDA} and {PyOpenCL}: A Scripting-Based Approach to GPU Run-Time Code Generation},
joss/paper.bib:  title = {PyGBe: Python, GPUs and Boundary elements for biomolecular electrostatics},
joss/paper.md:  - OpenCL
joss/paper.md:everything that is required for Calderón preconditioned Maxwell [@maxwellbempp] problems. Bempp-cl uses PyOpenCL [@pyopencl]
joss/paper.md:to just-in-time compile its computational kernels on a wide range of CPU and GPU devices and modern architectures. Alternatively,
joss/paper.md:OpenCL is used as it is able to compile C-based kernels to run on a wide range of CPU and GPU devices, without the need to
joss/paper.md:Operators are assembled using OpenCL or Numba based dense assembly, or via interface to fast multipole methods.
joss/paper.md:Internally, Bempp-cl uses PyOpenCL [@pyopencl] to just-in-time compile its operator assembly routines on a wide range of CPU
joss/paper.md:and GPU compute devices. On systems without OpenCL support, Numba [@numba] is used to just-in-time compile
joss/paper.md:Python-based assembly kernels, giving a slower but still viable alternative to OpenCL.
codemeta.json:{"@context": "https://raw.githubusercontent.com/codemeta/codemeta/master/codemeta.jsonld", "@type": "Code", "author": [{"@id": "0000-0002-3323-2110", "@type": "Person", "email": "t.betcke@ucl.ac.uk", "name": "Timo Betcke", "affiliation": "Department of Mathematics, University College London"}, {"@id": "0000-0002-4658-2443", "@type": "Person", "email": "mws48@cam.ac.uk", "name": "Matthew Scroggs", "affiliation": "Department of Engineering, University of Cambridge"}], "identifier": "", "codeRepository": "https://github.com/bempp/bempp-cl", "datePublished": "2020-09-14", "dateModified": "2021-03-18", "dateCreated": "2020-09-14", "description": "A fast Python based just-in-time compiling boundary element library", "keywords": "Python, OpenCL, boundary element method, partial differential equations, integral equations, numerical analysis", "license": "MIT", "title": "Bempp-cl", "version": "v0.3.2"}
test/conftest.py:        help="Valid values: numba opencl",
test/conftest.py:    elif value == "opencl":
test/conftest.py:        bempp.api.DEFAULT_DEVICE_INTERFACE = "opencl"
test/conftest.py:        raise ValueError("device must be one of: 'numba', 'opencl'")
README.md:optimised just-in-time compiled OpenCL kernels, or alternatively, by just-in-time compiled Numba routines, which are
README.md:automatically used on systems that do not provide OpenCL drivers. User visible functionality is strictly separated from the
benchmarks/conftest.py:        help="Valid values: numba opencl",
benchmarks/conftest.py:    elif value == "opencl":
benchmarks/conftest.py:        bempp.api.DEFAULT_DEVICE_INTERFACE = "opencl"
benchmarks/conftest.py:        raise ValueError("device must be one of: 'numba', 'opencl'")
bempp/api/fmm/exafmm.py:            Either 'numba' or 'opencl'. If not provided, the DEFAULT_DEVICE_INTERFACE
bempp/api/fmm/helpers.py:        elif device_interface == "opencl":
bempp/api/fmm/helpers.py:            evaluator = get_local_interaction_evaluator_opencl(
bempp/api/fmm/helpers.py:            raise ValueError("Device interface must be one of 'numba', 'opencl'.")
bempp/api/fmm/helpers.py:def get_local_interaction_evaluator_opencl(
bempp/api/fmm/helpers.py:    import pyopencl as _cl
bempp/api/fmm/helpers.py:    from bempp.core.opencl_kernels import get_kernel_from_name
bempp/api/fmm/helpers.py:    from bempp.core.opencl_kernels import default_context, default_device
bempp/api/utils/pool.py:        single precision for GPU devices and double precision for CPU devices.
bempp/api/utils/pool.py:        bempp.api.DEVICE_PRECISION_GPU = precision
bempp/api/utils/helpers.py:TypeContainer = _collections.namedtuple("TypeContainer", "real complex opencl")
bempp/api/__init__.py:# Try importing OpenCL routines
bempp/api/__init__.py:    from bempp.core.opencl_kernels import set_default_cpu_device
bempp/api/__init__.py:    from bempp.core.opencl_kernels import set_default_cpu_device_by_name
bempp/api/__init__.py:    from bempp.core.opencl_kernels import set_default_gpu_device_by_name
bempp/api/__init__.py:    from bempp.core.opencl_kernels import set_default_gpu_device
bempp/api/__init__.py:CPU_OPENCL_DRIVER_FOUND = False
bempp/api/__init__.py:GPU_OPENCL_DRIVER_FOUND = False
bempp/api/__init__.py:        from bempp.core.opencl_kernels import find_cpu_driver
bempp/api/__init__.py:        CPU_OPENCL_DRIVER_FOUND = find_cpu_driver()
bempp/api/__init__.py:        from bempp.core.opencl_kernels import find_gpu_driver
bempp/api/__init__.py:        GPU_OPENCL_DRIVER_FOUND = find_gpu_driver()
bempp/api/__init__.py:    if CPU_OPENCL_DRIVER_FOUND:
bempp/api/__init__.py:        DEFAULT_DEVICE_INTERFACE = "opencl"
bempp/api/__init__.py:        "Numba backend activated. For full performance the OpenCL backend with an OpenCL CPU driver is required."
bempp/core/opencl_kernels.py:"""OpenCL routines."""
bempp/core/opencl_kernels.py:import pyopencl as _cl
bempp/core/opencl_kernels.py:_DEFAULT_GPU_DEVICE = None
bempp/core/opencl_kernels.py:_DEFAULT_GPU_CONTEXT = None
bempp/core/opencl_kernels.py:    """Select OpenCL kernel."""
bempp/core/opencl_kernels.py:    if device_type == "gpu":
bempp/core/opencl_kernels.py:            raise RuntimeError("Could not find suitable OpenCL CPU driver.")
bempp/core/opencl_kernels.py:        bempp.api.log(f"OpenCL CPU Device set to: {_DEFAULT_CPU_DEVICE.name}")
bempp/core/opencl_kernels.py:def default_gpu_device():
bempp/core/opencl_kernels.py:    """Return the default GPU device."""
bempp/core/opencl_kernels.py:    global _DEFAULT_GPU_DEVICE
bempp/core/opencl_kernels.py:    global _DEFAULT_GPU_CONTEXT
bempp/core/opencl_kernels.py:    if "BEMPP_GPU_DRIVER" in os.environ:
bempp/core/opencl_kernels.py:        name = os.environ["BEMPP_GPU_DRIVER"]
bempp/core/opencl_kernels.py:    if _DEFAULT_GPU_DEVICE is None:
bempp/core/opencl_kernels.py:            ctx, device = find_gpu_driver(name)
bempp/core/opencl_kernels.py:            raise RuntimeError("Could not find a suitable OpenCL GPU driver.")
bempp/core/opencl_kernels.py:        _DEFAULT_GPU_CONTEXT = ctx
bempp/core/opencl_kernels.py:        _DEFAULT_GPU_DEVICE = device
bempp/core/opencl_kernels.py:        bempp.api.log(f"OpenCL GPU Device set to: {_DEFAULT_GPU_DEVICE.name}")
bempp/core/opencl_kernels.py:    return _DEFAULT_GPU_DEVICE
bempp/core/opencl_kernels.py:    elif device_type == "gpu":
bempp/core/opencl_kernels.py:        return default_gpu_device()
bempp/core/opencl_kernels.py:    elif device_type == "gpu":
bempp/core/opencl_kernels.py:        return default_gpu_context()
bempp/core/opencl_kernels.py:def default_gpu_context():
bempp/core/opencl_kernels.py:    """Return default GPU context."""
bempp/core/opencl_kernels.py:    if _DEFAULT_GPU_CONTEXT is None:
bempp/core/opencl_kernels.py:        default_gpu_device()
bempp/core/opencl_kernels.py:    return _DEFAULT_GPU_CONTEXT
bempp/core/opencl_kernels.py:    """Find the first available CPU OpenCL driver."""
bempp/core/opencl_kernels.py:def find_gpu_driver(name=None):
bempp/core/opencl_kernels.py:    """Find the first available GPU OpenCL driver."""
bempp/core/opencl_kernels.py:            if device.type == _cl.device_type.GPU:
bempp/core/opencl_kernels.py:        raise ValueError(f"Could not find GPU driver containing name {name}.")
bempp/core/opencl_kernels.py:    This method looks for the given string in the available OpenCL
bempp/core/opencl_kernels.py:def set_default_gpu_device_by_name(name):
bempp/core/opencl_kernels.py:    Set default GPU device by name.
bempp/core/opencl_kernels.py:    This method looks for the given string in the available OpenCL
bempp/core/opencl_kernels.py:    global _DEFAULT_GPU_CONTEXT
bempp/core/opencl_kernels.py:    global _DEFAULT_GPU_DEVICE
bempp/core/opencl_kernels.py:        pair = find_gpu_driver(name)
bempp/core/opencl_kernels.py:        raise RuntimeError("No GPU driver with given name found.")
bempp/core/opencl_kernels.py:    _DEFAULT_GPU_CONTEXT = context
bempp/core/opencl_kernels.py:    _DEFAULT_GPU_DEVICE = device
bempp/core/opencl_kernels.py:    vector_width_single = _DEFAULT_GPU_DEVICE.native_vector_width_float
bempp/core/opencl_kernels.py:    vector_width_double = _DEFAULT_GPU_DEVICE.native_vector_width_double
bempp/core/opencl_kernels.py:        f"Default GPU device: {_DEFAULT_GPU_DEVICE.name}. "
bempp/core/opencl_kernels.py:def set_default_gpu_device(platform_index, device_index):
bempp/core/opencl_kernels.py:    """Set the default GPU device."""
bempp/core/opencl_kernels.py:    global _DEFAULT_GPU_DEVICE
bempp/core/opencl_kernels.py:    global _DEFAULT_GPU_CONTEXT
bempp/core/opencl_kernels.py:    _DEFAULT_GPU_CONTEXT = _cl.Context(
bempp/core/opencl_kernels.py:    _DEFAULT_GPU_DEVICE = _DEFAULT_GPU_CONTEXT.devices[0]
bempp/core/opencl_kernels.py:    vector_width_single = _DEFAULT_GPU_DEVICE.native_vector_width_float
bempp/core/opencl_kernels.py:    vector_width_double = _DEFAULT_GPU_DEVICE.native_vector_width_double
bempp/core/opencl_kernels.py:        f"Default GPU device: {_DEFAULT_GPU_DEVICE.name}. "
bempp/core/opencl_assemblers.py:Actual implementation of OpenCL assemblers.
bempp/core/opencl_assemblers.py:import pyopencl as _cl
bempp/core/opencl_assemblers.py:    """Assemble singular part of integral operators with OpenCL."""
bempp/core/opencl_assemblers.py:    from bempp.core.opencl_kernels import get_kernel_from_operator_descriptor
bempp/core/opencl_assemblers.py:    from bempp.core.opencl_kernels import default_context, default_device
bempp/core/opencl_assemblers.py:    # Initialize OpenCL Buffers
bempp/core/opencl_assemblers.py:    """Assemble dense with OpenCL."""
bempp/core/opencl_assemblers.py:    from bempp.core.opencl_kernels import get_kernel_from_operator_descriptor
bempp/core/opencl_assemblers.py:    from bempp.core.opencl_kernels import (
bempp/core/opencl_assemblers.py:    if bempp.api.BOUNDARY_OPERATOR_DEVICE_TYPE == "gpu":
bempp/core/opencl_assemblers.py:        device_type = "gpu"
bempp/core/opencl_assemblers.py:    """Assemble dense with OpenCL."""
bempp/core/opencl_assemblers.py:    from bempp.core.opencl_kernels import get_kernel_from_name
bempp/core/opencl_assemblers.py:    from bempp.core.opencl_kernels import get_kernel_from_operator_descriptor
bempp/core/opencl_assemblers.py:    from bempp.core.opencl_kernels import (
bempp/core/opencl_assemblers.py:    if bempp.api.POTENTIAL_OPERATOR_DEVICE_TYPE == "gpu":
bempp/core/opencl_assemblers.py:        device_type = "gpu"
bempp/core/dispatcher.py:    if interface_type == "opencl":
bempp/core/dispatcher.py:        from bempp.core.opencl_assemblers import singular_assembler
bempp/core/dispatcher.py:        raise ValueError("Device interface must be one of 'numba', 'opencl'.")
bempp/core/dispatcher.py:    if interface_type == "opencl":
bempp/core/dispatcher.py:        from bempp.core.opencl_assemblers import dense_assembler
bempp/core/dispatcher.py:        raise ValueError("Device interface must be one of 'numba', 'opencl'.")
bempp/core/dispatcher.py:    elif interface_type == "opencl":
bempp/core/dispatcher.py:        from bempp.core.opencl_assemblers import potential_assembler
bempp/core/dispatcher.py:        raise ValueError("Device interface must be one of 'numba', 'opencl'.")
examples/test.py:    # Examples in this list will be skipped as the problems are very large or require GPUs
examples/test.py:    if script in ["reentrant_cube_capacity.py", "opencl_benchmark.py", "dirichlet_weak_imposition.py"]:
examples/other/opencl_benchmark.py:# # Bempp OpenCL performance benchmarks
examples/other/opencl_benchmark.py:# frequency of 5GHz. The GPU device is an NVIDIA Quadro RTX 3000 GPU with 6GB Ram.
examples/other/opencl_benchmark.py:# As OpenCL CPU driver we test both POCL (in Version 1.5) and the Intel OpenCL CPU driver, both with default vectorization options.
examples/other/opencl_benchmark.py:# We are testing all operators in single and double precision. For the GPU we only perform single precision tests as it is significantly
examples/other/opencl_benchmark.py:driver_labels = ["Portable Computing Language", "Intel(R) OpenCL"]
examples/other/opencl_benchmark.py:driver_labels = ["Portable Computing Language", "Intel(R) OpenCL", "NVIDIA CUDA"]
examples/other/opencl_benchmark.py:bempp.api.set_default_gpu_device_by_name("NVIDIA CUDA")
examples/other/opencl_benchmark.py:        if driver_name == "NVIDIA CUDA":
examples/other/opencl_benchmark.py:            bempp.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "gpu"

```
