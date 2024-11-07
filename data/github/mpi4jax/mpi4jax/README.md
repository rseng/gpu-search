# https://github.com/mpi4jax/mpi4jax

```console
setup.py:# Cuda detection
setup.py:def get_cuda_paths_from_nvidia_pypi():
setup.py:    # try to check if nvidia-cuda-nvcc-cu* is installed
setup.py:    # If the pip package nvidia-cuda-nvcc-cu11 is installed, it should have
setup.py:    # both of the things XLA looks for in the cuda path, namely bin/ptxas and
setup.py:    maybe_cuda_paths = [
setup.py:        depot_path / "nvidia" / "cuda_nvcc",
setup.py:        depot_path / "nvidia" / "cuda_runtime",
setup.py:    if all(p.is_dir() for p in maybe_cuda_paths):
setup.py:        return [str(p) for p in maybe_cuda_paths]
setup.py:def get_cuda_path():
setup.py:    cuda_path_default = None
setup.py:        cuda_path_default = os.path.normpath(
setup.py:    cuda_path = os.getenv("CUDA_PATH", "")  # Nvidia default on Windows
setup.py:    if len(cuda_path) == 0:
setup.py:        cuda_path = os.getenv("CUDA_ROOT", "")  # Nvidia default on Windows
setup.py:    if len(cuda_path) > 0 and cuda_path != cuda_path_default:
setup.py:            "nvcc path != CUDA_PATH",
setup.py:            "nvcc path: %s" % cuda_path_default,
setup.py:            "CUDA_PATH: %s" % cuda_path,
setup.py:    if os.path.exists(cuda_path):
setup.py:        _cuda_path = cuda_path
setup.py:    elif cuda_path_default is not None:
setup.py:        _cuda_path = cuda_path_default
setup.py:    elif os.path.exists("/usr/local/cuda"):
setup.py:        _cuda_path = "/usr/local/cuda"
setup.py:        _cuda_path = None
setup.py:    return _cuda_path
setup.py:def get_cuda_info():
setup.py:    cuda_info = {"compile": [], "libdirs": [], "libs": [], "rpaths": []}
setup.py:    # First check if the nvidia-cuda-nvcc-cu* package is installed. We ignore CUDA_ROOT
setup.py:    cuda_paths = get_cuda_paths_from_nvidia_pypi()
setup.py:    # If not, try to find the CUDA_PATH by hand
setup.py:    if len(cuda_paths) > 0:
setup.py:        nvidia_pypi_package = True
setup.py:        nvidia_pypi_package = False
setup.py:        _cuda_path = get_cuda_path()
setup.py:        if _cuda_path is None:
setup.py:            cuda_paths = []
setup.py:            cuda_paths = [_cuda_path]
setup.py:    if len(cuda_paths) == 0:
setup.py:        return cuda_info
setup.py:    for cuda_path in cuda_paths:
setup.py:        incdir = os.path.join(cuda_path, "include")
setup.py:            cuda_info["compile"].append(incdir)
setup.py:            full_dir = os.path.join(cuda_path, libdir)
setup.py:                cuda_info["libdirs"].append(full_dir)
setup.py:    # We need to link against libcudart.so
setup.py:    #   - If we are using standard CUDA installations, we simply add a link flag to
setup.py:    #     libcudart.so
setup.py:    #   - If we are using the nvidia-cuda-nvcc-cu* package, we need to find the exact
setup.py:    #     version of libcudart.so to link against because the the package does not provide
setup.py:    #     a generic binding to libcudart.so but only libcudart.so.XX.
setup.py:    # Moreover, if we are using nvidia-cuda-nvcc we must add @rpath (runtime search paths)
setup.py:    # because we do not expect the user to set LD_LIBRARY_PATH to the nvidia-cuda-nvcc
setup.py:    if not nvidia_pypi_package:
setup.py:        cuda_info["libs"].append("cudart")
setup.py:        possible_libcudart = find_files(cuda_paths, "libcudart.so*")
setup.py:        if "libcudart.so" in possible_libcudart:
setup.py:            # In theory with nvidia-cuda-nvcc-cu12 we should never reach this point
setup.py:            cuda_info["libs"].append("cudart")
setup.py:        elif len(possible_libcudart) > 0:
setup.py:            # This should be the standard case for nvidia-cuda-nvcc-cu*
setup.py:            # where we find a library libcudart.so.XX . The syntax to link to a
setup.py:            # specific version is -l:libcudart.so.XX
setup.py:            lib_to_link = possible_libcudart[0]
setup.py:            cuda_info["libs"].append(f":{os.path.basename(lib_to_link)}")
setup.py:            cuda_info["rpaths"].append(os.path.dirname(lib_to_link))
setup.py:            # If we cannot find libcudart.so, we cannot build the extension
setup.py:            # This should never happen with nvidia-cuda-nvcc-cu* package
setup.py:            cuda_info["libs"].append("cudart")
setup.py:    print("\n\nCUDA INFO:", cuda_info, "\n\n")
setup.py:    return cuda_info
setup.py:# /end Cuda detection
setup.py:    cuda_info = get_cuda_info()
setup.py:    if cuda_info["compile"] and cuda_info["libdirs"]:
setup.py:        if len(cuda_info["rpaths"]) > 0:
setup.py:            extra_extension_args["runtime_library_dirs"] = cuda_info["rpaths"]
setup.py:                name=f"{CYTHON_SUBMODULE_NAME}.mpi_xla_bridge_cuda",
setup.py:                sources=[f"{CYTHON_SUBMODULE_PATH}/mpi_xla_bridge_cuda.pyx"],
setup.py:                include_dirs=cuda_info["compile"],
setup.py:                library_dirs=cuda_info["libdirs"],
setup.py:                libraries=cuda_info["libs"],
setup.py:        print_warning("CUDA path not found", "(GPU extensions will not be built)")
setup.py:        "Environment :: GPU :: NVIDIA CUDA",
README.rst:``mpi4jax`` enables zero-copy, multi-host communication of `JAX <https://jax.readthedocs.io/>`_ arrays, even from jitted code and from GPU memory.
README.rst:With ``mpi4jax``, you can scale your JAX-based simulations to *entire CPU and GPU clusters* (without ever leaving ``jax.jit``).
README.rst:   # pip install -U 'jax[cuda12]'
README.rst:   # pip install -U 'jax[cuda12_local]'
README.rst:   $ CUDA_ROOT=XXX pip install mpi4jax
README.rst:(for more informations on jax GPU distributions, `see the JAX installation instructions <https://github.com/google/jax#installation>`_)
docs/api.rst:has_cuda_support
docs/api.rst:.. autofunction:: mpi4jax.has_cuda_support
docs/shallow-water.rst:    WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
docs/shallow-water.rst:    WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
docs/shallow-water.rst:    WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
docs/shallow-water.rst:    WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
docs/shallow-water.rst:Using the shallow water solver, we can observe how the performance behaves when we increase the number of MPI processes or switch to GPUs. Here we show some benchmark results on a machine with 2x Intel Xeon E5-2650 v4 CPUs and 2x NVIDIA Tesla P100 GPUs.
docs/shallow-water.rst:    # GPU
docs/shallow-water.rst:    $ JAX_PLATFORM_NAME=gpu mpirun -n 1 -- python examples/shallow_water.py --benchmark
docs/shallow-water.rst:    $ JAX_PLATFORM_NAME=gpu MPI4JAX_USE_CUDA_MPI=0 mpirun -n 2 -- python examples/shallow_water.py --benchmark
docs/shallow-water.rst:    $ JAX_PLATFORM_NAME=gpu MPI4JAX_USE_CUDA_MPI=1 mpirun -n 2 -- python examples/shallow_water.py --benchmark
docs/installation.rst:Start by `installing a suitable version of JAX and jaxlib <https://github.com/google/jax#installation>`_. If you don't plan on using ``mpi4jax`` on GPU, the following will do:
docs/installation.rst:Installation with NVIDIA GPU support (CUDA)
docs/installation.rst:   There are 3 ways to install jax with CUDA support:
docs/installation.rst:   - using a pypi-distributed CUDA installation (suggested by jax developers) ``pip install -U 'jax[cuda12]'`` 
docs/installation.rst:   - using the locally-installed CUDA version, which must be compatible with jax. ``pip install -U 'jax[cuda12_local]'`` 
docs/installation.rst:To use ``mpi4jax`` with pypi-distributed nvidia packages, which is the preferred way to install jax, you **must** install ``mpi4jax`` disabling
docs/installation.rst:the build-time-isolation in order for it to link to the libraries in the nvidia-cuda-nvcc-cu12 package. To do so, run the following command:
docs/installation.rst:   # assuming pip install -U 'jax[cuda12]' has been run
docs/installation.rst:Alternatively, if you want to install ``mpi4jax`` with a locally-installed CUDA version, you can run the following command we need 
docs/installation.rst:to be able to locate the CUDA headers on your system. If they are not detected automatically, you can set the environment 
docs/installation.rst:variable :envvar:`CUDA_ROOT` when installing ``mpi4jax``::
docs/installation.rst:   $ CUDA_ROOT=/usr/local/cuda pip install --no-build-isolation mpi4jax
docs/installation.rst:This is sufficient for most situations. However, ``mpi4jax`` will copy all data from GPU to CPU and back before and after invoking MPI.
docs/installation.rst:If this is a bottleneck in your application, you can build MPI with CUDA support and *communicate directly from GPU memory*. This requires that you re-build the entire stack:
docs/installation.rst:- Your MPI library, e.g. `OpenMPI <https://www.open-mpi.org/faq/?category=buildcuda>`_, with CUDA support.
docs/installation.rst:- ``mpi4py``, linked to your CUDA-enabled MPI installation.
docs/installation.rst:   Read :ref:`here <gpu-usage>` on how to use zero-copy GPU communication after installation.
docs/installation.rst:Installation with Intel GPU/XPU support
docs/installation.rst:``mpi4jax`` supports communication of JAX arrays stored in Intel GPU/XPU memory, via JAX's ``xpu`` backend.
docs/installation.rst:- Optionally, `Intel MPI <https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/mpi-library.html>`__ with Intel XPU/GPU support.
docs/installation.rst:  To leverage this, you also need to rebuild `mpi4py <https://mpi4py.readthedocs.io/en/stable/install.html>`__ to ensure it is linked to the XPU/GPU aware MPI implementation.
docs/sharp-bits.rst:.. _gpu-usage:
docs/sharp-bits.rst:Using CUDA MPI
docs/sharp-bits.rst:``mpi4jax`` is able to communicate data directly from and to GPU memory. :doc:`This requires that MPI, JAX, and mpi4jax are built with CUDA support. <installation>`
docs/sharp-bits.rst:Currently, we cannot detect whether MPI was built with CUDA support.
docs/sharp-bits.rst:Therefore, by default, ``mpi4jax`` will not read directly from GPU
docs/sharp-bits.rst:If you are certain that the underlying MPI library was built with CUDA
docs/sharp-bits.rst:   $ export MPI4JAX_USE_CUDA_MPI=1
docs/sharp-bits.rst:Data will then be copied directly from GPU to GPU. If your MPI library
docs/sharp-bits.rst:does not have CUDA support, you will receive a segmentation fault when
docs/sharp-bits.rst:trying to access GPU memory.
docs/sharp-bits.rst:and Intel GPU memory. This requires that you have installed MPI that is
docs/sharp-bits.rst:Intel GPU/XPU aware (MPI calls can work directly with XPU/GPU memory)
docs/sharp-bits.rst:Currently, we cannot detect whether MPI is XPU/GPU aware. Therefore, by
docs/sharp-bits.rst:default, ``mpi4jax`` will not read directly from XPU/GPU memory, but
docs/sharp-bits.rst:If you are certain that the underlying MPI library is XPU/GPU aware
docs/sharp-bits.rst:cannot work with Intel GPU/XPU buffers, you will receive a segmentation
docs/sharp-bits.rst:fault when trying to access mentioned GPU/XPU memory.
tests/test_decorators.py:def test_ensure_cuda_ext(monkeypatch):
tests/test_decorators.py:    from mpi4jax._src.decorators import ensure_cuda_ext
tests/test_decorators.py:        m.setattr(xla_bridge, "HAS_CUDA_EXT", False)
tests/test_decorators.py:            ensure_cuda_ext()
tests/test_decorators.py:        assert "GPU extensions could not be imported" in str(excinfo.value)
tests/test_jax_compat.py:    "0.1.61+cuda110": (0, 1, 61),
tests/test_has_cuda.py:    from mpi4jax import has_cuda_support
tests/test_has_cuda.py:    assert isinstance(has_cuda_support(), bool)
tests/conftest.py:    # don't hog memory if running on GPU
conf/install-cuda-ubuntu.sh:# Ideally choose from the list of meta-packages to minimise variance between cuda versions (although it does change too)
conf/install-cuda-ubuntu.sh:CUDA_PACKAGES_IN=(
conf/install-cuda-ubuntu.sh:    "cuda"
conf/install-cuda-ubuntu.sh:## Select CUDA version
conf/install-cuda-ubuntu.sh:# Get the cuda version from the environment as $cuda.
conf/install-cuda-ubuntu.sh:CUDA_VERSION_MAJOR_MINOR=${cuda}
conf/install-cuda-ubuntu.sh:CUDA_MAJOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f1)
conf/install-cuda-ubuntu.sh:CUDA_MINOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f2)
conf/install-cuda-ubuntu.sh:CUDA_PATCH=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f3)
conf/install-cuda-ubuntu.sh:echo "CUDA_MAJOR: ${CUDA_MAJOR}"
conf/install-cuda-ubuntu.sh:echo "CUDA_MINOR: ${CUDA_MINOR}"
conf/install-cuda-ubuntu.sh:echo "CUDA_PATCH: ${CUDA_PATCH}"
conf/install-cuda-ubuntu.sh:# If we don't know the CUDA_MAJOR or MINOR, error.
conf/install-cuda-ubuntu.sh:if [ -z "${CUDA_MAJOR}" ] ; then
conf/install-cuda-ubuntu.sh:    echo "Error: Unknown CUDA Major version. Aborting."
conf/install-cuda-ubuntu.sh:if [ -z "${CUDA_MINOR}" ] ; then
conf/install-cuda-ubuntu.sh:    echo "Error: Unknown CUDA Minor version. Aborting."
conf/install-cuda-ubuntu.sh:## Select CUDA packages to install
conf/install-cuda-ubuntu.sh:CUDA_PACKAGES=""
conf/install-cuda-ubuntu.sh:for package in "${CUDA_PACKAGES_IN[@]}"
conf/install-cuda-ubuntu.sh:    CUDA_PACKAGES+=" ${package}-${CUDA_MAJOR}-${CUDA_MINOR}"
conf/install-cuda-ubuntu.sh:echo "CUDA_PACKAGES ${CUDA_PACKAGES}"
conf/install-cuda-ubuntu.sh:PIN_FILENAME="cuda-ubuntu${UBUNTU_VERSION}.pin"
conf/install-cuda-ubuntu.sh:PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/${PIN_FILENAME}"
conf/install-cuda-ubuntu.sh:APT_KEY_URL="http://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/3bf863cc.pub"
conf/install-cuda-ubuntu.sh:REPO_URL="http://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/"
conf/install-cuda-ubuntu.sh:echo "Adding CUDA Repository"
conf/install-cuda-ubuntu.sh:sudo mv ${PIN_FILENAME} /etc/apt/preferences.d/cuda-repository-pin-600
conf/install-cuda-ubuntu.sh:echo "Installing CUDA packages ${CUDA_PACKAGES}"
conf/install-cuda-ubuntu.sh:sudo apt-get -y install ${CUDA_PACKAGES}
conf/install-cuda-ubuntu.sh:    echo "CUDA Installation Error."
conf/install-cuda-ubuntu.sh:CUDA_PATH=/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}
conf/install-cuda-ubuntu.sh:echo "CUDA_PATH=${CUDA_PATH}"
conf/install-cuda-ubuntu.sh:export CUDA_PATH=${CUDA_PATH}
conf/install-cuda-ubuntu.sh:export PATH="$CUDA_PATH/bin:$PATH"
conf/install-cuda-ubuntu.sh:export LD_LIBRARY_PATH="$CUDA_PATH/lib:$LD_LIBRARY_PATH"
examples/shallow_water.py:# on GPU, put each process on its own device
examples/shallow_water.py:os.environ["CUDA_VISIBLE_DEVICES"] = str(mpi_rank)
mpi4jax/_src/xla_bridge/cuda_runtime_api.pxd:cdef extern from "cuda_runtime_api.h":
mpi4jax/_src/xla_bridge/cuda_runtime_api.pxd:    cdef enum cudaError:
mpi4jax/_src/xla_bridge/cuda_runtime_api.pxd:        cudaSuccess = 0
mpi4jax/_src/xla_bridge/cuda_runtime_api.pxd:    cdef enum cudaMemcpyKind:
mpi4jax/_src/xla_bridge/cuda_runtime_api.pxd:        cudaMemcpyHostToHost = 0
mpi4jax/_src/xla_bridge/cuda_runtime_api.pxd:        cudaMemcpyHostToDevice = 1
mpi4jax/_src/xla_bridge/cuda_runtime_api.pxd:        cudaMemcpyDeviceToHost = 2
mpi4jax/_src/xla_bridge/cuda_runtime_api.pxd:        cudaMemcpyDeviceToDevice = 3
mpi4jax/_src/xla_bridge/cuda_runtime_api.pxd:        cudaMemcpyDefault = 4
mpi4jax/_src/xla_bridge/cuda_runtime_api.pxd:    ctypedef cudaError cudaError_t
mpi4jax/_src/xla_bridge/cuda_runtime_api.pxd:    ctypedef void* cudaStream_t
mpi4jax/_src/xla_bridge/cuda_runtime_api.pxd:    cudaError_t cudaStreamSynchronize(cudaStream_t stream) nogil
mpi4jax/_src/xla_bridge/cuda_runtime_api.pxd:    cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) nogil
mpi4jax/_src/xla_bridge/cuda_runtime_api.pxd:    cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) nogil
mpi4jax/_src/xla_bridge/cuda_runtime_api.pxd:    const char* cudaGetErrorName(cudaError_t error) nogil
mpi4jax/_src/xla_bridge/cuda_runtime_api.pxd:    const char* cudaGetErrorString(cudaError_t error) nogil
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:from .cuda_runtime_api cimport (
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    cudaGetErrorName,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    cudaGetErrorString,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    cudaError_t,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    cudaMemcpyAsync,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    cudaMemcpyDeviceToDevice,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    cudaMemcpyDeviceToHost,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    cudaMemcpyKind,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    cudaMemcpyHostToDevice,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    cudaStream_t,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    cudaStreamSynchronize,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    cudaSuccess,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:cpdef inline unicode get_error_name(cudaError_t ierr):
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    return py_string(cudaGetErrorName(ierr))
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:cpdef inline unicode get_error_string(cudaError_t ierr):
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    return py_string(cudaGetErrorString(ierr))
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:cdef inline cudaError_t checked_cuda_memcpy(void* dst, void* src, size_t count,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:                                            cudaMemcpyKind kind, cudaStream_t stream, MPI_Comm comm) nogil:
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    cdef cudaError_t ierr
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    ierr = cudaMemcpyAsync(dst, src, count, kind, stream)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    if ierr != cudaSuccess:
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:                f"cudaMemcpyAsync failed with the following error:\n"
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    ierr = checked_cuda_stream_synchronize(stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:cdef inline cudaError_t checked_cuda_stream_synchronize(
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    cudaStream_t stream, MPI_Comm comm
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    cdef cudaError_t ierr
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    ierr = cudaStreamSynchronize(stream)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:    if ierr != cudaSuccess:
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:                f"cudaStreamSynchronize failed with the following error:\n"
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:cdef void mpi_allgather_cuda(cudaStream_t stream, void** buffers,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_memcpy(in_buf, data, sendbytes, cudaMemcpyDeviceToHost, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_stream_synchronize(stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_memcpy(out_data, out_buf, recvbytes, cudaMemcpyHostToDevice, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:cdef void mpi_allreduce_cuda(cudaStream_t stream, void** buffers,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_memcpy(in_buf, data, count, cudaMemcpyDeviceToHost, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_stream_synchronize(stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_memcpy(out_data, out_buf, count, cudaMemcpyHostToDevice, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:cdef void mpi_alltoall_cuda(cudaStream_t stream, void** buffers,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_memcpy(in_buf, data, sendbytes, cudaMemcpyDeviceToHost, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_stream_synchronize(stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_memcpy(out_data, out_buf, recvbytes, cudaMemcpyHostToDevice, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:cdef void mpi_barrier_cuda(cudaStream_t stream, void** buffers,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:cdef void mpi_bcast_cuda(cudaStream_t stream, void** buffers,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:            checked_cuda_memcpy(buf, data, count, cudaMemcpyDeviceToHost, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_stream_synchronize(stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:            checked_cuda_memcpy(out_data, buf, count, cudaMemcpyHostToDevice, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:cdef void mpi_gather_cuda(cudaStream_t stream, void** buffers,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_memcpy(in_buf, data, sendbytes, cudaMemcpyDeviceToHost, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_stream_synchronize(stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:            checked_cuda_memcpy(
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:                out_data, out_buf, recvbytes, cudaMemcpyHostToDevice, stream, comm
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:cdef void mpi_recv_cuda(cudaStream_t stream, void** buffers,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_stream_synchronize(stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_memcpy(out_buf, recvbuf, count, cudaMemcpyHostToDevice, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:cdef void mpi_reduce_cuda(cudaStream_t stream, void** buffers,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_memcpy(in_buf, data, count, cudaMemcpyDeviceToHost, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_stream_synchronize(stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:            checked_cuda_memcpy(out_data, out_buf, count, cudaMemcpyHostToDevice, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:cdef void mpi_scan_cuda(cudaStream_t stream, void** buffers,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_memcpy(in_buf, data, count, cudaMemcpyDeviceToHost, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_stream_synchronize(stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_memcpy(out_data, out_buf, count, cudaMemcpyHostToDevice, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:cdef void mpi_scatter_cuda(cudaStream_t stream, void** buffers,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_memcpy(in_buf, data, sendbytes, cudaMemcpyDeviceToHost, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_stream_synchronize(stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_memcpy(out_data, out_buf, recvbytes, cudaMemcpyHostToDevice, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:cdef void mpi_send_cuda(cudaStream_t stream, void** buffers,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_memcpy(sendbuf, data, count, cudaMemcpyDeviceToHost, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_stream_synchronize(stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:cdef void mpi_sendrecv_cuda(cudaStream_t stream, void** buffers,
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_memcpy(sendbuf, in_buf, bytes_send, cudaMemcpyDeviceToHost, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_stream_synchronize(stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:        checked_cuda_memcpy(out_buf, recvbuf, bytes_recv, cudaMemcpyHostToDevice, stream, comm)
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:declare_custom_call_target("mpi_allgather", <void*>(mpi_allgather_cuda))
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:declare_custom_call_target("mpi_allreduce", <void*>(mpi_allreduce_cuda))
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:declare_custom_call_target("mpi_alltoall", <void*>(mpi_alltoall_cuda))
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:declare_custom_call_target("mpi_barrier", <void*>(mpi_barrier_cuda))
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:declare_custom_call_target("mpi_bcast", <void*>(mpi_bcast_cuda))
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:declare_custom_call_target("mpi_gather", <void*>(mpi_gather_cuda))
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:declare_custom_call_target("mpi_recv", <void*>(mpi_recv_cuda))
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:declare_custom_call_target("mpi_reduce", <void*>(mpi_reduce_cuda))
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:declare_custom_call_target("mpi_scan", <void*>(mpi_scan_cuda))
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:declare_custom_call_target("mpi_scatter", <void*>(mpi_scatter_cuda))
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:declare_custom_call_target("mpi_send", <void*>(mpi_send_cuda))
mpi4jax/_src/xla_bridge/mpi_xla_bridge_cuda.pyx:declare_custom_call_target("mpi_sendrecv", <void*>(mpi_sendrecv_cuda))
mpi4jax/_src/xla_bridge/__init__.py:    from . import mpi_xla_bridge_cuda
mpi4jax/_src/xla_bridge/__init__.py:    HAS_CUDA_EXT = False
mpi4jax/_src/xla_bridge/__init__.py:    HAS_CUDA_EXT = True
mpi4jax/_src/xla_bridge/__init__.py:if HAS_CUDA_EXT:
mpi4jax/_src/xla_bridge/__init__.py:    for name, fn in mpi_xla_bridge_cuda.custom_call_targets.items():
mpi4jax/_src/xla_bridge/__init__.py:        register_custom_call_target(name, fn, platform="CUDA", api_version=0)
mpi4jax/_src/decorators.py:_cuda_mpi_setup_done = False
mpi4jax/_src/decorators.py:def ensure_cuda_ext():
mpi4jax/_src/decorators.py:    from .xla_bridge import HAS_CUDA_EXT
mpi4jax/_src/decorators.py:    if not HAS_CUDA_EXT:
mpi4jax/_src/decorators.py:            "The mpi4jax GPU extensions could not be imported. "
mpi4jax/_src/decorators.py:            "Please re-build mpi4jax with CUDA support and try again."
mpi4jax/_src/decorators.py:def setup_cuda_mpi():
mpi4jax/_src/decorators.py:    global _cuda_mpi_setup_done
mpi4jax/_src/decorators.py:    if _cuda_mpi_setup_done:
mpi4jax/_src/decorators.py:    _cuda_mpi_setup_done = True
mpi4jax/_src/decorators.py:    cuda_copy_behavior = os.getenv("MPI4JAX_USE_CUDA_MPI", "")
mpi4jax/_src/decorators.py:    if _is_truthy(cuda_copy_behavior):
mpi4jax/_src/decorators.py:        has_cuda_mpi = True
mpi4jax/_src/decorators.py:    elif _is_falsy(cuda_copy_behavior):
mpi4jax/_src/decorators.py:        has_cuda_mpi = False
mpi4jax/_src/decorators.py:        has_cuda_mpi = False
mpi4jax/_src/decorators.py:            "Not using CUDA-enabled MPI. "
mpi4jax/_src/decorators.py:            "If you are sure that your MPI library is built with CUDA support, "
mpi4jax/_src/decorators.py:            "set MPI4JAX_USE_CUDA_MPI=1. To silence this warning, "
mpi4jax/_src/decorators.py:            "set MPI4JAX_USE_CUDA_MPI=0."
mpi4jax/_src/decorators.py:    from .xla_bridge import mpi_xla_bridge_cuda
mpi4jax/_src/decorators.py:    mpi_xla_bridge_cuda.set_copy_to_host(not has_cuda_mpi)
mpi4jax/_src/decorators.py:def translation_rule_cuda(func):
mpi4jax/_src/decorators.py:    """XLA primitive translation rule on GPU for mpi4jax custom calls.
mpi4jax/_src/decorators.py:        ensure_cuda_ext,
mpi4jax/_src/decorators.py:        setup_cuda_mpi,
mpi4jax/_src/utils.py:def has_cuda_support() -> bool:
mpi4jax/_src/utils.py:    """Returns True if mpi4jax is built with CUDA support and can be used with GPU-based
mpi4jax/_src/utils.py:    return xla_bridge.HAS_CUDA_EXT
mpi4jax/_src/collective_ops/reduce.py:    translation_rule_cuda,
mpi4jax/_src/collective_ops/reduce.py:mpi_reduce_xla_encode_cuda = translation_rule_cuda(mpi_reduce_xla_encode_device)
mpi4jax/_src/collective_ops/reduce.py:register_lowering(mpi_reduce_p, mpi_reduce_xla_encode_cuda, platform="cuda")
mpi4jax/_src/collective_ops/gather.py:    translation_rule_cuda,
mpi4jax/_src/collective_ops/gather.py:mpi_gather_xla_encode_cuda = translation_rule_cuda(mpi_gather_xla_encode_device)
mpi4jax/_src/collective_ops/gather.py:register_lowering(mpi_gather_p, mpi_gather_xla_encode_cuda, platform="cuda")
mpi4jax/_src/collective_ops/send.py:    translation_rule_cuda,
mpi4jax/_src/collective_ops/send.py:mpi_send_xla_encode_cuda = translation_rule_cuda(mpi_send_xla_encode_device)
mpi4jax/_src/collective_ops/send.py:register_lowering(mpi_send_p, mpi_send_xla_encode_cuda, platform="cuda")
mpi4jax/_src/collective_ops/bcast.py:    translation_rule_cuda,
mpi4jax/_src/collective_ops/bcast.py:mpi_bcast_xla_encode_cuda = translation_rule_cuda(mpi_bcast_xla_encode_device)
mpi4jax/_src/collective_ops/bcast.py:register_lowering(mpi_bcast_p, mpi_bcast_xla_encode_cuda, platform="cuda")
mpi4jax/_src/collective_ops/barrier.py:    translation_rule_cuda,
mpi4jax/_src/collective_ops/barrier.py:mpi_barrier_xla_encode_cuda = translation_rule_cuda(mpi_barrier_xla_encode_device)
mpi4jax/_src/collective_ops/barrier.py:register_lowering(mpi_barrier_p, mpi_barrier_xla_encode_cuda, platform="cuda")
mpi4jax/_src/collective_ops/allreduce.py:    translation_rule_cuda,
mpi4jax/_src/collective_ops/allreduce.py:mpi_allreduce_xla_encode_cuda = translation_rule_cuda(mpi_allreduce_xla_encode_device)
mpi4jax/_src/collective_ops/allreduce.py:register_lowering(mpi_allreduce_p, mpi_allreduce_xla_encode_cuda, platform="cuda")
mpi4jax/_src/collective_ops/scatter.py:    translation_rule_cuda,
mpi4jax/_src/collective_ops/scatter.py:mpi_scatter_xla_encode_cuda = translation_rule_cuda(mpi_scatter_xla_encode_device)
mpi4jax/_src/collective_ops/scatter.py:register_lowering(mpi_scatter_p, mpi_scatter_xla_encode_cuda, platform="cuda")
mpi4jax/_src/collective_ops/sendrecv.py:    translation_rule_cuda,
mpi4jax/_src/collective_ops/sendrecv.py:mpi_sendrecv_xla_encode_cuda = translation_rule_cuda(mpi_sendrecv_xla_encode_device)
mpi4jax/_src/collective_ops/sendrecv.py:register_lowering(mpi_sendrecv_p, mpi_sendrecv_xla_encode_cuda, platform="cuda")
mpi4jax/_src/collective_ops/scan.py:    translation_rule_cuda,
mpi4jax/_src/collective_ops/scan.py:mpi_scan_xla_encode_cuda = translation_rule_cuda(mpi_scan_xla_encode_device)
mpi4jax/_src/collective_ops/scan.py:register_lowering(mpi_scan_p, mpi_scan_xla_encode_cuda, platform="cuda")
mpi4jax/_src/collective_ops/alltoall.py:    translation_rule_cuda,
mpi4jax/_src/collective_ops/alltoall.py:mpi_alltoall_xla_encode_cuda = translation_rule_cuda(mpi_alltoall_xla_encode_device)
mpi4jax/_src/collective_ops/alltoall.py:register_lowering(mpi_alltoall_p, mpi_alltoall_xla_encode_cuda, platform="cuda")
mpi4jax/_src/collective_ops/allgather.py:    translation_rule_cuda,
mpi4jax/_src/collective_ops/allgather.py:mpi_allgather_xla_encode_cuda = translation_rule_cuda(mpi_allgather_xla_encode_device)
mpi4jax/_src/collective_ops/allgather.py:register_lowering(mpi_allgather_p, mpi_allgather_xla_encode_cuda, platform="cuda")
mpi4jax/_src/collective_ops/recv.py:    translation_rule_cuda,
mpi4jax/_src/collective_ops/recv.py:mpi_recv_xla_encode_cuda = translation_rule_cuda(mpi_recv_xla_encode_device)
mpi4jax/_src/collective_ops/recv.py:register_lowering(mpi_recv_p, mpi_recv_xla_encode_cuda, platform="cuda")
mpi4jax/_src/__init__.py:from .utils import has_cuda_support, has_sycl_support  # noqa: F401, E402
mpi4jax/experimental/notoken/collective_ops/reduce.py:    translation_rule_cuda,
mpi4jax/experimental/notoken/collective_ops/reduce.py:mpi_reduce_xla_encode_cuda = translation_rule_cuda(mpi_reduce_xla_encode_device)
mpi4jax/experimental/notoken/collective_ops/reduce.py:register_lowering(mpi_reduce_p, mpi_reduce_xla_encode_cuda, platform="cuda")
mpi4jax/experimental/notoken/collective_ops/gather.py:    translation_rule_cuda,
mpi4jax/experimental/notoken/collective_ops/gather.py:mpi_gather_xla_encode_cuda = translation_rule_cuda(mpi_gather_xla_encode_device)
mpi4jax/experimental/notoken/collective_ops/gather.py:register_lowering(mpi_gather_p, mpi_gather_xla_encode_cuda, platform="cuda")
mpi4jax/experimental/notoken/collective_ops/send.py:    translation_rule_cuda,
mpi4jax/experimental/notoken/collective_ops/send.py:mpi_send_xla_encode_cuda = translation_rule_cuda(mpi_send_xla_encode_device)
mpi4jax/experimental/notoken/collective_ops/send.py:register_lowering(mpi_send_p, mpi_send_xla_encode_cuda, platform="cuda")
mpi4jax/experimental/notoken/collective_ops/bcast.py:    translation_rule_cuda,
mpi4jax/experimental/notoken/collective_ops/bcast.py:mpi_bcast_xla_encode_cuda = translation_rule_cuda(mpi_bcast_xla_encode_device)
mpi4jax/experimental/notoken/collective_ops/bcast.py:register_lowering(mpi_bcast_p, mpi_bcast_xla_encode_cuda, platform="cuda")
mpi4jax/experimental/notoken/collective_ops/barrier.py:    translation_rule_cuda,
mpi4jax/experimental/notoken/collective_ops/barrier.py:mpi_barrier_xla_encode_cuda = translation_rule_cuda(mpi_barrier_xla_encode_device)
mpi4jax/experimental/notoken/collective_ops/barrier.py:register_lowering(mpi_barrier_p, mpi_barrier_xla_encode_cuda, platform="cuda")
mpi4jax/experimental/notoken/collective_ops/allreduce.py:    translation_rule_cuda,
mpi4jax/experimental/notoken/collective_ops/allreduce.py:mpi_allreduce_xla_encode_cuda = translation_rule_cuda(mpi_allreduce_xla_encode_device)
mpi4jax/experimental/notoken/collective_ops/allreduce.py:register_lowering(mpi_allreduce_p, mpi_allreduce_xla_encode_cuda, platform="cuda")
mpi4jax/experimental/notoken/collective_ops/scatter.py:    translation_rule_cuda,
mpi4jax/experimental/notoken/collective_ops/scatter.py:mpi_scatter_xla_encode_cuda = translation_rule_cuda(mpi_scatter_xla_encode_device)
mpi4jax/experimental/notoken/collective_ops/scatter.py:register_lowering(mpi_scatter_p, mpi_scatter_xla_encode_cuda, platform="cuda")
mpi4jax/experimental/notoken/collective_ops/sendrecv.py:    translation_rule_cuda,
mpi4jax/experimental/notoken/collective_ops/sendrecv.py:@translation_rule_cuda
mpi4jax/experimental/notoken/collective_ops/sendrecv.py:def mpi_sendrecv_xla_encode_cuda(
mpi4jax/experimental/notoken/collective_ops/sendrecv.py:register_lowering(mpi_sendrecv_p, mpi_sendrecv_xla_encode_cuda, platform="cuda")
mpi4jax/experimental/notoken/collective_ops/scan.py:    translation_rule_cuda,
mpi4jax/experimental/notoken/collective_ops/scan.py:mpi_scan_xla_encode_cuda = translation_rule_cuda(mpi_scan_xla_encode_device)
mpi4jax/experimental/notoken/collective_ops/scan.py:register_lowering(mpi_scan_p, mpi_scan_xla_encode_cuda, platform="cuda")
mpi4jax/experimental/notoken/collective_ops/alltoall.py:    translation_rule_cuda,
mpi4jax/experimental/notoken/collective_ops/alltoall.py:mpi_alltoall_xla_encode_cuda = translation_rule_cuda(mpi_alltoall_xla_encode_device)
mpi4jax/experimental/notoken/collective_ops/alltoall.py:register_lowering(mpi_alltoall_p, mpi_alltoall_xla_encode_cuda, platform="cuda")
mpi4jax/experimental/notoken/collective_ops/allgather.py:    translation_rule_cuda,
mpi4jax/experimental/notoken/collective_ops/allgather.py:mpi_allgather_xla_encode_cuda = translation_rule_cuda(mpi_allgather_xla_encode_device)
mpi4jax/experimental/notoken/collective_ops/allgather.py:register_lowering(mpi_allgather_p, mpi_allgather_xla_encode_cuda, platform="cuda")
mpi4jax/experimental/notoken/collective_ops/recv.py:    translation_rule_cuda,
mpi4jax/experimental/notoken/collective_ops/recv.py:mpi_recv_xla_encode_cuda = translation_rule_cuda(mpi_recv_xla_encode_device)
mpi4jax/experimental/notoken/collective_ops/recv.py:register_lowering(mpi_recv_p, mpi_recv_xla_encode_cuda, platform="cuda")
mpi4jax/__init__.py:    has_cuda_support,
mpi4jax/__init__.py:    "has_cuda_support",

```
