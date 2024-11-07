# https://github.com/numpy/numpy

```console
doc/source/user/basics.dispatch.rst:a GPU.
doc/source/user/basics.interoperability.rst:GPU arrays (CuPy_), Sparse arrays (`scipy.sparse`, `PyData/Sparse <Sparse_>`_)
doc/source/user/basics.interoperability.rst:   stream, e.g. in the case of multiple GPUs) and to access the data.
doc/source/user/basics.interoperability.rst:devices other than the CPU (e.g. Vulkan or GPU). Since NumPy only supports CPU,
doc/source/user/basics.interoperability.rst:like PyTorch_ and CuPy_, may exchange data on GPU using this protocol.
doc/source/user/basics.interoperability.rst:another function (for example, a GPU or parallel implementation) in a way that
doc/source/user/basics.interoperability.rst:learning using GPUs and CPUs. PyTorch arrays are commonly called *tensors*.
doc/source/user/basics.interoperability.rst:Tensors are similar to NumPy's ndarrays, except that tensors can run on GPUs or
doc/source/user/basics.interoperability.rst:CuPy is a NumPy/SciPy-compatible array library for GPU-accelerated computing
doc/source/user/basics.interoperability.rst: >>> x_gpu = cp.array([1, 2, 3, 4])
doc/source/user/basics.interoperability.rst:the matching CuPy CUDA/ROCm implementation of the ufunc):
doc/source/user/basics.interoperability.rst: >>> np.mean(np.exp(x_gpu))
doc/source/user/basics.interoperability.rst: >>> a_gpu = cp.asarray(a)
doc/source/user/basics.interoperability.rst: >>> qr_gpu = np.linalg.qr(a_gpu)
doc/source/user/basics.interoperability.rst:  Note that GPU tensors can't be converted to NumPy arrays since NumPy doesn't
doc/source/user/basics.interoperability.rst:  support GPU devices:
doc/source/user/basics.interoperability.rst:   >>> x_torch = torch.arange(5, device='cuda')
doc/source/user/basics.interoperability.rst:   >>> x_torch = torch.arange(5, device='cuda')
doc/source/reference/arrays.classes.rst:    computation and cupy arrays for GPU-based computation, subclassing is
doc/source/f2py/windows/index.rst:   natively, but has been superseded by the Nvidia HPC SDK, with no `native
doc/source/f2py/windows/index.rst:.. _Nvidia HPC SDK: https://www.pgroup.com/index.html
doc/source/f2py/windows/index.rst:.. _native Windows support: https://developer.nvidia.com/nvidia-hpc-sdk-downloads#collapseFour
doc/source/f2py/windows/pgi.rst:	As of 29-01-2022, `PGI compiler toolchains`_ have been superseded by the Nvidia
doc/source/f2py/windows/pgi.rst:.. _native Windows support: https://developer.nvidia.com/nvidia-hpc-sdk-downloads#collapseFour
doc/source/release/1.14.0-notes.rst:The PGI flang compiler is a Fortran front end for LLVM released by NVIDIA under
doc/source/release/1.20.0-notes.rst:The NVIDIA HPC SDK nvfortran compiler is now supported
doc/source/release/1.19.3-notes.rst:* `#17522 <https://github.com/numpy/numpy/pull/17522>`__: ENH: Support for the NVIDIA HPC SDK nvfortran compiler
doc/neps/nep-0030-duck-array-protocol.rst::Author: Peter Andreas Entschev <pentschev@nvidia.com>
doc/neps/nep-0056-array-api-main-namespace.rst:one of the top user requests from the 2020 NumPy user survey [4]_ (GPU support).
doc/neps/nep-0056-array-api-main-namespace.rst:- ``LinearDiscriminantAnalysis.fit`` showed ~28x gain with PyTorch on GPU vs.
doc/neps/nep-0056-array-api-main-namespace.rst:``unique_*`` ones, are easier to implement on GPU and for JIT compilers as a
doc/neps/nep-0056-array-api-main-namespace.rst:portable across multiple array libraries and their supported features like GPUs
doc/neps/nep-0056-array-api-main-namespace.rst:list. It will also work for CuPy & co, where it may create a new array on a GPU
doc/neps/nep-0022-ndarray-duck-typing-overview.rst:stored in GPU memory, arrays stored in alternative formats such as
doc/neps/nep-0018-array-function-protocol.rst:outside of NumPy itself for different architectures, such as for GPU
doc/neps/nep-0018-array-function-protocol.rst:divert execution to another function (for example a GPU or parallel
doc/neps/roadmap.rst:other NumPy-like projects.* This will enable GPU support via, e.g, CuPy, JAX or PyTorch,
doc/neps/nep-0031-uarray.rst:        # Code that has distributed GPU arrays here
doc/neps/nep-0031-uarray.rst:users may find it easier to port existing code to GPU or distributed computing.
doc/neps/nep-0037-array-module.rst:For example, a library that supports arrays on both CPUs and GPUs might decide
doc/neps/nep-0037-array-module.rst:            prefer_gpu = any(a.prefer_gpu for a in useful_arrays)
doc/neps/nep-0037-array-module.rst:            return ArrayModule(prefer_gpu)
doc/neps/nep-0037-array-module.rst:        def __init__(self, prefer_gpu):
doc/neps/nep-0037-array-module.rst:            self.prefer_gpu = prefer_gpu
doc/neps/nep-0037-array-module.rst:            return functools.partial(base_func, prefer_gpu=self.prefer_gpu)
doc/neps/nep-0037-array-module.rst:on the CPU or GPU could be solved by `making array creation lazy
doc/neps/nep-0049.rst:.. _`here`: https://mail.python.org/archives/list/numpy-discussion@python.org/thread/YPC5BGPUMKT2MLBP6O3FMPC35LFM2CCH/#YPC5BGPUMKT2MLBP6O3FMPC35LFM2CCH
doc/neps/nep-0035-array-creation-dispatch-with-array-function.rst::Author: Peter Andreas Entschev <pentschev@nvidia.com>
doc/neps/nep-0035-array-creation-dispatch-with-array-function.rst:computing, CuPy for GPGPU computing, xarray for N-D labeled arrays, etc. Underneath,
doc/neps/nep-0042-new-dtypes.rst:   to allow specialized implementations such as a GPU float64 subclassing a
doc/neps/nep-0047-array-api-standard.rst:1. DLPack is the only protocol with device support (e.g., GPUs using CUDA or
doc/neps/nep-0047-array-api-standard.rst:   ROCm drivers, or OpenCL devices). NumPy is CPU-only, but other array
doc/neps/nep-0047-array-api-standard.rst:multiple types of devices: CPU, GPU, TPU, and more exotic hardware.
doc/neps/nep-0053-c-abi-evolution.rst::Author: Sebastian Berg <sebastianb@nvidia.com>
doc/neps/scope.rst:  - Not specialized hardware such as GPUs
doc/neps/nep-0041-improved-dtype-support.rst:example with GPU backends (CuPy) storing additional methods related to the
doc/neps/nep-0041-improved-dtype-support.rst:GPU rather than as a mechanism to define new datatypes.
doc/changelog/1.19.3-changelog.rst:* `#17522 <https://github.com/numpy/numpy/pull/17522>`__: ENH: Support for the NVIDIA HPC SDK nvfortran compiler
doc/changelog/1.20.0-changelog.rst:* `#17344 <https://github.com/numpy/numpy/pull/17344>`__: ENH, BLD: Support for the NVIDIA HPC SDK nvfortran compiler
numpy/_core/tests/test_dlpack.py:            np.from_dlpack(x, device="gpu")
numpy/_core/tests/test_array_api_info.py:        info.default_dtypes(device="gpu")
numpy/_core/tests/test_array_api_info.py:        info.dtypes(device="gpu")
numpy/_core/src/common/dlpack/dlpack.h:  /*! \brief CUDA GPU device */
numpy/_core/src/common/dlpack/dlpack.h:  kDLCUDA = 2,
numpy/_core/src/common/dlpack/dlpack.h:   * \brief Pinned CUDA CPU memory by cudaMallocHost
numpy/_core/src/common/dlpack/dlpack.h:  kDLCUDAHost = 3,
numpy/_core/src/common/dlpack/dlpack.h:  /*! \brief OpenCL devices. */
numpy/_core/src/common/dlpack/dlpack.h:  kDLOpenCL = 4,
numpy/_core/src/common/dlpack/dlpack.h:  /*! \brief Metal for Apple GPU. */
numpy/_core/src/common/dlpack/dlpack.h:  /*! \brief ROCm GPUs for AMD GPUs */
numpy/_core/src/common/dlpack/dlpack.h:  kDLROCM = 10,
numpy/_core/src/common/dlpack/dlpack.h:   * \brief Pinned ROCm CPU memory allocated by hipMallocHost
numpy/_core/src/common/dlpack/dlpack.h:  kDLROCMHost = 11,
numpy/_core/src/common/dlpack/dlpack.h:   * \brief CUDA managed/unified memory allocated by cudaMallocManaged
numpy/_core/src/common/dlpack/dlpack.h:  kDLCUDAManaged = 13,
numpy/_core/src/common/dlpack/dlpack.h:  /*! \brief GPU support for next generation WebGPU standard. */
numpy/_core/src/common/dlpack/dlpack.h:  kDLWebGPU = 15,
numpy/_core/src/common/dlpack/dlpack.h:   * \brief The data pointer points to the allocated data. This will be CUDA
numpy/_core/src/common/dlpack/dlpack.h:   * device pointer or cl_mem handle in OpenCL. It may be opaque on some device
numpy/_core/src/common/dlpack/dlpack.h:   * types. This pointer is always aligned to 256 bytes as in CUDA. The
numpy/_core/src/common/dlpack/dlpack.h:   * on CPU/CUDA/ROCm, and always use `byte_offset=0`.  This must be fixed
numpy/_core/src/multiarray/nditer_templ.c.src: *                BUF, INDuBUF, IDPuBUF, INDuIDPuBUF, NEGPuBUF, INDuNEGPuBUF#
numpy/_core/src/multiarray/nditer_templ.c.src: *                BUF, INDuBUF, IDPuBUF, INDuIDPuBUF, NEGPuBUF, INDuNEGPuBUF#
numpy/_core/src/multiarray/dlpack.c:            device_type != kDLCUDAHost &&
numpy/_core/src/multiarray/dlpack.c:            device_type != kDLROCMHost &&
numpy/_core/src/multiarray/dlpack.c:            device_type != kDLCUDAManaged) {
numpy/distutils/fcompiler/nv.py:    """ NVIDIA High Performance Computing (HPC) SDK Fortran Compiler
numpy/distutils/fcompiler/nv.py:    https://developer.nvidia.com/hpc-sdk
numpy/distutils/fcompiler/nv.py:    Since august 2020 the NVIDIA HPC SDK includes the compilers formerly known as The Portland Group compilers,
numpy/distutils/fcompiler/nv.py:    description = 'NVIDIA HPC SDK'

```
