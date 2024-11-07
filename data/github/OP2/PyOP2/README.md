# https://github.com/OP2/PyOP2

```console
Makefile:	@echo "Available OpenCL contexts: $(OPENCL_CTXS)"
doc/sphinx/source/architecture.rst:   code, which may initiate host to device data transfer for the CUDA and
doc/sphinx/source/architecture.rst:   OpenCL backends.
doc/sphinx/source/ir.rst:between distinct cores. On the other end, if the backend is a GPU or an
doc/sphinx/source/ir.rst:There's a huge amount of parallelism available, for example, in a GPU, so
doc/sphinx/source/mpi.rst:wrapper code and allows launching separate kernels for GPU execution of each
doc/sphinx/source/backends.rst:* ``cuda``: offloads computation to a NVIDA GPU (requires :ref:`CUDA and pycuda
doc/sphinx/source/backends.rst:  <cuda-installation>`)
doc/sphinx/source/backends.rst:* ``opencl``: offloads computation to an OpenCL device, either a multi-core
doc/sphinx/source/backends.rst:  CPU or a GPU (requires :ref:`OpenCL and pyopencl <opencl-installation>`)
doc/sphinx/source/backends.rst:``cuda`` and ``opencl`` only support parallel loops on :class:`Dats
doc/sphinx/source/backends.rst:.. _cuda_backend:
doc/sphinx/source/backends.rst:CUDA backend
doc/sphinx/source/backends.rst:The CUDA backend makes extensive use of PyCUDA_ and its infrastructure for
doc/sphinx/source/backends.rst:just-in-time compilation of CUDA kernels and interfacing them to Python.
doc/sphinx/source/backends.rst:requires no CUDA-specific modifications and is automatically annotated with a
doc/sphinx/source/backends.rst:``__device__`` qualifier. PyCUDA_ automatically generates a host stub for the
doc/sphinx/source/backends.rst:such that a CUDA kernel can be launched straight from Python. The entire CUDA
doc/sphinx/source/backends.rst:The CUDA kernel ``__midpoint_stub`` is launched on the GPU for a specific
doc/sphinx/source/backends.rst:.. _opencl_backend:
doc/sphinx/source/backends.rst:OpenCL backend
doc/sphinx/source/backends.rst:The other device backend OpenCL is structurally very similar to the CUDA
doc/sphinx/source/backends.rst:backend. It uses PyOpenCL_ to interface to the OpenCL drivers and runtime.
doc/sphinx/source/backends.rst:to the CUDA case.
doc/sphinx/source/backends.rst:the kernel signature are automatically annotated with OpenCL storage
doc/sphinx/source/backends.rst:qualifiers. PyOpenCL_ provides Python wrappers for OpenCL runtime functions to
doc/sphinx/source/backends.rst:Parallel computations in OpenCL are executed by *work items* organised into
doc/sphinx/source/backends.rst:*work groups*. OpenCL requires the annotation of all pointer arguments with
doc/sphinx/source/backends.rst:automatically for the user kernel if the OpenCL backend is used. Local memory
doc/sphinx/source/backends.rst:therefore corresponds to CUDA's shared memory and private memory is called
doc/sphinx/source/backends.rst:local memory in CUDA. The work item id within the work group is accessed via
doc/sphinx/source/backends.rst:the OpenCL runtime call ``get_local_id(0)``, the work group id via
doc/sphinx/source/backends.rst:these differences in mind, the OpenCL kernel stub is structurally almost
doc/sphinx/source/backends.rst:identical to the corresponding CUDA version above.
doc/sphinx/source/backends.rst:computed as part of the execution plan. In CUDA this value is a launch
doc/sphinx/source/backends.rst:parameter to the kernel, whereas in OpenCL it needs to be hard coded as a
doc/sphinx/source/backends.rst:.. _PyCUDA: http://mathema.tician.de/software/pycuda/
doc/sphinx/source/backends.rst:.. _PyOpenCL: http://mathema.tician.de/software/pyopencl/
doc/sphinx/source/linear_algebra.rst:.. _gpu_assembly:
doc/sphinx/source/linear_algebra.rst:GPU matrix assembly
doc/sphinx/source/linear_algebra.rst:In a :func:`~pyop2.par_loop` assembling a :class:`~pyop2.Mat` on the GPU, the
doc/sphinx/source/linear_algebra.rst:above, the generated CUDA wrapper code is as follows, again omitting
doc/sphinx/source/linear_algebra.rst:initialisation and staging code described in :ref:`cuda_backend`.  The user
doc/sphinx/source/linear_algebra.rst:A separate CUDA kernel given below is launched afterwards to compress the data
doc/sphinx/source/linear_algebra.rst:when building the sparsity on the host and subsequently transferred to GPU
doc/sphinx/source/linear_algebra.rst:be allocated on the GPU.
doc/sphinx/source/linear_algebra.rst:.. _gpu_solve:
doc/sphinx/source/linear_algebra.rst:GPU linear algebra
doc/sphinx/source/linear_algebra.rst:Linear algebra on the GPU with the ``cuda`` backend uses the Cusp_ library,
doc/sphinx/source/linear_algebra.rst:  supported by the ``cuda`` backend.
doc/sphinx/source/kernels.rst:the element ``i`` the kernel is currently called for. In CUDA/OpenCL
doc/pyop2.tex:\item Since this leaves effectively no source-to-source transformation to perform (only inserting an essentially unmodified kernel into generated code) it should be possible to avoid the use of ROSE altogether. Should transformation need to be performed on OP2 kernels in future, this functionality may be added, either by integrating ROSE or using a simpler framework, since the operations performed in a kernel are limited to a fairly restricted subset of C/CUDA.
doc/pyop2.tex:    \item PyCUDA/PyOpenCL from Andreas Kl\"ockner for GPU/accelerator code

```
