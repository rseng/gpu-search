# https://github.com/PyFR/PyFR

```console
setup.py:    'pyfr.backends.cuda',
setup.py:    'pyfr.backends.cuda.kernels',
setup.py:    'pyfr.backends.opencl',
setup.py:    'pyfr.backends.opencl.kernels',
setup.py:    'pyfr.backends.cuda.kernels': ['*.mako'],
setup.py:    'pyfr.backends.opencl.kernels': ['*.mako'],
pyfr/backends/hip/generator.py:from pyfr.backends.base.generator import BaseGPUKernelGenerator
pyfr/backends/hip/generator.py:class HIPKernelGenerator(BaseGPUKernelGenerator):
pyfr/backends/hip/compiler.py:        flags = [f'--gpu-architecture={arch}', '-munsafe-fp-atomics']
pyfr/backends/opencl/gimmik.py:from gimmik import OpenCLMatMul
pyfr/backends/opencl/gimmik.py:from pyfr.backends.opencl.provider import OpenCLKernel, OpenCLKernelProvider
pyfr/backends/opencl/gimmik.py:class OpenCLGiMMiKKernels(OpenCLKernelProvider):
pyfr/backends/opencl/gimmik.py:        self.max_nnz = backend.cfg.getint('backend-opencl', 'gimmik-max-nnz',
pyfr/backends/opencl/gimmik.py:        self.nkerns = backend.cfg.getint('backend-opencl', 'gimmik-nkerns', 8)
pyfr/backends/opencl/gimmik.py:        self.nbench = backend.cfg.getint('backend-opencl', 'gimmik-nbench', 5)
pyfr/backends/opencl/gimmik.py:            mm = OpenCLMatMul(alpha*arr, beta=beta, aligne=aligne, n=b.ncol,
pyfr/backends/opencl/gimmik.py:        class MulKernel(OpenCLKernel):
pyfr/backends/opencl/provider.py:from pyfr.backends.opencl.generator import OpenCLKernelGenerator
pyfr/backends/opencl/provider.py:class OpenCLKernel(Kernel):
pyfr/backends/opencl/provider.py:class OpenCLOrderedMetaKernel(BaseOrderedMetaKernel):
pyfr/backends/opencl/provider.py:class OpenCLUnorderedMetaKernel(BaseUnorderedMetaKernel):
pyfr/backends/opencl/provider.py:class OpenCLKernelProvider(BaseKernelProvider):
pyfr/backends/opencl/provider.py:class OpenCLPointwiseKernelProvider(OpenCLKernelProvider,
pyfr/backends/opencl/provider.py:        class KernelGenerator(OpenCLKernelGenerator):
pyfr/backends/opencl/provider.py:        class PointwiseKernel(OpenCLKernel):
pyfr/backends/opencl/types.py:class _OpenCLMatrixCommon:
pyfr/backends/opencl/types.py:class OpenCLMatrixBase(_OpenCLMatrixCommon, base.MatrixBase):
pyfr/backends/opencl/types.py:class OpenCLMatrixSlice(_OpenCLMatrixCommon, base.MatrixSlice):
pyfr/backends/opencl/types.py:class OpenCLMatrix(OpenCLMatrixBase, base.Matrix): pass
pyfr/backends/opencl/types.py:class OpenCLConstMatrix(OpenCLMatrixBase, base.ConstMatrix): pass
pyfr/backends/opencl/types.py:class OpenCLView(base.View): pass
pyfr/backends/opencl/types.py:class OpenCLXchgView(base.XchgView): pass
pyfr/backends/opencl/types.py:class OpenCLXchgMatrix(OpenCLMatrix, base.XchgMatrix):
pyfr/backends/opencl/types.py:class OpenCLGraph(base.Graph):
pyfr/backends/opencl/blasext.py:from pyfr.backends.opencl.provider import OpenCLKernel, OpenCLKernelProvider
pyfr/backends/opencl/blasext.py:class OpenCLBlasExtKernels(OpenCLKernelProvider):
pyfr/backends/opencl/blasext.py:        class AxnpbyKernel(OpenCLKernel):
pyfr/backends/opencl/blasext.py:        class CopyKernel(OpenCLKernel):
pyfr/backends/opencl/blasext.py:        class ReductionKernel(OpenCLKernel):
pyfr/backends/opencl/generator.py:from pyfr.backends.base.generator import BaseGPUKernelGenerator
pyfr/backends/opencl/generator.py:class OpenCLKernelGenerator(BaseGPUKernelGenerator):
pyfr/backends/opencl/packing.py:from pyfr.backends.opencl.provider import OpenCLKernel, OpenCLKernelProvider
pyfr/backends/opencl/packing.py:class OpenCLPackingKernels(OpenCLKernelProvider):
pyfr/backends/opencl/packing.py:        class PackXchgViewKernel(OpenCLKernel):
pyfr/backends/opencl/packing.py:        class UnpackXchgMatrixKernel(OpenCLKernel):
pyfr/backends/opencl/kernels/base.mako:#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
pyfr/backends/opencl/driver.py:# Possible OpenCL exception types
pyfr/backends/opencl/driver.py:class OpenCLError(Exception): pass
pyfr/backends/opencl/driver.py:class OpenCLDeviceNotFound(OpenCLError): pass
pyfr/backends/opencl/driver.py:class OpenCLDeviceNotAvailable(OpenCLError): pass
pyfr/backends/opencl/driver.py:class OpenCLAllocationFailure(OpenCLError): pass
pyfr/backends/opencl/driver.py:class OpenCLOutOfResources(OpenCLError): pass
pyfr/backends/opencl/driver.py:class OpenCLBuildProgramFailure(OpenCLError): pass
pyfr/backends/opencl/driver.py:class OpenCLMisalignedSubBufferOffset(OpenCLError): pass
pyfr/backends/opencl/driver.py:class OpenCLDevicePartitioningFailed(OpenCLError): pass
pyfr/backends/opencl/driver.py:class OpenCLInvalidValue(OpenCLError): pass
pyfr/backends/opencl/driver.py:class OpenCLInvalidKernelName(OpenCLError): pass
pyfr/backends/opencl/driver.py:class OpenCLInvalidKernelArgs(OpenCLError): pass
pyfr/backends/opencl/driver.py:class OpenCLInvalidWorkGroupSize(OpenCLError): pass
pyfr/backends/opencl/driver.py:class OpenCLInvalidWorkItemSize(OpenCLError): pass
pyfr/backends/opencl/driver.py:class OpenCLInvalidGlobalWorkSize(OpenCLError): pass
pyfr/backends/opencl/driver.py:class OpenCLWrappers(LibWrapper):
pyfr/backends/opencl/driver.py:    _libname = 'OpenCL'
pyfr/backends/opencl/driver.py:        -1: OpenCLDeviceNotFound,
pyfr/backends/opencl/driver.py:        -2: OpenCLDeviceNotAvailable,
pyfr/backends/opencl/driver.py:        -4: OpenCLAllocationFailure,
pyfr/backends/opencl/driver.py:        -5: OpenCLOutOfResources,
pyfr/backends/opencl/driver.py:        -11: OpenCLBuildProgramFailure,
pyfr/backends/opencl/driver.py:        -13: OpenCLMisalignedSubBufferOffset,
pyfr/backends/opencl/driver.py:        -18: OpenCLDevicePartitioningFailed,
pyfr/backends/opencl/driver.py:        -30: OpenCLInvalidValue,
pyfr/backends/opencl/driver.py:        -46: OpenCLInvalidKernelName,
pyfr/backends/opencl/driver.py:        -52: OpenCLInvalidKernelArgs,
pyfr/backends/opencl/driver.py:        -54: OpenCLInvalidWorkGroupSize,
pyfr/backends/opencl/driver.py:        -55: OpenCLInvalidWorkItemSize,
pyfr/backends/opencl/driver.py:        -63: OpenCLInvalidGlobalWorkSize,
pyfr/backends/opencl/driver.py:        '*': OpenCLError
pyfr/backends/opencl/driver.py:    DEVICE_TYPE_GPU = 0x4
pyfr/backends/opencl/driver.py:class _OpenCLBase:
pyfr/backends/opencl/driver.py:class _OpenCLWaitFor:
pyfr/backends/opencl/driver.py:class OpenCLPlatform(_OpenCLBase):
pyfr/backends/opencl/driver.py:        except OpenCLDeviceNotFound:
pyfr/backends/opencl/driver.py:        return [OpenCLDevice(self.lib, d) for d in devices]
pyfr/backends/opencl/driver.py:class OpenCLDevice(_OpenCLBase):
pyfr/backends/opencl/driver.py:        except OpenCLDevicePartitioningFailed:
pyfr/backends/opencl/driver.py:        return [OpenCLDevice(lib, d) for d in subdevices]
pyfr/backends/opencl/driver.py:class OpenCLBuffer(_OpenCLBase):
pyfr/backends/opencl/driver.py:        return OpenCLSubBuffer(self.lib, self, off, nbytes)
pyfr/backends/opencl/driver.py:class OpenCLSubBuffer(_OpenCLBase):
pyfr/backends/opencl/driver.py:class OpenCLHostAlloc(_OpenCLBase):
pyfr/backends/opencl/driver.py:class OpenCLEvent(_OpenCLBase):
pyfr/backends/opencl/driver.py:class OpenCLQueue(_OpenCLWaitFor, _OpenCLBase):
pyfr/backends/opencl/driver.py:        return OpenCLEvent(self.lib, evt_ptr)
pyfr/backends/opencl/driver.py:class OpenCLProgram(_OpenCLBase):
pyfr/backends/opencl/driver.py:        except OpenCLBuildProgramFailure:
pyfr/backends/opencl/driver.py:            raise OpenCLBuildProgramFailure(buf.value.decode()) from None
pyfr/backends/opencl/driver.py:        return OpenCLKernel(self.lib, ptr, argtypes)
pyfr/backends/opencl/driver.py:class OpenCLKernel(_OpenCLWaitFor, _OpenCLBase):
pyfr/backends/opencl/driver.py:        return OpenCLKernel(self.lib, ptr, self.argtypes)
pyfr/backends/opencl/driver.py:            return OpenCLEvent(self.lib, evt_ptr)
pyfr/backends/opencl/driver.py:class OpenCL(_OpenCLWaitFor):
pyfr/backends/opencl/driver.py:        self.lib = OpenCLWrappers()
pyfr/backends/opencl/driver.py:        return [OpenCLPlatform(self.lib, p) for p in platforms]
pyfr/backends/opencl/driver.py:        return OpenCLBuffer(self.lib, self.ctx, nbytes)
pyfr/backends/opencl/driver.py:        alloc = OpenCLHostAlloc(self.lib, self.ctx, self.qdflt, nbytes)
pyfr/backends/opencl/driver.py:            return OpenCLEvent(self.lib, evt_ptr)
pyfr/backends/opencl/driver.py:        return OpenCLProgram(self.lib, self.ctx, self.dev, src, flags or [])
pyfr/backends/opencl/driver.py:        return OpenCLEvent(self.lib, evt)
pyfr/backends/opencl/driver.py:        return OpenCLQueue(self.lib, self.ctx, self.dev, out_of_order,
pyfr/backends/opencl/base.py:class OpenCLBackend(BaseBackend):
pyfr/backends/opencl/base.py:    name = 'opencl'
pyfr/backends/opencl/base.py:        from pyfr.backends.opencl.driver import OpenCL
pyfr/backends/opencl/base.py:        # Load and wrap OpenCL
pyfr/backends/opencl/base.py:        self.cl = OpenCL()
pyfr/backends/opencl/base.py:        platid = cfg.get('backend-opencl', 'platform-id', '0').lower()
pyfr/backends/opencl/base.py:        devid = cfg.get('backend-opencl', 'device-id', 'local-rank').lower()
pyfr/backends/opencl/base.py:        devtype = cfg.get('backend-opencl', 'device-type', 'all').upper()
pyfr/backends/opencl/base.py:        # Determine the OpenCL platform to use
pyfr/backends/opencl/base.py:            raise ValueError('No suitable OpenCL platform found')
pyfr/backends/opencl/base.py:        # Determine the OpenCL device to use
pyfr/backends/opencl/base.py:            raise ValueError('No suitable OpenCL device found')
pyfr/backends/opencl/base.py:        from pyfr.backends.opencl import (blasext, clblast, gimmik, packing,
pyfr/backends/opencl/base.py:        self.const_matrix_cls = types.OpenCLConstMatrix
pyfr/backends/opencl/base.py:        self.graph_cls = types.OpenCLGraph
pyfr/backends/opencl/base.py:        self.matrix_cls = types.OpenCLMatrix
pyfr/backends/opencl/base.py:        self.matrix_slice_cls = types.OpenCLMatrixSlice
pyfr/backends/opencl/base.py:        self.view_cls = types.OpenCLView
pyfr/backends/opencl/base.py:        self.xchg_matrix_cls = types.OpenCLXchgMatrix
pyfr/backends/opencl/base.py:        self.xchg_view_cls = types.OpenCLXchgView
pyfr/backends/opencl/base.py:        self.ordered_meta_kernel_cls = provider.OpenCLOrderedMetaKernel
pyfr/backends/opencl/base.py:        self.unordered_meta_kernel_cls = provider.OpenCLUnorderedMetaKernel
pyfr/backends/opencl/base.py:        kprovs = [provider.OpenCLPointwiseKernelProvider,
pyfr/backends/opencl/base.py:                  blasext.OpenCLBlasExtKernels,
pyfr/backends/opencl/base.py:                  packing.OpenCLPackingKernels,
pyfr/backends/opencl/base.py:                  gimmik.OpenCLGiMMiKKernels]
pyfr/backends/opencl/base.py:            self._providers.append(clblast.OpenCLCLBlastKernels(self))
pyfr/backends/opencl/base.py:            self._providers.append(tinytc.OpenCLTinyTCKernels(self))
pyfr/backends/opencl/__init__.py:from pyfr.backends.opencl.base import OpenCLBackend
pyfr/backends/opencl/clblast.py:from pyfr.backends.opencl.provider import OpenCLKernel, OpenCLKernelProvider
pyfr/backends/opencl/clblast.py:class OpenCLCLBlastKernels(OpenCLKernelProvider):
pyfr/backends/opencl/clblast.py:        class MulKernel(OpenCLKernel):
pyfr/backends/opencl/tinytc.py:from pyfr.backends.opencl.provider import OpenCLKernel, OpenCLKernelProvider
pyfr/backends/opencl/tinytc.py:class OpenCLTinyTCKernels(OpenCLKernelProvider):
pyfr/backends/opencl/tinytc.py:        class MulKernel(OpenCLKernel):
pyfr/backends/metal/provider.py:        return (cbuf_bench.GPUEndTime() - cbuf_bench.GPUStartTime()) / nbench
pyfr/backends/metal/generator.py:from pyfr.backends.base.generator import BaseGPUKernelGenerator
pyfr/backends/metal/generator.py:class MetalKernelGenerator(BaseGPUKernelGenerator):
pyfr/backends/base/generator.py:class BaseGPUKernelGenerator(BaseKernelGenerator):
pyfr/backends/cuda/gimmik.py:from gimmik import CUDAMatMul
pyfr/backends/cuda/gimmik.py:from pyfr.backends.cuda.provider import CUDAKernel, CUDAKernelProvider
pyfr/backends/cuda/gimmik.py:class CUDAGiMMiKKernels(CUDAKernelProvider):
pyfr/backends/cuda/gimmik.py:        self.nkerns = backend.cfg.getint('backend-cuda', 'gimmik-nkerns', 8)
pyfr/backends/cuda/gimmik.py:        self.nbench = backend.cfg.getint('backend-cuda', 'gimmik-nbench', 5)
pyfr/backends/cuda/gimmik.py:            mm = CUDAMatMul(alpha*arr, beta=beta, aligne=aligne, n=b.ncol,
pyfr/backends/cuda/gimmik.py:                compute_capability=self.backend.cuda.compute_capability()
pyfr/backends/cuda/gimmik.py:        class MulKernel(CUDAKernel):
pyfr/backends/cuda/provider.py:from pyfr.backends.cuda.generator import CUDAKernelGenerator
pyfr/backends/cuda/provider.py:from pyfr.backends.cuda.compiler import SourceModule
pyfr/backends/cuda/provider.py:class CUDAKernel(Kernel):
pyfr/backends/cuda/provider.py:class CUDAOrderedMetaKernel(BaseOrderedMetaKernel):
pyfr/backends/cuda/provider.py:class CUDAUnorderedMetaKernel(BaseUnorderedMetaKernel):
pyfr/backends/cuda/provider.py:class CUDAKernelProvider(BaseKernelProvider):
pyfr/backends/cuda/provider.py:        stream = self.backend.cuda.create_stream()
pyfr/backends/cuda/provider.py:        start_evt = self.backend.cuda.create_event(timing=True)
pyfr/backends/cuda/provider.py:        stop_evt = self.backend.cuda.create_event(timing=True)
pyfr/backends/cuda/provider.py:class CUDAPointwiseKernelProvider(CUDAKernelProvider,
pyfr/backends/cuda/provider.py:        class KernelGenerator(CUDAKernelGenerator):
pyfr/backends/cuda/provider.py:        class PointwiseKernel(CUDAKernel):
pyfr/backends/cuda/types.py:class _CUDAMatrixCommon:
pyfr/backends/cuda/types.py:class CUDAMatrixBase(_CUDAMatrixCommon, base.MatrixBase):
pyfr/backends/cuda/types.py:        self.backend.cuda.memcpy(buf, self.data, self.nbytes)
pyfr/backends/cuda/types.py:        self.backend.cuda.memcpy(self.data, buf, self.nbytes)
pyfr/backends/cuda/types.py:class CUDAMatrixSlice(_CUDAMatrixCommon, base.MatrixSlice):
pyfr/backends/cuda/types.py:class CUDAMatrix(CUDAMatrixBase, base.Matrix): pass
pyfr/backends/cuda/types.py:class CUDAConstMatrix(CUDAMatrixBase, base.ConstMatrix): pass
pyfr/backends/cuda/types.py:class CUDAView(base.View): pass
pyfr/backends/cuda/types.py:class CUDAXchgView(base.XchgView): pass
pyfr/backends/cuda/types.py:class CUDAXchgMatrix(CUDAMatrix, base.XchgMatrix):
pyfr/backends/cuda/types.py:        # If MPI is CUDA-aware then simply annotate our device buffer
pyfr/backends/cuda/types.py:        if backend.mpitype == 'cuda-aware':
pyfr/backends/cuda/types.py:            self.hdata = backend.cuda.pagelocked_empty(shape, dtype)
pyfr/backends/cuda/types.py:class CUDAGraph(base.Graph):
pyfr/backends/cuda/types.py:        self.graph = backend.cuda.create_graph()
pyfr/backends/cuda/types.py:            event = self.backend.cuda.create_event()
pyfr/backends/cuda/blasext.py:from pyfr.backends.cuda.provider import (CUDAKernel, CUDAKernelProvider,
pyfr/backends/cuda/blasext.py:class CUDABlasExtKernels(CUDAKernelProvider):
pyfr/backends/cuda/blasext.py:        class AxnpbyKernel(CUDAKernel):
pyfr/backends/cuda/blasext.py:        cuda = self.backend.cuda
pyfr/backends/cuda/blasext.py:        class CopyKernel(CUDAKernel):
pyfr/backends/cuda/blasext.py:                cuda.memcpy(dst, src, dst.nbytes, stream)
pyfr/backends/cuda/blasext.py:        cuda = self.backend.cuda
pyfr/backends/cuda/blasext.py:        reduced_dev = cuda.mem_alloc(ncola*grid[0]*rs[0].itemsize)
pyfr/backends/cuda/blasext.py:        reduced_host = cuda.pagelocked_empty((ncola, grid[0]), fpdtype)
pyfr/backends/cuda/blasext.py:        class ReductionKernel(CUDAKernel):
pyfr/backends/cuda/blasext.py:                cuda.memcpy(reduced_host, reduced_dev, reduced_dev.nbytes,
pyfr/backends/cuda/cublaslt.py:from pyfr.backends.cuda.provider import CUDAKernel, CUDAKernelProvider
pyfr/backends/cuda/cublaslt.py:class CUDACUBLASLtKernels(CUDAKernelProvider):
pyfr/backends/cuda/cublaslt.py:        self.nkerns = backend.cfg.getint('backend-cuda', 'cublas-nkerns', 512)
pyfr/backends/cuda/cublaslt.py:        cuda = self.backend.cuda
pyfr/backends/cuda/cublaslt.py:            ws_ptr = cuda.mem_alloc(self.WORKSPACE_MAX_SIZE)
pyfr/backends/cuda/cublaslt.py:        ws_ptr = cuda.mem_alloc(desc.ws_size) if desc.ws_size else None
pyfr/backends/cuda/cublaslt.py:        class MulKernel(CUDAKernel):
pyfr/backends/cuda/cublaslt.py:                stream = cuda.create_stream()
pyfr/backends/cuda/generator.py:from pyfr.backends.base.generator import BaseGPUKernelGenerator
pyfr/backends/cuda/generator.py:class CUDAKernelGenerator(BaseGPUKernelGenerator):
pyfr/backends/cuda/packing.py:from pyfr.backends.cuda.provider import (CUDAKernel, CUDAKernelProvider,
pyfr/backends/cuda/packing.py:class CUDAPackingKernels(CUDAKernelProvider):
pyfr/backends/cuda/packing.py:        cuda = self.backend.cuda
pyfr/backends/cuda/packing.py:        # If MPI is CUDA aware then we just need to pack the buffer
pyfr/backends/cuda/packing.py:        if self.backend.mpitype == 'cuda-aware':
pyfr/backends/cuda/packing.py:            class PackXchgViewKernel(CUDAKernel):
pyfr/backends/cuda/packing.py:            class PackXchgViewKernel(CUDAKernel):
pyfr/backends/cuda/packing.py:                    cuda.memcpy(m.hdata, m.data, m.nbytes, stream)
pyfr/backends/cuda/packing.py:        cuda = self.backend.cuda
pyfr/backends/cuda/packing.py:        if self.backend.mpitype == 'cuda-aware':
pyfr/backends/cuda/packing.py:            class UnpackXchgMatrixKernel(CUDAKernel):
pyfr/backends/cuda/packing.py:                    cuda.memcpy(mv.data, mv.hdata, mv.nbytes, stream)
pyfr/backends/cuda/driver.py:# Possible CUDA exception types
pyfr/backends/cuda/driver.py:class CUDAError(Exception): pass
pyfr/backends/cuda/driver.py:class CUDAInvalidValue(CUDAError): pass
pyfr/backends/cuda/driver.py:class CUDAOutofMemory(CUDAError): pass
pyfr/backends/cuda/driver.py:class CUDANotInitalized(CUDAError): pass
pyfr/backends/cuda/driver.py:class CUDANoDevice(CUDAError): pass
pyfr/backends/cuda/driver.py:class CUDAInvalidDevice(CUDAError): pass
pyfr/backends/cuda/driver.py:class CUDAECCUncorrectable(CUDAError): pass
pyfr/backends/cuda/driver.py:class CUDAErrorInvalidPTX(CUDAError): pass
pyfr/backends/cuda/driver.py:class CUDAErrorUnsupportedPTXVersion(CUDAError): pass
pyfr/backends/cuda/driver.py:class CUDAOSError(CUDAError, OSError): pass
pyfr/backends/cuda/driver.py:class CUDAInvalidHandle(CUDAError): pass
pyfr/backends/cuda/driver.py:class CUDAIllegalAddress(CUDAError): pass
pyfr/backends/cuda/driver.py:class CUDALaunchOutOfResources(CUDAError): pass
pyfr/backends/cuda/driver.py:class CUDALaunchFailed(CUDAError): pass
pyfr/backends/cuda/driver.py:class CUDASystemDriverMismatch(CUDAError): pass
pyfr/backends/cuda/driver.py:class CUDAKernelNodeParams(Structure):
pyfr/backends/cuda/driver.py:class CUDAMemcpy3D(Structure):
pyfr/backends/cuda/driver.py:class CUDAWrappers(LibWrapper):
pyfr/backends/cuda/driver.py:    _libname = 'cuda'
pyfr/backends/cuda/driver.py:        1: CUDAInvalidValue,
pyfr/backends/cuda/driver.py:        2: CUDAOutofMemory,
pyfr/backends/cuda/driver.py:        3: CUDANotInitalized,
pyfr/backends/cuda/driver.py:        100: CUDANoDevice,
pyfr/backends/cuda/driver.py:        101: CUDAInvalidDevice,
pyfr/backends/cuda/driver.py:        214: CUDAECCUncorrectable,
pyfr/backends/cuda/driver.py:        218: CUDAErrorInvalidPTX,
pyfr/backends/cuda/driver.py:        222: CUDAErrorUnsupportedPTXVersion,
pyfr/backends/cuda/driver.py:        304: CUDAOSError,
pyfr/backends/cuda/driver.py:        400: CUDAInvalidHandle,
pyfr/backends/cuda/driver.py:        700: CUDAIllegalAddress,
pyfr/backends/cuda/driver.py:        701: CUDALaunchOutOfResources,
pyfr/backends/cuda/driver.py:        719: CUDALaunchFailed,
pyfr/backends/cuda/driver.py:        803: CUDASystemDriverMismatch,
pyfr/backends/cuda/driver.py:        '*': CUDAError
pyfr/backends/cuda/driver.py:         POINTER(c_void_p), c_size_t, POINTER(CUDAKernelNodeParams)),
pyfr/backends/cuda/driver.py:         POINTER(c_void_p), c_size_t, POINTER(CUDAMemcpy3D), c_void_p),
pyfr/backends/cuda/driver.py:         POINTER(CUDAKernelNodeParams)),
pyfr/backends/cuda/driver.py:            raise RuntimeError(f'CUDA version {major}.{minor} < 11.4')
pyfr/backends/cuda/driver.py:class _CUDABase:
pyfr/backends/cuda/driver.py:    def __init__(self, cuda, ptr):
pyfr/backends/cuda/driver.py:        self.cuda = cuda
pyfr/backends/cuda/driver.py:                getattr(self.cuda.lib, self._destroyfn)(self)
pyfr/backends/cuda/driver.py:class CUDADevAlloc(_CUDABase):
pyfr/backends/cuda/driver.py:    def __init__(self, cuda, nbytes):
pyfr/backends/cuda/driver.py:        cuda.lib.cuMemAlloc(ptr, nbytes)
pyfr/backends/cuda/driver.py:        super().__init__(cuda, ptr)
pyfr/backends/cuda/driver.py:class CUDAHostAlloc(_CUDABase):
pyfr/backends/cuda/driver.py:    def __init__(self, cuda, nbytes):
pyfr/backends/cuda/driver.py:        cuda.lib.cuMemAllocHost(ptr, nbytes)
pyfr/backends/cuda/driver.py:        super().__init__(cuda, ptr)
pyfr/backends/cuda/driver.py:class CUDAStream(_CUDABase):
pyfr/backends/cuda/driver.py:    def __init__(self, cuda):
pyfr/backends/cuda/driver.py:        cuda.lib.cuStreamCreate(ptr, 0)
pyfr/backends/cuda/driver.py:        super().__init__(cuda, ptr)
pyfr/backends/cuda/driver.py:        self.cuda.lib.cuStreamBeginCapture(self, 0)
pyfr/backends/cuda/driver.py:        self.cuda.lib.cuStreamEndCapture(self, graph)
pyfr/backends/cuda/driver.py:        return CUDAGraph(self.cuda, graph)
pyfr/backends/cuda/driver.py:        self.cuda.lib.cuStreamSynchronize(self)
pyfr/backends/cuda/driver.py:class CUDAEvent(_CUDABase):
pyfr/backends/cuda/driver.py:    def __init__(self, cuda, timing=False):
pyfr/backends/cuda/driver.py:            flags = cuda.lib.EVENT_DEFAULT
pyfr/backends/cuda/driver.py:            flags = cuda.lib.EVENT_DISABLE_TIMING
pyfr/backends/cuda/driver.py:        cuda.lib.cuEventCreate(ptr, flags)
pyfr/backends/cuda/driver.py:        super().__init__(cuda, ptr)
pyfr/backends/cuda/driver.py:        self.cuda.lib.cuEventRecord(self, stream)
pyfr/backends/cuda/driver.py:        self.cuda.lib.cuEventSynchronize(self)
pyfr/backends/cuda/driver.py:        self.cuda.lib.cuEventElapsedTime(dt, start, self)
pyfr/backends/cuda/driver.py:class CUDAModule(_CUDABase):
pyfr/backends/cuda/driver.py:    def __init__(self, cuda, cucode):
pyfr/backends/cuda/driver.py:        cuda.lib.cuModuleLoadDataEx(ptr, cucode, 0, None, None)
pyfr/backends/cuda/driver.py:        super().__init__(cuda, ptr)
pyfr/backends/cuda/driver.py:        return CUDAFunction(self.cuda, self, name, argspec)
pyfr/backends/cuda/driver.py:class CUDAFunction(_CUDABase):
pyfr/backends/cuda/driver.py:    def __init__(self, cuda, module, name, argtypes):
pyfr/backends/cuda/driver.py:        cuda.lib.cuModuleGetFunction(ptr, module, name.encode())
pyfr/backends/cuda/driver.py:        super().__init__(cuda, ptr)
pyfr/backends/cuda/driver.py:        attr = getattr(self.cuda.lib, f'FUNC_ATTR_{attr.upper()}')
pyfr/backends/cuda/driver.py:        self.cuda.lib.cuFuncGetAttribute(byref(v), attr, self)
pyfr/backends/cuda/driver.py:        attr = getattr(self.cuda.lib, f'FUNC_ATTR_{attr.upper()}')
pyfr/backends/cuda/driver.py:        self.cuda.lib.cuFuncSetAttribute(self, attr, val)
pyfr/backends/cuda/driver.py:        return CUDAKernelNodeParams(self, grid, block, sharedb)
pyfr/backends/cuda/driver.py:        self.cuda.lib.cuLaunchKernel(self, *params.grid, *params.block,
pyfr/backends/cuda/driver.py:class CUDAGraph(_CUDABase):
pyfr/backends/cuda/driver.py:    def __init__(self, cuda, ptr=None):
pyfr/backends/cuda/driver.py:            cuda.lib.cuGraphCreate(ptr, 0)
pyfr/backends/cuda/driver.py:        super().__init__(cuda, ptr)
pyfr/backends/cuda/driver.py:        self.cuda.lib.cuGraphAddEmptyNode(ptr, self, *self._make_deps(deps))
pyfr/backends/cuda/driver.py:        self.cuda.lib.cuGraphAddEventRecordNode(ptr, self,
pyfr/backends/cuda/driver.py:        self.cuda.lib.cuGraphAddKernelNode(ptr, self, *self._make_deps(deps),
pyfr/backends/cuda/driver.py:        params = CUDAMemcpy3D()
pyfr/backends/cuda/driver.py:        params.src_memory_type = self.cuda.lib.MEMORYTYPE_UNIFIED
pyfr/backends/cuda/driver.py:        params.dst_memory_type = self.cuda.lib.MEMORYTYPE_UNIFIED
pyfr/backends/cuda/driver.py:        self.cuda.lib.cuGraphAddMemcpyNode(ptr, self, *self._make_deps(deps),
pyfr/backends/cuda/driver.py:                                           params, self.cuda.ctx)
pyfr/backends/cuda/driver.py:        self.cuda.lib.cuGraphAddChildGraphNode(ptr, self,
pyfr/backends/cuda/driver.py:        return CUDAExecGraph(self.cuda, self)
pyfr/backends/cuda/driver.py:class CUDAExecGraph(_CUDABase):
pyfr/backends/cuda/driver.py:    def __init__(self, cuda, graph):
pyfr/backends/cuda/driver.py:        cuda.lib.cuGraphInstantiateWithFlags(ptr, graph, 0)
pyfr/backends/cuda/driver.py:        super().__init__(cuda, ptr)
pyfr/backends/cuda/driver.py:        self.cuda.lib.cuGraphExecKernelNodeSetParams(self, node, kparams)
pyfr/backends/cuda/driver.py:        self.cuda.lib.cuGraphLaunch(self, stream)
pyfr/backends/cuda/driver.py:class CUDA:
pyfr/backends/cuda/driver.py:        self.lib = CUDAWrappers()
pyfr/backends/cuda/driver.py:        return CUDADevAlloc(self, nbytes)
pyfr/backends/cuda/driver.py:        alloc = CUDAHostAlloc(self, nbytes)
pyfr/backends/cuda/driver.py:        return CUDAModule(self, cucode)
pyfr/backends/cuda/driver.py:        return CUDAStream(self)
pyfr/backends/cuda/driver.py:        return CUDAEvent(self, timing)
pyfr/backends/cuda/driver.py:        return CUDAGraph(self)
pyfr/backends/cuda/base.py:class CUDABackend(BaseBackend):
pyfr/backends/cuda/base.py:    name = 'cuda'
pyfr/backends/cuda/base.py:        from pyfr.backends.cuda.compiler import NVRTC
pyfr/backends/cuda/base.py:        from pyfr.backends.cuda.driver import CUDA, CUDAError
pyfr/backends/cuda/base.py:        # Load and wrap CUDA and NVRTC
pyfr/backends/cuda/base.py:        self.cuda = CUDA()
pyfr/backends/cuda/base.py:        # Get the desired CUDA device
pyfr/backends/cuda/base.py:        devid = cfg.get('backend-cuda', 'device-id', 'local-rank')
pyfr/backends/cuda/base.py:            for i in range(self.cuda.device_count()):
pyfr/backends/cuda/base.py:                    self.cuda.set_device(i)
pyfr/backends/cuda/base.py:                except CUDAError:
pyfr/backends/cuda/base.py:                raise RuntimeError('Unable to create a CUDA context')
pyfr/backends/cuda/base.py:            self.cuda.set_device(get_local_rank())
pyfr/backends/cuda/base.py:            for i in range(self.cuda.device_count()):
pyfr/backends/cuda/base.py:                if str(self.cuda.device_uuid(i)) == devid:
pyfr/backends/cuda/base.py:                    self.cuda.set_device(i)
pyfr/backends/cuda/base.py:                raise RuntimeError(f'Unable to find CUDA device {devid}')
pyfr/backends/cuda/base.py:            self.cuda.set_device(int(devid))
pyfr/backends/cuda/base.py:        self.mpitype = cfg.get('backend-cuda', 'mpi-type', 'standard')
pyfr/backends/cuda/base.py:        if self.mpitype not in {'standard', 'cuda-aware'}:
pyfr/backends/cuda/base.py:            raise ValueError('Invalid CUDA backend MPI type')
pyfr/backends/cuda/base.py:        from pyfr.backends.cuda import (blasext, cublaslt, gimmik, packing,
pyfr/backends/cuda/base.py:        self.const_matrix_cls = types.CUDAConstMatrix
pyfr/backends/cuda/base.py:        self.graph_cls = types.CUDAGraph
pyfr/backends/cuda/base.py:        self.matrix_cls = types.CUDAMatrix
pyfr/backends/cuda/base.py:        self.matrix_slice_cls = types.CUDAMatrixSlice
pyfr/backends/cuda/base.py:        self.view_cls = types.CUDAView
pyfr/backends/cuda/base.py:        self.xchg_matrix_cls = types.CUDAXchgMatrix
pyfr/backends/cuda/base.py:        self.xchg_view_cls = types.CUDAXchgView
pyfr/backends/cuda/base.py:        self.ordered_meta_kernel_cls = provider.CUDAOrderedMetaKernel
pyfr/backends/cuda/base.py:        self.unordered_meta_kernel_cls = provider.CUDAUnorderedMetaKernel
pyfr/backends/cuda/base.py:        kprovs = [provider.CUDAPointwiseKernelProvider,
pyfr/backends/cuda/base.py:                  blasext.CUDABlasExtKernels,
pyfr/backends/cuda/base.py:                  packing.CUDAPackingKernels,
pyfr/backends/cuda/base.py:                  gimmik.CUDAGiMMiKKernels,
pyfr/backends/cuda/base.py:                  cublaslt.CUDACUBLASLtKernels]
pyfr/backends/cuda/base.py:        self._stream = self.cuda.create_stream()
pyfr/backends/cuda/base.py:        # Submit the kernels to the CUDA stream
pyfr/backends/cuda/base.py:        data = self.cuda.mem_alloc(nbytes)
pyfr/backends/cuda/base.py:        self.cuda.memset(data, 0, nbytes)
pyfr/backends/cuda/__init__.py:from pyfr.backends.cuda.base import CUDABackend
pyfr/backends/cuda/compiler.py:        cmajor, cminor = backend.cuda.compute_capability()
pyfr/backends/cuda/compiler.py:            f'--gpu-architecture=sm_{cmajor}{cminor}',
pyfr/backends/cuda/compiler.py:        flags += shlex.split(backend.cfg.get('backend-cuda', 'cflags', ''))
pyfr/backends/cuda/compiler.py:        # Compile to CUDA code (either PTX or CUBIN depending on arch flag)
pyfr/backends/cuda/compiler.py:        self.mod = backend.cuda.load_module(cucode)
pyfr/backends/__init__.py:from pyfr.backends.cuda import CUDABackend
pyfr/backends/__init__.py:from pyfr.backends.opencl import OpenCLBackend
doc/src/examples.rst:        mpiexec -n 2 pyfr -p run -b cuda euler-vortex.pyfrm euler-vortex.ini
doc/src/examples.rst:        pyfr -p run -b cuda double-mach-reflection.pyfrm double-mach-reflection.ini
doc/src/examples.rst:        pyfr -p run -b cuda couette-flow.pyfrm couette-flow.ini
doc/src/examples.rst:        pyfr -p run -b cuda inc-cylinder.pyfrm inc-cylinder.ini
doc/src/examples.rst:        pyfr -p run -b cuda viscous-shock-tube.pyfrm viscous-shock-tube.ini
doc/src/examples.rst:        pyfr -p run -b cuda triangular-aerofoil.pyfrm triangular-aerofoil.ini
doc/src/examples.rst:        pyfr -p run -b cuda triangular-aerofoil.pyfrm triangular-aerofoil-ascent.ini
doc/src/examples.rst:        pyfr -p run -b cuda taylor-green.pyfrm taylor-green.ini
doc/src/examples.rst:        pyfr -p run -b cuda taylor-green.pyfrm taylor-green-ascent.ini
doc/src/backends/backend-cuda.rst:[backend-cuda]
doc/src/backends/backend-cuda.rst:Parameterises the CUDA backend with
doc/src/backends/backend-cuda.rst:     ``standard`` | ``cuda-aware``
doc/src/backends/backend-cuda.rst:3. ``cflags`` --- additional NVIDIA realtime compiler (``nvrtc``) flags:
doc/src/backends/backend-cuda.rst:    [backend-cuda]
doc/src/backends/backend-opencl.rst:[backend-opencl]
doc/src/backends/backend-opencl.rst:Parameterises the OpenCL backend with
doc/src/backends/backend-opencl.rst:    ``all`` | ``cpu`` | ``gpu`` | ``accelerator``
doc/src/backends/backend-opencl.rst:    [backend-opencl]
doc/src/backends/backend-opencl.rst:    device-type = gpu
doc/src/developer_guide.rst:    :header: *CUDABackend* **Click to show**
doc/src/developer_guide.rst:    .. autoclass:: pyfr.backends.cuda.base.CUDABackend
doc/src/developer_guide.rst:    :header: *OpenCLBackend* **Click to show**
doc/src/developer_guide.rst:    .. autoclass:: pyfr.backends.opencl.base.OpenCLBackend
doc/src/developer_guide.rst:.. inheritance-diagram:: pyfr.backends.cuda.base
doc/src/developer_guide.rst:                         pyfr.backends.opencl.base
doc/src/developer_guide.rst:    :header: *CUDAPointwiseKernelProvider* **Click to show**
doc/src/developer_guide.rst:    .. autoclass:: pyfr.backends.cuda.provider.CUDAPointwiseKernelProvider
doc/src/developer_guide.rst:    :header: *OpenCLPointwiseKernelProvider* **Click to show**
doc/src/developer_guide.rst:    .. autoclass:: pyfr.backends.opencl.provider.OpenCLPointwiseKernelProvider
doc/src/developer_guide.rst:                         pyfr.backends.cuda.provider
doc/src/developer_guide.rst:                         pyfr.backends.opencl.provider
doc/src/developer_guide.rst:    :header: *CUDAKernelGenerator* **Click to show**
doc/src/developer_guide.rst:    .. autoclass:: pyfr.backends.cuda.generator.CUDAKernelGenerator
doc/src/developer_guide.rst:    :header: *OpenCLKernelGenerator* **Click to show**
doc/src/developer_guide.rst:    .. autoclass:: pyfr.backends.opencl.generator.OpenCLKernelGenerator
doc/src/developer_guide.rst:.. inheritance-diagram:: pyfr.backends.cuda.generator.CUDAKernelGenerator
doc/src/developer_guide.rst:                         pyfr.backends.opencl.generator.OpenCLKernelGenerator
doc/src/performance_tuning.rst:.. _perf cuda backend:
doc/src/performance_tuning.rst:CUDA Backend
doc/src/performance_tuning.rst:CUDA-aware MPI
doc/src/performance_tuning.rst:PyFR is capable of taking advantage of CUDA-aware MPI.  This enables
doc/src/performance_tuning.rst:CUDA device pointers to be directly to passed MPI routines.  Under the
doc/src/performance_tuning.rst:mpi4py has been built against an MPI distribution which is CUDA-aware
doc/src/performance_tuning.rst:        [backend-cuda]
doc/src/performance_tuning.rst:        mpi-type = cuda-aware
doc/src/user_guide.rst:   backends/backend-cuda.rst
doc/src/user_guide.rst:   backends/backend-opencl.rst
doc/src/installation.rst:.. _install cuda backend:
doc/src/installation.rst:CUDA Backend
doc/src/installation.rst:The CUDA backend targets NVIDIA GPUs with a compute capability of 3.0
doc/src/installation.rst:1. `CUDA <https://developer.nvidia.com/cuda-downloads>`_ >= 11.4
doc/src/installation.rst:The HIP backend targets AMD GPUs which are supported by the ROCm stack.
doc/src/installation.rst:1. `ROCm <https://docs.amd.com/>`_ >= 6.0.0
doc/src/installation.rst:2. `rocBLAS <https://github.com/ROCmSoftwarePlatform/rocBLAS>`_ >=
doc/src/installation.rst:The Metal backend targets Apple silicon GPUs. The backend requires:
doc/src/installation.rst:OpenCL Backend
doc/src/installation.rst:The OpenCL backend targets a range of accelerators including GPUs from
doc/src/installation.rst:AMD, Intel, and NVIDIA. The backend requires:
doc/src/installation.rst:1. OpenCL >= 2.1
doc/src/installation.rst:Note that when running on NVIDIA GPUs the OpenCL backend may terminate
doc/src/installation.rst:due to a long-standing bug in how the NVIDIA OpenCL implementation

```
