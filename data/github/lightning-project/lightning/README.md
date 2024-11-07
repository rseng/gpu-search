# https://github.com/lightning-project/lightning

```console
lightning-cuda/Cargo.toml:name = "lightning-cuda"
lightning-cuda/Cargo.toml:cuda-driver-sys = "0.3.0"
lightning-cuda/src/error.rs:use cuda_driver_sys::*;
lightning-cuda/src/error.rs:/// Check the result of a CUDA driver API function.
lightning-cuda/src/error.rs:/// All functions in the CUDA driver API return a `CUresult` flag indicating whether the operation
lightning-cuda/src/error.rs:/// and `Err(Error)` otherwise, thus making it easy to wrap native CUDA functions.
lightning-cuda/src/error.rs:/// # use cuda_driver_sys::*;
lightning-cuda/src/error.rs:///     cuda_check(cuInit(0))
lightning-cuda/src/error.rs:pub fn cuda_check(code: CUresult) -> Result {
lightning-cuda/src/error.rs:/// Returns the output value of a CUDA driver API function.
lightning-cuda/src/error.rs:/// Many functions in the CUDA driver API follow the pattern where they take a pointer to an
lightning-cuda/src/error.rs:/// # use cuda_driver_sys::*;
lightning-cuda/src/error.rs:/// // Get the number of CUDA-capable devices in the systems. `result` will be either `Ok(n)`
lightning-cuda/src/error.rs:///     cuda_call(|count: *mut c_int| cuDeviceGetCount(count))
lightning-cuda/src/error.rs:pub unsafe fn cuda_call<T, F>(fun: F) -> Result<T>
lightning-cuda/src/error.rs:/// Error returned by the CUDA driver API.
lightning-cuda/src/error.rs:/// Nearly all functions in the CUDA driver API return an `CUresult` status code which is
lightning-cuda/src/error.rs:/// either `CUDA_SUCCESS` on success or one of the `CUDA_ERROR_*` values on an error. This object
lightning-cuda/src/error.rs:    /// Construct an error object. Returns `Ok(())` for `CUDA_SUCCESS` and `Err(Error)` for
lightning-cuda/src/error.rs:    /// `CUDA_ERROR_*`.
lightning-cuda/src/error.rs:        if code == CUresult::CUDA_SUCCESS {
lightning-cuda/src/error.rs:    /// Get the name of this error. Returns an error if this error code is not recognized by CUDA.
lightning-cuda/src/error.rs:            let result = cuda_call(|v| cuGetErrorName(self.raw(), v))?;
lightning-cuda/src/error.rs:    /// Get the description of this error. Returns an error if this error code is not recognized by CUDA.
lightning-cuda/src/error.rs:            let result = cuda_call(|v| cuGetErrorString(self.raw(), v))?;
lightning-cuda/src/error.rs:    /// The given status code cannot be `CUDA_SUCCESS`.
lightning-cuda/src/error.rs:        // CUDA_SUCCESS should be 0, so we can use a NonZeroU32 to store the remaining codes.
lightning-cuda/src/error.rs:        assert_eq!(CUresult::CUDA_SUCCESS as usize, 0);
lightning-cuda/src/error.rs:            f.debug_tuple("CudaError").field(&name).finish()
lightning-cuda/src/error.rs:            f.debug_tuple("CudaError").field(&self.0).finish()
lightning-cuda/src/error.rs:        unsafe { Error::from_raw(CUresult::CUDA_ERROR_UNKNOWN) }
lightning-cuda/src/mem.rs://! Memory allocation in CUDA.
lightning-cuda/src/mem.rs:use crate::{copy, copy_raw, cuda_call, cuda_check, CopyDestination, CopySource, Error, Result};
lightning-cuda/src/mem.rs:use cuda_driver_sys::*;
lightning-cuda/src/mem.rs:/// Raw memory pointer which can be accessed on a CUDA device.
lightning-cuda/src/mem.rs:/// Memory buffer allocated on a CUDA device.
lightning-cuda/src/mem.rs:        let raw = cuda_call(|p| cuMemAlloc_v2(p, size))?;
lightning-cuda/src/mem.rs:        f.debug_tuple("CudaPinnedMem")
lightning-cuda/src/mem.rs:        let hptr = cuda_call(|p| cuMemHostAlloc(p, size, flag))?;
lightning-cuda/src/mem.rs:        let dptr = cuda_call(|dptr| cuMemHostGetDevicePointer_v2(dptr, hptr, 0))?;
lightning-cuda/src/mem.rs:/// Fixed-size mutable slice of memory accessible to a CUDA device.
lightning-cuda/src/mem.rs:/// Fixed-size slice of memory accessible to a CUDA device.
lightning-cuda/src/mem.rs:                cuda_check(cuMemsetD8_v2(
lightning-cuda/src/mem.rs:                cuda_check(cuMemsetD16_v2(
lightning-cuda/src/mem.rs:                cuda_check(cuMemsetD32_v2(
lightning-cuda/src/mem.rs:                unsafe { cuda_check(cuMemsetD8_v2(dest_ptr, byte, size_bytes)) }
lightning-cuda/src/mem.rs:                Error::new(CUresult::CUDA_ERROR_INVALID_VALUE)
lightning-cuda/src/mem.rs:        cuda_check(cuMemsetD8_v2(ptr, value, size_bytes))
lightning-cuda/src/profiler.rs://! Management of CUDA profiler.
lightning-cuda/src/profiler.rs:use crate::{cuda_check, Result};
lightning-cuda/src/profiler.rs:use cuda_driver_sys::*;
lightning-cuda/src/profiler.rs:    unsafe { cuda_check(cuProfilerStart()) }
lightning-cuda/src/profiler.rs:    unsafe { cuda_check(cuProfilerStop()) }
lightning-cuda/src/device.rs://! CUDA device management.
lightning-cuda/src/device.rs:use crate::{cuda_call, cuda_check, Dim3, Error, Result};
lightning-cuda/src/device.rs:use cuda_driver_sys::*;
lightning-cuda/src/device.rs:/// CUDA-capable device.
lightning-cuda/src/device.rs:/// valid CUDA device.
lightning-cuda/src/device.rs:    /// Returns the number of CUDA-capable devices in the system.
lightning-cuda/src/device.rs:            let n = cuda_call(|ptr| cuDeviceGetCount(ptr))?;
lightning-cuda/src/device.rs:    /// Returns all CUDA-capable devices in the system.
lightning-cuda/src/device.rs:                .map_err(|_| Error::from_raw(CUresult::CUDA_ERROR_INVALID_VALUE))?;
lightning-cuda/src/device.rs:            let device = cuda_call(|ptr| cuDeviceGet(ptr, i))?;
lightning-cuda/src/device.rs:    /// Returns the `Device` for the CUDA context currently registered to the calling thread.
lightning-cuda/src/device.rs:        unsafe { cuda_call(|dev| cuCtxGetDevice(dev)).map(|d| Self::from_raw(d)) }
lightning-cuda/src/device.rs:            cuda_check(cuDeviceGetName(
lightning-cuda/src/device.rs:            let pi = cuda_call(|v| cuDeviceGetAttribute(v, attrib, self.0))?;
lightning-cuda/src/device.rs:        unsafe { cuda_call(|v| cuDeviceTotalMem_v2(v, self.0)) }
lightning-cuda/src/device.rs:        unsafe { cuda_call(|v| cuDeviceCanAccessPeer(v, self.0, peer.0)) }.map(|v| v != 0)
lightning-cuda/src/device.rs:        f.debug_tuple("CudaDevice")
lightning-cuda/src/device.rs:    GPU_OVERLAP = CU_DEVICE_ATTRIBUTE_GPU_OVERLAP as u32,
lightning-cuda/src/device.rs:    MULTI_GPU_BOARD = CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD as u32,
lightning-cuda/src/device.rs:    MULTI_GPU_BOARD_GROUP_ID = CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID as u32,
lightning-cuda/src/lib.rs:This crate offers transparent rustic wrappers around the CUDA driver API.
lightning-cuda/src/lib.rs:This crate offers Rust versions of functions and types from the CUDA driver API. It assumes the user
lightning-cuda/src/lib.rs:is aware of how the CUDA API looks like, but wants slightly more convenience when working with CUDA.
lightning-cuda/src/lib.rs:CUDA types and most functions simply forward the call directly to a CUDA function.
lightning-cuda/src/lib.rs:# Why choose cuba over other CUDA crates?
lightning-cuda/src/lib.rs:There exist many excellent CUDA crates already. The design of this crate is similar to that of
lightning-cuda/src/lib.rs:[RustaCUDA]. In fact, CUBA was born out of minor gripes with RustaCUDA's design, most notably related
lightning-cuda/src/lib.rs:written from scratch, in some sense, it can be seen as a "spriritual" fork of RustaCUDA.
lightning-cuda/src/lib.rs:[RustaCUDA]: https://crates.io/crates/rustacuda
lightning-cuda/src/lib.rs:Some are other crates worth considering when working with CUDA.
lightning-cuda/src/lib.rs:  * [cuda](https://crates.io/crates/cuda)
lightning-cuda/src/lib.rs:  * [cuda-driver-sys](https://crates.io/crates/cuda-driver-sys)
lightning-cuda/src/lib.rs:  * [cuda-runtime-sys](https://crates.io/crates/cuda-runtime-sys)
lightning-cuda/src/lib.rs:  * [cuda-sys](https://crates.io/crates/cuda-sys)
lightning-cuda/src/lib.rs:  * [cudart](https://crates.io/crates/cudart)
lightning-cuda/src/lib.rs:  * [RustaCUDA](https://crates.io/crates/rustacuda)
lightning-cuda/src/lib.rs:This crate allows safe functions where possible, but is not afraid to expose unsafe CUDA
lightning-cuda/src/lib.rs:Additionally, most types have a `raw` method which exposes the underlying CUDA type (for example, [`Stream::raw`]
lightning-cuda/src/lib.rs:a CUDA type (for example, [`Stream::from_raw`] which takes a `CUstream`). This enables FFI
lightning-cuda/src/lib.rs:integration without existing libraries which work with raw CUDA types.
lightning-cuda/src/lib.rs:Nearly all functions in the CUDA driver API return an status code indicating whether the operation
lightning-cuda/src/lib.rs:was successful. This crate mimics this behavior by returning a [`CudaResult`] which can be either
lightning-cuda/src/lib.rs:`Ok(_)` on success or a [`CudaError`] on failure.
lightning-cuda/src/lib.rs:[`CudaResult`]: error/type.Result.html
lightning-cuda/src/lib.rs:[`CudaError`]: error/struct.Error.html
lightning-cuda/src/lib.rs:exposes all items prefixed with "Cuda" to prevent name collisions. (for example, [`Stream`]
lightning-cuda/src/lib.rs:becomes `CudaStream`).
lightning-cuda/src/lib.rs:let device = CudaDevice::nth(0)?;
lightning-cuda/src/lib.rs:let context = cuda_create_context(device, CudaContextFlags::empty())?;
lightning-cuda/src/lib.rs:// Use CudaStream, CudaEvent, CudaModule, CudaDeviceMem, etc...
lightning-cuda/src/lib.rs:unsafe { cuda_destroy_context(context)?; }
lightning-cuda/src/lib.rs:use cuda_driver_sys::*;
lightning-cuda/src/lib.rs:/// Initialize the CUDA runtime.
lightning-cuda/src/lib.rs:/// Must be called before calling any other CUDA function.
lightning-cuda/src/lib.rs:    unsafe { cuda_check(cuInit(0))? }
lightning-cuda/src/lib.rs:                    enabled across all CUDA-capable devices.",
lightning-cuda/src/lib.rs:/// Returns the version of the CUDA driver API.
lightning-cuda/src/lib.rs:/// Returns a tuple `(major, minor)` indicating the version. For example, CUDA 9.2 would return `(9, 2)`.
lightning-cuda/src/lib.rs:    let value = unsafe { cuda_call(|v| cuDriverGetVersion(v)) }? as u32;
lightning-cuda/src/context.rs://! Management of CUDA contexts
lightning-cuda/src/context.rs:use crate::{cuda_call, cuda_check, Device, Error, Result};
lightning-cuda/src/context.rs:use cuda_driver_sys::*;
lightning-cuda/src/context.rs:/// Create a new CUDA context for the given device.
lightning-cuda/src/context.rs:        let handle = cuda_call(|c| cuCtxCreate_v2(c, flags.bits, device.raw()))?;
lightning-cuda/src/context.rs:        let popped = cuda_call(|c| cuCtxPopCurrent_v2(c))?;
lightning-cuda/src/context.rs:/// Access the primary CUDA context for the given device.
lightning-cuda/src/context.rs:/// Each CUDA device is associated with one _primary_ context. This function returns the
lightning-cuda/src/context.rs:/// multiple different libraries to safely interact without explicitly exchanging CUDA contexts.
lightning-cuda/src/context.rs:        let h = cuda_call(|c| cuDevicePrimaryCtxRetain(c, device.raw()))?;
lightning-cuda/src/context.rs:/// Release the primary CUDA context for the given device.
lightning-cuda/src/context.rs:/// drops to zero, the primary context is destroyed by the CUDA runtime.
lightning-cuda/src/context.rs:/// All CUDA resources (events, streams, memory, modules) associated with the given context will be
lightning-cuda/src/context.rs:    cuda_check(cuDevicePrimaryCtxRelease(device.raw()))
lightning-cuda/src/context.rs:/// Destroys the CUDA context associated with the given handle.
lightning-cuda/src/context.rs:/// All CUDA resources (events, streams, memory, modules) associated with the given context will be
lightning-cuda/src/context.rs:    cuda_check(cuCtxDestroy_v2(handle.raw()))
lightning-cuda/src/context.rs:/// Handle to a CUDA context.
lightning-cuda/src/context.rs:/// Represents a pointer to valid CUDA context. Use [`with`], [`try_with`], or [`activate`] to
lightning-cuda/src/context.rs:/// push this CUDA context on to the thread-local context stack managed by the CUDA runtime.
lightning-cuda/src/context.rs:// CUDA driver API is thread-safe, so sending a context across threads should be safe.
lightning-cuda/src/context.rs:        f.debug_tuple("CudaContextHandle").field(&self.0).finish()
lightning-cuda/src/context.rs:    /// Returns the top CUDA context in the thread-local stack of contexts.
lightning-cuda/src/context.rs:        unsafe { cuda_call(|c| cuCtxGetCurrent(c)).map(|c| Self::from_raw(c)) }
lightning-cuda/src/context.rs:    /// for CUDA operations. The context must be deactivated using [`pop`] when done. Usage of
lightning-cuda/src/context.rs:        unsafe { cuda_check(cuCtxPushCurrent_v2(self.raw())) }
lightning-cuda/src/context.rs:        unsafe { cuda_call(|c| cuCtxPopCurrent_v2(c)).map(|c| Self::from_raw(c)) }
lightning-cuda/src/context.rs:        self.try_with(|| unsafe { cuda_check(cuCtxSynchronize()) })
lightning-cuda/src/context.rs:            cuda_check(unsafe { cuMemGetInfo_v2(&mut free, &mut total) })?;
lightning-cuda/src/context.rs:        self.try_with(|| cuda_check(unsafe { cuCtxEnablePeerAccess(self.raw(), 0) }))
lightning-cuda/src/context.rs:        self.try_with(|| cuda_check(unsafe { cuCtxDisablePeerAccess(self.raw()) }))
lightning-cuda/src/context.rs:/// Pops the current CUDA context when dropped. See the above method for details.
lightning-cuda/src/context.rs:    /// Flags for configuring a CUDA context.
lightning-cuda/src/context.rs:    /// Represents the `CU_CTX_*` flags from the CUDA driver API.
lightning-cuda/src/stream.rs://! Management of CUDA streams.
lightning-cuda/src/stream.rs:use crate::{cuda_call, cuda_check, Error, Event, Result};
lightning-cuda/src/stream.rs:use cuda_driver_sys::cudaError_enum::{CUDA_ERROR_NOT_READY, CUDA_SUCCESS};
lightning-cuda/src/stream.rs:use cuda_driver_sys::*;
lightning-cuda/src/stream.rs:/// CUDA Compute stream.
lightning-cuda/src/stream.rs:/// CUDA uses the concept of streams to organize asynchronous operations. A stream is basically a
lightning-cuda/src/stream.rs:            let raw = cuda_call(|v| match priority {
lightning-cuda/src/stream.rs:    /// Returns the CUDA default stream.
lightning-cuda/src/stream.rs:    /// Returns true if this stream represents the CUDA default stream.
lightning-cuda/src/stream.rs:            cuda_check(cuCtxGetStreamPriorityRange(&mut least, &mut greatest))?;
lightning-cuda/src/stream.rs:        unsafe { cuda_call(|v| cuStreamGetPriority(self.0, v)) }
lightning-cuda/src/stream.rs:        unsafe { cuda_check(cuStreamWaitEvent(self.0, event.raw(), 0)) }
lightning-cuda/src/stream.rs:    /// The callback will be passed either `Ok(())` or an error code, either because CUDA failed
lightning-cuda/src/stream.rs:    /// panic from crossing from Rust into the CUDA runtime.
lightning-cuda/src/stream.rs:            if result != CUDA_SUCCESS {
lightning-cuda/src/stream.rs:            if result != CUDA_SUCCESS {
lightning-cuda/src/stream.rs:                CUDA_SUCCESS => Ok(true),
lightning-cuda/src/stream.rs:                CUDA_ERROR_NOT_READY => Ok(false),
lightning-cuda/src/stream.rs:        unsafe { cuda_check(cuStreamSynchronize(self.0)) }
lightning-cuda/src/stream.rs:    /// The given `CUstream` object should be a valid CUDA stream object.
lightning-cuda/src/stream.rs:        f.debug_tuple("CudaStream").field(&self.0).finish()
lightning-cuda/src/stream.rs:    /// Flags for configuring a CUDA stream.
lightning-cuda/src/stream.rs:    /// Represent the `CU_STREAM_*` flags from the CUDA API.
lightning-cuda/src/event.rs://! Management of CUDA events.
lightning-cuda/src/event.rs:use crate::{cuda_call, cuda_check, Error, Result, Stream};
lightning-cuda/src/event.rs:use cuda_driver_sys::*;
lightning-cuda/src/event.rs:/// CUDA Event.
lightning-cuda/src/event.rs:/// CUDA uses events to organize synchronization of [`Stream`]s. Events can record onto a compute
lightning-cuda/src/event.rs:// Events are thread-safe since all calls are passed ot the CUDA driver API which is thread-safe.
lightning-cuda/src/event.rs:        unsafe { cuda_call(|event| cuEventCreate(event, flags.bits)).map(Self) }
lightning-cuda/src/event.rs:        unsafe { cuda_check(cuEventRecord(self.0, stream.raw())) }
lightning-cuda/src/event.rs:                CUresult::CUDA_SUCCESS => Ok(true),
lightning-cuda/src/event.rs:                CUresult::CUDA_ERROR_NOT_READY => Ok(false),
lightning-cuda/src/event.rs:        unsafe { cuda_check(cuEventSynchronize(self.0)) }
lightning-cuda/src/event.rs:        unsafe { cuda_call(|t| cuEventElapsedTime(t, start.0, end.0)) }
lightning-cuda/src/event.rs:    /// The given `CUevent` object should be a valid CUDA event object.
lightning-cuda/src/event.rs:        f.debug_tuple("CudaEvent").field(&self.0).finish()
lightning-cuda/src/event.rs:    /// Flags for configuring CUDA events.
lightning-cuda/src/event.rs:    /// Represent the `CU_EVENT_*` flags from the CUDA API.
lightning-cuda/src/prelude.rs:pub use crate::context::create_context as cuda_create_context;
lightning-cuda/src/prelude.rs:pub use crate::context::destroy_context as cuda_destroy_context;
lightning-cuda/src/prelude.rs:pub use crate::context::release_device_context as cuda_release_device_context;
lightning-cuda/src/prelude.rs:pub use crate::context::retain_device_context as cuda_retain_device_context;
lightning-cuda/src/prelude.rs:pub use crate::context::ContextFlags as CudaContextFlags;
lightning-cuda/src/prelude.rs:pub use crate::context::ContextHandle as CudaContextHandle;
lightning-cuda/src/prelude.rs:pub use crate::copy::{copy as cuda_copy, copy_async as cuda_copy_async};
lightning-cuda/src/prelude.rs:pub use crate::copy::{copy_raw as cuda_copy_raw, copy_raw_async as cuda_copy_raw_async};
lightning-cuda/src/prelude.rs:pub use crate::device::Device as CudaDevice;
lightning-cuda/src/prelude.rs:pub use crate::device::DeviceAttribute as CudaDeviceAttribute;
lightning-cuda/src/prelude.rs:pub use crate::error::{cuda_call, cuda_check, Error as CudaError, Result as CudaResult};
lightning-cuda/src/prelude.rs:pub use crate::event::Event as CudaEvent;
lightning-cuda/src/prelude.rs:pub use crate::event::EventFlags as CudaEventFlags;
lightning-cuda/src/prelude.rs:pub use crate::init as cuda_init;
lightning-cuda/src/prelude.rs:pub use crate::mem::DeviceMem as CudaDeviceMem;
lightning-cuda/src/prelude.rs:pub use crate::mem::DevicePtr as CudaDevicePtr;
lightning-cuda/src/prelude.rs:pub use crate::mem::DeviceSlice as CudaDeviceSlice;
lightning-cuda/src/prelude.rs:pub use crate::mem::DeviceSliceMut as CudaDeviceSliceMut;
lightning-cuda/src/prelude.rs:pub use crate::mem::PinnedMem as CudaPinnedMem;
lightning-cuda/src/prelude.rs:pub use crate::module::Function as CudaFunction;
lightning-cuda/src/prelude.rs:pub use crate::module::Module as CudaModule;
lightning-cuda/src/prelude.rs:pub use crate::profiler::profiler_start as cuda_profiler_start;
lightning-cuda/src/prelude.rs:pub use crate::profiler::profiler_stop as cuda_profiler_stop;
lightning-cuda/src/prelude.rs:pub use crate::stream::Stream as CudaStream;
lightning-cuda/src/prelude.rs:pub use crate::stream::StreamFlags as CudaStreamFlags;
lightning-cuda/src/prelude.rs:pub use crate::version as cuda_version;
lightning-cuda/src/copy.rs:use crate::{cuda_check, DeviceMem, DeviceSlice, DeviceSliceMut, PinnedMem, Result, Stream};
lightning-cuda/src/copy.rs:use cuda_driver_sys::*;
lightning-cuda/src/copy.rs:    cuda_check(cuMemcpy(dst.raw(), src.raw(), n * size_of::<T>()))
lightning-cuda/src/copy.rs:    cuda_check(cuMemcpyAsync(
lightning-cuda/src/copy.rs:/// Parameters `src` and `dst` should be memory locations which can be read/written by CUDA
lightning-cuda/src/copy.rs:/// Parameters `src` and `dst` should be memory locations which can be read/written by CUDA
lightning-cuda/src/module.rs://! Loading CUDA modules and calling kernels.
lightning-cuda/src/module.rs:use crate::{cuda_call, cuda_check, Result, Stream};
lightning-cuda/src/module.rs:use cuda_driver_sys::*;
lightning-cuda/src/module.rs:        f.debug_tuple("CudaModule").field(&self.0).finish()
lightning-cuda/src/module.rs:        unsafe { cuda_call(|m| cuModuleLoad(m, file_name.as_ptr())).map(Self) }
lightning-cuda/src/module.rs:            cuda_call(|m| cuModuleLoadData(m, image)).map(Self)
lightning-cuda/src/module.rs:            cuda_call(|m| cuModuleLoadFatBinary(m, image)).map(Self)
lightning-cuda/src/module.rs:            let fun = cuda_call(|f| cuModuleGetFunction(f, self.0, name.as_ptr()))?;
lightning-cuda/src/module.rs:        f.debug_tuple("CudaFunction").field(&self.fun).finish()
lightning-cuda/src/module.rs:            cuda_call(|v| cuFuncGetAttribute(v, attr, self.fun))
lightning-cuda/src/module.rs:            cuda_check(cuFuncSetAttribute(self.fun, attr, value))
lightning-cuda/src/module.rs:        cuda_check(cuLaunchKernel(
lightning-memops/Cargo.toml:lightning-cuda = {path = "../lightning-cuda", features = ["serde"]}
lightning-memops/Cargo.toml:cuda-driver-sys = "0.3"
lightning-memops/src/lib.rs:pub use cuda::copy::cuda_copy;
lightning-memops/src/lib.rs:pub use cuda::fill::cuda_fill;
lightning-memops/src/lib.rs:pub use cuda::reduce::cuda_fold;
lightning-memops/src/lib.rs:pub use cuda::{cuda_reduce, MemOpsKernelsCache};
lightning-memops/src/lib.rs:mod cuda;
lightning-memops/src/cuda/fill.rs:use cuda_driver_sys::{cuMemsetD2D16Async, cuMemsetD2D32Async, cuMemsetD2D8Async};
lightning-memops/src/cuda/fill.rs:use lightning_core::accessor::{CudaMutAccessor, Strides};
lightning-memops/src/cuda/fill.rs:use lightning_cuda::{
lightning-memops/src/cuda/fill.rs:    cuda_check, ContextHandle as CudaContextHandle, DevicePtr as CudaDevicePtr,
lightning-memops/src/cuda/fill.rs:    Stream as CudaStream,
lightning-memops/src/cuda/fill.rs:pub unsafe fn cuda_fill(
lightning-memops/src/cuda/fill.rs:    context: CudaContextHandle,
lightning-memops/src/cuda/fill.rs:    stream: &CudaStream,
lightning-memops/src/cuda/fill.rs:    dst: CudaMutAccessor,
lightning-memops/src/cuda/fill.rs:            cuda_check(cuMemsetD2D8Async(
lightning-memops/src/cuda/fill.rs:            cuda_check(cuMemsetD2D16Async(
lightning-memops/src/cuda/fill.rs:            cuda_check(cuMemsetD2D32Async(
lightning-memops/src/cuda/fill.rs:    context: CudaContextHandle,
lightning-memops/src/cuda/fill.rs:    stream: &CudaStream,
lightning-memops/src/cuda/fill.rs:    dst_ptr: CudaDevicePtr,
lightning-memops/src/cuda/elementwise.rs:use lightning_cuda::prelude::*;
lightning-memops/src/cuda/elementwise.rs:    handle: CudaContextHandle,
lightning-memops/src/cuda/elementwise.rs:    stream: &CudaStream,
lightning-memops/src/cuda/mod.rs:use lightning_core::accessor::{CudaAccessor4, CudaMutAccessor3};
lightning-memops/src/cuda/mod.rs:use lightning_cuda::prelude::*;
lightning-memops/src/cuda/mod.rs:        handle: CudaContextHandle,
lightning-memops/src/cuda/mod.rs:        stream: &CudaStream,
lightning-memops/src/cuda/mod.rs:        handle: CudaContextHandle,
lightning-memops/src/cuda/mod.rs:        stream: &CudaStream,
lightning-memops/src/cuda/mod.rs:pub unsafe fn cuda_reduce(
lightning-memops/src/cuda/mod.rs:    context: CudaContextHandle,
lightning-memops/src/cuda/mod.rs:    stream: &CudaStream,
lightning-memops/src/cuda/mod.rs:    mut src: CudaAccessor4,
lightning-memops/src/cuda/mod.rs:    dst: CudaMutAccessor3,
lightning-memops/src/cuda/reduce.rs:use lightning_core::accessor::{CudaAccessor, CudaMutAccessor};
lightning-memops/src/cuda/reduce.rs:use lightning_cuda::{ContextHandle as CudaContextHandle, Stream as CudaStream};
lightning-memops/src/cuda/reduce.rs:pub unsafe fn cuda_fold(
lightning-memops/src/cuda/reduce.rs:    context: CudaContextHandle,
lightning-memops/src/cuda/reduce.rs:    stream: &CudaStream,
lightning-memops/src/cuda/reduce.rs:    src: CudaAccessor,
lightning-memops/src/cuda/reduce.rs:    dst: CudaMutAccessor,
lightning-memops/src/cuda/copy.rs:use cuda_driver_sys::CUmemorytype_enum::CU_MEMORYTYPE_UNIFIED;
lightning-memops/src/cuda/copy.rs:use cuda_driver_sys::{cuMemcpy2DAsync_v2, CUDA_MEMCPY2D};
lightning-memops/src/cuda/copy.rs:use lightning_core::accessor::{CudaAccessor, CudaMutAccessor, Strides};
lightning-memops/src/cuda/copy.rs:use lightning_cuda::{
lightning-memops/src/cuda/copy.rs:    cuda_check, ContextHandle as CudaContextHandle, DevicePtr as CudaDevicePtr,
lightning-memops/src/cuda/copy.rs:    Stream as CudaStream,
lightning-memops/src/cuda/copy.rs:pub unsafe fn cuda_copy(
lightning-memops/src/cuda/copy.rs:    context: CudaContextHandle,
lightning-memops/src/cuda/copy.rs:    stream: &CudaStream,
lightning-memops/src/cuda/copy.rs:    src: CudaAccessor,
lightning-memops/src/cuda/copy.rs:    dst: CudaMutAccessor,
lightning-memops/src/cuda/copy.rs:        cuda_check(cuMemcpy2DAsync_v2(
lightning-memops/src/cuda/copy.rs:            &CUDA_MEMCPY2D {
lightning-memops/src/cuda/copy.rs:    context: CudaContextHandle,
lightning-memops/src/cuda/copy.rs:    stream: &CudaStream,
lightning-memops/src/cuda/copy.rs:    dst_ptr: CudaDevicePtr,
lightning-memops/src/cuda/copy.rs:    src_ptr: CudaDevicePtr,
README.md:# Lightning: Fast data processing using GPUs on distributed platforms
README.md:Lightning is a framework for data processing using GPUs on distributed platforms.
README.md:The framework allows distributed multi-GPU execution of compute kernels functions written in CUDA in a way that is similar to programming a single GPU, without worrying about low-level details such as network communication, memory management, and data transfers.
README.md:This enables scaling of existing GPU kernels to much larger problem sizes, for beyond the memory capacity of a single GPU.
README.md:Lightning efficiently distributes the work/data across GPUS and maximizes efficiency by overlapping scheduling, data movement, and work when possible.
README.md:"Lightning: Scaling the GPU Programming Model Beyond a Single GPU",
lightning-codegen/Cargo.toml:lightning-cuda = {path = "../lightning-cuda", features = ["serde"]}
lightning-codegen/Cargo.toml:cuda-driver-sys = "0.3"
lightning-codegen/resources/lightning.h:#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 2
lightning-codegen/resources/lightning.h:#ifdef __CUDA_ARCH__
lightning-codegen/src/kernel.rs:use lightning_cuda::prelude::{CudaContextHandle, CudaStream};
lightning-codegen/src/kernel.rs:        context: CudaContextHandle,
lightning-codegen/src/kernel.rs:    pub fn compile(&mut self, context: CudaContextHandle, config: KernelConfig) -> Result {
lightning-codegen/src/kernel.rs:        context: CudaContextHandle,
lightning-codegen/src/kernel.rs:        stream: &CudaStream,
lightning-codegen/src/instance.rs:use cuda_driver_sys::{CUdeviceptr, CUfunction};
lightning-codegen/src/instance.rs:use lightning_cuda::prelude::*;
lightning-codegen/src/instance.rs:use lightning_cuda::Dim3 as CudaDim3;
lightning-codegen/src/instance.rs:        ptr: CudaDevicePtr,
lightning-codegen/src/instance.rs:        ptr: CudaDevicePtr,
lightning-codegen/src/instance.rs:    pub(super) module: CudaModule,
lightning-codegen/src/instance.rs:    Cuda(#[from] CudaError),
lightning-codegen/src/instance.rs:        context: CudaContextHandle,
lightning-codegen/src/instance.rs:        stream: &CudaStream,
lightning-codegen/src/instance.rs:        let max_grid_size = CudaDevice::current()?.max_grid_dim()?;
lightning-codegen/src/instance.rs:        stream: &CudaStream,
lightning-codegen/src/instance.rs:            (Ok(x), Ok(y), Ok(z)) => CudaDim3::new(x, y, z),
lightning-codegen/src/instance.rs:            (Ok(x), Ok(y), Ok(z)) => CudaDim3::new(x, y, z),
lightning-codegen/src/instance.rs:        CudaFunction::from_raw(self.fun_ptr).launch_async(
lightning-codegen/src/compile.rs:use lightning_cuda::prelude::*;
lightning-codegen/src/compile.rs:    Cuda(#[from] CudaError),
lightning-codegen/src/compile.rs:        context: CudaContextHandle,
lightning-codegen/src/compile.rs:    ) -> Result<CudaModule, CompilationError> {
lightning-codegen/src/compile.rs:        cmd.arg(format!("--gpu-architecture=sm_{}{}", major, minor));
lightning-codegen/src/compile.rs:            .try_with(|| unsafe { CudaModule::load_image(image.as_ptr() as *const c_void) })?;
Cargo.toml:cuda-driver-sys = "0.3"
Cargo.toml:lightning-cuda = { path = "lightning-cuda", features = ["serde"]}
Cargo.toml:    "lightning-cuda",
lightning-core/Cargo.toml:lightning-cuda = {path = "../lightning-cuda", features = ["serde"]}
lightning-core/src/data_type.rs:        // Size and alignments taken from the CUDA programming guide.
lightning-core/src/data_type.rs:        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-vector-types
lightning-core/src/data_type.rs:/// Alias for `float2` in CUDA.
lightning-core/src/data_type.rs:/// Alias for `float3` in CUDA.
lightning-core/src/data_type.rs:/// Alias for `float4` in CUDA.
lightning-core/src/data_type.rs:/// Alias for `double2` in CUDA.
lightning-core/src/accessor.rs:use lightning_cuda::prelude::*;
lightning-core/src/accessor.rs:use lightning_cuda::DevicePtr;
lightning-core/src/accessor.rs:    Device(CudaDevicePtr, DeviceId),
lightning-core/src/accessor.rs:    DeviceMut(CudaDevicePtr, DeviceId),
lightning-core/src/accessor.rs:impl Data for CudaDevicePtr {
lightning-core/src/accessor.rs:    type Ptr = CudaDevicePtr;
lightning-core/src/accessor.rs:impl DataMut for CudaDevicePtr {
lightning-core/src/accessor.rs:    type PtrMut = CudaDevicePtr;
lightning-core/src/accessor.rs:type CudaAccessorN<const N: usize> = Accessor<CudaDevicePtr, N>;
lightning-core/src/accessor.rs:type CudaMutAccessorN<const N: usize> = Accessor<CudaDevicePtr, N>;
lightning-core/src/accessor.rs:pub type CudaAccessor = CudaAccessorN<MAX_DIMS>;
lightning-core/src/accessor.rs:pub type CudaMutAccessor = CudaMutAccessorN<MAX_DIMS>;
lightning-core/src/accessor.rs:pub type CudaAccessor1 = CudaAccessorN<1>;
lightning-core/src/accessor.rs:pub type CudaMutAccessor1 = CudaMutAccessorN<1>;
lightning-core/src/accessor.rs:pub type CudaAccessor2 = CudaAccessorN<2>;
lightning-core/src/accessor.rs:pub type CudaMutAccessor2 = CudaMutAccessorN<2>;
lightning-core/src/accessor.rs:pub type CudaAccessor3 = CudaAccessorN<3>;
lightning-core/src/accessor.rs:pub type CudaMutAccessor3 = CudaMutAccessorN<3>;
lightning-core/src/accessor.rs:pub type CudaAccessor4 = CudaAccessorN<4>;
lightning-core/src/accessor.rs:pub type CudaMutAccessor4 = CudaMutAccessorN<4>;
lightning-core/src/accessor.rs:    pub fn as_device(&self, id: DeviceId) -> Option<CudaAccessorN<N>> {
lightning-core/src/accessor.rs:            UnifiedPtr::Host(ptr) => Some(DevicePtr::new(ptr as _)), // CUDA unified mem
lightning-core/src/accessor.rs:    pub fn as_device_mut(&self, id: DeviceId) -> Option<CudaAccessorN<N>> {
resources/lightning.h:#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 2
resources/lightning.h:#ifdef __CUDA_ARCH__
src/planner/distribution/rows.rs:pub struct RowBlockCyclic<P = AllGPUs> {
src/planner/distribution/rows.rs:impl RowBlockCyclic<AllGPUs> {
src/planner/distribution/rows.rs:        Self::with_memories(block_size, AllGPUs)
src/planner/distribution/rows.rs:        let places = AllGPUs.generate(system, None).unwrap();
src/planner/distribution/tile.rs:    AllGPUs, ChunkDescriptor, ChunkQueryResult, DataDistribution, IntoDataDistribution,
src/planner/distribution/tile.rs:pub struct TileDist<P = AllGPUs> {
src/planner/distribution/tile.rs:        Self::with_memories(tile_size.into(), AllGPUs)
src/planner/distribution/stencil3d.rs:pub struct Stencil3DDist<P = AllGPUs> {
src/planner/distribution/stencil3d.rs:impl Stencil3DDist<AllGPUs> {
src/planner/distribution/stencil3d.rs:        Self::with_memories(tile_size, halo, AllGPUs)
src/planner/distribution/stencil2d.rs:pub struct Stencil2DDist<P = AllGPUs> {
src/planner/distribution/stencil2d.rs:impl Stencil2DDist<AllGPUs> {
src/planner/distribution/stencil2d.rs:        Self::with_memories(block_size, halo_size, AllGPUs)
src/planner/distribution/columns.rs:pub struct ColumnBlockCyclic<P = AllGPUs> {
src/planner/distribution/columns.rs:impl ColumnBlockCyclic<AllGPUs> {
src/planner/distribution/columns.rs:        Self::with_memories(block_size, AllGPUs)
src/planner/distribution/columns.rs:        let places = AllGPUs.generate(system, None).unwrap();
src/planner/distribution/mod.rs:pub struct AllGPUs;
src/planner/distribution/mod.rs:impl MemoryDistribution for AllGPUs {
src/planner/distribution/stencil.rs:pub struct StencilDist<P = AllGPUs> {
src/planner/distribution/stencil.rs:impl StencilDist<AllGPUs> {
src/planner/cuda.rs:use super::task::CudaLaunchTasklet;
src/planner/cuda.rs:    Affine, CudaArg, CudaKernelId, DataValue, Dim, Dim3, ExecutorId, MemoryId, Point, Rect,
src/planner/cuda.rs:pub(crate) enum CudaLauncherArg {
src/planner/cuda.rs:pub(crate) struct CudaLauncher {
src/planner/cuda.rs:    pub(crate) kernel_id: CudaKernelId,
src/planner/cuda.rs:    pub(crate) args: Vec<CudaLauncherArg>,
src/planner/cuda.rs:impl CudaLauncher {
src/planner/cuda.rs:                CudaLauncherArg::Value { value } => CudaArg::Value(value.clone()),
src/planner/cuda.rs:                &CudaLauncherArg::Array {
src/planner/cuda.rs:        let tasklet = CudaLaunchTasklet {
src/planner/cuda.rs:    ) -> Result<CudaArg> {
src/planner/cuda.rs:            Ok(CudaArg::array(ndims, array_index, subdomain, transform))
src/planner/cuda.rs:            Ok(CudaArg::array_per_block(
src/planner/task/cuda_kernel.rs:use crate::types::{CudaArg, CudaKernelId, Dim, Executor, GenericAccessor, Point, Tasklet};
src/planner/task/cuda_kernel.rs:use crate::worker::executor::CudaExecutor;
src/planner/task/cuda_kernel.rs:pub(crate) struct CudaLaunchTasklet {
src/planner/task/cuda_kernel.rs:    pub kernel_id: CudaKernelId,
src/planner/task/cuda_kernel.rs:    pub args: Vec<CudaArg>,
src/planner/task/cuda_kernel.rs:impl Tasklet for CudaLaunchTasklet {
src/planner/task/cuda_kernel.rs:        let executor = executor.downcast_ref::<CudaExecutor>()?;
src/planner/task/fill.rs:use crate::worker::executor::{CudaExecutor, HostExecutor};
src/planner/task/fill.rs:        } else if let Ok(executor) = executor.downcast_ref::<CudaExecutor>() {
src/planner/task/mod.rs:mod cuda_kernel;
src/planner/task/mod.rs:pub(crate) use self::cuda_kernel::*;
src/planner/task/mod.rs:    register_tasklet::<CudaLaunchTasklet>();
src/planner/task/fold.rs:use crate::worker::executor::{CudaExecutor, HostExecutor};
src/planner/task/fold.rs:        if let Ok(executor) = executor.downcast_ref::<CudaExecutor>() {
src/planner/task/reduce.rs:use crate::worker::executor::CudaExecutor;
src/planner/task/reduce.rs:        let executor = executor.downcast_ref::<CudaExecutor>()?;
src/planner/mod.rs:pub(crate) mod cuda;
src/api/kernel.rs:use crate::planner::cuda::{CudaLauncher, CudaLauncherArg};
src/api/kernel.rs:    CudaKernelId, DataType, DataValue, Dim, Dim3, ExecutorId, Point, Rect, DTYPE_I64, MAX_DIMS,
src/api/kernel.rs:pub struct CudaKernelBuilder {
src/api/kernel.rs:impl CudaKernelBuilder {
src/api/kernel.rs:    pub fn compile(&mut self, context: &Context) -> Result<CudaKernel> {
src/api/kernel.rs:        Ok(CudaKernel {
src/api/kernel.rs:pub struct CudaKernel {
src/api/kernel.rs:    pub(crate) id: CudaKernelId,
src/api/kernel.rs:impl CudaKernel {
src/api/kernel.rs:    pub fn id(&self) -> CudaKernelId {
src/api/kernel.rs:    fn build_launcher<A: KernelArgs>(&self, block_size: Dim, args: A) -> Result<CudaLauncher> {
src/api/kernel.rs:        let mut builder = CudaLauncherBuilder::new(self, block_size);
src/api/kernel.rs:    fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result;
src/api/kernel.rs:    fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result {
src/api/kernel.rs:            fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result {
src/api/kernel.rs:    fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result;
src/api/kernel.rs:    fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result {
src/api/kernel.rs:    fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result {
src/api/kernel.rs:    fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result {
src/api/kernel.rs:    fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result {
src/api/kernel.rs:    fn apply(&self, launch: &mut CudaLauncherBuilder) -> Result {
src/api/kernel.rs:pub struct CudaLauncherBuilder<'a> {
src/api/kernel.rs:    kernel: &'a CudaKernel,
src/api/kernel.rs:    args: Vec<CudaLauncherArg>,
src/api/kernel.rs:impl<'a> CudaLauncherBuilder<'a> {
src/api/kernel.rs:    pub fn new(kernel: &'a CudaKernel, block_size: Dim3) -> Self {
src/api/kernel.rs:        self.args.push(CudaLauncherArg::Value { value });
src/api/kernel.rs:        self.args.push(CudaLauncherArg::Array {
src/api/kernel.rs:                if let (Param::Value { name, .. }, CudaLauncherArg::Value { value }) = (param, arg)
src/api/kernel.rs:    fn into_launcher(self) -> Result<CudaLauncher> {
src/api/kernel.rs:        Ok(CudaLauncher {
src/api/context.rs:use super::{Array, ArrayView, CudaKernel, CudaKernelBuilder};
src/api/context.rs:    pub fn compile_kernel(&self, mut def: CudaKernelBuilder) -> Result<CudaKernel> {
src/worker/executor/cuda.rs:use lightning_cuda::prelude::*;
src/worker/executor/cuda.rs:use lightning_cuda::Dim3 as CudaDim3;
src/worker/executor/cuda.rs:    Affine, CudaAccessor, CudaAccessor4, CudaArg, CudaKernelId, CudaMutAccessor, CudaMutAccessor3,
src/worker/executor/cuda.rs:    context: CudaContextHandle,
src/worker/executor/cuda.rs:    kernels: RwLock<HashMap<CudaKernelId, Mutex<Kernel>>>,
src/worker/executor/cuda.rs:        src: CudaAccessor,
src/worker/executor/cuda.rs:        dst: CudaMutAccessor,
src/worker/executor/cuda.rs:pub(crate) struct CudaExecutorThread {
src/worker/executor/cuda.rs:impl CudaExecutorThread {
src/worker/executor/cuda.rs:        context: CudaContextHandle,
src/worker/executor/cuda.rs:            executors.push(CudaExecutor {
src/worker/executor/cuda.rs:                stream: context.try_with(CudaStream::new)?,
src/worker/executor/cuda.rs:                    .name(format!("cuda-executor-{}", index))
src/worker/executor/cuda.rs:    pub(crate) fn context(&self) -> CudaContextHandle {
src/worker/executor/cuda.rs:        use lightning_cuda::DeviceAttribute::*;
src/worker/executor/cuda.rs:    pub(crate) fn compile_kernel(&self, id: CudaKernelId, def: ModuleDef) -> Result {
src/worker/executor/cuda.rs:        src: CudaAccessor,
src/worker/executor/cuda.rs:        dst: CudaMutAccessor,
src/worker/executor/cuda.rs:fn main_loop(executor: CudaExecutor, receiver: Receiver<QueueItem>) {
src/worker/executor/cuda.rs:pub struct CudaExecutor {
src/worker/executor/cuda.rs:    stream: CudaStream,
src/worker/executor/cuda.rs:impl Executor for CudaExecutor {
src/worker/executor/cuda.rs:impl Display for CudaExecutor {
src/worker/executor/cuda.rs:impl CudaExecutor {
src/worker/executor/cuda.rs:    pub fn stream(&self) -> &CudaStream {
src/worker/executor/cuda.rs:    pub fn context(&self) -> CudaContextHandle {
src/worker/executor/cuda.rs:    pub unsafe fn fill_async(&self, dst: CudaMutAccessor, value: DataValue) -> Result {
src/worker/executor/cuda.rs:        lightning_memops::cuda_fill(
src/worker/executor/cuda.rs:    pub unsafe fn copy_async(&self, src: CudaAccessor, dst: CudaMutAccessor) -> Result {
src/worker/executor/cuda.rs:        lightning_memops::cuda_copy(inner.context, &self.stream, &inner.memops_kernels, src, dst)
src/worker/executor/cuda.rs:        src: CudaAccessor4,
src/worker/executor/cuda.rs:        dst: CudaMutAccessor3,
src/worker/executor/cuda.rs:        lightning_memops::cuda_reduce(
src/worker/executor/cuda.rs:        src: CudaAccessor,
src/worker/executor/cuda.rs:        dst: CudaMutAccessor,
src/worker/executor/cuda.rs:        lightning_memops::cuda_fold(
src/worker/executor/cuda.rs:    pub fn with_kernel<F>(&self, id: CudaKernelId, callback: F) -> Result
src/worker/executor/cuda.rs:        F: FnOnce(&CudaStream, &mut Kernel) -> Result,
src/worker/executor/cuda.rs:        id: CudaKernelId,
src/worker/executor/cuda.rs:        args: &[CudaArg],
src/worker/executor/cuda.rs:        fn to_dim3(v: [u64; 3]) -> Result<CudaDim3> {
src/worker/executor/cuda.rs:                (Ok(x), Ok(y), Ok(z)) => Ok(CudaDim3::new(x, y, z)),
src/worker/executor/cuda.rs:            array: CudaAccessor,
src/worker/executor/cuda.rs:                CudaDevicePtr::from_raw(ptr),
src/worker/executor/cuda.rs:            array: CudaAccessor,
src/worker/executor/cuda.rs:            array: CudaAccessor,
src/worker/executor/cuda.rs:                    CudaArg::Value(ref v) => KernelArg::value(v.clone()),
src/worker/executor/cuda.rs:                    CudaArg::Array(arg) => {
src/worker/executor/mod.rs:mod cuda;
src/worker/executor/mod.rs:pub(crate) use cuda::CudaExecutorThread;
src/worker/executor/mod.rs:pub use cuda::CudaExecutor;
src/worker/task/executor_set.rs:use crate::worker::executor::{CudaExecutorThread, HostThreadPool};
src/worker/task/executor_set.rs:    devices: Vec<CudaExecutorThread>,
src/worker/task/executor_set.rs:        devices: Vec<CudaExecutorThread>,
src/worker/mod.rs:use self::executor::{CudaExecutorThread, HostThreadPool};
src/worker/mod.rs:    CudaKernelId, DeviceId, DeviceInfo, ExecutorId, ExecutorKind, MemoryId, MemoryKind,
src/worker/mod.rs:use cuda_driver_sys::cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY;
src/worker/mod.rs:use lightning_codegen::ModuleDef as CudaModuleDef;
src/worker/mod.rs:use lightning_cuda::prelude::*;
src/worker/mod.rs:    executors: Vec<CudaExecutorThread>,
src/worker/mod.rs:        executors: Vec<CudaExecutorThread>,
src/worker/mod.rs:    fn handle_compilation_request(&mut self, id: CudaKernelId, def: CudaModuleDef) -> Result {
src/worker/mod.rs:fn initialize_cuda() -> Result<Vec<CudaContextHandle>> {
src/worker/mod.rs:    cuda_init()?;
src/worker/mod.rs:    for device in CudaDevice::all()? {
src/worker/mod.rs:        use CudaDeviceAttribute::*;
src/worker/mod.rs:        let context = cuda_create_context(device, CudaContextFlags::MAP_HOST)?;
src/worker/mod.rs:        context.try_with(|| cuda_profiler_start())?;
src/worker/mod.rs:unsafe fn destroy_cuda(contexts: Vec<CudaContextHandle>) -> Result {
src/worker/mod.rs:        if let Err(e) = context.try_with(|| cuda_profiler_stop()) {
src/worker/mod.rs:    // Sleep for a second to allow CUDA to write back profiling results.
src/worker/mod.rs:        if let Err(e) = cuda_destroy_context(context) {
src/worker/mod.rs:            error!("failed to destroy CUDA context: {:?}", e);
src/worker/mod.rs:    contexts: &[CudaContextHandle],
src/worker/mod.rs:        let executor = CudaExecutorThread::new(node_id, id, ctx, 2, config.specialization_policy)?;
src/worker/mod.rs:                CudaDeviceMem::empty(size)
src/worker/mod.rs:                        break Err(CudaError::new(CUDA_ERROR_OUT_OF_MEMORY).unwrap_err());
src/worker/mod.rs:                    match CudaDeviceMem::empty(free - space) {
src/worker/mod.rs:                        Err(e) if e.raw() == CUDA_ERROR_OUT_OF_MEMORY => {}
src/worker/mod.rs:    match initialize_cuda() {
src/worker/mod.rs:            // Destroying the CUDA context is unsafe since it cannot be called while CUDA activities are
src/worker/mod.rs:            unsafe { destroy_cuda(contexts) }?;
src/worker/memory/copy_engine.rs:use cuda_driver_sys::cudaError_enum::CUDA_ERROR_INVALID_VALUE;
src/worker/memory/copy_engine.rs:use cuda_driver_sys::{
src/worker/memory/copy_engine.rs:    cuMemcpyPeerAsync, CUmemorytype_enum, CUDA_MEMCPY3D_PEER,
src/worker/memory/copy_engine.rs:use lightning_cuda::prelude::*;
src/worker/memory/copy_engine.rs:    context: CudaContextHandle,
src/worker/memory/copy_engine.rs:    d2h_lo: CudaStream,
src/worker/memory/copy_engine.rs:    h2d_lo: CudaStream,
src/worker/memory/copy_engine.rs:    d2h_hi: CudaStream,
src/worker/memory/copy_engine.rs:    h2d_hi: CudaStream,
src/worker/memory/copy_engine.rs:    pub(crate) fn new(contexts: Vec<CudaContextHandle>) -> CudaResult<Self> {
src/worker/memory/copy_engine.rs:                    d2h_lo: CudaStream::new()?,
src/worker/memory/copy_engine.rs:                    h2d_lo: CudaStream::new()?,
src/worker/memory/copy_engine.rs:                    d2h_hi: CudaStream::new()?,
src/worker/memory/copy_engine.rs:                    h2d_hi: CudaStream::new()?,
src/worker/memory/copy_engine.rs:                    CudaDevice::can_access_peer(&left, right)?
src/worker/memory/copy_engine.rs:        F: FnOnce(CudaResult) + Send + 'static,
src/worker/memory/copy_engine.rs:        G: FnOnce(&CudaStream) -> CudaResult,
src/worker/memory/copy_engine.rs:                    let result = CudaResult::and(result_submit, result_sync);
src/worker/memory/copy_engine.rs:        dst_ptr: CudaDevicePtr,
src/worker/memory/copy_engine.rs:        F: FnOnce(CudaResult) + Send + 'static,
src/worker/memory/copy_engine.rs:                cuda_check(cuMemcpyHtoDAsync_v2(
src/worker/memory/copy_engine.rs:        dst_ptr: CudaDevicePtr,
src/worker/memory/copy_engine.rs:        F: FnOnce(CudaResult) + Send + 'static,
src/worker/memory/copy_engine.rs:                cuda_check(cuMemcpy3DPeerAsync(&params, stream.raw()))
src/worker/memory/copy_engine.rs:        src_ptr: CudaDevicePtr,
src/worker/memory/copy_engine.rs:        F: FnOnce(CudaResult) + Send + 'static,
src/worker/memory/copy_engine.rs:                cuda_check(cuMemcpyDtoHAsync_v2(
src/worker/memory/copy_engine.rs:        src_ptr: CudaDevicePtr,
src/worker/memory/copy_engine.rs:        F: FnOnce(CudaResult) + Send + 'static,
src/worker/memory/copy_engine.rs:                cuda_check(cuMemcpy3DPeerAsync(&params, stream.raw()))
src/worker/memory/copy_engine.rs:        src_ptr: CudaDevicePtr,
src/worker/memory/copy_engine.rs:        dst_ptr: CudaDevicePtr,
src/worker/memory/copy_engine.rs:        F: FnOnce(CudaResult) + Send + 'static,
src/worker/memory/copy_engine.rs:                    cuda_check(cuMemcpyDtoDAsync_v2(
src/worker/memory/copy_engine.rs:                    cuda_check(cuMemcpyPeerAsync(
src/worker/memory/copy_engine.rs:        src_ptr: CudaDevicePtr,
src/worker/memory/copy_engine.rs:        dst_ptr: CudaDevicePtr,
src/worker/memory/copy_engine.rs:        F: FnOnce(CudaResult) + Send + 'static,
src/worker/memory/copy_engine.rs:                cuda_check(cuMemcpy3DPeerAsync(&params, stream.raw()))
src/worker/memory/copy_engine.rs:) -> CudaResult<CUDA_MEMCPY3D_PEER> {
src/worker/memory/copy_engine.rs:            return Err(CudaError::new(CUDA_ERROR_INVALID_VALUE).unwrap_err());
src/worker/memory/copy_engine.rs:        return Err(CudaError::new(CUDA_ERROR_INVALID_VALUE).unwrap_err());
src/worker/memory/copy_engine.rs:    let params = CUDA_MEMCPY3D_PEER {
src/worker/memory/manager.rs:use lightning_cuda::prelude::*;
src/worker/memory/manager.rs:        result: CudaResult,
src/worker/memory/manager.rs:    device_entries: [Entry<CudaDevicePtr>; MAX_DEVICES],
src/worker/memory/manager.rs:    type Ptr = CudaDevicePtr;
src/worker/memory/manager.rs:    ) -> &'a mut Entry<CudaDevicePtr> {
src/worker/memory/allocator.rs:use lightning_cuda::prelude::*;
src/worker/memory/allocator.rs:    mem: CudaDeviceMem<u8>,
src/worker/memory/allocator.rs:    pub(crate) fn new(mem: CudaDeviceMem<u8>) -> Self {
src/worker/memory/allocator.rs:    ) -> Result<CudaDevicePtr<u8>, MemoryError> {
src/worker/memory/allocator.rs:        let result = Ok(unsafe { CudaDevicePtr::from_raw(offset as _) });
src/worker/memory/allocator.rs:    pub(crate) fn deallocate(&mut self, ptr: CudaDevicePtr<u8>, size: usize) {
src/worker/memory/allocator.rs:    mem: CudaPinnedMem<u8>,
src/worker/memory/allocator.rs:    context: CudaContextHandle,
src/worker/memory/allocator.rs:        context: CudaContextHandle,
src/worker/memory/allocator.rs:                use cuda_driver_sys::cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY;
src/worker/memory/allocator.rs:                    .try_with(|| -> CudaResult<_> { CudaPinnedMem::empty(block_size) })
src/worker/memory/allocator.rs:                    Err(e) if e.raw() == CUDA_ERROR_OUT_OF_MEMORY => None,
src/network/message.rs:use lightning_codegen::ModuleDef as CudaModuleDef;
src/network/message.rs:use lightning_cuda::prelude::CudaError;
src/network/message.rs:use crate::types::{CudaKernelId, EventId, SyncId, TaskletOutput, WorkerInfo};
src/network/message.rs:    Compile(CudaKernelId, CudaModuleDef),
src/network/message.rs:    CompileResult(CudaKernelId, Result<(), SerializedError>),
src/network/message.rs:    Cuda(CudaError),
src/network/message.rs:        if let Some(e) = r.downcast_ref::<CudaError>() {
src/network/message.rs:            Cuda(e.clone())
src/network/message.rs:            Cuda(e) => anyhow::Error::from(e),
src/types/mod.rs:pub struct CudaKernelId(pub(crate) u64);
src/types/mod.rs:impl Display for CudaKernelId {
src/types/mod.rs:pub enum CudaArg {
src/types/mod.rs:    Array(CudaArgArray),
src/types/mod.rs:impl CudaArg {
src/types/mod.rs:        Self::Array(CudaArgArray {
src/types/mod.rs:        Self::Array(CudaArgArray {
src/types/mod.rs:pub struct CudaArgArray {
src/driver/trace.rs:use crate::types::{CudaKernelId, WorkerId};
src/driver/trace.rs:    kernel_names: HashMap<CudaKernelId, String>,
src/driver/trace.rs:    pub(super) fn kernel_compiled(&mut self, id: CudaKernelId, def: ModuleDef) {
src/driver/internal.rs:use lightning_codegen::ModuleDef as CudaModuleDef;
src/driver/internal.rs:    CudaKernelId, DriverConfig, EventId, SyncId, SystemInfo, TaskletCallback, TaskletOutput,
src/driver/internal.rs:    definition: CudaModuleDef,
src/driver/internal.rs:    pending_compilation: HashMap<CudaKernelId, KernelCompilation>,
src/driver/internal.rs:    fn compile_cuda_kernel(
src/driver/internal.rs:        definition: CudaModuleDef,
src/driver/internal.rs:    ) -> Result<CudaKernelId> {
src/driver/internal.rs:        let id = CudaKernelId(self.next_kernel_id.get_and_increment());
src/driver/internal.rs:        kernel_id: CudaKernelId,
src/driver/internal.rs:    pub fn compile_kernel(&self, kernel_def: CudaModuleDef) -> Result<CudaKernelId> {
src/driver/internal.rs:            .compile_cuda_kernel(kernel_def, promise)?;

```
