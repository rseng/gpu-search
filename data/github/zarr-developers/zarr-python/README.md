# https://github.com/zarr-developers/zarr-python

```console
docs/talks/scipy2019/submission.rst:* `Matthew Rocklin <https://github.com/mrocklin>`_, NVIDIA
tests/test_buffer.py:from zarr.core.buffer import ArrayLike, BufferPrototype, NDArrayLike, cpu, gpu
tests/test_buffer.py:from zarr.testing.utils import gpu_test
tests/test_buffer.py:@gpu_test
tests/test_buffer.py:async def test_async_array_gpu_prototype() -> None:
tests/test_buffer.py:    """Test the use of the GPU buffer prototype"""
tests/test_buffer.py:        StorePath(MemoryStore(mode="w")) / "test_async_array_gpu_prototype",
tests/test_buffer.py:        prototype=gpu.buffer_prototype,
tests/test_buffer.py:    got = await a.getitem(selection=(slice(0, 9), slice(0, 9)), prototype=gpu.buffer_prototype)
tests/test_buffer.py:@gpu_test
tests/test_buffer.py:async def test_codecs_use_of_gpu_prototype() -> None:
tests/test_buffer.py:        StorePath(MemoryStore(mode="w")) / "test_codecs_use_of_gpu_prototype",
tests/test_buffer.py:        prototype=gpu.buffer_prototype,
tests/test_buffer.py:    got = await a.getitem(selection=(slice(0, 10), slice(0, 10)), prototype=gpu.buffer_prototype)
tests/test_store/test_memory.py:from zarr.core.buffer import Buffer, cpu, gpu
tests/test_store/test_memory.py:from zarr.storage.memory import GpuMemoryStore, MemoryStore
tests/test_store/test_memory.py:from zarr.testing.utils import gpu_test
tests/test_store/test_memory.py:@gpu_test
tests/test_store/test_memory.py:class TestGpuMemoryStore(StoreTests[GpuMemoryStore, gpu.Buffer]):
tests/test_store/test_memory.py:    store_cls = GpuMemoryStore
tests/test_store/test_memory.py:    buffer_cls = gpu.Buffer
tests/test_store/test_memory.py:    async def set(self, store: GpuMemoryStore, key: str, value: Buffer) -> None:
tests/test_store/test_memory.py:    def store(self, store_kwargs: str | None | dict[str, gpu.Buffer]) -> GpuMemoryStore:
tests/test_store/test_memory.py:    def test_store_repr(self, store: GpuMemoryStore) -> None:
tests/test_store/test_memory.py:        assert str(store) == f"gpumemory://{id(store._store_dict)}"
tests/test_store/test_memory.py:    def test_store_supports_writes(self, store: GpuMemoryStore) -> None:
tests/test_store/test_memory.py:    def test_store_supports_listing(self, store: GpuMemoryStore) -> None:
tests/test_store/test_memory.py:    def test_store_supports_partial_writes(self, store: GpuMemoryStore) -> None:
tests/test_store/test_memory.py:    def test_list_prefix(self, store: GpuMemoryStore) -> None:
tests/test_store/test_memory.py:    def test_dict_reference(self, store: GpuMemoryStore) -> None:
tests/test_store/test_memory.py:        result = GpuMemoryStore(store_dict=store_dict)
tests/test_store/test_memory.py:            "a": gpu.Buffer.from_bytes(b"aaaa"),
tests/test_store/test_memory.py:        result = GpuMemoryStore.from_dict(d)
tests/test_store/test_memory.py:            assert type(v) is gpu.Buffer
tests/conftest.py:        request.node.add_marker(pytest.mark.gpu)
pyproject.toml:gpu = [
pyproject.toml:    "cupy-cuda12x",
pyproject.toml:features = ["gpu"]
pyproject.toml:run-coverage-gpu = "pip install cupy-cuda12x && pytest -m gpu --cov-config=pyproject.toml --cov=pkg --cov=tests"
pyproject.toml:[tool.hatch.envs.gputest]
pyproject.toml:features = ["test", "extra", "gpu"]
pyproject.toml:[[tool.hatch.envs.gputest.matrix]]
pyproject.toml:[tool.hatch.envs.gputest.scripts]
pyproject.toml:run-coverage = "pytest -m gpu --cov-config=pyproject.toml --cov=pkg --cov=tests"
pyproject.toml:    "ignore:Creating a zarr.buffer.gpu.*:UserWarning",
pyproject.toml:    "gpu: mark a test as requiring CuPy and GPU"
src/zarr/storage/memory.py:from zarr.core.buffer import Buffer, gpu
src/zarr/storage/memory.py:class GpuMemoryStore(MemoryStore):
src/zarr/storage/memory.py:    """A GPU only memory store that stores every chunk in GPU memory irrespective
src/zarr/storage/memory.py:    GPU Buffers.
src/zarr/storage/memory.py:    Writing data to this store through ``.set`` will move the buffer to the GPU
src/zarr/storage/memory.py:        A mutable mapping with string keys and :class:`zarr.core.buffer.gpu.Buffer`
src/zarr/storage/memory.py:    _store_dict: MutableMapping[str, gpu.Buffer]  # type: ignore[assignment]
src/zarr/storage/memory.py:        store_dict: MutableMapping[str, gpu.Buffer] | None = None,
src/zarr/storage/memory.py:        return f"gpumemory://{id(self._store_dict)}"
src/zarr/storage/memory.py:        return f"GpuMemoryStore({str(self)!r})"
src/zarr/storage/memory.py:        Create a GpuMemoryStore from a dictionary of buffers at any location.
src/zarr/storage/memory.py:        The dictionary backing the newly created ``GpuMemoryStore`` will not be
src/zarr/storage/memory.py:            will be moved into a :class:`gpu.Buffer`.
src/zarr/storage/memory.py:        GpuMemoryStore
src/zarr/storage/memory.py:        gpu_store_dict = {k: gpu.Buffer.from_buffer(v) for k, v in store_dict.items()}
src/zarr/storage/memory.py:        return cls(gpu_store_dict)
src/zarr/storage/memory.py:        # Convert to gpu.Buffer
src/zarr/storage/memory.py:        gpu_value = value if isinstance(value, gpu.Buffer) else gpu.Buffer.from_buffer(value)
src/zarr/storage/memory.py:        await super().set(key, gpu_value, byte_range=byte_range)
src/zarr/core/buffer/gpu.py:    """A flat contiguous memory block on the GPU
src/zarr/core/buffer/gpu.py:    CUDA device memory, or something else. The only requirement is that the
src/zarr/core/buffer/gpu.py:                "Cannot use zarr.buffer.gpu.Buffer without cupy. Please install cupy."
src/zarr/core/buffer/gpu.py:        if not hasattr(array_like, "__cuda_array_interface__"):
src/zarr/core/buffer/gpu.py:            # Slow copy based path for arrays that don't support the __cuda_array_interface__
src/zarr/core/buffer/gpu.py:                "Creating a zarr.buffer.gpu.Buffer with an array that does not support the "
src/zarr/core/buffer/gpu.py:                "__cuda_array_interface__ for zero-copy transfers, "
src/zarr/core/buffer/gpu.py:        """Create an GPU Buffer given an arbitrary Buffer
src/zarr/core/buffer/gpu.py:        GPU and will trigger a copy if not.
src/zarr/core/buffer/gpu.py:            New GPU Buffer constructed from `buffer`
src/zarr/core/buffer/gpu.py:        gpu_other = Buffer(other_array)
src/zarr/core/buffer/gpu.py:        gpu_other_array = gpu_other.as_array_like()
src/zarr/core/buffer/gpu.py:            cp.concatenate((cp.asanyarray(self._data), cp.asanyarray(gpu_other_array)))
src/zarr/core/buffer/gpu.py:    """A n-dimensional memory block on the GPU
src/zarr/core/buffer/gpu.py:    CUDA device memory, or something else. The only requirement is that the
src/zarr/core/buffer/gpu.py:                "Cannot use zarr.buffer.gpu.NDBuffer without cupy. Please install cupy."
src/zarr/core/buffer/gpu.py:        if not hasattr(array, "__cuda_array_interface__"):
src/zarr/core/buffer/gpu.py:            # Slow copy based path for arrays that don't support the __cuda_array_interface__
src/zarr/core/buffer/gpu.py:                "Creating a zarr.buffer.gpu.NDBuffer with an array that does not support the "
src/zarr/core/buffer/gpu.py:                "__cuda_array_interface__ for zero-copy transfers, "
src/zarr/core/buffer/gpu.py:            gpu_value = NDBuffer(value.as_ndarray_like())
src/zarr/core/buffer/gpu.py:            value = gpu_value._data
src/zarr/core/buffer/cpu.py:    CUDA device memory, or something else. The only requirement is that the
src/zarr/core/buffer/cpu.py:    CUDA device memory, or something else. The only requirement is that the
src/zarr/core/buffer/core.py:    CUDA device memory, or something else. The only requirement is that the
src/zarr/core/buffer/core.py:    CUDA device memory, or something else. The only requirement is that the
src/zarr/testing/utils.py:        return cast(bool, cupy.cuda.runtime.getDeviceCount() > 0)
src/zarr/testing/utils.py:    except cupy.cuda.runtime.CUDARuntimeError:
src/zarr/testing/utils.py:# Decorator for GPU tests
src/zarr/testing/utils.py:def gpu_test(func: T_Callable) -> T_Callable:
src/zarr/testing/utils.py:        pytest.mark.gpu(
src/zarr/testing/utils.py:            pytest.mark.skipif(not has_cupy(), reason="CuPy not installed or no GPU available")(

```
