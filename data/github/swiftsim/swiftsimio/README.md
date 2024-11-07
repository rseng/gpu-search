# https://github.com/SWIFTSIM/swiftsimio

```console
docs/source/visualisation/projection.rst:+ ``gpu``: The same as ``fast`` but uses CUDA for faster computation on supported
docs/source/visualisation/projection.rst:  GPUs. The parallel implementation is the same function as the non-parallel.
tests/test_visualisation.py:from swiftsimio.optional_packages import CudaSupportError, CUDA_AVAILABLE
tests/test_visualisation.py:        except CudaSupportError:
tests/test_visualisation.py:            if CUDA_AVAILABLE:
tests/test_visualisation.py:                raise ImportError("Optional loading of the CUDA module is broken")
tests/test_visualisation.py:        except CudaSupportError:
tests/test_visualisation.py:            if CUDA_AVAILABLE:
tests/test_visualisation.py:                raise ImportError("Optional loading of the CUDA module is broken")
tests/test_visualisation.py:        except CudaSupportError:
tests/test_visualisation.py:            if CUDA_AVAILABLE:
tests/test_visualisation.py:                raise ImportError("Optional loading of the CUDA module is broken")
swiftsimio/optional_packages.py:+ numba/cuda: visualisation
swiftsimio/optional_packages.py:# Numba/CUDA
swiftsimio/optional_packages.py:    from numba.cuda.cudadrv.error import CudaSupportError
swiftsimio/optional_packages.py:        import numba.cuda.cudadrv.driver as drv
swiftsimio/optional_packages.py:        from numba import cuda
swiftsimio/optional_packages.py:        from numba.cuda import jit as cuda_jit
swiftsimio/optional_packages.py:            CUDA_AVAILABLE = cuda.is_available()
swiftsimio/optional_packages.py:            CUDA_AVAILABLE = True
swiftsimio/optional_packages.py:    except CudaSupportError:
swiftsimio/optional_packages.py:        CUDA_AVAILABLE = False
swiftsimio/optional_packages.py:    # Mock the CudaSupportError so that we can raise it in cases
swiftsimio/optional_packages.py:    class CudaSupportError(Exception):
swiftsimio/optional_packages.py:    CUDA_AVAILABLE = False
swiftsimio/optional_packages.py:if not CUDA_AVAILABLE:
swiftsimio/optional_packages.py:    # Mock cuda-jit to prevent crashes
swiftsimio/optional_packages.py:    def cuda_jit(*args, **kwargs):
swiftsimio/optional_packages.py:    # For additional CUDA API access
swiftsimio/optional_packages.py:    cuda = None
swiftsimio/visualisation/projection_backends/gpu.py:    CUDA_AVAILABLE,
swiftsimio/visualisation/projection_backends/gpu.py:    cuda_jit,
swiftsimio/visualisation/projection_backends/gpu.py:    CudaSupportError,
swiftsimio/visualisation/projection_backends/gpu.py:    cuda,
swiftsimio/visualisation/projection_backends/gpu.py:@cuda_jit("float32(float32, float32)", device=True)
swiftsimio/visualisation/projection_backends/gpu.py:    This is the cuda-compiled version of the kernel, designed for use
swiftsimio/visualisation/projection_backends/gpu.py:    within the gpu backend. It has no double precision cousin.
swiftsimio/visualisation/projection_backends/gpu.py:@cuda_jit(
swiftsimio/visualisation/projection_backends/gpu.py:def scatter_gpu(
swiftsimio/visualisation/projection_backends/gpu.py:    for a performance improvement. This is the cuda version,
swiftsimio/visualisation/projection_backends/gpu.py:    GPU. Do not call this where cuda is not available (checks
swiftsimio/visualisation/projection_backends/gpu.py:    ``swiftsimio.optional_packages.CUDA_AVAILABLE``)
swiftsimio/visualisation/projection_backends/gpu.py:    i, dx, dy = cuda.grid(3)
swiftsimio/visualisation/projection_backends/gpu.py:                cuda.atomic.add(
swiftsimio/visualisation/projection_backends/gpu.py:                    cuda.atomic.add(img, (cell_x, cell_y), mass * kernel_eval)
swiftsimio/visualisation/projection_backends/gpu.py:    if not CUDA_AVAILABLE or cuda is None:
swiftsimio/visualisation/projection_backends/gpu.py:        raise CudaSupportError(
swiftsimio/visualisation/projection_backends/gpu.py:            "Unable to load the CUDA extension to numba. This function "
swiftsimio/visualisation/projection_backends/gpu.py:            "is only available on systems with supported GPUs."
swiftsimio/visualisation/projection_backends/gpu.py:    output = cuda.device_array((res, res), dtype=float32)
swiftsimio/visualisation/projection_backends/gpu.py:    scatter_gpu[blocks_per_grid, threads_per_block](x, y, m, h, box_x, box_y, output)
swiftsimio/visualisation/projection_backends/__init__.py:from swiftsimio.visualisation.projection_backends.gpu import scatter as gpu
swiftsimio/visualisation/projection_backends/__init__.py:from swiftsimio.visualisation.projection_backends.gpu import (
swiftsimio/visualisation/projection_backends/__init__.py:    scatter_parallel as gpu_parallel,
swiftsimio/visualisation/projection_backends/__init__.py:    "gpu": gpu,
swiftsimio/visualisation/projection_backends/__init__.py:    "gpu": gpu_parallel,
CHANGELOG.txt:+ Adds GPU-accelerated projection images (thanks @loikki!).

```
