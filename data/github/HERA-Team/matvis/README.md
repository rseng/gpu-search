# https://github.com/HERA-Team/matvis

```console
README.rst:Fast matrix-based visibility simulator capable of running on CPU and GPU.
README.rst:CPU and GPU implementations.
README.rst:* Supports both CPU and GPU implementations as drop-in replacements for each other.
README.rst:If you want to use the GPU functions, install
README.rst:with ``pip install matvis[gpu]``.
setup.cfg:description = Fast matrix-based visibility simulator with interface to CPU and GPU
setup.cfg:    matvis[gpu,profile,dev]
setup.cfg:gpu =
docs/api.rst:   matvis.gpu
docs/understanding_the_algorithm.rst:    2. The algorithm lends itself to implementation on GPUs, since the dominant parts
docs/understanding_the_algorithm.rst:       are fantastically fast on GPUs. Therefore, the ``matvis`` *algorithm* is seen
docs/understanding_the_algorithm.rst:       implementations: ``matvis_cpu`` and ``matvis_gpu``, which have the same API.
.coveragerc:    HAVE_GPU:
tests/test_cublas.py:from matvis.gpu import _cublas as cb
tests/test_cublas.py:    # agpu = cp.asarray(a, order='F')
tests/test_cublas.py:    # c = cp.cublas.gemm('N', 'H', agpu, agpu)#
tests/test_cpu_vs_gpu.py:"""Compare matvis CPU and GPU visibilities."""
tests/test_cpu_vs_gpu.py:pytest.importorskip("pycuda")
tests/test_cpu_vs_gpu.py:def test_cpu_vs_gpu(polarized, use_analytic_beam, precision, min_chunks, source_buffer):
tests/test_cpu_vs_gpu.py:    """Compare matvis CPU and GPU visibilities."""
tests/test_cpu_vs_gpu.py:    vis_cpu = simulate_vis(use_gpu=False, beam_spline_opts={"kx": 1, "ky": 1}, **kw)
tests/test_cpu_vs_gpu.py:    vis_gpu = simulate_vis(use_gpu=True, **kw)
tests/test_cpu_vs_gpu.py:    np.testing.assert_allclose(vis_gpu.real, vis_cpu.real, rtol=rtol, atol=atol)
tests/test_cpu_vs_gpu.py:    np.testing.assert_allclose(vis_gpu.imag, vis_cpu.imag, rtol=rtol, atol=atol)
tests/test_beam_interp_gpu.py:"""Test the GPU beam interpolation routine."""
tests/test_beam_interp_gpu.py:from matvis.gpu.beams import gpu_beam_interpolation, prepare_for_map_coords
tests/test_beam_interp_gpu.py:    new_beam = gpu_beam_interpolation(
tests/test_beam_interp_gpu.py:    new_beam = gpu_beam_interpolation(
tests/test_beam_interp_gpu.py:        gpu_beam_interpolation(
tests/test_beams.py:from matvis import HAVE_GPU
tests/test_beams.py:@pytest.mark.skipif(not HAVE_GPU, reason="GPU is not available")
tests/test_beams.py:class TestGPUBeamInterpolator:
tests/test_beams.py:    """Test the GPUBeamInterpolator."""
tests/test_beams.py:        """Import the GPUBeamInterpolator."""
tests/test_beams.py:        from matvis.gpu.beams import GPUBeamInterpolator
tests/test_beams.py:        self.gpuinterp = GPUBeamInterpolator
tests/test_beams.py:    @pytest.mark.skipif(not HAVE_GPU, reason="GPU is not available")
tests/test_beams.py:        bm = self.gpuinterp(
tests/test_beams.py:        bm = self.gpuinterp(
tests/test_beams.py:def test_gpu_beam_interp_against_cpu(efield_single_freq):
tests/test_beams.py:    """Test that GPU beam interpolation matches the CPU interpolation."""
tests/test_beams.py:    if not HAVE_GPU:
tests/test_beams.py:        pytest.skip("GPU is not available")
tests/test_beams.py:    from matvis.gpu.beams import GPUBeamInterpolator
tests/test_beams.py:    gpu_bmfunc = GPUBeamInterpolator(
tests/test_beams.py:        gpu_bmfunc.beam_list[0].data_array,
tests/test_beams.py:    gpu_bmfunc.setup()
tests/test_beams.py:    gpu_bmfunc(tx, ty)
tests/test_beams.py:        cpu_bmfunc.interpolated_beam, gpu_bmfunc.interpolated_beam.get(), atol=1e-6
tests/test_coordrot.py:from matvis import HAVE_GPU
tests/test_coordrot.py:if HAVE_GPU:
tests/test_coordrot.py:def get_random_coordrot(n, method, gpu, seed, precision=2, setup: bool = True, **kw):
tests/test_coordrot.py:        gpu=gpu,
tests/test_coordrot.py:@pytest.mark.parametrize("gpu", [False, True] if HAVE_GPU else [False])
tests/test_coordrot.py:def test_repeat_stays_same(method, gpu):
tests/test_coordrot.py:    if not gpu and method.requires_gpu:
tests/test_coordrot.py:    coords = get_random_coordrot(15, method, gpu, seed=35)
tests/test_coordrot.py:    xp = cp if gpu else np
tests/test_coordrot.py:@pytest.mark.parametrize("gpu", [False, True] if HAVE_GPU else [False])
tests/test_coordrot.py:def test_accuracy_against_astropy(method, gpu, precision):
tests/test_coordrot.py:    if not gpu and method.requires_gpu:
tests/test_coordrot.py:        1000, CoordinateRotationAstropy, gpu, seed=42, precision=precision
tests/test_coordrot.py:    coords = get_random_coordrot(1000, method, gpu, seed=42, precision=precision)
tests/test_coordrot.py:    if gpu:
tests/test_coordrot.py:        1000, CoordinateRotationERFA, gpu=False, seed=1, precision=1
tests/test_coordrot.py:        1000, CoordinateRotationERFA, gpu=False, seed=1, precision=1, setup=False
tests/test_coordrot.py:        10000, CoordinateRotationERFA, gpu=False, seed=1, precision=1, chunk_size=100
tests/test_coordrot.py:        10000, CoordinateRotationERFA, gpu=False, seed=1, precision=1, chunk_size=5000
tests/test_coordrot.py:        10000, CoordinateRotationERFA, gpu=False, seed=1, precision=1
tests/test_matvis_gpu.py:"""Tests of functionality of matvis_gpu."""
tests/test_matvis_gpu.py:        precision=2, use_gpu=True, beam_spline_opts={"kx": 1, "ky": 1}, **kw
tests/test_matvis_gpu.py:        "use_gpu": True,
tests/test_matvis_gpu.py:        ValueError, match="GPUBeamInterpolator only supports beam_lists with either"
tests/test_matvis_gpu.py:        simulate_vis(beams=cpu_beams, use_gpu=True, **kw)
tests/test_matvis_gpu.py:    """Test that using single precision on gpu works."""
tests/test_matvis_gpu.py:    kw |= {"use_gpu": True}
tests/test_matprod.py:    "method", ["CPUMatMul", "CPUVectorDot", "GPUMatMul", "GPUVectorDot"]
tests/test_matprod.py:    if method.startswith("GPU"):
tests/test_matprod.py:        from matvis.gpu import matprod as module
tests/test_matprod.py:    if method.startswith("GPU"):
tests/test_matprod.py:    if method.startswith("GPU"):
tests/__init__.py:        # comparing to the GPU interpolation, which first has to interpolate to a regular
src/matvis/coordinates.py:from . import HAVE_GPU
src/matvis/coordinates.py:if HAVE_GPU:
src/matvis/cpu/cpu.py:        for the CoordinateRotationERFA (and GPU version of the same) method, there
src/matvis/cpu/coords.py:        if self.gpu:
src/matvis/cpu/coords.py:            self.xp.cuda.Device().synchronize()
src/matvis/cpu/coords.py:        if self.gpu:
src/matvis/cpu/coords.py:            self.xp.cuda.Device().synchronize()
src/matvis/core/coords.py:    HAVE_CUDA = True
src/matvis/core/coords.py:    HAVE_CUDA = False
src/matvis/core/coords.py:    requires_gpu: bool = False
src/matvis/core/coords.py:        gpu: bool = False,
src/matvis/core/coords.py:        self.gpu = gpu
src/matvis/core/coords.py:        if self.gpu and not HAVE_CUDA:
src/matvis/core/coords.py:            raise ValueError("GPU requested but cupy not installed.")
src/matvis/core/coords.py:        self.xp = cp if self.gpu else np
src/matvis/core/coords.py:        if self.gpu:
src/matvis/core/coords.py:            self.xp.cuda.Device().synchronize()
src/matvis/core/getz.py:    HAVE_CUDA = True
src/matvis/core/getz.py:    HAVE_CUDA = False
src/matvis/core/getz.py:        self, nant: int, nfeed: int, nax: int, nsrc: int, ctype, gpu: bool = False
src/matvis/core/getz.py:        self.gpu = gpu
src/matvis/core/getz.py:        if gpu and not HAVE_CUDA:
src/matvis/core/getz.py:            raise ImportError("You need to install the [gpu] extra to use gpu!")
src/matvis/core/getz.py:        self.xp = cp if self.gpu else np
src/matvis/core/getz.py:        if self.gpu:
src/matvis/core/getz.py:            cp.cuda.Device().synchronize()
src/matvis/core/tau.py:    HAVE_CUDA = True
src/matvis/core/tau.py:    HAVE_CUDA = False
src/matvis/core/tau.py:        gpu: bool = False,
src/matvis/core/tau.py:        self.gpu = gpu
src/matvis/core/tau.py:        if gpu and not HAVE_CUDA:
src/matvis/core/tau.py:            raise ImportError("You need to install the [gpu] extra to use gpu!")
src/matvis/core/tau.py:        self._xp = cp if self.gpu else np
src/matvis/core/tau.py:        if self.gpu:
src/matvis/core/tau.py:            cp.cuda.Device().synchronize()
src/matvis/core/__init__.py:algorithm. These routines are then implemented in the ``cpu`` and ``gpu``
src/matvis/gpu_src/beam_interpolation.cu:#include <pycuda-helpers.hpp>
src/matvis/gpu_src/beam_interpolation.cu:// Runs on GPU only
src/matvis/gpu_src/beam_interpolation.cu:texture<fp_tex_{{ DTYPE }}, cudaTextureType3D, cudaReadModeElementType> bm_tex;
src/matvis/gpu_src/measurement_equation.cu:// CUDA code for computing "voltage" visibilities
src/matvis/gpu_src/measurement_equation.cu:// BLOCK_PX : # of sky pixels handled by one GPU block, used to size shared memory
src/matvis/gpu_src/measurement_equation.cu:#include <pycuda-helpers.hpp>
src/matvis/_utils.py:    HAVE_CUDA = True
src/matvis/_utils.py:    HAVE_CUDA = False
src/matvis/_utils.py:        loc = "GPU" if HAVE_CUDA and isinstance(x, cp.ndarray) else "CPU"
src/matvis/_utils.py:    gpusize = {"a": freemem}
src/matvis/_utils.py:    while sum(gpusize.values()) >= freemem and ch < 100:
src/matvis/_utils.py:        gpusize = {
src/matvis/_utils.py:            f"nchunks={ch}. Array Sizes (bytes)={gpusize}. Total={sum(gpusize.values())}"
src/matvis/_utils.py:        f"(estimate {sum(gpusize.values()) / 1024**3:.2f} GB)"
src/matvis/cli.py:from matvis import DATA_PATH, HAVE_GPU, coordinates, cpu, simulate_vis
src/matvis/cli.py:if HAVE_GPU:
src/matvis/cli.py:    from matvis import gpu
src/matvis/cli.py:if HAVE_GPU:
src/matvis/cli.py:    simgpu = gpu.simulate
src/matvis/cli.py:        "naz{naz}_nza{nza}_g{gpu}_pr{precision}_{matprod_method}_{coord_method}"
src/matvis/cli.py:    gpu,
src/matvis/cli.py:    if not HAVE_GPU and gpu:
src/matvis/cli.py:        raise RuntimeError("Cannot run GPU version without GPU dependencies installed!")
src/matvis/cli.py:    cns.print(f"  GPU:              {gpu:>7}")
src/matvis/cli.py:    if gpu:
src/matvis/cli.py:        profiler.add_function(simgpu)
src/matvis/cli.py:        use_gpu=gpu,
src/matvis/cli.py:        matprod_method=f"{'GPU' if gpu else 'CPU'}{matprod_method}",
src/matvis/cli.py:        gpu=gpu,
src/matvis/cli.py:        "--gpu/--cpu",
src/matvis/wrapper.py:from . import HAVE_GPU, cpu
src/matvis/wrapper.py:if HAVE_GPU:
src/matvis/wrapper.py:    from . import gpu
src/matvis/wrapper.py:    use_gpu: bool = False,
src/matvis/wrapper.py:    if use_gpu:
src/matvis/wrapper.py:        if not HAVE_GPU:
src/matvis/wrapper.py:            raise ImportError("You cannot use GPU without installing GPU-dependencies!")
src/matvis/wrapper.py:        device = cp.cuda.Device()
src/matvis/wrapper.py:            Your GPU has the following attributes:
src/matvis/wrapper.py:    fnc = gpu.simulate if use_gpu else cpu.simulate
src/matvis/wrapper.py:    # Loop over frequencies and call matvis_cpu/gpu
src/matvis/gpu/gpu.py:"""GPU implementation of the simulator."""
src/matvis/gpu/gpu.py:    HAVE_CUDA = True
src/matvis/gpu/gpu.py:    HAVE_CUDA = False
src/matvis/gpu/gpu.py:    # warn, but default back to non-gpu functionality
src/matvis/gpu/gpu.py:    HAVE_CUDA = False
src/matvis/gpu/gpu.py:        "GPUCoordinateRotationERFA",
src/matvis/gpu/gpu.py:    matprod_method: Literal["GPUMatMul", "GPUVectorLoop"] = "GPUMatMul",
src/matvis/gpu/gpu.py:    """GPU implementation of the visibility simulator."""
src/matvis/gpu/gpu.py:    if not HAVE_CUDA:
src/matvis/gpu/gpu.py:        raise ImportError("You need to install the [gpu] extra to use this function!")
src/matvis/gpu/gpu.py:        min(max_memory, cp.cuda.Device().mem_info[0]),
src/matvis/gpu/gpu.py:    bmfunc = beams.GPUBeamInterpolator(
src/matvis/gpu/gpu.py:        gpu=True,
src/matvis/gpu/gpu.py:        nsrc=nsrc_alloc, nfeed=nfeed, nant=nant, nax=nax, ctype=ctype, gpu=True
src/matvis/gpu/gpu.py:        antpos=antpos, freq=freq, precision=precision, nsrc=nsrc_alloc, gpu=True
src/matvis/gpu/gpu.py:    logger.debug("Starting GPU allocations...")
src/matvis/gpu/gpu.py:    init_mem = cp.cuda.Device().mem_info[0]
src/matvis/gpu/gpu.py:    logger.debug(f"Before GPU allocations, GPU mem avail is: {init_mem / 1024**3} GB")
src/matvis/gpu/gpu.py:    memnow = cp.cuda.Device().mem_info[0]
src/matvis/gpu/gpu.py:    logger.debug(f"After antpos, GPU mem avail is: {memnow / 1024**3} GB.")
src/matvis/gpu/gpu.py:    memnow = cp.cuda.Device().mem_info[0]
src/matvis/gpu/gpu.py:        logger.debug(f"After bmfunc, GPU mem avail is: {memnow / 1024**3} GB.")
src/matvis/gpu/gpu.py:    memnow = cp.cuda.Device().mem_info[0]
src/matvis/gpu/gpu.py:    logger.debug(f"After coords, GPU mem avail is: {memnow / 1024**3} GB.")
src/matvis/gpu/gpu.py:    memnow = cp.cuda.Device().mem_info[0]
src/matvis/gpu/gpu.py:    logger.debug(f"After zcalc, GPU mem avail is: {memnow / 1024**3} GB.")
src/matvis/gpu/gpu.py:    memnow = cp.cuda.Device().mem_info[0]
src/matvis/gpu/gpu.py:    logger.debug(f"After matprod, GPU mem avail is: {memnow / 1024**3} GB.")
src/matvis/gpu/gpu.py:    streams = [cp.cuda.Stream() for _ in range(nchunks)]
src/matvis/gpu/gpu.py:        events = [{e: cp.cuda.Event() for e in event_order} for _ in range(nchunks)]
src/matvis/gpu/gpu.py:                f"After coords, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
src/matvis/gpu/gpu.py:                f"After beam, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
src/matvis/gpu/gpu.py:                f"After exptau, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
src/matvis/gpu/gpu.py:                f"After Z, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
src/matvis/gpu/gpu.py:                f"After matprod, GPU mem: {cp.cuda.Device().mem_info[0] / 1024**3} GB."
src/matvis/gpu/beams.py:"""GPU beam interpolation routines."""
src/matvis/gpu/beams.py:class GPUBeamInterpolator(BeamInterpolator):
src/matvis/gpu/beams.py:    """Interpolate a UVBeam object on the GPU.
src/matvis/gpu/beams.py:                "GPUBeamInterpolator only supports beam_lists with either all UVBeam or all AnalyticBeam objects."
src/matvis/gpu/beams.py:            # grid, then use linear interpolation on the GPU with that high-res grid.
src/matvis/gpu/beams.py:        """Evaluate the beam on the GPU.
src/matvis/gpu/beams.py:        """Perform the beam interpolation, choosing between CPU and GPU as necessary."""
src/matvis/gpu/beams.py:        gpu_beam_interpolation(
src/matvis/gpu/beams.py:def gpu_beam_interpolation(
src/matvis/gpu/beams.py:    Interpolate beam values from a regular az/za grid using GPU.
src/matvis/gpu/beams.py:    cp.cuda.Device().synchronize()
src/matvis/gpu/matprod.py:"""GPU-accelerated source-summing operation."""
src/matvis/gpu/matprod.py:class GPUMatMul(MatProd):
src/matvis/gpu/matprod.py:        cp.cuda.Device().synchronize()
src/matvis/gpu/matprod.py:        cp.cuda.Device().synchronize()
src/matvis/gpu/matprod.py:class GPUVectorDot(MatProd):
src/matvis/gpu/matprod.py:        cp.cuda.Device().synchronize()
src/matvis/gpu/matprod.py:        cp.cuda.Device().synchronize()
src/matvis/gpu/_cublas.py:from cupy.cuda import device
src/matvis/gpu/_cublas.py:from cupy_backends.cuda.libs import cublas
src/matvis/gpu/coords.py:"""Coordinate rotation methods for the GPU."""
src/matvis/gpu/coords.py:class GPUCoordinateRotationERFA(CoordinateRotationERFA):
src/matvis/gpu/coords.py:    over-rides the light-deflection function to be computed as a custom CUDA kernel.
src/matvis/gpu/coords.py:    All other methods in the super-class are compatible both with GPU and CPU
src/matvis/gpu/coords.py:    requires_gpu: bool = True
src/matvis/gpu/__init__.py:"""GPU-accelerated matvis implementation."""
src/matvis/gpu/__init__.py:from .gpu import simulate
src/matvis/__init__.py:    HAVE_GPU = True
src/matvis/__init__.py:    HAVE_GPU = False
src/matvis/__init__.py:if HAVE_GPU:
src/matvis/__init__.py:    from . import gpu
CHANGELOG.rst:- Better handling of errors when GPUs are present but currently unavailable for some
CHANGELOG.rst:- When getting the raw beam data for GPU, there was a check for whether the beam covers
CHANGELOG.rst:Version 1.0 is a major update that brings the GPU implementation up to the same API
CHANGELOG.rst:- Support for ``bm_pix`` and ``use_pixel_beams`` (in both CPU and GPU implementations).
CHANGELOG.rst:  methods in ``UVBeam``, or via new GPU methods. If you input an ``AnalyticBeam``, the
CHANGELOG.rst:- Polarization support for GPU implementation.
CHANGELOG.rst:- **BREAKING CHANGE:** the output from the CPU and GPU implementations has changed
CHANGELOG.rst:- ``vis_cpu`` and ``vis_gpu`` *modules* renamed to ``cpu`` and ``gpu`` respectively, to
CHANGELOG.rst:- New more comprehensive tests comparing the GPU and CPU implementations against
CHANGELOG.rst:- Ability to do **polarization**! (Only in ``vis_cpu`` for now, not GPU).
CHANGELOG.rst:- Installation of gpu extras fixed.
CHANGELOG.rst:- Fix import logic for GPU.

```
