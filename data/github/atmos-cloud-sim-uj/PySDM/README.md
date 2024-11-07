# https://github.com/atmos-cloud-sim-uj/PySDM

```console
setup.py:    keywords="physics-simulation, monte-carlo-simulation, gpu-computing,"
tests/unit_tests/test_imports.py:    "backends.GPU",
tests/unit_tests/products/test_ambient_relative_humidity.py:from PySDM.backends import CPU, GPU
tests/unit_tests/products/test_ambient_relative_humidity.py:    "backend_class", (CPU, pytest.param(GPU, marks=pytest.mark.xfail(strict=True)))
tests/unit_tests/dynamics/condensation/test_parcel_sanity_checks.py:from PySDM.backends import CPU, GPU
tests/unit_tests/dynamics/condensation/test_parcel_sanity_checks.py:                GPU, marks=pytest.mark.xfail(strict=True)
tests/unit_tests/dynamics/condensation/test_parcel_sanity_checks.py:            ),  # TODO #1117 (works with CUDA!)
tests/unit_tests/backends/storage/test_setitem.py:from PySDM.backends import CPU, GPU
tests/unit_tests/backends/storage/test_setitem.py:    "backend", (pytest.param(GPU, marks=pytest.mark.xfail(strict=True)), CPU)
tests/unit_tests/backends/test_collisions_methods.py:from PySDM.backends import CPU, GPU
tests/unit_tests/backends/test_collisions_methods.py:        ((CPU, "counting_sort"), (CPU, "counting_sort_parallel"), (GPU, "default")),
tests/unit_tests/backends/test_ctor_defaults.py:from PySDM.backends import CPU, GPU
tests/unit_tests/backends/test_ctor_defaults.py:    def test_gpu_ctor_defaults():
tests/unit_tests/backends/test_ctor_defaults.py:        signature = inspect.signature(GPU.__init__)
tests/unit_tests/impl/test_particle_attributes.py:from PySDM.backends import CPU, GPU, ThrustRTC
tests/unit_tests/impl/test_particle_attributes.py:        "backend_cls", (CPU, pytest.param(GPU, marks=pytest.mark.xfail(strict=True)))
tests/unit_tests/impl/test_particle_attributes.py:        if backend_class is GPU:
tests/unit_tests/impl/test_particle_attributes.py:        if backend_class is GPU:
tests/unit_tests/conftest.py:from PySDM.backends import CPU, GPU
tests/unit_tests/conftest.py:@pytest.fixture(params=(CPU, GPU))
tests/unit_tests/conftest.py:    params=(pytest.param(CPU(), id="CPU"), pytest.param(GPU(), id="GPU")),
tests/smoke_tests/box/srivastava_1982/test_eq_13.py:    title, settings, plot=False  # TODO #987 (backend_class: CPU, GPU)
tests/smoke_tests/box/srivastava_1982/test_eq_10.py:):  # TODO #987 (backend_class: CPU, GPU)
tests/smoke_tests/box/dejong_and_mackay_et_al_2023/test_fig_8.py:from PySDM.backends import CPU, GPU
tests/smoke_tests/box/dejong_and_mackay_et_al_2023/test_fig_8.py:    (CPU, pytest.param(GPU, marks=pytest.mark.xfail(strict=True))),  # TODO #987
tests/smoke_tests/box/dejong_and_mackay_et_al_2023/test_fig_6.py:from PySDM.backends import CPU, GPU
tests/smoke_tests/box/dejong_and_mackay_et_al_2023/test_fig_6.py:    (CPU, pytest.param(GPU, marks=pytest.mark.xfail(strict=True))),  # TODO #987
tests/smoke_tests/box/dejong_and_mackay_et_al_2023/test_fig_7.py:from PySDM.backends import CPU, GPU
tests/smoke_tests/box/dejong_and_mackay_et_al_2023/test_fig_7.py:        (CPU, pytest.param(GPU, marks=pytest.mark.xfail(strict=True))),  # TODO #987
tests/smoke_tests/parcel_b/yang_et_al_2018/test_just_do_it.py:from PySDM.backends import CPU, GPU
tests/smoke_tests/parcel_b/yang_et_al_2018/test_just_do_it.py:    if scheme == "SciPy" and (not adaptive or backend_class is GPU):
tests/smoke_tests/parcel_b/yang_et_al_2018/test_just_do_it.py:    if backend_class is not GPU:
tests/smoke_tests/parcel_b/arabas_and_shima_2017/test_vs_scipy.py:schemes = ("CPU", "SciPy")  # ,'GPU')  # TODO #588
tests/smoke_tests/parcel_b/arabas_and_shima_2017/test_vs_scipy.py:@pytest.mark.parametrize("scheme", ("CPU",))  # 'GPU'))  # TODO #588
tests/smoke_tests/parcel_b/arabas_and_shima_2017/test_conservation.py:from PySDM.backends import CPU, GPU
tests/smoke_tests/parcel_b/arabas_and_shima_2017/test_conservation.py:@pytest.mark.parametrize("scheme", ("SciPy", "CPU", "GPU"))
tests/smoke_tests/parcel_b/arabas_and_shima_2017/test_conservation.py:    assert scheme in ("SciPy", "CPU", "GPU")
tests/smoke_tests/parcel_b/arabas_and_shima_2017/test_conservation.py:    simulation = Simulation(settings, GPU if scheme == "GPU" else CPU)
README.md:[![CUDA](https://img.shields.io/static/v1?label=CUDA&logo=nVidia&color=87ce3e&message=ThrustRTC)](https://pypi.org/project/ThrustRTC/)
README.md:  and GPU-resident backend built on top of [ThrustRTC](https://pypi.org/project/ThrustRTC/).
README.md:The [`ThrustRTC`](https://open-atmos.github.io/PySDM/PySDM/backends/thrust_rtc.html) backend (aliased ``GPU``) offers GPU-resident operation of PySDM
README.md:Using the ``GPU`` backend requires nVidia hardware and [CUDA driver](https://developer.nvidia.com/cuda-downloads).
README.md:- nvidia: [cudatoolkit](https://anaconda.org/nvidia/cudatoolkit)
README.md:      GPU
README.md:The ``backend`` argument may be set to ``CPU`` or ``GPU``
README.md:  GPU-resident computation mode, respectively.
README.md:[![CUDA](https://img.shields.io/static/v1?label=+&logo=nVidia&color=darkgreen&message=ThrustRTC/CUDA)](https://pypi.org/project/ThrustRTC/)
paper/paper.bib:  title = {ThrustRTC: CUDA tool set for non-{C}++ languages that provides similar functionality like {T}hrust, with NVRTC at its core},
paper/paper.bib:  title = {On the design of {M}onte-{C}arlo particle coagulation solver interface: a CPU/GPU Super-Droplet Method case study with PySDM},
paper/paperv1.md:  - gpu-computing 
paper/paperv1.md:`PySDM` has two alternative parallel number-crunching backends available: multi-threaded CPU backend based on `Numba` [@Numba] and GPU-resident backend built on top of `ThrustRTC` [@ThrustRTC].
paper/paperv1.md:The optional GPU backend relies on proprietary vendor-specific CUDA technology, the accompanying non-free software and drivers; `ThrustRTC` and `CURandRTC` packages are released under the Anti-996 license.
paper/paperv1.md:For the GPU backend, a purpose-built `FakeThrust` class is shipped with `PySDM` which implements a subset of the `ThrustRTC` API 
paper/paperv1.md:The `backend` argument may be set to an instance of either `CPU` or `GPU` what translates to choosing the multi-threaded `Numba`-based backend or the `ThrustRTC-based` GPU-resident computation mode, respectively. 
paper/paperv1.md:    \item[availability of tools for modern hardware]{depicted in PySDM with the GPU backend}.
paper/paperv1.md:KG and BP contributed to the GPU backend.
paper/paper.md:  of the CPU and GPU backend code.
paper/paper.md:KD contributed to setting up continuous integration workflows for the GPU backend. 
paper/paper.md:OB implemented breakup handling within the GPU backend and contributed code refactors and new tests for both CPU and GPU backends.
PySDM/formulae.py:Logic for enabling common CPU/GPU physics formulae code
PySDM/backends/thrust_rtc.py:GPU-resident backend using NVRTC runtime compilation library for CUDA
PySDM/backends/thrust_rtc.py:            warnings.warn("CUDA is not available, using FakeThrustRTC!")
PySDM/backends/impl_thrust_rtc/test_helpers/fake_thrust_rtc.py: testing ThrustRTC code on machines with no GPU/CUDA
PySDM/backends/impl_thrust_rtc/test_helpers/flag.py: (for tests of GPU code on machines with no GPU)
PySDM/backends/impl_thrust_rtc/bisection.py:C code of basic bisection root-finding algorithm for use within ThrustRTC CUDA codes
PySDM/backends/impl_thrust_rtc/methods/pair_methods.py:GPU implementation of pairwise operations backend methods
PySDM/backends/impl_thrust_rtc/methods/displacement_methods.py:GPU implementation of backend methods for particle displacement (advection and sedimentation)
PySDM/backends/impl_thrust_rtc/methods/terminal_velocity_methods.py:GPU implementation of backend methods for terminal velocities
PySDM/backends/impl_thrust_rtc/methods/collisions_methods.py:GPU implementation of backend methods for particle collisions
PySDM/backends/impl_thrust_rtc/methods/index_methods.py:GPU implementation of shuffling and sorting backend methods
PySDM/backends/impl_thrust_rtc/methods/physics_methods.py:GPU implementation of backend methods wrapping basic physics formulae
PySDM/backends/impl_thrust_rtc/methods/isotope_methods.py:GPU implementation of isotope-relates backend methods
PySDM/backends/impl_thrust_rtc/methods/freezing_methods.py:GPU implementation of backend methods for freezing (singular and time-dependent immersion freezing)
PySDM/backends/impl_thrust_rtc/methods/moments_methods.py:GPU implementation of moment calculation backend methods
PySDM/backends/impl_thrust_rtc/methods/moments_methods.py:# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
PySDM/backends/impl_thrust_rtc/methods/__init__.py:""" method classes of the GPU backend """
PySDM/backends/impl_thrust_rtc/methods/condensation_methods.py:GPU implementation of backend methods for water condensation/evaporation
PySDM/backends/impl_thrust_rtc/conf.py:}  # TODO #1120: move to GPU backend ctor
PySDM/backends/impl_thrust_rtc/__init__.py:""" the guts of the GPU backend """
PySDM/backends/__init__.py:and GPU=`PySDM.backends.thrust_rtc.ThrustRTC`
PySDM/backends/__init__.py:from numba import cuda
PySDM/backends/__init__.py:def _cuda_is_available():
PySDM/backends/__init__.py:    lib_names = ("libcuda.so", "libcuda.dylib", "cuda.dll")
PySDM/backends/__init__.py:            cuda_lib = ctypes.CDLL(libname)
PySDM/backends/__init__.py:    result = cuda_lib.cuInit(0)
PySDM/backends/__init__.py:    if result != 0:  # cuda.h: CUDA_SUCCESS = 0
PySDM/backends/__init__.py:        cuda_lib.cuGetErrorString(result, ctypes.byref(error_str))
PySDM/backends/__init__.py:            f"CUDA library found but cuInit() failed (error code: {result};"
PySDM/backends/__init__.py:                "to use GPU on Colab set hardware accelerator to 'GPU' before session start"
PySDM/backends/__init__.py:if "CI" not in os.environ and (_cuda_is_available() or cuda.is_available()):
PySDM/backends/__init__.py:GPU = ThrustRTC
PySDM/physics/__init__.py:    in the same way on both CPU and GPU backends (yes, please use `const.ONE/const.TWO`
examples/PySDM_examples/Bulenok_2023_MasterThesis/utils.py:from PySDM.backends import CPU, GPU
examples/PySDM_examples/Bulenok_2023_MasterThesis/utils.py:    backends=(CPU, GPU),
examples/PySDM_examples/Bulenok_2023_MasterThesis/utils.py:    if GPU in backends:
examples/PySDM_examples/Bulenok_2023_MasterThesis/utils.py:        backend_configs.append((GPU, None))
examples/PySDM_examples/Arabas_et_al_2015/example_benchmark.py:from PySDM.backends import CPU, GPU
examples/PySDM_examples/Arabas_et_al_2015/example_benchmark.py:    if GPU.ENABLE:
examples/PySDM_examples/Arabas_et_al_2015/example_benchmark.py:        backends.append((GPU, "async"))
examples/PySDM_examples/Bartman_2020_MasterThesis/fig_5_SCIPY_VS_ADAPTIVE.py:from PySDM.backends import CPU, GPU
examples/PySDM_examples/Bartman_2020_MasterThesis/fig_5_SCIPY_VS_ADAPTIVE.py:                        settings, backend=CPU if scheme == "CPU" else GPU

```
