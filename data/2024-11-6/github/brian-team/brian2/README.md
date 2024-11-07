# https://github.com/brian-team/brian2

```console
brian2/devices/cpp_standalone/device.py:        # Allow setting `profile` in the `set_device` call (used e.g. in brian2cuda
docs_sphinx/introduction/release_notes.rst:of Brian2 and use Nvidia's ``nvcc`` compiler. The release also fixes a minor string-formatting error (:issue:`1377`),
docs_sphinx/introduction/release_notes.rst:  with GPU code generation targets.
docs_sphinx/introduction/brian1_to_2/preferences.rst:* ``brianhears_usegpu``: removed because Brian Hears doesn't exist in Brian 2.
docs_sphinx/user/computation.rst:e.g. `here for Brian2CUDA benchmarks <https://github.com/brian-team/brian2cuda/blob/835c978ad758bc0621e34344c1fb7b811ef8a118/brian2cuda/tests/features/cuda_configuration.py#L148-L156>`_ or `here for Brian2GeNN benchmarks <https://github.com/brian-team/brian2genn_benchmarks/blob/6d1a6d9d97c05653cec2e413c9fd312cfe13e15c/benchmark_utils.py#L78-L136>`_.
docs_sphinx/advanced/state_update.rst:a big issue in C, GPU or even with Numba.
docs_sphinx/developer/guidelines/logging.rst:Extension packages such as `brian2cuda <https://brian2cuda.readthedocs.io>`_ can use Brian's logging infrastructure by
docs_sphinx/developer/guidelines/logging.rst:e.g. a name starting with ``brian2cuda.`` so that it is clear whether a log message comes from Brian or from an

```
