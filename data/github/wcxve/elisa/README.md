# https://github.com/wcxve/elisa

```console
src/elisa/util/config.py:def set_jax_platform(platform: Literal['cpu', 'gpu', 'tpu'] | None = None):
src/elisa/util/config.py:    """Set JAX platform to CPU, GPU, or TPU.
src/elisa/util/config.py:    platform : {'cpu', 'gpu', 'tpu'}, optional
src/elisa/util/config.py:        Either ``'cpu'``, ``'gpu'``, or ``'tpu'``.
src/elisa/util/config.py:    assert platform in {'cpu', 'gpu', 'tpu', None}
src/elisa/util/config.py:    if platform == 'gpu':
src/elisa/util/config.py:        # see https://jax.readthedocs.io/en/latest/gpu_performance_tips.html
src/elisa/util/config.py:        xla_gpu_flags = (
src/elisa/util/config.py:            '--xla_gpu_enable_triton_softmax_fusion=true '
src/elisa/util/config.py:            '--xla_gpu_triton_gemm_any=True '
src/elisa/util/config.py:            '--xla_gpu_enable_async_collectives=true '
src/elisa/util/config.py:            '--xla_gpu_enable_latency_hiding_scheduler=true '
src/elisa/util/config.py:            '--xla_gpu_enable_highest_priority_async_stream=true'
src/elisa/util/config.py:        if xla_gpu_flags not in xla_flags:
src/elisa/util/config.py:            os.environ['XLA_FLAGS'] = f'{xla_flags} {xla_gpu_flags}'

```
