# https://github.com/pymc-devs/pymc3

```console
pymc/sampling/jax.py:    backend: Literal["cpu", "gpu"] | None = None,
pymc/sampling/jax.py:    postprocessing_backend: Literal["cpu", "gpu"] | None = None,
pymc/sampling/jax.py:    postprocessing_backend: Optional[Literal["cpu", "gpu"]], default None,
pymc/sampling/jax.py:        Specify how postprocessing should be computed. gpu or cpu
pymc/sampling/jax.py:    postprocessing_backend: Literal["cpu", "gpu"] | None = None,
pymc/sampling/jax.py:    postprocessing_backend : Optional[Literal["cpu", "gpu"]], default None,
pymc/sampling/jax.py:        Specify how postprocessing should be computed. gpu or cpu
docs/source/learn/consulting.md:* Model speed-ups (reparameterizations, JAX, [GPU sampling](https://www.pymc-labs.io/blog-posts/pymc-stan-benchmark/))
docs/source/contributing/developer_guide.md:That is the reason we often see no advantage in using GPU, because the data is copying between GPU and CPU at each function call - and for a small model, the result is a slower inference under GPU than CPU.
docs/source/contributing/developer_guide.md:While having the samplers be written in Python allows for a lot of flexibility and intuitive for experiment (writing e.g. NUTS in PyTensor is also very difficult), it comes at a performance penalty and makes sampling on the GPU very inefficient because memory needs to be copied for every logp evaluation.
tests/test_math.py:        pytensor.config.device in ["cuda", "gpu"],
tests/test_math.py:        reason="No logDet implementation on GPU.",
RELEASE-NOTES.md:  - ⚠️ Support for JAX and JAX samplers, also allows sampling on GPUs. [This benchmark](https://www.pymc-labs.io/blog-posts/pymc-stan-benchmark/) shows speed-ups of up to 11x.
RELEASE-NOTES.md:* Improved support for theano's floatX setting to enable GPU computations (work in progress).
RELEASE-NOTES.md:* Use theano Psi and GammaLn functions to enable GPU support for them.

```
