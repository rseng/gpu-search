# https://github.com/thomaspinder/GPJax

```console
static/paper.bib:  title   = {{G{P}y{T}orch}: Blackbox matrix-matrix {G}aussian process inference with {GPU} acceleration},
static/paper.md:Gaussian processes [GPs, @rasmussen2006gaussian] are Bayesian nonparametric models that have been successfully used in applications such as geostatistics [@matheron1963principles], Bayesian optimisation [@mockus1978application], and reinforcement learning [@deisenroth2011pilco]. `GPJax` is a didactic GP library targeted at researchers who wish to develop novel GP methodology. The scope of `GPJax` is to provide users with a set of composable objects for constructing GP models that closely resemble the underlying maths that one would write on paper. Furthermore, by the virtue of being written in JAX [@jax2018github], `GPJax` natively supports CPUs, GPUs and TPUs through efficient compilation to XLA, automatic differentiation and vectorised operations. Consequently, `GPJax` provides a modern GP package that can effortlessly be tailored, extended and interleaved with other libraries to meet the individual needs of researchers and scientists.
docs/installation.md:## GPU/TPU support
docs/installation.md:Fancy using GPJax on GPU/TPU? Then you'll need to install JAX with the relevant
docs/index.md:GPJax is a didactic Gaussian process (GP) library in JAX, supporting GPU

```
