# https://github.com/blackjax-devs/blackjax

```console
docs/examples/howto_sample_multiple_chains.md:Sampling with a few chains has become ubiquitous in modern probabilistic programming because it allows to compute better convergence diagnostics such as $\hat{R}$. More recently a new trend has emerged where researchers try to sample with thousands of chains for only a few steps. Whatever your use case is, Blackjax has you covered: thanks to JAX's primitives you will be able to run multiple chains on CPU, GPU or TPU.
docs/examples/howto_sample_multiple_chains.md:- [jax.vmap](https://jax.readthedocs.io/en/latest/jax.html?highlight=vmap#jax.vmap) is used to SIMD vectorize `JAX` code. It is important to remember that vectorization happens at the *instruction level*, each CPU or GPU  instruction will the process the information from your different chains, *one intructions at a time*. This can have some unexpected consequences;
docs/examples/howto_sample_multiple_chains.md:- [jax.pmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html#jax.pmap) is a higher level abstraction, where processes are split across multiple devices: GPUs, TPUs, or CPU cores.
docs/examples/howto_sample_multiple_chains.md:Remember when we said SIMD vectorization happens at the instruction level? At each step, the NUTS sampler can perform from 1 to 1024 integration steps, and the CPU (GPU) has to wait for all the chains to complete before moving on to the next chain. As a result, each step is as long as the slowest chain.
docs/index.md:Blackjax is a library of samplers for [JAX](https://github.com/google/jax) that works on CPU as well as GPU. It is designed with two categories of users in mind:
docs/index.md::::{admonition} GPU instructions
docs/index.md:run on CPU only. **If you want to use BlackJAX on GPU/TPU** we recommend you follow
blackjax/smc/resampling.py:    # O(N) loop as our code is meant to work on GPU where searchsorted is
README.md:works on CPU as well as GPU.
README.md:- Want to sample on GPU;
README.md:run on CPU only. **If you want to use BlackJAX on GPU/TPU** we recommend you follow

```
