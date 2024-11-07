# https://github.com/kazewong/flowMC

```console
docs/quickstart.md:JAX does not install GPU support by default.
docs/quickstart.md:If you want to use GPU with JAX, you need to install JAX with GPU support according to [their document](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier).
docs/quickstart.md:At the time of writing this documentation page, this is the command to install JAX with GPU support:
docs/quickstart.md:pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
docs/quickstart.md:2. JAX uses [XLA](https://www.tensorflow.org/xla) to compile your code not only into machine code but also in a way that is more optimized for accelerators such as GPUs and TPUs. Having multiple MCMC chains helps speed up the training of the normalizing flow. Accelerators such as GPUs and TPUs provide parallel computing solutions that are more scalable compared to CPUs.
docs/index.md:- Native support for GPU acceleration.
joss/paper.md:- Use of accelerators such as GPUs and TPUs are natively supported. The code also supports the use of multiple accelerators with SIMD parallelism.
joss/paper.md:Despite the growing interest for these methods few accessible implementations for non-experts already exist and none of them propose GPU and TPU. Namely, a version of the NeuTra sampler `[@Hoffman2019]` available in Pyro `[@bingham2019pyro]` and the PocoMC package `[@Karamanis2022]` are both CPU bounded.
joss/paper.md:Modern accelerators such as GPU and TPU are designed to execute dense computation in parallel.
README.md:- Native support for GPU acceleration.
README.md:Jax does not install GPU support by default.
README.md:If you want to use GPU with Jax, you need to install Jax with GPU support according to their document.
README.md:At the time of writing this documentation page, this is the command to install Jax with GPU support:
README.md:pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
example/non_jax_likelihood.py:4. Your code won't run on GPU.
src/flowMC/nfmodel/rqSpline.py:    # binary search, but this is more GPU/TPU friendly.
src/flowMC/nfmodel/rqSpline.py:    # binary search, but this is more GPU/TPU friendly.

```
