# https://github.com/ben-cassese/squishyplanet

```console
docs/installation.md:## GPU Users
docs/installation.md:Since ``squishyplanet`` is written entirely in ``JAX``, it can technically run on a GPU (or a TPU) as well as a CPU with no changes to the code. However, anyone attempting to do this will likely be dissapointed with the performance, since in its current state  ``squishyplanet`` is not optimized for GPU use. Many of the operations are run  sequentially to save on memory and it was entirely developed on a CPU.
docs/installation.md:If you are interested in running ``squishyplanet`` on a GPU, be sure you first follow the instructions for installing ``jax`` and ``jaxlib`` on your specific system, then install ``squishyplanet`` as normal. If you run into any issues, or even better if you're interested in helping to optimize ``squishyplanet`` for GPU use, please open an issue on the GitHub repository.

```
