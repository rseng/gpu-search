# https://github.com/darthoctopus/reggae

```console
docs/installation.md:- What do I need to install in order to accelerate these calculations on my GPU?
docs/installation.md:Please check that your OS is on `jax`'s supported list of systems and hardware configurations, which can be found [here](https://jax.readthedocs.io/en/latest/installation.html#supported-platforms). In summary, while performing `jax` calculations on the CPU is supported on all platforms, GPU acceleration is only available on limited combinations of hardware and operating systems. Our `requirements.txt` file assumes only CPU support via `jaxlib`; Windows users may need additional external requirements for it to work.
reggae/reggae/__init__.py:        slow because there is no GPU implementation of `eig` and it's just a generally
reggae/reggae/__init__.py:        inefficient way of doing it. Future implementations should wrap cuda primitives.

```
