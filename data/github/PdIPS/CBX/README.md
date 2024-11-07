# https://github.com/PdIPS/CBX

```console
paper.md:Most of the [CBXPy](https://pdips.github.io/CBXpy/) implementation uses basic Python functionality, and the agents are handled as an array-like structure. For certain specific features, like broadcasting-behaviour, array copying, and index selection, we fall back to the `numpy` implementation [@harris2020array]. However, it should be noted that an adaptation to other array or tensor libraries like PyTorch [@paszke2019pytorch] is straightforward. Compatibility with the latter enables gradient-free deep learning directly on the GPU, as demonstrated in the documentation.\

```
