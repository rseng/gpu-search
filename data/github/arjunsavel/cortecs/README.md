# https://github.com/arjunsavel/cortecs

```console
paper/paper.bib:  title={3D radiative transfer for exoplanet atmospheres. gCMCRT: a GPU-accelerated MCRT code},
paper/paper.bib:  title={ooc\_cuDNN: Accommodating convolutional neural networks over GPU memory capacity},
paper/paper.md:GPU-friendly methods. The package is actively developed on GitHub (<https://github.com/arjunsavel/cortecs>), and it is
paper/paper.md:some codes have parallelized the problem on GPUs [e.g., @lee:2022; @line:2021 ]. However, GPUs cannot in general hold large amounts of
paper/paper.md:data in their video random-access memory (VRAM) [e.g., @ito:2017]; only the cutting-edge, most expensive GPUs are equipped with VRAM in excess of 30 GB
paper/paper.md:(such as the NVIDIA A100 or H100). RAM and VRAM management is therefore a clear concern when producing
paper/paper.md:for GPUs and are accelerated with the `JAX` code transformation framework [@jax:2018]. An example of this reconstruction
src/cortecs/eval/eval_pca.py:    Unfortunately, not all GPUs will support a simpler dot product, I believe,

```
