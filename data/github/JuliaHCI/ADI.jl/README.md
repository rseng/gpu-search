# https://github.com/JuliaHCI/ADI.jl

```console
paper/paper.bib:% primary paper, detailing the GPU compiler and relevant aspects
paper/paper.bib:@article{besard2018juliagpu,
paper/paper.bib:  title         = {Effective Extensible Programming: Unleashing {Julia} on {GPUs}},
paper/paper.md:ADI algorithms differ in how they estimate the noise model. For example instead of estimating the systematics from the *target* star, a *reference* star can be used (reference differential imaging, RDI). Instead of using the entire frame, the cube can be processed in annuli corresponding to different circumstellar regions. These geometric techniques are independent of the underlying description of the algorithms. Similarly, GPU programming or out-of-core processing are computational techniques which appear like implementation details in comparison to the algorithms or how they are applied. Creating a *modular* and *generic* framework for HCI enables scientists to explore different algorithms and techniques flexibly, allowing more thorough and deeper investigations into the capabilities of HCI for finding exoplanets.
paper/paper.md:Algorithm designers will find that Julia is highly composable, so extending or adding a new algorithm only requires writing the code that is *unique to that algorithm*. Julia's language interoperability also means the algorithm can be implemented in Python or C, for example. In other words, to be able to fully use the post-processing capabilities of `ADI.jl` a new algorithm only needs to implement one or two methods. Furthermore, computational techniques like GPU programming are available *generically* through packages like `CUDA.jl` [@besard2018juliagpu].

```
