# https://github.com/FlowModelingControl/flowtorch

```console
docs/source/overview/glossary.rst:      library for tensor operations on CPU and GPU
README.md:*flowTorch* uses the [PyTorch](https://github.com/pytorch/pytorch) library as a backend for data structures, data types, and linear algebra operations on CPU and GPU. Some cool features of *flowTorch* include:
README.md:- most algorithms run on CPU as well as on GPU
paper.md:CPU and GPU, and exploration of novel deep learning-based analysis and modeling approaches.
paper.md:are converted internally to PyTorch tensors [@paszke2015]. Once the data are available as PyTorch tensors, further processing steps like scaling, clipping, masking, splitting, or merging are readily available as single function calls. The same is true for computing the mean, the standard deviation, histograms, or quantiles. Modal analysis techniques, like dynamic mode decomposition (DMD)[@schmid2010; @kutz2016] and proper orthogonal decomposition (POD)[@brunton2019; @semaan2020], are available via the subpackage `flowtorch.analysis`. The third subpackage, `flowtorch.rom`, enables adding reduced-order models (ROMs), like cluster-based network modeling (CNM)[@fernex2021], to the post-processing pipeline. Computationally intensive tasks may be offloaded to the GPU if needed, which greatly accelerates parameter studies. The entire analysis workflow described in the previous section can be performed in a single ecosystem sketched in \autoref{fig:ft_structure}. Moreover, re-using an analysis pipeline in a different problem setting is straightforward.

```
