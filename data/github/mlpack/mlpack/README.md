# https://github.com/mlpack/mlpack

```console
doc/user/matrices.md:    - GPU matrices via [Bandicoot](https://coot.sourceforge.io) (`coot::mat`,
doc/joss_paper/paper.bib:  author={{NVIDIA}},
doc/joss_paper/paper.bib:  howpublished={\url{http://docs.nvidia.com/cuda/nvblas}}
doc/joss_paper/paper.md:[@nvblas] which would allow mlpack algorithms to be run on the GPU.  In
src/mlpack/core/math/quantile.hpp: * GPU Computing Gems, Volume 2.
src/mlpack/bindings/cli/third_party/CLI/CLI11.hpp:#ifdef __CUDACC__
src/mlpack/bindings/cli/third_party/CLI/CLI11.hpp:#ifdef __CUDACC__
src/mlpack/tests/catch.hpp:#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC) && !defined(__CUDACC__) && !defined(__LCC__)
src/mlpack/tests/catch.hpp:#  if !defined(__ibmxl__) && !defined(__CUDACC__)
src/mlpack/tests/catch.hpp:        // Code accompanying the article "Approximating the erfinv function" in GPU Computing Gems, Volume 2
src/mlpack/methods/ann/layer/grouped_convolution.hpp: * was to distribute the model over multiple GPUs as an engineering compromise.
src/mlpack/methods/ann/layer/grouped_convolution.hpp: *  url = {https://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_CondenseNet_An_Efficient_CVPR_2018_paper.pdf}

```
