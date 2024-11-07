# https://github.com/martinfleis/clustergram

```console
README.md:use CPU and RAPIDS.AI `cuML`, which uses GPU. Note that all are optional dependencies
README.md:offered by `scikit-learn`. Data which were originally computed on GPU are converted to
paper/paper.bib:  abstract       = {Smarter applications are making better use of the insights gleaned from data, having an impact on every industry and research discipline. At the core of this revolution lies the tools and the methods that are driving it, from processing the massive piles of data generated each day to learning from and taking useful action. Deep neural networks, along with advancements in classical machine learning and scalable general-purpose graphics processing unit (GPU) computing, have become critical components of artificial intelligence, enabling many of these astounding breakthroughs and lowering the barrier to adoption. Python continues to be the most preferred language for scientific computing, data science, and machine learning, boosting both performance and productivity by enabling the use of low-level libraries and clean high-level APIs. This survey offers insight into the field of machine learning with Python, taking a tour through important topics to identify some of the core hardware and software paradigms that have enabled it. We cover widely-used libraries and concepts, collected together for holistic comparison, with the goal of educating the reader and driving the field of Python machine learning forward.},
paper/paper.md:hierarchical (or agglomerative) algorithms) or a GPU (`cuML` [@raschka2020machine] for
paper/paper.md:K-Means), where the GPU path is computing both clustering and the underlying data for
paper/paper.md:clustergram visualization on GPU, minimizing the need of data transfer between both.
clustergram/clustergram.py:    ``scipy`` which use CPU and RAPIDS.AI ``cuML``, which uses GPU. Note that all
clustergram/clustergram.py:        ``sklearn`` does computation on CPU, ``cuml`` on GPU.
clustergram/clustergram.py:        GPU option is not implemented.
clustergram/clustergram.py:        GPU option is not implemented.

```
