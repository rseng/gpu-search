# https://github.com/cblearn/cblearn

```console
setup.cfg:    Environment :: GPU :: NVIDIA CUDA
docs/getting_started/index.rst:    Most estimators provide an (optional) implementation using ``pytorch`` to run large datasets on CPU and GPU.
docs/user_guide/index.rst:Pytorch backend (CPU/GPU)
docs/user_guide/index.rst:Second, the whole computation can run on a GPU, if available.
docs/user_guide/index.rst:If a CUDA GPU is available, the computations are automatically performed on the GPU (if the computation should be forced to run on a cpu,
paper/supplementary.md:We compared various CPU and GPU implementations in `cblearn` with third-party implementations in *R* [`loe` @terada_local_2014], and *MATLAB* [@van_der_maaten_stochastic_2012].
paper/supplementary.md:Every algorithm runs once per dataset on a compute node (8 CPU cores; 96GB RAM; NVIDIA RTX 2080ti) with a run-time limit of 24 hours. Some runs failed by exceeding those constraints: our FORTE implementation failed due to an "out of memory" error on the `imagenet-v2` dataset. The *MATLAB* implementation of tSTE timed out on `things` and `imagenet-v2` datasets. The run of the *R* SOE  implementation on the `imagenet-v2` dataset failed by an "unsupported long vector" error caused by the large size of the requested embedding.
paper/supplementary.md:The GPU implementations are slower on the tested datasets and noticeably less accurate for SOE and GNMDS.
paper/supplementary.md:\caption{The triplet error and runtime per estimator and dataset relative to the mean error or the fastest run. Thin lines show runs on the different datasets; the thick lines indicate the respective median. Except for STE, all CPU algorithms can embed the triplets similarly well. There are just minor differences in the runtime of the CPU implementations. The GPU implementations are usually significantly slower on the data sets used.  
paper/supplementary.md:## When should GPU implementations be preferred?
paper/supplementary.md:Regarding accuracy and runtime, our GPU implementations using the `torch` backend could not outperform the CPU pendants using the `scipy` backend on the tested datasets. However, \autoref{fig:performance-per-algorithm_cblearn} shows the GPU runtime grows slower with the number of triplets, such that they potentially outperform CPU implementations with large datasets of $10^7$ triplets and more. Sometimes, the `torch` implementations show the best accuracy.
paper/supplementary.md:![The runtime increases almost linearly with the number of triplets. However, GPU implementations have a flatter slope and thus can compensate for the initial time overhead on large datasets.
paper/supplementary.md:    \label{fig:time-per-triplets_gpu}](images/time-per-triplets_gpu.pdf){width=50%}
paper/references.bib:	abstract = {The objective of ordinal embedding is to find a Euclidean representation of a set of abstract items, using only answers to triplet comparisons of the form "Is item \$i\$ closer to the item \$j\$ or item \$k\$?". In recent years, numerous algorithms have been proposed to solve this problem. However, there does not exist a fair and thorough assessment of these embedding methods and therefore several key questions remain unanswered: Which algorithms scale better with increasing sample size or dimension? Which ones perform better when the embedding dimension is small or few triplet comparisons are available? In our paper, we address these questions and provide the first comprehensive and systematic empirical evaluation of existing algorithms as well as a new neural network approach. In the large triplet regime, we find that simple, relatively unknown, non-convex methods consistently outperform all other algorithms, including elaborate approaches based on neural networks or landmark approaches. This finding can be explained by our insight that many of the non-convex optimization approaches do not suffer from local optima. In the low triplet regime, our neural network approach is either competitive or significantly outperforms all the other methods. Our comprehensive assessment is enabled by our unified library of popular embedding algorithms that leverages GPU resources and allows for fast and accurate embeddings of millions of data points.},
paper/paper.md:## Algorithms implemented for CPU and GPU
paper/paper.md:Most algorithm implementations are built with the scientific ecosystem around `scipy` [@virtanenSciPyFundamentalAlgorithms2020;@harris_array_2020] to be fast and lightweight. Inspired by the work of @vankadara_insights_2020, we added GPU implementations with `torch` [@paszke2019pytorch;@anselPyTorchFasterMachine2024] that use stochastic optimization routines known from deep learning methods.
paper/paper.md:These GPU implementations can be used with large datasets and rapidly adapted thanks to `torch`'s automated differentiation methods.
paper/paper.md:: Algorithm implementations in `cblearn`. Most of these come in multiple variants: Different backends for small datasets on CPU and large datasets on GPU as well as variations of objective functions. \label{tablealgorithms}
cblearn/embedding/_ckl.py:        It can executed on CPU, but also CUDA GPUs.
cblearn/embedding/_ckl.py:            device: The device on which pytorch computes. {"auto", "cpu", "cuda"}
cblearn/embedding/_ckl.py:                "auto" chooses cuda (GPU) if available, but falls back on cpu if not.
cblearn/embedding/_soe.py:        It can executed on CPU, but also CUDA GPUs.
cblearn/embedding/_soe.py:        The following is running on the CUDA GPU, if available (but requires pytorch installed).
cblearn/embedding/_soe.py:            device: The device on which pytorch computes. {"auto", "cpu", "cuda"}
cblearn/embedding/_soe.py:                "auto" chooses cuda (GPU) if available, but falls back on cpu if not.
cblearn/embedding/_oenn.py:        It can executed on CPU, but also CUDA GPUs.
cblearn/embedding/_oenn.py:            device: The device on which pytorch computes. {"auto", "cpu", "cuda"}
cblearn/embedding/_oenn.py:                "auto" chooses cuda (GPU) if available, but falls back on cpu if not.
cblearn/embedding/_oenn.py:        if torch.cuda.is_available():
cblearn/embedding/_oenn.py:            torch.cuda.manual_seed_all(seed)
cblearn/embedding/_gnmds.py:        It can executed on CPU, but also CUDA GPUs.
cblearn/embedding/_gnmds.py:                 The device on which pytorch computes. {"auto", "cpu", "cuda"}
cblearn/embedding/_gnmds.py:                "auto" chooses cuda (GPU) if available, but falls back on cpu if not.
cblearn/embedding/_forte.py:        It can executed on CPU, but also CUDA GPUs. We optimize using BFSGS and Strong-Wolfe line search.
cblearn/embedding/_forte.py:            device: The device on which pytorch computes. {"auto", "cpu", "cuda"}
cblearn/embedding/_forte.py:                "auto" chooses cuda (GPU) if available, but falls back on cpu if not.
cblearn/embedding/_torch_utils.py:        if torch.cuda.is_available():
cblearn/embedding/_torch_utils.py:            return "cuda"
cblearn/embedding/_torch_utils.py:                Device to run the minimization on, usually "cpu" or "cuda".
cblearn/embedding/_torch_utils.py:                "auto" uses "cuda", if available.
cblearn/embedding/_ste.py:        It can executed on CPU, but also CUDA GPUs.
cblearn/embedding/_ste.py:        The following is running on the CUDA GPU, if available (but requires pytorch installed).
cblearn/embedding/_ste.py:            device: The device on which pytorch computes. {"auto", "cpu", "cuda"}
cblearn/embedding/_ste.py:                "auto" chooses cuda (GPU) if available, but falls back on cpu if not.

```
