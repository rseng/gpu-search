# https://github.com/Sm00thix/IKPLS

```console
ikpls/jax_ikpls_alg_1.py:allows CPU, GPU, and TPU execution.
ikpls/jax_ikpls_alg_2.py:allows CPU, GPU, and TPU execution.
tests/load_data.py:https://openaccess.thecvf.com/content/ICCV2023W/CVPPA/html/Engstrom_Improving_Deep_Learning_on_Hyperspectral_Images_of_Grain_by_Incorporating_ICCVW_2023_paper.html
tests/test_ikpls.py:    def check_cpu_gpu_equality(
tests/test_ikpls.py:        Check equality properties between CPU and GPU implementations of PLS algorithm.
tests/test_ikpls.py:        self.check_cpu_gpu_equality(
tests/test_ikpls.py:        self.check_cpu_gpu_equality(
tests/test_ikpls.py:        self.check_cpu_gpu_equality(
tests/test_ikpls.py:        self.check_cpu_gpu_equality(
tests/test_ikpls.py:        self.check_cpu_gpu_equality(
tests/test_ikpls.py:        self.check_cpu_gpu_equality(
tests/test_ikpls.py:        self.check_cpu_gpu_equality(
README.md:Dive into cutting-edge Python implementations of the IKPLS (Improved Kernel Partial Least Squares) Algorithms #1 and #2 [[1]](#references) for CPUs, GPUs, and TPUs. IKPLS is both fast [[2]](#references) and numerically stable [[3]](#references) making it optimal for PLS modeling.
README.md:- Use our JAX [[6]](#references) implementations on CPUs or **leverage powerful GPUs and TPUs for PLS modelling**.
README.md:The JAX implementations support running on both CPU, GPU, and TPU.
README.md:- To enable NVIDIA GPU execution, install JAX and CUDA with:
README.md:    pip3 install -U "jax[cuda12]"
paper/timings/timings.py:def cross_val_gpu_pls(pls, X, Y, n_components, n_splits, show_progress):
paper/timings/timings.py:    Perform cross-validation for PLS on GPU and measure the execution time.
paper/timings/timings.py:def single_fit_gpu_pls(pls, X, Y, n_components):
paper/timings/timings.py:    Fit PLS model on GPU and measure the execution time.
paper/reproducing-results-notebook.py:SKIP_GPU = False
paper/reproducing-results-notebook.py:    parser.add_argument("-g", "--skip-gpu", default=SKIP_GPU,
paper/reproducing-results-notebook.py:                        help="skip runs that require a GPU")
paper/reproducing-results-notebook.py:    SKIP_GPU = args.skip_gpu
paper/reproducing-results-notebook.py:print(f"using {OUR_TIMINGS=} {SKIP_RUNS_LONGER_THAN=} {SKIP_GPU=} {DRY_RUN=}")
paper/reproducing-results-notebook.py:if SKIP_GPU:
paper/reproducing-results-notebook.py:status("removed GPU-only runs")
paper/plot_timings.py:            model_name = "JAX IKPLS #1 (GPU)"
paper/plot_timings.py:            model_name = "JAX IKPLS #2 (GPU)"
paper/time_pls.py:    cross_val_gpu_pls,
paper/time_pls.py:    single_fit_gpu_pls,
paper/time_pls.py:        help="Number of parallel jobs to use. A value of -1 will use all available cores. Not used for JAX implementations as it is assumed these will run on a TPU or GPU.",
paper/time_pls.py:            time = single_fit_gpu_pls(pls, X, Y, n_components)
paper/time_pls.py:            time = cross_val_gpu_pls(
paper/time_pls.py:        num_cores = 'GPU'
paper/paper.md:title: 'IKPLS: Improved Kernel Partial Least Squares and Fast Cross-Validation Algorithms for Python with CPU and GPU Implementations Using NumPy and JAX'
paper/paper.md:`ikpls` offers NumPy-based CPU and JAX-based CPU/GPU/TPU implementations. The JAX implementations are also differentiable, allowing seamless integration with deep learning techniques. This versatility enables users to handle diverse data dimensions efficiently.
paper/paper.md:2. both variants of IKPLS for GPUs, both of which are end-to-end differentiable, allowing integration with deep learning models;
paper/paper.md:This work introduces the Python software package, `ikpls`, with novel, fast implementations of IKPLS Algorithm #1 and Algorithm #2 by @dayal1997improved, which have previously been compared with other PLS algorithms and shown to be fast [@alin2009comparison] and numerically stable [@andersson2009comparison]. The implementations introduced in this work use NumPy [@harris2020array] and JAX [@jax2018github]. The NumPy implementations can be executed on CPUs, and the JAX implementations can be executed on CPUs, GPUs, and TPUs. The JAX implementations are also end-to-end differentiable, allowing integration into deep learning methods. This work compares the execution time of the implementations on input data of varying dimensions. It reveals that choosing the implementation that best fits the data will yield orders of magnitude faster execution than the common NIPALS [@wold1966estimation] implementation of PLS, which is the one implemented by scikit-learn [@scikit-learn], an extensive machine learning library for Python. With the implementations introduced in this work, choosing the optimal number of components and the optimal preprocessing becomes much more feasible than previously. Indeed, derivatives of this work have previously been applied to do this precisely [@engstrom2023improving; @engstrom2023analyzing].
paper/paper.md:The `ikpls` package has been rigorously tested for equivalence against scikit-learn's NIPALS using NIR spectra data from @dreier2022hyperspectral and scikit-learn's PLS test-suite. [Examples](https://github.com/Sm00thix/IKPLS/blob/main/examples/) are provided for core functionalities, demonstrating fitting, predicting, cross-validating on CPU and GPU, and gradient propagation through PLS fitting.
paper/paper.md:For GPU/TPU acceleration, ikpls provides Python classes for each IKPLS algorithm using JAX. JAX combines Autograd [@maclaurin2015autograd] with [XLA (Accelerated Linear Algebra)](https://www.tensorflow.org/xla) for high-performance computation on various hardware. Automatic differentiation in forward and backward modes enables seamless integration with deep learning techniques, supporting user-defined metric functions.
paper/paper.md:The results in \autoref{fig:timings} suggest CPU IKPLS for single fits, with a preference for IKPLS #2 if $N \gg K$. GPU usage is advised for larger datasets. In cross-validation, IKPLS options consistently outperform scikit-learn's NIPALS, with CPU IKPLS #2 (fast cross-validation) excelling, especially for large datasets. GPU IKPLS #1 is optimal in specific cases, considering preprocessing constraints. Fast cross-validation delivers significant speedup, more pronounced for IKPLS #2, especially when dealing with a larger number of target variables ($M$) [@engstrøm2024shortcutting].
paper/paper.md:| GPU         | NVIDIA GeForce RTX3090 Ti, CUDA 11.8 |

```
