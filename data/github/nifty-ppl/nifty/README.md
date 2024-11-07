# https://github.com/NIFTy-PPL/NIFTy

```console
docs/source/user/paper.md:Being written in JAX, \texttt{NIFTy.re} effortlessly runs on accelerator hardware such as the GPU and TPU, vectorizes models whenever possible, and just-in-time compiles code for additional performance.
docs/source/user/paper.md:The latter publication extensively used \texttt{NIFTy.re}'s GPU support to reduce the runtime by two orders of magnitude compared to the CPU.
docs/source/user/paper.md:This model exploits nearest neighbor relations on various coarsenings of the discretized modeled space and runs very efficiently on GPUs.
docs/source/user/paper.md:![Median evaluation time of applying the Fisher metric plus the identity metric to random input for \texttt{NIFTy.re} and \texttt{NIFTy} on the CPU (one and eight core(s) of an Intel Xeon Platinum 8358 CPU clocked at 2.60G Hz) and the GPU (A100 SXM4 80 GB HBM2). The quantile range from the 16%- to the 84%-quantile is obscured by the marker symbols.\label{fig:benchmark_nthreads=1+8_devices=cpu+gpu}](benchmark_nthreads=1+8_devices=cpu+gpu.png)
docs/source/user/paper.md:\autoref{fig:benchmark_nthreads=1+8_devices=cpu+gpu} shows the median evaluation time in \texttt{NIFTy} of applying $M_p$ to new, random tangent positions and the evaluation time in \texttt{NIFTy.re} of building $M_p$ and applying it to new, random tangent positions for exponentially larger models.
docs/source/user/paper.md:We ran the benchmark on one CPU core, eight CPU cores, and on a GPU on a compute-node with an Intel Xeon Platinum 8358 CPU clocked at 2.60G Hz and an NVIDIA A100 SXM4 80 GB HBM2 GPU.
docs/source/user/paper.md:The benchmark used `jax==0.4.23` and `jaxlib==0.4.23+cuda12.cudnn89`.
docs/source/user/paper.md:On the GPU, \texttt{NIFTy.re} is consistently about one to two orders of magnitude faster than \texttt{NIFTy} for images larger than 100,000 pixels.
docs/source/user/paper.md:Models in \texttt{NIFTy.re} and \texttt{NIFTy} are often well aligned with GPU programming models and thus consistently perform well on the GPU.
docs/source/user/paper.md:Modeling components such as the new GP models implemented in \texttt{NIFTy.re} are even better aligned with GPU programming paradigms and yield even higher performance gains [@Edenhofer2022].
README.md:This model exploits nearest neighbor relations on various coarsening of the discretized modeled space and runs very efficiently on GPUs.
paper/minimal_benchmark_vis.py:    "1": "benchmark_nthreads=1_devices=NVIDIA A100-SXM4-80GB+cpu.npy",
paper/minimal_benchmark_vis.py:    "8": "benchmark_nthreads=8_devices=NVIDIA A100-SXM4-80GB+cpu.npy",
paper/minimal_benchmark_vis.py:magic_gpu_key = "GPU NIFTy.re"
paper/minimal_benchmark_vis.py:            if platform.lower().startswith("nvidia"):
paper/minimal_benchmark_vis.py:                nm = magic_gpu_key
paper/minimal_benchmark_vis.py:n_gpu_plots = 0
paper/minimal_benchmark_vis.py:fn_out_stem = "benchmark_nthreads=1+8_devices=cpu+gpu"
paper/paper.tex:on accelerator hardware such as the GPU and TPU, vectorizes models
paper/paper.tex:used \texttt{NIFTy.re}'s GPU support to reduce the runtime by two orders
paper/paper.tex:on GPUs. For one-dimensional problems with arbitrarily spaced pixels,
paper/paper.tex:clocked at 2.60G Hz) and the GPU (A100 SXM4 80 GB HBM2). The quantile
paper/paper.tex:symbols.\label{fig:benchmark_nthreads=1+8_devices=cpu+gpu}}
paper/paper.tex:\autoref{fig:benchmark_nthreads=1+8_devices=cpu+gpu} shows the median
paper/paper.tex:benchmark on one CPU core, eight CPU cores, and on a GPU on a
paper/paper.tex:and an NVIDIA A100 SXM4 80 GB HBM2 GPU. The benchmark used
paper/paper.tex:\texttt{jax==0.4.23} and \texttt{jaxlib==0.4.23+cuda12.cudnn89}. We vary
paper/paper.tex:slightly better at using the additional cores. On the GPU,
paper/paper.tex:\texttt{NIFTy} are often well aligned with GPU programming models and
paper/paper.tex:thus consistently perform well on the GPU. Modeling components such as
paper/paper.tex:aligned with GPU programming paradigms and yield even higher performance
paper/paper.md:Being written in JAX, \texttt{NIFTy.re} effortlessly runs on accelerator hardware such as the GPU and TPU, vectorizes models whenever possible, and just-in-time compiles code for additional performance.
paper/paper.md:The latter publication extensively used \texttt{NIFTy.re}'s GPU support to reduce the runtime by two orders of magnitude compared to the CPU.
paper/paper.md:This model exploits nearest neighbor relations on various coarsenings of the discretized modeled space and runs very efficiently on GPUs.
paper/paper.md:![Median evaluation time of applying the Fisher metric plus the identity metric to random input for \texttt{NIFTy.re} and \texttt{NIFTy} on the CPU (one and eight core(s) of an Intel Xeon Platinum 8358 CPU clocked at 2.60G Hz) and the GPU (A100 SXM4 80 GB HBM2). The quantile range from the 16%- to the 84%-quantile is obscured by the marker symbols.\label{fig:benchmark_nthreads=1+8_devices=cpu+gpu}](benchmark_nthreads=1+8_devices=cpu+gpu.png)
paper/paper.md:\autoref{fig:benchmark_nthreads=1+8_devices=cpu+gpu} shows the median evaluation time in \texttt{NIFTy} of applying $M_p$ to new, random tangent positions and the evaluation time in \texttt{NIFTy.re} of building $M_p$ and applying it to new, random tangent positions for exponentially larger models.
paper/paper.md:We ran the benchmark on one CPU core, eight CPU cores, and on a GPU on a compute-node with an Intel Xeon Platinum 8358 CPU clocked at 2.60G Hz and an NVIDIA A100 SXM4 80 GB HBM2 GPU.
paper/paper.md:The benchmark used `jax==0.4.23` and `jaxlib==0.4.23+cuda12.cudnn89`.
paper/paper.md:On the GPU, \texttt{NIFTy.re} is consistently about one to two orders of magnitude faster than \texttt{NIFTy} for images larger than 100,000 pixels.
paper/paper.md:Models in \texttt{NIFTy.re} and \texttt{NIFTy} are often well aligned with GPU programming models and thus consistently perform well on the GPU.
paper/paper.md:Modeling components such as the new GP models implemented in \texttt{NIFTy.re} are even better aligned with GPU programming paradigms and yield even higher performance gains [@Edenhofer2022].
nifty8/re/refine/charted_field.py:    # On the GPU a single `cov_from_loc` call is about twice as fast as three
src/re/refine/charted_field.py:    # On the GPU a single `cov_from_loc` call is about twice as fast as three

```
