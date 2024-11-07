# https://github.com/vavrines/Kinetic.jl

```console
docs/make.jl:    "CUDA" => "para_cuda.md",
docs/src/parallel.md:Kinetic integrates the the latter two mechanism along with the CUDA-based GPU computing.
docs/src/parallel.md:    if has_cuda()
docs/src/parallel.md:        @info "Kinetic will run with CUDA"
docs/src/parallel.md:        for (i, dev) in enumerate(CUDA.devices())
docs/src/parallel.md:            @info "$i: $(CUDA.name(dev))"
docs/src/parallel.md:        @info "Scalar operation is disabled in CUDA"
docs/src/parallel.md:        CUDA.allowscalar(false)
docs/src/parallel.md:As the package is imported, it will report the computational resources (processors, threads and CUDA devices) that are going to be utilized.
docs/src/index.md:Kinetic is a computational fluid dynamics toolbox written in Julia. Based on differentiable programming, mechanical and neural network models are fused and solved in a unified framework. Simultaneous 1-3 dimensional numerical simulations can be performed on CPUs and GPUs.
docs/src/para_cuda.md:# GPU computing
docs/src/para_cuda.md:The thriving development of GPUs provides an alternative choice for scientific computing.
docs/src/para_cuda.md:Kinetic enables computation on the graphical architecture on the basis of [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).
docs/src/para_cuda.md:It provides the main programming interface for working with NVIDIA CUDA GPUs. 
docs/src/para_cuda.md:It features a user-friendly array abstraction, a compiler for writing CUDA kernels in Julia, and wrappers for various CUDA libraries.
docs/src/para_cuda.md:The test is conducted on a Tesla K80 GPU on [nextjournal.com](https://nextjournal.com).
docs/src/para_cuda.md:Pkg.add("CUDA")
docs/src/para_cuda.md:using Revise, CUDA, BenchmarkTools, KitBase
docs/src/para_cuda.md:Then let's turn to GPU.
docs/src/para_cuda.md:As can be seen, due to the relative small input size, the GPU threads aren't fully occupied, and therefore CPU is more efficient in this case.
docs/src/para_cuda.md:The results become around `50.011 μs (6 allocations: 234.80 KiB)` for CPU and `33.640 μs (187 allocations: 10.73 KiB)` for GPU.
docs/src/para_cuda.md:The results become around `507.960 μs (6 allocations: 2.29 MiB)` for CPU and `32.021 μs (187 allocations: 10.73 KiB)` for GPU.
docs/src/para_cuda.md:Under this size of computation, the GPU brings about 16x efficiency increment.
README.md:Simultaneous 1-3 dimensional numerical simulations can be performed on CPUs and GPUs.
README.md:        CUDA
paper/paper.md:Different parallel computing techniques are provided, e.g., multi-threading, distributed computing, and CUDA programming.
benchmark/cpu-vs-gpu.jl:using KitBase, CUDA

```
