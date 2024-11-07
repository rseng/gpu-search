# https://github.com/JuliaOcean/PlanktonIndividuals.jl

```console
Manifest.toml:    StructArraysGPUArraysCoreExt = "GPUArraysCore"
Manifest.toml:    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
docs/make.jl:             "GPU Support" => "gpu_support.md",
docs/joss/paper.md:title: 'PlanktonIndividuals.jl: A GPU supported individual-based phytoplankton life cycle model'
docs/joss/paper.md:  - GPU support
docs/joss/paper.md:Marine phytoplankton contributes ~50% of the total primary production on Earth forming the basis of the food web in the oceans. Understanding the spatial distribution and temporal variations of the phytoplankton community is essential to the global carbon and nutrient cycles. `PlanktonIndividuals.jl` is a fast individual-based model that represents the phytoplankton life cycle in detail, is written in Julia, and runs on both CPU and GPU. The model is used to elucidate temporal and spatial variations in phytoplankton cell density and stoichiometry, as well as growth and division behaviors induced by diel cycle and physical motions ranging from sub-mesoscale to large scale processes. 
docs/joss/paper.md:Due to computational limitations, previous microbial individual-based models could only simulate a limited number of individuals, with each individual representing cell populations rather than individual cells [@hellweger2016advancing]. In order to overcome this obstacle, `PlanktonIndividuals.jl` exploits the power of GPU which was first developed for video rendering but now offer extremely efficient, highly parallelized computing power. With GPU support, the simulations in `PlanktonIndividuals.jl` are sped up over 50 times compared with CPU simulations.
docs/joss/paper.md:Our package is unique in the way that it is tailored to the analysis of marine ecosystems across a wide range of scales using HPC systems. To this end, `PlanktonIndividuals.jl` provides a comprehensive description of phytoplankton physiology and biogeochemistry, written in a fast language, Julia, and with GPU support. It further innovates in combining the Eulerian and Lagrangian perspectives. Plankton individuals (Lagrangian) indeed have two-way interactions with gridded nutrient fields (Eulerian) that are advected by the same flow fields (in one-, two-, or three-dimensions) in our package. 
docs/joss/paper.md:Further development plans include implementation of macro-molecular model [@Inomura2020] and support for distributed parallelism with CUDA-aware MPI.
docs/src/model_setup.md:Passing `arch = CPU()` or `arch = GPU()` to the `PlanktonModel` constructor will determine whether the model
docs/src/model_setup.md:is time stepped on a CPU or GPU.
docs/src/model_setup.md:The only thing that needs to be changed is `arch = CPU()` or `arch = GPU()`.
docs/src/model_setup.md:!!! tip "Running on GPUs"
docs/src/model_setup.md:    Please refer to [GPU Support](@ref) for more detail on running `PlanktonIndividuals` on GPUs and don't hesitate to [open an issue](https://github.com/JuliaOcean/PlanktonIndividuals.jl/issues/new) if you have any difficulty.
docs/src/benchmarks.md:  GPU: Tesla P100-PCIE-12GB
docs/src/benchmarks.md:  CUDA runtime 11.8, artifact installation
docs/src/benchmarks.md:  CUDA driver 11.2
docs/src/benchmarks.md:  NVIDIA driver 460.84.0
docs/src/benchmarks.md:|  GPU |    1024 |   7.085 ms |   7.158 ms |   7.364 ms |   9.323 ms |   1.92 MiB |  21327 |
docs/src/benchmarks.md:|  GPU |   32768 |   7.435 ms |   7.520 ms |   7.925 ms |  10.173 ms |   1.92 MiB |  21327 |
docs/src/benchmarks.md:|  GPU |  131072 |   7.053 ms |   9.161 ms |   9.851 ms |  19.812 ms |   1.92 MiB |  21294 |
docs/src/benchmarks.md:|  GPU | 1048576 |   8.005 ms |  46.217 ms |  47.484 ms | 122.516 ms |   1.92 MiB |  21294 |
docs/src/benchmarks.md:  GPU: Tesla P100-PCIE-12GB
docs/src/benchmarks.md:  CUDA runtime 11.8, artifact installation
docs/src/benchmarks.md:  CUDA driver 11.2
docs/src/benchmarks.md:  NVIDIA driver 460.84.0
docs/src/benchmarks.md:|  GPU |    1024 |  32 |   6.902 ms |   6.920 ms |   7.101 ms |   8.719 ms |  1.98 MiB |  21513 |
docs/src/benchmarks.md:|  GPU |    1024 |  64 |   7.417 ms |   7.622 ms |   7.755 ms |   8.430 ms |  2.07 MiB |  21632 |
docs/src/benchmarks.md:|  GPU |    1024 | 128 |   7.734 ms |   8.071 ms |   8.141 ms |   8.854 ms |  2.45 MiB |  21713 |
docs/src/benchmarks.md:|  GPU |   32768 |  32 |   7.011 ms |   7.092 ms |   7.392 ms |  10.142 ms |  1.98 MiB |  21513 |
docs/src/benchmarks.md:|  GPU |   32768 |  64 |   6.769 ms |   6.837 ms |   7.152 ms |  10.035 ms |  2.07 MiB |  21632 |
docs/src/benchmarks.md:|  GPU |   32768 | 128 |   7.027 ms |   8.381 ms |   8.561 ms |  11.845 ms |  2.45 MiB |  21713 |
docs/src/benchmarks.md:|  GPU |  131072 |  32 |   6.580 ms |   8.054 ms |   8.560 ms |  15.323 ms |  1.98 MiB |  21541 |
docs/src/benchmarks.md:|  GPU |  131072 |  64 |   7.491 ms |   9.106 ms |   9.664 ms |  16.128 ms |  2.07 MiB |  21599 |
docs/src/benchmarks.md:|  GPU |  131072 | 128 |   7.918 ms |  12.640 ms |  12.791 ms |  23.534 ms |  2.45 MiB |  21680 |
docs/src/benchmarks.md:|  GPU | 1048576 |  32 |   9.781 ms |  35.539 ms |  36.437 ms |  59.171 ms |  1.98 MiB |  21528 |
docs/src/benchmarks.md:|  GPU | 1048576 |  64 |  10.682 ms |  37.958 ms |  39.055 ms |  65.476 ms |  2.08 MiB |  21647 |
docs/src/benchmarks.md:|  GPU | 1048576 | 128 |   7.994 ms |  50.094 ms |  50.772 ms | 126.537 ms |  2.45 MiB |  21680 |
docs/src/benchmarks.md:  GPU: Tesla P100-PCIE-12GB
docs/src/benchmarks.md:  CUDA runtime 11.8, artifact installation
docs/src/benchmarks.md:  CUDA driver 11.2
docs/src/benchmarks.md:  NVIDIA driver 460.84.0
docs/src/benchmarks.md:|  GPU |    1024 |  32 |   6.229 ms |   6.286 ms |   6.466 ms |   7.329 ms | 2.94 MiB |  21053 |
docs/src/benchmarks.md:|  GPU |    1024 |  64 |   9.194 ms |  11.891 ms |  11.689 ms |  12.604 ms | 9.99 MiB |  21077 |
docs/src/benchmarks.md:|  GPU |   32768 |  32 |   6.570 ms |   6.638 ms |   6.966 ms |   8.974 ms | 2.94 MiB |  21053 |
docs/src/benchmarks.md:|  GPU |   32768 |  64 |   9.143 ms |  12.882 ms |  12.712 ms |  15.781 ms | 9.99 MiB |  21077 |
docs/src/benchmarks.md:|  GPU |  131072 |  32 |   6.481 ms |   9.150 ms |   9.469 ms |  16.907 ms | 2.94 MiB |  21081 |
docs/src/benchmarks.md:|  GPU |  131072 |  64 |   9.212 ms |  16.623 ms |  16.438 ms |  25.557 ms | 9.99 MiB |  21105 |
docs/src/benchmarks.md:|  GPU | 1048576 |  32 |   7.257 ms |  39.894 ms |  40.268 ms |  96.189 ms | 2.94 MiB |  21020 |
docs/src/benchmarks.md:|  GPU | 1048576 |  64 |   9.586 ms |  54.934 ms |  53.741 ms | 118.675 ms | 9.99 MiB |  21105 |
docs/src/gpu_support.md:# GPU Support
docs/src/gpu_support.md:`PlanktonIndividuals.jl` has support from `CUDA.jl` and `KernelAbstractions.jl` to be able to run on graphical processing unit (GPU) for higher performance. Depending on the combination of CPU and GPU you have, a speedup of 35x is possible. Please see [Benchmarks](@ref benchmarks) for more details.
docs/src/gpu_support.md:## How to use a GPU
docs/src/gpu_support.md:To use a GPU to run `PlanktonIndividuals.jl` is easy. Users do not need to rewrite the setup or simulation script to change the architecture to run on. See [Architecture](@ref) for detailed instructions on setting up a model on GPU.
docs/src/gpu_support.md:!!! tip "Running on GPUs"
docs/src/gpu_support.md:    If you are having issues with running `PlanktonIndividuals` on a GPU, please
docs/src/gpu_support.md:## When to use a GPU
docs/src/gpu_support.md:GPU is very useful when running large simulations (either large domain or huge number of individuals, or both). If you simulate over 10,000 individuals, you will probably benefit form GPU. Please note, GPU is usually memory-limited, that is to say, you will probably fill up the memory on GPU long before the model slows down.
docs/src/gpu_support.md:`Individuals` take up a large amount of GPU memory due to complicated physiological processes and diagnostics. Typically, one should not try more than 50,000 individuals for a 12GB GPU.
docs/src/gpu_support.md:## GPU resources
docs/src/gpu_support.md:There are a few resources you can try to acquire a GPU from.
docs/src/gpu_support.md:1. Google Colab provides GPUs but you need to install Julia manually. Please see [this post](https://discourse.julialang.org/t/julia-on-google-colab-free-gpu-accelerated-shareable-notebooks/15319/39) on the Julia Discourse for detailed instructions.
docs/src/gpu_support.md:2. [Code Ocean](https://codeocean.com/) also has [GPU support](https://help.codeocean.com/en/articles/1053107-gpu-support). You can use "Ubuntu Linux with GPU support (18.04.3)" but you still have to install Julia manually.
docs/src/index.md:`PlanktonIndividuals.jl` is a fast individual-based model written in Julia that runs on both CPU and GPU. It simulates the life cycle of ocean phytoplankton cells as Lagrangian particles while nutrients are represented as Eulerian tracers and advected over the gridded domain. The model is used to simulate and interpret the temporal and spatial variations in phytoplankton cell density, stoichiometry, as well as growth and division behaviors induced by diel cycle and physical motions ranging from sub-mesoscale to large scale processes.
ext/PI_CUDAExt.jl:module PI_CUDAExt
ext/PI_CUDAExt.jl:using CUDA
ext/PI_CUDAExt.jl:using CUDA.CUDAKernels
ext/PI_CUDAExt.jl:import PlanktonIndividuals.Architectures: GPU, device, array_type, rng_type, isfunctional
ext/PI_CUDAExt.jl:device(::GPU) = CUDABackend()
ext/PI_CUDAExt.jl:array_type(::GPU) = CuArray
ext/PI_CUDAExt.jl:rng_type(::GPU) = CURAND.default_rng()
ext/PI_CUDAExt.jl:isfunctional(::GPU) = CUDA.functional()
ext/PI_MetalExt.jl:using GPUArrays
ext/PI_MetalExt.jl:import PlanktonIndividuals.Architectures: GPU, device, array_type, rng_type, isfunctional
ext/PI_MetalExt.jl:device(::GPU) = MetalBackend()
ext/PI_MetalExt.jl:array_type(::GPU) = MtlArray
ext/PI_MetalExt.jl:rng_type(::GPU) = GPUArrays.default_rng(MtlArray)
ext/PI_MetalExt.jl:isfunctional(::GPU) = Metal.functional()
README.md:`PlanktonIndividuals.jl` is a fast individual-based model written in Julia that can be run on both CPU and GPU. It simulates the life cycle of phytoplankton cells as Lagrangian particles in the ocean while nutrients are represented as Eulerian, density-based tracers using a [3rd order advection scheme](https://mitgcm.readthedocs.io/en/latest/algorithm/adv-schemes.html#third-order-direct-space-time-with-flux-limiting). The model is used to simulate and interpret the temporal and spacial variations of phytoplankton cell densities and stoichiometry as well as growth and division behaviors induced by diel cycle and physical motions ranging from sub-mesoscale to large scale processes.
Project.toml:CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Project.toml:GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
Project.toml:PI_CUDAExt = "CUDA"
Project.toml:PI_MetalExt = ["Metal", "GPUArrays"]
Project.toml:CUDA = "^4, 5"
Project.toml:GPUArrays = "^10,11"
examples/global_ocean_2D_example.jl:[[deps.CUDA]]
examples/global_ocean_2D_example.jl:deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "StaticArrays", "Statistics"]
examples/global_ocean_2D_example.jl:    [deps.CUDA.extensions]
examples/global_ocean_2D_example.jl:    [deps.CUDA.weakdeps]
examples/global_ocean_2D_example.jl:[[deps.CUDA_Driver_jll]]
examples/global_ocean_2D_example.jl:[[deps.CUDA_Runtime_Discovery]]
examples/global_ocean_2D_example.jl:[[deps.CUDA_Runtime_jll]]
examples/global_ocean_2D_example.jl:deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
examples/global_ocean_2D_example.jl:[[deps.GPUArrays]]
examples/global_ocean_2D_example.jl:deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
examples/global_ocean_2D_example.jl:[[deps.GPUArraysCore]]
examples/global_ocean_2D_example.jl:[[deps.GPUCompiler]]
examples/global_ocean_2D_example.jl:deps = ["Adapt", "CUDA", "JLD2", "KernelAbstractions", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Statistics", "StructArrays"]
examples/global_ocean_2D_example.jl:weakdeps = ["Adapt", "GPUArraysCore", "SparseArrays", "StaticArrays"]
examples/global_ocean_2D_example.jl:    StructArraysGPUArraysCoreExt = "GPUArraysCore"
examples/vertical_2D_example.jl:[[deps.CUDA]]
examples/vertical_2D_example.jl:deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "StaticArrays", "Statistics"]
examples/vertical_2D_example.jl:    [deps.CUDA.extensions]
examples/vertical_2D_example.jl:    [deps.CUDA.weakdeps]
examples/vertical_2D_example.jl:[[deps.CUDA_Driver_jll]]
examples/vertical_2D_example.jl:[[deps.CUDA_Runtime_Discovery]]
examples/vertical_2D_example.jl:[[deps.CUDA_Runtime_jll]]
examples/vertical_2D_example.jl:deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
examples/vertical_2D_example.jl:[[deps.GPUArrays]]
examples/vertical_2D_example.jl:deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
examples/vertical_2D_example.jl:[[deps.GPUArraysCore]]
examples/vertical_2D_example.jl:[[deps.GPUCompiler]]
examples/vertical_2D_example.jl:deps = ["Adapt", "CUDA", "JLD2", "KernelAbstractions", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Statistics", "StructArrays"]
examples/vertical_2D_example.jl:weakdeps = ["Adapt", "GPUArraysCore", "SparseArrays", "StaticArrays"]
examples/vertical_2D_example.jl:    StructArraysGPUArraysCoreExt = "GPUArraysCore"
examples/0D_experiment.jl:[[deps.CUDA]]
examples/0D_experiment.jl:deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "StaticArrays", "Statistics"]
examples/0D_experiment.jl:    [deps.CUDA.extensions]
examples/0D_experiment.jl:    [deps.CUDA.weakdeps]
examples/0D_experiment.jl:[[deps.CUDA_Driver_jll]]
examples/0D_experiment.jl:[[deps.CUDA_Runtime_Discovery]]
examples/0D_experiment.jl:[[deps.CUDA_Runtime_jll]]
examples/0D_experiment.jl:deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
examples/0D_experiment.jl:[[deps.GPUArrays]]
examples/0D_experiment.jl:deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
examples/0D_experiment.jl:[[deps.GPUArraysCore]]
examples/0D_experiment.jl:[[deps.GPUCompiler]]
examples/0D_experiment.jl:deps = ["Adapt", "CUDA", "JLD2", "KernelAbstractions", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Statistics", "StructArrays"]
examples/0D_experiment.jl:weakdeps = ["Adapt", "GPUArraysCore", "SparseArrays", "StaticArrays"]
examples/0D_experiment.jl:    StructArraysGPUArraysCoreExt = "GPUArraysCore"
examples/global_ocean_3D_example.jl:[[deps.CUDA]]
examples/global_ocean_3D_example.jl:deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "StaticArrays", "Statistics"]
examples/global_ocean_3D_example.jl:    [deps.CUDA.extensions]
examples/global_ocean_3D_example.jl:    [deps.CUDA.weakdeps]
examples/global_ocean_3D_example.jl:[[deps.CUDA_Driver_jll]]
examples/global_ocean_3D_example.jl:[[deps.CUDA_Runtime_Discovery]]
examples/global_ocean_3D_example.jl:[[deps.CUDA_Runtime_jll]]
examples/global_ocean_3D_example.jl:deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
examples/global_ocean_3D_example.jl:[[deps.GPUArrays]]
examples/global_ocean_3D_example.jl:deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
examples/global_ocean_3D_example.jl:[[deps.GPUArraysCore]]
examples/global_ocean_3D_example.jl:[[deps.GPUCompiler]]
examples/global_ocean_3D_example.jl:deps = ["Adapt", "CUDA", "JLD2", "KernelAbstractions", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Statistics", "StructArrays"]
examples/global_ocean_3D_example.jl:weakdeps = ["Adapt", "GPUArraysCore", "SparseArrays", "StaticArrays"]
examples/global_ocean_3D_example.jl:    StructArraysGPUArraysCoreExt = "GPUArraysCore"
examples/surface_mixing_3D_example.jl:[[deps.CUDA]]
examples/surface_mixing_3D_example.jl:deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "StaticArrays", "Statistics"]
examples/surface_mixing_3D_example.jl:    [deps.CUDA.extensions]
examples/surface_mixing_3D_example.jl:    [deps.CUDA.weakdeps]
examples/surface_mixing_3D_example.jl:[[deps.CUDA_Driver_jll]]
examples/surface_mixing_3D_example.jl:[[deps.CUDA_Runtime_Discovery]]
examples/surface_mixing_3D_example.jl:[[deps.CUDA_Runtime_jll]]
examples/surface_mixing_3D_example.jl:deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
examples/surface_mixing_3D_example.jl:[[deps.GPUArrays]]
examples/surface_mixing_3D_example.jl:deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
examples/surface_mixing_3D_example.jl:[[deps.GPUArraysCore]]
examples/surface_mixing_3D_example.jl:[[deps.GPUCompiler]]
examples/surface_mixing_3D_example.jl:deps = ["Adapt", "CUDA", "JLD2", "KernelAbstractions", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Statistics", "StructArrays"]
examples/surface_mixing_3D_example.jl:weakdeps = ["Adapt", "GPUArraysCore", "SparseArrays", "StaticArrays"]
examples/surface_mixing_3D_example.jl:    StructArraysGPUArraysCoreExt = "GPUArraysCore"
examples/horizontal_2D_example.jl:[[deps.CUDA]]
examples/horizontal_2D_example.jl:deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "StaticArrays", "Statistics"]
examples/horizontal_2D_example.jl:    [deps.CUDA.extensions]
examples/horizontal_2D_example.jl:    [deps.CUDA.weakdeps]
examples/horizontal_2D_example.jl:[[deps.CUDA_Driver_jll]]
examples/horizontal_2D_example.jl:[[deps.CUDA_Runtime_Discovery]]
examples/horizontal_2D_example.jl:[[deps.CUDA_Runtime_jll]]
examples/horizontal_2D_example.jl:deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
examples/horizontal_2D_example.jl:[[deps.GPUArrays]]
examples/horizontal_2D_example.jl:deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
examples/horizontal_2D_example.jl:[[deps.GPUArraysCore]]
examples/horizontal_2D_example.jl:[[deps.GPUCompiler]]
examples/horizontal_2D_example.jl:deps = ["Adapt", "CUDA", "JLD2", "KernelAbstractions", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Statistics", "StructArrays"]
examples/horizontal_2D_example.jl:weakdeps = ["Adapt", "GPUArraysCore", "SparseArrays", "StaticArrays"]
examples/horizontal_2D_example.jl:    StructArraysGPUArraysCoreExt = "GPUArraysCore"
src/PlanktonIndividuals.jl:    Architecture, GPU, CPU,
src/Biogeochemistry/nutrient_fields.jl:- `arch`: `CPU()` or `GPU()`. The computer architecture used to time-step `model`.
src/Model/models.jl:- `arch` : `CPU()` or `GPU()`. Computer architecture being used to run the model.
src/Grids/utils.jl:##### adapt the grid struct to GPU
src/Architectures.jl:export CPU, GPU, Architecture
src/Architectures.jl:    GPU <: Architecture
src/Architectures.jl:Run PlanktonIndividuals on one CUDA GPU node.
src/Architectures.jl:struct GPU <: Architecture end

```
