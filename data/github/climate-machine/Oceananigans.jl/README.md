# https://github.com/CliMA/Oceananigans.jl

```console
docs/oceananigans.bib:	title = {High-level {GPU} programming in {Julia}},
docs/oceananigans.bib:@article{Besard19GPU,
docs/oceananigans.bib:	title = {Effective extensible programming: unleashing {Julia} on {GPUs}},
docs/oceananigans.bib:	title = {Dynamic automatic differentiation of {GPU} broadcast kernels},
docs/src/simulation_tips.md:Furthermore, in case of more complex GPU runs, some details could
docs/src/simulation_tips.md:## General (CPU/GPU) simulation tips
docs/src/simulation_tips.md:GPU kernels (such as functions defining boundary conditions and forcings). Otherwise the Julia GPU
docs/src/simulation_tips.md:compiler can fail with obscure errors. This is explained in more detail in the GPU simulation tips
docs/src/simulation_tips.md:certainty_, since Julia and KernelAbstractions.jl (needed for GPU runs) already inline some
docs/src/simulation_tips.md:## GPU simulation tips
docs/src/simulation_tips.md:Running on GPUs can be very different from running on CPUs. Oceananigans makes most of the necessary
docs/src/simulation_tips.md:changes in the background, so that for very simple simulations changing between CPUs and GPUs is
docs/src/simulation_tips.md:just a matter of changing the `architecture` argument in the model from `CPU()` to `GPU()`. However,
docs/src/simulation_tips.md:GPU computing (and Julia) is again desirable, an inexperienced user can also achieve high efficiency
docs/src/simulation_tips.md:in GPU simulations by following a few simple principles.
docs/src/simulation_tips.md:### Global variables that need to be used in GPU computations need to be defined as constants or passed as parameters
docs/src/simulation_tips.md:Any global variable that needs to be accessed by the GPU needs to be a constant or the simulation
docs/src/simulation_tips.md:will throw an error if run on the GPU (and will run more slowly than it should on the CPU).
docs/src/simulation_tips.md:### Complex diagnostics using computed `Field`s may not work on GPUs
docs/src/simulation_tips.md:the compiler can't translate them into GPU code and they fail for GPU runs. (This limitation is summarized 
docs/src/simulation_tips.md:For example, in the example below, calculating `u²` works in both CPUs and GPUs, but calculating 
docs/src/simulation_tips.md:`ε` will not compile on GPUs when we call the command `compute!`:
docs/src/simulation_tips.md:GPU runs are sometimes memory-limited. A state-of-the-art Tesla V100 GPU has 32GB of
docs/src/simulation_tips.md:For large simulations on the GPU, careful management of memory allocation may be required:
docs/src/simulation_tips.md:- Use the [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) command
docs/src/simulation_tips.md:  line utility to monitor the memory usage of the GPU. It should tell you how much memory there is
docs/src/simulation_tips.md:  on your GPU and how much of it you're using and you can run it from Julia via
docs/src/simulation_tips.md:  shell> run(`nvidia-smi`)
docs/src/simulation_tips.md:### Arrays in GPUs are usually different from arrays in CPUs
docs/src/simulation_tips.md:Oceananigans.jl uses [`CUDA.CuArray`](https://cuda.juliagpu.org/stable/usage/array/) to store 
docs/src/simulation_tips.md:data for GPU computations. One limitation of `CuArray`s compared to the `Array`s used for 
docs/src/simulation_tips.md:launched through CUDA.jl or KernelAbstractions.jl. (You can learn more about GPU kernels 
docs/src/simulation_tips.md:[here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels) and 
docs/src/simulation_tips.md:[here](https://cuda.juliagpu.org/stable/usage/overview/#Kernel-programming-with-@cuda).)
docs/src/simulation_tips.md:Doing so requires individual elements to be copied from or to the GPU for processing,
docs/src/simulation_tips.md:Oceananigans.jl disables CUDA scalar indexing by default. See the
docs/src/simulation_tips.md:[scalar indexing](https://juliagpu.github.io/CUDA.jl/dev/usage/workflow/#UsageWorkflowScalar)
docs/src/simulation_tips.md:section of the CUDA.jl documentation for more information on scalar indexing.
docs/src/simulation_tips.md:julia> grid = RectilinearGrid(GPU(); size=(1, 1, 1), extent=(1, 1, 1), halo=(1, 1, 1))
docs/src/simulation_tips.md:1×1×1 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on GPU with 1×1×1 halo
docs/src/simulation_tips.md:NonhydrostaticModel{GPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
docs/src/simulation_tips.md:├── grid: 1×1×1 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on GPU with 1×1×1 halo
docs/src/simulation_tips.md:OffsetArrays.OffsetArray{Float64, 3, CUDA.CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}}
docs/src/simulation_tips.md:3×3×3 OffsetArray(::CUDA.CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}, 0:2, 0:2, 0:2) with eltype Float64 with indices 0:2×0:2×0:2:
docs/src/simulation_tips.md:Error showing value of type OffsetArrays.OffsetArray{Float64, 3, CUDA.CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}}:
docs/src/simulation_tips.md:Here `CUDA.jl` throws an error because scalar `getindex` is not `allowed`. There are ways to
docs/src/simulation_tips.md:in the [CUDA.jl documentation](https://cuda.juliagpu.org/stable/usage/workflow/#UsageWorkflowScalar)), but this option
docs/src/simulation_tips.md:can be very slow on GPUs, so it is advised to only use this last method when using the REPL or 
docs/src/simulation_tips.md:forcing functions on a GPU. To learn more about working with `CuArray`s, see the
docs/src/simulation_tips.md:[array programming](https://juliagpu.github.io/CUDA.jl/dev/usage/array/) section
docs/src/simulation_tips.md:of the CUDA.jl documentation.
docs/src/contributing.md:* Report the Oceananigans version, Julia version, machine (especially if using a GPU) and any other possibly useful details of the computational environment in which the bug was created.
docs/src/appendix/benchmarks.md:on a CPU versus a GPU.  We find that with the `WENO` advection scheme we get a maximum speedup of more than 400 times on a `16384^2` grid.
docs/src/appendix/benchmarks.md:  GPU: Tesla V100-SXM2-32GB
docs/src/appendix/benchmarks.md:│           GPU │     Float64 │    32 │   3.468 ms │   3.656 ms │   3.745 ms │   4.695 ms │  1.82 MiB │   5687 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │     Float64 │    64 │   3.722 ms │   3.903 ms │   4.050 ms │   5.671 ms │  1.82 MiB │   5687 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │     Float64 │   128 │   3.519 ms │   3.808 ms │   4.042 ms │   6.372 ms │  1.82 MiB │   5687 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │     Float64 │   256 │   3.822 ms │   4.153 ms │   4.288 ms │   5.810 ms │  1.82 MiB │   5687 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │     Float64 │   512 │   4.637 ms │   4.932 ms │   4.961 ms │   5.728 ms │  1.82 MiB │   5765 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │     Float64 │  1024 │   3.240 ms │   3.424 ms │   3.527 ms │   4.553 ms │  1.82 MiB │   5799 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │     Float64 │  2048 │  10.783 ms │  10.800 ms │  11.498 ms │  17.824 ms │  1.98 MiB │  16305 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │     Float64 │  4096 │  41.880 ms │  41.911 ms │  42.485 ms │  47.627 ms │  2.67 MiB │  61033 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │     Float64 │  8192 │ 166.751 ms │ 166.800 ms │ 166.847 ms │ 167.129 ms │  5.21 MiB │ 227593 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │     Float64 │ 16384 │ 681.129 ms │ 681.249 ms │ 681.301 ms │ 681.583 ms │ 16.59 MiB │ 973627 │       8 │
docs/src/appendix/benchmarks.md:        Shallow water model CPU to GPU speedup
docs/src/appendix/benchmarks.md:The time graph below shows that execution times on GPU are negligibly small up until grid size `1024^2` where it starts to scale similarly to times on CPU.
docs/src/appendix/benchmarks.md:Similar to to shallow water model, the nonhydrostatic model benchmark tests for its performance on both a CPU and a GPU. It was also benchmarked with the `WENO` advection scheme. The nonhydrostatic model is 3-dimensional unlike the 2-dimensional shallow water model. Total number of grid points is Ns cubed.
docs/src/appendix/benchmarks.md:  GPU: Tesla V100-SXM2-32GB
docs/src/appendix/benchmarks.md:│           GPU │     Float32 │  32 │   4.154 ms │   4.250 ms │   4.361 ms │   5.557 ms │ 2.13 MiB │   6033 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │     Float32 │  64 │   3.383 ms │   3.425 ms │   3.889 ms │   8.028 ms │ 2.13 MiB │   6077 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │     Float32 │ 128 │   5.564 ms │   5.580 ms │   6.095 ms │  10.725 ms │ 2.15 MiB │   7477 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │     Float32 │ 256 │  38.685 ms │  38.797 ms │  39.548 ms │  46.442 ms │ 2.46 MiB │  27721 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │     Float64 │  32 │   3.309 ms │   3.634 ms │   3.802 ms │   5.844 ms │ 2.68 MiB │   6033 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │     Float64 │  64 │   3.330 ms │   3.648 ms │   4.008 ms │   7.808 ms │ 2.68 MiB │   6071 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │     Float64 │ 128 │   7.209 ms │   7.323 ms │   8.313 ms │  17.259 ms │ 2.71 MiB │   8515 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │     Float64 │ 256 │  46.614 ms │  56.444 ms │  55.461 ms │  56.563 ms │ 3.17 MiB │  38253 │      10 │
docs/src/appendix/benchmarks.md:      Nonhydrostatic model CPU to GPU speedup
docs/src/appendix/benchmarks.md:Like the shallow water model, it can be seen at grid size `64^3` that the GPU is beginning to be saturated as speedups rapidly increase. At grid sizes `128^3` and `256^3` we see the speedup stabilize to around 400 times.
docs/src/appendix/benchmarks.md:For both float types, the benchmarked GPU times of the nonhydrostatic model starts to scale like its CPU times when grid size reaches `128^3`.
docs/src/appendix/benchmarks.md:By using `MPI.jl` the shallow water model can be run on multiple CPUs and multiple GPUs. For the benchmark results shown below, each rank is run on one CPU core and each uses a distinct GPU if applicable. 
docs/src/appendix/benchmarks.md:As seen in the tables above and in the graph below, efficiency drops off to around 80% and remains as such from 16 to 128 ranks. GPUs are not used in this or the next benchmark setup. 
docs/src/appendix/benchmarks.md:### Multi-GPU Shallow Water Model
docs/src/appendix/benchmarks.md:While still a work in progress, it is possible to use CUDA-aware MPI to run the shallow water model on multiple GPUs. Though efficiencies may not be as high as multi-CPU, the multi-GPU architecture is still worthwhile when keeping in mind the baseline speedups generated by using a single GPU. Note that though it is possible for multiple ranks to share the use of a single GPU, efficiencies would significantly decrease and memory may be insufficient. The results below show up to three ranks each using a separate GPU.
docs/src/appendix/benchmarks.md:  JULIA_CUDA_USE_BINARYBUILDER = false
docs/src/appendix/benchmarks.md:  GPU: Tesla V100-SXM2-32GB
docs/src/appendix/benchmarks.md:and passive tracers and compares the difference in speedup going from CPU to GPU. Number of tracers are listed in the tracers column as (active, passive). 
docs/src/appendix/benchmarks.md:  GPU: Tesla V100-SXM2-32GB
docs/src/appendix/benchmarks.md:│           GPU │  (0, 0) │  9.702 ms │ 12.755 ms │ 12.458 ms │ 12.894 ms │   1.59 MiB │  12321 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │  (0, 1) │ 13.863 ms │ 13.956 ms │ 14.184 ms │ 16.297 ms │   2.20 MiB │  14294 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │  (0, 2) │ 15.166 ms │ 15.230 ms │ 15.700 ms │ 19.893 ms │   2.93 MiB │  15967 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │  (1, 0) │ 13.740 ms │ 13.838 ms │ 14.740 ms │ 22.940 ms │   2.20 MiB │  14278 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │  (2, 0) │ 15.103 ms │ 15.199 ms │ 16.265 ms │ 25.906 ms │   2.93 MiB │  15913 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │  (2, 3) │ 13.981 ms │ 18.856 ms │ 18.520 ms │ 20.519 ms │   5.56 MiB │  17974 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │  (2, 5) │ 15.824 ms │ 21.211 ms │ 21.064 ms │ 24.897 ms │   7.86 MiB │  23938 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │ (2, 10) │ 22.085 ms │ 27.236 ms │ 28.231 ms │ 38.295 ms │  15.02 MiB │  31086 │      10 │
docs/src/appendix/benchmarks.md:  Arbitrary tracers CPU to GPU speedup
docs/src/appendix/benchmarks.md:       Arbitrary tracers relative performance (GPU)
docs/src/appendix/benchmarks.md:│           GPU │  (0, 0) │      1.0 │     1.0 │     1.0 │
docs/src/appendix/benchmarks.md:│           GPU │  (0, 1) │   1.0941 │ 1.39053 │ 1.16013 │
docs/src/appendix/benchmarks.md:│           GPU │  (0, 2) │  1.19399 │ 1.85081 │ 1.29592 │
docs/src/appendix/benchmarks.md:│           GPU │  (1, 0) │  1.08489 │ 1.39037 │ 1.15883 │
docs/src/appendix/benchmarks.md:│           GPU │  (2, 0) │  1.19157 │ 1.85109 │ 1.29153 │
docs/src/appendix/benchmarks.md:│           GPU │  (2, 3) │  1.47824 │ 3.50924 │ 1.45881 │
docs/src/appendix/benchmarks.md:│           GPU │  (2, 5) │  1.66293 │ 4.95474 │ 1.94286 │
docs/src/appendix/benchmarks.md:│           GPU │ (2, 10) │  2.13524 │ 9.47276 │ 2.52301 │
docs/src/appendix/benchmarks.md:and large eddy simulation (LES) models as well as how much speedup they experience going from CPU to GPU.
docs/src/appendix/benchmarks.md:  GPU: Tesla V100-SXM2-32GB
docs/src/appendix/benchmarks.md:│           GPU │ AnisotropicBiharmonicDiffusivity │ 24.699 ms │ 24.837 ms │ 26.946 ms │ 46.029 ms │ 3.16 MiB │  29911 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │           AnisotropicDiffusivity │ 16.115 ms │ 16.184 ms │ 16.454 ms │ 18.978 ms │ 2.97 MiB │  17169 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │    AnisotropicMinimumDissipation │ 15.858 ms │ 25.856 ms │ 24.874 ms │ 26.014 ms │ 3.57 MiB │  24574 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │             IsotropicDiffusivity │ 14.442 ms │ 17.415 ms │ 17.134 ms │ 17.513 ms │ 2.99 MiB │  19135 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │                 SmagorinskyLilly │ 16.315 ms │ 23.969 ms │ 23.213 ms │ 24.059 ms │ 3.86 MiB │  24514 │      10 │
docs/src/appendix/benchmarks.md:│           GPU │              TwoDimensionalLeith │ 34.470 ms │ 34.628 ms │ 35.535 ms │ 43.798 ms │ 3.56 MiB │  45291 │      10 │
docs/src/appendix/benchmarks.md:              Turbulence closure CPU to GPU speedup
docs/src/model_setup/forcing_functions.md:However, if using the GPU, then `typeof(parameters)` may be restricted by the requirements
docs/src/model_setup/forcing_functions.md:of GPU-compiliability.
docs/src/model_setup/legacy_grids.md:architecture. By default `architecture = CPU()`. By providing `GPU()` as the `architecture` argument
docs/src/model_setup/legacy_grids.md:we can construct the grid on GPU:
docs/src/model_setup/legacy_grids.md:julia> grid = RectilinearGrid(GPU(), size = (32, 64, 256), extent = (128, 256, 512))
docs/src/model_setup/legacy_grids.md:32×64×256 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on GPU with 3×3×3 halo
docs/src/model_setup/lagrangian_particles.md:!!! warn "Lagrangian particles on GPUs"
docs/src/model_setup/lagrangian_particles.md:    Remember to use `CuArray` instead of regular `Array` when storing particle locations and properties on the GPU.
docs/src/model_setup/lagrangian_particles.md:!!! warn "Custom properties on GPUs"
docs/src/model_setup/lagrangian_particles.md:    Not all data types can be passed to GPU kernels. If you intend to advect particles on the GPU make sure
docs/src/model_setup/lagrangian_particles.md:    work on the GPU.
docs/src/model_setup/architecture.md:Passing `CPU()` or `GPU()` to the grid constructor determines whether the grid lives on a CPU or GPU.
docs/src/model_setup/architecture.md:Ideally a set up or simulation script does not need to be modified to run on a GPU but still we are smoothing
docs/src/model_setup/architecture.md:out rough edges. Generally the CPU wants `Array` objects while the GPU wants `CuArray` objects.
docs/src/model_setup/architecture.md:!!! tip "Running on GPUs"
docs/src/model_setup/architecture.md:    when running on GPUs.
docs/src/model_setup/architecture.md:    from CPU to GPU even smoother. Please 
docs/src/model_setup/boundary_conditions.md:    when running on the GPU.
docs/src/model_setup/boundary_conditions.md:    On the GPU, all functions are force-inlined by default.
docs/src/model_setup/boundary_conditions.md:When running on the GPU, `Q` must be converted to a `CuArray`.
docs/src/grids.md:3. The machine architecture (CPU, GPU, lots of CPUs or lots of GPUs); and
docs/src/grids.md:In building our first grid, we did not specify whether it should be constructed on the [`CPU`](@ref)` or [`GPU`](@ref).
docs/src/grids.md:Next we build a grid on the _GPU_ that's two-dimensional in ``x, z`` and has variably-spaced cell interfaces in the `z`-direction,
docs/src/grids.md:```jldoctest grids_gpu
docs/src/grids.md:architecture = GPU()
docs/src/grids.md:10×1×4 RectilinearGrid{Float64, Periodic, Flat, Bounded} on GPU with 3×0×3 halo
docs/src/grids.md:!!! note "GPU architecture requires a CUDA-enabled device"
docs/src/grids.md:    To run the above example and create a grid on the GPU, an Nvidia GPU has to be available
docs/src/grids.md:    and [`CUDA.jl`](https://cuda.juliagpu.org/stable/) must be working). For more information
docs/src/grids.md:    see the [`CUDA.jl` documentation](https://cuda.juliagpu.org/stable/).
docs/src/grids.md:A bit later in this tutorial, we'll give examples that illustrate how to build a grid thats [`Distributed`](@ref) across _multiple_ CPUs and GPUs.
docs/src/grids.md:* The machine architecture, or whether data is stored on the CPU, GPU, or distributed across multiple devices or nodes.
docs/src/grids.md:The positional argument `CPU()` or `GPU()`, specifies the "architecture" of the simulation.
docs/src/grids.md:By using `architecture = GPU()`, any fields constructed on `grid` store their data on
docs/src/grids.md:an Nvidia [`GPU`](@ref), if one is available. By default, the grid will be constructed on
docs/src/grids.md:which allows us to distributed computations across either CPUs or GPUs.
docs/src/quick_start.md:Fine, we'll re-run this code on the GPU. But we're a little greedy, so we'll also
docs/src/quick_start.md:```@setup gpu
docs/src/quick_start.md:```@example gpu
docs/src/quick_start.md:grid = RectilinearGrid(GPU(),
docs/src/quick_start.md:See how we did that? We passed the positional argument `GPU()` to `RectilinearGrid`.
docs/src/quick_start.md:(This only works if a GPU is available, of course, and
docs/src/quick_start.md:[CUDA.jl is configured](https://cuda.juliagpu.org/stable/installation/overview/).)
docs/src/index.md:*🌊 Fast and friendly fluid dynamics on CPUs and GPUs.*
docs/src/index.md:and hydrostatic Boussinesq equations on CPUs and GPUs.
docs/src/index.md:It runs on GPUs (wow, fast!), though we believe Oceananigans makes the biggest waves
docs/src/index.md:* The [Oceananigans wiki](https://github.com/CliMA/Oceananigans.jl/wiki), which contains practical tips for [getting started with Julia](https://github.com/CliMA/Oceananigans.jl/wiki/Installation-and-getting-started-with-Oceananigans), [accessing and using GPUs](https://github.com/CliMA/Oceananigans.jl/wiki/Oceananigans-on-GPUs), and [productive workflows when using Oceananigans](https://github.com/CliMA/Oceananigans.jl/wiki/Productive-Oceananigans-workflows-and-Julia-environments).
docs/src/index.md:  title = {Oceananigans.jl: Fast and friendly geophysical fluid dynamics on GPUs},
docs/src/index.md:1. Silvestri, S., Wagner, G. L., Constantinou, N. C., Hill, C., Campin, J.-M., Souza, A., Bishnu, S., Churavy, V., Marshall, J., and Ferrari, R. (2024) [A GPU-based ocean dynamical core for routine mesoscale-resolving climate simulations](https://doi.org/10.22541/essoar.171708158.82342448/v1), _ESS Open Archive_. DOI: [10.22541/essoar.171708158.82342448/v1](https://doi.org/10.22541/essoar.171708158.82342448/v1)
docs/src/numerical_implementation/elliptic_solvers.md:cuFFT library on the GPU. Along wall-bounded dimensions, the cosine transform is used. In particular, as the transforms
docs/src/numerical_implementation/elliptic_solvers.md:## Cosine transforms on the GPU
docs/src/numerical_implementation/elliptic_solvers.md:transforms for the GPU. We implemented the fast 1D and 2D cosine transforms described by [Makhoul80](@citet) 
test/dependencies_for_runtests.jl:using CUDA
test/dependencies_for_runtests.jl:using Oceananigans.Architectures: device, array_type # to resolve conflict with CUDA.device
test/test_halo_regions.jl:    return CUDA.@allowscalar (all(field.data[1-Hx:0,          :,          :] .== 0) &&
test/test_halo_regions.jl:    return CUDA.@allowscalar (all(data[1-Hx:0,   1:Ny,       1:Nz] .== data[Nx-Hx+1:Nx, 1:Ny,       1:Nz]) &&
test/utils_for_runtests.jl:test_child_arch() = CUDA.has_cuda() ? GPU() : CPU()
test/test_abstract_operations.jl:    return CUDA.@allowscalar a_b[2, 2, 2] == op(num1, num2)
test/test_abstract_operations.jl:    return CUDA.@allowscalar a_b_c[2, 2, 2] == num1 + num2 + num2
test/test_abstract_operations.jl:    return CUDA.@allowscalar dx_a[2, 2, 2] == 1
test/test_abstract_operations.jl:    return CUDA.@allowscalar dy_a[2, 2, 2] == 2
test/test_abstract_operations.jl:    return CUDA.@allowscalar dz_a[2, 2, 2] == 3
test/test_abstract_operations.jl:    return CUDA.@allowscalar dx_a[2, 2, 2] == 3
test/test_abstract_operations.jl:    return CUDA.@allowscalar a∇b[i, j, k] == answer
test/test_abstract_operations.jl:                    @test CUDA.@allowscalar typeof(op(ψ)[2, 2, 2]) <: Number
test/test_abstract_operations.jl:                    @test CUDA.@allowscalar typeof(d(ψ)[2, 2, 2]) <: Number
test/test_abstract_operations.jl:                    @test CUDA.@allowscalar typeof(op(ψ, ϕ)[2, 2, 2]) <: Number
test/test_abstract_operations.jl:                    @test CUDA.@allowscalar typeof(op((Center, Center, Center), ψ, ϕ, σ)[2, 2, 2]) <: Number
test/test_abstract_operations.jl:                        CUDA.@allowscalar begin
test/test_multi_region_cubed_sphere.jl:            CUDA.@allowscalar begin
test/test_multi_region_cubed_sphere.jl:                    CUDA.@allowscalar begin
test/test_multi_region_cubed_sphere.jl:            CUDA.@allowscalar begin
test/test_multi_region_cubed_sphere.jl:            end # CUDA.@allowscalar
test/test_multi_region_cubed_sphere.jl:            CUDA.@allowscalar begin
test/test_multi_region_cubed_sphere.jl:            end # CUDA.@allowscalar
test/test_multi_region_cubed_sphere.jl:            CUDA.@allowscalar begin
test/test_multi_region_cubed_sphere.jl:            end # CUDA.@allowscalar
test/test_multi_region_cubed_sphere.jl:            CUDA.@allowscalar begin
test/test_multi_region_cubed_sphere.jl:            end # CUDA.@allowscalar
test/test_regrid.jl:                CUDA.@allowscalar begin
test/test_regrid.jl:                CUDA.@allowscalar begin
test/test_regrid.jl:                CUDA.@allowscalar begin
test/test_regrid.jl:                CUDA.@allowscalar begin
test/test_regrid.jl:                CUDA.@allowscalar begin
test/test_regrid.jl:                CUDA.@allowscalar begin
test/test_regrid.jl:                CUDA.@allowscalar begin
test/test_cubed_sphere_halo_exchange.jl:    # These tests cause an undefined `Bound Access Error` on GPU's CI with the new CUDA version.
test/test_cubed_sphere_halo_exchange.jl:    if !(arch isa GPU)
test/test_cubed_sphere_halo_exchange.jl:                    CUDA.@allowscalar field_face[i, j, 1] = parse(Int, @sprintf("%d%02d%02d", face_number, i, j))
test/test_cubed_sphere_halo_exchange.jl:            CUDA.allowscalar(true)
test/test_cubed_sphere_halo_exchange.jl:            CUDA.allowscalar(false)
test/test_cubed_sphere_halo_exchange.jl:    # These tests cause an undefined `Bound Access Error` on GPU's CI with the new CUDA version.
test/test_cubed_sphere_halo_exchange.jl:    if !(arch isa GPU)
test/test_cubed_sphere_halo_exchange.jl:                    CUDA.@allowscalar u_field_face[i, j, 1] = parse(Int, @sprintf("%d%d%02d%02d", U_DIGIT, face_number, i, j))
test/test_cubed_sphere_halo_exchange.jl:                    CUDA.@allowscalar v_field_face[i, j, 1] = parse(Int, @sprintf("%d%d%02d%02d", V_DIGIT, face_number, i, j))
test/test_cubed_sphere_halo_exchange.jl:            CUDA.allowscalar(true)
test/test_cubed_sphere_halo_exchange.jl:            CUDA.allowscalar(false)
test/test_preconditioned_conjugate_gradient_solver.jl:    CUDA.@allowscalar begin
test/test_field_scans.jl:                @test CUDA.@allowscalar Txyz[1, 1, 1] ≈ 3
test/test_field_scans.jl:                @test CUDA.@allowscalar wxyz[1, 1, 1] ≈ 3
test/test_field_scans.jl:                @compute Txyz = CUDA.@allowscalar Field(Average(T, condition=T.>3))
test/test_field_scans.jl:                @compute Txy = CUDA.@allowscalar Field(Average(T, dims=(1, 2), condition=T.>3))
test/test_field_scans.jl:                @compute Tx = CUDA.@allowscalar Field(Average(T, dims=1, condition=T.>2))
test/test_field_scans.jl:                @test CUDA.@allowscalar Txyz[1, 1, 1] ≈ 3.75
test/test_field_scans.jl:                @compute wxyz = CUDA.@allowscalar Field(Average(w, condition=w.>3))
test/test_field_scans.jl:                @compute wxy = CUDA.@allowscalar Field(Average(w, dims=(1, 2), condition=w.>2))
test/test_field_scans.jl:                @compute wx = CUDA.@allowscalar Field(Average(w, dims=1, condition=w.>1))
test/test_field_scans.jl:                @test CUDA.@allowscalar wxyz[1, 1, 1] ≈ 4.25
test/test_field_scans.jl:            @test CUDA.@allowscalar Txyz[1, 1, 1] == mean(T)
test/test_field_scans.jl:            @test CUDA.@allowscalar wxyz[1, 1, 1] == mean(w)
test/test_multi_region_advection_diffusion.jl:    if architecture(grid) isa GPU
test/test_multi_region_advection_diffusion.jl:    if architecture(grid) isa GPU
test/test_multi_region_advection_diffusion.jl:    if architecture(grid) isa GPU
test/test_implicit_free_surface_solver.jl:    Δy = CUDA.@allowscalar Δyᶜᶠᶜ(i, j, k, grid)
test/test_implicit_free_surface_solver.jl:    Δz = CUDA.@allowscalar Δzᶜᶠᶜ(i, j, k, grid)
test/test_implicit_free_surface_solver.jl:    CUDA.@allowscalar u[i, j, k] = transport / (Δy * Δz)
test/test_implicit_free_surface_solver.jl:    CUDA.@allowscalar begin
test/test_hydrostatic_regression.jl:                # GPU + ImplicitFreeSurface + precompute metrics cannot be tested on sverdrup at the moment
test/test_hydrostatic_regression.jl:                if !(precompute_metrics && free_surface isa ImplicitFreeSurface && arch isa GPU) &&
test/test_multi_region_poisson_solver.jl:    if arch isa GPU
test/test_multi_region_poisson_solver.jl:    if arch isa GPU
test/test_multi_region_poisson_solver.jl:    CUDA.@allowscalar begin
test/test_vector_rotation_operators.jl:        # Note that on the GPU, there are (apparently?) larger numerical errors 
test/test_vector_rotation_operators.jl:        # Note that on the GPU, there are (apparently?) larger numerical errors 
test/test_multi_region_implicit_solver.jl:    if architecture(grid) isa GPU
test/test_diagnostics.jl:    Δz_min = CUDA.@allowscalar Oceananigans.Operators.Δzᵃᵃᶠ(1, 1, 2, grid)
test/test_diagnostics.jl:    Δx_min = CUDA.@allowscalar Oceananigans.Operators.Δxᶠᶜᵃ(1, Ny, 1, grid)
test/test_diagnostics.jl:    Δy_min = CUDA.@allowscalar Oceananigans.Operators.Δyᶜᶠᵃ(1, 1, 1, grid)
test/test_distributed_models.jl:    # Only test on CPU because we do not have a GPU pressure solver yet
test/test_conditional_reductions.jl:            @test CUDA.@allowscalar reduc!(redimm, fimm)[1, 1 , 1] == reduc(fcon, condition = (i, j, k, x, y) -> i > 3, dims = 1)[1, 1, 1]
test/test_netcdf_output_writer.jl:using CUDA
test/test_netcdf_output_writer.jl:    @test CUDA.@allowscalar ds["zC"][1] == grid.zᵃᵃᶜ[1]
test/test_netcdf_output_writer.jl:    @test CUDA.@allowscalar ds["zF"][1] == grid.zᵃᵃᶠ[1]
test/test_netcdf_output_writer.jl:    @test CUDA.@allowscalar  ds["zC"][end] == grid.zᵃᵃᶜ[Nz]
test/test_netcdf_output_writer.jl:    @test CUDA.@allowscalar  ds["zF"][end] == grid.zᵃᵃᶠ[Nz+1]  # z is Bounded
test/test_nonhydrostatic_models.jl:                arch isa GPU && topo == (Bounded, Bounded, Bounded) && continue
test/test_time_stepping.jl:    CUDA.@allowscalar model.timestepper.G⁻.u[1, 1, 1] = NaN
test/test_time_stepping.jl:    u111 = CUDA.@allowscalar model.velocities.u[1, 1, 1]
test/test_time_stepping.jl:    CUDA.@allowscalar interior(model.tracers.T)[8:24, 8:24, 8:24] .+= 0.01
test/test_time_stepping.jl:    min_div = CUDA.@allowscalar minimum(interior(div_U))
test/test_time_stepping.jl:    max_div = CUDA.@allowscalar maximum(interior(div_U))
test/test_time_stepping.jl:    max_abs_div = CUDA.@allowscalar maximum(abs, interior(div_U))
test/test_time_stepping.jl:    sum_div = CUDA.@allowscalar sum(interior(div_U))
test/test_time_stepping.jl:    sum_abs_div = CUDA.@allowscalar sum(abs, interior(div_U))
test/test_time_stepping.jl:    Tavg0 = CUDA.@allowscalar mean(interior(model.tracers.T))
test/test_time_stepping.jl:    Tavg = CUDA.@allowscalar mean(interior(model.tracers.T))
test/test_matrix_poisson_solver.jl:    CUDA.@allowscalar begin
test/test_computed_field.jl:    result = CUDA.@allowscalar ST[1, 1, 1]
test/test_computed_field.jl:                if (grid isa ImmersedBoundaryGrid) & (arch==GPU())
test/test_computed_field.jl:                if (grid isa ImmersedBoundaryGrid) & (arch==GPU())
test/regression_tests/ocean_large_eddy_simulation_regression_test.jl:    test_fields = CUDA.@allowscalar (u = Array(interior(model.velocities.u)),
test/regression_tests/hydrostatic_free_turbulence_regression_test.jl:    CUDA.allowscalar(true)
test/regression_tests/hydrostatic_free_turbulence_regression_test.jl:    CUDA.allowscalar(false)
test/regression_tests/rayleigh_benard_regression_test.jl:    test_fields =  CUDA.@allowscalar (u = Array(interior(model.velocities.u)),
test/regression_tests/rayleigh_benard_regression_test.jl:    CUDA.allowscalar(true)
test/regression_tests/rayleigh_benard_regression_test.jl:    CUDA.allowscalar(false)
test/test_multi_region_unit.jl:devices(::GPU, num) = Tuple(0 for i in 1:num)
test/test_field.jl:        @test CUDA.@allowscalar all(isapprox.(ϕ, ϕ_vals, atol=ε)) # if this isn't true, reduction tests can't pass
test/test_field.jl:        # Important to make sure no CUDA scalar operations occur!
test/test_field.jl:        CUDA.allowscalar(false)
test/test_field.jl:    # TODO: remove this allowscalar when `nodes` returns broadcastable object on GPU
test/test_field.jl:    f_max = CUDA.@allowscalar maximum(func.(xf, yf, zf))
test/test_field.jl:        CUDA.@allowscalar begin
test/test_field.jl:    CUDA.@allowscalar begin
test/test_field.jl:            if arch isa GPU
test/test_field.jl:                @test CUDA.@allowscalar field.data[1, 1, 1] == A[1, 1, 1]
test/test_field.jl:            @test CUDA.@allowscalar u[1, 2, 3] ≈ f(xu[1], yu[2], zu[3])
test/test_field.jl:            @test CUDA.@allowscalar v[1, 2, 3] ≈ f(xv[1], yv[2], zv[3])
test/test_field.jl:            @test CUDA.@allowscalar w[1, 2, 3] ≈ f(xw[1], yw[2], zw[3])
test/test_field.jl:            @test CUDA.@allowscalar c[1, 2, 3] ≈ f(xc[1], yc[2], zc[3])
test/test_field.jl:            if arch isa GPU
test/test_field.jl:            CUDA.@allowscalar @test all(cv[i, j, k] == c[i, j, k] for k in 1+1:k_top-1, j in 1:Ny, i in 1:Nx)
test/test_field.jl:            CUDA.@allowscalar @test all(cvv[i, j, k] == cv[i, j, k] for k in 1+2:k_top-2, j in 1:Ny, i in 1:Nx)
test/test_field.jl:            CUDA.@allowscalar @test all(cvvv[i, j, k] == cvv[i, j, k] for k in 1+3:k_top-3, j in 1:Ny, i in 1:Nx)
test/test_biogeochemistry.jl:using CUDA
test/test_biogeochemistry.jl:    @test CUDA.@allowscalar any(biogeochemistry.photosynthetic_active_radiation .!= 0) # update state did get called
test/test_biogeochemistry.jl:    @test CUDA.@allowscalar any(model.tracers.P .!= 1) # bgc forcing did something
test/test_dynamics.jl:    CUDA.@allowscalar begin
test/test_dynamics.jl:    CUDA.@allowscalar begin
test/test_dynamics.jl:        for arch in [CPU()] # Need some work to make these run on GPU
test/test_dynamics.jl:    # This test alone runs for 2 hours on the GPU!!!!! 
test/test_dynamics.jl:            if arch == CPU() # This test is removed on the GPU (see Issue #2647)
test/test_immersed_advection.jl:            @test CUDA.@allowscalar  _symmetric_interpolate_xᶠᵃᵃ(i+1, j, 1, ibg, scheme, c) ≈ 1.0
test/test_immersed_advection.jl:            @test CUDA.@allowscalar _biased_interpolate_xᶠᵃᵃ(i+1, j, 1, ibg, scheme, true,  c) ≈ 1.0
test/test_immersed_advection.jl:            @test CUDA.@allowscalar _biased_interpolate_xᶠᵃᵃ(i+1, j, 1, ibg, scheme, false, c) ≈ 1.0
test/test_immersed_advection.jl:            @test CUDA.@allowscalar _biased_interpolate_yᵃᶠᵃ(i, j+1, 1, ibg, scheme, true,  c) ≈ 1.0
test/test_immersed_advection.jl:            @test CUDA.@allowscalar _biased_interpolate_yᵃᶠᵃ(i, j+1, 1, ibg, scheme, false, c) ≈ 1.0
test/test_immersed_advection.jl:            @test CUDA.@allowscalar  _symmetric_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, u) ≈ 1.0
test/test_immersed_advection.jl:            @test CUDA.@allowscalar  _symmetric_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, v) ≈ 1.0
test/test_immersed_advection.jl:            @test CUDA.@allowscalar  _symmetric_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, u) ≈ 1.0
test/test_immersed_advection.jl:            @test CUDA.@allowscalar  _symmetric_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, v) ≈ 1.0
test/test_immersed_advection.jl:            @test CUDA.@allowscalar _biased_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, true,  u) ≈ 1.0
test/test_immersed_advection.jl:            @test CUDA.@allowscalar _biased_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, false, u) ≈ 1.0
test/test_immersed_advection.jl:            @test CUDA.@allowscalar _biased_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, true,  u) ≈ 1.0
test/test_immersed_advection.jl:            @test CUDA.@allowscalar _biased_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, false, u) ≈ 1.0
test/test_immersed_advection.jl:            @test CUDA.@allowscalar _biased_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, true,  v) ≈ 1.0
test/test_immersed_advection.jl:            @test CUDA.@allowscalar _biased_interpolate_xᶜᵃᵃ(i+1, j, 1, ibg, scheme, false, v) ≈ 1.0
test/test_immersed_advection.jl:            @test CUDA.@allowscalar _biased_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, true,  v) ≈ 1.0
test/test_immersed_advection.jl:            @test CUDA.@allowscalar _biased_interpolate_yᵃᶜᵃ(i, j+1, 1, ibg, scheme, false, v) ≈ 1.0
test/test_broadcasting.jl:        @test CUDA.@allowscalar all(a .== 1) 
test/test_broadcasting.jl:        @test CUDA.@allowscalar all(c .== 3)
test/test_broadcasting.jl:        @test CUDA.@allowscalar all(c .== 4)
test/test_broadcasting.jl:        CUDA.@allowscalar begin
test/test_broadcasting.jl:        CUDA.@allowscalar begin
test/test_broadcasting.jl:        CUDA.@allowscalar begin
test/test_broadcasting.jl:        CUDA.@allowscalar begin
test/test_broadcasting.jl:            @test CUDA.@allowscalar all(r .== 2)
test/test_broadcasting.jl:            @test CUDA.@allowscalar all(q .== 6)
test/test_broadcasting.jl:            @test CUDA.@allowscalar all(q .== 7)
test/dependencies_for_poisson_solvers.jl:using CUDA
test/dependencies_for_poisson_solvers.jl:    return CUDA.@allowscalar interior(∇²ϕ) ≈ R
test/dependencies_for_poisson_solvers.jl:    CUDA.@allowscalar interior(ϕ) .= real.(solver.storage)
test/test_lagrangian_particle_tracking.jl:    initial_z = CUDA.@allowscalar grid.zᵃᵃᶜ[grid.Nz-1]
test/test_lagrangian_particle_tracking.jl:    top_boundary = CUDA.@allowscalar grid.zᵃᵃᶠ[grid.Nz+1]
test/test_batched_tridiagonal_solver.jl:    # Solve the system with backslash on the CPU to avoid scalar operations on the GPU.
test/test_batched_tridiagonal_solver.jl:    # Solve the systems with backslash on the CPU to avoid scalar operations on the GPU.
test/test_grids.jl:    grid_gpu = RectilinearGrid(GPU(), topology=(Periodic, Periodic, Bounded), size=(3, 7, 9), x=(0, 1), y=(-1, 1), z=0:9)
test/test_grids.jl:    return grid_cpu == grid_gpu
test/test_grids.jl:    CUDA.allowscalar() do
test/test_grids.jl:    end # CUDA.allowscalar()
test/test_grids.jl:            if CUDA.has_cuda()
test/test_jld2_output_writer.jl:        u₀ = CUDA.@allowscalar model.velocities.u[3, 3, 3]
test/test_jld2_output_writer.jl:        v₀ = CUDA.@allowscalar model.velocities.v[3, 3, 3]
test/test_jld2_output_writer.jl:        w₀ = CUDA.@allowscalar model.velocities.w[3, 3, 3]
test/test_checkpointer.jl:    CUDA.@allowscalar begin
test/test_shallow_water_models.jl:        @test_throws MethodError ShallowWaterModel(architecture=GPU, grid=grid, gravitational_acceleration=1)
test/test_shallow_water_models.jl:                #arch isa GPU && topo == (Flat, Bounded, Flat) && continue
test/test_shallow_water_models.jl:               #arch isa GPU && topo == (Bounded, Bounded, Flat) && continue
test/test_cubed_sphere_circulation.jl:    # These tests cause an undefined `Bound Access Error` on GPU's CI with the new CUDA version.
test/test_cubed_sphere_circulation.jl:    if !(arch isa GPU)
test/test_cubed_sphere_circulation.jl:            CUDA.@allowscalar set_velocities_from_streamfunction!(u_field, v_field, ψ, arch, grid)
test/test_cubed_sphere_circulation.jl:            CUDA.allowscalar(true)
test/test_cubed_sphere_circulation.jl:            CUDA.allowscalar(false)
test/test_buoyancy.jl:    density_anomaly = CUDA.@allowscalar ρ′(2, 2, 2, grid, eos, C.T, C.S)
test/test_buoyancy.jl:    dbdx = CUDA.@allowscalar ∂x_b(2, 2, 2, grid, buoyancy, C)
test/test_buoyancy.jl:    dbdy = CUDA.@allowscalar ∂y_b(2, 2, 2, grid, buoyancy, C)
test/test_buoyancy.jl:    dbdz = CUDA.@allowscalar ∂z_b(2, 2, 2, grid, buoyancy, C)
test/test_buoyancy.jl:    α = CUDA.@allowscalar thermal_expansionᶜᶜᶜ(2, 2, 2, grid, eos, C.T, C.S)
test/test_buoyancy.jl:    α = CUDA.@allowscalar thermal_expansionᶠᶜᶜ(2, 2, 2, grid, eos, C.T, C.S)
test/test_buoyancy.jl:    α = CUDA.@allowscalar thermal_expansionᶜᶠᶜ(2, 2, 2, grid, eos, C.T, C.S)
test/test_buoyancy.jl:    α = CUDA.@allowscalar thermal_expansionᶜᶜᶠ(2, 2, 2, grid, eos, C.T, C.S)
test/test_buoyancy.jl:    β = CUDA.@allowscalar haline_contractionᶜᶜᶜ(2, 2, 2, grid, eos, C.T, C.S)
test/test_buoyancy.jl:    β = CUDA.@allowscalar haline_contractionᶠᶜᶜ(2, 2, 2, grid, eos, C.T, C.S)
test/test_buoyancy.jl:    β = CUDA.@allowscalar haline_contractionᶜᶠᶜ(2, 2, 2, grid, eos, C.T, C.S)
test/test_buoyancy.jl:    β = CUDA.@allowscalar haline_contractionᶜᶜᶠ(2, 2, 2, grid, eos, C.T, C.S)
test/runtests.jl:CUDA.allowscalar() do
test/runtests.jl:            CUDA.precompile_runtime()
test/runtests.jl:            CUDA.versioninfo()
test/runtests.jl:end #CUDA.allowscalar()
CITATION.cff:  title: "Oceananigans.jl: Fast and friendly geophysical fluid dynamics on GPUs"
ext/OceananigansMakieExt.jl:- transferring data from GPU to CPU if necessary.
README.md:  <strong>🌊 Fast and friendly ocean-flavored Julia software for simulating incompressible fluid dynamics in Cartesian and spherical shell domains on CPUs and GPUs. https://clima.github.io/OceananigansDocumentation/stable</strong>
README.md:    <img alt="Buildkite CPU+GPU build status" src="https://img.shields.io/buildkite/4d921fc17b95341ea5477fb62df0e6d9364b61b154e050a123/main?logo=buildkite&label=Buildkite%20CPU%2BGPU&style=flat-square">
README.md:and hydrostatic Boussinesq equations on CPUs and GPUs.
README.md:It runs on GPUs (wow, [fast!](http://arxiv.org/abs/2309.06662)), though we believe Oceananigans makes the biggest waves
README.md:But there's more: changing `CPU()` to `GPU()` makes this code run on a CUDA-enabled Nvidia GPU.
README.md:Below, you'll find movies from GPU simulations along with CPU and GPU [performance benchmarks](https://github.com/clima/Oceananigans.jl#performance-benchmarks).
README.md:* The [Oceananigans wiki](https://github.com/CliMA/Oceananigans.jl/wiki) contains practical tips for [getting started with Julia](https://github.com/CliMA/Oceananigans.jl/wiki/Installation-and-getting-started-with-Oceananigans), [accessing and using GPUs](https://github.com/CliMA/Oceananigans.jl/wiki/Accessing-GPUs-and-using-Oceananigans-on-GPUs), and [productive workflows when using Oceananigans](https://github.com/CliMA/Oceananigans.jl/wiki/Productive-Oceananigans-workflows-and-Julia-environments).
README.md:  title = {{Oceananigans.jl: Fast and friendly geophysical fluid dynamics on GPUs}},
README.md:This is not really a fair comparison as we haven't parallelized across all the CPU's cores so we will revisit these benchmarks once Oceananigans.jl can run on multiple CPUs and GPUs.
README.md:To make full use of or fully saturate the computing power of a GPU such as an Nvidia Tesla V100 or
README.md:GPU register pressure, `Float32` models may not provide much of a speedup so the main benefit becomes
paper/paper.bib:  title   = {Effective {Extensible} {Programming}: {Unleashing} {Julia} on {GPUs}},
paper/paper.md:title: 'Oceananigans.jl: Fast and friendly geophysical fluid dynamics on GPUs'
paper/paper.md:  - gpu
paper/paper.md:simulation of incompressible, stratified, rotating fluid flows on CPUs and GPUs.
paper/paper.md:the CPU or GPU with Julia’s native GPU compiler [@Besard2019]. Because Julia is
paper/paper.md:or `architecture=GPU()` will execute the model on the CPU or GPU. By pinning a
paper/paper.md:Performance benchmarks show significant speedups when running on a GPU. Large
paper/paper.md:simulations on an Nvidia Tesla V100 GPU require ~1 nanosecond per grid point per
paper/paper.md:iteration. GPU simulations are therefore roughly 3x more cost-effective
paper/paper.md:than CPU simulations on cloud computing platforms such as Google Cloud. A GPU
paper/paper.md:``Oceananigans.jl`` is continuously tested on CPUs and GPUs with unit tests,
paper/paper.md:parallelism with CUDA-aware MPI as well as topography.
paper/paper.md:with complex boundaries on parallel CPU and GPU architectures. ``Oceananigans.jl``
validation/stratified_couette_flow/stratified_couette_flow.jl:function simulate_stratified_couette_flow(; Nxy, Nz, arch=GPU(), h=1, U_wall=1,
validation/solid_body_rotation/rossby_haurwitz.jl:filepath_w = run_rossby_haurwitz(architecture=GPU(), Nx=512, Ny=256, advection_scheme=WENO(vector_invariant=VelocityStencil()), prefix = "WENOVectorInvariantVel")
validation/solid_body_rotation/rossby_haurwitz.jl:filepath_w = run_rossby_haurwitz(architecture=GPU(), Nx=512, Ny=256, advection_scheme=WENO(vector_invariant=VorticityStencil()), prefix = "WENOVectorInvariantVort")
validation/solid_body_rotation/rossby_haurwitz.jl:filepath_w = run_rossby_haurwitz(architecture=GPU(), Nx=512, Ny=256, advection_scheme=VectorInvariant(), prefix = "VectorInvariant")
validation/shallow_water_model/single_geostrophic_vortex.jl:    grid = RectilinearGrid(GPU(); size = (Nx, Ny), x = (-0.5, 0.5), y = (-0.5, 0.5),
validation/shallow_water_model/single_geostrophic_vortex.jl:    grid = RectilinearGrid(GPU(); size = (Nx, Ny), x = (-0.5, 0.5), y = (-0.5, 0.5),
validation/shallow_water_model/shallow_water_Bickley_jet.jl:grid = RectilinearGrid(GPU(), size = (Nx, Ny),
validation/shallow_water_model/vortex_merger.jl:arch = GPU()
validation/shallow_water_model/near_global_shallow_water_quarter_degree.jl:using CUDA: @allowscalar, device!
validation/shallow_water_model/near_global_shallow_water_quarter_degree.jl:arch = GPU()
validation/advection/gaussian_bump_advection.jl:arch = GPU()
validation/immersed_boundaries/immersed_bickley_jet.jl:using CUDA
validation/immersed_boundaries/immersed_bickley_jet.jl:        experiment_name = run_bickley_jet(arch=GPU(), momentum_advection=advection, Nh=Nx)
validation/immersed_boundaries/2D_rough_rayleighbenard.jl:    grid = RectilinearGrid(GPU(), Float64,
validation/immersed_boundaries/nonlinear_topography.jl:    grid = RectilinearGrid(GPU(), Float64,
validation/immersed_boundaries/internal_tide.jl:using CUDA
validation/immersed_boundaries/internal_tide.jl:grid = RectilinearGrid(GPU(), size=(512, 256), 
validation/immersed_boundaries/internal_tide.jl:    Δt = CUDA.@allowscalar 0.1 * minimum(grid.Δxᶜᵃᵃ) / gravity_wave_speed
validation/convergence_tests/two_dimensional_vortex_advection.jl:using CUDA
validation/convergence_tests/two_dimensional_vortex_advection.jl:arch = CUDA.has_cuda() ? GPU() : CPU()
validation/convergence_tests/one_dimensional_advection_schemes.jl:using CUDA
validation/convergence_tests/one_dimensional_advection_schemes.jl:arch = CUDA.has_cuda() ? GPU() : CPU()
validation/convergence_tests/one_dimensional_cosine_advection_diffusion.jl:using CUDA
validation/convergence_tests/one_dimensional_cosine_advection_diffusion.jl:arch = CUDA.has_cuda() ? GPU() : CPU()
validation/convergence_tests/one_dimensional_gaussian_advection_diffusion.jl:using CUDA
validation/convergence_tests/one_dimensional_gaussian_advection_diffusion.jl:arch = CUDA.has_cuda() ? GPU() : CPU()
validation/convergence_tests/Manifest.toml:[[CUDA]]
validation/convergence_tests/Manifest.toml:deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "DataStructures", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
validation/convergence_tests/Manifest.toml:[[GPUArrays]]
validation/convergence_tests/Manifest.toml:[[GPUCompiler]]
validation/convergence_tests/analyze_forced_fixed_slip.jl:arch = CUDA.has_cuda() ? GPU() : CPU()
validation/convergence_tests/analyze_taylor_green.jl:arch = CUDA.has_cuda() ? GPU() : CPU()
validation/convergence_tests/point_exponential_decay.jl:using CUDA
validation/convergence_tests/point_exponential_decay.jl:arch = CUDA.has_cuda() ? GPU() : CPU()
validation/convergence_tests/two_dimensional_diffusion.jl:using CUDA
validation/convergence_tests/two_dimensional_diffusion.jl:arch = CUDA.has_cuda() ? GPU() : CPU()
validation/convergence_tests/run_taylor_green.jl:using CUDA
validation/convergence_tests/run_taylor_green.jl:arch = CUDA.has_cuda() ? GPU() : CPU()
validation/convergence_tests/run_forced_free_slip.jl:using CUDA
validation/convergence_tests/run_forced_free_slip.jl:arch = CUDA.has_cuda() ? GPU() : CPU()
validation/convergence_tests/two_dimensional_advection_schemes.jl:using CUDA
validation/convergence_tests/two_dimensional_advection_schemes.jl:arch = CUDA.has_cuda() ? GPU() : CPU()
validation/convergence_tests/Project.toml:CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
validation/convergence_tests/analyze_forced_free_slip.jl:arch = CUDA.has_cuda() ? GPU() : CPU()
validation/convergence_tests/run_forced_fixed_slip.jl:using CUDA
validation/convergence_tests/run_forced_fixed_slip.jl:arch = CUDA.has_cuda() ? GPU() : CPU()
validation/bickley_jet/bickley_jet.jl:        experiment_name = run_bickley_jet(arch=GPU(), momentum_advection=advection, Nh=Nx)
validation/bickley_jet/immersed_bickley_jet.jl:using CUDA
validation/bickley_jet/spherical_bickley_jet.jl:using CUDA: device!
validation/bickley_jet/spherical_bickley_jet.jl:        experiment_name = run_bickley_jet(arch=GPU(), momentum_advection=advection, Nh=Nx)
validation/open_boundaries/cylinder.jl:architecture = GPU()
validation/stokes_drift/Langmuir_with_Stokes_y_jet.jl:# The `const` declarations ensure that Stokes drift functions compile on the GPU.
validation/stokes_drift/Langmuir_with_Stokes_y_jet.jl:# To run this example on the GPU, include `GPU()` in the
validation/stokes_drift/Manifest.toml:    ArrayInterfaceCUDAExt = "CUDA"
validation/stokes_drift/Manifest.toml:    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
validation/stokes_drift/Manifest.toml:    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
validation/stokes_drift/Manifest.toml:    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
validation/stokes_drift/Manifest.toml:[[deps.CUDA]]
validation/stokes_drift/Manifest.toml:deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "Statistics", "UnsafeAtomicsLLVM"]
validation/stokes_drift/Manifest.toml:    [deps.CUDA.extensions]
validation/stokes_drift/Manifest.toml:[[deps.CUDA_Driver_jll]]
validation/stokes_drift/Manifest.toml:[[deps.CUDA_Runtime_Discovery]]
validation/stokes_drift/Manifest.toml:[[deps.CUDA_Runtime_jll]]
validation/stokes_drift/Manifest.toml:deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
validation/stokes_drift/Manifest.toml:[[deps.GPUArrays]]
validation/stokes_drift/Manifest.toml:deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
validation/stokes_drift/Manifest.toml:[[deps.GPUArraysCore]]
validation/stokes_drift/Manifest.toml:[[deps.GPUCompiler]]
validation/stokes_drift/Manifest.toml:    AMDGPUExt = "AMDGPU"
validation/stokes_drift/Manifest.toml:    CUDAExt = "CUDA"
validation/stokes_drift/Manifest.toml:    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
validation/stokes_drift/Manifest.toml:    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
validation/stokes_drift/Manifest.toml:deps = ["Adapt", "CUDA", "Crayons", "CubedSphere", "Dates", "Distances", "DocStringExtensions", "FFTW", "Glob", "IncompleteLU", "InteractiveUtils", "IterativeSolvers", "JLD2", "KernelAbstractions", "LinearAlgebra", "Logging", "MPI", "NCDatasets", "OffsetArrays", "OrderedCollections", "PencilArrays", "PencilFFTs", "Pkg", "Printf", "Random", "Rotations", "SeawaterPolynomials", "SparseArrays", "Statistics", "StructArrays"]
validation/stokes_drift/Manifest.toml:weakdeps = ["CUDA"]
validation/stokes_drift/Manifest.toml:    StridedViewsCUDAExt = "CUDA"
validation/stokes_drift/Manifest.toml:deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
validation/stokes_drift/Langmuir_with_Stokes_x_jet.jl:# The `const` declarations ensure that Stokes drift functions compile on the GPU.
validation/stokes_drift/Langmuir_with_Stokes_x_jet.jl:# To run this example on the GPU, include `GPU()` in the
validation/vertical_mixing_closures/gpu_tkevd_ensemble.jl:using CUDA
validation/vertical_mixing_closures/gpu_tkevd_ensemble.jl:model = HydrostaticFreeSurfaceModel(architecture = GPU(),
validation/vertical_mixing_closures/gpu_tkevd_ensemble.jl:    CUDA.@sync time_step!(model, 1e-9)
validation/mesoscale_turbulence/baroclinic_adjustment.jl:architecture  = GPU()
validation/mesoscale_turbulence/eady_turbulence.jl:# to ensure this operation is efficient on the GPU.
validation/mesoscale_turbulence/eddying_channel.jl:architecture = GPU()
validation/implicit_free_surface/geostrophic_adjustment_test.jl:        if arch isa GPU
validation/implicit_free_surface/geostrophic_adjustment_test.jl:archs = [Oceananigans.CPU()] #, Oceananigans.GPU()]
validation/multi_region/multi_region_internal_tide.jl:using CUDA
validation/multi_region/multi_region_near_global_quarter_degree.jl:using CUDA: @allowscalar, device!
validation/multi_region/multi_region_near_global_quarter_degree.jl:arch = GPU()
validation/distributed_simulations/distributed_scaling/distributed_nonhydrostatic_simulation.jl:    arch  = Distributed(GPU(); partition = Partition(ranks...))
validation/distributed_simulations/distributed_scaling/distributed_hydrostatic_simulation.jl:    arch  = Distributed(GPU(), FT; partition = Partition(ranks...))
validation/distributed_simulations/distributed_scaling/job_script.sh:# Upload modules: cuda and cuda-aware mpi
validation/distributed_simulations/distributed_scaling/job_script.sh:# module add cuda/11.4
validation/distributed_simulations/distributed_scaling/job_script.sh:# module load openmpi/3.1.6-cuda-pmi-ucx-slurm-jhklron
validation/distributed_simulations/distributed_scaling/job_script.sh:export JULIA_CUDA_MEMORY_POOL=none
validation/distributed_simulations/distributed_scaling/job_script.sh:export CUDA_VISIBLE_DEVICES=0,1,2,3
validation/distributed_simulations/distributed_scaling/job_script.sh:   NSYS="nsys profile --trace=nvtx,cuda,mpi --output=${SIMULATION}_RX${RX}_RY${RY}_NX${NX}_NY${NY}"
validation/distributed_simulations/distributed_scaling/run_tests.sh:# 2) the NGPUS_PER_NODE variable is correct (in this file)
validation/distributed_simulations/distributed_scaling/run_tests.sh:# 	(run these lines in a gpu node substituting modules and paths)
validation/distributed_simulations/distributed_scaling/run_tests.sh:# 	$ module load my_cuda_module
validation/distributed_simulations/distributed_scaling/run_tests.sh:# 	$ module load my_cuda_aware_mpi_module
validation/distributed_simulations/distributed_scaling/run_tests.sh:# 7) The system has at least max(RX) * max(RY) gpus
validation/distributed_simulations/distributed_scaling/run_tests.sh:# Number of gpus per node
validation/distributed_simulations/distributed_scaling/run_tests.sh:export NGPUS_PER_NODE=4
validation/distributed_simulations/distributed_scaling/run_tests.sh:		export NNODES=$((RANKS / NGPUS_PER_NODE))
validation/distributed_simulations/distributed_scaling/run_tests.sh:		export NTASKS=$NGPUS_PER_NODE
validation/distributed_simulations/distributed_scaling/run_tests.sh:		sbatch -N ${NNODES} --gres=gpu:${NTASKS} --ntasks-per-node=${NTASKS} job_script.sh
validation/regridding/latitude_longitude_regridding.jl:arch = GPU()
validation/regridding/rectilinear_regridding.jl:arch = GPU()
Project.toml:CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Project.toml:CUDA = "4.1.1, 5"
benchmark/benchmark_abstract_operations.jl:using CUDA
benchmark/benchmark_abstract_operations.jl:Archs = has_cuda() ? [CPU, GPU] : [CPU]
benchmark/benchmark_abstract_operations.jl:            @sync_gpu compute!($computed_field)
benchmark/benchmark_multithreading_single.jl:    @sync_gpu time_step!($model, 1)
benchmark/benchmark_shallow_water_model.jl:using CUDA
benchmark/benchmark_shallow_water_model.jl:        CUDA.@sync blocking=true time_step!($model, 1)
benchmark/benchmark_shallow_water_model.jl:Architectures = has_cuda() ? [CPU, GPU] : [CPU]
benchmark/benchmark_shallow_water_model.jl:gpu_times = zeros(Float64, plot_num)
benchmark/benchmark_shallow_water_model.jl:    gpu_times[i] = mean(suite[plot_keys[i+plot_num]].times) / 1.0e6
benchmark/benchmark_shallow_water_model.jl:          xlabel="Nx", ylabel="Times (ms)", title="Shallow Water Benchmarks: CPU vs GPU")
benchmark/benchmark_shallow_water_model.jl:plot!(plt, Ns, gpu_times, lw=4, label="gpu")
benchmark/benchmark_shallow_water_model.jl:plt2 = plot(Ns, cpu_times./gpu_times, lw=4, xaxis=:log2, legend=:none,
benchmark/benchmark_shallow_water_model.jl:            xlabel="Nx", ylabel="Speedup Ratio", title="Shallow Water Benchmarks: CPU/GPU")
benchmark/benchmark_shallow_water_model.jl:if GPU in Architectures
benchmark/benchmark_shallow_water_model.jl:    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
benchmark/benchmark_shallow_water_model.jl:    benchmarks_pretty_table(df_Δ, title="Shallow water model CPU to GPU speedup")
benchmark/distributed_shallow_water_model_threaded.jl:using CUDA
benchmark/distributed_shallow_water_model_threaded.jl:    @sync_gpu time_step!($model, 1)
benchmark/distributed_shallow_water_model_threaded.jl:    #CUDA.@sync blocking=true time_step!($model, 1)
benchmark/benchmark_transforms.jl:using CUDA
benchmark/benchmark_transforms.jl:function benchmark_fft(::Type{GPU}, N, dims; FT=Float64, planner_flag=FFTW.PATIENT)
benchmark/benchmark_transforms.jl:    # Cannot do CUDA FFTs along non-batched dims so dim=2 must
benchmark/benchmark_transforms.jl:        FFT! = CUDA.CUFFT.plan_fft!(A, 1)
benchmark/benchmark_transforms.jl:            CUDA.@sync begin
benchmark/benchmark_transforms.jl:        FFT! = CUDA.CUFFT.plan_fft!(A, dims)
benchmark/benchmark_transforms.jl:            CUDA.@sync ($FFT! * $A)
benchmark/benchmark_transforms.jl:Architectures = has_cuda() ? [CPU, GPU] : [CPU]
benchmark/benchmark_turbulence_closures.jl:using CUDA
benchmark/benchmark_turbulence_closures.jl:        @sync_gpu time_step!($model, 1)
benchmark/benchmark_turbulence_closures.jl:Architectures = has_cuda() ? [CPU, GPU] : [CPU]
benchmark/benchmark_turbulence_closures.jl:if GPU in Architectures
benchmark/benchmark_turbulence_closures.jl:    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
benchmark/benchmark_turbulence_closures.jl:    benchmarks_pretty_table(df_Δ, title="Turbulence closure CPU to GPU speedup")
benchmark/Manifest.toml:    ArrayInterfaceCUDAExt = "CUDA"
benchmark/Manifest.toml:    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
benchmark/Manifest.toml:    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
benchmark/Manifest.toml:    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
benchmark/Manifest.toml:[[deps.CUDA]]
benchmark/Manifest.toml:deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "StaticArrays", "Statistics"]
benchmark/Manifest.toml:    [deps.CUDA.extensions]
benchmark/Manifest.toml:    [deps.CUDA.weakdeps]
benchmark/Manifest.toml:[[deps.CUDA_Driver_jll]]
benchmark/Manifest.toml:[[deps.CUDA_Runtime_Discovery]]
benchmark/Manifest.toml:[[deps.CUDA_Runtime_jll]]
benchmark/Manifest.toml:deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
benchmark/Manifest.toml:[[deps.GPUArrays]]
benchmark/Manifest.toml:deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
benchmark/Manifest.toml:[[deps.GPUArraysCore]]
benchmark/Manifest.toml:[[deps.GPUCompiler]]
benchmark/Manifest.toml:    AMDGPUExt = "AMDGPU"
benchmark/Manifest.toml:    CUDAExt = "CUDA"
benchmark/Manifest.toml:    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
benchmark/Manifest.toml:    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
benchmark/Manifest.toml:deps = ["Adapt", "CUDA", "Crayons", "CubedSphere", "Dates", "Distances", "DocStringExtensions", "FFTW", "Glob", "IncompleteLU", "InteractiveUtils", "IterativeSolvers", "JLD2", "KernelAbstractions", "LinearAlgebra", "Logging", "MPI", "NCDatasets", "OffsetArrays", "OrderedCollections", "PencilArrays", "PencilFFTs", "Pkg", "Printf", "Random", "Rotations", "SeawaterPolynomials", "SparseArrays", "Statistics", "StructArrays"]
benchmark/Manifest.toml:    PencilArraysAMDGPUExt = ["AMDGPU"]
benchmark/Manifest.toml:    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
benchmark/Manifest.toml:weakdeps = ["CUDA"]
benchmark/Manifest.toml:    StridedViewsCUDAExt = "CUDA"
benchmark/Manifest.toml:weakdeps = ["Adapt", "GPUArraysCore", "SparseArrays", "StaticArrays"]
benchmark/Manifest.toml:    StructArraysGPUArraysCoreExt = "GPUArraysCore"
benchmark/benchmark_lat_lon_hydrostatic_model.jl:using CUDA
benchmark/benchmark_lat_lon_hydrostatic_model.jl:for arch in [ has_cuda() ? [CPU(), GPU()] : [CPU()] ]
benchmark/benchmark_topologies.jl:using CUDA
benchmark/benchmark_topologies.jl:        @sync_gpu time_step!($model, 1)
benchmark/benchmark_topologies.jl:Architectures = has_cuda() ? [CPU, GPU] : [CPU]
benchmark/benchmark_topologies.jl:if GPU in Architectures
benchmark/benchmark_topologies.jl:    df = gpu_speedups_suite(suite) |> speedups_dataframe
benchmark/benchmark_topologies.jl:    benchmarks_pretty_table(df, title="Topologies CPU to GPU speedup")
benchmark/benchmark_lagrangian_particle_tracking.jl:using CUDA
benchmark/benchmark_lagrangian_particle_tracking.jl:        @sync_gpu time_step!($model, 1)
benchmark/benchmark_lagrangian_particle_tracking.jl:Architectures = has_cuda() ? [CPU, GPU] : [CPU]
benchmark/benchmark_lagrangian_particle_tracking.jl:if GPU in Architectures
benchmark/benchmark_lagrangian_particle_tracking.jl:    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
benchmark/benchmark_lagrangian_particle_tracking.jl:    benchmarks_pretty_table(df_Δ, title="Lagrangian particle tracking CPU to GPU speedup")
benchmark/benchmark_fourier_tridiagonal_poisson_solver.jl:using CUDA
benchmark/benchmark_fourier_tridiagonal_poisson_solver.jl:        @sync_gpu solve_poisson_equation!($solver)
benchmark/benchmark_fourier_tridiagonal_poisson_solver.jl:Architectures = has_cuda() ? [CPU, GPU] : [CPU]
benchmark/benchmark_fourier_tridiagonal_poisson_solver.jl:if GPU in Architectures
benchmark/benchmark_fourier_tridiagonal_poisson_solver.jl:    df = gpu_speedups_suite(suite) |> speedups_dataframe
benchmark/benchmark_fourier_tridiagonal_poisson_solver.jl:    benchmarks_pretty_table(df, title="Fourier-tridiagonal Poisson solver CPU to GPU speedup")
benchmark/README.md:Most scripts benchmark one feature (e.g. advection schemes, arbitrary tracers). If your machine contains a CUDA-compatible GPU, benchmarks will also run on the GPU. Tables with benchmark results will be printed (and each table will also be saved to an HTML file).
benchmark/benchmark_equations_of_state.jl:using CUDA
benchmark/benchmark_equations_of_state.jl:        @sync_gpu time_step!($model, 1)
benchmark/benchmark_equations_of_state.jl:Architectures = has_cuda() ? [CPU, GPU] : [CPU]
benchmark/benchmark_equations_of_state.jl:if GPU in Architectures
benchmark/benchmark_equations_of_state.jl:    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
benchmark/benchmark_equations_of_state.jl:    benchmarks_pretty_table(df_Δ, title="Equation of state CPU to GPU speedup")
benchmark/benchmark_two_dimensional_models.jl:for arch in (CPU(), GPU())
benchmark/benchmark_advection_schemes.jl:using CUDA
benchmark/benchmark_advection_schemes.jl:        @sync_gpu time_step!($model, 1)
benchmark/benchmark_advection_schemes.jl:Architectures = has_cuda() ? [CPU, GPU] : [CPU]
benchmark/benchmark_advection_schemes.jl:if GPU in Architectures
benchmark/benchmark_advection_schemes.jl:    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
benchmark/benchmark_advection_schemes.jl:    benchmarks_pretty_table(df_Δ, title="Advection schemes CPU to GPU speedup")
benchmark/benchmark_models_stepping.jl:using CUDA
benchmark/benchmark_models_stepping.jl:        @sync_gpu time_step!($model, 1)
benchmark/benchmark_models_stepping.jl:        @sync_gpu time_step!($model, 0.001)
benchmark/benchmark_models_stepping.jl:        @sync_gpu time_step!($model, 1)
benchmark/benchmark_models_stepping.jl:Architectures = has_cuda() ? [CPU, GPU] : [CPU]
benchmark/benchmark_tracers.jl:using CUDA
benchmark/benchmark_tracers.jl:        @sync_gpu time_step!($model, 1)
benchmark/benchmark_tracers.jl:Architectures = has_cuda() ? [CPU, GPU] : [CPU]
benchmark/benchmark_tracers.jl:if GPU in Architectures
benchmark/benchmark_tracers.jl:    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
benchmark/benchmark_tracers.jl:    benchmarks_pretty_table(df_Δ, title="Arbitrary tracers CPU to GPU speedup")
benchmark/benchmark_spai_preconditioner.jl:using CUDA
benchmark/benchmark_spai_preconditioner.jl:            CUDA.@sync blocking = true sparse_approximate_inverse($matrix, ε = $ε, nzrel = $nzrel)
benchmark/benchmark_spai_preconditioner.jl:            CUDA.@sync blocking = true inv(Array($matrix))
benchmark/benchmark_vertically_stretched_nonhydrostatic_model.jl:using CUDA
benchmark/benchmark_vertically_stretched_nonhydrostatic_model.jl:        @sync_gpu time_step!($model, 1)
benchmark/benchmark_vertically_stretched_nonhydrostatic_model.jl:Architectures = has_cuda() ? [CPU, GPU] : [CPU]
benchmark/benchmark_vertically_stretched_nonhydrostatic_model.jl:if GPU in Architectures
benchmark/benchmark_vertically_stretched_nonhydrostatic_model.jl:    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
benchmark/benchmark_vertically_stretched_nonhydrostatic_model.jl:    benchmarks_pretty_table(df_Δ, title="Vertically-stretched nonhydrostatic model CPU to GPU speedup")
benchmark/benchmarkable_nonhydrostatic_model.jl:using CUDA
benchmark/benchmarkable_nonhydrostatic_model.jl:Architectures = has_cuda() ? [CPU, GPU] : [CPU]
benchmark/benchmarkable_nonhydrostatic_model.jl:        @sync_gpu time_step!($model, 1)
benchmark/distributed_nonhydrostatic_model_threaded.jl:using CUDA
benchmark/distributed_nonhydrostatic_model_threaded.jl:    @sync_gpu time_step!($model, 1)
benchmark/distributed_nonhydrostatic_model_threaded.jl:    #CUDA.@sync blocking=true time_step!($model, 1)
benchmark/benchmark_fft_based_poisson_solvers.jl:using CUDA
benchmark/benchmark_fft_based_poisson_solvers.jl:        @sync_gpu solve_poisson_equation!($solver)
benchmark/benchmark_fft_based_poisson_solvers.jl:Architectures = has_cuda() ? [CPU, GPU] : [CPU]
benchmark/benchmark_fft_based_poisson_solvers.jl:if GPU in Architectures
benchmark/benchmark_fft_based_poisson_solvers.jl:    df = gpu_speedups_suite(suite) |> speedups_dataframe
benchmark/benchmark_fft_based_poisson_solvers.jl:    benchmarks_pretty_table(df, title="FFT-based Poisson solver CPU to GPU speedup")
benchmark/Project.toml:CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
benchmark/Project.toml:CUDA = "^5"
benchmark/distributed_shallow_water_model_mpi.jl: #assigns one GPU per rank, could increase efficiency but must have enough GPUs
benchmark/distributed_shallow_water_model_mpi.jl: #CUDA.device!(local_rank)
benchmark/distributed_shallow_water_model_mpi.jl:    @sync_gpu time_step!($model, 1)
benchmark/benchmark_hydrostatic_model.jl:using CUDA
benchmark/benchmark_hydrostatic_model.jl:    CUDA.@allowscalar u[imid, jmid, 1] = 1
benchmark/benchmark_hydrostatic_model.jl:    (GPU, :RectilinearGrid)       => RectilinearGrid(GPU(), size=(Nx, Ny, 1), extent=(1, 1, 1)),
benchmark/benchmark_hydrostatic_model.jl:    (GPU, :LatitudeLongitudeGrid) => LatitudeLongitudeGrid(GPU(), size=(Nx, Ny, 1), longitude=(-160, 160), latitude=(-80, 80), z=(-1, 0), precompute_metrics=true),
benchmark/benchmark_hydrostatic_model.jl:# (GPU, :ConformalCubedSphereGrid)     => ConformalCubedSphereGrid(datadep"cubed_sphere_510_grid/cubed_sphere_510_grid.jld2", Nz=1, z=(-1, 0), architecture=GPU()),
benchmark/benchmark_hydrostatic_model.jl:        CUDA.@sync blocking = true time_step!($model, $Δt)
benchmark/benchmark_hydrostatic_model.jl:#architectures = has_cuda() ? [GPU] : [CPU]
benchmark/benchmark_multi_GPU.jl:using CUDA
benchmark/benchmark_multi_GPU.jl:simulation_serial = run_solid_body_rotation(Nx=Nx, Ny=Ny, architecture=GPU()) # 104 ms
benchmark/benchmark_multi_GPU.jl:simulation_paral1 = run_solid_body_rotation(Nx=mult1*Nx, Ny=Ny, dev = (0, 1), architecture=GPU()) # 85 ms
benchmark/benchmark_multi_GPU.jl:simulation_paral2 = run_solid_body_rotation(Nx=mult2*Nx, Ny=Ny, dev = (0, 1, 2), architecture=GPU())  # 80 ms
benchmark/benchmark_multi_GPU.jl:CUDA.device!(0)
benchmark/benchmark_multi_GPU.jl:    CUDA.@sync time_step!(simulation_serial.model, 1)
benchmark/benchmark_multi_GPU.jl:    CUDA.@sync time_step!(simulation_paral1.model, 1)
benchmark/benchmark_multi_GPU.jl:    CUDA.@sync time_step!(simulation_paral2.model, 1)
benchmark/src/Benchmarks.jl:export @sync_gpu,
benchmark/src/Benchmarks.jl:       gpu_speedups_suite,
benchmark/src/Benchmarks.jl:using CUDA
benchmark/src/Benchmarks.jl:using Oceananigans.Architectures: CPU, GPU
benchmark/src/Benchmarks.jl:using Oceananigans.Utils: oceananigans_versioninfo, versioninfo_with_gpu
benchmark/src/Benchmarks.jl:macro sync_gpu(expr)
benchmark/src/Benchmarks.jl:    return CUDA.has_cuda() ? :($(esc(CUDA.@sync expr))) : :($(esc(expr)))
benchmark/src/Benchmarks.jl:    println(versioninfo_with_gpu())
benchmark/src/Benchmarks.jl:    CUDA.has_cuda_gpu() && println(CUDA.versioninfo())
benchmark/src/Benchmarks.jl:is_arch_type(e) = e == CPU || e == GPU
benchmark/src/Benchmarks.jl:gpu_case(case) = Tuple(is_arch_type(e) ? GPU : e for e in case)
benchmark/src/Benchmarks.jl:function gpu_speedups_suite(suite)
benchmark/src/Benchmarks.jl:        case_gpu = gpu_case(case)
benchmark/src/Benchmarks.jl:            suite_speedup[case_speedup] = ratio(median(suite[case_gpu]), median(suite[case_cpu]))
benchmark/benchmark_time_steppers.jl:using CUDA
benchmark/benchmark_time_steppers.jl:        @sync_gpu time_step!($model, 1)
benchmark/benchmark_time_steppers.jl:Architectures = has_cuda() ? [CPU, GPU] : [CPU]
benchmark/benchmark_time_steppers.jl:if GPU in Architectures
benchmark/benchmark_time_steppers.jl:    df_Δ = gpu_speedups_suite(suite) |> speedups_dataframe
benchmark/benchmark_time_steppers.jl:    benchmarks_pretty_table(df_Δ, title="Time stepping CPU to GPU speedup")
examples/ocean_wind_mixing_and_convection.jl:# # [Wind- and convection-driven mixing in an ocean surface boundary layer](@id gpu_example)
examples/ocean_wind_mixing_and_convection.jl:# * To change the architecture to `GPU`, replace `CPU()` with `GPU()` inside the
examples/langmuir_turbulence.jl:# The `const` declarations ensure that Stokes drift functions compile on the GPU.
examples/langmuir_turbulence.jl:# To run this example on the GPU, include `GPU()` in the `RectilinearGrid` constructor above.
CONTRIBUTING.md:* Report the Oceananigans version, Julia version, machine (especially if using a GPU) and any other possibly useful details of the computational environment in which the bug was created.
src/OutputReaders/field_time_series.jl:using CUDA: @allowscalar
src/OutputReaders/field_time_series.jl:##### Minimal implementation of FieldTimeSeries for use in GPU kernels
src/OutputReaders/field_time_series.jl:struct GPUAdaptedFieldTimeSeries{LX, LY, LZ, TI, K, ET, D, χ} <: AbstractField{LX, LY, LZ, Nothing, ET, 4}
src/OutputReaders/field_time_series.jl:    function GPUAdaptedFieldTimeSeries{LX, LY, LZ}(data::D,
src/OutputReaders/field_time_series.jl:    return GPUAdaptedFieldTimeSeries{LX, LY, LZ}(adapt(to, fts.data),
src/OutputReaders/field_time_series.jl:const GPUFTS{LX, LY, LZ, TI, K} = GPUAdaptedFieldTimeSeries{LX, LY, LZ, TI, K} where {LX, LY, LZ, TI, K}
src/OutputReaders/field_time_series.jl:const FlavorOfFTS{LX, LY, LZ, TI, K} = Union{GPUFTS{LX, LY, LZ, TI, K},
src/OutputReaders/field_time_series.jl:const GPUFTSBC = BoundaryCondition{<:Any, <:GPUAdaptedFieldTimeSeries}
src/OutputReaders/field_time_series.jl:const FTSBC = Union{CPUFTSBC, GPUFTSBC}
src/OutputReaders/set_field_time_series.jl:        # Potentially transfer from CPU to GPU
src/OutputReaders/field_time_series_indexing.jl:# for ranges. if `times` is a vector that resides on the GPU, it has to be moved to the CPU for safe indexing.
src/OutputReaders/field_time_series_indexing.jl:function cpu_interpolating_time_indices(::GPU, times::AbstractVector, time_indexing, t)
src/AbstractOperations/unary_operations.jl:##### GPU capabilities
src/AbstractOperations/unary_operations.jl:"Adapt `UnaryOperation` to work on the GPU via CUDAnative and CUDAdrv."
src/AbstractOperations/derivatives.jl:##### GPU capabilities
src/AbstractOperations/derivatives.jl:"Adapt `Derivative` to work on the GPU."
src/AbstractOperations/AbstractOperations.jl:using CUDA
src/AbstractOperations/multiary_operations.jl:##### GPU capabilities
src/AbstractOperations/multiary_operations.jl:"Adapt `MultiaryOperation` to work on the GPU via CUDAnative and CUDAdrv."
src/AbstractOperations/kernel_function_operation.jl:    random_kernel_function(i, j, k, grid) = rand(); # use CUDA.rand on the GPU
src/AbstractOperations/kernel_function_operation.jl:"Adapt `KernelFunctionOperation` to work on the GPU via CUDAnative and CUDAdrv."
src/AbstractOperations/binary_operations.jl:##### GPU capabilities
src/AbstractOperations/binary_operations.jl:"Adapt `BinaryOperation` to work on the GPU via CUDAnative and CUDAdrv."
src/Oceananigans.jl:data-driven, ocean-flavored fluid dynamics on CPUs and GPUs.
src/Oceananigans.jl:    CPU, GPU, 
src/Oceananigans.jl:using CUDA
src/Oceananigans.jl:    if CUDA.has_cuda()
src/Oceananigans.jl:        @debug "CUDA-enabled GPU(s) detected:"
src/Oceananigans.jl:        for (gpu, dev) in enumerate(CUDA.devices())
src/Oceananigans.jl:            @debug "$dev: $(CUDA.name(dev))"
src/Oceananigans.jl:        CUDA.allowscalar(false)
src/Forcings/forcing.jl:The object `parameters` is arbitrary in principle, however GPU compilation can place
src/Forcings/discrete_forcing.jl:Above, `parameters` is, in principle, arbitrary. Note, however, that GPU compilation
src/Diagnostics/Diagnostics.jl:using CUDA
src/Advection/weno_interpolants.jl:# Trick to force compilation of Val(stencil-1) and avoid loops on the GPU
src/Advection/weno_interpolants.jl:# Trick to force compilation of Val(stencil-1) and avoid loops on the GPU
src/DistributedComputations/distributed_architectures.jl:using CUDA: ndevices, device!
src/DistributedComputations/distributed_architectures.jl:- `child_architecture`: Specifies whether the computation is performed on CPUs or GPUs. 
src/DistributedComputations/distributed_architectures.jl:- `devices`: `GPU` device linked to local rank. The GPU will be assigned based on the 
src/DistributedComputations/distributed_architectures.jl:             local node rank as such `devices[node_rank]`. Make sure to run `--ntasks-per-node` <= `--gres=gpu`.
src/DistributedComputations/distributed_architectures.jl:    # Assign CUDA device if on GPUs
src/DistributedComputations/distributed_architectures.jl:    if child_architecture isa GPU
src/DistributedComputations/distributed_architectures.jl:const DistributedGPU = Distributed{GPU}
src/DistributedComputations/distributed_fft_tridiagonal_solver.jl:using CUDA: @allowscalar
src/DistributedComputations/distributed_fft_tridiagonal_solver.jl:    # We need to permute indices to apply bounded transforms on the GPU (r2r of r2c with twiddling)
src/DistributedComputations/distributed_fft_tridiagonal_solver.jl:    x_buffer_needed = child_arch isa GPU && TX == Bounded
src/DistributedComputations/distributed_fft_tridiagonal_solver.jl:    z_buffer_needed = child_arch isa GPU && TZ == Bounded 
src/DistributedComputations/distributed_fft_tridiagonal_solver.jl:    # We cannot really batch anything, so on GPUs we always have to permute indices in the y direction
src/DistributedComputations/distributed_fft_tridiagonal_solver.jl:    y_buffer_needed = child_arch isa GPU
src/DistributedComputations/distributed_fields.jl:using CUDA: CuArray
src/DistributedComputations/plan_distributed_transforms.jl:    if arch isa GPU
src/DistributedComputations/distributed_transpose.jl:           We need to synchronize the GPU afterwards before any communication can take place. The packing is
src/DistributedComputations/distributed_on_architecture.jl:using CUDA: CuArray
src/DistributedComputations/distributed_on_architecture.jl:# We only support moving a type from CPU to GPU and the other way around
src/DistributedComputations/distributed_fft_based_poisson_solver.jl:using CUDA: @allowscalar
src/DistributedComputations/distributed_fft_based_poisson_solver.jl:    # We need to permute indices to apply bounded transforms on the GPU (r2r of r2c with twiddling)
src/DistributedComputations/distributed_fft_based_poisson_solver.jl:    x_buffer_needed = child_arch isa GPU && TX == Bounded
src/DistributedComputations/distributed_fft_based_poisson_solver.jl:    z_buffer_needed = child_arch isa GPU && TZ == Bounded 
src/DistributedComputations/distributed_fft_based_poisson_solver.jl:    # We cannot really batch anything, so on GPUs we always have to permute indices in the y direction
src/DistributedComputations/distributed_fft_based_poisson_solver.jl:    y_buffer_needed = child_arch isa GPU
src/MultiRegion/multi_region_grid.jl:             allocated on the the `CPU`. For `GPU` computation it is possible to specify the total
src/MultiRegion/multi_region_grid.jl:             number of GPUs or the specific GPUs to allocate memory on. The number of devices does
src/MultiRegion/multi_region_grid.jl:    ## If we are on GPUs we want to enable peer access, which we do by just copying fake arrays between all devices
src/MultiRegion/cubed_sphere_grid.jl:    new_devices = arch == CPU() ? Tuple(CPU() for _ in 1:length(partition)) : Tuple(CUDA.device() for _ in 1:length(partition))
src/MultiRegion/multi_region_utils.jl:# If no device is specified on the GPU, use only the default device
src/MultiRegion/multi_region_utils.jl:validate_devices(p, ::GPU, ::Nothing) = 1
src/MultiRegion/multi_region_utils.jl:function validate_devices(partition, ::GPU, devices)
src/MultiRegion/multi_region_utils.jl:    @assert length(unique(devices)) ≤ length(CUDA.devices())
src/MultiRegion/multi_region_utils.jl:    @assert maximum(devices) ≤ length(CUDA.devices())
src/MultiRegion/multi_region_utils.jl:function validate_devices(partition, ::GPU, devices::Number)
src/MultiRegion/multi_region_utils.jl:    @assert devices ≤ length(CUDA.devices())
src/MultiRegion/multi_region_utils.jl:        CUDA.device!(i-1)
src/MultiRegion/multi_region_utils.jl:            push!(devices, CUDA.device())
src/MultiRegion/multi_region_utils.jl:            push!(devices, CUDA.device())
src/MultiRegion/multi_region_utils.jl:        CUDA.device!(dev[i])
src/MultiRegion/multi_region_utils.jl:            push!(devices, CUDA.device())
src/MultiRegion/multi_region_utils.jl:            push!(devices, CUDA.device())
src/MultiRegion/multi_region_utils.jl:function maybe_enable_peer_access!(devices::NTuple{<:Any, <:CUDA.CuDevice})
src/MultiRegion/MultiRegion.jl:using CUDA
src/ImmersedBoundaries/active_cells_map.jl:# REMEMBER: since the active map is stripped out of the grid when `Adapt`ing to the GPU, 
src/ImmersedBoundaries/active_cells_map.jl:    # Create the cells map on the CPU, then switch it to the GPU
src/ImmersedBoundaries/grid_fitted_bottom.jl:using CUDA: CuArray
src/TimeSteppers/TimeSteppers.jl:using CUDA
src/TimeSteppers/clock.jl:"""Adapt `Clock` for GPU."""
src/Fields/abstract_field.jl:using CUDA
src/Fields/field.jl:                return CUDA.@allowscalar first(r)
src/Fields/field.jl:    return CUDA.@allowscalar sqrt(r[1])
src/Fields/broadcasting_abstract_fields.jl:using CUDA
src/Fields/broadcasting_abstract_fields.jl:Base.Broadcast.BroadcastStyle(::FieldBroadcastStyle, ::CUDA.CuArrayStyle{N}) where N = FieldBroadcastStyle()
src/Fields/broadcasting_abstract_fields.jl:                                        Broadcasted{<:CUDA.CuArrayStyle}}
src/Fields/set!.jl:using CUDA
src/Fields/set!.jl:using Oceananigans.Architectures: child_architecture, device, GPU, CPU
src/Fields/set!.jl:    if arch isa GPU
src/Fields/set!.jl:    # Transfer data to GPU if u is on the GPU
src/Fields/set!.jl:    if child_architecture(u) isa GPU
src/Fields/interpolate.jl:# GPU-compatile middle point calculation
src/Fields/interpolate.jl:Note that this is a lower-level `interpolate` method defined for use in CPU/GPU kernels.
src/Fields/interpolate.jl:    # We use mod and trunc as CUDA.modf is not defined.
src/Grids/zeros_and_ones.jl:using CUDA
src/Grids/zeros_and_ones.jl:using Oceananigans.Architectures: CPU, GPU, AbstractArchitecture
src/Grids/zeros_and_ones.jl:zeros(FT, ::GPU, N...) = CUDA.zeros(FT, N...)
src/Grids/rectilinear_grid.jl:                  on the CPU or GPU. Default: `CPU()`.
src/Grids/rectilinear_grid.jl:    if architecture == GPU() && !has_cuda()
src/Grids/rectilinear_grid.jl:        throw(ArgumentError("Cannot create a GPU grid. No CUDA-enabled GPU was detected!"))
src/Grids/grid_utils.jl:using CUDA
src/Grids/grid_utils.jl:@inline domain(topo, N, ξ) = CUDA.@allowscalar ξ[1], ξ[N+1]
src/Grids/grid_generation.jl:get_domain_extent(coord::AbstractVector, N) = CUDA.@allowscalar (coord[1], coord[N+1])
src/Grids/grid_generation.jl:get_face_node(coord::AbstractVector, i) = CUDA.@allowscalar coord[i]
src/Grids/latitude_longitude_grid.jl:                  on the CPU or GPU. Default: `CPU()`.
src/Grids/latitude_longitude_grid.jl:    if architecture == GPU() && !has_cuda()
src/Grids/latitude_longitude_grid.jl:        throw(ArgumentError("Cannot create a GPU grid. No CUDA-enabled GPU was detected!"))
src/Grids/orthogonal_spherical_shell_grid.jl:                  on the CPU or GPU. Default: `CPU()`.
src/Grids/orthogonal_spherical_shell_grid.jl:    if architecture == GPU() && !has_cuda() 
src/Grids/orthogonal_spherical_shell_grid.jl:        throw(ArgumentError("Cannot create a GPU grid. No CUDA-enabled GPU was detected!"))
src/Grids/orthogonal_spherical_shell_grid.jl:    λ_center = CUDA.@allowscalar λnode(i_center, j_center, 1, grid, ℓx, ℓy, Center())
src/Grids/orthogonal_spherical_shell_grid.jl:    φ_center = CUDA.@allowscalar φnode(i_center, j_center, 1, grid, ℓx, ℓy, Center())
src/Grids/orthogonal_spherical_shell_grid.jl:        extent_λ = CUDA.@allowscalar maximum(rad2deg.(sum(grid.Δxᶜᶠᵃ[1:Nx, :], dims=1))) / grid.radius
src/Grids/orthogonal_spherical_shell_grid.jl:        extent_λ = CUDA.@allowscalar maximum(rad2deg.(sum(grid.Δxᶜᶜᵃ[1:Nx, :], dims=1))) / grid.radius
src/Grids/orthogonal_spherical_shell_grid.jl:        extent_φ = CUDA.@allowscalar maximum(rad2deg.(sum(grid.Δyᶠᶜᵃ[:, 1:Ny], dims=2))) / grid.radius
src/Grids/orthogonal_spherical_shell_grid.jl:        extent_φ = CUDA.@allowscalar maximum(rad2deg.(sum(grid.Δyᶠᶜᵃ[:, 1:Ny], dims=2))) / grid.radius
src/Grids/abstract_grid.jl:Return the architecture (CPU or GPU) that the `grid` lives on.
src/Grids/abstract_grid.jl:    CUDA.@allowscalar return x1 == x2 && y1 == y2 && z1 == z2
src/Grids/Grids.jl:using CUDA
src/Grids/Grids.jl:using CUDA: has_cuda
src/Models/ShallowWaterModels/shallow_water_model.jl:            architecture (CPU/GPU) that the model is solve is inferred from the architecture
src/Models/ShallowWaterModels/compute_shallow_water_tendencies.jl:    # in GPU computations.
src/Models/NonhydrostaticModels/nonhydrostatic_model.jl:using CUDA: has_cuda
src/Models/NonhydrostaticModels/nonhydrostatic_model.jl:            architecture (CPU/GPU) that the model is solved on is inferred from the architecture
src/Models/NonhydrostaticModels/compute_nonhydrostatic_tendencies.jl:    # in GPU computations.
src/Models/HydrostaticFreeSurfaceModels/single_column_model_mode.jl:using CUDA: @allowscalar
src/Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_model.jl:using CUDA: has_cuda
src/Models/HydrostaticFreeSurfaceModels/hydrostatic_free_surface_model.jl:            architecture (CPU/GPU) that the model is solved is inferred from the architecture
src/Models/HydrostaticFreeSurfaceModels/split_explicit_free_surface_kernels.jl:        # latency of argument conversion to GPU-compatible values.
src/TurbulenceClosures/TurbulenceClosures.jl:using CUDA
src/TurbulenceClosures/turbulence_closure_implementations/TKEBasedVerticalDiffusivities/tke_dissipation_equations.jl:using CUDA
src/TurbulenceClosures/turbulence_closure_implementations/TKEBasedVerticalDiffusivities/TKEBasedVerticalDiffusivities.jl:using CUDA
src/TurbulenceClosures/turbulence_closure_implementations/TKEBasedVerticalDiffusivities/TKEBasedVerticalDiffusivities.jl:    closure = CUDA.@allowscalar closure_array[1, 1]
src/TurbulenceClosures/turbulence_closure_implementations/TKEBasedVerticalDiffusivities/time_step_catke_equation.jl:using CUDA
src/TurbulenceClosures/turbulence_closure_implementations/isopycnal_skew_symmetric_diffusivity.jl:    if arch isa Architectures.GPU
src/Architectures.jl:export CPU, GPU
src/Architectures.jl:using CUDA
src/Architectures.jl:    GPU <: AbstractArchitecture
src/Architectures.jl:Run Oceananigans on a single NVIDIA CUDA GPU.
src/Architectures.jl:struct GPU <: AbstractSerialArchitecture end
src/Architectures.jl:device(::GPU) = CUDA.CUDABackend(; always_inline=true)
src/Architectures.jl:architecture(::CuArray) = GPU()
src/Architectures.jl:array_type(::GPU) = CuArray
src/Architectures.jl:on_architecture(::GPU, a::Array) = CuArray(a)
src/Architectures.jl:on_architecture(::GPU, a::CuArray) = a
src/Architectures.jl:on_architecture(::GPU, a::BitArray) = CuArray(a)
src/Architectures.jl:on_architecture(::GPU, a::SubArray{<:Any, <:Any, <:CuArray}) = a
src/Architectures.jl:on_architecture(::GPU, a::SubArray{<:Any, <:Any, <:Array}) = CuArray(a)
src/Architectures.jl:cpu_architecture(::GPU) = CPU()
src/Architectures.jl:unified_array(::GPU, a) = a
src/Architectures.jl:unified_array(::GPU, a::AbstractArray) = map(eltype(a), cu(a; unified = true))
src/Architectures.jl:## GPU to GPU copy of contiguous data
src/Architectures.jl:@inline unsafe_free!(a::CuArray) = CUDA.unsafe_free!(a)
src/Architectures.jl:# Convert arguments to GPU-compatible types
src/Architectures.jl:@inline convert_args(::GPU, args) = CUDA.cudaconvert(args)
src/Architectures.jl:@inline convert_args(::GPU, args::Tuple) = map(CUDA.cudaconvert, args)
src/Solvers/heptadiagonal_iterative_solver.jl:using CUDA, CUDA.CUSPARSE
src/Solvers/heptadiagonal_iterative_solver.jl:- `CuSparseMatrixCSC(constructors...)` for GPU
src/Solvers/Solvers.jl:using CUDA
src/Solvers/Solvers.jl:using Oceananigans.Architectures: device, CPU, GPU, array_type, on_architecture
src/Solvers/fourier_tridiagonal_poisson_solver.jl:    lower_diagonal = CUDA.@allowscalar [ 1 / Δξᶠ(q, grid) for q in 2:size(grid, irreg_dim) ]
src/Solvers/fourier_tridiagonal_poisson_solver.jl:    buffer_needed = arch isa GPU && Bounded in (regular_top1, regular_top2)
src/Solvers/sparse_approximate_inverse.jl:makes it very appealing to use on the GPU.
src/Solvers/sparse_approximate_inverse.jl:in Julia: Hooray! but not on GPUs... booo)
src/Solvers/matrix_solver_utils.jl:using CUDA, CUDA.CUSPARSE
src/Solvers/matrix_solver_utils.jl:@inline constructors(::GPU, A::SparseMatrixCSC) = (CuArray(A.colptr), CuArray(A.rowval), CuArray(A.nzval),  (A.m, A.n))
src/Solvers/matrix_solver_utils.jl:@inline constructors(::GPU, A::CuSparseMatrixCSC) = (A.colPtr, A.rowVal, A.nzVal,  A.dims)
src/Solvers/matrix_solver_utils.jl:@inline constructors(::GPU, m::Number, n::Number, constr::Tuple) = (constr..., (m, n))
src/Solvers/matrix_solver_utils.jl:@inline unpack_constructors(::GPU, constr::Tuple) = (constr[1], constr[2], constr[3])
src/Solvers/matrix_solver_utils.jl:@inline copy_unpack_constructors(::GPU, constr::Tuple) = deepcopy((constr[1], constr[2], constr[3]))
src/Solvers/matrix_solver_utils.jl:@inline arch_sparse_matrix(::GPU, constr::Tuple) = CuSparseMatrixCSC(constr...)
src/Solvers/matrix_solver_utils.jl:@inline arch_sparse_matrix(::GPU, A::SparseMatrixCSC)     = CuSparseMatrixCSC(constructors(GPU(), A)...)
src/Solvers/matrix_solver_utils.jl:@inline arch_sparse_matrix(::GPU, A::CuSparseMatrixCSC) = A
src/Solvers/matrix_solver_utils.jl:#unfortunately this cannot run on a GPU so we have to resort to that ugly loop in _update_diag!
src/Solvers/matrix_solver_utils.jl:For `architecture = CPU()` the matrix returned is a `SparseArrays.SparseMatrixCSC`; for `GPU()`
src/Solvers/matrix_solver_utils.jl:is a `CUDA.CuSparseMatrixCSC`.
src/Solvers/matrix_solver_utils.jl:function compute_matrix_for_linear_operation(::GPU, template_field, linear_operation!, args...;
src/Solvers/matrix_solver_utils.jl:    CUDA.@allowscalar colptr[1] = 1
src/Solvers/matrix_solver_utils.jl:        CUDA.@allowscalar eᵢⱼₖ[i, j, k] = 1
src/Solvers/matrix_solver_utils.jl:            Aeᵢⱼₖₗₘₙ = CUDA.@allowscalar Aeᵢⱼₖ[l, m, n]
src/Solvers/matrix_solver_utils.jl:        CUDA.@allowscalar colptr[Ny*Nx*(k-1) + Nx*(j-1) + i + 1] = colptr[Ny*Nx*(k-1) + Nx*(j-1) + i] + count
src/Solvers/fft_based_poisson_solver.jl:    buffer_needed = arch isa GPU && Bounded in topo
src/Solvers/fft_based_poisson_solver.jl:    m === 0 && CUDA.@allowscalar ϕc[1, 1, 1] = 0
src/Solvers/discrete_transforms.jl:    twiddle_factors(arch::GPU, grid, dims)
src/Solvers/discrete_transforms.jl:Twiddle factors are needed to perform DCTs on the GPU. See equations (19a) and (22) of [Makhoul80](@citet)
src/Solvers/discrete_transforms.jl:function twiddle_factors(arch::GPU, grid, dims)
src/Solvers/discrete_transforms.jl:    transpose = arch isa GPU && dims == [2] ? (2, 1, 3) : nothing
src/Solvers/discrete_transforms.jl:function maybe_permute_indices!(A, B, arch::GPU, grid, dim, ::Bounded)
src/Solvers/discrete_transforms.jl:function maybe_unpermute_indices!(A, B, arch::GPU, grid, dim, ::Bounded)
src/Solvers/sparse_preconditioners.jl:using CUDA, CUDA.CUSPARSE
src/Solvers/sparse_preconditioners.jl:on the `GPU`
src/Solvers/sparse_preconditioners.jl:`ilu()` cannot be used on the GPU because preconditioning the solver with a direct LU (or Choleski) type 
src/Solvers/sparse_preconditioners.jl:    if architecture(A) isa GPU 
src/Solvers/sparse_preconditioners.jl:        throw(ArgumentError("the ILU factorization is not available on the GPU! choose another method"))
src/Solvers/sparse_preconditioners.jl:@inline architecture(::CuSparseMatrixCSC) = GPU()
src/Solvers/sparse_preconditioners.jl:All preconditioners are calculated on CPU and, if the model is based on a GPU architecture, then moved to the GPU.
src/Solvers/sparse_preconditioners.jl:on the GPU `asymptotic_diagonal_inverse_preconditioner_first_order(A)` in case of variable
src/Solvers/plan_transforms.jl:    return CUDA.CUFFT.plan_fft!(A, dims)
src/Solvers/plan_transforms.jl:    return CUDA.CUFFT.plan_ifft!(A, dims)
src/Solvers/plan_transforms.jl:batchable_GPU_topologies = ((Periodic, Periodic, Periodic),
src/Solvers/plan_transforms.jl:# the GPU we take the real part after a forward transform, so if the `Periodic`
src/Solvers/plan_transforms.jl:    if arch isa GPU && !(unflattened_topo in batchable_GPU_topologies)
src/Solvers/plan_transforms.jl:        # `batchable_GPU_topologies` occurs when there are two adjacent `Periodic` dimensions:
src/Solvers/plan_transforms.jl:        # On the GPU and for vertically Bounded grids, batching is possible either in horizontally-periodic
src/Solvers/plan_transforms.jl:        # We're on the GPU and either (Periodic, Periodic), (Flat, Periodic), or
src/Solvers/plan_transforms.jl:    else # we are on the GPU and we cannot / should not batch!
src/OutputWriters/fetch_output.jl:using CUDA
src/OutputWriters/fetch_output.jl:    if architecture(output) isa GPU
src/OutputWriters/netcdf_output_writer.jl:using Oceananigans.Utils: versioninfo_with_gpu, oceananigans_versioninfo, prettykeys
src/OutputWriters/netcdf_output_writer.jl:    global_attributes["Julia"] = "This file was generated using " * versioninfo_with_gpu()
src/OutputWriters/OutputWriters.jl:using CUDA
src/Utils/versioninfo.jl:function versioninfo_with_gpu()
src/Utils/versioninfo.jl:    if CUDA.has_cuda()
src/Utils/versioninfo.jl:        gpu_name = CUDA.device() |> CUDA.name
src/Utils/versioninfo.jl:        s = s * "  GPU: $gpu_name\n"
src/Utils/multi_region_transformation.jl:using CUDA: CuArray, CuDevice, CuContext, CuPtr, device, device!, synchronize
src/Utils/multi_region_transformation.jl:const GPUVar = Union{CuArray, CuContext, CuPtr, Ptr}
src/Utils/multi_region_transformation.jl:@inline getdevice(cu::GPUVar, i)            = CUDA.device(cu)
src/Utils/multi_region_transformation.jl:@inline getdevice(cu::GPUVar)      = CUDA.device(cu)
src/Utils/multi_region_transformation.jl:@inline switch_device!(dev::Int)                 = CUDA.device!(dev)
src/Utils/multi_region_transformation.jl:@inline switch_device!(dev::CuDevice)            = CUDA.device!(dev)
src/Utils/multi_region_transformation.jl:@inline sync_device!(::GPU)      = CUDA.synchronize()
src/Utils/multi_region_transformation.jl:@inline sync_device!(::CuDevice) = CUDA.synchronize()
src/Utils/Utils.jl:export versioninfo_with_gpu, oceananigans_versioninfo
src/Utils/Utils.jl:import CUDA  # To avoid name conflicts
src/BoundaryConditions/BoundaryConditions.jl:using CUDA, Adapt
src/BoundaryConditions/BoundaryConditions.jl:using Oceananigans.Architectures: CPU, GPU, device
src/BoundaryConditions/boundary_condition.jl:# Adapt boundary condition struct to be GPU friendly and passable to GPU kernels.
src/BoundaryConditions/boundary_condition.jl:# Adapt boundary condition struct to be GPU friendly and passable to GPU kernels.
src/BoundaryConditions/boundary_condition.jl:validate_boundary_condition_architecture(::CuArray, ::GPU, bc, side) = nothing
src/BoundaryConditions/boundary_condition.jl:validate_boundary_condition_architecture(::Array, ::GPU, bc, side) =
src/BoundaryConditions/boundary_condition.jl:    throw(ArgumentError("$side $bc must use `CuArray` rather than `Array` on GPU architectures!"))

```
