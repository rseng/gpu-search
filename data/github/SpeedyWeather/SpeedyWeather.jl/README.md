# https://github.com/SpeedyWeather/SpeedyWeather.jl

```console
docs/joss/paper.bib:  title = {Oceananigans.Jl: {{Fast}} and {{Friendly Geophysical Fluid Dynamics}} on {{GPUs}}},
docs/joss/paper.md:GPU support is planned.
docs/src/lowertriangularmatrices.md:## GPU 
docs/src/lowertriangularmatrices.md:If this array is a GPU array (e.g. `CuArray`), all operations are performed on GPU as well (work in progress).
docs/src/lowertriangularmatrices.md:so that GPU operation should be performant.
docs/src/lowertriangularmatrices.md:To use `LowerTriangularArray` on GPU you can e.g. just `adapt` an existing `LowerTriangularArray`.
docs/src/lowertriangularmatrices.md:L_gpu = adapt(CuArray, L)
test/grids.jl:@testset "AbstractGridArray: GPU (JLArrays)" begin 
test/lower_triangular_matrix.jl:@testset "LowerTriangularArray: GPU (JLArrays)" begin 
test/lower_triangular_matrix.jl:    # TODO: so far very basic GPU test, might integrate them into the other tests, as I already did with the broadcast test, but there are some key differences to avoid scalar indexing
test/lower_triangular_matrix.jl:    # the core of this copyto! in a GPU compatible way, and is called by copyto! with CuArrays
test/lower_triangular_matrix.jl:    # test that GPU and CPU method yield the same
test/runtests.jl:# GPU/KERNELABSTRACTIONS
CHANGELOG.md:- Move CUDA dependency into extension [#586](https://github.com/SpeedyWeather/SpeedyWeather.jl/pull/586)
ext/SpeedyWeatherCUDAExt.jl:module SpeedyWeatherCUDAExt
ext/SpeedyWeatherCUDAExt.jl:import CUDA: CUDA, CUDAKernels, CuArray
ext/SpeedyWeatherCUDAExt.jl:SpeedyWeather.default_array_type(::Type{GPU}) = CuArray
ext/SpeedyWeatherCUDAExt.jl:# DEVICE SETUP FOR CUDA
ext/SpeedyWeatherCUDAExt.jl:Return default used device for internal purposes, either `CPU` or `GPU` if a GPU is available."""
ext/SpeedyWeatherCUDAExt.jl:Device() = CUDA.functional() ? GPU() : CPU()
ext/SpeedyWeatherCUDAExt.jl:Return default used device for KernelAbstractions, either `CPU` or `CUDADevice` if a GPU is available."""
ext/SpeedyWeatherCUDAExt.jl:SpeedyWeather.Device_KernelAbstractions() = CUDA.functional() ? KernelAbstractions.CUDADevice : KernelAbstractions.CPU
ext/SpeedyWeatherCUDAExt.jl:SpeedyWeather.Device_KernelAbstractions(::GPU) = KernelAbstractions.CUDADevice
ext/SpeedyWeatherCUDAExt.jl:SpeedyWeather.DeviceArray(::GPU, x) = Adapt.adapt(CuArray, x)
ext/SpeedyWeatherCUDAExt.jl:Returns a `CuArray` when `device<:GPU` is used. Doesn't uses `adapt`, therefore always returns CuArray."""
ext/SpeedyWeatherCUDAExt.jl:SpeedyWeather.DeviceArrayNotAdapt(::GPU, x) = CuArray(x)
README.md:- single GPU support to accelerate medium to high resolution simulations
Project.toml:GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
Project.toml:CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Project.toml:SpeedyWeatherCUDAExt = "CUDA"
Project.toml:CUDA = "4, 5"
Project.toml:GPUArrays = "10"
Project.toml:CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Project.toml:test = ["Test", "JLArrays", "CUDA"]
src/LowerTriangularMatrices/lower_triangular_array.jl:# helper function for conversion etc on GPU, returns indices of the lower triangle
src/LowerTriangularMatrices/lower_triangular_array.jl:# GPU version and fallback for higher dimensions
src/LowerTriangularMatrices/lower_triangular_array.jl:# Fallback / GPU version (the two versions _copyto! and copyto! are there to enable tests of this function with regular Arrays)
src/LowerTriangularMatrices/lower_triangular_array.jl:# Broadcast CPU/GPU
src/LowerTriangularMatrices/lower_triangular_array.jl:# GPU without scalar indexing
src/LowerTriangularMatrices/lower_triangular_array.jl:struct LowerTriangularGPUStyle{N, ArrayType} <: GPUArrays.AbstractGPUArrayStyle{N} end
src/LowerTriangularMatrices/lower_triangular_array.jl:) where {T, N, ArrayType <: GPUArrays.AbstractGPUArray}
src/LowerTriangularMatrices/lower_triangular_array.jl:    return LowerTriangularGPUStyle{N, ArrayType_}()
src/LowerTriangularMatrices/lower_triangular_array.jl:LowerTriangularGPUStyle{N, ArrayType}(::Val{M}) where {N, ArrayType, M} =
src/LowerTriangularMatrices/lower_triangular_array.jl:    LowerTriangularGPUStyle{N, ArrayType}()
src/LowerTriangularMatrices/lower_triangular_array.jl:# same function as above, but needs to be defined for both CPU and GPU style
src/LowerTriangularMatrices/lower_triangular_array.jl:    bc::Broadcasted{LowerTriangularGPUStyle{N, ArrayType}},
src/LowerTriangularMatrices/lower_triangular_array.jl:function GPUArrays.backend(
src/LowerTriangularMatrices/lower_triangular_array.jl:) where {T, N, ArrayType <: GPUArrays.AbstractGPUArray}
src/LowerTriangularMatrices/lower_triangular_array.jl:    return GPUArrays.backend(ArrayType)
src/LowerTriangularMatrices/LowerTriangularMatrices.jl:# GPU
src/LowerTriangularMatrices/LowerTriangularMatrices.jl:import GPUArrays
src/SpeedyWeather.jl:# GPU, PARALLEL
src/SpeedyWeather.jl:# Utility for GPU / KernelAbstractions
src/SpeedyWeather.jl:include("gpu.jl")                               
src/gpu.jl:export CPU, GPU
src/gpu.jl:    GPU <: AbstractDevice
src/gpu.jl:Indicates that SpeedyWeather.jl runs on a single GPU
src/gpu.jl:struct GPU <: AbstractDevice end 
src/gpu.jl:    return device isa GPU ? 32 : 4 
src/RingGrids/RingGrids.jl:# GPU
src/RingGrids/RingGrids.jl:import GPUArrays
src/RingGrids/general.jl:(Julia's `Array` for CPU or others for GPU).
src/RingGrids/general.jl:## GPU
src/RingGrids/general.jl:struct AbstractGPUGridArrayStyle{N, ArrayType, Grid} <: GPUArrays.AbstractGPUArrayStyle{N} end
src/RingGrids/general.jl:) where {Grid<:AbstractGridArray{T, N, ArrayType}} where {T, N, ArrayType <: GPUArrays.AbstractGPUArray}
src/RingGrids/general.jl:    return AbstractGPUGridArrayStyle{N, ArrayType, nonparametric_type(Grid)}()
src/RingGrids/general.jl:AbstractGPUGridArrayStyle{N, ArrayType, Grid}(::Val{N}) where {N, ArrayType, Grid} =
src/RingGrids/general.jl:    AbstractGPUGridArrayStyle{N, ArrayType, Grid}()
src/RingGrids/general.jl:AbstractGPUGridArrayStyle{1, ArrayType, Grid}(::Val{2}) where {ArrayType, Grid} = AbstractGPUGridArrayStyle{2, ArrayType, Grid}()
src/RingGrids/general.jl:AbstractGPUGridArrayStyle{1, ArrayType, Grid}(::Val{0}) where {ArrayType, Grid} = AbstractGPUGridArrayStyle{1, ArrayType, Grid}()
src/RingGrids/general.jl:AbstractGPUGridArrayStyle{2, ArrayType, Grid}(::Val{3}) where {ArrayType, Grid} = AbstractGPUGridArrayStyle{3, ArrayType, Grid}()
src/RingGrids/general.jl:AbstractGPUGridArrayStyle{2, ArrayType, Grid}(::Val{1}) where {ArrayType, Grid} = AbstractGPUGridArrayStyle{2, ArrayType, Grid}()
src/RingGrids/general.jl:AbstractGPUGridArrayStyle{3, ArrayType, Grid}(::Val{4}) where {ArrayType, Grid} = AbstractGPUGridArrayStyle{4, ArrayType, Grid}()
src/RingGrids/general.jl:AbstractGPUGridArrayStyle{3, ArrayType, Grid}(::Val{2}) where {ArrayType, Grid} = AbstractGPUGridArrayStyle{3, ArrayType, Grid}()
src/RingGrids/general.jl:function GPUArrays.backend(
src/RingGrids/general.jl:) where {Grid <: AbstractGridArray{T, N, ArrayType}} where {T, N, ArrayType <: GPUArrays.AbstractGPUArray}
src/RingGrids/general.jl:    return GPUArrays.backend(ArrayType)
src/RingGrids/general.jl:    bc::Broadcasted{AbstractGPUGridArrayStyle{N, ArrayType, Grid}},

```
