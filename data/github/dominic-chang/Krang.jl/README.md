# https://github.com/dominic-chang/Krang.jl

```console
docs/make.jl:blacklist = ["gpu", "JuKeBOX", "Krang_logo", "enzyme"]
docs/src/getting_started.md:The GPU arrays can be passed to the cameras on construction to raytrace enforce raytracing on the GPU.
docs/src/getting_started.md:An sketch of how to do this with a CUDA array is:
docs/src/getting_started.md:using CUDA
docs/src/getting_started.md:store = CUDA.fill(0.0, sze, sze)
docs/src/index.md:  - icon: <img width="64" height="64" src="https://metal.juliagpu.org/stable/assets/logo.png" />
docs/src/index.md:    title: GPU Compatible
docs/src/index.md:    details: Type stable and type preserving. GPU compatible with CUDA.jl and Metal.jl.
gpu_examples/README.MD:# GPU examples
gpu_examples/README.MD:Here are a collection of GPU versions of the examples from the example script folder.
README.md:The ray tracing scheme has been optimized for GPU compatibility and automatic differentiability with [Enzyme.jl](https://enzyme.mit.edu/julia/stable/). 
examples/coordinate-example.jl:# > The GPU can be used in this example with an appropriate broadcast.
examples/coordinate-example.jl:# using CUDA
src/cameras/IntensityCamera.jl:    - `A=Matrix`: Optional argument to specify the type of matrix to use. A GPUMatrix can be used for GPU computations.
src/cameras/IntensityCamera.jl:    - `A=Matrix`: Optional argument to specify the type of matrix to use. A GPUMatrix can be used for GPU computations.
src/cameras/SlowLightIntensityCamera.jl:    - `A=Matrix`: Data type that stores screen pixel information (default is `Matrix`). A GPUMatrix can be used for GPU computations.
src/cameras/SlowLightIntensityCamera.jl:    - `A`: Data type that stores screen pixel information (default is `Matrix`). A GPUMatrix can be used for GPU computations.

```
