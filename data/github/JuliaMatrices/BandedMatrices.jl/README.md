# https://github.com/JuliaMatrices/BandedMatrices.jl

```console
examples/cuarrays.jl:# This example demonstrates a BandedMatrix on the GPU using CLArrays
examples/cuarrays.jl:# slow when tried to do directly on the GPU.
examples/cuarrays.jl:using GPUArrays, CuArrays, FillArrays, BandedMatrices
examples/cuarrays.jl:    GPUArrays.synchronize(u)
examples/cuarrays.jl:    GPUArrays.synchronize(u)
examples/clarrays.jl:# This example demonstrates a BandedMatrix on the GPU using CLArrays
examples/clarrays.jl:# slow when tried to do directly on the GPU.
examples/clarrays.jl:using GPUArrays, CLArrays, FillArrays, BandedMatrices
examples/clarrays.jl:    GPUArrays.synchronize(u)
examples/clarrays.jl:    GPUArrays.synchronize(u)

```
