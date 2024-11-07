# https://github.com/JuliaMatrices/BlockBandedMatrices.jl

```console
examples/cuarrays.jl:using BandedMatrices, BlockBandedMatrices, ArrayLayouts, BlockArrays, FillArrays, CuArrays, GPUArrays, LinearAlgebra
examples/cuarrays.jl:GPUArrays.synchronize(A)
examples/sharedarrays_setup.jl:Pkg.activate(homedir() * "/Documents/Coding/gpublockbanded")
examples/blockarray_backend.jl: using CUDAnative
examples/blockarray_backend.jl:using CUDAnative.CUDAdrv: CuStream
examples/blockarray_backend.jl:using GPUArrays
examples/blockarray_backend.jl:  @testset "block-banded on NVIDIA gpus" begin
examples/blockarray_backend.jl:       cgpu = cu(cblock)
examples/blockarray_backend.jl:       @test streamed_mul!(cgpu, cu(A), cu(x)) ≈ A * x
examples/blockarray_backend.jl:    suite["gpu"] = BenchmarkGroup()
examples/blockarray_backend.jl:      gpus = Dict(:c => adapt(CuArray, c),
examples/blockarray_backend.jl:      suite["gpu"]["N=$N n=$n"] = @benchmarkable begin
examples/blockarray_backend.jl:        LinearAlgebra.mul!($(gpus[:c]), $(gpus[:A]), $(gpus[:x]))
examples/blockarray_backend.jl:        gpuc = streamed_mul!($(gpus[:c]), $(gpus[:A]), $(gpus[:x]))

```
