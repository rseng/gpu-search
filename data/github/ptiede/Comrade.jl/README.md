# https://github.com/ptiede/Comrade.jl

```console
benchmarks/Manifest.toml:    ArrayInterfaceCUDAExt = "CUDA"
benchmarks/Manifest.toml:    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
benchmarks/Manifest.toml:    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
benchmarks/Manifest.toml:    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
benchmarks/Manifest.toml:deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
benchmarks/Manifest.toml:    ComponentArraysGPUArraysExt = "GPUArrays"
benchmarks/Manifest.toml:    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
benchmarks/Manifest.toml:deps = ["CEnum", "EnzymeCore", "Enzyme_jll", "GPUCompiler", "LLVM", "Libdl", "LinearAlgebra", "ObjectFile", "Preferences", "Printf", "Random"]
benchmarks/Manifest.toml:[[deps.GPUArrays]]
benchmarks/Manifest.toml:deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
benchmarks/Manifest.toml:[[deps.GPUArraysCore]]
benchmarks/Manifest.toml:[[deps.GPUCompiler]]
benchmarks/Manifest.toml:weakdeps = ["Adapt", "GPUArraysCore", "SparseArrays", "StaticArrays"]
benchmarks/Manifest.toml:    StructArraysGPUArraysCoreExt = "GPUArraysCore"
benchmarks/Manifest.toml:deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
src/instrument/model.jl:    # we will revert to broadcast so it works on the GPU

```
