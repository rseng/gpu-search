# https://github.com/podondra/bayesian-sznet

```console
dr12_predict.jl:    X_va = gpu(read(datafile, "X_va"))
BayesianSZNet.jl:FCNN(modelfile::String) = FCNN(gpu(trainmode!(BSON.load(modelfile, @__MODULE__)[:model])))
BayesianSZNet.jl:FCNN(p::AbstractFloat) = FCNN(gpu(trainmode!(Chain(Dense(3752, 512, relu),
BayesianSZNet.jl:SZNet(modelfile::String) = SZNet(gpu(trainmode!(BSON.load(modelfile, @__MODULE__)[:model])))
BayesianSZNet.jl:SZNet(p::AbstractFloat) = SZNet(gpu(trainmode!(Chain(Flux.unsqueeze(2),
BayesianSZNet.jl:    X_tr_gpu, z_tr_gpu, X_va_gpu = gpu(X_tr), gpu(z_tr), gpu(X_va)
BayesianSZNet.jl:    loader = DataLoader((X_tr_gpu, z_tr_gpu), batchsize=256, shuffle=true)
BayesianSZNet.jl:            ẑs_va = sample(model, X_va_gpu)
Manifest.toml:[[CUDA]]
Manifest.toml:deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "DataStructures", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "Libdl", "LinearAlgebra", "Logging", "MacroTools", "NNlib", "Pkg", "Printf", "Random", "Reexport", "Requires", "SparseArrays", "Statistics", "TimerOutputs"]
Manifest.toml:deps = ["AbstractTrees", "Adapt", "CUDA", "CodecZlib", "Colors", "DelimitedFiles", "Functors", "Juno", "LinearAlgebra", "MacroTools", "NNlib", "Pkg", "Printf", "Random", "Reexport", "SHA", "Statistics", "StatsBase", "Test", "ZipFile", "Zygote"]
Manifest.toml:[[GPUArrays]]
Manifest.toml:[[GPUCompiler]]
dr16_predict.jl:    X = gpu(read(datafile, "X"))
Project.toml:CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

```
