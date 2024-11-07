# https://github.com/gaelforget/MITgcm.jl

```console
docs/joss/paper.bib:@article{Wu2022, doi = {10.21105/joss.04207}, url = {https://doi.org/10.21105/joss.04207}, year = {2022}, publisher = {The Open Journal}, volume = {7}, number = {73}, pages = {4207}, author = {Zhen Wu and Ga{\"{e}}l Forget}, title = {PlanktonIndividuals.jl: A GPU supported individual-based phytoplankton life cycle model}, journal = {Journal of Open Source Software} }
docs/joss/paper.bib:  title = {{Oceananigans.jl: Fast and friendly geophysical fluid dynamics on GPUs}},
docs/joss/paper.md:The cutting-edge of climate modeling, and much of its legacy, is based on numerical models written in compiled languages like `Fortran` or `C`. The MIT general circulation model (`MITgcm`) for example now runs globally at the kilometer scale to provide an unprecedented view of ocean dynamics (Fig. \ref{fig:examples}, @camp-etal:04,  @marshall1997fvi, @Gallmeier2023). With its unrivaled adjoint modeling capabilities, `MITgcm` is also the computational engine that powers the ECCO ocean reanalysis, a widely-used data-assimilating product in climate science (Fig. \ref{fig:examples}, @heimbach2002automatic, @Forget2015a,  @Forget2024). `MITgcm` additionally provides unique modeling capabilities for ocean biogeochemistry, ecology, and optics (@Dutkiewicz2015, @Cbiomes2019). While a new generation of models, written in languages like C++ and Julia, is poised to better exploit GPUs (@OceananigansJOSS, @Wu2022, @e3sm-model), `Fortran`-based models are expected to remain popular on other computer architectures for the foreseeable future. They also provide a crucial reference point to evaluate next generation models.
examples/MITgcm_scan_output.jl:deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
examples/MITgcm_scan_output.jl:[[deps.GPUArrays]]
examples/MITgcm_scan_output.jl:deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
examples/MITgcm_scan_output.jl:[[deps.GPUArraysCore]]
examples/MITgcm_scan_output.jl:weakdeps = ["Adapt", "GPUArraysCore", "SparseArrays", "StaticArrays"]
examples/MITgcm_scan_output.jl:    StructArraysGPUArraysCoreExt = "GPUArraysCore"
examples/MITgcm_scan_output.jl:deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
examples/HS94_Makie.jl:    StructArraysGPUArraysCoreExt = "GPUArraysCore"
examples/HS94_Makie.jl:    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
examples/HS94_particles.jl:    ArrayInterfaceCUDAExt = "CUDA"
examples/HS94_particles.jl:    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
examples/HS94_particles.jl:    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
examples/HS94_particles.jl:    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
examples/HS94_particles.jl:    DiffEqBaseCUDAExt = "CUDA"
examples/HS94_particles.jl:    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
examples/HS94_particles.jl:deps = ["Adapt", "ArrayInterface", "GPUArraysCore", "GenericSchur", "LinearAlgebra", "PrecompileTools", "Printf", "SparseArrays", "libblastrampoline_jll"]
examples/HS94_particles.jl:[[deps.GPUArraysCore]]
examples/HS94_particles.jl:deps = ["ArrayInterface", "ChainRulesCore", "ConcreteStructs", "DocStringExtensions", "EnumX", "FastLapackInterface", "GPUArraysCore", "InteractiveUtils", "KLU", "Krylov", "LazyArrays", "Libdl", "LinearAlgebra", "MKL_jll", "Markdown", "PrecompileTools", "Preferences", "RecursiveFactorization", "Reexport", "SciMLBase", "SciMLOperators", "Setfield", "SparseArrays", "Sparspak", "StaticArraysCore", "UnPack"]
examples/HS94_particles.jl:    LinearSolveCUDAExt = "CUDA"
examples/HS94_particles.jl:    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
examples/HS94_particles.jl:deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
examples/Darwin3_1D.jl:    StructArraysGPUArraysCoreExt = "GPUArraysCore"
examples/Darwin3_1D.jl:    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
src/ShellScripting.jl:module load cuda/11.0

```
