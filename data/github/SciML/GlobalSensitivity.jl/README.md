# https://github.com/SciML/GlobalSensitivity.jl

```console
docs/src/tutorials/parallelized_gsa.md:`EnsembleGPUArray` to perform automatic multithreaded-parallelization of the ODE solves.
docs/src/tutorials/parallelized_gsa.md:[DiffEqGPU.jl](https://github.com/SciML/DiffEqGPU.jl) for automated GPU-parallelism of
joss/paper.md:The ability to introduce parallelism with GlobalSensitivity.jl by using the batch keyword argument is shown in the below code snippet. In the batch interface, each column `p[:, i]` is a set of parameters, and we output a column for each set of parameters. Here we present the use of [Ensemble Interface](https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/) through `EnsembleGPUArray` to perform automatic multithreaded-parallelization of the ODE solves.

```
