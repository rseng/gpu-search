# https://github.com/pybamm-team/PyBaMM

```console
CHANGELOG.md:- Fixed a bug where the JaxSolver would fails when using GPU support with no input parameters ([#3423](https://github.com/pybamm-team/PyBaMM/pull/3423))
CHANGELOG.md:- Added parameter list support to JAX solver, permitting multithreading / GPU execution ([#3121](https://github.com/pybamm-team/PyBaMM/pull/3121))
src/pybamm/solvers/c_solvers/idaklu/Expressions/IREE/iree_jit.hpp:   *         IREE runtime. E.g. "local-sync" for CPU, "vulkan" for GPU, etc.
src/pybamm/solvers/c_solvers/idaklu/sundials_legacy_wrapper.hpp:  N_Vector N_VNew_Cuda(sunindextype vec_length, SUNContext sunctx)
src/pybamm/solvers/c_solvers/idaklu/sundials_legacy_wrapper.hpp:    return N_VNew_Cuda(vec_length);
src/pybamm/solvers/jax_solver.py:            platform.startswith("gpu")
src/pybamm/solvers/jax_solver.py:            # gpu execution runs faster when parallelised with vmap

```
