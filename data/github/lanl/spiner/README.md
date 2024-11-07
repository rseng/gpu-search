# https://github.com/lanl/spiner

```console
test/CMakeLists.txt:  if (SPINER_TEST_USE_KOKKOS_CUDA)
test/CMakeLists.txt:    list(APPEND _spiner_content_opts "Kokkos_ENABLE_CUDA")
test/CMakeLists.txt:    list(APPEND _spiner_content_opts "Kokkos_ENABLE_CUDA_LAMBDA")
test/CMakeLists.txt:    list(APPEND _spiner_content_opts "Kokkos_ENABLE_CUDA_CONSTEXPR")
doc/sphinx/index.rst:run on CPUs, GPUs and everything in-between. You can create a table on
doc/sphinx/index.rst:a CPU, copy it to a GPU, and interpolate on it in a GPU kernel, for
doc/sphinx/index.rst:slice shown) on a GPU, with second-order convergence:
doc/sphinx/index.rst:performance on CPU and GPU for several architectures and problem
doc/sphinx/src/databox.rst:  In the function signatures below, GPU/performance portability
doc/sphinx/src/databox.rst:If GPU support is enabled, a ``DataBox`` can be allocated on either
doc/sphinx/src/databox.rst:  // Allocates on the device (GPU)
doc/sphinx/src/databox.rst:  If GPU support is not enabled, these both allocate on host.
doc/sphinx/src/databox.rst:If GPU support is enabled, you can deep-copy a ``DataBox`` and any
doc/sphinx/src/databox.rst:GPU. An object-oriented method
doc/sphinx/src/databox.rst:copied to GPU.
doc/sphinx/src/databox.rst:  If GPU support is not enabled, ``getOnDevice`` and friends are
doc/sphinx/src/databox.rst:  // Before using the databox in, e.g., a GPU or Kokkos kernel, get a
doc/sphinx/src/building.rst:* ``SPINER_USE_CUDA`` enables the Kokkos cuda backend
doc/sphinx/src/building.rst:* ``+cuda`` enables the cuda backend. A ``cuda_arch`` must be specified.
doc/sphinx/src/statement-of-need.rst:specialized hardware, such as GPUs. A key tool in the toolbox for many
doc/sphinx/src/statement-of-need.rst:on, whether this is an NVIDIA GPU, an Intel CPU, or a next generation
doc/sphinx/src/statement-of-need.rst:ubiquitous that hardware primitives are provided by GPUs. These
doc/sphinx/src/statement-of-need.rst:application. For example, on NVIDIA GPUs, the values to be
doc/sphinx/src/statement-of-need.rst:scientific applications. As GPUs are inherently vector devices,
README.md:`Spiner` is compatible with code on CPU, GPU, and everything in between. We use [ports-of-call](https://lanl.github.io/ports-of-call/main/index.html) for this feature.
README.md:- `SPINER_USE_CUDA` enables or disables Cuda. Requires Kokkos. Default is `OFF`.
README.md:### CUDA and Kokkos
README.md:compilation with CUDA, Kokkos, or none of the above. If `Kokkos` is
README.md:The following spack install was tested with a V100 GPU:
README.md:spack install kokkos~shared+cuda+cuda_lambda+cuda_relocatable_device_code+wrapper cuda_arch=70
README.md:cmake -DSPINER_USE_KOKKOS=ON -DSPINER_USE_CUDA=ON -DBUILD_TESTING=ON -DCMAKE_CXX_COMPILER=nvcc_wrapper ..
README.md:builds the tests for CUDA.
spack-repo/packages/ports-of-call/package.py:        values=("Kokkos", "Cuda", "None"),
spack-repo/packages/ports-of-call/package.py:        values=("Kokkos", "Cuda", "None"),
spack-repo/packages/ports-of-call/package.py:        if self.spec.satisfies("test_portability_strategy=Kokkos ^kokkos+rocm"):
spack-repo/packages/ports-of-call/package.py:        if self.spec.satisfies("test_portability_strategy=Kokkos ^kokkos+cuda"):
spack-repo/packages/spiner/package.py:    # Currently the raw cuda backend of ports-of-call is not supported.
spack-repo/packages/spiner/package.py:        "kokkos ~shared+cuda_lambda+cuda_constexpr",
spack-repo/packages/spiner/package.py:        when="+kokkos ^kokkos+cuda",
spack-repo/packages/spiner/package.py:        if self.spec.satisfies("^kokkos+cuda"):
spack-repo/packages/spiner/package.py:                self.define("CMAKE_CUDA_ARCHITECTURES", self.spec["kokkos"].variants["cuda_arch"].value)
spack-repo/packages/spiner/package.py:        if self.spec.satisfies("^kokkos+rocm"):
spack-repo/packages/spiner/package.py:        if self.spec.satisfies("^kokkos+cuda"):
CMakeLists.txt:  SPINER_TEST_USE_KOKKOS_CUDA "Use kokkos cuda offloading for tests (affects submodule-build only)" ON
.gitignore:simple_test_cuda
.gitlab-ci.yml:openmpi_cuda_gcc_volta:
.gitlab-ci.yml:    SPACK_ENV_NAME: openmpi-cuda-gcc-volta
.gitlab-ci.yml:openmpi_cuda_gcc_ampere:
.gitlab-ci.yml:    SPACK_ENV_NAME: openmpi-cuda-gcc-ampere
.gitlab-ci.yml:    SCHEDULER_PARAMETERS: "-N 1 --qos=debug -p shared-gpu-ampere"
.gitlab-ci.yml:rzvernal_craympich_rocm_mi250_gcc:
.gitlab-ci.yml:    SPACK_ENV_NAME: craympich-rocm-gfx90a-gcc
.gitlab-ci.yml:rzadams_craympich_rocm_mi300_gcc:
.gitlab-ci.yml:    SPACK_ENV_NAME: craympich-rocm-gfx942-gcc

```
