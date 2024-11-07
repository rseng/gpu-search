# https://github.com/QuaCaTeam/quaca

```console
include/catch.hpp:#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC) && !defined(__CUDACC__) && !defined(__LCC__)
include/catch.hpp:#  if !defined(__ibmxl__) && !defined(__CUDACC__)
include/catch.hpp:        // Code accompanying the article "Approximating the erfinv function" in GPU Computing Gems, Volume 2
include/armadillo-9.870.2/include/armadillo_bits/compiler_setup.hpp:  #if (defined(__GNUG__) || defined(__GNUC__)) && (defined(__INTEL_COMPILER) || defined(__NVCC__) || defined(__CUDACC__) || defined(__PGI) || defined(__PATHSCALE__) || defined(__ARMCC_VERSION) || defined(__IBMCPP__))
include/armadillo-9.870.2/include/armadillo_bits/compiler_setup.hpp:  #if defined(__clang__) && (defined(__INTEL_COMPILER) || defined(__NVCC__) || defined(__CUDACC__) || defined(__PGI) || defined(__PATHSCALE__) || defined(__ARMCC_VERSION) || defined(__IBMCPP__))

```
