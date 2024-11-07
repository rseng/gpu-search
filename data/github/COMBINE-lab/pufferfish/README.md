# https://github.com/COMBINE-lab/pufferfish

```console
include/parallel_hashmap/phmap_config.h:        (defined(__CUDACC__) && __CUDACC_VER_MAJOR__ >= 9) ||                \
include/parallel_hashmap/phmap_config.h:        (defined(__GNUC__) && !defined(__clang__) && !defined(__CUDACC__))
include/parallel_hashmap/phmap_config.h:    #elif defined(__CUDACC__)
include/parallel_hashmap/phmap_config.h:        #if __CUDACC_VER__ >= 70000
include/parallel_hashmap/phmap_config.h:        #endif  // __CUDACC_VER__ >= 70000
include/parallel_hashmap/phmap_config.h:    #endif  // defined(__CUDACC__)
include/simde/x86/sse.h: *   2015      Brandon Rowlett <browlett@nvidia.com>
include/simde/x86/sse2.h: *   2015      Brandon Rowlett <browlett@nvidia.com>
include/sparsepp/spp_config.h:    // doesn't actually support __int128 as of CUDA_VERSION=7500
include/sparsepp/spp_config.h:    #if defined(__CUDACC__)
include/sparsepp/spp_config.h:    // Nevertheless, as of CUDA 7.5, using __float128 with the host

```
