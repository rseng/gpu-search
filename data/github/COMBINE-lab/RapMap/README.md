# https://github.com/COMBINE-lab/RapMap

```console
include/sparsepp/spp_config.h:    // doesn't actually support __int128 as of CUDA_VERSION=7500
include/sparsepp/spp_config.h:    #if defined(__CUDACC__)
include/sparsepp/spp_config.h:    // Nevertheless, as of CUDA 7.5, using __float128 with the host

```
