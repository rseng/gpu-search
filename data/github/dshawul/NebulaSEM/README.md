# https://github.com/dshawul/NebulaSEM

```console
README.md:Additional options to enable fine-grained parallelization on CPUs and GPUs
README.md:- [NVHPC](https://developer.nvidia.com/hpc-sdk) or other OpenACC compiler
README.md:### A note about MPI/OpenMP/OpenACC parallelization
paper/paper.bib:    title = {A {GPU}-accelerated continuous and discontinuous {Galerkin} non-hydrostatic atmospheric model},
paper/paper.md:the high arithmetic intensity per element, suitability for GPU acceleration, and support for both h- and p- refinement.
paper/paper.md:## Parallelization with MPI+OpenMP/OpenACC
paper/paper.md:such as OpenMP for CPUs and OpenACC for GPUs optimize fine-grained parallelism, minimizing communication overhead.
paper/paper.md:Efficient GPU implementation of dGSEM is achieved through offloading of all field computations to the GPU [@abdi8], 
paper/paper.md:managed memory to simplify the data transfer logic between CPU and GPU etc.
CMakeLists.txt:option( USE_ACC "Build with OpenACC enabled" OFF )
CMakeLists.txt:    set(OpenACC_ACCEL_TARGET tesla:managed)
CMakeLists.txt:    find_package(OpenACC COMPONENTS CXX REQUIRED)
src/field/field.h://Hack for OpenACC template instatiation issues
src/field/field.h:#ifdef _OPENACC
src/util/util.h:#ifdef _OPENACC
src/mesh/mesh.h:#ifdef _OPENACC
src/CMakeLists.txt:if(OpenACC_CXX_FOUND)
src/CMakeLists.txt:    target_link_libraries(nebulasem PUBLIC OpenACC::OpenACC_CXX)
src/CMakeLists.txt:    target_compile_options(nebulasem PUBLIC ${OpenACC_CXX_OPTIONS} "-Minfo=accel")
src/CMakeLists.txt:    target_link_options(nebulasem PUBLIC ${OpenACC_CXX_OPTIONS})

```
