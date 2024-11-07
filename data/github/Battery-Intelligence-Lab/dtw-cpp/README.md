# https://github.com/Battery-Intelligence-Lab/dtw-cpp

```console
develop/TODO.md:- [ ] GPU programming 
cmake/CompilerWarnings.cmake:  CUDA_WARNINGS)
cmake/CompilerWarnings.cmake:  if("${CUDA_WARNINGS}" STREQUAL "")
cmake/CompilerWarnings.cmake:    set(CUDA_WARNINGS
cmake/CompilerWarnings.cmake:        # TODO add more Cuda warnings
cmake/CompilerWarnings.cmake:  set(PROJECT_WARNINGS_CUDA "${CUDA_WARNINGS}")
cmake/CompilerWarnings.cmake:              # Cuda warnings
cmake/CompilerWarnings.cmake:              $<$<COMPILE_LANGUAGE:CUDA>:${PROJECT_WARNINGS_CUDA}>)
cmake/StandardProjectSettings.cmake:    # On Windows cuda nvcc uses cl and not clang
cmake/StandardProjectSettings.cmake:    # On Windows cuda nvcc uses cl and not gcc

```
