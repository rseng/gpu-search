# https://github.com/ANTsX/ANTs

```console
CMake/CCache.cmake:    set(CUDA_LAUNCHER "${CCACHE_PROGRAM}")
CMake/CCache.cmake:    # Cuda support only added in CMake 3.10
CMake/CCache.cmake:    file(WRITE "${CMAKE_BINARY_DIR}/launch-cuda" ""
CMake/CCache.cmake:        "if [ \"$1\" = \"${CMAKE_CUDA_COMPILER}\" ] ; then\n"
CMake/CCache.cmake:        "exec \"${CUDA_LAUNCHER}\" \"${CMAKE_CUDA_COMPILER}\" \"$@\"\n"
CMake/CCache.cmake:                     "${CMAKE_BINARY_DIR}/launch-cuda"
CMake/CCache.cmake:        set(CMAKE_CUDA_COMPILER_LAUNCHER "${CMAKE_BINARY_DIR}/launch-cuda")
Utilities/antsSCCANObject.h:  MatchingPursuit(MatrixType & A, VectorType & x_k, RealType convcrit, unsigned int);
Utilities/antsSCCANObject.hxx:antsSCCANObject<TInputImage, TRealType>::MatchingPursuit(
Utilities/antsSCCANObject.hxx:      std::cout << " MatchingPursuit " << approxerr << " deltaminerr " << deltaminerr << " ct " << ct << " diag "
Utilities/antsSCCANObject.hxx:        // minerr1 = this->MatchingPursuit(  matrixP ,  randv,  0, 90  );/** bad */
SuperBuild.cmake:  CUDA_LAUNCHER:STRING
SuperBuild.cmake:  CMAKE_CUDA_COMPILER_LAUNCHER:STRING

```
