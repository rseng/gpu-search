# https://github.com/hpc4cmb/toast

```console
wheels/install_deps_linux.sh:    GPU_CONFIG="" CFOPENMP="${OPENMP_CXXFLAGS}" BLAS="-lopenblas -fopenmp -lm" \
wheels/install_deps_linux.sh:    GPU_CONFIG="" CFOPENMP="${OPENMP_CXXFLAGS}" BLAS="-lopenblas -fopenmp -lm" \
wheels/install_deps_osx.sh:    GPU_CONFIG="" BLAS="-framework Accelerate" \
wheels/install_deps_osx.sh:    GPU_CONFIG="" BLAS="-framework Accelerate" \
src/libtoast/gtest/googlemock/test/gmock-matchers_test.cc:TEST(MatcherInterfaceTest, CanBeImplementedUsingPublishedAPI) {
src/libtoast/gtest/googletest/include/gtest/internal/gtest-port.h:// with a TR1 tuple implementation.  NVIDIA's CUDA NVCC compiler
src/libtoast/gtest/googletest/include/gtest/internal/gtest-port.h:# if (defined(__GNUC__) && !defined(__CUDACC__) && (GTEST_GCC_VER_ >= 40000) \
src/libtoast/src/Random123/features/iccfeatures.h:#ifndef R123_CUDA_DEVICE
src/libtoast/src/Random123/features/iccfeatures.h:#define R123_CUDA_DEVICE
src/libtoast/src/Random123/features/iccfeatures.h:#ifndef R123_USE_MULHILO64_CUDA_INTRIN
src/libtoast/src/Random123/features/iccfeatures.h:#define R123_USE_MULHILO64_CUDA_INTRIN 0
src/libtoast/src/Random123/features/iccfeatures.h:#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
src/libtoast/src/Random123/features/iccfeatures.h:#define R123_USE_MULHILO64_OPENCL_INTRIN 0
src/libtoast/src/Random123/features/sunprofeatures.h:#ifndef R123_CUDA_DEVICE
src/libtoast/src/Random123/features/sunprofeatures.h:#define R123_CUDA_DEVICE
src/libtoast/src/Random123/features/sunprofeatures.h:#ifndef R123_USE_MULHILO64_CUDA_INTRIN
src/libtoast/src/Random123/features/sunprofeatures.h:#define R123_USE_MULHILO64_CUDA_INTRIN 0
src/libtoast/src/Random123/features/sunprofeatures.h:#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
src/libtoast/src/Random123/features/sunprofeatures.h:#define R123_USE_MULHILO64_OPENCL_INTRIN 0
src/libtoast/src/Random123/features/nvccfeatures.h:#if !defined(CUDART_VERSION)
src/libtoast/src/Random123/features/nvccfeatures.h:#error "why are we in nvccfeatures.h if CUDART_VERSION is not defined"
src/libtoast/src/Random123/features/nvccfeatures.h:#if CUDART_VERSION < 4010
src/libtoast/src/Random123/features/nvccfeatures.h:#error "CUDA versions earlier than 4.1 produce incorrect results for some templated functions in namespaces.  Random123 isunsupported.  See comments in nvccfeatures.h"
src/libtoast/src/Random123/features/nvccfeatures.h:// T=uint64_t in examples/uniform.hpp produces -1 for CUDA4.0 and
src/libtoast/src/Random123/features/nvccfeatures.h:// Thus, we no longer trust CUDA versions earlier than 4.1 even though
src/libtoast/src/Random123/features/nvccfeatures.h:// we had previously tested and timed Random123 with CUDA 3.x and 4.0.
src/libtoast/src/Random123/features/nvccfeatures.h://#ifdef  __CUDA_ARCH__ allows Philox32 and Philox64 to be compiled
src/libtoast/src/Random123/features/nvccfeatures.h://for both device and host functions in CUDA by setting compiler flags
src/libtoast/src/Random123/features/nvccfeatures.h:#ifdef  __CUDA_ARCH__
src/libtoast/src/Random123/features/nvccfeatures.h:#ifndef R123_CUDA_DEVICE
src/libtoast/src/Random123/features/nvccfeatures.h:#define R123_CUDA_DEVICE __device__
src/libtoast/src/Random123/features/nvccfeatures.h:#ifndef R123_USE_MULHILO64_CUDA_INTRIN
src/libtoast/src/Random123/features/nvccfeatures.h:#define R123_USE_MULHILO64_CUDA_INTRIN 1
src/libtoast/src/Random123/features/nvccfeatures.h:// No exceptions in CUDA, at least upto 4.0
src/libtoast/src/Random123/features/nvccfeatures.h:#else // ! __CUDA_ARCH__
src/libtoast/src/Random123/features/nvccfeatures.h:// If we're using nvcc not compiling for the CUDA architecture,
src/libtoast/src/Random123/features/nvccfeatures.h:#endif // __CUDA_ARCH__
src/libtoast/src/Random123/features/compilerfeatures.h:The Random123 library is portable across C, C++, CUDA, OpenCL environments,
src/libtoast/src/Random123/features/compilerfeatures.h:         MULHILO64_CUDA_INTRIN
src/libtoast/src/Random123/features/compilerfeatures.h:         MULHILO64_OPENCL_INTRIN
src/libtoast/src/Random123/features/compilerfeatures.h:(i.e. u01_*_53()), e.g. on OpenCL without the cl_khr_fp64 extension.
src/libtoast/src/Random123/features/compilerfeatures.h:<li>R123_CUDA_DEVICE - which expands to __device__ (or something else with
src/libtoast/src/Random123/features/compilerfeatures.h:  sufficiently similar semantics) when CUDA is in use, and expands
src/libtoast/src/Random123/features/compilerfeatures.h:  call assert (I'm looking at you, CUDA and OpenCL), or even include
src/libtoast/src/Random123/features/compilerfeatures.h:  assert.h safely (OpenCL).
src/libtoast/src/Random123/features/compilerfeatures.h:  is not available, e.g., MSVC and OpenCL.
src/libtoast/src/Random123/features/compilerfeatures.h:#if defined(__OPENCL_VERSION__) && __OPENCL_VERSION__ > 0
src/libtoast/src/Random123/features/compilerfeatures.h:#include "openclfeatures.h"
src/libtoast/src/Random123/features/compilerfeatures.h:#elif defined(__CUDACC__)
src/libtoast/src/Random123/features/compilerfeatures.h:#define R123_USE_PHILOX_64BIT (R123_USE_MULHILO64_ASM || R123_USE_MULHILO64_MSVC_INTRIN || R123_USE_MULHILO64_CUDA_INTRIN || R123_USE_GNU_UINT128 || R123_USE_MULHILO64_C99 || R123_USE_MULHILO64_OPENCL_INTRIN || R123_USE_MULHILO64_MULHI_INTRIN)
src/libtoast/src/Random123/features/pgccfeatures.h:#ifndef R123_CUDA_DEVICE
src/libtoast/src/Random123/features/pgccfeatures.h:#define R123_CUDA_DEVICE
src/libtoast/src/Random123/features/pgccfeatures.h:#ifndef R123_USE_MULHILO64_CUDA_INTRIN
src/libtoast/src/Random123/features/pgccfeatures.h:#define R123_USE_MULHILO64_CUDA_INTRIN 0
src/libtoast/src/Random123/features/pgccfeatures.h:#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
src/libtoast/src/Random123/features/pgccfeatures.h:#define R123_USE_MULHILO64_OPENCL_INTRIN 0
src/libtoast/src/Random123/features/msvcfeatures.h:#ifndef R123_CUDA_DEVICE
src/libtoast/src/Random123/features/msvcfeatures.h:#define R123_CUDA_DEVICE
src/libtoast/src/Random123/features/msvcfeatures.h:#ifndef R123_USE_MULHILO64_CUDA_INTRIN
src/libtoast/src/Random123/features/msvcfeatures.h:#define R123_USE_MULHILO64_CUDA_INTRIN 0
src/libtoast/src/Random123/features/msvcfeatures.h:#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
src/libtoast/src/Random123/features/msvcfeatures.h:#define R123_USE_MULHILO64_OPENCL_INTRIN 0
src/libtoast/src/Random123/features/openclfeatures.h:#ifndef __openclfeatures_dot_hpp
src/libtoast/src/Random123/features/openclfeatures.h:#define __openclfeatures_dot_hpp
src/libtoast/src/Random123/features/openclfeatures.h:#ifndef R123_CUDA_DEVICE
src/libtoast/src/Random123/features/openclfeatures.h:#define R123_CUDA_DEVICE
src/libtoast/src/Random123/features/openclfeatures.h:#ifndef R123_USE_MULHILO64_CUDA_INTRIN
src/libtoast/src/Random123/features/openclfeatures.h:#define R123_USE_MULHILO64_CUDA_INTRIN 0
src/libtoast/src/Random123/features/openclfeatures.h:#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
src/libtoast/src/Random123/features/openclfeatures.h:#define R123_USE_MULHILO64_OPENCL_INTRIN 1
src/libtoast/src/Random123/features/xlcfeatures.h:#ifndef R123_CUDA_DEVICE
src/libtoast/src/Random123/features/xlcfeatures.h:#define R123_CUDA_DEVICE
src/libtoast/src/Random123/features/xlcfeatures.h:#ifndef R123_USE_MULHILO64_CUDA_INTRIN
src/libtoast/src/Random123/features/xlcfeatures.h:#define R123_USE_MULHILO64_CUDA_INTRIN 0
src/libtoast/src/Random123/features/xlcfeatures.h:#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
src/libtoast/src/Random123/features/xlcfeatures.h:#define R123_USE_MULHILO64_OPENCL_INTRIN 0
src/libtoast/src/Random123/features/gccfeatures.h:#ifndef R123_CUDA_DEVICE
src/libtoast/src/Random123/features/gccfeatures.h:#define R123_CUDA_DEVICE
src/libtoast/src/Random123/features/gccfeatures.h:#ifndef R123_USE_MULHILO64_CUDA_INTRIN
src/libtoast/src/Random123/features/gccfeatures.h:#define R123_USE_MULHILO64_CUDA_INTRIN 0
src/libtoast/src/Random123/features/gccfeatures.h:#ifndef R123_USE_MULHILO64_OPENCL_INTRIN
src/libtoast/src/Random123/features/gccfeatures.h:#define R123_USE_MULHILO64_OPENCL_INTRIN 0
src/libtoast/src/Random123/uniform.hpp:#if defined(__CUDACC__) || defined(_LIBCPP_HAS_NO_CONSTEXPR)
src/libtoast/src/Random123/uniform.hpp:// Amazing! cuda thinks numeric_limits::max() is a __host__ function, so
src/libtoast/src/Random123/uniform.hpp:R123_CONSTEXPR R123_STATIC_INLINE R123_CUDA_DEVICE T maxTvalue(){
src/libtoast/src/Random123/uniform.hpp:R123_CUDA_DEVICE R123_STATIC_INLINE Ftype u01(Itype in){
src/libtoast/src/Random123/uniform.hpp:R123_CUDA_DEVICE R123_STATIC_INLINE Ftype uneg11(Itype in){
src/libtoast/src/Random123/uniform.hpp:R123_CUDA_DEVICE R123_STATIC_INLINE Ftype u01fixedpt(Itype in){
src/libtoast/src/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE Word mulhilo##W(Word a, Word b, Word* hip){ \
src/libtoast/src/Random123/philox.h:   happens that CUDA was the first time we used the idiom. */
src/libtoast/src/Random123/philox.h:#define _mulhilo_cuda_intrin_tpl(W, Word, INTRIN)                       \
src/libtoast/src/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE Word mulhilo##W(Word a, Word b, Word* hip){ \
src/libtoast/src/Random123/philox.h:#elif R123_USE_MULHILO64_CUDA_INTRIN
src/libtoast/src/Random123/philox.h:_mulhilo_cuda_intrin_tpl(64, uint64_t, __umul64hi)
src/libtoast/src/Random123/philox.h:#elif R123_USE_MULHILO64_OPENCL_INTRIN
src/libtoast/src/Random123/philox.h:_mulhilo_cuda_intrin_tpl(64, uint64_t, mul_hi)
src/libtoast/src/Random123/philox.h:_mulhilo_cuda_intrin_tpl(64, uint64_t, R123_MULHILO64_MULHI_INTRIN)
src/libtoast/src/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(struct r123array2x##W _philox2x##W##round(struct r123array2x##W ctr, struct r123array1x##W key)); \
src/libtoast/src/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE struct r123array2x##W _philox2x##W##round(struct r123array2x##W ctr, struct r123array1x##W key){ \
src/libtoast/src/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE struct r123array1x##W _philox2x##W##bumpkey( struct r123array1x##W key) { \
src/libtoast/src/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(struct r123array4x##W _philox4x##W##round(struct r123array4x##W ctr, struct r123array2x##W key)); \
src/libtoast/src/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE struct r123array4x##W _philox4x##W##round(struct r123array4x##W ctr, struct r123array2x##W key){ \
src/libtoast/src/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE struct r123array2x##W _philox4x##W##bumpkey( struct r123array2x##W key) { \
src/libtoast/src/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE philox##N##x##W##_key_t philox##N##x##W##keyinit(philox##N##x##W##_ukey_t uk) { return uk; } \
src/libtoast/src/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(philox##N##x##W##_ctr_t philox##N##x##W##_R(unsigned int R, philox##N##x##W##_ctr_t ctr, philox##N##x##W##_key_t key)); \
src/libtoast/src/Random123/philox.h:R123_CUDA_DEVICE R123_STATIC_INLINE philox##N##x##W##_ctr_t philox##N##x##W##_R(unsigned int R, philox##N##x##W##_ctr_t ctr, philox##N##x##W##_key_t key) { \
src/libtoast/src/Random123/philox.h:    inline R123_CUDA_DEVICE R123_FORCE_INLINE(ctr_type operator()(ctr_type ctr, key_type key) const){ \
src/libtoast/src/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(uint64_t RotL_64(uint64_t x, unsigned int N));
src/libtoast/src/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE uint64_t RotL_64(uint64_t x, unsigned int N)
src/libtoast/src/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(uint32_t RotL_32(uint32_t x, unsigned int N));
src/libtoast/src/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE uint32_t RotL_32(uint32_t x, unsigned int N)
src/libtoast/src/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE threefry2x##W##_key_t threefry2x##W##keyinit(threefry2x##W##_ukey_t uk) { return uk; } \
src/libtoast/src/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(threefry2x##W##_ctr_t threefry2x##W##_R(unsigned int Nrounds, threefry2x##W##_ctr_t in, threefry2x##W##_key_t k)); \
src/libtoast/src/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE                                          \
src/libtoast/src/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(threefry2x##W##_ctr_t threefry2x##W(threefry2x##W##_ctr_t in, threefry2x##W##_key_t k)); \
src/libtoast/src/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE                                     \
src/libtoast/src/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE threefry4x##W##_key_t threefry4x##W##keyinit(threefry4x##W##_ukey_t uk) { return uk; } \
src/libtoast/src/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(threefry4x##W##_ctr_t threefry4x##W##_R(unsigned int Nrounds, threefry4x##W##_ctr_t in, threefry4x##W##_key_t k)); \
src/libtoast/src/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE                                          \
src/libtoast/src/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE R123_FORCE_INLINE(threefry4x##W##_ctr_t threefry4x##W(threefry4x##W##_ctr_t in, threefry4x##W##_key_t k)); \
src/libtoast/src/Random123/threefry.h:R123_CUDA_DEVICE R123_STATIC_INLINE                                     \
src/libtoast/src/Random123/threefry.h:   inline R123_CUDA_DEVICE R123_FORCE_INLINE(ctr_type operator()(ctr_type ctr, key_type key)){ \
src/libtoast/src/Random123/array.h:inline R123_CUDA_DEVICE value_type assemble_from_u32(uint32_t *p32){
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE reference operator[](size_type i){return v[i];}                     \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE const_reference operator[](size_type i) const {return v[i];}        \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE reference at(size_type i){ if(i >=  _N) R123_THROW(std::out_of_range("array index out of range")); return (*this)[i]; } \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE const_reference at(size_type i) const { if(i >=  _N) R123_THROW(std::out_of_range("array index out of range")); return (*this)[i]; } \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE size_type size() const { return  _N; }                              \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE size_type max_size() const { return _N; }                           \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE bool empty() const { return _N==0; };                               \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE iterator begin() { return &v[0]; }                                  \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE iterator end() { return &v[_N]; }                                   \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE const_iterator begin() const { return &v[0]; }                      \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE const_iterator end() const { return &v[_N]; }                       \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE const_iterator cbegin() const { return &v[0]; }                     \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE const_iterator cend() const { return &v[_N]; }                      \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE reverse_iterator rbegin(){ return reverse_iterator(end()); }        \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE const_reverse_iterator rbegin() const{ return const_reverse_iterator(end()); } \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE reverse_iterator rend(){ return reverse_iterator(begin()); }        \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE const_reverse_iterator rend() const{ return const_reverse_iterator(begin()); } \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE const_reverse_iterator crbegin() const{ return const_reverse_iterator(cend()); } \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE const_reverse_iterator crend() const{ return const_reverse_iterator(cbegin()); } \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE pointer data(){ return &v[0]; }                                     \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE const_pointer data() const{ return &v[0]; }                         \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE reference front(){ return v[0]; }                                   \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE const_reference front() const{ return v[0]; }                       \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE reference back(){ return v[_N-1]; }                                 \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE const_reference back() const{ return v[_N-1]; }                     \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE bool operator==(const r123array##_N##x##W& rhs) const{ \
src/libtoast/src/Random123/array.h:	/* CUDA3 does not have std::equal */ \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE bool operator!=(const r123array##_N##x##W& rhs) const{ return !(*this == rhs); } \
src/libtoast/src/Random123/array.h:    /* CUDA3 does not have std::fill_n */ \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE void fill(const value_type& val){ for (size_t i = 0; i < _N; ++i) v[i] = val; } \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE void swap(r123array##_N##x##W& rhs){ \
src/libtoast/src/Random123/array.h:	/* CUDA3 does not have std::swap_ranges */ \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE r123array##_N##x##W& incr(R123_ULONG_LONG n=1){                         \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE static r123array##_N##x##W seed(SeedSeq &ss){      \
src/libtoast/src/Random123/array.h:    R123_CUDA_DEVICE r123array##_N##x##W& incr_carefully(R123_ULONG_LONG n){ \
src/toast/pybind11/README.rst:6. NVCC (CUDA 11.0 tested in CI)
src/toast/pybind11/README.rst:7. NVIDIA PGI (20.9 tested in CI)
src/toast/pybind11/tools/pybind11Common.cmake:      # instance, projects that include other types of source files like CUDA
src/toast/pybind11/tools/pybind11Tools.cmake:  if(NOT DEFINED CMAKE_CUDA_VISIBILITY_PRESET)
src/toast/pybind11/tools/pybind11Tools.cmake:    set_target_properties(${target_name} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
src/toast/pybind11/tools/pybind11NewTools.cmake:  if(NOT DEFINED CMAKE_CUDA_VISIBILITY_PRESET)
src/toast/pybind11/tools/pybind11NewTools.cmake:    set_target_properties(${target_name} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
src/toast/pybind11/include/pybind11/detail/common.h:// For CUDA, GCC7, GCC8:
src/toast/pybind11/include/pybind11/detail/common.h:// 1.7% for CUDA, -0.2% for GCC7, and 0.0% for GCC8 (using -DCMAKE_BUILD_TYPE=MinSizeRel,
src/toast/pybind11/include/pybind11/detail/common.h:    && (defined(__CUDACC__) || (defined(__GNUC__) && (__GNUC__ == 7 || __GNUC__ == 8)))
src/toast/pybind11/include/pybind11/cast.h:    // static_cast works around compiler error with MSVC 17 and CUDA 10.2
src/toast/pybind11/include/pybind11/numpy.h:#ifdef __CUDACC__

```
