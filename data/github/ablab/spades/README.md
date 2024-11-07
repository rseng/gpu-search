# https://github.com/ablab/spades

```console
ext/include/parallel_hashmap/phmap_config.h:        (defined(__CUDACC__) && __CUDACC_VER_MAJOR__ >= 9) ||                \
ext/include/parallel_hashmap/phmap_config.h:        (defined(__GNUC__) && !defined(__clang__) && !defined(__CUDACC__))
ext/include/parallel_hashmap/phmap_config.h:    #elif defined(__CUDACC__)
ext/include/parallel_hashmap/phmap_config.h:        #if __CUDACC_VER__ >= 70000
ext/include/parallel_hashmap/phmap_config.h:        #endif  // __CUDACC_VER__ >= 70000
ext/include/parallel_hashmap/phmap_config.h:    #endif  // defined(__CUDACC__)
ext/include/blaze/blaze/math/Constraints.h:#include <blaze/math/constraints/CUDAAssignable.h>
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h://  \file blaze/math/typetraits/IsCUDAAssignable.h
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h://  \brief Header file for the IsCUDAAssignable type trait
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:#ifndef _BLAZE_MATH_TYPETRAITS_ISCUDAASSIGNABLE_H_
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:#define _BLAZE_MATH_TYPETRAITS_ISCUDAASSIGNABLE_H_
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:template< typename T > struct IsCUDAAssignable;
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:/*!\brief Auxiliary helper struct for the IsCUDAAssignable type trait.
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:struct IsCUDAAssignableHelper
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:struct IsCUDAAssignableHelper< T, Void_t< decltype( T::cudaAssignable ) > >
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:   : public BoolConstant< T::cudaAssignable >
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:struct IsCUDAAssignableHelper< T, EnableIf_t< IsExpression_v<T> > >
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:   : public IsCUDAAssignable< typename T::ResultType >::Type
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:// This type trait tests whether or not the given template parameter is an CUDA-assignable data
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:// CUDA-assignable, whereas several vector and matrix types (as for instance DynamicVector and
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:// DynamicMatrix) can be CUDA-assignable. If the type is CUDA-assignable, the \a value member
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:   blaze::IsCUDAAssignable< VectorType >::value            // Evaluates to 1
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:   blaze::IsCUDAAssignable< SubvectorType >::Type          // Results in TrueType
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:   blaze::IsCUDAAssignable< CUDADynamicMatrix<int> >       // Is derived from TrueType
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:   blaze::IsCUDAAssignable< int >::value                   // Evaluates to 0
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:   blaze::IsCUDAAssignable< StaticVector<int,3UL> >::Type  // Results in FalseType
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:   blaze::IsCUDAAssignable< StaticMatrix<int,4UL,5UL> >    // Is derived from FalseType
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:struct IsCUDAAssignable
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:   : public IsCUDAAssignableHelper<T>
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:/*!\brief Auxiliary variable template for the IsCUDAAssignable type trait.
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:// The IsCUDAAssignable_v variable template provides a convenient shortcut to access the nested
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:// \a value of the IsCUDAAssignable class template. For instance, given the type \a T the
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:   constexpr bool value1 = blaze::IsCUDAAssignable<T>::value;
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:   constexpr bool value2 = blaze::IsCUDAAssignable_v<T>;
ext/include/blaze/blaze/math/typetraits/IsCUDAAssignable.h:constexpr bool IsCUDAAssignable_v = IsCUDAAssignable<T>::value;
ext/include/blaze/blaze/math/TypeTraits.h:#include <blaze/math/typetraits/IsCUDAAssignable.h>
ext/include/blaze/blaze/math/constraints/CUDAAssignable.h://  \file blaze/math/constraints/CUDAAssignable.h
ext/include/blaze/blaze/math/constraints/CUDAAssignable.h:#ifndef _BLAZE_MATH_CONSTRAINTS_CUDAASSIGNABLE_H_
ext/include/blaze/blaze/math/constraints/CUDAAssignable.h:#define _BLAZE_MATH_CONSTRAINTS_CUDAASSIGNABLE_H_
ext/include/blaze/blaze/math/constraints/CUDAAssignable.h:#include <blaze/math/typetraits/IsCUDAAssignable.h>
ext/include/blaze/blaze/math/constraints/CUDAAssignable.h://  MUST_BE_CUDA_ASSIGNABLE CONSTRAINT
ext/include/blaze/blaze/math/constraints/CUDAAssignable.h:// In case the given data type \a T is not CUDA-assignable (i.e. does not provide the according
ext/include/blaze/blaze/math/constraints/CUDAAssignable.h:// CUDA member constants or returns \a false), a compilation error is created.
ext/include/blaze/blaze/math/constraints/CUDAAssignable.h:#define BLAZE_CONSTRAINT_MUST_BE_CUDA_ASSIGNABLE(T) \
ext/include/blaze/blaze/math/constraints/CUDAAssignable.h:   static_assert( ::blaze::IsCUDAAssignable_v<T>, "Non-CUDA assignable type detected" )
ext/include/blaze/blaze/math/constraints/CUDAAssignable.h://  MUST_NOT_BE_CUDA_ASSIGNABLE CONSTRAINT
ext/include/blaze/blaze/math/constraints/CUDAAssignable.h:// In case the given data type \a T is CUDA-assignable (i.e. does provide the according CUDA
ext/include/blaze/blaze/math/constraints/CUDAAssignable.h:#define BLAZE_CONSTRAINT_MUST_NOT_BE_CUDA_ASSIGNABLE(T) \
ext/include/blaze/blaze/math/constraints/CUDAAssignable.h:   static_assert( !::blaze::IsCUDAAssignable_v<T>, "CUDA-assignable type detected" )
ext/include/blaze/blaze/system/HostDevice.h://  \brief Macro for CUDA compatibility
ext/include/blaze/blaze/system/HostDevice.h://  CUDA MACRO DEFINITIONS
ext/include/blaze/blaze/system/HostDevice.h:// \brief Conditional macro that sets __global__ attribute when compiled with CUDA.
ext/include/blaze/blaze/system/HostDevice.h:#ifdef __CUDACC__
ext/include/blaze/blaze/system/HostDevice.h:// \brief Conditional macro that sets __device__ attribute when compiled with CUDA.
ext/include/blaze/blaze/system/HostDevice.h:#ifdef __CUDACC__
ext/include/blaze/blaze/system/HostDevice.h:// \brief Conditional macro that sets __host__ attribute when compiled with CUDA.
ext/include/blaze/blaze/system/HostDevice.h:#ifdef __CUDACC__
ext/include/blaze/blaze/system/HostDevice.h:// \brief Conditional macro that sets __host__ and __device__ attributes when compiled with CUDA.
ext/include/blaze/blaze/system/HostDevice.h:#ifdef __CUDACC__
ext/include/sse2neon/sse2neon.h://   Brandon Rowlett <browlett@nvidia.com>
ext/include/sse2neon/sse2neon.h://   Eric van Beurden <evanbeurden@nvidia.com>
ext/include/sse2neon/sse2neon.h://   Alexander Potylitsin <apotylitsin@nvidia.com>
ext/include/boost/math/tools/config.hpp:#    if defined(__CUDACC__)
ext/include/boost/math/special_functions/next.hpp:#if !defined(_CRAYC) && !defined(__CUDACC__) && (!defined(__GNUC__) || (__GNUC__ > 3) || ((__GNUC__ == 3) && (__GNUC_MINOR__ > 3)))
ext/include/boost/math/special_functions/lanczos.hpp:#if !defined(_CRAYC) && !defined(__CUDACC__) && (!defined(__GNUC__) || (__GNUC__ > 3) || ((__GNUC__ == 3) && (__GNUC_MINOR__ > 3)))
ext/include/boost/config/detail/select_compiler_config.hpp:#if defined __CUDACC__
ext/include/boost/config/detail/select_compiler_config.hpp://  NVIDIA CUDA C++ compiler for GPU
ext/include/boost/config/detail/suffix.hpp:// If we're on a CUDA device (note DEVICE not HOST, irrespective of compiler) then disable __int128 and __float128 support if present:
ext/include/boost/config/detail/suffix.hpp:#if defined(__CUDA_ARCH__) && defined(BOOST_HAS_FLOAT128)
ext/include/boost/config/detail/suffix.hpp:#if defined(__CUDA_ARCH__) && defined(BOOST_HAS_INT128)
ext/include/boost/config/detail/suffix.hpp:// Set some default values GPU support
ext/include/boost/config/detail/suffix.hpp:#  ifndef BOOST_GPU_ENABLED
ext/include/boost/config/detail/suffix.hpp:#  define BOOST_GPU_ENABLED
ext/include/boost/config/detail/suffix.hpp:#    if defined(__CUDACC__)
ext/include/boost/config/compiler/gcc.hpp:#if !defined(__CUDACC__)
ext/include/boost/config/compiler/gcc.hpp:// doesn't actually support __int128 as of CUDA_VERSION=7500
ext/include/boost/config/compiler/gcc.hpp:#if defined(__CUDACC__)
ext/include/boost/config/compiler/gcc.hpp:// Nevertheless, as of CUDA 7.5, using __float128 with the host
ext/include/boost/config/compiler/nvcc.hpp://  NVIDIA CUDA C++ compiler setup
ext/include/boost/config/compiler/nvcc.hpp:#  define BOOST_COMPILER "NVIDIA CUDA C++ Compiler"
ext/include/boost/config/compiler/nvcc.hpp:#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && defined(__CUDACC_VER_BUILD__)
ext/include/boost/config/compiler/nvcc.hpp:#  define BOOST_CUDA_VERSION (__CUDACC_VER_MAJOR__ * 1000000 + __CUDACC_VER_MINOR__ * 10000 + __CUDACC_VER_BUILD__)
ext/include/boost/config/compiler/nvcc.hpp:// We don't really know what the CUDA version is, but it's definitely before 7.5:
ext/include/boost/config/compiler/nvcc.hpp:#  define BOOST_CUDA_VERSION 7000000
ext/include/boost/config/compiler/nvcc.hpp:// NVIDIA Specific support
ext/include/boost/config/compiler/nvcc.hpp:// BOOST_GPU_ENABLED : Flag a function or a method as being enabled on the host and device
ext/include/boost/config/compiler/nvcc.hpp:#define BOOST_GPU_ENABLED __host__ __device__
ext/include/boost/config/compiler/nvcc.hpp:// A bug in version 7.0 of CUDA prevents use of variadic templates in some occasions
ext/include/boost/config/compiler/nvcc.hpp:#if BOOST_CUDA_VERSION < 7050000
ext/include/boost/config/compiler/nvcc.hpp:#if (BOOST_CUDA_VERSION > 8000000) && (BOOST_CUDA_VERSION < 8010000)
ext/include/boost/config/compiler/nvcc.hpp:// CUDA (8.0) has no constexpr support in msvc mode:
ext/include/boost/config/compiler/nvcc.hpp:#if defined(_MSC_VER) && (BOOST_CUDA_VERSION < 9000000)
ext/include/boost/config/compiler/nvcc.hpp:#ifdef __CUDACC__
ext/include/boost/config/compiler/nvcc.hpp:#if (BOOST_CUDA_VERSION >= 8000000) && (BOOST_CUDA_VERSION < 8010000)
ext/include/boost/config/compiler/intel.hpp:#if defined(__CUDACC__)
ext/include/boost/config/compiler/pgi.hpp://  Copyright 2017, NVIDIA CORPORATION.
ext/include/boost/config/compiler/clang.hpp:// doesn't actually support __int128 as of CUDA_VERSION=7500
ext/include/boost/config/compiler/clang.hpp:#if defined(__CUDACC__)
ext/include/boost/mpl/aux_/config/gpu.hpp:#ifndef BOOST_MPL_AUX_CONFIG_GPU_HPP_INCLUDED
ext/include/boost/mpl/aux_/config/gpu.hpp:#define BOOST_MPL_AUX_CONFIG_GPU_HPP_INCLUDED
ext/include/boost/mpl/aux_/config/gpu.hpp:#if !defined(BOOST_MPL_CFG_GPU_ENABLED) \
ext/include/boost/mpl/aux_/config/gpu.hpp:#   define BOOST_MPL_CFG_GPU_ENABLED BOOST_GPU_ENABLED
ext/include/boost/mpl/aux_/config/gpu.hpp:#if defined __CUDACC__
ext/include/boost/mpl/aux_/config/gpu.hpp:#    define BOOST_MPL_CFG_GPU 1
ext/include/boost/mpl/aux_/config/gpu.hpp:#    define BOOST_MPL_CFG_GPU 0
ext/include/boost/mpl/aux_/config/gpu.hpp:#endif // BOOST_MPL_AUX_CONFIG_GPU_HPP_INCLUDED
ext/include/boost/mpl/assert.hpp:#include <boost/mpl/aux_/config/gpu.hpp>
ext/include/boost/mpl/assert.hpp:    || (BOOST_MPL_CFG_GCC != 0) || (BOOST_MPL_CFG_GPU != 0) || defined(__PGI) || defined(__clang__)
ext/include/boost/mpl/has_xxx.hpp:      || (BOOST_WORKAROUND(BOOST_MSVC, BOOST_TESTED_AT(1800)) && defined(__CUDACC__)) \
ext/include/boost/core/invoke_swap.hpp:BOOST_GPU_ENABLED
ext/include/boost/core/invoke_swap.hpp:BOOST_GPU_ENABLED
ext/include/boost/core/invoke_swap.hpp:BOOST_GPU_ENABLED
ext/include/boost/move/detail/workaround.hpp:#if !defined(__has_cpp_attribute) || defined(__CUDACC__)
ext/include/boost/mp11/detail/mp_defer.hpp:#if BOOST_MP11_WORKAROUND( BOOST_MP11_CUDA, >= 9000000 && BOOST_MP11_CUDA < 10000000 )
ext/include/boost/mp11/detail/mp_defer.hpp:template<template<class...> class F, class... T> struct mp_defer_cuda_workaround
ext/include/boost/mp11/detail/mp_defer.hpp:#if BOOST_MP11_WORKAROUND( BOOST_MP11_CUDA, >= 9000000 && BOOST_MP11_CUDA < 10000000 )
ext/include/boost/mp11/detail/mp_defer.hpp:template<template<class...> class F, class... T> using mp_defer = typename detail::mp_defer_cuda_workaround< F, T...>::type;
ext/include/boost/mp11/detail/mp_append.hpp:#if BOOST_MP11_WORKAROUND( BOOST_MP11_CUDA, >= 9000000 && BOOST_MP11_CUDA < 10000000 )
ext/include/boost/mp11/detail/mp_append.hpp:struct mp_append_impl_cuda_workaround
ext/include/boost/mp11/detail/mp_append.hpp:template<class... L> struct mp_append_impl: mp_append_impl_cuda_workaround<L...>::type::template fn<L...>
ext/include/boost/mp11/detail/mp_append.hpp:#endif // #if BOOST_MP11_WORKAROUND( BOOST_MP11_CUDA, >= 9000000 && BOOST_MP11_CUDA < 10000000 )
ext/include/boost/mp11/detail/config.hpp:#define BOOST_MP11_CUDA 0
ext/include/boost/mp11/detail/config.hpp:#if defined( __CUDACC__ )
ext/include/boost/mp11/detail/config.hpp:# undef BOOST_MP11_CUDA
ext/include/boost/mp11/detail/config.hpp:# define BOOST_MP11_CUDA (__CUDACC_VER_MAJOR__ * 1000000 + __CUDACC_VER_MINOR__ * 10000 + __CUDACC_VER_BUILD__)
ext/include/boost/mp11/detail/config.hpp:// CUDA (8.0) has no constexpr support in msvc mode:
ext/include/boost/mp11/detail/config.hpp:# if defined(_MSC_VER) && (BOOST_MP11_CUDA < 9000000)
ext/include/boost/mp11/algorithm.hpp:#if BOOST_MP11_WORKAROUND( BOOST_MP11_CUDA, >= 9000000 && BOOST_MP11_CUDA < 10000000 )
ext/include/boost/mp11/algorithm.hpp:template<template<class...> class F, class... L> struct mp_transform_cuda_workaround
ext/include/boost/mp11/algorithm.hpp:#if BOOST_MP11_WORKAROUND( BOOST_MP11_CUDA, >= 9000000 && BOOST_MP11_CUDA < 10000000 )
ext/include/boost/mp11/algorithm.hpp:template<template<class...> class F, class... L> using mp_transform = typename detail::mp_transform_cuda_workaround< F, L...>::type::type;
ext/include/boost/mp11/algorithm.hpp:#if BOOST_MP11_WORKAROUND( BOOST_MP11_CUDA, >= 9000000 && BOOST_MP11_CUDA < 10000000 )
ext/include/boost/mp11/algorithm.hpp:template<class L, std::size_t I> struct mp_at_c_cuda_workaround
ext/include/boost/mp11/algorithm.hpp:#if BOOST_MP11_WORKAROUND( BOOST_MP11_CUDA, >= 9000000 && BOOST_MP11_CUDA < 10000000 )
ext/include/boost/mp11/algorithm.hpp:template<class L, std::size_t I> using mp_at_c = typename detail::mp_at_c_cuda_workaround< L, I >::type::type;
ext/include/boost/mp11/algorithm.hpp:#if BOOST_MP11_WORKAROUND( BOOST_MP11_CUDA, >= 9000000 && BOOST_MP11_CUDA < 10000000 )
ext/include/boost/mp11/algorithm.hpp:        struct mp_nth_element_impl_cuda_workaround
ext/include/boost/mp11/algorithm.hpp:    using type = typename detail::mp_nth_element_impl_cuda_workaround::type::type;
ext/include/boost/type_traits/detail/config.hpp:#if defined(__cpp_rvalue_references) && defined(__NVCC__) && defined(__CUDACC__) && !defined(BOOST_TT_NO_NOEXCEPT_SEPARATE_TYPE)
ext/include/boost/type_traits/intrinsics.hpp:#if defined(BOOST_CLANG) && defined(__has_feature) && defined(__has_builtin) && (!(defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ < 11)) || defined(__CUDA__))
ext/include/boost/type_traits/intrinsics.hpp:// Note that these intrinsics are disabled for the CUDA meta-compiler as it appears
ext/include/boost/predef/architecture/ptx.h:| `+__CUDA_ARCH__+` | {predef_detection}
ext/include/boost/predef/architecture/ptx.h:| `+__CUDA_ARCH__+` | V.R.0
ext/include/boost/predef/architecture/ptx.h:#if defined(__CUDA_ARCH__)
ext/include/boost/predef/architecture/ptx.h:#   define BOOST_ARCH_PTX BOOST_PREDEF_MAKE_10_VR0(__CUDA_ARCH__)
ext/include/boost/concept/detail/has_constraints.hpp:#if BOOST_WORKAROUND(__SUNPRO_CC, <= 0x580) || defined(__CUDACC__)
ext/src/llvm/TargetParser.h:namespace AMDGPU {
ext/src/llvm/TargetParser.h:/// GPU kinds supported by the AMDGPU target.
ext/src/llvm/TargetParser.h:enum GPUKind : uint32_t {
ext/src/llvm/TargetParser.h:StringRef getArchNameAMDGCN(GPUKind AK);
ext/src/llvm/TargetParser.h:StringRef getArchNameR600(GPUKind AK);
ext/src/llvm/TargetParser.h:GPUKind parseArchAMDGCN(StringRef CPU);
ext/src/llvm/TargetParser.h:GPUKind parseArchR600(StringRef CPU);
ext/src/llvm/TargetParser.h:unsigned getArchAttrAMDGCN(GPUKind AK);
ext/src/llvm/TargetParser.h:unsigned getArchAttrR600(GPUKind AK);
ext/src/llvm/TargetParser.h:IsaVersion getIsaVersion(StringRef GPU);
ext/src/llvm/TargetParser.h:} // namespace AMDGPU
ext/src/llvm/Process.inc:  if (sigprocmask(SIG_SETMASK, &FullSet, &SavedSet) < 0)
ext/src/llvm/Process.inc:  if (sigprocmask(SIG_SETMASK, &SavedSet, nullptr) < 0)
ext/src/llvm/CrashRecoveryContext.cpp:  sigprocmask(SIG_UNBLOCK, &SigMask, nullptr);
ext/src/llvm/Signals.inc:  sigprocmask(SIG_UNBLOCK, &SigMask, nullptr);
ext/src/googletest/googlemock/test/gmock-matchers_test.cc:TEST(MatcherInterfaceTest, CanBeImplementedUsingPublishedAPI) {
ext/src/easel/esl_config.h.cmake:#cmakedefine eslENABLE_CUDA  // Should we build CUDA acceleration?
ext/src/easel/esl_config.h.in:#undef eslENABLE_CUDA              // Should we build CUDA GPU acceleration?

```
