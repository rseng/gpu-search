# https://github.com/jordanlab/STing

```console
share/cmake/Modules/FindSeqAn.cmake:set(_SEQAN_ALL_LIBRARIES     ZLIB BZip2 OpenMP CUDA)
share/cmake/Modules/FindSeqAn.cmake:# CUDA
share/cmake/Modules/FindSeqAn.cmake:list(FIND SEQAN_FIND_DEPENDENCIES "CUDA" _SEQAN_FIND_CUDA)
share/cmake/Modules/FindSeqAn.cmake:mark_as_advanced(_SEQAN_FIND_CUDA)
share/cmake/Modules/FindSeqAn.cmake:set (SEQAN_HAS_CUDA FALSE)
share/cmake/Modules/FindSeqAn.cmake:if (SEQAN_ENABLE_CUDA AND NOT _SEQAN_FIND_CUDA EQUAL -1)
share/cmake/Modules/FindSeqAn.cmake:  find_package(CUDA QUIET)
share/cmake/Modules/FindSeqAn.cmake:  if (CUDA_FOUND)
share/cmake/Modules/FindSeqAn.cmake:    set (SEQAN_HAS_CUDA TRUE)
share/cmake/Modules/FindSeqAn.cmake:endif (SEQAN_ENABLE_CUDA AND NOT _SEQAN_FIND_CUDA EQUAL -1)
share/cmake/Modules/FindSeqAn.cmake:  message("  SEQAN_HAS_CUDA             ${SEQAN_HAS_CUDA}")
include/seqan/platform/platform_icc.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/platform/platform_nvcc.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/platform/platform_nvcc.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/platform/platform_nvcc.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/platform/platform_nvcc.h: * @macro PLATFORM_CUDA
include/seqan/platform/platform_nvcc.h: * @brief Defined if the compiler is nvcc (or CUDA-capable).
include/seqan/platform/platform_nvcc.h: * @signature #define PLATFORM_CUDA
include/seqan/platform/platform_nvcc.h: * This macro is a synonym for __CUDACC__.
include/seqan/platform/platform_nvcc.h: * This macro can be placed in front of CUDA-compatible functions that can be callable both on host and device side.
include/seqan/platform/platform_nvcc.h: * The macro expands to <tt>__host__ __device__</tt> on CUDA-capable compilers and is ignored otherwise.
include/seqan/platform/platform_nvcc.h: *     // I can run on the CPU and on the GPU, yay!
include/seqan/platform/platform_nvcc.h: * The macro expands to <tt>__host__</tt> on CUDA-capable compilers and is ignored otherwise.
include/seqan/platform/platform_nvcc.h: * The macro expands to <tt>__device__</tt> on CUDA-capable compilers and is ignored otherwise.
include/seqan/platform/platform_nvcc.h: * Note that a device function containing CUDA intrinsics will not compile on non CUDA-capable compilers. Therefore, to
include/seqan/platform/platform_nvcc.h: * insure graceful compilation, it is still necessary to wrap CUDA-intrinsic code inside __CUDA_ARCH__ defines.
include/seqan/platform/platform_nvcc.h: * The macro expands to <tt>__global__</tt> on CUDA-capable compilers and is ignored otherwise.
include/seqan/platform/platform_nvcc.h:#ifndef PLATFORM_CUDA_H_
include/seqan/platform/platform_nvcc.h:#define PLATFORM_CUDA_H_
include/seqan/platform/platform_nvcc.h:#ifdef __CUDACC__
include/seqan/platform/platform_nvcc.h:#define PLATFORM_CUDA
include/seqan/platform/platform_nvcc.h:#endif  // #ifdef __CUDACC__
include/seqan/platform/platform_nvcc.h:#endif  // #ifndef PLATFORM_CUDA_H_
include/seqan/platform/platform_gcc.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/platform/platform_windows.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/sequence.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/sequence.h:#ifdef PLATFORM_CUDA
include/seqan/sequence.h:#ifdef PLATFORM_CUDA
include/seqan/sequence.h:#ifdef PLATFORM_CUDA
include/seqan/index/index_fm_lf_table.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_fm_rank_dictionary_base.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_fm_rank_dictionary_base.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/index/index_fm_rank_dictionary_base.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/index/index_fm_rank_dictionary_levels.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_fm_rank_dictionary_levels.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/index/index_fm_rank_dictionary_levels.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/index/index_fm_rank_dictionary_levels.h:// NOTE(esiragusa): This is required on CUDA devices.
include/seqan/index/index_fm_rank_dictionary_levels.h:// TODO(esiragusa): move loadAndCache() in misc_cuda.h
include/seqan/index/index_fm_rank_dictionary_levels.h:#if __CUDA_ARCH__ >= 350
include/seqan/index/index_fm_rank_dictionary_levels.h:// TODO(esiragusa): move loadAndCache() in misc_cuda.h
include/seqan/index/index_fm_rank_dictionary_levels.h:#if __CUDA_ARCH__ >= 350
include/seqan/index/index_fm_rank_dictionary_levels.h://#if __CUDA_ARCH__ >= 350
include/seqan/index/index_fm_rank_dictionary_levels.h://#if __CUDA_ARCH__ >= 350
include/seqan/index/index_fm_rank_dictionary_levels.h://#if __CUDA_ARCH__ >= 350
include/seqan/index/index_fm.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_fm.h:#ifndef PLATFORM_CUDA
include/seqan/index/index_device.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_device.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/index/index_device.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/index/index_fm_sparse_string.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_esa_stree.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/shape_threshold.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/shape_threshold.h:#if defined(PLATFORM_WINDOWS) | defined(PLATFORM_CUDA)
include/seqan/index/find2_index.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/find2_index.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/index/find2_index.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/index/index_fm_rank_dictionary_wt.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_esa_base.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_view.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_view.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/index/index_view.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/index/find2_index_multi.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/find2_index_multi.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/index/find2_index_multi.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/index/find2_index_multi.h:#ifdef PLATFORM_CUDA
include/seqan/index/find2_index_multi.h:#ifdef PLATFORM_CUDA
include/seqan/index/find2_index_multi.h:#ifdef PLATFORM_CUDA
include/seqan/index/find2_index_multi.h:#ifdef PLATFORM_CUDA
include/seqan/index/find2_index_multi.h:#ifdef PLATFORM_CUDA
include/seqan/index/find2_index_multi.h:#ifdef PLATFORM_CUDA
include/seqan/index/find2_index_multi.h:    SEQAN_ASSERT_EQ(cudaGetLastError(), cudaSuccess);
include/seqan/index/find2_index_multi.h:#ifdef PLATFORM_CUDA
include/seqan/index/find2_index_multi.h://    unsigned activeBlocks = cudaMaxActiveBlocks(_findKernel<TTextView, TPatternsView, TSpec, TDelegateView>, ctaSize, 0);
include/seqan/index/find2_index_multi.h:    SEQAN_ASSERT_EQ(cudaGetLastError(), cudaSuccess);
include/seqan/index/find2_vstree_factory.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/find2_vstree_factory.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/index/find2_vstree_factory.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/index/index_base.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_shims.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_shims.h:#ifdef PLATFORM_CUDA
include/seqan/index/index_fm_right_array_binary_tree.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_fm_stree.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_fm_stree.h:    // NOTE(esiragusa): isLeaf() early exit is slower on CUDA.
include/seqan/index/index_fm_stree.h:#ifndef __CUDA_ARCH__
include/seqan/index/index_fm_stree.h:        // NOTE(esiragusa): isLeaf() early exit is slower on CUDA.
include/seqan/index/index_fm_stree.h:#ifdef __CUDA_ARCH__
include/seqan/index/index_fm_rank_dictionary_naive.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_fm_rank_dictionary_naive.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/index/index_fm_rank_dictionary_naive.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/index/index_fm_compressed_sa.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_fm_device.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_fm_device.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/index/index_fm_device.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/index/index_fm_device.h:// Typedef CudaFMIndexConfig
include/seqan/index/index_fm_device.h:struct CudaFMIndexConfig
include/seqan/index/index_fm_device.h:typedef FMIndex<void, CudaFMIndexConfig>        CudaFMIndexSpec;
include/seqan/index/index_fm_device.h:typedef Index<DnaString, CudaFMIndexSpec>       DnaStringFMIndex;
include/seqan/index/index_fm_device.h:typedef Index<DnaStringSet, CudaFMIndexSpec>    DnaStringSetFMIndex;
include/seqan/index/index_fm_device.h:struct Size<LF<DnaStringSet, void, CudaFMIndexConfig> >
include/seqan/index/index_fm_right_array_binary_tree_iterator.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/find2_functors.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/find2_functors.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/index/find2_functors.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/index/find2_functors.h:#ifdef PLATFORM_CUDA
include/seqan/index/find2_functors.h:#ifdef PLATFORM_CUDA
include/seqan/index/find2_base.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/find2_base.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/index/find2_base.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/index/index_sa_stree.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_sa_stree.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index/index_fm_compressed_sa_iterator.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/file/file_mapping.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/platform.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/platform.h:// NOTE(esiragusa): nvcc header must be included even if __CUDACC__ is not defined.
include/seqan/parallel/parallel_lock.h:#if defined(__SSE2__) && !defined(__CUDACC__)
include/seqan/parallel/parallel_lock.h:#if defined( __CUDACC__)
include/seqan/parallel/parallel_lock.h:    // don't wait on the GPU
include/seqan/parallel/parallel_macros.h:#if defined(__CUDA_ARCH__)
include/seqan/index.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/index.h:// NOTE(esiragusa): CUDA FM-index is broken.
include/seqan/index.h://#ifdef PLATFORM_CUDA
include/seqan/misc/cuda.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/misc/cuda.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/misc/cuda.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/misc/cuda.h:// Author: Jacopo Pantaleoni <jpantaleoni@nvidia.com>
include/seqan/misc/cuda.h:#ifndef SEQAN_MISC_CUDA_MISC_H_
include/seqan/misc/cuda.h:#define SEQAN_MISC_CUDA_MISC_H_
include/seqan/misc/cuda.h://#include <cuda_runtime.h>
include/seqan/misc/cuda.h://#include <thrust/detail/backend/cuda/arch.h>
include/seqan/misc/cuda.h://#include <thrust/system/cuda/detail/arch.h>
include/seqan/misc/cuda.h:// Function cudaPrintFreeMemory()
include/seqan/misc/cuda.h:inline void cudaPrintFreeMemory()
include/seqan/misc/cuda.h:    cudaMemGetInfo(&free, &total);
include/seqan/misc/cuda.h:// Function cudaOccupancy()
include/seqan/misc/cuda.h:float cudaOccupancy()
include/seqan/misc/cuda.h:// Function cudaSmemAllocationUnit()
include/seqan/misc/cuda.h:inline size_t cudaSmemAllocationUnit(cudaDeviceProp const & /* properties */)
include/seqan/misc/cuda.h:// Function cudaRegAllocationUnit()
include/seqan/misc/cuda.h:inline size_t cudaRegAllocationUnit(cudaDeviceProp const & properties)
include/seqan/misc/cuda.h:// Function cudaWarpAllocationMultiple()
include/seqan/misc/cuda.h:inline size_t cudaWarpAllocationMultiple(cudaDeviceProp const & /* properties */)
include/seqan/misc/cuda.h:// Function cudaMaxBlocksPerMultiprocessor()
include/seqan/misc/cuda.h:inline size_t cudaMaxBlocksPerMultiprocessor(cudaDeviceProp const & properties)
include/seqan/misc/cuda.h:// Function cudaKernelGetAttributes()
include/seqan/misc/cuda.h:inline cudaFuncAttributes cudaKernelGetAttributes(TKernel kernel)
include/seqan/misc/cuda.h:    cudaFuncAttributes attributes;
include/seqan/misc/cuda.h:    cudaFuncGetAttributes(&attributes, kernelPointer);
include/seqan/misc/cuda.h:// Function cudaRegistersUsed()
include/seqan/misc/cuda.h:inline size_t cudaRegistersUsed(TKernel kernel)
include/seqan/misc/cuda.h:    return cudaKernelGetAttributes(kernel).numRegs;
include/seqan/misc/cuda.h:// Function cudaMaxActiveBlocks()
include/seqan/misc/cuda.h:inline size_t cudaMaxActiveBlocks(TKernel kernel, TCTASize ctaSize, TSmemSize dynamicSmemBytes)
include/seqan/misc/cuda.h:    cudaGetDevice(&device);
include/seqan/misc/cuda.h:    cudaDeviceProp properties;
include/seqan/misc/cuda.h:    cudaGetDeviceProperties(&properties, device);
include/seqan/misc/cuda.h:    return properties.multiProcessorCount * cudaMaxActiveBlocksPerSM(kernel, ctaSize, dynamicSmemBytes, properties);
include/seqan/misc/cuda.h:// Function cudaMaxActiveBlocksPerSM()
include/seqan/misc/cuda.h:inline size_t cudaMaxActiveBlocksPerSM(TKernel kernel, TCTASize ctaSize, TSmemSize dynamicSmemBytes,
include/seqan/misc/cuda.h:                                       cudaDeviceProp & properties)
include/seqan/misc/cuda.h:    cudaFuncAttributes attributes = cudaKernelGetAttributes(kernel);
include/seqan/misc/cuda.h:    // This is equivalent to the calculation done in the CUDA Occupancy Calculator spreadsheet.
include/seqan/misc/cuda.h:    size_t regAllocationUnit      = cudaRegAllocationUnit(properties);
include/seqan/misc/cuda.h:    size_t warpAllocationMultiple = cudaWarpAllocationMultiple(properties);
include/seqan/misc/cuda.h:    size_t smemAllocationUnit     = cudaSmemAllocationUnit(properties);
include/seqan/misc/cuda.h:    size_t maxBlocksPerSM         = cudaMaxBlocksPerMultiprocessor(properties);
include/seqan/misc/cuda.h:// Function checkCudaError()
include/seqan/misc/cuda.h://inline void checkCudaError(const char *message)
include/seqan/misc/cuda.h://    cudaError_t error = cudaGetLastError();
include/seqan/misc/cuda.h://    if(error!=cudaSuccess) {
include/seqan/misc/cuda.h://        fprintf(stderr,"%s: %s\n", message, cudaGetErrorString(error) );
include/seqan/misc/cuda.h:    return thrust::detail::backend::cuda::arch::max_blocksize_with_highest_occupancy(kernel, dynamic_smem_bytes_per_thread);
include/seqan/misc/cuda.h:    return thrust::system::cuda::detail::arch::max_blocksize_with_highest_occupancy(kernel, dynamic_smem_bytes_per_thread);
include/seqan/misc/cuda.h:    cudaDeviceProp device_properties;
include/seqan/misc/cuda.h:    cudaGetDevice(&device);
include/seqan/misc/cuda.h:    cudaGetDeviceProperties( &device_properties, device );
include/seqan/misc/cuda.h:    #ifdef __CUDA_ARCH__
include/seqan/misc/cuda.h:    if ((N > cuda::Arch::WARP_SIZE) || (is_pow2<N>() == false))
include/seqan/misc/cuda.h:#endif  // SEQAN_MISC_CUDA_MISC_H_
include/seqan/misc/bit_twiddling.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/misc/bit_twiddling.h:#if defined(__CUDA_ARCH__)
include/seqan/misc/bit_twiddling.h:// CUDA implementations.
include/seqan/misc/bit_twiddling.h:#elif defined(_MSC_VER)   // #if !defined(__CUDA_ARCH__) && defined(_MSC_VER)
include/seqan/misc/bit_twiddling.h:#elif !defined(_MSC_VER)  // #if !defined(__CUDA_ARCH__) && !defined(_MSC_VER)
include/seqan/misc/bit_twiddling.h:#endif // #if !defined(__CUDA_ARCH__) && !defined(_MSC_VER)
include/seqan/sequence/string_set_device.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/sequence/string_set_device.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/sequence/string_set_device.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/sequence/string_set_concat_direct.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/sequence/string_set_concat_direct.h:#ifdef PLATFORM_CUDA
include/seqan/sequence/string_set_concat_direct.h:#endif // PLATFORM_CUDA
include/seqan/sequence/string_set_view.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/sequence/string_set_view.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/sequence/string_set_view.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/sequence/adapt_thrust_vector.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/sequence/adapt_thrust_vector.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/sequence/adapt_thrust_vector.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/sequence/string_block.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/sequence/sequence_forwards.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/sequence/sequence_forwards.h:#ifdef PLATFORM_CUDA
include/seqan/sequence/container_view.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/sequence/container_view.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/sequence/container_view.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/sequence/container_view.h:#ifdef PLATFORM_CUDA
include/seqan/sequence/string_packed.h:#ifdef PLATFORM_CUDA
include/seqan/basic/tuple_base.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/basic/alphabet_residue.h:#ifdef __CUDA_ARCH__
include/seqan/basic/alphabet_residue.h:#ifdef __CUDA_ARCH__
include/seqan/basic/iterator_range.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/basic/iterator_range.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/basic/iterator_range.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/basic/debug_test_system.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/basic/debug_test_system.h:#if SEQAN_ENABLE_DEBUG && !defined(__CUDA_ARCH__)
include/seqan/basic/debug_test_system.h:#elif SEQAN_ENABLE_DEBUG && defined(__CUDA_ARCH__)
include/seqan/basic/debug_test_system.h:#endif  // #if defined(SEQAN_ENABLE_DEBUG) && !defined(__CUDA_ARCH__)
include/seqan/basic/metaprogramming_math.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/basic/basic_device.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/basic/basic_device.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/basic/basic_device.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
include/seqan/basic/basic_type.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/basic/alphabet_residue_funcs.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/basic/alphabet_math.h:#ifdef PLATFORM_CUDA
include/seqan/basic/basic_view.h:// Copyright (c) 2013 NVIDIA Corporation
include/seqan/basic/basic_view.h://     * Neither the name of NVIDIA Corporation nor the names of
include/seqan/basic/basic_view.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE

```
