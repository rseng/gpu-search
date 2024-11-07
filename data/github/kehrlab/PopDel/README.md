# https://github.com/kehrlab/PopDel

```console
seqan/platform/platform_icc.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/platform/platform_nvcc.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/platform/platform_nvcc.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/platform/platform_nvcc.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/platform/platform_nvcc.h: * @macro PLATFORM_CUDA
seqan/platform/platform_nvcc.h: * @brief Defined if the compiler is nvcc (or CUDA-capable).
seqan/platform/platform_nvcc.h: * @signature #define PLATFORM_CUDA
seqan/platform/platform_nvcc.h: * This macro is a synonym for __CUDACC__.
seqan/platform/platform_nvcc.h: * This macro can be placed in front of CUDA-compatible functions that can be callable both on host and device side.
seqan/platform/platform_nvcc.h: * The macro expands to <tt>__host__ __device__</tt> on CUDA-capable compilers and is ignored otherwise.
seqan/platform/platform_nvcc.h: *     // I can run on the CPU and on the GPU, yay!
seqan/platform/platform_nvcc.h: * The macro expands to <tt>__host__</tt> on CUDA-capable compilers and is ignored otherwise.
seqan/platform/platform_nvcc.h: * The macro expands to <tt>__device__</tt> on CUDA-capable compilers and is ignored otherwise.
seqan/platform/platform_nvcc.h: * Note that a device function containing CUDA intrinsics will not compile on non CUDA-capable compilers. Therefore, to
seqan/platform/platform_nvcc.h: * insure graceful compilation, it is still necessary to wrap CUDA-intrinsic code inside __CUDA_ARCH__ defines.
seqan/platform/platform_nvcc.h: * The macro expands to <tt>__global__</tt> on CUDA-capable compilers and is ignored otherwise.
seqan/platform/platform_nvcc.h:#ifndef PLATFORM_CUDA_H_
seqan/platform/platform_nvcc.h:#define PLATFORM_CUDA_H_
seqan/platform/platform_nvcc.h:#ifdef __CUDACC__
seqan/platform/platform_nvcc.h:#define PLATFORM_CUDA
seqan/platform/platform_nvcc.h:#endif  // #ifdef __CUDACC__
seqan/platform/platform_nvcc.h:#endif  // #ifndef PLATFORM_CUDA_H_
seqan/platform/platform_mingw.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/platform/platform_gcc.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/platform/platform_pgi.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/platform/platform_windows.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/platform/platform_solaris.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/sequence.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/sequence.h:#ifdef PLATFORM_CUDA
seqan/sequence.h:#ifdef PLATFORM_CUDA
seqan/sequence.h:#ifdef PLATFORM_CUDA
seqan/index/index_fm_lf_table.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_fm_rank_dictionary_base.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_fm_rank_dictionary_base.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/index/index_fm_rank_dictionary_base.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/index/index_fm_rank_dictionary_levels.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_fm_rank_dictionary_levels.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/index/index_fm_rank_dictionary_levels.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/index/index_fm_rank_dictionary_levels.h:// NOTE(esiragusa): This is required on CUDA devices.
seqan/index/index_fm_rank_dictionary_levels.h:// TODO(esiragusa): move loadAndCache() in misc_cuda.h
seqan/index/index_fm_rank_dictionary_levels.h:#if __CUDA_ARCH__ >= 350
seqan/index/index_fm_rank_dictionary_levels.h:// TODO(esiragusa): move loadAndCache() in misc_cuda.h
seqan/index/index_fm_rank_dictionary_levels.h:#if __CUDA_ARCH__ >= 350
seqan/index/index_fm_rank_dictionary_levels.h://#if __CUDA_ARCH__ >= 350
seqan/index/index_fm_rank_dictionary_levels.h://#if __CUDA_ARCH__ >= 350
seqan/index/index_fm_rank_dictionary_levels.h://#if __CUDA_ARCH__ >= 350
seqan/index/index_fm.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_fm.h:#ifndef PLATFORM_CUDA
seqan/index/index_device.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_device.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/index/index_device.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/index/index_fm_sparse_string.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_esa_stree.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/shape_threshold.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/shape_threshold.h:#if defined(PLATFORM_WINDOWS) | defined(PLATFORM_CUDA)
seqan/index/find2_index.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/find2_index.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/index/find2_index.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/index/index_fm_rank_dictionary_wt.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_esa_base.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_view.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_view.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/index/index_view.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/index/find2_index_multi.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/find2_index_multi.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/index/find2_index_multi.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/index/find2_index_multi.h:#ifdef PLATFORM_CUDA
seqan/index/find2_index_multi.h:#ifdef PLATFORM_CUDA
seqan/index/find2_index_multi.h:#ifdef PLATFORM_CUDA
seqan/index/find2_index_multi.h:#ifdef PLATFORM_CUDA
seqan/index/find2_index_multi.h:#ifdef PLATFORM_CUDA
seqan/index/find2_index_multi.h:#ifdef PLATFORM_CUDA
seqan/index/find2_index_multi.h:    SEQAN_ASSERT_EQ(cudaGetLastError(), cudaSuccess);
seqan/index/find2_index_multi.h:#ifdef PLATFORM_CUDA
seqan/index/find2_index_multi.h://    unsigned activeBlocks = cudaMaxActiveBlocks(_findKernel<TTextView, TPatternsView, TSpec, TDelegateView>, ctaSize, 0);
seqan/index/find2_index_multi.h:    SEQAN_ASSERT_EQ(cudaGetLastError(), cudaSuccess);
seqan/index/find2_vstree_factory.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/find2_vstree_factory.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/index/find2_vstree_factory.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/index/index_base.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_shims.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_shims.h:#ifdef PLATFORM_CUDA
seqan/index/index_fm_right_array_binary_tree.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_fm_stree.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_fm_stree.h:    // NOTE(esiragusa): isLeaf() early exit is slower on CUDA.
seqan/index/index_fm_stree.h:#ifndef __CUDA_ARCH__
seqan/index/index_fm_stree.h:        // NOTE(esiragusa): isLeaf() early exit is slower on CUDA.
seqan/index/index_fm_stree.h:#ifdef __CUDA_ARCH__
seqan/index/index_fm_rank_dictionary_naive.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_fm_rank_dictionary_naive.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/index/index_fm_rank_dictionary_naive.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/index/index_fm_compressed_sa.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_fm_device.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_fm_device.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/index/index_fm_device.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/index/index_fm_device.h:// Typedef CudaFMIndexConfig
seqan/index/index_fm_device.h:struct CudaFMIndexConfig
seqan/index/index_fm_device.h:typedef FMIndex<void, CudaFMIndexConfig>        CudaFMIndexSpec;
seqan/index/index_fm_device.h:typedef Index<DnaString, CudaFMIndexSpec>       DnaStringFMIndex;
seqan/index/index_fm_device.h:typedef Index<DnaStringSet, CudaFMIndexSpec>    DnaStringSetFMIndex;
seqan/index/index_fm_device.h:struct Size<LF<DnaStringSet, void, CudaFMIndexConfig> >
seqan/index/index_fm_right_array_binary_tree_iterator.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/find2_functors.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/find2_functors.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/index/find2_functors.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/index/find2_functors.h:#ifdef PLATFORM_CUDA
seqan/index/find2_functors.h:#ifdef PLATFORM_CUDA
seqan/index/find2_base.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/find2_base.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/index/find2_base.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/index/index_sa_stree.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_sa_stree.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index/index_fm_compressed_sa_iterator.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/file/file_mapping.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/platform.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/platform.h:// NOTE(esiragusa): nvcc header must be included even if __CUDACC__ is not defined.
seqan/parallel/parallel_lock.h:#if defined(__SSE2__) && !defined(__CUDACC__)
seqan/parallel/parallel_lock.h:#if defined( __CUDACC__)
seqan/parallel/parallel_lock.h:    // don't wait on the GPU
seqan/parallel/parallel_macros.h:#if defined(__CUDA_ARCH__)
seqan/index.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/index.h:// NOTE(esiragusa): CUDA FM-index is broken.
seqan/index.h://#ifdef PLATFORM_CUDA
seqan/misc/cuda.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/misc/cuda.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/misc/cuda.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/misc/cuda.h:// Author: Jacopo Pantaleoni <jpantaleoni@nvidia.com>
seqan/misc/cuda.h:#ifndef SEQAN_MISC_CUDA_MISC_H_
seqan/misc/cuda.h:#define SEQAN_MISC_CUDA_MISC_H_
seqan/misc/cuda.h://#include <cuda_runtime.h>
seqan/misc/cuda.h://#include <thrust/detail/backend/cuda/arch.h>
seqan/misc/cuda.h://#include <thrust/system/cuda/detail/arch.h>
seqan/misc/cuda.h:// Function cudaPrintFreeMemory()
seqan/misc/cuda.h:inline void cudaPrintFreeMemory()
seqan/misc/cuda.h:    cudaMemGetInfo(&free, &total);
seqan/misc/cuda.h:// Function cudaOccupancy()
seqan/misc/cuda.h:float cudaOccupancy()
seqan/misc/cuda.h:// Function cudaSmemAllocationUnit()
seqan/misc/cuda.h:inline size_t cudaSmemAllocationUnit(cudaDeviceProp const & /* properties */)
seqan/misc/cuda.h:// Function cudaRegAllocationUnit()
seqan/misc/cuda.h:inline size_t cudaRegAllocationUnit(cudaDeviceProp const & properties)
seqan/misc/cuda.h:// Function cudaWarpAllocationMultiple()
seqan/misc/cuda.h:inline size_t cudaWarpAllocationMultiple(cudaDeviceProp const & /* properties */)
seqan/misc/cuda.h:// Function cudaMaxBlocksPerMultiprocessor()
seqan/misc/cuda.h:inline size_t cudaMaxBlocksPerMultiprocessor(cudaDeviceProp const & properties)
seqan/misc/cuda.h:// Function cudaKernelGetAttributes()
seqan/misc/cuda.h:inline cudaFuncAttributes cudaKernelGetAttributes(TKernel kernel)
seqan/misc/cuda.h:    cudaFuncAttributes attributes;
seqan/misc/cuda.h:    cudaFuncGetAttributes(&attributes, kernelPointer);
seqan/misc/cuda.h:// Function cudaRegistersUsed()
seqan/misc/cuda.h:inline size_t cudaRegistersUsed(TKernel kernel)
seqan/misc/cuda.h:    return cudaKernelGetAttributes(kernel).numRegs;
seqan/misc/cuda.h:// Function cudaMaxActiveBlocks()
seqan/misc/cuda.h:inline size_t cudaMaxActiveBlocks(TKernel kernel, TCTASize ctaSize, TSmemSize dynamicSmemBytes)
seqan/misc/cuda.h:    cudaGetDevice(&device);
seqan/misc/cuda.h:    cudaDeviceProp properties;
seqan/misc/cuda.h:    cudaGetDeviceProperties(&properties, device);
seqan/misc/cuda.h:    return properties.multiProcessorCount * cudaMaxActiveBlocksPerSM(kernel, ctaSize, dynamicSmemBytes, properties);
seqan/misc/cuda.h:// Function cudaMaxActiveBlocksPerSM()
seqan/misc/cuda.h:inline size_t cudaMaxActiveBlocksPerSM(TKernel kernel, TCTASize ctaSize, TSmemSize dynamicSmemBytes,
seqan/misc/cuda.h:                                       cudaDeviceProp & properties)
seqan/misc/cuda.h:    cudaFuncAttributes attributes = cudaKernelGetAttributes(kernel);
seqan/misc/cuda.h:    // This is equivalent to the calculation done in the CUDA Occupancy Calculator spreadsheet.
seqan/misc/cuda.h:    size_t regAllocationUnit      = cudaRegAllocationUnit(properties);
seqan/misc/cuda.h:    size_t warpAllocationMultiple = cudaWarpAllocationMultiple(properties);
seqan/misc/cuda.h:    size_t smemAllocationUnit     = cudaSmemAllocationUnit(properties);
seqan/misc/cuda.h:    size_t maxBlocksPerSM         = cudaMaxBlocksPerMultiprocessor(properties);
seqan/misc/cuda.h:// Function checkCudaError()
seqan/misc/cuda.h://inline void checkCudaError(const char *message)
seqan/misc/cuda.h://    cudaError_t error = cudaGetLastError();
seqan/misc/cuda.h://    if(error!=cudaSuccess) {
seqan/misc/cuda.h://        fprintf(stderr,"%s: %s\n", message, cudaGetErrorString(error) );
seqan/misc/cuda.h:    return thrust::detail::backend::cuda::arch::max_blocksize_with_highest_occupancy(kernel, dynamic_smem_bytes_per_thread);
seqan/misc/cuda.h:    return thrust::system::cuda::detail::arch::max_blocksize_with_highest_occupancy(kernel, dynamic_smem_bytes_per_thread);
seqan/misc/cuda.h:    cudaDeviceProp device_properties;
seqan/misc/cuda.h:    cudaGetDevice(&device);
seqan/misc/cuda.h:    cudaGetDeviceProperties( &device_properties, device );
seqan/misc/cuda.h:    #ifdef __CUDA_ARCH__
seqan/misc/cuda.h:    if ((N > cuda::Arch::WARP_SIZE) || (is_pow2<N>() == false))
seqan/misc/cuda.h:#endif  // SEQAN_MISC_CUDA_MISC_H_
seqan/misc/bit_twiddling.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/misc/bit_twiddling.h:// CUDA implementations.
seqan/misc/bit_twiddling.h:#if defined(__CUDA_ARCH__)
seqan/misc/bit_twiddling.h:#else   // #if defined(__CUDA_ARCH__)
seqan/misc/bit_twiddling.h:#endif    // #if !defined(__CUDA_ARCH__)
seqan/sequence/string_set_device.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/sequence/string_set_device.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/sequence/string_set_device.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/sequence/string_set_concat_direct.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/sequence/string_set_concat_direct.h:#ifdef PLATFORM_CUDA
seqan/sequence/string_set_concat_direct.h:#endif // PLATFORM_CUDA
seqan/sequence/string_set_view.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/sequence/string_set_view.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/sequence/string_set_view.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/sequence/adapt_thrust_vector.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/sequence/adapt_thrust_vector.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/sequence/adapt_thrust_vector.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/sequence/string_block.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/sequence/sequence_forwards.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/sequence/sequence_forwards.h:#ifdef PLATFORM_CUDA
seqan/sequence/container_view.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/sequence/container_view.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/sequence/container_view.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/sequence/container_view.h:#ifdef PLATFORM_CUDA
seqan/sequence/string_packed.h:#ifdef PLATFORM_CUDA
seqan/basic/tuple_base.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/basic/alphabet_residue.h:#ifdef __CUDA_ARCH__
seqan/basic/alphabet_residue.h:#ifdef __CUDA_ARCH__
seqan/basic/iterator_range.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/basic/iterator_range.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/basic/iterator_range.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/basic/debug_test_system.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/basic/debug_test_system.h:#if SEQAN_ENABLE_DEBUG && !defined(__CUDA_ARCH__)
seqan/basic/debug_test_system.h:#elif SEQAN_ENABLE_DEBUG && defined(__CUDA_ARCH__)
seqan/basic/debug_test_system.h:#endif  // #if defined(SEQAN_ENABLE_DEBUG) && !defined(__CUDA_ARCH__)
seqan/basic/metaprogramming_math.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/basic/basic_device.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/basic/basic_device.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/basic/basic_device.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
seqan/basic/basic_type.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/basic/alphabet_residue_funcs.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/basic/alphabet_math.h:#ifdef PLATFORM_CUDA
seqan/basic/basic_view.h:// Copyright (c) 2013 NVIDIA Corporation
seqan/basic/basic_view.h://     * Neither the name of NVIDIA Corporation nor the names of
seqan/basic/basic_view.h:// ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE

```
