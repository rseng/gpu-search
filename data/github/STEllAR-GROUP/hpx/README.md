# https://github.com/STEllAR-GROUP/hpx

```console
libs/core/threading_base/include/hpx/threading_base/scheduler_base.hpp:        void set_cuda_polling_functions(polling_function_ptr cuda_func,
libs/core/threading_base/include/hpx/threading_base/scheduler_base.hpp:            polling_work_count_function_ptr cuda_work_count_func);
libs/core/threading_base/include/hpx/threading_base/scheduler_base.hpp:        void clear_cuda_polling_function();
libs/core/threading_base/include/hpx/threading_base/scheduler_base.hpp:        std::atomic<polling_function_ptr> polling_function_cuda_;
libs/core/threading_base/include/hpx/threading_base/scheduler_base.hpp:            polling_work_count_function_cuda_;
libs/core/threading_base/src/scheduler_base.cpp:      , polling_function_cuda_(&null_polling_function)
libs/core/threading_base/src/scheduler_base.cpp:      , polling_work_count_function_cuda_(&null_polling_work_count_function)
libs/core/threading_base/src/scheduler_base.cpp:    void scheduler_base::set_cuda_polling_functions(
libs/core/threading_base/src/scheduler_base.cpp:        polling_function_ptr cuda_func,
libs/core/threading_base/src/scheduler_base.cpp:        polling_work_count_function_ptr cuda_work_count_func)
libs/core/threading_base/src/scheduler_base.cpp:        polling_function_cuda_.store(cuda_func, std::memory_order_relaxed);
libs/core/threading_base/src/scheduler_base.cpp:        polling_work_count_function_cuda_.store(
libs/core/threading_base/src/scheduler_base.cpp:            cuda_work_count_func, std::memory_order_relaxed);
libs/core/threading_base/src/scheduler_base.cpp:    void scheduler_base::clear_cuda_polling_function()
libs/core/threading_base/src/scheduler_base.cpp:        polling_function_cuda_.store(
libs/core/threading_base/src/scheduler_base.cpp:        polling_work_count_function_cuda_.store(
libs/core/threading_base/src/scheduler_base.cpp:#if defined(HPX_HAVE_MODULE_ASYNC_CUDA)
libs/core/threading_base/src/scheduler_base.cpp:        if ((*polling_function_cuda_.load(std::memory_order_relaxed))() ==
libs/core/threading_base/src/scheduler_base.cpp:#if defined(HPX_HAVE_MODULE_ASYNC_CUDA)
libs/core/threading_base/src/scheduler_base.cpp:            polling_work_count_function_cuda_.load(std::memory_order_relaxed)();
libs/core/iterator_support/include/hpx/iterator_support/traits/is_iterator.hpp:#if defined(HPX_MSVC) && defined(__CUDACC__)
libs/core/compute_local/tests/unit/CMakeLists.txt:  target_include_directories(${test}_test SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
libs/core/compute_local/tests/regressions/CMakeLists.txt:  target_include_directories(${test}_test SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
libs/core/compute_local/include/hpx/compute_local/vector.hpp:#if !defined(__CUDA_ARCH__)
libs/core/compute_local/include/hpx/compute_local/vector.hpp:#if !defined(__CUDA_ARCH__)
libs/core/compute_local/include/hpx/compute_local/vector.hpp:#if !defined(__CUDA_ARCH__)
libs/core/compute_local/include/hpx/compute_local/vector.hpp:#if !defined(__CUDA_ARCH__)
libs/core/compute_local/include/hpx/compute_local/vector.hpp:#if defined(__NVCC__) || defined(__CUDACC__)
libs/core/compute_local/include/hpx/compute_local/traits/allocator_traits.hpp:#if defined(__CUDA_ARCH__)
libs/core/compute_local/include/hpx/compute_local/serialization/vector.hpp:#if !defined(__CUDA_ARCH__)
libs/core/compute_local/include/hpx/compute_local/serialization/vector.hpp:#if !defined(__CUDA_ARCH__)
libs/core/pack_traversal/include/hpx/pack_traversal/detail/pack_traversal_impl.hpp:            // CUDA needs std::forward here
libs/core/pack_traversal/include/hpx/pack_traversal/detail/pack_traversal_async_impl.hpp:                // CUDA needs std::forward here
libs/core/pack_traversal/include/hpx/pack_traversal/detail/unwrap_impl.hpp:            // CUDA needs std::forward here
libs/core/pack_traversal/include/hpx/pack_traversal/detail/unwrap_impl.hpp:            // CUDA needs std::forward here
libs/core/pack_traversal/include/hpx/pack_traversal/detail/unwrap_impl.hpp:            // CUDA needs std::forward here
libs/core/pack_traversal/include/hpx/pack_traversal/detail/unwrap_impl.hpp:            // CUDA needs std::forward here
libs/core/include_local/include/hpx/compute.hpp.in:#if defined(HPX_HAVE_CUDA) || defined(HPX_HAVE_HIP)
libs/core/include_local/include/hpx/compute.hpp.in:#include <hpx/modules/async_cuda.hpp>
libs/core/config/include/hpx/config/attributes.hpp:#  if defined(__NVCC__) || defined(__CUDACC__) || defined(__HIPCC__)
libs/core/config/include/hpx/config/compiler_specific.hpp:// Detecting CUDA compilation mode
libs/core/config/include/hpx/config/compiler_specific.hpp:// Both clang and nvcc define __CUDACC__ during CUDA compilation.
libs/core/config/include/hpx/config/compiler_specific.hpp:// Source: Clang CUDA documentation
libs/core/config/include/hpx/config/compiler_specific.hpp:#if defined(__NVCC__) && defined(__CUDACC__)
libs/core/config/include/hpx/config/compiler_specific.hpp:#  define HPX_CUDA_VERSION (__CUDACC_VER_MAJOR__*100 + __CUDACC_VER_MINOR__)
libs/core/config/include/hpx/config/compiler_specific.hpp:#  if defined(__CUDA_ARCH__)
libs/core/config/include/hpx/config/compiler_specific.hpp:     // nvcc compiling CUDA code, device mode.
libs/core/config/include/hpx/config/compiler_specific.hpp:// Detecting Clang CUDA
libs/core/config/include/hpx/config/compiler_specific.hpp:#elif defined(__clang__) && defined(__CUDACC__)
libs/core/config/include/hpx/config/compiler_specific.hpp:#  if defined(__CUDA_ARCH__)
libs/core/config/include/hpx/config/compiler_specific.hpp:     // clang compiling CUDA code, device mode.
libs/core/config/include/hpx/config/compiler_specific.hpp:     // hipclang compiling CUDA/HIP code, device mode.
libs/core/config/include/hpx/config/export_definitions.hpp:#elif defined(__NVCC__) || defined(__CUDACC__)
libs/core/config/include/hpx/config/forceinline.hpp:#   if defined(__NVCC__) || defined(__CUDACC__)
libs/core/datastructures/include/hpx/datastructures/detail/small_vector.hpp:#if defined(HPX_MSVC) && defined(__CUDACC__)
libs/core/datastructures/include/hpx/datastructures/tuple.hpp:#if !defined(__NVCC__) && !defined(__CUDACC__)
libs/core/datastructures/include/hpx/datastructures/tuple.hpp:// Hipcc compiler bug (with rocm-3.7.0 and rocm-3.8.0) return a const-reference
libs/core/datastructures/include/hpx/datastructures/tuple.hpp:// https://github.com/ROCm-Developer-Tools/HIP/issues/2173
libs/core/properties/include/hpx/properties/property.hpp:#if !defined(HPX_CUDA_VERSION)
libs/core/async_sycl/docs/index.rst:The creation of the HPX futures using SYCL events is based on the same event polling mechanism that the CUDA HPX
libs/core/async_sycl/tests/unit/sycl_vector_add_get_future.cpp:// Should work with GPUs with >= 2GB memory
libs/core/async_sycl/tests/unit/sycl_vector_add_get_future.cpp:// advisor --collect=survey --profile-gpu -- ./bin/sycl_vector_add_test
libs/core/async_sycl/tests/unit/sycl_vector_add_get_future.cpp:// --profile-gpu -- ./bin/sycl_vector_add_test
libs/core/async_sycl/include/hpx/async_sycl/detail/sycl_event_callback.hpp:    /// Type of the event_callback function used. Unlike the CUDA counterpart we are
libs/core/async_sycl/include/hpx/async_sycl/sycl_polling_helper.hpp://  libs/core/async_cuda/include/hpx/async_cuda/cuda_polling_helper.hpp
libs/core/async_sycl/include/hpx/async_sycl/sycl_future.hpp:// This file is very similar to its CUDA counterpart (cuda_future.hpp) just
libs/core/async_sycl/src/sycl_event_callback.cpp:// This file is very similar to its CUDA counterpart (cuda_event_callback.cpp)
libs/core/async_sycl/src/sycl_event_callback.cpp:     * Unlike the CUDA counterpart, no event pool is used here since we have to
libs/core/resource_partitioner/examples/oversubscribing_resource_partitioner.cpp:        GPU = 2,
libs/core/type_support/include/hpx/type_support/unused.hpp:#if defined(__CUDA_ARCH__)
libs/core/execution_base/include/hpx/execution_base/completion_signatures.hpp:        // https://github.com/NVIDIA/stdexec/pull/733#issue-1537242117
libs/core/futures/include/hpx/futures/future.hpp:            // CUDA 11. Without this nvcc fails to compile some code with
libs/core/futures/include/hpx/futures/future.hpp:#if defined(HPX_CUDA_VERSION) && (HPX_CUDA_VERSION < 1104)
libs/core/futures/include/hpx/futures/future.hpp:#if defined(HPX_CUDA_VERSION) && (HPX_CUDA_VERSION < 1104)
libs/core/algorithms/include/hpx/parallel/algorithms/detail/dispatch.hpp:#if !defined(__CUDA_ARCH__)
libs/core/algorithms/include/hpx/parallel/algorithms/detail/dispatch.hpp:#if !defined(__CUDA_ARCH__)
libs/core/algorithms/include/hpx/parallel/algorithms/count.hpp:#if !defined(__NVCC__) && !defined(__CUDACC__)
libs/core/algorithms/include/hpx/parallel/algorithms/for_each.hpp:#if !defined(__NVCC__) && !defined(__CUDACC__)
libs/core/algorithms/include/hpx/parallel/algorithms/for_each.hpp:#if !defined(__NVCC__) && !defined(__CUDACC__)
libs/core/algorithms/include/hpx/parallel/algorithms/lexicographical_compare.hpp:                            // gcc10/cuda11 complains about using HPX_INVOKE here
libs/core/algorithms/include/hpx/parallel/algorithms/for_loop.hpp:                    // everything to a GPU device
libs/core/algorithms/include/hpx/parallel/algorithms/for_loop.hpp:                    // everything to a GPU device
libs/core/algorithms/include/hpx/parallel/algorithms/transform.hpp:#if !defined(__NVCC__) && !defined(__CUDACC__)
libs/core/algorithms/include/hpx/parallel/algorithms/transform.hpp:#if !defined(__NVCC__) && !defined(__CUDACC__)
libs/core/algorithms/include/hpx/parallel/algorithms/transform.hpp:#if !defined(__NVCC__) && !defined(__CUDACC__)
libs/core/algorithms/include/hpx/parallel/algorithms/transform.hpp:#if !defined(__NVCC__) && !defined(__CUDACC__)
libs/core/execution/include/hpx/execution/traits/vector_pack_load_store.hpp:#if !defined(__CUDACC__)
libs/core/execution/include/hpx/execution/traits/vector_pack_find.hpp:#if !defined(__CUDACC__)
libs/core/execution/include/hpx/execution/traits/vector_pack_all_any_none.hpp:#if !defined(__CUDACC__)
libs/core/execution/include/hpx/execution/traits/vector_pack_get_set.hpp:#if !defined(__CUDACC__)
libs/core/execution/include/hpx/execution/traits/vector_pack_reduce.hpp:#if !defined(__CUDACC__)
libs/core/execution/include/hpx/execution/traits/vector_pack_type.hpp:#if !defined(__CUDACC__)
libs/core/execution/include/hpx/execution/traits/vector_pack_alignment_size.hpp:#if !defined(__CUDACC__)
libs/core/execution/include/hpx/execution/traits/vector_pack_count_bits.hpp:#if !defined(__CUDACC__)
libs/core/execution/include/hpx/execution/traits/vector_pack_conditionals.hpp:#if !defined(__CUDACC__)
libs/core/execution/include/hpx/execution/algorithms/detail/predicates.hpp:#if (defined(HPX_HAVE_CUDA) && defined(__CUDACC__)) || defined(HPX_HAVE_HIP)
libs/core/execution/include/hpx/execution/algorithms/detail/predicates.hpp:#if (defined(HPX_HAVE_CUDA) && defined(__CUDACC__)) || defined(HPX_HAVE_HIP)
libs/core/execution/include/hpx/execution/algorithms/detail/predicates.hpp:#if (defined(HPX_HAVE_CUDA) && defined(__CUDACC__)) || defined(HPX_HAVE_HIP)
libs/core/execution/include/hpx/execution/algorithms/detail/predicates.hpp:#if (defined(HPX_HAVE_CUDA) && defined(__CUDACC__)) || defined(HPX_HAVE_HIP)
libs/core/execution/include/hpx/execution/algorithms/detail/predicates.hpp:#if (defined(HPX_HAVE_CUDA) && defined(__CUDACC__)) || defined(HPX_HAVE_HIP)
libs/core/execution/include/hpx/execution/algorithms/when_all.hpp:#if !defined(HPX_CUDA_VERSION)
libs/core/execution/include/hpx/execution/algorithms/when_all.hpp:#if defined(HPX_CUDA_VERSION)
libs/core/execution/include/hpx/execution/algorithms/when_all.hpp:#if !defined(HPX_CUDA_VERSION)
libs/core/execution/include/hpx/execution/algorithms/when_all.hpp:#if defined(HPX_CUDA_VERSION)
libs/core/execution/include/hpx/execution/executors/polymorphic_executor.hpp:        // NOTE: nvcc (at least CUDA 9.2 and 10.1) fails with an internal
libs/core/execution/include/hpx/execution/executors/polymorphic_executor.hpp:#if !defined(HPX_HAVE_GPU_SUPPORT)
libs/core/config_registry/include/hpx/modules/config_registry.hpp:#elif defined(__NVCC__) || defined(__CUDACC__)
libs/core/CMakeLists.txt:    async_cuda
libs/core/executors/tests/regressions/CMakeLists.txt:  if(HPX_WITH_CUDA OR HPX_WITH_HIP)
libs/core/executors/tests/regressions/CMakeLists.txt:    list(APPEND compile_tests service_executor_cuda)
libs/core/executors/tests/regressions/CMakeLists.txt:    set(service_executor_cuda_CUDA ON)
libs/core/executors/tests/regressions/CMakeLists.txt:    if(${${compile_test}_CUDA})
libs/core/executors/include/hpx/executors/dataflow.hpp:            // uses std::forward for HPX_CUDA_VERSION
libs/core/hardware/include/hpx/hardware/timestamp/linux_generic.hpp:#if defined(HPX_HAVE_CUDA) && defined(HPX_COMPUTE_CODE)
libs/core/hardware/include/hpx/hardware/timestamp/linux_generic.hpp:#include <hpx/hardware/timestamp/cuda.hpp>
libs/core/hardware/include/hpx/hardware/timestamp/linux_generic.hpp:#if defined(HPX_HAVE_CUDA) && defined(HPX_COMPUTE_DEVICE_CODE)
libs/core/hardware/include/hpx/hardware/timestamp/linux_generic.hpp:        return timestamp_cuda();
libs/core/hardware/include/hpx/hardware/timestamp/linux_x86_64.hpp:#if defined(HPX_HAVE_CUDA) && defined(HPX_COMPUTE_CODE)
libs/core/hardware/include/hpx/hardware/timestamp/linux_x86_64.hpp:#include <hpx/hardware/timestamp/cuda.hpp>
libs/core/hardware/include/hpx/hardware/timestamp/linux_x86_64.hpp:#if defined(HPX_HAVE_CUDA) && defined(HPX_COMPUTE_DEVICE_CODE)
libs/core/hardware/include/hpx/hardware/timestamp/linux_x86_64.hpp:        return timestamp_cuda();
libs/core/hardware/include/hpx/hardware/timestamp/msvc.hpp:#if defined(HPX_HAVE_CUDA) && defined(HPX_COMPUTE_CODE)
libs/core/hardware/include/hpx/hardware/timestamp/msvc.hpp:#include <hpx/hardware/timestamp/cuda.hpp>
libs/core/hardware/include/hpx/hardware/timestamp/msvc.hpp:        return timestamp_cuda();
libs/core/hardware/include/hpx/hardware/timestamp/cuda.hpp:    HPX_DEVICE inline std::uint64_t timestamp_cuda()
libs/core/hardware/include/hpx/hardware/timestamp/linux_x86_32.hpp:#if defined(HPX_HAVE_CUDA) && defined(HPX_COMPUTE_CODE)
libs/core/hardware/include/hpx/hardware/timestamp/linux_x86_32.hpp:#include <hpx/hardware/timestamp/cuda.hpp>
libs/core/hardware/include/hpx/hardware/timestamp/linux_x86_32.hpp:#if defined(HPX_HAVE_CUDA) && defined(HPX_COMPUTE_DEVICE_CODE)
libs/core/hardware/include/hpx/hardware/timestamp/linux_x86_32.hpp:        return timestamp_cuda();
libs/core/hardware/include/hpx/hardware/timestamp/bgq.hpp:#if defined(HPX_HAVE_CUDA) && defined(HPX_COMPUTE_CODE)
libs/core/hardware/include/hpx/hardware/timestamp/bgq.hpp:#include <hpx/hardware/timestamp/cuda.hpp>
libs/core/hardware/include/hpx/hardware/timestamp/bgq.hpp:#if defined(HPX_HAVE_CUDA) && defined(HPX_COMPUTE_DEVICE_CODE)
libs/core/hardware/include/hpx/hardware/timestamp/bgq.hpp:        return timestamp_cuda();
libs/core/hardware/CMakeLists.txt:    hpx/hardware/timestamp/cuda.hpp
libs/core/hardware/CMakeLists.txt:    "hpx/hardware/timestamp/cuda.hpp"
libs/core/tag_invoke/include/hpx/functional/detail/tag_fallback_invoke.hpp:    // CUDA versions less than 11.2 have a template instantiation bug that
libs/core/tag_invoke/include/hpx/functional/detail/tag_fallback_invoke.hpp:#if !defined(HPX_CUDA_VERSION) || (HPX_CUDA_VERSION >= 1102)
libs/core/tag_invoke/include/hpx/functional/detail/tag_priority_invoke.hpp:    // CUDA versions less than 11.2 have a template instantiation problem that
libs/core/tag_invoke/include/hpx/functional/detail/tag_priority_invoke.hpp:#if !defined(HPX_CUDA_VERSION) || (HPX_CUDA_VERSION >= 1102)
libs/core/tag_invoke/include/hpx/functional/tag_invoke.hpp:    // CUDA versions less than 11.2 have a template instantiation bug that
libs/core/tag_invoke/include/hpx/functional/tag_invoke.hpp:#if !defined(HPX_CUDA_VERSION) || (HPX_CUDA_VERSION >= 1102)
libs/core/allocator_support/include/hpx/allocator_support/thread_local_caching_allocator.hpp:    !((defined(HPX_HAVE_CUDA) && defined(__CUDACC__)) ||                       \
libs/core/functional/include/hpx/functional/detail/empty_function.hpp:    // NOTE: nvcc (at least CUDA 9.2 and 10.1) fails with an internal compiler
libs/core/functional/include/hpx/functional/detail/empty_function.hpp:#if !defined(HPX_HAVE_CUDA)
libs/core/functional/include/hpx/functional/bind_front.hpp:#if !defined(__NVCC__) && !defined(__CUDACC__)
libs/core/functional/include/hpx/functional/traits/get_function_address.hpp:    (!defined(__NVCC__) || (__CUDACC_VER_MAJOR__ >= 8))
libs/core/functional/include/hpx/functional/traits/get_function_address.hpp:    (!defined(__NVCC__) || (__CUDACC_VER_MAJOR__ >= 8))
libs/core/functional/include/hpx/functional/traits/get_function_address.hpp:    (!defined(__NVCC__) || (__CUDACC_VER_MAJOR__ >= 8))
libs/core/functional/include/hpx/functional/traits/get_function_address.hpp:    (!defined(__NVCC__) || (__CUDACC_VER_MAJOR__ >= 8))
libs/core/functional/include/hpx/functional/bind_back.hpp:#if !defined(__NVCC__) && !defined(__CUDACC__)
libs/core/functional/include/hpx/functional/bind.hpp:#if !defined(__NVCC__) && !defined(__CUDACC__)
libs/core/functional/include/hpx/functional/protect.hpp:#if !defined(__NVCC__) && !defined(__CUDACC__)
libs/core/functional/include/hpx/functional/deferred_call.hpp:#if !defined(__NVCC__) && !defined(__CUDACC__)
libs/core/lcos_local/include/hpx/lcos_local/channel.hpp:#if !(defined(HPX_CUDA_VERSION) && defined(HPX_GCC_VERSION) &&                 \
libs/core/static_reinit/include/hpx/static_reinit/reinitializable_static.hpp:#if !defined(__CUDACC__)
libs/core/static_reinit/include/hpx/static_reinit/reinitializable_static.hpp:#if !defined(__CUDACC__)
libs/core/preprocessor/include/hpx/preprocessor/config.hpp:#if defined _GCCXML_ || defined _CUDACC_ || defined _PATHSCALE_ ||             \
libs/core/preprocessor/include/hpx/preprocessor/config.hpp:        !(defined _EDG_ || defined _GCCXML_ || defined _CUDACC_ ||             \
libs/core/async_cuda/docs/index.rst:.. _modules_async_cuda:
libs/core/async_cuda/docs/index.rst:async_cuda
libs/core/async_cuda/docs/index.rst:a |cuda|_ stream. Typically, a user may launch one or more kernels and then get a
libs/core/async_cuda/docs/index.rst:See the :ref:`API reference <modules_async_cuda_api>` of this module for more
libs/core/async_cuda/tests/unit/saxpy.cu:#include <hpx/async_cuda/cuda_executor.hpp>
libs/core/async_cuda/tests/unit/saxpy.cu:#include <hpx/async_cuda/custom_gpu_api.hpp>
libs/core/async_cuda/tests/unit/saxpy.cu:void launch_saxpy_kernel(hpx::cuda::experimental::cuda_executor& cudaexec,
libs/core/async_cuda/tests/unit/saxpy.cu:    // Invoking hpx::post with cudaLaunchKernel<void> directly result in an
libs/core/async_cuda/tests/unit/saxpy.cu:    auto launch_kernel = cudaLaunchKernel;
libs/core/async_cuda/tests/unit/saxpy.cu:    auto launch_kernel = cudaLaunchKernel<void>;
libs/core/async_cuda/tests/unit/saxpy.cu:    hpx::post(cudaexec, launch_kernel, reinterpret_cast<void*>(&saxpy),
libs/core/async_cuda/tests/unit/cuda_future.cpp:#include <hpx/modules/async_cuda.hpp>
libs/core/async_cuda/tests/unit/cuda_future.cpp:// but the cuda functions go in their own .cu file and are compiled with
libs/core/async_cuda/tests/unit/cuda_future.cpp:extern void cuda_trivial_kernel(T, cudaStream_t stream);
libs/core/async_cuda/tests/unit/cuda_future.cpp:    hpx::cuda::experimental::cuda_executor& cudaexec, unsigned int& blocks,
libs/core/async_cuda/tests/unit/cuda_future.cpp:int test_saxpy(hpx::cuda::experimental::cuda_executor& cudaexec)
libs/core/async_cuda/tests/unit/cuda_future.cpp:    // host arrays (CUDA pinned host memory for asynchronous data transfers)
libs/core/async_cuda/tests/unit/cuda_future.cpp:    hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/unit/cuda_future.cpp:        cudaMallocHost((void**) &h_A, N * sizeof(float)));
libs/core/async_cuda/tests/unit/cuda_future.cpp:    hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/unit/cuda_future.cpp:        cudaMallocHost((void**) &h_B, N * sizeof(float)));
libs/core/async_cuda/tests/unit/cuda_future.cpp:    hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/unit/cuda_future.cpp:        cudaMalloc((void**) &d_A, N * sizeof(float)));
libs/core/async_cuda/tests/unit/cuda_future.cpp:    hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/unit/cuda_future.cpp:        cudaMalloc((void**) &d_B, N * sizeof(float)));
libs/core/async_cuda/tests/unit/cuda_future.cpp:    // copy both arrays from cpu to gpu, putting both copies onto the stream
libs/core/async_cuda/tests/unit/cuda_future.cpp:    hpx::post(cudaexec, cudaMemcpyAsync, d_A, h_A, N * sizeof(float),
libs/core/async_cuda/tests/unit/cuda_future.cpp:        cudaMemcpyHostToDevice);
libs/core/async_cuda/tests/unit/cuda_future.cpp:    hpx::post(cudaexec, cudaMemcpyAsync, d_B, h_B, N * sizeof(float),
libs/core/async_cuda/tests/unit/cuda_future.cpp:        cudaMemcpyHostToDevice);
libs/core/async_cuda/tests/unit/cuda_future.cpp:    launch_saxpy_kernel(cudaexec, blocks, threads, args);
libs/core/async_cuda/tests/unit/cuda_future.cpp:    // finally, perform a copy from the gpu back to the cpu all on the same stream
libs/core/async_cuda/tests/unit/cuda_future.cpp:    auto cuda_future = hpx::async(cudaexec, cudaMemcpyAsync, h_B, d_B,
libs/core/async_cuda/tests/unit/cuda_future.cpp:        N * sizeof(float), cudaMemcpyDeviceToHost);
libs/core/async_cuda/tests/unit/cuda_future.cpp:    cuda_future
libs/core/async_cuda/tests/unit/cuda_future.cpp:                << "saxpy completed on GPU, checking results in continuation"
libs/core/async_cuda/tests/unit/cuda_future.cpp:    // install cuda future polling handler
libs/core/async_cuda/tests/unit/cuda_future.cpp:    hpx::cuda::experimental::enable_user_polling poll("default");
libs/core/async_cuda/tests/unit/cuda_future.cpp:    // create a cuda target using device number 0,1,2...
libs/core/async_cuda/tests/unit/cuda_future.cpp:    hpx::cuda::experimental::target target(device);
libs/core/async_cuda/tests/unit/cuda_future.cpp:    hpx::cuda::experimental::print_local_targets();
libs/core/async_cuda/tests/unit/cuda_future.cpp:    hpx::cuda::experimental::cuda_executor cudaexec(
libs/core/async_cuda/tests/unit/cuda_future.cpp:        device, hpx::cuda::experimental::event_mode{});
libs/core/async_cuda/tests/unit/cuda_future.cpp:    std::cout << "apply : cuda kernel <float>  : " << testf << std::endl;
libs/core/async_cuda/tests/unit/cuda_future.cpp:    hpx::post(cudaexec, cuda_trivial_kernel<float>, testf);
libs/core/async_cuda/tests/unit/cuda_future.cpp:    std::cout << "async : cuda kernel <float>  : " << testf + 1 << std::endl;
libs/core/async_cuda/tests/unit/cuda_future.cpp:    auto f1 = hpx::async(cudaexec, cuda_trivial_kernel<float>, testf + 1);
libs/core/async_cuda/tests/unit/cuda_future.cpp:    std::cout << "apply : cuda kernel <double> : " << testd << std::endl;
libs/core/async_cuda/tests/unit/cuda_future.cpp:    hpx::post(cudaexec, cuda_trivial_kernel<double>, testd);
libs/core/async_cuda/tests/unit/cuda_future.cpp:    std::cout << "async : cuda kernel <double> : " << testd + 1 << std::endl;
libs/core/async_cuda/tests/unit/cuda_future.cpp:    auto f2 = hpx::async(cudaexec, cuda_trivial_kernel<double>, testd + 1);
libs/core/async_cuda/tests/unit/cuda_future.cpp:    // test adding a continuation to a cuda call
libs/core/async_cuda/tests/unit/cuda_future.cpp:    auto f3 = hpx::async(cudaexec, cuda_trivial_kernel<double>, testd2);
libs/core/async_cuda/tests/unit/cuda_future.cpp:    // test using a copy of a cuda executor
libs/core/async_cuda/tests/unit/cuda_future.cpp:    auto exec_copy = cudaexec;
libs/core/async_cuda/tests/unit/cuda_future.cpp:    auto f4 = hpx::async(exec_copy, cuda_trivial_kernel<double>, testd2 + 1);
libs/core/async_cuda/tests/unit/cuda_future.cpp:    HPX_TEST(test_saxpy(cudaexec));
libs/core/async_cuda/tests/unit/cuda_future.cpp:    printf("[HPX Cuda future] - Starting...\n");
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:// For compliance with the NVIDIA EULA:
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:// "This software contains source code provided by NVIDIA Corporation."
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:// This is a conversion of the NVIDIA cublas example matrixMulCUBLAS to use
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:// comparison/checks and makes no difference to the GPU execution.
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:// Note: The hpx::cuda::experimental::allocator makes use of device code and if used
libs/core/async_cuda/tests/unit/cublas_matmul.cpp://     PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:// cudaMalloc/cudaMemcpy etc, so we do not #define HPX_CUBLAS_DEMO_WITH_ALLOCATOR
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:#include <hpx/modules/async_cuda.hpp>
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:void matrixMultiply(hpx::cuda::experimental::cublas_executor& cublas,
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    // create a cublas executor we'll use to futurize cuda events
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    using namespace hpx::cuda::experimental;
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    using cublas_future = typename cuda_executor::future_type;
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:        cudaMalloc((void**) &d_A, size_A * sizeof(T)));
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:        cudaMalloc((void**) &d_B, size_B * sizeof(T)));
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:        cudaMalloc((void**) &d_C, size_C * sizeof(T)));
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    hpx::post(cublas, cudaMemcpyAsync, d_A, h_A.data(), size_A * sizeof(T),
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:        cudaMemcpyHostToDevice);
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    auto copy_future = hpx::async(cublas, cudaMemcpyAsync, d_B, h_B.data(),
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:        size_B * sizeof(T), cudaMemcpyHostToDevice);
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    auto copy_finished = hpx::async(cublas, cudaMemcpyAsync, h_CUBLAS.data(),
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:        d_C, size_C * sizeof(T), cudaMemcpyDeviceToHost);
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:        HPX_TEST_MSG(resCUBLAS, "matrix CPU/GPU comparison error");
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_A));
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_B));
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_C));
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    // install cuda future polling handler
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    hpx::cuda::experimental::enable_user_polling poll("default");
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    // use a larger block size for Fermi and above, query default cuda target properties
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    hpx::cuda::experimental::target target(device);
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    std::cout << "GPU Device " << target.native_handle().get_device() << ": \""
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    hpx::cuda::experimental::cublas_executor cublas(device,
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:        CUBLAS_POINTER_MODE_HOST, hpx::cuda::experimental::event_mode{});
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    hpx::cuda::experimental::cublas_executor cublas2 = cublas;
libs/core/async_cuda/tests/unit/cublas_matmul.cpp:    hpx::cuda::experimental::cublas_executor cublas3(std::move(cublas));
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:#include <hpx/modules/async_cuda.hpp>
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:// This example is similar to the unit/cuda_future.cpp example (hence it also uses
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:// the externally defined cuda_trivial_kernel. See unit/cuda_future.cpp for
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:// This example extends unit/cuda_future.cpp by testing the cuda event polling
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:extern void cuda_trivial_kernel(T, cudaStream_t stream);
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:    hpx::cuda::experimental::cuda_executor& cudaexec, unsigned int& blocks,
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:    // install cuda future polling handler
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:    hpx::cuda::experimental::enable_user_polling poll("default");
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:    hpx::cuda::experimental::print_local_targets();
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:    hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:        cudaGetDeviceCount(&number_devices));
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:        hpx::cuda::experimental::cuda_executor exec(
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:            device_id, hpx::cuda::experimental::event_mode{});
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:        auto fut = hpx::async(exec, cuda_trivial_kernel<float>,
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:        hpx::cuda::experimental::check_cuda_error(cudaSetDevice(device_id));
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:        cudaStream_t device_stream;
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:        hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:            cudaStreamCreate(&device_stream));
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:        cuda_trivial_kernel<float>(
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:        auto fut = hpx::cuda::experimental::detail::get_future_with_event(
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:        hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:            cudaStreamDestroy(device_stream));
libs/core/async_cuda/tests/unit/cuda_multi_device_polling.cpp:    std::cout << "[HPX Cuda multi device polling] - Starting...\n" << std::endl;
libs/core/async_cuda/tests/unit/trivial_demo.cu:#include <hpx/async_cuda/cuda_exception.hpp>
libs/core/async_cuda/tests/unit/trivial_demo.cu:#include <hpx/async_cuda/custom_gpu_api.hpp>
libs/core/async_cuda/tests/unit/trivial_demo.cu:// Here is a trivial kernel that can be invoked on the GPU
libs/core/async_cuda/tests/unit/trivial_demo.cu:    printf("hello from gpu with value %f\n", static_cast<double>(val));
libs/core/async_cuda/tests/unit/trivial_demo.cu:void cuda_trivial_kernel(T t, cudaStream_t stream)
libs/core/async_cuda/tests/unit/trivial_demo.cu:    ::hpx::cuda::experimental::check_cuda_error(cudaDeviceSynchronize());
libs/core/async_cuda/tests/unit/trivial_demo.cu:    ::hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/unit/trivial_demo.cu:        cudaMalloc((void**) &d_in, sizeof(T)));
libs/core/async_cuda/tests/unit/trivial_demo.cu:    ::hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/unit/trivial_demo.cu:        cudaMalloc((void**) &d_out, sizeof(T)));
libs/core/async_cuda/tests/unit/trivial_demo.cu:    ::hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/unit/trivial_demo.cu:        cudaMemcpy(d_in, &t, sizeof(T), cudaMemcpyHostToDevice));
libs/core/async_cuda/tests/unit/trivial_demo.cu:    ::hpx::cuda::experimental::check_cuda_error(cudaDeviceSynchronize());
libs/core/async_cuda/tests/unit/trivial_demo.cu:    ::hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/unit/trivial_demo.cu:        cudaMemcpy(out, d_out, sizeof(T), cudaMemcpyDeviceToHost));
libs/core/async_cuda/tests/unit/trivial_demo.cu:    ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_in));
libs/core/async_cuda/tests/unit/trivial_demo.cu:    ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_out));
libs/core/async_cuda/tests/unit/trivial_demo.cu:// we must instantiate them here first in the cuda compiled code.
libs/core/async_cuda/tests/unit/trivial_demo.cu:template void cuda_trivial_kernel<float>(float, cudaStream_t);
libs/core/async_cuda/tests/unit/trivial_demo.cu:template void cuda_trivial_kernel<double>(double, cudaStream_t);
libs/core/async_cuda/tests/unit/CMakeLists.txt:set(tests cuda_future cuda_multi_device_polling transform_stream)
libs/core/async_cuda/tests/unit/CMakeLists.txt:if(HPX_WITH_GPUBLAS)
libs/core/async_cuda/tests/unit/CMakeLists.txt:set(cuda_future_PARAMETERS THREADS_PER_LOCALITY 4)
libs/core/async_cuda/tests/unit/CMakeLists.txt:set(cuda_multi_device_polling_PARAMETERS THREADS_PER_LOCALITY 4)
libs/core/async_cuda/tests/unit/CMakeLists.txt:set(cuda_future_CUDA_SOURCE saxpy trivial_demo)
libs/core/async_cuda/tests/unit/CMakeLists.txt:set(cuda_multi_device_polling_CUDA_SOURCE trivial_demo)
libs/core/async_cuda/tests/unit/CMakeLists.txt:set(transform_stream_CUDA ON)
libs/core/async_cuda/tests/unit/CMakeLists.txt:  if(${${test}_CUDA})
libs/core/async_cuda/tests/unit/CMakeLists.txt:  if(${test}_CUDA_SOURCE)
libs/core/async_cuda/tests/unit/CMakeLists.txt:    foreach(src ${${test}_CUDA_SOURCE})
libs/core/async_cuda/tests/unit/CMakeLists.txt:    FOLDER "Tests/Unit/Modules/Core/AsyncCuda"
libs/core/async_cuda/tests/unit/CMakeLists.txt:    "modules.async_cuda" ${test} ${${test}_PARAMETERS} RUN_SERIAL
libs/core/async_cuda/tests/unit/transform_stream.cu:// Fixed in CUDA 12.3
libs/core/async_cuda/tests/unit/transform_stream.cu:#if !defined(HPX_CUDA_VERSION) || (HPX_CUDA_VERSION > 1202)
libs/core/async_cuda/tests/unit/transform_stream.cu:#include <hpx/modules/async_cuda.hpp>
libs/core/async_cuda/tests/unit/transform_stream.cu:    void operator()(cudaStream_t stream) const
libs/core/async_cuda/tests/unit/transform_stream.cu:    double operator()(int x, cudaStream_t stream) const
libs/core/async_cuda/tests/unit/transform_stream.cu:    int operator()(double x, cudaStream_t stream) const
libs/core/async_cuda/tests/unit/transform_stream.cu:    int* operator()(int* p, cudaStream_t stream) const
libs/core/async_cuda/tests/unit/transform_stream.cu:struct cuda_memcpy_async
libs/core/async_cuda/tests/unit/transform_stream.cu:        return cudaMemcpyAsync(std::forward<Ts>(ts)...);
libs/core/async_cuda/tests/unit/transform_stream.cu:    namespace cu = ::hpx::cuda::experimental;
libs/core/async_cuda/tests/unit/transform_stream.cu:        cu::check_cuda_error(cudaMalloc((void**) &p, sizeof(type)));
libs/core/async_cuda/tests/unit/transform_stream.cu:        auto s = ex::just(p, &p_h, sizeof(type), cudaMemcpyHostToDevice) |
libs/core/async_cuda/tests/unit/transform_stream.cu:            cu::transform_stream(cuda_memcpy_async{}) |
libs/core/async_cuda/tests/unit/transform_stream.cu:            ex::then(&cu::check_cuda_error) |
libs/core/async_cuda/tests/unit/transform_stream.cu:                ex::just(cudaMemcpyDeviceToHost)) |
libs/core/async_cuda/tests/unit/transform_stream.cu:            cu::transform_stream(cuda_memcpy_async{}) |
libs/core/async_cuda/tests/unit/transform_stream.cu:            ex::then(&cu::check_cuda_error) |
libs/core/async_cuda/tests/unit/transform_stream.cu:        cu::check_cuda_error(cudaFree(p));
libs/core/async_cuda/tests/CMakeLists.txt:    add_hpx_pseudo_target(tests.unit.modules.async_cuda)
libs/core/async_cuda/tests/CMakeLists.txt:      tests.unit.modules tests.unit.modules.async_cuda
libs/core/async_cuda/tests/CMakeLists.txt:    add_hpx_pseudo_target(tests.regressions.modules.async_cuda)
libs/core/async_cuda/tests/CMakeLists.txt:      tests.regressions.modules tests.regressions.modules.async_cuda
libs/core/async_cuda/tests/CMakeLists.txt:    add_hpx_pseudo_target(tests.performance.modules.async_cuda)
libs/core/async_cuda/tests/CMakeLists.txt:      tests.performance.modules tests.performance.modules.async_cuda
libs/core/async_cuda/tests/CMakeLists.txt:      modules.async_cuda
libs/core/async_cuda/tests/CMakeLists.txt:      HEADERS ${async_cuda_headers}
libs/core/async_cuda/tests/CMakeLists.txt:      DEPENDENCIES hpx_async_cuda
libs/core/async_cuda/tests/performance/synchronize.cu:// Fixed in CUDA 12.3
libs/core/async_cuda/tests/performance/synchronize.cu:#if !defined(HPX_CUDA_VERSION) || (HPX_CUDA_VERSION > 1202)
libs/core/async_cuda/tests/performance/synchronize.cu:#include <hpx/modules/async_cuda.hpp>
libs/core/async_cuda/tests/performance/synchronize.cu:    cudaStream_t cuda_stream;
libs/core/async_cuda/tests/performance/synchronize.cu:    hpx::cuda::experimental::check_cuda_error(cudaStreamCreate(&cuda_stream));
libs/core/async_cuda/tests/performance/synchronize.cu:            dummy<<<1, 1, 0, cuda_stream>>>();
libs/core/async_cuda/tests/performance/synchronize.cu:            hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/performance/synchronize.cu:                cudaStreamSynchronize(cuda_stream));
libs/core/async_cuda/tests/performance/synchronize.cu:            dummy<<<1, 1, 0, cuda_stream>>>();
libs/core/async_cuda/tests/performance/synchronize.cu:            hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/performance/synchronize.cu:                cudaStreamSynchronize(cuda_stream));
libs/core/async_cuda/tests/performance/synchronize.cu:                dummy<<<1, 1, 0, cuda_stream>>>();
libs/core/async_cuda/tests/performance/synchronize.cu:            hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/performance/synchronize.cu:                cudaStreamSynchronize(cuda_stream));
libs/core/async_cuda/tests/performance/synchronize.cu:            dummy<<<1, 1, 0, cuda_stream>>>();
libs/core/async_cuda/tests/performance/synchronize.cu:        hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/performance/synchronize.cu:            cudaStreamSynchronize(cuda_stream));
libs/core/async_cuda/tests/performance/synchronize.cu:        hpx::cuda::experimental::enable_user_polling poll("default");
libs/core/async_cuda/tests/performance/synchronize.cu:        namespace cu = hpx::cuda::experimental;
libs/core/async_cuda/tests/performance/synchronize.cu:        auto const f = [](cudaStream_t cuda_stream) {
libs/core/async_cuda/tests/performance/synchronize.cu:            dummy<<<1, 1, 0, cuda_stream>>>();
libs/core/async_cuda/tests/performance/synchronize.cu:            cu::transform_stream(ex::just(), f, cuda_stream) | tt::sync_wait();
libs/core/async_cuda/tests/performance/synchronize.cu:        hpx::cuda::experimental::enable_user_polling poll("default");
libs/core/async_cuda/tests/performance/synchronize.cu:        namespace cu = hpx::cuda::experimental;
libs/core/async_cuda/tests/performance/synchronize.cu:        auto const f = [](cudaStream_t cuda_stream) {
libs/core/async_cuda/tests/performance/synchronize.cu:            dummy<<<1, 1, 0, cuda_stream>>>();
libs/core/async_cuda/tests/performance/synchronize.cu:            cu::transform_stream(ex::just(), f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) | tt::sync_wait();
libs/core/async_cuda/tests/performance/synchronize.cu:            cu::transform_stream(ex::just(), f, cuda_stream) | tt::sync_wait();
libs/core/async_cuda/tests/performance/synchronize.cu:        hpx::cuda::experimental::enable_user_polling poll("default");
libs/core/async_cuda/tests/performance/synchronize.cu:        namespace cu = hpx::cuda::experimental;
libs/core/async_cuda/tests/performance/synchronize.cu:        auto const f = [](cudaStream_t cuda_stream) {
libs/core/async_cuda/tests/performance/synchronize.cu:            dummy<<<1, 1, 0, cuda_stream>>>();
libs/core/async_cuda/tests/performance/synchronize.cu:            cu::transform_stream(ex::just(), f, cuda_stream) | ex::then([] {}) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) | ex::then([] {}) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) | ex::then([] {}) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) | ex::then([] {}) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) | ex::then([] {}) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) | ex::then([] {}) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) | ex::then([] {}) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) | ex::then([] {}) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) | ex::then([] {}) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) | tt::sync_wait();
libs/core/async_cuda/tests/performance/synchronize.cu:            cu::transform_stream(ex::just(), f, cuda_stream) | tt::sync_wait();
libs/core/async_cuda/tests/performance/synchronize.cu:        hpx::cuda::experimental::enable_user_polling poll("default");
libs/core/async_cuda/tests/performance/synchronize.cu:        namespace cu = hpx::cuda::experimental;
libs/core/async_cuda/tests/performance/synchronize.cu:        auto const f = [](cudaStream_t cuda_stream) {
libs/core/async_cuda/tests/performance/synchronize.cu:            dummy<<<1, 1, 0, cuda_stream>>>();
libs/core/async_cuda/tests/performance/synchronize.cu:            cu::transform_stream(ex::just(), f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:        hpx::cuda::experimental::enable_user_polling poll("default");
libs/core/async_cuda/tests/performance/synchronize.cu:        namespace cu = hpx::cuda::experimental;
libs/core/async_cuda/tests/performance/synchronize.cu:        auto const f = [](cudaStream_t cuda_stream) {
libs/core/async_cuda/tests/performance/synchronize.cu:            dummy<<<1, 1, 0, cuda_stream>>>();
libs/core/async_cuda/tests/performance/synchronize.cu:            cu::transform_stream(ex::just(), f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:                cu::transform_stream(f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:            cu::transform_stream(ex::just(), f, cuda_stream) |
libs/core/async_cuda/tests/performance/synchronize.cu:    hpx::cuda::experimental::check_cuda_error(cudaStreamDestroy(cuda_stream));
libs/core/async_cuda/tests/performance/CMakeLists.txt:if(HPX_WITH_GPUBLAS)
libs/core/async_cuda/tests/performance/CMakeLists.txt:  list(APPEND benchmarks cuda_executor_throughput)
libs/core/async_cuda/tests/performance/CMakeLists.txt:set(synchronize_CUDA ON)
libs/core/async_cuda/tests/performance/CMakeLists.txt:set(cuda_executor_throughput_PARAMETERS THREADS_PER_LOCALITY 1)
libs/core/async_cuda/tests/performance/CMakeLists.txt:  if(${${benchmark}_CUDA})
libs/core/async_cuda/tests/performance/CMakeLists.txt:    DEPENDENCIES hpx_async_cuda ${${benchmark}_FLAGS}
libs/core/async_cuda/tests/performance/CMakeLists.txt:    FOLDER "Benchmarks/Modules/Core/AsyncCuda"
libs/core/async_cuda/tests/performance/CMakeLists.txt:    "modules.async_cuda" ${benchmark} ${${benchmark}_PARAMETERS}
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:// For compliance with the NVIDIA EULA:
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:// "This software contains source code provided by NVIDIA Corporation."
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:// This is a conversion of the NVIDIA cublas example matrixMulCUBLAS to use
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:// comparison/checks and makes no difference to the GPU execution.
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:// Note: The hpx::cuda::experimental::allocator makes use of device code and if used
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp://     PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:// cudaMalloc/cudaMemcpy etc, so we do not #define HPX_CUBLAS_DEMO_WITH_ALLOCATOR
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:#include <hpx/modules/async_cuda.hpp>
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:    // create a cublas executor we'll use to futurize cuda events
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:    hpx::cuda::experimental::cublas_executor cublas(device,
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:        CUBLAS_POINTER_MODE_HOST, hpx::cuda::experimental::callback_mode{});
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:        typename hpx::cuda::experimental::cublas_executor::future_type;
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:    hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:        cudaMalloc((void**) &d_A, size_A * sizeof(T)));
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:    hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:        cudaMalloc((void**) &d_B, size_B * sizeof(T)));
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:    hpx::cuda::experimental::check_cuda_error(
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:        cudaMalloc((void**) &d_C, size_C * sizeof(T)));
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:    hpx::post(cublas, cudaMemcpyAsync, d_A, h_A.data(), size_A * sizeof(T),
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:        cudaMemcpyHostToDevice);
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:    auto copy_future = hpx::async(cublas, cudaMemcpyAsync, d_B, h_B.data(),
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:        size_B * sizeof(T), cudaMemcpyHostToDevice);
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:    auto test_function = [&](hpx::cuda::experimental::cublas_executor& exec,
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:        // time many cuda kernels spawned one after each other when they complete
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:    hpx::cuda::experimental::cublas_executor exec_callback(
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:        0, CUBLAS_POINTER_MODE_HOST, hpx::cuda::experimental::callback_mode{});
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:        // install cuda future polling handler for this scope block
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:        hpx::cuda::experimental::enable_user_polling poll("default");
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:        hpx::cuda::experimental::cublas_executor exec_event(
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:            0, CUBLAS_POINTER_MODE_HOST, hpx::cuda::experimental::event_mode{});
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:    hpx::cuda::experimental::check_cuda_error(cudaFree(d_A));
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:    hpx::cuda::experimental::check_cuda_error(cudaFree(d_B));
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:    hpx::cuda::experimental::check_cuda_error(cudaFree(d_C));
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:    hpx::cuda::experimental::target target(device);
libs/core/async_cuda/tests/performance/cuda_executor_throughput.cpp:    std::cout << "GPU Device " << target.native_handle().get_device() << ": \""
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:#if defined(HPX_HAVE_GPU_SUPPORT) && defined(HPX_HAVE_GPUBLAS)
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:#include <hpx/async_cuda/cuda_exception.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:#include <hpx/async_cuda/cuda_executor.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:#include <hpx/async_cuda/cuda_future.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:#include <hpx/async_cuda/target.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:// CUDA runtime
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:#include <hpx/async_cuda/custom_gpu_api.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:#include <hpx/async_cuda/custom_blas_api.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:namespace hpx { namespace cuda { namespace experimental {
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:        // not all of these are supported by all cuda/cublas versions
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:    // exception type for failed launch of cuda functions
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:    struct cublas_executor : cuda_executor
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:          : hpx::cuda::experimental::cuda_executor(device, event_mode)
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:            check_cuda_error(cudaSetDevice(device_));
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:        // forward a cuda function through to the cuda executor base class
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:        // (we permit the use of a cublas executor for cuda calls)
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:        inline std::enable_if_t<std::is_same_v<cudaError_t, R>> post(
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:            R (*cuda_function)(Params...), Args&&... args) const
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:            return cuda_executor::post(
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:                cuda_function, HPX_FORWARD(Args, args)...);
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:        // when the task completes, this allows integration of GPU kernels with
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:                    check_cuda_error(cudaSetDevice(device_));
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:        // forward a cuda function through to the cuda executor base class
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:        inline hpx::future<std::enable_if_t<std::is_same_v<cudaError_t, R>>>
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:        async(R (*cuda_function)(Params...), Args&&... args) const
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:            return cuda_executor::async(
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:                cuda_function, HPX_FORWARD(Args, args)...);
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:}}}    // namespace hpx::cuda::experimental
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:    struct is_one_way_executor<hpx::cuda::experimental::cublas_executor>
libs/core/async_cuda/include/hpx/async_cuda/cublas_executor.hpp:    struct is_two_way_executor<hpx::cuda::experimental::cublas_executor>
libs/core/async_cuda/include/hpx/async_cuda/cuda_exception.hpp:// CUDA runtime
libs/core/async_cuda/include/hpx/async_cuda/cuda_exception.hpp:#include <hpx/async_cuda/custom_gpu_api.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cuda_exception.hpp:namespace hpx { namespace cuda { namespace experimental {
libs/core/async_cuda/include/hpx/async_cuda/cuda_exception.hpp:    // exception type for failed launch of cuda functions
libs/core/async_cuda/include/hpx/async_cuda/cuda_exception.hpp:    struct cuda_exception : hpx::exception
libs/core/async_cuda/include/hpx/async_cuda/cuda_exception.hpp:        cuda_exception(const std::string& msg, cudaError_t err)
libs/core/async_cuda/include/hpx/async_cuda/cuda_exception.hpp:        cudaError_t get_cuda_errorcode()
libs/core/async_cuda/include/hpx/async_cuda/cuda_exception.hpp:        cudaError_t err_;
libs/core/async_cuda/include/hpx/async_cuda/cuda_exception.hpp:    // Error message handler for cuda calls
libs/core/async_cuda/include/hpx/async_cuda/cuda_exception.hpp:    inline void check_cuda_error(cudaError_t err)
libs/core/async_cuda/include/hpx/async_cuda/cuda_exception.hpp:        if (err != cudaSuccess)
libs/core/async_cuda/include/hpx/async_cuda/cuda_exception.hpp:            auto temp = std::string("cuda function returned error code :") +
libs/core/async_cuda/include/hpx/async_cuda/cuda_exception.hpp:                cudaGetErrorString(err);
libs/core/async_cuda/include/hpx/async_cuda/cuda_exception.hpp:            throw cuda_exception(temp, err);
libs/core/async_cuda/include/hpx/async_cuda/cuda_exception.hpp:}}}    // namespace hpx::cuda::experimental
libs/core/async_cuda/include/hpx/async_cuda/detail/cuda_debug.hpp:namespace hpx { namespace cuda { namespace experimental { namespace detail {
libs/core/async_cuda/include/hpx/async_cuda/detail/cuda_debug.hpp:    static constexpr print_on cud_debug("CUDA");
libs/core/async_cuda/include/hpx/async_cuda/detail/cuda_debug.hpp:}}}}    // namespace hpx::cuda::experimental::detail
libs/core/async_cuda/include/hpx/async_cuda/detail/cuda_event_callback.hpp:// This file provides functionality similar to CUDA's built-in
libs/core/async_cuda/include/hpx/async_cuda/detail/cuda_event_callback.hpp:// cudaStreamAddCallback, with the difference that an event is recorded and an
libs/core/async_cuda/include/hpx/async_cuda/detail/cuda_event_callback.hpp:#include <hpx/async_cuda/custom_gpu_api.hpp>
libs/core/async_cuda/include/hpx/async_cuda/detail/cuda_event_callback.hpp:namespace hpx { namespace cuda { namespace experimental { namespace detail {
libs/core/async_cuda/include/hpx/async_cuda/detail/cuda_event_callback.hpp:        hpx::move_only_function<void(cudaError_t)>;
libs/core/async_cuda/include/hpx/async_cuda/detail/cuda_event_callback.hpp:        event_callback_function_type&& f, cudaStream_t stream, int device = 0);
libs/core/async_cuda/include/hpx/async_cuda/detail/cuda_event_callback.hpp:}}}}    // namespace hpx::cuda::experimental::detail
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:#include <hpx/async_cuda/cuda_exception.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:#include <hpx/async_cuda/cuda_future.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:#include <hpx/async_cuda/target.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:// CUDA runtime
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:#include <hpx/async_cuda/custom_gpu_api.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:namespace hpx { namespace cuda { namespace experimental {
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:        // A helper object to call a cudafunction returning a cudaError type
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:        // specialization for return type of cudaError_t
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:        struct dispatch_helper<cudaError_t, Args...>
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:                cudaError_t (*f)(Args...), Args... args) const
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:                check_cuda_error(f(args...));
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:    // Allows the launching of cuda functions and kernels on a stream with futures
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:    struct cuda_executor_base
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:        // constructors - create a cuda stream that all tasks invoked by
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:        cuda_executor_base(std::size_t device, bool event_mode)
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:            target_ = std::make_shared<hpx::cuda::experimental::target>(device);
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:        cudaStream_t stream_;
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:        std::shared_ptr<hpx::cuda::experimental::target> target_;
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:    struct cuda_executor : cuda_executor_base
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:        // construct - create a cuda stream that all tasks invoked by
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:        explicit cuda_executor(std::size_t device, bool event_mode = true)
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:          : cuda_executor_base(device, event_mode)
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:        ~cuda_executor() {}
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:            cuda_executor const& exec, F&& f, Ts&&... ts)
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:            cuda_executor const& exec, F&& f, Ts&&... ts)
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:        // the return value is the value returned from the cuda call
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:        // (typically this will be cudaError_t).
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:        // Throws cuda_exception if the async launch fails.
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:        void post(R (*cuda_function)(Params...), Args&&... args) const
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:            check_cuda_error(cudaSetDevice(device_));
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:            // insert the stream handle in the arg list and call the cuda function
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:            helper(cuda_function, HPX_FORWARD(Args, args)..., stream_);
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:        // when the task completes, this allows integregration of GPU kernels with
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:        // Puts a cuda_exception in the future if the async launch fails.
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:            R (*cuda_kernel)(Params...), Args&&... args) const
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:                    check_cuda_error(cudaSetDevice(device_));
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:                    // insert the stream handle in the arg list and call the cuda function
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:                    helper(cuda_kernel, HPX_FORWARD(Args, args)..., stream_);
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:}}}    // namespace hpx::cuda::experimental
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:    struct is_one_way_executor<hpx::cuda::experimental::cuda_executor>
libs/core/async_cuda/include/hpx/async_cuda/cuda_executor.hpp:    struct is_two_way_executor<hpx::cuda::experimental::cuda_executor>
libs/core/async_cuda/include/hpx/async_cuda/custom_blas_api.hpp:#if defined(HPX_HAVE_HIP) && defined(HPX_HAVE_GPUBLAS)
libs/core/async_cuda/include/hpx/async_cuda/custom_blas_api.hpp:#elif defined(HPX_HAVE_CUDA) && defined(HPX_HAVE_GPUBLAS)
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:#include <hpx/async_cuda/custom_gpu_api.hpp>
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:namespace hpx { namespace cuda { namespace experimental {
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:            cudaError_t status, R&& r, Ts&&... ts)
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:            HPX_ASSERT(status != cudaErrorNotReady);
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:            if (status == cudaSuccess)
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:                        cuda_exception(std::string("Getting event after "
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:                                                   "CUDA stream transform "
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:                                cudaGetErrorString(status),
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:        void extend_argument_lifetimes(cudaStream_t stream, Ts&&... ts)
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:                        cudaError_t status) {
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:                        HPX_ASSERT(status != cudaErrorNotReady);
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:        void set_value_immediate_void(cudaStream_t stream, R&& r, Ts&&... ts)
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:            cudaStream_t stream, R&& r, Ts&&... ts)
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:                    cudaError_t status) mutable {
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:            cudaStream_t stream, R&& r, T&& t, Ts&&... ts)
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:            cudaStream_t stream, R&& r, T&& t, Ts&&... ts)
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:                    cudaError_t status) mutable {
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:            cudaStream_t stream;
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:            transform_stream_receiver(R_&& r, F_&& f, cudaStream_t stream)
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:                                              Ts..., cudaStream_t>::type>)
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:            cudaStream_t stream{};
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:                static_assert(hpx::is_invocable_v<F, Args..., cudaStream_t>,
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:                    hpx::util::invoke_result_t<F, Args..., cudaStream_t>;
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:                static_assert(hpx::is_invocable_v<F, Ts..., cudaStream_t>,
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:                    hpx::util::invoke_result_t<F, Ts..., cudaStream_t>;
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:    // - a cudaStream_t is inserted as an additional argument into the call to f
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:                !std::is_same<std::decay_t<F>, cudaStream_t>::value>>
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:            transform_stream_t, S&& s, F&& f, cudaStream_t stream = {})
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:            transform_stream_t, F&& f, cudaStream_t stream = {})
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:                transform_stream_t, F, cudaStream_t>{HPX_FORWARD(F, f), stream};
libs/core/async_cuda/include/hpx/async_cuda/transform_stream.hpp:}}}    // namespace hpx::cuda::experimental
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:#include <hpx/async_cuda/cuda_exception.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:#include <hpx/async_cuda/custom_gpu_api.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:namespace hpx { namespace cuda { namespace experimental {
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:    // a pool of cudaEvent_t objects.
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:    // Since allocation of a cuda event passes into the cuda runtime
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:    struct cuda_event_pool
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:        HPX_CORE_EXPORT static cuda_event_pool& get_event_pool();
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:        ~cuda_event_pool()
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:                "Number of CUDA event pools does not match the number of "
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:                check_cuda_error(cudaSetDevice(device));
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:                cudaEvent_t event;
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:                        check_cuda_error(cudaEventDestroy(event));
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:        inline bool pop(cudaEvent_t& event, int device = 0)
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:                "Accessing CUDA event pool with invalid device ID!");
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:                check_cuda_error(cudaGetDevice(&original_device));
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:                    check_cuda_error(cudaSetDevice(original_device));
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:        inline bool push(cudaEvent_t event, int device = 0)
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:                "Accessing CUDA event pool with invalid device ID!");
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:        cuda_event_pool(cuda_event_pool&&) = delete;
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:        cuda_event_pool& operator=(cuda_event_pool&&) = delete;
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:        cuda_event_pool(cuda_event_pool const&) = delete;
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:        cuda_event_pool& operator=(cuda_event_pool const&) = delete;
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:        cuda_event_pool()
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:            check_cuda_error(cudaGetDeviceCount(&max_number_devices_));
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:                "CUDA polling enabled and called, yet no CUDA device found!");
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:            check_cuda_error(cudaGetDevice(&original_device));
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:                check_cuda_error(cudaSetDevice(device));
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:            check_cuda_error(cudaSetDevice(original_device));
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:            check_cuda_error(cudaSetDevice(device));
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:            cudaEvent_t event;
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:            // Create an cuda_event to query a CUDA/CUBLAS kernel for completion.
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:            // [1]: CUDA Runtime API, section 5.5 cuda_event Management
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:            check_cuda_error(
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:                cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:        // One pool per GPU - each pool is dynamically sized and can grow if needed
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:        std::deque<hpx::lockfree::stack<cudaEvent_t>> free_lists_;
libs/core/async_cuda/include/hpx/async_cuda/cuda_event.hpp:}}}    // namespace hpx::cuda::experimental
libs/core/async_cuda/include/hpx/async_cuda/get_targets.hpp:namespace hpx { namespace cuda { namespace experimental {
libs/core/async_cuda/include/hpx/async_cuda/get_targets.hpp:}}}    // namespace hpx::cuda::experimental
libs/core/async_cuda/include/hpx/async_cuda/target.hpp:#include <hpx/async_cuda/cuda_future.hpp>
libs/core/async_cuda/include/hpx/async_cuda/target.hpp:#include <hpx/async_cuda/get_targets.hpp>
libs/core/async_cuda/include/hpx/async_cuda/target.hpp:#include <hpx/async_cuda/custom_gpu_api.hpp>
libs/core/async_cuda/include/hpx/async_cuda/target.hpp:namespace hpx { namespace cuda { namespace experimental {
libs/core/async_cuda/include/hpx/async_cuda/target.hpp:            cudaStream_t get_stream() const;
libs/core/async_cuda/include/hpx/async_cuda/target.hpp:            mutable cudaStream_t stream_;
libs/core/async_cuda/include/hpx/async_cuda/target.hpp:            return cuda::experimental::get_local_targets();
libs/core/async_cuda/include/hpx/async_cuda/target.hpp:}}}    // namespace hpx::cuda::experimental
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:        #define CUDART_CB __stdcall
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:        #define CUDART_CB
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaDeviceProp hipDeviceProp_t
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaDeviceSynchronize hipDeviceSynchronize
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaError_t hipError_t
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaErrorNotReady hipErrorNotReady
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaEvent_t hipEvent_t
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaEventCreateWithFlags hipEventCreateWithFlags
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaEventDestroy hipEventDestroy
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaEventDisableTiming hipEventDisableTiming
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaEventQuery hipEventQuery
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaEventRecord hipEventRecord
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaFree hipFree
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaGetDevice hipGetDevice
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaGetDeviceCount hipGetDeviceCount
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaGetDeviceProperties hipGetDeviceProperties
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaGetErrorString hipGetErrorString
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaGetLastError hipGetLastError
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaGetParameterBuffer hipGetParameterBuffer
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaLaunchDevice hipLaunchDevice
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaLaunchKernel hipLaunchKernel
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaMalloc hipMalloc
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaMallocHost hipHostMalloc
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaMemcpy hipMemcpy
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaMemcpyAsync hipMemcpyAsync
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaMemcpyHostToDevice hipMemcpyHostToDevice
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaMemGetInfo hipMemGetInfo
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaMemsetAsync hipMemsetAsync
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaSetDevice hipSetDevice
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaStream_t hipStream_t
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaStreamAddCallback hipStreamAddCallback
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaStreamCreate hipStreamCreate
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaStreamCreateWithFlags hipStreamCreateWithFlags
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaStreamDestroy hipStreamDestroy
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaStreamNonBlocking hipStreamNonBlocking
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaStreamSynchronize hipStreamSynchronize
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #define cudaSuccess hipSuccess
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:#elif defined(HPX_HAVE_CUDA)
libs/core/async_cuda/include/hpx/async_cuda/custom_gpu_api.hpp:    #include <cuda_runtime.h>
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:#include <hpx/async_cuda/cuda_event.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:#include <hpx/async_cuda/cuda_exception.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:#include <hpx/async_cuda/custom_gpu_api.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:#include <hpx/async_cuda/detail/cuda_debug.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:#include <hpx/async_cuda/detail/cuda_event_callback.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:namespace hpx { namespace cuda { namespace experimental {
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:        // cuda future data implementation
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:        // by a cuda callback when the stream event occurs
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                cudaStream_t stream, int device)
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                        cudaError_t status) {
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                        HPX_ASSERT(status != cudaErrorNotReady);
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                        if (status == cudaSuccess)
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                                std::make_exception_ptr(cuda_exception(
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                                        "cuda function returned error code :") +
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                                        cudaGetErrorString(status),
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                cudaStream_t stream)
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                // Hold on to the shared state on behalf of the cuda runtime
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                cudaError_t error =
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                    cudaStreamAddCallback(stream, stream_callback, this, 0);
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                if (error != cudaSuccess)
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                    check_cuda_error(error);
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:            // this is called from the nvidia backend on a non-hpx thread
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:            static void CUDART_CB stream_callback(
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                cudaStream_t, cudaError_t error, void* user_data)
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                if (error != cudaSuccess)
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                            "cuda::detail::future_data::stream_callback()",
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                            std::string("cudaStreamAddCallback failed: ") +
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                                cudaGetErrorString(error)));
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:            Allocator const& a, cudaStream_t stream, int device)
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:            Allocator const& a, cudaStream_t stream)
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:            Allocator const& a, cudaStream_t stream, int device = -1)
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:                check_cuda_error(cudaGetDevice(&device));
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:            cudaStream_t);
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:            cudaStream_t stream, int device = -1);
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:}}}      // namespace hpx::cuda::experimental
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:        hpx::cuda::experimental::detail::future_data<Allocator, Mode>,
libs/core/async_cuda/include/hpx/async_cuda/cuda_future.hpp:            hpx::cuda::experimental::detail::future_data<NewAllocator, Mode>;
libs/core/async_cuda/include/hpx/async_cuda/cuda_polling_helper.hpp:#include <hpx/async_cuda/detail/cuda_event_callback.hpp>
libs/core/async_cuda/include/hpx/async_cuda/cuda_polling_helper.hpp:namespace hpx { namespace cuda { namespace experimental {
libs/core/async_cuda/include/hpx/async_cuda/cuda_polling_helper.hpp:}}}    // namespace hpx::cuda::experimental
libs/core/async_cuda/CMakeLists.txt:if(NOT (HPX_WITH_CUDA OR HPX_WITH_HIP))
libs/core/async_cuda/CMakeLists.txt:# Default location is $HPX_ROOT/libs/async_cuda/include
libs/core/async_cuda/CMakeLists.txt:set(async_cuda_headers
libs/core/async_cuda/CMakeLists.txt:    hpx/async_cuda/cuda_event.hpp
libs/core/async_cuda/CMakeLists.txt:    hpx/async_cuda/cuda_executor.hpp
libs/core/async_cuda/CMakeLists.txt:    hpx/async_cuda/cuda_exception.hpp
libs/core/async_cuda/CMakeLists.txt:    hpx/async_cuda/cuda_future.hpp
libs/core/async_cuda/CMakeLists.txt:    hpx/async_cuda/cuda_polling_helper.hpp
libs/core/async_cuda/CMakeLists.txt:    hpx/async_cuda/cublas_executor.hpp
libs/core/async_cuda/CMakeLists.txt:    hpx/async_cuda/custom_blas_api.hpp
libs/core/async_cuda/CMakeLists.txt:    hpx/async_cuda/custom_gpu_api.hpp
libs/core/async_cuda/CMakeLists.txt:    hpx/async_cuda/detail/cuda_debug.hpp
libs/core/async_cuda/CMakeLists.txt:    hpx/async_cuda/detail/cuda_event_callback.hpp
libs/core/async_cuda/CMakeLists.txt:    hpx/async_cuda/get_targets.hpp
libs/core/async_cuda/CMakeLists.txt:    hpx/async_cuda/target.hpp
libs/core/async_cuda/CMakeLists.txt:    hpx/async_cuda/transform_stream.hpp
libs/core/async_cuda/CMakeLists.txt:# Default location is $HPX_ROOT/libs/async_cuda/include_compatibility
libs/core/async_cuda/CMakeLists.txt:set(async_cuda_compat_headers
libs/core/async_cuda/CMakeLists.txt:    hpx/compute/cuda/get_targets.hpp => hpx/async_cuda/get_targets.hpp
libs/core/async_cuda/CMakeLists.txt:    hpx/compute/cuda/target.hpp => hpx/async_cuda/target.hpp
libs/core/async_cuda/CMakeLists.txt:set(async_cuda_sources cuda_event_callback.cpp cuda_future.cpp cuda_target.cpp
libs/core/async_cuda/CMakeLists.txt:                       get_targets.cpp cuda_event.cpp
libs/core/async_cuda/CMakeLists.txt:  set(async_cuda_extra_deps ${async_cuda_extra_deps} roc::hipblas)
libs/core/async_cuda/CMakeLists.txt:elseif(HPX_WITH_CUDA AND TARGET Cuda::cuda)
libs/core/async_cuda/CMakeLists.txt:  set(async_cuda_extra_deps ${async_cuda_extra_deps} Cuda::cuda
libs/core/async_cuda/CMakeLists.txt:                            ${CUDA_CUBLAS_LIBRARIES}
libs/core/async_cuda/CMakeLists.txt:  core async_cuda
libs/core/async_cuda/CMakeLists.txt:  SOURCES ${async_cuda_sources}
libs/core/async_cuda/CMakeLists.txt:  HEADERS ${async_cuda_headers}
libs/core/async_cuda/CMakeLists.txt:  COMPAT_HEADERS ${async_cuda_compat_headers}
libs/core/async_cuda/CMakeLists.txt:  DEPENDENCIES ${async_cuda_extra_deps}
libs/core/async_cuda/examples/CMakeLists.txt:  add_hpx_pseudo_target(examples.modules.async_cuda)
libs/core/async_cuda/examples/CMakeLists.txt:  add_hpx_pseudo_dependencies(examples.modules examples.modules.async_cuda)
libs/core/async_cuda/examples/CMakeLists.txt:    add_hpx_pseudo_target(tests.examples.modules.async_cuda)
libs/core/async_cuda/examples/CMakeLists.txt:      tests.examples.modules tests.examples.modules.async_cuda
libs/core/async_cuda/src/cuda_future.cpp:#include <hpx/async_cuda/cuda_future.hpp>
libs/core/async_cuda/src/cuda_future.cpp:#include <hpx/async_cuda/custom_gpu_api.hpp>
libs/core/async_cuda/src/cuda_future.cpp:namespace hpx { namespace cuda { namespace experimental { namespace detail {
libs/core/async_cuda/src/cuda_future.cpp:    hpx::future<void> get_future_with_callback(cudaStream_t stream)
libs/core/async_cuda/src/cuda_future.cpp:    hpx::future<void> get_future_with_event(cudaStream_t stream, int device)
libs/core/async_cuda/src/cuda_future.cpp:}}}}    // namespace hpx::cuda::experimental::detail
libs/core/async_cuda/src/cuda_event_callback.cpp:#include <hpx/async_cuda/cuda_event.hpp>
libs/core/async_cuda/src/cuda_event_callback.cpp:#include <hpx/async_cuda/custom_gpu_api.hpp>
libs/core/async_cuda/src/cuda_event_callback.cpp:#include <hpx/async_cuda/detail/cuda_debug.hpp>
libs/core/async_cuda/src/cuda_event_callback.cpp:#include <hpx/async_cuda/detail/cuda_event_callback.hpp>
libs/core/async_cuda/src/cuda_event_callback.cpp:namespace hpx { namespace cuda { namespace experimental { namespace detail {
libs/core/async_cuda/src/cuda_event_callback.cpp:    // Holds a CUDA event and a callback. The callback is intended to be called when
libs/core/async_cuda/src/cuda_event_callback.cpp:        cudaEvent_t event;
libs/core/async_cuda/src/cuda_event_callback.cpp:            "CUDA event polling has not been enabled on any pool. Make sure "
libs/core/async_cuda/src/cuda_event_callback.cpp:            "that CUDA event polling is enabled on at least one thread pool.");
libs/core/async_cuda/src/cuda_event_callback.cpp:        event_callback_function_type&& f, cudaStream_t stream, int device)
libs/core/async_cuda/src/cuda_event_callback.cpp:        cudaEvent_t event;
libs/core/async_cuda/src/cuda_event_callback.cpp:        if (!cuda_event_pool::get_event_pool().pop(event, device))
libs/core/async_cuda/src/cuda_event_callback.cpp:        check_cuda_error(cudaEventRecord(event, stream));
libs/core/async_cuda/src/cuda_event_callback.cpp:    // Background progress function for async CUDA operations. Checks for completed
libs/core/async_cuda/src/cuda_event_callback.cpp:    // cudaEvent_t and calls the associated callback when ready. We first process
libs/core/async_cuda/src/cuda_event_callback.cpp:        std::unique_lock<hpx::cuda::experimental::detail::mutex_type> lk(
libs/core/async_cuda/src/cuda_event_callback.cpp:        cuda_event_pool& pool =
libs/core/async_cuda/src/cuda_event_callback.cpp:            hpx::cuda::experimental::cuda_event_pool::get_event_pool();
libs/core/async_cuda/src/cuda_event_callback.cpp:                    cudaError_t status = cudaEventQuery(continuation.event);
libs/core/async_cuda/src/cuda_event_callback.cpp:                    if (status == cudaErrorNotReady)
libs/core/async_cuda/src/cuda_event_callback.cpp:            cudaError_t status = cudaEventQuery(continuation.event);
libs/core/async_cuda/src/cuda_event_callback.cpp:            if (status == cudaErrorNotReady)
libs/core/async_cuda/src/cuda_event_callback.cpp:        sched->set_cuda_polling_functions(
libs/core/async_cuda/src/cuda_event_callback.cpp:            &hpx::cuda::experimental::detail::poll, &get_work_count);
libs/core/async_cuda/src/cuda_event_callback.cpp:            std::unique_lock<hpx::cuda::experimental::detail::mutex_type> lk(
libs/core/async_cuda/src/cuda_event_callback.cpp:                "CUDA event polling was disabled while there are unprocessed "
libs/core/async_cuda/src/cuda_event_callback.cpp:                "CUDA events. Make sure CUDA event polling is not disabled too "
libs/core/async_cuda/src/cuda_event_callback.cpp:                "CUDA event polling was disabled while there are unprocessed "
libs/core/async_cuda/src/cuda_event_callback.cpp:                "CUDA events. Make sure CUDA event polling is not disabled too "
libs/core/async_cuda/src/cuda_event_callback.cpp:        sched->clear_cuda_polling_function();
libs/core/async_cuda/src/cuda_event_callback.cpp:}}}}    // namespace hpx::cuda::experimental::detail
libs/core/async_cuda/src/get_targets.cpp:#include <hpx/async_cuda/target.hpp>
libs/core/async_cuda/src/get_targets.cpp:#include <hpx/async_cuda/custom_gpu_api.hpp>
libs/core/async_cuda/src/get_targets.cpp:namespace hpx { namespace cuda { namespace experimental {
libs/core/async_cuda/src/get_targets.cpp:        cudaError_t error = cudaGetDeviceCount(&device_count);
libs/core/async_cuda/src/get_targets.cpp:        if (error != cudaSuccess)
libs/core/async_cuda/src/get_targets.cpp:                "cuda::experimental::get_local_targets()",
libs/core/async_cuda/src/get_targets.cpp:                std::string("cudaGetDeviceCount failed: ") +
libs/core/async_cuda/src/get_targets.cpp:                    cudaGetErrorString(error));
libs/core/async_cuda/src/get_targets.cpp:                "cuda::experimental::get_local_targets()",
libs/core/async_cuda/src/get_targets.cpp:                "cudaGetDeviceCount failed: could not find any devices");
libs/core/async_cuda/src/get_targets.cpp:            std::cout << "GPU Device " << target.native_handle().get_device()
libs/core/async_cuda/src/get_targets.cpp:}}}    // namespace hpx::cuda::experimental
libs/core/async_cuda/src/cuda_target.cpp:#include <hpx/async_cuda/target.hpp>
libs/core/async_cuda/src/cuda_target.cpp:#include <hpx/async_cuda/custom_gpu_api.hpp>
libs/core/async_cuda/src/cuda_target.cpp:namespace hpx { namespace cuda { namespace experimental {
libs/core/async_cuda/src/cuda_target.cpp:        cudaDeviceProp props;
libs/core/async_cuda/src/cuda_target.cpp:        cudaError_t error = cudaGetDeviceProperties(&props, device_);
libs/core/async_cuda/src/cuda_target.cpp:        if (error != cudaSuccess)
libs/core/async_cuda/src/cuda_target.cpp:                "cuda::init_processing_unit()",
libs/core/async_cuda/src/cuda_target.cpp:                std::string("cudaGetDeviceProperties failed: ") +
libs/core/async_cuda/src/cuda_target.cpp:                    cudaGetErrorString(error));
libs/core/async_cuda/src/cuda_target.cpp:            cudaError_t err = cudaStreamDestroy(stream_);    // ignore error
libs/core/async_cuda/src/cuda_target.cpp:    cudaStream_t target::native_handle_type::get_stream() const
libs/core/async_cuda/src/cuda_target.cpp:            cudaError_t error = cudaSetDevice(device_);
libs/core/async_cuda/src/cuda_target.cpp:            if (error != cudaSuccess)
libs/core/async_cuda/src/cuda_target.cpp:                    "cuda::experimental::target::native_handle::get_stream()",
libs/core/async_cuda/src/cuda_target.cpp:                    std::string("cudaSetDevice failed: ") +
libs/core/async_cuda/src/cuda_target.cpp:                        cudaGetErrorString(error));
libs/core/async_cuda/src/cuda_target.cpp:            error = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
libs/core/async_cuda/src/cuda_target.cpp:            if (error != cudaSuccess)
libs/core/async_cuda/src/cuda_target.cpp:                    "cuda::experimental::target::native_handle::get_stream()",
libs/core/async_cuda/src/cuda_target.cpp:                    std::string("cudaStreamCreate failed: ") +
libs/core/async_cuda/src/cuda_target.cpp:                        cudaGetErrorString(error));
libs/core/async_cuda/src/cuda_target.cpp:        cudaStream_t stream = handle_.get_stream();
libs/core/async_cuda/src/cuda_target.cpp:                "cuda::experimental::target::synchronize",
libs/core/async_cuda/src/cuda_target.cpp:        cudaError_t error = cudaStreamSynchronize(stream);
libs/core/async_cuda/src/cuda_target.cpp:        if (error != cudaSuccess)
libs/core/async_cuda/src/cuda_target.cpp:                "cuda::experimental::target::synchronize",
libs/core/async_cuda/src/cuda_target.cpp:                std::string("cudaStreamSynchronize failed: ") +
libs/core/async_cuda/src/cuda_target.cpp:                    cudaGetErrorString(error));
libs/core/async_cuda/src/cuda_target.cpp:}}}    // namespace hpx::cuda::experimental
libs/core/async_cuda/src/cuda_event.cpp:#include <hpx/async_cuda/cuda_event.hpp>
libs/core/async_cuda/src/cuda_event.cpp:#include <hpx/async_cuda/custom_gpu_api.hpp>
libs/core/async_cuda/src/cuda_event.cpp:namespace hpx { namespace cuda { namespace experimental {
libs/core/async_cuda/src/cuda_event.cpp:    cuda_event_pool& cuda_event_pool::get_event_pool()
libs/core/async_cuda/src/cuda_event.cpp:        static cuda_event_pool event_pool_;
libs/core/async_cuda/src/cuda_event.cpp:}}}    // namespace hpx::cuda::experimental
libs/core/modules.rst:   /libs/core/async_cuda/docs/index.rst
libs/full/include/include/hpx/compute.hpp:#if defined(HPX_HAVE_CUDA) || defined(HPX_HAVE_HIP)
libs/full/include/include/hpx/compute.hpp:#include <hpx/modules/async_cuda.hpp>
libs/full/include/include/hpx/include/compute.hpp:#if defined(HPX_HAVE_CUDA) || defined(HPX_HAVE_HIP)
libs/full/include/include/hpx/include/compute.hpp:#include <hpx/modules/async_cuda.hpp>
libs/full/parcelset/tests/unit/CMakeLists.txt:  target_include_directories(${test}_test SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
libs/full/actions_base/include/hpx/actions_base/plain_action.hpp:#if defined(__NVCC__) || defined(__CUDACC__)
libs/full/actions_base/include/hpx/actions_base/plain_action.hpp:#if defined(__NVCC__) || defined(__CUDACC__)
libs/full/components/include/hpx/components/make_client.hpp:    // this is broken at least up until CUDA V11.5
libs/full/components/include/hpx/components/make_client.hpp:#if !defined(HPX_CUDA_VERSION)
libs/full/components/include/hpx/components/make_client.hpp:    // this is broken at least up until CUDA V11.5
libs/full/components/include/hpx/components/make_client.hpp:#if !defined(HPX_CUDA_VERSION)
libs/full/components/include/hpx/components/make_client.hpp:    // this is broken at least up until CUDA V11.5
libs/full/components/include/hpx/components/make_client.hpp:#if !defined(HPX_CUDA_VERSION)
libs/full/async_distributed/include/hpx/async_distributed/bind_action.hpp:#if !defined(__NVCC__) && !defined(__CUDACC__)
docs/sphinx/api/public_api.rst::c:macro:`HPX_ASSERT` can also be used in CUDA device code.
docs/sphinx/api/public_api.rst:CUDA device code, unlike ``std::tuple``.
docs/sphinx/manual/building_hpx.rst:.. option:: HPX_WITH_CUDA
docs/sphinx/manual/building_hpx.rst:   Enable support for CUDA. Use ``CMAKE_CUDA_COMPILER`` to set the CUDA compiler. This is a standard
docs/sphinx/about_hpx/people.rst:* Weile Wei, for fixing |hpx| builds with CUDA on Summit.
docs/sphinx/about_hpx/people.rst:* Marcin Copik, who worked on implementing GPU support for C++AMP and HCC. He
docs/sphinx/about_hpx/people.rst:* Patrick Diehl, who worked on implementing CUDA support for our companion
docs/sphinx/about_hpx/people.rst:  library targeting GPGPUs (|hpxcl|_).
docs/sphinx/about_hpx/people.rst:  related to |opencl|_) and implementing an |hpx| backend for |pocl|_, a
docs/sphinx/about_hpx/people.rst:  portable computing language solution based on |opencl|_.
docs/sphinx/about_hpx/people.rst:  |cuda|_).
docs/sphinx/why_hpx.rst:sockets, and heterogeneous structures of GPUs. Both efficiency and scalability
docs/sphinx/why_hpx.rst:we use `GPGPUs <http://en.wikipedia.org/wiki/GPGPU>`_ today. It is important to
docs/sphinx/why_hpx.rst:time to and from `GPGPUs <http://en.wikipedia.org/wiki/GPGPU>`_ as much as
docs/sphinx/releases/whats_new_0_9_99.rst:  heterogeneous architectures (currently CUDA). This functionality is an early
docs/sphinx/releases/whats_new_0_9_99.rst:* :hpx-pr:`2227` - Support for HPXCL's opencl::event
docs/sphinx/releases/whats_new_0_9_99.rst:* :hpx-pr:`2187` - Mask 128-bit ints if CUDA is being used
docs/sphinx/releases/whats_new_1_7_0.rst:  configured with CUDA or HIP. In this release ``HPX_COMPUTE_HOST_CODE`` is
docs/sphinx/releases/whats_new_1_7_0.rst:* :hpx-issue:`5306` - asio fails to build with CUDA 10.0
docs/sphinx/releases/whats_new_1_7_0.rst:* :hpx-pr:`5434` - Update CUDA polling logging to be more verbose
docs/sphinx/releases/whats_new_1_7_0.rst:* :hpx-pr:`5373` - More changes to clang-cuda Jenkins configuration
docs/sphinx/releases/whats_new_1_7_0.rst:* :hpx-pr:`5363` - Update cudatoolkit module name in clang-cuda Jenkins
docs/sphinx/releases/whats_new_1_7_0.rst:* :hpx-pr:`5353` - Add CUDA timestamp support to HPX Hardware Clock
docs/sphinx/releases/whats_new_1_7_0.rst:* :hpx-pr:`5342` - Refactor CUDA event polling
docs/sphinx/releases/whats_new_1_7_0.rst:* :hpx-pr:`5293` - Fix Clang 11 cuda_future test bug
docs/sphinx/releases/whats_new_1_7_0.rst:* :hpx-pr:`5277` - Small fixes and improvements to CUDA/MPI polling
docs/sphinx/releases/whats_new_1_7_0.rst:* :hpx-pr:`5215` - Update ROCm to 4.0.1 on Rostam
docs/sphinx/releases/whats_new_1_8_0.rst:- CUDA  version required updated to 11.4.
docs/sphinx/releases/whats_new_1_8_0.rst:  - ``HPX_WITH_COMPUTE_CUDA``
docs/sphinx/releases/whats_new_1_8_0.rst:  - ``HPX_WITH_ASYNC_CUDA``
docs/sphinx/releases/whats_new_1_8_0.rst:* :hpx-issue:`5812` - OctoTiger does not compile with HPX master and CUDA 11.5
docs/sphinx/releases/whats_new_1_8_0.rst:* :hpx-issue:`5692` - Kokkos compilation fails when using both HPX and CUDA execution spaces with gcc 9.3.0
docs/sphinx/releases/whats_new_1_8_0.rst:* :hpx-issue:`5647` - [User input needed] Remove (CUDA) compute functionality?
docs/sphinx/releases/whats_new_1_8_0.rst:* :hpx-issue:`5472` - Compilation error with cuda/11.3 
docs/sphinx/releases/whats_new_1_8_0.rst:* :hpx-pr:`5813` - The CUDA problem is not fixed in V11.5 yet...
docs/sphinx/releases/whats_new_1_8_0.rst:* :hpx-pr:`5803` - Attempt to fix CUDA related OctoTiger problems
docs/sphinx/releases/whats_new_1_8_0.rst:* :hpx-pr:`5796` - Disable CUDA tests that cause NVCC to silently fail without error messages
docs/sphinx/releases/whats_new_1_8_0.rst:* :hpx-pr:`5677` - Remove compute_cuda module
docs/sphinx/releases/whats_new_1_8_0.rst:* :hpx-pr:`5550` - Update CUDA module in clang-cuda configuration
docs/sphinx/releases/whats_new_1_8_0.rst:* :hpx-pr:`5479` - Fix version check for CUDA noexcept/result_of bug
docs/sphinx/releases/whats_new_1_8_0.rst:* :hpx-pr:`5475` - Require CMake 3.18 as it is already a requirement for CUDA
docs/sphinx/releases/whats_new_1_8_0.rst:* :hpx-pr:`5474` - Make the cuda parameters of try_compile optional
docs/sphinx/releases/whats_new_1_8_0.rst:* :hpx-pr:`5473` - Update cuda arch and change cuda version
docs/sphinx/releases/whats_new_1_8_0.rst:* :hpx-pr:`5461` - Rename HPX_WITH_CUDA_COMPUTE with HPX_WITH_COMPUTE_CUDA
docs/sphinx/releases/whats_new_1_8_0.rst:* :hpx-pr:`5383` - Tentatively remove runtime_registration_wrapper from cuda futures
docs/sphinx/releases/whats_new_1_8_0.rst:* :hpx-pr:`5283` - Require minimum C++17 and change CUDA handling
docs/sphinx/releases/whats_new_1_0_0.rst:  of CUDA. ``hpx::partitioned_vector`` has been enabled to be usable with
docs/sphinx/releases/whats_new_1_0_0.rst:  more GPU devices.
docs/sphinx/releases/whats_new_1_0_0.rst:* :hpx-issue:`2594` - FindOpenCL.cmake mismatch with the official cmake module
docs/sphinx/releases/whats_new_1_0_0.rst:* :hpx-pr:`2576` - Add missing dependencies of cuda based tests
docs/sphinx/releases/whats_new_1_0_0.rst:* :hpx-pr:`2559` - Allowing CUDA callback to set the future directly from an OS
docs/sphinx/releases/whats_new_1_0_0.rst:* :hpx-pr:`2557` - Removing bogus handling of compile flags for CUDA
docs/sphinx/releases/whats_new_1_0_0.rst:* :hpx-pr:`2553` - Add cmake cuda_arch option
docs/sphinx/releases/whats_new_1_0_0.rst:* :hpx-pr:`2549` - Pre-include defines.hpp to get the macro HPX_HAVE_CUDA value
docs/sphinx/releases/whats_new_1_0_0.rst:* :hpx-pr:`2546` - Some fixes around cuda clang partitioned_vector example
docs/sphinx/releases/whats_new_1_0_0.rst:* :hpx-issue:`2538` - HPX_WITH_CUDA corrupts compilation flags
docs/sphinx/releases/whats_new_1_0_0.rst:* :hpx-pr:`2501` - Some other fixes around cuda examples
docs/sphinx/releases/whats_new_1_0_0.rst:* :hpx-issue:`2500` - nvcc / cuda clang issue due to a missing -DHPX_WITH_CUDA
docs/sphinx/releases/whats_new_1_0_0.rst:* :hpx-pr:`2486` - Only flag HPX code for CUDA if HPX_WITH_CUDA is set
docs/sphinx/releases/whats_new_1_0_0.rst:* :hpx-pr:`2413` - Enable cuda/nvcc or cuda/clang when using
docs/sphinx/releases/whats_new_1_0_0.rst:* :hpx-pr:`2412` - Fix issue in HPX_SetupTarget.cmake when cuda is used
docs/sphinx/releases/whats_new_1_0_0.rst:* :hpx-pr:`2395` - Fix target_link_libraries() issue when HPX Cuda is enabled
docs/sphinx/releases/whats_new_1_0_0.rst:* :hpx-pr:`2328` - fix LibGeoDecomp builds with HPX + GCC 5.3.0 + CUDA 8RC
docs/sphinx/releases/whats_new_1_10_0.rst:- We have applied many fixes to our CUDA, ROCm, and SYCL build environments.
docs/sphinx/releases/whats_new_1_10_0.rst:* :hpx-issue:`6405` - Spack Build Error with ROCm 5.7.0
docs/sphinx/releases/whats_new_1_10_0.rst:* :hpx-issue:`5799` - Investigate CUDA compilation problems
docs/sphinx/releases/whats_new_1_10_0.rst:* :hpx-pr:`6409` - Working around CUDA issue
docs/sphinx/releases/whats_new_1_10_0.rst:* :hpx-pr:`6406` - Working around ROCm compiler issue
docs/sphinx/releases/whats_new_1_6_0.rst:previous CUDA features to now be compiled with hipcc and run on AMD GPUs.
docs/sphinx/releases/whats_new_1_6_0.rst:  CUDA functionality in |hpx| can now be used with HIP. The HIP functionality is
docs/sphinx/releases/whats_new_1_6_0.rst:  for the time being exposed through the same API as the CUDA functionality,
docs/sphinx/releases/whats_new_1_6_0.rst:  i.e. no changes are required in user code. The CUDA, and now HIP,
docs/sphinx/releases/whats_new_1_6_0.rst:  functionality is in the ``hpx::cuda`` namespace.
docs/sphinx/releases/whats_new_1_6_0.rst:* :hpx-pr:`5145` - Adjust handling of CUDA/HIP options in CMake
docs/sphinx/releases/whats_new_1_6_0.rst:* :hpx-pr:`5079` - Add checks to make sure that MPI/CUDA polling is enabled/not
docs/sphinx/releases/whats_new_1_6_0.rst:* :hpx-pr:`5047` - Limit cuda jenkins run to nodes with exclusively Nvidia GPUs
docs/sphinx/releases/whats_new_1_6_0.rst:* :hpx-pr:`4947` - Add HIP support for AMD GPUs
docs/sphinx/releases/whats_new_0_9_9.rst:* :hpx-issue:`1186` - Fixed FindOpenCL to find current AMD APP SDK
docs/sphinx/releases/whats_new_1_2_0.rst:* :hpx-issue:`3270` - Error when compiling CUDA examples
docs/sphinx/releases/whats_new_1_2_0.rst:* :hpx-pr:`3497` - Note that cuda support requires cmake 3.9
docs/sphinx/releases/whats_new_1_2_0.rst:* :hpx-pr:`3492` - Add CUDA_LINK_LIBRARIES_KEYWORD to allow PRIVATE keyword in linkage t…
docs/sphinx/releases/whats_new_1_2_0.rst:* :hpx-pr:`3401` - Fix cuda_future_helper.h when compiling with C++11
docs/sphinx/releases/whats_new_1_2_0.rst:* :hpx-pr:`3292` - Add new cuda kernel synchronization with hpx::future demo
docs/sphinx/releases/whats_new_1_3_0.rst:* :hpx-issue:`3616` - HPX Fails to Build with CUDA 10
docs/sphinx/releases/whats_new_1_3_0.rst:* :hpx-pr:`3702` - Fixing CUDA compiler errors
docs/sphinx/releases/whats_new_1_3_0.rst:* :hpx-pr:`3692` - Only disable ``constexpr`` with clang-cuda, not nvcc+gcc
docs/sphinx/releases/whats_new_0_9_5.rst:* :hpx-issue:`502` - Adding OpenCL and OCLM support to HPX for Windows and Linux
docs/sphinx/releases/whats_new_0_9_5.rst:* :hpx-issue:`488` - Adding OpenCL and OCLM support to HPX for the MSVC platform
docs/sphinx/releases/whats_new_1_1_0.rst:* :hpx-pr:`3246` - Assorted fixes for CUDA
docs/sphinx/releases/whats_new_1_1_0.rst:* :hpx-pr:`3106` - Add cmake test for std::decay_t to fix cuda build
docs/sphinx/releases/whats_new_1_1_0.rst:* :hpx-pr:`2911` - Fixing CUDA problems
docs/sphinx/releases/whats_new_1_1_0.rst:  compilation with CUDA 8.0
docs/sphinx/releases/whats_new_1_1_0.rst:* :hpx-issue:`2815` - HPX fails to compile with HPX_WITH_CUDA=ON and the new
docs/sphinx/releases/whats_new_1_1_0.rst:  CUDA 9.0 RC
docs/sphinx/releases/whats_new_1_1_0.rst:* :hpx-issue:`2689` - HPX build fails when HPX_WITH_CUDA is enabled
docs/sphinx/releases/whats_new_1_1_0.rst:* :hpx-pr:`2688` - Make Cuda Clang builds pass
docs/sphinx/releases/whats_new_1_1_0.rst:* :hpx-pr:`2576` - Add missing dependencies of cuda based tests
docs/sphinx/releases/whats_new_1_5_0.rst:also added experimental asynchronous MPI and CUDA executors. Lastly this release
docs/sphinx/releases/whats_new_1_5_0.rst:* It is now possible to have a basic CUDA support including a helper function to
docs/sphinx/releases/whats_new_1_5_0.rst:  get a future from a CUDA stream and target handling. They are available under
docs/sphinx/releases/whats_new_1_5_0.rst:  the ``hpx::cuda::experimental`` namespace and they can be enabled with the
docs/sphinx/releases/whats_new_1_5_0.rst:  ``-DHPX_WITH_ASYNC_CUDA=ON`` |cmake| option.
docs/sphinx/releases/whats_new_1_5_0.rst:* :hpx-pr:`4919` - Make cuda event pool dynamic instead of fixed size
docs/sphinx/releases/whats_new_1_5_0.rst:* :hpx-pr:`4873` - Set CUDA compute capability on rostam Jenkins builds
docs/sphinx/releases/whats_new_1_5_0.rst:* :hpx-pr:`4839` - Fix async_cuda build problems when distributed runtime is disabled
docs/sphinx/releases/whats_new_1_5_0.rst:* :hpx-pr:`4825` - Move all CUDA functionality to hpx::cuda::experimental namespace
docs/sphinx/releases/whats_new_1_5_0.rst:* :hpx-pr:`4444` - Minor CUDA fixes
docs/sphinx/releases/whats_new_1_5_0.rst:* :hpx-pr:`4385` - Cuda futures
docs/sphinx/releases/whats_new_1_5_0.rst:* :hpx-pr:`4380` - Add a helper function to get a future from a cuda stream
docs/sphinx/releases/whats_new_1_9_1.rst:* :hpx-pr:`6248` - Fix CUDA/HIP Jenkins pipelines
docs/sphinx/releases/whats_new_1_4_0.rst:* :hpx-issue:`3883` - cuda compilation fails because of ``-faligned-new``
docs/sphinx/releases/whats_new_1_4_0.rst:* :hpx-pr:`4240` - Mostly fix clang CUDA compilation
docs/sphinx/releases/whats_new_1_4_0.rst:* :hpx-pr:`4209` - Fix CUDA 10 build
docs/sphinx/releases/whats_new_1_4_0.rst:* :hpx-pr:`4202` - Fix CUDA configuration
docs/sphinx/releases/whats_new_1_4_0.rst:* :hpx-pr:`4192` - Set up CUDA in HPXConfig.cmake
docs/sphinx/releases/whats_new_1_4_0.rst:* :hpx-pr:`4186` - correct vc to cuda in cuda cmake
docs/sphinx/releases/whats_new_1_4_0.rst:* :hpx-pr:`4012` - Fix CUDA compilation
docs/sphinx/releases/whats_new_1_4_0.rst:* :hpx-pr:`3960` - fix compilation with CUDA 10 and GCC 6
docs/sphinx/releases/whats_new_1_4_0.rst:* :hpx-pr:`3933` - Remove ``cudadevrt`` from compile/link flags as it breaks
docs/joss_paper/paper.bib:  booktitle={{Proceedings of the 5th International Workshop on OpenCL}},
docs/joss_paper/paper.bib:  title={{Integration of CUDA Processing within the C++ Library for Parallelism and Concurrency (HPX)}},
docs/joss_paper/paper.bib:  booktitle={{Proceedings of the International Workshop on OpenCL}},
docs/joss_paper/paper.bib:  booktitle={{Proceedings of the International Workshop on OpenCL}},
docs/joss_paper/paper.bib:keywords = {GPU, Manycore, Performance portability, Mini-application, Multidimensional array, Parallel computing, Thread parallelism}
docs/joss_paper/paper.md: - name: NVIDIA, CA, Santa Clara, United States of America
docs/joss_paper/paper.md: HPX has support for several methods of integration with GPUs:
docs/joss_paper/paper.md: HPXCL provides users the ability to manage GPU kernels through a
docs/joss_paper/paper.md: synchronization of CPU and GPU code.
docs/joss_paper/paper.md: solution to heterogeneity by automatically generating GPU kernels
docs/joss_paper/paper.md: from C++ code. This enables HPX to launch both CPU and GPU kernels
tests/unit/build/CMakeLists.txt:  if(HPX_WITH_CLANG_CUDA)
tests/unit/build/CMakeLists.txt:    set(cmake_cuda_compiler -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER})
tests/unit/build/CMakeLists.txt:      -DCMAKE_BUILD_TYPE=${build_type} ${cmake_cuda_compiler} --test-command
tests/performance/local/CMakeLists.txt:if(NOT HPX_WITH_CUDA_COMPUTE)
tests/performance/local/CMakeLists.txt:  set(stream_FLAGS CUDA)
tests/performance/local/stream_report.cpp:        return "gpu allocator";
tests/performance/network/network_storage/slurm-network-storage.sh.in:#SBATCH --constraint=gpu
CITATION.cff:  affiliation: NVIDIA
.cmake-format.py:                        'pargs': { 'flags': ['CUDA',
tools/inspect/ascii_check.cpp:    static const string gPunct("$_{}[]#()<>%:;.?*+-/?&|~!=,\\\"'@^`");
tools/inspect/ascii_check.cpp:            return gPunct.find(c) == string::npos;
tools/perftests_ci/CMakeLists.txt:if(HPX_WITH_HIP OR HPX_WITH_CUDA)
tools/perftests_ci/CMakeLists.txt:  if(TARGET Cuda::cuda)
tools/perftests_ci/CMakeLists.txt:        "${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_COMPILER_VERSION} (${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION})"
tools/perftests_ci/CMakeLists.txt:        "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION} (${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_COMPILER_VERSION})"
CMakeLists.txt:# HPX CUDA configuration
CMakeLists.txt:set(CUDA_OPTION_STRING "Enable support for CUDA (default: OFF)")
CMakeLists.txt:hpx_option(HPX_WITH_CUDA BOOL "${CUDA_OPTION_STRING}" OFF ADVANCED)
CMakeLists.txt:if(HPX_WITH_CUDA AND HPX_WITH_HIP)
CMakeLists.txt:    "HPX_WITH_CUDA=ON and HPX_WITH_HIP=ON. Only one of them can be on at the same time.\
CMakeLists.txt:   OR HPX_WITH_CUDA
CMakeLists.txt:  "Enable generation of pkgconfig files (default: ON on Linux without CUDA/HIP, otherwise OFF)"
CMakeLists.txt:# Need to include the CUDA setup before the config test to enable the CUDA
CMakeLists.txt:include(HPX_SetupCUDA)
CMakeLists.txt:if(HPX_WITH_CUDA OR HPX_WITH_HIP)
CMakeLists.txt:  hpx_add_config_define(HPX_HAVE_GPU_SUPPORT)
CMakeLists.txt:# Setup NVIDIA's stdexec if requested
cmake/HPX_AddModule.cmake:  set(options CUDA CONFIG_FILES NO_CONFIG_IN_GENERATED_HEADERS)
cmake/HPX_SetupHIP.cmake:  if(HPX_WITH_CUDA)
cmake/HPX_SetupHIP.cmake:      "Both HPX_WITH_CUDA and HPX_WITH_HIP are ON. Please choose one of \
cmake/HPX_SetupHIP.cmake:  endif(HPX_WITH_CUDA)
cmake/HPX_SetupHIP.cmake:    set(HPX_WITH_GPUBLAS OFF)
cmake/HPX_SetupHIP.cmake:    set(HPX_WITH_GPUBLAS ON)
cmake/HPX_SetupHIP.cmake:    hpx_add_config_define(HPX_HAVE_GPUBLAS)
cmake/HPX_AddExecutable.cmake:      CUDA
cmake/HPX_AddExecutable.cmake:  if(${name}_CUDA)
cmake/HPX_AddExecutable.cmake:    set_target_properties(${name} PROPERTIES LANGUAGE CUDA)
cmake/HPX_SetupCUDA.cmake:if(HPX_WITH_CUDA AND NOT TARGET Cuda::cuda)
cmake/HPX_SetupCUDA.cmake:    set(HPX_WITH_CLANG_CUDA ON)
cmake/HPX_SetupCUDA.cmake:  # cuda_std_17 not recognized for previous versions
cmake/HPX_SetupCUDA.cmake:  # Check CUDA standard
cmake/HPX_SetupCUDA.cmake:  if(NOT DEFINED CMAKE_CUDA_STANDARD)
cmake/HPX_SetupCUDA.cmake:      set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
cmake/HPX_SetupCUDA.cmake:      set(CMAKE_CUDA_STANDARD 17)
cmake/HPX_SetupCUDA.cmake:    if(CMAKE_CUDA_STANDARD LESS 17)
cmake/HPX_SetupCUDA.cmake:        "You've set CMAKE_CUDA_STANDARD to ${CMAKE_CUDA_STANDARD}, which is less than 17 (the minimum required by HPX)"
cmake/HPX_SetupCUDA.cmake:  set(CMAKE_CUDA_EXTENSIONS OFF)
cmake/HPX_SetupCUDA.cmake:  enable_language(CUDA)
cmake/HPX_SetupCUDA.cmake:    hpx_add_config_define(HPX_HAVE_CUDA)
cmake/HPX_SetupCUDA.cmake:  # CUDA libraries used
cmake/HPX_SetupCUDA.cmake:  add_library(Cuda::cuda INTERFACE IMPORTED)
cmake/HPX_SetupCUDA.cmake:  # Toolkit targets like CUDA::cudart, CUDA::cublas, CUDA::cufft, etc. available
cmake/HPX_SetupCUDA.cmake:  find_package(CUDAToolkit MODULE REQUIRED)
cmake/HPX_SetupCUDA.cmake:  if(CUDAToolkit_FOUND)
cmake/HPX_SetupCUDA.cmake:    target_link_libraries(Cuda::cuda INTERFACE CUDA::cudart)
cmake/HPX_SetupCUDA.cmake:    if(TARGET CUDA::cublas)
cmake/HPX_SetupCUDA.cmake:      set(HPX_WITH_GPUBLAS ON)
cmake/HPX_SetupCUDA.cmake:      hpx_add_config_define(HPX_HAVE_GPUBLAS)
cmake/HPX_SetupCUDA.cmake:      target_link_libraries(Cuda::cuda INTERFACE CUDA::cublas)
cmake/HPX_SetupCUDA.cmake:      set(HPX_WITH_GPUBLAS OFF)
cmake/HPX_SetupCUDA.cmake:  # Flag not working for CLANG CUDA
cmake/HPX_SetupCUDA.cmake:  target_compile_features(Cuda::cuda INTERFACE cuda_std_${CMAKE_CUDA_STANDARD})
cmake/HPX_SetupCUDA.cmake:    Cuda::cuda PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE ON
cmake/HPX_SetupCUDA.cmake:  if(NOT HPX_WITH_CLANG_CUDA)
cmake/HPX_SetupCUDA.cmake:        Cuda::cuda INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-w>
cmake/HPX_SetupCUDA.cmake:      set(CUDA_PROPAGATE_HOST_FLAGS OFF)
cmake/HPX_SetupCUDA.cmake:        Cuda::cuda
cmake/HPX_SetupCUDA.cmake:        INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:
cmake/HPX_SetupCUDA.cmake:        Cuda::cuda
cmake/HPX_SetupCUDA.cmake:        INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:RelWithDebInfo>:
cmake/HPX_SetupCUDA.cmake:        Cuda::cuda
cmake/HPX_SetupCUDA.cmake:        INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:MinSizeRel>: -DNDEBUG
cmake/HPX_SetupCUDA.cmake:        Cuda::cuda
cmake/HPX_SetupCUDA.cmake:        INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>: -DNDEBUG -O2
cmake/HPX_SetupCUDA.cmake:    set(CUDA_SEPARABLE_COMPILATION ON)
cmake/HPX_SetupCUDA.cmake:      Cuda::cuda
cmake/HPX_SetupCUDA.cmake:      INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: --extended-lambda --default-stream
cmake/HPX_SetupCUDA.cmake:    if(${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVIDIA")
cmake/HPX_SetupCUDA.cmake:        Cuda::cuda INTERFACE $<$<COMPILE_LANGUAGE:CUDA>: ASIO_DISABLE_CONSTEXPR
cmake/HPX_SetupCUDA.cmake:    target_link_libraries(hpx_base_libraries INTERFACE Cuda::cuda)
cmake/HPX_AddConfigTest.cmake:  set(options FILE EXECUTE CUDA)
cmake/HPX_AddConfigTest.cmake:  if(NOT DEFINED ${variable} AND NOT ${variable}_CUDA)
cmake/HPX_AddConfigTest.cmake:      if(${variable}_CUDA)
cmake/HPX_AddConfigTest.cmake:      if(HPX_WITH_CUDA)
cmake/HPX_AddConfigTest.cmake:        set(cuda_parameters CUDA_STANDARD ${CMAKE_CUDA_STANDARD})
cmake/HPX_AddConfigTest.cmake:      set(CMAKE_CUDA_FLAGS
cmake/HPX_AddConfigTest.cmake:          "${CMAKE_CUDA_FLAGS} ${additional_cmake_flags} ${${variable}_CXXFLAGS}"
cmake/HPX_AddConfigTest.cmake:        ${cuda_parameters}
cmake/templates/HPXConfig.cmake.in:# CUDA
cmake/templates/HPXConfig.cmake.in:include(HPX_SetupCUDA)
cmake/templates/conf.py.in:.. |nvidia| replace:: NVIDIA
cmake/templates/conf.py.in:.. _nvidia: https://nvidia.com/
cmake/templates/conf.py.in:.. |opencl| replace:: OpenCL
cmake/templates/conf.py.in:.. _opencl: https://www.khronos.org/opencl/
cmake/templates/conf.py.in:.. |cuda| replace:: CUDA
cmake/templates/conf.py.in:.. _cuda: https://www.nvidia.com/object/cuda_home_new.html
cmake/HPX_GeneratePackageUtils.cmake:function(hpx_filter_cuda_flags cflag_list)
cmake/HPX_GeneratePackageUtils.cmake:  string(REGEX REPLACE "\\$<\\$<COMPILE_LANGUAGE:CUDA>:[^>]*>;?" "" _cflag_list
cmake/HPX_GeneratePackageUtils.cmake:endfunction(hpx_filter_cuda_flags)
cmake/HPX_GeneratePackageUtils.cmake:  # Cannot generate one file per language yet so filter out cuda
cmake/HPX_GeneratePackageUtils.cmake:  hpx_filter_cuda_flags(hpx_cflags_list)
cmake/HPX_SetupStdexec.cmake:        GIT_REPOSITORY https://github.com/NVIDIA/stdexec.git
examples/spell_check/5desk.txt:barracuda
examples/spell_check/5desk.txt:Nagpur

```
