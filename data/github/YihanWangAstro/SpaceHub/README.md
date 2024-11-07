# https://github.com/YihanWangAstro/SpaceHub

```console
test/catch.hpp:#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC) && !defined(__CUDACC__)
test/catch.hpp:#  if !defined(__ibmxl__) && !defined(__CUDACC__)
test/catch.hpp:        // Code accompanying the article "Approximating the erfinv function" in GPU Computing Gems, Volume 2
src/taskflow/core/task.hpp:  CUDAFLOW,
src/taskflow/core/task.hpp:  TaskType::CUDAFLOW,
src/taskflow/core/task.hpp:    case TaskType::CUDAFLOW:    val = "cudaflow";    break;
src/taskflow/core/task.hpp:@brief determines if a callable is a cudaflow task
src/taskflow/core/task.hpp:A cudaFlow task is a callable object constructible from 
src/taskflow/core/task.hpp:std::function<void(tf::cudaFlow&)> or std::function<void(tf::cudaFlowCapturer&)>.
src/taskflow/core/task.hpp:constexpr bool is_cudaflow_task_v = std::is_invocable_r_v<void, C, cudaFlow&> ||
src/taskflow/core/task.hpp:                                    std::is_invocable_r_v<void, C, cudaFlowCapturer&>;
src/taskflow/core/task.hpp:    @param callable callable to construct one of the static, dynamic, condition, and cudaFlow tasks
src/taskflow/core/task.hpp:    case Node::CUDAFLOW:     return TaskType::CUDAFLOW;
src/taskflow/core/task.hpp:  else if constexpr(is_cudaflow_task_v<C>) {
src/taskflow/core/task.hpp:    _node->_handle.emplace<Node::cudaFlow>(std::forward<C>(c));
src/taskflow/core/task.hpp:@brief overload of ostream inserter operator for cudaTask
src/taskflow/core/task.hpp:    case Node::CUDAFLOW:     return TaskType::CUDAFLOW;
src/taskflow/core/declarations.hpp:// cudaFlow
src/taskflow/core/declarations.hpp:class cudaNode;
src/taskflow/core/declarations.hpp:class cudaGraph;
src/taskflow/core/declarations.hpp:class cudaTask;
src/taskflow/core/declarations.hpp:class cudaFlow;
src/taskflow/core/declarations.hpp:class cudaFlowCapturer;
src/taskflow/core/declarations.hpp:class cudaFlowCapturerBase;
src/taskflow/core/declarations.hpp:class cudaCapturingBase;
src/taskflow/core/declarations.hpp:class cudaSequentialCapturing;
src/taskflow/core/declarations.hpp:class cudaRoundRobinCapturing;
src/taskflow/core/executor.hpp:  friend class cudaFlow;
src/taskflow/core/executor.hpp:    void _invoke_cudaflow_task(Worker&, Node*);
src/taskflow/core/executor.hpp:      std::is_invocable_r_v<void, C, cudaFlow&>, void>* = nullptr
src/taskflow/core/executor.hpp:    void _invoke_cudaflow_task_entry(C&&, Node*);
src/taskflow/core/executor.hpp:      std::is_invocable_r_v<void, C, cudaFlowCapturer&>, void>* = nullptr
src/taskflow/core/executor.hpp:    void _invoke_cudaflow_task_entry(C&&, Node*);
src/taskflow/core/executor.hpp:    //void _invoke_cudaflow_task_internal(cudaFlow&, P&&, bool);
src/taskflow/core/executor.hpp:    //void _invoke_cudaflow_task_external(cudaFlow&, P&&, bool);
src/taskflow/core/executor.hpp:    // cudaflow task
src/taskflow/core/executor.hpp:    case Node::CUDAFLOW: {
src/taskflow/core/executor.hpp:      _invoke_cudaflow_task(worker, node);
src/taskflow/core/executor.hpp:// Procedure: _invoke_cudaflow_task
src/taskflow/core/executor.hpp:inline void Executor::_invoke_cudaflow_task(Worker& worker, Node* node) {
src/taskflow/core/executor.hpp:  std::get<Node::cudaFlow>(node->_handle).work(*this, node);
src/taskflow/core/taskflow.hpp:  5. %cudaFlow task: the callable constructible from 
src/taskflow/core/taskflow.hpp:                     @c std::function<void(tf::cudaFlow&)> or
src/taskflow/core/taskflow.hpp:                     @c std::function<void(tf::cudaFlowCapturer&)>
src/taskflow/core/taskflow.hpp:    case Node::CUDAFLOW:
src/taskflow/core/taskflow.hpp:    case Node::CUDAFLOW: {
src/taskflow/core/taskflow.hpp:      std::get<Node::cudaFlow>(node->_handle).graph->dump(
src/taskflow/core/graph.hpp:  // cudaFlow work handle
src/taskflow/core/graph.hpp:  struct cudaFlow {
src/taskflow/core/graph.hpp:    cudaFlow(C&& c, G&& g);
src/taskflow/core/graph.hpp:    cudaFlow         // cudaFlow
src/taskflow/core/graph.hpp:  constexpr static auto CUDAFLOW     = get_index_v<cudaFlow, handle_t>; 
src/taskflow/core/graph.hpp:// Definition for Node::cudaFlow
src/taskflow/core/graph.hpp:Node::cudaFlow::cudaFlow(C&& c, G&& g) :
src/taskflow/core/flow_builder.hpp:    @brief creates a %cudaFlow task on the caller's GPU device context
src/taskflow/core/flow_builder.hpp:    @tparam C callable type constructible from @c std::function<void(tf::cudaFlow&)>
src/taskflow/core/flow_builder.hpp:    The following example creates a %cudaFlow of two kernel tasks, @c task1 and 
src/taskflow/core/flow_builder.hpp:    taskflow.emplace([&](tf::cudaFlow& cf){
src/taskflow/core/flow_builder.hpp:      tf::cudaTask task1 = cf.kernel(grid1, block1, shm1, kernel1, args1);
src/taskflow/core/flow_builder.hpp:      tf::cudaTask task2 = cf.kernel(grid2, block2, shm2, kernel2, args2);
src/taskflow/core/flow_builder.hpp:    Please refer to @ref GPUTaskingcudaFlow and @ref GPUTaskingcudaFlowCapturer 
src/taskflow/core/flow_builder.hpp:      std::enable_if_t<is_cudaflow_task_v<C>, void>* = nullptr
src/taskflow/core/flow_builder.hpp:    @brief creates a %cudaFlow task on the given device
src/taskflow/core/flow_builder.hpp:    @tparam C callable type constructible from std::function<void(tf::cudaFlow&)>
src/taskflow/core/flow_builder.hpp:    The following example creates a %cudaFlow of two kernel tasks, @c task1 and 
src/taskflow/core/flow_builder.hpp:    @c task2 on GPU @c 2, where @c task1 runs before @c task2
src/taskflow/core/flow_builder.hpp:    taskflow.emplace_on([&](tf::cudaFlow& cf){
src/taskflow/core/flow_builder.hpp:      tf::cudaTask task1 = cf.kernel(grid1, block1, shm1, kernel1, args1);
src/taskflow/core/flow_builder.hpp:      tf::cudaTask task2 = cf.kernel(grid2, block2, shm2, kernel2, args2);
src/taskflow/core/flow_builder.hpp:      std::enable_if_t<is_cudaflow_task_v<C>, void>* = nullptr
src/taskflow/README.md:by harnessing the power of CPU-GPU collaborative computing.
src/taskflow/README.md:| [Concurrent CPU-GPU Tasking](#concurrent-cpu-gpu-tasking) |
src/taskflow/README.md:| ![](image/cudaflow.svg) |
src/taskflow/README.md:+ Nvidia CUDA Toolkit and Compiler (nvcc) at least v11.1 with -std=c++17
src/taskflow/README.md:[cuda-zone]:             https://developer.nvidia.com/cuda-zone
src/taskflow/README.md:[nvcc]:                  https://developer.nvidia.com/cuda-llvm-compiler
src/taskflow/README.md:[cudaGraph]:             https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html
src/taskflow/cublasflow.hpp:// cudaflow.hpp
src/taskflow/cublasflow.hpp:#include "cudaflow.hpp"
src/taskflow/cublasflow.hpp:#include "cuda/cublas/cublas_flow.hpp"
src/taskflow/cublasflow.hpp:#include "cuda/cublas/cublas_helper.hpp"
src/taskflow/cublasflow.hpp:#include "cuda/cublas/cublas_level1.hpp"
src/taskflow/cublasflow.hpp:#include "cuda/cublas/cublas_level2.hpp"
src/taskflow/cublasflow.hpp:#include "cuda/cublas/cublas_level3.hpp"
src/taskflow/taskflow.hpp:/** @dir taskflow/cuda
src/taskflow/taskflow.hpp:@brief taskflow CUDA include dir
src/taskflow/taskflow.hpp:/** @dir taskflow/cuda/cublas
src/taskflow/cudaflow.hpp:// cudaflow.hpp
src/taskflow/cudaflow.hpp:#include "cuda/cuda_flow.hpp"
src/taskflow/cudaflow.hpp:@file cudaflow.hpp
src/taskflow/cudaflow.hpp:@brief main cudaFlow include file
src/taskflow/cuda/cuda_error.hpp:#include <cuda.h>
src/taskflow/cuda/cuda_error.hpp:#define TF_CUDA_REMOVE_FIRST_HELPER(N, ...) __VA_ARGS__
src/taskflow/cuda/cuda_error.hpp:#define TF_CUDA_REMOVE_FIRST(...) TF_CUDA_REMOVE_FIRST_HELPER(__VA_ARGS__)
src/taskflow/cuda/cuda_error.hpp:#define TF_CUDA_GET_FIRST_HELPER(N, ...) N
src/taskflow/cuda/cuda_error.hpp:#define TF_CUDA_GET_FIRST(...) TF_CUDA_GET_FIRST_HELPER(__VA_ARGS__)
src/taskflow/cuda/cuda_error.hpp:#define TF_CHECK_CUDA(...)                                       \
src/taskflow/cuda/cuda_error.hpp:if(TF_CUDA_GET_FIRST(__VA_ARGS__) != cudaSuccess) {              \
src/taskflow/cuda/cuda_error.hpp:  auto ev = TF_CUDA_GET_FIRST(__VA_ARGS__);                      \
src/taskflow/cuda/cuda_error.hpp:  auto unknown_name = "cudaErrorUnknown";                        \
src/taskflow/cuda/cuda_error.hpp:  auto error_str  = ::cudaGetErrorString(ev);                    \
src/taskflow/cuda/cuda_error.hpp:  auto error_name = ::cudaGetErrorName(ev);                      \
src/taskflow/cuda/cuda_error.hpp:  tf::ostreamize(oss, TF_CUDA_REMOVE_FIRST(__VA_ARGS__));  \
src/taskflow/cuda/cuda_flow.hpp:#include "cuda_task.hpp"
src/taskflow/cuda/cuda_flow.hpp:#include "cuda_capturer.hpp"
src/taskflow/cuda/cuda_flow.hpp:#include "cuda_optimizer.hpp"
src/taskflow/cuda/cuda_flow.hpp:#include "cuda_algorithm/cuda_for_each.hpp"
src/taskflow/cuda/cuda_flow.hpp:#include "cuda_algorithm/cuda_transform.hpp"
src/taskflow/cuda/cuda_flow.hpp:#include "cuda_algorithm/cuda_reduce.hpp"
src/taskflow/cuda/cuda_flow.hpp:@file cuda_flow.hpp
src/taskflow/cuda/cuda_flow.hpp:@brief cudaFlow include file
src/taskflow/cuda/cuda_flow.hpp:// class definition: cudaFlow
src/taskflow/cuda/cuda_flow.hpp:@class cudaFlow
src/taskflow/cuda/cuda_flow.hpp:@brief class for building a CUDA task dependency graph
src/taskflow/cuda/cuda_flow.hpp:A %cudaFlow is a high-level interface over CUDA Graph to perform GPU operations 
src/taskflow/cuda/cuda_flow.hpp:on one or multiple CUDA devices,
src/taskflow/cuda/cuda_flow.hpp:The following example creates a %cudaFlow of two kernel tasks, @c task1 and 
src/taskflow/cuda/cuda_flow.hpp:taskflow.emplace([&](tf::cudaFlow& cf){
src/taskflow/cuda/cuda_flow.hpp:  tf::cudaTask task1 = cf.kernel(grid1, block1, shm_size1, kernel1, args1);
src/taskflow/cuda/cuda_flow.hpp:  tf::cudaTask task2 = cf.kernel(grid2, block2, shm_size2, kernel2, args2);
src/taskflow/cuda/cuda_flow.hpp:A %cudaFlow is a task (tf::Task) created from tf::Taskflow 
src/taskflow/cuda/cuda_flow.hpp:That is, the callable that describes a %cudaFlow 
src/taskflow/cuda/cuda_flow.hpp:Inside a %cudaFlow task, different GPU tasks (tf::cudaTask) may run
src/taskflow/cuda/cuda_flow.hpp:in parallel scheduled by the CUDA runtime.
src/taskflow/cuda/cuda_flow.hpp:Please refer to @ref GPUTaskingcudaFlow for details.
src/taskflow/cuda/cuda_flow.hpp:class cudaFlow {
src/taskflow/cuda/cuda_flow.hpp:    cudaGraph graph;
src/taskflow/cuda/cuda_flow.hpp:    @brief constructs a standalone %cudaFlow
src/taskflow/cuda/cuda_flow.hpp:    A standalone %cudaFlow does not go through any taskflow and
src/taskflow/cuda/cuda_flow.hpp:    (e.g., tf::cudaFlow::offload).
src/taskflow/cuda/cuda_flow.hpp:    cudaFlow();
src/taskflow/cuda/cuda_flow.hpp:    @brief destroys the %cudaFlow and its associated native CUDA graph
src/taskflow/cuda/cuda_flow.hpp:    ~cudaFlow();
src/taskflow/cuda/cuda_flow.hpp:    @brief dumps the %cudaFlow graph into a DOT format through an
src/taskflow/cuda/cuda_flow.hpp:    @brief dumps the native CUDA graph into a DOT format through an
src/taskflow/cuda/cuda_flow.hpp:    The native CUDA graph may be different from the upper-level %cudaFlow 
src/taskflow/cuda/cuda_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cuda_flow.hpp:    cudaTask noop();
src/taskflow/cuda/cuda_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cuda_flow.hpp:    A host task can only execute CPU-specific functions and cannot do any CUDA calls 
src/taskflow/cuda/cuda_flow.hpp:    (e.g., @c cudaMalloc).
src/taskflow/cuda/cuda_flow.hpp:    cudaTask host(C&& callable);
src/taskflow/cuda/cuda_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cuda_flow.hpp:    cudaTask kernel(dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args);
src/taskflow/cuda/cuda_flow.hpp:    @brief creates a kernel task on a specific GPU
src/taskflow/cuda/cuda_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cuda_flow.hpp:    cudaTask kernel_on(int d, dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args);
src/taskflow/cuda/cuda_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cuda_flow.hpp:    cudaTask memset(void* dst, int v, size_t count);
src/taskflow/cuda/cuda_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cuda_flow.hpp:    to a target location. Direction can be arbitrary among CPUs and GPUs.
src/taskflow/cuda/cuda_flow.hpp:    cudaTask memcpy(void* tgt, const void* src, size_t bytes);
src/taskflow/cuda/cuda_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cuda_flow.hpp:    cudaTask zero(T* dst, size_t count);
src/taskflow/cuda/cuda_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cuda_flow.hpp:    cudaTask fill(T* dst, T value, size_t count);
src/taskflow/cuda/cuda_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cuda_flow.hpp:    to a target location. Direction can be arbitrary among CPUs and GPUs.
src/taskflow/cuda/cuda_flow.hpp:    cudaTask copy(T* tgt, const T* src, size_t num);
src/taskflow/cuda/cuda_flow.hpp:    @brief offloads the %cudaFlow onto a GPU and repeatedly runs it until 
src/taskflow/cuda/cuda_flow.hpp:    Immediately offloads the present %cudaFlow onto a GPU and
src/taskflow/cuda/cuda_flow.hpp:    An offloaded %cudaFlow forces the underlying graph to be instantiated.
src/taskflow/cuda/cuda_flow.hpp:    By default, if users do not offload the %cudaFlow, 
src/taskflow/cuda/cuda_flow.hpp:    @brief offloads the %cudaFlow and executes it by the given times
src/taskflow/cuda/cuda_flow.hpp:    @brief offloads the %cudaFlow and executes it once
src/taskflow/cuda/cuda_flow.hpp:    @brief updates parameters of a host task created from tf::cudaFlow::host
src/taskflow/cuda/cuda_flow.hpp:    void update_host(cudaTask task, C&& callable);
src/taskflow/cuda/cuda_flow.hpp:    @brief updates parameters of a kernel task created from tf::cudaFlow::kernel
src/taskflow/cuda/cuda_flow.hpp:    void update_kernel(cudaTask task, dim3 g, dim3 b, size_t shm, ArgsT&&... args);
src/taskflow/cuda/cuda_flow.hpp:    void update_copy(cudaTask task, T* tgt, const T* src, size_t num);
src/taskflow/cuda/cuda_flow.hpp:    void update_memcpy(cudaTask task, void* tgt, const void* src, size_t bytes);
src/taskflow/cuda/cuda_flow.hpp:    void update_memset(cudaTask task, void* dst, int ch, size_t count);
src/taskflow/cuda/cuda_flow.hpp:    The given arguments and type must comply with the rules of tf::cudaFlow::fill.
src/taskflow/cuda/cuda_flow.hpp:    void update_fill(cudaTask task, T* dst, T value, size_t count);
src/taskflow/cuda/cuda_flow.hpp:    The given arguments and type must comply with the rules of tf::cudaFlow::zero.
src/taskflow/cuda/cuda_flow.hpp:    void update_zero(cudaTask task, T* dst, size_t count);
src/taskflow/cuda/cuda_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cuda_flow.hpp:    cudaTask single_task(C&& callable);
src/taskflow/cuda/cuda_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cuda_flow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
src/taskflow/cuda/cuda_flow.hpp:    cudaTask for_each(I first, I last, C&& callable);
src/taskflow/cuda/cuda_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cuda_flow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
src/taskflow/cuda/cuda_flow.hpp:    cudaTask for_each_index(I first, I last, I step, C&& callable);
src/taskflow/cuda/cuda_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cuda_flow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
src/taskflow/cuda/cuda_flow.hpp:    cudaTask transform(I first, I last, C&& callable, S... srcs);
src/taskflow/cuda/cuda_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cuda_flow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
src/taskflow/cuda/cuda_flow.hpp:    cudaTask reduce(I first, I last, T* result, C&& op);
src/taskflow/cuda/cuda_flow.hpp:    @brief similar to tf::cudaFlow::reduce but does not assume any initial
src/taskflow/cuda/cuda_flow.hpp:    on a GPU:
src/taskflow/cuda/cuda_flow.hpp:    cudaTask uninitialized_reduce(I first, I last, T* result, C&& op);
src/taskflow/cuda/cuda_flow.hpp:    @brief constructs a subflow graph through tf::cudaFlowCapturer
src/taskflow/cuda/cuda_flow.hpp:              @c std::function<void(tf::cudaFlowCapturer&)>
src/taskflow/cuda/cuda_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cuda_flow.hpp:    A captured subflow forms a sub-graph to the %cudaFlow and can be used to 
src/taskflow/cuda/cuda_flow.hpp:    from the %cudaFlow.
src/taskflow/cuda/cuda_flow.hpp:    taskflow.emplace([&](tf::cudaFlow& cf){
src/taskflow/cuda/cuda_flow.hpp:      tf::cudaTask my_kernel = cf.kernel(my_arguments);
src/taskflow/cuda/cuda_flow.hpp:      tf::cudaTask my_subflow = cf.capture([&](tf::cudaFlowCapturer& capturer){
src/taskflow/cuda/cuda_flow.hpp:        capturer.on([&](cudaStream_t stream){
src/taskflow/cuda/cuda_flow.hpp:    cudaTask capture(C&& callable);
src/taskflow/cuda/cuda_flow.hpp:    cudaGraph& _graph;
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphExec_t _executable {nullptr};
src/taskflow/cuda/cuda_flow.hpp:    cudaFlow(Executor&, cudaGraph&);
src/taskflow/cuda/cuda_flow.hpp:// Construct a standalone cudaFlow
src/taskflow/cuda/cuda_flow.hpp:inline cudaFlow::cudaFlow() :
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphCreate(&_graph._native_handle, 0), 
src/taskflow/cuda/cuda_flow.hpp:    "cudaFlow failed to create a native graph (external mode)"
src/taskflow/cuda/cuda_flow.hpp:// Construct the cudaFlow from executor (internal graph)
src/taskflow/cuda/cuda_flow.hpp:inline cudaFlow::cudaFlow(Executor& e, cudaGraph& g) :
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphCreate(&_graph._native_handle, 0), 
src/taskflow/cuda/cuda_flow.hpp:    "cudaFlow failed to create a native graph (internal mode)"
src/taskflow/cuda/cuda_flow.hpp:inline cudaFlow::~cudaFlow() {
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphExecDestroy(_executable);
src/taskflow/cuda/cuda_flow.hpp:  cudaGraphDestroy(_graph._native_handle);
src/taskflow/cuda/cuda_flow.hpp:inline bool cudaFlow::empty() const {
src/taskflow/cuda/cuda_flow.hpp:inline void cudaFlow::dump(std::ostream& os) const {
src/taskflow/cuda/cuda_flow.hpp:inline void cudaFlow::dump_native_graph(std::ostream& os) const {
src/taskflow/cuda/cuda_flow.hpp:  cuda_dump_graph(os, _graph._native_handle);
src/taskflow/cuda/cuda_flow.hpp:inline cudaTask cudaFlow::noop() {
src/taskflow/cuda/cuda_flow.hpp:    _graph, std::in_place_type_t<cudaNode::Empty>{}
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphAddEmptyNode(
src/taskflow/cuda/cuda_flow.hpp:  return cudaTask(node);
src/taskflow/cuda/cuda_flow.hpp:cudaTask cudaFlow::host(C&& c) {
src/taskflow/cuda/cuda_flow.hpp:    _graph, std::in_place_type_t<cudaNode::Host>{}, std::forward<C>(c)
src/taskflow/cuda/cuda_flow.hpp:  auto& h = std::get<cudaNode::Host>(node->_handle);
src/taskflow/cuda/cuda_flow.hpp:  cudaHostNodeParams p;
src/taskflow/cuda/cuda_flow.hpp:  p.fn = cudaNode::Host::callback;
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphAddHostNode(
src/taskflow/cuda/cuda_flow.hpp:  return cudaTask(node);
src/taskflow/cuda/cuda_flow.hpp:cudaTask cudaFlow::kernel(
src/taskflow/cuda/cuda_flow.hpp:    _graph, std::in_place_type_t<cudaNode::Kernel>{}, (void*)f
src/taskflow/cuda/cuda_flow.hpp:  cudaKernelNodeParams p;
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphAddKernelNode(
src/taskflow/cuda/cuda_flow.hpp:  return cudaTask(node);
src/taskflow/cuda/cuda_flow.hpp:cudaTask cudaFlow::kernel_on(
src/taskflow/cuda/cuda_flow.hpp:    _graph, std::in_place_type_t<cudaNode::Kernel>{}, (void*)f
src/taskflow/cuda/cuda_flow.hpp:  cudaKernelNodeParams p;
src/taskflow/cuda/cuda_flow.hpp:  cudaScopedDevice ctx(d);
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphAddKernelNode(
src/taskflow/cuda/cuda_flow.hpp:  return cudaTask(node);
src/taskflow/cuda/cuda_flow.hpp:cudaTask cudaFlow::zero(T* dst, size_t count) {
src/taskflow/cuda/cuda_flow.hpp:    _graph, std::in_place_type_t<cudaNode::Memset>{}
src/taskflow/cuda/cuda_flow.hpp:  auto p = cuda_get_zero_parms(dst, count);
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphAddMemsetNode(
src/taskflow/cuda/cuda_flow.hpp:  return cudaTask(node);
src/taskflow/cuda/cuda_flow.hpp:cudaTask cudaFlow::fill(T* dst, T value, size_t count) {
src/taskflow/cuda/cuda_flow.hpp:    _graph, std::in_place_type_t<cudaNode::Memset>{}
src/taskflow/cuda/cuda_flow.hpp:  auto p = cuda_get_fill_parms(dst, value, count);
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphAddMemsetNode(
src/taskflow/cuda/cuda_flow.hpp:  return cudaTask(node);
src/taskflow/cuda/cuda_flow.hpp:cudaTask cudaFlow::copy(T* tgt, const T* src, size_t num) {
src/taskflow/cuda/cuda_flow.hpp:    _graph, std::in_place_type_t<cudaNode::Memcpy>{}
src/taskflow/cuda/cuda_flow.hpp:  auto p = cuda_get_copy_parms(tgt, src, num);
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphAddMemcpyNode(
src/taskflow/cuda/cuda_flow.hpp:  return cudaTask(node);
src/taskflow/cuda/cuda_flow.hpp:inline cudaTask cudaFlow::memset(void* dst, int ch, size_t count) {
src/taskflow/cuda/cuda_flow.hpp:    _graph, std::in_place_type_t<cudaNode::Memset>{}
src/taskflow/cuda/cuda_flow.hpp:  auto p = cuda_get_memset_parms(dst, ch, count);
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphAddMemsetNode(
src/taskflow/cuda/cuda_flow.hpp:  return cudaTask(node);
src/taskflow/cuda/cuda_flow.hpp:inline cudaTask cudaFlow::memcpy(void* tgt, const void* src, size_t bytes) {
src/taskflow/cuda/cuda_flow.hpp:    _graph, std::in_place_type_t<cudaNode::Memcpy>{}
src/taskflow/cuda/cuda_flow.hpp:  auto p = cuda_get_memcpy_parms(tgt, src, bytes);
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphAddMemcpyNode(
src/taskflow/cuda/cuda_flow.hpp:  return cudaTask(node);
src/taskflow/cuda/cuda_flow.hpp:void cudaFlow::update_host(cudaTask task, C&& c) {
src/taskflow/cuda/cuda_flow.hpp:  if(task.type() != cudaTaskType::HOST) {
src/taskflow/cuda/cuda_flow.hpp:  auto& h = std::get<cudaNode::Host>(task._node->_handle);
src/taskflow/cuda/cuda_flow.hpp:void cudaFlow::update_kernel(
src/taskflow/cuda/cuda_flow.hpp:  cudaTask ct, dim3 g, dim3 b, size_t s, ArgsT&&... args
src/taskflow/cuda/cuda_flow.hpp:  if(ct.type() != cudaTaskType::KERNEL) {
src/taskflow/cuda/cuda_flow.hpp:  cudaKernelNodeParams p;
src/taskflow/cuda/cuda_flow.hpp:  p.func = std::get<cudaNode::Kernel>((ct._node)->_handle).func;
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphExecKernelNodeSetParams(
src/taskflow/cuda/cuda_flow.hpp:void cudaFlow::update_copy(cudaTask ct, T* tgt, const T* src, size_t num) {
src/taskflow/cuda/cuda_flow.hpp:  if(ct.type() != cudaTaskType::MEMCPY) {
src/taskflow/cuda/cuda_flow.hpp:  auto p = cuda_get_copy_parms(tgt, src, num);
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphExecMemcpyNodeSetParams(
src/taskflow/cuda/cuda_flow.hpp:inline void cudaFlow::update_memcpy(
src/taskflow/cuda/cuda_flow.hpp:  cudaTask ct, void* tgt, const void* src, size_t bytes
src/taskflow/cuda/cuda_flow.hpp:  if(ct.type() != cudaTaskType::MEMCPY) {
src/taskflow/cuda/cuda_flow.hpp:  auto p = cuda_get_memcpy_parms(tgt, src, bytes);
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphExecMemcpyNodeSetParams(_executable, ct._node->_native_handle, &p),
src/taskflow/cuda/cuda_flow.hpp:void cudaFlow::update_memset(cudaTask ct, void* dst, int ch, size_t count) {
src/taskflow/cuda/cuda_flow.hpp:  if(ct.type() != cudaTaskType::MEMSET) {
src/taskflow/cuda/cuda_flow.hpp:  auto p = cuda_get_memset_parms(dst, ch, count);
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphExecMemsetNodeSetParams(
src/taskflow/cuda/cuda_flow.hpp:void cudaFlow::update_fill(cudaTask task, T* dst, T value, size_t count) {
src/taskflow/cuda/cuda_flow.hpp:  if(task.type() != cudaTaskType::MEMSET) {
src/taskflow/cuda/cuda_flow.hpp:  auto p = cuda_get_fill_parms(dst, value, count);
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphExecMemsetNodeSetParams(
src/taskflow/cuda/cuda_flow.hpp:void cudaFlow::update_zero(cudaTask task, T* dst, size_t count) {
src/taskflow/cuda/cuda_flow.hpp:  if(task.type() != cudaTaskType::MEMSET) {
src/taskflow/cuda/cuda_flow.hpp:  auto p = cuda_get_zero_parms(dst, count);
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphExecMemsetNodeSetParams(
src/taskflow/cuda/cuda_flow.hpp:cudaTask cudaFlow::single_task(C&& c) {
src/taskflow/cuda/cuda_flow.hpp:    1, 1, 0, cuda_single_task<C>, std::forward<C>(c)
src/taskflow/cuda/cuda_flow.hpp:cudaTask cudaFlow::for_each(I first, I last, C&& c) {
src/taskflow/cuda/cuda_flow.hpp:  size_t B = cuda_default_threads_per_block(N);
src/taskflow/cuda/cuda_flow.hpp:    (N+B-1) / B, B, 0, cuda_for_each<I, C>, first, N, std::forward<C>(c)
src/taskflow/cuda/cuda_flow.hpp:cudaTask cudaFlow::for_each_index(I beg, I end, I inc, C&& c) {
src/taskflow/cuda/cuda_flow.hpp:  size_t B = cuda_default_threads_per_block(N);
src/taskflow/cuda/cuda_flow.hpp:    (N+B-1) / B, B, 0, cuda_for_each_index<I, C>, beg, inc, N, std::forward<C>(c)
src/taskflow/cuda/cuda_flow.hpp:cudaTask cudaFlow::transform(I first, I last, C&& c, S... srcs) {
src/taskflow/cuda/cuda_flow.hpp:  size_t B = cuda_default_threads_per_block(N);
src/taskflow/cuda/cuda_flow.hpp:    (N+B-1) / B, B, 0, cuda_transform<I, C, S...>, 
src/taskflow/cuda/cuda_flow.hpp:cudaTask cudaFlow::reduce(I first, I last, T* result, C&& op) {
src/taskflow/cuda/cuda_flow.hpp:  size_t B = cuda_default_threads_per_block(N);
src/taskflow/cuda/cuda_flow.hpp:    1, B, B*sizeof(T), cuda_reduce<I, T, C, false>, 
src/taskflow/cuda/cuda_flow.hpp:cudaTask cudaFlow::uninitialized_reduce(I first, I last, T* result, C&& op) {
src/taskflow/cuda/cuda_flow.hpp:  size_t B = cuda_default_threads_per_block(N);
src/taskflow/cuda/cuda_flow.hpp:    1, B, B*sizeof(T), cuda_reduce<I, T, C, true>, 
src/taskflow/cuda/cuda_flow.hpp:cudaTask cudaFlow::capture(C&& c) {
src/taskflow/cuda/cuda_flow.hpp:    _graph, std::in_place_type_t<cudaNode::Subflow>{}
src/taskflow/cuda/cuda_flow.hpp:  auto& node_handle = std::get<cudaNode::Subflow>(node->_handle);
src/taskflow/cuda/cuda_flow.hpp:  cudaFlowCapturer capturer(node_handle.graph);
src/taskflow/cuda/cuda_flow.hpp:  //cuda_dump_graph(std::cout, captured);
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:    cudaGraphAddChildGraphNode(
src/taskflow/cuda/cuda_flow.hpp:    "failed to add a cudaFlow capturer task"
src/taskflow/cuda/cuda_flow.hpp:  TF_CHECK_CUDA(cudaGraphDestroy(captured), "failed to destroy captured graph");
src/taskflow/cuda/cuda_flow.hpp:  return cudaTask(node);
src/taskflow/cuda/cuda_flow.hpp:void cudaFlow::offload_until(P&& predicate) {
src/taskflow/cuda/cuda_flow.hpp:  //_executor->_invoke_cudaflow_task_internal(
src/taskflow/cuda/cuda_flow.hpp:  // transforms cudaFlow to a native cudaGraph under the specified device
src/taskflow/cuda/cuda_flow.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:      cudaGraphInstantiate(
src/taskflow/cuda/cuda_flow.hpp:    //cuda_dump_graph(std::cout, cf._graph._native_handle);
src/taskflow/cuda/cuda_flow.hpp:  cudaScopedPerThreadStream s;
src/taskflow/cuda/cuda_flow.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:      cudaGraphLaunch(_executable, s), "failed to execute cudaFlow"
src/taskflow/cuda/cuda_flow.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_flow.hpp:      cudaStreamSynchronize(s), "failed to synchronize cudaFlow execution"
src/taskflow/cuda/cuda_flow.hpp:inline void cudaFlow::offload_n(size_t n) {
src/taskflow/cuda/cuda_flow.hpp:inline void cudaFlow::offload() {
src/taskflow/cuda/cuda_flow.hpp:  std::enable_if_t<is_cudaflow_task_v<C>, void>*
src/taskflow/cuda/cuda_flow.hpp:    std::in_place_type_t<Node::cudaFlow>{},
src/taskflow/cuda/cuda_flow.hpp:      cudaScopedDevice ctx(d);
src/taskflow/cuda/cuda_flow.hpp:      executor._invoke_cudaflow_task_entry(c, node);
src/taskflow/cuda/cuda_flow.hpp:    std::make_unique<cudaGraph>()
src/taskflow/cuda/cuda_flow.hpp:template <typename C, std::enable_if_t<is_cudaflow_task_v<C>, void>*>
src/taskflow/cuda/cuda_flow.hpp:  return emplace_on(std::forward<C>(c), tf::cuda_get_device());
src/taskflow/cuda/cuda_flow.hpp:// Procedure: _invoke_cudaflow_task_entry (cudaFlow)
src/taskflow/cuda/cuda_flow.hpp:  std::enable_if_t<std::is_invocable_r_v<void, C, cudaFlow&>, void>*
src/taskflow/cuda/cuda_flow.hpp:void Executor::_invoke_cudaflow_task_entry(C&& c, Node* node) {
src/taskflow/cuda/cuda_flow.hpp:  auto& h = std::get<Node::cudaFlow>(node->_handle);
src/taskflow/cuda/cuda_flow.hpp:  cudaGraph* g = dynamic_cast<cudaGraph*>(h.graph.get());
src/taskflow/cuda/cuda_flow.hpp:  cudaFlow cf(*this, *g);
src/taskflow/cuda/cuda_flow.hpp:  // join the cudaflow if never offloaded
src/taskflow/cuda/cuda_flow.hpp:// Procedure: _invoke_cudaflow_task_entry (cudaFlowCapturer)
src/taskflow/cuda/cuda_flow.hpp:  std::enable_if_t<std::is_invocable_r_v<void, C, cudaFlowCapturer&>, void>*
src/taskflow/cuda/cuda_flow.hpp:void Executor::_invoke_cudaflow_task_entry(C&& c, Node* node) {
src/taskflow/cuda/cuda_flow.hpp:  auto& h = std::get<Node::cudaFlow>(node->_handle);
src/taskflow/cuda/cuda_flow.hpp:  cudaGraph* g = dynamic_cast<cudaGraph*>(h.graph.get());
src/taskflow/cuda/cuda_flow.hpp:  cudaFlowCapturer fc(*g);
src/taskflow/cuda/cuda_memory.hpp:#include "cuda_device.hpp"
src/taskflow/cuda/cuda_memory.hpp:@file cuda_memory.hpp
src/taskflow/cuda/cuda_memory.hpp:@brief CUDA memory utilities include file
src/taskflow/cuda/cuda_memory.hpp:inline size_t cuda_get_free_mem(int d) {
src/taskflow/cuda/cuda_memory.hpp:  cudaScopedDevice ctx(d);
src/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_memory.hpp:    cudaMemGetInfo(&free, &total), "failed to get mem info on device ", d
src/taskflow/cuda/cuda_memory.hpp:inline size_t cuda_get_total_mem(int d) {
src/taskflow/cuda/cuda_memory.hpp:  cudaScopedDevice ctx(d);
src/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_memory.hpp:    cudaMemGetInfo(&free, &total), "failed to get mem info on device ", d
src/taskflow/cuda/cuda_memory.hpp:The function calls @c cudaMalloc to allocate <tt>N*sizeof(T)</tt> bytes of memory
src/taskflow/cuda/cuda_memory.hpp:T* cuda_malloc_device(size_t N, int d) {
src/taskflow/cuda/cuda_memory.hpp:  cudaScopedDevice ctx(d);
src/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_memory.hpp:    cudaMalloc(&ptr, N*sizeof(T)), 
src/taskflow/cuda/cuda_memory.hpp:The function calls cuda_malloc_device from the current device associated
src/taskflow/cuda/cuda_memory.hpp:T* cuda_malloc_device(size_t N) {
src/taskflow/cuda/cuda_memory.hpp:  return cuda_malloc_device<T>(N, cuda_get_device());
src/taskflow/cuda/cuda_memory.hpp:The function calls @c cudaMallocManaged to allocate <tt>N*sizeof(T)</tt> bytes
src/taskflow/cuda/cuda_memory.hpp:T* cuda_malloc_shared(size_t N) {
src/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_memory.hpp:    cudaMallocManaged(&ptr, N*sizeof(T)), 
src/taskflow/cuda/cuda_memory.hpp:@brief frees memory on the GPU device
src/taskflow/cuda/cuda_memory.hpp:This methods call @c cudaFree to free the memory space pointed to by @c ptr
src/taskflow/cuda/cuda_memory.hpp:void cuda_free(T* ptr, int d) {
src/taskflow/cuda/cuda_memory.hpp:  cudaScopedDevice ctx(d);
src/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(cudaFree(ptr), "failed to free memory ", ptr, " on GPU ", d);
src/taskflow/cuda/cuda_memory.hpp:@brief frees memory on the GPU device
src/taskflow/cuda/cuda_memory.hpp:This methods call @c cudaFree to free the memory space pointed to by @c ptr
src/taskflow/cuda/cuda_memory.hpp:void cuda_free(T* ptr) {
src/taskflow/cuda/cuda_memory.hpp:  cuda_free(ptr, cuda_get_device());
src/taskflow/cuda/cuda_memory.hpp:The method calls @c cudaMemcpyAsync with the given @c stream
src/taskflow/cuda/cuda_memory.hpp:using @c cudaMemcpyDefault to infer the memory space of the source and 
src/taskflow/cuda/cuda_memory.hpp:inline void cuda_memcpy_async(
src/taskflow/cuda/cuda_memory.hpp:  cudaStream_t stream, void* dst, const void* src, size_t count
src/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_memory.hpp:    cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream),
src/taskflow/cuda/cuda_memory.hpp:    "failed to perform cudaMemcpyAsync"
src/taskflow/cuda/cuda_memory.hpp:@brief initializes or sets GPU memory to the given value byte by byte
src/taskflow/cuda/cuda_memory.hpp:@param devPtr pointer to GPU mempry
src/taskflow/cuda/cuda_memory.hpp:The method calls @c cudaMemsetAsync with the given @c stream
src/taskflow/cuda/cuda_memory.hpp:inline void cuda_memset_async(
src/taskflow/cuda/cuda_memory.hpp:  cudaStream_t stream, void* devPtr, int value, size_t count
src/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_memory.hpp:    cudaMemsetAsync(devPtr, value, count, stream),
src/taskflow/cuda/cuda_memory.hpp:    "failed to perform cudaMemsetAsync"
src/taskflow/cuda/cuda_memory.hpp://      cudaSharedMemory<T> smem;
src/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory
src/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <int>
src/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <unsigned int>
src/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <char>
src/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <unsigned char>
src/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <short>
src/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <unsigned short>
src/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <long>
src/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <unsigned long>
src/taskflow/cuda/cuda_memory.hpp://struct cudaSharedMemory <size_t>
src/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <bool>
src/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <float>
src/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <double>
src/taskflow/cuda/cublas/cublas_level1.hpp:cudaTask cublasFlowCapturer::amax(
src/taskflow/cuda/cublas/cublas_level1.hpp:  return factory()->on([this, n, x, incx, result] (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level1.hpp:cudaTask cublasFlowCapturer::amin(
src/taskflow/cuda/cublas/cublas_level1.hpp:  return factory()->on([this, n, x, incx, result] (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level1.hpp:cudaTask cublasFlowCapturer::asum(
src/taskflow/cuda/cublas/cublas_level1.hpp:  return factory()->on([this, n, x, incx, result] (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level1.hpp:cudaTask cublasFlowCapturer::axpy(
src/taskflow/cuda/cublas/cublas_level1.hpp:  return factory()->on([this, n, alpha, x, incx, y, incy] (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level1.hpp:cudaTask cublasFlowCapturer::vcopy(
src/taskflow/cuda/cublas/cublas_level1.hpp:  return factory()->on([this, n, x, incx, y, incy] (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level1.hpp:cudaTask cublasFlowCapturer::dot(
src/taskflow/cuda/cublas/cublas_level1.hpp:  return factory()->on([this, n, x, incx, y, incy, result] (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level1.hpp:cudaTask cublasFlowCapturer::nrm2(int n, const T* x, int incx, T* result) {
src/taskflow/cuda/cublas/cublas_level1.hpp:  return factory()->on([this, n, x, incx, result] (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level1.hpp:cudaTask cublasFlowCapturer::scal(int n, const T* scalar, T* x, int incx) {
src/taskflow/cuda/cublas/cublas_level1.hpp:  return factory()->on([this, n, scalar, x, incx] (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level1.hpp:cudaTask cublasFlowCapturer::swap(int n, T* x, int incx, T* y, int incy) {
src/taskflow/cuda/cublas/cublas_level1.hpp:  return factory()->on([this, n, x, incx, y, incy] (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_flow.hpp:It inherits methods from tf::cudaFlowCapturerBase and must be used from
src/taskflow/cuda/cublas/cublas_flow.hpp:a tf::cudaFlowCapturer object.
src/taskflow/cuda/cublas/cublas_flow.hpp:All pointers used to %cublasFlowCapturer methods must be in GPU memory space or managed 
src/taskflow/cuda/cublas/cublas_flow.hpp:(i.e., @c cudaMallocManaged),
src/taskflow/cuda/cublas/cublas_flow.hpp:  cudaMalloc(&x, N*sizeof(float));
src/taskflow/cuda/cublas/cublas_flow.hpp:  cudaMalloc(&d_res, sizeof(int));
src/taskflow/cuda/cublas/cublas_flow.hpp:  taskflow.emplace([&](tf::cudaFlowCapturer& capturer){
src/taskflow/cuda/cublas/cublas_flow.hpp:    tf::cudaTask h2d      = capturer.copy(x, host.data(), N);
src/taskflow/cuda/cublas/cublas_flow.hpp:    tf::cudaTask find_max = cublas->amax(N, x, 1, d_res);  
src/taskflow/cuda/cublas/cublas_flow.hpp:    tf::cudaTask d2h      = capturer.copy(&h_res, d_res, 1);
src/taskflow/cuda/cublas/cublas_flow.hpp:class cublasFlowCapturer : public cudaFlowCapturerBase {
src/taskflow/cuda/cublas/cublas_flow.hpp:    to a vector @c d in GPU memory space. 
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask vset(size_t n, const T* h, int inch, T* d, int incd);
src/taskflow/cuda/cublas/cublas_flow.hpp:    This method copies @c n elements from a vector @c d in GPU memory space 
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask vget(size_t n, const T* d, int incd, T* h, int inch);
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask amax(int n, const T* x, int incx, int* result);
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask amin(int n, const T* x, int incx, int* result);
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask asum(int n, const T* x, int incx, T* result);
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask axpy(
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask vcopy(int n, const T* x, int incx, T* y, int incy);
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask dot(int n, const T* x, int incx, const T* y, int incy, T* result);
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask nrm2(int n, const T* x, int incx, T* result);
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask scal(int n, const T* scalar, T* x, int incx);
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask swap(int n, T* x, int incx, T* y, int incy);
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask gemv(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask c_gemv(
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask symv(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask c_symv(
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask syr(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask c_syr(
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask syr2(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask c_syr2(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask trmv(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask c_trmv(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask trsv(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask c_trsv(
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask geam(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask c_geam(
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask gemm(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask c_gemm(
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask gemm_batched(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask c_gemm_batched(
src/taskflow/cuda/cublas/cublas_flow.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask gemm_sbatched(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask c_gemm_sbatched(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask symm(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask c_symm(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask syrk(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask c_syrk(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask syr2k(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask c_syr2k(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask syrkx(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask c_syrkx(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask trmm(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask c_trmm(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask trsm(
src/taskflow/cuda/cublas/cublas_flow.hpp:    cudaTask c_trsm(
src/taskflow/cuda/cublas/cublas_flow.hpp:    void _stream(cudaStream_t);
src/taskflow/cuda/cublas/cublas_flow.hpp:inline void cublasFlowCapturer::_stream(cudaStream_t stream) {
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::geam(
src/taskflow/cuda/cublas/cublas_level3.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::c_geam(
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::gemm(
src/taskflow/cuda/cublas/cublas_level3.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::c_gemm(
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::gemm_batched(
src/taskflow/cuda/cublas/cublas_level3.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::c_gemm_batched(
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::gemm_sbatched(
src/taskflow/cuda/cublas/cublas_level3.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::c_gemm_sbatched(
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::symm(
src/taskflow/cuda/cublas/cublas_level3.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::c_symm(
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::syrk(
src/taskflow/cuda/cublas/cublas_level3.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::c_syrk(
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::syr2k(
src/taskflow/cuda/cublas/cublas_level3.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::c_syr2k(
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::syrkx(
src/taskflow/cuda/cublas/cublas_level3.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::c_syrkx(
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::trmm(
src/taskflow/cuda/cublas/cublas_level3.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::c_trmm(
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::trsm(
src/taskflow/cuda/cublas/cublas_level3.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level3.hpp:cudaTask cublasFlowCapturer::c_trsm(
src/taskflow/cuda/cublas/cublas_error.hpp:if(TF_CUDA_GET_FIRST(__VA_ARGS__) != CUBLAS_STATUS_SUCCESS) {  \
src/taskflow/cuda/cublas/cublas_error.hpp:  auto ev = TF_CUDA_GET_FIRST(__VA_ARGS__);                    \
src/taskflow/cuda/cublas/cublas_error.hpp:  tf::ostreamize(oss, TF_CUDA_REMOVE_FIRST(__VA_ARGS__));      \
src/taskflow/cuda/cublas/cublas_helper.hpp:cudaTask cublasFlowCapturer::vset(
src/taskflow/cuda/cublas/cublas_helper.hpp:  return factory()->on([n, h, inch, d, incd] (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_helper.hpp:cudaTask cublasFlowCapturer::vget(size_t n, const T* d, int incd, T* h, int inch) {
src/taskflow/cuda/cublas/cublas_helper.hpp:  return factory()->on([n, d, incd, h, inch] (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_handle.hpp:using cublasPerThreadHandlePool = cudaPerThreadDeviceObjectPool<
src/taskflow/cuda/cublas/cublas_handle.hpp:    _ptr {cublas_per_thread_handle_pool().acquire(cuda_get_device())} {
src/taskflow/cuda/cublas/cublas_level2.hpp:cudaTask cublasFlowCapturer::gemv(
src/taskflow/cuda/cublas/cublas_level2.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level2.hpp:cudaTask cublasFlowCapturer::c_gemv(
src/taskflow/cuda/cublas/cublas_level2.hpp:cudaTask cublasFlowCapturer::trmv(
src/taskflow/cuda/cublas/cublas_level2.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level2.hpp:cudaTask cublasFlowCapturer::c_trmv(
src/taskflow/cuda/cublas/cublas_level2.hpp:cudaTask cublasFlowCapturer::trsv(
src/taskflow/cuda/cublas/cublas_level2.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level2.hpp:cudaTask cublasFlowCapturer::c_trsv(
src/taskflow/cuda/cublas/cublas_level2.hpp:cudaTask cublasFlowCapturer::symv(
src/taskflow/cuda/cublas/cublas_level2.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level2.hpp:cudaTask cublasFlowCapturer::c_symv(
src/taskflow/cuda/cublas/cublas_level2.hpp:cudaTask cublasFlowCapturer::syr(
src/taskflow/cuda/cublas/cublas_level2.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level2.hpp:cudaTask cublasFlowCapturer::c_syr(
src/taskflow/cuda/cublas/cublas_level2.hpp:cudaTask cublasFlowCapturer::syr2(
src/taskflow/cuda/cublas/cublas_level2.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cublas/cublas_level2.hpp:cudaTask cublasFlowCapturer::c_syr2(
src/taskflow/cuda/cuda_pool.hpp:#include "cuda_error.hpp"
src/taskflow/cuda/cuda_pool.hpp:@brief per-thread object pool to manage CUDA device object
src/taskflow/cuda/cuda_pool.hpp:A CUDA device object has a lifetime associated with a device,
src/taskflow/cuda/cuda_pool.hpp:for example, @c cudaStream_t, @c cublasHandle_t, etc.
src/taskflow/cuda/cuda_pool.hpp:There exists an one-to-one relationship between CUDA devices in CUDA Runtime API
src/taskflow/cuda/cuda_pool.hpp:and CUcontexts in the CUDA Driver API within a process.
src/taskflow/cuda/cuda_pool.hpp:The specific context which the CUDA Runtime API uses for a device 
src/taskflow/cuda/cuda_pool.hpp:From the perspective of the CUDA Runtime API, 
src/taskflow/cuda/cuda_pool.hpp:class cudaPerThreadDeviceObjectPool {
src/taskflow/cuda/cuda_pool.hpp:  // Due to some ordering, cuda context may be destroyed when the master
src/taskflow/cuda/cuda_pool.hpp:  // program thread destroys the cuda object.
src/taskflow/cuda/cuda_pool.hpp:  // destroy cuda objects while the master thread only keeps a weak reference
src/taskflow/cuda/cuda_pool.hpp:  struct cudaGlobalDeviceObjectPool {
src/taskflow/cuda/cuda_pool.hpp:    cudaPerThreadDeviceObjectPool() = default;
src/taskflow/cuda/cuda_pool.hpp:    inline static cudaGlobalDeviceObjectPool _shared_pool;
src/taskflow/cuda/cuda_pool.hpp:// cudaPerThreadDeviceObject::cudaHanale definition
src/taskflow/cuda/cuda_pool.hpp:cudaPerThreadDeviceObjectPool<H, C, D>::Object::Object(int d) : 
src/taskflow/cuda/cuda_pool.hpp:  cudaScopedDevice ctx(device);
src/taskflow/cuda/cuda_pool.hpp:cudaPerThreadDeviceObjectPool<H, C, D>::Object::~Object() {
src/taskflow/cuda/cuda_pool.hpp:  cudaScopedDevice ctx(device);
src/taskflow/cuda/cuda_pool.hpp:// cudaPerThreadDeviceObject::cudaHanaldePool definition
src/taskflow/cuda/cuda_pool.hpp:std::shared_ptr<typename cudaPerThreadDeviceObjectPool<H, C, D>::Object>
src/taskflow/cuda/cuda_pool.hpp:cudaPerThreadDeviceObjectPool<H, C, D>::cudaGlobalDeviceObjectPool::acquire(int d) {
src/taskflow/cuda/cuda_pool.hpp:void cudaPerThreadDeviceObjectPool<H, C, D>::cudaGlobalDeviceObjectPool::release(
src/taskflow/cuda/cuda_pool.hpp:// cudaPerThreadDeviceObject definition
src/taskflow/cuda/cuda_pool.hpp:std::shared_ptr<typename cudaPerThreadDeviceObjectPool<H, C, D>::Object> 
src/taskflow/cuda/cuda_pool.hpp:cudaPerThreadDeviceObjectPool<H, C, D>::acquire(int d) {
src/taskflow/cuda/cuda_pool.hpp:void cudaPerThreadDeviceObjectPool<H, C, D>::release(
src/taskflow/cuda/cuda_pool.hpp:size_t cudaPerThreadDeviceObjectPool<H, C, D>::footprint_size() const {
src/taskflow/cuda/cuda_optimizer.hpp:#include "cuda_graph.hpp"
src/taskflow/cuda/cuda_optimizer.hpp:@file cuda_optimizer.hpp
src/taskflow/cuda/cuda_optimizer.hpp:@brief %cudaFlow capturing algorithms include file
src/taskflow/cuda/cuda_optimizer.hpp:// cudaCapturingBase
src/taskflow/cuda/cuda_optimizer.hpp:class cudaCapturingBase {
src/taskflow/cuda/cuda_optimizer.hpp:    std::vector<cudaNode*> _toposort(cudaGraph&);
src/taskflow/cuda/cuda_optimizer.hpp:    std::vector<std::vector<cudaNode*>> _levelize(cudaGraph&);
src/taskflow/cuda/cuda_optimizer.hpp:inline std::vector<cudaNode*> cudaCapturingBase::_toposort(cudaGraph& graph) {
src/taskflow/cuda/cuda_optimizer.hpp:  std::vector<cudaNode*> res;
src/taskflow/cuda/cuda_optimizer.hpp:  std::queue<cudaNode*> bfs;
src/taskflow/cuda/cuda_optimizer.hpp:    auto& hu = std::get<cudaNode::Capture>(u->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:    auto& hu = std::get<cudaNode::Capture>(u->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:      auto& hv = std::get<cudaNode::Capture>(v->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:inline std::vector<std::vector<cudaNode*>> 
src/taskflow/cuda/cuda_optimizer.hpp:cudaCapturingBase::_levelize(cudaGraph& graph) {
src/taskflow/cuda/cuda_optimizer.hpp:  std::queue<cudaNode*> bfs;
src/taskflow/cuda/cuda_optimizer.hpp:    auto& hu = std::get<cudaNode::Capture>(u->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:    auto& hu = std::get<cudaNode::Capture>(u->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:      auto& hv = std::get<cudaNode::Capture>(v->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:  std::vector<std::vector<cudaNode*>> level_graph(max_level+1);
src/taskflow/cuda/cuda_optimizer.hpp:    auto& hu = std::get<cudaNode::Capture>(u->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:    //  assert(hu.level < std::get<cudaNode::Capture>(s->_handle).level);
src/taskflow/cuda/cuda_optimizer.hpp:// class definition: cudaSequentialCapturing
src/taskflow/cuda/cuda_optimizer.hpp:@class cudaSequentialCapturing
src/taskflow/cuda/cuda_optimizer.hpp:@brief class to capture the described graph into a native cudaGraph
src/taskflow/cuda/cuda_optimizer.hpp:the described graph and captures dependent GPU tasks using a single stream.
src/taskflow/cuda/cuda_optimizer.hpp:All GPU tasks run sequentially without breaking inter dependencies.
src/taskflow/cuda/cuda_optimizer.hpp:class cudaSequentialCapturing : public cudaCapturingBase {
src/taskflow/cuda/cuda_optimizer.hpp:  friend class cudaFlowCapturer;
src/taskflow/cuda/cuda_optimizer.hpp:    cudaSequentialCapturing() = default;
src/taskflow/cuda/cuda_optimizer.hpp:    cudaGraph_t _optimize(cudaGraph& graph);
src/taskflow/cuda/cuda_optimizer.hpp:inline cudaGraph_t cudaSequentialCapturing::_optimize(cudaGraph& graph) {
src/taskflow/cuda/cuda_optimizer.hpp:  // we must use ThreadLocal mode to avoid clashing with CUDA global states
src/taskflow/cuda/cuda_optimizer.hpp:  cudaScopedPerThreadStream stream;
src/taskflow/cuda/cuda_optimizer.hpp:  cudaGraph_t native_g;
src/taskflow/cuda/cuda_optimizer.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:    cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), 
src/taskflow/cuda/cuda_optimizer.hpp:    std::get<cudaNode::Capture>(node->_handle).work(stream);  
src/taskflow/cuda/cuda_optimizer.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:    cudaStreamEndCapture(stream, &native_g), "failed to end capture"
src/taskflow/cuda/cuda_optimizer.hpp:// class definition: cudaRoundRobinCapturing
src/taskflow/cuda/cuda_optimizer.hpp:@class cudaRoundRobinCapturing
src/taskflow/cuda/cuda_optimizer.hpp:@brief class to capture the described graph into a native cudaGraph
src/taskflow/cuda/cuda_optimizer.hpp:class cudaRoundRobinCapturing : public cudaCapturingBase {
src/taskflow/cuda/cuda_optimizer.hpp:  friend class cudaFlowCapturer;
src/taskflow/cuda/cuda_optimizer.hpp:    cudaRoundRobinCapturing();
src/taskflow/cuda/cuda_optimizer.hpp:    cudaRoundRobinCapturing(size_t num_streams);
src/taskflow/cuda/cuda_optimizer.hpp:    cudaGraph_t _optimize(cudaGraph& graph);
src/taskflow/cuda/cuda_optimizer.hpp:    void _reset(std::vector<cudaNode*>& graph);
src/taskflow/cuda/cuda_optimizer.hpp:inline cudaRoundRobinCapturing::cudaRoundRobinCapturing(size_t num_streams) :
src/taskflow/cuda/cuda_optimizer.hpp:inline size_t cudaRoundRobinCapturing::num_streams() const {
src/taskflow/cuda/cuda_optimizer.hpp:inline void cudaRoundRobinCapturing::num_streams(size_t n) {
src/taskflow/cuda/cuda_optimizer.hpp:inline void cudaRoundRobinCapturing::_reset(std::vector<cudaNode*>& graph) {
src/taskflow/cuda/cuda_optimizer.hpp:    auto& hn = std::get<cudaNode::Capture>(node->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:inline cudaGraph_t cudaRoundRobinCapturing::_optimize(cudaGraph& graph) {
src/taskflow/cuda/cuda_optimizer.hpp:  std::vector<cudaScopedPerThreadStream> streams(_num_streams);
src/taskflow/cuda/cuda_optimizer.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:    cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeThreadLocal), 
src/taskflow/cuda/cuda_optimizer.hpp:  std::vector<cudaScopedPerThreadEvent> events;
src/taskflow/cuda/cuda_optimizer.hpp:  cudaEvent_t fork_event = events.emplace_back();
src/taskflow/cuda/cuda_optimizer.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:    cudaEventRecord(fork_event, streams[0]), "faid to record fork"
src/taskflow/cuda/cuda_optimizer.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:      cudaStreamWaitEvent(streams[i], fork_event, 0), "failed to wait on fork"
src/taskflow/cuda/cuda_optimizer.hpp:    auto& hn = std::get<cudaNode::Capture>(node->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:    cudaNode* wait_node{nullptr};
src/taskflow/cuda/cuda_optimizer.hpp:      auto& phn = std::get<cudaNode::Capture>(pn->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:        else if(std::get<cudaNode::Capture>(wait_node->_handle).level < phn.level) {
src/taskflow/cuda/cuda_optimizer.hpp:        TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:          cudaStreamWaitEvent(
src/taskflow/cuda/cuda_optimizer.hpp:      assert(std::get<cudaNode::Capture>(wait_node->_handle).event); 
src/taskflow/cuda/cuda_optimizer.hpp:      TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:        cudaStreamWaitEvent(
src/taskflow/cuda/cuda_optimizer.hpp:          std::get<cudaNode::Capture>(wait_node->_handle).event, 
src/taskflow/cuda/cuda_optimizer.hpp:      auto& shn = std::get<cudaNode::Capture>(sn->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:          TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:            cudaEventRecord(hn.event, streams[hn.level % _num_streams]), "faid to record node's stream"
src/taskflow/cuda/cuda_optimizer.hpp:    cudaEvent_t join_event = events.emplace_back();
src/taskflow/cuda/cuda_optimizer.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:      cudaEventRecord(join_event, streams[i]), "failed to record join"
src/taskflow/cuda/cuda_optimizer.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:      cudaStreamWaitEvent(streams[0], join_event), "failed to wait on join"
src/taskflow/cuda/cuda_optimizer.hpp:  cudaGraph_t native_g;
src/taskflow/cuda/cuda_optimizer.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:    cudaStreamEndCapture(streams[0], &native_g), "failed to end capture"
src/taskflow/cuda/cuda_optimizer.hpp:  //tf::cuda_dump_graph(std::cout, native_g);
src/taskflow/cuda/cuda_optimizer.hpp:/*class cudaGreedyCapturing: public cudaCapturingBase {
src/taskflow/cuda/cuda_optimizer.hpp:  friend class cudaFlowCapturer;
src/taskflow/cuda/cuda_optimizer.hpp:    cudaGreedyCapturing();
src/taskflow/cuda/cuda_optimizer.hpp:    cudaGreedyCapturing(size_t num_stream);
src/taskflow/cuda/cuda_optimizer.hpp:    cudaGraph_t _optimize(cudaGraph& graph);
src/taskflow/cuda/cuda_optimizer.hpp:inline cudaGreedyCapturing::cudaGreedyCapturing(size_t num_streams): 
src/taskflow/cuda/cuda_optimizer.hpp:inline cudaGraph_t cudaGreedyCapturing::_optimize(cudaGraph& graph) {
src/taskflow/cuda/cuda_optimizer.hpp:  std::vector<cudaScopedPerThreadStream> streams(_num_streams);
src/taskflow/cuda/cuda_optimizer.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:    cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeThreadLocal), 
src/taskflow/cuda/cuda_optimizer.hpp:  std::vector<cudaScopedPerThreadEvent> events;
src/taskflow/cuda/cuda_optimizer.hpp:  cudaEvent_t fork_event = events.emplace_back();
src/taskflow/cuda/cuda_optimizer.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:    cudaEventRecord(fork_event, streams[0]), "faid to record fork"
src/taskflow/cuda/cuda_optimizer.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:      cudaStreamWaitEvent(streams[i], fork_event, 0), "failed to wait on fork"
src/taskflow/cuda/cuda_optimizer.hpp:  std::vector<cudaNode*> assign(streams.size());
src/taskflow/cuda/cuda_optimizer.hpp:  std::vector<cudaNode*> prev_assign(streams.size());
src/taskflow/cuda/cuda_optimizer.hpp:  std::queue<cudaNode*> remains;;
src/taskflow/cuda/cuda_optimizer.hpp:    auto& hn = std::get<cudaNode::Capture>(node->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:      auto& hn = std::get<cudaNode::Capture>(node->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:          auto& phn = std::get<cudaNode::Capture>(pn->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:              auto& pahn = std::get<cudaNode::Capture>(pan->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:                auto& shn = std::get<cudaNode::Capture>(sn->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:          auto& lhn = std::get<cudaNode::Capture>(ln->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:      auto& hn = std::get<cudaNode::Capture>(node->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:        auto& phn = std::get<cudaNode::Capture>(pn->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:      auto& hn = std::get<cudaNode::Capture>(node->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:        auto& phn = std::get<cudaNode::Capture>(pn->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:          TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:            cudaStreamWaitEvent(streams[hn.sid], phn.event), "failed to wait on the node"
src/taskflow/cuda/cuda_optimizer.hpp:        auto& shn = std::get<cudaNode::Capture>(sn->_handle);
src/taskflow/cuda/cuda_optimizer.hpp:          TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:            cudaEventRecord(hn.event, streams[hn.sid]), "failed to record"
src/taskflow/cuda/cuda_optimizer.hpp:    cudaEvent_t join_event = events.emplace_back();
src/taskflow/cuda/cuda_optimizer.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:      cudaEventRecord(join_event, streams[i]), "failed to record join"
src/taskflow/cuda/cuda_optimizer.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:      cudaStreamWaitEvent(streams[0], join_event), "failed to wait on join"
src/taskflow/cuda/cuda_optimizer.hpp:  cudaGraph_t native_g;
src/taskflow/cuda/cuda_optimizer.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_optimizer.hpp:    cudaStreamEndCapture(streams[0], &native_g), "failed to end capture"
src/taskflow/cuda/cuda_optimizer.hpp:  //tf::cuda_dump_graph(std::cout, native_g);
src/taskflow/cuda/cuda_device.hpp:#include "cuda_error.hpp"
src/taskflow/cuda/cuda_device.hpp:@file cuda_device.hpp
src/taskflow/cuda/cuda_device.hpp:@brief CUDA device utilities include file
src/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_num_devices() {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(cudaGetDeviceCount(&N), "failed to get device count");
src/taskflow/cuda/cuda_device.hpp:inline int cuda_get_device() {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(cudaGetDevice(&id), "failed to get current device id");
src/taskflow/cuda/cuda_device.hpp:inline void cuda_set_device(int id) {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(cudaSetDevice(id), "failed to switch to device ", id);
src/taskflow/cuda/cuda_device.hpp:inline void cuda_get_device_property(int i, cudaDeviceProp& p) {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_device.hpp:    cudaGetDeviceProperties(&p, i), "failed to get property of device ", i
src/taskflow/cuda/cuda_device.hpp:inline cudaDeviceProp cuda_get_device_property(int i) {
src/taskflow/cuda/cuda_device.hpp:  cudaDeviceProp p;
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_device.hpp:    cudaGetDeviceProperties(&p, i), "failed to get property of device ", i
src/taskflow/cuda/cuda_device.hpp:inline void cuda_dump_device_property(std::ostream& os, const cudaDeviceProp& p) {
src/taskflow/cuda/cuda_device.hpp:     << "GPU sharing Host Memory:       " << p.integrated << '\n'
src/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_threads_per_block(int d) {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerBlock, d),
src/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_x_dim_per_block(int d) {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxBlockDimX, d),
src/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_y_dim_per_block(int d) {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxBlockDimY, d),
src/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_z_dim_per_block(int d) {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxBlockDimZ, d),
src/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_x_dim_per_grid(int d) {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxGridDimX, d),
src/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_y_dim_per_grid(int d) {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxGridDimY, d),
src/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_z_dim_per_grid(int d) {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxGridDimZ, d),
src/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_shm_per_block(int d) {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&num, cudaDevAttrMaxSharedMemoryPerBlock, d),
src/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_warp_size(int d) {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&num, cudaDevAttrWarpSize, d),
src/taskflow/cuda/cuda_device.hpp:inline int cuda_get_device_compute_capability_major(int d) {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&num, cudaDevAttrComputeCapabilityMajor, d),
src/taskflow/cuda/cuda_device.hpp:inline int cuda_get_device_compute_capability_minor(int d) {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&num, cudaDevAttrComputeCapabilityMinor, d),
src/taskflow/cuda/cuda_device.hpp:inline bool cuda_get_device_unified_addressing(int d) {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&num, cudaDevAttrUnifiedAddressing, d),
src/taskflow/cuda/cuda_device.hpp:// CUDA Version
src/taskflow/cuda/cuda_device.hpp:@brief queries the latest CUDA version (1000 * major + 10 * minor) supported by the driver 
src/taskflow/cuda/cuda_device.hpp:inline int cuda_get_driver_version() {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_device.hpp:    cudaDriverGetVersion(&num), 
src/taskflow/cuda/cuda_device.hpp:    "failed to query the latest cuda version supported by the driver"
src/taskflow/cuda/cuda_device.hpp:@brief queries the CUDA Runtime version (1000 * major + 10 * minor)
src/taskflow/cuda/cuda_device.hpp:inline int cuda_get_runtime_version() {
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_device.hpp:    cudaRuntimeGetVersion(&num), "failed to query cuda runtime version"
src/taskflow/cuda/cuda_device.hpp:// cudaScopedDevice
src/taskflow/cuda/cuda_device.hpp:/** @class cudaScopedDevice
src/taskflow/cuda/cuda_device.hpp:  tf::cudaScopedDevice device(1);  // switch to the device context 1
src/taskflow/cuda/cuda_device.hpp:  cudaStream_t stream;
src/taskflow/cuda/cuda_device.hpp:  cudaStreamCreate(&stream);
src/taskflow/cuda/cuda_device.hpp:%cudaScopedDevice is neither movable nor copyable.
src/taskflow/cuda/cuda_device.hpp:class cudaScopedDevice {
src/taskflow/cuda/cuda_device.hpp:    explicit cudaScopedDevice(int device);
src/taskflow/cuda/cuda_device.hpp:    ~cudaScopedDevice();
src/taskflow/cuda/cuda_device.hpp:    cudaScopedDevice() = delete;
src/taskflow/cuda/cuda_device.hpp:    cudaScopedDevice(const cudaScopedDevice&) = delete;
src/taskflow/cuda/cuda_device.hpp:    cudaScopedDevice(cudaScopedDevice&&) = delete;
src/taskflow/cuda/cuda_device.hpp:inline cudaScopedDevice::cudaScopedDevice(int dev) { 
src/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(cudaGetDevice(&_p), "failed to get current device scope");
src/taskflow/cuda/cuda_device.hpp:    TF_CHECK_CUDA(cudaSetDevice(dev), "failed to scope on device ", dev);
src/taskflow/cuda/cuda_device.hpp:inline cudaScopedDevice::~cudaScopedDevice() { 
src/taskflow/cuda/cuda_device.hpp:    cudaSetDevice(_p);
src/taskflow/cuda/cuda_device.hpp:    //TF_CHECK_CUDA(cudaSetDevice(_p), "failed to scope back to device ", _p);
src/taskflow/cuda/cuda_device.hpp:}  // end of namespace cuda ---------------------------------------------------
src/taskflow/cuda/cuda_graph.hpp:#include "cuda_memory.hpp"
src/taskflow/cuda/cuda_graph.hpp:#include "cuda_stream.hpp"
src/taskflow/cuda/cuda_graph.hpp:// cudaGraph_t routines
src/taskflow/cuda/cuda_graph.hpp:cudaMemcpy3DParms cuda_get_copy_parms(T* tgt, const T* src, size_t num) {
src/taskflow/cuda/cuda_graph.hpp:  cudaMemcpy3DParms p;
src/taskflow/cuda/cuda_graph.hpp:  p.srcPos = ::make_cudaPos(0, 0, 0);
src/taskflow/cuda/cuda_graph.hpp:  p.srcPtr = ::make_cudaPitchedPtr(const_cast<T*>(src), num*sizeof(U), num, 1);
src/taskflow/cuda/cuda_graph.hpp:  p.dstPos = ::make_cudaPos(0, 0, 0);
src/taskflow/cuda/cuda_graph.hpp:  p.dstPtr = ::make_cudaPitchedPtr(tgt, num*sizeof(U), num, 1);
src/taskflow/cuda/cuda_graph.hpp:  p.extent = ::make_cudaExtent(num*sizeof(U), 1, 1);
src/taskflow/cuda/cuda_graph.hpp:  p.kind = cudaMemcpyDefault;
src/taskflow/cuda/cuda_graph.hpp:inline cudaMemcpy3DParms cuda_get_memcpy_parms(
src/taskflow/cuda/cuda_graph.hpp:  // Parameters in cudaPitchedPtr
src/taskflow/cuda/cuda_graph.hpp:  cudaMemcpy3DParms p;
src/taskflow/cuda/cuda_graph.hpp:  p.srcPos = ::make_cudaPos(0, 0, 0);
src/taskflow/cuda/cuda_graph.hpp:  p.srcPtr = ::make_cudaPitchedPtr(const_cast<void*>(src), bytes, bytes, 1);
src/taskflow/cuda/cuda_graph.hpp:  p.dstPos = ::make_cudaPos(0, 0, 0);
src/taskflow/cuda/cuda_graph.hpp:  p.dstPtr = ::make_cudaPitchedPtr(tgt, bytes, bytes, 1);
src/taskflow/cuda/cuda_graph.hpp:  p.extent = ::make_cudaExtent(bytes, 1, 1);
src/taskflow/cuda/cuda_graph.hpp:  p.kind = cudaMemcpyDefault;
src/taskflow/cuda/cuda_graph.hpp:inline cudaMemsetParams cuda_get_memset_parms(void* dst, int ch, size_t count) {
src/taskflow/cuda/cuda_graph.hpp:  cudaMemsetParams p;
src/taskflow/cuda/cuda_graph.hpp:cudaMemsetParams cuda_get_fill_parms(T* dst, T value, size_t count) {
src/taskflow/cuda/cuda_graph.hpp:  cudaMemsetParams p;
src/taskflow/cuda/cuda_graph.hpp:cudaMemsetParams cuda_get_zero_parms(T* dst, size_t count) {
src/taskflow/cuda/cuda_graph.hpp:  cudaMemsetParams p;
src/taskflow/cuda/cuda_graph.hpp:@brief queries the number of root nodes in a native CUDA graph
src/taskflow/cuda/cuda_graph.hpp:inline size_t cuda_get_graph_num_root_nodes(cudaGraph_t graph) {
src/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetRootNodes(graph, nullptr, &num_nodes), 
src/taskflow/cuda/cuda_graph.hpp:@brief queries the number of nodes in a native CUDA graph
src/taskflow/cuda/cuda_graph.hpp:inline size_t cuda_get_graph_num_nodes(cudaGraph_t graph) {
src/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetNodes(graph, nullptr, &num_nodes), 
src/taskflow/cuda/cuda_graph.hpp:@brief queries the number of edges in a native CUDA graph
src/taskflow/cuda/cuda_graph.hpp:inline size_t cuda_get_graph_num_edges(cudaGraph_t graph) {
src/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetEdges(graph, nullptr, nullptr, &num_edges), 
src/taskflow/cuda/cuda_graph.hpp:@brief acquires the nodes in a native CUDA graph
src/taskflow/cuda/cuda_graph.hpp:inline std::vector<cudaGraphNode_t> cuda_get_graph_nodes(cudaGraph_t graph) {
src/taskflow/cuda/cuda_graph.hpp:  size_t num_nodes = cuda_get_graph_num_nodes(graph);
src/taskflow/cuda/cuda_graph.hpp:  std::vector<cudaGraphNode_t> nodes(num_nodes);
src/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetNodes(graph, nodes.data(), &num_nodes),
src/taskflow/cuda/cuda_graph.hpp:@brief acquires the root nodes in a native CUDA graph
src/taskflow/cuda/cuda_graph.hpp:inline std::vector<cudaGraphNode_t> cuda_get_graph_root_nodes(cudaGraph_t graph) {
src/taskflow/cuda/cuda_graph.hpp:  size_t num_nodes = cuda_get_graph_num_root_nodes(graph);
src/taskflow/cuda/cuda_graph.hpp:  std::vector<cudaGraphNode_t> nodes(num_nodes);
src/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetRootNodes(graph, nodes.data(), &num_nodes),
src/taskflow/cuda/cuda_graph.hpp:@brief acquires the edges in a native CUDA graph
src/taskflow/cuda/cuda_graph.hpp:inline std::vector<std::pair<cudaGraphNode_t, cudaGraphNode_t>>
src/taskflow/cuda/cuda_graph.hpp:cuda_get_graph_edges(cudaGraph_t graph) {
src/taskflow/cuda/cuda_graph.hpp:  size_t num_edges = cuda_get_graph_num_edges(graph);
src/taskflow/cuda/cuda_graph.hpp:  std::vector<cudaGraphNode_t> froms(num_edges), tos(num_edges);
src/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetEdges(graph, froms.data(), tos.data(), &num_edges),
src/taskflow/cuda/cuda_graph.hpp:  std::vector<std::pair<cudaGraphNode_t, cudaGraphNode_t>> edges(num_edges);
src/taskflow/cuda/cuda_graph.hpp:@brief queries the type of a native CUDA graph node
src/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeKernel      = 0x00
src/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeMemcpy      = 0x01
src/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeMemset      = 0x02
src/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeHost        = 0x03
src/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeGraph       = 0x04
src/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeEmpty       = 0x05
src/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeWaitEvent   = 0x06
src/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeEventRecord = 0x07
src/taskflow/cuda/cuda_graph.hpp:inline cudaGraphNodeType cuda_get_graph_node_type(cudaGraphNode_t node) {
src/taskflow/cuda/cuda_graph.hpp:  cudaGraphNodeType type;
src/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_graph.hpp:    cudaGraphNodeGetType(node, &type), "failed to get native graph node type"
src/taskflow/cuda/cuda_graph.hpp:@brief convert the type of a native CUDA graph node to a readable string
src/taskflow/cuda/cuda_graph.hpp:inline const char* cuda_graph_node_type_to_string(cudaGraphNodeType type) {
src/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeKernel      : return "kernel";
src/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeMemcpy      : return "memcpy";
src/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeMemset      : return "memset";
src/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeHost        : return "host";
src/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeGraph       : return "graph";
src/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeEmpty       : return "empty";
src/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeWaitEvent   : return "event_wait";
src/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeEventRecord : return "event_record";
src/taskflow/cuda/cuda_graph.hpp:@brief dumps a native CUDA graph and all associated child graphs to a DOT format
src/taskflow/cuda/cuda_graph.hpp:@param graph native CUDA graph
src/taskflow/cuda/cuda_graph.hpp:void cuda_dump_graph(T& os, cudaGraph_t graph) {
src/taskflow/cuda/cuda_graph.hpp:  os << "digraph cudaGraph {\n";
src/taskflow/cuda/cuda_graph.hpp:  std::stack<std::tuple<cudaGraph_t, cudaGraphNode_t, int>> stack;
src/taskflow/cuda/cuda_graph.hpp:       << "label=\"cudaGraph-L" << l << "\";\n"
src/taskflow/cuda/cuda_graph.hpp:    auto nodes = cuda_get_graph_nodes(graph);
src/taskflow/cuda/cuda_graph.hpp:    auto edges = cuda_get_graph_edges(graph);
src/taskflow/cuda/cuda_graph.hpp:      auto type = cuda_get_graph_node_type(node);
src/taskflow/cuda/cuda_graph.hpp:      if(type == cudaGraphNodeTypeGraph) {
src/taskflow/cuda/cuda_graph.hpp:        cudaGraph_t graph;
src/taskflow/cuda/cuda_graph.hpp:        TF_CHECK_CUDA(cudaGraphChildGraphNodeGetGraph(node, &graph), "");
src/taskflow/cuda/cuda_graph.hpp:           << "label=\"cudaGraph-L" << l+1
src/taskflow/cuda/cuda_graph.hpp:           << cuda_graph_node_type_to_string(type) 
src/taskflow/cuda/cuda_graph.hpp:      std::unordered_set<cudaGraphNode_t> successors;
src/taskflow/cuda/cuda_graph.hpp:// cudaGraph class
src/taskflow/cuda/cuda_graph.hpp:// class: cudaGraph
src/taskflow/cuda/cuda_graph.hpp:class cudaGraph : public CustomGraphBase {
src/taskflow/cuda/cuda_graph.hpp:  friend class cudaNode;
src/taskflow/cuda/cuda_graph.hpp:  friend class cudaTask;
src/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowCapturerBase;
src/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowCapturer;
src/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlow;
src/taskflow/cuda/cuda_graph.hpp:  friend class cudaCapturingBase;
src/taskflow/cuda/cuda_graph.hpp:  friend class cudaSequentialCapturing;
src/taskflow/cuda/cuda_graph.hpp:  friend class cudaRoundRobinCapturing;
src/taskflow/cuda/cuda_graph.hpp:    cudaGraph() = default;
src/taskflow/cuda/cuda_graph.hpp:    ~cudaGraph();
src/taskflow/cuda/cuda_graph.hpp:    cudaGraph(const cudaGraph&) = delete;
src/taskflow/cuda/cuda_graph.hpp:    cudaGraph(cudaGraph&&);
src/taskflow/cuda/cuda_graph.hpp:    cudaGraph& operator = (const cudaGraph&) = delete;
src/taskflow/cuda/cuda_graph.hpp:    cudaGraph& operator = (cudaGraph&&);
src/taskflow/cuda/cuda_graph.hpp:    cudaNode* emplace_back(ArgsT&&...);
src/taskflow/cuda/cuda_graph.hpp:    cudaGraph_t _native_handle {nullptr};
src/taskflow/cuda/cuda_graph.hpp:    //std::vector<std::unique_ptr<cudaNode>> _nodes;
src/taskflow/cuda/cuda_graph.hpp:    std::vector<cudaNode*> _nodes;
src/taskflow/cuda/cuda_graph.hpp:// cudaNode class
src/taskflow/cuda/cuda_graph.hpp:// class: cudaNode
src/taskflow/cuda/cuda_graph.hpp:// in order to work with gpu context
src/taskflow/cuda/cuda_graph.hpp:class cudaNode {
src/taskflow/cuda/cuda_graph.hpp:  friend class cudaGraph;
src/taskflow/cuda/cuda_graph.hpp:  friend class cudaTask;
src/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlow;
src/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowCapturer;
src/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowCapturerBase;
src/taskflow/cuda/cuda_graph.hpp:  friend class cudaCapturingBase;
src/taskflow/cuda/cuda_graph.hpp:  friend class cudaSequentialCapturing;
src/taskflow/cuda/cuda_graph.hpp:  friend class cudaRoundRobinCapturing;
src/taskflow/cuda/cuda_graph.hpp:  friend class cudaGreedyCapturing;
src/taskflow/cuda/cuda_graph.hpp:    cudaGraph graph;
src/taskflow/cuda/cuda_graph.hpp:    std::function<void(cudaStream_t)> work;
src/taskflow/cuda/cuda_graph.hpp:    cudaEvent_t event {nullptr};
src/taskflow/cuda/cuda_graph.hpp:    cudaNode() = delete;
src/taskflow/cuda/cuda_graph.hpp:    cudaNode(cudaGraph&, ArgsT&&...);
src/taskflow/cuda/cuda_graph.hpp:    cudaGraph& _graph;
src/taskflow/cuda/cuda_graph.hpp:    cudaGraphNode_t _native_handle {nullptr};
src/taskflow/cuda/cuda_graph.hpp:    std::vector<cudaNode*> _successors;
src/taskflow/cuda/cuda_graph.hpp:    std::vector<cudaNode*> _dependents;
src/taskflow/cuda/cuda_graph.hpp:    void _precede(cudaNode*);
src/taskflow/cuda/cuda_graph.hpp:// cudaNode definitions
src/taskflow/cuda/cuda_graph.hpp:cudaNode::Host::Host(C&& c) : func {std::forward<C>(c)} {
src/taskflow/cuda/cuda_graph.hpp:inline void cudaNode::Host::callback(void* data) { 
src/taskflow/cuda/cuda_graph.hpp:cudaNode::Kernel::Kernel(F&& f) : 
src/taskflow/cuda/cuda_graph.hpp:cudaNode::Capture::Capture(C&& work) : 
src/taskflow/cuda/cuda_graph.hpp:cudaNode::cudaNode(cudaGraph& graph, ArgsT&&... args) : 
src/taskflow/cuda/cuda_graph.hpp:inline void cudaNode::_precede(cudaNode* v) {
src/taskflow/cuda/cuda_graph.hpp:  if(_handle.index() != cudaNode::CAPTURE) {
src/taskflow/cuda/cuda_graph.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_graph.hpp:      ::cudaGraphAddDependencies(
src/taskflow/cuda/cuda_graph.hpp://inline void cudaNode::_set_state(int flag) { 
src/taskflow/cuda/cuda_graph.hpp://inline void cudaNode::_unset_state(int flag) { 
src/taskflow/cuda/cuda_graph.hpp://inline void cudaNode::_clear_state() { 
src/taskflow/cuda/cuda_graph.hpp://inline bool cudaNode::_has_state(int flag) const {
src/taskflow/cuda/cuda_graph.hpp:// cudaGraph definitions
src/taskflow/cuda/cuda_graph.hpp:inline cudaGraph::~cudaGraph() {
src/taskflow/cuda/cuda_graph.hpp:inline cudaGraph::cudaGraph(cudaGraph&& g) :
src/taskflow/cuda/cuda_graph.hpp:inline cudaGraph& cudaGraph::operator = (cudaGraph&& rhs) {
src/taskflow/cuda/cuda_graph.hpp:inline bool cudaGraph::empty() const {
src/taskflow/cuda/cuda_graph.hpp:inline void cudaGraph::clear() {
src/taskflow/cuda/cuda_graph.hpp:cudaNode* cudaGraph::emplace_back(ArgsT&&... args) {
src/taskflow/cuda/cuda_graph.hpp:  //auto node = std::make_unique<cudaNode>(std::forward<ArgsT>(args)...);
src/taskflow/cuda/cuda_graph.hpp:  auto node = new cudaNode(std::forward<ArgsT>(args)...);
src/taskflow/cuda/cuda_graph.hpp:inline void cudaGraph::dump(
src/taskflow/cuda/cuda_graph.hpp:  std::stack<std::tuple<const cudaGraph*, const cudaNode*, int>> stack;
src/taskflow/cuda/cuda_graph.hpp:        os << "subgraph cluster_p" << root << " {\nlabel=\"cudaFlow: ";
src/taskflow/cuda/cuda_graph.hpp:        os << "digraph cudaFlow {\n";
src/taskflow/cuda/cuda_graph.hpp:      os << "subgraph cluster_p" << parent << " {\nlabel=\"cudaSubflow: ";
src/taskflow/cuda/cuda_graph.hpp:        case cudaNode::KERNEL:
src/taskflow/cuda/cuda_graph.hpp:        case cudaNode::SUBFLOW:
src/taskflow/cuda/cuda_graph.hpp:            &std::get<cudaNode::Subflow>(v->_handle).graph, v, l+1)
src/taskflow/cuda/cuda_stream.hpp:#include "cuda_pool.hpp"
src/taskflow/cuda/cuda_stream.hpp:@file cuda_stream.hpp
src/taskflow/cuda/cuda_stream.hpp:@brief CUDA stream utilities include file
src/taskflow/cuda/cuda_stream.hpp:// cudaStreamCreator and cudaStreamDeleter for per-thread stream pool
src/taskflow/cuda/cuda_stream.hpp:struct cudaStreamCreator {
src/taskflow/cuda/cuda_stream.hpp:  @brief operator to create a CUDA stream
src/taskflow/cuda/cuda_stream.hpp:  cudaStream_t operator () () const {
src/taskflow/cuda/cuda_stream.hpp:    cudaStream_t stream;
src/taskflow/cuda/cuda_stream.hpp:    TF_CHECK_CUDA(cudaStreamCreate(&stream), "failed to create a CUDA stream");
src/taskflow/cuda/cuda_stream.hpp:struct cudaStreamDeleter {
src/taskflow/cuda/cuda_stream.hpp:  @brief operator to destroy a CUDA stream
src/taskflow/cuda/cuda_stream.hpp:  void operator () (cudaStream_t stream) const {
src/taskflow/cuda/cuda_stream.hpp:    cudaStreamDestroy(stream);
src/taskflow/cuda/cuda_stream.hpp:using cudaPerThreadStreamPool = cudaPerThreadDeviceObjectPool<
src/taskflow/cuda/cuda_stream.hpp:  cudaStream_t, cudaStreamCreator, cudaStreamDeleter
src/taskflow/cuda/cuda_stream.hpp:@brief acquires the per-thread cuda stream pool
src/taskflow/cuda/cuda_stream.hpp:inline cudaPerThreadStreamPool& cuda_per_thread_stream_pool() {
src/taskflow/cuda/cuda_stream.hpp:  thread_local cudaPerThreadStreamPool pool;
src/taskflow/cuda/cuda_stream.hpp:// cudaScopedPerThreadStream definition
src/taskflow/cuda/cuda_stream.hpp:  tf::cudaScopedPerThreadStream stream(1);  // acquires a stream on device 1
src/taskflow/cuda/cuda_stream.hpp:  // use stream as a normal cuda stream (cudaStream_t)
src/taskflow/cuda/cuda_stream.hpp:  cudaStreamWaitEvent(stream, ...);
src/taskflow/cuda/cuda_stream.hpp:CUDA tasks (e.g., tf::cudaFlow, tf::cudaFlowCapturer).
src/taskflow/cuda/cuda_stream.hpp:%cudaScopedPerThreadStream is non-copyable.
src/taskflow/cuda/cuda_stream.hpp:class cudaScopedPerThreadStream {
src/taskflow/cuda/cuda_stream.hpp:  explicit cudaScopedPerThreadStream(int device) : 
src/taskflow/cuda/cuda_stream.hpp:    _ptr {cuda_per_thread_stream_pool().acquire(device)} {
src/taskflow/cuda/cuda_stream.hpp:  cudaScopedPerThreadStream() : 
src/taskflow/cuda/cuda_stream.hpp:    _ptr {cuda_per_thread_stream_pool().acquire(cuda_get_device())} {
src/taskflow/cuda/cuda_stream.hpp:  ~cudaScopedPerThreadStream() {
src/taskflow/cuda/cuda_stream.hpp:      cuda_per_thread_stream_pool().release(std::move(_ptr));
src/taskflow/cuda/cuda_stream.hpp:  @brief implicit conversion to the native CUDA stream (cudaStream_t)
src/taskflow/cuda/cuda_stream.hpp:  operator cudaStream_t () const {
src/taskflow/cuda/cuda_stream.hpp:  cudaScopedPerThreadStream(const cudaScopedPerThreadStream&) = delete;
src/taskflow/cuda/cuda_stream.hpp:  cudaScopedPerThreadStream(cudaScopedPerThreadStream&&) = default;
src/taskflow/cuda/cuda_stream.hpp:  cudaScopedPerThreadStream& operator = (const cudaScopedPerThreadStream&) = delete;
src/taskflow/cuda/cuda_stream.hpp:  cudaScopedPerThreadStream& operator = (cudaScopedPerThreadStream&&) = delete;
src/taskflow/cuda/cuda_stream.hpp:  std::shared_ptr<cudaPerThreadStreamPool::Object> _ptr;
src/taskflow/cuda/cuda_stream.hpp:// cudaStreamCreator and cudaStreamDeleter for per-thread event pool
src/taskflow/cuda/cuda_stream.hpp:struct cudaEventCreator {
src/taskflow/cuda/cuda_stream.hpp:  @brief operator to create a CUDA event
src/taskflow/cuda/cuda_stream.hpp:  cudaEvent_t operator () () const {
src/taskflow/cuda/cuda_stream.hpp:    cudaEvent_t event;
src/taskflow/cuda/cuda_stream.hpp:    TF_CHECK_CUDA(cudaEventCreate(&event), "failed to create a CUDA event");
src/taskflow/cuda/cuda_stream.hpp:struct cudaEventDeleter {
src/taskflow/cuda/cuda_stream.hpp:  @brief operator to destroy a CUDA event
src/taskflow/cuda/cuda_stream.hpp:  void operator () (cudaEvent_t event) const {
src/taskflow/cuda/cuda_stream.hpp:    cudaEventDestroy(event);
src/taskflow/cuda/cuda_stream.hpp:using cudaPerThreadEventPool = cudaPerThreadDeviceObjectPool<
src/taskflow/cuda/cuda_stream.hpp:  cudaEvent_t, cudaEventCreator, cudaEventDeleter
src/taskflow/cuda/cuda_stream.hpp:@brief per-thread cuda event pool
src/taskflow/cuda/cuda_stream.hpp:inline cudaPerThreadEventPool& cuda_per_thread_event_pool() {
src/taskflow/cuda/cuda_stream.hpp:  thread_local cudaPerThreadEventPool pool;
src/taskflow/cuda/cuda_stream.hpp:// cudaScopedPerThreadEvent definition
src/taskflow/cuda/cuda_stream.hpp:  tf::cudaScopedPerThreadEvent event(1);  // acquires a event on device 1
src/taskflow/cuda/cuda_stream.hpp:  // use event as a normal cuda event (cudaEvent_t)
src/taskflow/cuda/cuda_stream.hpp:  cudaStreamWaitEvent(stream, event);
src/taskflow/cuda/cuda_stream.hpp:CUDA tasks (e.g., tf::cudaFlow, tf::cudaFlowCapturer).
src/taskflow/cuda/cuda_stream.hpp:%cudaScopedPerThreadEvent is non-copyable.
src/taskflow/cuda/cuda_stream.hpp:class cudaScopedPerThreadEvent {
src/taskflow/cuda/cuda_stream.hpp:  explicit cudaScopedPerThreadEvent(int device) : 
src/taskflow/cuda/cuda_stream.hpp:    _ptr {cuda_per_thread_event_pool().acquire(device)} {
src/taskflow/cuda/cuda_stream.hpp:  cudaScopedPerThreadEvent() : 
src/taskflow/cuda/cuda_stream.hpp:    _ptr {cuda_per_thread_event_pool().acquire(cuda_get_device())} {
src/taskflow/cuda/cuda_stream.hpp:  ~cudaScopedPerThreadEvent() {
src/taskflow/cuda/cuda_stream.hpp:      cuda_per_thread_event_pool().release(std::move(_ptr));
src/taskflow/cuda/cuda_stream.hpp:  @brief implicit conversion to the native CUDA event (cudaEvent_t)
src/taskflow/cuda/cuda_stream.hpp:  operator cudaEvent_t () const {
src/taskflow/cuda/cuda_stream.hpp:  cudaScopedPerThreadEvent(const cudaScopedPerThreadEvent&) = delete;
src/taskflow/cuda/cuda_stream.hpp:  cudaScopedPerThreadEvent(cudaScopedPerThreadEvent&&) = default;
src/taskflow/cuda/cuda_stream.hpp:  cudaScopedPerThreadEvent& operator = (const cudaScopedPerThreadEvent&) = delete;
src/taskflow/cuda/cuda_stream.hpp:  cudaScopedPerThreadEvent& operator = (cudaScopedPerThreadEvent&&) = delete;
src/taskflow/cuda/cuda_stream.hpp:  std::shared_ptr<cudaPerThreadEventPool::Object> _ptr;
src/taskflow/cuda/cuda_task.hpp:#include "cuda_graph.hpp"
src/taskflow/cuda/cuda_task.hpp:@file cuda_task.hpp
src/taskflow/cuda/cuda_task.hpp:@brief cudaTask include file
src/taskflow/cuda/cuda_task.hpp:// cudaTask Types
src/taskflow/cuda/cuda_task.hpp:@enum cudaTaskType
src/taskflow/cuda/cuda_task.hpp:@brief enumeration of all %cudaTask types
src/taskflow/cuda/cuda_task.hpp:enum class cudaTaskType : int {
src/taskflow/cuda/cuda_task.hpp:@brief convert a cuda_task type to a human-readable string
src/taskflow/cuda/cuda_task.hpp:constexpr const char* to_string(cudaTaskType type) {
src/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::EMPTY:   return "empty";
src/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::HOST:    return "host";
src/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::MEMSET:  return "memset";
src/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::MEMCPY:  return "memcpy";
src/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::KERNEL:  return "kernel";
src/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::SUBFLOW: return "subflow";
src/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::CAPTURE: return "capture";
src/taskflow/cuda/cuda_task.hpp:// cudaTask 
src/taskflow/cuda/cuda_task.hpp:@class cudaTask
src/taskflow/cuda/cuda_task.hpp:@brief handle to a node of the internal CUDA graph
src/taskflow/cuda/cuda_task.hpp:class cudaTask {
src/taskflow/cuda/cuda_task.hpp:  friend class cudaFlow;
src/taskflow/cuda/cuda_task.hpp:  friend class cudaFlowCapturer;
src/taskflow/cuda/cuda_task.hpp:  friend class cudaFlowCapturerBase;
src/taskflow/cuda/cuda_task.hpp:  friend std::ostream& operator << (std::ostream&, const cudaTask&);
src/taskflow/cuda/cuda_task.hpp:    @brief constructs an empty cudaTask
src/taskflow/cuda/cuda_task.hpp:    cudaTask() = default;
src/taskflow/cuda/cuda_task.hpp:    @brief copy-constructs a cudaTask
src/taskflow/cuda/cuda_task.hpp:    cudaTask(const cudaTask&) = default;
src/taskflow/cuda/cuda_task.hpp:    @brief copy-assigns a cudaTask
src/taskflow/cuda/cuda_task.hpp:    cudaTask& operator = (const cudaTask&) = default;
src/taskflow/cuda/cuda_task.hpp:    cudaTask& precede(Ts&&... tasks);
src/taskflow/cuda/cuda_task.hpp:    cudaTask& succeed(Ts&&... tasks);
src/taskflow/cuda/cuda_task.hpp:    cudaTask& name(const std::string& name);
src/taskflow/cuda/cuda_task.hpp:    @brief queries if the task is associated with a cudaNode
src/taskflow/cuda/cuda_task.hpp:    cudaTaskType type() const;
src/taskflow/cuda/cuda_task.hpp:    cudaTask(cudaNode*);
src/taskflow/cuda/cuda_task.hpp:    cudaNode* _node {nullptr};
src/taskflow/cuda/cuda_task.hpp:inline cudaTask::cudaTask(cudaNode* node) : _node {node} {
src/taskflow/cuda/cuda_task.hpp:cudaTask& cudaTask::precede(Ts&&... tasks) {
src/taskflow/cuda/cuda_task.hpp:cudaTask& cudaTask::succeed(Ts&&... tasks) {
src/taskflow/cuda/cuda_task.hpp:inline bool cudaTask::empty() const {
src/taskflow/cuda/cuda_task.hpp:inline cudaTask& cudaTask::name(const std::string& name) {
src/taskflow/cuda/cuda_task.hpp:inline const std::string& cudaTask::name() const {
src/taskflow/cuda/cuda_task.hpp:inline size_t cudaTask::num_successors() const {
src/taskflow/cuda/cuda_task.hpp:inline cudaTaskType cudaTask::type() const {
src/taskflow/cuda/cuda_task.hpp:    case cudaNode::EMPTY:   return cudaTaskType::HOST;
src/taskflow/cuda/cuda_task.hpp:    case cudaNode::MEMSET:  return cudaTaskType::MEMSET;
src/taskflow/cuda/cuda_task.hpp:    case cudaNode::MEMCPY:  return cudaTaskType::MEMCPY;
src/taskflow/cuda/cuda_task.hpp:    case cudaNode::KERNEL:  return cudaTaskType::KERNEL;
src/taskflow/cuda/cuda_task.hpp:    case cudaNode::SUBFLOW: return cudaTaskType::SUBFLOW;
src/taskflow/cuda/cuda_task.hpp:    case cudaNode::CAPTURE: return cudaTaskType::CAPTURE;
src/taskflow/cuda/cuda_task.hpp:    default:                return cudaTaskType::UNDEFINED;
src/taskflow/cuda/cuda_task.hpp:void cudaTask::dump(T& os) const {
src/taskflow/cuda/cuda_task.hpp:  os << "cudaTask ";
src/taskflow/cuda/cuda_task.hpp:@brief overload of ostream inserter operator for cudaTask
src/taskflow/cuda/cuda_task.hpp:inline std::ostream& operator << (std::ostream& os, const cudaTask& ct) {
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:#include "cuda_transpose.hpp"
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:#include "cuda_matmul.hpp"
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:// cudaBLAF definition
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:@brief basic linear algebra flow on top of cudaFlow
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:class cudaBLAF {
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:    @param cudaflow a cudaflow object
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:    cudaBLAF(cudaFlow& cudaflow);
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:    @return cudaTask handle
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:    cudaTask transpose(const T* d_in, T* d_out, size_t rows, size_t cols);
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:    cudaTask matmul(const T* A, const T* B, T* C, size_t M, size_t K, size_t N);
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:    void update_transpose(cudaTask ct, const T* d_in, T* d_out, size_t rows, size_t cols);
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:    void update_matmul(cudaTask ct, const T* A, const T* B, T* C, size_t M, size_t K, size_t N);
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:    cudaFlow& _cf;
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:inline cudaBLAF::cudaBLAF(cudaFlow& cf) : _cf{cf} {
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:cudaTask cudaBLAF::transpose(const T* d_in, T* d_out, size_t rows, size_t cols) {
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:    cuda_transpose<T>,
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:cudaTask cudaBLAF::matmul(const T* A, const T* B, T* C, size_t M, size_t K, size_t N) {
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:    cuda_matmul<T>,
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:void cudaBLAF::update_transpose(cudaTask ct, const T* d_in, T* d_out, size_t rows, size_t cols) {
src/taskflow/cuda/cuda_algorithm/cuda_blaf.hpp:void cudaBLAF::update_matmul(cudaTask ct, const T* A, const T* B, T* C, size_t M, size_t K, size_t N) {
src/taskflow/cuda/cuda_algorithm/cuda_reduce.hpp:#include "../cuda_error.hpp"
src/taskflow/cuda/cuda_algorithm/cuda_reduce.hpp:__device__ void cuda_warp_reduce(
src/taskflow/cuda/cuda_algorithm/cuda_reduce.hpp:// Kernel: cuda_reduce
src/taskflow/cuda/cuda_algorithm/cuda_reduce.hpp:__global__ void cuda_reduce(I first, size_t N, T* res, C op) {
src/taskflow/cuda/cuda_algorithm/cuda_reduce.hpp:  cudaSharedMemory<T> shared_memory;
src/taskflow/cuda/cuda_algorithm/cuda_reduce.hpp:    cuda_warp_reduce(shm, N, tid, op);
src/taskflow/cuda/cuda_algorithm/cuda_reduce.hpp://__device__ void cuda_warp_reduce(
src/taskflow/cuda/cuda_algorithm/cuda_reduce.hpp://__global__ void cuda_reduce(int* din, int* dout, size_t N, C op) {
src/taskflow/cuda/cuda_algorithm/cuda_for_each.hpp:#include "../cuda_error.hpp"
src/taskflow/cuda/cuda_algorithm/cuda_for_each.hpp:__global__ void cuda_single_task(C callable) {
src/taskflow/cuda/cuda_algorithm/cuda_for_each.hpp:__global__ void cuda_for_each(I first, size_t N, F op) {
src/taskflow/cuda/cuda_algorithm/cuda_for_each.hpp:__global__ void cuda_for_each_index(I beg, I inc, size_t N, F op) {
src/taskflow/cuda/cuda_algorithm/cuda_transpose.hpp:#include "../cuda_error.hpp"
src/taskflow/cuda/cuda_algorithm/cuda_transpose.hpp:__global__ void cuda_transpose(
src/taskflow/cuda/cuda_algorithm/cuda_transform.hpp:#include "../cuda_error.hpp"
src/taskflow/cuda/cuda_algorithm/cuda_transform.hpp:__global__ void cuda_transform(I first, size_t N, F op, S... srcs) {
src/taskflow/cuda/cuda_algorithm/cuda_matmul.hpp:#include "../cuda_error.hpp"
src/taskflow/cuda/cuda_algorithm/cuda_matmul.hpp:__global__ void cuda_matmul(
src/taskflow/cuda/cuda_capturer.hpp:#include "cuda_task.hpp"
src/taskflow/cuda/cuda_capturer.hpp:#include "cuda_algorithm/cuda_for_each.hpp"
src/taskflow/cuda/cuda_capturer.hpp:#include "cuda_algorithm/cuda_transform.hpp"
src/taskflow/cuda/cuda_capturer.hpp:#include "cuda_algorithm/cuda_reduce.hpp"
src/taskflow/cuda/cuda_capturer.hpp:#include "cuda_optimizer.hpp"
src/taskflow/cuda/cuda_capturer.hpp:@file cuda_capturer.hpp
src/taskflow/cuda/cuda_capturer.hpp:@brief %cudaFlow capturer include file
src/taskflow/cuda/cuda_capturer.hpp:constexpr size_t cuda_default_max_threads_per_block() {
src/taskflow/cuda/cuda_capturer.hpp:constexpr size_t cuda_default_threads_per_block(size_t N) {
src/taskflow/cuda/cuda_capturer.hpp:    return std::min(cuda_default_max_threads_per_block(), next_pow2(N));
src/taskflow/cuda/cuda_capturer.hpp:// class definition: cudaFlowCapturerBase
src/taskflow/cuda/cuda_capturer.hpp:@class cudaFlowCapturerBase
src/taskflow/cuda/cuda_capturer.hpp:@brief base class to construct a CUDA task graph through stream capture
src/taskflow/cuda/cuda_capturer.hpp:class cudaFlowCapturerBase {
src/taskflow/cuda/cuda_capturer.hpp:  friend class cudaFlowCapturer;
src/taskflow/cuda/cuda_capturer.hpp:    cudaFlowCapturerBase() = default;
src/taskflow/cuda/cuda_capturer.hpp:    virtual ~cudaFlowCapturerBase() = default;
src/taskflow/cuda/cuda_capturer.hpp:    cudaFlowCapturer* factory() const;
src/taskflow/cuda/cuda_capturer.hpp:    cudaFlowCapturer* _factory {nullptr};
src/taskflow/cuda/cuda_capturer.hpp:inline cudaFlowCapturer* cudaFlowCapturerBase::factory() const {
src/taskflow/cuda/cuda_capturer.hpp:// class definition: cudaFlowCapturer
src/taskflow/cuda/cuda_capturer.hpp:@class cudaFlowCapturer
src/taskflow/cuda/cuda_capturer.hpp:@brief class for building a CUDA task dependency graph through stream capture
src/taskflow/cuda/cuda_capturer.hpp:A %cudaFlowCapturer inherits all the base methods from tf::cudaFlowCapturerBase 
src/taskflow/cuda/cuda_capturer.hpp:to construct a CUDA task graph through <i>stream capturer</i>. 
src/taskflow/cuda/cuda_capturer.hpp:This class also defines a factory interface tf::cudaFlowCapturer::make_capturer 
src/taskflow/cuda/cuda_capturer.hpp:The usage of tf::cudaFlowCapturer is similar to tf::cudaFlow, except users can
src/taskflow/cuda/cuda_capturer.hpp:call the method tf::cudaFlowCapturer::on to capture a sequence of asynchronous 
src/taskflow/cuda/cuda_capturer.hpp:CUDA operations through the given stream.
src/taskflow/cuda/cuda_capturer.hpp:The following example creates a CUDA graph that captures two kernel tasks,
src/taskflow/cuda/cuda_capturer.hpp:taskflow.emplace([](tf::cudaFlowCapturer& capturer){
src/taskflow/cuda/cuda_capturer.hpp:  auto task_1 = capturer.on([&](cudaStream_t stream){ 
src/taskflow/cuda/cuda_capturer.hpp:  auto task_2 = capturer.on([&](cudaStream_t stream){ 
src/taskflow/cuda/cuda_capturer.hpp:Similar to tf::cudaFlow, a %cudaFlowCapturer is a task (tf::Task) 
src/taskflow/cuda/cuda_capturer.hpp:That is, the callable that describes a %cudaFlowCapturer 
src/taskflow/cuda/cuda_capturer.hpp:Inside a %cudaFlow capturer task, different GPU tasks (tf::cudaTask) may run
src/taskflow/cuda/cuda_capturer.hpp:in parallel scheduled by both our capturing algorithm and the CUDA runtime.
src/taskflow/cuda/cuda_capturer.hpp:Please refer to @ref GPUTaskingcudaFlowCapturer for details.
src/taskflow/cuda/cuda_capturer.hpp:class cudaFlowCapturer {
src/taskflow/cuda/cuda_capturer.hpp:  friend class cudaFlow;
src/taskflow/cuda/cuda_capturer.hpp:    cudaGraph graph;
src/taskflow/cuda/cuda_capturer.hpp:    cudaSequentialCapturing,
src/taskflow/cuda/cuda_capturer.hpp:    cudaRoundRobinCapturing
src/taskflow/cuda/cuda_capturer.hpp:    //cudaGreedyCapturing
src/taskflow/cuda/cuda_capturer.hpp:    @brief constrcts a standalone cudaFlowCapturer
src/taskflow/cuda/cuda_capturer.hpp:    A standalone %cudaFlow capturer does not go through any taskflow and
src/taskflow/cuda/cuda_capturer.hpp:    (e.g., tf::cudaFlow::offload).
src/taskflow/cuda/cuda_capturer.hpp:    cudaFlowCapturer();
src/taskflow/cuda/cuda_capturer.hpp:    @brief destructs the cudaFlowCapturer
src/taskflow/cuda/cuda_capturer.hpp:    virtual ~cudaFlowCapturer();
src/taskflow/cuda/cuda_capturer.hpp:    @brief creates a custom capturer derived from tf::cudaFlowCapturerBase
src/taskflow/cuda/cuda_capturer.hpp:    Each %cudaFlow capturer keeps a list of custom capturers
src/taskflow/cuda/cuda_capturer.hpp:    a user-described %cudaFlow:
src/taskflow/cuda/cuda_capturer.hpp:      + tf::cudaSequentialCapturing
src/taskflow/cuda/cuda_capturer.hpp:      + tf::cudaRoundRobinCapturing
src/taskflow/cuda/cuda_capturer.hpp:    @brief captures a sequential CUDA operations from the given callable
src/taskflow/cuda/cuda_capturer.hpp:    @tparam C callable type constructible with @c std::function<void(cudaStream_t)>
src/taskflow/cuda/cuda_capturer.hpp:    @param callable a callable to capture CUDA operations with the stream
src/taskflow/cuda/cuda_capturer.hpp:    a sequence of CUDA operations defined in the callable.
src/taskflow/cuda/cuda_capturer.hpp:      std::is_invocable_r_v<void, C, cudaStream_t>, void>* = nullptr
src/taskflow/cuda/cuda_capturer.hpp:    cudaTask on(C&& callable);
src/taskflow/cuda/cuda_capturer.hpp:    The method captures a @c cudaMemcpyAsync operation through an 
src/taskflow/cuda/cuda_capturer.hpp:    cudaTask memcpy(void* dst, const void* src, size_t count);
src/taskflow/cuda/cuda_capturer.hpp:    @return cudaTask handle
src/taskflow/cuda/cuda_capturer.hpp:    to a target location. Direction can be arbitrary among CPUs and GPUs.
src/taskflow/cuda/cuda_capturer.hpp:    cudaTask copy(T* tgt, const T* src, size_t num);
src/taskflow/cuda/cuda_capturer.hpp:    @brief initializes or sets GPU memory to the given value byte by byte
src/taskflow/cuda/cuda_capturer.hpp:    @param ptr pointer to GPU mempry
src/taskflow/cuda/cuda_capturer.hpp:    The method captures a @c cudaMemsetAsync operation through an
src/taskflow/cuda/cuda_capturer.hpp:    cudaTask memset(void* ptr, int v, size_t n);
src/taskflow/cuda/cuda_capturer.hpp:    @return cudaTask handle
src/taskflow/cuda/cuda_capturer.hpp:    cudaTask kernel(dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args);
src/taskflow/cuda/cuda_capturer.hpp:    cudaTask single_task(C&& callable);
src/taskflow/cuda/cuda_capturer.hpp:    @return cudaTask handle
src/taskflow/cuda/cuda_capturer.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
src/taskflow/cuda/cuda_capturer.hpp:    cudaTask for_each(I first, I last, C&& callable);
src/taskflow/cuda/cuda_capturer.hpp:    @return cudaTask handle
src/taskflow/cuda/cuda_capturer.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
src/taskflow/cuda/cuda_capturer.hpp:    cudaTask for_each_index(I first, I last, I step, C&& callable);
src/taskflow/cuda/cuda_capturer.hpp:    @return cudaTask handle
src/taskflow/cuda/cuda_capturer.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
src/taskflow/cuda/cuda_capturer.hpp:    cudaTask transform(I first, I last, C&& callable, S... srcs);
src/taskflow/cuda/cuda_capturer.hpp:    @return a tf::cudaTask handle
src/taskflow/cuda/cuda_capturer.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
src/taskflow/cuda/cuda_capturer.hpp:    cudaTask reduce(I first, I last, T* result, C&& op);
src/taskflow/cuda/cuda_capturer.hpp:    @brief similar to tf::cudaFlowCapturerBase::reduce but does not assum 
src/taskflow/cuda/cuda_capturer.hpp:    on a GPU:
src/taskflow/cuda/cuda_capturer.hpp:    cudaTask uninitialized_reduce(I first, I last, T* result, C&& op);
src/taskflow/cuda/cuda_capturer.hpp:    @brief rebinds a capture task to another sequential CUDA operations
src/taskflow/cuda/cuda_capturer.hpp:    The method is similar to cudaFlowCapturerBase::on but with an additional 
src/taskflow/cuda/cuda_capturer.hpp:      std::is_invocable_r_v<void, C, cudaStream_t>, void>* = nullptr
src/taskflow/cuda/cuda_capturer.hpp:    cudaTask rebind_on(cudaTask task, C&& callable);
src/taskflow/cuda/cuda_capturer.hpp:    The method is similar to cudaFlowCapturerBase::memcpy but with an additional 
src/taskflow/cuda/cuda_capturer.hpp:    cudaTask rebind_memcpy(cudaTask task, void* dst, const void* src, size_t count);
src/taskflow/cuda/cuda_capturer.hpp:    The method is similar to cudaFlowCapturerBase::copy but with an additional 
src/taskflow/cuda/cuda_capturer.hpp:    cudaTask rebind_copy(cudaTask task, T* tgt, const T* src, size_t num);
src/taskflow/cuda/cuda_capturer.hpp:    The method is similar to cudaFlowCapturerBase::memset but with an additional 
src/taskflow/cuda/cuda_capturer.hpp:    cudaTask rebind_memset(cudaTask task, void* ptr, int value, size_t n);
src/taskflow/cuda/cuda_capturer.hpp:    The method is similar to cudaFlowCapturerBase::kernel but with an additional 
src/taskflow/cuda/cuda_capturer.hpp:    cudaTask rebind_kernel(cudaTask task, dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args);
src/taskflow/cuda/cuda_capturer.hpp:    @brief offloads the captured %cudaFlow onto a GPU and repeatedly runs it until 
src/taskflow/cuda/cuda_capturer.hpp:    Immediately offloads the %cudaFlow captured so far onto a GPU and
src/taskflow/cuda/cuda_capturer.hpp:    By default, if users do not offload the %cudaFlow capturer, 
src/taskflow/cuda/cuda_capturer.hpp:    @brief offloads the captured %cudaFlow and executes it by the given times
src/taskflow/cuda/cuda_capturer.hpp:    @brief offloads the captured %cudaFlow and executes it once
src/taskflow/cuda/cuda_capturer.hpp:    cudaGraph& _graph;
src/taskflow/cuda/cuda_capturer.hpp:    cudaGraphExec_t _executable {nullptr};
src/taskflow/cuda/cuda_capturer.hpp:    std::vector<std::unique_ptr<cudaFlowCapturerBase>> _capturers;
src/taskflow/cuda/cuda_capturer.hpp:    cudaFlowCapturer(cudaGraph&);
src/taskflow/cuda/cuda_capturer.hpp:    cudaGraph_t _capture();
src/taskflow/cuda/cuda_capturer.hpp:// constructs a cudaFlow capturer from a taskflow
src/taskflow/cuda/cuda_capturer.hpp:inline cudaFlowCapturer::cudaFlowCapturer(cudaGraph& g) :
src/taskflow/cuda/cuda_capturer.hpp:// constructs a standalone cudaFlow capturer
src/taskflow/cuda/cuda_capturer.hpp:inline cudaFlowCapturer::cudaFlowCapturer() : 
src/taskflow/cuda/cuda_capturer.hpp:inline cudaFlowCapturer::~cudaFlowCapturer() {
src/taskflow/cuda/cuda_capturer.hpp:    cudaGraphExecDestroy(_executable);
src/taskflow/cuda/cuda_capturer.hpp://inline void cudaFlowCapturer::_create_executable() {
src/taskflow/cuda/cuda_capturer.hpp://  TF_CHECK_CUDA(
src/taskflow/cuda/cuda_capturer.hpp://    cudaGraphInstantiate(
src/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::dump(std::ostream& os) const {
src/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::_destroy_executable() {
src/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_capturer.hpp:      cudaGraphExecDestroy(_executable), "failed to destroy executable graph"
src/taskflow/cuda/cuda_capturer.hpp:  std::is_invocable_r_v<void, C, cudaStream_t>, void>*
src/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::on(C&& callable) {
src/taskflow/cuda/cuda_capturer.hpp:    std::in_place_type_t<cudaNode::Capture>{}, std::forward<C>(callable)
src/taskflow/cuda/cuda_capturer.hpp:  return cudaTask(node);
src/taskflow/cuda/cuda_capturer.hpp:inline cudaTask cudaFlowCapturer::memcpy(
src/taskflow/cuda/cuda_capturer.hpp:  return on([dst, src, count] (cudaStream_t stream) mutable {
src/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_capturer.hpp:      cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream),
src/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::copy(T* tgt, const T* src, size_t num) {
src/taskflow/cuda/cuda_capturer.hpp:  return on([tgt, src, num] (cudaStream_t stream) mutable {
src/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_capturer.hpp:      cudaMemcpyAsync(tgt, src, sizeof(T)*num, cudaMemcpyDefault, stream),
src/taskflow/cuda/cuda_capturer.hpp:inline cudaTask cudaFlowCapturer::memset(void* ptr, int v, size_t n) {
src/taskflow/cuda/cuda_capturer.hpp:  return on([ptr, v, n] (cudaStream_t stream) mutable {
src/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_capturer.hpp:      cudaMemsetAsync(ptr, v, n, stream), "failed to capture memset"
src/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::kernel(
src/taskflow/cuda/cuda_capturer.hpp:  return on([g, b, s, f, args...] (cudaStream_t stream) mutable {
src/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::single_task(C&& callable) {
src/taskflow/cuda/cuda_capturer.hpp:  return on([c=std::forward<C>(callable)] (cudaStream_t stream) mutable {
src/taskflow/cuda/cuda_capturer.hpp:    cuda_single_task<C><<<1, 1, 0, stream>>>(c);
src/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::for_each(I first, I last, C&& c) {
src/taskflow/cuda/cuda_capturer.hpp:  return on([first, last, c=std::forward<C>(c)](cudaStream_t stream) mutable {
src/taskflow/cuda/cuda_capturer.hpp:    size_t B = cuda_default_threads_per_block(N);
src/taskflow/cuda/cuda_capturer.hpp:    cuda_for_each<I, C><<<(N+B-1)/B, B, 0, stream>>>(first, N, c);
src/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::for_each_index(I beg, I end, I inc, C&& c) {
src/taskflow/cuda/cuda_capturer.hpp:  return on([beg, end, inc, c=std::forward<C>(c)] (cudaStream_t stream) mutable {
src/taskflow/cuda/cuda_capturer.hpp:    size_t B = cuda_default_threads_per_block(N);
src/taskflow/cuda/cuda_capturer.hpp:    cuda_for_each_index<I, C><<<(N+B-1)/B, B, 0, stream>>>(beg, inc, N, c);
src/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::transform(I first, I last, C&& c, S... srcs) {
src/taskflow/cuda/cuda_capturer.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cuda_capturer.hpp:    size_t B = cuda_default_threads_per_block(N);
src/taskflow/cuda/cuda_capturer.hpp:    cuda_transform<I, C, S...><<<(N+B-1)/B, B, 0, stream>>>(first, N, c, srcs...);
src/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::reduce(I first, I last, T* result, C&& c) {
src/taskflow/cuda/cuda_capturer.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cuda_capturer.hpp:    size_t B = cuda_default_threads_per_block(N);
src/taskflow/cuda/cuda_capturer.hpp:    cuda_reduce<I, T, C, false><<<1, B, B*sizeof(T), stream>>>(
src/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::uninitialized_reduce(
src/taskflow/cuda/cuda_capturer.hpp:  (cudaStream_t stream) mutable {
src/taskflow/cuda/cuda_capturer.hpp:    size_t B = cuda_default_threads_per_block(N);
src/taskflow/cuda/cuda_capturer.hpp:    cuda_reduce<I, T, C, true><<<1, B, B*sizeof(T), stream>>>(
src/taskflow/cuda/cuda_capturer.hpp:void cudaFlowCapturer::offload_until(P&& predicate) {
src/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_capturer.hpp:      cudaGraphInstantiate(
src/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(cudaGraphDestroy(captured), "failed to destroy captured graph");
src/taskflow/cuda/cuda_capturer.hpp:  cudaScopedPerThreadStream s;
src/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_capturer.hpp:      cudaGraphLaunch(_executable, s), "failed to launch the exec graph"
src/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(cudaStreamSynchronize(s), "failed to synchronize stream");
src/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::offload_n(size_t n) {
src/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::offload() {
src/taskflow/cuda/cuda_capturer.hpp:inline bool cudaFlowCapturer::empty() const {
src/taskflow/cuda/cuda_capturer.hpp:  std::is_invocable_r_v<void, C, cudaStream_t>, void>* 
src/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::rebind_on(cudaTask task, C&& callable) {
src/taskflow/cuda/cuda_capturer.hpp:  if(task.type() != cudaTaskType::CAPTURE) {
src/taskflow/cuda/cuda_capturer.hpp:    throw std::runtime_error("invalid cudaTask type (must be CAPTURE)");
src/taskflow/cuda/cuda_capturer.hpp:  std::get<cudaNode::Capture>((task._node)->_handle).work = std::forward<C>(callable);
src/taskflow/cuda/cuda_capturer.hpp:inline cudaTask cudaFlowCapturer::rebind_memcpy(
src/taskflow/cuda/cuda_capturer.hpp:  cudaTask task, void* dst, const void* src, size_t count
src/taskflow/cuda/cuda_capturer.hpp:  return rebind_on(task, [dst, src, count](cudaStream_t stream) mutable {
src/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_capturer.hpp:      cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream),
src/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::rebind_copy(
src/taskflow/cuda/cuda_capturer.hpp:  cudaTask task, T* tgt, const T* src, size_t num
src/taskflow/cuda/cuda_capturer.hpp:  return rebind_on(task, [tgt, src, num] (cudaStream_t stream) mutable {
src/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_capturer.hpp:      cudaMemcpyAsync(tgt, src, sizeof(T)*num, cudaMemcpyDefault, stream),
src/taskflow/cuda/cuda_capturer.hpp:inline cudaTask cudaFlowCapturer::rebind_memset(
src/taskflow/cuda/cuda_capturer.hpp:  cudaTask task, void* ptr, int v, size_t n
src/taskflow/cuda/cuda_capturer.hpp:  return rebind_on(task, [ptr, v, n] (cudaStream_t stream) mutable {
src/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
src/taskflow/cuda/cuda_capturer.hpp:      cudaMemsetAsync(ptr, v, n, stream), "failed to capture memset"
src/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::rebind_kernel(
src/taskflow/cuda/cuda_capturer.hpp:  cudaTask task, dim3 g, dim3 b, size_t s, F&& f, ArgsT&&... args
src/taskflow/cuda/cuda_capturer.hpp:  return rebind_on(task, [g, b, s, f, args...] (cudaStream_t stream) mutable {
src/taskflow/cuda/cuda_capturer.hpp:T* cudaFlowCapturer::make_capturer(ArgsT&&... args) {
src/taskflow/cuda/cuda_capturer.hpp:  static_assert(std::is_base_of_v<cudaFlowCapturerBase, T>);
src/taskflow/cuda/cuda_capturer.hpp:inline cudaGraph_t cudaFlowCapturer::_capture() {
src/taskflow/cuda/cuda_capturer.hpp:OPT& cudaFlowCapturer::make_optimizer(ArgsT&&... args) {

```
