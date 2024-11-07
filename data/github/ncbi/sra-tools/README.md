# https://github.com/ncbi/sra-tools

```console
libs/inc/bm/sse2neon.h://   Brandon Rowlett <browlett@nvidia.com>
libs/inc/bm/sse2neon.h://   Eric van Beurden <evanbeurden@nvidia.com>
libs/inc/bm/sse2neon.h://   Alexander Potylitsin <apotylitsin@nvidia.com>
tools/external/fasterq-dump/cleanup_task.c:#ifndef _h_kproc_procmgr_
tools/external/fasterq-dump/cleanup_task.c:#include <kproc/procmgr.h>
tools/external/fasterq-dump/cleanup_task.c:    struct KProcMgr * proc_mgr;
tools/external/fasterq-dump/cleanup_task.c:    rc_t rc = KProcMgrMakeSingleton ( &proc_mgr );
tools/external/fasterq-dump/cleanup_task.c:        rc = KProcMgrAddCleanupTask ( proc_mgr, &( task -> ticket ), ( KTask * )task );
tools/external/fasterq-dump/cleanup_task.c:            InfoMsg( "CleanupTask: added to ProcManager" );
tools/external/fasterq-dump/cleanup_task.c:            rc_t rc2 = KProcMgrRelease ( proc_mgr );
tools/external/fasterq-dump/cleanup_task.c:                ErrMsg( "clt_add_task_to_proc_mgr().KProcMgrRelease() -> %R", rc2 );
tools/external/fasterq-dump/cleanup_task.c:        struct KProcMgr * proc_mgr;
tools/external/fasterq-dump/cleanup_task.c:        rc = KProcMgrMakeSingleton ( &proc_mgr );
tools/external/fasterq-dump/cleanup_task.c:            rc = KProcMgrRemoveCleanupTask ( proc_mgr, &( self -> ticket ) );
tools/external/fasterq-dump/cleanup_task.c:                ErrMsg( "clt_terminate().KProcMgrRemoveCleanupTask() -> %R", rc );
tools/external/fasterq-dump/cleanup_task.c:                rc_t rc2 = KProcMgrRelease ( proc_mgr );
tools/external/fasterq-dump/cleanup_task.c:                    ErrMsg( "clt_terminate().KProcMgrRelease() -> %R", rc2 );
tools/external/fasterq-dump/temp_dir.c:#ifndef _h_kproc_procmgr_
tools/external/fasterq-dump/temp_dir.c:#include <kproc/procmgr.h>
tools/external/fasterq-dump/temp_dir.c:    struct KProcMgr * proc_mgr;
tools/external/fasterq-dump/temp_dir.c:    rc_t rc = KProcMgrMakeSingleton ( &proc_mgr );
tools/external/fasterq-dump/temp_dir.c:        rc = KProcMgrGetPID ( proc_mgr, & self -> pid );
tools/external/fasterq-dump/temp_dir.c:            rc = KProcMgrGetHostName ( proc_mgr, self -> hostname, sizeof self -> hostname );
tools/external/fasterq-dump/temp_dir.c:        KProcMgrRelease ( proc_mgr );
tools/external/vdb-config/util.cpp:rc_t CKDirectory::CreateNonExistingPublicDir(bool verbose,
tools/external/vdb-config/util.cpp:                    rc_t r2 = dir.CreateNonExistingPublicDir
tools/external/vdb-config/tui/tui_widget_label.c:        KTUIDlgPushEvent( w -> dlg, ktuidlg_event_select, w -> id, 0, 0, NULL );
tools/external/vdb-config/tui/tui_widget.c:        KTUIDlgPushEvent( w->dlg, ev_type, w->id, value_1, value_2, ptr_0 );
tools/external/vdb-config/tui/tui_widget_tabhdr.c:        KTUIDlgPushEvent( w -> dlg, ktuidlg_event_select, w -> id, 0, 0, NULL );
tools/external/vdb-config/tui/tui_dlg.c: void KTUIDlgPushEvent( struct KTUIDlg * self,
tools/external/vdb-config/tui/tui_dlg.c:            KTUIDlgPushEvent( self, ktuidlg_event_focus_lost, has_focus->id, 0, 0, NULL );
tools/external/vdb-config/tui/tui_dlg.c:        KTUIDlgPushEvent( self, ktuidlg_event_focus, w->id, 0, 0, NULL );
tools/external/vdb-config/tui/tui_widget_grid.c:        KTUIDlgPushEvent( w->dlg, ktuidlg_event_select, w->id, 0, 0, NULL );
tools/external/vdb-config/tui/tui_menu.c:            KTUIDlgPushEvent( dlg, ktuidlg_event_menu, node->id, 0, 0, NULL );
tools/external/vdb-config/tui/tui_menu.c:                                KTUIDlgPushEvent( dlg, ktuidlg_event_menu, sub_node->id, 0, 0, NULL );
tools/external/vdb-config/tui/tui_menu.c:                KTUIDlgPushEvent( dlg, ktuidlg_event_menu, node->id, 0, 0, NULL );
tools/external/vdb-config/tui/tui_menu.c:            KTUIDlgPushEvent( dlg, ktuidlg_event_menu, node->id, 0, 0, NULL );
tools/external/vdb-config/tui/tui_menu.c:                    KTUIDlgPushEvent( dlg, ktuidlg_event_menu, node->id, 0, 0, NULL );
tools/external/vdb-config/tui/tui_dlg.h:TUI_EXTERN void KTUIDlgPushEvent( struct KTUIDlg * self,
tools/external/vdb-config/tui/tui_widget_inputline.c:        KTUIDlgPushEvent( w->dlg, ktuidlg_event_select, w->id, 0, 0, NULL );
tools/external/vdb-config/tui/tui_widget_button.c:        KTUIDlgPushEvent( w -> dlg, ktuidlg_event_select, w -> id, 0, 0, NULL );
tools/external/vdb-config/util.hpp:    rc_t CreateNonExistingPublicDir(bool verbose, const char *path, ...) const;
tools/external/vdb-config/util.hpp:    rc_t CreateNonExistingPublicDir(const std::string &path,
tools/external/vdb-config/util.hpp:    rc_t CreateNonExistingPublicDir(const CString &path, bool verbose = false)
tools/loaders/sharq/taskflow/sycl/syclflow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/sharq/taskflow/sycl/syclflow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/sharq/taskflow/sycl/syclflow.hpp:    @brief offloads the %syclFlow onto a GPU and repeatedly runs it until 
tools/loaders/sharq/taskflow/sycl/sycl_task.hpp:@brief handle to a node of the internal CUDA graph
tools/loaders/sharq/taskflow/core/declarations.hpp:// cudaFlow
tools/loaders/sharq/taskflow/core/declarations.hpp:class cudaFlowNode;
tools/loaders/sharq/taskflow/core/declarations.hpp:class cudaFlowGraph;
tools/loaders/sharq/taskflow/core/declarations.hpp:class cudaTask;
tools/loaders/sharq/taskflow/core/declarations.hpp:class cudaFlow;
tools/loaders/sharq/taskflow/core/declarations.hpp:class cudaFlowCapturer;
tools/loaders/sharq/taskflow/core/declarations.hpp:class cudaFlowOptimizerBase;
tools/loaders/sharq/taskflow/core/declarations.hpp:class cudaFlowLinearOptimizer;
tools/loaders/sharq/taskflow/core/declarations.hpp:class cudaFlowSequentialOptimizer;
tools/loaders/sharq/taskflow/core/declarations.hpp:class cudaFlowRoundRobinOptimizer;
tools/loaders/sharq/taskflow/core/executor-module-opt.hpp:    void _invoke_cudaflow_task(Worker&, Node*);
tools/loaders/sharq/taskflow/core/executor-module-opt.hpp:      std::enable_if_t<is_cudaflow_task_v<C>, void>* = nullptr
tools/loaders/sharq/taskflow/core/executor-module-opt.hpp:    void _invoke_cudaflow_task_entry(Node*, C&&);
tools/loaders/sharq/taskflow/core/executor-module-opt.hpp:    // cudaflow task
tools/loaders/sharq/taskflow/core/executor-module-opt.hpp:    case Node::CUDAFLOW: {
tools/loaders/sharq/taskflow/core/executor-module-opt.hpp:      _invoke_cudaflow_task(worker, node);
tools/loaders/sharq/taskflow/core/executor-module-opt.hpp:// Procedure: _invoke_cudaflow_task
tools/loaders/sharq/taskflow/core/executor-module-opt.hpp:inline void Executor::_invoke_cudaflow_task(Worker& worker, Node* node) {
tools/loaders/sharq/taskflow/core/executor-module-opt.hpp:  std::get_if<Node::cudaFlow>(&node->_handle)->work(*this, node);
tools/loaders/sharq/taskflow/core/taskflow.hpp:    and GPU tasks, you need to run the taskflow first before you can
tools/loaders/sharq/taskflow/taskflow.hpp:@dir taskflow/cuda
tools/loaders/sharq/taskflow/taskflow.hpp:@brief taskflow CUDA include dir
tools/loaders/sharq/taskflow/cuda/cuda_error.hpp:#include <cuda.h>
tools/loaders/sharq/taskflow/cuda/cuda_error.hpp:#define TF_CUDA_EXPAND( x ) x
tools/loaders/sharq/taskflow/cuda/cuda_error.hpp:#define TF_CUDA_REMOVE_FIRST_HELPER(N, ...) __VA_ARGS__
tools/loaders/sharq/taskflow/cuda/cuda_error.hpp:#define TF_CUDA_REMOVE_FIRST(...) TF_CUDA_EXPAND(TF_CUDA_REMOVE_FIRST_HELPER(__VA_ARGS__))
tools/loaders/sharq/taskflow/cuda/cuda_error.hpp:#define TF_CUDA_GET_FIRST_HELPER(N, ...) N
tools/loaders/sharq/taskflow/cuda/cuda_error.hpp:#define TF_CUDA_GET_FIRST(...) TF_CUDA_EXPAND(TF_CUDA_GET_FIRST_HELPER(__VA_ARGS__))
tools/loaders/sharq/taskflow/cuda/cuda_error.hpp:#define TF_CHECK_CUDA(...)                                       \
tools/loaders/sharq/taskflow/cuda/cuda_error.hpp:if(TF_CUDA_GET_FIRST(__VA_ARGS__) != cudaSuccess) {              \
tools/loaders/sharq/taskflow/cuda/cuda_error.hpp:  auto __ev__ = TF_CUDA_GET_FIRST(__VA_ARGS__);                  \
tools/loaders/sharq/taskflow/cuda/cuda_error.hpp:      << (cudaGetErrorString(__ev__)) << " ("                    \
tools/loaders/sharq/taskflow/cuda/cuda_error.hpp:      << (cudaGetErrorName(__ev__)) << ") - ";                   \
tools/loaders/sharq/taskflow/cuda/cuda_error.hpp:  tf::ostreamize(oss, TF_CUDA_REMOVE_FIRST(__VA_ARGS__));        \
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:#include "../cudaflow.hpp"
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:@file taskflow/cuda/algorithm/for_each.hpp
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:@brief cuda parallel-iteration algorithms include file
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:__global__ void cuda_for_each_kernel(I first, unsigned count, C c) {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  auto tile = cuda_get_tile(bid, nt*vt, count);
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  cuda_strided_iterate<nt, vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:__global__ void cuda_for_each_index_kernel(I first, I inc, unsigned count, C c) {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  auto tile = cuda_get_tile(bid, nt*vt, count);
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  cuda_strided_iterate<nt, vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:// cuda standard algorithms: single_task/for_each/for_each_index
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:void cuda_single_task(P&& p, C c) {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  cuda_kernel<<<1, 1, 0, p.stream()>>>(
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:on a GPU:
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:void cuda_for_each(P&& p, I first, I last, C c) {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  detail::cuda_for_each_kernel<E::nt, E::vt, I, C><<<E::num_blocks(count), E::nt, 0, p.stream()>>>(
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:the following loop on a GPU:
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:void cuda_for_each_index(P&& p, I first, I last, I inc, C c) {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  detail::cuda_for_each_index_kernel<E::nt, E::vt, I, C><<<E::num_blocks(count), E::nt, 0, p.stream()>>>(
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:__global__ void cuda_single_task(C callable) {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:cudaTask cudaFlow::single_task(C c) {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  return kernel(1, 1, 0, cuda_single_task<C>, c);
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:void cudaFlow::single_task(cudaTask task, C c) {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  return kernel(task, 1, 1, 0, cuda_single_task<C>, c);
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:cudaTask cudaFlowCapturer::single_task(C callable) {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  return on([=] (cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:    cuda_single_task(cudaDefaultExecutionPolicy(stream), callable);
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:void cudaFlowCapturer::single_task(cudaTask task, C callable) {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  on(task, [=] (cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:    cuda_single_task(cudaDefaultExecutionPolicy(stream), callable);
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:// cudaFlow: for_each, for_each_index
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:cudaTask cudaFlow::for_each(I first, I last, C c) {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  using E = cudaDefaultExecutionPolicy;
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:    detail::cuda_for_each_kernel<E::nt, E::vt, I, C>, first, count, c
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:void cudaFlow::for_each(cudaTask task, I first, I last, C c) {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  using E = cudaDefaultExecutionPolicy;
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:    detail::cuda_for_each_kernel<E::nt, E::vt, I, C>, first, count, c
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:cudaTask cudaFlow::for_each_index(I first, I last, I inc, C c) {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  using E = cudaDefaultExecutionPolicy;
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:    detail::cuda_for_each_index_kernel<E::nt, E::vt, I, C>, first, inc, count, c
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:void cudaFlow::for_each_index(cudaTask task, I first, I last, I inc, C c) {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  using E = cudaDefaultExecutionPolicy;
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:    detail::cuda_for_each_index_kernel<E::nt, E::vt, I, C>, first, inc, count, c
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:// cudaFlowCapturer: for_each, for_each_index
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:cudaTask cudaFlowCapturer::for_each(I first, I last, C c) {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  return on([=](cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:    cuda_for_each(cudaDefaultExecutionPolicy(stream), first, last, c);
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:cudaTask cudaFlowCapturer::for_each_index(I beg, I end, I inc, C c) {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  return on([=] (cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:    cuda_for_each_index(cudaDefaultExecutionPolicy(stream), beg, end, inc, c);
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:void cudaFlowCapturer::for_each(cudaTask task, I first, I last, C c) {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  on(task, [=](cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:    cuda_for_each(cudaDefaultExecutionPolicy(stream), first, last, c);
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:void cudaFlowCapturer::for_each_index(
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  cudaTask task, I beg, I end, I inc, C c
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:  on(task, [=] (cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/algorithm/for_each.hpp:    cuda_for_each_index(cudaDefaultExecutionPolicy(stream), beg, end, inc, c);
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:@file taskflow/cuda/algorithm/scan.hpp
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:@brief CUDA scan algorithm include file
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:inline constexpr unsigned cudaScanRecursionThreshold = 8;
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:enum class cudaScanType : int {
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:struct cudaScanResult {
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:struct cudaScanResult<T, vt, true> {
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  cudaArray<T, vt> scan;
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:struct cudaBlockScan {
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  const static unsigned num_warps  = nt / CUDA_WARP_SIZE;
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  __device__ cudaScanResult<T> operator ()(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    cudaScanType type = cudaScanType::EXCLUSIVE
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  __device__ cudaScanResult<T, vt> operator()(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    cudaArray<T, vt> x,
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    cudaScanType type = cudaScanType::EXCLUSIVE
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:__device__ cudaScanResult<T> cudaBlockScan<nt, T>::operator () (
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  T init, cudaScanType type
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  cuda_iterate<num_passes>([&](auto pass) {
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  cudaScanResult<T> result;
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    (cudaScanType::INCLUSIVE == type ? x :
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:__device__ cudaScanResult<T, vt> cudaBlockScan<nt, T>::operator()(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  cudaArray<T, vt> x,
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  cudaScanType type
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    cuda_iterate<vt>([&](auto i) {
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    cuda_iterate<vt>([&](auto i) {
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    (count + vt - 1) / vt, op, init, cudaScanType::EXCLUSIVE
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  cudaArray<T, vt> y;
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  cuda_iterate<vt>([&](auto i) {
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    if(cudaScanType::EXCLUSIVE == type) {
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  return cudaScanResult<T, vt> { y, result.reduction };
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:void cuda_single_pass_scan(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  cudaScanType scan_type,
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  cuda_kernel<<<1, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    using scan_t = cudaBlockScan<E::nt, T>;
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:      auto x = cuda_mem_to_reg_thread<E::nt, E::vt>(input + cur,
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:      cuda_reg_to_mem_thread<E::nt, E::vt>(result.scan, tid, count2,
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:void cuda_scan_loop(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  cudaScanType scan_type,
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  if(B > cudaScanRecursionThreshold) {
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    //cudaDeviceVector<T> partials(B);
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:      __shared__ typename cudaBlockReduce<E::nt, T>::Storage shm;
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:      auto tile = cuda_get_tile(bid, E::nv, count);
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:      auto x = cuda_mem_to_reg_strided<E::nt, E::vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:      cuda_strided_iterate<E::nt, E::vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:      auto all_reduce = cudaBlockReduce<E::nt, T>()(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    //cuda_scan_loop(p, cudaScanType::EXCLUSIVE, buffer, B, buffer, op, S);
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    cuda_scan_loop(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:      p, cudaScanType::EXCLUSIVE, buffer, B, buffer, op, buffer+B
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:      using scan_t = cudaBlockScan<E::nt, T>;
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:      auto tile = cuda_get_tile(bid, E::nv, count);
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:      auto x = cuda_mem_to_reg_thread<E::nt, E::vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:      cuda_reg_to_mem_thread<E::nt, E::vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    cuda_single_pass_scan(p, scan_type, input, count, output, op);
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:unsigned cudaExecutionPolicy<NT, VT>::scan_bufsz(unsigned count) {
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  for(auto b=B; b>detail::cudaScanRecursionThreshold; b=num_blocks(b)) {
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:void cuda_inclusive_scan(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  detail::cuda_scan_loop(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    p, detail::cudaScanType::INCLUSIVE, first, count, output, op, buf
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:void cuda_transform_inclusive_scan(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  detail::cuda_scan_loop(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    p, detail::cudaScanType::INCLUSIVE,
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    cuda_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }),
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:void cuda_exclusive_scan(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  detail::cuda_scan_loop(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    p, detail::cudaScanType::EXCLUSIVE, first, count, output, op, buf
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:void cuda_transform_exclusive_scan(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:  detail::cuda_scan_loop(
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    p, detail::cudaScanType::EXCLUSIVE,
tools/loaders/sharq/taskflow/cuda/algorithm/scan.hpp:    cuda_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }),
tools/loaders/sharq/taskflow/cuda/algorithm/matmul.hpp:#include "../cudaflow.hpp"
tools/loaders/sharq/taskflow/cuda/algorithm/matmul.hpp:__global__ void cuda_matmul(
tools/loaders/sharq/taskflow/cuda/algorithm/transpose.hpp:#include "../cuda_error.hpp"
tools/loaders/sharq/taskflow/cuda/algorithm/transpose.hpp:__global__ void cuda_transpose(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:#include "../cudaflow.hpp"
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:@file taskflow/cuda/algorithm/reduce.hpp
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:@brief cuda reduce algorithms include file
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:struct cudaBlockReduce {
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  static const unsigned group_size = std::min(nt, CUDA_WARP_SIZE);
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:    nt && (0 == nt % CUDA_WARP_SIZE),
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:    "cudaBlockReduce requires num threads to be a multiple of warp_size (32)"
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:__device__ T cudaBlockReduce<nt, T>::operator ()(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:    cuda_strided_iterate<group_size, num_items>([&](auto i, auto j) {
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  cuda_iterate<num_passes>([&](auto pass) {
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:// cuda_reduce
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:__global__ void cuda_reduce_kernel(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  __shared__ typename cudaBlockReduce<nt, U>::Storage shm;
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  auto tile = cuda_get_tile(bid, nt*vt, count);
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  auto x = cuda_mem_to_reg_strided<nt, vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  cuda_strided_iterate<nt, vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  s = cudaBlockReduce<nt, U>()(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:void cuda_reduce_loop(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  cuda_reduce_kernel<E::nt, E::vt><<<B, E::nt, 0, p.stream()>>>(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:    cuda_reduce_loop(p, buf, B, res, op, buf+B);
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:// cuda_uninitialized_reduce
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:__global__ void cuda_uninitialized_reduce_kernel(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  __shared__ typename cudaBlockReduce<nt, U>::Storage shm;
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  auto tile = cuda_get_tile(bid, nt*vt, count);
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  auto x = cuda_mem_to_reg_strided<nt, vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  cuda_strided_iterate<nt, vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  s = cudaBlockReduce<nt, U>()(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:void cuda_uninitialized_reduce_loop(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  cuda_uninitialized_reduce_kernel<E::nt, E:: vt><<<B, E::nt, 0, p.stream()>>>(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:    cuda_uninitialized_reduce_loop(p, buf, B, res, op, buf+B);
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:unsigned cudaExecutionPolicy<NT, VT>::reduce_bufsz(unsigned count) {
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:// cuda_reduce
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:void cuda_reduce(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  detail::cuda_reduce_loop(p, first, count, res, op, buf);
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:// cuda_uninitialized_reduce
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:on a GPU:
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:void cuda_uninitialized_reduce(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  detail::cuda_uninitialized_reduce_loop(p, first, count, res, op, buf);
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:void cuda_transform_reduce(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  detail::cuda_reduce_loop(p,
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:    cuda_make_load_iterator<T>([=]__device__(auto i){
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:on a GPU:
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:void cuda_uninitialized_transform_reduce(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:  detail::cuda_uninitialized_reduce_loop(p,
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp:    cuda_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }),
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp://__device__ void cuda_warp_reduce(
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp://__global__ void cuda_reduce(I first, size_t N, T* res, C op) {
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp://  cudaSharedMemory<T> shared_memory;
tools/loaders/sharq/taskflow/cuda/algorithm/reduce.hpp://    cuda_warp_reduce(shm, N, tid, op);
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:#include "../cudaflow.hpp"
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:@file taskflow/cuda/algorithm/merge.hpp
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:@brief CUDA merge algorithm include file
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:enum class cudaMergeBoundType {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:struct cudaMergePair {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cudaArray<T, N> keys;
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cudaArray<unsigned, N> indices;
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:struct cudaMergeRange {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  __device__ cudaRange a_range() const {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:    return cudaRange { a_begin, a_end };
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  __device__ cudaRange b_range() const {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:    return cudaRange { b_begin, b_end };
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  __device__ cudaMergeRange to_local() const {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:    return cudaMergeRange { 0, a_count(), a_count(), total() };
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  __device__ cudaMergeRange partition(unsigned mp0, unsigned diag) const {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:    return cudaMergeRange { a_begin + mp0, a_end, b_begin + diag - mp0, b_end };
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  __device__ cudaMergeRange partition(unsigned mp0, unsigned diag0,
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:    return cudaMergeRange {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cudaMergeBoundType bounds = cudaMergeBoundType::LOWER,
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:__device__ auto cuda_merge_path(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:    bool pred = (cudaMergeBoundType::UPPER == bounds) ?
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:template<cudaMergeBoundType bounds, typename keys_it, typename comp_t>
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:__device__ auto cuda_merge_path(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  keys_it keys, cudaMergeRange range, unsigned diag, comp_t comp
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  return cuda_merge_path<bounds>(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:template<cudaMergeBoundType bounds, bool range_check, typename T, typename comp_t>
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:__device__ bool cuda_merge_predicate(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  T a_key, T b_key, cudaMergeRange range, comp_t comp
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:    p = (cudaMergeBoundType::UPPER == bounds) ? comp(a_key, b_key) :
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:inline __device__ auto cuda_compute_merge_range(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  return cudaMergeRange { mp0, mp1, diag0 - mp0, diag1 - mp1 };
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:__device__ auto cuda_load_two_streams_reg(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cudaArray<T, vt> x;
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cuda_strided_iterate<nt, vt>([&](auto i, auto index) {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cudaArray<T, vt>
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cudaArray<T, vt> x;
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cuda_strided_iterate<nt, vt>([&](auto i, auto index) {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:__device__ void cuda_load_two_streams_shared(A a, unsigned a_count,
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  auto x = cuda_load_two_streams_reg<nt, vt, T>(a, a_count, b, b_count, tid);
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cuda_reg_to_shared_strided<nt>(x, tid, shared, sync);
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:__device__ auto cuda_gather_two_streams_strided(const T* a,
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  unsigned a_count, const T* b, unsigned b_count, cudaArray<unsigned, vt> indices,
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cudaArray<T, vt> x;
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cuda_strided_iterate<nt, vt>([&](auto i, auto j) {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cudaArray<T, vt>
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:> cuda_gather_two_streams_strided(a_it a,
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  unsigned a_count, b_it b, unsigned b_count, cudaArray<unsigned, vt> indices, unsigned tid) {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cudaArray<T, vt> x;
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cuda_strided_iterate<nt, vt>([&](auto i, auto j) {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:__device__ void cuda_transfer_two_streams_strided(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cudaArray<unsigned, vt> indices, unsigned tid, c_it c
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  auto x = cuda_gather_two_streams_strided<nt, vt, T>(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cuda_reg_to_mem_strided<nt>(x, tid, a_count + b_count, c);
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:template<cudaMergeBoundType bounds, unsigned vt, typename T, typename comp_t>
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:__device__ auto cuda_serial_merge(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  const T* keys_shared, cudaMergeRange range, comp_t comp, bool sync = true
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cudaMergePair<T, vt> merge_pair;
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cuda_iterate<vt>([&](auto i) {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:    bool p = cuda_merge_predicate<bounds, true>(a_key, b_key, range, comp);
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:template<cudaMergeBoundType bounds,
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  a_it a, b_it b, cudaMergeRange range_mem, unsigned tid, comp_t comp, T (&keys_shared)[S]
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cuda_load_two_streams_shared<nt, vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  auto mp = cuda_merge_path<bounds>(keys_shared, range_local, diag, comp);
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  auto merged = cuda_serial_merge<bounds, vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:template<cudaMergeBoundType bounds,
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:void cuda_merge_path_partitions(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cuda_kernel<<<B, nt, 0, p.stream()>>>([=]__device__(auto tid, auto bid) {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:    auto range = cuda_get_tile(bid, nt * vt, num_partitions);
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:    cuda_strided_iterate<nt, vt>([=](auto, auto j) {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:      buf[index] = cuda_merge_path<bounds>(a, a_count, b, b_count, diag, comp);
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp://  transform([=]MGPU_DEVICE(int index) {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:void cuda_merge_loop(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  auto has_values = !std::is_same<V, cudaEmpty>::value;
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cuda_merge_path_partitions<cudaMergeBoundType::LOWER>(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:    auto range = cuda_compute_merge_range(a_count, b_count, bid, E::nv, mp0, mp1);
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:    auto merge = block_merge_from_mem<cudaMergeBoundType::LOWER, E::nt, E::vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:    cuda_reg_to_mem_thread<E::nt>(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:      auto indices = cuda_reg_thread_to_strided<E::nt>(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:      cuda_transfer_two_streams_strided<E::nt>(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:unsigned cudaExecutionPolicy<NT, VT>::merge_bufsz(unsigned a_count, unsigned b_count) {
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:void cuda_merge_by_key(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  detail::cuda_merge_loop(p,
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:This function is equivalent to tf::cuda_merge_by_key without values.
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:void cuda_merge(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:  cuda_merge_by_key(
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:    a_keys_first, a_keys_last, (const cudaEmpty*)nullptr,
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:    b_keys_first, b_keys_last, (const cudaEmpty*)nullptr,
tools/loaders/sharq/taskflow/cuda/algorithm/merge.hpp:    c_keys_first, (cudaEmpty*)nullptr, comp,
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:@file taskflow/cuda/algorithm/find.hpp
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:@brief cuda find algorithms include file
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:struct cudaFindPair {
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:void cuda_find_if_loop(P&& p, I input, unsigned count, unsigned* idx, U pred) {
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:    cuda_single_task(p, [=] __device__ () { *idx = 0; });
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:  cuda_single_task(p, [=] __device__ () { *idx = count; });
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:  cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:    auto tile = cuda_get_tile(bid, E::nv, count);
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:    auto x = cuda_mem_to_reg_strided<E::nt, E::vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:    //__shared__ typename cudaBlockReduce<E::nt, unsigned>::Storage shm;
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:    //id = cudaBlockReduce<E::nt, unsigned>()(
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:    //  cuda_minimum<unsigned>{},
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:void cuda_min_element_loop(
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:    cuda_single_task(p, [=] __device__ () { *idx = 0; });
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:  using T = cudaFindPair<typename std::iterator_traits<I>::value_type>;
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:  cuda_uninitialized_reduce_loop(p,
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:    cuda_make_load_iterator<T>([=]__device__(auto i){
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:void cuda_max_element_loop(
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:    cuda_single_task(p, [=] __device__ () { *idx = 0; });
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:  using T = cudaFindPair<typename std::iterator_traits<I>::value_type>;
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:  cuda_uninitialized_reduce_loop(p,
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:    cuda_make_load_iterator<T>([=]__device__(auto i){
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:// cuda_find_if
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:void cuda_find_if(
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:  detail::cuda_find_if_loop(p, first, std::distance(first, last), idx, op);
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:// cuda_min_element
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:unsigned cudaExecutionPolicy<NT, VT>::min_element_bufsz(unsigned count) {
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:  return reduce_bufsz<detail::cudaFindPair<T>>(count);
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:tf::cuda_min_element_bufsz bytes for internal use.
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:void cuda_min_element(P&& p, I first, I last, unsigned* idx, O op, void* buf) {
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:  detail::cuda_min_element_loop(
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:// cuda_max_element
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:unsigned cudaExecutionPolicy<NT, VT>::max_element_bufsz(unsigned count) {
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:  return reduce_bufsz<detail::cudaFindPair<T>>(count);
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:tf::cuda_max_element_bufsz bytes for internal use.
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:void cuda_max_element(P&& p, I first, I last, unsigned* idx, O op, void* buf) {
tools/loaders/sharq/taskflow/cuda/algorithm/find.hpp:  detail::cuda_max_element_loop(
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:#include "../cudaflow.hpp"
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:@file taskflow/cuda/algorithm/transform.hpp
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:@brief cuda parallel-transform algorithms include file
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:__global__ void cuda_transform_kernel(I first, unsigned count, O output, C op) {
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:  auto tile = cuda_get_tile(bid, nt*vt, count);
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:  cuda_strided_iterate<nt, vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:__global__ void cuda_transform_kernel(
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:  auto tile = cuda_get_tile(bid, nt*vt, count);
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:  cuda_strided_iterate<nt, vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:// CUDA standard algorithms: transform
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:void cuda_transform(P&& p, I first, I last, O output, C op) {
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:  detail::cuda_transform_kernel<E::nt, E::vt, I, O, C>
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:void cuda_transform(
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:  detail::cuda_transform_kernel<E::nt, E::vt, I1, I2, O, C>
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:// cudaFlow
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:cudaTask cudaFlow::transform(I first, I last, O output, C c) {
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:  using E = cudaDefaultExecutionPolicy;
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:    detail::cuda_transform_kernel<E::nt, E::vt, I, O, C>,
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:cudaTask cudaFlow::transform(I1 first1, I1 last1, I2 first2, O output, C c) {
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:  using E = cudaDefaultExecutionPolicy;
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:    detail::cuda_transform_kernel<E::nt, E::vt, I1, I2, O, C>,
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:void cudaFlow::transform(cudaTask task, I first, I last, O output, C c) {
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:  using E = cudaDefaultExecutionPolicy;
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:    detail::cuda_transform_kernel<E::nt, E::vt, I, O, C>,
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:void cudaFlow::transform(
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:  cudaTask task, I1 first1, I1 last1, I2 first2, O output, C c
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:  using E = cudaDefaultExecutionPolicy;
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:    detail::cuda_transform_kernel<E::nt, E::vt, I1, I2, O, C>,
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:// cudaFlowCapturer
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:cudaTask cudaFlowCapturer::transform(I first, I last, O output, C op) {
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:  return on([=](cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:    cuda_transform(p, first, last, output, op);
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:cudaTask cudaFlowCapturer::transform(
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:  return on([=](cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:    cuda_transform(p, first1, last1, first2, output, op);
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:void cudaFlowCapturer::transform(
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:  cudaTask task, I first, I last, O output, C op
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:  on(task, [=] (cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:    cuda_transform(p, first, last, output, op);
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:void cudaFlowCapturer::transform(
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:  cudaTask task, I1 first1, I1 last1, I2 first2, O output, C op
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:  on(task, [=] (cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/sharq/taskflow/cuda/algorithm/transform.hpp:    cuda_transform(p, first1, last1, first2, output, op);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:@file taskflow/cuda/algorithm/sort.hpp
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:@brief CUDA sort algorithm include file
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:constexpr int cuda_clz(int x) {
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:constexpr int cuda_find_log2(int x, bool round_up = false) {
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  int a = 31 - cuda_clz(x);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:__device__ auto cuda_odd_even_sort(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  cudaArray<T, vt> x, C comp, int flags = 0
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  cuda_iterate<vt>([&](auto I) {
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:        cuda_swap(x[i], x[i + 1]);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:__device__ auto cuda_odd_even_sort(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  cudaKVArray<K, V, vt> x, C comp, int flags = 0
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  cuda_iterate<vt>([&](auto I) {
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:        cuda_swap(x.keys[i], x.keys[i + 1]);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:        cuda_swap(x.vals[i], x.vals[i + 1]);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:__device__ inline int cuda_out_of_range_flags(int first, int vt, int count) {
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:__device__ inline auto cuda_compute_merge_sort_frame(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  return cudaMergeRange {
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:__device__ inline auto cuda_compute_merge_sort_range(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  auto frame = cuda_compute_merge_sort_frame(partition, coop, spacing);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  return cudaMergeRange {
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:__device__ inline auto cuda_compute_merge_sort_range(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  auto range = cuda_compute_merge_sort_range(count, partition, coop, spacing);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:struct cudaBlockSort {
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  static constexpr bool has_values = !std::is_same<V, cudaEmpty>::value;
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  static_assert(is_pow2(nt), "cudaBlockSort requires pow2 number of threads");
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:    cudaKVArray<K, V, vt> x,
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:    auto range = cuda_compute_merge_sort_range(count, tid, coop, vt);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:    cuda_reg_to_shared_thread<nt, vt>(x.keys, tid, storage.keys);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:    auto mp = cuda_merge_path<cudaMergeBoundType::LOWER>(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:    auto merge = cuda_serial_merge<cudaMergeBoundType::LOWER, vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:      cuda_reg_to_shared_thread<nt, vt>(x.vals, tid, storage.vals);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:      x.vals = cuda_shared_gather<nt, vt>(storage.vals, merge.indices);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  __device__ auto block_sort(cudaKVArray<K, V, vt> x,
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:      auto head_flags = cuda_out_of_range_flags(vt * tid, vt, count);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:      x = cuda_odd_even_sort(x, comp, head_flags);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:      x = cuda_odd_even_sort(x, comp);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:void cuda_merge_sort_partitions(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  cuda_kernel<<<B, nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:    auto range = cuda_get_tile(bid, nt * vt, num_partitions);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:    cuda_strided_iterate<nt, vt>([=](auto, auto j) {
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:      auto range = cuda_compute_merge_sort_range(count, index, coop, spacing);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:      buf[index] = cuda_merge_path<cudaMergeBoundType::LOWER>(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  const bool has_values = !std::is_same<V, cudaEmpty>::value;
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  unsigned R = cuda_find_log2(B, true);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  //cudaDeviceVector<K> keys_temp(R ? count : 0);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  //cudaDeviceVector<V> vals_temp((has_values && R) ? count : 0);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:    using sort_t = cudaBlockSort<E::nt, E::vt, K, V>;
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:    auto tile = cuda_get_tile(bid, E::nv, count);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:    cudaKVArray<K, V, E::vt> unsorted;
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:    unsorted.keys = cuda_mem_to_reg_thread<E::nt, E::vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:      unsorted.vals = cuda_mem_to_reg_thread<E::nt, E::vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:    cuda_reg_to_mem_thread<E::nt, E::vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:      cuda_reg_to_mem_thread<E::nt, E::vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  //cudaDeviceVector<unsigned> mem(num_partitions);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:    cuda_merge_sort_partitions(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:    cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=]__device__(auto tid, auto bid) {
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:      auto tile = cuda_get_tile(bid, E::nv, count);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:      auto range = cuda_compute_merge_sort_range(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:      auto merge = block_merge_from_mem<cudaMergeBoundType::LOWER, E::nt, E::vt>(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:      cuda_reg_to_mem_thread<E::nt>(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:        auto indices = cuda_reg_thread_to_strided<E::nt>(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:        cuda_transfer_two_streams_strided<E::nt>(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:@tparam V value type (default tf::cudaEmpty)
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:The function is used to allocate a buffer for calling tf::cuda_sort.
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:template <typename P, typename K, typename V = cudaEmpty>
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:unsigned cuda_sort_buffer_size(unsigned count) {
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  const bool has_values = !std::is_same<V, cudaEmpty>::value;
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  unsigned R = detail::cuda_find_log2(B, true);
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:void cuda_sort_by_key(
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:This method is equivalent to tf::cuda_sort_by_key without values.
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:void cuda_sort(P&& p, K_it k_first, K_it k_last, C comp, void* buf) {
tools/loaders/sharq/taskflow/cuda/algorithm/sort.hpp:  cuda_sort_by_key(p, k_first, k_last, (cudaEmpty*)nullptr, comp, buf);
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:#include "cuda_device.hpp"
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:@file cuda_memory.hpp
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:@brief CUDA memory utilities include file
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:inline size_t cuda_get_free_mem(int d) {
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  cudaScopedDevice ctx(d);
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    cudaMemGetInfo(&free, &total), "failed to get mem info on device ", d
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:inline size_t cuda_get_total_mem(int d) {
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  cudaScopedDevice ctx(d);
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    cudaMemGetInfo(&free, &total), "failed to get mem info on device ", d
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:The function calls @c cudaMalloc to allocate <tt>N*sizeof(T)</tt> bytes of memory
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:T* cuda_malloc_device(size_t N, int d) {
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  cudaScopedDevice ctx(d);
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    cudaMalloc(&ptr, N*sizeof(T)),
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:T* cuda_malloc_device(size_t N) {
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    cudaMalloc(&ptr, N*sizeof(T)), 
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:The function calls @c cudaMallocManaged to allocate <tt>N*sizeof(T)</tt> bytes
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:T* cuda_malloc_shared(size_t N) {
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    cudaMallocManaged(&ptr, N*sizeof(T)),
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:@brief frees memory on the GPU device
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:This methods call @c cudaFree to free the memory space pointed to by @c ptr
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:void cuda_free(T* ptr, int d) {
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  cudaScopedDevice ctx(d);
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(cudaFree(ptr), "failed to free memory ", ptr, " on GPU ", d);
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:@brief frees memory on the GPU device
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:This methods call @c cudaFree to free the memory space pointed to by @c ptr
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:void cuda_free(T* ptr) {
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(cudaFree(ptr), "failed to free memory ", ptr);
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:The method calls @c cudaMemcpyAsync with the given @c stream
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:using @c cudaMemcpyDefault to infer the memory space of the source and
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:inline void cuda_memcpy_async(
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  cudaStream_t stream, void* dst, const void* src, size_t count
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream),
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    "failed to perform cudaMemcpyAsync"
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:@brief initializes or sets GPU memory to the given value byte by byte
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:@param devPtr pointer to GPU mempry
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:The method calls @c cudaMemsetAsync with the given @c stream
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:inline void cuda_memset_async(
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  cudaStream_t stream, void* devPtr, int value, size_t count
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    cudaMemsetAsync(devPtr, value, count, stream),
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    "failed to perform cudaMemsetAsync"
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp://      cudaSharedMemory<T> smem;
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <int>
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <unsigned int>
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <char>
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <unsigned char>
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <short>
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <unsigned short>
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <long>
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <unsigned long>
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp://struct cudaSharedMemory <size_t>
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <bool>
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <float>
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <double>
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:// cudaDeviceAllocator
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:@class cudaDeviceAllocator
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:@brief class to create a CUDA device allocator 
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:A %cudaDeviceAllocator enables device-specific allocation for 
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:class cudaDeviceAllocator {
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    using other = cudaDeviceAllocator<U>; 
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  cudaDeviceAllocator() noexcept {}
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  cudaDeviceAllocator( const cudaDeviceAllocator& ) noexcept {}
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  cudaDeviceAllocator( const cudaDeviceAllocator<U>& ) noexcept {}
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  ~cudaDeviceAllocator() noexcept {}
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  The block of storage is allocated using cudaMalloc and throws std::bad_alloc 
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:      cudaMalloc( &ptr, n*sizeof(T) ),
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:      cudaFree(ptr);
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  bool operator == (const cudaDeviceAllocator<U>&) const noexcept {
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  bool operator != (const cudaDeviceAllocator<U>&) const noexcept {
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:// cudaUSMAllocator
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:@class cudaUSMAllocator
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:A %cudaUSMAllocator enables using unified shared memory (USM) allocation for 
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:class cudaUSMAllocator {
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    using other = cudaUSMAllocator<U>; 
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  cudaUSMAllocator() noexcept {}
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  cudaUSMAllocator( const cudaUSMAllocator& ) noexcept {}
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  cudaUSMAllocator( const cudaUSMAllocator<U>& ) noexcept {}
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  ~cudaUSMAllocator() noexcept {}
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  The block of storage is allocated using cudaMalloc and throws std::bad_alloc 
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:      cudaMallocManaged( &ptr, n*sizeof(T) ),
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:      cudaFree(ptr);
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  bool operator == (const cudaUSMAllocator<U>&) const noexcept {
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:  bool operator != (const cudaUSMAllocator<U>&) const noexcept {
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:// GPU vector object
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp://using cudaDeviceVector = std::vector<NoInit<T>, cudaDeviceAllocator<NoInit<T>>>;
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp://using cudaUSMVector = std::vector<T, cudaUSMAllocator<T>>;
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:class cudaDeviceVector {
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    cudaDeviceVector() = default;
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    cudaDeviceVector(size_t N) : _N {N} {
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:        TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:          cudaMalloc(&_data, N*sizeof(T)),
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    cudaDeviceVector(cudaDeviceVector&& rhs) : 
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    ~cudaDeviceVector() {
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:        cudaFree(_data);
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    cudaDeviceVector& operator = (cudaDeviceVector&& rhs) {
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:        cudaFree(_data);
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    cudaDeviceVector(const cudaDeviceVector&) = delete;
tools/loaders/sharq/taskflow/cuda/cuda_memory.hpp:    cudaDeviceVector& operator = (const cudaDeviceVector&) = delete;
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:#include "cuda_graph.hpp"
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:@file cuda_optimizer.hpp
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:@brief %cudaFlow capturing algorithms include file
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:// cudaFlowOptimizerBase
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:class cudaFlowOptimizerBase {
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:    std::vector<cudaFlowNode*> _toposort(cudaFlowGraph&);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:    std::vector<std::vector<cudaFlowNode*>> _levelize(cudaFlowGraph&);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:inline std::vector<cudaFlowNode*> cudaFlowOptimizerBase::_toposort(cudaFlowGraph& graph) {
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  std::vector<cudaFlowNode*> res;
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  std::queue<cudaFlowNode*> bfs;
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:    auto hu = std::get_if<cudaFlowNode::Capture>(&u->_handle);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:      auto hv = std::get_if<cudaFlowNode::Capture>(&v->_handle);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:inline std::vector<std::vector<cudaFlowNode*>>
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:cudaFlowOptimizerBase::_levelize(cudaFlowGraph& graph) {
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  std::queue<cudaFlowNode*> bfs;
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:    auto hu = std::get_if<cudaFlowNode::Capture>(&u->_handle);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:    auto hu = std::get_if<cudaFlowNode::Capture>(&u->_handle);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:      auto hv = std::get_if<cudaFlowNode::Capture>(&v->_handle);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  std::vector<std::vector<cudaFlowNode*>> level_graph(max_level+1);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:    auto hu = std::get_if<cudaFlowNode::Capture>(&u->_handle);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:    //  assert(hu.level < std::get_if<cudaFlowNode::Capture>(&s->_handle)->level);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:// class definition: cudaFlowSequentialOptimizer
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:@class cudaFlowSequentialOptimizer
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:@brief class to capture a CUDA graph using a sequential stream
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:the described graph and captures dependent GPU tasks using a single stream.
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:All GPU tasks run sequentially without breaking inter dependencies.
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:class cudaFlowSequentialOptimizer : public cudaFlowOptimizerBase {
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  friend class cudaFlowCapturer;
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:    cudaFlowSequentialOptimizer() = default;
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:    cudaGraph_t _optimize(cudaFlowGraph& graph);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:inline cudaGraph_t cudaFlowSequentialOptimizer::_optimize(cudaFlowGraph& graph) {
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  // we must use ThreadLocal mode to avoid clashing with CUDA global states
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  cudaStream stream;
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  stream.begin_capture(cudaStreamCaptureModeThreadLocal);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:    std::get_if<cudaFlowNode::Capture>(&node->_handle)->work(stream);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:// class definition: cudaFlowLinearOptimizer
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:@class cudaFlowLinearOptimizer
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:@brief class to capture a linear CUDA graph using a sequential stream
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:A linear capturing algorithm is a special case of tf::cudaFlowSequentialOptimizer
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:class cudaFlowLinearOptimizer : public cudaFlowOptimizerBase {
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  friend class cudaFlowCapturer;
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:    cudaFlowLinearOptimizer() = default;
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:    cudaGraph_t _optimize(cudaFlowGraph& graph);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:inline cudaGraph_t cudaFlowLinearOptimizer::_optimize(cudaFlowGraph& graph) {
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  // we must use ThreadLocal mode to avoid clashing with CUDA global states
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  cudaStream stream;
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  stream.begin_capture(cudaStreamCaptureModeThreadLocal);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  cudaFlowNode* src {nullptr};
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:        std::get_if<cudaFlowNode::Capture>(&src->_handle)->work(stream);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:// class definition: cudaFlowRoundRobinOptimizer
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:@class cudaFlowRoundRobinOptimizer
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:@brief class to capture a CUDA graph using a round-robin algorithm
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  + Dian-Lun Lin and Tsung-Wei Huang, &quot;Efficient GPU Computation using %Task Graph Parallelism,&quot; <i>European Conference on Parallel and Distributed Computing (Euro-Par)</i>, 2021
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:The round-robin optimization algorithm is best suited for large %cudaFlow graphs
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:that compose hundreds of or thousands of GPU operations
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:maximum kernel currency in the captured CUDA graph.
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:class cudaFlowRoundRobinOptimizer : public cudaFlowOptimizerBase {
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  friend class cudaFlowCapturer;
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:    cudaFlowRoundRobinOptimizer() = default;
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:    explicit cudaFlowRoundRobinOptimizer(size_t num_streams);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:    cudaGraph_t _optimize(cudaFlowGraph& graph);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:    void _reset(std::vector<std::vector<cudaFlowNode*>>& graph);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:inline cudaFlowRoundRobinOptimizer::cudaFlowRoundRobinOptimizer(size_t num_streams) :
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:inline size_t cudaFlowRoundRobinOptimizer::num_streams() const {
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:inline void cudaFlowRoundRobinOptimizer::num_streams(size_t n) {
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:inline void cudaFlowRoundRobinOptimizer::_reset(
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  std::vector<std::vector<cudaFlowNode*>>& graph
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:      auto hn = std::get_if<cudaFlowNode::Capture>(&node->_handle);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:inline cudaGraph_t cudaFlowRoundRobinOptimizer::_optimize(cudaFlowGraph& graph) {
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  std::vector<cudaStream> streams(_num_streams);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  streams[0].begin_capture(cudaStreamCaptureModeThreadLocal);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  std::vector<cudaEvent> events;
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:  cudaEvent_t fork_event = events.emplace_back();
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:      auto hn = std::get_if<cudaFlowNode::Capture>(&node->_handle);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:      cudaFlowNode* wait_node{nullptr};
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:        auto phn = std::get_if<cudaFlowNode::Capture>(&pn->_handle);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:             std::get_if<cudaFlowNode::Capture>(&wait_node->_handle)->level < phn->level) {
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:        assert(std::get_if<cudaFlowNode::Capture>(&wait_node->_handle)->event); 
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:        streams[sid].wait(std::get_if<cudaFlowNode::Capture>(&wait_node->_handle)->event);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:        auto shn = std::get_if<cudaFlowNode::Capture>(&sn->_handle);
tools/loaders/sharq/taskflow/cuda/cuda_optimizer.hpp:    cudaEvent_t join_event = events.emplace_back();
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:#include "cuda_error.hpp"
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:@file cuda_device.hpp
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:@brief CUDA device utilities include file
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_num_devices() {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(cudaGetDeviceCount(&N), "failed to get device count");
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline int cuda_get_device() {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(cudaGetDevice(&id), "failed to get current device id");
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline void cuda_set_device(int id) {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(cudaSetDevice(id), "failed to switch to device ", id);
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline void cuda_get_device_property(int i, cudaDeviceProp& p) {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaGetDeviceProperties(&p, i), "failed to get property of device ", i
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline cudaDeviceProp cuda_get_device_property(int i) {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  cudaDeviceProp p;
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaGetDeviceProperties(&p, i), "failed to get property of device ", i
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline void cuda_dump_device_property(std::ostream& os, const cudaDeviceProp& p) {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:     << "GPU sharing Host Memory:       " << p.integrated << '\n'
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_threads_per_block(int d) {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerBlock, d),
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_x_dim_per_block(int d) {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxBlockDimX, d),
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_y_dim_per_block(int d) {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxBlockDimY, d),
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_z_dim_per_block(int d) {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxBlockDimZ, d),
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_x_dim_per_grid(int d) {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxGridDimX, d),
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_y_dim_per_grid(int d) {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxGridDimY, d),
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_z_dim_per_grid(int d) {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxGridDimZ, d),
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_shm_per_block(int d) {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&num, cudaDevAttrMaxSharedMemoryPerBlock, d),
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_warp_size(int d) {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&num, cudaDevAttrWarpSize, d),
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline int cuda_get_device_compute_capability_major(int d) {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&num, cudaDevAttrComputeCapabilityMajor, d),
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline int cuda_get_device_compute_capability_minor(int d) {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&num, cudaDevAttrComputeCapabilityMinor, d),
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline bool cuda_get_device_unified_addressing(int d) {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&num, cudaDevAttrUnifiedAddressing, d),
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:// CUDA Version
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:@brief queries the latest CUDA version (1000 * major + 10 * minor) supported by the driver
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline int cuda_get_driver_version() {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaDriverGetVersion(&num),
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    "failed to query the latest cuda version supported by the driver"
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:@brief queries the CUDA Runtime version (1000 * major + 10 * minor)
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline int cuda_get_runtime_version() {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaRuntimeGetVersion(&num), "failed to query cuda runtime version"
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:// cudaScopedDevice
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:/** @class cudaScopedDevice
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  tf::cudaScopedDevice device(1);  // switch to the device context 1
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  cudaStream_t stream;
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  cudaStreamCreate(&stream);
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:%cudaScopedDevice is neither movable nor copyable.
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:class cudaScopedDevice {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    explicit cudaScopedDevice(int device);
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    ~cudaScopedDevice();
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaScopedDevice() = delete;
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaScopedDevice(const cudaScopedDevice&) = delete;
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaScopedDevice(cudaScopedDevice&&) = delete;
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline cudaScopedDevice::cudaScopedDevice(int dev) {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(cudaGetDevice(&_p), "failed to get current device scope");
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    TF_CHECK_CUDA(cudaSetDevice(dev), "failed to scope on device ", dev);
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:inline cudaScopedDevice::~cudaScopedDevice() {
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    cudaSetDevice(_p);
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:    //TF_CHECK_CUDA(cudaSetDevice(_p), "failed to scope back to device ", _p);
tools/loaders/sharq/taskflow/cuda/cuda_device.hpp:}  // end of namespace cuda ---------------------------------------------------
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:#include "cuda_error.hpp"
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:@file cuda_execution_policy.hpp
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:@brief CUDA execution policy include file
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:@class cudaExecutionPolicy
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:@brief class to define execution policy for CUDA standard algorithms
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:Execution policy configures the kernel execution parameters in CUDA algorithms.
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:Details can be referred to @ref CUDASTDExecutionPolicy.
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:class cudaExecutionPolicy {
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:  cudaExecutionPolicy() = default;
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:  explicit cudaExecutionPolicy(cudaStream_t s) : _stream{s} {}
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:  cudaStream_t stream() noexcept { return _stream; };
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:  void stream(cudaStream_t stream) noexcept { _stream = stream; }
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:  The function is used to allocate a buffer for calling tf::cuda_reduce,
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:  tf::cuda_uninitialized_reduce, tf::cuda_transform_reduce, and
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:  tf::cuda_uninitialized_transform_reduce.
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:  @brief queries the buffer size in bytes needed to call tf::cuda_min_element
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:  tf::cuda_min_element.
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:  @brief queries the buffer size in bytes needed to call tf::cuda_max_element
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:  tf::cuda_max_element.
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:  tf::cuda_inclusive_scan, tf::cuda_exclusive_scan,
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:  tf::cuda_transform_inclusive_scan, and tf::cuda_transform_exclusive_scan.
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:  @brief queries the buffer size in bytes needed for CUDA merge algorithms
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:  tf::cuda_merge and tf::cuda_merge_by_key.
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:  cudaStream_t _stream {0};
tools/loaders/sharq/taskflow/cuda/cuda_execution_policy.hpp:using cudaDefaultExecutionPolicy = cudaExecutionPolicy<512, 7>;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:#include "cuda_memory.hpp"
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:#include "cuda_stream.hpp"
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:#include "cuda_meta.hpp"
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:// cudaGraph_t routines
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:cudaMemcpy3DParms cuda_get_copy_parms(T* tgt, const T* src, size_t num) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  cudaMemcpy3DParms p;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  p.srcPos = ::make_cudaPos(0, 0, 0);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  p.srcPtr = ::make_cudaPitchedPtr(const_cast<T*>(src), num*sizeof(U), num, 1);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  p.dstPos = ::make_cudaPos(0, 0, 0);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  p.dstPtr = ::make_cudaPitchedPtr(tgt, num*sizeof(U), num, 1);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  p.extent = ::make_cudaExtent(num*sizeof(U), 1, 1);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  p.kind = cudaMemcpyDefault;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:inline cudaMemcpy3DParms cuda_get_memcpy_parms(
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  // Parameters in cudaPitchedPtr
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  cudaMemcpy3DParms p;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  p.srcPos = ::make_cudaPos(0, 0, 0);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  p.srcPtr = ::make_cudaPitchedPtr(const_cast<void*>(src), bytes, bytes, 1);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  p.dstPos = ::make_cudaPos(0, 0, 0);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  p.dstPtr = ::make_cudaPitchedPtr(tgt, bytes, bytes, 1);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  p.extent = ::make_cudaExtent(bytes, 1, 1);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  p.kind = cudaMemcpyDefault;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:inline cudaMemsetParams cuda_get_memset_parms(void* dst, int ch, size_t count) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  cudaMemsetParams p;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:cudaMemsetParams cuda_get_fill_parms(T* dst, T value, size_t count) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  cudaMemsetParams p;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:cudaMemsetParams cuda_get_zero_parms(T* dst, size_t count) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  cudaMemsetParams p;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:@brief queries the number of root nodes in a native CUDA graph
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:inline size_t cuda_graph_get_num_root_nodes(cudaGraph_t graph) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetRootNodes(graph, nullptr, &num_nodes),
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:@brief queries the number of nodes in a native CUDA graph
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:inline size_t cuda_graph_get_num_nodes(cudaGraph_t graph) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetNodes(graph, nullptr, &num_nodes),
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:@brief queries the number of edges in a native CUDA graph
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:inline size_t cuda_graph_get_num_edges(cudaGraph_t graph) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetEdges(graph, nullptr, nullptr, &num_edges),
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:@brief acquires the nodes in a native CUDA graph
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:inline std::vector<cudaGraphNode_t> cuda_graph_get_nodes(cudaGraph_t graph) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  size_t num_nodes = cuda_graph_get_num_nodes(graph);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  std::vector<cudaGraphNode_t> nodes(num_nodes);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetNodes(graph, nodes.data(), &num_nodes),
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:@brief acquires the root nodes in a native CUDA graph
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:inline std::vector<cudaGraphNode_t> cuda_graph_get_root_nodes(cudaGraph_t graph) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  size_t num_nodes = cuda_graph_get_num_root_nodes(graph);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  std::vector<cudaGraphNode_t> nodes(num_nodes);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetRootNodes(graph, nodes.data(), &num_nodes),
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:@brief acquires the edges in a native CUDA graph
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:inline std::vector<std::pair<cudaGraphNode_t, cudaGraphNode_t>>
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:cuda_graph_get_edges(cudaGraph_t graph) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  size_t num_edges = cuda_graph_get_num_edges(graph);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  std::vector<cudaGraphNode_t> froms(num_edges), tos(num_edges);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetEdges(graph, froms.data(), tos.data(), &num_edges),
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  std::vector<std::pair<cudaGraphNode_t, cudaGraphNode_t>> edges(num_edges);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:@brief queries the type of a native CUDA graph node
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeKernel      = 0x00
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeMemcpy      = 0x01
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeMemset      = 0x02
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeHost        = 0x03
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeGraph       = 0x04
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeEmpty       = 0x05
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeWaitEvent   = 0x06
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeEventRecord = 0x07
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:inline cudaGraphNodeType cuda_get_graph_node_type(cudaGraphNode_t node) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  cudaGraphNodeType type;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaGraphNodeGetType(node, &type), "failed to get native graph node type"
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:@brief convert the type of a native CUDA graph node to a readable string
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:inline const char* cuda_graph_node_type_to_string(cudaGraphNodeType type) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeKernel      : return "kernel";
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeMemcpy      : return "memcpy";
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeMemset      : return "memset";
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeHost        : return "host";
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeGraph       : return "graph";
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeEmpty       : return "empty";
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeWaitEvent   : return "event_wait";
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeEventRecord : return "event_record";
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:@brief dumps a native CUDA graph and all associated child graphs to a DOT format
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:@param graph native CUDA graph
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:void cuda_dump_graph(T& os, cudaGraph_t graph) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  os << "digraph cudaGraph {\n";
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  std::stack<std::tuple<cudaGraph_t, cudaGraphNode_t, int>> stack;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:       << "label=\"cudaGraph-L" << l << "\";\n"
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    auto nodes = cuda_graph_get_nodes(graph);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    auto edges = cuda_graph_get_edges(graph);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:      auto type = cuda_get_graph_node_type(node);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:      if(type == cudaGraphNodeTypeGraph) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:        cudaGraph_t graph;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:        TF_CHECK_CUDA(cudaGraphChildGraphNodeGetGraph(node, &graph), "");
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:           << "label=\"cudaGraph-L" << l+1
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:           << cuda_graph_node_type_to_string(type)
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:      std::unordered_set<cudaGraphNode_t> successors;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:// cudaGraph
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:struct cudaGraphCreator {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  cudaGraph_t operator () () const { 
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaGraph_t g;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    TF_CHECK_CUDA(cudaGraphCreate(&g, 0), "failed to create a CUDA native graph");
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:struct cudaGraphDeleter {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  void operator () (cudaGraph_t g) const {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:      cudaGraphDestroy(g);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:@class cudaGraph
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:@brief class to create an RAII-styled wrapper over a CUDA executable graph
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:A cudaGraph object is an RAII-styled wrapper over 
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:a native CUDA graph (@c cudaGraph_t).
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:A cudaGraph object is move-only.
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:class cudaGraph :
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  public cudaObject<cudaGraph_t, cudaGraphCreator, cudaGraphDeleter> {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  @brief constructs an RAII-styled object from the given CUDA exec
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  Constructs a cudaGraph object from the given CUDA graph @c native.
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  explicit cudaGraph(cudaGraph_t native) : cudaObject(native) { }
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  @brief constructs a cudaGraph object with a new CUDA graph
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  cudaGraph() = default;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:// cudaGraphExec
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:struct cudaGraphExecCreator {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  cudaGraphExec_t operator () () const { return nullptr; }
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:struct cudaGraphExecDeleter {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  void operator () (cudaGraphExec_t executable) const {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:      cudaGraphExecDestroy(executable);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:@class cudaGraphExec
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:@brief class to create an RAII-styled wrapper over a CUDA executable graph
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:A cudaGraphExec object is an RAII-styled wrapper over 
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:a native CUDA executable graph (@c cudaGraphExec_t).
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:A cudaGraphExec object is move-only.
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:class cudaGraphExec : 
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  public cudaObject<cudaGraphExec_t, cudaGraphExecCreator, cudaGraphExecDeleter> {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  @brief constructs an RAII-styled object from the given CUDA exec
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  Constructs a cudaGraphExec object which owns @c exec.
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  explicit cudaGraphExec(cudaGraphExec_t exec) : cudaObject(exec) { }
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  cudaGraphExec() = default;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  @brief instantiates the exexutable from the given CUDA graph
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  void instantiate(cudaGraph_t graph) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaGraphExecDeleter {} (object);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:      cudaGraphInstantiate(&object, graph, nullptr, nullptr, 0),
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  @brief updates the exexutable from the given CUDA graph
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  cudaGraphExecUpdateResult update(cudaGraph_t graph) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaGraphNode_t error_node;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaGraphExecUpdateResult error_result;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaGraphExecUpdate(object, graph, &error_node, &error_result);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  void launch(cudaStream_t stream) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:      cudaGraphLaunch(object, stream), "failed to launch a CUDA executable graph"
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:// cudaFlowGraph class
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:// class: cudaFlowGraph
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:class cudaFlowGraph {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowNode;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  friend class cudaTask;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowCapturer;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlow;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowOptimizerBase;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowSequentialOptimizer;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowLinearOptimizer;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowRoundRobinOptimizer;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaFlowGraph() = default;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    ~cudaFlowGraph() = default;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaFlowGraph(const cudaFlowGraph&) = delete;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaFlowGraph(cudaFlowGraph&&) = default;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaFlowGraph& operator = (const cudaFlowGraph&) = delete;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaFlowGraph& operator = (cudaFlowGraph&&) = default;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaFlowNode* emplace_back(ArgsT&&...);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaGraph _native_handle {nullptr};
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    std::vector<std::unique_ptr<cudaFlowNode>> _nodes;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:// cudaFlowNode class
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:@class: cudaFlowNode
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:class cudaFlowNode {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowGraph;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  friend class cudaTask;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlow;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowCapturer;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowOptimizerBase;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowSequentialOptimizer;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowLinearOptimizer;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowRoundRobinOptimizer;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaFlowGraph cfg;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    std::function<void(cudaStream_t)> work;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaEvent_t event;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaFlowNode() = delete;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaFlowNode(cudaFlowGraph&, ArgsT&&...);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaFlowGraph& _cfg;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    cudaGraphNode_t _native_handle {nullptr};
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    SmallVector<cudaFlowNode*> _successors;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    SmallVector<cudaFlowNode*> _dependents;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    void _precede(cudaFlowNode*);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:// cudaFlowNode definitions
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:cudaFlowNode::Host::Host(C&& c) : func {std::forward<C>(c)} {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:inline void cudaFlowNode::Host::callback(void* data) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:cudaFlowNode::Kernel::Kernel(F&& f) :
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:cudaFlowNode::Capture::Capture(C&& work) :
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:cudaFlowNode::cudaFlowNode(cudaFlowGraph& graph, ArgsT&&... args) :
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:inline void cudaFlowNode::_precede(cudaFlowNode* v) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  _cfg._state |= cudaFlowGraph::CHANGED;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  if(_handle.index() != cudaFlowNode::CAPTURE) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:    TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:      cudaGraphAddDependencies(
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:// cudaGraph definitions
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:inline bool cudaFlowGraph::empty() const {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:inline void cudaFlowGraph::clear() {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  _state |= cudaFlowGraph::CHANGED;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:cudaFlowNode* cudaFlowGraph::emplace_back(ArgsT&&... args) {
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  _state |= cudaFlowGraph::CHANGED;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  auto node = std::make_unique<cudaFlowNode>(std::forward<ArgsT>(args)...);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  //auto node = new cudaFlowNode(std::forward<ArgsT>(args)...);
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:inline void cudaFlowGraph::dump(
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:  std::stack<std::tuple<const cudaFlowGraph*, const cudaFlowNode*, int>> stack;
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:        os << "subgraph cluster_p" << root << " {\nlabel=\"cudaFlow: ";
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:        os << "digraph cudaFlow {\n";
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:      os << "subgraph cluster_p" << parent << " {\nlabel=\"cudaSubflow: ";
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:        case cudaFlowNode::KERNEL:
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:        case cudaFlowNode::SUBFLOW:
tools/loaders/sharq/taskflow/cuda/cuda_graph.hpp:            &(std::get_if<cudaFlowNode::Subflow>(&v->_handle)->cfg), v, l+1)
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:#include "cuda_error.hpp"
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:@brief per-thread object pool to manage CUDA device object
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:A CUDA device object has a lifetime associated with a device,
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:for example, @c cudaStream_t, @c cublasHandle_t, etc.
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:There exists an one-to-one relationship between CUDA devices in CUDA Runtime API
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:and CUcontexts in the CUDA Driver API within a process.
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:The specific context which the CUDA Runtime API uses for a device
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:From the perspective of the CUDA Runtime API,
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:class cudaPerThreadDeviceObjectPool {
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  // Due to some ordering, cuda context may be destroyed when the master
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  // program thread destroys the cuda object.
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  // destroy cuda objects while the master thread only keeps a weak reference
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  struct cudaGlobalDeviceObjectPool {
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:    cudaPerThreadDeviceObjectPool() = default;
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:    inline static cudaGlobalDeviceObjectPool _shared_pool;
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:// cudaPerThreadDeviceObject::cudaHanale definition
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:cudaPerThreadDeviceObjectPool<H, C, D>::Object::Object(int d) :
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  cudaScopedDevice ctx(device);
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:cudaPerThreadDeviceObjectPool<H, C, D>::Object::~Object() {
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  cudaScopedDevice ctx(device);
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:// cudaPerThreadDeviceObject::cudaHanaldePool definition
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:std::shared_ptr<typename cudaPerThreadDeviceObjectPool<H, C, D>::Object>
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:cudaPerThreadDeviceObjectPool<H, C, D>::cudaGlobalDeviceObjectPool::acquire(int d) {
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:void cudaPerThreadDeviceObjectPool<H, C, D>::cudaGlobalDeviceObjectPool::release(
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:// cudaPerThreadDeviceObject definition
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:std::shared_ptr<typename cudaPerThreadDeviceObjectPool<H, C, D>::Object>
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:cudaPerThreadDeviceObjectPool<H, C, D>::acquire(int d) {
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:void cudaPerThreadDeviceObjectPool<H, C, D>::release(
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:size_t cudaPerThreadDeviceObjectPool<H, C, D>::footprint_size() const {
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:// cudaObject
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:@class cudaObject
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:@brief class to create an RAII-styled and move-only wrapper for CUDA objects
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:class cudaObject {
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  @brief constructs a CUDA object from the given one
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  explicit cudaObject(T obj) : object(obj) {}
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  @brief constructs a new CUDA object
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  cudaObject() : object{ C{}() } {}
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  cudaObject(const cudaObject&) = delete;
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  cudaObject(cudaObject&& rhs) : object{rhs.object} {
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  @brief destructs the CUDA object
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  ~cudaObject() { D{}(object); }
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  cudaObject& operator = (const cudaObject&) = delete;
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  cudaObject& operator = (cudaObject&& rhs) {
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  @brief implicit conversion to the native CUDA stream (cudaObject_t)
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  Returns the underlying stream of type @c cudaObject_t.
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  @brief deletes the current CUDA object (if any) and creates a new one
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  @brief resets this CUDA object to the given one
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  @brief deletes the current CUDA object
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  @brief releases the ownership of the CUDA object
tools/loaders/sharq/taskflow/cuda/cuda_object.hpp:  @brief the CUDA object
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:#include "cuda_object.hpp"
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:@file cuda_stream.hpp
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:@brief CUDA stream utilities include file
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:// cudaStream
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:struct cudaStreamCreator {
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:  cudaStream_t operator () () const {
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    cudaStream_t stream;
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    TF_CHECK_CUDA(cudaStreamCreate(&stream), "failed to create a CUDA stream");
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:struct cudaStreamDeleter {
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:  void operator () (cudaStream_t stream) const {
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:      cudaStreamDestroy(stream);
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:@class cudaStream
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:@brief class to create an RAII-styled wrapper over a native CUDA stream
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:A cudaStream object is an RAII-styled wrapper over a native CUDA stream
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:(@c cudaStream_t).
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:A cudaStream object is move-only.
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:class cudaStream : 
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:  public cudaObject <cudaStream_t, cudaStreamCreator, cudaStreamDeleter> {
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    @brief constructs an RAII-styled object from the given CUDA stream
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    Constructs a cudaStream object which owns @c stream.
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    explicit cudaStream(cudaStream_t stream) : cudaObject(stream) {
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    cudaStream() = default;
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    Equivalently calling @c cudaStreamSynchronize to block 
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:      TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:        cudaStreamSynchronize(object), "failed to synchronize a CUDA stream"
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    which will be returned via cudaStream::end_capture. 
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    + @c cudaStreamCaptureModeGlobal: This is the default mode. 
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:      with @c cudaStreamCaptureModeRelaxed at @c cuStreamBeginCapture, 
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:      @c cudaStreamCaptureModeGlobal, this thread is prohibited from potentially 
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    + @c cudaStreamCaptureModeThreadLocal: If the local thread has an ongoing capture 
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:      sequence not initiated with @c cudaStreamCaptureModeRelaxed, 
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    + @c cudaStreamCaptureModeRelaxed: The local thread is not prohibited 
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:      attempting @c cudaEventQuery on an event that was last recorded 
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    void begin_capture(cudaStreamCaptureMode m = cudaStreamCaptureModeGlobal) const {
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:      TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:        cudaStreamBeginCapture(object, m), 
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    Equivalently calling @c cudaStreamEndCapture to
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    Capture must have been initiated on stream via a call to cudaStream::begin_capture. 
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    cudaGraph_t end_capture() const {
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:      cudaGraph_t native_g;
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:      TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:        cudaStreamEndCapture(object, &native_g), 
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    Equivalently calling @c cudaEventRecord to record an event on this stream,
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    both of which must be on the same CUDA context.
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    void record(cudaEvent_t event) const {
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:      TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:        cudaEventRecord(event, object), 
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    Equivalently calling @c cudaStreamWaitEvent to make all future work 
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    void wait(cudaEvent_t event) const {
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:      TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:        cudaStreamWaitEvent(object, event, 0), 
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:// cudaEvent
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:struct cudaEventCreator {
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:  cudaEvent_t operator () () const {
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    cudaEvent_t event;
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    TF_CHECK_CUDA(cudaEventCreate(&event), "failed to create a CUDA event");
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:  cudaEvent_t operator () (unsigned int flag) const {
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    cudaEvent_t event;
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:      cudaEventCreateWithFlags(&event, flag),
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:      "failed to create a CUDA event with flag=", flag
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:struct cudaEventDeleter {
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:  void operator () (cudaEvent_t event) const {
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    cudaEventDestroy(event);
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:@class cudaEvent
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:@brief class to create an RAII-styled wrapper over a native CUDA event
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:A cudaEvent object is an RAII-styled wrapper over a native CUDA event 
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:(@c cudaEvent_t).
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:A cudaEvent object is move-only.
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:class cudaEvent :
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:  public cudaObject<cudaEvent_t, cudaEventCreator, cudaEventDeleter> {
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    @brief constructs an RAII-styled CUDA event object from the given CUDA event
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    explicit cudaEvent(cudaEvent_t event) : cudaObject(event) { }   
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    @brief constructs an RAII-styled CUDA event object
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    cudaEvent() = default;
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    @brief constructs an RAII-styled CUDA event object with the given flag
tools/loaders/sharq/taskflow/cuda/cuda_stream.hpp:    explicit cudaEvent(unsigned int flag) : cudaObject(cudaEventCreator{}(flag)) { }
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:#include "cuda_task.hpp"
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:#include "cuda_capturer.hpp"
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:@file taskflow/cuda/cudaflow.hpp
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:@brief cudaFlow include file
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:// class definition: cudaFlow
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:@class cudaFlow
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:@brief class to create a %cudaFlow task dependency graph
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:A %cudaFlow is a high-level interface over CUDA Graph to perform GPU operations
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:on one or multiple CUDA devices,
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:The following example creates a %cudaFlow of two kernel tasks, @c task1 and
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:taskflow.emplace([&](tf::cudaFlow& cf){
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  tf::cudaTask task1 = cf.kernel(grid1, block1, shm_size1, kernel1, args1);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  tf::cudaTask task2 = cf.kernel(grid2, block2, shm_size2, kernel2, args2);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:A %cudaFlow is a task (tf::Task) created from tf::Taskflow
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:That is, the callable that describes a %cudaFlow
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:Inside a %cudaFlow task, different GPU tasks (tf::cudaTask) may run
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:in parallel scheduled by the CUDA runtime.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:Please refer to @ref GPUTaskingcudaFlow for details.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:class cudaFlow {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @brief constructs a %cudaFlow
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaFlow();
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @brief destroys the %cudaFlow and its associated native CUDA graph
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    ~cudaFlow() = default;
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaFlow(cudaFlow&&) = default;
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaFlow& operator = (cudaFlow&&) = default;
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @brief clears the %cudaFlow object
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @brief dumps the %cudaFlow graph into a DOT format through an
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @brief dumps the native CUDA graph into a DOT format through an
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    The native CUDA graph may be different from the upper-level %cudaFlow
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaTask noop();
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    A host task can only execute CPU-specific functions and cannot do any CUDA calls
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    (e.g., @c cudaMalloc).
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaTask host(C&& callable);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    The method is similar to tf::cudaFlow::host but operates on a task
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    of type tf::cudaTaskType::HOST.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    void host(cudaTask task, C&& callable);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaTask kernel(dim3 g, dim3 b, size_t s, F f, ArgsT... args);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    The method is similar to tf::cudaFlow::kernel but operates on a task
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    of type tf::cudaTaskType::KERNEL.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:      cudaTask task, dim3 g, dim3 b, size_t shm, F f, ArgsT... args
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaTask memset(void* dst, int v, size_t count);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    The method is similar to tf::cudaFlow::memset but operates on a task
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    of type tf::cudaTaskType::MEMSET.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    void memset(cudaTask task, void* dst, int ch, size_t count);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    to a target location. Direction can be arbitrary among CPUs and GPUs.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaTask memcpy(void* tgt, const void* src, size_t bytes);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    The method is similar to tf::cudaFlow::memcpy but operates on a task
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    of type tf::cudaTaskType::MEMCPY.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    void memcpy(cudaTask task, void* tgt, const void* src, size_t bytes);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaTask zero(T* dst, size_t count);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    The method is similar to tf::cudaFlow::zero but operates on
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    a task of type tf::cudaTaskType::MEMSET.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    void zero(cudaTask task, T* dst, size_t count);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaTask fill(T* dst, T value, size_t count);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    The method is similar to tf::cudaFlow::fill but operates on a task
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    of type tf::cudaTaskType::MEMSET.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    void fill(cudaTask task, T* dst, T value, size_t count);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    to a target location. Direction can be arbitrary among CPUs and GPUs.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaTask copy(T* tgt, const T* src, size_t num);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    The method is similar to tf::cudaFlow::copy but operates on a task
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    of type tf::cudaTaskType::MEMCPY.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    void copy(cudaTask task, T* tgt, const T* src, size_t num);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @brief offloads the %cudaFlow onto a GPU asynchronously via a stream
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    Offloads the present %cudaFlow onto a GPU asynchronously via
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    An offloaded %cudaFlow forces the underlying graph to be instantiated.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    void run(cudaStream_t stream);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @brief acquires a reference to the underlying CUDA graph
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraph_t native_graph();
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @brief acquires a reference to the underlying CUDA graph executable
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphExec_t native_executable();
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaTask single_task(C c);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    This method is similar to cudaFlow::single_task but operates
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    void single_task(cudaTask task, C c);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaTask for_each(I first, I last, C callable);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::for_each
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    the task created from tf::cudaFlow::for_each.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    void for_each(cudaTask task, I first, I last, C callable);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaTask for_each_index(I first, I last, I step, C callable);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::for_each_index
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    the task created from tf::cudaFlow::for_each_index.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:      cudaTask task, I first, I last, I step, C callable
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaTask transform(I first, I last, O output, C op);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::transform
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    the task created from tf::cudaFlow::for_each.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    void transform(cudaTask task, I first, I last, O output, C c);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @return cudaTask handle
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaTask transform(I1 first1, I1 last1, I2 first2, O output, C op);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::transform
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    the task created from tf::cudaFlow::for_each.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:      cudaTask task, I1 first1, I1 last1, I2 first2, O output, C c
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @brief constructs a subflow graph through tf::cudaFlowCapturer
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:              @c std::function<void(tf::cudaFlowCapturer&)>
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    A captured subflow forms a sub-graph to the %cudaFlow and can be used to
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    from the %cudaFlow.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    taskflow.emplace([&](tf::cudaFlow& cf){
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:      tf::cudaTask my_kernel = cf.kernel(my_arguments);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:      tf::cudaTask my_subflow = cf.capture([&](tf::cudaFlowCapturer& capturer){
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:        capturer.on([&](cudaStream_t stream){
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaTask capture(C&& callable);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    The method is similar to tf::cudaFlow::capture but operates on a task
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    of type tf::cudaTaskType::SUBFLOW.
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    void capture(cudaTask task, C callable);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaFlowGraph _cfg;
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphExec _exe {nullptr};
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:// Construct a standalone cudaFlow
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:inline cudaFlow::cudaFlow() {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:inline void cudaFlow::clear() {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:inline bool cudaFlow::empty() const {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:inline size_t cudaFlow::num_tasks() const {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:inline void cudaFlow::dump(std::ostream& os) const {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:inline void cudaFlow::dump_native_graph(std::ostream& os) const {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  cuda_dump_graph(os, _cfg._native_handle);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:inline cudaTask cudaFlow::noop() {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    _cfg, std::in_place_type_t<cudaFlowNode::Empty>{}
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphAddEmptyNode(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:cudaTask cudaFlow::host(C&& c) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    _cfg, std::in_place_type_t<cudaFlowNode::Host>{}, std::forward<C>(c)
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  auto h = std::get_if<cudaFlowNode::Host>(&node->_handle);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  cudaHostNodeParams p;
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  p.fn = cudaFlowNode::Host::callback;
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphAddHostNode(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:cudaTask cudaFlow::kernel(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    _cfg, std::in_place_type_t<cudaFlowNode::Kernel>{}, (void*)f
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  cudaKernelNodeParams p;
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphAddKernelNode(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:cudaTask cudaFlow::zero(T* dst, size_t count) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    _cfg, std::in_place_type_t<cudaFlowNode::Memset>{}
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_zero_parms(dst, count);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphAddMemsetNode(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:cudaTask cudaFlow::fill(T* dst, T value, size_t count) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    _cfg, std::in_place_type_t<cudaFlowNode::Memset>{}
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_fill_parms(dst, value, count);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphAddMemsetNode(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:cudaTask cudaFlow::copy(T* tgt, const T* src, size_t num) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    _cfg, std::in_place_type_t<cudaFlowNode::Memcpy>{}
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_copy_parms(tgt, src, num);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphAddMemcpyNode(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:inline cudaTask cudaFlow::memset(void* dst, int ch, size_t count) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    _cfg, std::in_place_type_t<cudaFlowNode::Memset>{}
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_memset_parms(dst, ch, count);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphAddMemsetNode(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:inline cudaTask cudaFlow::memcpy(void* tgt, const void* src, size_t bytes) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    _cfg, std::in_place_type_t<cudaFlowNode::Memcpy>{}
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_memcpy_parms(tgt, src, bytes);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphAddMemcpyNode(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:void cudaFlow::host(cudaTask task, C&& c) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  if(task.type() != cudaTaskType::HOST) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  auto h = std::get_if<cudaFlowNode::Host>(&task._node->_handle);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:void cudaFlow::kernel(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  cudaTask task, dim3 g, dim3 b, size_t s, F f, ArgsT... args
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  if(task.type() != cudaTaskType::KERNEL) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  cudaKernelNodeParams p;
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphExecKernelNodeSetParams(_exe, task._node->_native_handle, &p),
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:void cudaFlow::copy(cudaTask task, T* tgt, const T* src, size_t num) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  if(task.type() != cudaTaskType::MEMCPY) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_copy_parms(tgt, src, num);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphExecMemcpyNodeSetParams(_exe, task._node->_native_handle, &p),
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:inline void cudaFlow::memcpy(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  cudaTask task, void* tgt, const void* src, size_t bytes
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  if(task.type() != cudaTaskType::MEMCPY) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_memcpy_parms(tgt, src, bytes);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphExecMemcpyNodeSetParams(_exe, task._node->_native_handle, &p),
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:inline void cudaFlow::memset(cudaTask task, void* dst, int ch, size_t count) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  if(task.type() != cudaTaskType::MEMSET) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_memset_parms(dst, ch, count);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphExecMemsetNodeSetParams(_exe, task._node->_native_handle, &p),
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:void cudaFlow::fill(cudaTask task, T* dst, T value, size_t count) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  if(task.type() != cudaTaskType::MEMSET) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_fill_parms(dst, value, count);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphExecMemsetNodeSetParams(_exe, task._node->_native_handle, &p),
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:void cudaFlow::zero(cudaTask task, T* dst, size_t count) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  if(task.type() != cudaTaskType::MEMSET) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_zero_parms(dst, count);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphExecMemsetNodeSetParams(_exe, task._node->_native_handle, &p),
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:void cudaFlow::capture(cudaTask task, C c) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  if(task.type() != cudaTaskType::SUBFLOW) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  auto node_handle = std::get_if<cudaFlowNode::Subflow>(&task._node->_handle);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  cudaFlowCapturer capturer;
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphExecChildGraphNodeSetParams(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:cudaTask cudaFlow::capture(C&& c) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    _cfg, std::in_place_type_t<cudaFlowNode::Subflow>{}
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  auto node_handle = std::get_if<cudaFlowNode::Subflow>(&node->_handle);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  cudaFlowCapturer capturer;
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  // move capturer's cudaFlow graph into node
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    cudaGraphAddChildGraphNode(
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:    "failed to add a cudaFlow capturer task"
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:inline void cudaFlow::run(cudaStream_t stream) {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:  _cfg._state = cudaFlowGraph::OFFLOADED;
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:inline cudaGraph_t cudaFlow::native_graph() {
tools/loaders/sharq/taskflow/cuda/cudaflow.hpp:inline cudaGraphExec_t cudaFlow::native_executable() {
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:#include "cuda_graph.hpp"
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:@file cuda_task.hpp
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:@brief cudaTask include file
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:// cudaTask Types
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:@enum cudaTaskType
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:@brief enumeration of all %cudaTask types
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:enum class cudaTaskType : int {
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:@brief convert a cuda_task type to a human-readable string
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:constexpr const char* to_string(cudaTaskType type) {
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::EMPTY:   return "empty";
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::HOST:    return "host";
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::MEMSET:  return "memset";
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::MEMCPY:  return "memcpy";
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::KERNEL:  return "kernel";
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::SUBFLOW: return "subflow";
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::CAPTURE: return "capture";
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:// cudaTask
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:@class cudaTask
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:@brief class to create a task handle over an internal node of a %cudaFlow graph
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:class cudaTask {
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:  friend class cudaFlow;
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:  friend class cudaFlowCapturer;
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:  friend class cudaFlowCapturerBase;
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:  friend std::ostream& operator << (std::ostream&, const cudaTask&);
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    @brief constructs an empty cudaTask
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    cudaTask() = default;
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    @brief copy-constructs a cudaTask
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    cudaTask(const cudaTask&) = default;
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    @brief copy-assigns a cudaTask
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    cudaTask& operator = (const cudaTask&) = default;
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    cudaTask& precede(Ts&&... tasks);
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    cudaTask& succeed(Ts&&... tasks);
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    cudaTask& name(const std::string& name);
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    @brief queries if the task is associated with a cudaFlowNode
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    cudaTaskType type() const;
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    cudaTask(cudaFlowNode*);
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    cudaFlowNode* _node {nullptr};
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:inline cudaTask::cudaTask(cudaFlowNode* node) : _node {node} {
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:cudaTask& cudaTask::precede(Ts&&... tasks) {
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:cudaTask& cudaTask::succeed(Ts&&... tasks) {
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:inline bool cudaTask::empty() const {
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:inline cudaTask& cudaTask::name(const std::string& name) {
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:inline const std::string& cudaTask::name() const {
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:inline size_t cudaTask::num_successors() const {
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:inline size_t cudaTask::num_dependents() const {
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:inline cudaTaskType cudaTask::type() const {
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    case cudaFlowNode::EMPTY:   return cudaTaskType::EMPTY;
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    case cudaFlowNode::HOST:    return cudaTaskType::HOST;
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    case cudaFlowNode::MEMSET:  return cudaTaskType::MEMSET;
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    case cudaFlowNode::MEMCPY:  return cudaTaskType::MEMCPY;
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    case cudaFlowNode::KERNEL:  return cudaTaskType::KERNEL;
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    case cudaFlowNode::SUBFLOW: return cudaTaskType::SUBFLOW;
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    case cudaFlowNode::CAPTURE: return cudaTaskType::CAPTURE;
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    default:                return cudaTaskType::UNDEFINED;
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:void cudaTask::dump(T& os) const {
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:  os << "cudaTask ";
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:void cudaTask::for_each_successor(V&& visitor) const {
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    visitor(cudaTask(_node->_successors[i]));
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:void cudaTask::for_each_dependent(V&& visitor) const {
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:    visitor(cudaTask(_node->_dependents[i]));
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:@brief overload of ostream inserter operator for cudaTask
tools/loaders/sharq/taskflow/cuda/cuda_task.hpp:inline std::ostream& operator << (std::ostream& os, const cudaTask& ct) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:#include "cuda_execution_policy.hpp"
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:inline constexpr unsigned CUDA_WARP_SIZE = 32;
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:struct cudaEmpty { };
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:struct cudaIterate {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:    cudaIterate<i + 1, count>::eval(f);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:struct cudaIterate<i, count, false> {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_iterate(F f) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaIterate<begin, end>::eval(f);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_iterate(F f) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_iterate<0, count>(f);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_iterate<count>([&](auto i) { y = i ? x[i] + y : x[i]; });
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_iterate<count>([&](auto i) { x[i] = val; });
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_strided_iterate(F f, unsigned tid) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_iterate<vt>([=](auto i) { f(i, nt * i + tid); });
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_strided_iterate(F f, unsigned tid, unsigned count) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:    cuda_strided_iterate<nt, vt0>(f, tid);    // No checking
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:    cuda_iterate<vt0>([=](auto i) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_iterate<vt0, vt>([=](auto i) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_thread_iterate(F f, unsigned tid) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_iterate<vt>([=](auto i) { f(i, vt * tid + i); });
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:// cudaRange
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:// cudaRange
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:struct cudaRange {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:inline __device__ cudaRange cuda_get_tile(unsigned b, unsigned nv, unsigned count) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  return cudaRange { nv * b, min(count, nv * (b + 1)) };
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:// cudaArray
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:struct cudaArray {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaArray() = default;
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaArray(const cudaArray&) = default;
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaArray& operator=(const cudaArray&) = default;
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  __device__ cudaArray(T x) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:    cuda_iterate<size>([&](unsigned i) { data[i] = x; });
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:struct cudaArray<T, 0> {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:struct cudaKVArray {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, size> keys;
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaArray<V, size> vals;
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_mem_to_reg_strided(I mem, unsigned tid, unsigned count) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x;
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_strided_iterate<nt, vt, vt0>(
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_reg_to_mem_strided(
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x, unsigned tid, unsigned count, it_t mem) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_strided_iterate<nt, vt, vt0>(
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_transform_mem_to_reg_strided(
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x;
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_strided_iterate<nt, vt, vt0>(
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_reg_to_shared_thread(
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x, unsigned tid, T (&shared)[shared_size], bool sync = true
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_thread_iterate<vt>([&](auto i, auto j) { shared[j] = x[i]; }, tid);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_shared_to_reg_thread(
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x;
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_thread_iterate<vt>([&](auto i, auto j) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_reg_to_shared_strided(
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x, unsigned tid, T (&shared)[shared_size], bool sync = true
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_strided_iterate<nt, vt>(
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_shared_to_reg_strided(
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x;
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_strided_iterate<nt, vt>([&](auto i, auto j) { x[i] = shared[j]; }, tid);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_reg_to_mem_thread(
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x, unsigned tid,
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_reg_to_shared_thread<nt>(x, tid, shared);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  auto y = cuda_shared_to_reg_strided<nt, vt>(shared, tid);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_reg_to_mem_strided<nt, vt, vt0>(y, tid, count, mem);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_mem_to_reg_thread(
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  auto x = cuda_mem_to_reg_strided<nt, vt, vt0>(mem, tid, count);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_reg_to_shared_strided<nt, vt>(x, tid, shared);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  auto y = cuda_shared_to_reg_thread<nt, vt>(shared, tid);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_shared_gather(
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  const T(&data)[S], cudaArray<unsigned, vt> indices, bool sync = true
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x;
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_iterate<vt>([&](auto i) { x[i] = data[indices[i]]; });
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_reg_thread_to_strided(
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x, unsigned tid, T (&shared)[S]
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_reg_to_shared_thread<nt>(x, tid, shared);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  return cuda_shared_to_reg_strided<nt, vt>(shared, tid);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_reg_strided_to_thread(
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x, unsigned tid, T (&shared)[S]
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cuda_reg_to_shared_strided<nt>(x, tid, shared);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  return cuda_shared_to_reg_thread<nt, vt>(shared, tid);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:// cudaLoadStoreIterator
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:struct cudaLoadStoreIterator : std::iterator_traits<const T*> {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  cudaLoadStoreIterator(L load_, S store_, I base_) :
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:      static_assert(!std::is_same<S, cudaEmpty>::value,
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:      static_assert(!std::is_same<L, cudaEmpty>::value,
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  __device__ cudaLoadStoreIterator operator+(I offset) const {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:    cudaLoadStoreIterator cp = *this;
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  __device__ cudaLoadStoreIterator& operator+=(I offset) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  __device__ cudaLoadStoreIterator operator-(I offset) const {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:    cudaLoadStoreIterator cp = *this;
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  __device__ cudaLoadStoreIterator& operator-=(I offset) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:auto cuda_make_load_store_iterator(L load, S store, I base = 0) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  return cudaLoadStoreIterator<L, S, T, I>(load, store, base);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:auto cuda_make_load_iterator(L load, I base = 0) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  return cuda_make_load_store_iterator<T>(load, cudaEmpty(), base);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:auto cuda_make_store_iterator(S store, I base = 0) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:  return cuda_make_load_store_iterator<T>(cudaEmpty(), store, base);
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_swap(T& a, T& b) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:__global__ void cuda_kernel(F f, args_t... args) {
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:struct cuda_plus{
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:struct cuda_minus{
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:struct cuda_multiplies{
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:struct cuda_maximum{
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:struct cuda_minimum{
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:struct cuda_less{
tools/loaders/sharq/taskflow/cuda/cuda_meta.hpp:struct cuda_greater{
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:#include "cuda_task.hpp"
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:#include "cuda_optimizer.hpp"
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:@file cuda_capturer.hpp
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:@brief %cudaFlow capturer include file
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:// class definition: cudaFlowCapturer
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:@class cudaFlowCapturer
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:@brief class to create a %cudaFlow graph using stream capture
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:The usage of tf::cudaFlowCapturer is similar to tf::cudaFlow, except users can
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:call the method tf::cudaFlowCapturer::on to capture a sequence of asynchronous
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:CUDA operations through the given stream.
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:The following example creates a CUDA graph that captures two kernel tasks,
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:taskflow.emplace([](tf::cudaFlowCapturer& capturer){
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  auto task_1 = capturer.on([&](cudaStream_t stream){
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  auto task_2 = capturer.on([&](cudaStream_t stream){
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:Similar to tf::cudaFlow, a %cudaFlowCapturer is a task (tf::Task)
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:That is, the callable that describes a %cudaFlowCapturer
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:Inside a %cudaFlow capturer task, different GPU tasks (tf::cudaTask) may run
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:By default, we use tf::cudaFlowRoundRobinOptimizer to transform a user-level
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:graph into a native CUDA graph.
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:Please refer to @ref GPUTaskingcudaFlowCapturer for details.
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:class cudaFlowCapturer {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  friend class cudaFlow;
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaFlowGraph graph;
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  // created from cudaFlow
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaFlowRoundRobinOptimizer,
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaFlowSequentialOptimizer,
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaFlowLinearOptimizer
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @brief constrcts a standalone cudaFlowCapturer
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    A standalone %cudaFlow capturer does not go through any taskflow and
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    can be run by the caller thread using tf::cudaFlowCapturer::run.
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaFlowCapturer() = default;
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @brief destructs the cudaFlowCapturer
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    ~cudaFlowCapturer() = default;
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaFlowCapturer(cudaFlowCapturer&&) = default;
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaFlowCapturer& operator = (cudaFlowCapturer&&) = default;
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @brief clear this %cudaFlow capturer
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @brief dumps the %cudaFlow graph into a DOT format through an
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @brief captures a sequential CUDA operations from the given callable
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @tparam C callable type constructible with @c std::function<void(cudaStream_t)>
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @param callable a callable to capture CUDA operations with the stream
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    a sequence of CUDA operations defined in the callable.
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:      std::is_invocable_r_v<void, C, cudaStream_t>, void>* = nullptr
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaTask on(C&& callable);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @brief updates a capture task to another sequential CUDA operations
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    The method is similar to cudaFlowCapturer::on but operates
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:      std::is_invocable_r_v<void, C, cudaStream_t>, void>* = nullptr
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    void on(cudaTask task, C&& callable);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @return a tf::cudaTask handle
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaTask noop();
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    The method is similar to tf::cudaFlowCapturer::noop but
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    void noop(cudaTask task);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    The method captures a @c cudaMemcpyAsync operation through an
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaTask memcpy(void* dst, const void* src, size_t count);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    The method is similar to cudaFlowCapturer::memcpy but operates on an
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    void memcpy(cudaTask task, void* dst, const void* src, size_t count);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @return cudaTask handle
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    to a target location. Direction can be arbitrary among CPUs and GPUs.
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaTask copy(T* tgt, const T* src, size_t num);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    The method is similar to cudaFlowCapturer::copy but operates on
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    void copy(cudaTask task, T* tgt, const T* src, size_t num);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @brief initializes or sets GPU memory to the given value byte by byte
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @param ptr pointer to GPU mempry
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    The method captures a @c cudaMemsetAsync operation through an
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaTask memset(void* ptr, int v, size_t n);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    The method is similar to cudaFlowCapturer::memset but operates on
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    void memset(cudaTask task, void* ptr, int value, size_t n);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @return cudaTask handle
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaTask kernel(dim3 g, dim3 b, size_t s, F f, ArgsT&&... args);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    The method is similar to cudaFlowCapturer::kernel but operates on
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:      cudaTask task, dim3 g, dim3 b, size_t s, F f, ArgsT&&... args
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaTask single_task(C c);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::single_task but operates
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    void single_task(cudaTask task, C c);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @return cudaTask handle
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaTask for_each(I first, I last, C callable);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::for_each but operates
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    void for_each(cudaTask task, I first, I last, C callable);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @return cudaTask handle
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaTask for_each_index(I first, I last, I step, C callable);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::for_each_index but operates
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:      cudaTask task, I first, I last, I step, C callable
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @return cudaTask handle
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaTask transform(I first, I last, O output, C op);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::transform but operates
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    void transform(cudaTask task, I first, I last, O output, C op);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @return cudaTask handle
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaTask transform(I1 first1, I1 last1, I2 first2, O output, C op);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::transform but operates
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:      cudaTask task, I1 first1, I1 last1, I2 first2, O output, C op
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    a user-described %cudaFlow:
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:      + tf::cudaFlowSequentialOptimizer
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:      + tf::cudaFlowRoundRobinOptimizer
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:      + tf::cudaFlowLinearOptimizer
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    By default, tf::cudaFlowCapturer uses the round-robin optimization
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    a native CUDA graph.
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @brief captures the cudaFlow and turns it into a CUDA Graph
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaGraph_t capture();
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @brief offloads the %cudaFlowCapturer onto a GPU asynchronously via a stream
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    Offloads the present %cudaFlowCapturer onto a GPU asynchronously via
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    An offloaded %cudaFlowCapturer forces the underlying graph to be instantiated.
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    void run(cudaStream_t stream);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @brief acquires a reference to the underlying CUDA graph
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaGraph_t native_graph();
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    @brief acquires a reference to the underlying CUDA graph executable
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaGraphExec_t native_executable();
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaFlowGraph _cfg;
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    cudaGraphExec _exe {nullptr};
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:inline bool cudaFlowCapturer::empty() const {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:inline size_t cudaFlowCapturer::num_tasks() const {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::clear() {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::dump(std::ostream& os) const {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::dump_native_graph(std::ostream& os) const {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  cuda_dump_graph(os, _cfg._native_handle);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  std::is_invocable_r_v<void, C, cudaStream_t>, void>*
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::on(C&& callable) {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    std::in_place_type_t<cudaFlowNode::Capture>{}, std::forward<C>(callable)
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  return cudaTask(node);
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:inline cudaTask cudaFlowCapturer::noop() {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  return on([](cudaStream_t){});
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::noop(cudaTask task) {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  on(task, [](cudaStream_t){});
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:inline cudaTask cudaFlowCapturer::memcpy(
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  return on([dst, src, count] (cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:      cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream),
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::copy(T* tgt, const T* src, size_t num) {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  return on([tgt, src, num] (cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:      cudaMemcpyAsync(tgt, src, sizeof(T)*num, cudaMemcpyDefault, stream),
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:inline cudaTask cudaFlowCapturer::memset(void* ptr, int v, size_t n) {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  return on([ptr, v, n] (cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:      cudaMemsetAsync(ptr, v, n, stream), "failed to capture memset"
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::kernel(
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  return on([g, b, s, f, args...] (cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:inline cudaGraph_t cudaFlowCapturer::capture() {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::run(cudaStream_t stream) {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  if(_cfg._state & cudaFlowGraph::CHANGED) {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  else if(_cfg._state & cudaFlowGraph::UPDATED) {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    if(_exe.update(_cfg._native_handle) != cudaGraphExecUpdateSuccess) {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  _cfg._state = cudaFlowGraph::OFFLOADED;
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:inline cudaGraph_t cudaFlowCapturer::native_graph() {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:inline cudaGraphExec_t cudaFlowCapturer::native_executable() {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  std::is_invocable_r_v<void, C, cudaStream_t>, void>*
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:void cudaFlowCapturer::on(cudaTask task, C&& callable) {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  if(task.type() != cudaTaskType::CAPTURE) {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    TF_THROW("invalid cudaTask type (must be CAPTURE)");
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  _cfg._state |= cudaFlowGraph::UPDATED;
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  std::get_if<cudaFlowNode::Capture>(&task._node->_handle)->work =
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::memcpy(
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  cudaTask task, void* dst, const void* src, size_t count
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  on(task, [dst, src, count](cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:      cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream),
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:void cudaFlowCapturer::copy(
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  cudaTask task, T* tgt, const T* src, size_t num
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  on(task, [tgt, src, num] (cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:      cudaMemcpyAsync(tgt, src, sizeof(T)*num, cudaMemcpyDefault, stream),
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::memset(
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  cudaTask task, void* ptr, int v, size_t n
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  on(task, [ptr, v, n] (cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:      cudaMemsetAsync(ptr, v, n, stream), "failed to capture memset"
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:void cudaFlowCapturer::kernel(
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  cudaTask task, dim3 g, dim3 b, size_t s, F f, ArgsT&&... args
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:  on(task, [g, b, s, f, args...] (cudaStream_t stream) mutable {
tools/loaders/sharq/taskflow/cuda/cuda_capturer.hpp:OPT& cudaFlowCapturer::make_optimizer(ArgsT&&... args) {
tools/loaders/sharq/spdlog/fmt/bundled/format-inl.h:  // Chosen instead of std::abort to satisfy Clang in CUDA mode during device
tools/loaders/sharq/spdlog/fmt/bundled/format.h:#  define FMT_CUDA_VERSION (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__)
tools/loaders/sharq/spdlog/fmt/bundled/format.h:#  define FMT_CUDA_VERSION 0
tools/loaders/sharq/spdlog/fmt/bundled/format.h:// EDG based compilers (Intel, NVIDIA, Elbrus, etc), GCC and MSVC support UDLs.
tools/loaders/sharq/CLI11.hpp:#ifdef __CUDACC__
tools/loaders/sharq/CLI11.hpp:#ifdef __CUDACC__
tools/loaders/bam-loader/taskflow/sycl/syclflow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/sycl/syclflow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/sycl/syclflow.hpp:    @brief offloads the %syclFlow onto a GPU and repeatedly runs it until 
tools/loaders/bam-loader/taskflow/sycl/sycl_task.hpp:@brief handle to a node of the internal CUDA graph
tools/loaders/bam-loader/taskflow/core/task.hpp:  /** @brief cudaFlow task type */
tools/loaders/bam-loader/taskflow/core/task.hpp:  CUDAFLOW,
tools/loaders/bam-loader/taskflow/core/task.hpp:  TaskType::CUDAFLOW,
tools/loaders/bam-loader/taskflow/core/task.hpp:TaskType::CUDAFLOW        ->  "cudaflow"
tools/loaders/bam-loader/taskflow/core/task.hpp:    case TaskType::CUDAFLOW:         val = "cudaflow";        break;
tools/loaders/bam-loader/taskflow/core/task.hpp:@brief determines if a callable is a %cudaFlow task
tools/loaders/bam-loader/taskflow/core/task.hpp:A cudaFlow task is a callable object constructible from
tools/loaders/bam-loader/taskflow/core/task.hpp:std::function<void(tf::cudaFlow&)> or std::function<void(tf::cudaFlowCapturer&)>.
tools/loaders/bam-loader/taskflow/core/task.hpp:constexpr bool is_cudaflow_task_v = std::is_invocable_r_v<void, C, cudaFlow&> ||
tools/loaders/bam-loader/taskflow/core/task.hpp:                                    std::is_invocable_r_v<void, C, cudaFlowCapturer&>;
tools/loaders/bam-loader/taskflow/core/task.hpp:           and cudaFlow tasks
tools/loaders/bam-loader/taskflow/core/task.hpp:    case Node::CUDAFLOW:        return TaskType::CUDAFLOW;
tools/loaders/bam-loader/taskflow/core/task.hpp:  else if constexpr(is_cudaflow_task_v<C>) {
tools/loaders/bam-loader/taskflow/core/task.hpp:    _node->_handle.emplace<Node::cudaFlow>(std::forward<C>(c));
tools/loaders/bam-loader/taskflow/core/task.hpp:@brief overload of ostream inserter operator for cudaTask
tools/loaders/bam-loader/taskflow/core/task.hpp:    case Node::CUDAFLOW:        return TaskType::CUDAFLOW;
tools/loaders/bam-loader/taskflow/core/declarations.hpp:// cudaFlow
tools/loaders/bam-loader/taskflow/core/declarations.hpp:class cudaNode;
tools/loaders/bam-loader/taskflow/core/declarations.hpp:class cudaGraph;
tools/loaders/bam-loader/taskflow/core/declarations.hpp:class cudaTask;
tools/loaders/bam-loader/taskflow/core/declarations.hpp:class cudaFlow;
tools/loaders/bam-loader/taskflow/core/declarations.hpp:class cudaFlowCapturer;
tools/loaders/bam-loader/taskflow/core/declarations.hpp:class cudaFlowCapturerBase;
tools/loaders/bam-loader/taskflow/core/declarations.hpp:class cudaCapturingBase;
tools/loaders/bam-loader/taskflow/core/declarations.hpp:class cudaLinearCapturing;
tools/loaders/bam-loader/taskflow/core/declarations.hpp:class cudaSequentialCapturing;
tools/loaders/bam-loader/taskflow/core/declarations.hpp:class cudaRoundRobinCapturing;
tools/loaders/bam-loader/taskflow/core/executor-module-opt.hpp:    void _invoke_cudaflow_task(Worker&, Node*);
tools/loaders/bam-loader/taskflow/core/executor-module-opt.hpp:      std::enable_if_t<is_cudaflow_task_v<C>, void>* = nullptr
tools/loaders/bam-loader/taskflow/core/executor-module-opt.hpp:    void _invoke_cudaflow_task_entry(Node*, C&&);
tools/loaders/bam-loader/taskflow/core/executor-module-opt.hpp:    // cudaflow task
tools/loaders/bam-loader/taskflow/core/executor-module-opt.hpp:    case Node::CUDAFLOW: {
tools/loaders/bam-loader/taskflow/core/executor-module-opt.hpp:      _invoke_cudaflow_task(worker, node);
tools/loaders/bam-loader/taskflow/core/executor-module-opt.hpp:// Procedure: _invoke_cudaflow_task
tools/loaders/bam-loader/taskflow/core/executor-module-opt.hpp:inline void Executor::_invoke_cudaflow_task(Worker& worker, Node* node) {
tools/loaders/bam-loader/taskflow/core/executor-module-opt.hpp:  std::get_if<Node::cudaFlow>(&node->_handle)->work(*this, node);
tools/loaders/bam-loader/taskflow/core/executor.hpp:    void _invoke_cudaflow_task(Worker&, Node*);
tools/loaders/bam-loader/taskflow/core/executor.hpp:      std::enable_if_t<is_cudaflow_task_v<C>, void>* = nullptr
tools/loaders/bam-loader/taskflow/core/executor.hpp:    void _invoke_cudaflow_task_entry(Node*, C&&);
tools/loaders/bam-loader/taskflow/core/executor.hpp:    // cudaflow task
tools/loaders/bam-loader/taskflow/core/executor.hpp:    case Node::CUDAFLOW: {
tools/loaders/bam-loader/taskflow/core/executor.hpp:      _invoke_cudaflow_task(worker, node);
tools/loaders/bam-loader/taskflow/core/executor.hpp:// Procedure: _invoke_cudaflow_task
tools/loaders/bam-loader/taskflow/core/executor.hpp:inline void Executor::_invoke_cudaflow_task(Worker& worker, Node* node) {
tools/loaders/bam-loader/taskflow/core/executor.hpp:  std::get_if<Node::cudaFlow>(&node->_handle)->work(*this, node);
tools/loaders/bam-loader/taskflow/core/taskflow.hpp:  7. %cudaFlow task      : the callable constructible from
tools/loaders/bam-loader/taskflow/core/taskflow.hpp:                           @c std::function<void(tf::cudaFlow&)> or
tools/loaders/bam-loader/taskflow/core/taskflow.hpp:                           @c std::function<void(tf::cudaFlowCapturer&)>
tools/loaders/bam-loader/taskflow/core/taskflow.hpp:    and GPU tasks, you need to run the taskflow first before you can
tools/loaders/bam-loader/taskflow/core/taskflow.hpp:    case Node::CUDAFLOW:
tools/loaders/bam-loader/taskflow/core/taskflow.hpp:    case Node::CUDAFLOW: {
tools/loaders/bam-loader/taskflow/core/taskflow.hpp:      std::get_if<Node::cudaFlow>(&node->_handle)->graph->dump(
tools/loaders/bam-loader/taskflow/core/graph.hpp:  // cudaFlow work handle
tools/loaders/bam-loader/taskflow/core/graph.hpp:  struct cudaFlow {
tools/loaders/bam-loader/taskflow/core/graph.hpp:    cudaFlow(C&& c, G&& g);
tools/loaders/bam-loader/taskflow/core/graph.hpp:    cudaFlow,        // cudaFlow
tools/loaders/bam-loader/taskflow/core/graph.hpp:  constexpr static auto CUDAFLOW        = get_index_v<cudaFlow, handle_t>;
tools/loaders/bam-loader/taskflow/core/graph.hpp:// Definition for Node::cudaFlow
tools/loaders/bam-loader/taskflow/core/graph.hpp:Node::cudaFlow::cudaFlow(C&& c, G&& g) :
tools/loaders/bam-loader/taskflow/core/flow_builder.hpp:    @brief creates a %cudaFlow task on the caller's GPU device context
tools/loaders/bam-loader/taskflow/core/flow_builder.hpp:    @tparam C callable type constructible from @c std::function<void(tf::cudaFlow&)>
tools/loaders/bam-loader/taskflow/core/flow_builder.hpp:    The following example creates a %cudaFlow of two kernel tasks, @c task1 and
tools/loaders/bam-loader/taskflow/core/flow_builder.hpp:    taskflow.emplace([&](tf::cudaFlow& cf){
tools/loaders/bam-loader/taskflow/core/flow_builder.hpp:      tf::cudaTask task1 = cf.kernel(grid1, block1, shm1, kernel1, args1);
tools/loaders/bam-loader/taskflow/core/flow_builder.hpp:      tf::cudaTask task2 = cf.kernel(grid2, block2, shm2, kernel2, args2);
tools/loaders/bam-loader/taskflow/core/flow_builder.hpp:    Please refer to @ref GPUTaskingcudaFlow and @ref GPUTaskingcudaFlowCapturer
tools/loaders/bam-loader/taskflow/core/flow_builder.hpp:      std::enable_if_t<is_cudaflow_task_v<C>, void>* = nullptr
tools/loaders/bam-loader/taskflow/core/flow_builder.hpp:    @brief creates a %cudaFlow task on the given device
tools/loaders/bam-loader/taskflow/core/flow_builder.hpp:    @tparam C callable type constructible from std::function<void(tf::cudaFlow&)>
tools/loaders/bam-loader/taskflow/core/flow_builder.hpp:    The following example creates a %cudaFlow of two kernel tasks, @c task1 and
tools/loaders/bam-loader/taskflow/core/flow_builder.hpp:    @c task2 on GPU @c 2, where @c task1 runs before @c task2
tools/loaders/bam-loader/taskflow/core/flow_builder.hpp:    taskflow.emplace_on([&](tf::cudaFlow& cf){
tools/loaders/bam-loader/taskflow/core/flow_builder.hpp:      tf::cudaTask task1 = cf.kernel(grid1, block1, shm1, kernel1, args1);
tools/loaders/bam-loader/taskflow/core/flow_builder.hpp:      tf::cudaTask task2 = cf.kernel(grid2, block2, shm2, kernel2, args2);
tools/loaders/bam-loader/taskflow/core/flow_builder.hpp:      std::enable_if_t<is_cudaflow_task_v<C>, void>* = nullptr
tools/loaders/bam-loader/taskflow/taskflow.hpp:@dir taskflow/cuda
tools/loaders/bam-loader/taskflow/taskflow.hpp:@brief taskflow CUDA include dir
tools/loaders/bam-loader/taskflow/cuda/cuda_error.hpp:#include <cuda.h>
tools/loaders/bam-loader/taskflow/cuda/cuda_error.hpp:#define TF_CUDA_EXPAND( x ) x
tools/loaders/bam-loader/taskflow/cuda/cuda_error.hpp:#define TF_CUDA_REMOVE_FIRST_HELPER(N, ...) __VA_ARGS__
tools/loaders/bam-loader/taskflow/cuda/cuda_error.hpp:#define TF_CUDA_REMOVE_FIRST(...) TF_CUDA_EXPAND(TF_CUDA_REMOVE_FIRST_HELPER(__VA_ARGS__))
tools/loaders/bam-loader/taskflow/cuda/cuda_error.hpp:#define TF_CUDA_GET_FIRST_HELPER(N, ...) N
tools/loaders/bam-loader/taskflow/cuda/cuda_error.hpp:#define TF_CUDA_GET_FIRST(...) TF_CUDA_EXPAND(TF_CUDA_GET_FIRST_HELPER(__VA_ARGS__))
tools/loaders/bam-loader/taskflow/cuda/cuda_error.hpp:#define TF_CHECK_CUDA(...)                                       \
tools/loaders/bam-loader/taskflow/cuda/cuda_error.hpp:if(TF_CUDA_GET_FIRST(__VA_ARGS__) != cudaSuccess) {              \
tools/loaders/bam-loader/taskflow/cuda/cuda_error.hpp:  auto __ev__ = TF_CUDA_GET_FIRST(__VA_ARGS__);                  \
tools/loaders/bam-loader/taskflow/cuda/cuda_error.hpp:      << (cudaGetErrorString(__ev__)) << " ("                    \
tools/loaders/bam-loader/taskflow/cuda/cuda_error.hpp:      << (cudaGetErrorName(__ev__)) << ") - ";                   \
tools/loaders/bam-loader/taskflow/cuda/cuda_error.hpp:  tf::ostreamize(oss, TF_CUDA_REMOVE_FIRST(__VA_ARGS__));        \
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:#include "../cudaflow.hpp"
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:@file taskflow/cuda/algorithm/for_each.hpp
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:@brief cuda parallel-iteration algorithms include file
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:void cuda_for_each_loop(P&& p, I first, unsigned count, C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  cuda_kernel<<<B, E::nt, 0, p.stream()>>>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    auto tile = cuda_get_tile(bid, E::nv, count);
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cuda_strided_iterate<E::nt, E::vt>([=](auto, auto j) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:void cuda_for_each_index_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  cuda_kernel<<<B, E::nt, 0, p.stream()>>>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    auto tile = cuda_get_tile(bid, E::nv, count);
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cuda_strided_iterate<E::nt, E::vt>([=]__device__(auto, auto j) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:// cuda standard algorithms: single_task/for_each/for_each_index
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:void cuda_single_task(P&& p, C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  cuda_kernel<<<1, 1, 0, p.stream()>>>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:on a GPU:
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:void cuda_for_each(P&& p, I first, I last, C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  detail::cuda_for_each_loop(p, first, count, c);
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:void cuda_for_each_index(P&& p, I first, I last, I inc, C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  detail::cuda_for_each_index_loop(p, first, inc, count, c);
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:__global__ void cuda_single_task(C callable) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:cudaTask cudaFlow::single_task(C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  return kernel(1, 1, 0, cuda_single_task<C>, c);
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:void cudaFlow::single_task(cudaTask task, C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  return kernel(task, 1, 1, 0, cuda_single_task<C>, c);
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:// cudaFlow
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:cudaTask cudaFlow::for_each(I first, I last, C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  return capture([=](cudaFlowCapturer& cap) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:cudaTask cudaFlow::for_each_index(I first, I last, I inc, C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  return capture([=](cudaFlowCapturer& cap) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:void cudaFlow::for_each(cudaTask task, I first, I last, C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  capture(task, [=](cudaFlowCapturer& cap) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:void cudaFlow::for_each_index(cudaTask task, I first, I last, I inc, C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  capture(task, [=](cudaFlowCapturer& cap) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:// cudaFlowCapturer
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:cudaTask cudaFlowCapturer::for_each(I first, I last, C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  return on([=](cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cuda_for_each(p, first, last, c);
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:cudaTask cudaFlowCapturer::for_each_index(I beg, I end, I inc, C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  return on([=] (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cuda_for_each_index(p, beg, end, inc, c);
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:void cudaFlowCapturer::for_each(cudaTask task, I first, I last, C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  on(task, [=](cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cuda_for_each(p, first, last, c);
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:void cudaFlowCapturer::for_each_index(
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  cudaTask task, I beg, I end, I inc, C c
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  on(task, [=] (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cuda_for_each_index(p, beg, end, inc, c);
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:cudaTask cudaFlowCapturer::single_task(C callable) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  return on([=] (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cuda_single_task(p, callable);
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:void cudaFlowCapturer::single_task(cudaTask task, C callable) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:  on(task, [=] (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/for_each.hpp:    cuda_single_task(p, callable);
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:@file taskflow/cuda/algorithm/scan.hpp
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:@brief CUDA scan algorithm include file
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:inline constexpr unsigned cudaScanRecursionThreshold = 8;
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:enum class cudaScanType : int {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:struct cudaScanResult {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:struct cudaScanResult<T, vt, true> {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  cudaArray<T, vt> scan;
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:struct cudaBlockScan {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  const static unsigned num_warps  = nt / CUDA_WARP_SIZE;
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  __device__ cudaScanResult<T> operator ()(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cudaScanType type = cudaScanType::EXCLUSIVE
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  __device__ cudaScanResult<T, vt> operator()(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cudaArray<T, vt> x,
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cudaScanType type = cudaScanType::EXCLUSIVE
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:__device__ cudaScanResult<T> cudaBlockScan<nt, T>::operator () (
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  T init, cudaScanType type
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  cuda_iterate<num_passes>([&](auto pass) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  cudaScanResult<T> result;
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    (cudaScanType::INCLUSIVE == type ? x :
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:__device__ cudaScanResult<T, vt> cudaBlockScan<nt, T>::operator()(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  cudaArray<T, vt> x,
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  cudaScanType type
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cuda_iterate<vt>([&](auto i) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cuda_iterate<vt>([&](auto i) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    (count + vt - 1) / vt, op, init, cudaScanType::EXCLUSIVE
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  cudaArray<T, vt> y;
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  cuda_iterate<vt>([&](auto i) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    if(cudaScanType::EXCLUSIVE == type) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  return cudaScanResult<T, vt> { y, result.reduction };
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:void cuda_single_pass_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  cudaScanType scan_type,
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  cuda_kernel<<<1, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    using scan_t = cudaBlockScan<E::nt, T>;
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:      auto x = cuda_mem_to_reg_thread<E::nt, E::vt>(input + cur,
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:      cuda_reg_to_mem_thread<E::nt, E::vt>(result.scan, tid, count2,
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:void cuda_scan_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  cudaScanType scan_type,
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  if(B > cudaScanRecursionThreshold) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    //cudaDeviceVector<T> partials(B);
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:      __shared__ typename cudaBlockReduce<E::nt, T>::Storage shm;
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:      auto tile = cuda_get_tile(bid, E::nv, count);
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:      auto x = cuda_mem_to_reg_strided<E::nt, E::vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:      cuda_strided_iterate<E::nt, E::vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:      auto all_reduce = cudaBlockReduce<E::nt, T>()(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    //cuda_scan_loop(p, cudaScanType::EXCLUSIVE, buffer, B, buffer, op, S);
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cuda_scan_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:      p, cudaScanType::EXCLUSIVE, buffer, B, buffer, op, buffer+B
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:      using scan_t = cudaBlockScan<E::nt, T>;
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:      auto tile = cuda_get_tile(bid, E::nv, count);
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:      auto x = cuda_mem_to_reg_thread<E::nt, E::vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:      cuda_reg_to_mem_thread<E::nt, E::vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cuda_single_pass_scan(p, scan_type, input, count, output, op);
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:tf::cuda_inclusive_scan, tf::cuda_exclusive_scan,
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:tf::cuda_transform_inclusive_scan, and tf::cuda_transform_exclusive_scan.
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:unsigned cuda_scan_buffer_size(unsigned count) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  for(auto b=B; b>detail::cudaScanRecursionThreshold; b=(b+E::nv-1)/E::nv) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://void cuda_inclusive_scan(P&& p, I first, I last, O output, C op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://  cudaDeviceVector<std::byte> temp(cuda_scan_buffer_size<P, T>(count));
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://  detail::cuda_scan_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://    p, detail::cudaScanType::INCLUSIVE, first, count, output, op, temp.data()
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:void cuda_inclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  detail::cuda_scan_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    p, detail::cudaScanType::INCLUSIVE, first, count, output, op, buf
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://void cuda_transform_inclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://  cudaDeviceVector<std::byte> temp(cuda_scan_buffer_size<P, T>(count));
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://  detail::cuda_scan_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://    p, detail::cudaScanType::INCLUSIVE,
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://    cuda_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }),
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:void cuda_transform_inclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  detail::cuda_scan_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    p, detail::cudaScanType::INCLUSIVE,
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cuda_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }),
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://void cuda_exclusive_scan(P&& p, I first, I last, O output, C op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://  cudaDeviceVector<std::byte> temp(cuda_scan_buffer_size<P, T>(count));
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://  detail::cuda_scan_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://    p, detail::cudaScanType::EXCLUSIVE, first, count, output, op, buf
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:void cuda_exclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  detail::cuda_scan_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    p, detail::cudaScanType::EXCLUSIVE, first, count, output, op, buf
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://void cuda_transform_exclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://  cudaDeviceVector<std::byte> temp(cuda_scan_buffer_size<P, T>(count));
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://  detail::cuda_scan_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://    p, detail::cudaScanType::EXCLUSIVE,
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp://    cuda_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }),
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:void cuda_transform_exclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  detail::cuda_scan_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    p, detail::cudaScanType::EXCLUSIVE,
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cuda_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }),
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:// cudaFlow
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:cudaTask cudaFlow::inclusive_scan(I first, I last, O output, C op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  return capture([=](cudaFlowCapturer& cap) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:void cudaFlow::inclusive_scan(cudaTask task, I first, I last, O output, C op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  capture(task, [=](cudaFlowCapturer& cap) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:cudaTask cudaFlow::exclusive_scan(I first, I last, O output, C op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  return capture([=](cudaFlowCapturer& cap) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:void cudaFlow::exclusive_scan(cudaTask task, I first, I last, O output, C op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  capture(task, [=](cudaFlowCapturer& cap) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:cudaTask cudaFlow::transform_inclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  return capture([=](cudaFlowCapturer& cap) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:void cudaFlow::transform_inclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  cudaTask task, I first, I last, O output, B bop, U uop
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  capture(task, [=](cudaFlowCapturer& cap) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:cudaTask cudaFlow::transform_exclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  return capture([=](cudaFlowCapturer& cap) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:void cudaFlow::transform_exclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  cudaTask task, I first, I last, O output, B bop, U uop
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  capture(task, [=](cudaFlowCapturer& cap) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:// cudaFlowCapturer
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:cudaTask cudaFlowCapturer::inclusive_scan(I first, I last, O output, C op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  auto bufsz = cuda_scan_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  return on([=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cuda_inclusive_scan(p, first, last, output, op, buf.get().data());
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:void cudaFlowCapturer::inclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  cudaTask task, I first, I last, O output, C op
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  auto bufsz = cuda_scan_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  on(task, [=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cuda_inclusive_scan(p, first, last, output, op, buf.get().data());
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:cudaTask cudaFlowCapturer::exclusive_scan(I first, I last, O output, C op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  auto bufsz = cuda_scan_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  return on([=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cuda_exclusive_scan(p, first, last, output, op, buf.get().data());
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:void cudaFlowCapturer::exclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  cudaTask task, I first, I last, O output, C op
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  auto bufsz = cuda_scan_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  on(task, [=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cuda_exclusive_scan(p, first, last, output, op, buf.get().data());
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:cudaTask cudaFlowCapturer::transform_inclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  auto bufsz = cuda_scan_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  return on([=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cuda_transform_inclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:void cudaFlowCapturer::transform_inclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  cudaTask task, I first, I last, O output, B bop, U uop
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  auto bufsz = cuda_scan_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  on(task, [=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cuda_transform_inclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:cudaTask cudaFlowCapturer::transform_exclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  auto bufsz = cuda_scan_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  return on([=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cuda_transform_exclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:void cudaFlowCapturer::transform_exclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  cudaTask task, I first, I last, O output, B bop, U uop
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  auto bufsz = cuda_scan_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  on(task, [=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/scan.hpp:    cuda_transform_exclusive_scan(
tools/loaders/bam-loader/taskflow/cuda/algorithm/matmul.hpp:#include "../cudaflow.hpp"
tools/loaders/bam-loader/taskflow/cuda/algorithm/matmul.hpp:__global__ void cuda_matmul(
tools/loaders/bam-loader/taskflow/cuda/algorithm/transpose.hpp:#include "../cuda_error.hpp"
tools/loaders/bam-loader/taskflow/cuda/algorithm/transpose.hpp:__global__ void cuda_transpose(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:#include "../cudaflow.hpp"
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:@file taskflow/cuda/algorithm/reduce.hpp
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:@brief cuda reduce algorithms include file
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:struct cudaBlockReduce {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  static const unsigned group_size = std::min(nt, CUDA_WARP_SIZE);
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    nt && (0 == nt % CUDA_WARP_SIZE),
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    "cudaBlockReduce requires num threads to be a multiple of warp_size (32)"
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:__device__ T cudaBlockReduce<nt, T>::operator ()(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cuda_strided_iterate<group_size, num_items>([&](auto i, auto j) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  cuda_iterate<num_passes>([&](auto pass) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:void cuda_reduce_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    __shared__ typename cudaBlockReduce<E::nt, U>::Storage shm;
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    auto tile = cuda_get_tile(bid, E::nv, count);
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    auto x = cuda_mem_to_reg_strided<E::nt, E::vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cuda_strided_iterate<E::nt, E::vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    s = cudaBlockReduce<E::nt, U>()(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cuda_reduce_loop(p, buf, B, res, op, buf+B);
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:void cuda_uninitialized_reduce_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    __shared__ typename cudaBlockReduce<E::nt, U>::Storage shm;
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    auto tile = cuda_get_tile(bid, E::nv, count);
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    auto x = cuda_mem_to_reg_strided<E::nt, E::vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cuda_strided_iterate<E::nt, E::vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    s = cudaBlockReduce<E::nt, U>()(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cuda_uninitialized_reduce_loop(p, buf, B, res, op, buf+B);
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:The function is used to allocate a buffer for calling tf::cuda_reduce,
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:tf::cuda_uninitialized_reduce, tf::cuda_transform_reduce, and
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:tf::cuda_transform_uninitialized_reduce.
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:unsigned cuda_reduce_buffer_size(unsigned count) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:// cuda_reduce
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:void cuda_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  detail::cuda_reduce_loop(p, first, count, res, op, buf);
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:// cuda_uninitialized_reduce
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:on a GPU:
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:void cuda_uninitialized_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  detail::cuda_uninitialized_reduce_loop(p, first, count, res, op, buf);
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:void cuda_transform_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  detail::cuda_reduce_loop(p,
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cuda_make_load_iterator<T>([=]__device__(auto i){
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:on a GPU:
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:void cuda_transform_uninitialized_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  //detail::cuda_transform_reduce_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  detail::cuda_uninitialized_reduce_loop(p,
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cuda_make_load_iterator<T>([=]__device__(auto i){ return uop(*(first+i)); }),
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp://__device__ void cuda_warp_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp://__global__ void cuda_reduce(I first, size_t N, T* res, C op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp://  cudaSharedMemory<T> shared_memory;
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp://    cuda_warp_reduce(shm, N, tid, op);
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:// cudaFlowCapturer
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:cudaTask cudaFlowCapturer::reduce(I first, I last, T* result, C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  auto bufsz = cuda_reduce_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  return on([=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cuda_reduce(p, first, last, result, c, buf.get().data());
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:cudaTask cudaFlowCapturer::uninitialized_reduce(I first, I last, T* result, C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  auto bufsz = cuda_reduce_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  return on([=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cuda_uninitialized_reduce(p, first, last, result, c, buf.get().data());
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:cudaTask cudaFlowCapturer::transform_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  auto bufsz = cuda_reduce_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  return on([=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cuda_transform_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:cudaTask cudaFlowCapturer::transform_uninitialized_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  auto bufsz = cuda_reduce_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  return on([=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cuda_transform_uninitialized_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:void cudaFlowCapturer::reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  cudaTask task, I first, I last, T* result, C c
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  auto bufsz = cuda_reduce_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  on(task, [=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cuda_reduce(p, first, last, result, c, buf.get().data());
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:void cudaFlowCapturer::uninitialized_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  cudaTask task, I first, I last, T* result, C c
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  auto bufsz = cuda_reduce_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  on(task, [=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cuda_uninitialized_reduce(p, first, last, result, c, buf.get().data());
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:void cudaFlowCapturer::transform_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  cudaTask task, I first, I last, T* result, C bop, U uop
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  auto bufsz = cuda_reduce_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  on(task, [=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cuda_transform_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:void cudaFlowCapturer::transform_uninitialized_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  cudaTask task, I first, I last, T* result, C bop, U uop
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  auto bufsz = cuda_reduce_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  on(task, [=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cuda_transform_uninitialized_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:// cudaFlow
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:cudaTask cudaFlow::reduce(I first, I last, T* result, B bop) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  return capture([=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:cudaTask cudaFlow::uninitialized_reduce(I first, I last, T* result, B bop) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  return capture([=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:cudaTask cudaFlow::transform_reduce(I first, I last, T* result, B bop, U uop) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  return capture([=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:cudaTask cudaFlow::transform_uninitialized_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  return capture([=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:void cudaFlow::reduce(cudaTask task, I first, I last, T* result, C op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  capture(task, [=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:void cudaFlow::uninitialized_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  cudaTask task, I first, I last, T* result, C op
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  capture(task, [=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:void cudaFlow::transform_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  cudaTask task, I first, I last, T* result, B bop, U uop
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  capture(task, [=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:void cudaFlow::transform_uninitialized_reduce(
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  cudaTask task, I first, I last, T* result, B bop, U uop
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:  capture(task, [=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/reduce.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:#include "../cudaflow.hpp"
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:@file taskflow/cuda/algorithm/merge.hpp
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:@brief CUDA merge algorithm include file
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:enum class cudaMergeBoundType {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:struct cudaMergePair {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cudaArray<T, N> keys;
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cudaArray<unsigned, N> indices;
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:struct cudaMergeRange {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  __device__ cudaRange a_range() const {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    return cudaRange { a_begin, a_end };
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  __device__ cudaRange b_range() const {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    return cudaRange { b_begin, b_end };
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  __device__ cudaMergeRange to_local() const {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    return cudaMergeRange { 0, a_count(), a_count(), total() };
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  __device__ cudaMergeRange partition(unsigned mp0, unsigned diag) const {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    return cudaMergeRange { a_begin + mp0, a_end, b_begin + diag - mp0, b_end };
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  __device__ cudaMergeRange partition(unsigned mp0, unsigned diag0,
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    return cudaMergeRange {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cudaMergeBoundType bounds = cudaMergeBoundType::LOWER,
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:__device__ auto cuda_merge_path(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    bool pred = (cudaMergeBoundType::UPPER == bounds) ?
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:template<cudaMergeBoundType bounds, typename keys_it, typename comp_t>
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:__device__ auto cuda_merge_path(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  keys_it keys, cudaMergeRange range, unsigned diag, comp_t comp
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  return cuda_merge_path<bounds>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:template<cudaMergeBoundType bounds, bool range_check, typename T, typename comp_t>
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:__device__ bool cuda_merge_predicate(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  T a_key, T b_key, cudaMergeRange range, comp_t comp
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    p = (cudaMergeBoundType::UPPER == bounds) ? comp(a_key, b_key) :
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:inline __device__ auto cuda_compute_merge_range(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  return cudaMergeRange { mp0, mp1, diag0 - mp0, diag1 - mp1 };
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:__device__ auto cuda_load_two_streams_reg(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cudaArray<T, vt> x;
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cuda_strided_iterate<nt, vt>([&](auto i, auto index) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cudaArray<T, vt>
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cudaArray<T, vt> x;
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cuda_strided_iterate<nt, vt>([&](auto i, auto index) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:__device__ void cuda_load_two_streams_shared(A a, unsigned a_count,
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  auto x = cuda_load_two_streams_reg<nt, vt, T>(a, a_count, b, b_count, tid);
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cuda_reg_to_shared_strided<nt>(x, tid, shared, sync);
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:__device__ auto cuda_gather_two_streams_strided(const T* a,
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  unsigned a_count, const T* b, unsigned b_count, cudaArray<unsigned, vt> indices,
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cudaArray<T, vt> x;
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cuda_strided_iterate<nt, vt>([&](auto i, auto j) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cudaArray<T, vt>
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:> cuda_gather_two_streams_strided(a_it a,
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  unsigned a_count, b_it b, unsigned b_count, cudaArray<unsigned, vt> indices, unsigned tid) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cudaArray<T, vt> x;
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cuda_strided_iterate<nt, vt>([&](auto i, auto j) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:__device__ void cuda_transfer_two_streams_strided(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cudaArray<unsigned, vt> indices, unsigned tid, c_it c
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  auto x = cuda_gather_two_streams_strided<nt, vt, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cuda_reg_to_mem_strided<nt>(x, tid, a_count + b_count, c);
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:template<cudaMergeBoundType bounds, unsigned vt, typename T, typename comp_t>
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:__device__ auto cuda_serial_merge(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  const T* keys_shared, cudaMergeRange range, comp_t comp, bool sync = true
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cudaMergePair<T, vt> merge_pair;
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cuda_iterate<vt>([&](auto i) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    bool p = cuda_merge_predicate<bounds, true>(a_key, b_key, range, comp);
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:template<cudaMergeBoundType bounds,
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  a_it a, b_it b, cudaMergeRange range_mem, unsigned tid, comp_t comp, T (&keys_shared)[S]
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cuda_load_two_streams_shared<nt, vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  auto mp = cuda_merge_path<bounds>(keys_shared, range_local, diag, comp);
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  auto merged = cuda_serial_merge<bounds, vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:template<cudaMergeBoundType bounds,
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:void cuda_merge_path_partitions(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cuda_kernel<<<B, nt, 0, p.stream()>>>([=]__device__(auto tid, auto bid) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    auto range = cuda_get_tile(bid, nt * vt, num_partitions);
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    cuda_strided_iterate<nt, vt>([=](auto, auto j) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:      buf[index] = cuda_merge_path<bounds>(a, a_count, b, b_count, diag, comp);
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp://  transform([=]MGPU_DEVICE(int index) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:void cuda_merge_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  auto has_values = !std::is_same<V, cudaEmpty>::value;
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cuda_merge_path_partitions<cudaMergeBoundType::LOWER>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    auto range = cuda_compute_merge_range(a_count, b_count, bid, E::nv, mp0, mp1);
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    auto merge = block_merge_from_mem<cudaMergeBoundType::LOWER, E::nt, E::vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    cuda_reg_to_mem_thread<E::nt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:      auto indices = cuda_reg_thread_to_strided<E::nt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:      cuda_transfer_two_streams_strided<E::nt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:tf::cuda_merge.
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:unsigned cuda_merge_buffer_size(unsigned a_count, unsigned b_count) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp://void cuda_merge(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp://  cudaDeviceVector<std::byte> temp(cuda_merge_buffer_size<P>(a_count, b_count));
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp://  detail::cuda_merge_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:void cuda_merge_by_key(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  detail::cuda_merge_loop(p,
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp://void cuda_merge(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp://  cuda_merge(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp://    a_keys_first, (const cudaEmpty*)nullptr, a_keys_last,
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp://    b_keys_first, (const cudaEmpty*)nullptr, b_keys_last,
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp://    c_keys_first, (cudaEmpty*)nullptr, comp
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:This function is equivalent to tf::cuda_merge_by_key without values.
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:void cuda_merge(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cuda_merge_by_key(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    a_keys_first, a_keys_last, (const cudaEmpty*)nullptr,
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    b_keys_first, b_keys_last, (const cudaEmpty*)nullptr,
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    c_keys_first, (cudaEmpty*)nullptr, comp,
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:// cudaFlow merge algorithms
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:cudaTask cudaFlow::merge(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  return capture([=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:void cudaFlow::merge(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cudaTask task, A a_first, A a_last, B b_first, B b_last, C c_first, Comp comp
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  capture(task, [=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:cudaTask cudaFlow::merge_by_key(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  return capture([=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:void cudaFlow::merge_by_key(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cudaTask task,
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  capture(task, [=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:// cudaFlowCapturer merge algorithms
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:cudaTask cudaFlowCapturer::merge(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  auto bufsz = cuda_merge_buffer_size<cudaDefaultExecutionPolicy>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  return on([=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    cuda_merge(cudaDefaultExecutionPolicy{stream},
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:void cudaFlowCapturer::merge(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cudaTask task, A a_first, A a_last, B b_first, B b_last, C c_first, Comp comp
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  auto bufsz = cuda_merge_buffer_size<cudaDefaultExecutionPolicy>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  on(task, [=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    cuda_merge(cudaDefaultExecutionPolicy{stream},
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:cudaTask cudaFlowCapturer::merge_by_key(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  auto bufsz = cuda_merge_buffer_size<cudaDefaultExecutionPolicy>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  return on([=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    cuda_merge_by_key(cudaDefaultExecutionPolicy{stream},
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:void cudaFlowCapturer::merge_by_key(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  cudaTask task,
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  auto bufsz = cuda_merge_buffer_size<cudaDefaultExecutionPolicy>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  on(task, [=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/merge.hpp:    cuda_merge_by_key(cudaDefaultExecutionPolicy{stream},
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:@file taskflow/cuda/algorithm/find.hpp
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:@brief cuda find algorithms include file
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:struct cudaFindPair {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:void cuda_find_if_loop(P&& p, I input, unsigned count, unsigned* idx, U pred) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cuda_single_task(p, [=] __device__ () { *idx = 0; });
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  cuda_single_task(p, [=] __device__ () { *idx = count; });
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    auto tile = cuda_get_tile(bid, E::nv, count);
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    auto x = cuda_mem_to_reg_strided<E::nt, E::vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    //__shared__ typename cudaBlockReduce<E::nt, unsigned>::Storage shm;
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    //id = cudaBlockReduce<E::nt, unsigned>()(
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    //  cuda_minimum<unsigned>{},
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:void cuda_min_element_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cuda_single_task(p, [=] __device__ () { *idx = 0; });
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  using T = cudaFindPair<typename std::iterator_traits<I>::value_type>;
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  cuda_uninitialized_reduce_loop(p,
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cuda_make_load_iterator<T>([=]__device__(auto i){
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:void cuda_max_element_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cuda_single_task(p, [=] __device__ () { *idx = 0; });
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  using T = cudaFindPair<typename std::iterator_traits<I>::value_type>;
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  cuda_uninitialized_reduce_loop(p,
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cuda_make_load_iterator<T>([=]__device__(auto i){
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:// cuda_find_if
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:void cuda_find_if(
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  detail::cuda_find_if_loop(p, first, std::distance(first, last), idx, op);
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:// cudaFlow
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:cudaTask cudaFlow::find_if(I first, I last, unsigned* idx, U op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  return capture([=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:void cudaFlow::find_if(cudaTask task, I first, I last, unsigned* idx, U op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  capture(task, [=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:// cudaFlowCapturer
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:cudaTask cudaFlowCapturer::find_if(I first, I last, unsigned* idx, U op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  return on([=](cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cuda_find_if(p, first, last, idx, op);
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:void cudaFlowCapturer::find_if(
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  cudaTask task, I first, I last, unsigned* idx, U op
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  on(task, [=](cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cuda_find_if(p, first, last, idx, op);
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:// cuda_min_element
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:@brief queries the buffer size in bytes needed to call tf::cuda_min_element
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:tf::cuda_min_element.
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:unsigned cuda_min_element_buffer_size(unsigned count) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  return cuda_reduce_buffer_size<P, detail::cudaFindPair<T>>(count);
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:tf::cuda_min_element_buffer_size bytes for internal use.
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:void cuda_min_element(P&& p, I first, I last, unsigned* idx, O op, void* buf) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  detail::cuda_min_element_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:// cudaFlowCapturer::min_element
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:cudaTask cudaFlowCapturer::min_element(I first, I last, unsigned* idx, O op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  auto bufsz = cuda_min_element_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  return on([=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cuda_min_element(p, first, last, idx, op, buf.get().data());
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:void cudaFlowCapturer::min_element(
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  cudaTask task, I first, I last, unsigned* idx, O op
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  auto bufsz = cuda_min_element_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  on(task, [=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cuda_min_element(p, first, last, idx, op, buf.get().data());
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:// cudaFlow::min_element
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:cudaTask cudaFlow::min_element(I first, I last, unsigned* idx, O op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  return capture([=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:void cudaFlow::min_element(
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  cudaTask task, I first, I last, unsigned* idx, O op
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  capture(task, [=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:// cuda_max_element
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:@brief queries the buffer size in bytes needed to call tf::cuda_max_element
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:tf::cuda_max_element.
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:unsigned cuda_max_element_buffer_size(unsigned count) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  return cuda_reduce_buffer_size<P, detail::cudaFindPair<T>>(count);
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:tf::cuda_max_element_buffer_size bytes for internal use.
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:void cuda_max_element(P&& p, I first, I last, unsigned* idx, O op, void* buf) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  detail::cuda_max_element_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:// cudaFlowCapturer::max_element
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:cudaTask cudaFlowCapturer::max_element(I first, I last, unsigned* idx, O op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  auto bufsz = cuda_max_element_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  return on([=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cuda_max_element(p, first, last, idx, op, buf.get().data());
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:void cudaFlowCapturer::max_element(
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  cudaTask task, I first, I last, unsigned* idx, O op
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  auto bufsz = cuda_max_element_buffer_size<cudaDefaultExecutionPolicy, T>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  on(task, [=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cuda_max_element(p, first, last, idx, op, buf.get().data());
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:// cudaFlow::max_element
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:cudaTask cudaFlow::max_element(I first, I last, unsigned* idx, O op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  return capture([=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:void cudaFlow::max_element(
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  cudaTask task, I first, I last, unsigned* idx, O op
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:  capture(task, [=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/find.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:#include "../cudaflow.hpp"
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:@file taskflow/cuda/algorithm/transform.hpp
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:@brief cuda parallel-transform algorithms include file
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:void cuda_transform_loop(P&& p, I first, unsigned count, O output, C op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:  cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=]__device__(auto tid, auto bid) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:    auto tile = cuda_get_tile(bid, E::nv, count);
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:    cuda_strided_iterate<E::nt, E::vt>([=]__device__(auto, auto j) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:void cuda_transform_loop(
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:  cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=]__device__(auto tid, auto bid) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:    auto tile = cuda_get_tile(bid, E::nv, count);
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:    cuda_strided_iterate<E::nt, E::vt>([=]__device__(auto, auto j) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:// CUDA standard algorithms: transform
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:void cuda_transform(P&& p, I first, I last, O output, C op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:  detail::cuda_transform_loop(p, first, count, output, op);
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:void cuda_transform(
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:  detail::cuda_transform_loop(p, first1, first2, count, output, op);
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:// cudaFlow
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:cudaTask cudaFlow::transform(I first, I last, O output, C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:  return capture([=](cudaFlowCapturer& cap) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:cudaTask cudaFlow::transform(I1 first1, I1 last1, I2 first2, O output, C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:  return capture([=](cudaFlowCapturer& cap) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:void cudaFlow::transform(cudaTask task, I first, I last, O output, C c) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:  capture(task, [=](cudaFlowCapturer& cap) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:void cudaFlow::transform(
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:  cudaTask task, I1 first1, I1 last1, I2 first2, O output, C c
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:  capture(task, [=](cudaFlowCapturer& cap) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:// cudaFlowCapturer
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:cudaTask cudaFlowCapturer::transform(I first, I last, O output, C op) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:  return on([=](cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:    cuda_transform(p, first, last, output, op);
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:cudaTask cudaFlowCapturer::transform(
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:  return on([=](cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:    cuda_transform(p, first1, last1, first2, output, op);
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:void cudaFlowCapturer::transform(
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:  cudaTask task, I first, I last, O output, C op
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:  on(task, [=] (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:    cuda_transform(p, first, last, output, op);
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:void cudaFlowCapturer::transform(
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:  cudaTask task, I1 first1, I1 last1, I2 first2, O output, C op
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:  on(task, [=] (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:    cudaDefaultExecutionPolicy p(stream);
tools/loaders/bam-loader/taskflow/cuda/algorithm/transform.hpp:    cuda_transform(p, first1, last1, first2, output, op);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:@file taskflow/cuda/algorithm/sort.hpp
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:@brief CUDA sort algorithm include file
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:constexpr int cuda_clz(int x) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:constexpr int cuda_find_log2(int x, bool round_up = false) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  int a = 31 - cuda_clz(x);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:__device__ auto cuda_odd_even_sort(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  cudaArray<T, vt> x, C comp, int flags = 0
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  cuda_iterate<vt>([&](auto I) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:        cuda_swap(x[i], x[i + 1]);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:__device__ auto cuda_odd_even_sort(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  cudaKVArray<K, V, vt> x, C comp, int flags = 0
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  cuda_iterate<vt>([&](auto I) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:        cuda_swap(x.keys[i], x.keys[i + 1]);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:        cuda_swap(x.vals[i], x.vals[i + 1]);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:__device__ inline int cuda_out_of_range_flags(int first, int vt, int count) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:__device__ inline auto cuda_compute_merge_sort_frame(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  return cudaMergeRange {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:__device__ inline auto cuda_compute_merge_sort_range(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  auto frame = cuda_compute_merge_sort_frame(partition, coop, spacing);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  return cudaMergeRange {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:__device__ inline auto cuda_compute_merge_sort_range(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  auto range = cuda_compute_merge_sort_range(count, partition, coop, spacing);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:struct cudaBlockSort {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  static constexpr bool has_values = !std::is_same<V, cudaEmpty>::value;
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  static_assert(is_pow2(nt), "cudaBlockSort requires pow2 number of threads");
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    cudaKVArray<K, V, vt> x,
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    auto range = cuda_compute_merge_sort_range(count, tid, coop, vt);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    cuda_reg_to_shared_thread<nt, vt>(x.keys, tid, storage.keys);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    auto mp = cuda_merge_path<cudaMergeBoundType::LOWER>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    auto merge = cuda_serial_merge<cudaMergeBoundType::LOWER, vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:      cuda_reg_to_shared_thread<nt, vt>(x.vals, tid, storage.vals);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:      x.vals = cuda_shared_gather<nt, vt>(storage.vals, merge.indices);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  __device__ auto block_sort(cudaKVArray<K, V, vt> x,
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:      auto head_flags = cuda_out_of_range_flags(vt * tid, vt, count);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:      x = cuda_odd_even_sort(x, comp, head_flags);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:      x = cuda_odd_even_sort(x, comp);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:void cuda_merge_sort_partitions(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  cuda_kernel<<<B, nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    auto range = cuda_get_tile(bid, nt * vt, num_partitions);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    cuda_strided_iterate<nt, vt>([=](auto, auto j) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:      auto range = cuda_compute_merge_sort_range(count, index, coop, spacing);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:      buf[index] = cuda_merge_path<cudaMergeBoundType::LOWER>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  const bool has_values = !std::is_same<V, cudaEmpty>::value;
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  unsigned R = cuda_find_log2(B, true);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  //cudaDeviceVector<K> keys_temp(R ? count : 0);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  //cudaDeviceVector<V> vals_temp((has_values && R) ? count : 0);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=] __device__ (auto tid, auto bid) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    using sort_t = cudaBlockSort<E::nt, E::vt, K, V>;
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    auto tile = cuda_get_tile(bid, E::nv, count);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    cudaKVArray<K, V, E::vt> unsorted;
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    unsorted.keys = cuda_mem_to_reg_thread<E::nt, E::vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:      unsorted.vals = cuda_mem_to_reg_thread<E::nt, E::vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    cuda_reg_to_mem_thread<E::nt, E::vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:      cuda_reg_to_mem_thread<E::nt, E::vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  //cudaDeviceVector<unsigned> mem(num_partitions);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    cuda_merge_sort_partitions(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    cuda_kernel<<<B, E::nt, 0, p.stream()>>>([=]__device__(auto tid, auto bid) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:      auto tile = cuda_get_tile(bid, E::nv, count);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:      auto range = cuda_compute_merge_sort_range(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:      auto merge = block_merge_from_mem<cudaMergeBoundType::LOWER, E::nt, E::vt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:      cuda_reg_to_mem_thread<E::nt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:        auto indices = cuda_reg_thread_to_strided<E::nt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:        cuda_transfer_two_streams_strided<E::nt>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:@tparam V value type (default tf::cudaEmpty)
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:The function is used to allocate a buffer for calling tf::cuda_sort.
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:template <typename P, typename K, typename V = cudaEmpty>
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:unsigned cuda_sort_buffer_size(unsigned count) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  const bool has_values = !std::is_same<V, cudaEmpty>::value;
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  unsigned R = detail::cuda_find_log2(B, true);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:void cuda_sort_by_key(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:This method is equivalent to tf::cuda_sort_by_key without values.
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:void cuda_sort(P&& p, K_it k_first, K_it k_last, C comp, void* buf) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  cuda_sort_by_key(p, k_first, k_last, (cudaEmpty*)nullptr, comp, buf);
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:// cudaFlow
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:cudaTask cudaFlow::sort(I first, I last, C comp) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  return capture([=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:void cudaFlow::sort(cudaTask task, I first, I last, C comp) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  capture(task, [=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:cudaTask cudaFlow::sort_by_key(K_it k_first, K_it k_last, V_it v_first, C comp) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  return capture([=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:void cudaFlow::sort_by_key(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  cudaTask task, K_it k_first, K_it k_last, V_it v_first, C comp
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  capture(task, [=](cudaFlowCapturer& cap){
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    cap.make_optimizer<cudaLinearCapturing>();
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:// cudaFlowCapturer
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:cudaTask cudaFlowCapturer::sort(I first, I last, C comp) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  auto bufsz = cuda_sort_buffer_size<cudaDefaultExecutionPolicy, K>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  return on([=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    cuda_sort(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:      cudaDefaultExecutionPolicy{stream}, first, last, comp, buf.get().data()
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:void cudaFlowCapturer::sort(cudaTask task, I first, I last, C comp) {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  auto bufsz = cuda_sort_buffer_size<cudaDefaultExecutionPolicy, K>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  on(task, [=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    cuda_sort(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:      cudaDefaultExecutionPolicy{stream}, first, last, comp, buf.get().data()
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:cudaTask cudaFlowCapturer::sort_by_key(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  auto bufsz = cuda_sort_buffer_size<cudaDefaultExecutionPolicy, K, V>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  return on([=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    cuda_sort_by_key(cudaDefaultExecutionPolicy{stream},
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:void cudaFlowCapturer::sort_by_key(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  cudaTask task, K_it k_first, K_it k_last, V_it v_first, C comp
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  auto bufsz = cuda_sort_buffer_size<cudaDefaultExecutionPolicy, K, V>(
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  on(task, [=, buf=MoC{cudaDeviceVector<std::byte>(bufsz)}] 
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:  (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/algorithm/sort.hpp:    cuda_sort_by_key(cudaDefaultExecutionPolicy{stream},
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:#include "cuda_device.hpp"
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:@file cuda_memory.hpp
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:@brief CUDA memory utilities include file
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:inline size_t cuda_get_free_mem(int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  cudaScopedDevice ctx(d);
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    cudaMemGetInfo(&free, &total), "failed to get mem info on device ", d
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:inline size_t cuda_get_total_mem(int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  cudaScopedDevice ctx(d);
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    cudaMemGetInfo(&free, &total), "failed to get mem info on device ", d
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:The function calls @c cudaMalloc to allocate <tt>N*sizeof(T)</tt> bytes of memory
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:T* cuda_malloc_device(size_t N, int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  cudaScopedDevice ctx(d);
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    cudaMalloc(&ptr, N*sizeof(T)),
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:T* cuda_malloc_device(size_t N) {
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    cudaMalloc(&ptr, N*sizeof(T)), 
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:The function calls @c cudaMallocManaged to allocate <tt>N*sizeof(T)</tt> bytes
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:T* cuda_malloc_shared(size_t N) {
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    cudaMallocManaged(&ptr, N*sizeof(T)),
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:@brief frees memory on the GPU device
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:This methods call @c cudaFree to free the memory space pointed to by @c ptr
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:void cuda_free(T* ptr, int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  cudaScopedDevice ctx(d);
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(cudaFree(ptr), "failed to free memory ", ptr, " on GPU ", d);
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:@brief frees memory on the GPU device
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:This methods call @c cudaFree to free the memory space pointed to by @c ptr
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:void cuda_free(T* ptr) {
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(cudaFree(ptr), "failed to free memory ", ptr);
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:The method calls @c cudaMemcpyAsync with the given @c stream
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:using @c cudaMemcpyDefault to infer the memory space of the source and
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:inline void cuda_memcpy_async(
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  cudaStream_t stream, void* dst, const void* src, size_t count
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream),
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    "failed to perform cudaMemcpyAsync"
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:@brief initializes or sets GPU memory to the given value byte by byte
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:@param devPtr pointer to GPU mempry
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:The method calls @c cudaMemsetAsync with the given @c stream
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:inline void cuda_memset_async(
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  cudaStream_t stream, void* devPtr, int value, size_t count
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    cudaMemsetAsync(devPtr, value, count, stream),
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    "failed to perform cudaMemsetAsync"
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp://      cudaSharedMemory<T> smem;
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <int>
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <unsigned int>
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <char>
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <unsigned char>
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <short>
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <unsigned short>
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <long>
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <unsigned long>
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp://struct cudaSharedMemory <size_t>
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <bool>
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <float>
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:struct cudaSharedMemory <double>
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:// cudaDeviceAllocator
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:@class cudaDeviceAllocator
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:@brief class to create a CUDA device allocator 
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:A %cudaDeviceAllocator enables device-specific allocation for 
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:class cudaDeviceAllocator {
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    using other = cudaDeviceAllocator<U>; 
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  cudaDeviceAllocator() noexcept {}
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  cudaDeviceAllocator( const cudaDeviceAllocator& ) noexcept {}
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  cudaDeviceAllocator( const cudaDeviceAllocator<U>& ) noexcept {}
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  ~cudaDeviceAllocator() noexcept {}
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  The block of storage is allocated using cudaMalloc and throws std::bad_alloc 
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:      cudaMalloc( &ptr, n*sizeof(T) ),
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:      cudaFree(ptr);
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  bool operator == (const cudaDeviceAllocator<U>&) const noexcept {
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  bool operator != (const cudaDeviceAllocator<U>&) const noexcept {
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:// cudaUSMAllocator
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:@class cudaUSMAllocator
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:A %cudaUSMAllocator enables using unified shared memory (USM) allocation for 
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:class cudaUSMAllocator {
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    using other = cudaUSMAllocator<U>; 
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  cudaUSMAllocator() noexcept {}
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  cudaUSMAllocator( const cudaUSMAllocator& ) noexcept {}
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  cudaUSMAllocator( const cudaUSMAllocator<U>& ) noexcept {}
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  ~cudaUSMAllocator() noexcept {}
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  The block of storage is allocated using cudaMalloc and throws std::bad_alloc 
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:      cudaMallocManaged( &ptr, n*sizeof(T) ),
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:      cudaFree(ptr);
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  bool operator == (const cudaUSMAllocator<U>&) const noexcept {
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:  bool operator != (const cudaUSMAllocator<U>&) const noexcept {
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:// GPU vector object
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp://using cudaDeviceVector = std::vector<NoInit<T>, cudaDeviceAllocator<NoInit<T>>>;
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp://using cudaUSMVector = std::vector<T, cudaUSMAllocator<T>>;
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:class cudaDeviceVector {
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    cudaDeviceVector() = default;
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    cudaDeviceVector(size_t N) : _N {N} {
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:        TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:          cudaMalloc(&_data, N*sizeof(T)),
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    cudaDeviceVector(cudaDeviceVector&& rhs) : 
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    ~cudaDeviceVector() {
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:        cudaFree(_data);
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    cudaDeviceVector& operator = (cudaDeviceVector&& rhs) {
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:        cudaFree(_data);
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    cudaDeviceVector(const cudaDeviceVector&) = delete;
tools/loaders/bam-loader/taskflow/cuda/cuda_memory.hpp:    cudaDeviceVector& operator = (const cudaDeviceVector&) = delete;
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:#include "cuda_error.hpp"
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:@brief per-thread object pool to manage CUDA device object
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:A CUDA device object has a lifetime associated with a device,
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:for example, @c cudaStream_t, @c cublasHandle_t, etc.
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:There exists an one-to-one relationship between CUDA devices in CUDA Runtime API
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:and CUcontexts in the CUDA Driver API within a process.
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:The specific context which the CUDA Runtime API uses for a device
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:From the perspective of the CUDA Runtime API,
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:class cudaPerThreadDeviceObjectPool {
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:  // Due to some ordering, cuda context may be destroyed when the master
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:  // program thread destroys the cuda object.
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:  // destroy cuda objects while the master thread only keeps a weak reference
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:  struct cudaGlobalDeviceObjectPool {
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:    cudaPerThreadDeviceObjectPool() = default;
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:    inline static cudaGlobalDeviceObjectPool _shared_pool;
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:// cudaPerThreadDeviceObject::cudaHanale definition
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:cudaPerThreadDeviceObjectPool<H, C, D>::Object::Object(int d) :
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:  cudaScopedDevice ctx(device);
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:cudaPerThreadDeviceObjectPool<H, C, D>::Object::~Object() {
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:  cudaScopedDevice ctx(device);
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:// cudaPerThreadDeviceObject::cudaHanaldePool definition
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:std::shared_ptr<typename cudaPerThreadDeviceObjectPool<H, C, D>::Object>
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:cudaPerThreadDeviceObjectPool<H, C, D>::cudaGlobalDeviceObjectPool::acquire(int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:void cudaPerThreadDeviceObjectPool<H, C, D>::cudaGlobalDeviceObjectPool::release(
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:// cudaPerThreadDeviceObject definition
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:std::shared_ptr<typename cudaPerThreadDeviceObjectPool<H, C, D>::Object>
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:cudaPerThreadDeviceObjectPool<H, C, D>::acquire(int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:void cudaPerThreadDeviceObjectPool<H, C, D>::release(
tools/loaders/bam-loader/taskflow/cuda/cuda_pool.hpp:size_t cudaPerThreadDeviceObjectPool<H, C, D>::footprint_size() const {
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:#include "cuda_graph.hpp"
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:@file cuda_optimizer.hpp
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:@brief %cudaFlow capturing algorithms include file
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:// cudaCapturingBase
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:class cudaCapturingBase {
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:    std::vector<cudaNode*> _toposort(cudaGraph&);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:    std::vector<std::vector<cudaNode*>> _levelize(cudaGraph&);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:inline std::vector<cudaNode*> cudaCapturingBase::_toposort(cudaGraph& graph) {
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  std::vector<cudaNode*> res;
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  std::queue<cudaNode*> bfs;
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:    auto hu = std::get_if<cudaNode::Capture>(&u->_handle);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:      auto hv = std::get_if<cudaNode::Capture>(&v->_handle);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:inline std::vector<std::vector<cudaNode*>>
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:cudaCapturingBase::_levelize(cudaGraph& graph) {
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  std::queue<cudaNode*> bfs;
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:    auto hu = std::get_if<cudaNode::Capture>(&u->_handle);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:    auto hu = std::get_if<cudaNode::Capture>(&u->_handle);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:      auto hv = std::get_if<cudaNode::Capture>(&v->_handle);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  std::vector<std::vector<cudaNode*>> level_graph(max_level+1);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:    auto hu = std::get_if<cudaNode::Capture>(&u->_handle);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:    //  assert(hu.level < std::get_if<cudaNode::Capture>(&s->_handle)->level);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:// class definition: cudaSequentialCapturing
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:@class cudaSequentialCapturing
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:@brief class to capture a CUDA graph using a sequential stream
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:the described graph and captures dependent GPU tasks using a single stream.
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:All GPU tasks run sequentially without breaking inter dependencies.
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:class cudaSequentialCapturing : public cudaCapturingBase {
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  friend class cudaFlowCapturer;
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:    cudaSequentialCapturing() = default;
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:    cudaGraph_t _optimize(cudaGraph& graph);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:inline cudaGraph_t cudaSequentialCapturing::_optimize(cudaGraph& graph) {
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  // we must use ThreadLocal mode to avoid clashing with CUDA global states
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  cudaStream stream;
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  stream.begin_capture(cudaStreamCaptureModeThreadLocal);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:    std::get_if<cudaNode::Capture>(&node->_handle)->work(stream);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:// class definition: cudaLinearCapturing
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:@class cudaLinearCapturing
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:@brief class to capture a linear CUDA graph using a sequential stream
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:A linear capturing algorithm is a special case of tf::cudaSequentialCapturing
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:class cudaLinearCapturing : public cudaCapturingBase {
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  friend class cudaFlowCapturer;
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:    cudaLinearCapturing() = default;
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:    cudaGraph_t _optimize(cudaGraph& graph);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:inline cudaGraph_t cudaLinearCapturing::_optimize(cudaGraph& graph) {
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  // we must use ThreadLocal mode to avoid clashing with CUDA global states
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  cudaStream stream;
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  stream.begin_capture(cudaStreamCaptureModeThreadLocal);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  cudaNode* src {nullptr};
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:        std::get_if<cudaNode::Capture>(&src->_handle)->work(stream);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:// class definition: cudaRoundRobinCapturing
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:@class cudaRoundRobinCapturing
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:@brief class to capture a CUDA graph using a round-robin algorithm
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  + Dian-Lun Lin and Tsung-Wei Huang, &quot;Efficient GPU Computation using %Task Graph Parallelism,&quot; <i>European Conference on Parallel and Distributed Computing (Euro-Par)</i>, 2021
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:The round-robin optimization algorithm is best suited for large %cudaFlow graphs
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:that compose hundreds of or thousands of GPU operations
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:maximum kernel currency in the captured CUDA graph.
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:class cudaRoundRobinCapturing : public cudaCapturingBase {
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  friend class cudaFlowCapturer;
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:    cudaRoundRobinCapturing() = default;
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:    explicit cudaRoundRobinCapturing(size_t num_streams);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:    cudaGraph_t _optimize(cudaGraph& graph);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:    void _reset(std::vector<std::vector<cudaNode*>>& graph);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:inline cudaRoundRobinCapturing::cudaRoundRobinCapturing(size_t num_streams) :
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:inline size_t cudaRoundRobinCapturing::num_streams() const {
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:inline void cudaRoundRobinCapturing::num_streams(size_t n) {
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:inline void cudaRoundRobinCapturing::_reset(
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  std::vector<std::vector<cudaNode*>>& graph
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:      auto hn = std::get_if<cudaNode::Capture>(&node->_handle);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:inline cudaGraph_t cudaRoundRobinCapturing::_optimize(cudaGraph& graph) {
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  std::vector<cudaStream> streams(_num_streams);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  streams[0].begin_capture(cudaStreamCaptureModeThreadLocal);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  std::vector<cudaEvent> events;
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:  cudaEvent_t fork_event = events.emplace_back();
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:      auto hn = std::get_if<cudaNode::Capture>(&node->_handle);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:      cudaNode* wait_node{nullptr};
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:        auto phn = std::get_if<cudaNode::Capture>(&pn->_handle);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:             std::get_if<cudaNode::Capture>(&wait_node->_handle)->level < phn->level) {
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:        assert(std::get_if<cudaNode::Capture>(&wait_node->_handle)->event); 
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:        streams[sid].wait(std::get_if<cudaNode::Capture>(&wait_node->_handle)->event);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:        auto shn = std::get_if<cudaNode::Capture>(&sn->_handle);
tools/loaders/bam-loader/taskflow/cuda/cuda_optimizer.hpp:    cudaEvent_t join_event = events.emplace_back();
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:#include "cuda_error.hpp"
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:@file cuda_device.hpp
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:@brief CUDA device utilities include file
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_num_devices() {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(cudaGetDeviceCount(&N), "failed to get device count");
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline int cuda_get_device() {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(cudaGetDevice(&id), "failed to get current device id");
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline void cuda_set_device(int id) {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(cudaSetDevice(id), "failed to switch to device ", id);
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline void cuda_get_device_property(int i, cudaDeviceProp& p) {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaGetDeviceProperties(&p, i), "failed to get property of device ", i
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline cudaDeviceProp cuda_get_device_property(int i) {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  cudaDeviceProp p;
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaGetDeviceProperties(&p, i), "failed to get property of device ", i
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline void cuda_dump_device_property(std::ostream& os, const cudaDeviceProp& p) {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:     << "GPU sharing Host Memory:       " << p.integrated << '\n'
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_threads_per_block(int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerBlock, d),
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_x_dim_per_block(int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxBlockDimX, d),
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_y_dim_per_block(int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxBlockDimY, d),
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_z_dim_per_block(int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxBlockDimZ, d),
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_x_dim_per_grid(int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxGridDimX, d),
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_y_dim_per_grid(int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxGridDimY, d),
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_z_dim_per_grid(int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&dim, cudaDevAttrMaxGridDimZ, d),
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_max_shm_per_block(int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&num, cudaDevAttrMaxSharedMemoryPerBlock, d),
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline size_t cuda_get_device_warp_size(int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&num, cudaDevAttrWarpSize, d),
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline int cuda_get_device_compute_capability_major(int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&num, cudaDevAttrComputeCapabilityMajor, d),
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline int cuda_get_device_compute_capability_minor(int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&num, cudaDevAttrComputeCapabilityMinor, d),
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline bool cuda_get_device_unified_addressing(int d) {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaDeviceGetAttribute(&num, cudaDevAttrUnifiedAddressing, d),
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:// CUDA Version
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:@brief queries the latest CUDA version (1000 * major + 10 * minor) supported by the driver
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline int cuda_get_driver_version() {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaDriverGetVersion(&num),
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    "failed to query the latest cuda version supported by the driver"
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:@brief queries the CUDA Runtime version (1000 * major + 10 * minor)
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline int cuda_get_runtime_version() {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaRuntimeGetVersion(&num), "failed to query cuda runtime version"
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:// cudaScopedDevice
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:/** @class cudaScopedDevice
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  tf::cudaScopedDevice device(1);  // switch to the device context 1
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  cudaStream_t stream;
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  cudaStreamCreate(&stream);
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:%cudaScopedDevice is neither movable nor copyable.
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:class cudaScopedDevice {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    explicit cudaScopedDevice(int device);
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    ~cudaScopedDevice();
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaScopedDevice() = delete;
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaScopedDevice(const cudaScopedDevice&) = delete;
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaScopedDevice(cudaScopedDevice&&) = delete;
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline cudaScopedDevice::cudaScopedDevice(int dev) {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:  TF_CHECK_CUDA(cudaGetDevice(&_p), "failed to get current device scope");
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    TF_CHECK_CUDA(cudaSetDevice(dev), "failed to scope on device ", dev);
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:inline cudaScopedDevice::~cudaScopedDevice() {
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    cudaSetDevice(_p);
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:    //TF_CHECK_CUDA(cudaSetDevice(_p), "failed to scope back to device ", _p);
tools/loaders/bam-loader/taskflow/cuda/cuda_device.hpp:}  // end of namespace cuda ---------------------------------------------------
tools/loaders/bam-loader/taskflow/cuda/cuda_execution_policy.hpp:#include "cuda_error.hpp"
tools/loaders/bam-loader/taskflow/cuda/cuda_execution_policy.hpp:@file cuda_execution_policy.hpp
tools/loaders/bam-loader/taskflow/cuda/cuda_execution_policy.hpp:@brief CUDA execution policy include file
tools/loaders/bam-loader/taskflow/cuda/cuda_execution_policy.hpp:@class cudaExecutionPolicy
tools/loaders/bam-loader/taskflow/cuda/cuda_execution_policy.hpp:@brief class to define execution policy for CUDA standard algorithms
tools/loaders/bam-loader/taskflow/cuda/cuda_execution_policy.hpp:Execution policy configures the kernel execution parameters in CUDA algorithms.
tools/loaders/bam-loader/taskflow/cuda/cuda_execution_policy.hpp:Details can be referred to @ref CUDASTDExecutionPolicy.
tools/loaders/bam-loader/taskflow/cuda/cuda_execution_policy.hpp:class cudaExecutionPolicy {
tools/loaders/bam-loader/taskflow/cuda/cuda_execution_policy.hpp:  cudaExecutionPolicy() = default;
tools/loaders/bam-loader/taskflow/cuda/cuda_execution_policy.hpp:  explicit cudaExecutionPolicy(cudaStream_t s) : _stream{s} {}
tools/loaders/bam-loader/taskflow/cuda/cuda_execution_policy.hpp:  cudaStream_t stream() noexcept { return _stream; };
tools/loaders/bam-loader/taskflow/cuda/cuda_execution_policy.hpp:  void stream(cudaStream_t stream) noexcept { _stream = stream; }
tools/loaders/bam-loader/taskflow/cuda/cuda_execution_policy.hpp:  cudaStream_t _stream {0};
tools/loaders/bam-loader/taskflow/cuda/cuda_execution_policy.hpp:using cudaDefaultExecutionPolicy = cudaExecutionPolicy<512, 9>;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:#include "cuda_memory.hpp"
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:#include "cuda_stream.hpp"
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:#include "cuda_meta.hpp"
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:// cudaGraph_t routines
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:cudaMemcpy3DParms cuda_get_copy_parms(T* tgt, const T* src, size_t num) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  cudaMemcpy3DParms p;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  p.srcPos = ::make_cudaPos(0, 0, 0);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  p.srcPtr = ::make_cudaPitchedPtr(const_cast<T*>(src), num*sizeof(U), num, 1);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  p.dstPos = ::make_cudaPos(0, 0, 0);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  p.dstPtr = ::make_cudaPitchedPtr(tgt, num*sizeof(U), num, 1);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  p.extent = ::make_cudaExtent(num*sizeof(U), 1, 1);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  p.kind = cudaMemcpyDefault;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline cudaMemcpy3DParms cuda_get_memcpy_parms(
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  // Parameters in cudaPitchedPtr
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  cudaMemcpy3DParms p;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  p.srcPos = ::make_cudaPos(0, 0, 0);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  p.srcPtr = ::make_cudaPitchedPtr(const_cast<void*>(src), bytes, bytes, 1);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  p.dstPos = ::make_cudaPos(0, 0, 0);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  p.dstPtr = ::make_cudaPitchedPtr(tgt, bytes, bytes, 1);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  p.extent = ::make_cudaExtent(bytes, 1, 1);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  p.kind = cudaMemcpyDefault;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline cudaMemsetParams cuda_get_memset_parms(void* dst, int ch, size_t count) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  cudaMemsetParams p;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:cudaMemsetParams cuda_get_fill_parms(T* dst, T value, size_t count) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  cudaMemsetParams p;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:cudaMemsetParams cuda_get_zero_parms(T* dst, size_t count) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  cudaMemsetParams p;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:@brief queries the number of root nodes in a native CUDA graph
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline size_t cuda_get_graph_num_root_nodes(cudaGraph_t graph) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetRootNodes(graph, nullptr, &num_nodes),
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:@brief queries the number of nodes in a native CUDA graph
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline size_t cuda_get_graph_num_nodes(cudaGraph_t graph) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetNodes(graph, nullptr, &num_nodes),
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:@brief queries the number of edges in a native CUDA graph
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline size_t cuda_get_graph_num_edges(cudaGraph_t graph) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetEdges(graph, nullptr, nullptr, &num_edges),
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:@brief acquires the nodes in a native CUDA graph
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline std::vector<cudaGraphNode_t> cuda_get_graph_nodes(cudaGraph_t graph) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  size_t num_nodes = cuda_get_graph_num_nodes(graph);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  std::vector<cudaGraphNode_t> nodes(num_nodes);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetNodes(graph, nodes.data(), &num_nodes),
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:@brief acquires the root nodes in a native CUDA graph
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline std::vector<cudaGraphNode_t> cuda_get_graph_root_nodes(cudaGraph_t graph) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  size_t num_nodes = cuda_get_graph_num_root_nodes(graph);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  std::vector<cudaGraphNode_t> nodes(num_nodes);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetRootNodes(graph, nodes.data(), &num_nodes),
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:@brief acquires the edges in a native CUDA graph
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline std::vector<std::pair<cudaGraphNode_t, cudaGraphNode_t>>
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:cuda_get_graph_edges(cudaGraph_t graph) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  size_t num_edges = cuda_get_graph_num_edges(graph);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  std::vector<cudaGraphNode_t> froms(num_edges), tos(num_edges);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaGraphGetEdges(graph, froms.data(), tos.data(), &num_edges),
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  std::vector<std::pair<cudaGraphNode_t, cudaGraphNode_t>> edges(num_edges);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:@brief queries the type of a native CUDA graph node
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeKernel      = 0x00
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeMemcpy      = 0x01
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeMemset      = 0x02
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeHost        = 0x03
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeGraph       = 0x04
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeEmpty       = 0x05
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeWaitEvent   = 0x06
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  + cudaGraphNodeTypeEventRecord = 0x07
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline cudaGraphNodeType cuda_get_graph_node_type(cudaGraphNode_t node) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  cudaGraphNodeType type;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaGraphNodeGetType(node, &type), "failed to get native graph node type"
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:@brief convert the type of a native CUDA graph node to a readable string
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline const char* cuda_graph_node_type_to_string(cudaGraphNodeType type) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeKernel      : return "kernel";
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeMemcpy      : return "memcpy";
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeMemset      : return "memset";
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeHost        : return "host";
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeGraph       : return "graph";
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeEmpty       : return "empty";
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeWaitEvent   : return "event_wait";
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    case cudaGraphNodeTypeEventRecord : return "event_record";
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:@brief dumps a native CUDA graph and all associated child graphs to a DOT format
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:@param graph native CUDA graph
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:void cuda_dump_graph(T& os, cudaGraph_t graph) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  os << "digraph cudaGraph {\n";
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  std::stack<std::tuple<cudaGraph_t, cudaGraphNode_t, int>> stack;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:       << "label=\"cudaGraph-L" << l << "\";\n"
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    auto nodes = cuda_get_graph_nodes(graph);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    auto edges = cuda_get_graph_edges(graph);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:      auto type = cuda_get_graph_node_type(node);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:      if(type == cudaGraphNodeTypeGraph) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:        cudaGraph_t graph;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:        TF_CHECK_CUDA(cudaGraphChildGraphNodeGetGraph(node, &graph), "");
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:           << "label=\"cudaGraph-L" << l+1
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:           << cuda_graph_node_type_to_string(type)
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:      std::unordered_set<cudaGraphNode_t> successors;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:// cudaGraph class
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:// class: cudaGraph
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:class cudaGraph : public CustomGraphBase {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaNode;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaTask;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowCapturerBase;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowCapturer;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlow;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaCapturingBase;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaSequentialCapturing;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaLinearCapturing;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaRoundRobinCapturing;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaGraph() = default;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    ~cudaGraph();
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaGraph(const cudaGraph&) = delete;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaGraph(cudaGraph&&);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaGraph& operator = (const cudaGraph&) = delete;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaGraph& operator = (cudaGraph&&);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaNode* emplace_back(ArgsT&&...);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaGraph_t _native_handle {nullptr};
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    std::vector<std::unique_ptr<cudaNode>> _nodes;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    //std::vector<cudaNode*> _nodes;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:// cudaNode class
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:@class: cudaNode
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:class cudaNode {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaGraph;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaTask;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlow;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowCapturer;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaFlowCapturerBase;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaCapturingBase;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaSequentialCapturing;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaLinearCapturing;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  friend class cudaRoundRobinCapturing;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaGraph graph;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    std::function<void(cudaStream_t)> work;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaEvent_t event;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaNode() = delete;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaNode(cudaGraph&, ArgsT&&...);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaGraph& _graph;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    cudaGraphNode_t _native_handle {nullptr};
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    SmallVector<cudaNode*> _successors;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    SmallVector<cudaNode*> _dependents;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    void _precede(cudaNode*);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:// cudaNode definitions
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:cudaNode::Host::Host(C&& c) : func {std::forward<C>(c)} {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline void cudaNode::Host::callback(void* data) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:cudaNode::Kernel::Kernel(F&& f) :
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:cudaNode::Capture::Capture(C&& work) :
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:cudaNode::cudaNode(cudaGraph& graph, ArgsT&&... args) :
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline void cudaNode::_precede(cudaNode* v) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  _graph._state |= cudaGraph::CHANGED;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  if(_handle.index() != cudaNode::CAPTURE) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:    TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:      cudaGraphAddDependencies(
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp://inline void cudaNode::_set_state(int flag) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp://inline void cudaNode::_unset_state(int flag) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp://inline void cudaNode::_clear_state() {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp://inline bool cudaNode::_has_state(int flag) const {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:// cudaGraph definitions
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline cudaGraph::~cudaGraph() {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline cudaGraph::cudaGraph(cudaGraph&& g) :
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline cudaGraph& cudaGraph::operator = (cudaGraph&& rhs) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline bool cudaGraph::empty() const {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline void cudaGraph::clear() {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  _state = cudaGraph::CHANGED;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:cudaNode* cudaGraph::emplace_back(ArgsT&&... args) {
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  _state |= cudaGraph::CHANGED;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  auto node = std::make_unique<cudaNode>(std::forward<ArgsT>(args)...);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  //auto node = new cudaNode(std::forward<ArgsT>(args)...);
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:inline void cudaGraph::dump(
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:  std::stack<std::tuple<const cudaGraph*, const cudaNode*, int>> stack;
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:        os << "subgraph cluster_p" << root << " {\nlabel=\"cudaFlow: ";
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:        os << "digraph cudaFlow {\n";
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:      os << "subgraph cluster_p" << parent << " {\nlabel=\"cudaSubflow: ";
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:        case cudaNode::KERNEL:
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:        case cudaNode::SUBFLOW:
tools/loaders/bam-loader/taskflow/cuda/cuda_graph.hpp:            &(std::get_if<cudaNode::Subflow>(&v->_handle)->graph), v, l+1)
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:#include "cuda_pool.hpp"
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:@file cuda_stream.hpp
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:@brief CUDA stream utilities include file
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:// cudaStreamCreator and cudaStreamDeleter for per-thread stream pool
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:struct cudaStreamCreator {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:  @brief operator to create a CUDA stream
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:  cudaStream_t operator () () const {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    cudaStream_t stream;
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    TF_CHECK_CUDA(cudaStreamCreate(&stream), "failed to create a CUDA stream");
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:struct cudaStreamDeleter {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:  @brief operator to destroy a CUDA stream
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:  void operator () (cudaStream_t stream) const {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      cudaStreamDestroy(stream);
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:struct cudaStreamSynchronizer {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:  void operator () (cudaStream_t stream) const {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      cudaStreamSynchronize(stream), "failed to synchronize a CUDA stream"
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:// cudaStream
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:@class cudaStream
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:@brief class to create a CUDA stream in an RAII-styled wrapper
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:A cudaStream object is an RAII-styled wrapper over a native CUDA stream
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:(@c cudaStream_t).
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:A cudaStream object is move-only.
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:class cudaStream {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    @brief constructs an RAII-styled object from the given CUDA stream
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    Constructs a cudaStream object which owns @c stream.
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    explicit cudaStream(cudaStream_t stream) : _stream(stream) {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    @brief constructs an RAII-styled object for a new CUDA stream
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    Equivalently calling @c cudaStreamCreate to create a stream.
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    cudaStream() : _stream{ cudaStreamCreator{}() } {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    cudaStream(const cudaStream&) = delete;
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    cudaStream(cudaStream&& rhs) : _stream{rhs._stream} {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    @brief destructs the CUDA stream
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    ~cudaStream() {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      cudaStreamDeleter {} (_stream);
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    cudaStream& operator = (const cudaStream&) = delete;
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    cudaStream& operator = (cudaStream&& rhs) {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      cudaStreamDeleter {} (_stream);
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    void reset(cudaStream_t stream = nullptr) {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      cudaStreamDeleter {} (_stream);
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    @brief implicit conversion to the native CUDA stream (cudaStream_t)
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    Returns the underlying stream of type @c cudaStream_t.
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    operator cudaStream_t () const {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    Equivalently calling @c cudaStreamSynchronize to block 
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      cudaStreamSynchronizer{}(_stream);
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    which will be returned via cudaStream::end_capture. 
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    + @c cudaStreamCaptureModeGlobal: This is the default mode. 
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      with @c cudaStreamCaptureModeRelaxed at @c cuStreamBeginCapture, 
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      @c cudaStreamCaptureModeGlobal, this thread is prohibited from potentially 
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    + @c cudaStreamCaptureModeThreadLocal: If the local thread has an ongoing capture 
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      sequence not initiated with @c cudaStreamCaptureModeRelaxed, 
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    + @c cudaStreamCaptureModeRelaxed: The local thread is not prohibited 
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      attempting @c cudaEventQuery on an event that was last recorded 
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    void begin_capture(cudaStreamCaptureMode m = cudaStreamCaptureModeGlobal) const {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:        cudaStreamBeginCapture(_stream, m), 
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    Equivalently calling @c cudaStreamEndCapture to
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    Capture must have been initiated on stream via a call to cudaStream::begin_capture. 
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    cudaGraph_t end_capture() const {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      cudaGraph_t native_g;
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:        cudaStreamEndCapture(_stream, &native_g), 
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    Equivalently calling @c cudaEventRecord to record an event on this stream,
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    both of which must be on the same CUDA context.
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    void record(cudaEvent_t event) const {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:        cudaEventRecord(event, _stream), 
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    Equivalently calling @c cudaStreamWaitEvent to make all future work 
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    void wait(cudaEvent_t event) const {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:        cudaStreamWaitEvent(_stream, event, 0), 
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    cudaStream_t _stream {nullptr};
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:// cudaEventCreator and cudaEventDeleter for per-thread event pool
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:struct cudaEventCreator {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:  @brief operator to create a CUDA event
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:  cudaEvent_t operator () () const {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    cudaEvent_t event;
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    TF_CHECK_CUDA(cudaEventCreate(&event), "failed to create a CUDA event");
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:struct cudaEventDeleter {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:  @brief operator to destroy a CUDA event
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:  void operator () (cudaEvent_t event) const {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    cudaEventDestroy(event);
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:// cudaEvent
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:@class cudaEvent
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:@brief class to create a CUDA event in an RAII-styled wrapper
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:A cudaEvent object is an RAII-styled wrapper over a native CUDA stream
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:(@c cudaEvent_t).
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:A cudaEvent object is move-only.
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:class cudaEvent {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    @brief constructs an RAII-styled object from the given CUDA event
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    explicit cudaEvent(cudaEvent_t event) : _event(event) {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    @brief constructs an RAII-styled object for a new CUDA event
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    cudaEvent() : _event{ cudaEventCreator{}() } {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    cudaEvent(const cudaEvent&) = delete;
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    cudaEvent(cudaEvent&& rhs) : _event{rhs._event} {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    @brief destructs the CUDA event
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    ~cudaEvent() {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      cudaEventDeleter {} (_event);
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    cudaEvent& operator = (const cudaEvent&) = delete;
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    cudaEvent& operator = (cudaEvent&& rhs) {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      cudaEventDeleter {} (_event);
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    @brief implicit conversion to the native CUDA event (cudaEvent_t)
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    Returns the underlying event of type @c cudaEvent_t.
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    operator cudaEvent_t () const {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    void reset(cudaEvent_t event = nullptr) {
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:      cudaEventDeleter {} (_event);
tools/loaders/bam-loader/taskflow/cuda/cuda_stream.hpp:    cudaEvent_t _event {nullptr};
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:#include "cuda_task.hpp"
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:#include "cuda_capturer.hpp"
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:@file taskflow/cuda/cudaflow.hpp
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:@brief cudaFlow include file
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:// class definition: cudaFlow
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:@class cudaFlow
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:@brief class to create a %cudaFlow task dependency graph
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:A %cudaFlow is a high-level interface over CUDA Graph to perform GPU operations
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:on one or multiple CUDA devices,
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:The following example creates a %cudaFlow of two kernel tasks, @c task1 and
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:taskflow.emplace([&](tf::cudaFlow& cf){
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  tf::cudaTask task1 = cf.kernel(grid1, block1, shm_size1, kernel1, args1);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  tf::cudaTask task2 = cf.kernel(grid2, block2, shm_size2, kernel2, args2);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:A %cudaFlow is a task (tf::Task) created from tf::Taskflow
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:That is, the callable that describes a %cudaFlow
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:Inside a %cudaFlow task, different GPU tasks (tf::cudaTask) may run
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:in parallel scheduled by the CUDA runtime.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:Please refer to @ref GPUTaskingcudaFlow for details.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:class cudaFlow {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraph graph;
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @brief constructs a standalone %cudaFlow
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    A standalone %cudaFlow does not go through any taskflow and
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    (e.g., tf::cudaFlow::offload).
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaFlow();
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @brief destroys the %cudaFlow and its associated native CUDA graph
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    ~cudaFlow();
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @brief clears the %cudaFlow object
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @brief dumps the %cudaFlow graph into a DOT format through an
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @brief dumps the native CUDA graph into a DOT format through an
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    The native CUDA graph may be different from the upper-level %cudaFlow
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask noop();
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    A host task can only execute CPU-specific functions and cannot do any CUDA calls
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    (e.g., @c cudaMalloc).
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask host(C&& callable);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    The method is similar to tf::cudaFlow::host but operates on a task
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    of type tf::cudaTaskType::HOST.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void host(cudaTask task, C&& callable);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask kernel(dim3 g, dim3 b, size_t s, F f, ArgsT&&... args);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    The method is similar to tf::cudaFlow::kernel but operates on a task
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    of type tf::cudaTaskType::KERNEL.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:      cudaTask task, dim3 g, dim3 b, size_t shm, F f, ArgsT&&... args
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask memset(void* dst, int v, size_t count);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    The method is similar to tf::cudaFlow::memset but operates on a task
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    of type tf::cudaTaskType::MEMSET.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void memset(cudaTask task, void* dst, int ch, size_t count);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    to a target location. Direction can be arbitrary among CPUs and GPUs.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask memcpy(void* tgt, const void* src, size_t bytes);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    The method is similar to tf::cudaFlow::memcpy but operates on a task
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    of type tf::cudaTaskType::MEMCPY.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void memcpy(cudaTask task, void* tgt, const void* src, size_t bytes);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask zero(T* dst, size_t count);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    The method is similar to tf::cudaFlow::zero but operates on
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    a task of type tf::cudaTaskType::MEMSET.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void zero(cudaTask task, T* dst, size_t count);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask fill(T* dst, T value, size_t count);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    The method is similar to tf::cudaFlow::fill but operates on a task
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    of type tf::cudaTaskType::MEMSET.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void fill(cudaTask task, T* dst, T value, size_t count);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    to a target location. Direction can be arbitrary among CPUs and GPUs.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask copy(T* tgt, const T* src, size_t num);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    The method is similar to tf::cudaFlow::copy but operates on a task
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    of type tf::cudaTaskType::MEMCPY.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void copy(cudaTask task, T* tgt, const T* src, size_t num);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @brief offloads the %cudaFlow onto a GPU and repeatedly runs it until
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    Immediately offloads the present %cudaFlow onto a GPU and
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    An offloaded %cudaFlow forces the underlying graph to be instantiated.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    By default, if users do not offload the %cudaFlow,
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @brief offloads the %cudaFlow and executes it by the given times
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @brief offloads the %cudaFlow and executes it once
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask single_task(C c);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    This method is similar to cudaFlow::single_task but operates
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void single_task(cudaTask task, C c);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask for_each(I first, I last, C callable);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::for_each
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    the task created from tf::cudaFlow::for_each.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void for_each(cudaTask task, I first, I last, C callable);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask for_each_index(I first, I last, I step, C callable);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::for_each_index
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    the task created from tf::cudaFlow::for_each_index.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:      cudaTask task, I first, I last, I step, C callable
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask transform(I first, I last, O output, C op);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::transform
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    the task created from tf::cudaFlow::for_each.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void transform(cudaTask task, I first, I last, O output, C c);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask transform(I1 first1, I1 last1, I2 first2, O output, C op);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::transform
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    the task created from tf::cudaFlow::for_each.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:      cudaTask task, I1 first1, I1 last1, I2 first2, O output, C c
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask reduce(I first, I last, T* result, B bop);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::reduce
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    the task created from tf::cudaFlow::reduce.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void reduce(cudaTask task, I first, I last, T* result, C op);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @brief similar to tf::cudaFlow::reduce but does not assume any initial
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask uninitialized_reduce(I first, I last, T* result, B bop);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::uninitialized_reduce
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    the task created from tf::cudaFlow::uninitialized_reduce.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:      cudaTask task, I first, I last, T* result, C op
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask transform_reduce(I first, I last, T* result, B bop, U uop);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::transform_reduce
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void transform_reduce(cudaTask, I first, I last, T* result, B bop, U uop);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @brief similar to tf::cudaFlow::transform_reduce but does not assume any initial
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask transform_uninitialized_reduce(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::transform_uninitialized_reduce
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:      cudaTask task, I first, I last, T* result, B bop, U uop
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask inclusive_scan(I first, I last, O output, C op);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           from tf::cudaFlow::inclusive_scan
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    This method is similar to tf::cudaFlow::inclusive_scan
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void inclusive_scan(cudaTask task, I first, I last, O output, C op);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @brief similar to cudaFlow::inclusive_scan but excludes the first value
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask exclusive_scan(I first, I last, O output, C op);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::exclusive_scan
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    This method is similar to tf::cudaFlow::exclusive_scan
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void exclusive_scan(cudaTask task, I first, I last, O output, C op);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask transform_inclusive_scan(I first, I last, O output, B bop, U uop);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::transform_inclusive_scan
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    This method is similar to tf::cudaFlow::transform_inclusive_scan
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:      cudaTask task, I first, I last, O output, B bop, U uop
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @brief similar to cudaFlow::transform_inclusive_scan but
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask transform_exclusive_scan(I first, I last, O output, B bop, U uop);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::transform_exclusive_scan
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    This method is similar to tf::cudaFlow::transform_exclusive_scan
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:      cudaTask task, I first, I last, O output, B bop, U uop
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask merge(A a_first, A a_last, B b_first, B b_last, C c_first, Comp comp);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::merge
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    This method is similar to tf::cudaFlow::merge but operates on
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:      cudaTask task, A a_first, A a_last, B b_first, B b_last, C c_first, Comp comp
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask sort(I first, I last, C comp);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::sort
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    This method is similar to tf::cudaFlow::sort but operates on
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void sort(cudaTask task, I first, I last, C comp);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask sort_by_key(K_it k_first, K_it k_last, V_it v_first, C comp);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::sort_by_key
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    This method is similar to tf::cudaFlow::sort_by_key but operates on
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:      cudaTask task, K_it k_first, K_it k_last, V_it v_first, C comp
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask merge_by_key(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::merge_by_key
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    This method is similar to tf::cudaFlow::merge_by_key but operates
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:      cudaTask task,
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask find_if(I first, I last, unsigned* idx, U op);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::find_if
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void find_if(cudaTask task, I first, I last, unsigned* idx, U op);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask min_element(I first, I last, unsigned* idx, O op);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::min_element
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void min_element(cudaTask task, I first, I last, unsigned* idx, O op);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask max_element(I first, I last, unsigned* idx, O op);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:           tf::cudaFlow::max_element
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void max_element(cudaTask task, I first, I last, unsigned* idx, O op);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @brief constructs a subflow graph through tf::cudaFlowCapturer
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:              @c std::function<void(tf::cudaFlowCapturer&)>
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    A captured subflow forms a sub-graph to the %cudaFlow and can be used to
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    from the %cudaFlow.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    taskflow.emplace([&](tf::cudaFlow& cf){
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:      tf::cudaTask my_kernel = cf.kernel(my_arguments);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:      tf::cudaTask my_subflow = cf.capture([&](tf::cudaFlowCapturer& capturer){
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:        capturer.on([&](cudaStream_t stream){
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaTask capture(C&& callable);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    The method is similar to tf::cudaFlow::capture but operates on a task
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    of type tf::cudaTaskType::SUBFLOW.
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    void capture(cudaTask task, C callable);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraph& _graph;
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphExec_t _executable {nullptr};
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaFlow(cudaGraph&);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:// Construct a standalone cudaFlow
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:inline cudaFlow::cudaFlow() :
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphCreate(&_graph._native_handle, 0),
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    "cudaFlow failed to create a native graph (external mode)"
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:// Construct the cudaFlow from executor (internal graph)
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:inline cudaFlow::cudaFlow(cudaGraph& g) :
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphCreate(&_graph._native_handle, 0),
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:inline cudaFlow::~cudaFlow() {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphExecDestroy(_executable);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  cudaGraphDestroy(_graph._native_handle);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:inline void cudaFlow::clear() {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:      cudaGraphExecDestroy(_executable), "failed to destroy executable graph"
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphDestroy(_graph._native_handle), "failed to destroy native graph"
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphCreate(&_graph._native_handle, 0), "failed to create native graph"
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:inline bool cudaFlow::empty() const {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:inline size_t cudaFlow::num_tasks() const {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:inline void cudaFlow::dump(std::ostream& os) const {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:inline void cudaFlow::dump_native_graph(std::ostream& os) const {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  cuda_dump_graph(os, _graph._native_handle);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:inline cudaTask cudaFlow::noop() {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    _graph, std::in_place_type_t<cudaNode::Empty>{}
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphAddEmptyNode(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:cudaTask cudaFlow::host(C&& c) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    _graph, std::in_place_type_t<cudaNode::Host>{}, std::forward<C>(c)
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  auto h = std::get_if<cudaNode::Host>(&node->_handle);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  cudaHostNodeParams p;
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  p.fn = cudaNode::Host::callback;
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphAddHostNode(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:cudaTask cudaFlow::kernel(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    _graph, std::in_place_type_t<cudaNode::Kernel>{}, (void*)f
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  cudaKernelNodeParams p;
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphAddKernelNode(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:cudaTask cudaFlow::zero(T* dst, size_t count) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    _graph, std::in_place_type_t<cudaNode::Memset>{}
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_zero_parms(dst, count);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphAddMemsetNode(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:cudaTask cudaFlow::fill(T* dst, T value, size_t count) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    _graph, std::in_place_type_t<cudaNode::Memset>{}
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_fill_parms(dst, value, count);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphAddMemsetNode(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:cudaTask cudaFlow::copy(T* tgt, const T* src, size_t num) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    _graph, std::in_place_type_t<cudaNode::Memcpy>{}
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_copy_parms(tgt, src, num);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphAddMemcpyNode(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:inline cudaTask cudaFlow::memset(void* dst, int ch, size_t count) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    _graph, std::in_place_type_t<cudaNode::Memset>{}
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_memset_parms(dst, ch, count);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphAddMemsetNode(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:inline cudaTask cudaFlow::memcpy(void* tgt, const void* src, size_t bytes) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    _graph, std::in_place_type_t<cudaNode::Memcpy>{}
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_memcpy_parms(tgt, src, bytes);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphAddMemcpyNode(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:void cudaFlow::host(cudaTask task, C&& c) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  if(task.type() != cudaTaskType::HOST) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  auto h = std::get_if<cudaNode::Host>(&task._node->_handle);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:void cudaFlow::kernel(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  cudaTask task, dim3 g, dim3 b, size_t s, F f, ArgsT&&... args
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  if(task.type() != cudaTaskType::KERNEL) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  cudaKernelNodeParams p;
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphExecKernelNodeSetParams(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:void cudaFlow::copy(cudaTask task, T* tgt, const T* src, size_t num) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  if(task.type() != cudaTaskType::MEMCPY) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_copy_parms(tgt, src, num);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphExecMemcpyNodeSetParams(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:inline void cudaFlow::memcpy(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  cudaTask task, void* tgt, const void* src, size_t bytes
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  if(task.type() != cudaTaskType::MEMCPY) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_memcpy_parms(tgt, src, bytes);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphExecMemcpyNodeSetParams(_executable, task._node->_native_handle, &p),
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:void cudaFlow::memset(cudaTask task, void* dst, int ch, size_t count) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  if(task.type() != cudaTaskType::MEMSET) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_memset_parms(dst, ch, count);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphExecMemsetNodeSetParams(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:void cudaFlow::fill(cudaTask task, T* dst, T value, size_t count) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  if(task.type() != cudaTaskType::MEMSET) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_fill_parms(dst, value, count);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphExecMemsetNodeSetParams(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:void cudaFlow::zero(cudaTask task, T* dst, size_t count) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  if(task.type() != cudaTaskType::MEMSET) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  auto p = cuda_get_zero_parms(dst, count);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphExecMemsetNodeSetParams(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:void cudaFlow::capture(cudaTask task, C c) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  if(task.type() != cudaTaskType::SUBFLOW) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  auto node_handle = std::get_if<cudaNode::Subflow>(&task._node->_handle);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  cudaFlowCapturer capturer(node_handle->graph);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  //cuda_dump_graph(std::cout, captured);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphExecChildGraphNodeSetParams(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(cudaGraphDestroy(captured), "failed to destroy captured graph");
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:cudaTask cudaFlow::capture(C&& c) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    _graph, std::in_place_type_t<cudaNode::Subflow>{}
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  auto node_handle = std::get_if<cudaNode::Subflow>(&node->_handle);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  cudaFlowCapturer capturer(node_handle->graph);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  //cuda_dump_graph(std::cout, captured);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    cudaGraphAddChildGraphNode(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    "failed to add a cudaFlow capturer task"
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  TF_CHECK_CUDA(cudaGraphDestroy(captured), "failed to destroy captured graph");
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  return cudaTask(node);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:void cudaFlow::offload_until(P&& predicate) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  // transforms cudaFlow to a native cudaGraph under the specified device
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:      cudaGraphInstantiate(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    //cuda_dump_graph(std::cout, cf._graph._native_handle);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  //cudaScopedPerThreadStream s;
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  cudaStream s;
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:      cudaGraphLaunch(_executable, s), "failed to execute cudaFlow"
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    //TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    //  cudaStreamSynchronize(s), "failed to synchronize cudaFlow execution"
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  _graph._state = cudaGraph::OFFLOADED;
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:inline void cudaFlow::offload_n(size_t n) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:inline void cudaFlow::offload() {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  std::enable_if_t<is_cudaflow_task_v<C>, void>*
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    std::in_place_type_t<Node::cudaFlow>{},
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:      cudaScopedDevice ctx(d);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:      e._invoke_cudaflow_task_entry(p, c);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    std::make_unique<cudaGraph>()
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:template <typename C, std::enable_if_t<is_cudaflow_task_v<C>, void>*>
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  return emplace_on(std::forward<C>(c), tf::cuda_get_device());
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:// Procedure: _invoke_cudaflow_task_entry
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:template <typename C, std::enable_if_t<is_cudaflow_task_v<C>, void>*>
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:void Executor::_invoke_cudaflow_task_entry(Node* node, C&& c) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:    std::is_invocable_r_v<void, C, cudaFlow&>, cudaFlow, cudaFlowCapturer
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  auto h = std::get_if<Node::cudaFlow>(&node->_handle);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  cudaGraph* g = dynamic_cast<cudaGraph*>(h->graph.get());
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  if(!(g->_state & cudaGraph::OFFLOADED)) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:/*// Procedure: _invoke_cudaflow_task_entry (cudaFlow)
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  std::enable_if_t<std::is_invocable_r_v<void, C, cudaFlow&>, void>*
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:void Executor::_invoke_cudaflow_task_entry(Node* node, C&& c) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  auto h = std::get_if<Node::cudaFlow>(&node->_handle);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  cudaGraph* g = dynamic_cast<cudaGraph*>(h->graph.get());
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  cudaFlow cf(*g);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:// Procedure: _invoke_cudaflow_task_entry (cudaFlowCapturer)
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  std::enable_if_t<std::is_invocable_r_v<void, C, cudaFlowCapturer&>, void>*
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:void Executor::_invoke_cudaflow_task_entry(Node* node, C&& c) {
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  auto h = std::get_if<Node::cudaFlow>(&node->_handle);
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  cudaGraph* g = dynamic_cast<cudaGraph*>(h->graph.get());
tools/loaders/bam-loader/taskflow/cuda/cudaflow.hpp:  cudaFlowCapturer fc(*g);
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:#include "cuda_graph.hpp"
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:@file cuda_task.hpp
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:@brief cudaTask include file
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:// cudaTask Types
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:@enum cudaTaskType
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:@brief enumeration of all %cudaTask types
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:enum class cudaTaskType : int {
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:@brief convert a cuda_task type to a human-readable string
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:constexpr const char* to_string(cudaTaskType type) {
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::EMPTY:   return "empty";
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::HOST:    return "host";
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::MEMSET:  return "memset";
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::MEMCPY:  return "memcpy";
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::KERNEL:  return "kernel";
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::SUBFLOW: return "subflow";
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    case cudaTaskType::CAPTURE: return "capture";
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:// cudaTask
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:@class cudaTask
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:@brief class to create a task handle over an internal node of a %cudaFlow graph
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:class cudaTask {
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:  friend class cudaFlow;
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:  friend class cudaFlowCapturer;
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:  friend class cudaFlowCapturerBase;
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:  friend std::ostream& operator << (std::ostream&, const cudaTask&);
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    @brief constructs an empty cudaTask
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    cudaTask() = default;
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    @brief copy-constructs a cudaTask
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    cudaTask(const cudaTask&) = default;
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    @brief copy-assigns a cudaTask
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    cudaTask& operator = (const cudaTask&) = default;
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    cudaTask& precede(Ts&&... tasks);
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    cudaTask& succeed(Ts&&... tasks);
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    cudaTask& name(const std::string& name);
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    @brief queries if the task is associated with a cudaNode
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    cudaTaskType type() const;
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    cudaTask(cudaNode*);
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    cudaNode* _node {nullptr};
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:inline cudaTask::cudaTask(cudaNode* node) : _node {node} {
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:cudaTask& cudaTask::precede(Ts&&... tasks) {
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:cudaTask& cudaTask::succeed(Ts&&... tasks) {
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:inline bool cudaTask::empty() const {
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:inline cudaTask& cudaTask::name(const std::string& name) {
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:inline const std::string& cudaTask::name() const {
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:inline size_t cudaTask::num_successors() const {
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:inline size_t cudaTask::num_dependents() const {
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:inline cudaTaskType cudaTask::type() const {
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    case cudaNode::EMPTY:   return cudaTaskType::EMPTY;
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    case cudaNode::HOST:    return cudaTaskType::HOST;
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    case cudaNode::MEMSET:  return cudaTaskType::MEMSET;
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    case cudaNode::MEMCPY:  return cudaTaskType::MEMCPY;
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    case cudaNode::KERNEL:  return cudaTaskType::KERNEL;
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    case cudaNode::SUBFLOW: return cudaTaskType::SUBFLOW;
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    case cudaNode::CAPTURE: return cudaTaskType::CAPTURE;
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    default:                return cudaTaskType::UNDEFINED;
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:void cudaTask::dump(T& os) const {
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:  os << "cudaTask ";
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:void cudaTask::for_each_successor(V&& visitor) const {
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    visitor(cudaTask(_node->_successors[i]));
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:void cudaTask::for_each_dependent(V&& visitor) const {
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:    visitor(cudaTask(_node->_dependents[i]));
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:@brief overload of ostream inserter operator for cudaTask
tools/loaders/bam-loader/taskflow/cuda/cuda_task.hpp:inline std::ostream& operator << (std::ostream& os, const cudaTask& ct) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:#include "cuda_execution_policy.hpp"
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:inline constexpr unsigned CUDA_WARP_SIZE = 32;
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:struct cudaEmpty { };
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:struct cudaIterate {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:    cudaIterate<i + 1, count>::eval(f);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:struct cudaIterate<i, count, false> {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_iterate(F f) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaIterate<begin, end>::eval(f);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_iterate(F f) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_iterate<0, count>(f);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_iterate<count>([&](auto i) { y = i ? x[i] + y : x[i]; });
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_iterate<count>([&](auto i) { x[i] = val; });
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_strided_iterate(F f, unsigned tid) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_iterate<vt>([=](auto i) { f(i, nt * i + tid); });
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_strided_iterate(F f, unsigned tid, unsigned count) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:    cuda_strided_iterate<nt, vt0>(f, tid);    // No checking
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:    cuda_iterate<vt0>([=](auto i) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_iterate<vt0, vt>([=](auto i) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_thread_iterate(F f, unsigned tid) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_iterate<vt>([=](auto i) { f(i, vt * tid + i); });
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:// cudaRange
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:// cudaRange
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:struct cudaRange {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:inline __device__ cudaRange cuda_get_tile(unsigned b, unsigned nv, unsigned count) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  return cudaRange { nv * b, min(count, nv * (b + 1)) };
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:// cudaArray
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:struct cudaArray {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaArray() = default;
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaArray(const cudaArray&) = default;
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaArray& operator=(const cudaArray&) = default;
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  __device__ cudaArray(T x) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:    cuda_iterate<size>([&](unsigned i) { data[i] = x; });
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:struct cudaArray<T, 0> {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:struct cudaKVArray {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, size> keys;
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaArray<V, size> vals;
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_mem_to_reg_strided(I mem, unsigned tid, unsigned count) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x;
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_strided_iterate<nt, vt, vt0>(
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_reg_to_mem_strided(
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x, unsigned tid, unsigned count, it_t mem) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_strided_iterate<nt, vt, vt0>(
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_transform_mem_to_reg_strided(
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x;
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_strided_iterate<nt, vt, vt0>(
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_reg_to_shared_thread(
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x, unsigned tid, T (&shared)[shared_size], bool sync = true
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_thread_iterate<vt>([&](auto i, auto j) { shared[j] = x[i]; }, tid);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_shared_to_reg_thread(
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x;
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_thread_iterate<vt>([&](auto i, auto j) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_reg_to_shared_strided(
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x, unsigned tid, T (&shared)[shared_size], bool sync = true
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_strided_iterate<nt, vt>(
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_shared_to_reg_strided(
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x;
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_strided_iterate<nt, vt>([&](auto i, auto j) { x[i] = shared[j]; }, tid);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_reg_to_mem_thread(
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x, unsigned tid,
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_reg_to_shared_thread<nt>(x, tid, shared);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  auto y = cuda_shared_to_reg_strided<nt, vt>(shared, tid);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_reg_to_mem_strided<nt, vt, vt0>(y, tid, count, mem);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_mem_to_reg_thread(
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  auto x = cuda_mem_to_reg_strided<nt, vt, vt0>(mem, tid, count);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_reg_to_shared_strided<nt, vt>(x, tid, shared);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  auto y = cuda_shared_to_reg_thread<nt, vt>(shared, tid);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_shared_gather(
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  const T(&data)[S], cudaArray<unsigned, vt> indices, bool sync = true
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x;
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_iterate<vt>([&](auto i) { x[i] = data[indices[i]]; });
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_reg_thread_to_strided(
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x, unsigned tid, T (&shared)[S]
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_reg_to_shared_thread<nt>(x, tid, shared);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  return cuda_shared_to_reg_strided<nt, vt>(shared, tid);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ auto cuda_reg_strided_to_thread(
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaArray<T, vt> x, unsigned tid, T (&shared)[S]
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cuda_reg_to_shared_strided<nt>(x, tid, shared);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  return cuda_shared_to_reg_thread<nt, vt>(shared, tid);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:// cudaLoadStoreIterator
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:struct cudaLoadStoreIterator : std::iterator_traits<const T*> {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  cudaLoadStoreIterator(L load_, S store_, I base_) :
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:      static_assert(!std::is_same<S, cudaEmpty>::value,
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:      static_assert(!std::is_same<L, cudaEmpty>::value,
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  __device__ cudaLoadStoreIterator operator+(I offset) const {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:    cudaLoadStoreIterator cp = *this;
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  __device__ cudaLoadStoreIterator& operator+=(I offset) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  __device__ cudaLoadStoreIterator operator-(I offset) const {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:    cudaLoadStoreIterator cp = *this;
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  __device__ cudaLoadStoreIterator& operator-=(I offset) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:auto cuda_make_load_store_iterator(L load, S store, I base = 0) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  return cudaLoadStoreIterator<L, S, T, I>(load, store, base);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:auto cuda_make_load_iterator(L load, I base = 0) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  return cuda_make_load_store_iterator<T>(load, cudaEmpty(), base);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:auto cuda_make_store_iterator(S store, I base = 0) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:  return cuda_make_load_store_iterator<T>(cudaEmpty(), store, base);
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__device__ void cuda_swap(T& a, T& b) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:__global__ void cuda_kernel(F f, args_t... args) {
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:struct cuda_plus{
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:struct cuda_minus{
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:struct cuda_multiplies{
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:struct cuda_maximum{
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:struct cuda_minimum{
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:struct cuda_less{
tools/loaders/bam-loader/taskflow/cuda/cuda_meta.hpp:struct cuda_greater{
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:#include "cuda_task.hpp"
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:#include "cuda_optimizer.hpp"
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:@file cuda_capturer.hpp
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:@brief %cudaFlow capturer include file
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:// class definition: cudaFlowCapturer
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:@class cudaFlowCapturer
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:@brief class to create a %cudaFlow graph using stream capture
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:The usage of tf::cudaFlowCapturer is similar to tf::cudaFlow, except users can
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:call the method tf::cudaFlowCapturer::on to capture a sequence of asynchronous
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:CUDA operations through the given stream.
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:The following example creates a CUDA graph that captures two kernel tasks,
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:taskflow.emplace([](tf::cudaFlowCapturer& capturer){
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  auto task_1 = capturer.on([&](cudaStream_t stream){
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  auto task_2 = capturer.on([&](cudaStream_t stream){
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:Similar to tf::cudaFlow, a %cudaFlowCapturer is a task (tf::Task)
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:That is, the callable that describes a %cudaFlowCapturer
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:Inside a %cudaFlow capturer task, different GPU tasks (tf::cudaTask) may run
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:By default, we use tf::cudaRoundRobinCapturing to transform a user-level
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:graph into a native CUDA graph.
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:Please refer to @ref GPUTaskingcudaFlowCapturer for details.
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:class cudaFlowCapturer {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  friend class cudaFlow;
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaGraph graph;
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaRoundRobinCapturing,
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaSequentialCapturing,
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaLinearCapturing
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @brief constrcts a standalone cudaFlowCapturer
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    A standalone %cudaFlow capturer does not go through any taskflow and
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    (e.g., tf::cudaFlow::offload).
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaFlowCapturer();
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @brief destructs the cudaFlowCapturer
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    virtual ~cudaFlowCapturer();
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @brief clear this %cudaFlow capturer
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    a user-described %cudaFlow:
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      + tf::cudaSequentialCapturing
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      + tf::cudaRoundRobinCapturing
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      + tf::cudaLinearCapturing
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    By default, tf::cudaFlowCapturer uses the round-robin optimization
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    a native CUDA graph.
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @brief captures a sequential CUDA operations from the given callable
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @tparam C callable type constructible with @c std::function<void(cudaStream_t)>
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @param callable a callable to capture CUDA operations with the stream
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    a sequence of CUDA operations defined in the callable.
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      std::is_invocable_r_v<void, C, cudaStream_t>, void>* = nullptr
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask on(C&& callable);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @brief updates a capture task to another sequential CUDA operations
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    The method is similar to cudaFlowCapturer::on but operates
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      std::is_invocable_r_v<void, C, cudaStream_t>, void>* = nullptr
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    void on(cudaTask task, C&& callable);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask noop();
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    The method is similar to tf::cudaFlowCapturer::noop but
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    void noop(cudaTask task);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    The method captures a @c cudaMemcpyAsync operation through an
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask memcpy(void* dst, const void* src, size_t count);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    The method is similar to cudaFlowCapturer::memcpy but operates on an
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    void memcpy(cudaTask task, void* dst, const void* src, size_t count);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @return cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    to a target location. Direction can be arbitrary among CPUs and GPUs.
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask copy(T* tgt, const T* src, size_t num);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    The method is similar to cudaFlowCapturer::copy but operates on
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    void copy(cudaTask task, T* tgt, const T* src, size_t num);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @brief initializes or sets GPU memory to the given value byte by byte
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @param ptr pointer to GPU mempry
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    The method captures a @c cudaMemsetAsync operation through an
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask memset(void* ptr, int v, size_t n);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    The method is similar to cudaFlowCapturer::memset but operates on
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    void memset(cudaTask task, void* ptr, int value, size_t n);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @return cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask kernel(dim3 g, dim3 b, size_t s, F f, ArgsT&&... args);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    The method is similar to cudaFlowCapturer::kernel but operates on
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaTask task, dim3 g, dim3 b, size_t s, F f, ArgsT&&... args
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask single_task(C c);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::single_task but operates
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    void single_task(cudaTask task, C c);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @return cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask for_each(I first, I last, C callable);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::for_each but operates
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    void for_each(cudaTask task, I first, I last, C callable);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @return cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask for_each_index(I first, I last, I step, C callable);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::for_each_index but operates
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaTask task, I first, I last, I step, C callable
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @return cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask transform(I first, I last, O output, C op);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::transform but operates
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    void transform(cudaTask task, I first, I last, O output, C op);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @return cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask transform(I1 first1, I1 last1, I2 first2, O output, C op);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::transform but operates
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaTask task, I1 first1, I1 last1, I2 first2, O output, C op
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask reduce(I first, I last, T* result, C op);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::reduce but operates
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    void reduce(cudaTask task, I first, I last, T* result, C op);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @brief similar to tf::cudaFlowCapturer::reduce but does not assume
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask uninitialized_reduce(I first, I last, T* result, C op);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::uninitialized_reduce
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaTask task, I first, I last, T* result, C op
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask transform_reduce(I first, I last, T* result, C bop, U uop);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::transform_reduce but
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaTask task, I first, I last, T* result, C bop, U uop
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @brief similar to tf::cudaFlowCapturer::transform_reduce but does not assume
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask transform_uninitialized_reduce(I first, I last, T* result, C bop, U uop);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::transform_uninitialized_reduce
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaTask task, I first, I last, T* result, C bop, U uop
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is equivalent to the parallel execution of the following loop on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask inclusive_scan(I first, I last, O output, C op);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::inclusive_scan
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    void inclusive_scan(cudaTask task, I first, I last, O output, C op);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @brief similar to cudaFlowCapturer::inclusive_scan
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask exclusive_scan(I first, I last, O output, C op);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::exclusive_scan
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    void exclusive_scan(cudaTask task, I first, I last, O output, C op);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    on a GPU:
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask transform_inclusive_scan(I first, I last, O output, B bop, U uop);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::transform_inclusive_scan
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaTask task, I first, I last, O output, B bop, U uop
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @brief similar to cudaFlowCapturer::transform_inclusive_scan but
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask transform_exclusive_scan(I first, I last, O output, B bop, U uop);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::transform_exclusive_scan
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaTask task, I first, I last, O output, B bop, U uop
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask merge(A a_first, A a_last, B b_first, B b_last, C c_first, Comp comp);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::merge but operates
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaTask task, A a_first, A a_last, B b_first, B b_last, C c_first, Comp comp
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask merge_by_key(
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to tf::cudaFlowCapturer::merge_by_key but operates
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaTask task,
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask sort(I first, I last, C comp);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::sort but operates on
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    void sort(cudaTask task, I first, I last, C comp);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @return a tf::cudaTask handle
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask sort_by_key(K_it k_first, K_it k_last, V_it v_first, C comp);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to tf::cudaFlowCapturer::sort_by_key
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaTask task, K_it k_first, K_it k_last, V_it v_first, C comp
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask find_if(I first, I last, unsigned* idx, U op);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to tf::cudaFlowCapturer::find_if but operates
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    void find_if(cudaTask task, I first, I last, unsigned* idx, U op);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask min_element(I first, I last, unsigned* idx, O op);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::min_element but operates
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    void min_element(cudaTask task, I first, I last, unsigned* idx, O op);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaTask max_element(I first, I last, unsigned* idx, O op);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    This method is similar to cudaFlowCapturer::max_element but operates
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    void max_element(cudaTask task, I first, I last, unsigned* idx, O op);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @brief offloads the captured %cudaFlow onto a GPU and repeatedly runs it until
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    Immediately offloads the %cudaFlow captured so far onto a GPU and
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    By default, if users do not offload the %cudaFlow capturer,
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @brief offloads the captured %cudaFlow and executes it by the given times
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    @brief offloads the captured %cudaFlow and executes it once
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaGraph& _graph;
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaGraphExec_t _executable {nullptr};
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaFlowCapturer(cudaGraph&);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaGraph_t _capture();
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:// constructs a cudaFlow capturer from a taskflow
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:inline cudaFlowCapturer::cudaFlowCapturer(cudaGraph& g) :
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:// constructs a standalone cudaFlow capturer
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:inline cudaFlowCapturer::cudaFlowCapturer() :
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:inline cudaFlowCapturer::~cudaFlowCapturer() {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaGraphExecDestroy(_executable);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:inline bool cudaFlowCapturer::empty() const {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:inline size_t cudaFlowCapturer::num_tasks() const {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::clear() {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::dump(std::ostream& os) const {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::_destroy_executable() {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaGraphExecDestroy(_executable), "failed to destroy executable graph"
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  std::is_invocable_r_v<void, C, cudaStream_t>, void>*
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::on(C&& callable) {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    std::in_place_type_t<cudaNode::Capture>{}, std::forward<C>(callable)
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  return cudaTask(node);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:inline cudaTask cudaFlowCapturer::noop() {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  return on([](cudaStream_t){});
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::noop(cudaTask task) {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  on(task, [](cudaStream_t){});
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:inline cudaTask cudaFlowCapturer::memcpy(
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  return on([dst, src, count] (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream),
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::copy(T* tgt, const T* src, size_t num) {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  return on([tgt, src, num] (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaMemcpyAsync(tgt, src, sizeof(T)*num, cudaMemcpyDefault, stream),
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:inline cudaTask cudaFlowCapturer::memset(void* ptr, int v, size_t n) {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  return on([ptr, v, n] (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaMemsetAsync(ptr, v, n, stream), "failed to capture memset"
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:cudaTask cudaFlowCapturer::kernel(
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  return on([g, b, s, f, args...] (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:inline cudaGraph_t cudaFlowCapturer::_capture() {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:void cudaFlowCapturer::offload_until(P&& predicate) {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  if(_graph._state & cudaGraph::CHANGED) {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaGraphInstantiate(&_executable, g, nullptr, nullptr, 0),
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    //cuda_dump_graph(std::cout, g);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(cudaGraphDestroy(g), "failed to destroy captured graph");
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  else if(_graph._state & cudaGraph::UPDATED) {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaGraphNode_t error_node;
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaGraphExecUpdateResult error_result;
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaGraphExecUpdate(_executable, g, &error_node, &error_result);
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    if(error_result != cudaGraphExecUpdateSuccess) {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:        cudaGraphInstantiate(&_executable, g, nullptr, nullptr, 0),
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(cudaGraphDestroy(g), "failed to destroy captured graph");
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    //cudaScopedPerThreadStream s;
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    cudaStream s;
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:        cudaGraphLaunch(_executable, s), "failed to launch the exec graph"
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      //TF_CHECK_CUDA(cudaStreamSynchronize(s), "failed to synchronize stream");
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  _graph._state = cudaGraph::OFFLOADED;
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::offload_n(size_t n) {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::offload() {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  std::is_invocable_r_v<void, C, cudaStream_t>, void>*
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:void cudaFlowCapturer::on(cudaTask task, C&& callable) {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  if(task.type() != cudaTaskType::CAPTURE) {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    TF_THROW("invalid cudaTask type (must be CAPTURE)");
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  _graph._state |= cudaGraph::UPDATED;
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  std::get_if<cudaNode::Capture>(&task._node->_handle)->work =
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::memcpy(
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  cudaTask task, void* dst, const void* src, size_t count
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  on(task, [dst, src, count](cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream),
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:void cudaFlowCapturer::copy(
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  cudaTask task, T* tgt, const T* src, size_t num
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  on(task, [tgt, src, num] (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaMemcpyAsync(tgt, src, sizeof(T)*num, cudaMemcpyDefault, stream),
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:inline void cudaFlowCapturer::memset(
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  cudaTask task, void* ptr, int v, size_t n
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  on(task, [ptr, v, n] (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:    TF_CHECK_CUDA(
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:      cudaMemsetAsync(ptr, v, n, stream), "failed to capture memset"
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:void cudaFlowCapturer::kernel(
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  cudaTask task, dim3 g, dim3 b, size_t s, F f, ArgsT&&... args
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:  on(task, [g, b, s, f, args...] (cudaStream_t stream) mutable {
tools/loaders/bam-loader/taskflow/cuda/cuda_capturer.hpp:OPT& cudaFlowCapturer::make_optimizer(ArgsT&&... args) {
tools/loaders/bam-loader/spdlog/fmt/bundled/format-inl.h:  // Chosen instead of std::abort to satisfy Clang in CUDA mode during device
tools/loaders/bam-loader/spdlog/fmt/bundled/format.h:#  define FMT_CUDA_VERSION (__CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__)
tools/loaders/bam-loader/spdlog/fmt/bundled/format.h:#  define FMT_CUDA_VERSION 0
tools/loaders/bam-loader/spdlog/fmt/bundled/format.h:// EDG based compilers (Intel, NVIDIA, Elbrus, etc), GCC and MSVC support UDLs.

```
