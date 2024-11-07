# https://github.com/lsds/KungFu

```console
setup.py:        use_nccl = os.getenv('KUNGFU_ENABLE_NCCL')
setup.py:        if use_nccl:
setup.py:            cmake_args.append(cmake_flag('KUNGFU_ENABLE_NCCL', use_nccl))
setup.py:            nccl_home = os.getenv('NCCL_HOME')
setup.py:            if nccl_home:
setup.py:                cmake_args.append(cmake_flag('NCCL_HOME', nccl_home))
docs/index.rst:or heavy dependencies like OpenMPI and NCCL as in Horovod.
docs/index.rst:and your server, with and without GPUs.
experimental/adapt_strategy/adapt_strategy.py:parser.add_argument('--no-cuda',
experimental/adapt_strategy/adapt_strategy.py:                    help='disables CUDA training')
experimental/adapt_strategy/adapt_strategy.py:args.cuda = not args.no_cuda
experimental/adapt_strategy/adapt_strategy.py:if args.cuda:
experimental/adapt_strategy/adapt_strategy.py:    config.gpu_options.allow_growth = True
experimental/adapt_strategy/adapt_strategy.py:    from kungfu.python import _get_cuda_index
experimental/adapt_strategy/adapt_strategy.py:    config.gpu_options.visible_device_list = str(_get_cuda_index())
experimental/adapt_strategy/adapt_strategy.py:    config.gpu_options.allow_growth = False
experimental/adapt_strategy/adapt_strategy.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
experimental/adapt_strategy/adapt_strategy.py:    config.gpu_options.visible_device_list = ''
experimental/adapt_strategy/adapt_strategy.py:    elif args.kf_optimizer == 'sync-sgd-nccl':
experimental/adapt_strategy/adapt_strategy.py:        opt = SynchronousSGDOptimizer(opt, nccl=True, nccl_fusion=args.fuse)
experimental/adapt_strategy/adapt_strategy.py:device = '/gpu:0' if args.cuda else 'CPU'
tests/python/integration/test_torch_ops.py:if torch.cuda.is_available():
tests/python/integration/test_torch_ops.py:    test_device('cuda:0')
tests/python/integration/test_tensorflow_resize.py:    p.add_argument('--use-nccl', action='store_true', default=False, help='')
tests/python/integration/test_tensorflow_resize.py:def build_fake_train_op(use_nccl):
tests/python/integration/test_tensorflow_resize.py:    if use_nccl:
tests/python/integration/test_tensorflow_resize.py:        from kungfu.tensorflow.ops import group_nccl_all_reduce
tests/python/integration/test_tensorflow_resize.py:        ys = group_nccl_all_reduce(xs)
tests/python/integration/test_tensorflow_resize.py:    train_op = build_fake_train_op(args.use_nccl)
tests/cpp/integration/test_nccl_helper.cpp:#include "cuda_vector.hpp"
tests/cpp/integration/test_nccl_helper.cpp:#include <kungfu/nccl/helper.hpp>
tests/cpp/integration/test_nccl_helper.cpp:    auto &nccl_helper = kungfu::NCCLHelper::GetDefault(true);
tests/cpp/integration/test_nccl_helper.cpp:        cuda_vector<R> x(n);
tests/cpp/integration/test_nccl_helper.cpp:        cuda_vector<R> y(n);
tests/cpp/integration/test_nccl_helper.cpp:        kungfu::CudaStream s;
tests/cpp/integration/test_nccl_helper.cpp:        auto controller = nccl_helper->EnsureController(KungFu_NCCL_GLOBAL);
tests/cpp/integration/test_nccl_helper.cpp:        controller->AllReduce(w, KungFu_SUM, static_cast<cudaStream_t>(s));
tests/cpp/integration/test_nccl_helper.cpp:        cuda_vector<R> x(n);
tests/cpp/integration/test_nccl_helper.cpp:        cuda_vector<R> y(n);
tests/cpp/integration/test_nccl_helper.cpp:        kungfu::CudaStream s;
tests/cpp/integration/test_nccl_helper.cpp:        s.memcpy(x.data(), x_cpu.data(), n * sizeof(R), cudaMemcpyHostToDevice);
tests/cpp/integration/test_nccl_helper.cpp:        s.memcpy(y.data(), y_cpu.data(), n * sizeof(R), cudaMemcpyHostToDevice);
tests/cpp/integration/test_nccl_helper.cpp:        auto controller = nccl_helper->EnsureGroupController(topology);
tests/cpp/integration/test_nccl_helper.cpp:        controller->AllReduce(w, KungFu_SUM, static_cast<cudaStream_t>(s));
tests/cpp/integration/test_nccl_helper.cpp:        s.memcpy(y_cpu.data(), y.data(), n * sizeof(R), cudaMemcpyDeviceToHost);
tests/cpp/integration/test_nccl_helper.cpp:    nccl_helper.reset(nullptr);
tests/cpp/integration/fake_nccl_trainer.cpp:#include <nccl.h>
tests/cpp/integration/fake_nccl_trainer.cpp:#include "collective_nccl_impl.hpp"
tests/cpp/integration/fake_nccl_trainer.cpp:#include "cuda_vector.hpp"
tests/cpp/integration/fake_nccl_trainer.cpp:template <typename T> struct fake_gpu_buffer_t {
tests/cpp/integration/fake_nccl_trainer.cpp:    cuda_vector<T> send_buf;
tests/cpp/integration/fake_nccl_trainer.cpp:    cuda_vector<T> recv_buf;
tests/cpp/integration/fake_nccl_trainer.cpp:    fake_gpu_buffer_t(const std::string &name, int count)
tests/cpp/integration/fake_nccl_trainer.cpp:void simple_test(int size, nccl_collective &nccl)
tests/cpp/integration/fake_nccl_trainer.cpp:    const int rank = nccl.rank();
tests/cpp/integration/fake_nccl_trainer.cpp:    const int np   = nccl.cluster_size();
tests/cpp/integration/fake_nccl_trainer.cpp:    cuda_vector<int32_t> x(n);
tests/cpp/integration/fake_nccl_trainer.cpp:    cuda_vector<int32_t> y(n);
tests/cpp/integration/fake_nccl_trainer.cpp:    nccl.all_reduce(x.data(), y.data(), n, "test-tensor");
tests/cpp/integration/fake_nccl_trainer.cpp:    ncclUniqueId id;
tests/cpp/integration/fake_nccl_trainer.cpp:        KUNGFU_CHECK(nccl_checker) << ncclGetUniqueId(&id);
tests/cpp/integration/fake_nccl_trainer.cpp:    bootstrap.template bcast<uint8_t>((uint8_t *)&id, sizeof(id), "nccl id");
tests/cpp/integration/fake_nccl_trainer.cpp:    nccl_collective nccl(id, bootstrap.cluster_size(), bootstrap.rank());
tests/cpp/integration/fake_nccl_trainer.cpp:        for (int i = 1; i < 10; ++i) { simple_test(i * Mi, nccl); }
tests/cpp/integration/fake_nccl_trainer.cpp:        for (int i = 1; i < 10; ++i) { simple_test(i * 10 * Mi, nccl); }
tests/cpp/integration/fake_nccl_trainer.cpp:    run_experiment<nccl_collective, fake_gpu_buffer_t<float>>(grad_sizes, nccl);
tests/cpp/integration/cuda_vector.hpp:#include <cuda_runtime.h>
tests/cpp/integration/cuda_vector.hpp:#include <kungfu/cuda/stream.hpp>
tests/cpp/integration/cuda_vector.hpp:struct cuda_mem_allocator {
tests/cpp/integration/cuda_vector.hpp:        // KUNGFU_CHECK(cuda_checker) << cudaMalloc<T>(&deviceMem, count);
tests/cpp/integration/cuda_vector.hpp:        KUNGFU_CHECK(kungfu::cuda_checker)
tests/cpp/integration/cuda_vector.hpp:            << cudaMalloc(&deviceMem, count * sizeof(T));
tests/cpp/integration/cuda_vector.hpp:struct cuda_mem_deleter {
tests/cpp/integration/cuda_vector.hpp:        KUNGFU_CHECK(kungfu::cuda_checker) << cudaFree(ptr);
tests/cpp/integration/cuda_vector.hpp:class cuda_vector
tests/cpp/integration/cuda_vector.hpp:    std::unique_ptr<T, cuda_mem_deleter> data_;
tests/cpp/integration/cuda_vector.hpp:    explicit cuda_vector(size_t count)
tests/cpp/integration/cuda_vector.hpp:        : count(count), data_(cuda_mem_allocator<T>()(count))
tests/cpp/integration/cuda_vector.hpp:        KUNGFU_CHECK(kungfu::cuda_checker) << cudaMemcpy(
tests/cpp/integration/cuda_vector.hpp:            data_.get(), buffer, count * sizeof(T), cudaMemcpyHostToDevice);
tests/cpp/integration/cuda_vector.hpp:        KUNGFU_CHECK(kungfu::cuda_checker) << cudaMemcpy(
tests/cpp/integration/cuda_vector.hpp:            buffer, data_.get(), count * sizeof(T), cudaMemcpyDeviceToHost);
tests/cpp/integration/collective_nccl_impl.hpp:#include <cuda_runtime.h>
tests/cpp/integration/collective_nccl_impl.hpp:#include <nccl.h>
tests/cpp/integration/collective_nccl_impl.hpp:#include <kungfu/cuda/stream.hpp>
tests/cpp/integration/collective_nccl_impl.hpp:using kungfu::cuda_checker;
tests/cpp/integration/collective_nccl_impl.hpp:struct show_nccl_error {
tests/cpp/integration/collective_nccl_impl.hpp:    std::string operator()(ncclResult_t err) const
tests/cpp/integration/collective_nccl_impl.hpp:        return ncclGetErrorString(err);
tests/cpp/integration/collective_nccl_impl.hpp:using nccl_checker = error_checker<ncclResult_t, ncclSuccess, show_nccl_error>;
tests/cpp/integration/collective_nccl_impl.hpp:struct nccl_type;
tests/cpp/integration/collective_nccl_impl.hpp:struct nccl_type<int32_t> {
tests/cpp/integration/collective_nccl_impl.hpp:    static auto value() { return ncclInt32; }
tests/cpp/integration/collective_nccl_impl.hpp:struct nccl_type<float> {
tests/cpp/integration/collective_nccl_impl.hpp:    static auto value() { return ncclFloat; }
tests/cpp/integration/collective_nccl_impl.hpp:class nccl_collective
tests/cpp/integration/collective_nccl_impl.hpp:    ncclComm_t comm;
tests/cpp/integration/collective_nccl_impl.hpp:    nccl_collective(ncclUniqueId id, int cluster_size, int rank)
tests/cpp/integration/collective_nccl_impl.hpp:            KUNGFU_CHECK(cuda_checker) << cudaSetDevice(0);
tests/cpp/integration/collective_nccl_impl.hpp:            printf("cuda device selected to %d\n", 0);
tests/cpp/integration/collective_nccl_impl.hpp:            KUNGFU_CHECK(cuda_checker) << cudaSetDevice(rank);
tests/cpp/integration/collective_nccl_impl.hpp:            printf("cuda device selected to %d\n", rank);
tests/cpp/integration/collective_nccl_impl.hpp:        KUNGFU_CHECK(nccl_checker)
tests/cpp/integration/collective_nccl_impl.hpp:            << ncclCommInitRank(&comm, cluster_size, id, rank);
tests/cpp/integration/collective_nccl_impl.hpp:        printf("nccl inited: %d/%d.\n", rank, cluster_size);
tests/cpp/integration/collective_nccl_impl.hpp:    ~nccl_collective()
tests/cpp/integration/collective_nccl_impl.hpp:        ncclCommDestroy(comm);
tests/cpp/integration/collective_nccl_impl.hpp:        printf("nccl destroyed: %d/%d.\n", _rank, _cluster_size);
tests/cpp/integration/collective_nccl_impl.hpp:        cudaStream_t stream;
tests/cpp/integration/collective_nccl_impl.hpp:        KUNGFU_CHECK(cuda_checker) << cudaStreamCreate(&stream);
tests/cpp/integration/collective_nccl_impl.hpp:        KUNGFU_CHECK(nccl_checker)
tests/cpp/integration/collective_nccl_impl.hpp:            << ncclAllReduce(send_buf, recv_buf, count, nccl_type<T>::value(),
tests/cpp/integration/collective_nccl_impl.hpp:                             ncclSum, comm, stream);
tests/cpp/integration/collective_nccl_impl.hpp:        KUNGFU_CHECK(cuda_checker) << cudaStreamSynchronize(stream);
tests/cpp/integration/collective_nccl_impl.hpp:        KUNGFU_CHECK(cuda_checker) << cudaStreamDestroy(stream);
tests/cpp/integration/collective_nccl_impl.hpp:        std::cerr << "nccl_collective::all_reduce<async> is not implemted"
tests/go/cmd/kungfu-bench-allreduce/kungfu-bench-allreduce.go:	"github.com/lsds/KungFu/srcs/go/nccl"
tests/go/cmd/kungfu-bench-allreduce/kungfu-bench-allreduce.go:	randomNcclFailure = flag.Bool("rand-nccl-failure", false, "")
tests/go/cmd/kungfu-bench-allreduce/kungfu-bench-allreduce.go:	if *randomNcclFailure {
tests/go/cmd/kungfu-bench-allreduce/kungfu-bench-allreduce.go:		nccl.RandomFailure()
README.md:KungFu provides ``kungfu-run`` to launch a KungFu process on a multi-GPU server.
README.md:You can use KungFu with Docker. Check out the docker files for [GPU](docker/Dockerfile.tf-gpu) and [CPU](docker/Dockerfile.tf-cpu) machines.
README.md:You can run this example on two machines (assuming each with 8 GPUs) using the below command (NOTE: this command must be called on each machine):
README.md:# Assume NUM_GPU_SLOTS=8, NUM_GPUS=16
README.md:kungfu-run -np $NUM_GPUS \
README.md:    -H 192.168.0.1:$NUM_GPU_SLOTS,192.168.0.2:$NUM_GPU_SLOTS -nic eth0 \
README.md:All these tests use a per-GPU batch size as 64 and [hyper-parameters](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks#getting-started)
README.md:We benchmark KungFu in a cluster that has 16 V100 GPUs hosted by 2 DGX-1 machines.
README.md:This batch size is evenly shared by the 16 GPUs.
srcs/python/kungfu/python/__init__.py:    has_nccl = _call_method(_python_lib, 'kungfu_python_init_nccl')
srcs/python/kungfu/python/__init__.py:    return _python_lib, has_nccl
srcs/python/kungfu/python/__init__.py:_has_nccl = None
srcs/python/kungfu/python/__init__.py:_python_lib, _has_nccl = _load_and_init_python_lib()
srcs/python/kungfu/python/__init__.py:    global _has_nccl
srcs/python/kungfu/python/__init__.py:    _has_nccl = _call_method(_python_lib, 'kungfu_python_init_nccl')
srcs/python/kungfu/python/__init__.py:    if _has_nccl:
srcs/python/kungfu/python/__init__.py:        _call_method(_python_lib, 'kungfu_python_finialize_nccl')
srcs/python/kungfu/python/__init__.py:def _get_cuda_index():
srcs/python/kungfu/python/__init__.py:    return _python_lib.kungfu_get_cuda_index()
srcs/python/kungfu/python/__init__.py:def show_cuda_version():
srcs/python/kungfu/python/__init__.py:    if _has_nccl:
srcs/python/kungfu/python/__init__.py:        _call_method(_python_lib, 'kungfu_show_cuda_version', force=True)
srcs/python/kungfu/python/__init__.py:        print('NCCL is NOT enabled')
srcs/python/kungfu/python/__init__.py:def show_nccl_version():
srcs/python/kungfu/python/__init__.py:    if _has_nccl:
srcs/python/kungfu/python/__init__.py:        _call_method(_python_lib, 'kungfu_show_nccl_version', force=True)
srcs/python/kungfu/python/__init__.py:        print('NCCL is NOT enabled')
srcs/python/kungfu/torch/ops/clib.py:if hasattr(ops, 'all_reduce_cuda'):
srcs/python/kungfu/torch/ops/clib.py:    all_reduce_op_map['torch.cuda.FloatTensor'] = ops.all_reduce_cuda
srcs/python/kungfu/torch/ops/clib.py:if hasattr(ops, 'all_reduce_cuda_async'):
srcs/python/kungfu/torch/ops/clib.py:        'torch.cuda.FloatTensor'] = ops.all_reduce_cuda_async
srcs/python/kungfu/torch/ops/clib.py:if hasattr(ops, 'broadcast_cuda_async'):
srcs/python/kungfu/torch/ops/clib.py:        'torch.cuda.FloatTensor'] = ops.broadcast_cuda_async
srcs/python/kungfu/torch/ops/clib.py:if hasattr(ops, 'all_gather_cuda'):
srcs/python/kungfu/torch/ops/clib.py:        'torch.cuda.FloatTensor'] = ops.all_gather_cuda
srcs/python/kungfu/torch/optimizers/sync_sgd.py:                if '.cuda.' in p.grad.type():
srcs/python/kungfu/torch/__init__.py:from kungfu.python import (_get_cuda_index, current_cluster_size,
srcs/python/kungfu/torch/__init__.py:get_cuda_index = _get_cuda_index
srcs/python/kungfu/torch/__init__.py:def nccl_built():
srcs/python/kungfu/info/__main__.py:from kungfu.python import show_cuda_version, show_nccl_version
srcs/python/kungfu/info/__main__.py:    show_cuda_version()
srcs/python/kungfu/info/__main__.py:    show_nccl_version()
srcs/python/kungfu/tensorflow/ops/collective.py:def _nccl_all_reduce(t):
srcs/python/kungfu/tensorflow/ops/collective.py:    return _op_lib.kungfu_nccl_all_reduce(t)
srcs/python/kungfu/tensorflow/ops/collective.py:def _scheduled_nccl_all_reduce(t, op_name=None):
srcs/python/kungfu/tensorflow/ops/collective.py:    return _op_lib.kungfu_scheduled_nccl_all_reduce(t, op_name=op_name)
srcs/python/kungfu/tensorflow/ops/collective.py:def _scheduled_hierarchical_nccl_all_reduce(t, op_names):
srcs/python/kungfu/tensorflow/ops/collective.py:    return _op_lib.kungfu_scheduled_hierarchical_nccl_all_reduce(
srcs/python/kungfu/tensorflow/ops/collective.py:def _start_nccl_scheduler(*args, **kwargs):
srcs/python/kungfu/tensorflow/ops/collective.py:    if hasattr(_op_lib, 'kungfu_start_nccl_scheduler'):
srcs/python/kungfu/tensorflow/ops/collective.py:        return _op_lib.kungfu_start_nccl_scheduler(*args, **kwargs)
srcs/python/kungfu/tensorflow/ops/collective.py:            "KungFu is not installed with NCCL. Please reinstall with KUNGFU_ENABLE_NCCL=1"
srcs/python/kungfu/tensorflow/ops/collective.py:def group_nccl_all_reduce(ts):
srcs/python/kungfu/tensorflow/ops/collective.py:    """Create a list of all_reduce operators for given tensor list, using NCCL."""
srcs/python/kungfu/tensorflow/ops/collective.py:        return map_maybe(_nccl_all_reduce, ts)  # exactly one of ts is not None
srcs/python/kungfu/tensorflow/ops/collective.py:                _start_nccl_scheduler(names, scope='global'),
srcs/python/kungfu/tensorflow/ops/collective.py:            return map_maybe(_scheduled_nccl_all_reduce, ts)
srcs/python/kungfu/tensorflow/ops/collective.py:def group_hierarchical_nccl_all_reduce(ts):
srcs/python/kungfu/tensorflow/ops/collective.py:        return _scheduled_hierarchical_nccl_all_reduce(
srcs/python/kungfu/tensorflow/ops/collective.py:            _start_nccl_scheduler(all_op_names, scope='local'),
srcs/python/kungfu/tensorflow/ops/adapt.py:    if hasattr(_op_lib, 'kungfu_reset_nccl_helper'):
srcs/python/kungfu/tensorflow/ops/adapt.py:        return _op_lib.kungfu_reset_nccl_helper(changed, detached)
srcs/python/kungfu/tensorflow/ops/adapt.py:    if hasattr(_op_lib, 'kungfu_reset_nccl_helper'):
srcs/python/kungfu/tensorflow/ops/adapt.py:        changed, detached = _op_lib.kungfu_reset_nccl_helper(changed, detached)
srcs/python/kungfu/tensorflow/ops/__init__.py:                         group_all_reduce, group_nccl_all_reduce,
srcs/python/kungfu/tensorflow/v1/benchmarks/__main__.py:    kungfu-run -q -np 4 python3 -m kungfu.tensorflow.v1.benchmarks --method NCCL
srcs/python/kungfu/tensorflow/v1/benchmarks/__main__.py:    kungfu-run -q -np 4 python3 -m kungfu.tensorflow.v1.benchmarks --method NCCL+CPU
srcs/python/kungfu/tensorflow/v1/benchmarks/__main__.py:from kungfu.python import _get_cuda_index
srcs/python/kungfu/tensorflow/v1/benchmarks/__main__.py:                                   group_all_reduce, group_nccl_all_reduce)
srcs/python/kungfu/tensorflow/v1/benchmarks/__main__.py:from kungfu.tensorflow.ops.collective import group_hierarchical_nccl_all_reduce
srcs/python/kungfu/tensorflow/v1/benchmarks/__main__.py:    'NCCL': group_nccl_all_reduce,
srcs/python/kungfu/tensorflow/v1/benchmarks/__main__.py:    'NCCL+CPU': group_hierarchical_nccl_all_reduce,
srcs/python/kungfu/tensorflow/v1/benchmarks/__main__.py:    config.gpu_options.allow_growth = True
srcs/python/kungfu/tensorflow/v1/benchmarks/__main__.py:        config.gpu_options.visible_device_list = str(hvd.local_rank())
srcs/python/kungfu/tensorflow/v1/benchmarks/__main__.py:        config.gpu_options.visible_device_list = str(_get_cuda_index())
srcs/python/kungfu/tensorflow/v1/benchmarks/__main__.py:                   help='CPU | NCCL | HOROVOD')
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:                                   group_nccl_all_reduce, monitored_all_reduce,
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:    group_all_reduce, group_hierarchical_nccl_all_reduce,
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:    group_nccl_all_reduce)
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:                            nccl=False,
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:                            nccl_fusion=False,
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:                            hierarchical_nccl=False,
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:        - nccl {bool} -- using NCCL to average gradients. (default: {False})
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:        - nccl_fusion {bool} -- fusing all gradients to amortise NCCL operation launch cost. (default: {True})
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:    sync_sgd_algo = _SynchronousSGD(nccl=nccl,
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:                                    nccl_fusion=nccl_fusion,
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:                                    hierarchical_nccl=hierarchical_nccl,
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:                 nccl=False,
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:                 nccl_fusion=True,
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:                 hierarchical_nccl=False,
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:        self._nccl = nccl
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:        self._nccl_fusion = nccl_fusion
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:        if self._nccl:
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:            if hierarchical_nccl:
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:                self._group_all_reduce_fn = group_hierarchical_nccl_all_reduce
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:                self._group_all_reduce_fn = group_nccl_all_reduce
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:        if self._nccl:
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:            # FIXME: We have a limitation that KungFu schedules NCCL operations
srcs/python/kungfu/tensorflow/optimizers/sync_sgd.py:            if self._nccl_fusion:
srcs/python/kungfu/tensorflow/optimizers/sma_sgd.py:    .. [SMA] CrossBow: Scaling Deep Learning with Small Batch Sizes on Multi-GPU Servers, VLDB 2019, `SMA Paper <http://www.vldb.org/pvldb/vol12/p1399-koliousis.pdf>`_
srcs/python/kungfu/tensorflow/optimizers/async_sgd.py:        - fuse_requests {bool} -- Fusing requests to amortise communication cost at the cost of extra GPU memory and cycles. (default: {True})
srcs/cmake/nccl.cmake:FIND_PACKAGE(CUDA REQUIRED)
srcs/cmake/nccl.cmake:FUNCTION(USE_NCCL target)
srcs/cmake/nccl.cmake:    IF(NCCL_HOME)
srcs/cmake/nccl.cmake:        TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${NCCL_HOME}/include)
srcs/cmake/nccl.cmake:        TARGET_LINK_LIBRARIES(${target} ${NCCL_HOME}/lib/libnccl.so)
srcs/cmake/nccl.cmake:        TARGET_LINK_LIBRARIES(${target} nccl)
srcs/cmake/nccl.cmake:    TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${CUDA_INCLUDE_DIRS})
srcs/cmake/nccl.cmake:    TARGET_LINK_LIBRARIES(${target} ${CUDA_LIBRARIES})
srcs/cmake/tests.cmake:IF(KUNGFU_ENABLE_NCCL)
srcs/cmake/tests.cmake:    ADD_TEST_BIN(fake-nccl-trainer
srcs/cmake/tests.cmake:                 ${KUNGFU_TESTS_DIR}/integration/fake_nccl_trainer.cpp)
srcs/cmake/tests.cmake:    USE_NCCL(fake-nccl-trainer)
srcs/cmake/tests.cmake:    USE_MPI(fake-nccl-trainer) # FIXME: don't use MPI for bootsrtap
srcs/cmake/tests.cmake:    ADD_TEST_BIN(test-nccl-helper
srcs/cmake/tests.cmake:                 ${KUNGFU_TESTS_DIR}/integration/test_nccl_helper.cpp)
srcs/cmake/tests.cmake:    USE_NCCL(test-nccl-helper)
srcs/cmake/tests.cmake:    TARGET_LINK_LIBRARIES(test-nccl-helper kungfu_nccl)
srcs/cmake/tests.cmake:    TARGET_LINK_LIBRARIES(test-nccl-helper kungfu)
srcs/cmake/tests.cmake:    IF(KUNGFU_ENABLE_NCCL)
srcs/cmake/tests.cmake:        USE_NCCL(fake-tf-agent)
srcs/cpp/include/kungfu/python/c_api.h:extern void kungfu_python_init_nccl();
srcs/cpp/include/kungfu/python/c_api.h:extern void kungfu_python_finialize_nccl();
srcs/cpp/include/kungfu/python/c_api.h:extern int kungfu_get_cuda_index();
srcs/cpp/include/kungfu/python/c_api.h:class NCCLHelper;
srcs/cpp/include/kungfu/torch/common.hpp:class CudaStream;
srcs/cpp/include/kungfu/torch/common.hpp:class TorchCudaHelper
srcs/cpp/include/kungfu/torch/common.hpp:    std::unique_ptr<CudaStream> up_stream_;
srcs/cpp/include/kungfu/torch/common.hpp:    std::unique_ptr<CudaStream> down_stream_;
srcs/cpp/include/kungfu/torch/common.hpp:    TorchCudaHelper();
srcs/cpp/include/kungfu/torch/common.hpp:    void from_cuda(void *buffer, const torch::Tensor &t);
srcs/cpp/include/kungfu/torch/common.hpp:    void from_cuda(void *dst, const void *src, size_t size);
srcs/cpp/include/kungfu/torch/common.hpp:    void to_cuda(torch::Tensor &t, const void *buffer);
srcs/cpp/include/kungfu/torch/common.hpp:    void to_cuda(void *dst, const void *src, size_t size);
srcs/cpp/include/kungfu/torch/common.hpp:extern TorchCudaHelper _torch_cuda_helper;
srcs/cpp/include/kungfu/torch/common.hpp:    Torch_Cuda_Float,
srcs/cpp/include/kungfu/nccl/helper.hpp:#include <kungfu/cuda/stream.hpp>
srcs/cpp/include/kungfu/nccl/helper.hpp:#include <kungfu/nccl/common.hpp>
srcs/cpp/include/kungfu/nccl/helper.hpp:#include <kungfu/nccl/controller.hpp>
srcs/cpp/include/kungfu/nccl/helper.hpp:#include <kungfu/nccl/scheduler.hpp>
srcs/cpp/include/kungfu/nccl/helper.hpp:// NCCLHelper is a singleton class that contains NCCL related global variables
srcs/cpp/include/kungfu/nccl/helper.hpp:class NCCLHelper
srcs/cpp/include/kungfu/nccl/helper.hpp:    std::map<KungFu_NCCLScope, std::unique_ptr<NCCLScheduler>> schedulers_;
srcs/cpp/include/kungfu/nccl/helper.hpp:    std::map<KungFu_NCCLScope, std::unique_ptr<NCCLController>> controllers_;
srcs/cpp/include/kungfu/nccl/helper.hpp:    std::map<std::string, std::unique_ptr<NCCLController>> group_controllers_;
srcs/cpp/include/kungfu/nccl/helper.hpp:    NCCLHelper();
srcs/cpp/include/kungfu/nccl/helper.hpp:    NCCLScheduler *EnsureScheduler(const KungFu_NCCLScope scope);
srcs/cpp/include/kungfu/nccl/helper.hpp:    NCCLController *EnsureController(const KungFu_NCCLScope scope);
srcs/cpp/include/kungfu/nccl/helper.hpp:    NCCLController *EnsureGroupController(std::vector<int32_t> topology);
srcs/cpp/include/kungfu/nccl/helper.hpp:    static std::unique_ptr<NCCLHelper> &GetDefault(bool reinit = false);
srcs/cpp/include/kungfu/nccl/gpu_collective.hpp:extern void kungfu_show_cuda_version();
srcs/cpp/include/kungfu/nccl/gpu_collective.hpp:extern void kungfu_show_nccl_version();
srcs/cpp/include/kungfu/nccl/gpu_collective.hpp:// gpu_collective wraps NCCL APIs for internal use.
srcs/cpp/include/kungfu/nccl/gpu_collective.hpp:// User should use NCCLController
srcs/cpp/include/kungfu/nccl/gpu_collective.hpp:class gpu_collective
srcs/cpp/include/kungfu/nccl/gpu_collective.hpp:    virtual ~gpu_collective() = default;
srcs/cpp/include/kungfu/nccl/gpu_collective.hpp:    static gpu_collective *new_global(kungfu::Peer &);
srcs/cpp/include/kungfu/nccl/gpu_collective.hpp:    static gpu_collective *new_local(kungfu::Peer &);
srcs/cpp/include/kungfu/nccl/gpu_collective.hpp:    static gpu_collective *new_group(kungfu::Peer &,
srcs/cpp/include/kungfu/nccl/scheduler.hpp:#include <kungfu/nccl/common.hpp>
srcs/cpp/include/kungfu/nccl/scheduler.hpp:class NCCLThread
srcs/cpp/include/kungfu/nccl/scheduler.hpp:    NCCLThread();
srcs/cpp/include/kungfu/nccl/scheduler.hpp:    ~NCCLThread();
srcs/cpp/include/kungfu/nccl/scheduler.hpp:    NCCLThread *nccl_thread_;
srcs/cpp/include/kungfu/nccl/scheduler.hpp:                   const std::vector<int32_t> &order, NCCLThread *nccl_thread);
srcs/cpp/include/kungfu/nccl/scheduler.hpp:class NCCLScheduler
srcs/cpp/include/kungfu/nccl/scheduler.hpp:    const KungFu_NCCLScope scope_;
srcs/cpp/include/kungfu/nccl/scheduler.hpp:    std::unique_ptr<NCCLThread> nccl_thread_;
srcs/cpp/include/kungfu/nccl/scheduler.hpp:    NCCLScheduler(const KungFu_NCCLScope scope, const bool auto_order = true);
srcs/cpp/include/kungfu/nccl/scheduler.hpp:    // Run a task in the dedicated NCCL thread
srcs/cpp/include/kungfu/nccl/controller.hpp:#include <kungfu/nccl/common.hpp>
srcs/cpp/include/kungfu/nccl/controller.hpp:#include <kungfu/nccl/gpu_collective.hpp>
srcs/cpp/include/kungfu/nccl/controller.hpp:void CrossAllReduceGpu(Peer *peer, const Workspace &w, KungFu_Op op,
srcs/cpp/include/kungfu/nccl/controller.hpp:// NCCLController exposes user-facing APIs
srcs/cpp/include/kungfu/nccl/controller.hpp:class NCCLController
srcs/cpp/include/kungfu/nccl/controller.hpp:    KungFu_NCCLScope scope_;
srcs/cpp/include/kungfu/nccl/controller.hpp:    // only used when scope_ == KungFu_NCCL_GROUP
srcs/cpp/include/kungfu/nccl/controller.hpp:    std::unique_ptr<gpu_collective> gpu_collective_;
srcs/cpp/include/kungfu/nccl/controller.hpp:    gpu_collective *new_gpu_collective(Peer *peer);
srcs/cpp/include/kungfu/nccl/controller.hpp:    NCCLController(const KungFu_NCCLScope scope);
srcs/cpp/include/kungfu/nccl/controller.hpp:    NCCLController(std::vector<int32_t> topology);
srcs/cpp/include/kungfu/nccl/common.hpp:enum KungFu_NCCLScope {
srcs/cpp/include/kungfu/nccl/common.hpp:    KungFu_NCCL_GLOBAL,
srcs/cpp/include/kungfu/nccl/common.hpp:    KungFu_NCCL_LOCAL,
srcs/cpp/include/kungfu/nccl/common.hpp:    KungFu_NCCL_GROUP,
srcs/cpp/include/kungfu/nccl/common.hpp:extern const std::map<std::string, KungFu_NCCLScope> _nccl_scopes;
srcs/cpp/include/kungfu/cuda/stream.hpp:#include <cuda_runtime.h>
srcs/cpp/include/kungfu/cuda/stream.hpp:struct show_cuda_error {
srcs/cpp/include/kungfu/cuda/stream.hpp:    std::string operator()(cudaError_t err) const
srcs/cpp/include/kungfu/cuda/stream.hpp:               cudaGetErrorString(err);
srcs/cpp/include/kungfu/cuda/stream.hpp:using cuda_checker = error_checker<cudaError_t, cudaSuccess, show_cuda_error>;
srcs/cpp/include/kungfu/cuda/stream.hpp:// CudaStream wraps cudaStream_t
srcs/cpp/include/kungfu/cuda/stream.hpp:class CudaStream
srcs/cpp/include/kungfu/cuda/stream.hpp:    cudaStream_t stream_;
srcs/cpp/include/kungfu/cuda/stream.hpp:    CudaStream();
srcs/cpp/include/kungfu/cuda/stream.hpp:    ~CudaStream();
srcs/cpp/include/kungfu/cuda/stream.hpp:    operator cudaStream_t() const { return stream_; }
srcs/cpp/include/kungfu/cuda/stream.hpp:                const cudaMemcpyKind dir);
srcs/cpp/include/kungfu/cuda/stream.hpp:    std::queue<std::unique_ptr<CudaStream>> queue_;
srcs/cpp/include/kungfu/cuda/stream.hpp:    std::unique_ptr<CudaStream> Get();
srcs/cpp/include/kungfu/cuda/stream.hpp:    void Put(std::unique_ptr<CudaStream> stream);
srcs/cpp/include/kungfu/tensorflow/ops.h:#include <kungfu/nccl/common.hpp>
srcs/cpp/src/python/cuda.cpp:std::vector<int> parse_cuda_visible_devices(const std::string &val)
srcs/cpp/src/python/cuda.cpp:int kungfu_get_cuda_index()
srcs/cpp/src/python/cuda.cpp:        const char *ptr = std::getenv("KUNGFU_CUDA_VISIBLE_DEVICES");
srcs/cpp/src/python/cuda.cpp:        const char *ptr = std::getenv("CUDA_VISIBLE_DEVICES");
srcs/cpp/src/python/cuda.cpp:            const auto devs = parse_cuda_visible_devices(ptr);
srcs/cpp/src/python/init_nccl.cpp:#include <kungfu/nccl/helper.hpp>
srcs/cpp/src/python/init_nccl.cpp:void kungfu_python_init_nccl() { kungfu::NCCLHelper::GetDefault(true); }
srcs/cpp/src/python/init_nccl.cpp:void kungfu_python_finialize_nccl()
srcs/cpp/src/python/init_nccl.cpp:    auto &p = kungfu::NCCLHelper::GetDefault();
srcs/cpp/src/torch/ops/cuda/helper.cpp:#include <kungfu/cuda/stream.hpp>
srcs/cpp/src/torch/ops/cuda/helper.cpp:TorchCudaHelper::TorchCudaHelper()
srcs/cpp/src/torch/ops/cuda/helper.cpp:    : up_stream_(new CudaStream), down_stream_(new CudaStream)
srcs/cpp/src/torch/ops/cuda/helper.cpp:HandleManager<int> &TorchCudaHelper::handle_manager() { return hm_; }
srcs/cpp/src/torch/ops/cuda/helper.cpp:void TorchCudaHelper::from_cuda(void *buffer, const torch::Tensor &t)
srcs/cpp/src/torch/ops/cuda/helper.cpp:                         cudaMemcpyDeviceToHost);
srcs/cpp/src/torch/ops/cuda/helper.cpp:void TorchCudaHelper::from_cuda(void *dst, const void *src, size_t size)
srcs/cpp/src/torch/ops/cuda/helper.cpp:    down_stream_->memcpy(dst, src, size, cudaMemcpyDeviceToHost);
srcs/cpp/src/torch/ops/cuda/helper.cpp:void TorchCudaHelper::to_cuda(torch::Tensor &t, const void *buffer)
srcs/cpp/src/torch/ops/cuda/helper.cpp:                       cudaMemcpyHostToDevice);
srcs/cpp/src/torch/ops/cuda/helper.cpp:void TorchCudaHelper::to_cuda(void *dst, const void *src, size_t size)
srcs/cpp/src/torch/ops/cuda/helper.cpp:    up_stream_->memcpy(dst, src, size, cudaMemcpyHostToDevice);
srcs/cpp/src/torch/ops/cuda/helper.cpp:    _torch_cuda_helper.handle_manager().wait(handle);
srcs/cpp/src/torch/ops/cuda/helper.cpp:    _torch_cuda_helper.handle_manager().wait_all(handles);
srcs/cpp/src/torch/ops/cuda/helper.cpp:TorchCudaHelper _torch_cuda_helper;
srcs/cpp/src/torch/ops/cuda/collective.cpp:    case Torch_Cuda_Float:
srcs/cpp/src/torch/ops/cuda/collective.cpp:void all_reduce_cuda(torch::Tensor input, torch::Tensor output,
srcs/cpp/src/torch/ops/cuda/collective.cpp:    _torch_cuda_helper.from_cuda(buffer.data(), input);
srcs/cpp/src/torch/ops/cuda/collective.cpp:    _torch_cuda_helper.to_cuda(output, buffer.data());
srcs/cpp/src/torch/ops/cuda/collective.cpp:int all_reduce_cuda_async(torch::Tensor input, torch::Tensor output,
srcs/cpp/src/torch/ops/cuda/collective.cpp:    const int handle = _torch_cuda_helper.handle_manager().create();
srcs/cpp/src/torch/ops/cuda/collective.cpp:        _torch_cuda_helper.from_cuda(buffer, px, size);
srcs/cpp/src/torch/ops/cuda/collective.cpp:                _torch_cuda_helper.to_cuda(py, buffer, size);
srcs/cpp/src/torch/ops/cuda/collective.cpp:                _torch_cuda_helper.handle_manager().done(handle);
srcs/cpp/src/torch/ops/cuda/collective.cpp:int broadcast_cuda_async(torch::Tensor input, torch::Tensor output,
srcs/cpp/src/torch/ops/cuda/collective.cpp:    const int handle = _torch_cuda_helper.handle_manager().create();
srcs/cpp/src/torch/ops/cuda/collective.cpp:        _torch_cuda_helper.from_cuda(buffer, px, size);
srcs/cpp/src/torch/ops/cuda/collective.cpp:                _torch_cuda_helper.to_cuda(py, buffer, size);
srcs/cpp/src/torch/ops/cuda/collective.cpp:                _torch_cuda_helper.handle_manager().done(handle);
srcs/cpp/src/torch/ops/cuda/collective.cpp:void all_gather_cuda(torch::Tensor input, torch::Tensor output,
srcs/cpp/src/torch/ops/cuda/collective.cpp:    _torch_cuda_helper.from_cuda(send_buffer.data(), input);
srcs/cpp/src/torch/ops/cuda/collective.cpp:    _torch_cuda_helper.to_cuda(output, receive_buffer.data());
srcs/cpp/src/torch/common.cpp:    {"torch.cuda.FloatTensor", Torch_Cuda_Float},
srcs/cpp/src/torch/module_cuda.cpp:void all_reduce_cuda(torch::Tensor input, torch::Tensor output,
srcs/cpp/src/torch/module_cuda.cpp:int all_reduce_cuda_async(torch::Tensor input, torch::Tensor output,
srcs/cpp/src/torch/module_cuda.cpp:int broadcast_cuda_async(torch::Tensor input, torch::Tensor output,
srcs/cpp/src/torch/module_cuda.cpp:void all_gather_cuda(torch::Tensor input, torch::Tensor output,
srcs/cpp/src/torch/module_cuda.cpp:    m.def("all_reduce_cuda", &kungfu::all_reduce_cuda);
srcs/cpp/src/torch/module_cuda.cpp:    m.def("all_reduce_cuda_async", &kungfu::all_reduce_cuda_async);
srcs/cpp/src/torch/module_cuda.cpp:    m.def("broadcast_cuda_async", &kungfu::broadcast_cuda_async);
srcs/cpp/src/torch/module_cuda.cpp:    m.def("all_gather_cuda", &kungfu::all_gather_cuda);
srcs/cpp/src/nccl/controller.cpp:#include <kungfu/cuda/stream.hpp>
srcs/cpp/src/nccl/controller.cpp:#include <kungfu/nccl/controller.hpp>
srcs/cpp/src/nccl/controller.cpp:#include <kungfu/nccl/helper.hpp>
srcs/cpp/src/nccl/controller.cpp:void CrossAllReduceGpu(Peer *peer, const Workspace &w, KungFu_Op op,
srcs/cpp/src/nccl/controller.cpp:            CudaStream stream;
srcs/cpp/src/nccl/controller.cpp:                          cudaMemcpyDeviceToDevice);
srcs/cpp/src/nccl/controller.cpp:        CudaStream stream;
srcs/cpp/src/nccl/controller.cpp:        stream.memcpy(buffer, w.sendbuf, data_size, cudaMemcpyDeviceToHost);
srcs/cpp/src/nccl/controller.cpp:                                 CudaStream stream;
srcs/cpp/src/nccl/controller.cpp:                                               cudaMemcpyHostToDevice);
srcs/cpp/src/nccl/controller.cpp:NCCLController::NCCLController(const KungFu_NCCLScope scope) : scope_(scope)
srcs/cpp/src/nccl/controller.cpp:    if (scope != KungFu_NCCL_LOCAL && scope != KungFu_NCCL_GLOBAL) {
srcs/cpp/src/nccl/controller.cpp:NCCLController::NCCLController(std::vector<int32_t> topology)
srcs/cpp/src/nccl/controller.cpp:    : scope_(KungFu_NCCL_GROUP), topology_(std::move(topology))
srcs/cpp/src/nccl/controller.cpp:gpu_collective *NCCLController::new_gpu_collective(Peer *peer)
srcs/cpp/src/nccl/controller.cpp:    case KungFu_NCCL_LOCAL:
srcs/cpp/src/nccl/controller.cpp:        return gpu_collective::new_local(*peer);
srcs/cpp/src/nccl/controller.cpp:    case KungFu_NCCL_GLOBAL:
srcs/cpp/src/nccl/controller.cpp:        return gpu_collective::new_global(*peer);
srcs/cpp/src/nccl/controller.cpp:        return gpu_collective::new_group(*peer, topology_);
srcs/cpp/src/nccl/controller.cpp:void NCCLController::InitOnce(Peer *peer)
srcs/cpp/src/nccl/controller.cpp:    if (gpu_collective_.get() == nullptr) {
srcs/cpp/src/nccl/controller.cpp:        gpu_collective_.reset(new_gpu_collective(peer));
srcs/cpp/src/nccl/controller.cpp:void NCCLController::ReInit(Peer *peer)
srcs/cpp/src/nccl/controller.cpp:    gpu_collective_.reset(new_gpu_collective(peer));
srcs/cpp/src/nccl/controller.cpp:int NCCLController::Reduce(const Workspace &w, KungFu_Op op, DoneCallback done)
srcs/cpp/src/nccl/controller.cpp:    gpu_collective_->reduce(w.sendbuf, w.recvbuf, w.count, w.dtype);
srcs/cpp/src/nccl/controller.cpp:int NCCLController::Broadcast(const Workspace &w, DoneCallback done)
srcs/cpp/src/nccl/controller.cpp:    gpu_collective_->broadcast(w.sendbuf, w.recvbuf, w.count, w.dtype);
srcs/cpp/src/nccl/controller.cpp:int NCCLController::Broadcast(const Workspace &w, void *stream_ptr)
srcs/cpp/src/nccl/controller.cpp:    gpu_collective_->broadcast(w.sendbuf, w.recvbuf, w.count, w.dtype,
srcs/cpp/src/nccl/controller.cpp:int NCCLController::AllReduce(const Workspace &w, KungFu_Op op,
srcs/cpp/src/nccl/controller.cpp:    gpu_collective_->all_reduce(w.sendbuf, w.recvbuf, w.count, w.dtype, op);
srcs/cpp/src/nccl/controller.cpp:int NCCLController::AllReduce(const Workspace &w, KungFu_Op op,
srcs/cpp/src/nccl/controller.cpp:    gpu_collective_->all_reduce(w.sendbuf, w.recvbuf, w.count, w.dtype, op,
srcs/cpp/src/nccl/controller.cpp:int NCCLController::AllGather(const Workspace &w, DoneCallback done)
srcs/cpp/src/nccl/controller.cpp:    gpu_collective_->all_gather(w.sendbuf, w.recvbuf, w.count, w.dtype);
srcs/cpp/src/nccl/controller.cpp:int NCCLController::AllGather(const Workspace &w, void *stream_ptr)
srcs/cpp/src/nccl/controller.cpp:    gpu_collective_->all_gather(w.sendbuf, w.recvbuf, w.count, w.dtype,
srcs/cpp/src/nccl/helper.cpp:#include <kungfu/nccl/helper.hpp>
srcs/cpp/src/nccl/helper.cpp:NCCLHelper::NCCLHelper() {}
srcs/cpp/src/nccl/helper.cpp:NCCLController *NCCLHelper::EnsureController(const KungFu_NCCLScope scope)
srcs/cpp/src/nccl/helper.cpp:    if (ptr.get() == nullptr) { ptr.reset(new NCCLController(scope)); }
srcs/cpp/src/nccl/helper.cpp:NCCLScheduler *NCCLHelper::EnsureScheduler(const KungFu_NCCLScope scope)
srcs/cpp/src/nccl/helper.cpp:    if (ptr.get() == nullptr) { ptr.reset(new NCCLScheduler(scope)); }
srcs/cpp/src/nccl/helper.cpp:NCCLController *NCCLHelper::EnsureGroupController(std::vector<int32_t> topology)
srcs/cpp/src/nccl/helper.cpp:        ptr.reset(new NCCLController(std::move(topology)));
srcs/cpp/src/nccl/helper.cpp:std::unique_ptr<NCCLHelper> &NCCLHelper::GetDefault(bool reinit)
srcs/cpp/src/nccl/helper.cpp:    static std::unique_ptr<NCCLHelper> instance;
srcs/cpp/src/nccl/helper.cpp:    if (reinit) { instance.reset(new NCCLHelper); }
srcs/cpp/src/nccl/common.cpp:#include <kungfu/nccl/common.hpp>
srcs/cpp/src/nccl/common.cpp:const std::map<std::string, KungFu_NCCLScope> _nccl_scopes({
srcs/cpp/src/nccl/common.cpp:    {"global", KungFu_NCCL_GLOBAL},
srcs/cpp/src/nccl/common.cpp:    {"local", KungFu_NCCL_LOCAL},
srcs/cpp/src/nccl/scheduler.cpp:#include <kungfu/nccl/scheduler.hpp>
srcs/cpp/src/nccl/scheduler.cpp:NCCLThread::NCCLThread()
srcs/cpp/src/nccl/scheduler.cpp:NCCLThread::~NCCLThread()
srcs/cpp/src/nccl/scheduler.cpp:void NCCLThread::Put(std::function<void()> task) { queue_.put(task); }
srcs/cpp/src/nccl/scheduler.cpp:void NCCLThread::Do(std::function<void()> task)
srcs/cpp/src/nccl/scheduler.cpp:                               NCCLThread *nccl_thread)
srcs/cpp/src/nccl/scheduler.cpp:      is_started_(size_), tasks_(size_), nccl_thread_(nccl_thread)
srcs/cpp/src/nccl/scheduler.cpp:            nccl_thread_->Put(tasks_[i]);
srcs/cpp/src/nccl/scheduler.cpp:        nccl_thread_->Do([] {});  // do an empty task to wait all previous tasks
srcs/cpp/src/nccl/scheduler.cpp:NCCLScheduler::NCCLScheduler(const KungFu_NCCLScope scope,
srcs/cpp/src/nccl/scheduler.cpp:    : name_("NCCLScheduler_" + std::to_string(int(scope))),
srcs/cpp/src/nccl/scheduler.cpp:      nccl_thread_(new NCCLThread)
srcs/cpp/src/nccl/scheduler.cpp:void NCCLScheduler::ResetOrder(int n)
srcs/cpp/src/nccl/scheduler.cpp:void NCCLScheduler::Reset(const std::vector<std::string> &names, Peer *peer)
srcs/cpp/src/nccl/scheduler.cpp:                if (scope_ == KungFu_NCCL_LOCAL) {
srcs/cpp/src/nccl/scheduler.cpp:    executor_.reset(new LinearExecutor(names, order_, nccl_thread_.get()));
srcs/cpp/src/nccl/scheduler.cpp:void NCCLScheduler::Start(const std::string &name, const DoneCallback &task)
srcs/cpp/src/nccl/scheduler.cpp:void NCCLScheduler::Do(std::function<void()> task)
srcs/cpp/src/nccl/scheduler.cpp:    nccl_thread_->Do(std::move(task));
srcs/cpp/src/nccl/gpu_collective.cpp:#include <kungfu/cuda/stream.hpp>
srcs/cpp/src/nccl/gpu_collective.cpp:#include <kungfu/nccl/gpu_collective.hpp>
srcs/cpp/src/nccl/gpu_collective.cpp:#include <nccl.h>
srcs/cpp/src/nccl/gpu_collective.cpp:struct show_nccl_error {
srcs/cpp/src/nccl/gpu_collective.cpp:    std::string operator()(ncclResult_t err) const
srcs/cpp/src/nccl/gpu_collective.cpp:        std::string msg = ncclGetErrorString(err);
srcs/cpp/src/nccl/gpu_collective.cpp:using nccl_checker = error_checker<ncclResult_t, ncclSuccess, show_nccl_error>;
srcs/cpp/src/nccl/gpu_collective.cpp:void kungfu_show_cuda_version()
srcs/cpp/src/nccl/gpu_collective.cpp:    KUNGFU_CHECK(kungfu::cuda_checker) << cudaDriverGetVersion(&driverVersion);
srcs/cpp/src/nccl/gpu_collective.cpp:    printf("CUDA Driver Veresion: %d\n", driverVersion);
srcs/cpp/src/nccl/gpu_collective.cpp:    KUNGFU_CHECK(kungfu::cuda_checker)
srcs/cpp/src/nccl/gpu_collective.cpp:        << cudaRuntimeGetVersion(&runtimeVersion);
srcs/cpp/src/nccl/gpu_collective.cpp:    printf("CUDA Runtime Veresion: %d\n", runtimeVersion);
srcs/cpp/src/nccl/gpu_collective.cpp:void kungfu_show_nccl_version()
srcs/cpp/src/nccl/gpu_collective.cpp:    KUNGFU_CHECK(nccl_checker) << ncclGetVersion(&version);
srcs/cpp/src/nccl/gpu_collective.cpp:    printf("NCCL Version: %d\n", version);
srcs/cpp/src/nccl/gpu_collective.cpp:struct nccl_type;
srcs/cpp/src/nccl/gpu_collective.cpp:struct nccl_type<int32_t> {
srcs/cpp/src/nccl/gpu_collective.cpp:    static ncclDataType_t value() { return ncclInt32; }
srcs/cpp/src/nccl/gpu_collective.cpp:struct nccl_type<kungfu::float16> {
srcs/cpp/src/nccl/gpu_collective.cpp:    static ncclDataType_t value() { return ncclFloat16; }
srcs/cpp/src/nccl/gpu_collective.cpp:struct nccl_type<float> {
srcs/cpp/src/nccl/gpu_collective.cpp:    static ncclDataType_t value() { return ncclFloat; }
srcs/cpp/src/nccl/gpu_collective.cpp:ncclDataType_t to_nccl_type(const KungFu_Datatype dtype)
srcs/cpp/src/nccl/gpu_collective.cpp:        return nccl_type<int32_t>::value();
srcs/cpp/src/nccl/gpu_collective.cpp:        return nccl_type<kungfu::float16>::value();
srcs/cpp/src/nccl/gpu_collective.cpp:        return nccl_type<float>::value();
srcs/cpp/src/nccl/gpu_collective.cpp:ncclRedOp_t to_nccl_op(const KungFu_Op op)
srcs/cpp/src/nccl/gpu_collective.cpp:        return ncclSum;
srcs/cpp/src/nccl/gpu_collective.cpp:        return ncclMin;
srcs/cpp/src/nccl/gpu_collective.cpp:        return ncclMax;
srcs/cpp/src/nccl/gpu_collective.cpp:        return ncclProd;
srcs/cpp/src/nccl/gpu_collective.cpp:class gpu_collective_nccl : public gpu_collective
srcs/cpp/src/nccl/gpu_collective.cpp:    ncclComm_t comm_;
srcs/cpp/src/nccl/gpu_collective.cpp:    CudaStream stream_;
srcs/cpp/src/nccl/gpu_collective.cpp:    gpu_collective_nccl(ncclUniqueId id, int cluster_size, int rank, int root)
srcs/cpp/src/nccl/gpu_collective.cpp:        KUNGFU_CHECK(nccl_checker)
srcs/cpp/src/nccl/gpu_collective.cpp:            << ncclCommInitRank(&comm_, cluster_size, id, rank);
srcs/cpp/src/nccl/gpu_collective.cpp:    ~gpu_collective_nccl()
srcs/cpp/src/nccl/gpu_collective.cpp:        KUNGFU_CHECK(nccl_checker) << ncclCommDestroy(comm_);
srcs/cpp/src/nccl/gpu_collective.cpp:        // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclreduce
srcs/cpp/src/nccl/gpu_collective.cpp:        KUNGFU_CHECK_HINT(nccl_checker, __func__)
srcs/cpp/src/nccl/gpu_collective.cpp:            << ncclReduce(send_buf, recv_buf, count, to_nccl_type(dtype),
srcs/cpp/src/nccl/gpu_collective.cpp:                          ncclSum, root_, comm_, stream_);
srcs/cpp/src/nccl/gpu_collective.cpp:        // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclbroadcast
srcs/cpp/src/nccl/gpu_collective.cpp:        KUNGFU_CHECK_HINT(nccl_checker, __func__)
srcs/cpp/src/nccl/gpu_collective.cpp:            << ncclBroadcast(send_buf, recv_buf, count, to_nccl_type(dtype),
srcs/cpp/src/nccl/gpu_collective.cpp:        // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclbroadcast
srcs/cpp/src/nccl/gpu_collective.cpp:        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
srcs/cpp/src/nccl/gpu_collective.cpp:        KUNGFU_CHECK_HINT(nccl_checker, __func__)
srcs/cpp/src/nccl/gpu_collective.cpp:            << ncclBroadcast(send_buf, recv_buf, count, to_nccl_type(dtype),
srcs/cpp/src/nccl/gpu_collective.cpp:        // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallreduce
srcs/cpp/src/nccl/gpu_collective.cpp:        KUNGFU_CHECK_HINT(nccl_checker, __func__)
srcs/cpp/src/nccl/gpu_collective.cpp:            << ncclAllReduce(send_buf, recv_buf, count, to_nccl_type(dtype),
srcs/cpp/src/nccl/gpu_collective.cpp:                             to_nccl_op(op), comm_, stream_);
srcs/cpp/src/nccl/gpu_collective.cpp:        // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallreduce
srcs/cpp/src/nccl/gpu_collective.cpp:        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
srcs/cpp/src/nccl/gpu_collective.cpp:        KUNGFU_CHECK_HINT(nccl_checker, __func__)
srcs/cpp/src/nccl/gpu_collective.cpp:            << ncclAllReduce(send_buf, recv_buf, count, to_nccl_type(dtype),
srcs/cpp/src/nccl/gpu_collective.cpp:                             to_nccl_op(op), comm_, stream);
srcs/cpp/src/nccl/gpu_collective.cpp:        // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallgather
srcs/cpp/src/nccl/gpu_collective.cpp:        KUNGFU_CHECK_HINT(nccl_checker, __func__)
srcs/cpp/src/nccl/gpu_collective.cpp:            << ncclAllGather(send_buf, recv_buf, send_count,
srcs/cpp/src/nccl/gpu_collective.cpp:                             to_nccl_type(dtype), comm_, stream_);
srcs/cpp/src/nccl/gpu_collective.cpp:        // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallgather
srcs/cpp/src/nccl/gpu_collective.cpp:        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
srcs/cpp/src/nccl/gpu_collective.cpp:        KUNGFU_CHECK_HINT(nccl_checker, __func__) << ncclAllGather(
srcs/cpp/src/nccl/gpu_collective.cpp:            send_buf, recv_buf, send_count, to_nccl_type(dtype), comm_, stream);
srcs/cpp/src/nccl/gpu_collective.cpp:gpu_collective *gpu_collective::new_global(kungfu::Peer &self)
srcs/cpp/src/nccl/gpu_collective.cpp:    ncclUniqueId id;
srcs/cpp/src/nccl/gpu_collective.cpp:    KUNGFU_CHECK(cuda_checker) << cudaSetDevice(kungfu_get_cuda_index());
srcs/cpp/src/nccl/gpu_collective.cpp:    if (rank == root) { KUNGFU_CHECK(nccl_checker) << ncclGetUniqueId(&id); }
srcs/cpp/src/nccl/gpu_collective.cpp:                   "nccl id");
srcs/cpp/src/nccl/gpu_collective.cpp:    return new gpu_collective_nccl(id, self.Size(), rank, root);
srcs/cpp/src/nccl/gpu_collective.cpp:gpu_collective *gpu_collective::new_local(kungfu::Peer &self)
srcs/cpp/src/nccl/gpu_collective.cpp:    ncclUniqueId id;
srcs/cpp/src/nccl/gpu_collective.cpp:    KUNGFU_CHECK(cuda_checker) << cudaSetDevice(kungfu_get_cuda_index());
srcs/cpp/src/nccl/gpu_collective.cpp:    if (rank == root) { KUNGFU_CHECK(nccl_checker) << ncclGetUniqueId(&id); }
srcs/cpp/src/nccl/gpu_collective.cpp:                        "local nccl id");
srcs/cpp/src/nccl/gpu_collective.cpp:    return new gpu_collective_nccl(id, self.LocalSize(), rank, root);
srcs/cpp/src/nccl/gpu_collective.cpp:gpu_collective *gpu_collective::new_group(kungfu::Peer &self,
srcs/cpp/src/nccl/gpu_collective.cpp:    ncclUniqueId id;
srcs/cpp/src/nccl/gpu_collective.cpp:    KUNGFU_CHECK(cuda_checker) << cudaSetDevice(kungfu_get_cuda_index());
srcs/cpp/src/nccl/gpu_collective.cpp:        KUNGFU_CHECK(nccl_checker) << ncclGetUniqueId(&id);
srcs/cpp/src/nccl/gpu_collective.cpp:                         topology.data(), "group nccl id");
srcs/cpp/src/nccl/gpu_collective.cpp:    return new gpu_collective_nccl(id, size, group_rank, 0);
srcs/cpp/src/cuda/stream.cpp:#include <kungfu/cuda/stream.hpp>
srcs/cpp/src/cuda/stream.cpp:CudaStream::CudaStream()
srcs/cpp/src/cuda/stream.cpp:    KUNGFU_CHECK(cuda_checker) << cudaStreamCreate(&stream_);
srcs/cpp/src/cuda/stream.cpp:CudaStream::~CudaStream()
srcs/cpp/src/cuda/stream.cpp:    const cudaError_t err = cudaStreamDestroy(stream_);
srcs/cpp/src/cuda/stream.cpp:    if (err == cudaErrorCudartUnloading ||
srcs/cpp/src/cuda/stream.cpp:        fprintf(stderr, "ignore cudaStreamDestroy error: %s\n",
srcs/cpp/src/cuda/stream.cpp:                show_cuda_error()(err).c_str());
srcs/cpp/src/cuda/stream.cpp:    KUNGFU_CHECK(cuda_checker) << err;
srcs/cpp/src/cuda/stream.cpp:void CudaStream::sync()
srcs/cpp/src/cuda/stream.cpp:    KUNGFU_CHECK(cuda_checker) << cudaStreamSynchronize(stream_);
srcs/cpp/src/cuda/stream.cpp:void CudaStream::memcpy(void *dst, const void *src, const size_t count,
srcs/cpp/src/cuda/stream.cpp:                        const cudaMemcpyKind dir)
srcs/cpp/src/cuda/stream.cpp:    KUNGFU_CHECK(cuda_checker)
srcs/cpp/src/cuda/stream.cpp:        << cudaMemcpyAsync(dst, src, count, dir, stream_);
srcs/cpp/src/cuda/stream.cpp:std::unique_ptr<CudaStream> StreamPool::Get()
srcs/cpp/src/cuda/stream.cpp:    return std::make_unique<CudaStream>();
srcs/cpp/src/cuda/stream.cpp:void StreamPool::Put(std::unique_ptr<CudaStream> stream)
srcs/cpp/src/tensorflow/ops/gpu/scheduler.cpp:#include <kungfu/nccl/helper.hpp>
srcs/cpp/src/tensorflow/ops/gpu/scheduler.cpp:REGISTER_KUNGFU_OP(StartNcclScheduler)
srcs/cpp/src/tensorflow/ops/gpu/scheduler.cpp:class StartNcclScheduler : public OpKernel
srcs/cpp/src/tensorflow/ops/gpu/scheduler.cpp:    KungFu_NCCLScope nccl_scope_;
srcs/cpp/src/tensorflow/ops/gpu/scheduler.cpp:    explicit StartNcclScheduler(OpKernelConstruction *context)
srcs/cpp/src/tensorflow/ops/gpu/scheduler.cpp:        OP_REQUIRES(context, kungfu::_nccl_scopes.count(scope_name) > 0,
srcs/cpp/src/tensorflow/ops/gpu/scheduler.cpp:        nccl_scope_ = kungfu::_nccl_scopes.at(scope_name);
srcs/cpp/src/tensorflow/ops/gpu/scheduler.cpp:            kungfu::NCCLHelper::GetDefault()->EnsureScheduler(nccl_scope_);
srcs/cpp/src/tensorflow/ops/gpu/scheduler.cpp:            kungfu::NCCLHelper::GetDefault()->EnsureController(nccl_scope_);
srcs/cpp/src/tensorflow/ops/gpu/scheduler.cpp:REGISTER_KUNGFU_KERNEL_BUILDER(StartNcclScheduler, DEVICE_CPU);
srcs/cpp/src/tensorflow/ops/gpu/scheduler.cpp:REGISTER_KUNGFU_OP(ResetNcclHelper)
srcs/cpp/src/tensorflow/ops/gpu/scheduler.cpp:class ResetNcclHelper : public OpKernel
srcs/cpp/src/tensorflow/ops/gpu/scheduler.cpp:        if (changed && !detached) { kungfu_python_init_nccl(); }
srcs/cpp/src/tensorflow/ops/gpu/scheduler.cpp:REGISTER_KUNGFU_KERNEL_BUILDER(ResetNcclHelper, DEVICE_CPU);
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:#include <kungfu/nccl/helper.hpp>
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:void spin_wait(perftools::gputools::Event *event, int ms = 100)
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:           perftools::gputools::Event::Status::kPending) {
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:perftools::gputools::Event *create_init_ready_event(OpKernelContext *context)
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:    auto ready_event    = new perftools::gputools::Event(executor);
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:void wait_delete_ready_event(perftools::gputools::Event *ready_event)
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:REGISTER_KUNGFU_OP(ScheduledNcclAllReduce)
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:class ScheduledNcclAllReduce : public AsyncOpKernel
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:    const KungFu_NCCLScope nccl_scope_;
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:    explicit ScheduledNcclAllReduce(OpKernelConstruction *context)
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:        : AsyncOpKernel(context), nccl_scope_(KungFu_NCCL_GLOBAL)
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:            kungfu::NCCLHelper::GetDefault()->EnsureScheduler(nccl_scope_);
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:            kungfu::NCCLHelper::GetDefault()->EnsureController(nccl_scope_);
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:REGISTER_KUNGFU_KERNEL_BUILDER(ScheduledNcclAllReduce, DEVICE_GPU);
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:REGISTER_KUNGFU_OP(NcclAllReduce)
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:class NcclAllReduce : public AsyncOpKernel
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:        auto scheduler_ = kungfu::NCCLHelper::GetDefault()->EnsureScheduler(
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:            KungFu_NCCL_GLOBAL);
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:        auto controller_ = kungfu::NCCLHelper::GetDefault()->EnsureController(
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:            KungFu_NCCL_GLOBAL);
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:REGISTER_KUNGFU_KERNEL_BUILDER(NcclAllReduce, DEVICE_GPU);
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:REGISTER_KUNGFU_OP(ScheduledHierarchicalNcclAllReduce)
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:class ScheduledHierarchicalNcclAllReduce : public AsyncOpKernel
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:    const KungFu_NCCLScope nccl_scope_;
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:    explicit ScheduledHierarchicalNcclAllReduce(OpKernelConstruction *context)
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:        : AsyncOpKernel(context), nccl_scope_(KungFu_NCCL_LOCAL)
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:            kungfu::NCCLHelper::GetDefault()->EnsureScheduler(nccl_scope_);
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:            kungfu::NCCLHelper::GetDefault()->EnsureController(nccl_scope_);
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:                CrossAllReduceGpu(peer, w_all_reduce, KungFu_SUM, name(), [=] {
srcs/cpp/src/tensorflow/ops/gpu/collective.cpp:REGISTER_KUNGFU_KERNEL_BUILDER(ScheduledHierarchicalNcclAllReduce, DEVICE_GPU);
srcs/go/nccl/bug.go:package nccl
srcs/go/libkungfu-comm/kungfu/python/c_api.h:extern void kungfu_python_init_nccl();
srcs/go/libkungfu-comm/kungfu/python/c_api.h:extern void kungfu_python_finialize_nccl();
srcs/go/libkungfu-comm/kungfu/python/c_api.h:extern int kungfu_get_cuda_index();
srcs/go/libkungfu-comm/kungfu/python/c_api.h:class NCCLHelper;
srcs/go/libkungfu-comm/kungfu/torch/common.hpp:class CudaStream;
srcs/go/libkungfu-comm/kungfu/torch/common.hpp:class TorchCudaHelper
srcs/go/libkungfu-comm/kungfu/torch/common.hpp:    std::unique_ptr<CudaStream> up_stream_;
srcs/go/libkungfu-comm/kungfu/torch/common.hpp:    std::unique_ptr<CudaStream> down_stream_;
srcs/go/libkungfu-comm/kungfu/torch/common.hpp:    TorchCudaHelper();
srcs/go/libkungfu-comm/kungfu/torch/common.hpp:    void from_cuda(void *buffer, const torch::Tensor &t);
srcs/go/libkungfu-comm/kungfu/torch/common.hpp:    void from_cuda(void *dst, const void *src, size_t size);
srcs/go/libkungfu-comm/kungfu/torch/common.hpp:    void to_cuda(torch::Tensor &t, const void *buffer);
srcs/go/libkungfu-comm/kungfu/torch/common.hpp:    void to_cuda(void *dst, const void *src, size_t size);
srcs/go/libkungfu-comm/kungfu/torch/common.hpp:extern TorchCudaHelper _torch_cuda_helper;
srcs/go/libkungfu-comm/kungfu/torch/common.hpp:    Torch_Cuda_Float,
srcs/go/libkungfu-comm/kungfu/nccl/helper.hpp:#include <kungfu/cuda/stream.hpp>
srcs/go/libkungfu-comm/kungfu/nccl/helper.hpp:#include <kungfu/nccl/common.hpp>
srcs/go/libkungfu-comm/kungfu/nccl/helper.hpp:#include <kungfu/nccl/controller.hpp>
srcs/go/libkungfu-comm/kungfu/nccl/helper.hpp:#include <kungfu/nccl/scheduler.hpp>
srcs/go/libkungfu-comm/kungfu/nccl/helper.hpp:// NCCLHelper is a singleton class that contains NCCL related global variables
srcs/go/libkungfu-comm/kungfu/nccl/helper.hpp:class NCCLHelper
srcs/go/libkungfu-comm/kungfu/nccl/helper.hpp:    std::map<KungFu_NCCLScope, std::unique_ptr<NCCLScheduler>> schedulers_;
srcs/go/libkungfu-comm/kungfu/nccl/helper.hpp:    std::map<KungFu_NCCLScope, std::unique_ptr<NCCLController>> controllers_;
srcs/go/libkungfu-comm/kungfu/nccl/helper.hpp:    std::map<std::string, std::unique_ptr<NCCLController>> group_controllers_;
srcs/go/libkungfu-comm/kungfu/nccl/helper.hpp:    NCCLHelper();
srcs/go/libkungfu-comm/kungfu/nccl/helper.hpp:    NCCLScheduler *EnsureScheduler(const KungFu_NCCLScope scope);
srcs/go/libkungfu-comm/kungfu/nccl/helper.hpp:    NCCLController *EnsureController(const KungFu_NCCLScope scope);
srcs/go/libkungfu-comm/kungfu/nccl/helper.hpp:    NCCLController *EnsureGroupController(std::vector<int32_t> topology);
srcs/go/libkungfu-comm/kungfu/nccl/helper.hpp:    static std::unique_ptr<NCCLHelper> &GetDefault(bool reinit = false);
srcs/go/libkungfu-comm/kungfu/nccl/gpu_collective.hpp:extern void kungfu_show_cuda_version();
srcs/go/libkungfu-comm/kungfu/nccl/gpu_collective.hpp:extern void kungfu_show_nccl_version();
srcs/go/libkungfu-comm/kungfu/nccl/gpu_collective.hpp:// gpu_collective wraps NCCL APIs for internal use.
srcs/go/libkungfu-comm/kungfu/nccl/gpu_collective.hpp:// User should use NCCLController
srcs/go/libkungfu-comm/kungfu/nccl/gpu_collective.hpp:class gpu_collective
srcs/go/libkungfu-comm/kungfu/nccl/gpu_collective.hpp:    virtual ~gpu_collective() = default;
srcs/go/libkungfu-comm/kungfu/nccl/gpu_collective.hpp:    static gpu_collective *new_global(kungfu::Peer &);
srcs/go/libkungfu-comm/kungfu/nccl/gpu_collective.hpp:    static gpu_collective *new_local(kungfu::Peer &);
srcs/go/libkungfu-comm/kungfu/nccl/gpu_collective.hpp:    static gpu_collective *new_group(kungfu::Peer &,
srcs/go/libkungfu-comm/kungfu/nccl/scheduler.hpp:#include <kungfu/nccl/common.hpp>
srcs/go/libkungfu-comm/kungfu/nccl/scheduler.hpp:class NCCLThread
srcs/go/libkungfu-comm/kungfu/nccl/scheduler.hpp:    NCCLThread();
srcs/go/libkungfu-comm/kungfu/nccl/scheduler.hpp:    ~NCCLThread();
srcs/go/libkungfu-comm/kungfu/nccl/scheduler.hpp:    NCCLThread *nccl_thread_;
srcs/go/libkungfu-comm/kungfu/nccl/scheduler.hpp:                   const std::vector<int32_t> &order, NCCLThread *nccl_thread);
srcs/go/libkungfu-comm/kungfu/nccl/scheduler.hpp:class NCCLScheduler
srcs/go/libkungfu-comm/kungfu/nccl/scheduler.hpp:    const KungFu_NCCLScope scope_;
srcs/go/libkungfu-comm/kungfu/nccl/scheduler.hpp:    std::unique_ptr<NCCLThread> nccl_thread_;
srcs/go/libkungfu-comm/kungfu/nccl/scheduler.hpp:    NCCLScheduler(const KungFu_NCCLScope scope, const bool auto_order = true);
srcs/go/libkungfu-comm/kungfu/nccl/scheduler.hpp:    // Run a task in the dedicated NCCL thread
srcs/go/libkungfu-comm/kungfu/nccl/controller.hpp:#include <kungfu/nccl/common.hpp>
srcs/go/libkungfu-comm/kungfu/nccl/controller.hpp:#include <kungfu/nccl/gpu_collective.hpp>
srcs/go/libkungfu-comm/kungfu/nccl/controller.hpp:void CrossAllReduceGpu(Peer *peer, const Workspace &w, KungFu_Op op,
srcs/go/libkungfu-comm/kungfu/nccl/controller.hpp:// NCCLController exposes user-facing APIs
srcs/go/libkungfu-comm/kungfu/nccl/controller.hpp:class NCCLController
srcs/go/libkungfu-comm/kungfu/nccl/controller.hpp:    KungFu_NCCLScope scope_;
srcs/go/libkungfu-comm/kungfu/nccl/controller.hpp:    // only used when scope_ == KungFu_NCCL_GROUP
srcs/go/libkungfu-comm/kungfu/nccl/controller.hpp:    std::unique_ptr<gpu_collective> gpu_collective_;
srcs/go/libkungfu-comm/kungfu/nccl/controller.hpp:    gpu_collective *new_gpu_collective(Peer *peer);
srcs/go/libkungfu-comm/kungfu/nccl/controller.hpp:    NCCLController(const KungFu_NCCLScope scope);
srcs/go/libkungfu-comm/kungfu/nccl/controller.hpp:    NCCLController(std::vector<int32_t> topology);
srcs/go/libkungfu-comm/kungfu/nccl/common.hpp:enum KungFu_NCCLScope {
srcs/go/libkungfu-comm/kungfu/nccl/common.hpp:    KungFu_NCCL_GLOBAL,
srcs/go/libkungfu-comm/kungfu/nccl/common.hpp:    KungFu_NCCL_LOCAL,
srcs/go/libkungfu-comm/kungfu/nccl/common.hpp:    KungFu_NCCL_GROUP,
srcs/go/libkungfu-comm/kungfu/nccl/common.hpp:extern const std::map<std::string, KungFu_NCCLScope> _nccl_scopes;
srcs/go/libkungfu-comm/kungfu/cuda/stream.hpp:#include <cuda_runtime.h>
srcs/go/libkungfu-comm/kungfu/cuda/stream.hpp:struct show_cuda_error {
srcs/go/libkungfu-comm/kungfu/cuda/stream.hpp:    std::string operator()(cudaError_t err) const
srcs/go/libkungfu-comm/kungfu/cuda/stream.hpp:               cudaGetErrorString(err);
srcs/go/libkungfu-comm/kungfu/cuda/stream.hpp:using cuda_checker = error_checker<cudaError_t, cudaSuccess, show_cuda_error>;
srcs/go/libkungfu-comm/kungfu/cuda/stream.hpp:// CudaStream wraps cudaStream_t
srcs/go/libkungfu-comm/kungfu/cuda/stream.hpp:class CudaStream
srcs/go/libkungfu-comm/kungfu/cuda/stream.hpp:    cudaStream_t stream_;
srcs/go/libkungfu-comm/kungfu/cuda/stream.hpp:    CudaStream();
srcs/go/libkungfu-comm/kungfu/cuda/stream.hpp:    ~CudaStream();
srcs/go/libkungfu-comm/kungfu/cuda/stream.hpp:    operator cudaStream_t() const { return stream_; }
srcs/go/libkungfu-comm/kungfu/cuda/stream.hpp:                const cudaMemcpyKind dir);
srcs/go/libkungfu-comm/kungfu/cuda/stream.hpp:    std::queue<std::unique_ptr<CudaStream>> queue_;
srcs/go/libkungfu-comm/kungfu/cuda/stream.hpp:    std::unique_ptr<CudaStream> Get();
srcs/go/libkungfu-comm/kungfu/cuda/stream.hpp:    void Put(std::unique_ptr<CudaStream> stream);
srcs/go/libkungfu-comm/kungfu/tensorflow/ops.h:#include <kungfu/nccl/common.hpp>
srcs/go/cmd/kungfu-distribute/kungfu-distribute.go:		utils.LogCudaEnv()
srcs/go/cmd/kungfu-distribute/kungfu-distribute.go:		utils.LogNCCLEnv()
srcs/go/utils/utils.go:func LogCudaEnv() {
srcs/go/utils/utils.go:	LogEnvWithPrefix(`CUDA_`, `cuda-env`)
srcs/go/utils/utils.go:func LogNCCLEnv() {
srcs/go/utils/utils.go:	LogEnvWithPrefix(`NCCL_`, `nccl-env`)
srcs/go/utils/utils.go:func ListNvidiaGPUNames() []string {
srcs/go/utils/utils.go:	files, err := filepath.Glob(prefix + `nvidia*`)
srcs/go/utils/utils.go:		n, err := fmt.Sscanf(name, "nvidia%d", &x)
srcs/go/utils/utils.go:		if n == 1 && err == nil && fmt.Sprintf("nvidia%d", x) == name {
srcs/go/utils/runner/local/hack.go:	"github.com/lsds/KungFu/srcs/go/nccl"
srcs/go/utils/runner/local/hack.go:	if strings.HasPrefix(firstStderr.First, nccl.Bug) {
srcs/go/kungfu/job/gpu_resource.go:// GPUPool manages GPU ids
srcs/go/kungfu/job/gpu_resource.go:type GPUPool struct {
srcs/go/kungfu/job/gpu_resource.go:// NewGPUPool create a new GPUPool of given size
srcs/go/kungfu/job/gpu_resource.go:func NewGPUPool(n int) *GPUPool {
srcs/go/kungfu/job/gpu_resource.go:	return &GPUPool{cap: n, mask: mask}
srcs/go/kungfu/job/gpu_resource.go:// Get returns the smallest GPU id that is available
srcs/go/kungfu/job/gpu_resource.go:func (p *GPUPool) Get() int {
srcs/go/kungfu/job/gpu_resource.go:var errGPUNotAllocated = errors.New("GPU not allocated")
srcs/go/kungfu/job/gpu_resource.go:// Put puts an GPU id back to the pool
srcs/go/kungfu/job/gpu_resource.go:func (p *GPUPool) Put(id int) {
srcs/go/kungfu/job/gpu_resource.go:			utils.ExitErr(errGPUNotAllocated)
srcs/go/kungfu/job/cuda_visible_device.go:// https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/
srcs/go/kungfu/job/cuda_visible_device.go:const cudaVisibleDevicesKey = `CUDA_VISIBLE_DEVICES`
srcs/go/kungfu/job/cuda_visible_device.go:func getCudaIndex(localRank int) int {
srcs/go/kungfu/job/cuda_visible_device.go:	val, ok := lookupEnv(cudaVisibleDevicesKey)
srcs/go/kungfu/job/cuda_visible_device.go:	ids, err := parseCudaVisibleDevices(val)
srcs/go/kungfu/job/cuda_visible_device.go:		log.Warnf("invalid valud of %s: %q", cudaVisibleDevicesKey, val)
srcs/go/kungfu/job/cuda_visible_device.go:		log.Warnf("%s=%s is not enough for local rank %d", cudaVisibleDevicesKey, val, localRank)
srcs/go/kungfu/job/cuda_visible_device.go:var errInvalidCudaVisibleDevices = errors.New("invalid " + cudaVisibleDevicesKey)
srcs/go/kungfu/job/cuda_visible_device.go:func parseCudaVisibleDevices(val string) ([]int, error) {
srcs/go/kungfu/job/cuda_visible_device.go:			return nil, errInvalidCudaVisibleDevices
srcs/go/kungfu/job/cuda_visible_device_test.go:var cudaEnv = map[string]string{}
srcs/go/kungfu/job/cuda_visible_device_test.go:	val, ok := cudaEnv[key]
srcs/go/kungfu/job/cuda_visible_device_test.go:func Test_getCudaIndex(t *testing.T) {
srcs/go/kungfu/job/cuda_visible_device_test.go:	if id := getCudaIndex(1); id != 1 {
srcs/go/kungfu/job/cuda_visible_device_test.go:	cudaEnv[`CUDA_VISIBLE_DEVICES`] = "2,3"
srcs/go/kungfu/job/cuda_visible_device_test.go:	if id := getCudaIndex(1); id != 3 {
srcs/go/kungfu/job/cuda_visible_device_test.go:	cudaEnv[`CUDA_VISIBLE_DEVICES`] = ""
srcs/go/kungfu/job/cuda_visible_device_test.go:	if id := getCudaIndex(0); id != -1 {
srcs/go/kungfu/job/job.go:var warnCudaOption = new(sync.Once)
srcs/go/kungfu/job/job.go:func (j Job) NewProc(peer plan.PeerID, gpuID int, initClusterVersion int, cluster plan.Cluster, initProgress uint64) proc.Proc {
srcs/go/kungfu/job/job.go:	cudaIdx := strconv.Itoa(getCudaIndex(gpuID))
srcs/go/kungfu/job/job.go:	envs[`KUNGFU_`+cudaVisibleDevicesKey] = cudaIdx
srcs/go/kungfu/job/job.go:		warnCudaOption.Do(func() {
srcs/go/kungfu/job/job.go:			log.Warnf("Please set `config.gpu_options.visible_device_list = str(local_rank)`")
srcs/go/kungfu/job/job.go:		envs[cudaVisibleDevicesKey] = cudaIdx
srcs/go/kungfu/base/kungfu/python/c_api.h:extern void kungfu_python_init_nccl();
srcs/go/kungfu/base/kungfu/python/c_api.h:extern void kungfu_python_finialize_nccl();
srcs/go/kungfu/base/kungfu/python/c_api.h:extern int kungfu_get_cuda_index();
srcs/go/kungfu/base/kungfu/python/c_api.h:class NCCLHelper;
srcs/go/kungfu/base/kungfu/torch/common.hpp:class CudaStream;
srcs/go/kungfu/base/kungfu/torch/common.hpp:class TorchCudaHelper
srcs/go/kungfu/base/kungfu/torch/common.hpp:    std::unique_ptr<CudaStream> up_stream_;
srcs/go/kungfu/base/kungfu/torch/common.hpp:    std::unique_ptr<CudaStream> down_stream_;
srcs/go/kungfu/base/kungfu/torch/common.hpp:    TorchCudaHelper();
srcs/go/kungfu/base/kungfu/torch/common.hpp:    void from_cuda(void *buffer, const torch::Tensor &t);
srcs/go/kungfu/base/kungfu/torch/common.hpp:    void from_cuda(void *dst, const void *src, size_t size);
srcs/go/kungfu/base/kungfu/torch/common.hpp:    void to_cuda(torch::Tensor &t, const void *buffer);
srcs/go/kungfu/base/kungfu/torch/common.hpp:    void to_cuda(void *dst, const void *src, size_t size);
srcs/go/kungfu/base/kungfu/torch/common.hpp:extern TorchCudaHelper _torch_cuda_helper;
srcs/go/kungfu/base/kungfu/torch/common.hpp:    Torch_Cuda_Float,
srcs/go/kungfu/base/kungfu/nccl/helper.hpp:#include <kungfu/cuda/stream.hpp>
srcs/go/kungfu/base/kungfu/nccl/helper.hpp:#include <kungfu/nccl/common.hpp>
srcs/go/kungfu/base/kungfu/nccl/helper.hpp:#include <kungfu/nccl/controller.hpp>
srcs/go/kungfu/base/kungfu/nccl/helper.hpp:#include <kungfu/nccl/scheduler.hpp>
srcs/go/kungfu/base/kungfu/nccl/helper.hpp:// NCCLHelper is a singleton class that contains NCCL related global variables
srcs/go/kungfu/base/kungfu/nccl/helper.hpp:class NCCLHelper
srcs/go/kungfu/base/kungfu/nccl/helper.hpp:    std::map<KungFu_NCCLScope, std::unique_ptr<NCCLScheduler>> schedulers_;
srcs/go/kungfu/base/kungfu/nccl/helper.hpp:    std::map<KungFu_NCCLScope, std::unique_ptr<NCCLController>> controllers_;
srcs/go/kungfu/base/kungfu/nccl/helper.hpp:    std::map<std::string, std::unique_ptr<NCCLController>> group_controllers_;
srcs/go/kungfu/base/kungfu/nccl/helper.hpp:    NCCLHelper();
srcs/go/kungfu/base/kungfu/nccl/helper.hpp:    NCCLScheduler *EnsureScheduler(const KungFu_NCCLScope scope);
srcs/go/kungfu/base/kungfu/nccl/helper.hpp:    NCCLController *EnsureController(const KungFu_NCCLScope scope);
srcs/go/kungfu/base/kungfu/nccl/helper.hpp:    NCCLController *EnsureGroupController(std::vector<int32_t> topology);
srcs/go/kungfu/base/kungfu/nccl/helper.hpp:    static std::unique_ptr<NCCLHelper> &GetDefault(bool reinit = false);
srcs/go/kungfu/base/kungfu/nccl/gpu_collective.hpp:extern void kungfu_show_cuda_version();
srcs/go/kungfu/base/kungfu/nccl/gpu_collective.hpp:extern void kungfu_show_nccl_version();
srcs/go/kungfu/base/kungfu/nccl/gpu_collective.hpp:// gpu_collective wraps NCCL APIs for internal use.
srcs/go/kungfu/base/kungfu/nccl/gpu_collective.hpp:// User should use NCCLController
srcs/go/kungfu/base/kungfu/nccl/gpu_collective.hpp:class gpu_collective
srcs/go/kungfu/base/kungfu/nccl/gpu_collective.hpp:    virtual ~gpu_collective() = default;
srcs/go/kungfu/base/kungfu/nccl/gpu_collective.hpp:    static gpu_collective *new_global(kungfu::Peer &);
srcs/go/kungfu/base/kungfu/nccl/gpu_collective.hpp:    static gpu_collective *new_local(kungfu::Peer &);
srcs/go/kungfu/base/kungfu/nccl/gpu_collective.hpp:    static gpu_collective *new_group(kungfu::Peer &,
srcs/go/kungfu/base/kungfu/nccl/scheduler.hpp:#include <kungfu/nccl/common.hpp>
srcs/go/kungfu/base/kungfu/nccl/scheduler.hpp:class NCCLThread
srcs/go/kungfu/base/kungfu/nccl/scheduler.hpp:    NCCLThread();
srcs/go/kungfu/base/kungfu/nccl/scheduler.hpp:    ~NCCLThread();
srcs/go/kungfu/base/kungfu/nccl/scheduler.hpp:    NCCLThread *nccl_thread_;
srcs/go/kungfu/base/kungfu/nccl/scheduler.hpp:                   const std::vector<int32_t> &order, NCCLThread *nccl_thread);
srcs/go/kungfu/base/kungfu/nccl/scheduler.hpp:class NCCLScheduler
srcs/go/kungfu/base/kungfu/nccl/scheduler.hpp:    const KungFu_NCCLScope scope_;
srcs/go/kungfu/base/kungfu/nccl/scheduler.hpp:    std::unique_ptr<NCCLThread> nccl_thread_;
srcs/go/kungfu/base/kungfu/nccl/scheduler.hpp:    NCCLScheduler(const KungFu_NCCLScope scope, const bool auto_order = true);
srcs/go/kungfu/base/kungfu/nccl/scheduler.hpp:    // Run a task in the dedicated NCCL thread
srcs/go/kungfu/base/kungfu/nccl/controller.hpp:#include <kungfu/nccl/common.hpp>
srcs/go/kungfu/base/kungfu/nccl/controller.hpp:#include <kungfu/nccl/gpu_collective.hpp>
srcs/go/kungfu/base/kungfu/nccl/controller.hpp:void CrossAllReduceGpu(Peer *peer, const Workspace &w, KungFu_Op op,
srcs/go/kungfu/base/kungfu/nccl/controller.hpp:// NCCLController exposes user-facing APIs
srcs/go/kungfu/base/kungfu/nccl/controller.hpp:class NCCLController
srcs/go/kungfu/base/kungfu/nccl/controller.hpp:    KungFu_NCCLScope scope_;
srcs/go/kungfu/base/kungfu/nccl/controller.hpp:    // only used when scope_ == KungFu_NCCL_GROUP
srcs/go/kungfu/base/kungfu/nccl/controller.hpp:    std::unique_ptr<gpu_collective> gpu_collective_;
srcs/go/kungfu/base/kungfu/nccl/controller.hpp:    gpu_collective *new_gpu_collective(Peer *peer);
srcs/go/kungfu/base/kungfu/nccl/controller.hpp:    NCCLController(const KungFu_NCCLScope scope);
srcs/go/kungfu/base/kungfu/nccl/controller.hpp:    NCCLController(std::vector<int32_t> topology);
srcs/go/kungfu/base/kungfu/nccl/common.hpp:enum KungFu_NCCLScope {
srcs/go/kungfu/base/kungfu/nccl/common.hpp:    KungFu_NCCL_GLOBAL,
srcs/go/kungfu/base/kungfu/nccl/common.hpp:    KungFu_NCCL_LOCAL,
srcs/go/kungfu/base/kungfu/nccl/common.hpp:    KungFu_NCCL_GROUP,
srcs/go/kungfu/base/kungfu/nccl/common.hpp:extern const std::map<std::string, KungFu_NCCLScope> _nccl_scopes;
srcs/go/kungfu/base/kungfu/cuda/stream.hpp:#include <cuda_runtime.h>
srcs/go/kungfu/base/kungfu/cuda/stream.hpp:struct show_cuda_error {
srcs/go/kungfu/base/kungfu/cuda/stream.hpp:    std::string operator()(cudaError_t err) const
srcs/go/kungfu/base/kungfu/cuda/stream.hpp:               cudaGetErrorString(err);
srcs/go/kungfu/base/kungfu/cuda/stream.hpp:using cuda_checker = error_checker<cudaError_t, cudaSuccess, show_cuda_error>;
srcs/go/kungfu/base/kungfu/cuda/stream.hpp:// CudaStream wraps cudaStream_t
srcs/go/kungfu/base/kungfu/cuda/stream.hpp:class CudaStream
srcs/go/kungfu/base/kungfu/cuda/stream.hpp:    cudaStream_t stream_;
srcs/go/kungfu/base/kungfu/cuda/stream.hpp:    CudaStream();
srcs/go/kungfu/base/kungfu/cuda/stream.hpp:    ~CudaStream();
srcs/go/kungfu/base/kungfu/cuda/stream.hpp:    operator cudaStream_t() const { return stream_; }
srcs/go/kungfu/base/kungfu/cuda/stream.hpp:                const cudaMemcpyKind dir);
srcs/go/kungfu/base/kungfu/cuda/stream.hpp:    std::queue<std::unique_ptr<CudaStream>> queue_;
srcs/go/kungfu/base/kungfu/cuda/stream.hpp:    std::unique_ptr<CudaStream> Get();
srcs/go/kungfu/base/kungfu/cuda/stream.hpp:    void Put(std::unique_ptr<CudaStream> stream);
srcs/go/kungfu/base/kungfu/tensorflow/ops.h:#include <kungfu/nccl/common.hpp>
srcs/go/kungfu/runner/watch.go:	gpuPool *job.GPUPool
srcs/go/kungfu/runner/watch.go:	gpuID := w.gpuPool.Get()
srcs/go/kungfu/runner/watch.go:	if gpuID < 0 {
srcs/go/kungfu/runner/watch.go:		log.Errorf("gpuID = %d", gpuID)
srcs/go/kungfu/runner/watch.go:	proc := w.job.NewProc(id, gpuID, s.Version, s.Cluster, s.Progress)
srcs/go/kungfu/runner/watch.go:		w.gpuPool.Put(gpuID)
srcs/go/kungfu/runner/watch.go:		gpuPool: job.NewGPUPool(j.HostList.SlotOf(self.IPv4)),
srcs/go/kungfu/runner/flags.go:		utils.LogCudaEnv()
srcs/go/kungfu/runner/flags.go:		utils.LogNCCLEnv()
srcs/go/kungfu/runner/flags.go:	flag.BoolVar(&f.AllowNVLink, "allow-nvlink", false, "allow NCCL to discover NVLink")
CMakeLists.txt:IF(KUNGFU_ENABLE_NCCL)
CMakeLists.txt:    INCLUDE(srcs/cmake/nccl.cmake)
CMakeLists.txt:        kungfu_nccl
CMakeLists.txt:        srcs/cpp/src/cuda/stream.cpp
CMakeLists.txt:        srcs/cpp/src/nccl/common.cpp
CMakeLists.txt:        srcs/cpp/src/nccl/controller.cpp
CMakeLists.txt:        srcs/cpp/src/nccl/gpu_collective.cpp
CMakeLists.txt:        srcs/cpp/src/nccl/helper.cpp
CMakeLists.txt:        srcs/cpp/src/nccl/scheduler.cpp
CMakeLists.txt:        srcs/cpp/src/python/cuda.cpp
CMakeLists.txt:    TARGET_LINK_LIBRARIES(kungfu_nccl kungfu)
CMakeLists.txt:    USE_NCCL(kungfu_nccl)
CMakeLists.txt:        TARGET_COMPILE_DEFINITIONS(kungfu_nccl
CMakeLists.txt:        srcs/cpp/src/nccl/common.cpp srcs/cpp/src/nccl/scheduler.cpp
CMakeLists.txt:        srcs/cpp/src/python/c_api.cpp srcs/cpp/src/python/cuda.cpp
CMakeLists.txt:    IF(KUNGFU_ENABLE_NCCL)
CMakeLists.txt:            PRIVATE srcs/cpp/src/python/init_nccl.cpp
CMakeLists.txt:                    srcs/cpp/src/cuda/stream.cpp
CMakeLists.txt:                    srcs/cpp/src/nccl/controller.cpp
CMakeLists.txt:                    srcs/cpp/src/nccl/helper.cpp
CMakeLists.txt:                    srcs/cpp/src/nccl/gpu_collective.cpp)
CMakeLists.txt:        USE_NCCL(kungfu_python)
CMakeLists.txt:    FILE(GLOB GPU_OP_SRCS
CMakeLists.txt:         ${CMAKE_SOURCE_DIR}/srcs/cpp/src/tensorflow/ops/gpu/*.cpp)
CMakeLists.txt:    IF(KUNGFU_ENABLE_NCCL)
CMakeLists.txt:        TARGET_SOURCES(kungfu_tensorflow_ops PRIVATE ${GPU_OP_SRCS})
CMakeLists.txt:        USE_NCCL(kungfu_tensorflow_ops)
scripts/install-nccl.sh:filename=nccl-${version}.tar.gz
scripts/install-nccl.sh:URL=https://github.com/NVIDIA/nccl/archive/v${version}.tar.gz
scripts/install-nccl.sh:    PREFIX=$HOME/local/nccl
scripts/tests/run-nccl-train-test.sh:export NCCL_HOME=$HOME/local/nccl
scripts/tests/run-nccl-train-test.sh:    KUNGFU_ENABLE_NCCL=1 \
scripts/tests/run-nccl-train-test.sh:export LD_LIBRARY_PATH=$NCCL_HOME/lib
scripts/tests/run-nccl-train-test.sh:run_nccl_experiment() {
scripts/tests/run-nccl-train-test.sh:run_nccl_experiment_all() {
scripts/tests/run-nccl-train-test.sh:        run_nccl_experiment $np $@
scripts/tests/run-nccl-train-test.sh:run_nccl_experiment_all ./tests/python/integration/fake_tf_trainer.py
scripts/tests/run-nccl-train-test.sh:run_nccl_experiment_all ./experiments/kungfu/kf_tensorflow_synthetic_benchmark.py
scripts/tests/run-fake-trainer.sh:run_fake_nccl_trainer() {
scripts/tests/run-fake-trainer.sh:    #     ./bin/fake-nccl-trainer
scripts/tests/run-fake-trainer.sh:        ./bin/fake-nccl-trainer
scripts/tests/run-fake-trainer.sh:    elif [ "$collective" = "nccl" ]; then
scripts/tests/run-fake-trainer.sh:        run_fake_trainer_all run_fake_nccl_trainer
scripts/tests/run-fake-trainer.sh:        if [ -f /usr/include/nccl.h ]; then
scripts/tests/run-fake-trainer.sh:            run_fake_trainer_all run_fake_nccl_trainer
scripts/tests/run-tensorflow-resize-test.sh:# Test NCCL
scripts/tests/run-tensorflow-resize-test.sh:kungfu_run python3 tests/python/integration/test_tensorflow_resize.py --use-nccl
benchmarks/monitoring/benchmark.py:parser.add_argument('--no-cuda',
benchmarks/monitoring/benchmark.py:                    help='disables CUDA training')
benchmarks/monitoring/benchmark.py:args.cuda = not args.no_cuda
benchmarks/monitoring/benchmark.py:if args.cuda:
benchmarks/monitoring/benchmark.py:    config.gpu_options.allow_growth = True
benchmarks/monitoring/benchmark.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
benchmarks/monitoring/benchmark.py:    config.gpu_options.allow_growth = False
benchmarks/monitoring/benchmark.py:    config.gpu_options.visible_device_list = ''
benchmarks/monitoring/benchmark.py:device = '/gpu:0' if args.cuda else 'CPU'
benchmarks/monitoring/run_benchmark.sh:# Assume each host has 8 GPUs.
benchmarks/system/benchmark_horovod_torch.py:# Assuming you have 4 GPUs locally.
benchmarks/system/benchmark_horovod_torch.py:parser.add_argument('--no-cuda',
benchmarks/system/benchmark_horovod_torch.py:                    help='disables CUDA training')
benchmarks/system/benchmark_horovod_torch.py:args.cuda = not args.no_cuda and torch.cuda.is_available()
benchmarks/system/benchmark_horovod_torch.py:if args.cuda:
benchmarks/system/benchmark_horovod_torch.py:    # Horovod: pin GPU to local rank.
benchmarks/system/benchmark_horovod_torch.py:    torch.cuda.set_device(hvd.local_rank())
benchmarks/system/benchmark_horovod_torch.py:if args.cuda:
benchmarks/system/benchmark_horovod_torch.py:    # Move model to GPU.
benchmarks/system/benchmark_horovod_torch.py:    model.cuda()
benchmarks/system/benchmark_horovod_torch.py:    # If using GPU Adasum allreduce, scale learning rate by local_size.
benchmarks/system/benchmark_horovod_torch.py:    if args.use_adasum and hvd.nccl_built():
benchmarks/system/benchmark_horovod_torch.py:if args.cuda:
benchmarks/system/benchmark_horovod_torch.py:    data, target = data.cuda(), target.cuda()
benchmarks/system/benchmark_horovod_torch.py:device = 'GPU' if args.cuda else 'CPU'
benchmarks/system/benchmark_kungfu.py:parser.add_argument('--no-cuda',
benchmarks/system/benchmark_kungfu.py:                    help='disables CUDA training')
benchmarks/system/benchmark_kungfu.py:args.cuda = not args.no_cuda
benchmarks/system/benchmark_kungfu.py:if args.cuda:
benchmarks/system/benchmark_kungfu.py:    config.gpu_options.allow_growth = True
benchmarks/system/benchmark_kungfu.py:    from kungfu.python import _get_cuda_index
benchmarks/system/benchmark_kungfu.py:    config.gpu_options.visible_device_list = str(_get_cuda_index())
benchmarks/system/benchmark_kungfu.py:    config.gpu_options.allow_growth = False
benchmarks/system/benchmark_kungfu.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
benchmarks/system/benchmark_kungfu.py:    config.gpu_options.visible_device_list = ''
benchmarks/system/benchmark_kungfu.py:    elif args.kf_optimizer == 'sync-sgd-nccl':
benchmarks/system/benchmark_kungfu.py:        opt = SynchronousSGDOptimizer(opt, nccl=True, nccl_fusion=args.fuse)
benchmarks/system/benchmark_kungfu.py:    elif args.kf_optimizer == 'sync-sgd-hierarchical-nccl':
benchmarks/system/benchmark_kungfu.py:                                      nccl=True,
benchmarks/system/benchmark_kungfu.py:                                      nccl_fusion=args.fuse,
benchmarks/system/benchmark_kungfu.py:                                      hierarchical_nccl=True)
benchmarks/system/benchmark_kungfu.py:device = '/gpu:0' if args.cuda else 'CPU'
benchmarks/system/benchmark_ps.py:        # Set the GPU
benchmarks/system/benchmark_ps.py:        use_cuda = not FLAGS.no_cuda
benchmarks/system/benchmark_ps.py:        if use_cuda:
benchmarks/system/benchmark_ps.py:            config.gpu_options.allow_growth = True
benchmarks/system/benchmark_ps.py:            config.gpu_options.visible_device_list = os.environ[
benchmarks/system/benchmark_ps.py:                "CUDA_VISIBLE_DEVICES"]
benchmarks/system/benchmark_ps.py:            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
benchmarks/system/benchmark_ps.py:            config.gpu_options.allow_growth = False
benchmarks/system/benchmark_ps.py:            config.gpu_options.visible_device_list = ''
benchmarks/system/benchmark_ps.py:    # Flags for GPU
benchmarks/system/benchmark_ps.py:    parser.add_argument('--no-cuda',
benchmarks/system/benchmark_ps.py:                        help='disables CUDA training')
benchmarks/system/benchmark_horovod.py:parser.add_argument('--no-cuda',
benchmarks/system/benchmark_horovod.py:                    help='disables CUDA training')
benchmarks/system/benchmark_horovod.py:args.cuda = not args.no_cuda
benchmarks/system/benchmark_horovod.py:# Horovod: pin GPU to be used to process local rank (one GPU per process)
benchmarks/system/benchmark_horovod.py:if args.cuda:
benchmarks/system/benchmark_horovod.py:    config.gpu_options.allow_growth = True
benchmarks/system/benchmark_horovod.py:    config.gpu_options.visible_device_list = str(hvd.local_rank())
benchmarks/system/benchmark_horovod.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
benchmarks/system/benchmark_horovod.py:    config.gpu_options.allow_growth = False
benchmarks/system/benchmark_horovod.py:    config.gpu_options.visible_device_list = ''
benchmarks/system/benchmark_horovod.py:device = 'GPU' if args.cuda else 'CPU'
benchmarks/system/benchmark_horovod.py:        attrs['nccl_built'] = hvd.nccl_built()
benchmarks/system/README.md:We assume the benchmark runs on a server with 4 GPUs.
benchmarks/system/README.md:CUDA_VISIBLE_DEVICES="" python3 benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=ps --task_index=0 &
benchmarks/system/README.md:# Start workers on different GPUs
benchmarks/system/README.md:CUDA_VISIBLE_DEVICES="0" python3 benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=0 --model=ResNet50 --batch-size=64 &
benchmarks/system/README.md:CUDA_VISIBLE_DEVICES="1" python3 benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=1 --model=ResNet50 --batch-size=64 &
benchmarks/system/README.md:CUDA_VISIBLE_DEVICES="2" python3 benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=2 --model=ResNet50 --batch-size=64 &
benchmarks/system/README.md:CUDA_VISIBLE_DEVICES="3" python3 benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=3 --model=ResNet50 --batch-size=64 &
benchmarks/system/benchmark_kungfu_elastic.py:parser.add_argument('--no-cuda',
benchmarks/system/benchmark_kungfu_elastic.py:                    help='disables CUDA training')
benchmarks/system/benchmark_kungfu_elastic.py:args.cuda = not args.no_cuda
benchmarks/system/benchmark_kungfu_elastic.py:if args.cuda:
benchmarks/system/benchmark_kungfu_elastic.py:    config.gpu_options.allow_growth = True
benchmarks/system/benchmark_kungfu_elastic.py:    from kungfu.python import _get_cuda_index
benchmarks/system/benchmark_kungfu_elastic.py:    config.gpu_options.visible_device_list = str(_get_cuda_index())
benchmarks/system/benchmark_kungfu_elastic.py:    config.gpu_options.allow_growth = False
benchmarks/system/benchmark_kungfu_elastic.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
benchmarks/system/benchmark_kungfu_elastic.py:    config.gpu_options.visible_device_list = ''
benchmarks/system/benchmark_kungfu_elastic.py:    elif args.kf_optimizer == 'sync-sgd-nccl':
benchmarks/system/benchmark_kungfu_elastic.py:        opt = SynchronousSGDOptimizer(opt, nccl=True, nccl_fusion=args.fuse)
benchmarks/system/benchmark_kungfu_elastic.py:device = '/gpu:0' if args.cuda else 'CPU'
benchmarks/system/benchmark_kungfu_torch.py:parser.add_argument('--no-cuda',
benchmarks/system/benchmark_kungfu_torch.py:                    help='disables CUDA training')
benchmarks/system/benchmark_kungfu_torch.py:args.cuda = not args.no_cuda and torch.cuda.is_available()
benchmarks/system/benchmark_kungfu_torch.py:if args.cuda:
benchmarks/system/benchmark_kungfu_torch.py:    torch.cuda.set_device(kf.get_cuda_index())
benchmarks/system/benchmark_kungfu_torch.py:if args.cuda:
benchmarks/system/benchmark_kungfu_torch.py:    # Move model to GPU.
benchmarks/system/benchmark_kungfu_torch.py:    model.cuda()
benchmarks/system/benchmark_kungfu_torch.py:    # If using GPU Adasum allreduce, scale learning rate by local_size.
benchmarks/system/benchmark_kungfu_torch.py:    if args.use_adasum and kf.nccl_built():
benchmarks/system/benchmark_kungfu_torch.py:if args.cuda:
benchmarks/system/benchmark_kungfu_torch.py:    data, target = data.cuda(), target.cuda()
benchmarks/system/benchmark_kungfu_torch.py:device = 'GPU' if args.cuda else 'CPU'
benchmarks/system/benchmark_kungfu_tf2.py:parser.add_argument('--no-cuda',
benchmarks/system/benchmark_kungfu_tf2.py:                    help='disables CUDA training')
benchmarks/system/benchmark_kungfu_tf2.py:args.cuda = not args.no_cuda
benchmarks/system/benchmark_kungfu_tf2.py:device = 'GPU' if args.cuda else 'CPU'
experiments/tfkeras/tf-keras.go:	SyncSgdNccl KFOptimizer = `sync-sgd-nccl`
experiments/tfkeras/tf-keras.go:	SyncSgdNccl,
experiments/tfkeras/tf-keras.go:			SyncSgdNccl,
experiments/cmd/kungfu-remote-install/kungfu-remote-install.go:	enableNCCL *bool
experiments/cmd/kungfu-remote-install/kungfu-remote-install.go:	enableNCCL: flag.Bool("nccl", false, ""),
experiments/cmd/kungfu-remote-install/kungfu-remote-install.go:			Env(`KUNGFU_ENABLE_NCCL`, str(b2i(*flg.enableNCCL))).
docker/Dockerfile.tf-gpu:FROM tensorflow/tensorflow:1.13.1-gpu-py3
docker/Dockerfile.tf-gpu:RUN ldconfig /usr/local/cuda-10.0/targets/x86_64-linux/lib/stubs && pip3 install --no-index -U .
setup_mindspore.py:        use_nccl = os.getenv('KUNGFU_ENABLE_NCCL')
setup_mindspore.py:        if use_nccl:
setup_mindspore.py:            cmake_args.append(cmake_flag('KUNGFU_ENABLE_NCCL', use_nccl))
setup_mindspore.py:            nccl_home = os.getenv('NCCL_HOME')
setup_mindspore.py:            if nccl_home:
setup_mindspore.py:                cmake_args.append(cmake_flag('NCCL_HOME', nccl_home))
configure:ENABLE_NCCL=0
configure:NCCL_HOME=$HOME/local/nccl
configure:        --enable-nccl)
configure:            ENABLE_NCCL=1
configure:        --with-nccl=*)
configure:            NCCL_HOME="${i#*=}"
configure:            ENABLE_NCCL=1
configure:    if [ ${ENABLE_NCCL} -gt 0 ]; then
configure:        add_cmake_flag NCCL_HOME ${NCCL_HOME}
configure:        add_cmake_flag KUNGFU_ENABLE_NCCL 1
setup_pytorch.py:def find_cuda():
setup_pytorch.py:    # TODO: find cuda
setup_pytorch.py:    return '/usr/local/cuda'
setup_pytorch.py:    with_cuda = None
setup_pytorch.py:    if torch.cuda.is_available():
setup_pytorch.py:        srcs += glob.glob('srcs/cpp/src/cuda/*.cpp')
setup_pytorch.py:        srcs += glob.glob('srcs/cpp/src/torch/ops/cuda/*.cpp')
setup_pytorch.py:        with_cuda = True
setup_pytorch.py:        include_dirs += [os.path.join(find_cuda(), 'include')]
setup_pytorch.py:        library_dirs += [os.path.join(find_cuda(), 'lib64')]
setup_pytorch.py:        libraries += ['cudart']
setup_pytorch.py:        srcs += ['srcs/cpp/src/torch/module_cuda.cpp']
setup_pytorch.py:        with_cuda=with_cuda,
examples/keras_mnist.py:config.gpu_options.allow_growth = True
examples/keras_mnist.py:# KungFu: adjust number of epochs based on number of GPUs.
examples/keras_mnist.py:# KungFu: adjust learning rate based on number of GPUs.
examples/tf2_mnist_gradient_tape.py:# KungFu: adjust learning rate based on number of GPUs.
examples/tf2_mnist_gradient_tape.py:# KungFu: adjust number of steps based on number of GPUs.
examples/tf2_mnist_keras_dataset.py:# KungFu: adjust learning rate based on number of GPUs.
examples/tf2_mnist_keras_dataset.py:# KungFu: adjust number of steps based on number of GPUs.
examples/torch_elastic/torch_mnist_example.py:    use_cuda = not args.no_cuda and torch.cuda.is_available()
examples/torch_elastic/torch_mnist_example.py:    config = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
examples/torch_elastic/torch_mnist_example.py:    p.add_argument('--no-cuda', action='store_true', default=False)
examples/torch_elastic/torch_mnist_example.py:    use_cuda = not args.no_cuda and torch.cuda.is_available()
examples/torch_elastic/torch_mnist_example.py:    device = torch.device("cuda" if use_cuda else "cpu")
examples/torch_mnist_example.py:    parser.add_argument('--no-cuda',
examples/torch_mnist_example.py:                        help='disables CUDA training')
examples/torch_mnist_example.py:    use_cuda = not args.no_cuda and torch.cuda.is_available()
examples/torch_mnist_example.py:    device = torch.device("cuda" if use_cuda else "cpu")
examples/torch_mnist_example.py:    config = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
CONTRIBUTING.md:docker build -f docker/Dockerfile.tf-gpu -t kungfu:gpu .
CONTRIBUTING.md:docker run -it kungfu:gpu
CONTRIBUTING.md:## Use NVIDIA NCCL
CONTRIBUTING.md:KungFu can use [NCCL](https://developer.nvidia.com/nccl) to leverage GPU-GPU direct communication.
CONTRIBUTING.md:# uncomment to use your own NCCL
CONTRIBUTING.md:# export NCCL_HOME=$HOME/local/nccl
CONTRIBUTING.md:KUNGFU_ENABLE_NCCL=1 pip3 install .
CONTRIBUTING.md:config.gpu_options.allow_growth = True
CONTRIBUTING.md:from kungfu.python import _get_cuda_index
CONTRIBUTING.md:config.gpu_options.visible_device_list = str(_get_cuda_index())
CONTRIBUTING.md:# export NCCL_DEBUG=INFO # uncomment to enable
setup_tensorflow.py:        use_nccl = os.getenv('KUNGFU_ENABLE_NCCL')
setup_tensorflow.py:        if use_nccl:
setup_tensorflow.py:            cmake_args.append(cmake_flag('KUNGFU_ENABLE_NCCL', use_nccl))
setup_tensorflow.py:            nccl_home = os.getenv('NCCL_HOME')
setup_tensorflow.py:            if nccl_home:
setup_tensorflow.py:                cmake_args.append(cmake_flag('NCCL_HOME', nccl_home))

```
