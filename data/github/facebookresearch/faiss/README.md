# https://github.com/facebookresearch/faiss

```console
Doxyfile:EXCLUDE                = gpu/test
CHANGELOG.md:- ROCm support (#3462)
CHANGELOG.md:- Implement reconstruct_n for GPU IVFFlat indexes (#3338)
CHANGELOG.md:- Change index_cpu_to_gpu to throw for indices not implemented on GPU (#3336)
CHANGELOG.md:- Throw when attempting to move IndexPQ to GPU (#3328)
CHANGELOG.md:- Added a new conda package faiss-gpu-raft alongside faiss-cpu and faiss-gpu
CHANGELOG.md:- Integrated IVF-Flat and IVF-PQ implementations in faiss-gpu-raft from RAFT by Nvidia [thanks Corey Nolet and Tarang Jain]
CHANGELOG.md:- Added support for CUDA 12
CHANGELOG.md:- Fixed DeviceVector reallocations in Faiss GPU
CHANGELOG.md:- Fixed warp synchronous behavior in Faiss GPU CUDA 12
CHANGELOG.md:- 64-bit indexing arithmetic support in FAISS GPU
CHANGELOG.md:- CUDA 10 is no longer supported in precompiled packages
CHANGELOG.md:- Faiss GPU IVF large query batch fix
CHANGELOG.md:- Support LSQ on GPU (by @KinglittleQ)
CHANGELOG.md:- The order of xb an xq was different between `faiss.knn` and `faiss.knn_gpu`.
CHANGELOG.md:- Arbitrary dimensions per sub-quantizer now allowed for `GpuIndexIVFPQ`.
CHANGELOG.md:- Brute-force kNN on GPU (`bfKnn`) now accepts `int32` indices.
CHANGELOG.md:- Support alternative distances on GPU for GpuIndexFlat, including L1, Linf and
CHANGELOG.md:- Support METRIC_INNER_PRODUCT for GpuIndexIVFPQ.
CHANGELOG.md:- Support float16 coarse quantizer for GpuIndexIVFFlat and GpuIndexIVFPQ. GPU
CHANGELOG.md:- Removed support for useFloat16Accumulator for accumulators on GPU (all
CHANGELOG.md:- Fixed GpuCloner (some fields were not copied, default to no precomputed tables
CHANGELOG.md:- ScalarQuantizer support for GPU, see gpu/GpuIndexIVFScalarQuantizer.h. This is
CHANGELOG.md:particularly useful as GPU memory is often less abundant than CPU.
CHANGELOG.md:- The Python KMeans object can be used to use the GPU directly, just add
CHANGELOG.md:gpu=True to the constuctor see gpu/test/test_gpu_index.py test TestGPUKmeans.
CHANGELOG.md:- Support for k = 2048 search on GPU (instead of 1024).
CHANGELOG.md:- Simplified build system (with --with-cuda/--with-cuda-arch options).
CHANGELOG.md:- Most CUDA mem alloc failures now throw exceptions instead of terminating on an
CHANGELOG.md:- Conda packages now depend on the cudatoolkit packages, which fixes some
CHANGELOG.md:interferences with pytorch. Consequentially, faiss-gpu should now be installed
CHANGELOG.md:by conda install -c pytorch faiss-gpu cudatoolkit=10.0.
CHANGELOG.md:- New GpuIndexBinaryFlat index.
CHANGELOG.md:- GpuIndexIVFFlat issues for float32 with 64 / 128 dims.
CHANGELOG.md:- Sharding of flat indexes on GPU with index_cpu_to_gpu_multiple.
CHANGELOG.md:- k-selection for CUDA 9.
CHANGELOG.md:- Extended tutorial to GPU indices.
tests/test_contrib.py:        # decimal = 4 required when run on GPU
tests/test_contrib.py:            xq, ds.database_iterator(bs=100), threshold, ngpu=0,
tests/CMakeLists.txt:  $<$<BOOL:${FAISS_ENABLE_ROCM}>:hip::host>
README.md:Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python/numpy. Some of the most useful algorithms are implemented on the GPU. It is developed primarily at Meta's [Fundamental AI Research](https://ai.facebook.com/) group.
README.md:The GPU implementation can accept input from either CPU or GPU memory. On a server with GPUs, the GPU indexes can be used a drop-in replacement for the CPU indexes (e.g., replace `IndexFlatL2` with `GpuIndexFlatL2`) and copies to/from GPU memory are handled automatically. Results will be faster however if both input and output remain resident on the GPU. Both single and multi-GPU usage is supported.
README.md:Faiss comes with precompiled libraries for Anaconda in Python, see [faiss-cpu](https://anaconda.org/pytorch/faiss-cpu) and [faiss-gpu](https://anaconda.org/pytorch/faiss-gpu). The library is mostly implemented in C++, the only dependency is a [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) implementation. Optional GPU support is provided via CUDA or AMD ROCm, and the Python interface is also optional. It compiles with cmake. See [INSTALL.md](INSTALL.md) for details.
README.md:The optional GPU implementation provides what is likely (as of March 2017) the fastest exact and approximate (compressed-domain) nearest neighbor search implementation for high-dimensional vectors, fastest Lloyd's k-means, and fastest small k-selection algorithm known. [The implementation is detailed here](https://arxiv.org/abs/1702.08734).
README.md:- to reproduce results from our research papers, [Polysemous codes](https://arxiv.org/abs/1609.01882) and [Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734), refer to the [benchmarks README](benchs/README.md). For [
README.md:- [Jeff Johnson](https://github.com/wickedfoo) implemented all of the GPU Faiss
README.md:For the GPU version of Faiss, please cite:
README.md:  title={Billion-scale similarity search with {GPUs}},
benchs/bench_gpu_sift1m.py:# we need only a StandardGpuResources per GPU
benchs/bench_gpu_sift1m.py:res = faiss.StandardGpuResources()
benchs/bench_gpu_sift1m.py:flat_config = faiss.GpuIndexFlatConfig()
benchs/bench_gpu_sift1m.py:index = faiss.GpuIndexFlatL2(res, d, flat_config)
benchs/bench_gpu_sift1m.py:co = faiss.GpuClonerOptions()
benchs/bench_gpu_sift1m.py:index = faiss.index_cpu_to_gpu(res, 0, index, co)
benchs/kmeans_mnist.py:ngpu = int(sys.argv[2])
benchs/kmeans_mnist.py:def train_kmeans(x, k, ngpu):
benchs/kmeans_mnist.py:    "Runs kmeans on one or several GPUs"
benchs/kmeans_mnist.py:    res = [faiss.StandardGpuResources() for i in range(ngpu)]
benchs/kmeans_mnist.py:    for i in range(ngpu):
benchs/kmeans_mnist.py:        cfg = faiss.GpuIndexFlatConfig()
benchs/kmeans_mnist.py:    if ngpu == 1:
benchs/kmeans_mnist.py:        index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
benchs/kmeans_mnist.py:        indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i])
benchs/kmeans_mnist.py:                   for i in range(ngpu)]
benchs/kmeans_mnist.py:train_kmeans(x, k, ngpu)
benchs/bench_fw/index.py:        elif name == "lsq_gpu":
benchs/bench_fw/index.py:                ngpus = faiss.get_num_gpus()
benchs/bench_fw/index.py:                icm_encoder_factory = faiss.GpuIcmEncoderFactory(ngpus)
benchs/bench_fw/index.py:                "lsq_gpu",
benchs/bench_fw/benchmark_io.py:            gpus_per_node=8,
benchs/bench_fw/descriptors.py:            kmeans = faiss.Kmeans(d=x.shape[1], k=k, gpu=True)
benchs/bench_fw/descriptors.py:            if name == "lsq_gpu" and val == 0:
benchs/bench_ivfpq_raft.py:# Copyright (c) 2023, NVIDIA CORPORATION.
benchs/bench_ivfpq_raft.py:res = faiss.StandardGpuResources()
benchs/bench_ivfpq_raft.py:mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource())
benchs/bench_ivfpq_raft.py:    co = faiss.GpuMultipleClonerOptions()
benchs/bench_ivfpq_raft.py:    index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
benchs/bench_ivfpq_raft.py:    index_gpu.train(trainVecs)
benchs/bench_ivfpq_raft.py:print("GPU Train Benchmarks")
benchs/bench_ivfpq_raft.py:raft_gpu_train_time = bench_train_milliseconds(index, xt, True)
benchs/bench_ivfpq_raft.py:    print("Method: IVFPQ, Operation: TRAIN, dim: %d, n_centroids %d, numSubQuantizers %d, bitsPerCode %d, numTrain: %d, RAFT enabled GPU train time: %.3f milliseconds" % (
benchs/bench_ivfpq_raft.py:        n_cols, nlist, M, args.bits_per_code, n_train, raft_gpu_train_time))
benchs/bench_ivfpq_raft.py:    classical_gpu_train_time = bench_train_milliseconds(
benchs/bench_ivfpq_raft.py:    print("Method: IVFPQ, Operation: TRAIN, dim: %d, n_centroids %d, numSubQuantizers %d, bitsPerCode %d, numTrain: %d, classical GPU train time: %.3f milliseconds, RAFT enabled GPU train time: %.3f milliseconds" % (
benchs/bench_ivfpq_raft.py:        n_cols, nlist, M, args.bits_per_code, n_train, classical_gpu_train_time, raft_gpu_train_time))
benchs/bench_ivfpq_raft.py:    co = faiss.GpuMultipleClonerOptions()
benchs/bench_ivfpq_raft.py:    index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
benchs/bench_ivfpq_raft.py:    index_gpu.copyFrom(index)
benchs/bench_ivfpq_raft.py:    index_gpu.add(addVecs)
benchs/bench_ivfpq_raft.py:print("GPU Add Benchmarks")
benchs/bench_ivfpq_raft.py:raft_gpu_add_time = bench_add_milliseconds(index, xb, True)
benchs/bench_ivfpq_raft.py:    print("Method: IVFPQ, Operation: ADD, dim: %d, n_centroids %d numSubQuantizers %d, bitsPerCode %d, numAdd %d, RAFT enabled GPU add time: %.3f milliseconds" % (
benchs/bench_ivfpq_raft.py:        n_cols, nlist, M, args.bits_per_code, n_rows, raft_gpu_add_time))
benchs/bench_ivfpq_raft.py:    classical_gpu_add_time = bench_add_milliseconds(
benchs/bench_ivfpq_raft.py:    print("Method: IVFFPQ, Operation: ADD, dim: %d, n_centroids %d, numSubQuantizers %d, bitsPerCode %d, numAdd %d, classical GPU add time: %.3f milliseconds, RAFT enabled GPU add time: %.3f milliseconds" % (
benchs/bench_ivfpq_raft.py:        n_cols, nlist, M, args.bits_per_code, n_rows, classical_gpu_add_time, raft_gpu_add_time))
benchs/bench_ivfpq_raft.py:    co = faiss.GpuMultipleClonerOptions()
benchs/bench_ivfpq_raft.py:    index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
benchs/bench_ivfpq_raft.py:    index_gpu.copyFrom(index)
benchs/bench_ivfpq_raft.py:    index_gpu.add(addVecs)
benchs/bench_ivfpq_raft.py:    index_gpu.nprobe = nprobe
benchs/bench_ivfpq_raft.py:    index_gpu.search(queryVecs, k)
benchs/bench_ivfpq_raft.py:    print("GPU Search Benchmarks")
benchs/bench_ivfpq_raft.py:        raft_gpu_search_time = bench_search_milliseconds(
benchs/bench_ivfpq_raft.py:            print("Method: IVFPQ, Operation: SEARCH, dim: %d, n_centroids: %d, numSubQuantizers %d, bitsPerCode %d, numVecs: %d, numQuery: %d, nprobe: %d, k: %d, RAFT enabled GPU search time: %.3f milliseconds" % (
benchs/bench_ivfpq_raft.py:                n_cols, nlist, M, args.bits_per_code, n_add, n_rows, args.nprobe, args.k, raft_gpu_search_time))
benchs/bench_ivfpq_raft.py:            classical_gpu_search_time = bench_search_milliseconds(
benchs/bench_ivfpq_raft.py:            print("Method: IVFPQ, Operation: SEARCH, dim: %d, n_centroids: %d, numSubQuantizers %d, bitsPerCode %d, numVecs: %d, numQuery: %d, nprobe: %d, k: %d, classical GPU search time: %.3f milliseconds, RAFT enabled GPU search time: %.3f milliseconds" % (
benchs/bench_ivfpq_raft.py:                n_cols, nlist, M, args.bits_per_code, n_add, n_rows, args.nprobe, args.k, classical_gpu_search_time, raft_gpu_search_time))
benchs/bench_ivfflat_raft.py:# Copyright (c) 2023, NVIDIA CORPORATION.
benchs/bench_ivfflat_raft.py:   help='whether to benchmark train operation on GPU index')
benchs/bench_ivfflat_raft.py:   help='whether to benchmark add operation on GPU index')
benchs/bench_ivfflat_raft.py:   help='whether to benchmark search operation on GPU index')
benchs/bench_ivfflat_raft.py:res = faiss.StandardGpuResources()
benchs/bench_ivfflat_raft.py:mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource())
benchs/bench_ivfflat_raft.py:    co = faiss.GpuMultipleClonerOptions()
benchs/bench_ivfflat_raft.py:    index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
benchs/bench_ivfflat_raft.py:    index_gpu.train(trainVecs)
benchs/bench_ivfflat_raft.py:    print("GPU Train Benchmarks")
benchs/bench_ivfflat_raft.py:            raft_gpu_train_time = bench_train_milliseconds(
benchs/bench_ivfflat_raft.py:                print("Method: IVFFlat, Operation: TRAIN, dim: %d, n_centroids %d, numTrain: %d, RAFT enabled GPU train time: %.3f milliseconds" % (
benchs/bench_ivfflat_raft.py:                    n_cols, args.n_centroids, n_rows, raft_gpu_train_time))
benchs/bench_ivfflat_raft.py:                classical_gpu_train_time = bench_train_milliseconds(
benchs/bench_ivfflat_raft.py:                print("Method: IVFFlat, Operation: TRAIN, dim: %d, n_centroids %d, numTrain: %d, classical GPU train time: %.3f milliseconds, RAFT enabled GPU train time: %.3f milliseconds" % (
benchs/bench_ivfflat_raft.py:                    n_cols, args.n_centroids, n_rows, classical_gpu_train_time, raft_gpu_train_time))
benchs/bench_ivfflat_raft.py:    co = faiss.GpuMultipleClonerOptions()
benchs/bench_ivfflat_raft.py:    index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
benchs/bench_ivfflat_raft.py:    index_gpu.copyFrom(index)
benchs/bench_ivfflat_raft.py:    index_gpu.add(addVecs)
benchs/bench_ivfflat_raft.py:    print("GPU Add Benchmarks")
benchs/bench_ivfflat_raft.py:            raft_gpu_add_time = bench_add_milliseconds(index, addVecs, True)
benchs/bench_ivfflat_raft.py:                print("Method: IVFFlat, Operation: ADD, dim: %d, n_centroids %d, numAdd: %d, RAFT enabled GPU add time: %.3f milliseconds" % (
benchs/bench_ivfflat_raft.py:                    n_train, n_rows, n_cols, args.n_centroids, raft_gpu_add_time))
benchs/bench_ivfflat_raft.py:                classical_gpu_add_time = bench_add_milliseconds(
benchs/bench_ivfflat_raft.py:                print("Method: IVFFlat, Operation: ADD, dim: %d, n_centroids %d, numAdd: %d, classical GPU add time: %.3f milliseconds, RAFT enabled GPU add time: %.3f milliseconds" % (
benchs/bench_ivfflat_raft.py:                    n_train, n_rows, n_cols, args.n_centroids, classical_gpu_add_time, raft_gpu_add_time))
benchs/bench_ivfflat_raft.py:    co = faiss.GpuMultipleClonerOptions()
benchs/bench_ivfflat_raft.py:    index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
benchs/bench_ivfflat_raft.py:    index_gpu.copyFrom(index)
benchs/bench_ivfflat_raft.py:    index_gpu.add(addVecs)
benchs/bench_ivfflat_raft.py:    index_gpu.nprobe = nprobe
benchs/bench_ivfflat_raft.py:    index_gpu.search(queryVecs, k)
benchs/bench_ivfflat_raft.py:    print("GPU Search Benchmarks")
benchs/bench_ivfflat_raft.py:            raft_gpu_search_time = bench_search_milliseconds(
benchs/bench_ivfflat_raft.py:                print("Method: IVFFlat, Operation: SEARCH, dim: %d, n_centroids: %d, numVecs: %d, numQuery: %d, nprobe: %d, k: %d, RAFT enabled GPU search time: %.3f milliseconds" % (
benchs/bench_ivfflat_raft.py:                    n_cols, args.n_centroids, n_add, n_rows, args.nprobe, args.k, raft_gpu_search_time))
benchs/bench_ivfflat_raft.py:                classical_gpu_search_time = bench_search_milliseconds(
benchs/bench_ivfflat_raft.py:                print("Method: IVFFlat, Operation: SEARCH, dim: %d, n_centroids: %d, numVecs: %d, numQuery: %d, nprobe: %d, k: %d, classical GPU search time: %.3f milliseconds, RAFT enabled GPU search time: %.3f milliseconds" % (
benchs/bench_ivfflat_raft.py:                    n_cols, args.n_centroids, n_add, n_rows, args.nprobe, args.k, classical_gpu_search_time, raft_gpu_search_time))
benchs/bench_ivfflat_raft.py:    # Avoid classical GPU Benchmarks for large datasets because of OOM for more than 500000 queries and/or large dims as well as for large k
benchs/bench_ivfflat_raft.py:            raft_gpu_search_time = bench_search_milliseconds(
benchs/bench_ivfflat_raft.py:            print("Method: IVFFlat, Operation: SEARCH, numTrain: %d, dim: %d, n_centroids: %d, numVecs: %d, numQuery: %d, nprobe: %d, k: %d, RAFT enabled GPU search time: %.3f milliseconds" % (
benchs/bench_ivfflat_raft.py:                n_cols, args.n_centroids, n_add, n_rows, args.nprobe, args.k, raft_gpu_search_time))
benchs/bench_all_ivf/bench_kmeans.py:ngpu = faiss.get_num_gpus()
benchs/bench_all_ivf/bench_kmeans.py:print("nb GPUs:", ngpu)
benchs/bench_all_ivf/bench_kmeans.py:if ngpu > 0:
benchs/bench_all_ivf/bench_kmeans.py:    print("moving index to GPU")
benchs/bench_all_ivf/bench_kmeans.py:    index = faiss.index_cpu_to_all_gpus(index)
benchs/bench_all_ivf/bench_all_ivf.py:        if args.train_on_gpu:
benchs/bench_all_ivf/bench_all_ivf.py:            print("add a training index on GPU")
benchs/bench_all_ivf/bench_all_ivf.py:            train_index = faiss.index_cpu_to_all_gpus(
benchs/bench_all_ivf/bench_all_ivf.py:    aa('--train_on_gpu', default=False, action='store_true',
benchs/bench_all_ivf/bench_all_ivf.py:        help='do training on GPU')
benchs/bench_all_ivf/README.md:The code depends on Faiss and can use 1 to 8 GPUs to do the k-means clustering for large vocabularies. 
benchs/bench_all_ivf/run_on_cluster_generic.bash:#    run_on_8gpu: runs a command on one machine with 8 GPUs
benchs/bench_all_ivf/run_on_cluster_generic.bash:    run_on "--cpus-per-task=80 --gres=gpu:0 --mem=500G --time=70:00:00 --partition=priority" "$@"
benchs/bench_all_ivf/run_on_cluster_generic.bash:    run_on "--cpus-per-task=80 --gres=gpu:2 --mem=100G --time=1:00:00 --partition=priority" "$@"
benchs/bench_all_ivf/run_on_cluster_generic.bash:    run_on "--cpus-per-task=80 --gres=gpu:2 --mem=100G --time=3:00:00 --partition=priority" "$@"
benchs/bench_all_ivf/run_on_cluster_generic.bash:function run_on_4gpu_3h {
benchs/bench_all_ivf/run_on_cluster_generic.bash:    run_on "--cpus-per-task=40 --gres=gpu:4 --mem=100G --time=3:00:00 --partition=priority" "$@"
benchs/bench_all_ivf/run_on_cluster_generic.bash:function run_on_8gpu () {
benchs/bench_all_ivf/run_on_cluster_generic.bash:    run_on "--cpus-per-task=80 --gres=gpu:8 --mem=100G --time=70:00:00 --partition=priority" "$@"
benchs/bench_all_ivf/run_on_cluster_generic.bash:# precompute centroids on GPU for large vocabularies
benchs/bench_all_ivf/run_on_cluster_generic.bash:        run_on_4gpu_3h $key.e \
benchs/bench_all_ivf/run_on_cluster_generic.bash:                --train_on_gpu
benchs/bench_hybrid_cpu_gpu.py:class ShardedGPUIndex:
benchs/bench_hybrid_cpu_gpu.py:    Multiple GPU indexes, each on its GPU, with a common coarse quantizer.
benchs/bench_hybrid_cpu_gpu.py:        ngpu = index.count()
benchs/bench_hybrid_cpu_gpu.py:        self.pool = ThreadPool(ngpu)
benchs/bench_hybrid_cpu_gpu.py:        ngpu = index.count()
benchs/bench_hybrid_cpu_gpu.py:        Dall = np.empty((ngpu, nq, k), dtype='float32')
benchs/bench_hybrid_cpu_gpu.py:        Iall = np.empty((ngpu, nq, k), dtype='int64')
benchs/bench_hybrid_cpu_gpu.py:                gpu_index = faiss.downcast_index(index.at(rank))
benchs/bench_hybrid_cpu_gpu.py:                Dall[rank], Iall[rank] = gpu_index.search_preassigned(
benchs/bench_hybrid_cpu_gpu.py:            list(self.pool.map(do_search, range(ngpu)))
benchs/bench_hybrid_cpu_gpu.py:                gpu_index = faiss.downcast_index(index.at(rank))
benchs/bench_hybrid_cpu_gpu.py:                    gpu_index.search_preassigned(xq[i0:i0 + bs], k, Iq, Dq)
benchs/bench_hybrid_cpu_gpu.py:                    range(ngpu)
benchs/bench_hybrid_cpu_gpu.py:    """ extract the IVF sub-index from the index, supporting GpuIndexes
benchs/bench_hybrid_cpu_gpu.py:        if isinstance(index, faiss.GpuIndexIVF):
benchs/bench_hybrid_cpu_gpu.py:    if index.__class__ == ShardedGPUIndex:
benchs/bench_hybrid_cpu_gpu.py:       help='GPU cloner options')
benchs/bench_hybrid_cpu_gpu.py:       help='GPU cloner options')
benchs/bench_hybrid_cpu_gpu.py:       help='GPU cloner options')
benchs/bench_hybrid_cpu_gpu.py:            "cpu", "gpu", "gpu_flat_quantizer",
benchs/bench_hybrid_cpu_gpu.py:            "cpu_flat_gpu_quantizer", "gpu_tiled", "gpu_ivf_quantizer",
benchs/bench_hybrid_cpu_gpu.py:            "multi_gpu", "multi_gpu_flat_quantizer",
benchs/bench_hybrid_cpu_gpu.py:            "multi_gpu_sharded", "multi_gpu_flat_quantizer_sharded",
benchs/bench_hybrid_cpu_gpu.py:            "multi_gpu_sharded1", "multi_gpu_sharded1_flat",
benchs/bench_hybrid_cpu_gpu.py:            "multi_gpu_sharded1_ivf",
benchs/bench_hybrid_cpu_gpu.py:            "multi_gpu_Csharded1", "multi_gpu_Csharded1_flat",
benchs/bench_hybrid_cpu_gpu.py:            "multi_gpu_Csharded1_ivf",
benchs/bench_hybrid_cpu_gpu.py:       help="batch size for tiled CPU / GPU computation (-1= no tiling)")
benchs/bench_hybrid_cpu_gpu.py:        os.system("nvidia-smi")
benchs/bench_hybrid_cpu_gpu.py:    print("Faiss nb GPUs:", faiss.get_num_gpus())
benchs/bench_hybrid_cpu_gpu.py:    # prepare options for GPU clone
benchs/bench_hybrid_cpu_gpu.py:    co = faiss.GpuMultipleClonerOptions()
benchs/bench_hybrid_cpu_gpu.py:    elif args.search_type == "gpu":
benchs/bench_hybrid_cpu_gpu.py:        print("move index to 1 GPU")
benchs/bench_hybrid_cpu_gpu.py:        res = faiss.StandardGpuResources()
benchs/bench_hybrid_cpu_gpu.py:        index = faiss.index_cpu_to_gpu(res, 0, index, co)
benchs/bench_hybrid_cpu_gpu.py:    elif args.search_type == "gpu_tiled":
benchs/bench_hybrid_cpu_gpu.py:        print("move index to 1 GPU")
benchs/bench_hybrid_cpu_gpu.py:        res = faiss.StandardGpuResources()
benchs/bench_hybrid_cpu_gpu.py:        index = faiss.index_cpu_to_gpu(res, 0, index, co)
benchs/bench_hybrid_cpu_gpu.py:    elif args.search_type == "gpu_ivf_quantizer":
benchs/bench_hybrid_cpu_gpu.py:        res = faiss.StandardGpuResources()
benchs/bench_hybrid_cpu_gpu.py:            faiss.index_cpu_to_gpu(res, 0, new_quantizer, co),
benchs/bench_hybrid_cpu_gpu.py:            faiss.index_cpu_to_gpu(res, 0, index, co),
benchs/bench_hybrid_cpu_gpu.py:    elif args.search_type == "gpu_flat_quantizer":
benchs/bench_hybrid_cpu_gpu.py:        res = faiss.StandardGpuResources()
benchs/bench_hybrid_cpu_gpu.py:        index = faiss.index_cpu_to_gpu(res, 0, index, co)
benchs/bench_hybrid_cpu_gpu.py:    elif args.search_type == "cpu_flat_gpu_quantizer":
benchs/bench_hybrid_cpu_gpu.py:        res = faiss.StandardGpuResources()
benchs/bench_hybrid_cpu_gpu.py:        quantizer = faiss.index_cpu_to_gpu(res, 0, quantizer, co)
benchs/bench_hybrid_cpu_gpu.py:    elif args.search_type in ("multi_gpu", "multi_gpu_sharded"):
benchs/bench_hybrid_cpu_gpu.py:        print(f"move index to {faiss.get_num_gpus()} GPU")
benchs/bench_hybrid_cpu_gpu.py:        index = faiss.index_cpu_to_all_gpus(index, co=co)
benchs/bench_hybrid_cpu_gpu.py:            "multi_gpu_flat_quantizer", "multi_gpu_flat_quantizer_sharded"):
benchs/bench_hybrid_cpu_gpu.py:        index = faiss.index_cpu_to_all_gpus(index, co=co)
benchs/bench_hybrid_cpu_gpu.py:            "multi_gpu_sharded1", "multi_gpu_sharded1_flat",
benchs/bench_hybrid_cpu_gpu.py:            "multi_gpu_sharded1_ivf"):
benchs/bench_hybrid_cpu_gpu.py:        print(f"move index to {faiss.get_num_gpus()} GPU")
benchs/bench_hybrid_cpu_gpu.py:        gpus = list(range(faiss.get_num_gpus()))
benchs/bench_hybrid_cpu_gpu.py:        res = [faiss.StandardGpuResources() for _ in gpus]
benchs/bench_hybrid_cpu_gpu.py:        index = faiss.index_cpu_to_gpu_multiple_py(res, index, co, gpus)
benchs/bench_hybrid_cpu_gpu.py:        if args.search_type == "multi_gpu_sharded1":
benchs/bench_hybrid_cpu_gpu.py:            index = ShardedGPUIndex(hnsw_quantizer, index, bs=args.batch_size)
benchs/bench_hybrid_cpu_gpu.py:        elif args.search_type == "multi_gpu_sharded1_ivf":
benchs/bench_hybrid_cpu_gpu.py:            quantizer = faiss.index_cpu_to_gpu_multiple_py(
benchs/bench_hybrid_cpu_gpu.py:                res, quantizer, co, gpus)
benchs/bench_hybrid_cpu_gpu.py:            index = ShardedGPUIndex(quantizer, index, bs=args.batch_size)
benchs/bench_hybrid_cpu_gpu.py:        elif args.search_type == "multi_gpu_sharded1_flat":
benchs/bench_hybrid_cpu_gpu.py:            quantizer = faiss.index_cpu_to_gpu_multiple_py(
benchs/bench_hybrid_cpu_gpu.py:                res, quantizer, co, gpus)
benchs/bench_hybrid_cpu_gpu.py:            index = ShardedGPUIndex(quantizer, index, bs=args.batch_size)
benchs/bench_hybrid_cpu_gpu.py:            "multi_gpu_Csharded1", "multi_gpu_Csharded1_flat",
benchs/bench_hybrid_cpu_gpu.py:            "multi_gpu_Csharded1_ivf"):
benchs/bench_hybrid_cpu_gpu.py:        print(f"move index to {faiss.get_num_gpus()} GPU")
benchs/bench_hybrid_cpu_gpu.py:        if args.search_type == "multi_gpu_Csharded1":
benchs/bench_hybrid_cpu_gpu.py:            index = faiss.index_cpu_to_all_gpus(index, co)
benchs/bench_hybrid_cpu_gpu.py:        elif args.search_type == "multi_gpu_Csharded1_flat":
benchs/bench_hybrid_cpu_gpu.py:            index = faiss.index_cpu_to_all_gpus(index, co)
benchs/bench_hybrid_cpu_gpu.py:        elif args.search_type == "multi_gpu_Csharded1_ivf":
benchs/bench_hybrid_cpu_gpu.py:            index = faiss.index_cpu_to_all_gpus(index, co)
benchs/bench_hybrid_cpu_gpu.py:        "GPU": list(os.popen("nvidia-smi", "r")),
benchs/README.md:   Title = {Billion-scale similarity search with GPUs},
benchs/README.md:## GPU experiments
benchs/README.md:The benchmarks below run 1 or 4 Titan X GPUs and reproduce the results of the "GPU paper". They are also a good starting point on how to use GPU Faiss.
benchs/README.md:See above on how to get SIFT1M into subdirectory sift1M/. The script [`bench_gpu_sift1m.py`](bench_gpu_sift1m.py) reproduces the "exact k-NN time" plot in the ArXiv paper, and the SIFT1M numbers.
benchs/README.md:WARN: increase temp memory to avoid cudaMalloc, or decrease query/add size (alloc 256000000 B, highwater 256000000 B)
benchs/README.md:To index small datasets, it is more efficient to use a `GpuIVFFlat`, which just stores the full vectors in the inverted lists. We did not mention this in the the paper because it is not as scalable. To experiment with this setting, change the `index_factory` string from "IVF4096,PQ64" to "IVF16384,Flat". This gives:
benchs/README.md:The script [`bench_gpu_1bn.py`](bench_gpu_1bn.py) runs multi-gpu searches on the two 1-billion vector datasets we considered. It is more complex than the previous scripts, because it supports many search options and decomposes the dataset build process in Python to exploit the best possible CPU/GPU parallelism and GPU distribution.
benchs/README.md:Even on multiple GPUs, building the 1B datasets can last several hours. It is often a good idea to validate that everything is working fine on smaller datasets like SIFT1M, SIFT2M, etc.
benchs/README.md:The search results on SIFT1B in the "GPU paper" can be obtained with
benchs/README.md:python bench_gpu_1bn.py SIFT1000M OPQ8_32,IVF262144,PQ8 -nnn 10 -ngpu 1 -tempmem $[1536*1024*1024]
benchs/README.md:We use the `-tempmem` option to reduce the temporary memory allocation to 1.5G, otherwise the dataset does not fit in GPU memory
benchs/README.md:The same script generates the GPU search results on Deep1B.
benchs/README.md:python bench_gpu_1bn.py  Deep1B OPQ20_80,IVF262144,PQ20 -nnn 10 -R 2 -ngpu 4 -altadd -noptables -tempmem $[1024*1024*1024]
benchs/README.md:Here we are a bit tight on memory so we disable precomputed tables (`-noptables`) and restrict the amount of temporary memory. The `-altadd` option avoids GPU memory overflows during add.
benchs/README.md:python bench_gpu_1bn.py Deep1B OPQ20_80,IVF262144,PQ20 -nnn 10 -altadd -knngraph  -R 2 -noptables -tempmem $[1<<30] -ngpu 4
benchs/README.md:CPU index contains 1000000000 vectors, move to GPU
benchs/README.md:Copy CPU index to 2 sharded GPU indexes
benchs/README.md:   dispatch to GPUs 0:2
benchs/README.md:  IndexIVFPQ size 500000000 -> GpuIndexIVFPQ indicesOptions=0 usePrecomputed=0 useFloat16=0 reserveVecs=0
benchs/README.md:  IndexIVFPQ size 500000000 -> GpuIndexIVFPQ indicesOptions=0 usePrecomputed=0 useFloat16=0 reserveVecs=0
benchs/README.md:   dispatch to GPUs 2:4
benchs/README.md:  IndexIVFPQ size 500000000 -> GpuIndexIVFPQ indicesOptions=0 usePrecomputed=0 useFloat16=0 reserveVecs=0
benchs/README.md:  IndexIVFPQ size 500000000 -> GpuIndexIVFPQ indicesOptions=0 usePrecomputed=0 useFloat16=0 reserveVecs=0
benchs/README.md:move to GPU done in 151.535 s
benchs/bench_quantizer.py:if 'lsq-gpu' in todo:
benchs/bench_quantizer.py:    ngpus = faiss.get_num_gpus()
benchs/bench_quantizer.py:    lsq.icm_encoder_factory = faiss.GpuIcmEncoderFactory(ngpus)
benchs/bench_quantizer.py:    eval_quantizer(lsq, xb, xt, 'lsq-gpu')
benchs/bench_fw_codecs.py:        (f"LSQ{cs // b}x{b}", [{"encode_ils_iters": 16}], 256 * (2 ** b), {"encode_ils_iters": eii, "lsq_gpu": lg}) 
benchs/bench_fw_codecs.py:        (f"PLSQ{sub}x{cs // sub // b}x{b}", [{"encode_ils_iters": 16}], 256 * (2 ** b), {"encode_ils_iters": eii, "lsq_gpu": lg}) 
benchs/distributed_ondisk/distributed_kmeans.py:from faiss.contrib.clustering import DatasetAssign, DatasetAssignGPU, kmeans
benchs/distributed_ondisk/distributed_kmeans.py:        # same, with GPU
benchs/distributed_ondisk/distributed_kmeans.py:        ngpu = faiss.get_num_gpus()
benchs/distributed_ondisk/distributed_kmeans.py:        print('using %d GPUs' % ngpu)
benchs/distributed_ondisk/distributed_kmeans.py:            DatasetAssignGPU(xx[100000 * i // ngpu: 100000 * (i + 1) // ngpu], i)
benchs/distributed_ondisk/distributed_kmeans.py:            for i in range(ngpu)
benchs/distributed_ondisk/distributed_kmeans.py:    aa('--gpu', default=-2, type=int, help='GPU to use (-2:none, -1: all)')
benchs/distributed_ondisk/distributed_kmeans.py:        if args.gpu == -2:
benchs/distributed_ondisk/distributed_kmeans.py:            print('moving to GPU')
benchs/distributed_ondisk/distributed_kmeans.py:            data = DatasetAssignGPU(x, args.gpu)
benchs/distributed_ondisk/README.md:The object can be a Faiss CPU index, a GPU index or a set of remote GPU or CPU indexes.
benchs/distributed_ondisk/README.md:# same, with GPUs
benchs/distributed_ondisk/README.md:# using all the machine's GPUs
benchs/distributed_ondisk/README.md:# distributed run, with one local server per GPU
benchs/distributed_ondisk/README.md:The test `test_kmeans_2` simulates a distributed run on a single machine by starting one server process per GPU and connecting to the servers via the rpc protocol.
benchs/distributed_ondisk/README.md:asks SLURM for 5 machines with 4 GPUs each with the `srun` command.
benchs/distributed_ondisk/README.md:This is just a matter of using as many machines / GPUs as possible in setting the output centroids with the `--out filename` option.
benchs/distributed_ondisk/run_on_cluster.bash:    # using all the machine's GPUs
benchs/distributed_ondisk/run_on_cluster.bash:           --k $k --gpu -1
benchs/distributed_ondisk/run_on_cluster.bash:    # distrbuted run, with one local server per GPU
benchs/distributed_ondisk/run_on_cluster.bash:    ngpu=$( echo /dev/nvidia? | wc -w )
benchs/distributed_ondisk/run_on_cluster.bash:    for((gpu=0;gpu<ngpu;gpu++)); do
benchs/distributed_ondisk/run_on_cluster.bash:        i0=$((nvec * gpu / ngpu))
benchs/distributed_ondisk/run_on_cluster.bash:        i1=$((nvec * (gpu + 1) / ngpu))
benchs/distributed_ondisk/run_on_cluster.bash:        port=$(( baseport + gpu ))
benchs/distributed_ondisk/run_on_cluster.bash:        echo "start server $gpu for range $i0:$i1"
benchs/distributed_ondisk/run_on_cluster.bash:               --server --gpu $gpu \
benchs/distributed_ondisk/run_on_cluster.bash:         --cpus-per-task=40 --gres=gpu:4 --mem=100G \
benchs/distributed_ondisk/run_on_cluster.bash:              --server --gpu -1 \
benchs/distributed_ondisk/run_on_cluster.bash:              --server --gpu -1 \
benchs/distributed_ondisk/run_on_cluster.bash:         --cpus-per-task=40 --gres=gpu:4 --mem=100G \
benchs/distributed_ondisk/run_on_cluster.bash:             --cpus-per-task=40 --gres=gpu:0 --mem=200G \
benchs/distributed_ondisk/run_on_cluster.bash:             --cpus-per-task=20 --gres=gpu:0 --mem=200G \
benchs/distributed_ondisk/run_on_cluster.bash:         --cpus-per-task=64 --gres=gpu:0 --mem=100G \
benchs/bench_gpu_1bn.py:Usage: bench_gpu_1bn.py dataset indextype [options]
benchs/bench_gpu_1bn.py:indextype: any index type supported by index_factory that runs on GPU.
benchs/bench_gpu_1bn.py:-ngpu ngpu         nb of GPUs to use (default = all)
benchs/bench_gpu_1bn.py:-tempmem N         use N bytes of temporary GPU memory
benchs/bench_gpu_1bn.py:-float16           use 16-bit floats on the GPU side
benchs/bench_gpu_1bn.py:                   on GPU during add. Slightly faster for big datasets on
benchs/bench_gpu_1bn.py:                   slow GPUs
benchs/bench_gpu_1bn.py:                   will be copied across ngpu/R, default R=1)
benchs/bench_gpu_1bn.py:ngpu = faiss.get_num_gpus()
benchs/bench_gpu_1bn.py:    elif a == '-ngpu':      ngpu = int(args.pop(0))
benchs/bench_gpu_1bn.py:cacheroot = '/tmp/bench_gpu_1bn'
benchs/bench_gpu_1bn.py:# Wake up GPUs
benchs/bench_gpu_1bn.py:print("preparing resources for %d GPUs" % ngpu)
benchs/bench_gpu_1bn.py:gpu_resources = []
benchs/bench_gpu_1bn.py:for i in range(ngpu):
benchs/bench_gpu_1bn.py:    res = faiss.StandardGpuResources()
benchs/bench_gpu_1bn.py:    gpu_resources.append(res)
benchs/bench_gpu_1bn.py:    " return vectors of device ids and resources useful for gpu_multiple"
benchs/bench_gpu_1bn.py:    vres = faiss.GpuResourcesVector()
benchs/bench_gpu_1bn.py:        i1 = ngpu
benchs/bench_gpu_1bn.py:        vres.push_back(gpu_resources[i])
benchs/bench_gpu_1bn.py:    db_gt_gpu = faiss.index_cpu_to_gpu_multiple(
benchs/bench_gpu_1bn.py:        db_gt_gpu.add(xsl)
benchs/bench_gpu_1bn.py:        D, I = db_gt_gpu.search(xqs, gt_sl)
benchs/bench_gpu_1bn.py:        db_gt_gpu.reset()
benchs/bench_gpu_1bn.py:    index = faiss.index_cpu_to_gpu_multiple(
benchs/bench_gpu_1bn.py:    a sharded gpu_index that contains the same data. """
benchs/bench_gpu_1bn.py:    co = faiss.GpuMultipleClonerOptions()
benchs/bench_gpu_1bn.py:    gpu_index = faiss.index_cpu_to_gpu_multiple(
benchs/bench_gpu_1bn.py:        gpu_index.add_with_ids(xs, np.arange(i0, i1))
benchs/bench_gpu_1bn.py:        if max_add > 0 and gpu_index.ntotal > max_add:
benchs/bench_gpu_1bn.py:            for i in range(ngpu):
benchs/bench_gpu_1bn.py:                index_src_gpu = faiss.downcast_index(gpu_index.at(i))
benchs/bench_gpu_1bn.py:                index_src = faiss.index_gpu_to_cpu(index_src_gpu)
benchs/bench_gpu_1bn.py:                index_src_gpu.reset()
benchs/bench_gpu_1bn.py:                index_src_gpu.reserveMemory(max_add)
benchs/bench_gpu_1bn.py:            gpu_index.sync_with_shard_indexes()
benchs/bench_gpu_1bn.py:    if hasattr(gpu_index, 'at'):
benchs/bench_gpu_1bn.py:        for i in range(ngpu):
benchs/bench_gpu_1bn.py:            index_src = faiss.index_gpu_to_cpu(gpu_index.at(i))
benchs/bench_gpu_1bn.py:        index_src = faiss.index_gpu_to_cpu(gpu_index)
benchs/bench_gpu_1bn.py:        gpu_index = None
benchs/bench_gpu_1bn.py:    return gpu_index, indexall
benchs/bench_gpu_1bn.py:    # - stage 2: assign on GPU
benchs/bench_gpu_1bn.py:    coarse_quantizer_gpu = faiss.index_cpu_to_gpu_multiple(
benchs/bench_gpu_1bn.py:        _, assign = coarse_quantizer_gpu.search(xs, 1)
benchs/bench_gpu_1bn.py:            gpu_index, indexall = compute_populated_index(preproc)
benchs/bench_gpu_1bn.py:            gpu_index, indexall = compute_populated_index_2(preproc)
benchs/bench_gpu_1bn.py:        gpu_index = None
benchs/bench_gpu_1bn.py:    co = faiss.GpuMultipleClonerOptions()
benchs/bench_gpu_1bn.py:    print("CPU index contains %d vectors, move to GPU" % indexall.ntotal)
benchs/bench_gpu_1bn.py:        if not gpu_index:
benchs/bench_gpu_1bn.py:            print("copying loaded index to GPUs")
benchs/bench_gpu_1bn.py:            index = faiss.index_cpu_to_gpu_multiple(
benchs/bench_gpu_1bn.py:            index = gpu_index
benchs/bench_gpu_1bn.py:        del gpu_index # We override the GPU index
benchs/bench_gpu_1bn.py:        print("Copy CPU index to %d sharded GPU indexes" % replicas)
benchs/bench_gpu_1bn.py:            gpu0 = ngpu * i / replicas
benchs/bench_gpu_1bn.py:            gpu1 = ngpu * (i + 1) / replicas
benchs/bench_gpu_1bn.py:            vres, vdev = make_vres_vdev(gpu0, gpu1)
benchs/bench_gpu_1bn.py:            print("   dispatch to GPUs %d:%d" % (gpu0, gpu1))
benchs/bench_gpu_1bn.py:            index1 = faiss.index_cpu_to_gpu_multiple(
benchs/bench_gpu_1bn.py:    print("move to GPU done in %.3f s" % (time.time() - t0))
benchs/bench_gpu_1bn.py:    ps = faiss.GpuParameterSpace()
tutorial/python/5-Multiple-GPUs.py:ngpus = faiss.get_num_gpus()
tutorial/python/5-Multiple-GPUs.py:print("number of GPUs:", ngpus)
tutorial/python/5-Multiple-GPUs.py:gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
tutorial/python/5-Multiple-GPUs.py:gpu_index.add(xb)              # add vectors to the index
tutorial/python/5-Multiple-GPUs.py:print(gpu_index.ntotal)
tutorial/python/5-Multiple-GPUs.py:D, I = gpu_index.search(xq, k) # actual search
tutorial/python/4-GPU.py:res = faiss.StandardGpuResources()  # use a single GPU
tutorial/python/4-GPU.py:# make it a flat GPU index
tutorial/python/4-GPU.py:gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
tutorial/python/4-GPU.py:gpu_index_flat.add(xb)         # add vectors to the index
tutorial/python/4-GPU.py:print(gpu_index_flat.ntotal)
tutorial/python/4-GPU.py:D, I = gpu_index_flat.search(xq, k)  # actual search
tutorial/python/4-GPU.py:# make it an IVF GPU index
tutorial/python/4-GPU.py:gpu_index_ivf = faiss.index_cpu_to_gpu(res, 0, index_ivf)
tutorial/python/4-GPU.py:assert not gpu_index_ivf.is_trained
tutorial/python/4-GPU.py:gpu_index_ivf.train(xb)        # add vectors to the index
tutorial/python/4-GPU.py:assert gpu_index_ivf.is_trained
tutorial/python/4-GPU.py:gpu_index_ivf.add(xb)          # add vectors to the index
tutorial/python/4-GPU.py:print(gpu_index_ivf.ntotal)
tutorial/python/4-GPU.py:D, I = gpu_index_ivf.search(xq, k)  # actual search
tutorial/cpp/CMakeLists.txt:add_executable(4-GPU EXCLUDE_FROM_ALL 4-GPU.cpp)
tutorial/cpp/CMakeLists.txt:target_link_libraries(4-GPU PRIVATE faiss)
tutorial/cpp/CMakeLists.txt:add_executable(5-Multiple-GPUs EXCLUDE_FROM_ALL 5-Multiple-GPUs.cpp)
tutorial/cpp/CMakeLists.txt:target_link_libraries(5-Multiple-GPUs PRIVATE faiss)
tutorial/cpp/5-Multiple-GPUs.cpp:#include <faiss/gpu/GpuAutoTune.h>
tutorial/cpp/5-Multiple-GPUs.cpp:#include <faiss/gpu/GpuCloner.h>
tutorial/cpp/5-Multiple-GPUs.cpp:#include <faiss/gpu/GpuIndexFlat.h>
tutorial/cpp/5-Multiple-GPUs.cpp:#include <faiss/gpu/StandardGpuResources.h>
tutorial/cpp/5-Multiple-GPUs.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
tutorial/cpp/5-Multiple-GPUs.cpp:    int ngpus = faiss::gpu::getNumDevices();
tutorial/cpp/5-Multiple-GPUs.cpp:    printf("Number of GPUs: %d\n", ngpus);
tutorial/cpp/5-Multiple-GPUs.cpp:    std::vector<faiss::gpu::GpuResourcesProvider*> res;
tutorial/cpp/5-Multiple-GPUs.cpp:    for (int i = 0; i < ngpus; i++) {
tutorial/cpp/5-Multiple-GPUs.cpp:        res.push_back(new faiss::gpu::StandardGpuResources);
tutorial/cpp/5-Multiple-GPUs.cpp:    faiss::Index* gpu_index =
tutorial/cpp/5-Multiple-GPUs.cpp:            faiss::gpu::index_cpu_to_gpu_multiple(res, devs, &cpu_index);
tutorial/cpp/5-Multiple-GPUs.cpp:    printf("is_trained = %s\n", gpu_index->is_trained ? "true" : "false");
tutorial/cpp/5-Multiple-GPUs.cpp:    gpu_index->add(nb, xb); // add vectors to the index
tutorial/cpp/5-Multiple-GPUs.cpp:    printf("ntotal = %ld\n", gpu_index->ntotal);
tutorial/cpp/5-Multiple-GPUs.cpp:        gpu_index->search(nq, xq, k, D, I);
tutorial/cpp/5-Multiple-GPUs.cpp:    delete gpu_index;
tutorial/cpp/5-Multiple-GPUs.cpp:    for (int i = 0; i < ngpus; i++) {
tutorial/cpp/4-GPU.cpp:#include <faiss/gpu/GpuIndexFlat.h>
tutorial/cpp/4-GPU.cpp:#include <faiss/gpu/GpuIndexIVFFlat.h>
tutorial/cpp/4-GPU.cpp:#include <faiss/gpu/StandardGpuResources.h>
tutorial/cpp/4-GPU.cpp:    faiss::gpu::StandardGpuResources res;
tutorial/cpp/4-GPU.cpp:    faiss::gpu::GpuIndexFlatL2 index_flat(&res, d);
tutorial/cpp/4-GPU.cpp:    faiss::gpu::GpuIndexIVFFlat index_ivf(&res, d, nlist, faiss::METRIC_L2);
INSTALL.md:- faiss-gpu, containing both CPU and GPU indices, is available on Linux (x86-64 only) for CUDA 11.4 and 12.1
INSTALL.md:- faiss-gpu-raft containing both CPU and GPU indices provided by NVIDIA RAFT, is available on Linux (x86-64 only) for CUDA 11.8 and 12.1.
INSTALL.md:# GPU(+CPU) version
INSTALL.md:$ conda install -c pytorch -c nvidia faiss-gpu=1.9.0
INSTALL.md:# GPU(+CPU) version with NVIDIA RAFT
INSTALL.md:$ conda install -c pytorch -c nvidia -c rapidsai -c conda-forge faiss-gpu-raft=1.9.0
INSTALL.md:# GPU(+CPU) version using AMD ROCm not yet available
INSTALL.md:For faiss-gpu, the nvidia channel is required for CUDA, which is not
INSTALL.md:For faiss-gpu-raft, the nvidia, rapidsai and conda-forge channels are required.
INSTALL.md:# GPU(+CPU) version
INSTALL.md:$ conda install -c pytorch/label/nightly -c nvidia faiss-gpu=1.9.0
INSTALL.md:# GPU(+CPU) version with NVIDIA RAFT
INSTALL.md:conda install -c pytorch -c nvidia -c rapidsai -c conda-forge faiss-gpu-raft=1.9.0 pytorch pytorch-cuda numpy
INSTALL.md:# GPU(+CPU) version using AMD ROCm not yet available
INSTALL.md:In the above commands, pytorch-cuda=11 or pytorch-cuda=12 would select a specific CUDA version, if it’s required.
INSTALL.md:A combination of versions that installs GPU Faiss with CUDA and Pytorch (as of 2024-05-15):
INSTALL.md:conda install -c pytorch -c nvidia faiss-gpu=1.8.0 pytorch=*=*cuda* pytorch-cuda=11 numpy
INSTALL.md:# GPU version
INSTALL.md:$ conda install -c conda-forge faiss-gpu
INSTALL.md:# AMD ROCm version not yet available
INSTALL.md:- for GPU indices:
INSTALL.md:  - the CUDA toolkit,
INSTALL.md:- for AMD GPUs:
INSTALL.md:  - AMD ROCm,
INSTALL.md:  - `-DFAISS_ENABLE_GPU=OFF` in order to disable building GPU indices (possible
INSTALL.md:    of the IVF-Flat and IVF-PQ GPU-accelerated indices (default is `OFF`, possible
INSTALL.md:- GPU-related options:
INSTALL.md:  - `-DCUDAToolkit_ROOT=/path/to/cuda-10.1` in order to hint to the path of
INSTALL.md:  the CUDA toolkit (for more information, see
INSTALL.md:  [CMake docs](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html)),
INSTALL.md:  - `-DCMAKE_CUDA_ARCHITECTURES="75;72"` for specifying which GPU architectures
INSTALL.md:  to build against (see [CUDA docs](https://developer.nvidia.com/cuda-gpus) to
INSTALL.md:  - `-DFAISS_ENABLE_ROCM=ON` in order to enable building GPU indices for AMD GPUs.
INSTALL.md: `-DFAISS_ENABLE_GPU` must be `ON` when using this option. (possible values are `ON` and `OFF`),
INSTALL.md:### Basic GPU example
INSTALL.md:$ make -C build demo_ivfpq_indexing_gpu
INSTALL.md:$ ./build/demos/demo_ivfpq_indexing_gpu
INSTALL.md:This produce the GPU code equivalent to the CPU `demo_ivfpq_indexing`. It also
INSTALL.md:shows how to translate indexes from/to a GPU.
INSTALL.md:### Real-life test on GPU
INSTALL.md:The example above also runs on GPU. Edit `demos/demo_auto_tune.py` at line 100
INSTALL.md:keys_to_test = keys_gpu
INSTALL.md:use_gpu = True
INSTALL.md:to test the GPU code.
conda/faiss-gpu/build-lib.sh:# Workaround for CUDA 11.4.4 builds. Moves all necessary headers to include root.
conda/faiss-gpu/build-lib.sh:      -DFAISS_ENABLE_GPU=ON \
conda/faiss-gpu/build-lib.sh:      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
conda/faiss-gpu/meta.yaml:{% if cudatoolkit == '11.4.4' %}
conda/faiss-gpu/meta.yaml:{% set cuda_constraints=">=11.4,<12" %}
conda/faiss-gpu/meta.yaml:{% elif cudatoolkit == '12.1.1' %}
conda/faiss-gpu/meta.yaml:{% set cuda_constraints=">=12.1,<13" %}
conda/faiss-gpu/meta.yaml:      string: "h{{ PKG_HASH }}_{{ number }}_cuda{{ cudatoolkit }}{{ suffix }}"
conda/faiss-gpu/meta.yaml:        - CUDA_ARCHS
conda/faiss-gpu/meta.yaml:        - cuda-toolkit {{ cudatoolkit }}
conda/faiss-gpu/meta.yaml:        - cuda-cudart {{ cuda_constraints }}
conda/faiss-gpu/meta.yaml:  - name: faiss-gpu
conda/faiss-gpu/meta.yaml:      string: "py{{ PY_VER }}_h{{ PKG_HASH }}_{{ number }}_cuda{{ cudatoolkit }}{{ suffix }}"
conda/faiss-gpu/meta.yaml:        - cuda-toolkit {{ cudatoolkit }}
conda/faiss-gpu/meta.yaml:        - pytorch-cuda {{ cuda_constraints }}
conda/faiss-gpu/meta.yaml:        - cp tests/common_faiss_tests.py faiss/gpu/test
conda/faiss-gpu/meta.yaml:        - python -X faulthandler -m unittest discover -v -s faiss/gpu/test/ -p "test_*"
conda/faiss-gpu/meta.yaml:        - python -X faulthandler -m unittest discover -v -s faiss/gpu/test/ -p "torch_*"
conda/faiss-gpu/meta.yaml:        - faiss/gpu/test/
conda/faiss-gpu/build-pkg.sh:      -DFAISS_ENABLE_GPU=ON \
conda/faiss-gpu-raft/build-lib.sh:      -DFAISS_ENABLE_GPU=ON \
conda/faiss-gpu-raft/build-lib.sh:      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
conda/faiss-gpu-raft/meta.yaml:{% if cudatoolkit == '11.8.0' %}
conda/faiss-gpu-raft/meta.yaml:{% set cuda_constraints=">=11.8,<12" %}
conda/faiss-gpu-raft/meta.yaml:{% elif cudatoolkit == '12.1.1' %}
conda/faiss-gpu-raft/meta.yaml:{% set cuda_constraints=">=12.1,<13" %}
conda/faiss-gpu-raft/meta.yaml:      string: "h{{ PKG_HASH }}_{{ number }}_cuda{{ cudatoolkit }}_raft{{ suffix }}"
conda/faiss-gpu-raft/meta.yaml:        - CUDA_ARCHS
conda/faiss-gpu-raft/meta.yaml:        - cuda-toolkit {{ cudatoolkit }}
conda/faiss-gpu-raft/meta.yaml:        - cuda-version {{ cuda_constraints }}
conda/faiss-gpu-raft/meta.yaml:        - cuda-cudart {{ cuda_constraints }}
conda/faiss-gpu-raft/meta.yaml:        - cuda-version {{ cuda_constraints }}
conda/faiss-gpu-raft/meta.yaml:  - name: faiss-gpu-raft
conda/faiss-gpu-raft/meta.yaml:      string: "py{{ PY_VER }}_h{{ PKG_HASH }}_{{ number }}_cuda{{ cudatoolkit }}{{ suffix }}"
conda/faiss-gpu-raft/meta.yaml:        - cuda-toolkit {{ cudatoolkit }}
conda/faiss-gpu-raft/meta.yaml:        - pytorch-cuda {{ cuda_constraints }}
conda/faiss-gpu-raft/meta.yaml:        - cp tests/common_faiss_tests.py faiss/gpu/test
conda/faiss-gpu-raft/meta.yaml:        - python -X faulthandler -m unittest discover -v -s faiss/gpu/test/ -p "test_*"
conda/faiss-gpu-raft/meta.yaml:        - python -X faulthandler -m unittest discover -v -s faiss/gpu/test/ -p "torch_*"
conda/faiss-gpu-raft/meta.yaml:        - faiss/gpu/test/
conda/faiss-gpu-raft/build-pkg.sh:      -DFAISS_ENABLE_GPU=ON \
conda/faiss/build-pkg-osx.sh:      -DFAISS_ENABLE_GPU=OFF \
conda/faiss/build-lib.sh:      -DFAISS_ENABLE_GPU=OFF \
conda/faiss/build-lib-arm64.sh:      -DFAISS_ENABLE_GPU=OFF \
conda/faiss/build-lib.bat:      -DFAISS_ENABLE_GPU=OFF ^
conda/faiss/build-pkg.sh:      -DFAISS_ENABLE_GPU=OFF \
conda/faiss/build-lib-osx.sh:      -DFAISS_ENABLE_GPU=OFF \
conda/faiss/build-pkg.bat:      -DFAISS_ENABLE_GPU=OFF ^
conda/faiss/build-pkg-arm64.sh:      -DFAISS_ENABLE_GPU=OFF \
CMakeLists.txt:# Copyright (c) 2023, NVIDIA CORPORATION.
CMakeLists.txt:if(FAISS_ENABLE_GPU)
CMakeLists.txt:  if (FAISS_ENABLE_ROCM)
CMakeLists.txt:    list(PREPEND CMAKE_MODULE_PATH "/opt/rocm/lib/cmake")
CMakeLists.txt:    list(PREPEND CMAKE_PREFIX_PATH "/opt/rocm")
CMakeLists.txt:    list(APPEND FAISS_LANGUAGES CUDA)
CMakeLists.txt:include(rapids-cuda)
CMakeLists.txt:rapids_cuda_init_architectures(faiss)
CMakeLists.txt:rapids_cuda_init_architectures(pyfaiss)
CMakeLists.txt:rapids_cuda_init_architectures(faiss_c_library)
CMakeLists.txt:option(FAISS_ENABLE_GPU "Enable support for GPU indexes." ON)
CMakeLists.txt:option(FAISS_ENABLE_RAFT "Enable RAFT for GPU indexes." OFF)
CMakeLists.txt:option(FAISS_ENABLE_ROCM "Enable ROCm for GPU indexes." OFF)
CMakeLists.txt:if(FAISS_ENABLE_GPU)
CMakeLists.txt:  if(FAISS_ENABLE_ROCM)
CMakeLists.txt:    add_definitions(-DUSE_AMD_ROCM)
CMakeLists.txt:    set(GPU_EXT_PREFIX "hip")
CMakeLists.txt:    execute_process(COMMAND ${PROJECT_SOURCE_DIR}/faiss/gpu/hipify.sh)
CMakeLists.txt:    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
CMakeLists.txt:    enable_language(CUDA)
CMakeLists.txt:    set(GPU_EXT_PREFIX "cu")
CMakeLists.txt:if(FAISS_ENABLE_GPU)
CMakeLists.txt:  if(FAISS_ENABLE_ROCM)
CMakeLists.txt:    add_subdirectory(faiss/gpu-rocm)
CMakeLists.txt:    add_subdirectory(faiss/gpu)
CMakeLists.txt:  if(FAISS_ENABLE_GPU)
CMakeLists.txt:    if(FAISS_ENABLE_ROCM)
CMakeLists.txt:      add_subdirectory(faiss/gpu-rocm/test)
CMakeLists.txt:      add_subdirectory(faiss/gpu/test)
contrib/exhaustive_search.py:    if faiss.get_num_gpus():
contrib/exhaustive_search.py:        LOG.info('running on %d GPUs' % faiss.get_num_gpus())
contrib/exhaustive_search.py:        index = faiss.index_cpu_to_all_gpus(index)
contrib/exhaustive_search.py:def range_search_gpu(xq, r2, index_gpu, index_cpu, gpu_k=1024):
contrib/exhaustive_search.py:    """GPU does not support range search, so we emulate it with
contrib/exhaustive_search.py:    - None. In that case, at most gpu_k results will be returned
contrib/exhaustive_search.py:    is_binary_index = isinstance(index_gpu, faiss.IndexBinary)
contrib/exhaustive_search.py:    keep_max = faiss.is_similarity_metric(index_gpu.metric_type)
contrib/exhaustive_search.py:    k = min(index_gpu.ntotal, gpu_k)
contrib/exhaustive_search.py:        f"GPU search {nq} queries with {k=:} {is_binary_index=:} {keep_max=:}")
contrib/exhaustive_search.py:    D, I = index_gpu.search(xq, k)
contrib/exhaustive_search.py:                    index_cpu = faiss.IndexFlat(d, index_gpu.metric_type)
contrib/exhaustive_search.py:                if index_gpu.metric_type == faiss.METRIC_L2:
contrib/exhaustive_search.py:                       shard=False, ngpu=-1):
contrib/exhaustive_search.py:    if ngpu == -1:
contrib/exhaustive_search.py:        ngpu = faiss.get_num_gpus()
contrib/exhaustive_search.py:    if ngpu:
contrib/exhaustive_search.py:        LOG.info('running on %d GPUs' % ngpu)
contrib/exhaustive_search.py:        co = faiss.GpuMultipleClonerOptions()
contrib/exhaustive_search.py:        index_gpu = faiss.index_cpu_to_all_gpus(index, co=co, ngpu=ngpu)
contrib/exhaustive_search.py:        if ngpu > 0:
contrib/exhaustive_search.py:            index_gpu.add(xbi)
contrib/exhaustive_search.py:            lims_i, Di, Ii = range_search_gpu(xq, threshold, index_gpu, xbi)
contrib/exhaustive_search.py:            index_gpu.reset()
contrib/exhaustive_search.py:                             shard=False, ngpu=0, clip_to_min=False):
contrib/exhaustive_search.py:    If ngpu != 0, the function moves the index to this many GPUs to
contrib/exhaustive_search.py:    if ngpu == -1:
contrib/exhaustive_search.py:        ngpu = faiss.get_num_gpus()
contrib/exhaustive_search.py:    if ngpu:
contrib/exhaustive_search.py:        LOG.info('running on %d GPUs' % ngpu)
contrib/exhaustive_search.py:        co = faiss.GpuMultipleClonerOptions()
contrib/exhaustive_search.py:        index_gpu = faiss.index_cpu_to_all_gpus(index, co=co, ngpu=ngpu)
contrib/exhaustive_search.py:        index_gpu = None
contrib/exhaustive_search.py:        if index_gpu:
contrib/exhaustive_search.py:            lims_i, Di, Ii = range_search_gpu(xqi, radius, index_gpu, index)
contrib/torch/README.md:The code is designed to work with CPU and GPU tensors.
contrib/torch/clustering.py:class DatasetAssignGPU(DatasetAssign):
contrib/torch/clustering.py:        return faiss.knn_gpu(self.res, self.x, centroids, 1)
contrib/torch_utils.py:pytorch Tensors (CPU or GPU) to be used as arguments to Faiss indexes and
contrib/torch_utils.py:other functions. Torch GPU tensors can only be used with Faiss GPU indexes.
contrib/torch_utils.py:If this is imported with a package that supports Faiss GPU, the necessary
contrib/torch_utils.py:    """ gets a Faiss SWIG pointer from a pytorch tensor (on CPU or GPU) """
contrib/torch_utils.py:    """ gets a Faiss SWIG pointer from a pytorch tensor (on CPU or GPU) """
contrib/torch_utils.py:    """ gets a Faiss SWIG pointer from a pytorch tensor (on CPU or GPU) """
contrib/torch_utils.py:    """ gets a Faiss SWIG pointer from a pytorch tensor (on CPU or GPU) """
contrib/torch_utils.py:    """ gets a Faiss SWIG pointer from a pytorch tensor (on CPU or GPU) """
contrib/torch_utils.py:    """ Creates a scoping object to make Faiss GPU use the same stream
contrib/torch_utils.py:        as pytorch, based on torch.cuda.current_stream().
contrib/torch_utils.py:        pytorch_stream = torch.cuda.current_stream()
contrib/torch_utils.py:    # This is the cudaStream_t that we wish to use
contrib/torch_utils.py:    cuda_stream_s = faiss.cast_integer_to_cudastream_t(pytorch_stream.cuda_stream)
contrib/torch_utils.py:    # So we can revert GpuResources stream state upon exit
contrib/torch_utils.py:    prior_dev = torch.cuda.current_device()
contrib/torch_utils.py:    prior_stream = res.getDefaultStream(torch.cuda.current_device())
contrib/torch_utils.py:    res.setDefaultStream(torch.cuda.current_device(), cuda_stream_s)
contrib/torch_utils.py:        if x.is_cuda:
contrib/torch_utils.py:            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'
contrib/torch_utils.py:            # On the GPU, use proper stream ordering
contrib/torch_utils.py:        if x.is_cuda:
contrib/torch_utils.py:            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'
contrib/torch_utils.py:            # On the GPU, use proper stream ordering
contrib/torch_utils.py:        if x.is_cuda:
contrib/torch_utils.py:            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'
contrib/torch_utils.py:            # On the GPU, use proper stream ordering
contrib/torch_utils.py:        if x.is_cuda:
contrib/torch_utils.py:            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'
contrib/torch_utils.py:            # On the GPU, use proper stream ordering
contrib/torch_utils.py:        if x.is_cuda:
contrib/torch_utils.py:            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'
contrib/torch_utils.py:            # On the GPU, use proper stream ordering
contrib/torch_utils.py:        if x.is_cuda:
contrib/torch_utils.py:            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'
contrib/torch_utils.py:            # On the GPU, use proper stream ordering
contrib/torch_utils.py:        if x.is_cuda:
contrib/torch_utils.py:            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'
contrib/torch_utils.py:            # On the GPU, use proper stream ordering
contrib/torch_utils.py:        # produce a GPU tensor
contrib/torch_utils.py:            device = torch.device('cuda', self.getDevice())
contrib/torch_utils.py:        if x.is_cuda:
contrib/torch_utils.py:            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'
contrib/torch_utils.py:            # On the GPU, use proper stream ordering
contrib/torch_utils.py:        # produce a GPU tensor
contrib/torch_utils.py:            device = torch.device('cuda', self.getDevice())
contrib/torch_utils.py:        if x.is_cuda:
contrib/torch_utils.py:            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'
contrib/torch_utils.py:            # On the GPU, use proper stream ordering
contrib/torch_utils.py:        if x.is_cuda:
contrib/torch_utils.py:            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'
contrib/torch_utils.py:            # On the GPU, use proper stream ordering
contrib/torch_utils.py:    # Until the GPU version is implemented, we do not support pre-allocated
contrib/torch_utils.py:        assert not x.is_cuda, 'Range search using GPU tensor not yet implemented'
contrib/torch_utils.py:        assert not hasattr(self, 'getDevice'), 'Range search on GPU index not yet implemented'
contrib/torch_utils.py:        if x.is_cuda:
contrib/torch_utils.py:            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'
contrib/torch_utils.py:            # On the GPU, use proper stream ordering
contrib/torch_utils.py:        if codes.is_cuda:
contrib/torch_utils.py:            assert hasattr(self, 'getDevice'), 'GPU tensor on CPU index not allowed'
contrib/torch_utils.py:            # On the GPU, use proper stream ordering
contrib/torch_utils.py:    assert not xb.is_cuda, "use knn_gpu for GPU tensors"
contrib/torch_utils.py:    assert not xq.is_cuda, "use knn_gpu for GPU tensors"
contrib/torch_utils.py:def torch_replacement_knn_gpu(res, xq, xb, k, D=None, I=None, metric=faiss.METRIC_L2, device=-1, use_raft=False):
contrib/torch_utils.py:        return faiss.knn_gpu_numpy(res, xq, xb, k, D, I, metric, device)
contrib/torch_utils.py:    args = faiss.GpuDistanceParams()
contrib/torch_utils.py:torch_replace_method(faiss_module, 'knn_gpu', torch_replacement_knn_gpu, True, True)
contrib/torch_utils.py:def torch_replacement_pairwise_distance_gpu(res, xq, xb, D=None, metric=faiss.METRIC_L2, device=-1):
contrib/torch_utils.py:        return faiss.pairwise_distance_gpu_numpy(res, xq, xb, D, metric)
contrib/torch_utils.py:    args = faiss.GpuDistanceParams()
contrib/torch_utils.py:torch_replace_method(faiss_module, 'pairwise_distance_gpu', torch_replacement_pairwise_distance_gpu, True, True)
contrib/README.md:Note that although some of the modules may depend on additional modules (eg. GPU Faiss, pytorch, hdf5), they are not necessarily compiled in to avoid adding dependencies. It is the user's responsibility to provide them.
contrib/README.md:Computes the ground-truth search results for a dataset that possibly does not fit in RAM. Uses GPU if available.
contrib/README.md:Interoperability functions for pytorch and Faiss: Importing this will allow pytorch Tensors (CPU or GPU) to be used as arguments to Faiss indexes and other functions. Torch GPU tensors can only be used with Faiss GPU indexes. If this is imported with a package that supports Faiss GPU, the necessary stream synchronization with the current pytorch stream will be automatically performed.
contrib/README.md:Tested in `tests/test_contrib_torch.py` (CPU) and `gpu/test/test_contrib_torch_gpu.py` (GPU).
contrib/big_batch_search.py:    the writeback of results (useful to maximize GPU utilization).
contrib/big_batch_search.py:    use_float16: convert all matrices to float16 (faster for GPU gemm)
contrib/clustering.py:class DatasetAssignGPU(DatasetAssign):
contrib/clustering.py:    """ GPU version of the previous """
contrib/clustering.py:    def __init__(self, x, gpu_id, verbose=False):
contrib/clustering.py:        if gpu_id >= 0:
contrib/clustering.py:            self.index = faiss.index_cpu_to_gpu(
contrib/clustering.py:                faiss.StandardGpuResources(),
contrib/clustering.py:                gpu_id, index)
contrib/clustering.py:            # -1 -> assign to all GPUs
contrib/clustering.py:            self.index = faiss.index_cpu_to_all_gpus(index)
contrib/clustering.py:    For the torch implementation, the centroids are tensors (possibly on GPU),
faiss/clone_index.cpp:        // make sure we don't get a GPU index here
faiss/python/setup.py:shutil.copyfile("gpu_wrappers.py", "faiss/gpu_wrappers.py")
faiss/python/setup.py:are implemented on the GPU. It is developed by Facebook AI Research.
faiss/python/gpu_wrappers.py:# GPU functions
faiss/python/gpu_wrappers.py:def index_cpu_to_gpu_multiple_py(resources, index, co=None, gpus=None):
faiss/python/gpu_wrappers.py:    """ builds the C++ vectors for the GPU indices and the
faiss/python/gpu_wrappers.py:    the list of GPUs """
faiss/python/gpu_wrappers.py:    if gpus is None:
faiss/python/gpu_wrappers.py:        gpus = range(len(resources))
faiss/python/gpu_wrappers.py:    vres = GpuResourcesVector()
faiss/python/gpu_wrappers.py:    for i, res in zip(gpus, resources):
faiss/python/gpu_wrappers.py:        return index_binary_cpu_to_gpu_multiple(vres, vdev, index, co)
faiss/python/gpu_wrappers.py:        return index_cpu_to_gpu_multiple(vres, vdev, index, co)
faiss/python/gpu_wrappers.py:def index_cpu_to_all_gpus(index, co=None, ngpu=-1):
faiss/python/gpu_wrappers.py:    index_gpu = index_cpu_to_gpus_list(index, co=co, gpus=None, ngpu=ngpu)
faiss/python/gpu_wrappers.py:    return index_gpu
faiss/python/gpu_wrappers.py:def index_cpu_to_gpus_list(index, co=None, gpus=None, ngpu=-1):
faiss/python/gpu_wrappers.py:    """ Here we can pass list of GPU ids as a parameter or ngpu to
faiss/python/gpu_wrappers.py:    use first n GPU's. gpus mut be a list or None.
faiss/python/gpu_wrappers.py:    co is a GpuMultipleClonerOptions
faiss/python/gpu_wrappers.py:    if (gpus is None) and (ngpu == -1):  # All blank
faiss/python/gpu_wrappers.py:        gpus = range(get_num_gpus())
faiss/python/gpu_wrappers.py:    elif (gpus is None) and (ngpu != -1):  # Get number of GPU's only
faiss/python/gpu_wrappers.py:        gpus = range(ngpu)
faiss/python/gpu_wrappers.py:    res = [StandardGpuResources() for _ in gpus]
faiss/python/gpu_wrappers.py:    index_gpu = index_cpu_to_gpu_multiple_py(res, index, co, gpus)
faiss/python/gpu_wrappers.py:    return index_gpu
faiss/python/gpu_wrappers.py:def knn_gpu(res, xq, xb, k, D=None, I=None, metric=METRIC_L2, device=-1, use_raft=False, vectorsMemoryLimit=0, queriesMemoryLimit=0):
faiss/python/gpu_wrappers.py:    Compute the k nearest neighbors of a vector on one GPU without constructing an index
faiss/python/gpu_wrappers.py:    res : StandardGpuResources
faiss/python/gpu_wrappers.py:        GPU resources to use during computation
faiss/python/gpu_wrappers.py:        Which CUDA device in the system to run the search on. -1 indicates that
faiss/python/gpu_wrappers.py:        the current thread-local device state (via cudaGetDevice) should be used
faiss/python/gpu_wrappers.py:        (can also be set via torch.cuda.set_device in PyTorch)
faiss/python/gpu_wrappers.py:        Otherwise, an integer 0 <= device < numDevices indicates the GPU on which
faiss/python/gpu_wrappers.py:        If not 0, the GPU will use at most this amount of memory
faiss/python/gpu_wrappers.py:    args = GpuDistanceParams()
faiss/python/gpu_wrappers.py:def pairwise_distance_gpu(res, xq, xb, D=None, metric=METRIC_L2, device=-1):
faiss/python/gpu_wrappers.py:    Compute all pairwise distances between xq and xb on one GPU without constructing an index
faiss/python/gpu_wrappers.py:    res : StandardGpuResources
faiss/python/gpu_wrappers.py:        GPU resources to use during computation
faiss/python/gpu_wrappers.py:        Which CUDA device in the system to run the search on. -1 indicates that
faiss/python/gpu_wrappers.py:        the current thread-local device state (via cudaGetDevice) should be used
faiss/python/gpu_wrappers.py:        (can also be set via torch.cuda.set_device in PyTorch)
faiss/python/gpu_wrappers.py:        Otherwise, an integer 0 <= device < numDevices indicates the GPU on which
faiss/python/gpu_wrappers.py:    args = GpuDistanceParams()
faiss/python/extra_wrappers.py:    gpu: bool or int, optional
faiss/python/extra_wrappers.py:       False: don't use GPU
faiss/python/extra_wrappers.py:       True: use all GPUs
faiss/python/extra_wrappers.py:       number: use this many GPUs
faiss/python/extra_wrappers.py:        self.gpu = False
faiss/python/extra_wrappers.py:            if k == 'gpu':
faiss/python/extra_wrappers.py:                    v = get_num_gpus()
faiss/python/extra_wrappers.py:                self.gpu = v
faiss/python/extra_wrappers.py:            if self.gpu:
faiss/python/extra_wrappers.py:                self.index = faiss.index_cpu_to_all_gpus(self.index, ngpu=self.gpu)
faiss/python/extra_wrappers.py:            if self.gpu:
faiss/python/extra_wrappers.py:                fac = GpuProgressiveDimIndexFactory(ngpu=self.gpu)
faiss/python/CMakeLists.txt:  if(FAISS_ENABLE_GPU)
faiss/python/CMakeLists.txt:      COMPILE_DEFINITIONS GPU_WRAPPER
faiss/python/CMakeLists.txt:    if (FAISS_ENABLE_ROCM)
faiss/python/CMakeLists.txt:        COMPILE_DEFINITIONS FAISS_ENABLE_ROCM
faiss/python/CMakeLists.txt:  if(FAISS_ENABLE_ROCM)
faiss/python/CMakeLists.txt:    foreach(h ${FAISS_GPU_HEADERS})
faiss/python/CMakeLists.txt:        "${faiss_SOURCE_DIR}/faiss/gpu-rocm/${h}")
faiss/python/CMakeLists.txt:        "${faiss_SOURCE_DIR}/faiss/gpu-rocm/${h}")
faiss/python/CMakeLists.txt:        "${faiss_SOURCE_DIR}/faiss/gpu-rocm/${h}")
faiss/python/CMakeLists.txt:        "${faiss_SOURCE_DIR}/faiss/gpu-rocm/${h}")
faiss/python/CMakeLists.txt:    foreach(h ${FAISS_GPU_HEADERS})
faiss/python/CMakeLists.txt:        "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
faiss/python/CMakeLists.txt:        "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
faiss/python/CMakeLists.txt:        "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
faiss/python/CMakeLists.txt:        "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
faiss/python/CMakeLists.txt:        "${faiss_SOURCE_DIR}/faiss/gpu/${h}")
faiss/python/CMakeLists.txt:if(FAISS_ENABLE_GPU)
faiss/python/CMakeLists.txt:  if(FAISS_ENABLE_ROCM)
faiss/python/CMakeLists.txt:    find_package(CUDAToolkit REQUIRED)
faiss/python/CMakeLists.txt:    target_link_libraries(swigfaiss PRIVATE CUDA::cudart
faiss/python/CMakeLists.txt:      $<$<BOOL:${FAISS_ENABLE_RAFT}>:nvidia::cutlass::cutlass>)
faiss/python/CMakeLists.txt:    target_link_libraries(swigfaiss_avx2 PRIVATE CUDA::cudart
faiss/python/CMakeLists.txt:      $<$<BOOL:${FAISS_ENABLE_RAFT}>:nvidia::cutlass::cutlass>)
faiss/python/CMakeLists.txt:    target_link_libraries(swigfaiss_avx512 PRIVATE CUDA::cudart
faiss/python/CMakeLists.txt:      $<$<BOOL:${FAISS_ENABLE_RAFT}>:nvidia::cutlass::cutlass>)
faiss/python/CMakeLists.txt:    target_link_libraries(swigfaiss_sve PRIVATE CUDA::cudart
faiss/python/CMakeLists.txt:      $<$<BOOL:${FAISS_ENABLE_RAFT}>:nvidia::cutlass::cutlass>)
faiss/python/CMakeLists.txt:configure_file(gpu_wrappers.py gpu_wrappers.py COPYONLY)
faiss/python/swigfaiss.swig:// GPU_WRAPPER: also compile interfaces for GPU.
faiss/python/swigfaiss.swig:#ifdef GPU_WRAPPER
faiss/python/swigfaiss.swig:%template(GpuResourcesVector) std::vector<faiss::gpu::GpuResourcesProvider*>;
faiss/python/swigfaiss.swig:int get_num_gpus();
faiss/python/swigfaiss.swig:void gpu_profiler_start();
faiss/python/swigfaiss.swig:void gpu_profiler_stop();
faiss/python/swigfaiss.swig:void gpu_sync_all_devices();
faiss/python/swigfaiss.swig:#ifdef GPU_WRAPPER
faiss/python/swigfaiss.swig:#ifdef FAISS_ENABLE_ROCM
faiss/python/swigfaiss.swig:%shared_ptr(faiss::gpu::GpuResources);
faiss/python/swigfaiss.swig:%shared_ptr(faiss::gpu::StandardGpuResourcesImpl);
faiss/python/swigfaiss.swig:#include <faiss/gpu-rocm/StandardGpuResources.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu-rocm/GpuIndicesOptions.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu-rocm/GpuClonerOptions.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu-rocm/GpuIndex.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu-rocm/GpuIndexFlat.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu-rocm/GpuIndexIVF.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu-rocm/GpuIndexIVFPQ.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu-rocm/GpuIndexIVFFlat.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu-rocm/GpuIndexIVFScalarQuantizer.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu-rocm/GpuIndexBinaryFlat.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu-rocm/GpuAutoTune.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu-rocm/GpuCloner.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu-rocm/GpuDistance.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu-rocm/GpuIcmEncoder.h>
faiss/python/swigfaiss.swig:int get_num_gpus()
faiss/python/swigfaiss.swig:    return faiss::gpu::getNumDevices();
faiss/python/swigfaiss.swig:void gpu_profiler_start()
faiss/python/swigfaiss.swig:    return faiss::gpu::profilerStart();
faiss/python/swigfaiss.swig:void gpu_profiler_stop()
faiss/python/swigfaiss.swig:    return faiss::gpu::profilerStop();
faiss/python/swigfaiss.swig:void gpu_sync_all_devices()
faiss/python/swigfaiss.swig:    return faiss::gpu::synchronizeAllDevices();
faiss/python/swigfaiss.swig:%ignore faiss::gpu::GpuMemoryReservation;
faiss/python/swigfaiss.swig:%ignore faiss::gpu::GpuMemoryReservation::operator=(GpuMemoryReservation&&);
faiss/python/swigfaiss.swig:%ignore faiss::gpu::AllocType;
faiss/python/swigfaiss.swig:%include  <faiss/gpu-rocm/GpuResources.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu-rocm/StandardGpuResources.h>
faiss/python/swigfaiss.swig:hipStream_t cast_integer_to_cudastream_t(int64_t x) {
faiss/python/swigfaiss.swig:int64_t cast_cudastream_t_to_integer(hipStream_t x) {
faiss/python/swigfaiss.swig:#else // FAISS_ENABLE_ROCM
faiss/python/swigfaiss.swig:%shared_ptr(faiss::gpu::GpuResources);
faiss/python/swigfaiss.swig:%shared_ptr(faiss::gpu::StandardGpuResourcesImpl);
faiss/python/swigfaiss.swig:#include <faiss/gpu/StandardGpuResources.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu/GpuIndicesOptions.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu/GpuClonerOptions.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu/GpuIndex.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu/GpuIndexCagra.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu/GpuIndexFlat.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu/GpuIndexIVF.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu/GpuIndexIVFPQ.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu/GpuIndexIVFFlat.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu/GpuIndexBinaryFlat.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu/GpuAutoTune.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu/GpuCloner.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu/GpuDistance.h>
faiss/python/swigfaiss.swig:#include <faiss/gpu/GpuIcmEncoder.h>
faiss/python/swigfaiss.swig:int get_num_gpus()
faiss/python/swigfaiss.swig:    return faiss::gpu::getNumDevices();
faiss/python/swigfaiss.swig:void gpu_profiler_start()
faiss/python/swigfaiss.swig:    return faiss::gpu::profilerStart();
faiss/python/swigfaiss.swig:void gpu_profiler_stop()
faiss/python/swigfaiss.swig:    return faiss::gpu::profilerStop();
faiss/python/swigfaiss.swig:void gpu_sync_all_devices()
faiss/python/swigfaiss.swig:    return faiss::gpu::synchronizeAllDevices();
faiss/python/swigfaiss.swig:%ignore faiss::gpu::GpuMemoryReservation;
faiss/python/swigfaiss.swig:%ignore faiss::gpu::GpuMemoryReservation::operator=(GpuMemoryReservation&&);
faiss/python/swigfaiss.swig:%ignore faiss::gpu::AllocType;
faiss/python/swigfaiss.swig:%include  <faiss/gpu/GpuResources.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu/StandardGpuResources.h>
faiss/python/swigfaiss.swig:typedef CUstream_st* cudaStream_t;
faiss/python/swigfaiss.swig:// interop between pytorch exposed cudaStream_t and faiss
faiss/python/swigfaiss.swig:cudaStream_t cast_integer_to_cudastream_t(int64_t x) {
faiss/python/swigfaiss.swig:  return (cudaStream_t) x;
faiss/python/swigfaiss.swig:int64_t cast_cudastream_t_to_integer(cudaStream_t x) {
faiss/python/swigfaiss.swig:#endif // FAISS_ENABLE_ROCM
faiss/python/swigfaiss.swig:#else // GPU_WRAPPER
faiss/python/swigfaiss.swig:int get_num_gpus()
faiss/python/swigfaiss.swig:void gpu_profiler_start()
faiss/python/swigfaiss.swig:void gpu_profiler_stop()
faiss/python/swigfaiss.swig:void gpu_sync_all_devices()
faiss/python/swigfaiss.swig:#endif // GPU_WRAPPER
faiss/python/swigfaiss.swig:#ifdef GPU_WRAPPER
faiss/python/swigfaiss.swig:#ifdef FAISS_ENABLE_ROCM
faiss/python/swigfaiss.swig:%ignore faiss::gpu::GpuIndexIVF::GpuIndexIVF;
faiss/python/swigfaiss.swig:%include  <faiss/gpu-rocm/GpuIndicesOptions.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu-rocm/GpuClonerOptions.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu-rocm/GpuIndex.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu-rocm/GpuIndexFlat.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu-rocm/GpuIndexIVF.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu-rocm/GpuIndexIVFPQ.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu-rocm/GpuIndexIVFFlat.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu-rocm/GpuIndexIVFScalarQuantizer.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu-rocm/GpuIndexBinaryFlat.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu-rocm/GpuDistance.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu-rocm/GpuIcmEncoder.h>
faiss/python/swigfaiss.swig:#else // FAISS_ENABLE_ROCM
faiss/python/swigfaiss.swig:%ignore faiss::gpu::GpuIndexIVF::GpuIndexIVF;
faiss/python/swigfaiss.swig:%include  <faiss/gpu/GpuIndicesOptions.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu/GpuClonerOptions.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu/GpuIndex.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu/GpuIndexCagra.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu/GpuIndexFlat.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu/GpuIndexIVF.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu/GpuIndexIVFPQ.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu/GpuIndexIVFFlat.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu/GpuIndexBinaryFlat.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu/GpuDistance.h>
faiss/python/swigfaiss.swig:%include  <faiss/gpu/GpuIcmEncoder.h>
faiss/python/swigfaiss.swig:#endif // FAISS_ENABLE_ROCM
faiss/python/swigfaiss.swig:%define DOWNCAST_GPU(subclass)
faiss/python/swigfaiss.swig:    if (dynamic_cast<faiss::gpu::subclass *> ($1)) {
faiss/python/swigfaiss.swig:      $result = SWIG_NewPointerObj($1,SWIGTYPE_p_faiss__gpu__ ## subclass,$owner);
faiss/python/swigfaiss.swig:#ifdef GPU_WRAPPER
faiss/python/swigfaiss.swig:    DOWNCAST_GPU ( GpuIndexCagra )
faiss/python/swigfaiss.swig:    DOWNCAST_GPU ( GpuIndexIVFPQ )
faiss/python/swigfaiss.swig:    DOWNCAST_GPU ( GpuIndexIVFFlat )
faiss/python/swigfaiss.swig:    DOWNCAST_GPU ( GpuIndexIVFScalarQuantizer )
faiss/python/swigfaiss.swig:    DOWNCAST_GPU ( GpuIndexFlat )
faiss/python/swigfaiss.swig:#ifdef GPU_WRAPPER
faiss/python/swigfaiss.swig:    DOWNCAST_GPU ( GpuIndexBinaryFlat )
faiss/python/swigfaiss.swig:#ifdef GPU_WRAPPER
faiss/python/swigfaiss.swig:#ifdef FAISS_ENABLE_ROCM
faiss/python/swigfaiss.swig:%include  <faiss/gpu-rocm/GpuAutoTune.h>
faiss/python/swigfaiss.swig:%newobject index_gpu_to_cpu;
faiss/python/swigfaiss.swig:%newobject index_cpu_to_gpu;
faiss/python/swigfaiss.swig:%newobject index_cpu_to_gpu_multiple;
faiss/python/swigfaiss.swig:%include  <faiss/gpu-rocm/GpuCloner.h>
faiss/python/swigfaiss.swig:#else // FAISS_ENABLE_ROCM
faiss/python/swigfaiss.swig:%include  <faiss/gpu/GpuAutoTune.h>
faiss/python/swigfaiss.swig:%newobject index_gpu_to_cpu;
faiss/python/swigfaiss.swig:%newobject index_cpu_to_gpu;
faiss/python/swigfaiss.swig:%newobject index_cpu_to_gpu_multiple;
faiss/python/swigfaiss.swig:%include  <faiss/gpu/GpuCloner.h>
faiss/python/swigfaiss.swig:#endif // FAISS_ENABLE_ROCM
faiss/python/__init__.py:from faiss.gpu_wrappers import *
faiss/cppcontrib/docker_dev/Dockerfile:RUN cd /root/faiss && /usr/local/bin/cmake -B build -DFAISS_ENABLE_GPU=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release .
faiss/IndexHNSW.h:    // This option is used by GpuIndexCagra::copyTo(IndexHNSWCagra*)
faiss/IndexHNSW.h:    // GpuIndexCagra.
faiss/IndexHNSW.h:    // used when GpuIndexCagra::copyFrom(IndexHNSWCagra*) is invoked.
faiss/IndexHNSW.h:    /// This option is used to copy the knn graph from GpuIndexCagra
faiss/impl/platform_macros.h:// cudatoolkit provides __builtin_ctz for NVCC >= 11.0
faiss/impl/platform_macros.h:#if !defined(__CUDACC__) || __CUDACC_VER_MAJOR__ < 11
faiss/impl/CodePacker.h: * the "fast_scan" indexes on CPU and for some GPU kernels.
faiss/impl/HNSW.cpp:    // `faiss::gpu::GpuIndexCagra::copyFrom(IndexHNSWCagra*)` is functional
faiss/gpu/GpuCloner.cpp:#include <faiss/gpu/GpuCloner.h>
faiss/gpu/GpuCloner.cpp:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/GpuCloner.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuCloner.cpp:#include <faiss/gpu/GpuIndex.h>
faiss/gpu/GpuCloner.cpp:#include <faiss/gpu/GpuIndexBinaryFlat.h>
faiss/gpu/GpuCloner.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuCloner.cpp:#include <faiss/gpu/GpuIndexCagra.h>
faiss/gpu/GpuCloner.cpp:#include <faiss/gpu/GpuIndexFlat.h>
faiss/gpu/GpuCloner.cpp:#include <faiss/gpu/GpuIndexIVFFlat.h>
faiss/gpu/GpuCloner.cpp:#include <faiss/gpu/GpuIndexIVFPQ.h>
faiss/gpu/GpuCloner.cpp:#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
faiss/gpu/GpuCloner.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/GpuCloner.cpp:namespace gpu {
faiss/gpu/GpuCloner.cpp:    if (auto ifl = dynamic_cast<const GpuIndexFlat*>(index)) {
faiss/gpu/GpuCloner.cpp:    } else if (auto ifl = dynamic_cast<const GpuIndexIVFFlat*>(index)) {
faiss/gpu/GpuCloner.cpp:            auto ifl = dynamic_cast<const GpuIndexIVFScalarQuantizer*>(index)) {
faiss/gpu/GpuCloner.cpp:    } else if (auto ipq = dynamic_cast<const GpuIndexIVFPQ*>(index)) {
faiss/gpu/GpuCloner.cpp:        // (inverse op of ToGpuClonerMultiple)
faiss/gpu/GpuCloner.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuCloner.cpp:    else if (auto icg = dynamic_cast<const GpuIndexCagra*>(index)) {
faiss/gpu/GpuCloner.cpp:faiss::Index* index_gpu_to_cpu(const faiss::Index* gpu_index) {
faiss/gpu/GpuCloner.cpp:    return cl.clone_Index(gpu_index);
faiss/gpu/GpuCloner.cpp: * Cloning to 1 GPU
faiss/gpu/GpuCloner.cpp:ToGpuCloner::ToGpuCloner(
faiss/gpu/GpuCloner.cpp:        GpuResourcesProvider* prov,
faiss/gpu/GpuCloner.cpp:        const GpuClonerOptions& options)
faiss/gpu/GpuCloner.cpp:        : GpuClonerOptions(options), provider(prov), device(device) {}
faiss/gpu/GpuCloner.cpp:Index* ToGpuCloner::clone_Index(const Index* index) {
faiss/gpu/GpuCloner.cpp:        GpuIndexFlatConfig config;
faiss/gpu/GpuCloner.cpp:        return new GpuIndexFlat(provider, ifl, config);
faiss/gpu/GpuCloner.cpp:        GpuIndexFlatConfig config;
faiss/gpu/GpuCloner.cpp:        GpuIndexFlat* gif = new GpuIndexFlat(
faiss/gpu/GpuCloner.cpp:        GpuIndexIVFFlatConfig config;
faiss/gpu/GpuCloner.cpp:        GpuIndexIVFFlat* res = new GpuIndexIVFFlat(
faiss/gpu/GpuCloner.cpp:        GpuIndexIVFScalarQuantizerConfig config;
faiss/gpu/GpuCloner.cpp:        GpuIndexIVFScalarQuantizer* res = new GpuIndexIVFScalarQuantizer(
faiss/gpu/GpuCloner.cpp:            printf("  IndexIVFPQ size %ld -> GpuIndexIVFPQ "
faiss/gpu/GpuCloner.cpp:        GpuIndexIVFPQConfig config;
faiss/gpu/GpuCloner.cpp:        GpuIndexIVFPQ* res = new GpuIndexIVFPQ(provider, ipq, config);
faiss/gpu/GpuCloner.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuCloner.cpp:        GpuIndexCagraConfig config;
faiss/gpu/GpuCloner.cpp:        GpuIndexCagra* res =
faiss/gpu/GpuCloner.cpp:                new GpuIndexCagra(provider, icg->d, icg->metric_type, config);
faiss/gpu/GpuCloner.cpp:        FAISS_THROW_MSG("This index type is not implemented on GPU.");
faiss/gpu/GpuCloner.cpp:faiss::Index* index_cpu_to_gpu(
faiss/gpu/GpuCloner.cpp:        GpuResourcesProvider* provider,
faiss/gpu/GpuCloner.cpp:        const GpuClonerOptions* options) {
faiss/gpu/GpuCloner.cpp:    GpuClonerOptions defaults;
faiss/gpu/GpuCloner.cpp:    ToGpuCloner cl(provider, device, options ? *options : defaults);
faiss/gpu/GpuCloner.cpp: * Cloning to multiple GPUs
faiss/gpu/GpuCloner.cpp:ToGpuClonerMultiple::ToGpuClonerMultiple(
faiss/gpu/GpuCloner.cpp:        std::vector<GpuResourcesProvider*>& provider,
faiss/gpu/GpuCloner.cpp:        const GpuMultipleClonerOptions& options)
faiss/gpu/GpuCloner.cpp:        : GpuMultipleClonerOptions(options) {
faiss/gpu/GpuCloner.cpp:ToGpuClonerMultiple::ToGpuClonerMultiple(
faiss/gpu/GpuCloner.cpp:        const std::vector<ToGpuCloner>& sub_cloners,
faiss/gpu/GpuCloner.cpp:        const GpuMultipleClonerOptions& options)
faiss/gpu/GpuCloner.cpp:        : GpuMultipleClonerOptions(options), sub_cloners(sub_cloners) {}
faiss/gpu/GpuCloner.cpp:void ToGpuClonerMultiple::copy_ivf_shard(
faiss/gpu/GpuCloner.cpp:Index* ToGpuClonerMultiple::clone_Index_to_shards(const Index* index) {
faiss/gpu/GpuCloner.cpp:            // on GPU
faiss/gpu/GpuCloner.cpp:        // are short-lived, translated immediately to GPU indexes.
faiss/gpu/GpuCloner.cpp:Index* ToGpuClonerMultiple::clone_Index(const Index* index) {
faiss/gpu/GpuCloner.cpp:            // which GPU(s) will be assigned to this sub-quantizer
faiss/gpu/GpuCloner.cpp:            std::vector<ToGpuCloner> sub_cloners_2;
faiss/gpu/GpuCloner.cpp:            ToGpuClonerMultiple cm(sub_cloners_2, *this);
faiss/gpu/GpuCloner.cpp:faiss::Index* index_cpu_to_gpu_multiple(
faiss/gpu/GpuCloner.cpp:        std::vector<GpuResourcesProvider*>& provider,
faiss/gpu/GpuCloner.cpp:        const GpuMultipleClonerOptions* options) {
faiss/gpu/GpuCloner.cpp:    GpuMultipleClonerOptions defaults;
faiss/gpu/GpuCloner.cpp:    ToGpuClonerMultiple cl(provider, devices, options ? *options : defaults);
faiss/gpu/GpuCloner.cpp:GpuProgressiveDimIndexFactory::GpuProgressiveDimIndexFactory(int ngpu) {
faiss/gpu/GpuCloner.cpp:    FAISS_THROW_IF_NOT(ngpu >= 1);
faiss/gpu/GpuCloner.cpp:    devices.resize(ngpu);
faiss/gpu/GpuCloner.cpp:    vres.resize(ngpu);
faiss/gpu/GpuCloner.cpp:    for (int i = 0; i < ngpu; i++) {
faiss/gpu/GpuCloner.cpp:        vres[i] = new StandardGpuResources();
faiss/gpu/GpuCloner.cpp:GpuProgressiveDimIndexFactory::~GpuProgressiveDimIndexFactory() {
faiss/gpu/GpuCloner.cpp:Index* GpuProgressiveDimIndexFactory::operator()(int dim) {
faiss/gpu/GpuCloner.cpp:    return index_cpu_to_gpu_multiple(vres, devices, &index, &options);
faiss/gpu/GpuCloner.cpp:faiss::IndexBinary* index_binary_gpu_to_cpu(
faiss/gpu/GpuCloner.cpp:        const faiss::IndexBinary* gpu_index) {
faiss/gpu/GpuCloner.cpp:    if (auto ii = dynamic_cast<const GpuIndexBinaryFlat*>(gpu_index)) {
faiss/gpu/GpuCloner.cpp:faiss::IndexBinary* index_binary_cpu_to_gpu(
faiss/gpu/GpuCloner.cpp:        GpuResourcesProvider* provider,
faiss/gpu/GpuCloner.cpp:        const GpuClonerOptions* options) {
faiss/gpu/GpuCloner.cpp:        GpuIndexBinaryFlatConfig config;
faiss/gpu/GpuCloner.cpp:        return new GpuIndexBinaryFlat(provider, ii, config);
faiss/gpu/GpuCloner.cpp:faiss::IndexBinary* index_binary_cpu_to_gpu_multiple(
faiss/gpu/GpuCloner.cpp:        std::vector<GpuResourcesProvider*>& provider,
faiss/gpu/GpuCloner.cpp:        const GpuMultipleClonerOptions* options) {
faiss/gpu/GpuCloner.cpp:    GpuMultipleClonerOptions defaults;
faiss/gpu/GpuCloner.cpp:        return index_binary_cpu_to_gpu(provider[0], devices[0], index, options);
faiss/gpu/GpuCloner.cpp:            ret->addIndex(index_binary_cpu_to_gpu(
faiss/gpu/GpuCloner.cpp:            ret->addIndex(index_binary_cpu_to_gpu(
faiss/gpu/GpuCloner.cpp:} // namespace gpu
faiss/gpu/GpuIndexCagra.cu: * Copyright (c) 2024, NVIDIA CORPORATION.
faiss/gpu/GpuIndexCagra.cu:#include <faiss/gpu/GpuIndexCagra.h>
faiss/gpu/GpuIndexCagra.cu:#include <faiss/gpu/impl/RaftCagra.cuh>
faiss/gpu/GpuIndexCagra.cu:namespace gpu {
faiss/gpu/GpuIndexCagra.cu:GpuIndexCagra::GpuIndexCagra(
faiss/gpu/GpuIndexCagra.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexCagra.cu:        GpuIndexCagraConfig config)
faiss/gpu/GpuIndexCagra.cu:        : GpuIndex(provider->getResources(), dims, metric, 0.0f, config),
faiss/gpu/GpuIndexCagra.cu:void GpuIndexCagra::train(idx_t n, const float* x) {
faiss/gpu/GpuIndexCagra.cu:bool GpuIndexCagra::addImplRequiresIDs_() const {
faiss/gpu/GpuIndexCagra.cu:void GpuIndexCagra::addImpl_(idx_t n, const float* x, const idx_t* ids) {
faiss/gpu/GpuIndexCagra.cu:    FAISS_THROW_MSG("adding vectors is not supported by GpuIndexCagra.");
faiss/gpu/GpuIndexCagra.cu:void GpuIndexCagra::searchImpl_(
faiss/gpu/GpuIndexCagra.cu:void GpuIndexCagra::copyFrom(const faiss::IndexHNSWCagra* index) {
faiss/gpu/GpuIndexCagra.cu:    GpuIndex::copyFrom(index);
faiss/gpu/GpuIndexCagra.cu:void GpuIndexCagra::copyTo(faiss::IndexHNSWCagra* index) const {
faiss/gpu/GpuIndexCagra.cu:    GpuIndex::copyTo(index);
faiss/gpu/GpuIndexCagra.cu:void GpuIndexCagra::reset() {
faiss/gpu/GpuIndexCagra.cu:std::vector<idx_t> GpuIndexCagra::get_knngraph() const {
faiss/gpu/GpuIndexCagra.cu:} // namespace gpu
faiss/gpu/GpuIndexFlat.h:#include <faiss/gpu/GpuIndex.h>
faiss/gpu/GpuIndexFlat.h:namespace gpu {
faiss/gpu/GpuIndexFlat.h:struct GpuIndexFlatConfig : public GpuIndexConfig {
faiss/gpu/GpuIndexFlat.h:/// Wrapper around the GPU implementation that looks like
faiss/gpu/GpuIndexFlat.h:class GpuIndexFlat : public GpuIndex {
faiss/gpu/GpuIndexFlat.h:    /// data over to the given GPU
faiss/gpu/GpuIndexFlat.h:    GpuIndexFlat(
faiss/gpu/GpuIndexFlat.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexFlat.h:            GpuIndexFlatConfig config = GpuIndexFlatConfig());
faiss/gpu/GpuIndexFlat.h:    GpuIndexFlat(
faiss/gpu/GpuIndexFlat.h:            std::shared_ptr<GpuResources> resources,
faiss/gpu/GpuIndexFlat.h:            GpuIndexFlatConfig config = GpuIndexFlatConfig());
faiss/gpu/GpuIndexFlat.h:    GpuIndexFlat(
faiss/gpu/GpuIndexFlat.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexFlat.h:            GpuIndexFlatConfig config = GpuIndexFlatConfig());
faiss/gpu/GpuIndexFlat.h:    GpuIndexFlat(
faiss/gpu/GpuIndexFlat.h:            std::shared_ptr<GpuResources> resources,
faiss/gpu/GpuIndexFlat.h:            GpuIndexFlatConfig config = GpuIndexFlatConfig());
faiss/gpu/GpuIndexFlat.h:    ~GpuIndexFlat() override;
faiss/gpu/GpuIndexFlat.h:    inline FlatIndex* getGpuData() {
faiss/gpu/GpuIndexFlat.h:    /// Called from GpuIndex for add
faiss/gpu/GpuIndexFlat.h:    /// Called from GpuIndex for search
faiss/gpu/GpuIndexFlat.h:    const GpuIndexFlatConfig flatConfig_;
faiss/gpu/GpuIndexFlat.h:    /// Holds our GPU data containing the list of vectors
faiss/gpu/GpuIndexFlat.h:/// Wrapper around the GPU implementation that looks like
faiss/gpu/GpuIndexFlat.h:class GpuIndexFlatL2 : public GpuIndexFlat {
faiss/gpu/GpuIndexFlat.h:    /// data over to the given GPU
faiss/gpu/GpuIndexFlat.h:    GpuIndexFlatL2(
faiss/gpu/GpuIndexFlat.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexFlat.h:            GpuIndexFlatConfig config = GpuIndexFlatConfig());
faiss/gpu/GpuIndexFlat.h:    GpuIndexFlatL2(
faiss/gpu/GpuIndexFlat.h:            std::shared_ptr<GpuResources> resources,
faiss/gpu/GpuIndexFlat.h:            GpuIndexFlatConfig config = GpuIndexFlatConfig());
faiss/gpu/GpuIndexFlat.h:    GpuIndexFlatL2(
faiss/gpu/GpuIndexFlat.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexFlat.h:            GpuIndexFlatConfig config = GpuIndexFlatConfig());
faiss/gpu/GpuIndexFlat.h:    GpuIndexFlatL2(
faiss/gpu/GpuIndexFlat.h:            std::shared_ptr<GpuResources> resources,
faiss/gpu/GpuIndexFlat.h:            GpuIndexFlatConfig config = GpuIndexFlatConfig());
faiss/gpu/GpuIndexFlat.h:/// Wrapper around the GPU implementation that looks like
faiss/gpu/GpuIndexFlat.h:class GpuIndexFlatIP : public GpuIndexFlat {
faiss/gpu/GpuIndexFlat.h:    /// data over to the given GPU
faiss/gpu/GpuIndexFlat.h:    GpuIndexFlatIP(
faiss/gpu/GpuIndexFlat.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexFlat.h:            GpuIndexFlatConfig config = GpuIndexFlatConfig());
faiss/gpu/GpuIndexFlat.h:    GpuIndexFlatIP(
faiss/gpu/GpuIndexFlat.h:            std::shared_ptr<GpuResources> resources,
faiss/gpu/GpuIndexFlat.h:            GpuIndexFlatConfig config = GpuIndexFlatConfig());
faiss/gpu/GpuIndexFlat.h:    GpuIndexFlatIP(
faiss/gpu/GpuIndexFlat.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexFlat.h:            GpuIndexFlatConfig config = GpuIndexFlatConfig());
faiss/gpu/GpuIndexFlat.h:    GpuIndexFlatIP(
faiss/gpu/GpuIndexFlat.h:            std::shared_ptr<GpuResources> resources,
faiss/gpu/GpuIndexFlat.h:            GpuIndexFlatConfig config = GpuIndexFlatConfig());
faiss/gpu/GpuIndexFlat.h:} // namespace gpu
faiss/gpu/GpuAutoTune.h:namespace gpu {
faiss/gpu/GpuAutoTune.h:/// parameter space and setters for GPU indexes
faiss/gpu/GpuAutoTune.h:struct GpuParameterSpace : faiss::ParameterSpace {
faiss/gpu/GpuAutoTune.h:} // namespace gpu
faiss/gpu/GpuDistance.cu: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/GpuDistance.cu:#include <faiss/gpu/GpuDistance.h>
faiss/gpu/GpuDistance.cu:#include <faiss/gpu/GpuResources.h>
faiss/gpu/GpuDistance.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/GpuDistance.cu:#include <faiss/gpu/impl/Distance.cuh>
faiss/gpu/GpuDistance.cu:#include <faiss/gpu/utils/ConversionOperators.cuh>
faiss/gpu/GpuDistance.cu:#include <faiss/gpu/utils/CopyUtils.cuh>
faiss/gpu/GpuDistance.cu:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/GpuDistance.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuDistance.cu:#include <faiss/gpu/utils/RaftUtils.h>
faiss/gpu/GpuDistance.cu:namespace gpu {
faiss/gpu/GpuDistance.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuDistance.cu:bool should_use_raft(GpuDistanceParams args) {
faiss/gpu/GpuDistance.cu:        cudaDeviceProp prop;
faiss/gpu/GpuDistance.cu:        cudaGetDeviceProperties(&prop, dev);
faiss/gpu/GpuDistance.cu:void bfKnnConvert(GpuResourcesProvider* prov, const GpuDistanceParams& args) {
faiss/gpu/GpuDistance.cu:        // Original behavior if no device is specified, use the current CUDA
faiss/gpu/GpuDistance.cu:                "bfKnn: device specified must be -1 (current CUDA thread local device) "
faiss/gpu/GpuDistance.cu:void bfKnn(GpuResourcesProvider* prov, const GpuDistanceParams& args) {
faiss/gpu/GpuDistance.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuDistance.cu:        GpuResourcesProvider* prov,
faiss/gpu/GpuDistance.cu:        const GpuDistanceParams& args,
faiss/gpu/GpuDistance.cu:    GpuDistanceParams args_batch = args;
faiss/gpu/GpuDistance.cu:        GpuResourcesProvider* prov,
faiss/gpu/GpuDistance.cu:        const GpuDistanceParams& args,
faiss/gpu/GpuDistance.cu:        GpuResourcesProvider* prov,
faiss/gpu/GpuDistance.cu:        const GpuDistanceParams& args,
faiss/gpu/GpuDistance.cu:        GpuDistanceParams args_batch = args;
faiss/gpu/GpuDistance.cu:        GpuResourcesProvider* res,
faiss/gpu/GpuDistance.cu:    GpuDistanceParams args;
faiss/gpu/GpuDistance.cu:} // namespace gpu
faiss/gpu/GpuIndexFlat.cu:#include <faiss/gpu/GpuIndex.h>
faiss/gpu/GpuIndexFlat.cu:#include <faiss/gpu/GpuIndexFlat.h>
faiss/gpu/GpuIndexFlat.cu:#include <faiss/gpu/GpuResources.h>
faiss/gpu/GpuIndexFlat.cu:#include <faiss/gpu/impl/IndexUtils.h>
faiss/gpu/GpuIndexFlat.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/GpuIndexFlat.cu:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/GpuIndexFlat.cu:#include <faiss/gpu/impl/FlatIndex.cuh>
faiss/gpu/GpuIndexFlat.cu:#include <faiss/gpu/utils/ConversionOperators.cuh>
faiss/gpu/GpuIndexFlat.cu:#include <faiss/gpu/utils/CopyUtils.cuh>
faiss/gpu/GpuIndexFlat.cu:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/GpuIndexFlat.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuIndexFlat.cu:#include <faiss/gpu/impl/RaftFlatIndex.cuh>
faiss/gpu/GpuIndexFlat.cu:namespace gpu {
faiss/gpu/GpuIndexFlat.cu:GpuIndexFlat::GpuIndexFlat(
faiss/gpu/GpuIndexFlat.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexFlat.cu:        GpuIndexFlatConfig config)
faiss/gpu/GpuIndexFlat.cu:        : GpuIndex(
faiss/gpu/GpuIndexFlat.cu:GpuIndexFlat::GpuIndexFlat(
faiss/gpu/GpuIndexFlat.cu:        std::shared_ptr<GpuResources> resources,
faiss/gpu/GpuIndexFlat.cu:        GpuIndexFlatConfig config)
faiss/gpu/GpuIndexFlat.cu:        : GpuIndex(
faiss/gpu/GpuIndexFlat.cu:GpuIndexFlat::GpuIndexFlat(
faiss/gpu/GpuIndexFlat.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexFlat.cu:        GpuIndexFlatConfig config)
faiss/gpu/GpuIndexFlat.cu:        : GpuIndex(provider->getResources(), dims, metric, 0, config),
faiss/gpu/GpuIndexFlat.cu:GpuIndexFlat::GpuIndexFlat(
faiss/gpu/GpuIndexFlat.cu:        std::shared_ptr<GpuResources> resources,
faiss/gpu/GpuIndexFlat.cu:        GpuIndexFlatConfig config)
faiss/gpu/GpuIndexFlat.cu:        : GpuIndex(resources, dims, metric, 0, config), flatConfig_(config) {
faiss/gpu/GpuIndexFlat.cu:GpuIndexFlat::~GpuIndexFlat() {}
faiss/gpu/GpuIndexFlat.cu:void GpuIndexFlat::resetIndex_(int dims) {
faiss/gpu/GpuIndexFlat.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuIndexFlat.cu:void GpuIndexFlat::copyFrom(const faiss::IndexFlat* index) {
faiss/gpu/GpuIndexFlat.cu:    GpuIndex::copyFrom(index);
faiss/gpu/GpuIndexFlat.cu:void GpuIndexFlat::copyTo(faiss::IndexFlat* index) const {
faiss/gpu/GpuIndexFlat.cu:    GpuIndex::copyTo(index);
faiss/gpu/GpuIndexFlat.cu:        // FIXME: there is an extra GPU allocation here and copy if the flat
faiss/gpu/GpuIndexFlat.cu:size_t GpuIndexFlat::getNumVecs() const {
faiss/gpu/GpuIndexFlat.cu:void GpuIndexFlat::reset() {
faiss/gpu/GpuIndexFlat.cu:void GpuIndexFlat::train(idx_t n, const float* x) {
faiss/gpu/GpuIndexFlat.cu:void GpuIndexFlat::add(idx_t n, const float* x) {
faiss/gpu/GpuIndexFlat.cu:        GpuIndex::add(n, x);
faiss/gpu/GpuIndexFlat.cu:bool GpuIndexFlat::addImplRequiresIDs_() const {
faiss/gpu/GpuIndexFlat.cu:void GpuIndexFlat::addImpl_(idx_t n, const float* x, const idx_t* ids) {
faiss/gpu/GpuIndexFlat.cu:void GpuIndexFlat::searchImpl_(
faiss/gpu/GpuIndexFlat.cu:    // Input and output data are already resident on the GPU
faiss/gpu/GpuIndexFlat.cu:void GpuIndexFlat::reconstruct(idx_t key, float* out) const {
faiss/gpu/GpuIndexFlat.cu:void GpuIndexFlat::reconstruct_n(idx_t i0, idx_t n, float* out) const {
faiss/gpu/GpuIndexFlat.cu:void GpuIndexFlat::reconstruct_batch(idx_t n, const idx_t* keys, float* out)
faiss/gpu/GpuIndexFlat.cu:void GpuIndexFlat::compute_residual(const float* x, float* residual, idx_t key)
faiss/gpu/GpuIndexFlat.cu:void GpuIndexFlat::compute_residual_n(
faiss/gpu/GpuIndexFlat.cu:// GpuIndexFlatL2
faiss/gpu/GpuIndexFlat.cu:GpuIndexFlatL2::GpuIndexFlatL2(
faiss/gpu/GpuIndexFlat.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexFlat.cu:        GpuIndexFlatConfig config)
faiss/gpu/GpuIndexFlat.cu:        : GpuIndexFlat(provider, index, config) {}
faiss/gpu/GpuIndexFlat.cu:GpuIndexFlatL2::GpuIndexFlatL2(
faiss/gpu/GpuIndexFlat.cu:        std::shared_ptr<GpuResources> resources,
faiss/gpu/GpuIndexFlat.cu:        GpuIndexFlatConfig config)
faiss/gpu/GpuIndexFlat.cu:        : GpuIndexFlat(resources, index, config) {}
faiss/gpu/GpuIndexFlat.cu:GpuIndexFlatL2::GpuIndexFlatL2(
faiss/gpu/GpuIndexFlat.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexFlat.cu:        GpuIndexFlatConfig config)
faiss/gpu/GpuIndexFlat.cu:        : GpuIndexFlat(provider, dims, faiss::METRIC_L2, config) {}
faiss/gpu/GpuIndexFlat.cu:GpuIndexFlatL2::GpuIndexFlatL2(
faiss/gpu/GpuIndexFlat.cu:        std::shared_ptr<GpuResources> resources,
faiss/gpu/GpuIndexFlat.cu:        GpuIndexFlatConfig config)
faiss/gpu/GpuIndexFlat.cu:        : GpuIndexFlat(resources, dims, faiss::METRIC_L2, config) {}
faiss/gpu/GpuIndexFlat.cu:void GpuIndexFlatL2::copyFrom(faiss::IndexFlat* index) {
faiss/gpu/GpuIndexFlat.cu:            "Cannot copy a GpuIndexFlatL2 from an index of "
faiss/gpu/GpuIndexFlat.cu:    GpuIndexFlat::copyFrom(index);
faiss/gpu/GpuIndexFlat.cu:void GpuIndexFlatL2::copyTo(faiss::IndexFlat* index) {
faiss/gpu/GpuIndexFlat.cu:            "Cannot copy a GpuIndexFlatL2 to an index of "
faiss/gpu/GpuIndexFlat.cu:    GpuIndexFlat::copyTo(index);
faiss/gpu/GpuIndexFlat.cu:// GpuIndexFlatIP
faiss/gpu/GpuIndexFlat.cu:GpuIndexFlatIP::GpuIndexFlatIP(
faiss/gpu/GpuIndexFlat.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexFlat.cu:        GpuIndexFlatConfig config)
faiss/gpu/GpuIndexFlat.cu:        : GpuIndexFlat(provider, index, config) {}
faiss/gpu/GpuIndexFlat.cu:GpuIndexFlatIP::GpuIndexFlatIP(
faiss/gpu/GpuIndexFlat.cu:        std::shared_ptr<GpuResources> resources,
faiss/gpu/GpuIndexFlat.cu:        GpuIndexFlatConfig config)
faiss/gpu/GpuIndexFlat.cu:        : GpuIndexFlat(resources, index, config) {}
faiss/gpu/GpuIndexFlat.cu:GpuIndexFlatIP::GpuIndexFlatIP(
faiss/gpu/GpuIndexFlat.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexFlat.cu:        GpuIndexFlatConfig config)
faiss/gpu/GpuIndexFlat.cu:        : GpuIndexFlat(provider, dims, faiss::METRIC_INNER_PRODUCT, config) {}
faiss/gpu/GpuIndexFlat.cu:GpuIndexFlatIP::GpuIndexFlatIP(
faiss/gpu/GpuIndexFlat.cu:        std::shared_ptr<GpuResources> resources,
faiss/gpu/GpuIndexFlat.cu:        GpuIndexFlatConfig config)
faiss/gpu/GpuIndexFlat.cu:        : GpuIndexFlat(resources, dims, faiss::METRIC_INNER_PRODUCT, config) {}
faiss/gpu/GpuIndexFlat.cu:void GpuIndexFlatIP::copyFrom(faiss::IndexFlat* index) {
faiss/gpu/GpuIndexFlat.cu:            "Cannot copy a GpuIndexFlatIP from an index of "
faiss/gpu/GpuIndexFlat.cu:    GpuIndexFlat::copyFrom(index);
faiss/gpu/GpuIndexFlat.cu:void GpuIndexFlatIP::copyTo(faiss::IndexFlat* index) {
faiss/gpu/GpuIndexFlat.cu:            "Cannot copy a GpuIndexFlatIP to an index of "
faiss/gpu/GpuIndexFlat.cu:    GpuIndexFlat::copyTo(index);
faiss/gpu/GpuIndexFlat.cu:} // namespace gpu
faiss/gpu/StandardGpuResources.cpp: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/StandardGpuResources.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/StandardGpuResources.cpp:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/StandardGpuResources.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/StandardGpuResources.cpp:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/StandardGpuResources.cpp:namespace gpu {
faiss/gpu/StandardGpuResources.cpp:// Use 256 MiB of pinned memory for async CPU <-> GPU copies by default
faiss/gpu/StandardGpuResources.cpp:// Default temporary memory allocation for <= 4 GiB memory GPUs
faiss/gpu/StandardGpuResources.cpp:// Default temporary memory allocation for <= 8 GiB memory GPUs
faiss/gpu/StandardGpuResources.cpp:// Maximum temporary memory allocation for all GPUs
faiss/gpu/StandardGpuResources.cpp:// StandardGpuResourcesImpl
faiss/gpu/StandardGpuResources.cpp:StandardGpuResourcesImpl::StandardGpuResourcesImpl()
faiss/gpu/StandardGpuResources.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/StandardGpuResources.cpp:          tempMemSize_(getDefaultTempMemForGPU(
faiss/gpu/StandardGpuResources.cpp:StandardGpuResourcesImpl::~StandardGpuResourcesImpl() {
faiss/gpu/StandardGpuResources.cpp:                    << "StandardGpuResources destroyed with allocations outstanding:\n"
faiss/gpu/StandardGpuResources.cpp:            !allocError, "GPU memory allocations not properly cleaned up");
faiss/gpu/StandardGpuResources.cpp:        CUDA_VERIFY(cudaStreamDestroy(entry.second));
faiss/gpu/StandardGpuResources.cpp:            CUDA_VERIFY(cudaStreamDestroy(stream));
faiss/gpu/StandardGpuResources.cpp:        CUDA_VERIFY(cudaStreamDestroy(entry.second));
faiss/gpu/StandardGpuResources.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/StandardGpuResources.cpp:        auto err = cudaFreeHost(pinnedMemAlloc_);
faiss/gpu/StandardGpuResources.cpp:                err == cudaSuccess,
faiss/gpu/StandardGpuResources.cpp:                "Failed to cudaFreeHost pointer %p (error %d %s)",
faiss/gpu/StandardGpuResources.cpp:                cudaGetErrorString(err));
faiss/gpu/StandardGpuResources.cpp:size_t StandardGpuResourcesImpl::getDefaultTempMemForGPU(
faiss/gpu/StandardGpuResources.cpp:        // If the GPU has <= 4 GiB of memory, reserve 512 MiB
faiss/gpu/StandardGpuResources.cpp:        // If the GPU has <= 8 GiB of memory, reserve 1 GiB
faiss/gpu/StandardGpuResources.cpp:void StandardGpuResourcesImpl::noTempMemory() {
faiss/gpu/StandardGpuResources.cpp:void StandardGpuResourcesImpl::setTempMemory(size_t size) {
faiss/gpu/StandardGpuResources.cpp:        tempMemSize_ = getDefaultTempMemForGPU(-1, size);
faiss/gpu/StandardGpuResources.cpp:        // currently running work, because the cudaFree call that this implies
faiss/gpu/StandardGpuResources.cpp:        // will force-synchronize all GPUs with the CPU
faiss/gpu/StandardGpuResources.cpp:                    getDefaultTempMemForGPU(device, tempMemSize_));
faiss/gpu/StandardGpuResources.cpp:void StandardGpuResourcesImpl::setPinnedMemory(size_t size) {
faiss/gpu/StandardGpuResources.cpp:void StandardGpuResourcesImpl::setDefaultStream(
faiss/gpu/StandardGpuResources.cpp:        cudaStream_t stream) {
faiss/gpu/StandardGpuResources.cpp:        cudaStream_t prevStream = nullptr;
faiss/gpu/StandardGpuResources.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/StandardGpuResources.cpp:void StandardGpuResourcesImpl::revertDefaultStream(int device) {
faiss/gpu/StandardGpuResources.cpp:            cudaStream_t prevStream = userDefaultStreams_[device];
faiss/gpu/StandardGpuResources.cpp:            cudaStream_t newStream = defaultStreams_[device];
faiss/gpu/StandardGpuResources.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/StandardGpuResources.cpp:void StandardGpuResourcesImpl::setDefaultNullStreamAllDevices() {
faiss/gpu/StandardGpuResources.cpp:void StandardGpuResourcesImpl::setLogMemoryAllocations(bool enable) {
faiss/gpu/StandardGpuResources.cpp:bool StandardGpuResourcesImpl::isInitialized(int device) const {
faiss/gpu/StandardGpuResources.cpp:void StandardGpuResourcesImpl::initializeForDevice(int device) {
faiss/gpu/StandardGpuResources.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/StandardGpuResources.cpp:                FAISS_THROW_MSG("CUDA memory allocation error");
faiss/gpu/StandardGpuResources.cpp:        auto err = cudaHostAlloc(
faiss/gpu/StandardGpuResources.cpp:                &pinnedMemAlloc_, pinnedMemSize_, cudaHostAllocDefault);
faiss/gpu/StandardGpuResources.cpp:                err == cudaSuccess,
faiss/gpu/StandardGpuResources.cpp:                "failed to cudaHostAlloc %zu bytes for CPU <-> GPU "
faiss/gpu/StandardGpuResources.cpp:                cudaGetErrorString(err));
faiss/gpu/StandardGpuResources.cpp:#if USE_AMD_ROCM
faiss/gpu/StandardGpuResources.cpp:    cudaStream_t defaultStream = nullptr;
faiss/gpu/StandardGpuResources.cpp:    CUDA_VERIFY(
faiss/gpu/StandardGpuResources.cpp:            cudaStreamCreateWithFlags(&defaultStream, cudaStreamNonBlocking));
faiss/gpu/StandardGpuResources.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/StandardGpuResources.cpp:    cudaStream_t asyncCopyStream = 0;
faiss/gpu/StandardGpuResources.cpp:    CUDA_VERIFY(
faiss/gpu/StandardGpuResources.cpp:            cudaStreamCreateWithFlags(&asyncCopyStream, cudaStreamNonBlocking));
faiss/gpu/StandardGpuResources.cpp:    std::vector<cudaStream_t> deviceStreams;
faiss/gpu/StandardGpuResources.cpp:        cudaStream_t stream = nullptr;
faiss/gpu/StandardGpuResources.cpp:        CUDA_VERIFY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
faiss/gpu/StandardGpuResources.cpp:    // For CUDA 10 on V100, enabling tensor core usage would enable automatic
faiss/gpu/StandardGpuResources.cpp:    // in unacceptable loss of precision in general. For CUDA 11 / A100, only
faiss/gpu/StandardGpuResources.cpp:#if CUDA_VERSION >= 11000
faiss/gpu/StandardGpuResources.cpp:            getDefaultTempMemForGPU(device, tempMemSize_));
faiss/gpu/StandardGpuResources.cpp:cublasHandle_t StandardGpuResourcesImpl::getBlasHandle(int device) {
faiss/gpu/StandardGpuResources.cpp:cudaStream_t StandardGpuResourcesImpl::getDefaultStream(int device) {
faiss/gpu/StandardGpuResources.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/StandardGpuResources.cpp:raft::device_resources& StandardGpuResourcesImpl::getRaftHandle(int device) {
faiss/gpu/StandardGpuResources.cpp:        // to the current GpuResources
faiss/gpu/StandardGpuResources.cpp:std::vector<cudaStream_t> StandardGpuResourcesImpl::getAlternateStreams(
faiss/gpu/StandardGpuResources.cpp:std::pair<void*, size_t> StandardGpuResourcesImpl::getPinnedMemory() {
faiss/gpu/StandardGpuResources.cpp:cudaStream_t StandardGpuResourcesImpl::getAsyncCopyStream(int device) {
faiss/gpu/StandardGpuResources.cpp:void* StandardGpuResourcesImpl::allocMemory(const AllocRequest& req) {
faiss/gpu/StandardGpuResources.cpp:    // cudaMalloc guarantees allocation alignment to 256 bytes; do the same here
faiss/gpu/StandardGpuResources.cpp:                        << "StandardGpuResources: alloc fail "
faiss/gpu/StandardGpuResources.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/StandardGpuResources.cpp:                            rmm::cuda_device_id{adjReq.device});
faiss/gpu/StandardGpuResources.cpp:            FAISS_THROW_MSG("CUDA memory allocation error");
faiss/gpu/StandardGpuResources.cpp:        auto err = cudaMalloc(&p, adjReq.size);
faiss/gpu/StandardGpuResources.cpp:        if (err != cudaSuccess) {
faiss/gpu/StandardGpuResources.cpp:            // FIXME: as of CUDA 11, a memory allocation error appears to be
faiss/gpu/StandardGpuResources.cpp:            // presented via cudaGetLastError as well, and needs to be
faiss/gpu/StandardGpuResources.cpp:            cudaGetLastError();
faiss/gpu/StandardGpuResources.cpp:            ss << "StandardGpuResources: alloc fail " << adjReq.toString()
faiss/gpu/StandardGpuResources.cpp:               << " (cudaMalloc error " << cudaGetErrorString(err) << " ["
faiss/gpu/StandardGpuResources.cpp:            FAISS_THROW_IF_NOT_FMT(err == cudaSuccess, "%s", str.c_str());
faiss/gpu/StandardGpuResources.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/StandardGpuResources.cpp:            FAISS_THROW_MSG("CUDA memory allocation error");
faiss/gpu/StandardGpuResources.cpp:        auto err = cudaMallocManaged(&p, adjReq.size);
faiss/gpu/StandardGpuResources.cpp:        if (err != cudaSuccess) {
faiss/gpu/StandardGpuResources.cpp:            // FIXME: as of CUDA 11, a memory allocation error appears to be
faiss/gpu/StandardGpuResources.cpp:            // presented via cudaGetLastError as well, and needs to be cleared.
faiss/gpu/StandardGpuResources.cpp:            cudaGetLastError();
faiss/gpu/StandardGpuResources.cpp:            ss << "StandardGpuResources: alloc fail " << adjReq.toString()
faiss/gpu/StandardGpuResources.cpp:               << " failed (cudaMallocManaged error " << cudaGetErrorString(err)
faiss/gpu/StandardGpuResources.cpp:            FAISS_THROW_IF_NOT_FMT(err == cudaSuccess, "%s", str.c_str());
faiss/gpu/StandardGpuResources.cpp:        std::cout << "StandardGpuResources: alloc ok " << adjReq.toString()
faiss/gpu/StandardGpuResources.cpp:void StandardGpuResourcesImpl::deallocMemory(int device, void* p) {
faiss/gpu/StandardGpuResources.cpp:        std::cout << "StandardGpuResources: dealloc " << req.toString() << "\n";
faiss/gpu/StandardGpuResources.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/StandardGpuResources.cpp:        auto err = cudaFree(p);
faiss/gpu/StandardGpuResources.cpp:                err == cudaSuccess,
faiss/gpu/StandardGpuResources.cpp:                "Failed to cudaFree pointer %p (error %d %s)",
faiss/gpu/StandardGpuResources.cpp:                cudaGetErrorString(err));
faiss/gpu/StandardGpuResources.cpp:size_t StandardGpuResourcesImpl::getTempMemoryAvailable(int device) const {
faiss/gpu/StandardGpuResources.cpp:StandardGpuResourcesImpl::getMemoryInfo() const {
faiss/gpu/StandardGpuResources.cpp:// StandardGpuResources
faiss/gpu/StandardGpuResources.cpp:StandardGpuResources::StandardGpuResources()
faiss/gpu/StandardGpuResources.cpp:        : res_(new StandardGpuResourcesImpl) {}
faiss/gpu/StandardGpuResources.cpp:StandardGpuResources::~StandardGpuResources() = default;
faiss/gpu/StandardGpuResources.cpp:std::shared_ptr<GpuResources> StandardGpuResources::getResources() {
faiss/gpu/StandardGpuResources.cpp:void StandardGpuResources::noTempMemory() {
faiss/gpu/StandardGpuResources.cpp:void StandardGpuResources::setTempMemory(size_t size) {
faiss/gpu/StandardGpuResources.cpp:void StandardGpuResources::setPinnedMemory(size_t size) {
faiss/gpu/StandardGpuResources.cpp:void StandardGpuResources::setDefaultStream(int device, cudaStream_t stream) {
faiss/gpu/StandardGpuResources.cpp:void StandardGpuResources::revertDefaultStream(int device) {
faiss/gpu/StandardGpuResources.cpp:void StandardGpuResources::setDefaultNullStreamAllDevices() {
faiss/gpu/StandardGpuResources.cpp:StandardGpuResources::getMemoryInfo() const {
faiss/gpu/StandardGpuResources.cpp:cudaStream_t StandardGpuResources::getDefaultStream(int device) {
faiss/gpu/StandardGpuResources.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/StandardGpuResources.cpp:raft::device_resources& StandardGpuResources::getRaftHandle(int device) {
faiss/gpu/StandardGpuResources.cpp:size_t StandardGpuResources::getTempMemoryAvailable(int device) const {
faiss/gpu/StandardGpuResources.cpp:void StandardGpuResources::syncDefaultStreamCurrentDevice() {
faiss/gpu/StandardGpuResources.cpp:void StandardGpuResources::setLogMemoryAllocations(bool enable) {
faiss/gpu/StandardGpuResources.cpp:} // namespace gpu
faiss/gpu/GpuIndexIVFScalarQuantizer.h:#include <faiss/gpu/GpuIndexIVF.h>
faiss/gpu/GpuIndexIVFScalarQuantizer.h:namespace gpu {
faiss/gpu/GpuIndexIVFScalarQuantizer.h:class GpuIndexFlat;
faiss/gpu/GpuIndexIVFScalarQuantizer.h:struct GpuIndexIVFScalarQuantizerConfig : public GpuIndexIVFConfig {
faiss/gpu/GpuIndexIVFScalarQuantizer.h:/// Wrapper around the GPU implementation that looks like
faiss/gpu/GpuIndexIVFScalarQuantizer.h:class GpuIndexIVFScalarQuantizer : public GpuIndexIVF {
faiss/gpu/GpuIndexIVFScalarQuantizer.h:    /// copying data over to the given GPU, if the input index is trained.
faiss/gpu/GpuIndexIVFScalarQuantizer.h:    GpuIndexIVFScalarQuantizer(
faiss/gpu/GpuIndexIVFScalarQuantizer.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFScalarQuantizer.h:            GpuIndexIVFScalarQuantizerConfig config =
faiss/gpu/GpuIndexIVFScalarQuantizer.h:                    GpuIndexIVFScalarQuantizerConfig());
faiss/gpu/GpuIndexIVFScalarQuantizer.h:    GpuIndexIVFScalarQuantizer(
faiss/gpu/GpuIndexIVFScalarQuantizer.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFScalarQuantizer.h:            GpuIndexIVFScalarQuantizerConfig config =
faiss/gpu/GpuIndexIVFScalarQuantizer.h:                    GpuIndexIVFScalarQuantizerConfig());
faiss/gpu/GpuIndexIVFScalarQuantizer.h:    /// Constructs a new instance with a provided CPU or GPU coarse quantizer;
faiss/gpu/GpuIndexIVFScalarQuantizer.h:    GpuIndexIVFScalarQuantizer(
faiss/gpu/GpuIndexIVFScalarQuantizer.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFScalarQuantizer.h:            GpuIndexIVFScalarQuantizerConfig config =
faiss/gpu/GpuIndexIVFScalarQuantizer.h:                    GpuIndexIVFScalarQuantizerConfig());
faiss/gpu/GpuIndexIVFScalarQuantizer.h:    ~GpuIndexIVFScalarQuantizer() override;
faiss/gpu/GpuIndexIVFScalarQuantizer.h:    /// Reserve GPU memory in our inverted lists for this number of vectors
faiss/gpu/GpuIndexIVFScalarQuantizer.h:    const GpuIndexIVFScalarQuantizerConfig ivfSQConfig_;
faiss/gpu/GpuIndexIVFScalarQuantizer.h:} // namespace gpu
faiss/gpu/StandardGpuResources.h: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/StandardGpuResources.h:#if defined USE_NVIDIA_RAFT
faiss/gpu/StandardGpuResources.h:#include <faiss/gpu/GpuResources.h>
faiss/gpu/StandardGpuResources.h:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/StandardGpuResources.h:#include <faiss/gpu/utils/StackDeviceMemory.h>
faiss/gpu/StandardGpuResources.h:namespace gpu {
faiss/gpu/StandardGpuResources.h:/// Standard implementation of the GpuResources object that provides for a
faiss/gpu/StandardGpuResources.h:class StandardGpuResourcesImpl : public GpuResources {
faiss/gpu/StandardGpuResources.h:    StandardGpuResourcesImpl();
faiss/gpu/StandardGpuResources.h:    ~StandardGpuResourcesImpl() override;
faiss/gpu/StandardGpuResources.h:    /// requests will call cudaMalloc / cudaFree at the point of use
faiss/gpu/StandardGpuResources.h:    /// all devices as temporary memory. This is the upper bound for the GPU
faiss/gpu/StandardGpuResources.h:    /// memory that we will reserve. We will never go above 1.5 GiB on any GPU;
faiss/gpu/StandardGpuResources.h:    /// smaller GPUs (with <= 4 GiB or <= 8 GiB) will use less memory than that.
faiss/gpu/StandardGpuResources.h:    /// Set amount of pinned memory to allocate, for async GPU <-> CPU
faiss/gpu/StandardGpuResources.h:    /// i.e., it will not be destroyed when the GpuResources object gets cleaned
faiss/gpu/StandardGpuResources.h:    /// We are guaranteed that all Faiss GPU work is ordered with respect to
faiss/gpu/StandardGpuResources.h:    /// this stream upon exit from an index or other Faiss GPU call.
faiss/gpu/StandardGpuResources.h:    void setDefaultStream(int device, cudaStream_t stream) override;
faiss/gpu/StandardGpuResources.h:    /// Returns the stream for the given device on which all Faiss GPU work is
faiss/gpu/StandardGpuResources.h:    /// We are guaranteed that all Faiss GPU work is ordered with respect to
faiss/gpu/StandardGpuResources.h:    /// this stream upon exit from an index or other Faiss GPU call.
faiss/gpu/StandardGpuResources.h:    cudaStream_t getDefaultStream(int device) override;
faiss/gpu/StandardGpuResources.h:#if defined USE_NVIDIA_RAFT
faiss/gpu/StandardGpuResources.h:    /// If enabled, will print every GPU memory allocation and deallocation to
faiss/gpu/StandardGpuResources.h:    std::vector<cudaStream_t> getAlternateStreams(int device) override;
faiss/gpu/StandardGpuResources.h:    /// Allocate non-temporary GPU memory
faiss/gpu/StandardGpuResources.h:    cudaStream_t getAsyncCopyStream(int device) override;
faiss/gpu/StandardGpuResources.h:    /// Have GPU resources been initialized for this device yet?
faiss/gpu/StandardGpuResources.h:    /// Adjust the default temporary memory allocation based on the total GPU
faiss/gpu/StandardGpuResources.h:    static size_t getDefaultTempMemForGPU(int device, size_t requested);
faiss/gpu/StandardGpuResources.h:    std::unordered_map<int, cudaStream_t> defaultStreams_;
faiss/gpu/StandardGpuResources.h:    std::unordered_map<int, cudaStream_t> userDefaultStreams_;
faiss/gpu/StandardGpuResources.h:    std::unordered_map<int, std::vector<cudaStream_t>> alternateStreams_;
faiss/gpu/StandardGpuResources.h:    /// Async copy stream to use for GPU <-> CPU pinned memory copies
faiss/gpu/StandardGpuResources.h:    std::unordered_map<int, cudaStream_t> asyncCopyStreams_;
faiss/gpu/StandardGpuResources.h:#if defined USE_NVIDIA_RAFT
faiss/gpu/StandardGpuResources.h:    /// Pinned memory allocation for use with this GPU
faiss/gpu/StandardGpuResources.h:    /// Whether or not we log every GPU memory allocation and deallocation
faiss/gpu/StandardGpuResources.h:/// Default implementation of GpuResources that allocates a cuBLAS
faiss/gpu/StandardGpuResources.h:/// Internally, the Faiss GPU code uses the instance managed by getResources,
faiss/gpu/StandardGpuResources.h:class StandardGpuResources : public GpuResourcesProvider {
faiss/gpu/StandardGpuResources.h:    StandardGpuResources();
faiss/gpu/StandardGpuResources.h:    ~StandardGpuResources() override;
faiss/gpu/StandardGpuResources.h:    std::shared_ptr<GpuResources> getResources() override;
faiss/gpu/StandardGpuResources.h:    /// requests will call cudaMalloc / cudaFree at the point of use
faiss/gpu/StandardGpuResources.h:    /// all devices as temporary memory. This is the upper bound for the GPU
faiss/gpu/StandardGpuResources.h:    /// memory that we will reserve. We will never go above 1.5 GiB on any GPU;
faiss/gpu/StandardGpuResources.h:    /// smaller GPUs (with <= 4 GiB or <= 8 GiB) will use less memory than that.
faiss/gpu/StandardGpuResources.h:    /// Set amount of pinned memory to allocate, for async GPU <-> CPU
faiss/gpu/StandardGpuResources.h:    /// i.e., it will not be destroyed when the GpuResources object gets cleaned
faiss/gpu/StandardGpuResources.h:    /// We are guaranteed that all Faiss GPU work is ordered with respect to
faiss/gpu/StandardGpuResources.h:    /// this stream upon exit from an index or other Faiss GPU call.
faiss/gpu/StandardGpuResources.h:    void setDefaultStream(int device, cudaStream_t stream);
faiss/gpu/StandardGpuResources.h:    cudaStream_t getDefaultStream(int device);
faiss/gpu/StandardGpuResources.h:#if defined USE_NVIDIA_RAFT
faiss/gpu/StandardGpuResources.h:    /// If enabled, will print every GPU memory allocation and deallocation to
faiss/gpu/StandardGpuResources.h:    std::shared_ptr<StandardGpuResourcesImpl> res_;
faiss/gpu/StandardGpuResources.h:} // namespace gpu
faiss/gpu/GpuIndexIVFPQ.cu:#include <faiss/gpu/GpuIndexFlat.h>
faiss/gpu/GpuIndexIVFPQ.cu:#include <faiss/gpu/GpuIndexIVFPQ.h>
faiss/gpu/GpuIndexIVFPQ.cu:#include <faiss/gpu/GpuResources.h>
faiss/gpu/GpuIndexIVFPQ.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/GpuIndexIVFPQ.cu:#include <faiss/gpu/impl/IVFPQ.cuh>
faiss/gpu/GpuIndexIVFPQ.cu:#include <faiss/gpu/utils/CopyUtils.cuh>
faiss/gpu/GpuIndexIVFPQ.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuIndexIVFPQ.cu:#include <faiss/gpu/utils/RaftUtils.h>
faiss/gpu/GpuIndexIVFPQ.cu:#include <faiss/gpu/impl/RaftIVFPQ.cuh>
faiss/gpu/GpuIndexIVFPQ.cu:namespace gpu {
faiss/gpu/GpuIndexIVFPQ.cu:GpuIndexIVFPQ::GpuIndexIVFPQ(
faiss/gpu/GpuIndexIVFPQ.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFPQ.cu:        GpuIndexIVFPQConfig config)
faiss/gpu/GpuIndexIVFPQ.cu:        : GpuIndexIVF(
faiss/gpu/GpuIndexIVFPQ.cu:GpuIndexIVFPQ::GpuIndexIVFPQ(
faiss/gpu/GpuIndexIVFPQ.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFPQ.cu:        GpuIndexIVFPQConfig config)
faiss/gpu/GpuIndexIVFPQ.cu:        : GpuIndexIVF(provider, dims, metric, 0, nlist, config),
faiss/gpu/GpuIndexIVFPQ.cu:GpuIndexIVFPQ::GpuIndexIVFPQ(
faiss/gpu/GpuIndexIVFPQ.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFPQ.cu:        GpuIndexIVFPQConfig config)
faiss/gpu/GpuIndexIVFPQ.cu:        : GpuIndexIVF(
faiss/gpu/GpuIndexIVFPQ.cu:            "GpuIndexIVFPQ: RAFT does not support separate coarseQuantizer");
faiss/gpu/GpuIndexIVFPQ.cu:GpuIndexIVFPQ::~GpuIndexIVFPQ() {}
faiss/gpu/GpuIndexIVFPQ.cu:void GpuIndexIVFPQ::copyFrom(const faiss::IndexIVFPQ* index) {
faiss/gpu/GpuIndexIVFPQ.cu:    // This will copy GpuIndexIVF data such as the coarse quantizer
faiss/gpu/GpuIndexIVFPQ.cu:    GpuIndexIVF::copyFrom(index);
faiss/gpu/GpuIndexIVFPQ.cu:            "GPU: only pq.nbits == 8 is supported");
faiss/gpu/GpuIndexIVFPQ.cu:            index->by_residual, "GPU: only by_residual = true is supported");
faiss/gpu/GpuIndexIVFPQ.cu:            index->polysemous_ht == 0, "GPU: polysemous codes not supported");
faiss/gpu/GpuIndexIVFPQ.cu:        // copied in GpuIndex::copyFrom
faiss/gpu/GpuIndexIVFPQ.cu:void GpuIndexIVFPQ::copyTo(faiss::IndexIVFPQ* index) const {
faiss/gpu/GpuIndexIVFPQ.cu:            "Cannot copy to CPU as GPU index doesn't retain "
faiss/gpu/GpuIndexIVFPQ.cu:    GpuIndexIVF::copyTo(index);
faiss/gpu/GpuIndexIVFPQ.cu:void GpuIndexIVFPQ::reserveMemory(size_t numVecs) {
faiss/gpu/GpuIndexIVFPQ.cu:void GpuIndexIVFPQ::setPrecomputedCodes(bool enable) {
faiss/gpu/GpuIndexIVFPQ.cu:bool GpuIndexIVFPQ::getPrecomputedCodes() const {
faiss/gpu/GpuIndexIVFPQ.cu:int GpuIndexIVFPQ::getNumSubQuantizers() const {
faiss/gpu/GpuIndexIVFPQ.cu:int GpuIndexIVFPQ::getBitsPerCode() const {
faiss/gpu/GpuIndexIVFPQ.cu:int GpuIndexIVFPQ::getCentroidsPerSubQuantizer() const {
faiss/gpu/GpuIndexIVFPQ.cu:size_t GpuIndexIVFPQ::reclaimMemory() {
faiss/gpu/GpuIndexIVFPQ.cu:void GpuIndexIVFPQ::reset() {
faiss/gpu/GpuIndexIVFPQ.cu:void GpuIndexIVFPQ::updateQuantizer() {
faiss/gpu/GpuIndexIVFPQ.cu:void GpuIndexIVFPQ::trainResidualQuantizer_(idx_t n, const float* x) {
faiss/gpu/GpuIndexIVFPQ.cu:    // For PQ training purposes, accelerate it by using a GPU clustering index
faiss/gpu/GpuIndexIVFPQ.cu:            GpuIndexFlatConfig config;
faiss/gpu/GpuIndexIVFPQ.cu:            GpuIndexFlatL2 pqIndex(resources_, pq.dsub, config);
faiss/gpu/GpuIndexIVFPQ.cu:void GpuIndexIVFPQ::train(idx_t n, const float* x) {
faiss/gpu/GpuIndexIVFPQ.cu:    // to the classical GPU impl
faiss/gpu/GpuIndexIVFPQ.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuIndexIVFPQ.cu:        // FIXME: GPUize more of this
faiss/gpu/GpuIndexIVFPQ.cu:void GpuIndexIVFPQ::setIndex_(
faiss/gpu/GpuIndexIVFPQ.cu:        GpuResources* resources,
faiss/gpu/GpuIndexIVFPQ.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuIndexIVFPQ.cu:void GpuIndexIVFPQ::verifyPQSettings_() const {
faiss/gpu/GpuIndexIVFPQ.cu:} // namespace gpu
faiss/gpu/GpuIcmEncoder.cu:#include <faiss/gpu/GpuIcmEncoder.h>
faiss/gpu/GpuIcmEncoder.cu:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/GpuIcmEncoder.cu:#include <faiss/gpu/impl/IcmEncoder.cuh>
faiss/gpu/GpuIcmEncoder.cu:namespace gpu {
faiss/gpu/GpuIcmEncoder.cu:///< A helper structure to support multi-GPU
faiss/gpu/GpuIcmEncoder.cu:GpuIcmEncoder::GpuIcmEncoder(
faiss/gpu/GpuIcmEncoder.cu:        const std::vector<GpuResourcesProvider*>& provs,
faiss/gpu/GpuIcmEncoder.cu:GpuIcmEncoder::~GpuIcmEncoder() {}
faiss/gpu/GpuIcmEncoder.cu:void GpuIcmEncoder::set_binary_term() {
faiss/gpu/GpuIcmEncoder.cu:void GpuIcmEncoder::encode(
faiss/gpu/GpuIcmEncoder.cu:GpuIcmEncoderFactory::GpuIcmEncoderFactory(int ngpus) {
faiss/gpu/GpuIcmEncoder.cu:    for (int i = 0; i < ngpus; i++) {
faiss/gpu/GpuIcmEncoder.cu:        provs.push_back(new StandardGpuResources());
faiss/gpu/GpuIcmEncoder.cu:lsq::IcmEncoder* GpuIcmEncoderFactory::get(const LocalSearchQuantizer* lsq) {
faiss/gpu/GpuIcmEncoder.cu:    return new GpuIcmEncoder(lsq, provs, devices);
faiss/gpu/GpuIcmEncoder.cu:} // namespace gpu
faiss/gpu/GpuAutoTune.cpp:#include <faiss/gpu/GpuAutoTune.h>
faiss/gpu/GpuAutoTune.cpp:#include <faiss/gpu/GpuIndex.h>
faiss/gpu/GpuAutoTune.cpp:#include <faiss/gpu/GpuIndexFlat.h>
faiss/gpu/GpuAutoTune.cpp:#include <faiss/gpu/GpuIndexIVFFlat.h>
faiss/gpu/GpuAutoTune.cpp:#include <faiss/gpu/GpuIndexIVFPQ.h>
faiss/gpu/GpuAutoTune.cpp:#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
faiss/gpu/GpuAutoTune.cpp:#include <faiss/gpu/impl/IndexUtils.h>
faiss/gpu/GpuAutoTune.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/GpuAutoTune.cpp:namespace gpu {
faiss/gpu/GpuAutoTune.cpp: * Parameters to auto-tune on GpuIndex'es
faiss/gpu/GpuAutoTune.cpp:void GpuParameterSpace::initialize(const Index* index) {
faiss/gpu/GpuAutoTune.cpp:    if (DC(GpuIndexIVF)) {
faiss/gpu/GpuAutoTune.cpp:void GpuParameterSpace::set_index_parameter(
faiss/gpu/GpuAutoTune.cpp:        if (DC(GpuIndexIVF)) {
faiss/gpu/GpuAutoTune.cpp:        if (DC(GpuIndexIVFPQ)) {
faiss/gpu/GpuAutoTune.cpp:        if (DC(GpuIndexIVF)) {
faiss/gpu/GpuAutoTune.cpp:} // namespace gpu
faiss/gpu/test/test_gpu_index_ivfsq.py:    res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index_ivfsq.py:    config = faiss.GpuIndexIVFScalarQuantizerConfig()
faiss/gpu/test/test_gpu_index_ivfsq.py:    idx_gpu = faiss.GpuIndexIVFScalarQuantizer(res, idx_cpu, config)
faiss/gpu/test/test_gpu_index_ivfsq.py:    return idx_cpu, idx_gpu
faiss/gpu/test/test_gpu_index_ivfsq.py:def make_indices_copy_from_gpu(nlist, d, qtype, by_residual, metric, clamp):
faiss/gpu/test/test_gpu_index_ivfsq.py:    res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index_ivfsq.py:    config = faiss.GpuIndexIVFScalarQuantizerConfig()
faiss/gpu/test/test_gpu_index_ivfsq.py:    idx_gpu = faiss.GpuIndexIVFScalarQuantizer(res, d, nlist,
faiss/gpu/test/test_gpu_index_ivfsq.py:    idx_gpu.train(to_train)
faiss/gpu/test/test_gpu_index_ivfsq.py:    idx_gpu.add(to_train)
faiss/gpu/test/test_gpu_index_ivfsq.py:    idx_gpu.copyTo(idx_cpu)
faiss/gpu/test/test_gpu_index_ivfsq.py:    return idx_cpu, idx_gpu
faiss/gpu/test/test_gpu_index_ivfsq.py:    res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index_ivfsq.py:    config = faiss.GpuIndexIVFScalarQuantizerConfig()
faiss/gpu/test/test_gpu_index_ivfsq.py:    idx_gpu = faiss.GpuIndexIVFScalarQuantizer(res, d, nlist,
faiss/gpu/test/test_gpu_index_ivfsq.py:    assert(by_residual == idx_gpu.by_residual)
faiss/gpu/test/test_gpu_index_ivfsq.py:    idx_gpu.train(to_train)
faiss/gpu/test/test_gpu_index_ivfsq.py:    idx_gpu.add(to_train)
faiss/gpu/test/test_gpu_index_ivfsq.py:    return idx_cpu, idx_gpu
faiss/gpu/test/test_gpu_index_ivfsq.py:    ci, gi = make_indices_copy_from_gpu(nlist, d, qtype,
faiss/gpu/test/test_gpu_index.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index.py:        config = faiss.GpuIndexIVFFlatConfig()
faiss/gpu/test/test_gpu_index.py:        idx_gpu = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2, config)
faiss/gpu/test/test_gpu_index.py:        idx_gpu.nprobe = nprobe
faiss/gpu/test/test_gpu_index.py:        idx_gpu.train(xb)
faiss/gpu/test/test_gpu_index.py:        idx_gpu.add(xb)
faiss/gpu/test/test_gpu_index.py:        q_d, q_i = idx_gpu.quantizer.search(xq, nprobe)
faiss/gpu/test/test_gpu_index.py:            idx_gpu, xq, k, q_i, q_d)
faiss/gpu/test/test_gpu_index.py:        d, i = idx_gpu.search(xq, k)
faiss/gpu/test/test_gpu_index.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index.py:        config = faiss.GpuIndexIVFPQConfig()
faiss/gpu/test/test_gpu_index.py:        idx_gpu = faiss.GpuIndexIVFPQ(res, d, nlist, 4, 8, faiss.METRIC_L2, config)
faiss/gpu/test/test_gpu_index.py:        idx_gpu.nprobe = nprobe
faiss/gpu/test/test_gpu_index.py:        idx_gpu.train(xb)
faiss/gpu/test/test_gpu_index.py:        idx_gpu.add(xb)
faiss/gpu/test/test_gpu_index.py:        q_d, q_i = idx_gpu.quantizer.search(xq, nprobe)
faiss/gpu/test/test_gpu_index.py:            idx_gpu, xq, k, q_i, q_d)
faiss/gpu/test/test_gpu_index.py:        d, i = idx_gpu.search(xq, k)
faiss/gpu/test/test_gpu_index.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index.py:        idx_gpu = faiss.GpuIndexIVFScalarQuantizer(
faiss/gpu/test/test_gpu_index.py:        idx_gpu.nprobe = nprobe
faiss/gpu/test/test_gpu_index.py:        idx_gpu.train(xb)
faiss/gpu/test/test_gpu_index.py:        idx_gpu.add(xb)
faiss/gpu/test/test_gpu_index.py:        q_d, q_i = idx_gpu.quantizer.search(xq, nprobe)
faiss/gpu/test/test_gpu_index.py:            idx_gpu, xq, k, q_i, q_d)
faiss/gpu/test/test_gpu_index.py:        d, i = idx_gpu.search(xq, k)
faiss/gpu/test/test_gpu_index.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index.py:        # construct a GPU index using the same trained coarse quantizer
faiss/gpu/test/test_gpu_index.py:        config = faiss.GpuIndexIVFFlatConfig()
faiss/gpu/test/test_gpu_index.py:        idx_gpu = faiss.GpuIndexIVFFlat(res, q, d, nlist, faiss.METRIC_L2, config)
faiss/gpu/test/test_gpu_index.py:        assert(idx_gpu.is_trained)
faiss/gpu/test/test_gpu_index.py:        idx_gpu.add(xb)
faiss/gpu/test/test_gpu_index.py:        idx_gpu.nprobe = nprobe
faiss/gpu/test/test_gpu_index.py:        d_g, i_g = idx_gpu.search(xq, k)
faiss/gpu/test/test_gpu_index.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index.py:        # construct a GPU index using the same trained coarse quantizer
faiss/gpu/test/test_gpu_index.py:        idx_gpu = faiss.GpuIndexIVFScalarQuantizer(
faiss/gpu/test/test_gpu_index.py:        assert(not idx_gpu.is_trained)
faiss/gpu/test/test_gpu_index.py:        idx_gpu.train(xb)
faiss/gpu/test/test_gpu_index.py:        idx_gpu.add(xb)
faiss/gpu/test/test_gpu_index.py:        idx_gpu.nprobe = nprobe
faiss/gpu/test/test_gpu_index.py:        d_g, i_g = idx_gpu.search(xq, k)
faiss/gpu/test/test_gpu_index.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index.py:        # construct a GPU index using the same trained coarse quantizer
faiss/gpu/test/test_gpu_index.py:        config = faiss.GpuIndexIVFPQConfig()
faiss/gpu/test/test_gpu_index.py:        idx_gpu = faiss.GpuIndexIVFPQ(
faiss/gpu/test/test_gpu_index.py:        assert(not idx_gpu.is_trained)
faiss/gpu/test/test_gpu_index.py:        idx_gpu.train(xb)
faiss/gpu/test/test_gpu_index.py:        idx_gpu.add(xb)
faiss/gpu/test/test_gpu_index.py:        idx_gpu.nprobe = nprobe_lvl_2
faiss/gpu/test/test_gpu_index.py:            idx_gpu.setPrecomputedCodes(use_precomputed)
faiss/gpu/test/test_gpu_index.py:            d_g, i_g = idx_gpu.search(xq, k)
faiss/gpu/test/test_gpu_index.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index.py:            config = faiss.GpuIndexIVFPQConfig()
faiss/gpu/test/test_gpu_index.py:            idx_gpu = faiss.GpuIndexIVFPQ(res, d, nlist, sub_q, bits_per_code, faiss.METRIC_L2, config)
faiss/gpu/test/test_gpu_index.py:            idx_gpu.train(xb)
faiss/gpu/test/test_gpu_index.py:            idx_gpu.add(xb)
faiss/gpu/test/test_gpu_index.py:            idx_gpu.copyTo(idx_cpu)
faiss/gpu/test/test_gpu_index.py:            idx_gpu.nprobe = nprobe
faiss/gpu/test/test_gpu_index.py:            d_g, i_g = idx_gpu.search(xq, k)
faiss/gpu/test/test_gpu_index.py:            idx_gpu.setPrecomputedCodes(True)
faiss/gpu/test/test_gpu_index.py:            d_g, i_g = idx_gpu.search(xq, k)
faiss/gpu/test/test_gpu_index.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index.py:            config = faiss.GpuIndexIVFPQConfig()
faiss/gpu/test/test_gpu_index.py:            idx_gpu = faiss.GpuIndexIVFPQ(res, d, nlist, sub_q, bits_per_code, faiss.METRIC_L2, config)
faiss/gpu/test/test_gpu_index.py:            idx_gpu.train(xb)
faiss/gpu/test/test_gpu_index.py:            idx_gpu.add(xb)
faiss/gpu/test/test_gpu_index.py:            idx_gpu.copyTo(idx_cpu)
faiss/gpu/test/test_gpu_index.py:            idx_gpu.nprobe = nprobe
faiss/gpu/test/test_gpu_index.py:            d_g, i_g = idx_gpu.search(xq, 10)
faiss/gpu/test/test_gpu_index.py:            idx_gpu.setPrecomputedCodes(True)
faiss/gpu/test/test_gpu_index.py:            d_g, i_g = idx_gpu.search(xq, 10)
faiss/gpu/test/test_gpu_index.py:    def test_copy_to_gpu(self):
faiss/gpu/test/test_gpu_index.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index.py:            config = faiss.GpuIndexIVFPQConfig()
faiss/gpu/test/test_gpu_index.py:            idx_gpu = faiss.GpuIndexIVFPQ(res, d, nlist, sub_q, bits_per_code, faiss.METRIC_L2, config)
faiss/gpu/test/test_gpu_index.py:            idx_gpu.copyFrom(idx_cpu)
faiss/gpu/test/test_gpu_index.py:            idx_gpu.nprobe = nprobe
faiss/gpu/test/test_gpu_index.py:            d_g, i_g = idx_gpu.search(xq, 10)
faiss/gpu/test/test_gpu_index.py:            idx_gpu.setPrecomputedCodes(True)
faiss/gpu/test/test_gpu_index.py:            d_g, i_g = idx_gpu.search(xq, 10)
faiss/gpu/test/test_gpu_index.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index.py:        config = faiss.GpuIndexIVFFlatConfig()
faiss/gpu/test/test_gpu_index.py:        idx = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2, config)
faiss/gpu/test/test_gpu_index.py:        idx = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2, config)
faiss/gpu/test/test_gpu_index.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index.py:        config = faiss.GpuIndexIVFPQConfig()
faiss/gpu/test/test_gpu_index.py:        idx = faiss.GpuIndexIVFPQ(res, d, nlist, M, nbits,
faiss/gpu/test/test_gpu_index.py:        idx = faiss.GpuIndexIVFPQ(res, d, nlist, M, nbits,
faiss/gpu/test/test_gpu_index.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index.py:        config = faiss.GpuIndexIVFScalarQuantizerConfig()
faiss/gpu/test/test_gpu_index.py:        idx = faiss.GpuIndexIVFScalarQuantizer(res, d, nlist, qtype,
faiss/gpu/test/test_gpu_index.py:        idx = faiss.GpuIndexIVFScalarQuantizer(res, d, nlist, qtype,
faiss/gpu/test/test_gpu_index.py:class TestSQ_to_gpu(unittest.TestCase):
faiss/gpu/test/test_gpu_index.py:    def test_sq_cpu_to_gpu(self):
faiss/gpu/test/test_gpu_index.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index.py:        config = faiss.GpuClonerOptions()
faiss/gpu/test/test_gpu_index.py:        gpu_index = faiss.index_cpu_to_gpu(res, 0, index, config)
faiss/gpu/test/test_gpu_index.py:        self.assertIsInstance(gpu_index, faiss.GpuIndexFlat)
faiss/gpu/test/test_gpu_index.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index.py:        config = faiss.GpuIndexIVFPQConfig()
faiss/gpu/test/test_gpu_index.py:        idx = faiss.GpuIndexIVFPQ(res, d, nlist, M, nbits,
faiss/gpu/test/test_gpu_index.py:    def subtest_gpu_encoding(self, ngpus):
faiss/gpu/test/test_gpu_index.py:        lsq.icm_encoder_factory = faiss.GpuIcmEncoderFactory(ngpus)
faiss/gpu/test/test_gpu_index.py:        err_gpu = self.eval_codec(lsq, xb)
faiss/gpu/test/test_gpu_index.py:        # 13804.411 vs 13814.794, 1 gpu
faiss/gpu/test/test_gpu_index.py:        print(err_gpu, err_cpu)
faiss/gpu/test/test_gpu_index.py:        self.assertLess(err_gpu, err_cpu * 1.05)
faiss/gpu/test/test_gpu_index.py:    def test_one_gpu(self):
faiss/gpu/test/test_gpu_index.py:        self.subtest_gpu_encoding(1)
faiss/gpu/test/test_gpu_index.py:    def test_multiple_gpu(self):
faiss/gpu/test/test_gpu_index.py:        ngpu = faiss.get_num_gpus()
faiss/gpu/test/test_gpu_index.py:        self.subtest_gpu_encoding(ngpu)
faiss/gpu/test/test_gpu_index.py:class TestGpuAutoTune(unittest.TestCase):
faiss/gpu/test/test_gpu_index.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index.py:        options = faiss.GpuClonerOptions()
faiss/gpu/test/test_gpu_index.py:        index = faiss.index_cpu_to_gpu(res, 0, index, options)
faiss/gpu/test/test_gpu_index.py:        ps = faiss.GpuParameterSpace()
faiss/gpu/test/TestUtils.cpp:#include <cuda_fp16.h>
faiss/gpu/test/TestUtils.cpp:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/test/TestUtils.cpp:namespace gpu {
faiss/gpu/test/TestUtils.cpp:    faiss::gpu::compareLists(
faiss/gpu/test/TestUtils.cpp:    auto queryVecs = faiss::gpu::randVecs(numQuery, dim);
faiss/gpu/test/TestUtils.cpp:} // namespace gpu
faiss/gpu/test/test_multi_gpu.py:    @unittest.skipIf(faiss.get_num_gpus() < 2, "multiple GPU only test")
faiss/gpu/test/test_multi_gpu.py:        assert faiss.get_num_gpus() > 1
faiss/gpu/test/test_multi_gpu.py:        co = faiss.GpuMultipleClonerOptions()
faiss/gpu/test/test_multi_gpu.py:        index = faiss.index_cpu_to_all_gpus(index_cpu, co, ngpu=2)
faiss/gpu/test/test_multi_gpu.py:        index2 = faiss.index_cpu_to_all_gpus(index_cpu, co, ngpu=2)
faiss/gpu/test/test_multi_gpu.py:    @unittest.skipIf(faiss.get_num_gpus() < 2, "multiple GPU only test")
faiss/gpu/test/test_multi_gpu.py:        co = faiss.GpuMultipleClonerOptions()
faiss/gpu/test/test_multi_gpu.py:        index = faiss.index_cpu_to_all_gpus(index, co, ngpu=2)
faiss/gpu/test/test_multi_gpu.py:        faiss.GpuParameterSpace().set_index_parameter(index, 'nprobe', 8)
faiss/gpu/test/test_multi_gpu.py:    def test_binary_clone(self, ngpu=1, shard=False):
faiss/gpu/test/test_multi_gpu.py:        co = faiss.GpuMultipleClonerOptions()
faiss/gpu/test/test_multi_gpu.py:        # index2 = faiss.index_cpu_to_all_gpus(index, ngpu=ngpu)
faiss/gpu/test/test_multi_gpu.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_multi_gpu.py:        index2 = faiss.GpuIndexBinaryFlat(res, index)
faiss/gpu/test/test_multi_gpu.py:        self.test_binary_clone(ngpu=2, shard=False)
faiss/gpu/test/test_multi_gpu.py:        self.test_binary_clone(ngpu=2, shard=True)
faiss/gpu/test/test_multi_gpu.py:# This class also has a multi-GPU test within
faiss/gpu/test/test_multi_gpu.py:    def do_cpu_to_gpu(self, index_key):
faiss/gpu/test/test_multi_gpu.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_multi_gpu.py:        co = faiss.GpuClonerOptions()
faiss/gpu/test/test_multi_gpu.py:        gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)
faiss/gpu/test/test_multi_gpu.py:        gpu_index.nprobe = 4
faiss/gpu/test/test_multi_gpu.py:        Dnew, Inew = gpu_index.search(xq, 10)
faiss/gpu/test/test_multi_gpu.py:        if faiss.get_num_gpus() == 1:
faiss/gpu/test/test_multi_gpu.py:            # test on just 2 GPUs
faiss/gpu/test/test_multi_gpu.py:            res = [faiss.StandardGpuResources() for i in range(2)]
faiss/gpu/test/test_multi_gpu.py:            co = faiss.GpuMultipleClonerOptions()
faiss/gpu/test/test_multi_gpu.py:            gpu_index = faiss.index_cpu_to_gpu_multiple_py(res, index, co)
faiss/gpu/test/test_multi_gpu.py:            faiss.GpuParameterSpace().set_index_parameter(
faiss/gpu/test/test_multi_gpu.py:                gpu_index, 'nprobe', 4)
faiss/gpu/test/test_multi_gpu.py:            Dnew, Inew = gpu_index.search(xq, 10)
faiss/gpu/test/test_multi_gpu.py:    def test_cpu_to_gpu_IVFPQ(self):
faiss/gpu/test/test_multi_gpu.py:        self.do_cpu_to_gpu('IVF128,PQ4')
faiss/gpu/test/test_multi_gpu.py:    def test_cpu_to_gpu_IVFFlat(self):
faiss/gpu/test/test_multi_gpu.py:        self.do_cpu_to_gpu('IVF128,Flat')
faiss/gpu/test/test_multi_gpu.py:    def test_set_gpu_param(self):
faiss/gpu/test/test_multi_gpu.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_multi_gpu.py:        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
faiss/gpu/test/test_multi_gpu.py:        faiss.GpuParameterSpace().set_index_parameter(gpu_index, "nprobe", 3)
faiss/gpu/test/TestGpuSelect.cu:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/test/TestGpuSelect.cu:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/test/TestGpuSelect.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/test/TestGpuSelect.cu:#include <faiss/gpu/utils/BlockSelectKernel.cuh>
faiss/gpu/test/TestGpuSelect.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/test/TestGpuSelect.cu:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/test/TestGpuSelect.cu:#include <faiss/gpu/utils/HostTensor.cuh>
faiss/gpu/test/TestGpuSelect.cu:#include <faiss/gpu/utils/WarpSelectKernel.cuh>
faiss/gpu/test/TestGpuSelect.cu:    using namespace faiss::gpu;
faiss/gpu/test/TestGpuSelect.cu:    StandardGpuResources res;
faiss/gpu/test/TestGpuSelect.cu:    // Select top-k on GPU
faiss/gpu/test/TestGpuSelect.cu:    DeviceTensor<float, 2, true> gpuVal(
faiss/gpu/test/TestGpuSelect.cu:    DeviceTensor<float, 2, true> gpuOutVal(
faiss/gpu/test/TestGpuSelect.cu:    DeviceTensor<idx_t, 2, true> gpuOutInd(
faiss/gpu/test/TestGpuSelect.cu:        runWarpSelect(gpuVal, gpuOutVal, gpuOutInd, dir, k, 0);
faiss/gpu/test/TestGpuSelect.cu:        runBlockSelect(gpuVal, gpuOutVal, gpuOutInd, dir, k, 0);
faiss/gpu/test/TestGpuSelect.cu:    HostTensor<float, 2, true> outVal(gpuOutVal, 0);
faiss/gpu/test/TestGpuSelect.cu:    HostTensor<idx_t, 2, true> outInd(gpuOutInd, 0);
faiss/gpu/test/TestGpuSelect.cu:            float gpuV = outVal[r][i];
faiss/gpu/test/TestGpuSelect.cu:            EXPECT_EQ(gpuV, cpuV)
faiss/gpu/test/TestGpuSelect.cu:            // differ, because the order in which the GPU will see the
faiss/gpu/test/TestGpuSelect.cu:            idx_t gpuInd = outInd[r][i];
faiss/gpu/test/TestGpuSelect.cu:            auto itSeenIndex = seenIndices.find(gpuInd);
faiss/gpu/test/TestGpuSelect.cu:                    << "Row " << r << " user index " << gpuInd
faiss/gpu/test/TestGpuSelect.cu:            seenIndices[gpuInd] = i;
faiss/gpu/test/TestGpuSelect.cu:            if (gpuInd != cpuInd) {
faiss/gpu/test/TestGpuSelect.cu:                float gpuGatherV = hostVal[r][gpuInd];
faiss/gpu/test/TestGpuSelect.cu:                EXPECT_EQ(gpuGatherV, cpuGatherV)
faiss/gpu/test/TestGpuSelect.cu:                        << " source ind " << gpuInd << " " << cpuInd;
faiss/gpu/test/TestGpuSelect.cu:TEST(TestGpuSelect, test) {
faiss/gpu/test/TestGpuSelect.cu:        int rows = faiss::gpu::randVal(10, 100);
faiss/gpu/test/TestGpuSelect.cu:        int cols = faiss::gpu::randVal(1, 30000);
faiss/gpu/test/TestGpuSelect.cu:        int k = std::min(cols, faiss::gpu::randVal(1, GPU_MAX_SELECTION_K));
faiss/gpu/test/TestGpuSelect.cu:        bool dir = faiss::gpu::randBool();
faiss/gpu/test/TestGpuSelect.cu:TEST(TestGpuSelect, test1) {
faiss/gpu/test/TestGpuSelect.cu:        int rows = faiss::gpu::randVal(10, 100);
faiss/gpu/test/TestGpuSelect.cu:        int cols = faiss::gpu::randVal(1, 30000);
faiss/gpu/test/TestGpuSelect.cu:        bool dir = faiss::gpu::randBool();
faiss/gpu/test/TestGpuSelect.cu:TEST(TestGpuSelect, testExact) {
faiss/gpu/test/TestGpuSelect.cu:        int rows = faiss::gpu::randVal(10, 100);
faiss/gpu/test/TestGpuSelect.cu:        int cols = faiss::gpu::randVal(1, GPU_MAX_SELECTION_K);
faiss/gpu/test/TestGpuSelect.cu:        bool dir = faiss::gpu::randBool();
faiss/gpu/test/TestGpuSelect.cu:TEST(TestGpuSelect, testWarp) {
faiss/gpu/test/TestGpuSelect.cu:        int rows = faiss::gpu::randVal(10, 100);
faiss/gpu/test/TestGpuSelect.cu:        int cols = faiss::gpu::randVal(1, 30000);
faiss/gpu/test/TestGpuSelect.cu:        int k = std::min(cols, faiss::gpu::randVal(1, GPU_MAX_SELECTION_K));
faiss/gpu/test/TestGpuSelect.cu:        bool dir = faiss::gpu::randBool();
faiss/gpu/test/TestGpuSelect.cu:TEST(TestGpuSelect, test1Warp) {
faiss/gpu/test/TestGpuSelect.cu:        int rows = faiss::gpu::randVal(10, 100);
faiss/gpu/test/TestGpuSelect.cu:        int cols = faiss::gpu::randVal(1, 30000);
faiss/gpu/test/TestGpuSelect.cu:        bool dir = faiss::gpu::randBool();
faiss/gpu/test/TestGpuSelect.cu:TEST(TestGpuSelect, testExactWarp) {
faiss/gpu/test/TestGpuSelect.cu:        int rows = faiss::gpu::randVal(10, 100);
faiss/gpu/test/TestGpuSelect.cu:        int cols = faiss::gpu::randVal(1, GPU_MAX_SELECTION_K);
faiss/gpu/test/TestGpuSelect.cu:        bool dir = faiss::gpu::randBool();
faiss/gpu/test/TestGpuSelect.cu:    faiss::gpu::setTestSeed(100);
faiss/gpu/test/torch_test_contrib_gpu.py:class TestTorchUtilsGPU(unittest.TestCase):
faiss/gpu/test/torch_test_contrib_gpu.py:        # Add to CPU index with torch GPU (should fail)
faiss/gpu/test/torch_test_contrib_gpu.py:        xb_torch_gpu = torch.rand(10000, 128, device=torch.device('cuda', 0), dtype=torch.float32)
faiss/gpu/test/torch_test_contrib_gpu.py:            cpu_index.add(xb_torch_gpu)
faiss/gpu/test/torch_test_contrib_gpu.py:        # Add to GPU with torch GPU
faiss/gpu/test/torch_test_contrib_gpu.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/torch_test_contrib_gpu.py:        gpu_index = faiss.GpuIndexFlatL2(res, 128)
faiss/gpu/test/torch_test_contrib_gpu.py:        gpu_index.add(xb_torch.cuda())
faiss/gpu/test/torch_test_contrib_gpu.py:        d_torch_cpu, i_torch_cpu = gpu_index.search(xq_torch_cpu, 10)
faiss/gpu/test/torch_test_contrib_gpu.py:        # Search with torch GPU
faiss/gpu/test/torch_test_contrib_gpu.py:        xq_torch_gpu = xq_torch_cpu.cuda()
faiss/gpu/test/torch_test_contrib_gpu.py:        d_torch_gpu, i_torch_gpu = gpu_index.search(xq_torch_gpu, 10)
faiss/gpu/test/torch_test_contrib_gpu.py:        self.assertTrue(d_torch_gpu.is_cuda)
faiss/gpu/test/torch_test_contrib_gpu.py:        self.assertTrue(i_torch_gpu.is_cuda)
faiss/gpu/test/torch_test_contrib_gpu.py:        self.assertTrue(torch.equal(d_torch_cpu.cuda(), d_torch_gpu))
faiss/gpu/test/torch_test_contrib_gpu.py:        self.assertTrue(torch.equal(i_torch_cpu.cuda(), i_torch_gpu))
faiss/gpu/test/torch_test_contrib_gpu.py:        # Search with torch GPU using pre-allocated arrays
faiss/gpu/test/torch_test_contrib_gpu.py:        new_d_torch_gpu = torch.zeros(10, 10, device=torch.device('cuda', 0), dtype=torch.float32)
faiss/gpu/test/torch_test_contrib_gpu.py:        new_i_torch_gpu = torch.zeros(10, 10, device=torch.device('cuda', 0), dtype=torch.int64)
faiss/gpu/test/torch_test_contrib_gpu.py:        gpu_index.search(xq_torch_gpu, 10, new_d_torch_gpu, new_i_torch_gpu)
faiss/gpu/test/torch_test_contrib_gpu.py:        self.assertTrue(torch.equal(d_torch_cpu.cuda(), new_d_torch_gpu))
faiss/gpu/test/torch_test_contrib_gpu.py:        self.assertTrue(torch.equal(i_torch_cpu.cuda(), new_i_torch_gpu))
faiss/gpu/test/torch_test_contrib_gpu.py:        d_np_cpu, i_np_cpu = gpu_index.search(xq_np_cpu, 10)
faiss/gpu/test/torch_test_contrib_gpu.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/torch_test_contrib_gpu.py:        index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2)
faiss/gpu/test/torch_test_contrib_gpu.py:        xb = torch.rand(1000, d, device=torch.device('cuda', 0), dtype=torch.float32)
faiss/gpu/test/torch_test_contrib_gpu.py:        ids = torch.arange(1000, 1000 + xb.shape[0], device=torch.device('cuda', 0), dtype=torch.int64)
faiss/gpu/test/torch_test_contrib_gpu.py:        # Test add_with_ids with torch gpu
faiss/gpu/test/torch_test_contrib_gpu.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/torch_test_contrib_gpu.py:        index = faiss.GpuIndexFlatL2(res, d)
faiss/gpu/test/torch_test_contrib_gpu.py:        xb = torch.rand(100, d, device=torch.device('cuda', 0), dtype=torch.float32)
faiss/gpu/test/torch_test_contrib_gpu.py:        # Test reconstruct with torch gpu (native return)
faiss/gpu/test/torch_test_contrib_gpu.py:        self.assertTrue(y.is_cuda)
faiss/gpu/test/torch_test_contrib_gpu.py:        # Test reconstruct with torch gpu output providesd
faiss/gpu/test/torch_test_contrib_gpu.py:        y = torch.empty(d, device=torch.device('cuda', 0), dtype=torch.float32)
faiss/gpu/test/torch_test_contrib_gpu.py:        # Test reconstruct_n with torch gpu (native return)
faiss/gpu/test/torch_test_contrib_gpu.py:        self.assertTrue(y.is_cuda)
faiss/gpu/test/torch_test_contrib_gpu.py:        # Test reconstruct_n with torch gpu output provided
faiss/gpu/test/torch_test_contrib_gpu.py:        y = torch.empty(10, d, device=torch.device('cuda', 0), dtype=torch.float32)
faiss/gpu/test/torch_test_contrib_gpu.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/torch_test_contrib_gpu.py:        config = faiss.GpuIndexIVFFlatConfig()
faiss/gpu/test/torch_test_contrib_gpu.py:        index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2, config)
faiss/gpu/test/torch_test_contrib_gpu.py:        xb = torch.rand(100, d, device=torch.device('cuda', 0), dtype=torch.float32)
faiss/gpu/test/torch_test_contrib_gpu.py:        # Test reconstruct_n with torch gpu (native return)
faiss/gpu/test/torch_test_contrib_gpu.py:        self.assertTrue(y.is_cuda)
faiss/gpu/test/torch_test_contrib_gpu.py:        # Test reconstruct_n with torch gpu output provided
faiss/gpu/test/torch_test_contrib_gpu.py:        y = torch.empty(10, d, device=torch.device('cuda', 0), dtype=torch.float32)
faiss/gpu/test/torch_test_contrib_gpu.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/torch_test_contrib_gpu.py:        index = faiss.GpuIndexFlatL2(res, d)
faiss/gpu/test/torch_test_contrib_gpu.py:        xb = torch.rand(10000, d, device=torch.device('cuda', 0), dtype=torch.float32)
faiss/gpu/test/torch_test_contrib_gpu.py:        # Test assign with native gpu output
faiss/gpu/test/torch_test_contrib_gpu.py:        # both input as gpu torch and input as cpu torch
faiss/gpu/test/torch_test_contrib_gpu.py:        xq = torch.rand(10, d, device=torch.device('cuda', 0), dtype=torch.float32)
faiss/gpu/test/torch_test_contrib_gpu.py:        # This is not currently implemented on GPU indices
faiss/gpu/test/torch_test_contrib_gpu.py:        # This is not currently implemented on GPU indices
faiss/gpu/test/torch_test_contrib_gpu.py:        # This is not currently implemented on GPU indices
faiss/gpu/test/torch_test_contrib_gpu.py:        # This is not currently implemented on GPU indices
faiss/gpu/test/torch_test_contrib_gpu.py:class TestTorchUtilsKnnGpu(unittest.TestCase):
faiss/gpu/test/torch_test_contrib_gpu.py:    def test_knn_gpu(self, use_raft=False):
faiss/gpu/test/torch_test_contrib_gpu.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/torch_test_contrib_gpu.py:        # for the GPU, we'll use a non-default stream
faiss/gpu/test/torch_test_contrib_gpu.py:        s = torch.cuda.Stream()
faiss/gpu/test/torch_test_contrib_gpu.py:        with torch.cuda.stream(s):
faiss/gpu/test/torch_test_contrib_gpu.py:                    D, I = faiss.knn_gpu(res, xq_c, xb_c, k, use_raft=use_raft)
faiss/gpu/test/torch_test_contrib_gpu.py:            # test torch (cpu, gpu) inputs
faiss/gpu/test/torch_test_contrib_gpu.py:            for is_cuda in True, False:
faiss/gpu/test/torch_test_contrib_gpu.py:                        if is_cuda:
faiss/gpu/test/torch_test_contrib_gpu.py:                            xq_c = xq.cuda()
faiss/gpu/test/torch_test_contrib_gpu.py:                            xb_c = xb.cuda()
faiss/gpu/test/torch_test_contrib_gpu.py:                        D, I = faiss.knn_gpu(res, xq_c, xb_c, k, use_raft=use_raft)
faiss/gpu/test/torch_test_contrib_gpu.py:                            D, I = faiss.knn_gpu(res, xq_c[6:8], xb_c, k, use_raft=use_raft)
faiss/gpu/test/torch_test_contrib_gpu.py:    def test_knn_gpu_raft(self):
faiss/gpu/test/torch_test_contrib_gpu.py:        self.test_knn_gpu(use_raft=True)
faiss/gpu/test/torch_test_contrib_gpu.py:    def test_knn_gpu_datatypes(self, use_raft=False):
faiss/gpu/test/torch_test_contrib_gpu.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/torch_test_contrib_gpu.py:        xb_c = xb.cuda().half()
faiss/gpu/test/torch_test_contrib_gpu.py:        xq_c = xq.cuda().half()
faiss/gpu/test/torch_test_contrib_gpu.py:        faiss.knn_gpu(res, xq_c, xb_c, k, D, I, use_raft=use_raft)
faiss/gpu/test/torch_test_contrib_gpu.py:        faiss.knn_gpu(res, xq_c, xb_c, k, D, I, use_raft=use_raft)
faiss/gpu/test/torch_test_contrib_gpu.py:class TestTorchUtilsPairwiseDistanceGpu(unittest.TestCase):
faiss/gpu/test/torch_test_contrib_gpu.py:    def test_pairwise_distance_gpu(self):
faiss/gpu/test/torch_test_contrib_gpu.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/torch_test_contrib_gpu.py:        # for the GPU, we'll use a non-default stream
faiss/gpu/test/torch_test_contrib_gpu.py:        s = torch.cuda.Stream()
faiss/gpu/test/torch_test_contrib_gpu.py:        with torch.cuda.stream(s):
faiss/gpu/test/torch_test_contrib_gpu.py:                    D = faiss.pairwise_distance_gpu(res, xq_c, xb_c)
faiss/gpu/test/torch_test_contrib_gpu.py:            # test torch (cpu, gpu) inputs
faiss/gpu/test/torch_test_contrib_gpu.py:            for is_cuda in True, False:
faiss/gpu/test/torch_test_contrib_gpu.py:                        if is_cuda:
faiss/gpu/test/torch_test_contrib_gpu.py:                            xq_c = xq.cuda()
faiss/gpu/test/torch_test_contrib_gpu.py:                            xb_c = xb.cuda()
faiss/gpu/test/torch_test_contrib_gpu.py:                        D = faiss.pairwise_distance_gpu(res, xq_c, xb_c)
faiss/gpu/test/torch_test_contrib_gpu.py:                            D = faiss.pairwise_distance_gpu(res, xq_c[4:8], xb_c)
faiss/gpu/test/torch_test_contrib_gpu.py:        xt_torch = torch.from_numpy(xt).to("cuda:0")
faiss/gpu/test/torch_test_contrib_gpu.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/torch_test_contrib_gpu.py:        data = clustering.DatasetAssignGPU(res, xt_torch)
faiss/gpu/test/test_gpu_index_ivfflat.py:class TestGpuIndexIvfflat(unittest.TestCase):
faiss/gpu/test/test_gpu_index_ivfflat.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index_ivfflat.py:        config = faiss.GpuIndexIVFFlatConfig()
faiss/gpu/test/test_gpu_index_ivfflat.py:        index2 = faiss.GpuIndexIVFFlat(res, index, config)
faiss/gpu/test/test_gpu_basics.py:        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0,
faiss/gpu/test/test_gpu_basics.py:        index = faiss.GpuIndexFlat(faiss.StandardGpuResources(),
faiss/gpu/test/test_gpu_basics.py:        index = faiss.GpuIndexIVFFlat(
faiss/gpu/test/test_gpu_basics.py:            faiss.StandardGpuResources(),
faiss/gpu/test/test_gpu_basics.py:        index = faiss.GpuIndexIVFPQ(
faiss/gpu/test/test_gpu_basics.py:            faiss.StandardGpuResources(), index_cpu)
faiss/gpu/test/test_gpu_basics.py:        index = faiss.GpuIndexBinaryFlat(faiss.StandardGpuResources(),
faiss/gpu/test/test_gpu_basics.py:        num_gpu = 4
faiss/gpu/test/test_gpu_basics.py:        for _i in range(num_gpu):
faiss/gpu/test/test_gpu_basics.py:            config = faiss.GpuIndexFlatConfig()
faiss/gpu/test/test_gpu_basics.py:            config.device = 0   # simulate on a single GPU
faiss/gpu/test/test_gpu_basics.py:            sub_index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), dim, config)
faiss/gpu/test/test_gpu_basics.py:class TestGPUKmeans(unittest.TestCase):
faiss/gpu/test/test_gpu_basics.py:        km2 = faiss.Kmeans(d, k, gpu=True)
faiss/gpu/test/test_gpu_basics.py:        kmeans = faiss.Kmeans(d, k, gpu=True)
faiss/gpu/test/test_gpu_basics.py:        kmeans2 = faiss.Kmeans(d, k, progressive_dim_steps=5, gpu=True)
faiss/gpu/test/test_gpu_basics.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_basics.py:        index = faiss.GpuIndexFlat(res, index_ref)
faiss/gpu/test/test_gpu_basics.py:        index = faiss.GpuIndexFlat(res, d, metric)
faiss/gpu/test/test_gpu_basics.py:class TestGpuRef(unittest.TestCase):
faiss/gpu/test/test_gpu_basics.py:    def test_gpu_ref(self):
faiss/gpu/test/test_gpu_basics.py:        def create_gpu(dim):
faiss/gpu/test/test_gpu_basics.py:            gpu_quantizer = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(dim))
faiss/gpu/test/test_gpu_basics.py:            index.clustering_index = gpu_quantizer
faiss/gpu/test/test_gpu_basics.py:            index.dont_dealloc_me = gpu_quantizer
faiss/gpu/test/test_gpu_basics.py:        index = create_gpu(dim)
faiss/gpu/test/test_gpu_basics.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_basics.py:        gpu_id = random.randrange(0, faiss.get_num_gpus())
faiss/gpu/test/test_gpu_basics.py:        params = faiss.GpuDistanceParams()
faiss/gpu/test/test_gpu_basics.py:        params.device = gpu_id
faiss/gpu/test/test_gpu_basics.py:        faiss.knn_gpu(
faiss/gpu/test/test_gpu_basics.py:            res, qs, xs, k, out_d, out_i, device=gpu_id,
faiss/gpu/test/test_gpu_basics.py:        params.device = random.randrange(0, faiss.get_num_gpus())
faiss/gpu/test/test_gpu_basics.py:        params.device = random.randrange(0, faiss.get_num_gpus())
faiss/gpu/test/test_gpu_basics.py:            res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_basics.py:            params = faiss.GpuDistanceParams()
faiss/gpu/test/test_gpu_basics.py:            params.device = random.randrange(0, faiss.get_num_gpus())
faiss/gpu/test/test_gpu_basics.py:            params.device = random.randrange(0, faiss.get_num_gpus())
faiss/gpu/test/test_gpu_basics.py:    def test_with_gpu(self):
faiss/gpu/test/test_gpu_basics.py:        """ check that we get the same results with a GPU quantizer and a CPU quantizer """
faiss/gpu/test/test_gpu_basics.py:        fac = faiss.GpuProgressiveDimIndexFactory(1)
faiss/gpu/test/test_gpu_basics.py:class TestGpuFlags(unittest.TestCase):
faiss/gpu/test/test_gpu_basics.py:    def test_gpu_flag(self):
faiss/gpu/test/test_gpu_basics.py:        assert "GPU" in faiss.get_compile_options().split()
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:        numAdd = 2 * faiss::gpu::randVal(2000, 5000);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:        dim = faiss::gpu::randVal(64, 200);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:        nprobe = faiss::gpu::randVal(std::min(10, numCentroids), numCentroids);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:        numQuery = faiss::gpu::randVal(32, 100);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:        // differences between GPU and CPU, to stay within our error bounds,
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:        k = std::min(faiss::gpu::randVal(10, 30), numAdd / 40);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:        indicesOpt = faiss::gpu::randSelect(
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:                {faiss::gpu::INDICES_CPU,
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:                 faiss::gpu::INDICES_32_BIT,
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:                 faiss::gpu::INDICES_64_BIT});
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:        device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    faiss::gpu::IndicesOptions indicesOpt;
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    using namespace faiss::gpu;
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    auto config = GpuIndexIVFScalarQuantizerConfig();
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    GpuIndexIVFScalarQuantizer gpuIndex(
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    gpuIndex.train(opt.numTrain, trainVecs.data());
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    gpuIndex.add(opt.numAdd, addVecs.data());
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    gpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    gpuIndex.copyTo(&cpuIndex);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    EXPECT_EQ(gpuIndex.ntotal, opt.numAdd);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    EXPECT_EQ(cpuIndex.d, gpuIndex.d);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    EXPECT_EQ(cpuIndex.quantizer->d, gpuIndex.quantizer->d);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    EXPECT_EQ(cpuIndex.nlist, gpuIndex.getNumLists());
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    EXPECT_EQ(cpuIndex.nprobe, gpuIndex.nprobe);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    testIVFEquality(cpuIndex, gpuIndex);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:            gpuIndex,
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:TEST(TestGpuIndexIVFScalarQuantizer, CopyTo_fp16) {
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:TEST(TestGpuIndexIVFScalarQuantizer, CopyTo_8bit) {
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:TEST(TestGpuIndexIVFScalarQuantizer, CopyTo_8bit_uniform) {
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:TEST(TestGpuIndexIVFScalarQuantizer, CopyTo_6bit) {
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:TEST(TestGpuIndexIVFScalarQuantizer, CopyTo_4bit) {
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:TEST(TestGpuIndexIVFScalarQuantizer, CopyTo_4bit_uniform) {
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    using namespace faiss::gpu;
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    auto config = GpuIndexIVFScalarQuantizerConfig();
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    GpuIndexIVFScalarQuantizer gpuIndex(
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    gpuIndex.nprobe = 1;
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    gpuIndex.copyFrom(&cpuIndex);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    EXPECT_EQ(gpuIndex.ntotal, opt.numAdd);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    EXPECT_EQ(cpuIndex.d, gpuIndex.d);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    EXPECT_EQ(cpuIndex.nlist, gpuIndex.getNumLists());
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    EXPECT_EQ(cpuIndex.nprobe, gpuIndex.nprobe);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    testIVFEquality(cpuIndex, gpuIndex);
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:            gpuIndex,
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:TEST(TestGpuIndexIVFScalarQuantizer, CopyFrom_fp16) {
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:TEST(TestGpuIndexIVFScalarQuantizer, CopyFrom_8bit) {
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:TEST(TestGpuIndexIVFScalarQuantizer, CopyFrom_8bit_uniform) {
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:TEST(TestGpuIndexIVFScalarQuantizer, CopyFrom_6bit) {
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:TEST(TestGpuIndexIVFScalarQuantizer, CopyFrom_4bit) {
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:TEST(TestGpuIndexIVFScalarQuantizer, CopyFrom_4bit_uniform) {
faiss/gpu/test/TestGpuIndexIVFScalarQuantizer.cpp:    faiss::gpu::setTestSeed(100);
faiss/gpu/test/test_raft.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_raft.py:        Dnew, Inew = faiss.knn_gpu(
faiss/gpu/test/test_raft.py:        Dnew, Inew = faiss.knn_gpu(
faiss/gpu/test/test_raft.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_raft.py:        co = faiss.GpuClonerOptions()
faiss/gpu/test/test_raft.py:        index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
faiss/gpu/test/test_raft.py:        Dnew, Inew = index_gpu.search(ds.get_queries(), 13)
faiss/gpu/test/test_raft.py:        index_gpu.add(xb[2000:])
faiss/gpu/test/test_raft.py:        Dnew, Inew = index_gpu.search(ds.get_queries(), 13)
faiss/gpu/test/test_raft.py:        index2 = faiss.index_gpu_to_cpu(index_gpu)
faiss/gpu/test/TestGpuDistance.cu: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/test/TestGpuDistance.cu:#include <faiss/gpu/GpuDistance.h>
faiss/gpu/test/TestGpuDistance.cu:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/test/TestGpuDistance.cu:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/test/TestGpuDistance.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/test/TestGpuDistance.cu:#include <faiss/gpu/utils/CopyUtils.cuh>
faiss/gpu/test/TestGpuDistance.cu:#include <faiss/gpu/utils/Transpose.cuh>
faiss/gpu/test/TestGpuDistance.cu:        faiss::gpu::GpuDistanceParams& args,
faiss/gpu/test/TestGpuDistance.cu:        faiss::gpu::GpuResourcesProvider* res,
faiss/gpu/test/TestGpuDistance.cu:        std::vector<float>& gpuDistance,
faiss/gpu/test/TestGpuDistance.cu:        std::vector<faiss::idx_t>& gpuIndices,
faiss/gpu/test/TestGpuDistance.cu:    using namespace faiss::gpu;
faiss/gpu/test/TestGpuDistance.cu:            gpuDistance.data(),
faiss/gpu/test/TestGpuDistance.cu:            gpuIndices.data(),
faiss/gpu/test/TestGpuDistance.cu:    using namespace faiss::gpu;
faiss/gpu/test/TestGpuDistance.cu:    StandardGpuResources res;
faiss/gpu/test/TestGpuDistance.cu:    // Copy input data to GPU, and pre-transpose both vectors and queries for
faiss/gpu/test/TestGpuDistance.cu:    auto gpuVecs = toDeviceNonTemporary<float, 2>(
faiss/gpu/test/TestGpuDistance.cu:    auto gpuQueries = toDeviceNonTemporary<float, 2>(
faiss/gpu/test/TestGpuDistance.cu:    runTransposeAny(gpuVecs, 0, 1, vecsT, stream);
faiss/gpu/test/TestGpuDistance.cu:    runTransposeAny(gpuQueries, 0, 1, queriesT, stream);
faiss/gpu/test/TestGpuDistance.cu:    std::vector<float> gpuDistance(numQuery * k, 0);
faiss/gpu/test/TestGpuDistance.cu:    std::vector<faiss::idx_t> gpuIndices(numQuery * k, -1);
faiss/gpu/test/TestGpuDistance.cu:    GpuDistanceParams args;
faiss/gpu/test/TestGpuDistance.cu:    args.vectors = colMajorVecs ? vecsT.data() : gpuVecs.data();
faiss/gpu/test/TestGpuDistance.cu:    args.queries = colMajorQueries ? queriesT.data() : gpuQueries.data();
faiss/gpu/test/TestGpuDistance.cu:    args.outDistances = gpuDistance.data();
faiss/gpu/test/TestGpuDistance.cu:    args.outIndices = gpuIndices.data();
faiss/gpu/test/TestGpuDistance.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuDistance.cu:            gpuDistance,
faiss/gpu/test/TestGpuDistance.cu:            gpuIndices,
faiss/gpu/test/TestGpuDistance.cu:TEST(TestGpuDistance, Transposition_RR) {
faiss/gpu/test/TestGpuDistance.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuDistance.cu:TEST(TestRaftGpuDistance, Transposition_RR) {
faiss/gpu/test/TestGpuDistance.cu:TEST(TestGpuDistance, Transposition_RC) {
faiss/gpu/test/TestGpuDistance.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuDistance.cu:TEST(TestRaftGpuDistance, Transposition_RC) {
faiss/gpu/test/TestGpuDistance.cu:TEST(TestGpuDistance, Transposition_CR) {
faiss/gpu/test/TestGpuDistance.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuDistance.cu:TEST(TestRaftGpuDistance, Transposition_CR) {
faiss/gpu/test/TestGpuDistance.cu:TEST(TestGpuDistance, Transposition_CC) {
faiss/gpu/test/TestGpuDistance.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuDistance.cu:TEST(TestRaftGpuDistance, Transposition_CC) {
faiss/gpu/test/TestGpuDistance.cu:TEST(TestGpuDistance, L1) {
faiss/gpu/test/TestGpuDistance.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuDistance.cu:TEST(TestRaftGpuDistance, L1) {
faiss/gpu/test/TestGpuDistance.cu:TEST(TestGpuDistance, L1_RC) {
faiss/gpu/test/TestGpuDistance.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuDistance.cu:TEST(TestRaftGpuDistance, L1_RC) {
faiss/gpu/test/TestGpuDistance.cu:TEST(TestGpuDistance, L1_CR) {
faiss/gpu/test/TestGpuDistance.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuDistance.cu:TEST(TestRaftGpuDistance, L1_CR) {
faiss/gpu/test/TestGpuDistance.cu:TEST(TestGpuDistance, L1_CC) {
faiss/gpu/test/TestGpuDistance.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuDistance.cu:TEST(TestRaftGpuDistance, L1_CC) {
faiss/gpu/test/TestGpuDistance.cu:TEST(TestGpuDistance, Linf) {
faiss/gpu/test/TestGpuDistance.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuDistance.cu:TEST(TestRaftGpuDistance, Linf) {
faiss/gpu/test/TestGpuDistance.cu:TEST(TestGpuDistance, Lp) {
faiss/gpu/test/TestGpuDistance.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuDistance.cu:TEST(TestRaftGpuDistance, Lp) {
faiss/gpu/test/TestGpuDistance.cu:TEST(TestGpuDistance, Canberra) {
faiss/gpu/test/TestGpuDistance.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuDistance.cu:TEST(TestRaftGpuDistance, Canberra) {
faiss/gpu/test/TestGpuDistance.cu:TEST(TestGpuDistance, BrayCurtis) {
faiss/gpu/test/TestGpuDistance.cu:TEST(TestGpuDistance, JensenShannon) {
faiss/gpu/test/TestGpuDistance.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuDistance.cu:TEST(TestRaftGpuDistance, JensenShannon) {
faiss/gpu/test/TestGpuDistance.cu:TEST(TestGpuDistance, Jaccard) {
faiss/gpu/test/TestGpuDistance.cu:    faiss::gpu::setTestSeed(100);
faiss/gpu/test/test_gpu_index_serialize.py:class TestGpuSerialize(unittest.TestCase):
faiss/gpu/test/test_gpu_index_serialize.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_gpu_index_serialize.py:        # Construct various GPU index types
faiss/gpu/test/test_gpu_index_serialize.py:        indexes.append(faiss.GpuIndexFlatL2(res, d))
faiss/gpu/test/test_gpu_index_serialize.py:        indexes.append(faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2))
faiss/gpu/test/test_gpu_index_serialize.py:        config = faiss.GpuIndexIVFScalarQuantizerConfig()
faiss/gpu/test/test_gpu_index_serialize.py:        indexes.append(faiss.GpuIndexIVFScalarQuantizer(res, d, nlist, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2, True, config))
faiss/gpu/test/test_gpu_index_serialize.py:        indexes.append(faiss.GpuIndexIVFPQ(res, d, nlist, 4, 8, faiss.METRIC_L2))
faiss/gpu/test/test_gpu_index_serialize.py:            ser = faiss.serialize_index(faiss.index_gpu_to_cpu(index))
faiss/gpu/test/test_gpu_index_serialize.py:            gpu_cloner_options = faiss.GpuClonerOptions()
faiss/gpu/test/test_gpu_index_serialize.py:            if isinstance(index, faiss.GpuIndexIVFScalarQuantizer):
faiss/gpu/test/test_gpu_index_serialize.py:                gpu_cloner_options.use_raft = False
faiss/gpu/test/test_gpu_index_serialize.py:            gpu_index_restore = faiss.index_cpu_to_gpu(res, 0, cpu_index, gpu_cloner_options)
faiss/gpu/test/test_gpu_index_serialize.py:            restore_d, restore_i = gpu_index_restore.search(query, k)
faiss/gpu/test/test_gpu_index_serialize.py:            gpu_index_restore.add(query)
faiss/gpu/test/CMakeLists.txt:# Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/test/CMakeLists.txt:add_library(faiss_gpu_test_helper TestUtils.cpp)
faiss/gpu/test/CMakeLists.txt:if(FAISS_ENABLE_ROCM)
faiss/gpu/test/CMakeLists.txt:  target_link_libraries(faiss_gpu_test_helper PUBLIC faiss gtest hip::host)
faiss/gpu/test/CMakeLists.txt:  find_package(CUDAToolkit REQUIRED)
faiss/gpu/test/CMakeLists.txt:  target_link_libraries(faiss_gpu_test_helper PUBLIC
faiss/gpu/test/CMakeLists.txt:    faiss gtest CUDA::cudart
faiss/gpu/test/CMakeLists.txt:macro(faiss_gpu_test file)
faiss/gpu/test/CMakeLists.txt:  target_link_libraries(${test_name} PRIVATE faiss_gpu_test_helper)
faiss/gpu/test/CMakeLists.txt:faiss_gpu_test(TestCodePacking.cpp)
faiss/gpu/test/CMakeLists.txt:faiss_gpu_test(TestGpuIndexFlat.cpp)
faiss/gpu/test/CMakeLists.txt:faiss_gpu_test(TestGpuIndexIVFFlat.cpp)
faiss/gpu/test/CMakeLists.txt:faiss_gpu_test(TestGpuIndexBinaryFlat.cpp)
faiss/gpu/test/CMakeLists.txt:faiss_gpu_test(TestGpuMemoryException.cpp)
faiss/gpu/test/CMakeLists.txt:faiss_gpu_test(TestGpuIndexIVFPQ.cpp)
faiss/gpu/test/CMakeLists.txt:faiss_gpu_test(TestGpuIndexIVFScalarQuantizer.cpp)
faiss/gpu/test/CMakeLists.txt:faiss_gpu_test(TestGpuDistance.${GPU_EXT_PREFIX})
faiss/gpu/test/CMakeLists.txt:faiss_gpu_test(TestGpuSelect.${GPU_EXT_PREFIX})
faiss/gpu/test/CMakeLists.txt:  faiss_gpu_test(TestGpuIndexCagra.cu)
faiss/gpu/test/CMakeLists.txt:add_executable(demo_ivfpq_indexing_gpu EXCLUDE_FROM_ALL
faiss/gpu/test/CMakeLists.txt:  demo_ivfpq_indexing_gpu.cpp)
faiss/gpu/test/CMakeLists.txt:if (FAISS_ENABLE_ROCM)
faiss/gpu/test/CMakeLists.txt:  target_link_libraries(demo_ivfpq_indexing_gpu
faiss/gpu/test/CMakeLists.txt:  target_link_libraries(demo_ivfpq_indexing_gpu
faiss/gpu/test/CMakeLists.txt:    PRIVATE faiss gtest_main CUDA::cudart)
faiss/gpu/test/TestCodePacking.cpp:#include <faiss/gpu/impl/InterleavedCodes.h>
faiss/gpu/test/TestCodePacking.cpp:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/test/TestCodePacking.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/test/TestCodePacking.cpp:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/test/TestCodePacking.cpp:    using namespace faiss::gpu;
faiss/gpu/test/TestCodePacking.cpp:    using namespace faiss::gpu;
faiss/gpu/test/TestCodePacking.cpp:    using namespace faiss::gpu;
faiss/gpu/test/TestCodePacking.cpp:    using namespace faiss::gpu;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:#include <faiss/gpu/GpuIndexIVFPQ.h>
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        codes = codeSizes[faiss::gpu::randVal(0, codeSizes.size() - 1)];
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        dim = codes * dimSizes[faiss::gpu::randVal(0, dimSizes.size() - 1)];
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        codes = faiss::gpu::randVal(0, 96);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        dim = codes * dimSizes[faiss::gpu::randVal(0, dimSizes.size() - 1)];
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        numAdd = faiss::gpu::randVal(2000, 5000);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        // TODO: Change back to `faiss::gpu::randVal(3, 7)` when we
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        nprobe = std::min(faiss::gpu::randVal(40, 1000), numCentroids);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        numQuery = faiss::gpu::randVal(4, 8);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        // differences between GPU and CPU, to stay within our error bounds,
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        k = std::min(faiss::gpu::randVal(5, 20), numAdd / 40);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        usePrecomputed = faiss::gpu::randBool();
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        indicesOpt = faiss::gpu::randSelect(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:                {faiss::gpu::INDICES_CPU,
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:                 faiss::gpu::INDICES_32_BIT,
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:                 faiss::gpu::INDICES_64_BIT});
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:            useFloat16 = faiss::gpu::randBool();
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::IndicesOptions indicesOpt;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::GpuIndexIVFPQConfig config;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::GpuIndexIVFPQ gpuIndex(&res, &cpuIndex, config);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    gpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:            gpuIndex,
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, Query_L2) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, Query_IP) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, LargeBatch) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:                faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        faiss::gpu::GpuIndexIVFPQConfig config;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        faiss::gpu::GpuIndexIVFPQ gpuIndex(&res, &cpuIndex, config);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        gpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:                gpuIndex,
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:                faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        faiss::gpu::GpuIndexIVFPQConfig config;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        faiss::gpu::GpuIndexIVFPQ gpuIndex(&res, &cpuIndex, config);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        gpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:                gpuIndex,
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, Query_L2_MMCodeDistance) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, Query_IP_MMCodeDistance) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, Float16Coarse) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::GpuIndexIVFPQConfig config;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::GpuIndexIVFPQ gpuIndex(&res, &cpuIndex, config);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    gpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    gpuIndex.add(opt.numAdd, addVecs.data());
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:            gpuIndex,
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::GpuIndexIVFPQConfig config;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::GpuIndexIVFPQ gpuIndex(&res, &cpuIndex, config);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    gpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    gpuIndex.add(opt.numAdd, addVecs.data());
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:            gpuIndex,
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, Add_L2) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, Add_IP) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:                faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        faiss::gpu::GpuIndexIVFPQConfig config;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        faiss::gpu::GpuIndexIVFPQ gpuIndex(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        gpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        gpuIndex.train(opt.numTrain, trainVecs.data());
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        gpuIndex.add(opt.numAdd, addVecs.data());
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        gpuIndex.copyTo(&cpuIndex);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        EXPECT_EQ(gpuIndex.ntotal, opt.numAdd);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        EXPECT_EQ(cpuIndex.d, gpuIndex.d);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        EXPECT_EQ(cpuIndex.nlist, gpuIndex.getNumLists());
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        EXPECT_EQ(cpuIndex.nprobe, gpuIndex.nprobe);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        EXPECT_EQ(cpuIndex.pq.M, gpuIndex.getNumSubQuantizers());
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        EXPECT_EQ(gpuIndex.getNumSubQuantizers(), opt.codes);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        EXPECT_EQ(cpuIndex.pq.nbits, gpuIndex.getBitsPerCode());
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        EXPECT_EQ(gpuIndex.getBitsPerCode(), opt.bitsPerCode);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        testIVFEquality(cpuIndex, gpuIndex);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:                gpuIndex,
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, CopyTo) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::GpuIndexIVFPQConfig config;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::GpuIndexIVFPQ gpuIndex(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    gpuIndex.nprobe = 1;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    gpuIndex.copyFrom(&cpuIndex);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    EXPECT_EQ(gpuIndex.ntotal, opt.numAdd);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    EXPECT_EQ(cpuIndex.d, gpuIndex.d);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    EXPECT_EQ(cpuIndex.nlist, gpuIndex.getNumLists());
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    EXPECT_EQ(cpuIndex.nprobe, gpuIndex.nprobe);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    EXPECT_EQ(cpuIndex.pq.M, gpuIndex.getNumSubQuantizers());
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    EXPECT_EQ(gpuIndex.getNumSubQuantizers(), opt.codes);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    EXPECT_EQ(cpuIndex.pq.nbits, gpuIndex.getBitsPerCode());
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    EXPECT_EQ(gpuIndex.getBitsPerCode(), opt.bitsPerCode);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    testIVFEquality(cpuIndex, gpuIndex);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:            gpuIndex,
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, CopyFrom) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::GpuIndexIVFPQConfig config;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::GpuIndexIVFPQ gpuIndex(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    gpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    gpuIndex.train(opt.numTrain, trainVecs.data());
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    gpuIndex.add(opt.numAdd, addVecs.data());
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    gpuIndex.search(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, QueryNaN) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::GpuIndexIVFPQConfig config;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::GpuIndexIVFPQ gpuIndex(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    gpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    gpuIndex.train(opt.numTrain, trainVecs.data());
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    EXPECT_EQ(gpuIndex.ntotal, 0);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    gpuIndex.add(numNans, nans.data());
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    std::vector<float> queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    gpuIndex.search(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, AddNaN) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, Query_L2_Raft) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        opt.bitsPerCode = faiss::gpu::randVal(4, 8);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, Query_IP_Raft) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        opt.bitsPerCode = faiss::gpu::randVal(4, 8);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, LargeBatch_Raft) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, CopyFrom_Raft) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    opt.bitsPerCode = faiss::gpu::randVal(4, 8);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, Add_L2_Raft) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        opt.bitsPerCode = faiss::gpu::randVal(4, 8);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, Add_IP_Raft) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        opt.bitsPerCode = faiss::gpu::randVal(4, 8);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:        opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, QueryNaN_Raft) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    opt.bitsPerCode = faiss::gpu::randVal(4, 8);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, AddNaN_Raft) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    opt.bitsPerCode = faiss::gpu::randVal(4, 8);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, CopyTo_Raft) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    opt.bitsPerCode = faiss::gpu::randVal(4, 8);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:TEST(TestGpuIndexIVFPQ, UnifiedMemory) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    if (!faiss::gpu::getFullUnifiedMemSupport(device)) {
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    std::vector<float> trainVecs = faiss::gpu::randVecs(numTrain, dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    std::vector<float> addVecs = faiss::gpu::randVecs(numAdd, dim);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::GpuIndexIVFPQConfig config;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    config.memorySpace = faiss::gpu::MemorySpace::Unified;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::GpuIndexIVFPQ gpuIndex(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    gpuIndex.copyFrom(&cpuIndex);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    gpuIndex.nprobe = nprobe;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:            gpuIndex,
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    config.indicesOptions = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::GpuIndexIVFPQ raftGpuIndex(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    raftGpuIndex.copyFrom(&cpuIndex);
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    raftGpuIndex.nprobe = nprobe;
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:            raftGpuIndex,
faiss/gpu/test/TestGpuIndexIVFPQ.cpp:    faiss::gpu::setTestSeed(100);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:#include <faiss/gpu/GpuIndexBinaryFlat.h>
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:#include <faiss/gpu/impl/IndexUtils.h>
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:        const std::vector<int>& gpuDist,
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:        const std::vector<faiss::idx_t>& gpuLabels,
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:        std::set<faiss::idx_t> gpuLabelSet;
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:                EXPECT_EQ(cpuLabelSet, gpuLabelSet);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:                gpuLabelSet.clear();
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:            gpuLabelSet.insert(gpuLabels[idx]);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:            EXPECT_EQ(cpuDist[idx], gpuDist[idx]);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:void testGpuIndexBinaryFlat(int kOverride = -1) {
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    faiss::gpu::GpuIndexBinaryFlatConfig config;
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    config.device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    int dims = faiss::gpu::randVal(1, 20) * DimMultiple;
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    faiss::gpu::GpuIndexBinaryFlat gpuIndex(&res, dims, config);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:            : faiss::gpu::randVal(1, faiss::gpu::getMaxKSelection());
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    int numVecs = faiss::gpu::randVal(k + 1, 20000);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    int numQuery = faiss::gpu::randVal(1, 1000);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    auto data = faiss::gpu::randBinaryVecs(numVecs, dims);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    gpuIndex.add(numVecs, data.data());
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    auto query = faiss::gpu::randBinaryVecs(numQuery, dims);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    std::vector<int> gpuDist(numQuery * k);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    std::vector<faiss::idx_t> gpuLabels(numQuery * k);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    gpuIndex.search(
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:            numQuery, query.data(), k, gpuDist.data(), gpuLabels.data());
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    compareBinaryDist(cpuDist, cpuLabels, gpuDist, gpuLabels, numQuery, k);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:TEST(TestGpuIndexBinaryFlat, Test8) {
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:        testGpuIndexBinaryFlat<8>();
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:TEST(TestGpuIndexBinaryFlat, Test32) {
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:        testGpuIndexBinaryFlat<32>();
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:TEST(TestGpuIndexBinaryFlat, LargeIndex) {
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    if (faiss::gpu::getFreeMemory(device) < kMem) {
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:        std::cerr << "TestGpuIndexFlat.LargeIndex: skipping due "
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    faiss::gpu::GpuIndexBinaryFlatConfig config;
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    faiss::gpu::GpuIndexBinaryFlat gpuIndex(&res, dims, config);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    auto xb = faiss::gpu::randBinaryVecs(nb, dims);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    auto xq = faiss::gpu::randBinaryVecs(nq, dims);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    gpuIndex.add(nb, xb.data());
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    std::vector<int> gpuDist(nq * k);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    std::vector<faiss::idx_t> gpuLabels(nq * k);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    gpuIndex.search(nq, xq.data(), k, gpuDist.data(), gpuLabels.data());
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    compareBinaryDist(cpuDist, cpuLabels, gpuDist, gpuLabels, nq, k);
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:TEST(TestGpuIndexBinaryFlat, Reconstruct) {
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    std::unique_ptr<faiss::gpu::GpuIndexBinaryFlat> index2(
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:            new faiss::gpu::GpuIndexBinaryFlat(&res, index.get()));
faiss/gpu/test/TestGpuIndexBinaryFlat.cpp:    faiss::gpu::setTestSeed(100);
faiss/gpu/test/test_contrib_gpu.py:    range_ground_truth, range_search_gpu, \
faiss/gpu/test/test_contrib_gpu.py:    def test_max_results_binary(self, ngpu=1):
faiss/gpu/test/test_contrib_gpu.py:            ngpu=1
faiss/gpu/test/test_contrib_gpu.py:    @unittest.skipIf(faiss.get_num_gpus() < 2, "multiple GPU only test")
faiss/gpu/test/test_contrib_gpu.py:    def test_max_results_binary_multigpu(self):
faiss/gpu/test/test_contrib_gpu.py:        self.test_max_results_binary(ngpu=2)
faiss/gpu/test/test_contrib_gpu.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_contrib_gpu.py:            return faiss.pairwise_distance_gpu(
faiss/gpu/test/test_contrib_gpu.py:            return faiss.knn_gpu(res, xq, xb, k, metric=faiss.METRIC_L2)
faiss/gpu/test/test_contrib_gpu.py:class TestBigBatchSearchMultiGPU(unittest.TestCase):
faiss/gpu/test/test_contrib_gpu.py:    @unittest.skipIf(faiss.get_num_gpus() < 2, "multiple GPU only test")
faiss/gpu/test/test_contrib_gpu.py:        ngpu = faiss.get_num_gpus()
faiss/gpu/test/test_contrib_gpu.py:        res = [faiss.StandardGpuResources() for _ in range(ngpu)]
faiss/gpu/test/test_contrib_gpu.py:            return faiss.knn_gpu(
faiss/gpu/test/test_contrib_gpu.py:            computation_threads=ngpu
faiss/gpu/test/test_contrib_gpu.py:class TestRangeSearchGpu(unittest.TestCase):
faiss/gpu/test/test_contrib_gpu.py:        index_gpu = faiss.index_cpu_to_all_gpus(
faiss/gpu/test/test_contrib_gpu.py:        index_gpu.train(ds.get_train())
faiss/gpu/test/test_contrib_gpu.py:        index_gpu.add(ds.get_database())
faiss/gpu/test/test_contrib_gpu.py:        D, _ = index_gpu.search(ds.get_queries(), k)
faiss/gpu/test/test_contrib_gpu.py:        index_cpu = faiss.index_gpu_to_cpu(index_gpu)
faiss/gpu/test/test_contrib_gpu.py:        # make sure some entries were computed by CPU and some by GPU
faiss/gpu/test/test_contrib_gpu.py:        # mixed GPU / CPU run
faiss/gpu/test/test_contrib_gpu.py:        Lnew, Dnew, Inew = range_search_gpu(
faiss/gpu/test/test_contrib_gpu.py:            ds.get_queries(), threshold, index_gpu, index_cpu, gpu_k=4)
faiss/gpu/test/test_contrib_gpu.py:        Lnew2, Dnew2, Inew2 = range_search_gpu(
faiss/gpu/test/test_contrib_gpu.py:            ds.get_queries(), threshold, index_gpu, None, gpu_k=4)
faiss/gpu/test/TestUtils.h:namespace gpu {
faiss/gpu/test/TestUtils.h:/// Compare IVF lists between a CPU and GPU index
faiss/gpu/test/TestUtils.h:void testIVFEquality(A& cpuIndex, B& gpuIndex) {
faiss/gpu/test/TestUtils.h:    EXPECT_EQ(cpuIndex.nlist, gpuIndex.nlist);
faiss/gpu/test/TestUtils.h:        EXPECT_EQ(cpuLists->list_size(i), gpuIndex.getListLength(i));
faiss/gpu/test/TestUtils.h:        auto gpuCodes = gpuIndex.getListVectorData(i, false);
faiss/gpu/test/TestUtils.h:        EXPECT_EQ(cpuCodes, gpuCodes);
faiss/gpu/test/TestUtils.h:        EXPECT_EQ(cpuIndices, gpuIndex.getListIndices(i));
faiss/gpu/test/TestUtils.h:} // namespace gpu
faiss/gpu/test/test_cagra.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_cagra.py:        index = faiss.GpuIndexCagra(res, d, metric)
faiss/gpu/test/test_cagra.py:        res = faiss.StandardGpuResources()
faiss/gpu/test/test_cagra.py:        index = faiss.GpuIndexCagra(res, d, metric)
faiss/gpu/test/test_cagra.py:        cpu_index = faiss.index_gpu_to_cpu(index)
faiss/gpu/test/test_cagra.py:        gpu_index = faiss.index_cpu_to_gpu(res, 0, deserialized_index)
faiss/gpu/test/test_cagra.py:        Dnew2, Inew2 = gpu_index.search(ds.get_queries(), k)
faiss/gpu/test/TestGpuIndexFlat.cpp:#include <faiss/gpu/GpuIndexFlat.h>
faiss/gpu/test/TestGpuIndexFlat.cpp:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/test/TestGpuIndexFlat.cpp:#include <faiss/gpu/impl/IndexUtils.h>
faiss/gpu/test/TestGpuIndexFlat.cpp:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/test/TestGpuIndexFlat.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/test/TestGpuIndexFlat.cpp:                                          : faiss::gpu::randVal(1000, 5000);
faiss/gpu/test/TestGpuIndexFlat.cpp:                                  : faiss::gpu::randVal(50, 800);
faiss/gpu/test/TestGpuIndexFlat.cpp:                                              : faiss::gpu::randVal(1, 512);
faiss/gpu/test/TestGpuIndexFlat.cpp:            ? std::min(faiss::gpu::randVal(1, 50), numVecs)
faiss/gpu/test/TestGpuIndexFlat.cpp:                      faiss::gpu::randVal(1, faiss::gpu::getMaxKSelection()),
faiss/gpu/test/TestGpuIndexFlat.cpp:    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::GpuIndexFlatConfig config;
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::GpuIndexFlat gpuIndex(&res, dim, opt.metric, config);
faiss/gpu/test/TestGpuIndexFlat.cpp:    gpuIndex.metric_arg = opt.metricArg;
faiss/gpu/test/TestGpuIndexFlat.cpp:    std::vector<float> vecs = faiss::gpu::randVecs(numVecs, dim);
faiss/gpu/test/TestGpuIndexFlat.cpp:    gpuIndex.add(numVecs, vecs.data());
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexFlat.cpp:            gpuIndex,
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, IP_Float32) {
faiss/gpu/test/TestGpuIndexFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, L1_Float32) {
faiss/gpu/test/TestGpuIndexFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, Lp_Float32) {
faiss/gpu/test/TestGpuIndexFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, L2_Float32) {
faiss/gpu/test/TestGpuIndexFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, L2_k_2048) {
faiss/gpu/test/TestGpuIndexFlat.cpp:    if (faiss::gpu::getMaxKSelection() >= 2048) {
faiss/gpu/test/TestGpuIndexFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, L2_Float32_K1) {
faiss/gpu/test/TestGpuIndexFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, IP_Float16) {
faiss/gpu/test/TestGpuIndexFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, L2_Float16) {
faiss/gpu/test/TestGpuIndexFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, L2_Float16_K1) {
faiss/gpu/test/TestGpuIndexFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, L2_Tiling) {
faiss/gpu/test/TestGpuIndexFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, QueryEmpty) {
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::GpuIndexFlatConfig config;
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::GpuIndexFlatL2 gpuIndex(&res, dim, config);
faiss/gpu/test/TestGpuIndexFlat.cpp:    gpuIndex.search(numQuery, queries.data(), k, dist.data(), ind.data());
faiss/gpu/test/TestGpuIndexFlat.cpp:    int numVecs = faiss::gpu::randVal(100, 200);
faiss/gpu/test/TestGpuIndexFlat.cpp:    int dim = faiss::gpu::randVal(1, 1000);
faiss/gpu/test/TestGpuIndexFlat.cpp:    std::vector<float> vecs = faiss::gpu::randVecs(numVecs, dim);
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexFlat.cpp:    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuIndexFlat.cpp:        faiss::gpu::GpuIndexFlatConfig config;
faiss/gpu/test/TestGpuIndexFlat.cpp:        faiss::gpu::GpuIndexFlatL2 gpuIndex(&res, 2000, config);
faiss/gpu/test/TestGpuIndexFlat.cpp:        gpuIndex.copyFrom(&cpuIndex);
faiss/gpu/test/TestGpuIndexFlat.cpp:        EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
faiss/gpu/test/TestGpuIndexFlat.cpp:        EXPECT_EQ(gpuIndex.ntotal, numVecs);
faiss/gpu/test/TestGpuIndexFlat.cpp:        EXPECT_EQ(cpuIndex.d, gpuIndex.d);
faiss/gpu/test/TestGpuIndexFlat.cpp:        std::vector<float> gpuVals(numVecs * dim);
faiss/gpu/test/TestGpuIndexFlat.cpp:        gpuIndex.reconstruct_n(0, gpuIndex.ntotal, gpuVals.data());
faiss/gpu/test/TestGpuIndexFlat.cpp:        cpuIndex.reconstruct_n(0, gpuIndex.ntotal, cpuVals.data());
faiss/gpu/test/TestGpuIndexFlat.cpp:        // The CPU is the source of (float32) truth here, while the GPU index
faiss/gpu/test/TestGpuIndexFlat.cpp:            EXPECT_EQ(gpuVals, faiss::gpu::roundToHalf(cpuVals));
faiss/gpu/test/TestGpuIndexFlat.cpp:            EXPECT_EQ(gpuVals, cpuVals);
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, CopyFrom) {
faiss/gpu/test/TestGpuIndexFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestRaftGpuIndexFlat, CopyFrom) {
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexFlat.cpp:    int numVecs = faiss::gpu::randVal(100, 200);
faiss/gpu/test/TestGpuIndexFlat.cpp:    int dim = faiss::gpu::randVal(1, 1000);
faiss/gpu/test/TestGpuIndexFlat.cpp:    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuIndexFlat.cpp:    std::vector<float> vecs = faiss::gpu::randVecs(numVecs, dim);
faiss/gpu/test/TestGpuIndexFlat.cpp:        faiss::gpu::GpuIndexFlatConfig config;
faiss/gpu/test/TestGpuIndexFlat.cpp:        faiss::gpu::GpuIndexFlatL2 gpuIndex(&res, dim, config);
faiss/gpu/test/TestGpuIndexFlat.cpp:        gpuIndex.add(numVecs, vecs.data());
faiss/gpu/test/TestGpuIndexFlat.cpp:        gpuIndex.copyTo(&cpuIndex);
faiss/gpu/test/TestGpuIndexFlat.cpp:        EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
faiss/gpu/test/TestGpuIndexFlat.cpp:        EXPECT_EQ(gpuIndex.ntotal, numVecs);
faiss/gpu/test/TestGpuIndexFlat.cpp:        EXPECT_EQ(cpuIndex.d, gpuIndex.d);
faiss/gpu/test/TestGpuIndexFlat.cpp:        std::vector<float> gpuVals(numVecs * dim);
faiss/gpu/test/TestGpuIndexFlat.cpp:        gpuIndex.reconstruct_n(0, gpuIndex.ntotal, gpuVals.data());
faiss/gpu/test/TestGpuIndexFlat.cpp:        cpuIndex.reconstruct_n(0, gpuIndex.ntotal, cpuVals.data());
faiss/gpu/test/TestGpuIndexFlat.cpp:        // The GPU is the source of truth here, so the float32 exact comparison
faiss/gpu/test/TestGpuIndexFlat.cpp:        EXPECT_EQ(gpuVals, cpuVals);
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, CopyTo) {
faiss/gpu/test/TestGpuIndexFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestRaftGpuIndexFlat, CopyTo) {
faiss/gpu/test/TestGpuIndexFlat.cpp:    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuIndexFlat.cpp:    if (!faiss::gpu::getFullUnifiedMemSupport(device)) {
faiss/gpu/test/TestGpuIndexFlat.cpp:    // FIXME: GpuIndexFlat doesn't support > 2^31 (vecs * dims) due to
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::GpuIndexFlatConfig config;
faiss/gpu/test/TestGpuIndexFlat.cpp:    config.memorySpace = faiss::gpu::MemorySpace::Unified;
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::GpuIndexFlatL2 gpuIndexL2(&res, dim, config);
faiss/gpu/test/TestGpuIndexFlat.cpp:    std::vector<float> vecs = faiss::gpu::randVecs(numVecs, dim);
faiss/gpu/test/TestGpuIndexFlat.cpp:    gpuIndexL2.add(numVecs, vecs.data());
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexFlat.cpp:            gpuIndexL2,
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, UnifiedMemory) {
faiss/gpu/test/TestGpuIndexFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestRaftGpuIndexFlat, UnifiedMemory) {
faiss/gpu/test/TestGpuIndexFlat.cpp:    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexFlat.cpp:    if (faiss::gpu::getFreeMemory(device) < kMem) {
faiss/gpu/test/TestGpuIndexFlat.cpp:        std::cout << "TestGpuIndexFlat.LargeIndex: skipping due "
faiss/gpu/test/TestGpuIndexFlat.cpp:    auto xb = faiss::gpu::randVecs(nb, dim);
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::GpuIndexFlatConfig config;
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::GpuIndexFlatL2 gpuIndexL2(&res, dim, config);
faiss/gpu/test/TestGpuIndexFlat.cpp:    gpuIndexL2.add(nb, xb.data());
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexFlat.cpp:            gpuIndexL2,
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, LargeIndex) {
faiss/gpu/test/TestGpuIndexFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestRaftGpuIndexFlat, LargeIndex) {
faiss/gpu/test/TestGpuIndexFlat.cpp:    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::GpuIndexFlatConfig config;
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::GpuIndexFlat gpuIndex(
faiss/gpu/test/TestGpuIndexFlat.cpp:    auto vecs = faiss::gpu::randVecs(numVecs, dim);
faiss/gpu/test/TestGpuIndexFlat.cpp:    gpuIndex.add(numVecs, vecs.data());
faiss/gpu/test/TestGpuIndexFlat.cpp:    auto queryVecs = faiss::gpu::randVecs(indexVecs.size(), dim);
faiss/gpu/test/TestGpuIndexFlat.cpp:    auto residualsGpu = std::vector<float>(indexVecs.size() * dim);
faiss/gpu/test/TestGpuIndexFlat.cpp:    gpuIndex.compute_residual_n(
faiss/gpu/test/TestGpuIndexFlat.cpp:            residualsGpu.data(),
faiss/gpu/test/TestGpuIndexFlat.cpp:    EXPECT_EQ(residualsCpu, residualsGpu);
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, Residual) {
faiss/gpu/test/TestGpuIndexFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestRaftGpuIndexFlat, Residual) {
faiss/gpu/test/TestGpuIndexFlat.cpp:    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexFlat.cpp:    auto vecs = faiss::gpu::randVecs(numVecs, dim);
faiss/gpu/test/TestGpuIndexFlat.cpp:    auto vecs16 = faiss::gpu::roundToHalf(vecs);
faiss/gpu/test/TestGpuIndexFlat.cpp:        faiss::gpu::GpuIndexFlatConfig config;
faiss/gpu/test/TestGpuIndexFlat.cpp:        faiss::gpu::GpuIndexFlat gpuIndex(
faiss/gpu/test/TestGpuIndexFlat.cpp:        gpuIndex.add(numVecs, vecs.data());
faiss/gpu/test/TestGpuIndexFlat.cpp:            gpuIndex.reconstruct(15, reconstructVecs.data());
faiss/gpu/test/TestGpuIndexFlat.cpp:            gpuIndex.reconstruct_n(
faiss/gpu/test/TestGpuIndexFlat.cpp:            gpuIndex.reconstruct_batch(
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, Reconstruct) {
faiss/gpu/test/TestGpuIndexFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestRaftGpuIndexFlat, Reconstruct) {
faiss/gpu/test/TestGpuIndexFlat.cpp:    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexFlat.cpp:    auto xb = faiss::gpu::randVecs(nb, dim);
faiss/gpu/test/TestGpuIndexFlat.cpp:    auto xq = faiss::gpu::randVecs(nq, dim);
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::GpuIndexFlatConfig config;
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::GpuIndexFlatL2 gpuIndex(&res, dim, config);
faiss/gpu/test/TestGpuIndexFlat.cpp:    gpuIndex.add(nb, xb.data());
faiss/gpu/test/TestGpuIndexFlat.cpp:    gpuIndex.search_and_reconstruct(
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::compareLists(
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestGpuIndexFlat, SearchAndReconstruct) {
faiss/gpu/test/TestGpuIndexFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexFlat.cpp:TEST(TestRaftGpuIndexFlat, SearchAndReconstruct) {
faiss/gpu/test/TestGpuIndexFlat.cpp:    faiss::gpu::setTestSeed(100);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#include <faiss/gpu/GpuIndexIVFFlat.h>
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#include <faiss/gpu/GpuIndicesOptions.h>
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        numAdd = 2 * faiss::gpu::randVal(2000, 5000);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        dim = faiss::gpu::randVal(64, 200);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        nprobe = faiss::gpu::randVal(std::min(10, numCentroids), numCentroids);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        numQuery = faiss::gpu::randVal(32, 100);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        // differences between GPU and CPU, to stay within our error bounds,
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        k = std::min(faiss::gpu::randVal(10, 30), numAdd / 40);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        indicesOpt = faiss::gpu::randSelect(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:                {faiss::gpu::INDICES_CPU,
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:                 faiss::gpu::INDICES_32_BIT,
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:                 faiss::gpu::INDICES_64_BIT});
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::IndicesOptions indicesOpt;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:                faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        faiss::gpu::GpuIndexIVFFlatConfig config;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        faiss::gpu::GpuIndexIVFFlat gpuIndex(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        gpuIndex.copyFrom(&cpuIndex);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        gpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:                gpuIndex,
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:                faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        faiss::gpu::GpuIndexIVFFlatConfig config;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:                useRaft ? faiss::gpu::INDICES_64_BIT : opt.indicesOpt;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        faiss::gpu::GpuIndexIVFFlat gpuIndex(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        gpuIndex.copyFrom(&cpuIndex);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        gpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        gpuIndex.add(opt.numAdd, addVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:                gpuIndex,
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlatConfig config;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:            useRaft ? faiss::gpu::INDICES_64_BIT : opt.indicesOpt;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlat gpuIndex(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.train(opt.numTrain, trainVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.add(opt.numAdd, addVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.copyTo(&cpuIndex);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    EXPECT_EQ(gpuIndex.ntotal, opt.numAdd);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    EXPECT_EQ(cpuIndex.d, gpuIndex.d);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    EXPECT_EQ(cpuIndex.quantizer->d, gpuIndex.quantizer->d);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    EXPECT_EQ(cpuIndex.nlist, gpuIndex.getNumLists());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    EXPECT_EQ(cpuIndex.nprobe, gpuIndex.nprobe);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    testIVFEquality(cpuIndex, gpuIndex);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:            gpuIndex,
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlatConfig config;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:            useRaft ? faiss::gpu::INDICES_64_BIT : opt.indicesOpt;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlat gpuIndex(&res, 1, 1, faiss::METRIC_L2, config);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.nprobe = 1;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.copyFrom(&cpuIndex);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    EXPECT_EQ(cpuIndex.ntotal, gpuIndex.ntotal);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    EXPECT_EQ(gpuIndex.ntotal, opt.numAdd);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    EXPECT_EQ(cpuIndex.d, gpuIndex.d);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    EXPECT_EQ(cpuIndex.nlist, gpuIndex.getNumLists());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    EXPECT_EQ(cpuIndex.nprobe, gpuIndex.nprobe);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    testIVFEquality(cpuIndex, gpuIndex);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:            gpuIndex,
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, Float32_32_Add_L2) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, Float32_32_Add_IP) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, Float16_32_Add_L2) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, Float16_32_Add_IP) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, Float32_Query_L2) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, Float32_Query_IP) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, LargeBatch) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, Float16_32_Query_L2) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, Float16_32_Query_IP) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, Float32_Query_L2_64) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, Float32_Query_IP_64) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, Float32_Query_L2_128) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, Float32_Query_IP_128) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    opt.indicesOpt = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, Float32_32_CopyTo) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, Float32_32_CopyFrom) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, Float32_negative) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    auto trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    auto addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlatConfig config;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlat gpuIndex(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.copyFrom(&cpuIndex);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:            gpuIndex,
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    config.indicesOptions = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlat raftGpuIndex(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    raftGpuIndex.copyFrom(&cpuIndex);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    raftGpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:            raftGpuIndex,
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, QueryNaN) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlatConfig config;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    config.flatConfig.useFloat16 = faiss::gpu::randBool();
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlat gpuIndex(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.train(opt.numTrain, trainVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.add(opt.numAdd, addVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.search(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    config.indicesOptions = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlat raftGpuIndex(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    raftGpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    raftGpuIndex.train(opt.numTrain, trainVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    raftGpuIndex.add(opt.numAdd, addVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    raftGpuIndex.search(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, AddNaN) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlatConfig config;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    config.flatConfig.useFloat16 = faiss::gpu::randBool();
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlat gpuIndex(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.train(opt.numTrain, trainVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    EXPECT_EQ(gpuIndex.ntotal, 0);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.add(numNans, nans.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    std::vector<float> queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.search(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    config.indicesOptions = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlat raftGpuIndex(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    raftGpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    raftGpuIndex.train(opt.numTrain, trainVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    EXPECT_EQ(raftGpuIndex.ntotal, 0);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    raftGpuIndex.add(numNans, nans.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    raftGpuIndex.search(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, UnifiedMemory) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    if (!faiss::gpu::getFullUnifiedMemSupport(device)) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    std::vector<float> trainVecs = faiss::gpu::randVecs(numTrain, dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    std::vector<float> addVecs = faiss::gpu::randVecs(numAdd, dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlatConfig config;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    config.memorySpace = faiss::gpu::MemorySpace::Unified;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlat gpuIndex(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.copyFrom(&cpuIndex);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.nprobe = nprobe;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:            gpuIndex,
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    config.indicesOptions = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlat raftGpuIndex(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    raftGpuIndex.copyFrom(&cpuIndex);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    raftGpuIndex.nprobe = nprobe;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:            raftGpuIndex,
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, LongIVFList) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    int device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    if (faiss::gpu::getFreeMemory(device) < kMem) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:        std::cout << "TestGpuIndexIVFFlat.LongIVFList: skipping due "
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    std::vector<float> trainVecs = faiss::gpu::randVecs(numTrain, dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    std::vector<float> addVecs = faiss::gpu::randVecs(numAdd, dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlatConfig config;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlat gpuIndex(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.train(numTrain, trainVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.add(numAdd, addVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.nprobe = 1;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:            gpuIndex,
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    config.indicesOptions = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlat raftGpuIndex(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    raftGpuIndex.train(numTrain, trainVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    raftGpuIndex.add(numAdd, addVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    raftGpuIndex.nprobe = 1;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::compareIndices(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:            raftGpuIndex,
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:TEST(TestGpuIndexIVFFlat, Reconstruct_n) {
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    std::vector<float> trainVecs = faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlatConfig config;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    config.indicesOptions = faiss::gpu::INDICES_64_BIT;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlat gpuIndex(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.train(opt.numTrain, trainVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.add(opt.numAdd, addVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    std::vector<float> gpuVals(opt.numAdd * opt.dim);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex.reconstruct_n(0, gpuIndex.ntotal, gpuVals.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    EXPECT_EQ(gpuVals, cpuVals);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    config.indicesOptions = faiss::gpu::INDICES_32_BIT;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlat gpuIndex1(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex1.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex1.train(opt.numTrain, trainVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex1.add(opt.numAdd, addVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex1.reconstruct_n(0, gpuIndex1.ntotal, gpuVals.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    EXPECT_EQ(gpuVals, cpuVals);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    config.indicesOptions = faiss::gpu::INDICES_CPU;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::GpuIndexIVFFlat gpuIndex2(
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex2.nprobe = opt.nprobe;
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex2.train(opt.numTrain, trainVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex2.add(opt.numAdd, addVecs.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    gpuIndex2.reconstruct_n(0, gpuIndex2.ntotal, gpuVals.data());
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    EXPECT_EQ(gpuVals, cpuVals);
faiss/gpu/test/TestGpuIndexIVFFlat.cpp:    faiss::gpu::setTestSeed(100);
faiss/gpu/test/demo_ivfpq_indexing_gpu.cpp:#include <faiss/gpu/GpuAutoTune.h>
faiss/gpu/test/demo_ivfpq_indexing_gpu.cpp:#include <faiss/gpu/GpuCloner.h>
faiss/gpu/test/demo_ivfpq_indexing_gpu.cpp:#include <faiss/gpu/GpuIndexIVFPQ.h>
faiss/gpu/test/demo_ivfpq_indexing_gpu.cpp:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/test/demo_ivfpq_indexing_gpu.cpp:    faiss::gpu::StandardGpuResources resources;
faiss/gpu/test/demo_ivfpq_indexing_gpu.cpp:    faiss::gpu::GpuIndexIVFPQConfig config;
faiss/gpu/test/demo_ivfpq_indexing_gpu.cpp:    faiss::gpu::GpuIndexIVFPQ index(
faiss/gpu/test/demo_ivfpq_indexing_gpu.cpp:        faiss::Index* cpu_index = faiss::gpu::index_gpu_to_cpu(&index);
faiss/gpu/test/TestGpuMemoryException.cpp:#include <faiss/gpu/GpuIndexFlat.h>
faiss/gpu/test/TestGpuMemoryException.cpp:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/test/TestGpuMemoryException.cpp:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/test/TestGpuMemoryException.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/test/TestGpuMemoryException.cpp:// Test to see if we can recover after attempting to allocate too much GPU
faiss/gpu/test/TestGpuMemoryException.cpp:TEST(TestGpuMemoryException, AddException) {
faiss/gpu/test/TestGpuMemoryException.cpp:    CUDA_VERIFY(cudaMemGetInfo(&devFree, &devTotal));
faiss/gpu/test/TestGpuMemoryException.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuMemoryException.cpp:    faiss::gpu::GpuIndexFlatConfig config;
faiss/gpu/test/TestGpuMemoryException.cpp:    config.device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuMemoryException.cpp:    faiss::gpu::GpuIndexFlatL2 gpuIndexL2Broken(
faiss/gpu/test/TestGpuMemoryException.cpp:    faiss::gpu::GpuIndexFlatL2 gpuIndexL2(&res, (int)realAddDims, config);
faiss/gpu/test/TestGpuMemoryException.cpp:                gpuIndexL2Broken.add(numBrokenAdd, vecs.get()),
faiss/gpu/test/TestGpuMemoryException.cpp:        auto vecs = faiss::gpu::randVecs(numRealAdd, realAddDims);
faiss/gpu/test/TestGpuMemoryException.cpp:        EXPECT_NO_THROW(gpuIndexL2.add(numRealAdd, vecs.data()));
faiss/gpu/test/TestGpuMemoryException.cpp:                gpuIndexL2Broken.add(numBrokenAdd, vecs.get()),
faiss/gpu/test/TestGpuMemoryException.cpp:        auto vecs = faiss::gpu::randVecs(numQuery, realAddDims);
faiss/gpu/test/TestGpuMemoryException.cpp:                gpuIndexL2,
faiss/gpu/test/TestGpuMemoryException.cpp:    faiss::gpu::setTestSeed(100);
faiss/gpu/test/TestGpuIndexCagra.cu: * Copyright (c) 2024, NVIDIA CORPORATION.
faiss/gpu/test/TestGpuIndexCagra.cu:#include <faiss/gpu/GpuIndexCagra.h>
faiss/gpu/test/TestGpuIndexCagra.cu:#include <faiss/gpu/GpuResources.h>
faiss/gpu/test/TestGpuIndexCagra.cu:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/test/TestGpuIndexCagra.cu:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/test/TestGpuIndexCagra.cu:#include <faiss/gpu/utils/CopyUtils.cuh>
faiss/gpu/test/TestGpuIndexCagra.cu:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/test/TestGpuIndexCagra.cu:#include <raft/core/resource/cuda_stream.hpp>
faiss/gpu/test/TestGpuIndexCagra.cu:        numTrain = 2 * faiss::gpu::randVal(2000, 5000);
faiss/gpu/test/TestGpuIndexCagra.cu:        dim = faiss::gpu::randVal(4, 10);
faiss/gpu/test/TestGpuIndexCagra.cu:        numAdd = faiss::gpu::randVal(1000, 3000);
faiss/gpu/test/TestGpuIndexCagra.cu:        graphDegree = faiss::gpu::randSelect({32, 64});
faiss/gpu/test/TestGpuIndexCagra.cu:        intermediateGraphDegree = faiss::gpu::randSelect({64, 98});
faiss/gpu/test/TestGpuIndexCagra.cu:        buildAlgo = faiss::gpu::randSelect(
faiss/gpu/test/TestGpuIndexCagra.cu:                {faiss::gpu::graph_build_algo::IVF_PQ,
faiss/gpu/test/TestGpuIndexCagra.cu:                 faiss::gpu::graph_build_algo::NN_DESCENT});
faiss/gpu/test/TestGpuIndexCagra.cu:        numQuery = faiss::gpu::randVal(32, 100);
faiss/gpu/test/TestGpuIndexCagra.cu:        k = faiss::gpu::randVal(10, 30);
faiss/gpu/test/TestGpuIndexCagra.cu:        device = faiss::gpu::randVal(0, faiss::gpu::getNumDevices() - 1);
faiss/gpu/test/TestGpuIndexCagra.cu:    faiss::gpu::graph_build_algo buildAlgo;
faiss/gpu/test/TestGpuIndexCagra.cu:        if (opt.buildAlgo == faiss::gpu::graph_build_algo::NN_DESCENT &&
faiss/gpu/test/TestGpuIndexCagra.cu:                faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexCagra.cu:        // train gpu index
faiss/gpu/test/TestGpuIndexCagra.cu:        faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexCagra.cu:        faiss::gpu::GpuIndexCagraConfig config;
faiss/gpu/test/TestGpuIndexCagra.cu:        faiss::gpu::GpuIndexCagra gpuIndex(&res, cpuIndex.d, metric, config);
faiss/gpu/test/TestGpuIndexCagra.cu:        gpuIndex.train(opt.numTrain, trainVecs.data());
faiss/gpu/test/TestGpuIndexCagra.cu:        auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
faiss/gpu/test/TestGpuIndexCagra.cu:        auto gpuRes = res.getResources();
faiss/gpu/test/TestGpuIndexCagra.cu:        auto devAlloc = faiss::gpu::makeDevAlloc(
faiss/gpu/test/TestGpuIndexCagra.cu:                faiss::gpu::AllocType::FlatData,
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes->getDefaultStreamCurrentDevice());
faiss/gpu/test/TestGpuIndexCagra.cu:        faiss::gpu::DeviceTensor<float, 2, true> testDistance(
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
faiss/gpu/test/TestGpuIndexCagra.cu:        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> testIndices(
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
faiss/gpu/test/TestGpuIndexCagra.cu:        gpuIndex.search(
faiss/gpu/test/TestGpuIndexCagra.cu:        auto refDistanceDev = faiss::gpu::toDeviceTemporary(
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes.get(),
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes->getDefaultStreamCurrentDevice());
faiss/gpu/test/TestGpuIndexCagra.cu:        auto refIndicesDev = faiss::gpu::toDeviceTemporary(
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes.get(),
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes->getDefaultStreamCurrentDevice());
faiss/gpu/test/TestGpuIndexCagra.cu:        auto raft_handle = gpuRes->getRaftHandleCurrentDevice();
faiss/gpu/test/TestGpuIndexCagra.cu:TEST(TestGpuIndexCagra, Float32_Query_L2) {
faiss/gpu/test/TestGpuIndexCagra.cu:TEST(TestGpuIndexCagra, Float32_Query_IP) {
faiss/gpu/test/TestGpuIndexCagra.cu:        if (opt.buildAlgo == faiss::gpu::graph_build_algo::NN_DESCENT &&
faiss/gpu/test/TestGpuIndexCagra.cu:                faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexCagra.cu:        std::vector<float> addVecs = faiss::gpu::randVecs(opt.numAdd, opt.dim);
faiss/gpu/test/TestGpuIndexCagra.cu:        faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexCagra.cu:        // train gpu index and copy to cpu index
faiss/gpu/test/TestGpuIndexCagra.cu:        faiss::gpu::GpuIndexCagraConfig config;
faiss/gpu/test/TestGpuIndexCagra.cu:        faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metric, config);
faiss/gpu/test/TestGpuIndexCagra.cu:        gpuIndex.train(opt.numTrain, trainVecs.data());
faiss/gpu/test/TestGpuIndexCagra.cu:        gpuIndex.copyTo(&copiedCpuIndex);
faiss/gpu/test/TestGpuIndexCagra.cu:        auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
faiss/gpu/test/TestGpuIndexCagra.cu:        auto gpuRes = res.getResources();
faiss/gpu/test/TestGpuIndexCagra.cu:        auto refDistanceDev = faiss::gpu::toDeviceTemporary(
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes.get(),
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes->getDefaultStreamCurrentDevice());
faiss/gpu/test/TestGpuIndexCagra.cu:        auto refIndicesDev = faiss::gpu::toDeviceTemporary(
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes.get(),
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes->getDefaultStreamCurrentDevice());
faiss/gpu/test/TestGpuIndexCagra.cu:        auto copyRefDistanceDev = faiss::gpu::toDeviceTemporary(
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes.get(),
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes->getDefaultStreamCurrentDevice());
faiss/gpu/test/TestGpuIndexCagra.cu:        auto copyRefIndicesDev = faiss::gpu::toDeviceTemporary(
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes.get(),
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes->getDefaultStreamCurrentDevice());
faiss/gpu/test/TestGpuIndexCagra.cu:        auto raft_handle = gpuRes->getRaftHandleCurrentDevice();
faiss/gpu/test/TestGpuIndexCagra.cu:TEST(TestGpuIndexCagra, Float32_CopyTo_L2) {
faiss/gpu/test/TestGpuIndexCagra.cu:TEST(TestGpuIndexCagra, Float32_CopyTo_L2_BaseLevelOnly) {
faiss/gpu/test/TestGpuIndexCagra.cu:TEST(TestGpuIndexCagra, Float32_CopyTo_IP) {
faiss/gpu/test/TestGpuIndexCagra.cu:TEST(TestGpuIndexCagra, Float32_CopyTo_IP_BaseLevelOnly) {
faiss/gpu/test/TestGpuIndexCagra.cu:        if (opt.buildAlgo == faiss::gpu::graph_build_algo::NN_DESCENT &&
faiss/gpu/test/TestGpuIndexCagra.cu:                faiss::gpu::randVecs(opt.numTrain, opt.dim);
faiss/gpu/test/TestGpuIndexCagra.cu:        faiss::gpu::StandardGpuResources res;
faiss/gpu/test/TestGpuIndexCagra.cu:        // convert to gpu index
faiss/gpu/test/TestGpuIndexCagra.cu:        faiss::gpu::GpuIndexCagra copiedGpuIndex(&res, cpuIndex.d, metric);
faiss/gpu/test/TestGpuIndexCagra.cu:        copiedGpuIndex.copyFrom(&cpuIndex);
faiss/gpu/test/TestGpuIndexCagra.cu:        // train gpu index
faiss/gpu/test/TestGpuIndexCagra.cu:        faiss::gpu::GpuIndexCagraConfig config;
faiss/gpu/test/TestGpuIndexCagra.cu:        faiss::gpu::GpuIndexCagra gpuIndex(&res, opt.dim, metric, config);
faiss/gpu/test/TestGpuIndexCagra.cu:        gpuIndex.train(opt.numTrain, trainVecs.data());
faiss/gpu/test/TestGpuIndexCagra.cu:        auto queryVecs = faiss::gpu::randVecs(opt.numQuery, opt.dim);
faiss/gpu/test/TestGpuIndexCagra.cu:        auto gpuRes = res.getResources();
faiss/gpu/test/TestGpuIndexCagra.cu:        auto devAlloc = faiss::gpu::makeDevAlloc(
faiss/gpu/test/TestGpuIndexCagra.cu:                faiss::gpu::AllocType::FlatData,
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes->getDefaultStreamCurrentDevice());
faiss/gpu/test/TestGpuIndexCagra.cu:        faiss::gpu::DeviceTensor<float, 2, true> copyTestDistance(
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
faiss/gpu/test/TestGpuIndexCagra.cu:        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> copyTestIndices(
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
faiss/gpu/test/TestGpuIndexCagra.cu:        copiedGpuIndex.search(
faiss/gpu/test/TestGpuIndexCagra.cu:        faiss::gpu::DeviceTensor<float, 2, true> testDistance(
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
faiss/gpu/test/TestGpuIndexCagra.cu:        faiss::gpu::DeviceTensor<faiss::idx_t, 2, true> testIndices(
faiss/gpu/test/TestGpuIndexCagra.cu:                gpuRes.get(), devAlloc, {opt.numQuery, opt.k});
faiss/gpu/test/TestGpuIndexCagra.cu:        gpuIndex.search(
faiss/gpu/test/TestGpuIndexCagra.cu:        auto raft_handle = gpuRes->getRaftHandleCurrentDevice();
faiss/gpu/test/TestGpuIndexCagra.cu:TEST(TestGpuIndexCagra, Float32_CopyFrom_L2) {
faiss/gpu/test/TestGpuIndexCagra.cu:TEST(TestGpuIndexCagra, Float32_CopyFrom_IP) {
faiss/gpu/test/TestGpuIndexCagra.cu:    faiss::gpu::setTestSeed(100);
faiss/gpu/test/test_index_cpu_to_gpu.py:class TestMoveToGpu(unittest.TestCase):
faiss/gpu/test/test_index_cpu_to_gpu.py:        cls.res = faiss.StandardGpuResources()
faiss/gpu/test/test_index_cpu_to_gpu.py:        config = faiss.GpuClonerOptions()
faiss/gpu/test/test_index_cpu_to_gpu.py:        faiss.index_cpu_to_gpu(self.res, 0, idx, config)
faiss/gpu/GpuIndexBinaryFlat.cu:#include <faiss/gpu/GpuIndexBinaryFlat.h>
faiss/gpu/GpuIndexBinaryFlat.cu:#include <faiss/gpu/GpuResources.h>
faiss/gpu/GpuIndexBinaryFlat.cu:#include <faiss/gpu/impl/IndexUtils.h>
faiss/gpu/GpuIndexBinaryFlat.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/GpuIndexBinaryFlat.cu:#include <faiss/gpu/impl/BinaryFlatIndex.cuh>
faiss/gpu/GpuIndexBinaryFlat.cu:#include <faiss/gpu/utils/ConversionOperators.cuh>
faiss/gpu/GpuIndexBinaryFlat.cu:#include <faiss/gpu/utils/CopyUtils.cuh>
faiss/gpu/GpuIndexBinaryFlat.cu:namespace gpu {
faiss/gpu/GpuIndexBinaryFlat.cu:GpuIndexBinaryFlat::GpuIndexBinaryFlat(
faiss/gpu/GpuIndexBinaryFlat.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexBinaryFlat.cu:        GpuIndexBinaryFlatConfig config)
faiss/gpu/GpuIndexBinaryFlat.cu:GpuIndexBinaryFlat::GpuIndexBinaryFlat(
faiss/gpu/GpuIndexBinaryFlat.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexBinaryFlat.cu:        GpuIndexBinaryFlatConfig config)
faiss/gpu/GpuIndexBinaryFlat.cu:GpuIndexBinaryFlat::~GpuIndexBinaryFlat() {}
faiss/gpu/GpuIndexBinaryFlat.cu:int GpuIndexBinaryFlat::getDevice() const {
faiss/gpu/GpuIndexBinaryFlat.cu:std::shared_ptr<GpuResources> GpuIndexBinaryFlat::getResources() {
faiss/gpu/GpuIndexBinaryFlat.cu:void GpuIndexBinaryFlat::copyFrom(const faiss::IndexBinaryFlat* index) {
faiss/gpu/GpuIndexBinaryFlat.cu:void GpuIndexBinaryFlat::copyTo(faiss::IndexBinaryFlat* index) const {
faiss/gpu/GpuIndexBinaryFlat.cu:void GpuIndexBinaryFlat::add(idx_t n, const uint8_t* x) {
faiss/gpu/GpuIndexBinaryFlat.cu:void GpuIndexBinaryFlat::reset() {
faiss/gpu/GpuIndexBinaryFlat.cu:void GpuIndexBinaryFlat::search(
faiss/gpu/GpuIndexBinaryFlat.cu:    // The input vectors may be too large for the GPU, but we still
faiss/gpu/GpuIndexBinaryFlat.cu:    // GPU.
faiss/gpu/GpuIndexBinaryFlat.cu:        // `x` that won't fit on the GPU.
faiss/gpu/GpuIndexBinaryFlat.cu:        // -> GPU.
faiss/gpu/GpuIndexBinaryFlat.cu:        // fit on the GPU (e.g., n * k is too large for the GPU memory).
faiss/gpu/GpuIndexBinaryFlat.cu:void GpuIndexBinaryFlat::searchNonPaged_(
faiss/gpu/GpuIndexBinaryFlat.cu:void GpuIndexBinaryFlat::searchFromCpuPaged_(
faiss/gpu/GpuIndexBinaryFlat.cu:    // Just page without overlapping copy with compute (as GpuIndexFlat does)
faiss/gpu/GpuIndexBinaryFlat.cu:void GpuIndexBinaryFlat::reconstruct(faiss::idx_t key, uint8_t* out) const {
faiss/gpu/GpuIndexBinaryFlat.cu:} // namespace gpu
faiss/gpu/GpuIndexBinaryFlat.h:#include <faiss/gpu/GpuIndex.h>
faiss/gpu/GpuIndexBinaryFlat.h:#include <faiss/gpu/GpuResources.h>
faiss/gpu/GpuIndexBinaryFlat.h:namespace gpu {
faiss/gpu/GpuIndexBinaryFlat.h:struct GpuIndexBinaryFlatConfig : public GpuIndexConfig {};
faiss/gpu/GpuIndexBinaryFlat.h:/// A GPU version of IndexBinaryFlat for brute-force comparison of bit vectors
faiss/gpu/GpuIndexBinaryFlat.h:class GpuIndexBinaryFlat : public IndexBinary {
faiss/gpu/GpuIndexBinaryFlat.h:    /// data over to the given GPU
faiss/gpu/GpuIndexBinaryFlat.h:    GpuIndexBinaryFlat(
faiss/gpu/GpuIndexBinaryFlat.h:            GpuResourcesProvider* resources,
faiss/gpu/GpuIndexBinaryFlat.h:            GpuIndexBinaryFlatConfig config = GpuIndexBinaryFlatConfig());
faiss/gpu/GpuIndexBinaryFlat.h:    GpuIndexBinaryFlat(
faiss/gpu/GpuIndexBinaryFlat.h:            GpuResourcesProvider* resources,
faiss/gpu/GpuIndexBinaryFlat.h:            GpuIndexBinaryFlatConfig config = GpuIndexBinaryFlatConfig());
faiss/gpu/GpuIndexBinaryFlat.h:    ~GpuIndexBinaryFlat() override;
faiss/gpu/GpuIndexBinaryFlat.h:    /// Returns a reference to our GpuResources object that manages memory,
faiss/gpu/GpuIndexBinaryFlat.h:    /// stream and handle resources on the GPU
faiss/gpu/GpuIndexBinaryFlat.h:    std::shared_ptr<GpuResources> getResources();
faiss/gpu/GpuIndexBinaryFlat.h:    std::shared_ptr<GpuResources> resources_;
faiss/gpu/GpuIndexBinaryFlat.h:    const GpuIndexBinaryFlatConfig binaryFlatConfig_;
faiss/gpu/GpuIndexBinaryFlat.h:    /// Holds our GPU data containing the list of vectors
faiss/gpu/GpuIndexBinaryFlat.h:} // namespace gpu
faiss/gpu/GpuIndexCagra.h: * Copyright (c) 2024, NVIDIA CORPORATION.
faiss/gpu/GpuIndexCagra.h:#include <faiss/gpu/GpuIndex.h>
faiss/gpu/GpuIndexCagra.h:#include <faiss/gpu/GpuIndexIVFPQ.h>
faiss/gpu/GpuIndexCagra.h:namespace gpu {
faiss/gpu/GpuIndexCagra.h:    /// to use as little GPU memory for the database as possible.
faiss/gpu/GpuIndexCagra.h:    /// Possible values: [CUDA_R_32F, CUDA_R_16F, CUDA_R_8U]
faiss/gpu/GpuIndexCagra.h:    cudaDataType_t lut_dtype = CUDA_R_32F;
faiss/gpu/GpuIndexCagra.h:    /// Possible values: [CUDA_R_16F, CUDA_R_32F]
faiss/gpu/GpuIndexCagra.h:    cudaDataType_t internal_distance_dtype = CUDA_R_32F;
faiss/gpu/GpuIndexCagra.h:    /// One wants to increase the carveout to make sure a good GPU occupancy for
faiss/gpu/GpuIndexCagra.h:    /// Moreover, a GPU usually allows only a fixed set of cache configurations,
faiss/gpu/GpuIndexCagra.h:    /// to the NVIDIA tuning guide for the target GPU architecture.
faiss/gpu/GpuIndexCagra.h:struct GpuIndexCagraConfig : public GpuIndexConfig {
faiss/gpu/GpuIndexCagra.h:struct GpuIndexCagra : public GpuIndex {
faiss/gpu/GpuIndexCagra.h:    GpuIndexCagra(
faiss/gpu/GpuIndexCagra.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexCagra.h:            GpuIndexCagraConfig config = GpuIndexCagraConfig());
faiss/gpu/GpuIndexCagra.h:    /// Called from GpuIndex for search
faiss/gpu/GpuIndexCagra.h:    const GpuIndexCagraConfig cagraConfig_;
faiss/gpu/GpuIndexCagra.h:} // namespace gpu
faiss/gpu/GpuResources.h: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/GpuResources.h:#include <cuda_runtime.h>
faiss/gpu/GpuResources.h:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuResources.h:namespace gpu {
faiss/gpu/GpuResources.h:class GpuResources;
faiss/gpu/GpuResources.h:    /// Primary data storage for GpuIndexFlat (the raw matrix of vectors and
faiss/gpu/GpuResources.h:    /// Primary data storage for GpuIndexIVF* (the storage for each individual
faiss/gpu/GpuResources.h:    /// For GpuIndexIVFPQ, "precomputed codes" for more efficient PQ lookup
faiss/gpu/GpuResources.h:    /// StandardGpuResources implementation specific types
faiss/gpu/GpuResources.h:    /// When using StandardGpuResources, temporary memory allocations
faiss/gpu/GpuResources.h:    /// allocated up front for each gpu (e.g., 1.5 GiB upon initialization).
faiss/gpu/GpuResources.h:    /// allocation by StandardGpuResources is marked with this AllocType.
faiss/gpu/GpuResources.h:    /// When using StandardGpuResources, any MemorySpace::Temporary allocations
faiss/gpu/GpuResources.h:    /// to calling cudaMalloc which are sized to just the request at hand. These
faiss/gpu/GpuResources.h:/// Memory regions accessible to the GPU
faiss/gpu/GpuResources.h:    /// top-level index call, and where the streams using it have completed GPU
faiss/gpu/GpuResources.h:    /// work). Typically backed by Device memory (cudaMalloc/cudaFree).
faiss/gpu/GpuResources.h:    /// Managed using cudaMalloc/cudaFree (typical GPU device memory)
faiss/gpu/GpuResources.h:    /// Managed using cudaMallocManaged/cudaFree (typical Unified CPU/GPU
faiss/gpu/GpuResources.h:    inline AllocInfo(AllocType at, int dev, MemorySpace sp, cudaStream_t st)
faiss/gpu/GpuResources.h:    cudaStream_t stream = nullptr;
faiss/gpu/GpuResources.h:AllocInfo makeDevAlloc(AllocType at, cudaStream_t st);
faiss/gpu/GpuResources.h:AllocInfo makeTempAlloc(AllocType at, cudaStream_t st);
faiss/gpu/GpuResources.h:AllocInfo makeSpaceAlloc(AllocType at, MemorySpace sp, cudaStream_t st);
faiss/gpu/GpuResources.h:            cudaStream_t st,
faiss/gpu/GpuResources.h:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuResources.h:struct GpuMemoryReservation {
faiss/gpu/GpuResources.h:    GpuMemoryReservation();
faiss/gpu/GpuResources.h:    GpuMemoryReservation(
faiss/gpu/GpuResources.h:            GpuResources* r,
faiss/gpu/GpuResources.h:            cudaStream_t str,
faiss/gpu/GpuResources.h:    GpuMemoryReservation(GpuMemoryReservation&& m) noexcept;
faiss/gpu/GpuResources.h:    ~GpuMemoryReservation();
faiss/gpu/GpuResources.h:    GpuMemoryReservation& operator=(GpuMemoryReservation&& m);
faiss/gpu/GpuResources.h:    GpuResources* res;
faiss/gpu/GpuResources.h:    cudaStream_t stream;
faiss/gpu/GpuResources.h:/// Base class of GPU-side resource provider; hides provision of
faiss/gpu/GpuResources.h:/// cuBLAS handles, CUDA streams and all device memory allocation performed
faiss/gpu/GpuResources.h:class GpuResources {
faiss/gpu/GpuResources.h:    virtual ~GpuResources();
faiss/gpu/GpuResources.h:    virtual cudaStream_t getDefaultStream(int device) = 0;
faiss/gpu/GpuResources.h:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuResources.h:    virtual void setDefaultStream(int device, cudaStream_t stream) = 0;
faiss/gpu/GpuResources.h:    virtual std::vector<cudaStream_t> getAlternateStreams(int device) = 0;
faiss/gpu/GpuResources.h:    /// without cudaMalloc allocation?
faiss/gpu/GpuResources.h:    /// Returns the stream on which we perform async CPU <-> GPU copies
faiss/gpu/GpuResources.h:    virtual cudaStream_t getAsyncCopyStream(int device) = 0;
faiss/gpu/GpuResources.h:    cudaStream_t getDefaultStreamCurrentDevice();
faiss/gpu/GpuResources.h:    GpuMemoryReservation allocMemoryHandle(const AllocRequest& req);
faiss/gpu/GpuResources.h:    // equivalent to cudaDeviceSynchronize(getDefaultStream(device))
faiss/gpu/GpuResources.h:    std::vector<cudaStream_t> getAlternateStreamsCurrentDevice();
faiss/gpu/GpuResources.h:    cudaStream_t getAsyncCopyStreamCurrentDevice();
faiss/gpu/GpuResources.h:class GpuResourcesProvider {
faiss/gpu/GpuResources.h:    virtual ~GpuResourcesProvider();
faiss/gpu/GpuResources.h:    virtual std::shared_ptr<GpuResources> getResources() = 0;
faiss/gpu/GpuResources.h:/// A simple wrapper for a GpuResources object to make a GpuResourcesProvider
faiss/gpu/GpuResources.h:class GpuResourcesProviderFromInstance : public GpuResourcesProvider {
faiss/gpu/GpuResources.h:    explicit GpuResourcesProviderFromInstance(std::shared_ptr<GpuResources> p);
faiss/gpu/GpuResources.h:    ~GpuResourcesProviderFromInstance() override;
faiss/gpu/GpuResources.h:    std::shared_ptr<GpuResources> getResources() override;
faiss/gpu/GpuResources.h:    std::shared_ptr<GpuResources> res_;
faiss/gpu/GpuResources.h:} // namespace gpu
faiss/gpu/GpuIndicesOptions.h:namespace gpu {
faiss/gpu/GpuIndicesOptions.h:/// How user vector index data is stored on the GPU
faiss/gpu/GpuIndicesOptions.h:    /// The user indices are only stored on the CPU; the GPU returns
faiss/gpu/GpuIndicesOptions.h:    /// GPU. Only (inverted list, offset) is returned to the user as the
faiss/gpu/GpuIndicesOptions.h:    /// Indices are stored as 32 bit integers on the GPU, but returned
faiss/gpu/GpuIndicesOptions.h:    /// Indices are stored as 64 bit integers on the GPU
faiss/gpu/GpuIndicesOptions.h:} // namespace gpu
faiss/gpu/GpuIcmEncoder.h:namespace gpu {
faiss/gpu/GpuIcmEncoder.h:class GpuResourcesProvider;
faiss/gpu/GpuIcmEncoder.h:/** Perform LSQ encoding on GPU.
faiss/gpu/GpuIcmEncoder.h:class GpuIcmEncoder : public lsq::IcmEncoder {
faiss/gpu/GpuIcmEncoder.h:    GpuIcmEncoder(
faiss/gpu/GpuIcmEncoder.h:            const std::vector<GpuResourcesProvider*>& provs,
faiss/gpu/GpuIcmEncoder.h:    ~GpuIcmEncoder();
faiss/gpu/GpuIcmEncoder.h:    GpuIcmEncoder(const GpuIcmEncoder&) = delete;
faiss/gpu/GpuIcmEncoder.h:    GpuIcmEncoder& operator=(const GpuIcmEncoder&) = delete;
faiss/gpu/GpuIcmEncoder.h:struct GpuIcmEncoderFactory : public lsq::IcmEncoderFactory {
faiss/gpu/GpuIcmEncoder.h:    explicit GpuIcmEncoderFactory(int ngpus = 1);
faiss/gpu/GpuIcmEncoder.h:    std::vector<GpuResourcesProvider*> provs;
faiss/gpu/GpuIcmEncoder.h:} // namespace gpu
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:#include <faiss/gpu/GpuIndexFlat.h>
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:#include <faiss/gpu/GpuResources.h>
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:#include <faiss/gpu/impl/IVFFlat.cuh>
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:#include <faiss/gpu/utils/CopyUtils.cuh>
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:namespace gpu {
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:GpuIndexIVFScalarQuantizer::GpuIndexIVFScalarQuantizer(
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:        GpuIndexIVFScalarQuantizerConfig config)
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:        : GpuIndexIVF(
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:GpuIndexIVFScalarQuantizer::GpuIndexIVFScalarQuantizer(
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:        GpuIndexIVFScalarQuantizerConfig config)
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:        : GpuIndexIVF(provider, dims, metric, 0, nlist, config),
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:GpuIndexIVFScalarQuantizer::GpuIndexIVFScalarQuantizer(
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:        GpuIndexIVFScalarQuantizerConfig config)
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:        : GpuIndexIVF(
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:GpuIndexIVFScalarQuantizer::~GpuIndexIVFScalarQuantizer() {}
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:void GpuIndexIVFScalarQuantizer::verifySQSettings_() const {
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:            isSQSupported(sq.qtype), "Unsupported scalar QuantizerType on GPU");
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:                "GpuIndexIVFScalarQuantizer: Insufficient shared memory "
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:                "available on the GPU for QT_8bit or QT_4bit with %d "
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:void GpuIndexIVFScalarQuantizer::reserveMemory(size_t numVecs) {
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:void GpuIndexIVFScalarQuantizer::copyFrom(
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:    GpuIndexIVF::copyFrom(index);
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:void GpuIndexIVFScalarQuantizer::copyTo(
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:            "Cannot copy to CPU as GPU index doesn't retain "
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:    GpuIndexIVF::copyTo(index);
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:size_t GpuIndexIVFScalarQuantizer::reclaimMemory() {
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:void GpuIndexIVFScalarQuantizer::updateQuantizer() {
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:void GpuIndexIVFScalarQuantizer::reset() {
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:void GpuIndexIVFScalarQuantizer::trainResiduals_(idx_t n, const float* x) {
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:void GpuIndexIVFScalarQuantizer::train(idx_t n, const float* x) {
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:    // FIXME: GPUize more of this
faiss/gpu/GpuIndexIVFScalarQuantizer.cu:} // namespace gpu
faiss/gpu/GpuCloner.h:#include <faiss/gpu/GpuClonerOptions.h>
faiss/gpu/GpuCloner.h:#include <faiss/gpu/GpuIndex.h>
faiss/gpu/GpuCloner.h:#include <faiss/gpu/GpuIndicesOptions.h>
faiss/gpu/GpuCloner.h:namespace gpu {
faiss/gpu/GpuCloner.h:class GpuResourcesProvider;
faiss/gpu/GpuCloner.h:/// Cloner specialized for GPU -> CPU
faiss/gpu/GpuCloner.h:/// Cloner specialized for CPU -> 1 GPU
faiss/gpu/GpuCloner.h:struct ToGpuCloner : faiss::Cloner, GpuClonerOptions {
faiss/gpu/GpuCloner.h:    GpuResourcesProvider* provider;
faiss/gpu/GpuCloner.h:    ToGpuCloner(
faiss/gpu/GpuCloner.h:            GpuResourcesProvider* prov,
faiss/gpu/GpuCloner.h:            const GpuClonerOptions& options);
faiss/gpu/GpuCloner.h:/// Cloner specialized for CPU -> multiple GPUs
faiss/gpu/GpuCloner.h:struct ToGpuClonerMultiple : faiss::Cloner, GpuMultipleClonerOptions {
faiss/gpu/GpuCloner.h:    std::vector<ToGpuCloner> sub_cloners;
faiss/gpu/GpuCloner.h:    ToGpuClonerMultiple(
faiss/gpu/GpuCloner.h:            std::vector<GpuResourcesProvider*>& provider,
faiss/gpu/GpuCloner.h:            const GpuMultipleClonerOptions& options);
faiss/gpu/GpuCloner.h:    ToGpuClonerMultiple(
faiss/gpu/GpuCloner.h:            const std::vector<ToGpuCloner>& sub_cloners,
faiss/gpu/GpuCloner.h:            const GpuMultipleClonerOptions& options);
faiss/gpu/GpuCloner.h:/// converts any GPU index inside gpu_index to a CPU index
faiss/gpu/GpuCloner.h:faiss::Index* index_gpu_to_cpu(const faiss::Index* gpu_index);
faiss/gpu/GpuCloner.h:/// converts any CPU index that can be converted to GPU
faiss/gpu/GpuCloner.h:faiss::Index* index_cpu_to_gpu(
faiss/gpu/GpuCloner.h:        GpuResourcesProvider* provider,
faiss/gpu/GpuCloner.h:        const GpuClonerOptions* options = nullptr);
faiss/gpu/GpuCloner.h:faiss::Index* index_cpu_to_gpu_multiple(
faiss/gpu/GpuCloner.h:        std::vector<GpuResourcesProvider*>& provider,
faiss/gpu/GpuCloner.h:        const GpuMultipleClonerOptions* options = nullptr);
faiss/gpu/GpuCloner.h:struct GpuProgressiveDimIndexFactory : ProgressiveDimIndexFactory {
faiss/gpu/GpuCloner.h:    GpuMultipleClonerOptions options;
faiss/gpu/GpuCloner.h:    std::vector<GpuResourcesProvider*> vres;
faiss/gpu/GpuCloner.h:    explicit GpuProgressiveDimIndexFactory(int ngpu);
faiss/gpu/GpuCloner.h:    virtual ~GpuProgressiveDimIndexFactory() override;
faiss/gpu/GpuCloner.h:faiss::IndexBinary* index_binary_gpu_to_cpu(
faiss/gpu/GpuCloner.h:        const faiss::IndexBinary* gpu_index);
faiss/gpu/GpuCloner.h:/// converts any CPU index that can be converted to GPU
faiss/gpu/GpuCloner.h:faiss::IndexBinary* index_binary_cpu_to_gpu(
faiss/gpu/GpuCloner.h:        GpuResourcesProvider* provider,
faiss/gpu/GpuCloner.h:        const GpuClonerOptions* options = nullptr);
faiss/gpu/GpuCloner.h:faiss::IndexBinary* index_binary_cpu_to_gpu_multiple(
faiss/gpu/GpuCloner.h:        std::vector<GpuResourcesProvider*>& provider,
faiss/gpu/GpuCloner.h:        const GpuMultipleClonerOptions* options = nullptr);
faiss/gpu/GpuCloner.h:} // namespace gpu
faiss/gpu/GpuIndex.cu:#include <faiss/gpu/GpuIndex.h>
faiss/gpu/GpuIndex.cu:#include <faiss/gpu/GpuResources.h>
faiss/gpu/GpuIndex.cu:#include <faiss/gpu/impl/IndexUtils.h>
faiss/gpu/GpuIndex.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/GpuIndex.cu:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/GpuIndex.cu:#include <faiss/gpu/utils/CopyUtils.cuh>
faiss/gpu/GpuIndex.cu:namespace gpu {
faiss/gpu/GpuIndex.cu:/// Size above which we page copies from the CPU to GPU (non-paged
faiss/gpu/GpuIndex.cu:bool should_use_raft(GpuIndexConfig config_) {
faiss/gpu/GpuIndex.cu:        cudaDeviceProp prop;
faiss/gpu/GpuIndex.cu:        cudaGetDeviceProperties(&prop, config_.device);
faiss/gpu/GpuIndex.cu:GpuIndex::GpuIndex(
faiss/gpu/GpuIndex.cu:        std::shared_ptr<GpuResources> resources,
faiss/gpu/GpuIndex.cu:        GpuIndexConfig config)
faiss/gpu/GpuIndex.cu:            "Invalid GPU device %d",
faiss/gpu/GpuIndex.cu:            "Device %d does not support full CUDA 8 Unified Memory (CC 6.0+)",
faiss/gpu/GpuIndex.cu:int GpuIndex::getDevice() const {
faiss/gpu/GpuIndex.cu:void GpuIndex::copyFrom(const faiss::Index* index) {
faiss/gpu/GpuIndex.cu:void GpuIndex::copyTo(faiss::Index* index) const {
faiss/gpu/GpuIndex.cu:void GpuIndex::setMinPagingSize(size_t size) {
faiss/gpu/GpuIndex.cu:size_t GpuIndex::getMinPagingSize() const {
faiss/gpu/GpuIndex.cu:void GpuIndex::add(idx_t n, const float* x) {
faiss/gpu/GpuIndex.cu:void GpuIndex::add_with_ids(idx_t n, const float* x, const idx_t* ids) {
faiss/gpu/GpuIndex.cu:void GpuIndex::addPaged_(idx_t n, const float* x, const idx_t* ids) {
faiss/gpu/GpuIndex.cu:void GpuIndex::addPage_(idx_t n, const float* x, const idx_t* ids) {
faiss/gpu/GpuIndex.cu:    // At this point, `x` can be resident on CPU or GPU, and `ids` may be
faiss/gpu/GpuIndex.cu:    // resident on CPU, GPU or may be null.
faiss/gpu/GpuIndex.cu:    // GPU.
faiss/gpu/GpuIndex.cu:void GpuIndex::assign(idx_t n, const float* x, idx_t* labels, idx_t k) const {
faiss/gpu/GpuIndex.cu:void GpuIndex::search(
faiss/gpu/GpuIndex.cu:    // The input vectors may be too large for the GPU, but we still
faiss/gpu/GpuIndex.cu:    // GPU.
faiss/gpu/GpuIndex.cu:        // `x` that won't fit on the GPU.
faiss/gpu/GpuIndex.cu:        // -> GPU.
faiss/gpu/GpuIndex.cu:        // fit on the GPU (e.g., n * k is too large for the GPU memory).
faiss/gpu/GpuIndex.cu:void GpuIndex::search_and_reconstruct(
faiss/gpu/GpuIndex.cu:void GpuIndex::searchNonPaged_(
faiss/gpu/GpuIndex.cu:void GpuIndex::searchFromCpuPaged_(
faiss/gpu/GpuIndex.cu:    // 2 pinned copy -> GPU
faiss/gpu/GpuIndex.cu:    // 3 GPU compute
faiss/gpu/GpuIndex.cu:    // Reserve space on the GPU for the destination of the pinned buffer
faiss/gpu/GpuIndex.cu:    DeviceTensor<float, 2, true> bufGpuA(
faiss/gpu/GpuIndex.cu:    DeviceTensor<float, 2, true> bufGpuB(
faiss/gpu/GpuIndex.cu:    DeviceTensor<float, 2, true>* bufGpus[2] = {&bufGpuA, &bufGpuB};
faiss/gpu/GpuIndex.cu:    std::unique_ptr<CudaEvent> eventPinnedCopyDone[2];
faiss/gpu/GpuIndex.cu:    // Execute completion events for the GPU buffers
faiss/gpu/GpuIndex.cu:    std::unique_ptr<CudaEvent> eventGpuExecuteDone[2];
faiss/gpu/GpuIndex.cu:        // Start async pinned -> GPU copy first (buf 2)
faiss/gpu/GpuIndex.cu:            // Copy pinned to GPU
faiss/gpu/GpuIndex.cu:            auto& eventPrev = eventGpuExecuteDone[cur2BufIndex];
faiss/gpu/GpuIndex.cu:            CUDA_VERIFY(cudaMemcpyAsync(
faiss/gpu/GpuIndex.cu:                    bufGpus[cur2BufIndex]->data(),
faiss/gpu/GpuIndex.cu:                    cudaMemcpyHostToDevice,
faiss/gpu/GpuIndex.cu:            eventPinnedCopyDone[cur2BufIndex].reset(new CudaEvent(copyStream));
faiss/gpu/GpuIndex.cu:            // Process on GPU
faiss/gpu/GpuIndex.cu:            // DeviceTensor<float, 2, true> input(bufGpus[cur3BufIndex]->data(),
faiss/gpu/GpuIndex.cu:                    bufGpus[cur3BufIndex]->data(),
faiss/gpu/GpuIndex.cu:            eventGpuExecuteDone[cur3BufIndex].reset(
faiss/gpu/GpuIndex.cu:                    new CudaEvent(defaultStream));
faiss/gpu/GpuIndex.cu:void GpuIndex::compute_residual(const float* x, float* residual, idx_t key)
faiss/gpu/GpuIndex.cu:void GpuIndex::compute_residual_n(
faiss/gpu/GpuIndex.cu:std::shared_ptr<GpuResources> GpuIndex::getResources() {
faiss/gpu/GpuIndex.cu:GpuIndex* tryCastGpuIndex(faiss::Index* index) {
faiss/gpu/GpuIndex.cu:    return dynamic_cast<GpuIndex*>(index);
faiss/gpu/GpuIndex.cu:bool isGpuIndex(faiss::Index* index) {
faiss/gpu/GpuIndex.cu:    return tryCastGpuIndex(index) != nullptr;
faiss/gpu/GpuIndex.cu:bool isGpuIndexImplemented(faiss::Index* index) {
faiss/gpu/GpuIndex.cu:} // namespace gpu
faiss/gpu/GpuIndex.cu:// Crossing fingers that the InitGpuCompileOptions_instance will
faiss/gpu/GpuIndex.cu:extern std::string gpu_compile_options;
faiss/gpu/GpuIndex.cu:struct InitGpuCompileOptions {
faiss/gpu/GpuIndex.cu:    InitGpuCompileOptions() {
faiss/gpu/GpuIndex.cu:        gpu_compile_options = "GPU ";
faiss/gpu/GpuIndex.cu:#ifdef USE_NVIDIA_RAFT
faiss/gpu/GpuIndex.cu:        gpu_compile_options += "NVIDIA_RAFT ";
faiss/gpu/GpuIndex.cu:#ifdef USE_AMD_ROCM
faiss/gpu/GpuIndex.cu:        gpu_compile_options += "AMD_ROCM ";
faiss/gpu/GpuIndex.cu:InitGpuCompileOptions InitGpuCompileOptions_instance;
faiss/gpu/CMakeLists.txt:# Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/CMakeLists.txt:set(FAISS_GPU_SRC
faiss/gpu/CMakeLists.txt:  GpuAutoTune.cpp
faiss/gpu/CMakeLists.txt:  GpuCloner.cpp
faiss/gpu/CMakeLists.txt:  GpuDistance.cu
faiss/gpu/CMakeLists.txt:  GpuIcmEncoder.cu
faiss/gpu/CMakeLists.txt:  GpuIndex.cu
faiss/gpu/CMakeLists.txt:  GpuIndexBinaryFlat.cu
faiss/gpu/CMakeLists.txt:  GpuIndexFlat.cu
faiss/gpu/CMakeLists.txt:  GpuIndexIVF.cu
faiss/gpu/CMakeLists.txt:  GpuIndexIVFFlat.cu
faiss/gpu/CMakeLists.txt:  GpuIndexIVFPQ.cu
faiss/gpu/CMakeLists.txt:  GpuIndexIVFScalarQuantizer.cu
faiss/gpu/CMakeLists.txt:  GpuResources.cpp
faiss/gpu/CMakeLists.txt:  StandardGpuResources.cpp
faiss/gpu/CMakeLists.txt:set(FAISS_GPU_HEADERS
faiss/gpu/CMakeLists.txt:  GpuAutoTune.h
faiss/gpu/CMakeLists.txt:  GpuCloner.h
faiss/gpu/CMakeLists.txt:  GpuClonerOptions.h
faiss/gpu/CMakeLists.txt:  GpuDistance.h
faiss/gpu/CMakeLists.txt:  GpuIcmEncoder.h
faiss/gpu/CMakeLists.txt:  GpuFaissAssert.h
faiss/gpu/CMakeLists.txt:  GpuIndex.h
faiss/gpu/CMakeLists.txt:  GpuIndexBinaryFlat.h
faiss/gpu/CMakeLists.txt:  GpuIndexFlat.h
faiss/gpu/CMakeLists.txt:  GpuIndexIVF.h
faiss/gpu/CMakeLists.txt:  GpuIndexIVFFlat.h
faiss/gpu/CMakeLists.txt:  GpuIndexIVFPQ.h
faiss/gpu/CMakeLists.txt:  GpuIndexIVFScalarQuantizer.h
faiss/gpu/CMakeLists.txt:  GpuIndicesOptions.h
faiss/gpu/CMakeLists.txt:  GpuResources.h
faiss/gpu/CMakeLists.txt:  StandardGpuResources.h
faiss/gpu/CMakeLists.txt:  impl/GpuScalarQuantizer.cuh
faiss/gpu/CMakeLists.txt:    "faiss::gpu::Codec<0, 1>"
faiss/gpu/CMakeLists.txt:    "faiss::gpu::Codec<1, 1>"
faiss/gpu/CMakeLists.txt:    "faiss::gpu::Codec<2, 1>"
faiss/gpu/CMakeLists.txt:    "faiss::gpu::Codec<3, 1>"
faiss/gpu/CMakeLists.txt:    "faiss::gpu::Codec<4, 1>"
faiss/gpu/CMakeLists.txt:    "faiss::gpu::Codec<5, 1>"
faiss/gpu/CMakeLists.txt:    "faiss::gpu::Codec<6, 1>"
faiss/gpu/CMakeLists.txt:    "faiss::gpu::CodecFloat"
faiss/gpu/CMakeLists.txt:    "faiss::gpu::IPDistance"
faiss/gpu/CMakeLists.txt:    "faiss::gpu::L2Distance"
faiss/gpu/CMakeLists.txt:  if (FAISS_ENABLE_ROCM)
faiss/gpu/CMakeLists.txt:     list(TRANSFORM FAISS_GPU_SRC REPLACE cu$ hip)
faiss/gpu/CMakeLists.txt:    set(output_file "${CMAKE_CURRENT_BINARY_DIR}/${filename}.${GPU_EXT_PREFIX}")
faiss/gpu/CMakeLists.txt:    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/impl/scan/IVFInterleavedScanKernelTemplate.${GPU_EXT_PREFIX}" template_content)
faiss/gpu/CMakeLists.txt:    list(APPEND FAISS_GPU_SRC "${output_file}")
faiss/gpu/CMakeLists.txt:  set(FAISS_GPU_SRC "${FAISS_GPU_SRC}" PARENT_SCOPE)
faiss/gpu/CMakeLists.txt:  list(APPEND FAISS_GPU_HEADERS
faiss/gpu/CMakeLists.txt:          GpuIndexCagra.h
faiss/gpu/CMakeLists.txt:  list(APPEND FAISS_GPU_SRC
faiss/gpu/CMakeLists.txt:          GpuIndexCagra.cu
faiss/gpu/CMakeLists.txt:add_library(faiss_gpu STATIC ${FAISS_GPU_SRC})
faiss/gpu/CMakeLists.txt:set_target_properties(faiss_gpu PROPERTIES
faiss/gpu/CMakeLists.txt:target_include_directories(faiss_gpu PUBLIC
faiss/gpu/CMakeLists.txt:  target_compile_definitions(faiss PUBLIC USE_NVIDIA_RAFT=1)
faiss/gpu/CMakeLists.txt:  target_compile_definitions(faiss_avx2 PUBLIC USE_NVIDIA_RAFT=1)
faiss/gpu/CMakeLists.txt:  target_compile_definitions(faiss_avx512 PUBLIC USE_NVIDIA_RAFT=1)
faiss/gpu/CMakeLists.txt:  # dynamic library + CUDA runtime context which are requirements
faiss/gpu/CMakeLists.txt:    GpuDistance.cu
faiss/gpu/CMakeLists.txt:    StandardGpuResources.cpp
faiss/gpu/CMakeLists.txt:  target_compile_definitions(faiss_gpu PUBLIC USE_NVIDIA_RAFT=1)
faiss/gpu/CMakeLists.txt:if (FAISS_ENABLE_ROCM)
faiss/gpu/CMakeLists.txt:  list(TRANSFORM FAISS_GPU_SRC REPLACE cu$ hip)
faiss/gpu/CMakeLists.txt:# Export FAISS_GPU_HEADERS variable to parent scope.
faiss/gpu/CMakeLists.txt:set(FAISS_GPU_HEADERS ${FAISS_GPU_HEADERS} PARENT_SCOPE)
faiss/gpu/CMakeLists.txt:target_link_libraries(faiss PRIVATE  "$<LINK_LIBRARY:WHOLE_ARCHIVE,faiss_gpu>")
faiss/gpu/CMakeLists.txt:target_link_libraries(faiss_avx2 PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,faiss_gpu>")
faiss/gpu/CMakeLists.txt:target_link_libraries(faiss_avx512 PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,faiss_gpu>")
faiss/gpu/CMakeLists.txt:target_link_libraries(faiss_sve PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,faiss_gpu>")
faiss/gpu/CMakeLists.txt:foreach(header ${FAISS_GPU_HEADERS})
faiss/gpu/CMakeLists.txt:    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/gpu/${dir}
faiss/gpu/CMakeLists.txt:if (FAISS_ENABLE_ROCM)
faiss/gpu/CMakeLists.txt:  target_link_libraries(faiss_gpu PRIVATE hip::host roc::hipblas)
faiss/gpu/CMakeLists.txt:  target_compile_options(faiss_gpu PRIVATE)
faiss/gpu/CMakeLists.txt:  # This is what CUDA 11.5+ `nvcc -hls=gen-lcs -aug-hls` would generate
faiss/gpu/CMakeLists.txt:  target_link_options(faiss_gpu PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")
faiss/gpu/CMakeLists.txt:  find_package(CUDAToolkit REQUIRED)
faiss/gpu/CMakeLists.txt:  target_link_libraries(faiss_gpu PRIVATE CUDA::cudart CUDA::cublas
faiss/gpu/CMakeLists.txt:    $<$<BOOL:${FAISS_ENABLE_RAFT}>:nvidia::cutlass::cutlass>
faiss/gpu/CMakeLists.txt:  target_compile_options(faiss_gpu PRIVATE
faiss/gpu/CMakeLists.txt:    $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin=-compress-all
faiss/gpu/impl/FlatIndex.cuh: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/impl/FlatIndex.cuh:#include <faiss/gpu/GpuResources.h>
faiss/gpu/impl/FlatIndex.cuh:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/impl/FlatIndex.cuh:#include <faiss/gpu/utils/DeviceVector.cuh>
faiss/gpu/impl/FlatIndex.cuh:namespace gpu {
faiss/gpu/impl/FlatIndex.cuh:class GpuResources;
faiss/gpu/impl/FlatIndex.cuh:/// Holder of GPU resources for a particular flat index
faiss/gpu/impl/FlatIndex.cuh:    FlatIndex(GpuResources* res, int dim, bool useFloat16, MemorySpace space);
faiss/gpu/impl/FlatIndex.cuh:    void reserve(size_t numVecs, cudaStream_t stream);
faiss/gpu/impl/FlatIndex.cuh:    void add(const float* data, idx_t numVecs, cudaStream_t stream);
faiss/gpu/impl/FlatIndex.cuh:    /// Collection of GPU resources that we use
faiss/gpu/impl/FlatIndex.cuh:    GpuResources* resources_;
faiss/gpu/impl/FlatIndex.cuh:} // namespace gpu
faiss/gpu/impl/BinaryFlatIndex.cu:#include <faiss/gpu/GpuResources.h>
faiss/gpu/impl/BinaryFlatIndex.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/BinaryFlatIndex.cu:#include <faiss/gpu/impl/BinaryDistance.cuh>
faiss/gpu/impl/BinaryFlatIndex.cu:#include <faiss/gpu/impl/BinaryFlatIndex.cuh>
faiss/gpu/impl/BinaryFlatIndex.cu:namespace gpu {
faiss/gpu/impl/BinaryFlatIndex.cu:BinaryFlatIndex::BinaryFlatIndex(GpuResources* res, int dim, MemorySpace space)
faiss/gpu/impl/BinaryFlatIndex.cu:void BinaryFlatIndex::reserve(size_t numVecs, cudaStream_t stream) {
faiss/gpu/impl/BinaryFlatIndex.cu:        cudaStream_t stream) {
faiss/gpu/impl/BinaryFlatIndex.cu:} // namespace gpu
faiss/gpu/impl/L2Norm.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/L2Norm.cu:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/impl/L2Norm.cu:#include <faiss/gpu/impl/L2Norm.cuh>
faiss/gpu/impl/L2Norm.cu:#include <faiss/gpu/utils/ConversionOperators.cuh>
faiss/gpu/impl/L2Norm.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/impl/L2Norm.cu:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/impl/L2Norm.cu:#include <faiss/gpu/utils/MathOperators.cuh>
faiss/gpu/impl/L2Norm.cu:#include <faiss/gpu/utils/PtxUtils.cuh>
faiss/gpu/impl/L2Norm.cu:#include <faiss/gpu/utils/Reductions.cuh>
faiss/gpu/impl/L2Norm.cu:namespace gpu {
faiss/gpu/impl/L2Norm.cu:        cudaStream_t stream) {
faiss/gpu/impl/L2Norm.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/L2Norm.cu:        cudaStream_t stream) {
faiss/gpu/impl/L2Norm.cu:        cudaStream_t stream) {
faiss/gpu/impl/L2Norm.cu:} // namespace gpu
faiss/gpu/impl/IVFUtilsSelect1.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/IVFUtilsSelect1.cu:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/impl/IVFUtilsSelect1.cu:#include <faiss/gpu/impl/IVFUtils.cuh>
faiss/gpu/impl/IVFUtilsSelect1.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/impl/IVFUtilsSelect1.cu:#include <faiss/gpu/utils/Limits.cuh>
faiss/gpu/impl/IVFUtilsSelect1.cu:#include <faiss/gpu/utils/Select.cuh>
faiss/gpu/impl/IVFUtilsSelect1.cu:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/IVFUtilsSelect1.cu:namespace gpu {
faiss/gpu/impl/IVFUtilsSelect1.cu:        cudaStream_t stream) {
faiss/gpu/impl/IVFUtilsSelect1.cu:    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);
faiss/gpu/impl/IVFUtilsSelect1.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/impl/IVFUtilsSelect1.cu:#endif // GPU_MAX_SELECTION_K
faiss/gpu/impl/IVFUtilsSelect1.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/IVFUtilsSelect1.cu:} // namespace gpu
faiss/gpu/impl/BroadcastSum.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/BroadcastSum.cu:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/impl/BroadcastSum.cu:#include <faiss/gpu/utils/MathOperators.cuh>
faiss/gpu/impl/BroadcastSum.cu:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/BroadcastSum.cu:namespace gpu {
faiss/gpu/impl/BroadcastSum.cu:        cudaStream_t stream) {
faiss/gpu/impl/BroadcastSum.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/BroadcastSum.cu:        cudaStream_t stream) {
faiss/gpu/impl/BroadcastSum.cu:        cudaStream_t stream) {
faiss/gpu/impl/BroadcastSum.cu:        cudaStream_t stream) {
faiss/gpu/impl/BroadcastSum.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/BroadcastSum.cu:        cudaStream_t stream) {
faiss/gpu/impl/BroadcastSum.cu:        cudaStream_t stream) {
faiss/gpu/impl/BroadcastSum.cu:        cudaStream_t stream) {
faiss/gpu/impl/BroadcastSum.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/BroadcastSum.cu:        cudaStream_t stream) {
faiss/gpu/impl/BroadcastSum.cu:        cudaStream_t stream) {
faiss/gpu/impl/BroadcastSum.cu:} // namespace gpu
faiss/gpu/impl/FlatIndex.cu: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/impl/FlatIndex.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/FlatIndex.cu:#include <faiss/gpu/impl/Distance.cuh>
faiss/gpu/impl/FlatIndex.cu:#include <faiss/gpu/impl/FlatIndex.cuh>
faiss/gpu/impl/FlatIndex.cu:#include <faiss/gpu/impl/L2Norm.cuh>
faiss/gpu/impl/FlatIndex.cu:#include <faiss/gpu/impl/VectorResidual.cuh>
faiss/gpu/impl/FlatIndex.cu:#include <faiss/gpu/utils/ConversionOperators.cuh>
faiss/gpu/impl/FlatIndex.cu:#include <faiss/gpu/utils/CopyUtils.cuh>
faiss/gpu/impl/FlatIndex.cu:#include <faiss/gpu/utils/Transpose.cuh>
faiss/gpu/impl/FlatIndex.cu:namespace gpu {
faiss/gpu/impl/FlatIndex.cu:        GpuResources* res,
faiss/gpu/impl/FlatIndex.cu:void FlatIndex::reserve(size_t numVecs, cudaStream_t stream) {
faiss/gpu/impl/FlatIndex.cu:void FlatIndex::add(const float* data, idx_t numVecs, cudaStream_t stream) {
faiss/gpu/impl/FlatIndex.cu:} // namespace gpu
faiss/gpu/impl/BinaryDistance.cuh:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/impl/BinaryDistance.cuh:namespace gpu {
faiss/gpu/impl/BinaryDistance.cuh:        cudaStream_t stream);
faiss/gpu/impl/BinaryDistance.cuh:} // namespace gpu
faiss/gpu/impl/DistanceUtils.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/DistanceUtils.cuh:namespace gpu {
faiss/gpu/impl/DistanceUtils.cuh:        cudaStream_t stream) {
faiss/gpu/impl/DistanceUtils.cuh:    CUDA_TEST_ERROR();
faiss/gpu/impl/DistanceUtils.cuh:    // (or not). For <= 4 GB GPUs, prefer 512 MB of usage. For <= 8 GB GPUs,
faiss/gpu/impl/DistanceUtils.cuh:} // namespace gpu
faiss/gpu/impl/IVFAppend.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/IVFAppend.cu:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/impl/IVFAppend.cu:#include <faiss/gpu/impl/IVFAppend.cuh>
faiss/gpu/impl/IVFAppend.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/impl/IVFAppend.cu:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/impl/IVFAppend.cu:#include <faiss/gpu/utils/PtxUtils.cuh>
faiss/gpu/impl/IVFAppend.cu:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/IVFAppend.cu:#include <faiss/gpu/utils/WarpPackedBits.cuh>
faiss/gpu/impl/IVFAppend.cu:#include <faiss/gpu/utils/WarpShuffles.cuh>
faiss/gpu/impl/IVFAppend.cu:namespace gpu {
faiss/gpu/impl/IVFAppend.cu:        cudaStream_t stream) {
faiss/gpu/impl/IVFAppend.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/IVFAppend.cu:        cudaStream_t stream) {
faiss/gpu/impl/IVFAppend.cu:        CUDA_TEST_ERROR();
faiss/gpu/impl/IVFAppend.cu:        GpuScalarQuantizer* scalarQ,
faiss/gpu/impl/IVFAppend.cu:        cudaStream_t stream) {
faiss/gpu/impl/IVFAppend.cu:                        scalarQ->gpuTrained.data(),
faiss/gpu/impl/IVFAppend.cu:                        scalarQ->gpuTrained.data() + dim);
faiss/gpu/impl/IVFAppend.cu:                        scalarQ->gpuTrained.data(),
faiss/gpu/impl/IVFAppend.cu:                        scalarQ->gpuTrained.data() + dim);
faiss/gpu/impl/IVFAppend.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/IVFAppend.cu:        cudaStream_t stream) {
faiss/gpu/impl/IVFAppend.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/IVFAppend.cu:        cudaStream_t stream) {
faiss/gpu/impl/IVFAppend.cu:    // FIXME: some issue with getLaneId() and CUDA 10.1 and P4 GPUs?
faiss/gpu/impl/IVFAppend.cu:            (faiss::gpu::Tensor<long, 1, true>::DataType)warpId * kWarpSize;
faiss/gpu/impl/IVFAppend.cu:        GpuScalarQuantizer* scalarQ,
faiss/gpu/impl/IVFAppend.cu:        GpuResources* res,
faiss/gpu/impl/IVFAppend.cu:        cudaStream_t stream) {
faiss/gpu/impl/IVFAppend.cu:                        scalarQ->gpuTrained.data(),
faiss/gpu/impl/IVFAppend.cu:                        scalarQ->gpuTrained.data() + dim);
faiss/gpu/impl/IVFAppend.cu:                        scalarQ->gpuTrained.data(),
faiss/gpu/impl/IVFAppend.cu:                        scalarQ->gpuTrained.data() + dim);
faiss/gpu/impl/IVFAppend.cu:                        scalarQ->gpuTrained.data(),
faiss/gpu/impl/IVFAppend.cu:                        scalarQ->gpuTrained.data() + dim);
faiss/gpu/impl/IVFAppend.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/IVFAppend.cu:        cudaStream_t stream) {
faiss/gpu/impl/IVFAppend.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/IVFAppend.cu:} // namespace gpu
faiss/gpu/impl/IVFUtils.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/IVFUtils.cu:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/impl/IVFUtils.cu:#include <faiss/gpu/impl/IVFUtils.cuh>
faiss/gpu/impl/IVFUtils.cu:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/IVFUtils.cu:#include <faiss/gpu/utils/ThrustUtils.cuh>
faiss/gpu/impl/IVFUtils.cu:namespace gpu {
faiss/gpu/impl/IVFUtils.cu:        // the GPU can likely allocate.
faiss/gpu/impl/IVFUtils.cu:        GpuResources* res,
faiss/gpu/impl/IVFUtils.cu:        cudaStream_t stream) {
faiss/gpu/impl/IVFUtils.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/IVFUtils.cu:    // one, so it won't call cudaMalloc/Free if we size it sufficiently
faiss/gpu/impl/IVFUtils.cu:            thrust::cuda::par(alloc).on(stream),
faiss/gpu/impl/IVFUtils.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/IVFUtils.cu:} // namespace gpu
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:#include <faiss/gpu/GpuResources.h>
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:#include <faiss/gpu/impl/IVFUtils.cuh>
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:#include <faiss/gpu/impl/PQCodeDistances.cuh>
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:#include <faiss/gpu/impl/PQCodeLoad.cuh>
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:#include <faiss/gpu/utils/ConversionOperators.cuh>
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:#include <faiss/gpu/utils/HostTensor.cuh>
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:#include <faiss/gpu/utils/LoadStoreOperators.cuh>
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:#include <faiss/gpu/utils/NoTypeTensor.cuh>
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:#include <faiss/gpu/utils/WarpPackedBits.cuh>
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:namespace gpu {
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:    // FIXME: some issue with getLaneId() and CUDA 10.1 and P4 GPUs?
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:        GpuResources* res,
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:        cudaStream_t stream) {
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:    CUDA_TEST_ERROR();
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:        GpuResources* res) {
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:    CUDA_VERIFY(cudaMemsetAsync(
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:    CUDA_VERIFY(cudaMemsetAsync(
faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh:} // namespace gpu
faiss/gpu/impl/IVFAppend.cuh:#include <faiss/gpu/GpuIndicesOptions.h>
faiss/gpu/impl/IVFAppend.cuh:#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
faiss/gpu/impl/IVFAppend.cuh:#include <faiss/gpu/utils/DeviceVector.cuh>
faiss/gpu/impl/IVFAppend.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/IVFAppend.cuh:namespace gpu {
faiss/gpu/impl/IVFAppend.cuh:        cudaStream_t stream);
faiss/gpu/impl/IVFAppend.cuh:        cudaStream_t stream);
faiss/gpu/impl/IVFAppend.cuh:        cudaStream_t stream);
faiss/gpu/impl/IVFAppend.cuh:        cudaStream_t stream);
faiss/gpu/impl/IVFAppend.cuh:        GpuScalarQuantizer* scalarQ,
faiss/gpu/impl/IVFAppend.cuh:        cudaStream_t stream);
faiss/gpu/impl/IVFAppend.cuh:        GpuScalarQuantizer* scalarQ,
faiss/gpu/impl/IVFAppend.cuh:        GpuResources* res,
faiss/gpu/impl/IVFAppend.cuh:        cudaStream_t stream);
faiss/gpu/impl/IVFAppend.cuh:} // namespace gpu
faiss/gpu/impl/IcmEncoder.cuh:#include <faiss/gpu/GpuResources.h>
faiss/gpu/impl/IcmEncoder.cuh:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/impl/IcmEncoder.cuh:namespace gpu {
faiss/gpu/impl/IcmEncoder.cuh:    GpuResourcesProvider* prov;
faiss/gpu/impl/IcmEncoder.cuh:    std::shared_ptr<GpuResources> res;
faiss/gpu/impl/IcmEncoder.cuh:            GpuResourcesProvider* prov,
faiss/gpu/impl/IcmEncoder.cuh:} // namespace gpu
faiss/gpu/impl/IndexUtils.h:namespace gpu {
faiss/gpu/impl/IndexUtils.h:/// Returns the maximum k-selection value supported based on the CUDA SDK that
faiss/gpu/impl/IndexUtils.h:/// non-CUDA files
faiss/gpu/impl/IndexUtils.h:} // namespace gpu
faiss/gpu/impl/BinaryFlatIndex.cuh:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/impl/BinaryFlatIndex.cuh:#include <faiss/gpu/utils/DeviceVector.cuh>
faiss/gpu/impl/BinaryFlatIndex.cuh:namespace gpu {
faiss/gpu/impl/BinaryFlatIndex.cuh:class GpuResources;
faiss/gpu/impl/BinaryFlatIndex.cuh:/// Holder of GPU resources for a particular flat index
faiss/gpu/impl/BinaryFlatIndex.cuh:    BinaryFlatIndex(GpuResources* res, int dim, MemorySpace space);
faiss/gpu/impl/BinaryFlatIndex.cuh:    void reserve(size_t numVecs, cudaStream_t stream);
faiss/gpu/impl/BinaryFlatIndex.cuh:    void add(const unsigned char* data, idx_t numVecs, cudaStream_t stream);
faiss/gpu/impl/BinaryFlatIndex.cuh:    /// Collection of GPU resources that we use
faiss/gpu/impl/BinaryFlatIndex.cuh:    GpuResources* resources_;
faiss/gpu/impl/BinaryFlatIndex.cuh:} // namespace gpu
faiss/gpu/impl/VectorResidual.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/VectorResidual.cu:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/impl/VectorResidual.cu:#ifdef USE_AMD_ROCM
faiss/gpu/impl/VectorResidual.cu:#define CUDART_NAN_F __int_as_float(0x7fffffff)
faiss/gpu/impl/VectorResidual.cu:#include <math_constants.h> // in CUDA SDK, for CUDART_NAN_F
faiss/gpu/impl/VectorResidual.cu:#include <faiss/gpu/impl/VectorResidual.cuh>
faiss/gpu/impl/VectorResidual.cu:#include <faiss/gpu/utils/ConversionOperators.cuh>
faiss/gpu/impl/VectorResidual.cu:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/VectorResidual.cu:namespace gpu {
faiss/gpu/impl/VectorResidual.cu:                residual[i] = CUDART_NAN_F;
faiss/gpu/impl/VectorResidual.cu:            residual[threadIdx.x] = CUDART_NAN_F;
faiss/gpu/impl/VectorResidual.cu:        cudaStream_t stream) {
faiss/gpu/impl/VectorResidual.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/VectorResidual.cu:        cudaStream_t stream) {
faiss/gpu/impl/VectorResidual.cu:        cudaStream_t stream) {
faiss/gpu/impl/VectorResidual.cu:        cudaStream_t stream) {
faiss/gpu/impl/VectorResidual.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/VectorResidual.cu:        cudaStream_t stream) {
faiss/gpu/impl/VectorResidual.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/VectorResidual.cu:        cudaStream_t stream) {
faiss/gpu/impl/VectorResidual.cu:        cudaStream_t stream) {
faiss/gpu/impl/VectorResidual.cu:        cudaStream_t stream) {
faiss/gpu/impl/VectorResidual.cu:        cudaStream_t stream) {
faiss/gpu/impl/VectorResidual.cu:} // namespace gpu
faiss/gpu/impl/RemapIndices.h:namespace gpu {
faiss/gpu/impl/RemapIndices.h:} // namespace gpu
faiss/gpu/impl/L2Select.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/L2Select.cuh:namespace gpu {
faiss/gpu/impl/L2Select.cuh:        cudaStream_t stream);
faiss/gpu/impl/IcmEncoder.cu:#include <faiss/gpu/impl/IcmEncoder.cuh>
faiss/gpu/impl/IcmEncoder.cu:#include <faiss/gpu/GpuResources.h>
faiss/gpu/impl/IcmEncoder.cu:#include <faiss/gpu/impl/L2Norm.cuh>
faiss/gpu/impl/IcmEncoder.cu:#include <faiss/gpu/utils/CopyUtils.cuh>
faiss/gpu/impl/IcmEncoder.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/impl/IcmEncoder.cu:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/impl/IcmEncoder.cu:#include <faiss/gpu/utils/MatrixMult.cuh>
faiss/gpu/impl/IcmEncoder.cu:#include <faiss/gpu/utils/Pair.cuh>
faiss/gpu/impl/IcmEncoder.cu:#include <faiss/gpu/utils/Reductions.cuh>
faiss/gpu/impl/IcmEncoder.cu:namespace gpu {
faiss/gpu/impl/IcmEncoder.cu:        GpuResourcesProvider* prov,
faiss/gpu/impl/IcmEncoder.cu:} // namespace gpu
faiss/gpu/impl/RaftCagra.cu: * Copyright (c) 2024, NVIDIA CORPORATION.
faiss/gpu/impl/RaftCagra.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/RaftCagra.cu:#include <faiss/gpu/impl/RaftCagra.cuh>
faiss/gpu/impl/RaftCagra.cu:namespace gpu {
faiss/gpu/impl/RaftCagra.cu:        GpuResources* resources,
faiss/gpu/impl/RaftCagra.cu:            indicesOptions == faiss::gpu::INDICES_64_BIT,
faiss/gpu/impl/RaftCagra.cu:        GpuResources* resources,
faiss/gpu/impl/RaftCagra.cu:            indicesOptions == faiss::gpu::INDICES_64_BIT,
faiss/gpu/impl/RaftCagra.cu:    auto distances_on_gpu = getDeviceForAddress(distances) >= 0;
faiss/gpu/impl/RaftCagra.cu:    auto knn_graph_on_gpu = getDeviceForAddress(knn_graph) >= 0;
faiss/gpu/impl/RaftCagra.cu:    FAISS_ASSERT(distances_on_gpu == knn_graph_on_gpu);
faiss/gpu/impl/RaftCagra.cu:    if (distances_on_gpu && knn_graph_on_gpu) {
faiss/gpu/impl/RaftCagra.cu:    } else if (!distances_on_gpu && !knn_graph_on_gpu) {
faiss/gpu/impl/RaftCagra.cu:    RAFT_CUDA_TRY(cudaMemcpy2DAsync(
faiss/gpu/impl/RaftCagra.cu:            cudaMemcpyDefault,
faiss/gpu/impl/RaftCagra.cu:} // namespace gpu
faiss/gpu/impl/RaftFlatIndex.cuh: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/impl/RaftFlatIndex.cuh:#include <faiss/gpu/GpuResources.h>
faiss/gpu/impl/RaftFlatIndex.cuh:#include <faiss/gpu/impl/FlatIndex.cuh>
faiss/gpu/impl/RaftFlatIndex.cuh:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/impl/RaftFlatIndex.cuh:#include <faiss/gpu/utils/DeviceVector.cuh>
faiss/gpu/impl/RaftFlatIndex.cuh:namespace gpu {
faiss/gpu/impl/RaftFlatIndex.cuh:class GpuResources;
faiss/gpu/impl/RaftFlatIndex.cuh:/// Holder of GPU resources for a particular flat index
faiss/gpu/impl/RaftFlatIndex.cuh:            GpuResources* res,
faiss/gpu/impl/RaftFlatIndex.cuh:} // namespace gpu
faiss/gpu/impl/VectorResidual.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/VectorResidual.cuh:#include <cuda_fp16.h>
faiss/gpu/impl/VectorResidual.cuh:namespace gpu {
faiss/gpu/impl/VectorResidual.cuh:        cudaStream_t stream);
faiss/gpu/impl/VectorResidual.cuh:        cudaStream_t stream);
faiss/gpu/impl/VectorResidual.cuh:        cudaStream_t stream);
faiss/gpu/impl/VectorResidual.cuh:        cudaStream_t stream);
faiss/gpu/impl/VectorResidual.cuh:        cudaStream_t stream);
faiss/gpu/impl/VectorResidual.cuh:        cudaStream_t stream);
faiss/gpu/impl/VectorResidual.cuh:} // namespace gpu
faiss/gpu/impl/RaftCagra.cuh: * Copyright (c) 2024, NVIDIA CORPORATION.
faiss/gpu/impl/RaftCagra.cuh:#include <faiss/gpu/GpuIndicesOptions.h>
faiss/gpu/impl/RaftCagra.cuh:#include <faiss/gpu/GpuResources.h>
faiss/gpu/impl/RaftCagra.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/RaftCagra.cuh:namespace gpu {
faiss/gpu/impl/RaftCagra.cuh:            GpuResources* resources,
faiss/gpu/impl/RaftCagra.cuh:            GpuResources* resources,
faiss/gpu/impl/RaftCagra.cuh:    /// Collection of GPU resources that we use
faiss/gpu/impl/RaftCagra.cuh:    GpuResources* resources_;
faiss/gpu/impl/RaftCagra.cuh:} // namespace gpu
faiss/gpu/impl/PQCodeLoad.cuh:#include <faiss/gpu/utils/PtxUtils.cuh>
faiss/gpu/impl/PQCodeLoad.cuh:namespace gpu {
faiss/gpu/impl/PQCodeLoad.cuh:#if __CUDA_ARCH__ >= 350
faiss/gpu/impl/PQCodeLoad.cuh:#endif // __CUDA_ARCH__
faiss/gpu/impl/PQCodeLoad.cuh:#ifdef USE_AMD_ROCM
faiss/gpu/impl/PQCodeLoad.cuh:#else // USE_AMD_ROCM
faiss/gpu/impl/PQCodeLoad.cuh:#endif // USE_AMD_ROCM
faiss/gpu/impl/PQCodeLoad.cuh:} // namespace gpu
faiss/gpu/impl/Distance.cu:#include <faiss/gpu/GpuResources.h>
faiss/gpu/impl/Distance.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/Distance.cu:#include <faiss/gpu/impl/BroadcastSum.cuh>
faiss/gpu/impl/Distance.cu:#include <faiss/gpu/impl/Distance.cuh>
faiss/gpu/impl/Distance.cu:#include <faiss/gpu/impl/DistanceUtils.cuh>
faiss/gpu/impl/Distance.cu:#include <faiss/gpu/impl/L2Norm.cuh>
faiss/gpu/impl/Distance.cu:#include <faiss/gpu/impl/L2Select.cuh>
faiss/gpu/impl/Distance.cu:#include <faiss/gpu/utils/BlockSelectKernel.cuh>
faiss/gpu/impl/Distance.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/impl/Distance.cu:#include <faiss/gpu/utils/Limits.cuh>
faiss/gpu/impl/Distance.cu:#include <faiss/gpu/utils/MatrixMult.cuh>
faiss/gpu/impl/Distance.cu:namespace gpu {
faiss/gpu/impl/Distance.cu:        GpuResources* res,
faiss/gpu/impl/Distance.cu:        cudaStream_t stream,
faiss/gpu/impl/Distance.cu:                thrust::cuda::par.on(stream),
faiss/gpu/impl/Distance.cu:        GpuResources* res,
faiss/gpu/impl/Distance.cu:        cudaStream_t stream,
faiss/gpu/impl/Distance.cu:                thrust::cuda::par.on(stream),
faiss/gpu/impl/Distance.cu:                thrust::cuda::par.on(stream),
faiss/gpu/impl/Distance.cu:    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K); // select limitation
faiss/gpu/impl/Distance.cu:        GpuResources* res,
faiss/gpu/impl/Distance.cu:        cudaStream_t stream,
faiss/gpu/impl/Distance.cu:        GpuResources* res,
faiss/gpu/impl/Distance.cu:        cudaStream_t stream,
faiss/gpu/impl/Distance.cu:        GpuResources* res,
faiss/gpu/impl/Distance.cu:        cudaStream_t stream,
faiss/gpu/impl/Distance.cu:        GpuResources* res,
faiss/gpu/impl/Distance.cu:        cudaStream_t stream,
faiss/gpu/impl/Distance.cu:        GpuResources* res,
faiss/gpu/impl/Distance.cu:        cudaStream_t stream,
faiss/gpu/impl/Distance.cu:        GpuResources* res,
faiss/gpu/impl/Distance.cu:        cudaStream_t stream,
faiss/gpu/impl/Distance.cu:        GpuResources* res,
faiss/gpu/impl/Distance.cu:        cudaStream_t stream,
faiss/gpu/impl/Distance.cu:        GpuResources* res,
faiss/gpu/impl/Distance.cu:        cudaStream_t stream,
faiss/gpu/impl/Distance.cu:        GpuResources* res,
faiss/gpu/impl/Distance.cu:        cudaStream_t stream,
faiss/gpu/impl/Distance.cu:        GpuResources* res,
faiss/gpu/impl/Distance.cu:        cudaStream_t stream,
faiss/gpu/impl/Distance.cu:} // namespace gpu
faiss/gpu/impl/RaftIVFFlat.cuh: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/impl/RaftIVFFlat.cuh:#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
faiss/gpu/impl/RaftIVFFlat.cuh:#include <faiss/gpu/impl/IVFFlat.cuh>
faiss/gpu/impl/RaftIVFFlat.cuh:namespace gpu {
faiss/gpu/impl/RaftIVFFlat.cuh:            GpuResources* resources,
faiss/gpu/impl/RaftIVFFlat.cuh:    /// Reserve GPU memory in our inverted lists for this number of vectors
faiss/gpu/impl/RaftIVFFlat.cuh:    /// (GpuIndexIVF::search_preassigned implementation)
faiss/gpu/impl/RaftIVFFlat.cuh:    std::vector<uint8_t> getListVectorData(idx_t listId, bool gpuFormat)
faiss/gpu/impl/RaftIVFFlat.cuh:    /// or GPU quantizer
faiss/gpu/impl/RaftIVFFlat.cuh:    size_t getGpuVectorsEncodingSize_(idx_t numVecs) const override;
faiss/gpu/impl/RaftIVFFlat.cuh:} // namespace gpu
faiss/gpu/impl/InterleavedCodes.h:// Utilities for bit packing and unpacking CPU non-interleaved and GPU
faiss/gpu/impl/InterleavedCodes.h:namespace gpu {
faiss/gpu/impl/InterleavedCodes.h:} // namespace gpu
faiss/gpu/impl/PQScanMultiPassPrecomputed.cuh:#include <faiss/gpu/GpuIndicesOptions.h>
faiss/gpu/impl/PQScanMultiPassPrecomputed.cuh:#include <faiss/gpu/utils/DeviceVector.cuh>
faiss/gpu/impl/PQScanMultiPassPrecomputed.cuh:#include <faiss/gpu/utils/NoTypeTensor.cuh>
faiss/gpu/impl/PQScanMultiPassPrecomputed.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/PQScanMultiPassPrecomputed.cuh:namespace gpu {
faiss/gpu/impl/PQScanMultiPassPrecomputed.cuh:class GpuResources;
faiss/gpu/impl/PQScanMultiPassPrecomputed.cuh:        GpuResources* res);
faiss/gpu/impl/PQScanMultiPassPrecomputed.cuh:} // namespace gpu
faiss/gpu/impl/Distance.cuh:#include <faiss/gpu/impl/GeneralDistance.cuh>
faiss/gpu/impl/Distance.cuh:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/impl/Distance.cuh:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/impl/Distance.cuh:namespace gpu {
faiss/gpu/impl/Distance.cuh:class GpuResources;
faiss/gpu/impl/Distance.cuh:/// FIXME: the output distances must fit in GPU memory
faiss/gpu/impl/Distance.cuh:        GpuResources* res,
faiss/gpu/impl/Distance.cuh:        cudaStream_t stream,
faiss/gpu/impl/Distance.cuh:        GpuResources* res,
faiss/gpu/impl/Distance.cuh:        cudaStream_t stream,
faiss/gpu/impl/Distance.cuh:        GpuResources* res,
faiss/gpu/impl/Distance.cuh:        cudaStream_t stream,
faiss/gpu/impl/Distance.cuh:        GpuResources* res,
faiss/gpu/impl/Distance.cuh:        cudaStream_t stream,
faiss/gpu/impl/Distance.cuh:        GpuResources* resources,
faiss/gpu/impl/Distance.cuh:        cudaStream_t stream,
faiss/gpu/impl/Distance.cuh:        GpuResources* resources,
faiss/gpu/impl/Distance.cuh:        cudaStream_t stream,
faiss/gpu/impl/Distance.cuh:        GpuResources* resources,
faiss/gpu/impl/Distance.cuh:        cudaStream_t stream,
faiss/gpu/impl/Distance.cuh:        GpuResources* resources,
faiss/gpu/impl/Distance.cuh:        cudaStream_t stream,
faiss/gpu/impl/Distance.cuh:        GpuResources* resources,
faiss/gpu/impl/Distance.cuh:        cudaStream_t stream,
faiss/gpu/impl/Distance.cuh:        GpuResources* resources,
faiss/gpu/impl/Distance.cuh:        cudaStream_t stream,
faiss/gpu/impl/Distance.cuh:} // namespace gpu
faiss/gpu/impl/RaftIVFPQ.cuh: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/impl/RaftIVFPQ.cuh:#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
faiss/gpu/impl/RaftIVFPQ.cuh:#include <faiss/gpu/impl/IVFPQ.cuh>
faiss/gpu/impl/RaftIVFPQ.cuh:namespace gpu {
faiss/gpu/impl/RaftIVFPQ.cuh:/// Implementing class for IVFPQ on the GPU
faiss/gpu/impl/RaftIVFPQ.cuh:            GpuResources* resources,
faiss/gpu/impl/RaftIVFPQ.cuh:    /// Reserve GPU memory in our inverted lists for this number of vectors
faiss/gpu/impl/RaftIVFPQ.cuh:    /// (GpuIndexIVF::search_preassigned implementation)
faiss/gpu/impl/RaftIVFPQ.cuh:    std::vector<uint8_t> getListVectorData(idx_t listId, bool gpuFormat)
faiss/gpu/impl/RaftIVFPQ.cuh:    /// or GPU quantizer
faiss/gpu/impl/RaftIVFPQ.cuh:    size_t getGpuListEncodingSize_(idx_t listId);
faiss/gpu/impl/RaftIVFPQ.cuh:} // namespace gpu
faiss/gpu/impl/BroadcastSum.cuh:#include <cuda_fp16.h>
faiss/gpu/impl/BroadcastSum.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/BroadcastSum.cuh:namespace gpu {
faiss/gpu/impl/BroadcastSum.cuh:        cudaStream_t stream);
faiss/gpu/impl/BroadcastSum.cuh:        cudaStream_t stream);
faiss/gpu/impl/BroadcastSum.cuh:        cudaStream_t stream);
faiss/gpu/impl/BroadcastSum.cuh:        cudaStream_t stream);
faiss/gpu/impl/BroadcastSum.cuh:        cudaStream_t stream);
faiss/gpu/impl/BroadcastSum.cuh:        cudaStream_t stream);
faiss/gpu/impl/BroadcastSum.cuh:} // namespace gpu
faiss/gpu/impl/IVFFlat.cuh:#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
faiss/gpu/impl/IVFFlat.cuh:#include <faiss/gpu/impl/IVFBase.cuh>
faiss/gpu/impl/IVFFlat.cuh:namespace gpu {
faiss/gpu/impl/IVFFlat.cuh:    IVFFlat(GpuResources* resources,
faiss/gpu/impl/IVFFlat.cuh:    /// (GpuIndexIVF::search_preassigned implementation)
faiss/gpu/impl/IVFFlat.cuh:    size_t getGpuVectorsEncodingSize_(idx_t numVecs) const override;
faiss/gpu/impl/IVFFlat.cuh:    /// Translate to our preferred GPU encoding
faiss/gpu/impl/IVFFlat.cuh:    virtual std::vector<uint8_t> translateCodesToGpu_(
faiss/gpu/impl/IVFFlat.cuh:    /// Translate from our preferred GPU encoding
faiss/gpu/impl/IVFFlat.cuh:    virtual std::vector<uint8_t> translateCodesFromGpu_(
faiss/gpu/impl/IVFFlat.cuh:            cudaStream_t stream) override;
faiss/gpu/impl/IVFFlat.cuh:    std::unique_ptr<GpuScalarQuantizer> scalarQ_;
faiss/gpu/impl/IVFFlat.cuh:} // namespace gpu
faiss/gpu/impl/RemapIndices.cpp:#include <faiss/gpu/impl/RemapIndices.h>
faiss/gpu/impl/RemapIndices.cpp:namespace gpu {
faiss/gpu/impl/RemapIndices.cpp:} // namespace gpu
faiss/gpu/impl/scan/IVFInterleavedScanKernelTemplate.cu:#include <faiss/gpu/impl/scan/IVFInterleavedImpl.cuh>
faiss/gpu/impl/scan/IVFInterleavedScanKernelTemplate.cu:namespace gpu {
faiss/gpu/impl/scan/IVFInterleavedScanKernelTemplate.cu:        GpuScalarQuantizer* scalarQ,
faiss/gpu/impl/scan/IVFInterleavedScanKernelTemplate.cu:        GpuResources* res) {
faiss/gpu/impl/scan/IVFInterleavedScanKernelTemplate.cu:} // namespace gpu
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:#include <faiss/gpu/impl/IVFInterleaved.cuh>
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:#include <faiss/gpu/utils/DeviceVector.cuh>
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:namespace gpu {
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:        GpuScalarQuantizer* scalarQ,
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:        GpuResources* res);
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:        GpuScalarQuantizer* scalarQ,
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:        GpuResources* res) {
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:                        scalarQ->gpuTrained.data(),
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:                        scalarQ->gpuTrained.data() + dim);
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:                        scalarQ->gpuTrained.data(),
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:                        scalarQ->gpuTrained.data() + dim);
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:                        scalarQ->gpuTrained.data(),
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:                        scalarQ->gpuTrained.data() + dim);
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:        GpuScalarQuantizer* scalarQ,
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:        GpuResources* res) {
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:    CUDA_TEST_ERROR();
faiss/gpu/impl/scan/IVFInterleavedImpl.cuh:} // namespace gpu
faiss/gpu/impl/IVFUtils.cuh:#include <faiss/gpu/GpuIndicesOptions.h>
faiss/gpu/impl/IVFUtils.cuh:#include <faiss/gpu/utils/DeviceVector.cuh>
faiss/gpu/impl/IVFUtils.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/IVFUtils.cuh:namespace gpu {
faiss/gpu/impl/IVFUtils.cuh:class GpuResources;
faiss/gpu/impl/IVFUtils.cuh:        GpuResources* res,
faiss/gpu/impl/IVFUtils.cuh:        cudaStream_t stream);
faiss/gpu/impl/IVFUtils.cuh:        cudaStream_t stream);
faiss/gpu/impl/IVFUtils.cuh:        cudaStream_t stream);
faiss/gpu/impl/IVFUtils.cuh:} // namespace gpu
faiss/gpu/impl/RaftIVFPQ.cu: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/impl/RaftIVFPQ.cu:#include <faiss/gpu/GpuIndexFlat.h>
faiss/gpu/impl/RaftIVFPQ.cu:#include <faiss/gpu/utils/RaftUtils.h>
faiss/gpu/impl/RaftIVFPQ.cu:#include <faiss/gpu/impl/FlatIndex.cuh>
faiss/gpu/impl/RaftIVFPQ.cu:#include <faiss/gpu/impl/RaftIVFPQ.cuh>
faiss/gpu/impl/RaftIVFPQ.cu:#include <faiss/gpu/utils/Transpose.cuh>
faiss/gpu/impl/RaftIVFPQ.cu:namespace gpu {
faiss/gpu/impl/RaftIVFPQ.cu:        GpuResources* resources,
faiss/gpu/impl/RaftIVFPQ.cu:    // If the index instance is a GpuIndexFlat, then we can use direct access to
faiss/gpu/impl/RaftIVFPQ.cu:    auto gpuQ = dynamic_cast<GpuIndexFlat*>(quantizer);
faiss/gpu/impl/RaftIVFPQ.cu:    if (gpuQ) {
faiss/gpu/impl/RaftIVFPQ.cu:        auto gpuData = gpuQ->getGpuData();
faiss/gpu/impl/RaftIVFPQ.cu:        if (gpuData->getUseFloat16()) {
faiss/gpu/impl/RaftIVFPQ.cu:            gpuData->reconstruct(0, gpuData->getSize(), centroids);
faiss/gpu/impl/RaftIVFPQ.cu:            auto centroids = gpuData->getVectorsFloat32Ref();
faiss/gpu/impl/RaftIVFPQ.cu:        // them to the GPU, in order to have access as needed for residual
faiss/gpu/impl/RaftIVFPQ.cu:/// (GpuIndexIVF::search_preassigned implementation)
faiss/gpu/impl/RaftIVFPQ.cu:size_t RaftIVFPQ::getGpuListEncodingSize_(idx_t listId) {
faiss/gpu/impl/RaftIVFPQ.cu:std::vector<uint8_t> RaftIVFPQ::getListVectorData(idx_t listId, bool gpuFormat)
faiss/gpu/impl/RaftIVFPQ.cu:    if (gpuFormat) {
faiss/gpu/impl/RaftIVFPQ.cu:                "gpuFormat should be false for RAFT indices. Unpacked codes are flat.");
faiss/gpu/impl/RaftIVFPQ.cu:    // Device is already set in GpuIndex::search
faiss/gpu/impl/RaftIVFPQ.cu:    pams.lut_dtype = useFloat16LookupTables_ ? CUDA_R_16F : CUDA_R_32F;
faiss/gpu/impl/RaftIVFPQ.cu:        // GPU index can only support max int entries per list
faiss/gpu/impl/RaftIVFPQ.cu:                "GPU inverted list can only support "
faiss/gpu/impl/RaftIVFPQ.cu:    // The GPU might have a different layout of the memory
faiss/gpu/impl/RaftIVFPQ.cu:    auto gpuListSizeInBytes = getGpuListEncodingSize_(listId);
faiss/gpu/impl/RaftIVFPQ.cu:    // We only have int32 length representations on the GPU per each
faiss/gpu/impl/RaftIVFPQ.cu:    FAISS_ASSERT(gpuListSizeInBytes <= (size_t)std::numeric_limits<int>::max());
faiss/gpu/impl/RaftIVFPQ.cu:} // namespace gpu
faiss/gpu/impl/L2Norm.cuh:#include <cuda_fp16.h>
faiss/gpu/impl/L2Norm.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/L2Norm.cuh:namespace gpu {
faiss/gpu/impl/L2Norm.cuh:        cudaStream_t stream);
faiss/gpu/impl/L2Norm.cuh:        cudaStream_t stream);
faiss/gpu/impl/L2Norm.cuh:} // namespace gpu
faiss/gpu/impl/IVFUtilsSelect2.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/IVFUtilsSelect2.cu:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/impl/IVFUtilsSelect2.cu:#include <faiss/gpu/impl/IVFUtils.cuh>
faiss/gpu/impl/IVFUtilsSelect2.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/impl/IVFUtilsSelect2.cu:#include <faiss/gpu/utils/Limits.cuh>
faiss/gpu/impl/IVFUtilsSelect2.cu:#include <faiss/gpu/utils/Select.cuh>
faiss/gpu/impl/IVFUtilsSelect2.cu:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/IVFUtilsSelect2.cu:namespace gpu {
faiss/gpu/impl/IVFUtilsSelect2.cu:        cudaStream_t stream) {
faiss/gpu/impl/IVFUtilsSelect2.cu:    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);
faiss/gpu/impl/IVFUtilsSelect2.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/impl/IVFUtilsSelect2.cu:#endif // GPU_MAX_SELECTION_K
faiss/gpu/impl/IVFUtilsSelect2.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/IVFUtilsSelect2.cu:} // namespace gpu
faiss/gpu/impl/IVFBase.cuh:#include <faiss/gpu/GpuIndicesOptions.h>
faiss/gpu/impl/IVFBase.cuh:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/impl/IVFBase.cuh:#include <faiss/gpu/utils/DeviceVector.cuh>
faiss/gpu/impl/IVFBase.cuh:namespace gpu {
faiss/gpu/impl/IVFBase.cuh:class GpuResources;
faiss/gpu/impl/IVFBase.cuh:    IVFBase(GpuResources* resources,
faiss/gpu/impl/IVFBase.cuh:    /// Reserve GPU memory in our inverted lists for this number of vectors
faiss/gpu/impl/IVFBase.cuh:    virtual std::vector<uint8_t> getListVectorData(idx_t listId, bool gpuFormat)
faiss/gpu/impl/IVFBase.cuh:    /// or GPU quantizer
faiss/gpu/impl/IVFBase.cuh:    /// (GpuIndexIVF::search_preassigned implementation)
faiss/gpu/impl/IVFBase.cuh:    /// Performs search in a CPU or GPU coarse quantizer for IVF cells,
faiss/gpu/impl/IVFBase.cuh:    virtual size_t getGpuVectorsEncodingSize_(idx_t numVecs) const = 0;
faiss/gpu/impl/IVFBase.cuh:    /// Translate to our preferred GPU encoding
faiss/gpu/impl/IVFBase.cuh:    virtual std::vector<uint8_t> translateCodesToGpu_(
faiss/gpu/impl/IVFBase.cuh:    /// Translate from our preferred GPU encoding
faiss/gpu/impl/IVFBase.cuh:    virtual std::vector<uint8_t> translateCodesFromGpu_(
faiss/gpu/impl/IVFBase.cuh:            cudaStream_t stream) = 0;
faiss/gpu/impl/IVFBase.cuh:    void updateDeviceListInfo_(cudaStream_t stream);
faiss/gpu/impl/IVFBase.cuh:            cudaStream_t stream);
faiss/gpu/impl/IVFBase.cuh:    /// Shared function to copy indices from CPU to GPU
faiss/gpu/impl/IVFBase.cuh:    /// Collection of GPU resources that we use
faiss/gpu/impl/IVFBase.cuh:    GpuResources* resources_;
faiss/gpu/impl/IVFBase.cuh:    /// Coarse quantizer centroids available on GPU
faiss/gpu/impl/IVFBase.cuh:    /// How are user indices stored on the GPU?
faiss/gpu/impl/IVFBase.cuh:        DeviceIVFList(GpuResources* res, const AllocInfo& info);
faiss/gpu/impl/IVFBase.cuh:} // namespace gpu
faiss/gpu/impl/IVFFlatScan.cuh:#include <faiss/gpu/GpuIndicesOptions.h>
faiss/gpu/impl/IVFFlatScan.cuh:#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
faiss/gpu/impl/IVFFlatScan.cuh:#include <faiss/gpu/utils/DeviceVector.cuh>
faiss/gpu/impl/IVFFlatScan.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/IVFFlatScan.cuh:namespace gpu {
faiss/gpu/impl/IVFFlatScan.cuh:class GpuResources;
faiss/gpu/impl/IVFFlatScan.cuh:        GpuScalarQuantizer* scalarQ,
faiss/gpu/impl/IVFFlatScan.cuh:        GpuResources* res);
faiss/gpu/impl/IVFFlatScan.cuh:} // namespace gpu
faiss/gpu/impl/PQScanMultiPassNoPrecomputed.cuh:#include <faiss/gpu/GpuIndicesOptions.h>
faiss/gpu/impl/PQScanMultiPassNoPrecomputed.cuh:#include <faiss/gpu/utils/DeviceVector.cuh>
faiss/gpu/impl/PQScanMultiPassNoPrecomputed.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/PQScanMultiPassNoPrecomputed.cuh:namespace gpu {
faiss/gpu/impl/PQScanMultiPassNoPrecomputed.cuh:class GpuResources;
faiss/gpu/impl/PQScanMultiPassNoPrecomputed.cuh:        GpuResources* res);
faiss/gpu/impl/PQScanMultiPassNoPrecomputed.cuh:} // namespace gpu
faiss/gpu/impl/PQScanMultiPassNoPrecomputed.cuh:#include <faiss/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh>
faiss/gpu/impl/GeneralDistance.cuh:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/GeneralDistance.cuh:#include <faiss/gpu/impl/DistanceUtils.cuh>
faiss/gpu/impl/GeneralDistance.cuh:#include <faiss/gpu/utils/BlockSelectKernel.cuh>
faiss/gpu/impl/GeneralDistance.cuh:#include <faiss/gpu/utils/ConversionOperators.cuh>
faiss/gpu/impl/GeneralDistance.cuh:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/impl/GeneralDistance.cuh:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/impl/GeneralDistance.cuh:#include <faiss/gpu/utils/Select.cuh>
faiss/gpu/impl/GeneralDistance.cuh:namespace gpu {
faiss/gpu/impl/GeneralDistance.cuh:        cudaStream_t stream) {
faiss/gpu/impl/GeneralDistance.cuh:        GpuResources* res,
faiss/gpu/impl/GeneralDistance.cuh:        cudaStream_t stream,
faiss/gpu/impl/GeneralDistance.cuh:                thrust::cuda::par.on(stream),
faiss/gpu/impl/GeneralDistance.cuh:                thrust::cuda::par.on(stream),
faiss/gpu/impl/GeneralDistance.cuh:    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K); // select limitation
faiss/gpu/impl/GeneralDistance.cuh:    CUDA_TEST_ERROR();
faiss/gpu/impl/GeneralDistance.cuh:} // namespace gpu
faiss/gpu/impl/BinaryDistance.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/BinaryDistance.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/impl/BinaryDistance.cu:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/impl/BinaryDistance.cu:#include <faiss/gpu/utils/Select.cuh>
faiss/gpu/impl/BinaryDistance.cu:namespace gpu {
faiss/gpu/impl/BinaryDistance.cu:        cudaStream_t stream) {
faiss/gpu/impl/BinaryDistance.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/impl/BinaryDistance.cu:        cudaStream_t stream) {
faiss/gpu/impl/BinaryDistance.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/impl/BinaryDistance.cu:        cudaStream_t stream) {
faiss/gpu/impl/BinaryDistance.cu:    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);
faiss/gpu/impl/BinaryDistance.cu:} // namespace gpu
faiss/gpu/impl/RaftFlatIndex.cu: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/impl/RaftFlatIndex.cu:#include <faiss/gpu/utils/RaftUtils.h>
faiss/gpu/impl/RaftFlatIndex.cu:#include <faiss/gpu/impl/RaftFlatIndex.cuh>
faiss/gpu/impl/RaftFlatIndex.cu:#include <faiss/gpu/utils/ConversionOperators.cuh>
faiss/gpu/impl/RaftFlatIndex.cu:namespace gpu {
faiss/gpu/impl/RaftFlatIndex.cu:        GpuResources* res,
faiss/gpu/impl/RaftFlatIndex.cu:} // namespace gpu
faiss/gpu/impl/PQCodeDistances.cuh:#include <faiss/gpu/utils/NoTypeTensor.cuh>
faiss/gpu/impl/PQCodeDistances.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/PQCodeDistances.cuh:namespace gpu {
faiss/gpu/impl/PQCodeDistances.cuh:        GpuResources* res,
faiss/gpu/impl/PQCodeDistances.cuh:} // namespace gpu
faiss/gpu/impl/PQCodeDistances.cuh:#include <faiss/gpu/impl/PQCodeDistances-inl.cuh>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/GpuIndex.h>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/GpuResources.h>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/impl/InterleavedCodes.h>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/impl/RemapIndices.h>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/impl/BroadcastSum.cuh>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/impl/Distance.cuh>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/impl/FlatIndex.cuh>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/impl/IVFAppend.cuh>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/impl/IVFPQ.cuh>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/impl/L2Norm.cuh>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/impl/PQCodeDistances.cuh>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/impl/PQScanMultiPassNoPrecomputed.cuh>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/impl/PQScanMultiPassPrecomputed.cuh>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/impl/VectorResidual.cuh>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/utils/ConversionOperators.cuh>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/utils/HostTensor.cuh>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/utils/MatrixMult.cuh>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/utils/NoTypeTensor.cuh>
faiss/gpu/impl/IVFPQ.cu:#include <faiss/gpu/utils/Transpose.cuh>
faiss/gpu/impl/IVFPQ.cu:namespace gpu {
faiss/gpu/impl/IVFPQ.cu:        GpuResources* resources,
faiss/gpu/impl/IVFPQ.cu:                "Precomputed codes are not needed for GpuIndexIVFPQ "
faiss/gpu/impl/IVFPQ.cu:        cudaStream_t stream) {
faiss/gpu/impl/IVFPQ.cu:size_t IVFPQ::getGpuVectorsEncodingSize_(idx_t numVecs) const {
faiss/gpu/impl/IVFPQ.cu:// Convert the CPU layout to the GPU layout
faiss/gpu/impl/IVFPQ.cu:std::vector<uint8_t> IVFPQ::translateCodesToGpu_(
faiss/gpu/impl/IVFPQ.cu:// Conver the GPU layout to the CPU layout
faiss/gpu/impl/IVFPQ.cu:std::vector<uint8_t> IVFPQ::translateCodesFromGpu_(
faiss/gpu/impl/IVFPQ.cu:    // Whether or not there is a CPU or GPU coarse quantizer, updateQuantizer()
faiss/gpu/impl/IVFPQ.cu:    // have the data available on the GPU
faiss/gpu/impl/IVFPQ.cu:            "not synchronized on GPU; must call updateQuantizer() "
faiss/gpu/impl/IVFPQ.cu:    FAISS_ASSERT(nprobe <= GPU_MAX_SELECTION_K);
faiss/gpu/impl/IVFPQ.cu:    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);
faiss/gpu/impl/IVFPQ.cu:    // If the GPU isn't storing indices (they are on the CPU side), we
faiss/gpu/impl/IVFPQ.cu:        // Copy back to GPU, since the input to this function is on the
faiss/gpu/impl/IVFPQ.cu:        // GPU
faiss/gpu/impl/IVFPQ.cu:} // namespace gpu
faiss/gpu/impl/InterleavedCodes.cpp:#include <faiss/gpu/impl/InterleavedCodes.h>
faiss/gpu/impl/InterleavedCodes.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/InterleavedCodes.cpp:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/impl/InterleavedCodes.cpp:namespace gpu {
faiss/gpu/impl/InterleavedCodes.cpp:} // namespace gpu
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:#include <faiss/gpu/GpuResources.h>
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:#include <faiss/gpu/impl/IVFUtils.cuh>
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:#include <faiss/gpu/impl/PQCodeLoad.cuh>
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:#include <faiss/gpu/impl/PQScanMultiPassPrecomputed.cuh>
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:#include <faiss/gpu/utils/ConversionOperators.cuh>
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:#include <faiss/gpu/utils/LoadStoreOperators.cuh>
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:#include <faiss/gpu/utils/MathOperators.cuh>
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:#include <faiss/gpu/utils/WarpPackedBits.cuh>
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:namespace gpu {
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:    // FIXME: some issue with getLaneId() and CUDA 10.1 and P4 GPUs?
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:        GpuResources* res,
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:        cudaStream_t stream) {
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:        CUDA_TEST_ERROR();
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:        GpuResources* res) {
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:    CUDA_VERIFY(cudaMemsetAsync(
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:    CUDA_VERIFY(cudaMemsetAsync(
faiss/gpu/impl/PQScanMultiPassPrecomputed.cu:} // namespace gpu
faiss/gpu/impl/IVFBase.cu:#include <faiss/gpu/GpuIndex.h>
faiss/gpu/impl/IVFBase.cu:#include <faiss/gpu/GpuIndexFlat.h>
faiss/gpu/impl/IVFBase.cu:#include <faiss/gpu/GpuResources.h>
faiss/gpu/impl/IVFBase.cu:#include <faiss/gpu/impl/RemapIndices.h>
faiss/gpu/impl/IVFBase.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/IVFBase.cu:#include <faiss/gpu/impl/FlatIndex.cuh>
faiss/gpu/impl/IVFBase.cu:#include <faiss/gpu/impl/IVFAppend.cuh>
faiss/gpu/impl/IVFBase.cu:#include <faiss/gpu/impl/IVFBase.cuh>
faiss/gpu/impl/IVFBase.cu:#include <faiss/gpu/utils/CopyUtils.cuh>
faiss/gpu/impl/IVFBase.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/impl/IVFBase.cu:#include <faiss/gpu/utils/HostTensor.cuh>
faiss/gpu/impl/IVFBase.cu:#include <faiss/gpu/utils/ThrustUtils.cuh>
faiss/gpu/impl/IVFBase.cu:namespace gpu {
faiss/gpu/impl/IVFBase.cu:IVFBase::DeviceIVFList::DeviceIVFList(GpuResources* res, const AllocInfo& info)
faiss/gpu/impl/IVFBase.cu:        GpuResources* resources,
faiss/gpu/impl/IVFBase.cu:    auto bytesPerDataList = getGpuVectorsEncodingSize_(vecsPerList);
faiss/gpu/impl/IVFBase.cu:void IVFBase::updateDeviceListInfo_(cudaStream_t stream) {
faiss/gpu/impl/IVFBase.cu:        cudaStream_t stream) {
faiss/gpu/impl/IVFBase.cu:    // Copy the above update sets to the GPU
faiss/gpu/impl/IVFBase.cu:        // The data is stored as int32 on the GPU
faiss/gpu/impl/IVFBase.cu:        // The data is stored as int64 on the GPU
faiss/gpu/impl/IVFBase.cu:        // The data is not stored on the GPU
faiss/gpu/impl/IVFBase.cu:        // encoded on the GPU
faiss/gpu/impl/IVFBase.cu:std::vector<uint8_t> IVFBase::getListVectorData(idx_t listId, bool gpuFormat)
faiss/gpu/impl/IVFBase.cu:    auto gpuCodes = list->data.copyToHost<uint8_t>(stream);
faiss/gpu/impl/IVFBase.cu:    if (gpuFormat) {
faiss/gpu/impl/IVFBase.cu:        return gpuCodes;
faiss/gpu/impl/IVFBase.cu:        // The GPU layout may be different than the CPU layout (e.g., vectors
faiss/gpu/impl/IVFBase.cu:        return translateCodesFromGpu_(std::move(gpuCodes), list->numVecs);
faiss/gpu/impl/IVFBase.cu:    // The GPU might have a different layout of the memory
faiss/gpu/impl/IVFBase.cu:    auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(numVecs);
faiss/gpu/impl/IVFBase.cu:    auto translatedCodes = translateCodesToGpu_(std::move(codesV), numVecs);
faiss/gpu/impl/IVFBase.cu:            gpuListSizeInBytes,
faiss/gpu/impl/IVFBase.cu:    // If the index instance is a GpuIndexFlat, then we can use direct access to
faiss/gpu/impl/IVFBase.cu:    auto gpuQ = dynamic_cast<GpuIndexFlat*>(quantizer);
faiss/gpu/impl/IVFBase.cu:    if (gpuQ) {
faiss/gpu/impl/IVFBase.cu:        auto gpuData = gpuQ->getGpuData();
faiss/gpu/impl/IVFBase.cu:        if (gpuData->getUseFloat16()) {
faiss/gpu/impl/IVFBase.cu:            gpuData->reconstruct(0, gpuData->getSize(), centroids);
faiss/gpu/impl/IVFBase.cu:            auto ref32 = gpuData->getVectorsFloat32Ref();
faiss/gpu/impl/IVFBase.cu:        // them to the GPU, in order to have access as needed for residual
faiss/gpu/impl/IVFBase.cu:    // The provided IVF quantizer may be CPU or GPU resident.
faiss/gpu/impl/IVFBase.cu:    // If GPU resident, we can simply call it passing the above output device
faiss/gpu/impl/IVFBase.cu:    auto gpuQuantizer = tryCastGpuIndex(coarseQuantizer);
faiss/gpu/impl/IVFBase.cu:    if (gpuQuantizer) {
faiss/gpu/impl/IVFBase.cu:        gpuQuantizer->search(
faiss/gpu/impl/IVFBase.cu:            gpuQuantizer->compute_residual_n(
faiss/gpu/impl/IVFBase.cu:            gpuQuantizer->reconstruct_batch(
faiss/gpu/impl/IVFBase.cu:    // the GPUs, which means that they might need reallocation, which
faiss/gpu/impl/IVFBase.cu:            auto newSizeBytes = getGpuVectorsEncodingSize_(newNumVecs);
faiss/gpu/impl/IVFBase.cu:                // indices are not stored on the GPU or CPU side
faiss/gpu/impl/IVFBase.cu:    // Copy the offsets to the GPU
faiss/gpu/impl/IVFBase.cu:} // namespace gpu
faiss/gpu/impl/L2Select.cu:#include <faiss/gpu/impl/L2Select.cuh>
faiss/gpu/impl/L2Select.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/L2Select.cu:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/impl/L2Select.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/impl/L2Select.cu:#include <faiss/gpu/utils/MathOperators.cuh>
faiss/gpu/impl/L2Select.cu:#include <faiss/gpu/utils/Pair.cuh>
faiss/gpu/impl/L2Select.cu:#include <faiss/gpu/utils/Reductions.cuh>
faiss/gpu/impl/L2Select.cu:#include <faiss/gpu/utils/Select.cuh>
faiss/gpu/impl/L2Select.cu:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/impl/L2Select.cu:namespace gpu {
faiss/gpu/impl/L2Select.cu:        cudaStream_t stream) {
faiss/gpu/impl/L2Select.cu:    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);
faiss/gpu/impl/L2Select.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/impl/L2Select.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/L2Select.cu:        cudaStream_t stream) {
faiss/gpu/impl/L2Select.cu:} // namespace gpu
faiss/gpu/impl/IVFPQ.cuh:#include <faiss/gpu/impl/IVFBase.cuh>
faiss/gpu/impl/IVFPQ.cuh:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/impl/IVFPQ.cuh:namespace gpu {
faiss/gpu/impl/IVFPQ.cuh:/// Implementing class for IVFPQ on the GPU
faiss/gpu/impl/IVFPQ.cuh:    IVFPQ(GpuResources* resources,
faiss/gpu/impl/IVFPQ.cuh:    /// (GpuIndexIVF::search_preassigned implementation)
faiss/gpu/impl/IVFPQ.cuh:    size_t getGpuVectorsEncodingSize_(idx_t numVecs) const override;
faiss/gpu/impl/IVFPQ.cuh:    /// Translate to our preferred GPU encoding
faiss/gpu/impl/IVFPQ.cuh:    std::vector<uint8_t> translateCodesToGpu_(
faiss/gpu/impl/IVFPQ.cuh:    /// Translate from our preferred GPU encoding
faiss/gpu/impl/IVFPQ.cuh:    std::vector<uint8_t> translateCodesFromGpu_(
faiss/gpu/impl/IVFPQ.cuh:            cudaStream_t stream) override;
faiss/gpu/impl/IVFPQ.cuh:    /// On the GPU, we prefer different PQ centroid data layouts for
faiss/gpu/impl/IVFPQ.cuh:} // namespace gpu
faiss/gpu/impl/IVFFlat.cu:#include <faiss/gpu/GpuIndex.h>
faiss/gpu/impl/IVFFlat.cu:#include <faiss/gpu/GpuResources.h>
faiss/gpu/impl/IVFFlat.cu:#include <faiss/gpu/impl/InterleavedCodes.h>
faiss/gpu/impl/IVFFlat.cu:#include <faiss/gpu/impl/RemapIndices.h>
faiss/gpu/impl/IVFFlat.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/IVFFlat.cu:#include <faiss/gpu/impl/FlatIndex.cuh>
faiss/gpu/impl/IVFFlat.cu:#include <faiss/gpu/impl/IVFAppend.cuh>
faiss/gpu/impl/IVFFlat.cu:#include <faiss/gpu/impl/IVFFlat.cuh>
faiss/gpu/impl/IVFFlat.cu:#include <faiss/gpu/impl/IVFFlatScan.cuh>
faiss/gpu/impl/IVFFlat.cu:#include <faiss/gpu/impl/IVFInterleaved.cuh>
faiss/gpu/impl/IVFFlat.cu:#include <faiss/gpu/utils/ConversionOperators.cuh>
faiss/gpu/impl/IVFFlat.cu:#include <faiss/gpu/utils/CopyUtils.cuh>
faiss/gpu/impl/IVFFlat.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/impl/IVFFlat.cu:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/impl/IVFFlat.cu:#include <faiss/gpu/utils/HostTensor.cuh>
faiss/gpu/impl/IVFFlat.cu:#include <faiss/gpu/utils/Transpose.cuh>
faiss/gpu/impl/IVFFlat.cu:namespace gpu {
faiss/gpu/impl/IVFFlat.cu:        GpuResources* res,
faiss/gpu/impl/IVFFlat.cu:          scalarQ_(scalarQ ? new GpuScalarQuantizer(res, *scalarQ) : nullptr) {}
faiss/gpu/impl/IVFFlat.cu:size_t IVFFlat::getGpuVectorsEncodingSize_(idx_t numVecs) const {
faiss/gpu/impl/IVFFlat.cu:std::vector<uint8_t> IVFFlat::translateCodesToGpu_(
faiss/gpu/impl/IVFFlat.cu:std::vector<uint8_t> IVFFlat::translateCodesFromGpu_(
faiss/gpu/impl/IVFFlat.cu:        cudaStream_t stream) {
faiss/gpu/impl/IVFFlat.cu:    FAISS_ASSERT(nprobe <= GPU_MAX_SELECTION_K);
faiss/gpu/impl/IVFFlat.cu:    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);
faiss/gpu/impl/IVFFlat.cu:    auto gpuQuantizer = tryCastGpuIndex(coarseQuantizer);
faiss/gpu/impl/IVFFlat.cu:    if (gpuQuantizer) {
faiss/gpu/impl/IVFFlat.cu:        gpuQuantizer->reconstruct_batch(
faiss/gpu/impl/IVFFlat.cu:    // If the GPU isn't storing indices (they are on the CPU side), we
faiss/gpu/impl/IVFFlat.cu:        // Copy back to GPU, since the input to this function is on the
faiss/gpu/impl/IVFFlat.cu:        // GPU
faiss/gpu/impl/IVFFlat.cu:} // namespace gpu
faiss/gpu/impl/IVFFlatScan.cu:#include <faiss/gpu/GpuResources.h>
faiss/gpu/impl/IVFFlatScan.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/IVFFlatScan.cu:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/impl/IVFFlatScan.cu:#include <faiss/gpu/impl/DistanceUtils.cuh>
faiss/gpu/impl/IVFFlatScan.cu:#include <faiss/gpu/impl/IVFFlatScan.cuh>
faiss/gpu/impl/IVFFlatScan.cu:#include <faiss/gpu/impl/IVFUtils.cuh>
faiss/gpu/impl/IVFFlatScan.cu:#include <faiss/gpu/utils/Comparators.cuh>
faiss/gpu/impl/IVFFlatScan.cu:#include <faiss/gpu/utils/ConversionOperators.cuh>
faiss/gpu/impl/IVFFlatScan.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/impl/IVFFlatScan.cu:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/impl/IVFFlatScan.cu:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/impl/IVFFlatScan.cu:#include <faiss/gpu/utils/MathOperators.cuh>
faiss/gpu/impl/IVFFlatScan.cu:#include <faiss/gpu/utils/PtxUtils.cuh>
faiss/gpu/impl/IVFFlatScan.cu:#include <faiss/gpu/utils/Reductions.cuh>
faiss/gpu/impl/IVFFlatScan.cu:namespace gpu {
faiss/gpu/impl/IVFFlatScan.cu:        GpuResources* res,
faiss/gpu/impl/IVFFlatScan.cu:        GpuScalarQuantizer* scalarQ,
faiss/gpu/impl/IVFFlatScan.cu:        cudaStream_t stream) {
faiss/gpu/impl/IVFFlatScan.cu:                        scalarQ->gpuTrained.data(),
faiss/gpu/impl/IVFFlatScan.cu:                        scalarQ->gpuTrained.data() + dim);
faiss/gpu/impl/IVFFlatScan.cu:                        scalarQ->gpuTrained.data(),
faiss/gpu/impl/IVFFlatScan.cu:                        scalarQ->gpuTrained.data() + dim);
faiss/gpu/impl/IVFFlatScan.cu:    CUDA_TEST_ERROR();
faiss/gpu/impl/IVFFlatScan.cu:        GpuScalarQuantizer* scalarQ,
faiss/gpu/impl/IVFFlatScan.cu:        GpuResources* res) {
faiss/gpu/impl/IVFFlatScan.cu:    CUDA_VERIFY(cudaMemsetAsync(
faiss/gpu/impl/IVFFlatScan.cu:    CUDA_VERIFY(cudaMemsetAsync(
faiss/gpu/impl/IVFFlatScan.cu:} // namespace gpu
faiss/gpu/impl/IndexUtils.cu:#include <faiss/gpu/impl/IndexUtils.h>
faiss/gpu/impl/IndexUtils.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/impl/IndexUtils.cu:namespace gpu {
faiss/gpu/impl/IndexUtils.cu:/// Returns the maximum k-selection value supported based on the CUDA SDK that
faiss/gpu/impl/IndexUtils.cu:/// non-CUDA files
faiss/gpu/impl/IndexUtils.cu:    return GPU_MAX_SELECTION_K;
faiss/gpu/impl/IndexUtils.cu:            "GPU index only supports min/max-K selection up to %d (requested %d)",
faiss/gpu/impl/IndexUtils.cu:            "GPU IVF index only supports nprobe selection up to %d (requested %zu)",
faiss/gpu/impl/IndexUtils.cu:} // namespace gpu
faiss/gpu/impl/GpuScalarQuantizer.cuh:#include <faiss/gpu/utils/ConversionOperators.cuh>
faiss/gpu/impl/GpuScalarQuantizer.cuh:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/impl/GpuScalarQuantizer.cuh:#include <faiss/gpu/utils/HostTensor.cuh>
faiss/gpu/impl/GpuScalarQuantizer.cuh:#include <faiss/gpu/utils/PtxUtils.cuh>
faiss/gpu/impl/GpuScalarQuantizer.cuh:#include <faiss/gpu/utils/WarpShuffles.cuh>
faiss/gpu/impl/GpuScalarQuantizer.cuh:namespace gpu {
faiss/gpu/impl/GpuScalarQuantizer.cuh:// GPU memory
faiss/gpu/impl/GpuScalarQuantizer.cuh:struct GpuScalarQuantizer : public ScalarQuantizer {
faiss/gpu/impl/GpuScalarQuantizer.cuh:    GpuScalarQuantizer(GpuResources* res, const ScalarQuantizer& sq)
faiss/gpu/impl/GpuScalarQuantizer.cuh:              gpuTrained(DeviceTensor<float, 1, true>(
faiss/gpu/impl/GpuScalarQuantizer.cuh:        gpuTrained.copyFrom(cpuTrained, stream);
faiss/gpu/impl/GpuScalarQuantizer.cuh:    // ScalarQuantizer::trained copied to GPU memory
faiss/gpu/impl/GpuScalarQuantizer.cuh:    DeviceTensor<float, 1, true> gpuTrained;
faiss/gpu/impl/GpuScalarQuantizer.cuh:} // namespace gpu
faiss/gpu/impl/IVFInterleaved.cu:#include <faiss/gpu/impl/IVFInterleaved.cuh>
faiss/gpu/impl/IVFInterleaved.cu:#include <faiss/gpu/impl/scan/IVFInterleavedImpl.cuh>
faiss/gpu/impl/IVFInterleaved.cu:namespace gpu {
faiss/gpu/impl/IVFInterleaved.cu:        // GPU_MAX_SELECTION_K.
faiss/gpu/impl/IVFInterleaved.cu:            static_assert(GPU_MAX_SELECTION_K <= 65536, "");
faiss/gpu/impl/IVFInterleaved.cu:        cudaStream_t stream) {
faiss/gpu/impl/IVFInterleaved.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/impl/IVFInterleaved.cu:        GpuScalarQuantizer* scalarQ,
faiss/gpu/impl/IVFInterleaved.cu:        GpuResources* res) {
faiss/gpu/impl/IVFInterleaved.cu:    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);
faiss/gpu/impl/IVFInterleaved.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/impl/IVFInterleaved.cu:} // namespace gpu
faiss/gpu/impl/IVFInterleaved.cuh:#include <faiss/gpu/GpuIndicesOptions.h>
faiss/gpu/impl/IVFInterleaved.cuh:#include <faiss/gpu/GpuResources.h>
faiss/gpu/impl/IVFInterleaved.cuh:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/IVFInterleaved.cuh:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/impl/IVFInterleaved.cuh:#include <faiss/gpu/impl/DistanceUtils.cuh>
faiss/gpu/impl/IVFInterleaved.cuh:#include <faiss/gpu/impl/GpuScalarQuantizer.cuh>
faiss/gpu/impl/IVFInterleaved.cuh:#include <faiss/gpu/utils/Comparators.cuh>
faiss/gpu/impl/IVFInterleaved.cuh:#include <faiss/gpu/utils/ConversionOperators.cuh>
faiss/gpu/impl/IVFInterleaved.cuh:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/impl/IVFInterleaved.cuh:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/impl/IVFInterleaved.cuh:#include <faiss/gpu/utils/DeviceVector.cuh>
faiss/gpu/impl/IVFInterleaved.cuh:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/impl/IVFInterleaved.cuh:#include <faiss/gpu/utils/MathOperators.cuh>
faiss/gpu/impl/IVFInterleaved.cuh:#include <faiss/gpu/utils/PtxUtils.cuh>
faiss/gpu/impl/IVFInterleaved.cuh:#include <faiss/gpu/utils/Select.cuh>
faiss/gpu/impl/IVFInterleaved.cuh:#include <faiss/gpu/utils/WarpPackedBits.cuh>
faiss/gpu/impl/IVFInterleaved.cuh:namespace gpu {
faiss/gpu/impl/IVFInterleaved.cuh:            // FIXME: some issue with getLaneId() and CUDA 10.1 and P4 GPUs?
faiss/gpu/impl/IVFInterleaved.cuh:        GpuScalarQuantizer* scalarQ,
faiss/gpu/impl/IVFInterleaved.cuh:        GpuResources* res);
faiss/gpu/impl/IVFInterleaved.cuh:        cudaStream_t stream);
faiss/gpu/impl/IVFInterleaved.cuh:} // namespace gpu
faiss/gpu/impl/RaftIVFFlat.cu: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/impl/RaftIVFFlat.cu:#include <faiss/gpu/GpuIndexFlat.h>
faiss/gpu/impl/RaftIVFFlat.cu:#include <faiss/gpu/utils/RaftUtils.h>
faiss/gpu/impl/RaftIVFFlat.cu:#include <faiss/gpu/impl/FlatIndex.cuh>
faiss/gpu/impl/RaftIVFFlat.cu:#include <faiss/gpu/impl/IVFFlat.cuh>
faiss/gpu/impl/RaftIVFFlat.cu:#include <faiss/gpu/impl/RaftIVFFlat.cuh>
faiss/gpu/impl/RaftIVFFlat.cu:#include <faiss/gpu/utils/Transpose.cuh>
faiss/gpu/impl/RaftIVFFlat.cu:namespace gpu {
faiss/gpu/impl/RaftIVFFlat.cu:        GpuResources* res,
faiss/gpu/impl/RaftIVFFlat.cu:    // Device is already set in GpuIndex::search
faiss/gpu/impl/RaftIVFFlat.cu:        bool gpuFormat) const {
faiss/gpu/impl/RaftIVFFlat.cu:    if (gpuFormat) {
faiss/gpu/impl/RaftIVFFlat.cu:        FAISS_THROW_MSG("gpuFormat should be false for RAFT indices");
faiss/gpu/impl/RaftIVFFlat.cu:    auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(listSize);
faiss/gpu/impl/RaftIVFFlat.cu:    std::vector<uint8_t> interleaved_codes(gpuListSizeInBytes);
faiss/gpu/impl/RaftIVFFlat.cu:            gpuListSizeInBytes,
faiss/gpu/impl/RaftIVFFlat.cu:/// (GpuIndexIVF::search_preassigned implementation)
faiss/gpu/impl/RaftIVFFlat.cu:    // If the index instance is a GpuIndexFlat, then we can use direct access to
faiss/gpu/impl/RaftIVFFlat.cu:    auto gpuQ = dynamic_cast<GpuIndexFlat*>(quantizer);
faiss/gpu/impl/RaftIVFFlat.cu:    if (gpuQ) {
faiss/gpu/impl/RaftIVFFlat.cu:        auto gpuData = gpuQ->getGpuData();
faiss/gpu/impl/RaftIVFFlat.cu:        if (gpuData->getUseFloat16()) {
faiss/gpu/impl/RaftIVFFlat.cu:            gpuData->reconstruct(0, gpuData->getSize(), centroids);
faiss/gpu/impl/RaftIVFFlat.cu:            auto centroids = gpuData->getVectorsFloat32Ref();
faiss/gpu/impl/RaftIVFFlat.cu:        // them to the GPU, in order to have access as needed for residual
faiss/gpu/impl/RaftIVFFlat.cu:        // GPU index can only support max int entries per list
faiss/gpu/impl/RaftIVFFlat.cu:                "GPU inverted list can only support "
faiss/gpu/impl/RaftIVFFlat.cu:size_t RaftIVFFlat::getGpuVectorsEncodingSize_(idx_t numVecs) const {
faiss/gpu/impl/RaftIVFFlat.cu:    // The GPU might have a different layout of the memory
faiss/gpu/impl/RaftIVFFlat.cu:    auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(numVecs);
faiss/gpu/impl/RaftIVFFlat.cu:    // We only have int32 length representations on the GPU per each
faiss/gpu/impl/RaftIVFFlat.cu:    FAISS_ASSERT(gpuListSizeInBytes <= (size_t)std::numeric_limits<int>::max());
faiss/gpu/impl/RaftIVFFlat.cu:    std::vector<uint8_t> interleaved_codes(gpuListSizeInBytes);
faiss/gpu/impl/RaftIVFFlat.cu:            gpuListSizeInBytes,
faiss/gpu/impl/RaftIVFFlat.cu:} // namespace gpu
faiss/gpu/impl/PQCodeDistances-inl.cuh:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/impl/PQCodeDistances-inl.cuh:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/impl/PQCodeDistances-inl.cuh:#include <faiss/gpu/impl/BroadcastSum.cuh>
faiss/gpu/impl/PQCodeDistances-inl.cuh:#include <faiss/gpu/impl/Distance.cuh>
faiss/gpu/impl/PQCodeDistances-inl.cuh:#include <faiss/gpu/impl/L2Norm.cuh>
faiss/gpu/impl/PQCodeDistances-inl.cuh:#include <faiss/gpu/utils/ConversionOperators.cuh>
faiss/gpu/impl/PQCodeDistances-inl.cuh:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/impl/PQCodeDistances-inl.cuh:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/impl/PQCodeDistances-inl.cuh:#include <faiss/gpu/utils/MatrixMult.cuh>
faiss/gpu/impl/PQCodeDistances-inl.cuh:#include <faiss/gpu/utils/PtxUtils.cuh>
faiss/gpu/impl/PQCodeDistances-inl.cuh:#include <faiss/gpu/utils/Transpose.cuh>
faiss/gpu/impl/PQCodeDistances-inl.cuh:namespace gpu {
faiss/gpu/impl/PQCodeDistances-inl.cuh:#if defined(USE_AMD_ROCM) && __AMDGCN_WAVEFRONT_SIZE == 64u
faiss/gpu/impl/PQCodeDistances-inl.cuh:        cudaStream_t stream) {
faiss/gpu/impl/PQCodeDistances-inl.cuh:    CUDA_TEST_ERROR();
faiss/gpu/impl/PQCodeDistances-inl.cuh:        cudaStream_t stream) {
faiss/gpu/impl/PQCodeDistances-inl.cuh:        GpuResources* res,
faiss/gpu/impl/PQCodeDistances-inl.cuh:        cudaStream_t stream) {
faiss/gpu/impl/PQCodeDistances-inl.cuh:        GpuResources* res,
faiss/gpu/impl/PQCodeDistances-inl.cuh:        cudaStream_t stream) {
faiss/gpu/impl/PQCodeDistances-inl.cuh:    CUDA_TEST_ERROR();
faiss/gpu/impl/PQCodeDistances-inl.cuh:} // namespace gpu
faiss/gpu/GpuClonerOptions.h:#include <faiss/gpu/GpuIndicesOptions.h>
faiss/gpu/GpuClonerOptions.h:namespace gpu {
faiss/gpu/GpuClonerOptions.h:/// set some options on how to copy to GPU
faiss/gpu/GpuClonerOptions.h:struct GpuClonerOptions {
faiss/gpu/GpuClonerOptions.h:    /// (anything but GpuIndexFlat*)?
faiss/gpu/GpuClonerOptions.h:    /// for GpuIndexIVFFlat, is storage in float16?
faiss/gpu/GpuClonerOptions.h:    /// for GpuIndexIVFPQ, are intermediate calculations in float16?
faiss/gpu/GpuClonerOptions.h:    /// For GpuIndexFlat, store data in transposed layout?
faiss/gpu/GpuClonerOptions.h:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuClonerOptions.h:    /// throw an exception for indices not implemented on GPU. When set to
faiss/gpu/GpuClonerOptions.h:struct GpuMultipleClonerOptions : public GpuClonerOptions {
faiss/gpu/GpuClonerOptions.h:    /// Whether to shard the index across GPUs, versus replication
faiss/gpu/GpuClonerOptions.h:    /// across GPUs
faiss/gpu/GpuClonerOptions.h:    /// set to true if an IndexIVF is to be dispatched to multiple GPUs with a
faiss/gpu/GpuClonerOptions.h:} // namespace gpu
faiss/gpu/perf/PerfIVFPQAdd.cpp:#include <cuda_profiler_api.h>
faiss/gpu/perf/PerfIVFPQAdd.cpp:#include <faiss/gpu/GpuIndexIVFPQ.h>
faiss/gpu/perf/PerfIVFPQAdd.cpp:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/perf/PerfIVFPQAdd.cpp:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/perf/PerfIVFPQAdd.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/perf/PerfIVFPQAdd.cpp:#include <faiss/gpu/utils/Timer.h>
faiss/gpu/perf/PerfIVFPQAdd.cpp:DEFINE_int32(index, 2, "0 = no indices on GPU; 1 = 32 bit, 2 = 64 bit on GPU");
faiss/gpu/perf/PerfIVFPQAdd.cpp:DEFINE_bool(time_gpu, true, "time add to GPU");
faiss/gpu/perf/PerfIVFPQAdd.cpp:    cudaProfilerStop();
faiss/gpu/perf/PerfIVFPQAdd.cpp:    faiss::gpu::StandardGpuResources res;
faiss/gpu/perf/PerfIVFPQAdd.cpp:    std::vector<float> trainVecs = faiss::gpu::randVecs(numTrain, dim);
faiss/gpu/perf/PerfIVFPQAdd.cpp:    faiss::gpu::GpuIndexIVFPQConfig config;
faiss/gpu/perf/PerfIVFPQAdd.cpp:    config.indicesOptions = (faiss::gpu::IndicesOptions)FLAGS_index;
faiss/gpu/perf/PerfIVFPQAdd.cpp:    faiss::gpu::GpuIndexIVFPQ gpuIndex(
faiss/gpu/perf/PerfIVFPQAdd.cpp:    if (FLAGS_time_gpu) {
faiss/gpu/perf/PerfIVFPQAdd.cpp:        gpuIndex.train(numTrain, trainVecs.data());
faiss/gpu/perf/PerfIVFPQAdd.cpp:            gpuIndex.reserveMemory(numVecs);
faiss/gpu/perf/PerfIVFPQAdd.cpp:    cudaDeviceSynchronize();
faiss/gpu/perf/PerfIVFPQAdd.cpp:    CUDA_VERIFY(cudaProfilerStart());
faiss/gpu/perf/PerfIVFPQAdd.cpp:    float totalGpuTime = 0.0f;
faiss/gpu/perf/PerfIVFPQAdd.cpp:        auto addVecs = faiss::gpu::randVecs(FLAGS_batch_size, dim);
faiss/gpu/perf/PerfIVFPQAdd.cpp:        if (FLAGS_time_gpu) {
faiss/gpu/perf/PerfIVFPQAdd.cpp:            faiss::gpu::CpuTimer timer;
faiss/gpu/perf/PerfIVFPQAdd.cpp:            gpuIndex.add(FLAGS_batch_size, addVecs.data());
faiss/gpu/perf/PerfIVFPQAdd.cpp:            CUDA_VERIFY(cudaDeviceSynchronize());
faiss/gpu/perf/PerfIVFPQAdd.cpp:            totalGpuTime += time;
faiss/gpu/perf/PerfIVFPQAdd.cpp:                printf("Batch %d | GPU time to add %d vecs: %.3f ms (%.5f ms per)\n",
faiss/gpu/perf/PerfIVFPQAdd.cpp:            faiss::gpu::CpuTimer timer;
faiss/gpu/perf/PerfIVFPQAdd.cpp:    CUDA_VERIFY(cudaProfilerStop());
faiss/gpu/perf/PerfIVFPQAdd.cpp:    if (FLAGS_time_gpu) {
faiss/gpu/perf/PerfIVFPQAdd.cpp:               "GPU time to add %d vectors (%d batches, %d per batch): "
faiss/gpu/perf/PerfIVFPQAdd.cpp:               totalGpuTime,
faiss/gpu/perf/PerfIVFPQAdd.cpp:               totalGpuTime * 1000.0f / (float)total);
faiss/gpu/perf/WriteIndex.cpp:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/perf/WriteIndex.cpp:    auto trainVecs = faiss::gpu::randVecs(numTrain, dim);
faiss/gpu/perf/WriteIndex.cpp:        auto vecs = faiss::gpu::randVecs(numRemaining, dim);
faiss/gpu/perf/PerfIVFPQ.cu:#include <faiss/gpu/GpuIndexIVFPQ.h>
faiss/gpu/perf/PerfIVFPQ.cu:#include <faiss/gpu/perf/IndexWrapper.h>
faiss/gpu/perf/PerfIVFPQ.cu:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/perf/PerfIVFPQ.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/perf/PerfIVFPQ.cu:#include <faiss/gpu/utils/Timer.h>
faiss/gpu/perf/PerfIVFPQ.cu:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/perf/PerfIVFPQ.cu:#include <faiss/gpu/utils/HostTensor.cuh>
faiss/gpu/perf/PerfIVFPQ.cu:#include <cuda_profiler_api.h>
faiss/gpu/perf/PerfIVFPQ.cu:DEFINE_int32(num_gpus, 1, "number of gpus to use");
faiss/gpu/perf/PerfIVFPQ.cu:DEFINE_int32(index, 2, "0 = no indices on GPU; 1 = 32 bit, 2 = 64 bit on GPU");
faiss/gpu/perf/PerfIVFPQ.cu:using namespace faiss::gpu;
faiss/gpu/perf/PerfIVFPQ.cu:    CUDA_VERIFY(cudaProfilerStop());
faiss/gpu/perf/PerfIVFPQ.cu:    // Convert to GPU index
faiss/gpu/perf/PerfIVFPQ.cu:    printf("Copying index to %d GPU(s)...\n", FLAGS_num_gpus);
faiss/gpu/perf/PerfIVFPQ.cu:    auto indicesOpt = (faiss::gpu::IndicesOptions)FLAGS_index;
faiss/gpu/perf/PerfIVFPQ.cu:                    faiss::gpu::GpuResourcesProvider* res,
faiss/gpu/perf/PerfIVFPQ.cu:                    int dev) -> std::unique_ptr<faiss::gpu::GpuIndexIVFPQ> {
faiss/gpu/perf/PerfIVFPQ.cu:        faiss::gpu::GpuIndexIVFPQConfig config;
faiss/gpu/perf/PerfIVFPQ.cu:        auto p = std::unique_ptr<faiss::gpu::GpuIndexIVFPQ>(
faiss/gpu/perf/PerfIVFPQ.cu:                new faiss::gpu::GpuIndexIVFPQ(res, index.get(), config));
faiss/gpu/perf/PerfIVFPQ.cu:    IndexWrapper<faiss::gpu::GpuIndexIVFPQ> gpuIndex(FLAGS_num_gpus, initFn);
faiss/gpu/perf/PerfIVFPQ.cu:    gpuIndex.setNumProbes(FLAGS_nprobe);
faiss/gpu/perf/PerfIVFPQ.cu:    HostTensor<float, 2, true> gpuDistances({numQueries, FLAGS_k});
faiss/gpu/perf/PerfIVFPQ.cu:    HostTensor<faiss::idx_t, 2, true> gpuIndices({numQueries, FLAGS_k});
faiss/gpu/perf/PerfIVFPQ.cu:    CUDA_VERIFY(cudaProfilerStart());
faiss/gpu/perf/PerfIVFPQ.cu:    faiss::gpu::synchronizeAllDevices();
faiss/gpu/perf/PerfIVFPQ.cu:    float gpuTime = 0.0f;
faiss/gpu/perf/PerfIVFPQ.cu:    // Time GPU
faiss/gpu/perf/PerfIVFPQ.cu:        gpuIndex.getIndex()->search(
faiss/gpu/perf/PerfIVFPQ.cu:                gpuDistances.data(),
faiss/gpu/perf/PerfIVFPQ.cu:                gpuIndices.data());
faiss/gpu/perf/PerfIVFPQ.cu:        // additional synchronization with the GPU
faiss/gpu/perf/PerfIVFPQ.cu:        gpuTime = timer.elapsedMilliseconds();
faiss/gpu/perf/PerfIVFPQ.cu:    CUDA_VERIFY(cudaProfilerStop());
faiss/gpu/perf/PerfIVFPQ.cu:    printf("GPU time %.3f ms\n", gpuTime);
faiss/gpu/perf/PerfIVFPQ.cu:            gpuDistances.data(),
faiss/gpu/perf/PerfIVFPQ.cu:            gpuIndices.data(),
faiss/gpu/perf/PerfIVFPQ.cu:    CUDA_VERIFY(cudaDeviceSynchronize());
faiss/gpu/perf/PerfBinaryFlat.cu:#include <faiss/gpu/GpuIndexBinaryFlat.h>
faiss/gpu/perf/PerfBinaryFlat.cu:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/perf/PerfBinaryFlat.cu:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/perf/PerfBinaryFlat.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/perf/PerfBinaryFlat.cu:#include <faiss/gpu/utils/Timer.h>
faiss/gpu/perf/PerfBinaryFlat.cu:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/perf/PerfBinaryFlat.cu:#include <faiss/gpu/utils/HostTensor.cuh>
faiss/gpu/perf/PerfBinaryFlat.cu:#include <cuda_profiler_api.h>
faiss/gpu/perf/PerfBinaryFlat.cu:using namespace faiss::gpu;
faiss/gpu/perf/PerfBinaryFlat.cu:    cudaProfilerStop();
faiss/gpu/perf/PerfBinaryFlat.cu:    // Convert to GPU index
faiss/gpu/perf/PerfBinaryFlat.cu:    printf("Copying index to GPU...\n");
faiss/gpu/perf/PerfBinaryFlat.cu:    GpuIndexBinaryFlatConfig config;
faiss/gpu/perf/PerfBinaryFlat.cu:    faiss::gpu::StandardGpuResources res;
faiss/gpu/perf/PerfBinaryFlat.cu:    faiss::gpu::GpuIndexBinaryFlat gpuIndex(&res, index.get(), config);
faiss/gpu/perf/PerfBinaryFlat.cu:    HostTensor<int, 2, true> gpuDistances({numQueries, FLAGS_k});
faiss/gpu/perf/PerfBinaryFlat.cu:    HostTensor<faiss::idx_t, 2, true> gpuIndices({numQueries, FLAGS_k});
faiss/gpu/perf/PerfBinaryFlat.cu:    CUDA_VERIFY(cudaProfilerStart());
faiss/gpu/perf/PerfBinaryFlat.cu:    faiss::gpu::synchronizeAllDevices();
faiss/gpu/perf/PerfBinaryFlat.cu:    float gpuTime = 0.0f;
faiss/gpu/perf/PerfBinaryFlat.cu:    // Time GPU
faiss/gpu/perf/PerfBinaryFlat.cu:        gpuIndex.search(
faiss/gpu/perf/PerfBinaryFlat.cu:                gpuDistances.data(),
faiss/gpu/perf/PerfBinaryFlat.cu:                gpuIndices.data());
faiss/gpu/perf/PerfBinaryFlat.cu:        // additional synchronization with the GPU
faiss/gpu/perf/PerfBinaryFlat.cu:        gpuTime = timer.elapsedMilliseconds();
faiss/gpu/perf/PerfBinaryFlat.cu:    CUDA_VERIFY(cudaProfilerStop());
faiss/gpu/perf/PerfBinaryFlat.cu:    printf("GPU time %.3f ms\n", gpuTime);
faiss/gpu/perf/PerfBinaryFlat.cu:    CUDA_VERIFY(cudaDeviceSynchronize());
faiss/gpu/perf/PerfIVFFlat.cu:#include <faiss/gpu/GpuIndexIVFFlat.h>
faiss/gpu/perf/PerfIVFFlat.cu:#include <faiss/gpu/perf/IndexWrapper.h>
faiss/gpu/perf/PerfIVFFlat.cu:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/perf/PerfIVFFlat.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/perf/PerfIVFFlat.cu:#include <faiss/gpu/utils/Timer.h>
faiss/gpu/perf/PerfIVFFlat.cu:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/perf/PerfIVFFlat.cu:#include <faiss/gpu/utils/HostTensor.cuh>
faiss/gpu/perf/PerfIVFFlat.cu:#include <cuda_profiler_api.h>
faiss/gpu/perf/PerfIVFFlat.cu:DEFINE_int32(num_gpus, 1, "number of gpus to use");
faiss/gpu/perf/PerfIVFFlat.cu:DEFINE_int32(index, 2, "0 = no indices on GPU; 1 = 32 bit, 2 = 64 bit on GPU");
faiss/gpu/perf/PerfIVFFlat.cu:using namespace faiss::gpu;
faiss/gpu/perf/PerfIVFFlat.cu:    cudaProfilerStop();
faiss/gpu/perf/PerfIVFFlat.cu:    // Convert to GPU index
faiss/gpu/perf/PerfIVFFlat.cu:    printf("Copying index to %d GPU(s)...\n", FLAGS_num_gpus);
faiss/gpu/perf/PerfIVFFlat.cu:    auto initFn = [&index](faiss::gpu::GpuResourcesProvider* res, int dev)
faiss/gpu/perf/PerfIVFFlat.cu:            -> std::unique_ptr<faiss::gpu::GpuIndexIVFFlat> {
faiss/gpu/perf/PerfIVFFlat.cu:        GpuIndexIVFFlatConfig config;
faiss/gpu/perf/PerfIVFFlat.cu:        config.indicesOptions = (faiss::gpu::IndicesOptions)FLAGS_index;
faiss/gpu/perf/PerfIVFFlat.cu:        auto p = std::unique_ptr<faiss::gpu::GpuIndexIVFFlat>(
faiss/gpu/perf/PerfIVFFlat.cu:                new faiss::gpu::GpuIndexIVFFlat(
faiss/gpu/perf/PerfIVFFlat.cu:    IndexWrapper<faiss::gpu::GpuIndexIVFFlat> gpuIndex(FLAGS_num_gpus, initFn);
faiss/gpu/perf/PerfIVFFlat.cu:    gpuIndex.setNumProbes(FLAGS_nprobe);
faiss/gpu/perf/PerfIVFFlat.cu:    HostTensor<float, 2, true> gpuDistances({numQueries, FLAGS_k});
faiss/gpu/perf/PerfIVFFlat.cu:    HostTensor<faiss::idx_t, 2, true> gpuIndices({numQueries, FLAGS_k});
faiss/gpu/perf/PerfIVFFlat.cu:    CUDA_VERIFY(cudaProfilerStart());
faiss/gpu/perf/PerfIVFFlat.cu:    faiss::gpu::synchronizeAllDevices();
faiss/gpu/perf/PerfIVFFlat.cu:    float gpuTime = 0.0f;
faiss/gpu/perf/PerfIVFFlat.cu:    // Time GPU
faiss/gpu/perf/PerfIVFFlat.cu:        gpuIndex.getIndex()->search(
faiss/gpu/perf/PerfIVFFlat.cu:                gpuDistances.data(),
faiss/gpu/perf/PerfIVFFlat.cu:                gpuIndices.data());
faiss/gpu/perf/PerfIVFFlat.cu:        // additional synchronization with the GPU
faiss/gpu/perf/PerfIVFFlat.cu:        gpuTime = timer.elapsedMilliseconds();
faiss/gpu/perf/PerfIVFFlat.cu:    CUDA_VERIFY(cudaProfilerStop());
faiss/gpu/perf/PerfIVFFlat.cu:    printf("GPU time %.3f ms\n", gpuTime);
faiss/gpu/perf/PerfIVFFlat.cu:            gpuDistances.data(),
faiss/gpu/perf/PerfIVFFlat.cu:            gpuIndices.data(),
faiss/gpu/perf/PerfIVFFlat.cu:    CUDA_VERIFY(cudaDeviceSynchronize());
faiss/gpu/perf/slow.py:    index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(),
faiss/gpu/perf/PerfClustering.cpp:#include <faiss/gpu/GpuIndexFlat.h>
faiss/gpu/perf/PerfClustering.cpp:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/perf/PerfClustering.cpp:#include <faiss/gpu/perf/IndexWrapper.h>
faiss/gpu/perf/PerfClustering.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/perf/PerfClustering.cpp:#include <faiss/gpu/utils/Timer.h>
faiss/gpu/perf/PerfClustering.cpp:#include <cuda_profiler_api.h>
faiss/gpu/perf/PerfClustering.cpp:DEFINE_int32(num_gpus, 1, "number of gpus to use");
faiss/gpu/perf/PerfClustering.cpp:        "minimum size to use CPU -> GPU paged copies");
faiss/gpu/perf/PerfClustering.cpp:using namespace faiss::gpu;
faiss/gpu/perf/PerfClustering.cpp:    cudaProfilerStop();
faiss/gpu/perf/PerfClustering.cpp:    auto initFn = [](faiss::gpu::GpuResourcesProvider* res,
faiss/gpu/perf/PerfClustering.cpp:                     int dev) -> std::unique_ptr<faiss::gpu::GpuIndexFlat> {
faiss/gpu/perf/PerfClustering.cpp:            ((faiss::gpu::StandardGpuResources*)res)
faiss/gpu/perf/PerfClustering.cpp:        GpuIndexFlatConfig config;
faiss/gpu/perf/PerfClustering.cpp:        auto p = std::unique_ptr<faiss::gpu::GpuIndexFlat>(
faiss/gpu/perf/PerfClustering.cpp:                        ? (faiss::gpu::GpuIndexFlat*)new faiss::gpu::
faiss/gpu/perf/PerfClustering.cpp:                                  GpuIndexFlatL2(res, FLAGS_dim, config)
faiss/gpu/perf/PerfClustering.cpp:                        : (faiss::gpu::GpuIndexFlat*)new faiss::gpu::
faiss/gpu/perf/PerfClustering.cpp:                                  GpuIndexFlatIP(res, FLAGS_dim, config));
faiss/gpu/perf/PerfClustering.cpp:    IndexWrapper<faiss::gpu::GpuIndexFlat> gpuIndex(FLAGS_num_gpus, initFn);
faiss/gpu/perf/PerfClustering.cpp:    CUDA_VERIFY(cudaProfilerStart());
faiss/gpu/perf/PerfClustering.cpp:    faiss::gpu::synchronizeAllDevices();
faiss/gpu/perf/PerfClustering.cpp:    float gpuTime = 0.0f;
faiss/gpu/perf/PerfClustering.cpp:        kmeans.train(FLAGS_num, vecs.data(), *(gpuIndex.getIndex()));
faiss/gpu/perf/PerfClustering.cpp:        // additional synchronization with the GPU
faiss/gpu/perf/PerfClustering.cpp:        gpuTime = timer.elapsedMilliseconds();
faiss/gpu/perf/PerfClustering.cpp:    CUDA_VERIFY(cudaProfilerStop());
faiss/gpu/perf/PerfClustering.cpp:    printf("k-means time %.3f ms\n", gpuTime);
faiss/gpu/perf/PerfClustering.cpp:    CUDA_VERIFY(cudaDeviceSynchronize());
faiss/gpu/perf/IndexWrapper-inl.h:namespace gpu {
faiss/gpu/perf/IndexWrapper-inl.h:template <typename GpuIndex>
faiss/gpu/perf/IndexWrapper-inl.h:IndexWrapper<GpuIndex>::IndexWrapper(
faiss/gpu/perf/IndexWrapper-inl.h:        int numGpus,
faiss/gpu/perf/IndexWrapper-inl.h:        std::function<std::unique_ptr<GpuIndex>(GpuResourcesProvider*, int)>
faiss/gpu/perf/IndexWrapper-inl.h:    FAISS_ASSERT(numGpus <= faiss::gpu::getNumDevices());
faiss/gpu/perf/IndexWrapper-inl.h:    for (int i = 0; i < numGpus; ++i) {
faiss/gpu/perf/IndexWrapper-inl.h:        auto res = std::unique_ptr<faiss::gpu::StandardGpuResources>(
faiss/gpu/perf/IndexWrapper-inl.h:                new StandardGpuResources);
faiss/gpu/perf/IndexWrapper-inl.h:    if (numGpus > 1) {
faiss/gpu/perf/IndexWrapper-inl.h:template <typename GpuIndex>
faiss/gpu/perf/IndexWrapper-inl.h:faiss::Index* IndexWrapper<GpuIndex>::getIndex() {
faiss/gpu/perf/IndexWrapper-inl.h:template <typename GpuIndex>
faiss/gpu/perf/IndexWrapper-inl.h:void IndexWrapper<GpuIndex>::runOnIndices(std::function<void(GpuIndex*)> f) {
faiss/gpu/perf/IndexWrapper-inl.h:            f(dynamic_cast<GpuIndex*>(index));
faiss/gpu/perf/IndexWrapper-inl.h:template <typename GpuIndex>
faiss/gpu/perf/IndexWrapper-inl.h:void IndexWrapper<GpuIndex>::setNumProbes(size_t nprobe) {
faiss/gpu/perf/IndexWrapper-inl.h:    runOnIndices([nprobe](GpuIndex* index) { index->nprobe = nprobe; });
faiss/gpu/perf/IndexWrapper-inl.h:} // namespace gpu
faiss/gpu/perf/PerfSelect.cu:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/perf/PerfSelect.cu:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/perf/PerfSelect.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/perf/PerfSelect.cu:#include <faiss/gpu/utils/BlockSelectKernel.cuh>
faiss/gpu/perf/PerfSelect.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/perf/PerfSelect.cu:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/perf/PerfSelect.cu:#include <faiss/gpu/utils/HostTensor.cuh>
faiss/gpu/perf/PerfSelect.cu:#include <faiss/gpu/utils/WarpSelectKernel.cuh>
faiss/gpu/perf/PerfSelect.cu:    using namespace faiss::gpu;
faiss/gpu/perf/PerfSelect.cu:    StandardGpuResources res;
faiss/gpu/perf/PerfSelect.cu:    // Select top-k on GPU
faiss/gpu/perf/PerfSelect.cu:    DeviceTensor<float, 2, true> gpuVal(
faiss/gpu/perf/PerfSelect.cu:        limitK = GPU_MAX_SELECTION_K;
faiss/gpu/perf/PerfSelect.cu:        DeviceTensor<float, 2, true> gpuOutVal(
faiss/gpu/perf/PerfSelect.cu:        DeviceTensor<faiss::idx_t, 2, true> gpuOutInd(
faiss/gpu/perf/PerfSelect.cu:                runWarpSelect(gpuVal, gpuOutVal, gpuOutInd, FLAGS_dir, k, 0);
faiss/gpu/perf/PerfSelect.cu:                runBlockSelect(gpuVal, gpuOutVal, gpuOutInd, FLAGS_dir, k, 0);
faiss/gpu/perf/PerfSelect.cu:    cudaDeviceSynchronize();
faiss/gpu/perf/IndexWrapper.h:#include <faiss/gpu/StandardGpuResources.h>
faiss/gpu/perf/IndexWrapper.h:namespace gpu {
faiss/gpu/perf/IndexWrapper.h:// If we want to run multi-GPU, create a proxy to wrap the indices.
faiss/gpu/perf/IndexWrapper.h:// If we don't want multi-GPU, don't involve the proxy, so it doesn't
faiss/gpu/perf/IndexWrapper.h:template <typename GpuIndex>
faiss/gpu/perf/IndexWrapper.h:    std::vector<std::unique_ptr<faiss::gpu::StandardGpuResources>> resources;
faiss/gpu/perf/IndexWrapper.h:    std::vector<std::unique_ptr<GpuIndex>> subIndex;
faiss/gpu/perf/IndexWrapper.h:            int numGpus,
faiss/gpu/perf/IndexWrapper.h:            std::function<std::unique_ptr<GpuIndex>(GpuResourcesProvider*, int)>
faiss/gpu/perf/IndexWrapper.h:    void runOnIndices(std::function<void(GpuIndex*)> f);
faiss/gpu/perf/IndexWrapper.h:} // namespace gpu
faiss/gpu/perf/IndexWrapper.h:#include <faiss/gpu/perf/IndexWrapper-inl.h>
faiss/gpu/perf/PerfFlat.cu:#include <faiss/gpu/GpuIndexFlat.h>
faiss/gpu/perf/PerfFlat.cu:#include <faiss/gpu/perf/IndexWrapper.h>
faiss/gpu/perf/PerfFlat.cu:#include <faiss/gpu/test/TestUtils.h>
faiss/gpu/perf/PerfFlat.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/perf/PerfFlat.cu:#include <faiss/gpu/utils/Timer.h>
faiss/gpu/perf/PerfFlat.cu:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/perf/PerfFlat.cu:#include <faiss/gpu/utils/HostTensor.cuh>
faiss/gpu/perf/PerfFlat.cu:#include <cuda_profiler_api.h>
faiss/gpu/perf/PerfFlat.cu:DEFINE_int32(num_gpus, 1, "number of gpus to use");
faiss/gpu/perf/PerfFlat.cu:using namespace faiss::gpu;
faiss/gpu/perf/PerfFlat.cu:    cudaProfilerStop();
faiss/gpu/perf/PerfFlat.cu:    // Convert to GPU index
faiss/gpu/perf/PerfFlat.cu:    printf("Copying index to %d GPU(s)...\n", FLAGS_num_gpus);
faiss/gpu/perf/PerfFlat.cu:    auto initFn = [&index](faiss::gpu::GpuResourcesProvider* res, int dev)
faiss/gpu/perf/PerfFlat.cu:            -> std::unique_ptr<faiss::gpu::GpuIndexFlat> {
faiss/gpu/perf/PerfFlat.cu:        ((faiss::gpu::StandardGpuResources*)res)
faiss/gpu/perf/PerfFlat.cu:        GpuIndexFlatConfig config;
faiss/gpu/perf/PerfFlat.cu:        auto p = std::unique_ptr<faiss::gpu::GpuIndexFlat>(
faiss/gpu/perf/PerfFlat.cu:                new faiss::gpu::GpuIndexFlat(res, index.get(), config));
faiss/gpu/perf/PerfFlat.cu:    IndexWrapper<faiss::gpu::GpuIndexFlat> gpuIndex(FLAGS_num_gpus, initFn);
faiss/gpu/perf/PerfFlat.cu:    HostTensor<float, 2, true> gpuDistances({numQueries, FLAGS_k});
faiss/gpu/perf/PerfFlat.cu:    HostTensor<faiss::idx_t, 2, true> gpuIndices({numQueries, FLAGS_k});
faiss/gpu/perf/PerfFlat.cu:    CUDA_VERIFY(cudaProfilerStart());
faiss/gpu/perf/PerfFlat.cu:    faiss::gpu::synchronizeAllDevices();
faiss/gpu/perf/PerfFlat.cu:    float gpuTime = 0.0f;
faiss/gpu/perf/PerfFlat.cu:    // Time GPU
faiss/gpu/perf/PerfFlat.cu:        gpuIndex.getIndex()->search(
faiss/gpu/perf/PerfFlat.cu:                gpuDistances.data(),
faiss/gpu/perf/PerfFlat.cu:                gpuIndices.data());
faiss/gpu/perf/PerfFlat.cu:        // additional synchronization with the GPU
faiss/gpu/perf/PerfFlat.cu:        gpuTime = timer.elapsedMilliseconds();
faiss/gpu/perf/PerfFlat.cu:    CUDA_VERIFY(cudaProfilerStop());
faiss/gpu/perf/PerfFlat.cu:    printf("GPU time %.3f ms\n", gpuTime);
faiss/gpu/perf/PerfFlat.cu:                gpuDistances.data(),
faiss/gpu/perf/PerfFlat.cu:                gpuIndices.data(),
faiss/gpu/perf/PerfFlat.cu:    CUDA_VERIFY(cudaDeviceSynchronize());
faiss/gpu/GpuIndexIVFFlat.cu:#include <faiss/gpu/GpuIndexFlat.h>
faiss/gpu/GpuIndexIVFFlat.cu:#include <faiss/gpu/GpuIndexIVFFlat.h>
faiss/gpu/GpuIndexIVFFlat.cu:#include <faiss/gpu/GpuResources.h>
faiss/gpu/GpuIndexIVFFlat.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/GpuIndexIVFFlat.cu:#include <faiss/gpu/impl/IVFFlat.cuh>
faiss/gpu/GpuIndexIVFFlat.cu:#include <faiss/gpu/utils/CopyUtils.cuh>
faiss/gpu/GpuIndexIVFFlat.cu:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/GpuIndexIVFFlat.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuIndexIVFFlat.cu:#include <faiss/gpu/utils/RaftUtils.h>
faiss/gpu/GpuIndexIVFFlat.cu:#include <faiss/gpu/impl/RaftIVFFlat.cuh>
faiss/gpu/GpuIndexIVFFlat.cu:namespace gpu {
faiss/gpu/GpuIndexIVFFlat.cu:GpuIndexIVFFlat::GpuIndexIVFFlat(
faiss/gpu/GpuIndexIVFFlat.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFFlat.cu:        GpuIndexIVFFlatConfig config)
faiss/gpu/GpuIndexIVFFlat.cu:        : GpuIndexIVF(
faiss/gpu/GpuIndexIVFFlat.cu:GpuIndexIVFFlat::GpuIndexIVFFlat(
faiss/gpu/GpuIndexIVFFlat.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFFlat.cu:        GpuIndexIVFFlatConfig config)
faiss/gpu/GpuIndexIVFFlat.cu:        : GpuIndexIVF(provider, dims, metric, 0, nlist, config),
faiss/gpu/GpuIndexIVFFlat.cu:GpuIndexIVFFlat::GpuIndexIVFFlat(
faiss/gpu/GpuIndexIVFFlat.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFFlat.cu:        GpuIndexIVFFlatConfig config)
faiss/gpu/GpuIndexIVFFlat.cu:        : GpuIndexIVF(
faiss/gpu/GpuIndexIVFFlat.cu:            "GpuIndexIVFFlat: RAFT does not support separate coarseQuantizer");
faiss/gpu/GpuIndexIVFFlat.cu:GpuIndexIVFFlat::~GpuIndexIVFFlat() {}
faiss/gpu/GpuIndexIVFFlat.cu:void GpuIndexIVFFlat::reserveMemory(size_t numVecs) {
faiss/gpu/GpuIndexIVFFlat.cu:void GpuIndexIVFFlat::copyFrom(const faiss::IndexIVFFlat* index) {
faiss/gpu/GpuIndexIVFFlat.cu:    // This will copy GpuIndexIVF data such as the coarse quantizer
faiss/gpu/GpuIndexIVFFlat.cu:    GpuIndexIVF::copyFrom(index);
faiss/gpu/GpuIndexIVFFlat.cu:void GpuIndexIVFFlat::copyTo(faiss::IndexIVFFlat* index) const {
faiss/gpu/GpuIndexIVFFlat.cu:            "Cannot copy to CPU as GPU index doesn't retain "
faiss/gpu/GpuIndexIVFFlat.cu:    GpuIndexIVF::copyTo(index);
faiss/gpu/GpuIndexIVFFlat.cu:size_t GpuIndexIVFFlat::reclaimMemory() {
faiss/gpu/GpuIndexIVFFlat.cu:void GpuIndexIVFFlat::reset() {
faiss/gpu/GpuIndexIVFFlat.cu:void GpuIndexIVFFlat::updateQuantizer() {
faiss/gpu/GpuIndexIVFFlat.cu:void GpuIndexIVFFlat::train(idx_t n, const float* x) {
faiss/gpu/GpuIndexIVFFlat.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuIndexIVFFlat.cu:        // FIXME: GPUize more of this
faiss/gpu/GpuIndexIVFFlat.cu:void GpuIndexIVFFlat::setIndex_(
faiss/gpu/GpuIndexIVFFlat.cu:        GpuResources* resources,
faiss/gpu/GpuIndexIVFFlat.cu:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuIndexIVFFlat.cu:void GpuIndexIVFFlat::reconstruct_n(idx_t i0, idx_t ni, float* out) const {
faiss/gpu/GpuIndexIVFFlat.cu:} // namespace gpu
faiss/gpu/GpuIndexIVFFlat.h:#include <faiss/gpu/GpuIndexIVF.h>
faiss/gpu/GpuIndexIVFFlat.h:namespace gpu {
faiss/gpu/GpuIndexIVFFlat.h:class GpuIndexFlat;
faiss/gpu/GpuIndexIVFFlat.h:struct GpuIndexIVFFlatConfig : public GpuIndexIVFConfig {
faiss/gpu/GpuIndexIVFFlat.h:/// Wrapper around the GPU implementation that looks like
faiss/gpu/GpuIndexIVFFlat.h:class GpuIndexIVFFlat : public GpuIndexIVF {
faiss/gpu/GpuIndexIVFFlat.h:    /// data over to the given GPU, if the input index is trained.
faiss/gpu/GpuIndexIVFFlat.h:    GpuIndexIVFFlat(
faiss/gpu/GpuIndexIVFFlat.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFFlat.h:            GpuIndexIVFFlatConfig config = GpuIndexIVFFlatConfig());
faiss/gpu/GpuIndexIVFFlat.h:    GpuIndexIVFFlat(
faiss/gpu/GpuIndexIVFFlat.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFFlat.h:            GpuIndexIVFFlatConfig config = GpuIndexIVFFlatConfig());
faiss/gpu/GpuIndexIVFFlat.h:    /// Constructs a new instance with a provided CPU or GPU coarse quantizer;
faiss/gpu/GpuIndexIVFFlat.h:    GpuIndexIVFFlat(
faiss/gpu/GpuIndexIVFFlat.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFFlat.h:            GpuIndexIVFFlatConfig config = GpuIndexIVFFlatConfig());
faiss/gpu/GpuIndexIVFFlat.h:    ~GpuIndexIVFFlat() override;
faiss/gpu/GpuIndexIVFFlat.h:    /// Reserve GPU memory in our inverted lists for this number of vectors
faiss/gpu/GpuIndexIVFFlat.h:            GpuResources* resources,
faiss/gpu/GpuIndexIVFFlat.h:    const GpuIndexIVFFlatConfig ivfFlatConfig_;
faiss/gpu/GpuIndexIVFFlat.h:} // namespace gpu
faiss/gpu/GpuIndex.h: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/GpuIndex.h:#include <faiss/gpu/GpuResources.h>
faiss/gpu/GpuIndex.h:namespace gpu {
faiss/gpu/GpuIndex.h:struct GpuIndexConfig {
faiss/gpu/GpuIndex.h:    /// GPU device on which the index is resident
faiss/gpu/GpuIndex.h:    /// On Pascal and above (CC 6+) architectures, allows GPUs to use
faiss/gpu/GpuIndex.h:    /// more memory than is available on the GPU.
faiss/gpu/GpuIndex.h:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuIndex.h:bool should_use_raft(GpuIndexConfig config_);
faiss/gpu/GpuIndex.h:class GpuIndex : public faiss::Index {
faiss/gpu/GpuIndex.h:    GpuIndex(
faiss/gpu/GpuIndex.h:            std::shared_ptr<GpuResources> resources,
faiss/gpu/GpuIndex.h:            GpuIndexConfig config);
faiss/gpu/GpuIndex.h:    /// Returns a reference to our GpuResources object that manages memory,
faiss/gpu/GpuIndex.h:    /// stream and handle resources on the GPU
faiss/gpu/GpuIndex.h:    std::shared_ptr<GpuResources> getResources();
faiss/gpu/GpuIndex.h:    /// CPU -> GPU paging
faiss/gpu/GpuIndex.h:    /// `x` can be resident on the CPU or any GPU; copies are performed
faiss/gpu/GpuIndex.h:    /// `x` and `ids` can be resident on the CPU or any GPU; copies are
faiss/gpu/GpuIndex.h:    /// `x` and `labels` can be resident on the CPU or any GPU; copies are
faiss/gpu/GpuIndex.h:    /// GPU; copies are performed as needed
faiss/gpu/GpuIndex.h:    /// any GPU; copies are performed as needed
faiss/gpu/GpuIndex.h:    /// Overridden to force GPU indices to provide their own GPU-friendly
faiss/gpu/GpuIndex.h:    /// Overridden to force GPU indices to provide their own GPU-friendly
faiss/gpu/GpuIndex.h:    /// Calls addImpl_ for a single page of GPU-resident data
faiss/gpu/GpuIndex.h:    /// Calls searchImpl_ for a single page of GPU-resident data
faiss/gpu/GpuIndex.h:    /// Calls searchImpl_ for a single page of GPU-resident data,
faiss/gpu/GpuIndex.h:    std::shared_ptr<GpuResources> resources_;
faiss/gpu/GpuIndex.h:    const GpuIndexConfig config_;
faiss/gpu/GpuIndex.h:    /// Size above which we page copies from the CPU to GPU
faiss/gpu/GpuIndex.h:/// If the given index is a GPU index, this returns the index instance
faiss/gpu/GpuIndex.h:GpuIndex* tryCastGpuIndex(faiss::Index* index);
faiss/gpu/GpuIndex.h:/// Is the given index instance a GPU index?
faiss/gpu/GpuIndex.h:bool isGpuIndex(faiss::Index* index);
faiss/gpu/GpuIndex.h:/// Does the given CPU index instance have a corresponding GPU implementation?
faiss/gpu/GpuIndex.h:bool isGpuIndexImplemented(faiss::Index* index);
faiss/gpu/GpuIndex.h:} // namespace gpu
faiss/gpu/GpuIndexIVF.h:#include <faiss/gpu/GpuIndex.h>
faiss/gpu/GpuIndexIVF.h:#include <faiss/gpu/GpuIndexFlat.h>
faiss/gpu/GpuIndexIVF.h:#include <faiss/gpu/GpuIndicesOptions.h>
faiss/gpu/GpuIndexIVF.h:namespace gpu {
faiss/gpu/GpuIndexIVF.h:class GpuIndexFlat;
faiss/gpu/GpuIndexIVF.h:struct GpuIndexIVFConfig : public GpuIndexConfig {
faiss/gpu/GpuIndexIVF.h:    /// Index storage options for the GPU
faiss/gpu/GpuIndexIVF.h:    GpuIndexFlatConfig flatConfig;
faiss/gpu/GpuIndexIVF.h:    /// throw an exception for indices not implemented on GPU. When set to
faiss/gpu/GpuIndexIVF.h:/// Base class of all GPU IVF index types. This (for now) deliberately does not
faiss/gpu/GpuIndexIVF.h:/// in IndexIVF is not supported in the same manner on the GPU.
faiss/gpu/GpuIndexIVF.h:class GpuIndexIVF : public GpuIndex, public IndexIVFInterface {
faiss/gpu/GpuIndexIVF.h:    GpuIndexIVF(
faiss/gpu/GpuIndexIVF.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVF.h:            GpuIndexIVFConfig config = GpuIndexIVFConfig());
faiss/gpu/GpuIndexIVF.h:    /// Version that takes a coarse quantizer instance. The GpuIndexIVF does not
faiss/gpu/GpuIndexIVF.h:    GpuIndexIVF(
faiss/gpu/GpuIndexIVF.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVF.h:            GpuIndexIVFConfig config = GpuIndexIVFConfig());
faiss/gpu/GpuIndexIVF.h:    ~GpuIndexIVF() override;
faiss/gpu/GpuIndexIVF.h:    /// If gpuFormat is true, the data is returned as it is encoded in the
faiss/gpu/GpuIndexIVF.h:    /// GPU-side representation.
faiss/gpu/GpuIndexIVF.h:    /// compliant format, while the native GPU format may differ.
faiss/gpu/GpuIndexIVF.h:            bool gpuFormat = false) const;
faiss/gpu/GpuIndexIVF.h:    // not implemented for GPU
faiss/gpu/GpuIndexIVF.h:    /// Called from GpuIndex for add/add_with_ids
faiss/gpu/GpuIndexIVF.h:    /// Called from GpuIndex for search
faiss/gpu/GpuIndexIVF.h:    const GpuIndexIVFConfig ivfConfig_;
faiss/gpu/GpuIndexIVF.h:} // namespace gpu
faiss/gpu/GpuDistance.h:namespace gpu {
faiss/gpu/GpuDistance.h:class GpuResourcesProvider;
faiss/gpu/GpuDistance.h:/// Arguments to brute-force GPU k-nearest neighbor searching
faiss/gpu/GpuDistance.h:struct GpuDistanceParams {
faiss/gpu/GpuDistance.h:    /// On which GPU device should the search run?
faiss/gpu/GpuDistance.h:    /// -1 indicates that the current CUDA thread-local device
faiss/gpu/GpuDistance.h:    /// (via cudaGetDevice/cudaSetDevice) is used
faiss/gpu/GpuDistance.h:bool should_use_raft(GpuDistanceParams args);
faiss/gpu/GpuDistance.h:/// A wrapper for gpu/impl/Distance.cuh to expose direct brute-force k-nearest
faiss/gpu/GpuDistance.h:/// GPU or the CPU, but all calculations are performed on the GPU. If the result
faiss/gpu/GpuDistance.h:/// All GPU computation is performed on the current CUDA device, and ordered
faiss/gpu/GpuDistance.h:void bfKnn(GpuResourcesProvider* resources, const GpuDistanceParams& args);
faiss/gpu/GpuDistance.h:// bfKnn which takes two extra parameters to control the maximum GPU
faiss/gpu/GpuDistance.h:// If 0, the corresponding input must fit into GPU memory.
faiss/gpu/GpuDistance.h:// If greater than 0, the function will use at most this much GPU
faiss/gpu/GpuDistance.h:// chunks are processed sequentially on the GPU.
faiss/gpu/GpuDistance.h:        GpuResourcesProvider* resources,
faiss/gpu/GpuDistance.h:        const GpuDistanceParams& args,
faiss/gpu/GpuDistance.h:        GpuResourcesProvider* resources,
faiss/gpu/GpuDistance.h:} // namespace gpu
faiss/gpu/GpuIndexIVFPQ.h:#include <faiss/gpu/GpuIndexIVF.h>
faiss/gpu/GpuIndexIVFPQ.h:namespace gpu {
faiss/gpu/GpuIndexIVFPQ.h:class GpuIndexFlat;
faiss/gpu/GpuIndexIVFPQ.h:struct GpuIndexIVFPQConfig : public GpuIndexIVFConfig {
faiss/gpu/GpuIndexIVFPQ.h:/// IVFPQ index for the GPU
faiss/gpu/GpuIndexIVFPQ.h:class GpuIndexIVFPQ : public GpuIndexIVF {
faiss/gpu/GpuIndexIVFPQ.h:    /// data over to the given GPU, if the input index is trained.
faiss/gpu/GpuIndexIVFPQ.h:    GpuIndexIVFPQ(
faiss/gpu/GpuIndexIVFPQ.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFPQ.h:            GpuIndexIVFPQConfig config = GpuIndexIVFPQConfig());
faiss/gpu/GpuIndexIVFPQ.h:    GpuIndexIVFPQ(
faiss/gpu/GpuIndexIVFPQ.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFPQ.h:            GpuIndexIVFPQConfig config = GpuIndexIVFPQConfig());
faiss/gpu/GpuIndexIVFPQ.h:    /// Constructs a new instance with a provided CPU or GPU coarse quantizer;
faiss/gpu/GpuIndexIVFPQ.h:    GpuIndexIVFPQ(
faiss/gpu/GpuIndexIVFPQ.h:            GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVFPQ.h:            GpuIndexIVFPQConfig config = GpuIndexIVFPQConfig());
faiss/gpu/GpuIndexIVFPQ.h:    ~GpuIndexIVFPQ() override;
faiss/gpu/GpuIndexIVFPQ.h:    /// Reserve space on the GPU for the inverted lists for `num`
faiss/gpu/GpuIndexIVFPQ.h:    /// Reserve GPU memory in our inverted lists for this number of vectors
faiss/gpu/GpuIndexIVFPQ.h:            GpuResources* resources,
faiss/gpu/GpuIndexIVFPQ.h:    const GpuIndexIVFPQConfig ivfpqConfig_;
faiss/gpu/GpuIndexIVFPQ.h:} // namespace gpu
faiss/gpu/utils/DeviceDefs.cuh:#include <cuda.h>
faiss/gpu/utils/DeviceDefs.cuh:namespace gpu {
faiss/gpu/utils/DeviceDefs.cuh:#ifdef USE_AMD_ROCM
faiss/gpu/utils/DeviceDefs.cuh:#define GPU_MAX_SELECTION_K 2048
faiss/gpu/utils/DeviceDefs.cuh:#else // USE_AMD_ROCM
faiss/gpu/utils/DeviceDefs.cuh:// We require at least CUDA 8.0 for compilation
faiss/gpu/utils/DeviceDefs.cuh:#if CUDA_VERSION < 8000
faiss/gpu/utils/DeviceDefs.cuh:#error "CUDA >= 8.0 is required"
faiss/gpu/utils/DeviceDefs.cuh:#if CUDA_VERSION >= 9000
faiss/gpu/utils/DeviceDefs.cuh:#if CUDA_VERSION > 9000
faiss/gpu/utils/DeviceDefs.cuh:// Based on the CUDA version (we assume what version of nvcc/ptxas we were
faiss/gpu/utils/DeviceDefs.cuh:#define GPU_MAX_SELECTION_K 2048
faiss/gpu/utils/DeviceDefs.cuh:#define GPU_MAX_SELECTION_K 1024
faiss/gpu/utils/DeviceDefs.cuh:#endif // USE_AMD_ROCM
faiss/gpu/utils/DeviceDefs.cuh:} // namespace gpu
faiss/gpu/utils/Tensor.cuh:#include <cuda.h>
faiss/gpu/utils/Tensor.cuh:#include <cuda_runtime.h>
faiss/gpu/utils/Tensor.cuh:/// Multi-dimensional array class for CUDA device and host usage.
faiss/gpu/utils/Tensor.cuh:/// Originally from Facebook's fbcunn, since added to the Torch GPU
faiss/gpu/utils/Tensor.cuh:namespace gpu {
faiss/gpu/utils/Tensor.cuh:            cudaStream_t stream);
faiss/gpu/utils/Tensor.cuh:            cudaStream_t stream);
faiss/gpu/utils/Tensor.cuh:    __host__ void copyFrom(const std::vector<T>& v, cudaStream_t stream);
faiss/gpu/utils/Tensor.cuh:    __host__ std::vector<T> copyToVector(cudaStream_t stream);
faiss/gpu/utils/Tensor.cuh:#if __CUDA_ARCH__ >= 350 || defined(USE_AMD_ROCM)
faiss/gpu/utils/Tensor.cuh:#if __CUDA_ARCH__ >= 350 || defined(USE_AMD_ROCM)
faiss/gpu/utils/Tensor.cuh:#if __CUDA_ARCH__ >= 350 || defined(USE_AMD_ROCM)
faiss/gpu/utils/Tensor.cuh:#if __CUDA_ARCH__ >= 350 || defined(USE_AMD_ROCM)
faiss/gpu/utils/Tensor.cuh:} // namespace gpu
faiss/gpu/utils/Tensor.cuh:#include <faiss/gpu/utils/Tensor-inl.cuh>
faiss/gpu/utils/ThrustUtils.cuh:#include <cuda.h>
faiss/gpu/utils/ThrustUtils.cuh:#include <faiss/gpu/GpuResources.h>
faiss/gpu/utils/ThrustUtils.cuh:namespace gpu {
faiss/gpu/utils/ThrustUtils.cuh:            GpuResources* res,
faiss/gpu/utils/ThrustUtils.cuh:            cudaStream_t stream,
faiss/gpu/utils/ThrustUtils.cuh:        // didn't cudaMalloc
faiss/gpu/utils/ThrustUtils.cuh:    GpuResources* res_;
faiss/gpu/utils/ThrustUtils.cuh:    cudaStream_t stream_;
faiss/gpu/utils/ThrustUtils.cuh:} // namespace gpu
faiss/gpu/utils/Select.cuh:#include <faiss/gpu/utils/Comparators.cuh>
faiss/gpu/utils/Select.cuh:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/utils/Select.cuh:#include <faiss/gpu/utils/MergeNetworkBlock.cuh>
faiss/gpu/utils/Select.cuh:#include <faiss/gpu/utils/MergeNetworkWarp.cuh>
faiss/gpu/utils/Select.cuh:#include <faiss/gpu/utils/PtxUtils.cuh>
faiss/gpu/utils/Select.cuh:#include <faiss/gpu/utils/ReductionOperators.cuh>
faiss/gpu/utils/Select.cuh:#include <faiss/gpu/utils/Reductions.cuh>
faiss/gpu/utils/Select.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/utils/Select.cuh:namespace gpu {
faiss/gpu/utils/Select.cuh:#if CUDA_VERSION < 9000 || defined(USE_AMD_ROCM)
faiss/gpu/utils/Select.cuh:#if CUDA_VERSION < 9000 || defined(USE_AMD_ROCM)
faiss/gpu/utils/Select.cuh:} // namespace gpu
faiss/gpu/utils/WarpSelectKernel.cuh:#include <faiss/gpu/utils/Select.cuh>
faiss/gpu/utils/WarpSelectKernel.cuh:namespace gpu {
faiss/gpu/utils/WarpSelectKernel.cuh:        cudaStream_t stream);
faiss/gpu/utils/WarpSelectKernel.cuh:        cudaStream_t stream);
faiss/gpu/utils/WarpSelectKernel.cuh:} // namespace gpu
faiss/gpu/utils/StaticUtils.h:#include <cuda.h>
faiss/gpu/utils/StaticUtils.h:// allow usage for non-CUDA files
faiss/gpu/utils/StaticUtils.h:namespace gpu {
faiss/gpu/utils/StaticUtils.h:} // namespace gpu
faiss/gpu/utils/DeviceUtils.h:#include <cuda_runtime.h>
faiss/gpu/utils/DeviceUtils.h:namespace gpu {
faiss/gpu/utils/DeviceUtils.h:/// Returns the current thread-local GPU device
faiss/gpu/utils/DeviceUtils.h:/// Sets the current thread-local GPU device
faiss/gpu/utils/DeviceUtils.h:/// Returns the number of available GPU devices
faiss/gpu/utils/DeviceUtils.h:/// Starts the CUDA profiler (exposed via SWIG)
faiss/gpu/utils/DeviceUtils.h:/// Stops the CUDA profiler (exposed via SWIG)
faiss/gpu/utils/DeviceUtils.h:/// cudaDeviceSynchronize for each device)
faiss/gpu/utils/DeviceUtils.h:/// Returns a cached cudaDeviceProp for the given device
faiss/gpu/utils/DeviceUtils.h:const cudaDeviceProp& getDeviceProperties(int device);
faiss/gpu/utils/DeviceUtils.h:/// Returns the cached cudaDeviceProp for the current device
faiss/gpu/utils/DeviceUtils.h:const cudaDeviceProp& getCurrentDeviceProperties();
faiss/gpu/utils/DeviceUtils.h:/// Returns the maximum number of threads available for the given GPU
faiss/gpu/utils/DeviceUtils.h:/// Returns the maximum grid size for the given GPU device
faiss/gpu/utils/DeviceUtils.h:/// Returns the maximum smem available for the given GPU device
faiss/gpu/utils/DeviceUtils.h:/// Returns the warp size of the given GPU device
faiss/gpu/utils/DeviceUtils.h:// RAII object to manage a cudaEvent_t
faiss/gpu/utils/DeviceUtils.h:class CudaEvent {
faiss/gpu/utils/DeviceUtils.h:    explicit CudaEvent(cudaStream_t stream, bool timer = false);
faiss/gpu/utils/DeviceUtils.h:    CudaEvent(const CudaEvent& event) = delete;
faiss/gpu/utils/DeviceUtils.h:    CudaEvent(CudaEvent&& event) noexcept;
faiss/gpu/utils/DeviceUtils.h:    ~CudaEvent();
faiss/gpu/utils/DeviceUtils.h:    inline cudaEvent_t get() {
faiss/gpu/utils/DeviceUtils.h:    void streamWaitOnEvent(cudaStream_t stream);
faiss/gpu/utils/DeviceUtils.h:    CudaEvent& operator=(CudaEvent&& event) noexcept;
faiss/gpu/utils/DeviceUtils.h:    CudaEvent& operator=(CudaEvent& event) = delete;
faiss/gpu/utils/DeviceUtils.h:    cudaEvent_t event_;
faiss/gpu/utils/DeviceUtils.h:/// Wrapper to test return status of CUDA functions
faiss/gpu/utils/DeviceUtils.h:#define CUDA_VERIFY(X)                      \
faiss/gpu/utils/DeviceUtils.h:                err__ == cudaSuccess,       \
faiss/gpu/utils/DeviceUtils.h:                "CUDA error %d %s",         \
faiss/gpu/utils/DeviceUtils.h:                cudaGetErrorString(err__)); \
faiss/gpu/utils/DeviceUtils.h:/// Wrapper to synchronously probe for CUDA errors
faiss/gpu/utils/DeviceUtils.h:// #define FAISS_GPU_SYNC_ERROR 1
faiss/gpu/utils/DeviceUtils.h:#ifdef FAISS_GPU_SYNC_ERROR
faiss/gpu/utils/DeviceUtils.h:#define CUDA_TEST_ERROR()                     \
faiss/gpu/utils/DeviceUtils.h:        CUDA_VERIFY(cudaDeviceSynchronize()); \
faiss/gpu/utils/DeviceUtils.h:#define CUDA_TEST_ERROR()                \
faiss/gpu/utils/DeviceUtils.h:        CUDA_VERIFY(cudaGetLastError()); \
faiss/gpu/utils/DeviceUtils.h:    std::vector<cudaEvent_t> events;
faiss/gpu/utils/DeviceUtils.h:        cudaEvent_t event;
faiss/gpu/utils/DeviceUtils.h:        CUDA_VERIFY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
faiss/gpu/utils/DeviceUtils.h:        CUDA_VERIFY(cudaEventRecord(event, stream));
faiss/gpu/utils/DeviceUtils.h:            CUDA_VERIFY(cudaStreamWaitEvent(stream, event, 0));
faiss/gpu/utils/DeviceUtils.h:        CUDA_VERIFY(cudaEventDestroy(event));
faiss/gpu/utils/DeviceUtils.h:void streamWait(const L1& a, const std::initializer_list<cudaStream_t>& b) {
faiss/gpu/utils/DeviceUtils.h:void streamWait(const std::initializer_list<cudaStream_t>& a, const L2& b) {
faiss/gpu/utils/DeviceUtils.h:        const std::initializer_list<cudaStream_t>& a,
faiss/gpu/utils/DeviceUtils.h:        const std::initializer_list<cudaStream_t>& b) {
faiss/gpu/utils/DeviceUtils.h:} // namespace gpu
faiss/gpu/utils/Timer.h:#include <cuda_runtime.h>
faiss/gpu/utils/Timer.h:namespace gpu {
faiss/gpu/utils/Timer.h:    KernelTimer(cudaStream_t stream = nullptr);
faiss/gpu/utils/Timer.h:    /// actual GPU-side kernel timings for any kernels launched in the
faiss/gpu/utils/Timer.h:    cudaEvent_t startEvent_;
faiss/gpu/utils/Timer.h:    cudaEvent_t stopEvent_;
faiss/gpu/utils/Timer.h:    cudaStream_t stream_;
faiss/gpu/utils/Timer.h:} // namespace gpu
faiss/gpu/utils/HostTensor-inl.cuh:namespace gpu {
faiss/gpu/utils/HostTensor-inl.cuh:        cudaStream_t stream)
faiss/gpu/utils/HostTensor-inl.cuh:} // namespace gpu
faiss/gpu/utils/StackDeviceMemory.h:#include <cuda_runtime.h>
faiss/gpu/utils/StackDeviceMemory.h:#include <faiss/gpu/GpuResources.h>
faiss/gpu/utils/StackDeviceMemory.h:namespace gpu {
faiss/gpu/utils/StackDeviceMemory.h:    StackDeviceMemory(GpuResources* res, int device, size_t allocPerDevice);
faiss/gpu/utils/StackDeviceMemory.h:    void* allocMemory(cudaStream_t stream, size_t size);
faiss/gpu/utils/StackDeviceMemory.h:    void deallocMemory(int device, cudaStream_t, size_t size, void* p);
faiss/gpu/utils/StackDeviceMemory.h:        inline Range(char* s, char* e, cudaStream_t str)
faiss/gpu/utils/StackDeviceMemory.h:        cudaStream_t stream_;
faiss/gpu/utils/StackDeviceMemory.h:        /// Constructor that allocates memory via cudaMalloc
faiss/gpu/utils/StackDeviceMemory.h:        Stack(GpuResources* res, int device, size_t size);
faiss/gpu/utils/StackDeviceMemory.h:        /// calling cudaMalloc
faiss/gpu/utils/StackDeviceMemory.h:        char* getAlloc(size_t size, cudaStream_t stream);
faiss/gpu/utils/StackDeviceMemory.h:        void returnAlloc(char* p, size_t size, cudaStream_t stream);
faiss/gpu/utils/StackDeviceMemory.h:        /// Our GpuResources object
faiss/gpu/utils/StackDeviceMemory.h:        GpuResources* res_;
faiss/gpu/utils/StackDeviceMemory.h:} // namespace gpu
faiss/gpu/utils/DeviceUtils.cu:#include <cuda_profiler_api.h>
faiss/gpu/utils/DeviceUtils.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/utils/DeviceUtils.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/utils/DeviceUtils.cu:namespace gpu {
faiss/gpu/utils/DeviceUtils.cu:    CUDA_VERIFY(cudaGetDevice(&dev));
faiss/gpu/utils/DeviceUtils.cu:    CUDA_VERIFY(cudaSetDevice(device));
faiss/gpu/utils/DeviceUtils.cu:    cudaError_t err = cudaGetDeviceCount(&numDev);
faiss/gpu/utils/DeviceUtils.cu:    if (cudaErrorNoDevice == err) {
faiss/gpu/utils/DeviceUtils.cu:        CUDA_VERIFY(err);
faiss/gpu/utils/DeviceUtils.cu:    CUDA_VERIFY(cudaProfilerStart());
faiss/gpu/utils/DeviceUtils.cu:    CUDA_VERIFY(cudaProfilerStop());
faiss/gpu/utils/DeviceUtils.cu:        CUDA_VERIFY(cudaDeviceSynchronize());
faiss/gpu/utils/DeviceUtils.cu:const cudaDeviceProp& getDeviceProperties(int device) {
faiss/gpu/utils/DeviceUtils.cu:    static std::unordered_map<int, cudaDeviceProp> properties;
faiss/gpu/utils/DeviceUtils.cu:        cudaDeviceProp prop;
faiss/gpu/utils/DeviceUtils.cu:        CUDA_VERIFY(cudaGetDeviceProperties(&prop, device));
faiss/gpu/utils/DeviceUtils.cu:const cudaDeviceProp& getCurrentDeviceProperties() {
faiss/gpu/utils/DeviceUtils.cu:    cudaPointerAttributes att;
faiss/gpu/utils/DeviceUtils.cu:    cudaError_t err = cudaPointerGetAttributes(&att, p);
faiss/gpu/utils/DeviceUtils.cu:            err == cudaSuccess || err == cudaErrorInvalidValue,
faiss/gpu/utils/DeviceUtils.cu:    if (err == cudaErrorInvalidValue) {
faiss/gpu/utils/DeviceUtils.cu:        err = cudaGetLastError();
faiss/gpu/utils/DeviceUtils.cu:                err == cudaErrorInvalidValue, "unknown error %d", (int)err);
faiss/gpu/utils/DeviceUtils.cu:#if USE_AMD_ROCM
faiss/gpu/utils/DeviceUtils.cu:    // memoryType is deprecated for CUDA 10.0+
faiss/gpu/utils/DeviceUtils.cu:#if CUDA_VERSION < 10000
faiss/gpu/utils/DeviceUtils.cu:    if (att.memoryType == cudaMemoryTypeHost) {
faiss/gpu/utils/DeviceUtils.cu:    if (att.type == cudaMemoryTypeDevice) {
faiss/gpu/utils/DeviceUtils.cu:    CUDA_VERIFY(cudaMemGetInfo(&free, &total));
faiss/gpu/utils/DeviceUtils.cu:    CUDA_VERIFY(cudaMemGetInfo(&free, &total));
faiss/gpu/utils/DeviceUtils.cu:CudaEvent::CudaEvent(cudaStream_t stream, bool timer) : event_(0) {
faiss/gpu/utils/DeviceUtils.cu:    CUDA_VERIFY(cudaEventCreateWithFlags(
faiss/gpu/utils/DeviceUtils.cu:            &event_, timer ? cudaEventDefault : cudaEventDisableTiming));
faiss/gpu/utils/DeviceUtils.cu:    CUDA_VERIFY(cudaEventRecord(event_, stream));
faiss/gpu/utils/DeviceUtils.cu:CudaEvent::CudaEvent(CudaEvent&& event) noexcept
faiss/gpu/utils/DeviceUtils.cu:CudaEvent::~CudaEvent() {
faiss/gpu/utils/DeviceUtils.cu:        CUDA_VERIFY(cudaEventDestroy(event_));
faiss/gpu/utils/DeviceUtils.cu:CudaEvent& CudaEvent::operator=(CudaEvent&& event) noexcept {
faiss/gpu/utils/DeviceUtils.cu:void CudaEvent::streamWaitOnEvent(cudaStream_t stream) {
faiss/gpu/utils/DeviceUtils.cu:    CUDA_VERIFY(cudaStreamWaitEvent(stream, event_, 0));
faiss/gpu/utils/DeviceUtils.cu:void CudaEvent::cpuWaitOnEvent() {
faiss/gpu/utils/DeviceUtils.cu:    CUDA_VERIFY(cudaEventSynchronize(event_));
faiss/gpu/utils/DeviceUtils.cu:} // namespace gpu
faiss/gpu/utils/DeviceVector.cuh:#include <cuda.h>
faiss/gpu/utils/DeviceVector.cuh:#include <faiss/gpu/GpuResources.h>
faiss/gpu/utils/DeviceVector.cuh:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/utils/DeviceVector.cuh:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/utils/DeviceVector.cuh:namespace gpu {
faiss/gpu/utils/DeviceVector.cuh:// For growing GPU allocations:
faiss/gpu/utils/DeviceVector.cuh:    DeviceVector(GpuResources* res, AllocInfo allocInfo)
faiss/gpu/utils/DeviceVector.cuh:    std::vector<OutT> copyToHost(cudaStream_t stream) const {
faiss/gpu/utils/DeviceVector.cuh:            CUDA_VERIFY(cudaMemcpyAsync(
faiss/gpu/utils/DeviceVector.cuh:                    cudaMemcpyDeviceToHost,
faiss/gpu/utils/DeviceVector.cuh:            cudaStream_t stream,
faiss/gpu/utils/DeviceVector.cuh:                CUDA_VERIFY(cudaMemcpyAsync(
faiss/gpu/utils/DeviceVector.cuh:                        cudaMemcpyHostToDevice,
faiss/gpu/utils/DeviceVector.cuh:                CUDA_VERIFY(cudaMemcpyAsync(
faiss/gpu/utils/DeviceVector.cuh:                        cudaMemcpyDeviceToDevice,
faiss/gpu/utils/DeviceVector.cuh:    bool resize(size_t newSize, cudaStream_t stream) {
faiss/gpu/utils/DeviceVector.cuh:    void setAll(const T& value, cudaStream_t stream) {
faiss/gpu/utils/DeviceVector.cuh:                    thrust::cuda::par.on(stream), data(), data() + num_, value);
faiss/gpu/utils/DeviceVector.cuh:    void setAt(size_t idx, const T& value, cudaStream_t stream) {
faiss/gpu/utils/DeviceVector.cuh:        CUDA_VERIFY(cudaMemcpyAsync(
faiss/gpu/utils/DeviceVector.cuh:                cudaMemcpyHostToDevice,
faiss/gpu/utils/DeviceVector.cuh:    T getAt(size_t idx, cudaStream_t stream) {
faiss/gpu/utils/DeviceVector.cuh:        CUDA_VERIFY(cudaMemcpyAsync(
faiss/gpu/utils/DeviceVector.cuh:                &out, data() + idx, sizeof(T), cudaMemcpyDeviceToHost, stream));
faiss/gpu/utils/DeviceVector.cuh:    size_t reclaim(bool exact, cudaStream_t stream) {
faiss/gpu/utils/DeviceVector.cuh:    bool reserve(size_t newCapacity, cudaStream_t stream) {
faiss/gpu/utils/DeviceVector.cuh:    void realloc_(size_t newCapacity, cudaStream_t stream) {
faiss/gpu/utils/DeviceVector.cuh:        CUDA_VERIFY(cudaMemcpyAsync(
faiss/gpu/utils/DeviceVector.cuh:                cudaMemcpyDeviceToDevice,
faiss/gpu/utils/DeviceVector.cuh:        CUDA_VERIFY(cudaMemsetAsync(
faiss/gpu/utils/DeviceVector.cuh:    GpuMemoryReservation alloc_;
faiss/gpu/utils/DeviceVector.cuh:    GpuResources* res_;
faiss/gpu/utils/DeviceVector.cuh:} // namespace gpu
faiss/gpu/utils/WarpPackedBits.cuh:#include <faiss/gpu/utils/PtxUtils.cuh>
faiss/gpu/utils/WarpPackedBits.cuh:#include <faiss/gpu/utils/WarpShuffles.cuh>
faiss/gpu/utils/WarpPackedBits.cuh:namespace gpu {
faiss/gpu/utils/WarpPackedBits.cuh:} // namespace gpu
faiss/gpu/utils/HostTensor.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/utils/HostTensor.cuh:namespace gpu {
faiss/gpu/utils/HostTensor.cuh:    /// locally. If the tensor is on the GPU, then we will copy it to
faiss/gpu/utils/HostTensor.cuh:            cudaStream_t stream);
faiss/gpu/utils/HostTensor.cuh:        /// cudaFree
faiss/gpu/utils/HostTensor.cuh:} // namespace gpu
faiss/gpu/utils/HostTensor.cuh:#include <faiss/gpu/utils/HostTensor-inl.cuh>
faiss/gpu/utils/BlockSelectKernel.cuh:#include <faiss/gpu/utils/Select.cuh>
faiss/gpu/utils/BlockSelectKernel.cuh:namespace gpu {
faiss/gpu/utils/BlockSelectKernel.cuh:        cudaStream_t stream);
faiss/gpu/utils/BlockSelectKernel.cuh:        cudaStream_t stream);
faiss/gpu/utils/BlockSelectKernel.cuh:        cudaStream_t stream);
faiss/gpu/utils/BlockSelectKernel.cuh:        cudaStream_t stream);
faiss/gpu/utils/BlockSelectKernel.cuh:} // namespace gpu
faiss/gpu/utils/DeviceTensor-inl.cuh:namespace gpu {
faiss/gpu/utils/DeviceTensor-inl.cuh:        GpuResources* res,
faiss/gpu/utils/DeviceTensor-inl.cuh:        GpuResources* res,
faiss/gpu/utils/DeviceTensor-inl.cuh:        GpuResources* res,
faiss/gpu/utils/DeviceTensor-inl.cuh:        PtrTraits>::zero(cudaStream_t stream) {
faiss/gpu/utils/DeviceTensor-inl.cuh:        CUDA_VERIFY(cudaMemsetAsync(
faiss/gpu/utils/DeviceTensor-inl.cuh:} // namespace gpu
faiss/gpu/utils/DeviceTensor.cuh:#include <faiss/gpu/GpuResources.h>
faiss/gpu/utils/DeviceTensor.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/utils/DeviceTensor.cuh:namespace gpu {
faiss/gpu/utils/DeviceTensor.cuh:            GpuResources* res,
faiss/gpu/utils/DeviceTensor.cuh:            GpuResources* res,
faiss/gpu/utils/DeviceTensor.cuh:            GpuResources* res,
faiss/gpu/utils/DeviceTensor.cuh:            cudaStream_t stream);
faiss/gpu/utils/DeviceTensor.cuh:    GpuMemoryReservation reservation_;
faiss/gpu/utils/DeviceTensor.cuh:} // namespace gpu
faiss/gpu/utils/DeviceTensor.cuh:#include <faiss/gpu/utils/DeviceTensor-inl.cuh>
faiss/gpu/utils/warpselect/WarpSelectFloat1.cu:#include <faiss/gpu/utils/warpselect/WarpSelectImpl.cuh>
faiss/gpu/utils/warpselect/WarpSelectFloat1.cu:namespace gpu {
faiss/gpu/utils/warpselect/WarpSelectFloat1.cu:} // namespace gpu
faiss/gpu/utils/warpselect/WarpSelectImpl.cuh:#include <faiss/gpu/utils/Limits.cuh>
faiss/gpu/utils/warpselect/WarpSelectImpl.cuh:#include <faiss/gpu/utils/WarpSelectKernel.cuh>
faiss/gpu/utils/warpselect/WarpSelectImpl.cuh:            cudaStream_t stream)
faiss/gpu/utils/warpselect/WarpSelectImpl.cuh:            cudaStream_t stream) {                                             \
faiss/gpu/utils/warpselect/WarpSelectImpl.cuh:        CUDA_TEST_ERROR();                                                     \
faiss/gpu/utils/warpselect/WarpSelectFloatT1024.cu:#include <faiss/gpu/utils/warpselect/WarpSelectImpl.cuh>
faiss/gpu/utils/warpselect/WarpSelectFloatT1024.cu:namespace gpu {
faiss/gpu/utils/warpselect/WarpSelectFloatF1024.cu:#include <faiss/gpu/utils/warpselect/WarpSelectImpl.cuh>
faiss/gpu/utils/warpselect/WarpSelectFloatF1024.cu:namespace gpu {
faiss/gpu/utils/warpselect/WarpSelectFloat128.cu:#include <faiss/gpu/utils/warpselect/WarpSelectImpl.cuh>
faiss/gpu/utils/warpselect/WarpSelectFloat128.cu:namespace gpu {
faiss/gpu/utils/warpselect/WarpSelectFloat128.cu:} // namespace gpu
faiss/gpu/utils/warpselect/WarpSelectFloat256.cu:#include <faiss/gpu/utils/warpselect/WarpSelectImpl.cuh>
faiss/gpu/utils/warpselect/WarpSelectFloat256.cu:namespace gpu {
faiss/gpu/utils/warpselect/WarpSelectFloat256.cu:} // namespace gpu
faiss/gpu/utils/warpselect/WarpSelectFloatT512.cu:#include <faiss/gpu/utils/warpselect/WarpSelectImpl.cuh>
faiss/gpu/utils/warpselect/WarpSelectFloatT512.cu:namespace gpu {
faiss/gpu/utils/warpselect/WarpSelectFloat64.cu:#include <faiss/gpu/utils/warpselect/WarpSelectImpl.cuh>
faiss/gpu/utils/warpselect/WarpSelectFloat64.cu:namespace gpu {
faiss/gpu/utils/warpselect/WarpSelectFloat64.cu:} // namespace gpu
faiss/gpu/utils/warpselect/WarpSelectFloatF2048.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/utils/warpselect/WarpSelectFloatF2048.cu:#include <faiss/gpu/utils/warpselect/WarpSelectImpl.cuh>
faiss/gpu/utils/warpselect/WarpSelectFloatF2048.cu:namespace gpu {
faiss/gpu/utils/warpselect/WarpSelectFloatF2048.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/utils/warpselect/WarpSelectFloatF2048.cu:} // namespace gpu
faiss/gpu/utils/warpselect/WarpSelectFloatT2048.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/utils/warpselect/WarpSelectFloatT2048.cu:#include <faiss/gpu/utils/warpselect/WarpSelectImpl.cuh>
faiss/gpu/utils/warpselect/WarpSelectFloatT2048.cu:namespace gpu {
faiss/gpu/utils/warpselect/WarpSelectFloatT2048.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/utils/warpselect/WarpSelectFloatT2048.cu:} // namespace gpu
faiss/gpu/utils/warpselect/WarpSelectFloat32.cu:#include <faiss/gpu/utils/warpselect/WarpSelectImpl.cuh>
faiss/gpu/utils/warpselect/WarpSelectFloat32.cu:namespace gpu {
faiss/gpu/utils/warpselect/WarpSelectFloat32.cu:} // namespace gpu
faiss/gpu/utils/warpselect/WarpSelectFloatF512.cu:#include <faiss/gpu/utils/warpselect/WarpSelectImpl.cuh>
faiss/gpu/utils/warpselect/WarpSelectFloatF512.cu:namespace gpu {
faiss/gpu/utils/NoTypeTensor.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/utils/NoTypeTensor.cuh:namespace gpu {
faiss/gpu/utils/NoTypeTensor.cuh:} // namespace gpu
faiss/gpu/utils/MatrixMult-inl.cuh:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/utils/MatrixMult-inl.cuh:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/utils/MatrixMult-inl.cuh:#include <faiss/gpu/utils/HostTensor.cuh>
faiss/gpu/utils/MatrixMult-inl.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/utils/MatrixMult-inl.cuh:namespace gpu {
faiss/gpu/utils/MatrixMult-inl.cuh:struct GetCudaType;
faiss/gpu/utils/MatrixMult-inl.cuh:#ifdef USE_AMD_ROCM
faiss/gpu/utils/MatrixMult-inl.cuh:struct GetCudaType<float> {
faiss/gpu/utils/MatrixMult-inl.cuh:struct GetCudaType<half> {
faiss/gpu/utils/MatrixMult-inl.cuh:struct GetCudaType<float> {
faiss/gpu/utils/MatrixMult-inl.cuh:    static constexpr cudaDataType_t Type = CUDA_R_32F;
faiss/gpu/utils/MatrixMult-inl.cuh:struct GetCudaType<half> {
faiss/gpu/utils/MatrixMult-inl.cuh:    static constexpr cudaDataType_t Type = CUDA_R_16F;
faiss/gpu/utils/MatrixMult-inl.cuh:    auto cAT = GetCudaType<AT>::Type;
faiss/gpu/utils/MatrixMult-inl.cuh:    auto cBT = GetCudaType<BT>::Type;
faiss/gpu/utils/MatrixMult-inl.cuh:#ifdef USE_AMD_ROCM
faiss/gpu/utils/MatrixMult-inl.cuh:    // FIXME: some weird CUDA 11 bug? where cublasSgemmEx on
faiss/gpu/utils/MatrixMult-inl.cuh:    // and we are on CUDA 11+
faiss/gpu/utils/MatrixMult-inl.cuh:#if CUDA_VERSION >= 11000
faiss/gpu/utils/MatrixMult-inl.cuh:    if (cAT == CUDA_R_16F || cBT == CUDA_R_16F) {
faiss/gpu/utils/MatrixMult-inl.cuh:                CUDA_R_32F,
faiss/gpu/utils/MatrixMult-inl.cuh:            CUDA_R_32F,
faiss/gpu/utils/MatrixMult-inl.cuh:#endif // USE_AMD_ROCM
faiss/gpu/utils/MatrixMult-inl.cuh:    auto cAT = GetCudaType<AT>::Type;
faiss/gpu/utils/MatrixMult-inl.cuh:    auto cBT = GetCudaType<BT>::Type;
faiss/gpu/utils/MatrixMult-inl.cuh:#ifdef USE_AMD_ROCM
faiss/gpu/utils/MatrixMult-inl.cuh:            CUDA_R_32F,
faiss/gpu/utils/MatrixMult-inl.cuh:            CUDA_R_32F,
faiss/gpu/utils/MatrixMult-inl.cuh:        cudaStream_t stream) {
faiss/gpu/utils/MatrixMult-inl.cuh:    CUDA_TEST_ERROR();
faiss/gpu/utils/MatrixMult-inl.cuh:        cudaStream_t stream) {
faiss/gpu/utils/MatrixMult-inl.cuh:    CUDA_TEST_ERROR();
faiss/gpu/utils/MatrixMult-inl.cuh:} // namespace gpu
faiss/gpu/utils/WarpSelectFloat.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/utils/WarpSelectFloat.cu:#include <faiss/gpu/utils/warpselect/WarpSelectImpl.cuh>
faiss/gpu/utils/WarpSelectFloat.cu:namespace gpu {
faiss/gpu/utils/WarpSelectFloat.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/utils/WarpSelectFloat.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/utils/WarpSelectFloat.cu:        cudaStream_t stream) {
faiss/gpu/utils/WarpSelectFloat.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/utils/WarpSelectFloat.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/utils/WarpSelectFloat.cu:} // namespace gpu
faiss/gpu/utils/Timer.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/utils/Timer.cpp:#include <faiss/gpu/utils/Timer.h>
faiss/gpu/utils/Timer.cpp:namespace gpu {
faiss/gpu/utils/Timer.cpp:KernelTimer::KernelTimer(cudaStream_t stream)
faiss/gpu/utils/Timer.cpp:    CUDA_VERIFY(cudaEventCreate(&startEvent_));
faiss/gpu/utils/Timer.cpp:    CUDA_VERIFY(cudaEventCreate(&stopEvent_));
faiss/gpu/utils/Timer.cpp:    CUDA_VERIFY(cudaEventRecord(startEvent_, stream_));
faiss/gpu/utils/Timer.cpp:    CUDA_VERIFY(cudaEventDestroy(startEvent_));
faiss/gpu/utils/Timer.cpp:    CUDA_VERIFY(cudaEventDestroy(stopEvent_));
faiss/gpu/utils/Timer.cpp:    CUDA_VERIFY(cudaEventRecord(stopEvent_, stream_));
faiss/gpu/utils/Timer.cpp:    CUDA_VERIFY(cudaEventSynchronize(stopEvent_));
faiss/gpu/utils/Timer.cpp:    CUDA_VERIFY(cudaEventElapsedTime(&time, startEvent_, stopEvent_));
faiss/gpu/utils/Timer.cpp:} // namespace gpu
faiss/gpu/utils/ConversionOperators.cuh:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/utils/ConversionOperators.cuh:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/utils/ConversionOperators.cuh:#include <cuda.h>
faiss/gpu/utils/ConversionOperators.cuh:namespace gpu {
faiss/gpu/utils/ConversionOperators.cuh:void runConvert(const From* in, To* out, size_t num, cudaStream_t stream) {
faiss/gpu/utils/ConversionOperators.cuh:            thrust::cuda::par.on(stream),
faiss/gpu/utils/ConversionOperators.cuh:        cudaStream_t stream,
faiss/gpu/utils/ConversionOperators.cuh:        GpuResources* res,
faiss/gpu/utils/ConversionOperators.cuh:        cudaStream_t stream,
faiss/gpu/utils/ConversionOperators.cuh:        GpuResources* res,
faiss/gpu/utils/ConversionOperators.cuh:        cudaStream_t stream,
faiss/gpu/utils/ConversionOperators.cuh:} // namespace gpu
faiss/gpu/utils/MathOperators.cuh:#include <faiss/gpu/utils/ConversionOperators.cuh>
faiss/gpu/utils/MathOperators.cuh:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/utils/MathOperators.cuh:namespace gpu {
faiss/gpu/utils/MathOperators.cuh:#if CUDA_VERSION >= 9000 || defined(USE_AMD_ROCM)
faiss/gpu/utils/MathOperators.cuh:} // namespace gpu
faiss/gpu/utils/CopyUtils.cuh:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/utils/CopyUtils.cuh:#include <faiss/gpu/utils/HostTensor.cuh>
faiss/gpu/utils/CopyUtils.cuh:namespace gpu {
faiss/gpu/utils/CopyUtils.cuh:        GpuResources* resources,
faiss/gpu/utils/CopyUtils.cuh:        cudaStream_t stream,
faiss/gpu/utils/CopyUtils.cuh:        GpuResources* resources,
faiss/gpu/utils/CopyUtils.cuh:        cudaStream_t stream,
faiss/gpu/utils/CopyUtils.cuh:        GpuResources* resources,
faiss/gpu/utils/CopyUtils.cuh:        cudaStream_t stream,
faiss/gpu/utils/CopyUtils.cuh:        cudaStream_t stream,
faiss/gpu/utils/CopyUtils.cuh:inline void fromDevice(T* src, T* dst, size_t num, cudaStream_t stream) {
faiss/gpu/utils/CopyUtils.cuh:        CUDA_VERIFY(cudaMemcpyAsync(
faiss/gpu/utils/CopyUtils.cuh:                dst, src, num * sizeof(T), cudaMemcpyDeviceToHost, stream));
faiss/gpu/utils/CopyUtils.cuh:        cudaStreamSynchronize(stream);
faiss/gpu/utils/CopyUtils.cuh:        CUDA_VERIFY(cudaMemcpyAsync(
faiss/gpu/utils/CopyUtils.cuh:                dst, src, num * sizeof(T), cudaMemcpyDeviceToDevice, stream));
faiss/gpu/utils/CopyUtils.cuh:void fromDevice(Tensor<T, Dim, true>& src, T* dst, cudaStream_t stream) {
faiss/gpu/utils/CopyUtils.cuh:} // namespace gpu
faiss/gpu/utils/PtxUtils.cuh:#include <cuda.h>
faiss/gpu/utils/PtxUtils.cuh:#ifdef USE_AMD_ROCM
faiss/gpu/utils/PtxUtils.cuh:namespace gpu {
faiss/gpu/utils/PtxUtils.cuh:#ifdef USE_AMD_ROCM
faiss/gpu/utils/PtxUtils.cuh:#else // USE_AMD_ROCM
faiss/gpu/utils/PtxUtils.cuh:#endif // USE_AMD_ROCM
faiss/gpu/utils/PtxUtils.cuh:} // namespace gpu
faiss/gpu/utils/MergeNetworkWarp.cuh:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/utils/MergeNetworkWarp.cuh:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/utils/MergeNetworkWarp.cuh:#include <faiss/gpu/utils/MergeNetworkUtils.cuh>
faiss/gpu/utils/MergeNetworkWarp.cuh:#include <faiss/gpu/utils/PtxUtils.cuh>
faiss/gpu/utils/MergeNetworkWarp.cuh:#include <faiss/gpu/utils/WarpShuffles.cuh>
faiss/gpu/utils/MergeNetworkWarp.cuh:namespace gpu {
faiss/gpu/utils/MergeNetworkWarp.cuh:} // namespace gpu
faiss/gpu/utils/Reductions.cuh:#include <cuda.h>
faiss/gpu/utils/Reductions.cuh:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/utils/Reductions.cuh:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/utils/Reductions.cuh:#include <faiss/gpu/utils/PtxUtils.cuh>
faiss/gpu/utils/Reductions.cuh:#include <faiss/gpu/utils/ReductionOperators.cuh>
faiss/gpu/utils/Reductions.cuh:#include <faiss/gpu/utils/WarpShuffles.cuh>
faiss/gpu/utils/Reductions.cuh:namespace gpu {
faiss/gpu/utils/Reductions.cuh:} // namespace gpu
faiss/gpu/utils/Pair.cuh:#include <cuda.h>
faiss/gpu/utils/Pair.cuh:#include <faiss/gpu/utils/MathOperators.cuh>
faiss/gpu/utils/Pair.cuh:#include <faiss/gpu/utils/WarpShuffles.cuh>
faiss/gpu/utils/Pair.cuh:namespace gpu {
faiss/gpu/utils/Pair.cuh:/// A simple pair type for CUDA device usage
faiss/gpu/utils/Pair.cuh:} // namespace gpu
faiss/gpu/utils/MergeNetworkUtils.cuh:namespace gpu {
faiss/gpu/utils/MergeNetworkUtils.cuh:} // namespace gpu
faiss/gpu/utils/Transpose.cuh:#include <cuda.h>
faiss/gpu/utils/Transpose.cuh:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/utils/Transpose.cuh:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/utils/Transpose.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/utils/Transpose.cuh:namespace gpu {
faiss/gpu/utils/Transpose.cuh:#if __CUDA_ARCH__ >= 350 || defined(USE_AMD_ROCM)
faiss/gpu/utils/Transpose.cuh:        cudaStream_t stream) {
faiss/gpu/utils/Transpose.cuh:    CUDA_TEST_ERROR();
faiss/gpu/utils/Transpose.cuh:} // namespace gpu
faiss/gpu/utils/RaftUtils.h: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/utils/RaftUtils.h:#include <faiss/gpu/GpuResources.h>
faiss/gpu/utils/RaftUtils.h:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/utils/RaftUtils.h:namespace gpu {
faiss/gpu/utils/RaftUtils.h:        GpuResources* res,
faiss/gpu/utils/RaftUtils.h:        GpuResources* res,
faiss/gpu/utils/RaftUtils.h:} // namespace gpu
faiss/gpu/utils/Comparators.cuh:#include <cuda.h>
faiss/gpu/utils/Comparators.cuh:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/utils/Comparators.cuh:namespace gpu {
faiss/gpu/utils/Comparators.cuh:} // namespace gpu
faiss/gpu/utils/BlockSelectFloat.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/utils/BlockSelectFloat.cu:#include <faiss/gpu/utils/blockselect/BlockSelectImpl.cuh>
faiss/gpu/utils/BlockSelectFloat.cu:namespace gpu {
faiss/gpu/utils/BlockSelectFloat.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/utils/BlockSelectFloat.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/utils/BlockSelectFloat.cu:        cudaStream_t stream) {
faiss/gpu/utils/BlockSelectFloat.cu:    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);
faiss/gpu/utils/BlockSelectFloat.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/utils/BlockSelectFloat.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/utils/BlockSelectFloat.cu:        cudaStream_t stream) {
faiss/gpu/utils/BlockSelectFloat.cu:    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);
faiss/gpu/utils/BlockSelectFloat.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/utils/BlockSelectFloat.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/utils/BlockSelectFloat.cu:} // namespace gpu
faiss/gpu/utils/MergeNetworkBlock.cuh:#include <cuda.h>
faiss/gpu/utils/MergeNetworkBlock.cuh:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/utils/MergeNetworkBlock.cuh:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/utils/MergeNetworkBlock.cuh:#include <faiss/gpu/utils/MergeNetworkUtils.cuh>
faiss/gpu/utils/MergeNetworkBlock.cuh:#include <faiss/gpu/utils/PtxUtils.cuh>
faiss/gpu/utils/MergeNetworkBlock.cuh:#include <faiss/gpu/utils/WarpShuffles.cuh>
faiss/gpu/utils/MergeNetworkBlock.cuh:namespace gpu {
faiss/gpu/utils/MergeNetworkBlock.cuh:        // FIXME: is this a CUDA 9 compiler bug?
faiss/gpu/utils/MergeNetworkBlock.cuh:            // FIXME: is this a CUDA 9 compiler bug?
faiss/gpu/utils/MergeNetworkBlock.cuh:        // FIXME: is this a CUDA 9 compiler bug?
faiss/gpu/utils/MergeNetworkBlock.cuh:            // FIXME: is this a CUDA 9 compiler bug?
faiss/gpu/utils/MergeNetworkBlock.cuh:} // namespace gpu
faiss/gpu/utils/WarpShuffles.cuh:#include <cuda.h>
faiss/gpu/utils/WarpShuffles.cuh:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/utils/WarpShuffles.cuh:namespace gpu {
faiss/gpu/utils/WarpShuffles.cuh:#if CUDA_VERSION >= 9000
faiss/gpu/utils/WarpShuffles.cuh:#if CUDA_VERSION >= 9000
faiss/gpu/utils/WarpShuffles.cuh:// CUDA SDK does not provide specializations for T*
faiss/gpu/utils/WarpShuffles.cuh:#if CUDA_VERSION >= 9000
faiss/gpu/utils/WarpShuffles.cuh:// CUDA SDK does not provide specializations for T*
faiss/gpu/utils/WarpShuffles.cuh:#if CUDA_VERSION >= 9000
faiss/gpu/utils/WarpShuffles.cuh:// CUDA SDK does not provide specializations for T*
faiss/gpu/utils/WarpShuffles.cuh:#if CUDA_VERSION >= 9000
faiss/gpu/utils/WarpShuffles.cuh:// CUDA SDK does not provide specializations for T*
faiss/gpu/utils/WarpShuffles.cuh:#ifdef USE_AMD_ROCM
faiss/gpu/utils/WarpShuffles.cuh:// CUDA 9.0+ has half shuffle
faiss/gpu/utils/WarpShuffles.cuh:#if CUDA_VERSION < 9000
faiss/gpu/utils/WarpShuffles.cuh:#endif // CUDA_VERSION
faiss/gpu/utils/WarpShuffles.cuh:#endif // USE_AMD_ROCM
faiss/gpu/utils/WarpShuffles.cuh:} // namespace gpu
faiss/gpu/utils/blockselect/BlockSelectFloat1.cu:#include <faiss/gpu/utils/blockselect/BlockSelectImpl.cuh>
faiss/gpu/utils/blockselect/BlockSelectFloat1.cu:namespace gpu {
faiss/gpu/utils/blockselect/BlockSelectFloat1.cu:} // namespace gpu
faiss/gpu/utils/blockselect/BlockSelectFloat128.cu:#include <faiss/gpu/utils/blockselect/BlockSelectImpl.cuh>
faiss/gpu/utils/blockselect/BlockSelectFloat128.cu:namespace gpu {
faiss/gpu/utils/blockselect/BlockSelectFloat128.cu:} // namespace gpu
faiss/gpu/utils/blockselect/BlockSelectFloat256.cu:#include <faiss/gpu/utils/blockselect/BlockSelectImpl.cuh>
faiss/gpu/utils/blockselect/BlockSelectFloat256.cu:namespace gpu {
faiss/gpu/utils/blockselect/BlockSelectFloat256.cu:} // namespace gpu
faiss/gpu/utils/blockselect/BlockSelectFloatF2048.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/utils/blockselect/BlockSelectFloatF2048.cu:#include <faiss/gpu/utils/blockselect/BlockSelectImpl.cuh>
faiss/gpu/utils/blockselect/BlockSelectFloatF2048.cu:namespace gpu {
faiss/gpu/utils/blockselect/BlockSelectFloatF2048.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/utils/blockselect/BlockSelectFloatF2048.cu:} // namespace gpu
faiss/gpu/utils/blockselect/BlockSelectFloatF512.cu:#include <faiss/gpu/utils/blockselect/BlockSelectImpl.cuh>
faiss/gpu/utils/blockselect/BlockSelectFloatF512.cu:namespace gpu {
faiss/gpu/utils/blockselect/BlockSelectFloatF1024.cu:#include <faiss/gpu/utils/blockselect/BlockSelectImpl.cuh>
faiss/gpu/utils/blockselect/BlockSelectFloatF1024.cu:namespace gpu {
faiss/gpu/utils/blockselect/BlockSelectFloatT2048.cu:#include <faiss/gpu/utils/DeviceDefs.cuh>
faiss/gpu/utils/blockselect/BlockSelectFloatT2048.cu:#include <faiss/gpu/utils/blockselect/BlockSelectImpl.cuh>
faiss/gpu/utils/blockselect/BlockSelectFloatT2048.cu:namespace gpu {
faiss/gpu/utils/blockselect/BlockSelectFloatT2048.cu:#if GPU_MAX_SELECTION_K >= 2048
faiss/gpu/utils/blockselect/BlockSelectFloatT2048.cu:} // namespace gpu
faiss/gpu/utils/blockselect/BlockSelectImpl.cuh:#include <faiss/gpu/utils/BlockSelectKernel.cuh>
faiss/gpu/utils/blockselect/BlockSelectImpl.cuh:#include <faiss/gpu/utils/Limits.cuh>
faiss/gpu/utils/blockselect/BlockSelectImpl.cuh:            cudaStream_t stream);                                \
faiss/gpu/utils/blockselect/BlockSelectImpl.cuh:            cudaStream_t stream)
faiss/gpu/utils/blockselect/BlockSelectImpl.cuh:            cudaStream_t stream) {                                             \
faiss/gpu/utils/blockselect/BlockSelectImpl.cuh:        CUDA_TEST_ERROR();                                                     \
faiss/gpu/utils/blockselect/BlockSelectImpl.cuh:            cudaStream_t stream) {                                             \
faiss/gpu/utils/blockselect/BlockSelectImpl.cuh:        CUDA_TEST_ERROR();                                                     \
faiss/gpu/utils/blockselect/BlockSelectFloatT1024.cu:#include <faiss/gpu/utils/blockselect/BlockSelectImpl.cuh>
faiss/gpu/utils/blockselect/BlockSelectFloatT1024.cu:namespace gpu {
faiss/gpu/utils/blockselect/BlockSelectFloat64.cu:#include <faiss/gpu/utils/blockselect/BlockSelectImpl.cuh>
faiss/gpu/utils/blockselect/BlockSelectFloat64.cu:namespace gpu {
faiss/gpu/utils/blockselect/BlockSelectFloat64.cu:} // namespace gpu
faiss/gpu/utils/blockselect/BlockSelectFloat32.cu:#include <faiss/gpu/utils/blockselect/BlockSelectImpl.cuh>
faiss/gpu/utils/blockselect/BlockSelectFloat32.cu:namespace gpu {
faiss/gpu/utils/blockselect/BlockSelectFloat32.cu:} // namespace gpu
faiss/gpu/utils/blockselect/BlockSelectFloatT512.cu:#include <faiss/gpu/utils/blockselect/BlockSelectImpl.cuh>
faiss/gpu/utils/blockselect/BlockSelectFloatT512.cu:namespace gpu {
faiss/gpu/utils/StackDeviceMemory.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/utils/StackDeviceMemory.cpp:#include <faiss/gpu/utils/StackDeviceMemory.h>
faiss/gpu/utils/StackDeviceMemory.cpp:#include <faiss/gpu/utils/StaticUtils.h>
faiss/gpu/utils/StackDeviceMemory.cpp:namespace gpu {
faiss/gpu/utils/StackDeviceMemory.cpp:StackDeviceMemory::Stack::Stack(GpuResources* res, int d, size_t sz)
faiss/gpu/utils/StackDeviceMemory.cpp:char* StackDeviceMemory::Stack::getAlloc(size_t size, cudaStream_t stream) {
faiss/gpu/utils/StackDeviceMemory.cpp:        cudaStream_t stream) {
faiss/gpu/utils/StackDeviceMemory.cpp:        GpuResources* res,
faiss/gpu/utils/StackDeviceMemory.cpp:void* StackDeviceMemory::allocMemory(cudaStream_t stream, size_t size) {
faiss/gpu/utils/StackDeviceMemory.cpp:        cudaStream_t stream,
faiss/gpu/utils/StackDeviceMemory.cpp:} // namespace gpu
faiss/gpu/utils/Tensor-inl.cuh:#include <faiss/gpu/GpuFaissAssert.h>
faiss/gpu/utils/Tensor-inl.cuh:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/utils/Tensor-inl.cuh:namespace gpu {
faiss/gpu/utils/Tensor-inl.cuh:    GPU_FAISS_ASSERT(sizes.size() == Dim);
faiss/gpu/utils/Tensor-inl.cuh:        cudaStream_t stream) {
faiss/gpu/utils/Tensor-inl.cuh:    GPU_FAISS_ASSERT(this->isContiguous());
faiss/gpu/utils/Tensor-inl.cuh:    GPU_FAISS_ASSERT(this->numElements() == t.numElements());
faiss/gpu/utils/Tensor-inl.cuh:        GPU_FAISS_ASSERT(this->data_);
faiss/gpu/utils/Tensor-inl.cuh:        GPU_FAISS_ASSERT(t.data());
faiss/gpu/utils/Tensor-inl.cuh:            CUDA_VERIFY(cudaMemcpyAsync(
faiss/gpu/utils/Tensor-inl.cuh:                    ourDev == -1 ? cudaMemcpyHostToHost
faiss/gpu/utils/Tensor-inl.cuh:                                 : cudaMemcpyHostToDevice,
faiss/gpu/utils/Tensor-inl.cuh:            CUDA_VERIFY(cudaMemcpyAsync(
faiss/gpu/utils/Tensor-inl.cuh:                    ourDev == -1 ? cudaMemcpyDeviceToHost
faiss/gpu/utils/Tensor-inl.cuh:                                 : cudaMemcpyDeviceToDevice,
faiss/gpu/utils/Tensor-inl.cuh:        cudaStream_t stream) {
faiss/gpu/utils/Tensor-inl.cuh:    GPU_FAISS_ASSERT(this->isContiguous());
faiss/gpu/utils/Tensor-inl.cuh:    GPU_FAISS_ASSERT(this->numElements() == t.numElements());
faiss/gpu/utils/Tensor-inl.cuh:        GPU_FAISS_ASSERT(this->data_);
faiss/gpu/utils/Tensor-inl.cuh:        GPU_FAISS_ASSERT(t.data());
faiss/gpu/utils/Tensor-inl.cuh:            CUDA_VERIFY(cudaMemcpyAsync(
faiss/gpu/utils/Tensor-inl.cuh:                    ourDev == -1 ? cudaMemcpyHostToHost
faiss/gpu/utils/Tensor-inl.cuh:                                 : cudaMemcpyDeviceToHost,
faiss/gpu/utils/Tensor-inl.cuh:            CUDA_VERIFY(cudaMemcpyAsync(
faiss/gpu/utils/Tensor-inl.cuh:                    ourDev == -1 ? cudaMemcpyHostToDevice
faiss/gpu/utils/Tensor-inl.cuh:                                 : cudaMemcpyDeviceToDevice,
faiss/gpu/utils/Tensor-inl.cuh:        cudaStream_t stream) {
faiss/gpu/utils/Tensor-inl.cuh:    GPU_FAISS_ASSERT(this->isContiguous());
faiss/gpu/utils/Tensor-inl.cuh:    GPU_FAISS_ASSERT(this->numElements() == v.size());
faiss/gpu/utils/Tensor-inl.cuh:        GPU_FAISS_ASSERT(this->data_);
faiss/gpu/utils/Tensor-inl.cuh:        CUDA_VERIFY(cudaMemcpyAsync(
faiss/gpu/utils/Tensor-inl.cuh:                ourDev == -1 ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice,
faiss/gpu/utils/Tensor-inl.cuh:        copyToVector(cudaStream_t stream) {
faiss/gpu/utils/Tensor-inl.cuh:    GPU_FAISS_ASSERT(this->isContiguous());
faiss/gpu/utils/Tensor-inl.cuh:            CUDA_VERIFY(cudaMemcpyAsync(
faiss/gpu/utils/Tensor-inl.cuh:                    cudaMemcpyDeviceToHost,
faiss/gpu/utils/Tensor-inl.cuh:    GPU_FAISS_ASSERT(canCastResize<U>());
faiss/gpu/utils/Tensor-inl.cuh:        GPU_FAISS_ASSERT(this->canUseIndexType<NewIndexT>());
faiss/gpu/utils/Tensor-inl.cuh:    GPU_FAISS_ASSERT(dim1 >= 0 && dim1 < Dim);
faiss/gpu/utils/Tensor-inl.cuh:    GPU_FAISS_ASSERT(dim2 >= 0 && dim2 < Dim);
faiss/gpu/utils/Tensor-inl.cuh:        GPU_FAISS_ASSERT(dim1 != Dim - 1 && dim2 != Dim - 1);
faiss/gpu/utils/Tensor-inl.cuh:    GPU_FAISS_ASSERT(dim1 >= 0 && dim1 < Dim);
faiss/gpu/utils/Tensor-inl.cuh:        GPU_FAISS_ASSERT(cont);
faiss/gpu/utils/Tensor-inl.cuh:        GPU_FAISS_ASSERT(isContiguousDim(i));
faiss/gpu/utils/Tensor-inl.cuh:    GPU_FAISS_ASSERT(
faiss/gpu/utils/Tensor-inl.cuh:            GPU_FAISS_ASSERT(start + size <= size_[dim]);
faiss/gpu/utils/Tensor-inl.cuh:    GPU_FAISS_ASSERT(this->isContiguous());
faiss/gpu/utils/Tensor-inl.cuh:    GPU_FAISS_ASSERT(sizes.size() == NewDim);
faiss/gpu/utils/Tensor-inl.cuh:    GPU_FAISS_ASSERT(curSize == newSize);
faiss/gpu/utils/Tensor-inl.cuh:} // namespace gpu
faiss/gpu/utils/RaftUtils.cu: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/utils/RaftUtils.cu:#include <faiss/gpu/GpuIndex.h>
faiss/gpu/utils/RaftUtils.cu:#include <faiss/gpu/utils/RaftUtils.h>
faiss/gpu/utils/RaftUtils.cu:namespace gpu {
faiss/gpu/utils/RaftUtils.cu:        GpuResources* res,
faiss/gpu/utils/RaftUtils.cu:        GpuResources* res,
faiss/gpu/utils/RaftUtils.cu:} // namespace gpu
faiss/gpu/utils/ReductionOperators.cuh:#include <cuda.h>
faiss/gpu/utils/ReductionOperators.cuh:#include <faiss/gpu/utils/Limits.cuh>
faiss/gpu/utils/ReductionOperators.cuh:#include <faiss/gpu/utils/MathOperators.cuh>
faiss/gpu/utils/ReductionOperators.cuh:#include <faiss/gpu/utils/Pair.cuh>
faiss/gpu/utils/ReductionOperators.cuh:namespace gpu {
faiss/gpu/utils/ReductionOperators.cuh:} // namespace gpu
faiss/gpu/utils/Float16.cuh:#include <cuda.h>
faiss/gpu/utils/Float16.cuh:#include <faiss/gpu/GpuResources.h>
faiss/gpu/utils/Float16.cuh:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/utils/Float16.cuh:#if __CUDA_ARCH__ >= 530 || defined(USE_AMD_ROCM)
faiss/gpu/utils/Float16.cuh:#endif // __CUDA_ARCH__ types
faiss/gpu/utils/Float16.cuh:#include <cuda_fp16.h>
faiss/gpu/utils/Float16.cuh:namespace gpu {
faiss/gpu/utils/Float16.cuh:} // namespace gpu
faiss/gpu/utils/LoadStoreOperators.cuh:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/utils/LoadStoreOperators.cuh:// cuda_fp16.hpp doesn't export this
faiss/gpu/utils/LoadStoreOperators.cuh:namespace gpu {
faiss/gpu/utils/LoadStoreOperators.cuh:#ifdef USE_AMD_ROCM
faiss/gpu/utils/LoadStoreOperators.cuh:#else // USE_AMD_ROCM
faiss/gpu/utils/LoadStoreOperators.cuh:#if CUDA_VERSION >= 9000
faiss/gpu/utils/LoadStoreOperators.cuh:#if CUDA_VERSION >= 9000
faiss/gpu/utils/LoadStoreOperators.cuh:#if CUDA_VERSION >= 9000
faiss/gpu/utils/LoadStoreOperators.cuh:#if CUDA_VERSION >= 9000
faiss/gpu/utils/LoadStoreOperators.cuh:#endif // USE_AMD_ROCM
faiss/gpu/utils/LoadStoreOperators.cuh:} // namespace gpu
faiss/gpu/utils/Limits.cuh:#include <faiss/gpu/utils/Pair.cuh>
faiss/gpu/utils/Limits.cuh:namespace gpu {
faiss/gpu/utils/Limits.cuh:#if CUDA_VERSION >= 9000 || defined(USE_AMD_ROCM)
faiss/gpu/utils/Limits.cuh:} // namespace gpu
faiss/gpu/utils/MatrixMult.cuh:#include <faiss/gpu/utils/DeviceTensor.cuh>
faiss/gpu/utils/MatrixMult.cuh:#include <faiss/gpu/utils/Float16.cuh>
faiss/gpu/utils/MatrixMult.cuh:#include <faiss/gpu/utils/HostTensor.cuh>
faiss/gpu/utils/MatrixMult.cuh:#include <faiss/gpu/utils/Tensor.cuh>
faiss/gpu/utils/MatrixMult.cuh:namespace gpu {
faiss/gpu/utils/MatrixMult.cuh:class GpuResources;
faiss/gpu/utils/MatrixMult.cuh:        cudaStream_t stream);
faiss/gpu/utils/MatrixMult.cuh:        cudaStream_t stream);
faiss/gpu/utils/MatrixMult.cuh:} // namespace gpu
faiss/gpu/utils/MatrixMult.cuh:#include <faiss/gpu/utils/MatrixMult-inl.cuh>
faiss/gpu/GpuIndexIVF.cu:#include <faiss/gpu/GpuCloner.h>
faiss/gpu/GpuIndexIVF.cu:#include <faiss/gpu/GpuIndexFlat.h>
faiss/gpu/GpuIndexIVF.cu:#include <faiss/gpu/GpuIndexIVF.h>
faiss/gpu/GpuIndexIVF.cu:#include <faiss/gpu/impl/IndexUtils.h>
faiss/gpu/GpuIndexIVF.cu:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/GpuIndexIVF.cu:#include <faiss/gpu/impl/IVFBase.cuh>
faiss/gpu/GpuIndexIVF.cu:#include <faiss/gpu/utils/CopyUtils.cuh>
faiss/gpu/GpuIndexIVF.cu:namespace gpu {
faiss/gpu/GpuIndexIVF.cu:GpuIndexIVF::GpuIndexIVF(
faiss/gpu/GpuIndexIVF.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVF.cu:        GpuIndexIVFConfig config)
faiss/gpu/GpuIndexIVF.cu:        : GpuIndex(provider->getResources(), dims, metric, metricArg, config),
faiss/gpu/GpuIndexIVF.cu:GpuIndexIVF::GpuIndexIVF(
faiss/gpu/GpuIndexIVF.cu:        GpuResourcesProvider* provider,
faiss/gpu/GpuIndexIVF.cu:        GpuIndexIVFConfig config)
faiss/gpu/GpuIndexIVF.cu:        : GpuIndex(provider->getResources(), dims, metric, metricArg, config),
faiss/gpu/GpuIndexIVF.cu:void GpuIndexIVF::init_() {
faiss/gpu/GpuIndexIVF.cu:        // The passed in quantizer may be either a CPU or GPU index
faiss/gpu/GpuIndexIVF.cu:        // Construct a GPU empty flat quantizer as our coarse quantizer
faiss/gpu/GpuIndexIVF.cu:        GpuIndexFlatConfig config = ivfConfig_.flatConfig;
faiss/gpu/GpuIndexIVF.cu:            quantizer = new GpuIndexFlatL2(resources_, d, config);
faiss/gpu/GpuIndexIVF.cu:            quantizer = new GpuIndexFlatIP(resources_, d, config);
faiss/gpu/GpuIndexIVF.cu:GpuIndexIVF::~GpuIndexIVF() {}
faiss/gpu/GpuIndexIVF.cu:void GpuIndexIVF::verifyIVFSettings_() const {
faiss/gpu/GpuIndexIVF.cu:    // If the quantizer is a GPU index, then it must be resident on the same
faiss/gpu/GpuIndexIVF.cu:    auto gpuQuantizer = tryCastGpuIndex(quantizer);
faiss/gpu/GpuIndexIVF.cu:    if (gpuQuantizer && gpuQuantizer->getDevice() != getDevice()) {
faiss/gpu/GpuIndexIVF.cu:                "GpuIndexIVF: not allowed to instantiate a GPU IVF "
faiss/gpu/GpuIndexIVF.cu:                "index that is resident on a different GPU (%d) "
faiss/gpu/GpuIndexIVF.cu:                "than its GPU coarse quantizer (%d)",
faiss/gpu/GpuIndexIVF.cu:                gpuQuantizer->getDevice());
faiss/gpu/GpuIndexIVF.cu:void GpuIndexIVF::copyFrom(const faiss::IndexIVF* index) {
faiss/gpu/GpuIndexIVF.cu:    GpuIndex::copyFrom(index);
faiss/gpu/GpuIndexIVF.cu:    if (!isGpuIndex(index->quantizer)) {
faiss/gpu/GpuIndexIVF.cu:        // The coarse quantizer used in the IndexIVF is non-GPU.
faiss/gpu/GpuIndexIVF.cu:        // If it is something that we support on the GPU, we wish to copy it
faiss/gpu/GpuIndexIVF.cu:        // over to the GPU, on the same device that we are on.
faiss/gpu/GpuIndexIVF.cu:        GpuResourcesProviderFromInstance pfi(getResources());
faiss/gpu/GpuIndexIVF.cu:        // Attempt to clone the index to GPU. If it fails because the coarse
faiss/gpu/GpuIndexIVF.cu:        // quantizer is not implemented on GPU and the flag to allow CPU
faiss/gpu/GpuIndexIVF.cu:            GpuClonerOptions options;
faiss/gpu/GpuIndexIVF.cu:            auto cloner = ToGpuCloner(&pfi, getDevice(), options);
faiss/gpu/GpuIndexIVF.cu:            if (strstr(e.what(), "not implemented on GPU")) {
faiss/gpu/GpuIndexIVF.cu:                            "GPU and allowCpuCoarseQuantizer is set to false. "
faiss/gpu/GpuIndexIVF.cu:        // Otherwise, this is a GPU coarse quantizer index instance found in a
faiss/gpu/GpuIndexIVF.cu:                "GpuIndexIVF::copyFrom: copying a CPU IVF index to GPU "
faiss/gpu/GpuIndexIVF.cu:                "that already contains a GPU coarse (level 1) quantizer "
faiss/gpu/GpuIndexIVF.cu:void GpuIndexIVF::copyTo(faiss::IndexIVF* index) const {
faiss/gpu/GpuIndexIVF.cu:    GpuIndex::copyTo(index);
faiss/gpu/GpuIndexIVF.cu:    index->quantizer = index_gpu_to_cpu(quantizer);
faiss/gpu/GpuIndexIVF.cu:idx_t GpuIndexIVF::getNumLists() const {
faiss/gpu/GpuIndexIVF.cu:idx_t GpuIndexIVF::getListLength(idx_t listId) const {
faiss/gpu/GpuIndexIVF.cu:std::vector<uint8_t> GpuIndexIVF::getListVectorData(
faiss/gpu/GpuIndexIVF.cu:        bool gpuFormat) const {
faiss/gpu/GpuIndexIVF.cu:    return baseIndex_->getListVectorData(listId, gpuFormat);
faiss/gpu/GpuIndexIVF.cu:std::vector<idx_t> GpuIndexIVF::getListIndices(idx_t listId) const {
faiss/gpu/GpuIndexIVF.cu:void GpuIndexIVF::addImpl_(idx_t n, const float* x, const idx_t* xids) {
faiss/gpu/GpuIndexIVF.cu:    // Device is already set in GpuIndex::add
faiss/gpu/GpuIndexIVF.cu:    // Data is already resident on the GPU
faiss/gpu/GpuIndexIVF.cu:int GpuIndexIVF::getCurrentNProbe_(const SearchParameters* params) const {
faiss/gpu/GpuIndexIVF.cu:                    "GPU IVF index does not currently support "
faiss/gpu/GpuIndexIVF.cu:                    "GPU IVF index: passed unhandled SearchParameters "
faiss/gpu/GpuIndexIVF.cu:void GpuIndexIVF::searchImpl_(
faiss/gpu/GpuIndexIVF.cu:    // Device was already set in GpuIndex::search
faiss/gpu/GpuIndexIVF.cu:    // Data is already resident on the GPU
faiss/gpu/GpuIndexIVF.cu:void GpuIndexIVF::search_preassigned(
faiss/gpu/GpuIndexIVF.cu:            "GpuIndexIVF::search_preassigned does not "
faiss/gpu/GpuIndexIVF.cu:    FAISS_THROW_IF_NOT_MSG(this->is_trained, "GpuIndexIVF not trained");
faiss/gpu/GpuIndexIVF.cu:            "GPU IVF index does not currently support "
faiss/gpu/GpuIndexIVF.cu:    // If the output was not already on the GPU, copy it back
faiss/gpu/GpuIndexIVF.cu:void GpuIndexIVF::range_search_preassigned(
faiss/gpu/GpuIndexIVF.cu:bool GpuIndexIVF::addImplRequiresIDs_() const {
faiss/gpu/GpuIndexIVF.cu:void GpuIndexIVF::trainQuantizer_(idx_t n, const float* x) {
faiss/gpu/GpuIndexIVF.cu:    // leverage the CPU-side k-means code, which works for the GPU
faiss/gpu/GpuIndexIVF.cu:} // namespace gpu
faiss/gpu/GpuResources.cpp: * Copyright (c) 2023, NVIDIA CORPORATION.
faiss/gpu/GpuResources.cpp:#include <faiss/gpu/GpuResources.h>
faiss/gpu/GpuResources.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
faiss/gpu/GpuResources.cpp:namespace gpu {
faiss/gpu/GpuResources.cpp:AllocInfo makeDevAlloc(AllocType at, cudaStream_t st) {
faiss/gpu/GpuResources.cpp:AllocInfo makeTempAlloc(AllocType at, cudaStream_t st) {
faiss/gpu/GpuResources.cpp:AllocInfo makeSpaceAlloc(AllocType at, MemorySpace sp, cudaStream_t st) {
faiss/gpu/GpuResources.cpp:// GpuMemoryReservation
faiss/gpu/GpuResources.cpp:GpuMemoryReservation::GpuMemoryReservation()
faiss/gpu/GpuResources.cpp:GpuMemoryReservation::GpuMemoryReservation(
faiss/gpu/GpuResources.cpp:        GpuResources* r,
faiss/gpu/GpuResources.cpp:        cudaStream_t str,
faiss/gpu/GpuResources.cpp:GpuMemoryReservation::GpuMemoryReservation(GpuMemoryReservation&& m) noexcept {
faiss/gpu/GpuResources.cpp:GpuMemoryReservation& GpuMemoryReservation::operator=(
faiss/gpu/GpuResources.cpp:        GpuMemoryReservation&& m) {
faiss/gpu/GpuResources.cpp:void GpuMemoryReservation::release() {
faiss/gpu/GpuResources.cpp:GpuMemoryReservation::~GpuMemoryReservation() {
faiss/gpu/GpuResources.cpp:// GpuResources
faiss/gpu/GpuResources.cpp:GpuResources::~GpuResources() = default;
faiss/gpu/GpuResources.cpp:cublasHandle_t GpuResources::getBlasHandleCurrentDevice() {
faiss/gpu/GpuResources.cpp:cudaStream_t GpuResources::getDefaultStreamCurrentDevice() {
faiss/gpu/GpuResources.cpp:#if defined USE_NVIDIA_RAFT
faiss/gpu/GpuResources.cpp:raft::device_resources& GpuResources::getRaftHandleCurrentDevice() {
faiss/gpu/GpuResources.cpp:std::vector<cudaStream_t> GpuResources::getAlternateStreamsCurrentDevice() {
faiss/gpu/GpuResources.cpp:cudaStream_t GpuResources::getAsyncCopyStreamCurrentDevice() {
faiss/gpu/GpuResources.cpp:void GpuResources::syncDefaultStream(int device) {
faiss/gpu/GpuResources.cpp:    CUDA_VERIFY(cudaStreamSynchronize(getDefaultStream(device)));
faiss/gpu/GpuResources.cpp:void GpuResources::syncDefaultStreamCurrentDevice() {
faiss/gpu/GpuResources.cpp:GpuMemoryReservation GpuResources::allocMemoryHandle(const AllocRequest& req) {
faiss/gpu/GpuResources.cpp:    return GpuMemoryReservation(
faiss/gpu/GpuResources.cpp:size_t GpuResources::getTempMemoryAvailableCurrentDevice() const {
faiss/gpu/GpuResources.cpp:// GpuResourcesProvider
faiss/gpu/GpuResources.cpp:GpuResourcesProvider::~GpuResourcesProvider() = default;
faiss/gpu/GpuResources.cpp:// GpuResourcesProviderFromResourceInstance
faiss/gpu/GpuResources.cpp:GpuResourcesProviderFromInstance::GpuResourcesProviderFromInstance(
faiss/gpu/GpuResources.cpp:        std::shared_ptr<GpuResources> p)
faiss/gpu/GpuResources.cpp:GpuResourcesProviderFromInstance::~GpuResourcesProviderFromInstance() = default;
faiss/gpu/GpuResources.cpp:std::shared_ptr<GpuResources> GpuResourcesProviderFromInstance::getResources() {
faiss/gpu/GpuResources.cpp:} // namespace gpu
faiss/gpu/GpuFaissAssert.h:#ifndef GPU_FAISS_ASSERT_INCLUDED
faiss/gpu/GpuFaissAssert.h:#define GPU_FAISS_ASSERT_INCLUDED
faiss/gpu/GpuFaissAssert.h:#include <cuda.h>
faiss/gpu/GpuFaissAssert.h:#if defined(__CUDA_ARCH__) || defined(USE_AMD_ROCM)
faiss/gpu/GpuFaissAssert.h:#define GPU_FAISS_ASSERT(X) assert(X)
faiss/gpu/GpuFaissAssert.h:#define GPU_FAISS_ASSERT_MSG(X, MSG) assert(X)
faiss/gpu/GpuFaissAssert.h:#define GPU_FAISS_ASSERT_FMT(X, FMT, ...) assert(X)
faiss/gpu/GpuFaissAssert.h:#define GPU_FAISS_ASSERT(X) FAISS_ASSERT(X)
faiss/gpu/GpuFaissAssert.h:#define GPU_FAISS_ASSERT_MSG(X, MSG) FAISS_ASSERT_MSG(X, MSG)
faiss/gpu/GpuFaissAssert.h:#define GPU_FAISS_ASSERT_FMT(X, FMT, ...) FAISS_ASSERT_FMT(X, FMT, __VA_ARGS)
faiss/gpu/GpuFaissAssert.h:#endif // __CUDA_ARCH__
faiss/gpu/hipify.sh:    # create all destination directories for hipified files into sibling 'gpu-rocm' directory
faiss/gpu/hipify.sh:        dst="${src//gpu/gpu-rocm}"
faiss/gpu/hipify.sh:    done <   <(find ./gpu -type d -print0)
faiss/gpu/hipify.sh:            dst="${src//\.\/gpu/\.\/gpu-rocm}"
faiss/gpu/hipify.sh:        done <   <(find ./gpu -name "*.$ext" -print0)
faiss/gpu/hipify.sh:    done <   <(find ./gpu-rocm -name "*.cu.tmp" -print0)
faiss/gpu/hipify.sh:    # replace header include statements "<faiss/gpu/" with "<faiss/gpu-rocm"
faiss/gpu/hipify.sh:    # replace thrust::cuda::par with thrust::hip::par
faiss/gpu/hipify.sh:            sed -i 's@#include <faiss/gpu/@#include <faiss/gpu-rocm/@' "$src"
faiss/gpu/hipify.sh:            sed -i 's@thrust::cuda::par@thrust::hip::par@' "$src"
faiss/gpu/hipify.sh:        done <   <(find ./gpu-rocm -name "*.$ext.tmp" -print0)
faiss/gpu/hipify.sh:        done <   <(find ./gpu-rocm -name "*.$ext.tmp" -print0)
faiss/gpu/hipify.sh:        dst="${src//\.\/gpu/\.\/gpu-rocm}"
faiss/gpu/hipify.sh:    done <   <(find ./gpu -name "CMakeLists.txt" -print0)
faiss/gpu/hipify.sh:            dst="${src//\.\/gpu/\.\/gpu-rocm}"
faiss/gpu/hipify.sh:        done <   <(find ./gpu -name "*.$ext" -print0)
faiss/gpu/hipify.sh:# Convert the faiss/gpu dir
faiss/utils/partitioning.cpp:struct PreprocMinShift {
faiss/utils/partitioning.cpp:    explicit PreprocMinShift(uint16_t min) {
faiss/utils/partitioning.cpp:        a16 = histogram_8(data, PreprocMinShift<s, 8>(min), (n & ~15)); \
faiss/utils/partitioning.cpp:        a16 = histogram_16(data, PreprocMinShift<s, 16>(min), (n & ~15)); \
faiss/utils/utils.cpp:// this will be set at load time from GPU Faiss
faiss/utils/utils.cpp:std::string gpu_compile_options;
faiss/utils/utils.cpp:    options += gpu_compile_options;
faiss/utils/utils.h: * in contrib.exhaustive_search.range_search_gpu */
faiss/utils/fp16-inl.h:    //   with Intel CPUs, but might be a problem on GPUs or PS3 SPUs),
c_api/INSTALL.md:Building with GPU support
c_api/INSTALL.md:For GPU support, a separate dynamic library in the "c_api/gpu" directory needs to be built.
c_api/INSTALL.md:The "gpufaiss_c" dynamic library contains the GPU and CPU implementations of Faiss, which means that
c_api/INSTALL.md:it can be used in place of "faiss_c". The same library will dynamically link with the CUDA runtime
c_api/INSTALL.md:Using the GPU with the C API
c_api/INSTALL.md:A standard GPU resources object can be obtained by the name `FaissStandardGpuResources`:
c_api/INSTALL.md:FaissStandardGpuResources* gpu_res = NULL;
c_api/INSTALL.md:int c = faiss_StandardGpuResources_new(&gpu_res);
c_api/INSTALL.md:Similarly to the C++ API, a CPU index can be converted to a GPU index:
c_api/INSTALL.md:FaissGpuIndex* gpu_index = NULL;
c_api/INSTALL.md:c = faiss_index_cpu_to_gpu(gpu_res, 0, cpu_index, &gpu_index);
c_api/INSTALL.md:A more complete example is available by the name `bin/example_gpu_c`.
c_api/CMakeLists.txt:if(FAISS_ENABLE_GPU)
c_api/CMakeLists.txt:  if(FAISS_ENABLE_ROCM)
c_api/CMakeLists.txt:    add_subdirectory(gpu-rocm)
c_api/CMakeLists.txt:    add_subdirectory(gpu)
c_api/gpu/StandardGpuResources_c.cpp:#include "StandardGpuResources_c.h"
c_api/gpu/StandardGpuResources_c.cpp:#include <faiss/gpu/StandardGpuResources.h>
c_api/gpu/StandardGpuResources_c.cpp:using faiss::gpu::StandardGpuResources;
c_api/gpu/StandardGpuResources_c.cpp:DEFINE_DESTRUCTOR(StandardGpuResources)
c_api/gpu/StandardGpuResources_c.cpp:int faiss_StandardGpuResources_new(FaissStandardGpuResources** p_res) {
c_api/gpu/StandardGpuResources_c.cpp:        auto p = new StandardGpuResources();
c_api/gpu/StandardGpuResources_c.cpp:        *p_res = reinterpret_cast<FaissStandardGpuResources*>(p);
c_api/gpu/StandardGpuResources_c.cpp:int faiss_StandardGpuResources_noTempMemory(FaissStandardGpuResources* res) {
c_api/gpu/StandardGpuResources_c.cpp:        reinterpret_cast<StandardGpuResources*>(res)->noTempMemory();
c_api/gpu/StandardGpuResources_c.cpp:int faiss_StandardGpuResources_setTempMemory(
c_api/gpu/StandardGpuResources_c.cpp:        FaissStandardGpuResources* res,
c_api/gpu/StandardGpuResources_c.cpp:        reinterpret_cast<StandardGpuResources*>(res)->setTempMemory(size);
c_api/gpu/StandardGpuResources_c.cpp:int faiss_StandardGpuResources_setPinnedMemory(
c_api/gpu/StandardGpuResources_c.cpp:        FaissStandardGpuResources* res,
c_api/gpu/StandardGpuResources_c.cpp:        reinterpret_cast<StandardGpuResources*>(res)->setPinnedMemory(size);
c_api/gpu/StandardGpuResources_c.cpp:int faiss_StandardGpuResources_setDefaultStream(
c_api/gpu/StandardGpuResources_c.cpp:        FaissStandardGpuResources* res,
c_api/gpu/StandardGpuResources_c.cpp:        cudaStream_t stream) {
c_api/gpu/StandardGpuResources_c.cpp:        reinterpret_cast<StandardGpuResources*>(res)->setDefaultStream(
c_api/gpu/StandardGpuResources_c.cpp:int faiss_StandardGpuResources_setDefaultNullStreamAllDevices(
c_api/gpu/StandardGpuResources_c.cpp:        FaissStandardGpuResources* res) {
c_api/gpu/StandardGpuResources_c.cpp:        reinterpret_cast<StandardGpuResources*>(res)
c_api/gpu/GpuAutoTune_c.cpp:#include "GpuAutoTune_c.h"
c_api/gpu/GpuAutoTune_c.cpp:#include <faiss/gpu/GpuAutoTune.h>
c_api/gpu/GpuAutoTune_c.cpp:#include <faiss/gpu/GpuCloner.h>
c_api/gpu/GpuAutoTune_c.cpp:#include <faiss/gpu/GpuClonerOptions.h>
c_api/gpu/GpuAutoTune_c.cpp:#include <faiss/gpu/GpuResources.h>
c_api/gpu/GpuAutoTune_c.cpp:#include "GpuClonerOptions_c.h"
c_api/gpu/GpuAutoTune_c.cpp:using faiss::gpu::GpuClonerOptions;
c_api/gpu/GpuAutoTune_c.cpp:using faiss::gpu::GpuMultipleClonerOptions;
c_api/gpu/GpuAutoTune_c.cpp:using faiss::gpu::GpuResourcesProvider;
c_api/gpu/GpuAutoTune_c.cpp:int faiss_index_gpu_to_cpu(const FaissIndex* gpu_index, FaissIndex** p_out) {
c_api/gpu/GpuAutoTune_c.cpp:        auto cpu_index = faiss::gpu::index_gpu_to_cpu(
c_api/gpu/GpuAutoTune_c.cpp:                reinterpret_cast<const Index*>(gpu_index));
c_api/gpu/GpuAutoTune_c.cpp:/// converts any CPU index that can be converted to GPU
c_api/gpu/GpuAutoTune_c.cpp:int faiss_index_cpu_to_gpu(
c_api/gpu/GpuAutoTune_c.cpp:        FaissGpuResourcesProvider* provider,
c_api/gpu/GpuAutoTune_c.cpp:        FaissGpuIndex** p_out) {
c_api/gpu/GpuAutoTune_c.cpp:        auto res = reinterpret_cast<GpuResourcesProvider*>(provider);
c_api/gpu/GpuAutoTune_c.cpp:        auto gpu_index = faiss::gpu::index_cpu_to_gpu(
c_api/gpu/GpuAutoTune_c.cpp:        *p_out = reinterpret_cast<FaissGpuIndex*>(gpu_index);
c_api/gpu/GpuAutoTune_c.cpp:int faiss_index_cpu_to_gpu_with_options(
c_api/gpu/GpuAutoTune_c.cpp:        FaissGpuResourcesProvider* provider,
c_api/gpu/GpuAutoTune_c.cpp:        const FaissGpuClonerOptions* options,
c_api/gpu/GpuAutoTune_c.cpp:        FaissGpuIndex** p_out) {
c_api/gpu/GpuAutoTune_c.cpp:        auto res = reinterpret_cast<GpuResourcesProvider*>(provider);
c_api/gpu/GpuAutoTune_c.cpp:        auto gpu_index = faiss::gpu::index_cpu_to_gpu(
c_api/gpu/GpuAutoTune_c.cpp:                reinterpret_cast<const GpuClonerOptions*>(options));
c_api/gpu/GpuAutoTune_c.cpp:        *p_out = reinterpret_cast<FaissGpuIndex*>(gpu_index);
c_api/gpu/GpuAutoTune_c.cpp:int faiss_index_cpu_to_gpu_multiple(
c_api/gpu/GpuAutoTune_c.cpp:        FaissGpuResourcesProvider* const* providers_vec,
c_api/gpu/GpuAutoTune_c.cpp:        FaissGpuIndex** p_out) {
c_api/gpu/GpuAutoTune_c.cpp:        std::vector<GpuResourcesProvider*> res(devices_size);
c_api/gpu/GpuAutoTune_c.cpp:            res[i] = reinterpret_cast<GpuResourcesProvider*>(providers_vec[i]);
c_api/gpu/GpuAutoTune_c.cpp:        auto gpu_index = faiss::gpu::index_cpu_to_gpu_multiple(
c_api/gpu/GpuAutoTune_c.cpp:        *p_out = reinterpret_cast<FaissGpuIndex*>(gpu_index);
c_api/gpu/GpuAutoTune_c.cpp:int faiss_index_cpu_to_gpu_multiple_with_options(
c_api/gpu/GpuAutoTune_c.cpp:        FaissGpuResourcesProvider* const* providers_vec,
c_api/gpu/GpuAutoTune_c.cpp:        const FaissGpuMultipleClonerOptions* options,
c_api/gpu/GpuAutoTune_c.cpp:        FaissGpuIndex** p_out) {
c_api/gpu/GpuAutoTune_c.cpp:        std::vector<GpuResourcesProvider*> res(providers_vec_size);
c_api/gpu/GpuAutoTune_c.cpp:            res[i] = reinterpret_cast<GpuResourcesProvider*>(providers_vec[i]);
c_api/gpu/GpuAutoTune_c.cpp:        auto gpu_index = faiss::gpu::index_cpu_to_gpu_multiple(
c_api/gpu/GpuAutoTune_c.cpp:                reinterpret_cast<const GpuMultipleClonerOptions*>(options));
c_api/gpu/GpuAutoTune_c.cpp:        *p_out = reinterpret_cast<FaissGpuIndex*>(gpu_index);
c_api/gpu/DeviceUtils_c.h:#include <cuda_runtime_api.h>
c_api/gpu/DeviceUtils_c.h:/// Returns the number of available GPU devices
c_api/gpu/DeviceUtils_c.h:int faiss_get_num_gpus(int* p_output);
c_api/gpu/DeviceUtils_c.h:/// Starts the CUDA profiler (exposed via SWIG)
c_api/gpu/DeviceUtils_c.h:int faiss_gpu_profiler_start();
c_api/gpu/DeviceUtils_c.h:/// Stops the CUDA profiler (exposed via SWIG)
c_api/gpu/DeviceUtils_c.h:int faiss_gpu_profiler_stop();
c_api/gpu/DeviceUtils_c.h:/// cudaDeviceSynchronize for each device)
c_api/gpu/DeviceUtils_c.h:int faiss_gpu_sync_all_devices();
c_api/gpu/example_gpu_c.c:#include "GpuAutoTune_c.h"
c_api/gpu/example_gpu_c.c:#include "StandardGpuResources_c.h"
c_api/gpu/example_gpu_c.c:    int gpus = -1;
c_api/gpu/example_gpu_c.c:    FAISS_TRY(faiss_get_num_gpus(&gpus));
c_api/gpu/example_gpu_c.c:    printf("%d GPU devices are available\n", gpus);
c_api/gpu/example_gpu_c.c:    printf("Loading standard GPU resources...\n");
c_api/gpu/example_gpu_c.c:    FaissStandardGpuResources* gpu_res = NULL;
c_api/gpu/example_gpu_c.c:    FAISS_TRY(faiss_StandardGpuResources_new(&gpu_res));
c_api/gpu/example_gpu_c.c:    printf("Moving index to the GPU...\n");
c_api/gpu/example_gpu_c.c:    FaissGpuIndex* index = NULL;
c_api/gpu/example_gpu_c.c:    FaissGpuClonerOptions* options = NULL;
c_api/gpu/example_gpu_c.c:    FAISS_TRY(faiss_GpuClonerOptions_new(&options));
c_api/gpu/example_gpu_c.c:    FAISS_TRY(faiss_index_cpu_to_gpu_with_options(
c_api/gpu/example_gpu_c.c:            gpu_res, 0, cpu_index, options, &index));
c_api/gpu/example_gpu_c.c:    printf("Freeing GPU resources...\n");
c_api/gpu/example_gpu_c.c:    faiss_GpuResources_free(gpu_res);
c_api/gpu/example_gpu_c.c:    faiss_GpuClonerOptions_free(options);
c_api/gpu/GpuIndicesOptions_c.h:#ifndef FAISS_GPU_INDICES_OPTIONS_C_H
c_api/gpu/GpuIndicesOptions_c.h:#define FAISS_GPU_INDICES_OPTIONS_C_H
c_api/gpu/GpuIndicesOptions_c.h:/// How user vector index data is stored on the GPU
c_api/gpu/GpuIndicesOptions_c.h:    /// The user indices are only stored on the CPU; the GPU returns
c_api/gpu/GpuIndicesOptions_c.h:    /// GPU. Only (inverted list, offset) is returned to the user as the
c_api/gpu/GpuIndicesOptions_c.h:    /// Indices are stored as 32 bit integers on the GPU, but returned
c_api/gpu/GpuIndicesOptions_c.h:    /// Indices are stored as 64 bit integers on the GPU
c_api/gpu/GpuClonerOptions_c.cpp:#include "GpuClonerOptions_c.h"
c_api/gpu/GpuClonerOptions_c.cpp:#include <faiss/gpu/GpuClonerOptions.h>
c_api/gpu/GpuClonerOptions_c.cpp:using faiss::gpu::GpuClonerOptions;
c_api/gpu/GpuClonerOptions_c.cpp:using faiss::gpu::GpuMultipleClonerOptions;
c_api/gpu/GpuClonerOptions_c.cpp:using faiss::gpu::IndicesOptions;
c_api/gpu/GpuClonerOptions_c.cpp:int faiss_GpuClonerOptions_new(FaissGpuClonerOptions** p) {
c_api/gpu/GpuClonerOptions_c.cpp:        *p = reinterpret_cast<FaissGpuClonerOptions*>(new GpuClonerOptions());
c_api/gpu/GpuClonerOptions_c.cpp:int faiss_GpuMultipleClonerOptions_new(FaissGpuMultipleClonerOptions** p) {
c_api/gpu/GpuClonerOptions_c.cpp:        *p = reinterpret_cast<FaissGpuMultipleClonerOptions*>(
c_api/gpu/GpuClonerOptions_c.cpp:                new GpuMultipleClonerOptions());
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_DESTRUCTOR(GpuClonerOptions)
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_DESTRUCTOR(GpuMultipleClonerOptions)
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_GETTER(GpuClonerOptions, FaissIndicesOptions, indicesOptions)
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_GETTER(GpuClonerOptions, int, useFloat16CoarseQuantizer)
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_GETTER(GpuClonerOptions, int, useFloat16)
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_GETTER(GpuClonerOptions, int, usePrecomputed)
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_GETTER(GpuClonerOptions, long, reserveVecs)
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_GETTER(GpuClonerOptions, int, storeTransposed)
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_GETTER(GpuClonerOptions, int, verbose)
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_GETTER(GpuMultipleClonerOptions, int, shard)
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_GETTER(GpuMultipleClonerOptions, int, shard_type)
c_api/gpu/GpuClonerOptions_c.cpp:        GpuClonerOptions,
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_SETTER_STATIC(GpuClonerOptions, bool, int, useFloat16CoarseQuantizer)
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_SETTER_STATIC(GpuClonerOptions, bool, int, useFloat16)
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_SETTER_STATIC(GpuClonerOptions, bool, int, usePrecomputed)
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_SETTER(GpuClonerOptions, long, reserveVecs)
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_SETTER_STATIC(GpuClonerOptions, bool, int, storeTransposed)
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_SETTER_STATIC(GpuClonerOptions, bool, int, verbose)
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_SETTER_STATIC(GpuMultipleClonerOptions, bool, int, shard)
c_api/gpu/GpuClonerOptions_c.cpp:DEFINE_SETTER(GpuMultipleClonerOptions, int, shard_type)
c_api/gpu/GpuResources_c.cpp:#include "GpuResources_c.h"
c_api/gpu/GpuResources_c.cpp:#include <faiss/gpu/GpuResources.h>
c_api/gpu/GpuResources_c.cpp:using faiss::gpu::GpuResources;
c_api/gpu/GpuResources_c.cpp:using faiss::gpu::GpuResourcesProvider;
c_api/gpu/GpuResources_c.cpp:DEFINE_DESTRUCTOR(GpuResources)
c_api/gpu/GpuResources_c.cpp:int faiss_GpuResources_initializeForDevice(FaissGpuResources* res, int device) {
c_api/gpu/GpuResources_c.cpp:        reinterpret_cast<GpuResources*>(res)->initializeForDevice(device);
c_api/gpu/GpuResources_c.cpp:int faiss_GpuResources_getBlasHandle(
c_api/gpu/GpuResources_c.cpp:        FaissGpuResources* res,
c_api/gpu/GpuResources_c.cpp:        auto o = reinterpret_cast<GpuResources*>(res)->getBlasHandle(device);
c_api/gpu/GpuResources_c.cpp:int faiss_GpuResources_getDefaultStream(
c_api/gpu/GpuResources_c.cpp:        FaissGpuResources* res,
c_api/gpu/GpuResources_c.cpp:        cudaStream_t* out) {
c_api/gpu/GpuResources_c.cpp:        auto o = reinterpret_cast<GpuResources*>(res)->getDefaultStream(device);
c_api/gpu/GpuResources_c.cpp:int faiss_GpuResources_getPinnedMemory(
c_api/gpu/GpuResources_c.cpp:        FaissGpuResources* res,
c_api/gpu/GpuResources_c.cpp:        auto o = reinterpret_cast<GpuResources*>(res)->getPinnedMemory();
c_api/gpu/GpuResources_c.cpp:int faiss_GpuResources_getAsyncCopyStream(
c_api/gpu/GpuResources_c.cpp:        FaissGpuResources* res,
c_api/gpu/GpuResources_c.cpp:        cudaStream_t* out) {
c_api/gpu/GpuResources_c.cpp:        auto o = reinterpret_cast<GpuResources*>(res)->getAsyncCopyStream(
c_api/gpu/GpuResources_c.cpp:int faiss_GpuResources_getBlasHandleCurrentDevice(
c_api/gpu/GpuResources_c.cpp:        FaissGpuResources* res,
c_api/gpu/GpuResources_c.cpp:        auto o = reinterpret_cast<GpuResources*>(res)
c_api/gpu/GpuResources_c.cpp:int faiss_GpuResources_getDefaultStreamCurrentDevice(
c_api/gpu/GpuResources_c.cpp:        FaissGpuResources* res,
c_api/gpu/GpuResources_c.cpp:        cudaStream_t* out) {
c_api/gpu/GpuResources_c.cpp:        auto o = reinterpret_cast<GpuResources*>(res)
c_api/gpu/GpuResources_c.cpp:int faiss_GpuResources_syncDefaultStream(FaissGpuResources* res, int device) {
c_api/gpu/GpuResources_c.cpp:        reinterpret_cast<GpuResources*>(res)->syncDefaultStream(device);
c_api/gpu/GpuResources_c.cpp:int faiss_GpuResources_syncDefaultStreamCurrentDevice(FaissGpuResources* res) {
c_api/gpu/GpuResources_c.cpp:        reinterpret_cast<GpuResources*>(res)->syncDefaultStreamCurrentDevice();
c_api/gpu/GpuResources_c.cpp:int faiss_GpuResources_getAsyncCopyStreamCurrentDevice(
c_api/gpu/GpuResources_c.cpp:        FaissGpuResources* res,
c_api/gpu/GpuResources_c.cpp:        cudaStream_t* out) {
c_api/gpu/GpuResources_c.cpp:        auto o = reinterpret_cast<GpuResources*>(res)
c_api/gpu/GpuResources_c.cpp:DEFINE_DESTRUCTOR(GpuResourcesProvider)
c_api/gpu/GpuResources_c.cpp:int faiss_GpuResourcesProvider_getResources(
c_api/gpu/GpuResources_c.cpp:        FaissGpuResourcesProvider* res,
c_api/gpu/GpuResources_c.cpp:        FaissGpuResources** out) {
c_api/gpu/GpuResources_c.cpp:        auto o = reinterpret_cast<GpuResourcesProvider*>(res)->getResources();
c_api/gpu/GpuResources_c.cpp:        *out = reinterpret_cast<FaissGpuResources*>(o.get());
c_api/gpu/CMakeLists.txt:  GpuAutoTune_c.cpp
c_api/gpu/CMakeLists.txt:  GpuClonerOptions_c.cpp
c_api/gpu/CMakeLists.txt:  GpuIndex_c.cpp
c_api/gpu/CMakeLists.txt:  GpuResources_c.cpp
c_api/gpu/CMakeLists.txt:  StandardGpuResources_c.cpp
c_api/gpu/CMakeLists.txt:file(GLOB FAISS_C_API_GPU_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")
c_api/gpu/CMakeLists.txt:faiss_install_headers("${FAISS_C_API_GPU_HEADERS}" c_api/gpu)
c_api/gpu/CMakeLists.txt:if (FAISS_ENABLE_ROCM)
c_api/gpu/CMakeLists.txt:  find_package(CUDAToolkit REQUIRED)
c_api/gpu/CMakeLists.txt:  target_link_libraries(faiss_c PUBLIC CUDA::cudart CUDA::cublas
c_api/gpu/CMakeLists.txt:    $<$<BOOL:${FAISS_ENABLE_RAFT}>:nvidia::cutlass::cutlass>)
c_api/gpu/CMakeLists.txt:add_executable(example_gpu_c EXCLUDE_FROM_ALL example_gpu_c.c)
c_api/gpu/CMakeLists.txt:target_link_libraries(example_gpu_c PRIVATE faiss_c)
c_api/gpu/GpuIndex_c.h:#ifndef FAISS_GPU_INDEX_C_H
c_api/gpu/GpuIndex_c.h:#define FAISS_GPU_INDEX_C_H
c_api/gpu/GpuIndex_c.h:FAISS_DECLARE_CLASS(GpuIndexConfig)
c_api/gpu/GpuIndex_c.h:FAISS_DECLARE_GETTER(GpuIndexConfig, int, device)
c_api/gpu/GpuIndex_c.h:FAISS_DECLARE_CLASS_INHERITED(GpuIndex, Index)
c_api/gpu/GpuResources_c.h:#ifndef FAISS_GPU_RESOURCES_C_H
c_api/gpu/GpuResources_c.h:#define FAISS_GPU_RESOURCES_C_H
c_api/gpu/GpuResources_c.h:#include <cuda_runtime_api.h>
c_api/gpu/GpuResources_c.h:/// Base class of GPU-side resource provider; hides provision of
c_api/gpu/GpuResources_c.h:/// cuBLAS handles, CUDA streams and a temporary memory manager
c_api/gpu/GpuResources_c.h:FAISS_DECLARE_CLASS(GpuResources)
c_api/gpu/GpuResources_c.h:FAISS_DECLARE_DESTRUCTOR(GpuResources)
c_api/gpu/GpuResources_c.h:int faiss_GpuResources_initializeForDevice(FaissGpuResources*, int);
c_api/gpu/GpuResources_c.h:int faiss_GpuResources_getBlasHandle(FaissGpuResources*, int, cublasHandle_t*);
c_api/gpu/GpuResources_c.h:int faiss_GpuResources_getDefaultStream(FaissGpuResources*, int, cudaStream_t*);
c_api/gpu/GpuResources_c.h:int faiss_GpuResources_getPinnedMemory(FaissGpuResources*, void**, size_t*);
c_api/gpu/GpuResources_c.h:/// Returns the stream on which we perform async CPU <-> GPU copies
c_api/gpu/GpuResources_c.h:int faiss_GpuResources_getAsyncCopyStream(
c_api/gpu/GpuResources_c.h:        FaissGpuResources*,
c_api/gpu/GpuResources_c.h:        cudaStream_t*);
c_api/gpu/GpuResources_c.h:int faiss_GpuResources_getBlasHandleCurrentDevice(
c_api/gpu/GpuResources_c.h:        FaissGpuResources*,
c_api/gpu/GpuResources_c.h:int faiss_GpuResources_getDefaultStreamCurrentDevice(
c_api/gpu/GpuResources_c.h:        FaissGpuResources*,
c_api/gpu/GpuResources_c.h:        cudaStream_t*);
c_api/gpu/GpuResources_c.h:// equivalent to cudaDeviceSynchronize(getDefaultStream(device))
c_api/gpu/GpuResources_c.h:int faiss_GpuResources_syncDefaultStream(FaissGpuResources*, int);
c_api/gpu/GpuResources_c.h:int faiss_GpuResources_syncDefaultStreamCurrentDevice(FaissGpuResources*);
c_api/gpu/GpuResources_c.h:int faiss_GpuResources_getAsyncCopyStreamCurrentDevice(
c_api/gpu/GpuResources_c.h:        FaissGpuResources*,
c_api/gpu/GpuResources_c.h:        cudaStream_t*);
c_api/gpu/GpuResources_c.h:FAISS_DECLARE_CLASS(GpuResourcesProvider)
c_api/gpu/GpuResources_c.h:FAISS_DECLARE_DESTRUCTOR(GpuResourcesProvider)
c_api/gpu/GpuResources_c.h:int faiss_GpuResourcesProvider_getResources(
c_api/gpu/GpuResources_c.h:        FaissGpuResourcesProvider*,
c_api/gpu/GpuResources_c.h:        FaissGpuResources**);
c_api/gpu/DeviceUtils_c.cpp:#include <faiss/gpu/utils/DeviceUtils.h>
c_api/gpu/DeviceUtils_c.cpp:/// Returns the number of available GPU devices
c_api/gpu/DeviceUtils_c.cpp:int faiss_get_num_gpus(int* p_output) {
c_api/gpu/DeviceUtils_c.cpp:        int output = faiss::gpu::getNumDevices();
c_api/gpu/DeviceUtils_c.cpp:/// Starts the CUDA profiler (exposed via SWIG)
c_api/gpu/DeviceUtils_c.cpp:int faiss_gpu_profiler_start() {
c_api/gpu/DeviceUtils_c.cpp:        faiss::gpu::profilerStart();
c_api/gpu/DeviceUtils_c.cpp:/// Stops the CUDA profiler (exposed via SWIG)
c_api/gpu/DeviceUtils_c.cpp:int faiss_gpu_profiler_stop() {
c_api/gpu/DeviceUtils_c.cpp:        faiss::gpu::profilerStop();
c_api/gpu/DeviceUtils_c.cpp:/// cudaDeviceSynchronize for each device)
c_api/gpu/DeviceUtils_c.cpp:int faiss_gpu_sync_all_devices() {
c_api/gpu/DeviceUtils_c.cpp:        faiss::gpu::synchronizeAllDevices();
c_api/gpu/macros_impl.h:#ifndef GPU_MACROS_IMPL_H
c_api/gpu/macros_impl.h:#define GPU_MACROS_IMPL_H
c_api/gpu/macros_impl.h:                reinterpret_cast<const faiss::gpu::clazz*>(obj)->name); \
c_api/gpu/macros_impl.h:        reinterpret_cast<faiss::gpu::clazz*>(obj)->name = val;   \
c_api/gpu/macros_impl.h:        reinterpret_cast<faiss::gpu::clazz*>(obj)->name =             \
c_api/gpu/macros_impl.h:        delete reinterpret_cast<faiss::gpu::clazz*>(obj); \
c_api/gpu/StandardGpuResources_c.h:#ifndef FAISS_STANDARD_GPURESOURCES_C_H
c_api/gpu/StandardGpuResources_c.h:#define FAISS_STANDARD_GPURESOURCES_C_H
c_api/gpu/StandardGpuResources_c.h:#include <cuda_runtime_api.h>
c_api/gpu/StandardGpuResources_c.h:#include "GpuResources_c.h"
c_api/gpu/StandardGpuResources_c.h:/// Default implementation of GpuResourcesProvider that allocates a cuBLAS
c_api/gpu/StandardGpuResources_c.h:FAISS_DECLARE_CLASS_INHERITED(StandardGpuResources, GpuResourcesProvider)
c_api/gpu/StandardGpuResources_c.h:FAISS_DECLARE_DESTRUCTOR(StandardGpuResources)
c_api/gpu/StandardGpuResources_c.h:/// Default constructor for StandardGpuResources
c_api/gpu/StandardGpuResources_c.h:int faiss_StandardGpuResources_new(FaissStandardGpuResources**);
c_api/gpu/StandardGpuResources_c.h:/// requests will call cudaMalloc / cudaFree at the point of use
c_api/gpu/StandardGpuResources_c.h:int faiss_StandardGpuResources_noTempMemory(FaissStandardGpuResources*);
c_api/gpu/StandardGpuResources_c.h:int faiss_StandardGpuResources_setTempMemory(
c_api/gpu/StandardGpuResources_c.h:        FaissStandardGpuResources*,
c_api/gpu/StandardGpuResources_c.h:/// Set amount of pinned memory to allocate, for async GPU <-> CPU
c_api/gpu/StandardGpuResources_c.h:int faiss_StandardGpuResources_setPinnedMemory(
c_api/gpu/StandardGpuResources_c.h:        FaissStandardGpuResources*,
c_api/gpu/StandardGpuResources_c.h:int faiss_StandardGpuResources_setDefaultStream(
c_api/gpu/StandardGpuResources_c.h:        FaissStandardGpuResources*,
c_api/gpu/StandardGpuResources_c.h:        cudaStream_t stream);
c_api/gpu/StandardGpuResources_c.h:int faiss_StandardGpuResources_setDefaultNullStreamAllDevices(
c_api/gpu/StandardGpuResources_c.h:        FaissStandardGpuResources*);
c_api/gpu/GpuAutoTune_c.h:#ifndef FAISS_GPU_AUTO_TUNE_C_H
c_api/gpu/GpuAutoTune_c.h:#define FAISS_GPU_AUTO_TUNE_C_H
c_api/gpu/GpuAutoTune_c.h:#include "GpuClonerOptions_c.h"
c_api/gpu/GpuAutoTune_c.h:#include "GpuIndex_c.h"
c_api/gpu/GpuAutoTune_c.h:#include "GpuResources_c.h"
c_api/gpu/GpuAutoTune_c.h:/// converts any GPU index inside gpu_index to a CPU index
c_api/gpu/GpuAutoTune_c.h:int faiss_index_gpu_to_cpu(const FaissIndex* gpu_index, FaissIndex** p_out);
c_api/gpu/GpuAutoTune_c.h:/// converts any CPU index that can be converted to GPU
c_api/gpu/GpuAutoTune_c.h:int faiss_index_cpu_to_gpu(
c_api/gpu/GpuAutoTune_c.h:        FaissGpuResourcesProvider* provider,
c_api/gpu/GpuAutoTune_c.h:        FaissGpuIndex** p_out);
c_api/gpu/GpuAutoTune_c.h:/// converts any CPU index that can be converted to GPU
c_api/gpu/GpuAutoTune_c.h:int faiss_index_cpu_to_gpu_with_options(
c_api/gpu/GpuAutoTune_c.h:        FaissGpuResourcesProvider* provider,
c_api/gpu/GpuAutoTune_c.h:        const FaissGpuClonerOptions* options,
c_api/gpu/GpuAutoTune_c.h:        FaissGpuIndex** p_out);
c_api/gpu/GpuAutoTune_c.h:/// converts any CPU index that can be converted to GPU
c_api/gpu/GpuAutoTune_c.h:int faiss_index_cpu_to_gpu_multiple(
c_api/gpu/GpuAutoTune_c.h:        FaissGpuResourcesProvider* const* providers_vec,
c_api/gpu/GpuAutoTune_c.h:        FaissGpuIndex** p_out);
c_api/gpu/GpuAutoTune_c.h:/// converts any CPU index that can be converted to GPU
c_api/gpu/GpuAutoTune_c.h:int faiss_index_cpu_to_gpu_multiple_with_options(
c_api/gpu/GpuAutoTune_c.h:        FaissGpuResourcesProvider* const* providers_vec,
c_api/gpu/GpuAutoTune_c.h:        const FaissGpuMultipleClonerOptions* options,
c_api/gpu/GpuAutoTune_c.h:        FaissGpuIndex** p_out);
c_api/gpu/GpuAutoTune_c.h:/// parameter space and setters for GPU indexes
c_api/gpu/GpuAutoTune_c.h:FAISS_DECLARE_CLASS_INHERITED(GpuParameterSpace, ParameterSpace)
c_api/gpu/GpuIndex_c.cpp:#include "GpuIndex_c.h"
c_api/gpu/GpuIndex_c.cpp:#include <faiss/gpu/GpuIndex.h>
c_api/gpu/GpuIndex_c.cpp:using faiss::gpu::GpuIndexConfig;
c_api/gpu/GpuIndex_c.cpp:DEFINE_GETTER(GpuIndexConfig, int, device)
c_api/gpu/GpuClonerOptions_c.h:#ifndef FAISS_GPU_CLONER_OPTIONS_C_H
c_api/gpu/GpuClonerOptions_c.h:#define FAISS_GPU_CLONER_OPTIONS_C_H
c_api/gpu/GpuClonerOptions_c.h:#include "GpuIndicesOptions_c.h"
c_api/gpu/GpuClonerOptions_c.h:FAISS_DECLARE_CLASS(GpuClonerOptions)
c_api/gpu/GpuClonerOptions_c.h:FAISS_DECLARE_DESTRUCTOR(GpuClonerOptions)
c_api/gpu/GpuClonerOptions_c.h:/// Default constructor for GpuClonerOptions
c_api/gpu/GpuClonerOptions_c.h:int faiss_GpuClonerOptions_new(FaissGpuClonerOptions**);
c_api/gpu/GpuClonerOptions_c.h:/// (anything but GpuIndexFlat*)?
c_api/gpu/GpuClonerOptions_c.h:        GpuClonerOptions,
c_api/gpu/GpuClonerOptions_c.h:FAISS_DECLARE_GETTER_SETTER(GpuClonerOptions, int, useFloat16CoarseQuantizer)
c_api/gpu/GpuClonerOptions_c.h:/// (boolean) for GpuIndexIVFFlat, is storage in float16?
c_api/gpu/GpuClonerOptions_c.h:/// for GpuIndexIVFPQ, are intermediate calculations in float16?
c_api/gpu/GpuClonerOptions_c.h:FAISS_DECLARE_GETTER_SETTER(GpuClonerOptions, int, useFloat16)
c_api/gpu/GpuClonerOptions_c.h:FAISS_DECLARE_GETTER_SETTER(GpuClonerOptions, int, usePrecomputed)
c_api/gpu/GpuClonerOptions_c.h:FAISS_DECLARE_GETTER_SETTER(GpuClonerOptions, long, reserveVecs)
c_api/gpu/GpuClonerOptions_c.h:/// (boolean) For GpuIndexFlat, store data in transposed layout?
c_api/gpu/GpuClonerOptions_c.h:FAISS_DECLARE_GETTER_SETTER(GpuClonerOptions, int, storeTransposed)
c_api/gpu/GpuClonerOptions_c.h:FAISS_DECLARE_GETTER_SETTER(GpuClonerOptions, int, verbose)
c_api/gpu/GpuClonerOptions_c.h:FAISS_DECLARE_CLASS_INHERITED(GpuMultipleClonerOptions, GpuClonerOptions)
c_api/gpu/GpuClonerOptions_c.h:FAISS_DECLARE_DESTRUCTOR(GpuMultipleClonerOptions)
c_api/gpu/GpuClonerOptions_c.h:/// Default constructor for GpuMultipleClonerOptions
c_api/gpu/GpuClonerOptions_c.h:int faiss_GpuMultipleClonerOptions_new(FaissGpuMultipleClonerOptions**);
c_api/gpu/GpuClonerOptions_c.h:/// (boolean) Whether to shard the index across GPUs, versus replication
c_api/gpu/GpuClonerOptions_c.h:/// across GPUs
c_api/gpu/GpuClonerOptions_c.h:FAISS_DECLARE_GETTER_SETTER(GpuMultipleClonerOptions, int, shard)
c_api/gpu/GpuClonerOptions_c.h:FAISS_DECLARE_GETTER_SETTER(GpuMultipleClonerOptions, int, shard_type)
cmake/thirdparty/fetch_rapids.cmake:# Copyright (c) 2023, NVIDIA CORPORATION.
.gitignore:/c_api/gpu/bin/
demos/demo_auto_tune.py:# indexes that can run on the GPU
demos/demo_auto_tune.py:keys_gpu = [
demos/demo_auto_tune.py:use_gpu = False
demos/demo_auto_tune.py:if use_gpu:
demos/demo_auto_tune.py:    # if this fails, it means that the GPU version was not comp
demos/demo_auto_tune.py:    assert faiss.StandardGpuResources, \
demos/demo_auto_tune.py:        "FAISS was not compiled with GPU support, or loading _swigfaiss_gpu.so failed"
demos/demo_auto_tune.py:    res = faiss.StandardGpuResources()
demos/demo_auto_tune.py:    if use_gpu:
demos/demo_auto_tune.py:        # transfer to GPU (may be partial)
demos/demo_auto_tune.py:        index = faiss.index_cpu_to_gpu(res, dev_no, index)
demos/demo_auto_tune.py:        params = faiss.GpuParameterSpace()
demos/offline_ivf/run.py:            gpus_per_node=args.gpus_per_node,
demos/offline_ivf/run.py:        "--gpus_per_node",
demos/offline_ivf/README.md:`conda install -c pytorch/label/nightly -c nvidia faiss-gpu=1.7.4`
demos/offline_ivf/utils.py:    index_gpu = faiss.index_cpu_to_all_gpus(index_cpu)
demos/offline_ivf/utils.py:    index_gpu.add(xb)
demos/offline_ivf/utils.py:    D_gpu, I_gpu = index_gpu.search(xq, 10)
demos/offline_ivf/utils.py:    assert np.all(I_cpu == I_gpu), "faiss sanity check failed"
demos/offline_ivf/utils.py:    assert np.all(np.isclose(D_cpu, D_gpu)), "faiss sanity check failed"
demos/offline_ivf/offline_ivf.py:        os.system("nvidia-smi")
demos/offline_ivf/offline_ivf.py:        gpu_quantizer = faiss.index_cpu_to_all_gpus(cpu_quantizer)
demos/offline_ivf/offline_ivf.py:                index_ivf.quantizer = gpu_quantizer
demos/offline_ivf/offline_ivf.py:            return faiss.knn_gpu(
demos/offline_ivf/offline_ivf.py:                self.all_gpu_resources[thread_id],
demos/offline_ivf/offline_ivf.py:        quantizer = faiss.index_cpu_to_all_gpus(index_ivf.quantizer)
demos/offline_ivf/offline_ivf.py:        ngpu = faiss.get_num_gpus()
demos/offline_ivf/offline_ivf.py:        logging.info(f"number of gpus: {ngpu}")
demos/offline_ivf/offline_ivf.py:        self.all_gpu_resources = [
demos/offline_ivf/offline_ivf.py:            faiss.StandardGpuResources() for _ in range(ngpu)
demos/offline_ivf/offline_ivf.py:        # quantizer = faiss.index_cpu_to_all_gpus(index_ivf.quantizer)
demos/offline_ivf/offline_ivf.py:                    quantizer = faiss.index_cpu_to_all_gpus(
demos/offline_ivf/offline_ivf.py:                    prefetch_threads = faiss.get_num_gpus()
demos/offline_ivf/offline_ivf.py:                        threaded=faiss.get_num_gpus() * 8,
demos/offline_ivf/offline_ivf.py:                        computation_threads=faiss.get_num_gpus(),
demos/demo_distributed_kmeans_torch.py:class DatasetAssignDistributedGPU(clustering.DatasetAssign):
demos/demo_distributed_kmeans_torch.py:        D, I = faiss.knn_gpu(
demos/demo_distributed_kmeans_torch.py:        backend="nccl",
demos/demo_distributed_kmeans_torch.py:    device = torch.device(f"cuda:{rank}")
demos/demo_distributed_kmeans_torch.py:    torch.cuda.set_device(device)
demos/demo_distributed_kmeans_torch.py:    res = faiss.StandardGpuResources()
demos/demo_distributed_kmeans_torch.py:    da = DatasetAssignDistributedGPU(res, x, rank, nproc)

```
