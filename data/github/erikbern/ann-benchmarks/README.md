# https://github.com/erikbern/ann-benchmarks

```console
README.md:* We mainly support CPU-based ANN algorithms. GPU support exists for FAISS, but it has to be compiled with GPU support locally and experiments must be run using the flags `--local --batch`. 
ann_benchmarks/algorithms/faiss_gpu/config.yml:    constructor: FaissGPU
ann_benchmarks/algorithms/faiss_gpu/config.yml:    module: ann_benchmarks.algorithms.faiss_gpu
ann_benchmarks/algorithms/faiss_gpu/config.yml:    name: faiss-gpu
ann_benchmarks/algorithms/faiss_gpu/module.py:# https://github.com/facebookresearch/faiss/blob/master/benchs/bench_gpu_sift1m.py  # noqa
ann_benchmarks/algorithms/faiss_gpu/module.py:class FaissGPU(BaseANN):
ann_benchmarks/algorithms/faiss_gpu/module.py:        self.name = "FaissGPU(n_bits={}, n_probes={})".format(n_bits, n_probes)
ann_benchmarks/algorithms/faiss_gpu/module.py:        self._res = faiss.StandardGpuResources()
ann_benchmarks/algorithms/faiss_gpu/module.py:        self._index = faiss.GpuIndexIVFFlat(self._res, len(X[0]), self._n_bits, faiss.METRIC_L2)
ann_benchmarks/algorithms/faiss_gpu/module.py:        # co = faiss.GpuClonerOptions()
ann_benchmarks/algorithms/faiss_gpu/module.py:        # self._index = faiss.index_cpu_to_gpu(self._res, 0,
ann_benchmarks/datasets.py:    # (on my desktop/gpu this only takes 4-5 seconds to train - but

```
