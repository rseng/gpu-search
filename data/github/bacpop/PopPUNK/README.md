# https://github.com/bacpop/PopPUNK

```console
web/s_pneumoniae/args.txt:        "gpu_sketch":false,
web/s_pneumoniae/args.txt:        "gpu_dist":false,
web/s_pneumoniae/args.txt:        "gpu_dist":false,
docs/troubleshooting.rst:- Consider the ``--gpu-sketch`` and ``--gpu-dists`` options is applicable,
docs/troubleshooting.rst:  and a GPU is available.
docs/troubleshooting.rst:with ``--gpu-dists``. Please update to ``PopPUNK >=v2.4.0``
docs/index.rst:   gpu.rst
docs/visualisation.rst:- ``--threads``, ``--gpu-dist``, ``--deviceid``, ``--strand-preserved`` -- querying options used if extra distance calculations are needed.
docs/gpu.rst:Using GPUs
docs/gpu.rst:PopPUNK can use GPU acceleration of sketching (only when using sequence reads), distance
docs/gpu.rst:Installing GPU packages
docs/gpu.rst:To use GPU acceleration, PopPUNK uses ``cupy``, ``numba`` and the ``cudatoolkit``
docs/gpu.rst:using conda. The ``cudatoolkit`` packages need to be matched to your CUDA version.
docs/gpu.rst:The command ``nvidia-smi`` can be used to find the supported `CUDA version <https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi>`__.
docs/gpu.rst:Installation of the ``cudatoolkit`` with conda (or the faster conda alternative,
docs/gpu.rst:such as (modify the ``CUDA_VERSION`` variable as appropriate)::
docs/gpu.rst:    export CUDA_VERSION=11.3
docs/gpu.rst:    conda create -n poppunk_gpu -c rapidsai -c nvidia -c conda-forge \
docs/gpu.rst:    -c bioconda -c defaults rapids>=22.12 python=3.8 cudatoolkit=$CUDA_VERSION \
docs/gpu.rst:    conda activate poppunk_gpu
docs/gpu.rst:The version of ``pp-sketchlib`` on conda only supports some GPUs. A more general approach
docs/gpu.rst:versions of the CUDA compiler (``cuda-nvcc``) and runtime API (``cuda-cudart``)
docs/gpu.rst:that match the CUDA version. Although conda can be used, creating such a complex
docs/gpu.rst:    export CUDA_VERSION=11.3
docs/gpu.rst:    mamba create -n poppunk_gpu -c rapidsai -c nvidia -c conda-forge \
docs/gpu.rst:    -c bioconda -c defaults rapids=22.12 python>=3.8 cudatoolkit=$CUDA_VERSION \
docs/gpu.rst:    cuda-nvcc=$CUDA_VERSION cuda-cudart=$CUDA_VERSION networkx cupy numba cmake \
docs/gpu.rst:To correctly build ``pp-sketchlib``, the GPU architecture needs to be correctly
docs/gpu.rst:specified. The ``nvidia-smi`` command can be used to display the GPUs available
docs/gpu.rst:compilation (typically of the form ``sm_*``) using this `guide <https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/>`__
docs/gpu.rst:the compute version to that used by your GPU. See `the CMAKE_CUDA_COMPILER_VERSION
docs/gpu.rst:.. table:: GPU compute versions
docs/gpu.rst:    GPU                Compute version
docs/gpu.rst:You should see a message that the CUDA compiler is found, in which case the compilation
docs/gpu.rst:and installation of sketchlib will include GPU components::
docs/gpu.rst:    -- Looking for a CUDA compiler
docs/gpu.rst:    -- Looking for a CUDA compiler - /usr/local/cuda-11.1/bin/nvcc
docs/gpu.rst:    -- CUDA found, compiling both GPU and CPU code
docs/gpu.rst:    -- The CUDA compiler identification is NVIDIA 11.1.105
docs/gpu.rst:    -- Detecting CUDA compiler ABI info
docs/gpu.rst:    -- Detecting CUDA compiler ABI info - done
docs/gpu.rst:    -- Check for working CUDA compiler: /usr/local/cuda-11.1/bin/nvcc - skipped
docs/gpu.rst:    -- Detecting CUDA compile features
docs/gpu.rst:    -- Detecting CUDA compile features - done
docs/gpu.rst:Selecting a GPU
docs/gpu.rst:A single GPU will be selected on systems where multiple devices are available. For
docs/gpu.rst:Alternatively, all GPU-enabled functions will used device 0 by default. Any GPU can
docs/gpu.rst:be set to device 0 using the system ``CUDA_VISIBLE_DEVICES`` variable, which can be set
docs/gpu.rst:before running PopPUNK; e.g. to use GPU device 1::
docs/gpu.rst:    export CUDA_VISIBLE_DEVICES=1
docs/gpu.rst:Using a GPU
docs/gpu.rst:By default, PopPUNK will use not use GPUs. To use them, you will need to add
docs/gpu.rst:the flag ``--gpu-sketch`` (when constructing or querying a database using reads),
docs/gpu.rst:``--gpu-dist`` (when constructing or querying a database from assemblies or reads),
docs/gpu.rst:``--gpu-model`` (when fitting a DBSCAN model on the GPU), or ``--gpu-graph``
docs/gpu.rst:Note that fitting a model with a GPU is fast, even with a large subsample of points,
docs/gpu.rst:but may be limited by the memory of the GPU device. Therefore it is recommended that
docs/gpu.rst:the transfer of data between CPU and GPU is optimised using the ``--assign-subsample``
docs/gpu.rst:transferred to the GPU memory, speeding up the process, but also increasing the risks
docs/sketching.rst:GPU acceleration
docs/sketching.rst:There are two pieces of heavy computation that can be accelerated with the use of a CUDA-enabled
docs/sketching.rst:GPU:
docs/sketching.rst:- Sketching read data ``--gpu-sketch``.
docs/sketching.rst:- Calculating core and accessory distances ``--gpu-dist``.
docs/sketching.rst:We assume you have a GPU of at least compute capability v7.0 (Tesla) with drivers
docs/sketching.rst:correctly installed. You do not need the CUDA toolkit installed, as all libraries are
docs/sketching.rst:   You will see 'GPU' in the progress message if a GPU is successfully being used. If you
docs/sketching.rst:   see the usual CPU version your install may not have been compiled with CUDA.
docs/sketching.rst:Sketching read data with the GPU is a hybrid algorithm which can take advantage of
docs/sketching.rst:up to around 16 ``--threads`` to keep a typical consumer GPU busy. The sequence data
docs/sketching.rst:   Sketching 128 read sets on GPU device 0
docs/sketching.rst:Calculating distances with the GPU will give slightly different results to CPU distances,
docs/sketching.rst:   Calculating distances on GPU device 0
docs/sketching.rst:   Progress (GPU): 100.0%
docs/sketching.rst:   The GPU which is device 0 will be used by default. If you wish to target another
docs/sketching.rst:   GPU, use the ``--deviceid`` option. This may be important on computing clusters
docs/sketching.rst:   where you must use your job's allocated GPU.
docs/installation.rst:If you want to use GPUs, take a look at :doc:`gpu`.
docs/mst.rst:Using GPU acceleration for the graph
docs/mst.rst:As an extra optimisation, you may add ``--gpu-graph`` to use `cuGraph <https://docs.rapids.ai/api>`__
docs/mst.rst:from the RAPIDS library to calculate the MST on a GPU::
docs/mst.rst:    --microreact --output sparse_mst_viz --threads 4 --gpu-graph
docs/mst.rst:    Calculating MST (GPU part)
docs/query_assignment.rst:- Add ``--gpu-dist``, if you have a GPU available.
docs/query_assignment.rst:- Add ``--gpu-sketch``, if your input is all reads, and you have a GPU available. If
test/test-gpu.py:"""Tests for PopPUNK using GPUs where possible"""
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files references.txt --min-k 13 --k-step 3 --plot-fit 5 --output example_db --overwrite --gpu-dist", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " test-update-gpu.py", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model bgmm --ref-db example_db --K 4 --overwrite --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model dbscan --ref-db example_db --output example_dbscan --overwrite --gpu-graph --gpu-model", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model dbscan --ref-db example_db --output example_dbscan --overwrite --gpu-graph --gpu-model --for-refine", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model dbscan --ref-db example_db --output example_dbscan --overwrite --gpu-graph --gpu-model --assign-subsample 1000", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model dbscan --ref-db example_db --output example_dbscan --overwrite --gpu-graph --gpu-model --model-subsample 5000", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model dbscan --ref-db example_db --output example_dbscan --overwrite --graph-weights --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.2 --overwrite --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --manual-start manual.txt --overwrite --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.2 --overwrite --indiv-refine both --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.2 --overwrite --indiv-refine both --no-local --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.2 --overwrite --unconstrained --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.2 --overwrite --score-idx 1 --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.2 --overwrite --score-idx 2 --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model threshold --threshold 0.003 --ref-db example_db --output example_threshold --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model refine --ref-db example_db --output example_refine --neg-shift 0.15 --summary-sample 15 --overwrite --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model lineage --output example_lineages --ranks 1,2,3,5 --ref-db example_db --overwrite --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --use-model --ref-db example_db --model-dir example_db --output example_use --overwrite --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query some_queries.txt --db example_db --model-dir example_refine --output example_query --overwrite --gpu-dist --gpu-graph --core --accessory", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query some_queries.txt --db example_db --model-dir example_dbscan --output example_query_update --update-db --graph-weights --overwrite --gpu-dist  --gpu-graph", shell=True, check=True) # uses graph weights
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query single_query.txt --db example_db --model-dir example_refine --output example_single_query --update-db --overwrite --gpu-dist  --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk_assign-runner.py --query some_queries.txt --db example_db --model-dir example_lineages --output example_lineage_query --overwrite --gpu-graph --gpu-dist", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_viz --microreact --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_viz --cytoscape --network-file example_db/example_db_graph.csv.gz --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_viz --phandango --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_viz --grapetree --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_viz_subset --microreact --include-files subset.txt --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --query-db example_query --output example_viz_query --microreact --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --previous-clustering example_lineages/example_lineages_lineages.csv --model-dir example_lineages --output example_lineage_viz --microreact --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --distances example_query/example_query.dists --ref-db example_db --model-dir example_lineages --query-db example_lineage_query --output example_viz_query_lineages --microreact --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk_visualise-runner.py --ref-db example_db --output example_mst --microreact --tree both --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk_mst-runner.py --distance-pkl example_db/example_db.dists.pkl --rank-fit example_lineages/example_lineages_rank_5_fit.npz --previous-clustering example_dbscan/example_dbscan_clusters.csv --output example_sparse_mst --no-plot --gpu-graph", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk_mandrake-runner.py --distances example_db/example_db.dists --output example_mandrake --perplexity 5 --use-gpu", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk_references-runner.py --network example_db/example_db_graph.csv.gz --distances example_db/example_db.dists --ref-db example_db --output example_refs --model example_db --use-gpu", shell=True, check=True)
test/test-gpu.py:subprocess.run(python_cmd + " ../poppunk_info-runner.py --db example_db --output example_db.info.csv --use-gpu", shell=True, check=True)
test/test-update-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files rfile12.txt --output batch12 --overwrite --gpu-dist", shell=True, check=True)
test/test-update-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model lineage --ref-db batch12 --ranks 1,2 --gpu-graph", shell=True, check=True)
test/test-update-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files rfile1.txt --output batch1 --overwrite --gpu-dist", shell=True, check=True)
test/test-update-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model lineage --ref-db batch1 --ranks 1,2 --gpu-graph", shell=True, check=True)
test/test-update-gpu.py:subprocess.run(python_cmd + " ../poppunk_assign-runner.py --db batch1 --query rfile2.txt --output batch2 --update-db --overwrite --gpu-graph --gpu-dist", shell=True, check=True)
test/test-update-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --create-db --r-files rfile123.txt --output batch123 --overwrite  --gpu-dist", shell=True, check=True)
test/test-update-gpu.py:subprocess.run(python_cmd + " ../poppunk-runner.py --fit-model lineage --ref-db batch123 --ranks 1,2  --gpu-graph  --gpu-dist", shell=True, check=True)
test/test-update-gpu.py:subprocess.run(python_cmd + " ../poppunk_assign-runner.py --db batch2 --query rfile3.txt --output batch3 --update-db --overwrite --gpu-graph --gpu-dist", shell=True, check=True)
PopPUNK/visualise.py:    from numba import cuda
PopPUNK/visualise.py:    gpu_lib = True
PopPUNK/visualise.py:    gpu_lib = False
PopPUNK/visualise.py:    other.add_argument('--gpu-dist', default=False, action='store_true', help='Use a GPU when calculating distances [default = False]')
PopPUNK/visualise.py:    other.add_argument('--gpu-graph', default=False, action='store_true', help='Use a GPU when calculating graphs [default = False]')
PopPUNK/visualise.py:    other.add_argument('--deviceid', default=0, type=int, help='CUDA device ID, if using GPU [default = 0]')
PopPUNK/visualise.py:                            gpu_dist,
PopPUNK/visualise.py:                            gpu_graph,
PopPUNK/visualise.py:                                                        use_gpu = gpu_dist,
PopPUNK/visualise.py:                                                        use_gpu=gpu_dist,
PopPUNK/visualise.py:                                                            use_gpu=gpu_dist,
PopPUNK/visualise.py:                                                            gpu_graph = gpu_graph)
PopPUNK/visualise.py:                                                                use_gpu = gpu_graph,
PopPUNK/visualise.py:                        if gpu_graph:
PopPUNK/visualise.py:                    mst_graph = generate_minimum_spanning_tree(G, gpu_graph)
PopPUNK/visualise.py:                                    use_gpu = gpu_graph)
PopPUNK/visualise.py:                    if gpu_graph:
PopPUNK/visualise.py:                                                    use_gpu = False)
PopPUNK/visualise.py:                                                use_gpu=gpu_graph,
PopPUNK/visualise.py:            genomeNetwork = load_network_file(network_file, use_gpu = gpu_graph)
PopPUNK/visualise.py:            if gpu_graph:
PopPUNK/visualise.py:                genomeNetwork = remove_nodes_from_graph(genomeNetwork, all_seq, viz_subset, use_gpu = gpu_graph)
PopPUNK/visualise.py:            genomeNetwork = sparse_mat_to_network(sparse_mat, combined_seq, use_gpu = gpu_graph)
PopPUNK/visualise.py:                            args.gpu_dist,
PopPUNK/visualise.py:                            args.gpu_graph,
PopPUNK/network.py:# Load GPU libraries
PopPUNK/network.py:    from numba import cuda
PopPUNK/network.py:from .utils import check_and_set_gpu
PopPUNK/network.py:                  core_only = False, accessory_only = False, use_gpu = False):
PopPUNK/network.py:            use_gpu (bool)
PopPUNK/network.py:    if use_gpu:
PopPUNK/network.py:    genomeNetwork = load_network_file(network_file, use_gpu = use_gpu)
PopPUNK/network.py:    checkNetworkVertexCount(refList, genomeNetwork, use_gpu)
PopPUNK/network.py:def load_network_file(fn, use_gpu = False):
PopPUNK/network.py:            use_gpu (bool)
PopPUNK/network.py:    if use_gpu:
PopPUNK/network.py:def checkNetworkVertexCount(seq_list, G, use_gpu):
PopPUNK/network.py:        use_gpu (bool)
PopPUNK/network.py:    vertex_list = set(get_vertex_list(G, use_gpu = use_gpu))
PopPUNK/network.py:                        existingRefs = None, threads = 1, use_gpu = False):
PopPUNK/network.py:           use_gpu (bool)
PopPUNK/network.py:    if use_gpu:
PopPUNK/network.py:                        use_gpu = False):
PopPUNK/network.py:        use_gpu (bool)
PopPUNK/network.py:        prev_G = load_network_file(prev_G_fn, use_gpu = use_gpu)
PopPUNK/network.py:    if use_gpu:
PopPUNK/network.py:def print_network_summary(G, sample_size = None, betweenness_sample = betweenness_sample_default, use_gpu = False):
PopPUNK/network.py:            a GPU. Smaller numbers are faster but less precise [default = 100]
PopPUNK/network.py:        use_gpu (bool)
PopPUNK/network.py:            Whether to use GPUs for network construction
PopPUNK/network.py:                                        use_gpu = use_gpu)
PopPUNK/network.py:                                previous_pkl = None, vertex_labels = None, weights = False, use_gpu = False):
PopPUNK/network.py:        use_gpu (bool)
PopPUNK/network.py:            Whether to use GPUs for network construction
PopPUNK/network.py:                                                                            use_gpu = use_gpu)
PopPUNK/network.py:                                                            use_gpu = use_gpu)
PopPUNK/network.py:                                        use_gpu = False):
PopPUNK/network.py:            a GPU. Smaller numbers are faster but less precise [default = 100]
PopPUNK/network.py:        use_gpu (bool)
PopPUNK/network.py:            Whether to use GPUs for network construction
PopPUNK/network.py:    if use_gpu:
PopPUNK/network.py:            edge_gpu_matrix = cuda.to_device(edge_array)
PopPUNK/network.py:            G_df = cudf.DataFrame(edge_gpu_matrix, columns = ['source','destination'])
PopPUNK/network.py:                                        use_gpu = use_gpu)
PopPUNK/network.py:                                            use_gpu = use_gpu)
PopPUNK/network.py:                              use_gpu = use_gpu)
PopPUNK/network.py:                                use_gpu = False):
PopPUNK/network.py:            a GPU. Smaller numbers are faster but less precise [default = 100]
PopPUNK/network.py:        use_gpu (bool)
PopPUNK/network.py:            Whether to use GPUs for network construction
PopPUNK/network.py:                                                                                use_gpu = use_gpu)
PopPUNK/network.py:        if use_gpu:
PopPUNK/network.py:    if use_gpu:
PopPUNK/network.py:                                            use_gpu = use_gpu)
PopPUNK/network.py:                              use_gpu = use_gpu)
PopPUNK/network.py:                                            use_gpu = False):
PopPUNK/network.py:            a GPU. Smaller numbers are faster but less precise [default = 100]
PopPUNK/network.py:        use_gpu (bool)
PopPUNK/network.py:            Whether to use GPUs for network construction
PopPUNK/network.py:    if use_gpu:
PopPUNK/network.py:                                    use_gpu = use_gpu)
PopPUNK/network.py:                              use_gpu = use_gpu)
PopPUNK/network.py:def construct_dense_weighted_network(rlist, distMat, weights_type = None, use_gpu = False):
PopPUNK/network.py:        use_gpu (bool)
PopPUNK/network.py:            Whether to use GPUs for network construction
PopPUNK/network.py:    if use_gpu:
PopPUNK/network.py:        # Construct network with GPU via data frame
PopPUNK/network.py:    summarise = True, sample_size = None, use_gpu = False):
PopPUNK/network.py:            a GPU. Smaller numbers are faster but less precise [default = 100]
PopPUNK/network.py:        use_gpu (bool)
PopPUNK/network.py:            Whether to use GPUs for network construction
PopPUNK/network.py:                                            use_gpu = use_gpu)
PopPUNK/network.py:                              use_gpu = use_gpu)
PopPUNK/network.py:                    subsample = None, use_gpu = False):
PopPUNK/network.py:            a GPU. Smaller numbers are faster but less precise [default = 100]
PopPUNK/network.py:        use_gpu (bool)
PopPUNK/network.py:    if use_gpu:
PopPUNK/network.py:        if use_gpu:
PopPUNK/network.py:                      use_gpu = False):
PopPUNK/network.py:        use_gpu (bool)
PopPUNK/network.py:                                            use_gpu = use_gpu)
PopPUNK/network.py:        if use_gpu:
PopPUNK/network.py:                                                    use_gpu = use_gpu)
PopPUNK/network.py:                  use_gpu = False):
PopPUNK/network.py:        use_gpu (bool)
PopPUNK/network.py:    if use_gpu:
PopPUNK/network.py:def get_vertex_list(G, use_gpu = False):
PopPUNK/network.py:       use_gpu (bool)
PopPUNK/network.py:    if use_gpu:
PopPUNK/network.py:                use_gpu = False):
PopPUNK/network.py:       use_gpu (bool)
PopPUNK/network.py:    if use_gpu:
PopPUNK/network.py:def sparse_mat_to_network(sparse_mat, rlist, use_gpu = False):
PopPUNK/network.py:       use_gpu (bool)
PopPUNK/network.py:         Whether GPU libraries should be used
PopPUNK/network.py:    if use_gpu:
PopPUNK/network.py:def prune_graph(prefix, reflist, samples_to_keep, output_db_name, threads, use_gpu):
PopPUNK/network.py:       use_gpu (bool)
PopPUNK/network.py:    if use_gpu:
PopPUNK/network.py:          G = load_network_file(network_fn, use_gpu = use_gpu)
PopPUNK/network.py:          G_new = remove_nodes_from_graph(G, reflist, samples_to_keep, use_gpu)
PopPUNK/network.py:                      use_gpu = use_gpu)
PopPUNK/network.py:def remove_nodes_from_graph(G,reflist, samples_to_keep, use_gpu):
PopPUNK/network.py:       use_gpu (bool)
PopPUNK/network.py:    if use_gpu:
PopPUNK/network.py:def remove_non_query_components(G, rlist, qlist, use_gpu = False):
PopPUNK/network.py:        use_gpu (bool)
PopPUNK/network.py:            Whether to use GPUs for network construction
PopPUNK/network.py:    if use_gpu:
PopPUNK/network.py:        sys.stderr.write('Saving partial query graphs is not compatible with GPU networks yet\n')
PopPUNK/info.py:# Load GPU libraries
PopPUNK/info.py:    gpu_lib = True
PopPUNK/info.py:    gpu_lib = False
PopPUNK/info.py:    parser.add_argument('--use-gpu',
PopPUNK/info.py:                        help='Whether GPU libraries should be used in network analysis')
PopPUNK/info.py:    from .utils import check_and_set_gpu
PopPUNK/info.py:    # Check whether GPU libraries can be loaded
PopPUNK/info.py:    use_gpu = check_and_set_gpu(args.use_gpu, gpu_lib, quit_on_fail = False)
PopPUNK/info.py:        if use_gpu:
PopPUNK/info.py:        G = load_network_file(network_file, use_gpu = False)
PopPUNK/info.py:        if use_gpu:
PopPUNK/info.py:            G = load_network_file(network_file, use_gpu = True)
PopPUNK/info.py:            sys.stderr.write('Unable to load necessary GPU libraries\n')
PopPUNK/info.py:        G = sparse_mat_to_network(sparse_mat, sample_names, use_gpu = use_gpu)
PopPUNK/info.py:    print_network_summary(G, betweenness_sample = betweenness_sample_default, use_gpu = args.use_gpu)
PopPUNK/info.py:        if use_gpu:
PopPUNK/sparse_mst.py:# Load GPU libraries
PopPUNK/sparse_mst.py:    from numba import cuda
PopPUNK/sparse_mst.py:    gpu_lib = True
PopPUNK/sparse_mst.py:    gpu_lib = False
PopPUNK/sparse_mst.py:from .utils import check_and_set_gpu
PopPUNK/sparse_mst.py:    other.add_argument('--gpu-graph', default=False, action='store_true',
PopPUNK/sparse_mst.py:def generate_mst_from_sparse_input(sparse_mat, rlist, old_rlist = None, previous_mst = None, gpu_graph = False):
PopPUNK/sparse_mst.py:    if gpu_graph:
PopPUNK/sparse_mst.py:                                                                                  use_gpu = gpu_graph)
PopPUNK/sparse_mst.py:    G = generate_minimum_spanning_tree(G, gpu_graph)
PopPUNK/sparse_mst.py:                                        gpu_graph = args.gpu_graph)
PopPUNK/sparse_mst.py:                    use_gpu = args.gpu_graph)
PopPUNK/sparse_mst.py:    mst_as_tree = mst_to_phylogeny(G, rlist, use_gpu = args.gpu_graph)
PopPUNK/sparse_mst.py:        if args.gpu_graph:
PopPUNK/trees.py:from .utils import check_and_set_gpu
PopPUNK/trees.py:# Load GPU libraries
PopPUNK/trees.py:    from numba import cuda
PopPUNK/trees.py:def mst_to_phylogeny(mst_network, names, use_gpu = False):
PopPUNK/trees.py:       use_gpu (bool)
PopPUNK/trees.py:            Whether to use GPU-specific libraries for processing
PopPUNK/trees.py:    if use_gpu:
PopPUNK/trees.py:        if use_gpu:
PopPUNK/models.py:# Load GPU libraries
PopPUNK/models.py:    from numba import cuda
PopPUNK/models.py:from .utils import check_and_set_gpu
PopPUNK/models.py:                   use_gpu = False):
PopPUNK/models.py:        use_gpu (bool)
PopPUNK/models.py:            Whether to load npz file with GPU libraries
PopPUNK/models.py:    def __init__(self, outPrefix, use_gpu = False, max_batch_size = 5000, max_samples = 100000, assign_points = True):
PopPUNK/models.py:        self.use_gpu = use_gpu # Updated below
PopPUNK/models.py:    def fit(self, X, max_num_clusters, min_cluster_prop, use_gpu = False):
PopPUNK/models.py:            use_gpu (bool)
PopPUNK/models.py:                Whether GPU algorithms should be used in DBSCAN fitting
PopPUNK/models.py:        # Check on initialisation of GPU libraries and memory
PopPUNK/models.py:        # Convert to cupy if using GPU to avoid implicit numpy conversion below
PopPUNK/models.py:        if use_gpu:
PopPUNK/models.py:                gpu_lib = True
PopPUNK/models.py:                gpu_lib = False
PopPUNK/models.py:            # check on GPU
PopPUNK/models.py:            use_gpu = check_and_set_gpu(use_gpu,
PopPUNK/models.py:                                gpu_lib,
PopPUNK/models.py:            if use_gpu:
PopPUNK/models.py:                self.use_gpu = True
PopPUNK/models.py:                self.use_gpu = False
PopPUNK/models.py:                                                                use_gpu = use_gpu)
PopPUNK/models.py:              if use_gpu:
PopPUNK/models.py:                              use_gpu = use_gpu)
PopPUNK/models.py:        elif not use_gpu:
PopPUNK/models.py:            y = self.assign(X, max_batch_size = self.max_batch_size, use_gpu = use_gpu)
PopPUNK/models.py:            y = self.assign(self.subsampled_X, max_batch_size = self.max_batch_size, use_gpu = use_gpu)
PopPUNK/models.py:             use_gpu=self.use_gpu)
PopPUNK/models.py:        if 'use_gpu' in fit_npz.keys():
PopPUNK/models.py:            self.use_gpu = fit_npz['use_gpu']
PopPUNK/models.py:            self.use_gpu = False
PopPUNK/models.py:        if self.use_gpu:
PopPUNK/models.py:                                        use_gpu=self.use_gpu),
PopPUNK/models.py:                            self.use_gpu)
PopPUNK/models.py:    def assign(self, X, no_scale = False, progress = True, max_batch_size = 5000, use_gpu = False):
PopPUNK/models.py:            use_gpu (bool)
PopPUNK/models.py:                Use GPU-enabled algorithms for clustering
PopPUNK/models.py:            if use_gpu:
PopPUNK/models.py:            betweenness_sample = betweenness_sample_default, sample_size = None, use_gpu = False):
PopPUNK/models.py:                a GPU. Smaller numbers are faster but less precise [default = 100]
PopPUNK/models.py:            use_gpu (bool)
PopPUNK/models.py:                    use_gpu = use_gpu)
PopPUNK/models.py:                        use_gpu = use_gpu)
PopPUNK/models.py:                                    use_gpu = use_gpu)
PopPUNK/models.py:    def __init__(self, outPrefix, ranks, max_search_depth, reciprocal_only, count_unique_distances, dist_col = None, use_gpu = False):
PopPUNK/models.py:        self.use_gpu = use_gpu
PopPUNK/models.py:        if self.use_gpu:
PopPUNK/models.py:            if self.use_gpu:
PopPUNK/models.py:        # Convert data structures if using GPU
PopPUNK/models.py:        if self.use_gpu:
PopPUNK/plot.py:def plot_dbscan_results(X, y, n_clusters, out_prefix, use_gpu):
PopPUNK/plot.py:        use_gpu (bool)
PopPUNK/plot.py:            Whether model was fitted with GPU-enabled code
PopPUNK/plot.py:    # Convert data if from GPU
PopPUNK/plot.py:    if use_gpu:
PopPUNK/plot.py:                         use_gpu = False, device_id = 0):
PopPUNK/plot.py:        use_gpu (bool)
PopPUNK/plot.py:            Whether to use a GPU for t-SNE generation
PopPUNK/plot.py:            Device ID of GPU to be used
PopPUNK/plot.py:                       use_gpu=use_gpu, device_id=device_id)
PopPUNK/refine.py:# Load GPU libraries
PopPUNK/refine.py:    from numba import cuda
PopPUNK/refine.py:from .utils import check_and_set_gpu
PopPUNK/refine.py:              use_gpu = False):
PopPUNK/refine.py:            a GPU. Smaller numbers are faster but less precise [default = 100]
PopPUNK/refine.py:        use_gpu (bool)
PopPUNK/refine.py:        if use_gpu:
PopPUNK/refine.py:                                   use_gpu = True),
PopPUNK/refine.py:                                                use_gpu = False),
PopPUNK/refine.py:                                        use_gpu = use_gpu))
PopPUNK/refine.py:                            betweenness_sample, sample_size, use_gpu)
PopPUNK/refine.py:                 use_gpu = False):
PopPUNK/refine.py:            a GPU. Smaller numbers are faster but less precise [default = 100]
PopPUNK/refine.py:        use_gpu (bool)
PopPUNK/refine.py:                use_gpu = use_gpu)
PopPUNK/refine.py:                write_clusters = None, sample_size = None, use_gpu = False):
PopPUNK/refine.py:            a GPU. Smaller numbers are faster but less precise [default = 100]
PopPUNK/refine.py:        use_gpu (bool)
PopPUNK/refine.py:    if use_gpu:
PopPUNK/refine.py:    if use_gpu:
PopPUNK/refine.py:                                              use_gpu = use_gpu)
PopPUNK/refine.py:                if use_gpu:
PopPUNK/refine.py:                                use_gpu = use_gpu)
PopPUNK/refine.py:                                  use_gpu=use_gpu)
PopPUNK/refine.py:               sample_size = None, use_gpu = False):
PopPUNK/refine.py:            a GPU. Smaller numbers are faster but less precise [default = 100]
PopPUNK/refine.py:        use_gpu (bool)
PopPUNK/refine.py:                                        use_gpu = use_gpu)
PopPUNK/refine.py:                            use_gpu = use_gpu)[1][score_idx]
PopPUNK/refine.py:                 use_gpu = False):
PopPUNK/refine.py:            a GPU. Smaller numbers are faster but less precise [default = 100]
PopPUNK/refine.py:        use_gpu (bool)
PopPUNK/refine.py:                                use_gpu = use_gpu)
PopPUNK/assign.py:    other.add_argument('--gpu-sketch', default=False, action='store_true', help='Use a GPU when calculating sketches (read data only) [default = False]')
PopPUNK/assign.py:    other.add_argument('--gpu-dist', default=False, action='store_true', help='Use a GPU when calculating distances [default = False]')
PopPUNK/assign.py:    other.add_argument('--gpu-graph', default=False, action='store_true', help='Use a GPU when constructing networks [default = False]')
PopPUNK/assign.py:    other.add_argument('--deviceid', default=0, type=int, help='CUDA device ID, if using GPU [default = 0]')
PopPUNK/assign.py:                 args.gpu_sketch,
PopPUNK/assign.py:                 args.gpu_dist,
PopPUNK/assign.py:                 args.gpu_graph,
PopPUNK/assign.py:                 gpu_sketch,
PopPUNK/assign.py:                 gpu_dist,
PopPUNK/assign.py:                 gpu_graph,
PopPUNK/assign.py:                                use_gpu = gpu_sketch,
PopPUNK/assign.py:                    gpu_dist,
PopPUNK/assign.py:                    gpu_graph,
PopPUNK/assign.py:                 gpu_dist,
PopPUNK/assign.py:                 gpu_graph,
PopPUNK/assign.py:                                      use_gpu = gpu_dist)
PopPUNK/assign.py:                                      use_gpu = gpu_dist)
PopPUNK/assign.py:                                                                       use_gpu = gpu_graph,
PopPUNK/assign.py:                                  use_gpu = gpu_graph)
PopPUNK/assign.py:                             use_gpu = gpu_graph)
PopPUNK/assign.py:            n_vertices = len(get_vertex_list(genomeNetwork, use_gpu = gpu_graph))
PopPUNK/assign.py:                                        use_gpu = gpu_graph)
PopPUNK/assign.py:                                                use_gpu = gpu_graph)}
PopPUNK/assign.py:                                                    use_gpu = gpu_graph)
PopPUNK/assign.py:                                use_gpu = gpu_graph)
PopPUNK/assign.py:                                use_gpu = gpu_graph)
PopPUNK/assign.py:                                            use_gpu = gpu_graph)
PopPUNK/assign.py:                                    use_gpu = gpu_graph)
PopPUNK/assign.py:                genomeNetwork, pruned_isolate_lists = remove_non_query_components(genomeNetwork, rNames, qNames, use_gpu = gpu_graph)
PopPUNK/assign.py:                    save_network(genomeNetwork[min(model.ranks)], prefix = output, suffix = '_graph', use_gpu = gpu_graph)
PopPUNK/assign.py:                    save_network(genomeNetwork, prefix = output, suffix = graph_suffix, use_gpu = gpu_graph)
PopPUNK/sketchlib.py:                        use_gpu = False, deviceid = 0):
PopPUNK/sketchlib.py:        use_gpu (bool)
PopPUNK/sketchlib.py:            Use GPU for read sketching
PopPUNK/sketchlib.py:            GPU device id
PopPUNK/sketchlib.py:                                   use_gpu=use_gpu,
PopPUNK/sketchlib.py:                  threads = 1, use_gpu = False, deviceid = 0):
PopPUNK/sketchlib.py:        use_gpu (bool)
PopPUNK/sketchlib.py:            Use a GPU for querying
PopPUNK/sketchlib.py:            Index of the CUDA GPU device to use
PopPUNK/sketchlib.py:                                             use_gpu=use_gpu,
PopPUNK/sketchlib.py:                                                    use_gpu = False)
PopPUNK/sketchlib.py:                                                        use_gpu = False)
PopPUNK/sketchlib.py:                                             use_gpu=use_gpu,
PopPUNK/sketchlib.py:                                                use_gpu = False)
PopPUNK/sketchlib.py:                                                    use_gpu = False)
PopPUNK/qc.py:                   strand_preserved=False, threads=1, use_gpu=False):
PopPUNK/qc.py:        use_gpu (bool)
PopPUNK/qc.py:            Whether GPU libraries were used to generate the original network.
PopPUNK/qc.py:                    use_gpu)
PopPUNK/__main__.py:            help='Number of sequences used to estimate betweeness with a GPU [default = 100]',
PopPUNK/__main__.py:    other.add_argument('--gpu-sketch', default=False, action='store_true', help='Use a GPU when calculating sketches (read data only) [default = False]')
PopPUNK/__main__.py:    other.add_argument('--gpu-dist', default=False, action='store_true', help='Use a GPU when calculating distances [default = False]')
PopPUNK/__main__.py:    other.add_argument('--gpu-model', default=False, action='store_true', help='Use a GPU when fitting a model [default = False]')
PopPUNK/__main__.py:    other.add_argument('--gpu-graph', default=False, action='store_true', help='Use a GPU when calculating networks [default = False]')
PopPUNK/__main__.py:    other.add_argument('--deviceid', default=0, type=int, help='CUDA device ID, if using GPU [default = 0]')
PopPUNK/__main__.py:    from .utils import check_and_set_gpu
PopPUNK/__main__.py:    # Check on initialisation of GPU libraries and memory
PopPUNK/__main__.py:        from numba import cuda
PopPUNK/__main__.py:        gpu_lib = True
PopPUNK/__main__.py:        gpu_lib = False
PopPUNK/__main__.py:    args.gpu_graph = check_and_set_gpu(args.gpu_graph,
PopPUNK/__main__.py:                                        gpu_lib,
PopPUNK/__main__.py:                           args.gpu_graph)
PopPUNK/__main__.py:                                   use_gpu = args.gpu_graph)
PopPUNK/__main__.py:                                        args.gpu_model)
PopPUNK/__main__.py:                                            args.gpu_graph)
PopPUNK/__main__.py:                                    use_gpu = args.gpu_graph)
PopPUNK/__main__.py:                                                     use_gpu = args.gpu_graph)
PopPUNK/__main__.py:                                                                        use_gpu = args.gpu_graph,
PopPUNK/__main__.py:                                    use_gpu = args.gpu_graph)
PopPUNK/__main__.py:                                  use_gpu = args.gpu_graph)
PopPUNK/__main__.py:        checkNetworkVertexCount(refList, genomeNetwork, use_gpu = args.gpu_graph)
PopPUNK/__main__.py:                                                     use_gpu = args.gpu_graph)}
PopPUNK/__main__.py:        save_network(genomeNetwork, prefix = output, suffix = "_graph", use_gpu = args.gpu_graph)
PopPUNK/__main__.py:                                                             use_gpu = args.gpu_graph)
PopPUNK/__main__.py:                                      use_gpu = args.gpu_graph)
PopPUNK/__main__.py:                                    use_gpu = args.gpu_graph)
PopPUNK/__main__.py:                                        use_gpu = args.gpu_graph)
PopPUNK/__main__.py:                                    use_gpu = args.gpu_graph)
PopPUNK/lineages.py:    aGroup.add_argument('--gpu-sketch', help="Use GPU for sketching",
PopPUNK/lineages.py:    aGroup.add_argument('--gpu-dist', help="Use GPU for distance calculations",
PopPUNK/lineages.py:    aGroup.add_argument('--gpu-graph', help="Use GPU for graph analysis",
PopPUNK/lineages.py:    aGroup.add_argument('--deviceid',   help="Device ID of GPU",
PopPUNK/lineages.py:                      use_gpu = args.gpu_graph)
PopPUNK/lineages.py:                                                            use_gpu = args.gpu_graph,
PopPUNK/lineages.py:                            use_gpu = args.gpu_graph)
PopPUNK/lineages.py:                                  use_gpu = args.gpu_graph)
PopPUNK/lineages.py:                                use_gpu = args.gpu_sketch,
PopPUNK/lineages.py:                    args.gpu_dist,
PopPUNK/lineages.py:                    args.gpu_graph,
PopPUNK/lineages.py:                            args.gpu_dist,
PopPUNK/lineages.py:                            args.gpu_graph,
PopPUNK/utils.py:    from numba import cuda
PopPUNK/utils.py:                                use_gpu = args.gpu_sketch,
PopPUNK/utils.py:                            use_gpu = args.gpu_dist,
PopPUNK/utils.py:def check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = False):
PopPUNK/utils.py:    """Check GPU libraries can be loaded and set managed memory.
PopPUNK/utils.py:        use_gpu (bool)
PopPUNK/utils.py:            Whether GPU packages have been requested
PopPUNK/utils.py:        gpu_lib (bool)
PopPUNK/utils.py:            Whether GPU packages are available
PopPUNK/utils.py:        use_gpu (bool)
PopPUNK/utils.py:            Whether GPU packages can be used
PopPUNK/utils.py:    # load CUDA libraries
PopPUNK/utils.py:    if use_gpu and not gpu_lib:
PopPUNK/utils.py:            sys.stderr.write('Unable to load GPU libraries; exiting\n')
PopPUNK/utils.py:            sys.stderr.write('Unable to load GPU libraries; using CPU libraries '
PopPUNK/utils.py:            use_gpu = False
PopPUNK/utils.py:    if use_gpu:
PopPUNK/utils.py:            cupy.cuda.set_allocator(rmm.allocators.cupy.rmm_cupy_allocator)
PopPUNK/utils.py:        if "cuda" in sys.modules:
PopPUNK/utils.py:            cuda.set_memory_manager(rmm.allocators.numba.RMMNumbaManager)
PopPUNK/utils.py:    return use_gpu
PopPUNK/mandrake.py:    from SCE import wtsne_gpu_fp64
PopPUNK/mandrake.py:    gpu_fn_available = True
PopPUNK/mandrake.py:    gpu_fn_available = False
PopPUNK/mandrake.py:                       maxIter = 10000000, n_threads = 1, use_gpu = False, device_id = 0):
PopPUNK/mandrake.py:        use_gpu (bool)
PopPUNK/mandrake.py:            Whether to use GPU libraries
PopPUNK/mandrake.py:            Device ID of GPU to be used
PopPUNK/mandrake.py:        # Set up function call with either CPU or GPU
PopPUNK/mandrake.py:        gpu_analysis_complete = False
PopPUNK/mandrake.py:          if use_gpu and gpu_fn_available:
PopPUNK/mandrake.py:              sys.stderr.write("Running on GPU\n")
PopPUNK/mandrake.py:              wtsne_call = partial(wtsne_gpu_fp64,
PopPUNK/mandrake.py:              gpu_analysis_complete = True
PopPUNK/mandrake.py:          # If installed through conda/mamba mandrake is not GPU-enabled by default
PopPUNK/mandrake.py:          sys.stderr.write('Mandrake analysis with GPU failed; trying with CPU\n')
PopPUNK/mandrake.py:        if not gpu_analysis_complete:
PopPUNK/mandrake.py:    parser.add_argument('--use-gpu', help='Whether to use GPU libraries for t-SNE calculation', default = False, action='store_true')
PopPUNK/mandrake.py:    parser.add_argument('--device-id', help="Device ID of GPU to use", type=int, default=0)
PopPUNK/mandrake.py:                       use_gpu = args.use_gpu,
PopPUNK/reference_pick.py:    other.add_argument('--use-gpu', default=False, action='store_true', help='Whether to use GPUs')
PopPUNK/reference_pick.py:    genomeNetwork = load_network_file(args.network, use_gpu = args.use_gpu)
PopPUNK/reference_pick.py:    if args.use_gpu:
PopPUNK/reference_pick.py:                            use_gpu = args.use_gpu)
PopPUNK/reference_pick.py:                    use_gpu = args.use_gpu)
PopPUNK/dbscan.py:from .utils import check_and_set_gpu
PopPUNK/dbscan.py:def fitDbScan(X, min_samples, min_cluster_size, cache_out, use_gpu = False):
PopPUNK/dbscan.py:        use_gpu (bool)
PopPUNK/dbscan.py:            Whether GPU algorithms should be used in DBSCAN fitting
PopPUNK/dbscan.py:    if use_gpu:
PopPUNK/dbscan.py:      sys.stderr.write('Fitting HDBSCAN model using a GPU\n')
README.md:`--update-db`. We have also fixed a number of bugs with GPU distances. These are
scripts/poppunk_batch_mst.py:    aGroup.add_argument('--gpu-dist', help='Use GPU for distance calculations',
scripts/poppunk_batch_mst.py:    aGroup.add_argument('--gpu-graph', help='Use GPU for network analysis',
scripts/poppunk_batch_mst.py:    aGroup.add_argument('--deviceid', help='GPU device ID (int)',
scripts/poppunk_batch_mst.py:        # GPU options
scripts/poppunk_batch_mst.py:        if args.gpu_dist:
scripts/poppunk_batch_mst.py:            create_db_cmd += " --gpu-dist --deviceid " + str(args.deviceid)
scripts/poppunk_batch_mst.py:            # GPU options
scripts/poppunk_batch_mst.py:            if args.gpu_graph:
scripts/poppunk_batch_mst.py:                mst_command = mst_command + " --gpu-graph"
scripts/poppunk_batch_mst.py:            # GPU options
scripts/poppunk_batch_mst.py:            if args.gpu_dist:
scripts/poppunk_batch_mst.py:                assign_cmd = assign_cmd + " --gpu-dist --deviceid " + str(args.deviceid)
scripts/poppunk_batch_mst.py:                if args.gpu_graph:
scripts/poppunk_batch_mst.py:                    mst_command = mst_command + " --gpu-graph"
scripts/poppunk_batch_mst.py:            if args.gpu_graph:
scripts/poppunk_batch_mst.py:                mst_command = mst_command + " --gpu-graph"
scripts/poppunk_iterate.py:    use_gpu = False
scripts/poppunk_iterate.py:                use_gpu,

```
