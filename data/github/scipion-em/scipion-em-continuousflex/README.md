# https://github.com/scipion-em/scipion-em-continuousflex

```console
continuousflex/protocols/protocol_deep_hemnma_infer.py:DEVICE_CUDA = 0
continuousflex/protocols/protocol_deep_hemnma_infer.py:                      choices=['train on GPUs',
continuousflex/protocols/protocol_deep_hemnma_infer.py:                               'tain on CPUs'], default=DEVICE_CUDA,
continuousflex/protocols/protocol_tomoflow_refine_alignment.py:        group.addParam('N_GPU', params.IntParam, default=1, important=True, allowsNull=True,
continuousflex/protocols/protocol_tomoflow_refine_alignment.py:                              label = 'Parallel processes on GPU',
continuousflex/protocols/protocol_tomoflow_refine_alignment.py:                                   ' (independently). The more powerful your GPU, the higher the number you can choose.')
continuousflex/protocols/protocol_tomoflow_refine_alignment.py:        group.addParam('GPU_list', params.NumericRangeParam,
continuousflex/protocols/protocol_tomoflow_refine_alignment.py:                       label="GPU id(s)",
continuousflex/protocols/protocol_tomoflow_refine_alignment.py:                       help='Select the GPU id(s) that will be used for optical flow calculation.'
continuousflex/protocols/protocol_tomoflow_refine_alignment.py:                            'You can select a list like 0-4, and it will take the GPUs 0 1 2 3 4'
continuousflex/protocols/protocol_tomoflow_refine_alignment.py:        GPUids = np.array(getListFromRangeString(self.GPU_list.get()))
continuousflex/protocols/protocol_tomoflow_refine_alignment.py:        gpu_ps = np.tile(GPUids, mdImgs.size())
continuousflex/protocols/protocol_tomoflow_refine_alignment.py:            gpu_p = gpu_ps[objId-1]
continuousflex/protocols/protocol_tomoflow_refine_alignment.py:                                                                   path_flowx, path_flowy, path_flowz, gpu_p)
continuousflex/protocols/protocol_tomoflow_refine_alignment.py:        Parallel(n_jobs=self.N_GPU.get(), backend="multiprocessing")(delayed(segment)(p) for p in ps)
continuousflex/protocols/protocol_batch_pdb_cluster.py:from pyworkflow.protocol.params import PointerParam, FileParam, USE_GPU, GPU_LIST, BooleanParam, StringParam,LEVEL_ADVANCED
continuousflex/protocols/protocol_batch_pdb_cluster.py:        form.addHidden(USE_GPU, BooleanParam, default=True,
continuousflex/protocols/protocol_batch_pdb_cluster.py:                       label="Use GPU for execution",
continuousflex/protocols/protocol_batch_pdb_cluster.py:                       help="This protocol has both CPU and GPU implementation.\
continuousflex/protocols/protocol_batch_pdb_cluster.py:        form.addHidden(GPU_LIST, StringParam, default='0',
continuousflex/protocols/protocol_batch_pdb_cluster.py:                       label="Choose GPU IDs",
continuousflex/protocols/protocol_batch_pdb_cluster.py:                       help="Add a list of GPU devices that can be used")
continuousflex/protocols/protocol_batch_pdb_cluster.py:                    if self.useGpu.get():
continuousflex/protocols/protocol_batch_pdb_cluster.py:                    if self.useGpu.get():
continuousflex/protocols/protocol_batch_pdb_cluster.py:                            self.runJob('xmipp_cuda_reconstruct_fourier', args,
continuousflex/protocols/protocol_batch_pdb_cluster.py:                                        numberOfMpi=len((self.gpuList.get()).split(',')) + 1)
continuousflex/protocols/protocol_batch_pdb_cluster.py:                            self.runJob('xmipp_cuda_reconstruct_fourier', args)
continuousflex/protocols/protocol_tomoflow.py:        group.addParam('N_GPU', params.IntParam, default=1, important=True, allowsNull=True,
continuousflex/protocols/protocol_tomoflow.py:                              label = 'Parallel processes on GPU',
continuousflex/protocols/protocol_tomoflow.py:                                   ' (independently). The more powerful your GPU, the higher the number you can choose.')
continuousflex/protocols/protocol_tomoflow.py:        group.addParam('GPU_list', params.NumericRangeParam,
continuousflex/protocols/protocol_tomoflow.py:                       label="GPU id(s)",
continuousflex/protocols/protocol_tomoflow.py:                       help='Select the GPU id(s) that will be used for optical flow calculation.'
continuousflex/protocols/protocol_tomoflow.py:                            'You can select a list like 0-4, and it will take the GPUs 0 1 2 3 4'
continuousflex/protocols/protocol_tomoflow.py:        GPUids = np.array(getListFromRangeString(self.GPU_list.get()))
continuousflex/protocols/protocol_tomoflow.py:        gpu_ps = np.tile(GPUids, mdImgs.size())
continuousflex/protocols/protocol_tomoflow.py:            gpu_p = gpu_ps[objId-1]
continuousflex/protocols/protocol_tomoflow.py:                                                                   path_flowx, path_flowy, path_flowz, gpu_p)
continuousflex/protocols/protocol_tomoflow.py:        Parallel(n_jobs=self.N_GPU.get(), backend="multiprocessing")(delayed(segment)(p) for p in ps)
continuousflex/protocols/utilities/processing_dh/models/deep_hemnma.py:            return mlp, proj_imgs.to('cuda:0')
continuousflex/protocols/utilities/deep_hemnma_infer.py:        DEVICE = 'cuda'
continuousflex/protocols/utilities/optflow_run.py:                factor2=100, path_volx='x_OF_3D.vol', path_voly='y_OF_3D.vol', path_volz='z_OF_3D.vol', gpu_id=0):
continuousflex/protocols/utilities/optflow_run.py:    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
continuousflex/protocols/utilities/optflow_run.py:    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
continuousflex/protocols/utilities/optflow_run.py:    import pycuda.autoinit
continuousflex/protocols/utilities/deep_hemnma.py:        DEVICE = 'cuda'
continuousflex/protocols/protocol_deep_hemnma_train.py:DEVICE_CUDA = 0
continuousflex/protocols/protocol_deep_hemnma_train.py:                      choices=['train on GPUs',
continuousflex/protocols/protocol_deep_hemnma_train.py:                               'tain on CPUs'], default = DEVICE_CUDA,
continuousflex/bibtex.py:abstract = {GENeralized-Ensemble SImulation System (GENESIS) is a software package for molecular dynamics (MD) simulation of biological systems. It is designed to extend limitations in system size and accessible time scale by adopting highly parallelized schemes and enhanced conformational sampling algorithms. In this new version, GENESIS 1.1, new functions and advanced algorithms have been added. The all-atom and coarse-grained potential energy functions used in AMBER and GROMACS packages now become available in addition to CHARMM energy functions. The performance of MD simulations has been greatly improved by further optimization, multiple time-step integration, and hybrid (CPU + GPU) computing. The string method and replica-exchange umbrella sampling with flexible collective variable choice are used for finding the minimum free-energy pathway and obtaining free-energy profiles for conformational changes of a macromolecule. These new features increase the usefulness and power of GENESIS for modeling and simulation in biological research. © 2017 Wiley Periodicals, Inc.},
continuousflex/tests/test_workflow_TomoFlow.py:        protRefine = self.newProtocol(FlexProtRefineSubtomoAlign,N_GPU = 1)
continuousflex/tests/test_workflow_TomoFlow.py:        protRefine = self.newProtocol(FlexProtRefineSubtomoAlign,N_GPU = 1)
continuousflex/conda.yaml:      - pycuda==2020.1
continuousflex/templates/mdspace.json.template:        "useGpu": true,
continuousflex/templates/mdspace.json.template:        "gpuList": "0",
continuousflex/__init__.py:            config_path = continuousflex.__path__[0] + '/conda_noCuda.yaml'

```
