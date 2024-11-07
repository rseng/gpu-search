# https://github.com/DeepRank/deeprank

```console
setup.py:        'cuda': ['pycuda'],
docs/conf.py:    'torch.cuda',
test/test_learn.py:                          cuda=False, plot=True, outdir=out)
test/test_learn.py:                          cuda=False, plot=True, outdir=out)
test/test_learn.py:                          cuda=False, plot=True, outdir=out)
test/test_learn.py:                          cuda=False, plot=True, outdir=out)
test/test_generate_cuda.py:    import pycuda
test/test_generate_cuda.py:class TestGenerateCUDA(unittest.TestCase):
test/test_generate_cuda.py:    gpu_block = [8, 8, 8]
test/test_generate_cuda.py:    h5file = '1ak4_cuda.hdf5'
test/test_generate_cuda.py:    def test_generate_cuda(self):
test/test_generate_cuda.py:            database.tune_cuda_kernel(grid_info, func='gaussian')
test/test_generate_cuda.py:            database.test_cuda(grid_info, self.gpu_block, func='gaussian')
test/test_generate_cuda.py:                cuda=True,
test/test_generate_cuda.py:                gpu_block=self.gpu_block)
README.md:                  cuda=False,plot=True,outdir=out)
README.md:We then create a `NeuralNet` instance that takes the dataset as input argument. Several options are available to specify the task to do, the GPU use, etc ... We then have simply to train the model. Simple !
deeprank/generate/GridTools.py:                 cuda=False, gpu_block=None, cuda_func=None, cuda_atomic=None,
deeprank/generate/GridTools.py:            cuda(bool, optional): Use CUDA or not.
deeprank/generate/GridTools.py:            gpu_block(tuple(int), optional): GPU block size to use.
deeprank/generate/GridTools.py:            cuda_func(None, optional): Name of the CUDA function to be
deeprank/generate/GridTools.py:                Must be present in kernel_cuda.c.
deeprank/generate/GridTools.py:            cuda_atomic(None, optional): Name of the CUDA function to be
deeprank/generate/GridTools.py:                Must be present in kernel_cuda.c.
deeprank/generate/GridTools.py:        # cuda support
deeprank/generate/GridTools.py:        self.cuda = cuda
deeprank/generate/GridTools.py:        if self.cuda:  # pragma: no cover
deeprank/generate/GridTools.py:            self.gpu_block = gpu_block
deeprank/generate/GridTools.py:            self.gpu_grid = [int(np.ceil(n / b))
deeprank/generate/GridTools.py:                             for b, n in zip(self.gpu_block, self.npts)]
deeprank/generate/GridTools.py:        # cuda
deeprank/generate/GridTools.py:        self.cuda_func = cuda_func
deeprank/generate/GridTools.py:        self.cuda_atomic = cuda_atomic
deeprank/generate/GridTools.py:        # prepare the cuda memory
deeprank/generate/GridTools.py:        if self.cuda:  # pragma: no cover
deeprank/generate/GridTools.py:            # try to import pycuda
deeprank/generate/GridTools.py:                from pycuda import driver, compiler, gpuarray, tools
deeprank/generate/GridTools.py:                import pycuda.autoinit
deeprank/generate/GridTools.py:                raise ImportError("Error when importing pyCuda in GridTools")
deeprank/generate/GridTools.py:            # book mem on the gpu
deeprank/generate/GridTools.py:            x_gpu = gpuarray.to_gpu(self.x.astype(np.float32))
deeprank/generate/GridTools.py:            y_gpu = gpuarray.to_gpu(self.y.astype(np.float32))
deeprank/generate/GridTools.py:            z_gpu = gpuarray.to_gpu(self.z.astype(np.float32))
deeprank/generate/GridTools.py:            grid_gpu = gpuarray.zeros(self.npts, np.float32)
deeprank/generate/GridTools.py:            # if we use CUDA
deeprank/generate/GridTools.py:            if self.cuda:  # pragma: no cover
deeprank/generate/GridTools.py:                grid_gpu *= 0
deeprank/generate/GridTools.py:                    self.cuda_atomic(
deeprank/generate/GridTools.py:                        vdw, x0, y0, z0, x_gpu, y_gpu, z_gpu, grid_gpu, block=tuple(
deeprank/generate/GridTools.py:                            self.gpu_block), grid=tuple(
deeprank/generate/GridTools.py:                            self.gpu_grid))
deeprank/generate/GridTools.py:                    atdensA = grid_gpu.get()
deeprank/generate/GridTools.py:                grid_gpu *= 0
deeprank/generate/GridTools.py:                    self.cuda_atomic(
deeprank/generate/GridTools.py:                        vdw, x0, y0, z0, x_gpu, y_gpu, z_gpu, grid_gpu, block=tuple(
deeprank/generate/GridTools.py:                            self.gpu_block), grid=tuple(
deeprank/generate/GridTools.py:                            self.gpu_grid))
deeprank/generate/GridTools.py:                    atdensB = grid_gpu.get()
deeprank/generate/GridTools.py:            # if we don't use CUDA
deeprank/generate/GridTools.py:        # prepare the cuda memory
deeprank/generate/GridTools.py:        if self.cuda:  # pragma: no cover
deeprank/generate/GridTools.py:            # try to import pycuda
deeprank/generate/GridTools.py:                from pycuda import driver, compiler, gpuarray, tools
deeprank/generate/GridTools.py:                import pycuda.autoinit
deeprank/generate/GridTools.py:                raise ImportError("Error when importing pyCuda in GridTools")
deeprank/generate/GridTools.py:            # book mem on the gpu
deeprank/generate/GridTools.py:            x_gpu = gpuarray.to_gpu(self.x.astype(np.float32))
deeprank/generate/GridTools.py:            y_gpu = gpuarray.to_gpu(self.y.astype(np.float32))
deeprank/generate/GridTools.py:            z_gpu = gpuarray.to_gpu(self.z.astype(np.float32))
deeprank/generate/GridTools.py:            grid_gpu = gpuarray.zeros(self.npts, np.float32)
deeprank/generate/GridTools.py:            if self.cuda:  # pragma: no cover
deeprank/generate/GridTools.py:                grid_gpu *= 0
deeprank/generate/GridTools.py:                if not self.cuda:
deeprank/generate/GridTools.py:                # try to use cuda to speed it up
deeprank/generate/GridTools.py:                        self.cuda_func(alpha,
deeprank/generate/GridTools.py:                                       x_gpu, y_gpu, z_gpu,
deeprank/generate/GridTools.py:                                       grid_gpu,
deeprank/generate/GridTools.py:                                       block=tuple(self.gpu_block),
deeprank/generate/GridTools.py:                                       grid=tuple(self.gpu_grid))
deeprank/generate/GridTools.py:                            'CUDA only possible for single-valued features')
deeprank/generate/GridTools.py:            if self.cuda:  # pragma: no cover
deeprank/generate/GridTools.py:                dict_data[fname] = grid_gpu.get()
deeprank/generate/DataGenerator.py:    from pycuda import driver, compiler, gpuarray, tools
deeprank/generate/DataGenerator.py:    import pycuda.autoinit
deeprank/generate/DataGenerator.py:                     cuda=False, gpu_block=None,
deeprank/generate/DataGenerator.py:                     cuda_kernel='kernel_map.c',
deeprank/generate/DataGenerator.py:                     cuda_func_name='gaussian',
deeprank/generate/DataGenerator.py:            cuda (bool, optional): Use CUDA
deeprank/generate/DataGenerator.py:            gpu_block (None, optional): GPU block size to be used
deeprank/generate/DataGenerator.py:            cuda_kernel (str, optional): filename containing CUDA kernel
deeprank/generate/DataGenerator.py:            cuda_func_name (str, optional): The name of the function in the kernel
deeprank/generate/DataGenerator.py:        # default CUDA
deeprank/generate/DataGenerator.py:        cuda_func = None
deeprank/generate/DataGenerator.py:        cuda_atomic = None
deeprank/generate/DataGenerator.py:        # disable CUDA when using MPI
deeprank/generate/DataGenerator.py:                if cuda:
deeprank/generate/DataGenerator.py:                        'CUDA mapping disabled when using MPI')
deeprank/generate/DataGenerator.py:                    cuda = False
deeprank/generate/DataGenerator.py:        # sanity check for cuda
deeprank/generate/DataGenerator.py:        if cuda and gpu_block is None:  # pragma: no cover
deeprank/generate/DataGenerator.py:                f'GPU block automatically set to 8 x 8 x 8. '
deeprank/generate/DataGenerator.py:                f'You can set block size with gpu_block=[n,m,k]')
deeprank/generate/DataGenerator.py:            gpu_block = [8, 8, 8]
deeprank/generate/DataGenerator.py:        # initialize cuda
deeprank/generate/DataGenerator.py:        if cuda:  # pragma: no cover
deeprank/generate/DataGenerator.py:            # compile cuda module
deeprank/generate/DataGenerator.py:            module = self._compile_cuda_kernel(cuda_kernel, npts, res)
deeprank/generate/DataGenerator.py:            # get the cuda function for the atomic/residue feature
deeprank/generate/DataGenerator.py:            cuda_func = self._get_cuda_function(
deeprank/generate/DataGenerator.py:                module, cuda_func_name)
deeprank/generate/DataGenerator.py:            # get the cuda function for the atomic densties
deeprank/generate/DataGenerator.py:            cuda_atomic_name = 'atomic_densities'
deeprank/generate/DataGenerator.py:            cuda_atomic = self._get_cuda_function(
deeprank/generate/DataGenerator.py:                module, cuda_atomic_name)
deeprank/generate/DataGenerator.py:                    cuda=cuda,
deeprank/generate/DataGenerator.py:                    gpu_block=gpu_block,
deeprank/generate/DataGenerator.py:                    cuda_func=cuda_func,
deeprank/generate/DataGenerator.py:                    cuda_atomic=cuda_atomic,
deeprank/generate/DataGenerator.py:    def _tune_cuda_kernel(self, grid_info, cuda_kernel='kernel_map.c', func='gaussian'):  # pragma: no cover
deeprank/generate/DataGenerator.py:        """Tune the CUDA kernel using the kernel tuner
deeprank/generate/DataGenerator.py:            cuda_kernel (str, optional): file containing the kernel
deeprank/generate/DataGenerator.py:        # arguments of the CUDA function
deeprank/generate/DataGenerator.py:            os.path.abspath(__file__)) + '/' + cuda_kernel
deeprank/generate/DataGenerator.py:    def _test_cuda(self, grid_info, gpu_block=8, cuda_kernel='kernel_map.c', func='gaussian'):  # pragma: no cover
deeprank/generate/DataGenerator.py:        """Test the CUDA kernel.
deeprank/generate/DataGenerator.py:            gpu_block (int, optional): GPU block size to be used
deeprank/generate/DataGenerator.py:            cuda_kernel (str, optional): File containing the kernel
deeprank/generate/DataGenerator.py:        # get the cuda function
deeprank/generate/DataGenerator.py:        module = self._compile_cuda_kernel(cuda_kernel, npts, res)
deeprank/generate/DataGenerator.py:        cuda_func = self._get_cuda_function(module, func)
deeprank/generate/DataGenerator.py:        # book memp on the gpu
deeprank/generate/DataGenerator.py:        x_gpu = gpuarray.to_gpu(x.astype(np.float32))
deeprank/generate/DataGenerator.py:        y_gpu = gpuarray.to_gpu(y.astype(np.float32))
deeprank/generate/DataGenerator.py:        z_gpu = gpuarray.to_gpu(z.astype(np.float32))
deeprank/generate/DataGenerator.py:        grid_gpu = gpuarray.zeros(
deeprank/generate/DataGenerator.py:        if not isinstance(gpu_block, list):
deeprank/generate/DataGenerator.py:            gpu_block = [gpu_block] * 3
deeprank/generate/DataGenerator.py:        gpu_grid = [int(np.ceil(n / b))
deeprank/generate/DataGenerator.py:                    for b, n in zip(gpu_block, grid_info['number_of_points'])]
deeprank/generate/DataGenerator.py:        print('GPU BLOCK:', gpu_block)
deeprank/generate/DataGenerator.py:        print('GPU GRID :', gpu_grid)
deeprank/generate/DataGenerator.py:            cuda_func(alpha, x0, y0, z0, x_gpu, y_gpu, z_gpu, grid_gpu,
deeprank/generate/DataGenerator.py:                      block=tuple(gpu_block), grid=tuple(gpu_grid))
deeprank/generate/DataGenerator.py:#       Routines needed to handle CUDA
deeprank/generate/DataGenerator.py:    def _compile_cuda_kernel(cuda_kernel, npts, res):  # pragma: no cover
deeprank/generate/DataGenerator.py:        """Compile the cuda kernel.
deeprank/generate/DataGenerator.py:            cuda_kernel (str): filename
deeprank/generate/DataGenerator.py:        # get the cuda kernel path
deeprank/generate/DataGenerator.py:            os.path.abspath(__file__)) + '/' + cuda_kernel
deeprank/generate/DataGenerator.py:    def _get_cuda_function(module, func_name):  # pragma: no cover
deeprank/generate/DataGenerator.py:            func: cuda function
deeprank/generate/DataGenerator.py:        cuda_func = module.get_function(func_name)
deeprank/generate/DataGenerator.py:        return cuda_func
deeprank/learn/metaqnn.py:    def train_model(self, cuda=False, ngpu=0):
deeprank/learn/metaqnn.py:        model = NeuralNet(self.data_set, cnn, plot=False, cuda=cuda, ngpu=ngpu)
deeprank/learn/NeuralNet.py:import torch.cuda
deeprank/learn/NeuralNet.py:                 cuda=False, ngpu=0,
deeprank/learn/NeuralNet.py:            cuda (bool): Use CUDA.
deeprank/learn/NeuralNet.py:            ngpu (int): number of GPU to be used.
deeprank/learn/NeuralNet.py:            if not cuda:
deeprank/learn/NeuralNet.py:        # CUDA
deeprank/learn/NeuralNet.py:        # CUDA required
deeprank/learn/NeuralNet.py:        self.cuda = cuda
deeprank/learn/NeuralNet.py:        self.ngpu = ngpu
deeprank/learn/NeuralNet.py:        # handles GPU/CUDA
deeprank/learn/NeuralNet.py:        if self.ngpu > 0:
deeprank/learn/NeuralNet.py:            self.cuda = True
deeprank/learn/NeuralNet.py:        if self.ngpu == 0 and self.cuda:
deeprank/learn/NeuralNet.py:            self.ngpu = 1
deeprank/learn/NeuralNet.py:        if cuda is True:
deeprank/learn/NeuralNet.py:            device = torch.device("cuda")  # PyTorch v0.4.0
deeprank/learn/NeuralNet.py:            if self.state['cuda']:
deeprank/learn/NeuralNet.py:        # multi-gpu
deeprank/learn/NeuralNet.py:        if self.ngpu > 1:
deeprank/learn/NeuralNet.py:            ids = [i for i in range(self.ngpu)]
deeprank/learn/NeuralNet.py:            self.net = nn.DataParallel(self.net, device_ids=ids).cuda()
deeprank/learn/NeuralNet.py:        # cuda compatible
deeprank/learn/NeuralNet.py:        elif self.cuda:
deeprank/learn/NeuralNet.py:            self.net = self.net.cuda()
deeprank/learn/NeuralNet.py:        logger.info(f'=\t CUDA     : {str(self.cuda)}')
deeprank/learn/NeuralNet.py:        if self.cuda:
deeprank/learn/NeuralNet.py:            logger.info(f'=\t nGPU     : {self.ngpu}')
deeprank/learn/NeuralNet.py:        # check if CUDA works
deeprank/learn/NeuralNet.py:        if self.cuda and not torch.cuda.is_available():
deeprank/learn/NeuralNet.py:                f' --> CUDA not deteceted: Make sure that CUDA is installed '
deeprank/learn/NeuralNet.py:                f'and that you are running on GPUs.\n'
deeprank/learn/NeuralNet.py:                f' --> To turn CUDA of set cuda=False in NeuralNet.\n'
deeprank/learn/NeuralNet.py:        if self.cuda:
deeprank/learn/NeuralNet.py:            logger.info(f': NGPU      : {self.ngpu}')
deeprank/learn/NeuralNet.py:                 'cuda': self.cuda
deeprank/learn/NeuralNet.py:        # pin memory for cuda
deeprank/learn/NeuralNet.py:        if self.cuda:
deeprank/learn/NeuralNet.py:            if self.cuda:
deeprank/learn/NeuralNet.py:        # if cuda is available
deeprank/learn/NeuralNet.py:        if self.cuda:
deeprank/learn/NeuralNet.py:            inputs = inputs.cuda(non_blocking=True)
deeprank/learn/NeuralNet.py:            targets = targets.cuda(non_blocking=True)
deeprank/utils/launch.py:            cuda=True,
deeprank/utils/launch.py:            gpu_block=[
deeprank/utils/launch.py:        help="GPU device to use",
deeprank/utils/launch.py:        # set the cuda device
deeprank/utils/launch.py:        #os.environ['CUDA_DEVICE'] = args.device
scripts/launch.py:        # database.map_features(grid_info,time=False,try_sparse=True,cuda=True,gpu_block=[8,8,8])
scripts/launch.py:        help="GPU device to use",
scripts/launch.py:        # set the cuda device
scripts/launch.py:        #os.environ['CUDA_DEVICE'] = args.device
example/train_001_smallNN_Kfold_balance_nofilt.py:os.environ["CUDA_VISIBLE_DEVICES"] = "1"
example/train_001_smallNN_Kfold_balance_nofilt.py:    class_weights = torch.FloatTensor(weights).cuda()
example/train_001_smallNN_Kfold_balance_nofilt.py:                    cuda=True,
example/train_001_smallNN_Kfold_balance_nofilt.py:                    ngpu=1,
example/learn.py:                  cuda=False, plot=True, outdir=out)
example/learn_batch_new.py:os.environ["CUDA_VISIBLE_DEVICES"] = "0"
example/learn_batch_new.py:                      cuda=True,
example/learn_batch_new.py:                      ngpu=1,
example/learn_batch.py:# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
example/learn_batch.py:                      cuda=False, plot=True, outdir=out)
example/learn_batch.py:    #model = NeuralNet(data_set, model3d.cnn,cuda=True,ngpu=1,plot=False, task='class')
example/kfold_train-balance_nofilt.slurm:#SBATCH -p gpu

```
