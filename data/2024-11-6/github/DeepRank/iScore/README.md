# https://github.com/DeepRank/iScore

```console
docs/gpu.rst:GPU Kernels
docs/conf.py:                'Bio','pycuda','svmutil']
iScore/graphrank/kernel.py:    import pycuda.autoinit
iScore/graphrank/kernel.py:    from pycuda import driver, compiler, gpuarray, tools
iScore/graphrank/kernel.py:    from pycuda.reduction import ReductionKernel
iScore/graphrank/kernel.py:    print('Warning : pycuda not found')
iScore/graphrank/kernel.py:    import skcuda.linalg as culinalg
iScore/graphrank/kernel.py:    import skcuda.misc as cumisc
iScore/graphrank/kernel.py:    print('Warning : scikit-cuda not found')
iScore/graphrank/kernel.py:                 gpu_block=(8,8,1),method='vect'):
iScore/graphrank/kernel.py:            gpu_block (tuple, optional): Block size to use for GPU
iScore/graphrank/kernel.py:        self.gpu_block = gpu_block
iScore/graphrank/kernel.py:        # the cuda kernel
iScore/graphrank/kernel.py:        self.kernel = os.path.dirname(os.path.abspath(__file__)) + '/cuda/cuda_kernel.cu'
iScore/graphrank/kernel.py:            raise FileNotFoundError('Cuda kernel %s not found' %self.kernel)
iScore/graphrank/kernel.py:    def run(self,lamb,walk,outfile='kernel.pkl',cuda=False,gpu_block=(8,8,1),check=None,
iScore/graphrank/kernel.py:            cuda (bool, optional): Use CUDA or not
iScore/graphrank/kernel.py:            gpu_block (tuple, optional): Size of the gpu block
iScore/graphrank/kernel.py:        if cuda and mpi_size > 1:
iScore/graphrank/kernel.py:            print('MPI and CUDA implementation not supported (yet)\n CUDA disabled.')
iScore/graphrank/kernel.py:            cuda = False
iScore/graphrank/kernel.py:        # do all the single-time cuda operations
iScore/graphrank/kernel.py:        if cuda:
iScore/graphrank/kernel.py:            self.weight_product = gpuarray.zeros(n_edges_prod, np.float32)
iScore/graphrank/kernel.py:            self.index_product = gpuarray.zeros((n_edges_prod,2), np.int32)
iScore/graphrank/kernel.py:                print('GPU - Mem  : %f' %(time()-t0))
iScore/graphrank/kernel.py:        K['param'] = {'lambda':lamb,'walk':walk,'cuda':cuda,'gpu_block':gpu_block}
iScore/graphrank/kernel.py:                if cuda:
iScore/graphrank/kernel.py:                    self.compute_kron_mat_cuda(G1,G2)
iScore/graphrank/kernel.py:                    self.compute_px_cuda(G1,G2)
iScore/graphrank/kernel.py:                    self.compute_W0_cuda(G1,G2)
iScore/graphrank/kernel.py:            print('GPU - Kron : %f' %(time()-t0))
iScore/graphrank/kernel.py:        d1_ = gpuarray.to_gpu(data1.astype(np.float32))
iScore/graphrank/kernel.py:        d2_ = gpuarray.to_gpu(data2.astype(np.float32))
iScore/graphrank/kernel.py:        mgpu  = -2*culinalg.dot(d1_,d2_,transa='N',transb='T')
iScore/graphrank/kernel.py:        vgpu = cumisc.sum(d1_**2,axis=1)[:,None]
iScore/graphrank/kernel.py:        cumisc.add_matvec(mgpu,vgpu,out=mgpu)
iScore/graphrank/kernel.py:        vgpu = cumisc.sum(d2_**2,axis=1)
iScore/graphrank/kernel.py:        cumisc.add_matvec(mgpu,vgpu,out=mgpu)
iScore/graphrank/kernel.py:        mcpu = mgpu.get()
iScore/graphrank/kernel.py:    #  CUDA Routines
iScore/graphrank/kernel.py:        """Compile the file containing the CUDA kernels."""
iScore/graphrank/kernel.py:            print('GPU - Kern : %f' %(time()-t0))
iScore/graphrank/kernel.py:    def compute_kron_mat_cuda(self,g1,g2,kernel_name='create_kron_mat',gpu_block=None): # pragma: no cover
iScore/graphrank/kernel.py:            gpu_block (None, optional): Size of the GPU block
iScore/graphrank/kernel.py:        # get the gpu block size if specified
iScore/graphrank/kernel.py:        if gpu_block is not None:
iScore/graphrank/kernel.py:            block = gpu_block
iScore/graphrank/kernel.py:            block = self.gpu_block
iScore/graphrank/kernel.py:        create_kron_mat_gpu = self.mod.get_function(kernel_name)
iScore/graphrank/kernel.py:        # put the raw pssm on the GPU
iScore/graphrank/kernel.py:        pssm1 = gpuarray.to_gpu(np.array(g1.edges_pssm).astype(np.float32))
iScore/graphrank/kernel.py:        pssm2 = gpuarray.to_gpu(np.array(g2.edges_pssm).astype(np.float32))
iScore/graphrank/kernel.py:        # we have to put the index on the gpu as well
iScore/graphrank/kernel.py:        ind1 = gpuarray.to_gpu(np.array(g1.edges_index).astype(np.int32))
iScore/graphrank/kernel.py:        ind2 = gpuarray.to_gpu(np.array(g2.edges_index).astype(np.int32))
iScore/graphrank/kernel.py:        # create the gpu arrays only if we have to
iScore/graphrank/kernel.py:            self.weight_product = gpuarray.zeros(n_edges_prod, np.float32)
iScore/graphrank/kernel.py:            self.index_product = gpuarray.zeros((n_edges_prod,2), np.int32)
iScore/graphrank/kernel.py:            print('GPU - Mem  : %f \t (block size:%dx%d)' %(time()-t0,block[0],block[1]))
iScore/graphrank/kernel.py:        create_kron_mat_gpu (ind1,ind2,
iScore/graphrank/kernel.py:            print('GPU - Kron : %f \t (block size:%dx%d)' %(time()-t0,block[0],block[1]))
iScore/graphrank/kernel.py:    def compute_px_cuda(self,g1,g2,gpu_block=None): # pragma: no cover
iScore/graphrank/kernel.py:            gpu_block (None, optional): Size of the GPU block
iScore/graphrank/kernel.py:        info1 = gpuarray.to_gpu(np.array(g1.nodes_info_data).astype(np.float32))
iScore/graphrank/kernel.py:        info2 = gpuarray.to_gpu(np.array(g2.nodes_info_data).astype(np.float32))
iScore/graphrank/kernel.py:        pvect = gpuarray.zeros(n_nodes_prod,np.float32)
iScore/graphrank/kernel.py:        if gpu_block is not None:
iScore/graphrank/kernel.py:            block = gpu_block
iScore/graphrank/kernel.py:            block = self.gpu_block
iScore/graphrank/kernel.py:            print('GPU - Px   : %f \t (block size:%dx%d)' %(time()-t0,block[0],block[1]))
iScore/graphrank/kernel.py:    def compute_W0_cuda(self,g1,g2,gpu_block=None): # pragma: no cover
iScore/graphrank/kernel.py:            gpu_block (None, optional): Size of the GPU block
iScore/graphrank/kernel.py:        pssm1 = gpuarray.to_gpu(np.array(g1.nodes_pssm_data).astype(np.float32))
iScore/graphrank/kernel.py:        pssm2 = gpuarray.to_gpu(np.array(g2.nodes_pssm_data).astype(np.float32))
iScore/graphrank/kernel.py:        w0 = gpuarray.zeros(n_nodes_prod,np.float32)
iScore/graphrank/kernel.py:        if gpu_block is not None:
iScore/graphrank/kernel.py:            block = gpu_block
iScore/graphrank/kernel.py:            block = self.gpu_block
iScore/graphrank/kernel.py:            print('GPU - W0   : %f \t (block size:%dx%d)' %(time()-t0,block[0],block[1]))
iScore/graphrank/kernel.py:                  tune_kernel=False,func='all',cuda=False, gpu_block=[8,8,1]):
iScore/graphrank/kernel.py:                 gpu_block=tuple(gpu_block),method=method)
iScore/graphrank/kernel.py:               cuda=cuda,
iScore/graphrank/kernel.py:               gpu_block=tuple(gpu_block),
iScore/graphrank/cuda/main.cu:#include <cuda.h>
iScore/graphrank/cuda/main.cu:#include <cuda_runtime_api.h>
iScore/graphrank/cuda/main.cu:    const cudaError_t error = call;\
iScore/graphrank/cuda/main.cu:    if (error != cudaSuccess)\
iScore/graphrank/cuda/main.cu:        printf("code %d, reason %s", error, cudaGetErrorString(error));\
iScore/graphrank/cuda/main.cu:    struct cudaDeviceProp deviceProp;
iScore/graphrank/cuda/main.cu:    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
iScore/graphrank/cuda/main.cu:    CHECK(cudaSetDevice(dev));
iScore/graphrank/cuda/main.cu:    cudaMalloc((void**)&d_edge_index_1,nBytes_index1);
iScore/graphrank/cuda/main.cu:    cudaMemcpy(d_edge_index_1, edge_index_1.data(), nBytes_index1,cudaMemcpyHostToDevice);
iScore/graphrank/cuda/main.cu:    cudaMalloc((void**)&d_edge_pssm_1,nBytes_pssm1);
iScore/graphrank/cuda/main.cu:    cudaMemcpy(d_edge_pssm_1, edge_pssm_1.data(),nBytes_pssm1,cudaMemcpyHostToDevice);
iScore/graphrank/cuda/main.cu:    cudaMalloc((void**)&d_edge_index_2,nBytes_index2);
iScore/graphrank/cuda/main.cu:    cudaMemcpy(d_edge_index_2, edge_index_2.data(),nBytes_index2,cudaMemcpyHostToDevice);
iScore/graphrank/cuda/main.cu:    cudaMalloc((void**)&d_edge_pssm_2,nBytes_pssm2);
iScore/graphrank/cuda/main.cu:    cudaMemcpy(d_edge_pssm_2, edge_pssm_2.data(), nBytes_pssm2, cudaMemcpyHostToDevice);
iScore/graphrank/cuda/main.cu:    cudaMalloc((void**)&d_edge_index_prod,nBytes_index_prod);
iScore/graphrank/cuda/main.cu:    cudaMalloc((void**)&d_edge_weight_prod,nBytes_weight_prod);
iScore/graphrank/cuda/main.cu:    cudaDeviceSynchronize();
iScore/graphrank/cuda/main.cu:    cudaMalloc((void**)&d_node_data_1,nBytes_node1);
iScore/graphrank/cuda/main.cu:    cudaMemcpy(d_node_data_1, node_data_1.data(), nBytes_node1, cudaMemcpyHostToDevice);
iScore/graphrank/cuda/main.cu:    cudaMalloc((void**)&d_node_data_2,nBytes_node2);
iScore/graphrank/cuda/main.cu:    cudaMemcpy(d_node_data_2, node_data_2.data(), nBytes_node2, cudaMemcpyHostToDevice);
iScore/graphrank/cuda/main.cu:    cudaMalloc((void**)&d_pvect,nBytes_pvect);
iScore/graphrank/cuda/main.cu:    cudaDeviceSynchronize();
iScore/graphrank/cuda/main.cu:    cudaFree(d_edge_index_1);
iScore/graphrank/cuda/main.cu:    cudaFree(d_edge_pssm_1);
iScore/graphrank/cuda/main.cu:    cudaFree(d_edge_index_2);
iScore/graphrank/cuda/main.cu:    cudaFree(d_edge_pssm_2);
iScore/graphrank/cuda/main.cu:    cudaFree(d_edge_index_prod);
iScore/graphrank/cuda/main.cu:    cudaFree(d_edge_weight_prod);
iScore/graphrank/cuda/main.cu:    cudaFree(d_node_data_1);
iScore/graphrank/cuda/main.cu:    cudaFree(d_node_data_2);
iScore/graphrank/cuda/main.cu:    cudaFree(d_pvect);
iScore/graphrank/cuda/main.cu:    cudaDeviceReset();
iScore/graphrank/cuda/Makefile:CUDA_ROOT = /usr/local/cuda-9.0
iScore/graphrank/cuda/Makefile:CUDA_INC = -I${CUDA_ROOT}/include/
iScore/graphrank/cuda/Makefile:CUDA_LIB = -L${CUDA_ROOT}/lib64/ 
iScore/graphrank/cuda/Makefile:test : main.o cuda_kernel.o
iScore/graphrank/cuda/Makefile:	nvcc ${CXX_FLAGS} ${CUDA_INC} main.o cuda_kernel.o -o test ${CUDA_LIB} -lcuda -lcudart
iScore/graphrank/cuda/Makefile:cuda_kernel.o: cuda_kernel.cu
iScore/graphrank/cuda/Makefile:	nvcc ${CXX_FLAGS} -c  ${CUDA_INC} $< -o $@ ${CUDA_LIB} -lcuda -lcudart
iScore/graphrank/cuda/Makefile:	rm test cuda_kernel.o main.o 
iScore/graphrank/cuda/cuda_kernel.cu:#include <cuda_runtime.h>
MANIFEST.in:include iScore/graphrank/cuda/*
bin/iScore.kernel:parser.add_argument('--tune_kernel',action='store_true',help='Only tune the CUDA kernel')
bin/iScore.kernel:# cuda parameters
bin/iScore.kernel:parser.add_argument('--cuda',action='store_true', help='Use CUDA kernel')
bin/iScore.kernel:parser.add_argument('--gpu_block',nargs='+',default=[8,8,1],type=int,help='number of gpu block to use. Default: 8 8 1')
bin/iScore.kernel:	              tune_kernel=args.tune_kernel,func=args.func,cuda=args.cuda, gpu_block=args.gpu_block)

```
