# https://github.com/tum-pbs/PhiML

```console
setup.py:def check_tf_cuda_compatibility():
setup.py:    build = tensorflow.sysconfig.get_build_info()  # is_rocm_build, cuda_compute_capabilities
setup.py:    is_cuda_build = build['is_cuda_build']
setup.py:    if not is_cuda_build:
setup.py:        raise AssertionError("Your TensorFlow build does not support CUDA.")
setup.py:        cuda_version = build['cuda_version']
setup.py:        print(f"TensorFlow was compiled against CUDA {cuda_version} and cuDNN {cudnn_version}.")
setup.py:def compile_cuda(file_names, nvcc, source_dir, target_dir, logfile):
setup.py:            '-D GOOGLE_CUDA=1',
setup.py:def compile_gcc(file_names, gcc, source_dir, target_dir, cuda_lib, logfile):
setup.py:    link_cuda_lib = '-L' + cuda_lib
setup.py:                '-lcudart',
setup.py:                link_cuda_lib
setup.py:class CudaCommand(distutils.cmd.Command):
setup.py:    description = 'Compile CUDA sources'
setup.py:        ('nvcc=', None, 'Path to the Nvidia nvcc compiler.'),
setup.py:        ('cuda-lib=', None, 'Path to the CUDA libraries.'),
setup.py:        tf_gcc = check_tf_cuda_compatibility()
setup.py:        self.nvcc = '/usr/local/cuda/bin/nvcc' if isfile('/usr/local/cuda/bin/nvcc') else 'nvcc'
setup.py:        self.cuda_lib = '/usr/local/cuda/lib64/'
setup.py:        src_path = abspath('./phiml/tf/cuda/src')
setup.py:        build_path = abspath('./phiml/tf/cuda/build')
setup.py:        logfile_path = abspath('./phiml/tf/cuda/log.txt')
setup.py:        print("CUDA lib:\t" + self.cuda_lib)
setup.py:        print('Compiling CUDA code...')
setup.py:                compile_cuda('resample', self.nvcc, src_path, build_path, logfile=logfile)
setup.py:                compile_gcc('resample', self.gcc, src_path, build_path, self.cuda_lib, logfile=logfile)
setup.py:                compile_cuda('resample_gradient', self.nvcc, src_path, build_path, logfile=logfile)
setup.py:                compile_gcc('resample_gradient', self.gcc, src_path, build_path, self.cuda_lib, logfile=logfile)
setup.py:                # compile_cuda('bicgstab_ilu_linear_solve_op', self.nvcc, src_path, build_path, logfile=logfile)
setup.py:                # compile_gcc('bicgstab_ilu_linear_solve_op', self.gcc, src_path, build_path, self.cuda_lib, logfile=logfile)
setup.py:        'tf_cuda': CudaCommand,
phiml/math/_ops.py:            Either `ComputeDevice` or category `str`, such as `'CPU'` or `'GPU'`.
phiml/math/_ops.py:            Whether to use the DLPack library to convert from one GPU-enabled backend to another.
phiml/math/_ops.py:            * `'sparse'`: GPU-supported hash grid implementation with fully sparse connectivity.
phiml/backend/torch/_torch_backend.py:        for index in range(torch.cuda.device_count()):
phiml/backend/torch/_torch_backend.py:            properties = torch.cuda.get_device_properties(index)
phiml/backend/torch/_torch_backend.py:            devices.append(ComputeDevice(self, properties.name, 'GPU', properties.total_memory, properties.multi_processor_count, f"compute capability {properties.major}.{properties.minor}", f'cuda:{index}'))
phiml/backend/torch/_torch_backend.py:        torch.cuda.manual_seed(seed)
phiml/backend/jax/_jax_backend.py:        for device_type in ['cpu', 'gpu', 'tpu']:
phiml/backend/_backend.py:        assert device_type in ('CPU', 'GPU', 'TPU')
phiml/backend/_backend.py:        """ Type of device such as `'CPU'`, `'GPU'` or `'TPU'`. """
phiml/backend/_backend.py:        """ Number of CPU cores or GPU multiprocessors. -1 for n/a. """
phiml/backend/_backend.py:        * PyTorch: [`torch.cuda.get_device_properties`](https://pytorch.org/docs/stable/cuda.html#torch.cuda.get_device_properties)
phiml/backend/_backend.py:            device_type: (optional) Return only devices of this type, e.g. `'GPU'` or `'CPU'`. See `ComputeDevice.device_type`.
phiml/backend/_backend.py:            assert device_type in ('CPU', 'GPU', 'TPU'), "Device"
phiml/backend/_backend.py:            device: `ComputeDevice` or device type as `str`, such as `'CPU'` or `'GPU'`.
phiml/backend/tensorflow/_tf_backend.py:from ._tf_cuda_resample import resample_cuda, use_cuda
phiml/backend/tensorflow/_tf_backend.py:        if use_cuda(grid):
phiml/backend/tensorflow/_tf_backend.py:            return resample_cuda(grid, coordinates, extrapolation)
phiml/backend/tensorflow/_tf_cuda_resample.py:    resample_op_path = os.path.join(current_dir, "cuda/build/resample.so")
phiml/backend/tensorflow/_tf_cuda_resample.py:        current_dir, "cuda/build/resample_gradient.so"
phiml/backend/tensorflow/_tf_cuda_resample.py:        'CUDA binaries not found at %s. Run "python setup.py tf_cuda" to compile them'
phiml/backend/tensorflow/_tf_cuda_resample.py:        'CUDA binaries not found at %s. Run "python setup.py tf_cuda" to '
phiml/backend/tensorflow/_tf_cuda_resample.py:    # e.g.: tensorflow.python.framework.errors_impl.NotFoundError: libcudart.so.10.0: cannot open shared object file: No such file or directory
phiml/backend/tensorflow/_tf_cuda_resample.py:def use_cuda(inputs):
phiml/backend/tensorflow/_tf_cuda_resample.py:    if not tf.test.is_gpu_available(True, (3, 0)):
phiml/backend/tensorflow/_tf_cuda_resample.py:def resample_cuda(inputs, sample_coords, extrapolation):
phiml/backend/tensorflow/_compile_cuda.py:def compile_cuda_ops(gcc: str = None,
phiml/backend/tensorflow/_compile_cuda.py:                     cuda_lib: str = None):
phiml/backend/tensorflow/_compile_cuda.py:    tf_gcc = check_tf_cuda_compatibility()
phiml/backend/tensorflow/_compile_cuda.py:        nvcc = '/usr/local/cuda/bin/nvcc' if isfile('/usr/local/cuda/bin/nvcc') else 'nvcc'
phiml/backend/tensorflow/_compile_cuda.py:    if cuda_lib is None:
phiml/backend/tensorflow/_compile_cuda.py:        cuda_lib = '/usr/local/cuda/lib64/'
phiml/backend/tensorflow/_compile_cuda.py:    src_path = join(uml_tf_path, 'cuda', 'src')
phiml/backend/tensorflow/_compile_cuda.py:    build_path = join(uml_tf_path, 'cuda', 'build')
phiml/backend/tensorflow/_compile_cuda.py:    logfile_path = join(uml_tf_path, 'cuda', 'log.txt')
phiml/backend/tensorflow/_compile_cuda.py:    print("CUDA lib:\t" + cuda_lib)
phiml/backend/tensorflow/_compile_cuda.py:    print('Compiling CUDA code...')
phiml/backend/tensorflow/_compile_cuda.py:            compile_cuda('resample', nvcc, src_path, build_path, logfile=logfile)
phiml/backend/tensorflow/_compile_cuda.py:            compile_gcc('resample', gcc, src_path, build_path, cuda_lib, logfile=logfile)
phiml/backend/tensorflow/_compile_cuda.py:            compile_cuda('resample_gradient', nvcc, src_path, build_path, logfile=logfile)
phiml/backend/tensorflow/_compile_cuda.py:            compile_gcc('resample_gradient', gcc, src_path, build_path, cuda_lib, logfile=logfile)
phiml/backend/tensorflow/_compile_cuda.py:            # compile_cuda('bicgstab_ilu_linear_solve_op', self.nvcc, src_path, build_path, logfile=logfile)
phiml/backend/tensorflow/_compile_cuda.py:            # compile_gcc('bicgstab_ilu_linear_solve_op', self.gcc, src_path, build_path, self.cuda_lib, logfile=logfile)
phiml/backend/tensorflow/_compile_cuda.py:def check_tf_cuda_compatibility():
phiml/backend/tensorflow/_compile_cuda.py:    build = tensorflow.sysconfig.get_build_info()  # is_rocm_build, cuda_compute_capabilities
phiml/backend/tensorflow/_compile_cuda.py:    is_cuda_build = build['is_cuda_build']
phiml/backend/tensorflow/_compile_cuda.py:    if not is_cuda_build:
phiml/backend/tensorflow/_compile_cuda.py:        raise AssertionError("Your TensorFlow build does not support CUDA.")
phiml/backend/tensorflow/_compile_cuda.py:        cuda_version = build['cuda_version']
phiml/backend/tensorflow/_compile_cuda.py:        print(f"TensorFlow was compiled against CUDA {cuda_version} and cuDNN {cudnn_version}.")
phiml/backend/tensorflow/_compile_cuda.py:def compile_cuda(file_names, nvcc, source_dir, target_dir, logfile):
phiml/backend/tensorflow/_compile_cuda.py:            '-D GOOGLE_CUDA=1',
phiml/backend/tensorflow/_compile_cuda.py:def compile_gcc(file_names, gcc, source_dir, target_dir, cuda_lib, logfile):
phiml/backend/tensorflow/_compile_cuda.py:    link_cuda_lib = '-L' + cuda_lib
phiml/backend/tensorflow/_compile_cuda.py:                '-lcudart',
phiml/backend/tensorflow/_compile_cuda.py:                link_cuda_lib
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:#include <cuda_runtime.h>
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:    if (err == cudaSuccess) return;
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:                          int* iterations_gpu) 
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, calcZ_v4, 0, 0);
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, initVariablesWithGuess, 0, 0);
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:    CUDA_CHECK_RETURN(cudaMemcpy(threshold_reached_cpu, threshold_reached, sizeof(bool) * batch_size, cudaMemcpyDeviceToHost));
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:            CUDA_CHECK_RETURN(cudaMemcpy(threshold_reached_cpu, threshold_reached, sizeof(bool) * batch_size, cudaMemcpyDeviceToHost));
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:            CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:            CUDA_CHECK_RETURN(cudaMemset(threshold_reached, 1, sizeof(bool) * batch_size));
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:    CUDA_CHECK_RETURN(cudaMemcpy(iterations_gpu, &iterations, sizeof(int), cudaMemcpyHostToDevice));
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cu.cc:    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/resample_gradient.cc:// Register the GPU kernels.
phiml/backend/tensorflow/cuda/src/resample_gradient.cc:REGISTER_KERNEL_BUILDER(Name("ResampleGradient").Device(DEVICE_GPU), ResampleGradientOp);
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:#include <cuda_runtime_api.h>
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  if (err == cudaSuccess) return;
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:inline void cdpAssert(cudaError_t code, const char *file, int line, bool abort=true)
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:   if (code != cudaSuccess)
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:      printf("GPU kernel assert: %s %p %d\n", cudaGetErrorString(code), file, line);
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:      printf("GPU kernel assert: %i %p %d\n", code, file, line);
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:      printf("GPU kernel assert: %i %p %d\n", code, file, line);
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:      printf("GPU kernel assert: %i %p %d\n", code, file, line);
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:#define DTYPE CUDA_R_32F
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:    cudaOccupancyMaxPotentialBlockSize(  &minGridSize, &blockSize, transpose_csr_into_cpy, 0, 0 );
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:    cdpErrchk(cudaMalloc((void**) &csr_values_transposed, nnz_a*sizeof(dtype)));
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  cdpErrchk(cudaMalloc((void**)&pBuffer, pBufferSize));
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:cdpErrchk(cudaMalloc((void**) &s,     matrix_shape*sizeof(dtype)));
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:cdpErrchk(cudaMalloc((void**) &s_hat, matrix_shape*sizeof(dtype)));
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:cdpErrchk(cudaMalloc((void**) &p,     matrix_shape*sizeof(dtype)));
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:cdpErrchk(cudaMalloc((void**) &p_hat, matrix_shape*sizeof(dtype)));
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:cdpErrchk(cudaMalloc((void**) &r,     matrix_shape*sizeof(dtype)));
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:cdpErrchk(cudaMalloc((void**) &rh,    matrix_shape*sizeof(dtype)));
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:cdpErrchk(cudaMalloc((void**) &v,     matrix_shape*sizeof(dtype)));
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:cdpErrchk(cudaMalloc((void**) &t,     matrix_shape*sizeof(dtype)));
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:cdpErrchk(cudaMalloc((void**) &z,     matrix_shape*sizeof(dtype)));*/
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:cdpErrchk(cudaMalloc((void**) &helpvec,     matrix_shape*sizeof(dtype)));
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:cdpErrchk(cudaMalloc((void**) &helpvec2,     matrix_shape*sizeof(dtype)));
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc: CUDA_CHECK_RETURN(cudaDeviceSynchronize());*/
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:cdpErrchk(cudaMalloc((void**)&bicgBuffer, bicgBufferSize));
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  CUDA_CHECK_RETURN(cudaDeviceSynchronize());*/
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:    cudaFree(csr_values_transposed);
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  cudaFree(pBuffer);
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  cudaFree(bicgBuffer);
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  /*cudaFree(s);
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  cudaFree(s_hat);
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  cudaFree(p);
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  cudaFree(p_hat);
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  cudaFree(r);
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  cudaFree(rh);
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  cudaFree(v);
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  cudaFree(t);
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  cudaFree(z);*/
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  //cudaFree(csr_values);
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cu.cc:  //cudaFree(csr_values);
phiml/backend/tensorflow/cuda/src/laplace_op.cc:REGISTER_KERNEL_BUILDER(Name("LaplaceMatrix").Device(DEVICE_GPU), LaplaceMatrixOp);
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cc:            int* iterations_gpu);
phiml/backend/tensorflow/cuda/src/pressure_solve_op.cc:REGISTER_KERNEL_BUILDER(Name("PressureSolve").Device(DEVICE_GPU), PressureSolveOp);
phiml/backend/tensorflow/cuda/src/laplace_op.cu.cc:#include <cuda_runtime.h>
phiml/backend/tensorflow/cuda/src/laplace_op.cu.cc:static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
phiml/backend/tensorflow/cuda/src/laplace_op.cu.cc:  if (err == cudaSuccess) return;
phiml/backend/tensorflow/cuda/src/laplace_op.cu.cc:  std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
phiml/backend/tensorflow/cuda/src/laplace_op.cu.cc:#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
phiml/backend/tensorflow/cuda/src/laplace_op.cu.cc:    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
phiml/backend/tensorflow/cuda/src/laplace_op.cu.cc:    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/laplace_op.cu.cc:    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
phiml/backend/tensorflow/cuda/src/laplace_op.cu.cc:    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cc:    // SPACE FOR COPY OF X (actual copy done in cuda) - TF PYTHON REUSES  SOME TENSORS (WIERD)
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cc:    //std::cout << "launch cuda";
phiml/backend/tensorflow/cuda/src/bicgstab_ilu_linear_solve_op.cc:REGISTER_KERNEL_BUILDER(Name("BicgstabIluLinearSolve").Device(DEVICE_GPU), BicgstabIluLinearSolveOp);
phiml/backend/tensorflow/cuda/src/helpers.h:#ifdef __CUDACC__
phiml/backend/tensorflow/cuda/src/helpers.h:#define CUDA_HOSTDEV __host__ __device__
phiml/backend/tensorflow/cuda/src/helpers.h:#define CUDA_HOSTDEV
phiml/backend/tensorflow/cuda/src/helpers.h:CUDA_HOSTDEV
phiml/backend/tensorflow/cuda/src/helpers.h:CUDA_HOSTDEV
phiml/backend/tensorflow/cuda/src/helpers.h:CUDA_HOSTDEV
phiml/backend/tensorflow/cuda/src/helpers.h:CUDA_HOSTDEV
phiml/backend/tensorflow/cuda/src/helpers.h:CUDA_HOSTDEV
phiml/backend/tensorflow/cuda/src/helpers.h:CUDA_HOSTDEV
phiml/backend/tensorflow/cuda/src/helpers.h:#ifdef __CUDACC__
phiml/backend/tensorflow/cuda/src/helpers.h:#if __CUDA_ARCH__ >= 350
phiml/backend/tensorflow/cuda/src/helpers.h:// https://stackoverflow.com/questions/13245258/handle-error-not-found-error-in-cuda/13245319
phiml/backend/tensorflow/cuda/src/helpers.h:static void HandleError( cudaError_t err,
phiml/backend/tensorflow/cuda/src/helpers.h:    if (err != cudaSuccess) {
phiml/backend/tensorflow/cuda/src/helpers.h:        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
phiml/backend/tensorflow/cuda/src/helpers.h:cudaArray* createArray(const unsigned int xSize, const unsigned int ySize, const unsigned int zSize, const unsigned int components) {
phiml/backend/tensorflow/cuda/src/helpers.h:	cudaArray* cuArray;
phiml/backend/tensorflow/cuda/src/helpers.h:	// Create cuda extent
phiml/backend/tensorflow/cuda/src/helpers.h:	cudaExtent extent = make_cudaExtent(xSize, ySize, zSize);
phiml/backend/tensorflow/cuda/src/helpers.h:	cudaChannelFormatDesc channelDesc;
phiml/backend/tensorflow/cuda/src/helpers.h:		channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
phiml/backend/tensorflow/cuda/src/helpers.h:		channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
phiml/backend/tensorflow/cuda/src/helpers.h:		channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
phiml/backend/tensorflow/cuda/src/helpers.h:	cudaMalloc3DArray(&cuArray, &channelDesc, extent, cudaArraySurfaceLoadStore);
phiml/backend/tensorflow/cuda/src/helpers.h:cudaMemcpy3DParms createCopyParams(const T* __restrict__ data, cudaArray* cuArray, const unsigned int xSize, const unsigned int ySize, const unsigned int zSize, const unsigned int components) {
phiml/backend/tensorflow/cuda/src/helpers.h:	cudaMemcpy3DParms copyParams = {0};
phiml/backend/tensorflow/cuda/src/helpers.h:	cudaExtent extent = make_cudaExtent(xSize, ySize, zSize);
phiml/backend/tensorflow/cuda/src/helpers.h:	copyParams.srcPtr = make_cudaPitchedPtr((void*) data, xSize * components * sizeof(T), xSize, ySize);
phiml/backend/tensorflow/cuda/src/helpers.h:	copyParams.kind = cudaMemcpyDeviceToDevice;
phiml/backend/tensorflow/cuda/src/helpers.h:cudaResourceDesc createResDesc(cudaArray* cuArray) {
phiml/backend/tensorflow/cuda/src/helpers.h:	struct cudaResourceDesc resDesc;
phiml/backend/tensorflow/cuda/src/helpers.h:	resDesc.resType = cudaResourceTypeArray;
phiml/backend/tensorflow/cuda/src/helpers.h:cudaTextureObject_t createTextureObject(cudaArray* cuArray) {
phiml/backend/tensorflow/cuda/src/helpers.h:	cudaTextureObject_t dataTexture = 0;
phiml/backend/tensorflow/cuda/src/helpers.h:	cudaResourceDesc resDesc = createResDesc(cuArray);
phiml/backend/tensorflow/cuda/src/helpers.h:	struct cudaTextureDesc texDesc;
phiml/backend/tensorflow/cuda/src/helpers.h:	texDesc.addressMode[0] = cudaAddressModeClamp;
phiml/backend/tensorflow/cuda/src/helpers.h:	texDesc.addressMode[1] = cudaAddressModeClamp;
phiml/backend/tensorflow/cuda/src/helpers.h:	texDesc.addressMode[2] = cudaAddressModeClamp;
phiml/backend/tensorflow/cuda/src/helpers.h:	texDesc.filterMode = cudaFilterModePoint;
phiml/backend/tensorflow/cuda/src/helpers.h:	//texDesc.filterMode = cudaFilterModeLinear;
phiml/backend/tensorflow/cuda/src/helpers.h:	texDesc.readMode = cudaReadModeElementType;
phiml/backend/tensorflow/cuda/src/helpers.h:	cudaCreateTextureObject(&dataTexture, &resDesc, &texDesc, NULL);
phiml/backend/tensorflow/cuda/src/helpers.h:cudaSurfaceObject_t createSurfaceObject(cudaArray* cuArray){
phiml/backend/tensorflow/cuda/src/helpers.h:	cudaSurfaceObject_t surfaceObject = 0;
phiml/backend/tensorflow/cuda/src/helpers.h:	cudaResourceDesc resDesc = createResDesc(cuArray);
phiml/backend/tensorflow/cuda/src/helpers.h:	cudaCreateSurfaceObject(&surfaceObject, &resDesc);
phiml/backend/tensorflow/cuda/src/helpers.h:void CopyKernel (const float* data, cudaSurfaceObject_t surfaceObject, int dims, const unsigned int xSize, const unsigned int ySize, const unsigned int zSize, const unsigned int batch) {
phiml/backend/tensorflow/cuda/src/helpers.h:void copyDataToArray(const T* __restrict__ data, cudaArray* cuArray, cudaSurfaceObject_t surfaceObject, cudaMemcpy3DParms copyParams, const int dims, const unsigned int xSize, const unsigned int ySize, const unsigned int zSize, const unsigned int batch, const unsigned int components) {
phiml/backend/tensorflow/cuda/src/helpers.h:		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, CopyKernel, 0, 0);
phiml/backend/tensorflow/cuda/src/helpers.h:		HANDLE_ERROR(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/helpers.h:		cudaMemcpyToArray(cuArray, 0, 0, data + batch * xSize * ySize * zSize * components, xSize * ySize * zSize * components * sizeof(T), cudaMemcpyDeviceToDevice);
phiml/backend/tensorflow/cuda/src/helpers.h:		copyParams.srcPtr = make_cudaPitchedPtr((void*) (data + batch * xSize * ySize * zSize * components), xSize * components * sizeof(T), xSize, ySize);
phiml/backend/tensorflow/cuda/src/helpers.h:		cudaMemcpy3D(&copyParams);
phiml/backend/tensorflow/cuda/src/helpers.h:inline T tex1DHelper(cudaTextureObject_t texObj, float x) {
phiml/backend/tensorflow/cuda/src/helpers.h:inline float3 tex1DHelper(cudaTextureObject_t texObj, float x) {
phiml/backend/tensorflow/cuda/src/helpers.h:inline T tex1DHelper(cudaTextureObject_t texObj, float x, const Boundary* boundaries, const unsigned int xSize) {
phiml/backend/tensorflow/cuda/src/helpers.h:inline T tex2DHelper(cudaTextureObject_t texObj, float x, float y) {
phiml/backend/tensorflow/cuda/src/helpers.h:inline float3 tex2DHelper(cudaTextureObject_t texObj, float x, float y) {
phiml/backend/tensorflow/cuda/src/helpers.h:inline T tex2DHelper(cudaTextureObject_t texObj, float x, float y, const Boundary* boundaries, const unsigned int xSize, const unsigned int ySize) {
phiml/backend/tensorflow/cuda/src/helpers.h:inline T tex3DHelper(cudaTextureObject_t texObj, float x, float y, float z) {
phiml/backend/tensorflow/cuda/src/helpers.h:inline float3 tex3DHelper(cudaTextureObject_t texObj, float x, float y, float z) {
phiml/backend/tensorflow/cuda/src/helpers.h:inline T tex3DHelper(cudaTextureObject_t texObj, float x, float y, float z, const Boundary* boundaries, const unsigned int xSize, const unsigned int ySize, const unsigned int zSize) {
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:// Naive CUDA kernel
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:void ResampleGradientCudaKernel(
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:void ResampleGradient1DCudaKernel(
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:	cudaTextureObject_t dataTexture,
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:void ResampleGradient2DCudaKernel (
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:	cudaTextureObject_t dataTexture,
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:void ResampleGradient3DCudaKernel (
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:	cudaTextureObject_t dataTexture,
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:	cudaTextureObject_t dataTexture,
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient1DCudaKernel<float, float>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:			ResampleGradient1DCudaKernel<float, float><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient1DCudaKernel<float, float2>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:			ResampleGradient1DCudaKernel<float, float2><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient1DCudaKernel<float, float3>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:			ResampleGradient1DCudaKernel<float, float3><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient1DCudaKernel<float, float4>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:			ResampleGradient1DCudaKernel<float, float4><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient2DCudaKernel<float, float>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:			ResampleGradient2DCudaKernel<float, float><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, ySize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient2DCudaKernel<float, float2>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:			ResampleGradient2DCudaKernel<float, float2><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, ySize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient2DCudaKernel<float, float3>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:			ResampleGradient2DCudaKernel<float, float3><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, ySize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient2DCudaKernel<float, float4>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:			ResampleGradient2DCudaKernel<float, float4><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, ySize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient3DCudaKernel<float, float>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:			ResampleGradient3DCudaKernel<float, float><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient3DCudaKernel<float, float2>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:			ResampleGradient3DCudaKernel<float, float2><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient3DCudaKernel<float, float3>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:			ResampleGradient3DCudaKernel<float, float3><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient3DCudaKernel<float, float4>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:			ResampleGradient3DCudaKernel<float, float4><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		// Allocate cuda array
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		cudaArray* cuArray = createArray(xSize, ySize, zSize, components);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		cudaMemcpy3DParms copyParams = {0};
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		cudaSurfaceObject_t surfaceObject = createSurfaceObject(cuArray);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		cudaTextureObject_t dataTexture = createTextureObject(cuArray);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:			HANDLE_ERROR(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		HANDLE_ERROR(cudaDestroySurfaceObject(surfaceObject));
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		cudaDestroyTextureObject(dataTexture);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:		HANDLE_ERROR(cudaFreeArray(cuArray));
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:// Define the GPU implementation that launches the CUDA kernel.
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:	cudaMemset(dataGradient, 0, dataSize * sizeof(float));
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:	cudaMemset(pointsGradient, 0, pointsSize * sizeof(float));
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:	// Launch the cuda kernel.
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradientCudaKernel<float>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:	cudaMalloc(&dimSizesDevice, dims * sizeof(unsigned int));
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:	cudaMemcpy(dimSizesDevice, dimSizes, dims * sizeof(unsigned int), cudaMemcpyHostToDevice);
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:    cudaMalloc(&q, gridSize * blockSize * dims * sizeof(float));
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:    cudaMalloc(&weights, gridSize * blockSize * (dims + 1) * sizeof(float));
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:	ResampleGradientCudaKernel<float><<<gridSize, blockSize>>>(
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:	HANDLE_ERROR(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:	HANDLE_ERROR(cudaFree(dimSizesDevice));
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:	HANDLE_ERROR(cudaFree(q));
phiml/backend/tensorflow/cuda/src/resample_gradient.cu.cc:	HANDLE_ERROR(cudaFree(weights));
phiml/backend/tensorflow/cuda/src/resample.cc:// Register the GPU kernel.
phiml/backend/tensorflow/cuda/src/resample.cc:REGISTER_KERNEL_BUILDER(Name("Resample").Device(DEVICE_GPU), ResampleOp);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:// Naive CUDA kernel.
phiml/backend/tensorflow/cuda/src/resample.cu.cc:void ResampleCudaKernel(
phiml/backend/tensorflow/cuda/src/resample.cu.cc:// https://devblogs.nvidia.com/lerp-faster-cuda/
phiml/backend/tensorflow/cuda/src/resample.cu.cc:void Resample1DCudaKernel(
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	cudaTextureObject_t dataTexture,
phiml/backend/tensorflow/cuda/src/resample.cu.cc:void Resample2DCudaKernel (
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	cudaTextureObject_t dataTexture,
phiml/backend/tensorflow/cuda/src/resample.cu.cc:void Resample3DCudaKernel (
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	cudaTextureObject_t dataTexture,
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	cudaTextureObject_t dataTexture,
phiml/backend/tensorflow/cuda/src/resample.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample1DCudaKernel<float,float>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:			Resample1DCudaKernel<float, float><<<gridSize, blockSize>>>(batch, xSize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample1DCudaKernel<float,float2>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:			Resample1DCudaKernel<float, float2><<<gridSize, blockSize>>>(batch, xSize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample1DCudaKernel<float,float3>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:			Resample1DCudaKernel<float, float3><<<gridSize, blockSize>>>(batch, xSize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample1DCudaKernel<float,float4>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:			Resample1DCudaKernel<float, float4><<<gridSize, blockSize>>>(batch, xSize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample2DCudaKernel<float,float>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:			Resample2DCudaKernel<float, float><<<gridSize, blockSize>>>(batch, xSize, ySize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample2DCudaKernel<float,float2>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:			Resample2DCudaKernel<float, float2><<<gridSize, blockSize>>>(batch, xSize, ySize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample2DCudaKernel<float,float3>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:			Resample2DCudaKernel<float, float3><<<gridSize, blockSize>>>(batch, xSize, ySize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample2DCudaKernel<float,float4>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:			Resample2DCudaKernel<float, float4><<<gridSize, blockSize>>>(batch, xSize, ySize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample3DCudaKernel<float,float>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:			Resample3DCudaKernel<float, float><<<gridSize, blockSize>>>(batch, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample3DCudaKernel<float,float2>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:			Resample3DCudaKernel<float, float2><<<gridSize, blockSize>>>(batch, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample3DCudaKernel<float,float3>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:			Resample3DCudaKernel<float, float3><<<gridSize, blockSize>>>(batch, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample3DCudaKernel<float,float4>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:			Resample3DCudaKernel<float, float4><<<gridSize, blockSize>>>(batch, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	// Allocate cuda array
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	cudaArray* cuArray = createArray(xSize, ySize, zSize, components);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	cudaMemcpy3DParms copyParams = {0};
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	cudaSurfaceObject_t surfaceObject = createSurfaceObject(cuArray);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	cudaTextureObject_t dataTexture = createTextureObject(cuArray);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:		HANDLE_ERROR(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	HANDLE_ERROR(cudaDestroySurfaceObject(surfaceObject));
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	cudaDestroyTextureObject(dataTexture);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	HANDLE_ERROR(cudaFreeArray(cuArray));
phiml/backend/tensorflow/cuda/src/resample.cu.cc:// Define the GPU implementation that launches the CUDA kernel.
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	cudaMemset(output, 0, outputSize * sizeof(float));
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleCudaKernel<float>, 0, 0);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	cudaMalloc(&dimSizesDevice, dims * sizeof(unsigned int));
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	cudaMemcpy(dimSizesDevice, dimSizes, dims * sizeof(unsigned int), cudaMemcpyHostToDevice);
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	cudaMalloc(&q, gridSize * blockSize * dims * sizeof(float));
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	ResampleCudaKernel<float><<<gridSize, blockSize>>>(
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	HANDLE_ERROR(cudaDeviceSynchronize());
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	HANDLE_ERROR(cudaFree(dimSizesDevice));
phiml/backend/tensorflow/cuda/src/resample.cu.cc:	HANDLE_ERROR(cudaFree(q));
phiml/backend/tensorflow/__init__.py:    for i, device in enumerate(_tf.config.list_physical_devices('GPU')):
phiml/backend/tensorflow/__init__.py:        _LOGGER.info(f"phiml.backend.tf: Setting memory_growth on GPU {i} to True to prevent Blas errors")
phiml/backend/tensorflow/__init__.py:from ._compile_cuda import compile_cuda_ops
phiml/_troubleshoot.py:        gpu_count = len(tf.TENSORFLOW.list_devices('GPU'))
phiml/_troubleshoot.py:    if gpu_count == 0:
phiml/_troubleshoot.py:        return f"Installed ({tf_version}), {gpu_count} GPUs available.\n{tf_prob}"
phiml/_troubleshoot.py:        from .backend.tensorflow._tf_cuda_resample import librariesLoaded
phiml/_troubleshoot.py:            cuda_str = 'CUDA kernels available.'
phiml/_troubleshoot.py:                cuda_str = f"Optional TensorFlow CUDA kernels not available and compilation not recommended on {platform.system()}. GPU will be used nevertheless."
phiml/_troubleshoot.py:                cuda_str = f"Optional TensorFlow CUDA kernels not available. GPU will be used nevertheless. Clone the Φ-ML source from GitHub and run 'python setup.py tf_cuda' to compile them. See https://tum-pbs.github.io/PhiML/Installation_Instructions.html"
phiml/_troubleshoot.py:        return f"Installed ({tf_version}), {gpu_count} GPUs available.\n{cuda_str}\n{tf_prob}"
phiml/_troubleshoot.py:        gpu_count = len(torch_.TORCH.list_devices('GPU'))
phiml/_troubleshoot.py:        return f"Installed ({torch_version}), {gpu_count} GPUs available. This version has known bugs with JIT compilation. Recommended: 1.11+ or 1.8.2 LTS"
phiml/_troubleshoot.py:        return f"Installed ({torch_version}), {gpu_count} GPUs available. You may encounter problems importing torch.fft. Recommended: 1.11+ or 1.8.2 LTS"
phiml/_troubleshoot.py:    return f"Installed ({torch_version}), {gpu_count} GPUs available."
phiml/_troubleshoot.py:        gpu_count = len(jax_.JAX.list_devices('GPU'))
phiml/_troubleshoot.py:        return f"Installed ({version}), {gpu_count} GPUs available. This is an old version of Jax that may not support all required features, e.g. sparse matrices."
phiml/_troubleshoot.py:    return f"Installed ({version}), {gpu_count} GPUs available."
docs/Installation_Instructions.md:For GPU acceleration, deep learning and optimization, either
docs/Installation_Instructions.md:Note that these also require a CUDA installation with *cuDNN* libraries for GPU execution.
docs/Installation_Instructions.md:We recommend CUDA 11.0 with cuDNN 8.
docs/Installation_Instructions.md:*Note*: If you want to use the Φ<sub>ML</sub> CUDA operations with TensorFlow, you have to build Φ<sub>ML</sub> from source instead (see below).
docs/Installation_Instructions.md:Install [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/install) or [Jax](https://github.com/google/jax#installation) in addition to Φ<sub>ML</sub> to enable machine learning capabilities and GPU execution.
docs/Installation_Instructions.md:With this done, you may compile CUDA kernels for better performance, see below.
docs/Installation_Instructions.md:## Compiling the CUDA Kernels for TensorFlow
docs/Installation_Instructions.md:The Φ<sub>ML</sub> source includes several custom CUDA kernels to speed up certain operations when using TensorFlow.
docs/Installation_Instructions.md:To use these, you must have a [TensorFlow compatible CUDA SDK with cuDNN](https://www.tensorflow.org/install/gpu#software_requirements) as well as a compatible C++ compiler installed.
docs/Installation_Instructions.md:To compile the CUDA kernels for TensorFlow, clone the repository as described above, then run `$ python <target directory>/setup.py tf_cuda`.
docs/Installation_Instructions.md:This will add the compiled CUDA binaries to the \<target directory\>.
docs/index.md:* [Selecting compute devices (CPU/GPU/TPU)](Devices.html)
docs/Package_Info.md:See the [installation Instructions](https://tum-pbs.github.io/PhiML/Installation_Instructions.html) on how to compile the optional custom CUDA operations.
.coveragerc:    phiml/backend/tensorflow/_tf_cuda_resample.py
.coveragerc:    phiml/backend/tensorflow/_compile_cuda.py
tests/commit/test_nn.py:# import os; os.environ['CUDA_VISIBLE_DEVICES'] = ""
tests/commit/backend/test__backend.py:    def test_convert(self):  # TODO this causes RuntimeError when GPU capsule is given to Jax in CPU mode
tests/gpu/test_tf_cuda_resample.py:from phiml.backend.tensorflow._tf_cuda_resample import resample_cuda
tests/gpu/test_tf_cuda_resample.py:class TestTfCudaResample(TestCase):
tests/gpu/test_tf_cuda_resample.py:                cuda_resampled = resample_cuda(data_placeholder, points_placeholder, boundary)
tests/gpu/test_tf_cuda_resample.py:                gradient = np.zeros(cuda_resampled.shape, np.float32)
tests/gpu/test_tf_cuda_resample.py:                cuda_data_gradient = (tf.gradients(cuda_resampled, data_placeholder, gradient_placeholder))[0]
tests/gpu/test_tf_cuda_resample.py:                cuda_points_gradient = (tf.gradients(cuda_resampled, points_placeholder, gradient_placeholder))[0]
tests/gpu/test_tf_cuda_resample.py:                    result = sess.run([cuda_resampled, nifty_resampled, cuda_data_gradient, nifty_data_gradient,
tests/gpu/test_tf_cuda_resample.py:                                       cuda_points_gradient, nifty_points_gradient], feed_dict={data_placeholder: data,
tests/gpu/test_tf_cuda_resample.py:        self.global_boundaries('/device:GPU:0')
tests/gpu/test_tf_cuda_resample.py:                    cuda_resampled = resample_cuda(data_placeholder, points_placeholder, boundary)
tests/gpu/test_tf_cuda_resample.py:                        result = sess.run(cuda_resampled, feed_dict={data_placeholder: data, points_placeholder: points})
tests/gpu/test_tf_cuda_resample.py:            cuda_resampled = resample_cuda(data_placeholder, points_placeholder, boundary)
tests/gpu/test_tf_cuda_resample.py:                result = sess.run(cuda_resampled, feed_dict={data_placeholder: data, points_placeholder: points})
tests/gpu/test_tf_cuda_resample.py:        self.mixed_boundaries('/device:GPU:0')
tests/gpu/test_tf_cuda_resample.py:                single = resample_cuda(data_placeholder, points_placeholder, boundary)
tests/gpu/test_tf_cuda_resample.py:                combined = resample_cuda(data_combined_placeholder, points_combined_placeholder, boundary)
tests/gpu/test_tf_cuda_resample.py:                combined = resample_cuda(data_combined_placeholder, points_placeholder, boundary)
tests/gpu/test_tf_cuda_resample.py:                combined = resample_cuda(data_placeholder, points_combined_placeholder, boundary)
tests/gpu/test_tf_cuda_resample.py:        self.batch_sizes('/device:GPU:0')
README.md:Install [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/install) or [Jax](https://github.com/google/jax#installation) to enable machine learning capabilities and GPU execution.
README.md:For optimal GPU performance, you may compile the custom CUDA operators, see the [detailed installation instructions](https://tum-pbs.github.io/PhiML/Installation_Instructions.html).
README.md:* Φ<sub>ML</sub> [abstracts compute devices](https://tum-pbs.github.io/PhiML/Devices.html) but does not currently allow mapping operations onto multiple GPUs.
MANIFEST.in:recursive-include phiml/backend/tf/cuda/build *
CONTRIBUTING.md:- Code optimizations or native (CUDA) implementations.

```
