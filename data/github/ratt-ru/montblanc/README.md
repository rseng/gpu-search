# https://github.com/ratt-ru/montblanc

```console
setup.py:minimum_cuda_version = 8000
setup.py:class InspectCudaException(Exception):
setup.py:    """ Find nvcc and the CUDA installation """
setup.py:    default_cuda_path = os.path.join('usr', 'local', 'cuda')
setup.py:    cuda_path = os.environ.get('CUDA_PATH', default_cuda_path)
setup.py:    cuda_path_found = os.path.exists(cuda_path)
setup.py:    # Can't find either NVCC or some CUDA_PATH
setup.py:    if not nvcc_found and not cuda_path_found:
setup.py:        raise InspectCudaException("Neither nvcc '{}' "
setup.py:            "or the CUDA_PATH '{}' were found!".format(
setup.py:                nvcc_path, cuda_path))
setup.py:    # No NVCC, try find it in the CUDA_PATH
setup.py:            "Searching within the CUDA_PATH '{}'"
setup.py:                .format(nvcc_path, cuda_path))
setup.py:        bin_dir = os.path.join(cuda_path, 'bin')
setup.py:            raise InspectCudaException("nvcc not found in '{}' "
setup.py:                "or under the CUDA_PATH at '{}' "
setup.py:                .format(search_paths, cuda_path))
setup.py:    # No CUDA_PATH found, infer it from NVCC
setup.py:    if not cuda_path_found:
setup.py:        cuda_path = os.path.normpath(
setup.py:        log.warning("CUDA_PATH not found, inferring it as '{}' "
setup.py:                cuda_path, nvcc_path))
setup.py:        cuda_path_found = True
setup.py:    if cuda_path_found:
setup.py:        include_dirs.append(os.path.join(cuda_path, 'include'))
setup.py:            library_dirs.append(os.path.join(cuda_path, 'bin'))
setup.py:            library_dirs.append(os.path.join(cuda_path, 'lib', 'x64'))
setup.py:            library_dirs.append(os.path.join(cuda_path, 'lib64'))
setup.py:            library_dirs.append(os.path.join(cuda_path, 'lib'))
setup.py:        library_dirs.append(os.path.join(default_cuda_path, 'lib'))
setup.py:        'cuda_available' : True,
setup.py:        'libraries' : ['cudart', 'cuda'],
setup.py:def inspect_cuda_version_and_devices(compiler, settings):
setup.py:    Poor mans deviceQuery. Returns CUDA_VERSION information and
setup.py:    CUDA device information in JSON format
setup.py:            #include <cuda.h>
setup.py:              printf("  \\"cuda_version\\": %d,\\n", CUDA_VERSION);
setup.py:              cudaGetDeviceCount(&nr_of_devices);
setup.py:                cudaDeviceProp p;
setup.py:                cudaGetDeviceProperties(&p, d);
setup.py:        msg = ("Running the CUDA device check "
setup.py:        raise InspectCudaException(msg)
setup.py:            raise InspectCudaException(msg)
setup.py:            raise InspectCudaException(msg)
setup.py:        # reset the default compiler_so, which we might have changed for cuda
setup.py:def inspect_cuda():
setup.py:    """ Return cuda device information and nvcc/cuda setup """
setup.py:    output = inspect_cuda_version_and_devices(nvcc_compiler, nvcc_settings)
setup.py:        log.info("NVIDIA cub installation found "
setup.py:    log.info("No NVIDIA cub installation found")
setup.py:        log.info("Valid NVIDIA cub archive found '{}'".format(cub_zip_file))
setup.py:        log.info("NVIDIA cub archive unzipped into '{}'".format(
setup.py:            # use the cuda for .cu files
setup.py:        # reset the default compiler_so, which we might have changed for cuda
setup.py:def cuda_architecture_flags(device_info):
setup.py:    Emit a list of architecture flags for each CUDA device found
setup.py:    ['--gpu-architecture=sm_30', '--gpu-architecture=sm_52']
setup.py:        archs = ['--gpu-architecture=sm_30']
setup.py:        log.info("No CUDA devices found, defaulting to architecture '{}'".format(archs[0]))
setup.py:            arch_str = '--gpu-architecture=sm_{}{}'.format(device['major'], device['minor'])
setup.py:    use_cuda = nvcc_settings is not None and (bool(nvcc_settings['cuda_available'])
setup.py:        and tf.test.is_built_with_cuda())
setup.py:    # Add cuda specific build information, if it is available
setup.py:    if use_cuda:
setup.py:        # CUDA source files
setup.py:        # CUDA include directories
setup.py:        # CUDA header dependencies
setup.py:        # CUDA libraries
setup.py:        # --gpu-architecture=sm_xy flags
setup.py:        nvcc_flags += cuda_architecture_flags(device_info)
setup.py:        nvcc_flags += ['-DGOOGLE_CUDA=%d' % int(use_cuda)]
setup.py:        self.cuda_devices = device_info
setup.py:            self.nvcc_settings, self.cuda_devices, 
setup.py:    use_tf_cuda = False
setup.py:    use_tf_cuda = tf.test.is_built_with_cuda()
setup.py:# Detect CUDA and GPU Devices
setup.py:# See if CUDA is installed and if any NVIDIA devices are available
setup.py:# Choose the tensorflow flavour to install (CPU or GPU)
setup.py:if use_tf_cuda:
setup.py:        # Look for CUDA devices and NVCC/CUDA installation
setup.py:        device_info, nvcc_settings = inspect_cuda()
setup.py:        tensorflow_package = 'tensorflow-gpu'
setup.py:        cuda_version = device_info['cuda_version']
setup.py:        log.info("CUDA '{}' found. "
setup.py:                 "Installing tensorflow GPU".format(cuda_version))
setup.py:        log.info("CUDA installation settings:\n{}"
setup.py:        log.info("CUDA code will be compiled for the following devices:\n{}"
setup.py:    except InspectCudaException as e:
setup.py:        # Can't find a reasonable NVCC/CUDA install. Go with the CPU version
setup.py:        log.exception("CUDA not found: {}. ".format(str(e)))
setup.py:        log.exception("NVIDIA cub install failed.")
setup.py:    device_info, nvcc_settings = {}, {'cuda_available': False}
setup.py:    # Pass NVCC and CUDA settings through to the build extension
setup.py:            'cuda_devices': device_info
setup.py:    description='GPU-accelerated RIME implementations.',
Dockerfile:FROM radioastro/cuda:devel
docs/concepts.rst:Montblanc predicts the model visibilities of an radio interferometer from a parametric sky model. Internally, this computation is performed via either CPUs or GPUs by Google's tensorflow_ framework.
docs/concepts.rst:When the number of visibilities and radio source is large, it becomes more computationally efficient to compute on GPUs. However, the problem space also becomes commensurately larger and therefore requires subdividing the problem so that *tiles*, or chunks, can fit both within the memory budget of a GPU and a CPU-only node.
docs/installation.rst:If you wish to take advantage of GPU Acceleration, the following are required:
docs/installation.rst:- `CUDA 8.0  <CUDA_>`_.
docs/installation.rst:- `cuDNN 6.0 <cudnn_>`_ for CUDA 8.0.
docs/installation.rst:- A Kepler or later model NVIDIA GPU.
docs/installation.rst:- .. _install_tf_gpu:
docs/installation.rst:  Montblanc depends on tensorflow_ for CPU and GPU acceleration.
docs/installation.rst:  If you require GPU acceleration, the GPU version of tensorflow
docs/installation.rst:    $ pip install tensorflow-gpu==1.8.0
docs/installation.rst:- GPU Acceleration requires `CUDA 8.0 <CUDA_>`_ and `cuDNN 6.0 for CUDA 8.0 <cudnn_>`_.
docs/installation.rst:  - It is often easier to CUDA install from the `NVIDIA <CUDA_>`_ site on Linux systems.
docs/installation.rst:  - You will need to sign up for the `NVIDIA Developer Program <cudnn_>`_ to download cudNN.
docs/installation.rst:  During the installation process, Montblanc will inspect your CUDA installation
docs/installation.rst:  to determine if a GPU-supported installation can proceed.
docs/installation.rst:  If your CUDA installation does not live in ``/usr``, it  helps to set a
docs/installation.rst:  **For example**, if CUDA is installed in ``/usr/local/cuda-8.0`` and cuDNN is unzipped
docs/installation.rst:  into ``/usr/local/cudnn-6.0-cuda-8.0``, run the following on the command line or
docs/installation.rst:      # CUDA 8
docs/installation.rst:      $ export CUDA_PATH=/usr/local/cuda-8.0
docs/installation.rst:      $ export PATH=$CUDA_PATH/bin:$PATH
docs/installation.rst:      $ export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
docs/installation.rst:      $ export LD_LIBRARY_PATH=$CUDA_PATH/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
docs/installation.rst:      # CUDNN 6.0 (CUDA 8.0)
docs/installation.rst:      $ export CUDNN_HOME=/usr/local/cudnn-6.0-cuda-8.0
docs/installation.rst:      # Latest NVIDIA drivers
docs/installation.rst:      $ export LD_LIBRARY_PATH=/usr/lib/nvidia-375:$LD_LIBRARY_PATH
docs/installation.rst:Set the ``CUDA_PATH`` so that the setup script can find CUDA:
docs/installation.rst:    $ export CUDA_PATH=/usr/local/cuda-8.0
docs/installation.rst:or somewhere on your ``PATH``, you can leave ``CUDA_PATH`` unset. In this case
docs/installation.rst:setup will infer the CUDA_PATH as ``/usr``
docs/installation.rst:- Montblanc doesn't use your GPU or compile GPU tensorflow operators.
docs/installation.rst:  1. Check if the `GPU version of tensorflow <install_tf_gpu_>`_ is installed.
docs/installation.rst:     It is possible to see if the GPU version of tensorflow is installed by running
docs/installation.rst:     If tensorflow knows about your GPU it will log some information about it:
docs/installation.rst:          2017-05-16 14:24:38.571320: I tensorflow/core/common_runtime/gpu/gpu_device.cc:887] Found device 0 with properties:
docs/installation.rst:          2017-05-16 14:24:38.571352: I tensorflow/core/common_runtime/gpu/gpu_device.cc:908] DMA: 0
docs/installation.rst:          2017-05-16 14:24:38.571372: I tensorflow/core/common_runtime/gpu/gpu_device.cc:918] 0:   Y
docs/installation.rst:          2017-05-16 14:24:38.571403: I tensorflow/core/common_runtime/gpu/gpu_device.cc:977] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 960M, pci bus id: 0000:01:00.0)
docs/installation.rst:  2. The installation process couldn't find your CUDA install.
docs/installation.rst:     It will log information about where it thinks this is and which GPU devices
docs/installation.rst:.. _cuda: https://developer.nvidia.com/cuda-downloads
docs/installation.rst:.. _cudnn: https://developer.nvidia.com/cudnn
docs/installation.rst:.. _tensorflow-gpu: https://pypi.python.org/pypi/tensorflow-gpu
montblanc/util/__init__.py:    """ Context Manager Wrapper for CUDA Contexts! """
montblanc/configuration.py:            'allowed': ['CPU', 'GPU'],
montblanc/configuration.py:            'default': 'GPU',
montblanc/configuration.py:                               "tile of the problem on a CPU/GPU "
montblanc/include/montblanc/abstraction.cuh:// CUDA include required for CUDART_PI_F and CUDART_PI
montblanc/include/montblanc/abstraction.cuh:	constexpr static float cuda_pi = CUDART_PI_F;
montblanc/include/montblanc/abstraction.cuh:	constexpr static double cuda_pi = CUDART_PI;
montblanc/impl/rime/tensorflow/RimeSolver.py:        gpus = [d.name for d in devices if d.device_type == 'GPU']
montblanc/impl/rime/tensorflow/RimeSolver.py:        if device_type == 'GPU' and len(gpus) == 0:
montblanc/impl/rime/tensorflow/RimeSolver.py:            montblanc.log.warning("No GPUs are present, falling back to CPU.")
montblanc/impl/rime/tensorflow/RimeSolver.py:        self._devices = cpus if use_cpus else gpus
montblanc/impl/rime/tensorflow/config.py:        # between the CPU and GPU versions. Go with
montblanc/impl/rime/tensorflow/rime_ops/test_e_beam.py:        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
montblanc/impl/rime/tensorflow/rime_ops/test_e_beam.py:        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
montblanc/impl/rime/tensorflow/rime_ops/test_e_beam.py:                                if d.device_type == 'GPU']
montblanc/impl/rime/tensorflow/rime_ops/test_e_beam.py:        # Run the op on all GPUs
montblanc/impl/rime/tensorflow/rime_ops/test_e_beam.py:        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]
montblanc/impl/rime/tensorflow/rime_ops/test_e_beam.py:            # Compare with GPU ejones
montblanc/impl/rime/tensorflow/rime_ops/test_e_beam.py:            for gpu_ejones in S.run(gpu_ops):
montblanc/impl/rime/tensorflow/rime_ops/test_e_beam.py:                self.assertTrue(np.allclose(cpu_ejones, gpu_ejones),
montblanc/impl/rime/tensorflow/rime_ops/test_e_beam.py:                    "As the CPU and GPU may slightly differ, "
montblanc/impl/rime/tensorflow/rime_ops/test_e_beam.py:                    "and slghtly above on the GPU for instance. "
montblanc/impl/rime/tensorflow/rime_ops/test_e_beam.py:                d = np.invert(np.isclose(cpu_ejones, gpu_ejones))
montblanc/impl/rime/tensorflow/rime_ops/test_radec_to_lm.py:        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
montblanc/impl/rime/tensorflow/rime_ops/test_radec_to_lm.py:        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
montblanc/impl/rime/tensorflow/rime_ops/test_radec_to_lm.py:                         if d.device_type == 'GPU']
montblanc/impl/rime/tensorflow/rime_ops/test_radec_to_lm.py:        # Run the op on all GPUs
montblanc/impl/rime/tensorflow/rime_ops/test_radec_to_lm.py:        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]
montblanc/impl/rime/tensorflow/rime_ops/test_radec_to_lm.py:            for gpu_result in S.run(gpu_ops):
montblanc/impl/rime/tensorflow/rime_ops/test_radec_to_lm.py:                assert np.allclose(cpu_result, gpu_result)
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cuh:#ifndef RIME_PHASE_OP_GPU_H_
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cuh:#define RIME_PHASE_OP_GPU_H_
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cuh:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cuh:// Required in order for Eigen::GpuDevice to be an actual type
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cuh:#define EIGEN_USE_GPU
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cuh:typedef Eigen::GpuDevice GPUDevice;
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cuh:// CUDA kernel computing the phase term
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cuh:// Partially specialise Phase for GPUDevice
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cuh:class Phase<GPUDevice, FT, CT> : public tensorflow::OpKernel {
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cuh:        // Cast input into CUDA types defined within the Traits class
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cuh:        // Cast to the cuda types expected by the kernel
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cuh:        const auto & stream = context->eigen_device<GPUDevice>().stream();
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cuh:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cuh:#endif // #define RIME_PHASE_OP_GPU_H
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op.h://   - post_process_visibilities_op_gpu.cuh for CUDA devices
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op.h://   - post_process_visibilities_op_gpu.cu for CUDA devices
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op.h://   - parallactic_angle_sin_cos_op_gpu.cuh for CUDA devices
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op.h://   - parallactic_angle_sin_cos_op_gpu.cu for CUDA devices
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cu:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cu:#include "sersic_shape_op_gpu.cuh"
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU)
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cu:    SersicShape<GPUDevice, float>);
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU)
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cu:    SersicShape<GPUDevice, double>);
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cu:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cuh:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cuh:#ifndef RIME_SERSIC_SHAPE_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cuh:#define RIME_SERSIC_SHAPE_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cuh:// Required in order for Eigen::GpuDevice to be an actual type
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cuh:#define EIGEN_USE_GPU
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cuh:typedef Eigen::GpuDevice GPUDevice;
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cuh:// CUDA kernel outline
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cuh:// Specialise the SersicShape op for GPUs
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cuh:class SersicShape<GPUDevice, FT> : public tensorflow::OpKernel
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cuh:        const auto & stream = context->eigen_device<GPUDevice>().stream();
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cuh:#endif // #ifndef RIME_SERSIC_SHAPE_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op_gpu.cuh:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cu:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cu:#include "post_process_visibilities_op_gpu.cuh"
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cu:// Register a GPU kernel for PostProcessVisibilities
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU),
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cu:    PostProcessVisibilities<GPUDevice, float, tensorflow::complex64>);
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cu:// Register a GPU kernel for PostProcessVisibilities
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU),
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cu:    PostProcessVisibilities<GPUDevice, double, tensorflow::complex128>);
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cu:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/test_feed_rotation.py:        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
montblanc/impl/rime/tensorflow/rime_ops/test_feed_rotation.py:        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
montblanc/impl/rime/tensorflow/rime_ops/test_feed_rotation.py:                         if d.device_type == 'GPU']
montblanc/impl/rime/tensorflow/rime_ops/test_feed_rotation.py:        # Run the op on all GPUs
montblanc/impl/rime/tensorflow/rime_ops/test_feed_rotation.py:        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]
montblanc/impl/rime/tensorflow/rime_ops/test_feed_rotation.py:            for gpu_feed_rotation in S.run(gpu_ops):
montblanc/impl/rime/tensorflow/rime_ops/test_feed_rotation.py:                                            gpu_feed_rotation))
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cu:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cu:#include "sum_coherencies_op_gpu.cuh"
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cu:// Register a GPU kernel for SumCoherencies that handles floats
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU),
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cu:    SumCoherencies<GPUDevice, float, tensorflow::complex64>);
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cu:// Register a GPU kernel for SumCoherencies that handles doubles
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU),
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cu:    SumCoherencies<GPUDevice, double, tensorflow::complex128>);
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cu:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op.h:// General definition of the GaussShape op, which will be specialised for CPUs and GPUs in
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op.h:// gauss_shape_op_cpu.h and gauss_shape_op_gpu.cuh respectively.
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op.h:// gauss_shape_op_cpu.cpp and gauss_shape_op_gpu.cu respectively
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cuh:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cuh:#ifndef RIME_RADEC_TO_LM_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cuh:#define RIME_RADEC_TO_LM_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cuh:// Required in order for Eigen::GpuDevice to be an actual type
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cuh:#define EIGEN_USE_GPU
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cuh:typedef Eigen::GpuDevice GPUDevice;
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cuh:// CUDA kernel outline
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cuh:// Specialise the RadecToLm op for GPUs
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cuh:class RadecToLm<GPUDevice, FT> : public tensorflow::OpKernel
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cuh:        // Set up our CUDA thread block and grid
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cuh:        // Get the GPU device
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cuh:        const auto & device = context->eigen_device<GPUDevice>();
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cuh:        // Call the rime_radec_to_lm CUDA kernel
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cuh:#endif // #ifndef RIME_RADEC_TO_LM_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cuh:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cu:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cu:#include "gauss_shape_op_gpu.cuh"
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU)
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cu:    GaussShape<GPUDevice, float>);
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU)
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cu:    GaussShape<GPUDevice, double>);
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cu:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op.h:// General definition of the SumCoherencies op, which will be specialised for CPUs and GPUs in
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op.h:// sum_coherencies_op_cpu.h and sum_coherencies_op_gpu.cuh respectively, as well as float types (FT).
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op.h:// sum_coherencies_op_cpu.cpp and sum_coherencies_op_gpu.cu respectively
montblanc/impl/rime/tensorflow/rime_ops/e_beam_op_gpu.cu:#include "e_beam_op_gpu.cuh"
montblanc/impl/rime/tensorflow/rime_ops/e_beam_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU)
montblanc/impl/rime/tensorflow/rime_ops/e_beam_op_gpu.cu:    EBeam<GPUDevice, float, tensorflow::complex64>);
montblanc/impl/rime/tensorflow/rime_ops/e_beam_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU)
montblanc/impl/rime/tensorflow/rime_ops/e_beam_op_gpu.cu:    EBeam<GPUDevice, double, tensorflow::complex128>);
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cu:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cu:#include "parallactic_angle_sin_cos_op_gpu.cuh"
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cu:// Register a GPU kernel for ParallacticAngleSinCos
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU),
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cu:    ParallacticAngleSinCos<GPUDevice, float>);
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cu:// Register a GPU kernel for ParallacticAngleSinCos
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU),
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cu:    ParallacticAngleSinCos<GPUDevice, double>);
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cu:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cu:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cu:#include "radec_to_lm_op_gpu.cuh"
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cu:// Register a GPU kernel for RadecToLm
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU),
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cu:    RadecToLm<GPUDevice, float>);
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cu:// Register a GPU kernel for RadecToLm
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU),
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cu:    RadecToLm<GPUDevice, double>);
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op_gpu.cu:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cu:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cu:#include "feed_rotation_op_gpu.cuh"
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cu:// Register a GPU kernel for FeedRotation
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU),
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cu:    FeedRotation<GPUDevice, float, tensorflow::complex64>);
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cu:// Register a GPU kernel for FeedRotation
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU),
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cu:    FeedRotation<GPUDevice, double, tensorflow::complex128>);
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cu:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/Makefile:TF_CUDA = $(shell python -c 'import tensorflow as tf; print int(tf.test.is_built_with_cuda())')
montblanc/impl/rime/tensorflow/rime_ops/Makefile:# Define our sources, compiling CUDA code if it's enabled
montblanc/impl/rime/tensorflow/rime_ops/Makefile:ifeq ($(TF_CUDA), 1)
montblanc/impl/rime/tensorflow/rime_ops/Makefile:NVCCFLAGS =-std=c++11 -DGOOGLE_CUDA=$(TF_CUDA) $(TF_CFLAGS) $(INCLUDES) \
montblanc/impl/rime/tensorflow/rime_ops/Makefile:	-x cu --compiler-options "-fPIC" --gpu-architecture=sm_30 -lineinfo
montblanc/impl/rime/tensorflow/rime_ops/Makefile:ifeq ($(TF_CUDA), 1)
montblanc/impl/rime/tensorflow/rime_ops/Makefile:	LDFLAGS := $(LDFLAGS) -lcuda -lcudart
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cu:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cu:#include "create_antenna_jones_op_gpu.cuh"
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cu:// Register a GPU kernel for CreateAntennaJones that handles floats
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU),
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cu:    CreateAntennaJones<GPUDevice, float, tensorflow::complex64>);
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cu:// Register a GPU kernel for CreateAntennaJones that handles doubles
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU),
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cu:    CreateAntennaJones<GPUDevice, double, tensorflow::complex128>);
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cu:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cuh:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cuh:#ifndef RIME_PARALLACTIC_ANGLE_SIN_COS_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cuh:#define RIME_PARALLACTIC_ANGLE_SIN_COS_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cuh:// Required in order for Eigen::GpuDevice to be an actual type
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cuh:#define EIGEN_USE_GPU
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cuh:typedef Eigen::GpuDevice GPUDevice;
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cuh:// CUDA kernel outline
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cuh:// Specialise the ParallacticAngleSinCos op for GPUs
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cuh:class ParallacticAngleSinCos<GPUDevice, FT> : public tensorflow::OpKernel
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cuh:        // Set up our CUDA thread block and grid
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cuh:        // Get the GPU device
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cuh:        const auto & device = context->eigen_device<GPUDevice>();
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cuh:        // Call the rime_parallactic_angle_sin_cos CUDA kernel
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cuh:#endif // #ifndef RIME_PARALLACTIC_ANGLE_SIN_COS_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/parallactic_angle_sin_cos_op_gpu.cuh:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cu:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cu:#define EIGEN_USE_GPU
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cu:#include "phase_op_gpu.cuh"
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU)
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cu:    Phase<GPUDevice, float, tensorflow::complex64>);
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU)
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cu:    Phase<GPUDevice, double, tensorflow::complex128>);
montblanc/impl/rime/tensorflow/rime_ops/phase_op_gpu.cu:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/rime_constant_structures.h:#ifdef GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/e_beam_op_gpu.cuh:#ifndef RIME_E_BEAM_OP_GPU_CUH_
montblanc/impl/rime/tensorflow/rime_ops/e_beam_op_gpu.cuh:#define RIME_E_BEAM_OP_GPU_CUH_
montblanc/impl/rime/tensorflow/rime_ops/e_beam_op_gpu.cuh:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/e_beam_op_gpu.cuh:// Required in order for Eigen::GpuDevice to be an actual type
montblanc/impl/rime/tensorflow/rime_ops/e_beam_op_gpu.cuh:#define EIGEN_USE_GPU
montblanc/impl/rime/tensorflow/rime_ops/e_beam_op_gpu.cuh:typedef Eigen::GpuDevice GPUDevice;
montblanc/impl/rime/tensorflow/rime_ops/e_beam_op_gpu.cuh:class EBeam<GPUDevice, FT, CT> : public tensorflow::OpKernel
montblanc/impl/rime/tensorflow/rime_ops/e_beam_op_gpu.cuh:                std::to_string(BEAM_NUD_LIMIT) + "' for the GPU beam."));
montblanc/impl/rime/tensorflow/rime_ops/e_beam_op_gpu.cuh:        const auto & stream = context->eigen_device<GPUDevice>().stream();
montblanc/impl/rime/tensorflow/rime_ops/e_beam_op_gpu.cuh:        // Cast to the cuda types expected by the kernel
montblanc/impl/rime/tensorflow/rime_ops/e_beam_op_gpu.cuh:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/e_beam_op_gpu.cuh:#endif // #ifndef RIME_E_BEAM_OP_GPU_CUH_
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cuh:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cuh:#ifndef RIME_CREATE_ANTENNA_JONES_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cuh:#define RIME_CREATE_ANTENNA_JONES_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cuh:// Required in order for Eigen::GpuDevice to be an actual type
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cuh:#define EIGEN_USE_GPU
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cuh:typedef Eigen::GpuDevice GPUDevice;
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cuh:// CUDA kernel outline
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cuh:// Specialise the CreateAntennaJones op for GPUs
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cuh:class CreateAntennaJones<GPUDevice, FT, CT> : public tensorflow::OpKernel
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cuh:        //GPU kernel above requires this hard-coded number
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cuh:        // Set up our CUDA thread block and grid
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cuh:        // Get the GPU device
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cuh:        const auto & device = context->eigen_device<GPUDevice>();
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cuh:        // Call the rime_create_antenna_jones CUDA kernel
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cuh:#endif // #ifndef RIME_CREATE_ANTENNA_JONES_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_gpu.cuh:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/b_sqrt_op_gpu.cu:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/b_sqrt_op_gpu.cu:#include "b_sqrt_op_gpu.cuh"
montblanc/impl/rime/tensorflow/rime_ops/b_sqrt_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU)
montblanc/impl/rime/tensorflow/rime_ops/b_sqrt_op_gpu.cu:    BSqrt<GPUDevice, float, tensorflow::complex64>);
montblanc/impl/rime/tensorflow/rime_ops/b_sqrt_op_gpu.cu:    .Device(tensorflow::DEVICE_GPU)
montblanc/impl/rime/tensorflow/rime_ops/b_sqrt_op_gpu.cu:    BSqrt<GPUDevice, double, tensorflow::complex128>);
montblanc/impl/rime/tensorflow/rime_ops/test_sersic_shape.py:        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
montblanc/impl/rime/tensorflow/rime_ops/test_sersic_shape.py:        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
montblanc/impl/rime/tensorflow/rime_ops/test_sersic_shape.py:                         if d.device_type == 'GPU']
montblanc/impl/rime/tensorflow/rime_ops/test_sersic_shape.py:        # Run the op on all GPUs
montblanc/impl/rime/tensorflow/rime_ops/test_sersic_shape.py:        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]
montblanc/impl/rime/tensorflow/rime_ops/test_sersic_shape.py:            for gpu_sersic_shape in S.run(gpu_ops):
montblanc/impl/rime/tensorflow/rime_ops/test_sersic_shape.py:                                            gpu_sersic_shape))
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:// General definition of the ${opname} op, which will be specialised for CPUs and GPUs in
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:// ${cpp_header_file} and ${cuda_header_file} respectively, as well as float types (FT).
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:// ${cpp_source_file} and ${cuda_source_file} respectively
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:# Template for the cuda header file (GPU)
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:CUDA_HEADER_TEMPLATE = string.Template(
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:"""#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:#ifndef ${cuda_header_guard}
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:#define ${cuda_header_guard}
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:// Required in order for Eigen::GpuDevice to be an actual type
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:#define EIGEN_USE_GPU
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:typedef Eigen::GpuDevice GPUDevice;
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:// CUDA kernel outline
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:// Specialise the ${opname} op for GPUs
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:class ${opname}<GPUDevice, FT> : public tensorflow::OpKernel
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:        // Set up our CUDA thread block and grid
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:        // Get the GPU device
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:        const auto & device = context->eigen_device<GPUDevice>();
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:        // Call the ${kernel_name} CUDA kernel
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:#endif // #ifndef ${cuda_header_guard}
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:# Template for the cuda source file (GPU)
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:CUDA_SOURCE_TEMPLATE = string.Template(
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:"""#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:#include "${cuda_header_file}"
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:// Register a GPU kernel for ${opname} that handles floats
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:    .Device(tensorflow::DEVICE_GPU),
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:    ${opname}<GPUDevice, float>);
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:// Register a GPU kernel for ${opname} that handles doubles
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:    .Device(tensorflow::DEVICE_GPU),
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:    ${opname}<GPUDevice, double>);
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:# Pin the compute to the GPU
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:with tf.device('/gpu:0'):
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:    expr_gpu = ${module}.${snake_case}(tf_array)
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:    # Run our expressions on CPU and GPU
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:    result_gpu = S.run(expr_gpu)
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:    # and that CPU and GPU results agree
montblanc/impl/rime/tensorflow/rime_ops/op_source_templates.py:    assert np.allclose(result_cpu, result_gpu)
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cuh:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cuh:#ifndef RIME_GAUSS_SHAPE_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cuh:#define RIME_GAUSS_SHAPE_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cuh:// Required in order for Eigen::GpuDevice to be an actual type
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cuh:#define EIGEN_USE_GPU
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cuh:typedef Eigen::GpuDevice GPUDevice;
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cuh:// CUDA kernel outline
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cuh:// Specialise the GaussShape op for GPUs
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cuh:class GaussShape<GPUDevice, FT> : public tensorflow::OpKernel
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cuh:        const auto & stream = context->eigen_device<GPUDevice>().stream();
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cuh:#endif // #ifndef RIME_GAUSS_SHAPE_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/gauss_shape_op_gpu.cuh:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/test_b_sqrt.py:        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
montblanc/impl/rime/tensorflow/rime_ops/test_b_sqrt.py:        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
montblanc/impl/rime/tensorflow/rime_ops/test_b_sqrt.py:                                if d.device_type == 'GPU']
montblanc/impl/rime/tensorflow/rime_ops/test_b_sqrt.py:        # Run the op on all GPUs
montblanc/impl/rime/tensorflow/rime_ops/test_b_sqrt.py:        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]
montblanc/impl/rime/tensorflow/rime_ops/test_b_sqrt.py:            # Compare with GPU bsqrt and invert flag
montblanc/impl/rime/tensorflow/rime_ops/test_b_sqrt.py:            for gpu_bsqrt, gpu_invert in S.run(gpu_ops):
montblanc/impl/rime/tensorflow/rime_ops/test_b_sqrt.py:                self.assertTrue(np.all(cpu_invert == gpu_invert))
montblanc/impl/rime/tensorflow/rime_ops/test_b_sqrt.py:                # Compare cpu and gpu
montblanc/impl/rime/tensorflow/rime_ops/test_b_sqrt.py:                d = np.isclose(cpu_bsqrt, gpu_bsqrt, **tols)
montblanc/impl/rime/tensorflow/rime_ops/test_b_sqrt.py:                it = (np.asarray(p).T, cpu_bsqrt[d], gpu_bsqrt[d])
montblanc/impl/rime/tensorflow/rime_ops/test_b_sqrt.py:                self.fail("CPU/GPU bsqrt failed likely because the "
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op.h://   - feed_rotation_op_gpu.cuh for CUDA devices
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op.h://   - feed_rotation_op_gpu.cu for CUDA devices
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cuh:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cuh:#ifndef RIME_SUM_COHERENCIES_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cuh:#define RIME_SUM_COHERENCIES_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cuh:// Required in order for Eigen::GpuDevice to be an actual type
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cuh:#define EIGEN_USE_GPU
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cuh:typedef Eigen::GpuDevice GPUDevice;
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cuh:// CUDA kernel outline
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cuh:// Specialise the SumCoherencies op for GPUs
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cuh:class SumCoherencies<GPUDevice, FT, CT> : public tensorflow::OpKernel
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cuh:        // Cast input into CUDA types defined within the Traits class
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cuh:        // Set up our CUDA thread block and grid
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cuh:        // Get the GPU device
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cuh:        const auto & device = context->eigen_device<GPUDevice>();
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cuh:        // Call the rime_sum_coherencies CUDA kernel
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cuh:#endif // #ifndef RIME_SUM_COHERENCIES_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/sum_coherencies_op_gpu.cuh:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op.h:// General definition of the SersicShape op, which will be specialised for CPUs and GPUs in
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op.h:// sersic_shape_op_cpu.h and sersic_shape_op_gpu.cuh respectively.
montblanc/impl/rime/tensorflow/rime_ops/sersic_shape_op.h:// sersic_shape_op_cpu.cpp and sersic_shape_op_gpu.cu respectively
montblanc/impl/rime/tensorflow/rime_ops/test_post_process_visibilities.py:        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
montblanc/impl/rime/tensorflow/rime_ops/test_post_process_visibilities.py:        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
montblanc/impl/rime/tensorflow/rime_ops/test_post_process_visibilities.py:                                if d.device_type == 'GPU']
montblanc/impl/rime/tensorflow/rime_ops/test_post_process_visibilities.py:        # Run the op on all GPUs
montblanc/impl/rime/tensorflow/rime_ops/test_post_process_visibilities.py:        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]
montblanc/impl/rime/tensorflow/rime_ops/test_post_process_visibilities.py:            # Compare against the gpu visibilities and chi squared values
montblanc/impl/rime/tensorflow/rime_ops/test_post_process_visibilities.py:            for gpu_vis, gpu_X2 in S.run(gpu_ops):
montblanc/impl/rime/tensorflow/rime_ops/test_post_process_visibilities.py:                self.assertTrue(np.allclose(cpu_vis, gpu_vis))
montblanc/impl/rime/tensorflow/rime_ops/test_post_process_visibilities.py:                self.assertTrue(np.allclose(cpu_X2, gpu_X2))
montblanc/impl/rime/tensorflow/rime_ops/create_op_outline.py:    CUDA_HEADER_TEMPLATE,
montblanc/impl/rime/tensorflow/rime_ops/create_op_outline.py:    CUDA_SOURCE_TEMPLATE,
montblanc/impl/rime/tensorflow/rime_ops/create_op_outline.py:    'cuda_header_file' : ''.join([snake_case, '_op_gpu.cuh']),
montblanc/impl/rime/tensorflow/rime_ops/create_op_outline.py:    'cuda_source_file' : ''.join([snake_case, '_op_gpu.cu']),
montblanc/impl/rime/tensorflow/rime_ops/create_op_outline.py:    'cuda_header_guard' : header_guard(D['cuda_header_file']),
montblanc/impl/rime/tensorflow/rime_ops/create_op_outline.py:with open(D['cuda_header_file'], 'w') as f:
montblanc/impl/rime/tensorflow/rime_ops/create_op_outline.py:    f.write(CUDA_HEADER_TEMPLATE.substitute(**D))
montblanc/impl/rime/tensorflow/rime_ops/create_op_outline.py:with open(D['cuda_source_file'], 'w') as f:
montblanc/impl/rime/tensorflow/rime_ops/create_op_outline.py:    f.write(CUDA_SOURCE_TEMPLATE.substitute(**D))
montblanc/impl/rime/tensorflow/rime_ops/test_sum_coherencies.py:        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
montblanc/impl/rime/tensorflow/rime_ops/test_sum_coherencies.py:        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
montblanc/impl/rime/tensorflow/rime_ops/test_sum_coherencies.py:                         if d.device_type == 'GPU']
montblanc/impl/rime/tensorflow/rime_ops/test_sum_coherencies.py:        # Run the op on all GPUs
montblanc/impl/rime/tensorflow/rime_ops/test_sum_coherencies.py:        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]
montblanc/impl/rime/tensorflow/rime_ops/test_sum_coherencies.py:            #Compare against the GPU coherencies
montblanc/impl/rime/tensorflow/rime_ops/test_sum_coherencies.py:            for gpu_coherencies in S.run(gpu_ops):
montblanc/impl/rime/tensorflow/rime_ops/test_sum_coherencies.py:               self.assertTrue(np.allclose(cpu_coherencies, gpu_coherencies,
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cuh:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cuh:#ifndef RIME_FEED_ROTATION_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cuh:#define RIME_FEED_ROTATION_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cuh:// Required in order for Eigen::GpuDevice to be an actual type
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cuh:#define EIGEN_USE_GPU
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cuh:typedef Eigen::GpuDevice GPUDevice;
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cuh:// CUDA kernel outline
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cuh:// CUDA kernel outline
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cuh:// Specialise the FeedRotation op for GPUs
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cuh:class FeedRotation<GPUDevice, FT, CT> : public tensorflow::OpKernel
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cuh:        // Set up our CUDA thread block and grid
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cuh:        // Get the GPU device
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cuh:        const auto & device = context->eigen_device<GPUDevice>();
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cuh:            // Call the linear feed rotation CUDA kernel
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cuh:            // Call the circular feed rotation CUDA kernel
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cuh:#endif // #ifndef RIME_FEED_ROTATION_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/feed_rotation_op_gpu.cuh:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:#ifndef RIME_POST_PROCESS_VISIBILITIES_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:#define RIME_POST_PROCESS_VISIBILITIES_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:// Required in order for Eigen::GpuDevice to be an actual type
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:#define EIGEN_USE_GPU
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:typedef Eigen::GpuDevice GPUDevice;
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:// CUDA kernel outline
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:// Specialise the PostProcessVisibilities op for GPUs
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:class PostProcessVisibilities<GPUDevice, FT, CT> : public tensorflow::OpKernel
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:        // Get the GPU device
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:        const auto & device = context->eigen_device<GPUDevice>();
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:        // Create a GPU Allocator
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:        tf::AllocatorAttributes gpu_allocator;
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:        gpu_allocator.set_gpu_compatible(true);
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:            &chi_squared_terms, gpu_allocator));
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:            &temp_storage, gpu_allocator));
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:        // Set up our CUDA thread block and grid
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:        // Call the rime_post_process_visibilities CUDA kernel
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:#endif // #ifndef RIME_POST_PROCESS_VISIBILITIES_OP_GPU_CUH
montblanc/impl/rime/tensorflow/rime_ops/post_process_visibilities_op_gpu.cuh:#endif // #if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/test_phase.py:    # No GPU implementation of exp yet
montblanc/impl/rime/tensorflow/rime_ops/test_phase.py:        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
montblanc/impl/rime/tensorflow/rime_ops/test_phase.py:        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
montblanc/impl/rime/tensorflow/rime_ops/test_phase.py:                         if d.device_type == 'GPU']
montblanc/impl/rime/tensorflow/rime_ops/test_phase.py:        # Run the op on all GPUs
montblanc/impl/rime/tensorflow/rime_ops/test_phase.py:        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]
montblanc/impl/rime/tensorflow/rime_ops/test_phase.py:            for phase_gpu in S.run(gpu_ops):
montblanc/impl/rime/tensorflow/rime_ops/test_phase.py:                assert np.allclose(phase_cpu, phase_gpu)
montblanc/impl/rime/tensorflow/rime_ops/test_parallactic_angle_sin_cos.py:        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
montblanc/impl/rime/tensorflow/rime_ops/test_parallactic_angle_sin_cos.py:        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
montblanc/impl/rime/tensorflow/rime_ops/test_parallactic_angle_sin_cos.py:                                if d.device_type == 'GPU']
montblanc/impl/rime/tensorflow/rime_ops/test_parallactic_angle_sin_cos.py:        # Run the op on all GPUs
montblanc/impl/rime/tensorflow/rime_ops/test_parallactic_angle_sin_cos.py:        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]
montblanc/impl/rime/tensorflow/rime_ops/test_parallactic_angle_sin_cos.py:            # Compare with GPU sincos
montblanc/impl/rime/tensorflow/rime_ops/test_parallactic_angle_sin_cos.py:            for gpu_pa_sin, gpu_pa_cos in S.run(gpu_ops):
montblanc/impl/rime/tensorflow/rime_ops/test_parallactic_angle_sin_cos.py:                self.assertTrue(np.allclose(cpu_pa_sin, gpu_pa_sin))
montblanc/impl/rime/tensorflow/rime_ops/test_parallactic_angle_sin_cos.py:                self.assertTrue(np.allclose(cpu_pa_cos, gpu_pa_cos))
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op.h:// General definition of the CreateAntennaJones op, which will be specialised for CPUs and GPUs in
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op.h:// create_antenna_jones_op_cpu.h and create_antenna_jones_op_gpu.cuh respectively, as well as float types (FT).
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op.h:// create_antenna_jones_op_cpu.cpp and create_antenna_jones_op_gpu.cu respectively
montblanc/impl/rime/tensorflow/rime_ops/b_sqrt_op_gpu.cuh:#ifndef RIME_B_SQRT_OP_GPU_H_
montblanc/impl/rime/tensorflow/rime_ops/b_sqrt_op_gpu.cuh:#define RIME_B_SQRT_OP_GPU_H_
montblanc/impl/rime/tensorflow/rime_ops/b_sqrt_op_gpu.cuh:#if GOOGLE_CUDA
montblanc/impl/rime/tensorflow/rime_ops/b_sqrt_op_gpu.cuh:// Required in order for Eigen::GpuDevice to be an actual type
montblanc/impl/rime/tensorflow/rime_ops/b_sqrt_op_gpu.cuh:#define EIGEN_USE_GPU
montblanc/impl/rime/tensorflow/rime_ops/b_sqrt_op_gpu.cuh:typedef Eigen::GpuDevice GPUDevice;
montblanc/impl/rime/tensorflow/rime_ops/b_sqrt_op_gpu.cuh:class BSqrt<GPUDevice, FT, CT> : public tensorflow::OpKernel
montblanc/impl/rime/tensorflow/rime_ops/b_sqrt_op_gpu.cuh:        // Cast input into CUDA types defined within the Traits class
montblanc/impl/rime/tensorflow/rime_ops/b_sqrt_op_gpu.cuh:        // Get the device pointers of our GPU memory arrays
montblanc/impl/rime/tensorflow/rime_ops/b_sqrt_op_gpu.cuh:        const auto & stream = context->eigen_device<GPUDevice>().stream();
montblanc/impl/rime/tensorflow/rime_ops/b_sqrt_op_gpu.cuh:#endif // #ifndef RIME_B_SQRT_OP_GPU_H_
montblanc/impl/rime/tensorflow/rime_ops/test_gauss_shape.py:        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
montblanc/impl/rime/tensorflow/rime_ops/test_gauss_shape.py:        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
montblanc/impl/rime/tensorflow/rime_ops/test_gauss_shape.py:                         if d.device_type == 'GPU']
montblanc/impl/rime/tensorflow/rime_ops/test_gauss_shape.py:        # Run the op on all GPUs
montblanc/impl/rime/tensorflow/rime_ops/test_gauss_shape.py:        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]
montblanc/impl/rime/tensorflow/rime_ops/test_gauss_shape.py:            for gpu_gauss_shape in S.run(gpu_ops):
montblanc/impl/rime/tensorflow/rime_ops/test_gauss_shape.py:                                            gpu_gauss_shape))
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op.h://   - radec_to_lm_op_gpu.cuh for CUDA devices
montblanc/impl/rime/tensorflow/rime_ops/radec_to_lm_op.h://   - radec_to_lm_op_gpu.cu for CUDA devices
montblanc/impl/rime/tensorflow/rime_ops/test_create_antenna_jones.py:        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
montblanc/impl/rime/tensorflow/rime_ops/test_create_antenna_jones.py:        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
montblanc/impl/rime/tensorflow/rime_ops/test_create_antenna_jones.py:                                if d.device_type == 'GPU']
montblanc/impl/rime/tensorflow/rime_ops/test_create_antenna_jones.py:        # Run the op on all GPUs
montblanc/impl/rime/tensorflow/rime_ops/test_create_antenna_jones.py:        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]
montblanc/impl/rime/tensorflow/rime_ops/test_create_antenna_jones.py:            # Compare with GPU sincos
montblanc/impl/rime/tensorflow/rime_ops/test_create_antenna_jones.py:            for gpu_aj in S.run(gpu_ops):
montblanc/impl/rime/tensorflow/rime_ops/test_create_antenna_jones.py:                self.assertTrue(np.allclose(cpu_aj, gpu_aj))
montblanc/impl/rime/tensorflow/rime_ops/create_antenna_jones_op_cpu.h:        //GPU kernel above requires this hard-coded number
montblanc/solvers/rime_solver.py:        Used in templated GPU kernels.

```
