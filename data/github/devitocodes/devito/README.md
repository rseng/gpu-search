# https://github.com/devitocodes/devito

```console
setup.py:with open('requirements-nvidia.txt') as f:
setup.py:    nvidias = f.read().splitlines()
setup.py:for mreqs, mode in (zip([optionals, mpis, nvidias, testing],
setup.py:                        ['extras', 'mpi', 'nvidia', 'tests'])):
requirements-nvidia.txt:cupy-cuda12x
requirements-nvidia.txt:dask-cuda
tests/test_gpu_common.py:from devito.arch import get_gpu_info, get_cpu_info, Device, Cpu64
tests/test_gpu_common.py:class TestGPUInfo:
tests/test_gpu_common.py:    def test_get_gpu_info(self):
tests/test_gpu_common.py:        info = get_gpu_info()
tests/test_gpu_common.py:        known = ['nvidia', 'tesla', 'geforce', 'quadro', 'amd', 'unspecified']
tests/test_gpu_common.py:            # There might be than one GPUs, but for now we don't care
tests/test_gpu_common.py:            pytest.xfail("Unsupported platform for get_gpu_info")
tests/test_gpu_common.py:        b = 17 if configuration['language'] == 'openacc' else 16  # No `qid` w/ OMP
tests/test_gpu_common.py:        b = 18 if configuration['language'] == 'openacc' else 17  # No `qid` w/ OMP
tests/test_gpu_common.py:        b = 21 if configuration['language'] == 'openacc' else 20  # No `qid` w/ OMP
tests/test_gpu_common.py:        op0 = Operator(eqns, opt=('noop', {'gpu-fit': usave}))
tests/test_gpu_common.py:        b = 21 if configuration['language'] == 'openacc' else 20  # No `qid` w/ OMP
tests/test_gpu_common.py:        if configuration['language'] == 'openacc':
tests/test_gpu_common.py:        if configuration['language'] == 'openacc':
tests/test_gpu_common.py:        op0 = Operator(eqn, opt=('noop', {'gpu-fit': u}))
tests/test_gpu_common.py:        if configuration['language'] == 'openacc':
tests/test_gpu_common.py:        op0 = Operator(eqns, opt=('noop', {'gpu-fit': u}))
tests/test_gpu_common.py:        op0 = Operator(eqns, opt=('noop', {'gpu-fit': u}))
tests/test_gpu_common.py:        op0 = Operator(eqns, opt=('noop', {'gpu-fit': usave}))
tests/test_gpu_common.py:        if configuration['language'] == 'openacc':
tests/test_gpu_common.py:        op0 = Operator(eqns, opt=('noop', {'gpu-fit': usave}))
tests/test_gpu_common.py:        op0 = Operator(eqns, opt=('noop', {'gpu-fit': (usave, vsave)}))
tests/test_gpu_common.py:        if configuration['language'] == 'openacc':
tests/test_gpu_common.py:        op0 = Operator(eqns, opt=('noop', {'gpu-fit': (fsave, usave)}))
tests/test_gpu_common.py:        op0 = Operator(eqns, opt=('noop', {'gpu-fit': (u, v)}))
tests/test_gpu_common.py:        op0 = Operator(eqns, opt=('cire-sops', {'gpu-fit': usave}))
tests/test_gpu_common.py:    @pytest.mark.parametrize('opt,gpu_fit,async_degree,linearize', [
tests/test_gpu_common.py:    def test_save(self, opt, gpu_fit, async_degree, linearize):
tests/test_gpu_common.py:                      opt=(opt, {'gpu-fit': usave if gpu_fit else None,
tests/test_gpu_common.py:        if configuration['language'] == 'openacc':
tests/test_gpu_common.py:        if configuration['language'] == 'openacc':
tests/test_gpu_common.py:        if configuration['language'] == 'openacc':
tests/test_gpu_common.py:                                   'gpu-fit': [vsave]}))
tests/test_gpu_common.py:    @pytest.mark.parametrize('opt,opt_options,gpu_fit', [
tests/test_gpu_common.py:    def test_xcor_from_saved(self, opt, opt_options, gpu_fit):
tests/test_gpu_common.py:        opt_options = {'gpu-fit': usave if gpu_fit else None, **opt_options}
tests/test_gpu_common.py:    def test_gpu_create_forward(self):
tests/test_gpu_common.py:                      opt=('buffering', 'streaming', 'orchestrate', {'gpu-create': u}))
tests/test_gpu_common.py:        if language == 'openacc':
tests/test_gpu_common.py:    def test_gpu_create_backward(self):
tests/test_gpu_common.py:                      opt=('buffering', 'streaming', 'orchestrate', {'gpu-create': u}))
tests/test_gpu_common.py:        if language == 'openacc':
tests/test_gpu_common.py:        if language == 'openacc':
tests/test_gpu_common.py:                                 {'gpu-fit': [gsave]}))
tests/test_gpu_common.py:    def test_gpu_fit_w_tensor_functions(self):
tests/test_gpu_common.py:        op = Operator(eqns, opt=('noop', {'gpu-fit': usave}))
tests/test_gpu_common.py:        assert set(op._options['gpu-fit']) - set(usave.values()) == set()
tests/test_gpu_common.py:        op = Operator(eqns, opt=('noop', {'gpu-fit': [usave, usave2]}))
tests/test_gpu_common.py:        assert set(op._options['gpu-fit']) - vals == set()
tests/test_pickle.py:        # for multi-node multi-gpu execution, when DeviceID will have
tests/test_gpu_openmp.py:             '  int ngpus = omp_get_num_devices();\n'
tests/test_gpu_openmp.py:             '  omp_set_default_device((rank)%(ngpus));\n}')
tests/test_gpu_openmp.py:            Operator(Eq(u.forward, u + 1), language='openmp', opt='openacc')
tests/test_gpu_openmp.py:                      platform='nvidiaX', language='openmp', opt=opt)
tests/test_operator.py:        op2 = Operator(Eq(u, u + 1), platform='nvidiaX')
tests/test_operator.py:        Operator(Eq(u, u + 1), platform='nvidiaX', compiler='gcc')
tests/test_operator.py:            Operator(Eq(u, u + 1), platform='nvidiaX', compiler='asf')
tests/test_operator.py:        op3 = Operator(Eq(u, u + 1), platform='nvidiaX', language='openacc')
tests/test_operator.py:            Operator(Eq(u, u + 1), platform='bdw', language='openacc')
tests/test_iet.py:def test_make_cuda_stream():
tests/test_iet.py:    class CudaStream(LocalObject):
tests/test_iet.py:        dtype = type('cudaStream_t', (c_void_p,), {})
tests/test_iet.py:            return Call('cudaStreamCreate', Byref(self))
tests/test_iet.py:            return Call('cudaStreamDestroy', self)
tests/test_iet.py:    stream = CudaStream('stream')
tests/test_iet.py:  cudaStream_t stream;
tests/test_iet.py:  cudaStreamCreate(&(stream));
tests/test_iet.py:  cudaStreamDestroy(stream);
tests/test_gpu_openacc.py:        op = Operator(Eq(u.forward, u + 1), platform='nvidiaX', language='openacc')
tests/test_gpu_openacc.py:        op1 = Operator(Eq(u.forward, u + 1), platform='nvidiaX', language='openacc',
tests/test_gpu_openacc.py:                      platform='nvidiaX', language='openacc', opt='openacc')
tests/test_gpu_openacc.py:                     platform='nvidiaX', language='openacc', opt='openmp')
tests/test_gpu_openacc.py:                      platform='nvidiaX', language='openacc', opt=opt)
tests/test_gpu_openacc.py:        op = Operator(eqns, platform='nvidiaX', language='openacc',
tests/test_gpu_openacc.py:        op = Operator(eqns, platform='nvidiaX', language='openacc',
tests/test_gpu_openacc.py:        op = Operator(eqns, platform='nvidiaX', language='openacc',
tests/test_gpu_openacc.py:        # Make sure we've indeed generated OpenACC code
tests/test_gpu_openacc.py:        op = Operator([stencil] + src_term + rec_term, opt=opt, language='openacc')
tests/test_gpu_openacc.py:        # Make sure we've indeed generated OpenACC code
tests/test_gpu_openacc.py:        op = Operator(Eq(u.forward, expr), platform='nvidiaX', language='openacc')
tests/test_gpu_openacc.py:        # Make sure we've indeed generated OpenACC+MPI code
tests/test_dse.py:        op1 = Operator(eq, platform='nvidiaX', language='openacc')
README.md:[![Build Status on GPU](https://github.com/devitocodes/devito/workflows/CI-gpu/badge.svg)](https://github.com/devitocodes/devito/actions?query=workflow%3ACI-gpu)
README.md:kernels on several computer platforms, including CPUs, GPUs, and clusters
README.md:  GPU parallelism via OpenMP and OpenACC, multi-node parallelism via MPI,
README.md:* Generation of parallel code (CPU, GPU, multi-node via MPI);
docker-compose.yml:  devito.nvidia:
docker-compose.yml:        base: devitocodes/bases:nvidia-nvc
docker-compose.yml:    runtime: nvidia
MANIFEST.in:include requirements-nvidia.txt
docker/nvdashboard.json:{"data":{"layout-restorer:data":{"main":{"dock":{"type":"split-area","orientation":"horizontal","sizes":[0.631830167247976,0.36816983275202403],"children":[{"type":"tab-area","currentIndex":1,"widgets":["terminal:1"]},{"type":"split-area","orientation":"vertical","sizes":[0.300012537867128,0.3293396034114645,0.3706478587214075],"children":[{"type":"tab-area","currentIndex":0,"widgets":["nvdashboard-launcher:/GPU-Utilization"]},{"type":"tab-area","currentIndex":0,"widgets":["nvdashboard-launcher:/GPU-Memory"]},{"type":"split-area","orientation":"horizontal","sizes":[0.5,0.5],"children":[{"type":"tab-area","currentIndex":0,"widgets":["nvdashboard-launcher:/PCIe-Throughput"]},{"type":"tab-area","currentIndex":0,"widgets":["nvdashboard-launcher:/NVLink-Throughput"]}]}]}]},"mode":"multiple-document","current":"terminal:1"},"left":{"collapsed":false,"current":"filebrowser","widgets":["filebrowser","running-sessions","nvdashboard-launcher","command-palette","tab-manager"]},"right":{"collapsed":true,"widgets":[]}},"nvdashboard-launcher:/GPU-Utilization":{"data":{"label":"GPU Utilization","route":"/GPU-Utilization"}},"terminal:1":{"data":{"name":"1"}},"nvdashboard-launcher:/GPU-Memory":{"data":{"label":"GPU Memory","route":"/GPU-Memory"}},"nvdashboard-launcher:/PCIe-Throughput":{"data":{"label":"PCIe Throughput","route":"/PCIe-Throughput"}},"nvdashboard-launcher:/NVLink-Throughput":{"data":{"label":"NVLink Throughput","route":"/NVLink-Throughput"}}},"metadata":{"id":"/lab"}}
docker/Dockerfile.devito:    cmake .. -DNVIDIA_SUPPORT=ON -DAMDGPU_SUPPORT=ON -DINTEL_SUPPORT=ON && \
docker/entrypoint.sh:if [[ "$DEVITO_PLATFORM" = "nvidiaX" ]]; then
docker/Dockerfile.nvidia:# This Dockerfile contains the NVidia HPC SDK (nvc, cuda, OpenMPI) for Devito
docker/Dockerfile.nvidia:RUN curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | gpg --yes --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
docker/Dockerfile.nvidia:RUN echo 'deb [trusted=yes, signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' | tee /etc/apt/sources.list.d/nvhpc.list
docker/Dockerfile.nvidia:        wget https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64/nvhpc_${year}.${minor}_amd64.deb && \
docker/Dockerfile.nvidia:# nvidia-container-runtime
docker/Dockerfile.nvidia:ENV NVIDIA_VISIBLE_DEVICES all
docker/Dockerfile.nvidia:ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
docker/Dockerfile.nvidia:ENV NCCL_UCX_RNDV_THRESH=0
docker/Dockerfile.nvidia:ENV NCCL_UCX_RNDV_SCHEME=get_zcopy
docker/Dockerfile.nvidia:ENV NCCL_PLUGIN_P2P=ucx
docker/Dockerfile.nvidia:ENV UCX_TLS=cuda,cuda_copy,cuda_ipc,sm,shm,self
docker/Dockerfile.nvidia:#ENV UCX_TLS=cuda,cuda_copy,cuda_ipc,sm,shm,self,rc_x,gdr_copy
docker/Dockerfile.nvidia:RUN export NVARCH=$(ls -1 /opt/nvidia/hpc_sdk/Linux_x86_64/ | grep '\.' | head -n 1) && \
docker/Dockerfile.nvidia:    export CUDA_V=$(ls /opt/nvidia/hpc_sdk/Linux_x86_64/${NVARCH}/cuda/ | grep '\.') && \
docker/Dockerfile.nvidia:    ln -sf /opt/nvidia/hpc_sdk/Linux_x86_64/${NVARCH} /opt/nvhpc && \
docker/Dockerfile.nvidia:    ln -sf /opt/nvidia/hpc_sdk/Linux_x86_64/${NVARCH}/cuda/${CUDA_V}/extras/CUPTI /opt/CUPTI && \
docker/Dockerfile.nvidia:    ln -sf /opt/nvidia/hpc_sdk/Linux_x86_64/comm_libs/${CUDA_V}/nvshmem /opt/nvhpc/comm_libs/nvshmem && \
docker/Dockerfile.nvidia:    ln -sf /opt/nvidia/hpc_sdk/Linux_x86_64/comm_libs/${CUDA_V}/nccl /opt/nvhpc/comm_libs/nccl
docker/Dockerfile.nvidia:# Starting nvhpc 23.5 and cuda 12.1, hpcx and openmpi are inside the cuda version folder, only the bin is in the comm_libs path
docker/Dockerfile.nvidia:RUN export CUDA_V=$(/opt/nvhpc/cuda/bin/nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p') && \
docker/Dockerfile.nvidia:    ls /opt/nvhpc/comm_libs/${CUDA_V}/hpcx/ &&\
docker/Dockerfile.nvidia:    if [ -d /opt/nvhpc/comm_libs/${CUDA_V}/hpcx ]; then \
docker/Dockerfile.nvidia:        ln -sf /opt/nvhpc/comm_libs/${CUDA_V}/hpcx /opt/nvhpc/comm_libs/hpcx && \
docker/Dockerfile.nvidia:        ln -sf /opt/nvhpc/comm_libs/${CUDA_V}/openmpi4 /opt/nvhpc/comm_libs/openmpi4;\
docker/Dockerfile.nvidia:# required for nvidia-docker v1
docker/Dockerfile.nvidia:RUN echo "$HPCSDK_HOME/cuda/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
docker/Dockerfile.nvidia:    echo "$HPCSDK_HOME/cuda/lib64" >> /etc/ld.so.conf.d/nvidia.conf && \
docker/Dockerfile.nvidia:    echo "$HPCSDK_HOME/compilers/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
docker/Dockerfile.nvidia:    echo "$HPCSDK_HOME/comm_libs/mpi/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
docker/Dockerfile.nvidia:    echo "$HPCSDK_CUPTI/lib64" >> /etc/ld.so.conf.d/nvidia.conf && \
docker/Dockerfile.nvidia:    echo "$HPCSDK_HOME/math_libs/lib64" >> /etc/ld.so.conf.d/nvidia.conf    
docker/Dockerfile.nvidia:# Compiler, CUDA, and Library paths
docker/Dockerfile.nvidia:# CUDA_HOME has been deprecated but keep for now because of other dependencies (@mloubout).
docker/Dockerfile.nvidia:ENV CUDA_HOME $HPCSDK_HOME/cuda
docker/Dockerfile.nvidia:ENV NVHPC_CUDA_HOME $HPCSDK_HOME/cuda
docker/Dockerfile.nvidia:ENV CUDA_ROOT $HPCSDK_HOME/cuda/bin
docker/Dockerfile.nvidia:ENV PATH $HPCSDK_HOME/compilers/bin:$HPCSDK_HOME/cuda/bin:$HPCSDK_HOME/comm_libs/mpi/bin:${PATH}
docker/Dockerfile.nvidia:ENV LD_LIBRARY_PATH $HPCSDK_HOME/cuda/lib:$HPCSDK_HOME/cuda/lib64:$HPCSDK_HOME/compilers/lib:$HPCSDK_HOME/math_libs/lib64:$HPCSDK_HOME/comm_libs/mpi/lib:$HPCSDK_CUPTI/lib64:bitcomp_DIR:${LD_LIBRARY_PATH}
docker/Dockerfile.nvidia:ENV CPATH $HPCSDK_HOME/comm_libs/mpi/include:$HPCSDK_HOME/comm_libs/nvshmem/include:$HPCSDK_HOME/comm_libs/nccl/include:${CPATH}
docker/Dockerfile.nvidia:# Install python nvidia dependencies
docker/Dockerfile.nvidia:    /venv/bin/pip install --no-cache-dir -r https://raw.githubusercontent.com/devitocodes/devito/master/requirements-nvidia.txt && \
docker/Dockerfile.nvidia:    # Install jupyter and setup nvidia configs.
docker/Dockerfile.nvidia:# NVC for GPUs via OpenACC config
docker/Dockerfile.nvidia:ENV DEVITO_PLATFORM="nvidiaX"
docker/Dockerfile.nvidia:ENV DEVITO_LANGUAGE="openacc"
docker/Dockerfile.nvidia:# NVC for GPUs via CUDA config
docker/Dockerfile.nvidia:ENV DEVITO_ARCH="cuda"
docker/Dockerfile.nvidia:ENV DEVITO_PLATFORM="nvidiaX"
docker/Dockerfile.nvidia:ENV DEVITO_LANGUAGE="cuda"
docker/README.md:### [Devito] on GPU
docker/README.md:Second, we provide three images to run [Devito] on GPUs, tagged `devito:nvidia-nvc-*`, and `devito:amd-*`.
docker/README.md:- `devito:nvidia-nvc-*` is intended to be used on NVidia GPUs. It comes with the configuration to use the `nvc` compiler for `openacc` offloading. This image also comes with CUDA-aware MPI for multi-GPU deployment.
docker/README.md:- `devito:amd-*` is intended to be used on AMD GPUs. It comes with the configuration to use the `aoompcc` compiler for `openmp` offloading. This image also comes with ROCm-aware MPI for multi-GPU deployment. This image can also be used on AMD CPUs since the ROCm compilers are preinstalled.
docker/README.md:#### NVidia
docker/README.md:To run the NVidia GPU version, you will need [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed and to specify the GPUs to use at runtime with the `--gpus` flag. See, for example, a few runtime commands for the NVidia `nvc` images.
docker/README.md:docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 devitocodes/devito:nvidia-nvc-latest
docker/README.md:docker run --gpus all --rm -it devitocodes/devito:nvidia-nvc-latest python examples/seismic/acoustic/acoustic_example.py
docker/README.md:docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --device=/dev/infiniband/uverbs0 --device=/dev/infiniband/rdma_cm  devitocodes/devito:nvidia-nvc-latest
docker/README.md:docker run --gpus all --rm -it -v `pwd`:`pwd` -w `pwd` -u $(id -u):$(id -g) devitocodes/devito:nvidia-nvc-latest python examples/seismic/acoustic/acoustic_example.py
docker/README.md:Unlike NVidia, AMD does not require an additional Docker setup and runs with the standard docker. You will, however, need to pass some flags so that the image is linked to the GPU devices. You can find a short walkthrough in these [AMD notes](https://developer.amd.com/wp-content/resources/ROCm%20Learning%20Centre/chapter5/Chapter5.3_%20KerasMultiGPU_ROCm.pdf) for their TensorFlow GPU Docker image.
docker/README.md:- `devito:gpu-*` that corresponds to `devito:nvidia-nvc-*`
docker/README.md:To build the images yourself, you only need to run the standard build command using the provided Dockerfile. The main difference between the CPU and GPU images will be the base image used.
docker/README.md:To build the GPU image with `openacc` offloading and the `nvc` compiler, run:
docker/README.md:docker build --build-arg base=devitocodes/bases:nvidia-nvc --network=host --file docker/Dockerfile.devito --tag devito .
docker/README.md:### Example GPU
docker/README.md:# Start a terminal to develop/run for GPUs using docker compose
docker/README.md:docker-compose run devito.nvidia /bin/bash
docker/Dockerfile.intel:# Drivers mandatory for intel gpu
docker/Dockerfile.intel:# https://dgpu-docs.intel.com/driver/installation.html#ubuntu-install-steps
docker/Dockerfile.intel:RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy unified" >  /etc/apt/sources.list.d/intel-gpu-jammy.list
docker/Dockerfile.intel:    apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero \
docker/Dockerfile.intel:# ICX SYCL GPU image
docker/Dockerfile.intel:FROM icx as gpu-sycl
docker/Dockerfile.intel:ENV DEVITO_PLATFORM="intelgpuX"
docker/Dockerfile.amd:# Based on  https://github.com/amd/InfinityHub-CI/tree/main/base-gpu-mpi-rocm-docker
docker/Dockerfile.amd:ARG ROCM_VERSION=5.5.1
docker/Dockerfile.amd:FROM rocm/dev-ubuntu-22.04:${ROCM_VERSION}-complete as sdk-base
docker/Dockerfile.amd:ENV ROCM_HOME=/opt/rocm \
docker/Dockerfile.amd:# Until rocm base has it fixed
docker/Dockerfile.amd:RUN ln -s /opt/rocm/llvm/bin/offload-arch /opt/rocm/bin/offload-arch
docker/Dockerfile.amd:# Adding rocm/cmake to the Environment 
docker/Dockerfile.amd:ENV PATH=$ROCM_HOME/bin:$ROCM_HOME/profiler/bin:$ROCM_HOME/opencl/bin:/opt/cmake/bin:$PATH \
docker/Dockerfile.amd:    LD_LIBRARY_PATH=$ROCM_HOME/lib:$ROCM_HOME/lib64:$ROCM_HOME/llvm/lib:$LD_LIBRARY_PATH \
docker/Dockerfile.amd:    C_INCLUDE_PATH=$ROCM_HOME/include:$C_INCLUDE_PATH \
docker/Dockerfile.amd:    CPLUS_INCLUDE_PATH=$ROCM_HOME/include:$CPLUS_INCLUDE_PATH \
docker/Dockerfile.amd:    CPATH=$ROCM_HOME/include:$CPATH \
docker/Dockerfile.amd:    INCLUDE=$ROCM_HOME/include:$INCLUDE
docker/Dockerfile.amd:        --with-rocm=$ROCM_HOME \
docker/Dockerfile.amd:        --without-cuda \
docker/Dockerfile.amd:# AOMP for GPUs (OpenMP offloading)
docker/Dockerfile.amd:ENV DEVITO_PLATFORM="amdgpuX"
docker/Dockerfile.amd:# HIPCC for GPUs (HIP)
docker/Dockerfile.amd:ENV DEVITO_PLATFORM="amdgpuX"
docker/Singularity.nvidia.def:# This Dockerfile contains the additional NVIDIA compilers, 
docker/Singularity.nvidia.def:# libraries, and plugins to enable OpenACC and NVIDIA GPU 
docker/Singularity.nvidia.def:#   singularity build --fakeroot devito.nvidia.sif docker/Singularity.nvidia.def
docker/Singularity.nvidia.def:#   singularity run --nv --writable-tmpfs devito.nvidia.sif
docker/Singularity.nvidia.def:./requirements-nvidia.txt /app/requirements-nvidia.txt
docker/Singularity.nvidia.def:export NVIDIA_VISIBLE_DEVICES=all
docker/Singularity.nvidia.def:export NVIDIA_DRIVER_CAPABILITIES=compute,utility
docker/Singularity.nvidia.def:export HPCSDK_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/2022
docker/Singularity.nvidia.def:export HPCSDK_CUPTI=/opt/nvidia/hpc_sdk/Linux_x86_64/2022/cuda/11.6/extras/CUPTI
docker/Singularity.nvidia.def:export CUDA_HOME=$HPCSDK_HOME/cuda
docker/Singularity.nvidia.def:export NVHPC_CUDA_HOME=$HPCSDK_HOME/cuda
docker/Singularity.nvidia.def:export CUDA_ROOT=$HPCSDK_HOME/cuda/bin
docker/Singularity.nvidia.def:export PATH=$HPCSDK_HOME/compilers/bin:$HPCSDK_HOME/cuda/bin:$HPCSDK_HOME/comm_libs/mpi/bin:${PATH}
docker/Singularity.nvidia.def:export LD_LIBRARY_PATH=$HPCSDK_HOME/cuda/lib:$HPCSDK_HOME/cuda/lib64:$HPCSDK_HOME/compilers/lib:$HPCSDK_HOME/math_libs/lib64:$HPCSDK_HOME/comm_libs/mpi/lib:$HPCSDK_CUPTI/lib64:bitcomp_DIR:${LD_LIBRARY_PATH}
docker/Singularity.nvidia.def:export NCCL_UCX_RNDV_THRESH=0
docker/Singularity.nvidia.def:export NCCL_UCX_RNDV_SCHEME=get_zcopy
docker/Singularity.nvidia.def:export NCCL_PLUGIN_P2P=ucx
docker/Singularity.nvidia.def:export UCX_TLS=rc_x,sm,shm,cuda_copy,gdr_copy,cuda_ipc
docker/Singularity.nvidia.def:#export UCX_TLS=sm,shm,cuda_copy,cuda_ipc
docker/Singularity.nvidia.def:## Environment Variables for OpenACC Builds
docker/Singularity.nvidia.def:export DEVITO_LANGUAGE="openacc"
docker/Singularity.nvidia.def:export DEVITO_PLATFORM=nvidiaX
docker/Singularity.nvidia.def:# Options: [unset, 1] For PGI openacc; Should only be set after a first execution of the benchmark
docker/Singularity.nvidia.def:echo 'deb [trusted=yes] https://developer.download.nvidia.com/hpc-sdk/ubuntu/amd64 /' > /etc/apt/sources.list.d/nvhpc.list && \
docker/Singularity.nvidia.def:  nvhpc-22-2 nvhpc-22-2-cuda-multi && \
docker/Singularity.nvidia.def:  http://developer.download.nvidia.com/compute/nvcomp/2.2/local_installers/nvcomp_exts_x86_64_ubuntu18.04-2.2.tar.gz && \
docker/Singularity.nvidia.def:export HPCSDK_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/2022
docker/Singularity.nvidia.def:export HPCSDK_CUPTI=/opt/nvidia/hpc_sdk/Linux_x86_64/2022/cuda/11.6/extras/CUPTI
docker/Singularity.nvidia.def:# Compiler, CUDA, and Library paths
docker/Singularity.nvidia.def:export CUDA_HOME=$HPCSDK_HOME/cuda
docker/Singularity.nvidia.def:export NVHPC_CUDA_HOME=$HPCSDK_HOME/cuda
docker/Singularity.nvidia.def:export CUDA_ROOT=$HPCSDK_HOME/cuda/bin
docker/Singularity.nvidia.def:export PATH=$HPCSDK_HOME/compilers/bin:$HPCSDK_HOME/cuda/bin:$HPCSDK_HOME/comm_libs/mpi/bin:${PATH}
docker/Singularity.nvidia.def:export LD_LIBRARY_PATH=$HPCSDK_HOME/cuda/lib:$HPCSDK_HOME/cuda/lib64:$HPCSDK_HOME/compilers/lib:$HPCSDK_HOME/math_libs/lib64:$HPCSDK_HOME/comm_libs/mpi/lib:$HPCSDK_CUPTI/lib64:${LD_LIBRARY_PATH}
docker/Singularity.nvidia.def:	/venv/bin/pip install --no-cache-dir -r /app/requirements-nvidia.txt && \
examples/README.md:* `performance`: Jupyter notebooks explaining the optimizations applied by Devito, the options available to steer the optimization process, how to run on GPUs, and much more.
examples/performance/README.md:    * `gpu-fit` (boolean, False): list of saved TimeFunctions that fit in the device memory
examples/performance/README.md:|                     |        CPU          |         GPU        |
examples/performance/README.md:|                     |        CPU          |         GPU        |
examples/performance/README.md:|                     |        CPU          |         GPU        |
examples/performance/README.md:|                     |        CPU          |         GPU        |
examples/performance/README.md:| gpu-fit             |        :x:          | :heavy_check_mark: |
examples/performance/README.md:|                     |        CPU          |         GPU        |
FAQ.md:This environment variable is mostly needed when running on GPUs, to ask Devito to generate code for a particular device (see for example this [tutorial](https://github.com/devitocodes/devito/blob/master/examples/gpu/01_diffusion_with_openmp_offloading.ipynb)). Can be also used to specify CPU architectures such as Intel's -- Haswell, Broadwell, SKL and KNL -- ARM, AMD, and Power. Often one can ignore this variable because Devito typically does a decent job at auto-detecting the underlying platform.
FAQ.md:Specify the generated code language. The default is `C`, which means sequential C. Use `openmp` to emit C+OpenMP or `openacc` for C+OpenACC.
FAQ.md:With a device PLATFORM (e.g., `nvidiaX`, `amdgpuX`, or `intelgpuX`), the compiler will generate OpenMP code for device offloading.
FAQ.md:When using OpenMP offloading, it is recommended to stick to the corresponding vendor compiler, so `ARCH=amdclang` for AMD, `ARCH={icc,icx,intel}` for Intel, and `ARCH=nvc` for NVidia.
FAQ.md:#### LANGUAGE=openacc
FAQ.md:Requires: `PLATFORM=nvidiaX` and `ARCH=nvc`.
FAQ.md:#### LANGUAGE=cuda
FAQ.md:Requires: `PLATFORM=nvidiaX` and `ARCH=cuda`.
FAQ.md:Requires: `PLATFORM=amdgpuX` and `ARCH=hip`.
FAQ.md:The code generated by devito is designed to run fast on CPU, GPU and clusters thereof. Broadly outlined, some of the mechanics for generating fast code are:
FAQ.md:### GPU
FAQ.md:* Longer pipelines, less travel to host (do more work on the GPU before communicating data between host and GPU)
FAQ.md:### Clusters of CPUs/GPUs
FAQ.md:* There are other tricks for the creation of an MFE. If your code by default uses MPI, or OpenMP, or OpenACC, or combinations thereof, but the bug appears even when running sequentially, then explicitly disable parallelism. Also try with disabling the performance optimizations applied by an Operator -- `Operator(..., opt='noop')`.
FAQ.md:* Did you try tuning the performance of the Devito `Operator`? E.g., on GPUs it is worth giving the `par-tile` option a go.
devito/core/gpu.py:    GPU_FIT = 'all-fallback'
devito/core/gpu.py:    Assuming all functions fit into the gpu memory.
devito/core/gpu.py:        # GPU parallelism
devito/core/gpu.py:        o['gpu-fit'] = cls._normalize_gpu_fit(oo, **kwargs)
devito/core/gpu.py:        o['gpu-create'] = as_tuple(oo.pop('gpu-create', ()))
devito/core/gpu.py:    def _normalize_gpu_fit(cls, oo, **kwargs):
devito/core/gpu.py:            gfit = as_tuple(oo.pop('gpu-fit'))
devito/core/gpu.py:                return as_tuple(cls.GPU_FIT)
devito/core/gpu.py:        # GPU parallelism
devito/core/gpu.py:        # GPU parallelism
devito/core/gpu.py:        callback = lambda f: not is_on_device(f, options['gpu-fit'])
devito/core/gpu.py:        if len(oo['gpu-create']):
devito/core/gpu.py:            raise InvalidOperator("Unsupported gpu-create option for omp operators")
devito/core/gpu.py:# OpenACC
devito/core/gpu.py:        oo['openacc'] = True
devito/core/gpu.py:        mapper['openacc'] = mapper['parallel']
devito/core/gpu.py:    _known_passes = DeviceCustomOperator._known_passes + ('openacc',)
devito/core/cpu.py:        oo.pop('gpu-fit', None)
devito/core/cpu.py:        oo.pop('gpu-create', None)
devito/core/cpu.py:    _known_passes_disabled = ('tasking', 'streaming', 'openacc')
devito/core/operator.py:        # Enforce pthreads if CPU-GPU orchestration requested
devito/core/__init__.py:from devito.core.gpu import (DeviceNoopOmpOperator, DeviceNoopAccOperator,
devito/core/__init__.py:operator_registry.add(DeviceCustomAccOperator, Device, 'custom', 'openacc')
devito/core/__init__.py:operator_registry.add(DeviceNoopAccOperator, Device, 'noop', 'openacc')
devito/core/__init__.py:operator_registry.add(DeviceAdvAccOperator, Device, 'advanced', 'openacc')
devito/core/__init__.py:operator_registry.add(DeviceFsgAccOperator, Device, 'advanced-fsg', 'openacc')
devito/tools/dtypes_lowering.py:# NOTE: the following is inspired by pyopencl.cltypes
devito/tools/dtypes_lowering.py:        CUDA or HIP.
devito/operator/registry.py:    _languages = ('C', 'openmp', 'openacc', 'cuda', 'hip', 'sycl')
devito/operator/operator.py:            # Same as above, but excluding the setup phase, e.g. the CPU-GPU
devito/operator/operator.py:            # data transfers in the case of a GPU run, mallocs, frees, etc.
devito/operator/operator.py:                if not is_on_device(i, self.options['gpu-fit']):
devito/types/basic.py:            * the device DRAM if platform=GPU
devito/types/basic.py:            * the device DRAM if platform=GPU
devito/types/basic.py:            * the host DRAM if platform=GPU
devito/types/grid.py:        # MultiSubDomains might break on GPUs
devito/builtins/initializers.py:    # Note: generally not enough parallelism to be performant on a gpu device
devito/builtins/initializers.py:    # TODO: Add openacc support for CPUs and set platform = 'cpu64'
devito/passes/clusters/blocking.py:        self.gpu_fit = options.get('gpu-fit', ())
devito/passes/clusters/blocking.py:        return (is_on_device(cluster.functions, self.gpu_fit),)
devito/passes/clusters/blocking.py:            if is_on_device(c.functions, self.gpu_fit):
devito/passes/iet/linearization.py:        # being perfectly legal OpenACC code. The workaround consists of
devito/passes/iet/languages/openacc.py:from devito.arch import AMDGPUX, NVIDIAX
devito/passes/iet/languages/openacc.py:                                  if (is_on_device(i, kwargs['gpu_fit']) and
devito/passes/iet/languages/openacc.py:        'name': 'OpenACC',
devito/passes/iet/languages/openacc.py:        'header': 'openacc.h',
devito/passes/iet/languages/openacc.py:        AMDGPUX: Macro('acc_device_radeon'),
devito/passes/iet/languages/openacc.py:        NVIDIAX: Macro('acc_device_nvidia'),
devito/passes/iet/languages/openacc.py:            body = self.DeviceIteration(gpu_fit=self.gpu_fit, tile=tile,
devito/passes/iet/languages/openacc.py:        OpenACC provides multiple mechanisms to this purpose, including the
devito/passes/iet/languages/openacc.py:            https://forums.developer.nvidia.com/t/acc-deviceptr-does-not-work-in-\
devito/passes/iet/languages/openacc.py:                openacc-code-dynamically-loaded-from-a-shared-library/211599
devito/passes/iet/languages/openacc.py:        Basically, the issue crops up when OpenACC code is part of a shared library
devito/passes/iet/languages/openacc.py:        OpenACC runtime library. That's our case, since our executable is Python.
devito/passes/iet/languages/openacc.py:        functions = [f for f in symbols if needs_transfer(f, self.gpu_fit)]
devito/passes/iet/languages/openmp.py:from devito.arch import AMDGPUX, NVIDIAX, INTELGPUX, PVC
devito/passes/iet/languages/openmp.py:        AMDGPUX: None,
devito/passes/iet/languages/openmp.py:        NVIDIAX: None,
devito/passes/iet/languages/openmp.py:        INTELGPUX: None,
devito/passes/iet/languages/targets.py:from devito.passes.iet.languages.openacc import (DeviceAccizer, DeviceAccDataManager,
devito/passes/iet/langbase.py:            1. Calling the init function (e.g., `acc_init(...)` for OpenACC)
devito/passes/iet/langbase.py:            # so that we have nranks == ngpus (as long as the user has launched
devito/passes/iet/langbase.py:            # number of GPUs per node)
devito/passes/iet/langbase.py:                # nonetheless to perform the rank-GPU assignment
devito/passes/iet/langbase.py:                ngpus, call_ngpus = self.lang._get_num_devices(self.platform)
devito/passes/iet/langbase.py:                osdd_else = self.lang['set-device']([rank % ngpus] + devicetype)
devito/passes/iet/langbase.py:                    List(body=[rank_decl, rank_init, call_ngpus, osdd_else]),
devito/passes/iet/langbase.py:        if any(not is_on_device(e.write, self.gpu_fit) for e in expressions):
devito/passes/iet/langbase.py:        hostfuncs = [f for f in functions if not is_on_device(f, self.gpu_fit)]
devito/passes/iet/parpragma.py:                 gpu_fit=None, **kwargs):
devito/passes/iet/parpragma.py:            reduction=reduction, schedule=schedule, tile=tile, gpu_fit=gpu_fit,
devito/passes/iet/parpragma.py:        self.gpu_fit = gpu_fit
devito/passes/iet/parpragma.py:        self.gpu_fit = options['gpu-fit']
devito/passes/iet/parpragma.py:            body = self.DeviceIteration(gpu_fit=self.gpu_fit,
devito/passes/iet/parpragma.py:        ngpus = Symbol(name='ngpus')
devito/passes/iet/parpragma.py:        return ngpus, cls.mapper['num-devices'](devicetype, ngpus)
devito/passes/iet/engine.py:        gpu_fit = (options or {}).get('gpu-fit', ())
devito/passes/iet/engine.py:                          if needs_transfer(w, gpu_fit)})
devito/passes/iet/engine.py:                          if needs_transfer(f, gpu_fit) and f not in wmovs})
devito/passes/iet/definitions.py:from devito.passes import is_gpu_create
devito/passes/iet/definitions.py:        self.gpu_fit = kwargs['options']['gpu-fit']
devito/passes/iet/definitions.py:        self.gpu_create = kwargs['options']['gpu-create']
devito/passes/iet/definitions.py:            if is_gpu_create(obj, self.gpu_create):
devito/passes/__init__.py:def is_on_device(obj, gpu_fit):
devito/passes/__init__.py:    gpu_fit : list of Function
devito/passes/__init__.py:        `gpu-fit` and is propagated down here through the various stages of lowering.
devito/passes/__init__.py:    if 'all-fallback' in gpu_fit and fsave:
devito/passes/__init__.py:        warning("TimeFunction %s assumed to fit the GPU memory" % fsave)
devito/passes/__init__.py:    return all(f in gpu_fit for f in fsave)
devito/passes/__init__.py:def needs_transfer(f, gpu_fit):
devito/passes/__init__.py:    gpu_fit : list of Function
devito/passes/__init__.py:        `gpu-fit` and is propagated down here through the various stages of lowering.
devito/passes/__init__.py:    return f._mem_mapped and not f.alias and is_on_device(f, gpu_fit)
devito/passes/__init__.py:def is_gpu_create(obj, gpu_create):
devito/passes/__init__.py:    gpu-create : list of Function
devito/passes/__init__.py:    return all(f in gpu_create for f in functions)
devito/arch/archinfo.py:__all__ = ['platform_registry', 'get_cpu_info', 'get_gpu_info', 'get_nvidia_cc',
devito/arch/archinfo.py:           'get_cuda_path', 'get_hip_path', 'check_cuda_runtime', 'get_m1_llvm_path',
devito/arch/archinfo.py:           'Device', 'NvidiaDevice', 'AmdDevice', 'IntelDevice',
devito/arch/archinfo.py:           'ANYCPU', 'ANYGPU',
devito/arch/archinfo.py:           # Generic GPUs
devito/arch/archinfo.py:           'AMDGPUX', 'NVIDIAX', 'INTELGPUX',
devito/arch/archinfo.py:           # Intel GPUs
devito/arch/archinfo.py:           'PVC', 'INTELGPUMAX', 'MAX1100', 'MAX1550']
devito/arch/archinfo.py:def get_gpu_info():
devito/arch/archinfo.py:    """Attempt GPU info autodetection."""
devito/arch/archinfo.py:    # Filter out virtual GPUs from a list of GPU dictionaries
devito/arch/archinfo.py:    def filter_real_gpus(gpus):
devito/arch/archinfo.py:        def is_real_gpu(gpu):
devito/arch/archinfo.py:            return 'virtual' not in gpu['product'].lower()
devito/arch/archinfo.py:        return list(filter(is_real_gpu, gpus))
devito/arch/archinfo.py:    def homogenise_gpus(gpu_infos):
devito/arch/archinfo.py:        Run homogeneity checks on a list of GPUs, return GPU with count if
devito/arch/archinfo.py:        if gpu_infos == []:
devito/arch/archinfo.py:        for gpu_info in gpu_infos:
devito/arch/archinfo.py:            gpu_info.pop('physicalid', None)
devito/arch/archinfo.py:        if all_equal(gpu_infos):
devito/arch/archinfo.py:            gpu_infos[0]['ncards'] = len(gpu_infos)
devito/arch/archinfo.py:            return gpu_infos[0]
devito/arch/archinfo.py:    # Parse textual gpu info into a dict
devito/arch/archinfo.py:    # *** First try: `nvidia-smi`, clearly only works with NVidia cards
devito/arch/archinfo.py:        gpu_infos = []
devito/arch/archinfo.py:        info_cmd = ['nvidia-smi', '-L']
devito/arch/archinfo.py:            gpu_info = {}
devito/arch/archinfo.py:            if 'GPU' in line:
devito/arch/archinfo.py:                gpu_info = {}
devito/arch/archinfo.py:                match = re.match(r'GPU *[0-9]*\: ([\w]*) (.*) \(', line)
devito/arch/archinfo.py:                        gpu_info['architecture'] = 'unspecified'
devito/arch/archinfo.py:                        gpu_info['architecture'] = match.group(1)
devito/arch/archinfo.py:                        gpu_info['product'] = 'unspecified'
devito/arch/archinfo.py:                        gpu_info['product'] = match.group(2)
devito/arch/archinfo.py:                    gpu_info['vendor'] = 'NVIDIA'
devito/arch/archinfo.py:                    gpu_infos.append(gpu_info)
devito/arch/archinfo.py:        gpu_info = homogenise_gpus(gpu_infos)
devito/arch/archinfo.py:                    info_cmd = ['nvidia-smi', '--query-gpu=memory.%s' % i, '--format=csv']
devito/arch/archinfo.py:                        # We shouldn't really end up here, unless nvidia-smi changes
devito/arch/archinfo.py:            gpu_info['mem.%s' % i] = make_cbk(i)
devito/arch/archinfo.py:        return gpu_info
devito/arch/archinfo.py:    # *** Second try: `rocm-smi`, clearly only works with AMD cards
devito/arch/archinfo.py:        gpu_infos = {}
devito/arch/archinfo.py:        # Base gpu info
devito/arch/archinfo.py:        info_cmd = ['rocm-smi', '--showproductname']
devito/arch/archinfo.py:            if 'GPU' in line:
devito/arch/archinfo.py:                pattern = r'GPU\[(\d+)\].*?Card series:\s*(.*?)\s*$'
devito/arch/archinfo.py:                    gpu_infos.setdefault(gid, dict())
devito/arch/archinfo.py:                    gpu_infos[gid]['physicalid'] = gid
devito/arch/archinfo.py:                    gpu_infos[gid]['product'] = match1.group(2)
devito/arch/archinfo.py:                pattern = r'GPU\[(\d+)\].*?Card model:\s*(.*?)\s*$'
devito/arch/archinfo.py:                    gpu_infos.setdefault(gid, dict())
devito/arch/archinfo.py:                    gpu_infos[gid]['physicalid'] = match2.group(1)
devito/arch/archinfo.py:                    gpu_infos[gid]['model'] = match2.group(2)
devito/arch/archinfo.py:        gpu_info = homogenise_gpus(list(gpu_infos.values()))
devito/arch/archinfo.py:        info_cmd = ['rocm-smi', '--showmeminfo', 'vram', '--json']
devito/arch/archinfo.py:                        # We shouldn't really end up here, unless nvidia-smi changes
devito/arch/archinfo.py:            gpu_info['mem.%s' % i] = make_cbk(i)
devito/arch/archinfo.py:        gpu_infos['architecture'] = 'AMD'
devito/arch/archinfo.py:        return gpu_info
devito/arch/archinfo.py:        def lshw_single_gpu_info(raw_info):
devito/arch/archinfo.py:                def extract_gpu_info(keyword):
devito/arch/archinfo.py:                gpu_info = {}
devito/arch/archinfo.py:                gpu_info['product'] = extract_gpu_info('product')
devito/arch/archinfo.py:                gpu_info['architecture'] = parse_product_arch()
devito/arch/archinfo.py:                gpu_info['vendor'] = extract_gpu_info('vendor')
devito/arch/archinfo.py:                gpu_info['physicalid'] = extract_gpu_info('physical id')
devito/arch/archinfo.py:                return gpu_info
devito/arch/archinfo.py:        gpu_infos = [lshw_single_gpu_info(device) for device in devices]
devito/arch/archinfo.py:        gpu_infos = filter_real_gpus(gpu_infos)
devito/arch/archinfo.py:        return homogenise_gpus(gpu_infos)
devito/arch/archinfo.py:        gpu_infos = []
devito/arch/archinfo.py:                gpu_info = {}
devito/arch/archinfo.py:                #   0001:00:00.0 3D controller: NVIDIA Corp... [Tesla K80] (rev a1)
devito/arch/archinfo.py:                    gpu_info['product'] = name_match.group(1)
devito/arch/archinfo.py:                        gpu_info['architecture'] = arch_match.group(1)
devito/arch/archinfo.py:                        gpu_info['architecture'] = 'unspecified'
devito/arch/archinfo.py:                gpu_infos.append(gpu_info)
devito/arch/archinfo.py:        gpu_infos = filter_real_gpus(gpu_infos)
devito/arch/archinfo.py:        return homogenise_gpus(gpu_infos)
devito/arch/archinfo.py:def get_nvidia_cc():
devito/arch/archinfo.py:    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
devito/arch/archinfo.py:            cuda = ctypes.CDLL(libname)
devito/arch/archinfo.py:    if cuda.cuInit(0) != 0:
devito/arch/archinfo.py:    elif (cuda.cuDeviceComputeCapability(ctypes.byref(cc_major),
devito/arch/archinfo.py:def get_cuda_path():
devito/arch/archinfo.py:    for i in ['CUDA_HOME', 'CUDA_ROOT']:
devito/arch/archinfo.py:        cuda_home = os.environ.get(i)
devito/arch/archinfo.py:        if cuda_home:
devito/arch/archinfo.py:            return cuda_home
devito/arch/archinfo.py:        if re.match('.*/nvidia/hpc_sdk/.*/compilers/lib', i):
devito/arch/archinfo.py:            cuda_home = os.path.join(os.path.dirname(os.path.dirname(i)), 'cuda')
devito/arch/archinfo.py:            if os.path.exists(cuda_home):
devito/arch/archinfo.py:                return cuda_home
devito/arch/archinfo.py:def check_cuda_runtime():
devito/arch/archinfo.py:    libnames = ('libcudart.so', 'libcudart.dylib', 'cudart.dll')
devito/arch/archinfo.py:            cuda = ctypes.CDLL(libname)
devito/arch/archinfo.py:        warning("Unable to check compatibility of NVidia driver and runtime")
devito/arch/archinfo.py:    if cuda.cudaDriverGetVersion(ctypes.byref(driver_version)) == 0 and \
devito/arch/archinfo.py:       cuda.cudaRuntimeGetVersion(ctypes.byref(runtime_version)) == 0:
devito/arch/archinfo.py:            warning("The NVidia driver (v%d) on this system may not be compatible "
devito/arch/archinfo.py:                    "with the CUDA runtime (v%d)" % (driver_version, runtime_version))
devito/arch/archinfo.py:        warning("Unable to check compatibility of NVidia driver and runtime")
devito/arch/archinfo.py:        info = get_gpu_info()
devito/arch/archinfo.py:        info = get_gpu_info()
devito/arch/archinfo.py:        info = get_gpu_info()
devito/arch/archinfo.py:class NvidiaDevice(Device):
devito/arch/archinfo.py:        info = get_gpu_info()
devito/arch/archinfo.py:        # The AMD's AOMP compiler toolkit ships the `mygpu` program to (quoting
devito/arch/archinfo.py:        #     Print out the real gpu name for the current system
devito/arch/archinfo.py:        #     or for the codename specified with -getgpuname option.
devito/arch/archinfo.py:        #     mygpu will only print values accepted by cuda clang in
devito/arch/archinfo.py:        #     the clang argument --cuda-gpu-arch.
devito/arch/archinfo.py:                p1 = Popen(['mygpu', '-d', fallback], stdout=PIPE, stderr=PIPE)
devito/arch/archinfo.py:ANYGPU = Cpu64('gpu')
devito/arch/archinfo.py:NVIDIAX = NvidiaDevice('nvidiaX')
devito/arch/archinfo.py:AMDGPUX = AmdDevice('amdgpuX')
devito/arch/archinfo.py:INTELGPUX = IntelDevice('intelgpuX')
devito/arch/archinfo.py:PVC = IntelDevice('pvc')  # Legacy codename for MAX GPUs
devito/arch/archinfo.py:INTELGPUMAX = IntelDevice('intelgpuMAX')
devito/arch/compiler.py:from devito.arch import (AMDGPUX, Cpu64, AppleArm, NVIDIAX, POWER8, POWER9, Graviton,
devito/arch/compiler.py:                         IntelDevice, get_nvidia_cc, check_cuda_runtime,
devito/arch/compiler.py:        if platform is NVIDIAX:
devito/arch/compiler.py:                cc = get_nvidia_cc()
devito/arch/compiler.py:                self.ldflags += ['-fopenmp', '-fopenmp-targets=nvptx64-nvidia-cuda']
devito/arch/compiler.py:        elif platform is AMDGPUX:
devito/arch/compiler.py:    """AMD's fork of Clang for OpenMP offloading on both AMD and NVidia cards."""
devito/arch/compiler.py:        if platform is NVIDIAX:
devito/arch/compiler.py:        elif platform is AMDGPUX:
devito/arch/compiler.py:        if platform is NVIDIAX:
devito/arch/compiler.py:                self.cflags.append('-gpu=mem:separate:pinnedalloc')
devito/arch/compiler.py:                self.cflags.append('-gpu=pinned')
devito/arch/compiler.py:            if language == 'openacc':
devito/arch/compiler.py:                self.cflags.extend(['-mp', '-acc:gpu'])
devito/arch/compiler.py:                self.cflags.extend(['-mp=gpu'])
devito/arch/compiler.py:        # Default PGI compile for a target is GPU and single threaded host.
devito/arch/compiler.py:class NvidiaCompiler(PGICompiler):
devito/arch/compiler.py:class CudaCompiler(Compiler):
devito/arch/compiler.py:        cc = get_nvidia_cc()
devito/arch/compiler.py:        self.cflags.extend(['-Xcudafe', '--display_error_number',
devito/arch/compiler.py:        # to be executed once to warn the user in case there's a CUDA/driver
devito/arch/compiler.py:        # garbage, since the CUDA kernel behaviour would be undefined
devito/arch/compiler.py:        check_cuda_runtime()
devito/arch/compiler.py:            if platform is NVIDIAX:
devito/arch/compiler.py:                self.cflags.append('-fopenmp-targets=nvptx64-cuda')
devito/arch/compiler.py:        # to enable GPU-aware MPI (that is, passing device pointers to MPI calls)
devito/arch/compiler.py:        elif platform is NVIDIAX:
devito/arch/compiler.py:            self.cflags.append('-fsycl-targets=nvptx64-cuda')
devito/arch/compiler.py:        elif platform is NVIDIAX:
devito/arch/compiler.py:            if language == 'cuda':
devito/arch/compiler.py:                _base = CudaCompiler
devito/arch/compiler.py:                _base = NvidiaCompiler
devito/arch/compiler.py:        elif platform is AMDGPUX:
devito/arch/compiler.py:        # will set CXX to nvc++ breaking  the cuda backend
devito/arch/compiler.py:    'nvc': NvidiaCompiler,
devito/arch/compiler.py:    'nvc++': NvidiaCompiler,
devito/arch/compiler.py:    'nvidia': NvidiaCompiler,
devito/arch/compiler.py:    'cuda': CudaCompiler,
conftest.py:                                  NvidiaCompiler)
conftest.py:    accepted.update({'device', 'device-C', 'device-openmp', 'device-openacc',
conftest.py:        # Skip if won't run on GPUs
conftest.py:        # Skip if won't run on a specific GPU backend
conftest.py:        # Skip if must run on GPUs but not currently on a GPU
conftest.py:           isinstance(configuration['compiler'], NvidiaCompiler) and \
conftest.py:# regarding GPU spatial and/or temporal blocking.

```
