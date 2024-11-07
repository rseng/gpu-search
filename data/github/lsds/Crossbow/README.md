# https://github.com/lsds/Crossbow

```console
README.md:# Crossbow: A Multi-GPU Deep Learning System for Training with Small Batch Sizes
README.md:**Crossbow** is a multi-GPU system for training deep learning models that
README.md:  while scaling to multiple GPUs. 
README.md:**Crossbow** utilises modern GPUs better than other systems by training multiple  _model replicas_ on the same GPU. When the batch size is sufficiently small to leave GPU resources unused, **Crossbow** trains a second model replica, a third, etc., as long as training throughput increases.
README.md:To synchronise many model replicas, **Crossbow** uses _synchronous model averaging_ to adjust the trajectory of each individual replica based on the average of all. With model averaging, the batch size does not increase linearly with the number of model replicas, as it would with synchronous SGD. This yields better statistical efficiency without cumbersome hyper-parameter tuning when trying to scale training to a larger number of GPUs.
README.md:**Crossbow** requires NVIDIA's [CUDA](https://developer.nvidia.com/cuda-toolkit) toolkit, the [cuDDN](https://developer.nvidia.com/cudnn) library and the [NCCL](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html) library (currently using versions 8.0, 6.0, and 2.1.15, respectively). After successful installation, make sure that:
README.md:* `CUDA_HOME` is set (the default location is `/usr/local/cuda`)
README.md:* `NCCL_HOME` is set
README.md:* `PATH` includes `$CUDA_HOME/bin` and
README.md:* `LD_LIBRARY_PATH` includes `$CUDA_HOME/lib64` and `$NCCL_HOME/lib`
README.md:**Crossbow** uses page-locked memory regions to speed up data transfers from CPU to GPU and vice versa. The amount of memory locked by the system usually exceeds the default OS limit. Edit `/etc/security/limits.conf` and append the following lines to the end of the file:
tools/measurements/parse-measurements.py:class GPU(object):
tools/measurements/parse-measurements.py:        self.gpu = GPU()
tools/measurements/parse-measurements.py:        # 2: GPU temperature    (C)
tools/measurements/parse-measurements.py:        # 3: GPU utilization    (%), 
tools/measurements/parse-measurements.py:        # GPU
tools/measurements/parse-measurements.py:        m.gpu.temperature = float(s[2])
tools/measurements/parse-measurements.py:        m.gpu.utilisation = float(s[3])
tools/measurements/parse-measurements.py:    print("%d GPU devices" % len(keys))
tools/measurements/parse-measurements.py:        sys.stderr.write("error: different number of measurements per GPU")
tools/measurements/parse-measurements.py:    print("%d measurements per GPU" % K[0])
tools/measurements/parse-measurements.py:    agggpuutil = []
tools/measurements/parse-measurements.py:        gpuutilvalues = []
tools/measurements/parse-measurements.py:            gpuutilvalues.append(m.gpu.utilisation)
tools/measurements/parse-measurements.py:        if np.mean(gpuutilvalues) < 1:
tools/measurements/parse-measurements.py:        gpuutilvalues = gpuutilvalues[20:]
tools/measurements/parse-measurements.py:        gpuutilvalues = gpuutilvalues[:-100]
tools/measurements/parse-measurements.py:        print(gpuutilvalues)
tools/measurements/parse-measurements.py:        agggpuutil.extend(gpuutilvalues)
tools/measurements/parse-measurements.py:        print("GPU utilization stats for GPU " + key)
tools/measurements/parse-measurements.py:        printStats(key, gpuutilvalues)
tools/measurements/parse-measurements.py:        print("Memory utilization stats for GPU " + key)
tools/measurements/parse-measurements.py:    printStats("agg_gpu", agggpuutil)
tools/measurements/gpu-measurements.sh:# GPU query metrics
tools/measurements/gpu-measurements.sh:# gpu_serial:         The serial number of each GPU, a globally unique 
tools/measurements/gpu-measurements.sh:# utilization.gpu:    Percentage of time over the past sampling period 
tools/measurements/gpu-measurements.sh:# gpu_name
tools/measurements/gpu-measurements.sh:# gpu_uuid
tools/measurements/gpu-measurements.sh:# temperature_gpu
tools/measurements/gpu-measurements.sh:# gpu_bus_id
tools/measurements/gpu-measurements.sh:METRICS="$METRICS,gpu_serial"
tools/measurements/gpu-measurements.sh:METRICS="$METRICS,temperature.gpu"
tools/measurements/gpu-measurements.sh:METRICS="$METRICS,utilization.gpu"
tools/measurements/gpu-measurements.sh:nvidia-smi --query-gpu=${METRICS} -l${UNIT} ${DURATION} --format=${FORMAT} >>${FILENAME} 2>&1
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:# Crossbow Docker image based on NVIDIA docker CUDA 9.2
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:FROM nvidia/cuda:9.2-base-ubuntu${UBUNTU_VERSION} as base
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:        cuda-command-line-tools-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:        cuda-cublas-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:        cuda-cufft-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:        cuda-curand-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:        cuda-cusolver-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:        cuda-cusparse-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:        libcudnn7=7.1.4.18-1+cuda9.2 \ 
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:        libnccl2 \ 
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:        libnccl-dev \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:        libcudnn7-dev=7.1.4.18-1+cuda9.2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:        cuda-libraries-dev-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:        cuda-nvml-dev-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:        cuda-minimal-build-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:        cuda-command-line-tools-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:ENV CUDA_HOME /usr/local/cuda
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2.Dockerfile:    && cd clib-multigpu \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2-modelarts.Dockerfile:# CrossBow Docker image based on Huawei ModelArts CUDA 9.2
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2-modelarts.Dockerfile:FROM swr.cn-north-1.myhuaweicloud.com/eiwizard/custom-gpu-cuda92-base:1.0 as base
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2-modelarts.Dockerfile:        cuda-command-line-tools-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2-modelarts.Dockerfile:        cuda-cublas-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2-modelarts.Dockerfile:        cuda-cufft-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2-modelarts.Dockerfile:        cuda-curand-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2-modelarts.Dockerfile:        cuda-cusolver-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2-modelarts.Dockerfile:        cuda-cusparse-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2-modelarts.Dockerfile:        libnccl2 \ 
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2-modelarts.Dockerfile:        libnccl-dev \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2-modelarts.Dockerfile:        cuda-libraries-dev-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2-modelarts.Dockerfile:        cuda-nvml-dev-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2-modelarts.Dockerfile:        cuda-minimal-build-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2-modelarts.Dockerfile:        cuda-command-line-tools-9-2 \
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2-modelarts.Dockerfile:ENV CUDA_HOME /usr/local/cuda
tools/dockerfiles/dockerfiles/crossbow-cuda-9.2-modelarts.Dockerfile:    && cd clib-multigpu \
tools/dockerfiles/README.md:The following command builds a custom image for Crossbow based on CUDA 9.2.
tools/dockerfiles/README.md:$ docker build -f ./dockerfiles/crossbow-cuda-9.2.Dockerfile -t crossbow:latest .
tools/dockerfiles/README.md:$ docker run --runtime=nvidia -u root:root --ulimit memlock=1073741824:1073741824 -it crossbow:latest /crossbow/scripts/benchmarks/lenet.sh
scripts/reset.sh:rm -f "$CROSSBOW_HOME"/clib-multigpu/Makefile*
scripts/reset.sh:rm -f "$CROSSBOW_HOME"/clib-multigpu/build.log
scripts/reset.sh:rm -f "$CROSSBOW_HOME"/clib-multigpu/*.so
scripts/reset.sh:rm -f "$CROSSBOW_HOME"/clib-multigpu/*.o
scripts/reset.sh:# Delete CUDA files
scripts/prepare-software.sh:# Check libraries: Is CUDA_HOME set?
scripts/run.sh:                cuda-memcheck java $OPTS -cp $JCP $CLS $ARGS
scripts/benchmarks/resnet-32.sh:# Enable GPU utilisation measurements
scripts/benchmarks/resnet-32.sh:MEASUREMENTSCRIPT="$CROSSBOW_HOME/tools/measurements/gpu-measurements.sh"
scripts/benchmarks/resnet-32.sh:numgpus=2
scripts/benchmarks/resnet-32.sh:devices=`crossbowCreateDeviceList ${numgpus}`
scripts/benchmarks/resnet-32.sh:echo "[INFO] Running on $numgpus devices (${devices})"
scripts/benchmarks/resnet-32.sh:echo "[INFO] $numreplicas replicas per GPU"
scripts/benchmarks/resnet-32.sh:wpc=$(($numreplicas * $numgpus * $wpcscale))
scripts/benchmarks/resnet-32.sh:    $MEASUREMENTSCRIPT "resnet-32-b-${batchsize}-g-${numgpus}-m-${numreplicas}" &
scripts/benchmarks/resnet-32.sh:    --gpu true \
scripts/benchmarks/resnet-32.sh:    --gpu-devices ${devices} \
scripts/benchmarks/resnet-32.sh:    --number-of-gpu-models ${numreplicas} \
scripts/benchmarks/resnet-32.sh:    --number-of-gpu-streams ${numreplicas} \
scripts/benchmarks/resnet-32.sh:    # Stop GPU measurements script
scripts/benchmarks/resnet-32.sh:    echo "Stop GPU measurements"
scripts/benchmarks/vgg.sh:numgpus=1
scripts/benchmarks/vgg.sh:devices=`crossbowCreateDeviceList ${numgpus}`
scripts/benchmarks/vgg.sh:echo "[INFO] Running on $numgpus devices (${devices})"
scripts/benchmarks/vgg.sh:echo "[INFO] $numreplicas replicas per GPU"
scripts/benchmarks/vgg.sh:wpc=$(($numreplicas * $numgpus * $wpcscale))
scripts/benchmarks/vgg.sh:    --gpu true \
scripts/benchmarks/vgg.sh:    --gpu-devices ${devices} \
scripts/benchmarks/vgg.sh:    --number-of-gpu-models ${numreplicas} \
scripts/benchmarks/vgg.sh:    --number-of-gpu-streams ${numreplicas} \
scripts/benchmarks/lenet.sh:numgpus=1
scripts/benchmarks/lenet.sh:devices=`crossbowCreateDeviceList ${numgpus}`
scripts/benchmarks/lenet.sh:echo "[INFO] Running on $numgpus devices (${devices})"
scripts/benchmarks/lenet.sh:echo "[INFO] $numreplicas replicas per GPU"
scripts/benchmarks/lenet.sh:wpc=$(($numreplicas * $numgpus * $wpcscale))
scripts/benchmarks/lenet.sh:    --gpu true \
scripts/benchmarks/lenet.sh:    --gpu-devices ${devices} \
scripts/benchmarks/lenet.sh:    --number-of-gpu-models ${numreplicas} \
scripts/benchmarks/lenet.sh:    --number-of-gpu-streams ${numreplicas} \
scripts/benchmarks/resnet-50.sh:numgpus=8
scripts/benchmarks/resnet-50.sh:devices=`crossbowCreateDeviceList ${numgpus}`
scripts/benchmarks/resnet-50.sh:echo "[INFO] Running on $numgpus devices (${devices})"
scripts/benchmarks/resnet-50.sh:echo "[INFO] $numreplicas replicas per GPU"
scripts/benchmarks/resnet-50.sh:wpc=$(($numreplicas * $numgpus * $wpcscale))
scripts/benchmarks/resnet-50.sh:NCCL_DEBUG=WARN java $OPTS -cp $JCP $CLASS \
scripts/benchmarks/resnet-50.sh:    --gpu true \
scripts/benchmarks/resnet-50.sh:    --gpu-devices ${devices} \
scripts/benchmarks/resnet-50.sh:    --number-of-gpu-models ${numreplicas} \
scripts/benchmarks/resnet-50.sh:    --number-of-gpu-streams ${numreplicas} \
scripts/benchmarks/resnet-101.sh:# Enable GPU utilisation measurements
scripts/benchmarks/resnet-101.sh:MEASUREMENTSCRIPT="$CROSSBOW_HOME/tools/measurements/gpu-measurements.sh"
scripts/benchmarks/resnet-101.sh:numgpus=8
scripts/benchmarks/resnet-101.sh:devices=`crossbowCreateDeviceList ${numgpus}`
scripts/benchmarks/resnet-101.sh:echo "[INFO] Running on $numgpus devices (${devices})"
scripts/benchmarks/resnet-101.sh:echo "[INFO] $numreplicas replicas per GPU"
scripts/benchmarks/resnet-101.sh:wpc=$(($numreplicas * $numgpus * $wpcscale))
scripts/benchmarks/resnet-101.sh:resultfile="resnet-101-b-${batchsize}-g-${numgpus}-m-${numreplicas}.out"
scripts/benchmarks/resnet-101.sh:    $MEASUREMENTSCRIPT "resnet-101-b-${batchsize}-g-${numgpus}-m-${numreplicas}.csv" &
scripts/benchmarks/resnet-101.sh:NCCL_DEBUG=WARN java $OPTS -cp $JCP $CLASS \
scripts/benchmarks/resnet-101.sh:    --gpu true \
scripts/benchmarks/resnet-101.sh:    --gpu-devices ${devices} \
scripts/benchmarks/resnet-101.sh:    --number-of-gpu-models ${numreplicas} \
scripts/benchmarks/resnet-101.sh:    --number-of-gpu-streams ${numreplicas} \
scripts/benchmarks/resnet-101.sh:    # Stop GPU measurements script
scripts/benchmarks/resnet-101.sh:    echo "Stop GPU measurements"
scripts/benchmarks/resnet-101.sh:        killall "nvidia-smi" >/dev/null 2>&1
scripts/crossbow.conf:CLIB_DIR="$CROSSBOW_HOME/clib-multigpu"
scripts/crossbow.conf:# Execution mode: either "cpu", "gpu" or "hybrid" (default is "cpu")
scripts/simple-run.sh:	cuda-memcheck java $OPTS -cp $JCP $CLASS $@
scripts/build.sh:GPULIB="$CLIB_DIR"/libGPU.so
scripts/build.sh:for lib in $CPULIB $GPULIB $BLASLIB $RNGLIB $DATALIB
clib-multigpu/datasetfile.h:#include <cuda.h>
clib-multigpu/datasetfile.h:#include <cuda_runtime.h>
clib-multigpu/debug.h:#include <cuda_runtime.h>
clib-multigpu/debug.h:#include <nccl.h>
clib-multigpu/debug.h:#undef GPU_VERBOSE
clib-multigpu/debug.h:/* #define GPU_VERBOSE */
clib-multigpu/debug.h:#define CUDANN_NOOP
clib-multigpu/debug.h:#define CUDART_NOOP
clib-multigpu/debug.h:#ifdef GPU_VERBOSE
clib-multigpu/debug.h:/* CUDA error handling */
clib-multigpu/debug.h:static const char *mycudaGetErrorEnum(cublasStatus_t error) {
clib-multigpu/debug.h:#define checkCudaErrors(x) do { if (x != cudaSuccess) { fprintf(stderr, "cuda error: %s (in %s, %s:%d)\n", cudaGetErrorString(x), __func__, __FILE__, __LINE__); exit (1); } } while (0)
clib-multigpu/debug.h:#define checkCublasStatus(x) do { if (x != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublas error: %s (in %s, %s:%d)\n", mycudaGetErrorEnum(x), __func__, __FILE__, __LINE__); exit (1); } } while (0)
clib-multigpu/debug.h:#define checkNcclErrors(x) do { if (x != ncclSuccess) { fprintf(stderr, "nccl error: %s (in %s, %s:%d)\n", ncclGetErrorString(x), __func__, __FILE__, __LINE__); exit (1); } } while (0)
clib-multigpu/debug.h:#endif /* __GPU_DEBUG_H_ */
clib-multigpu/stream.c:	checkCudaErrors(cudaSetDevice(p->deviceId));
clib-multigpu/stream.c:	p->stream = crossbowMalloc (p->branches * sizeof(cudaStream_t));
clib-multigpu/stream.c:		checkCudaErrors(cudaStreamCreateWithFlags(&(p->stream[i]), cudaStreamNonBlocking));
clib-multigpu/stream.c:	checkCudaErrors(cudaEventCreateWithFlags(&p->event, cudaEventBlockingSync));
clib-multigpu/stream.c:	checkCudaErrors(cudaEventCreateWithFlags(&p->start, cudaEventBlockingSync));
clib-multigpu/stream.c:	checkCudaErrors(cudaEventCreateWithFlags(&p->event, cudaEventDefault));
clib-multigpu/stream.c:	checkCudaErrors(cudaEventCreateWithFlags(&p->start, cudaEventDefault));
clib-multigpu/stream.c:	checkCudaErrors(cudaEventCreateWithFlags(&p->event, cudaEventBlockingSync | cudaEventDisableTiming));
clib-multigpu/stream.c:	checkCudaErrors(cudaEventCreateWithFlags(&p->event, cudaEventDefault | cudaEventDisableTiming));
clib-multigpu/stream.c:			checkCudaErrors(cudaMemcpyAsync(last, gradient, length, cudaMemcpyDeviceToDevice, p->stream));
clib-multigpu/stream.c:			checkCudaErrors(cudaEventRecord (p->model->client[p->op->peer->id], p->stream));
clib-multigpu/stream.c:			checkCudaErrors(cudaStreamWaitEvent(p->modelSynchronisationStream, p->model->client[p->op->peer->id], 0));
clib-multigpu/stream.c:			checkCudaErrors(cudaEventRecord(p->model->server[p->op->peer->id], p->modelSynchronisationStream));
clib-multigpu/stream.c:			checkCudaErrors(cudaEventRecord (p->model->client[p->op->peer->id], p->stream));
clib-multigpu/stream.c:			checkCudaErrors(cudaStreamWaitEvent(p->modelSynchronisationStream, p->model->client[p->op->peer->id], 0));
clib-multigpu/stream.c:			checkCudaErrors(cudaEventRecord(p->model->server[p->op->peer->id], p->modelSynchronisationStream));
clib-multigpu/stream.c:	checkCudaErrors(cudaEventRecord (p->model->client[p->op->peer->id], p->stream));
clib-multigpu/stream.c:			checkCudaErrors(cudaMemcpyAsync(last, gradient, length, cudaMemcpyDeviceToDevice, p->stream));
clib-multigpu/stream.c:		checkCudaErrors(cudaStreamDestroy(p->stream[i]));
clib-multigpu/stream.c:	crossbowFree (p->stream, (p->branches * sizeof(cudaStream_t)));
clib-multigpu/stream.c:	checkCudaErrors(cudaEventDestroy(p->event));
clib-multigpu/dataset.c:			p->gpu = filemanager[phase][i]->gpu;
clib-multigpu/dataset.c:	(JNIEnv *env, jobject obj, jint phase, jint parts, jboolean gpu, jintArray tasksize) {
clib-multigpu/dataset.c:		filemanager[phase][i] = crossbowDatasetFileManagerCreate (parts, (gpu == JNI_TRUE) ? 1 : 0, argv[i]);
clib-multigpu/dataset.c:#ifdef GPU_VERBOSE
clib-multigpu/model.c:	/* Redirect all CUDA calls to the correct device */
clib-multigpu/model.c:	checkCudaErrors (cudaSetDevice(p->dev));
clib-multigpu/model.c:	/* The model's gradient is allocated only on GPU memory. */
clib-multigpu/model.c:	checkCudaErrors(cudaMemset (p->gradient->dev, 0, p->bytes));
clib-multigpu/model.c:    checkCudaErrors(cudaMemset (p->hist->dev, 0, p->bytes));
clib-multigpu/model.c:	checkCudaErrors(cudaMemset (p->tmp1->dev, 0, p->bytes));
clib-multigpu/model.c:	/* GPU events to synchronise with the parameter server */
clib-multigpu/model.c:	p->client = crossbowMalloc (p->ops * sizeof(cudaEvent_t));
clib-multigpu/model.c:	p->server = crossbowMalloc (p->ops * sizeof(cudaEvent_t));
clib-multigpu/model.c:		checkCudaErrors(cudaEventCreateWithFlags(&(p->client[i]),  cudaEventBlockingSync | cudaEventDisableTiming));
clib-multigpu/model.c:		checkCudaErrors(cudaEventCreateWithFlags(&(p->server[i]),  cudaEventBlockingSync | cudaEventDisableTiming));
clib-multigpu/model.c:		checkCudaErrors(cudaEventCreateWithFlags(&(p->client[i]),  cudaEventDisableTiming));
clib-multigpu/model.c:		checkCudaErrors(cudaEventCreateWithFlags(&(p->server[i]),  cudaEventDisableTiming));
clib-multigpu/model.c:	checkCudaErrors(cudaEventCreateWithFlags(&(p->client),  cudaEventBlockingSync | cudaEventDisableTiming));
clib-multigpu/model.c:	checkCudaErrors(cudaEventCreateWithFlags(&(p->server),  cudaEventBlockingSync | cudaEventDisableTiming));
clib-multigpu/model.c:	checkCudaErrors(cudaEventCreateWithFlags(&(p->client), cudaEventDisableTiming));
clib-multigpu/model.c:	checkCudaErrors(cudaEventCreateWithFlags(&(p->server), cudaEventDisableTiming));
clib-multigpu/model.c:	/* checkCudaErrors(cudaEventCreateWithFlags(&(p->updated), cudaEventBlockingSync | cudaEventDisableTiming)); */
clib-multigpu/model.c:	checkCudaErrors(cudaEventCreateWithFlags(&(p->updated), cudaEventDisableTiming));
clib-multigpu/model.c:	/* checkCudaErrors(cudaEventCreateWithFlags(&(p->accumulated), cudaEventBlockingSync | cudaEventDisableTiming)); */
clib-multigpu/model.c:	checkCudaErrors(cudaEventCreateWithFlags(&(p->accumulated), cudaEventDisableTiming));
clib-multigpu/model.c:	/* Redirect all CUDA calls to the correct device */
clib-multigpu/model.c:	checkCudaErrors (cudaSetDevice(model->dev));
clib-multigpu/model.c:		checkCudaErrors(cudaMemset (model->diff->dev, 0, model->bytes));
clib-multigpu/model.c:	checkCudaErrors(cudaMemset (model->diff->dev, 0, model->bytes));
clib-multigpu/model.c:		checkCudaErrors(cudaMemset (model->last->dev, 0, model->bytes));
clib-multigpu/model.c:    checkCudaErrors(cudaMemset (model->temp->dev, 0, model->bytes));
clib-multigpu/model.c:	/* Redirect all CUDA calls to the correct device */
clib-multigpu/model.c:	checkCudaErrors (cudaSetDevice(p->dev));
clib-multigpu/model.c:	/* Init GPU events to synchronise with the parameter server */
clib-multigpu/model.c:	p->client = crossbowMalloc (p->ops * sizeof(cudaEvent_t));
clib-multigpu/model.c:	p->server = crossbowMalloc (p->ops * sizeof(cudaEvent_t));
clib-multigpu/model.c:		checkCudaErrors(cudaEventCreateWithFlags(&(p->client[i]),  cudaEventBlockingSync | cudaEventDisableTiming));
clib-multigpu/model.c:		checkCudaErrors(cudaEventCreateWithFlags(&(p->server[i]),  cudaEventBlockingSync | cudaEventDisableTiming));
clib-multigpu/model.c:		checkCudaErrors(cudaEventCreateWithFlags(&(p->client[i]),  cudaEventDisableTiming));
clib-multigpu/model.c:		checkCudaErrors(cudaEventCreateWithFlags(&(p->server[i]),  cudaEventDisableTiming));
clib-multigpu/model.c:    checkCudaErrors(cudaEventCreateWithFlags(&(p->client), cudaEventBlockingSync | cudaEventDisableTiming));
clib-multigpu/model.c:    checkCudaErrors(cudaEventCreateWithFlags(&(p->server), cudaEventBlockingSync | cudaEventDisableTiming));
clib-multigpu/model.c:	checkCudaErrors(cudaEventCreateWithFlags(&(p->client), cudaEventDisableTiming));
clib-multigpu/model.c:	checkCudaErrors(cudaEventCreateWithFlags(&(p->server), cudaEventDisableTiming));
clib-multigpu/model.c:	/* checkCudaErrors(cudaEventCreateWithFlags(&(p->updated), cudaEventBlockingSync | cudaEventDisableTiming)); */
clib-multigpu/model.c:	checkCudaErrors(cudaEventCreateWithFlags(&(p->updated), cudaEventDisableTiming));
clib-multigpu/model.c:	/* checkCudaErrors(cudaEventCreateWithFlags(&(p->accumulated), cudaEventBlockingSync | cudaEventDisableTiming)); */
clib-multigpu/model.c:	checkCudaErrors(cudaEventCreateWithFlags(&(p->accumulated), cudaEventDisableTiming));
clib-multigpu/model.c:		checkCudaErrors(cudaEventDestroy(p->client[i]));
clib-multigpu/model.c:		checkCudaErrors(cudaEventDestroy(p->server[i]));
clib-multigpu/model.c:	crossbowFree(p->client, p->ops * sizeof(cudaEvent_t));
clib-multigpu/model.c:	crossbowFree(p->server, p->ops * sizeof(cudaEvent_t));
clib-multigpu/model.c:	checkCudaErrors(cudaEventDestroy(p->client));
clib-multigpu/model.c:	checkCudaErrors(cudaEventDestroy(p->server));
clib-multigpu/model.c:	checkCudaErrors(cudaEventDestroy(p->updated));
clib-multigpu/model.c:	checkCudaErrors(cudaEventDestroy(p->accumulated));
clib-multigpu/variable.h:void crossbowVariablePush (crossbowVariableP, cudaStream_t);
clib-multigpu/lightweightdatasetprocessortask.h:	unsigned GPU;
clib-multigpu/datasetfilemanager.h:	/* Use of GPU register or not */
clib-multigpu/datasetfilemanager.h:	unsigned gpu;
clib-multigpu/resulthandler.c:		fprintf(stderr, "warning: GPU result handler blocked at task %d (index %d)\n", taskid, ndx);
clib-multigpu/synch/common.c:#ifndef USE_NCCL
clib-multigpu/synch/common.c:	err("NCCL is disabled");
clib-multigpu/synch/common.c:	checkNcclErrors(ncclGroupStart());
clib-multigpu/synch/common.c:		/* Redirect all CUDA calls to the current device */
clib-multigpu/synch/common.c:		checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/synch/common.c:		checkCudaErrors(cudaStreamWaitEvent(dev->modelSynchronisationStream, model->accumulated, 0));
clib-multigpu/synch/common.c:		checkCudaErrors(cudaMemsetAsync(model->diff->dev, 0, model->bytes, dev->modelSynchronisationStream));
clib-multigpu/synch/common.c:		checkNcclErrors(ncclAllReduce(
clib-multigpu/synch/common.c:			ncclFloat, 
clib-multigpu/synch/common.c:			ncclSum, 
clib-multigpu/synch/common.c:    checkNcclErrors(ncclGroupEnd());
clib-multigpu/synch/common.c:#ifndef USE_NCCL
clib-multigpu/synch/common.c:	/* Redirect all CUDA calls to the default device (the master) */
clib-multigpu/synch/common.c:	checkCudaErrors (cudaSetDevice(defaultDev->id));
clib-multigpu/synch/common.c:		checkCudaErrors(cudaStreamWaitEvent(defaultDev->modelSynchronisationStream, model->accumulated, 0));
clib-multigpu/synch/common.c:		cudaMemcpyPeerAsync (defaultModel->temp->dev, defaultDev->id, model->gradient->dev, dev->id, 
clib-multigpu/synch/common.c:	checkNcclErrors(ncclGroupStart());
clib-multigpu/synch/common.c:		checkNcclErrors(ncclReduce(model->gradient->dev, defaultModel->gradient->dev, model->bytes, 
clib-multigpu/synch/common.c:            ncclChar, ncclSum, defaultDev->id, ctx->comms[dev->id], dev->modelSynchronisationStream));
clib-multigpu/synch/common.c:	checkNcclErrors(ncclGroupEnd());
clib-multigpu/synch/common.c:	/* Assumes that CUDA calls are already redirected to default device */
clib-multigpu/synch/common.c:#ifndef USE_NCCL
clib-multigpu/synch/common.c:			cudaMemcpyPeerAsync (model->data->dev, dev->id, defaultModel->data->dev, 
clib-multigpu/synch/common.c:				cudaMemcpyPeerAsync (model->last->dev, dev->id, defaultModel->last->dev, 
clib-multigpu/synch/common.c:		/* Record multi-GPU synchronisation event */
clib-multigpu/synch/common.c:		checkCudaErrors(cudaEventRecord (ctx->modelmanager->synched [dev->id], 
clib-multigpu/synch/common.c:	checkNcclErrors(ncclGroupStart());
clib-multigpu/synch/common.c:		checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/synch/common.c:		checkNcclErrors(ncclBcast(model->data->dev, model->bytes, ncclChar, defaultDev->id, 
clib-multigpu/synch/common.c:	checkNcclErrors(ncclGroupEnd());
clib-multigpu/synch/common.c:	checkCudaErrors (cudaSetDevice(defaultDev->id));
clib-multigpu/synch/common.c:	/* Record multi-GPU synchronisation event */
clib-multigpu/synch/common.c:		checkCudaErrors(cudaEventRecord (ctx->modelmanager->synched [dev->id], 
clib-multigpu/synch/common.c:	/* Copy `model` to all replicas on device `dev`. Assumes that all CUDA calls have been redirected to that device. */
clib-multigpu/synch/common.c:	checkCudaErrors(cudaStreamWaitEvent(dev->modelSynchronisationStream, ctx->modelmanager->synched[dev->id], 0));
clib-multigpu/synch/common.c:			checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, dev->modelSynchronisationStream));
clib-multigpu/synch/synchronoussgd.c:static void crossbowSynchronisationSingleGPUSynchronousSGD (crossbowExecutionContextP ctx, int first) {
clib-multigpu/synch/synchronoussgd.c:static void crossbowSynchronisationMultiGPUSynchronousSGD (crossbowExecutionContextP ctx, int first) {
clib-multigpu/synch/synchronoussgd.c:		/* Redirect all CUDA calls to the current device */
clib-multigpu/synch/synchronoussgd.c:		checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/synch/synchronoussgd.c:		checkCudaErrors(cudaEventRecord (model->accumulated, dev->modelSynchronisationStream));
clib-multigpu/synch/synchronoussgd.c:		/* Redirect all CUDA calls to the current device */
clib-multigpu/synch/synchronoussgd.c:		checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/synch/synchronoussgd.c:		checkCudaErrors(cudaEventRecord (ctx->modelmanager->synched [dev->id], dev->modelSynchronisationStream));
clib-multigpu/synch/synchronoussgd.c:		/* Redirect all CUDA calls to the current device */
clib-multigpu/synch/synchronoussgd.c:		checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/synch/synchronoussgd.c:		checkCudaErrors(cudaMemsetAsync (model->gradient->dev, 0, model->bytes, dev->modelSynchronisationStream));
clib-multigpu/synch/synchronoussgd.c:		case SINGLE_GPU:
clib-multigpu/synch/synchronoussgd.c:			crossbowSynchronisationSingleGPUSynchronousSGD (ctx, first);
clib-multigpu/synch/synchronoussgd.c:		case MULTI_GPU:
clib-multigpu/synch/synchronoussgd.c:			crossbowSynchronisationMultiGPUSynchronousSGD  (ctx, first);
clib-multigpu/synch/downpour.c:static void crossbowSynchronisationSingleGPUDownpour (crossbowExecutionContextP ctx, int first, int clock) {
clib-multigpu/synch/downpour.c:	err ("Single-GPU DOWNPOUR SGD model synchronisation is not supported yet\n");
clib-multigpu/synch/downpour.c:static void crossbowSynchronisationMultiGPUDownpour (crossbowExecutionContextP ctx, int first, int clock) {
clib-multigpu/synch/downpour.c:	err ("Multi-GPU DOWNPOUR SGD model synchronisation is not supported yet\n");
clib-multigpu/synch/downpour.c:		case SINGLE_GPU:
clib-multigpu/synch/downpour.c:			crossbowSynchronisationSingleGPUDownpour (ctx, first, clock);
clib-multigpu/synch/downpour.c:		case MULTI_GPU:
clib-multigpu/synch/downpour.c:			crossbowSynchronisationMultiGPUDownpour  (ctx, first, clock);
clib-multigpu/synch/default.c:static void crossbowSynchronisationSingleGPUDefault (crossbowExecutionContextP ctx, int first) {
clib-multigpu/synch/default.c:	/* Redirect all CUDA calls to the default device */
clib-multigpu/synch/default.c:	checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/synch/default.c:	/* checkCudaErrors(cudaDeviceSynchronize()); */
clib-multigpu/synch/default.c:			checkCudaErrors(cudaStreamWaitEvent(dev->modelSynchronisationStream, 
clib-multigpu/synch/default.c:			checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, 
clib-multigpu/synch/default.c:	/* checkCudaErrors(cudaDeviceSynchronize()); */
clib-multigpu/synch/default.c:static void crossbowSynchronisationMultiGPUDefault (crossbowExecutionContextP ctx, int first) {
clib-multigpu/synch/default.c:	err ("Multi-GPU default SGD model synchronisation is not supported yet\n");
clib-multigpu/synch/default.c:		case SINGLE_GPU: 
clib-multigpu/synch/default.c:			crossbowSynchronisationSingleGPUDefault (ctx, first); 
clib-multigpu/synch/default.c:		case MULTI_GPU:
clib-multigpu/synch/default.c:			crossbowSynchronisationMultiGPUDefault (ctx, first); 
clib-multigpu/synch/eamsgd.c:static void crossbowSynchronisationSingleGPUElasticAveragingSGD (crossbowExecutionContextP ctx, int first, int clock) {
clib-multigpu/synch/eamsgd.c:	err ("Single-GPU asynchronous Elastic Averaging SGD model synchronisation is not supported yet\n");
clib-multigpu/synch/eamsgd.c:static void crossbowSynchronisationMultiGPUElasticAveragingSGD (crossbowExecutionContextP ctx, int first, int clock) {
clib-multigpu/synch/eamsgd.c:	err ("Multi-GPU asynchronous Elastic Averaging SGD model synchronisation is not supported yet\n");
clib-multigpu/synch/eamsgd.c:		case SINGLE_GPU:
clib-multigpu/synch/eamsgd.c:			crossbowSynchronisationSingleGPUElasticAveragingSGD (ctx, first, clock);
clib-multigpu/synch/eamsgd.c:		case MULTI_GPU:
clib-multigpu/synch/eamsgd.c:			crossbowSynchronisationMultiGPUElasticAveragingSGD  (ctx, first, clock);
clib-multigpu/synch/polyakruppert.c:static void crossbowSynchronisationSingleGPUPolyakRuppert (crossbowExecutionContextP ctx, int first, int clock) {
clib-multigpu/synch/polyakruppert.c:	/* Redirect all CUDA calls to the default device */
clib-multigpu/synch/polyakruppert.c:	checkCudaErrors (cudaSetDevice(defaultDev->id));
clib-multigpu/synch/polyakruppert.c:	checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/synch/polyakruppert.c:	checkCudaErrors(cudaStreamWaitEvent(defaultDev->modelSynchronisationStream, defaultModel->updated, 0));
clib-multigpu/synch/polyakruppert.c:	checkCudaErrors(cudaMemsetAsync(defaultModel->gradient->dev, 0, defaultModel->bytes, defaultDev->modelSynchronisationStream));
clib-multigpu/synch/polyakruppert.c:				checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, defaultDev->modelSynchronisationStream));
clib-multigpu/synch/polyakruppert.c:	checkCudaErrors(cudaEventRecord(defaultModel->updated, defaultDev->modelSynchronisationStream));
clib-multigpu/synch/polyakruppert.c:	checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/synch/polyakruppert.c:				checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, defaultDev->modelSynchronisationStream));
clib-multigpu/synch/polyakruppert.c:static void crossbowSynchronisationMultiGPUPolyakRuppert (crossbowExecutionContextP ctx, int first, int clock) {
clib-multigpu/synch/polyakruppert.c:		/* Redirect all CUDA calls to the current device */
clib-multigpu/synch/polyakruppert.c:		checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/synch/polyakruppert.c:		checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/synch/polyakruppert.c:		checkCudaErrors(cudaStreamWaitEvent(dev->modelSynchronisationStream, model->updated, 0));
clib-multigpu/synch/polyakruppert.c:		checkCudaErrors(cudaMemsetAsync (model->gradient->dev, 0, model->bytes, dev->modelSynchronisationStream));
clib-multigpu/synch/polyakruppert.c:					checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, dev->modelSynchronisationStream));
clib-multigpu/synch/polyakruppert.c:		checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/synch/polyakruppert.c:		checkCudaErrors(cudaEventRecord (model->accumulated, dev->modelSynchronisationStream));
clib-multigpu/synch/polyakruppert.c:	/* CUDA calls redirected to default device */
clib-multigpu/synch/polyakruppert.c:	checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/synch/polyakruppert.c:		/* Redirect all CUDA calls to the current device */
clib-multigpu/synch/polyakruppert.c:		checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/synch/polyakruppert.c:		checkCudaErrors(cudaStreamWaitEvent(dev->modelSynchronisationStream, ctx->modelmanager->synched[dev->id], 0));
clib-multigpu/synch/polyakruppert.c:		checkCudaErrors(cudaEventRecord(model->updated, dev->modelSynchronisationStream));
clib-multigpu/synch/polyakruppert.c:					checkCudaErrors(cudaEventRecord (ctx->modelmanager->replicas[id]->updated, dev->modelSynchronisationStream));
clib-multigpu/synch/polyakruppert.c:		case SINGLE_GPU:
clib-multigpu/synch/polyakruppert.c:			crossbowSynchronisationSingleGPUPolyakRuppert (ctx, first, clock);
clib-multigpu/synch/polyakruppert.c:		case MULTI_GPU:
clib-multigpu/synch/polyakruppert.c:			crossbowSynchronisationMultiGPUPolyakRuppert  (ctx, first, clock);
clib-multigpu/synch/synchronouseamsgd.c:static void crossbowSynchronisationSingleGPUSynchronousElasticAveragingSGD (crossbowExecutionContextP ctx, int first, int clock) {
clib-multigpu/synch/synchronouseamsgd.c:#ifdef GPU_VERBOSE
clib-multigpu/synch/synchronouseamsgd.c:	/* Redirect all CUDA calls to the default device */
clib-multigpu/synch/synchronouseamsgd.c:	checkCudaErrors (cudaSetDevice(defaultDev->id));
clib-multigpu/synch/synchronouseamsgd.c:	checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/synch/synchronouseamsgd.c:	checkCudaErrors(cudaStreamWaitEvent(defaultDev->modelSynchronisationStream, defaultModel->updated, 0));
clib-multigpu/synch/synchronouseamsgd.c:	checkCudaErrors(cudaMemsetAsync(defaultModel->gradient->dev, 0, defaultModel->bytes, defaultDev->modelSynchronisationStream));
clib-multigpu/synch/synchronouseamsgd.c:			checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, defaultDev->modelSynchronisationStream));
clib-multigpu/synch/synchronouseamsgd.c:	checkCudaErrors(cudaEventRecord(defaultModel->updated, defaultDev->modelSynchronisationStream));
clib-multigpu/synch/synchronouseamsgd.c:	checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/synch/synchronouseamsgd.c:static void crossbowSynchronisationMultiGPUSynchronousElasticAveragingSGD (crossbowExecutionContextP ctx, int first, int clock) {
clib-multigpu/synch/synchronouseamsgd.c:#ifdef GPU_VERBOSE
clib-multigpu/synch/synchronouseamsgd.c:		/* Redirect all CUDA calls to the current device */
clib-multigpu/synch/synchronouseamsgd.c:		checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/synch/synchronouseamsgd.c:		checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/synch/synchronouseamsgd.c:		checkCudaErrors(cudaStreamWaitEvent(dev->modelSynchronisationStream, model->updated, 0));
clib-multigpu/synch/synchronouseamsgd.c:		checkCudaErrors(cudaMemsetAsync (model->gradient->dev, 0, model->bytes, dev->modelSynchronisationStream));
clib-multigpu/synch/synchronouseamsgd.c:				checkCudaErrors(cudaStreamWaitEvent(dev->modelSynchronisationStream, defaultModel->accumulated, 0));
clib-multigpu/synch/synchronouseamsgd.c:				checkCudaErrors(cudaEventRecord (ctx->modelmanager->replicas[id]->accumulated, dev->modelSynchronisationStream));
clib-multigpu/synch/synchronouseamsgd.c:				checkCudaErrors (cudaSetDevice(defaultDev->id));
clib-multigpu/synch/synchronouseamsgd.c:				checkCudaErrors(cudaStreamWaitEvent(defaultDev->modelSynchronisationStream, ctx->modelmanager->replicas[id]->accumulated, 0));
clib-multigpu/synch/synchronouseamsgd.c:				cudaMemcpyPeerAsync (defaultModel->temp->dev, defaultDev->id, model->diff->dev, dev->id, defaultModel->bytes, defaultDev->modelSynchronisationStream);
clib-multigpu/synch/synchronouseamsgd.c:				checkCudaErrors(cudaEventRecord (defaultModel->accumulated, defaultDev->modelSynchronisationStream));
clib-multigpu/synch/synchronouseamsgd.c:				checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/synch/synchronouseamsgd.c:				checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, 
clib-multigpu/synch/synchronouseamsgd.c:		checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/synch/synchronouseamsgd.c:		checkCudaErrors(cudaEventRecord (model->accumulated, dev->modelSynchronisationStream));
clib-multigpu/synch/synchronouseamsgd.c:	/* CUDA calls redirected to default device, but reset it anyway */
clib-multigpu/synch/synchronouseamsgd.c:	checkCudaErrors (cudaSetDevice(defaultDev->id));
clib-multigpu/synch/synchronouseamsgd.c:	checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/synch/synchronouseamsgd.c:		/* Redirect all CUDA calls to the current device */
clib-multigpu/synch/synchronouseamsgd.c:		checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/synch/synchronouseamsgd.c:		checkCudaErrors(cudaStreamWaitEvent(dev->modelSynchronisationStream, ctx->modelmanager->synched[dev->id], 0));
clib-multigpu/synch/synchronouseamsgd.c:		checkCudaErrors(cudaEventRecord(model->updated, dev->modelSynchronisationStream));
clib-multigpu/synch/synchronouseamsgd.c:		case SINGLE_GPU:
clib-multigpu/synch/synchronouseamsgd.c:			crossbowSynchronisationSingleGPUSynchronousElasticAveragingSGD (ctx, first, clock);
clib-multigpu/synch/synchronouseamsgd.c:		case MULTI_GPU:
clib-multigpu/synch/synchronouseamsgd.c:			crossbowSynchronisationMultiGPUSynchronousElasticAveragingSGD  (ctx, first, clock);
clib-multigpu/synch/sma.c:static void crossbowSynchronisationSingleGPUSynchronousModelAveraging (crossbowExecutionContextP ctx, int first) {
clib-multigpu/synch/sma.c:static void crossbowSynchronisationMultiGPUSynchronousModelAveraging (crossbowExecutionContextP ctx, int first) {
clib-multigpu/synch/sma.c:#ifdef GPU_VERBOSE
clib-multigpu/synch/sma.c:		/* Redirect all CUDA calls to the current device */
clib-multigpu/synch/sma.c:		checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/synch/sma.c:		checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/synch/sma.c:		checkCudaErrors(cudaStreamWaitEvent(dev->modelSynchronisationStream, model->updated, 0));
clib-multigpu/synch/sma.c:		checkCudaErrors(cudaMemsetAsync (model->gradient->dev, 0, model->bytes, dev->modelSynchronisationStream));
clib-multigpu/synch/sma.c:					checkCudaErrors(cudaEventRecord(ctx->modelmanager->replicas[id]->updated, 
clib-multigpu/synch/sma.c:		checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/synch/sma.c:		checkCudaErrors(cudaEventRecord (model->accumulated, dev->modelSynchronisationStream));
clib-multigpu/synch/sma.c:	/* Redirected CUDA calls to default device */
clib-multigpu/synch/sma.c:	checkCudaErrors (cudaSetDevice(defaultDev->id));
clib-multigpu/synch/sma.c:		/* Redirect all CUDA calls to the current device */
clib-multigpu/synch/sma.c:		checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/synch/sma.c:		/* Record multi-GPU synchronisation event */
clib-multigpu/synch/sma.c:                checkCudaErrors(cudaEventRecord (ctx->modelmanager->synched [dev->id], dev->modelSynchronisationStream));
clib-multigpu/synch/sma.c:	checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/synch/sma.c:		/* Redirect all CUDA calls to the current device */
clib-multigpu/synch/sma.c:		checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/synch/sma.c:		checkCudaErrors(cudaStreamWaitEvent(dev->modelSynchronisationStream, ctx->modelmanager->synched[dev->id], 0));
clib-multigpu/synch/sma.c:		checkCudaErrors(cudaEventRecord(model->updated, dev->modelSynchronisationStream));
clib-multigpu/synch/sma.c:					checkCudaErrors(cudaEventRecord (ctx->modelmanager->replicas[id]->updated, 
clib-multigpu/synch/sma.c:		case SINGLE_GPU:
clib-multigpu/synch/sma.c:			crossbowSynchronisationSingleGPUSynchronousModelAveraging (ctx, first);
clib-multigpu/synch/sma.c:		case MULTI_GPU:
clib-multigpu/synch/sma.c:			crossbowSynchronisationMultiGPUSynchronousModelAveraging  (ctx, first);
clib-multigpu/synch/hogwild.c:static void crossbowSynchronisationSingleGPUHogwild (crossbowExecutionContextP ctx, int first, int clock) {
clib-multigpu/synch/hogwild.c:	err ("Single-GPU Hogwild! SGD model synchronisation is not supported yet\n");
clib-multigpu/synch/hogwild.c:static void crossbowSynchronisationMultiGPUHogwild (crossbowExecutionContextP ctx, int first, int clock) {
clib-multigpu/synch/hogwild.c:	err ("Multi-GPU Hogwild! SGD model synchronisation is not supported yet\n");
clib-multigpu/synch/hogwild.c:		case SINGLE_GPU:
clib-multigpu/synch/hogwild.c:			crossbowSynchronisationSingleGPUHogwild (ctx, first, clock);
clib-multigpu/synch/hogwild.c:		case MULTI_GPU:
clib-multigpu/synch/hogwild.c:			crossbowSynchronisationMultiGPUHogwild  (ctx, first, clock);
clib-multigpu/GPU.c:#include "uk_ac_imperial_lsds_crossbow_device_TheGPU.h"
clib-multigpu/GPU.c:static crossbowExecutionContextP theGPU = NULL;
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_init
clib-multigpu/GPU.c:	if (theGPU) {
clib-multigpu/GPU.c:		fprintf(stderr, "error: GPU execution context already initialised\n");
clib-multigpu/GPU.c:	theGPU = crossbowExecutionContextInit (argv, argc, 
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_free
clib-multigpu/GPU.c:	crossbowExecutionContextFree (env, theGPU);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_dump
clib-multigpu/GPU.c:	crossbowExecutionContextDump (theGPU);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_configureBatchExamples
clib-multigpu/GPU.c:	crossbowExecutionContextSetBatchExamples (theGPU, argc, argv, size);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_configureBatchLabels
clib-multigpu/GPU.c:	crossbowExecutionContextSetBatchLabels (theGPU, argc, argv, size);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_configureBatchSplits
clib-multigpu/GPU.c:	crossbowExecutionContextSetBatchSplits (theGPU, splits);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_configureStreams
clib-multigpu/GPU.c:	crossbowExecutionContextCreateStreams (theGPU, branches);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setRandomSeed
clib-multigpu/GPU.c:	crossbowExecutionContextSetRandomSeed (theGPU, (unsigned long long) seed);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernel
clib-multigpu/GPU.c:	crossbowExecutionContextSetKernel (theGPU, id, binding, inputs, variables, outputs, (pull == JNI_TRUE) ? 1 : 0);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelInput
clib-multigpu/GPU.c:	crossbowExecutionContextSetKernelInput (theGPU, id, ndx, argc, argv, capacity);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelOutput
clib-multigpu/GPU.c:	crossbowExecutionContextSetKernelOutput (theGPU, id, argc, argv, capacity);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelLocalVariable
clib-multigpu/GPU.c:	crossbowExecutionContextSetKernelLocalVariable (theGPU, id, ndx, binding, argc, argv, capacity, readonly);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelLocalVariableBuffer
clib-multigpu/GPU.c:	crossbowExecutionContextSetKernelLocalVariableBuffer (theGPU, id, ndx, src);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelScalars
clib-multigpu/GPU.c:	crossbowExecutionContextSetKernelScalars (theGPU, id, count);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelScalarAsInt
clib-multigpu/GPU.c:	crossbowExecutionContextSetKernelScalarAsInt (theGPU, id, ndx, binding, value);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelScalarAsFloat
clib-multigpu/GPU.c:	crossbowExecutionContextSetKernelScalarAsFloat (theGPU, id, ndx, binding, value);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelScalarAsDouble
clib-multigpu/GPU.c:	crossbowExecutionContextSetKernelScalarAsDouble (theGPU, id, ndx, binding, value);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetKernelType
clib-multigpu/GPU.c:	crossbowExecutionContextCudnnSetKernelType (theGPU, id, type);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetKernelInputDescriptor
clib-multigpu/GPU.c:	crossbowExecutionContextCudnnSetKernelInputDescriptor (theGPU, id, count, channels, height, width);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetKernelOutputDescriptor
clib-multigpu/GPU.c:	crossbowExecutionContextCudnnSetKernelOutputDescriptor (theGPU, id, count, channels, height, width);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetConvolutionDescriptor
clib-multigpu/GPU.c:	crossbowExecutionContextCudnnSetConvolutionDescriptor (theGPU, id, paddingHeight, paddingWidth, strideHeight, strideWidth);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetConvolutionFilterDescriptor
clib-multigpu/GPU.c:	crossbowExecutionContextCudnnSetConvolutionFilterDescriptor (theGPU, id, count, channels, height, width);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetConvolutionBiasDescriptor
clib-multigpu/GPU.c:	crossbowExecutionContextCudnnSetConvolutionBiasDescriptor (theGPU, id, count, channels, height, width);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnConfigureConvolutionForwardAlgorithm
clib-multigpu/GPU.c:	size_t workSpaceSize = crossbowExecutionContextCudnnConfigureConvolutionForwardAlgorithm (theGPU, id, limit, threshold);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnConfigureConvolutionBackwardFilterAlgorithm
clib-multigpu/GPU.c:	size_t workSpaceSize = crossbowExecutionContextCudnnConfigureConvolutionBackwardFilterAlgorithm (theGPU, id, limit, threshold);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnConfigureConvolutionBackwardDataAlgorithm
clib-multigpu/GPU.c:	size_t workSpaceSize = crossbowExecutionContextCudnnConfigureConvolutionBackwardDataAlgorithm (theGPU, id, limit, threshold);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetPoolingMode
clib-multigpu/GPU.c:	crossbowExecutionContextCudnnSetPoolingMode (theGPU, id, mode);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetPoolingDescriptor
clib-multigpu/GPU.c:	crossbowExecutionContextCudnnSetPoolingDescriptor (theGPU, id, windowHeight, windowWidth, paddingHeight, paddingWidth, strideHeight, strideWidth);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetActivationDescriptor
clib-multigpu/GPU.c:	crossbowExecutionContextCudnnSetActivationDescriptor (theGPU, id, mode, ceiling);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetBatchNormDescriptor
clib-multigpu/GPU.c:	crossbowExecutionContextCudnnSetBatchNormDescriptor (theGPU, id);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetBatchNormEstimatedMeanAndVariance
clib-multigpu/GPU.c:	crossbowExecutionContextCudnnSetBatchNormEstimatedMeanAndVariance (theGPU, id, capacity);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetDropoutDescriptor
clib-multigpu/GPU.c:	crossbowExecutionContextCudnnSetDropoutDescriptor (theGPU, id, dropout, seed);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnGetDropoutReserveSpaceSize
clib-multigpu/GPU.c:	size_t reserveSpaceSize = crossbowExecutionContextCudnnGetDropoutReserveSpaceSize (theGPU, id);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameters
clib-multigpu/GPU.c:	crossbowExecutionContextSetKernelConfigurationParameters (theGPU, id, count);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameterAsInt
clib-multigpu/GPU.c:	crossbowExecutionContextSetKernelConfigurationParameterAsInt (theGPU, id, ndx, binding, value);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameterAsFloat
clib-multigpu/GPU.c:	crossbowExecutionContextSetKernelConfigurationParameterAsFloat (theGPU, id, ndx, binding, value);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameterAsIntArray
clib-multigpu/GPU.c:	crossbowExecutionContextSetKernelConfigurationParameterAsIntArray (theGPU, id, ndx, binding, argc, argv);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameterAsFloatArray
clib-multigpu/GPU.c:	crossbowExecutionContextSetKernelConfigurationParameterAsFloatArray (theGPU, id, ndx, binding, argc, argv);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameterAsDouble
clib-multigpu/GPU.c:	crossbowExecutionContextSetKernelConfigurationParameterAsDouble (theGPU, id, ndx, binding, value);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowGraph
clib-multigpu/GPU.c:	crossbowExecutionContextSetDataflowGraph (theGPU, id, argc, argv);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowStream
clib-multigpu/GPU.c:	crossbowExecutionContextSetDataflowStream (theGPU, id, ord, branch);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowDependency
clib-multigpu/GPU.c:	crossbowExecutionContextSetDataflowDependency (theGPU, id, ord, type, guard, (internal == JNI_TRUE) ? 0 : 1);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowUpstreamNeighbours
clib-multigpu/GPU.c:	crossbowExecutionContextSetDataflowUpstreamNeighbours (theGPU, id, ord, argc, argv);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowDownstreamNeighbours
clib-multigpu/GPU.c:	crossbowExecutionContextSetDataflowDownstreamNeighbours (theGPU, id, ord, argc, argv);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowLossOperator
clib-multigpu/GPU.c:	crossbowExecutionContextSetDataflowLossOperator(theGPU, id, op);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowAccuracyOperator
clib-multigpu/GPU.c:	crossbowExecutionContextSetDataflowAccuracyOperator(theGPU, id, op);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowDataTransformOperator
clib-multigpu/GPU.c:	crossbowExecutionContextSetDataflowDataTransformOperator(theGPU, id, op);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowPeers
clib-multigpu/GPU.c:	crossbowExecutionContextSetDataflowPeers (theGPU, id, argc, argv);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowMemoryPlan
clib-multigpu/GPU.c:	crossbowExecutionContextSetDataflowMemoryPlan (theGPU, id, order, provider, position);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModel
clib-multigpu/GPU.c:	crossbowExecutionContextSetModel (theGPU, variables, size);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModelVariable
clib-multigpu/GPU.c:	crossbowExecutionContextSetModelVariable (theGPU, id, order, argc, argv, capacity);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModelVariableBuffer
clib-multigpu/GPU.c:	crossbowExecutionContextSetModelVariableBuffer (theGPU, id, order, src);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModelVariableLearningRateMultiplier
clib-multigpu/GPU.c:	crossbowExecutionContextSetModelVariableLearningRateMultiplier (theGPU, id, order, multiplier);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModelWorkPerClock
clib-multigpu/GPU.c:	crossbowExecutionContextSetModelWorkPerClock (theGPU, wpc);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setUpdateModelType
clib-multigpu/GPU.c:	crossbowExecutionContextSetUpdateModelType (theGPU, type);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyFixed
clib-multigpu/GPU.c:	crossbowExecutionContextSetLearningRateDecayPolicyFixed (theGPU, rate);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyInv
clib-multigpu/GPU.c:	crossbowExecutionContextSetLearningRateDecayPolicyInv (theGPU, learningRate, gamma, power);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyStep
clib-multigpu/GPU.c:	crossbowExecutionContextSetLearningRateDecayPolicyStep (theGPU, learningRate, gamma, size);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyMultiStep
clib-multigpu/GPU.c:	crossbowExecutionContextSetLearningRateDecayPolicyMultiStep (theGPU, learningRate, gamma, warmuptasks, argc, argv);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyExp
clib-multigpu/GPU.c:	crossbowExecutionContextSetLearningRateDecayPolicyExp (theGPU, learningRate, gamma);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyCircular
clib-multigpu/GPU.c:	crossbowExecutionContextSetLearningRateDecayPolicyCircular (theGPU, H, superconvergence, M, step);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setBaseModelMomentum
clib-multigpu/GPU.c:	crossbowExecutionContextSetBaseModelMomentum (theGPU, momentum);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setMomentum
clib-multigpu/GPU.c:	crossbowExecutionContextSetMomentum (theGPU, momentum, method);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setWeightDecay
clib-multigpu/GPU.c:	crossbowExecutionContextSetWeightDecay (theGPU, weigthDecay);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setEamsgdAlpha
clib-multigpu/GPU.c:	crossbowExecutionContextSetEamsgdAlpha (theGPU, alpha);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setEamsgdTau
clib-multigpu/GPU.c:	crossbowExecutionContextSetEamsgdTau (theGPU, tau);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModelManager
clib-multigpu/GPU.c:	crossbowExecutionContextSetModelManager (env, theGPU, replicas, type);
clib-multigpu/GPU.c:JNIEXPORT jobject JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_acquireAccess
clib-multigpu/GPU.c:	result = crossbowExecutionContextAcquireAccess (env, theGPU, &argv[0]);
clib-multigpu/GPU.c:JNIEXPORT jobject JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_upgradeAccess
clib-multigpu/GPU.c:	result = crossbowExecutionContextUpgradeAccess (env, theGPU, replicaId, &argv[0]);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_release
clib-multigpu/GPU.c:	err("Cannot release a GPU model replica id object from the GPU\n");
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setResultHandler
clib-multigpu/GPU.c:	crossbowExecutionContextSetResultHandler (theGPU, id, slots, count);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLightWeightDatasetHandler
clib-multigpu/GPU.c:	crossbowExecutionContextSetLightWeightDatasetHandler (theGPU, id, slots, count);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_execute
clib-multigpu/GPU.c:		theGPU,
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_executeNext
clib-multigpu/GPU.c:			theGPU,
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_schedule
clib-multigpu/GPU.c:			theGPU,
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_scheduleNext
clib-multigpu/GPU.c:			theGPU,
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_lockAny
clib-multigpu/GPU.c:	return crossbowExecutionContextLockModels (theGPU);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_merge
clib-multigpu/GPU.c:	int result = crossbowExecutionContextMergeModels (theGPU, (pull == JNI_TRUE) ? 1 : 0);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_synchronise
clib-multigpu/GPU.c:	return crossbowExecutionContextSynchroniseModels (theGPU, first, clock, autotune, (push == JNI_TRUE) ? 1 : 0);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_unlockAny
clib-multigpu/GPU.c:	return crossbowExecutionContextUnlockModels(theGPU);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_checkpointModel
clib-multigpu/GPU.c:	crossbowExecutionContextCheckpointModels (theGPU, binding);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_overrideModelData
clib-multigpu/GPU.c:		crossbowExecutionContextOverrideModelData (theGPU, binding);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_addModel
clib-multigpu/GPU.c:	crossbowExecutionContextAddModel (theGPU);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_delModel
clib-multigpu/GPU.c:	crossbowExecutionContextDelModel (theGPU);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_recordDatasetInit
clib-multigpu/GPU.c:	crossbowExecutionContextRecordDatasetInit (theGPU, phase, workers, capacity, NB, b, padding);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_recordDatasetRegister
clib-multigpu/GPU.c:	crossbowExecutionContextRecordDatasetRegister (theGPU, phase, id, binding);
clib-multigpu/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_recordDatasetFinalise
clib-multigpu/GPU.c:	crossbowExecutionContextRecordDatasetFinalise (theGPU, phase);
clib-multigpu/doublebuffer.c:#include <cuda.h>
clib-multigpu/doublebuffer.c:#include <cuda_runtime.h>
clib-multigpu/doublebuffer.c:    	checkCudaErrors(cudaHostRegister(p->theImages[i], p->capacity[0], cudaHostRegisterMapped | cudaHostRegisterPortable));
clib-multigpu/doublebuffer.c:    	checkCudaErrors(cudaHostRegister(p->theLabels[i], p->capacity[1], cudaHostRegisterMapped | cudaHostRegisterPortable));
clib-multigpu/doublebuffer.c:    checkCudaErrors(cudaHostUnregister(p->theImages[0]));
clib-multigpu/doublebuffer.c:    checkCudaErrors(cudaHostUnregister(p->theImages[1]));
clib-multigpu/doublebuffer.c:    checkCudaErrors(cudaHostUnregister(p->theLabels[0]));
clib-multigpu/doublebuffer.c:    checkCudaErrors(cudaHostUnregister(p->theLabels[1]));
clib-multigpu/databuffer.c:		p->host = crossbowCudaMallocHost (p->size);
clib-multigpu/databuffer.c:	p->dev = crossbowCudaMalloc (p->size);
clib-multigpu/databuffer.c:		p->host = crossbowCudaMallocHost (p->size);
clib-multigpu/databuffer.c:	p->dev = crossbowCudaMalloc (p->size);
clib-multigpu/databuffer.c:	/* Copy GPU device memory region as well; use default stream */
clib-multigpu/databuffer.c:	checkCudaErrors(cudaMemcpy (p->dev, buffer->dev, p->size, cudaMemcpyDeviceToDevice));
clib-multigpu/databuffer.c:void crossbowDataBufferCopyDeviceRegion (crossbowDataBufferP p, crossbowDataBufferP q, cudaStream_t stream) {
clib-multigpu/databuffer.c:	/* Copy q's GPU device buffer to p's GPU device buffer */
clib-multigpu/databuffer.c:	checkCudaErrors(cudaMemcpyAsync (p->dev, q->dev, p->size, cudaMemcpyDeviceToDevice, stream));
clib-multigpu/databuffer.c:/* Push `host` to GPU device */
clib-multigpu/databuffer.c:void crossbowDataBufferPush (crossbowDataBufferP p, cudaStream_t stream) {
clib-multigpu/databuffer.c:	checkCudaErrors(cudaMemcpyAsync (p->dev, p->host, p->size, cudaMemcpyHostToDevice, stream));
clib-multigpu/databuffer.c:	checkCudaErrors(cudaMemcpy (p->dev, p->host, p->size, cudaMemcpyHostToDevice));
clib-multigpu/databuffer.c:/* Push `data` to GPU device */
clib-multigpu/databuffer.c:void crossbowDataBufferPushRegion (crossbowDataBufferP p, void *data, int offset, int length, cudaStream_t stream) {
clib-multigpu/databuffer.c:	checkCudaErrors(cudaMemcpyAsync (((char *) (p->dev) + offset), data, length, cudaMemcpyHostToDevice, stream));
clib-multigpu/databuffer.c:/* Pull data from GPU device */
clib-multigpu/databuffer.c:void crossbowDataBufferPull (crossbowDataBufferP p, cudaStream_t stream) {
clib-multigpu/databuffer.c:	checkCudaErrors(cudaMemcpyAsync (p->host, p->dev, p->size, cudaMemcpyDeviceToHost, stream));
clib-multigpu/databuffer.c:	checkCudaErrors(cudaMemcpy (p->host, p->dev, p->size, cudaMemcpyDeviceToHost));
clib-multigpu/databuffer.c:	checkCudaErrors(cudaMemcpy (t, (void *)(((char *) p->dev) + offset), bytes, cudaMemcpyDeviceToHost));
clib-multigpu/databuffer.c:	checkCudaErrors(cudaMemcpy (t, (void *)(((char *) p->dev) + offset), bytes, cudaMemcpyDeviceToHost));
clib-multigpu/databuffer.c:	checkCudaErrors(cudaMemcpy (t, (void *)(((char *) p->dev) + offset), bytes, cudaMemcpyDeviceToHost));
clib-multigpu/databuffer.c:#ifdef GPU_VERBOSE
clib-multigpu/databuffer.c:	/* Fetch GPU data to host memory */
clib-multigpu/databuffer.c:	checkCudaErrors(cudaMemcpy (t, p->dev, p->size, cudaMemcpyDeviceToHost));
clib-multigpu/databuffer.c:	/* Copy host buffer to GPU memory */
clib-multigpu/databuffer.c:	checkCudaErrors(cudaMemcpy (p->dev, t, p->size, cudaMemcpyHostToDevice));
clib-multigpu/databuffer.c:#ifdef GPU_VERBOSE
clib-multigpu/databuffer.c:		crossbowCudaFreeHost (p->host, p->size);
clib-multigpu/databuffer.c:	crossbowCudaFree (p->dev, p->size);
clib-multigpu/kernel.c:	p->parameters = NULL; /* Initialised by TheGPU_setKernelConfigurationParameters() in GPU.c */
clib-multigpu/kernel.c:	p->scalars = NULL; /* Initialised by TheGPU_setKernelScalars() in GPU.c */
clib-multigpu/kernel.c:	 * from GPU memory. */
clib-multigpu/kernel.c:		checkCudaErrors (cudaSetDevice (dev->id));
clib-multigpu/kernel.c:	 * Lazy materialisation of output buffers: note that `cudaMalloc` and `cudaFree`
clib-multigpu/kernel.c:	 * are implicit GPU synchronisation points.
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:/* Header for class uk_ac_imperial_lsds_crossbow_device_TheGPU */
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:#ifndef _Included_uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:#define _Included_uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_init
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_free
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_dump
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_configureBatchExamples
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_configureBatchLabels
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_configureBatchSplits
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_configureStreams
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setRandomSeed
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernel
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelInput
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelOutput
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelLocalVariable
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelLocalVariableBuffer
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameters
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameterAsInt
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameterAsFloat
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameterAsIntArray
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameterAsFloatArray
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelConfigurationParameterAsDouble
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelScalars
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelScalarAsInt
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelScalarAsFloat
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setKernelScalarAsDouble
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetKernelType
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetKernelInputDescriptor
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetKernelOutputDescriptor
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetConvolutionDescriptor
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetConvolutionFilterDescriptor
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetConvolutionBiasDescriptor
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnConfigureConvolutionForwardAlgorithm
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnConfigureConvolutionBackwardFilterAlgorithm
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnConfigureConvolutionBackwardDataAlgorithm
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetPoolingMode
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetPoolingDescriptor
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetActivationDescriptor
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetBatchNormDescriptor
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetBatchNormEstimatedMeanAndVariance
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnSetDropoutDescriptor
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_cudnnGetDropoutReserveSpaceSize
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowGraph
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowStream
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowDependency
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowUpstreamNeighbours
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowDownstreamNeighbours
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowLossOperator
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowAccuracyOperator
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowDataTransformOperator
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowPeers
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setDataflowMemoryPlan
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModel
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModelVariable
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModelVariableBuffer
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModelVariableLearningRateMultiplier
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModelWorkPerClock
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setUpdateModelType
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyFixed
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyInv
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyStep
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyMultiStep
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyExp
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLearningRateDecayPolicyCircular
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setBaseModelMomentum
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setMomentum
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setWeightDecay
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setEamsgdAlpha
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setEamsgdTau
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setModelManager
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jobject JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_acquireAccess
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_release
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jobject JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_upgradeAccess
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_execute
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_executeNext
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_schedule
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_scheduleNext
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setResultHandler
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_setLightWeightDatasetHandler
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_lockAny
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_merge
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_synchronise
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_unlockAny
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_checkpointModel
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_overrideModelData
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_addModel
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_delModel
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_recordDatasetInit
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_recordDatasetRegister
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h: * Class:     uk_ac_imperial_lsds_crossbow_device_TheGPU
clib-multigpu/uk_ac_imperial_lsds_crossbow_device_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_crossbow_device_TheGPU_recordDatasetFinalise
clib-multigpu/databuffer.h:#include <cuda_runtime.h>
clib-multigpu/databuffer.h:void crossbowDataBufferCopyDeviceRegion (crossbowDataBufferP, crossbowDataBufferP, cudaStream_t);
clib-multigpu/databuffer.h:void crossbowDataBufferPush (crossbowDataBufferP, cudaStream_t);
clib-multigpu/databuffer.h:void crossbowDataBufferPushRegion (crossbowDataBufferP, void *, int, int, cudaStream_t);
clib-multigpu/databuffer.h:void crossbowDataBufferPull (crossbowDataBufferP, cudaStream_t);
clib-multigpu/genmakefile.sh:MAKEFILE="$CROSSBOW_HOME"/clib-multigpu/Makefile
clib-multigpu/genmakefile.sh:CH="$CUDA_HOME"
clib-multigpu/genmakefile.sh:# If CUDA_HOME is not set, try to find it
clib-multigpu/genmakefile.sh:[ -z "$CH" ] && [ -d "/usr/local/cuda" ] && CH="/usr/local/cuda"
clib-multigpu/genmakefile.sh:NH="$NCCL_HOME"
clib-multigpu/genmakefile.sh:# If NCCL_HOME is not set, try to find it
clib-multigpu/genmakefile.sh:[ -z "$NH" ] && [ -d "/opt/nccl" ] && BH="/opt/nccl"
clib-multigpu/genmakefile.sh:echo "CUDA_PATH := $CH" >>"$MAKEFILE"
clib-multigpu/genmakefile.sh:echo "NCCL_PATH := $NH" >>"$MAKEFILE"
clib-multigpu/genmakefile.sh:NV := \$(CUDA_PATH)/bin/nvcc -ccbin \$(CC)
clib-multigpu/genmakefile.sh:# Added to supress warnings after switch to CUDA 8.0
clib-multigpu/genmakefile.sh:WARN := -Wno-deprecated-gpu-targets
clib-multigpu/genmakefile.sh:	# CCFLAGS += -rpath \$(CUDA_PATH)/lib
clib-multigpu/genmakefile.sh:	LFL += -Xlinker -framework -Xlinker CUDA
clib-multigpu/genmakefile.sh:# CUDA
clib-multigpu/genmakefile.sh:INCLUDES += -I\$(CUDA_PATH)/include
clib-multigpu/genmakefile.sh:# NCCL
clib-multigpu/genmakefile.sh:ifneq (\$(NCCL_PATH),)
clib-multigpu/genmakefile.sh:	INCLUDES += -I\$(NCCL_PATH)/include
clib-multigpu/genmakefile.sh:# CUDA
clib-multigpu/genmakefile.sh:LIBS += -L\$(CUDA_PATH)/lib\$(POSTFIX) -lcudart -lcublas -lcudnn -lcurand -lnvToolsExt
clib-multigpu/genmakefile.sh:# NCCL
clib-multigpu/genmakefile.sh:ifneq (\$(NCCL_PATH),)
clib-multigpu/genmakefile.sh:	LIBS += -L\$(NCCL_PATH)/lib -lnccl
clib-multigpu/genmakefile.sh:	LIBS += -lnccl
clib-multigpu/genmakefile.sh:# With CUDA 9.0 and higher, 20 is no longer supported
clib-multigpu/genmakefile.sh:all: libCPU.so libdataset.so liblightweightdataset.so libGPU.so libBLAS.so libRNG.so librecords.so
clib-multigpu/genmakefile.sh:libGPU.so: GPU.o image/recordreader.o image/recordfile.o image/record.o image/image.o image/boundingbox.o image/rectangle.o image/yarng.o random/generator.o \$(OBJS) \$(KNLS)
clib-multigpu/genmakefile.sh:	\$(NV) \$(LFL) -shared -o libGPU.so GPU.o image/recordreader.o image/recordfile.o image/record.o image/image.o image/boundingbox.o image/rectangle.o image/yarng.o random/generator.o \$(OBJS) \$(KNLS) \$(LIBS)
clib-multigpu/genmakefile.sh:GPU.o: GPU.c uk_ac_imperial_lsds_crossbow_device_TheGPU.h executioncontext.h
clib-multigpu/genmakefile.sh:uk_ac_imperial_lsds_crossbow_device_TheGPU.h:
clib-multigpu/genmakefile.sh:	javah -classpath \$(CLASS_PATH) uk.ac.imperial.lsds.crossbow.device.TheGPU
clib-multigpu/genmakefile.sh:	\$(NV) \$(INCLUDES) \$(LFL) image/testrecordreader.c -o image/testrecordreader -L\$(CBOW_PATH)/clib-multigpu -lGPU -lCPU -lBLAS -lRNG -lrecords -lm \$(LIBS)
clib-multigpu/genmakefile.sh:	\$(NV) \$(INCLUDES) \$(LFL) image/testbatchreader.c -o image/testbatchreader -L\$(CBOW_PATH)/clib-multigpu -lGPU -lCPU -lBLAS -lRNG -lrecords -lm \$(LIBS)
clib-multigpu/genmakefile.sh:	\$(NV) \$(INCLUDES) \$(LFL) testrecorddataset.c -o testrecorddataset -L\$(CBOW_PATH)/clib-multigpu -lGPU -lCPU -lBLAS -lRNG -lrecords -lm \$(LIBS)
clib-multigpu/stream.h:#include <cuda_runtime.h>
clib-multigpu/stream.h:	cudaStream_t *stream;
clib-multigpu/stream.h:	cudaEvent_t event;
clib-multigpu/stream.h:	cudaEvent_t start;
clib-multigpu/stream.h:	cudaEvent_t barrier;
clib-multigpu/stream.h:	cudaStream_t modelSynchronisationStream;
clib-multigpu/model.h:	 * The lock is acquired by the GPU worker thread before a dataflow is scheduled.
clib-multigpu/model.h:	 * the GPU worker thread.
clib-multigpu/model.h:	 * A solution would be to acquire the (read or write) lock, record a CUDA event
clib-multigpu/model.h:	 * the operator's GPU functions the event would fire and then it would be  safe
clib-multigpu/model.h:	 * More locks: GPU events to synchronise with the parameter server (realised as another stream).
clib-multigpu/model.h:	cudaEvent_t *client;
clib-multigpu/model.h:	cudaEvent_t *server;
clib-multigpu/model.h:	cudaEvent_t client;
clib-multigpu/model.h:	cudaEvent_t server;
clib-multigpu/model.h:	cudaEvent_t updated;
clib-multigpu/model.h:	cudaEvent_t accumulated;
clib-multigpu/memoryregion.c:	checkCudaErrors(cudaHostRegister(ptr, length, cudaHostRegisterMapped | cudaHostRegisterPortable));
clib-multigpu/memoryregion.c:	checkCudaErrors(cudaHostUnregister(ptr));
clib-multigpu/lightweightdatasetmanager.h:	/* Use of GPU register or not */
clib-multigpu/lightweightdatasetmanager.h:	unsigned GPU;
clib-multigpu/datasetfilemanager.c:crossbowDatasetFileManagerP crossbowDatasetFileManagerCreate (int size, unsigned gpu, int blocksize) {
clib-multigpu/datasetfilemanager.c:	p->gpu = gpu;
clib-multigpu/datasetfile.c:		checkCudaErrors(cudaHostRegister(ptr, blocksize, cudaHostRegisterMapped | cudaHostRegisterPortable));
clib-multigpu/datasetfile.c:	checkCudaErrors(cudaHostRegister(ptr, length, cudaHostRegisterMapped | cudaHostRegisterPortable));
clib-multigpu/datasetfile.c:		checkCudaErrors(cudaHostUnregister(ptr));
clib-multigpu/datasetfile.c:	checkCudaErrors(cudaHostUnregister(ptr));
clib-multigpu/executioncontext.c: * Based on <helper_cuda.h>
clib-multigpu/executioncontext.c:static void CUDART_CB callback (cudaStream_t stream, cudaError_t error, void *args) {
clib-multigpu/executioncontext.c:#ifdef GPU_VERBOSE
clib-multigpu/executioncontext.c:	dbg ("CUDA stream %p, crossbow stream %p\n", stream, s->stream);
clib-multigpu/executioncontext.c:	checkCudaErrors(error);
clib-multigpu/executioncontext.c:	crossbowCallbackHandlerP handler = crossbowExecutionContextNextCallbackHandler (theGPU);
clib-multigpu/executioncontext.c:	p->mode = (__count > 1) ? MULTI_GPU : SINGLE_GPU;
clib-multigpu/executioncontext.c:#ifdef USE_NCCL
clib-multigpu/executioncontext.c:	/* Configure NCCL library */
clib-multigpu/executioncontext.c:	p->comms = (ncclComm_t *) crossbowMalloc(p->nc * sizeof(ncclComm_t));
clib-multigpu/executioncontext.c:	checkNcclErrors(ncclCommInitAll(p->comms, __count, p->devs));
clib-multigpu/executioncontext.c:	 * Task handlers, once pinned to a CPU core, were able to schedule tasks to any GPU.
clib-multigpu/executioncontext.c:	 * But this did not guarantee optimal GPU affinity.
clib-multigpu/executioncontext.c:	 * These handlers, one per device, will be used by the GPU worker thread, responsible
clib-multigpu/executioncontext.c:		/* Redirect CUDA calls to specific device */
clib-multigpu/executioncontext.c:		checkCudaErrors(cudaSetDevice(dev->id));
clib-multigpu/executioncontext.c:		// checkCudaErrors(cudaEventCreateWithFlags(&(dev->barrier), cudaEventBlockingSync));
clib-multigpu/executioncontext.c:		checkCudaErrors(cudaEventCreateWithFlags(&(dev->barrier), cudaEventDefault));
clib-multigpu/executioncontext.c:		checkCudaErrors(cudaStreamCreateWithFlags(&(dev->modelSynchronisationStream), cudaStreamNonBlocking));
clib-multigpu/executioncontext.c:	checkCudaErrors(cudaSetDevice(p->defaultDeviceId));
clib-multigpu/executioncontext.c:#ifdef GPU_VERBOSE
clib-multigpu/executioncontext.c:	/* All CUDA calls already redirected to specific device */
clib-multigpu/executioncontext.c:		checkCudaErrors(cudaEventRecord(s->barrier, NULL));
clib-multigpu/executioncontext.c:	/* Double-check that there are no pending asynchronous tasks on this CUDA streams */
clib-multigpu/executioncontext.c:	checkCudaErrors(cudaEventQuery(s->event));
clib-multigpu/executioncontext.c:		checkCudaErrors(cudaStreamWaitEvent(s->stream[i], s->barrier, 0));
clib-multigpu/executioncontext.c:	checkCudaErrors(cudaEventRecord(s->start, s->stream[0]));
clib-multigpu/executioncontext.c:		 * checkCudaErrors(cudaEventRecord(s->op->start[s->id], s->stream[s->op->branch]));
clib-multigpu/executioncontext.c:		 * 	checkCudaErrors(cudaStreamWaitEvent(s->stream, s->op->end[ctx->previousStreamId], 0));
clib-multigpu/executioncontext.c:				checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], d->guard->end[s->id], 0));
clib-multigpu/executioncontext.c:				checkCudaErrors(cudaEventRecord(s->op->end[s->id], s->stream[s->op->branch]));
clib-multigpu/executioncontext.c:				checkCudaErrors(cudaStreamWaitEvent(s->stream[0], s->op->end[s->id], 0));
clib-multigpu/executioncontext.c:	checkCudaErrors(cudaEventRecord(s->event, s->stream[0]));
clib-multigpu/executioncontext.c:			checkCudaErrors(cudaEventDestroy(dev->barrier));
clib-multigpu/executioncontext.c:			checkCudaErrors(cudaStreamDestroy(dev->modelSynchronisationStream));
clib-multigpu/executioncontext.c:#ifdef USE_NCCL
clib-multigpu/executioncontext.c:	/* Free NCCL context */
clib-multigpu/executioncontext.c:		ncclCommDestroy(p->comms[i]);
clib-multigpu/executioncontext.c:	crossbowFree(p->comms, p->nc * sizeof(ncclComm_t));
clib-multigpu/executioncontext.c:	cudaDeviceReset();
clib-multigpu/executioncontext.c:	checkCudaErrors (cudaGetDeviceCount (&numberofdevices));
clib-multigpu/executioncontext.c:	if (0 == numberofdevices) err("No CUDA device found\n");
clib-multigpu/executioncontext.c:	/* Configure and get GPU device properties */
clib-multigpu/executioncontext.c:	struct cudaDeviceProp properties;
clib-multigpu/executioncontext.c:		checkCudaErrors(cudaGetDeviceProperties (&properties, dev->id));
clib-multigpu/executioncontext.c:			checkCudaErrors(cudaSetDevice (dev->id));
clib-multigpu/executioncontext.c:			checkCudaErrors(cudaSetDeviceFlags (cudaDeviceScheduleSpin | cudaDeviceMapHost));
clib-multigpu/executioncontext.c:			 * checkCudaErrors(cudaDeviceSetLimit (cudaLimitDevRuntimePendingLaunchCount,  limit));
clib-multigpu/executioncontext.c:			 * checkCudaErrors(cudaDeviceGetLimit (&limit, cudaLimitDevRuntimePendingLaunchCount));
clib-multigpu/executioncontext.c:		/* Copy data to GPU */
clib-multigpu/executioncontext.c:			checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/executioncontext.c:	checkCudaErrors (cudaSetDevice(device->id));
clib-multigpu/executioncontext.c:	checkCudaErrors (cudaSetDevice(device->id));
clib-multigpu/executioncontext.c:	checkCudaErrors (cudaSetDevice(device->id));
clib-multigpu/executioncontext.c:			checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/executioncontext.c:#ifdef GPU_VERBOSE
clib-multigpu/executioncontext.c:	/* Push the model's initialised variables to GPU memory */
clib-multigpu/executioncontext.c:	checkCudaErrors (cudaSetDevice (ctx->theModel->dev));
clib-multigpu/executioncontext.c:	checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/executioncontext.c:	 * aligned (since it has been memory-mapped) and registered against CUDA's address space.
clib-multigpu/executioncontext.c:	checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/executioncontext.c:	 * aligned (since it has been memory-mapped) and registered against CUDA's address space.
clib-multigpu/executioncontext.c:	checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/executioncontext.c:	 * aligned (since it has been memory-mapped) and registered against CUDA's address space.
clib-multigpu/executioncontext.c:	checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/executioncontext.c:	 * aligned (since it has been memory-mapped) and registered against CUDA's address space.
clib-multigpu/executioncontext.c:				fprintf(stderr, "error: failed to lock all GPU model replicas at synchronisation barrier\n");
clib-multigpu/executioncontext.c:	dbg("Synchronise GPU models at clock %d\n", clock);
clib-multigpu/executioncontext.c:		dbg("Synchronise replicas using default model (%s-GPU mode)\n", (ctx->mode == SINGLE_GPU) ? "single" : "multi");
clib-multigpu/executioncontext.c:		dbg("Synchronise replicas using S-SGD model (%s-GPU mode)\n", (ctx->mode == SINGLE_GPU) ? "single" : "multi");
clib-multigpu/executioncontext.c:		dbg("Synchronise replicas using EAMSGD model (%s-GPU mode)\n", (ctx->mode == SINGLE_GPU) ? "single" : "multi");
clib-multigpu/executioncontext.c:		dbg("Synchronise replicas using synchronous EAMSGD model (%s-GPU mode)\n", (ctx->mode == SINGLE_GPU) ? "single" : "multi");
clib-multigpu/executioncontext.c:		dbg("Synchronise replicas using DOWNPOUR model (%s-GPU mode)\n", (ctx->mode == SINGLE_GPU) ? "single" : "multi");
clib-multigpu/executioncontext.c:		dbg("Synchronise replicas using Hogwild! model (%s-GPU mode)\n", (ctx->mode == SINGLE_GPU) ? "single" : "multi");
clib-multigpu/executioncontext.c:		dbg("Synchronise replicas using Polyak-Ruppert model (%s-GPU mode)\n", (ctx->mode == SINGLE_GPU) ? "single" : "multi");
clib-multigpu/executioncontext.c:		dbg("Synchronise replicas using SMA model (%s-GPU mode)\n", (ctx->mode == SINGLE_GPU) ? "single" : "multi");
clib-multigpu/callbackhandler.c:#include <cuda.h>
clib-multigpu/callbackhandler.c:#include <cuda_runtime.h>
clib-multigpu/callbackhandler.c:		checkCudaErrors(cudaEventSynchronize(s->event));
clib-multigpu/callbackhandler.c:		 * while (cudaEventQuery(s->event) != cudaSuccess);
clib-multigpu/callbackhandler.c:		checkCudaErrors(cudaEventElapsedTime (&dt, s->start, s->event));
clib-multigpu/callbackhandler.c:		checkCudaErrors(cudaEventElapsedTime (&dt, s->barrier, s->event));
clib-multigpu/executioncontext.h:#include <cuda.h>
clib-multigpu/executioncontext.h:#include <cuda_runtime.h>
clib-multigpu/executioncontext.h:#include <cuda_runtime_api.h>
clib-multigpu/executioncontext.h:#include <nccl.h>
clib-multigpu/executioncontext.h:#ifdef USE_NCCL
clib-multigpu/executioncontext.h:	ncclComm_t *comms;
clib-multigpu/operator.h:#include <cuda.h>
clib-multigpu/operator.h:#include <cuda_runtime.h>
clib-multigpu/operator.h:#include <cuda_runtime_api.h>
clib-multigpu/operator.h:	cudaEvent_t *start;
clib-multigpu/operator.h:	cudaEvent_t *end;
clib-multigpu/kernels/cudnnpoolgradient.cu:#ifndef CUDART_NOOP
clib-multigpu/kernels/cudnnpoolgradient.cu:	checkCudaErrors (cudaMemsetAsync (output->dev, 0, s->op->kernel->output->schema->bytes, s->stream[s->op->branch]));
clib-multigpu/kernels/cudnnpoolgradient.cu:#ifndef CUDANN_NOOP
clib-multigpu/kernels/cudnnrelugradient.cu:#ifndef CUDANN_NOOP
clib-multigpu/kernels/softmaxgradient.cu:#ifndef CUDART_NOOP
clib-multigpu/kernels/softmaxgradient.cu:	cudaMemcpyAsync (output->dev, input->dev, length, cudaMemcpyDeviceToDevice, s->stream[s->op->branch]);
clib-multigpu/kernels/gradientdescentoptimiser.cu:		case SINGLE_GPU:
clib-multigpu/kernels/gradientdescentoptimiser.cu:		case  MULTI_GPU:
clib-multigpu/kernels/gradientdescentoptimiser.cu:		case SINGLE_GPU:
clib-multigpu/kernels/gradientdescentoptimiser.cu:		case  MULTI_GPU:
clib-multigpu/kernels/gradientdescentoptimiser.cu:		case SINGLE_GPU:
clib-multigpu/kernels/gradientdescentoptimiser.cu:		case MULTI_GPU:
clib-multigpu/kernels/gradientdescentoptimiser.cu:		case SINGLE_GPU:
clib-multigpu/kernels/gradientdescentoptimiser.cu:		case MULTI_GPU:
clib-multigpu/kernels/gradientdescentoptimiser.cu:		case SINGLE_GPU:
clib-multigpu/kernels/gradientdescentoptimiser.cu:		case MULTI_GPU:
clib-multigpu/kernels/gradientdescentoptimiser.cu:		case SINGLE_GPU:
clib-multigpu/kernels/gradientdescentoptimiser.cu:		case MULTI_GPU:
clib-multigpu/kernels/gradientdescentoptimiser.cu:		case SINGLE_GPU:
clib-multigpu/kernels/gradientdescentoptimiser.cu:		case MULTI_GPU:
clib-multigpu/kernels/gradientdescentoptimiser.cu:		case SINGLE_GPU:
clib-multigpu/kernels/gradientdescentoptimiser.cu:		case MULTI_GPU:
clib-multigpu/kernels/cudnnconv.cu:#include <nvToolsExtCuda.h>
clib-multigpu/kernels/cudnnconv.cu:#include <nvToolsExtCudaRt.h>
clib-multigpu/kernels/cudnnconv.cu:	/* The GPU worker should wait for the model to be updated with the latest gradients */
clib-multigpu/kernels/cudnnconv.cu:	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->updated, 0));
clib-multigpu/kernels/cudnnconv.cu:#ifdef GPU_VERBOSE
clib-multigpu/kernels/cudnnconv.cu:#ifndef CUDANN_NOOP
clib-multigpu/kernels/innerproduct.cu:	/* The GPU worker should wait for the model to be updated with the latest gradients */
clib-multigpu/kernels/innerproduct.cu:	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->updated, 0));
clib-multigpu/kernels/cudnnsoftmaxgradient.cu:#ifndef CUDART_NOOP
clib-multigpu/kernels/cudnnsoftmaxgradient.cu:    /* checkCudaErrors (cudaMemsetAsync (output->dev, 0, s->op->kernel->output->schema->bytes, s->stream[s->op->branch])); */
clib-multigpu/kernels/cudnnsoftmaxgradient.cu:#ifndef CUDANN_NOOP
clib-multigpu/kernels/matfact.cu:	CUDA_KERNEL_LOOP(ndx, cells) {
clib-multigpu/kernels/matfact.cu:	} /* End of CUDA_KERNEL_LOOP */
clib-multigpu/kernels/matfact.cu:	/* The GPU worker should wait for the model to be updated with the latest gradients */
clib-multigpu/kernels/matfact.cu:	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->updated, 0));
clib-multigpu/kernels/matfact.cu:	/* The GPU worker should wait for the application of the previously computed gradient (if any)
clib-multigpu/kernels/matfact.cu:	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->server[s->op->id], 0));
clib-multigpu/kernels/matfact.cu:	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->server, 0));
clib-multigpu/kernels/matfact.cu:	cudaMemsetAsync (s->model->gradient->dev, 0, s->model->bytes, s->stream[s->op->branch]);
clib-multigpu/kernels/matfact.cu:	checkCudaErrors(cudaMemsetAsync(output->dev, 0, s->op->kernel->output->schema->bytes, s->stream[s->op->branch]));
clib-multigpu/kernels/matfact.cu:	checkCudaErrors(cudaMemsetAsync(losses->dev, 0, losses_length, s->stream[s->op->branch]));
clib-multigpu/kernels/matfact.cu:	crossbowKernelMatFactCompute<<<GET_BLOCKS(cells), CUDA_NUM_THREADS, 0, s->stream[s->op->branch]>>>(
clib-multigpu/kernels/matfact.cu:	checkCudaErrors(cudaEventRecord(s->model->client[s->op->id], s->stream[s->op->branch]));
clib-multigpu/kernels/matfact.cu:	checkCudaErrors(cudaEventRecord(s->model->client, s->stream[s->op->branch]));
clib-multigpu/kernels/matfact.cu:	checkCudaErrors(cudaStreamWaitEvent(s->modelSynchronisationStream, s->model->client[s->op->id], 0));
clib-multigpu/kernels/matfact.cu:	checkCudaErrors(cudaStreamWaitEvent(s->modelSynchronisationStream, s->model->client, 0));
clib-multigpu/kernels/matfact.cu:	checkCudaErrors(cudaEventRecord(s->model->server[s->op->id], s->modelSynchronisationStream));
clib-multigpu/kernels/matfact.cu:	cudaEventSynchronize (s->model->server[s->op->id]);
clib-multigpu/kernels/matfact.cu:	checkCudaErrors(cudaEventRecord(s->model->server, s->modelSynchronisationStream));
clib-multigpu/kernels/matfact.cu:	cudaEventSynchronize (s->model->server);
clib-multigpu/kernels/datatransform.cu:	/* cudaMemsetAsync (((void *) output->dev), 0, 4 * examples * channels * outputImageHeight * outputImageWidth, s->stream); */
clib-multigpu/kernels/concat.cu:	CUDA_KERNEL_LOOP(index, count) {
clib-multigpu/kernels/concat.cu:#ifndef CUDART_NOOP
clib-multigpu/kernels/concat.cu:	checkCudaErrors (cudaMemsetAsync (output->dev, 0, s->op->kernel->output->schema->bytes, s->stream[s->op->branch]));
clib-multigpu/kernels/concat.cu:		crossbowKernelConcatCompute<<< GET_BLOCKS(threads), CUDA_NUM_THREADS, 0, s->stream[s->op->branch] >>>(
clib-multigpu/kernels/softmaxlossgradient.cu:	CUDA_KERNEL_LOOP(index, nthreads) {
clib-multigpu/kernels/softmaxlossgradient.cu:#ifndef CUDART_NOOP
clib-multigpu/kernels/softmaxlossgradient.cu:	cudaMemcpyAsync (output->dev, peer_input->dev, peer_input_length, cudaMemcpyDeviceToDevice, s->stream[s->op->branch]);
clib-multigpu/kernels/softmaxlossgradient.cu:	crossbowKernelSoftMaxLossGradientCompute<<<GET_BLOCKS(nthreads), CUDA_NUM_THREADS, 0, s->stream[s->op->branch]>>>(
clib-multigpu/kernels/cudnnconvgradient.cu:	 * The GPU worker should wait for the application of the previously computed gradient (if any)
clib-multigpu/kernels/cudnnconvgradient.cu:	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->server[s->op->peer->id], 0));
clib-multigpu/kernels/cudnnconvgradient.cu:	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->server, 0));
clib-multigpu/kernels/cudnnconvgradient.cu:#ifndef CUDART_NOOP
clib-multigpu/kernels/cudnnconvgradient.cu:	cudaMemsetAsync ((void *)(((char *) weightGradient->dev) + weightGradientOffset), 0, weightGradientLength + biasGradientLength, s->stream[s->op->branch]);
clib-multigpu/kernels/cudnnconvgradient.cu:#ifndef CUDANN_NOOP
clib-multigpu/kernels/cudnnconvgradient.cu:#ifndef CUDANN_NOOP
clib-multigpu/kernels/cudnnconvgradient.cu:#ifndef CUDANN_NOOP
clib-multigpu/kernels/macros.h:#ifndef __CROSSBOW_CUDA_MACROS_H_
clib-multigpu/kernels/macros.h:#define __CROSSBOW_CUDA_MACROS_H_
clib-multigpu/kernels/macros.h:#define CUDA_KERNEL_LOOP(ndx, max) \
clib-multigpu/kernels/macros.h:#if __CUDA_ARCH__ >= 200
clib-multigpu/kernels/macros.h:	const int CUDA_NUM_THREADS = 1024;
clib-multigpu/kernels/macros.h:	const int CUDA_NUM_THREADS = 512;
clib-multigpu/kernels/macros.h:inline int GET_BLOCKS (const int n) { return ((n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS); }
clib-multigpu/kernels/macros.h:#endif /* __CROSSBOW_CUDA_MACROS_H_ */
clib-multigpu/kernels/elementwiseopgradient.cu:#ifndef CUDART_NOOP
clib-multigpu/kernels/elementwiseopgradient.cu:		checkCudaErrors (cudaMemsetAsync (
clib-multigpu/kernels/cudnnsoftmax.cu:#ifndef CUDANN_NOOP
clib-multigpu/kernels/cudnnbatchnormgradient.cu:	 * The GPU worker should wait for the application of the previously computed gradient (if any)
clib-multigpu/kernels/cudnnbatchnormgradient.cu:	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->server[s->op->peer->id], 0));
clib-multigpu/kernels/cudnnbatchnormgradient.cu:	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->server, 0));
clib-multigpu/kernels/cudnnbatchnormgradient.cu:#ifndef CUDART_NOOP
clib-multigpu/kernels/cudnnbatchnormgradient.cu:	cudaMemsetAsync ((void *)(((char *) scaleGradient->dev) + scaleGradientOffset), 0, scaleGradientLength + biasGradientLength, s->stream[s->op->branch]);
clib-multigpu/kernels/cudnnbatchnormgradient.cu:#ifndef CUDANN_NOOP
clib-multigpu/kernels/cudnndropoutgradient.cu:#ifdef GPU_VERBOSE
clib-multigpu/kernels/cudnndropoutgradient.cu:#ifndef CUDANN_NOOP
clib-multigpu/kernels/cudnnbatchnorm.cu:	/* The GPU worker should wait for the model to be updated with the latest gradients */
clib-multigpu/kernels/cudnnbatchnorm.cu:	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->updated, 0));
clib-multigpu/kernels/cudnnbatchnorm.cu:#ifdef GPU_VERBOSE
clib-multigpu/kernels/cudnnbatchnorm.cu:#ifndef CUDANN_NOOP
clib-multigpu/kernels/cudnnbatchnorm.cu:#ifndef CUDANN_NOOP
clib-multigpu/kernels/classify.cu:	CUDA_KERNEL_LOOP(index, elements) {
clib-multigpu/kernels/classify.cu:	crossbowKernelClassifyCompute<<<GET_BLOCKS(elements), CUDA_NUM_THREADS, 0, s->stream[s->op->branch]>>>
clib-multigpu/kernels/concatgradient.cu:	CUDA_KERNEL_LOOP(index, count) {
clib-multigpu/kernels/concatgradient.cu:#ifndef CUDART_NOOP
clib-multigpu/kernels/concatgradient.cu:	checkCudaErrors (cudaMemsetAsync (output->dev, 0, s->op->kernel->output->schema->bytes, s->stream[s->op->branch]));
clib-multigpu/kernels/concatgradient.cu:	crossbowKernelConcatGradientCompute<<< GET_BLOCKS(threads), CUDA_NUM_THREADS, 0, s->stream[s->op->branch] >>>(
clib-multigpu/kernels/softmaxloss.cu:	CUDA_KERNEL_LOOP(index, nthreads) {
clib-multigpu/kernels/softmaxloss.cu:	/* struct cudaPointerAttributes attributes; */
clib-multigpu/kernels/softmaxloss.cu:	checkCudaErrors(cudaPointerGetAttributes (&attributes, labels->dev));
clib-multigpu/kernels/softmaxloss.cu:	crossbowKernelSoftMaxLossCompute<<<GET_BLOCKS(nthreads), CUDA_NUM_THREADS, 0, s->stream[s->op->branch]>>>(
clib-multigpu/kernels/softmaxloss.cu:	checkCudaErrors(cudaPointerGetAttributes (&attributes, output->dev));
clib-multigpu/kernels/softmaxloss.cu:    checkCudaErrors(cudaPointerGetAttributes (&attributes, losses->dev));
clib-multigpu/kernels/accuracy.cu:	CUDA_KERNEL_LOOP(index, elements) {
clib-multigpu/kernels/accuracy.cu:	crossbowKernelAccuracyCompute<<<GET_BLOCKS(elements), CUDA_NUM_THREADS, 0, s->stream[s->op->branch]>>>
clib-multigpu/kernels/noopstateless.cu:#ifndef CUDART_NOOP
clib-multigpu/kernels/noopstateless.cu:	cudaMemcpyAsync (output->dev, input->dev, length, cudaMemcpyDeviceToDevice, s->stream[s->op->branch]);
clib-multigpu/kernels/cudnndropout.cu:#ifdef GPU_VERBOSE
clib-multigpu/kernels/cudnndropout.cu:#ifndef CUDART_NOOP
clib-multigpu/kernels/cudnndropout.cu:		cudaMemcpyAsync (output->dev, input->dev, length, cudaMemcpyDeviceToDevice, s->stream[s->op->branch]);
clib-multigpu/kernels/cudnndropout.cu:#ifndef CUDANN_NOOP
clib-multigpu/kernels/cudnnpool.cu:#ifdef GPU_VERBOSE
clib-multigpu/kernels/cudnnpool.cu:#ifndef CUDANN_NOOP
clib-multigpu/kernels/optimisers/synchronouseamsgd.cu:	checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/kernels/optimisers/synchronouseamsgd.cu:		checkCudaErrors(cudaMemcpyAsync(last->dev, gradient->dev, s->model->bytes, cudaMemcpyDeviceToDevice, s->stream[s->op->branch]));
clib-multigpu/kernels/optimisers/synchronouseamsgd.cu:		checkCudaErrors(cudaMemcpyAsync(s->model->diff->dev, model->dev, s->model->bytes, cudaMemcpyDeviceToDevice, s->stream[s->op->branch]));
clib-multigpu/kernels/optimisers/synchronouseamsgd.cu:		checkCudaErrors(cudaEventRecord (s->model->client, s->stream[s->op->branch]));
clib-multigpu/kernels/optimisers/synchronouseamsgd.cu:		checkCudaErrors(cudaStreamWaitEvent(s->modelSynchronisationStream, s->model->client, 0));
clib-multigpu/kernels/optimisers/synchronouseamsgd.cu:		checkCudaErrors(cudaEventRecord(s->model->server, s->modelSynchronisationStream));
clib-multigpu/kernels/optimisers/synchronouseamsgd.cu:		checkCudaErrors(cudaMemcpyAsync(s->model->diff->dev, model->dev, s->model->bytes, cudaMemcpyDeviceToDevice, s->stream[s->op->branch]));
clib-multigpu/kernels/optimisers/synchronouseamsgd.cu:		checkCudaErrors(cudaEventRecord (s->model->client, s->stream[s->op->branch]));
clib-multigpu/kernels/optimisers/synchronouseamsgd.cu:		checkCudaErrors(cudaStreamWaitEvent(s->modelSynchronisationStream, s->model->client, 0));
clib-multigpu/kernels/optimisers/synchronouseamsgd.cu:		checkCudaErrors(cudaEventRecord(s->model->server, s->modelSynchronisationStream));
clib-multigpu/kernels/optimisers/default.cu:		checkCudaErrors(cudaEventRecord (s->model->client, s->stream[s->op->branch]));
clib-multigpu/kernels/optimisers/default.cu:		checkCudaErrors(cudaMemcpyAsync(
clib-multigpu/kernels/optimisers/default.cu:			cudaMemcpyDeviceToDevice, 
clib-multigpu/kernels/optimisers/default.cu:		checkCudaErrors(cudaStreamWaitEvent(s->modelSynchronisationStream, s->model->client, 0));
clib-multigpu/kernels/optimisers/default.cu:		checkCudaErrors(cudaEventRecord(s->model->server, s->modelSynchronisationStream));
clib-multigpu/kernels/optimisers/default.cu:		checkCudaErrors(cudaEventRecord (s->model->client, s->stream[s->op->branch]));
clib-multigpu/kernels/optimisers/default.cu:		checkCudaErrors(cudaStreamWaitEvent(s->modelSynchronisationStream, s->model->client, 0));
clib-multigpu/kernels/optimisers/default.cu:		checkCudaErrors(cudaEventRecord(s->model->server, s->modelSynchronisationStream));
clib-multigpu/kernels/optimisers/sma.cu:	 * checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/kernels/optimisers/sma.cu:		checkCudaErrors(cudaMemcpyAsync(last->dev, gradient->dev, s->model->bytes, cudaMemcpyDeviceToDevice, s->stream[s->op->branch]));
clib-multigpu/kernels/optimisers/sma.cu:		checkCudaErrors(cudaMemcpyAsync(s->model->diff->dev, model->dev, s->model->bytes, cudaMemcpyDeviceToDevice, s->stream[s->op->branch]));
clib-multigpu/kernels/optimisers/sma.cu:		checkCudaErrors(cudaEventRecord (s->model->client, s->stream[s->op->branch]));
clib-multigpu/kernels/optimisers/sma.cu:		checkCudaErrors(cudaStreamWaitEvent(s->modelSynchronisationStream, s->model->client, 0));
clib-multigpu/kernels/optimisers/sma.cu:		checkCudaErrors(cudaEventRecord(s->model->server, s->modelSynchronisationStream));
clib-multigpu/kernels/optimisers/sma.cu:		checkCudaErrors(cudaMemcpyAsync(s->model->diff->dev, model->dev, s->model->bytes, cudaMemcpyDeviceToDevice, s->stream[s->op->branch]));
clib-multigpu/kernels/optimisers/sma.cu:		checkCudaErrors(cudaEventRecord (s->model->client, s->stream[s->op->branch]));
clib-multigpu/kernels/optimisers/sma.cu:		checkCudaErrors(cudaStreamWaitEvent(s->modelSynchronisationStream, s->model->client, 0));
clib-multigpu/kernels/optimisers/sma.cu:		checkCudaErrors(cudaEventRecord(s->model->server, s->modelSynchronisationStream));
clib-multigpu/kernels/optimisers/synchronoussgd.cu:	 * checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/kernels/optimisers/synchronoussgd.cu:	checkCudaErrors(cudaEventRecord (s->model->client, s->stream[s->op->branch]));
clib-multigpu/kernels/optimisers/synchronoussgd.cu:	checkCudaErrors(cudaStreamWaitEvent(s->modelSynchronisationStream, s->model->client, 0));
clib-multigpu/kernels/optimisers/synchronoussgd.cu:	checkCudaErrors(cudaEventRecord(s->model->server, s->modelSynchronisationStream));
clib-multigpu/kernels/elementwiseop.cu:#ifndef CUDART_NOOP
clib-multigpu/kernels/elementwiseop.cu:    checkCudaErrors (cudaMemsetAsync (
clib-multigpu/kernels/crossbow.h:#include <cuda_runtime.h>
clib-multigpu/kernels/cudnnrelu.cu:#ifndef CUDANN_NOOP
clib-multigpu/kernels/innerproductgradient.cu:	 * The GPU worker should wait for the application of the previously computed gradient (if any)
clib-multigpu/kernels/innerproductgradient.cu:	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->server[s->op->peer->id], 0));
clib-multigpu/kernels/innerproductgradient.cu:	checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], s->model->server, 0));
clib-multigpu/device.h:#include <cuda.h>
clib-multigpu/device.h:#include <cuda_runtime.h>
clib-multigpu/device.h:#include <cuda_runtime_api.h>
clib-multigpu/device.h:	cudaStream_t   modelSynchronisationStream;
clib-multigpu/device.h:	cudaEvent_t barrier;
clib-multigpu/recorddataset.c:#ifdef GPU_VERBOSE	
clib-multigpu/operator.c:	p->start = crossbowMalloc (p->events * sizeof(cudaEvent_t));
clib-multigpu/operator.c:	p->end   = crossbowMalloc (p->events * sizeof(cudaEvent_t));
clib-multigpu/operator.c:		// checkCudaErrors(cudaEventCreateWithFlags(&(p->start[i]),  cudaEventBlockingSync | cudaEventDisableTiming));
clib-multigpu/operator.c:		// checkCudaErrors(cudaEventCreateWithFlags(&(p->end  [i]),  cudaEventBlockingSync | cudaEventDisableTiming));
clib-multigpu/operator.c:		checkCudaErrors(cudaEventCreateWithFlags(&(p->start[i]),  cudaEventDisableTiming));
clib-multigpu/operator.c:		checkCudaErrors(cudaEventCreateWithFlags(&(p->end  [i]),  cudaEventDisableTiming));
clib-multigpu/operator.c:			checkCudaErrors(cudaEventDestroy(p->start[i]));
clib-multigpu/operator.c:			checkCudaErrors(cudaEventDestroy(p->end  [i]));
clib-multigpu/operator.c:		crossbowFree (p->start, (p->events * sizeof(cudaEvent_t)));
clib-multigpu/operator.c:		crossbowFree (p->end,   (p->events * sizeof(cudaEvent_t)));
clib-multigpu/utils.h:#define USE_NCCL
clib-multigpu/utils.h:/* #undef USE_NCCL */
clib-multigpu/utils.h:typedef enum crossbow_model_synchronisation_mode_type { SINGLE_GPU = 0, MULTI_GPU } crossbowModelSynchronisationMode_t;
clib-multigpu/datasetfileblock.h:	unsigned gpu;
clib-multigpu/memorymanager.c: * b) cudaMallocHost
clib-multigpu/memorymanager.c: * c) cudaMalloc
clib-multigpu/memorymanager.c: * b) cudaFreeHost
clib-multigpu/memorymanager.c: * c) cudaFree
clib-multigpu/memorymanager.c:static long cudaMallocHostCalls = 0L;
clib-multigpu/memorymanager.c:static long cudaMallocCalls = 0L;
clib-multigpu/memorymanager.c:static long cudaFreeHostCalls = 0L;
clib-multigpu/memorymanager.c:static long cudaFreeCalls = 0L;
clib-multigpu/memorymanager.c:static long cudaMallocHostBytes = 0L;
clib-multigpu/memorymanager.c:static long cudaMallocBytes = 0L;
clib-multigpu/memorymanager.c:void *crossbowCudaMallocHost (int size) {
clib-multigpu/memorymanager.c:	checkCudaErrors(cudaMallocHost (&p, size));
clib-multigpu/memorymanager.c:	/* checkCudaErrors(cudaHostAlloc (&p, size, cudaHostAllocPortable)); */
clib-multigpu/memorymanager.c:	cudaMallocHostBytes += size;
clib-multigpu/memorymanager.c:	cudaMallocHostCalls ++;
clib-multigpu/memorymanager.c:	__sync_add_and_fetch (&(cudaMallocHostBytes), size);
clib-multigpu/memorymanager.c:	__sync_add_and_fetch (&(cudaMallocHostCalls),    1);
clib-multigpu/memorymanager.c:void *crossbowCudaMalloc (int size) {
clib-multigpu/memorymanager.c:	checkCudaErrors(cudaMalloc (&p, size));
clib-multigpu/memorymanager.c:	cudaMallocBytes += size;
clib-multigpu/memorymanager.c:	cudaMallocCalls ++;
clib-multigpu/memorymanager.c:	__sync_add_and_fetch (&(cudaMallocBytes), size);
clib-multigpu/memorymanager.c:	__sync_add_and_fetch (&(cudaMallocCalls),    1);
clib-multigpu/memorymanager.c:void *crossbowCudaFreeHost (void *item, int size) {
clib-multigpu/memorymanager.c:		checkCudaErrors(cudaFreeHost (item));
clib-multigpu/memorymanager.c:		cudaMallocHostBytes -= size;
clib-multigpu/memorymanager.c:		cudaFreeHostCalls ++;
clib-multigpu/memorymanager.c:		__sync_sub_and_fetch (&(cudaMallocHostBytes), size);
clib-multigpu/memorymanager.c:		__sync_add_and_fetch (&(cudaFreeHostCalls),      1);
clib-multigpu/memorymanager.c:void *crossbowCudaFree (void *item, int size) {
clib-multigpu/memorymanager.c:		checkCudaErrors(cudaFree (item));
clib-multigpu/memorymanager.c:		cudaMallocBytes -= size;
clib-multigpu/memorymanager.c:		cudaFreeCalls ++;
clib-multigpu/memorymanager.c:		__sync_sub_and_fetch (&(cudaMallocBytes), size);
clib-multigpu/memorymanager.c:		__sync_add_and_fetch (&(cudaFreeCalls),      1);
clib-multigpu/memorymanager.c:	struct cudaPointerAttributes attributes;
clib-multigpu/memorymanager.c:		if (cudaPointerGetAttributes (&attributes, p) != cudaSuccess) {
clib-multigpu/memorymanager.c:			cudaGetLastError ();
clib-multigpu/memorymanager.c:	cudaMemGetInfo(&free, &total);
clib-multigpu/memorymanager.c:			cudaMallocHostBytes, cudaMallocHostCalls, cudaFreeHostCalls);
clib-multigpu/memorymanager.c:			cudaMallocBytes, cudaMallocCalls, cudaFreeCalls, free, total);
clib-multigpu/datasetfilehandler.c:                if (block->gpu)
clib-multigpu/datasetfilehandler.c:                if (block->gpu)
clib-multigpu/datasetfilehandler.c:				if (block->gpu)
clib-multigpu/datasetfilehandler.c:				if (block->gpu)
clib-multigpu/variable.c:void crossbowVariablePush (crossbowVariableP p, cudaStream_t stream) {
clib-multigpu/modelmanager.h:#include <cuda.h>
clib-multigpu/modelmanager.h:	cudaEvent_t *synched;
clib-multigpu/modelmanager.c:	checkCudaErrors (cudaSetDevice(p->theModel->dev));
clib-multigpu/modelmanager.c:		checkCudaErrors (cudaSetDevice(p->replicas[i]->dev));
clib-multigpu/modelmanager.c:	checkCudaErrors (cudaSetDevice(p->theModel->dev));
clib-multigpu/modelmanager.c:    //p->synched = (cudaEvent_t *) crossbowMalloc (numberofdevices * sizeof(cudaEvent_t));
clib-multigpu/modelmanager.c:    //    checkCudaErrors(cudaEventCreateWithFlags(&(p->synched[i]), cudaEventDisableTiming));
clib-multigpu/modelmanager.c:    //    // checkCudaErrors(cudaEventCreateWithFlags(&(p->synched[i]), cudaEventBlockingSync | cudaEventDisableTiming));
clib-multigpu/modelmanager.c:     * (See use of NCCL library is executioncontext.c)
clib-multigpu/modelmanager.c:	p->synched = (cudaEvent_t *) crossbowMalloc (numberofdevices * sizeof(cudaEvent_t));
clib-multigpu/modelmanager.c:		checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/modelmanager.c:		checkCudaErrors(cudaEventCreateWithFlags(&(p->synched[dev->id]), cudaEventBlockingSync | cudaEventDisableTiming));
clib-multigpu/modelmanager.c:	checkCudaErrors (cudaSetDevice(p->theModel->dev));
clib-multigpu/modelmanager.c:		/* Redirect CUDA calls to model device */
clib-multigpu/modelmanager.c:		checkCudaErrors (cudaSetDevice(model->dev));
clib-multigpu/modelmanager.c:		prefix = crossbowStringConcat ("%s/gpu-%02d-theModel", dir, model->dev);
clib-multigpu/modelmanager.c:		/* Redirect CUDA calls to model device */
clib-multigpu/modelmanager.c:		checkCudaErrors (cudaSetDevice(p->replicas[id]->dev));
clib-multigpu/modelmanager.c:		prefix = crossbowStringConcat ("%s/gpu-%02d-replica-%03d", dir, p->replicas[id]->dev, id);
clib-multigpu/modelmanager.c:		/* Redirect CUDA calls to model device */
clib-multigpu/modelmanager.c:		checkCudaErrors (cudaSetDevice(model->dev));
clib-multigpu/modelmanager.c:		prefix = crossbowStringConcat ("%s/gpu-%02d-theModel", dir, model->dev);
clib-multigpu/modelmanager.c:		/* Redirect CUDA calls to model device */
clib-multigpu/modelmanager.c:		checkCudaErrors (cudaSetDevice(p->replicas[id]->dev));
clib-multigpu/modelmanager.c:		prefix = crossbowStringConcat ("%s/gpu-%02d-replica-%03d", dir, p->replicas[id]->dev, id);
clib-multigpu/modelmanager.c: * 3. The GPU worker attempts to update the clock of one
clib-multigpu/modelmanager.c:		checkCudaErrors (cudaSetDevice (dev));
clib-multigpu/modelmanager.c:		checkCudaErrors (cudaDeviceSynchronize ());
clib-multigpu/modelmanager.c:		checkCudaErrors (cudaDeviceSynchronize ());
clib-multigpu/modelmanager.c:#ifdef GPU_VERBOSE
clib-multigpu/modelmanager.c:#ifdef GPU_VERBOSE
clib-multigpu/modelmanager.c:		checkCudaErrors (cudaSetDevice (dev));
clib-multigpu/modelmanager.c:		checkCudaErrors (cudaDeviceSynchronize ());
clib-multigpu/modelmanager.c:#ifdef GPU_VERBOSE
clib-multigpu/modelmanager.c:	/* Free multi-GPU synchronisation events */
clib-multigpu/modelmanager.c:	crossbowFree (p->synched, (size * sizeof(cudaEvent_t)));
clib-multigpu/cudnn/cudnnbatchnormparams.c:	p->ready = (cudaEvent_t *) crossbowMalloc (p->replicas * sizeof(cudaEvent_t));
clib-multigpu/cudnn/cudnnbatchnormparams.c:	memset (p->ready, 0, (p->replicas * sizeof(cudaEvent_t)));
clib-multigpu/cudnn/cudnnbatchnormparams.c:		checkCudaErrors(cudaMemset (buffer->dev, 0, buffer->size));
clib-multigpu/cudnn/cudnnbatchnormparams.c:		/* Copy data to GPU buffer */
clib-multigpu/cudnn/cudnnbatchnormparams.c:		checkCudaErrors(cudaMemcpy (buffer->dev, (void *) data, buffer->size, cudaMemcpyHostToDevice));
clib-multigpu/cudnn/cudnnbatchnormparams.c: * Assumes that all CUDA calls have been directed to device `ndx`.
clib-multigpu/cudnn/cudnnbatchnormparams.c:	// checkCudaErrors(cudaEventCreateWithFlags(&(p->ready[ndx]), cudaEventBlockingSync | cudaEventDisableTiming));
clib-multigpu/cudnn/cudnnbatchnormparams.c:	checkCudaErrors(cudaEventCreateWithFlags(&(p->ready[ndx]), cudaEventDisableTiming));
clib-multigpu/cudnn/cudnnbatchnormparams.c:void crossbowCudnnBatchNormParamsGetEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP p, int ndx, cudaStream_t stream, crossbowDataBufferP *m, crossbowDataBufferP *v, int *f) {
clib-multigpu/cudnn/cudnnbatchnormparams.c:	checkCudaErrors(cudaStreamWaitEvent(stream, p->ready[ndx], 0));
clib-multigpu/cudnn/cudnnbatchnormparams.c:		/* Redirect CUDA calls to corresponding device */
clib-multigpu/cudnn/cudnnbatchnormparams.c:		checkCudaErrors (cudaSetDevice(id));
clib-multigpu/cudnn/cudnnbatchnormparams.c:		char *f = crossbowStringConcat ("%s/gpu-%02d-bn-avg-%03d.dat", dir, id, op);
clib-multigpu/cudnn/cudnnbatchnormparams.c:		char *g = crossbowStringConcat ("%s/gpu-%02d-bn-var-%03d.dat", dir, id, op);
clib-multigpu/cudnn/cudnnbatchnormparams.c:		/* Redirect CUDA calls to corresponding device */
clib-multigpu/cudnn/cudnnbatchnormparams.c:		checkCudaErrors (cudaSetDevice(id));
clib-multigpu/cudnn/cudnnbatchnormparams.c:		char *f = crossbowStringConcat ("%s/gpu-%02d-bn-avg-%03d.dat", dir, id, op);
clib-multigpu/cudnn/cudnnbatchnormparams.c:		char *g = crossbowStringConcat ("%s/gpu-%02d-bn-var-%03d.dat", dir, id, op);
clib-multigpu/cudnn/cudnnbatchnormparams.c:void crossbowCudnnBatchNormParamsReleaseEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP p, int ndx, cudaStream_t stream) {
clib-multigpu/cudnn/cudnnbatchnormparams.c:	checkCudaErrors(cudaEventRecord (p->ready[ndx], stream));
clib-multigpu/cudnn/cudnnbatchnormparams.c:	checkCudaErrors (cudaSetDevice(defaultDev->id));
clib-multigpu/cudnn/cudnnbatchnormparams.c:	checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/cudnn/cudnnbatchnormparams.c:			cudaMemcpyPeerAsync (    p->_mean->dev, defaultDev->id, m->dev, id, M->size, defaultDev->modelSynchronisationStream);
clib-multigpu/cudnn/cudnnbatchnormparams.c:			cudaMemcpyPeerAsync (p->_variance->dev, defaultDev->id, v->dev, id, V->size, defaultDev->modelSynchronisationStream);
clib-multigpu/cudnn/cudnnbatchnormparams.c:			cudaMemcpyPeerAsync (m->dev, id, M->dev, defaultDev->id, M->size, defaultDev->modelSynchronisationStream);
clib-multigpu/cudnn/cudnnbatchnormparams.c:			cudaMemcpyPeerAsync (v->dev, id, V->dev, defaultDev->id, V->size, defaultDev->modelSynchronisationStream);
clib-multigpu/cudnn/cudnnbatchnormparams.c:	checkCudaErrors(cudaDeviceSynchronize());
clib-multigpu/cudnn/cudnnbatchnormparams.c:			checkCudaErrors(cudaEventDestroy(p->ready[ndx]));
clib-multigpu/cudnn/cudnnbatchnormparams.c:	crossbowFree (p->ready, (p->replicas * sizeof(cudaEvent_t)));
clib-multigpu/cudnn/cudnnconvparams.c:#ifdef GPU_VERBOSE
clib-multigpu/cudnn/cudnnconvparams.c:#ifdef GPU_VERBOSE
clib-multigpu/cudnn/cudnnconvparams.c:#ifdef GPU_VERBOSE
clib-multigpu/cudnn/cudnnbatchnormparams.h:	cudaEvent_t *ready;
clib-multigpu/cudnn/cudnnbatchnormparams.h:void crossbowCudnnBatchNormParamsGetEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP, int, cudaStream_t, crossbowDataBufferP *, crossbowDataBufferP *, int *);
clib-multigpu/cudnn/cudnnbatchnormparams.h:void crossbowCudnnBatchNormParamsReleaseEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP, int, cudaStream_t);
clib-multigpu/cudnn/cudnnreluparams.c:	 * Note: CUDA 6.0 supports ELU also.
clib-multigpu/cudnn/cudnnrnnparams.c:	p->ready = (cudaEvent_t *) crossbowMalloc (p->replicas * sizeof(cudaEvent_t));
clib-multigpu/cudnn/cudnnrnnparams.c:	memset (p->ready, 0, (p->replicas * sizeof(cudaEvent_t)));
clib-multigpu/cudnn/cudnnrnnparams.c: * Assumes that all CUDA calls have been directed to device `ndx`.
clib-multigpu/cudnn/cudnnrnnparams.c:	checkCudaErrors(cudaEventCreateWithFlags(&(p->ready[ndx]), cudaEventBlockingSync | cudaEventDisableTiming));
clib-multigpu/cudnn/cudnnrnnparams.c:void crossbowCudnnBatchNormParamsGetEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP p, int ndx, cudaStream_t stream, crossbowDataBufferP *m, crossbowDataBufferP *v, int *f) {
clib-multigpu/cudnn/cudnnrnnparams.c:	checkCudaErrors(cudaStreamWaitEvent(stream, p->ready[ndx], 0));
clib-multigpu/cudnn/cudnnrnnparams.c:void crossbowCudnnBatchNormParamsReleaseEstimatedMeanAndVariable (crossbowCudnnBatchNormParamsP p, int ndx, cudaStream_t stream) {
clib-multigpu/cudnn/cudnnrnnparams.c:	checkCudaErrors(cudaEventRecord (p->ready[ndx], stream));
clib-multigpu/cudnn/cudnnrnnparams.c: * and that all CUDA calls have been directed to the default device.
clib-multigpu/cudnn/cudnnrnnparams.c:	crossbowFree (p->ready, (p->replicas * sizeof(cudaEvent_t)));
clib-multigpu/localvariable.c:	 * If a local variable is read-only, we initialise its GPU buffer once.
clib-multigpu/localvariable.c:	 * The GPU buffer is accessible by all scheduled GPU execution streams.
clib-multigpu/localvariable.c:	 * Otherwise, we allocate a GPU-only buffer for each active stream.
clib-multigpu/localvariable.c:			checkCudaErrors (cudaSetDevice(dev->id));
clib-multigpu/localvariable.c:			checkCudaErrors (cudaSetDevice (dev->id));
clib-multigpu/lightweightdataset.c:	invalidConditionException (manager[phi][0]->GPU == manager[phi][1]->GPU);
clib-multigpu/lightweightdataset.c:	task->GPU = manager[phi][0]->GPU;
clib-multigpu/lightweightdataset.c:	(JNIEnv *env, jobject obj, jint phase, jint parts, jboolean gpu, jintArray tasksize) {
clib-multigpu/lightweightdataset.c:		manager[phase][i] = crossbowLightWeightDatasetManagerCreate (parts, (gpu == JNI_TRUE) ? 1 : 0, argv[i]);
clib-multigpu/lightweightdataset.c:		if (manager[phase][i]->GPU)
clib-multigpu/lightweightdataset.c:#ifdef GPU_VERBOSE
clib-multigpu/lightweightdataset.c:#ifdef GPU_VERBOSE
clib-multigpu/lightweightdatasetmanager.c:crossbowLightWeightDatasetManagerP crossbowLightWeightDatasetManagerCreate (int numberofnodes, unsigned GPU, int blocksize) {
clib-multigpu/lightweightdatasetmanager.c:	p->GPU = GPU;
clib-multigpu/lightweightdatasetbuffer.c:#include <cuda.h>
clib-multigpu/lightweightdatasetbuffer.c:#include <cuda_runtime.h>
clib-multigpu/lightweightdatasetbuffer.c:		checkCudaErrors(cudaHostRegister(ptr, blocksize, cudaHostRegisterMapped | cudaHostRegisterPortable));
clib-multigpu/lightweightdatasetbuffer.c:		checkCudaErrors(cudaHostUnregister(ptr));
clib-multigpu/taskhandler.c:#include <cuda.h>
clib-multigpu/taskhandler.c:#include <cuda_runtime.h>
clib-multigpu/taskhandler.c:	/* Redirect all CUDA calls to specific device */
clib-multigpu/taskhandler.c:	checkCudaErrors (cudaSetDevice(s->deviceId));
clib-multigpu/taskhandler.c:	/* Double-check that there are no pending asynchronous tasks on this CUDA streams */
clib-multigpu/taskhandler.c:	checkCudaErrors(cudaEventQuery(s->event));
clib-multigpu/taskhandler.c:	checkCudaErrors(cudaEventRecord(s->start, s->stream[0]));
clib-multigpu/taskhandler.c:					checkCudaErrors(cudaStreamWaitEvent(s->stream[s->op->branch], d->guard->end[s->id], 0));
clib-multigpu/taskhandler.c:					checkCudaErrors(cudaEventRecord(s->op->end[s->id], s->stream[s->op->branch]));
clib-multigpu/taskhandler.c:				checkCudaErrors(cudaStreamWaitEvent(s->stream[0], s->op->end[s->id], 0));
clib-multigpu/taskhandler.c:	checkCudaErrors(cudaEventRecord(s->event, s->stream[0]));
clib-multigpu/memorymanager.h:#include <cuda.h>
clib-multigpu/memorymanager.h:#include <cuda_runtime.h>
clib-multigpu/memorymanager.h:#include <cuda_runtime_api.h>
clib-multigpu/memorymanager.h:void *crossbowCudaMallocHost (int);
clib-multigpu/memorymanager.h:void *crossbowCudaMalloc (int);
clib-multigpu/memorymanager.h:void *crossbowCudaFreeHost (void *, int);
clib-multigpu/memorymanager.h:void *crossbowCudaFree (void *, int);
.gitignore:clib-multigpu/Makefile.save
.gitignore:clib-multigpu/Makefile
src/test/java/uk/ac/imperial/lsds/crossbow/LogisticRegression.java:			.setGPU (true)
src/test/java/uk/ac/imperial/lsds/crossbow/LogisticRegression.java:			.setNumberOfGPUModelReplicas (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/LogisticRegression.java:			.setNumberOfGPUStreams (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/LogisticRegression.java:			.setNumberOfGPUCallbackHandlers (8)
src/test/java/uk/ac/imperial/lsds/crossbow/LogisticRegression.java:			.setNumberOfGPUTaskHandlers(1)
src/test/java/uk/ac/imperial/lsds/crossbow/LogisticRegression.java:			.setGPUDevices(devices)
src/test/java/uk/ac/imperial/lsds/crossbow/ConvNet.java:			.setGPU (true)
src/test/java/uk/ac/imperial/lsds/crossbow/ConvNet.java:			.setNumberOfGPUModelReplicas (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/ConvNet.java:			.setNumberOfGPUStreams (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/ConvNet.java:			.setNumberOfGPUCallbackHandlers (8)
src/test/java/uk/ac/imperial/lsds/crossbow/ConvNet.java:			.setGPUDevices(devices);
src/test/java/uk/ac/imperial/lsds/crossbow/LeNetBatchNorm.java:			.setGPU (true)
src/test/java/uk/ac/imperial/lsds/crossbow/LeNetBatchNorm.java:			.setNumberOfGPUModelReplicas (numberofreplicas) /* wpc too !*/
src/test/java/uk/ac/imperial/lsds/crossbow/LeNetBatchNorm.java:			.setNumberOfGPUStreams (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/LeNetBatchNorm.java:			.setNumberOfGPUCallbackHandlers (8)
src/test/java/uk/ac/imperial/lsds/crossbow/LeNetBatchNorm.java:			.setGPUDevices(0);
src/test/java/uk/ac/imperial/lsds/crossbow/Inceptionv3.java:			.setGPU (true)
src/test/java/uk/ac/imperial/lsds/crossbow/Inceptionv3.java:			.setNumberOfGPUModelReplicas (numreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/Inceptionv3.java:			.setNumberOfGPUStreams (numreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/Inceptionv3.java:			.setNumberOfGPUCallbackHandlers (8)
src/test/java/uk/ac/imperial/lsds/crossbow/Inceptionv3.java:			.setGPUDevices(1)
src/test/java/uk/ac/imperial/lsds/crossbow/MatrixFactorisation.java:		boolean cpu = false, gpu = true;
src/test/java/uk/ac/imperial/lsds/crossbow/MatrixFactorisation.java:		// boolean cpu = true, gpu = false;
src/test/java/uk/ac/imperial/lsds/crossbow/MatrixFactorisation.java:		int __number_of_gpu_model_replicas = 1;
src/test/java/uk/ac/imperial/lsds/crossbow/MatrixFactorisation.java:		int __number_of_gpu_streams = 1;
src/test/java/uk/ac/imperial/lsds/crossbow/MatrixFactorisation.java:			if (args[i].equals("--gpu")) {
src/test/java/uk/ac/imperial/lsds/crossbow/MatrixFactorisation.java:				gpu = Boolean.parseBoolean(args[j]);
src/test/java/uk/ac/imperial/lsds/crossbow/MatrixFactorisation.java:			if (args[i].equals("--number-of-gpu-model-replicas")) {
src/test/java/uk/ac/imperial/lsds/crossbow/MatrixFactorisation.java:				__number_of_gpu_model_replicas =  Integer.parseInt(args[j]);
src/test/java/uk/ac/imperial/lsds/crossbow/MatrixFactorisation.java:			if (args[i].equals("--number-of-gpu-streams")) {
src/test/java/uk/ac/imperial/lsds/crossbow/MatrixFactorisation.java:				__number_of_gpu_streams = Integer.parseInt(args[j]);
src/test/java/uk/ac/imperial/lsds/crossbow/MatrixFactorisation.java:			.setGPU (gpu)
src/test/java/uk/ac/imperial/lsds/crossbow/MatrixFactorisation.java:			.setNumberOfGPUModelReplicas (__number_of_gpu_model_replicas)
src/test/java/uk/ac/imperial/lsds/crossbow/MatrixFactorisation.java:			.setNumberOfGPUStreams(__number_of_gpu_streams)
src/test/java/uk/ac/imperial/lsds/crossbow/TestRecordDataset.java:			.setGPU (true)
src/test/java/uk/ac/imperial/lsds/crossbow/TestRecordDataset.java:			.setNumberOfGPUModelReplicas (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/TestRecordDataset.java:			.setNumberOfGPUStreams (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/TestRecordDataset.java:			.setNumberOfGPUCallbackHandlers (2)
src/test/java/uk/ac/imperial/lsds/crossbow/TestRecordDataset.java:			.setNumberOfGPUTaskHandlers (2)
src/test/java/uk/ac/imperial/lsds/crossbow/TestRecordDataset.java:			.setGPUDevices (devices)
src/test/java/uk/ac/imperial/lsds/crossbow/TestMemoryPlanner.java:		SystemConf.getInstance().setCPU(true).setGPU(false);
src/test/java/uk/ac/imperial/lsds/crossbow/ResNetv1.java:			.setGPU (true)
src/test/java/uk/ac/imperial/lsds/crossbow/ResNetv1.java:			.setNumberOfGPUModelReplicas (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/ResNetv1.java:			.setNumberOfGPUStreams (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/ResNetv1.java:			.setNumberOfGPUCallbackHandlers (4)
src/test/java/uk/ac/imperial/lsds/crossbow/ResNetv1.java:			.setNumberOfGPUTaskHandlers (4)
src/test/java/uk/ac/imperial/lsds/crossbow/ResNetv1.java:			.setGPUDevices (devices)
src/test/java/uk/ac/imperial/lsds/crossbow/NoopApp.java:			.setGPU (true)
src/test/java/uk/ac/imperial/lsds/crossbow/NoopApp.java:			.setGPUDevices(0,1)
src/test/java/uk/ac/imperial/lsds/crossbow/NoopApp.java:			.setNumberOfGPUModelReplicas (1)
src/test/java/uk/ac/imperial/lsds/crossbow/NoopApp.java:			.setNumberOfGPUStreams (1)
src/test/java/uk/ac/imperial/lsds/crossbow/NoopApp.java:			.setNumberOfGPUCallbackHandlers (8)
src/test/java/uk/ac/imperial/lsds/crossbow/NoopStatelessApp.java:			.setGPU (false)
src/test/java/uk/ac/imperial/lsds/crossbow/NoopStatelessApp.java:			.setNumberOfGPUModelReplicas (1)
src/test/java/uk/ac/imperial/lsds/crossbow/NoopStatelessApp.java:			.setNumberOfGPUStreams (1)
src/test/java/uk/ac/imperial/lsds/crossbow/NoopStatelessApp.java:			.setNumberOfGPUCallbackHandlers (1)
src/test/java/uk/ac/imperial/lsds/crossbow/microbenchmarks/kernels/TestDataTransformAndConcat.java:		SystemConf.getInstance().setCPU(false).setGPU(true);
src/test/java/uk/ac/imperial/lsds/crossbow/microbenchmarks/kernels/TestDataTransform.java:		SystemConf.getInstance().setCPU(false).setGPU(true);
src/test/java/uk/ac/imperial/lsds/crossbow/microbenchmarks/kernels/TestBatchNorm.java:		SystemConf.getInstance().setCPU(true).setGPU(false);
src/test/java/uk/ac/imperial/lsds/crossbow/LeNet.java:			.setGPU (true)
src/test/java/uk/ac/imperial/lsds/crossbow/LeNet.java:			.setNumberOfGPUModelReplicas (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/LeNet.java:			.setNumberOfGPUStreams (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/LeNet.java:			.setNumberOfGPUCallbackHandlers (8)
src/test/java/uk/ac/imperial/lsds/crossbow/LeNet.java:			.setNumberOfGPUTaskHandlers(2)
src/test/java/uk/ac/imperial/lsds/crossbow/LeNet.java:			.setGPUDevices(devices);
src/test/java/uk/ac/imperial/lsds/crossbow/VGG.java:			.setGPU (true)
src/test/java/uk/ac/imperial/lsds/crossbow/VGG.java:			.setNumberOfGPUModelReplicas (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/VGG.java:			.setNumberOfGPUStreams (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/VGG.java:			.setNumberOfGPUCallbackHandlers (8)
src/test/java/uk/ac/imperial/lsds/crossbow/VGG.java:			.setNumberOfGPUTaskHandlers (1)
src/test/java/uk/ac/imperial/lsds/crossbow/VGG.java:			.setGPUDevices (devices)
src/test/java/uk/ac/imperial/lsds/crossbow/convnet/benchmarks/App.java:			.setGPU (true)
src/test/java/uk/ac/imperial/lsds/crossbow/convnet/benchmarks/App.java:			.setNumberOfGPUModelReplicas (replicas)
src/test/java/uk/ac/imperial/lsds/crossbow/convnet/benchmarks/App.java:			.setNumberOfGPUStreams (replicas)
src/test/java/uk/ac/imperial/lsds/crossbow/convnet/benchmarks/App.java:			.setNumberOfGPUTaskHandlers(1)
src/test/java/uk/ac/imperial/lsds/crossbow/convnet/benchmarks/App.java:			.setNumberOfGPUCallbackHandlers (8)
src/test/java/uk/ac/imperial/lsds/crossbow/convnet/benchmarks/App.java:			.setGPUDevices(devices)
src/test/java/uk/ac/imperial/lsds/crossbow/AlexNetv2.java:				.setGPU (true)
src/test/java/uk/ac/imperial/lsds/crossbow/AlexNetv2.java:				.setNumberOfGPUModelReplicas (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/AlexNetv2.java:				.setNumberOfGPUStreams (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/AlexNetv2.java:				.setNumberOfGPUCallbackHandlers (8)
src/test/java/uk/ac/imperial/lsds/crossbow/AlexNetv2.java:				.setGPUDevices(devices)
src/test/java/uk/ac/imperial/lsds/crossbow/ResNetv1ForCifar.java:			.setGPU (true)
src/test/java/uk/ac/imperial/lsds/crossbow/ResNetv1ForCifar.java:			.setNumberOfGPUModelReplicas (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/ResNetv1ForCifar.java:			.setNumberOfGPUStreams (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/ResNetv1ForCifar.java:			.setNumberOfGPUCallbackHandlers (8)
src/test/java/uk/ac/imperial/lsds/crossbow/ResNetv1ForCifar.java:			.setNumberOfGPUTaskHandlers (4)
src/test/java/uk/ac/imperial/lsds/crossbow/ResNetv1ForCifar.java:			.setGPUDevices (devices)
src/test/java/uk/ac/imperial/lsds/crossbow/TestDynamics.java:			.setGPU (true)
src/test/java/uk/ac/imperial/lsds/crossbow/TestDynamics.java:			.setNumberOfGPUModelReplicas (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/TestDynamics.java:			.setNumberOfGPUStreams (numberofreplicas)
src/test/java/uk/ac/imperial/lsds/crossbow/TestDynamics.java:			.setNumberOfGPUCallbackHandlers (8)
src/test/java/uk/ac/imperial/lsds/crossbow/TestDynamics.java:			.setNumberOfGPUTaskHandlers (4)
src/test/java/uk/ac/imperial/lsds/crossbow/TestDynamics.java:			.setGPUDevices (devices)
src/test/java/uk/ac/imperial/lsds/crossbow/dlbench/MLP.java:			.setGPU (true)
src/test/java/uk/ac/imperial/lsds/crossbow/dlbench/MLP.java:			.setNumberOfGPUModelReplicas (1)
src/test/java/uk/ac/imperial/lsds/crossbow/dlbench/MLP.java:			.setNumberOfGPUStreams (1)
src/test/java/uk/ac/imperial/lsds/crossbow/dlbench/MLP.java:			.setNumberOfGPUCallbackHandlers (8)
src/test/java/uk/ac/imperial/lsds/crossbow/dlbench/MLP.java:			.setGPUDevices(0)
src/test/java/uk/ac/imperial/lsds/crossbow/BabyResNet.java:		boolean __cpu = true, __gpu = false;
src/test/java/uk/ac/imperial/lsds/crossbow/BabyResNet.java:		/* GPU */
src/test/java/uk/ac/imperial/lsds/crossbow/BabyResNet.java:		int __number_of_gpu_model_replicas = 1;
src/test/java/uk/ac/imperial/lsds/crossbow/BabyResNet.java:		int __number_of_gpu_streams = 1;
src/test/java/uk/ac/imperial/lsds/crossbow/BabyResNet.java:			if (args[i].equals("--gpu")) {
src/test/java/uk/ac/imperial/lsds/crossbow/BabyResNet.java:				__gpu = Boolean.parseBoolean(args[j]);
src/test/java/uk/ac/imperial/lsds/crossbow/BabyResNet.java:			if (args[i].equals("--number-of-gpu-model-replicas")) {
src/test/java/uk/ac/imperial/lsds/crossbow/BabyResNet.java:				__number_of_gpu_model_replicas =  Integer.parseInt(args[j]);
src/test/java/uk/ac/imperial/lsds/crossbow/BabyResNet.java:			if (args[i].equals("--number-of-gpu-streams")) {
src/test/java/uk/ac/imperial/lsds/crossbow/BabyResNet.java:				__number_of_gpu_streams = Integer.parseInt(args[j]);
src/test/java/uk/ac/imperial/lsds/crossbow/BabyResNet.java:			.setGPU(__gpu)
src/test/java/uk/ac/imperial/lsds/crossbow/BabyResNet.java:			.setNumberOfGPUModelReplicas(__number_of_gpu_model_replicas)
src/test/java/uk/ac/imperial/lsds/crossbow/BabyResNet.java:			.setNumberOfGPUStreams(__number_of_gpu_streams)
src/test/java/uk/ac/imperial/lsds/crossbow/BabyResNet.java:			.setNumberOfGPUCallbackHandlers(__number_of_callback_handlers)
src/main/java/uk/ac/imperial/lsds/crossbow/ExecutionContext.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/ExecutionContext.java:		TheGPU.getInstance().init();
src/main/java/uk/ac/imperial/lsds/crossbow/ExecutionContext.java:		TheGPU.getInstance().register (this);
src/main/java/uk/ac/imperial/lsds/crossbow/ExecutionContext.java:		TheGPU.getInstance().destroy ();
src/main/java/uk/ac/imperial/lsds/crossbow/model/Model.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/model/Model.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/model/Model.java:		TheGPU.getInstance().setModel (variables.length, bytes);
src/main/java/uk/ac/imperial/lsds/crossbow/model/Model.java:				TheGPU.getInstance().setModelVariable (ndx, p.getOrder(), p.getShape().array(), p.capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/model/Model.java:				TheGPU.getInstance().setModelVariableData (ndx, p.getOrder(), p.getDataBuffer());
src/main/java/uk/ac/imperial/lsds/crossbow/model/Model.java:				TheGPU.getInstance().setModelVariableLearningRateMultiplier(ndx, p.getOrder(), p.getLearningRateMultiplier());
src/main/java/uk/ac/imperial/lsds/crossbow/model/Model.java:		// TheGPU.getInstance().overrideModelData (SystemConf.getInstance().getModelDirectory ());
src/main/java/uk/ac/imperial/lsds/crossbow/model/Model.java:		TheGPU.getInstance().setModelWorkPerClock (ModelConf.getInstance().getWpc());
src/main/java/uk/ac/imperial/lsds/crossbow/model/Model.java:		TheGPU.getInstance().setUpdateModelType (ModelConf.getInstance().getUpdateModel().getId());
src/main/java/uk/ac/imperial/lsds/crossbow/model/Model.java:		ModelConf.getInstance().getSolverConf().GPURegister ();
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:	 * Apply accumulated CPU gradient to GPU model, and vice versa 
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:		/* Auto-tune GPU model replicas (add or remove one replica per GPU) */
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:				log.info("Add a new model replica per GPU");
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:				/* TheGPU.getInstance().addModel (); */
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:				log.info("Remove a model replica per GPU");
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:				/* TheGPU.getInstance().delModel (); */
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:				if (SystemConf.getInstance().getGPU()) {
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:					TheGPU.getInstance().checkpointModel (SystemConf.getInstance().getCheckpointDirectory());
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:	 * Function has been modified for GPU-only execution; model check-pointing is disabled.
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:		 * Synchronise GPU models
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:		 * Distinguishing between BSP, SSP, and ASP is done at the GPU engine level.
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:		 * to lock all GPU model replicas (otherwise an exception will be thrown).
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:		if (SystemConf.getInstance().getGPU ()) {
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:			TheGPU.getInstance().lockAny();
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:			TheGPU.getInstance().merge (SystemConf.getInstance().isHybrid());
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:			TheGPU.getInstance().synchronise(0, clock, autotune(), false);
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:		TheGPU.getInstance().lockAny();
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:		TheGPU.getInstance().synchronise(0, clock, autotune(), false);
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:		/* Synchronise models across CPU and GPU boundary */
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:		if (SystemConf.getInstance().getGPU())
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:			TheGPU.getInstance().unlockAny();
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:		TheGPU.getInstance().unlockAny();
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:		TheGPU.getInstance().setModelManager
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:			(SystemConf.getInstance().numberOfGPUModelReplicas(), SystemConf.getInstance().getSynchronisationModel().getId());
src/main/java/uk/ac/imperial/lsds/crossbow/model/ModelManager.java:	 * So all models, both on CPU and GPU, are unlocked.
src/main/java/uk/ac/imperial/lsds/crossbow/ModelConf.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/ModelConf.java:					TheGPU.getInstance().setLightWeightDatasetHandler (i, datasets[i].getDatasetSlots(), datasets[i].numberOfSlots());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Classify.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Classify.java:		/* Are there any GPU-specific local variables? No */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Classify.java:		memoryRequirements.setLocalGPUMemoryRequirements (0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Classify.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Classify.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Classify.java:		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, true);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Classify.java:		TheGPU.getInstance().setKernelInput (id, 0, input[0].getShape().array(), input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Classify.java:		TheGPU.getInstance().setKernelOutput (id, output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dummy.java:		/* Are there any GPU-specific local variables? No */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dummy.java:		memoryRequirements.setLocalGPUMemoryRequirements (0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dummy.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/MatFact.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/MatFact.java:		memoryRequirements.setLocalGPUMemoryRequirements(labels.capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/MatFact.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/MatFact.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/MatFact.java:		TheGPU.getInstance().setKernel (id, name, 2, 1, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/MatFact.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/MatFact.java:		TheGPU.getInstance().setKernelInput  (id, 1, labels[0].getShape().array(), labels[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/MatFact.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/MatFact.java:		TheGPU.getInstance().setKernelLocalVariable (id, 0, "loss", labels[0].getShape().array(), labels[0].capacity(), false);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/MatFact.java:		/* Configure GPU kernel-specific parameters */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/MatFact.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 5);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/MatFact.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt   (id, 0, "latents", conf.numberOfLatentVariables());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/MatFact.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt   (id, 1,    "rows", conf.numberOfRows());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/MatFact.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt   (id, 2, "columns", conf.numberOfColumns());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/MatFact.java:		TheGPU.getInstance().setKernelConfigurationParameterAsFloat (id, 3,  "lambda", conf.getLambda());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/MatFact.java:		TheGPU.getInstance().setKernelConfigurationParameterAsFloat (id, 4,    "rate", conf.getLearningRateEta0());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMax.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMax.java:		/* Are there any GPU-specific local variables? No */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMax.java:		memoryRequirements.setLocalGPUMemoryRequirements (0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMax.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMax.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMax.java:		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMax.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMax.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMax.java:		/* Configure GPU kernel-specific parameters */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMax.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMax.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0, "axis", conf.getAxis());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMax.java:		TheGPU.getInstance().cudnnSetKernelType(id, CudnnKernelType.SOFTMAX.getId());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMax.java:		TheGPU.getInstance().cudnnSetKernelInputDescriptor  (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMax.java:		TheGPU.getInstance().cudnnSetKernelOutputDescriptor (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProductGradient.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProductGradient.java:		/* Are there any GPU-specific local variables? Yes, `biasmultiplier` */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProductGradient.java:			memoryRequirements.setLocalGPUMemoryRequirements (var.capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProductGradient.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProductGradient.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProductGradient.java:		TheGPU.getInstance().setKernel (id, name, 1, 1, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProductGradient.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProductGradient.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProductGradient.java:		TheGPU.getInstance().setKernelLocalVariable (id, 0, "biasmultiplier", local[0].getShape().array(), local[0].capacity(),  true);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProductGradient.java:		/* Initialise `_biasmultiplier` variable on GPU */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProductGradient.java:		TheGPU.getInstance().setKernelLocalVariableData (id, 0, local[0].getDataBuffer());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProductGradient.java:		/* Configure GPU kernel-specific parameters */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProductGradient.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 3);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProductGradient.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0,    "axis", conf.getAxis());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProductGradient.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 1, "outputs", conf.numberOfOutputs());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProductGradient.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 2,    "bias", conf.hasBias() ? 1 : 0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/GradientDescentOptimiser.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/GradientDescentOptimiser.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/GradientDescentOptimiser.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/GradientDescentOptimiser.java:		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/GradientDescentOptimiser.java:		TheGPU.getInstance().setKernelInput  (id, 0, new int [] {1}, 4);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/GradientDescentOptimiser.java:		TheGPU.getInstance().setKernelOutput (id,    new int [] {1}, 4);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/GradientDescentOptimiser.java:		/* Are there any GPU-specific local variables? No */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/GradientDescentOptimiser.java:		memoryRequirements.setLocalGPUMemoryRequirements (0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConvGradient.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConvGradient.java:		/* Are there any GPU-specific local variables? No */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConvGradient.java:		memoryRequirements.setLocalGPUMemoryRequirements (0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConvGradient.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConvGradient.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConvGradient.java:		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConvGradient.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConvGradient.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConvGradient.java:		/* Set GPU kernel-specific parameters */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConvGradient.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConvGradient.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0, "bias", conf.hasBias() ? 1 : 0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxGradient.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxGradient.java:		/* Are there any GPU-specific local variables? No */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxGradient.java:		memoryRequirements.setLocalGPUMemoryRequirements (0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxGradient.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxGradient.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxGradient.java:		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxGradient.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxGradient.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dropout.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dropout.java:		/* Are there any GPU-specific local variables? Yes, but they cannot be defined at the moment */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dropout.java:		memoryRequirements.setLocalGPUMemoryRequirements (0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dropout.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dropout.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dropout.java:		TheGPU.getInstance().setKernel (id, name, 1, 1, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dropout.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dropout.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dropout.java:		TheGPU.getInstance().cudnnSetKernelType (id, CudnnKernelType.DROPOUT.getId());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dropout.java:		TheGPU.getInstance().cudnnSetKernelInputDescriptor (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dropout.java:		TheGPU.getInstance().cudnnSetKernelOutputDescriptor (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dropout.java:		int reserveSpaceSizeInBytes = TheGPU.getInstance().cudnnGetDropoutReserveSpaceSize (id);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dropout.java:		TheGPU.getInstance().cudnnSetDropoutDescriptor (id, conf.getRatio(), SystemConf.getInstance().getRandomSeed());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dropout.java:			TheGPU.getInstance().setKernelLocalVariable (id, 0, "reserveSpace", 
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Dropout.java:			memoryRequirements.setLocalGPUMemoryRequirements (reserveSpaceSizeInBytes);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/IKernel.java:	public void GPURegister ();
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNormGradient.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNormGradient.java:		/* Are there any GPU-specific local variables?  No */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNormGradient.java:		memoryRequirements.setLocalGPUMemoryRequirements (0); 
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNormGradient.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNormGradient.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNormGradient.java:		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNormGradient.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNormGradient.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNormGradient.java:		/* Configure GPU kernel-specific parameters */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNormGradient.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 1); 
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNormGradient.java:		TheGPU.getInstance().setKernelConfigurationParameterAsDouble (id, 0, "epsilon", conf.getEpsilon());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DropoutGradient.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DropoutGradient.java:		/* Are there any GPU-specific local variables? No */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DropoutGradient.java:		memoryRequirements.setLocalGPUMemoryRequirements (0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DropoutGradient.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DropoutGradient.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DropoutGradient.java:		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DropoutGradient.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DropoutGradient.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/LRN.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLoss.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLoss.java:		/* Are there any GPU-specific local variables? Yes, `losses` and `count` */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLoss.java:		memoryRequirements.setLocalGPUMemoryRequirements (labels.capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLoss.java:		memoryRequirements.incLocalGPUMemoryRequirements (labels.capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLoss.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLoss.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLoss.java:		TheGPU.getInstance().setKernel (id, name, 2, 2, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLoss.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLoss.java:		TheGPU.getInstance().setKernelInput  (id, 1, labels[0].getShape().array(), labels[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLoss.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLoss.java:		TheGPU.getInstance().setKernelLocalVariable (id, 0, "losses", labels[0].getShape().array(), labels[0].capacity(),  false);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLoss.java:		TheGPU.getInstance().setKernelLocalVariable (id, 1, "counts", labels[0].getShape().array(), labels[0].capacity(),  false);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLoss.java:		/* Configure GPU kernel-specific parameters */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLoss.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLoss.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0, "ignorelabel", conf.getIgnoredLabelValue());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/LRNGradient.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:		/* Are there any GPU-specific local variables? Yes, `means` and `randoms` */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:			memoryRequirements.incLocalGPUMemoryRequirements (var.capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:			memoryRequirements.incLocalGPUMemoryRequirements (12 * inputShape[0].numberOfExamples());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:		log.debug(String.format ("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:		TheGPU.getInstance().setKernel (id, name, 1, numberoflocalvariables, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:			TheGPU.getInstance().setKernelLocalVariable (id, localvariableid ++, "means", local[0].getShape().array(), local[0].capacity(), true);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:			/* Initialise `_meanvalues` variable on GPU */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:			TheGPU.getInstance().setKernelLocalVariableData (id, 0, local[0].getDataBuffer());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:			/* The second local variable is GPU-specific used to store generated random numbers on device memory */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:			TheGPU.getInstance().setKernelLocalVariable (id, localvariableid ++, "randoms", new int [] { 3 * examples }, (12 * examples), false);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:		/* Configure GPU kernel-specific parameters */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 5);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt   (id, 0,     "cropSize", conf.getCropSize    ());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt   (id, 1,       "mirror", conf.getMirror      () ? 1 : 0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:		TheGPU.getInstance().setKernelConfigurationParameterAsFloat (id, 2,  "scaleFactor", conf.getScaleFactor ());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt   (id, 3, "subtractMean", conf.subtractMean   () ? 1 : 0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/DataTransform.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt   (id, 4, "hasMeanImage", conf.hasMeanImage   () ? 1 : 0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOpGradient.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOpGradient.java:		/* Are there any GPU-specific local variables? No */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOpGradient.java:		memoryRequirements.setLocalGPUMemoryRequirements (0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOpGradient.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOpGradient.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOpGradient.java:		/* TheGPU.getInstance().setKernel (id, name, 1, 0, operator.getPeer().numberOfInputs(), (isLossKernel() || isAccuracyKernel())); */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOpGradient.java:		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOpGradient.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOpGradient.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOpGradient.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOpGradient.java:		TheGPU.getInstance().setKernelConfigurationParameterAsFloatArray (id, 0, "coefficients", conf.getCoefficients());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/NoopStateless.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/NoopStateless.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/NoopStateless.java:		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/NoopStateless.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/NoopStateless.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Accuracy.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Accuracy.java:		/* Are there any GPU-specific local variables? Yes, `count` */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Accuracy.java:		memoryRequirements.setLocalGPUMemoryRequirements (labels.capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Accuracy.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Accuracy.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Accuracy.java:		TheGPU.getInstance().setKernel (id, name, 2, 1, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Accuracy.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Accuracy.java:		TheGPU.getInstance().setKernelInput  (id, 1, labels[0].getShape().array(), labels[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Accuracy.java:		TheGPU.getInstance().setKernelOutput (id, new int [] { 1 }, 4);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Accuracy.java:		TheGPU.getInstance().setKernelLocalVariable (id, 0, "count", labels[0].getShape().array(), labels[0].capacity(),  false);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLossGradient.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLossGradient.java:		/* Are there any GPU-specific local variables? Yes, `count` */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLossGradient.java:		memoryRequirements.setLocalGPUMemoryRequirements (labels.capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLossGradient.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLossGradient.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLossGradient.java:		TheGPU.getInstance().setKernel (id, name, 2, 1, 1, (isLossKernel() || isAccuracyKernel())); 
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLossGradient.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLossGradient.java:		TheGPU.getInstance().setKernelInput  (id, 1, labels[0].getShape().array(), labels[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLossGradient.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLossGradient.java:		TheGPU.getInstance().setKernelLocalVariable (id, 0, "count", labels[0].getShape().array(), labels[0].capacity(), false);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLossGradient.java:		/* Configure GPU kernel-specific parameters */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLossGradient.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/SoftMaxLossGradient.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0, "ignorelabel", conf.getIgnoredLabelValue());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOp.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOp.java:		/* Are there any GPU-specific local variables? No */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOp.java:		memoryRequirements.setLocalGPUMemoryRequirements (0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOp.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOp.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOp.java:		TheGPU.getInstance().setKernel (id, name, input.length, 0, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOp.java:			TheGPU.getInstance().setKernelInput (id, i, input[i].getShape().array(), input[i].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOp.java:		TheGPU.getInstance().setKernelOutput (id, output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOp.java:		/* Set GPU kernel-specific parameters */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOp.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ElementWiseOp.java:		TheGPU.getInstance().setKernelConfigurationParameterAsFloatArray (id, 0, "coefficients", conf.getCoefficients());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLUGradient.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLUGradient.java:		/* Are there any GPU-specific local variables? No */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLUGradient.java:		memoryRequirements.setLocalGPUMemoryRequirements (0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLUGradient.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLUGradient.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLUGradient.java:		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLUGradient.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLUGradient.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLUGradient.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLUGradient.java:		TheGPU.getInstance().setKernelConfigurationParameterAsFloat (id, 0, "slope", conf.getNegativeSlope());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/KernelMemoryRequirements.java:	/* Local variable length used for CPU and/or GPU execution. */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/KernelMemoryRequirements.java:	long localOnCPU, localOnGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/KernelMemoryRequirements.java:		output = model = localOnCPU = localOnGPU = 0L;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/KernelMemoryRequirements.java:	public KernelMemoryRequirements setLocalGPUMemoryRequirements (long bytes) {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/KernelMemoryRequirements.java:		localOnGPU = bytes;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/KernelMemoryRequirements.java:	public KernelMemoryRequirements incLocalGPUMemoryRequirements (long bytes) {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/KernelMemoryRequirements.java:		localOnGPU += bytes;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/KernelMemoryRequirements.java:	public long getLocalGPUMemoryRequirements () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/KernelMemoryRequirements.java:		return localOnGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Concat.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Concat.java:		/* Are there any GPU-specific local variables? No */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Concat.java:		memoryRequirements.setLocalGPUMemoryRequirements (0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Concat.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Concat.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Concat.java:		TheGPU.getInstance().setKernel (id, name, input.length, 0, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Concat.java:			TheGPU.getInstance().setKernelInput (id, i, input[i].getShape().array(), input[i].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Concat.java:		TheGPU.getInstance().setKernelOutput (id, output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Concat.java:		/* Set GPU kernel-specific parameters */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Concat.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Concat.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0, "axis", conf.getAxis());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/conf/SolverConf.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/conf/SolverConf.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/conf/SolverConf.java:		TheGPU.getInstance().setEamsgdAlpha (alpha);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/conf/SolverConf.java:		TheGPU.getInstance().setEamsgdTau   (tau);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/conf/SolverConf.java:		TheGPU.getInstance().setMomentum    (momentum, momentumMethod.getId ());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/conf/SolverConf.java:		TheGPU.getInstance().setWeightDecay (weightDecay);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/conf/SolverConf.java:			TheGPU.getInstance().setLearningRateDecayPolicyFixed (baseLearningRate); 
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/conf/SolverConf.java:			TheGPU.getInstance().setLearningRateDecayPolicyInv (baseLearningRate, gamma, power);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/conf/SolverConf.java:			TheGPU.getInstance().setLearningRateDecayPolicyStep (baseLearningRate, gamma, getStepSize());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/conf/SolverConf.java:			TheGPU.getInstance().setLearningRateDecayPolicyMultiStep (baseLearningRate, gamma, 0, getStepValues());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/conf/SolverConf.java:			TheGPU.getInstance().setLearningRateDecayPolicyMultiStep (baseLearningRate, gamma, 
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/conf/SolverConf.java:			TheGPU.getInstance().setLearningRateDecayPolicyExp (baseLearningRate, gamma);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/conf/SolverConf.java:			TheGPU.getInstance().setLearningRateDecayPolicyCircular (
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/conf/SolverConf.java:		TheGPU.getInstance().setBaseModelMomentum (baseModelMomentum);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		/* Are there any GPU-specific local variables? Yes */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		memoryRequirements.incLocalGPUMemoryRequirements (newMean.getInitialValue()[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		memoryRequirements.incLocalGPUMemoryRequirements ( newVar.getInitialValue()[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		TheGPU.getInstance().setKernel (id, name, 1, 2, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		/* Initialise local variables on GPU */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		TheGPU.getInstance().setKernelLocalVariable (id, 0, "newMean",     local[0].getShape().array(), local[0].capacity(), false);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		TheGPU.getInstance().setKernelLocalVariable (id, 1, "newVariance", local[0].getShape().array(), local[0].capacity(), false);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		/* Configure GPU kernel-specific parameters */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 4);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt	 (id, 0, "globalStatistics", conf.useGlobalStatistics() ? 1 : 0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		TheGPU.getInstance().setKernelConfigurationParameterAsDouble (id, 1, "epsilon",          conf.getEpsilon());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		TheGPU.getInstance().setKernelConfigurationParameterAsDouble (id, 2, "fraction",         conf.getMovingAverageFraction());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt	 (id, 3, "CMA",             (conf.getEstimatedMeanAndVarianceType() == BatchNormEstimatedMeanAndVarianceType.CMA) ? 1 : 0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		TheGPU.getInstance().cudnnSetKernelType(id, CudnnKernelType.BATCHNORM.getId());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		TheGPU.getInstance().cudnnSetBatchNormEstimatedMeanAndVariance (id, local[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		TheGPU.getInstance().cudnnSetKernelInputDescriptor  (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		TheGPU.getInstance().cudnnSetKernelOutputDescriptor	(id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/BatchNorm.java:		TheGPU.getInstance().cudnnSetBatchNormDescriptor (id);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Noop.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Noop.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Noop.java:		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, true); // (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Noop.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Noop.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Noop.java:		/* Configure GPU kernel-specific parameters */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Noop.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Noop.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0, "axis", conf.getAxis());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/PoolGradient.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/PoolGradient.java:		/* Are there any GPU-specific local variables? No */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/PoolGradient.java:		memoryRequirements.setLocalGPUMemoryRequirements (0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/PoolGradient.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/PoolGradient.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/PoolGradient.java:		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/PoolGradient.java:		TheGPU.getInstance().setKernelInput  (id, 0, input[0].getShape().array(), input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/PoolGradient.java:		TheGPU.getInstance().setKernelOutput (id, output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		/* Are there any GPU-specific local variables? Yes, but they cannot be defined at the moment */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		memoryRequirements.setLocalGPUMemoryRequirements (0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		TheGPU.getInstance().setKernel (id, name, 1, 3, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		/* Set GPU kernel-specific parameters */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0, "bias", conf.hasBias() ? 1 : 0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		TheGPU.getInstance().cudnnSetKernelType(id, CudnnKernelType.CONV.getId());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		TheGPU.getInstance().cudnnSetKernelInputDescriptor  (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		TheGPU.getInstance().cudnnSetKernelOutputDescriptor (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		TheGPU.getInstance().cudnnSetConvolutionDescriptor(id, padding.get(0), padding.get(1), stride.get(0), stride.get(1));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		TheGPU.getInstance().cudnnSetConvolutionFilterDescriptor(id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:			TheGPU.getInstance().cudnnSetConvolutionBiasDescriptor(id, 1, biasShape.get(0), 1, 1);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		/* Change -1 (unlimited memory) to 0 to eliminate workspace memory requirements on GPU */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		int forwardWorkspaceSizeInBytes        = TheGPU.getInstance().cudnnConfigureConvolutionForwardAlgorithm (id, __limit_fwd, __threshold); 
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		int backwardFilterWorkspaceSizeInBytes = TheGPU.getInstance().cudnnConfigureConvolutionBackwardFilterAlgorithm (id, __limit_bwd, __threshold);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		int backwardDataWorkspaceSizeInBytes   = TheGPU.getInstance().cudnnConfigureConvolutionBackwardDataAlgorithm (id, __limit_bwd, __threshold);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:		memoryRequirements.setLocalGPUMemoryRequirements
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:			TheGPU.getInstance().setKernelLocalVariable (id, 0, "forwardWorkSpace", 
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:			TheGPU.getInstance().setKernelLocalVariable (id, 1, "backwardFilterWorkSpace", 
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Conv.java:			TheGPU.getInstance().setKernelLocalVariable (id, 2, "backwardDataWorkSpace", 
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProduct.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProduct.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProduct.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProduct.java:		TheGPU.getInstance().setKernel (id, name, 1, 1, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProduct.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProduct.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProduct.java:		TheGPU.getInstance().setKernelLocalVariable (id, 0, "biasmultiplier", local[0].getShape().array(), local[0].capacity(), true);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProduct.java:		/* Initialise `_biasmultiplier` variable on GPU */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProduct.java:		TheGPU.getInstance().setKernelLocalVariableData (id, 0, local[0].getDataBuffer());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProduct.java:		/* Configure GPU kernel-specific parameters */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProduct.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 3);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProduct.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0,    "axis", conf.getAxis());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProduct.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 1, "outputs", conf.numberOfOutputs());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProduct.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 2,    "bias", conf.hasBias() ? 1 : 0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProduct.java:		/* Are there any GPU-specific local variables? Yes, `biasmultiplier` */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/InnerProduct.java:			memoryRequirements.setLocalGPUMemoryRequirements (var.capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConcatGradient.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConcatGradient.java:		/* Are there any GPU-specific local variables? No */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConcatGradient.java:		memoryRequirements.setLocalGPUMemoryRequirements (0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConcatGradient.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConcatGradient.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConcatGradient.java:		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConcatGradient.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConcatGradient.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConcatGradient.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 2);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConcatGradient.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 0, "axis",   conf.getAxis());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ConcatGradient.java:		TheGPU.getInstance().setKernelConfigurationParameterAsInt (id, 1, "offset", conf.getOffset());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Pool.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Pool.java:		/* Are there any GPU-specific local variables? No */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Pool.java:		memoryRequirements.setLocalGPUMemoryRequirements (0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Pool.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Pool.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Pool.java:		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Pool.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Pool.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Pool.java:		TheGPU.getInstance().cudnnSetKernelType(id, CudnnKernelType.POOL.getId());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Pool.java:		TheGPU.getInstance().cudnnSetKernelInputDescriptor  (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Pool.java:		TheGPU.getInstance().cudnnSetKernelOutputDescriptor (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Pool.java:		TheGPU.getInstance().cudnnSetPoolingMode (id, conf.getMethod().getId());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/Pool.java:		TheGPU.getInstance().cudnnSetPoolingDescriptor (id, kernelHeight, kernelWidth, paddingHeight, paddingWidth, strideHeight, strideWidth);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLU.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLU.java:		/* Are there any GPU-specific local variables? No */
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLU.java:		memoryRequirements.setLocalGPUMemoryRequirements (0);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLU.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLU.java:		log.debug(String.format("Register kernel with GPU for operator %s", operator.getName()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLU.java:		TheGPU.getInstance().setKernel (id, name, 1, 0, 1, (isLossKernel() || isAccuracyKernel()));
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLU.java:		TheGPU.getInstance().setKernelInput  (id, 0,  input[0].getShape().array(),  input[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLU.java:		TheGPU.getInstance().setKernelOutput (id,    output[0].getShape().array(), output[0].capacity());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLU.java:		TheGPU.getInstance().setKernelConfigurationParameters (id, 1);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLU.java:		TheGPU.getInstance().setKernelConfigurationParameterAsFloat (id, 0, "slope", conf.getNegativeSlope());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLU.java:		TheGPU.getInstance().cudnnSetKernelType (id, CudnnKernelType.RELU.getId());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLU.java:		TheGPU.getInstance().cudnnSetActivationDescriptor (id, conf.getActivationMode().getId(), conf.getReLUCeiling());
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLU.java:		TheGPU.getInstance().cudnnSetKernelInputDescriptor (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
src/main/java/uk/ac/imperial/lsds/crossbow/kernel/ReLU.java:		TheGPU.getInstance().cudnnSetKernelOutputDescriptor (id, dimensions [0], dimensions [1], dimensions [2], dimensions [3]);
src/main/java/uk/ac/imperial/lsds/crossbow/Dataset.java:		DatasetMemoryManager.getInstance().init (phase.getId(), parts, SystemConf.getInstance().getGPU(), 
src/main/java/uk/ac/imperial/lsds/crossbow/CoreMapper.java:		if (SystemConf.getInstance().getGPU()) {
src/main/java/uk/ac/imperial/lsds/crossbow/CoreMapper.java:			/* Core #0 is reserved for the task dispatcher and core #1 is reserved for the GPU worker thread.
src/main/java/uk/ac/imperial/lsds/crossbow/CoreMapper.java:			if (SystemConf.getInstance().numberOfGPUTaskHandlers() > 0) {
src/main/java/uk/ac/imperial/lsds/crossbow/CoreMapper.java:				pivot += SystemConf.getInstance().numberOfGPUTaskHandlers();
src/main/java/uk/ac/imperial/lsds/crossbow/CoreMapper.java:			// pivot += SystemConf.getInstance().numberOfGPUCallbackHandlers();
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheCPU.java:			String library = String.format("%s/clib-multigpu/libCPU.so", SystemConf.getInstance().getHomeDirectory());
src/main/java/uk/ac/imperial/lsds/crossbow/device/blas/BLAS.java:				String library = String.format("%s/clib-multigpu/libBLAS.so", SystemConf.getInstance().getHomeDirectory());
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:public class TheGPU {
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:	private final static Logger log = LogManager.getLogger (TheGPU.class);
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:	private static final TheGPU gpuInstance = new TheGPU ();
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:	public static TheGPU getInstance () { return gpuInstance; }
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:	public TheGPU () {
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:		if (! SystemConf.getInstance().getGPU())
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:			String library = String.format("%s/clib-multigpu/libGPU.so", SystemConf.getInstance().getHomeDirectory());
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:			SystemConf.getInstance().getGPUDevices(), 
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:			SystemConf.getInstance().numberOfGPUStreams(), 
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:			SystemConf.getInstance().numberOfGPUCallbackHandlers(),
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:			SystemConf.getInstance().numberOfGPUTaskHandlers(),
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:		if (! SystemConf.getInstance().getGPU())
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:		if (! SystemConf.getInstance().getGPU())
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:			throw new IllegalStateException ("error: GPU library is not loaded");
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:					node.getOperator().GPURegister();
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:				next.GPURegister();
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:		context.getModel().GPURegister();
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:		context.getModelManager().GPURegister();
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:		TheGPU.getInstance().overrideModelData (SystemConf.getInstance().getModelDirectory ());
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:			throw new UnsupportedOperationException ("error: GPU does not support indirect byte buffers yet");
src/main/java/uk/ac/imperial/lsds/crossbow/device/TheGPU.java:			throw new UnsupportedOperationException ("error: GPU does not support indirect byte buffers yet");
src/main/java/uk/ac/imperial/lsds/crossbow/device/dataset/DatasetMemoryManager.java:			String library = String.format("%s/clib-multigpu/libdataset.so", SystemConf.getInstance().getHomeDirectory());
src/main/java/uk/ac/imperial/lsds/crossbow/device/dataset/DatasetMemoryManager.java:	public native int init (int phase, int parts, boolean gpu, int [] block);
src/main/java/uk/ac/imperial/lsds/crossbow/device/dataset/LightWeightDatasetMemoryManager.java:			String library = String.format("%s/clib-multigpu/liblightweightdataset.so", SystemConf.getInstance().getHomeDirectory());
src/main/java/uk/ac/imperial/lsds/crossbow/device/dataset/LightWeightDatasetMemoryManager.java:	public native int init (int phase, int parts, boolean gpu, int [] block);
src/main/java/uk/ac/imperial/lsds/crossbow/device/ObjectRef.java:		String library = String.format("%s/clib-multigpu/libobjectref.so", SystemConf.getInstance().getHomeDirectory());
src/main/java/uk/ac/imperial/lsds/crossbow/device/random/RandomGenerator.java:				String library = String.format("%s/clib-multigpu/libRNG.so", SystemConf.getInstance().getHomeDirectory());
src/main/java/uk/ac/imperial/lsds/crossbow/task/AbstractTask.java:	protected boolean GPU = false;
src/main/java/uk/ac/imperial/lsds/crossbow/task/AbstractTask.java:	public void setGPU (boolean GPU) {
src/main/java/uk/ac/imperial/lsds/crossbow/task/AbstractTask.java:		this.GPU = GPU;
src/main/java/uk/ac/imperial/lsds/crossbow/task/AbstractTask.java:	public boolean isGPUTask () {
src/main/java/uk/ac/imperial/lsds/crossbow/task/AbstractTask.java:		return GPU;
src/main/java/uk/ac/imperial/lsds/crossbow/task/Task.java:		graph.process(batch, replicaId, this /* API */, GPU);
src/main/java/uk/ac/imperial/lsds/crossbow/task/Task.java:			if (! GPU) {
src/main/java/uk/ac/imperial/lsds/crossbow/task/Task.java:				handler.setSlot(taskId, batch.getFreeOffsets(), batch.getLoss(), batch.getAccuracy(), batch.getModelGradient(), GPU);
src/main/java/uk/ac/imperial/lsds/crossbow/task/Task.java:			if (! GPU) {
src/main/java/uk/ac/imperial/lsds/crossbow/task/Task.java:				 * We cannot release the model replica of a GPU task
src/main/java/uk/ac/imperial/lsds/crossbow/task/ITask.java:	public boolean isGPUTask ();
src/main/java/uk/ac/imperial/lsds/crossbow/dispatcher/TaskDispatcher.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/dispatcher/TaskDispatcher.java:			TheGPU.getInstance().schedule(graph.getId(), taskid, buffer[0], _p[0], _q[0], buffer[1], _p[1], _q[1], f, (test ? 1 : 0), bound);
src/main/java/uk/ac/imperial/lsds/crossbow/dispatcher/ResNet50TaskDispatcher.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/dispatcher/ResNet50TaskDispatcher.java:			TheGPU.getInstance().scheduleNext(graph.getId(), taskid, _p[0], _q[0], _p[1], _q[1], f, (test ? 1 : 0), bound);
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessor.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessor.java:	private boolean GPU;
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessor.java:	private int deviceId = 0; /* Processor class: GPU (0) or CPU (1) */
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessor.java:	public TaskProcessor (int pid, TaskQueue queue, int [][] matrix, ModelManager modelmanager, boolean GPU) {
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessor.java:		this.GPU = GPU;
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessor.java:		if (GPU) deviceId = 0;
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessor.java:		if (GPU) {
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessor.java:				log.info(String.format("GPU worker exits (tasks are scheduled directly by the dispatcher)"));
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessor.java:			log.info(String.format("GPU worker is thread %s", Thread.currentThread()));
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessor.java:			Thread.currentThread().setName("GPU task processor");
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessor.java:				if (GPU) {
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessor.java:					replicaId = TheGPU.getInstance().acquireAccess (clock);
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessor.java:					if (GPU) {
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessor.java:						replicaId = TheGPU.getInstance().upgradeAccess(replicaId, clock);
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessor.java:						if (GPU)
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessor.java:							throw new IllegalStateException ("error: cannot release a GPU model replica too early");
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessor.java:				task.setGPU(GPU);
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessorPool.java:			/* Assign the first processor to be the GPU worker */
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessorPool.java:		case GPU:
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessorPool.java:					("error: number of workers must be equal to 1 in GPU-only execution mode");
src/main/java/uk/ac/imperial/lsds/crossbow/processor/TaskProcessorPool.java:		if (SystemConf.getInstance().getExecutionMode().equals(ExecutionMode.GPU) && SystemConf.getInstance().useDirectScheduling())
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:	public void process (Batch batch, Integer replicaId, Task task, boolean GPU) {
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:		if (! GPU) {
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:			/* Schedule task on the GPU */
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:			TheGPU.getInstance().execute(getId(), batch, replicaId, task);
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:	public void GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:		TheGPU.getInstance().setDataflowGraph (id, ops);
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:				TheGPU.getInstance().setDataflowUpstreamNeighbours (id, next.getOrder(), upstream);
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:				TheGPU.getInstance().setDataflowDownstreamNeighbours (id, next.getOrder(), downstream);
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:			TheGPU.getInstance().setDataflowLossOperator(id, op.getId());
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:			TheGPU.getInstance().setDataflowAccuracyOperator(id, op.getId());
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:			TheGPU.getInstance().setDataflowDataTransformOperator(id, op.getId());
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:		TheGPU.getInstance().setDataflowPeers (id, peers);
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:			TheGPU.getInstance().setDataflowMemoryPlan (id, next.getOrder(), provider, position);
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java://			TheGPU.getInstance().setDataflowStream (id, next.getOrder(), next.getLabel());
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java://						TheGPU.getInstance().setDataflowDependency(id, next.getOrder(), DependencyType.END_BEFORE_START.getId(), prev.getOrder(), true);
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:		s.append(String.format("%5s:\t%9s\t%9s\t%9s\t%9s (%s)\n", "Id", "Output", "Model", "CPU vars", "GPU vars", "Name"));
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:					KernelMemoryRequirements.bytesToString (requirements.getLocalGPUMemoryRequirements ()),
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:			total.incLocalGPUMemoryRequirements (requirements.getLocalGPUMemoryRequirements ());
src/main/java/uk/ac/imperial/lsds/crossbow/SubGraph.java:				KernelMemoryRequirements.bytesToString (total.getLocalGPUMemoryRequirements ())
src/main/java/uk/ac/imperial/lsds/crossbow/Operator.java:	public boolean GPURegister () {
src/main/java/uk/ac/imperial/lsds/crossbow/Operator.java:		kernel.GPURegister();
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:	private boolean CPU, GPU;
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:	private int [] GPUDevices;
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:	 * GPU-only mode.
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		opts.add (new Option ("--gpu"                        ).setType (Boolean.class));
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		opts.add (new Option ("--gpu-devices"                ).setType ( String.class));
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		opts.add (new Option ("--number-of-gpu-models"       ).setType (Integer.class));
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		opts.add (new Option ("--number-of-gpu-streams"      ).setType (Integer.class));
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		GPU = false;
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		GPUDevices = new int [] { 0 }; /* Use first available device */
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:	public SystemConf setGPU (boolean GPU) {
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		this.GPU = GPU;
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:	public boolean getGPU () {
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		return GPU;
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		if (CPU && (! GPU)) return ExecutionMode.CPU;
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		if ((! CPU) && GPU) return ExecutionMode.GPU;
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:	public SystemConf setNumberOfGPUModelReplicas (int replicas) {
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:	public int numberOfGPUModelReplicas () {
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:	public SystemConf setNumberOfGPUStreams (int streams) {
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:	public int numberOfGPUStreams () {
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:	public SystemConf setNumberOfGPUCallbackHandlers (int callbackhandlers) {
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:	public int numberOfGPUCallbackHandlers () {
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:	public SystemConf setNumberOfGPUTaskHandlers (int taskhandlers) {
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:	public int numberOfGPUTaskHandlers () {
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:	public SystemConf setGPUDevices (int ... GPUDevices) {
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		this.GPUDevices = GPUDevices;
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:	public int [] getGPUDevices () {
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		return GPUDevices;
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		else if (arg.equals("--gpu")) {
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:			setGPU (opt.getBooleanValue ());
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		else if (arg.equals("--gpu-devices")) {
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:			setGPUDevices (list);
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		else if (arg.equals("--number-of-gpu-models")) {
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:			setNumberOfGPUModelReplicas (opt.getIntValue ());
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		else if (arg.equals("--number-of-gpu-streams")) {
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:			setNumberOfGPUStreams (opt.getIntValue ());
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:			setNumberOfGPUCallbackHandlers (opt.getIntValue ());
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:			setNumberOfGPUTaskHandlers (opt.getIntValue ());
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		s.append(String.format("%s execution mode\n", (isHybrid() ? "Hybrid" : (getGPU() ? "GPU-only" : "CPU-only"))));
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		s.append(String.format("%d GPU model replicas\n", replicas[1]));
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		s.append(String.format("%d GPU streams\n", streams));
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		s.append(String.format("%d GPU task callback handlers\n", callbackhandlers));
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		s.append(String.format("%d GPU task handlers\n", taskhandlers));
src/main/java/uk/ac/imperial/lsds/crossbow/SystemConf.java:		s.append(String.format("%s number of model replicas per GPU\n", (autotune ? "Auto-tune" : "Don't auto-tune")));
src/main/java/uk/ac/imperial/lsds/crossbow/types/ExecutionMode.java:	CPU (0), GPU (1), HYBRID (2);
src/main/java/uk/ac/imperial/lsds/crossbow/types/ExecutionMode.java:		case 1: return "GPU-only";
src/main/java/uk/ac/imperial/lsds/crossbow/RecordDataset.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/RecordDataset.java:		TheGPU.getInstance().recordDatasetInit (
src/main/java/uk/ac/imperial/lsds/crossbow/RecordDataset.java:			TheGPU.getInstance().recordDatasetRegister(phase.getId(), id, filename);
src/main/java/uk/ac/imperial/lsds/crossbow/RecordDataset.java:		TheGPU.getInstance().recordDatasetFinalise(phase.getId());
src/main/java/uk/ac/imperial/lsds/crossbow/DatasetMetadata.java:	 * they can be registered with CUDA memory. 
src/main/java/uk/ac/imperial/lsds/crossbow/result/TestResultHandler.java:	public void setSlot (int taskid, long [] free, float loss, float accuracy, ModelGradient gradient, boolean GPU) {
src/main/java/uk/ac/imperial/lsds/crossbow/result/IResultHandler.java:	public void setSlot (int taskid, long [] free, float loss, float accuracy, ModelGradient gradient, boolean GPU);
src/main/java/uk/ac/imperial/lsds/crossbow/result/TrainingResultHandler.java:			/* We need a way to differentiate between CPU and GPU tasks.
src/main/java/uk/ac/imperial/lsds/crossbow/result/TrainingResultHandler.java:			 * have been set by the GPU.
src/main/java/uk/ac/imperial/lsds/crossbow/result/TrainingResultHandler.java:	public void setSlot (int taskid, long [] free, float loss, float accuracy, ModelGradient gradient, boolean GPU) {
src/main/java/uk/ac/imperial/lsds/crossbow/result/TrainingResultHandler.java:		 * If null, then it must have been a GPU task.
src/main/java/uk/ac/imperial/lsds/crossbow/Dataflow.java:import uk.ac.imperial.lsds.crossbow.device.TheGPU;
src/main/java/uk/ac/imperial/lsds/crossbow/Dataflow.java:		if (SystemConf.getInstance().getGPU())
src/main/java/uk/ac/imperial/lsds/crossbow/Dataflow.java:			TheGPU.getInstance().setResultHandler(phase.getId(), handler.getResultSlots(), handler.numberOfSlots());
src/main/java/uk/ac/imperial/lsds/crossbow/LightWeightDataset.java:		LightWeightDatasetMemoryManager.getInstance().init (phase.getId(), parts, SystemConf.getInstance().getGPU(), 

```
