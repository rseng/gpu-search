# https://github.com/lsds/Saber

```console
cuda-6.5/URL:http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_6.5-14_amd64.deb
doc/paper/16-sigmod/plots/data/original/figure-08/hybrid-operator-boxes.dat:# System "Saber (CPU only)" "Saber (GPGPU only)" "Saber"
doc/paper/16-sigmod/plots/bin/eps/select-task-size.eps:[ [(Helvetica) 300.0 0.0 true true 0 (Saber \(GPGPU only\))]
doc/paper/16-sigmod/plots/bin/eps/select-complexity.eps:[ [(Helvetica) 420.0 0.0 true true 0 (Saber \(GPGPU only\))]
doc/paper/16-sigmod/plots/bin/eps/select-sliding-window.eps:[ [(Helvetica) 380.0 0.0 true true 0 (Saber \(GPGPU only\))]
doc/paper/16-sigmod/plots/bin/eps/select-task-size-window-1024-1024.eps:[ [(Helvetica) 300.0 0.0 true true 0 (Saber \(GPGPU only\))]
doc/paper/16-sigmod/plots/bin/eps/agg-avg-sliding-window.eps:[ [(Helvetica) 380.0 0.0 true true 0 (Saber \(GPGPU only\))]
doc/paper/16-sigmod/plots/bin/eps/join-task-size.eps:[ [(Helvetica) 300.0 0.0 true true 0 (Saber \(GPGPU only\))]
doc/paper/16-sigmod/plots/bin/eps/join-complexity.eps:[ [(Helvetica) 420.0 0.0 true true 0 (Saber \(GPGPU only\))]
doc/paper/16-sigmod/plots/bin/eps/hybrid-operators.eps:[ [(Helvetica) 280.0 0.0 true true 0 (Saber \(GPGPU only\))]
doc/paper/16-sigmod/plots/bin/eps/comparisons.eps:[ [(Helvetica) 280.0 0.0 true true 0 (Saber \(GPGPU contrib.\))]
doc/paper/16-sigmod/plots/bin/eps/groupby-task-size.eps:[ [(Helvetica) 300.0 0.0 true true 0 (Saber \(GPGPU only\))]
doc/paper/16-sigmod/plots/bin/eps/hlsadaptivity-with-google-cluster-data.eps:[ [(Helvetica) 240.0 0.0 true true 0 (Saber \(GPGPU contrib.\))]
doc/paper/16-sigmod/plots/bin/eps/select-task-size-window-1024-1.eps:[ [(Helvetica) 300.0 0.0 true true 0 (Saber \(GPGPU only\))]
doc/paper/16-sigmod/plots/bin/eps/select-task-size-window-1-1.eps:[ [(Helvetica) 300.0 0.0 true true 0 (Saber \(GPGPU only\))]
doc/paper/16-sigmod/plots/plotall.gnuplot:set style line  5 lt -1 lw 6 pt 1 ps 3 dashtype 3    # Throughput (gpu-only)
doc/paper/16-sigmod/plots/plotall.gnuplot:gpuThroughputStyle=5
doc/paper/16-sigmod/plots/plotall.gnuplot:saberGpuBox="fs solid 0.5 lt -1 lw 2"
doc/paper/16-sigmod/plots/plotall.gnuplot:saberGpu="Saber (GPGPU only)"
doc/paper/16-sigmod/plots/plotall.gnuplot:saberGpuContrib="Saber (GPGPU contrib.)"
doc/paper/16-sigmod/plots/plotall.gnuplot:@saberGpuBox \
doc/paper/16-sigmod/plots/plotall.gnuplot:title saberGpuContrib, \
doc/paper/16-sigmod/plots/plotall.gnuplot:@saberGpuBox \
doc/paper/16-sigmod/plots/plotall.gnuplot:title saberGpu, \
doc/paper/16-sigmod/plots/plotall.gnuplot:@saberGpuBox \
doc/paper/16-sigmod/plots/plotall.gnuplot:title saberGpu, \
doc/paper/16-sigmod/plots/plotall.gnuplot:fig10a.'gpu.dat' \
doc/paper/16-sigmod/plots/plotall.gnuplot:ls gpuThroughputStyle \
doc/paper/16-sigmod/plots/plotall.gnuplot:title saberGpu,\
doc/paper/16-sigmod/plots/plotall.gnuplot:fig10b.'gpu.dat' \
doc/paper/16-sigmod/plots/plotall.gnuplot:ls gpuThroughputStyle \
doc/paper/16-sigmod/plots/plotall.gnuplot:title saberGpu, \
doc/paper/16-sigmod/plots/plotall.gnuplot:fig11a.'gpu-throughput.dat' \
doc/paper/16-sigmod/plots/plotall.gnuplot:ls gpuThroughputStyle \
doc/paper/16-sigmod/plots/plotall.gnuplot:title saberGpu, \
doc/paper/16-sigmod/plots/plotall.gnuplot:fig11b.'gpu-throughput-avg.dat' \
doc/paper/16-sigmod/plots/plotall.gnuplot:ls gpuThroughputStyle \
doc/paper/16-sigmod/plots/plotall.gnuplot:title saberGpu, \
doc/paper/16-sigmod/plots/plotall.gnuplot:fig12a.'gpu-throughput.dat' \
doc/paper/16-sigmod/plots/plotall.gnuplot:ls gpuThroughputStyle \
doc/paper/16-sigmod/plots/plotall.gnuplot:title saberGpu, \
doc/paper/16-sigmod/plots/plotall.gnuplot:fig12b.'gpu-throughput.dat' \
doc/paper/16-sigmod/plots/plotall.gnuplot:ls gpuThroughputStyle \
doc/paper/16-sigmod/plots/plotall.gnuplot:title saberGpu, \
doc/paper/16-sigmod/plots/plotall.gnuplot:fig12c.'gpu-throughput.dat' \
doc/paper/16-sigmod/plots/plotall.gnuplot:ls gpuThroughputStyle \
doc/paper/16-sigmod/plots/plotall.gnuplot:title saberGpu, \
doc/paper/16-sigmod/plots/plotall.gnuplot:fig13a.'gpu-throughput.dat' \
doc/paper/16-sigmod/plots/plotall.gnuplot:ls gpuThroughputStyle \
doc/paper/16-sigmod/plots/plotall.gnuplot:title saberGpu, \
doc/paper/16-sigmod/plots/plotall.gnuplot:fig13b.'gpu-throughput.dat' \
doc/paper/16-sigmod/plots/plotall.gnuplot:ls gpuThroughputStyle \
doc/paper/16-sigmod/plots/plotall.gnuplot:title saberGpu
doc/paper/16-sigmod/plots/plotall.gnuplot:fig13c.'gpu-throughput.dat' \
doc/paper/16-sigmod/plots/plotall.gnuplot:ls gpuThroughputStyle \
doc/paper/16-sigmod/plots/plotall.gnuplot:title saberGpu
doc/paper/16-sigmod/plots/plotall.gnuplot:with boxes @saberGpuBox title saberGpuContrib, \
README.md:The CPU and GPU query operators have been implemented in Java and OpenCL, respectively. The GPU OpenCL operators are compiled _just-in-time_, when an operator is instantiated by a Saber application.
README.md:#### The libOpenCL.so library
README.md:Saber requires libOpenCL.so to be present on the OS library path (`LD_LIBRARY_PATH`). In our installation, it is located under `/usr/local/cuda/lib64/`. The OpenCL headers are located under `/usr/local/cuda/include`.
README.md:### GPU
README.md:The GPU used in our experiments is an NVIDIA Quadro K5200 with 2,304 CUDA cores and 8,123 MB of RAM, attached to the host via PCIe 3.0 x16. The GPU has **two copy engines**, allowing for concurrent copy and kernel execution.
README.md:  CUDA Driver Version / Runtime Version          8.0 / 7.0  
README.md:  CUDA Capability Major/Minor version number:    3.5  
README.md:  (12) Multiprocessors, (192) CUDA Cores/MP:     2304 CUDA Cores  
README.md:  GPU Max Clock rate:                            771 MHz (0.77 GHz)  
README.md:  Integrated GPU sharing Host Memory:            No  
README.md:######--execution-mode `cpu`|`gpu`|`hybrid`
README.md:Sets the execution mode to either CPU-only, GPU-only or Hybrid, respectively. In the latter case, both processors execute query tasks opportunistically. The default execution mode is `cpu`.
README.md:Sets the number of CPU worker threads. The default value is `1`. In GPU-only execution mode, the value must be `1`. **CPU worker threads are pinned to physical cores**. The first thread is pinned to core id 1, the second to core id 2, and so on.
README.md:Sets the GPU pipeline depth - the number of query tasks scheduled on the GPU before the result of the first one is returned. The default value is `4`. 
scripts/reset.sh:# Delete cuda-6.5 files
scripts/reset.sh:rm -rf "$SABER_HOME"/cuda-6.5/deb
scripts/reset.sh:rm -f "$SABER_HOME"/cuda-6.5/*.log
scripts/prepare-software.sh:# Check libraries: Is CUDA_HOME set?
scripts/try-install-cuda-6.5.sh:# Scripts that tries to install cuda-6.5 on Ubuntu 14.04
scripts/try-install-cuda-6.5.sh:# usage: ./try-install-cuda-6.5.sh
scripts/try-install-cuda-6.5.sh:# Download package to $SABER_HOME/cuda-6.5/deb
scripts/try-install-cuda-6.5.sh:if [ -d "$SABER_HOME/cuda-6.5/deb" ]; then
scripts/try-install-cuda-6.5.sh:	rm -f "$SABER_HOME"/cuda-6.5/deb/*.deb
scripts/try-install-cuda-6.5.sh:	mkdir -p "$SABER_HOME/cuda-6.5/deb"
scripts/try-install-cuda-6.5.sh:wget --input-file="$SABER_HOME/cuda-6.5/URL" --output-file="$SABER_HOME/cuda-6.5/wget.log" --directory-prefix="$SABER_HOME/cuda-6.5/deb"
scripts/try-install-cuda-6.5.sh:echo "error: failed to download cuda-6.5 package (transcript written on $SABER_HOME/cuda-6.5/wget.log)"
scripts/try-install-cuda-6.5.sh:# At this point, try install cuda-6.5
scripts/try-install-cuda-6.5.sh:PKG=$(ls "$SABER_HOME/cuda-6.5/deb")
scripts/try-install-cuda-6.5.sh:LOG="$SABER_HOME/cuda-6.5/install.log"
scripts/try-install-cuda-6.5.sh:sudo dpkg -i "$SABER_HOME/cuda-6.5/deb/$PKG" >>"$LOG" 2>&1
scripts/try-install-cuda-6.5.sh:echo "error: failed to install cuda-6.5 package (transcript written on $LOG)"
scripts/try-install-cuda-6.5.sh:sudo apt-get update >>"$SABER_HOME/cuda-6.5/install.log" 2>&1
scripts/try-install-cuda-6.5.sh:echo "Installing cuda-6.5..."
scripts/try-install-cuda-6.5.sh:saberInstallPackage "cuda-6-5" "$LOG" || exit 1
scripts/try-install-cuda-6.5.sh:echo "CUDA 6.5 library install successful (transcript written on $LOG)"
scripts/try-install-cuda-6.5.sh:echo "export CUDA_HOME=/usr/local/cuda-6.5"
scripts/try-install-cuda-6.5.sh:echo "export PATH=\$CUDA_HOME/bin:\$PATH"
scripts/try-install-cuda-6.5.sh:echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
scripts/saber.conf:# Execution mode: either "cpu", "gpu" or "hybrid" (default is "cpu")
scripts/saber.conf:# Pipeline depth: number of GPU execution pipelines (default is 4)
scripts/common.sh:	saberOptionInSet "$1" "$2" "cpu" "gpu" "hybrid" || result=1
scripts/experiments/figure-12/a/figure-12a.sh:	GPU_OPTS="$GPU_OPTS --number-of-worker-threads  1 --pipeline-depth 4"
scripts/experiments/figure-12/a/figure-12a.sh:	# 	Run Selection GPU-only with task size $N (N-2)
scripts/experiments/figure-12/a/figure-12a.sh:	for mode in "cpu" "gpu" "hybrid"; do
scripts/experiments/figure-12/a/figure-12a.sh:		"gpu")
scripts/experiments/figure-12/a/figure-12a.sh:		line="'Saber (GPU-only) throughput'"
scripts/experiments/figure-12/a/figure-12a.sh:		modeoptions="$GPU_OPTS"
scripts/experiments/figure-12/a/figure-12a.sh:	for mode in "cpu" "gpu" "hybrid"; do
scripts/experiments/figure-12/a/figure-12a.sh:		"gpu")
scripts/experiments/figure-12/a/figure-12a.sh:	for mode in "cpu" "gpu" "hybrid"; do
scripts/experiments/figure-12/a/figure-12a.sh:		"gpu")
scripts/experiments/figure-12/a/figure-12a.sh:		line="'Saber (GPU-only) throughput'"
scripts/experiments/figure-12/a/figure-12a.sh:GPU_OPTS=
scripts/experiments/figure-10/b/figure-10b.sh:GPU_OPTS=
scripts/experiments/figure-10/a/figure-10a.sh:	GPU_OPTS="$GPU_OPTS --number-of-worker-threads  1 --pipeline-depth 4"
scripts/experiments/figure-10/a/figure-10a.sh:	# 	Run Selection GPU-only with $N predicates (N-2)
scripts/experiments/figure-10/a/figure-10a.sh:	for mode in "cpu" "gpu" "hybrid"; do
scripts/experiments/figure-10/a/figure-10a.sh:		"gpu")
scripts/experiments/figure-10/a/figure-10a.sh:		line="'Saber (GPU-only) throughput'"
scripts/experiments/figure-10/a/figure-10a.sh:		modeoptions="$GPU_OPTS"
scripts/experiments/figure-10/a/figure-10a.sh:	for mode in "cpu" "gpu" "hybrid"; do
scripts/experiments/figure-10/a/figure-10a.sh:		"gpu")
scripts/experiments/figure-10/a/figure-10a.sh:	for mode in "cpu" "gpu" "hybrid"; do
scripts/experiments/figure-10/a/figure-10a.sh:		"gpu")
scripts/experiments/figure-10/a/figure-10a.sh:		line="'Saber (GPU-only) throughput'"
scripts/experiments/figure-10/a/figure-10a.sh:GPU_OPTS=
scripts/experiments/figure-08/figure-08.sh:GPU_OPTS=
scripts/experiments/figure-13/c/figure-13c.sh:	GPU_OPTS="$GPU_OPTS --number-of-worker-threads 1 --pipeline-depth 4"
scripts/experiments/figure-13/c/figure-13c.sh:	# 	Run Selection GPU-only with task size $N (window 1024 rows, slide 1024 rows) (N-2)
scripts/experiments/figure-13/c/figure-13c.sh:	for mode in "cpu" "gpu"; do
scripts/experiments/figure-13/c/figure-13c.sh:		line="'Saber (GPU-only) throughput'"
scripts/experiments/figure-13/c/figure-13c.sh:		modeoptions="$GPU_OPTS"
scripts/experiments/figure-13/c/figure-13c.sh:	for mode in "cpu" "gpu"; do
scripts/experiments/figure-13/c/figure-13c.sh:	for mode in "cpu" "gpu"; do
scripts/experiments/figure-13/c/figure-13c.sh:		line="'Saber (GPU-only) throughput'"
scripts/experiments/figure-13/c/figure-13c.sh:GPU_OPTS=
scripts/experiments/figure-13/a/figure-13a.sh:	GPU_OPTS="$GPU_OPTS --number-of-worker-threads 1 --pipeline-depth 4"
scripts/experiments/figure-13/a/figure-13a.sh:	# 	Run Selection GPU-only with task size $N (window 1 row, slide 1 row) (N-2)
scripts/experiments/figure-13/a/figure-13a.sh:	for mode in "cpu" "gpu"; do
scripts/experiments/figure-13/a/figure-13a.sh:		line="'Saber (GPU-only) throughput'"
scripts/experiments/figure-13/a/figure-13a.sh:		modeoptions="$GPU_OPTS"
scripts/experiments/figure-13/a/figure-13a.sh:	for mode in "cpu" "gpu"; do
scripts/experiments/figure-13/a/figure-13a.sh:	for mode in "cpu" "gpu"; do
scripts/experiments/figure-13/a/figure-13a.sh:		line="'Saber (GPU-only) throughput'"
scripts/experiments/figure-13/a/figure-13a.sh:GPU_OPTS=
scripts/experiments/figure-11/b/figure-11b.sh:	GPU_OPTS="$GPU_OPTS --number-of-worker-threads  1 --pipeline-depth 4"
scripts/experiments/figure-11/b/figure-11b.sh:	# 	Run Aggregation GPU-only with $N tuples/slide (N-2)
scripts/experiments/figure-11/b/figure-11b.sh:	for mode in "cpu" "gpu" "hybrid"; do
scripts/experiments/figure-11/b/figure-11b.sh:		"gpu")
scripts/experiments/figure-11/b/figure-11b.sh:		line="'Saber (GPU-only) throughput'"
scripts/experiments/figure-11/b/figure-11b.sh:		modeoptions="$GPU_OPTS"
scripts/experiments/figure-11/b/figure-11b.sh:	for mode in "cpu" "gpu" "hybrid"; do
scripts/experiments/figure-11/b/figure-11b.sh:		"gpu")
scripts/experiments/figure-11/b/figure-11b.sh:	for mode in "cpu" "gpu" "hybrid"; do
scripts/experiments/figure-11/b/figure-11b.sh:		"gpu")
scripts/experiments/figure-11/b/figure-11b.sh:		line="'Saber (GPU-only) throughput'"
scripts/experiments/figure-11/b/figure-11b.sh:GPU_OPTS=
scripts/experiments/figure-11/a/figure-11a.sh:	GPU_OPTS="$GPU_OPTS --number-of-worker-threads  1 --pipeline-depth 4"
scripts/experiments/figure-11/a/figure-11a.sh:	# 	Run Selection GPU-only with $N tuples/slide (N-2)
scripts/experiments/figure-11/a/figure-11a.sh:	for mode in "cpu" "gpu" "hybrid"; do
scripts/experiments/figure-11/a/figure-11a.sh:		"gpu")
scripts/experiments/figure-11/a/figure-11a.sh:		line="'Saber (GPU-only) throughput'"
scripts/experiments/figure-11/a/figure-11a.sh:		modeoptions="$GPU_OPTS"
scripts/experiments/figure-11/a/figure-11a.sh:	for mode in "cpu" "gpu" "hybrid"; do
scripts/experiments/figure-11/a/figure-11a.sh:		"gpu")
scripts/experiments/figure-11/a/figure-11a.sh:	for mode in "cpu" "gpu" "hybrid"; do
scripts/experiments/figure-11/a/figure-11a.sh:		"gpu")
scripts/experiments/figure-11/a/figure-11a.sh:		line="'Saber (GPU-only) throughput'"
scripts/experiments/figure-11/a/figure-11a.sh:GPU_OPTS=
scripts/build.sh:# Make libGPU.so
scripts/build.sh:make gpu >>$TRANSCRIPT 2>&1
scripts/build.sh:echo "error: Saber GPU library compilation failed (transcript written on $SABER_HOME/clib/$TRANSCRIPT)"
scripts/build.sh:GPULIB="$SABER_HOME/clib/libGPU.so"
scripts/build.sh:saberFileExistsOrExit "$GPULIB"
scripts/build.sh:gpulibsize=`wc -c < "$GPULIB" | sed -e 's/^[ \t]*//'`
scripts/build.sh:echo "Saber GPU library build successful (transcript written on $SABER_HOME/clib/$TRANSCRIPT)"
scripts/build.sh:echo "Output written on $GPULIB ($gpulibsize bytes)"
clib/debug.h:#ifndef __GPU_DEBUG_H_
clib/debug.h:#define __GPU_DEBUG_H_
clib/debug.h:#undef GPU_VERBOSE
clib/debug.h:// #define GPU_VERBOSE
clib/debug.h:#undef GPU_PROFILE
clib/debug.h:// #define GPU_PROFILE
clib/debug.h:#ifdef GPU_VERBOSE
clib/debug.h:#endif /* __GPU_DEBUG_H_ */
clib/inputbuffer.c:#include "openclerrorcode.h"
clib/inputbuffer.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/inputbuffer.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/inputbuffer.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/resulthandler.c:	/* Pin this thread to a particular core: 0 is the dispatcher, 1 is the GPU */
clib/resulthandler.c:		gpu_context_readOutput (p->context, p->readOutput, env, p->obj, p->qid);
clib/resulthandler.c:	gpuContextP context,
clib/resulthandler.c:	void (*callback)(gpuContextP, JNIEnv *, jobject, int, int, int),
clib/GPU.c:#include "GPU.h"
clib/GPU.c:#include "uk_ac_imperial_lsds_saber_devices_TheGPU.h"
clib/GPU.c:#include "gpuquery.h"
clib/GPU.c:#include "openclerrorcode.h"
clib/GPU.c:#include <OpenCL/opencl.h>
clib/GPU.c:static gpuQueryP queries [MAX_QUERIES];
clib/GPU.c:static gpuContextP pipeline [MAX_DEPTH];
clib/GPU.c:void callback_setKernelDummy     (cl_kernel, gpuContextP, int *, long *);
clib/GPU.c:void callback_setKernelProject   (cl_kernel, gpuContextP, int *, long *);
clib/GPU.c:void callback_setKernelSelect    (cl_kernel, gpuContextP, int *, long *);
clib/GPU.c:void callback_setKernelCompact   (cl_kernel, gpuContextP, int *, long *);
clib/GPU.c:void callback_setKernelThetaJoin (cl_kernel, gpuContextP, int *, long *);
clib/GPU.c:void callback_setKernelReduce    (cl_kernel, gpuContextP, int *, long *);
clib/GPU.c:void callback_setKernelAggregate (cl_kernel, gpuContextP, int *, long *);
clib/GPU.c:void callback_configureReduce    (cl_kernel, gpuContextP, int *, long *);
clib/GPU.c:void callback_configureAggregate (cl_kernel, gpuContextP, int *, long *);
clib/GPU.c:void callback_writeInput (gpuContextP, JNIEnv *, jobject, int, int);
clib/GPU.c:void callback_readOutput (gpuContextP, JNIEnv *, jobject, int, int, int);
clib/GPU.c:gpuContextP callback_execKernel (gpuContextP);
clib/GPU.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/GPU.c:	error = clGetDeviceIDs (platform, CL_DEVICE_TYPE_GPU, 2, &device, &count);
clib/GPU.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/GPU.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/GPU.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/GPU.c:	fprintf(stdout, "GPU name: %s\n", name);
clib/GPU.c:	fprintf(stdout, "GPU supported extensions are: %s\n", extensions);
clib/GPU.c:	fprintf(stdout, "GPU memory addresses are %u bits aligned\n", value);
clib/GPU.c:void gpu_init (JNIEnv *env, int _queries, int _depth) {
clib/GPU.c:	#ifdef GPU_HANDLER
clib/GPU.c:void gpu_free () {
clib/GPU.c:			gpu_query_free (queries[i]);
clib/GPU.c:int gpu_getQuery (const char *source, int _kernels, int _inputs, int _outputs) {
clib/GPU.c:	queries[ndx] = gpu_query_new (ndx, device, context, source, _kernels, _inputs, _outputs);
clib/GPU.c:	gpu_query_setResultHandler (queries[ndx], resultHandler);
clib/GPU.c:int gpu_setInput  (int qid, int ndx, int size) {
clib/GPU.c:	gpuQueryP p = queries[qid];
clib/GPU.c:	return gpu_query_setInput (p, ndx, size);
clib/GPU.c:int gpu_setOutput (int qid, int ndx, int size, int writeOnly, int doNotMove, int bearsMark, int readEvent, int ignoreMark) {
clib/GPU.c:	gpuQueryP p = queries[qid];
clib/GPU.c:	return gpu_query_setOutput (p, ndx, size, writeOnly, doNotMove, bearsMark, readEvent, ignoreMark);
clib/GPU.c:int gpu_setKernel (int qid, int ndx,
clib/GPU.c:	void (*callback)(cl_kernel, gpuContextP, int *, long *),
clib/GPU.c:	gpuQueryP p = queries[qid];
clib/GPU.c:	return gpu_query_setKernel (p, ndx, name, callback, args1, args2);
clib/GPU.c:int gpu_exec (int qid,
clib/GPU.c:	gpuQueryP p = queries[qid];
clib/GPU.c:	return gpu_query_exec (p, threads, threadsPerGroup, operator, env, obj);
clib/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_init
clib/GPU.c:	gpu_init (env, N, D);
clib/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_free
clib/GPU.c:	gpu_free ();
clib/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_getQuery
clib/GPU.c:	int qid = gpu_getQuery (_source, _kernels, _inputs, _outputs);
clib/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_setInput
clib/GPU.c:	return gpu_setInput (qid, ndx, size);
clib/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_setOutput
clib/GPU.c:	return gpu_setOutput (qid, ndx, size, writeOnly, doNotMove, bearsMark, readEvent, ignoreMark);
clib/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_setKernelDummy
clib/GPU.c:	gpu_setKernel (qid, 0, "dummyKernel", &callback_setKernelDummy, NULL, NULL);
clib/GPU.c:void callback_setKernelDummy (cl_kernel kernel, gpuContextP context, int *args1, long *args2) {
clib/GPU.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_setKernelProject
clib/GPU.c:	gpu_setKernel (qid, 0, "projectKernel", &callback_setKernelProject, args, NULL);
clib/GPU.c:void callback_setKernelProject (cl_kernel kernel, gpuContextP context, int *args1, long *args2) {
clib/GPU.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_setKernelSelect
clib/GPU.c:	gpu_setKernel (qid, 0,  "selectKernel",  &callback_setKernelSelect, args, NULL);
clib/GPU.c:	gpu_setKernel (qid, 1, "compactKernel", &callback_setKernelCompact, args, NULL);
clib/GPU.c:void callback_setKernelSelect (cl_kernel kernel, gpuContextP context, int *args1, long *args2) {
clib/GPU.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/GPU.c:void callback_setKernelCompact (cl_kernel kernel, gpuContextP context, int *args1, long *args2) {
clib/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_setKernelThetaJoin
clib/GPU.c:	gpu_setKernel (qid, 0, "countKernel",   &callback_setKernelThetaJoin, args, NULL);
clib/GPU.c:	gpu_setKernel (qid, 1, "scanKernel",    &callback_setKernelThetaJoin, args, NULL);
clib/GPU.c:	gpu_setKernel (qid, 2, "compactKernel", &callback_setKernelThetaJoin, args, NULL);
clib/GPU.c:	gpu_setKernel (qid, 3, "joinKernel",    &callback_setKernelThetaJoin, args, NULL);
clib/GPU.c:void callback_setKernelThetaJoin (cl_kernel kernel, gpuContextP context, int *args1, long *args2) {
clib/GPU.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_setKernelReduce
clib/GPU.c:	gpu_setKernel (qid, 0, "clearKernel",           &callback_setKernelReduce, args1, args2);
clib/GPU.c:	gpu_setKernel (qid, 1, "computeOffsetKernel",   &callback_setKernelReduce, args1, args2);
clib/GPU.c:	gpu_setKernel (qid, 2, "computePointersKernel", &callback_setKernelReduce, args1, args2);
clib/GPU.c:	gpu_setKernel (qid, 3, "reduceKernel",          &callback_setKernelReduce, args1, args2);
clib/GPU.c:void callback_setKernelReduce (cl_kernel kernel, gpuContextP context, int *args1, long *args2) {
clib/GPU.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_setKernelAggregate
clib/GPU.c:	gpu_setKernel (qid, 0, "clearKernel",                    &callback_setKernelAggregate, args1, args2);
clib/GPU.c:	gpu_setKernel (qid, 1, "computeOffsetKernel",            &callback_setKernelAggregate, args1, args2);
clib/GPU.c:	gpu_setKernel (qid, 2, "computePointersKernel",          &callback_setKernelAggregate, args1, args2);
clib/GPU.c:	gpu_setKernel (qid, 3, "countWindowsKernel",             &callback_setKernelAggregate, args1, args2);
clib/GPU.c:	gpu_setKernel (qid, 4, "aggregateClosingWindowsKernel",  &callback_setKernelAggregate, args1, args2);
clib/GPU.c:	gpu_setKernel (qid, 5, "aggregateCompleteWindowsKernel", &callback_setKernelAggregate, args1, args2);
clib/GPU.c:	gpu_setKernel (qid, 6, "aggregateOpeningWindowsKernel",  &callback_setKernelAggregate, args1, args2);
clib/GPU.c:	gpu_setKernel (qid, 7, "aggregatePendingWindowsKernel",  &callback_setKernelAggregate, args1, args2);
clib/GPU.c:	gpu_setKernel (qid, 8, "packKernel",                     &callback_setKernelAggregate, args1, args2);
clib/GPU.c:void callback_setKernelAggregate (cl_kernel kernel, gpuContextP context, int *args1, long *args2) {
clib/GPU.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_execute
clib/GPU.c:	gpu_exec (qid, threads, threadsPerGroup, operator, env, obj);
clib/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_executeReduce
clib/GPU.c:	gpu_exec (qid, threads, threadsPerGroup, operator, env, obj);
clib/GPU.c:void callback_configureReduce (cl_kernel kernel, gpuContextP context, int *args1, long *args2) {
clib/GPU.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/GPU.c:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_executeAggregate
clib/GPU.c:	gpu_exec (qid, threads, threadsPerGroup, operator, env, obj);
clib/GPU.c:void callback_configureAggregate (cl_kernel kernel, gpuContextP context, int *args1, long *args2) {
clib/GPU.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/GPU.c:void callback_writeInput (gpuContextP context, JNIEnv *env, jobject obj, int qid, int ndx) {
clib/GPU.c:void callback_readOutput (gpuContextP context, JNIEnv *env, jobject obj, int qid, int ndx, int mark) {
clib/GPU.c:gpuContextP callback_execKernel (gpuContextP context) {
clib/GPU.c:	gpuContextP p = pipeline[0];
clib/GPU.c:	#ifdef GPU_VERBOSE
clib/inputbuffer.h:#include <OpenCL/opencl.h>
clib/outputbuffer.c:#include "openclerrorcode.h"
clib/outputbuffer.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/outputbuffer.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/outputbuffer.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/outputbuffer.h:#include <OpenCL/opencl.h>
clib/genmakefile.sh:CH="$CUDA_HOME"
clib/genmakefile.sh:# If CUDA_HOME is not set, try to find it
clib/genmakefile.sh:[ -z "$CH" ] && [ -d "/usr/local/cuda" ] && CH="/usr/local/cuda"
clib/genmakefile.sh:echo "CUDA_HOME = $CH" >>"$MAKEFILE"
clib/genmakefile.sh:# Set path to CUDA_HOME/include on Linux only
clib/genmakefile.sh:ifneq (\$(CUDA_HOME),)
clib/genmakefile.sh:	GFLAGS += -I\$(CUDA_HOME)/include
clib/genmakefile.sh:# Set OpenCL library
clib/genmakefile.sh:	GLIBS += -framework opencl 
clib/genmakefile.sh:	GLIBS += -lOpenCL
clib/genmakefile.sh:SABERLIBS += -lGPU
clib/genmakefile.sh:OBJS = GPU.o timer.o openclerrorcode.o resulthandler.o inputbuffer.o outputbuffer.o gpucontext.o gpuquery.o
clib/genmakefile.sh:all: libCPU.so libGPU.so
clib/genmakefile.sh:gpu: libGPU.so
clib/genmakefile.sh:libGPU.so: \$(OBJS)
clib/genmakefile.sh:	\$(CC) -shared -o libGPU.so \$(OBJS) \$(GLIBS) \$(CLIBS)
clib/genmakefile.sh:GPU.o: GPU.c GPU.h uk_ac_imperial_lsds_saber_devices_TheGPU.h timer.h openclerrorcode.h 
clib/genmakefile.sh:uk_ac_imperial_lsds_saber_devices_TheGPU.h:
clib/genmakefile.sh:	javah -classpath \$(CLASS_PATH) uk.ac.imperial.lsds.saber.devices.TheGPU
clib/genmakefile.sh:openclerrorcode.o: openclerrorcode.c openclerrorcode.h
clib/genmakefile.sh:gpucontext.o: gpucontext.c gpucontext.h
clib/genmakefile.sh:gpuquery.o: gpuquery.c gpuquery.h
clib/resulthandler.h:#include "gpucontext.h"
clib/resulthandler.h:	gpuContextP context;
clib/resulthandler.h:	void (*readOutput) (gpuContextP, JNIEnv *, jobject, int, int, int);
clib/resulthandler.h:void result_handler_readOutput (resultHandlerP, int, gpuContextP,
clib/resulthandler.h:		void (*callback)(gpuContextP, JNIEnv *, jobject, int, int, int), jobject);
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h:/* Header for class uk_ac_imperial_lsds_saber_devices_TheGPU */
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h:#ifndef _Included_uk_ac_imperial_lsds_saber_devices_TheGPU
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h:#define _Included_uk_ac_imperial_lsds_saber_devices_TheGPU
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h: * Class:     uk_ac_imperial_lsds_saber_devices_TheGPU
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_init
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h: * Class:     uk_ac_imperial_lsds_saber_devices_TheGPU
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_free
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h: * Class:     uk_ac_imperial_lsds_saber_devices_TheGPU
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_getQuery
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h: * Class:     uk_ac_imperial_lsds_saber_devices_TheGPU
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_setInput
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h: * Class:     uk_ac_imperial_lsds_saber_devices_TheGPU
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_setOutput
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h: * Class:     uk_ac_imperial_lsds_saber_devices_TheGPU
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_setKernelDummy
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h: * Class:     uk_ac_imperial_lsds_saber_devices_TheGPU
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_setKernelProject
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h: * Class:     uk_ac_imperial_lsds_saber_devices_TheGPU
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_setKernelSelect
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h: * Class:     uk_ac_imperial_lsds_saber_devices_TheGPU
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_setKernelThetaJoin
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h: * Class:     uk_ac_imperial_lsds_saber_devices_TheGPU
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_setKernelReduce
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h: * Class:     uk_ac_imperial_lsds_saber_devices_TheGPU
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_setKernelAggregate
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h: * Class:     uk_ac_imperial_lsds_saber_devices_TheGPU
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_execute
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h: * Class:     uk_ac_imperial_lsds_saber_devices_TheGPU
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_executeReduce
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h: * Class:     uk_ac_imperial_lsds_saber_devices_TheGPU
clib/uk_ac_imperial_lsds_saber_devices_TheGPU.h:JNIEXPORT jint JNICALL Java_uk_ac_imperial_lsds_saber_devices_TheGPU_executeAggregate
clib/gpuquery.h:#ifndef __GPU_QUERY_H_
clib/gpuquery.h:#define __GPU_QUERY_H_
clib/gpuquery.h:#include "gpucontext.h"
clib/gpuquery.h:#include "GPU.h"
clib/gpuquery.h:typedef struct gpu_query *gpuQueryP;
clib/gpuquery.h:typedef struct gpu_query {
clib/gpuquery.h:	gpuContextP contexts [NCONTEXTS];
clib/gpuquery.h:} gpu_query_t;
clib/gpuquery.h:gpuQueryP gpu_query_new (int, cl_device_id, cl_context, const char *, int, int, int);
clib/gpuquery.h:void gpu_query_init (gpuQueryP, JNIEnv *, int);
clib/gpuquery.h:void gpu_query_setResultHandler (gpuQueryP, resultHandlerP);
clib/gpuquery.h:void gpu_query_free (gpuQueryP);
clib/gpuquery.h:gpuContextP gpu_context_switch (gpuQueryP);
clib/gpuquery.h:int gpu_query_setInput (gpuQueryP, int, int);
clib/gpuquery.h:int gpu_query_setOutput (gpuQueryP, int, int, int, int, int, int, int);
clib/gpuquery.h:int gpu_query_setKernel (gpuQueryP,
clib/gpuquery.h:		void (*callback)(cl_kernel, gpuContextP, int *, long *),
clib/gpuquery.h:int gpu_query_exec (gpuQueryP, size_t *, size_t *, queryOperatorP, JNIEnv *, jobject);
clib/gpuquery.h:#endif /* __GPU_QUERY_H_ */
clib/GPU.h:#ifndef __GPU_H_
clib/GPU.h:#define __GPU_H_
clib/GPU.h:#include "gpucontext.h"
clib/GPU.h:#include "openclerrorcode.h"
clib/GPU.h:	void (*writeInput) (gpuContextP, JNIEnv *, jobject, int, int);
clib/GPU.h:	void (*readOutput) (gpuContextP, JNIEnv *, jobject, int, int, int);
clib/GPU.h:	void (*configure) (cl_kernel, gpuContextP, int *, long *);
clib/GPU.h:	gpuContextP (*execKernel) (gpuContextP);
clib/GPU.h:void gpu_init (JNIEnv *, int, int);
clib/GPU.h:void gpu_free ();
clib/GPU.h:int gpu_getQuery (const char *, int, int, int);
clib/GPU.h:int gpu_setInput (int, int, int);
clib/GPU.h:int gpu_setOutput (int, int, int, int, int, int, int, int);
clib/GPU.h:int gpu_setKernel (int, int, const char *, void (*callback) (cl_kernel, gpuContextP, int *, long *), int *, long *);
clib/GPU.h:int gpu_exec (int, size_t *, size_t *, queryOperatorP, JNIEnv *, jobject);
clib/GPU.h:#endif /* SEEP_GPU_H_ */
clib/openclerrorcode.c:#include "openclerrorcode.h"
clib/openclerrorcode.c:#include <OpenCL/opencl.h>
clib/openclerrorcode.h:#ifndef OPENCL_ERROR_CODE_H_
clib/openclerrorcode.h:#define OPENCL_ERROR_CODE_H_
clib/openclerrorcode.h:#include <OpenCL/opencl.h>
clib/openclerrorcode.h:#endif /* OPENCL_ERROR_CODE_H_ */
clib/utils.h:#ifndef __GPU_UTILS_H_
clib/utils.h:#define __GPU_UTILS_H_
clib/utils.h:// #undef GPU_HANDLER
clib/utils.h:#define GPU_HANDLER
clib/utils.h:#endif /* __GPU_UTILS_H_ */
clib/gpucontext.c:#include "gpucontext.h"
clib/gpucontext.c:#include "openclerrorcode.h"
clib/gpucontext.c:#include <OpenCL/opencl.h>
clib/gpucontext.c:#ifdef GPU_VERBOSE
clib/gpucontext.c:		fprintf(stderr, "opencl error (%d): %s (%s)\n", error, getErrorMessage(error), __FUNCTION__);
clib/gpucontext.c:		fprintf(stderr, "opencl error (%d): %s (%s)\n", error, getErrorMessage(error), __FUNCTION__);
clib/gpucontext.c:		fprintf(stderr, "opencl error (%d): %s (%s)\n", error, getErrorMessage(error), __FUNCTION__);
clib/gpucontext.c:gpuContextP gpu_context (int qid, cl_device_id device, cl_context context, cl_program program,
clib/gpucontext.c:	gpuContextP q = (gpuContextP) malloc (sizeof(gpu_context_t));
clib/gpucontext.c:		fprintf(stderr, "opencl error (%d): %s (%s)\n", error, getErrorMessage(error), __FUNCTION__);
clib/gpucontext.c:		fprintf(stderr, "opencl error (%d): %s (%s)\n", error, getErrorMessage(error), __FUNCTION__);
clib/gpucontext.c:void gpu_context_free (gpuContextP q) {
clib/gpucontext.c:void gpu_context_setInput (gpuContextP q, int ndx, int size) {
clib/gpucontext.c:void gpu_context_setOutput (gpuContextP q, int ndx, int size,
clib/gpucontext.c:void gpu_context_setKernel (gpuContextP q,
clib/gpucontext.c:	void (*callback)(cl_kernel, gpuContextP, int *, long *),
clib/gpucontext.c:			fprintf(stderr, "opencl error (%d): %s (%s)\n", error, getErrorMessage(error), __FUNCTION__);
clib/gpucontext.c:void gpu_context_configureKernel (gpuContextP q,
clib/gpucontext.c:	void (*callback)(cl_kernel, gpuContextP, int *, long *),
clib/gpucontext.c:#ifdef GPU_PROFILE
clib/gpucontext.c:void gpu_context_profileQuery (gpuContextP q) {
clib/gpucontext.c:void gpu_context_waitForReadEvent (gpuContextP q) {
clib/gpucontext.c:		fprintf(stderr, "opencl error (%d): %s (%s)\n", 
clib/gpucontext.c:void gpu_context_waitForWriteEvent (gpuContextP q) {
clib/gpucontext.c:		fprintf(stderr, "opencl error (%d): %s (%s)\n", 
clib/gpucontext.c:void gpu_context_flush (gpuContextP q) {
clib/gpucontext.c:		fprintf(stderr, "opencl error (%d): %s (%s)\n", error, getErrorMessage(error), __FUNCTION__);
clib/gpucontext.c:void gpu_context_finish (gpuContextP q) {
clib/gpucontext.c:		fprintf(stderr, "opencl error (%d): %s (%s), q=%d @%p\n", error, getErrorMessage(error), __FUNCTION__, q->qid, q);
clib/gpucontext.c:void gpu_context_moveInputBuffers (gpuContextP q) {
clib/gpucontext.c:#ifdef GPU_PROFILE				
clib/gpucontext.c:			fprintf(stderr, "opencl error (%d): %s (%s)\n", error, getErrorMessage(error), __FUNCTION__);
clib/gpucontext.c:void gpu_context_submitKernel (gpuContextP q, size_t *threads, size_t *threadsPerGroup) {
clib/gpucontext.c:#ifdef GPU_PROFILE
clib/gpucontext.c:		fprintf(stderr, "opencl error (%d): %s (%s)\n", error, getErrorMessage(error), __FUNCTION__);
clib/gpucontext.c:void gpu_context_moveOutputBuffers (gpuContextP q) {
clib/gpucontext.c:#ifdef GPU_PROFILE
clib/gpucontext.c:		fprintf(stderr, "opencl error (%d): %s (%s)\n",
clib/gpucontext.c:void gpu_context_writeInput (gpuContextP q,
clib/gpucontext.c:	void (*callback)(gpuContextP, JNIEnv *, jobject, int, int),
clib/gpucontext.c:void gpu_context_readOutput (gpuContextP q,
clib/gpucontext.c:	void (*callback)(gpuContextP, JNIEnv *, jobject, int, int, int),
clib/templates/byteorder.h:#ifndef __GPU_BYTEORDER_H_
clib/templates/byteorder.h:#define __GPU_BYTEORDER_H_
clib/templates/byteorder.h:#endif /* __GPU_BYTEORDER_H_ */
clib/gpucontext.h:#ifndef __GPU_CONTEXT_H_
clib/gpucontext.h:#define __GPU_CONTEXT_H_
clib/gpucontext.h:#include <OpenCL/opencl.h>
clib/gpucontext.h:typedef struct gpu_kernel_input {
clib/gpucontext.h:} gpu_kernel_input_t;
clib/gpucontext.h:typedef struct gpu_kernel_output {
clib/gpucontext.h:} gpu_kernel_output_t;
clib/gpucontext.h:typedef struct gpu_kernel {
clib/gpucontext.h:} gpu_kernel_t;
clib/gpucontext.h:typedef struct gpu_context *gpuContextP;
clib/gpucontext.h:typedef struct gpu_context {
clib/gpucontext.h:	gpu_kernel_t kernel;
clib/gpucontext.h:	gpu_kernel_input_t kernelInput;
clib/gpucontext.h:	gpu_kernel_output_t kernelOutput;
clib/gpucontext.h:#ifdef GPU_PROFILE
clib/gpucontext.h:} gpu_context_t;
clib/gpucontext.h:gpuContextP gpu_context (int, cl_device_id, cl_context, cl_program, int, int, int);
clib/gpucontext.h:void gpu_context_free (gpuContextP);
clib/gpucontext.h:void gpu_context_setInput  (gpuContextP, int, int);
clib/gpucontext.h:void gpu_context_setOutput (gpuContextP, int, int, int, int, int, int, int);
clib/gpucontext.h:void gpu_context_setKernel (gpuContextP,
clib/gpucontext.h:		void (*callback)(cl_kernel, gpuContextP, int *, long *),
clib/gpucontext.h:void gpu_context_configureKernel (gpuContextP,
clib/gpucontext.h:		void (*callback)(cl_kernel, gpuContextP, int *, long *),
clib/gpucontext.h:void gpu_context_waitForReadEvent (gpuContextP);
clib/gpucontext.h:void gpu_context_waitForWriteEvent (gpuContextP);
clib/gpucontext.h:void gpu_context_profileQuery (gpuContextP);
clib/gpucontext.h:void gpu_context_flush (gpuContextP);
clib/gpucontext.h:void gpu_context_finish (gpuContextP);
clib/gpucontext.h:void gpu_context_moveInputBuffers (gpuContextP);
clib/gpucontext.h:void gpu_context_submitKernel (gpuContextP, size_t *, size_t *);
clib/gpucontext.h:void gpu_context_moveOutputBuffers (gpuContextP);
clib/gpucontext.h:void gpu_context_writeInput (gpuContextP,
clib/gpucontext.h:		void (*callback)(gpuContextP, JNIEnv *, jobject, int, int),
clib/gpucontext.h:void gpu_context_readOutput (gpuContextP,
clib/gpucontext.h:		void (*callback)(gpuContextP, JNIEnv *, jobject, int, int, int),
clib/gpucontext.h:#endif /* __GPU_CONTEXT_H_ */
clib/gpuquery.c:#include "gpuquery.h"
clib/gpuquery.c:#include "openclerrorcode.h"
clib/gpuquery.c:#include <OpenCL/opencl.h>
clib/gpuquery.c:static int gpu_query_exec_1 (gpuQueryP, size_t *, size_t *, queryOperatorP, JNIEnv *, jobject); /* w/o  pipelining */
clib/gpuquery.c:static int gpu_query_exec_2 (gpuQueryP, size_t *, size_t *, queryOperatorP, JNIEnv *, jobject); /* with pipelining */
clib/gpuquery.c:gpuQueryP gpu_query_new (int qid, cl_device_id device, cl_context context, const char *source,
clib/gpuquery.c:	 * based on the type of GPU device (i.e. NVIDIA or not)
clib/gpuquery.c:	gpuQueryP p = (gpuQueryP) malloc (sizeof(gpu_query_t));
clib/gpuquery.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/gpuquery.c:		fprintf(stderr, "opencl error (%d): %s\n", error, getErrorMessage(error));
clib/gpuquery.c:		p->contexts[i] = gpu_context (p->qid, p->device, p->context, p->program, _kernels, _inputs, _outputs);
clib/gpuquery.c:void gpu_query_setResultHandler (gpuQueryP q, resultHandlerP handler) {
clib/gpuquery.c:void gpu_query_free (gpuQueryP p) {
clib/gpuquery.c:			gpu_context_free (p->contexts[i]);
clib/gpuquery.c:int gpu_query_setInput (gpuQueryP q, int ndx, int size) {
clib/gpuquery.c:		gpu_context_setInput (q->contexts[i], ndx, size);
clib/gpuquery.c:int gpu_query_setOutput (gpuQueryP q, int ndx, int size, int writeOnly, int doNotMove, int bearsMark, int readEvent, int ignoreMark) {
clib/gpuquery.c:		gpu_context_setOutput (q->contexts[i], ndx, size, writeOnly, doNotMove, bearsMark, readEvent, ignoreMark);
clib/gpuquery.c:int gpu_query_setKernel (gpuQueryP q,
clib/gpuquery.c:	void (*callback)(cl_kernel, gpuContextP, int *, long *),
clib/gpuquery.c:		gpu_context_setKernel (q->contexts[i], ndx, name, callback, args1, args2);
clib/gpuquery.c:gpuContextP gpu_context_switch (gpuQueryP p) {
clib/gpuquery.c:#ifdef GPU_VERBOSE
clib/gpuquery.c:#ifdef GPU_VERBOSE
clib/gpuquery.c:int gpu_query_exec (gpuQueryP q, size_t *threads, size_t *threadsPerGroup, queryOperatorP operator, JNIEnv *env, jobject obj) {
clib/gpuquery.c:		return gpu_query_exec_1 (q, threads, threadsPerGroup, operator, env, obj);
clib/gpuquery.c:		return gpu_query_exec_2 (q, threads, threadsPerGroup, operator, env, obj);
clib/gpuquery.c:static int gpu_query_exec_1 (gpuQueryP q, size_t *threads, size_t *threadsPerGroup, queryOperatorP operator, JNIEnv *env, jobject obj) {
clib/gpuquery.c:	gpuContextP p = gpu_context_switch (q);
clib/gpuquery.c:	gpu_context_writeInput (p, operator->writeInput, env, obj, q->qid);
clib/gpuquery.c:	gpu_context_moveInputBuffers (p);
clib/gpuquery.c:		gpu_context_configureKernel (p, operator->configure, operator->args1, operator->args2);
clib/gpuquery.c:	gpu_context_submitKernel (p, threads, threadsPerGroup);
clib/gpuquery.c:	gpu_context_moveOutputBuffers (p);
clib/gpuquery.c:	gpu_context_flush (p);
clib/gpuquery.c:	gpu_context_finish(p);
clib/gpuquery.c:	gpu_context_readOutput (p, operator->readOutput, env, obj, q->qid);
clib/gpuquery.c:static int gpu_query_exec_2 (gpuQueryP q, size_t *threads, size_t *threadsPerGroup, queryOperatorP operator, JNIEnv *env, jobject obj) {
clib/gpuquery.c:	gpuContextP p = gpu_context_switch (q);
clib/gpuquery.c:	gpuContextP theOther = (operator->execKernel(p));
clib/gpuquery.c:		gpu_context_finish(theOther);
clib/gpuquery.c:#ifdef GPU_PROFILE
clib/gpuquery.c:		gpu_context_profileQuery (theOther);
clib/gpuquery.c:			gpu_context_readOutput (theOther, operator->readOutput, env, obj, q->qid);
clib/gpuquery.c:	gpu_context_writeInput (p, operator->writeInput, env, obj, q->qid);
clib/gpuquery.c:	gpu_context_moveInputBuffers (p);
clib/gpuquery.c:		gpu_context_configureKernel (p, operator->configure, operator->args1, operator->args2);
clib/gpuquery.c:	gpu_context_submitKernel (p, threads, threadsPerGroup);
clib/gpuquery.c:	gpu_context_moveOutputBuffers (p);
clib/gpuquery.c:	gpu_context_flush (p);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestSelection.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.SelectionKernel;
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestSelection.java:		IOperatorCode gpuCode = new SelectionKernel (schema, predicate, null, batchSize);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestSelection.java:		operator = new QueryOperator (cpuCode, gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestAggregation.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.AggregationKernel;
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestAggregation.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.ReductionKernel;
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestAggregation.java:		IOperatorCode gpuCode = null;
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestAggregation.java:			gpuCode = new ReductionKernel (window, aggregationTypes, aggregationAttributes, schema, batchSize);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestAggregation.java:			gpuCode = new AggregationKernel (window, aggregationTypes, aggregationAttributes, groupByAttributes, schema, batchSize);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestAggregation.java:		operator = new QueryOperator (cpuCode, gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestAggregation.java:			query.setAggregateOperator((IAggregateOperator) gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W1.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.AggregationKernel;
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W1.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.ProjectionKernel;
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W1.java:		IOperatorCode gpuCode1 = new ProjectionKernel (schema1, expressions, batchSize, 100);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W1.java:		operator1 = new QueryOperator (cpuCode1, gpuCode1);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W1.java:		IOperatorCode gpuCode2 = new AggregationKernel (window2, aggregationTypes, aggregationAttributes, groupByAttributes, schema2, batchSize);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W1.java:		operator2 = new QueryOperator (cpuCode2, gpuCode2);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W1.java:			query2.setAggregateOperator((IAggregateOperator) gpuCode2);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W2.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.ProjectionKernel;
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W2.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.ReductionKernel;
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W2.java:		 SystemConf.GPU = true;
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W2.java:		IOperatorCode gpuCode1 = new ProjectionKernel (schema1, expressions, batchSize, 1);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W2.java:		operator1 = new QueryOperator (cpuCode1, gpuCode1);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W2.java:		IOperatorCode gpuCode2 = new ReductionKernel (window2, aggregationTypes, aggregationAttributes, schema2, batchSize);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W2.java:		operator2 = new QueryOperator (cpuCode2, gpuCode2);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W2.java:			query2.setAggregateOperator((IAggregateOperator) gpuCode2);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W3.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.SelectionKernel;
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W3.java:		SystemConf.GPU = false;
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W3.java:		if (executionMode.toLowerCase().contains("gpu") || executionMode.toLowerCase().contains("hybrid"))
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W3.java:			SystemConf.GPU = true;
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W3.java:		SystemConf.HYBRID = SystemConf.CPU && SystemConf.GPU;
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W3.java:		IOperatorCode gpuCode = new SelectionKernel (schema, predicate, customPredicate.toString(), batchSize);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/scheduling/W3.java:		operator = new QueryOperator (cpuCode, gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestNoop.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.NoOpKernel;
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestNoop.java:		IOperatorCode gpuCode = new NoOpKernel (schema, batchSize);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestNoop.java:		operator = new QueryOperator (cpuCode, gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestProjection.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.ProjectionKernel;
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestProjection.java:		IOperatorCode gpuCode = new ProjectionKernel (schema, expressions, batchSize, projectionExpressionDepth);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestProjection.java:		operator = new QueryOperator (cpuCode, gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestThetaJoin.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.ThetaJoinKernel;
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestThetaJoin.java:		IOperatorCode gpuCode = new ThetaJoinKernel (schema1, schema2, predicate, null, batchSize, SystemConf.UNBOUNDED_BUFFER_SIZE);
src/test/java/uk/ac/imperial/lsds/saber/experiments/microbenchmarks/TestThetaJoin.java:		operator = new QueryOperator (cpuCode, gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestNoOp.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.NoOpKernel;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestNoOp.java://		String executionMode = "gpu";
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestNoOp.java://		SystemConf.GPU = false;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestNoOp.java://		if (executionMode.toLowerCase().contains("gpu") || executionMode.toLowerCase().contains("hybrid"))
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestNoOp.java://			SystemConf.GPU = true;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestNoOp.java://		SystemConf.HYBRID = SystemConf.CPU && SystemConf.GPU;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestNoOp.java://		IOperatorCode gpuCode = new NoOpKernel (schema, batchSize);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestNoOp.java://		operator = new QueryOperator (cpuCode, gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/DemoWithGoogleClusterData.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.SelectionKernel;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/DemoWithGoogleClusterData.java://		SystemConf.GPU = false;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/DemoWithGoogleClusterData.java://		if (executionMode.toLowerCase().contains("gpu") || executionMode.toLowerCase().contains("hybrid"))
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/DemoWithGoogleClusterData.java://			SystemConf.GPU = true;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/DemoWithGoogleClusterData.java://		SystemConf.HYBRID = SystemConf.CPU && SystemConf.GPU;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/DemoWithGoogleClusterData.java://		/* GPU operator predicate */
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/DemoWithGoogleClusterData.java://		IOperatorCode gpuCode = new SelectionKernel (schema, predicate, customPredicate.toString(), batchSize);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/DemoWithGoogleClusterData.java://		operator = new QueryOperator (cpuCode, gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestThetaJoinSelectivity.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.ThetaJoinKernel;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestThetaJoinSelectivity.java://		SystemConf.GPU = false;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestThetaJoinSelectivity.java://		if (executionMode.toLowerCase().contains("gpu") || executionMode.toLowerCase().contains("hybrid"))
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestThetaJoinSelectivity.java://			SystemConf.GPU = true;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestThetaJoinSelectivity.java://		SystemConf.HYBRID = SystemConf.CPU && SystemConf.GPU;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestThetaJoinSelectivity.java://		IOperatorCode gpuCode = new ThetaJoinKernel (schema1, schema2, predicate, null, batchSize, 1048576);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestThetaJoinSelectivity.java://		operator = new QueryOperator (cpuCode, gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAggregation.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.AggregationKernel;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAggregation.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.ReductionKernel;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAggregation.java://		String executionMode = "gpu";
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAggregation.java://		SystemConf.GPU = false;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAggregation.java://		if (executionMode.toLowerCase().contains("gpu") || executionMode.toLowerCase().contains("hybrid"))
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAggregation.java://			SystemConf.GPU = true;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAggregation.java://		SystemConf.HYBRID = SystemConf.CPU && SystemConf.GPU;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAggregation.java://		IOperatorCode gpuCode;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAggregation.java://			gpuCode = new ReductionKernel (window, aggregationTypes, aggregationAttributes, schema, batchSize);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAggregation.java://			gpuCode = new AggregationKernel (window, aggregationTypes, aggregationAttributes, groupByAttributes, schema, batchSize);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAggregation.java://		operator = new QueryOperator (cpuCode, gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAggregation.java://			query.setAggregateOperator((IAggregateOperator) gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestSelectionPredicates.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.SelectionKernel;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestSelectionPredicates.java://		SystemConf.GPU = false;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestSelectionPredicates.java://		if (executionMode.toLowerCase().contains("gpu") || executionMode.toLowerCase().contains("hybrid"))
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestSelectionPredicates.java://			SystemConf.GPU = true;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestSelectionPredicates.java://		SystemConf.HYBRID = SystemConf.CPU && SystemConf.GPU;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestSelectionPredicates.java://		IOperatorCode gpuCode = new SelectionKernel (schema, predicate, null, batchSize);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestSelectionPredicates.java://		operator = new QueryOperator (cpuCode, gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAdaptivity.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.SelectionKernel;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAdaptivity.java://		SystemConf.GPU = false;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAdaptivity.java://		if (executionMode.toLowerCase().contains("gpu") || executionMode.toLowerCase().contains("hybrid"))
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAdaptivity.java://			SystemConf.GPU = true;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAdaptivity.java://		SystemConf.HYBRID = SystemConf.CPU && SystemConf.GPU;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAdaptivity.java://		IOperatorCode gpuCode = new SelectionKernel (schema, predicate, customPredicate.toString(), batchSize);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAdaptivity.java://		operator = new QueryOperator (cpuCode, gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestTwoNoOps.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.NoOpKernel;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestTwoNoOps.java://		SystemConf.GPU = false;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestTwoNoOps.java://		if (executionMode.toLowerCase().contains("gpu") || executionMode.toLowerCase().contains("hybrid"))
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestTwoNoOps.java://			SystemConf.GPU = true;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestTwoNoOps.java://		SystemConf.HYBRID = SystemConf.CPU && SystemConf.GPU;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestTwoNoOps.java://		IOperatorCode gpuCode1 = new NoOpKernel (schema1, batchSize);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestTwoNoOps.java://		operator1 = new QueryOperator (cpuCode1, gpuCode1);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestTwoNoOps.java://		IOperatorCode gpuCode2 = new NoOpKernel (schema2, batchSize);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestTwoNoOps.java://		operator2 = new QueryOperator (cpuCode2, gpuCode2);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/SyntheticReduction.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.AggregationKernel;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/SyntheticReduction.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.ReductionKernel;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/SyntheticReduction.java://		String executionMode = "gpu";
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/SyntheticReduction.java://		SystemConf.GPU = false;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/SyntheticReduction.java://		if (executionMode.toLowerCase().contains("gpu") || executionMode.toLowerCase().contains("hybrid"))
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/SyntheticReduction.java://			SystemConf.GPU = true;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/SyntheticReduction.java://		SystemConf.HYBRID = SystemConf.CPU && SystemConf.GPU;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/SyntheticReduction.java://		IOperatorCode gpuCode;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/SyntheticReduction.java://			gpuCode = new ReductionKernel (window, aggregationTypes, aggregationAttributes, schema, batchSize);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/SyntheticReduction.java://			gpuCode = new AggregationKernel (window, aggregationTypes, aggregationAttributes, groupByAttributes, schema, batchSize);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/SyntheticReduction.java://		operator = new QueryOperator (cpuCode, gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/SyntheticReduction.java://			query.setAggregateOperator((IAggregateOperator) gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAdaptivityWithGoogleClusterData.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.SelectionKernel;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAdaptivityWithGoogleClusterData.java://		SystemConf.GPU = false;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAdaptivityWithGoogleClusterData.java://		if (executionMode.toLowerCase().contains("gpu") || executionMode.toLowerCase().contains("hybrid"))
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAdaptivityWithGoogleClusterData.java://			SystemConf.GPU = true;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAdaptivityWithGoogleClusterData.java://		SystemConf.HYBRID = SystemConf.CPU && SystemConf.GPU;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAdaptivityWithGoogleClusterData.java://		IOperatorCode gpuCode = new SelectionKernel (schema, predicate, customPredicate.toString(), batchSize);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestAdaptivityWithGoogleClusterData.java://		operator = new QueryOperator (cpuCode, gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestThetaJoinPredicates.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.ThetaJoinKernel;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestThetaJoinPredicates.java://		String executionMode = "gpu";
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestThetaJoinPredicates.java://		SystemConf.GPU = false;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestThetaJoinPredicates.java://		if (executionMode.toLowerCase().contains("gpu") || executionMode.toLowerCase().contains("hybrid"))
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestThetaJoinPredicates.java://			SystemConf.GPU = true;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestThetaJoinPredicates.java://		SystemConf.HYBRID = SystemConf.CPU && SystemConf.GPU;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestThetaJoinPredicates.java://		IOperatorCode gpuCode = new ThetaJoinKernel (schema1, schema2, predicate, null, batchSize, 1048576);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestThetaJoinPredicates.java://		operator = new QueryOperator (cpuCode, gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/hls/Test1.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.NoOpKernel;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/hls/Test1.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.ProjectionKernel;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/hls/Test1.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.ReductionKernel;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/hls/Test1.java://		SystemConf.GPU = false;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/hls/Test1.java://		if (executionMode.toLowerCase().contains("gpu") || executionMode.toLowerCase().contains("hybrid"))
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/hls/Test1.java://			SystemConf.GPU = true;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/hls/Test1.java://		SystemConf.HYBRID = SystemConf.CPU && SystemConf.GPU;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/hls/Test1.java://		IOperatorCode gpuCode1 = new ProjectionKernel (schema1, expressions, batchSize, 1);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/hls/Test1.java://		operator1 = new QueryOperator (cpuCode1, gpuCode1);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/hls/Test1.java://		IOperatorCode gpuCode2 = new ReductionKernel (window2, aggregationTypes, aggregationAttributes, schema2, batchSize);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/hls/Test1.java://		operator2 = new QueryOperator (cpuCode2, gpuCode2);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/hls/Test1.java://			query2.setAggregateOperator((IAggregateOperator) gpuCode2);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestProjection.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.ProjectionKernel;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestProjection.java://		SystemConf.GPU = false;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestProjection.java://		if (executionMode.toLowerCase().contains("gpu") || executionMode.toLowerCase().contains("hybrid"))
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestProjection.java://			SystemConf.GPU = true;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestProjection.java://		SystemConf.HYBRID = SystemConf.CPU && SystemConf.GPU;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestProjection.java://		IOperatorCode gpuCode = new ProjectionKernel (schema, expressions, batchSize, expressionDepth);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestProjection.java://		operator = new QueryOperator (cpuCode, gpuCode);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestSelectionSelectivity.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.SelectionKernel;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestSelectionSelectivity.java://		String executionMode = "gpu";
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestSelectionSelectivity.java://		SystemConf.GPU = false;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestSelectionSelectivity.java://		if (executionMode.toLowerCase().contains("gpu") || executionMode.toLowerCase().contains("hybrid"))
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestSelectionSelectivity.java://			SystemConf.GPU = true;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestSelectionSelectivity.java://		SystemConf.HYBRID = SystemConf.CPU && SystemConf.GPU;
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestSelectionSelectivity.java://		IOperatorCode gpuCode = new SelectionKernel (schema, predicate, null, batchSize);
src/test/java/uk/ac/imperial/lsds/saber/microbenchmarks/TestSelectionSelectivity.java://		operator = new QueryOperator (cpuCode, gpuCode);
src/main/java/uk/ac/imperial/lsds/saber/devices/TheGPU.java:public class TheGPU {
src/main/java/uk/ac/imperial/lsds/saber/devices/TheGPU.java:	private static final String gpuLibrary = SystemConf.SABER_HOME + "/clib/libGPU.so";
src/main/java/uk/ac/imperial/lsds/saber/devices/TheGPU.java:	private static final TheGPU gpuInstance = new TheGPU (5, 10);
src/main/java/uk/ac/imperial/lsds/saber/devices/TheGPU.java:	public static TheGPU getInstance () { return gpuInstance; }
src/main/java/uk/ac/imperial/lsds/saber/devices/TheGPU.java:	/* Managing GPU pipelining of multiple queries */
src/main/java/uk/ac/imperial/lsds/saber/devices/TheGPU.java:	public TheGPU (int q, int b) {
src/main/java/uk/ac/imperial/lsds/saber/devices/TheGPU.java:			System.load (gpuLibrary);
src/main/java/uk/ac/imperial/lsds/saber/devices/TheGPU.java:			System.err.println("error: failed to load GPU library");
src/main/java/uk/ac/imperial/lsds/saber/handlers/PartialResultSlot.java:	boolean GPU = false;
src/main/java/uk/ac/imperial/lsds/saber/handlers/PartialResultSlot.java:		GPU = false;
src/main/java/uk/ac/imperial/lsds/saber/handlers/PartialResultSlot.java:		GPU = false;
src/main/java/uk/ac/imperial/lsds/saber/handlers/PartialResultSlot.java:		s.append(String.format("GPU: %5s", GPU));
src/main/java/uk/ac/imperial/lsds/saber/www/MeasurementQueue.java:		float cpuValue, gpuValue;
src/main/java/uk/ac/imperial/lsds/saber/www/MeasurementQueue.java:			cpuValue = gpuValue = 0F;
src/main/java/uk/ac/imperial/lsds/saber/www/MeasurementQueue.java:		public void set (long timestamp, float cpuValue, float gpuValue) {
src/main/java/uk/ac/imperial/lsds/saber/www/MeasurementQueue.java:			this.gpuValue = gpuValue;
src/main/java/uk/ac/imperial/lsds/saber/www/MeasurementQueue.java:	public void add (long timestamp, float cpuValue, float gpuValue) { /* Thread-safe */
src/main/java/uk/ac/imperial/lsds/saber/www/MeasurementQueue.java:		p.set (timestamp, cpuValue, gpuValue);
src/main/java/uk/ac/imperial/lsds/saber/www/MeasurementQueue.java:					if (mode.equals("gpu")) results.add(p.gpuValue);
src/main/java/uk/ac/imperial/lsds/saber/www/MeasurementQueue.java:						results.add(p.cpuValue + p.gpuValue);
src/main/java/uk/ac/imperial/lsds/saber/www/ThroughputHandler.java:	public void addMeasurement (long timestamp, float cpuValue, float gpuValue) {
src/main/java/uk/ac/imperial/lsds/saber/www/ThroughputHandler.java:		queue.add (timestamp, cpuValue, gpuValue);
src/main/java/uk/ac/imperial/lsds/saber/www/RESTfulHandler.java:	public void addMeasurement (int qid, long timestamp, float cpuValue, float gpuValue) {
src/main/java/uk/ac/imperial/lsds/saber/www/RESTfulHandler.java:		t[qid].addMeasurement(timestamp, cpuValue, gpuValue);
src/main/java/uk/ac/imperial/lsds/saber/buffers/PartialWindowResults.java:	 * This method is called when we fetch complete windows from the GPU.
src/main/java/uk/ac/imperial/lsds/saber/SystemConf.java:	public static boolean GPU = false;
src/main/java/uk/ac/imperial/lsds/saber/SystemConf.java:				CPU = true; GPU = false; HYBRID = false;
src/main/java/uk/ac/imperial/lsds/saber/SystemConf.java:			if (arg.compareTo("gpu") == 0) {
src/main/java/uk/ac/imperial/lsds/saber/SystemConf.java:				GPU = true; CPU = false; HYBRID = false;
src/main/java/uk/ac/imperial/lsds/saber/SystemConf.java:				HYBRID = CPU = GPU = true;
src/main/java/uk/ac/imperial/lsds/saber/SystemConf.java:		s.append(String.format("Execution mode               : CPU %s GPU %s\n", CPU, GPU));
src/main/java/uk/ac/imperial/lsds/saber/SystemConf.java:		s.append(String.format("GPU pipeline depth           : %d\n", PIPELINE_DEPTH));
src/main/java/uk/ac/imperial/lsds/saber/tasks/AbstractTask.java:	protected boolean GPU = false;
src/main/java/uk/ac/imperial/lsds/saber/tasks/AbstractTask.java:	public void setGPU (boolean GPU) {
src/main/java/uk/ac/imperial/lsds/saber/tasks/AbstractTask.java:		this.GPU = GPU;
src/main/java/uk/ac/imperial/lsds/saber/tasks/Task.java:			next.process(batch1, this, GPU);
src/main/java/uk/ac/imperial/lsds/saber/tasks/Task.java:			next.process(batch1, batch2, this, GPU);
src/main/java/uk/ac/imperial/lsds/saber/dispatchers/TaskDispatcher.java:		/* The single, system-wide task queue for either CPU or GPU tasks */
src/main/java/uk/ac/imperial/lsds/saber/QueryApplication.java:import uk.ac.imperial.lsds.saber.devices.TheGPU;
src/main/java/uk/ac/imperial/lsds/saber/QueryApplication.java:	private int M = 2; /* CPU and GPGPU */
src/main/java/uk/ac/imperial/lsds/saber/QueryApplication.java:		if (SystemConf.GPU) {
src/main/java/uk/ac/imperial/lsds/saber/QueryApplication.java:			TheGPU.getInstance().load ();
src/main/java/uk/ac/imperial/lsds/saber/QueryApplication.java:			TheGPU.getInstance().init (N, SystemConf.PIPELINE_DEPTH);
src/main/java/uk/ac/imperial/lsds/saber/QueryApplication.java:		workerPool = new TaskProcessorPool(threads, queue, policy, SystemConf.GPU, SystemConf.HYBRID);
src/main/java/uk/ac/imperial/lsds/saber/processors/TaskProcessor.java:	boolean GPU, hybrid;
src/main/java/uk/ac/imperial/lsds/saber/processors/TaskProcessor.java:	private int cid = 0; /* Processor class: GPU (0) or CPU (1) */
src/main/java/uk/ac/imperial/lsds/saber/processors/TaskProcessor.java:	public TaskProcessor (int pid, TaskQueue queue, int [][] policy, boolean GPU, boolean hybrid) {
src/main/java/uk/ac/imperial/lsds/saber/processors/TaskProcessor.java:		this.GPU = GPU;
src/main/java/uk/ac/imperial/lsds/saber/processors/TaskProcessor.java:		if (GPU) 
src/main/java/uk/ac/imperial/lsds/saber/processors/TaskProcessor.java:		int min = (hybrid ? 3 : 1); /* +1 dispatcher, +1 GPU, if available */
src/main/java/uk/ac/imperial/lsds/saber/processors/TaskProcessor.java:		if (GPU) {
src/main/java/uk/ac/imperial/lsds/saber/processors/TaskProcessor.java:			System.out.println ("[DBG] GPU thread is " + Thread.currentThread());
src/main/java/uk/ac/imperial/lsds/saber/processors/TaskProcessor.java:				// System.out.println(String.format("[DBG] processor %2d task %d.%6d (GPU %5s)", pid, task.queryid, task.taskid, GPU));
src/main/java/uk/ac/imperial/lsds/saber/processors/TaskProcessor.java:				task.setGPU(GPU);
src/main/java/uk/ac/imperial/lsds/saber/processors/TaskProcessorPool.java:	public TaskProcessorPool (int workers, final TaskQueue queue, int [][] policy, boolean GPU, boolean hybrid) {
src/main/java/uk/ac/imperial/lsds/saber/processors/TaskProcessorPool.java:			/* Assign the first processor to be the GPU worker */
src/main/java/uk/ac/imperial/lsds/saber/processors/TaskProcessorPool.java:			if (GPU) {
src/main/java/uk/ac/imperial/lsds/saber/processors/TaskProcessorPool.java:				/* GPGPU-only */
src/main/java/uk/ac/imperial/lsds/saber/processors/TaskProcessorPool.java:				System.out.println("[DBG] GPGPU-only execution");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/cpu/Projection.java:		throw new UnsupportedOperationException("error: `configureOutput` method is applicable only to GPU operators");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/cpu/Projection.java:		throw new UnsupportedOperationException("error: `processOutput` method is applicable only to GPU operators");	
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/cpu/Projection.java:		throw new UnsupportedOperationException("error: `setup` method is applicable only to GPU operators");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/cpu/NoOp.java:		throw new UnsupportedOperationException("error: `configureOutput` method is applicable only to GPU operators");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/cpu/NoOp.java:		throw new UnsupportedOperationException("error: `processOutput` method is applicable only to GPU operators");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/cpu/NoOp.java:		throw new UnsupportedOperationException("error: `setup` method is applicable only to GPU operators");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/cpu/Aggregation.java:		throw new UnsupportedOperationException("error: `configureOutput` method is applicable only to GPU operators");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/cpu/Aggregation.java:		throw new UnsupportedOperationException("error: `processOutput` method is applicable only to GPU operators");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/cpu/Aggregation.java:		throw new UnsupportedOperationException("error: `setup` method is applicable only to GPU operators");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/cpu/ThetaJoin.java:		throw new UnsupportedOperationException("error: `configureOutput` method is applicable only to GPU operators");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/cpu/ThetaJoin.java:		throw new UnsupportedOperationException("error: `processOutput` method is applicable only to GPU operators");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/cpu/ThetaJoin.java:		throw new UnsupportedOperationException("error: `setup` method is applicable only to GPU operators");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/cpu/Selection.java:		throw new UnsupportedOperationException("error: `configureOutput` method is applicable only to GPU operators");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/cpu/Selection.java:		throw new UnsupportedOperationException("error: `processOutput` method is applicable only to GPU operators");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/cpu/Selection.java:		throw new UnsupportedOperationException("error: `setup` method is applicable only to GPU operators");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoin.java:		throw new UnsupportedOperationException("error: `configureOutput` method is applicable only to GPU operators");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoin.java:		throw new UnsupportedOperationException("error: `processOutput` method is applicable only to GPU operators");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoin.java:		throw new UnsupportedOperationException("error: `setup` method is applicable only to GPU operators");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.code.generators.KernelGenerator;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:import uk.ac.imperial.lsds.saber.devices.TheGPU;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		qid = TheGPU.getInstance().getQuery(source, 4, 4, 4);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		TheGPU.getInstance().setInput(qid, 0, batchSize);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		TheGPU.getInstance().setInput(qid, 1, batchSize);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		TheGPU.getInstance().setInput(qid, 2, startPointers.capacity());
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		TheGPU.getInstance().setInput(qid, 3,   endPointers.capacity());
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		TheGPU.getInstance().setOutput(qid, 0,              records * 4, 0, 1, 1, 0, 1); /* counts */
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		TheGPU.getInstance().setOutput(qid, 1,              records * 4, 0, 1, 0, 0, 1); /* offsets */
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		TheGPU.getInstance().setOutput(qid, 2, numberOfThreadGroups * 4, 0, 1, 0, 0, 1); /* partitions */
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		TheGPU.getInstance().setOutput(qid, 3,           outputSize    , 1, 0, 0, 1, 0);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		TheGPU.getInstance().setKernelThetaJoin(qid, args);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		TheGPU.getInstance().setInputBuffer(qid, 0, inputBuffer1, start1, end1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		TheGPU.getInstance().setInputBuffer(qid, 1, inputBuffer2, start2, end2);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		TheGPU.getInstance().setInputBuffer (qid, 2, startPointers);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		TheGPU.getInstance().setInputBuffer (qid, 3,   endPointers);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		WindowBatch pipelinedBatch = TheGPU.getInstance().shiftUp(first);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:			pipelinedOperator = pipelinedBatch.getQuery().getMostUpstreamOperator().getGpuCode();
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		TheGPU.getInstance().execute(qid, threads, threadsPerGroup);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		TheGPU.getInstance().setOutputBuffer(queryId, 3, outputBuffer);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/udfs/MonetDBComparisonThetaJoinKernel.java:		IQueryBuffer buffer = TheGPU.getInstance().getOutputBuffer(queryId, 3);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:package uk.ac.imperial.lsds.saber.cql.operators.gpu;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.code.generators.KernelGenerator;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:import uk.ac.imperial.lsds.saber.devices.TheGPU;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:	 * in a batch. This buffer is thread-safe, since only the GPU thread uses it.
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:		qid = TheGPU.getInstance().getQuery (source, 4, 1, 5);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:		TheGPU.getInstance().setInput (qid, 0, inputSize);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:		TheGPU.getInstance().setOutput(qid, 0, windowPointersSize, 0, 1, 0, 0, 1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:		TheGPU.getInstance().setOutput(qid, 1, windowPointersSize, 0, 1, 0, 0, 1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:		TheGPU.getInstance().setOutput(qid, 2, offsetSize, 0, 1, 0, 0, 1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:		TheGPU.getInstance().setOutput(qid, 3, windowCountsSize, 0, 0, 1, 0, 1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:		TheGPU.getInstance().setOutput(qid, 4, outputSize, 1, 0, 0, 1, 0);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:		TheGPU.getInstance().setKernelReduce (qid, args1, args2);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:		TheGPU.getInstance().setInputBuffer(qid, 0, inputBuffer, start, end);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:		WindowBatch pipelinedBatch = TheGPU.getInstance().shiftUp(batch);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:			pipelinedOperator = pipelinedBatch.getQuery().getMostUpstreamOperator().getGpuCode();
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:		TheGPU.getInstance().executeReduce (qid, threads, threadsPerGroup, args2);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:		TheGPU.getInstance().setOutputBuffer(queryId, 3, windowCounts);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:		TheGPU.getInstance().setOutputBuffer(queryId, 4, outputBuffer);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ReductionKernel.java:		IQueryBuffer buffer = TheGPU.getInstance().getOutputBuffer(queryId, 4);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:package uk.ac.imperial.lsds.saber.cql.operators.gpu;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.code.generators.KernelGenerator;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:import uk.ac.imperial.lsds.saber.devices.TheGPU;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		qid = TheGPU.getInstance().getQuery(source, 4, 4, 4);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		TheGPU.getInstance().setInput(qid, 0, batchSize);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		TheGPU.getInstance().setInput(qid, 1, batchSize);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		TheGPU.getInstance().setInput(qid, 2, startPointers.capacity());
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		TheGPU.getInstance().setInput(qid, 3,   endPointers.capacity());
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		TheGPU.getInstance().setOutput(qid, 0,              records * 4, 0, 1, 1, 0, 1); /* counts */
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		TheGPU.getInstance().setOutput(qid, 1,              records * 4, 0, 1, 0, 0, 1); /* offsets */
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		TheGPU.getInstance().setOutput(qid, 2, numberOfThreadGroups * 4, 0, 1, 0, 0, 1); /* partitions */
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		TheGPU.getInstance().setOutput(qid, 3,           outputSize    , 1, 0, 0, 1, 0);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		TheGPU.getInstance().setKernelThetaJoin(qid, args);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		TheGPU.getInstance().setInputBuffer(qid, 0, inputBuffer1, start1, end1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		TheGPU.getInstance().setInputBuffer(qid, 1, inputBuffer2, start2, end2);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		TheGPU.getInstance().setInputBuffer (qid, 2, startPointers);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		TheGPU.getInstance().setInputBuffer (qid, 3,   endPointers);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		WindowBatch pipelinedBatch = TheGPU.getInstance().shiftUp(first);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:			pipelinedOperator = pipelinedBatch.getQuery().getMostUpstreamOperator().getGpuCode();
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		TheGPU.getInstance().execute(qid, threads, threadsPerGroup);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		TheGPU.getInstance().setOutputBuffer(queryId, 3, outputBuffer);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ThetaJoinKernel.java:		IQueryBuffer buffer = TheGPU.getInstance().getOutputBuffer(queryId, 3);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ProjectionKernel.java:package uk.ac.imperial.lsds.saber.cql.operators.gpu;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ProjectionKernel.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.code.generators.KernelGenerator;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ProjectionKernel.java:import uk.ac.imperial.lsds.saber.devices.TheGPU;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ProjectionKernel.java:		qid = TheGPU.getInstance().getQuery (source, 1, 1, 1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ProjectionKernel.java:		TheGPU.getInstance().setInput (qid, 0, inputSize);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ProjectionKernel.java:		TheGPU.getInstance().setOutput (qid, 0, outputSize, 1, 0, 0, 1, 1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ProjectionKernel.java:		TheGPU.getInstance().setKernelProject (qid, args);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ProjectionKernel.java:		TheGPU.getInstance().setInputBuffer(qid, 0, inputBuffer, start, end);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ProjectionKernel.java:		WindowBatch pipelinedBatch = TheGPU.getInstance().shiftUp(batch);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ProjectionKernel.java:			pipelinedOperator = pipelinedBatch.getQuery().getMostUpstreamOperator().getGpuCode();
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ProjectionKernel.java:		TheGPU.getInstance().execute (qid, threads, threadsPerGroup);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ProjectionKernel.java:		TheGPU.getInstance().setOutputBuffer (queryId, 0, outputBuffer);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/ProjectionKernel.java:		IQueryBuffer buffer = TheGPU.getInstance().getOutputBuffer (queryId, 0);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/SelectionKernel.java:package uk.ac.imperial.lsds.saber.cql.operators.gpu;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/SelectionKernel.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.code.generators.KernelGenerator;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/SelectionKernel.java:import uk.ac.imperial.lsds.saber.devices.TheGPU;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/SelectionKernel.java:		qid = TheGPU.getInstance().getQuery (source, 2, 1, 4);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/SelectionKernel.java:		TheGPU.getInstance().setInput (qid, 0, inputSize);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/SelectionKernel.java:		TheGPU.getInstance().setOutput (qid, 0, 4 * records,              0, 1, 1, 0, 1); /*      Flags */
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/SelectionKernel.java:		TheGPU.getInstance().setOutput (qid, 1, 4 * records,              0, 1, 0, 0, 1); /*    Offsets */
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/SelectionKernel.java:		TheGPU.getInstance().setOutput (qid, 2, 4 * numberOfThreadGroups, 0, 1, 0, 0, 1); /* Partitions */
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/SelectionKernel.java:		TheGPU.getInstance().setOutput (qid, 3, outputSize,               1, 0, 0, 1, 0); /*    Results */
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/SelectionKernel.java:		TheGPU.getInstance().setKernelSelect (qid, args);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/SelectionKernel.java:		TheGPU.getInstance().setInputBuffer(qid, 0, inputBuffer, start, end);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/SelectionKernel.java:		WindowBatch pipelinedBatch = TheGPU.getInstance().shiftUp(batch);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/SelectionKernel.java:			pipelinedOperator = pipelinedBatch.getQuery().getMostUpstreamOperator().getGpuCode();
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/SelectionKernel.java:		TheGPU.getInstance().execute(qid, threads, threadsPerGroup);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/SelectionKernel.java:		TheGPU.getInstance().setOutputBuffer(queryId, 3, outputBuffer);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/SelectionKernel.java:		IQueryBuffer buffer = TheGPU.getInstance().getOutputBuffer(queryId, 3);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/code/generators/NoopKernelGenerator.java:package uk.ac.imperial.lsds.saber.cql.operators.gpu.code.generators;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/code/generators/ReductionKernelGenerator.java:package uk.ac.imperial.lsds.saber.cql.operators.gpu.code.generators;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/code/generators/AggregationKernelGenerator.java:package uk.ac.imperial.lsds.saber.cql.operators.gpu.code.generators;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/code/generators/ProjectionKernelGenerator.java:package uk.ac.imperial.lsds.saber.cql.operators.gpu.code.generators;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/code/generators/KernelGenerator.java:package uk.ac.imperial.lsds.saber.cql.operators.gpu.code.generators;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/code/generators/KernelGenerator.java:		b.append("#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable\n");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/code/generators/KernelGenerator.java:		b.append("#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable\n");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/code/generators/KernelGenerator.java:		b.append("#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable\n");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/code/generators/KernelGenerator.java:		b.append("#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable\n");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/code/generators/KernelGenerator.java:		b.append("#pragma OPENCL EXTENSION cl_khr_fp64: enable\n");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/code/generators/KernelGenerator.java:		b.append("#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable\n");
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/code/generators/ThetaJoinKernelGenerator.java:package uk.ac.imperial.lsds.saber.cql.operators.gpu.code.generators;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/code/generators/SelectionKernelGenerator.java:package uk.ac.imperial.lsds.saber.cql.operators.gpu.code.generators;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:package uk.ac.imperial.lsds.saber.cql.operators.gpu;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.code.generators.KernelGenerator;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:import uk.ac.imperial.lsds.saber.devices.TheGPU;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:	 * in a batch. This buffer is thread-safe, since only the GPU thread uses it.
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		qid = TheGPU.getInstance().getQuery(source, 9, 1, 9);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		TheGPU.getInstance().setInput (qid, 0, inputSize);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		TheGPU.getInstance().setOutput(qid, 0, windowPointersSize, 0, 1, 0, 0, 1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		TheGPU.getInstance().setOutput(qid, 1, windowPointersSize, 0, 1, 0, 0, 1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		TheGPU.getInstance().setOutput(qid, 2, failedFlagsSize, 1, 1, 0, 0, 1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		TheGPU.getInstance().setOutput(qid, 3, offsetSize, 0, 1, 0, 0, 1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		TheGPU.getInstance().setOutput(qid, 4, windowCountsSize, 0, 0, 1, 0, 1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		TheGPU.getInstance().setOutput(qid, 5, outputSize, 1, 0, 0, 1, 1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		TheGPU.getInstance().setOutput(qid, 6, outputSize, 1, 0, 0, 1, 1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		TheGPU.getInstance().setOutput(qid, 7, outputSize, 1, 0, 0, 1, 1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		TheGPU.getInstance().setOutput(qid, 8, outputSize, 1, 0, 0, 1, 1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		TheGPU.getInstance().setKernelAggregate (qid, args1, args2);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		TheGPU.getInstance().setInputBuffer(qid, 0, inputBuffer, start, end);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		WindowBatch pipelinedBatch = TheGPU.getInstance().shiftUp(batch);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:			pipelinedOperator = pipelinedBatch.getQuery().getMostUpstreamOperator().getGpuCode();
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		TheGPU.getInstance().executeAggregate (qid, threads, threadsPerGroup, args2);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		TheGPU.getInstance().setOutputBuffer(queryId, 4, windowCounts);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		TheGPU.getInstance().setOutputBuffer(queryId, 5, outputBuffer5);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/AggregationKernel.java:		IQueryBuffer buffer = TheGPU.getInstance().getOutputBuffer(queryId, 5);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/NoOpKernel.java:package uk.ac.imperial.lsds.saber.cql.operators.gpu;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/NoOpKernel.java:import uk.ac.imperial.lsds.saber.cql.operators.gpu.code.generators.KernelGenerator;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/NoOpKernel.java:import uk.ac.imperial.lsds.saber.devices.TheGPU;
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/NoOpKernel.java:		qid = TheGPU.getInstance().getQuery(source, 1, 1, 1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/NoOpKernel.java:		TheGPU.getInstance().setInput (qid, 0, inputSize);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/NoOpKernel.java:		TheGPU.getInstance().setOutput (qid, 0, inputSize, 1, 0, 0, 1, 1);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/NoOpKernel.java:		TheGPU.getInstance().setKernelDummy (qid, null);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/NoOpKernel.java:		TheGPU.getInstance().setInputBuffer(qid, 0, inputBuffer, start, end);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/NoOpKernel.java:		WindowBatch pipelinedBatch = TheGPU.getInstance().shiftUp(batch);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/NoOpKernel.java:			pipelinedOperator = pipelinedBatch.getQuery().getMostUpstreamOperator().getGpuCode();
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/NoOpKernel.java:		TheGPU.getInstance().execute (qid, threads, threadsPerGroup);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/NoOpKernel.java:		TheGPU.getInstance().setOutputBuffer (queryId, 0, outputBuffer);
src/main/java/uk/ac/imperial/lsds/saber/cql/operators/gpu/NoOpKernel.java:		IQueryBuffer buffer = TheGPU.getInstance().getOutputBuffer (queryId, 0);
src/main/java/uk/ac/imperial/lsds/saber/QueryOperator.java:	private IOperatorCode gpuCode;
src/main/java/uk/ac/imperial/lsds/saber/QueryOperator.java:	public QueryOperator (IOperatorCode cpuCode, IOperatorCode gpuCode) {
src/main/java/uk/ac/imperial/lsds/saber/QueryOperator.java:		this.gpuCode = gpuCode;
src/main/java/uk/ac/imperial/lsds/saber/QueryOperator.java:	public IOperatorCode getGpuCode() {
src/main/java/uk/ac/imperial/lsds/saber/QueryOperator.java:		return gpuCode;
src/main/java/uk/ac/imperial/lsds/saber/QueryOperator.java:	public void process (WindowBatch batch, IWindowAPI api, boolean GPU) {
src/main/java/uk/ac/imperial/lsds/saber/QueryOperator.java:		if (GPU)
src/main/java/uk/ac/imperial/lsds/saber/QueryOperator.java:			gpuCode.processData(batch, api);
src/main/java/uk/ac/imperial/lsds/saber/QueryOperator.java:	public void process (WindowBatch first, WindowBatch second, IWindowAPI api, boolean GPU) {
src/main/java/uk/ac/imperial/lsds/saber/QueryOperator.java:		if (GPU)
src/main/java/uk/ac/imperial/lsds/saber/QueryOperator.java:			gpuCode.processData(first, second, api);
src/main/java/uk/ac/imperial/lsds/saber/QueryOperator.java:		if (gpuCode != null && SystemConf.GPU)
src/main/java/uk/ac/imperial/lsds/saber/QueryOperator.java:			gpuCode.setup();
src/main/java/uk/ac/imperial/lsds/saber/QueryOperator.java:			return gpuCode.toString();

```
