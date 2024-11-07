# https://github.com/MBoemo/DNAscent

```console
docs/source/releaseNotes.rst:* v4.0.1 had an issue where one of the model layers was not properly optimised for GPU usage by TensorFlow 2.12.0. This was causing slow runtimes of ``DNAscent detect`` on certain GPUs. The issue is fixed in this release, although it required a rollback to TensorFlow 2.4.1 (at least for now). Part of our aim in releasing and supporting Singularity images was to mitigate any inconvenience that might have been caused by the need to change CUDA and CuDNN versions.
docs/source/releaseNotes.rst:* TensorFlow updated to 2.12.0 and, correspondingly, GPU usage now requires CUDA 11.8 and cuDNN 8.9.
docs/source/releaseNotes.rst:* Migration from TensorFlow 1.14 to 2.4.1 and, correspondingly, GPU usage now requires CUDA 11 and cuDNN 8,
docs/source/releaseNotes.rst:* Support for BrdU detection on GPUs,
docs/source/index.rst:* **Singularity image or compile from source?** Singularity image. We load it with DNAscent's dependencies so that all you need is a valid NVIDIA driver. 
docs/source/index.rst:* **GPU or CPU?** GPU for DNAscent detect (along with as many CPUs as your GPU node has available) and CPU for DNAscent align and forkSense. DNAscent index is quick and should run in a few seconds on a single CPU.
docs/source/detect.rst:     --GPU                     use the GPU device indicated for prediction (default is CPU),
docs/source/detect.rst:The number of threads is specified using the ``-t`` flag. ``DNAscent detect`` multithreads quite well by analysing a separate read on each thread so multithreading is recommended. By default, the signal alignments and base analogue predictions are run on CPUs.  If a CUDA-compatible GPU device is specified using the ``--GPU`` flag, then the signal alignments will be run on CPUs using the threads specified with ``-t`` and the base analogue prediction will be run on the GPU. Your GPU device number can be found with the command ``nvidia-smi``. GPU use requires that CUDA and cuDNN are set up correctly on your system and that these libraries can be accessed. If they're not, DNAscent will default back to using CPUs.
docs/source/workflows.rst:If the system has a CUDA-compatible GPU in it, we can run ``nvidia-smi`` to get an output that looks like the following:
docs/source/workflows.rst:   | NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
docs/source/workflows.rst:   | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
docs/source/workflows.rst:   | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
docs/source/workflows.rst:   |  GPU   GI   CI        PID   Type   Process name                  GPU Memory | 
docs/source/workflows.rst:From this, we can see that the GPU's device ID is 0 (just to the left of Tesla) so we can run:
docs/source/workflows.rst:   DNAscent detect -b alignment.bam -r /full/path/to/reference.fasta -i index.dnascent -o detect_output.bam -t 10 --GPU 0
docs/source/workflows.rst:Note that we're assuming the CUDA libraries for the GPU have been set up properly (see :ref:`installation`). If these libraries can't be accessed, DNAscent will splash a warning saying so and default back to using CPUs.
docs/source/installation.rst:We recommend running DNAscent using one of our supported Singularity images. These images contain all necessary dependencies including TensorFlow, CUDA, CuDNN, and compression plugins so that your system only needs a valid NVIDIA driver for GPU usage. If your system does not have Singularity installed, instructions are available `here <https://docs.sylabs.io/guides/3.0/user-guide/installation.html>`_.
docs/source/installation.rst:The ``DNAscent detect`` executable can make use of a GPU, although this is optional (see :ref:`detect`).  DNAscent requires CUDA 11.1 and cuDNN 8.0. Information about these can be found at the following links:
docs/source/installation.rst:* cuDNN: https://developer.nvidia.com/cudnn
docs/source/installation.rst:* CUDA: https://developer.nvidia.com/cuda-11.0-download-archive
DNAscent.def:From: nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
DNAscent.def:    # cuda paths
DNAscent.def:    export CUDA_HOME=/usr/local/cuda
DNAscent.def:    export CPATH=/usr/local/cuda/include:$CPATH
DNAscent.def:    export CUDA_PATH=/usr/local/cuda
DNAscent.def:    export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
DNAscent.def:    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
DNAscent.def:    export PATH=/usr/local/cuda/bin:$PATH
Makefile:		wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.4.1.tar.gz; \
Makefile:		tar -xzf libtensorflow-gpu-linux-x86_64-2.4.1.tar.gz || exit 255; \
src/data_IO.cpp:				bool useGPU){
src/data_IO.cpp:	if (useGPU) compMode = "GPU";
src/error_handling.h:		const char* message = "Invalid GPU device ID (expected single int): ";
src/detect.cpp:"  --GPU                     use the GPU device indicated for prediction (default is CPU),\n"
src/detect.cpp:	bool useGPU = false;
src/detect.cpp:	unsigned char GPUdevice = '0';
src/detect.cpp:		else if ( flag == "--GPU" ){
src/detect.cpp:			args.useGPU = true;
src/detect.cpp:			args.GPUdevice = *argv[ i + 1 ];
src/detect.cpp:	if (not args.useGPU){
src/detect.cpp:		modelPair = model_load_gpu_twoInputs(modelPath.c_str(), args.GPUdevice, args.threads);
src/detect.cpp:		std::string outHeader = writeDetectHeader(args.bamFilename, args.referenceFilename, args.indexFilename, args.threads, false, args.minQ, args.minL, args.useGPU);	
src/trainCNN.cpp:"  --GPU                     use the GPU device indicated for prediction (default is CPU),\n"
src/trainCNN.cpp:	bool useGPU = false;
src/trainCNN.cpp:	unsigned char GPUdevice = '0';
src/trainCNN.cpp:		else if ( flag == "--GPU" ){
src/trainCNN.cpp:			args.useGPU = true;
src/trainCNN.cpp:			args.GPUdevice = *argv[ i + 1 ];
src/trainCNN.cpp:	if (not args.useGPU){
src/trainCNN.cpp:		modelPair = model_load_gpu_twoInputs(modelPath.c_str(), args.GPUdevice, args.threads);
src/tensor.h:std::shared_ptr<ModelSession> model_load_gpu(const char *filename, unsigned char device, unsigned int threads, const char *);
src/tensor.h:std::pair< std::shared_ptr<ModelSession>, std::shared_ptr<TF_Graph *> > model_load_gpu_twoInputs(const char *filename, unsigned char device, unsigned int threads);
src/tensor.cpp:	int vis = setenv("CUDA_VISIBLE_DEVICES", "", 1);
src/tensor.cpp:		std::cerr << "Suppression of GPU devices failed." << std::endl;
src/tensor.cpp:	//for CPU-only useage, the tensorflow gpu library will still print out warnings about not finding GPU/CUDA - suppress them here
src/tensor.cpp:std::pair< std::shared_ptr<ModelSession>, std::shared_ptr<TF_Graph *> > model_load_gpu_twoInputs(const char *saved_model_dir, unsigned char device, unsigned int threads){
src/tensor.cpp:std::shared_ptr<ModelSession> model_load_gpu(const char *saved_model_dir, unsigned char device, unsigned int threads,const char *input_layer_name){

```
