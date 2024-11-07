# https://github.com/NBISweden/MrBayes

```console
aclocal.m4:#   "clang" (LLVM), "cray", "fujitsu", "sdcc", "sx", "nvhpc" (NVIDIA HPC
INSTALL-Docker-Ubuntu.md:    - CUDA: N/A
INSTALL-Docker-Ubuntu.md:    - OpenCL: OK
INSTALL-Docker-Ubuntu.md:  (since we asked for not using CUDA or Java):
INSTALL-Docker-Ubuntu.md:    - `WARNING: NVIDIA CUDA nvcc compiler not found`
INSTALL-Docker-Ubuntu.md:    # OpenCL
INSTALL-Docker-Ubuntu.md:        ocl-icd-opencl-dev \
INSTALL-Docker-Ubuntu.md:        pocl-opencl-icd
INSTALL-Docker-CentOS.md:    - CUDA: N/A
INSTALL-Docker-CentOS.md:    - OpenCL: FAIL
INSTALL-Docker-CentOS.md:  (since we asked for not using OpenCL, CUDA or Java):
INSTALL-Docker-CentOS.md:    - `WARNING: OpenCL not found or disabled`
INSTALL-Docker-CentOS.md:    - `WARNING: NVIDIA CUDA nvcc compiler not found`
INSTALL-Docker-CentOS.md:    # OpenCL
INSTALL-Docker-CentOS.md:    # Tue 30 Jun 2020: Beagle compiles with OpenCL, but when run in MrBayes, we get
INSTALL-Docker-CentOS.md:    # OpenCL error: Unknown error from file <GPUInterfaceOpenCL.cpp>, line 115.
INSTALL-Docker-CentOS.md:    #ln -s /usr/lib64/libOpenCL.so.1 /usr/lib/libOpenCL.so # https://unix.stackexchange.com/questions/292630/how-to-install-opencl-in-centos-7-using-yum
INSTALL-Docker-CentOS.md:    LDFLAGS=-Wl,-rpath=/usr/local/lib ./configure --without-jdk --without-cuda --without-opencl --disable-doxygen-doc
doc/manual/src/Manual_MrBayes_v3.2.tex:slower than amino-acid models. If you are able to use BEAGLE to compute likelihoods on the GPU, you
doc/manual/src/Manual_MrBayes_v3.2.tex:efficient likelihood calculations on both CPUs and GPUs. The CPU (central processing unit) is the
doc/manual/src/Manual_MrBayes_v3.2.tex:core component of all computers, while a GPU (graphics processing unit) belongs on the video card
doc/manual/src/Manual_MrBayes_v3.2.tex:that comes with most modern computers. The GPU specializes in performing small tasks on large sets
doc/manual/src/Manual_MrBayes_v3.2.tex:but recently it has become much easier to use GPUs for ordinary calculations. BEAGLE takes
doc/manual/src/Manual_MrBayes_v3.2.tex:Regardless of your machine, you should be able to use BEAGLE on your CPU. However, it is the GPU
doc/manual/src/Manual_MrBayes_v3.2.tex:code that will give you the dramatic speedups, and it currently requires that you have an NVIDIA
doc/manual/src/Manual_MrBayes_v3.2.tex:video card and the appropriate CUDA drivers.
doc/manual/src/Manual_MrBayes_v3.2.tex:be a good idea to examine the available BEAGLE CPU and GPU resources. You can find out by using the
doc/manual/src/Manual_MrBayes_v3.2.tex:If BEAGLE finds an available GPU resource, it will be listed here. This is the listing from a
doc/manual/src/Manual_MrBayes_v3.2.tex:machine with an available GPU detected by BEAGLE:
doc/manual/src/Manual_MrBayes_v3.2.tex: Flags: PROCESSOR_GPU PRECISION_SINGLE COMPUTATION_SYNCH EIGEN_REAL
doc/manual/src/Manual_MrBayes_v3.2.tex:It is the second entry that corresponds to the GPU. If you do not have one or more GPUs listed
doc/manual/src/Manual_MrBayes_v3.2.tex:Beagledevice       CPU/GPU               CPU            
doc/manual/src/Manual_MrBayes_v3.2.tex:To use the GPU, you simply switch from CPU to GPU (if you have an available GPU). With the GPU, the
doc/manual/src/Manual_MrBayes_v3.2.tex:applicable in the GPU case. The fastest GPU option should therefore be:
doc/manual/src/Manual_MrBayes_v3.2.tex:MrBayes > set usebeagle=yes beagledevice=gpu
doc/manual/src/Manual_MrBayes_v3.2.tex:The GPU code can be a lot faster than the CPU code, particularly for amino acid and codon models.
doc/manual/src/Manual_MrBayes_v3.2.tex:sequences are, the better GPU performance you can expect. If the sequences are short, the overhead
doc/manual/src/Manual_MrBayes_v3.2.tex:involved in shuffling data to and from the GPU may well overshadow any performance gain you get in
doc/manual/src/Manual_MrBayes_v3.2.tex:more cores than GPUs. This means that BEAGLE may not help much in an MPI setting unless you are
INSTALL:as setting up the required prerequisites for things like OpenCL etc. is
src/mbbeagle.h:void   BeagleAddGPUDevicesToList (int **beagleResource, int *beagleResourceCount);
src/mbbeagle.h:void   BeagleRemoveGPUDevicesFromList (int **beagleResource, int *beagleResourceCount);
src/mbbeagle.c:    /* use level-order traversal with CUDA implementation or OpenCL with multi-partition */
src/mbbeagle.c:    if(((details.flags & BEAGLE_FLAG_FRAMEWORK_CUDA) && division < 1 ) ||
src/mbbeagle.c:        ((details.flags & BEAGLE_FLAG_FRAMEWORK_OPENCL) && division < 0))
src/mbbeagle.c:void BeagleAddGPUDevicesToList (int **newResourceList, int *beagleResourceCount)
src/mbbeagle.c:    int i, gpuCount;
src/mbbeagle.c:    gpuCount = 0;
src/mbbeagle.c:        if (beagleResources->list[i].supportFlags & BEAGLE_FLAG_PROCESSOR_GPU) {
src/mbbeagle.c:            (*newResourceList)[gpuCount] = i;
src/mbbeagle.c:            gpuCount++;
src/mbbeagle.c:    *beagleResourceCount = gpuCount;            
src/mbbeagle.c:void BeagleRemoveGPUDevicesFromList (int **beagleResource, int *beagleResourceCount)
src/mbbeagle.c:    if (inFlags & BEAGLE_FLAG_PROCESSOR_GPU) {
src/mbbeagle.c:            MrBayesPrint ("%s   Simultaneous use of GPU and SSE not available.\n", spacer);
src/mbbeagle.c:            MrBayesPrint ("%s   Simultaneous use of GPU and OpenMP not available.\n", spacer);
src/mbbeagle.c:                      "PROCESSOR_GPU",
src/mbbeagle.c:                     BEAGLE_FLAG_PROCESSOR_GPU,
src/command.c:                    if (!strcmp(tempStr, "Gpu"))
src/command.c:                        beagleFlags |= BEAGLE_FLAG_PROCESSOR_GPU;
src/command.c:                        BeagleAddGPUDevicesToList(&beagleResource, &beagleResourceCount);                       
src/command.c:                        beagleFlags &= ~BEAGLE_FLAG_PROCESSOR_GPU;
src/command.c:                        BeagleRemoveGPUDevicesFromList(&beagleResource, &beagleResourceCount);
src/command.c:                        if (beagleFlags & BEAGLE_FLAG_PROCESSOR_GPU)
src/command.c:                            MrBayesPrint ("%s   Setting beagledevice to GPU\n", spacer);
src/command.c:        MrBayesPrint ("                   performance hardware including multicore CPUs and GPUs. Some  \n"); 
src/command.c:        MrBayesPrint ("   Beagledevice -- Set this option to 'GPU' or 'CPU' to select processor.        \n"); 
src/command.c:        MrBayesPrint ("                   Can result in improved performance on GPU devices at the cost \n");
src/command.c:        MrBayesPrint ("   Beagledevice       CPU/GPU               %s                                   \n", beagleFlags & BEAGLE_FLAG_PROCESSOR_GPU ? "GPU" : "CPU");
src/command.c:    PARAM (234, "Beagledevice",   DoSetParm,         "Cpu|Gpu|\0");

```
