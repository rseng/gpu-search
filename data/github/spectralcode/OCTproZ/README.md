# https://github.com/spectralcode/OCTproZ

```console
vision.md:- Implement OCT processing pipeline in other frameworks besides CUDA (e.g. OpenCL, OpenMP, C ++ AMP) to support various computer hardware for processing
README.md:* **Real-time OCT processing and visualization with single GPU**  </br>
README.md:The full [OCT processing pipeline](https://spectralcode.github.io/OCTproZ/#processing-section) is implemented in [CUDA](https://developer.nvidia.com/cuda-zone) and visualization is performed with [OpenGL](https://www.opengl.org). Depending on the GPU used, OCTproZ can be used for MHz-OCT. 
README.md:GPU           | A-scan rate without live 3D view | A-scan rate with live 3D view
README.md:NVIDIA Quadro K620  | ~ 300 kHz ( ~2.2 volumes/s) | ~ 250 kHz ( ~1.9 volumes/s)
README.md:NVIDIA GeForce GTX 1080 Ti  | ~ 4.8 MHz (~ 36 volumes/s) | ~ 4.0 MHz (~ 30 volumes/s)
README.md:To run OCTproZ a cuda-compatible graphics card with current drivers is required.
README.md:|[OCTSharp](https://github.com/OCTSharpImaging/OCTSharp) | Cuda, C# |
README.md:|[vortex](https://www.vortex-oct.dev/) | Cuda, C++ with Python bindings |
paper/paper.bib:  title={Scalable, high performance Fourier domain optical coherence tomography: Why FPGAs and not GPGPUs},
paper/paper.bib:  title={GPU-accelerated single-pass raycaster},
paper/paper.bib:@misc{cuda,
paper/paper.bib:	title={CUDA},
paper/paper.bib:  author={​NVIDIA},
paper/paper.bib:	url={https://developer.nvidia.com/cuda-zone},
paper/paper.md:  - CUDA
paper/paper.md:Optical coherence tomography (OCT) is a non-invasive imaging technique used primarily in the medical field, especially in ophthalmology. The core element of any OCT system is an optical interferometer that generates a spectral fringe pattern by combining a reference beam and the backscattered light from a sample. To obtain an interpretable image from this acquired raw OCT signal several processing steps are necessary, whereby the inverse Fourier transform represents an essential step. As the possible acquisition speed for raw OCT data has increased constantly, more sophisticated methods were needed for processing and live visualization of the acquired OCT data. A particularly impressive setup was presented by Choi et al. [@choi2012spectral] that utilizes twenty FPGA-modules for real-time OCT signal processing and a graphics processing unit (GPU) for volume rendering. Nowadays, processing is typically done on graphics cards [@zhang2010real; @rasakanthan2011processing; @sylwestrzak2012four; @jian2013graphics; @wieser2014high], not FPGAs, because implementing algorithms on GPUs is more flexible and takes less time [@li2011scalable]. Most of the publications that describe OCT GPU processing do not provide the actual software implementation.
paper/paper.md:A commendable exemption is the GPU accelerated OCT processing pipeline published by Jian et al. The associated source code, which demonstrates an implementation of OCT data processing and visualization and does not include any advanced features such as a graphical user interface (GUI), already consists of several thousand lines. Thus, the most time consuming task of Fourier Domain OCT (FD-OCT) system development is not the optical setup, but the software development. The software can be separated into hardware control and signal processing, whereby the former being a highly individual, hardware-dependent software module and the latter being a generic software module, which is almost identical for many systems. To drastically reduce OCT system development time, we present OCTproZ, an open source OCT processing software package that can easily be extended, via a plug-in system, for many different hardware setups. In this paper we give a brief overview of the key functionality and structure of the software.
paper/paper.md:OCTproZ performs live signal processing and visualization of OCT data. It is written in C++, uses the cross-platform application framework Qt [@qt] for the GUI and utilizes Nvidia’s computer unified device architecture (CUDA) [@cuda] for GPU parallel computing. A screenshot of the application can be seen in Figure \ref{fig:screenshot}.
paper/paper.md:Raw data, i.e. acquired spectral fringe pattern, from the OCT system is transferred to RAM until enough data for a user-defined amount of cross-sectional images, so-called B-scans, is acquired. Via direct memory access (DMA) this raw data batch is then copied asynchronously to GPU memory where OCT signal processing is executed. If the processed data needs to be stored or post processing steps are desired the processed OCT data can be transferred back to RAM with the use of DMA. An overview of the processing steps is depicted in Figure \ref{fig:processing}.
paper/paper.md: ![Processing pipeline of OCTproZ v1.2.0. Each box inside "OCTproZ GPU Processing" represents a CUDA kernel. Some processing steps are combinend into a single kernel (e.g. k-linearization, dispersion compensation and windowing) to enhance processing performance. \label{fig:processing}](figures/processing_pipeline_linear_v2.png) 
paper/paper.md:In order to avoid unnecessary data transfer to host memory, CUDA-OpenGL interoperability is used which allows the processed data to remain in GPU memory for visualization. 
paper/paper.md:Processing rate highly depends on the size of the raw data, the used computer hardware and resource usage by background or system processes. With common modern computer systems and typical data dimensions for OCT, OCTproZ achieves A-scan rates in the MHz range. Exemplary, Table 1 shows two computer systems and their respective processing rates for the full processing pipeline. However, since the 3D live view is computationally intensive the processing rate changes noticeably depending on whether the volume viewer is activated or not. The used raw data set consists of samples with a bit depth of 12, 1024 samples per raw A-scan, 512 A-scans per B-scan and 256 B-scans per volume. As the volume is processed in batches, the batch size was set for each system to a reasonable number of B-scans per buffer to avoid GPU memory overflow. It should be noted that this performance evaluation was done with OCTproZ v1.0.0 but is also valid for v1.2.0 if the newly introduced processing step for sinusoidal scan distortion correction is disabled.
paper/paper.md:GPU|NVIDIA Quadro K620|NVIDIA GeForce GTX 1080 Ti
performance.md:|GPU|NVIDIA Quadro K620|NVIDIA GeForce GTX 1080 Ti| NVIDIA GeForce GTX 1080
performance.md:| Embedded System |**NVIDIA Jetson Nano**|
performance.md:|GPU|NVIDIA Tegra X1 (128-core Maxwell)|
performance.md: It is also possible to use the [NVIDIA Visual Profiler](https://developer.nvidia.com/nvidia-visual-profiler) to analyze performance in more detail.
performance.md: For example, the following screenshot from the NVIDIA Visual Profiler shows the performance analysis of the measurement (without 3D live view) from the table at the beginning of this document with the lab computer:
performance.md:- Processing happens in batches. One batch is equal to one buffer and the size of the buffer has impact on processing performance. If it is too small the processing may be slower than possible. If it is too large the application may crash as a larger buffer size results in higher GPU memory usage, which can exceed the available memory on the used GPU 
performance.md:- The optimal buffer size for a specific GPU needs to be determined experimentally 
BUILD.md:- Installation of [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (version 8 or newer)
BUILD.md:- __Windows:__ MSVC compiler that is compatible with your CUDA version (see [CUDA installation guide for Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements)) To get the MSVC compiler it is the easiest to search online where/how to get it as this changes from time to time. Pay attention that you get the right version of the MSVC compiler as described in the CUDA guide. <br>
BUILD.md:__Linux:__ Development environment that is compatible with your CUDA version (see [CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)) and the third-party libraries mentioned in the [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#install-libraries)
BUILD.md:4. Change the CUDA architecture flags in [cuda.pri](octproz_project/octproz/pri/cuda.pri) if necessary for your hardware ([more info](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/))
BUILD.md:OCTproZ can be compiled with different versions of Qt, CUDA and MSVC. One of the easiest setups is possible with:
BUILD.md:- [CUDA 11.5.2](https://developer.nvidia.com/cuda-toolkit-archive)
BUILD.md:### 2 Install CUDA:
BUILD.md:1. Download [CUDA 11.5.2](https://developer.nvidia.com/cuda-toolkit-archive)
BUILD.md:2. Start CUDA installer and follow instructions on screen
BUILD.md:### 2. Install CUDA
BUILD.md:Follow the [CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux) carefully.
BUILD.md:Now insert the cuda relevant paths as stated in the cuda installation guide at the end of the file.
BUILD.md:After this you should verify [that the CUDA toolkit can find and communicate correctly with the CUDA-capable hardware.](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#verify-installation)
BUILD.md:Finally you need to [install some third-party libraries](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#install-libraries):
BUILD.md:# OCTproZ on the NVIDIA Jetson Nano with JetPack
BUILD.md:- [stackoverflow.com Qt-default version issue on migration from RPi4 to NVIDIA Jetson Nano](https://stackoverflow.com/questions/62190967/qt-default-version-issue-on-migration-from-rpi4-to-nvidia-jetson-nano)
octproz_project/config.pri:CUDA_RELEVANT_DEFINES += ENABLE_CUDA_ZERO_COPY  # This enables zero copy memory on Jetson Nano (comment out to disable)
octproz_project/config.pri:DEFINES += $$CUDA_RELEVANT_DEFINES
octproz_project/thirdparty/cuda/readme.txt:All files in this folder are from the NVIDIA CUDA samples.
octproz_project/thirdparty/cuda/readme.txt:The CUDA samples can be found here: https://github.com/nvidia/cuda-samples 
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h: *  * Neither the name of NVIDIA CORPORATION nor the names of its
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:// These are CUDA Helper functions for initialization and error checking
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:#ifndef COMMON_HELPER_CUDA_H_
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:#define COMMON_HELPER_CUDA_H_
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:// files, please refer the CUDA examples for examples of the needed CUDA
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:// headers, which may change depending on which CUDA functions are used.
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:// CUDA Runtime error messages
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:static const char *_cudaGetErrorEnum(cudaError_t error) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  return cudaGetErrorName(error);
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:#ifdef CUDA_DRIVER_API
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:// CUDA Driver API errors
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:static const char *_cudaGetErrorEnum(CUresult error) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:static const char *_cudaGetErrorEnum(cublasStatus_t error) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:static const char *_cudaGetErrorEnum(cufftResult error) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:static const char *_cudaGetErrorEnum(cusparseStatus_t error) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:static const char *_cudaGetErrorEnum(cusolverStatus_t error) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:static const char *_cudaGetErrorEnum(curandStatus_t error) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:static const char *_cudaGetErrorEnum(nvjpegStatus_t error) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:static const char *_cudaGetErrorEnum(NppStatus error) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    // These are for CUDA 5.5 or higher
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    case NPP_CUDA_KERNEL_EXECUTION_ERROR:
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:      return "NPP_CUDA_KERNEL_EXECUTION_ERROR";
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:// This will output the proper CUDA error strings in the event
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:// that a CUDA host call returns an error
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:// This will output the proper error string when calling cudaGetLastError
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:inline void __getLastCudaError(const char *errorMessage, const char *file,
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  cudaError_t err = cudaGetLastError();
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  if (cudaSuccess != err) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:            "%s(%i) : getLastCudaError() CUDA error :"
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:            cudaGetErrorString(err));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:// This will only print the proper error string when calling cudaGetLastError
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:inline void __printLastCudaError(const char *errorMessage, const char *file,
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  cudaError_t err = cudaGetLastError();
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  if (cudaSuccess != err) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:            "%s(%i) : getLastCudaError() CUDA error :"
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:            cudaGetErrorString(err));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:// Beginning of GPU Architecture definitions
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  // Defines for GPU Architecture types (using the SM version to determine
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  sSMtoCores nGpuArchCoresPerSM[] = {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  while (nGpuArchCoresPerSM[index].SM != -1) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:      return nGpuArchCoresPerSM[index].Cores;
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  return nGpuArchCoresPerSM[index - 1].Cores;
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  // Defines for GPU Architecture types (using the SM version to determine
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  // the GPU Arch name)
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  sSMtoArchName nGpuArchNameSM[] = {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  while (nGpuArchNameSM[index].SM != -1) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:      return nGpuArchNameSM[index].name;
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:      major, minor, nGpuArchNameSM[index - 1].name);
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  return nGpuArchNameSM[index - 1].name;
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  // end of GPU Architecture definitions
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:#ifdef __CUDA_RUNTIME_H__
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:// General GPU Device CUDA Initialization
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:inline int gpuDeviceInit(int devID) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  checkCudaErrors(cudaGetDeviceCount(&device_count));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:            "gpuDeviceInit() CUDA error: "
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:            "no devices supporting CUDA.\n");
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n",
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:            ">> gpuDeviceInit (-device=%d) is not a valid"
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:            " GPU device. <<\n",
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, devID));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  if (computeMode == cudaComputeModeProhibited) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:            "Prohibited>, no threads can use cudaSetDevice().\n");
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  checkCudaErrors(cudaSetDevice(devID));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, _ConvertSMVer2ArchName(major, minor));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:// This function returns the best GPU (with maximum GFLOPS)
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:inline int gpuGetMaxGflopsDeviceId() {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  checkCudaErrors(cudaGetDeviceCount(&device_count));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:            "gpuGetMaxGflopsDeviceId() CUDA error:"
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:            " no devices supporting CUDA.\n");
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  // Find the best CUDA capable GPU device
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    // If this GPU is not running on Compute Mode prohibited,
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    if (computeMode != cudaComputeModeProhibited) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:      checkCudaErrors(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:      cudaError_t result = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, current_device);
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:      if (result != cudaSuccess) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:        // If cudaDevAttrClockRate attribute is not supported we
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:        // set clockRate as 1, to consider GPU with most SMs and CUDA Cores.
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:        if(result == cudaErrorInvalidValue) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:          fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", __FILE__, __LINE__,
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:            static_cast<unsigned int>(result), _cudaGetErrorEnum(result));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:            "gpuGetMaxGflopsDeviceId() CUDA error:"
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:// Initialization code to find the best CUDA Device
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:inline int findCudaDevice(int argc, const char **argv) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:      devID = gpuDeviceInit(devID);
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    devID = gpuGetMaxGflopsDeviceId();
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    checkCudaErrors(cudaSetDevice(devID));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:inline int findIntegratedGPU() {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  checkCudaErrors(cudaGetDeviceCount(&device_count));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  // Find the integrated GPU which is compute capable
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    checkCudaErrors(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, current_device));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    // If GPU is integrated and is not running on Compute Mode prohibited,
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    // then cuda can map to GLES resource
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:    if (integrated && (computeMode != cudaComputeModeProhibited)) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:      checkCudaErrors(cudaSetDevice(current_device));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:      checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:      checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:      printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:            "CUDA error:"
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:            " No GLES-CUDA Interop capable GPU found.\n");
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:// General check for CUDA GPU SM Capabilities
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:inline bool checkCudaCapabilities(int major_version, int minor_version) {
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  checkCudaErrors(cudaGetDevice(&dev));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:        "  No GPU device was found that can support "
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:        "CUDA compute capability %d.%d.\n",
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:  // end of CUDA Helper Functions
octproz_project/thirdparty/cuda/common/inc/helper_cuda.h:#endif  // COMMON_HELPER_CUDA_H_
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:/* --------------------------- GL_ARB_gpu_shader5 -------------------------- */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#ifndef GL_ARB_gpu_shader5
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GL_ARB_gpu_shader5 1
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GLEW_ARB_gpu_shader5 GLEW_GET_VAR(__GLEW_ARB_gpu_shader5)
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#endif /* GL_ARB_gpu_shader5 */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:/* ------------------------- GL_ARB_gpu_shader_fp64 ------------------------ */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#ifndef GL_ARB_gpu_shader_fp64
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GL_ARB_gpu_shader_fp64 1
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GLEW_ARB_gpu_shader_fp64 GLEW_GET_VAR(__GLEW_ARB_gpu_shader_fp64)
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#endif /* GL_ARB_gpu_shader_fp64 */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:/* --------------------- GL_EXT_gpu_program_parameters --------------------- */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#ifndef GL_EXT_gpu_program_parameters
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GL_EXT_gpu_program_parameters 1
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GLEW_EXT_gpu_program_parameters GLEW_GET_VAR(__GLEW_EXT_gpu_program_parameters)
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#endif /* GL_EXT_gpu_program_parameters */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:/* --------------------------- GL_EXT_gpu_shader4 -------------------------- */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#ifndef GL_EXT_gpu_shader4
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GL_EXT_gpu_shader4 1
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GLEW_EXT_gpu_shader4 GLEW_GET_VAR(__GLEW_EXT_gpu_shader4)
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#endif /* GL_EXT_gpu_shader4 */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:/* --------------------------- GL_NV_gpu_program4 -------------------------- */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#ifndef GL_NV_gpu_program4
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GL_NV_gpu_program4 1
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GLEW_NV_gpu_program4 GLEW_GET_VAR(__GLEW_NV_gpu_program4)
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#endif /* GL_NV_gpu_program4 */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:/* -------------------------- GL_NV_gpu_program4_1 ------------------------- */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#ifndef GL_NV_gpu_program4_1
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GL_NV_gpu_program4_1 1
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GLEW_NV_gpu_program4_1 GLEW_GET_VAR(__GLEW_NV_gpu_program4_1)
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#endif /* GL_NV_gpu_program4_1 */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:/* --------------------------- GL_NV_gpu_program5 -------------------------- */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#ifndef GL_NV_gpu_program5
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GL_NV_gpu_program5 1
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GLEW_NV_gpu_program5 GLEW_GET_VAR(__GLEW_NV_gpu_program5)
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#endif /* GL_NV_gpu_program5 */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:/* ------------------------- GL_NV_gpu_program_fp64 ------------------------ */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#ifndef GL_NV_gpu_program_fp64
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GL_NV_gpu_program_fp64 1
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GLEW_NV_gpu_program_fp64 GLEW_GET_VAR(__GLEW_NV_gpu_program_fp64)
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#endif /* GL_NV_gpu_program_fp64 */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:/* --------------------------- GL_NV_gpu_shader5 --------------------------- */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#ifndef GL_NV_gpu_shader5
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GL_NV_gpu_shader5 1
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GLEW_NV_gpu_shader5 GLEW_GET_VAR(__GLEW_NV_gpu_shader5)
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#endif /* GL_NV_gpu_shader5 */
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GL_BUFFER_GPU_ADDRESS_NV 0x8F1D
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:#define GL_GPU_ADDRESS_NV 0x8F34
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_ARB_gpu_shader5;
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_ARB_gpu_shader_fp64;
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_EXT_gpu_program_parameters;
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_EXT_gpu_shader4;
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_NV_gpu_program4;
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_NV_gpu_program4_1;
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_NV_gpu_program5;
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_NV_gpu_program_fp64;
octproz_project/thirdparty/cuda/common/inc/GL/glew.h:        GLEW_VAR_EXPORT GLboolean __GLEW_NV_gpu_shader5;
octproz_project/thirdparty/cuda/common/inc/GL/wglext.h:#ifndef WGL_NV_gpu_affinity
octproz_project/thirdparty/cuda/common/inc/GL/wglext.h:#ifndef WGL_NV_gpu_affinity
octproz_project/thirdparty/cuda/common/inc/GL/wglext.h:DECLARE_HANDLE(HGPUNV);
octproz_project/thirdparty/cuda/common/inc/GL/wglext.h:typedef struct _GPU_DEVICE
octproz_project/thirdparty/cuda/common/inc/GL/wglext.h:} GPU_DEVICE, *PGPU_DEVICE;
octproz_project/thirdparty/cuda/common/inc/GL/wglext.h:#ifndef WGL_NV_gpu_affinity
octproz_project/thirdparty/cuda/common/inc/GL/wglext.h:#define WGL_NV_gpu_affinity 1
octproz_project/thirdparty/cuda/common/inc/GL/wglext.h:extern BOOL WINAPI wglEnumGpusNV(UINT iIndex, HGPUNV *hGpu);
octproz_project/thirdparty/cuda/common/inc/GL/wglext.h:extern BOOL WINAPI wglEnumGpuDevicesNV(HGPUNV hGpu, UINT iIndex, PGPU_DEVICE pGpuDevice);
octproz_project/thirdparty/cuda/common/inc/GL/wglext.h:extern HDC WINAPI wglCreateAffinityDCNV(const HGPUNV *pGpuList);
octproz_project/thirdparty/cuda/common/inc/GL/wglext.h:extern BOOL WINAPI wglEnumGpusFromAffinityDCNV(HDC hAffinityDC, UINT iIndex, HGPUNV *hGpu);
octproz_project/thirdparty/cuda/common/inc/GL/wglext.h:typedef BOOL (WINAPI *PFNWGLENUMGPUSNVPROC)(UINT iIndex, HGPUNV *hGpu);
octproz_project/thirdparty/cuda/common/inc/GL/wglext.h:typedef BOOL (WINAPI *PFNWGLENUMGPUDEVICESNVPROC)(HGPUNV hGpu, UINT iIndex, PGPU_DEVICE pGpuDevice);
octproz_project/thirdparty/cuda/common/inc/GL/wglext.h:typedef HDC(WINAPI *PFNWGLCREATEAFFINITYDCNVPROC)(const HGPUNV *pGpuList);
octproz_project/thirdparty/cuda/common/inc/GL/wglext.h:typedef BOOL (WINAPI *PFNWGLENUMGPUSFROMAFFINITYDCNVPROC)(HDC hAffinityDC, UINT iIndex, HGPUNV *hGpu);
octproz_project/thirdparty/cuda/common/inc/GL/glext.h: Copyright NVIDIA Corporation 2006
octproz_project/thirdparty/cuda/common/inc/GL/glext.h: *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
octproz_project/thirdparty/cuda/common/inc/GL/glext.h: NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
octproz_project/thirdparty/cuda/common/inc/GL/glext.h: THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
octproz_project/thirdparty/cuda/common/inc/GL/glext.h:#ifndef GL_EXT_gpu_shader4
octproz_project/thirdparty/cuda/common/inc/GL/glext.h:#ifndef GL_NV_gpu_program4
octproz_project/thirdparty/cuda/common/inc/GL/glext.h:#ifndef GL_EXT_gpu_shader4
octproz_project/thirdparty/cuda/common/inc/GL/glext.h:#define GL_EXT_gpu_shader4 1
octproz_project/thirdparty/cuda/common/inc/GL/glext.h:#ifndef GL_NV_gpu_program4
octproz_project/thirdparty/cuda/common/inc/GL/glext.h:#define GL_NV_gpu_program4 1
octproz_project/thirdparty/cuda/common/inc/helper_functions.h:/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
octproz_project/thirdparty/cuda/common/inc/helper_functions.h: *  * Neither the name of NVIDIA CORPORATION nor the names of its
octproz_project/thirdparty/cuda/common/inc/helper_timer.h:/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
octproz_project/thirdparty/cuda/common/inc/helper_timer.h: *  * Neither the name of NVIDIA CORPORATION nor the names of its
octproz_project/thirdparty/cuda/common/inc/exception.h:/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
octproz_project/thirdparty/cuda/common/inc/exception.h: *  * Neither the name of NVIDIA CORPORATION nor the names of its
octproz_project/thirdparty/cuda/common/inc/exception.h:/* CUda UTility Library */
octproz_project/thirdparty/cuda/common/inc/helper_string.h:/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
octproz_project/thirdparty/cuda/common/inc/helper_string.h: *  * Neither the name of NVIDIA CORPORATION nor the names of its
octproz_project/thirdparty/cuda/common/inc/helper_string.h:// CUDA Utility Helper Functions
octproz_project/thirdparty/cuda/common/inc/helper_string.h:// This function wraps the CUDA Driver API into a template function
octproz_project/thirdparty/cuda/common/inc/helper_string.h:      "../../../../Samples/3_CUDA_Features/<executable_name>/",  // up 4 in tree
octproz_project/thirdparty/cuda/common/inc/helper_string.h:      "../../../Samples/3_CUDA_Features/<executable_name>/",     // up 3 in tree
octproz_project/thirdparty/cuda/common/inc/helper_string.h:      "../../Samples/3_CUDA_Features/<executable_name>/",        // up 2 in tree
octproz_project/thirdparty/cuda/common/inc/helper_string.h:      "../../../../Samples/4_CUDA_Libraries/<executable_name>/",  // up 4 in tree
octproz_project/thirdparty/cuda/common/inc/helper_string.h:      "../../../Samples/4_CUDA_Libraries/<executable_name>/",     // up 3 in tree
octproz_project/thirdparty/cuda/common/inc/helper_string.h:      "../../Samples/4_CUDA_Libraries/<executable_name>/",        // up 2 in tree
octproz_project/thirdparty/cuda/common/inc/helper_string.h:      "../../../../Samples/3_CUDA_Features/<executable_name>/data/",  // up 4 in tree
octproz_project/thirdparty/cuda/common/inc/helper_string.h:      "../../../Samples/3_CUDA_Features/<executable_name>/data/",     // up 3 in tree
octproz_project/thirdparty/cuda/common/inc/helper_string.h:      "../../Samples/3_CUDA_Features/<executable_name>/data/",        // up 2 in tree
octproz_project/thirdparty/cuda/common/inc/helper_string.h:      "../../../../Samples/4_CUDA_Libraries/<executable_name>/data/",  // up 4 in tree
octproz_project/thirdparty/cuda/common/inc/helper_string.h:      "../../../Samples/4_CUDA_Libraries/<executable_name>/data/",     // up 3 in tree
octproz_project/thirdparty/cuda/common/inc/helper_string.h:      "../../Samples/4_CUDA_Libraries/<executable_name>/data/",        // up 2 in tree
octproz_project/thirdparty/cuda/common/inc/helper_image.h:/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
octproz_project/thirdparty/cuda/common/inc/helper_image.h: *  * Neither the name of NVIDIA CORPORATION nor the names of its
octproz_project/octproz_devkit/octproz_devkit.pro:#include cuda.pri only for Jetson Nano
octproz_project/octproz_devkit/octproz_devkit.pro:	include(../octproz/pri/cuda.pri)
octproz_project/octproz_devkit/src/acquisitionbuffer.cpp:	#include <cuda_runtime.h>
octproz_project/octproz_devkit/src/acquisitionbuffer.cpp:	#include <helper_cuda.h>
octproz_project/octproz_devkit/src/acquisitionbuffer.cpp:			#ifdef ENABLE_CUDA_ZERO_COPY
octproz_project/octproz_devkit/src/acquisitionbuffer.cpp:				cudaError_t err = cudaHostAlloc((void**)&(this->bufferArray[bufferIndex]), this->bytesPerBuffer, cudaHostAllocMapped);
octproz_project/octproz_devkit/src/acquisitionbuffer.cpp:				cudaError_t err = cudaHostAlloc((void**)&(this->bufferArray[bufferIndex]), this->bytesPerBuffer, cudaHostAllocPortable);
octproz_project/octproz_devkit/src/acquisitionbuffer.cpp:			if (err != cudaSuccess || this->bufferArray[bufferIndex] == nullptr){
octproz_project/octproz_devkit/src/acquisitionbuffer.cpp:				emit error(tr("Buffer memory allocation error. cudaHostAlloc() error code: ") + QString::number(err));
octproz_project/octproz_devkit/src/acquisitionbuffer.cpp:				cudaMemset(this->bufferArray[bufferIndex], 0, this->bytesPerBuffer);
octproz_project/octproz_devkit/src/acquisitionbuffer.cpp:				cudaFreeHost(this->bufferArray[i]);
octproz_project/octproz/aboutdata/thirdparty.txt:<li><b>Cuda</b>: <a href="https://developer.nvidia.com/cuda-zone">https://developer.nvidia.com/cuda-zone</a></li>
octproz_project/octproz/octproz.pro:	$$SOURCEDIR/gpu2hostnotifier.cpp \
octproz_project/octproz/octproz.pro:		SOURCES += $$SOURCEDIR/cuda_code.cu
octproz_project/octproz/octproz.pro:		SOURCES -= $$SOURCEDIR/cuda_code.cu
octproz_project/octproz/octproz.pro:	$$SOURCEDIR/gpu2hostnotifier.h \
octproz_project/octproz/octproz.pro:	$$SOURCEDIR/cuda_code.cu \
octproz_project/octproz/octproz.pro:#include cuda configuration
octproz_project/octproz/octproz.pro:include(pri/cuda.pri)
octproz_project/octproz/pri/cuda.pri:#CUDA system and compiler settings
octproz_project/octproz/pri/cuda.pri:#path of cuda source files
octproz_project/octproz/pri/cuda.pri:CUDA_SOURCES += $$SOURCEDIR/cuda_code.cu \
octproz_project/octproz/pri/cuda.pri:#cuda architecture flags
octproz_project/octproz/pri/cuda.pri:#change these flags according to your GPU
octproz_project/octproz/pri/cuda.pri:#see https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/ for more information
octproz_project/octproz/pri/cuda.pri:#use this for maximum compatibility with CUDA 9
octproz_project/octproz/pri/cuda.pri:#CUDA_ARCH += sm_30 \
octproz_project/octproz/pri/cuda.pri:#use this for maximum compatibility with CUDA 11.0
octproz_project/octproz/pri/cuda.pri:CUDA_ARCH += sm_52 \
octproz_project/octproz/pri/cuda.pri:#use this for Jetson Nano with JetPack 4.6.1 (Cuda 10.2, Ubuntu 18.04)
octproz_project/octproz/pri/cuda.pri:#CUDA_ARCH += sm_53 \
octproz_project/octproz/pri/cuda.pri:CUDA_DEFINES_FLAGS = $$join(CUDA_RELEVANT_DEFINES, '-D', '-D', '')
octproz_project/octproz/pri/cuda.pri:#cuda include paths
octproz_project/octproz/pri/cuda.pri:	CUDA_DIR = /usr/local/cuda
octproz_project/octproz/pri/cuda.pri:	QMAKE_LIBDIR += $$CUDA_DIR/lib64
octproz_project/octproz/pri/cuda.pri:	exists($$shell_path($$CUDA_DIR/samples)){
octproz_project/octproz/pri/cuda.pri:		NVCUDASAMPLES_ROOT = $$shell_path($$CUDA_DIR/samples)
octproz_project/octproz/pri/cuda.pri:		NVCUDASAMPLES_ROOT = $$shell_path($$PWD/../../thirdparty/cuda)
octproz_project/octproz/pri/cuda.pri:	CUDA_DIR = $$(CUDA_PATH)
octproz_project/octproz/pri/cuda.pri:	isEmpty(NVCUDASAMPLES_ROOT){
octproz_project/octproz/pri/cuda.pri:		NVCUDASAMPLES_ROOT = $$shell_path($$PWD/../../thirdparty/cuda) #in older CUDA versions NVCUDASAMPLES_ROOT is defined. it is the location of the CUDA samples folder
octproz_project/octproz/pri/cuda.pri:	INCLUDEPATH += $$shell_path($$(NVCUDASAMPLES_ROOT)/common/inc)
octproz_project/octproz/pri/cuda.pri:	INCLUDEPATH_CUDA += $$shell_path($$(NVCUDASAMPLES_ROOT)/common/inc)
octproz_project/octproz/pri/cuda.pri:	QMAKE_LIBDIR += $$CUDA_DIR/lib/$$SYSTEM_NAME
octproz_project/octproz/pri/cuda.pri:INCLUDEPATH += $$CUDA_DIR/include \
octproz_project/octproz/pri/cuda.pri:	$$NVCUDASAMPLES_ROOT/common/inc
octproz_project/octproz/pri/cuda.pri:INCLUDEPATH_CUDA += $$[QT_INSTALL_HEADERS] \
octproz_project/octproz/pri/cuda.pri:	$$CUDA_DIR/include \
octproz_project/octproz/pri/cuda.pri:	$$NVCUDASAMPLES_ROOT/common/inc
octproz_project/octproz/pri/cuda.pri:#cuda libraries
octproz_project/octproz/pri/cuda.pri:	CUDA_LIBS += -lcudart -lcuda -lcufft -lculibos
octproz_project/octproz/pri/cuda.pri:	CUDA_LIBS += -lcudart -lcuda -lcufft
octproz_project/octproz/pri/cuda.pri:LIBS += $$CUDA_LIBS
octproz_project/octproz/pri/cuda.pri:CUDA_INC = $$join(INCLUDEPATH_CUDA,'" -I"','-I"','"')
octproz_project/octproz/pri/cuda.pri:	QMAKE_LFLAGS += -Wl,-rpath,$$CUDA_DIR/lib
octproz_project/octproz/pri/cuda.pri:	NVCCFLAGS = -Xlinker -rpath,$$CUDA_DIR/lib
octproz_project/octproz/pri/cuda.pri:#cuda compiler configuration
octproz_project/octproz/pri/cuda.pri:CUDA_OBJECTS_DIR = ./
octproz_project/octproz/pri/cuda.pri:		cuda_d.input = CUDA_SOURCES
octproz_project/octproz/pri/cuda.pri:		cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
octproz_project/octproz/pri/cuda.pri:		cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_DEFINES_FLAGS $$CUDA_INC --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
octproz_project/octproz/pri/cuda.pri:		cuda_d.dependency_type = TYPE_C
octproz_project/octproz/pri/cuda.pri:		QMAKE_EXTRA_COMPILERS += cuda_d
octproz_project/octproz/pri/cuda.pri:		cuda.input = CUDA_SOURCES
octproz_project/octproz/pri/cuda.pri:		cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
octproz_project/octproz/pri/cuda.pri:		cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_DEFINES_FLAGS $$CUDA_INC --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
octproz_project/octproz/pri/cuda.pri:		cuda.dependency_type = TYPE_C
octproz_project/octproz/pri/cuda.pri:		QMAKE_EXTRA_COMPILERS += cuda
octproz_project/octproz/pri/cuda.pri:		cuda_d.input = CUDA_SOURCES
octproz_project/octproz/pri/cuda.pri:		cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
octproz_project/octproz/pri/cuda.pri:		cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_DEFINES_FLAGS $$CUDA_INC --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
octproz_project/octproz/pri/cuda.pri:					  --compile -cudart static -g -DWIN32 -D_MBCS \
octproz_project/octproz/pri/cuda.pri:		cuda_d.dependency_type = TYPE_C
octproz_project/octproz/pri/cuda.pri:		QMAKE_EXTRA_COMPILERS += cuda_d
octproz_project/octproz/pri/cuda.pri:		cuda.input = CUDA_SOURCES
octproz_project/octproz/pri/cuda.pri:		cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
octproz_project/octproz/pri/cuda.pri:		cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_DEFINES_FLAGS $$CUDA_INC --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
octproz_project/octproz/pri/cuda.pri:					--compile -cudart static -DWIN32 -D_MBCS \
octproz_project/octproz/pri/cuda.pri:		cuda.dependency_type = TYPE_C
octproz_project/octproz/pri/cuda.pri:		QMAKE_EXTRA_COMPILERS += cuda
octproz_project/octproz/pri/cuda.pri:message(CUDA_INC is $$CUDA_INC)
octproz_project/octproz/src/octproz.cpp:	connect(this->signalProcessing, &Processing::processedRecordDone, this, &OCTproZ::slot_resetGpu2HostSettings);
octproz_project/octproz/src/octproz.cpp:	connect(this->bscanWindow, &GLWindow2D::registerBufferCudaGL, this->signalProcessing, &Processing::slot_registerBscanOpenGLbufferWithCuda);
octproz_project/octproz/src/octproz.cpp:	connect(this->enFaceViewWindow, &GLWindow2D::registerBufferCudaGL, this->signalProcessing, &Processing::slot_registerEnFaceViewOpenGLbufferWithCuda);
octproz_project/octproz/src/octproz.cpp:	connect(this->volumeWindow, &GLWindow3D::registerBufferCudaGL, this->signalProcessing, &Processing::slot_registerVolumeViewOpenGLbufferWithCuda);
octproz_project/octproz/src/octproz.cpp:	connect(this->signalProcessing, &Processing::initOpenGLenFaceView, this->enFaceViewWindow, &GLWindow2D::registerOpenGLBufferWithCuda);
octproz_project/octproz/src/octproz.cpp:	connect(this->signalProcessing, &Processing::initOpenGLenFaceView, this->volumeWindow, &GLWindow3D::registerOpenGLBufferWithCuda);
octproz_project/octproz/src/octproz.cpp:	this->processedDataNotifier = Gpu2HostNotifier::getInstance();
octproz_project/octproz/src/octproz.cpp:	connect(this->processedDataNotifier, &Gpu2HostNotifier::newGpuDataAvailible, this->plot1D, &PlotWindow1D::slot_plotProcessedData);
octproz_project/octproz/src/octproz.cpp:	connect(this->processedDataNotifier, &Gpu2HostNotifier::backgroundRecorded, this->sidebar, &Sidebar::updateBackgroundPlot);
octproz_project/octproz/src/octproz.cpp://	connect(this->processedDataNotifier, &Gpu2HostNotifier::bscanDisplayBufferReady, this->bscanWindow, QOverload<>::of(&GLWindow2D::update));
octproz_project/octproz/src/octproz.cpp://	connect(this->processedDataNotifier, &Gpu2HostNotifier::enfaceDisplayBufferReady, this->enFaceViewWindow, QOverload<>::of(&GLWindow2D::update));
octproz_project/octproz/src/octproz.cpp://	connect(this->processedDataNotifier, &Gpu2HostNotifier::volumeDisplayBufferReady, this->volumeWindow, QOverload<>::of(&GLWindow3D::update));
octproz_project/octproz/src/octproz.cpp:	connect(&notifierThread, &QThread::finished, this->processedDataNotifier, &Gpu2HostNotifier::deleteLater);
octproz_project/octproz/src/octproz.cpp:			QThread::msleep(1000); //provide some time to let gpu computation finish
octproz_project/octproz/src/octproz.cpp:		this->slot_prepareGpu2HostForProcessedRecording();
octproz_project/octproz/src/octproz.cpp:				connect(this->processedDataNotifier, &Gpu2HostNotifier::newGpuDataAvailible, extension, &Extension::processedDataReceived);
octproz_project/octproz/src/octproz.cpp:					disconnect(this->processedDataNotifier, &Gpu2HostNotifier::newGpuDataAvailible, extension, &Extension::processedDataReceived);
octproz_project/octproz/src/octproz.cpp:	disconnect(this->processedDataNotifier, &Gpu2HostNotifier::newGpuDataAvailible, extension, &Extension::processedDataReceived);
octproz_project/octproz/src/octproz.cpp:void OCTproZ::slot_prepareGpu2HostForProcessedRecording() {
octproz_project/octproz/src/octproz.cpp:void OCTproZ::slot_resetGpu2HostSettings() {
octproz_project/octproz/src/glwindow2d.cpp:	emit registerBufferCudaGL(this->buf);
octproz_project/octproz/src/glwindow2d.cpp:void GLWindow2D::registerOpenGLBufferWithCuda() {
octproz_project/octproz/src/glwindow2d.cpp:	emit registerBufferCudaGL(this->buf);
octproz_project/octproz/src/glwindow2d.cpp:		emit registerBufferCudaGL(this->buf); //registerBufferCudaGL is necessary here because as soon as the openglwidget/dock is removed from the main window initializeGL() is called again.
octproz_project/octproz/src/gpu2hostnotifier.cpp:#include "gpu2hostnotifier.h"
octproz_project/octproz/src/gpu2hostnotifier.cpp:Gpu2HostNotifier* Gpu2HostNotifier::gpu2hostNotifier = nullptr;
octproz_project/octproz/src/gpu2hostnotifier.cpp:Gpu2HostNotifier::Gpu2HostNotifier(QObject *parent)
octproz_project/octproz/src/gpu2hostnotifier.cpp:Gpu2HostNotifier* Gpu2HostNotifier::getInstance(QObject* parent) {
octproz_project/octproz/src/gpu2hostnotifier.cpp:	gpu2hostNotifier = gpu2hostNotifier != nullptr ? gpu2hostNotifier : new Gpu2HostNotifier(parent);
octproz_project/octproz/src/gpu2hostnotifier.cpp:	return gpu2hostNotifier;
octproz_project/octproz/src/gpu2hostnotifier.cpp:Gpu2HostNotifier::~Gpu2HostNotifier()
octproz_project/octproz/src/gpu2hostnotifier.cpp:void Gpu2HostNotifier::emitCurrentStreamingBuffer(void* streamingBuffer) {
octproz_project/octproz/src/gpu2hostnotifier.cpp:	emit newGpuDataAvailible(streamingBuffer, params->bitDepth, params->samplesPerLine / 2, params->ascansPerBscan, params->bscansPerBuffer, params->buffersPerVolume, params->currentBufferNr);
octproz_project/octproz/src/gpu2hostnotifier.cpp:void Gpu2HostNotifier::emitBackgroundRecorded() {
octproz_project/octproz/src/gpu2hostnotifier.cpp:void Gpu2HostNotifier::emitBscanDisplayBufferReady(void* data) {
octproz_project/octproz/src/gpu2hostnotifier.cpp:void Gpu2HostNotifier::emitEnfaceDisplayBufferReady(void* data) {
octproz_project/octproz/src/gpu2hostnotifier.cpp:void Gpu2HostNotifier::emitVolumeDisplayBufferReady(void* data) {
octproz_project/octproz/src/gpu2hostnotifier.cpp:void CUDART_CB Gpu2HostNotifier::dh2StreamingCallback(void* currStreamingBuffer) {
octproz_project/octproz/src/gpu2hostnotifier.cpp:	Gpu2HostNotifier::getInstance()->emitCurrentStreamingBuffer(currStreamingBuffer);
octproz_project/octproz/src/gpu2hostnotifier.cpp:void CUDART_CB Gpu2HostNotifier::backgroundSignalCallback(void* backgroundSignal) {
octproz_project/octproz/src/gpu2hostnotifier.cpp:	Gpu2HostNotifier::getInstance()->emitBackgroundRecorded();
octproz_project/octproz/src/gpu2hostnotifier.cpp:void CUDART_CB Gpu2HostNotifier::bscanDisblayBufferReadySignalCallback(void* data) {
octproz_project/octproz/src/gpu2hostnotifier.cpp:	Gpu2HostNotifier::getInstance()->emitBscanDisplayBufferReady(data);
octproz_project/octproz/src/gpu2hostnotifier.cpp:void CUDART_CB Gpu2HostNotifier::enfaceDisplayBufferReadySignalCallback(void* data) {
octproz_project/octproz/src/gpu2hostnotifier.cpp:	Gpu2HostNotifier::getInstance()->emitEnfaceDisplayBufferReady(data);
octproz_project/octproz/src/gpu2hostnotifier.cpp:void CUDART_CB Gpu2HostNotifier::volumeDisblayBufferReadySignalCallback(void* data) {
octproz_project/octproz/src/gpu2hostnotifier.cpp:	Gpu2HostNotifier::getInstance()->emitVolumeDisplayBufferReady(data);
octproz_project/octproz/src/octproz.h:	void slot_prepareGpu2HostForProcessedRecording();
octproz_project/octproz/src/octproz.h:	void slot_resetGpu2HostSettings();
octproz_project/octproz/src/octproz.h:	Gpu2HostNotifier* processedDataNotifier;
octproz_project/octproz/src/glwindow3d.cpp:		emit registerBufferCudaGL(this->raycastingVolume->getVolumeTexture()); //registerBufferCudaGL is necessary here because as soon as the openglwidget/dock is removed from the main window initializeGL() is called again. //todo: check if opengl context (buffer, texture,...) cleanup is necessary!
octproz_project/octproz/src/glwindow3d.cpp:			QTimer::singleShot(REFRESH_INTERVAL_IN_ms, this, QOverload<>::of(&GLWindow3D::delayedUpdate)); //todo: consider using Gpu2HostNotifier to notify GLWindow3D when new volume data is available
octproz_project/octproz/src/glwindow3d.cpp:	emit registerBufferCudaGL(this->raycastingVolume->getVolumeTexture());
octproz_project/octproz/src/glwindow3d.cpp:void GLWindow3D::registerOpenGLBufferWithCuda() {
octproz_project/octproz/src/glwindow3d.cpp:		emit registerBufferCudaGL(this->raycastingVolume->getVolumeTexture());
octproz_project/octproz/src/processing.h:#include "gpu2hostnotifier.h"
octproz_project/octproz/src/processing.h:	void initCudaOpenGlInterop();
octproz_project/octproz/src/processing.h:	bool waitForCudaOpenGlInteropReady(int interval, int timeout);
octproz_project/octproz/src/processing.h:	bool isCudaOpenGlInteropReady();
octproz_project/octproz/src/processing.h:	bool bscanGlBufferRegisteredWithCuda;
octproz_project/octproz/src/processing.h:	bool enfaceGlBufferRegisteredWithCuda;
octproz_project/octproz/src/processing.h:	bool volumeGlBufferRegisteredWithCuda;
octproz_project/octproz/src/processing.h:	void slot_registerBscanOpenGLbufferWithCuda(unsigned int openGLbufferId);
octproz_project/octproz/src/processing.h:	void slot_registerEnFaceViewOpenGLbufferWithCuda(unsigned int openGLbufferId);
octproz_project/octproz/src/processing.h:	void slot_registerVolumeViewOpenGLbufferWithCuda(unsigned int openGLbufferId);
octproz_project/octproz/src/processing.h:	void enableGpu2HostStreaming(bool enableStreaming);
octproz_project/octproz/src/glwindow3d.h:	void registerBufferCudaGL(unsigned int bufferId);
octproz_project/octproz/src/glwindow3d.h:	void initCudaGl();
octproz_project/octproz/src/glwindow3d.h:	void registerOpenGLBufferWithCuda();
octproz_project/octproz/src/sidebar.cpp:	//GPU to RAM Streaming
octproz_project/octproz/src/sidebar.cpp:	//GPU to RAM Streaming
octproz_project/octproz/src/cuda_code.cu:#ifndef CUDA_CODE_CU
octproz_project/octproz/src/cuda_code.cu:#define CUDA_CODE_CU
octproz_project/octproz/src/cuda_code.cu:#if __CUDACC_VER_MAJOR__ <12
octproz_project/octproz/src/cuda_code.cu:surface<void, cudaSurfaceType3D> surfaceWrite;
octproz_project/octproz/src/cuda_code.cu:cudaStream_t stream[nStreams];
octproz_project/octproz/src/cuda_code.cu:cudaStream_t userRequestStream;
octproz_project/octproz/src/cuda_code.cu:cudaEvent_t syncEvent;
octproz_project/octproz/src/cuda_code.cu:cudaGraphicsResource* cuBufHandleBscan = NULL;
octproz_project/octproz/src/cuda_code.cu:cudaGraphicsResource* cuBufHandleEnFaceView = NULL;
octproz_project/octproz/src/cuda_code.cu:cudaGraphicsResource* cuBufHandleVolumeView = NULL;
octproz_project/octproz/src/cuda_code.cu:bool cudaInitialized = false;
octproz_project/octproz/src/cuda_code.cu://todo: use/evaluate cuda texture for interpolation in klinearization kernel
octproz_project/octproz/src/cuda_code.cu://todo: optimize cuda code
octproz_project/octproz/src/cuda_code.cu:extern "C" void cuda_updateDispersionCurve(float* h_dispersionCurve, int size, cudaStream_t stream) {
octproz_project/octproz/src/cuda_code.cu:		checkCudaErrors(cudaMemcpyAsync(d_dispersionCurve, h_dispersionCurve, size * sizeof(float), cudaMemcpyHostToDevice, stream));
octproz_project/octproz/src/cuda_code.cu:extern "C" void cuda_updateWindowCurve(float* h_windowCurve, int size, cudaStream_t stream) {
octproz_project/octproz/src/cuda_code.cu:		checkCudaErrors(cudaMemcpyAsync(d_windowCurve, h_windowCurve, size * sizeof(float), cudaMemcpyHostToDevice, stream));
octproz_project/octproz/src/cuda_code.cu:extern "C" void cuda_updatePostProcessBackground(float* h_postProcessBackground, int size, cudaStream_t stream) {
octproz_project/octproz/src/cuda_code.cu:		checkCudaErrors(cudaMemcpy(d_postProcBackgroundLine, h_postProcessBackground, size * sizeof(float), cudaMemcpyHostToDevice));
octproz_project/octproz/src/cuda_code.cu:		cudaStreamSynchronize(stream);
octproz_project/octproz/src/cuda_code.cu:		checkCudaErrors(cudaMemcpyAsync(d_postProcBackgroundLine, h_postProcessBackground, size * sizeof(float), cudaMemcpyHostToDevice, stream));
octproz_project/octproz/src/cuda_code.cu:extern "C" void cuda_copyPostProcessBackgroundToHost(float* h_postProcessBackground, int size, cudaStream_t stream) {
octproz_project/octproz/src/cuda_code.cu:		checkCudaErrors(cudaMemcpy(h_postProcessBackground, d_postProcBackgroundLine, size * sizeof(float), cudaMemcpyDeviceToHost));
octproz_project/octproz/src/cuda_code.cu:		cudaStreamSynchronize(stream);
octproz_project/octproz/src/cuda_code.cu:		Gpu2HostNotifier::backgroundSignalCallback(h_postProcessBackground);
octproz_project/octproz/src/cuda_code.cu:		checkCudaErrors(cudaMemcpyAsync(h_postProcessBackground, d_postProcBackgroundLine, size * sizeof(float), cudaMemcpyDeviceToHost, stream));
octproz_project/octproz/src/cuda_code.cu:		checkCudaErrors(cudaLaunchHostFunc(stream, Gpu2HostNotifier::backgroundSignalCallback, h_postProcessBackground));
octproz_project/octproz/src/cuda_code.cu:extern "C" void cuda_registerStreamingBuffers(void* h_streamingBuffer1, void* h_streamingBuffer2, size_t bytesPerBuffer) {
octproz_project/octproz/src/cuda_code.cu:	checkCudaErrors(cudaHostRegister(h_streamingBuffer1, bytesPerBuffer, cudaHostRegisterPortable));
octproz_project/octproz/src/cuda_code.cu:	checkCudaErrors(cudaHostRegister(h_streamingBuffer2, bytesPerBuffer, cudaHostRegisterPortable));
octproz_project/octproz/src/cuda_code.cu:extern "C" void cuda_unregisterStreamingBuffers() {
octproz_project/octproz/src/cuda_code.cu:	checkCudaErrors(cudaHostUnregister(host_streamingBuffer1));
octproz_project/octproz/src/cuda_code.cu:	checkCudaErrors(cudaHostUnregister(host_streamingBuffer2));
octproz_project/octproz/src/cuda_code.cu:__global__ void cuda_bscanFlip_slow(float *output, float *input, int samplesPerAscan, int ascansPerBscan, int samplesInVolume) {
octproz_project/octproz/src/cuda_code.cu://todo: optimize! cuda_bscanFlip should be possible with just index < samplesPerBuffer/4
octproz_project/octproz/src/cuda_code.cu:__global__ void cuda_bscanFlip(float *output, float *input, const int samplesPerAscan, const int ascansPerBscan, const int samplesPerBscan, const int halfSamplesInVolume) {
octproz_project/octproz/src/cuda_code.cu:#if __CUDACC_VER_MAJOR__ >=12
octproz_project/octproz/src/cuda_code.cu:__global__ void updateDisplayedVolume(cudaSurfaceObject_t surfaceWrite, const float* processedBuffer, const unsigned int samplesInBuffer, const unsigned int currBufferNr, const unsigned int bscansPerBuffer, dim3 textureDim) {
octproz_project/octproz/src/cuda_code.cu:extern "C" void cuda_updateResampleCurve(float* h_resampleCurve, int size, cudaStream_t stream) {
octproz_project/octproz/src/cuda_code.cu:		checkCudaErrors(cudaMemcpyAsync(d_resampleCurve, h_resampleCurve, size * sizeof(float), cudaMemcpyHostToDevice, stream));
octproz_project/octproz/src/cuda_code.cu:	cudaError_t status;
octproz_project/octproz/src/cuda_code.cu:	status = cudaMemGetInfo(&freeMem, &totalMem);
octproz_project/octproz/src/cuda_code.cu:	if(status != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:		printf("Cuda: Error retrieving memory info: %s\n", cudaGetErrorString(status));
octproz_project/octproz/src/cuda_code.cu:		status = cudaMalloc(d_buffer, bufferSize);
octproz_project/octproz/src/cuda_code.cu:		if(status != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:			printf("cudaMalloc failed: %s\n", cudaGetErrorString(status));
octproz_project/octproz/src/cuda_code.cu:				freeCudaMem(d_buffer); //cleanup on failure
octproz_project/octproz/src/cuda_code.cu:		status = cudaMemset(*d_buffer, 0, bufferSize);
octproz_project/octproz/src/cuda_code.cu:		if(status != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:			printf("cudaMemsetAsync failed: %s\n", cudaGetErrorString(status));
octproz_project/octproz/src/cuda_code.cu:				freeCudaMem(d_buffer); //cleanup on failure
octproz_project/octproz/src/cuda_code.cu:		printf("Cuda: Not enough memory available.\n");
octproz_project/octproz/src/cuda_code.cu:	cudaError_t err;
octproz_project/octproz/src/cuda_code.cu:	err = cudaStreamCreate(&userRequestStream);
octproz_project/octproz/src/cuda_code.cu:	if (err != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:		printf("Cuda: Failed to create stream: %s\n", cudaGetErrorString(err));
octproz_project/octproz/src/cuda_code.cu:		err = cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
octproz_project/octproz/src/cuda_code.cu:		if (err != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:			printf("cuda: Failed to create stream %d: %s\n", i, cudaGetErrorString(err));
octproz_project/octproz/src/cuda_code.cu:				cudaStreamDestroy(stream[j]);
octproz_project/octproz/src/cuda_code.cu:			cudaStreamDestroy(userRequestStream);
octproz_project/octproz/src/cuda_code.cu:	err = cudaEventCreateWithFlags(&syncEvent, cudaEventBlockingSync);
octproz_project/octproz/src/cuda_code.cu:	if (err != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:		printf("Failed to create synchronization event: %s\n", cudaGetErrorString(err));
octproz_project/octproz/src/cuda_code.cu:			cudaStreamDestroy(stream[i]);
octproz_project/octproz/src/cuda_code.cu:		cudaStreamDestroy(userRequestStream);
octproz_project/octproz/src/cuda_code.cu:	cudaGetDevice(&currentDevice);
octproz_project/octproz/src/cuda_code.cu:	cudaDeviceProp deviceProp;
octproz_project/octproz/src/cuda_code.cu:	cudaGetDeviceProperties(&deviceProp, currentDevice);
octproz_project/octproz/src/cuda_code.cu:extern "C" bool initializeCuda(void* h_buffer1, void* h_buffer2, OctAlgorithmParameters* parameters) {
octproz_project/octproz/src/cuda_code.cu:	checkCudaErrors(cudaPeekAtLastError());
octproz_project/octproz/src/cuda_code.cu:	checkCudaErrors(cudaDeviceSynchronize());
octproz_project/octproz/src/cuda_code.cu:	//allocate device memory for the raw signal. On the Jetson Nano (__aarch64__), this allocation can be skipped if the acquisition buffer is created with cudaHostAlloc with the flag cudaHostAllocMapped, this way it can be accessed by both CPU and GPU; no extra device buffer is necessary.
octproz_project/octproz/src/cuda_code.cu:#if !defined(__aarch64__) || !defined(ENABLE_CUDA_ZERO_COPY)
octproz_project/octproz/src/cuda_code.cu:	//register existing host memory for use by cuda to accelerate cudaMemcpy. 
octproz_project/octproz/src/cuda_code.cu:	//cudaHostAlloc and the cudaHostAllocMapped flag, which allows for zero-copy access.
octproz_project/octproz/src/cuda_code.cu:	checkCudaErrors(cudaHostRegister(host_buffer1, samplesPerBuffer * bytesPerSample, cudaHostRegisterPortable));
octproz_project/octproz/src/cuda_code.cu:	checkCudaErrors(cudaHostRegister(host_buffer2, samplesPerBuffer * bytesPerSample, cudaHostRegisterPortable));
octproz_project/octproz/src/cuda_code.cu:	checkCudaErrors(cufftPlan1d(&d_plan, signalLength, CUFFT_C2C, ascansPerBscan*bscansPerBuffer));
octproz_project/octproz/src/cuda_code.cu:	checkCudaErrors(cudaPeekAtLastError());
octproz_project/octproz/src/cuda_code.cu:	checkCudaErrors(cudaDeviceSynchronize());
octproz_project/octproz/src/cuda_code.cu:	cudaInitialized = true;
octproz_project/octproz/src/cuda_code.cu:#if !defined(__aarch64__) || !defined(ENABLE_CUDA_ZERO_COPY)
octproz_project/octproz/src/cuda_code.cu:			freeCudaMem((void**)&d_inputBuffer[i]);
octproz_project/octproz/src/cuda_code.cu:		freeCudaMem((void**)&d_outputBuffer);
octproz_project/octproz/src/cuda_code.cu:		freeCudaMem((void**)&d_windowCurve);
octproz_project/octproz/src/cuda_code.cu:		freeCudaMem((void**)&d_fftBuffer);
octproz_project/octproz/src/cuda_code.cu:		freeCudaMem((void**)&d_meanALine);
octproz_project/octproz/src/cuda_code.cu:		freeCudaMem((void**)&d_postProcBackgroundLine);
octproz_project/octproz/src/cuda_code.cu:		freeCudaMem((void**)&d_processedBuffer);
octproz_project/octproz/src/cuda_code.cu:		freeCudaMem((void**)&d_sinusoidalScanTmpBuffer);
octproz_project/octproz/src/cuda_code.cu:		freeCudaMem((void**)&d_inputLinearized);
octproz_project/octproz/src/cuda_code.cu:		freeCudaMem((void**)&d_phaseCartesian);
octproz_project/octproz/src/cuda_code.cu:		freeCudaMem((void**)&d_resampleCurve);
octproz_project/octproz/src/cuda_code.cu:		freeCudaMem((void**)&d_dispersionCurve);
octproz_project/octproz/src/cuda_code.cu:		freeCudaMem((void**)&d_sinusoidalResampleCurve);
octproz_project/octproz/src/cuda_code.cu:		checkCudaErrors(cudaStreamDestroy(userRequestStream));
octproz_project/octproz/src/cuda_code.cu:			checkCudaErrors(cudaStreamDestroy(stream[i]));
octproz_project/octproz/src/cuda_code.cu:		checkCudaErrors(cudaEventDestroy(syncEvent));
octproz_project/octproz/src/cuda_code.cu:extern "C" void cleanupCuda() {
octproz_project/octproz/src/cuda_code.cu:	if (cudaInitialized) {
octproz_project/octproz/src/cuda_code.cu:			cudaHostUnregister(host_buffer1);
octproz_project/octproz/src/cuda_code.cu:			cudaHostUnregister(host_buffer2);
octproz_project/octproz/src/cuda_code.cu:		cudaInitialized = false;
octproz_project/octproz/src/cuda_code.cu:extern "C" void freeCudaMem(void** data) {
octproz_project/octproz/src/cuda_code.cu:		checkCudaErrors(cudaFree(*data));
octproz_project/octproz/src/cuda_code.cu:		printf("Cuda: Failed to free memory.\n");
octproz_project/octproz/src/cuda_code.cu:		d_bscanDisplayBuffer = cuda_map(cuBufHandleBscan, userRequestStream);
octproz_project/octproz/src/cuda_code.cu:		cuda_unmap(cuBufHandleBscan, userRequestStream);
octproz_project/octproz/src/cuda_code.cu:		d_enFaceViewDisplayBuffer = cuda_map(cuBufHandleEnFaceView, userRequestStream);
octproz_project/octproz/src/cuda_code.cu:		cuda_unmap(cuBufHandleEnFaceView, userRequestStream);
octproz_project/octproz/src/cuda_code.cu:extern "C" inline void updateBscanDisplayBuffer(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction, cudaStream_t stream) {
octproz_project/octproz/src/cuda_code.cu:		d_bscanDisplayBuffer = cuda_map(cuBufHandleBscan, stream);
octproz_project/octproz/src/cuda_code.cu:		cuda_unmap(cuBufHandleBscan, stream);
octproz_project/octproz/src/cuda_code.cu:extern "C" inline void updateEnFaceDisplayBuffer(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction, cudaStream_t stream) {
octproz_project/octproz/src/cuda_code.cu:		d_enFaceViewDisplayBuffer = cuda_map(cuBufHandleEnFaceView, stream);
octproz_project/octproz/src/cuda_code.cu:		cuda_unmap(cuBufHandleEnFaceView, stream);
octproz_project/octproz/src/cuda_code.cu:extern "C" inline void updateVolumeDisplayBuffer(const float* d_currBuffer, const unsigned int currentBufferNr, const unsigned int bscansPerBuffer, cudaStream_t stream) {
octproz_project/octproz/src/cuda_code.cu:	//map graphics resource for access by cuda
octproz_project/octproz/src/cuda_code.cu:	cudaArray* d_volumeViewDisplayBuffer = NULL;
octproz_project/octproz/src/cuda_code.cu:		d_volumeViewDisplayBuffer = cuda_map3dTexture(cuBufHandleVolumeView, stream);
octproz_project/octproz/src/cuda_code.cu:#if __CUDACC_VER_MAJOR__ >=12
octproz_project/octproz/src/cuda_code.cu:	        cudaResourceDesc surfRes;
octproz_project/octproz/src/cuda_code.cu:	        surfRes.resType = cudaResourceTypeArray;
octproz_project/octproz/src/cuda_code.cu:	        cudaSurfaceObject_t surfaceWrite;
octproz_project/octproz/src/cuda_code.cu:	        cudaError_t error_id = cudaCreateSurfaceObject(&surfaceWrite, &surfRes);
octproz_project/octproz/src/cuda_code.cu:	        if (error_id != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:	            printf("Cuda: Failed to create surface object: %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
octproz_project/octproz/src/cuda_code.cu:	        cudaDestroySurfaceObject(surfaceWrite);
octproz_project/octproz/src/cuda_code.cu:		//bind voxel array to a writable cuda surface
octproz_project/octproz/src/cuda_code.cu:		cudaError_t error_id = cudaBindSurfaceToArray(surfaceWrite, d_volumeViewDisplayBuffer);
octproz_project/octproz/src/cuda_code.cu:		if (error_id != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:			printf("Cuda: Failed to bind surface to cuda array:  %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
octproz_project/octproz/src/cuda_code.cu:		//write to cuda surface
octproz_project/octproz/src/cuda_code.cu:		cuda_unmap(cuBufHandleVolumeView, stream);
octproz_project/octproz/src/cuda_code.cu:inline void streamProcessedData(float* d_currProcessedBuffer, cudaStream_t stream) {
octproz_project/octproz/src/cuda_code.cu:#if defined(__aarch64__) && defined(ENABLE_CUDA_ZERO_COPY)
octproz_project/octproz/src/cuda_code.cu:		checkCudaErrors(cudaHostGetDevicePointer((void**)&d_outputBuffer, (void*)hostDestBuffer, 0));
octproz_project/octproz/src/cuda_code.cu:#if !defined(__aarch64__) || !defined(ENABLE_CUDA_ZERO_COPY)
octproz_project/octproz/src/cuda_code.cu:		checkCudaErrors(cudaMemcpyAsync(hostDestBuffer, (void*)d_outputBuffer, (samplesPerBuffer / 2) * bytesPerSample, cudaMemcpyDeviceToHost, stream));
octproz_project/octproz/src/cuda_code.cu:		checkCudaErrors(cudaLaunchHostFunc(stream, Gpu2HostNotifier::dh2StreamingCallback, hostDestBuffer));
octproz_project/octproz/src/cuda_code.cu:extern "C" void octCudaPipeline(void* h_inputSignal) {
octproz_project/octproz/src/cuda_code.cu:	//check if cuda buffers are initialized
octproz_project/octproz/src/cuda_code.cu:	if (!cudaInitialized) {
octproz_project/octproz/src/cuda_code.cu:		printf("Cuda: Device buffers are not initialized!\n");
octproz_project/octproz/src/cuda_code.cu:#if defined(__aarch64__) && defined(ENABLE_CUDA_ZERO_COPY)
octproz_project/octproz/src/cuda_code.cu:		checkCudaErrors(cudaHostGetDevicePointer((void**)&d_inputBuffer[currBuffer], (void*)h_inputSignal, 0));
octproz_project/octproz/src/cuda_code.cu:		checkCudaErrors(cudaMemcpyAsync(d_inputBuffer[currBuffer], h_inputSignal, samplesPerBuffer * bytesPerSample, cudaMemcpyHostToDevice, stream[currStream]));
octproz_project/octproz/src/cuda_code.cu:	//synchronization: block the host during cudaMemcpyAsync and inputToCufftComplex to prevent the data acquisition of the virtual OCT system from outpacing the processing, ensuring proper synchronization in the pipeline.
octproz_project/octproz/src/cuda_code.cu:#if !defined(__aarch64__) || !defined(ENABLE_CUDA_ZERO_COPY)
octproz_project/octproz/src/cuda_code.cu:	cudaEventRecord(syncEvent, stream[currStream]);
octproz_project/octproz/src/cuda_code.cu:	cudaEventSynchronize(syncEvent);
octproz_project/octproz/src/cuda_code.cu:		cuda_updateResampleCurve(params->resampleCurve, params->resampleCurveLength, stream[currStream]);
octproz_project/octproz/src/cuda_code.cu:		cuda_updateDispersionCurve(params->dispersionCurve, signalLength, stream[currStream]);
octproz_project/octproz/src/cuda_code.cu:		cuda_updateWindowCurve(params->windowCurve, signalLength, stream[currStream]);
octproz_project/octproz/src/cuda_code.cu:	checkCudaErrors(cufftExecC2C(d_plan, d_fftBuffer2, d_fftBuffer2, CUFFT_INVERSE));
octproz_project/octproz/src/cuda_code.cu:		cuda_bscanFlip<<<gridSize/2, blockSize, 0, stream[currStream]>>> (d_currBuffer, d_currBuffer, signalLength / 2, ascansPerBscan, (signalLength*ascansPerBscan)/2, samplesPerBuffer/4);
octproz_project/octproz/src/cuda_code.cu:		checkCudaErrors(cudaMemcpyAsync(d_sinusoidalScanTmpBuffer, d_currBuffer, sizeof(float)*samplesPerBuffer/2, cudaMemcpyDeviceToDevice,stream[currStream]));
octproz_project/octproz/src/cuda_code.cu:			cuda_copyPostProcessBackgroundToHost(params->postProcessBackground, signalLength/2, stream[currStream]);
octproz_project/octproz/src/cuda_code.cu:			cuda_updatePostProcessBackground(params->postProcessBackground, signalLength/2, stream[currStream]);
octproz_project/octproz/src/cuda_code.cu:		//checkCudaErrors(cudaLaunchHostFunc(stream[currStream], Gpu2HostNotifier::bscanDisblayBufferReadySignalCallback, 0));
octproz_project/octproz/src/cuda_code.cu:		//checkCudaErrors(cudaLaunchHostFunc(stream[currStream], Gpu2HostNotifier::enfaceDisplayBufferReadySignalCallback, 0));
octproz_project/octproz/src/cuda_code.cu:		//checkCudaErrors(cudaLaunchHostFunc(stream[currStream], Gpu2HostNotifier::volumeDisblayBufferReadySignalCallback, 0));
octproz_project/octproz/src/cuda_code.cu:#if defined(__aarch64__) && defined(ENABLE_CUDA_ZERO_COPY)
octproz_project/octproz/src/cuda_code.cu:	cudaEventRecord(syncEvent, stream[currStream]);
octproz_project/octproz/src/cuda_code.cu:	cudaEventSynchronize(syncEvent);
octproz_project/octproz/src/cuda_code.cu:	cudaError_t err = cudaGetLastError();
octproz_project/octproz/src/cuda_code.cu:	if (err != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:		printf("Cuda error: %s\n", cudaGetErrorString(err));
octproz_project/octproz/src/cuda_code.cu:extern "C" bool cuda_registerGlBufferBscan(GLuint buf) {
octproz_project/octproz/src/cuda_code.cu:		cudaError_t unregisterResult = cudaGraphicsUnregisterResource(cuBufHandleBscan);
octproz_project/octproz/src/cuda_code.cu:		if (unregisterResult != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:			printf("Cuda: Failed to unregister existing resource. Error: %s\n", cudaGetErrorString(unregisterResult));
octproz_project/octproz/src/cuda_code.cu:	cudaError_t registerResult = cudaGraphicsGLRegisterBuffer(&cuBufHandleBscan, buf, cudaGraphicsRegisterFlagsWriteDiscard);
octproz_project/octproz/src/cuda_code.cu:	if (registerResult != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:		printf("Cuda: Failed to register buffer %u. Error: %s\n", buf, cudaGetErrorString(registerResult));
octproz_project/octproz/src/cuda_code.cu:extern "C" bool cuda_registerGlBufferEnFaceView(GLuint buf) {
octproz_project/octproz/src/cuda_code.cu:		cudaError_t unregisterResult = cudaGraphicsUnregisterResource(cuBufHandleEnFaceView);
octproz_project/octproz/src/cuda_code.cu:		if (unregisterResult != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:			printf("Cuda: Failed to unregister existing resource. Error: %s\n", cudaGetErrorString(unregisterResult));
octproz_project/octproz/src/cuda_code.cu:	if (cudaGraphicsGLRegisterBuffer(&cuBufHandleEnFaceView, buf, cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:		printf("Cuda: Failed to register buffer %u\n", buf);
octproz_project/octproz/src/cuda_code.cu:extern "C" bool cuda_registerGlBufferVolumeView(GLuint buf) {
octproz_project/octproz/src/cuda_code.cu:		cudaError_t unregisterResult = cudaGraphicsUnregisterResource(cuBufHandleVolumeView);
octproz_project/octproz/src/cuda_code.cu:		if (unregisterResult != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:			printf("Cuda: Failed to unregister existing resource. Error: %s\n", cudaGetErrorString(unregisterResult));
octproz_project/octproz/src/cuda_code.cu:	cudaError_t err = cudaGraphicsGLRegisterImage(&cuBufHandleVolumeView, buf, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
octproz_project/octproz/src/cuda_code.cu:	if (err != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:		printf("Cuda: Failed to register buffer %u\n", buf);
octproz_project/octproz/src/cuda_code.cu:void* cuda_map(cudaGraphicsResource* res, cudaStream_t stream) {
octproz_project/octproz/src/cuda_code.cu:	cudaError_t error_id = cudaGraphicsMapResources(1, &res, stream);
octproz_project/octproz/src/cuda_code.cu:	if (error_id != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:		printf("Cuda: Failed to map resource:  %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
octproz_project/octproz/src/cuda_code.cu:	error_id = cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, res);
octproz_project/octproz/src/cuda_code.cu:	if (error_id != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:		printf("Cuda: Failed to get device pointer");
octproz_project/octproz/src/cuda_code.cu:cudaArray* cuda_map3dTexture(cudaGraphicsResource* res, cudaStream_t stream) {
octproz_project/octproz/src/cuda_code.cu:	cudaArray* d_ArrayPtr = 0;
octproz_project/octproz/src/cuda_code.cu:	cudaError_t error_id = cudaGraphicsMapResources(1, &res, stream);
octproz_project/octproz/src/cuda_code.cu:	if (error_id != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:		printf("Cuda: Failed to map 3D texture resource:  %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
octproz_project/octproz/src/cuda_code.cu:	error_id = cudaGraphicsSubResourceGetMappedArray(&d_ArrayPtr, res, 0, 0);
octproz_project/octproz/src/cuda_code.cu:	if (error_id != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:		printf("Cuda: Failed to get device array pointer");
octproz_project/octproz/src/cuda_code.cu:void cuda_unmap(cudaGraphicsResource* res, cudaStream_t stream) {
octproz_project/octproz/src/cuda_code.cu:	if (cudaGraphicsUnmapResources(1, &res, stream) != cudaSuccess) {
octproz_project/octproz/src/cuda_code.cu:		printf("Cuda: Failed to unmap resource");
octproz_project/octproz/src/processing.cpp:	this->bscanGlBufferRegisteredWithCuda = false;
octproz_project/octproz/src/processing.cpp:	this->enfaceGlBufferRegisteredWithCuda = false;
octproz_project/octproz/src/processing.cpp:	this->volumeGlBufferRegisteredWithCuda = false;
octproz_project/octproz/src/processing.cpp:	Gpu2HostNotifier* notifier = Gpu2HostNotifier::getInstance();
octproz_project/octproz/src/processing.cpp:	connect(notifier, &Gpu2HostNotifier::newGpuDataAvailible, this->processedRecorder, &Recorder::slot_record);
octproz_project/octproz/src/processing.cpp:	cleanupCuda();
octproz_project/octproz/src/processing.cpp:void Processing::initCudaOpenGlInterop(){
octproz_project/octproz/src/processing.cpp:	this->bscanGlBufferRegisteredWithCuda = false;
octproz_project/octproz/src/processing.cpp:	this->enfaceGlBufferRegisteredWithCuda = false;
octproz_project/octproz/src/processing.cpp:	this->volumeGlBufferRegisteredWithCuda = false;
octproz_project/octproz/src/processing.cpp:	peekInterval = 220; //this value was determined on a Jetson Nano through trial and error. Lower values can cause issues where CUDA registration for one, two or all OpenGL buffers does not occur at all.
octproz_project/octproz/src/processing.cpp:	this->waitForCudaOpenGlInteropReady(peekInterval, 2000); //this is necessary because initOpenGL(...) triggers a signal emission in glwindow2d containing the OpenGL buffer. This buffer is subsequently registered with CUDA in a slot within this processing class. Thus, this wait time and processEvents() serve as a workaround to ensure the slot executes - meaning the OpenGL buffer gets registered with CUDA - before continuation. //todo: re-examine how the steps for interoperability are called, this many signal slot connections for this straight forward task are too convoluted, probably there is an easyier way for the sequence: create QOffscreenSurface in GUI thread --> allocate OpenGL buffer --> register with cuda --> map buffer to get cuda pointer --> pass pointer to cuda kernel --> unmap pointer
octproz_project/octproz/src/processing.cpp:bool Processing::waitForCudaOpenGlInteropReady(int interval, int timeout){
octproz_project/octproz/src/processing.cpp:	while (!this->isCudaOpenGlInteropReady()) {
octproz_project/octproz/src/processing.cpp:			emit error(tr("Cuda-OpenGL Interoperability initialization timeout."));
octproz_project/octproz/src/processing.cpp:bool Processing::isCudaOpenGlInteropReady(){
octproz_project/octproz/src/processing.cpp:	return (this->bscanGlBufferRegisteredWithCuda || !this->octParams->bscanViewEnabled) &&
octproz_project/octproz/src/processing.cpp:		(this->enfaceGlBufferRegisteredWithCuda || !this->octParams->enFaceViewEnabled) &&
octproz_project/octproz/src/processing.cpp:		(this->volumeGlBufferRegisteredWithCuda || !this->octParams->volumeViewEnabled) &&
octproz_project/octproz/src/processing.cpp:		emit info(tr("GPU processing initialization..."));
octproz_project/octproz/src/processing.cpp:		this->initCudaOpenGlInterop();
octproz_project/octproz/src/processing.cpp:		bool gpuInitialized = initializeCuda(h_buffer1, h_buffer2, this->octParams);
octproz_project/octproz/src/processing.cpp:		if(!gpuInitialized){
octproz_project/octproz/src/processing.cpp:			emit error(tr("GPU buffer initialization failed."));
octproz_project/octproz/src/processing.cpp:			this->enableGpu2HostStreaming(this->octParams->streamToHost);
octproz_project/octproz/src/processing.cpp:		emit info(tr("GPU processing initialized."));
octproz_project/octproz/src/processing.cpp:					//make OpenGL context current and process raw data on GPU
octproz_project/octproz/src/processing.cpp:					octCudaPipeline(buffer->bufferArray[bufferPos]); //todo: wrap cuda functions in extra class such that oct processing implementations with other gpu/multi threading frameworks (OpenCL, OpenMP, C++ AMP) can be used interchangeably
octproz_project/octproz/src/processing.cpp:					//gpu 2 host-ram streaming
octproz_project/octproz/src/processing.cpp:						this->enableGpu2HostStreaming(this->octParams->streamToHost);
octproz_project/octproz/src/processing.cpp:			this->enableGpu2HostStreaming(false);
octproz_project/octproz/src/processing.cpp:		cleanupCuda();
octproz_project/octproz/src/processing.cpp:void Processing::slot_registerBscanOpenGLbufferWithCuda(unsigned int bufferId){
octproz_project/octproz/src/processing.cpp:		this->bscanGlBufferRegisteredWithCuda = cuda_registerGlBufferBscan(bufferId);
octproz_project/octproz/src/processing.cpp:void Processing::slot_registerEnFaceViewOpenGLbufferWithCuda(unsigned int bufferId){
octproz_project/octproz/src/processing.cpp:		this->enfaceGlBufferRegisteredWithCuda = cuda_registerGlBufferEnFaceView(bufferId);
octproz_project/octproz/src/processing.cpp:void Processing::slot_registerVolumeViewOpenGLbufferWithCuda(unsigned int bufferId){
octproz_project/octproz/src/processing.cpp:		this->volumeGlBufferRegisteredWithCuda = cuda_registerGlBufferVolumeView(bufferId);
octproz_project/octproz/src/processing.cpp:void Processing::enableGpu2HostStreaming(bool enableStreaming) {
octproz_project/octproz/src/processing.cpp:		emit info(tr("GPU to Host-Ram Streaming enabled."));
octproz_project/octproz/src/processing.cpp:		emit info(tr("GPU to Host-Ram Streaming disabled."));
octproz_project/octproz/src/processing.cpp:	cuda_registerStreamingBuffers(h_streamingBuffer1, h_streamingBuffer2, bytesPerBuffer);
octproz_project/octproz/src/processing.cpp:	cuda_unregisterStreamingBuffers();
octproz_project/octproz/src/kernels.h://CUDA FFT
octproz_project/octproz/src/kernels.h://include gl headers before cuda_gl_interop.h for aarch64 (jetson nano)
octproz_project/octproz/src/kernels.h://CUDA Runtime, Interop, and includes
octproz_project/octproz/src/kernels.h:#include <cuda_gl_interop.h>
octproz_project/octproz/src/kernels.h:#include <cuda_profiler_api.h>
octproz_project/octproz/src/kernels.h://Helper functions, CUDA utilities
octproz_project/octproz/src/kernels.h:#include <helper_cuda.h>
octproz_project/octproz/src/kernels.h:#include <cuda_fp16.h>
octproz_project/octproz/src/kernels.h:#include "gpu2hostnotifier.h"
octproz_project/octproz/src/kernels.h://cuda_code.cu
octproz_project/octproz/src/kernels.h:extern "C" bool initializeCuda(void* h_buffer1, void* h_buffer2, OctAlgorithmParameters* dispParameters);
octproz_project/octproz/src/kernels.h:extern "C" void octCudaPipeline(void* h_inputSignal);
octproz_project/octproz/src/kernels.h:extern "C" void cleanupCuda();
octproz_project/octproz/src/kernels.h:extern "C" void freeCudaMem(void** data);
octproz_project/octproz/src/kernels.h:extern "C" void cuda_registerStreamingBuffers(void* h_streamingBuffer1, void* h_streamingBuffer2, size_t bytesPerBuffer);
octproz_project/octproz/src/kernels.h:extern "C" void cuda_unregisterStreamingBuffers();
octproz_project/octproz/src/kernels.h:extern "C" bool cuda_registerGlBufferBscan(GLuint buf);
octproz_project/octproz/src/kernels.h:extern "C" bool cuda_registerGlBufferEnFaceView(GLuint buf);
octproz_project/octproz/src/kernels.h:extern "C" bool cuda_registerGlBufferVolumeView(GLuint buf);
octproz_project/octproz/src/kernels.h:extern void* cuda_map(cudaGraphicsResource* res, cudaStream_t stream);
octproz_project/octproz/src/kernels.h:extern void cuda_unmap(cudaGraphicsResource* res, cudaStream_t stream);
octproz_project/octproz/src/kernels.h:extern cudaArray* cuda_map3dTexture(cudaGraphicsResource* res, cudaStream_t stream);
octproz_project/octproz/src/kernels.h:extern "C" inline void updateBscanDisplayBuffer(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction, cudaStream_t stream); ///as soon as new buffer is acquired this function is called and the display buffer gets updated
octproz_project/octproz/src/kernels.h:extern "C" inline void updateEnFaceDisplayBuffer(unsigned int frameNr, unsigned int displayFunctionFrames, int displayFunction, cudaStream_t stream); ///as soon as new buffer is acquired this function is called and the display buffer gets updated
octproz_project/octproz/src/glwindow2d.h:// CUDA Runtime, Interop  includes
octproz_project/octproz/src/glwindow2d.h:#include <cuda_runtime.h>
octproz_project/octproz/src/glwindow2d.h:#include <cuda_gl_interop.h>
octproz_project/octproz/src/glwindow2d.h:#include <cuda_profiler_api.h>
octproz_project/octproz/src/glwindow2d.h:// CUDA helper functions
octproz_project/octproz/src/glwindow2d.h:#include <helper_cuda.h>
octproz_project/octproz/src/glwindow2d.h:	void registerOpenGLBufferWithCuda();
octproz_project/octproz/src/glwindow2d.h:	void registerBufferCudaGL(unsigned int bufferId);
octproz_project/octproz/src/glwindow2d.h:	void initCudaGl();
octproz_project/octproz/src/gpu2hostnotifier.h:#ifndef GPU2HOSTNOTIFIER_H
octproz_project/octproz/src/gpu2hostnotifier.h:#define GPU2HOSTNOTIFIER_H
octproz_project/octproz/src/gpu2hostnotifier.h:#include "cuda_runtime_api.h"
octproz_project/octproz/src/gpu2hostnotifier.h:#include "helper_cuda.h"
octproz_project/octproz/src/gpu2hostnotifier.h:class Gpu2HostNotifier : public QObject
octproz_project/octproz/src/gpu2hostnotifier.h:	static Gpu2HostNotifier* getInstance(QObject* parent = nullptr);
octproz_project/octproz/src/gpu2hostnotifier.h:	~Gpu2HostNotifier();
octproz_project/octproz/src/gpu2hostnotifier.h:	static void CUDART_CB dh2StreamingCallback(void* currStreamingBuffer);
octproz_project/octproz/src/gpu2hostnotifier.h:	static void CUDART_CB backgroundSignalCallback(void* backgroundSignal);
octproz_project/octproz/src/gpu2hostnotifier.h:	static void CUDART_CB bscanDisblayBufferReadySignalCallback(void* data);
octproz_project/octproz/src/gpu2hostnotifier.h:	static void CUDART_CB enfaceDisplayBufferReadySignalCallback(void* data);
octproz_project/octproz/src/gpu2hostnotifier.h:	static void CUDART_CB volumeDisblayBufferReadySignalCallback(void* data);
octproz_project/octproz/src/gpu2hostnotifier.h:	Gpu2HostNotifier(QObject *parent);
octproz_project/octproz/src/gpu2hostnotifier.h:	static Gpu2HostNotifier* gpu2hostNotifier;
octproz_project/octproz/src/gpu2hostnotifier.h:	void newGpuDataAvailible(void* rawBuffer, unsigned bitDepth, unsigned int samplesPerLine, unsigned int linesPerFrame, unsigned int framesPerBuffer, unsigned int buffersPerVolume, unsigned int currentBufferNr);
octproz_project/octproz/src/gpu2hostnotifier.h:#endif //GPU2HOSTNOTIFIER_H
octproz_project/octproz_plugins/octproz_virtual_oct_system/src/virtualoctsystemsettingsdialog.ui:          <string> For processing buffer by buffer is copied to GPU memory and all a-scans of each buffer are processed in parallel. The size of the buffer affects the processing speed. The optimal value depends on the GPU used.</string>
octproz_project/octproz_plugins/octproz_virtual_oct_system/src/virtualoctsystemsettingsdialog.ui:          <string>Buffers per volume. A volume can consist of one or more buffers. For processing buffer by buffer is copied to GPU memory and all a-scans of each buffer are processed in parallel.</string>

```
