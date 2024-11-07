# https://github.com/mp3guy/ElasticFusion

```console
README.md:Ubuntu 22.04 on Xorg, NVIDIA drivers 510.73.05, CUDA driver 11.6, CUDA toolkit 11.5 (essentially whatever is in the Ubuntu repos).
README.md:A [very fast nVidia GPU (3.5TFLOPS+)](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units#GeForce_900_Series), and a fast CPU (something like an i7). If you want to use a non-nVidia GPU you can rewrite the tracking code or substitute it with something else, as the rest of the pipeline is actually written in the OpenGL Shading Language. 
README.md:***I have a nice new laptop with a good GPU but it's still slow***
README.md:If your laptop is running on battery power the GPU will throttle down to save power, so that's unlikely to work (as an aside, [Kintinuous](https://github.com/mp3guy/Kintinuous) will run at 30Hz on a modern laptop on battery power these days). You can try disabling SO(3) pre-alignment, enabling fast odometry, only using either ICP or RGB tracking and not both, running in open loop mode or disabling the tracking pyramid. All of these will cost you accuracy. 
README.md:No. The system relies on an extremely fast and tight coupling between the mapping and tracking on the GPU, which I don't believe ROS supports natively in terms of message passing. 
CMakeLists.txt:find_package(CUDA REQUIRED)
CMakeLists.txt:include_directories(${CUDA_INCLUDE_DIRS})
CMakeLists.txt:                      ${CUDA_LIBRARIES}
MainController.cpp:    for (std::map<std::string, GPUTexture*>::const_iterator it = eFusion->getTextures().begin();
Core/Ferns.cpp:    GPUTexture* imageTexture,
Core/Ferns.cpp:    GPUTexture* vertexTexture,
Core/Ferns.cpp:    GPUTexture* normalTexture,
Core/Ferns.cpp:    GPUTexture* vertexTexture,
Core/Ferns.cpp:    GPUTexture* normalTexture,
Core/Ferns.cpp:    GPUTexture* imageTexture,
Core/IndexMap.h:#include "GPUTexture.h"
Core/IndexMap.h:  GPUTexture* indexTex() {
Core/IndexMap.h:  GPUTexture* vertConfTex() {
Core/IndexMap.h:  GPUTexture* colorTimeTex() {
Core/IndexMap.h:  GPUTexture* normalRadTex() {
Core/IndexMap.h:  GPUTexture* drawTex() {
Core/IndexMap.h:  GPUTexture* depthTex() {
Core/IndexMap.h:  GPUTexture* imageTex() {
Core/IndexMap.h:  GPUTexture* vertexTex() {
Core/IndexMap.h:  GPUTexture* normalTex() {
Core/IndexMap.h:  GPUTexture* timeTex() {
Core/IndexMap.h:  GPUTexture* oldImageTex() {
Core/IndexMap.h:  GPUTexture* oldVertexTex() {
Core/IndexMap.h:  GPUTexture* oldNormalTex() {
Core/IndexMap.h:  GPUTexture* oldTimeTex() {
Core/IndexMap.h:  GPUTexture* colorInfoTex() {
Core/IndexMap.h:  GPUTexture* vertexInfoTex() {
Core/IndexMap.h:  GPUTexture* normalInfoTex() {
Core/IndexMap.h:  GPUTexture indexTexture;
Core/IndexMap.h:  GPUTexture vertConfTexture;
Core/IndexMap.h:  GPUTexture colorTimeTexture;
Core/IndexMap.h:  GPUTexture normalRadTexture;
Core/IndexMap.h:  GPUTexture drawTexture;
Core/IndexMap.h:  GPUTexture depthTexture;
Core/IndexMap.h:  GPUTexture imageTexture;
Core/IndexMap.h:  GPUTexture vertexTexture;
Core/IndexMap.h:  GPUTexture normalTexture;
Core/IndexMap.h:  GPUTexture timeTexture;
Core/IndexMap.h:  GPUTexture oldImageTexture;
Core/IndexMap.h:  GPUTexture oldVertexTexture;
Core/IndexMap.h:  GPUTexture oldNormalTexture;
Core/IndexMap.h:  GPUTexture oldTimeTexture;
Core/IndexMap.h:  GPUTexture colorInfoTexture;
Core/IndexMap.h:  GPUTexture vertexInfoTexture;
Core/IndexMap.h:  GPUTexture normalInfoTexture;
Core/GPUTexture.h:#ifndef GPUTEXTURE_H_
Core/GPUTexture.h:#define GPUTEXTURE_H_
Core/GPUTexture.h:#include <cuda_gl_interop.h>
Core/GPUTexture.h:#include <cuda_runtime_api.h>
Core/GPUTexture.h:class GPUTexture {
Core/GPUTexture.h:  GPUTexture(
Core/GPUTexture.h:  virtual ~GPUTexture();
Core/GPUTexture.h:  cudaGraphicsResource* cudaRes;
Core/GPUTexture.h:  GPUTexture()
Core/GPUTexture.h:        cudaRes(0),
Core/GPUTexture.h:#endif /* GPUTEXTURE_H_ */
Core/ElasticFusion.h:#include <pangolin/gl/glcuda.h>
Core/ElasticFusion.h:  std::map<std::string, GPUTexture*>& getTextures();
Core/ElasticFusion.h:   * (this is stored under textures[GPUTexture::DEPTH_NORM]
Core/ElasticFusion.h:  std::map<std::string, GPUTexture*> textures;
Core/CMakeLists.txt:include_directories(${CUDA_INCLUDE_DIRS})
Core/CMakeLists.txt:file(GLOB cuda Cuda/*.cpp Cuda/*.h Cuda/*.cu Cuda/*.cuh)
Core/CMakeLists.txt:file(GLOB containers Cuda/containers/*.cpp Cuda/containers/*.h Cuda/containers/*.cu Cuda/containers/*.cuh)
Core/CMakeLists.txt:set(CUDA_ARCH_BIN "" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
Core/CMakeLists.txt:set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")              
Core/CMakeLists.txt:include("${CMAKE_MODULE_PATH}/CudaDetect.cmake")
Core/CMakeLists.txt:detect_installed_gpus(CUDA_NVCC_ARCHS)
Core/CMakeLists.txt:foreach(NVCC_ARCH IN LISTS CUDA_NVCC_ARCHS)
Core/CMakeLists.txt:    list(APPEND CUDA_ARCH_BIN "${NVCC_ARCH} ")
Core/CMakeLists.txt:include("${CMAKE_MODULE_PATH}/CudaComputeTargetFlags.cmake")
Core/CMakeLists.txt:set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;--expt-relaxed-constexpr;--disable-warnings")
Core/CMakeLists.txt:CUDA_COMPILE(cuda_objs ${cuda})
Core/CMakeLists.txt:            ${cuda} 
Core/CMakeLists.txt:            ${cuda_objs} 
Core/CMakeLists.txt:                      ${CUDA_LIBRARIES}
Core/Shaders/FillIn.h:#include "../GPUTexture.h"
Core/Shaders/FillIn.h:  void image(GPUTexture* existingRgb, GPUTexture* rawRgb, bool passthrough);
Core/Shaders/FillIn.h:  void vertex(GPUTexture* existingVertex, GPUTexture* rawDepth, bool passthrough);
Core/Shaders/FillIn.h:  void normal(GPUTexture* existingNormal, GPUTexture* rawDepth, bool passthrough);
Core/Shaders/FillIn.h:  GPUTexture imageTexture;
Core/Shaders/FillIn.h:  GPUTexture vertexTexture;
Core/Shaders/FillIn.h:  GPUTexture normalTexture;
Core/Shaders/FillIn.cpp:void FillIn::image(GPUTexture* existingRgb, GPUTexture* rawRgb, bool passthrough) {
Core/Shaders/FillIn.cpp:void FillIn::vertex(GPUTexture* existingVertex, GPUTexture* rawDepth, bool passthrough) {
Core/Shaders/FillIn.cpp:void FillIn::normal(GPUTexture* existingNormal, GPUTexture* rawDepth, bool passthrough) {
Core/Shaders/Resize.h:#include "../GPUTexture.h"
Core/Shaders/Resize.h:  void image(GPUTexture* source, Img<Eigen::Matrix<uint8_t, 3, 1>>& dest);
Core/Shaders/Resize.h:  void vertex(GPUTexture* source, Img<Eigen::Vector4f>& dest);
Core/Shaders/Resize.h:  void time(GPUTexture* source, Img<uint16_t>& dest);
Core/Shaders/Resize.h:  GPUTexture imageTexture;
Core/Shaders/Resize.h:  GPUTexture vertexTexture;
Core/Shaders/Resize.h:  GPUTexture timeTexture;
Core/Shaders/Resize.cpp:void Resize::image(GPUTexture* source, Img<Eigen::Matrix<uint8_t, 3, 1>>& dest) {
Core/Shaders/Resize.cpp:void Resize::vertex(GPUTexture* source, Img<Eigen::Vector4f>& dest) {
Core/Shaders/Resize.cpp:void Resize::time(GPUTexture* source, Img<uint16_t>& dest) {
Core/ElasticFusion.cpp:  for (std::map<std::string, GPUTexture*>::iterator it = textures.begin(); it != textures.end();
Core/ElasticFusion.cpp:  textures[GPUTexture::RGB] = new GPUTexture(
Core/ElasticFusion.cpp:  textures[GPUTexture::DEPTH_RAW] = new GPUTexture(
Core/ElasticFusion.cpp:  textures[GPUTexture::DEPTH_FILTERED] = new GPUTexture(
Core/ElasticFusion.cpp:  textures[GPUTexture::DEPTH_METRIC] = new GPUTexture(
Core/ElasticFusion.cpp:  textures[GPUTexture::DEPTH_METRIC_FILTERED] = new GPUTexture(
Core/ElasticFusion.cpp:  textures[GPUTexture::DEPTH_NORM] = new GPUTexture(
Core/ElasticFusion.cpp:      textures[GPUTexture::DEPTH_NORM]->texture);
Core/ElasticFusion.cpp:      textures[GPUTexture::DEPTH_FILTERED]->texture);
Core/ElasticFusion.cpp:      textures[GPUTexture::DEPTH_METRIC]->texture);
Core/ElasticFusion.cpp:      textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture);
Core/ElasticFusion.cpp:      textures[GPUTexture::RGB]->texture,
Core/ElasticFusion.cpp:      textures[GPUTexture::DEPTH_METRIC]->texture,
Core/ElasticFusion.cpp:      textures[GPUTexture::RGB]->texture,
Core/ElasticFusion.cpp:      textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture,
Core/ElasticFusion.cpp:  textures[GPUTexture::DEPTH_RAW]->texture->Upload(
Core/ElasticFusion.cpp:  textures[GPUTexture::RGB]->texture->Upload(rgb, GL_RGB, GL_UNSIGNED_BYTE);
Core/ElasticFusion.cpp:    frameToModel.initFirstRGB(textures[GPUTexture::RGB]);
Core/ElasticFusion.cpp:      frameToModel.initICP(textures[GPUTexture::DEPTH_FILTERED], maxDepthProcessed);
Core/ElasticFusion.cpp:      frameToModel.initRGB(textures[GPUTexture::RGB]);
Core/ElasticFusion.cpp:          textures[GPUTexture::RGB],
Core/ElasticFusion.cpp:          textures[GPUTexture::DEPTH_METRIC],
Core/ElasticFusion.cpp:          textures[GPUTexture::DEPTH_METRIC_FILTERED],
Core/ElasticFusion.cpp:  fillIn.vertex(indexMap.vertexTex(), textures[GPUTexture::DEPTH_FILTERED], lost);
Core/ElasticFusion.cpp:  fillIn.normal(indexMap.normalTex(), textures[GPUTexture::DEPTH_FILTERED], lost);
Core/ElasticFusion.cpp:  fillIn.image(indexMap.imageTex(), textures[GPUTexture::RGB], lost || frameToFrameRGB);
Core/ElasticFusion.cpp:  computePacks[ComputePack::METRIC]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
Core/ElasticFusion.cpp:      textures[GPUTexture::DEPTH_FILTERED]->texture, &uniforms);
Core/ElasticFusion.cpp:  computePacks[ComputePack::FILTER]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
Core/ElasticFusion.cpp:  computePacks[ComputePack::NORM]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
Core/ElasticFusion.cpp:std::map<std::string, GPUTexture*>& ElasticFusion::getTextures() {
Core/GlobalModel.cpp:    GPUTexture* rgb,
Core/GlobalModel.cpp:    GPUTexture* depthRaw,
Core/GlobalModel.cpp:    GPUTexture* depthFiltered,
Core/GlobalModel.cpp:    GPUTexture* indexMap,
Core/GlobalModel.cpp:    GPUTexture* vertConfMap,
Core/GlobalModel.cpp:    GPUTexture* colorTimeMap,
Core/GlobalModel.cpp:    GPUTexture* normRadMap,
Core/GlobalModel.cpp:    GPUTexture* indexMap,
Core/GlobalModel.cpp:    GPUTexture* vertConfMap,
Core/GlobalModel.cpp:    GPUTexture* colorTimeMap,
Core/GlobalModel.cpp:    GPUTexture* normRadMap,
Core/GlobalModel.cpp:    GPUTexture* depthMap,
Core/Cuda/convenience.cuh:#ifndef CUDA_CONVENIENCE_CUH_
Core/Cuda/convenience.cuh:#define CUDA_CONVENIENCE_CUH_
Core/Cuda/convenience.cuh:#include <cuda_runtime_api.h>
Core/Cuda/convenience.cuh:static inline void cudaSafeCall(cudaError_t err) {
Core/Cuda/convenience.cuh:  if (cudaSuccess != err) {
Core/Cuda/convenience.cuh:    std::cout << "Error: " << cudaGetErrorString(err) << ": " << __FILE__ << ":" << __LINE__
Core/Cuda/convenience.cuh:#endif /* CUDA_CONVENIENCE_CUH_ */
Core/Cuda/operators.cuh:#ifndef CUDA_OPERATORS_CUH_
Core/Cuda/operators.cuh:#define CUDA_OPERATORS_CUH_
Core/Cuda/operators.cuh:#endif /* CUDA_OPERATORS_CUH_ */
Core/Cuda/cudafuncs.cu:#include "cudafuncs.cuh"
Core/Cuda/cudafuncs.cu:texture<uchar4, 2, cudaReadModeElementType> uchar4Tex;
Core/Cuda/cudafuncs.cu:texture<float4, 2, cudaReadModeElementType> float4Tex0;
Core/Cuda/cudafuncs.cu:texture<float4, 2, cudaReadModeElementType> float4Tex1;
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaGetLastError());
Core/Cuda/cudafuncs.cu:      vmap.ptr(v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaGetLastError());
Core/Cuda/cudafuncs.cu:    nmap.ptr(v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
Core/Cuda/cudafuncs.cu:    nmap.ptr(v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaGetLastError());
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaGetLastError());
Core/Cuda/cudafuncs.cu:    const cudaArray_t& vmap_src,
Core/Cuda/cudafuncs.cu:    const cudaArray_t& nmap_src,
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaBindTextureToArray(float4Tex0, vmap_src));
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaBindTextureToArray(float4Tex1, nmap_src));
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaGetLastError());
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaUnbindTexture(float4Tex0));
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaUnbindTexture(float4Tex1));
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaGetLastError());
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaGetLastError());
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaDeviceSynchronize());
Core/Cuda/cudafuncs.cu:  float* gauss_cuda;
Core/Cuda/cudafuncs.cu:  cudaMalloc((void**)&gauss_cuda, sizeof(float) * 25);
Core/Cuda/cudafuncs.cu:  cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25, cudaMemcpyHostToDevice);
Core/Cuda/cudafuncs.cu:  pyrDownKernelGaussF<<<grid, block>>>(src, dst, gauss_cuda);
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaGetLastError());
Core/Cuda/cudafuncs.cu:  cudaFree(gauss_cuda);
Core/Cuda/cudafuncs.cu:  float* gauss_cuda;
Core/Cuda/cudafuncs.cu:  cudaMalloc((void**)&gauss_cuda, sizeof(float) * 25);
Core/Cuda/cudafuncs.cu:  cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25, cudaMemcpyHostToDevice);
Core/Cuda/cudafuncs.cu:  pyrDownKernelIntensityGauss<<<grid, block>>>(src, dst, gauss_cuda);
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaGetLastError());
Core/Cuda/cudafuncs.cu:  cudaFree(gauss_cuda);
Core/Cuda/cudafuncs.cu:  dst.ptr(y)[x] = z > cutOff || z <= 0 ? __int_as_float(0x7fffffff) /*CUDART_NAN_F*/ : z;
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaGetLastError());
Core/Cuda/cudafuncs.cu:void imageBGRToIntensity(cudaArray_t cuArr, DeviceArray2D<uint8_t>& dst) {
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaBindTextureToArray(uchar4Tex, cuArr));
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaGetLastError());
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaUnbindTexture(uchar4Tex));
Core/Cuda/cudafuncs.cu:    cudaMemcpyToSymbol(gsobel_x3x3, gsx3x3, sizeof(float) * 9);
Core/Cuda/cudafuncs.cu:    cudaMemcpyToSymbol(gsobel_y3x3, gsy3x3, sizeof(float) * 9);
Core/Cuda/cudafuncs.cu:    cudaSafeCall(cudaGetLastError());
Core/Cuda/cudafuncs.cu:    cudaSafeCall(cudaDeviceSynchronize());
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaGetLastError());
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaDeviceSynchronize());
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaGetLastError());
Core/Cuda/cudafuncs.cu:  cudaSafeCall(cudaDeviceSynchronize());
Core/Cuda/reduce.cu:#include "cudafuncs.cuh"
Core/Cuda/reduce.cu:  cudaSafeCall(cudaGetLastError());
Core/Cuda/reduce.cu:  cudaSafeCall(cudaDeviceSynchronize());
Core/Cuda/reduce.cu:  cudaSafeCall(cudaGetLastError());
Core/Cuda/reduce.cu:  cudaSafeCall(cudaDeviceSynchronize());
Core/Cuda/reduce.cu:  cudaMalloc(&out, sizeof(int2));
Core/Cuda/reduce.cu:  cudaMemcpy(out, &out_host, sizeof(int2), cudaMemcpyHostToDevice);
Core/Cuda/reduce.cu:  cudaSafeCall(cudaGetLastError());
Core/Cuda/reduce.cu:  cudaSafeCall(cudaDeviceSynchronize());
Core/Cuda/reduce.cu:  cudaMemcpy(&out_host, out, sizeof(int2), cudaMemcpyDeviceToHost);
Core/Cuda/reduce.cu:  cudaFree(out);
Core/Cuda/reduce.cu:  cudaSafeCall(cudaGetLastError());
Core/Cuda/reduce.cu:  cudaSafeCall(cudaDeviceSynchronize());
Core/Cuda/cudafuncs.cuh:#ifndef CUDA_CUDAFUNCS_CUH_
Core/Cuda/cudafuncs.cuh:#define CUDA_CUDAFUNCS_CUH_
Core/Cuda/cudafuncs.cuh:#include <cuda_runtime.h>
Core/Cuda/cudafuncs.cuh:    const cudaArray_t& vmap_src,
Core/Cuda/cudafuncs.cuh:    const cudaArray_t& nmap_src,
Core/Cuda/cudafuncs.cuh:void imageBGRToIntensity(cudaArray_t cuArr, DeviceArray2D<uint8_t>& dst);
Core/Cuda/cudafuncs.cuh:#endif /* CUDA_CUDAFUNCS_CUH_ */
Core/Cuda/containers/kernel_containers.hpp:#if defined(__CUDACC__)
Core/Cuda/containers/kernel_containers.hpp:    #define GPU_HOST_DEVICE__ __host__ __device__ __forceinline__
Core/Cuda/containers/kernel_containers.hpp:    #define GPU_HOST_DEVICE__
Core/Cuda/containers/kernel_containers.hpp:    GPU_HOST_DEVICE__ DevPtr() : data(0) {}
Core/Cuda/containers/kernel_containers.hpp:    GPU_HOST_DEVICE__ DevPtr(T* data_arg) : data(data_arg) {}
Core/Cuda/containers/kernel_containers.hpp:    GPU_HOST_DEVICE__ size_t elemSize() const { return elem_size; }
Core/Cuda/containers/kernel_containers.hpp:    GPU_HOST_DEVICE__ operator       T*()       { return data; }
Core/Cuda/containers/kernel_containers.hpp:    GPU_HOST_DEVICE__ operator const T*() const { return data; }
Core/Cuda/containers/kernel_containers.hpp:    GPU_HOST_DEVICE__ PtrSz() : size(0) {}
Core/Cuda/containers/kernel_containers.hpp:    GPU_HOST_DEVICE__ PtrSz(T* data_arg, size_t size_arg) : DevPtr<T>(data_arg), size(size_arg) {}
Core/Cuda/containers/kernel_containers.hpp:    GPU_HOST_DEVICE__ PtrStep() : step(0) {}
Core/Cuda/containers/kernel_containers.hpp:    GPU_HOST_DEVICE__ PtrStep(T* data_arg, size_t step_arg) : DevPtr<T>(data_arg), step(step_arg) {}
Core/Cuda/containers/kernel_containers.hpp:    GPU_HOST_DEVICE__       T* ptr(int y = 0)       { return (      T*)( (      char*)DevPtr<T>::data + y * step); }
Core/Cuda/containers/kernel_containers.hpp:    GPU_HOST_DEVICE__ const T* ptr(int y = 0) const { return (const T*)( (const char*)DevPtr<T>::data + y * step); }
Core/Cuda/containers/kernel_containers.hpp:    GPU_HOST_DEVICE__ PtrStepSz() : cols(0), rows(0) {}
Core/Cuda/containers/kernel_containers.hpp:    GPU_HOST_DEVICE__ PtrStepSz(int rows_arg, int cols_arg, T* data_arg, size_t step_arg)
Core/Cuda/containers/device_memory.cpp:#include "cuda_runtime_api.h"
Core/Cuda/containers/device_memory.cpp:    cudaSafeCall(cudaMalloc(&data_, sizeBytes_));
Core/Cuda/containers/device_memory.cpp:    cudaSafeCall(cudaMemcpy(other.data_, data_, sizeBytes_, cudaMemcpyDeviceToDevice));
Core/Cuda/containers/device_memory.cpp:    cudaSafeCall(cudaDeviceSynchronize());
Core/Cuda/containers/device_memory.cpp:    cudaSafeCall(cudaFree(data_));
Core/Cuda/containers/device_memory.cpp:  cudaSafeCall(cudaMemcpy(data_, host_ptr_arg, sizeBytes_, cudaMemcpyHostToDevice));
Core/Cuda/containers/device_memory.cpp:  cudaSafeCall(cudaDeviceSynchronize());
Core/Cuda/containers/device_memory.cpp:  cudaSafeCall(cudaMemcpy(host_ptr_arg, data_, sizeBytes_, cudaMemcpyDeviceToHost));
Core/Cuda/containers/device_memory.cpp:  cudaSafeCall(cudaDeviceSynchronize());
Core/Cuda/containers/device_memory.cpp:    cudaSafeCall(cudaMallocPitch((void**)&data_, &step_, colsBytes_, rows_));
Core/Cuda/containers/device_memory.cpp:    cudaSafeCall(cudaFree(data_));
Core/Cuda/containers/device_memory.cpp:    cudaSafeCall(cudaMemcpy2D(
Core/Cuda/containers/device_memory.cpp:        other.data_, other.step_, data_, step_, colsBytes_, rows_, cudaMemcpyDeviceToDevice));
Core/Cuda/containers/device_memory.cpp:    cudaSafeCall(cudaDeviceSynchronize());
Core/Cuda/containers/device_memory.cpp:  cudaSafeCall(cudaMemcpy2D(
Core/Cuda/containers/device_memory.cpp:      data_, step_, host_ptr_arg, host_step_arg, colsBytes_, rows_, cudaMemcpyHostToDevice));
Core/Cuda/containers/device_memory.cpp:  cudaSafeCall(cudaMemcpy2D(
Core/Cuda/containers/device_memory.cpp:      host_ptr_arg, host_step_arg, data_, step_, colsBytes_, rows_, cudaMemcpyDeviceToHost));
Core/Cuda/containers/device_array.hpp:  * \note Typed container for GPU memory with reference counting.
Core/Cuda/containers/device_array.hpp:        /** \brief Allocates internal buffer in GPU memory
Core/Cuda/containers/device_array.hpp:        /** \brief Allocates internal buffer in GPU memory. If internal buffer was created before the function recreates it with new size. If new and old sizes are equal it does nothing.
Core/Cuda/containers/device_array.hpp:        /** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is enough.
Core/Cuda/containers/device_array.hpp:        /** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is enough.
Core/Cuda/containers/device_array.hpp:        /** \brief Returns pointer for internal buffer in GPU memory. */
Core/Cuda/containers/device_array.hpp:        /** \brief Returns const pointer for internal buffer in GPU memory. */
Core/Cuda/containers/device_array.hpp:        /** \brief Returns pointer for internal buffer in GPU memory. */
Core/Cuda/containers/device_array.hpp:        /** \brief Returns const pointer for internal buffer in GPU memory. */
Core/Cuda/containers/device_array.hpp:  * \note Typed container for pitched GPU memory with reference counting.
Core/Cuda/containers/device_array.hpp:        /** \brief Allocates internal buffer in GPU memory
Core/Cuda/containers/device_array.hpp:        /** \brief Allocates internal buffer in GPU memory. If internal buffer was created before the function recreates it with new size. If new and old sizes are equal it does nothing.
Core/Cuda/containers/device_array.hpp:        /** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is enough.
Core/Cuda/containers/device_array.hpp:        /** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is enough.
Core/Cuda/containers/device_array.hpp:        /** \brief Returns pointer for internal buffer in GPU memory. */
Core/Cuda/containers/device_array.hpp:        /** \brief Returns const pointer for internal buffer in GPU memory. */
Core/Cuda/containers/device_memory.hpp:  * \note This is a BLOB container class with reference counting for GPU memory.
Core/Cuda/containers/device_memory.hpp:        /** \brief Allocates internal buffer in GPU memory
Core/Cuda/containers/device_memory.hpp:         /** \brief Allocates internal buffer in GPU memory. If internal buffer was created before the function recreates it with new size. If new and old sizes are equal it does nothing.
Core/Cuda/containers/device_memory.hpp:        /** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is enough.
Core/Cuda/containers/device_memory.hpp:        /** \brief Returns pointer for internal buffer in GPU memory. */
Core/Cuda/containers/device_memory.hpp:        /** \brief Returns constant pointer for internal buffer in GPU memory. */
Core/Cuda/containers/device_memory.hpp:  * \note This is a BLOB container class with reference counting for pitched GPU memory.
Core/Cuda/containers/device_memory.hpp:        /** \brief Allocates internal buffer in GPU memory
Core/Cuda/containers/device_memory.hpp:        /** \brief Allocates internal buffer in GPU memory. If internal buffer was created before the function recreates it with new size. If new and old sizes are equal it does nothing.
Core/Cuda/containers/device_memory.hpp:        /** \brief Uploads data to internal buffer in GPU memory. It calls create() inside to ensure that intenal buffer size is enough.
Core/Cuda/types.cuh:#ifndef CUDA_TYPES_CUH_
Core/Cuda/types.cuh:#define CUDA_TYPES_CUH_
Core/Cuda/types.cuh:#if !defined(__CUDACC__)
Core/Cuda/types.cuh:#if !defined(__CUDACC__)
Core/Cuda/types.cuh:#endif /* CUDA_TYPES_CUH_ */
Core/GlobalModel.h:#include "GPUTexture.h"
Core/GlobalModel.h:      GPUTexture* rgb,
Core/GlobalModel.h:      GPUTexture* depthRaw,
Core/GlobalModel.h:      GPUTexture* depthFiltered,
Core/GlobalModel.h:      GPUTexture* indexMap,
Core/GlobalModel.h:      GPUTexture* vertConfMap,
Core/GlobalModel.h:      GPUTexture* colorTimeMap,
Core/GlobalModel.h:      GPUTexture* normRadMap,
Core/GlobalModel.h:      GPUTexture* indexMap,
Core/GlobalModel.h:      GPUTexture* vertConfMap,
Core/GlobalModel.h:      GPUTexture* colorTimeMap,
Core/GlobalModel.h:      GPUTexture* normRadMap,
Core/GlobalModel.h:      GPUTexture* depthMap,
Core/GlobalModel.h:  GPUTexture updateMapVertsConfs;
Core/GlobalModel.h:  GPUTexture updateMapColorsTime;
Core/GlobalModel.h:  GPUTexture updateMapNormsRadii;
Core/GlobalModel.h:  GPUTexture deformationNodes;
Core/Deformation.h:#include "GPUTexture.h"
Core/GPUTexture.cpp:#include "GPUTexture.h"
Core/GPUTexture.cpp:const std::string GPUTexture::RGB = "RGB";
Core/GPUTexture.cpp:const std::string GPUTexture::DEPTH_RAW = "DEPTH";
Core/GPUTexture.cpp:const std::string GPUTexture::DEPTH_FILTERED = "DEPTH_FILTERED";
Core/GPUTexture.cpp:const std::string GPUTexture::DEPTH_METRIC = "DEPTH_METRIC";
Core/GPUTexture.cpp:const std::string GPUTexture::DEPTH_METRIC_FILTERED = "DEPTH_METRIC_FILTERED";
Core/GPUTexture.cpp:const std::string GPUTexture::DEPTH_NORM = "DEPTH_NORM";
Core/GPUTexture.cpp:GPUTexture::GPUTexture(
Core/GPUTexture.cpp:  cudaGraphicsGLRegisterImage(
Core/GPUTexture.cpp:      &cudaRes, texture->tid, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
Core/GPUTexture.cpp:GPUTexture::~GPUTexture() {
Core/GPUTexture.cpp:  if (cudaRes) {
Core/GPUTexture.cpp:    cudaGraphicsUnregisterResource(cudaRes);
Core/Ferns.h:      GPUTexture* imageTexture,
Core/Ferns.h:      GPUTexture* vertexTexture,
Core/Ferns.h:      GPUTexture* normalTexture,
Core/Ferns.h:      GPUTexture* vertexTexture,
Core/Ferns.h:      GPUTexture* normalTexture,
Core/Ferns.h:      GPUTexture* imageTexture,
Core/Ferns.h:  GPUTexture vertFern;
Core/Ferns.h:  GPUTexture vertCurrent;
Core/Ferns.h:  GPUTexture normFern;
Core/Ferns.h:  GPUTexture normCurrent;
Core/Ferns.h:  GPUTexture colorFern;
Core/Ferns.h:  GPUTexture colorCurrent;
Core/Utils/RGBDOdometry.cpp:void RGBDOdometry::initICP(GPUTexture* filteredDepth, const float depthCutoff) {
Core/Utils/RGBDOdometry.cpp:  cudaArray_t textPtr;
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsMapResources(1, &filteredDepth->cudaRes);
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsSubResourceGetMappedArray(&textPtr, filteredDepth->cudaRes, 0, 0);
Core/Utils/RGBDOdometry.cpp:  cudaMemcpy2DFromArray(
Core/Utils/RGBDOdometry.cpp:      cudaMemcpyDeviceToDevice);
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsUnmapResources(1, &filteredDepth->cudaRes);
Core/Utils/RGBDOdometry.cpp:  cudaDeviceSynchronize();
Core/Utils/RGBDOdometry.cpp:void RGBDOdometry::initICP(GPUTexture* predictedVertices, GPUTexture* predictedNormals) {
Core/Utils/RGBDOdometry.cpp:  cudaArray_t vmapPtr, nmapPtr;
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsMapResources(1, &predictedVertices->cudaRes);
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsSubResourceGetMappedArray(&vmapPtr, predictedVertices->cudaRes, 0, 0);
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsMapResources(1, &predictedNormals->cudaRes);
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsSubResourceGetMappedArray(&nmapPtr, predictedNormals->cudaRes, 0, 0);
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsUnmapResources(1, &predictedVertices->cudaRes);
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsUnmapResources(1, &predictedNormals->cudaRes);
Core/Utils/RGBDOdometry.cpp:  cudaDeviceSynchronize();
Core/Utils/RGBDOdometry.cpp:    GPUTexture* predictedVertices,
Core/Utils/RGBDOdometry.cpp:    GPUTexture* predictedNormals,
Core/Utils/RGBDOdometry.cpp:  cudaArray_t vmapPtr, nmapPtr;
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsMapResources(1, &predictedVertices->cudaRes);
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsSubResourceGetMappedArray(&vmapPtr, predictedVertices->cudaRes, 0, 0);
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsMapResources(1, &predictedNormals->cudaRes);
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsSubResourceGetMappedArray(&nmapPtr, predictedNormals->cudaRes, 0, 0);
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsUnmapResources(1, &predictedVertices->cudaRes);
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsUnmapResources(1, &predictedNormals->cudaRes);
Core/Utils/RGBDOdometry.cpp:  cudaDeviceSynchronize();
Core/Utils/RGBDOdometry.cpp:    GPUTexture* rgb,
Core/Utils/RGBDOdometry.cpp:  cudaArray_t textPtr;
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsMapResources(1, &rgb->cudaRes);
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsUnmapResources(1, &rgb->cudaRes);
Core/Utils/RGBDOdometry.cpp:  cudaDeviceSynchronize();
Core/Utils/RGBDOdometry.cpp:void RGBDOdometry::initRGBModel(GPUTexture* rgb) {
Core/Utils/RGBDOdometry.cpp:void RGBDOdometry::initRGB(GPUTexture* rgb) {
Core/Utils/RGBDOdometry.cpp:void RGBDOdometry::initFirstRGB(GPUTexture* rgb) {
Core/Utils/RGBDOdometry.cpp:  cudaArray_t textPtr;
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsMapResources(1, &rgb->cudaRes);
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);
Core/Utils/RGBDOdometry.cpp:  cudaGraphicsUnmapResources(1, &rgb->cudaRes);
Core/Utils/RGBDOdometry.h:#include "../Cuda/cudafuncs.cuh"
Core/Utils/RGBDOdometry.h:#include "../GPUTexture.h"
Core/Utils/RGBDOdometry.h:  void initICP(GPUTexture* filteredDepth, const float depthCutoff);
Core/Utils/RGBDOdometry.h:  void initICP(GPUTexture* predictedVertices, GPUTexture* predictedNormals);
Core/Utils/RGBDOdometry.h:      GPUTexture* predictedVertices,
Core/Utils/RGBDOdometry.h:      GPUTexture* predictedNormals,
Core/Utils/RGBDOdometry.h:  void initRGB(GPUTexture* rgb);
Core/Utils/RGBDOdometry.h:  void initRGBModel(GPUTexture* rgb);
Core/Utils/RGBDOdometry.h:  void initFirstRGB(GPUTexture* rgb);
Core/Utils/RGBDOdometry.h:      GPUTexture* rgb,
CMakeModules/CudaDetect.cmake:# Taken from https://github.com/BVLC/caffe/blob/master/cmake/Cuda.cmake
CMakeModules/CudaDetect.cmake:# A function for automatic detection of GPUs installed  (if autodetection is enabled)
CMakeModules/CudaDetect.cmake:#   detect_installed_gpus(out_variable)
CMakeModules/CudaDetect.cmake:function(detect_installed_gpus out_variable)
CMakeModules/CudaDetect.cmake:  if(NOT CUDA_gpu_detect_output)
CMakeModules/CudaDetect.cmake:    set(__cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)
CMakeModules/CudaDetect.cmake:      "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
CMakeModules/CudaDetect.cmake:      "    cudaDeviceProp prop;\n"
CMakeModules/CudaDetect.cmake:      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
CMakeModules/CudaDetect.cmake:    execute_process(COMMAND "${CUDA_NVCC_EXECUTABLE}" "-Wno-deprecated-gpu-targets" "--run" "${__cufile}"
CMakeModules/CudaDetect.cmake:      set(CUDA_gpu_detect_output ${__nvcc_out} CACHE INTERNAL "Returned GPU architectures from detect_gpus tool" FORCE)
CMakeModules/CudaDetect.cmake:  if(NOT CUDA_gpu_detect_output)
CMakeModules/CudaDetect.cmake:    message(STATUS "Automatic GPU detection failed. Is CUDA properly installed? .")
CMakeModules/CudaDetect.cmake:    set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
CMakeModules/CudaComputeTargetFlags.cmake:#   	include(CudaComputeTargetFlags.cmake)
CMakeModules/CudaComputeTargetFlags.cmake:MACRO(CUDA_COMPUTE_TARGET_FLAGS arch_bin arch_ptx cuda_nvcc_target_flags)
CMakeModules/CudaComputeTargetFlags.cmake:	set(cuda_computer_target_flags_temp "") 
CMakeModules/CudaComputeTargetFlags.cmake:	# Tell NVCC to add binaries for the specified GPUs
CMakeModules/CudaComputeTargetFlags.cmake:			set(cuda_computer_target_flags_temp ${cuda_computer_target_flags_temp} -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})					
CMakeModules/CudaComputeTargetFlags.cmake:			set(cuda_computer_target_flags_temp ${cuda_computer_target_flags_temp} -gencode arch=compute_${ARCH},code=sm_${ARCH})					
CMakeModules/CudaComputeTargetFlags.cmake:		set(cuda_computer_target_flags_temp ${cuda_computer_target_flags_temp} -gencode arch=compute_${ARCH},code=compute_${ARCH})				
CMakeModules/CudaComputeTargetFlags.cmake:	set(${cuda_nvcc_target_flags} ${cuda_computer_target_flags_temp})		
CMakeModules/CudaComputeTargetFlags.cmake:	set(cuda_nvcc_target_flags "")
CMakeModules/CudaComputeTargetFlags.cmake:	CUDA_COMPUTE_TARGET_FLAGS(CUDA_ARCH_BIN CUDA_ARCH_PTX cuda_nvcc_target_flags)		
CMakeModules/CudaComputeTargetFlags.cmake:	if (cuda_nvcc_target_flags)
CMakeModules/CudaComputeTargetFlags.cmake:		message(STATUS "CUDA NVCC target flags: ${cuda_nvcc_target_flags}")
CMakeModules/CudaComputeTargetFlags.cmake:		list(APPEND CUDA_NVCC_FLAGS ${cuda_nvcc_target_flags})
Tools/GUI.h:#include "../Core/GPUTexture.h"
Tools/GUI.h:#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049
Tools/GUI.h:    colorTexture = new GPUTexture(
Tools/GUI.h:    pangolin::Display(GPUTexture::RGB).SetAspect(640.0f / 480.0f);
Tools/GUI.h:    pangolin::Display(GPUTexture::DEPTH_NORM).SetAspect(640.0f / 480.0f);
Tools/GUI.h:          .AddDisplay(pangolin::Display(GPUTexture::RGB))
Tools/GUI.h:          .AddDisplay(pangolin::Display(GPUTexture::DEPTH_NORM))
Tools/GUI.h:    gpuMem = new pangolin::Var<int>("ui.GPU memory free", 0);
Tools/GUI.h:    delete gpuMem;
Tools/GUI.h:  void displayImg(const std::string& id, GPUTexture* img) {
Tools/GUI.h:    glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);
Tools/GUI.h:    gpuMem->operator=(memFree);
Tools/GUI.h:  pangolin::Var<int>* gpuMem;
Tools/GUI.h:  GPUTexture* colorTexture;

```
