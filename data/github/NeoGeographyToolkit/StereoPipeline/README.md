# https://github.com/NeoGeographyToolkit/StereoPipeline

```console
graveyard/cuda_code.cu:#include <cuda.h>
graveyard/cuda_code.cu:    cudaError_t         cudaStat; 
graveyard/cuda_code.cu:    cout << "Allocating GPU Memory.\n";
graveyard/cuda_code.cu:    cudaStat = cudaMalloc ((void **)&in, N * sizeof(in[0])); 
graveyard/cuda_code.cu:    cudaStat = cudaMalloc ((void **)&out, N * sizeof(out[0])); 
graveyard/cuda_code.cu:    cudaStat = cudaMemcpy (in, img, N * sizeof(img[0]), cudaMemcpyHostToDevice); 
graveyard/cuda_code.cu:    cudaStat = cudaMemcpy (img, out, N * sizeof(out[0]), cudaMemcpyDeviceToHost); 
graveyard/cuda_code.cu:  class CudaImageResource { 
graveyard/cuda_code.cu:    CudaImageResource(ImageFormat format): 
graveyard/cuda_code.cu:      cudaStat = cudaMalloc ((void **)&m_buffer, size * sizeof(float)); 
graveyard/cuda_code.cu:    virtual ~CudaImageResource() {
graveyard/cuda_code.cu:      cudaFree(m_buffer);
graveyard/cuda_code.cu:      cudaStat = cudaMemcpy (img, out, N * sizeof(out[0]), cudaMemcpyDeviceToHost); 
graveyard/cuda_code.cu:      cudaStat = cudaMemcpy (in, img, N * sizeof(img[0]), cudaMemcpyHostToDevice); 
graveyard/m4/ax_common_options.m4:AX_ARG_ENABLE(pkg-paths-default, [${HOME} ${HOME}/local /sw /opt /opt/local /usr/local /usr/X11R6 /usr /usr/local/cuda], [none], [Whether to use a built-in search path])
graveyard/cuda_test.cc:#include <cuda_code.h>
graveyard/cuda_test.cc:  // Call out to our CUDA code
graveyard/HRSC/HRSC.h:#include <vw/Camera/OrbitingPushbroomModel.h>
graveyard/MOC/Ephemeris.h:#include <vw/Camera/OrbitingPushbroomModel.h>
graveyard/MOC/Metadata.cc:#include <vw/Camera/OrbitingPushbroomModel.h>
graveyard/MOC/Metadata.cc:  return new vw::camera::OrbitingPushbroomModel( rows(), // number of lines
graveyard/MOC/Ephemeris.cc:#include <vw/Camera/OrbitingPushbroomModel.h>
graveyard/MOC/Metadata.h:#include <vw/Camera/OrbitingPushbroomModel.h>
thirdparty/gtest/include/gtest/gtest_ASP.h:// implementation.  NVIDIA's CUDA NVCC compiler pretends to be GCC by
thirdparty/gtest/include/gtest/gtest_ASP.h:# if (defined(__GNUC__) && !defined(__CUDACC__) && (GTEST_GCC_VER_ >= 40000)) \
src/asp/Core/SoftwareRenderer.cc:// identical to NVIDIA's OpenGL implementation OpenGL version. This
src/asp/Core/SoftwareRenderer.cc:// (NVIDIA may use the 0.5 pixel offset). We also may be using a

```
