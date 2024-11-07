# https://github.com/PMBio/peer

```console
External/googletest-read-only/include/gtest/internal/gtest-port.h:// implementation.  NVIDIA's CUDA NVCC compiler pretends to be GCC by
External/googletest-read-only/include/gtest/internal/gtest-port.h:#if (defined(__GNUC__) && !defined(__CUDACC__) && (GTEST_GCC_VER_ >= 40000)) \
External/EigenOld/demos/opengl/gpuhelper.cpp:#include "gpuhelper.h"
External/EigenOld/demos/opengl/gpuhelper.cpp:GpuHelper gpu;
External/EigenOld/demos/opengl/gpuhelper.cpp:GpuHelper::GpuHelper()
External/EigenOld/demos/opengl/gpuhelper.cpp:GpuHelper::~GpuHelper()
External/EigenOld/demos/opengl/gpuhelper.cpp:void GpuHelper::pushProjectionMode2D(ProjectionMode2D pm)
External/EigenOld/demos/opengl/gpuhelper.cpp:void GpuHelper::popProjectionMode2D(void)
External/EigenOld/demos/opengl/gpuhelper.cpp:void GpuHelper::drawVector(const Vector3f& position, const Vector3f& vec, const Color& color, float aspect /* = 50.*/)
External/EigenOld/demos/opengl/gpuhelper.cpp:void GpuHelper::drawVectorBox(const Vector3f& position, const Vector3f& vec, const Color& color, float aspect)
External/EigenOld/demos/opengl/gpuhelper.cpp:void GpuHelper::drawUnitCube(void)
External/EigenOld/demos/opengl/gpuhelper.cpp:void GpuHelper::drawUnitSphere(int level)
External/EigenOld/demos/opengl/quaternion_demo.h:#include "gpuhelper.h"
External/EigenOld/demos/opengl/CMakeLists.txt:set(quaternion_demo_SRCS  gpuhelper.cpp icosphere.cpp camera.cpp trackball.cpp quaternion_demo.cpp)
External/EigenOld/demos/opengl/camera.cpp:#include "gpuhelper.h"
External/EigenOld/demos/opengl/camera.cpp:  gpu.loadMatrix(projectionMatrix(),GL_PROJECTION);
External/EigenOld/demos/opengl/camera.cpp:  gpu.loadMatrix(viewMatrix().matrix(),GL_MODELVIEW);
External/EigenOld/demos/opengl/gpuhelper.h:#ifndef EIGEN_GPUHELPER_H
External/EigenOld/demos/opengl/gpuhelper.h:#define EIGEN_GPUHELPER_H
External/EigenOld/demos/opengl/gpuhelper.h:class GpuHelper
External/EigenOld/demos/opengl/gpuhelper.h:    GpuHelper();
External/EigenOld/demos/opengl/gpuhelper.h:    ~GpuHelper();
External/EigenOld/demos/opengl/gpuhelper.h:extern GpuHelper gpu;
External/EigenOld/demos/opengl/gpuhelper.h:inline void GpuHelper::setMatrixTarget(GLenum matrixTarget)
External/EigenOld/demos/opengl/gpuhelper.h:void GpuHelper::multMatrix(const Matrix<Scalar,4,4, _Flags, 4,4>& mat, GLenum matrixTarget)
External/EigenOld/demos/opengl/gpuhelper.h:void GpuHelper::loadMatrix(
External/EigenOld/demos/opengl/gpuhelper.h:void GpuHelper::loadMatrix(const Eigen::Matrix<Scalar,4,4, _Flags, 4,4>& mat, GLenum matrixTarget)
External/EigenOld/demos/opengl/gpuhelper.h:inline void GpuHelper::pushMatrix(GLenum matrixTarget)
External/EigenOld/demos/opengl/gpuhelper.h:inline void GpuHelper::pushMatrix(const Matrix<Scalar,4,4, _Flags, 4,4>& mat, GLenum matrixTarget)
External/EigenOld/demos/opengl/gpuhelper.h:void GpuHelper::pushMatrix(
External/EigenOld/demos/opengl/gpuhelper.h:inline void GpuHelper::popMatrix(GLenum matrixTarget)
External/EigenOld/demos/opengl/gpuhelper.h:inline void GpuHelper::draw(GLenum mode, uint nofElement)
External/EigenOld/demos/opengl/gpuhelper.h:inline void GpuHelper::draw(GLenum mode, const std::vector<uint>* pIndexes)
External/EigenOld/demos/opengl/gpuhelper.h:inline void GpuHelper::draw(GLenum mode, uint start, uint end)
External/EigenOld/demos/opengl/gpuhelper.h:#endif // EIGEN_GPUHELPER_H
External/EigenOld/demos/opengl/quaternion_demo.cpp:        gpu.pushMatrix(GL_MODELVIEW);
External/EigenOld/demos/opengl/quaternion_demo.cpp:        gpu.multMatrix(t.matrix(),GL_MODELVIEW);
External/EigenOld/demos/opengl/quaternion_demo.cpp:        gpu.popMatrix(GL_MODELVIEW);
External/EigenOld/demos/opengl/quaternion_demo.cpp:  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitX(), Color(1,0,0,1));
External/EigenOld/demos/opengl/quaternion_demo.cpp:  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitY(), Color(0,1,0,1));
External/EigenOld/demos/opengl/quaternion_demo.cpp:  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitZ(), Color(0,0,1,1));
External/gtest.framework/Headers/internal/gtest-port.h:// implementation.  NVIDIA's CUDA NVCC compiler pretends to be GCC by
External/gtest.framework/Headers/internal/gtest-port.h:#if (defined(__GNUC__) && !defined(__CUDACC__) && (GTEST_GCC_VER_ >= 40000)) \
External/gtest.framework/Versions/A/Headers/internal/gtest-port.h:// implementation.  NVIDIA's CUDA NVCC compiler pretends to be GCC by
External/gtest.framework/Versions/A/Headers/internal/gtest-port.h:#if (defined(__GNUC__) && !defined(__CUDACC__) && (GTEST_GCC_VER_ >= 40000)) \
External/gtest.framework/Versions/Current/Headers/internal/gtest-port.h:// implementation.  NVIDIA's CUDA NVCC compiler pretends to be GCC by
External/gtest.framework/Versions/Current/Headers/internal/gtest-port.h:#if (defined(__GNUC__) && !defined(__CUDACC__) && (GTEST_GCC_VER_ >= 40000)) \
External/Eigen-3.1.0-alpha2/unsupported/test/mpreal/dlmalloc.c:        Thanks to Tony E. Bennett <tbennett@nvidia.com> and others.
External/Eigen-3.1.0-alpha2/unsupported/test/openglsupport.cpp:    #ifdef GLEW_ARB_gpu_shader_fp64
External/Eigen-3.1.0-alpha2/unsupported/test/openglsupport.cpp:    if(GLEW_ARB_gpu_shader_fp64)
External/Eigen-3.1.0-alpha2/unsupported/test/openglsupport.cpp:      #ifdef GL_ARB_gpu_shader_fp64
External/Eigen-3.1.0-alpha2/unsupported/test/openglsupport.cpp:      std::cerr << "Warning: GLEW_ARB_gpu_shader_fp64 was not tested\n";
External/Eigen-3.1.0-alpha2/unsupported/test/openglsupport.cpp:      std::cerr << "Warning: GLEW_ARB_gpu_shader_fp64 was not tested\n";
External/Eigen-3.1.0-alpha2/unsupported/Eigen/OpenGLSupport:#ifdef GL_ARB_gpu_shader_fp64
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.cpp:#include "gpuhelper.h"
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.cpp:GpuHelper gpu;
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.cpp:GpuHelper::GpuHelper()
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.cpp:GpuHelper::~GpuHelper()
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.cpp:void GpuHelper::pushProjectionMode2D(ProjectionMode2D pm)
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.cpp:void GpuHelper::popProjectionMode2D(void)
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.cpp:void GpuHelper::drawVector(const Vector3f& position, const Vector3f& vec, const Color& color, float aspect /* = 50.*/)
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.cpp:void GpuHelper::drawVectorBox(const Vector3f& position, const Vector3f& vec, const Color& color, float aspect)
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.cpp:void GpuHelper::drawUnitCube(void)
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.cpp:void GpuHelper::drawUnitSphere(int level)
External/Eigen-3.1.0-alpha2/demos/opengl/quaternion_demo.h:#include "gpuhelper.h"
External/Eigen-3.1.0-alpha2/demos/opengl/CMakeLists.txt:set(quaternion_demo_SRCS  gpuhelper.cpp icosphere.cpp camera.cpp trackball.cpp quaternion_demo.cpp)
External/Eigen-3.1.0-alpha2/demos/opengl/camera.cpp:#include "gpuhelper.h"
External/Eigen-3.1.0-alpha2/demos/opengl/camera.cpp:  gpu.loadMatrix(projectionMatrix(),GL_PROJECTION);
External/Eigen-3.1.0-alpha2/demos/opengl/camera.cpp:  gpu.loadMatrix(viewMatrix().matrix(),GL_MODELVIEW);
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:#ifndef EIGEN_GPUHELPER_H
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:#define EIGEN_GPUHELPER_H
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:class GpuHelper
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:    GpuHelper();
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:    ~GpuHelper();
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:extern GpuHelper gpu;
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:inline void GpuHelper::setMatrixTarget(GLenum matrixTarget)
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:void GpuHelper::multMatrix(const Matrix<Scalar,4,4, _Flags, 4,4>& mat, GLenum matrixTarget)
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:void GpuHelper::loadMatrix(
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:void GpuHelper::loadMatrix(const Eigen::Matrix<Scalar,4,4, _Flags, 4,4>& mat, GLenum matrixTarget)
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:inline void GpuHelper::pushMatrix(GLenum matrixTarget)
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:inline void GpuHelper::pushMatrix(const Matrix<Scalar,4,4, _Flags, 4,4>& mat, GLenum matrixTarget)
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:void GpuHelper::pushMatrix(
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:inline void GpuHelper::popMatrix(GLenum matrixTarget)
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:inline void GpuHelper::draw(GLenum mode, uint nofElement)
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:inline void GpuHelper::draw(GLenum mode, const std::vector<uint>* pIndexes)
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:inline void GpuHelper::draw(GLenum mode, uint start, uint end)
External/Eigen-3.1.0-alpha2/demos/opengl/gpuhelper.h:#endif // EIGEN_GPUHELPER_H
External/Eigen-3.1.0-alpha2/demos/opengl/quaternion_demo.cpp:        gpu.pushMatrix(GL_MODELVIEW);
External/Eigen-3.1.0-alpha2/demos/opengl/quaternion_demo.cpp:        gpu.multMatrix(t.matrix(),GL_MODELVIEW);
External/Eigen-3.1.0-alpha2/demos/opengl/quaternion_demo.cpp:        gpu.popMatrix(GL_MODELVIEW);
External/Eigen-3.1.0-alpha2/demos/opengl/quaternion_demo.cpp:  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitX(), Color(1,0,0,1));
External/Eigen-3.1.0-alpha2/demos/opengl/quaternion_demo.cpp:  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitY(), Color(0,1,0,1));
External/Eigen-3.1.0-alpha2/demos/opengl/quaternion_demo.cpp:  gpu.drawVector(Vector3f::Zero(), length*Vector3f::UnitZ(), Color(0,0,1,1));

```
