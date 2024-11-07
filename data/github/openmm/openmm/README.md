# https://github.com/openmm/openmm

```console
plugins/drude/CMakeLists.txt:IF(OPENMM_BUILD_OPENCL_LIB)
plugins/drude/CMakeLists.txt:    SET(OPENMM_BUILD_DRUDE_OPENCL_LIB ON CACHE BOOL "Build Drude implementation for OpenCL")
plugins/drude/CMakeLists.txt:ELSE(OPENMM_BUILD_OPENCL_LIB)
plugins/drude/CMakeLists.txt:    SET(OPENMM_BUILD_DRUDE_OPENCL_LIB OFF CACHE BOOL "Build Drude implementation for OpenCL")
plugins/drude/CMakeLists.txt:ENDIF(OPENMM_BUILD_OPENCL_LIB)
plugins/drude/CMakeLists.txt:IF(OPENMM_BUILD_DRUDE_OPENCL_LIB)
plugins/drude/CMakeLists.txt:    ADD_SUBDIRECTORY(platforms/opencl)
plugins/drude/CMakeLists.txt:ENDIF(OPENMM_BUILD_DRUDE_OPENCL_LIB)
plugins/drude/CMakeLists.txt:IF(OPENMM_BUILD_CUDA_LIB)
plugins/drude/CMakeLists.txt:    SET(OPENMM_BUILD_DRUDE_CUDA_LIB ON CACHE BOOL "Build Drude implementation for CUDA")
plugins/drude/CMakeLists.txt:ELSE(OPENMM_BUILD_CUDA_LIB)
plugins/drude/CMakeLists.txt:    SET(OPENMM_BUILD_DRUDE_CUDA_LIB OFF CACHE BOOL "Build Drude implementation for CUDA")
plugins/drude/CMakeLists.txt:ENDIF(OPENMM_BUILD_CUDA_LIB)
plugins/drude/CMakeLists.txt:IF(OPENMM_BUILD_DRUDE_CUDA_LIB)
plugins/drude/CMakeLists.txt:    ADD_SUBDIRECTORY(platforms/cuda)
plugins/drude/CMakeLists.txt:ENDIF(OPENMM_BUILD_DRUDE_CUDA_LIB)
plugins/drude/platforms/opencl/tests/TestOpenCLDrudeNoseHoover.cpp:#include "OpenCLDrudeTests.h"
plugins/drude/platforms/opencl/tests/TestOpenCLDrudeLangevinIntegrator.cpp:#include "OpenCLDrudeTests.h"
plugins/drude/platforms/opencl/tests/TestOpenCLDrudeSCFIntegrator.cpp:#include "OpenCLDrudeTests.h"
plugins/drude/platforms/opencl/tests/CMakeLists.txt:INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIR})
plugins/drude/platforms/opencl/tests/CMakeLists.txt:INCLUDE_DIRECTORIES(${OPENMM_DIR}/platforms/opencl/tests)
plugins/drude/platforms/opencl/tests/CMakeLists.txt:    ADD_TEST(${TEST_ROOT}Single ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT} single ${OPENCL_TEST_PLATFORM_INDEX} ${OPENCL_TEST_DEVICE_INDEX})
plugins/drude/platforms/opencl/tests/CMakeLists.txt:    IF (OPENMM_BUILD_OPENCL_DOUBLE_PRECISION_TESTS)
plugins/drude/platforms/opencl/tests/CMakeLists.txt:        ADD_TEST(${TEST_ROOT}Mixed ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT} mixed ${OPENCL_TEST_PLATFORM_INDEX} ${OPENCL_TEST_DEVICE_INDEX})
plugins/drude/platforms/opencl/tests/CMakeLists.txt:        ADD_TEST(${TEST_ROOT}Double ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT} double ${OPENCL_TEST_PLATFORM_INDEX} ${OPENCL_TEST_DEVICE_INDEX})
plugins/drude/platforms/opencl/tests/CMakeLists.txt:    ENDIF(OPENMM_BUILD_OPENCL_DOUBLE_PRECISION_TESTS)
plugins/drude/platforms/opencl/tests/OpenCLDrudeTests.h:#include "OpenCLTests.h"
plugins/drude/platforms/opencl/tests/OpenCLDrudeTests.h:extern "C" void registerDrudeOpenCLKernelFactories();
plugins/drude/platforms/opencl/tests/OpenCLDrudeTests.h:    registerDrudeOpenCLKernelFactories();
plugins/drude/platforms/opencl/tests/OpenCLDrudeTests.h:    platform = dynamic_cast<OpenCLPlatform&>(Platform::getPlatformByName("OpenCL"));
plugins/drude/platforms/opencl/tests/TestOpenCLDrudeForce.cpp:#include "OpenCLDrudeTests.h"
plugins/drude/platforms/opencl/include/OpenCLDrudeKernelFactory.h:#ifndef OPENMM_OPENCLDRUDEKERNELFACTORY_H_
plugins/drude/platforms/opencl/include/OpenCLDrudeKernelFactory.h:#define OPENMM_OPENCLDRUDEKERNELFACTORY_H_
plugins/drude/platforms/opencl/include/OpenCLDrudeKernelFactory.h: * This KernelFactory creates kernels for the OpenCL implementation of the Drude plugin.
plugins/drude/platforms/opencl/include/OpenCLDrudeKernelFactory.h:class OpenCLDrudeKernelFactory : public KernelFactory {
plugins/drude/platforms/opencl/include/OpenCLDrudeKernelFactory.h:#endif /*OPENMM_OPENCLDRUDEKERNELFACTORY_H_*/
plugins/drude/platforms/opencl/CMakeLists.txt:# OpenMM OpenCL Drude Integrator
plugins/drude/platforms/opencl/CMakeLists.txt:# Creates OpenMMDrudeOpenCL library.
plugins/drude/platforms/opencl/CMakeLists.txt:#   OpenMMDrudeOpenCL.dll
plugins/drude/platforms/opencl/CMakeLists.txt:#   OpenMMDrudeOpenCL.lib
plugins/drude/platforms/opencl/CMakeLists.txt:#   libOpenMMDrudeOpenCL.so
plugins/drude/platforms/opencl/CMakeLists.txt:SET(OPENMMDRUDEOPENCL_LIBRARY_NAME OpenMMDrudeOpenCL)
plugins/drude/platforms/opencl/CMakeLists.txt:SET(SHARED_TARGET ${OPENMMDRUDEOPENCL_LIBRARY_NAME})
plugins/drude/platforms/opencl/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/opencl/include)
plugins/drude/platforms/opencl/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/opencl/src)
plugins/drude/platforms/opencl/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_BINARY_DIR}/platforms/opencl/src)
plugins/drude/platforms/opencl/CMakeLists.txt:INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIR})
plugins/drude/platforms/opencl/CMakeLists.txt:TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_LIBRARY_NAME}  ${OPENCL_LIBRARIES} ${PTHREADS_LIB})
plugins/drude/platforms/opencl/CMakeLists.txt:TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_LIBRARY_NAME}OpenCL)
plugins/drude/platforms/opencl/CMakeLists.txt:# Ensure that links to the main OpenCL library will be resolved.
plugins/drude/platforms/opencl/CMakeLists.txt:    SET(OPENCL_LIBRARY libOpenMMOpenCL.dylib)
plugins/drude/platforms/opencl/CMakeLists.txt:    INSTALL(CODE "EXECUTE_PROCESS(COMMAND install_name_tool -change ${OPENCL_LIBRARY} @loader_path/${OPENCL_LIBRARY} ${CMAKE_INSTALL_PREFIX}/lib/plugins/lib${SHARED_TARGET}.dylib)")
plugins/drude/platforms/opencl/CMakeLists.txt:if(BUILD_TESTING AND OPENMM_BUILD_OPENCL_TESTS)
plugins/drude/platforms/opencl/CMakeLists.txt:endif(BUILD_TESTING AND OPENMM_BUILD_OPENCL_TESTS)
plugins/drude/platforms/opencl/src/OpenCLDrudeKernelFactory.cpp:#include "OpenCLDrudeKernelFactory.h"
plugins/drude/platforms/opencl/src/OpenCLDrudeKernelFactory.cpp:#include "OpenCLContext.h"
plugins/drude/platforms/opencl/src/OpenCLDrudeKernelFactory.cpp:        Platform& platform = Platform::getPlatformByName("OpenCL");
plugins/drude/platforms/opencl/src/OpenCLDrudeKernelFactory.cpp:        OpenCLDrudeKernelFactory* factory = new OpenCLDrudeKernelFactory();
plugins/drude/platforms/opencl/src/OpenCLDrudeKernelFactory.cpp:extern "C" OPENMM_EXPORT void registerDrudeOpenCLKernelFactories() {
plugins/drude/platforms/opencl/src/OpenCLDrudeKernelFactory.cpp:        Platform::getPlatformByName("OpenCL");
plugins/drude/platforms/opencl/src/OpenCLDrudeKernelFactory.cpp:        Platform::registerPlatform(new OpenCLPlatform());
plugins/drude/platforms/opencl/src/OpenCLDrudeKernelFactory.cpp:KernelImpl* OpenCLDrudeKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
plugins/drude/platforms/opencl/src/OpenCLDrudeKernelFactory.cpp:    OpenCLContext& cl = *static_cast<OpenCLPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
plugins/drude/platforms/cuda/tests/TestCudaDrudeLangevinIntegrator.cpp:#include "CudaDrudeTests.h"
plugins/drude/platforms/cuda/tests/TestCudaDrudeNoseHoover.cpp:#include "CudaDrudeTests.h"
plugins/drude/platforms/cuda/tests/CudaDrudeTests.h:#include "CudaTests.h"
plugins/drude/platforms/cuda/tests/CudaDrudeTests.h:extern "C" void registerDrudeCudaKernelFactories();
plugins/drude/platforms/cuda/tests/CudaDrudeTests.h:    registerDrudeCudaKernelFactories();
plugins/drude/platforms/cuda/tests/CudaDrudeTests.h:    platform = dynamic_cast<CudaPlatform&>(Platform::getPlatformByName("CUDA"));
plugins/drude/platforms/cuda/tests/CMakeLists.txt:INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIR})
plugins/drude/platforms/cuda/tests/CMakeLists.txt:INCLUDE_DIRECTORIES(${OPENMM_DIR}/platforms/cuda/tests)
plugins/drude/platforms/cuda/tests/CMakeLists.txt:        SET_TARGET_PROPERTIES(${TEST_ROOT} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS} -F/Library/Frameworks -framework CUDA" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")
plugins/drude/platforms/cuda/tests/CMakeLists.txt:    IF (OPENMM_BUILD_CUDA_DOUBLE_PRECISION_TESTS)
plugins/drude/platforms/cuda/tests/CMakeLists.txt:    ENDIF(OPENMM_BUILD_CUDA_DOUBLE_PRECISION_TESTS)
plugins/drude/platforms/cuda/tests/TestCudaDrudeForce.cpp:#include "CudaDrudeTests.h"
plugins/drude/platforms/cuda/tests/TestCudaDrudeSCFIntegrator.cpp:#include "CudaDrudeTests.h"
plugins/drude/platforms/cuda/include/CudaDrudeKernelFactory.h:#ifndef OPENMM_CUDADRUDEKERNELFACTORY_H_
plugins/drude/platforms/cuda/include/CudaDrudeKernelFactory.h:#define OPENMM_CUDADRUDEKERNELFACTORY_H_
plugins/drude/platforms/cuda/include/CudaDrudeKernelFactory.h: * This KernelFactory creates kernels for the CUDA implementation of the Drude plugin.
plugins/drude/platforms/cuda/include/CudaDrudeKernelFactory.h:class CudaDrudeKernelFactory : public KernelFactory {
plugins/drude/platforms/cuda/include/CudaDrudeKernelFactory.h:#endif /*OPENMM_CUDADRUDEKERNELFACTORY_H_*/
plugins/drude/platforms/cuda/CMakeLists.txt:# OpenMM CUDA Drude Integrator
plugins/drude/platforms/cuda/CMakeLists.txt:# Creates OpenMMDrudeCUDA library.
plugins/drude/platforms/cuda/CMakeLists.txt:#   OpenMMDrudeCUDA.dll
plugins/drude/platforms/cuda/CMakeLists.txt:#   OpenMMDrudeCUDA.lib
plugins/drude/platforms/cuda/CMakeLists.txt:#   libOpenMMDrudeCUDA.so
plugins/drude/platforms/cuda/CMakeLists.txt:SET(OPENMMDRUDECUDA_LIBRARY_NAME OpenMMDrudeCUDA)
plugins/drude/platforms/cuda/CMakeLists.txt:SET(SHARED_TARGET ${OPENMMDRUDECUDA_LIBRARY_NAME})
plugins/drude/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/cuda/include)
plugins/drude/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/cuda/src)
plugins/drude/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_BINARY_DIR}/platforms/cuda/src)
plugins/drude/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_INCLUDE})
plugins/drude/platforms/cuda/CMakeLists.txt:TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_LIBRARY_NAME}CUDA)
plugins/drude/platforms/cuda/CMakeLists.txt:    SET_TARGET_PROPERTIES(${SHARED_TARGET} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS} -F/Library/Frameworks -framework CUDA")
plugins/drude/platforms/cuda/CMakeLists.txt:# Ensure that links to the main CUDA library will be resolved.
plugins/drude/platforms/cuda/CMakeLists.txt:    SET(CUDA_LIBRARY libOpenMMCUDA.dylib)
plugins/drude/platforms/cuda/CMakeLists.txt:    INSTALL(CODE "EXECUTE_PROCESS(COMMAND install_name_tool -change ${CUDA_LIBRARY} @loader_path/${CUDA_LIBRARY} ${CMAKE_INSTALL_PREFIX}/lib/plugins/lib${SHARED_TARGET}.dylib)")
plugins/drude/platforms/cuda/CMakeLists.txt:if(BUILD_TESTING AND OPENMM_BUILD_CUDA_TESTS)
plugins/drude/platforms/cuda/CMakeLists.txt:endif(BUILD_TESTING AND OPENMM_BUILD_CUDA_TESTS)
plugins/drude/platforms/cuda/src/CudaDrudeKernelFactory.cpp:#include "CudaDrudeKernelFactory.h"
plugins/drude/platforms/cuda/src/CudaDrudeKernelFactory.cpp:#include "CudaContext.h"
plugins/drude/platforms/cuda/src/CudaDrudeKernelFactory.cpp:        Platform& platform = Platform::getPlatformByName("CUDA");
plugins/drude/platforms/cuda/src/CudaDrudeKernelFactory.cpp:        CudaDrudeKernelFactory* factory = new CudaDrudeKernelFactory();
plugins/drude/platforms/cuda/src/CudaDrudeKernelFactory.cpp:extern "C" OPENMM_EXPORT void registerDrudeCudaKernelFactories() {
plugins/drude/platforms/cuda/src/CudaDrudeKernelFactory.cpp:        Platform::getPlatformByName("CUDA");
plugins/drude/platforms/cuda/src/CudaDrudeKernelFactory.cpp:        Platform::registerPlatform(new CudaPlatform());
plugins/drude/platforms/cuda/src/CudaDrudeKernelFactory.cpp:KernelImpl* CudaDrudeKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
plugins/drude/platforms/cuda/src/CudaDrudeKernelFactory.cpp:    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
plugins/cpupme/src/CpuPmeKernels.cpp:                // error of 1.5e-7.  Stolen by ACS from the CUDA platform's AMOEBA plugin.
plugins/rpmd/tests/TestRpmd.h: * This tests the OpenCL implementation of RPMDIntegrator.
plugins/rpmd/CMakeLists.txt:IF(OPENMM_BUILD_OPENCL_LIB)
plugins/rpmd/CMakeLists.txt:    SET(OPENMM_BUILD_RPMD_OPENCL_LIB ON CACHE BOOL "Build RPMD implementation for OpenCL")
plugins/rpmd/CMakeLists.txt:ELSE(OPENMM_BUILD_OPENCL_LIB)
plugins/rpmd/CMakeLists.txt:    SET(OPENMM_BUILD_RPMD_OPENCL_LIB OFF CACHE BOOL "Build RPMD implementation for OpenCL")
plugins/rpmd/CMakeLists.txt:ENDIF(OPENMM_BUILD_OPENCL_LIB)
plugins/rpmd/CMakeLists.txt:IF(OPENMM_BUILD_RPMD_OPENCL_LIB)
plugins/rpmd/CMakeLists.txt:    ADD_SUBDIRECTORY(platforms/opencl)
plugins/rpmd/CMakeLists.txt:ENDIF(OPENMM_BUILD_RPMD_OPENCL_LIB)
plugins/rpmd/CMakeLists.txt:IF(OPENMM_BUILD_CUDA_LIB)
plugins/rpmd/CMakeLists.txt:    SET(OPENMM_BUILD_RPMD_CUDA_LIB ON CACHE BOOL "Build RPMD implementation for CUDA")
plugins/rpmd/CMakeLists.txt:ELSE(OPENMM_BUILD_CUDA_LIB)
plugins/rpmd/CMakeLists.txt:    SET(OPENMM_BUILD_RPMD_CUDA_LIB OFF CACHE BOOL "Build RPMD implementation for CUDA")
plugins/rpmd/CMakeLists.txt:ENDIF(OPENMM_BUILD_CUDA_LIB)
plugins/rpmd/CMakeLists.txt:IF(OPENMM_BUILD_RPMD_CUDA_LIB)
plugins/rpmd/CMakeLists.txt:    ADD_SUBDIRECTORY(platforms/cuda)
plugins/rpmd/CMakeLists.txt:ENDIF(OPENMM_BUILD_RPMD_CUDA_LIB)
plugins/rpmd/platforms/opencl/tests/TestOpenCLRpmd.cpp:#include "OpenCLTests.h"
plugins/rpmd/platforms/opencl/tests/TestOpenCLRpmd.cpp:extern "C" void registerRPMDOpenCLKernelFactories();
plugins/rpmd/platforms/opencl/tests/TestOpenCLRpmd.cpp:    registerRPMDOpenCLKernelFactories();
plugins/rpmd/platforms/opencl/tests/TestOpenCLRpmd.cpp:    platform = dynamic_cast<OpenCLPlatform&>(Platform::getPlatformByName("OpenCL"));
plugins/rpmd/platforms/opencl/tests/CMakeLists.txt:INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIR})
plugins/rpmd/platforms/opencl/tests/CMakeLists.txt:INCLUDE_DIRECTORIES(${OPENMM_DIR}/platforms/opencl/tests)
plugins/rpmd/platforms/opencl/tests/CMakeLists.txt:    ADD_TEST(${TEST_ROOT}Single ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT} single ${OPENCL_TEST_PLATFORM_INDEX} ${OPENCL_TEST_DEVICE_INDEX})
plugins/rpmd/platforms/opencl/tests/CMakeLists.txt:    IF (OPENMM_BUILD_OPENCL_DOUBLE_PRECISION_TESTS)
plugins/rpmd/platforms/opencl/tests/CMakeLists.txt:        ADD_TEST(${TEST_ROOT}Mixed ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT} mixed ${OPENCL_TEST_PLATFORM_INDEX} ${OPENCL_TEST_DEVICE_INDEX})
plugins/rpmd/platforms/opencl/tests/CMakeLists.txt:        ADD_TEST(${TEST_ROOT}Double ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT} double ${OPENCL_TEST_PLATFORM_INDEX} ${OPENCL_TEST_DEVICE_INDEX})
plugins/rpmd/platforms/opencl/tests/CMakeLists.txt:    ENDIF(OPENMM_BUILD_OPENCL_DOUBLE_PRECISION_TESTS)
plugins/rpmd/platforms/opencl/include/OpenCLRpmdKernelFactory.h:#ifndef OPENMM_OPENCLRPMDKERNELFACTORY_H_
plugins/rpmd/platforms/opencl/include/OpenCLRpmdKernelFactory.h:#define OPENMM_OPENCLRPMDKERNELFACTORY_H_
plugins/rpmd/platforms/opencl/include/OpenCLRpmdKernelFactory.h: * This KernelFactory creates kernels for the OpenCL implementation of RPMDIntegrator.
plugins/rpmd/platforms/opencl/include/OpenCLRpmdKernelFactory.h:class OpenCLRpmdKernelFactory : public KernelFactory {
plugins/rpmd/platforms/opencl/include/OpenCLRpmdKernelFactory.h:#endif /*OPENMM_OPENCLRPMDKERNELFACTORY_H_*/
plugins/rpmd/platforms/opencl/CMakeLists.txt:# OpenMM OpenCL RPMD Integrator
plugins/rpmd/platforms/opencl/CMakeLists.txt:# Creates OpenMMRPMDOpenCL library.
plugins/rpmd/platforms/opencl/CMakeLists.txt:#   OpenMMRPMDOpenCL.dll
plugins/rpmd/platforms/opencl/CMakeLists.txt:#   OpenMMRPMDOpenCL.lib
plugins/rpmd/platforms/opencl/CMakeLists.txt:#   libOpenMMRPMDOpenCL.so
plugins/rpmd/platforms/opencl/CMakeLists.txt:SET(OPENMMRPMDOPENCL_LIBRARY_NAME OpenMMRPMDOpenCL)
plugins/rpmd/platforms/opencl/CMakeLists.txt:SET(SHARED_TARGET ${OPENMMRPMDOPENCL_LIBRARY_NAME})
plugins/rpmd/platforms/opencl/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/opencl/include)
plugins/rpmd/platforms/opencl/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/opencl/src)
plugins/rpmd/platforms/opencl/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_BINARY_DIR}/platforms/opencl/src)
plugins/rpmd/platforms/opencl/CMakeLists.txt:INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIR})
plugins/rpmd/platforms/opencl/CMakeLists.txt:TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_LIBRARY_NAME}  ${OPENCL_LIBRARIES} ${PTHREADS_LIB})
plugins/rpmd/platforms/opencl/CMakeLists.txt:TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_LIBRARY_NAME}OpenCL)
plugins/rpmd/platforms/opencl/CMakeLists.txt:# Ensure that links to the main OpenCL library will be resolved.
plugins/rpmd/platforms/opencl/CMakeLists.txt:    SET(OPENCL_LIBRARY libOpenMMOpenCL.dylib)
plugins/rpmd/platforms/opencl/CMakeLists.txt:    INSTALL(CODE "EXECUTE_PROCESS(COMMAND install_name_tool -change ${OPENCL_LIBRARY} @loader_path/${OPENCL_LIBRARY} ${CMAKE_INSTALL_PREFIX}/lib/plugins/lib${SHARED_TARGET}.dylib)")
plugins/rpmd/platforms/opencl/CMakeLists.txt:if(BUILD_TESTING AND OPENMM_BUILD_OPENCL_TESTS)
plugins/rpmd/platforms/opencl/CMakeLists.txt:endif(BUILD_TESTING AND OPENMM_BUILD_OPENCL_TESTS)
plugins/rpmd/platforms/opencl/src/OpenCLRpmdKernelFactory.cpp:#include "OpenCLRpmdKernelFactory.h"
plugins/rpmd/platforms/opencl/src/OpenCLRpmdKernelFactory.cpp:#include "OpenCLContext.h"
plugins/rpmd/platforms/opencl/src/OpenCLRpmdKernelFactory.cpp:        Platform& platform = Platform::getPlatformByName("OpenCL");
plugins/rpmd/platforms/opencl/src/OpenCLRpmdKernelFactory.cpp:        OpenCLRpmdKernelFactory* factory = new OpenCLRpmdKernelFactory();
plugins/rpmd/platforms/opencl/src/OpenCLRpmdKernelFactory.cpp:extern "C" OPENMM_EXPORT void registerRPMDOpenCLKernelFactories() {
plugins/rpmd/platforms/opencl/src/OpenCLRpmdKernelFactory.cpp:        Platform::getPlatformByName("OpenCL");
plugins/rpmd/platforms/opencl/src/OpenCLRpmdKernelFactory.cpp:        Platform::registerPlatform(new OpenCLPlatform());
plugins/rpmd/platforms/opencl/src/OpenCLRpmdKernelFactory.cpp:KernelImpl* OpenCLRpmdKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
plugins/rpmd/platforms/opencl/src/OpenCLRpmdKernelFactory.cpp:    OpenCLContext& cl = *static_cast<OpenCLPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
plugins/rpmd/platforms/cuda/tests/TestCudaRpmd.cpp:#include "CudaTests.h"
plugins/rpmd/platforms/cuda/tests/TestCudaRpmd.cpp:extern "C" void registerRPMDCudaKernelFactories();
plugins/rpmd/platforms/cuda/tests/TestCudaRpmd.cpp:    registerRPMDCudaKernelFactories();
plugins/rpmd/platforms/cuda/tests/TestCudaRpmd.cpp:    platform = dynamic_cast<CudaPlatform&>(Platform::getPlatformByName("CUDA"));
plugins/rpmd/platforms/cuda/tests/CMakeLists.txt:INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIR})
plugins/rpmd/platforms/cuda/tests/CMakeLists.txt:INCLUDE_DIRECTORIES(${OPENMM_DIR}/platforms/cuda/tests)
plugins/rpmd/platforms/cuda/tests/CMakeLists.txt:        SET_TARGET_PROPERTIES(${TEST_ROOT} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS} -F/Library/Frameworks -framework CUDA" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")
plugins/rpmd/platforms/cuda/tests/CMakeLists.txt:    IF (OPENMM_BUILD_CUDA_DOUBLE_PRECISION_TESTS)
plugins/rpmd/platforms/cuda/tests/CMakeLists.txt:    ENDIF(OPENMM_BUILD_CUDA_DOUBLE_PRECISION_TESTS)
plugins/rpmd/platforms/cuda/include/CudaRpmdKernelFactory.h:#ifndef OPENMM_CUDARPMDKERNELFACTORY_H_
plugins/rpmd/platforms/cuda/include/CudaRpmdKernelFactory.h:#define OPENMM_CUDARPMDKERNELFACTORY_H_
plugins/rpmd/platforms/cuda/include/CudaRpmdKernelFactory.h: * This KernelFactory creates kernels for the CUDA implementation of RPMDIntegrator.
plugins/rpmd/platforms/cuda/include/CudaRpmdKernelFactory.h:class CudaRpmdKernelFactory : public KernelFactory {
plugins/rpmd/platforms/cuda/include/CudaRpmdKernelFactory.h:#endif /*OPENMM_CUDARPMDKERNELFACTORY_H_*/
plugins/rpmd/platforms/cuda/CMakeLists.txt:# OpenMM CUDA RPMD Integrator
plugins/rpmd/platforms/cuda/CMakeLists.txt:# Creates OpenMMRPMDCUDA library,.
plugins/rpmd/platforms/cuda/CMakeLists.txt:#   OpenMMRPMDCUDA.dll
plugins/rpmd/platforms/cuda/CMakeLists.txt:#   OpenMMRPMDCUDA.lib
plugins/rpmd/platforms/cuda/CMakeLists.txt:#   libOpenMMRPMDCUDA.so
plugins/rpmd/platforms/cuda/CMakeLists.txt:SET(OPENMMRPMDCUDA_LIBRARY_NAME OpenMMRPMDCUDA)
plugins/rpmd/platforms/cuda/CMakeLists.txt:SET(SHARED_TARGET ${OPENMMRPMDCUDA_LIBRARY_NAME})
plugins/rpmd/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/cuda/include)
plugins/rpmd/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/cuda/src)
plugins/rpmd/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_BINARY_DIR}/platforms/cuda/src)
plugins/rpmd/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_INCLUDE})
plugins/rpmd/platforms/cuda/CMakeLists.txt:TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_LIBRARY_NAME}CUDA)
plugins/rpmd/platforms/cuda/CMakeLists.txt:    SET_TARGET_PROPERTIES(${SHARED_TARGET} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS} -F/Library/Frameworks -framework CUDA")
plugins/rpmd/platforms/cuda/CMakeLists.txt:# Ensure that links to the main CUDA library will be resolved.
plugins/rpmd/platforms/cuda/CMakeLists.txt:    SET(CUDA_LIBRARY libOpenMMCUDA.dylib)
plugins/rpmd/platforms/cuda/CMakeLists.txt:    INSTALL(CODE "EXECUTE_PROCESS(COMMAND install_name_tool -change ${CUDA_LIBRARY} @loader_path/${CUDA_LIBRARY} ${CMAKE_INSTALL_PREFIX}/lib/plugins/lib${SHARED_TARGET}.dylib)")
plugins/rpmd/platforms/cuda/CMakeLists.txt:if(BUILD_TESTING AND OPENMM_BUILD_CUDA_TESTS)
plugins/rpmd/platforms/cuda/CMakeLists.txt:endif(BUILD_TESTING AND OPENMM_BUILD_CUDA_TESTS)
plugins/rpmd/platforms/cuda/src/CudaRpmdKernelFactory.cpp:#include "CudaRpmdKernelFactory.h"
plugins/rpmd/platforms/cuda/src/CudaRpmdKernelFactory.cpp:#include "CudaContext.h"
plugins/rpmd/platforms/cuda/src/CudaRpmdKernelFactory.cpp:        Platform& platform = Platform::getPlatformByName("CUDA");
plugins/rpmd/platforms/cuda/src/CudaRpmdKernelFactory.cpp:        CudaRpmdKernelFactory* factory = new CudaRpmdKernelFactory();
plugins/rpmd/platforms/cuda/src/CudaRpmdKernelFactory.cpp:extern "C" OPENMM_EXPORT void registerRPMDCudaKernelFactories() {
plugins/rpmd/platforms/cuda/src/CudaRpmdKernelFactory.cpp:        Platform::getPlatformByName("CUDA");
plugins/rpmd/platforms/cuda/src/CudaRpmdKernelFactory.cpp:        Platform::registerPlatform(new CudaPlatform());
plugins/rpmd/platforms/cuda/src/CudaRpmdKernelFactory.cpp:KernelImpl* CudaRpmdKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
plugins/rpmd/platforms/cuda/src/CudaRpmdKernelFactory.cpp:    CudaContext& cl = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
plugins/amoeba/openmmapi/src/AmoebaVdwForceImpl.cpp:    // Compute the VdW tapering coefficients.  Mostly copied from amoebaCudaGpu.cpp.
plugins/amoeba/tests/TestHippoNonbondedForce.h: * This tests the CUDA implementation of HippoNonbondedForce.
plugins/amoeba/CMakeLists.txt:IF(OPENMM_BUILD_CUDA_LIB)
plugins/amoeba/CMakeLists.txt:    SET(OPENMM_BUILD_AMOEBA_CUDA_LIB ON CACHE BOOL "Build OpenMMAmoebaCuda library for Nvidia GPUs")
plugins/amoeba/CMakeLists.txt:ELSE(OPENMM_BUILD_CUDA_LIB)
plugins/amoeba/CMakeLists.txt:    SET(OPENMM_BUILD_AMOEBA_CUDA_LIB OFF CACHE BOOL "Build OpenMMAmoebaCuda library for Nvidia GPUs")
plugins/amoeba/CMakeLists.txt:ENDIF(OPENMM_BUILD_CUDA_LIB)
plugins/amoeba/CMakeLists.txt:IF(OPENMM_BUILD_OPENCL_LIB)
plugins/amoeba/CMakeLists.txt:    SET(OPENMM_BUILD_AMOEBA_OPENCL_LIB ON CACHE BOOL "Build OpenMMAmoebaOpenCL library")
plugins/amoeba/CMakeLists.txt:ELSE(OPENMM_BUILD_OPENCL_LIB)
plugins/amoeba/CMakeLists.txt:    SET(OPENMM_BUILD_AMOEBA_OPENCL_LIB OFF CACHE BOOL "Build OpenMMAmoebaOpenCL library")
plugins/amoeba/CMakeLists.txt:ENDIF(OPENMM_BUILD_OPENCL_LIB)
plugins/amoeba/CMakeLists.txt:SET(OPENMM_BUILD_AMOEBA_CUDA_PATH)
plugins/amoeba/CMakeLists.txt:IF(OPENMM_BUILD_AMOEBA_CUDA_LIB)
plugins/amoeba/CMakeLists.txt:    ADD_SUBDIRECTORY(platforms/cuda)
plugins/amoeba/CMakeLists.txt:    SET(OPENMM_BUILD_AMOEBA_CUDA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/platforms/cuda)
plugins/amoeba/CMakeLists.txt:    SET(OPENMM_AMOEBA_CUDA_SOURCE_SUBDIRS . openmmapi olla platforms/cuda)
plugins/amoeba/CMakeLists.txt:ENDIF(OPENMM_BUILD_AMOEBA_CUDA_LIB)
plugins/amoeba/CMakeLists.txt:SET(OPENMM_BUILD_AMOEBA_OPENCL_PATH)
plugins/amoeba/CMakeLists.txt:IF(OPENMM_BUILD_AMOEBA_OPENCL_LIB)
plugins/amoeba/CMakeLists.txt:    ADD_SUBDIRECTORY(platforms/opencl)
plugins/amoeba/CMakeLists.txt:    SET(OPENMM_BUILD_AMOEBA_OPENCL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/platforms/opencl)
plugins/amoeba/CMakeLists.txt:    SET(OPENMM_AMOEBA_OPENCL_SOURCE_SUBDIRS . openmmapi olla platforms/opencl)
plugins/amoeba/CMakeLists.txt:ENDIF(OPENMM_BUILD_AMOEBA_OPENCL_LIB)
plugins/amoeba/platforms/opencl/tests/TestOpenCLAmoebaVdwForce.cpp:#include "OpenCLAmoebaTests.h"
plugins/amoeba/platforms/opencl/tests/TestOpenCLAmoebaExtrapolatedPolarization.cpp:#include "OpenCLAmoebaTests.h"
plugins/amoeba/platforms/opencl/tests/OpenCLAmoebaTests.h:#include "OpenCLTests.h"
plugins/amoeba/platforms/opencl/tests/OpenCLAmoebaTests.h:extern "C" void registerAmoebaOpenCLKernelFactories();
plugins/amoeba/platforms/opencl/tests/OpenCLAmoebaTests.h:    registerAmoebaOpenCLKernelFactories();
plugins/amoeba/platforms/opencl/tests/OpenCLAmoebaTests.h:    platform = dynamic_cast<OpenCLPlatform&>(Platform::getPlatformByName("OpenCL"));
plugins/amoeba/platforms/opencl/tests/TestOpenCLAmoebaGeneralizedKirkwoodForce.cpp:#include "OpenCLAmoebaTests.h"
plugins/amoeba/platforms/opencl/tests/CMakeLists.txt:INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIR})
plugins/amoeba/platforms/opencl/tests/CMakeLists.txt:INCLUDE_DIRECTORIES(${OPENMM_DIR}/platforms/opencl/tests)
plugins/amoeba/platforms/opencl/tests/CMakeLists.txt:    ADD_TEST(${TEST_ROOT}Single ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT} single ${OPENCL_TEST_PLATFORM_INDEX} ${OPENCL_TEST_DEVICE_INDEX})
plugins/amoeba/platforms/opencl/tests/CMakeLists.txt:    IF (OPENMM_BUILD_OPENCL_DOUBLE_PRECISION_TESTS)
plugins/amoeba/platforms/opencl/tests/CMakeLists.txt:        ADD_TEST(${TEST_ROOT}Mixed ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT} mixed ${OPENCL_TEST_PLATFORM_INDEX} ${OPENCL_TEST_DEVICE_INDEX})
plugins/amoeba/platforms/opencl/tests/CMakeLists.txt:        # Some AMOEBA classes don't work correctly in double precision on some OpenCL implementations.
plugins/amoeba/platforms/opencl/tests/CMakeLists.txt:        #ADD_TEST(${TEST_ROOT}Double ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT} double ${OPENCL_TEST_PLATFORM_INDEX} ${OPENCL_TEST_DEVICE_INDEX})
plugins/amoeba/platforms/opencl/tests/CMakeLists.txt:    ENDIF(OPENMM_BUILD_OPENCL_DOUBLE_PRECISION_TESTS)
plugins/amoeba/platforms/opencl/tests/TestOpenCLAmoebaMultipoleForce.cpp:#include "OpenCLAmoebaTests.h"
plugins/amoeba/platforms/opencl/tests/TestOpenCLHippoNonbondedForce.cpp:#include "OpenCLAmoebaTests.h"
plugins/amoeba/platforms/opencl/tests/TestOpenCLAmoebaTorsionTorsionForce.cpp:#include "OpenCLAmoebaTests.h"
plugins/amoeba/platforms/opencl/tests/TestOpenCLWcaDispersionForce.cpp:#include "OpenCLAmoebaTests.h"
plugins/amoeba/platforms/opencl/include/AmoebaOpenCLKernelFactory.h:#ifndef AMOEBA_OPENMM_OPENCLKERNELFACTORY_H_
plugins/amoeba/platforms/opencl/include/AmoebaOpenCLKernelFactory.h:#define AMOEBA_OPENMM_OPENCLKERNELFACTORY_H_
plugins/amoeba/platforms/opencl/include/AmoebaOpenCLKernelFactory.h: * This KernelFactory creates all kernels for the AMOEBA OpenCL platform.
plugins/amoeba/platforms/opencl/include/AmoebaOpenCLKernelFactory.h:class AmoebaOpenCLKernelFactory : public KernelFactory {
plugins/amoeba/platforms/opencl/include/AmoebaOpenCLKernelFactory.h:#endif /*AMOEBA_OPENMM_OPENCLKERNELFACTORY_H_*/
plugins/amoeba/platforms/opencl/CMakeLists.txt:# OpenMM OpenCL Amoeba Implementation
plugins/amoeba/platforms/opencl/CMakeLists.txt:# Creates OpenMMAmoebaOpenCL library.
plugins/amoeba/platforms/opencl/CMakeLists.txt:#   OpenMMAmoebaOpenCL.dll
plugins/amoeba/platforms/opencl/CMakeLists.txt:#   OpenMMAmoebaOpenCL.lib
plugins/amoeba/platforms/opencl/CMakeLists.txt:#   libOpenMMAmoebaOpenCL.so
plugins/amoeba/platforms/opencl/CMakeLists.txt:SET(OPENMMAMOEBAOPENCL_LIBRARY_NAME OpenMMAmoebaOpenCL)
plugins/amoeba/platforms/opencl/CMakeLists.txt:SET(SHARED_TARGET ${OPENMMAMOEBAOPENCL_LIBRARY_NAME})
plugins/amoeba/platforms/opencl/CMakeLists.txt:SET(STATIC_TARGET ${OPENMMAMOEBAOPENCL_LIBRARY_NAME}_static)
plugins/amoeba/platforms/opencl/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/opencl/include)
plugins/amoeba/platforms/opencl/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/opencl/src)
plugins/amoeba/platforms/opencl/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_BINARY_DIR}/platforms/opencl/src)
plugins/amoeba/platforms/opencl/CMakeLists.txt:INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIR})
plugins/amoeba/platforms/opencl/CMakeLists.txt:    TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_LIBRARY_NAME}OpenCL)
plugins/amoeba/platforms/opencl/CMakeLists.txt:    TARGET_LINK_LIBRARIES(${STATIC_TARGET} ${OPENMM_LIBRARY_NAME}OpenCL)
plugins/amoeba/platforms/opencl/CMakeLists.txt:# Ensure that links to the main OpenCL library will be resolved.
plugins/amoeba/platforms/opencl/CMakeLists.txt:    SET(OPENCL_LIBRARY libOpenMMOpenCL.dylib)
plugins/amoeba/platforms/opencl/CMakeLists.txt:    INSTALL(CODE "EXECUTE_PROCESS(COMMAND install_name_tool -change ${OPENCL_LIBRARY} @loader_path/${OPENCL_LIBRARY} ${CMAKE_INSTALL_PREFIX}/lib/plugins/lib${SHARED_TARGET}.dylib)")
plugins/amoeba/platforms/opencl/CMakeLists.txt:if(BUILD_TESTING AND OPENMM_BUILD_OPENCL_TESTS)
plugins/amoeba/platforms/opencl/CMakeLists.txt:endif(BUILD_TESTING AND OPENMM_BUILD_OPENCL_TESTS)
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.h:#ifndef AMOEBA_OPENMM_OPENCLKERNELS_H
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.h:#define AMOEBA_OPENMM_OPENCLKERNELS_H
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.h:#include "OpenCLContext.h"
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.h:#include "OpenCLFFT3D.h"
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.h:#include "OpenCLSort.h"
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.h:class OpenCLCalcAmoebaMultipoleForceKernel : public CommonCalcAmoebaMultipoleForceKernel {
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.h:    OpenCLCalcAmoebaMultipoleForceKernel(const std::string& name, const Platform& platform, OpenCLContext& cl, const System& system) :
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.h:    ~OpenCLCalcAmoebaMultipoleForceKernel();
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.h:    OpenCLFFT3D* fft;
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.h:class OpenCLCalcHippoNonbondedForceKernel : public CommonCalcHippoNonbondedForceKernel {
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.h:    OpenCLCalcHippoNonbondedForceKernel(const std::string& name, const Platform& platform, OpenCLContext& cl, const System& system) :
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.h:    ~OpenCLCalcHippoNonbondedForceKernel();
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.h:    class SortTrait : public OpenCLSort::SortTrait {
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.h:    OpenCLSort* sort;
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.h:    OpenCLFFT3D *fftForward, *fftBackward, *dfftForward, *dfftBackward;
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.h:#endif /*AMOEBA_OPENMM_OPENCLKERNELS_H*/
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:#include "AmoebaOpenCLKernels.h"
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:OpenCLCalcAmoebaMultipoleForceKernel::~OpenCLCalcAmoebaMultipoleForceKernel() {
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:void OpenCLCalcAmoebaMultipoleForceKernel::initialize(const System& system, const AmoebaMultipoleForce& force) {
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:        fft = new OpenCLFFT3D(dynamic_cast<OpenCLContext&>(cc), gridSizeX, gridSizeY, gridSizeZ, false);
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:void OpenCLCalcAmoebaMultipoleForceKernel::computeFFT(bool forward) {
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:    OpenCLArray& grid1 = dynamic_cast<OpenCLContext&>(cc).unwrap(pmeGrid1);
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:    OpenCLArray& grid2 = dynamic_cast<OpenCLContext&>(cc).unwrap(pmeGrid2);
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:OpenCLCalcHippoNonbondedForceKernel::~OpenCLCalcHippoNonbondedForceKernel() {
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:void OpenCLCalcHippoNonbondedForceKernel::initialize(const System& system, const HippoNonbondedForce& force) {
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:        OpenCLContext& cl = dynamic_cast<OpenCLContext&>(cc);
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:        sort = new OpenCLSort(cl, new SortTrait(), cc.getNumAtoms());
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:        fftForward = new OpenCLFFT3D(cl, gridSizeX, gridSizeY, gridSizeZ, true);
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:        dfftForward = new OpenCLFFT3D(cl, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, true);
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:void OpenCLCalcHippoNonbondedForceKernel::computeFFT(bool forward, bool dispersion) {
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:    OpenCLArray& grid1 = dynamic_cast<OpenCLContext&>(cc).unwrap(pmeGrid1);
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:    OpenCLArray& grid2 = dynamic_cast<OpenCLContext&>(cc).unwrap(pmeGrid2);
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:    OpenCLFFT3D* fft = (dispersion ? dfftForward : fftForward);
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:void OpenCLCalcHippoNonbondedForceKernel::sortGridIndex() {
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernels.cpp:    sort->sort(dynamic_cast<OpenCLContext&>(cc).unwrap(pmeAtomGridIndex));
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernelFactory.cpp:#include "AmoebaOpenCLKernelFactory.h"
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernelFactory.cpp:#include "AmoebaOpenCLKernels.h"
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernelFactory.cpp:#include "OpenCLContext.h"
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernelFactory.cpp:#include "OpenCLPlatform.h"
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernelFactory.cpp:        Platform& platform = Platform::getPlatformByName("OpenCL");
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernelFactory.cpp:        AmoebaOpenCLKernelFactory* factory = new AmoebaOpenCLKernelFactory();
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernelFactory.cpp:        // Ignore.  The OpenCL platform isn't available.
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernelFactory.cpp:extern "C" OPENMM_EXPORT void registerAmoebaOpenCLKernelFactories() {
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernelFactory.cpp:        Platform::getPlatformByName("OpenCL");
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernelFactory.cpp:        Platform::registerPlatform(new OpenCLPlatform());
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernelFactory.cpp:KernelImpl* AmoebaOpenCLKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernelFactory.cpp:    OpenCLPlatform::PlatformData& data = *static_cast<OpenCLPlatform::PlatformData*>(context.getPlatformData());
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernelFactory.cpp:    OpenCLContext& cc = *data.contexts[0];
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernelFactory.cpp:        return new OpenCLCalcAmoebaMultipoleForceKernel(name, platform, cc, context.getSystem());
plugins/amoeba/platforms/opencl/src/AmoebaOpenCLKernelFactory.cpp:        return new OpenCLCalcHippoNonbondedForceKernel(name, platform, cc, context.getSystem());
plugins/amoeba/platforms/reference/include/windowsExportAmoebaReference.h: *     OpenMMCUDA_BUILDING_{SHARED|STATIC}_LIBRARY
plugins/amoeba/platforms/common/src/AmoebaCommonKernels.cpp:    // Copy the grid points to the GPU.
plugins/amoeba/platforms/common/src/kernels/hippoNonbonded.cc:#if defined(__CUDA_ARCH__) || defined(USE_HIP)
plugins/amoeba/platforms/common/src/kernels/multipolePme.cc:#if __CUDA_ARCH__ < 500
plugins/amoeba/platforms/common/src/kernels/multipolePme.cc:#if __CUDA_ARCH__ < 500
plugins/amoeba/platforms/common/src/kernels/multipolePme.cc:#if __CUDA_ARCH__ < 500
plugins/amoeba/platforms/common/src/kernels/multipolePme.cc:#if __CUDA_ARCH__ < 500
plugins/amoeba/platforms/common/src/kernels/multipoleInducedField.cc:// OpenCL requires a second version of this function, since the signature depends
plugins/amoeba/platforms/cuda/tests/TestCudaAmoebaTorsionTorsionForce.cpp:#include "CudaAmoebaTests.h"
plugins/amoeba/platforms/cuda/tests/TestCudaAmoebaMultipoleForce.cpp:#include "CudaAmoebaTests.h"
plugins/amoeba/platforms/cuda/tests/CudaAmoebaTests.h:#include "CudaTests.h"
plugins/amoeba/platforms/cuda/tests/CudaAmoebaTests.h:extern "C" void registerAmoebaCudaKernelFactories();
plugins/amoeba/platforms/cuda/tests/CudaAmoebaTests.h:    registerAmoebaCudaKernelFactories();
plugins/amoeba/platforms/cuda/tests/CudaAmoebaTests.h:    platform = dynamic_cast<CudaPlatform&>(Platform::getPlatformByName("CUDA"));
plugins/amoeba/platforms/cuda/tests/CMakeLists.txt:INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIR})
plugins/amoeba/platforms/cuda/tests/CMakeLists.txt:INCLUDE_DIRECTORIES(${OPENMM_DIR}/platforms/cuda/tests)
plugins/amoeba/platforms/cuda/tests/CMakeLists.txt:        SET_TARGET_PROPERTIES(${TEST_ROOT} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS} -F/Library/Frameworks -framework CUDA" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")
plugins/amoeba/platforms/cuda/tests/CMakeLists.txt:    IF (OPENMM_BUILD_CUDA_DOUBLE_PRECISION_TESTS)
plugins/amoeba/platforms/cuda/tests/CMakeLists.txt:    ENDIF(OPENMM_BUILD_CUDA_DOUBLE_PRECISION_TESTS)
plugins/amoeba/platforms/cuda/tests/TestCudaAmoebaExtrapolatedPolarization.cpp:#include "CudaAmoebaTests.h"
plugins/amoeba/platforms/cuda/tests/TestCudaAmoebaGeneralizedKirkwoodForce.cpp:#include "CudaAmoebaTests.h"
plugins/amoeba/platforms/cuda/tests/TestCudaAmoebaVdwForce.cpp:#include "CudaAmoebaTests.h"
plugins/amoeba/platforms/cuda/tests/TestCudaHippoNonbondedForce.cpp:#include "CudaAmoebaTests.h"
plugins/amoeba/platforms/cuda/tests/TestCudaWcaDispersionForce.cpp:#include "CudaAmoebaTests.h"
plugins/amoeba/platforms/cuda/include/AmoebaCudaKernelFactory.h:#ifndef AMOEBA_OPENMM_CUDAKERNELFACTORY_H_
plugins/amoeba/platforms/cuda/include/AmoebaCudaKernelFactory.h:#define AMOEBA_OPENMM_CUDAKERNELFACTORY_H_
plugins/amoeba/platforms/cuda/include/AmoebaCudaKernelFactory.h: * This KernelFactory creates all kernels for AmoebaCudaPlatform.
plugins/amoeba/platforms/cuda/include/AmoebaCudaKernelFactory.h:class AmoebaCudaKernelFactory : public KernelFactory {
plugins/amoeba/platforms/cuda/include/AmoebaCudaKernelFactory.h:#endif /*AMOEBA_OPENMM_CUDAKERNELFACTORY_H_*/
plugins/amoeba/platforms/cuda/CMakeLists.txt:# OpenMM CUDA Amoeba Implementation
plugins/amoeba/platforms/cuda/CMakeLists.txt:# Creates OpenMMAmoebaCUDA library.
plugins/amoeba/platforms/cuda/CMakeLists.txt:#   OpenMMAmoebaCUDA.dll
plugins/amoeba/platforms/cuda/CMakeLists.txt:#   OpenMMAmoebaCUDA.lib
plugins/amoeba/platforms/cuda/CMakeLists.txt:#   libOpenMMAmoebaCUDA.so
plugins/amoeba/platforms/cuda/CMakeLists.txt:SET(OPENMMAMOEBACUDA_LIBRARY_NAME OpenMMAmoebaCUDA)
plugins/amoeba/platforms/cuda/CMakeLists.txt:SET(SHARED_TARGET ${OPENMMAMOEBACUDA_LIBRARY_NAME})
plugins/amoeba/platforms/cuda/CMakeLists.txt:SET(STATIC_TARGET ${OPENMMAMOEBACUDA_LIBRARY_NAME}_static)
plugins/amoeba/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/cuda/include)
plugins/amoeba/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/cuda/src)
plugins/amoeba/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_BINARY_DIR}/platforms/cuda/src)
plugins/amoeba/platforms/cuda/CMakeLists.txt:SET(KERNEL_SOURCE_CLASS CudaAmoebaKernelSources)
plugins/amoeba/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_INCLUDE})
plugins/amoeba/platforms/cuda/CMakeLists.txt:FILE(GLOB CUDA_KERNELS ${KERNEL_SOURCE_DIR}/kernels/*.cu)
plugins/amoeba/platforms/cuda/CMakeLists.txt:    DEPENDS ${CUDA_KERNELS}
plugins/amoeba/platforms/cuda/CMakeLists.txt:    TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_LIBRARY_NAME}CUDA)
plugins/amoeba/platforms/cuda/CMakeLists.txt:        SET_TARGET_PROPERTIES(${SHARED_TARGET} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS} -F/Library/Frameworks -framework CUDA")
plugins/amoeba/platforms/cuda/CMakeLists.txt:    TARGET_LINK_LIBRARIES(${STATIC_TARGET} ${OPENMM_LIBRARY_NAME}CUDA)
plugins/amoeba/platforms/cuda/CMakeLists.txt:        SET_TARGET_PROPERTIES(${STATIC_TARGET} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS} -F/Library/Frameworks -framework CUDA")
plugins/amoeba/platforms/cuda/CMakeLists.txt:# Ensure that links to the main CUDA library will be resolved.
plugins/amoeba/platforms/cuda/CMakeLists.txt:    SET(CUDA_LIBRARY libOpenMMCUDA.dylib)
plugins/amoeba/platforms/cuda/CMakeLists.txt:    INSTALL(CODE "EXECUTE_PROCESS(COMMAND install_name_tool -change ${CUDA_LIBRARY} @loader_path/${CUDA_LIBRARY} ${CMAKE_INSTALL_PREFIX}/lib/plugins/lib${SHARED_TARGET}.dylib)")
plugins/amoeba/platforms/cuda/CMakeLists.txt:if(BUILD_TESTING AND OPENMM_BUILD_CUDA_TESTS)
plugins/amoeba/platforms/cuda/CMakeLists.txt:endif(BUILD_TESTING AND OPENMM_BUILD_CUDA_TESTS)
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:#include "AmoebaCudaKernels.h"
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:#include "CudaAmoebaKernelSources.h"
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:#include "CudaBondedUtilities.h"
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:#include "CudaFFT3D.h"
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:#include "CudaForceInfo.h"
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:#include "CudaKernelSources.h"
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:    if (result != CUDA_SUCCESS) { \
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:CudaCalcAmoebaMultipoleForceKernel::~CudaCalcAmoebaMultipoleForceKernel() {
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:void CudaCalcAmoebaMultipoleForceKernel::initialize(const System& system, const AmoebaMultipoleForce& force) {
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:void CudaCalcAmoebaMultipoleForceKernel::computeFFT(bool forward) {
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:    CudaArray& grid1 = dynamic_cast<CudaContext&>(cc).unwrap(pmeGrid1);
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:    CudaArray& grid2 = dynamic_cast<CudaContext&>(cc).unwrap(pmeGrid2);
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:CudaCalcHippoNonbondedForceKernel::~CudaCalcHippoNonbondedForceKernel() {
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:void CudaCalcHippoNonbondedForceKernel::initialize(const System& system, const HippoNonbondedForce& force) {
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:        CudaContext& cu = dynamic_cast<CudaContext&>(cc);
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:        sort = new CudaSort(cu, new SortTrait(), cc.getNumAtoms());
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:void CudaCalcHippoNonbondedForceKernel::computeFFT(bool forward, bool dispersion) {
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:    CudaArray& grid1 = dynamic_cast<CudaContext&>(cc).unwrap(pmeGrid1);
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:    CudaArray& grid2 = dynamic_cast<CudaContext&>(cc).unwrap(pmeGrid2);
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:void CudaCalcHippoNonbondedForceKernel::sortGridIndex() {
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.cpp:    sort->sort(dynamic_cast<CudaContext&>(cc).unwrap(pmeAtomGridIndex));
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.h:#ifndef AMOEBA_OPENMM_CUDAKERNELS_H_
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.h:#define AMOEBA_OPENMM_CUDAKERNELS_H_
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.h:#include "CudaContext.h"
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.h:#include "CudaNonbondedUtilities.h"
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.h:#include "CudaSort.h"
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.h:class CudaCalcAmoebaMultipoleForceKernel : public CommonCalcAmoebaMultipoleForceKernel {
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.h:    CudaCalcAmoebaMultipoleForceKernel(const std::string& name, const Platform& platform, CudaContext& cu, const System& system) :
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.h:    ~CudaCalcAmoebaMultipoleForceKernel();
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.h:class CudaCalcHippoNonbondedForceKernel : public CommonCalcHippoNonbondedForceKernel {
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.h:    CudaCalcHippoNonbondedForceKernel(const std::string& name, const Platform& platform, CudaContext& cu, const System& system) :
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.h:    ~CudaCalcHippoNonbondedForceKernel();
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.h:    class SortTrait : public CudaSort::SortTrait {
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.h:    CudaSort* sort;
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernels.h:#endif /*AMOEBA_OPENMM_CUDAKERNELS_H*/
plugins/amoeba/platforms/cuda/src/CudaAmoebaKernelSources.h.in:#ifndef OPENMM_CUDAAMOEBAKERNELSOURCES_H_
plugins/amoeba/platforms/cuda/src/CudaAmoebaKernelSources.h.in:#define OPENMM_CUDAAMOEBAKERNELSOURCES_H_
plugins/amoeba/platforms/cuda/src/CudaAmoebaKernelSources.h.in: * This class is a central holding place for the source code of CUDA kernels.
plugins/amoeba/platforms/cuda/src/CudaAmoebaKernelSources.h.in:class OPENMM_EXPORT CudaAmoebaKernelSources {
plugins/amoeba/platforms/cuda/src/CudaAmoebaKernelSources.h.in:#endif /*OPENMM_CUDAAMOEBAKERNELSOURCES_H_*/
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernelFactory.cpp:#include "AmoebaCudaKernelFactory.h"
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernelFactory.cpp:#include "AmoebaCudaKernels.h"
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernelFactory.cpp:#include "CudaPlatform.h"
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernelFactory.cpp:        Platform& platform = Platform::getPlatformByName("CUDA");
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernelFactory.cpp:        AmoebaCudaKernelFactory* factory = new AmoebaCudaKernelFactory();
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernelFactory.cpp:        // Ignore.  The CUDA platform isn't available.
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernelFactory.cpp:extern "C" OPENMM_EXPORT void registerAmoebaCudaKernelFactories() {
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernelFactory.cpp:        Platform::getPlatformByName("CUDA");
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernelFactory.cpp:        Platform::registerPlatform(new CudaPlatform());
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernelFactory.cpp:KernelImpl* AmoebaCudaKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernelFactory.cpp:    CudaPlatform::PlatformData& data = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData());
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernelFactory.cpp:    CudaContext& cu = *data.contexts[0];
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernelFactory.cpp:        return new CudaCalcAmoebaMultipoleForceKernel(name, platform, cu, context.getSystem());
plugins/amoeba/platforms/cuda/src/AmoebaCudaKernelFactory.cpp:        return new CudaCalcHippoNonbondedForceKernel(name, platform, cu, context.getSystem());
plugins/amoeba/platforms/cuda/src/kernels/multipolePme.cu:#if __CUDA_ARCH__ < 500
plugins/amoeba/platforms/cuda/src/kernels/multipolePme.cu:#if __CUDA_ARCH__ < 500
plugins/amoeba/platforms/cuda/src/kernels/multipolePme.cu:#if __CUDA_ARCH__ < 500
plugins/amoeba/platforms/cuda/src/kernels/multipolePme.cu:#if __CUDA_ARCH__ < 500
plugins/amoeba/platforms/cuda/src/CudaAmoebaKernelSources.cpp.in:#include "CudaAmoebaKernelSources.h"
openmmapi/include/openmm/internal/ContextImpl.h:     * values as this one.  With the CUDA and OpenCL platforms, it also shares the same GPU context, allowing data
openmmapi/include/openmm/internal/ContextImpl.h:     * to be transferred between them without leaving the GPU.
openmmapi/include/openmm/internal/CustomCPPForceImpl.h: * code that can run on a GPU and interact directly with internal data structures
openmmapi/src/LocalEnergyMinimizer.cpp:        checkLargeForces = (platformName == "CUDA" || platformName == "OpenCL" || platformName == "HIP" || platformName == "Metal");
openmmapi/src/LocalEnergyMinimizer.cpp:        // The CUDA, OpenCL and HIP platforms accumulate forces in fixed point, so they
devtools/CI-README.md:OpenMM can be described as a C++ library with wrappers available in different programming languages (Python, C, Fortran). The heavy lifting is performed by the backend platforms, which can be based on CPU, CUDA and/or OpenCL (and possibly more in the future). All of this is supported for different operating systems and architectures. As a result, the CI setup can get a bit involved, but this document will try to clarify how it works and what we support.
devtools/CI-README.md:OpenMM's CI runs mainly on GitHub Actions, with one separate Jenkins box running the GPU tests (generously provided by Jason Swails).
devtools/CI-README.md:- CUDA versions
devtools/CI-README.md:- OpenCL implementations
devtools/CI-README.md:  - Nvidia (tested along CUDA)
devtools/CI-README.md:- Steps can be run or skipped based on conditions expressed inside an `if:` key. This is how we control whether we need to install CUDA or not, for example. Jobs can have `if` check, if needed.
devtools/CI-README.md:- Depending on the matrix configuration, we also install CUDA and/or AMD's OpenCL. These conditional steps are evaluated using GHA's builtin `if` mechanism. Ideally we would install this within the conda environment, but sometimes they are not available (licensing issues, etc(), so we delegate that to the system packages or vendor installers.
devtools/CI-README.md:  - For CUDA, we check whether `cuda-version` is not empty, and pass it to `devtools/ci/gh-actions/scripts/install_cuda.sh` as an environment variable.
devtools/CI-README.md:  - For OpenCL, we check whether `OPENCL` is `true` and run `devtools/ci/gh-actions/scripts/install_amd_opencl.sh`. This relies on a installer located in a S3 bucket. This could be refactored to install different OpenCL implementations (ROCm, Intel, etc).
devtools/CI-README.md:- Neither CUDA nor OpenCL installation scripts are run. Instead, we download and install the 10.9 SDK using `devtools/ci/gh-actions/scripts/install_macos_sdk.sh`. This is done so we can mimic what Conda Forge does in their feedstocks. Check the scripts comments for more info.
devtools/CI-README.md:- Installs CUDA with the Nvidia installers using `devtools/ci/gh-actions/scripts/install_cuda.bat`, which requires an environment variable `CUDA_VERSION`, exported from the corresponding matrix entry. Again, this only runs if `matrix.cuda-version` is not empty.
devtools/CI-README.md:- These run on a Docker image on top of `ubuntu-latest`. The Docker image itself depends on the architecture chosen (ppc64le, aarch64) and what CUDA version we want. These are provided by Conda Forge, so they have `conda` preinstalled and ready to go.
devtools/CI-README.md:  - We don't need to install CUDA or setup Miniconda, because they are preinstalled in the Docker image.
devtools/ci/jenkins/install.sh:if [ ! -z "$OPENMM_CUDA_COMPILER" ]; then
devtools/ci/jenkins/install.sh:    echo "Using nvcc ($OPENMM_CUDA_COMPILER) version:"
devtools/ci/jenkins/install.sh:    $OPENMM_CUDA_COMPILER --version
devtools/ci/jenkins/install.sh:    CUDA_ARGS="-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} -DOPENMM_BUILD_CUDA_LIB=true"
devtools/ci/jenkins/install.sh:      -DSWIG_EXECUTABLE=`which swig` $CUDA_ARGS $EXTRA_CMAKE_ARGS .
devtools/ci/jenkins/install_and_test_cpu.sh:EXTRA_CMAKE_ARGS="-DOPENMM_BUILD_CUDA_LIB=false -DOPENMM_BUILD_OPENCL_LIB=false"
devtools/ci/gh-actions/start_docker_locally.sh:# This is the image for PowerPC + CUDA
devtools/ci/gh-actions/start_docker_locally.sh:export DOCKER_IMAGE="quay.io/condaforge/linux-anvil-ppc64le-cuda:10.2"
devtools/ci/gh-actions/conda-envs/build-windows-latest.yml:- khronos-opencl-icd-loader
devtools/ci/gh-actions/conda-envs/build-ubuntu-latest-hip.yml:- rocm-cmake
devtools/ci/gh-actions/conda-envs/build-ubuntu-latest-hip.yml:- rocm-device-libs
devtools/ci/gh-actions/scripts/install_cuda.sh:# This script install CUDA on Ubuntu-based systems
devtools/ci/gh-actions/scripts/install_cuda.sh:# It uses the Nvidia repos for Ubuntu 22.04
devtools/ci/gh-actions/scripts/install_cuda.sh:# It expects a $CUDA_VERSION environment variable set to major.minor (e.g. 10.0)
devtools/ci/gh-actions/scripts/install_cuda.sh:wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
devtools/ci/gh-actions/scripts/install_cuda.sh:sudo dpkg -i cuda-keyring_1.1-1_all.deb
devtools/ci/gh-actions/scripts/install_cuda.sh:CUDA_APT=${CUDA_VERSION/./-}
devtools/ci/gh-actions/scripts/install_cuda.sh:    libgl1-mesa-dev cuda-compiler-${CUDA_APT} \
devtools/ci/gh-actions/scripts/install_cuda.sh:    cuda-drivers cuda-driver-dev-${CUDA_APT} \
devtools/ci/gh-actions/scripts/install_cuda.sh:    cuda-cudart-${CUDA_APT} cuda-cudart-dev-${CUDA_APT} \
devtools/ci/gh-actions/scripts/install_cuda.sh:    libcufft-${CUDA_APT} libcufft-dev-${CUDA_APT} \
devtools/ci/gh-actions/scripts/install_cuda.sh:    cuda-nvrtc-${CUDA_APT} cuda-nvrtc-dev-${CUDA_APT} \
devtools/ci/gh-actions/scripts/install_cuda.sh:    cuda-nvprof-${CUDA_APT} cuda-profiler-api-${CUDA_APT}
devtools/ci/gh-actions/scripts/install_cuda.sh:export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
devtools/ci/gh-actions/scripts/install_cuda.sh:export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
devtools/ci/gh-actions/scripts/install_cuda.sh:export PATH=${CUDA_HOME}/bin:${PATH}
devtools/ci/gh-actions/scripts/install_cuda.sh:echo "CUDA_HOME=${CUDA_HOME}" >> ${GITHUB_ENV}
devtools/ci/gh-actions/scripts/install_cuda.bat::: This script installs CUDA on Windows.
devtools/ci/gh-actions/scripts/install_cuda.bat::: It downloads the offline installer from the Nvidia servers
devtools/ci/gh-actions/scripts/install_cuda.bat::: It uses the default installation path, which is exported as CUDA_PATH
devtools/ci/gh-actions/scripts/install_cuda.bat::: For CMake compatibility, CUDA_TOOLKIT_ROOT_DIR is also exported
devtools/ci/gh-actions/scripts/install_cuda.bat::: It expects a %CUDA_VERSION% environment variable, set to major.minor (e.g. 10.0)
devtools/ci/gh-actions/scripts/install_cuda.bat::: https://docs.nvidia.com/cuda/archive/%CUDA_VERSION%/cuda-installation-guide-microsoft-windows/index.html
devtools/ci/gh-actions/scripts/install_cuda.bat:set "VAR=nvcc_%CUDA_VERSION% cuobjdump_%CUDA_VERSION% nvprune_%CUDA_VERSION% cupti_%CUDA_VERSION%"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "VAR=%VAR% memcheck_%CUDA_VERSION% nvdisasm_%CUDA_VERSION% nvprof_%CUDA_VERSION% cublas_%CUDA_VERSION%"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "VAR=%VAR% cublas_dev_%CUDA_VERSION% cudart_%CUDA_VERSION% cufft_%CUDA_VERSION% cufft_dev_%CUDA_VERSION%"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "VAR=%VAR% curand_%CUDA_VERSION% curand_dev_%CUDA_VERSION% cusolver_%CUDA_VERSION% cusolver_dev_%CUDA_VERSION%"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "VAR=%VAR% cusparse_%CUDA_VERSION% cusparse_dev_%CUDA_VERSION% npp_%CUDA_VERSION% npp_dev_%CUDA_VERSION%"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "VAR=%VAR% nvrtc_%CUDA_VERSION% nvrtc_dev_%CUDA_VERSION% nvml_dev_%CUDA_VERSION%"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "VAR=%VAR% visual_studio_integration_%CUDA_VERSION%"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_COMPONENTS=%VAR%"
devtools/ci/gh-actions/scripts/install_cuda.bat:if "%CUDA_VERSION%" == "9.2"  goto cuda92
devtools/ci/gh-actions/scripts/install_cuda.bat:if "%CUDA_VERSION%" == "10.0" goto cuda100
devtools/ci/gh-actions/scripts/install_cuda.bat:if "%CUDA_VERSION%" == "10.1" goto cuda101
devtools/ci/gh-actions/scripts/install_cuda.bat:if "%CUDA_VERSION%" == "10.2" goto cuda102
devtools/ci/gh-actions/scripts/install_cuda.bat:if "%CUDA_VERSION%" == "11.0" goto cuda110
devtools/ci/gh-actions/scripts/install_cuda.bat:if "%CUDA_VERSION%" == "11.1" goto cuda111
devtools/ci/gh-actions/scripts/install_cuda.bat:if "%CUDA_VERSION%" == "11.2" goto cuda112
devtools/ci/gh-actions/scripts/install_cuda.bat:echo CUDA '%CUDA_VERSION%' is not supported
devtools/ci/gh-actions/scripts/install_cuda.bat::cuda92
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_NETWORK_INSTALLER_URL=https://developer.nvidia.com/compute/cuda/9.2/Prod2/network_installers2/cuda_9.2.148_win10_network"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_NETWORK_INSTALLER_CHECKSUM=2bf9ae67016867b68f361bf50d2b9e7b"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_INSTALLER_URL=https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers2/cuda_9.2.148_win10"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_INSTALLER_CHECKSUM=f6c170a7452098461070dbba3e6e58f1"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_PATCH_URL=https://developer.nvidia.com/compute/cuda/9.2/Prod2/patches/1/cuda_9.2.148.1_windows"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_PATCH_CHECKSUM=09e20653f1346d2461a9f8f1a7178ba2"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_COMPONENTS=%CUDA_COMPONENTS% nvgraph_%CUDA_VERSION% nvgraph_dev_%CUDA_VERSION%"
devtools/ci/gh-actions/scripts/install_cuda.bat:goto cuda_common
devtools/ci/gh-actions/scripts/install_cuda.bat::cuda100
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_NETWORK_INSTALLER_URL=https://developer.nvidia.com/compute/cuda/10.0/Prod/network_installers/cuda_10.0.130_win10_network"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_NETWORK_INSTALLER_CHECKSUM=3312deac9c939bd78d0e7555606c22fc"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_INSTALLER_URL=https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_411.31_win10"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_INSTALLER_CHECKSUM=90fafdfe2167ac25432db95391ca954e"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_COMPONENTS=%CUDA_COMPONENTS% nvgraph_%CUDA_VERSION% nvgraph_dev_%CUDA_VERSION%"
devtools/ci/gh-actions/scripts/install_cuda.bat:goto cuda_common
devtools/ci/gh-actions/scripts/install_cuda.bat::cuda101
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_NETWORK_INSTALLER_URL=http://developer.download.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.243_win10_network.exe"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_NETWORK_INSTALLER_CHECKSUM=fae0c958440511576691b825d4599e93"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_INSTALLER_URL=http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_426.00_win10.exe"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_INSTALLER_CHECKSUM=b54cf32683f93e787321dcc2e692ff69"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_COMPONENTS=%CUDA_COMPONENTS% nvgraph_%CUDA_VERSION% nvgraph_dev_%CUDA_VERSION%"
devtools/ci/gh-actions/scripts/install_cuda.bat:goto cuda_common
devtools/ci/gh-actions/scripts/install_cuda.bat::cuda102
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_NETWORK_INSTALLER_URL=http://developer.download.nvidia.com/compute/cuda/10.2/Prod/network_installers/cuda_10.2.89_win10_network.exe"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_NETWORK_INSTALLER_CHECKSUM=60e0f16845d731b690179606f385041e"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_INSTALLER_URL=http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_441.22_win10.exe"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_INSTALLER_CHECKSUM=d9f5b9f24c3d3fc456a3c789f9b43419"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_PATCH_URL=http://developer.download.nvidia.com/compute/cuda/10.2/Prod/patches/1/cuda_10.2.1_win10.exe"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_PATCH_CHECKSUM=9d751ae129963deb7202f1d85149c69d"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_COMPONENTS=%CUDA_COMPONENTS% nvgraph_%CUDA_VERSION% nvgraph_dev_%CUDA_VERSION%"
devtools/ci/gh-actions/scripts/install_cuda.bat:goto cuda_common
devtools/ci/gh-actions/scripts/install_cuda.bat::cuda110
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_NETWORK_INSTALLER_URL=http://developer.download.nvidia.com/compute/cuda/11.0.3/network_installers/cuda_11.0.3_win10_network.exe"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_NETWORK_INSTALLER_CHECKSUM=1b88bf7bb8e50207bbb53ed2033f93f3"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_INSTALLER_URL=http://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_451.82_win10.exe"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_INSTALLER_CHECKSUM=80ae0fdbe04759123f3cab81f2aadabd"
devtools/ci/gh-actions/scripts/install_cuda.bat:goto cuda_common
devtools/ci/gh-actions/scripts/install_cuda.bat::cuda111
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_NETWORK_INSTALLER_URL=https://developer.download.nvidia.com/compute/cuda/11.1.1/network_installers/cuda_11.1.1_win10_network.exe"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_NETWORK_INSTALLER_CHECKSUM=7e36e50ee486a84612adfd85500a9971"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_INSTALLER_URL=https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_456.81_win10.exe"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_INSTALLER_CHECKSUM=a89dfad35fc1adf02a848a9c06cfff15"
devtools/ci/gh-actions/scripts/install_cuda.bat:goto cuda_common
devtools/ci/gh-actions/scripts/install_cuda.bat::cuda112
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_NETWORK_INSTALLER_URL=https://developer.download.nvidia.com/compute/cuda/11.2.0/network_installers/cuda_11.2.0_win10_network.exe"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_NETWORK_INSTALLER_CHECKSUM=ab02a25eed1201cc3e414be943a242df"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_INSTALLER_URL=https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.89_win10.exe"
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_INSTALLER_CHECKSUM=92f38c37ce9c6c11d27c10701b040256"
devtools/ci/gh-actions/scripts/install_cuda.bat:goto cuda_common
devtools/ci/gh-actions/scripts/install_cuda.bat::cuda_common
devtools/ci/gh-actions/scripts/install_cuda.bat:::We expect this CUDA_PATH
devtools/ci/gh-actions/scripts/install_cuda.bat:set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION%"
devtools/ci/gh-actions/scripts/install_cuda.bat:echo Downloading CUDA version %CUDA_VERSION% installer from %CUDA_INSTALLER_URL%
devtools/ci/gh-actions/scripts/install_cuda.bat:echo Expected MD5: %CUDA_INSTALLER_CHECKSUM%
devtools/ci/gh-actions/scripts/install_cuda.bat:curl --retry 3 -k -L %CUDA_INSTALLER_URL% --output cuda_installer.exe
devtools/ci/gh-actions/scripts/install_cuda.bat:openssl md5 cuda_installer.exe | findstr %CUDA_INSTALLER_CHECKSUM%
devtools/ci/gh-actions/scripts/install_cuda.bat:start /wait cuda_installer.exe -s %CUDA_COMPONENTS%
devtools/ci/gh-actions/scripts/install_cuda.bat:    echo Problem installing CUDA toolkit...
devtools/ci/gh-actions/scripts/install_cuda.bat:del cuda_installer.exe
devtools/ci/gh-actions/scripts/install_cuda.bat:if not "%CUDA_PATCH_URL%"=="" (
devtools/ci/gh-actions/scripts/install_cuda.bat:    curl --retry 3 -k -L %CUDA_PATCH_URL% --output cuda_patch.exe
devtools/ci/gh-actions/scripts/install_cuda.bat:    openssl md5 cuda_patch.exe | findstr %CUDA_PATCH_CHECKSUM%
devtools/ci/gh-actions/scripts/install_cuda.bat:    start /wait cuda_patch.exe -s
devtools/ci/gh-actions/scripts/install_cuda.bat:    del cuda_patch.exe
devtools/ci/gh-actions/scripts/install_cuda.bat:if not exist "%CUDA_PATH%\bin\nvcc.exe" (
devtools/ci/gh-actions/scripts/install_cuda.bat:    echo CUDA toolkit installation failed!
devtools/ci/gh-actions/scripts/install_cuda.bat:echo CUDA_PATH=%CUDA_PATH% >> %GITHUB_ENV%
devtools/ci/gh-actions/scripts/install_cuda.bat:echo CUDA_TOOLKIT_ROOT_DIR=%CUDA_PATH:\=/% >> %GITHUB_ENV%
devtools/ci/gh-actions/scripts/install_cuda.bat::: Notes about nvcuda.dll
devtools/ci/gh-actions/scripts/install_cuda.bat::: We should also provide the drivers (nvcuda.dll), but the installer will not
devtools/ci/gh-actions/scripts/install_cuda.bat::: proceed without a physical Nvidia card attached (not the case in the CI).
devtools/ci/gh-actions/scripts/install_cuda.bat::: Expanding `<installer.exe>\Display.Driver\nvcuda.64.dl_` to `C:\Windows\System32`
devtools/ci/gh-actions/scripts/install_cuda.bat::: ncvuda.dll in a GPU-less machine without breaking the EULA (aka zipping nvcuda.dll
devtools/ci/gh-actions/scripts/install_amd_opencl.sh:# This script installs AMD's SDK 3.0 to provide their OpenCL implementation
devtools/ci/gh-actions/scripts/install_amd_opencl.sh:export OPENCL_VENDOR_PATH=${AMDAPPSDK}/etc/OpenCL/vendors
devtools/ci/gh-actions/scripts/install_amd_opencl.sh:mkdir -p ${OPENCL_VENDOR_PATH}
devtools/ci/gh-actions/scripts/install_amd_opencl.sh:echo libamdocl64.so > ${OPENCL_VENDOR_PATH}/amdocl64.icd
devtools/ci/gh-actions/scripts/install_amd_opencl.sh:echo "OPENCL_VENDOR_PATH=${OPENCL_VENDOR_PATH}" >> ${GITHUB_ENV}
devtools/ci/gh-actions/scripts/run_steps_inside_docker_image.sh:if [[ -d /usr/local/cuda ]]; then
devtools/ci/gh-actions/scripts/run_steps_inside_docker_image.sh:    export CUDA_PATH="/usr/local/cuda"
devtools/ci/gh-actions/scripts/run_steps_inside_docker_image.sh:    export CUDA_LIB_PATH="${CUDA_PATH}/lib64/stubs"
devtools/ci/gh-actions/scripts/run_steps_inside_docker_image.sh:    export LD_LIBRARY_PATH="${CUDA_PATH}/lib64/stubs:${LD_LIBRARY_PATH:-}"
devtools/ci/gh-actions/scripts/run_steps_inside_docker_image.sh:    export PATH="${CUDA_PATH}/bin:${PATH}"
devtools/ci/gh-actions/scripts/run_steps_inside_docker_image.sh:    -DOPENMM_BUILD_CUDA_TESTS=OFF \
devtools/ci/gh-actions/scripts/run_steps_inside_docker_image.sh:    -DOPENMM_BUILD_OPENCL_TESTS=OFF
devtools/ci/gh-actions/scripts/run_steps_inside_docker_image.sh:if [[ ! -z ${CUDA_VER} ]]; then
devtools/ci/gh-actions/scripts/run_steps_inside_docker_image.sh:    test -f ${CONDA_PREFIX}/lib/plugins/libOpenMMCUDA.so
devtools/ci/gh-actions/scripts/run_steps_inside_docker_image.sh:    test -f ${CONDA_PREFIX}/lib/plugins/libOpenMMOpenCL.so
devtools/packaging/README.md:docker run -i -t --rm -e TAG -v `pwd`:/io jchodera/omnia-build-box:cuda80-amd30-clang38 bash
devtools/packaging/README.md:docker run -i -t --rm -e TAG -v `pwd`:/io jchodera/omnia-build-box:cuda80-amd30-clang38 bash
devtools/packaging/scripts/windows/prepare.ps1:# Install CUDA.
devtools/packaging/scripts/windows/prepare.ps1:wget https://developer.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.168_win10_network.exe -UseBasicParsing -OutFile cuda_10.1.168_win10_network.exe
devtools/packaging/scripts/windows/prepare.ps1:.\cuda_10.1.168_win10_network.exe -s nvcc_10.1 cudart_10.1 cufft_10.1 cufft_dev_10.1 nvrtc_10.1 nvrtc_dev_10.1 | Out-Null
devtools/packaging/scripts/windows/build.bat:    -DOPENCL_INCLUDE_DIR="%APPSDK%\include" -DOPENCL_LIBRARY="%APPSDK%\lib\x86_64\OpenCL.lib"
devtools/packaging/scripts/linux/build.sh:# Use NVIDIA CUDA 8.0
devtools/packaging/scripts/linux/build.sh:CMAKE_FLAGS+=" -DCUDA_CUDART_LIBRARY=/usr/local/cuda-8.0/lib64/libcudart.so"
devtools/packaging/scripts/linux/build.sh:CMAKE_FLAGS+=" -DCUDA_NVCC_EXECUTABLE=/usr/local/cuda-8.0/bin/nvcc"
devtools/packaging/scripts/linux/build.sh:CMAKE_FLAGS+=" -DCUDA_SDK_ROOT_DIR=/usr/local/cuda-8.0/"
devtools/packaging/scripts/linux/build.sh:CMAKE_FLAGS+=" -DCUDA_TOOLKIT_INCLUDE=/usr/local/cuda-8.0/include"
devtools/packaging/scripts/linux/build.sh:CMAKE_FLAGS+=" -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0/"
devtools/packaging/scripts/linux/build.sh:CMAKE_FLAGS+=" -DOPENCL_INCLUDE_DIR=/opt/AMDAPPSDK-3.0/include/"
devtools/packaging/scripts/linux/build.sh:CMAKE_FLAGS+=" -DOPENCL_LIBRARY=/opt/AMDAPPSDK-3.0/lib/x86_64/libOpenCL.so"
devtools/packaging/scripts/linux/build.sh:CMAKE_FLAGS+=" -DCMAKE_CXX_FLAGS_RELEASE=-I/usr/include/nvidia/"
devtools/packaging/scripts/source/build.sh:# Use NVIDIA CUDA 8.0
devtools/packaging/scripts/source/build.sh:CMAKE_FLAGS+=" -DCUDA_CUDART_LIBRARY=/usr/local/cuda-8.0/lib64/libcudart.so"
devtools/packaging/scripts/source/build.sh:CMAKE_FLAGS+=" -DCUDA_NVCC_EXECUTABLE=/usr/local/cuda-8.0/bin/nvcc"
devtools/packaging/scripts/source/build.sh:CMAKE_FLAGS+=" -DCUDA_SDK_ROOT_DIR=/usr/local/cuda-8.0/"
devtools/packaging/scripts/source/build.sh:CMAKE_FLAGS+=" -DCUDA_TOOLKIT_INCLUDE=/usr/local/cuda-8.0/include"
devtools/packaging/scripts/source/build.sh:CMAKE_FLAGS+=" -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0/"
devtools/packaging/scripts/source/build.sh:CMAKE_FLAGS+=" -DOPENCL_INCLUDE_DIR=/opt/AMDAPPSDK-3.0/include/"
devtools/packaging/scripts/source/build.sh:CMAKE_FLAGS+=" -DOPENCL_LIBRARY=/opt/AMDAPPSDK-3.0/lib/x86_64/libOpenCL.so"
devtools/packaging/scripts/source/build.sh:CMAKE_FLAGS+=" -DCMAKE_CXX_FLAGS_RELEASE=-I/usr/include/nvidia/"
devtools/Jenkinsfile:                stage("Build and test CUDA platform") {
devtools/Jenkinsfile:                            label "cuda && docker"
devtools/Jenkinsfile:                            args '--gpus all'
devtools/Jenkinsfile:                        sh "devtools/ci/jenkins/test.sh -R 'TestCuda' --parallel 2"
devtools/Jenkinsfile:                stage("Build and test OpenCL platform") {
devtools/Jenkinsfile:                            label "cuda && docker"
devtools/Jenkinsfile:                            args '--gpus all'
devtools/Jenkinsfile:                        sh "devtools/ci/jenkins/test.sh -R 'TestOpenCL' --parallel 2"
libraries/vkfft/include/vkFFT.h:#include <cuda.h>
libraries/vkfft/include/vkFFT.h:#include <cuda_runtime.h>
libraries/vkfft/include/vkFFT.h:#include <cuda_runtime_api.h>
libraries/vkfft/include/vkFFT.h:#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
libraries/vkfft/include/vkFFT.h:#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
libraries/vkfft/include/vkFFT.h:#include <OpenCL/opencl.h>
libraries/vkfft/include/vkFFT.h:	CUdevice* device;//pointer to CUDA device, obtained from cuDeviceGet
libraries/vkfft/include/vkFFT.h:	//CUcontext* context;//pointer to CUDA context, obtained from cuDeviceGet
libraries/vkfft/include/vkFFT.h:	cudaStream_t* stream;//pointer to streams (can be more than 1), where to execute the kernels
libraries/vkfft/include/vkFFT.h:	uint64_t num_streams;//try to submit CUDA kernels in multiple streams for asynchronous execution. Default 1
libraries/vkfft/include/vkFFT.h:	uint64_t coalescedMemory;//in bytes, for Nvidia and AMD is equal to 32, Intel is equal 64, scaled for half precision. Gonna work regardles, but if specified by user correctly, the performance will be higher.
libraries/vkfft/include/vkFFT.h:	uint64_t fixMaxRadixBluestein;//controls the padding of sequences in Bluestein convolution. If specified, padded sequence will be made of up to fixMaxRadixBluestein primes. Default: 2 for CUDA and Vulkan/OpenCL/HIP up to 1048576 combined dimension FFT system, 7 for Vulkan/OpenCL/HIP past after. Min = 2, Max = 13.
libraries/vkfft/include/vkFFT.h:	uint64_t fixMinRaderPrimeFFT;//start FFT convolution version of Rader for radix primes from this number. Better than direct multiplication version for almost all primes (except small ones, like 17-23 on some GPUs). Must be bigger or equal to fixMinRaderPrimeMult. Deafult 29 on AMD and 17 on other GPUs. 
libraries/vkfft/include/vkFFT.h:	uint64_t fixMaxRaderPrimeFFT;//switch to Bluestein's algorithm for radix primes from this number. Switch may happen earlier if prime can't fit in shared memory. Default is 16384, which is bigger than most current GPU's shared memory.
libraries/vkfft/include/vkFFT.h:	uint64_t registerBoost; //specify if register file size is bigger than shared memory and can be used to extend it X times (on Nvidia 256KB register file can be used instead of 32KB of shared memory, set this constant to 4 to emulate 128KB of shared memory). Default 1
libraries/vkfft/include/vkFFT.h:	uint64_t devicePageSize;//in KB, the size of a page on the GPU. Setting to 0 disables local buffer split in pages
libraries/vkfft/include/vkFFT.h:	uint64_t computeCapabilityMajor; // CUDA/HIP compute capability of the device
libraries/vkfft/include/vkFFT.h:	uint64_t computeCapabilityMinor; // CUDA/HIP compute capability of the device
libraries/vkfft/include/vkFFT.h:	uint64_t vendorID; // vendorID 0x10DE - NVIDIA, 0x8086 - Intel, 0x1002 - AMD, etc.
libraries/vkfft/include/vkFFT.h:	cudaEvent_t* stream_event;//Filled at app creation
libraries/vkfft/include/vkFFT.h:	void* saveApplicationString;//memory array(uint32_t* for Vulkan, char* for CUDA/HIP/OpenCL) through which user can access VkFFT generated binaries. (will be allocated by VkFFT, deallocated with deleteVkFFT call)
libraries/vkfft/include/vkFFT.h:#extension GL_ARB_gpu_shader_fp64 : enable\n\
libraries/vkfft/include/vkFFT.h:#extension GL_ARB_gpu_shader_int64 : enable\n\n");
libraries/vkfft/include/vkFFT.h:#ifdef VKFFT_OLD_ROCM
libraries/vkfft/include/vkFFT.h:#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n");
libraries/vkfft/include/vkFFT.h:#if(!((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))//OpenCL, Level Zero and Metal are  not handling barrier with thread-conditional writes to local memory - so this is a work-around
libraries/vkfft/include/vkFFT.h:#if(!((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))//OpenCL, Level Zero and Metal are  not handling barrier with thread-conditional writes to local memory - so this is a work-around
libraries/vkfft/include/vkFFT.h:						sc->tempLen = sprintf(sc->tempStr, "		if (((combinedID %% %" PRIu64 ")%%2) == 1) {\n", 2 * sc->fftDim);//another OpenCL bugfix
libraries/vkfft/include/vkFFT.h:#if((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5))//OpenCL is not handling barrier with thread-conditional writes to local memory - so this is a work-around
libraries/vkfft/include/vkFFT.h:						sc->tempLen = sprintf(sc->tempStr, "		if (((combinedID %% %" PRIu64 ")%%2) == 1) {\n", 2 * sc->fftDim);//another OpenCL bugfix
libraries/vkfft/include/vkFFT.h:						sc->tempLen = sprintf(sc->tempStr, "		if (((combinedID %% %" PRIu64 ")%%2) == 1) {\n", 2 * sc->fftDim);//another OpenCL bugfix
libraries/vkfft/include/vkFFT.h:#if(!((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))//OpenCL, Level Zero and Metal are  not handling barrier with thread-conditional writes to local memory - so this is a work-around
libraries/vkfft/include/vkFFT.h:#if(!((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))//OpenCL, Level Zero and Metal are  not handling barrier with thread-conditional writes to local memory - so this is a work-around
libraries/vkfft/include/vkFFT.h:#if(!((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))//OpenCL, Level Zero and Metal are  not handling barrier with thread-conditional writes to local memory - so this is a work-around
libraries/vkfft/include/vkFFT.h:#if((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5))//OpenCL is not handling barrier with thread-conditional writes to local memory - so this is a work-around
libraries/vkfft/include/vkFFT.h:#if(!((VKFFT_BACKEND==3)||(VKFFT_BACKEND==4)||(VKFFT_BACKEND==5)))//OpenCL, Level Zero and Metal are  not handling barrier with thread-conditional writes to local memory - so this is a work-around
libraries/vkfft/include/vkFFT.h:	cudaError_t res = cudaSuccess;
libraries/vkfft/include/vkFFT.h:	res = cudaMemcpy(buffer, cpu_arr, transferSize, cudaMemcpyHostToDevice);
libraries/vkfft/include/vkFFT.h:	if (res != cudaSuccess) {
libraries/vkfft/include/vkFFT.h:	cudaError_t res = cudaSuccess;
libraries/vkfft/include/vkFFT.h:	res = cudaMemcpy(cpu_arr, buffer, transferSize, cudaMemcpyDeviceToHost);
libraries/vkfft/include/vkFFT.h:	if (res != cudaSuccess) {
libraries/vkfft/include/vkFFT.h:	CUresult res = CUDA_SUCCESS;
libraries/vkfft/include/vkFFT.h:	cudaError_t res_t = cudaSuccess;
libraries/vkfft/include/vkFFT.h:		res_t = cudaFree(axis->bufferLUT);
libraries/vkfft/include/vkFFT.h:		if (res_t == cudaSuccess) axis->bufferLUT = 0;
libraries/vkfft/include/vkFFT.h:		if (res == CUDA_SUCCESS) axis->VkFFTModule = 0;
libraries/vkfft/include/vkFFT.h:		cudaError_t res_t = cudaSuccess;
libraries/vkfft/include/vkFFT.h:				res_t = cudaEventDestroy(app->configuration.stream_event[i]);
libraries/vkfft/include/vkFFT.h:				if (res_t == cudaSuccess) app->configuration.stream_event[i] = 0;
libraries/vkfft/include/vkFFT.h:			cudaError_t res_t = cudaSuccess;
libraries/vkfft/include/vkFFT.h:				res_t = cudaFree(app->configuration.tempBuffer[0]);
libraries/vkfft/include/vkFFT.h:				if (res_t == cudaSuccess) app->configuration.tempBuffer[0] = 0;
libraries/vkfft/include/vkFFT.h:					cudaError_t res_t = cudaSuccess;
libraries/vkfft/include/vkFFT.h:					res_t = cudaFree(app->bufferRaderUintLUT[i][j]);
libraries/vkfft/include/vkFFT.h:					if (res_t == cudaSuccess) app->bufferRaderUintLUT[i][j] = 0;
libraries/vkfft/include/vkFFT.h:			cudaError_t res_t = cudaSuccess;
libraries/vkfft/include/vkFFT.h:				res_t = cudaFree(app->bufferBluestein[i]);
libraries/vkfft/include/vkFFT.h:				if (res_t == cudaSuccess) app->bufferBluestein[i] = 0;
libraries/vkfft/include/vkFFT.h:				res_t = cudaFree(app->bufferBluesteinFFT[i]);
libraries/vkfft/include/vkFFT.h:				if (res_t == cudaSuccess) app->bufferBluesteinFFT[i] = 0;
libraries/vkfft/include/vkFFT.h:				res_t = cudaFree(app->bufferBluesteinIFFT[i]);
libraries/vkfft/include/vkFFT.h:				if (res_t == cudaSuccess) app->bufferBluesteinIFFT[i] = 0;
libraries/vkfft/include/vkFFT.h:			rader_min_registers = (rader_min_registers / 2 + scale_registers_rader) * 2;//min number of registers for Rader (can be more than min_registers_per_thread, but min_registers_per_thread should be at least 4 for Nvidiaif you have >256 threads)
libraries/vkfft/include/vkFFT.h:	cudaError_t res = cudaSuccess;
libraries/vkfft/include/vkFFT.h:	res = cudaMalloc((void**)&app->bufferBluestein[axis_id], bufferSize);
libraries/vkfft/include/vkFFT.h:	if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
libraries/vkfft/include/vkFFT.h:		res = cudaMalloc((void**)&app->bufferBluesteinFFT[axis_id], bufferSize);
libraries/vkfft/include/vkFFT.h:		if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
libraries/vkfft/include/vkFFT.h:		res = cudaMalloc((void**)&app->bufferBluesteinIFFT[axis_id], bufferSize);
libraries/vkfft/include/vkFFT.h:		if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
libraries/vkfft/include/vkFFT.h:		kernelPreparationConfiguration.queue = app->configuration.queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
libraries/vkfft/include/vkFFT.h:			res = cudaDeviceSynchronize();
libraries/vkfft/include/vkFFT.h:			if (res != cudaSuccess) {
libraries/vkfft/include/vkFFT.h:			res = cudaDeviceSynchronize();
libraries/vkfft/include/vkFFT.h:			if (res != cudaSuccess) {
libraries/vkfft/include/vkFFT.h:			res = cudaDeviceSynchronize();
libraries/vkfft/include/vkFFT.h:			if (res != cudaSuccess) {
libraries/vkfft/include/vkFFT.h:				kernelPreparationConfiguration.queue = app->configuration.queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
libraries/vkfft/include/vkFFT.h:				cudaError_t res = cudaSuccess;
libraries/vkfft/include/vkFFT.h:				res = cudaMalloc(&bufferRaderFFT, bufferSize);
libraries/vkfft/include/vkFFT.h:				if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
libraries/vkfft/include/vkFFT.h:				res = cudaDeviceSynchronize();
libraries/vkfft/include/vkFFT.h:				if (res != cudaSuccess) {
libraries/vkfft/include/vkFFT.h:				cudaFree(bufferRaderFFT);
libraries/vkfft/include/vkFFT.h:	cudaError_t res = cudaSuccess;
libraries/vkfft/include/vkFFT.h:				res = cudaMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
libraries/vkfft/include/vkFFT.h:				if (res != cudaSuccess) {
libraries/vkfft/include/vkFFT.h:				res = cudaMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
libraries/vkfft/include/vkFFT.h:				if (res != cudaSuccess) {
libraries/vkfft/include/vkFFT.h:#if (CUDA_VERSION >= 11030)
libraries/vkfft/include/vkFFT.h:			sprintf(opts[0], "--gpu-architecture=sm_%" PRIu64 "%" PRIu64 "", app->configuration.computeCapabilityMajor, app->configuration.computeCapabilityMinor);
libraries/vkfft/include/vkFFT.h:#if (CUDA_VERSION >= 11030)
libraries/vkfft/include/vkFFT.h:#if (CUDA_VERSION >= 11030)
libraries/vkfft/include/vkFFT.h:#if (CUDA_VERSION >= 11030)
libraries/vkfft/include/vkFFT.h:#if (CUDA_VERSION >= 11030)
libraries/vkfft/include/vkFFT.h:		if (result2 != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:		if (result2 != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:			if (result2 != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:			if (result2 != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:	cudaError_t res = cudaSuccess;
libraries/vkfft/include/vkFFT.h:		res = cudaMalloc(app->configuration.tempBuffer, app->configuration.tempBufferSize[0]);
libraries/vkfft/include/vkFFT.h:		if (res != cudaSuccess) {
libraries/vkfft/include/vkFFT.h:							res = cudaMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
libraries/vkfft/include/vkFFT.h:							if (res != cudaSuccess) {
libraries/vkfft/include/vkFFT.h:							res = cudaMalloc((void**)&axis->bufferLUT, axis->bufferLUTSize);
libraries/vkfft/include/vkFFT.h:							if (res != cudaSuccess) {
libraries/vkfft/include/vkFFT.h:			res = cudaMalloc((void**)&app->bufferRaderUintLUT[axis_id][axis_upload_id], app->bufferRaderUintLUTSize[axis_id][axis_upload_id]);
libraries/vkfft/include/vkFFT.h:			if (res != cudaSuccess) {
libraries/vkfft/include/vkFFT.h:#if (CUDA_VERSION >= 11030)
libraries/vkfft/include/vkFFT.h:			sprintf(opts[0], "--gpu-architecture=sm_%" PRIu64 "%" PRIu64 "", app->configuration.computeCapabilityMajor, app->configuration.computeCapabilityMinor);
libraries/vkfft/include/vkFFT.h:#if (CUDA_VERSION >= 11030)
libraries/vkfft/include/vkFFT.h:#if (CUDA_VERSION >= 11030)
libraries/vkfft/include/vkFFT.h:#if (CUDA_VERSION >= 11030)
libraries/vkfft/include/vkFFT.h:#if (CUDA_VERSION >= 11030)
libraries/vkfft/include/vkFFT.h:		if (result2 != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:		if (result2 != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:		if (result2 != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:			if (result2 != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:			if (result2 != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:		case 0x10DE://NVIDIA
libraries/vkfft/include/vkFFT.h:		case 0x10DE://Nvidia
libraries/vkfft/include/vkFFT.h:	case 0x10DE://NVIDIA
libraries/vkfft/include/vkFFT.h:	CUresult res = CUDA_SUCCESS;
libraries/vkfft/include/vkFFT.h:	cudaError_t res_t = cudaSuccess;
libraries/vkfft/include/vkFFT.h:	if (res != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:	if (res != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:	if (res != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:	if (res != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:	if (res != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:	if (res != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:	if (res != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:	if (res != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:	if (res != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:	if (res != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:	if (res != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:	if (res != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:	if (res != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:	//we don't need this in CUDA
libraries/vkfft/include/vkFFT.h:		app->configuration.stream_event = (cudaEvent_t*)malloc(app->configuration.num_streams * sizeof(cudaEvent_t));
libraries/vkfft/include/vkFFT.h:			res_t = cudaEventCreate(&app->configuration.stream_event[i]);
libraries/vkfft/include/vkFFT.h:			if (res_t != cudaSuccess) {
libraries/vkfft/include/vkFFT.h:	case 0x10DE://NVIDIA
libraries/vkfft/include/vkFFT.h:		//The dummy kernel approach (above) does not work for some DCT-IV kernels (like 256x256x256). They refuse to have more than 256 threads. I will just force OpenCL thread limits for now.
libraries/vkfft/include/vkFFT.h:	case 0x10DE://NVIDIA
libraries/vkfft/include/vkFFT.h:				CUresult result = CUDA_SUCCESS;
libraries/vkfft/include/vkFFT.h:						if (result != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:				if (result != CUDA_SUCCESS) {
libraries/vkfft/include/vkFFT.h:						cudaError_t res2 = cudaEventRecord(app->configuration.stream_event[app->configuration.streamID], app->configuration.stream[app->configuration.streamID]);
libraries/vkfft/include/vkFFT.h:						if (res2 != cudaSuccess) return VKFFT_ERROR_FAILED_TO_EVENT_RECORD;
libraries/vkfft/include/vkFFT.h:		cudaError_t res = cudaSuccess;
libraries/vkfft/include/vkFFT.h:			res = cudaEventSynchronize(app->configuration.stream_event[s]);
libraries/vkfft/include/vkFFT.h:			if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SYNCHRONIZE;
cmake_modules/FindOpenCL.cmake:### OPENCL_INCLUDE_DIR ###
cmake_modules/FindOpenCL.cmake:# Try OPENCL_DIR variable before looking elsewhere
cmake_modules/FindOpenCL.cmake:find_path(OPENCL_INCLUDE_DIR
cmake_modules/FindOpenCL.cmake:    NAMES OpenCL/opencl.h CL/opencl.h
cmake_modules/FindOpenCL.cmake:    PATHS $ENV{OPENCL_DIR}
cmake_modules/FindOpenCL.cmake:# Next look in environment variables set by OpenCL SDK installations
cmake_modules/FindOpenCL.cmake:find_path(OPENCL_INCLUDE_DIR
cmake_modules/FindOpenCL.cmake:    NAMES OpenCL/opencl.h CL/opencl.h
cmake_modules/FindOpenCL.cmake:        $ENV{CUDA_PATH}
cmake_modules/FindOpenCL.cmake:    find_path(OPENCL_INCLUDE_DIR
cmake_modules/FindOpenCL.cmake:        NAMES opencl.h opencl.h
cmake_modules/FindOpenCL.cmake:            "${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/OpenCL.framework/Headers"
cmake_modules/FindOpenCL.cmake:find_path(OPENCL_INCLUDE_DIR
cmake_modules/FindOpenCL.cmake:    NAMES OpenCL/opencl.h CL/opencl.h
cmake_modules/FindOpenCL.cmake:        "C:/CUDA"
cmake_modules/FindOpenCL.cmake:        "/usr/local/cuda"
cmake_modules/FindOpenCL.cmake:        "${CUDA_TOOLKIT_ROOT_DIR}"
cmake_modules/FindOpenCL.cmake:### OPENCL_LIBRARY ###
cmake_modules/FindOpenCL.cmake:# Try OPENCL_DIR variable before looking elsewhere
cmake_modules/FindOpenCL.cmake:find_library(OPENCL_LIBRARY
cmake_modules/FindOpenCL.cmake:    NAMES OpenCL
cmake_modules/FindOpenCL.cmake:      $ENV{OPENCL_DIR}
cmake_modules/FindOpenCL.cmake:      ${OPENCL_LIB_SEARCH_PATH}
cmake_modules/FindOpenCL.cmake:# Next look in environment variables set by OpenCL SDK installations
cmake_modules/FindOpenCL.cmake:find_library(OPENCL_LIBRARY
cmake_modules/FindOpenCL.cmake:    NAMES OpenCL
cmake_modules/FindOpenCL.cmake:      $ENV{CUDA_PATH}
cmake_modules/FindOpenCL.cmake:find_library(OPENCL_LIBRARY
cmake_modules/FindOpenCL.cmake:    NAMES OpenCL
cmake_modules/FindOpenCL.cmake:        "C:/CUDA"
cmake_modules/FindOpenCL.cmake:        "/usr/local/cuda"
cmake_modules/FindOpenCL.cmake:        "${CUDA_TOOLKIT_ROOT_DIR}"
cmake_modules/FindOpenCL.cmake:find_package_handle_standard_args(OpenCL DEFAULT_MSG OPENCL_LIBRARY OPENCL_INCLUDE_DIR)
cmake_modules/FindOpenCL.cmake:if(OPENCL_FOUND)
cmake_modules/FindOpenCL.cmake:  set(OPENCL_LIBRARIES ${OPENCL_LIBRARY})
cmake_modules/FindOpenCL.cmake:  mark_as_advanced(CLEAR OPENCL_INCLUDE_DIR)
cmake_modules/FindOpenCL.cmake:  mark_as_advanced(CLEAR OPENCL_LIBRARY)
cmake_modules/FindOpenCL.cmake:else(OPENCL_FOUND)
cmake_modules/FindOpenCL.cmake:  set(OPENCL_LIBRARIES)
cmake_modules/FindOpenCL.cmake:  mark_as_advanced(OPENCL_INCLUDE_DIR)
cmake_modules/FindOpenCL.cmake:  mark_as_advanced(OPENCL_LIBRARY)
cmake_modules/FindOpenCL.cmake:endif(OPENCL_FOUND)
tests/TestCustomCVForce.h:    // reordering on the GPU.
tests/TestATMForce.h:    // Also add nonbonded forces to trigger atom reordering on the GPU.
README.md:provides a combination of extreme flexibility (through custom forces and integrators), openness, and high performance (especially on recent GPUs) that make it truly unique among simulation codes.  
CMakeLists.txt:# CUDA platform
CMakeLists.txt:FIND_PACKAGE(CUDAToolkit QUIET)
CMakeLists.txt:IF(CUDAToolkit_FOUND)
CMakeLists.txt:    SET(OPENMM_BUILD_CUDA_LIB ON CACHE BOOL "Build OpenMMCuda library for Nvidia GPUs")
CMakeLists.txt:ELSE(CUDAToolkit_FOUND)
CMakeLists.txt:    SET(OPENMM_BUILD_CUDA_LIB OFF CACHE BOOL "Build OpenMMCuda library for Nvidia GPUs")
CMakeLists.txt:ENDIF(CUDAToolkit_FOUND)
CMakeLists.txt:IF(OPENMM_BUILD_CUDA_LIB)
CMakeLists.txt:    ADD_SUBDIRECTORY(platforms/cuda)
CMakeLists.txt:ENDIF(OPENMM_BUILD_CUDA_LIB)
CMakeLists.txt:# OpenCL platform
CMakeLists.txt:FIND_PACKAGE(OpenCL QUIET)
CMakeLists.txt:IF(OPENCL_FOUND)
CMakeLists.txt:    SET(OPENMM_BUILD_OPENCL_LIB ON CACHE BOOL "Build OpenMMOpenCL library")
CMakeLists.txt:ELSE(OPENCL_FOUND)
CMakeLists.txt:    SET(OPENMM_BUILD_OPENCL_LIB OFF CACHE BOOL "Build OpenMMOpenCL library")
CMakeLists.txt:ENDIF(OPENCL_FOUND)
CMakeLists.txt:IF(OPENMM_BUILD_OPENCL_LIB)
CMakeLists.txt:    ADD_SUBDIRECTORY(platforms/opencl)
CMakeLists.txt:ENDIF(OPENMM_BUILD_OPENCL_LIB)
CMakeLists.txt:LIST(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} /opt/rocm)
CMakeLists.txt:    SET(OPENMM_BUILD_HIP_LIB ON CACHE BOOL "Build OpenMMHIP library for AMD GPUs")
CMakeLists.txt:    SET(OPENMM_BUILD_HIP_LIB OFF CACHE BOOL "Build OpenMMHIP library for AMD GPUs")
CMakeLists.txt:SET(OPENMM_BUILD_COMMON OFF CACHE BOOL "Build common files even if CUDA or OpenCL platforms are not built")
CMakeLists.txt:IF(OPENMM_BUILD_CUDA_LIB OR OPENMM_BUILD_OPENCL_LIB OR OPENMM_BUILD_HIP_LIB OR OPENMM_BUILD_COMMON)
olla/include/openmm/kernels.h: * GPU based platforms sometimes use a CPU based implementation provided by a separate
olla/include/openmm/kernels.h: * GPU based platforms sometimes use a CPU based implementation provided by a separate
appveyor.yml:# Download OpenCL Headers and build the ICD loader
appveyor.yml:  - ps: $opencl_registry = "https://www.khronos.org/registry/cl"
appveyor.yml:  - ps: $opencl_github = "KhronosGroup/OpenCL-Headers"
appveyor.yml:  - ps: mkdir C:/opencl > $null
appveyor.yml:  - ps: cd C:/opencl
appveyor.yml:  - ps: wget $opencl_registry/specs/opencl-icd-1.2.11.0.tgz -OutFile opencl-icd-1.2.11.0.tgz
appveyor.yml:  - ps: 7z x opencl-icd-1.2.11.0.tgz > $null
appveyor.yml:  - ps: 7z x opencl-icd-1.2.11.0.tar > $null
appveyor.yml:  - ps: wget https://github.com/$opencl_github/tree/master/CL -UseBasicParsing | select -ExpandProperty links | where {$_.href -like "*.h*"} | select -ExpandProperty title | foreach{ wget https://raw.githubusercontent.com/$opencl_github/master/CL/$_ -OutFile inc/CL/$_ -UseBasicParsing}
appveyor.yml:       -DOPENCL_INCLUDE_DIR=C:/opencl/inc
appveyor.yml:       -DOPENCL_LIBRARY=C:/opencl/lib/OpenCL.lib
appveyor.yml:       -DOPENMM_BUILD_OPENCL_TESTS=OFF
wrappers/python/setup.py:    (especially on recent GPUs) that make it truly unique among simulation codes.
wrappers/python/openmm/app/data/charmm36.xml:    <Residue name="LNCCL1">
wrappers/python/openmm/app/data/charmm36.xml:    <Residue name="LNCCL2">
wrappers/python/openmm/__init__.py:(especially on recent GPUs) that make it truly unique among simulation codes.
wrappers/python/src/swig_doxygen/swigInputConfig.py:                ('CudaKernelFactory',),
wrappers/python/src/swig_doxygen/swigInputConfig.py:                ('CudaPlatform',),
platforms/hip/include/HipContext.h:  - Hip only marginally supports the CUDA context API, and will remove
platforms/hip/include/HipContext.h:     * may be more efficient on CPUs and GPUs.
platforms/hip/include/HipContext.h:    std::string tempDir, cacheDir, gpuArchitecture;
platforms/hip/include/HipSort.h: * Sorting Algorithm with CUDA"  Journal of the Chinese Institute of Engineers, 32(7),
platforms/hip/src/HipContext.cpp:    gpuArchitecture = props.gcnArchName;
platforms/hip/src/HipContext.cpp:    // GPUs starting from CDNA1 and RDNA3 support atomic add for floats (global_atomic_add_f32),
platforms/hip/src/HipContext.cpp:    // which can be used in PME. Older GPUs use fixed point charge spreading instead.
platforms/hip/src/HipContext.cpp:    if (gpuArchitecture.find("gfx900") == 0 ||
platforms/hip/src/HipContext.cpp:        gpuArchitecture.find("gfx906") == 0 ||
platforms/hip/src/HipContext.cpp:        gpuArchitecture.find("gfx10") == 0) {
platforms/hip/src/HipContext.cpp:    // For RDNA GPUs hipDeviceAttributeMultiprocessorCount means WGP (work-group processors, two compute units), not CUs.
platforms/hip/src/HipContext.cpp:    cacheFile << cacheDir << "openmm-hip-" << getHash(src + gpuArchitecture);
platforms/hip/src/HipContext.cpp:    options += " --offload-arch=" + gpuArchitecture;
platforms/hip/src/HipContext.cpp:    if (gpuArchitecture.find("gfx90a") == 0 ||
platforms/hip/src/HipContext.cpp:        gpuArchitecture.find("gfx94") == 0) {
platforms/hip/src/HipContext.cpp:        options += " --gpu-max-threads-per-block=" + std::to_string(getMaxThreadBlockSize());
platforms/hip/src/HipContext.cpp:        // ROCm 6.0 headers. This issue has been fixed in 6.1. hipRTC includes amd_hip_complex.h
platforms/hip/src/kernels/findInteractingBlocks.hip: * [in] maxTiles               - maximum number of tiles to process, used for multi-GPUs
platforms/hip/src/kernels/findInteractingBlocks.hip: * [in] startBlockIndex        - first block to process, used for multi-GPUs,
platforms/hip/src/HipPlatform.cpp:    // so the OpenCL plaform can be selected as default
platforms/opencl/staticTarget/CMakeLists.txt:# Include OpenCL related files.
platforms/opencl/staticTarget/CMakeLists.txt:# INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/../FindOpenCL.cmake)
platforms/opencl/staticTarget/CMakeLists.txt:INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIR})
platforms/opencl/staticTarget/CMakeLists.txt:FILE(GLOB OPENCL_KERNELS ${KERNEL_SOURCE_DIR}/kernels/*.cl)
platforms/opencl/staticTarget/CMakeLists.txt:    DEPENDS ${OPENCL_KERNELS}
platforms/opencl/staticTarget/CMakeLists.txt:TARGET_LINK_LIBRARIES(${STATIC_TARGET} ${OPENMM_LIBRARY_NAME}  ${OPENCL_LIBRARIES} ${PTHREADS_LIB_STATIC})
platforms/opencl/sharedTarget/CMakeLists.txt:# Include OpenCL related files.
platforms/opencl/sharedTarget/CMakeLists.txt:# INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/../FindOpenCL.cmake)
platforms/opencl/sharedTarget/CMakeLists.txt:INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIR})
platforms/opencl/sharedTarget/CMakeLists.txt:FILE(GLOB OPENCL_KERNELS ${KERNEL_SOURCE_DIR}/kernels/*.cl)
platforms/opencl/sharedTarget/CMakeLists.txt:    DEPENDS ${OPENCL_KERNELS}
platforms/opencl/sharedTarget/CMakeLists.txt:TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_LIBRARY_NAME}  ${OPENCL_LIBRARIES} ${PTHREADS_LIB})
platforms/opencl/tests/TestOpenCLGayBerneForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLCustomCentroidBondForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLHarmonicBondForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLPeriodicTorsionForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLDispersionPME.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLVariableLangevinIntegrator.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLNoseHooverIntegrator.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLSort.cpp: * This tests the OpenCL implementation of sorting.
platforms/opencl/tests/TestOpenCLSort.cpp:#include "OpenCLArray.h"
platforms/opencl/tests/TestOpenCLSort.cpp:#include "OpenCLContext.h"
platforms/opencl/tests/TestOpenCLSort.cpp:#include "OpenCLSort.h"
platforms/opencl/tests/TestOpenCLSort.cpp:static OpenCLPlatform platform;
platforms/opencl/tests/TestOpenCLSort.cpp:class SortTrait : public OpenCLSort::SortTrait {
platforms/opencl/tests/TestOpenCLSort.cpp:    OpenCLPlatform::PlatformData platformData(system, "", "", platform.getPropertyDefaultValue("OpenCLPrecision"), "false", "false", 1, NULL);
platforms/opencl/tests/TestOpenCLSort.cpp:    OpenCLContext& context = *platformData.contexts[0];
platforms/opencl/tests/TestOpenCLSort.cpp:    OpenCLArray data(context, array.size(), sizeof(float), "sortData");
platforms/opencl/tests/TestOpenCLSort.cpp:    OpenCLSort sort(context, new SortTrait(), array.size(), uniform);
platforms/opencl/tests/TestOpenCLSort.cpp:            platform.setPropertyDefaultValue("OpenCLPrecision", string(argv[1]));
platforms/opencl/tests/TestOpenCLCMAPTorsionForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLFFT.cpp: * This tests the OpenCL implementation of FFT.
platforms/opencl/tests/TestOpenCLFFT.cpp:#include "OpenCLArray.h"
platforms/opencl/tests/TestOpenCLFFT.cpp:#include "OpenCLContext.h"
platforms/opencl/tests/TestOpenCLFFT.cpp:#include "OpenCLFFT3D.h"
platforms/opencl/tests/TestOpenCLFFT.cpp:#include "OpenCLSort.h"
platforms/opencl/tests/TestOpenCLFFT.cpp:static OpenCLPlatform platform;
platforms/opencl/tests/TestOpenCLFFT.cpp:    OpenCLPlatform::PlatformData platformData(system, "", "", platform.getPropertyDefaultValue("OpenCLPrecision"), "false", "false", 1, NULL);
platforms/opencl/tests/TestOpenCLFFT.cpp:    OpenCLContext& context = *platformData.contexts[0];
platforms/opencl/tests/TestOpenCLFFT.cpp:    OpenCLArray grid1(context, original.size(), sizeof(Real2), "grid1");
platforms/opencl/tests/TestOpenCLFFT.cpp:    OpenCLArray grid2(context, original.size(), sizeof(Real2), "grid2");
platforms/opencl/tests/TestOpenCLFFT.cpp:    OpenCLFFT3D fft(context, xsize, ysize, zsize, realToComplex);
platforms/opencl/tests/TestOpenCLFFT.cpp:            platform.setPropertyDefaultValue("OpenCLPrecision", string(argv[1]));
platforms/opencl/tests/TestOpenCLFFT.cpp:        if (platform.getPropertyDefaultValue("OpenCLPrecision") == "double") {
platforms/opencl/tests/TestOpenCLCustomCompoundBondForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLCustomCVForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLCheckpoints.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLCustomHbondForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLMonteCarloFlexibleBarostat.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLCustomTorsionForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLVirtualSites.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLLangevinMiddleIntegrator.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/OpenCLTests.h:#include "OpenCLPlatform.h"
platforms/opencl/tests/OpenCLTests.h:OpenMM::OpenCLPlatform platform;
platforms/opencl/tests/OpenCLTests.h:        platform.setPropertyDefaultValue("OpenCLPlatformIndex", std::string(argv[2]));
platforms/opencl/tests/TestOpenCLCustomExternalForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLCustomAngleForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLCustomBondForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLMonteCarloAnisotropicBarostat.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLLangevinIntegrator.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLATMForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/has_opencl_gpu.c: * file has_opencl_gpu.c 
platforms/opencl/tests/has_opencl_gpu.c: * Returns zero if an OpenCL-capable GPU is found.
platforms/opencl/tests/has_opencl_gpu.c:#include <OpenCL/cl.h>
platforms/opencl/tests/has_opencl_gpu.c: * check_devices() looks for a GPU among all OpenCL devices 
platforms/opencl/tests/has_opencl_gpu.c: * in a particular OpenCL platform.
platforms/opencl/tests/has_opencl_gpu.c: * Returns zero if a GPU is found.  Returns one otherwise.
platforms/opencl/tests/has_opencl_gpu.c:    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 10, devices, &num_devices);
platforms/opencl/tests/has_opencl_gpu.c:                printf(" CL_DEVICE_NOT_FOUND; no OpenCL devices that matched device_type were found.\n");
platforms/opencl/tests/has_opencl_gpu.c:        printf("No OpenCL platforms found.\n");
platforms/opencl/tests/has_opencl_gpu.c:            return status; // found GPU
platforms/opencl/tests/has_opencl_gpu.c:    return 1; // did NOT find GPU
platforms/opencl/tests/TestOpenCLCustomCPPForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/CMakeLists.txt:# INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/../FindOpenCL.cmake)
platforms/opencl/tests/CMakeLists.txt:INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIR})
platforms/opencl/tests/CMakeLists.txt:SET(OPENMM_BUILD_OPENCL_DOUBLE_PRECISION_TESTS TRUE CACHE BOOL "Whether to build double precision versions of OpenCL test cases")
platforms/opencl/tests/CMakeLists.txt:set(OPENCL_TEST_PLATFORM_INDEX -1 CACHE STRING "OpenCL platform index used for running OpenCL test cases. The default, -1, selects the fastest platform")
platforms/opencl/tests/CMakeLists.txt:set(OPENCL_TEST_DEVICE_INDEX -1 CACHE STRING "OpenCL device index used for running OpenCL test cases. The default, -1, selects the fastest device")
platforms/opencl/tests/CMakeLists.txt:MARK_AS_ADVANCED(OPENCL_TEST_PLATFORM_INDEX)
platforms/opencl/tests/CMakeLists.txt:MARK_AS_ADVANCED(OPENCL_TEST_DEVICE_INDEX)
platforms/opencl/tests/CMakeLists.txt:    ADD_TEST(${TEST_ROOT}Single ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT} single ${OPENCL_TEST_PLATFORM_INDEX} ${OPENCL_TEST_DEVICE_INDEX})
platforms/opencl/tests/CMakeLists.txt:    IF (OPENMM_BUILD_OPENCL_DOUBLE_PRECISION_TESTS)
platforms/opencl/tests/CMakeLists.txt:        ADD_TEST(${TEST_ROOT}Mixed ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT} mixed ${OPENCL_TEST_PLATFORM_INDEX} ${OPENCL_TEST_DEVICE_INDEX})
platforms/opencl/tests/CMakeLists.txt:        ADD_TEST(${TEST_ROOT}Double ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT} double ${OPENCL_TEST_PLATFORM_INDEX} ${OPENCL_TEST_DEVICE_INDEX})
platforms/opencl/tests/CMakeLists.txt:    ENDIF(OPENMM_BUILD_OPENCL_DOUBLE_PRECISION_TESTS)
platforms/opencl/tests/TestOpenCLMultipleForces.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLVariableVerletIntegrator.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLMonteCarloBarostat.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLEwald.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLBrownianIntegrator.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLRandom.cpp: * This tests the OpenCL implementation of random number generation.
platforms/opencl/tests/TestOpenCLRandom.cpp:#include "OpenCLArray.h"
platforms/opencl/tests/TestOpenCLRandom.cpp:#include "OpenCLContext.h"
platforms/opencl/tests/TestOpenCLRandom.cpp:#include "OpenCLIntegrationUtilities.h"
platforms/opencl/tests/TestOpenCLRandom.cpp:#include "OpenCLPlatform.h"
platforms/opencl/tests/TestOpenCLRandom.cpp:static OpenCLPlatform platform;
platforms/opencl/tests/TestOpenCLRandom.cpp:    OpenCLPlatform::PlatformData platformData(system, "", "", platform.getPropertyDefaultValue("OpenCLPrecision"), "false", "false", 1, NULL);
platforms/opencl/tests/TestOpenCLRandom.cpp:    OpenCLContext& context = *platformData.contexts[0];
platforms/opencl/tests/TestOpenCLRandom.cpp:    OpenCLArray& random = context.getIntegrationUtilities().getRandom();
platforms/opencl/tests/TestOpenCLRandom.cpp:            platform.setPropertyDefaultValue("OpenCLPrecision", string(argv[1]));
platforms/opencl/tests/TestOpenCLCompoundIntegrator.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLAndersenThermostat.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLCMMotionRemover.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLHarmonicAngleForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLCustomNonbondedForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLSettle.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLCustomIntegrator.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLRMSDForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLCustomManyParticleForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLNonbondedForce.cpp:#define CL_HPP_TARGET_OPENCL_VERSION 120
platforms/opencl/tests/TestOpenCLNonbondedForce.cpp:#define CL_HPP_MINIMUM_OPENCL_VERSION 120
platforms/opencl/tests/TestOpenCLNonbondedForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLNonbondedForce.cpp:#include "opencl.hpp"
platforms/opencl/tests/TestOpenCLNonbondedForce.cpp:    int platformIndex = stoi(platform.getPropertyValue(context, OpenCLPlatform::OpenCLPlatformIndex()));
platforms/opencl/tests/TestOpenCLNonbondedForce.cpp:    int deviceIndex = stoi(platform.getPropertyValue(context, OpenCLPlatform::OpenCLDeviceIndex()));
platforms/opencl/tests/TestOpenCLCustomGBForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLLocalEnergyMinimizer.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLDeviceQuery.cpp:#include "OpenCLContext.h"
platforms/opencl/tests/TestOpenCLDeviceQuery.cpp:    cout << "OpenCL devices:" << endl << endl;
platforms/opencl/tests/TestOpenCLDeviceQuery.cpp:            cout << "OpenCLPlatformIndex " << j << ", OpenCLDeviceIndex " << i << ": \"" << d.getInfo<CL_DEVICE_NAME>()
platforms/opencl/tests/TestOpenCLDeviceQuery.cpp:            } else if (d.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) {
platforms/opencl/tests/TestOpenCLDeviceQuery.cpp:                cout << "CL_DEVICE_TYPE_GPU" << endl;
platforms/opencl/tests/TestOpenCLDeviceQuery.cpp:            if (d.getInfo<CL_DEVICE_TYPE>() != CL_DEVICE_TYPE_GPU) {
platforms/opencl/tests/TestOpenCLGBSAOBCForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLRBTorsionForce.cpp:#include "OpenCLTests.h"
platforms/opencl/tests/TestOpenCLVerletIntegrator.cpp:#include "OpenCLTests.h"
platforms/opencl/include/OpenCLSort.h:#ifndef __OPENMM_OPENCLSORT_H__
platforms/opencl/include/OpenCLSort.h:#define __OPENMM_OPENCLSORT_H__
platforms/opencl/include/OpenCLSort.h:#include "OpenCLArray.h"
platforms/opencl/include/OpenCLSort.h:#include "OpenCLContext.h"
platforms/opencl/include/OpenCLSort.h: * class FloatTrait : public OpenCLSort::SortTrait {
platforms/opencl/include/OpenCLSort.h: * Sorting Algorithm with CUDA"  Journal of the Chinese Institute of Engineers, 32(7),
platforms/opencl/include/OpenCLSort.h:class OPENMM_EXPORT_COMMON OpenCLSort {
platforms/opencl/include/OpenCLSort.h:     * Create an OpenCLSort object for sorting data of a particular type.
platforms/opencl/include/OpenCLSort.h:     *                   and deletes it when the OpenCLSort is deleted.
platforms/opencl/include/OpenCLSort.h:    OpenCLSort(OpenCLContext& context, SortTrait* trait, unsigned int length, bool uniform=true);
platforms/opencl/include/OpenCLSort.h:    ~OpenCLSort();
platforms/opencl/include/OpenCLSort.h:    void sort(OpenCLArray& data);
platforms/opencl/include/OpenCLSort.h:    OpenCLContext& context;
platforms/opencl/include/OpenCLSort.h:    OpenCLArray dataRange;
platforms/opencl/include/OpenCLSort.h:    OpenCLArray bucketOfElement;
platforms/opencl/include/OpenCLSort.h:    OpenCLArray offsetInBucket;
platforms/opencl/include/OpenCLSort.h:    OpenCLArray bucketOffset;
platforms/opencl/include/OpenCLSort.h:    OpenCLArray buckets;
platforms/opencl/include/OpenCLSort.h:class OpenCLSort::SortTrait {
platforms/opencl/include/OpenCLSort.h:     * Get the CUDA code to select the key from the data value.
platforms/opencl/include/OpenCLSort.h:#endif // __OPENMM_OPENCLSORT_H__
platforms/opencl/include/OpenCLNonbondedUtilities.h:#ifndef OPENMM_OPENCLNONBONDEDUTILITIES_H_
platforms/opencl/include/OpenCLNonbondedUtilities.h:#define OPENMM_OPENCLNONBONDEDUTILITIES_H_
platforms/opencl/include/OpenCLNonbondedUtilities.h:#include "OpenCLArray.h"
platforms/opencl/include/OpenCLNonbondedUtilities.h:#include "OpenCLExpressionUtilities.h"
platforms/opencl/include/OpenCLNonbondedUtilities.h:class OpenCLContext;
platforms/opencl/include/OpenCLNonbondedUtilities.h:class OpenCLSort;
platforms/opencl/include/OpenCLNonbondedUtilities.h:class OPENMM_EXPORT_COMMON OpenCLNonbondedUtilities : public NonbondedUtilities {
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLNonbondedUtilities(OpenCLContext& context);
platforms/opencl/include/OpenCLNonbondedUtilities.h:    ~OpenCLNonbondedUtilities();
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray& getBlockCenters() {
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray& getBlockBoundingBoxes() {
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray& getInteractionCount() {
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray& getInteractingTiles() {
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray& getInteractingAtoms() {
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray& getExclusions() {
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray& getExclusionTiles() {
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray& getExclusionIndices() {
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray& getExclusionRowIndices() {
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray& getRebuildNeighborList() {
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLContext& context;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray exclusionTiles;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray exclusions;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray exclusionIndices;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray exclusionRowIndices;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray interactingTiles;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray interactingAtoms;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray interactionCount;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray blockCenter;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray blockBoundingBox;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray sortedBlocks;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray sortedBlockCenter;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray sortedBlockBoundingBox;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray blockSizeRange;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray largeBlockCenter;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray largeBlockBoundingBox;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray oldPositions;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLArray rebuildNeighborList;
platforms/opencl/include/OpenCLNonbondedUtilities.h:    OpenCLSort* blockSorter;
platforms/opencl/include/OpenCLNonbondedUtilities.h:class OpenCLNonbondedUtilities::KernelSet {
platforms/opencl/include/OpenCLNonbondedUtilities.h:class OpenCLNonbondedUtilities::ParameterInfo {
platforms/opencl/include/OpenCLNonbondedUtilities.h:#endif /*OPENMM_OPENCLNONBONDEDUTILITIES_H_*/
platforms/opencl/include/OpenCLIntegrationUtilities.h:#ifndef OPENMM_OPENCLINTEGRATIONUTILITIES_H_
platforms/opencl/include/OpenCLIntegrationUtilities.h:#define OPENMM_OPENCLINTEGRATIONUTILITIES_H_
platforms/opencl/include/OpenCLIntegrationUtilities.h:#include "OpenCLArray.h"
platforms/opencl/include/OpenCLIntegrationUtilities.h:class OpenCLContext;
platforms/opencl/include/OpenCLIntegrationUtilities.h:class OPENMM_EXPORT_COMMON OpenCLIntegrationUtilities : public IntegrationUtilities {
platforms/opencl/include/OpenCLIntegrationUtilities.h:    OpenCLIntegrationUtilities(OpenCLContext& context, const System& system);
platforms/opencl/include/OpenCLIntegrationUtilities.h:    OpenCLArray& getPosDelta();
platforms/opencl/include/OpenCLIntegrationUtilities.h:    OpenCLArray& getRandom();
platforms/opencl/include/OpenCLIntegrationUtilities.h:    OpenCLArray& getStepSize();
platforms/opencl/include/OpenCLIntegrationUtilities.h:    OpenCLArray ccmaConvergedHostBuffer;
platforms/opencl/include/OpenCLIntegrationUtilities.h:#endif /*OPENMM_OPENCLINTEGRATIONUTILITIES_H_*/
platforms/opencl/include/OpenCLEvent.h:#ifndef OPENMM_OPENCLEVENT_H_
platforms/opencl/include/OpenCLEvent.h:#define OPENMM_OPENCLEVENT_H_
platforms/opencl/include/OpenCLEvent.h:#include "OpenCLContext.h"
platforms/opencl/include/OpenCLEvent.h: * This is the OpenCL implementation of the ComputeKernelImpl interface. 
platforms/opencl/include/OpenCLEvent.h:class OpenCLEvent : public ComputeEventImpl {
platforms/opencl/include/OpenCLEvent.h:    OpenCLEvent(OpenCLContext& context);
platforms/opencl/include/OpenCLEvent.h:    OpenCLContext& context;
platforms/opencl/include/OpenCLEvent.h:#endif /*OPENMM_OPENCLEVENT_H_*/
platforms/opencl/include/OpenCLBondedUtilities.h:#ifndef OPENMM_OPENCLBONDEDUTILITIES_H_
platforms/opencl/include/OpenCLBondedUtilities.h:#define OPENMM_OPENCLBONDEDUTILITIES_H_
platforms/opencl/include/OpenCLBondedUtilities.h:class OPENMM_EXPORT_COMMON OpenCLBondedUtilities : public BondedUtilities {
platforms/opencl/include/OpenCLBondedUtilities.h:    OpenCLBondedUtilities(ComputeContext& context) : BondedUtilities(context) {
platforms/opencl/include/OpenCLBondedUtilities.h:#endif /*OPENMM_OPENCLBONDEDUTILITIES_H_*/
platforms/opencl/include/OpenCLForceInfo.h:#ifndef OPENMM_OPENCLFORCEINFO_H_
platforms/opencl/include/OpenCLForceInfo.h:#define OPENMM_OPENCLFORCEINFO_H_
platforms/opencl/include/OpenCLForceInfo.h: * Using this mechanism is equivalent to calling requestForceBuffers() on the OpenCLContext.
platforms/opencl/include/OpenCLForceInfo.h:class OPENMM_EXPORT_COMMON OpenCLForceInfo : public ComputeForceInfo {
platforms/opencl/include/OpenCLForceInfo.h:    OpenCLForceInfo(int requiredForceBuffers) : requiredForceBuffers(requiredForceBuffers) {
platforms/opencl/include/OpenCLForceInfo.h:#endif /*OPENMM_OPENCLFORCEINFO_H_*/
platforms/opencl/include/OpenCLCompact.h:#ifndef __OPENMM_OPENCLCOMPACT_H__
platforms/opencl/include/OpenCLCompact.h:#define __OPENMM_OPENCLCOMPACT_H__
platforms/opencl/include/OpenCLCompact.h:/* Code for OPENCL stream compaction. Roughly based on:
platforms/opencl/include/OpenCLCompact.h:          it'd be easy to take the CUDA SDK scanLargeArray sample, and do a prefix sum over dgBlockCounts in
platforms/opencl/include/OpenCLCompact.h:  Author:       CUDA version by Imran Haque (ihaque@cs.stanford.edu), converted to OpenCL by Peter Eastman
platforms/opencl/include/OpenCLCompact.h:#include "OpenCLArray.h"
platforms/opencl/include/OpenCLCompact.h:#include "OpenCLContext.h"
platforms/opencl/include/OpenCLCompact.h:class OPENMM_EXPORT_COMMON OpenCLCompact {
platforms/opencl/include/OpenCLCompact.h:    OpenCLCompact(OpenCLContext& context);
platforms/opencl/include/OpenCLCompact.h:    void compactStream(OpenCLArray& dOut, OpenCLArray& dIn, OpenCLArray& dValid, OpenCLArray& numValid);
platforms/opencl/include/OpenCLCompact.h:    OpenCLContext& context;
platforms/opencl/include/OpenCLCompact.h:    OpenCLArray dgBlockCounts;
platforms/opencl/include/OpenCLCompact.h:#endif // __OPENMM_OPENCLCOMPACT_H__
platforms/opencl/include/OpenCLExpressionUtilities.h:#ifndef OPENMM_OPENCLEXPRESSIONUTILITIES_H_
platforms/opencl/include/OpenCLExpressionUtilities.h:#define OPENMM_OPENCLEXPRESSIONUTILITIES_H_
platforms/opencl/include/OpenCLExpressionUtilities.h:class OPENMM_EXPORT_COMMON OpenCLExpressionUtilities : public ExpressionUtilities {
platforms/opencl/include/OpenCLExpressionUtilities.h:    OpenCLExpressionUtilities(ComputeContext& context) : ExpressionUtilities(context) {
platforms/opencl/include/OpenCLExpressionUtilities.h:#endif /*OPENMM_OPENCLEXPRESSIONUTILITIES_H_*/
platforms/opencl/include/OpenCLParameterSet.h:#ifndef OPENMM_OPENCLPARAMETERSET_H_
platforms/opencl/include/OpenCLParameterSet.h:#define OPENMM_OPENCLPARAMETERSET_H_
platforms/opencl/include/OpenCLParameterSet.h:#include "OpenCLContext.h"
platforms/opencl/include/OpenCLParameterSet.h:#include "OpenCLNonbondedUtilities.h"
platforms/opencl/include/OpenCLParameterSet.h:class OpenCLNonbondedUtilities;
platforms/opencl/include/OpenCLParameterSet.h:class OPENMM_EXPORT_COMMON OpenCLParameterSet : public ComputeParameterSet {
platforms/opencl/include/OpenCLParameterSet.h:     * Create an OpenCLParameterSet.
platforms/opencl/include/OpenCLParameterSet.h:    OpenCLParameterSet(OpenCLContext& context, int numParameters, int numObjects, const std::string& name, bool bufferPerParameter=false, bool useDoublePrecision=false);
platforms/opencl/include/OpenCLParameterSet.h:     * Get a set of OpenCLNonbondedUtilities::ParameterInfo objects which describe the Buffers
platforms/opencl/include/OpenCLParameterSet.h:    std::vector<OpenCLNonbondedUtilities::ParameterInfo>& getBuffers() {
platforms/opencl/include/OpenCLParameterSet.h:     * Get a set of OpenCLNonbondedUtilities::ParameterInfo objects which describe the Buffers
platforms/opencl/include/OpenCLParameterSet.h:    const std::vector<OpenCLNonbondedUtilities::ParameterInfo>& getBuffers() const {
platforms/opencl/include/OpenCLParameterSet.h:    std::vector<OpenCLNonbondedUtilities::ParameterInfo> buffers;
platforms/opencl/include/OpenCLParameterSet.h:#endif /*OPENMM_OPENCLPARAMETERSET_H_*/
platforms/opencl/include/OpenCLFFT3D.h:#ifndef __OPENMM_OPENCLFFT3D_H__
platforms/opencl/include/OpenCLFFT3D.h:#define __OPENMM_OPENCLFFT3D_H__
platforms/opencl/include/OpenCLFFT3D.h:#include "OpenCLArray.h"
platforms/opencl/include/OpenCLFFT3D.h:class OPENMM_EXPORT_COMMON OpenCLFFT3D {
platforms/opencl/include/OpenCLFFT3D.h:     * Create an OpenCLFFT3D object for performing transforms of a particular size.
platforms/opencl/include/OpenCLFFT3D.h:    OpenCLFFT3D(OpenCLContext& context, int xsize, int ysize, int zsize, bool realToComplex=false);
platforms/opencl/include/OpenCLFFT3D.h:    ~OpenCLFFT3D();
platforms/opencl/include/OpenCLFFT3D.h:    void execFFT(OpenCLArray& in, OpenCLArray& out, bool forward = true);
platforms/opencl/include/OpenCLFFT3D.h:    OpenCLContext& context;
platforms/opencl/include/OpenCLFFT3D.h:#endif // __OPENMM_OPENCLFFT3D_H__
platforms/opencl/include/OpenCLKernels.h:#ifndef OPENMM_OPENCLKERNELS_H_
platforms/opencl/include/OpenCLKernels.h:#define OPENMM_OPENCLKERNELS_H_
platforms/opencl/include/OpenCLKernels.h:#include "OpenCLPlatform.h"
platforms/opencl/include/OpenCLKernels.h:#include "OpenCLArray.h"
platforms/opencl/include/OpenCLKernels.h:#include "OpenCLContext.h"
platforms/opencl/include/OpenCLKernels.h:#include "OpenCLFFT3D.h"
platforms/opencl/include/OpenCLKernels.h:#include "OpenCLSort.h"
platforms/opencl/include/OpenCLKernels.h:class OpenCLCalcForcesAndEnergyKernel : public CalcForcesAndEnergyKernel {
platforms/opencl/include/OpenCLKernels.h:    OpenCLCalcForcesAndEnergyKernel(std::string name, const Platform& platform, OpenCLContext& cl) : CalcForcesAndEnergyKernel(name, platform), cl(cl) {
platforms/opencl/include/OpenCLKernels.h:   OpenCLContext& cl;
platforms/opencl/include/OpenCLKernels.h:class OpenCLCalcNonbondedForceKernel : public CalcNonbondedForceKernel {
platforms/opencl/include/OpenCLKernels.h:    OpenCLCalcNonbondedForceKernel(std::string name, const Platform& platform, OpenCLContext& cl, const System& system) : CalcNonbondedForceKernel(name, platform),
platforms/opencl/include/OpenCLKernels.h:    ~OpenCLCalcNonbondedForceKernel();
platforms/opencl/include/OpenCLKernels.h:    class SortTrait : public OpenCLSort::SortTrait {
platforms/opencl/include/OpenCLKernels.h:    OpenCLContext& cl;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray charges;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray sigmaEpsilon;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray exceptionParams;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray exclusionAtoms;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray exclusionParams;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray baseParticleParams;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray baseExceptionParams;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray particleParamOffsets;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray exceptionParamOffsets;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray particleOffsetIndices;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray exceptionOffsetIndices;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray globalParams;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray cosSinSums;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray pmeGrid1;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray pmeGrid2;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray pmeBsplineModuliX;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray pmeBsplineModuliY;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray pmeBsplineModuliZ;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray pmeDispersionBsplineModuliX;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray pmeDispersionBsplineModuliY;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray pmeDispersionBsplineModuliZ;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray pmeBsplineTheta;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray pmeAtomRange;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray pmeAtomGridIndex;
platforms/opencl/include/OpenCLKernels.h:    OpenCLArray pmeEnergyBuffer;
platforms/opencl/include/OpenCLKernels.h:    OpenCLSort* sort;
platforms/opencl/include/OpenCLKernels.h:    OpenCLFFT3D* fft;
platforms/opencl/include/OpenCLKernels.h:    OpenCLFFT3D* dispersionFft;
platforms/opencl/include/OpenCLKernels.h:class OpenCLCalcCustomCVForceKernel : public CommonCalcCustomCVForceKernel {
platforms/opencl/include/OpenCLKernels.h:    OpenCLCalcCustomCVForceKernel(std::string name, const Platform& platform, ComputeContext& cc) : CommonCalcCustomCVForceKernel(name, platform, cc) {
platforms/opencl/include/OpenCLKernels.h:        return *reinterpret_cast<OpenCLPlatform::PlatformData*>(innerContext.getPlatformData())->contexts[0];
platforms/opencl/include/OpenCLKernels.h:class OpenCLCalcATMForceKernel : public CommonCalcATMForceKernel {
platforms/opencl/include/OpenCLKernels.h:    OpenCLCalcATMForceKernel(std::string name, const Platform& platform, ComputeContext& cc) : CommonCalcATMForceKernel(name, platform, cc) {
platforms/opencl/include/OpenCLKernels.h:        return *reinterpret_cast<OpenCLPlatform::PlatformData*>(innerContext.getPlatformData())->contexts[0];
platforms/opencl/include/OpenCLKernels.h:#endif /*OPENMM_OPENCLKERNELS_H_*/
platforms/opencl/include/OpenCLKernel.h:#ifndef OPENMM_OPENCLKERNEL_H_
platforms/opencl/include/OpenCLKernel.h:#define OPENMM_OPENCLKERNEL_H_
platforms/opencl/include/OpenCLKernel.h:#include "OpenCLArray.h"
platforms/opencl/include/OpenCLKernel.h:#include "OpenCLContext.h"
platforms/opencl/include/OpenCLKernel.h: * This is the OpenCL implementation of the ComputeKernelImpl interface. 
platforms/opencl/include/OpenCLKernel.h:class OpenCLKernel : public ComputeKernelImpl {
platforms/opencl/include/OpenCLKernel.h:     * Create a new OpenCLKernel.
platforms/opencl/include/OpenCLKernel.h:    OpenCLKernel(OpenCLContext& context, cl::Kernel kernel);
platforms/opencl/include/OpenCLKernel.h:    OpenCLContext& context;
platforms/opencl/include/OpenCLKernel.h:    std::vector<OpenCLArray*> arrayArgs;
platforms/opencl/include/OpenCLKernel.h:#endif /*OPENMM_OPENCLKERNEL_H_*/
platforms/opencl/include/OpenCLContext.h:#ifndef OPENMM_OPENCLCONTEXT_H_
platforms/opencl/include/OpenCLContext.h:#define OPENMM_OPENCLCONTEXT_H_
platforms/opencl/include/OpenCLContext.h:#define CL_HPP_TARGET_OPENCL_VERSION 120
platforms/opencl/include/OpenCLContext.h:#define CL_HPP_MINIMUM_OPENCL_VERSION 120
platforms/opencl/include/OpenCLContext.h:#include "opencl.hpp"
platforms/opencl/include/OpenCLContext.h:#include "OpenCLArray.h"
platforms/opencl/include/OpenCLContext.h:#include "OpenCLBondedUtilities.h"
platforms/opencl/include/OpenCLContext.h:#include "OpenCLExpressionUtilities.h"
platforms/opencl/include/OpenCLContext.h:#include "OpenCLIntegrationUtilities.h"
platforms/opencl/include/OpenCLContext.h:#include "OpenCLNonbondedUtilities.h"
platforms/opencl/include/OpenCLContext.h:#include "OpenCLPlatform.h"
platforms/opencl/include/OpenCLContext.h:class OpenCLForceInfo;
platforms/opencl/include/OpenCLContext.h: * This class contains the information associated with a Context by the OpenCL Platform.  Each OpenCLContext is
platforms/opencl/include/OpenCLContext.h: * in parallel on multiple devices, there is a separate OpenCLContext for each one.  The list of all contexts is
platforms/opencl/include/OpenCLContext.h: * stored in the OpenCLPlatform::PlatformData.
platforms/opencl/include/OpenCLContext.h: * In addition, a worker thread is created for each OpenCLContext.  This is used for parallel computations, so that
platforms/opencl/include/OpenCLContext.h:class OPENMM_EXPORT_COMMON OpenCLContext : public ComputeContext {
platforms/opencl/include/OpenCLContext.h:    OpenCLContext(const System& system, int platformIndex, int deviceIndex, const std::string& precision, OpenCLPlatform::PlatformData& platformData,
platforms/opencl/include/OpenCLContext.h:        OpenCLContext* originalContext);
platforms/opencl/include/OpenCLContext.h:    ~OpenCLContext();
platforms/opencl/include/OpenCLContext.h:    OpenCLPlatform::PlatformData& getPlatformData() {
platforms/opencl/include/OpenCLContext.h:     * one OpenCLContext is created for each device.
platforms/opencl/include/OpenCLContext.h:    OpenCLArray* createArray();
platforms/opencl/include/OpenCLContext.h:     * Convert an array to an OpenCLArray.  If the argument is already an OpenCLArray, this simply casts it.
platforms/opencl/include/OpenCLContext.h:     * If the argument is a ComputeArray that wraps an OpenCLArray, this returns the wrapped array.  For any
platforms/opencl/include/OpenCLContext.h:    OpenCLArray& unwrap(ArrayInterface& array) const;
platforms/opencl/include/OpenCLContext.h:    OpenCLArray& getPosq() {
platforms/opencl/include/OpenCLContext.h:    OpenCLArray& getPosqCorrection() {
platforms/opencl/include/OpenCLContext.h:    OpenCLArray& getVelm() {
platforms/opencl/include/OpenCLContext.h:    OpenCLArray& getForce() {
platforms/opencl/include/OpenCLContext.h:    OpenCLArray& getForceBuffers() {
platforms/opencl/include/OpenCLContext.h:    OpenCLArray& getLongForceBuffer() {
platforms/opencl/include/OpenCLContext.h:    OpenCLArray& getEnergyBuffer() {
platforms/opencl/include/OpenCLContext.h:    OpenCLArray& getEnergyParamDerivBuffer() {
platforms/opencl/include/OpenCLContext.h:    OpenCLArray& getAtomIndexArray() {
platforms/opencl/include/OpenCLContext.h:     * Create an OpenCL Program from source code.
platforms/opencl/include/OpenCLContext.h:     * @param optimizationFlags  the optimization flags to pass to the OpenCL compiler.  If this is
platforms/opencl/include/OpenCLContext.h:     * Create an OpenCL Program from source code.
platforms/opencl/include/OpenCLContext.h:     * @param optimizationFlags  the optimization flags to pass to the OpenCL compiler.  If this is
platforms/opencl/include/OpenCLContext.h:    void reduceBuffer(OpenCLArray& array, OpenCLArray& longBuffer, int numBuffers);
platforms/opencl/include/OpenCLContext.h:     * may be more efficient on CPUs and GPUs.
platforms/opencl/include/OpenCLContext.h:     * Get the OpenCLIntegrationUtilities for this context.
platforms/opencl/include/OpenCLContext.h:    OpenCLIntegrationUtilities& getIntegrationUtilities() {
platforms/opencl/include/OpenCLContext.h:     * Get the OpenCLExpressionUtilities for this context.
platforms/opencl/include/OpenCLContext.h:    OpenCLExpressionUtilities& getExpressionUtilities() {
platforms/opencl/include/OpenCLContext.h:     * Get the OpenCLBondedUtilities for this context.
platforms/opencl/include/OpenCLContext.h:    OpenCLBondedUtilities& getBondedUtilities() {
platforms/opencl/include/OpenCLContext.h:     * Get the OpenCLNonbondedUtilities for this context.
platforms/opencl/include/OpenCLContext.h:    OpenCLNonbondedUtilities& getNonbondedUtilities() {
platforms/opencl/include/OpenCLContext.h:    OpenCLNonbondedUtilities* createNonbondedUtilities() {
platforms/opencl/include/OpenCLContext.h:        return new OpenCLNonbondedUtilities(*this);
platforms/opencl/include/OpenCLContext.h:    OpenCLPlatform::PlatformData& platformData;
platforms/opencl/include/OpenCLContext.h:    OpenCLArray posq;
platforms/opencl/include/OpenCLContext.h:    OpenCLArray posqCorrection;
platforms/opencl/include/OpenCLContext.h:    OpenCLArray velm;
platforms/opencl/include/OpenCLContext.h:    OpenCLArray force;
platforms/opencl/include/OpenCLContext.h:    OpenCLArray forceBuffers;
platforms/opencl/include/OpenCLContext.h:    OpenCLArray longForceBuffer;
platforms/opencl/include/OpenCLContext.h:    OpenCLArray energyBuffer;
platforms/opencl/include/OpenCLContext.h:    OpenCLArray energySum;
platforms/opencl/include/OpenCLContext.h:    OpenCLArray energyParamDerivBuffer;
platforms/opencl/include/OpenCLContext.h:    OpenCLArray atomIndexDevice;
platforms/opencl/include/OpenCLContext.h:    OpenCLArray chargeBuffer;
platforms/opencl/include/OpenCLContext.h:    OpenCLIntegrationUtilities* integration;
platforms/opencl/include/OpenCLContext.h:    OpenCLExpressionUtilities* expression;
platforms/opencl/include/OpenCLContext.h:    OpenCLBondedUtilities* bonded;
platforms/opencl/include/OpenCLContext.h:    OpenCLNonbondedUtilities* nonbonded;
platforms/opencl/include/OpenCLContext.h:class OPENMM_EXPORT_COMMON OpenCLContext::WorkTask : public ComputeContext::WorkTask {
platforms/opencl/include/OpenCLContext.h:class OPENMM_EXPORT_COMMON OpenCLContext::ReorderListener : public ComputeContext::ReorderListener {
platforms/opencl/include/OpenCLContext.h:class OPENMM_EXPORT_COMMON OpenCLContext::ForcePreComputation : public ComputeContext::ForcePreComputation {
platforms/opencl/include/OpenCLContext.h:class OPENMM_EXPORT_COMMON OpenCLContext::ForcePostComputation : public ComputeContext::ForcePostComputation {
platforms/opencl/include/OpenCLContext.h:#endif /*OPENMM_OPENCLCONTEXT_H_*/
platforms/opencl/include/OpenCLArray.h:#ifndef OPENMM_OPENCLARRAY_H_
platforms/opencl/include/OpenCLArray.h:#define OPENMM_OPENCLARRAY_H_
platforms/opencl/include/OpenCLArray.h:#define CL_HPP_TARGET_OPENCL_VERSION 120
platforms/opencl/include/OpenCLArray.h:#define CL_HPP_MINIMUM_OPENCL_VERSION 120
platforms/opencl/include/OpenCLArray.h:#include "opencl.hpp"
platforms/opencl/include/OpenCLArray.h:class OpenCLContext;
platforms/opencl/include/OpenCLArray.h: * This class encapsulates an OpenCL Buffer.  It provides a simplified API for working with it,
platforms/opencl/include/OpenCLArray.h: * and for copying data to and from the OpenCL Buffer.
platforms/opencl/include/OpenCLArray.h:class OPENMM_EXPORT_COMMON OpenCLArray : public ArrayInterface {
platforms/opencl/include/OpenCLArray.h:     * Create an OpenCLArray object.  The object is allocated on the heap with the "new" operator.
platforms/opencl/include/OpenCLArray.h:     * @param flags             the set of flags to specify when creating the OpenCL Buffer
platforms/opencl/include/OpenCLArray.h:    static OpenCLArray* create(OpenCLContext& context, size_t size, const std::string& name, cl_int flags = CL_MEM_READ_WRITE) {
platforms/opencl/include/OpenCLArray.h:        return new OpenCLArray(context, size, sizeof(T), name, flags);
platforms/opencl/include/OpenCLArray.h:     * Create an OpenCLArray object that uses a preexisting Buffer.  The object is allocated on the heap with the "new" operator.
platforms/opencl/include/OpenCLArray.h:     * @param buffer            the OpenCL Buffer this object encapsulates
platforms/opencl/include/OpenCLArray.h:    static OpenCLArray* create(OpenCLContext& context, cl::Buffer* buffer, size_t size, const std::string& name) {
platforms/opencl/include/OpenCLArray.h:        return new OpenCLArray(context, buffer, size, sizeof(T), name);
platforms/opencl/include/OpenCLArray.h:     * Create an uninitialized OpenCLArray object.  It does not point to any OpenCL Buffer,
platforms/opencl/include/OpenCLArray.h:    OpenCLArray();
platforms/opencl/include/OpenCLArray.h:     * Create an OpenCLArray object.
platforms/opencl/include/OpenCLArray.h:     * @param flags             the set of flags to specify when creating the OpenCL Buffer
platforms/opencl/include/OpenCLArray.h:    OpenCLArray(OpenCLContext& context, size_t size, int elementSize, const std::string& name, cl_int flags = CL_MEM_READ_WRITE);
platforms/opencl/include/OpenCLArray.h:     * Create an OpenCLArray object that uses a preexisting Buffer.
platforms/opencl/include/OpenCLArray.h:     * @param buffer            the OpenCL Buffer this object encapsulates
platforms/opencl/include/OpenCLArray.h:    OpenCLArray(OpenCLContext& context, cl::Buffer* buffer, size_t size, int elementSize, const std::string& name);
platforms/opencl/include/OpenCLArray.h:    ~OpenCLArray();
platforms/opencl/include/OpenCLArray.h:     * @param flags             the set of flags to specify when creating the OpenCL Buffer
platforms/opencl/include/OpenCLArray.h:    void initialize(OpenCLContext& context, size_t size, int elementSize, const std::string& name, cl_int flags);
platforms/opencl/include/OpenCLArray.h:     * @param buffer            the OpenCL Buffer this object encapsulates
platforms/opencl/include/OpenCLArray.h:    void initialize(OpenCLContext& context, cl::Buffer* buffer, size_t size, int elementSize, const std::string& name);
platforms/opencl/include/OpenCLArray.h:     * @param flags             the set of flags to specify when creating the OpenCL Buffer
platforms/opencl/include/OpenCLArray.h:    void initialize(OpenCLContext& context, size_t size, const std::string& name, cl_int flags = CL_MEM_READ_WRITE) {
platforms/opencl/include/OpenCLArray.h:     * @param buffer            the OpenCL Buffer this object encapsulates
platforms/opencl/include/OpenCLArray.h:    void initialize(OpenCLContext& context, cl::Buffer* buffer, size_t size, const std::string& name) {
platforms/opencl/include/OpenCLArray.h:     * Get the OpenCL Buffer object.
platforms/opencl/include/OpenCLArray.h:     * Copy the values in the Buffer to a second OpenCLArray.
platforms/opencl/include/OpenCLArray.h:    OpenCLContext* context;
platforms/opencl/include/OpenCLArray.h:#endif /*OPENMM_OPENCLARRAY_H_*/
platforms/opencl/include/OpenCLProgram.h:#ifndef OPENMM_OPENCLPROGRAM_H_
platforms/opencl/include/OpenCLProgram.h:#define OPENMM_OPENCLPROGRAM_H_
platforms/opencl/include/OpenCLProgram.h:#include "OpenCLContext.h"
platforms/opencl/include/OpenCLProgram.h: * This is the OpenCL implementation of the ComputeProgramImpl interface. 
platforms/opencl/include/OpenCLProgram.h:class OpenCLProgram : public ComputeProgramImpl {
platforms/opencl/include/OpenCLProgram.h:     * Create a new OpenCLProgram.
platforms/opencl/include/OpenCLProgram.h:    OpenCLProgram(OpenCLContext& context, cl::Program program);
platforms/opencl/include/OpenCLProgram.h:    OpenCLContext& context;
platforms/opencl/include/OpenCLProgram.h:#endif /*OPENMM_OPENCLPROGRAM_H_*/
platforms/opencl/include/OpenCLPlatform.h:#ifndef OPENMM_OPENCLPLATFORM_H_
platforms/opencl/include/OpenCLPlatform.h:#define OPENMM_OPENCLPLATFORM_H_
platforms/opencl/include/OpenCLPlatform.h:class OpenCLContext;
platforms/opencl/include/OpenCLPlatform.h: * This Platform subclass uses OpenCL implementations of the OpenMM kernels.
platforms/opencl/include/OpenCLPlatform.h:class OPENMM_EXPORT_COMMON OpenCLPlatform : public Platform {
platforms/opencl/include/OpenCLPlatform.h:    OpenCLPlatform();
platforms/opencl/include/OpenCLPlatform.h:        static const std::string name = "OpenCL";
platforms/opencl/include/OpenCLPlatform.h:     * This is the name of the parameter for selecting which OpenCL device or devices to use.
platforms/opencl/include/OpenCLPlatform.h:    static const std::string& OpenCLDeviceIndex() {
platforms/opencl/include/OpenCLPlatform.h:     * This is the name of the parameter that reports the OpenCL device or devices being used.
platforms/opencl/include/OpenCLPlatform.h:    static const std::string& OpenCLDeviceName() {
platforms/opencl/include/OpenCLPlatform.h:     * This is the name of the parameter for selecting which OpenCL platform to use.
platforms/opencl/include/OpenCLPlatform.h:    static const std::string& OpenCLPlatformIndex() {
platforms/opencl/include/OpenCLPlatform.h:        static const std::string key = "OpenCLPlatformIndex";
platforms/opencl/include/OpenCLPlatform.h:     * This is the name of the parameter that reports the OpenCL platform being used.
platforms/opencl/include/OpenCLPlatform.h:    static const std::string& OpenCLPlatformName() {
platforms/opencl/include/OpenCLPlatform.h:        static const std::string key = "OpenCLPlatformName";
platforms/opencl/include/OpenCLPlatform.h:    static const std::string& OpenCLPrecision() {
platforms/opencl/include/OpenCLPlatform.h:    static const std::string& OpenCLUseCpuPme() {
platforms/opencl/include/OpenCLPlatform.h:    static const std::string& OpenCLDisablePmeStream() {
platforms/opencl/include/OpenCLPlatform.h:class OPENMM_EXPORT_COMMON OpenCLPlatform::PlatformData {
platforms/opencl/include/OpenCLPlatform.h:    std::vector<OpenCLContext*> contexts;
platforms/opencl/include/OpenCLPlatform.h:#endif /*OPENMM_OPENCLPLATFORM_H_*/
platforms/opencl/include/OpenCLKernelFactory.h:#ifndef OPENMM_OPENCLKERNELFACTORY_H_
platforms/opencl/include/OpenCLKernelFactory.h:#define OPENMM_OPENCLKERNELFACTORY_H_
platforms/opencl/include/OpenCLKernelFactory.h: * This KernelFactory creates all kernels for OpenCLPlatform.
platforms/opencl/include/OpenCLKernelFactory.h:class OpenCLKernelFactory : public KernelFactory {
platforms/opencl/include/OpenCLKernelFactory.h:#endif /*OPENMM_OPENCLKERNELFACTORY_H_*/
platforms/opencl/include/OpenCLParallelKernels.h:#ifndef OPENMM_OPENCLPARALLELKERNELS_H_
platforms/opencl/include/OpenCLParallelKernels.h:#define OPENMM_OPENCLPARALLELKERNELS_H_
platforms/opencl/include/OpenCLParallelKernels.h:#include "OpenCLPlatform.h"
platforms/opencl/include/OpenCLParallelKernels.h:#include "OpenCLContext.h"
platforms/opencl/include/OpenCLParallelKernels.h:#include "OpenCLKernels.h"
platforms/opencl/include/OpenCLParallelKernels.h:class OpenCLParallelCalcForcesAndEnergyKernel : public CalcForcesAndEnergyKernel {
platforms/opencl/include/OpenCLParallelKernels.h:    OpenCLParallelCalcForcesAndEnergyKernel(std::string name, const Platform& platform, OpenCLPlatform::PlatformData& data);
platforms/opencl/include/OpenCLParallelKernels.h:    ~OpenCLParallelCalcForcesAndEnergyKernel();
platforms/opencl/include/OpenCLParallelKernels.h:    OpenCLCalcForcesAndEnergyKernel& getKernel(int index) {
platforms/opencl/include/OpenCLParallelKernels.h:        return dynamic_cast<OpenCLCalcForcesAndEnergyKernel&>(kernels[index].getImpl());
platforms/opencl/include/OpenCLParallelKernels.h:    OpenCLPlatform::PlatformData& data;
platforms/opencl/include/OpenCLParallelKernels.h:    OpenCLArray contextForces;
platforms/opencl/include/OpenCLParallelKernels.h:class OpenCLParallelCalcNonbondedForceKernel : public CalcNonbondedForceKernel {
platforms/opencl/include/OpenCLParallelKernels.h:    OpenCLParallelCalcNonbondedForceKernel(std::string name, const Platform& platform, OpenCLPlatform::PlatformData& data, const System& system);
platforms/opencl/include/OpenCLParallelKernels.h:    OpenCLCalcNonbondedForceKernel& getKernel(int index) {
platforms/opencl/include/OpenCLParallelKernels.h:        return dynamic_cast<OpenCLCalcNonbondedForceKernel&>(kernels[index].getImpl());
platforms/opencl/include/OpenCLParallelKernels.h:    OpenCLPlatform::PlatformData& data;
platforms/opencl/include/OpenCLParallelKernels.h:#endif /*OPENMM_OPENCLPARALLELKERNELS_H_*/
platforms/opencl/CMakeLists.txt:# OpenMM OpenCL Platform
platforms/opencl/CMakeLists.txt:# Creates OpenMMOpenCL library.
platforms/opencl/CMakeLists.txt:#   OpenMMOpenCL.dll
platforms/opencl/CMakeLists.txt:#   OpenMMOpenCL.lib
platforms/opencl/CMakeLists.txt:#   OpenMMOpenCL_static.lib
platforms/opencl/CMakeLists.txt:#   libOpenMMOpenCL.so
platforms/opencl/CMakeLists.txt:#   libOpenMMOpenCL_static.a
platforms/opencl/CMakeLists.txt:set(OPENMM_BUILD_OPENCL_TESTS TRUE CACHE BOOL "Whether to build OpenCL test cases")
platforms/opencl/CMakeLists.txt:if(BUILD_TESTING AND OPENMM_BUILD_OPENCL_TESTS)
platforms/opencl/CMakeLists.txt:endif(BUILD_TESTING AND OPENMM_BUILD_OPENCL_TESTS)
platforms/opencl/CMakeLists.txt:SET(OPENMMOPENCL_LIBRARY_NAME OpenMMOpenCL)
platforms/opencl/CMakeLists.txt:SET(SHARED_TARGET ${OPENMMOPENCL_LIBRARY_NAME})
platforms/opencl/CMakeLists.txt:SET(STATIC_TARGET ${OPENMMOPENCL_LIBRARY_NAME}_static)
platforms/opencl/CMakeLists.txt:SET(KERNEL_SOURCE_CLASS OpenCLKernelSources)
platforms/opencl/CMakeLists.txt:FILE(GLOB CORE_HEADERS include/*.h src/opencl.hpp ${KERNELS_H})
platforms/opencl/CMakeLists.txt:INSTALL_FILES(/include/openmm/opencl FILES ${CORE_HEADERS})
platforms/opencl/src/OpenCLCompact.cpp:/* Code for OpenCL stream compaction. Roughly based on:
platforms/opencl/src/OpenCLCompact.cpp:          it'd be easy to take the CUDA SDK scanLargeArray sample, and do a prefix sum over dgBlockCounts in
platforms/opencl/src/OpenCLCompact.cpp:  Author:       CUDA version by Imran Haque (ihaque@cs.stanford.edu), converted to OpenCL by Peter Eastman
platforms/opencl/src/OpenCLCompact.cpp:#include "OpenCLCompact.h"
platforms/opencl/src/OpenCLCompact.cpp:#include "OpenCLKernelSources.h"
platforms/opencl/src/OpenCLCompact.cpp:OpenCLCompact::OpenCLCompact(OpenCLContext& context) : context(context) {
platforms/opencl/src/OpenCLCompact.cpp:    cl::Program program = context.createProgram(OpenCLKernelSources::compact);
platforms/opencl/src/OpenCLCompact.cpp:void OpenCLCompact::compactStream(OpenCLArray& dOut, OpenCLArray& dIn, OpenCLArray& dValid, OpenCLArray& numValid) {
platforms/opencl/src/OpenCLKernelSources.h.in:#ifndef OPENMM_OPENCLKERNELSOURCES_H_
platforms/opencl/src/OpenCLKernelSources.h.in:#define OPENMM_OPENCLKERNELSOURCES_H_
platforms/opencl/src/OpenCLKernelSources.h.in: * This class is a central holding place for the source code of OpenCL kernels.
platforms/opencl/src/OpenCLKernelSources.h.in:class OPENMM_EXPORT_COMMON OpenCLKernelSources {
platforms/opencl/src/OpenCLKernelSources.h.in:#endif /*OPENMM_OPENCLKERNELSOURCES_H_*/
platforms/opencl/src/OpenCLProgram.cpp:#include "OpenCLProgram.h"
platforms/opencl/src/OpenCLProgram.cpp:#include "OpenCLKernel.h"
platforms/opencl/src/OpenCLProgram.cpp:OpenCLProgram::OpenCLProgram(OpenCLContext& context, cl::Program program) : context(context), program(program) {
platforms/opencl/src/OpenCLProgram.cpp:ComputeKernel OpenCLProgram::createKernel(const string& name) {
platforms/opencl/src/OpenCLProgram.cpp:    return shared_ptr<ComputeKernelImpl>(new OpenCLKernel(context, kernel));
platforms/opencl/src/OpenCLKernelFactory.cpp:#include "OpenCLKernelFactory.h"
platforms/opencl/src/OpenCLKernelFactory.cpp:#include "OpenCLParallelKernels.h"
platforms/opencl/src/OpenCLKernelFactory.cpp:KernelImpl* OpenCLKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
platforms/opencl/src/OpenCLKernelFactory.cpp:    OpenCLPlatform::PlatformData& data = *static_cast<OpenCLPlatform::PlatformData*>(context.getPlatformData());
platforms/opencl/src/OpenCLKernelFactory.cpp:    OpenCLContext& cl = *data.contexts[0];
platforms/opencl/src/OpenCLKernelFactory.cpp:            return new OpenCLParallelCalcForcesAndEnergyKernel(name, platform, data);
platforms/opencl/src/OpenCLKernelFactory.cpp:            return new OpenCLParallelCalcNonbondedForceKernel(name, platform, data, context.getSystem());
platforms/opencl/src/OpenCLKernelFactory.cpp:        return new OpenCLCalcForcesAndEnergyKernel(name, platform, cl);
platforms/opencl/src/OpenCLKernelFactory.cpp:        return new OpenCLCalcNonbondedForceKernel(name, platform, cl, context.getSystem());
platforms/opencl/src/OpenCLKernelFactory.cpp:        return new OpenCLCalcCustomCVForceKernel(name, platform, cl);
platforms/opencl/src/OpenCLKernelFactory.cpp:        return new OpenCLCalcATMForceKernel(name, platform, cl);
platforms/opencl/src/OpenCLPlatform.cpp:#include "OpenCLContext.h"
platforms/opencl/src/OpenCLPlatform.cpp:#include "OpenCLPlatform.h"
platforms/opencl/src/OpenCLPlatform.cpp:#include "OpenCLKernelFactory.h"
platforms/opencl/src/OpenCLPlatform.cpp:#include "OpenCLKernels.h"
platforms/opencl/src/OpenCLPlatform.cpp:extern "C" void registerOpenCLPlatform() {
platforms/opencl/src/OpenCLPlatform.cpp:    if (OpenCLPlatform::isPlatformSupported())
platforms/opencl/src/OpenCLPlatform.cpp:        Platform::registerPlatform(new OpenCLPlatform());
platforms/opencl/src/OpenCLPlatform.cpp:    if (OpenCLPlatform::isPlatformSupported())
platforms/opencl/src/OpenCLPlatform.cpp:        Platform::registerPlatform(new OpenCLPlatform());
platforms/opencl/src/OpenCLPlatform.cpp:OpenCLPlatform::OpenCLPlatform() {
platforms/opencl/src/OpenCLPlatform.cpp:    deprecatedPropertyReplacements["OpenCLDeviceIndex"] = OpenCLDeviceIndex();
platforms/opencl/src/OpenCLPlatform.cpp:    deprecatedPropertyReplacements["OpenCLDeviceName"] = OpenCLDeviceName();
platforms/opencl/src/OpenCLPlatform.cpp:    deprecatedPropertyReplacements["OpenCLPrecision"] = OpenCLPrecision();
platforms/opencl/src/OpenCLPlatform.cpp:    deprecatedPropertyReplacements["OpenCLUseCpuPme"] = OpenCLUseCpuPme();
platforms/opencl/src/OpenCLPlatform.cpp:    deprecatedPropertyReplacements["OpenCLDisablePmeStream"] = OpenCLDisablePmeStream();
platforms/opencl/src/OpenCLPlatform.cpp:    OpenCLKernelFactory* factory = new OpenCLKernelFactory();
platforms/opencl/src/OpenCLPlatform.cpp:    platformProperties.push_back(OpenCLDeviceIndex());
platforms/opencl/src/OpenCLPlatform.cpp:    platformProperties.push_back(OpenCLDeviceName());
platforms/opencl/src/OpenCLPlatform.cpp:    platformProperties.push_back(OpenCLPlatformIndex());
platforms/opencl/src/OpenCLPlatform.cpp:    platformProperties.push_back(OpenCLPlatformName());
platforms/opencl/src/OpenCLPlatform.cpp:    platformProperties.push_back(OpenCLPrecision());
platforms/opencl/src/OpenCLPlatform.cpp:    platformProperties.push_back(OpenCLUseCpuPme());
platforms/opencl/src/OpenCLPlatform.cpp:    platformProperties.push_back(OpenCLDisablePmeStream());
platforms/opencl/src/OpenCLPlatform.cpp:    setPropertyDefaultValue(OpenCLDeviceIndex(), "");
platforms/opencl/src/OpenCLPlatform.cpp:    setPropertyDefaultValue(OpenCLDeviceName(), "");
platforms/opencl/src/OpenCLPlatform.cpp:    setPropertyDefaultValue(OpenCLPlatformIndex(), "");
platforms/opencl/src/OpenCLPlatform.cpp:    setPropertyDefaultValue(OpenCLPlatformName(), "");
platforms/opencl/src/OpenCLPlatform.cpp:    setPropertyDefaultValue(OpenCLPrecision(), "single");
platforms/opencl/src/OpenCLPlatform.cpp:    setPropertyDefaultValue(OpenCLUseCpuPme(), "false");
platforms/opencl/src/OpenCLPlatform.cpp:    setPropertyDefaultValue(OpenCLDisablePmeStream(), "false");
platforms/opencl/src/OpenCLPlatform.cpp:double OpenCLPlatform::getSpeed() const {
platforms/opencl/src/OpenCLPlatform.cpp:bool OpenCLPlatform::supportsDoublePrecision() const {
platforms/opencl/src/OpenCLPlatform.cpp:bool OpenCLPlatform::isPlatformSupported() {
platforms/opencl/src/OpenCLPlatform.cpp:    // Return false for OpenCL implementations that are known
platforms/opencl/src/OpenCLPlatform.cpp:        // contained a number of serious bugs in the Apple OpenCL libraries.
platforms/opencl/src/OpenCLPlatform.cpp:    // Make sure at least one OpenCL implementation is installed.
platforms/opencl/src/OpenCLPlatform.cpp:const string& OpenCLPlatform::getPropertyValue(const Context& context, const string& property) const {
platforms/opencl/src/OpenCLPlatform.cpp:void OpenCLPlatform::setPropertyValue(Context& context, const string& property, const string& value) const {
platforms/opencl/src/OpenCLPlatform.cpp:void OpenCLPlatform::contextCreated(ContextImpl& context, const map<string, string>& properties) const {
platforms/opencl/src/OpenCLPlatform.cpp:    const string& platformPropValue = (properties.find(OpenCLPlatformIndex()) == properties.end() ?
platforms/opencl/src/OpenCLPlatform.cpp:            getPropertyDefaultValue(OpenCLPlatformIndex()) : properties.find(OpenCLPlatformIndex())->second);
platforms/opencl/src/OpenCLPlatform.cpp:    const string& devicePropValue = (properties.find(OpenCLDeviceIndex()) == properties.end() ?
platforms/opencl/src/OpenCLPlatform.cpp:            getPropertyDefaultValue(OpenCLDeviceIndex()) : properties.find(OpenCLDeviceIndex())->second);
platforms/opencl/src/OpenCLPlatform.cpp:    string precisionPropValue = (properties.find(OpenCLPrecision()) == properties.end() ?
platforms/opencl/src/OpenCLPlatform.cpp:            getPropertyDefaultValue(OpenCLPrecision()) : properties.find(OpenCLPrecision())->second);
platforms/opencl/src/OpenCLPlatform.cpp:    string cpuPmePropValue = (properties.find(OpenCLUseCpuPme()) == properties.end() ?
platforms/opencl/src/OpenCLPlatform.cpp:            getPropertyDefaultValue(OpenCLUseCpuPme()) : properties.find(OpenCLUseCpuPme())->second);
platforms/opencl/src/OpenCLPlatform.cpp:    string pmeStreamPropValue = (properties.find(OpenCLDisablePmeStream()) == properties.end() ?
platforms/opencl/src/OpenCLPlatform.cpp:            getPropertyDefaultValue(OpenCLDisablePmeStream()) : properties.find(OpenCLDisablePmeStream())->second);
platforms/opencl/src/OpenCLPlatform.cpp:void OpenCLPlatform::linkedContextCreated(ContextImpl& context, ContextImpl& originalContext) const {
platforms/opencl/src/OpenCLPlatform.cpp:    string platformPropValue = platform.getPropertyValue(originalContext.getOwner(), OpenCLPlatformIndex());
platforms/opencl/src/OpenCLPlatform.cpp:    string devicePropValue = platform.getPropertyValue(originalContext.getOwner(), OpenCLDeviceIndex());
platforms/opencl/src/OpenCLPlatform.cpp:    string precisionPropValue = platform.getPropertyValue(originalContext.getOwner(), OpenCLPrecision());
platforms/opencl/src/OpenCLPlatform.cpp:    string cpuPmePropValue = platform.getPropertyValue(originalContext.getOwner(), OpenCLUseCpuPme());
platforms/opencl/src/OpenCLPlatform.cpp:    string pmeStreamPropValue = platform.getPropertyValue(originalContext.getOwner(), OpenCLDisablePmeStream());
platforms/opencl/src/OpenCLPlatform.cpp:void OpenCLPlatform::contextDestroyed(ContextImpl& context) const {
platforms/opencl/src/OpenCLPlatform.cpp:OpenCLPlatform::PlatformData::PlatformData(const System& system, const string& platformPropValue, const string& deviceIndexProperty,
platforms/opencl/src/OpenCLPlatform.cpp:                contexts.push_back(new OpenCLContext(system, platformIndex, deviceIndex, precisionProperty, *this, (originalData == NULL ? NULL : originalData->contexts[i])));
platforms/opencl/src/OpenCLPlatform.cpp:            contexts.push_back(new OpenCLContext(system, platformIndex, -1, precisionProperty, *this, (originalData == NULL ? NULL : originalData->contexts[0])));
platforms/opencl/src/OpenCLPlatform.cpp:    propertyValues[OpenCLPlatform::OpenCLDeviceIndex()] = deviceIndex.str();
platforms/opencl/src/OpenCLPlatform.cpp:    propertyValues[OpenCLPlatform::OpenCLDeviceName()] = deviceName.str();
platforms/opencl/src/OpenCLPlatform.cpp:    propertyValues[OpenCLPlatform::OpenCLPlatformIndex()] = contexts[0]->intToString(platformIndex);
platforms/opencl/src/OpenCLPlatform.cpp:    propertyValues[OpenCLPlatform::OpenCLPlatformName()] = platforms[platformIndex].getInfo<CL_PLATFORM_NAME>();
platforms/opencl/src/OpenCLPlatform.cpp:    propertyValues[OpenCLPlatform::OpenCLPrecision()] = precisionProperty;
platforms/opencl/src/OpenCLPlatform.cpp:    propertyValues[OpenCLPlatform::OpenCLUseCpuPme()] = useCpuPme ? "true" : "false";
platforms/opencl/src/OpenCLPlatform.cpp:    propertyValues[OpenCLPlatform::OpenCLDisablePmeStream()] = disablePmeStream ? "true" : "false";
platforms/opencl/src/OpenCLPlatform.cpp:OpenCLPlatform::PlatformData::~PlatformData() {
platforms/opencl/src/OpenCLPlatform.cpp:void OpenCLPlatform::PlatformData::initializeContexts(const System& system) {
platforms/opencl/src/OpenCLPlatform.cpp:void OpenCLPlatform::PlatformData::syncContexts() {
platforms/opencl/src/OpenCLSort.cpp:#include "OpenCLSort.h"
platforms/opencl/src/OpenCLSort.cpp:#include "OpenCLKernelSources.h"
platforms/opencl/src/OpenCLSort.cpp:OpenCLSort::OpenCLSort(OpenCLContext& context, SortTrait* trait, unsigned int length, bool uniform) :
platforms/opencl/src/OpenCLSort.cpp:    cl::Program program = context.createProgram(context.replaceStrings(OpenCLKernelSources::sort, replacements));
platforms/opencl/src/OpenCLSort.cpp:    int maxShortList = max(maxLocalBuffer, (int) OpenCLContext::ThreadBlockSize*context.getNumThreadBlocks());
platforms/opencl/src/OpenCLSort.cpp:    if (vendor.size() >= 6 && vendor.substr(0, 6) == "NVIDIA") {
platforms/opencl/src/OpenCLSort.cpp:        useShortList2 = (dataLength <= OpenCLContext::ThreadBlockSize*context.getNumThreadBlocks());
platforms/opencl/src/OpenCLSort.cpp:OpenCLSort::~OpenCLSort() {
platforms/opencl/src/OpenCLSort.cpp:void OpenCLSort::sort(OpenCLArray& data) {
platforms/opencl/src/OpenCLSort.cpp:        throw OpenMMException("OpenCLSort called with different data size");
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:#include "OpenCLNonbondedUtilities.h"
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:#include "OpenCLArray.h"
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:#include "OpenCLContext.h"
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:#include "OpenCLKernelSources.h"
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:#include "OpenCLExpressionUtilities.h"
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:#include "OpenCLSort.h"
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:class OpenCLNonbondedUtilities::BlockSortTrait : public OpenCLSort::SortTrait {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:OpenCLNonbondedUtilities::OpenCLNonbondedUtilities(OpenCLContext& context) : context(context), useCutoff(false), usePeriodic(false), useNeighborList(false), anyExclusions(false), usePadding(true),
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:            // 1536 threads per GPU core.
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:        forceThreadBlockSize = (context.getSIMDWidth() >= 32 ? OpenCLContext::ThreadBlockSize : 32);
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:    setKernelSource(deviceIsCpu ? OpenCLKernelSources::nonbonded_cpu : OpenCLKernelSources::nonbonded);
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:OpenCLNonbondedUtilities::~OpenCLNonbondedUtilities() {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:void OpenCLNonbondedUtilities::addInteraction(bool usesCutoff, bool usesPeriodic, bool usesExclusions, double cutoffDistance, const vector<vector<int> >& exclusionList, const string& kernel, int forceGroup, bool useNeighborList) {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:void OpenCLNonbondedUtilities::addParameter(ComputeParameterInfo parameter) {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:void OpenCLNonbondedUtilities::addParameter(const ParameterInfo& parameter) {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:void OpenCLNonbondedUtilities::addArgument(ComputeParameterInfo parameter) {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:void OpenCLNonbondedUtilities::addArgument(const ParameterInfo& parameter) {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:string OpenCLNonbondedUtilities::addEnergyParameterDerivative(const string& param) {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:void OpenCLNonbondedUtilities::requestExclusions(const vector<vector<int> >& exclusionList) {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:void OpenCLNonbondedUtilities::initialize(const System& system) {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:        int x = atom1/OpenCLContext::TileSize;
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:            int y = atom2/OpenCLContext::TileSize;
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:    exclusions.initialize<cl_uint>(context, tilesWithExclusions.size()*OpenCLContext::TileSize, "exclusions");
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:        int x = atom1/OpenCLContext::TileSize;
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:        int offset1 = atom1-x*OpenCLContext::TileSize;
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:            int y = atom2/OpenCLContext::TileSize;
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:            int offset2 = atom2-y*OpenCLContext::TileSize;
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:                int index = exclusionTileMap[make_pair(x, y)]*OpenCLContext::TileSize;
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:                int index = exclusionTileMap[make_pair(y, x)]*OpenCLContext::TileSize;
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:        interactingAtoms.initialize<cl_int>(context, OpenCLContext::TileSize*maxTiles, "interactingAtoms");
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:        blockSorter = new OpenCLSort(context, new BlockSortTrait(), numAtomBlocks, false);
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:static void setPeriodicBoxArgs(OpenCLContext& cl, cl::Kernel& kernel, int index) {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:double OpenCLNonbondedUtilities::getMaxCutoffDistance() {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:double OpenCLNonbondedUtilities::padCutoff(double cutoff) {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:void OpenCLNonbondedUtilities::prepareInteractions(int forceGroups) {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:void OpenCLNonbondedUtilities::computeInteractions(int forceGroups, bool includeForces, bool includeEnergy) {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:bool OpenCLNonbondedUtilities::updateNeighborListSize() {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:    interactingAtoms.resize(OpenCLContext::TileSize*(size_t) maxTiles);
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:void OpenCLNonbondedUtilities::setUsePadding(bool padding) {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:void OpenCLNonbondedUtilities::setAtomBlockRange(double startFraction, double endFraction) {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:void OpenCLNonbondedUtilities::createKernelsForGroups(int groups) {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:        defines["TILE_SIZE"] = context.intToString(OpenCLContext::TileSize);
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:        string file = (deviceIsCpu ? OpenCLKernelSources::findInteractingBlocks_cpu : OpenCLKernelSources::findInteractingBlocks);
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:cl::Kernel OpenCLNonbondedUtilities::createInteractionKernel(const string& source, const vector<ParameterInfo>& params, const vector<ParameterInfo>& arguments, bool useExclusions, bool isSymmetric, int groups, bool includeForces, bool includeEnergy) {
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:    defines["TILE_SIZE"] = context.intToString(OpenCLContext::TileSize);
platforms/opencl/src/OpenCLNonbondedUtilities.cpp:void OpenCLNonbondedUtilities::setKernelSource(const string& source) {
platforms/opencl/src/OpenCLParallelKernels.cpp:#include "OpenCLParallelKernels.h"
platforms/opencl/src/OpenCLParallelKernels.cpp:class OpenCLParallelCalcForcesAndEnergyKernel::BeginComputationTask : public OpenCLContext::WorkTask {
platforms/opencl/src/OpenCLParallelKernels.cpp:    BeginComputationTask(ContextImpl& context, OpenCLContext& cl, OpenCLCalcForcesAndEnergyKernel& kernel,
platforms/opencl/src/OpenCLParallelKernels.cpp:    OpenCLContext& cl;
platforms/opencl/src/OpenCLParallelKernels.cpp:    OpenCLCalcForcesAndEnergyKernel& kernel;
platforms/opencl/src/OpenCLParallelKernels.cpp:class OpenCLParallelCalcForcesAndEnergyKernel::FinishComputationTask : public OpenCLContext::WorkTask {
platforms/opencl/src/OpenCLParallelKernels.cpp:    FinishComputationTask(ContextImpl& context, OpenCLContext& cl, OpenCLCalcForcesAndEnergyKernel& kernel,
platforms/opencl/src/OpenCLParallelKernels.cpp:    OpenCLContext& cl;
platforms/opencl/src/OpenCLParallelKernels.cpp:    OpenCLCalcForcesAndEnergyKernel& kernel;
platforms/opencl/src/OpenCLParallelKernels.cpp:OpenCLParallelCalcForcesAndEnergyKernel::OpenCLParallelCalcForcesAndEnergyKernel(string name, const Platform& platform, OpenCLPlatform::PlatformData& data) :
platforms/opencl/src/OpenCLParallelKernels.cpp:        kernels.push_back(Kernel(new OpenCLCalcForcesAndEnergyKernel(name, platform, *data.contexts[i])));
platforms/opencl/src/OpenCLParallelKernels.cpp:OpenCLParallelCalcForcesAndEnergyKernel::~OpenCLParallelCalcForcesAndEnergyKernel() {
platforms/opencl/src/OpenCLParallelKernels.cpp:void OpenCLParallelCalcForcesAndEnergyKernel::initialize(const System& system) {
platforms/opencl/src/OpenCLParallelKernels.cpp:void OpenCLParallelCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups) {
platforms/opencl/src/OpenCLParallelKernels.cpp:    OpenCLContext& cl0 = *data.contexts[0];
platforms/opencl/src/OpenCLParallelKernels.cpp:        OpenCLContext& cl = *data.contexts[i];
platforms/opencl/src/OpenCLParallelKernels.cpp:double OpenCLParallelCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups, bool& valid) {
platforms/opencl/src/OpenCLParallelKernels.cpp:        OpenCLContext& cl = *data.contexts[i];
platforms/opencl/src/OpenCLParallelKernels.cpp:        OpenCLContext& cl = *data.contexts[0];
platforms/opencl/src/OpenCLParallelKernels.cpp:class OpenCLParallelCalcNonbondedForceKernel::Task : public OpenCLContext::WorkTask {
platforms/opencl/src/OpenCLParallelKernels.cpp:    Task(ContextImpl& context, OpenCLCalcNonbondedForceKernel& kernel, bool includeForce,
platforms/opencl/src/OpenCLParallelKernels.cpp:    OpenCLCalcNonbondedForceKernel& kernel;
platforms/opencl/src/OpenCLParallelKernels.cpp:OpenCLParallelCalcNonbondedForceKernel::OpenCLParallelCalcNonbondedForceKernel(std::string name, const Platform& platform, OpenCLPlatform::PlatformData& data, const System& system) :
platforms/opencl/src/OpenCLParallelKernels.cpp:        kernels.push_back(Kernel(new OpenCLCalcNonbondedForceKernel(name, platform, *data.contexts[i], system)));
platforms/opencl/src/OpenCLParallelKernels.cpp:void OpenCLParallelCalcNonbondedForceKernel::initialize(const System& system, const NonbondedForce& force) {
platforms/opencl/src/OpenCLParallelKernels.cpp:double OpenCLParallelCalcNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
platforms/opencl/src/OpenCLParallelKernels.cpp:        OpenCLContext& cl = *data.contexts[i];
platforms/opencl/src/OpenCLParallelKernels.cpp:void OpenCLParallelCalcNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const NonbondedForce& force, int firstParticle, int lastParticle, int firstException, int lastException) {
platforms/opencl/src/OpenCLParallelKernels.cpp:void OpenCLParallelCalcNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
platforms/opencl/src/OpenCLParallelKernels.cpp:    dynamic_cast<const OpenCLCalcNonbondedForceKernel&>(kernels[0].getImpl()).getPMEParameters(alpha, nx, ny, nz);
platforms/opencl/src/OpenCLParallelKernels.cpp:void OpenCLParallelCalcNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
platforms/opencl/src/OpenCLParallelKernels.cpp:    dynamic_cast<const OpenCLCalcNonbondedForceKernel&>(kernels[0].getImpl()).getLJPMEParameters(alpha, nx, ny, nz);
platforms/opencl/src/OpenCLArray.cpp:#include "OpenCLArray.h"
platforms/opencl/src/OpenCLArray.cpp:#include "OpenCLContext.h"
platforms/opencl/src/OpenCLArray.cpp:OpenCLArray::OpenCLArray() : buffer(NULL), ownsBuffer(false) {
platforms/opencl/src/OpenCLArray.cpp:OpenCLArray::OpenCLArray(OpenCLContext& context, size_t size, int elementSize, const std::string& name, cl_int flags) : buffer(NULL) {
platforms/opencl/src/OpenCLArray.cpp:OpenCLArray::OpenCLArray(OpenCLContext& context, cl::Buffer* buffer, size_t size, int elementSize, const std::string& name) : buffer(NULL) {
platforms/opencl/src/OpenCLArray.cpp:OpenCLArray::~OpenCLArray() {
platforms/opencl/src/OpenCLArray.cpp:void OpenCLArray::initialize(ComputeContext& context, size_t size, int elementSize, const std::string& name) {
platforms/opencl/src/OpenCLArray.cpp:    initialize(dynamic_cast<OpenCLContext&>(context), size, elementSize, name, CL_MEM_READ_WRITE);
platforms/opencl/src/OpenCLArray.cpp:void OpenCLArray::initialize(OpenCLContext& context, size_t size, int elementSize, const std::string& name, cl_int flags) {
platforms/opencl/src/OpenCLArray.cpp:        throw OpenMMException("OpenCLArray has already been initialized");
platforms/opencl/src/OpenCLArray.cpp:void OpenCLArray::initialize(OpenCLContext& context, cl::Buffer* buffer, size_t size, int elementSize, const std::string& name) {
platforms/opencl/src/OpenCLArray.cpp:        throw OpenMMException("OpenCLArray has already been initialized");
platforms/opencl/src/OpenCLArray.cpp:void OpenCLArray::resize(size_t size) {
platforms/opencl/src/OpenCLArray.cpp:        throw OpenMMException("OpenCLArray has not been initialized");
platforms/opencl/src/OpenCLArray.cpp:ComputeContext& OpenCLArray::getContext() {
platforms/opencl/src/OpenCLArray.cpp:void OpenCLArray::uploadSubArray(const void* data, int offset, int elements, bool blocking) {
platforms/opencl/src/OpenCLArray.cpp:        throw OpenMMException("OpenCLArray has not been initialized");
platforms/opencl/src/OpenCLArray.cpp:void OpenCLArray::download(void* data, bool blocking) const {
platforms/opencl/src/OpenCLArray.cpp:        throw OpenMMException("OpenCLArray has not been initialized");
platforms/opencl/src/OpenCLArray.cpp:void OpenCLArray::copyTo(ArrayInterface& dest) const {
platforms/opencl/src/OpenCLArray.cpp:        throw OpenMMException("OpenCLArray has not been initialized");
platforms/opencl/src/OpenCLArray.cpp:    OpenCLArray& clDest = context->unwrap(dest);
platforms/opencl/src/OpenCLContext.cpp:#include "OpenCLContext.h"
platforms/opencl/src/OpenCLContext.cpp:#include "OpenCLArray.h"
platforms/opencl/src/OpenCLContext.cpp:#include "OpenCLBondedUtilities.h"
platforms/opencl/src/OpenCLContext.cpp:#include "OpenCLEvent.h"
platforms/opencl/src/OpenCLContext.cpp:#include "OpenCLForceInfo.h"
platforms/opencl/src/OpenCLContext.cpp:#include "OpenCLIntegrationUtilities.h"
platforms/opencl/src/OpenCLContext.cpp:#include "OpenCLKernelSources.h"
platforms/opencl/src/OpenCLContext.cpp:#include "OpenCLNonbondedUtilities.h"
platforms/opencl/src/OpenCLContext.cpp:#include "OpenCLProgram.h"
platforms/opencl/src/OpenCLContext.cpp:const int OpenCLContext::ThreadBlockSize = 64;
platforms/opencl/src/OpenCLContext.cpp:const int OpenCLContext::TileSize = 32;
platforms/opencl/src/OpenCLContext.cpp:    string skip = "OpenCL Build Warning : Compiler build log:";
platforms/opencl/src/OpenCLContext.cpp:    std::cerr << "OpenCL internal error: " << errinfo << std::endl;
platforms/opencl/src/OpenCLContext.cpp:    return (vendor.find("NVIDIA") == 0 ||
platforms/opencl/src/OpenCLContext.cpp:OpenCLContext::OpenCLContext(const System& system, int platformIndex, int deviceIndex, const string& precision, OpenCLPlatform::PlatformData& platformData, OpenCLContext* originalContext) :
platforms/opencl/src/OpenCLContext.cpp:            throw OpenMMException("Illegal value for OpenCLPlatformIndex: "+intToString(platformIndex));
platforms/opencl/src/OpenCLContext.cpp:            throw OpenMMException("Specified DeviceIndex but not OpenCLPlatformIndex.  When multiple platforms are available, a platform index is needed to specify a device.");
platforms/opencl/src/OpenCLContext.cpp:                if (devices[i].getInfo<CL_DEVICE_TYPE>() != CL_DEVICE_TYPE_GPU) {
platforms/opencl/src/OpenCLContext.cpp:                            // AMD GPUs either have a single VLIW SIMD or multiple scalar SIMDs.
platforms/opencl/src/OpenCLContext.cpp:            throw OpenMMException("No compatible OpenCL platform is available");
platforms/opencl/src/OpenCLContext.cpp:            throw OpenMMException("No compatible OpenCL device is available");
platforms/opencl/src/OpenCLContext.cpp:            cout << "WARNING: Using an unsupported OpenCL implementation.  Results may be incorrect." << endl;
platforms/opencl/src/OpenCLContext.cpp:            throw OpenMMException("The specified OpenCL device is not compatible with OpenMM");
platforms/opencl/src/OpenCLContext.cpp:            // 768 threads per GPU core.
platforms/opencl/src/OpenCLContext.cpp:        else if (vendor.size() >= 6 && vendor.substr(0, 6) == "NVIDIA") {
platforms/opencl/src/OpenCLContext.cpp:                // Compute level 1.2 and later Nvidia GPUs support 64 bit atomics, even though they don't list the
platforms/opencl/src/OpenCLContext.cpp:                // slow on earlier GPUs.
platforms/opencl/src/OpenCLContext.cpp:                    // Workaround for a bug in Maxwell on CUDA 6.x.
platforms/opencl/src/OpenCLContext.cpp:                    if (platformVersion.find("CUDA 6") != string::npos)
platforms/opencl/src/OpenCLContext.cpp:            if (device.getInfo<CL_DEVICE_TYPE>() != CL_DEVICE_TYPE_GPU) {
platforms/opencl/src/OpenCLContext.cpp:                /// \todo Is 6 a good value for the OpenCL CPU device?
platforms/opencl/src/OpenCLContext.cpp:                        // If the GPU has multiple SIMDs per compute unit then it is uses the scalar instruction
platforms/opencl/src/OpenCLContext.cpp:                                numThreadBlocksPerComputeUnit = 6*simdPerComputeUnit; // Navi seems to like more thread blocks than older GPUs
platforms/opencl/src/OpenCLContext.cpp:                        // Runtime does not support the query so is unlikely to be the newer scalar GPU.
platforms/opencl/src/OpenCLContext.cpp:    cl::Program utilities = createProgram(OpenCLKernelSources::utilities);
platforms/opencl/src/OpenCLContext.cpp:        OpenCLArray valuesArray(*this, 20, sizeof(mm_float8), "values");
platforms/opencl/src/OpenCLContext.cpp:    bonded = new OpenCLBondedUtilities(*this);
platforms/opencl/src/OpenCLContext.cpp:    nonbonded = new OpenCLNonbondedUtilities(*this);
platforms/opencl/src/OpenCLContext.cpp:    integration = new OpenCLIntegrationUtilities(*this, system);
platforms/opencl/src/OpenCLContext.cpp:    expression = new OpenCLExpressionUtilities(*this);
platforms/opencl/src/OpenCLContext.cpp:OpenCLContext::~OpenCLContext() {
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::initialize() {
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::initializeContexts() {
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::addForce(ComputeForceInfo* force) {
platforms/opencl/src/OpenCLContext.cpp:    OpenCLForceInfo* clinfo = dynamic_cast<OpenCLForceInfo*>(force);
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::requestForceBuffers(int minBuffers) {
platforms/opencl/src/OpenCLContext.cpp:cl::Program OpenCLContext::createProgram(const string source, const char* optimizationFlags) {
platforms/opencl/src/OpenCLContext.cpp:cl::Program OpenCLContext::createProgram(const string source, const map<string, string>& defines, const char* optimizationFlags) {
platforms/opencl/src/OpenCLContext.cpp:    src << OpenCLKernelSources::common << endl;
platforms/opencl/src/OpenCLContext.cpp:vector<ComputeContext*> OpenCLContext::getAllContexts() {
platforms/opencl/src/OpenCLContext.cpp:    for (OpenCLContext* c : platformData.contexts)
platforms/opencl/src/OpenCLContext.cpp:double& OpenCLContext::getEnergyWorkspace() {
platforms/opencl/src/OpenCLContext.cpp:cl::CommandQueue& OpenCLContext::getQueue() {
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::setQueue(cl::CommandQueue& queue) {
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::restoreDefaultQueue() {
platforms/opencl/src/OpenCLContext.cpp:OpenCLArray* OpenCLContext::createArray() {
platforms/opencl/src/OpenCLContext.cpp:    return new OpenCLArray();
platforms/opencl/src/OpenCLContext.cpp:ComputeEvent OpenCLContext::createEvent() {
platforms/opencl/src/OpenCLContext.cpp:    return shared_ptr<ComputeEventImpl>(new OpenCLEvent(*this));
platforms/opencl/src/OpenCLContext.cpp:ComputeProgram OpenCLContext::compileProgram(const std::string source, const std::map<std::string, std::string>& defines) {
platforms/opencl/src/OpenCLContext.cpp:    return shared_ptr<ComputeProgramImpl>(new OpenCLProgram(*this, program));
platforms/opencl/src/OpenCLContext.cpp:OpenCLArray& OpenCLContext::unwrap(ArrayInterface& array) const {
platforms/opencl/src/OpenCLContext.cpp:    OpenCLArray* clarray;
platforms/opencl/src/OpenCLContext.cpp:        clarray = dynamic_cast<OpenCLArray*>(&wrapper->getArray());
platforms/opencl/src/OpenCLContext.cpp:        clarray = dynamic_cast<OpenCLArray*>(&array);
platforms/opencl/src/OpenCLContext.cpp:        throw OpenMMException("Array argument is not an OpenCLArray");
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::executeKernel(cl::Kernel& kernel, int workUnits, int blockSize) {
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::printProfilingEvents() {
platforms/opencl/src/OpenCLContext.cpp:int OpenCLContext::computeThreadBlockSize(double memory) const {
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::clearBuffer(ArrayInterface& array) {
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::clearBuffer(cl::Memory& memory, int size) {
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::addAutoclearBuffer(ArrayInterface& array) {
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::addAutoclearBuffer(cl::Memory& memory, int size) {
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::clearAutoclearBuffers() {
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::reduceForces() {
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::reduceBuffer(OpenCLArray& array, OpenCLArray& longBuffer, int numBuffers) {
platforms/opencl/src/OpenCLContext.cpp:double OpenCLContext::reduceEnergy() {
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::setCharges(const vector<double>& charges) {
platforms/opencl/src/OpenCLContext.cpp:bool OpenCLContext::requestPosqCharges() {
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::addEnergyParameterDerivative(const string& param) {
platforms/opencl/src/OpenCLContext.cpp:void OpenCLContext::flushQueue() {
platforms/opencl/src/kernels/compact.cl:/* Code for CUDA stream compaction. Roughly based on:
platforms/opencl/src/kernels/compact.cl:          it'd be easy to take the CUDA SDK scanLargeArray sample, and do a prefix sum over dgBlockCounts in
platforms/opencl/src/kernels/compact.cl:  Author:       CUDA version by Imran Haque (ihaque@cs.stanford.edu), converted to OpenCL by Peter Eastman
platforms/opencl/src/kernels/compact.cl:// Taken from cuda SDK "scan" sample for naive scan, with small modifications
platforms/opencl/src/kernels/findInteractingBlocks_cpu.cl:#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
platforms/opencl/src/kernels/findInteractingBlocks_cpu.cl:#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
platforms/opencl/src/kernels/findInteractingBlocks.cl:#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
platforms/opencl/src/kernels/findInteractingBlocks.cl:#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
platforms/opencl/src/kernels/findInteractingBlocks.cl:// and slower on most GPUs.  On AMD, however, it is faster, so we keep it around to use there.
platforms/opencl/src/kernels/common.cl: * This file contains OpenCL definitions for the macros and functions needed for the
platforms/opencl/src/kernels/common.cl:#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
platforms/opencl/src/kernels/common.cl:#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
platforms/opencl/src/kernels/common.cl:// OpenCL has overloaded versions of standard math functions for single and double
platforms/opencl/src/kernels/common.cl:// precision arguments.  CUDA has separate functions.  To allow them to be called
platforms/opencl/src/kernels/sort.cl:#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
platforms/opencl/src/opencl.hpp: *   \brief C++ bindings for OpenCL 1.0, OpenCL 1.1, OpenCL 1.2,
platforms/opencl/src/opencl.hpp: *       OpenCL 2.0, OpenCL 2.1, OpenCL 2.2, and OpenCL 3.0.
platforms/opencl/src/opencl.hpp: *   Derived from the OpenCL 1.x C++ bindings written by
platforms/opencl/src/opencl.hpp: *       http://khronosgroup.github.io/OpenCL-CLHPP/
platforms/opencl/src/opencl.hpp: *       https://github.com/KhronosGroup/OpenCL-CLHPP/releases
platforms/opencl/src/opencl.hpp: *       https://github.com/KhronosGroup/OpenCL-CLHPP
platforms/opencl/src/opencl.hpp: * reasonable to define C++ bindings for OpenCL.
platforms/opencl/src/opencl.hpp: * The interface is contained with a single C++ header file \em opencl.hpp and all
platforms/opencl/src/opencl.hpp: * bindings; it is enough to simply include \em opencl.hpp.
platforms/opencl/src/opencl.hpp: * fixes in the new header as well as additional OpenCL 2.0 features.
platforms/opencl/src/opencl.hpp: * reason we release it as opencl.hpp rather than a new version of cl.hpp.
platforms/opencl/src/opencl.hpp: * Due to the evolution of the underlying OpenCL API the 2.0 C++ bindings
platforms/opencl/src/opencl.hpp: * and the range of valid underlying OpenCL runtime versions supported.
platforms/opencl/src/opencl.hpp: * The combination of preprocessor macros CL_HPP_TARGET_OPENCL_VERSION and 
platforms/opencl/src/opencl.hpp: * CL_HPP_MINIMUM_OPENCL_VERSION control this range. These are three digit
platforms/opencl/src/opencl.hpp: * decimal values representing OpenCL runime versions. The default for 
platforms/opencl/src/opencl.hpp: * the target is 200, representing OpenCL 2.0 and the minimum is also 
platforms/opencl/src/opencl.hpp: * The OpenCL 1.x versions of the C++ bindings included a size_t wrapper
platforms/opencl/src/opencl.hpp: * In OpenCL 2.0 OpenCL C is not entirely backward compatibility with 
platforms/opencl/src/opencl.hpp: * earlier versions. As a result a flag must be passed to the OpenCL C
platforms/opencl/src/opencl.hpp: * compiled to request OpenCL 2.0 compilation of kernels with 1.2 as
platforms/opencl/src/opencl.hpp: * For those cases the compilation defaults to OpenCL C 2.0.
platforms/opencl/src/opencl.hpp: * - CL_HPP_TARGET_OPENCL_VERSION
platforms/opencl/src/opencl.hpp: *   Defines the target OpenCL runtime version to build the header
platforms/opencl/src/opencl.hpp: *   against. Defaults to 200, representing OpenCL 2.0.
platforms/opencl/src/opencl.hpp: *   defined and may be defined by the user before opencl.hpp is
platforms/opencl/src/opencl.hpp: *   defined and may be defined by the user before opencl.hpp is
platforms/opencl/src/opencl.hpp: *   defined and may be defined by the user before opencl.hpp is
platforms/opencl/src/opencl.hpp: *   defined by the user before opencl.hpp is included.
platforms/opencl/src/opencl.hpp: *   Enables device fission for OpenCL 1.2 platforms.
platforms/opencl/src/opencl.hpp: *   Default to OpenCL C 1.2 compilation rather than OpenCL C 2.0
platforms/opencl/src/opencl.hpp:    #define CL_HPP_TARGET_OPENCL_VERSION 200
platforms/opencl/src/opencl.hpp:    #include <CL/opencl.hpp>
platforms/opencl/src/opencl.hpp:            if (platver.find("OpenCL 2.") != std::string::npos) {
platforms/opencl/src/opencl.hpp:            std::cout << "No OpenCL 2.0 platform found.";
platforms/opencl/src/opencl.hpp:# pragma message("opencl.hpp: USE_DX_INTEROP is deprecated. Define CL_HPP_USE_DX_INTEROP instead")
platforms/opencl/src/opencl.hpp:# pragma message("opencl.hpp: USE_CL_DEVICE_FISSION is deprecated. Define CL_HPP_USE_CL_DEVICE_FISSION instead")
platforms/opencl/src/opencl.hpp:# pragma message("opencl.hpp: __CL_ENABLE_EXCEPTIONS is deprecated. Define CL_HPP_ENABLE_EXCEPTIONS instead")
platforms/opencl/src/opencl.hpp:# pragma message("opencl.hpp: __NO_STD_VECTOR is deprecated. Define CL_HPP_NO_STD_VECTOR instead")
platforms/opencl/src/opencl.hpp:# pragma message("opencl.hpp: __NO_STD_STRING is deprecated. Define CL_HPP_NO_STD_STRING instead")
platforms/opencl/src/opencl.hpp:# pragma message("opencl.hpp: VECTOR_CLASS is deprecated. Alias cl::vector instead")
platforms/opencl/src/opencl.hpp:# pragma message("opencl.hpp: STRING_CLASS is deprecated. Alias cl::string instead.")
platforms/opencl/src/opencl.hpp:# pragma message("opencl.hpp: __CL_USER_OVERRIDE_ERROR_STRINGS is deprecated. Define CL_HPP_USER_OVERRIDE_ERROR_STRINGS instead")
platforms/opencl/src/opencl.hpp:# pragma message("opencl.hpp: __USE_DEV_VECTOR is no longer supported. Expect compilation errors")
platforms/opencl/src/opencl.hpp:# pragma message("opencl.hpp: __USE_DEV_STRING is no longer supported. Expect compilation errors")
platforms/opencl/src/opencl.hpp:#if !defined(CL_HPP_TARGET_OPENCL_VERSION)
platforms/opencl/src/opencl.hpp:# pragma message("opencl.hpp: CL_HPP_TARGET_OPENCL_VERSION is not defined. It will default to 300 (OpenCL 3.0)")
platforms/opencl/src/opencl.hpp:# define CL_HPP_TARGET_OPENCL_VERSION 300
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION != 100 && \
platforms/opencl/src/opencl.hpp:    CL_HPP_TARGET_OPENCL_VERSION != 110 && \
platforms/opencl/src/opencl.hpp:    CL_HPP_TARGET_OPENCL_VERSION != 120 && \
platforms/opencl/src/opencl.hpp:    CL_HPP_TARGET_OPENCL_VERSION != 200 && \
platforms/opencl/src/opencl.hpp:    CL_HPP_TARGET_OPENCL_VERSION != 210 && \
platforms/opencl/src/opencl.hpp:    CL_HPP_TARGET_OPENCL_VERSION != 220 && \
platforms/opencl/src/opencl.hpp:    CL_HPP_TARGET_OPENCL_VERSION != 300
platforms/opencl/src/opencl.hpp:# pragma message("opencl.hpp: CL_HPP_TARGET_OPENCL_VERSION is not a valid value (100, 110, 120, 200, 210, 220 or 300). It will be set to 300 (OpenCL 3.0).")
platforms/opencl/src/opencl.hpp:# undef CL_HPP_TARGET_OPENCL_VERSION
platforms/opencl/src/opencl.hpp:# define CL_HPP_TARGET_OPENCL_VERSION 300
platforms/opencl/src/opencl.hpp:/* Forward target OpenCL version to C headers if necessary */
platforms/opencl/src/opencl.hpp:#if defined(CL_TARGET_OPENCL_VERSION)
platforms/opencl/src/opencl.hpp:/* Warn if prior definition of CL_TARGET_OPENCL_VERSION is lower than
platforms/opencl/src/opencl.hpp:#if CL_TARGET_OPENCL_VERSION < CL_HPP_TARGET_OPENCL_VERSION
platforms/opencl/src/opencl.hpp:# pragma message("CL_TARGET_OPENCL_VERSION is already defined as is lower than CL_HPP_TARGET_OPENCL_VERSION")
platforms/opencl/src/opencl.hpp:# define CL_TARGET_OPENCL_VERSION CL_HPP_TARGET_OPENCL_VERSION
platforms/opencl/src/opencl.hpp:#if !defined(CL_HPP_MINIMUM_OPENCL_VERSION)
platforms/opencl/src/opencl.hpp:# define CL_HPP_MINIMUM_OPENCL_VERSION 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION != 100 && \
platforms/opencl/src/opencl.hpp:    CL_HPP_MINIMUM_OPENCL_VERSION != 110 && \
platforms/opencl/src/opencl.hpp:    CL_HPP_MINIMUM_OPENCL_VERSION != 120 && \
platforms/opencl/src/opencl.hpp:    CL_HPP_MINIMUM_OPENCL_VERSION != 200 && \
platforms/opencl/src/opencl.hpp:    CL_HPP_MINIMUM_OPENCL_VERSION != 210 && \
platforms/opencl/src/opencl.hpp:    CL_HPP_MINIMUM_OPENCL_VERSION != 220 && \
platforms/opencl/src/opencl.hpp:    CL_HPP_MINIMUM_OPENCL_VERSION != 300
platforms/opencl/src/opencl.hpp:# pragma message("opencl.hpp: CL_HPP_MINIMUM_OPENCL_VERSION is not a valid value (100, 110, 120, 200, 210, 220 or 300). It will be set to 100")
platforms/opencl/src/opencl.hpp:# undef CL_HPP_MINIMUM_OPENCL_VERSION
platforms/opencl/src/opencl.hpp:# define CL_HPP_MINIMUM_OPENCL_VERSION 100
platforms/opencl/src/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION > CL_HPP_TARGET_OPENCL_VERSION
platforms/opencl/src/opencl.hpp:# error "CL_HPP_MINIMUM_OPENCL_VERSION must not be greater than CL_HPP_TARGET_OPENCL_VERSION"
platforms/opencl/src/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 100 && !defined(CL_USE_DEPRECATED_OPENCL_1_0_APIS)
platforms/opencl/src/opencl.hpp:# define CL_USE_DEPRECATED_OPENCL_1_0_APIS
platforms/opencl/src/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 110 && !defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
platforms/opencl/src/opencl.hpp:# define CL_USE_DEPRECATED_OPENCL_1_1_APIS
platforms/opencl/src/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 120 && !defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
platforms/opencl/src/opencl.hpp:# define CL_USE_DEPRECATED_OPENCL_1_2_APIS
platforms/opencl/src/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 200 && !defined(CL_USE_DEPRECATED_OPENCL_2_0_APIS)
platforms/opencl/src/opencl.hpp:# define CL_USE_DEPRECATED_OPENCL_2_0_APIS
platforms/opencl/src/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 210 && !defined(CL_USE_DEPRECATED_OPENCL_2_1_APIS)
platforms/opencl/src/opencl.hpp:# define CL_USE_DEPRECATED_OPENCL_2_1_APIS
platforms/opencl/src/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 220 && !defined(CL_USE_DEPRECATED_OPENCL_2_2_APIS)
platforms/opencl/src/opencl.hpp:# define CL_USE_DEPRECATED_OPENCL_2_2_APIS
platforms/opencl/src/opencl.hpp:#include <OpenCL/opencl.h>
platforms/opencl/src/opencl.hpp:#include <CL/opencl.h>
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:        *  OpenCL C calls that require arrays of size_t values, whose
platforms/opencl/src/opencl.hpp: * \brief The OpenCL C++ bindings are defined within this namespace.
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 220
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
platforms/opencl/src/opencl.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
platforms/opencl/src/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
platforms/opencl/src/opencl.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:    F(cl_device_info, CL_DEVICE_OPENCL_C_VERSION, string) \
platforms/opencl/src/opencl.hpp:    F(cl_device_info, CL_DEVICE_OPENCL_C_NUMERIC_VERSION_KHR, cl_version_khr)
platforms/opencl/src/opencl.hpp:    F(cl_device_info, CL_DEVICE_OPENCL_C_ALL_VERSIONS, cl::vector<cl_name_version>) \
platforms/opencl/src/opencl.hpp:    F(cl_device_info, CL_DEVICE_OPENCL_C_FEATURES, cl::vector<cl_name_version>) \
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 220
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 220
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 300
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 300
platforms/opencl/src/opencl.hpp:#if defined(CL_HPP_USE_CL_SUB_GROUPS_KHR) && CL_HPP_TARGET_OPENCL_VERSION < 210
platforms/opencl/src/opencl.hpp:#endif // #if defined(CL_HPP_USE_CL_SUB_GROUPS_KHR) && CL_HPP_TARGET_OPENCL_VERSION < 210
platforms/opencl/src/opencl.hpp:// Flags deprecated in OpenCL 2.0
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 100 && CL_HPP_MINIMUM_OPENCL_VERSION < 200 && CL_HPP_TARGET_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 110
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 110 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION < 300
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION < 300
platforms/opencl/src/opencl.hpp:#ifdef CL_DEVICE_GPU_OVERLAP_NV
platforms/opencl/src/opencl.hpp:CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_GPU_OVERLAP_NV, cl_bool)
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp: * OpenCL 1.2 devices do have retain/release.
platforms/opencl/src/opencl.hpp:#else // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp: * OpenCL 1.1 devices do not have retain/release.
platforms/opencl/src/opencl.hpp:#endif // ! (CL_HPP_TARGET_OPENCL_VERSION >= 120)
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
platforms/opencl/src/opencl.hpp:#else // CL_HPP_MINIMUM_OPENCL_VERSION < 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:     *  \param devices returns a vector of OpenCL D3D10 devices found. The cl::Device
platforms/opencl/src/opencl.hpp:     *  values returned in devices can be used to identify a specific OpenCL
platforms/opencl/src/opencl.hpp:     *  The application can query specific capabilities of the OpenCL device(s)
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
platforms/opencl/src/opencl.hpp: * Unload the OpenCL compiler.
platforms/opencl/src/opencl.hpp: * \note Deprecated for OpenCL 1.2. Use Platform::unloadCompiler instead.
platforms/opencl/src/opencl.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:/*! \brief Class interface for creating OpenCL buffers from ID3D10Buffer's.
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
platforms/opencl/src/opencl.hpp:            useCreateImage = (version >= 0x10002); // OpenCL 1.2 or above
platforms/opencl/src/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 || defined(CL_HPP_USE_CL_IMAGE2D_FROM_BUFFER_KHR)
platforms/opencl/src/opencl.hpp:#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 200 || defined(CL_HPP_USE_CL_IMAGE2D_FROM_BUFFER_KHR)
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:    *              The channel order may differ as described in the OpenCL 
platforms/opencl/src/opencl.hpp:#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
platforms/opencl/src/opencl.hpp: *  \note Deprecated for OpenCL 1.2. Please use ImageGL instead.
platforms/opencl/src/opencl.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
platforms/opencl/src/opencl.hpp:            useCreateImage = (version >= 0x10002); // OpenCL 1.2 or above
platforms/opencl/src/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif  // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
platforms/opencl/src/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
platforms/opencl/src/opencl.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp: * was performed by OpenCL anyway.
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if (CL_HPP_TARGET_OPENCL_VERSION >= 200 && defined(CL_HPP_USE_CL_SUB_GROUPS_KHR)) || CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#else // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210 || (CL_HPP_TARGET_OPENCL_VERSION==200 && defined(CL_HPP_USE_IL_KHR))
platforms/opencl/src/opencl.hpp:     * Valid for either OpenCL >= 2.1 or when CL_HPP_USE_IL_KHR is defined.
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#else // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:     * Valid for either OpenCL >= 2.1 or when CL_HPP_USE_IL_KHR is defined.
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#else // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:     * \param context A valid OpenCL context in which to construct the program.
platforms/opencl/src/opencl.hpp:     * \param devices A vector of OpenCL device objects for which the program will be created.
platforms/opencl/src/opencl.hpp:     *   CL_INVALID_DEVICE if OpenCL devices listed in devices are not in the list of devices associated with context.
platforms/opencl/src/opencl.hpp:     *   CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required by the OpenCL implementation on the host.
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 220
platforms/opencl/src/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_2_2_APIS)
platforms/opencl/src/opencl.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_2_2_APIS)
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 220
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 220
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 220
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:                useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
platforms/opencl/src/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:               useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
platforms/opencl/src/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
platforms/opencl/src/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
platforms/opencl/src/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
platforms/opencl/src/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
platforms/opencl/src/opencl.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#else // CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:     *     The pattern type must be an accepted OpenCL data type.
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
platforms/opencl/src/opencl.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
platforms/opencl/src/opencl.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
platforms/opencl/src/opencl.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
platforms/opencl/src/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
platforms/opencl/src/opencl.hpp:#endif // defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
platforms/opencl/src/opencl.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 210
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp: * SVM buffer back to the OpenCL runtime.
platforms/opencl/src/opencl.hpp: * SVM buffer back to the OpenCL runtime.
platforms/opencl/src/opencl.hpp: * SVM buffer back to the OpenCL runtime.
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
platforms/opencl/src/opencl.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
platforms/opencl/src/opencl.hpp:     * Backward compatibility class to ensure that cl.hpp code works with opencl.hpp.
platforms/opencl/src/OpenCLKernel.cpp:#include "OpenCLKernel.h"
platforms/opencl/src/OpenCLKernel.cpp:OpenCLKernel::OpenCLKernel(OpenCLContext& context, cl::Kernel kernel) : context(context), kernel(kernel) {
platforms/opencl/src/OpenCLKernel.cpp:string OpenCLKernel::getName() const {
platforms/opencl/src/OpenCLKernel.cpp:int OpenCLKernel::getMaxBlockSize() const {
platforms/opencl/src/OpenCLKernel.cpp:void OpenCLKernel::execute(int threads, int blockSize) {
platforms/opencl/src/OpenCLKernel.cpp:    // Set args that are specified by OpenCLArrays.  We can't do this earlier, because it's
platforms/opencl/src/OpenCLKernel.cpp:void OpenCLKernel::addArrayArg(ArrayInterface& value) {
platforms/opencl/src/OpenCLKernel.cpp:void OpenCLKernel::addPrimitiveArg(const void* value, int size) {
platforms/opencl/src/OpenCLKernel.cpp:void OpenCLKernel::addEmptyArg() {
platforms/opencl/src/OpenCLKernel.cpp:void OpenCLKernel::setArrayArg(int index, ArrayInterface& value) {
platforms/opencl/src/OpenCLKernel.cpp:void OpenCLKernel::setPrimitiveArg(int index, const void* value, int size) {
platforms/opencl/src/OpenCLKernel.cpp:    // The const_cast is needed because of a bug in the OpenCL C++ wrappers.  clSetKernelArg()
platforms/opencl/src/OpenCLIntegrationUtilities.cpp:#include "OpenCLIntegrationUtilities.h"
platforms/opencl/src/OpenCLIntegrationUtilities.cpp:#include "OpenCLContext.h"
platforms/opencl/src/OpenCLIntegrationUtilities.cpp:OpenCLIntegrationUtilities::OpenCLIntegrationUtilities(OpenCLContext& context, const System& system) : IntegrationUtilities(context, system) {
platforms/opencl/src/OpenCLIntegrationUtilities.cpp:        // Different communication mechanisms give optimal performance on AMD and on NVIDIA.
platforms/opencl/src/OpenCLIntegrationUtilities.cpp:OpenCLArray& OpenCLIntegrationUtilities::getPosDelta() {
platforms/opencl/src/OpenCLIntegrationUtilities.cpp:    return dynamic_cast<OpenCLContext&>(context).unwrap(posDelta);
platforms/opencl/src/OpenCLIntegrationUtilities.cpp:OpenCLArray& OpenCLIntegrationUtilities::getRandom() {
platforms/opencl/src/OpenCLIntegrationUtilities.cpp:    return dynamic_cast<OpenCLContext&>(context).unwrap(random);
platforms/opencl/src/OpenCLIntegrationUtilities.cpp:OpenCLArray& OpenCLIntegrationUtilities::getStepSize() {
platforms/opencl/src/OpenCLIntegrationUtilities.cpp:    return dynamic_cast<OpenCLContext&>(context).unwrap(stepSize);
platforms/opencl/src/OpenCLIntegrationUtilities.cpp:void OpenCLIntegrationUtilities::applyConstraintsImpl(bool constrainVelocities, double tol) {
platforms/opencl/src/OpenCLIntegrationUtilities.cpp:            OpenCLContext& cl = dynamic_cast<OpenCLContext&>(context);
platforms/opencl/src/OpenCLIntegrationUtilities.cpp:void OpenCLIntegrationUtilities::distributeForcesFromVirtualSites() {
platforms/opencl/src/OpenCLKernelSources.cpp.in:#include "OpenCLKernelSources.h"
platforms/opencl/src/OpenCLFFT3D.cpp:#include "OpenCLFFT3D.h"
platforms/opencl/src/OpenCLFFT3D.cpp:#include "OpenCLContext.h"
platforms/opencl/src/OpenCLFFT3D.cpp:#include "OpenCLExpressionUtilities.h"
platforms/opencl/src/OpenCLFFT3D.cpp:#include "OpenCLKernelSources.h"
platforms/opencl/src/OpenCLFFT3D.cpp:OpenCLFFT3D::OpenCLFFT3D(OpenCLContext& context, int xsize, int ysize, int zsize, bool realToComplex) :
platforms/opencl/src/OpenCLFFT3D.cpp:OpenCLFFT3D::~OpenCLFFT3D() {
platforms/opencl/src/OpenCLFFT3D.cpp:void OpenCLFFT3D::execFFT(OpenCLArray& in, OpenCLArray& out, bool forward) {
platforms/opencl/src/OpenCLFFT3D.cpp:OpenCLFFT3D::OpenCLFFT3D(OpenCLContext& context, int xsize, int ysize, int zsize, bool realToComplex) :
platforms/opencl/src/OpenCLFFT3D.cpp:            cl::Program program = context.createProgram(OpenCLKernelSources::fftR2C, defines);
platforms/opencl/src/OpenCLFFT3D.cpp:void OpenCLFFT3D::execFFT(OpenCLArray& in, OpenCLArray& out, bool forward) {
platforms/opencl/src/OpenCLFFT3D.cpp:cl::Kernel OpenCLFFT3D::createKernel(int xsize, int ysize, int zsize, int& threads, int axis, bool forward, bool inputIsReal) {
platforms/opencl/src/OpenCLFFT3D.cpp:        cl::Program program = context.createProgram(context.replaceStrings(OpenCLKernelSources::fft, replacements));
platforms/opencl/src/OpenCLFFT3D.cpp:int OpenCLFFT3D::findLegalDimension(int minimum) {
platforms/opencl/src/OpenCLParameterSet.cpp:#include "OpenCLParameterSet.h"
platforms/opencl/src/OpenCLParameterSet.cpp:OpenCLParameterSet::OpenCLParameterSet(OpenCLContext& context, int numParameters, int numObjects, const string& name, bool bufferPerParameter, bool useDoublePrecision) :
platforms/opencl/src/OpenCLParameterSet.cpp:        buffers.push_back(OpenCLNonbondedUtilities::ParameterInfo(info.getName(), info.getComponentType(), info.getNumComponents(), info.getSize(), context.unwrap(info.getArray()).getDeviceBuffer()));
platforms/opencl/src/OpenCLEvent.cpp:#include "OpenCLEvent.h"
platforms/opencl/src/OpenCLEvent.cpp:OpenCLEvent::OpenCLEvent(OpenCLContext& context) : context(context) {
platforms/opencl/src/OpenCLEvent.cpp:void OpenCLEvent::enqueue() {
platforms/opencl/src/OpenCLEvent.cpp:void OpenCLEvent::wait() {
platforms/opencl/src/OpenCLKernels.cpp:#include "OpenCLKernels.h"
platforms/opencl/src/OpenCLKernels.cpp:#include "OpenCLForceInfo.h"
platforms/opencl/src/OpenCLKernels.cpp:#include "OpenCLBondedUtilities.h"
platforms/opencl/src/OpenCLKernels.cpp:#include "OpenCLExpressionUtilities.h"
platforms/opencl/src/OpenCLKernels.cpp:#include "OpenCLIntegrationUtilities.h"
platforms/opencl/src/OpenCLKernels.cpp:#include "OpenCLNonbondedUtilities.h"
platforms/opencl/src/OpenCLKernels.cpp:#include "OpenCLKernelSources.h"
platforms/opencl/src/OpenCLKernels.cpp:static void setPeriodicBoxSizeArg(OpenCLContext& cl, cl::Kernel& kernel, int index) {
platforms/opencl/src/OpenCLKernels.cpp:static void setPeriodicBoxArgs(OpenCLContext& cl, cl::Kernel& kernel, int index) {
platforms/opencl/src/OpenCLKernels.cpp:void OpenCLCalcForcesAndEnergyKernel::initialize(const System& system) {
platforms/opencl/src/OpenCLKernels.cpp:void OpenCLCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
platforms/opencl/src/OpenCLKernels.cpp:    OpenCLNonbondedUtilities& nb = cl.getNonbondedUtilities();
platforms/opencl/src/OpenCLKernels.cpp:double OpenCLCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups, bool& valid) {
platforms/opencl/src/OpenCLKernels.cpp:class OpenCLCalcNonbondedForceKernel::ForceInfo : public OpenCLForceInfo {
platforms/opencl/src/OpenCLKernels.cpp:    ForceInfo(int requiredBuffers, const NonbondedForce& force) : OpenCLForceInfo(requiredBuffers), force(force) {
platforms/opencl/src/OpenCLKernels.cpp:class OpenCLCalcNonbondedForceKernel::PmeIO : public CalcPmeReciprocalForceKernel::IO {
platforms/opencl/src/OpenCLKernels.cpp:    PmeIO(OpenCLContext& cl, cl::Kernel addForcesKernel) : cl(cl), addForcesKernel(addForcesKernel) {
platforms/opencl/src/OpenCLKernels.cpp:    OpenCLContext& cl;
platforms/opencl/src/OpenCLKernels.cpp:    OpenCLArray forceTemp;
platforms/opencl/src/OpenCLKernels.cpp:class OpenCLCalcNonbondedForceKernel::PmePreComputation : public OpenCLContext::ForcePreComputation {
platforms/opencl/src/OpenCLKernels.cpp:    PmePreComputation(OpenCLContext& cl, Kernel& pme, CalcPmeReciprocalForceKernel::IO& io) : cl(cl), pme(pme), io(io) {
platforms/opencl/src/OpenCLKernels.cpp:    OpenCLContext& cl;
platforms/opencl/src/OpenCLKernels.cpp:class OpenCLCalcNonbondedForceKernel::PmePostComputation : public OpenCLContext::ForcePostComputation {
platforms/opencl/src/OpenCLKernels.cpp:class OpenCLCalcNonbondedForceKernel::SyncQueuePreComputation : public OpenCLContext::ForcePreComputation {
platforms/opencl/src/OpenCLKernels.cpp:    SyncQueuePreComputation(OpenCLContext& cl, cl::CommandQueue queue, int forceGroup) : cl(cl), queue(queue), forceGroup(forceGroup) {
platforms/opencl/src/OpenCLKernels.cpp:    OpenCLContext& cl;
platforms/opencl/src/OpenCLKernels.cpp:class OpenCLCalcNonbondedForceKernel::SyncQueuePostComputation : public OpenCLContext::ForcePostComputation {
platforms/opencl/src/OpenCLKernels.cpp:    SyncQueuePostComputation(OpenCLContext& cl, cl::Event& event, OpenCLArray& pmeEnergyBuffer, int forceGroup) : cl(cl), event(event),
platforms/opencl/src/OpenCLKernels.cpp:    OpenCLContext& cl;
platforms/opencl/src/OpenCLKernels.cpp:    OpenCLArray& pmeEnergyBuffer;
platforms/opencl/src/OpenCLKernels.cpp:OpenCLCalcNonbondedForceKernel::~OpenCLCalcNonbondedForceKernel() {
platforms/opencl/src/OpenCLKernels.cpp:void OpenCLCalcNonbondedForceKernel::initialize(const System& system, const NonbondedForce& force) {
platforms/opencl/src/OpenCLKernels.cpp:        gridSizeX = OpenCLFFT3D::findLegalDimension(gridSizeX);
platforms/opencl/src/OpenCLKernels.cpp:        gridSizeY = OpenCLFFT3D::findLegalDimension(gridSizeY);
platforms/opencl/src/OpenCLKernels.cpp:        gridSizeZ = OpenCLFFT3D::findLegalDimension(gridSizeZ);
platforms/opencl/src/OpenCLKernels.cpp:            dispersionGridSizeX = OpenCLFFT3D::findLegalDimension(dispersionGridSizeX);
platforms/opencl/src/OpenCLKernels.cpp:            dispersionGridSizeY = OpenCLFFT3D::findLegalDimension(dispersionGridSizeY);
platforms/opencl/src/OpenCLKernels.cpp:            dispersionGridSizeZ = OpenCLFFT3D::findLegalDimension(dispersionGridSizeZ);
platforms/opencl/src/OpenCLKernels.cpp:                pmeEnergyBuffer.initialize(cl, cl.getNumThreadBlocks()*OpenCLContext::ThreadBlockSize, energyElementSize, "pmeEnergyBuffer");
platforms/opencl/src/OpenCLKernels.cpp:                sort = new OpenCLSort(cl, new SortTrait(), cl.getNumAtoms());
platforms/opencl/src/OpenCLKernels.cpp:                fft = new OpenCLFFT3D(cl, gridSizeX, gridSizeY, gridSizeZ, true);
platforms/opencl/src/OpenCLKernels.cpp:                    dispersionFft = new OpenCLFFT3D(cl, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, true);
platforms/opencl/src/OpenCLKernels.cpp:                bool isNvidia = (vendor.size() >= 6 && vendor.substr(0, 6) == "NVIDIA");
platforms/opencl/src/OpenCLKernels.cpp:                usePmeQueue = (!cl.getPlatformData().disablePmeStream && !cl.getPlatformData().useCpuPme && isNvidia);
platforms/opencl/src/OpenCLKernels.cpp:                    OpenCLArray *xmoduli, *ymoduli, *zmoduli;
platforms/opencl/src/OpenCLKernels.cpp:        cl.getNonbondedUtilities().addParameter(OpenCLNonbondedUtilities::ParameterInfo(prefix+"charge", "real", 1, charges.getElementSize(), charges.getDeviceBuffer()));
platforms/opencl/src/OpenCLKernels.cpp:        cl.getNonbondedUtilities().addParameter(OpenCLNonbondedUtilities::ParameterInfo(prefix+"sigmaEpsilon", "float", 2, sizeof(cl_float2), sigmaEpsilon.getDeviceBuffer()));
platforms/opencl/src/OpenCLKernels.cpp:double OpenCLCalcNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
platforms/opencl/src/OpenCLKernels.cpp:void OpenCLCalcNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const NonbondedForce& force, int firstParticle, int lastParticle, int firstException, int lastException) {
platforms/opencl/src/OpenCLKernels.cpp:void OpenCLCalcNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
platforms/opencl/src/OpenCLKernels.cpp:void OpenCLCalcNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
platforms/reference/src/SimTKReference/ReferencePME.cpp:         * Both for Cuda and modern CPUs it is nice to avoid conditionals, but we still need to apply periodic boundary conditions.
platforms/reference/src/SimTKReference/ReferencePME.cpp: * This probably very sub-optimal in Cuda? Separate kernel?
platforms/reference/src/SimTKReference/ReferencePME.cpp: * In practice, it might help to require order=4 for the cuda port.
platforms/reference/src/SimTKReference/ReferencePME.cpp:                 * In cuda you could probably work around this by setting something to 0.0 instead, but the short story is that we
platforms/common/include/openmm/common/ComputeContext.h:     * the thread.  Platforms that rely on binding contexts to threads (such as CUDA) need to
platforms/common/include/openmm/common/ComputeContext.h:     * This provides better interoperability with other libraries that use CUDA and create
platforms/common/include/openmm/common/ComputeContext.h:     * Platforms that rely on binding contexts to threads (such as CUDA) need to implement this.
platforms/common/include/openmm/common/ComputeContext.h:     * contexts to threads (such as CUDA) need to implement this.
platforms/common/include/openmm/common/ComputeContext.h:     * may be more efficient on CPUs and GPUs.
platforms/common/include/openmm/common/ComputeContext.h:     * Add a listener that should be called whenever atoms get reordered.  The OpenCLContext
platforms/common/include/openmm/common/ComputeContext.h:     * The OpenCLContext assumes ownership of the object, and deletes it when the context itself is deleted.
platforms/common/include/openmm/common/ComputeContext.h:     * The OpenCLContext assumes ownership of the object, and deletes it when the context itself is deleted.
platforms/common/include/openmm/common/ComputeArray.h: * array implementation (typically CudaArray or OpenCLArray).  This class can be used in code that
platforms/common/src/CommonKernels.cpp:        cc.getFloatForceBuffer(); // This will throw an exception on the CUDA platform.
platforms/common/src/CommonKernels.cpp:        // The CUDA platform doesn't have a floating point force buffer, so we don't need to copy it.
platforms/common/src/CommonKernels.cpp:    // When using multiple GPUs, this method is itself called from the worker thread.
platforms/common/src/kernels/customNonbondedGroups.cc:#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
platforms/common/src/kernels/customNonbondedGroups.cc:    // CUDA lets us do this slightly more efficiently by using shuffle operations.
platforms/common/src/kernels/customManyParticle.cc:#if !(defined(__CUDA_ARCH__) || defined(USE_HIP))
platforms/common/src/kernels/customManyParticle.cc:#if defined(__CUDA_ARCH__) || defined(USE_HIP)
platforms/cuda/staticTarget/CMakeLists.txt:# Include CUDA related files.
platforms/cuda/staticTarget/CMakeLists.txt:INCLUDE_DIRECTORIES(${CUDAToolkit_LIBRARY_DIR})
platforms/cuda/staticTarget/CMakeLists.txt:FILE(GLOB CUDA_KERNELS ${KERNEL_SOURCE_DIR}/kernels/*.cu)
platforms/cuda/staticTarget/CMakeLists.txt:    DEPENDS ${CUDA_KERNELS}
platforms/cuda/staticTarget/CMakeLists.txt:TARGET_LINK_LIBRARIES(${STATIC_TARGET} ${OPENMM_LIBRARY_NAME} CUDA::cuda_driver CUDA::cufft_static CUDA::nvrtc_static ${PTHREADS_LIB_STATIC})
platforms/cuda/staticTarget/CMakeLists.txt:    SET_TARGET_PROPERTIES(${STATIC_TARGET} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS} -F/Library/Frameworks -framework CUDA")
platforms/cuda/sharedTarget/CMakeLists.txt:# Include CUDA related files.
platforms/cuda/sharedTarget/CMakeLists.txt:INCLUDE_DIRECTORIES(${CUDAToolkit_LIBRARY_DIR})
platforms/cuda/sharedTarget/CMakeLists.txt:FILE(GLOB CUDA_KERNELS ${KERNEL_SOURCE_DIR}/kernels/*.cu)
platforms/cuda/sharedTarget/CMakeLists.txt:    DEPENDS ${CUDA_KERNELS}
platforms/cuda/sharedTarget/CMakeLists.txt:FIND_LIBRARY(NVRTC_LIB nvrtc PATHS "${CUDAToolkit_LIBRARY_DIR}")
platforms/cuda/sharedTarget/CMakeLists.txt:TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_LIBRARY_NAME} CUDA::cuda_driver CUDA::cufft ${NVRTC_LIB} ${PTHREADS_LIB})
platforms/cuda/sharedTarget/CMakeLists.txt:    SET_TARGET_PROPERTIES(${SHARED_TARGET} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS} -F/Library/Frameworks -framework CUDA")
platforms/cuda/tests/TestCudaATMForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaVirtualSites.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaCustomCentroidBondForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaMonteCarloFlexibleBarostat.cpp:#include "CudaTests.h"
platforms/cuda/tests/CudaTests.h:#include "CudaPlatform.h"
platforms/cuda/tests/CudaTests.h:OpenMM::CudaPlatform platform;
platforms/cuda/tests/TestCudaCompoundIntegrator.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaCustomAngleForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaVariableVerletIntegrator.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaLangevinIntegrator.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaCustomCVForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaCMMotionRemover.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaRBTorsionForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaLangevinMiddleIntegrator.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaHarmonicAngleForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaCustomManyParticleForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaMonteCarloBarostat.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaCustomCompoundBondForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaFFT3D.cpp: * This tests the CUDA implementation of FFT.
platforms/cuda/tests/TestCudaFFT3D.cpp:#include "CudaArray.h"
platforms/cuda/tests/TestCudaFFT3D.cpp:#include "CudaContext.h"
platforms/cuda/tests/TestCudaFFT3D.cpp:#include "CudaFFT3D.h"
platforms/cuda/tests/TestCudaFFT3D.cpp:#include "CudaSort.h"
platforms/cuda/tests/TestCudaFFT3D.cpp:static CudaPlatform platform;
platforms/cuda/tests/TestCudaFFT3D.cpp:    CudaPlatform::PlatformData platformData(NULL, system, "", "true", platform.getPropertyDefaultValue("CudaPrecision"), "false",
platforms/cuda/tests/TestCudaFFT3D.cpp:            platform.getPropertyDefaultValue(CudaPlatform::CudaTempDirectory()),
platforms/cuda/tests/TestCudaFFT3D.cpp:            platform.getPropertyDefaultValue(CudaPlatform::CudaDisablePmeStream()), "false", 1, NULL);
platforms/cuda/tests/TestCudaFFT3D.cpp:    CudaContext& context = *platformData.contexts[0];
platforms/cuda/tests/TestCudaFFT3D.cpp:    CudaArray grid1(context, original.size(), sizeof(Real2), "grid1");
platforms/cuda/tests/TestCudaFFT3D.cpp:    CudaArray grid2(context, original.size(), sizeof(Real2), "grid2");
platforms/cuda/tests/TestCudaFFT3D.cpp:    CudaFFT3D fft(context, xsize, ysize, zsize, realToComplex);
platforms/cuda/tests/TestCudaFFT3D.cpp:            platform.setPropertyDefaultValue("CudaPrecision", string(argv[1]));
platforms/cuda/tests/TestCudaFFT3D.cpp:        if (platform.getPropertyDefaultValue("CudaPrecision") == "double") {
platforms/cuda/tests/TestCudaCMAPTorsionForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaCustomExternalForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaSort.cpp: * This tests the CUDA implementation of sorting.
platforms/cuda/tests/TestCudaSort.cpp:#include "CudaArray.h"
platforms/cuda/tests/TestCudaSort.cpp:#include "CudaContext.h"
platforms/cuda/tests/TestCudaSort.cpp:#include "CudaSort.h"
platforms/cuda/tests/TestCudaSort.cpp:CudaPlatform platform;
platforms/cuda/tests/TestCudaSort.cpp:class SortTrait : public CudaSort::SortTrait {
platforms/cuda/tests/TestCudaSort.cpp:    CudaPlatform::PlatformData platformData(NULL, system, "", "true", platform.getPropertyDefaultValue("CudaPrecision"), "false",
platforms/cuda/tests/TestCudaSort.cpp:            platform.getPropertyDefaultValue(CudaPlatform::CudaTempDirectory()),
platforms/cuda/tests/TestCudaSort.cpp:            platform.getPropertyDefaultValue(CudaPlatform::CudaDisablePmeStream()), "false", 1, NULL);
platforms/cuda/tests/TestCudaSort.cpp:    CudaContext& context = *platformData.contexts[0];
platforms/cuda/tests/TestCudaSort.cpp:    CudaArray data(context, array.size(), 4, "sortData");
platforms/cuda/tests/TestCudaSort.cpp:    CudaSort sort(context, new SortTrait(), array.size(), uniform);
platforms/cuda/tests/TestCudaSort.cpp:            platform.setPropertyDefaultValue("CudaPrecision", string(argv[1]));
platforms/cuda/tests/TestCudaEwald.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaRandom.cpp: * This tests the CUDA implementation of random number generation.
platforms/cuda/tests/TestCudaRandom.cpp:#include "CudaArray.h"
platforms/cuda/tests/TestCudaRandom.cpp:#include "CudaContext.h"
platforms/cuda/tests/TestCudaRandom.cpp:#include "CudaIntegrationUtilities.h"
platforms/cuda/tests/TestCudaRandom.cpp:#include "CudaPlatform.h"
platforms/cuda/tests/TestCudaRandom.cpp:CudaPlatform platform;
platforms/cuda/tests/TestCudaRandom.cpp:    CudaPlatform::PlatformData platformData(NULL, system, "", "true", platform.getPropertyDefaultValue("CudaPrecision"), "false",
platforms/cuda/tests/TestCudaRandom.cpp:            platform.getPropertyDefaultValue(CudaPlatform::CudaTempDirectory()),
platforms/cuda/tests/TestCudaRandom.cpp:            platform.getPropertyDefaultValue(CudaPlatform::CudaDisablePmeStream()), "false", 1, NULL);
platforms/cuda/tests/TestCudaRandom.cpp:    CudaContext& context = *platformData.contexts[0];
platforms/cuda/tests/TestCudaRandom.cpp:    CudaArray& random = context.getIntegrationUtilities().getRandom();
platforms/cuda/tests/TestCudaRandom.cpp:            platform.setPropertyDefaultValue("CudaPrecision", string(argv[1]));
platforms/cuda/tests/TestCudaCustomGBForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaMultipleForces.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaCustomIntegrator.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaCustomHbondForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaCustomTorsionForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaSettle.cpp:#include "CudaTests.h"
platforms/cuda/tests/CMakeLists.txt:INCLUDE_DIRECTORIES(${CUDAToolkit_LIBRARY_DIR})
platforms/cuda/tests/CMakeLists.txt:SET(OPENMM_BUILD_CUDA_DOUBLE_PRECISION_TESTS TRUE CACHE BOOL "Whether to build double precision versions of CUDA test cases")
platforms/cuda/tests/CMakeLists.txt:        SET_TARGET_PROPERTIES(${TEST_ROOT} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS} -F/Library/Frameworks -framework CUDA" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")
platforms/cuda/tests/CMakeLists.txt:    IF (OPENMM_BUILD_CUDA_DOUBLE_PRECISION_TESTS)
platforms/cuda/tests/CMakeLists.txt:    ENDIF(OPENMM_BUILD_CUDA_DOUBLE_PRECISION_TESTS)
platforms/cuda/tests/TestCudaCheckpoints.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaHarmonicBondForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaVariableLangevinIntegrator.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaGayBerneForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaLocalEnergyMinimizer.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaMonteCarloAnisotropicBarostat.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaCustomCPPForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaBrownianIntegrator.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaDispersionPME.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaAndersenThermostat.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaPeriodicTorsionForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaRMSDForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaNonbondedForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaNonbondedForce.cpp:#include <cuda.h>
platforms/cuda/tests/TestCudaNonbondedForce.cpp:    // Check that the CudaDeterministicForces property works correctly.
platforms/cuda/tests/TestCudaNonbondedForce.cpp:    properties[CudaPlatform::CudaDeterministicForces()] = "true";
platforms/cuda/tests/TestCudaNonbondedForce.cpp:    int deviceIndex = stoi(platform.getPropertyValue(context, CudaPlatform::CudaDeviceIndex()));
platforms/cuda/tests/TestCudaVerletIntegrator.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaCustomBondForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaCustomNonbondedForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaGBSAOBCForce.cpp:#include "CudaTests.h"
platforms/cuda/tests/TestCudaNoseHooverIntegrator.cpp:#include "CudaTests.h"
platforms/cuda/include/CudaContext.h:#ifndef OPENMM_CUDACONTEXT_H_
platforms/cuda/include/CudaContext.h:#define OPENMM_CUDACONTEXT_H_
platforms/cuda/include/CudaContext.h:#include <cuda.h>
platforms/cuda/include/CudaContext.h:#include "CudaArray.h"
platforms/cuda/include/CudaContext.h:#include "CudaBondedUtilities.h"
platforms/cuda/include/CudaContext.h:#include "CudaExpressionUtilities.h"
platforms/cuda/include/CudaContext.h:#include "CudaIntegrationUtilities.h"
platforms/cuda/include/CudaContext.h:#include "CudaNonbondedUtilities.h"
platforms/cuda/include/CudaContext.h:#include "CudaPlatform.h"
platforms/cuda/include/CudaContext.h: * This class contains the information associated with a Context by the CUDA Platform.  Each CudaContext is
platforms/cuda/include/CudaContext.h: * in parallel on multiple devices, there is a separate CudaContext for each one.  The list of all contexts is
platforms/cuda/include/CudaContext.h: * stored in the CudaPlatform::PlatformData.
platforms/cuda/include/CudaContext.h: * In addition, a worker thread is created for each CudaContext.  This is used for parallel computations, so that
platforms/cuda/include/CudaContext.h:class OPENMM_EXPORT_COMMON CudaContext : public ComputeContext {
platforms/cuda/include/CudaContext.h:    CudaContext(const System& system, int deviceIndex, bool useBlockingSync, const std::string& precision,
platforms/cuda/include/CudaContext.h:            const std::string& tempDir, CudaPlatform::PlatformData& platformData, CudaContext* originalContext);
platforms/cuda/include/CudaContext.h:    ~CudaContext();
platforms/cuda/include/CudaContext.h:    CudaPlatform::PlatformData& getPlatformData() {
platforms/cuda/include/CudaContext.h:     * one CudaContext is created for each device.
platforms/cuda/include/CudaContext.h:    CudaArray* createArray();
platforms/cuda/include/CudaContext.h:     * Convert an array to an CudaArray.  If the argument is already an CudaArray, this simply casts it.
platforms/cuda/include/CudaContext.h:     * If the argument is a ComputeArray that wraps a CudaArray, this returns the wrapped array.  For any
platforms/cuda/include/CudaContext.h:    CudaArray& unwrap(ArrayInterface& array) const;
platforms/cuda/include/CudaContext.h:    CudaArray& getPosq() {
platforms/cuda/include/CudaContext.h:    CudaArray& getPosqCorrection() {
platforms/cuda/include/CudaContext.h:    CudaArray& getVelm() {
platforms/cuda/include/CudaContext.h:    CudaArray& getForce() {
platforms/cuda/include/CudaContext.h:     * The CUDA platform does not use floating point force buffers, so this throws an exception.
platforms/cuda/include/CudaContext.h:        throw OpenMMException("CUDA platform does not use floating point force buffers");
platforms/cuda/include/CudaContext.h:    CudaArray& getLongForceBuffer() {
platforms/cuda/include/CudaContext.h:     * All CUDA devices support 64 bit atomics, so this throws an exception.
platforms/cuda/include/CudaContext.h:        throw OpenMMException("CUDA platform does not use floating point force buffers");
platforms/cuda/include/CudaContext.h:    CudaArray& getEnergyBuffer() {
platforms/cuda/include/CudaContext.h:    CudaArray& getEnergyParamDerivBuffer() {
platforms/cuda/include/CudaContext.h:    CudaArray& getAtomIndexArray() {
platforms/cuda/include/CudaContext.h:     * Create a CUDA module from source code.
platforms/cuda/include/CudaContext.h:     * @param optimizationFlags  the optimization flags to pass to the CUDA compiler.  If this is
platforms/cuda/include/CudaContext.h:     * Create a CUDA module from source code.
platforms/cuda/include/CudaContext.h:     * @param optimizationFlags  the optimization flags to pass to the CUDA compiler.  If this is
platforms/cuda/include/CudaContext.h:     * Get a kernel from a CUDA module.
platforms/cuda/include/CudaContext.h:     * may be more efficient on CPUs and GPUs.
platforms/cuda/include/CudaContext.h:     * Convert a CUDA result code to the corresponding string description.
platforms/cuda/include/CudaContext.h:     * Get the CudaIntegrationUtilities for this context.
platforms/cuda/include/CudaContext.h:    CudaIntegrationUtilities& getIntegrationUtilities() {
platforms/cuda/include/CudaContext.h:     * Get the CudaExpressionUtilities for this context.
platforms/cuda/include/CudaContext.h:    CudaExpressionUtilities& getExpressionUtilities() {
platforms/cuda/include/CudaContext.h:     * Get the CudaBondedUtilities for this context.
platforms/cuda/include/CudaContext.h:    CudaBondedUtilities& getBondedUtilities() {
platforms/cuda/include/CudaContext.h:     * Get the CudaNonbondedUtilities for this context.
platforms/cuda/include/CudaContext.h:    CudaNonbondedUtilities& getNonbondedUtilities() {
platforms/cuda/include/CudaContext.h:    CudaNonbondedUtilities* createNonbondedUtilities() {
platforms/cuda/include/CudaContext.h:        return new CudaNonbondedUtilities(*this);
platforms/cuda/include/CudaContext.h:    static bool hasInitializedCuda;
platforms/cuda/include/CudaContext.h:    CudaPlatform::PlatformData& platformData;
platforms/cuda/include/CudaContext.h:    int gpuArchitecture;
platforms/cuda/include/CudaContext.h:    CudaArray posq;
platforms/cuda/include/CudaContext.h:    CudaArray posqCorrection;
platforms/cuda/include/CudaContext.h:    CudaArray velm;
platforms/cuda/include/CudaContext.h:    CudaArray force;
platforms/cuda/include/CudaContext.h:    CudaArray energyBuffer;
platforms/cuda/include/CudaContext.h:    CudaArray energySum;
platforms/cuda/include/CudaContext.h:    CudaArray energyParamDerivBuffer;
platforms/cuda/include/CudaContext.h:    CudaArray atomIndexDevice;
platforms/cuda/include/CudaContext.h:    CudaArray chargeBuffer;
platforms/cuda/include/CudaContext.h:    CudaIntegrationUtilities* integration;
platforms/cuda/include/CudaContext.h:    CudaExpressionUtilities* expression;
platforms/cuda/include/CudaContext.h:    CudaBondedUtilities* bonded;
platforms/cuda/include/CudaContext.h:    CudaNonbondedUtilities* nonbonded;
platforms/cuda/include/CudaContext.h:class OPENMM_EXPORT_COMMON CudaContext::WorkTask : public ComputeContext::WorkTask {
platforms/cuda/include/CudaContext.h:class OPENMM_EXPORT_COMMON CudaContext::ReorderListener : public ComputeContext::ReorderListener {
platforms/cuda/include/CudaContext.h:class OPENMM_EXPORT_COMMON CudaContext::ForcePreComputation : public ComputeContext::ForcePreComputation {
platforms/cuda/include/CudaContext.h:class OPENMM_EXPORT_COMMON CudaContext::ForcePostComputation : public ComputeContext::ForcePostComputation {
platforms/cuda/include/CudaContext.h:#endif /*OPENMM_CUDACONTEXT_H_*/
platforms/cuda/include/CudaArray.h:#ifndef OPENMM_CUDAARRAY_H_
platforms/cuda/include/CudaArray.h:#define OPENMM_CUDAARRAY_H_
platforms/cuda/include/CudaArray.h:#include <cuda.h>
platforms/cuda/include/CudaArray.h:class CudaContext;
platforms/cuda/include/CudaArray.h: * This class encapsulates a block of CUDA device memory.  It provides a simplified API
platforms/cuda/include/CudaArray.h:class OPENMM_EXPORT_COMMON CudaArray : public ArrayInterface {
platforms/cuda/include/CudaArray.h:     * Create a CudaArray object.  The object is allocated on the heap with the "new" operator.
platforms/cuda/include/CudaArray.h:    static CudaArray* create(CudaContext& context, size_t size, const std::string& name) {
platforms/cuda/include/CudaArray.h:        return new CudaArray(context, size, sizeof(T), name);
platforms/cuda/include/CudaArray.h:     * Create an uninitialized CudaArray object.  It does not point to any device memory,
platforms/cuda/include/CudaArray.h:    CudaArray();
platforms/cuda/include/CudaArray.h:     * Create a CudaArray object.
platforms/cuda/include/CudaArray.h:    CudaArray(CudaContext& context, size_t size, int elementSize, const std::string& name);
platforms/cuda/include/CudaArray.h:    ~CudaArray();
platforms/cuda/include/CudaArray.h:    CudaContext* context;
platforms/cuda/include/CudaArray.h:#endif /*OPENMM_CUDAARRAY_H_*/
platforms/cuda/include/CudaIntegrationUtilities.h:#ifndef OPENMM_CUDAINTEGRATIONUTILITIES_H_
platforms/cuda/include/CudaIntegrationUtilities.h:#define OPENMM_CUDAINTEGRATIONUTILITIES_H_
platforms/cuda/include/CudaIntegrationUtilities.h:#include "CudaArray.h"
platforms/cuda/include/CudaIntegrationUtilities.h:#include <cuda.h>
platforms/cuda/include/CudaIntegrationUtilities.h:class CudaContext;
platforms/cuda/include/CudaIntegrationUtilities.h:class OPENMM_EXPORT_COMMON CudaIntegrationUtilities : public IntegrationUtilities {
platforms/cuda/include/CudaIntegrationUtilities.h:    CudaIntegrationUtilities(CudaContext& context, const System& system);
platforms/cuda/include/CudaIntegrationUtilities.h:    ~CudaIntegrationUtilities();
platforms/cuda/include/CudaIntegrationUtilities.h:    CudaArray& getPosDelta();
platforms/cuda/include/CudaIntegrationUtilities.h:    CudaArray& getRandom();
platforms/cuda/include/CudaIntegrationUtilities.h:    CudaArray& getStepSize();
platforms/cuda/include/CudaIntegrationUtilities.h:#endif /*OPENMM_CUDAINTEGRATIONUTILITIES_H_*/
platforms/cuda/include/CudaPlatform.h:#ifndef OPENMM_CUDAPLATFORM_H_
platforms/cuda/include/CudaPlatform.h:#define OPENMM_CUDAPLATFORM_H_
platforms/cuda/include/CudaPlatform.h:class CudaContext;
platforms/cuda/include/CudaPlatform.h: * This Platform subclass uses CUDA implementations of the OpenMM kernels.
platforms/cuda/include/CudaPlatform.h:class OPENMM_EXPORT_COMMON CudaPlatform : public Platform {
platforms/cuda/include/CudaPlatform.h:    CudaPlatform();
platforms/cuda/include/CudaPlatform.h:        static const std::string name = "CUDA";
platforms/cuda/include/CudaPlatform.h:     * This is the name of the parameter for selecting which CUDA device or devices to use.
platforms/cuda/include/CudaPlatform.h:    static const std::string& CudaDeviceIndex() {
platforms/cuda/include/CudaPlatform.h:     * This is the name of the parameter that reports the CUDA device or devices being used.
platforms/cuda/include/CudaPlatform.h:    static const std::string& CudaDeviceName() {
platforms/cuda/include/CudaPlatform.h:     * This is the name of the parameter for selecting whether CUDA should sync or spin loop while waiting for results.
platforms/cuda/include/CudaPlatform.h:    static const std::string& CudaUseBlockingSync() {
platforms/cuda/include/CudaPlatform.h:    static const std::string& CudaPrecision() {
platforms/cuda/include/CudaPlatform.h:    static const std::string& CudaUseCpuPme() {
platforms/cuda/include/CudaPlatform.h:    static const std::string& CudaCompiler() {
platforms/cuda/include/CudaPlatform.h:        static const std::string key = "CudaCompiler";
platforms/cuda/include/CudaPlatform.h:    static const std::string& CudaHostCompiler() {
platforms/cuda/include/CudaPlatform.h:        static const std::string key = "CudaHostCompiler";
platforms/cuda/include/CudaPlatform.h:    static const std::string& CudaTempDirectory() {
platforms/cuda/include/CudaPlatform.h:    static const std::string& CudaDisablePmeStream() {
platforms/cuda/include/CudaPlatform.h:    static const std::string& CudaDeterministicForces() {
platforms/cuda/include/CudaPlatform.h:class OPENMM_EXPORT_COMMON CudaPlatform::PlatformData {
platforms/cuda/include/CudaPlatform.h:    std::vector<CudaContext*> contexts;
platforms/cuda/include/CudaPlatform.h:#endif /*OPENMM_CUDAPLATFORM_H_*/
platforms/cuda/include/CudaNonbondedUtilities.h:#ifndef OPENMM_CUDANONBONDEDUTILITIES_H_
platforms/cuda/include/CudaNonbondedUtilities.h:#define OPENMM_CUDANONBONDEDUTILITIES_H_
platforms/cuda/include/CudaNonbondedUtilities.h:#include "CudaArray.h"
platforms/cuda/include/CudaNonbondedUtilities.h:#include "CudaExpressionUtilities.h"
platforms/cuda/include/CudaNonbondedUtilities.h:#include <cuda.h>
platforms/cuda/include/CudaNonbondedUtilities.h:class CudaContext;
platforms/cuda/include/CudaNonbondedUtilities.h:class CudaSort;
platforms/cuda/include/CudaNonbondedUtilities.h:class OPENMM_EXPORT_COMMON CudaNonbondedUtilities : public NonbondedUtilities  {
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaNonbondedUtilities(CudaContext& context);
platforms/cuda/include/CudaNonbondedUtilities.h:    ~CudaNonbondedUtilities();
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray& getBlockCenters() {
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray& getBlockBoundingBoxes() {
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray& getInteractionCount() {
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray& getInteractingTiles() {
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray& getInteractingAtoms() {
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray& getSinglePairs() {
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray& getExclusions() {
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray& getExclusionTiles() {
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray& getExclusionIndices() {
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray& getExclusionRowIndices() {
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray& getRebuildNeighborList() {
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaContext& context;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray exclusionTiles;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray exclusions;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray exclusionIndices;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray exclusionRowIndices;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray interactingTiles;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray interactingAtoms;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray interactionCount;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray singlePairs;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray singlePairCount;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray blockCenter;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray blockBoundingBox;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray sortedBlocks;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray sortedBlockCenter;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray sortedBlockBoundingBox;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray blockSizeRange;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray largeBlockCenter;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray largeBlockBoundingBox;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray oldPositions;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaArray rebuildNeighborList;
platforms/cuda/include/CudaNonbondedUtilities.h:    CudaSort* blockSorter;
platforms/cuda/include/CudaNonbondedUtilities.h:class CudaNonbondedUtilities::KernelSet {
platforms/cuda/include/CudaNonbondedUtilities.h:class CudaNonbondedUtilities::ParameterInfo {
platforms/cuda/include/CudaNonbondedUtilities.h:#endif /*OPENMM_CUDANONBONDEDUTILITIES_H_*/
platforms/cuda/include/CudaProgram.h:#ifndef OPENMM_CUDAPROGRAM_H_
platforms/cuda/include/CudaProgram.h:#define OPENMM_CUDAPROGRAM_H_
platforms/cuda/include/CudaProgram.h:#include "CudaContext.h"
platforms/cuda/include/CudaProgram.h: * This is the CUDA implementation of the ComputeProgramImpl interface. 
platforms/cuda/include/CudaProgram.h:class CudaProgram : public ComputeProgramImpl {
platforms/cuda/include/CudaProgram.h:     * Create a new CudaProgram.
platforms/cuda/include/CudaProgram.h:    CudaProgram(CudaContext& context, CUmodule module);
platforms/cuda/include/CudaProgram.h:    CudaContext& context;
platforms/cuda/include/CudaProgram.h:#endif /*OPENMM_CUDAPROGRAM_H_*/
platforms/cuda/include/CudaParallelKernels.h:#ifndef OPENMM_CUDAPARALLELKERNELS_H_
platforms/cuda/include/CudaParallelKernels.h:#define OPENMM_CUDAPARALLELKERNELS_H_
platforms/cuda/include/CudaParallelKernels.h:#include "CudaPlatform.h"
platforms/cuda/include/CudaParallelKernels.h:#include "CudaContext.h"
platforms/cuda/include/CudaParallelKernels.h:#include "CudaKernels.h"
platforms/cuda/include/CudaParallelKernels.h:class CudaParallelCalcForcesAndEnergyKernel : public CalcForcesAndEnergyKernel {
platforms/cuda/include/CudaParallelKernels.h:    CudaParallelCalcForcesAndEnergyKernel(std::string name, const Platform& platform, CudaPlatform::PlatformData& data);
platforms/cuda/include/CudaParallelKernels.h:    ~CudaParallelCalcForcesAndEnergyKernel();
platforms/cuda/include/CudaParallelKernels.h:    CudaCalcForcesAndEnergyKernel& getKernel(int index) {
platforms/cuda/include/CudaParallelKernels.h:        return dynamic_cast<CudaCalcForcesAndEnergyKernel&>(kernels[index].getImpl());
platforms/cuda/include/CudaParallelKernels.h:    CudaPlatform::PlatformData& data;
platforms/cuda/include/CudaParallelKernels.h:    CudaArray contextForces;
platforms/cuda/include/CudaParallelKernels.h:class CudaParallelCalcNonbondedForceKernel : public CalcNonbondedForceKernel {
platforms/cuda/include/CudaParallelKernels.h:    CudaParallelCalcNonbondedForceKernel(std::string name, const Platform& platform, CudaPlatform::PlatformData& data, const System& system);
platforms/cuda/include/CudaParallelKernels.h:    CudaCalcNonbondedForceKernel& getKernel(int index) {
platforms/cuda/include/CudaParallelKernels.h:        return dynamic_cast<CudaCalcNonbondedForceKernel&>(kernels[index].getImpl());
platforms/cuda/include/CudaParallelKernels.h:    CudaPlatform::PlatformData& data;
platforms/cuda/include/CudaParallelKernels.h:#endif /*OPENMM_CUDAPARALLELKERNELS_H_*/
platforms/cuda/include/CudaForceInfo.h:#ifndef OPENMM_CUDAFORCEINFO_H_
platforms/cuda/include/CudaForceInfo.h:#define OPENMM_CUDAFORCEINFO_H_
platforms/cuda/include/CudaForceInfo.h:class OPENMM_EXPORT_COMMON CudaForceInfo : public ComputeForceInfo {
platforms/cuda/include/CudaForceInfo.h:#endif /*OPENMM_CUDAFORCEINFO_H_*/
platforms/cuda/include/CudaFFT3D.h:#ifndef __OPENMM_CUDAFFT3D_H__
platforms/cuda/include/CudaFFT3D.h:#define __OPENMM_CUDAFFT3D_H__
platforms/cuda/include/CudaFFT3D.h:#include "CudaArray.h"
platforms/cuda/include/CudaFFT3D.h:class OPENMM_EXPORT_COMMON CudaFFT3D {
platforms/cuda/include/CudaFFT3D.h:     * Create an CudaFFT3D object for performing transforms of a particular size.
platforms/cuda/include/CudaFFT3D.h:    CudaFFT3D(CudaContext& context, int xsize, int ysize, int zsize, bool realToComplex=false);
platforms/cuda/include/CudaFFT3D.h:    void execFFT(CudaArray& in, CudaArray& out, bool forward = true);
platforms/cuda/include/CudaFFT3D.h:    CudaContext& context;
platforms/cuda/include/CudaFFT3D.h:#endif // __OPENMM_CUDAFFT3D_H__
platforms/cuda/include/CudaBondedUtilities.h:#ifndef OPENMM_CUDABONDEDUTILITIES_H_
platforms/cuda/include/CudaBondedUtilities.h:#define OPENMM_CUDABONDEDUTILITIES_H_
platforms/cuda/include/CudaBondedUtilities.h:class OPENMM_EXPORT_COMMON CudaBondedUtilities : public BondedUtilities {
platforms/cuda/include/CudaBondedUtilities.h:    CudaBondedUtilities(ComputeContext& context) : BondedUtilities(context) {
platforms/cuda/include/CudaBondedUtilities.h:#endif /*OPENMM_CUDABONDEDUTILITIES_H_*/
platforms/cuda/include/CudaExpressionUtilities.h:#ifndef OPENMM_CUDAEXPRESSIONUTILITIES_H_
platforms/cuda/include/CudaExpressionUtilities.h:#define OPENMM_CUDAEXPRESSIONUTILITIES_H_
platforms/cuda/include/CudaExpressionUtilities.h:class OPENMM_EXPORT_COMMON CudaExpressionUtilities : public ExpressionUtilities {
platforms/cuda/include/CudaExpressionUtilities.h:    CudaExpressionUtilities(ComputeContext& context) : ExpressionUtilities(context) {
platforms/cuda/include/CudaExpressionUtilities.h:#endif /*OPENMM_CUDAEXPRESSIONUTILITIES_H_*/
platforms/cuda/include/CudaEvent.h:#ifndef OPENMM_CUDAEVENT_H_
platforms/cuda/include/CudaEvent.h:#define OPENMM_CUDAEVENT_H_
platforms/cuda/include/CudaEvent.h:#include "CudaContext.h"
platforms/cuda/include/CudaEvent.h: * This is the CUDA implementation of the ComputeKernelImpl interface. 
platforms/cuda/include/CudaEvent.h:class CudaEvent : public ComputeEventImpl {
platforms/cuda/include/CudaEvent.h:    CudaEvent(CudaContext& context);
platforms/cuda/include/CudaEvent.h:    ~CudaEvent();
platforms/cuda/include/CudaEvent.h:    CudaContext& context;
platforms/cuda/include/CudaEvent.h:#endif /*OPENMM_CUDAEVENT_H_*/
platforms/cuda/include/CudaSort.h:#ifndef __OPENMM_CUDASORT_H__
platforms/cuda/include/CudaSort.h:#define __OPENMM_CUDASORT_H__
platforms/cuda/include/CudaSort.h:#include "CudaArray.h"
platforms/cuda/include/CudaSort.h:#include "CudaContext.h"
platforms/cuda/include/CudaSort.h: * class FloatTrait : public CudaSort::SortTrait {
platforms/cuda/include/CudaSort.h: * Sorting Algorithm with CUDA"  Journal of the Chinese Institute of Engineers, 32(7),
platforms/cuda/include/CudaSort.h:class OPENMM_EXPORT_COMMON CudaSort {
platforms/cuda/include/CudaSort.h:     * Create a CudaSort object for sorting data of a particular type.
platforms/cuda/include/CudaSort.h:     *                   and deletes it when the CudaSort is deleted.
platforms/cuda/include/CudaSort.h:    CudaSort(CudaContext& context, SortTrait* trait, unsigned int length, bool uniform=true);
platforms/cuda/include/CudaSort.h:    ~CudaSort();
platforms/cuda/include/CudaSort.h:    void sort(CudaArray& data);
platforms/cuda/include/CudaSort.h:    CudaContext& context;
platforms/cuda/include/CudaSort.h:    CudaArray dataRange;
platforms/cuda/include/CudaSort.h:    CudaArray bucketOfElement;
platforms/cuda/include/CudaSort.h:    CudaArray offsetInBucket;
platforms/cuda/include/CudaSort.h:    CudaArray bucketOffset;
platforms/cuda/include/CudaSort.h:    CudaArray buckets;
platforms/cuda/include/CudaSort.h:class CudaSort::SortTrait {
platforms/cuda/include/CudaSort.h:     * Get the CUDA code to select the key from the data value.
platforms/cuda/include/CudaSort.h:#endif // __OPENMM_CUDASORT_H__
platforms/cuda/include/CudaKernels.h:#ifndef OPENMM_CUDAKERNELS_H_
platforms/cuda/include/CudaKernels.h:#define OPENMM_CUDAKERNELS_H_
platforms/cuda/include/CudaKernels.h:#include "CudaPlatform.h"
platforms/cuda/include/CudaKernels.h:#include "CudaArray.h"
platforms/cuda/include/CudaKernels.h:#include "CudaContext.h"
platforms/cuda/include/CudaKernels.h:#include "CudaFFT3D.h"
platforms/cuda/include/CudaKernels.h:#include "CudaSort.h"
platforms/cuda/include/CudaKernels.h:class CudaCalcForcesAndEnergyKernel : public CalcForcesAndEnergyKernel {
platforms/cuda/include/CudaKernels.h:    CudaCalcForcesAndEnergyKernel(std::string name, const Platform& platform, CudaContext& cu) : CalcForcesAndEnergyKernel(name, platform), cu(cu) {
platforms/cuda/include/CudaKernels.h:   CudaContext& cu;
platforms/cuda/include/CudaKernels.h:class CudaCalcNonbondedForceKernel : public CalcNonbondedForceKernel {
platforms/cuda/include/CudaKernels.h:    CudaCalcNonbondedForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) : CalcNonbondedForceKernel(name, platform),
platforms/cuda/include/CudaKernels.h:    ~CudaCalcNonbondedForceKernel();
platforms/cuda/include/CudaKernels.h:    class SortTrait : public CudaSort::SortTrait {
platforms/cuda/include/CudaKernels.h:    CudaContext& cu;
platforms/cuda/include/CudaKernels.h:    CudaArray charges;
platforms/cuda/include/CudaKernels.h:    CudaArray sigmaEpsilon;
platforms/cuda/include/CudaKernels.h:    CudaArray exceptionParams;
platforms/cuda/include/CudaKernels.h:    CudaArray exclusionAtoms;
platforms/cuda/include/CudaKernels.h:    CudaArray exclusionParams;
platforms/cuda/include/CudaKernels.h:    CudaArray baseParticleParams;
platforms/cuda/include/CudaKernels.h:    CudaArray baseExceptionParams;
platforms/cuda/include/CudaKernels.h:    CudaArray particleParamOffsets;
platforms/cuda/include/CudaKernels.h:    CudaArray exceptionParamOffsets;
platforms/cuda/include/CudaKernels.h:    CudaArray particleOffsetIndices;
platforms/cuda/include/CudaKernels.h:    CudaArray exceptionOffsetIndices;
platforms/cuda/include/CudaKernels.h:    CudaArray globalParams;
platforms/cuda/include/CudaKernels.h:    CudaArray cosSinSums;
platforms/cuda/include/CudaKernels.h:    CudaArray pmeGrid1;
platforms/cuda/include/CudaKernels.h:    CudaArray pmeGrid2;
platforms/cuda/include/CudaKernels.h:    CudaArray pmeBsplineModuliX;
platforms/cuda/include/CudaKernels.h:    CudaArray pmeBsplineModuliY;
platforms/cuda/include/CudaKernels.h:    CudaArray pmeBsplineModuliZ;
platforms/cuda/include/CudaKernels.h:    CudaArray pmeDispersionBsplineModuliX;
platforms/cuda/include/CudaKernels.h:    CudaArray pmeDispersionBsplineModuliY;
platforms/cuda/include/CudaKernels.h:    CudaArray pmeDispersionBsplineModuliZ;
platforms/cuda/include/CudaKernels.h:    CudaArray pmeAtomGridIndex;
platforms/cuda/include/CudaKernels.h:    CudaArray pmeEnergyBuffer;
platforms/cuda/include/CudaKernels.h:    CudaSort* sort;
platforms/cuda/include/CudaKernels.h:    CudaFFT3D* fft;
platforms/cuda/include/CudaKernels.h:    CudaFFT3D* dispersionFft;
platforms/cuda/include/CudaKernels.h:    bool hasCoulomb, hasLJ, useFixedPointChargeSpreading, usePmeStream, useCudaFFT, doLJPME, usePosqCharges, recomputeParams, hasOffsets;
platforms/cuda/include/CudaKernels.h:class CudaCalcCustomCVForceKernel : public CommonCalcCustomCVForceKernel {
platforms/cuda/include/CudaKernels.h:    CudaCalcCustomCVForceKernel(std::string name, const Platform& platform, ComputeContext& cc) : CommonCalcCustomCVForceKernel(name, platform, cc) {
platforms/cuda/include/CudaKernels.h:        return *reinterpret_cast<CudaPlatform::PlatformData*>(innerContext.getPlatformData())->contexts[0];
platforms/cuda/include/CudaKernels.h:class CudaCalcATMForceKernel : public CommonCalcATMForceKernel {
platforms/cuda/include/CudaKernels.h:    CudaCalcATMForceKernel(std::string name, const Platform& platform, ComputeContext& cc) : CommonCalcATMForceKernel(name, platform, cc) {
platforms/cuda/include/CudaKernels.h:        return *reinterpret_cast<CudaPlatform::PlatformData*>(innerContext.getPlatformData())->contexts[0];
platforms/cuda/include/CudaKernels.h:#endif /*OPENMM_CUDAKERNELS_H_*/
platforms/cuda/include/CudaKernel.h:#ifndef OPENMM_CUDAKERNEL_H_
platforms/cuda/include/CudaKernel.h:#define OPENMM_CUDAKERNEL_H_
platforms/cuda/include/CudaKernel.h:#include "CudaArray.h"
platforms/cuda/include/CudaKernel.h:#include "CudaContext.h"
platforms/cuda/include/CudaKernel.h: * This is the CUDA implementation of the ComputeKernelImpl interface. 
platforms/cuda/include/CudaKernel.h:class CudaKernel : public ComputeKernelImpl {
platforms/cuda/include/CudaKernel.h:     * Create a new CudaKernel.
platforms/cuda/include/CudaKernel.h:    CudaKernel(CudaContext& context, CUfunction kernel, const std::string& name);
platforms/cuda/include/CudaKernel.h:    CudaContext& context;
platforms/cuda/include/CudaKernel.h:    std::vector<CudaArray*> arrayArgs;
platforms/cuda/include/CudaKernel.h:#endif /*OPENMM_CUDAKERNEL_H_*/
platforms/cuda/include/CudaKernelFactory.h:#ifndef OPENMM_CUDAKERNELFACTORY_H_
platforms/cuda/include/CudaKernelFactory.h:#define OPENMM_CUDAKERNELFACTORY_H_
platforms/cuda/include/CudaKernelFactory.h: * This KernelFactory creates all kernels for CudaPlatform.
platforms/cuda/include/CudaKernelFactory.h:class CudaKernelFactory : public KernelFactory {
platforms/cuda/include/CudaKernelFactory.h:#endif /*OPENMM_CUDAKERNELFACTORY_H_*/
platforms/cuda/include/CudaParameterSet.h:#ifndef OPENMM_CUDAPARAMETERSET_H_
platforms/cuda/include/CudaParameterSet.h:#define OPENMM_CUDAPARAMETERSET_H_
platforms/cuda/include/CudaParameterSet.h:#include "CudaContext.h"
platforms/cuda/include/CudaParameterSet.h:#include "CudaNonbondedUtilities.h"
platforms/cuda/include/CudaParameterSet.h:class CudaNonbondedUtilities;
platforms/cuda/include/CudaParameterSet.h:class OPENMM_EXPORT_COMMON CudaParameterSet : public ComputeParameterSet {
platforms/cuda/include/CudaParameterSet.h:     * Create an CudaParameterSet.
platforms/cuda/include/CudaParameterSet.h:    CudaParameterSet(CudaContext& context, int numParameters, int numObjects, const std::string& name, bool bufferPerParameter=false, bool useDoublePrecision=false);
platforms/cuda/include/CudaParameterSet.h:     * Get a set of CudaNonbondedUtilities::ParameterInfo objects which describe the Buffers
platforms/cuda/include/CudaParameterSet.h:    std::vector<CudaNonbondedUtilities::ParameterInfo>& getBuffers() {
platforms/cuda/include/CudaParameterSet.h:    std::vector<CudaNonbondedUtilities::ParameterInfo> buffers;
platforms/cuda/include/CudaParameterSet.h:#endif /*OPENMM_CUDAPARAMETERSET_H_*/
platforms/cuda/CMakeLists.txt:# OpenMM CUDA Platform
platforms/cuda/CMakeLists.txt:# Creates OpenMMCUDA library.
platforms/cuda/CMakeLists.txt:#   OpenMMCUDA.dll
platforms/cuda/CMakeLists.txt:#   OpenMMCUDA.lib
platforms/cuda/CMakeLists.txt:#   OpenMMCUDA_static.lib
platforms/cuda/CMakeLists.txt:#   libOpenMMCUDA.so
platforms/cuda/CMakeLists.txt:#   libOpenMMCUDA_static.a
platforms/cuda/CMakeLists.txt:set(OPENMM_BUILD_CUDA_TESTS TRUE CACHE BOOL "Whether to build CUDA test cases")
platforms/cuda/CMakeLists.txt:if(BUILD_TESTING AND OPENMM_BUILD_CUDA_TESTS)
platforms/cuda/CMakeLists.txt:endif(BUILD_TESTING AND OPENMM_BUILD_CUDA_TESTS)
platforms/cuda/CMakeLists.txt:SET(OPENMMCUDA_LIBRARY_NAME OpenMMCUDA)
platforms/cuda/CMakeLists.txt:SET(SHARED_TARGET ${OPENMMCUDA_LIBRARY_NAME})
platforms/cuda/CMakeLists.txt:SET(STATIC_TARGET ${OPENMMCUDA_LIBRARY_NAME}_static)
platforms/cuda/CMakeLists.txt:SET(KERNEL_SOURCE_CLASS CudaKernelSources)
platforms/cuda/CMakeLists.txt:INSTALL_FILES(/include/openmm/cuda FILES ${CORE_HEADERS})
platforms/cuda/src/CudaKernelFactory.cpp:#include "CudaKernelFactory.h"
platforms/cuda/src/CudaKernelFactory.cpp:#include "CudaKernels.h"
platforms/cuda/src/CudaKernelFactory.cpp:#include "CudaParallelKernels.h"
platforms/cuda/src/CudaKernelFactory.cpp:#include "CudaPlatform.h"
platforms/cuda/src/CudaKernelFactory.cpp:KernelImpl* CudaKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
platforms/cuda/src/CudaKernelFactory.cpp:    CudaPlatform::PlatformData& data = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData());
platforms/cuda/src/CudaKernelFactory.cpp:    CudaContext& cu = *data.contexts[0];
platforms/cuda/src/CudaKernelFactory.cpp:            return new CudaParallelCalcForcesAndEnergyKernel(name, platform, data);
platforms/cuda/src/CudaKernelFactory.cpp:            return new CudaParallelCalcNonbondedForceKernel(name, platform, data, context.getSystem());
platforms/cuda/src/CudaKernelFactory.cpp:        return new CudaCalcForcesAndEnergyKernel(name, platform, cu);
platforms/cuda/src/CudaKernelFactory.cpp:        return new CudaCalcNonbondedForceKernel(name, platform, cu, context.getSystem());
platforms/cuda/src/CudaKernelFactory.cpp:        return new CudaCalcCustomCVForceKernel(name, platform, cu);
platforms/cuda/src/CudaKernelFactory.cpp:        return new CudaCalcATMForceKernel(name, platform, cu);
platforms/cuda/src/CudaKernelSources.h.in:#ifndef OPENMM_CUDAKERNELSOURCES_H_
platforms/cuda/src/CudaKernelSources.h.in:#define OPENMM_CUDAKERNELSOURCES_H_
platforms/cuda/src/CudaKernelSources.h.in: * This class is a central holding place for the source code of CUDA kernels.
platforms/cuda/src/CudaKernelSources.h.in:class OPENMM_EXPORT_COMMON CudaKernelSources {
platforms/cuda/src/CudaKernelSources.h.in:#endif /*OPENMM_CUDAKERNELSOURCES_H_*/
platforms/cuda/src/CudaIntegrationUtilities.cpp:#include "CudaIntegrationUtilities.h"
platforms/cuda/src/CudaIntegrationUtilities.cpp:#include "CudaContext.h"
platforms/cuda/src/CudaIntegrationUtilities.cpp:    if (result != CUDA_SUCCESS) { \
platforms/cuda/src/CudaIntegrationUtilities.cpp:        m<<prefix<<": "<<dynamic_cast<CudaContext&>(context).getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
platforms/cuda/src/CudaIntegrationUtilities.cpp:CudaIntegrationUtilities::CudaIntegrationUtilities(CudaContext& context, const System& system) : IntegrationUtilities(context, system),
platforms/cuda/src/CudaIntegrationUtilities.cpp:CudaIntegrationUtilities::~CudaIntegrationUtilities() {
platforms/cuda/src/CudaIntegrationUtilities.cpp:CudaArray& CudaIntegrationUtilities::getPosDelta() {
platforms/cuda/src/CudaIntegrationUtilities.cpp:    return dynamic_cast<CudaContext&>(context).unwrap(posDelta);
platforms/cuda/src/CudaIntegrationUtilities.cpp:CudaArray& CudaIntegrationUtilities::getRandom() {
platforms/cuda/src/CudaIntegrationUtilities.cpp:    return dynamic_cast<CudaContext&>(context).unwrap(random);
platforms/cuda/src/CudaIntegrationUtilities.cpp:CudaArray& CudaIntegrationUtilities::getStepSize() {
platforms/cuda/src/CudaIntegrationUtilities.cpp:    return dynamic_cast<CudaContext&>(context).unwrap(stepSize);
platforms/cuda/src/CudaIntegrationUtilities.cpp:void CudaIntegrationUtilities::applyConstraintsImpl(bool constrainVelocities, double tol) {
platforms/cuda/src/CudaIntegrationUtilities.cpp:void CudaIntegrationUtilities::distributeForcesFromVirtualSites() {
platforms/cuda/src/CudaKernel.cpp:#include "CudaKernel.h"
platforms/cuda/src/CudaKernel.cpp:CudaKernel::CudaKernel(CudaContext& context, CUfunction kernel, const string& name) : context(context), kernel(kernel), name(name) {
platforms/cuda/src/CudaKernel.cpp:string CudaKernel::getName() const {
platforms/cuda/src/CudaKernel.cpp:int CudaKernel::getMaxBlockSize() const {
platforms/cuda/src/CudaKernel.cpp:    if (result != CUDA_SUCCESS)
platforms/cuda/src/CudaKernel.cpp:void CudaKernel::execute(int threads, int blockSize) {
platforms/cuda/src/CudaKernel.cpp:void CudaKernel::addArrayArg(ArrayInterface& value) {
platforms/cuda/src/CudaKernel.cpp:void CudaKernel::addPrimitiveArg(const void* value, int size) {
platforms/cuda/src/CudaKernel.cpp:void CudaKernel::addEmptyArg() {
platforms/cuda/src/CudaKernel.cpp:void CudaKernel::setArrayArg(int index, ArrayInterface& value) {
platforms/cuda/src/CudaKernel.cpp:void CudaKernel::setPrimitiveArg(int index, const void* value, int size) {
platforms/cuda/src/CudaProgram.cpp:#include "CudaProgram.h"
platforms/cuda/src/CudaProgram.cpp:#include "CudaKernel.h"
platforms/cuda/src/CudaProgram.cpp:CudaProgram::CudaProgram(CudaContext& context, CUmodule module) : context(context), module(module) {
platforms/cuda/src/CudaProgram.cpp:ComputeKernel CudaProgram::createKernel(const string& name) {
platforms/cuda/src/CudaProgram.cpp:    return shared_ptr<ComputeKernelImpl>(new CudaKernel(context, kernel, name));
platforms/cuda/src/CudaArray.cpp:#include "CudaArray.h"
platforms/cuda/src/CudaArray.cpp:#include "CudaContext.h"
platforms/cuda/src/CudaArray.cpp:CudaArray::CudaArray() : pointer(0), ownsMemory(false) {
platforms/cuda/src/CudaArray.cpp:CudaArray::CudaArray(CudaContext& context, size_t size, int elementSize, const std::string& name) : pointer(0) {
platforms/cuda/src/CudaArray.cpp:CudaArray::~CudaArray() {
platforms/cuda/src/CudaArray.cpp:        if (result != CUDA_SUCCESS) {
platforms/cuda/src/CudaArray.cpp:            str<<"Error deleting array "<<name<<": "<<CudaContext::getErrorString(result)<<" ("<<result<<")";
platforms/cuda/src/CudaArray.cpp:void CudaArray::initialize(ComputeContext& context, size_t size, int elementSize, const std::string& name) {
platforms/cuda/src/CudaArray.cpp:        throw OpenMMException("CudaArray has already been initialized");
platforms/cuda/src/CudaArray.cpp:    this->context = &dynamic_cast<CudaContext&>(context);
platforms/cuda/src/CudaArray.cpp:    if (result != CUDA_SUCCESS) {
platforms/cuda/src/CudaArray.cpp:        str<<"Error creating array "<<name<<": "<<CudaContext::getErrorString(result)<<" ("<<result<<")";
platforms/cuda/src/CudaArray.cpp:void CudaArray::resize(size_t size) {
platforms/cuda/src/CudaArray.cpp:        throw OpenMMException("CudaArray has not been initialized");
platforms/cuda/src/CudaArray.cpp:    if (result != CUDA_SUCCESS) {
platforms/cuda/src/CudaArray.cpp:        str<<"Error deleting array "<<name<<": "<<CudaContext::getErrorString(result)<<" ("<<result<<")";
platforms/cuda/src/CudaArray.cpp:ComputeContext& CudaArray::getContext() {
platforms/cuda/src/CudaArray.cpp:void CudaArray::uploadSubArray(const void* data, int offset, int elements, bool blocking) {
platforms/cuda/src/CudaArray.cpp:        throw OpenMMException("CudaArray has not been initialized");
platforms/cuda/src/CudaArray.cpp:    if (result != CUDA_SUCCESS) {
platforms/cuda/src/CudaArray.cpp:        str<<"Error uploading array "<<name<<": "<<CudaContext::getErrorString(result)<<" ("<<result<<")";
platforms/cuda/src/CudaArray.cpp:void CudaArray::download(void* data, bool blocking) const {
platforms/cuda/src/CudaArray.cpp:        throw OpenMMException("CudaArray has not been initialized");
platforms/cuda/src/CudaArray.cpp:    if (result != CUDA_SUCCESS) {
platforms/cuda/src/CudaArray.cpp:        str<<"Error downloading array "<<name<<": "<<CudaContext::getErrorString(result)<<" ("<<result<<")";
platforms/cuda/src/CudaArray.cpp:void CudaArray::copyTo(ArrayInterface& dest) const {
platforms/cuda/src/CudaArray.cpp:        throw OpenMMException("CudaArray has not been initialized");
platforms/cuda/src/CudaArray.cpp:    CudaArray& cuDest = context->unwrap(dest);
platforms/cuda/src/CudaArray.cpp:    if (result != CUDA_SUCCESS) {
platforms/cuda/src/CudaArray.cpp:        str<<"Error copying array "<<name<<" to "<<dest.getName()<<": "<<CudaContext::getErrorString(result)<<" ("<<result<<")";
platforms/cuda/src/kernels/common.cu: * This file contains CUDA definitions for the macros and functions needed for the
platforms/cuda/src/kernels/nonbonded.cu: * On CUDA devices that support the shuffle intrinsic, on diagonal exclusion tiles use
platforms/cuda/src/kernels/findInteractingBlocks.cu: * To use half precision, we're supposed to include cuda_fp16.h.  Unfortunately,
platforms/cuda/src/kernels/findInteractingBlocks.cu: * [in] maxTiles               - maximum number of tiles to process, used for multi-GPUs
platforms/cuda/src/kernels/findInteractingBlocks.cu: * [in] startBlockIndex        - first block to process, used for multi-GPUs,
platforms/cuda/src/CudaFFT3D.cpp:#include "CudaFFT3D.h"
platforms/cuda/src/CudaFFT3D.cpp:#include "CudaContext.h"
platforms/cuda/src/CudaFFT3D.cpp:#include "CudaKernelSources.h"
platforms/cuda/src/CudaFFT3D.cpp:CudaFFT3D::CudaFFT3D(CudaContext& context, int xsize, int ysize, int zsize, bool realToComplex) :
platforms/cuda/src/CudaFFT3D.cpp:            CUmodule module = context.createModule(CudaKernelSources::vectorOps+CudaKernelSources::fftR2C, defines);
platforms/cuda/src/CudaFFT3D.cpp:void CudaFFT3D::execFFT(CudaArray& in, CudaArray& out, bool forward) {
platforms/cuda/src/CudaFFT3D.cpp:int CudaFFT3D::findLegalDimension(int minimum) {
platforms/cuda/src/CudaFFT3D.cpp:CUfunction CudaFFT3D::createKernel(int xsize, int ysize, int zsize, int& threads, int axis, bool forward, bool inputIsReal) {
platforms/cuda/src/CudaFFT3D.cpp:    CUmodule module = context.createModule(CudaKernelSources::vectorOps+context.replaceStrings(CudaKernelSources::fft, replacements));
platforms/cuda/src/CudaNonbondedUtilities.cpp:#include "CudaNonbondedUtilities.h"
platforms/cuda/src/CudaNonbondedUtilities.cpp:#include "CudaArray.h"
platforms/cuda/src/CudaNonbondedUtilities.cpp:#include "CudaContext.h"
platforms/cuda/src/CudaNonbondedUtilities.cpp:#include "CudaKernelSources.h"
platforms/cuda/src/CudaNonbondedUtilities.cpp:#include "CudaExpressionUtilities.h"
platforms/cuda/src/CudaNonbondedUtilities.cpp:#include "CudaSort.h"
platforms/cuda/src/CudaNonbondedUtilities.cpp:    if (result != CUDA_SUCCESS) { \
platforms/cuda/src/CudaNonbondedUtilities.cpp:class CudaNonbondedUtilities::BlockSortTrait : public CudaSort::SortTrait {
platforms/cuda/src/CudaNonbondedUtilities.cpp:CudaNonbondedUtilities::CudaNonbondedUtilities(CudaContext& context) : context(context), useCutoff(false), usePeriodic(false), useNeighborList(false), anyExclusions(false), usePadding(true),
platforms/cuda/src/CudaNonbondedUtilities.cpp:    setKernelSource(CudaKernelSources::nonbonded);
platforms/cuda/src/CudaNonbondedUtilities.cpp:CudaNonbondedUtilities::~CudaNonbondedUtilities() {
platforms/cuda/src/CudaNonbondedUtilities.cpp:void CudaNonbondedUtilities::addInteraction(bool usesCutoff, bool usesPeriodic, bool usesExclusions, double cutoffDistance, const vector<vector<int> >& exclusionList, const string& kernel, int forceGroup, bool useNeighborList) {
platforms/cuda/src/CudaNonbondedUtilities.cpp:void CudaNonbondedUtilities::addInteraction(bool usesCutoff, bool usesPeriodic, bool usesExclusions, double cutoffDistance, const vector<vector<int> >& exclusionList, const string& kernel, int forceGroup, bool useNeighborList, bool supportsPairList) {
platforms/cuda/src/CudaNonbondedUtilities.cpp:void CudaNonbondedUtilities::addParameter(ComputeParameterInfo parameter) {
platforms/cuda/src/CudaNonbondedUtilities.cpp:void CudaNonbondedUtilities::addParameter(const ParameterInfo& parameter) {
platforms/cuda/src/CudaNonbondedUtilities.cpp:void CudaNonbondedUtilities::addArgument(ComputeParameterInfo parameter) {
platforms/cuda/src/CudaNonbondedUtilities.cpp:void CudaNonbondedUtilities::addArgument(const ParameterInfo& parameter) {
platforms/cuda/src/CudaNonbondedUtilities.cpp:string CudaNonbondedUtilities::addEnergyParameterDerivative(const string& param) {
platforms/cuda/src/CudaNonbondedUtilities.cpp:void CudaNonbondedUtilities::requestExclusions(const vector<vector<int> >& exclusionList) {
platforms/cuda/src/CudaNonbondedUtilities.cpp:void CudaNonbondedUtilities::initialize(const System& system) {
platforms/cuda/src/CudaNonbondedUtilities.cpp:        int x = atom1/CudaContext::TileSize;
platforms/cuda/src/CudaNonbondedUtilities.cpp:            int y = atom2/CudaContext::TileSize;
platforms/cuda/src/CudaNonbondedUtilities.cpp:    exclusions.initialize<tileflags>(context, tilesWithExclusions.size()*CudaContext::TileSize, "exclusions");
platforms/cuda/src/CudaNonbondedUtilities.cpp:        int x = atom1/CudaContext::TileSize;
platforms/cuda/src/CudaNonbondedUtilities.cpp:        int offset1 = atom1-x*CudaContext::TileSize;
platforms/cuda/src/CudaNonbondedUtilities.cpp:            int y = atom2/CudaContext::TileSize;
platforms/cuda/src/CudaNonbondedUtilities.cpp:            int offset2 = atom2-y*CudaContext::TileSize;
platforms/cuda/src/CudaNonbondedUtilities.cpp:                int index = exclusionTileMap[make_pair(x, y)]*CudaContext::TileSize;
platforms/cuda/src/CudaNonbondedUtilities.cpp:                int index = exclusionTileMap[make_pair(y, x)]*CudaContext::TileSize;
platforms/cuda/src/CudaNonbondedUtilities.cpp:        interactingAtoms.initialize<int>(context, CudaContext::TileSize*maxTiles, "interactingAtoms");
platforms/cuda/src/CudaNonbondedUtilities.cpp:        blockSorter = new CudaSort(context, new BlockSortTrait(), numAtomBlocks, false);
platforms/cuda/src/CudaNonbondedUtilities.cpp:double CudaNonbondedUtilities::getMaxCutoffDistance() {
platforms/cuda/src/CudaNonbondedUtilities.cpp:double CudaNonbondedUtilities::padCutoff(double cutoff) {
platforms/cuda/src/CudaNonbondedUtilities.cpp:void CudaNonbondedUtilities::prepareInteractions(int forceGroups) {
platforms/cuda/src/CudaNonbondedUtilities.cpp:void CudaNonbondedUtilities::computeInteractions(int forceGroups, bool includeForces, bool includeEnergy) {
platforms/cuda/src/CudaNonbondedUtilities.cpp:bool CudaNonbondedUtilities::updateNeighborListSize() {
platforms/cuda/src/CudaNonbondedUtilities.cpp:        interactingAtoms.resize(CudaContext::TileSize*(size_t) maxTiles);
platforms/cuda/src/CudaNonbondedUtilities.cpp:void CudaNonbondedUtilities::setUsePadding(bool padding) {
platforms/cuda/src/CudaNonbondedUtilities.cpp:void CudaNonbondedUtilities::setAtomBlockRange(double startFraction, double endFraction) {
platforms/cuda/src/CudaNonbondedUtilities.cpp:void CudaNonbondedUtilities::createKernelsForGroups(int groups) {
platforms/cuda/src/CudaNonbondedUtilities.cpp:        defines["TILE_SIZE"] = context.intToString(CudaContext::TileSize);
platforms/cuda/src/CudaNonbondedUtilities.cpp:        CUmodule interactingBlocksProgram = context.createModule(CudaKernelSources::vectorOps+CudaKernelSources::findInteractingBlocks, defines);
platforms/cuda/src/CudaNonbondedUtilities.cpp:CUfunction CudaNonbondedUtilities::createInteractionKernel(const string& source, vector<ParameterInfo>& params, vector<ParameterInfo>& arguments, bool useExclusions, bool isSymmetric, int groups, bool includeForces, bool includeEnergy) {
platforms/cuda/src/CudaNonbondedUtilities.cpp:    defines["TILE_SIZE"] = context.intToString(CudaContext::TileSize);
platforms/cuda/src/CudaNonbondedUtilities.cpp:    CUmodule program = context.createModule(CudaKernelSources::vectorOps+context.replaceStrings(kernelSource, replacements), defines);
platforms/cuda/src/CudaNonbondedUtilities.cpp:void CudaNonbondedUtilities::setKernelSource(const string& source) {
platforms/cuda/src/CudaParameterSet.cpp:#include "CudaParameterSet.h"
platforms/cuda/src/CudaParameterSet.cpp:CudaParameterSet::CudaParameterSet(CudaContext& context, int numParameters, int numObjects, const string& name, bool bufferPerParameter, bool useDoublePrecision) :
platforms/cuda/src/CudaParameterSet.cpp:        buffers.push_back(CudaNonbondedUtilities::ParameterInfo(info.getName(), info.getComponentType(), info.getNumComponents(), info.getSize(), context.unwrap(info.getArray()).getDevicePointer()));
platforms/cuda/src/CudaEvent.cpp:#include "CudaEvent.h"
platforms/cuda/src/CudaEvent.cpp:CudaEvent::CudaEvent(CudaContext& context) : context(context), eventCreated(false) {
platforms/cuda/src/CudaEvent.cpp:    if (result != CUDA_SUCCESS)
platforms/cuda/src/CudaEvent.cpp:        throw OpenMMException("Error creating CUDA event:"+CudaContext::getErrorString(result));
platforms/cuda/src/CudaEvent.cpp:CudaEvent::~CudaEvent() {
platforms/cuda/src/CudaEvent.cpp:void CudaEvent::enqueue() {
platforms/cuda/src/CudaEvent.cpp:void CudaEvent::wait() {
platforms/cuda/src/CudaKernelSources.cpp.in:#include "CudaKernelSources.h"
platforms/cuda/src/CudaSort.cpp:#include "CudaSort.h"
platforms/cuda/src/CudaSort.cpp:#include "CudaKernelSources.h"
platforms/cuda/src/CudaSort.cpp:CudaSort::CudaSort(CudaContext& context, SortTrait* trait, unsigned int length, bool uniform) :
platforms/cuda/src/CudaSort.cpp:    CUmodule module = context.createModule(context.replaceStrings(CudaKernelSources::sort, replacements));
platforms/cuda/src/CudaSort.cpp:    int maxShortList = min(3000, max(maxLocalBuffer, CudaContext::ThreadBlockSize*context.getNumThreadBlocks()));
platforms/cuda/src/CudaSort.cpp:CudaSort::~CudaSort() {
platforms/cuda/src/CudaSort.cpp:void CudaSort::sort(CudaArray& data) {
platforms/cuda/src/CudaSort.cpp:        throw OpenMMException("CudaSort called with different data size");
platforms/cuda/src/CudaSort.cpp:        if (dataLength <= CudaContext::ThreadBlockSize*context.getNumThreadBlocks()) {
platforms/cuda/src/CudaKernels.cpp:#include "CudaKernels.h"
platforms/cuda/src/CudaKernels.cpp:#include "CudaForceInfo.h"
platforms/cuda/src/CudaKernels.cpp:#include "CudaBondedUtilities.h"
platforms/cuda/src/CudaKernels.cpp:#include "CudaExpressionUtilities.h"
platforms/cuda/src/CudaKernels.cpp:#include "CudaIntegrationUtilities.h"
platforms/cuda/src/CudaKernels.cpp:#include "CudaNonbondedUtilities.h"
platforms/cuda/src/CudaKernels.cpp:#include "CudaKernelSources.h"
platforms/cuda/src/CudaKernels.cpp:    if (result != CUDA_SUCCESS) { \
platforms/cuda/src/CudaKernels.cpp:        m<<prefix<<": "<<CudaContext::getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
platforms/cuda/src/CudaKernels.cpp:void CudaCalcForcesAndEnergyKernel::initialize(const System& system) {
platforms/cuda/src/CudaKernels.cpp:void CudaCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
platforms/cuda/src/CudaKernels.cpp:    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
platforms/cuda/src/CudaKernels.cpp:double CudaCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForces, bool includeEnergy, int groups, bool& valid) {
platforms/cuda/src/CudaKernels.cpp:class CudaCalcNonbondedForceKernel::ForceInfo : public CudaForceInfo {
platforms/cuda/src/CudaKernels.cpp:class CudaCalcNonbondedForceKernel::PmeIO : public CalcPmeReciprocalForceKernel::IO {
platforms/cuda/src/CudaKernels.cpp:    PmeIO(CudaContext& cu, CUfunction addForcesKernel) : cu(cu), addForcesKernel(addForcesKernel) {
platforms/cuda/src/CudaKernels.cpp:    CudaContext& cu;
platforms/cuda/src/CudaKernels.cpp:    CudaArray forceTemp;
platforms/cuda/src/CudaKernels.cpp:class CudaCalcNonbondedForceKernel::PmePreComputation : public CudaContext::ForcePreComputation {
platforms/cuda/src/CudaKernels.cpp:    PmePreComputation(CudaContext& cu, Kernel& pme, CalcPmeReciprocalForceKernel::IO& io) : cu(cu), pme(pme), io(io) {
platforms/cuda/src/CudaKernels.cpp:    CudaContext& cu;
platforms/cuda/src/CudaKernels.cpp:class CudaCalcNonbondedForceKernel::PmePostComputation : public CudaContext::ForcePostComputation {
platforms/cuda/src/CudaKernels.cpp:class CudaCalcNonbondedForceKernel::SyncStreamPreComputation : public CudaContext::ForcePreComputation {
platforms/cuda/src/CudaKernels.cpp:    SyncStreamPreComputation(CudaContext& cu, CUstream stream, CUevent event, int forceGroup) : cu(cu), stream(stream), event(event), forceGroup(forceGroup) {
platforms/cuda/src/CudaKernels.cpp:    CudaContext& cu;
platforms/cuda/src/CudaKernels.cpp:class CudaCalcNonbondedForceKernel::SyncStreamPostComputation : public CudaContext::ForcePostComputation {
platforms/cuda/src/CudaKernels.cpp:    SyncStreamPostComputation(CudaContext& cu, CUevent event, CUfunction addEnergyKernel, CudaArray& pmeEnergyBuffer, int forceGroup) : cu(cu), event(event),
platforms/cuda/src/CudaKernels.cpp:    CudaContext& cu;
platforms/cuda/src/CudaKernels.cpp:    CudaArray& pmeEnergyBuffer;
platforms/cuda/src/CudaKernels.cpp:CudaCalcNonbondedForceKernel::~CudaCalcNonbondedForceKernel() {
platforms/cuda/src/CudaKernels.cpp:        if (useCudaFFT) {
platforms/cuda/src/CudaKernels.cpp:void CudaCalcNonbondedForceKernel::initialize(const System& system, const NonbondedForce& force) {
platforms/cuda/src/CudaKernels.cpp:            CUmodule module = cu.createModule(CudaKernelSources::vectorOps+CommonKernelSources::ewald, replacements);
platforms/cuda/src/CudaKernels.cpp:        gridSizeX = CudaFFT3D::findLegalDimension(gridSizeX);
platforms/cuda/src/CudaKernels.cpp:        gridSizeY = CudaFFT3D::findLegalDimension(gridSizeY);
platforms/cuda/src/CudaKernels.cpp:        gridSizeZ = CudaFFT3D::findLegalDimension(gridSizeZ);
platforms/cuda/src/CudaKernels.cpp:            dispersionGridSizeX = CudaFFT3D::findLegalDimension(dispersionGridSizeX);
platforms/cuda/src/CudaKernels.cpp:            dispersionGridSizeY = CudaFFT3D::findLegalDimension(dispersionGridSizeY);
platforms/cuda/src/CudaKernels.cpp:            dispersionGridSizeZ = CudaFFT3D::findLegalDimension(dispersionGridSizeZ);
platforms/cuda/src/CudaKernels.cpp:            CUmodule module = cu.createModule(CudaKernelSources::vectorOps+cu.replaceStrings(CommonKernelSources::pme, replacements), pmeDefines);
platforms/cuda/src/CudaKernels.cpp:                    module = cu.createModule(CudaKernelSources::vectorOps+CommonKernelSources::pme, pmeDefines);
platforms/cuda/src/CudaKernels.cpp:                pmeEnergyBuffer.initialize(cu, cu.getNumThreadBlocks()*CudaContext::ThreadBlockSize, energyElementSize, "pmeEnergyBuffer");
platforms/cuda/src/CudaKernels.cpp:                sort = new CudaSort(cu, new SortTrait(), cu.getNumAtoms());
platforms/cuda/src/CudaKernels.cpp:                useCudaFFT = (cufftVersion >= 7050); // There was a critical bug in version 7.0
platforms/cuda/src/CudaKernels.cpp:                if (useCudaFFT) {
platforms/cuda/src/CudaKernels.cpp:                    fft = new CudaFFT3D(cu, gridSizeX, gridSizeY, gridSizeZ, true);
platforms/cuda/src/CudaKernels.cpp:                        dispersionFft = new CudaFFT3D(cu, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, true);
platforms/cuda/src/CudaKernels.cpp:                    if (useCudaFFT) {
platforms/cuda/src/CudaKernels.cpp:                    CudaArray *xmoduli, *ymoduli, *zmoduli;
platforms/cuda/src/CudaKernels.cpp:        cu.getNonbondedUtilities().addParameter(CudaNonbondedUtilities::ParameterInfo(prefix+"charge", "real", 1, charges.getElementSize(), charges.getDevicePointer()));
platforms/cuda/src/CudaKernels.cpp:        cu.getNonbondedUtilities().addParameter(CudaNonbondedUtilities::ParameterInfo(prefix+"sigmaEpsilon", "float", 2, sizeof(float2), sigmaEpsilon.getDevicePointer()));
platforms/cuda/src/CudaKernels.cpp:double CudaCalcNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
platforms/cuda/src/CudaKernels.cpp:        CudaArray& pmeSpreadDstGrid = useFixedPointChargeSpreading ? pmeGrid2 : pmeGrid1;
platforms/cuda/src/CudaKernels.cpp:            if (useCudaFFT) {
platforms/cuda/src/CudaKernels.cpp:            if (useCudaFFT) {
platforms/cuda/src/CudaKernels.cpp:            if (useCudaFFT) {
platforms/cuda/src/CudaKernels.cpp:            if (useCudaFFT) {
platforms/cuda/src/CudaKernels.cpp:void CudaCalcNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const NonbondedForce& force, int firstParticle, int lastParticle, int firstException, int lastException) {
platforms/cuda/src/CudaKernels.cpp:void CudaCalcNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
platforms/cuda/src/CudaKernels.cpp:void CudaCalcNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
platforms/cuda/src/CudaContext.cpp:#include "CudaContext.h"
platforms/cuda/src/CudaContext.cpp:#include "CudaArray.h"
platforms/cuda/src/CudaContext.cpp:#include "CudaBondedUtilities.h"
platforms/cuda/src/CudaContext.cpp:#include "CudaEvent.h"
platforms/cuda/src/CudaContext.cpp:#include "CudaIntegrationUtilities.h"
platforms/cuda/src/CudaContext.cpp:#include "CudaKernels.h"
platforms/cuda/src/CudaContext.cpp:#include "CudaKernelSources.h"
platforms/cuda/src/CudaContext.cpp:#include "CudaNonbondedUtilities.h"
platforms/cuda/src/CudaContext.cpp:#include "CudaProgram.h"
platforms/cuda/src/CudaContext.cpp:#include "CudaExpressionUtilities.h"
platforms/cuda/src/CudaContext.cpp:#include <cudaProfiler.h>
platforms/cuda/src/CudaContext.cpp:    if (result != CUDA_SUCCESS) { \
platforms/cuda/src/CudaContext.cpp:const int CudaContext::ThreadBlockSize = 64;
platforms/cuda/src/CudaContext.cpp:const int CudaContext::TileSize = sizeof(tileflags)*8;
platforms/cuda/src/CudaContext.cpp:bool CudaContext::hasInitializedCuda = false;
platforms/cuda/src/CudaContext.cpp:CudaContext::CudaContext(const System& system, int deviceIndex, bool useBlockingSync, const string& precision, const string& tempDir, CudaPlatform::PlatformData& platformData,
platforms/cuda/src/CudaContext.cpp:        CudaContext* originalContext) : ComputeContext(system), currentStream(0), platformData(platformData), contextIsValid(false), hasAssignedPosqCharges(false),
platforms/cuda/src/CudaContext.cpp:    int cudaDriverVersion;
platforms/cuda/src/CudaContext.cpp:    cuDriverGetVersion(&cudaDriverVersion);
platforms/cuda/src/CudaContext.cpp:    if (!hasInitializedCuda) {
platforms/cuda/src/CudaContext.cpp:        CHECK_RESULT2(cuInit(0), "Error initializing CUDA");
platforms/cuda/src/CudaContext.cpp:        hasInitializedCuda = true;
platforms/cuda/src/CudaContext.cpp:            if (cuCtxCreate(&context, flags, device) == CUDA_SUCCESS) {
platforms/cuda/src/CudaContext.cpp:                throw OpenMMException("The requested CUDA device could not be loaded");
platforms/cuda/src/CudaContext.cpp:                throw OpenMMException("No compatible CUDA device is available");
platforms/cuda/src/CudaContext.cpp:    if (cudaDriverVersion < 7000) {
platforms/cuda/src/CudaContext.cpp:        // This is a workaround to support GTX 980 with CUDA 6.5.  It reports
platforms/cuda/src/CudaContext.cpp:    if (cudaDriverVersion < 8000) {
platforms/cuda/src/CudaContext.cpp:        // This is a workaround to support Pascal with CUDA 7.5.  It reports
platforms/cuda/src/CudaContext.cpp:    gpuArchitecture = 10*major+minor;
platforms/cuda/src/CudaContext.cpp:    if (cudaDriverVersion >= 9000) {
platforms/cuda/src/CudaContext.cpp:    CUmodule utilities = createModule(CudaKernelSources::vectorOps+CudaKernelSources::utilities);
platforms/cuda/src/CudaContext.cpp:    bonded = new CudaBondedUtilities(*this);
platforms/cuda/src/CudaContext.cpp:    nonbonded = new CudaNonbondedUtilities(*this);
platforms/cuda/src/CudaContext.cpp:    integration = new CudaIntegrationUtilities(*this, system);
platforms/cuda/src/CudaContext.cpp:    expression = new CudaExpressionUtilities(*this);
platforms/cuda/src/CudaContext.cpp:CudaContext::~CudaContext() {
platforms/cuda/src/CudaContext.cpp:void CudaContext::initialize() {
platforms/cuda/src/CudaContext.cpp:    CHECK_RESULT2(cuDeviceGetAttribute(&multiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device), "Error checking GPU properties");
platforms/cuda/src/CudaContext.cpp:void CudaContext::initializeContexts() {
platforms/cuda/src/CudaContext.cpp:void CudaContext::setAsCurrent() {
platforms/cuda/src/CudaContext.cpp:void CudaContext::pushAsCurrent() {
platforms/cuda/src/CudaContext.cpp:void CudaContext::popAsCurrent() {
platforms/cuda/src/CudaContext.cpp:CUmodule CudaContext::createModule(const string source, const char* optimizationFlags) {
platforms/cuda/src/CudaContext.cpp:CUmodule CudaContext::createModule(const string source, const map<string, string>& defines, const char* optimizationFlags) {
platforms/cuda/src/CudaContext.cpp:    src << CudaKernelSources::common << endl;
platforms/cuda/src/CudaContext.cpp:#if CUDA_VERSION < 11020
platforms/cuda/src/CudaContext.cpp:    // CUDA versions before 11.2 can't query the compiler to see what it supports.
platforms/cuda/src/CudaContext.cpp:    string compileArchitecture = intToString(min(gpuArchitecture, maxCompilerArchitecture));
platforms/cuda/src/CudaContext.cpp:    if (cuModuleLoad(&module, cacheFile.str().c_str()) == CUDA_SUCCESS)
platforms/cuda/src/CudaContext.cpp:            CHECK_RESULT2(cuModuleLoadDataEx(&module, &ptx[0], 0, NULL, NULL), "Error loading CUDA module");
platforms/cuda/src/CudaContext.cpp:        if (result != CUDA_SUCCESS) {
platforms/cuda/src/CudaContext.cpp:            m<<"Error loading CUDA module: "<<getErrorString(result)<<" ("<<result<<")";
platforms/cuda/src/CudaContext.cpp:CUfunction CudaContext::getKernel(CUmodule& module, const string& name) {
platforms/cuda/src/CudaContext.cpp:    if (result != CUDA_SUCCESS) {
platforms/cuda/src/CudaContext.cpp:vector<ComputeContext*> CudaContext::getAllContexts() {
platforms/cuda/src/CudaContext.cpp:    for (CudaContext* c : platformData.contexts)
platforms/cuda/src/CudaContext.cpp:double& CudaContext::getEnergyWorkspace() {
platforms/cuda/src/CudaContext.cpp:CUstream CudaContext::getCurrentStream() {
platforms/cuda/src/CudaContext.cpp:void CudaContext::setCurrentStream(CUstream stream) {
platforms/cuda/src/CudaContext.cpp:void CudaContext::restoreDefaultStream() {
platforms/cuda/src/CudaContext.cpp:CudaArray* CudaContext::createArray() {
platforms/cuda/src/CudaContext.cpp:    return new CudaArray();
platforms/cuda/src/CudaContext.cpp:ComputeEvent CudaContext::createEvent() {
platforms/cuda/src/CudaContext.cpp:    return shared_ptr<ComputeEventImpl>(new CudaEvent(*this));
platforms/cuda/src/CudaContext.cpp:ComputeProgram CudaContext::compileProgram(const std::string source, const std::map<std::string, std::string>& defines) {
platforms/cuda/src/CudaContext.cpp:    CUmodule module = createModule(CudaKernelSources::vectorOps+source, defines);
platforms/cuda/src/CudaContext.cpp:    return shared_ptr<ComputeProgramImpl>(new CudaProgram(*this, module));
platforms/cuda/src/CudaContext.cpp:CudaArray& CudaContext::unwrap(ArrayInterface& array) const {
platforms/cuda/src/CudaContext.cpp:    CudaArray* cuarray;
platforms/cuda/src/CudaContext.cpp:        cuarray = dynamic_cast<CudaArray*>(&wrapper->getArray());
platforms/cuda/src/CudaContext.cpp:        cuarray = dynamic_cast<CudaArray*>(&array);
platforms/cuda/src/CudaContext.cpp:        throw OpenMMException("Array argument is not an CudaArray");
platforms/cuda/src/CudaContext.cpp:std::string CudaContext::getErrorString(CUresult result) {
platforms/cuda/src/CudaContext.cpp:    if (cuGetErrorName(result, &message) == CUDA_SUCCESS)
platforms/cuda/src/CudaContext.cpp:    return "CUDA error";
platforms/cuda/src/CudaContext.cpp:void CudaContext::executeKernel(CUfunction kernel, void** arguments, int threads, int blockSize, unsigned int sharedSize) {
platforms/cuda/src/CudaContext.cpp:    if (result != CUDA_SUCCESS) {
platforms/cuda/src/CudaContext.cpp:int CudaContext::computeThreadBlockSize(double memory) const {
platforms/cuda/src/CudaContext.cpp:void CudaContext::clearBuffer(ArrayInterface& array) {
platforms/cuda/src/CudaContext.cpp:void CudaContext::clearBuffer(CUdeviceptr memory, int size) {
platforms/cuda/src/CudaContext.cpp:void CudaContext::addAutoclearBuffer(ArrayInterface& array) {
platforms/cuda/src/CudaContext.cpp:void CudaContext::addAutoclearBuffer(CUdeviceptr memory, int size) {
platforms/cuda/src/CudaContext.cpp:void CudaContext::clearAutoclearBuffers() {
platforms/cuda/src/CudaContext.cpp:double CudaContext::reduceEnergy() {
platforms/cuda/src/CudaContext.cpp:void CudaContext::setCharges(const vector<double>& charges) {
platforms/cuda/src/CudaContext.cpp:bool CudaContext::requestPosqCharges() {
platforms/cuda/src/CudaContext.cpp:void CudaContext::addEnergyParameterDerivative(const string& param) {
platforms/cuda/src/CudaContext.cpp:void CudaContext::flushQueue() {
platforms/cuda/src/CudaContext.cpp:vector<int> CudaContext::getDevicePrecedence() {
platforms/cuda/src/CudaContext.cpp:unsigned int CudaContext::getEventFlags() {
platforms/cuda/src/CudaParallelKernels.cpp:#include "CudaParallelKernels.h"
platforms/cuda/src/CudaParallelKernels.cpp:#include "CudaKernelSources.h"
platforms/cuda/src/CudaParallelKernels.cpp:if (result != CUDA_SUCCESS) { \
platforms/cuda/src/CudaParallelKernels.cpp:class CudaParallelCalcForcesAndEnergyKernel::BeginComputationTask : public CudaContext::WorkTask {
platforms/cuda/src/CudaParallelKernels.cpp:    BeginComputationTask(ContextImpl& context, CudaContext& cu, CudaCalcForcesAndEnergyKernel& kernel,
platforms/cuda/src/CudaParallelKernels.cpp:    CudaContext& cu;
platforms/cuda/src/CudaParallelKernels.cpp:    CudaCalcForcesAndEnergyKernel& kernel;
platforms/cuda/src/CudaParallelKernels.cpp:class CudaParallelCalcForcesAndEnergyKernel::FinishComputationTask : public CudaContext::WorkTask {
platforms/cuda/src/CudaParallelKernels.cpp:    FinishComputationTask(ContextImpl& context, CudaContext& cu, CudaCalcForcesAndEnergyKernel& kernel,
platforms/cuda/src/CudaParallelKernels.cpp:            bool includeForce, bool includeEnergy, int groups, double& energy, double& completionTime, long long* pinnedMemory, CudaArray& contextForces,
platforms/cuda/src/CudaParallelKernels.cpp:            CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context");
platforms/cuda/src/CudaParallelKernels.cpp:                    CudaContext& context0 = *cu.getPlatformData().contexts[0];
platforms/cuda/src/CudaParallelKernels.cpp:    CudaContext& cu;
platforms/cuda/src/CudaParallelKernels.cpp:    CudaCalcForcesAndEnergyKernel& kernel;
platforms/cuda/src/CudaParallelKernels.cpp:    CudaArray& contextForces;
platforms/cuda/src/CudaParallelKernels.cpp:CudaParallelCalcForcesAndEnergyKernel::CudaParallelCalcForcesAndEnergyKernel(string name, const Platform& platform, CudaPlatform::PlatformData& data) :
platforms/cuda/src/CudaParallelKernels.cpp:        kernels.push_back(Kernel(new CudaCalcForcesAndEnergyKernel(name, platform, *data.contexts[i])));
platforms/cuda/src/CudaParallelKernels.cpp:CudaParallelCalcForcesAndEnergyKernel::~CudaParallelCalcForcesAndEnergyKernel() {
platforms/cuda/src/CudaParallelKernels.cpp:void CudaParallelCalcForcesAndEnergyKernel::initialize(const System& system) {
platforms/cuda/src/CudaParallelKernels.cpp:    CudaContext& cu = *data.contexts[0];
platforms/cuda/src/CudaParallelKernels.cpp:    CUmodule module = cu.createModule(CudaKernelSources::parallel);
platforms/cuda/src/CudaParallelKernels.cpp:        CudaContext& cuLocal = *data.contexts[i];
platforms/cuda/src/CudaParallelKernels.cpp:void CudaParallelCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups) {
platforms/cuda/src/CudaParallelKernels.cpp:    CudaContext& cu = *data.contexts[0];
platforms/cuda/src/CudaParallelKernels.cpp:        CudaContext& cu = *data.contexts[i];
platforms/cuda/src/CudaParallelKernels.cpp:double CudaParallelCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups, bool& valid) {
platforms/cuda/src/CudaParallelKernels.cpp:        CudaContext& cu = *data.contexts[i];
platforms/cuda/src/CudaParallelKernels.cpp:    CudaContext& cu = *data.contexts[0];
platforms/cuda/src/CudaParallelKernels.cpp:class CudaParallelCalcNonbondedForceKernel::Task : public CudaContext::WorkTask {
platforms/cuda/src/CudaParallelKernels.cpp:    Task(ContextImpl& context, CudaCalcNonbondedForceKernel& kernel, bool includeForce,
platforms/cuda/src/CudaParallelKernels.cpp:    CudaCalcNonbondedForceKernel& kernel;
platforms/cuda/src/CudaParallelKernels.cpp:CudaParallelCalcNonbondedForceKernel::CudaParallelCalcNonbondedForceKernel(std::string name, const Platform& platform, CudaPlatform::PlatformData& data, const System& system) :
platforms/cuda/src/CudaParallelKernels.cpp:        kernels.push_back(Kernel(new CudaCalcNonbondedForceKernel(name, platform, *data.contexts[i], system)));
platforms/cuda/src/CudaParallelKernels.cpp:void CudaParallelCalcNonbondedForceKernel::initialize(const System& system, const NonbondedForce& force) {
platforms/cuda/src/CudaParallelKernels.cpp:double CudaParallelCalcNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
platforms/cuda/src/CudaParallelKernels.cpp:        CudaContext& cu = *data.contexts[i];
platforms/cuda/src/CudaParallelKernels.cpp:void CudaParallelCalcNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const NonbondedForce& force, int firstParticle, int lastParticle, int firstException, int lastException) {
platforms/cuda/src/CudaParallelKernels.cpp:void CudaParallelCalcNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
platforms/cuda/src/CudaParallelKernels.cpp:    dynamic_cast<const CudaCalcNonbondedForceKernel&>(kernels[0].getImpl()).getPMEParameters(alpha, nx, ny, nz);
platforms/cuda/src/CudaParallelKernels.cpp:void CudaParallelCalcNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
platforms/cuda/src/CudaParallelKernels.cpp:    dynamic_cast<const CudaCalcNonbondedForceKernel&>(kernels[0].getImpl()).getLJPMEParameters(alpha, nx, ny, nz);
platforms/cuda/src/CudaPlatform.cpp:#include "CudaContext.h"
platforms/cuda/src/CudaPlatform.cpp:#include "CudaExpressionUtilities.h"
platforms/cuda/src/CudaPlatform.cpp:#include "CudaPlatform.h"
platforms/cuda/src/CudaPlatform.cpp:#include "CudaKernelFactory.h"
platforms/cuda/src/CudaPlatform.cpp:#include "CudaKernels.h"
platforms/cuda/src/CudaPlatform.cpp:    if (result != CUDA_SUCCESS) { \
platforms/cuda/src/CudaPlatform.cpp:        m<<prefix<<": "<<CudaContext::getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
platforms/cuda/src/CudaPlatform.cpp:extern "C" void registerCudaPlatform() {
platforms/cuda/src/CudaPlatform.cpp:    Platform::registerPlatform(new CudaPlatform());
platforms/cuda/src/CudaPlatform.cpp:    Platform::registerPlatform(new CudaPlatform());
platforms/cuda/src/CudaPlatform.cpp:CudaPlatform::CudaPlatform() {
platforms/cuda/src/CudaPlatform.cpp:    deprecatedPropertyReplacements["CudaDeviceIndex"] = CudaDeviceIndex();
platforms/cuda/src/CudaPlatform.cpp:    deprecatedPropertyReplacements["CudaDeviceName"] = CudaDeviceName();
platforms/cuda/src/CudaPlatform.cpp:    deprecatedPropertyReplacements["CudaUseBlockingSync"] = CudaUseBlockingSync();
platforms/cuda/src/CudaPlatform.cpp:    deprecatedPropertyReplacements["CudaPrecision"] = CudaPrecision();
platforms/cuda/src/CudaPlatform.cpp:    deprecatedPropertyReplacements["CudaUseCpuPme"] = CudaUseCpuPme();
platforms/cuda/src/CudaPlatform.cpp:    deprecatedPropertyReplacements["CudaTempDirectory"] = CudaTempDirectory();
platforms/cuda/src/CudaPlatform.cpp:    deprecatedPropertyReplacements["CudaDisablePmeStream"] = CudaDisablePmeStream();
platforms/cuda/src/CudaPlatform.cpp:    deprecatedPropertyReplacements["CudaDeterministicForces"] = CudaDeterministicForces();
platforms/cuda/src/CudaPlatform.cpp:    CudaKernelFactory* factory = new CudaKernelFactory();
platforms/cuda/src/CudaPlatform.cpp:    platformProperties.push_back(CudaDeviceIndex());
platforms/cuda/src/CudaPlatform.cpp:    platformProperties.push_back(CudaDeviceName());
platforms/cuda/src/CudaPlatform.cpp:    platformProperties.push_back(CudaUseBlockingSync());
platforms/cuda/src/CudaPlatform.cpp:    platformProperties.push_back(CudaPrecision());
platforms/cuda/src/CudaPlatform.cpp:    platformProperties.push_back(CudaUseCpuPme());
platforms/cuda/src/CudaPlatform.cpp:    platformProperties.push_back(CudaCompiler());
platforms/cuda/src/CudaPlatform.cpp:    platformProperties.push_back(CudaTempDirectory());
platforms/cuda/src/CudaPlatform.cpp:    platformProperties.push_back(CudaHostCompiler());
platforms/cuda/src/CudaPlatform.cpp:    platformProperties.push_back(CudaDisablePmeStream());
platforms/cuda/src/CudaPlatform.cpp:    platformProperties.push_back(CudaDeterministicForces());
platforms/cuda/src/CudaPlatform.cpp:    setPropertyDefaultValue(CudaDeviceIndex(), "");
platforms/cuda/src/CudaPlatform.cpp:    setPropertyDefaultValue(CudaDeviceName(), "");
platforms/cuda/src/CudaPlatform.cpp:    setPropertyDefaultValue(CudaUseBlockingSync(), "false");
platforms/cuda/src/CudaPlatform.cpp:    setPropertyDefaultValue(CudaPrecision(), "single");
platforms/cuda/src/CudaPlatform.cpp:    setPropertyDefaultValue(CudaUseCpuPme(), "false");
platforms/cuda/src/CudaPlatform.cpp:    setPropertyDefaultValue(CudaDisablePmeStream(), "false");
platforms/cuda/src/CudaPlatform.cpp:    setPropertyDefaultValue(CudaDeterministicForces(), "false");
platforms/cuda/src/CudaPlatform.cpp:    setPropertyDefaultValue(CudaCompiler(), "");
platforms/cuda/src/CudaPlatform.cpp:    setPropertyDefaultValue(CudaHostCompiler(), "");
platforms/cuda/src/CudaPlatform.cpp:    setPropertyDefaultValue(CudaTempDirectory(), string(getenv("TEMP")));
platforms/cuda/src/CudaPlatform.cpp:    setPropertyDefaultValue(CudaTempDirectory(), tmp);
platforms/cuda/src/CudaPlatform.cpp:double CudaPlatform::getSpeed() const {
platforms/cuda/src/CudaPlatform.cpp:bool CudaPlatform::supportsDoublePrecision() const {
platforms/cuda/src/CudaPlatform.cpp:const string& CudaPlatform::getPropertyValue(const Context& context, const string& property) const {
platforms/cuda/src/CudaPlatform.cpp:void CudaPlatform::setPropertyValue(Context& context, const string& property, const string& value) const {
platforms/cuda/src/CudaPlatform.cpp:void CudaPlatform::contextCreated(ContextImpl& context, const map<string, string>& properties) const {
platforms/cuda/src/CudaPlatform.cpp:    const string& devicePropValue = (properties.find(CudaDeviceIndex()) == properties.end() ?
platforms/cuda/src/CudaPlatform.cpp:            getPropertyDefaultValue(CudaDeviceIndex()) : properties.find(CudaDeviceIndex())->second);
platforms/cuda/src/CudaPlatform.cpp:    string blockingPropValue = (properties.find(CudaUseBlockingSync()) == properties.end() ?
platforms/cuda/src/CudaPlatform.cpp:            getPropertyDefaultValue(CudaUseBlockingSync()) : properties.find(CudaUseBlockingSync())->second);
platforms/cuda/src/CudaPlatform.cpp:    string precisionPropValue = (properties.find(CudaPrecision()) == properties.end() ?
platforms/cuda/src/CudaPlatform.cpp:            getPropertyDefaultValue(CudaPrecision()) : properties.find(CudaPrecision())->second);
platforms/cuda/src/CudaPlatform.cpp:    string cpuPmePropValue = (properties.find(CudaUseCpuPme()) == properties.end() ?
platforms/cuda/src/CudaPlatform.cpp:            getPropertyDefaultValue(CudaUseCpuPme()) : properties.find(CudaUseCpuPme())->second);
platforms/cuda/src/CudaPlatform.cpp:    const string& tempPropValue = (properties.find(CudaTempDirectory()) == properties.end() ?
platforms/cuda/src/CudaPlatform.cpp:            getPropertyDefaultValue(CudaTempDirectory()) : properties.find(CudaTempDirectory())->second);
platforms/cuda/src/CudaPlatform.cpp:    string pmeStreamPropValue = (properties.find(CudaDisablePmeStream()) == properties.end() ?
platforms/cuda/src/CudaPlatform.cpp:            getPropertyDefaultValue(CudaDisablePmeStream()) : properties.find(CudaDisablePmeStream())->second);
platforms/cuda/src/CudaPlatform.cpp:    string deterministicForcesValue = (properties.find(CudaDeterministicForces()) == properties.end() ?
platforms/cuda/src/CudaPlatform.cpp:            getPropertyDefaultValue(CudaDeterministicForces()) : properties.find(CudaDeterministicForces())->second);
platforms/cuda/src/CudaPlatform.cpp:void CudaPlatform::linkedContextCreated(ContextImpl& context, ContextImpl& originalContext) const {
platforms/cuda/src/CudaPlatform.cpp:    string devicePropValue = platform.getPropertyValue(originalContext.getOwner(), CudaDeviceIndex());
platforms/cuda/src/CudaPlatform.cpp:    string blockingPropValue = platform.getPropertyValue(originalContext.getOwner(), CudaUseBlockingSync());
platforms/cuda/src/CudaPlatform.cpp:    string precisionPropValue = platform.getPropertyValue(originalContext.getOwner(), CudaPrecision());
platforms/cuda/src/CudaPlatform.cpp:    string cpuPmePropValue = platform.getPropertyValue(originalContext.getOwner(), CudaUseCpuPme());
platforms/cuda/src/CudaPlatform.cpp:    string tempPropValue = platform.getPropertyValue(originalContext.getOwner(), CudaTempDirectory());
platforms/cuda/src/CudaPlatform.cpp:    string pmeStreamPropValue = platform.getPropertyValue(originalContext.getOwner(), CudaDisablePmeStream());
platforms/cuda/src/CudaPlatform.cpp:    string deterministicForcesValue = platform.getPropertyValue(originalContext.getOwner(), CudaDeterministicForces());
platforms/cuda/src/CudaPlatform.cpp:void CudaPlatform::contextDestroyed(ContextImpl& context) const {
platforms/cuda/src/CudaPlatform.cpp:CudaPlatform::PlatformData::PlatformData(ContextImpl* context, const System& system, const string& deviceIndexProperty, const string& blockingProperty, const string& precisionProperty,
platforms/cuda/src/CudaPlatform.cpp:                contexts.push_back(new CudaContext(system, deviceIndex, blocking, precisionProperty, tempProperty, *this, (originalData == NULL ? NULL : originalData->contexts[i])));
platforms/cuda/src/CudaPlatform.cpp:            contexts.push_back(new CudaContext(system, -1, blocking, precisionProperty, tempProperty, *this, (originalData == NULL ? NULL : originalData->contexts[0])));
platforms/cuda/src/CudaPlatform.cpp:    propertyValues[CudaPlatform::CudaDeviceIndex()] = deviceIndex.str();
platforms/cuda/src/CudaPlatform.cpp:    propertyValues[CudaPlatform::CudaDeviceName()] = deviceName.str();
platforms/cuda/src/CudaPlatform.cpp:    propertyValues[CudaPlatform::CudaUseBlockingSync()] = blocking ? "true" : "false";
platforms/cuda/src/CudaPlatform.cpp:    propertyValues[CudaPlatform::CudaPrecision()] = precisionProperty;
platforms/cuda/src/CudaPlatform.cpp:    propertyValues[CudaPlatform::CudaUseCpuPme()] = useCpuPme ? "true" : "false";
platforms/cuda/src/CudaPlatform.cpp:    propertyValues[CudaPlatform::CudaCompiler()] = "";
platforms/cuda/src/CudaPlatform.cpp:    propertyValues[CudaPlatform::CudaTempDirectory()] = tempProperty;
platforms/cuda/src/CudaPlatform.cpp:    propertyValues[CudaPlatform::CudaHostCompiler()] = "";
platforms/cuda/src/CudaPlatform.cpp:    propertyValues[CudaPlatform::CudaDisablePmeStream()] = disablePmeStream ? "true" : "false";
platforms/cuda/src/CudaPlatform.cpp:    propertyValues[CudaPlatform::CudaDeterministicForces()] = deterministicForces ? "true" : "false";
platforms/cuda/src/CudaPlatform.cpp:CudaPlatform::PlatformData::~PlatformData() {
platforms/cuda/src/CudaPlatform.cpp:void CudaPlatform::PlatformData::initializeContexts(const System& system) {
platforms/cuda/src/CudaPlatform.cpp:void CudaPlatform::PlatformData::syncContexts() {
examples/benchmark.py:    if options.opencl_platform is not None and 'OpenCLPlatformIndex' in platform.getPropertyNames():
examples/benchmark.py:        properties['OpenCLPlatformIndex'] = options.opencl_platform
examples/benchmark.py:Example: run the full suite of benchmarks for the CUDA platform, printing the results as a table
examples/benchmark.py:    python benchmark.py --platform=CUDA --style=table
examples/benchmark.py:    python benchmark.py --platform=CUDA --precision=mixed --outfile=benchmark.yaml""")
examples/benchmark.py:parser.add_argument('--disable-pme-stream', default=False, action='store_true', dest='disable_pme_stream', help='disable use of a separate GPU stream for PME')
examples/benchmark.py:parser.add_argument('--device', default=None, dest='device', help='device index for CUDA, HIP, or OpenCL')
examples/benchmark.py:parser.add_argument('--opencl-platform', default=None, dest='opencl_platform', help='platform index for OpenCL')
examples/benchmark.py:parser.add_argument('--precision', default='single', dest='precision', help=f'precision modes for CUDA, HIP, or OpenCL: {PRECISIONS} [default: single]')
examples/benchmark.py:# Attempt to get GPU info
examples/benchmark.py:    if shutil.which('nvidia-smi') is not None:
examples/benchmark.py:        cmd = 'nvidia-smi --query-gpu=driver_version,gpu_name --format=csv,noheader'
examples/benchmark.py:        system_info['nvidia_driver'], system_info['gpu'] = output.strip().split(', ')
examples/HelloSodiumChloride.cpp:// GPU-accelerated constant temperature simulation of a very simple system with
examples/HelloSodiumChloride.cpp:// (1) Load any available OpenMM plugins, e.g. Cuda and Brook.
examples/HelloSodiumChlorideInC.c: * GPU-accelerated constant temperature simulation of a very simple system with
examples/HelloSodiumChlorideInC.c: * (1) Load any available OpenMM plugins, e.g. Cuda and Brook.
examples/HelloEthane.cpp: * GPU-accelerated simulation of a system with both bonded and nonbonded forces, 
examples/HelloEthane.cpp:// (1) Load any available OpenMM plugins, e.g. Cuda and Brook.
examples/HelloArgonInFortran.f90:! API for GPU-accelerated molecular dynamics simulation. The primary goal is
examples/HelloArgonInFortran.f90:    ! Load any shared libraries containing GPU implementations.
examples/HelloArgonInC.c: * API for GPU-accelerated molecular dynamics simulation. The primary goal is
examples/HelloArgonInC.c:    /* Load any shared libraries containing GPU implementations. */
examples/HelloArgon.cpp:// API for GPU-accelerated molecular dynamics simulation. The primary goal is
examples/HelloArgon.cpp:    // Load any shared libraries containing GPU implementations.
examples/HelloWaterBox.cpp: * GPU-accelerated simulation of a system with both bonded and nonbonded forces, 
examples/HelloWaterBox.cpp:// (1) Load any available OpenMM plugins, e.g. Cuda and Brook.
examples/MakefileNotes.txt:instructions -- if you are hoping to get GPU acceleration you 
examples/HelloSodiumChlorideInFortran.f90:! GPU-accelerated constant temperature simulation of a very simple system with
examples/HelloSodiumChlorideInFortran.f90:    ! many steps to take on the GPU in between.
examples/HelloSodiumChlorideInFortran.f90:! (1) Load any available OpenMM plugins, e.g. Cuda and Brook.
.travis.yml:      name: "CPU OpenCL"
.travis.yml:      env: OPENCL=true
.travis.yml:           CUDA=false
.travis.yml:           -OPENMM_BUILD_OPENCL_LIB=ON
.travis.yml:           -DOPENMM_BUILD_OPENCL_TESTS=ON
.travis.yml:           -DOPENCL_INCLUDE_DIR=$HOME/AMDAPPSDK/include
.travis.yml:           -DOPENCL_LIBRARY=$HOME/AMDAPPSDK/lib/x86_64/libOpenCL.so"
.travis.yml:      name: "CUDA Compile"
.travis.yml:      env: CUDA=true
.travis.yml:           OPENCL=false
.travis.yml:           CUDA_VERSION="7.5-18"
.travis.yml:             -DOPENMM_BUILD_CUDA_TESTS=OFF
.travis.yml:             -DOPENMM_BUILD_OPENCL_TESTS=OFF
.travis.yml:             -DOPENCL_LIBRARY=/usr/local/cuda-7.5/lib64/libOpenCL.so
.travis.yml:             -DCUDA_CUDART_LIBRARY=/usr/local/cuda-7.5/lib64/libcudart.so
.travis.yml:             -DCUDA_NVCC_EXECUTABLE=/usr/local/cuda-7.5/bin/nvcc
.travis.yml:             -DCUDA_SDK_ROOT_DIR=/usr/local/cuda-7.5/
.travis.yml:             -DCUDA_TOOLKIT_INCLUDE=/usr/local/cuda-7.5/include
.travis.yml:             -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-7.5/"
.travis.yml:      env: OPENCL=false
.travis.yml:           CUDA=false
.travis.yml:             -DOPENMM_BUILD_OPENCL_TESTS=OFF"
.travis.yml:      env: OPENCL=false
.travis.yml:           CUDA=false
.travis.yml:      env: OPENCL=false
.travis.yml:           CUDA=false
.travis.yml:      env: OPENCL=false
.travis.yml:           CUDA=false
.travis.yml:      env: OPENCL=false
.travis.yml:           CUDA=false
.travis.yml:      env: OPENCL=false
.travis.yml:           CUDA=false
.travis.yml:  - if [[ "$OPENCL" == "true" ]]; then
.travis.yml:      export OPENCL_VENDOR_PATH=${AMDAPPSDK}/etc/OpenCL/vendors;
.travis.yml:      mkdir -p ${OPENCL_VENDOR_PATH};
.travis.yml:      echo libamdocl64.so > ${OPENCL_VENDOR_PATH}/amdocl64.icd;
.travis.yml:  - if [[ "$OPENCL" == "false" && "$CUDA" == "false" && "$TRAVIS_OS_NAME" == "linux" && "${TRAVIS_CPU_ARCH}" != "ppc64le" && "${TRAVIS_CPU_ARCH}" != "arm64" ]]; then
.travis.yml:  - if [[ "$OPENCL" == "false" && "$CUDA" == "false" && "$TRAVIS_OS_NAME" == "osx" ]]; then
.travis.yml:  - if [[ "$CUDA" == "true" ]]; then
.travis.yml:      wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_${CUDA_VERSION}_amd64.deb";
.travis.yml:      sudo dpkg -i cuda-repo-ubuntu1404_${CUDA_VERSION}_amd64.deb;
.travis.yml:      export CUDA_APT=${CUDA_VERSION%-*};
.travis.yml:      export CUDA_APT=${CUDA_APT/./-};
.travis.yml:      sudo apt-get install -y cuda-drivers cuda-core-${CUDA_APT} cuda-cudart-dev-${CUDA_APT} cuda-cufft-dev-${CUDA_APT};
.travis.yml:      export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION%%-*};
.travis.yml:      export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH};
.travis.yml:      export PATH=${CUDA_HOME}/bin:${PATH};
.travis.yml:  - if [[ "$OPENCL" == "true" ]]; then ./TestOpenCLDeviceQuery; fi
.travis.yml:  - if [[ "$OPENCL" == "false" && "$CUDA" == "false" ]]; then
docs-source/developerguide/03_writing_plugins.rst:one for the CUDA Platform, and one for the OpenCL Platform.
docs-source/developerguide/03_writing_plugins.rst:standard platforms: Reference, CPU, CUDA, and OpenCL.
docs-source/developerguide/03_writing_plugins.rst:Now suppose you are writing the OpenCL implementation of StringForce.  Here are
docs-source/developerguide/03_writing_plugins.rst:#. OpenCLCalcStringForceKernel.  This extends CalcStringForceKernel and provides
docs-source/developerguide/03_writing_plugins.rst:   execute many different GPU kernels and create its own internal data structures.
docs-source/developerguide/03_writing_plugins.rst:#. OpenCLStringForceKernelFactory.  This is a KernelFactory subclass that knows
docs-source/developerguide/03_writing_plugins.rst:   how to create instances of OpenCLCalcStringForceKernel.
docs-source/developerguide/03_writing_plugins.rst:\ :code:`Platform::getPlatform("OpenCL")` to get the OpenCL Platform,
docs-source/developerguide/03_writing_plugins.rst:then create a new OpenCLStringForceKernelFactory and call
docs-source/developerguide/03_writing_plugins.rst::code:`registerKernelFactory()` on the Platform to register it.  If the OpenCL
docs-source/developerguide/03_writing_plugins.rst:doing anything.  Most likely this means there is no OpenCL runtime on the
docs-source/developerguide/01_introduction.rst:* Chapter :numref:`the-opencl-platform` discusses the architecture of the OpenCL Platform, providing
docs-source/developerguide/01_introduction.rst:  information relevant to writing OpenCL implementations of new features.
docs-source/developerguide/01_introduction.rst:* Chapter :numref:`the-cuda-platform` discusses the architecture of the CUDA Platform, providing
docs-source/developerguide/01_introduction.rst:  information relevant to writing CUDA implementations of new features.
docs-source/developerguide/01_introduction.rst:  write a single implementation of a feature that can be used for both OpenCL and CUDA.
docs-source/developerguide/07_cuda_platform.rst:.. _the-cuda-platform:
docs-source/developerguide/07_cuda_platform.rst:The CUDA and HIP Platforms
docs-source/developerguide/07_cuda_platform.rst:The CUDA and HIP platforms are very similar to the OpenCL platform, and most of the
docs-source/developerguide/07_cuda_platform.rst:previous chapter applies equally well to them, just changing “OpenCL” to “Cuda” or
docs-source/developerguide/07_cuda_platform.rst:Like the OpenCL platform, the CUDA and HIP platforms compile their kernels at runtime.
docs-source/developerguide/07_cuda_platform.rst:The OpenCL platform, as described in Section :numref:`computing-forces`\ , uses two types of buffers for
docs-source/developerguide/07_cuda_platform.rst:buffer.  In contrast, the CUDA and HIP platforms use *only* the fixed point buffer
docs-source/developerguide/07_cuda_platform.rst:(represented by the CUDA type :code:`long` :code:`long`\ ).
docs-source/developerguide/index.rst:   06_opencl_platform.rst
docs-source/developerguide/index.rst:   07_cuda_platform.rst
docs-source/developerguide/06_opencl_platform.rst:.. _the-opencl-platform:
docs-source/developerguide/06_opencl_platform.rst:The OpenCL Platform
docs-source/developerguide/06_opencl_platform.rst:The OpenCL Platform is much more complicated than the reference Platform.  It
docs-source/developerguide/06_opencl_platform.rst:program with OpenCL.  There are many tutorials on that subject available
docs-source/developerguide/06_opencl_platform.rst:When using the OpenCL Platform, the “platform-specific data” stored in
docs-source/developerguide/06_opencl_platform.rst:ContextImpl is of type OpenCLPlatform::PlatformData, which is declared in
docs-source/developerguide/06_opencl_platform.rst:OpenCLPlatform.h.  The most important field of this class is :code:`contexts`
docs-source/developerguide/06_opencl_platform.rst:, which is a vector of OpenCLContexts.  (There is one OpenCLContext for each
docs-source/developerguide/06_opencl_platform.rst:on a single device, in which case there will be only one OpenCLContext.
docs-source/developerguide/06_opencl_platform.rst:OpenCLContext stores most of the important information about a simulation:
docs-source/developerguide/06_opencl_platform.rst:positions, velocities, forces, an OpenCL CommandQueue used for executing
docs-source/developerguide/06_opencl_platform.rst:OpenCLIntegrationUtilities, OpenCLNonbondedUtilities, and OpenCLBondedUtilities.
docs-source/developerguide/06_opencl_platform.rst:Allocation of device memory is generally done through the OpenCLArray class.  It
docs-source/developerguide/06_opencl_platform.rst:Every kernel is specific to a particular OpenCLContext, which in turn is
docs-source/developerguide/06_opencl_platform.rst::code:`OpenCLContext::createProgram()` makes it easy to specify a list of
docs-source/developerguide/06_opencl_platform.rst:the OpenCLContext.  It allows you to specify the total number of work-items to
docs-source/developerguide/06_opencl_platform.rst:The OpenCL platform supports three precision modes:
docs-source/developerguide/06_opencl_platform.rst:   forces, and energies (returned by the OpenCLContext’s :code:`getPosq()`\ ,
docs-source/developerguide/06_opencl_platform.rst:   the OpenCLContext’s getPosqCorrection() method).  Adding the position and the
docs-source/developerguide/06_opencl_platform.rst::code:`getUseDoublePrecision()` on the OpenCLContext to determine which mode
docs-source/developerguide/06_opencl_platform.rst:format.  Most GPUs support atomic operations on 64 bit integers, which allows
docs-source/developerguide/06_opencl_platform.rst:Some low end GPUs do not support this, however, especially the embedded GPUs
docs-source/developerguide/06_opencl_platform.rst:next multiple of 32.  Call :code:`getPaddedNumAtoms()` on the OpenCLContext
docs-source/developerguide/06_opencl_platform.rst:The OpenCL implementation of each Force object should define a subclass of
docs-source/developerguide/06_opencl_platform.rst:the OpenCLContext.  It implements methods for determining whether particular
docs-source/developerguide/06_opencl_platform.rst:best of cases.  It is even more complicated on a GPU.  Furthermore, the
docs-source/developerguide/06_opencl_platform.rst:The OpenCLNonbondedUtilities class tries to simplify all of this.  To use it you
docs-source/developerguide/06_opencl_platform.rst:You may still be able to use features of OpenCLNonbondedUtilities, but you
docs-source/developerguide/06_opencl_platform.rst:OpenCLNonbondedUtilities, providing a block of OpenCL source code for computing
docs-source/developerguide/06_opencl_platform.rst::code:`addParameter()` on the OpenCLNonbondedUtilities.  You provide an array
docs-source/developerguide/06_opencl_platform.rst:where :code:`nb` is the OpenCLNonbondedUtilities for the context.  Now we
docs-source/developerguide/06_opencl_platform.rst:Just as OpenCLNonbondedUtilities simplifies the task of creating nonbonded
docs-source/developerguide/06_opencl_platform.rst:interactions, OpenCLBondedUtilities simplifies the process for many types of
docs-source/developerguide/06_opencl_platform.rst:Using OpenCLBondedUtilities is very similar to the process described above.  You
docs-source/developerguide/06_opencl_platform.rst:provide a block of OpenCL code for evaluating a single interaction.  This block
docs-source/developerguide/06_opencl_platform.rst:OpenCLBondedUtilities object.  You also provide a list of the particles involved
docs-source/developerguide/06_opencl_platform.rst:Nonbonded calculations are done a bit differently in the OpenCL Platform than in
docs-source/developerguide/06_opencl_platform.rst:The OpenCL Platform addresses this by periodically reordering particles so that
docs-source/developerguide/06_opencl_platform.rst:sequential particles are close together.  This means that what the OpenCL
docs-source/developerguide/06_opencl_platform.rst:needed to be updated, that would make it prohibitively slow.  The OpenCL
docs-source/developerguide/06_opencl_platform.rst:ComputeForceInfo it adds to the OpenCLContext.  It can specify two types of
docs-source/developerguide/06_opencl_platform.rst:#. It can define *particle groups*\ .  The OpenCL Platform will ensure that
docs-source/developerguide/06_opencl_platform.rst:The OpenCLContext’s OpenCLIntegrationUtilities provides features that are used
docs-source/developerguide/06_opencl_platform.rst:OpenCLContext plus the delta stored in the OpenCLIntegrationUtilities.  It then
docs-source/developerguide/08_common_compute.rst:to reduce code duplication between the OpenCL and CUDA platforms.  It allows a
docs-source/developerguide/08_common_compute.rst:OpenCL and CUDA are very similar to each other.  Their computational models are
docs-source/developerguide/08_common_compute.rst:Even their languages for writing kernels are very similar.  Here is an OpenCL
docs-source/developerguide/08_common_compute.rst:Here is the corresponding CUDA kernel.
docs-source/developerguide/08_common_compute.rst:kernel can be compiled equally well either as OpenCL or as CUDA.
docs-source/developerguide/08_common_compute.rst:|Macro                          |OpenCL Definition                                           |CUDA Definition                             |
docs-source/developerguide/08_common_compute.rst:based on the features supported by the device.  In addition, the CUDA compiler
docs-source/developerguide/08_common_compute.rst:defines the symbol :code:`__CUDA_ARCH__`\ , so you can check for this symbol if
docs-source/developerguide/08_common_compute.rst:you want to have different code blocks for CUDA and OpenCL.
docs-source/developerguide/08_common_compute.rst:Both OpenCL and CUDA define vector types like :code:`int2` and :code:`float4`\ .
docs-source/developerguide/08_common_compute.rst:use only the vector types that are supported by both OpenCL and CUDA: 2, 3, and 4
docs-source/developerguide/08_common_compute.rst:CUDA uses functions to construct vector values, such as :code:`make_float2(x, y)`\ .
docs-source/developerguide/08_common_compute.rst:OpenCL instead uses a typecast like syntax: :code:`(float2) (x, y)`\ .  In common
docs-source/developerguide/08_common_compute.rst:code, use the CUDA style :code:`make_` functions.  OpenMM provides definitions
docs-source/developerguide/08_common_compute.rst:of these functions when compiling as OpenCL.
docs-source/developerguide/08_common_compute.rst:In CUDA, vector types are simply data structures.  You can access their elements,
docs-source/developerguide/08_common_compute.rst:but not do much more with them.  In contrast, OpenCL's vectors are mathematical
docs-source/developerguide/08_common_compute.rst:CUDA, OpenMM provides definitions of these operators and functions.
docs-source/developerguide/08_common_compute.rst:OpenCL also supports "swizzle" notation for vectors.  For example, if :code:`f`
docs-source/developerguide/08_common_compute.rst:in CUDA, so swizzle notation cannot be used in common code.  Because stripping
docs-source/developerguide/08_common_compute.rst:64 bit integers are another data type that needs special handling.  Both OpenCL
docs-source/developerguide/08_common_compute.rst:and CUDA support them, but they use different names for them: :code:`long` in OpenCL,
docs-source/developerguide/08_common_compute.rst::code:`long long` in CUDA.  To work around this inconsistency, OpenMM provides
docs-source/developerguide/08_common_compute.rst:Host code for Common Compute is very similar to host code for OpenCL or CUDA.
docs-source/developerguide/08_common_compute.rst:In fact, most of the classes provided by the OpenCL and CUDA platforms are
docs-source/developerguide/08_common_compute.rst:subclasses of Common Compute classes.  For example, OpenCLContext and
docs-source/developerguide/08_common_compute.rst:CudaContext are both subclasses of ComputeContext.  When writing common code,
docs-source/developerguide/08_common_compute.rst:either OpenCL or CUDA just based on the particular context passed to it at
docs-source/developerguide/08_common_compute.rst:runtime.  Similarly, OpenCLNonbondedUtilities and CudaNonbondedUtilities are
docs-source/developerguide/08_common_compute.rst:the device.  OpenCLArray and CudaArray are both subclasses of it.  To simplify
docs-source/developerguide/08_common_compute.rst:ComputeArray.  It acts as a wrapper around an OpenCLArray or CudaArray,
docs-source/developerguide/08_common_compute.rst:OpenCL and CUDA have quite different APIs for compiling and invoking kernels.
docs-source/developerguide/08_common_compute.rst:value for a kernel argument or to access the elements of an array.  OpenCL and
docs-source/developerguide/08_common_compute.rst:CUDA both define types for them, but they have different names, and in any case
docs-source/developerguide/08_common_compute.rst:you want to avoid using OpenCL-specific or CUDA-specific types in common code.
docs-source/developerguide/08_common_compute.rst:define them differently.  In OpenCL, a three component vector is essentially a
docs-source/developerguide/08_common_compute.rst::code:`sizeof(float3)` is 12 in CUDA but 16 in OpenCL.  Within a kernel this
docs-source/developerguide/08_common_compute.rst::code:`mm_` host types defined for three component vectors, because CUDA and
docs-source/developerguide/08_common_compute.rst:OpenCL would require them to be defined in different ways.
docs-source/developerguide/09_customcppforceimpl.rst:will automatically work on all platforms.  You do not need to write any GPU
docs-source/api-c++/index.rst:hardware, from GPUs to supercomputers. A ``Platform`` implements some set of
docs-source/usersguide/library/07_testing_validation.rst:hardware platforms (e.g. different models of GPU), software platforms (e.g.
docs-source/usersguide/library/07_testing_validation.rst:operating systems and OpenCL implementations), and types of simulations.
docs-source/usersguide/library/07_testing_validation.rst:both the CUDA and OpenCL platforms, using both single and double precision (and
docs-source/usersguide/library/07_testing_validation.rst:several days) depending on the speed of your GPU.
docs-source/usersguide/library/07_testing_validation.rst:computed with the OpenCL or CUDA platform are shown in
docs-source/usersguide/library/07_testing_validation.rst:is the force computed by the platform being tested (OpenCL or CUDA).  The median
docs-source/usersguide/library/07_testing_validation.rst:Force                                 OpenCL (single)           OpenCL (double)       CUDA (single)        CUDA (double)
docs-source/usersguide/library/07_testing_validation.rst:OpenCL/CUDA platform
docs-source/usersguide/library/07_testing_validation.rst:ubiquitin in OBC implicit solvent.  All three simulations used the CUDA
docs-source/usersguide/library/07_testing_validation.rst:repeated for OpenCL, CUDA, and CPU platforms.
docs-source/usersguide/library/07_testing_validation.rst:Solvent Model   OpenCL               CUDA                 CPU
docs-source/usersguide/library/01_introduction.rst:The CUDA, HIP, and OpenCL platforms are distributed under the GNU Lesser General
docs-source/usersguide/library/01_introduction.rst:GPU, for example, it will be stored in video memory, and must be transferred to
docs-source/usersguide/library/01_introduction.rst:that are appropriate.  For example, code for GPUs will be written in stream
docs-source/usersguide/library/01_introduction.rst:processing languages such as OpenCL or CUDA, code written to run on clusters
docs-source/usersguide/library/01_introduction.rst:GPU, one would create one or more KernelImpl subclasses that implemented the
docs-source/usersguide/library/01_introduction.rst:computations with GPU kernels, and one or more KernelFactory subclasses to
docs-source/usersguide/library/01_introduction.rst:use (a GPU implementation, a multithreaded CPU implementation, an MPI-based
docs-source/usersguide/library/01_introduction.rst:but that need not be the case.  For a GPU implementation, for example, a single
docs-source/usersguide/library/01_introduction.rst:KernelImpl might invoke several GPU kernels.  Alternatively, a single GPU kernel
docs-source/usersguide/library/01_introduction.rst:**CudaPlatform**\ : This platform is implemented using the CUDA language, and
docs-source/usersguide/library/01_introduction.rst:performs calculations on Nvidia GPUs.
docs-source/usersguide/library/01_introduction.rst:performs calculations on ROCm-compatible AMD GPUs.
docs-source/usersguide/library/01_introduction.rst:**OpenCLPlatform**\ : This platform is implemented using the OpenCL language,
docs-source/usersguide/library/01_introduction.rst:and performs calculations on a variety of types of GPUs and CPUs.
docs-source/usersguide/library/01_introduction.rst:#. The CPU platform is usually the fastest choice when a fast GPU is not
docs-source/usersguide/library/01_introduction.rst:   OpenCL platform running on the CPU.
docs-source/usersguide/library/01_introduction.rst:#. The CUDA platform can be used with NVIDIA GPUs.  For using an AMD GPU,
docs-source/usersguide/library/01_introduction.rst:   use the HIP platform (or the OpenCL platform which is usually slower).  For
docs-source/usersguide/library/01_introduction.rst:   using an Intel or Apple GPU, use the OpenCL platform.
docs-source/usersguide/library/02_compiling.rst:CUDA, OpenCL, or HIP Support
docs-source/usersguide/library/02_compiling.rst:If you want to compile OpenMM with support for running on GPUs, you will need
docs-source/usersguide/library/02_compiling.rst:CUDA, HIP, or OpenCL.  MacOS comes with OpenCL built in, so nothing else needs to
docs-source/usersguide/library/02_compiling.rst:The most recent CUDA Toolkit can be obtained from https://developer.nvidia.com/cuda-downloads.
docs-source/usersguide/library/02_compiling.rst:It includes the headers and libraries needed to compile both CUDA and OpenCL
docs-source/usersguide/library/02_compiling.rst:CUDA applications.  The runtime components for OpenCL applications are included
docs-source/usersguide/library/02_compiling.rst:with the GPU drivers from NVIDIA, AMD, and Intel, so make sure you have an
docs-source/usersguide/library/02_compiling.rst:    conda install -c conda-forge hip-devel hipcc rocm-cmake rocm-device-libs
docs-source/usersguide/library/02_compiling.rst:* Usually the OpenCL library and headers will be detected automatically.  If for
docs-source/usersguide/library/02_compiling.rst:  any reason CMake is unable to find them, set OPENCL_INCLUDE_DIR to point to
docs-source/usersguide/library/02_compiling.rst:  the directory containing the headers (usually /usr/local/cuda/include on Linux)
docs-source/usersguide/library/02_compiling.rst:  and OPENCL_LIBRARY to point to the library (usually /usr/local/cuda/lib64/libOpenCL.so
docs-source/usersguide/library/02_compiling.rst:CUDA, OpenCL, or HIP Support
docs-source/usersguide/library/02_compiling.rst:If you want to compile OpenMM with support for running on GPUs, you will need
docs-source/usersguide/library/02_compiling.rst:CUDA, HIP, or OpenCL.
docs-source/usersguide/library/02_compiling.rst:The most recent CUDA Toolkit can be obtained from https://developer.nvidia.com/cuda-downloads.
docs-source/usersguide/library/02_compiling.rst:It includes the headers and libraries needed to compile both CUDA and OpenCL
docs-source/usersguide/library/02_compiling.rst:CUDA applications.  The runtime components for OpenCL applications are included
docs-source/usersguide/library/02_compiling.rst:with the GPU drivers from NVIDIA, AMD, and Intel, so make sure you have an
docs-source/usersguide/library/02_compiling.rst:To build the HIP platform, install the HIP SDK from https://rocm.docs.amd.com/projects/install-on-windows.
docs-source/usersguide/library/02_compiling.rst:* Usually the OpenCL library and headers will be detected automatically.  If for
docs-source/usersguide/library/02_compiling.rst:  any reason CMake is unable to find them, set OPENCL_INCLUDE_DIR to point to
docs-source/usersguide/library/02_compiling.rst:  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/include", except
docs-source/usersguide/library/02_compiling.rst:  OPENCL_LIBRARY to point to the library (usually "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/lib/x64/OpenCL.lib").
docs-source/usersguide/library/02_compiling.rst:  it is not, set HIP_DIR to "C:\AMD\ROCm\6.1\lib\cmake\hip" and HIPRTC_DIR to
docs-source/usersguide/library/02_compiling.rst:  "C:\AMD\ROCm\6.1\lib\cmake\hiprtc" (substituting the correct version number
docs-source/usersguide/library/06_integration_examples.rst:   resources being used on the GPU.
docs-source/usersguide/library/03_tutorials.rst:  extremely fast on a GPU but very, very slowly on a CPU, so it is an excellent
docs-source/usersguide/library/03_tutorials.rst:  example to use to compare performance on the GPU versus the CPU.  The other
docs-source/usersguide/library/03_tutorials.rst:CPU (Reference platform) or a GPU (CUDA or OpenCL platform).  It will say one of
docs-source/usersguide/library/03_tutorials.rst:    REMARK  Using OpenMM platform Cuda
docs-source/usersguide/library/03_tutorials.rst:    REMARK  Using OpenMM platform OpenCL
docs-source/usersguide/library/03_tutorials.rst:If you have a supported GPU, the program should, by default, run on the GPU.
docs-source/usersguide/library/03_tutorials.rst:CPU (Reference platform) or a GPU (CUDA or OpenCL platform).  It will say one of
docs-source/usersguide/library/03_tutorials.rst:    REMARK  Using OpenMM platform Cuda
docs-source/usersguide/library/03_tutorials.rst:    REMARK  Using OpenMM platform OpenCL
docs-source/usersguide/library/03_tutorials.rst:If you have a supported GPU, the program should, by default, run on the GPU.
docs-source/usersguide/library/03_tutorials.rst:Running a program on GPU platforms
docs-source/usersguide/library/03_tutorials.rst:program on another platform (e.g., an NVIDIA or AMD GPU), you need to load the
docs-source/usersguide/library/03_tutorials.rst:required shared libraries for that other platform (e.g., Cuda, OpenCL).  The
docs-source/usersguide/library/03_tutorials.rst:   information can reside on the GPUs and requires communication overhead to
docs-source/usersguide/library/03_tutorials.rst::code:`platformName` so the calling program knows what platform (e.g., CUDA,
docs-source/usersguide/library/03_tutorials.rst:OpenCL, Reference) was used.
docs-source/usersguide/library/05_languages_not_cpp.rst:GPU or is running (slowly) on the Reference platform. However, the argon example
docs-source/usersguide/library/05_languages_not_cpp.rst:is executing on the GPU or is running (slowly) on the Reference platform.
docs-source/usersguide/library/05_languages_not_cpp.rst:When using the Python API, be sure to include the GPU support
docs-source/usersguide/library/08_amoeba_plugin.rst:  calculated using an algorithm that does not map well to GPUs.  Instead the
docs-source/usersguide/library/08_amoeba_plugin.rst:* Calculations using the CUDA platform may be done in either single or double
docs-source/usersguide/library/08_amoeba_plugin.rst:used the CUDA platform, and were repeated for both single and double precision.
docs-source/usersguide/library/04_platform_specifics.rst:    Platform& platform = Platform::getPlatform("OpenCL");
docs-source/usersguide/library/04_platform_specifics.rst:OpenCL Platform
docs-source/usersguide/library/04_platform_specifics.rst:The OpenCL Platform recognizes the following Platform-specific properties:
docs-source/usersguide/library/04_platform_specifics.rst:* OpenCLPlatformIndex: When multiple OpenCL implementations are installed on
docs-source/usersguide/library/04_platform_specifics.rst:  zero-based index of the platform (in the OpenCL sense, not the OpenMM sense) to use,
docs-source/usersguide/library/04_platform_specifics.rst:  in the order they are returned by the OpenCL platform API.  This is useful, for
docs-source/usersguide/library/04_platform_specifics.rst:  example, in selecting whether to use a GPU or CPU based OpenCL implementation.
docs-source/usersguide/library/04_platform_specifics.rst:* DeviceIndex: When multiple OpenCL devices are available on your
docs-source/usersguide/library/04_platform_specifics.rst:  index of the device to use, in the order they are returned by the OpenCL device
docs-source/usersguide/library/04_platform_specifics.rst:The OpenCL Platform also supports parallelizing a simulation across multiple
docs-source/usersguide/library/04_platform_specifics.rst:GPUs.  To do that, set the DeviceIndex property to a comma separated list
docs-source/usersguide/library/04_platform_specifics.rst:CUDA Platform
docs-source/usersguide/library/04_platform_specifics.rst:The CUDA Platform recognizes the following Platform-specific properties:
docs-source/usersguide/library/04_platform_specifics.rst:* DeviceIndex: When multiple CUDA devices are available on your computer,
docs-source/usersguide/library/04_platform_specifics.rst:  the device to use, in the order they are returned by the CUDA API.
docs-source/usersguide/library/04_platform_specifics.rst:* UseBlockingSync: This is used to control how the CUDA runtime
docs-source/usersguide/library/04_platform_specifics.rst:  synchronizes between the CPU and GPU.  If this is set to “true” (the default),
docs-source/usersguide/library/04_platform_specifics.rst:  CUDA will allow the calling thread to sleep while the GPU is performing a
docs-source/usersguide/library/04_platform_specifics.rst:  computation, allowing the CPU to do other work.  If it is set to “false”, CUDA
docs-source/usersguide/library/04_platform_specifics.rst:  will spin-lock while the GPU is working.  Setting it to "false" can improve performance slightly,
docs-source/usersguide/library/04_platform_specifics.rst:  but also prevents the CPU from doing anything else while the GPU is working.
docs-source/usersguide/library/04_platform_specifics.rst:* DeterministicForces: In some cases, the CUDA platform may compute forces
docs-source/usersguide/library/04_platform_specifics.rst:The CUDA Platform also supports parallelizing a simulation across multiple GPUs.
docs-source/usersguide/library/04_platform_specifics.rst:the CUDA platform.
docs-source/usersguide/library/04_platform_specifics.rst:using PME on the Reference, OpenCL, and double-precision CUDA will result in
docs-source/usersguide/library/04_platform_specifics.rst:deterministic simulations. Single-precision CUDA and CPU platforms are not
docs-source/usersguide/application/02_running_sims.rst:OpenMM includes five platforms: :class:`Reference`, :class:`CPU`, :class:`CUDA`, :class:`OpenCL`, and :class:`HIP`.  For a
docs-source/usersguide/application/02_running_sims.rst::class:`Simulation`.  The following lines specify to use the :class:`CUDA` platform:
docs-source/usersguide/application/02_running_sims.rst:    platform = Platform.getPlatform('CUDA')
docs-source/usersguide/application/02_running_sims.rst:The platform name should be one of :code:`OpenCL`, :code:`CUDA`, :code:`HIP`, :code:`CPU`, or
docs-source/usersguide/application/02_running_sims.rst:work across two different GPUs (CUDA devices 0 and 1), doing all computations in
docs-source/usersguide/application/02_running_sims.rst:    platform = Platform.getPlatform('CUDA')
docs-source/usersguide/application/01_getting_started.rst:\2. (Optional) If you want to run OpenMM on a GPU, make sure you have installed
docs-source/usersguide/application/01_getting_started.rst:  * If you have an NVIDIA GPU, download the latest drivers from
docs-source/usersguide/application/01_getting_started.rst:    https://www.nvidia.com/Download/index.aspx. CUDA itself will be installed
docs-source/usersguide/application/01_getting_started.rst:  * If you have an AMD GPU and are using Linux or Windows, download the latest
docs-source/usersguide/application/01_getting_started.rst:    platform (recommended), you also need to install HIP/ROCm by following the
docs-source/usersguide/application/01_getting_started.rst:    instructions at https://rocm.docs.amd.com.
docs-source/usersguide/application/01_getting_started.rst:  * On macOS, OpenCL is included with the operating system and is supported on
docs-source/usersguide/application/01_getting_started.rst:OpenMM compiled with the latest version of CUDA supported by your drivers.
docs-source/usersguide/application/01_getting_started.rst:Alternatively you can request a version that is compiled for a specific CUDA
docs-source/usersguide/application/01_getting_started.rst:    conda install -c conda-forge openmm cuda-version=12
docs-source/usersguide/application/01_getting_started.rst:where :code:`12` should be replaced with the particular CUDA version
docs-source/usersguide/application/01_getting_started.rst:you want to target.  We build packages for CUDA 11 and above.  Because different
docs-source/usersguide/application/01_getting_started.rst:CUDA releases are not binary compatible with each other, OpenMM can only work
docs-source/usersguide/application/01_getting_started.rst:with the particular CUDA version it was compiled with.
docs-source/usersguide/application/01_getting_started.rst:GPU, we recommend installing with pip instead.
docs-source/usersguide/application/01_getting_started.rst:The package installed with that command includes the OpenCL, CPU, and Reference
docs-source/usersguide/application/01_getting_started.rst:platforms.  To also install the CUDA platform (recommended if you have an NVIDIA
docs-source/usersguide/application/01_getting_started.rst:GPU), type
docs-source/usersguide/application/01_getting_started.rst:    pip install openmm[cuda12]
docs-source/usersguide/application/01_getting_started.rst:This will install a copy of the CUDA platform compiled with CUDA 12.  Alternatively,
docs-source/usersguide/application/01_getting_started.rst:if you have an AMD GPU, use this command to include the HIP platform (compiled
docs-source/usersguide/application/01_getting_started.rst:This command confirms that OpenMM is installed, checks whether GPU acceleration
docs-source/usersguide/application/01_getting_started.rst:is available (via the CUDA, OpenCL, and/or HIP platforms), and verifies that all
docs-source/licenses/Licenses.txt:2. CUDA and OpenCL Platforms
docs-source/licenses/Licenses.txt:The CUDA Platform and OpenCL Platform may be used under the terms of the GNU

```
