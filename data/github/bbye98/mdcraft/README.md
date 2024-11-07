# https://github.com/bbye98/mdcraft

```console
paper/paper.bib:  abstract = {GROMACS is one of the most widely used open-source and free software codes in chemistry, used primarily for dynamical simulations of biomolecules. It provides a rich set of calculation types, preparation and analysis tools. Several advanced techniques for free-energy calculations are supported. In version 5, it reaches new performance heights, through several new and enhanced parallelization algorithms. These work on every level; SIMD registers inside cores, multithreading, heterogeneous CPU--GPU acceleration, state-of-the-art 3D domain decomposition, and ensemble-level parallelization through built-in replica exchange and the separate Copernicus framework. The latest best-in-class compressed trajectory storage format is supported.},
paper/paper.bib:  keywords = {Free energy,GPU,Molecular dynamics,SIMD},
lib/openmm-ic-plugin/README.md:Currently, only the CUDA platform is supported.
lib/openmm-ic-plugin/README.md:   `OPENMM_CUDA_COMPILER=$(which nvcc)`, etc.
lib/openmm-ic-plugin/README.md:7. Make sure that `CUDA_TOOLKIT_ROOT_DIR` is set correctly and that 
lib/openmm-ic-plugin/README.md:   `IC_BUILD_CUDA_LIB` is enabled.
lib/openmm-ic-plugin/CMakeLists.txt:FIND_PACKAGE(CUDA QUIET)
lib/openmm-ic-plugin/CMakeLists.txt:IF(CUDA_FOUND)
lib/openmm-ic-plugin/CMakeLists.txt:    SET(IC_BUILD_CUDA_LIB ON CACHE BOOL "Build implementation for CUDA")
lib/openmm-ic-plugin/CMakeLists.txt:ELSE(CUDA_FOUND)
lib/openmm-ic-plugin/CMakeLists.txt:    SET(IC_BUILD_CUDA_LIB OFF CACHE BOOL "Build implementation for CUDA")
lib/openmm-ic-plugin/CMakeLists.txt:ENDIF(CUDA_FOUND)
lib/openmm-ic-plugin/CMakeLists.txt:IF(IC_BUILD_CUDA_LIB)
lib/openmm-ic-plugin/CMakeLists.txt:    ADD_SUBDIRECTORY(platforms/cuda)
lib/openmm-ic-plugin/CMakeLists.txt:ENDIF(IC_BUILD_CUDA_LIB)
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernels.h:#ifndef CUDA_IC_KERNELS_H_
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernels.h:#define CUDA_IC_KERNELS_H_
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernels.h:#include "CudaArray.h"
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernels.h:#include "CudaContext.h"
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernels.h:class CudaIntegrateICLangevinStepKernel : public IntegrateICLangevinStepKernel {
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernels.h:    CudaIntegrateICLangevinStepKernel(std::string name,
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernels.h:                                      const Platform& platform, CudaContext& cu)
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernels.h:    ~CudaIntegrateICLangevinStepKernel();
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernels.h:    CudaContext& cu;
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernels.h:    CudaArray *params, *invAtomIndex;
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernels.h:class CudaIntegrateICDrudeLangevinStepKernel
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernels.h:    CudaIntegrateICDrudeLangevinStepKernel(std::string name,
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernels.h:                                           CudaContext& cu)
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernels.h:    CudaContext& cu;
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernels.h:    CudaArray normalParticles, pairParticles, invAtomIndex;
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernels.h:#endif /*CUDA_IC_KERNELS_H_*/
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernelFactory.h:#ifndef OPENMM_CUDAICKERNELFACTORY_H_
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernelFactory.h:#define OPENMM_CUDAICKERNELFACTORY_H_
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernelFactory.h:class CudaICKernelFactory : public KernelFactory {
lib/openmm-ic-plugin/platforms/cuda/include/CudaICKernelFactory.h:#endif /*OPENMM_CUDAICKERNELFACTORY_H_*/
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:# OpenMM Image Charge Plugin CUDA Platform
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:SET(IC_CUDA_LIBRARY_NAME OpenMMICCUDA)
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:SET(SHARED_TARGET ${IC_CUDA_LIBRARY_NAME})
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE "${OPENMM_DIR}/include" "${OPENMM_DIR}/include/openmm" "${OPENMM_DIR}/include/openmm/reference" "${OPENMM_DIR}/include/openmm/cuda")
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/cuda/include)
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/cuda/src)
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_BINARY_DIR}/platforms/cuda/src)
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:SET(CUDA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:SET(CUDA_SOURCE_CLASS CudaICKernelSources)
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:SET(CUDA_KERNELS_CPP ${CMAKE_CURRENT_BINARY_DIR}/src/${CUDA_SOURCE_CLASS}.cpp)
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:SET(CUDA_KERNELS_H ${CMAKE_CURRENT_BINARY_DIR}/src/${CUDA_SOURCE_CLASS}.h)
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:SET(SOURCE_FILES ${SOURCE_FILES} ${CUDA_KERNELS_CPP} ${CUDA_KERNELS_H})
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_INCLUDE})
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:FILE(GLOB CUDA_KERNELS ${CUDA_SOURCE_DIR}/kernels/*.cu)
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:ADD_CUSTOM_COMMAND(OUTPUT ${CUDA_KERNELS_CPP} ${CUDA_KERNELS_H}
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:    ARGS -D CUDA_SOURCE_DIR=${CUDA_SOURCE_DIR} -D CUDA_KERNELS_CPP=${CUDA_KERNELS_CPP} -D CUDA_KERNELS_H=${CUDA_KERNELS_H} -D CUDA_SOURCE_CLASS=${CUDA_SOURCE_CLASS} -P ${CMAKE_SOURCE_DIR}/platforms/cuda/EncodeCUDAFiles.cmake
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:    DEPENDS ${CUDA_KERNELS}
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:SET_SOURCE_FILES_PROPERTIES(${CUDA_KERNELS_CPP} ${CUDA_KERNELS_H} PROPERTIES GENERATED TRUE)
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${CUDA_LIBRARIES})
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:TARGET_LINK_LIBRARIES(${SHARED_TARGET} OpenMMCUDA)
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:    SET_TARGET_PROPERTIES(${SHARED_TARGET} PROPERTIES LINK_FLAGS "-F/Library/Frameworks -framework CUDA ${EXTRA_COMPILE_FLAGS}")
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:# Ensure that links to the main CUDA library will be resolved.
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:    SET(CUDA_LIBRARY libOpenMMCUDA.dylib)
lib/openmm-ic-plugin/platforms/cuda/CMakeLists.txt:    INSTALL(CODE "EXECUTE_PROCESS(COMMAND install_name_tool -change ${CUDA_LIBRARY} @loader_path/${CUDA_LIBRARY} ${CMAKE_INSTALL_PREFIX}/lib/plugins/lib${SHARED_TARGET}.dylib)")
lib/openmm-ic-plugin/platforms/cuda/EncodeCUDAFiles.cmake:FILE(GLOB CUDA_KERNELS ${CUDA_SOURCE_DIR}/kernels/*.cu)
lib/openmm-ic-plugin/platforms/cuda/EncodeCUDAFiles.cmake:SET(CUDA_FILE_DECLARATIONS)
lib/openmm-ic-plugin/platforms/cuda/EncodeCUDAFiles.cmake:SET(CUDA_FILE_DEFINITIONS)
lib/openmm-ic-plugin/platforms/cuda/EncodeCUDAFiles.cmake:CONFIGURE_FILE(${CUDA_SOURCE_DIR}/${CUDA_SOURCE_CLASS}.cpp.in ${CUDA_KERNELS_CPP})
lib/openmm-ic-plugin/platforms/cuda/EncodeCUDAFiles.cmake:FOREACH(file ${CUDA_KERNELS})
lib/openmm-ic-plugin/platforms/cuda/EncodeCUDAFiles.cmake:    FILE(RELATIVE_PATH filename ${CUDA_SOURCE_DIR}/kernels ${file})
lib/openmm-ic-plugin/platforms/cuda/EncodeCUDAFiles.cmake:    SET(CUDA_FILE_DECLARATIONS ${CUDA_FILE_DECLARATIONS}static\ const\ std::string\ ${variable_name};\n)
lib/openmm-ic-plugin/platforms/cuda/EncodeCUDAFiles.cmake:    FILE(APPEND ${CUDA_KERNELS_CPP} const\ string\ ${CUDA_SOURCE_CLASS}::${variable_name}\ =\ \"${file_content}\"\;\n)
lib/openmm-ic-plugin/platforms/cuda/EncodeCUDAFiles.cmake:CONFIGURE_FILE(${CUDA_SOURCE_DIR}/${CUDA_SOURCE_CLASS}.h.in ${CUDA_KERNELS_H})
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernelSources.cpp.in:#include "CudaICKernelSources.h"
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernelSources.h.in:#ifndef OPENMM_CUDAICKERNELSOURCES_H_
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernelSources.h.in:#define OPENMM_CUDAICKERNELSOURCES_H_
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernelSources.h.in:class CudaICKernelSources {
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernelSources.h.in:    @CUDA_FILE_DECLARATIONS@
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernelSources.h.in:#endif /*OPENMM_CUDAICKERNELSOURCES_H_*/
lib/openmm-ic-plugin/platforms/cuda/src/kernels/ICLangevin.cu:#if __CUDA_ARCH__ >= 130
lib/openmm-ic-plugin/platforms/cuda/src/kernels/ICLangevin.cu:#if __CUDA_ARCH__ >= 130
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernelFactory.cpp:#include "CudaICKernelFactory.h"
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernelFactory.cpp:#include "CudaICKernels.h"
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernelFactory.cpp:        Platform& platform = Platform::getPlatformByName("CUDA");
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernelFactory.cpp:        CudaICKernelFactory* factory = new CudaICKernelFactory();
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernelFactory.cpp:extern "C" OPENMM_EXPORT_IC void registerCudaICKernelFactories() {
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernelFactory.cpp:        Platform::getPlatformByName("CUDA");
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernelFactory.cpp:        Platform::registerPlatform(new CudaPlatform());
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernelFactory.cpp:KernelImpl* CudaICKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernelFactory.cpp:    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernelFactory.cpp:        return new CudaIntegrateICLangevinStepKernel(name, platform, cu);
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernelFactory.cpp:        return new CudaIntegrateICDrudeLangevinStepKernel(name, platform, cu);
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:#include "CudaBondedUtilities.h"
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:#include "CudaForceInfo.h"
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:#include "CudaICKernelSources.h"
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:#include "CudaICKernels.h"
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:#include "CudaIntegrationUtilities.h"
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:CudaIntegrateICLangevinStepKernel::~CudaIntegrateICLangevinStepKernel() {
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:void CudaIntegrateICLangevinStepKernel::initialize(
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:        CudaICKernelSources::vectorOps + CudaICKernelSources::ICLangevin,
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:    params = new CudaArray(
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:        CudaArray::create<int>(cu, cu.getPaddedNumAtoms(), "invAtomIndex");
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:void CudaIntegrateICLangevinStepKernel::execute(
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:    CudaIntegrationUtilities& integration = cu.getIntegrationUtilities();
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:double CudaIntegrateICLangevinStepKernel::computeKineticEnergy(
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:void CudaIntegrateICDrudeLangevinStepKernel::initialize(
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:    CUmodule module = cu.createModule(CudaICKernelSources::vectorOps +
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:                                          CudaICKernelSources::ICDrudeLangevin +
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:                                          CudaICKernelSources::ICLangevin,
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:void CudaIntegrateICDrudeLangevinStepKernel::execute(
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:    CudaIntegrationUtilities& integration = cu.getIntegrationUtilities();
lib/openmm-ic-plugin/platforms/cuda/src/CudaICKernels.cpp:double CudaIntegrateICDrudeLangevinStepKernel::computeKineticEnergy(
src/mdcraft/openmm/utility.py:    with the particle mesh Ewald (PME) method on a GPU (CUDA or OpenCL).
src/mdcraft/openmm/utility.py:    calculations are being evaluated on the GPU.
src/mdcraft/openmm/utility.py:        logging.info(f"  GPU: {steps:14,} ts " f"===> {time:{time_width}.5f} s elapsed")
src/mdcraft/openmm/utility.py:    cutoffs = {"gpu": {min_cutoff}}
src/mdcraft/openmm/utility.py:                    cutoffs["gpu"].add(cutoff)

```
