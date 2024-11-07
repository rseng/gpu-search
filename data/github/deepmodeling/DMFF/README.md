# https://github.com/deepmodeling/DMFF

```console
package/docker/develop_cpu.dockerfile:    conda remove cudatoolkit --force -y && \
package/docker/develop_gpu.dockerfile:	export CONDA_OVERRIDE_CUDA="12.0" && \
package/docker/develop_gpu.dockerfile:	conda create -y -n dmff_omm -c conda-forge python=3.11 openmm libtensorflow_cc tensorflow-gpu swig numpy && \
package/docker/develop_gpu.dockerfile:# wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-$TF_VERSION.tar.gz
package/docker/develop_gpu.dockerfile:# tar -xf libtensorflow-gpu-linux-x86_64-$TF_VERSION.tar.gz -C /usr/local
docs/user_guide/2.installation.md:+ Install [jax](https://github.com/google/jax) (select the correct cuda version, see more details in the Jax installation guide):
docs/user_guide/2.installation.md:# GPU version
docs/user_guide/2.installation.md:pip install "jax[cuda11_local]==0.4.14" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
docs/index.md:**DMFF** (**D**ifferentiable **M**olecular **F**orce **F**ield) is a Jax-based python package that provides a full differentiable implementation of molecular force field models. This project aims to establish an extensible codebase to minimize the efforts in force field parameterization, and to ease the force and virial tensor evaluations for advanced complicated potentials (e.g., polarizable models with geometry-dependent atomic parameters). Currently, this project mainly focuses on the molecular systems such as: water, biological macromolecules (peptides, proteins, nucleic acids), organic polymers, and small organic molecules (organic electrolyte, drug-like molecules) etc. We support both the conventional point charge models (OPLS and AMBER like) and multipolar polarizable models (AMOEBA and MPID like). The entire project is backed by the XLA technique in JAX, thus can be "jitted" and run in GPU devices much more efficiently compared to normal python codes.
backend/save_dmff2tf.py:gpus = tf.config.experimental.list_physical_devices('GPU')
backend/save_dmff2tf.py:for gpu in gpus:
backend/save_dmff2tf.py:  tf.config.experimental.set_memory_growth(gpu, True)
backend/openmm_dmff_plugin/python/setup.py:      package_data={"OpenMMDMFFPlugin":['data/lj_fluid/*.pb', 'data/lj_fluid/variables/variables.index', 'data/lj_fluid/variables/variables.data-00000-of-00001', 'data/lj_fluid_gpu/*.pb', 'data/lj_fluid_gpu/variables/variables.index', 'data/lj_fluid_gpu/variables/variables.data-00000-of-00001', 'data/*.pdb']},
backend/openmm_dmff_plugin/python/tests/test_dmff_plugin_nve.py:        model_dir = os.path.join(os.path.dirname(__file__), "../data", "lj_fluid_gpu")
backend/openmm_dmff_plugin/python/OpenMMDMFFPlugin/tests/test_dmff_plugin_nve.py:        model_dir = os.path.join(os.path.dirname(__file__), "../data", "lj_fluid_gpu")
backend/openmm_dmff_plugin/python/OpenMMDMFFPlugin/__init__.py:(especially on recent GPUs) that make it truly unique among simulation codes.
backend/openmm_dmff_plugin/README.md:Install the python, openmm and cudatoolkit.
backend/openmm_dmff_plugin/README.md:conda create -n dmff_omm -c conda-forge python=3.9 openmm cudatoolkit=11.6
backend/openmm_dmff_plugin/README.md:   python -m OpenMMDMFFPlugin.tests.test_dmff_plugin_nvt -n 100 --platform CUDA
backend/openmm_dmff_plugin/CMakeLists.txt:FIND_PACKAGE(CUDA QUIET)
backend/openmm_dmff_plugin/CMakeLists.txt:IF(CUDA_FOUND)
backend/openmm_dmff_plugin/CMakeLists.txt:    message(STATUS "CUDA found, building CUDA implementation")
backend/openmm_dmff_plugin/CMakeLists.txt:    SET(PLUGIN_BUILD_CUDA_LIB ON CACHE BOOL "Build implementation for CUDA: ON")
backend/openmm_dmff_plugin/CMakeLists.txt:ELSE(CUDA_FOUND)
backend/openmm_dmff_plugin/CMakeLists.txt:    message(STATUS "CUDA not found, not building CUDA implementation")
backend/openmm_dmff_plugin/CMakeLists.txt:    SET(PLUGIN_BUILD_CUDA_LIB OFF CACHE BOOL "Build implementation for CUDA: OFF")
backend/openmm_dmff_plugin/CMakeLists.txt:ENDIF(CUDA_FOUND)
backend/openmm_dmff_plugin/CMakeLists.txt:IF(PLUGIN_BUILD_CUDA_LIB)
backend/openmm_dmff_plugin/CMakeLists.txt:    ADD_SUBDIRECTORY(platforms/cuda)
backend/openmm_dmff_plugin/CMakeLists.txt:ENDIF(PLUGIN_BUILD_CUDA_LIB)
backend/openmm_dmff_plugin/serialization/tests/TestSerializeDMFFForce.cpp:    const string graph = "../python/OpenMMDMFFPlugin/data/lj_fluid_gpu";
backend/openmm_dmff_plugin/platforms/reference/tests/TestDMFFPlugin4Reference.cpp:const string graph = "../python/OpenMMDMFFPlugin/data/lj_fluid_gpu";
backend/openmm_dmff_plugin/platforms/cuda/tests/TestDMFFPlugin4CUDA.cpp:extern "C" OPENMM_EXPORT void registerDMFFCudaKernelFactories();
backend/openmm_dmff_plugin/platforms/cuda/tests/TestDMFFPlugin4CUDA.cpp:const string graph = "../python/OpenMMDMFFPlugin/data/lj_fluid_gpu";
backend/openmm_dmff_plugin/platforms/cuda/tests/TestDMFFPlugin4CUDA.cpp:    Platform& platform = Platform::getPlatformByName("CUDA");
backend/openmm_dmff_plugin/platforms/cuda/tests/TestDMFFPlugin4CUDA.cpp:        registerDMFFCudaKernelFactories();
backend/openmm_dmff_plugin/platforms/cuda/tests/TestDMFFPlugin4CUDA.cpp:            Platform::getPlatformByName("CUDA").setPropertyDefaultValue("CudaPrecision", string(argv[1]));
backend/openmm_dmff_plugin/platforms/cuda/tests/CMakeLists.txt:INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIR})
backend/openmm_dmff_plugin/platforms/cuda/tests/CMakeLists.txt:        SET_TARGET_PROPERTIES(${TEST_ROOT} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS} -F/Library/Frameworks -framework CUDA" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")
backend/openmm_dmff_plugin/platforms/cuda/include/CudaDMFFKernels.h:#ifndef CUDA_DMFF_KERNELS_H_
backend/openmm_dmff_plugin/platforms/cuda/include/CudaDMFFKernels.h:#define CUDA_DMFF_KERNELS_H_
backend/openmm_dmff_plugin/platforms/cuda/include/CudaDMFFKernels.h:#include "openmm/cuda/CudaContext.h"
backend/openmm_dmff_plugin/platforms/cuda/include/CudaDMFFKernels.h:#include "openmm/cuda/CudaArray.h"
backend/openmm_dmff_plugin/platforms/cuda/include/CudaDMFFKernels.h:class CudaCalcDMFFForceKernel : public CalcDMFFForceKernel{
backend/openmm_dmff_plugin/platforms/cuda/include/CudaDMFFKernels.h:    CudaCalcDMFFForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu):CalcDMFFForceKernel(name, platform), cu(cu){};
backend/openmm_dmff_plugin/platforms/cuda/include/CudaDMFFKernels.h:    ~CudaCalcDMFFForceKernel();
backend/openmm_dmff_plugin/platforms/cuda/include/CudaDMFFKernels.h:    // Used for CUDA Platform.
backend/openmm_dmff_plugin/platforms/cuda/include/CudaDMFFKernels.h:    OpenMM::CudaContext& cu;
backend/openmm_dmff_plugin/platforms/cuda/include/CudaDMFFKernels.h:    OpenMM::CudaArray dmffForces;
backend/openmm_dmff_plugin/platforms/cuda/include/CudaDMFFKernels.h:#endif /*CUDA_DMFF_KERNELS_H_*/
backend/openmm_dmff_plugin/platforms/cuda/include/CudaDMFFKernelFactory.h:#ifndef OPENMM_CUDA_DMFF_KERNEL_FACTORY_H_
backend/openmm_dmff_plugin/platforms/cuda/include/CudaDMFFKernelFactory.h:#define OPENMM_CUDA_DMFF_KERNEL_FACTORY_H_
backend/openmm_dmff_plugin/platforms/cuda/include/CudaDMFFKernelFactory.h: * This KernelFactory creates kernels for the CUDA implementation of the DMFF plugin.
backend/openmm_dmff_plugin/platforms/cuda/include/CudaDMFFKernelFactory.h:class CudaDMFFKernelFactory : public KernelFactory {
backend/openmm_dmff_plugin/platforms/cuda/include/CudaDMFFKernelFactory.h:#endif /*OPENMM_CUDA_DMFF_KERNEL_FACTORY_H_*/
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:# OpenMM DMFF Plugin CUDA Platform
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:SET(PLUGIN_CUDA_LIBRARY_NAME OpenMMDMFFCUDA)
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:SET(SHARED_TARGET ${PLUGIN_CUDA_LIBRARY_NAME})
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/cuda/include)
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/cuda/src)
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(BEFORE ${CMAKE_BINARY_DIR}/platforms/cuda/src)
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:SET(CUDA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:SET(CUDA_SOURCE_CLASS CudaDMFFKernelSources)
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:SET(CUDA_KERNELS_CPP ${CMAKE_CURRENT_BINARY_DIR}/src/${CUDA_SOURCE_CLASS}.cpp)
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:SET(CUDA_KERNELS_H ${CMAKE_CURRENT_BINARY_DIR}/src/${CUDA_SOURCE_CLASS}.h)
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:SET(SOURCE_FILES ${SOURCE_FILES} ${CUDA_KERNELS_CPP} ${CUDA_KERNELS_H})
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_INCLUDE})
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:FILE(GLOB CUDA_KERNELS ${CUDA_SOURCE_DIR}/kernels/*.cu)
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:ADD_CUSTOM_COMMAND(OUTPUT ${CUDA_KERNELS_CPP} ${CUDA_KERNELS_H}
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:    ARGS -D CUDA_SOURCE_DIR=${CUDA_SOURCE_DIR} -D CUDA_KERNELS_CPP=${CUDA_KERNELS_CPP} -D CUDA_KERNELS_H=${CUDA_KERNELS_H} -D CUDA_SOURCE_CLASS=${CUDA_SOURCE_CLASS} -P ${CMAKE_SOURCE_DIR}/platforms/cuda/EncodeCUDAFiles.cmake
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:    DEPENDS ${CUDA_KERNELS}
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:SET_SOURCE_FILES_PROPERTIES(${CUDA_KERNELS_CPP} ${CUDA_KERNELS_H} PROPERTIES GENERATED TRUE)
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${CUDA_LIBRARIES})
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:TARGET_LINK_LIBRARIES(${SHARED_TARGET} OpenMMCUDA)
backend/openmm_dmff_plugin/platforms/cuda/CMakeLists.txt:    SET_TARGET_PROPERTIES(${SHARED_TARGET} PROPERTIES LINK_FLAGS "-F/Library/Frameworks -framework CUDA ${EXTRA_COMPILE_FLAGS}")
backend/openmm_dmff_plugin/platforms/cuda/EncodeCUDAFiles.cmake:FILE(GLOB CUDA_KERNELS ${CUDA_SOURCE_DIR}/kernels/*.cu)
backend/openmm_dmff_plugin/platforms/cuda/EncodeCUDAFiles.cmake:SET(CUDA_FILE_DECLARATIONS)
backend/openmm_dmff_plugin/platforms/cuda/EncodeCUDAFiles.cmake:SET(CUDA_FILE_DEFINITIONS)
backend/openmm_dmff_plugin/platforms/cuda/EncodeCUDAFiles.cmake:CONFIGURE_FILE(${CUDA_SOURCE_DIR}/${CUDA_SOURCE_CLASS}.cpp.in ${CUDA_KERNELS_CPP})
backend/openmm_dmff_plugin/platforms/cuda/EncodeCUDAFiles.cmake:FOREACH(file ${CUDA_KERNELS})
backend/openmm_dmff_plugin/platforms/cuda/EncodeCUDAFiles.cmake:    FILE(RELATIVE_PATH filename ${CUDA_SOURCE_DIR}/kernels ${file})
backend/openmm_dmff_plugin/platforms/cuda/EncodeCUDAFiles.cmake:    SET(CUDA_FILE_DECLARATIONS ${CUDA_FILE_DECLARATIONS}static\ const\ std::string\ ${variable_name};\n)
backend/openmm_dmff_plugin/platforms/cuda/EncodeCUDAFiles.cmake:    FILE(APPEND ${CUDA_KERNELS_CPP} const\ string\ ${CUDA_SOURCE_CLASS}::${variable_name}\ =\ \"${file_content}\"\;\n)
backend/openmm_dmff_plugin/platforms/cuda/EncodeCUDAFiles.cmake:CONFIGURE_FILE(${CUDA_SOURCE_DIR}/${CUDA_SOURCE_CLASS}.h.in ${CUDA_KERNELS_H})
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernelFactory.cpp:#include "CudaDMFFKernelFactory.h"
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernelFactory.cpp:#include "CudaDMFFKernels.h"
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernelFactory.cpp:        Platform& platform = Platform::getPlatformByName("CUDA");
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernelFactory.cpp:        CudaDMFFKernelFactory* factory = new CudaDMFFKernelFactory();
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernelFactory.cpp:extern "C" OPENMM_EXPORT void registerDMFFCudaKernelFactories() {
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernelFactory.cpp:        Platform::getPlatformByName("CUDA");
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernelFactory.cpp:        Platform::registerPlatform(new CudaPlatform());
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernelFactory.cpp:KernelImpl* CudaDMFFKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernelFactory.cpp:    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernelFactory.cpp:        return new CudaCalcDMFFForceKernel(name, platform, cu);
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernels.cpp:#include "CudaDMFFKernels.h"
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernels.cpp:#include "CudaDMFFKernelSources.h"
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernels.cpp:CudaCalcDMFFForceKernel::~CudaCalcDMFFForceKernel(){
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernels.cpp:void CudaCalcDMFFForceKernel::initialize(const System& system, const DMFFForce& force){
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernels.cpp:    // Set for CUDA context.
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernels.cpp:    CUmodule module = cu.createModule(CudaDMFFKernelSources::DMFFForce, defines);
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernels.cpp:double CudaCalcDMFFForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernels.cpp:        // Change to OpenMM CUDA context.
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernelSources.cpp.in:#include "CudaDMFFKernelSources.h"
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernelSources.h.in:#ifndef OPENMM_CUDA_DMFF_KERNEL_SOURCES_H_
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernelSources.h.in:#define OPENMM_CUDA_DMFF_KERNEL_SOURCES_H_
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernelSources.h.in: * This class is a central holding place for the source code of CUDA kernels.
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernelSources.h.in:class CudaDMFFKernelSources {
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernelSources.h.in:@CUDA_FILE_DECLARATIONS@
backend/openmm_dmff_plugin/platforms/cuda/src/CudaDMFFKernelSources.h.in:#endif /*OPENMM_CUDA_DMFF_KERNEL_SOURCES_H_*/
README.md:**DMFF** (**D**ifferentiable **M**olecular **F**orce **F**ield) is a Jax-based python package that provides a full differentiable implementation of molecular force field models. This project aims to establish an extensible codebase to minimize the efforts in force field parameterization, and to ease the force and virial tensor evaluations for advanced complicated potentials (e.g., polarizable models with geometry-dependent atomic parameters). Currently, this project mainly focuses on the molecular systems such as: water, biological macromolecules (peptides, proteins, nucleic acids), organic polymers, and small organic molecules (organic electrolyte, drug-like molecules) etc. We support both the conventional point charge models (OPLS and AMBER like) and multipolar polarizable models (AMOEBA and MPID like). The entire project is backed by the XLA technique in JAX, thus can be "jitted" and run in GPU devices much more efficiently compared to normal python codes.
examples/peg_slater_isa/fit.sh:#SBATCH -N 1 -n 1 --gres=gpu:1
examples/peg_slater_isa/run_amoeba.py:    platform_AB = Platform.getPlatformByName('CUDA')
examples/peg_slater_isa/run_amoeba.py:    platform_A = Platform.getPlatformByName('CUDA')
examples/peg_slater_isa/run_amoeba.py:    platform_B = Platform.getPlatformByName('CUDA')
examples/eann/run_ref.py:    # used for select a unoccupied GPU
examples/eann/run_ref.py:    # gpu/cpu
.gitignore:[Dd]ebugPublic/
.gitignore:# NVidia Nsight GPU debugger configuration file

```
