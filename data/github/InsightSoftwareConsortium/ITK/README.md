# https://github.com/InsightSoftwareConsortium/ITK

```console
CMake/ITKModuleMacros.cmake:      if("${itk-module}" MATCHES ITKGPU)
CMake/ITKModuleMacros.cmake:        if(${ITK_USE_GPU})
CMake/itkOpenCL.cmake:if(ITK_USE_GPU)
CMake/itkOpenCL.cmake:  find_package(OpenCL REQUIRED)
CMake/itkOpenCL.cmake:  if(NOT ${OPENCL_FOUND})
CMake/itkOpenCL.cmake:    message(FATAL "Could not find OpenCL")
CMake/itkOpenCL.cmake:    write_gpu_kernel_to_file
CMake/itkOpenCL.cmake:    OPENCL_FILE
CMake/itkOpenCL.cmake:    GPUFILTER_NAME
CMake/itkOpenCL.cmake:    GPUFILTER_KERNELNAME
CMake/itkOpenCL.cmake:    sourcefile_to_string(${OPENCL_FILE} ${GPUFILTER_KERNELNAME}_SourceString)
CMake/itkOpenCL.cmake:    set(${GPUFILTER_KERNELNAME}_KernelString "#include \"itk${GPUFILTER_NAME}.h\"\n\n")
CMake/itkOpenCL.cmake:    set(${GPUFILTER_KERNELNAME}_KernelString "${${GPUFILTER_KERNELNAME}_KernelString}namespace itk\n")
CMake/itkOpenCL.cmake:    set(${GPUFILTER_KERNELNAME}_KernelString "${${GPUFILTER_KERNELNAME}_KernelString}{\n\n")
CMake/itkOpenCL.cmake:    set(${GPUFILTER_KERNELNAME}_KernelString
CMake/itkOpenCL.cmake:        "${${GPUFILTER_KERNELNAME}_KernelString}const char* ${GPUFILTER_KERNELNAME}::GetOpenCLSource()\n")
CMake/itkOpenCL.cmake:    set(${GPUFILTER_KERNELNAME}_KernelString "${${GPUFILTER_KERNELNAME}_KernelString}{\n")
CMake/itkOpenCL.cmake:    set(${GPUFILTER_KERNELNAME}_KernelString
CMake/itkOpenCL.cmake:        "${${GPUFILTER_KERNELNAME}_KernelString}  return ${${GPUFILTER_KERNELNAME}_SourceString};\n")
CMake/itkOpenCL.cmake:    set(${GPUFILTER_KERNELNAME}_KernelString "${${GPUFILTER_KERNELNAME}_KernelString}}\n\n")
CMake/itkOpenCL.cmake:    set(${GPUFILTER_KERNELNAME}_KernelString "${${GPUFILTER_KERNELNAME}_KernelString}}\n")
CMake/itkOpenCL.cmake:    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE} "${${GPUFILTER_KERNELNAME}_KernelString}")
CMake/itkOpenCL.cmake:    configure_file(${OPENCL_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}.cl COPYONLY)
CMake/itkOpenCL.cmake:      ${GPUFILTER_KERNELNAME}_Target
CMake/itkOpenCL.cmake:  macro(write_gpu_kernels GPUKernels GPU_SRC)
CMake/itkOpenCL.cmake:    foreach(GPUKernel ${GPUKernels})
CMake/itkOpenCL.cmake:      get_filename_component(FilterName ${GPUKernel} NAME_WE)
CMake/itkOpenCL.cmake:      write_gpu_kernel_to_file(
CMake/itkOpenCL.cmake:        ${GPUKernel}
CMake/itkOpenCL.cmake:        ${GPU_SRC})
CMake/ITKConfig.cmake.in:# Add configuration with GPU
CMake/ITKConfig.cmake.in:set(ITK_USE_GPU "@ITK_USE_GPU@")
Utilities/Maintenance/git-clang-format:      'cu',  # CUDA
Utilities/Maintenance/RestyleHttpToHttps.sh:    -e 's#http://docs.nvidia.com#https://docs.nvidia.com#g' \
CMakeLists.txt:# Enable GPU support. Requires OpenCL to be installed
CMakeLists.txt:option(ITK_USE_GPU "GPU acceleration via OpenCL" OFF)
CMakeLists.txt:mark_as_advanced(ITK_USE_GPU)
CMakeLists.txt:if(ITK_USE_GPU)
CMakeLists.txt:  include(itkOpenCL)
CMakeLists.txt:option(ITK_USE_CUFFTW "Use NVidia CUDA cuFFT with its FFTW interface for FFT computation." OFF)
CMakeLists.txt:  Module_CudaCommon
Modules/Video/BridgeOpenCV/CMakeLists.txt:set(OpenCV_CUDA ${OpenCV_CUDA}) # Windows specific option
Modules/Video/BridgeOpenCV/CMakeLists.txt:  set(OpenCV_CUDA ${OpenCV_CUDA}) # Windows specific option
Modules/Registration/GPUCommon/itk-module.cmake:set(DOCUMENTATION "This module contains some common components to support GPU-based
Modules/Registration/GPUCommon/itk-module.cmake:  ITKGPURegistrationCommon
Modules/Registration/GPUCommon/itk-module.cmake:  ITKGPUCommon
Modules/Registration/GPUCommon/itk-module.cmake:  ITKGPUFiniteDifference
Modules/Registration/GPUCommon/include/itkGPUPDEDeformableRegistrationFunction.h:#ifndef itkGPUPDEDeformableRegistrationFunction_h
Modules/Registration/GPUCommon/include/itkGPUPDEDeformableRegistrationFunction.h:#define itkGPUPDEDeformableRegistrationFunction_h
Modules/Registration/GPUCommon/include/itkGPUPDEDeformableRegistrationFunction.h:#include "itkGPUFiniteDifferenceFunction.h"
Modules/Registration/GPUCommon/include/itkGPUPDEDeformableRegistrationFunction.h:/** \class GPUPDEDeformableRegistrationFunction
Modules/Registration/GPUCommon/include/itkGPUPDEDeformableRegistrationFunction.h: * \ingroup ITKGPURegistrationCommon
Modules/Registration/GPUCommon/include/itkGPUPDEDeformableRegistrationFunction.h:class ITK_TEMPLATE_EXPORT GPUPDEDeformableRegistrationFunction : public GPUFiniteDifferenceFunction<TDisplacementField>
Modules/Registration/GPUCommon/include/itkGPUPDEDeformableRegistrationFunction.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUPDEDeformableRegistrationFunction);
Modules/Registration/GPUCommon/include/itkGPUPDEDeformableRegistrationFunction.h:  using Self = GPUPDEDeformableRegistrationFunction;
Modules/Registration/GPUCommon/include/itkGPUPDEDeformableRegistrationFunction.h:  using Superclass = GPUFiniteDifferenceFunction<TDisplacementField>;
Modules/Registration/GPUCommon/include/itkGPUPDEDeformableRegistrationFunction.h:  itkOverrideGetNameOfClassMacro(GPUPDEDeformableRegistrationFunction);
Modules/Registration/GPUCommon/include/itkGPUPDEDeformableRegistrationFunction.h:  GPUPDEDeformableRegistrationFunction()
Modules/Registration/GPUCommon/include/itkGPUPDEDeformableRegistrationFunction.h:  ~GPUPDEDeformableRegistrationFunction() override = default;
Modules/Registration/GPUCommon/CMakeLists.txt:project(ITKGPURegistrationCommon)
Modules/Registration/GPUCommon/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Registration/GPUCommon/CMakeLists.txt:  set(ITKGPURegistrationCommon_SYSTEM_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
Modules/Registration/GPUCommon/CMakeLists.txt:  set(ITKGPURegistrationCommon_SYSTEM_LIBRARY_DIRS ${OPENCL_LIBRARIES})
Modules/Registration/GPUPDEDeformable/itk-module.cmake:    "This module contains the GPU implementation of classes
Modules/Registration/GPUPDEDeformable/itk-module.cmake:  ITKGPUPDEDeformableRegistration
Modules/Registration/GPUPDEDeformable/itk-module.cmake:  ITKGPUCommon
Modules/Registration/GPUPDEDeformable/itk-module.cmake:  ITKGPUFiniteDifference
Modules/Registration/GPUPDEDeformable/itk-module.cmake:  ITKGPURegistrationCommon
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest2.cxx:#include "itkGPUDemonsRegistrationFilter.h"
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest2.cxx:itkGPUDemonsRegistrationFilterTest2(int argc, char * argv[])
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest2.cxx:  using ImageType = itk::GPUImage<PixelType, ImageDimension>;
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest2.cxx:  using FieldType = itk::GPUImage<VectorType, ImageDimension>;
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest2.cxx:  using RegistrationType = itk::GPUDemonsRegistrationFilter<ImageType, ImageType, FieldType>;
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest2.cxx:  using FunctionType = RegistrationType::GPUDemonsRegistrationFunctionType;
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:  set(ITKGPUPDEDeformableRegistration-tests itkGPUDemonsRegistrationFilterTest.cxx
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:                                            itkGPUDemonsRegistrationFilterTest2.cxx)
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:  set(ITKGPUPDEDeformableRegistrationTest_LIBRARIES
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:      ITKCommon;ITKGPUCommon;ITKGPUFiniteDifference;ITKGPUPDEDeformableRegistration;ITKStatistics;ITKSpatialObjects;ITKPath;ITKOptimizers;ITKIOBMP;ITKIOBioRad;ITKIOGDCM;ITKIOGIPL;ITKIOJPEG;ITKIOLSM;ITKIOMeta;ITKIONIFTI;ITKIONRRD;ITKIOPNG;ITKIOStimulate;ITKIOTIFF;ITKIOVTK;itksys
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:  createtestdriver(ITKGPUPDEDeformableRegistration "${ITKGPUPDEDeformableRegistrationTest_LIBRARIES}"
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:                   "${ITKGPUPDEDeformableRegistration-tests}")
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:    itkGPUDemonsRegistrationFilterTestDim2
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:    ITKGPUPDEDeformableRegistrationTestDriver
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:    itkGPUDemonsRegistrationFilterTest
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuDemonsRegistrationTest2D.mha)
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:    itkGPUDemonsRegistrationFilterTestDim3
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:    ITKGPUPDEDeformableRegistrationTestDriver
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:    itkGPUDemonsRegistrationFilterTest
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuDemonsRegistrationTest2D.mha)
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:    itkGPUDemonsRegistrationFilterTest2
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:    ITKGPUPDEDeformableRegistrationTestDriver
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:    itkGPUDemonsRegistrationFilterTest2
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuDemonsRegistrationTest2Fixed.mha
Modules/Registration/GPUPDEDeformable/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuDemonsRegistrationTest2Warped.mha)
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx: * Test program for itkGPUDemonsRegistrationFilter class
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx: * This program creates a GPU Mean filter and a CPU threshold filter using
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx: * object factory framework and test pipelining of GPU and CPU filters.
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:// #include "pathToOpenCLSourceCode.h"
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:#include "itkGPUDemonsRegistrationFilter.h"
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:#include "itkGPUImage.h"
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:#include "itkGPUKernelManager.h"
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:#include "itkGPUContextManager.h"
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:itk::TimeProbe m_GPUTime;
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:GPUDemonsRegistrationFilterTestTemplate(int argc, char * argv[]);
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:itkGPUDemons(int argc, char * argv[]);
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:itkGPUDemonsRegistrationFilterTest(int argc, char * argv[])
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:      returnValue = GPUDemonsRegistrationFilterTestTemplate<2>(argc, argv);
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:      returnValue = GPUDemonsRegistrationFilterTestTemplate<3>(argc, argv);
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:GPUDemonsRegistrationFilterTestTemplate(int argc, char * argv[])
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:  using GPUDisplacementFieldType = itk::GPUImage<VectorPixelType, ImageDimension>;
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:  using GPUDisplacementFieldPointer = typename GPUDisplacementFieldType::Pointer;
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:  GPUDisplacementFieldPointer gpuOut;
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:    std::cout << "Starting GPU Demons" << std::endl;
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:    gpuOut = (itkGPUDemons<ImageDimension, GPUDisplacementFieldPointer>(argc, argv));
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:    std::cout << "Finished GPU Demons" << std::endl;
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:  std::cout << "Average GPU registration time in seconds = " << m_GPUTime.GetMean() << std::endl;
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:  InternalPixelType *gpuBuf, *cpuBuf;
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:  gpuBuf = (InternalPixelType *)gpuOut->GetBufferPointer();
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:  size1 = gpuOut->GetLargestPossibleRegion().GetNumberOfPixels();
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:      tmp = gpuBuf[i * ImageDimension + d] - cpuBuf[i * ImageDimension + d];
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:  // std::cout << "Total GPU time in seconds = " << m_GPUTime.GetMean() <<
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:  // std::cout << "Initial GPU time in seconds = " << gpuInitTime.GetMean() <<
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:itkGPUDemons(int, char * argv[])
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:  using InternalImageType = itk::GPUImage<InternalPixelType, Dimension>;
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:    itk::GPUDemonsRegistrationFilter<InternalImageType, InternalImageType, DisplacementFieldType>;
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:  m_GPUTime.Start();
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:  m_GPUTime.Stop();
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:  std::cout << "GPU InitTime in seconds = " << filter->GetInitTime().GetTotal() << std::endl;
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:  std::cout << "GPU ComputeUpdateTime in seconds = " << filter->GetComputeUpdateTime().GetTotal() << std::endl;
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:  std::cout << "GPU ApplyUpdateTime in seconds = " << filter->GetApplyUpdateTime().GetTotal() << std::endl;
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:  std::cout << "GPU SmoothFieldTime in seconds = " << filter->GetSmoothFieldTime().GetTotal() << std::endl;
Modules/Registration/GPUPDEDeformable/test/itkGPUDemonsRegistrationFilterTest.cxx:  char * outName = AppendFileName(argv[5], "_gpu");
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:#ifndef itkGPUPDEDeformableRegistrationFilter_h
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:#define itkGPUPDEDeformableRegistrationFilter_h
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:#include "itkGPUDenseFiniteDifferenceImageFilter.h"
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:#include "itkGPUPDEDeformableRegistrationFunction.h"
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h: * \class GPUPDEDeformableRegistrationFilter
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h: * GPUPDEDeformableRegistrationFilter is a base case for filter implementing
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h: * \ingroup ITKGPUPDEDeformableRegistration
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:/** Create a helper GPU Kernel class for GPUPDEDeformableRegistrationFilter */
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:itkGPUKernelClassMacro(GPUPDEDeformableRegistrationFilterKernel);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:class ITK_TEMPLATE_EXPORT GPUPDEDeformableRegistrationFilter
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  : public GPUDenseFiniteDifferenceImageFilter<TDisplacementField, TDisplacementField, TParentImageFilter>
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUPDEDeformableRegistrationFilter);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  using Self = GPUPDEDeformableRegistrationFilter;
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  using GPUSuperclass = GPUDenseFiniteDifferenceImageFilter<TDisplacementField, TDisplacementField, TParentImageFilter>;
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  itkOverrideGetNameOfClassMacro(GPUPDEDeformableRegistrationFilter);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  /** Types inherited from the GPUSuperclass */
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  using OutputImageType = typename GPUSuperclass::OutputImageType;
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  using FiniteDifferenceFunctionType = typename GPUSuperclass::FiniteDifferenceFunctionType;
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  /** GPUPDEDeformableRegistrationFilterFunction type. */
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  using GPUPDEDeformableRegistrationFunctionType =
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:    GPUPDEDeformableRegistrationFunction<FixedImageType, MovingImageType, DisplacementFieldType>;
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  /** Inherit some enums and type alias from the GPUSuperclass. */
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  static constexpr unsigned int ImageDimension = GPUSuperclass::ImageDimension;
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  /** Get OpenCL Kernel source as a string, creates a GetOpenCLSource method */
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  itkGetOpenCLSourceFromKernelMacro(GPUPDEDeformableRegistrationFilterKernel);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  GPUPDEDeformableRegistrationFilter();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  ~GPUPDEDeformableRegistrationFilter() override = default;
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  GPUSmoothVectorField(DisplacementFieldPointer         field,
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:                       typename GPUDataManager::Pointer GPUSmoothingKernels[],
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:                       int                              GPUSmoothingKernelSizes[]);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  typename GPUDataManager::Pointer m_GPUSmoothingKernels[ImageDimension]{};
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  typename GPUDataManager::Pointer m_UpdateFieldGPUSmoothingKernels[ImageDimension]{};
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  typename GPUDataManager::Pointer m_GPUImageSizes{};
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  /* GPU kernel handle for GPUSmoothDisplacementField */
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:  int m_SmoothDisplacementFieldGPUKernelHandle{};
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.h:#  include "itkGPUPDEDeformableRegistrationFilter.hxx"
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:#ifndef itkGPUDemonsRegistrationFunction_h
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:#define itkGPUDemonsRegistrationFunction_h
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:#include "itkGPUPDEDeformableRegistrationFunction.h"
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:#include "itkGPUReduction.h"
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h: * \class GPUDemonsRegistrationFunction
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h: * \ingroup ITKGPUPDEDeformableRegistration
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:/** Create a helper GPU Kernel class for GPUDemonsRegistrationFunction */
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:itkGPUKernelClassMacro(GPUDemonsRegistrationFunctionKernel);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:class ITK_TEMPLATE_EXPORT GPUDemonsRegistrationFunction
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  : public GPUPDEDeformableRegistrationFunction<TFixedImage, TMovingImage, TDisplacementField>
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUDemonsRegistrationFunction);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  using Self = GPUDemonsRegistrationFunction;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  using Superclass = GPUPDEDeformableRegistrationFunction<TFixedImage, TMovingImage, TDisplacementField>;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  itkOverrideGetNameOfClassMacro(GPUDemonsRegistrationFunction);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  /** GPU data pointer type. */
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  using GPUDataPointer = GPUDataManager::Pointer;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  /** Get OpenCL Kernel source as a string, creates a GetOpenCLSource method */
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  itkGetOpenCLSourceFromKernelMacro(GPUDemonsRegistrationFunctionKernel);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  /** Allocate GPU buffers for computing metric statistics
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  GPUAllocateMetricData(unsigned int numPixels) override;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  /** Release GPU buffers for computing metric statistics. */
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  GPUReleaseMetricData() override;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  GPUComputeUpdate(const DisplacementFieldTypePointer output, DisplacementFieldTypePointer update, void * gd) override;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  GPUDemonsRegistrationFunction();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  ~GPUDemonsRegistrationFunction() override = default;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  /* GPU kernel handle for GPUComputeUpdate */
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  int m_ComputeUpdateGPUKernelHandle{};
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  mutable GPUReduction<int>::Pointer   m_GPUPixelCounter{};
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  mutable GPUReduction<float>::Pointer m_GPUSquaredChange{};
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:  mutable GPUReduction<float>::Pointer m_GPUSquaredDifference{};
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.h:#  include "itkGPUDemonsRegistrationFunction.hxx"
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:#ifndef itkGPUPDEDeformableRegistrationFilter_hxx
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:#define itkGPUPDEDeformableRegistrationFilter_hxx
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:// #define NOT_REORDER_GPU_MEMORY
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:GPUPDEDeformableRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  GPUPDEDeformableRegistrationFilter()
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  // Build GPU program
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    itkExceptionMacro("GPUDenseFiniteDifferenceImageFilter supports 1/2/3D image.");
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  using GPUCodeType = const char *;
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  GPUCodeType GPUSource = GPUPDEDeformableRegistrationFilter::GetOpenCLSource();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  this->m_GPUKernelManager->LoadProgramFromString(GPUSource, defines.str().c_str());
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:#ifdef NOT_REORDER_GPU_MEMORY
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  m_SmoothDisplacementFieldGPUKernelHandle = this->m_GPUKernelManager->CreateKernel("SmoothingFilter");
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  m_SmoothDisplacementFieldGPUKernelHandle = this->m_GPUKernelManager->CreateKernel("SmoothingFilterReorder");
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:GPUPDEDeformableRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::PrintSelf(
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  GPUSuperclass::PrintSelf(os, indent);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  os << indent << "GPUSmoothingKernels: ";
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    os << indent.GetNextIndent() << m_GPUSmoothingKernels[d] << std::endl;
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  os << indent << "UpdateFieldGPUSmoothingKernels: ";
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    os << indent.GetNextIndent() << m_UpdateFieldGPUSmoothingKernels[d] << std::endl;
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  itkPrintSelfObjectMacro(GPUImageSizes);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  os << indent << "SmoothDisplacementFieldGPUKernelHandle: " << m_SmoothDisplacementFieldGPUKernelHandle << std::endl;
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:GPUPDEDeformableRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  auto * f = dynamic_cast<GPUPDEDeformableRegistrationFunctionType *>(this->GetDifferenceFunction().GetPointer());
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  this->GPUSuperclass::InitializeIteration();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:GPUPDEDeformableRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  typename GPUSuperclass::InputImageType::ConstPointer inputPtr = this->GetInput();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    this->GPUSuperclass::CopyInputToOutput();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    typename GPUSuperclass::PixelType zeros;
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    using GPUOutputImage = typename itk::GPUTraits<TDisplacementField>::Type;
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    typename GPUOutputImage::Pointer output = dynamic_cast<GPUOutputImage *>(this->GetOutput());
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    // copy the deformation output to gpu
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    output->GetGPUDataManager()->SetGPUDirtyFlag(true);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:GPUPDEDeformableRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    this->GPUSuperclass::GenerateOutputInformation();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:GPUPDEDeformableRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  // call the GPUSuperclass's implementation
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  GPUSuperclass::GenerateInputRequestedRegion();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:GPUPDEDeformableRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  // Call GPUFiniteDifferenceImageFilter::PostProcessOutput().
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  this->GPUSuperclass::PostProcessOutput();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    m_GPUSmoothingKernels[dir]->Initialize();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    m_UpdateFieldGPUSmoothingKernels[dir]->Initialize();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  m_GPUImageSizes->Initialize();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  auto * f = dynamic_cast<GPUPDEDeformableRegistrationFunctionType *>(this->GetDifferenceFunction().GetPointer());
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  f->GPUReleaseMetricData();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  // update the cpu buffer from gpu
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:GPUPDEDeformableRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::Initialize()
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  this->GPUSuperclass::Initialize();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  auto * f = dynamic_cast<GPUPDEDeformableRegistrationFunctionType *>(this->GetDifferenceFunction().GetPointer());
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  f->GPUAllocateMetricData(numPixels);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:GPUPDEDeformableRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  GPUSmoothVectorField(DisplacementFieldPointer         field,
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:                       typename GPUDataManager::Pointer GPUSmoothingKernels[],
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:                       int                              GPUSmoothingKernelSizes[])
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  using GPUBufferImage = typename itk::GPUTraits<TDisplacementField>::Type;
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  using GPUOutputImage = typename itk::GPUTraits<TDisplacementField>::Type;
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  typename GPUBufferImage::Pointer bfPtr = dynamic_cast<GPUBufferImage *>(m_TempField.GetPointer());
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  typename GPUOutputImage::Pointer otPtr = dynamic_cast<GPUOutputImage *>(field.GetPointer());
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  typename GPUOutputImage::SizeType outSize = otPtr->GetLargestPossibleRegion().GetSize();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    itkExceptionMacro("GPUSmoothDisplacementField supports 1/2/3D images.");
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  this->m_GPUKernelManager->GetKernelWorkGroupInfo(
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    m_SmoothDisplacementFieldGPUKernelHandle, CL_KERNEL_WORK_GROUP_SIZE, &kernelWorkGroupSize);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  this->m_GPUKernelManager->GetDeviceInfo(
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:      this->m_GPUKernelManager->SetKernelArgWithImage(
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:        m_SmoothDisplacementFieldGPUKernelHandle, argidx++, otPtr->GetGPUDataManager());
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:      this->m_GPUKernelManager->SetKernelArgWithImage(
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:        m_SmoothDisplacementFieldGPUKernelHandle, argidx++, bfPtr->GetGPUDataManager());
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:      this->m_GPUKernelManager->SetKernelArgWithImage(
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:        m_SmoothDisplacementFieldGPUKernelHandle, argidx++, bfPtr->GetGPUDataManager());
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:      this->m_GPUKernelManager->SetKernelArgWithImage(
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:        m_SmoothDisplacementFieldGPUKernelHandle, argidx++, otPtr->GetGPUDataManager());
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    this->m_GPUKernelManager->SetKernelArgWithImage(
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:      m_SmoothDisplacementFieldGPUKernelHandle, argidx++, m_GPUImageSizes);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    this->m_GPUKernelManager->SetKernelArg(
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:      m_SmoothDisplacementFieldGPUKernelHandle, argidx++, sizeof(int), &(ImageDim));
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    this->m_GPUKernelManager->SetKernelArgWithImage(
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:      m_SmoothDisplacementFieldGPUKernelHandle, argidx++, GPUSmoothingKernels[indir]);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    this->m_GPUKernelManager->SetKernelArg(
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:      m_SmoothDisplacementFieldGPUKernelHandle, argidx++, sizeof(int), &(GPUSmoothingKernelSizes[indir]));
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    this->m_GPUKernelManager->SetKernelArg(m_SmoothDisplacementFieldGPUKernelHandle, argidx++, sizeof(int), &(indir));
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    this->m_GPUKernelManager->SetKernelArg(m_SmoothDisplacementFieldGPUKernelHandle, argidx++, sizeof(int), &(outdir));
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    this->m_GPUKernelManager->SetKernelArg(m_SmoothDisplacementFieldGPUKernelHandle,
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:                                           sizeof(DeformationScalarType) * GPUSmoothingKernelSizes[indir],
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    this->m_GPUKernelManager->SetKernelArg(m_SmoothDisplacementFieldGPUKernelHandle,
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:                                             (localSize[indir] + GPUSmoothingKernelSizes[indir] - 1),
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    this->m_GPUKernelManager->LaunchKernel(m_SmoothDisplacementFieldGPUKernelHandle,
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:GPUPDEDeformableRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  this->GPUSmoothVectorField(this->GetDisplacementField(), this->m_GPUSmoothingKernels, this->m_SmoothingKernelSizes);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:GPUPDEDeformableRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  this->GPUSmoothVectorField(
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    this->GetUpdateBuffer(), this->m_UpdateFieldGPUSmoothingKernels, this->m_UpdateFieldSmoothingKernelSizes);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:GPUPDEDeformableRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  // Therefore, we will avoid the data copy to GPU at every
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    // convolution kernel buffer on GPU
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    m_GPUSmoothingKernels[dir] = GPUDataManager::New();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    m_GPUSmoothingKernels[dir]->SetBufferSize(sizeof(DeformationScalarType) * ksize);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    m_GPUSmoothingKernels[dir]->SetCPUBufferPointer(m_SmoothingKernels[dir]);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    m_GPUSmoothingKernels[dir]->SetBufferFlag(CL_MEM_READ_ONLY);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    m_GPUSmoothingKernels[dir]->Allocate();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    m_GPUSmoothingKernels[dir]->SetGPUDirtyFlag(true);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    // convolution kernel buffer on GPU
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    m_UpdateFieldGPUSmoothingKernels[dir] = GPUDataManager::New();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    m_UpdateFieldGPUSmoothingKernels[dir]->SetBufferSize(sizeof(DeformationScalarType) * ksize);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    m_UpdateFieldGPUSmoothingKernels[dir]->SetCPUBufferPointer(m_UpdateFieldSmoothingKernels[dir]);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    m_UpdateFieldGPUSmoothingKernels[dir]->SetBufferFlag(CL_MEM_READ_ONLY);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    m_UpdateFieldGPUSmoothingKernels[dir]->Allocate();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    m_UpdateFieldGPUSmoothingKernels[dir]->SetGPUDirtyFlag(true);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:    itkExceptionMacro("GPUSmoothDisplacementField supports 1/2/3D images.");
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  m_GPUImageSizes = GPUDataManager::New();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  m_GPUImageSizes->SetBufferSize(sizeof(int) * 3);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  m_GPUImageSizes->SetCPUBufferPointer(m_ImageSizes);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  m_GPUImageSizes->SetBufferFlag(CL_MEM_READ_ONLY);
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  m_GPUImageSizes->Allocate();
Modules/Registration/GPUPDEDeformable/include/itkGPUPDEDeformableRegistrationFilter.hxx:  m_GPUImageSizes->SetGPUDirtyFlag(true);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:#ifndef itkGPUDemonsRegistrationFilter_hxx
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:#define itkGPUDemonsRegistrationFilter_hxx
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:GPUDemonsRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:  GPUDemonsRegistrationFilter()
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:  typename GPUDemonsRegistrationFunctionType::Pointer drfp;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:  drfp = GPUDemonsRegistrationFunctionType::New();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:GPUDemonsRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::PrintSelf(
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:  GPUSuperclass::PrintSelf(os, indent);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:GPUDemonsRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::InitializeIteration()
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:  // call the GPUSuperclass  implementation
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:  GPUSuperclass::InitializeIteration();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:  auto * drfp = dynamic_cast<GPUDemonsRegistrationFunctionType *>(this->GetDifferenceFunction().GetPointer());
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:GPUDemonsRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::GetMetric() const
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:  auto * drfp = dynamic_cast<GPUDemonsRegistrationFunctionType *>(this->GetDifferenceFunction().GetPointer());
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:GPUDemonsRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:  auto * drfp = dynamic_cast<GPUDemonsRegistrationFunctionType *>(this->GetDifferenceFunction().GetPointer());
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:GPUDemonsRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:  auto * drfp = dynamic_cast<GPUDemonsRegistrationFunctionType *>(this->GetDifferenceFunction().GetPointer());
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:GPUDemonsRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>::ApplyUpdate(
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:  this->GPUSuperclass::ApplyUpdate(dt);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.hxx:  auto * drfp = dynamic_cast<GPUDemonsRegistrationFunctionType *>(this->GetDifferenceFunction().GetPointer());
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:#ifndef itkGPUDemonsRegistrationFunction_hxx
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:#define itkGPUDemonsRegistrationFunction_hxx
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:GPUDemonsRegistrationFunction<TFixedImage, TMovingImage, TDisplacementField>::GPUDemonsRegistrationFunction()
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  /*** Prepare GPU opencl program ***/
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUPixelCounter = nullptr;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUSquaredChange = nullptr;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUSquaredDifference = nullptr;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:    itkExceptionMacro("GPUDenseFiniteDifferenceImageFilter supports 1/2/3D image.");
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  const char * GPUSource = GPUDemonsRegistrationFunction::GetOpenCLSource();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  this->m_GPUKernelManager->LoadProgramFromString(GPUSource, defines.str().c_str());
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_ComputeUpdateGPUKernelHandle = this->m_GPUKernelManager->CreateKernel("ComputeUpdate");
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:GPUDemonsRegistrationFunction<TFixedImage, TMovingImage, TDisplacementField>::PrintSelf(std::ostream & os,
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  itkPrintSelfObjectMacro(GPUPixelCounter);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  itkPrintSelfObjectMacro(GPUSquaredChange);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  itkPrintSelfObjectMacro(GPUSquaredDifference);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:GPUDemonsRegistrationFunction<TFixedImage, TMovingImage, TDisplacementField>::SetIntensityDifferenceThreshold(
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:GPUDemonsRegistrationFunction<TFixedImage, TMovingImage, TDisplacementField>::GetIntensityDifferenceThreshold() const
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:GPUDemonsRegistrationFunction<TFixedImage, TMovingImage, TDisplacementField>::InitializeIteration()
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:GPUDemonsRegistrationFunction<TFixedImage, TMovingImage, TDisplacementField>::GPUAllocateMetricData(
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  // allocate gpu buffers for statistics
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  // if (m_GPUPixelCounter == (GPUReduction<int>::Pointer)nullptr)
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUPixelCounter = GPUReduction<int>::New();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUSquaredChange = GPUReduction<float>::New();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUSquaredDifference = GPUReduction<float>::New();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUPixelCounter->InitializeKernel(numPixels);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUSquaredChange->InitializeKernel(numPixels);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUSquaredDifference->InitializeKernel(numPixels);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUPixelCounter->AllocateGPUInputBuffer();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUSquaredChange->AllocateGPUInputBuffer();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUSquaredDifference->AllocateGPUInputBuffer();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:GPUDemonsRegistrationFunction<TFixedImage, TMovingImage, TDisplacementField>::GPUReleaseMetricData()
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUPixelCounter->ReleaseGPUInputBuffer();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUSquaredChange->ReleaseGPUInputBuffer();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUSquaredDifference->ReleaseGPUInputBuffer();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:GPUDemonsRegistrationFunction<TFixedImage, TMovingImage, TDisplacementField>::GPUComputeUpdate(
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  localSize[0] = localSize[1] = localSize[2] = OpenCLGetLocalBlockSize(ImageDim);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:    m_ComputeUpdateGPUKernelHandle, argidx++, fixedImage->GetGPUDataManager());
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:    m_ComputeUpdateGPUKernelHandle, argidx++, movingImage->GetGPUDataManager());
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:    m_ComputeUpdateGPUKernelHandle, argidx++, output->GetGPUDataManager());
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:    m_ComputeUpdateGPUKernelHandle, argidx++, update->GetGPUDataManager());
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:    m_ComputeUpdateGPUKernelHandle, argidx++, m_GPUPixelCounter->GetGPUDataManager());
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:    m_ComputeUpdateGPUKernelHandle, argidx++, m_GPUSquaredChange->GetGPUDataManager());
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:    m_ComputeUpdateGPUKernelHandle, argidx++, m_GPUSquaredDifference->GetGPUDataManager());
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  this->m_GPUKernelManager->SetKernelArg(m_ComputeUpdateGPUKernelHandle, argidx++, sizeof(float), &(normalizer));
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:    this->m_GPUKernelManager->SetKernelArg(m_ComputeUpdateGPUKernelHandle, argidx++, sizeof(int), &(imgSize[i]));
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  this->m_GPUKernelManager->LaunchKernel(
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:    m_ComputeUpdateGPUKernelHandle, static_cast<int>(DisplacementFieldType::ImageDimension), globalSize, localSize);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUPixelCounter->GPUGenerateData();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUSquaredChange->GPUGenerateData();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_GPUSquaredDifference->GPUGenerateData();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_SumOfSquaredDifference = m_GPUSquaredDifference->GetGPUResult();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_NumberOfPixelsProcessed = m_GPUPixelCounter->GetGPUResult();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:  m_SumOfSquaredChange = m_GPUSquaredChange->GetGPUResult();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:GPUDemonsRegistrationFunction<TFixedImage, TMovingImage, TDisplacementField>::ComputeUpdate(
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFunction.hxx:GPUDemonsRegistrationFunction<TFixedImage, TMovingImage, TDisplacementField>::ReleaseGlobalDataPointer(void * gd) const
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:#ifndef itkGPUDemonsRegistrationFilter_h
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:#define itkGPUDemonsRegistrationFilter_h
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:#include "itkOpenCLUtil.h"
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:#include "itkGPUPDEDeformableRegistrationFilter.h"
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:#include "itkGPUDemonsRegistrationFunction.h"
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:/** \class GPUDemonsRegistrationFilter
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h: * \brief Deformably register two images using the demons algorithm with GPU.
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h: * GPUDemonsRegistrationFilter implements the demons deformable algorithm that
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h: * for each iteration is computed in GPUDemonsRegistrationFunction.
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h: * \sa GPUDemonsRegistrationFunction
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h: * \ingroup ITKGPUPDEDeformableRegistration
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:class ITK_TEMPLATE_EXPORT GPUDemonsRegistrationFilter
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  : public GPUPDEDeformableRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUDemonsRegistrationFilter);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  using Self = GPUDemonsRegistrationFilter;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  using GPUSuperclass =
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:    GPUPDEDeformableRegistrationFilter<TFixedImage, TMovingImage, TDisplacementField, TParentImageFilter>;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  itkOverrideGetNameOfClassMacro(GPUDemonsRegistrationFilter);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  /** Inherit types from GPUSuperclass. */
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  using TimeStepType = typename GPUSuperclass::TimeStepType;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  using FixedImageType = typename GPUSuperclass::FixedImageType;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  using FixedImagePointer = typename GPUSuperclass::FixedImagePointer;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  using MovingImageType = typename GPUSuperclass::MovingImageType;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  using MovingImagePointer = typename GPUSuperclass::MovingImagePointer;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  using DisplacementFieldType = typename GPUSuperclass::DisplacementFieldType;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  using DisplacementFieldPointer = typename GPUSuperclass::DisplacementFieldPointer;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  using FiniteDifferenceFunctionType = typename GPUSuperclass::FiniteDifferenceFunctionType;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  /** GPUDemonsRegistrationFilterFunction type. */
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  using GPUDemonsRegistrationFunctionType =
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:    GPUDemonsRegistrationFunction<FixedImageType, MovingImageType, DisplacementFieldType>;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  GPUDemonsRegistrationFilter();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  ~GPUDemonsRegistrationFilter() override = default;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:/** \class GPUDemonsRegistrationFilterFactory
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h: * \brief Object Factory implementation for GPUDemonsRegistrationFilter
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h: * \ingroup ITKGPUPDEDeformableRegistration
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:class GPUDemonsRegistrationFilterFactory : public itk::ObjectFactoryBase
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUDemonsRegistrationFilterFactory);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  using Self = GPUDemonsRegistrationFilterFactory;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  using GPUSuperclass = ObjectFactoryBase;
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:    return "A Factory for GPUDemonsRegistrationFilter";
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  itkOverrideGetNameOfClassMacro(GPUDemonsRegistrationFilterFactory);
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:    auto factory = GPUDemonsRegistrationFilterFactory::New();
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:    using InputImageType = GPUImage<ipt, dm>;                                                             \
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:    using OutputImageType = GPUImage<opt, dm>;                                                            \
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:    using DisplacementFieldType = GPUImage<VectorPixelType, dm>;                                          \
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:      typeid(GPUDemonsRegistrationFilter<InputImageType, OutputImageType, DisplacementFieldType>).name(), \
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:      "GPU Demons Registration Filter Override",                                                          \
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:        GPUDemonsRegistrationFilter<InputImageType, OutputImageType, DisplacementFieldType>>::New());     \
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:  GPUDemonsRegistrationFilterFactory()
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:    if (IsGPUAvailable())
Modules/Registration/GPUPDEDeformable/include/itkGPUDemonsRegistrationFilter.h:#  include "itkGPUDemonsRegistrationFilter.hxx"
Modules/Registration/GPUPDEDeformable/CMakeLists.txt:project(ITKGPUPDEDeformableRegistration)
Modules/Registration/GPUPDEDeformable/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Registration/GPUPDEDeformable/CMakeLists.txt:  set(ITKGPUPDEDeformableRegistration_LIBRARIES ITKGPUPDEDeformableRegistration)
Modules/Registration/GPUPDEDeformable/CMakeLists.txt:  set(ITKGPUPDEDeformableRegistration_SYSTEM_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
Modules/Registration/GPUPDEDeformable/CMakeLists.txt:  set(ITKGPUPDEDeformableRegistration_SYSTEM_LIBRARY_DIRS ${OPENCL_LIBRARIES})
Modules/Registration/GPUPDEDeformable/src/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Registration/GPUPDEDeformable/src/CMakeLists.txt:  set(ITKGPUPDEDeformableRegistration_SRCS)
Modules/Registration/GPUPDEDeformable/src/CMakeLists.txt:  set(ITKGPUPDEDeformableRegistration_Kernels GPUDemonsRegistrationFunction.cl GPUPDEDeformableRegistrationFilter.cl)
Modules/Registration/GPUPDEDeformable/src/CMakeLists.txt:  write_gpu_kernels("${ITKGPUPDEDeformableRegistration_Kernels}" ITKGPUPDEDeformableRegistration_SRCS)
Modules/Registration/GPUPDEDeformable/src/CMakeLists.txt:  itk_module_add_library(ITKGPUPDEDeformableRegistration ${ITKGPUPDEDeformableRegistration_SRCS})
Modules/Registration/GPUPDEDeformable/src/CMakeLists.txt:  target_link_libraries(ITKGPUPDEDeformableRegistration LINK_PUBLIC ${OPENCL_LIBRARIES})
Modules/Registration/GPUPDEDeformable/src/GPUPDEDeformableRegistrationFilter.cl:       execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
Modules/Registration/GPUPDEDeformable/src/GPUPDEDeformableRegistrationFilter.cl:       execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
Modules/Registration/GPUPDEDeformable/src/GPUDemonsRegistrationFunction.cl:     execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
Modules/ThirdParty/KWSys/src/KWSys/SystemInformation.hxx.in:  long long GetProcMemoryAvailable(const char* hostLimitEnvVarName = nullptr,
Modules/ThirdParty/KWSys/src/KWSys/SystemInformation.hxx.in:  long long GetProcMemoryUsed();
Modules/ThirdParty/KWSys/src/KWSys/ProcessUNIX.c:  if (sigprocmask(SIG_BLOCK, &mask, &old_mask) < 0) {
Modules/ThirdParty/KWSys/src/KWSys/ProcessUNIX.c:    sigprocmask(SIG_SETMASK, &old_mask, 0);
Modules/ThirdParty/KWSys/src/KWSys/ProcessUNIX.c:    sigprocmask(SIG_SETMASK, &old_mask, 0);
Modules/ThirdParty/KWSys/src/KWSys/ProcessUNIX.c:    sigprocmask(SIG_SETMASK, &old_mask, 0);
Modules/ThirdParty/KWSys/src/KWSys/ProcessUNIX.c:  if (sigprocmask(SIG_SETMASK, &old_mask, 0) < 0) {
Modules/ThirdParty/KWSys/src/KWSys/ProcessUNIX.c:  if (sigprocmask(SIG_BLOCK, &mask, &old_mask) < 0) {
Modules/ThirdParty/KWSys/src/KWSys/ProcessUNIX.c:  sigprocmask(SIG_SETMASK, &old_mask, 0);
Modules/ThirdParty/KWSys/src/KWSys/ProcessUNIX.c:     TODO: sigprocmask is undefined for threaded apps.  See
Modules/ThirdParty/KWSys/src/KWSys/ProcessUNIX.c:  sigprocmask(SIG_BLOCK, &newset, &oldset);
Modules/ThirdParty/KWSys/src/KWSys/ProcessUNIX.c:  sigprocmask(SIG_SETMASK, &oldset, 0);
Modules/ThirdParty/KWSys/src/KWSys/ProcessUNIX.c:        sigprocmask(SIG_UNBLOCK, &unblockSet, 0);
Modules/ThirdParty/KWSys/src/KWSys/SystemInformation.cxx:  long long GetProcMemoryAvailable(const char* hostLimitEnvVarName,
Modules/ThirdParty/KWSys/src/KWSys/SystemInformation.cxx:  long long GetProcMemoryUsed();
Modules/ThirdParty/KWSys/src/KWSys/SystemInformation.cxx:      << this->GetProcMemoryAvailable(hostLimitEnvVarName, procLimitEnvVarName)
Modules/ThirdParty/KWSys/src/KWSys/SystemInformation.cxx:long long SystemInformation::GetProcMemoryAvailable(
Modules/ThirdParty/KWSys/src/KWSys/SystemInformation.cxx:  return this->Implementation->GetProcMemoryAvailable(hostLimitEnvVarName,
Modules/ThirdParty/KWSys/src/KWSys/SystemInformation.cxx:long long SystemInformation::GetProcMemoryUsed()
Modules/ThirdParty/KWSys/src/KWSys/SystemInformation.cxx:  return this->Implementation->GetProcMemoryUsed();
Modules/ThirdParty/KWSys/src/KWSys/SystemInformation.cxx:long long SystemInformationImplementation::GetProcMemoryAvailable(
Modules/ThirdParty/KWSys/src/KWSys/SystemInformation.cxx:long long SystemInformationImplementation::GetProcMemoryUsed()
Modules/ThirdParty/KWSys/src/KWSys/SystemInformation.cxx:    std::bitset<std::numeric_limits<ULONG_PTR>::digits> ProcMask(
Modules/ThirdParty/KWSys/src/KWSys/SystemInformation.cxx:    unsigned int count = (unsigned int)ProcMask.count();
Modules/ThirdParty/KWSys/src/KWSys/SystemTools.cxx:#  include <csignal> /* sigprocmask */
Modules/ThirdParty/KWSys/src/KWSys/testSystemInformation.cxx:  printMethod3(info, GetProcMemoryAvailable("KWSHL", "KWSPL"), "KiB");
Modules/ThirdParty/KWSys/src/KWSys/testSystemInformation.cxx:  printMethod3(info, GetProcMemoryUsed(), "KiB");
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:#define  DECLAREContigPutFunc(name) \
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(put8bitcmaptile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(put4bitcmaptile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(put2bitcmaptile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(put1bitcmaptile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putgreytile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putagreytile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(put16bitbwtile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(put1bitbwtile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(put2bitbwtile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(put4bitbwtile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putRGBcontig8bittile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putRGBAAcontig8bittile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putRGBUAcontig8bittile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putRGBcontig16bittile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putRGBAAcontig16bittile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putRGBUAcontig16bittile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putRGBcontig8bitCMYKtile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putRGBcontig8bitCMYKMaptile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putcontig8bitCIELab)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putcontig8bitYCbCr44tile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putcontig8bitYCbCr42tile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putcontig8bitYCbCr41tile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putcontig8bitYCbCr22tile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putcontig8bitYCbCr21tile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putcontig8bitYCbCr12tile)
Modules/ThirdParty/TIFF/src/itktiff/tif_getimage.c:DECLAREContigPutFunc(putcontig8bitYCbCr11tile)
Modules/ThirdParty/GDCM/src/gdcm/Utilities/socketxx/socket++/sig.h:#ifndef sigprocmask
Modules/ThirdParty/GDCM/src/gdcm/Utilities/socketxx/socket++/sig.h:  #define sigprocmask(a, b, c)
Modules/ThirdParty/GDCM/src/gdcm/Utilities/socketxx/socket++/sig.cpp:  if (sigprocmask (SIG_BLOCK, &s, 0) == -1) throw sigerr();
Modules/ThirdParty/GDCM/src/gdcm/Utilities/socketxx/socket++/sig.cpp:  if (sigprocmask (SIG_UNBLOCK, &s, 0) == -1) throw sigerr();
Modules/ThirdParty/GDCM/src/gdcm/Source/DataStructureAndEncodingDefinition/gdcmCSAHeader.cxx:ECGPulsing:     MlOff
Modules/ThirdParty/HDF5/src/itkhdf5/config/cmake_ext_mod/ConfigureChecks.cmake:CHECK_FUNCTION_EXISTS (sigprocmask       ${HDF_PREFIX}_HAVE_SIGPROCMASK)
Modules/ThirdParty/HDF5/src/itkhdf5/config/cmake/H5pubconf.h.in:/* Define to 1 if you have the `sigprocmask' function. */
Modules/ThirdParty/HDF5/src/itkhdf5/config/cmake/H5pubconf.h.in:#cmakedefine H5_HAVE_SIGPROCMASK @H5_HAVE_SIGPROCMASK@
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5detect.c: * do. If sigsetjmp/siglongjmp are not supported, need to use sigprocmask to
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5detect.c:#if !defined(H5HAVE_SIGJMP) && defined(H5_HAVE_SIGPROCMASK)
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5detect.c:    /* Use sigprocmask to unblock the signal if sigsetjmp/siglongjmp are not */
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5detect.c:    HDsigprocmask(SIG_UNBLOCK, &set, NULL);
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5detect.c:#if !defined(H5HAVE_SIGJMP) && defined(H5_HAVE_SIGPROCMASK)
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5detect.c:    /* Use sigprocmask to unblock the signal if sigsetjmp/siglongjmp are not */
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5detect.c:    HDsigprocmask(SIG_UNBLOCK, &set, NULL);
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5detect.c:#if !defined(H5HAVE_SIGJMP) && defined(H5_HAVE_SIGPROCMASK)
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5detect.c:    /* Use sigprocmask to unblock the signal if sigsetjmp/siglongjmp are not */
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5detect.c:    HDsigprocmask(SIG_UNBLOCK, &set, NULL);
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5detect.c:#ifdef H5_HAVE_SIGPROCMASK
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5detect.c:    fprintf(rawoutstream, "/* sigprocmask() support: yes */\n");
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5detect.c:    fprintf(rawoutstream, "/* sigprocmask() support: no */\n");
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5Rpublic.h:#include "H5Gpublic.h"
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5Rpublic.h: *          following valid object type values (defined in H5Gpublic.h):
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5Rpublic.h: *          \snippet H5Gpublic.h H5G_obj_t_snip
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5config.h.in:/* Define to 1 if you have the `sigprocmask' function. */
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5config.h.in:#undef HAVE_SIGPROCMASK
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5Rdeprec.c: * Return:      Success:    Object type (as defined in H5Gpublic.h)
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5Gprivate.h:#include "H5Gpublic.h"
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5private.h:#ifndef HDsigprocmask
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5private.h:#define HDsigprocmask(H, S, O) sigprocmask(H, S, O)
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5private.h:#endif /* HDsigprocmask */
Modules/ThirdParty/HDF5/src/itkhdf5/src/hdf5.h:#include "H5Gpublic.h"  /* Groups                                   */
Modules/ThirdParty/HDF5/src/itkhdf5/src/CMakeLists.txt:    ${HDF5_SRC_DIR}/H5Gpublic.h
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5Gpublic.h: * Created:             H5Gpublic.h
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5Gpublic.h:#ifndef H5Gpublic_H
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5Gpublic.h:#define H5Gpublic_H
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5Gpublic.h: *          \p ginfo is an H5G_info_t struct and is defined (in H5Gpublic.h)
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5Gpublic.h: *          \p ginfo is an H5G_info_t struct and is defined (in H5Gpublic.h)
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5Gpublic.h: *          H5Gpublic.h):
Modules/ThirdParty/HDF5/src/itkhdf5/src/H5Gpublic.h:#endif /* H5Gpublic_H */
Modules/ThirdParty/MetaIO/src/MetaIO/src/metaImage.cxx:#  include <csignal> /* sigprocmask */
Modules/ThirdParty/MINC/src/libminc/libsrc2/minc_compat2.h:#define ncclose miclose
Modules/ThirdParty/MINC/src/libminc/libsrc/netcdf_convenience.c:#undef ncclose
Modules/ThirdParty/MINC/src/libminc/libsrc/netcdf_convenience.c:      (void) ncclose(status);
Modules/ThirdParty/MINC/src/libminc/libsrc/netcdf_convenience.c:@DESCRIPTION: A wrapper for routine ncclose, allowing future enhancements.
Modules/ThirdParty/MINC/src/libminc/libsrc/netcdf_convenience.c:       status = ncclose(cdfid);
Modules/ThirdParty/MINC/src/libminc/libsrc/netcdf_convenience.c:   status = ncclose(cdfid);
Modules/ThirdParty/MINC/src/libminc/libsrc/minc_compat.h:#define ncclose miclose
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/Core:// We need cuda_runtime.h/hip_runtime.h to ensure that
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/Core:#if defined(EIGEN_CUDACC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/Core:  #include <cuda_runtime.h>
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/Core:#if EIGEN_COMP_ICC && defined(EIGEN_GPU_COMPILE_PHASE) \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/Core:#if defined(EIGEN_HAS_CUDA_FP16) || defined(EIGEN_HAS_HIP_FP16)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/Core:  #define EIGEN_HAS_GPU_FP16
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/Core:#if defined(EIGEN_HAS_CUDA_BF16) || defined(EIGEN_HAS_HIP_BF16)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/Core:  #define EIGEN_HAS_GPU_BF16
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/Core:#if defined EIGEN_VECTORIZE_GPU
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/Core:  #include "src/Core/arch/GPU/PacketMath.h"
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/Core:  #include "src/Core/arch/GPU/MathFunctions.h"
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/Core:  #include "src/Core/arch/GPU/TypeCasting.h"
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/Core:// Specialized functors for GPU.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/Core:#ifdef EIGEN_GPUCC
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/Core:#include "src/Core/arch/GPU/Complex.h"
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/AssignEvaluator.h:#ifndef EIGEN_GPU_COMPILE_PHASE
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/Reverse.h://reverse const overload moved DenseBase.h due to a CUDA compiler bug
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Memory.h:#if ((EIGEN_COMP_GNUC) || __has_feature(cxx_thread_local) || EIGEN_COMP_MSVC >= 1900) && !defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Memory.h:#if ! defined EIGEN_ALLOCA && ! defined EIGEN_GPU_COMPILE_PHASE
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Assert.h://  - we are not compiling for GPU, or
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Assert.h://  - gpu debugging is enabled.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Assert.h:#if !defined(EIGEN_NO_DEBUG) && (!defined(EIGEN_GPU_COMPILE_PHASE) || !defined(EIGEN_NO_DEBUG_GPU))
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Assert.h://   - nvcc+gcc supports __builtin_FILE() on host, and on device after CUDA 11.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Assert.h://   - nvcc+msvc supports __builtin_FILE() only after CUDA 11.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Assert.h:#if (EIGEN_HAS_BUILTIN(__builtin_FILE) && (EIGEN_COMP_CLANG || !defined(EIGEN_CUDA_ARCH))) || \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Assert.h:    (EIGEN_GNUC_STRICT_AT_LEAST(5, 0, 0) && (EIGEN_COMP_NVCC >= 110000 || !defined(EIGEN_CUDA_ARCH))) || \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Assert.h:#ifdef EIGEN_GPU_COMPILE_PHASE
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Assert.h:    // GPU device code doesn't allow stderr or abort, so use printf and raise an
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Assert.h:#else  // EIGEN_GPU_COMPILE_PHASE
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Assert.h:#endif  // EIGEN_GPU_COMPILE_PHASE
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Serializer.h:// the CPU and GPU.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/EmulateArray.h:// CUDA doesn't support the STL containers, so we use our own instead.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/EmulateArray.h:#if defined(EIGEN_GPUCC) || defined(EIGEN_AVOID_STL_ARRAY)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/EmulateArray.h:#if !defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/EmulateArray.h:// The compiler supports c++11, and we're not targeting cuda: use std::array as Eigen::array
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/DisableStupidWarnings.h:  // MSVC 14.16 (required by CUDA 9.*) does not support the _Pragma keyword, so
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Meta.h: #if defined(EIGEN_CUDA_ARCH)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Meta.h:// GPU devices treat `long double` as `double`.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Meta.h:#ifndef EIGEN_GPU_COMPILE_PHASE
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Meta.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Meta.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) || (!defined(EIGEN_CUDA_ARCH) && defined(EIGEN_CONSTEXPR_ARE_DEVICE_FUNC))
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Meta.h:#if !defined(EIGEN_GPU_COMPILE_PHASE) || (!defined(EIGEN_CUDA_ARCH) && defined(EIGEN_CONSTEXPR_ARE_DEVICE_FUNC))
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:  #define EIGEN_COMP_NVCC  ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#elif defined(__CUDACC_VER__)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:  #define EIGEN_COMP_NVCC __CUDACC_VER__
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:// Detect GPU compilers and architectures
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:// Note that this also makes EIGEN_CUDACC and EIGEN_HIPCC mutually exclusive
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#if defined(__CUDACC__) && !defined(EIGEN_NO_CUDA) && !defined(__SYCL_DEVICE_ONLY__)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:  // Means the compiler is either nvcc or clang with CUDA enabled
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDACC __CUDACC__
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#if defined(__CUDA_ARCH__) && !defined(EIGEN_NO_CUDA) && !defined(__SYCL_DEVICE_ONLY__)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDA_ARCH __CUDA_ARCH__
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#include <cuda.h>
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDA_SDK_VER (CUDA_VERSION * 10)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:  #define EIGEN_CUDA_SDK_VER 0
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:  // Means the compiler is HIPCC (analogous to EIGEN_CUDACC, but for HIP)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:    // analogous to EIGEN_CUDA_ARCH, but for HIP
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:  // For HIP (ROCm 3.5 and higher), we need to explicitly set the launch_bounds attribute
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:  // specified. This results in failures on the HIP platform, for cases when a GPU kernel
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:  // couple of ROCm releases (compiler will go back to using 1024 value as the default)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:// Unify CUDA/HIPCC
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC) || defined(EIGEN_HIPCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:// If either EIGEN_CUDACC or EIGEN_HIPCC is defined, then define EIGEN_GPUCC
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#define EIGEN_GPUCC
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:// EIGEN_CUDACC implies the CUDA compiler and is used to tweak Eigen code for use in CUDA kernels
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:// In most cases the same tweaks are required to the Eigen code to enable in both the HIP and CUDA kernels.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_CUDACC) || defined(EIGEN_HIPCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:// For cases where the tweak is specific to CUDA, the code should be guarded with
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_CUDACC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIP_DEVICE_COMPILE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:// If either EIGEN_CUDA_ARCH or EIGEN_HIP_DEVICE_COMPILE is defined, then define EIGEN_GPU_COMPILE_PHASE
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#define EIGEN_GPU_COMPILE_PHASE
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:// GPU compilers (HIPCC, NVCC) typically do two passes over the source code,
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h://   + another to compile the source for the "device" (ie. GPU)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:// EIGEN_CUDA_ARCH implies the device compilation phase in CUDA
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:// In most cases, the "host" / "device" specific code is the same for both HIP and CUDA
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h://       #if defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h://       #if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIP_DEVICE_COMPILE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:// For cases where the tweak is specific to CUDA, the code should be guarded with
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h://      #if defined(EIGEN_CUDA_ARCH)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:    #if EIGEN_ARCH_ARM64 && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:    #if EIGEN_ARCH_ARM64 && defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) && !defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#if !defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_CUDACC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:    #ifdef __CUDACC_RELAXED_CONSTEXPR__
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:  #elif defined(__clang__) && defined(__CUDA__) && __has_feature(cxx_relaxed_constexpr)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#if (EIGEN_COMP_MSVC || EIGEN_COMP_ICC) && !defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:// GPU stuff
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:// Disable some features when compiling with GPU compilers (SYCL/HIPCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#if defined(SYCL_DEVICE_ONLY) || defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:// All functions callable from CUDA/HIP code must be qualified with __device__
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#elif defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#if defined(EIGEN_NO_DEBUG) || (defined(EIGEN_GPU_COMPILE_PHASE) && defined(EIGEN_NO_DEBUG_GPU))
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:// When compiling CUDA/HIP device code with NVCC or HIPCC
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#if (defined(EIGEN_CUDA_ARCH) && defined(__NVCC__)) || defined(EIGEN_HIP_DEVICE_COMPILE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(EIGEN_CUDA_ARCH) && !defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_USE_SYCL) && !defined(EIGEN_HIP_DEVICE_COMPILE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/Macros.h:#  if defined(EIGEN_CUDA_ARCH)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/MaxSizeVector.h:  * std::vector is not an option (e.g. on GPU or when compiling using
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/ConfigureVectorization.h:#if (defined EIGEN_CUDACC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/ConfigureVectorization.h:  #if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/ConfigureVectorization.h:    // GPU code is always vectorized and requires memory alignment for
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/ConfigureVectorization.h:#if !(defined(EIGEN_DONT_VECTORIZE) || defined(EIGEN_GPUCC))
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/ConfigureVectorization.h:#if defined(__F16C__) && !defined(EIGEN_GPUCC) && (!EIGEN_COMP_CLANG_STRICT || EIGEN_CLANG_STRICT_AT_LEAST(3,8,0))
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/ConfigureVectorization.h:#if defined EIGEN_CUDACC
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/ConfigureVectorization.h:  #define EIGEN_VECTORIZE_GPU
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/ConfigureVectorization.h:  #if EIGEN_CUDA_SDK_VER >= 70500
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/ConfigureVectorization.h:    #define EIGEN_HAS_CUDA_FP16
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/ConfigureVectorization.h:#if defined(EIGEN_HAS_CUDA_FP16)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/ConfigureVectorization.h:  #include <cuda_runtime_api.h>
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/ConfigureVectorization.h:  #include <cuda_fp16.h>
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/util/ConfigureVectorization.h:  #define EIGEN_VECTORIZE_GPU
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:// GPU, and correctly handles special cases (unlike MSVC).
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:    // There is no official ::arg on device in CUDA/HIP, so we always need to use std::arg.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if (!defined(EIGEN_GPUCC) || defined(EIGEN_CONSTEXPR_ARE_DEVICE_FUNC))
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#ifndef EIGEN_GPU_COMPILE_PHASE
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#ifndef EIGEN_GPU_COMPILE_PHASE
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:// HIP and CUDA do not support long double.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#ifndef EIGEN_GPU_COMPILE_PHASE
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if (!defined(EIGEN_GPUCC)) && EIGEN_FAST_MATH && !defined(SYCL_DEVICE_ONLY)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/MathFunctions.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/functors/UnaryFunctors.h:#ifdef EIGEN_GPU_CC
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/functors/UnaryFunctors.h:#endif  // #ifndef EIGEN_GPU_COMPILE_PHASE
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/functors/AssignmentFunctors.h:#ifdef EIGEN_GPUCC
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/functors/AssignmentFunctors.h:    // FIXME is there some kind of cuda::swap?
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/functors/BinaryFunctors.h:#ifndef EIGEN_GPU_COMPILE_PHASE
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/VectorwiseOp.h:    // isVertical*Factor+isHorizontal instead of (isVertical?Factor:1) to handle CUDA bug with ternary operator
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/VectorwiseOp.h://const colwise moved to DenseBase.h due to CUDA compiler bug
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/VectorwiseOp.h://const rowwise moved to DenseBase.h due to CUDA compiler bug
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/GeneralProduct.h:    #ifndef EIGEN_GPU_COMPILE_PHASE
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/GenericPacketMath.h:#elif defined(EIGEN_CUDA_ARCH)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/GenericPacketMath.h:// Eigen+CUDA does not support complexes.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/GenericPacketMath.h:#if !defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/ProductEvaluators.h:#ifndef EIGEN_GPUCC
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/ProductEvaluators.h:#ifndef EIGEN_GPUCC
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/DenseBase.h:    //Code moved here due to a CUDA compiler bug
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/NumTraits.h:  // Load src into registers first. This allows the memcpy to be elided by CUDA.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/NumTraits.h:// GPU devices treat `long double` as `double`.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/NumTraits.h:#ifndef EIGEN_GPU_COMPILE_PHASE
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#ifndef EIGEN_PACKET_MATH_GPU_H
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#define EIGEN_PACKET_MATH_GPU_H
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_HIP_DEVICE_COMPILE) || (defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 350)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#define EIGEN_GPU_HAS_LDG 1
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if (defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#define EIGEN_CUDA_HAS_FP16_ARITHMETIC 1
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_HIP_DEVICE_COMPILE) || defined(EIGEN_CUDA_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#define EIGEN_GPU_HAS_FP16_ARITHMETIC 1
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:// Make sure this is only available when targeting a GPU: we don't want to
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:// We need to distinguish ‘clang as the CUDA compiler’ from ‘clang as the host compiler,
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIPCC) || (defined(EIGEN_CUDACC) && EIGEN_COMP_CLANG && !EIGEN_COMP_NVCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#endif // defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIPCC) || (defined(EIGEN_CUDACC) && EIGEN_COMP_CLANG && !EIGEN_COMP_NVCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#endif // defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:// Half-packet functions are not available on the host for CUDA 9.0-9.2, only
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if (defined(EIGEN_HAS_CUDA_FP16) || defined(EIGEN_HAS_HIP_FP16)) && defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined(EIGEN_CUDA_HAS_FP16_ARITHMETIC)) || \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_LDG)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#elif defined(EIGEN_CUDA_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_CUDA_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:// the implementation of GPU half reduction.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#if defined(EIGEN_GPU_HAS_FP16_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#endif // (defined(EIGEN_HAS_CUDA_FP16) || defined(EIGEN_HAS_HIP_FP16)) && defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#undef EIGEN_GPU_HAS_LDG
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#undef EIGEN_CUDA_HAS_FP16_ARITHMETIC
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#undef EIGEN_GPU_HAS_FP16_ARITHMETIC
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/PacketMath.h:#endif // EIGEN_PACKET_MATH_GPU_H
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/MathFunctions.h:#ifndef EIGEN_MATH_FUNCTIONS_GPU_H
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/MathFunctions.h:#define EIGEN_MATH_FUNCTIONS_GPU_H
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/MathFunctions.h:#if defined(EIGEN_GPUCC) && defined(EIGEN_USE_GPU)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/MathFunctions.h:#endif // EIGEN_MATH_FUNCTIONS_GPU_H
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/Tuple.h:#ifndef EIGEN_TUPLE_GPU
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/Tuple.h:#define EIGEN_TUPLE_GPU
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/Tuple.h:#endif  // EIGEN_TUPLE_GPU
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/TypeCasting.h:#ifndef EIGEN_TYPE_CASTING_GPU_H
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/TypeCasting.h:#define EIGEN_TYPE_CASTING_GPU_H
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/TypeCasting.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/TypeCasting.h:#endif // EIGEN_TYPE_CASTING_GPU_H
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/Complex.h:#ifndef EIGEN_COMPLEX_GPU_H
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/Complex.h:#define EIGEN_COMPLEX_GPU_H
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/Complex.h:// operators and functors for complex types when building for CUDA to enable
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/Complex.h:#if defined(EIGEN_GPUCC) && defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/Complex.h:#endif  // EIGEN_GPUCC && EIGEN_GPU_COMPILE_PHASE
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/GPU/Complex.h:#endif  // EIGEN_COMPLEX_GPU_H
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/SYCL/MathFunctions.h:// Make sure this is only available when targeting a GPU: we don't want to
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/SYCL/InteropHeaders.h:// Make sure this is only available when targeting a GPU: we don't want to
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/HIP/hcc/math_constants.h: *  HIP equivalent of the CUDA header of the same name
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// Standard 16-bit float type, mostly useful for GPUs. Defines a new
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// type Eigen::half (inheriting either from CUDA's or HIP's __half struct) with
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// to disk and the likes), but fast on GPUs.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16) || defined(EIGEN_HAS_ARM64_FP16_SCALAR_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// When compiling with GPU support, the "__half_raw" base class as well as
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// some other routines are defined in the GPU compiler header files
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// (cuda_fp16.h, hip_fp16.h), and they are not tagged constexpr
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// GPU support. Hence the need to disable EIGEN_CONSTEXPR when building
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// Eigen with GPU support
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// This is required because of a quirk in the way TensorFlow GPU builds are done.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// When compiling TensorFlow source code with GPU support, files that
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h://  * contain GPU kernels (i.e. *.cu.cc files) are compiled via hipcc
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h://  * do not contain GPU kernels ( i.e. *.cc files) are compiled via gcc (typically)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if !defined(EIGEN_HAS_GPU_FP16) || !defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// Make our own __half_raw definition that is similar to CUDA's.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_GPU_FP16) && !defined(EIGEN_GPU_COMPILE_PHASE))
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#elif defined(EIGEN_HAS_CUDA_FP16)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:  #if EIGEN_CUDA_SDK_VER < 90000
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:    // In CUDA < 9.0, __half is the equivalent of CUDA 9's __half_raw
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:  #endif // defined(EIGEN_HAS_CUDA_FP16)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h: #elif defined(EIGEN_HAS_CUDA_FP16)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:  #if EIGEN_CUDA_SDK_VER >= 90000
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if !defined(EIGEN_HAS_GPU_FP16) || !defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:  // * when compiling without GPU support enabled
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:  // * during host compile phase when compiling with GPU support enabled
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#elif defined(EIGEN_HAS_CUDA_FP16)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:  // Note that EIGEN_CUDA_SDK_VER is set to 0 even when compiling with HIP, so
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:  // (EIGEN_CUDA_SDK_VER < 90000) is true even for HIP!  So keeping this within
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:  // #if defined(EIGEN_HAS_CUDA_FP16) is needed
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:  #if defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER < 90000
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h: #elif defined(EIGEN_HAS_CUDA_FP16)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:  #if defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER >= 90000
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16) && !defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:     EIGEN_CUDA_ARCH >= 530) ||                                  \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// fp16 type since GPU halfs are rather different from native CPU halfs.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// TODO: Rename to something like EIGEN_HAS_NATIVE_GPU_FP16
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER >= 90000
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_CUDA_SDK_VER) && EIGEN_CUDA_SDK_VER >= 90000
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_ARM64_FP16_SCALAR_ARITHMETIC) && !defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// We need to distinguish ‘clang as the CUDA compiler’ from ‘clang as the host compiler,
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if EIGEN_COMP_CLANG && defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_HAS_NATIVE_FP16)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// Definitions for CPUs and older HIP+CUDA, mostly working through conversion
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if EIGEN_COMP_CLANG && defined(EIGEN_GPUCC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// Conversion routines, including fallbacks for the host or older CUDA.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:  // Fortunately, since we need to disable EIGEN_CONSTEXPR for GPU anyway, we can get out
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:  // of this catch22 by having separate bodies for GPU / non GPU
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:  // HIP/CUDA/Default have a member 'x' of type uint16_t.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530) || \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && EIGEN_CUDA_SDK_VER >= 80000 && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 530) || \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 300) || \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if (EIGEN_CUDA_SDK_VER >= 80000 && defined EIGEN_CUDA_ARCH && EIGEN_CUDA_ARCH >= 300) || \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 530) || \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_GPU_FP16) || defined(EIGEN_HAS_ARM64_FP16_SCALAR_ARITHMETIC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// The __shfl* functions are only valid on HIP or _CUDA_ARCH_ >= 300.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h://   CUDA defines them for (__CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__))
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// HIP and CUDA prior to SDK 9.0 define
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:// CUDA since 9.0 deprecates those and instead defines
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_CUDACC) && (!defined(EIGEN_CUDA_ARCH) || EIGEN_CUDA_ARCH >= 300)) \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if defined(EIGEN_HAS_CUDA_FP16) && EIGEN_CUDA_SDK_VER >= 90000
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#else // HIP or CUDA SDK < 9.0
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#endif // HIP vs CUDA
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_CUDACC) && (!defined(EIGEN_CUDA_ARCH) || EIGEN_CUDA_ARCH >= 350)) \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/Half.h:#if (defined(EIGEN_HAS_CUDA_FP16) && defined(EIGEN_CUDA_ARCH) && EIGEN_CUDA_ARCH >= 300) || \
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/BFloat16.h:// When compiling with GPU support, the "hip_bfloat16" base class as well as
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/BFloat16.h:// some other routines are defined in the GPU compiler header files
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/BFloat16.h:// GPU support. Hence the need to disable EIGEN_CONSTEXPR when building
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/BFloat16.h:// Eigen with GPU support
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/BFloat16.h:// Only use HIP GPU bf16 in kernels
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/BFloat16.h:#if defined(EIGEN_HAS_HIP_BF16) && defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/BFloat16.h:#if defined(EIGEN_HAS_HIP_BF16) && !defined(EIGEN_GPU_COMPILE_PHASE)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/BFloat16.h:// We need to distinguish ‘clang as the CUDA compiler’ from ‘clang as the host compiler,
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/BFloat16.h:#if EIGEN_COMP_CLANG && defined(EIGEN_CUDACC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/BFloat16.h:#if (defined(EIGEN_HAS_GPU_BF16) && defined(EIGEN_HAS_NATIVE_BF16))
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/BFloat16.h:#if EIGEN_COMP_CLANG && defined(EIGEN_CUDACC)
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/BFloat16.h:// The __shfl* functions are only valid on HIP or _CUDA_ARCH_ >= 300.
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/BFloat16.h://   CUDA defines them for (__CUDA_ARCH__ >= 300 || !defined(__CUDA_ARCH__))
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/BFloat16.h:// HIP and CUDA prior to SDK 9.0 define
Modules/ThirdParty/Eigen3/src/itkeigen/Eigen/src/Core/arch/Default/BFloat16.h:// CUDA since 9.0 deprecates those and instead defines
Modules/ThirdParty/Eigen3/src/itkeigen/CMakeLists.txt:set(EIGEN_CUDA_CXX_FLAGS "" CACHE STRING "Additional flags to pass to the cuda compiler.")
Modules/ThirdParty/Eigen3/src/itkeigen/CMakeLists.txt:set(EIGEN_CUDA_COMPUTE_ARCH 30 CACHE STRING "The CUDA compute architecture(s) to target when compiling CUDA code")
Modules/ThirdParty/Eigen3/src/itkeigen/CMakeLists.txt:    set(DPCPP_SYCL_TARGET "spir64" CACHE STRING "Default target for Intel CPU/GPU")
Modules/Remote/VkFFTBackend.remote.cmake:  "ITK FFT accelerated backends using the VkFFT library for Vulkan/CUDA/HIP/OpenCL compatibility."
Modules/Remote/SmoothingRecursiveYvvGaussianFilter.remote.cmake:  "GPU and CPU Young & Van Vliet Recursive Gaussian Smoothing Filter: https://doi.org/10.54294/cpyaig"
Modules/Remote/CudaCommon.remote.cmake:  CudaCommon
Modules/Remote/CudaCommon.remote.cmake:  "Framework for processing images with Cuda."
Modules/Remote/CudaCommon.remote.cmake:  GIT_REPOSITORY https://github.com/RTKConsortium/ITKCudaCommon.git
Modules/Filtering/LabelMap/test/Baseline/itkLabelMapMaskImageFilterTestCrop-0-0-0-0.png.cid:bafkreicgtdpaw2rbtkhvoydgpuwnmmjbd4qnk2g3bn47veizhapfw4czf4
Modules/Filtering/LabelMap/test/Baseline/itkLabelMapMaskImageFilterTest-0-0-0.png.cid:bafkreicgtdpaw2rbtkhvoydgpuwnmmjbd4qnk2g3bn47veizhapfw4czf4
Modules/Filtering/LabelMap/test/Baseline/itkLabelMapMaskImageFilterTestCrop-0-0-0-10.png.cid:bafkreicgtdpaw2rbtkhvoydgpuwnmmjbd4qnk2g3bn47veizhapfw4czf4
Modules/Filtering/MathematicalMorphology/test/itkAnchorOpenCloseImageFilterTest.cxx:#include "itkAnchorOpenCloseImageFilter.h"
Modules/Filtering/MathematicalMorphology/test/itkAnchorOpenCloseImageFilterTest.cxx:itkAnchorOpenCloseImageFilterTest(int, char ** const)
Modules/Filtering/MathematicalMorphology/test/itkAnchorOpenCloseImageFilterTest.cxx:  using FilterType = itk::AnchorOpenCloseImageFilter<ImageType, KernelType, CompateType1, CompateType2>;
Modules/Filtering/MathematicalMorphology/test/itkAnchorOpenCloseImageFilterTest.cxx:  ITK_EXERCISE_BASIC_OBJECT_METHODS(filter, AnchorOpenCloseImageFilter, KernelImageFilter);
Modules/Filtering/MathematicalMorphology/test/CMakeLists.txt:    itkAnchorOpenCloseImageFilterTest.cxx
Modules/Filtering/MathematicalMorphology/test/CMakeLists.txt:  itkAnchorOpenCloseImageFilterTest
Modules/Filtering/MathematicalMorphology/test/CMakeLists.txt:  itkAnchorOpenCloseImageFilterTest)
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseLine.hxx:#ifndef itkAnchorOpenCloseLine_hxx
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseLine.hxx:#define itkAnchorOpenCloseLine_hxx
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseLine.hxx:AnchorOpenCloseLine<TInputPix, TCompare>::AnchorOpenCloseLine()
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseLine.hxx:AnchorOpenCloseLine<TInputPix, TCompare>::DoLine(std::vector<InputImagePixelType> & buffer, unsigned int bufflength)
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseLine.hxx:AnchorOpenCloseLine<TInputPix, TCompare>::StartLine(std::vector<InputImagePixelType> & buffer,
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseLine.hxx:AnchorOpenCloseLine<TInputPix, TCompare>::FinishLine(std::vector<InputImagePixelType> & buffer,
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseLine.hxx:AnchorOpenCloseLine<TInputPix, TCompare>::PrintSelf(std::ostream & os, Indent indent) const
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenImageFilter.h:#include "itkAnchorOpenCloseImageFilter.h"
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenImageFilter.h:  : public AnchorOpenCloseImageFilter<TImage,
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenImageFilter.h:  using Superclass = AnchorOpenCloseImageFilter<TImage,
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.hxx:#ifndef itkAnchorOpenCloseImageFilter_hxx
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.hxx:#define itkAnchorOpenCloseImageFilter_hxx
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.hxx:AnchorOpenCloseImageFilter<TImage, TKernel, TCompare1, TCompare2>::AnchorOpenCloseImageFilter()
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.hxx:AnchorOpenCloseImageFilter<TImage, TKernel, TCompare1, TCompare2>::DynamicThreadedGenerateData(
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.hxx:AnchorOpenCloseImageFilter<TImage, TKernel, TCompare1, TCompare2>::DoFaceOpen(
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.hxx:AnchorOpenCloseImageFilter<TImage, TKernel, TCompare1, TCompare2>::PrintSelf(std::ostream & os, Indent indent) const
Modules/Filtering/MathematicalMorphology/include/itkAnchorCloseImageFilter.h:#include "itkAnchorOpenCloseImageFilter.h"
Modules/Filtering/MathematicalMorphology/include/itkAnchorCloseImageFilter.h:  : public AnchorOpenCloseImageFilter<TImage,
Modules/Filtering/MathematicalMorphology/include/itkAnchorCloseImageFilter.h:  using Superclass = AnchorOpenCloseImageFilter<TImage,
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseLine.h:#ifndef itkAnchorOpenCloseLine_h
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseLine.h:#define itkAnchorOpenCloseLine_h
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseLine.h: * \class AnchorOpenCloseLine
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseLine.h:class ITK_TEMPLATE_EXPORT AnchorOpenCloseLine
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseLine.h:  AnchorOpenCloseLine();
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseLine.h:  ~AnchorOpenCloseLine() = default;
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseLine.h:#  include "itkAnchorOpenCloseLine.hxx"
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.h:#ifndef itkAnchorOpenCloseImageFilter_h
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.h:#define itkAnchorOpenCloseImageFilter_h
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.h:#include "itkAnchorOpenCloseLine.h"
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.h: * \class AnchorOpenCloseImageFilter
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.h:class ITK_TEMPLATE_EXPORT AnchorOpenCloseImageFilter : public KernelImageFilter<TImage, TImage, TKernel>
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(AnchorOpenCloseImageFilter);
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.h:  using Self = AnchorOpenCloseImageFilter;
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.h:  itkOverrideGetNameOfClassMacro(AnchorOpenCloseImageFilter);
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.h:  AnchorOpenCloseImageFilter();
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.h:  ~AnchorOpenCloseImageFilter() override = default;
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.h:  //  using AnchorLineOpenType = AnchorOpenCloseLine<InputImagePixelType, THistogramCompare,
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.h:  using AnchorLineOpenType = AnchorOpenCloseLine<InputImagePixelType, TCompare1>;
Modules/Filtering/MathematicalMorphology/include/itkAnchorOpenCloseImageFilter.h:#  include "itkAnchorOpenCloseImageFilter.hxx"
Modules/Filtering/GPUSmoothing/itk-module.cmake:    "This module contains the GPU implementation of the
Modules/Filtering/GPUSmoothing/itk-module.cmake:  ITKGPUSmoothing
Modules/Filtering/GPUSmoothing/itk-module.cmake:  ITKGPUCommon
Modules/Filtering/GPUSmoothing/itk-module.cmake:  ITKGPUImageFilterBase
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:#include "itkGPUImage.h"
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:#include "itkGPUKernelManager.h"
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:#include "itkGPUContextManager.h"
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:#include "itkGPUImageToImageFilter.h"
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:#include "itkGPUNeighborhoodOperatorImageFilter.h"
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:#include "itkGPUDiscreteGaussianImageFilter.h"
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx: * Testing GPU Discrete Gaussian Image Filter
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:runGPUDiscreteGaussianImageFilterTest(const std::string & inFile, const std::string & outFile)
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:  using InputImageType = itk::GPUImage<InputPixelType, VImageDimension>;
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:  using OutputImageType = itk::GPUImage<OutputPixelType, VImageDimension>;
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:  using GPUFilterType = itk::GPUDiscreteGaussianImageFilter<InputImageType, OutputImageType>;
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:      auto GPUFilter = GPUFilterType::New();
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:      itk::TimeProbe gputimer;
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:      gputimer.Start();
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:      GPUFilter->SetInput(reader->GetOutput());
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:      GPUFilter->SetVariance(variance);
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:      GPUFilter->Update();
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:      GPUFilter->GetOutput()->UpdateBuffers(); // synchronization point (GPU->CPU memcpy)
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:      gputimer.Stop();
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:      std::cout << "GPU Gaussian Filter took " << gputimer.GetMean() << " seconds.\n" << std::endl;
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:      itk::ImageRegionIterator<OutputImageType> git(GPUFilter->GetOutput(),
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:                                                    GPUFilter->GetOutput()->GetLargestPossibleRegion());
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:        //         ", GPU : "
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:      writer->SetInput(GPUFilter->GetOutput());
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:        // but the double precision is not well-supported on most GPUs
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:        // and by most drivers at this time.  Therefore, the GPU filter
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:itkGPUDiscreteGaussianImageFilterTest(int argc, char * argv[])
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:  if (!itk::IsGPUAvailable())
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:    std::cerr << "OpenCL-enabled GPU is not present." << std::endl;
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:    return runGPUDiscreteGaussianImageFilterTest<2>(inFile, outFile);
Modules/Filtering/GPUSmoothing/test/itkGPUDiscreteGaussianImageFilterTest.cxx:    return runGPUDiscreteGaussianImageFilterTest<3>(inFile, outFile);
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:  set(ITKGPUSmoothing-tests itkGPUMeanImageFilterTest.cxx itkGPUDiscreteGaussianImageFilterTest.cxx)
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:  createtestdriver(ITKGPUSmoothing "${ITKGPUSmoothing-Test_LIBRARIES}" "${ITKGPUSmoothing-tests}")
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    itkGPUMeanImageFilterTest2D
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    ITKGPUSmoothingTestDriver
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuMeanImageFilterTest2D.png
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    itkGPUMeanImageFilterTest
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuMeanImageFilterTest2D.png
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    itkGPUMeanImageFilterTest3D
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    ITKGPUSmoothingTestDriver
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    DATA{Baseline/gpuMeanImageFilterTest3D.mha}
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuMeanImageFilterTest3D.mha
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    itkGPUMeanImageFilterTest
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuMeanImageFilterTest3D.mha
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    itkGPUDiscreteGaussianImageFilterTest2D
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    ITKGPUSmoothingTestDriver
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    itkGPUDiscreteGaussianImageFilterTest
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuGradientDiscreteGaussianImageFilterTest2D.mha
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    itkGPUDiscreteGaussianImageFilterTest3D
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    ITKGPUSmoothingTestDriver
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    itkGPUDiscreteGaussianImageFilterTest
Modules/Filtering/GPUSmoothing/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuGradientDiscreteGaussianImageFilterTest3D.mha
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx: * Test program for itkGPUMeanImageFilter class
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx: * This program creates a GPU Mean filter test pipelining.
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:#include "itkGPUImage.h"
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:#include "itkGPUKernelManager.h"
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:#include "itkGPUContextManager.h"
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:#include "itkGPUImageToImageFilter.h"
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:#include "itkGPUMeanImageFilter.h"
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx: * Testing GPU Mean Image Filter
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:runGPUMeanImageFilterTest(const std::string & inFile, const std::string & outFile)
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:  using InputImageType = itk::GPUImage<InputPixelType, VImageDimension>;
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:  using OutputImageType = itk::GPUImage<OutputPixelType, VImageDimension>;
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:  //       GPU filter for Median filter and CPU filter for threshold filter.
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:  using GPUMeanFilterType = itk::GPUMeanImageFilter<InputImageType, OutputImageType>;
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:      auto GPUFilter = GPUMeanFilterType::New();
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:      itk::TimeProbe gputimer;
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:      gputimer.Start();
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:      GPUFilter->SetInput(reader->GetOutput());
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:      GPUFilter->SetRadius(indexRadius);
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:      GPUFilter->Update();
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:      GPUFilter->GetOutput()->UpdateBuffers(); // synchronization point (GPU->CPU memcpy)
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:      gputimer.Stop();
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:      std::cout << "GPU mean filter took " << gputimer.GetMean() << " seconds.\n" << std::endl;
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:      writer->SetInput(GPUFilter->GetOutput());
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:itkGPUMeanImageFilterTest(int argc, char * argv[])
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:  if (!itk::IsGPUAvailable())
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:    std::cerr << "OpenCL-enabled GPU is not present." << std::endl;
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:    return runGPUMeanImageFilterTest<2>(inFile, outFile);
Modules/Filtering/GPUSmoothing/test/itkGPUMeanImageFilterTest.cxx:    return runGPUMeanImageFilterTest<3>(inFile, outFile);
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:#ifndef itkGPUMeanImageFilter_hxx
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:#define itkGPUMeanImageFilter_hxx
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:GPUMeanImageFilter<TInputImage, TOutputImage>::GPUMeanImageFilter()
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:    itkExceptionMacro("GPUMeanImageFilter supports 1/2/3D image.");
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:  const char * GPUSource = GPUMeanImageFilter::GetOpenCLSource();
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:  this->m_GPUKernelManager->LoadProgramFromString(GPUSource, defines.str().c_str());
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:  m_MeanFilterGPUKernelHandle = this->m_GPUKernelManager->CreateKernel("MeanFilter");
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:GPUMeanImageFilter<TInputImage, TOutputImage>::~GPUMeanImageFilter() = default;
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:GPUMeanImageFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream & os, Indent indent) const
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:GPUMeanImageFilter<TInputImage, TOutputImage>::GPUGenerateData()
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:  using GPUInputImage = typename itk::GPUTraits<TInputImage>::Type;
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:  using GPUOutputImage = typename itk::GPUTraits<TOutputImage>::Type;
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:  typename GPUInputImage::Pointer  inPtr = dynamic_cast<GPUInputImage *>(this->ProcessObject::GetInput(0));
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:  typename GPUOutputImage::Pointer otPtr = dynamic_cast<GPUOutputImage *>(this->ProcessObject::GetOutput(0));
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:  typename GPUOutputImage::SizeType outSize = otPtr->GetLargestPossibleRegion().GetSize();
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:  localSize[0] = localSize[1] = localSize[2] = OpenCLGetLocalBlockSize(ImageDim);
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(m_MeanFilterGPUKernelHandle, argidx++, inPtr->GetGPUDataManager());
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(m_MeanFilterGPUKernelHandle, argidx++, otPtr->GetGPUDataManager());
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:    this->m_GPUKernelManager->SetKernelArg(m_MeanFilterGPUKernelHandle, argidx++, sizeof(int), &(radius[i]));
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:    this->m_GPUKernelManager->SetKernelArg(m_MeanFilterGPUKernelHandle, argidx++, sizeof(int), &(imgSize[i]));
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:  this->m_GPUKernelManager->LaunchKernel(
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.hxx:    m_MeanFilterGPUKernelHandle, static_cast<int>(TInputImage::ImageDimension), globalSize, localSize);
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:#ifndef itkGPUDiscreteGaussianImageFilter_h
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:#define itkGPUDiscreteGaussianImageFilter_h
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:#include "itkGPUImage.h"
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:#include "itkGPUImageToImageFilter.h"
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:#include "itkGPUNeighborhoodOperatorImageFilter.h"
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h: * \class GPUDiscreteGaussianImageFilter
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h: * and a discrete Gaussian operator (kernel). GPUNeighborhoodOperatorImageFilter
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h: * \ingroup ITKGPUSmoothing
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:class ITK_TEMPLATE_EXPORT GPUDiscreteGaussianImageFilter
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:  : public GPUImageToImageFilter<TInputImage, TOutputImage, DiscreteGaussianImageFilter<TInputImage, TOutputImage>>
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUDiscreteGaussianImageFilter);
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:  using Self = GPUDiscreteGaussianImageFilter;
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:  using GPUSuperclass = GPUImageToImageFilter<TInputImage, TOutputImage, CPUSuperclass>;
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:  itkOverrideGetNameOfClassMacro(GPUDiscreteGaussianImageFilter);
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:  using RealOutputImageType = GPUImage<OutputPixelType, ImageDimension>;
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:    GPUNeighborhoodOperatorImageFilter<InputImageType, RealOutputImageType, RealOutputPixelValueType>;
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:    GPUNeighborhoodOperatorImageFilter<RealOutputImageType, RealOutputImageType, RealOutputPixelValueType>;
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:    GPUNeighborhoodOperatorImageFilter<RealOutputImageType, OutputImageType, RealOutputPixelValueType>;
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:    GPUNeighborhoodOperatorImageFilter<InputImageType, OutputImageType, RealOutputPixelValueType>;
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:  GPUDiscreteGaussianImageFilter();
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:  ~GPUDiscreteGaussianImageFilter() override = default;
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:  /** Standard GPU pipeline method. */
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:  GPUGenerateData() override;
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.h:#  include "itkGPUDiscreteGaussianImageFilter.hxx"
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.hxx:#ifndef itkGPUDiscreteGaussianImageFilter_hxx
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.hxx:#define itkGPUDiscreteGaussianImageFilter_hxx
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.hxx:#include "itkGPUNeighborhoodOperatorImageFilter.h"
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.hxx:GPUDiscreteGaussianImageFilter<TInputImage, TOutputImage>::GPUDiscreteGaussianImageFilter()
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.hxx:    itkExceptionMacro("GPUDiscreteGaussianImageFilter only supports n-dimensional image.");
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.hxx:GPUDiscreteGaussianImageFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion()
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.hxx:GPUDiscreteGaussianImageFilter<TInputImage, TOutputImage>::GPUGenerateData()
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.hxx:  using GPUInputImage = typename itk::GPUTraits<TInputImage>::Type;
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.hxx:  using GPUOutputImage = typename itk::GPUTraits<TOutputImage>::Type;
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.hxx:  typename GPUOutputImage::Pointer output =
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.hxx:    dynamic_cast<GPUOutputImage *>(this->GetOutput()); // this->ProcessObject::GetOutput(0)
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.hxx:  auto localInput = GPUInputImage::New();
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.hxx:GPUDiscreteGaussianImageFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream & os, Indent indent) const
Modules/Filtering/GPUSmoothing/include/itkGPUDiscreteGaussianImageFilter.hxx:  GPUSuperclass::PrintSelf(os, indent);
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:#ifndef itkGPUMeanImageFilter_h
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:#define itkGPUMeanImageFilter_h
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:#include "itkGPUBoxImageFilter.h"
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:#include "itkOpenCLUtil.h"
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h: * \class GPUMeanImageFilter
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h: * \brief GPU-enabled implementation of the MeanImageFilter.
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h: * Current GPU mean filter reads in neighborhood pixels from global memory.
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h: * \ingroup ITKGPUSmoothing
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:/** Create a helper GPU Kernel class for GPUMeanImageFilter */
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:itkGPUKernelClassMacro(GPUMeanImageFilterKernel);
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:class ITK_TEMPLATE_EXPORT GPUMeanImageFilter
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:  : // public GPUImageToImageFilter<
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:    public GPUBoxImageFilter<TInputImage, TOutputImage, MeanImageFilter<TInputImage, TOutputImage>>
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUMeanImageFilter);
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:  using Self = GPUMeanImageFilter;
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:  using Superclass = GPUBoxImageFilter<TInputImage, TOutputImage, MeanImageFilter<TInputImage, TOutputImage>>;
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:  itkOverrideGetNameOfClassMacro(GPUMeanImageFilter);
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:  /** Get OpenCL Kernel source as a string, creates a GetOpenCLSource method */
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:  itkGetOpenCLSourceFromKernelMacro(GPUMeanImageFilterKernel);
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:  GPUMeanImageFilter();
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:  ~GPUMeanImageFilter() override;
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:  GPUGenerateData() override;
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:  int m_MeanFilterGPUKernelHandle{};
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h: * \class GPUMeanImageFilterFactory
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h: * \brief Object Factory implementation for GPUMeanImageFilter
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h: * \ingroup ITKGPUSmoothing
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:class GPUMeanImageFilterFactory : public ObjectFactoryBase
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUMeanImageFilterFactory);
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:  using Self = GPUMeanImageFilterFactory;
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:    return "A Factory for GPUMeanImageFilter";
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:  itkOverrideGetNameOfClassMacro(GPUMeanImageFilterFactory);
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:    auto factory = GPUMeanImageFilterFactory::New();
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:                           typeid(GPUMeanImageFilter<InputImageType, OutputImageType>).name(),                \
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:                           "GPU Mean Image Filter Override",                                                  \
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:                           CreateObjectFunction<GPUMeanImageFilter<InputImageType, OutputImageType>>::New()); \
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:  GPUMeanImageFilterFactory()
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:    if (IsGPUAvailable())
Modules/Filtering/GPUSmoothing/include/itkGPUMeanImageFilter.h:#  include "itkGPUMeanImageFilter.hxx"
Modules/Filtering/GPUSmoothing/CMakeLists.txt:project(ITKGPUSmoothing)
Modules/Filtering/GPUSmoothing/CMakeLists.txt:  set(ITK_USE_GPU
Modules/Filtering/GPUSmoothing/CMakeLists.txt:      CACHE BOOL "Enable OpenCL GPU support." FORCE)
Modules/Filtering/GPUSmoothing/CMakeLists.txt:  include(itkOpenCL)
Modules/Filtering/GPUSmoothing/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Filtering/GPUSmoothing/CMakeLists.txt:  set(ITKGPUSmoothing_LIBRARIES ITKGPUSmoothing)
Modules/Filtering/GPUSmoothing/CMakeLists.txt:  set(ITKGPUSmoothing_SYSTEM_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
Modules/Filtering/GPUSmoothing/CMakeLists.txt:  set(ITKGPUSmoothing_SYSTEM_LIBRARY_DIRS ${OPENCL_LIBRARIES})
Modules/Filtering/GPUSmoothing/wrapping/test/itkGPUMeanImageFilterTest.py:# Testing GPU Mean Image Filter
Modules/Filtering/GPUSmoothing/wrapping/test/itkGPUMeanImageFilterTest.py:InputGPUImageType = itk.GPUImage[InputPixelType, Dimension]
Modules/Filtering/GPUSmoothing/wrapping/test/itkGPUMeanImageFilterTest.py:OutputGPUImageType = itk.GPUImage[OutputPixelType, Dimension]
Modules/Filtering/GPUSmoothing/wrapping/test/itkGPUMeanImageFilterTest.py:input_gpu_image = itk.cast_image_filter(
Modules/Filtering/GPUSmoothing/wrapping/test/itkGPUMeanImageFilterTest.py:    input_image, in_place=False, ttype=(InputImageType, InputGPUImageType)
Modules/Filtering/GPUSmoothing/wrapping/test/itkGPUMeanImageFilterTest.py:input_gpu_image.UpdateBuffers()
Modules/Filtering/GPUSmoothing/wrapping/test/itkGPUMeanImageFilterTest.py:GPUMeanFilterType = itk.GPUMeanImageFilter[InputGPUImageType, OutputGPUImageType]
Modules/Filtering/GPUSmoothing/wrapping/test/itkGPUMeanImageFilterTest.py:gpu_filter = GPUMeanFilterType.New()
Modules/Filtering/GPUSmoothing/wrapping/test/itkGPUMeanImageFilterTest.py:gpu_timer = itk.TimeProbe()
Modules/Filtering/GPUSmoothing/wrapping/test/itkGPUMeanImageFilterTest.py:gpu_timer.Start()
Modules/Filtering/GPUSmoothing/wrapping/test/itkGPUMeanImageFilterTest.py:gpu_filter.SetInput(input_gpu_image)
Modules/Filtering/GPUSmoothing/wrapping/test/itkGPUMeanImageFilterTest.py:gpu_filter.Update()
Modules/Filtering/GPUSmoothing/wrapping/test/itkGPUMeanImageFilterTest.py:gpu_filter.GetOutput().UpdateBuffers()  # synchronization point (GPU->CPU memcpy)
Modules/Filtering/GPUSmoothing/wrapping/test/itkGPUMeanImageFilterTest.py:gpu_timer.Stop()
Modules/Filtering/GPUSmoothing/wrapping/test/itkGPUMeanImageFilterTest.py:print(f"GPU MeanFilter took {gpu_timer.GetMean()} seconds.\n")
Modules/Filtering/GPUSmoothing/wrapping/test/itkGPUMeanImageFilterTest.py:    gpu_filter.GetOutput(), ttype=(OutputGPUImageType, OutputImageType)
Modules/Filtering/GPUSmoothing/wrapping/test/itkGPUMeanImageFilterTest.py:output_gpu_image = gpu_filter.GetOutput()
Modules/Filtering/GPUSmoothing/wrapping/test/CMakeLists.txt:    itkGPUMeanImageFilterPythonTest2D
Modules/Filtering/GPUSmoothing/wrapping/test/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/itkGPUMeanImageFilterTest.py
Modules/Filtering/GPUSmoothing/wrapping/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/itkGPUMeanImageFilterPythonTest2D.mha
Modules/Filtering/GPUSmoothing/wrapping/CMakeLists.txt:if(ITK_USE_GPU OR NOT ITK_SOURCE_DIR)
Modules/Filtering/GPUSmoothing/wrapping/CMakeLists.txt:  itk_wrap_module(ITKGPUSmoothing)
Modules/Filtering/GPUSmoothing/wrapping/itkGPUMeanImageFilter.wrap:itk_wrap_include("itkGPUImage.h")
Modules/Filtering/GPUSmoothing/wrapping/itkGPUMeanImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >")
Modules/Filtering/GPUSmoothing/wrapping/itkGPUMeanImageFilter.wrap:itk_wrap_class("itk::GPUImageToImageFilter" POINTER)
Modules/Filtering/GPUSmoothing/wrapping/itkGPUMeanImageFilter.wrap:      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::MeanImageFilter< itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} > >"
Modules/Filtering/GPUSmoothing/wrapping/itkGPUMeanImageFilter.wrap:itk_wrap_class("itk::GPUBoxImageFilter" POINTER)
Modules/Filtering/GPUSmoothing/wrapping/itkGPUMeanImageFilter.wrap:      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::MeanImageFilter< itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} > >"
Modules/Filtering/GPUSmoothing/wrapping/itkGPUMeanImageFilter.wrap:itk_wrap_class("itk::GPUMeanImageFilter" POINTER)
Modules/Filtering/GPUSmoothing/wrapping/itkGPUMeanImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >")
Modules/Filtering/GPUSmoothing/wrapping/itkGPUDiscreteGaussianImageFilter.wrap:itk_wrap_include("itkGPUImage.h")
Modules/Filtering/GPUSmoothing/wrapping/itkGPUDiscreteGaussianImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >")
Modules/Filtering/GPUSmoothing/wrapping/itkGPUDiscreteGaussianImageFilter.wrap:itk_wrap_class("itk::GPUImageToImageFilter" POINTER)
Modules/Filtering/GPUSmoothing/wrapping/itkGPUDiscreteGaussianImageFilter.wrap:      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::DiscreteGaussianImageFilter< itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} > >"
Modules/Filtering/GPUSmoothing/wrapping/itkGPUDiscreteGaussianImageFilter.wrap:itk_wrap_class("itk::GPUDiscreteGaussianImageFilter" POINTER)
Modules/Filtering/GPUSmoothing/wrapping/itkGPUDiscreteGaussianImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >")
Modules/Filtering/GPUSmoothing/src/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Filtering/GPUSmoothing/src/CMakeLists.txt:  set(ITKGPUSmoothing_SRCS)
Modules/Filtering/GPUSmoothing/src/CMakeLists.txt:  set(ITKGPUSmoothing_Kernels GPUMeanImageFilter.cl)
Modules/Filtering/GPUSmoothing/src/CMakeLists.txt:  write_gpu_kernels("${ITKGPUSmoothing_Kernels}" ITKGPUSmoothing_SRCS)
Modules/Filtering/GPUSmoothing/src/CMakeLists.txt:  itk_module_add_library(ITKGPUSmoothing ${ITKGPUSmoothing_SRCS})
Modules/Filtering/GPUSmoothing/src/CMakeLists.txt:  target_link_libraries(ITKGPUSmoothing LINK_PUBLIC ${OPENCL_LIBRARIES})
Modules/Filtering/GPUSmoothing/src/GPUMeanImageFilter.cl:// very slow on older GPUs (pre-Fermi) that do not have hardware cache.
Modules/Filtering/GPUSmoothing/src/GPUMeanImageFilter.cl:     execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
Modules/Filtering/GPUThresholding/itk-module.cmake:set(DOCUMENTATION "This module contains the GPU implementation for image
Modules/Filtering/GPUThresholding/itk-module.cmake:  ITKGPUThresholding
Modules/Filtering/GPUThresholding/itk-module.cmake:  ITKGPUCommon
Modules/Filtering/GPUThresholding/itk-module.cmake:  ITKGPUSmoothing
Modules/Filtering/GPUThresholding/test/itkGPUImageFilterTest.cxx: * Test program for itkGPUImageToImageFilter class
Modules/Filtering/GPUThresholding/test/itkGPUImageFilterTest.cxx: * This program creates a GPU Mean filter and a CPU threshold filter using
Modules/Filtering/GPUThresholding/test/itkGPUImageFilterTest.cxx: * object factory framework and test pipelining of GPU and CPU filters.
Modules/Filtering/GPUThresholding/test/itkGPUImageFilterTest.cxx://#include "pathToOpenCLSourceCode.h"
Modules/Filtering/GPUThresholding/test/itkGPUImageFilterTest.cxx:#include "itkGPUImage.h"
Modules/Filtering/GPUThresholding/test/itkGPUImageFilterTest.cxx:#include "itkGPUKernelManager.h"
Modules/Filtering/GPUThresholding/test/itkGPUImageFilterTest.cxx:#include "itkGPUContextManager.h"
Modules/Filtering/GPUThresholding/test/itkGPUImageFilterTest.cxx:#include "itkGPUImageToImageFilter.h"
Modules/Filtering/GPUThresholding/test/itkGPUImageFilterTest.cxx:#include "itkGPUMeanImageFilter.h"
Modules/Filtering/GPUThresholding/test/itkGPUImageFilterTest.cxx:runGPUImageFilterTest(const std::string & inFile, const std::string & outFile)
Modules/Filtering/GPUThresholding/test/itkGPUImageFilterTest.cxx:  //       GPU filter for Median filter and CPU filter for threshold filter.
Modules/Filtering/GPUThresholding/test/itkGPUImageFilterTest.cxx:  filter1->SetInput(reader->GetOutput()); // copy CPU->GPU implicilty
Modules/Filtering/GPUThresholding/test/itkGPUImageFilterTest.cxx:  writer->SetInput(filter3->GetOutput()); // copy GPU->CPU implicilty
Modules/Filtering/GPUThresholding/test/itkGPUImageFilterTest.cxx:itkGPUImageFilterTest(int argc, char * argv[])
Modules/Filtering/GPUThresholding/test/itkGPUImageFilterTest.cxx:    return runGPUImageFilterTest<2>(inFile, outFile);
Modules/Filtering/GPUThresholding/test/itkGPUImageFilterTest.cxx:    return runGPUImageFilterTest<3>(inFile, outFile);
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx: * Test program for itkGPUBinaryThresholdImageFilter class
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:#include "itkGPUImage.h"
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:#include "itkGPUKernelManager.h"
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:#include "itkGPUContextManager.h"
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:#include "itkGPUImageToImageFilter.h"
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:#include "itkGPUBinaryThresholdImageFilter.h"
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:runGPUBinaryThresholdImageFilterTest(const std::string & inFile, const std::string & outFile)
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:  using InputImageType = itk::GPUImage<InputPixelType, VImageDimension>;
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:  using OutputImageType = itk::GPUImage<OutputPixelType, VImageDimension>;
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:  if (!itk::IsGPUAvailable())
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:    std::cerr << "OpenCL-enabled GPU is not present." << std::endl;
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:  using GPUThresholdFilterType = itk::GPUBinaryThresholdImageFilter<InputImageType, OutputImageType>;
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:      auto GPUFilter = GPUThresholdFilterType::New();
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:      itk::TimeProbe gputimer;
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:      gputimer.Start();
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:      GPUFilter->SetOutsideValue(outsideValue);
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:      GPUFilter->SetInsideValue(insideValue);
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:      GPUFilter->SetUpperThreshold(upperThreshold);
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:      GPUFilter->SetLowerThreshold(lowerThreshold);
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:      // GPUFilter->SetInPlace( true );
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:      GPUFilter->SetInput(reader->GetOutput());
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:      GPUFilter->Update();
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:      GPUFilter->GetOutput()->UpdateBuffers(); // synchronization point
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:      gputimer.Stop();
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:      std::cout << "GPU binary threshold took " << gputimer.GetMean() << " seconds.\n" << std::endl;
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:      itk::ImageRegionIterator<OutputImageType> git(GPUFilter->GetOutput(),
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:                                                    GPUFilter->GetOutput()->GetLargestPossibleRegion());
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:        writer->SetInput(GPUFilter->GetOutput());
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:itkGPUBinaryThresholdImageFilterTest(int argc, char * argv[])
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:    return runGPUBinaryThresholdImageFilterTest<2>(inFile, outFile);
Modules/Filtering/GPUThresholding/test/itkGPUBinaryThresholdImageFilterTest.cxx:    return runGPUBinaryThresholdImageFilterTest<3>(inFile, outFile);
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:  set(ITKGPUThresholding-tests
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:      itkGPUImageFilterTest.cxx
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:      #itkGPUImageFilterTestTemp.cxx
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:      itkGPUBinaryThresholdImageFilterTest.cxx)
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:  createtestdriver(ITKGPUThresholding "${ITKGPUThresholding-Test_LIBRARIES}" "${ITKGPUThresholding-tests}")
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:    itkGPUImageFilterTest2D
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:    ITKGPUThresholdingTestDriver
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:    itkGPUImageFilterTest
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuImageFilterTest2D.png
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:    itkGPUImageFilterTest3D
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:    ITKGPUThresholdingTestDriver
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:    itkGPUImageFilterTest
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuImageFilterTest3D.png
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:  #itk_add_test(NAME itkGPUImageFilterTestTemp
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:  #        COMMAND ITKGPUThresholdingTestDriver itkGPUImageFilterTestTemp
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:  #        ${ITK_TEST_OUTPUT_DIR}/gpuImageFilterTest.png)
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:    itkGPUBinaryThresholdImageFilterTest2D
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:    ITKGPUThresholdingTestDriver
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:    itkGPUBinaryThresholdImageFilterTest
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuBinaryThresholdImageFilterTest2D.mha
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:    itkGPUBinaryThresholdImageFilterTest3D
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:    ITKGPUThresholdingTestDriver
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:    itkGPUBinaryThresholdImageFilterTest
Modules/Filtering/GPUThresholding/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuBinaryThresholdImageFilterTest3D.mha
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:#ifndef itkGPUBinaryThresholdImageFilter_h
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:#define itkGPUBinaryThresholdImageFilter_h
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:#include "itkOpenCLUtil.h"
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:#include "itkGPUFunctorBase.h"
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:#include "itkGPUKernelManager.h"
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:#include "itkGPUUnaryFunctorImageFilter.h"
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:class ITK_TEMPLATE_EXPORT GPUBinaryThreshold : public GPUFunctorBase
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  GPUBinaryThreshold()
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  ~GPUBinaryThreshold() override = default;
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  /** Setup GPU kernel arguments for this functor.
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:   * Returns current argument index to set additional arguments in the GPU kernel */
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  SetGPUKernelArguments(GPUKernelManager::Pointer KernelManager, int KernelHandle) override
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:/** Create a helper GPU Kernel class for GPUBinaryThresholdImageFilter */
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:itkGPUKernelClassMacro(GPUBinaryThresholdImageFilterKernel);
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h: * \class GPUBinaryThresholdImageFilter
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h: * \brief GPU version of binary threshold image filter.
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h: * \ingroup ITKGPUThresholding
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:class ITK_TEMPLATE_EXPORT GPUBinaryThresholdImageFilter
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  : public GPUUnaryFunctorImageFilter<
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:      Functor::GPUBinaryThreshold<typename TInputImage::PixelType, typename TOutputImage::PixelType>,
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUBinaryThresholdImageFilter);
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  using Self = GPUBinaryThresholdImageFilter;
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  using GPUSuperclass = GPUUnaryFunctorImageFilter<
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:    Functor::GPUBinaryThreshold<typename TInputImage::PixelType, typename TOutputImage::PixelType>,
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  itkOverrideGetNameOfClassMacro(GPUBinaryThresholdImageFilter);
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  /** Get OpenCL Kernel source as a string, creates a GetOpenCLSource method */
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  itkGetOpenCLSourceFromKernelMacro(GPUBinaryThresholdImageFilterKernel);
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  GPUBinaryThresholdImageFilter();
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  ~GPUBinaryThresholdImageFilter() override = default;
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  /** Unlike CPU version, GPU version of binary threshold filter is not
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  GPUGenerateData() override;
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h: * \class GPUBinaryThresholdImageFilterFactory
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h: * Object Factory implementation for GPUBinaryThresholdImageFilter
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h: * \ingroup ITKGPUThresholding
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:class GPUBinaryThresholdImageFilterFactory : public ObjectFactoryBase
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUBinaryThresholdImageFilterFactory);
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  using Self = GPUBinaryThresholdImageFilterFactory;
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:    return "A Factory for GPUBinaryThresholdImageFilter";
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  itkOverrideGetNameOfClassMacro(GPUBinaryThresholdImageFilterFactory);
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:    auto factory = GPUBinaryThresholdImageFilterFactory::New();
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:      typeid(itk::GPUBinaryThresholdImageFilter<InputImageType, OutputImageType>).name(),                \
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:      "GPU Binary Threshold Image Filter Override",                                                      \
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:      itk::CreateObjectFunction<GPUBinaryThresholdImageFilter<InputImageType, OutputImageType>>::New()); \
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:  GPUBinaryThresholdImageFilterFactory()
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:    if (IsGPUAvailable())
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.h:#  include "itkGPUBinaryThresholdImageFilter.hxx"
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.hxx:#ifndef itkGPUBinaryThresholdImageFilter_hxx
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.hxx:#define itkGPUBinaryThresholdImageFilter_hxx
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.hxx:GPUBinaryThresholdImageFilter<TInputImage, TOutputImage>::GPUBinaryThresholdImageFilter()
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.hxx:    itkExceptionMacro("GPUBinaryThresholdImageFilter supports 1/2/3D image.");
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.hxx:    // This is to work around a bug in the OpenCL compiler on Mac OS 10.6 and 10.7 with NVidia drivers
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.hxx:    excpMsg << "GPUBinaryThresholdImageFilter supports";
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.hxx:  const char * GPUSource = GPUBinaryThresholdImageFilter::GetOpenCLSource();
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.hxx:  this->m_GPUKernelManager->LoadProgramFromString(GPUSource, defines.str().c_str());
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.hxx:  this->m_UnaryFunctorImageFilterGPUKernelHandle = this->m_GPUKernelManager->CreateKernel("BinaryThresholdFilter");
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.hxx:GPUBinaryThresholdImageFilter<TInputImage, TOutputImage>::GPUGenerateData()
Modules/Filtering/GPUThresholding/include/itkGPUBinaryThresholdImageFilter.hxx:  GPUSuperclass::GPUGenerateData();
Modules/Filtering/GPUThresholding/CMakeLists.txt:project(ITKGPUThresholding)
Modules/Filtering/GPUThresholding/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Filtering/GPUThresholding/CMakeLists.txt:  set(ITKGPUThresholding_LIBRARIES ITKGPUThresholding)
Modules/Filtering/GPUThresholding/CMakeLists.txt:  set(ITKGPUThresholding_SYSTEM_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
Modules/Filtering/GPUThresholding/CMakeLists.txt:  set(ITKGPUThresholding_SYSTEM_LIBRARY_DIRS ${OPENCL_LIBRARIES})
Modules/Filtering/GPUThresholding/src/GPUBinaryThresholdImageFilter.cl:     execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
Modules/Filtering/GPUThresholding/src/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Filtering/GPUThresholding/src/CMakeLists.txt:  set(ITKGPUThresholding_SRCS)
Modules/Filtering/GPUThresholding/src/CMakeLists.txt:  set(ITKGPUThresholding_Kernels GPUBinaryThresholdImageFilter.cl)
Modules/Filtering/GPUThresholding/src/CMakeLists.txt:  write_gpu_kernels("${ITKGPUThresholding_Kernels}" ITKGPUThresholding_SRCS)
Modules/Filtering/GPUThresholding/src/CMakeLists.txt:  itk_module_add_library(ITKGPUThresholding ${ITKGPUThresholding_SRCS})
Modules/Filtering/GPUThresholding/src/CMakeLists.txt:  target_link_libraries(ITKGPUThresholding LINK_PUBLIC ${OPENCL_LIBRARIES})
Modules/Filtering/GPUAnisotropicSmoothing/itk-module.cmake:    "This module contains the GPU implementation of filters that
Modules/Filtering/GPUAnisotropicSmoothing/itk-module.cmake:\\\\ref ITKGPUSmoothing.")
Modules/Filtering/GPUAnisotropicSmoothing/itk-module.cmake:  ITKGPUAnisotropicSmoothing
Modules/Filtering/GPUAnisotropicSmoothing/itk-module.cmake:  ITKGPUCommon
Modules/Filtering/GPUAnisotropicSmoothing/itk-module.cmake:  ITKGPUFiniteDifference
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx: * Test program for GPUGradientAnisotropicDiffusionImageFilter class
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:#include "itkOpenCLUtil.h"
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:#include "itkGPUImage.h"
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:#include "itkGPUKernelManager.h"
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:#include "itkGPUContextManager.h"
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:#include "itkGPUImageToImageFilter.h"
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:#include "itkGPUGradientAnisotropicDiffusionImageFilter.h"
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:runGPUGradientAnisotropicDiffusionImageFilterTest(const std::string & inFile, const std::string & outFile)
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:  using InputImageType = itk::GPUImage<InputPixelType, VImageDimension>;
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:  using OutputImageType = itk::GPUImage<OutputPixelType, VImageDimension>;
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:  // Create CPU/GPU anistorpic diffusion filter
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:  using GPUAnisoDiffFilterType = itk::GPUGradientAnisotropicDiffusionImageFilter<InputImageType, OutputImageType>;
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:  auto GPUFilter = GPUAnisoDiffFilterType::New();
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:    GPUFilter, GPUGradientAnisotropicDiffusionImageFilter, GPUAnisotropicDiffusionImageFilter);
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:      itk::TimeProbe gputimer;
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:      gputimer.Start();
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:      GPUFilter->SetInput(reader->GetOutput());
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:      GPUFilter->SetNumberOfIterations(10);
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:      GPUFilter->SetTimeStep(0.0625); // 125 );
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:      GPUFilter->SetConductanceParameter(3.0);
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:      GPUFilter->UseImageSpacingOn();
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:        GPUFilter->Update();
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:        std::cout << "Caught exception during GPUFilter->Update() " << excp << std::endl;
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:        GPUFilter->GetOutput()->UpdateBuffers(); // synchronization point
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:        std::cout << "Caught exception during GPUFilter->GetOutput()->UpdateBuffers() " << excp << std::endl;
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:      gputimer.Stop();
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:      std::cout << "GPU Anisotropic diffusion took " << gputimer.GetMean() << " seconds.\n" << std::endl;
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:      itk::ImageRegionIterator<OutputImageType> git(GPUFilter->GetOutput(),
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:                                                    GPUFilter->GetOutput()->GetLargestPossibleRegion());
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:      writer->SetInput(GPUFilter->GetOutput()); // copy GPU->CPU implicilty
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:  GPUFilter = nullptr;                                      // explicit GPU object destruction test
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:  itk::GPUContextManager::GetInstance()->DestroyInstance(); // GPUContextManager singleton destruction test
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:itkGPUGradientAnisotropicDiffusionImageFilterTest(int argc, char * argv[])
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:  if (!itk::IsGPUAvailable())
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:    std::cerr << "OpenCL-enabled GPU is not present." << std::endl;
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:    return runGPUGradientAnisotropicDiffusionImageFilterTest<2>(inFile, outFile);
Modules/Filtering/GPUAnisotropicSmoothing/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx:    return runGPUGradientAnisotropicDiffusionImageFilterTest<3>(inFile, outFile);
Modules/Filtering/GPUAnisotropicSmoothing/test/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Filtering/GPUAnisotropicSmoothing/test/CMakeLists.txt:  set(ITKGPUAnisotropicSmoothing-tests itkGPUGradientAnisotropicDiffusionImageFilterTest.cxx)
Modules/Filtering/GPUAnisotropicSmoothing/test/CMakeLists.txt:  createtestdriver(ITKGPUAnisotropicSmoothing "${ITKGPUAnisotropicSmoothing-Test_LIBRARIES}"
Modules/Filtering/GPUAnisotropicSmoothing/test/CMakeLists.txt:                   "${ITKGPUAnisotropicSmoothing-tests}")
Modules/Filtering/GPUAnisotropicSmoothing/test/CMakeLists.txt:    itkGPUGradientAnisotropicDiffusionImageFilterTest2D
Modules/Filtering/GPUAnisotropicSmoothing/test/CMakeLists.txt:    ITKGPUAnisotropicSmoothingTestDriver
Modules/Filtering/GPUAnisotropicSmoothing/test/CMakeLists.txt:    itkGPUGradientAnisotropicDiffusionImageFilterTest
Modules/Filtering/GPUAnisotropicSmoothing/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuGradientAnisotropicDiffusionImageFilterTest2D.mha
Modules/Filtering/GPUAnisotropicSmoothing/test/CMakeLists.txt:    itkGPUGradientAnisotropicDiffusionImageFilterTest3D
Modules/Filtering/GPUAnisotropicSmoothing/test/CMakeLists.txt:    ITKGPUAnisotropicSmoothingTestDriver
Modules/Filtering/GPUAnisotropicSmoothing/test/CMakeLists.txt:    itkGPUGradientAnisotropicDiffusionImageFilterTest
Modules/Filtering/GPUAnisotropicSmoothing/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuGradientAnisotropicDiffusionImageFilterTest3D.mha
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:#ifndef itkGPUGradientAnisotropicDiffusionImageFilter_h
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:#define itkGPUGradientAnisotropicDiffusionImageFilter_h
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:#include "itkOpenCLUtil.h"
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:#include "itkGPUAnisotropicDiffusionImageFilter.h"
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:#include "itkGPUGradientNDAnisotropicDiffusionFunction.h"
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h: * \class GPUGradientAnisotropicDiffusionImageFilter
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h: * \ingroup ITKGPUAnisotropicSmoothing
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:class GPUGradientAnisotropicDiffusionImageFilter
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:  : public GPUAnisotropicDiffusionImageFilter<TInputImage, TOutputImage, TParentImageFilter>
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUGradientAnisotropicDiffusionImageFilter);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:  using Self = GPUGradientAnisotropicDiffusionImageFilter;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:  using Superclass = GPUAnisotropicDiffusionImageFilter<TInputImage, TOutputImage, TParentImageFilter>;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:  using GPUSuperclass = GPUAnisotropicDiffusionImageFilter<TInputImage, TOutputImage, TParentImageFilter>;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:  itkOverrideGetNameOfClassMacro(GPUGradientAnisotropicDiffusionImageFilter);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:  using UpdateBufferType = typename GPUSuperclass::UpdateBufferType;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:  static constexpr unsigned int ImageDimension = GPUSuperclass::ImageDimension;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:  GPUGradientAnisotropicDiffusionImageFilter()
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:    typename GPUGradientNDAnisotropicDiffusionFunction<UpdateBufferType>::Pointer p =
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:      GPUGradientNDAnisotropicDiffusionFunction<UpdateBufferType>::New();
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilter.h:  ~GPUGradientAnisotropicDiffusionImageFilter() override = default;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:#ifndef itkGPUScalarAnisotropicDiffusionFunction_hxx
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:#define itkGPUScalarAnisotropicDiffusionFunction_hxx
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:GPUScalarAnisotropicDiffusionFunction<TImage>::GPUScalarAnisotropicDiffusionFunction()
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  this->m_AnisotropicDiffusionFunctionGPUBuffer = GPUDataManager::New();
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  this->m_AnisotropicDiffusionFunctionGPUKernelManager = GPUKernelManager::New();
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  // load GPU kernel
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:    itkExceptionMacro("GPUScalarAnisotropicDiffusionFunction supports 1/2/3D image.");
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  const char * GPUSource = GPUScalarAnisotropicDiffusionFunction::GetOpenCLSource();
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  this->m_AnisotropicDiffusionFunctionGPUKernelManager->LoadProgramFromString(GPUSource, defines.str().c_str());
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  this->m_AverageGradientMagnitudeSquaredGPUKernelHandle =
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:    this->m_AnisotropicDiffusionFunctionGPUKernelManager->CreateKernel("AverageGradientMagnitudeSquared");
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:GPUScalarAnisotropicDiffusionFunction<TImage>::GPUCalculateAverageGradientMagnitudeSquared(TImage * ip)
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  // GPU kernel to compute Average Squared Gradient Magnitude
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  using GPUImageType = typename itk::GPUTraits<TImage>::Type;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  typename GPUImageType::Pointer  inPtr = dynamic_cast<GPUImageType *>(ip);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  typename GPUImageType::SizeType outSize = inPtr->GetLargestPossibleRegion().GetSize();
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  unsigned int blockSize = OpenCLGetLocalBlockSize(ImageDim);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  // Initialize & Allocate GPU Buffer
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  if (bufferSize != this->m_AnisotropicDiffusionFunctionGPUBuffer->GetBufferSize())
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:    this->m_AnisotropicDiffusionFunctionGPUBuffer->Initialize();
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:    this->m_AnisotropicDiffusionFunctionGPUBuffer->SetBufferSize(sizeof(float) * bufferSize);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:    this->m_AnisotropicDiffusionFunctionGPUBuffer->Allocate();
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  typename GPUKernelManager::Pointer kernelManager = this->m_AnisotropicDiffusionFunctionGPUKernelManager;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  int                                kernelHandle = this->m_AverageGradientMagnitudeSquaredGPUKernelHandle;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  kernelManager->SetKernelArgWithImage(kernelHandle, argidx++, inPtr->GetGPUDataManager());
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  kernelManager->SetKernelArgWithImage(kernelHandle, argidx++, this->m_AnisotropicDiffusionFunctionGPUBuffer);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  // Read back intermediate sums from GPU and compute final value
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  this->m_AnisotropicDiffusionFunctionGPUBuffer->SetCPUBufferPointer(intermSum.get());
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  this->m_AnisotropicDiffusionFunctionGPUBuffer->SetCPUDirtyFlag(true); //
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  this->m_AnisotropicDiffusionFunctionGPUBuffer->SetGPUDirtyFlag(false);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:  this->m_AnisotropicDiffusionFunctionGPUBuffer->UpdateCPUBuffer(); //
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.hxx:                                                                    // GPU->CPU
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h:#ifndef itkGPUScalarAnisotropicDiffusionFunction_h
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h:#define itkGPUScalarAnisotropicDiffusionFunction_h
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h:#include "itkGPUAnisotropicDiffusionFunction.h"
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h: * \class GPUScalarAnisotropicDiffusionFunction
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h: * This class forms the base for any GPU anisotropic diffusion function that
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h: * operates on scalar data (see itkGPUAnisotropicDiffusionFunction).
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h: * \ingroup ITKGPUAnisotropicSmoothing
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h:/** Create a helper GPU Kernel class for GPUScalarAnisotropicDiffusionFunction */
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h:itkGPUKernelClassMacro(GPUScalarAnisotropicDiffusionFunctionKernel);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h:class ITK_TEMPLATE_EXPORT GPUScalarAnisotropicDiffusionFunction : public GPUAnisotropicDiffusionFunction<TImage>
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUScalarAnisotropicDiffusionFunction);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h:  using Self = GPUScalarAnisotropicDiffusionFunction;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h:  using Superclass = GPUAnisotropicDiffusionFunction<TImage>;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h:  itkOverrideGetNameOfClassMacro(GPUScalarAnisotropicDiffusionFunction);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h:  /** Get OpenCL Kernel source as a string, creates a GetOpenCLSource method */
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h:  itkGetOpenCLSourceFromKernelMacro(GPUScalarAnisotropicDiffusionFunctionKernel);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h:  /** Compute average squared gradient of magnitude using the GPU */
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h:  GPUCalculateAverageGradientMagnitudeSquared(TImage *) override;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h:  GPUScalarAnisotropicDiffusionFunction();
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h:  ~GPUScalarAnisotropicDiffusionFunction() override = default;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUScalarAnisotropicDiffusionFunction.h:#  include "itkGPUScalarAnisotropicDiffusionFunction.hxx"
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:#ifndef itkGPUAnisotropicDiffusionImageFilter_h
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:#define itkGPUAnisotropicDiffusionImageFilter_h
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:#include "itkGPUDenseFiniteDifferenceImageFilter.h"
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h: * \class GPUAnisotropicDiffusionImageFilter
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h: * This filter is the GPU base class for AnisotropicDiffusionImageFilter.
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h: * InitializeIteration() calls GPUCalculateAverageGradientMagnitudeSquared().
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h: * \ingroup ITKGPUAnisotropicSmoothing
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:class ITK_TEMPLATE_EXPORT GPUAnisotropicDiffusionImageFilter
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:  : public GPUDenseFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUAnisotropicDiffusionImageFilter);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:  using Self = GPUAnisotropicDiffusionImageFilter;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:  using GPUSuperclass = GPUDenseFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:  itkOverrideGetNameOfClassMacro(GPUAnisotropicDiffusionImageFilter);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:  using InputImageType = typename GPUSuperclass::InputImageType;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:  using OutputImageType = typename GPUSuperclass::OutputImageType;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:  using UpdateBufferType = typename GPUSuperclass::UpdateBufferType;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:  static constexpr unsigned int ImageDimension = GPUSuperclass::ImageDimension;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:  using PixelType = typename GPUSuperclass::PixelType;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:  using TimeStepType = typename GPUSuperclass::TimeStepType;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:  GPUAnisotropicDiffusionImageFilter() = default;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:  ~GPUAnisotropicDiffusionImageFilter() override = default;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.h:#  include "itkGPUAnisotropicDiffusionImageFilter.hxx"
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h:#ifndef itkGPUGradientNDAnisotropicDiffusionFunction_h
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h:#define itkGPUGradientNDAnisotropicDiffusionFunction_h
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h:#include "itkGPUScalarAnisotropicDiffusionFunction.h"
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h: * \class GPUGradientNDAnisotropicDiffusionFunction
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h: * anisotropic diffusion equation for scalar-valued images on the GPU.  See
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h: * \ingroup ITKGPUAnisotropicSmoothing
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h:/** Create a helper GPU Kernel class for GPUGradientNDAnisotropicDiffusionFunction */
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h:itkGPUKernelClassMacro(GPUGradientNDAnisotropicDiffusionFunctionKernel);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h:class ITK_TEMPLATE_EXPORT GPUGradientNDAnisotropicDiffusionFunction
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h:  : public GPUScalarAnisotropicDiffusionFunction<TImage>
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUGradientNDAnisotropicDiffusionFunction);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h:  using Self = GPUGradientNDAnisotropicDiffusionFunction;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h:  using Superclass = GPUScalarAnisotropicDiffusionFunction<TImage>;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h:  itkOverrideGetNameOfClassMacro(GPUGradientNDAnisotropicDiffusionFunction);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h:  /** Get OpenCL Kernel source as a string, creates a GetOpenCLSource method */
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h:  itkGetOpenCLSourceFromKernelMacro(GPUGradientNDAnisotropicDiffusionFunctionKernel);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h:  GPUComputeUpdate(const typename TImage::Pointer output, typename TImage::Pointer buffer, void * globalData) override;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h:  GPUGradientNDAnisotropicDiffusionFunction();
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h:  ~GPUGradientNDAnisotropicDiffusionFunction() override = default;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.h:#  include "itkGPUGradientNDAnisotropicDiffusionFunction.hxx"
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.hxx:#ifndef itkGPUAnisotropicDiffusionImageFilter_hxx
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.hxx:#define itkGPUAnisotropicDiffusionImageFilter_hxx
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.hxx:#include "itkGPUAnisotropicDiffusionFunction.h"
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.hxx:GPUAnisotropicDiffusionImageFilter<TInputImage, TOutputImage, TParentImageFilter>::InitializeIteration()
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.hxx:    dynamic_cast<GPUAnisotropicDiffusionFunction<UpdateBufferType> *>(this->GetDifferenceFunction().GetPointer());
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.hxx:    throw ExceptionObject(__FILE__, __LINE__, "GPU anisotropic diffusion function is not set.", ITK_LOCATION);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.hxx:      /** GPU version of average squared gradient magnitude calculation */
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionImageFilter.hxx:      f->GPUCalculateAverageGradientMagnitudeSquared(this->GetOutput());
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:#ifndef itkGPUGradientNDAnisotropicDiffusionFunction_hxx
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:#define itkGPUGradientNDAnisotropicDiffusionFunction_hxx
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:#include "itkOpenCLUtil.h"
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:double GPUGradientNDAnisotropicDiffusionFunction<TImage>::m_MIN_NORM = 1.0e-10;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:GPUGradientNDAnisotropicDiffusionFunction<TImage>::GPUGradientNDAnisotropicDiffusionFunction()
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:  // Create GPU Kernel
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:    itkExceptionMacro("GPUGradientNDAnisotropicDiffusionFunction supports 1/2/3D image.");
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:          << "#define BLOCK_SIZE " << OpenCLGetLocalBlockSize(TImage::ImageDimension) << '\n';
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:  // This is to work around a bug in the OpenCL compiler on Mac OS 10.6 and 10.7 with NVidia drivers
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:  const char * GPUSource = GPUGradientNDAnisotropicDiffusionFunction::GetOpenCLSource();
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:  this->m_GPUKernelManager->LoadProgramFromString(GPUSource, defines.str().c_str());
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:  this->m_ComputeUpdateGPUKernelHandle = this->m_GPUKernelManager->CreateKernel("ComputeUpdate");
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:GPUGradientNDAnisotropicDiffusionFunction<TImage>::GPUComputeUpdate(const typename TImage::Pointer output,
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:  /** Launch GPU kernel to update buffer with output
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:   * GPU version of ComputeUpdate() - compute entire update buffer */
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:  using GPUImageType = typename itk::GPUTraits<TImage>::Type;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:  typename GPUImageType::Pointer  inPtr = dynamic_cast<GPUImageType *>(output.GetPointer());
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:  typename GPUImageType::Pointer  bfPtr = dynamic_cast<GPUImageType *>(buffer.GetPointer());
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:  typename GPUImageType::SizeType outSize = bfPtr->GetLargestPossibleRegion().GetSize();
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:  localSize[0] = localSize[1] = localSize[2] = OpenCLGetLocalBlockSize(ImageDim);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:    this->m_ComputeUpdateGPUKernelHandle, argidx++, inPtr->GetGPUDataManager());
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:    this->m_ComputeUpdateGPUKernelHandle, argidx++, bfPtr->GetGPUDataManager());
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:  this->m_GPUKernelManager->SetKernelArg(
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:    this->m_ComputeUpdateGPUKernelHandle, argidx++, sizeof(typename TImage::PixelType), &(m_K));
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:    this->m_GPUKernelManager->SetKernelArg(
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:      this->m_ComputeUpdateGPUKernelHandle, argidx++, sizeof(float), &(imgScale[i]));
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:    this->m_GPUKernelManager->SetKernelArg(this->m_ComputeUpdateGPUKernelHandle, argidx++, sizeof(int), &(imgSize[i]));
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientNDAnisotropicDiffusionFunction.hxx:  this->m_GPUKernelManager->LaunchKernel(this->m_ComputeUpdateGPUKernelHandle, ImageDim, globalSize, localSize);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h:#ifndef itkGPUAnisotropicDiffusionFunction_h
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h:#define itkGPUAnisotropicDiffusionFunction_h
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h:#include "itkGPUFiniteDifferenceFunction.h"
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h: * \class GPUAnisotropicDiffusionFunction
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h: * This is the GPU version of AnisotropicDiffusionFunction class.
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h: * This class must be subclassed to provide the GPUCalculateUpdate() methods of
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h: * GPUFiniteDifferenceFunction and the function
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h: * \ingroup ITKGPUAnisotropicSmoothing
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h:class ITK_TEMPLATE_EXPORT GPUAnisotropicDiffusionFunction : public GPUFiniteDifferenceFunction<TImage>
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUAnisotropicDiffusionFunction);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h:  using Self = GPUAnisotropicDiffusionFunction;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h:  using Superclass = GPUFiniteDifferenceFunction<TImage>;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h:  itkOverrideGetNameOfClassMacro(GPUAnisotropicDiffusionFunction);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h:  GPUCalculateAverageGradientMagnitudeSquared(ImageType *) = 0;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h:  GPUAnisotropicDiffusionFunction()
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h:  ~GPUAnisotropicDiffusionFunction() override = default;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h:  // GPU buffer for Computing Average Squared Gradient Magnitude
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h:  typename GPUDataManager::Pointer   m_AnisotropicDiffusionFunctionGPUBuffer{};
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h:  typename GPUKernelManager::Pointer m_AnisotropicDiffusionFunctionGPUKernelManager{};
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h:  // GPU Kernel Handles
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUAnisotropicDiffusionFunction.h:  int m_AverageGradientMagnitudeSquaredGPUKernelHandle{};
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h:#ifndef itkGPUGradientAnisotropicDiffusionImageFilterFactory_h
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h:#define itkGPUGradientAnisotropicDiffusionImageFilterFactory_h
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h:#include "itkGPUGradientAnisotropicDiffusionImageFilter.h"
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h: * \class GPUGradientAnisotropicDiffusionImageFilterFactory
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h: * Object Factory implementation for GPUGradientAnisotropicDiffusionImageFilter
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h: * \ingroup ITKGPUAnisotropicSmoothing
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h:class GPUGradientAnisotropicDiffusionImageFilterFactory : public ObjectFactoryBase
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUGradientAnisotropicDiffusionImageFilterFactory);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h:  using Self = GPUGradientAnisotropicDiffusionImageFilterFactory;
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h:    return "A Factory for GPUGradientAnisotropicDiffusionImageFilter";
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h:  itkOverrideGetNameOfClassMacro(GPUGradientAnisotropicDiffusionImageFilterFactory);
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h:    GPUGradientAnisotropicDiffusionImageFilterFactory::Pointer factory =
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h:      GPUGradientAnisotropicDiffusionImageFilterFactory::New();
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h:      typeid(itk::GPUGradientAnisotropicDiffusionImageFilter<InputImageType, OutputImageType>).name(),                \
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h:      "GPU GradientAnisotropicDiffusionImageFilter Override",                                                         \
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h:      itk::CreateObjectFunction<GPUGradientAnisotropicDiffusionImageFilter<InputImageType, OutputImageType>>::New()); \
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h:  GPUGradientAnisotropicDiffusionImageFilterFactory()
Modules/Filtering/GPUAnisotropicSmoothing/include/itkGPUGradientAnisotropicDiffusionImageFilterFactory.h:    if (IsGPUAvailable())
Modules/Filtering/GPUAnisotropicSmoothing/CMakeLists.txt:project(ITKGPUAnisotropicSmoothing)
Modules/Filtering/GPUAnisotropicSmoothing/CMakeLists.txt:  set(ITK_USE_GPU
Modules/Filtering/GPUAnisotropicSmoothing/CMakeLists.txt:      CACHE BOOL "Enable OpenCL GPU support." FORCE)
Modules/Filtering/GPUAnisotropicSmoothing/CMakeLists.txt:  include(itkOpenCL)
Modules/Filtering/GPUAnisotropicSmoothing/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Filtering/GPUAnisotropicSmoothing/CMakeLists.txt:  set(ITKGPUAnisotropicSmoothing_LIBRARIES ITKGPUAnisotropicSmoothing)
Modules/Filtering/GPUAnisotropicSmoothing/CMakeLists.txt:  set(ITKGPUAnisotropicSmoothing_SYSTEM_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
Modules/Filtering/GPUAnisotropicSmoothing/CMakeLists.txt:  set(ITKGPUAnisotropicSmoothing_SYSTEM_LIBRARY_DIRS ${OPENCL_LIBRARIES})
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/itkGPUAnisotropicDiffusionImageFilter.wrap:itk_wrap_include("itkGPUImage.h")
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/itkGPUAnisotropicDiffusionImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >")
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/itkGPUAnisotropicDiffusionImageFilter.wrap:  #itk_wrap_template("GI${ITKM_${t}${d}}${d}GI${ITKM_${t}${d}}${d}" "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} >")
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/CMakeLists.txt:    itkGPUGradientAnisotropicDiffusionImageFilterPythonTest2D
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/itkGPUGradientAnisotropicDiffusionImageFilterTest.py
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuGradientAnisotropicDiffusionImageFilterTest2DPython.mha
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/CMakeLists.txt:    itkGPUGradientAnisotropicDiffusionImageFilterPythonTest3D
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/tkGPUGradientAnisotropicDiffusionImageFilterTest.py
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuGradientAnisotropicDiffusionImageFilterTest3DPython.mha
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.py:InputGPUImageType = itk.GPUImage[InputPixelType, Dimension]
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.py:OutputGPUImageType = itk.GPUImage[OutputPixelType, Dimension]
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.py:input_gpu_image = itk.cast_image_filter(
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.py:    input_image, ttype=(InputImageType, InputGPUImageType)
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.py:GPUFilterType = itk.GPUGradientAnisotropicDiffusionImageFilter[
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.py:    InputGPUImageType, OutputGPUImageType
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.py:gpu_filter = GPUFilterType.New(
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.py:gpu_timer = itk.TimeProbe()
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.py:gpu_timer.Start()
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.py:gpu_filter.SetInput(input_gpu_image)
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.py:gpu_filter.Update()
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.py:gpu_filter.GetOutput().UpdateBuffers()  # synchronization point (GPU->CPU memcpy)
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.py:gpu_timer.Stop()
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.py:print(f"GPU NeighborhoodFilter took {gpu_timer.GetMean()} seconds.\n")
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/test/itkGPUGradientAnisotropicDiffusionImageFilterTest.py:    gpu_filter.GetOutput(), ttype=(OutputGPUImageType, OutputImageType)
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/itkGPUGradientAnisotropicDiffusionImageFilter.wrap:itk_wrap_include("itkGPUImage.h")
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/itkGPUGradientAnisotropicDiffusionImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >")
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/itkGPUGradientAnisotropicDiffusionImageFilter.wrap:itk_wrap_class("itk::GPUFiniteDifferenceImageFilter" POINTER)
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/itkGPUGradientAnisotropicDiffusionImageFilter.wrap:      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GradientAnisotropicDiffusionImageFilter< itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} > >"
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/itkGPUGradientAnisotropicDiffusionImageFilter.wrap:itk_wrap_class("itk::GPUDenseFiniteDifferenceImageFilter" POINTER)
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/itkGPUGradientAnisotropicDiffusionImageFilter.wrap:      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GradientAnisotropicDiffusionImageFilter< itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} > >"
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/itkGPUGradientAnisotropicDiffusionImageFilter.wrap:itk_wrap_class("itk::GPUImageToImageFilter" POINTER)
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/itkGPUGradientAnisotropicDiffusionImageFilter.wrap:      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GradientAnisotropicDiffusionImageFilter< itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} > >"
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/itkGPUGradientAnisotropicDiffusionImageFilter.wrap:itk_wrap_class("itk::GPUInPlaceImageFilter" POINTER)
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/itkGPUGradientAnisotropicDiffusionImageFilter.wrap:      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GradientAnisotropicDiffusionImageFilter< itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} > >"
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/itkGPUGradientAnisotropicDiffusionImageFilter.wrap:itk_wrap_class("itk::GPUAnisotropicDiffusionImageFilter" POINTER)
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/itkGPUGradientAnisotropicDiffusionImageFilter.wrap:      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GradientAnisotropicDiffusionImageFilter< itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} > >"
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/itkGPUGradientAnisotropicDiffusionImageFilter.wrap:itk_wrap_class("itk::GPUGradientAnisotropicDiffusionImageFilter" POINTER)
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/itkGPUGradientAnisotropicDiffusionImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >")
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/CMakeLists.txt:if(ITK_USE_GPU OR NOT ITK_SOURCE_DIR)
Modules/Filtering/GPUAnisotropicSmoothing/wrapping/CMakeLists.txt:  itk_wrap_module(ITKGPUAnisotropicSmoothing)
Modules/Filtering/GPUAnisotropicSmoothing/src/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Filtering/GPUAnisotropicSmoothing/src/CMakeLists.txt:  set(ITKGPUAnisotropicSmoothing_SRCS)
Modules/Filtering/GPUAnisotropicSmoothing/src/CMakeLists.txt:  set(ITKGPUAnisotropicSmoothing_Kernels GPUGradientNDAnisotropicDiffusionFunction.cl
Modules/Filtering/GPUAnisotropicSmoothing/src/CMakeLists.txt:                                         GPUScalarAnisotropicDiffusionFunction.cl)
Modules/Filtering/GPUAnisotropicSmoothing/src/CMakeLists.txt:  write_gpu_kernels("${ITKGPUAnisotropicSmoothing_Kernels}" ITKGPUAnisotropicSmoothing_SRCS)
Modules/Filtering/GPUAnisotropicSmoothing/src/CMakeLists.txt:  itk_module_add_library(ITKGPUAnisotropicSmoothing ${ITKGPUAnisotropicSmoothing_SRCS})
Modules/Filtering/GPUAnisotropicSmoothing/src/CMakeLists.txt:  target_link_libraries(ITKGPUAnisotropicSmoothing LINK_PUBLIC ${OPENCL_LIBRARIES})
Modules/Filtering/GPUAnisotropicSmoothing/src/GPUScalarAnisotropicDiffusionFunction.cl:    // NVIDIA OpenCL implementations
Modules/Filtering/GPUAnisotropicSmoothing/src/GPUScalarAnisotropicDiffusionFunction.cl:    // NVIDIA OpenCL implementations
Modules/Filtering/GPUAnisotropicSmoothing/src/GPUScalarAnisotropicDiffusionFunction.cl:    //if(interval > 16) barrier(CLK_LOCAL_MEM_FENCE);  // don't need to synchronize if within a warp (only for NVIDIA)
Modules/Filtering/GPUAnisotropicSmoothing/src/GPUScalarAnisotropicDiffusionFunction.cl:    // NVIDIA OpenCL implementations
Modules/Filtering/GPUImageFilterBase/itk-module.cmake:    "This module contains GPU implementations of the base
Modules/Filtering/GPUImageFilterBase/itk-module.cmake:  ITKGPUImageFilterBase
Modules/Filtering/GPUImageFilterBase/itk-module.cmake:  ITKGPUCommon
Modules/Filtering/GPUImageFilterBase/test/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Filtering/GPUImageFilterBase/test/CMakeLists.txt:  set(ITKGPUImageFilterBase-tests itkGPUNeighborhoodOperatorImageFilterTest.cxx)
Modules/Filtering/GPUImageFilterBase/test/CMakeLists.txt:  createtestdriver(ITKGPUImageFilterBase "${ITKGPUImageFilterBase-Test_LIBRARIES}" "${ITKGPUImageFilterBase-tests}")
Modules/Filtering/GPUImageFilterBase/test/CMakeLists.txt:    itkGPUNeighborhoodOperatorImageFilterTest2D
Modules/Filtering/GPUImageFilterBase/test/CMakeLists.txt:    ITKGPUImageFilterBaseTestDriver
Modules/Filtering/GPUImageFilterBase/test/CMakeLists.txt:    itkGPUNeighborhoodOperatorImageFilterTest
Modules/Filtering/GPUImageFilterBase/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuNeighborhoodOperatorImageFilterTest2D.mha
Modules/Filtering/GPUImageFilterBase/test/CMakeLists.txt:    itkGPUNeighborhoodOperatorImageFilterTest3D
Modules/Filtering/GPUImageFilterBase/test/CMakeLists.txt:    ITKGPUImageFilterBaseTestDriver
Modules/Filtering/GPUImageFilterBase/test/CMakeLists.txt:    itkGPUNeighborhoodOperatorImageFilterTest
Modules/Filtering/GPUImageFilterBase/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuNeighborhoodOperatorImageFilterTest3D.mha
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:#include "itkGPUImage.h"
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:#include "itkGPUKernelManager.h"
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:#include "itkGPUContextManager.h"
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:#include "itkGPUImageToImageFilter.h"
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:#include "itkGPUNeighborhoodOperatorImageFilter.h"
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx: * Testing GPU Neighborhood Operator Image Filter
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:runGPUNeighborhoodOperatorImageFilterTest(const std::string & inFile, const std::string & outFile)
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:  using InputImageType = itk::GPUImage<InputPixelType, VImageDimension>;
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:  using OutputImageType = itk::GPUImage<OutputPixelType, VImageDimension>;
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:  using GPUNeighborhoodFilterType =
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:    itk::GPUNeighborhoodOperatorImageFilter<InputImageType, OutputImageType, RealOutputPixelValueType>;
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:      auto GPUFilter = GPUNeighborhoodFilterType::New();
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:      itk::TimeProbe gputimer;
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:      gputimer.Start();
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:      GPUFilter->SetInput(reader->GetOutput());
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:      GPUFilter->SetOperator(oper);
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:      GPUFilter->Update();
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:      GPUFilter->GetOutput()->UpdateBuffers(); // synchronization point (GPU->CPU memcpy)
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:      gputimer.Stop();
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:      std::cout << "GPU NeighborhoodFilter took " << gputimer.GetMean() << " seconds.\n" << std::endl;
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:      itk::ImageRegionIterator<OutputImageType> git(GPUFilter->GetOutput(),
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:                                                    GPUFilter->GetOutput()->GetLargestPossibleRegion());
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:        //         static_cast<double>(cit.Get()) << ", GPU : " << static_cast<double>(git.Get()) << std::endl;
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:      writer->SetInput(GPUFilter->GetOutput());
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:        // but the double precision is not well-supported on most GPUs
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:        // and by most drivers at this time.  Therefore, the GPU filter
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:itkGPUNeighborhoodOperatorImageFilterTest(int argc, char * argv[])
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:  if (!itk::IsGPUAvailable())
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:    std::cerr << "OpenCL-enabled GPU is not present." << std::endl;
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:    return runGPUNeighborhoodOperatorImageFilterTest<2>(inFile, outFile);
Modules/Filtering/GPUImageFilterBase/test/itkGPUNeighborhoodOperatorImageFilterTest.cxx:    return runGPUNeighborhoodOperatorImageFilterTest<3>(inFile, outFile);
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:#ifndef itkGPUNeighborhoodOperatorImageFilter_hxx
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:#define itkGPUNeighborhoodOperatorImageFilter_hxx
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:GPUNeighborhoodOperatorImageFilter< TInputImage, TOutputImage, TOperatorValueType >
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:GPUNeighborhoodOperatorImageFilter<TInputImage, TOutputImage, TOperatorValueType, TParentImageFilter>::
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  GPUNeighborhoodOperatorImageFilter()
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  // Create GPU buffer to store neighborhood coefficient.
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  // This will be used as __constant memory in the GPU kernel.
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  m_NeighborhoodGPUBuffer = NeighborhoodGPUBufferType::New();
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:    itkExceptionMacro("GPUneighborhoodOperatorImageFilter supports 1/2/3D image.");
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  const char * GPUSource = GPUNeighborhoodOperatorImageFilter::GetOpenCLSource();
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  this->m_GPUKernelManager->LoadProgramFromString(GPUSource, defines.str().c_str());
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  m_NeighborhoodOperatorFilterGPUKernelHandle = this->m_GPUKernelManager->CreateKernel("NeighborOperatorFilter");
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:GPUNeighborhoodOperatorImageFilter<TInputImage, TOutputImage, TOperatorValueType, TParentImageFilter>::SetOperator(
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  /** Create GPU memory for operator coefficients */
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  m_NeighborhoodGPUBuffer->Initialize();
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  typename NeighborhoodGPUBufferType::IndexType index;
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  typename NeighborhoodGPUBufferType::SizeType  size;
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  const typename NeighborhoodGPUBufferType::RegionType region(index, size);
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  m_NeighborhoodGPUBuffer->SetRegions(region);
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  m_NeighborhoodGPUBuffer->Allocate();
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  ImageRegionIterator<NeighborhoodGPUBufferType> iit(m_NeighborhoodGPUBuffer,
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:                                                     m_NeighborhoodGPUBuffer->GetLargestPossibleRegion());
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:    iit.Set(static_cast<typename NeighborhoodGPUBufferType::PixelType>(*nit));
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  /** Mark GPU dirty */
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  m_NeighborhoodGPUBuffer->GetGPUDataManager()->SetGPUBufferDirty();
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:GPUNeighborhoodOperatorImageFilter<TInputImage, TOutputImage, TOperatorValueType, TParentImageFilter>::GPUGenerateData()
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  int kHd = m_NeighborhoodOperatorFilterGPUKernelHandle;
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  using GPUInputImage = typename itk::GPUTraits<TInputImage>::Type;
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  using GPUOutputImage = typename itk::GPUTraits<TOutputImage>::Type;
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  using GPUInputManagerType = GPUImageDataManager<GPUInputImage>;
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  using GPUOutputManagerType = GPUImageDataManager<GPUOutputImage>;
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  typename GPUInputImage::Pointer  inPtr = dynamic_cast<GPUInputImage *>(this->ProcessObject::GetInput(0));
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  typename GPUOutputImage::Pointer otPtr = dynamic_cast<GPUOutputImage *>(this->ProcessObject::GetOutput(0));
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  // typename GPUOutputImage::SizeType outSize = otPtr->GetLargestPossibleRegion().GetSize();
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  typename GPUOutputImage::SizeType outSize = otPtr->GetBufferedRegion().GetSize();
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  localSize[0] = localSize[1] = localSize[2] = OpenCLGetLocalBlockSize(ImageDim);
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  this->m_GPUKernelManager->template SetKernelArgWithImageAndBufferedRegion<GPUInputManagerType>(
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  this->m_GPUKernelManager->template SetKernelArgWithImageAndBufferedRegion<GPUOutputManagerType>(
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(kHd, argidx++, m_NeighborhoodGPUBuffer->GetGPUDataManager());
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:    this->m_GPUKernelManager->SetKernelArg(kHd, argidx++, sizeof(int), &(radius[i]));
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  //  this->m_GPUKernelManager->SetKernelArg(kHd, argidx++, sizeof(int), &(imgSize[i]) );
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.hxx:  this->m_GPUKernelManager->LaunchKernel(kHd, ImageDim, globalSize, localSize);
Modules/Filtering/GPUImageFilterBase/include/itkGPUBoxImageFilter.h:#ifndef itkGPUBoxImageFilter_h
Modules/Filtering/GPUImageFilterBase/include/itkGPUBoxImageFilter.h:#define itkGPUBoxImageFilter_h
Modules/Filtering/GPUImageFilterBase/include/itkGPUBoxImageFilter.h:#include "itkGPUImageToImageFilter.h"
Modules/Filtering/GPUImageFilterBase/include/itkGPUBoxImageFilter.h: * \class GPUBoxImageFilter
Modules/Filtering/GPUImageFilterBase/include/itkGPUBoxImageFilter.h: * \brief A base class for all the GPU filters working on a box neighborhood
Modules/Filtering/GPUImageFilterBase/include/itkGPUBoxImageFilter.h: * \ingroup ITKGPUImageFilterBase
Modules/Filtering/GPUImageFilterBase/include/itkGPUBoxImageFilter.h:class ITK_TEMPLATE_EXPORT GPUBoxImageFilter
Modules/Filtering/GPUImageFilterBase/include/itkGPUBoxImageFilter.h:  : public GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>
Modules/Filtering/GPUImageFilterBase/include/itkGPUBoxImageFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUBoxImageFilter);
Modules/Filtering/GPUImageFilterBase/include/itkGPUBoxImageFilter.h:  using Self = GPUBoxImageFilter;
Modules/Filtering/GPUImageFilterBase/include/itkGPUBoxImageFilter.h:  using GPUSuperclass = GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>;
Modules/Filtering/GPUImageFilterBase/include/itkGPUBoxImageFilter.h:  itkOverrideGetNameOfClassMacro(GPUBoxImageFilter);
Modules/Filtering/GPUImageFilterBase/include/itkGPUBoxImageFilter.h:  GPUBoxImageFilter() = default;
Modules/Filtering/GPUImageFilterBase/include/itkGPUBoxImageFilter.h:  ~GPUBoxImageFilter() override = default;
Modules/Filtering/GPUImageFilterBase/include/itkGPUBoxImageFilter.h:    GPUSuperclass::PrintSelf(os, indent);
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:#ifndef itkGPUNeighborhoodOperatorImageFilter_h
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:#define itkGPUNeighborhoodOperatorImageFilter_h
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:#include "itkGPUImage.h"
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:#include "itkGPUImageToImageFilter.h"
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:/** \class GPUNeighborhoodOperatorImageFilter
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h: * using the GPU.
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h: * This GPU filter calculates successive inner products between a single
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h: * \ingroup ITKGPUImageFilterBase
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:/** Create a helper GPU Kernel class for GPUNeighborhoodOperatorImageFilter */
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:itkGPUKernelClassMacro(GPUNeighborhoodOperatorImageFilterKernel);
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:class ITK_TEMPLATE_EXPORT GPUNeighborhoodOperatorImageFilter
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:  : public GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUNeighborhoodOperatorImageFilter);
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:  using Self = GPUNeighborhoodOperatorImageFilter;
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:  using GPUSuperclass = GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>;
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:  itkOverrideGetNameOfClassMacro(GPUNeighborhoodOperatorImageFilter);
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:  using NeighborhoodGPUBufferType = GPUImage<TOperatorValueType, Self::ImageDimension>;
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:  using OutputImageRegionType = typename GPUSuperclass::OutputImageRegionType;
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:  /** Get OpenCL Kernel source as a string, creates a GetOpenCLSource method */
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:  itkGetOpenCLSourceFromKernelMacro(GPUNeighborhoodOperatorImageFilterKernel);
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:  GPUNeighborhoodOperatorImageFilter();
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:  ~GPUNeighborhoodOperatorImageFilter() override = default;
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:  GPUGenerateData() override;
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:    GPUSuperclass::PrintSelf(os, indent);
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:  int m_NeighborhoodOperatorFilterGPUKernelHandle{};
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:  typename NeighborhoodGPUBufferType::Pointer m_NeighborhoodGPUBuffer{};
Modules/Filtering/GPUImageFilterBase/include/itkGPUNeighborhoodOperatorImageFilter.h:#  include "itkGPUNeighborhoodOperatorImageFilter.hxx"
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:#ifndef itkGPUCastImageFilter_h
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:#define itkGPUCastImageFilter_h
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:#include "itkGPUFunctorBase.h"
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:#include "itkGPUKernelManager.h"
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:#include "itkGPUUnaryFunctorImageFilter.h"
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:/** Create a helper GPU Kernel class for GPUCastImageFilter */
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:itkGPUKernelClassMacro(GPUCastImageFilterKernel);
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:/** \class GPUCastImageFilter
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h: * \brief GPU version of CastImageFilter.
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h: * \ingroup ITKGPUImageFilterBase
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:class ITK_TEMPLATE_EXPORT GPUCast : public GPUFunctorBase
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:  GPUCast() {}
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:  ~GPUCast() {}
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:  /** Setup GPU kernel arguments for this functor.
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:   * Returns current argument index to set additional arguments in the GPU kernel.
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:  SetGPUKernelArguments(GPUKernelManager::Pointer itkNotUsed(kernelManager), int itkNotUsed(kernelHandle))
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:class ITK_TEMPLATE_EXPORT GPUCastImageFilter
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:  : public GPUUnaryFunctorImageFilter<
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:      Functor::GPUCast<typename TInputImage::PixelType, typename TOutputImage::PixelType>,
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUCastImageFilter);
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:  using Self = GPUCastImageFilter;
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:  using GPUSuperclass =
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:    GPUUnaryFunctorImageFilter<TInputImage,
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:                               Functor::GPUCast<typename TInputImage::PixelType, typename TOutputImage::PixelType>,
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:  itkOverrideGetNameOfClassMacro(GPUCastImageFilter);
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:  GPUCastImageFilter();
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:  ~GPUCastImageFilter() override {}
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:  /** Unlike CPU version, GPU version of binary threshold filter is not
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:  GPUGenerateData() override;
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:#  include "itkGPUCastImageFilter.hxx"
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.h:#endif /* itkGPUCastImageFilter_h */
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.hxx:#ifndef itkGPUCastImageFilter_hxx
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.hxx:#define itkGPUCastImageFilter_hxx
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.hxx:#include "itkOpenCLUtil.h"
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.hxx:GPUCastImageFilter<TInputImage, TOutputImage>::GPUCastImageFilter()
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.hxx:    itkExceptionMacro("GPUCastImageFilter supports 1/2/3D image.");
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.hxx:  // OpenCL kernel source
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.hxx:  const char * GPUSource = GPUCastImageFilterKernel::GetOpenCLSource();
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.hxx:  this->m_GPUKernelManager->LoadProgramFromString(GPUSource, defines.str().c_str());
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.hxx:  this->m_UnaryFunctorImageFilterGPUKernelHandle = this->m_GPUKernelManager->CreateKernel("CastImageFilter");
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.hxx:GPUCastImageFilter<TInputImage, TOutputImage>::GPUGenerateData()
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.hxx:  itkDebugMacro("Calling GPUCastImageFilter::GPUGenerateData()");
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.hxx:  GPUSuperclass::GPUGenerateData();
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.hxx:  itkDebugMacro("GPUCastImageFilter::GPUGenerateData() finished");
Modules/Filtering/GPUImageFilterBase/include/itkGPUCastImageFilter.hxx:#endif /* itkGPUCastImageFilter_hxx */
Modules/Filtering/GPUImageFilterBase/CMakeLists.txt:project(ITKGPUImageFilterBase)
Modules/Filtering/GPUImageFilterBase/CMakeLists.txt:  set(ITK_USE_GPU
Modules/Filtering/GPUImageFilterBase/CMakeLists.txt:      CACHE BOOL "Enable OpenCL GPU support." FORCE)
Modules/Filtering/GPUImageFilterBase/CMakeLists.txt:  include(itkOpenCL)
Modules/Filtering/GPUImageFilterBase/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Filtering/GPUImageFilterBase/CMakeLists.txt:  set(ITKGPUImageFilterBase_LIBRARIES ITKGPUImageFilterBase)
Modules/Filtering/GPUImageFilterBase/CMakeLists.txt:  set(ITKGPUImageFilterBase_SYSTEM_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
Modules/Filtering/GPUImageFilterBase/CMakeLists.txt:  set(ITKGPUImageFilterBase_SYSTEM_LIBRARY_DIRS ${OPENCL_LIBRARIES})
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUBoxImageFilter.wrap:itk_wrap_include("itkGPUImage.h")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUBoxImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUNeighborhoodOperatorImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUNeighborhoodOperatorImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >, ${ITKT_${t}}")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUNeighborhoodOperatorImageFilter.wrap:itk_wrap_class("itk::GPUImageToImageFilter" POINTER)
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUNeighborhoodOperatorImageFilter.wrap:      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::NeighborhoodOperatorImageFilter< itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >, ${ITKT_${t}} >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUNeighborhoodOperatorImageFilter.wrap:itk_wrap_class("itk::GPUNeighborhoodOperatorImageFilter" POINTER)
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUNeighborhoodOperatorImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >, ${ITKT_${t}}")
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:# Testing GPU Neighborhood Operator Image Filter
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:InputGPUImageType = itk.GPUImage[InputPixelType, Dimension]
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:OutputGPUImageType = itk.GPUImage[OutputPixelType, Dimension]
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:input_gpu_image = itk.cast_image_filter(
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:    input_image, in_place=False, ttype=(InputImageType, InputGPUImageType)
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:input_gpu_image.UpdateBuffers()
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:GPUNeighborhoodFilterType = itk.GPUNeighborhoodOperatorImageFilter[
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:    InputGPUImageType, OutputGPUImageType, RealOutputPixelType
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:gpu_filter = GPUNeighborhoodFilterType.New()
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:gpu_timer = itk.TimeProbe()
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:gpu_timer.Start()
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:gpu_filter.SetInput(input_gpu_image)
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:gpu_filter.SetOperator(oper)
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:gpu_filter.Update()
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:gpu_filter.GetOutput().UpdateBuffers()  # synchronization point (GPU->CPU memcpy)
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:gpu_timer.Stop()
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:print(f"GPU NeighborhoodFilter took {gpu_timer.GetMean()} seconds.\n")
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:    gpu_filter.GetOutput(), ttype=(OutputGPUImageType, OutputImageType)
Modules/Filtering/GPUImageFilterBase/wrapping/test/itkGPUNeighborhoodOperatorImageFilterTest.py:output_gpu_image = gpu_filter.GetOutput()
Modules/Filtering/GPUImageFilterBase/wrapping/test/CMakeLists.txt:    itkGPUNeighborhoodOperatorImageFilterPythonTest2D
Modules/Filtering/GPUImageFilterBase/wrapping/test/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/itkGPUNeighborhoodOperatorImageFilterTest.py
Modules/Filtering/GPUImageFilterBase/wrapping/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuNeighborhoodOperatorImageFilterTest2DPython.mha
Modules/Filtering/GPUImageFilterBase/wrapping/test/CMakeLists.txt:    itkGPUNeighborhoodOperatorImageFilterPythonTest3D
Modules/Filtering/GPUImageFilterBase/wrapping/test/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/itkGPUNeighborhoodOperatorImageFilterTest.py
Modules/Filtering/GPUImageFilterBase/wrapping/test/CMakeLists.txt:    ${ITK_TEST_OUTPUT_DIR}/gpuNeighborhoodOperatorImageFilterTest3DPython.mha
Modules/Filtering/GPUImageFilterBase/wrapping/CMakeLists.txt:if(ITK_USE_GPU OR NOT ITK_SOURCE_DIR)
Modules/Filtering/GPUImageFilterBase/wrapping/CMakeLists.txt:  itk_wrap_module(ITKGPUImageFilterBase)
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:itk_wrap_include("itkGPUImage.h")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:itk_wrap_include("itkGPUCastImageFilter.h")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:itk_wrap_class("itk::Functor::GPUCast")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:                      "itk::Image< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} >")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:                          "itk::Image< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:                          "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} >")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:                          "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:                      "itk::Image< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} >")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::Image< ${ITKT_${t}${d}}, ${d} >")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} >")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:itk_wrap_class("itk::GPUImageToImageFilter" POINTER)
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:      "itk::Image< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::CastImageFilter< itk::Image< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:      "itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} >, itk::CastImageFilter< itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:      "itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::CastImageFilter< itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:          "itk::Image< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::CastImageFilter< itk::Image< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:          "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} >, itk::CastImageFilter< itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:          "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::CastImageFilter< itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:itk_wrap_class("itk::GPUInPlaceImageFilter" POINTER)
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:      "itk::Image< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::CastImageFilter< itk::Image< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:      "itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} >, itk::CastImageFilter< itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:      "itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::CastImageFilter< itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:          "itk::Image< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::CastImageFilter< itk::Image< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:          "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} >, itk::CastImageFilter< itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:          "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::CastImageFilter< itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:itk_wrap_class("itk::GPUUnaryFunctorImageFilter" POINTER)
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:      "I${ITKM_${to}}${d}GI${ITKM_${to}}${d}GPUCast${ITKM_${to}}${ITKM_${to}}CastImageFilter"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:      "itk::Image< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::Functor::GPUCast< ${ITKT_${to}}, ${ITKT_${to}} >, itk::CastImageFilter< itk::Image< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:      "itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} >, itk::Functor::GPUCast< ${ITKT_${to}}, ${ITKT_${to}} >, itk::CastImageFilter< itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:      "itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::Functor::GPUCast< ${ITKT_${to}}, ${ITKT_${to}} >, itk::CastImageFilter< itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:          "itk::Image< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::Functor::GPUCast< ${ITKT_${from}}, ${ITKT_${to}} >, itk::CastImageFilter< itk::Image< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:          "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} >, itk::Functor::GPUCast< ${ITKT_${from}}, ${ITKT_${to}} >, itk::CastImageFilter< itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:          "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::Functor::GPUCast< ${ITKT_${from}}, ${ITKT_${to}} >, itk::CastImageFilter< itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} > >"
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:itk_wrap_class("itk::GPUCastImageFilter" POINTER_WITH_SUPERCLASS)
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:                      "itk::Image< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} >")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:                          "itk::Image< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:                          "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} >")
Modules/Filtering/GPUImageFilterBase/wrapping/itkGPUCastImageFilter.wrap:                          "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Filtering/GPUImageFilterBase/src/GPUCastImageFilter.cl:// OpenCL implementation of itk::CastImageFilter
Modules/Filtering/GPUImageFilterBase/src/GPUCastImageFilter.cl:// Apple OpenCL 1.0 support function
Modules/Filtering/GPUImageFilterBase/src/GPUCastImageFilter.cl:  execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
Modules/Filtering/GPUImageFilterBase/src/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Filtering/GPUImageFilterBase/src/CMakeLists.txt:  set(ITKGPUImageFilterBase_SRCS)
Modules/Filtering/GPUImageFilterBase/src/CMakeLists.txt:  set(ITKGPUImageFilterBase_Kernels GPUCastImageFilter.cl GPUNeighborhoodOperatorImageFilter.cl)
Modules/Filtering/GPUImageFilterBase/src/CMakeLists.txt:  write_gpu_kernels("${ITKGPUImageFilterBase_Kernels}" ITKGPUImageFilterBase_SRCS)
Modules/Filtering/GPUImageFilterBase/src/CMakeLists.txt:  itk_module_add_library(ITKGPUImageFilterBase ${ITKGPUImageFilterBase_SRCS})
Modules/Filtering/GPUImageFilterBase/src/CMakeLists.txt:  target_link_libraries(ITKGPUImageFilterBase LINK_PUBLIC ${OPENCL_LIBRARIES})
Modules/Filtering/GPUImageFilterBase/src/GPUNeighborhoodOperatorImageFilter.cl:     execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
Modules/Core/GPUCommon/CMake/itkCheckHasBlocks.cxx:// Check if blocks can be used for the GPU implementation.
Modules/Core/GPUCommon/itk-module.cmake:with the GPU.  This includes base classes for the GPU image filters, some
Modules/Core/GPUCommon/itk-module.cmake:OpenCL utilities, and classes to manage the interface between the CPU and
Modules/Core/GPUCommon/itk-module.cmake:the GPU.  These classes manage the GPU kernel, transferring the data to and
Modules/Core/GPUCommon/itk-module.cmake:from the GPU, and managing the GPU contexts.")
Modules/Core/GPUCommon/itk-module.cmake:  ITKGPUCommon
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx: * Test program for itkGPUImage class.
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx: * This program shows how to use GPU image and GPU program.
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx://#include "pathToOpenCLSourceCode.h"
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:#include "itkGPUImage.h"
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:#include "itkGPUKernelManager.h"
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:#include "itkGPUContextManager.h"
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:#include "itkGPUImageOps.h"
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:using ItkImage1f = itk::GPUImage<float, 2>;
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:itkGPUImageTest(int argc, char * argv[])
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  // create GPUImage
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  // create GPU program object
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  itk::GPUKernelManager::Pointer kernelManager = itk::GPUKernelManager::New();
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  const char * GPUSource = itk::GPUImageOps::GetOpenCLSource();
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  kernelManager->LoadProgramFromString(GPUSource, "#define PIXELTYPE float\n");
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  std::cout << "Before GPU kernel execution" << std::endl;
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  kernelManager->SetKernelArgWithImage(kernel_add, 0, srcA->GetGPUDataManager());
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  kernelManager->SetKernelArgWithImage(kernel_add, 1, srcB->GetGPUDataManager());
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  kernelManager->SetKernelArgWithImage(kernel_add, 2, dest->GetGPUDataManager());
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  std::cout << "After GPU kernel execution" << std::endl;
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  std::cout << "Before GPU kernel execution" << std::endl;
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  kernelManager->SetKernelArgWithImage(kernel_mult, 0, srcA->GetGPUDataManager());
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  kernelManager->SetKernelArgWithImage(kernel_mult, 1, srcB->GetGPUDataManager());
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  kernelManager->SetKernelArgWithImage(kernel_mult, 2, dest->GetGPUDataManager());
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  std::cout << "After GPU kernel execution" << std::endl;
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  // Change Command Queue if more than one GPU device exists
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  itk::GPUContextManager * contextManager = itk::GPUContextManager::GetInstance();
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:    std::cout << "More than one GPU device available, switching command queues." << std::endl;
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:    std::cout << "Only one GPU device available, using same command queue." << std::endl;
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  std::cout << "Before GPU kernel execution" << std::endl;
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  kernelManager->SetKernelArgWithImage(kernel_sub, 0, srcA->GetGPUDataManager());
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  kernelManager->SetKernelArgWithImage(kernel_sub, 1, srcB->GetGPUDataManager());
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  kernelManager->SetKernelArgWithImage(kernel_sub, 2, dest->GetGPUDataManager());
Modules/Core/GPUCommon/test/itkGPUImageTest.cxx:  std::cout << "After GPU kernel execution" << std::endl;
Modules/Core/GPUCommon/test/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Core/GPUCommon/test/CMakeLists.txt:  set(ITKGPUCommon-tests itkGPUImageTest.cxx itkGPUReductionTest.cxx)
Modules/Core/GPUCommon/test/CMakeLists.txt:  createtestdriver(ITKGPUCommon "${ITKGPUCommon-Test_LIBRARIES}" "${ITKGPUCommon-tests}")
Modules/Core/GPUCommon/test/CMakeLists.txt:    itkGPUImageTest
Modules/Core/GPUCommon/test/CMakeLists.txt:    ITKGPUCommonTestDriver
Modules/Core/GPUCommon/test/CMakeLists.txt:    itkGPUImageTest)
Modules/Core/GPUCommon/test/CMakeLists.txt:    itkGPUReductionTest
Modules/Core/GPUCommon/test/CMakeLists.txt:    ITKGPUCommonTestDriver
Modules/Core/GPUCommon/test/CMakeLists.txt:    itkGPUReductionTest)
Modules/Core/GPUCommon/test/itkGPUReductionTest.cxx: * Test program for itkGPUImage class.
Modules/Core/GPUCommon/test/itkGPUReductionTest.cxx: * This program shows how to use GPU image and GPU program.
Modules/Core/GPUCommon/test/itkGPUReductionTest.cxx://#include "pathToOpenCLSourceCode.h"
Modules/Core/GPUCommon/test/itkGPUReductionTest.cxx:#include "itkGPUImage.h"
Modules/Core/GPUCommon/test/itkGPUReductionTest.cxx:#include "itkGPUReduction.h"
Modules/Core/GPUCommon/test/itkGPUReductionTest.cxx:itkGPUReductionTest(int argc, char * argv[])
Modules/Core/GPUCommon/test/itkGPUReductionTest.cxx:  itk::GPUReduction<ElementType>::Pointer summer = itk::GPUReduction<ElementType>::New();
Modules/Core/GPUCommon/test/itkGPUReductionTest.cxx:  summer->AllocateGPUInputBuffer(h_idata);
Modules/Core/GPUCommon/test/itkGPUReductionTest.cxx:  summer->GPUGenerateData();
Modules/Core/GPUCommon/test/itkGPUReductionTest.cxx:  int GPUsum = summer->GetGPUResult();
Modules/Core/GPUCommon/test/itkGPUReductionTest.cxx:  summer->ReleaseGPUInputBuffer();
Modules/Core/GPUCommon/test/itkGPUReductionTest.cxx:  if (GPUsum == static_cast<int>(numPixels))
Modules/Core/GPUCommon/test/itkGPUReductionTest.cxx:    std::cout << "GPU reduction to sum passed, sum = " << GPUsum << ", numPixels = " << numPixels << std::endl;
Modules/Core/GPUCommon/test/itkGPUReductionTest.cxx:    std::cout << "Expected sum to be " << numPixels << ", GPUReduction computed " << GPUsum << " which is wrong."
Modules/Core/GPUCommon/test/itkGPUReductionTest.cxx:    std::cout << "Expected CPU sum to be " << numPixels << ", GPUReduction computed " << CPUsum << " which is wrong."
Modules/Core/GPUCommon/test/itkGPUReductionTest.cxx:  summer = nullptr;                                         // explicit GPU object destruction test
Modules/Core/GPUCommon/test/itkGPUReductionTest.cxx:  itk::GPUContextManager::GetInstance()->DestroyInstance(); // GPUContextManager singleton destruction test
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:#ifndef itkGPUReduction_hxx
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:#define itkGPUReduction_hxx
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:GPUReduction<TElement>::GPUReduction()
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  // Prepare GPU opencl program
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  m_GPUKernelManager = GPUKernelManager::New();
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  m_GPUDataManager = nullptr;
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  m_ReduceGPUKernelHandle = 0;
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  m_TestGPUKernelHandle = 0;
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:GPUReduction<TElement>::~GPUReduction()
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  this->ReleaseGPUInputBuffer();
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:GPUReduction<TElement>::PrintSelf(std::ostream & os, Indent indent) const
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  itkPrintSelfObjectMacro(GPUKernelManager);
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  itkPrintSelfObjectMacro(GPUDataManager);
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  os << indent << "ReduceGPUKernelHandle: " << m_ReduceGPUKernelHandle << std::endl;
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  os << indent << "TestGPUKernelHandle: " << m_TestGPUKernelHandle << std::endl;
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  os << indent << "GPUResult: " << static_cast<typename NumericTraits<TElement>::PrintType>(m_GPUResult) << std::endl;
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:GPUReduction<TElement>::NextPow2(unsigned int x)
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:GPUReduction<TElement>::isPow2(unsigned int x)
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:GPUReduction<TElement>::GetNumBlocksAndThreads(int   whichKernel,
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:GPUReduction<TElement>::GetReductionKernel(int whichKernel, int blockSize, int isPowOf2)
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  const char * GPUSource = GPUReduction::GetOpenCLSource();
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  this->m_GPUKernelManager->LoadProgramFromString(GPUSource, defines.str().c_str());
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  unsigned int handle = this->m_GPUKernelManager->CreateKernel(kernelName.str().c_str());
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  cl_int ciErrNum = this->m_GPUKernelManager->GetKernelWorkGroupInfo(handle, CL_KERNEL_WORK_GROUP_SIZE, &wgSize);
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  OpenCLCheckError(ciErrNum, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  // this->m_GPUKernelManager->ReleaseProgram();
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:GPUReduction<TElement>::AllocateGPUInputBuffer(TElement * h_idata)
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  m_GPUDataManager = GPUDataManager::New();
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  m_GPUDataManager->SetBufferSize(bytes);
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  m_GPUDataManager->SetCPUBufferPointer(h_idata);
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  m_GPUDataManager->Allocate();
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:    m_GPUDataManager->SetGPUDirtyFlag(true);
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:GPUReduction<TElement>::ReleaseGPUInputBuffer()
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  if (m_GPUDataManager == (GPUDataPointer) nullptr)
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  m_GPUDataManager->Initialize();
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:GPUReduction<TElement>::RandomTest()
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  this->AllocateGPUInputBuffer(h_idata);
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  TElement gpu_result = this->GPUGenerateData();
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  std::cout << "GPU result = " << gpu_result << std::endl << std::flush;
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  this->ReleaseGPUInputBuffer();
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:GPUReduction<TElement>::InitializeKernel(unsigned int size)
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  //   m_TestGPUKernelHandle = this->GetReductionKernel(6, 64, 1);
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  // m_GPUKernelManager->ReleaseKernel(kernelHandle);
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  m_ReduceGPUKernelHandle = this->GetReductionKernel(whichKernel, numThreads, isPow2(size));
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:GPUReduction<TElement>::GPUGenerateData()
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  GPUDataPointer odata = GPUDataManager::New();
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  m_GPUResult = 0;
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  m_GPUResult = this->GPUReduce(size,
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:                                m_GPUDataManager,
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  return m_GPUResult;
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:GPUReduction<TElement>::GPUReduce(cl_int         n,
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:                                  GPUDataPointer idata,
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:                                  GPUDataPointer odata)
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  TElement gpu_result = 0;
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(m_ReduceGPUKernelHandle, argidx++, idata);
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(m_ReduceGPUKernelHandle, argidx++, odata);
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  this->m_GPUKernelManager->SetKernelArg(m_ReduceGPUKernelHandle, argidx++, sizeof(cl_int), &n);
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  this->m_GPUKernelManager->SetKernelArg(m_ReduceGPUKernelHandle, argidx++, sizeof(TElement) * numThreads, nullptr);
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  gpu_result = 0;
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  this->m_GPUKernelManager->LaunchKernel(m_ReduceGPUKernelHandle, 1, globalSize, localSize);
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:    gpu_result += h_odata[i];
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:  return gpu_result;
Modules/Core/GPUCommon/include/itkGPUReduction.hxx:GPUReduction<TElement>::CPUGenerateData(TElement * data, int size)
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:#ifndef itkGPUImageToImageFilter_h
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:#define itkGPUImageToImageFilter_h
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:#include "itkGPUKernelManager.h"
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:/** \class GPUImageToImageFilter
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h: * \brief class to abstract the behaviour of the GPU filters.
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h: * GPUImageToImageFilter is the GPU version of ImageToImageFilter.
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h: * This class can accept both CPU and GPU image as input and output,
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h: * and apply filter accordingly. If GPU is available for use, then
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h: * GPUGenerateData() is called. Otherwise, GenerateData() in the
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h: * \ingroup ITKGPUCommon
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:class ITK_TEMPLATE_EXPORT GPUImageToImageFilter : public TParentImageFilter
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUImageToImageFilter);
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:  using Self = GPUImageToImageFilter;
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:  itkOverrideGetNameOfClassMacro(GPUImageToImageFilter);
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:  // macro to set if GPU is used
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:  itkSetMacro(GPUEnabled, bool);
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:  itkGetConstMacro(GPUEnabled, bool);
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:  itkBooleanMacro(GPUEnabled);
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:  GraftOutput(typename itk::GPUTraits<TOutputImage>::Type * output);
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:  GraftOutput(const DataObjectIdentifierType & key, typename itk::GPUTraits<TOutputImage>::Type * output);
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:  GPUImageToImageFilter();
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:  ~GPUImageToImageFilter() override;
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:  GPUGenerateData()
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:  // GPU kernel manager
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:  typename GPUKernelManager::Pointer m_GPUKernelManager{};
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:  // GPU kernel handle - kernel should be defined in specific filter (not in the
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:  bool m_GPUEnabled{ true };
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.h:#  include "itkGPUImageToImageFilter.hxx"
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:#ifndef itkGPUImageDataManager_h
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:#define itkGPUImageDataManager_h
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:#include "itkOpenCLUtil.h"
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:#include "itkGPUDataManager.h"
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:#include "itkGPUContextManager.h"
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:class GPUImage;
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h: * \class GPUImageDataManager
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h: * DataManager for GPUImage. This class will take care of data synchronization
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h: * between CPU Image and GPU Image.
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h: * \ingroup ITKGPUCommon
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:class ITK_TEMPLATE_EXPORT GPUImageDataManager : public GPUDataManager
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:  // allow GPUKernelManager to access GPU buffer pointer
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:  friend class GPUKernelManager;
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:  friend class GPUImage<typename ImageType::PixelType, ImageType::ImageDimension>;
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUImageDataManager);
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:  using Self = GPUImageDataManager;
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:  using Superclass = GPUDataManager;
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:  itkOverrideGetNameOfClassMacro(GPUImageDataManager);
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:  itkGetModifiableObjectMacro(GPUBufferedRegionIndex, GPUDataManager);
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:  itkGetModifiableObjectMacro(GPUBufferedRegionSize, GPUDataManager);
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:  /** actual GPU->CPU memory copy takes place here */
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:  /** actual CPU->GPU memory copy takes place here */
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:  MakeGPUBufferUpToDate();
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:  GPUImageDataManager() = default;
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:  ~GPUImageDataManager() override = default;
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:  typename GPUDataManager::Pointer m_GPUBufferedRegionIndex{};
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:  typename GPUDataManager::Pointer m_GPUBufferedRegionSize{};
Modules/Core/GPUCommon/include/itkGPUImageDataManager.h:#  include "itkGPUImageDataManager.hxx"
Modules/Core/GPUCommon/include/itkGPUImage.hxx:#ifndef itkGPUImage_hxx
Modules/Core/GPUCommon/include/itkGPUImage.hxx:#define itkGPUImage_hxx
Modules/Core/GPUCommon/include/itkGPUImage.hxx:GPUImage<TPixel, VImageDimension>::GPUImage()
Modules/Core/GPUCommon/include/itkGPUImage.hxx:  m_DataManager = GPUImageDataManager<GPUImage<TPixel, VImageDimension>>::New();
Modules/Core/GPUCommon/include/itkGPUImage.hxx:GPUImage<TPixel, VImageDimension>::~GPUImage() = default;
Modules/Core/GPUCommon/include/itkGPUImage.hxx:GPUImage<TPixel, VImageDimension>::Allocate(bool initialize)
Modules/Core/GPUCommon/include/itkGPUImage.hxx:  // allocate GPU memory
Modules/Core/GPUCommon/include/itkGPUImage.hxx:  /* prevent unnecessary copy from CPU to GPU at the beginning */
Modules/Core/GPUCommon/include/itkGPUImage.hxx:GPUImage<TPixel, VImageDimension>::Initialize()
Modules/Core/GPUCommon/include/itkGPUImage.hxx:  // GPU image initialize
Modules/Core/GPUCommon/include/itkGPUImage.hxx:  /* prevent unnecessary copy from CPU to GPU at the beginning */
Modules/Core/GPUCommon/include/itkGPUImage.hxx:GPUImage<TPixel, VImageDimension>::FillBuffer(const TPixel & value)
Modules/Core/GPUCommon/include/itkGPUImage.hxx:  m_DataManager->SetGPUBufferDirty();
Modules/Core/GPUCommon/include/itkGPUImage.hxx:GPUImage<TPixel, VImageDimension>::SetPixel(const IndexType & index, const TPixel & value)
Modules/Core/GPUCommon/include/itkGPUImage.hxx:  m_DataManager->SetGPUBufferDirty();
Modules/Core/GPUCommon/include/itkGPUImage.hxx:GPUImage<TPixel, VImageDimension>::GetPixel(const IndexType & index) const
Modules/Core/GPUCommon/include/itkGPUImage.hxx:GPUImage<TPixel, VImageDimension>::GetPixel(const IndexType & index)
Modules/Core/GPUCommon/include/itkGPUImage.hxx:  m_DataManager->SetGPUBufferDirty();
Modules/Core/GPUCommon/include/itkGPUImage.hxx:TPixel & GPUImage<TPixel, VImageDimension>::operator[](const IndexType & index)
Modules/Core/GPUCommon/include/itkGPUImage.hxx:  m_DataManager->SetGPUBufferDirty();
Modules/Core/GPUCommon/include/itkGPUImage.hxx:const TPixel & GPUImage<TPixel, VImageDimension>::operator[](const IndexType & index) const
Modules/Core/GPUCommon/include/itkGPUImage.hxx:GPUImage<TPixel, VImageDimension>::SetPixelContainer(PixelContainer * container)
Modules/Core/GPUCommon/include/itkGPUImage.hxx:  m_DataManager->SetGPUDirtyFlag(true);
Modules/Core/GPUCommon/include/itkGPUImage.hxx:GPUImage<TPixel, VImageDimension>::UpdateBuffers()
Modules/Core/GPUCommon/include/itkGPUImage.hxx:  m_DataManager->UpdateGPUBuffer();
Modules/Core/GPUCommon/include/itkGPUImage.hxx:GPUImage<TPixel, VImageDimension>::GetBufferPointer()
Modules/Core/GPUCommon/include/itkGPUImage.hxx:   * Always set GPU dirty (even though pixel values are not modified)
Modules/Core/GPUCommon/include/itkGPUImage.hxx:  m_DataManager->SetGPUBufferDirty();
Modules/Core/GPUCommon/include/itkGPUImage.hxx:GPUImage<TPixel, VImageDimension>::GetBufferPointer() const
Modules/Core/GPUCommon/include/itkGPUImage.hxx:GPUDataManager *
Modules/Core/GPUCommon/include/itkGPUImage.hxx:GPUImage<TPixel, VImageDimension>::GetGPUDataManager()
Modules/Core/GPUCommon/include/itkGPUImage.hxx:GPUImage<TPixel, VImageDimension>::Graft(const Self * data)
Modules/Core/GPUCommon/include/itkGPUImage.hxx:  using GPUImageDataManagerType = GPUImageDataManager<GPUImage>;
Modules/Core/GPUCommon/include/itkGPUImage.hxx:  auto * ptr = const_cast<GPUImageDataManagerType *>(data->GetDataManager());
Modules/Core/GPUCommon/include/itkGPUImage.hxx:  // call GPU data graft function
Modules/Core/GPUCommon/include/itkGPUImage.hxx:  // Synchronize timestamp of GPUImage and GPUDataManager
Modules/Core/GPUCommon/include/itkGPUImage.hxx:GPUImage<TPixel, VImageDimension>::Graft(const DataObject * data)
Modules/Core/GPUCommon/include/itkGPUImage.hxx:    itkExceptionMacro("itk::GPUImage::Graft() cannot cast " << typeid(data).name() << " to "
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:#ifndef itkGPUImageToImageFilter_hxx
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:#define itkGPUImageToImageFilter_hxx
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GPUImageToImageFilter()
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  m_GPUKernelManager = GPUKernelManager::New();
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::~GPUImageToImageFilter() = default;
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::PrintSelf(std::ostream & os, Indent indent) const
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  os << indent << "GPU: " << (m_GPUEnabled ? "Enabled" : "Disabled") << std::endl;
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GenerateData()
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  if (!m_GPUEnabled) // call CPU update function
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  else // call GPU update function
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:    GPUGenerateData();
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GraftOutput(
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  typename itk::GPUTraits<TOutputImage>::Type * output)
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  using GPUOutputImage = typename itk::GPUTraits<TOutputImage>::Type;
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  typename GPUOutputImage::Pointer gpuImage = dynamic_cast<GPUOutputImage *>(this->GetOutput());
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  gpuImage->Graft(output);
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GraftOutput(DataObject * output)
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  using GPUOutputImage = typename itk::GPUTraits<TOutputImage>::Type;
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  auto * gpuImage = dynamic_cast<GPUOutputImage *>(output);
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  if (gpuImage)
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:    this->GraftOutput(gpuImage);
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:    itkExceptionMacro("itk::GPUImageToImageFilter::GraftOutput() cannot cast " << typeid(output).name() << " to "
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:                                                                               << typeid(GPUOutputImage *).name());
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GraftOutput(
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  typename itk::GPUTraits<TOutputImage>::Type * output)
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  using GPUOutputImage = typename itk::GPUTraits<TOutputImage>::Type;
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  typename GPUOutputImage::Pointer gpuImage = dynamic_cast<GPUOutputImage *>(this->ProcessObject::GetOutput(key));
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  gpuImage->Graft(output);
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GraftOutput(const DataObjectIdentifierType & key,
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  using GPUOutputImage = typename itk::GPUTraits<TOutputImage>::Type;
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  auto * gpuImage = dynamic_cast<GPUOutputImage *>(output);
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:  if (gpuImage)
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:    this->GraftOutput(key, gpuImage);
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:    itkExceptionMacro("itk::GPUImageToImageFilter::GraftOutput() cannot cast " << typeid(output).name() << " to "
Modules/Core/GPUCommon/include/itkGPUImageToImageFilter.hxx:                                                                               << typeid(GPUOutputImage *).name());
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:#ifndef itkGPUUnaryFunctorImageFilter_hxx
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:#define itkGPUUnaryFunctorImageFilter_hxx
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:GPUUnaryFunctorImageFilter<TInputImage, TOutputImage, TFunction, TParentImageFilter>::GenerateOutputInformation()
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:GPUUnaryFunctorImageFilter<TInputImage, TOutputImage, TFunction, TParentImageFilter>::GPUGenerateData()
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:  // Applying functor using GPU kernel
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:  using GPUInputImage = typename itk::GPUTraits<TInputImage>::Type;
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:  using GPUOutputImage = typename itk::GPUTraits<TOutputImage>::Type;
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:  typename GPUInputImage::Pointer  inPtr = dynamic_cast<GPUInputImage *>(this->ProcessObject::GetInput(0));
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:  typename GPUOutputImage::Pointer otPtr = dynamic_cast<GPUOutputImage *>(this->ProcessObject::GetOutput(0));
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:  typename GPUOutputImage::SizeType outSize = otPtr->GetLargestPossibleRegion().GetSize();
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:  localSize[0] = localSize[1] = localSize[2] = OpenCLGetLocalBlockSize(ImageDim);
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:    (this->GetFunctor()).SetGPUKernelArguments(this->m_GPUKernelManager, m_UnaryFunctorImageFilterGPUKernelHandle);
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:    m_UnaryFunctorImageFilterGPUKernelHandle, argidx++, inPtr->GetGPUDataManager());
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:    m_UnaryFunctorImageFilterGPUKernelHandle, argidx++, otPtr->GetGPUDataManager());
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:    this->m_GPUKernelManager->SetKernelArg(
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:      m_UnaryFunctorImageFilterGPUKernelHandle, argidx++, sizeof(int), &(imgSize[i]));
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.hxx:  this->m_GPUKernelManager->LaunchKernel(m_UnaryFunctorImageFilterGPUKernelHandle, ImageDim, globalSize, localSize);
Modules/Core/GPUCommon/include/itkGPUContextManager.h:#ifndef itkGPUContextManager_h
Modules/Core/GPUCommon/include/itkGPUContextManager.h:#define itkGPUContextManager_h
Modules/Core/GPUCommon/include/itkGPUContextManager.h:#include "itkOpenCLUtil.h"
Modules/Core/GPUCommon/include/itkGPUContextManager.h:/** \class GPUContextManager
Modules/Core/GPUCommon/include/itkGPUContextManager.h: * \brief Singleton class to store the GPU context.
Modules/Core/GPUCommon/include/itkGPUContextManager.h: * \ingroup ITKGPUCommon
Modules/Core/GPUCommon/include/itkGPUContextManager.h:class GPUContextManager : public LightObject
Modules/Core/GPUCommon/include/itkGPUContextManager.h:  static GPUContextManager *
Modules/Core/GPUCommon/include/itkGPUContextManager.h:  GPUContextManager();
Modules/Core/GPUCommon/include/itkGPUContextManager.h:  ~GPUContextManager() override;
Modules/Core/GPUCommon/include/itkGPUContextManager.h:  static GPUContextManager * m_Instance;
Modules/Core/GPUCommon/include/itkGPUReduction.h:#ifndef itkGPUReduction_h
Modules/Core/GPUCommon/include/itkGPUReduction.h:#define itkGPUReduction_h
Modules/Core/GPUCommon/include/itkGPUReduction.h:#include "itkGPUDataManager.h"
Modules/Core/GPUCommon/include/itkGPUReduction.h:#include "itkGPUKernelManager.h"
Modules/Core/GPUCommon/include/itkGPUReduction.h:#include "itkOpenCLUtil.h"
Modules/Core/GPUCommon/include/itkGPUReduction.h:/** Create a helper GPU Kernel class for GPUReduction */
Modules/Core/GPUCommon/include/itkGPUReduction.h:itkGPUKernelClassMacro(GPUReductionKernel);
Modules/Core/GPUCommon/include/itkGPUReduction.h: * \class GPUReduction
Modules/Core/GPUCommon/include/itkGPUReduction.h: * \ingroup ITKGPUCommon
Modules/Core/GPUCommon/include/itkGPUReduction.h:class ITK_TEMPLATE_EXPORT GPUReduction : public Object
Modules/Core/GPUCommon/include/itkGPUReduction.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUReduction);
Modules/Core/GPUCommon/include/itkGPUReduction.h:  using Self = GPUReduction;
Modules/Core/GPUCommon/include/itkGPUReduction.h:  itkOverrideGetNameOfClassMacro(GPUReduction);
Modules/Core/GPUCommon/include/itkGPUReduction.h:  using GPUDataPointer = GPUDataManager::Pointer;
Modules/Core/GPUCommon/include/itkGPUReduction.h:  itkGetMacro(GPUDataManager, GPUDataPointer);
Modules/Core/GPUCommon/include/itkGPUReduction.h:  itkGetMacro(GPUResult, TElement);
Modules/Core/GPUCommon/include/itkGPUReduction.h:  /** Get OpenCL Kernel source as a string, creates a GetOpenCLSource method */
Modules/Core/GPUCommon/include/itkGPUReduction.h:  itkGetOpenCLSourceFromKernelMacro(GPUReductionKernel);
Modules/Core/GPUCommon/include/itkGPUReduction.h:  AllocateGPUInputBuffer(TElement * h_idata = nullptr);
Modules/Core/GPUCommon/include/itkGPUReduction.h:  ReleaseGPUInputBuffer();
Modules/Core/GPUCommon/include/itkGPUReduction.h:  GPUGenerateData();
Modules/Core/GPUCommon/include/itkGPUReduction.h:  GPUReduce(cl_int         n,
Modules/Core/GPUCommon/include/itkGPUReduction.h:            GPUDataPointer idata,
Modules/Core/GPUCommon/include/itkGPUReduction.h:            GPUDataPointer odata);
Modules/Core/GPUCommon/include/itkGPUReduction.h:  GPUReduction();
Modules/Core/GPUCommon/include/itkGPUReduction.h:  ~GPUReduction() override;
Modules/Core/GPUCommon/include/itkGPUReduction.h:  /** GPU kernel manager for GPUFiniteDifferenceFunction class */
Modules/Core/GPUCommon/include/itkGPUReduction.h:  GPUKernelManager::Pointer m_GPUKernelManager{};
Modules/Core/GPUCommon/include/itkGPUReduction.h:  GPUDataPointer            m_GPUDataManager{};
Modules/Core/GPUCommon/include/itkGPUReduction.h:  /* GPU kernel handle for GPUComputeUpdate */
Modules/Core/GPUCommon/include/itkGPUReduction.h:  int m_ReduceGPUKernelHandle{};
Modules/Core/GPUCommon/include/itkGPUReduction.h:  int m_TestGPUKernelHandle{};
Modules/Core/GPUCommon/include/itkGPUReduction.h:  TElement m_GPUResult, m_CPUResult{};
Modules/Core/GPUCommon/include/itkGPUReduction.h:#  include "itkGPUReduction.hxx"
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:#ifndef itkGPUUnaryFunctorImageFilter_h
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:#define itkGPUUnaryFunctorImageFilter_h
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:#include "itkGPUFunctorBase.h"
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:#include "itkGPUInPlaceImageFilter.h"
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:/** \class GPUUnaryFunctorImageFilter
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h: * \brief Implements pixel-wise generic operation on one image using the GPU.
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h: * GPU version of unary functor image filter.
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h: * GPU Functor handles parameter setup for the GPU kernel.
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h: * \ingroup   ITKGPUCommon
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:class ITK_TEMPLATE_EXPORT GPUUnaryFunctorImageFilter
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:  : public GPUInPlaceImageFilter<TInputImage, TOutputImage, TParentImageFilter>
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUUnaryFunctorImageFilter);
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:  using Self = GPUUnaryFunctorImageFilter;
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:  using GPUSuperclass = GPUInPlaceImageFilter<TInputImage, TOutputImage>;
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:  itkOverrideGetNameOfClassMacro(GPUUnaryFunctorImageFilter);
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:  GPUUnaryFunctorImageFilter() = default;
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:  ~GPUUnaryFunctorImageFilter() override = default;
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:  GPUGenerateData() override;
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:  /** GPU kernel handle is defined here instead of in the child class
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:   * because GPUGenerateData() in this base class is used. */
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:  int m_UnaryFunctorImageFilterGPUKernelHandle{};
Modules/Core/GPUCommon/include/itkGPUUnaryFunctorImageFilter.h:#  include "itkGPUUnaryFunctorImageFilter.hxx"
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:// GPU Kernel Manager Class
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:#ifndef itkGPUKernelManager_h
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:#define itkGPUKernelManager_h
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:#include "itkOpenCLUtil.h"
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:#include "itkGPUImage.h"
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:#include "itkGPUContextManager.h"
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:#include "itkGPUDataManager.h"
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:/** \class GPUKernelManager
Modules/Core/GPUCommon/include/itkGPUKernelManager.h: * \brief GPU kernel manager implemented using OpenCL.
Modules/Core/GPUCommon/include/itkGPUKernelManager.h: * This class is responsible for managing the GPU kernel and
Modules/Core/GPUCommon/include/itkGPUKernelManager.h: * \ingroup ITKGPUCommon
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:class GPUKernelManager : public LightObject
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUKernelManager);
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:    GPUDataManager::Pointer m_GPUDataManager;
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:  using Self = GPUKernelManager;
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:  itkOverrideGetNameOfClassMacro(GPUKernelManager);
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:  SetKernelArgWithImage(int kernelIdx, cl_uint argIdx, GPUDataManager * manager);
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:  /** Pass to GPU both the pixel buffer and the buffered region. */
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:  // template< typename TGPUImageDataManager >
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:  //  typename TGPUImageDataManager::Pointer manager);
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:  template <typename TGPUImageDataManager>
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:  SetKernelArgWithImageAndBufferedRegion(int kernelIdx, cl_uint & argIdx, TGPUImageDataManager * manager)
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:    errid = clSetKernelArg(m_KernelContainer[kernelIdx], argIdx, sizeof(cl_mem), manager->GetGPUBufferPointer());
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:    m_KernelArgumentReady[kernelIdx][argIdx].m_GPUDataManager = manager;
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:    // this->SetKernelArg(kernelIdx, argIdx++, sizeof(int), &(TGPUImageDataManager::ImageDimension) );
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:                           manager->GetModifiableGPUBufferedRegionIndex()->GetGPUBufferPointer());
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:    m_KernelArgumentReady[kernelIdx][argIdx].m_GPUDataManager = manager->GetModifiableGPUBufferedRegionIndex();
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:                           manager->GetModifiableGPUBufferedRegionSize()->GetGPUBufferPointer());
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:    m_KernelArgumentReady[kernelIdx][argIdx].m_GPUDataManager = manager->GetModifiableGPUBufferedRegionSize();
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:  GPUKernelManager();
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:  ~GPUKernelManager() override;
Modules/Core/GPUCommon/include/itkGPUKernelManager.h:  GPUContextManager * m_Manager{};
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h:#ifndef itkGPUInPlaceImageFilter_h
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h:#define itkGPUInPlaceImageFilter_h
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h:#include "itkGPUImageToImageFilter.h"
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h:/** \class GPUInPlaceImageFilter
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h: * \brief Base class for GPU filters that take an image as input and overwrite that image as the output
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h: * This class is the base class for GPU inplace filter. The template parameter for parent class type
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h: * must be InPlaceImageFilter type so that the GPU superclass of this class can be correctly defined
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h: * (NOTE: TParentImageFilter::Superclass is used to define GPUImageToImageFilter class).
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h: * \ingroup ITKGPUCommon
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h:class ITK_TEMPLATE_EXPORT GPUInPlaceImageFilter
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h:  : public GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUInPlaceImageFilter);
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h:  using Self = GPUInPlaceImageFilter;
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h:  using GPUSuperclass = GPUImageToImageFilter<TInputImage, TOutputImage, TParentImageFilter>;
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h:  itkOverrideGetNameOfClassMacro(GPUInPlaceImageFilter);
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h:  using OutputImageType = typename GPUSuperclass::OutputImageType;
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h:  using OutputImagePointer = typename GPUSuperclass::OutputImagePointer;
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h:  using OutputImageRegionType = typename GPUSuperclass::OutputImageRegionType;
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h:  using OutputImagePixelType = typename GPUSuperclass::OutputImagePixelType;
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h:  GPUInPlaceImageFilter();
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h:  ~GPUInPlaceImageFilter() override;
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.h:#  include "itkGPUInPlaceImageFilter.hxx"
Modules/Core/GPUCommon/include/itkOpenCLUtil.h:#ifndef itkOpenCLUtil_h
Modules/Core/GPUCommon/include/itkOpenCLUtil.h:#define itkOpenCLUtil_h
Modules/Core/GPUCommon/include/itkOpenCLUtil.h:#ifndef CL_TARGET_OPENCL_VERSION
Modules/Core/GPUCommon/include/itkOpenCLUtil.h:#  define CL_TARGET_OPENCL_VERSION 120
Modules/Core/GPUCommon/include/itkOpenCLUtil.h:#  include <OpenCL/opencl.h>
Modules/Core/GPUCommon/include/itkOpenCLUtil.h:#  include <CL/opencl.h>
Modules/Core/GPUCommon/include/itkOpenCLUtil.h: * OpenCL workgroup (block) size for 1/2/3D - needs to be tuned based on the GPU architecture
Modules/Core/GPUCommon/include/itkOpenCLUtil.h:OpenCLGetLocalBlockSize(unsigned int ImageDim);
Modules/Core/GPUCommon/include/itkOpenCLUtil.h:OpenCLGetAvailableDevices(cl_platform_id platform, cl_device_type devType, cl_uint * numAvailableDevices);
Modules/Core/GPUCommon/include/itkOpenCLUtil.h:OpenCLGetMaxFlopsDev(cl_context cxGPUContext);
Modules/Core/GPUCommon/include/itkOpenCLUtil.h:OpenCLPrintDeviceInfo(cl_device_id device, bool verbose = false);
Modules/Core/GPUCommon/include/itkOpenCLUtil.h:/** Find the OpenCL platform that matches the "name" */
Modules/Core/GPUCommon/include/itkOpenCLUtil.h:OpenCLSelectPlatform(const char * name);
Modules/Core/GPUCommon/include/itkOpenCLUtil.h:/** Check OpenCL error */
Modules/Core/GPUCommon/include/itkOpenCLUtil.h:OpenCLCheckError(cl_int error, const char * filename = "", int lineno = 0, const char * location = "");
Modules/Core/GPUCommon/include/itkOpenCLUtil.h:/** Check if OpenCL-enabled GPU is present. */
Modules/Core/GPUCommon/include/itkOpenCLUtil.h:IsGPUAvailable();
Modules/Core/GPUCommon/include/itkGPUDataManager.h:#ifndef itkGPUDataManager_h
Modules/Core/GPUCommon/include/itkGPUDataManager.h:#define itkGPUDataManager_h
Modules/Core/GPUCommon/include/itkGPUDataManager.h:#include "itkOpenCLUtil.h"
Modules/Core/GPUCommon/include/itkGPUDataManager.h:#include "itkGPUContextManager.h"
Modules/Core/GPUCommon/include/itkGPUDataManager.h:/** \class GPUDataManager
Modules/Core/GPUCommon/include/itkGPUDataManager.h: * \brief GPU memory manager implemented using OpenCL. Required by GPUImage class.
Modules/Core/GPUCommon/include/itkGPUDataManager.h: * This class serves as a base class for GPU data container for GPUImage class,
Modules/Core/GPUCommon/include/itkGPUDataManager.h: * meta data will be already stored in image class (parent of GPUImage), therefore
Modules/Core/GPUCommon/include/itkGPUDataManager.h: * we did not name it GPUImageBase. Rather, this class is a GPU-specific data manager
Modules/Core/GPUCommon/include/itkGPUDataManager.h: * that provides functionalities for CPU-GPU data synchronization and grafting GPU data.
Modules/Core/GPUCommon/include/itkGPUDataManager.h: * \ingroup ITKGPUCommon
Modules/Core/GPUCommon/include/itkGPUDataManager.h:class GPUDataManager : public Object // DataObject//
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  /** allow GPUKernelManager to access GPU buffer pointer */
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  friend class GPUKernelManager;
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUDataManager);
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  using Self = GPUDataManager;
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  itkOverrideGetNameOfClassMacro(GPUDataManager);
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  SetGPUDirtyFlag(bool isDirty);
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  /** Make GPU up-to-date and mark CPU as dirty.
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  /** Make CPU up-to-date and mark GPU as dirty.
Modules/Core/GPUCommon/include/itkGPUDataManager.h:   * Call this function when you want to modify GPU data */
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  SetGPUBufferDirty();
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  IsGPUBufferDirty() const
Modules/Core/GPUCommon/include/itkGPUDataManager.h:    return m_IsGPUBufferDirty;
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  /** actual GPU->CPU memory copy takes place here */
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  /** actual CPU->GPU memory copy takes place here */
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  UpdateGPUBuffer();
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  /** Synchronize CPU and GPU buffers (using dirty flags) */
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  /** Method for grafting the content of one GPUDataManager into another one */
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  Graft(const GPUDataManager * data);
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  /** Initialize GPUDataManager */
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  /** Get GPU buffer pointer */
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  GetGPUBufferPointer();
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  /** Get GPU buffer pointer */
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  GPUDataManager();
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  ~GPUDataManager() override;
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  GPUContextManager * m_ContextManager{};
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  cl_mem m_GPUBuffer{};
Modules/Core/GPUCommon/include/itkGPUDataManager.h:  bool m_IsGPUBufferDirty{};
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.hxx:#ifndef itkGPUInPlaceImageFilter_hxx
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.hxx:#define itkGPUInPlaceImageFilter_hxx
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.hxx:GPUInPlaceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GPUInPlaceImageFilter() = default;
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.hxx:GPUInPlaceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::~GPUInPlaceImageFilter() = default;
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.hxx:GPUInPlaceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::PrintSelf(std::ostream & os, Indent indent) const
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.hxx:  GPUSuperclass::PrintSelf(os, indent);
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.hxx:GPUInPlaceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::ReleaseInputs()
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.hxx:    if( this->GetGPUEnabled() )
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.hxx:      std::cout << "ToDo: GPUInPlaceImageFilter::ReleaseInputs()" << std::endl;
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.hxx:GPUInPlaceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::AllocateOutputs()
Modules/Core/GPUCommon/include/itkGPUInPlaceImageFilter.hxx:  if (this->GetGPUEnabled())
Modules/Core/GPUCommon/include/itkGPUImage.h:#ifndef itkGPUImage_h
Modules/Core/GPUCommon/include/itkGPUImage.h:#define itkGPUImage_h
Modules/Core/GPUCommon/include/itkGPUImage.h:#include "itkGPUImageDataManager.h"
Modules/Core/GPUCommon/include/itkGPUImage.h:/** \class GPUImage
Modules/Core/GPUCommon/include/itkGPUImage.h: *  \brief Templated n-dimensional image class for the GPU.
Modules/Core/GPUCommon/include/itkGPUImage.h: * Derived from itk Image class to use with GPU image filters.
Modules/Core/GPUCommon/include/itkGPUImage.h: * This class manages both CPU and GPU memory implicitly, and
Modules/Core/GPUCommon/include/itkGPUImage.h: * can be used with non-GPU itk filters as well. Memory transfer
Modules/Core/GPUCommon/include/itkGPUImage.h: * between CPU and GPU is done automatically and implicitly.
Modules/Core/GPUCommon/include/itkGPUImage.h: * \ingroup ITKGPUCommon
Modules/Core/GPUCommon/include/itkGPUImage.h:class ITK_TEMPLATE_EXPORT GPUImage : public Image<TPixel, VImageDimension>
Modules/Core/GPUCommon/include/itkGPUImage.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUImage);
Modules/Core/GPUCommon/include/itkGPUImage.h:  using Self = GPUImage;
Modules/Core/GPUCommon/include/itkGPUImage.h:  itkOverrideGetNameOfClassMacro(GPUImage);
Modules/Core/GPUCommon/include/itkGPUImage.h:  // Allocate CPU and GPU memory space
Modules/Core/GPUCommon/include/itkGPUImage.h:  /** Explicit synchronize CPU/GPU buffers */
Modules/Core/GPUCommon/include/itkGPUImage.h:    m_DataManager->SetGPUBufferDirty();
Modules/Core/GPUCommon/include/itkGPUImage.h:    m_DataManager->SetGPUBufferDirty();
Modules/Core/GPUCommon/include/itkGPUImage.h:    m_DataManager->SetGPUBufferDirty();
Modules/Core/GPUCommon/include/itkGPUImage.h:  itkGetModifiableObjectMacro(DataManager, GPUImageDataManager<GPUImage>);
Modules/Core/GPUCommon/include/itkGPUImage.h:  GPUDataManager *
Modules/Core/GPUCommon/include/itkGPUImage.h:  GetGPUDataManager();
Modules/Core/GPUCommon/include/itkGPUImage.h:   * than GPU's. That is because Modified() is called at
Modules/Core/GPUCommon/include/itkGPUImage.h:   * increment GPU's time stamp in GPUGenerateData() the
Modules/Core/GPUCommon/include/itkGPUImage.h:  /** Graft the data and information from one GPUImage to another. */
Modules/Core/GPUCommon/include/itkGPUImage.h:  GPUImage();
Modules/Core/GPUCommon/include/itkGPUImage.h:  ~GPUImage() override;
Modules/Core/GPUCommon/include/itkGPUImage.h:  typename GPUImageDataManager<GPUImage>::Pointer m_DataManager{};
Modules/Core/GPUCommon/include/itkGPUImage.h:class ITK_TEMPLATE_EXPORT GPUImageFactory : public itk::ObjectFactoryBase
Modules/Core/GPUCommon/include/itkGPUImage.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUImageFactory);
Modules/Core/GPUCommon/include/itkGPUImage.h:  using Self = GPUImageFactory;
Modules/Core/GPUCommon/include/itkGPUImage.h:    return "A Factory for GPUImage";
Modules/Core/GPUCommon/include/itkGPUImage.h:  itkOverrideGetNameOfClassMacro(GPUImageFactory);
Modules/Core/GPUCommon/include/itkGPUImage.h:    auto factory = GPUImageFactory::New();
Modules/Core/GPUCommon/include/itkGPUImage.h:                         typeid(itk::GPUImage<pt, dm>).name(), \
Modules/Core/GPUCommon/include/itkGPUImage.h:                         "GPU Image Override",                 \
Modules/Core/GPUCommon/include/itkGPUImage.h:                         itk::CreateObjectFunction<GPUImage<pt, dm>>::New())
Modules/Core/GPUCommon/include/itkGPUImage.h:  GPUImageFactory()
Modules/Core/GPUCommon/include/itkGPUImage.h:    if (IsGPUAvailable())
Modules/Core/GPUCommon/include/itkGPUImage.h:class ITK_TEMPLATE_EXPORT GPUTraits
Modules/Core/GPUCommon/include/itkGPUImage.h:class ITK_TEMPLATE_EXPORT GPUTraits<Image<TPixelType, VDimension>>
Modules/Core/GPUCommon/include/itkGPUImage.h:  using Type = GPUImage<TPixelType, VDimension>;
Modules/Core/GPUCommon/include/itkGPUImage.h:#  include "itkGPUImage.hxx"
Modules/Core/GPUCommon/include/itkGPUFunctorBase.h:#ifndef itkGPUFunctorBase_h
Modules/Core/GPUCommon/include/itkGPUFunctorBase.h:#define itkGPUFunctorBase_h
Modules/Core/GPUCommon/include/itkGPUFunctorBase.h:#include "itkGPUKernelManager.h"
Modules/Core/GPUCommon/include/itkGPUFunctorBase.h:/** \class GPUFunctorBase
Modules/Core/GPUCommon/include/itkGPUFunctorBase.h: * \brief Base functor class for GPU functor image filters.
Modules/Core/GPUCommon/include/itkGPUFunctorBase.h: * \ingroup ITKGPUCommon
Modules/Core/GPUCommon/include/itkGPUFunctorBase.h:class GPUFunctorBase
Modules/Core/GPUCommon/include/itkGPUFunctorBase.h:  GPUFunctorBase() = default;
Modules/Core/GPUCommon/include/itkGPUFunctorBase.h:  virtual ~GPUFunctorBase() = default;
Modules/Core/GPUCommon/include/itkGPUFunctorBase.h:  /** Setup GPU kernel arguments for this functor.
Modules/Core/GPUCommon/include/itkGPUFunctorBase.h:   * \return Current argument index to set additional arguments in the GPU kernel. */
Modules/Core/GPUCommon/include/itkGPUFunctorBase.h:  SetGPUKernelArguments(GPUKernelManager::Pointer KernelManager, int KernelHandle) = 0;
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:#ifndef itkGPUImageDataManager_hxx
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:#define itkGPUImageDataManager_hxx
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:#include "itkOpenCLUtil.h"
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:GPUImageDataManager<ImageType>::SetImagePointer(ImageType * img)
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:  m_GPUBufferedRegionIndex = GPUDataManager::New();
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:  m_GPUBufferedRegionIndex->SetBufferSize(sizeof(int) * ImageDimension);
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:  m_GPUBufferedRegionIndex->SetCPUBufferPointer(m_BufferedRegionIndex);
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:  m_GPUBufferedRegionIndex->SetBufferFlag(CL_MEM_READ_ONLY);
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:  m_GPUBufferedRegionIndex->Allocate();
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:  m_GPUBufferedRegionIndex->SetGPUDirtyFlag(true);
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:  m_GPUBufferedRegionSize = GPUDataManager::New();
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:  m_GPUBufferedRegionSize->SetBufferSize(sizeof(int) * ImageDimension);
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:  m_GPUBufferedRegionSize->SetCPUBufferPointer(m_BufferedRegionSize);
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:  m_GPUBufferedRegionSize->SetBufferFlag(CL_MEM_READ_ONLY);
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:  m_GPUBufferedRegionSize->Allocate();
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:  m_GPUBufferedRegionSize->SetGPUDirtyFlag(true);
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:GPUImageDataManager<ImageType>::MakeCPUBufferUpToDate()
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:    ModifiedTimeType gpu_time = this->GetMTime();
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:     * access function in GPUImage and therefore dirty flag is not
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:     * CPU and GPU data as well
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:    if ((m_IsCPUBufferDirty || (gpu_time > cpu_time)) && m_GPUBuffer != nullptr && m_CPUBuffer != nullptr)
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:      itkDebugMacro("GPU->CPU data copy");
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:                                  m_GPUBuffer,
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:      OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:      m_IsGPUBufferDirty = false;
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:GPUImageDataManager<ImageType>::MakeGPUBufferUpToDate()
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:    ModifiedTimeType gpu_time = this->GetMTime();
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:     * access function in GPUImage and therefore dirty flag is not
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:     * CPU and GPU data as well
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:    if ((m_IsGPUBufferDirty || (gpu_time < cpu_time)) && m_CPUBuffer != nullptr && m_GPUBuffer != nullptr)
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:      itkDebugMacro("CPU->GPU data copy");
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:                                   m_GPUBuffer,
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:      OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/include/itkGPUImageDataManager.hxx:      m_IsGPUBufferDirty = false;
Modules/Core/GPUCommon/include/itkGPUImageOps.h:#ifndef itkGPUImageOps_h
Modules/Core/GPUCommon/include/itkGPUImageOps.h:#define itkGPUImageOps_h
Modules/Core/GPUCommon/include/itkGPUImageOps.h:/** Create a helper GPU Kernel class for GPUImageOps */
Modules/Core/GPUCommon/include/itkGPUImageOps.h:itkGPUKernelClassMacro(GPUImageOpsKernel);
Modules/Core/GPUCommon/include/itkGPUImageOps.h:/** \class GPUImageOps
Modules/Core/GPUCommon/include/itkGPUImageOps.h: * \brief Provides the kernels for some basic GPU Image Operations
Modules/Core/GPUCommon/include/itkGPUImageOps.h: * \ingroup ITKGPUCommon
Modules/Core/GPUCommon/include/itkGPUImageOps.h:class GPUImageOps
Modules/Core/GPUCommon/include/itkGPUImageOps.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUImageOps);
Modules/Core/GPUCommon/include/itkGPUImageOps.h:  GPUImageOps() = default;
Modules/Core/GPUCommon/include/itkGPUImageOps.h:  virtual ~GPUImageOps() = default;
Modules/Core/GPUCommon/include/itkGPUImageOps.h:  using Self = GPUImageOps;
Modules/Core/GPUCommon/include/itkGPUImageOps.h:  /** Get OpenCL Kernel source as a string, creates a GetOpenCLSource method */
Modules/Core/GPUCommon/include/itkGPUImageOps.h:  itkGetOpenCLSourceFromKernelMacro(GPUImageOpsKernel);
Modules/Core/GPUCommon/CMakeLists.txt:project(ITKGPUCommon)
Modules/Core/GPUCommon/CMakeLists.txt:  set(ITK_USE_GPU
Modules/Core/GPUCommon/CMakeLists.txt:      CACHE BOOL "Enable OpenCL GPU support." FORCE)
Modules/Core/GPUCommon/CMakeLists.txt:  set(ITK_USE_GPU ON)
Modules/Core/GPUCommon/CMakeLists.txt:  include(itkOpenCL)
Modules/Core/GPUCommon/CMakeLists.txt:if(ITK_USE_GPU
Modules/Core/GPUCommon/CMakeLists.txt:  message(FATAL_ERROR "Your compiler does not support Blocks (C language extension). This is needed for ITK_USE_GPU=ON")
Modules/Core/GPUCommon/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Core/GPUCommon/CMakeLists.txt:  set(ITKGPUCommon_LIBRARIES ITKGPUCommon ${OpenCL_LIBRARIES})
Modules/Core/GPUCommon/CMakeLists.txt:    list(APPEND ITKGPUCommon_LIBRARIES "-framework OpenCL")
Modules/Core/GPUCommon/CMakeLists.txt:      ITKGPUCommon_LIBRARIES
Modules/Core/GPUCommon/CMakeLists.txt:  set(ITKGPUCommon_SYSTEM_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS})
Modules/Core/GPUCommon/wrapping/itkGPUImageOps.wrap:itk_wrap_simple_class("itk::GPUImageOps")
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:GPUImageType = itk.GPUImage[itk.F, 2]
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:srcA = GPUImageType.New()
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:srcB = GPUImageType.New()
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:dest = GPUImageType.New()
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:kernel_manager = itk.GPUKernelManager.New()
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:gpu_source = itk.GPUImageOps.GetOpenCLSource()
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:print(gpu_source)
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:kernel_manager.LoadProgramFromString(gpu_source, "#define PIXELTYPE float\n")
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:print("Before GPU kernel execution")
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:kernel_manager.SetKernelArgWithImage(kernel_add, 0, srcA.GetGPUDataManager())
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:kernel_manager.SetKernelArgWithImage(kernel_add, 1, srcB.GetGPUDataManager())
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:kernel_manager.SetKernelArgWithImage(kernel_add, 2, dest.GetGPUDataManager())
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:print("After GPU kernel execution")
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:print("Before GPU kernel execution")
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:kernel_manager.SetKernelArgWithImage(kernel_mult, 0, srcA.GetGPUDataManager())
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:kernel_manager.SetKernelArgWithImage(kernel_mult, 1, srcB.GetGPUDataManager())
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:kernel_manager.SetKernelArgWithImage(kernel_mult, 2, dest.GetGPUDataManager())
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:print("After GPU kernel execution")
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:# Change Command Queue if more than one GPU device exists
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:context_manager = itk.GPUContextManager.GetInstance()
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:    print("More than one GPU device available, switching command queues.")
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:    print("Only one GPU device available, using same command queue.")
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:print("Before GPU kernel execution")
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:kernel_manager.SetKernelArgWithImage(kernel_sub, 0, srcA.GetGPUDataManager())
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:kernel_manager.SetKernelArgWithImage(kernel_sub, 1, srcB.GetGPUDataManager())
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:kernel_manager.SetKernelArgWithImage(kernel_sub, 2, dest.GetGPUDataManager())
Modules/Core/GPUCommon/wrapping/test/itkGPUImageTest.py:print("After GPU kernel execution")
Modules/Core/GPUCommon/wrapping/test/itkGPUReductionTest.py:summer = itk.GPUReduction[ElementType].New()
Modules/Core/GPUCommon/wrapping/test/CMakeLists.txt:    itkGPUImagePythonTest
Modules/Core/GPUCommon/wrapping/test/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/itkGPUImageTest.py)
Modules/Core/GPUCommon/wrapping/test/CMakeLists.txt:    itkGPUReductionPythonTest
Modules/Core/GPUCommon/wrapping/test/CMakeLists.txt:    ${CMAKE_CURRENT_SOURCE_DIR}/itkGPUReductionTest.py)
Modules/Core/GPUCommon/wrapping/itkGPUContextManager.wrap:itk_wrap_simple_class("itk::GPUContextManager" POINTER)
Modules/Core/GPUCommon/wrapping/itkGPUDataManager.wrap:itk_wrap_simple_class("itk::GPUDataManager" POINTER)
Modules/Core/GPUCommon/wrapping/itkGPUReduction.wrap:itk_wrap_class("itk::GPUReduction" POINTER)
Modules/Core/GPUCommon/wrapping/CMakeLists.txt:if(ITK_USE_GPU OR NOT ITK_SOURCE_DIR)
Modules/Core/GPUCommon/wrapping/CMakeLists.txt:  itk_wrap_module(ITKGPUCommon)
Modules/Core/GPUCommon/wrapping/CMakeLists.txt:      itkGPUContextManager
Modules/Core/GPUCommon/wrapping/CMakeLists.txt:      itkGPUKernelManager
Modules/Core/GPUCommon/wrapping/CMakeLists.txt:      itkGPUDataManager
Modules/Core/GPUCommon/wrapping/CMakeLists.txt:      itkGPUImage
Modules/Core/GPUCommon/wrapping/CMakeLists.txt:      itkGPUImageToImageFilter)
Modules/Core/GPUCommon/wrapping/itkGPUKernelManager.wrap:itk_wrap_simple_class("itk::GPUKernelManager" POINTER)
Modules/Core/GPUCommon/wrapping/itkGPUImage.wrap:itk_wrap_class("itk::GPUImage" POINTER)
Modules/Core/GPUCommon/wrapping/itkGPUImage.wrap:itk_wrap_include("itkGPUImageDataManager.h")
Modules/Core/GPUCommon/wrapping/itkGPUImage.wrap:itk_wrap_class("itk::GPUImageDataManager" POINTER)
Modules/Core/GPUCommon/wrapping/itkGPUImage.wrap:    itk_wrap_template("GI${ITKM_${t}}${d}" "itk::GPUImage< ${ITKT_${t}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUImage.wrap:    itk_wrap_template("GI${ITKM_${t}${d}}${d}" "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUImageToImageFilter.wrap:    itk_wrap_template("GI${ITKM_${to}}${d}" "itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUImageToImageFilter.wrap:    itk_wrap_template("GI${ITKM_${t}${d}}${d}" "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUImageToImageFilter.wrap:                        "itk::Image< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUImageToImageFilter.wrap:                        "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUImageToImageFilter.wrap:                        "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUImageToImageFilter.wrap:                      "itk::Image< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUImageToImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::Image< ${ITKT_${t}${d}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUImageToImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUImageToImageFilter.wrap:itk_wrap_class("itk::GPUImageToImageFilter" POINTER)
Modules/Core/GPUCommon/wrapping/itkGPUImageToImageFilter.wrap:                        "itk::Image< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUImageToImageFilter.wrap:                        "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUImageToImageFilter.wrap:                        "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUImageToImageFilter.wrap:                      "itk::Image< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUImageToImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::Image< ${ITKT_${t}${d}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUImageToImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUInPlaceImageFilter.wrap:                      "itk::Image< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUInPlaceImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUInPlaceImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUInPlaceImageFilter.wrap:                          "itk::Image< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUInPlaceImageFilter.wrap:                          "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::Image< ${ITKT_${to}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUInPlaceImageFilter.wrap:                          "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUInPlaceImageFilter.wrap:                      "itk::Image< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUInPlaceImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::Image< ${ITKT_${t}${d}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUInPlaceImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUInPlaceImageFilter.wrap:itk_wrap_class("itk::GPUImageToImageFilter" POINTER)
Modules/Core/GPUCommon/wrapping/itkGPUInPlaceImageFilter.wrap:        "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >, itk::InPlaceImageFilter< itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} > >"
Modules/Core/GPUCommon/wrapping/itkGPUInPlaceImageFilter.wrap:      "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::InPlaceImageFilter< itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} > >"
Modules/Core/GPUCommon/wrapping/itkGPUInPlaceImageFilter.wrap:itk_wrap_class("itk::GPUInPlaceImageFilter" POINTER)
Modules/Core/GPUCommon/wrapping/itkGPUInPlaceImageFilter.wrap:                        "itk::GPUImage< ${ITKT_${from}}, ${d} >, itk::GPUImage< ${ITKT_${to}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUInPlaceImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} >")
Modules/Core/GPUCommon/wrapping/itkGPUFunctorBase.wrap:itk_wrap_simple_class("itk::Functor::GPUFunctorBase")
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:#include "itkGPUKernelManager.h"
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::GPUKernelManager()
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  m_Manager = GPUContextManager::GetInstance();
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::~GPUKernelManager()
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::LoadProgramFromFile(const char * filename, const char * cPreamble)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  // open the OpenCL source code file
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    itkWarningMacro("Cannot open OpenCL source file");
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  // printout OpenCL source Path
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    itkWarningMacro("Cannot open OpenCL source file");
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  // Create OpenCL program from source strings
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    itkWarningMacro("Cannot create GPU program");
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    // itkWarningMacro("OpenCL program build error");
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:           << "OpenCL program build error:" << paramValue
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::LoadProgramFromString(const char * cSource, const char * cPreamble)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  // Create OpenCL program from source strings
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    itkWarningMacro("Cannot create GPU program");
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    // itkWarningMacro("OpenCL program build error");
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:           << "OpenCL program build error:" << paramValue
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::CreateKernel(const char * kernelName)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    itkWarningMacro("Fail to create GPU kernel");
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::GetKernelWorkGroupInfo(int kernelIdx, cl_kernel_work_group_info paramName, void * value)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::GetDeviceInfo(cl_kernel_work_group_info paramName, size_t argSize, void * argValue)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::SetKernelArg(int kernelIdx, cl_uint argIdx, size_t argSize, const void * argVal)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  m_KernelArgumentReady[kernelIdx][argIdx].m_GPUDataManager = (GPUDataManager::Pointer) nullptr;
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::SetTypedKernelArg(int kernelIdx, cl_uint argIdx, TArg argVal)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::SetKernelArgWithChar(int kernelIdx, cl_uint argIdx, char argVal)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::SetKernelArgWithUChar(int kernelIdx, cl_uint argIdx, unsigned char argVal)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::SetKernelArgWithShort(int kernelIdx, cl_uint argIdx, short argVal)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::SetKernelArgWithUShort(int kernelIdx, cl_uint argIdx, unsigned short argVal)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::SetKernelArgWithInt(int kernelIdx, cl_uint argIdx, int argVal)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::SetKernelArgWithUInt(int kernelIdx, cl_uint argIdx, unsigned int argVal)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::SetKernelArgWithLongLong(int kernelIdx, cl_uint argIdx, long long argVal)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::SetKernelArgWithULongLong(int kernelIdx, cl_uint argIdx, unsigned long long argVal)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::SetKernelArgWithFloat(int kernelIdx, cl_uint argIdx, float argVal)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::SetKernelArgWithDouble(int kernelIdx, cl_uint argIdx, double argVal)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::SetKernelArgWithImage(int kernelIdx, cl_uint argIdx, GPUDataManager * manager)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  errid = clSetKernelArg(m_KernelContainer[kernelIdx], argIdx, sizeof(cl_mem), manager->GetGPUBufferPointer());
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  m_KernelArgumentReady[kernelIdx][argIdx].m_GPUDataManager = manager;
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:// this function must be called right before GPU kernel is launched
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::CheckArgumentReady(int kernelIdx)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    if (m_KernelArgumentReady[kernelIdx][i].m_GPUDataManager != (GPUDataManager::Pointer) nullptr)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:      m_KernelArgumentReady[kernelIdx][i].m_GPUDataManager->SetCPUBufferDirty();
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::ResetArguments(int kernelIdx)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    m_KernelArgumentReady[kernelIdx][i].m_GPUDataManager = (GPUDataManager::Pointer) nullptr;
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::LaunchKernel1D(int kernelIdx, size_t globalWorkSize, size_t itkNotUsed(localWorkSize))
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    itkWarningMacro("GPU kernel arguments are not completely assigned");
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    itkWarningMacro("GPU kernel launch failed");
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::LaunchKernel2D(int    kernelIdx,
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    itkWarningMacro("GPU kernel arguments are not completely assigned");
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    itkWarningMacro("GPU kernel launch failed");
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::LaunchKernel3D(int    kernelIdx,
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    itkWarningMacro("GPU kernel arguments are not completely assigned");
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    itkWarningMacro("GPU kernel launch failed");
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::LaunchKernel(int kernelIdx, int dim, size_t * globalWorkSize, size_t * localWorkSize)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    itkWarningMacro("GPU kernel arguments are not completely assigned");
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:    itkWarningMacro("GPU kernel launch failed");
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::SetCurrentCommandQueue(int queueid)
Modules/Core/GPUCommon/src/itkGPUKernelManager.cxx:GPUKernelManager::GetCurrentCommandQueueID() const
Modules/Core/GPUCommon/src/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Core/GPUCommon/src/CMakeLists.txt:  set(ITKGPUCommon_SRCS
Modules/Core/GPUCommon/src/CMakeLists.txt:      itkGPUContextManager.cxx
Modules/Core/GPUCommon/src/CMakeLists.txt:      itkGPUDataManager.cxx
Modules/Core/GPUCommon/src/CMakeLists.txt:      itkGPUKernelManager.cxx
Modules/Core/GPUCommon/src/CMakeLists.txt:      itkOpenCLUtil.cxx)
Modules/Core/GPUCommon/src/CMakeLists.txt:  set(ITKGPUCommon_Kernels GPUImageOps.cl GPUReduction.cl)
Modules/Core/GPUCommon/src/CMakeLists.txt:  write_gpu_kernels("${ITKGPUCommon_Kernels}" ITKGPUCommon_SRCS)
Modules/Core/GPUCommon/src/CMakeLists.txt:  itk_module_add_library(ITKGPUCommon ${ITKGPUCommon_SRCS})
Modules/Core/GPUCommon/src/CMakeLists.txt:  target_link_libraries(ITKGPUCommon LINK_PUBLIC ${OPENCL_LIBRARIES})
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:#include "itkGPUDataManager.h"
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::GPUDataManager()
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  m_ContextManager = GPUContextManager::GetInstance();
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  m_GPUBuffer = nullptr;
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::~GPUDataManager()
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  if (m_GPUBuffer)
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:    clReleaseMemObject(m_GPUBuffer);
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::SetBufferSize(unsigned int num)
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::SetBufferFlag(cl_mem_flags flags)
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::Allocate()
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:    std::cout << this << "::Allocate Create GPU buffer of size " << m_BufferSize << " Bytes" << std::endl;
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:    m_GPUBuffer = clCreateBuffer(m_ContextManager->GetCurrentContext(), m_MemFlags, m_BufferSize, nullptr, &errid);
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:    m_IsGPUBufferDirty = true;
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  // this->UpdateGPUBuffer();
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::SetCPUBufferPointer(void * ptr)
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::SetCPUDirtyFlag(bool isDirty)
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::SetGPUDirtyFlag(bool isDirty)
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  m_IsGPUBufferDirty = isDirty;
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::SetGPUBufferDirty()
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  m_IsGPUBufferDirty = true;
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::SetCPUBufferDirty()
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  this->UpdateGPUBuffer();
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::UpdateCPUBuffer()
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  if (m_IsCPUBufferDirty && m_GPUBuffer != nullptr && m_CPUBuffer != nullptr)
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:    std::cout << this << "::UpdateCPUBuffer GPU->CPU data copy " << m_GPUBuffer << "->" << m_CPUBuffer << std::endl;
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:                                m_GPUBuffer,
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::UpdateGPUBuffer()
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  if (m_IsGPUBufferDirty && m_CPUBuffer != nullptr && m_GPUBuffer != nullptr)
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:    std::cout << this << "::UpdateGPUBuffer CPU->GPU data copy " << m_CPUBuffer << "->" << m_GPUBuffer << std::endl;
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:                                 m_GPUBuffer,
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:    m_IsGPUBufferDirty = false;
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::GetGPUBufferPointer()
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  return &m_GPUBuffer;
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::GetCPUBufferPointer()
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  SetGPUBufferDirty();
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::Update()
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  if (m_IsGPUBufferDirty && m_IsCPUBufferDirty)
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:    itkExceptionMacro("Cannot make up-to-date buffer because both CPU and GPU buffers are dirty");
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  this->UpdateGPUBuffer();
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  m_IsGPUBufferDirty = m_IsCPUBufferDirty = false;
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::SetCurrentCommandQueue(int queueid)
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:    m_IsGPUBufferDirty = true;
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::GetCurrentCommandQueueID() const
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::Graft(const GPUDataManager * data)
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:    if (m_GPUBuffer) // Decrease reference count to GPU memory
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:      clReleaseMemObject(m_GPUBuffer);
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:    if (data->m_GPUBuffer) // Increase reference count to GPU memory
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:      clRetainMemObject(data->m_GPUBuffer);
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:    m_GPUBuffer = data->m_GPUBuffer;
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:    m_IsGPUBufferDirty = data->m_IsGPUBufferDirty;
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::Initialize()
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  if (m_GPUBuffer) // Release GPU memory if exists
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:    clReleaseMemObject(m_GPUBuffer);
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  m_GPUBuffer = nullptr;
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  m_IsGPUBufferDirty = false;
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:GPUDataManager::PrintSelf(std::ostream & os, Indent indent) const
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  os << indent << "GPUBuffer: " << m_GPUBuffer << std::endl;
Modules/Core/GPUCommon/src/itkGPUDataManager.cxx:  itkPrintSelfBooleanMacro(IsGPUBufferDirty);
Modules/Core/GPUCommon/src/GPUReduction.cl:/** This is for parallel reduction and is modified from NVIDIA's example code */
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:#include "itkGPUContextManager.h"
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:GPUContextManager * GPUContextManager::m_Instance = nullptr;
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:GPUContextManager *
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:GPUContextManager::GetInstance()
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:    m_Instance = new GPUContextManager();
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:GPUContextManager::DestroyInstance()
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:  itkDebugStatement(std::cout << "OpenCL context is destroyed." << std::endl);
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:GPUContextManager::GPUContextManager()
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:  // Get NVIDIA platform by default
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:  m_Platform = OpenCLSelectPlatform("NVIDIA");
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:  cl_device_type devType = CL_DEVICE_TYPE_GPU; // CL_DEVICE_TYPE_CPU;//
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:  m_Devices = OpenCLGetAvailableDevices(m_Platform, devType, &m_NumberOfDevices);
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:    OpenCLPrintDeviceInfo(m_Devices[i], true);
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:GPUContextManager::~GPUContextManager()
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:GPUContextManager::GetCommandQueue(int i)
Modules/Core/GPUCommon/src/itkGPUContextManager.cxx:GPUContextManager::GetDeviceId(int i)
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:#include "itkOpenCLUtil.h"
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:OpenCLGetLocalBlockSize(unsigned int ImageDim)
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:   * OpenCL workgroup (block) size for 1/2/3D - needs to be tuned based on the GPU architecture
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:  int OPENCL_BLOCK_SIZE[3] = { 256, 16, 4 /*8*/ };
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:  return OPENCL_BLOCK_SIZE[ImageDim - 1];
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:OpenCLGetAvailableDevices(cl_platform_id platform, cl_device_type devType, cl_uint * numAvailableDevices)
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:  OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:OpenCLGetMaxFlopsDev(cl_context cxGPUContext)
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:  // get the list of GPU devices associated with context
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:  clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, nullptr, &szParmDataBytes);
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:  clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, nullptr);
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:OpenCLPrintDeviceInfo(cl_device_id device, bool verbose)
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:// Find the OpenCL platform that matches the "name"
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:OpenCLSelectPlatform(const char * name)
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:  // Get OpenCL platform count
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:      printf("No OpenCL platform found!\n\n");
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:OpenCLCheckError(cl_int error, const char * filename, int lineno, const char * location)
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:      errorMsg << "OpenCL Error : " << errorString[index] << std::endl;
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:      errorMsg << "OpenCL Error : Unspecified Error" << std::endl;
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:/** Check if OpenCL-enabled GPU is present. */
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:IsGPUAvailable()
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:  cl_platform_id platformId = OpenCLSelectPlatform("NVIDIA");
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:  cl_device_type devType = CL_DEVICE_TYPE_GPU;
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:  cl_device_id * device_id = OpenCLGetAvailableDevices(platformId, devType, &numDevices);
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:  msg << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
Modules/Core/GPUCommon/src/itkOpenCLUtil.cxx:      << "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n";
Modules/Core/Common/include/itkMacro.h:/**\def itkGPUKernelClassMacro
Modules/Core/Common/include/itkMacro.h: * provides the GPU kernel source code as a const char*
Modules/Core/Common/include/itkMacro.h:#define itkGPUKernelClassMacro(kernel) class itkGPUKernelMacro(kernel)
Modules/Core/Common/include/itkMacro.h:/**\def itkGPUKernelMacro
Modules/Core/Common/include/itkMacro.h: * Equivalent to the original `itkGPUKernelClassMacro(kernel)` macro, but
Modules/Core/Common/include/itkMacro.h:#define itkGPUKernelMacro(kernel)          \
Modules/Core/Common/include/itkMacro.h:    static const char * GetOpenCLSource(); \
Modules/Core/Common/include/itkMacro.h:#define itkGetOpenCLSourceFromKernelMacro(kernel)                             \
Modules/Core/Common/include/itkMacro.h:  static const char * GetOpenCLSource() { return kernel::GetOpenCLSource(); } \
Modules/Core/Common/src/itkConfigure.h.in: * The warning is disabled for NVCC (NVidia CUDA) for use of ITK macros with CUDA versions
Modules/Core/Common/src/itkConfigure.h.in:#if !defined(__CUDACC__)
Modules/Core/Common/src/itkConfigure.h.in:#cmakedefine ITK_USE_GPU
Modules/Core/GPUFiniteDifference/itk-module.cmake:    "This module contains the GPU implementations of base classes
Modules/Core/GPUFiniteDifference/itk-module.cmake:  ITKGPUFiniteDifference
Modules/Core/GPUFiniteDifference/itk-module.cmake:  ITKGPUCommon
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:#ifndef itkGPUFiniteDifferenceFunction_h
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:#define itkGPUFiniteDifferenceFunction_h
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:#include "itkGPUDataManager.h"
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:#include "itkGPUKernelManager.h"
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h: * \class GPUFiniteDifferenceFunction
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h: * This is a base class of GPU finite difference function.
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h: * Note that unlike most GPU classes, derived class of GPUFiniteDifferenceFunction
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h: * FiniteDifferenceFunction are reused by its derived GPU classes.
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h: * GPUFiniteDifferenceFunction must be subclassed to add functionality for
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h: * GPUComputeUpdate.
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h: * \ingroup ITKGPUFiniteDifference
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:class ITK_TEMPLATE_EXPORT GPUFiniteDifferenceFunction : public FiniteDifferenceFunction<TImageType>
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUFiniteDifferenceFunction);
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:  using Self = GPUFiniteDifferenceFunction;
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:  itkOverrideGetNameOfClassMacro(GPUFiniteDifferenceFunction);
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:  /** Empty implementation - this will not be used by GPU filters */
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:  /** GPU function to compute update buffer */
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:  GPUComputeUpdate(const typename TImageType::Pointer output, typename TImageType::Pointer update, void * gd) = 0;
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:  /** Allocate GPU buffers for computing metric statistics
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:  GPUAllocateMetricData(unsigned int itkNotUsed(numPixels))
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:  /** Release GPU buffers for computing metric statistics
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:  GPUReleaseMetricData()
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:  GPUFiniteDifferenceFunction() { m_GPUKernelManager = GPUKernelManager::New(); }
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:  ~GPUFiniteDifferenceFunction() override = default;
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:  /** GPU kernel manager for GPUFiniteDifferenceFunction class */
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:  typename GPUKernelManager::Pointer m_GPUKernelManager{};
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:  /** GPU kernel handle for GPUComputeUpdate() */
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFunction.h:  int m_ComputeUpdateGPUKernelHandle{};
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:#ifndef itkGPUDenseFiniteDifferenceImageFilter_hxx
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:#define itkGPUDenseFiniteDifferenceImageFilter_hxx
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:#include "itkOpenCLUtil.h"
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:GPUDenseFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  GPUDenseFiniteDifferenceImageFilter()
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:   * FiniteDifferenceImageFilter requires two GPU kernels
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:   * Kernel for 2 can be used for entire subclass of GPUDense..,
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:    itkExceptionMacro("GPUDenseFiniteDifferenceImageFilter supports 1/2/3D image.");
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  const char * GPUSource = GPUDenseFiniteDifferenceImageFilter::GetOpenCLSource();
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  this->m_GPUKernelManager->LoadProgramFromString(GPUSource, defines.str().c_str());
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  m_ApplyUpdateGPUKernelHandle = this->m_GPUKernelManager->CreateKernel("ApplyUpdate");
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:GPUDenseFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::CopyInputToOutput()
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  // Marking GPU is dirty by setting CPU as modified
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:GPUDenseFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::AllocateUpdateBuffer()
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  // CPUSuperclass will call Image::Allocate() which will call GPUImage::Allocate() .
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:GPUDenseFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::ApplyUpdate(const TimeStepType & dt)
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  this->GPUApplyUpdate(dt);
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:GPUDenseFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GPUApplyUpdate(
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  // GPU version of ApplyUpdate
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  using GPUBufferImage = typename itk::GPUTraits<UpdateBufferType>::Type;
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  using GPUOutputImage = typename itk::GPUTraits<TOutputImage>::Type;
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  typename GPUBufferImage::Pointer bfPtr = dynamic_cast<GPUBufferImage *>(GetUpdateBuffer());
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  typename GPUOutputImage::Pointer otPtr =
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:    dynamic_cast<GPUOutputImage *>(this->GetOutput()); // this->ProcessObject::GetOutput(0)
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  typename GPUOutputImage::SizeType outSize = otPtr->GetLargestPossibleRegion().GetSize();
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  localSize[0] = localSize[1] = localSize[2] = OpenCLGetLocalBlockSize(ImageDim);
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(m_ApplyUpdateGPUKernelHandle, argidx++, bfPtr->GetGPUDataManager());
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  this->m_GPUKernelManager->SetKernelArgWithImage(m_ApplyUpdateGPUKernelHandle, argidx++, otPtr->GetGPUDataManager());
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  this->m_GPUKernelManager->SetKernelArg(m_ApplyUpdateGPUKernelHandle, argidx++, sizeof(float), &(timeStep));
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:    this->m_GPUKernelManager->SetKernelArg(m_ApplyUpdateGPUKernelHandle, argidx++, sizeof(int), &(imgSize[i]));
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  this->m_GPUKernelManager->LaunchKernel(
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:    m_ApplyUpdateGPUKernelHandle, static_cast<int>(TInputImage::ImageDimension), globalSize, localSize);
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:GPUDenseFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GPUCalculateChange() -> TimeStepType
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  auto * df = dynamic_cast<GPUFiniteDifferenceFunction<OutputImageType> *>(this->GetDifferenceFunction().GetPointer());
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  df->GPUComputeUpdate(output, GetUpdateBuffer(), globalData);
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:GPUDenseFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::PrintSelf(std::ostream & os,
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.hxx:  GPUSuperclass::PrintSelf(os, indent);
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.hxx:#ifndef itkGPUFiniteDifferenceImageFilter_hxx
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.hxx:#define itkGPUFiniteDifferenceImageFilter_hxx
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.hxx:GPUFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GPUFiniteDifferenceImageFilter()
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.hxx:  m_State = GPUFiniteDifferenceFilterEnum::UNINITIALIZED;
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.hxx:GPUFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::~GPUFiniteDifferenceImageFilter() =
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.hxx:GPUFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GPUGenerateData()
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.hxx:  if (this->GetState() == GPUFiniteDifferenceFilterEnum::UNINITIALIZED)
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.hxx:    dt = this->GPUCalculateChange();
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.hxx:GPUFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::GenerateInputRequestedRegion()
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.hxx:  typename GPUSuperclass::InputImagePointer inputPtr = const_cast<TInputImage *>(this->GetInput());
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.hxx:GPUFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::ResolveTimeStep(
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.hxx:GPUFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::Halt()
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.hxx:GPUFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::InitializeFunctionCoefficients()
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.hxx:GPUFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>::PrintSelf(std::ostream & os,
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.hxx:  GPUSuperclass::PrintSelf(os, indent);
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:#ifndef itkGPUFiniteDifferenceImageFilter_h
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:#define itkGPUFiniteDifferenceImageFilter_h
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:#include "itkGPUInPlaceImageFilter.h"
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:#include "itkGPUFiniteDifferenceFunction.h"
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:#include "itkGPUFiniteDifferenceFilterEnum.h"
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h: * \class GPUFiniteDifferenceImageFilter
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h: * \brief Base class for GPU Finite Difference Image Filters.
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h: * \ingroup ITKGPUFiniteDifference
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h: * \sa GPUDenseFiniteDifferenceImageFilter */
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:class ITK_TEMPLATE_EXPORT GPUFiniteDifferenceImageFilter
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:  : public GPUInPlaceImageFilter<TInputImage, TOutputImage, TParentImageFilter>
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUFiniteDifferenceImageFilter);
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:  using Self = GPUFiniteDifferenceImageFilter;
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:  using GPUSuperclass = GPUInPlaceImageFilter<TInputImage, TOutputImage, TParentImageFilter>;
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:  itkOverrideGetNameOfClassMacro(GPUFiniteDifferenceImageFilter);
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:  using FiniteDifferenceFunctionType = typename GPUFiniteDifferenceFunction<TOutputImage>::DifferenceFunctionType;
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:  using FilterStateType = GPUFiniteDifferenceFilterEnum;
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:  static constexpr GPUFiniteDifferenceFilterEnum UNINITIALIZED = GPUFiniteDifferenceFilterEnum::UNINITIALIZED;
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:  static constexpr GPUFiniteDifferenceFilterEnum INITIALIZED = GPUFiniteDifferenceFilterEnum::INITIALIZED;
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:    this->SetState(GPUFiniteDifferenceFilterEnum::INITIALIZED);
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:    this->SetState(GPUFiniteDifferenceFilterEnum::UNINITIALIZED);
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:  itkSetMacro(State, GPUFiniteDifferenceFilterEnum);
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:  itkGetConstReferenceMacro(State, GPUFiniteDifferenceFilterEnum);
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:  GPUFiniteDifferenceImageFilter();
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:  ~GPUFiniteDifferenceImageFilter() override;
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:  GPUApplyUpdate(const TimeStepType & dt) = 0;
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:  GPUCalculateChange() = 0;
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:  GPUGenerateData() override;
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:  GPUFiniteDifferenceFilterEnum m_State{};
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceImageFilter.h:#  include "itkGPUFiniteDifferenceImageFilter.hxx"
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:#ifndef itkGPUDenseFiniteDifferenceImageFilter_h
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:#define itkGPUDenseFiniteDifferenceImageFilter_h
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:#include "itkGPUFiniteDifferenceImageFilter.h"
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:/** Create a helper GPU Kernel class for GPUDenseFiniteDifferenceImageFilter */
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:class ITKGPUFiniteDifference_EXPORT itkGPUKernelMacro(GPUDenseFiniteDifferenceImageFilterKernel);
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h: * \class GPUDenseFiniteDifferenceImageFilter
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h: * This is the GPU version of DenseFiniteDifferenceImageFilter class.
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h: * Currently only single-threaded, single GPU version is implemented.
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h: * GPUCalculateChange() and GPUApplyUpdate(), which are GPU version of
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h: * This filter can be used as a base class for GPU implementation of
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h: * \ingroup ITKGPUFiniteDifference
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:class ITK_TEMPLATE_EXPORT GPUDenseFiniteDifferenceImageFilter
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  : public GPUFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  ITK_DISALLOW_COPY_AND_MOVE(GPUDenseFiniteDifferenceImageFilter);
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  using Self = GPUDenseFiniteDifferenceImageFilter;
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  using GPUSuperclass = GPUFiniteDifferenceImageFilter<TInputImage, TOutputImage, TParentImageFilter>;
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  itkOverrideGetNameOfClassMacro(GPUDenseFiniteDifferenceImageFilter);
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  using InputImageType = typename GPUSuperclass::InputImageType;
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  using OutputImageType = typename GPUSuperclass::OutputImageType;
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  using FiniteDifferenceFunctionType = typename GPUSuperclass::FiniteDifferenceFunctionType;
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  static constexpr unsigned int ImageDimension = GPUSuperclass::ImageDimension;
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  using PixelType = typename GPUSuperclass::PixelType;
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  using TimeStepType = typename GPUSuperclass::TimeStepType;
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  /** Get OpenCL Kernel source as a string, creates a GetOpenCLSource method */
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  itkGetOpenCLSourceFromKernelMacro(GPUDenseFiniteDifferenceImageFilterKernel);
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  GPUDenseFiniteDifferenceImageFilter();
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  ~GPUDenseFiniteDifferenceImageFilter() override = default;
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:   * the GPU.  "dt" is the time step to use for the update of each pixel. */
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  GPUApplyUpdate(const TimeStepType & dt) override;
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:   * output using the GPU. Returns value is a time step to be used for the update. */
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  GPUCalculateChange() override;
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  /* GPU kernel handle for GPUApplyUpdate */
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:  int m_ApplyUpdateGPUKernelHandle{};
Modules/Core/GPUFiniteDifference/include/itkGPUDenseFiniteDifferenceImageFilter.h:#  include "itkGPUDenseFiniteDifferenceImageFilter.hxx"
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFilterEnum.h:#ifndef itkGPUFiniteDifferenceFilterEnum_h
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFilterEnum.h:#define itkGPUFiniteDifferenceFilterEnum_h
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFilterEnum.h:#include "ITKGPUFiniteDifferenceExport.h"
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFilterEnum.h: * \class GPUFiniteDifferenceFilterEnum
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFilterEnum.h: * \ingroup ITKGPUFiniteDifference
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFilterEnum.h:enum class GPUFiniteDifferenceFilterEnum : uint8_t
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFilterEnum.h:extern ITKGPUFiniteDifference_EXPORT std::ostream &
Modules/Core/GPUFiniteDifference/include/itkGPUFiniteDifferenceFilterEnum.h:                                     operator<<(std::ostream & out, const GPUFiniteDifferenceFilterEnum value);
Modules/Core/GPUFiniteDifference/CMakeLists.txt:project(ITKGPUFiniteDifference)
Modules/Core/GPUFiniteDifference/CMakeLists.txt:  set(ITK_USE_GPU
Modules/Core/GPUFiniteDifference/CMakeLists.txt:      CACHE BOOL "Enable OpenCL GPU support." FORCE)
Modules/Core/GPUFiniteDifference/CMakeLists.txt:  include(itkOpenCL)
Modules/Core/GPUFiniteDifference/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Core/GPUFiniteDifference/CMakeLists.txt:  set(ITKGPUFiniteDifference_SYSTEM_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
Modules/Core/GPUFiniteDifference/CMakeLists.txt:  set(ITKGPUFiniteDifference_SYSTEM_LIBRARY_DIRS ${OPENCL_LIBRARIES})
Modules/Core/GPUFiniteDifference/CMakeLists.txt:  set(ITKGPUFiniteDifference_LIBRARIES ITKGPUFiniteDifference)
Modules/Core/GPUFiniteDifference/wrapping/itkGPUFiniteDifferenceFunction.wrap:    itk_wrap_template("GI${ITKM_${t}}${d}" "itk::GPUImage< ${ITKT_${t}}, ${d} >")
Modules/Core/GPUFiniteDifference/wrapping/itkGPUFiniteDifferenceFunction.wrap:    itk_wrap_template("GI${ITKM_${t}${d}}${d}" "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >")
Modules/Core/GPUFiniteDifference/wrapping/itkGPUFiniteDifferenceFunction.wrap:itk_wrap_class("itk::GPUFiniteDifferenceFunction" POINTER)
Modules/Core/GPUFiniteDifference/wrapping/itkGPUFiniteDifferenceFunction.wrap:    itk_wrap_template("GI${ITKM_${t}}${d}" "itk::GPUImage< ${ITKT_${t}}, ${d} >")
Modules/Core/GPUFiniteDifference/wrapping/itkGPUFiniteDifferenceFunction.wrap:    itk_wrap_template("GI${ITKM_${t}${d}}${d}" "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >")
Modules/Core/GPUFiniteDifference/wrapping/itkGPUDenseFiniteDifferenceImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >")
Modules/Core/GPUFiniteDifference/wrapping/itkGPUDenseFiniteDifferenceImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} >")
Modules/Core/GPUFiniteDifference/wrapping/itkGPUDenseFiniteDifferenceImageFilter.wrap:itk_wrap_class("itk::GPUImageToImageFilter" POINTER)
Modules/Core/GPUFiniteDifference/wrapping/itkGPUDenseFiniteDifferenceImageFilter.wrap:      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::DenseFiniteDifferenceImageFilter< itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} > >"
Modules/Core/GPUFiniteDifference/wrapping/itkGPUDenseFiniteDifferenceImageFilter.wrap:      "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::DenseFiniteDifferenceImageFilter< itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} > >"
Modules/Core/GPUFiniteDifference/wrapping/itkGPUDenseFiniteDifferenceImageFilter.wrap:itk_wrap_class("itk::GPUInPlaceImageFilter" POINTER)
Modules/Core/GPUFiniteDifference/wrapping/itkGPUDenseFiniteDifferenceImageFilter.wrap:      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::DenseFiniteDifferenceImageFilter< itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} > >"
Modules/Core/GPUFiniteDifference/wrapping/itkGPUDenseFiniteDifferenceImageFilter.wrap:      "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::DenseFiniteDifferenceImageFilter< itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} > >"
Modules/Core/GPUFiniteDifference/wrapping/itkGPUDenseFiniteDifferenceImageFilter.wrap:itk_wrap_class("itk::GPUFiniteDifferenceImageFilter" POINTER)
Modules/Core/GPUFiniteDifference/wrapping/itkGPUDenseFiniteDifferenceImageFilter.wrap:      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::DenseFiniteDifferenceImageFilter< itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} > >"
Modules/Core/GPUFiniteDifference/wrapping/itkGPUDenseFiniteDifferenceImageFilter.wrap:      "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::DenseFiniteDifferenceImageFilter< itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} > >"
Modules/Core/GPUFiniteDifference/wrapping/itkGPUDenseFiniteDifferenceImageFilter.wrap:itk_wrap_class("itk::GPUDenseFiniteDifferenceImageFilter" POINTER)
Modules/Core/GPUFiniteDifference/wrapping/itkGPUDenseFiniteDifferenceImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >")
Modules/Core/GPUFiniteDifference/wrapping/itkGPUDenseFiniteDifferenceImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} >")
Modules/Core/GPUFiniteDifference/wrapping/itkGPUFiniteDifferenceFilterEnum.wrap:itk_wrap_simple_class("itk::GPUFiniteDifferenceFilterEnum" ENUM)
Modules/Core/GPUFiniteDifference/wrapping/CMakeLists.txt:if(ITK_USE_GPU OR NOT ITK_SOURCE_DIR)
Modules/Core/GPUFiniteDifference/wrapping/CMakeLists.txt:  itk_wrap_module(ITKGPUFiniteDifference)
Modules/Core/GPUFiniteDifference/wrapping/itkGPUFiniteDifferenceImageFilter.wrap:itk_wrap_include("itkGPUImage.h")
Modules/Core/GPUFiniteDifference/wrapping/itkGPUFiniteDifferenceImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}}, ${d} >, itk::GPUImage< ${ITKT_${t}}, ${d} >")
Modules/Core/GPUFiniteDifference/wrapping/itkGPUFiniteDifferenceImageFilter.wrap:                      "itk::GPUImage< ${ITKT_${t}${d}}, ${d} >, itk::GPUImage< ${ITKT_${t}${d}}, ${d} >")
Modules/Core/GPUFiniteDifference/src/GPUDenseFiniteDifferenceImageFilter.cl:     execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
Modules/Core/GPUFiniteDifference/src/CMakeLists.txt:set(ITKGPUFiniteDifference_SRCS itkGPUFiniteDifferenceFilterEnum.cxx)
Modules/Core/GPUFiniteDifference/src/CMakeLists.txt:if(ITK_USE_GPU)
Modules/Core/GPUFiniteDifference/src/CMakeLists.txt:  set(ITKGPUFiniteDifference_Kernels GPUDenseFiniteDifferenceImageFilter.cl)
Modules/Core/GPUFiniteDifference/src/CMakeLists.txt:  write_gpu_kernels("${ITKGPUFiniteDifference_Kernels}" ITKGPUFiniteDifference_SRCS)
Modules/Core/GPUFiniteDifference/src/CMakeLists.txt:  itk_module_add_library(ITKGPUFiniteDifference ${ITKGPUFiniteDifference_SRCS})
Modules/Core/GPUFiniteDifference/src/CMakeLists.txt:  target_link_libraries(ITKGPUFiniteDifference LINK_PUBLIC ${OPENCL_LIBRARIES})
Modules/Core/GPUFiniteDifference/src/itkGPUFiniteDifferenceFilterEnum.cxx:#include "itkGPUFiniteDifferenceFilterEnum.h"
Modules/Core/GPUFiniteDifference/src/itkGPUFiniteDifferenceFilterEnum.cxx:operator<<(std::ostream & out, const GPUFiniteDifferenceFilterEnum value)
Modules/Core/GPUFiniteDifference/src/itkGPUFiniteDifferenceFilterEnum.cxx:      case GPUFiniteDifferenceFilterEnum::UNINITIALIZED:
Modules/Core/GPUFiniteDifference/src/itkGPUFiniteDifferenceFilterEnum.cxx:        return "GPUFiniteDifferenceImageFilter<TInputImage,TOutputImage,TParentImageFilter>::"
Modules/Core/GPUFiniteDifference/src/itkGPUFiniteDifferenceFilterEnum.cxx:               "GPUFiniteDifferenceFilterEnum::"
Modules/Core/GPUFiniteDifference/src/itkGPUFiniteDifferenceFilterEnum.cxx:      case GPUFiniteDifferenceFilterEnum::INITIALIZED:
Modules/Core/GPUFiniteDifference/src/itkGPUFiniteDifferenceFilterEnum.cxx:        return "GPUFiniteDifferenceFilterEnum::INITIALIZED";
Modules/Core/GPUFiniteDifference/src/itkGPUFiniteDifferenceFilterEnum.cxx:        return "INVALID VALUE FOR GPUFiniteDifferenceFilterEnum";
Wrapping/Generators/Python/Tests/exclude-filters.txt:GPUUnaryFunctorImageFilter
Wrapping/Generators/Python/Tests/exclude-filters.txt:AnchorOpenCloseImageFilter
Documentation/docs/releases/5.2.md:- Add missing iostream include for CUDA classes ([3dba9889](https://github.com/SimonRit/RTK/commit/3dba9889))
Documentation/docs/releases/5.2.md:- rename guards for Cuda includes following ITK's style ([dd660a00](https://github.com/SimonRit/RTK/commit/dd660a00))
Documentation/docs/releases/5.2.md:-  Add missing iostream include for CUDA classes ([3dba9889](https://github.com/SimonRit/RTK/commit/3dba9889))
Documentation/docs/releases/5.2.md:-  rename guards for Cuda includes following ITK's style ([dd660a00](https://github.com/SimonRit/RTK/commit/dd660a00))
Documentation/docs/releases/5.4rc02.md:- [CudaCommon](https://github.com/RTKConsortium/ITKCudaCommon.git)
Documentation/docs/releases/3.14.md:    Code/Review/itkAnchorOpenCloseImageFilter
Documentation/docs/releases/4.2.md:-   Addition of GPU modules for Finite Difference, Smoothing,
Documentation/docs/releases/4.2.md:`     COMP: fix compilation warnings in GPU code`\
Documentation/docs/releases/4.2.md:`     COMP: there is not GPUCommonRegistration library`\
Documentation/docs/releases/4.2.md:`     BUG: fix GPUCommon tests`\
Documentation/docs/releases/4.2.md:`     ENH: add GPU Finite Difference module`\
Documentation/docs/releases/4.2.md:`     ENH: add GPU ImageFilterBase module`\
Documentation/docs/releases/4.2.md:`     ENH: add GPU Smoothing module`\
Documentation/docs/releases/4.2.md:`     ENH: add GPU Thresholding module`\
Documentation/docs/releases/4.2.md:`     ENH: add GPU RegistrationCommon module`\
Documentation/docs/releases/4.2.md:`     ENH: add GPU PDE Deformable Registration module`\
Documentation/docs/releases/4.2.md:`     ENH: add GPU Anisotropic Diffusion module`\
Documentation/docs/releases/4.2.md:`     COMP: Fix GPUCommon warnings and circular include error`
Documentation/docs/releases/4.2.md:`     COMP: Fix GPU class doxygen warnings.`\
Documentation/docs/releases/4.2.md:`     COMP: Fix doxygen warning in GPUFiniteDifferenceImageFilter.`\
Documentation/docs/releases/4.11.md:` - GPGPU system support for C++11`\
Documentation/docs/releases/4.11.md:`     COMP: Address CMake error that ITKGPUCommon is not in export`\
Documentation/docs/releases/4.11.md:`     ENH: Add the `“`ITK_USE_GPU`”` to ITK Config`\
Documentation/docs/releases/4.11.md:`     BUG: GPUImage regression due to API change.`\
Documentation/docs/releases/4.11.md:`     COMP: GPU baseclass ivars need to be protected`\
Documentation/docs/releases/4.11.md:`     COMP: remove C++11 compiler warning for CUDA compilations`\
Documentation/docs/releases/4.11.md:`     COMP: remove C++11 compiler warning for CUDA compilations`
Documentation/docs/releases/3.16.md:    Code/Review/itkAnchorOpenCloseImageFilter
Documentation/docs/releases/5.3rc01.md:- remove warnings due to the use of deprecated CUDA declarations ([184d0cad](https://github.com/SimonRit/RTK/commit/184d0cad))
Documentation/docs/releases/5.3rc01.md:- clean and modernize CMake handling of CUDA compilation ([642ad254](https://github.com/SimonRit/RTK/commit/642ad254))
Documentation/docs/releases/5.3rc01.md:- add missing depedency of lib RTK to lib ITKCudaCommon ([4d288a55](https://github.com/SimonRit/RTK/commit/4d288a55))
Documentation/docs/releases/5.3rc01.md:- Fix compilation error with VS 2019 (16.9.5) and CUDA 11.1 ([3bd80190](https://github.com/SimonRit/RTK/commit/3bd80190))
Documentation/docs/releases/4.5.md:`     ENH: Remote module: GPU and CPU Smoothing recursive YVV Gaussian Filter`\
Documentation/docs/releases/4.5.md:`     ENH: Remote module: GPU and CPU Smoothing recursive YVV Gaussian Filter`\
Documentation/docs/releases/5.0.md:      COMP: compile GPU classes with legacies disabled
Documentation/docs/releases/4.0.md:### GPU
Documentation/docs/releases/4.0.md:-   [GPU Acceleration - V4](https://itk.org/Wiki/ITK_Release_4/GPU_Acceleration)
Documentation/docs/releases/4.0.md:-   [GPU Acceleration - V4 (deprecated
Documentation/docs/releases/4.0.md:    link)](https://itk.org/Wiki/GPU_Acceleration_-_V4)
Documentation/docs/releases/5.4rc01.md:- [CudaCommon](https://github.com/RTKConsortium/ITKCudaCommon.git)
Documentation/docs/releases/5.4rc01.md:- Fix boolean member print statement in `itk::GPUDataManager` ([9b55bcec62](https://github.com/InsightSoftwareConsortium/ITK/commit/9b55bcec62))
Documentation/docs/releases/5.4rc01.md:- Add Superclass alias to GPUGradientAnisotropicDiffusionImageFilter ([296db6ac2b](https://github.com/InsightSoftwareConsortium/ITK/commit/296db6ac2b))
Documentation/docs/releases/5.4rc01.md:- Restore `PrintSelf` in itkGPUPDEDeformableRegistrationFilter.h ([fc1cce56a9](https://github.com/InsightSoftwareConsortium/ITK/commit/fc1cce56a9))
Documentation/docs/releases/5.4rc01.md:- Use `lock_guard<mutex>` in Logger classes and GPUImageDataManager ([a744aef227](https://github.com/InsightSoftwareConsortium/ITK/commit/a744aef227))
Documentation/docs/releases/5.4rc01.md:## CudaCommon:
Documentation/docs/releases/5.4rc01.md:- Upgrade GitHub actions following ITKRemoteModuleBuildTestPackageAction ([f08c035](https://github.com/RTKConsortium/ITKCudaCommon/commit/f08c035))
Documentation/docs/releases/5.4rc01.md:- Use cmake-options variable in GitHub action for Python packaging ([7763f2c](https://github.com/RTKConsortium/ITKCudaCommon/commit/7763f2c))
Documentation/docs/releases/5.4rc01.md:- Changed radART to medPhoton in RTK consortium ([87c2643](https://github.com/RTKConsortium/ITKCudaCommon/commit/87c2643))
Documentation/docs/releases/5.4rc01.md:- Change mailing list links to CREATIS website ([e5dba4a](https://github.com/RTKConsortium/ITKCudaCommon/commit/e5dba4a))
Documentation/docs/releases/5.4rc01.md:- Set CMP0135 to remove DOWNLOAD_EXTRACT_TIMESTAMP config warning ([82ce410](https://github.com/RTKConsortium/ITKCudaCommon/commit/82ce410))
Documentation/docs/releases/5.4rc01.md:- Fix unsigned int warning on image dimension ([319222c](https://github.com/RTKConsortium/ITKCudaCommon/commit/319222c))
Documentation/docs/releases/5.4rc01.md:- Add missing ITK config include for external compilation ([36ea588](https://github.com/RTKConsortium/ITKCudaCommon/commit/36ea588))
Documentation/docs/releases/5.4rc01.md:- Fix unused parameters argc / argv warnings ([cc77281](https://github.com/RTKConsortium/ITKCudaCommon/commit/cc77281))
Documentation/docs/releases/5.4rc01.md:- Fix typo in wheel name ([5cb1187](https://github.com/RTKConsortium/ITKCudaCommon/commit/5cb1187))
Documentation/docs/releases/5.4rc01.md:- Remove CUDA libraries from the CI generated Python packages ([cefcfbb](https://github.com/RTKConsortium/ITKCudaCommon/commit/cefcfbb))
Documentation/docs/releases/5.4rc01.md:- Fix CUDA/CPU inconsistency in DisplacedDetectorImageFilter ([e8299590](https://github.com/RTKConsortium/RTK/commit/e8299590))
Documentation/docs/releases/5.4rc01.md:- Use ITKRemoteModuleBuildTestPackageAction and separate CUDA actions ([bcce053e](https://github.com/RTKConsortium/RTK/commit/bcce053e))
Documentation/docs/releases/5.4rc01.md:- Remove ITKCudaCommon from the source tree ([138474a8](https://github.com/RTKConsortium/RTK/commit/138474a8))
Documentation/docs/releases/5.4rc01.md:- Add CudaCommon dependency to Cuda wheels ([9bc33997](https://github.com/RTKConsortium/RTK/commit/9bc33997))
Documentation/docs/releases/5.4rc01.md:- Import Cuda bin directory in the Windows environment for Cuda DLLs ([abe956fb](https://github.com/RTKConsortium/RTK/commit/abe956fb))
Documentation/docs/releases/5.4rc01.md:- Fix Cuda [[nodiscard]] warnings of Image "Transform" functions ([ab08d7a6](https://github.com/RTKConsortium/RTK/commit/ab08d7a6))
Documentation/docs/releases/5.4rc01.md:- Fix ITKPythonPackage git tag in Python CUDA action ([02005d75](https://github.com/RTKConsortium/RTK/commit/02005d75))
Documentation/docs/releases/5.4rc01.md:- Update AWS EC2 GPU AMI tag ([900f0db](https://github.com/InsightSoftwareConsortium/ITKVkFFTBackend/commit/900f0db))
Documentation/docs/releases/5.1.md:-  remove MacOS and Windows self-hosted Cuda python packages ([df54c6f7](https://github.com/SimonRit/RTK/commit/df54c6f7))
Documentation/docs/releases/5.1.md:-  remove unused kernel parameter in Cuda forward projection ([dcb015ee](https://github.com/SimonRit/RTK/commit/dcb015ee))
Documentation/docs/releases/5.3rc04.md:- [ITKCudaCommon](https://github.com/SimonRit/ITKCudaCommon): Framework for processing images with CUDA
Documentation/docs/releases/5.3rc04.md:- Fix GPUCommon compile errors ([9e78df2d7f](https://github.com/InsightSoftwareConsortium/ITK/commit/9e78df2d7f))
Documentation/docs/releases/5.3rc04.md:- Remove duplicate MSVC `/bigobj` flags from GPU test projects ([aec9e223a1](https://github.com/InsightSoftwareConsortium/ITK/commit/aec9e223a1))
Documentation/docs/releases/5.3rc04.md:- Fixed issue #1821 by conditionalizing ITKGPUCommon_* per-OS ([ce936dee3e](https://github.com/InsightSoftwareConsortium/ITK/commit/ce936dee3e))
Documentation/docs/releases/5.3rc04.md:## CudaCommon:
Documentation/docs/releases/5.3rc04.md:- Modules need updated version of ITK ([350ce95](https://github.com/SimonRit/ITKCudaCommon/commit/350ce95))
Documentation/docs/releases/5.3rc04.md:- Remove inclusion of .hxx files as headers ([2fdd6d5](https://github.com/SimonRit/ITKCudaCommon/commit/2fdd6d5))
Documentation/docs/releases/5.3rc04.md:- Add CUDA include dirs to CudaCommon_INCLUDE_DIRS ([8e2a7dc](https://github.com/SimonRit/ITKCudaCommon/commit/8e2a7dc))
Documentation/docs/releases/5.3rc04.md:- Build Cuda wheel on Linux self-hosted ([9aae2b3f](https://github.com/SimonRit/RTK/commit/9aae2b3f))
Documentation/docs/releases/5.3rc04.md:- Configure wheel name when CUDA is used ([1540c582](https://github.com/SimonRit/RTK/commit/1540c582))
Documentation/docs/releases/5.3rc04.md:- add Windows Cuda CI ([e7ae96d9](https://github.com/SimonRit/RTK/commit/e7ae96d9))
Documentation/docs/releases/5.3rc04.md:- fix compilation of external project depending on RTK with Cuda ([eff10ab1](https://github.com/SimonRit/RTK/commit/eff10ab1))
Documentation/docs/releases/5.3rc04.md:- set CMake vars for ITKCudaCommon targets ([4f33472e](https://github.com/SimonRit/RTK/commit/4f33472e))
Documentation/docs/releases/5.3rc04.md:- test compiled with rtk_add_cuda_test should use USE_CUDA ([f7dac1fe](https://github.com/SimonRit/RTK/commit/f7dac1fe))
Documentation/docs/releases/5.3rc04.md:- expect exception with Cuda conjugate gradient on CPU images ([7f23cbce](https://github.com/SimonRit/RTK/commit/7f23cbce))
Documentation/docs/releases/5.3rc04.md:- add missing checks after Cuda memory allocation ([fc9e8e2b](https://github.com/SimonRit/RTK/commit/fc9e8e2b))
Documentation/docs/releases/5.3.md:ITK 5.3.0 also includes Python dictionary conversions functions, `itk.dict_from_image`, `itk.image_from_dict`, `itk.dict_from_mesh`, `itk.mesh_from_dict`, and `itk.dict_from_transform`, `itk.transform_from_dict`.  Major improvements were made to the generation of Python interface, `*.pyi` files. Additional remote modules we contributed to support point set registration, [ITKFPFH](https://github.com/InsightSoftwareConsortium/ITKFPFH) computes feature points that could be used to obtain salient points while performing registration of two point clouds, and [ITKRANSAC](https://github.com/InsightSoftwareConsortium/ITKRANSAC) performs feature-based point cloud registration with the Random Sample Consensus (RANSAC) algorithm. A [new GitHub Action](https://github.com/InsightSoftwareConsortium/ITKRemoteModuleBuildTestPackageAction) was created to faciliate testing, packaging, and maintenance of remote modules. The Action includes recent developments to support the creation of 3.11 Python packages, ARM and GPGPU-capable Python packages.
Documentation/docs/releases/5.3.md:- [ITKCudaCommon](https://github.com/SimonRit/ITKCudaCommon): Framework for processing images with CUDA
Documentation/docs/releases/5.3.md:Major improvements to the toolkit in this release led to an extended release timeline as refinements were made in testing. For 5.4.0, we plan to return to our regular biannual release cadence. For 5.4, anticipated improvements include enhancements to GPU Python packages, Python packaging improvements via [scikit-build](https://scikit-build.org), improved [MONAI](https://monai.io/) support, and [WebAssembly support](https://wasm.itk.org). A few patch releases are expected before 5.4.0.
Documentation/docs/releases/5.3.md:- Upgrade RTK and CudaCommon with new RTKConsortium repository ([a75529e95d](https://github.com/InsightSoftwareConsortium/ITK/commit/a75529e95d))
Documentation/docs/releases/5.3.md:## CudaCommon:
Documentation/docs/releases/5.3.md:- Replace deprecated PUBLIC_LINK with PUBLIC ([1520198](https://github.com/RTKConsortium/ITKCudaCommon/commit/1520198))
Documentation/docs/releases/5.3.md:- Add python wrapping for CudaCommon ([095abb2](https://github.com/RTKConsortium/ITKCudaCommon/commit/095abb2))
Documentation/docs/releases/5.3.md:- Remove unneeded types from wrapping ([3531dd0](https://github.com/RTKConsortium/ITKCudaCommon/commit/3531dd0))
Documentation/docs/releases/5.3.md:- add CI for Windows and Linux CUDA packages ([2c5b730](https://github.com/RTKConsortium/ITKCudaCommon/commit/2c5b730))
Documentation/docs/releases/5.3.md:- Update GitHub actions for ITK 5.3 RC 4 ([19db3a3](https://github.com/RTKConsortium/ITKCudaCommon/commit/19db3a3))
Documentation/docs/releases/5.3.md:- Define CUDACOMMON_CUDA_VERSION for wheel names and verify it ([4b157fa](https://github.com/RTKConsortium/ITKCudaCommon/commit/4b157fa))
Documentation/docs/releases/5.3.md:- Bump CMake to v3.22.2 and Ninja to 1.10.2 ([de2796e](https://github.com/RTKConsortium/ITKCudaCommon/commit/de2796e))
Documentation/docs/releases/5.3.md:- Reduce downloads in self-hosted Github runners ([58ffcd1](https://github.com/RTKConsortium/ITKCudaCommon/commit/58ffcd1))
Documentation/docs/releases/5.3.md:- Update GitHub links to the new RTKConsortium repository ([57296fc](https://github.com/RTKConsortium/ITKCudaCommon/commit/57296fc))
Documentation/docs/releases/5.3.md:- use ITK's module mechanisme for EXPORT macros ([097151f](https://github.com/RTKConsortium/ITKCudaCommon/commit/097151f))
Documentation/docs/releases/5.3.md:- add missing export for Windows DLL ([afd01f3](https://github.com/RTKConsortium/ITKCudaCommon/commit/afd01f3))
Documentation/docs/releases/5.3.md:- Remove Cuda libraries from the Windows Python package ([4e89624](https://github.com/RTKConsortium/ITKCudaCommon/commit/4e89624))
Documentation/docs/releases/5.3.md:- Bump ITK and change http to https ([18ef765](https://github.com/RTKConsortium/ITKCudaCommon/commit/18ef765))
Documentation/docs/releases/5.3.md:- Add self-hosted CI for Windows python package with Cuda ([0751c885](https://github.com/RTKConsortium/RTK/commit/0751c885))
Documentation/docs/releases/5.3.md:- Wait for Cuda packages to publish artifacts ([e2ff4ce5](https://github.com/RTKConsortium/RTK/commit/e2ff4ce5))
Documentation/docs/releases/5.3.md:- Define RTK_CUDA_VERSION for wheel names and verify it in RTK ([21dd15e1](https://github.com/RTKConsortium/RTK/commit/21dd15e1))
Documentation/docs/releases/5.3.md:- Installation instructions for the CUDA-compatible Python package ([cbebbade](https://github.com/RTKConsortium/RTK/commit/cbebbade))
Documentation/docs/releases/5.3.md:- Remove Cuda libraries from the Windows python package ([4e68af80](https://github.com/RTKConsortium/RTK/commit/4e68af80))
Documentation/docs/releases/5.3.md:- Bump GPU CI ITK git tag ([b96b631](https://github.com/InsightSoftwareConsortium/ITKVkFFTBackend/commit/b96b631))
Documentation/docs/releases/5.0b03.md:In addition, four new remote modules are available, FFT's can be computed on the GPU via [cuFFTW](https://docs.nvidia.com/cuda/cufft/index.html), and ITK Python's `itk.imread` now supports image series. More information can be found in the feature summary below.
Documentation/docs/releases/5.0b03.md:- Support for Fast Fourier Transforms (FFTs) computed on NVIDIA GPUs via cuFFTW: enable by setting `ITK_USE_CUFFTW=ON`.
Documentation/docs/releases/5.0b03.md:      ENH: Add support for NVidia CUDA FFTs via cuFTTW
Documentation/docs/releases/4.3.md:`     BUG: fixed a few bugs related to GPU Demons`\
Documentation/docs/releases/4.3.md:`     ENH: added suport for buffered region in GPU image`
Documentation/docs/releases/4.3.md:`     BUG: GPU tests sometimes fails`\
Documentation/docs/releases/4.3.md:`     BUG: GPU tests sometimes fails`\
Documentation/docs/releases/5.0b01.md:      BUG: Missing SetCPUBufferPointer in GPUImage<...>::SetPixelContainer(...)
Documentation/docs/releases/4.4.md:`       COMP: Address unused parameter warning in GPU code`\
Documentation/docs/releases/4.4.md:`       COMP: Address unused parameter warning in GPU code`\
Documentation/docs/releases/4.4.md:`       COMP: Make GPUFunctorBase destructor virtual`\
Documentation/docs/releases/4.4.md:`       COMP: GPUPDEDeformableRegistrationFilter.cl fails to `“`compile`”`.`\
Documentation/docs/releases/4.4.md:`       COMP: GPUPDEDeformableRegistrationFilter.cl fails to `“`compile`”`.`
Documentation/docs/releases/4.9.md:`     COMP: Fixed the warning messages from itkGPU Module`\
Documentation/docs/releases/4.9.md:`     COMP: Fix inconsistant naming of GPUPDEDeformableRegistration`\
Documentation/docs/releases/4.8.md:`     COMP: Fix override warnings on clang with GPU module and MINC`\
Documentation/docs/releases/4.8.md:`     BUG: Fix memory leak at GPUCommon`
Documentation/docs/releases/4.13.md:      BUG: Fix marking of required inputs for GPU PDE registration
Documentation/docs/releases/4.13.md:      ENH: Use Input macros for set/get GPU PDE inputs
Documentation/docs/releases/4.13.md:      BUG: Correct GPUMeanImageFilter Superclass
Documentation/docs/releases/4.13.md:      BUG: Remove debug code from GPUMeanImageFilter
Documentation/docs/releases/4.13.md:      BUG: Do not use static_cast, SmartPointer in GPUImage::GetGPUDataManager
Documentation/docs/releases/4.13.md:      BUG: Remove debug code in itkGPUImage.hxx
Documentation/docs/releases/5.3rc03.md:ITK 5.3 RC 3 also includes FFT backend registration through the object factory, Python wrapping for more registration methods, metrics, and registration of point sets, and new remote modules to facilitate rendering of meshes and ITK filtering with CUDA. And, there any many more improvements and fixes detailed in the log below.
Documentation/docs/releases/5.3rc03.md:- [ITKCudaCommon](https://github.com/SimonRit/ITKCudaCommon): Framework for processing images with CUDA
Documentation/docs/releases/5.3rc03.md:We anticipate an additional release candidate following community testing before the 5.3.0 release. The following release candidate will provide an opportunity to test contributions for packaging, distributed computation, and GPU acceleration. Please try out the current release candidate, and discuss your experiences at [discourse.itk.org](https://discourse.itk.org). Contribute with pull requests, code reviews, and issue discussions in our [GitHub Organization](https://github.com/InsightSoftwareConsortium).
Documentation/docs/releases/5.3rc03.md:- add CudaCommon remote module ([2cd4d82307](https://github.com/InsightSoftwareConsortium/ITK/commit/2cd4d82307))
Documentation/docs/releases/5.3rc03.md:- Fixed CudaFDKConeBeamReconstructionFilter not using GPU default projection subset size ([7ba9a51c](https://github.com/SimonRit/RTK/commit/7ba9a51c))
Documentation/docs/releases/5.3rc03.md:- provide nvcc location to CMake package CUDAToolkit ([2a05f389](https://github.com/SimonRit/RTK/commit/2a05f389))
Documentation/docs/releases/5.3rc03.md:- add Linux x64 self-hosted CI for CUDA ([d3b52188](https://github.com/SimonRit/RTK/commit/d3b52188))
Documentation/docs/releases/5.3rc03.md:- use 5.2 as the default Cuda architecture ([baffa738](https://github.com/SimonRit/RTK/commit/baffa738))
Documentation/docs/releases/4.7.md:`     COMP: Remove unused typedef's in GPU code.`\
Documentation/docs/releases/4.7.md:`     COMP: Add a check for Blocks in GPU module`\
Documentation/docs/releases/5.4rc04.md:- [CudaCommon](https://github.com/RTKConsortium/ITKCudaCommon.git)
Documentation/docs/releases/5.4rc04.md:## CudaCommon:
Documentation/docs/releases/5.4rc04.md:- Remove CudaContextManager class and use cudaSetDevice ([09a9645](https://github.com/RTKConsortium/ITKCudaCommon/commit/09a9645))
Documentation/docs/releases/5.4rc04.md:- Upgrade CI to ITK v5.4rc01 ([4b362ac](https://github.com/RTKConsortium/ITKCudaCommon/commit/4b362ac))
Documentation/docs/releases/5.4rc04.md:- Upgrade CUDA packaging CI to ITK v5.4rc2 ([daf8676](https://github.com/RTKConsortium/ITKCudaCommon/commit/daf8676))
Documentation/docs/releases/5.4rc04.md:- Convert README to markdown ([bd8871e](https://github.com/RTKConsortium/ITKCudaCommon/commit/bd8871e))
Documentation/docs/releases/5.4rc04.md:- Fix new file name for README in itk-module.cmake ([c79cfc5](https://github.com/RTKConsortium/ITKCudaCommon/commit/c79cfc5))
Documentation/docs/releases/5.4rc04.md:- Add missing Python wrapping of ImageToImageFilter for CudaImage ([9d3fe9b](https://github.com/RTKConsortium/ITKCudaCommon/commit/9d3fe9b))
Documentation/docs/releases/5.4rc04.md:- Use older manylinux image with GCC 11 for Cuda 11.6 ([24266b1](https://github.com/RTKConsortium/ITKCudaCommon/commit/24266b1))
Documentation/docs/releases/5.4rc04.md:- Use newer CI packaging to exclude libraries ([0c20c4e](https://github.com/RTKConsortium/ITKCudaCommon/commit/0c20c4e))
Documentation/docs/releases/5.4rc04.md:- Replace itkTypeMacro calls with `itkOverrideGetNameOfClassMacro` ([dcc3401](https://github.com/RTKConsortium/ITKCudaCommon/commit/dcc3401))
Documentation/docs/releases/5.4rc04.md:- Release of ITKCudaCommon v1.0.1 ([7073f37](https://github.com/RTKConsortium/ITKCudaCommon/commit/7073f37))
Documentation/docs/releases/5.4rc04.md:- Step size accessors for Cuda ray tracing in iterative recon ([0a84555d](https://github.com/RTKConsortium/RTK/commit/0a84555d))
Documentation/docs/releases/5.4rc04.md:- Add Cuda wrapping of Parker, scatter glare and ramp filters ([777b75a3](https://github.com/RTKConsortium/RTK/commit/777b75a3))
Documentation/docs/releases/5.4rc04.md:- Allow compilation of CPU TV filter with RTK_USE_CUDA ON ([91fd0561](https://github.com/RTKConsortium/RTK/commit/91fd0561))
Documentation/docs/releases/5.4rc04.md:- Fix Windows compilations in non-CUDA CI ([ef740ffe](https://github.com/RTKConsortium/RTK/commit/ef740ffe))

```
