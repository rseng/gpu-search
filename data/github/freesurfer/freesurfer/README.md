# https://github.com/freesurfer/freesurfer

```console
gems/fs_build_GEMS2:-DBUILD_CUDA=OFF \
gems/README.txt:CUDA_64_BIT_DEVICE_CODE:BOOL=ON
gems/README.txt://Attach the build rule to the CUDA source file.  Enable only when
gems/README.txt:// the CUDA source file is added to at most one target.
gems/README.txt:CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE:BOOL=ON
gems/README.txt:CUDA_BUILD_CUBIN:BOOL=OFF
gems/README.txt:CUDA_BUILD_EMULATION:BOOL=OFF
gems/README.txt://"cudart" library
gems/README.txt:CUDA_CUDART_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libcudart.so
gems/README.txt://"cuda" library (older versions only).
gems/README.txt:CUDA_CUDA_LIBRARY:FILEPATH=/usr/lib64/libcuda.so
gems/README.txt:CUDA_GENERATED_OUTPUT_DIR:PATH=
gems/README.txt:CUDA_HOST_COMPILATION_CPP:BOOL=ON
gems/README.txt:CUDA_HOST_COMPILER:FILEPATH=/usr/bin/cc
gems/README.txt:CUDA_NVCC_EXECUTABLE:FILEPATH=/usr/local/cuda/bin/nvcc
gems/README.txt:CUDA_NVCC_FLAGS:STRING=
gems/README.txt:CUDA_NVCC_FLAGS_DEBUG:STRING=
gems/README.txt:CUDA_NVCC_FLAGS_MINSIZEREL:STRING=
gems/README.txt:CUDA_NVCC_FLAGS_RELEASE:STRING=
gems/README.txt:CUDA_NVCC_FLAGS_RELWITHDEBINFO:STRING=
gems/README.txt:CUDA_PROPAGATE_HOST_FLAGS:BOOL=ON
gems/README.txt:CUDA_SDK_ROOT_DIR:PATH=CUDA_SDK_ROOT_DIR-NOTFOUND
gems/README.txt://Compile CUDA objects with separable compilation enabled.  Requires
gems/README.txt:// CUDA 5.0+
gems/README.txt:CUDA_SEPARABLE_COMPILATION:BOOL=OFF
gems/README.txt:CUDA_TARGET_CPU_ARCH:STRING=
gems/README.txt:CUDA_TOOLKIT_INCLUDE:PATH=/usr/local/cuda/include
gems/README.txt:CUDA_TOOLKIT_ROOT_DIR:PATH=/usr/local/cuda
gems/README.txt:CUDA_TOOLKIT_TARGET_DIR:PATH=CUDA_TOOLKIT_ROOT_DIR-NOTFOUND
gems/README.txt://Use the static version of the CUDA runtime library if available
gems/README.txt:CUDA_USE_STATIC_CUDA_RUNTIME:BOOL=OFF
gems/README.txt://Print out the commands run while compiling the CUDA source file.
gems/README.txt:CUDA_VERBOSE_BUILD:BOOL=OFF
gems/README.txt://Version of CUDA as computed from nvcc.
gems/README.txt:CUDA_VERSION:STRING=8.0
gems/README.txt:CUDA_cublas_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libcublas.so
gems/README.txt:CUDA_cublasemu_LIBRARY:FILEPATH=CUDA_cublasemu_LIBRARY-NOTFOUND
gems/README.txt://static CUDA runtime library
gems/README.txt:CUDA_cudart_static_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libcudart_static.a
gems/README.txt:CUDA_cufft_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libcufft.so
gems/README.txt:CUDA_cufftemu_LIBRARY:FILEPATH=CUDA_cufftemu_LIBRARY-NOTFOUND
gems/README.txt:CUDA_cupti_LIBRARY:FILEPATH=/usr/local/cuda/extras/CUPTI/lib64/libcupti.so
gems/README.txt:CUDA_curand_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libcurand.so
gems/README.txt:CUDA_cusolver_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libcusolver.so
gems/README.txt:CUDA_cusparse_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libcusparse.so
gems/README.txt:CUDA_nppc_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libnppc.so
gems/README.txt:CUDA_nppi_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libnppi.so
gems/README.txt:CUDA_npps_LIBRARY:FILEPATH=/usr/local/cuda/lib64/libnpps.so
gems/README.txt:kvlGEMSCUDA_LIB_DEPENDS:STATIC=general;/usr/local/cuda/lib64/libcudart.so;
gems/README.txt://ADVANCED property for variable: CUDA_64_BIT_DEVICE_CODE
gems/README.txt:CUDA_64_BIT_DEVICE_CODE-ADVANCED:INTERNAL=1
gems/README.txt://List of intermediate files that are part of the cuda dependency
gems/README.txt:CUDA_ADDITIONAL_CLEAN_FILES:INTERNAL=/space/sand/1/users/zkaufman/git_repos/freesurfer.test/GEMS2/cuda/CMakeFiles/kvlGEMSCUDA.dir//kvlGEMSCUDA_generated_visitcountersimplecudaimpl.cu.o.depend
gems/README.txt://ADVANCED property for variable: CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE
gems/README.txt:CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_BUILD_CUBIN
gems/README.txt:CUDA_BUILD_CUBIN-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_BUILD_EMULATION
gems/README.txt:CUDA_BUILD_EMULATION-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_CUDART_LIBRARY
gems/README.txt:CUDA_CUDART_LIBRARY-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_CUDA_LIBRARY
gems/README.txt:CUDA_CUDA_LIBRARY-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_GENERATED_OUTPUT_DIR
gems/README.txt:CUDA_GENERATED_OUTPUT_DIR-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_HOST_COMPILATION_CPP
gems/README.txt:CUDA_HOST_COMPILATION_CPP-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_NVCC_EXECUTABLE
gems/README.txt:CUDA_NVCC_EXECUTABLE-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_NVCC_FLAGS
gems/README.txt:CUDA_NVCC_FLAGS-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_NVCC_FLAGS_DEBUG
gems/README.txt:CUDA_NVCC_FLAGS_DEBUG-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_NVCC_FLAGS_MINSIZEREL
gems/README.txt:CUDA_NVCC_FLAGS_MINSIZEREL-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_NVCC_FLAGS_RELEASE
gems/README.txt:CUDA_NVCC_FLAGS_RELEASE-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_NVCC_FLAGS_RELWITHDEBINFO
gems/README.txt:CUDA_NVCC_FLAGS_RELWITHDEBINFO-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_PROPAGATE_HOST_FLAGS
gems/README.txt:CUDA_PROPAGATE_HOST_FLAGS-ADVANCED:INTERNAL=1
gems/README.txt://This is the value of the last time CUDA_SDK_ROOT_DIR was set
gems/README.txt:CUDA_SDK_ROOT_DIR_INTERNAL:INTERNAL=CUDA_SDK_ROOT_DIR-NOTFOUND
gems/README.txt://ADVANCED property for variable: CUDA_SEPARABLE_COMPILATION
gems/README.txt:CUDA_SEPARABLE_COMPILATION-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_TARGET_CPU_ARCH
gems/README.txt:CUDA_TARGET_CPU_ARCH-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_TOOLKIT_INCLUDE
gems/README.txt:CUDA_TOOLKIT_INCLUDE-ADVANCED:INTERNAL=1
gems/README.txt://This is the value of the last time CUDA_TOOLKIT_ROOT_DIR was
gems/README.txt:CUDA_TOOLKIT_ROOT_DIR_INTERNAL:INTERNAL=/usr/local/cuda
gems/README.txt://ADVANCED property for variable: CUDA_TOOLKIT_TARGET_DIR
gems/README.txt:CUDA_TOOLKIT_TARGET_DIR-ADVANCED:INTERNAL=1
gems/README.txt://This is the value of the last time CUDA_TOOLKIT_TARGET_DIR was
gems/README.txt:CUDA_TOOLKIT_TARGET_DIR_INTERNAL:INTERNAL=CUDA_TOOLKIT_ROOT_DIR-NOTFOUND
gems/README.txt://ADVANCED property for variable: CUDA_VERBOSE_BUILD
gems/README.txt:CUDA_VERBOSE_BUILD-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_VERSION
gems/README.txt:CUDA_VERSION-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_cublas_LIBRARY
gems/README.txt:CUDA_cublas_LIBRARY-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_cublasemu_LIBRARY
gems/README.txt:CUDA_cublasemu_LIBRARY-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_cudart_static_LIBRARY
gems/README.txt:CUDA_cudart_static_LIBRARY-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_cufft_LIBRARY
gems/README.txt:CUDA_cufft_LIBRARY-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_cufftemu_LIBRARY
gems/README.txt:CUDA_cufftemu_LIBRARY-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_cupti_LIBRARY
gems/README.txt:CUDA_cupti_LIBRARY-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_curand_LIBRARY
gems/README.txt:CUDA_curand_LIBRARY-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_cusolver_LIBRARY
gems/README.txt:CUDA_cusolver_LIBRARY-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_cusparse_LIBRARY
gems/README.txt:CUDA_cusparse_LIBRARY-ADVANCED:INTERNAL=1
gems/README.txt:CUDA_make2cmake:INTERNAL=/autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/cmake/3.5.2/share/cmake-3.5/Modules/FindCUDA/make2cmake.cmake
gems/README.txt://ADVANCED property for variable: CUDA_nppc_LIBRARY
gems/README.txt:CUDA_nppc_LIBRARY-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_nppi_LIBRARY
gems/README.txt:CUDA_nppi_LIBRARY-ADVANCED:INTERNAL=1
gems/README.txt://ADVANCED property for variable: CUDA_npps_LIBRARY
gems/README.txt:CUDA_npps_LIBRARY-ADVANCED:INTERNAL=1
gems/README.txt:CUDA_parse_cubin:INTERNAL=/autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/cmake/3.5.2/share/cmake-3.5/Modules/FindCUDA/parse_cubin.cmake
gems/README.txt:CUDA_run_nvcc:INTERNAL=/autofs/cluster/pubsw/2/pubsw/Linux2-2.3-x86_64/packages/cmake/3.5.2/share/cmake-3.5/Modules/FindCUDA/run_nvcc.cmake
gems/README.txt://Details about finding CUDA
gems/README.txt:FIND_PACKAGE_MESSAGE_DETAILS_CUDA:INTERNAL=[/usr/local/cuda][/usr/local/cuda/bin/nvcc][/usr/local/cuda/include][/usr/local/cuda/lib64/libcudart.so][v8.0()]
gems/CMakeLists.txt:option(GEMS_BUILD_CUDA "Build CUDA stuff" OFF)
gems/CMakeLists.txt:if(GEMS_BUILD_CUDA)
gems/CMakeLists.txt:  find_package(CUDA REQUIRED)
gems/CMakeLists.txt:  include_directories(${CUDA_INCLUDE_DIRS})
gems/CMakeLists.txt:  set(CMAKE_CXX_FLAGS "-DCUDA_FOUND ${CMAKE_CXX_FLAGS}")
gems/CMakeLists.txt:  include_directories(cuda)
gems/CMakeLists.txt:  add_subdirectory(cuda)
gems/Testing/boosttests.cpp:#ifdef CUDA_FOUND
gems/Testing/boosttests.cpp:#include "cudaglobalfixture.hpp"
gems/Testing/boosttests.cpp:BOOST_GLOBAL_FIXTURE(CUDAGlobalFixture);
gems/Testing/testiosupport.hpp:#ifdef CUDA_FOUND
gems/Testing/testiosupport.hpp:#include "dimensioncuda.hpp"
gems/Testing/testiosupport.hpp:#ifdef CUDA_FOUND
gems/Testing/testiosupport.hpp:		      const kvl::cuda::Dimension<nDims,IndexType> d) {
gems/Testing/testatlasmeshvisitcounter.cpp:#ifdef CUDA_FOUND
gems/Testing/testatlasmeshvisitcounter.cpp:#include "cudaimage.hpp"
gems/Testing/testatlasmeshvisitcounter.cpp:#include "atlasmeshvisitcountercuda.hpp"
gems/Testing/testatlasmeshvisitcounter.cpp:#include "visitcountersimplecuda.hpp"
gems/Testing/testatlasmeshvisitcounter.cpp:#include "visitcountertetrahedralmeshcuda.hpp"
gems/Testing/testatlasmeshvisitcounter.cpp:#ifdef CUDA_FOUND
gems/Testing/testatlasmeshvisitcounter.cpp:#ifdef GPU_ALL_PRECISIONS
gems/Testing/testatlasmeshvisitcounter.cpp:  kvl::cuda::VisitCounterSimple<float,float>,
gems/Testing/testatlasmeshvisitcounter.cpp:  kvl::cuda::VisitCounterSimple<double,double>,
gems/Testing/testatlasmeshvisitcounter.cpp:  kvl::cuda::VisitCounterSimple<float,double>,
gems/Testing/testatlasmeshvisitcounter.cpp:  kvl::cuda::VisitCounterTetrahedralMesh
gems/Testing/testatlasmeshvisitcounter.cpp:  > CUDAImplTypes;
gems/Testing/testatlasmeshvisitcounter.cpp:  kvl::cuda::VisitCounterSimple<double,double>,
gems/Testing/testatlasmeshvisitcounter.cpp:  kvl::cuda::VisitCounterTetrahedralMesh
gems/Testing/testatlasmeshvisitcounter.cpp:  > CUDAImplTypes;
gems/Testing/testatlasmeshvisitcounter.cpp:#ifdef CUDA_FOUND
gems/Testing/testatlasmeshvisitcounter.cpp:BOOST_AUTO_TEST_CASE_TEMPLATE( LowerCornerGPUSimple, ImplType, CUDAImplTypes  )
gems/Testing/testatlasmeshvisitcounter.cpp:BOOST_AUTO_TEST_CASE_TEMPLATE( OriginOnlyGPUSimple, ImplType, CUDAImplTypes )
gems/Testing/testatlasmeshvisitcounter.cpp:BOOST_AUTO_TEST_CASE_TEMPLATE( XAxisOnlyGPUSimple, ImplType, CUDAImplTypes )
gems/Testing/testatlasmeshvisitcounter.cpp:BOOST_AUTO_TEST_CASE_TEMPLATE( FarCornerOnlyGPUSimple, ImplType, CUDAImplTypes )
gems/Testing/testatlasmeshvisitcounter.cpp:BOOST_AUTO_TEST_CASE_TEMPLATE( UpperCornerGPUSimple, ImplType, CUDAImplTypes )
gems/Testing/testatlasmeshvisitcounter.cpp:BOOST_AUTO_TEST_CASE_TEMPLATE( NoVerticesGPUSimple, ImplType, CUDAImplTypes )
gems/Testing/testatlasmeshvisitcounter.cpp:BOOST_AUTO_TEST_CASE_TEMPLATE( LowerCornerExactGPUSimple, ImplType, CUDAImplTypes )
gems/Testing/testatlasmeshvisitcounter.cpp:BOOST_AUTO_TEST_CASE_TEMPLATE( UpperCornerExactGPUSimple, ImplType, CUDAImplTypes )
gems/Testing/testatlasmeshvisitcounter.cpp:BOOST_AUTO_TEST_CASE_TEMPLATE( AutoCornersGPUSimple, ImplType, CUDAImplTypes )
gems/Testing/testatlasmeshvisitcounter.cpp:BOOST_AUTO_TEST_CASE_TEMPLATE( AutoCornersLargeImageLargeTetrahedronGPUSimple, ImplType, CUDAImplTypes )
gems/Testing/testatlasmeshvisitcounter.cpp:#ifdef CUDA_FOUND
gems/Testing/testatlasmeshvisitcounter.cpp:BOOST_AUTO_TEST_CASE_TEMPLATE( SimpleCUDAImpl, ImplType, CUDAImplTypes )
gems/Testing/testcudatetrahedralmesh.cpp:#include "cudatetrahedralmesh.hpp"
gems/Testing/testcudatetrahedralmesh.cpp:BOOST_AUTO_TEST_SUITE( CudaTetrahedralMesh )
gems/Testing/testcudatetrahedralmesh.cpp:BOOST_AUTO_TEST_CASE_TEMPLATE( SendToGPU, T, CoordTypes )
gems/Testing/testcudatetrahedralmesh.cpp:  kvl::cuda::CudaTetrahedralMesh<T,unsigned long,float> ctm;
gems/Testing/testcudatetrahedralmesh.cpp:  kvl::cuda::CudaTetrahedralMesh<double,unsigned char,float> ctm;
gems/Testing/testatlasmeshalphadrawer.cpp:#ifdef CUDA_FOUND
gems/Testing/testatlasmeshalphadrawer.cpp:#include "cudaimage.hpp"
gems/Testing/testatlasmeshalphadrawer.cpp:#include "atlasmeshalphadrawercuda.hpp"
gems/Testing/testatlasmeshalphadrawer.cpp:#ifdef CUDA_FOUND
gems/Testing/testatlasmeshalphadrawer.cpp:BOOST_DATA_TEST_CASE( ContainedUnitCubeGPU,  boost::unit_test::data::xrange(nAlphas), classNumber )
gems/Testing/testatlasmeshalphadrawer.cpp:  kvl::cuda::AtlasMeshAlphaDrawerCUDA ad;
gems/Testing/testatlasmeshalphadrawer.cpp:BOOST_DATA_TEST_CASE( ContainedLargeCubeGPU,  boost::unit_test::data::xrange(nAlphas), classNumber )
gems/Testing/testatlasmeshalphadrawer.cpp:  kvl::cuda::AtlasMeshAlphaDrawerCUDA ad;
gems/Testing/testatlasmeshalphadrawer.cpp:#ifdef CUDA_FOUND
gems/Testing/testatlasmeshalphadrawer.cpp:BOOST_AUTO_TEST_CASE( CudaImpl )
gems/Testing/testatlasmeshalphadrawer.cpp:  kvl::cuda::AtlasMeshAlphaDrawerCUDA ad;
gems/Testing/CMakeLists.txt:add_executable(kvlAtlasMeshRasterizorTestGPU kvlAtlasMeshRasterizorTestGPU.cxx)
gems/Testing/CMakeLists.txt:target_link_libraries(kvlAtlasMeshRasterizorTestGPU kvlGEMSCommon)
gems/Testing/CMakeLists.txt:  if(CUDA_FOUND)
gems/Testing/CMakeLists.txt:    list(APPEND testsrcs testcudaimage.cpp)
gems/Testing/CMakeLists.txt:    list(APPEND testsrcs cudaglobalfixture.cpp)
gems/Testing/CMakeLists.txt:    list(APPEND testsrcs cudaimagetests.cu)
gems/Testing/CMakeLists.txt:    list(APPEND testsrcs testcudatetrahedralmesh.cpp)
gems/Testing/CMakeLists.txt:    list(APPEND testsrcs testdimensioncuda.cpp)
gems/Testing/CMakeLists.txt:    list(APPEND GEMS2libs kvlGEMSCUDA)
gems/Testing/CMakeLists.txt:    cuda_add_executable( TestGEMS2 ${testsrcs} ${cudatestsrcs})
gems/Testing/cudaglobalfixture.cpp:#include <cuda_runtime.h>
gems/Testing/cudaglobalfixture.cpp:#include "cudacheck.hpp"
gems/Testing/cudaglobalfixture.cpp:#include "cudacontroller.hpp"
gems/Testing/cudaglobalfixture.cpp:#include "cudaglobalfixture.hpp"
gems/Testing/cudaglobalfixture.cpp:CUDAGlobalFixture::CUDAGlobalFixture() {
gems/Testing/cudaglobalfixture.cpp:  const int cudaDevice = 0;
gems/Testing/cudaglobalfixture.cpp:  kvl::cuda::InitialiseCUDA(cudaDevice);
gems/Testing/cudaglobalfixture.cpp:  cudaDeviceProp properties;
gems/Testing/cudaglobalfixture.cpp:  CUDA_SAFE_CALL( cudaGetDeviceProperties(&properties, cudaDevice) );
gems/Testing/cudaglobalfixture.cpp:  std::cout <<  "CUDA Device : " << properties.name << std::endl;
gems/Testing/testdimensioncuda.cpp:#include "dimensioncuda.hpp"
gems/Testing/testdimensioncuda.cpp:BOOST_AUTO_TEST_SUITE( DimensionCuda )
gems/Testing/testdimensioncuda.cpp:    kvl::cuda::Dimension<1,LengthType> testObject;
gems/Testing/testdimensioncuda.cpp:    kvl::cuda::Dimension<1,LengthType> d1, d2, d3;
gems/Testing/testdimensioncuda.cpp:  kvl::cuda::Dimension<1,LengthType> testObject;
gems/Testing/testdimensioncuda.cpp:    kvl::cuda::Dimension<2,LengthType> d;
gems/Testing/testdimensioncuda.cpp:  kvl::cuda::Dimension<3,LengthType> d;
gems/Testing/testdimensioncuda.cpp:    kvl::cuda::Dimension<1,LengthType> d;
gems/Testing/testdimensioncuda.cpp:    kvl::cuda::Dimension<2,LengthType> d;
gems/Testing/testdimensioncuda.cpp:    kvl::cuda::Dimension<3,LengthType> d;
gems/Testing/testdimensioncuda.cpp:  kvl::cuda::Dimension<3,LengthType> d;
gems/Testing/testdimensioncuda.cpp:  kvl::cuda::Dimension<3,LengthType> d;
gems/Testing/testdimensioncuda.cpp:    kvl::cuda::Dimension<1,LengthType> d1;
gems/Testing/testdimensioncuda.cpp:    kvl::cuda::Dimension<3,LengthType> d3;
gems/Testing/testdimensioncuda.cpp:  kvl::cuda::Dimension<nDims,LengthType> d;
gems/Testing/cudaglobalfixture.hpp:#ifdef CUDA_FOUND
gems/Testing/cudaglobalfixture.hpp:class CUDAGlobalFixture {
gems/Testing/cudaglobalfixture.hpp:  CUDAGlobalFixture();
gems/Testing/testcudaimage.cpp:// This will only be compiled if CUDA_FOUND is defined
gems/Testing/testcudaimage.cpp:#include "dimensioncuda.hpp"
gems/Testing/testcudaimage.cpp:#include "cudaimage.hpp"
gems/Testing/testcudaimage.cpp:#include "cudaimagetests.hpp"
gems/Testing/testcudaimage.cpp:    kvl::cuda::Dimension<nDims,size_t> srcDims, resultDims;
gems/Testing/testcudaimage.cpp:    kvl::cuda::CudaImage<T,nDims,size_t> d_image;
gems/Testing/testcudaimage.cpp:     kvl::cuda::Dimension<nDims,size_t> srcDims, resultDims;
gems/Testing/testcudaimage.cpp:     kvl::cuda::CudaImage<T,nDims,size_t> d_src, d_dst;
gems/Testing/testcudaimage.cpp:BOOST_AUTO_TEST_SUITE( CudaImage )
gems/Testing/cudaimagetests.hpp:#include "cudaimage.hpp"
gems/Testing/cudaimagetests.hpp:void runPlusTest( kvl::cuda::CudaImage<ElementType,nDims,IndexType>& dst,
gems/Testing/cudaimagetests.hpp:		  const kvl::cuda::CudaImage<ElementType,nDims,IndexType>& src,
gems/Testing/cudaimagetests.hpp:void runPlusTest<unsigned char,1,size_t>( kvl::cuda::CudaImage<unsigned char,1,size_t>& dst,
gems/Testing/cudaimagetests.hpp:					  const kvl::cuda::CudaImage<unsigned char,1,size_t>& src,
gems/Testing/cudaimagetests.hpp:void runPlusTest<int,1,size_t>( kvl::cuda::CudaImage<int,1,size_t>& dst,
gems/Testing/cudaimagetests.hpp:				const kvl::cuda::CudaImage<int,1,size_t>& src,
gems/Testing/cudaimagetests.hpp:void runPlusTest<float,1,size_t>( kvl::cuda::CudaImage<float,1,size_t>& dst,
gems/Testing/cudaimagetests.hpp:				  const kvl::cuda::CudaImage<float,1,size_t>& src,
gems/Testing/cudaimagetests.hpp:void runPlusTest<double,1,size_t>( kvl::cuda::CudaImage<double,1,size_t>& dst,
gems/Testing/cudaimagetests.hpp:				   const kvl::cuda::CudaImage<double,1,size_t>& src,
gems/Testing/cudaimagetests.hpp:void runPlusTest<unsigned char,2,size_t>( kvl::cuda::CudaImage<unsigned char,2,size_t>& dst,
gems/Testing/cudaimagetests.hpp:					  const kvl::cuda::CudaImage<unsigned char,2,size_t>& src,
gems/Testing/cudaimagetests.hpp:void runPlusTest<int,2,size_t>( kvl::cuda::CudaImage<int,2,size_t>& dst,
gems/Testing/cudaimagetests.hpp:				const kvl::cuda::CudaImage<int,2,size_t>& src,
gems/Testing/cudaimagetests.hpp:void runPlusTest<float,2,size_t>( kvl::cuda::CudaImage<float,2,size_t>& dst,
gems/Testing/cudaimagetests.hpp:				  const kvl::cuda::CudaImage<float,2,size_t>& src,
gems/Testing/cudaimagetests.hpp:void runPlusTest<double,2,size_t>( kvl::cuda::CudaImage<double,2,size_t>& dst,
gems/Testing/cudaimagetests.hpp:				   const kvl::cuda::CudaImage<double,2,size_t>& src,
gems/Testing/cudaimagetests.hpp:void runPlusTest<unsigned char,3,size_t>( kvl::cuda::CudaImage<unsigned char,3,size_t>& dst,
gems/Testing/cudaimagetests.hpp:					  const kvl::cuda::CudaImage<unsigned char,3,size_t>& src,
gems/Testing/cudaimagetests.hpp:void runPlusTest<int,3,size_t>( kvl::cuda::CudaImage<int,3,size_t>& dst,
gems/Testing/cudaimagetests.hpp:				const kvl::cuda::CudaImage<int,3,size_t>& src,
gems/Testing/cudaimagetests.hpp:void runPlusTest<float,3,size_t>( kvl::cuda::CudaImage<float,3,size_t>& dst,
gems/Testing/cudaimagetests.hpp:				  const kvl::cuda::CudaImage<float,3,size_t>& src,
gems/Testing/cudaimagetests.hpp:void runPlusTest<double,3,size_t>( kvl::cuda::CudaImage<double,3,size_t>& dst,
gems/Testing/cudaimagetests.hpp:				   const kvl::cuda::CudaImage<double,3,size_t>& src,
gems/Testing/cudaimagetests.cu:#include "cudaimagetests.hpp"
gems/Testing/cudaimagetests.cu:void PlusKernel1D( kvl::cuda::Image_GPU<ElementType,1,IndexType> dst,
gems/Testing/cudaimagetests.cu:		   const kvl::cuda::Image_GPU<ElementType,1,IndexType> src,
gems/Testing/cudaimagetests.cu:void PlusKernel2D( kvl::cuda::Image_GPU<ElementType,2,IndexType> dst,
gems/Testing/cudaimagetests.cu:		   const kvl::cuda::Image_GPU<ElementType,2,IndexType> src,
gems/Testing/cudaimagetests.cu:void PlusKernel3D( kvl::cuda::Image_GPU<ElementType,3,IndexType> dst,
gems/Testing/cudaimagetests.cu:		   const kvl::cuda::Image_GPU<ElementType,3,IndexType> src,
gems/Testing/cudaimagetests.cu:void LaunchPlusKernel( kvl::cuda::CudaImage<ElementType,1,IndexType>& dst,
gems/Testing/cudaimagetests.cu:		       const kvl::cuda::CudaImage<ElementType,1,IndexType>& src,
gems/Testing/cudaimagetests.cu:  auto err = cudaGetLastError();
gems/Testing/cudaimagetests.cu:  if( cudaSuccess != err ) {
gems/Testing/cudaimagetests.cu:    throw kvl::cuda::CUDAException(err);
gems/Testing/cudaimagetests.cu:  err = cudaDeviceSynchronize(); if( cudaSuccess != err ) {
gems/Testing/cudaimagetests.cu:    throw kvl::cuda::CUDAException(err);
gems/Testing/cudaimagetests.cu:void LaunchPlusKernel( kvl::cuda::CudaImage<ElementType,2,IndexType>& dst,
gems/Testing/cudaimagetests.cu:		       const kvl::cuda::CudaImage<ElementType,2,IndexType>& src,
gems/Testing/cudaimagetests.cu:  auto err = cudaGetLastError();
gems/Testing/cudaimagetests.cu:  if( cudaSuccess != err ) {
gems/Testing/cudaimagetests.cu:    throw kvl::cuda::CUDAException(err);
gems/Testing/cudaimagetests.cu:  err = cudaDeviceSynchronize(); if( cudaSuccess != err ) {
gems/Testing/cudaimagetests.cu:    throw kvl::cuda::CUDAException(err);
gems/Testing/cudaimagetests.cu:void LaunchPlusKernel( kvl::cuda::CudaImage<ElementType,3,IndexType>& dst,
gems/Testing/cudaimagetests.cu:		       const kvl::cuda::CudaImage<ElementType,3,IndexType>& src,
gems/Testing/cudaimagetests.cu:  auto err = cudaGetLastError();
gems/Testing/cudaimagetests.cu:  if( cudaSuccess != err ) {
gems/Testing/cudaimagetests.cu:    throw kvl::cuda::CUDAException(err);
gems/Testing/cudaimagetests.cu:  err = cudaDeviceSynchronize(); if( cudaSuccess != err ) {
gems/Testing/cudaimagetests.cu:    throw kvl::cuda::CUDAException(err);
gems/Testing/cudaimagetests.cu:void runPlusTest<unsigned char,1,size_t>( kvl::cuda::CudaImage<unsigned char,1,size_t>& dst,
gems/Testing/cudaimagetests.cu:					  const kvl::cuda::CudaImage<unsigned char,1,size_t>& src,
gems/Testing/cudaimagetests.cu:void runPlusTest<int,1,size_t>( kvl::cuda::CudaImage<int,1,size_t>& dst,
gems/Testing/cudaimagetests.cu:				const kvl::cuda::CudaImage<int,1,size_t>& src,
gems/Testing/cudaimagetests.cu:void runPlusTest<float,1,size_t>( kvl::cuda::CudaImage<float,1,size_t>& dst,
gems/Testing/cudaimagetests.cu:				  const kvl::cuda::CudaImage<float,1,size_t>& src,
gems/Testing/cudaimagetests.cu:void runPlusTest<double,1,size_t>( kvl::cuda::CudaImage<double,1,size_t>& dst,
gems/Testing/cudaimagetests.cu:				   const kvl::cuda::CudaImage<double,1,size_t>& src,
gems/Testing/cudaimagetests.cu:void runPlusTest<unsigned char,2,size_t>( kvl::cuda::CudaImage<unsigned char,2,size_t>& dst,
gems/Testing/cudaimagetests.cu:					  const kvl::cuda::CudaImage<unsigned char,2,size_t>& src,
gems/Testing/cudaimagetests.cu:void runPlusTest<int,2,size_t>( kvl::cuda::CudaImage<int,2,size_t>& dst,
gems/Testing/cudaimagetests.cu:				const kvl::cuda::CudaImage<int,2,size_t>& src,
gems/Testing/cudaimagetests.cu:void runPlusTest<float,2,size_t>( kvl::cuda::CudaImage<float,2,size_t>& dst,
gems/Testing/cudaimagetests.cu:				  const kvl::cuda::CudaImage<float,2,size_t>& src,
gems/Testing/cudaimagetests.cu:void runPlusTest<double,2,size_t>( kvl::cuda::CudaImage<double,2,size_t>& dst,
gems/Testing/cudaimagetests.cu:				   const kvl::cuda::CudaImage<double,2,size_t>& src,
gems/Testing/cudaimagetests.cu:void runPlusTest<unsigned char,3,size_t>( kvl::cuda::CudaImage<unsigned char,3,size_t>& dst,
gems/Testing/cudaimagetests.cu:					  const kvl::cuda::CudaImage<unsigned char,3,size_t>& src,
gems/Testing/cudaimagetests.cu:void runPlusTest<int,3,size_t>( kvl::cuda::CudaImage<int,3,size_t>& dst,
gems/Testing/cudaimagetests.cu:				const kvl::cuda::CudaImage<int,3,size_t>& src,
gems/Testing/cudaimagetests.cu:void runPlusTest<float,3,size_t>( kvl::cuda::CudaImage<float,3,size_t>& dst,
gems/Testing/cudaimagetests.cu:				  const kvl::cuda::CudaImage<float,3,size_t>& src,
gems/Testing/cudaimagetests.cu:void runPlusTest<double,3,size_t>( kvl::cuda::CudaImage<double,3,size_t>& dst,
gems/Testing/cudaimagetests.cu:				   const kvl::cuda::CudaImage<double,3,size_t>& src,
gems/.gitignore:cuda/CMakeFiles/
gems/.gitignore:cuda/cmake_install.cmake
gems/cuda/cudadeleters.hpp:#include "cudacheck.hpp"
gems/cuda/cudadeleters.hpp:  namespace cuda {
gems/cuda/cudadeleters.hpp:    class CudaDeviceDeleter {
gems/cuda/cudadeleters.hpp:      void operator()(cudaPitchedPtr* d_ptr) {
gems/cuda/cudadeleters.hpp:	    CUDA_SAFE_CALL(cudaFree(d_ptr->ptr));
gems/cuda/simplesharedtetrahedroninterior.hpp:#include "cudautils.hpp"
gems/cuda/simplesharedtetrahedroninterior.hpp:  namespace cuda {
gems/cuda/simplesharedtetrahedroninterior.hpp:      auto err = cudaGetLastError();
gems/cuda/simplesharedtetrahedroninterior.hpp:      if( cudaSuccess != err ) {
gems/cuda/simplesharedtetrahedroninterior.hpp:	throw CUDAException(err);
gems/cuda/simplesharedtetrahedroninterior.hpp:      typedef typename MeshSupplier::GPUType MeshArg;
gems/cuda/simplesharedtetrahedroninterior.hpp:      err = cudaDeviceSynchronize();
gems/cuda/simplesharedtetrahedroninterior.hpp:      if( cudaSuccess != err ) {
gems/cuda/simplesharedtetrahedroninterior.hpp:	throw CUDAException(err);
gems/cuda/alphadraweraction.hpp:  AlphaDrawerAction(kvl::cuda::Image_GPU<AlphasType,3,unsigned short> target,
gems/cuda/alphadraweraction.hpp:  kvl::cuda::Image_GPU<AlphasType,3,unsigned short> output;
gems/cuda/visitcountertetrahedralmeshcudaimpl.hpp:#include "cudaimage.hpp"
gems/cuda/visitcountertetrahedralmeshcudaimpl.hpp:#include "cudatetrahedralmesh.hpp"
gems/cuda/visitcountertetrahedralmeshcudaimpl.hpp:  namespace cuda {
gems/cuda/visitcountertetrahedralmeshcudaimpl.hpp:    void RunVisitCounterTetrahedralMeshCUDA( CudaImage<int,3,unsigned short>& d_output,
gems/cuda/visitcountertetrahedralmeshcudaimpl.hpp:					     const CudaTetrahedralMesh<double,unsigned long,float>& ctm );
gems/cuda/cudacontroller.hpp:  namespace cuda {
gems/cuda/cudacontroller.hpp:    void InitialiseCUDA(const int deviceID);
gems/cuda/cudacontroller.hpp:    void FinalizeCUDA();
gems/cuda/visitcountersimplecudaimpl.cu:#include "visitcountersimplecudaimpl.hpp"
gems/cuda/visitcountersimplecudaimpl.cu:class SimpleMesh_GPU {
gems/cuda/visitcountersimplecudaimpl.cu:  SimpleMesh_GPU( const kvl::cuda::Image_GPU<ArgType,3,size_t>& tetrahedra ) : tetInfo(tetrahedra) {}
gems/cuda/visitcountersimplecudaimpl.cu:  const kvl::cuda::Image_GPU<ArgType,3,size_t> tetInfo;
gems/cuda/visitcountersimplecudaimpl.cu:  namespace cuda {
gems/cuda/visitcountersimplecudaimpl.cu:      typedef SimpleMesh_GPU<CoordinateType> GPUType;
gems/cuda/visitcountersimplecudaimpl.cu:      SimpleMeshSupply( const kvl::cuda::CudaImage<CoordinateType,3,size_t>& tetrahedra ) : d_tetInfo(tetrahedra) {}
gems/cuda/visitcountersimplecudaimpl.cu:      SimpleMesh_GPU<CoordinateType> getArg() const {
gems/cuda/visitcountersimplecudaimpl.cu:	return SimpleMesh_GPU<CoordinateType>(this->d_tetInfo.getArg());
gems/cuda/visitcountersimplecudaimpl.cu:      const kvl::cuda::CudaImage<CoordinateType,3,size_t>& d_tetInfo;
gems/cuda/visitcountersimplecudaimpl.cu:    void RunVisitCounterSimpleCUDA<float,float>( CudaImage<int,3,unsigned short>& d_output,
gems/cuda/visitcountersimplecudaimpl.cu:						 const CudaImage<float,3,size_t>& d_tetrahedra ) {
gems/cuda/visitcountersimplecudaimpl.cu:    void RunVisitCounterSimpleCUDA<double,double>( CudaImage<int,3,unsigned short>& d_output,
gems/cuda/visitcountersimplecudaimpl.cu:						   const CudaImage<double,3,size_t>& d_tetrahedra ) {
gems/cuda/visitcountersimplecudaimpl.cu:    void RunVisitCounterSimpleCUDA<float,double>( CudaImage<int,3,unsigned short>& d_output,
gems/cuda/visitcountersimplecudaimpl.cu:						  const CudaImage<float,3,size_t>& d_tetrahedra ) {
gems/cuda/visitcounteraction.hpp:  VisitCounterAction(kvl::cuda::Image_GPU<int,3,unsigned short> target) : output(target) {} 
gems/cuda/visitcounteraction.hpp:  kvl::cuda::Image_GPU<int,3,unsigned short> output;
gems/cuda/visitcountertetrahedralmeshcuda.hpp:#include "cudatetrahedralmesh.hpp"
gems/cuda/visitcountertetrahedralmeshcuda.hpp:#include "visitcountertetrahedralmeshcudaimpl.hpp"
gems/cuda/visitcountertetrahedralmeshcuda.hpp:  namespace cuda {
gems/cuda/visitcountertetrahedralmeshcuda.hpp:	CudaTetrahedralMesh<double,unsigned long,float> ctm;
gems/cuda/visitcountertetrahedralmeshcuda.hpp:	RunVisitCounterTetrahedralMeshCUDA( this->d_Output, ctm );
gems/cuda/visitcountertetrahedralmeshcuda.hpp:	CudaImage<int,3,unsigned short>::DimensionType dims;
gems/cuda/visitcountertetrahedralmeshcuda.hpp:      CudaImage<int,3,unsigned short> d_Output;
gems/cuda/cudaimage.hpp:#include <cuda_runtime.h>
gems/cuda/cudaimage.hpp:#include "dimensioncuda.hpp"
gems/cuda/cudaimage.hpp:#include "cudadeleters.hpp"
gems/cuda/cudaimage.hpp:  namespace cuda {
gems/cuda/cudaimage.hpp:    class Image_GPU {
gems/cuda/cudaimage.hpp:    class CudaImage {
gems/cuda/cudaimage.hpp:	auto extent = this->GetCudaExtent();
gems/cuda/cudaimage.hpp:	auto dst = this->GetCudaPitchedPtr(dest);
gems/cuda/cudaimage.hpp:	cudaMemcpy3DParms cpyPrms = cudaMemcpy3DParms();
gems/cuda/cudaimage.hpp:	cpyPrms.kind = cudaMemcpyDeviceToHost;
gems/cuda/cudaimage.hpp:	CUDA_SAFE_CALL( cudaMemcpy3D(&cpyPrms) );
gems/cuda/cudaimage.hpp:	cudaExtent extent = this->GetCudaExtent();
gems/cuda/cudaimage.hpp:	cudaPitchedPtr src = this->GetCudaPitchedPtr(source);
gems/cuda/cudaimage.hpp:	cudaMemcpy3DParms cpyPrms = cudaMemcpy3DParms();
gems/cuda/cudaimage.hpp:	cpyPrms.kind = cudaMemcpyHostToDevice;
gems/cuda/cudaimage.hpp:	CUDA_SAFE_CALL( cudaMemcpy3D(&cpyPrms) );
gems/cuda/cudaimage.hpp:      cudaExtent GetCudaExtent() const {
gems/cuda/cudaimage.hpp:	// Creates the CudaExtent from the dims
gems/cuda/cudaimage.hpp:	cudaExtent res;
gems/cuda/cudaimage.hpp:      cudaPitchedPtr GetCudaPitchedPtr( const std::vector<ElementType>& target ) const {
gems/cuda/cudaimage.hpp:	cudaPitchedPtr res;
gems/cuda/cudaimage.hpp:	auto tmpExtent = this->GetCudaExtent();
gems/cuda/cudaimage.hpp:	  This is not nice, but unfortunately the cudaMemcpy3D API
gems/cuda/cudaimage.hpp:	  a const_cudaPitchedPtr, the cast wouldn't be necessary
gems/cuda/cudaimage.hpp:      Image_GPU<ElementType,nDims,IndexType> getArg() const {
gems/cuda/cudaimage.hpp:	Image_GPU<ElementType,nDims,IndexType> gpuArg;
gems/cuda/cudaimage.hpp:	  gpuArg.dims[i] = this->dims[i];
gems/cuda/cudaimage.hpp:	gpuArg.pitchedPtr =  this->d_elements->ptr;
gems/cuda/cudaimage.hpp:	gpuArg.dataPitch = this->d_elements->pitch;
gems/cuda/cudaimage.hpp:	return gpuArg;
gems/cuda/cudaimage.hpp:	CUDA_SAFE_CALL( cudaMemset3D( *(this->d_elements), value, this->GetCudaExtent() ) ); 
gems/cuda/cudaimage.hpp:      std::unique_ptr<cudaPitchedPtr,CudaDeviceDeleter> d_elements;
gems/cuda/cudaimage.hpp:	cudaExtent tmpExtent = this->GetCudaExtent();
gems/cuda/cudaimage.hpp:	this->d_elements = std::unique_ptr<cudaPitchedPtr,CudaDeviceDeleter>(new cudaPitchedPtr);
gems/cuda/cudaimage.hpp:	CUDA_SAFE_CALL( cudaMalloc3D(this->d_elements.get(), tmpExtent) );
gems/cuda/visitcountertetrahedralmeshcudaimpl.cu:#include "visitcountertetrahedralmeshcudaimpl.hpp"
gems/cuda/visitcountertetrahedralmeshcudaimpl.cu:  namespace cuda {
gems/cuda/visitcountertetrahedralmeshcudaimpl.cu:    void RunVisitCounterTetrahedralMeshCUDA( CudaImage<int,3,unsigned short>& d_output,
gems/cuda/visitcountertetrahedralmeshcudaimpl.cu:					     const CudaTetrahedralMesh<double,unsigned long,float>& ctm ) {
gems/cuda/atlasmeshalphadrawercudaimpl.hpp:#include "cudaimage.hpp"
gems/cuda/atlasmeshalphadrawercudaimpl.hpp:#include "cudatetrahedralmesh.hpp"
gems/cuda/atlasmeshalphadrawercudaimpl.hpp:  namespace cuda {
gems/cuda/atlasmeshalphadrawercudaimpl.hpp:    void RunAtlasMeshAlphaDrawerCUDA( CudaImage<float,3,unsigned short>& d_output,
gems/cuda/atlasmeshalphadrawercudaimpl.hpp:				      const CudaTetrahedralMesh<double,unsigned long,float>& ctm,
gems/cuda/CMakeLists.txt:set(cudasrcs cudacontroller.cpp)
gems/cuda/CMakeLists.txt:list(APPEND cudasrcs atlasmeshvisitcountercuda.cpp)
gems/cuda/CMakeLists.txt:list(APPEND cudasrcs visitcountersimplecudaimpl.cu)
gems/cuda/CMakeLists.txt:list(APPEND cudasrcs visitcountertetrahedralmeshcudaimpl.cu)
gems/cuda/CMakeLists.txt:list(APPEND cudasrcs atlasmeshalphadrawercuda.cpp)
gems/cuda/CMakeLists.txt:list(APPEND cudasrcs atlasmeshalphadrawercudaimpl.cu)
gems/cuda/CMakeLists.txt:cuda_add_library(kvlGEMSCUDA ${cudasrcs})
gems/cuda/visitcountersimplecuda.hpp:#include "dimensioncuda.hpp"
gems/cuda/visitcountersimplecuda.hpp:#include "cudaimage.hpp"
gems/cuda/visitcountersimplecuda.hpp:#include "visitcountersimplecudaimpl.hpp"
gems/cuda/visitcountersimplecuda.hpp:  namespace cuda {
gems/cuda/visitcountersimplecuda.hpp:	CudaImage<T,3,size_t> d_tetrahedra;
gems/cuda/visitcountersimplecuda.hpp:	RunVisitCounterSimpleCUDA<T,Internal>( d_Output, d_tetrahedra );
gems/cuda/visitcountersimplecuda.hpp:	CudaImage<int,3,unsigned short>::DimensionType dims;
gems/cuda/visitcountersimplecuda.hpp:      CudaImage<int,3,unsigned short> d_Output;
gems/cuda/atlasmeshvisitcountercuda.cpp:#include "atlasmeshvisitcountercuda.hpp"
gems/cuda/atlasmeshvisitcountercuda.cpp:  namespace cuda {
gems/cuda/atlasmeshvisitcountercuda.cpp:    void AtlasMeshVisitCounterCUDA::SetRegions( const kvl::interfaces::AtlasMeshVisitCounter::ImageType::RegionType& region ) {
gems/cuda/atlasmeshvisitcountercuda.cpp:    void AtlasMeshVisitCounterCUDA::VisitCount( const kvl::AtlasMesh* mesh ) {
gems/cuda/atlasmeshvisitcountercuda.cpp:    const AtlasMeshVisitCounterCUDA::ImageType* AtlasMeshVisitCounterCUDA::GetImage() const {
gems/cuda/atlasmeshvisitcountercuda.hpp:  namespace cuda {
gems/cuda/atlasmeshvisitcountercuda.hpp:    class AtlasMeshVisitCounterCUDA : public kvl::interfaces::AtlasMeshVisitCounter {
gems/cuda/atlasmeshvisitcountercuda.hpp:      AtlasMeshVisitCounterCUDA() {}
gems/cuda/atlasmeshvisitcountercuda.hpp:      virtual const AtlasMeshVisitCounterCUDA::ImageType*  GetImage() const override;
gems/cuda/cudautils.hpp:  namespace cuda {
gems/cuda/visitcountersimplecudaimpl.hpp:#include "cudaimage.hpp"
gems/cuda/visitcountersimplecudaimpl.hpp:  namespace cuda {
gems/cuda/visitcountersimplecudaimpl.hpp:    void RunVisitCounterSimpleCUDA( CudaImage<int,3,unsigned short>& d_output,
gems/cuda/visitcountersimplecudaimpl.hpp:				    const CudaImage<T,3,size_t>& d_tetrahedra ) {
gems/cuda/visitcountersimplecudaimpl.hpp:      throw std::runtime_error("Must call RunVisitCounterSimpleCUDA with float or double");
gems/cuda/visitcountersimplecudaimpl.hpp:    void RunVisitCounterSimpleCUDA<float,float>( CudaImage<int,3,unsigned short>& d_output,
gems/cuda/visitcountersimplecudaimpl.hpp:						 const CudaImage<float,3,size_t>& d_tetrahedra );
gems/cuda/visitcountersimplecudaimpl.hpp:    void RunVisitCounterSimpleCUDA<double,double>( CudaImage<int,3,unsigned short>& d_output,
gems/cuda/visitcountersimplecudaimpl.hpp:						   const CudaImage<double,3,size_t>& d_tetrahedra );
gems/cuda/visitcountersimplecudaimpl.hpp:    void RunVisitCounterSimpleCUDA<float,double>( CudaImage<int,3,unsigned short>& d_output,
gems/cuda/visitcountersimplecudaimpl.hpp:						  const CudaImage<float,3,size_t>& d_tetrahedra );
gems/cuda/atlasmeshalphadrawercuda.hpp:#include "cudatetrahedralmesh.hpp"
gems/cuda/atlasmeshalphadrawercuda.hpp:  namespace cuda {
gems/cuda/atlasmeshalphadrawercuda.hpp:    class AtlasMeshAlphaDrawerCUDA : public kvl::interfaces::AtlasMeshAlphaDrawer {
gems/cuda/atlasmeshalphadrawercuda.hpp:      AtlasMeshAlphaDrawerCUDA() {}
gems/cuda/atlasmeshalphadrawercuda.hpp:      virtual const AtlasMeshAlphaDrawerCUDA::ImageType* GetImage() const override;
gems/cuda/atlasmeshalphadrawercuda.hpp:      CudaImage<float,3,unsigned short> d_Output;
gems/cuda/atlasmeshalphadrawercuda.hpp:      AtlasMeshAlphaDrawerCUDA::ImageType::Pointer image;
gems/cuda/atlasmeshalphadrawercudaimpl.cu:#include "atlasmeshalphadrawercudaimpl.hpp"
gems/cuda/atlasmeshalphadrawercudaimpl.cu:  namespace cuda {
gems/cuda/atlasmeshalphadrawercudaimpl.cu:    void RunAtlasMeshAlphaDrawerCUDA( CudaImage<float,3,unsigned short>& d_output,
gems/cuda/atlasmeshalphadrawercudaimpl.cu:				      const CudaTetrahedralMesh<double,unsigned long,float>& ctm,
gems/cuda/dimensioncuda.hpp:  namespace cuda {
gems/cuda/cudaexception.hpp:#include <cuda_runtime.h>
gems/cuda/cudaexception.hpp:  namespace cuda {
gems/cuda/cudaexception.hpp:    class CUDAException : public std::runtime_error {
gems/cuda/cudaexception.hpp:      cudaError errorCode;
gems/cuda/cudaexception.hpp:      CUDAException(const cudaError error) : errorCode(error),
gems/cuda/cudaexception.hpp:					     runtime_error(cudaGetErrorString(errorCode)) {}
gems/cuda/cudacontroller.cpp:#include <cuda_runtime.h>
gems/cuda/cudacontroller.cpp:#include "cudacheck.hpp"
gems/cuda/cudacontroller.cpp:#include "cudacontroller.hpp"
gems/cuda/cudacontroller.cpp:  namespace cuda {
gems/cuda/cudacontroller.cpp:    void InitialiseCUDA(const int deviceID) {
gems/cuda/cudacontroller.cpp:      CUDA_SAFE_CALL( cudaSetDevice(deviceID) );
gems/cuda/cudacontroller.cpp:      CUDA_SAFE_CALL( cudaMalloc( &d_tmp, 1 ) );
gems/cuda/cudacontroller.cpp:      CUDA_SAFE_CALL( cudaFree( d_tmp ) );
gems/cuda/cudacontroller.cpp:    void FinalizeCUDA() {
gems/cuda/cudacheck.hpp:#include "cudaexception.hpp"
gems/cuda/cudacheck.hpp:#define CUDA_SAFE_CALL( call ) do {		\
gems/cuda/cudacheck.hpp:    cudaError err = call;			\
gems/cuda/cudacheck.hpp:    if( cudaSuccess != err ) {			\
gems/cuda/cudacheck.hpp:      throw kvl::cuda::CUDAException(err);	\
gems/cuda/cudacheck.hpp:    err = cudaDeviceSynchronize();		\
gems/cuda/cudacheck.hpp:    if( cudaSuccess != err ) {			\
gems/cuda/cudacheck.hpp:      throw kvl::cuda::CUDAException(err);	\
gems/cuda/cudatetrahedralmesh.hpp:#include "cudaimage.hpp"
gems/cuda/cudatetrahedralmesh.hpp:  namespace cuda {
gems/cuda/cudatetrahedralmesh.hpp:    class TetrahedralMesh_GPU {
gems/cuda/cudatetrahedralmesh.hpp:      Image_GPU<CoordinateType,2,MeshIndexType> vertices;
gems/cuda/cudatetrahedralmesh.hpp:      Image_GPU<MeshIndexType,2,MeshIndexType> vertexMap;
gems/cuda/cudatetrahedralmesh.hpp:      Image_GPU<AlphasType,2,MeshIndexType> alphas;
gems/cuda/cudatetrahedralmesh.hpp:    class CudaTetrahedralMesh {
gems/cuda/cudatetrahedralmesh.hpp:      typedef TetrahedralMesh_GPU<CoordinateType,MeshIndexType,AlphasType> GPUType;
gems/cuda/cudatetrahedralmesh.hpp:	// Transfer data to the GPU
gems/cuda/cudatetrahedralmesh.hpp:      TetrahedralMesh_GPU<CoordinateType,MeshIndexType,AlphasType> getArg() const {
gems/cuda/cudatetrahedralmesh.hpp:	TetrahedralMesh_GPU<CoordinateType,MeshIndexType,AlphasType> gpuArg;
gems/cuda/cudatetrahedralmesh.hpp:	gpuArg.vertices = this->d_vertices.getArg();
gems/cuda/cudatetrahedralmesh.hpp:	gpuArg.vertexMap = this->d_vertexMap.getArg();
gems/cuda/cudatetrahedralmesh.hpp:	gpuArg.alphas = this->d_alphas.getArg();
gems/cuda/cudatetrahedralmesh.hpp:	return gpuArg;
gems/cuda/cudatetrahedralmesh.hpp:      CudaImage<CoordinateType,2,MeshIndexType> d_vertices;
gems/cuda/cudatetrahedralmesh.hpp:      CudaImage<MeshIndexType,2,MeshIndexType> d_vertexMap;
gems/cuda/cudatetrahedralmesh.hpp:      CudaImage<AlphasType,2,MeshIndexType> d_alphas;
gems/cuda/atlasmeshalphadrawercuda.cpp:#include "atlasmeshalphadrawercuda.hpp"
gems/cuda/atlasmeshalphadrawercuda.cpp:#include "atlasmeshalphadrawercudaimpl.hpp"
gems/cuda/atlasmeshalphadrawercuda.cpp:  namespace cuda {
gems/cuda/atlasmeshalphadrawercuda.cpp:    void AtlasMeshAlphaDrawerCUDA::SetRegions( const ImageType::RegionType&  region ) {
gems/cuda/atlasmeshalphadrawercuda.cpp:	this->image = AtlasMeshAlphaDrawerCUDA::ImageType::New();
gems/cuda/atlasmeshalphadrawercuda.cpp:    void AtlasMeshAlphaDrawerCUDA::Interpolate( const kvl::AtlasMesh* mesh ) {
gems/cuda/atlasmeshalphadrawercuda.cpp:      CudaTetrahedralMesh<double,unsigned long,float> ctm;
gems/cuda/atlasmeshalphadrawercuda.cpp:      RunAtlasMeshAlphaDrawerCUDA( this->d_Output, ctm, this->classNumber );
gems/cuda/atlasmeshalphadrawercuda.cpp:    const AtlasMeshAlphaDrawerCUDA::ImageType* AtlasMeshAlphaDrawerCUDA::GetImage() const {
gems/cuda/atlasmeshalphadrawercuda.cpp:      CudaImage<float,3,unsigned short>::DimensionType dims;
gems/cuda/atlasmeshalphadrawercuda.cpp:    void AtlasMeshAlphaDrawerCUDA::SetClassNumber( const int targetClass ) {
python/fsdeeplearn/metrics.py:            #   It works on GPU because we do not perform index validation checking on GPU -- it's too
python/fsdeeplearn/models.py:#from tensorflow.keras.utils import multi_gpu_model
python/fsdeeplearn/models.py:from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
python/fsdeeplearn/models.py:               loss='mean_squared_error', initial_learning_rate=0.00001, deconvolution=False, num_gpus=1,
python/fsdeeplearn/models.py:                                                        loss, num_gpus, initial_learning_rate)
python/fsdeeplearn/models.py:                        pool_size=None, n_labels=0, num_outputs=1, num_gpus=1, GPnet=None,pooling='max',
python/fsdeeplearn/models.py:def build_compile_model(input_layer, input_shape, conv, n_labels, loss, num_gpus, initial_learning_rate):
python/fsdeeplearn/models.py:        if num_gpus > 1:
python/fsdeeplearn/models.py:                parallel_model = multi_gpu_model(model, gpus=num_gpus)
python/fsdeeplearn/models.py:            if num_gpus > 1:
python/fsdeeplearn/models.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
python/fsdeeplearn/models.py:            if num_gpus > 1:
python/fsdeeplearn/models.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
python/fsdeeplearn/models.py:                 num_gpus=1):
python/fsdeeplearn/models.py:        if num_gpus > 1:
python/fsdeeplearn/models.py:                parallel_model = multi_gpu_model(model, gpus=num_gpus)
python/fsdeeplearn/models.py:            if num_gpus > 1:
python/fsdeeplearn/models.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
python/fsdeeplearn/models.py:            if num_gpus > 1:
python/fsdeeplearn/models.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
python/fsdeeplearn/models.py:#                        num_gpus=1, num_outputs=1):
python/fsdeeplearn/models.py:#         if num_gpus > 1:
python/fsdeeplearn/models.py:#                 parallel_model = multi_gpu_model(model, gpus=num_gpus)
python/fsdeeplearn/models.py:#             if num_gpus > 1:
python/fsdeeplearn/models.py:#                     parallel_model = multi_gpu_model(model, gpus=num_gpus)
python/fsdeeplearn/models.py:#             if num_gpus > 1:
python/fsdeeplearn/models.py:#                     parallel_model = multi_gpu_model(model, gpus=num_gpus)
python/fsdeeplearn/models.py:#                   num_gpus=1, num_outputs=1):
python/fsdeeplearn/models.py:#         if num_gpus > 1:
python/fsdeeplearn/models.py:#                 parallel_model = multi_gpu_model(model, gpus=num_gpus)
python/fsdeeplearn/models.py:#             if num_gpus > 1:
python/fsdeeplearn/models.py:#                     parallel_model = multi_gpu_model(model, gpus=num_gpus)
python/fsdeeplearn/models.py:#             if num_gpus > 1:
python/fsdeeplearn/models.py:#                     parallel_model = multi_gpu_model(model, gpus=num_gpus)
python/fsdeeplearn/models.py:#                   num_gpus=1):
python/fsdeeplearn/models.py:#         if num_gpus > 1:
python/fsdeeplearn/models.py:#                 parallel_model = multi_gpu_model(model, gpus=num_gpus)
python/fsdeeplearn/models.py:#             if num_gpus > 1:
python/fsdeeplearn/models.py:#                     parallel_model = multi_gpu_model(model, gpus=num_gpus)
python/fsdeeplearn/models.py:#             if num_gpus > 1:
python/fsdeeplearn/models.py:#                     parallel_model = multi_gpu_model(model, gpus=num_gpus)
python/fsdeeplearn/scripts/train_ev_atlases.py:gpu_id = -1
python/fsdeeplearn/scripts/train_ev_atlases.py:if os.getenv('NGPUS'):
python/fsdeeplearn/scripts/train_ev_atlases.py:    ngpus = int(os.getenv('NGPUS'))
python/fsdeeplearn/scripts/train_ev_atlases.py:#    ngpus=1
python/fsdeeplearn/scripts/train_ev_atlases.py:    gpu_str = '0'
python/fsdeeplearn/scripts/train_ev_atlases.py:    for g in range(1,ngpus):
python/fsdeeplearn/scripts/train_ev_atlases.py:        gpu_str += ',%d' % g
python/fsdeeplearn/scripts/train_ev_atlases.py:    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
python/fsdeeplearn/scripts/train_ev_atlases.py:    print('reading %d GPUS from env and setting CUDA_VISIBLE_DEVICES to %s' % (ngpus, gpu_str))
python/fsdeeplearn/scripts/train_ev_atlases.py:elif os.getenv('CUDA_VISIBLE_DEVICES'):
python/fsdeeplearn/scripts/train_ev_atlases.py:    gpu_list = os.getenv('CUDA_VISIBLE_DEVICES')
python/fsdeeplearn/scripts/train_ev_atlases.py:    ngpus = len(gpu_list.split(','))
python/fsdeeplearn/scripts/train_ev_atlases.py:    gpu_id = 0
python/fsdeeplearn/scripts/train_ev_atlases.py:    fsd.configure(gpu=gpu_id)
python/fsdeeplearn/scripts/train_ev_atlases.py:if gpu_id >= 0:
python/fsdeeplearn/scripts/train_ev_atlases.py:    print('using gpu %d on host %s' % (gpu_id, host))
python/fsdeeplearn/scripts/train_ev_atlases.py:    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"]
python/fsdeeplearn/scripts/train_ev_atlases.py:    ngpus = len(gpu_list.split(','))
python/fsdeeplearn/scripts/train_ev_atlases.py:    batch_size = (batch_size // ngpus) * ngpus
python/fsdeeplearn/scripts/train_ev_atlases.py:    print('using %d gpus %s on host %s with batch_size %d' % (ngpus,gpu_list, host, batch_size))
python/fsdeeplearn/scripts/make_semi_folding_atlas.py:gpu_id = -1
python/fsdeeplearn/scripts/make_semi_folding_atlas.py:ngpus = 1 if os.getenv('NGPUS') is None else int(os.getenv('NGPUS'))
python/fsdeeplearn/scripts/make_semi_folding_atlas.py:print(f'using {ngpus} gpus')
python/fsdeeplearn/scripts/make_semi_folding_atlas.py:if ngpus > 1:
python/fsdeeplearn/scripts/make_semi_folding_atlas.py:    model_device = '/gpu:0'
python/fsdeeplearn/scripts/make_semi_folding_atlas.py:    synth_device = '/gpu:1'
python/fsdeeplearn/scripts/make_semi_folding_atlas.py:    synth_gpu = 1
python/fsdeeplearn/scripts/make_semi_folding_atlas.py:    model_device = '/gpu:0'
python/fsdeeplearn/scripts/make_semi_folding_atlas.py:    synth_gpu = -1
python/fsdeeplearn/scripts/make_semi_folding_atlas.py:os.environ["CUDA_VISIBLE_DEVICES"] = dev_str
python/fsdeeplearn/scripts/make_semi_folding_atlas.py:print(f'physical GPU # is {os.getenv("SLURM_STEP_GPUS")}')
python/fsdeeplearn/scripts/make_semi_folding_atlas.py:#ngpus = strategy.num_replicas_in_sync
python/fsdeeplearn/scripts/make_semi_folding_atlas.py:batch_size = (batch_size // ngpus) * ngpus
python/fsdeeplearn/scripts/make_semi_folding_atlas.py:print('found %d gpus after configuring device, resetting batch_size to %d' % (ngpus, batch_size))
python/fsdeeplearn/scripts/make_semi_folding_atlas.py:vbatch = int((1+(nval // ngpus)) * ngpus)
python/fsdeeplearn/scripts/train_ev_semi.py:gpuid = -1
python/fsdeeplearn/scripts/train_ev_semi.py:if os.getenv('NGPUS'):
python/fsdeeplearn/scripts/train_ev_semi.py:    ngpus = int(os.getenv('NGPUS'))
python/fsdeeplearn/scripts/train_ev_semi.py:#    ngpus=1
python/fsdeeplearn/scripts/train_ev_semi.py:    gpu_str = '0'
python/fsdeeplearn/scripts/train_ev_semi.py:    for g in range(1,ngpus):
python/fsdeeplearn/scripts/train_ev_semi.py:        gpu_str += ',%d' % g
python/fsdeeplearn/scripts/train_ev_semi.py:    if ngpus == 1:
python/fsdeeplearn/scripts/train_ev_semi.py:        gpuid = 0
python/fsdeeplearn/scripts/train_ev_semi.py:    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
python/fsdeeplearn/scripts/train_ev_semi.py:    print('reading %d GPUS from env and setting CUDA_VISIBLE_DEVICES to %s' % (ngpus, gpu_str))
python/fsdeeplearn/scripts/train_ev_semi.py:elif os.getenv('CUDA_VISIBLE_DEVICES'):
python/fsdeeplearn/scripts/train_ev_semi.py:    gpu_list = os.getenv('CUDA_VISIBLE_DEVICES')
python/fsdeeplearn/scripts/train_ev_semi.py:    ngpus = len(gpu_list.split(','))
python/fsdeeplearn/scripts/train_ev_semi.py:    gpuid = 0
python/fsdeeplearn/scripts/train_ev_semi.py:    fsd.configure(gpu=gpuid)
python/fsdeeplearn/scripts/train_ev_semi.py:if gpuid >= 0:
python/fsdeeplearn/scripts/train_ev_semi.py:    print('using gpu %d on host %s' % (gpuid, host))
python/fsdeeplearn/scripts/train_ev_semi.py:    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"]
python/fsdeeplearn/scripts/train_ev_semi.py:    ngpus = len(gpu_list.split(','))
python/fsdeeplearn/scripts/train_ev_semi.py:    batch_size = (batch_size // ngpus) * ngpus
python/fsdeeplearn/scripts/train_ev_semi.py:    print('using %d gpus %s on host %s with batch_size %d' % (ngpus,gpu_list, host, batch_size))
python/fsdeeplearn/scripts/train_ev_semi.py:# device, ngpus = ne.utils.setup_device(gpuid=gpuid)
python/fsdeeplearn/scripts/train_parcellation.py:gpu_id = -1
python/fsdeeplearn/scripts/train_parcellation.py:if os.getenv('NGPUS'):
python/fsdeeplearn/scripts/train_parcellation.py:    ngpus = int(os.getenv('NGPUS'))
python/fsdeeplearn/scripts/train_parcellation.py:    ngpus=1
python/fsdeeplearn/scripts/train_parcellation.py:    gpu_str = '0'
python/fsdeeplearn/scripts/train_parcellation.py:    for g in range(1,ngpus):
python/fsdeeplearn/scripts/train_parcellation.py:        gpu_str += ',%d' % g
python/fsdeeplearn/scripts/train_parcellation.py:    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
python/fsdeeplearn/scripts/train_parcellation.py:    print('reading %d GPUS from env and setting CUDA_VISIBLE_DEVICES to %s' % (ngpus, gpu_str))
python/fsdeeplearn/scripts/train_parcellation.py:elif os.getenv('CUDA_VISIBLE_DEVICES'):  # need to implement mirrored strategy for this
python/fsdeeplearn/scripts/train_parcellation.py:    gpu_list = os.getenv('CUDA_VISIBLE_DEVICES')
python/fsdeeplearn/scripts/train_parcellation.py:    ngpus = len(gpu_list.split(','))
python/fsdeeplearn/scripts/train_parcellation.py:    gpu_id = 0
python/fsdeeplearn/scripts/train_parcellation.py:    fsd.configure(gpu=gpu_id)
python/fsdeeplearn/scripts/train_parcellation.py:if gpu_id >= 0:
python/fsdeeplearn/scripts/train_parcellation.py:    print('using gpu %d on host %s' % (gpu_id, host))
python/fsdeeplearn/scripts/train_parcellation.py:    gpu_list = os.environ["CUDA_VISIBLE_DEVICES"]
python/fsdeeplearn/scripts/train_parcellation.py:    ngpus = len(gpu_list.split(','))
python/fsdeeplearn/scripts/train_parcellation.py:    batch_size = (batch_size // ngpus) * ngpus
python/fsdeeplearn/scripts/train_parcellation.py:    print('using %d gpus %s on host %s with batch_size %d' % (ngpus,gpu_list, host, batch_size))
python/fsdeeplearn/utils.py:def configure(gpu=0):
python/fsdeeplearn/utils.py:    Configures the appropriate TF device from a cuda device integer.
python/fsdeeplearn/utils.py:    gpuid = str(gpu)
python/fsdeeplearn/utils.py:    if gpuid is not None and (gpuid != '-1'):
python/fsdeeplearn/utils.py:        device = '/gpu:' + gpuid
python/fsdeeplearn/utils.py:        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
python/fsdeeplearn/utils.py:        # GPU memory configuration differs between TF 1 and 2
python/fsdeeplearn/utils.py:            config.gpu_options.allow_growth = True
python/fsdeeplearn/utils.py:            for pd in tf.config.list_physical_devices('GPU'):
python/fsdeeplearn/utils.py:        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
python/fsdeeplearn/deepnet.py:class MultiGPUCheckpointCallback(tf.keras.callbacks.Callback):
python/fsdeeplearn/deepnet.py:        super(MultiGPUCheckpointCallback, self).__init__()
python/fsdeeplearn/deepnet.py:                 fcn=True, num_gpus=1, preprocessing=False, augment=False, num_outputs=1,
python/fsdeeplearn/deepnet.py:        self.num_gpus = num_gpus
python/fsdeeplearn/deepnet.py:                                                            num_gpus=num_gpus)
python/fsdeeplearn/deepnet.py:                                                         num_gpus=num_gpus)
python/fsdeeplearn/deepnet.py:                                                            num_gpus=num_gpus, num_outputs=num_outputs)
python/fsdeeplearn/deepnet.py:                  wmp_standardize=True, rob_standardize=True, fcn=True, num_gpus=1,
python/fsdeeplearn/deepnet.py:                           wmp_standardize=wmp_standardize, rob_standardize=rob_standardize, fcn=fcn, num_gpus=1,
python/fsdeeplearn/deepnet.py:            if self.num_gpus == 1:
python/fsdeeplearn/deepnet.py:                                batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                                batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                                batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                                batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                                batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                                batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                            generator=self.feature_generator.training_generator(batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                                batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                callback = MultiGPUCheckpointCallback(output_prefix, save_per_epoch, save_weights, initial_epoch)
python/fsdeeplearn/deepnet.py:                    generator=self.feature_generator.training_generator(batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                    validation_data=self.feature_generator.validation_generator(batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:            if self.num_gpus == 1:
python/fsdeeplearn/deepnet.py:                    # self.model.fit_generator(generator=self.feature_generator.seg_training_generator_multichannel_nmr_t2(batch_size=batch_size*self.num_gpus),
python/fsdeeplearn/deepnet.py:                    #                  validation_data=self.feature_generator.seg_validation_generator_multichannel_nmr(batch_size=batch_size*self.num_gpus),
python/fsdeeplearn/deepnet.py:                            batch_size=batch_size * self.num_gpus,
python/fsdeeplearn/deepnet.py:                            batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                            batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                            batch_size=batch_size * self.num_gpus, focus=focus),
python/fsdeeplearn/deepnet.py:                            batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                            batch_size=batch_size * self.num_gpus, focus=focus),
python/fsdeeplearn/deepnet.py:                            batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                            batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                            batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                            batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                            batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                callback = MultiGPUCheckpointCallback(output_prefix, save_per_epoch, save_weights, initial_epoch)
python/fsdeeplearn/deepnet.py:                    generator=self.feature_generator.training_generator(batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                    validation_data=self.feature_generator.validation_generator(batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                generator=self.feature_generator.training_label_generator(batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:                    batch_size=batch_size * self.num_gpus),
python/fsdeeplearn/deepnet.py:        # if self.num_gpus > 1:
python/fsdeeplearn/deepnet.py:        if self.num_gpus > 1:
python/fsdeeplearn/deepnet.py:        if self.num_gpus > 1:
python/fsdeeplearn/deepnet.py:        if self.num_gpus > 1:
python/fsdeeplearn/deepnet.py:        if self.num_gpus > 1:
python/req.sh:nvidia_subdir="$install_path/python/lib/python3.8/site-packages/nvidia"
python/req.sh:   ## comment out entries not available on MacOS (nvidia, triton)
python/req.sh:   # $python_binary -m pip freeze | sort | uniq | sed 's; @ ;@;g' | sed 's;^qatools.*;#&;' | sed 's;^pyfs.*;#&;' | sed 's;^nvidia.*;#&;' | sed 's;^triton.*;#&;' > $build_req_new
python/req.sh:   # $python_binary -m pip freeze | sort | uniq | sed 's; @ ;@;g' | sed 's;^qatools.*;#&;' | sed 's;^pyfs.*;#&;' | sed 's;^nvidia.*;#&;' | sed 's;^triton.*;#&;' | sed 's;voxelmorph==.*;'${voxelmorph_url_when_version_invalid}';' | sed 's;neurite==.*;'${neurite_url_when_version_invalid}';' | sed 's;pystrum==.*;'${pystrum_url_when_version_invalid}';' | sed 's;surfa==.*;'${surfa_url_when_version_invalid}';' > $build_req_new
python/req.sh:   # $python_binary -m pip freeze | sort | uniq | sed 's; @ ;@;g' | sed 's;^qatools.*;#&;' | sed 's;^pyfs.*;#&;' | sed 's;^nvidia.*;#&;' | sed 's;^triton.*;#&;' | sed 's;voxelmorph==.*;'${voxelmorph_url_when_version_invalid}';' | sed 's;neurite==.*;'${neurite_url_when_version_invalid}';' | sed 's;pystrum==.*;'${pystrum_url_when_version_invalid}';' > $build_req_new
python/req.sh:   $python_binary -m pip freeze | sort | uniq | sed 's; @ ;@;g' | sed 's;^qatools.*;#&;' | sed 's;^pyfs.*;#&;' | sed 's;^nvidia.*;#&;' | sed 's;^triton.*;#&;' | sed 's;^fsutil.*;#&;' | sed 's;voxelmorph==.*;'${voxelmorph_url_when_version_invalid}';' | sed 's;neurite==.*;'${neurite_url_when_version_invalid}';' | sed 's;pystrum==.*;'${pystrum_url_when_version_invalid}';' | sed 's;spheremorph==.*;'${spheremorph_url_when_version_invalid}';' | sed 's;^torch==.*cpu;--find-links '${torch_cpu_url}'\n&;' > $build_req_new
python/req.sh:   ## check contents of nvidia directory
python/req.sh:   # echo $s: "Contents of nvidia subdir BEFORE UNINSTALL"
python/req.sh:   # if [ -e $nvidia_subdir ]; then ls $nvidia_subdir; fi
python/req.sh:   ## remove nvidia packages with cuda libs (installed as dependency on linux but not MacOS)
python/req.sh:   ## replace torch with torch+cpu version (no cuda libs) via --libtorch arg above
python/req.sh:   # $python_binary -m pip freeze | grep "^nvidia\|^triton\|^torch" > /dev/null
python/req.sh:   $python_binary -m pip freeze | grep "^nvidia\|^triton" > /dev/null
python/req.sh:      # $python_binary -m pip freeze | grep '^nvidia\|^triton\|^torch' | sed 's;==.*;;' >> postinstall.list
python/req.sh:      $python_binary -m pip freeze | grep '^nvidia\|^triton' | sed 's;==.*;;' >> postinstall.list
python/req.sh:      echo "$s: Found nothing to uninstall for nvidia and triton in output from pip freeze."
python/req.sh:      (cd ${tflow_subdir} && find -type f ! -name "*LICENSE*" ! -name "*.so*" -exec grep -i "copyright.*nvidia" {} \; -print | grep "^\.\/") > header.list
python/req.sh:           echo "$s: Removing $file_cnt NVIDIA copyrighted source files"
python/req.sh:           (cd ${tflow_subdir} && find -type f ! -name "*LICENSE*" ! -name "*.so*" -exec grep -i "copyright.*nvidia" {} \; -print | grep "^\.\/") > header.list
python/req.sh:      echo "%s: tensorflow does not appear to be installed to check for NVIDIA source files"
python/req.sh:   ## check contents of nvidia directory
python/req.sh:   # echo "$s: Contents of nvidia subdir AFTER UNINSTALL"
python/req.sh:   # if [ -e $nvidia_subdir ]; then ls $nvidia_subdir; fi
python/req.sh:      ## 03/2024 - exclude nvidia-cudnn-cu12 which breaks installation on Ubuntu linux
python/req.sh:      cat postinstall.list | grep -v "nvidia-cudnn-cu12" | tr -s '\n' ' ' >> postinstall.sh
python/req.sh:      ## check contents of nvidia directory
python/req.sh:      # echo "$s: Contents of nvidia subdir BEFORE REINSTALL"
python/req.sh:      # if [ -e $nvidia_subdir ]; then ls $nvidia_subdir; fi
python/req.sh:      ## check contents of nvidia directory
python/req.sh:      # echo "$s: Contents of nvidia subdir AFTER REINSTALL"
python/req.sh:      # if [ -e $nvidia_subdir ]; then ls $nvidia_subdir; fi
python/CMakeLists.txt:function(prune_cuda)
python/CMakeLists.txt:      message(STATUS \"Nothing to prune for Cuda in fspython on MacOS\")
python/CMakeLists.txt:   if(NOT FSPYTHON_INSTALL_CUDA)
python/CMakeLists.txt:      prune_cuda()
python/CMakeLists.txt:   if(NOT FSPYTHON_INSTALL_CUDA)
python/CMakeLists.txt:      prune_cuda()
distribution/docs/license.tensorflow-python-3rd-party.txt:./demos/opengl/gpuhelper.h
distribution/docs/license.tensorflow-python-3rd-party.txt:./demos/opengl/gpuhelper.cpp
distribution/docs/license.anaconda-miniconda-python.txt:Miniconda also provides access to cuDNN software binaries ("cuDNN binaries") from NVIDIA Corporation. You are specifically authorized to use the cuDNN binaries with your installation of Miniconda subject to your compliance with the license agreement located at https://docs.nvidia.com/deeplearning/sdk/cudnn-sla/index.html. You are also authorized to redistribute the cuDNN binaries with an Miniconda package that contains the cuDNN binaries. You can add or remove the cuDNN binaries utilizing the install and uninstall features in Miniconda.
distribution/docs/license.anaconda-miniconda-python.txt:cuDNN binaries contain source code provided by NVIDIA Corporation.
distribution/etc/recon-config.yaml:UseGPU:
distribution/etc/recon-config.yaml:    flags: -gpu
distribution/etc/recon-config.yaml:    descr: Use GPU for those GPU-enabled programs 
mri_histo_util/mri_histo_atlas_segment:  echo "   mri_histo_atlas_segment INPUT_SCAN OUTPUT_DIRECTORY ATLAS_MODE GPU THREADS [BF_MODE] [GMM_MODE]"
mri_histo_util/mri_histo_atlas_segment:  echo "GPU: set to 1 to use the GPU (*highly* recommended but requires a 24GB GPU!)"
mri_histo_util/mri_histo_atlas_segment:  echo "   mri_histo_atlas_segment INPUT_SCAN OUTPUT_DIRECTORY ATLAS_MODE GPU THREADS BF_MODE GMM_MODE"
mri_histo_util/README.md:segment.sh INPUT_SCAN OUTPUT_DIRECTORY ATLAS_MODE GPU THREADS [BF_MODE] [GMM_MODE]
mri_histo_util/README.md:- GPU: set to 1 to use the GPU (*highly* recommended but requires a 24GB GPU!)
mri_histo_util/README.md:Also, Using a GPU (minimum memory: 24GB) is highly recommended. On the GPU, the code runs in about an hour (30 mins/hemisphere).
mri_histo_util/README.md:use many (>10) threads! Even if you use the GPU, we recommend using a bunch of CPU threads (e.g., 8) if possible, so the CPU 
mri_histo_util/ERC_bayesian_segmentation/ext/LBFGS.py:                if torch.cuda.is_available():
mri_histo_util/ERC_bayesian_segmentation/ext/LBFGS.py:                    F_prev = torch.tensor(np.nan, dtype=dtype).cuda()
mri_histo_util/ERC_bayesian_segmentation/ext/LBFGS.py:                if (torch.cuda.is_available()):
mri_histo_util/ERC_bayesian_segmentation/ext/LBFGS.py:                    F_b = torch.tensor(np.nan, dtype=dtype).cuda()
mri_histo_util/ERC_bayesian_segmentation/ext/LBFGS.py:                    g_b = torch.tensor(np.nan, dtype=dtype).cuda()
mri_histo_util/ERC_bayesian_segmentation/ext/LBFGS.py:                        if torch.cuda.is_available():
mri_histo_util/ERC_bayesian_segmentation/ext/LBFGS.py:                            g_b = torch.tensor(np.nan, dtype=dtype).cuda()
mri_histo_util/ERC_bayesian_segmentation/ext/interpol/autograd.py:    from torch.cuda.amp import custom_fwd, custom_bwd
mri_histo_util/ERC_bayesian_segmentation/ext/interpol/tests/test_gradcheck_pushpull.py:if torch.cuda.is_available():
mri_histo_util/ERC_bayesian_segmentation/ext/interpol/tests/test_gradcheck_pushpull.py:    print('cuda backend available')
mri_histo_util/ERC_bayesian_segmentation/ext/interpol/tests/test_gradcheck_pushpull.py:    devices.append('cuda')
mri_histo_util/ERC_bayesian_segmentation/ext/interpol/tests/test_gradcheck_pushpull.py:    if device == 'cuda':
mri_histo_util/ERC_bayesian_segmentation/ext/interpol/tests/test_gradcheck_pushpull.py:        torch.cuda.set_device(param)
mri_histo_util/ERC_bayesian_segmentation/ext/interpol/tests/test_gradcheck_pushpull.py:        torch.cuda.init()
mri_histo_util/ERC_bayesian_segmentation/ext/interpol/tests/test_gradcheck_pushpull.py:            torch.cuda.empty_cache()
mri_histo_util/ERC_bayesian_segmentation/ext/interpol/utils.py:    current_version, *cuda_variant = torch.__version__.split('+')
mri_histo_util/ERC_bayesian_segmentation/ext/bias_field_correction_torch.py:    torch.cuda.empty_cache()
mri_histo_util/ERC_bayesian_segmentation/ext/my_functions.py:        torch.cuda.empty_cache()
mri_histo_util/ERC_bayesian_segmentation/ext/my_functions.py:# GPU
mri_histo_util/ERC_bayesian_segmentation/ERC_bayesian_segmentation/networks.py:        torch.cuda.empty_cache()
mri_histo_util/ERC_bayesian_segmentation/ERC_bayesian_segmentation/networks.py:            torch.cuda.empty_cache()
mri_histo_util/ERC_bayesian_segmentation/scripts/segment.py:parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
mri_histo_util/ERC_bayesian_segmentation/scripts/segment.py:    os.environ["CUDA_VISIBLE_DEVICES"] = ""
mri_histo_util/ERC_bayesian_segmentation/scripts/segment.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mri_histo_util/ERC_bayesian_segmentation/scripts/segment.py:    os.environ["CUDA_VISIBLE_DEVICES"] = ""
mri_histo_util/ERC_bayesian_segmentation/scripts/segment.py:device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mri_histo_util/ERC_bayesian_segmentation/scripts/segment.py:    slow = (device.type == 'cpu' or torch.cuda.get_device_properties(device).total_memory < 30*1024**3)
mri_histo_util/ERC_bayesian_segmentation/scripts/segment.py:            torch.cuda.empty_cache()
mri_histo_util/ERC_bayesian_segmentation/scripts/segment.py:                torch.cuda.empty_cache()
mri_histo_util/ERC_bayesian_segmentation/scripts/segment.py:    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mri_histo_util/ERC_bayesian_segmentation/scripts/segment.py:    torch.cuda.empty_cache()
packages/tiff/tif_getimage.c:#define	DECLAREContigPutFunc(name) \
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(put8bitcmaptile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(put4bitcmaptile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(put2bitcmaptile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(put1bitcmaptile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(putgreytile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(put16bitbwtile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(put1bitbwtile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(put2bitbwtile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(put4bitbwtile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(putRGBcontig8bittile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(putRGBcontig8bitMaptile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(putRGBAAcontig8bittile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(putRGBUAcontig8bittile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(putRGBcontig16bittile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(putRGBAAcontig16bittile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(putRGBUAcontig16bittile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(putRGBcontig8bitCMYKtile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(putRGBcontig8bitCMYKMaptile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(putcontig8bitCIELab)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(putcontig8bitYCbCr44tile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(putcontig8bitYCbCr42tile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(putcontig8bitYCbCr41tile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(putcontig8bitYCbCr22tile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(putcontig8bitYCbCr21tile)
packages/tiff/tif_getimage.c:DECLAREContigPutFunc(putcontig8bitYCbCr11tile)
packages/minc/minc_netcdf_convenience.c:      (void) ncclose(status);
packages/minc/minc_netcdf_convenience.c:@DESCRIPTION: A wrapper for routine ncclose, allowing future enhancements.
packages/minc/minc_netcdf_convenience.c:    status = ncclose(cdfid);
packages/minc/minc_netcdf_convenience.c:            (void) ncclose(status);
packages/minc/minc_netcdf_convenience.c:void ncclose(int file) {
packages/netcdf/v2i.c:ncclose(int ncid)
packages/netcdf/v2i.c:		nc_advise("ncclose", status, "ncid %d", ncid);
packages/netcdf/nc.c: *  Common code for ncendef, ncclose(endef)
packages/netcdf/nc.c: * In data mode, same as ncclose.
packages/netcdf/netcdf.h:ncclose(int ncid);
packages/xml2/xmlsave.c:	    xmlCharEncCloseFunc(buf->encoder);
packages/xml2/xmlsave.c:    xmlCharEncCloseFunc(buf->encoder);
packages/xml2/xmlsave.c:        if (handler) xmlCharEncCloseFunc(handler);
packages/xml2/encoding.c: * xmlCharEncCloseFunc:
packages/xml2/encoding.c:xmlCharEncCloseFunc(xmlCharEncodingHandler *handler) {
packages/xml2/xmlIO.c:        xmlCharEncCloseFunc(in->encoder);
packages/xml2/xmlIO.c:        xmlCharEncCloseFunc(out->encoder);
packages/xml2/parserInternals.c:            xmlCharEncCloseFunc(input->buf->encoder);
packages/xml2/catalog.c: * xmlParseSGMLCatalogPubid:
packages/xml2/catalog.c:xmlParseSGMLCatalogPubid(const xmlChar *cur, xmlChar **id) {
packages/xml2/catalog.c:		    cur = xmlParseSGMLCatalogPubid(cur, &sysid);
packages/xml2/catalog.c:		    cur = xmlParseSGMLCatalogPubid(cur, &name);
packages/xml2/catalog.c:		    cur = xmlParseSGMLCatalogPubid(cur, &sysid);
packages/xml2/catalog.c:		    cur = xmlParseSGMLCatalogPubid(cur, &sysid);
packages/xml2/libxml/encoding.h:	xmlCharEncCloseFunc		(xmlCharEncodingHandler *handler);
packages/xml2/elfgcchack.h:#undef xmlCharEncCloseFunc
packages/xml2/elfgcchack.h:extern __typeof (xmlCharEncCloseFunc) xmlCharEncCloseFunc __attribute((alias("xmlCharEncCloseFunc__internal_alias")));
packages/xml2/elfgcchack.h:#ifndef xmlCharEncCloseFunc
packages/xml2/elfgcchack.h:extern __typeof (xmlCharEncCloseFunc) xmlCharEncCloseFunc__internal_alias __attribute((visibility("hidden")));
packages/xml2/elfgcchack.h:#define xmlCharEncCloseFunc xmlCharEncCloseFunc__internal_alias
packages/pybind11/tools/pybind11Common.cmake:      # instance, projects that include other types of source files like CUDA
packages/pybind11/tools/pybind11Tools.cmake:  if(NOT DEFINED CMAKE_CUDA_VISIBILITY_PRESET)
packages/pybind11/tools/pybind11Tools.cmake:    set_target_properties(${target_name} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
packages/pybind11/tools/pybind11NewTools.cmake:  if(NOT DEFINED CMAKE_CUDA_VISIBILITY_PRESET)
packages/pybind11/tools/pybind11NewTools.cmake:    set_target_properties(${target_name} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
packages/pybind11/include/pybind11/detail/common.h:// For CUDA, GCC7, GCC8:
packages/pybind11/include/pybind11/detail/common.h:// 1.7% for CUDA, -0.2% for GCC7, and 0.0% for GCC8 (using -DCMAKE_BUILD_TYPE=MinSizeRel,
packages/pybind11/include/pybind11/detail/common.h:    && (defined(__CUDACC__) || (defined(__GNUC__) && (__GNUC__ == 7 || __GNUC__ == 8)))
packages/pybind11/include/pybind11/cast.h:    // static_cast works around compiler error with MSVC 17 and CUDA 10.2
packages/pybind11/include/pybind11/numpy.h:#ifdef __CUDACC__
psacnn_brain_segmentation/psacnn_brain_segmentation/predict.py:                                                        num_gpus=1,
psacnn_brain_segmentation/psacnn_brain_segmentation/predict.py:#                               contrast='t1w', gpu=0, patch_size=96,model_file=None, output_soft=False):
psacnn_brain_segmentation/psacnn_brain_segmentation/predict.py:    gpu = traits.Int(desc='seed value for non-deterministic functions', argstr='-gpu %d', position=3)
psacnn_brain_segmentation/psacnn_brain_segmentation/predict.py:    'run psacnn_segment.py -i t1.nii.gz -o output_dir/psacnn -gpu 0 -c 't2w' -p 96
psacnn_brain_segmentation/psacnn_brain_segmentation/predict.py:    psacnn = Node(PSACNN(output_dir=output_dir, contrast='t2w', gpu=0, patch_size=96, save_prob_output=False), name='psacnn')
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:from keras.utils import  multi_gpu_model
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                  loss='mean_absolute_error', initial_learning_rate=0.00001, deconvolution=False, use_patches=True, num_gpus=1):
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:        if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:            if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:            if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                  num_gpus=1, num_outputs=1):
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:        if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:            if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:            if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                  num_gpus=1, num_outputs=1):
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:        if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:            if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:            if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                  loss='mean_absolute_error', initial_learning_rate=0.00001, deconvolution=False, use_patches=True, num_gpus=1):
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:        if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:            if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:            if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:#         if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:#                 parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:#             if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:#                     parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:#             if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:#                     parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                         kernel_size=(3,3), pool_size=(2,2), n_labels=0, num_outputs=1, num_gpus=1):
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:def build_compile_model(input_layer, input_shape, conv, n_labels,  loss, num_gpus, initial_learning_rate):
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:        if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:            if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:            if num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                  loss='mean_squared_error', initial_learning_rate=0.00001, deconvolution=False, num_gpus=1, num_outputs=1, add_modality_channel=False):
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:                                                       loss, num_gpus, initial_learning_rate)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:    os.environ['LD_LIBRARY_PATH'] = '/usr/pubsw/packages/CUDA/lib64 '
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/unet_model.py:    #                    use_patches=True, num_gpus=1)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:class MultiGPUCheckpointCallback(Callback):
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:        super(MultiGPUCheckpointCallback, self).__init__()
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                 fcn=True, num_gpus=1, preprocessing=False, augment=False, num_outputs=1,
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:        self.num_gpus = num_gpus
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                                       deconvolution=False, use_patches=use_patches, num_gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                                                            num_gpus=num_gpus)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                                       deconvolution=False, use_patches=use_patches, num_gpus=num_gpus, num_outputs=num_outputs)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                                                         deconvolution=False, num_gpus=num_gpus, num_outputs=num_outputs,
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                  wmp_standardize=True, rob_standardize=True,fcn=True, num_gpus=1,
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                           wmp_standardize=wmp_standardize, rob_standardize=rob_standardize, fcn=fcn, num_gpus=1,
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:            if self.num_gpus == 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                            generator=self.feature_generator.training_generator_t1beta(batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                                batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                                batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                                batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                                batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                                batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                        self.model.fit_generator(generator=self.feature_generator.training_generator(batch_size=batch_size*self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                                         validation_data=self.feature_generator.validation_generator(batch_size=batch_size*self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                callback = MultiGPUCheckpointCallback(output_prefix, save_per_epoch, save_weights, initial_epoch)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                    generator=self.feature_generator.training_generator(batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                    validation_data=self.feature_generator.validation_generator(batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:            if self.num_gpus == 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                    # self.model.fit_generator(generator=self.feature_generator.seg_training_generator_multichannel_nmr_t2(batch_size=batch_size*self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                    #                  validation_data=self.feature_generator.seg_validation_generator_multichannel_nmr(batch_size=batch_size*self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                        generator=self.feature_generator.dynamic_seg_training_generator_multichannel_nmr_t2(batch_size=batch_size * self.num_gpus,
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                            batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                        batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                        batch_size=batch_size * self.num_gpus, focus=focus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                                                 batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                        batch_size=batch_size * self.num_gpus, focus=focus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                                                 batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                        batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                                                 batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                        generator=self.feature_generator.seg_training_generator_multichannel(batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                            batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                callback = MultiGPUCheckpointCallback(output_prefix, save_per_epoch, save_weights, initial_epoch)
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                    generator=self.feature_generator.training_generator(batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                    validation_data=self.feature_generator.validation_generator(batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:            self.model.fit_generator(generator=self.feature_generator.training_label_generator(batch_size=batch_size*self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                                         batch_size=batch_size * self.num_gpus),
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:        # if self.num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:        if self.num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:        if self.num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:        if self.num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:        if self.num_gpus > 1:
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
psacnn_brain_segmentation/psacnn_brain_segmentation/deeplearn_utils/DeepImageSynth.py:                                              num_gpus=1,
psacnn_brain_segmentation/README.md:$>./psacnn_segment.py  -i t1.nii.gz -o /loc/output/dir  -c t1w -gpu 0 -pre
psacnn_brain_segmentation/README.md:-gpu: for GPU to run on. Set to -1 if not running on a GPU.
psacnn_brain_segmentation/preprocess.py:                                                        num_gpus=1,
psacnn_brain_segmentation/preprocess.py:#                               contrast='t1w', gpu=0, patch_size=96,model_file=None, output_soft=False):
psacnn_brain_segmentation/preprocess.py:    # gpu = traits.Int(desc='seed value for non-deterministic functions', argstr='-gpu %d', position=3)
psacnn_brain_segmentation/preprocess.py:    batch_size = traits.Int(4, desc='batch size for GPU processing', argstr='-batch_size %d', position=8, usedefault=True)
psacnn_brain_segmentation/preprocess.py:    'run psacnn_segment.py -i t1.nii.gz -o output_dir/psacnn -gpu 0 -c 't2w' -p 96
psacnn_brain_segmentation/preprocess.py:def psacnn_workflow(input_file, output_dir, use_preprocess=True, model_file=None, contrast='t1w', use_gpu=True,
psacnn_brain_segmentation/preprocess.py:                    gpu_id=0, save_label_image=False, save_prob_image=False, patch_size=96, batch_size=4, sample_rate=20000):
psacnn_brain_segmentation/preprocess.py:    if use_gpu == False:
psacnn_brain_segmentation/preprocess.py:        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
psacnn_brain_segmentation/preprocess.py:        os.environ["CUDA_VISIBLE_DEVICES"] = ""
psacnn_brain_segmentation/preprocess.py:        gpu_id = -1
psacnn_brain_segmentation/preprocess.py:        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
psacnn_brain_segmentation/preprocess.py:        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
psacnn_brain_segmentation/preprocess.py:    psacnn_workflow(input_file, output_dir, use_preprocess=False, model_file=None, contrast='t1w', use_gpu=True,
psacnn_brain_segmentation/preprocess.py:                    gpu_id=0, save_label_image=False, save_prob_image=False, patch_size=96, batch_size=4,
psacnn_brain_segmentation/psacnn_segment.py:            -gpu: GPU id to be used for computation
psacnn_brain_segmentation/psacnn_segment.py:parser.add_argument('-gpu', nargs='?', const=0, default=0, type=int)
psacnn_brain_segmentation/psacnn_segment.py:#print(args.gpu)
psacnn_brain_segmentation/psacnn_segment.py:if args.gpu < 0 :
psacnn_brain_segmentation/psacnn_segment.py:    use_gpu = False
psacnn_brain_segmentation/psacnn_segment.py:    use_gpu = True
psacnn_brain_segmentation/psacnn_segment.py:                use_gpu=use_gpu,
psacnn_brain_segmentation/psacnn_segment.py:                gpu_id=args.gpu,
mri_3d_photo_recon/mri_3d_photo_recon:    parser.add_argument("--gpu", type=int, help="Index of GPU to use", default=None)
mri_3d_photo_recon/mri_3d_photo_recon:    # Set the GPU if needed
mri_3d_photo_recon/mri_3d_photo_recon:    if options.gpu is None:
mri_3d_photo_recon/mri_3d_photo_recon:        os.environ["CUDA_VISIBLE_DEVICES"] = ""
mri_3d_photo_recon/mri_3d_photo_recon:        os.environ["CUDA_VISIBLE_DEVICES"] = str(options.gpu)
mri_3d_photo_recon/mri_3d_photo_recon:        if torch.cuda.is_available():
mri_3d_photo_recon/mri_3d_photo_recon:            print("Using GPU device " + str(options.gpu))
mri_3d_photo_recon/mri_3d_photo_recon:            device = torch.device("cuda:0")
mri_3d_photo_recon/mri_3d_photo_recon:                "Tried to use GPU device "
mri_3d_photo_recon/mri_3d_photo_recon:                + str(options.gpu)
mri_3d_photo_recon/mri_3d_photo_recon:    torch.cuda.manual_seed_all(seed)
mri_3d_photo_recon/mri_3d_photo_recon:    torch.cuda.manual_seed(seed)
mri_3d_photo_recon/mri_3d_photo_recon:                if torch.cuda.is_available():
mri_3d_photo_recon/mri_3d_photo_recon:                    F_prev = torch.tensor(np.nan, dtype=dtype).cuda()
mri_3d_photo_recon/mri_3d_photo_recon:                if torch.cuda.is_available():
mri_3d_photo_recon/mri_3d_photo_recon:                    F_b = torch.tensor(np.nan, dtype=dtype).cuda()
mri_3d_photo_recon/mri_3d_photo_recon:                    g_b = torch.tensor(np.nan, dtype=dtype).cuda()
mri_3d_photo_recon/mri_3d_photo_recon:                        if torch.cuda.is_available():
mri_3d_photo_recon/mri_3d_photo_recon:                            g_b = torch.tensor(np.nan, dtype=dtype).cuda()
mri_synthseg/mri_synthseg:    parser.add_argument("--cpu", action="store_true", help="(optional) Enforce running with CPU rather than GPU.")
mri_synthseg/mri_synthseg:        print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
mri_synthseg/mri_synthseg:        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
mris_estimate_wm/mris_estimate_wm:parser.add_argument('-g', '--gpu', action='store_true', help='Use the GPU.')
mris_estimate_wm/mris_estimate_wm:# configure GPU device
mris_estimate_wm/mris_estimate_wm:if args.gpu:
mris_estimate_wm/mris_estimate_wm:    print('Configuring model on the GPU')
mris_estimate_wm/mris_estimate_wm:    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
mris_estimate_wm/mris_estimate_wm:    device = torch.device('cuda')
mris_estimate_wm/mris_estimate_wm:    device_name = 'GPU'
mris_estimate_wm/mris_estimate_wm:    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
dissection_photo/nnUNet/package/batchgenerators/dataloading/nondet_multi_threaded_augmenter.py:                 gpu: Union[int, None] = None, wait_time: float = 0.02):
dissection_photo/nnUNet/package/batchgenerators/dataloading/nondet_multi_threaded_augmenter.py:    do_pin_memory = torch is not None and pin_memory and gpu is not None and torch.cuda.is_available()
dissection_photo/nnUNet/package/batchgenerators/dataloading/nondet_multi_threaded_augmenter.py:        print('using pin_memory on device', gpu)
dissection_photo/nnUNet/package/batchgenerators/dataloading/nondet_multi_threaded_augmenter.py:        torch.cuda.set_device(gpu)
dissection_photo/nnUNet/package/batchgenerators/dataloading/nondet_multi_threaded_augmenter.py:            if torch is not None and torch.cuda.is_available():
dissection_photo/nnUNet/package/batchgenerators/dataloading/nondet_multi_threaded_augmenter.py:                gpu = torch.cuda.current_device()
dissection_photo/nnUNet/package/batchgenerators/dataloading/nondet_multi_threaded_augmenter.py:                gpu = None
dissection_photo/nnUNet/package/batchgenerators/dataloading/nondet_multi_threaded_augmenter.py:            # gpu: Union[int, None] = None, wait_time: float = 0.02
dissection_photo/nnUNet/package/batchgenerators/dataloading/nondet_multi_threaded_augmenter.py:                self._queue, self.results_loop_queue, self.abort_event, self.pin_memory, self._processes, gpu,
dissection_photo/nnUNet/package/batchgenerators/dataloading/multi_threaded_augmenter.py:                 gpu: Union[int, None], wait_time: float, worker_list: list):
dissection_photo/nnUNet/package/batchgenerators/dataloading/multi_threaded_augmenter.py:    do_pin_memory = torch is not None and pin_memory and gpu is not None and torch.cuda.is_available()
dissection_photo/nnUNet/package/batchgenerators/dataloading/multi_threaded_augmenter.py:        print('using pin_memory on device', gpu)
dissection_photo/nnUNet/package/batchgenerators/dataloading/multi_threaded_augmenter.py:        torch.cuda.set_device(gpu)
dissection_photo/nnUNet/package/batchgenerators/dataloading/multi_threaded_augmenter.py:            if torch is not None and torch.cuda.is_available():
dissection_photo/nnUNet/package/batchgenerators/dataloading/multi_threaded_augmenter.py:                gpu = torch.cuda.current_device()
dissection_photo/nnUNet/package/batchgenerators/dataloading/multi_threaded_augmenter.py:                gpu = None
dissection_photo/nnUNet/package/batchgenerators/dataloading/multi_threaded_augmenter.py:                self._queues, self.pin_memory_queue, self.abort_event, self.pin_memory, gpu, self.wait_time,
dissection_photo/nnUNet/package/documentation/benchmarking.md:be noted down, along with the GPU name, torch version and cudnn version. You can find the benchmark output in the 
dissection_photo/nnUNet/package/documentation/benchmarking.md:epoch time as well as the GPU used and the torch and cudnn versions. Useful for speed testing the entire pipeline 
dissection_photo/nnUNet/package/documentation/benchmarking.md:(data loading, augmentation, GPU training)
dissection_photo/nnUNet/package/documentation/benchmarking.md:just presents dummy arrays to the GPU. Useful for checking pure GPU speed.
dissection_photo/nnUNet/package/documentation/benchmarking.md:Then, for each dataset, run the following commands (only one per GPU! Or one after the other):
dissection_photo/nnUNet/package/documentation/benchmarking.md:Note that there can be multiple entries in this file if the benchmark was run on different GPU types, torch versions or cudnn versions!
dissection_photo/nnUNet/package/documentation/benchmarking.md:We have tested a variety of GPUs and summarized the results in a 
dissection_photo/nnUNet/package/documentation/benchmarking.md:both your numbers are worse, the problem is with your GPU:
dissection_photo/nnUNet/package/documentation/benchmarking.md:- Are you certain you compare the correct GPU? (duh)
dissection_photo/nnUNet/package/documentation/benchmarking.md:[PyTorch installation](https://pytorch.org/get-started/locally/) page, select the most recent cuda version your 
dissection_photo/nnUNet/package/documentation/benchmarking.md:- Finally, some very basic things that could impact your GPU performance: 
dissection_photo/nnUNet/package/documentation/benchmarking.md:  - Is the GPU cooled adequately? Check the temperature with `nvidia-smi`. Hot GPUs throttle performance in order to not self-destruct
dissection_photo/nnUNet/package/documentation/benchmarking.md:  - Is your OS using the GPU for displaying your desktop at the same time? If so then you can expect a performance 
dissection_photo/nnUNet/package/documentation/benchmarking.md:  - Are other users using the GPU as well?
dissection_photo/nnUNet/package/documentation/installation_instructions.md:We support GPU (recommended), CPU and Apple M1/M2 as devices (currently Apple mps does not implement 3D 
dissection_photo/nnUNet/package/documentation/installation_instructions.md:We recommend you use a GPU for training as this will take a really long time on CPU or MPS (Apple M1/M2). 
dissection_photo/nnUNet/package/documentation/installation_instructions.md:For training a GPU with at least 10 GB (popular non-datacenter options are the RTX 2080ti, RTX 3080/3090 or RTX 4080/4090) is 
dissection_photo/nnUNet/package/documentation/installation_instructions.md:required. We also recommend a strong CPU to go along with the GPU. 6 cores (12 threads) 
dissection_photo/nnUNet/package/documentation/installation_instructions.md:input channels and target structures. Plus, the faster the GPU, the better the CPU should be!
dissection_photo/nnUNet/package/documentation/installation_instructions.md:Again we recommend a GPU to make predictions as this will be substantially faster than the other options. However, 
dissection_photo/nnUNet/package/documentation/installation_instructions.md:inference times are typically still manageable on CPU and MPS (Apple M1/M2). If using a GPU, it should have at least 
dissection_photo/nnUNet/package/documentation/installation_instructions.md:- GPU: RTX 3090 or RTX 4090
dissection_photo/nnUNet/package/documentation/installation_instructions.md:- CPU: 2x AMD EPYC7763 for a total of 128C/256T. 16C/GPU are highly recommended for fast GPUs such as the A100!
dissection_photo/nnUNet/package/documentation/installation_instructions.md:- GPU: 8xA100 PCIe (price/performance superior to SXM variant + they use less power)
dissection_photo/nnUNet/package/documentation/installation_instructions.md:(nnU-net by default uses one GPU per training. The server configuration can run up to 8 model trainings simultaneously)
dissection_photo/nnUNet/package/documentation/installation_instructions.md:CPU/GPU ratio. For the server above (256 threads for 8 GPUs), a good value would be 24-30. You can do this by 
dissection_photo/nnUNet/package/documentation/installation_instructions.md:install the latest version with support for your hardware (cuda, mps, cpu).
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:You can specify the device nnU-net should use by using `-device DEVICE`. DEVICE can only be cpu, cuda or mps. If 
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:you have multiple GPUs, please select the gpu id using `CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...]` (requires device to be cuda).
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:Training times largely depend on the GPU. The smallest GPU we recommend for training is the Nvidia RTX 2080ti. With 
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:### Using multiple GPUs for training
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:If multiple GPUs are at your disposal, the best way of using them is to train multiple nnU-Net trainings at once, one 
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:on each GPU. This is because data parallelism never scales perfectly linearly, especially not with small networks such 
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_NAME_OR_ID 2d 0 [--npz] & # train on GPU 0
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:CUDA_VISIBLE_DEVICES=1 nnUNetv2_train DATASET_NAME_OR_ID 2d 1 [--npz] & # train on GPU 1
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:CUDA_VISIBLE_DEVICES=2 nnUNetv2_train DATASET_NAME_OR_ID 2d 2 [--npz] & # train on GPU 2
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:CUDA_VISIBLE_DEVICES=3 nnUNetv2_train DATASET_NAME_OR_ID 2d 3 [--npz] & # train on GPU 3
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:CUDA_VISIBLE_DEVICES=4 nnUNetv2_train DATASET_NAME_OR_ID 2d 4 [--npz] & # train on GPU 4
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:configuration! Wait with starting subsequent folds until the first training is using the GPU! Depending on the 
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:If you insist on running DDP multi-GPU training, we got you covered:
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:`nnUNetv2_train DATASET_NAME_OR_ID 2d 0 [--npz] -num_gpus X`
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:Again, note that this will be slower than running separate training on separate GPUs. DDP only makes sense if you have 
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:Important when using `-num_gpus`:
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:1) If you train using, say, 2 GPUs but have more GPUs in the system you need to specify which GPUs should be used via 
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:CUDA_VISIBLE_DEVICES=0,1 (or whatever your ids are).
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:2) You cannot specify more GPUs than you have samples in your minibatches. If the batch size is 2, 2 GPUs is the maximum!
dissection_photo/nnUNet/package/documentation/how_to_use_nnunet.md:3) Make sure your batch size is divisible by the numbers of GPUs you use or you will not make good use of your hardware.
dissection_photo/nnUNet/package/documentation/competitions/AutoPETII.md:Training each model requires 8 Nvidia A100 40GB GPUs. Expect training to run for 5-7 days. You'll need a really good 
dissection_photo/nnUNet/package/documentation/competitions/AutoPETII.md:nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_bs80 0 -num_gpus 8
dissection_photo/nnUNet/package/documentation/competitions/AutoPETII.md:nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_bs80 1 -num_gpus 8
dissection_photo/nnUNet/package/documentation/competitions/AutoPETII.md:nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_bs80 2 -num_gpus 8
dissection_photo/nnUNet/package/documentation/competitions/AutoPETII.md:nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_bs80 3 -num_gpus 8
dissection_photo/nnUNet/package/documentation/competitions/AutoPETII.md:nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_bs80 4 -num_gpus 8
dissection_photo/nnUNet/package/documentation/competitions/AutoPETII.md:nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_192x192x192_b24 0 -num_gpus 8
dissection_photo/nnUNet/package/documentation/competitions/AutoPETII.md:nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_192x192x192_b24 1 -num_gpus 8
dissection_photo/nnUNet/package/documentation/competitions/AutoPETII.md:nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_192x192x192_b24 2 -num_gpus 8
dissection_photo/nnUNet/package/documentation/competitions/AutoPETII.md:nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_192x192x192_b24 3 -num_gpus 8
dissection_photo/nnUNet/package/documentation/competitions/AutoPETII.md:nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 221 3d_fullres_resenc_192x192x192_b24 4 -num_gpus 8
dissection_photo/nnUNet/package/documentation/competitions/AutoPETII.md:(We also provide pretrained weights in case you don't want to invest the GPU resources, see below)
dissection_photo/nnUNet/package/documentation/competitions/AutoPETII.md:done to keep the inference time below 10 minutes per image on a T4 GPU (we actually never tested whether we could 
dissection_photo/nnUNet/package/documentation/extending_nnunet.md:  used by default. It needs to have some sort of GPU memory estimation method that can be used to evaluate whether 
dissection_photo/nnUNet/package/documentation/extending_nnunet.md:  topologies fit into a specified GPU memory target. Build a new `ExperimentPlanner` that can configure your new 
dissection_photo/nnUNet/package/documentation/extending_nnunet.md:- Remember that multi-GPU training, region-based training, ignore label and cascaded training are now simply integrated 
dissection_photo/nnUNet/package/documentation/resenc_presets.md:RT: training run time (measured on 1x Nvidia A100 PCIe 40GB)\
dissection_photo/nnUNet/package/documentation/resenc_presets.md:VRAM: GPU VRAM used during training, as reported by nvidia-smi\
dissection_photo/nnUNet/package/documentation/resenc_presets.md:We offer three new presets, each targeted for a different GPU VRAM and compute budget:
dissection_photo/nnUNet/package/documentation/resenc_presets.md:- **nnU-Net ResEnc M**: similar GPU budget to the standard UNet configuration. Best suited for GPUs with 9-11GB VRAM. Training time: ~12h on A100
dissection_photo/nnUNet/package/documentation/resenc_presets.md:- **nnU-Net ResEnc L**: requires a GPU with 24GB VRAM. Training time: ~35h on A100
dissection_photo/nnUNet/package/documentation/resenc_presets.md:- **nnU-Net ResEnc XL**: requires a GPU with 40GB VRAM. Training time: ~66h on A100
dissection_photo/nnUNet/package/documentation/resenc_presets.md:- They set new default values for `gpu_memory_target_in_gb` to target the respective VRAM consumptions
dissection_photo/nnUNet/package/documentation/resenc_presets.md:You can easily adapt the GPU memory target to match your GPU, and to scale beyond 40GB of GPU memory. 
dissection_photo/nnUNet/package/documentation/resenc_presets.md:`nnUNetv2_plan_experiment -d 3 -pl nnUNetPlannerResEncM -gpu_memory_target 80 -overwrite_plans_name nnUNetResEncUNetPlans_80G`
dissection_photo/nnUNet/package/documentation/resenc_presets.md:warning ("You are running nnUNetPlannerM with a non-standard gpu_memory_target_in_gb"). This warning can be ignored here.
dissection_photo/nnUNet/package/documentation/resenc_presets.md:### Scaling to multiple GPUs
dissection_photo/nnUNet/package/documentation/resenc_presets.md:When scaling to multiple GPUs, do not just specify the combined amount of VRAM to `nnUNetv2_plan_experiment` as this 
dissection_photo/nnUNet/package/documentation/resenc_presets.md:may result in patch sizes that are too large to be processed by individual GPUs. It is best to let this command run for 
dissection_photo/nnUNet/package/documentation/resenc_presets.md:the VRAM budget of one GPU, and then manually edit the plans file to increase the batch size. You can use [configuration inheritance](explanation_plans_files.md).
dissection_photo/nnUNet/package/documentation/resenc_presets.md:Where XX is the new batch size. If 3d_fullres has a batch size of 2 for one GPU and you are planning to scale to 8 GPUs, make the new batch size 2x8=16!
dissection_photo/nnUNet/package/documentation/resenc_presets.md:You can then train the new configuration using nnU-Net's multi-GPU settings:
dissection_photo/nnUNet/package/documentation/resenc_presets.md:nnUNetv2_train DATASETID 3d_fullres_bsXX FOLD -p nnUNetResEncUNetPlans_80G -num_gpus 8
dissection_photo/nnUNet/package/documentation/resenc_presets.md:variants. For a fair comparison, pick the variant that most closely matches the GPU memory and compute 
dissection_photo/nnUNet/package/documentation/changelog.md:- Cross-platform support. Cuda, mps (Apple M1/M2) and of course CPU support! Simply select the device with 
dissection_photo/nnUNet/package/documentation/changelog.md:- Native support for multi-GPU (DDP) TRAINING. 
dissection_photo/nnUNet/package/documentation/changelog.md:Multi-GPU INFERENCE should still be run with `CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] -num_parts Y -part_id X`. 
dissection_photo/nnUNet/package/documentation/changelog.md:There is no cross-GPU communication in inference, so it doesn't make sense to add additional complexity with DDP.
dissection_photo/nnUNet/package/documentation/changelog.md:- Integration into MONAI (together with our friends at Nvidia)
dissection_photo/nnUNet/package/acvl_utils/array_manipulation/resampling.py:def maybe_resample_on_gpu(data: torch.Tensor,
dissection_photo/nnUNet/package/acvl_utils/array_manipulation/resampling.py:                          compute_device: str = 'cuda:0', result_device: str = 'cuda:0',
dissection_photo/nnUNet/package/acvl_utils/array_manipulation/resampling.py:        if torch.cuda.is_available(): torch.cuda.empty_cache()
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:def gpu_binary_dilation(binary_array: Union[np.ndarray, torch.Tensor], selem: np.ndarray) \
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:    IMPORTANT: ALWAYS benchmark your image and kernel sizes first. Sometimes GPU is actually slower than CPU!
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:        # move source array to GPU first. Uses non-blocking (important!) so that copy operation can run in background.
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:        # Cast to half only on the GPU because that is much faster and because the source array is quicker to
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:        # again convert to 1 byte per element byte on GPU, then copy
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:    torch.cuda.empty_cache()
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:def gpu_binary_erosion(binary_array: Union[np.ndarray, torch.Tensor], selem: np.ndarray) \
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:    IMPORTANT: ALWAYS benchmark your image and kernel sizes first. Sometimes GPU is actually slower than CPU!
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:        # move source array to GPU first. Uses non-blocking (important!) so that copy operation can run in background.
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:        # Cast to half only on the GPU because that is much faster and because the source array is quicker to
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:        # again convert to 1 byte per element byte on GPU, then copy
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:    torch.cuda.empty_cache()
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:def gpu_binary_opening(binary_array: Union[np.ndarray, torch.Tensor], selem: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:    return gpu_binary_dilation(gpu_binary_erosion(binary_array, selem), selem)
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:def gpu_binary_closing(binary_array: Union[np.ndarray, torch.Tensor], selem: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:    return gpu_binary_erosion(gpu_binary_dilation(binary_array, selem), selem)
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:    _ = gpu_binary_dilation(inp, selem)
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:    output_gpu = gpu_binary_dilation(inp, selem)
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:    time_gpu = time() - start
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:    print(f'Dilation: GPU: {time_gpu}s')
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:    assert np.all(output_gpu == ref)
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:    output_gpu = gpu_binary_erosion(inp, selem)
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:    time_gpu = time() - start
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:    print(f'Erosion: GPU: {time_gpu}s')
dissection_photo/nnUNet/package/acvl_utils/morphology/gpu_binary_morphology.py:    assert np.all(output_gpu == ref)
dissection_photo/nnUNet/package/nnunetv2/dataset_conversion/Dataset027_ACDC.py:    # labelsTr_folder = '/home/isensee/drives/gpu_data_root/OE0441/isensee/nnUNet_raw/nnUNet_raw_remake/Dataset027_ACDC/labelsTr'
dissection_photo/nnUNet/package/nnunetv2/inference/JHU_inference.py:                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
dissection_photo/nnUNet/package/nnunetv2/inference/JHU_inference.py:        device=torch.device('cuda', 0),
dissection_photo/nnUNet/package/nnunetv2/inference/sliding_window_prediction.py:                     value_scaling_factor: float = 1, dtype=torch.float16, device=torch.device('cuda', 0)) \
dissection_photo/nnUNet/package/nnunetv2/inference/examples.py:        device=torch.device('cuda', 0),
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:                 device: torch.device = torch.device('cuda'),
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:        if device.type == 'cuda':
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:                                                self.configuration_manager, num_processes, self.device.type == 'cuda',
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:            self.device.type == 'cuda',
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:        # So autocast will only be active if we have a cuda device.
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:    parser.add_argument('-device', type=str, default='cuda', required=False,
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:    assert args.device in ['cpu', 'cuda',
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:    elif args.device == 'cuda':
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:        # multithreading in torch doesn't help nnU-Net if run on GPU
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:        device = torch.device('cuda')
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:    parser.add_argument('-device', type=str, default='cuda', required=False,
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:    assert args.device in ['cpu', 'cuda',
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:    elif args.device == 'cuda':
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:        # multithreading in torch doesn't help nnU-Net if run on GPU
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:        device = torch.device('cuda')
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:        device=torch.device('cuda', 0),
dissection_photo/nnUNet/package/nnunetv2/inference/predict_from_raw_data.py:    #     device=torch.device('cuda', 0),
dissection_photo/nnUNet/package/nnunetv2/inference/readme.md:focus on communicating with the compute device (i.e. your GPU) and does not have to do any other processing. 
dissection_photo/nnUNet/package/nnunetv2/inference/readme.md:        device=torch.device('cuda', 0),
dissection_photo/nnUNet/package/nnunetv2/tests/integration_tests/run_integration_test_trainingOnly_DDP.sh:nnUNetv2_train $1 3d_fullres 0 -tr nnUNetTrainer_10epochs -num_gpus 2
dissection_photo/nnUNet/package/nnunetv2/tests/integration_tests/lsf_commands.sh:bsub -q gpu.legacy -gpu num=1:j_exclusive=yes:gmem=1G -L /bin/bash ". /home/isensee/load_env_cluster4.sh && cd /home/isensee/git_repos/nnunet_remake && export nnUNet_keep_files_open=True && . nnunetv2/tests/integration_tests/run_integration_test.sh 996"
dissection_photo/nnUNet/package/nnunetv2/tests/integration_tests/lsf_commands.sh:bsub -q gpu.legacy -gpu num=1:j_exclusive=yes:gmem=1G -L /bin/bash ". /home/isensee/load_env_cluster4.sh && cd /home/isensee/git_repos/nnunet_remake && export nnUNet_keep_files_open=True && . nnunetv2/tests/integration_tests/run_integration_test.sh 997"
dissection_photo/nnUNet/package/nnunetv2/tests/integration_tests/lsf_commands.sh:bsub -q gpu.legacy -gpu num=1:j_exclusive=yes:gmem=1G -L /bin/bash ". /home/isensee/load_env_cluster4.sh && cd /home/isensee/git_repos/nnunet_remake && export nnUNet_keep_files_open=True && . nnunetv2/tests/integration_tests/run_integration_test.sh 998"
dissection_photo/nnUNet/package/nnunetv2/tests/integration_tests/lsf_commands.sh:bsub -q gpu.legacy -gpu num=1:j_exclusive=yes:gmem=1G -L /bin/bash ". /home/isensee/load_env_cluster4.sh && cd /home/isensee/git_repos/nnunet_remake && export nnUNet_keep_files_open=True && . nnunetv2/tests/integration_tests/run_integration_test.sh 999"
dissection_photo/nnUNet/package/nnunetv2/tests/integration_tests/lsf_commands.sh:bsub -q gpu.legacy -gpu num=2:j_exclusive=yes:gmem=1G -L /bin/bash ". /home/isensee/load_env_cluster4.sh && cd /home/isensee/git_repos/nnunet_remake && export nnUNet_keep_files_open=True && . nnunetv2/tests/integration_tests/run_integration_test_trainingOnly_DDP.sh 996"
dissection_photo/nnUNet/package/nnunetv2/tests/integration_tests/lsf_commands.sh:bsub -q gpu.legacy -gpu num=2:j_exclusive=yes:gmem=1G -L /bin/bash ". /home/isensee/load_env_cluster4.sh && cd /home/isensee/git_repos/nnunet_remake && export nnUNet_keep_files_open=True && . nnunetv2/tests/integration_tests/run_integration_test_trainingOnly_DDP.sh 997"
dissection_photo/nnUNet/package/nnunetv2/tests/integration_tests/lsf_commands.sh:bsub -q gpu.legacy -gpu num=2:j_exclusive=yes:gmem=1G -L /bin/bash ". /home/isensee/load_env_cluster4.sh && cd /home/isensee/git_repos/nnunet_remake && export nnUNet_keep_files_open=True && . nnunetv2/tests/integration_tests/run_integration_test_trainingOnly_DDP.sh 998"
dissection_photo/nnUNet/package/nnunetv2/tests/integration_tests/lsf_commands.sh:bsub -q gpu.legacy -gpu num=2:j_exclusive=yes:gmem=1G -L /bin/bash ". /home/isensee/load_env_cluster4.sh && cd /home/isensee/git_repos/nnunet_remake && export nnUNet_keep_files_open=True && . nnunetv2/tests/integration_tests/run_integration_test_trainingOnly_DDP.sh 999"
dissection_photo/nnUNet/package/nnunetv2/tests/integration_tests/readme.md:use DATSET_ID 996, 997, 998 and 999. You can run these independently on different GPUs/systems to speed things up. 
dissection_photo/nnUNet/package/nnunetv2/tests/integration_tests/readme.md:to verify DDP is working (needs 2 GPUs!)
dissection_photo/nnUNet/package/nnunetv2/batch_running/generate_lsf_runs_customDecathlon.py:    num_gpus = 1
dissection_photo/nnUNet/package/nnunetv2/batch_running/generate_lsf_runs_customDecathlon.py:    exclude_hosts = "-R \"select[hname!='e230-dgx2-2']\" -R \"select[hname!='e230-dgx2-1']\" -R \"select[hname!='lsf22-gpu02']\" -R \"select[hname!='lsf22-gpu06']\" -R \"select[hname!='e230-dgx1-1']\""
dissection_photo/nnUNet/package/nnunetv2/batch_running/generate_lsf_runs_customDecathlon.py:    gpu_requirements = f"-gpu num={num_gpus}:j_exclusive=yes:gmem=33G"#gmodel=NVIDIAA100_PCIE_40GB"
dissection_photo/nnUNet/package/nnunetv2/batch_running/generate_lsf_runs_customDecathlon.py:    queue = "-q gpu-lowprio"
dissection_photo/nnUNet/package/nnunetv2/batch_running/generate_lsf_runs_customDecathlon.py:    additional_arguments = f' -num_gpus {num_gpus} --disable_checkpointing'  # ''
dissection_photo/nnUNet/package/nnunetv2/batch_running/generate_lsf_runs_customDecathlon.py:                            command = f'bsub {exclude_hosts} {resources} {queue} {gpu_requirements} {preamble} {train_command} {dataset} {config} {fl} -tr {tr} -p {p}'
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/generate_benchmarking_commands.py:    gpu_models = [#'NVIDIAA100_PCIE_40GB', 'NVIDIAGeForceRTX2080Ti', 'NVIDIATITANRTX', 'TeslaV100_SXM2_32GB',
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/generate_benchmarking_commands.py:                  'NVIDIAA100_SXM4_40GB']#, 'TeslaV100_PCIE_32GB']
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/generate_benchmarking_commands.py:    num_gpus = 1
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/generate_benchmarking_commands.py:    queue = "-q gpu"
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/generate_benchmarking_commands.py:    train_command = 'nnUNet_compile=False nnUNet_results=/dkfz/cluster/gpu/checkpoints/OE0441/isensee/nnUNet_results_remake_benchmark nnUNetv2_train'
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/generate_benchmarking_commands.py:    additional_arguments = f' -num_gpus {num_gpus}'  # ''
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/generate_benchmarking_commands.py:        for g in gpu_models:
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/generate_benchmarking_commands.py:            gpu_requirements = f"-gpu num={num_gpus}:j_exclusive=yes:gmodel={g}"
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/generate_benchmarking_commands.py:                                command = f'bsub {exclude_hosts} {resources} {queue} {gpu_requirements} {preamble} {train_command} {dataset} {config} {fl} -tr {tr} -p {p}'
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py:    num_gpus = 1
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py:    unique_gpus = set()
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py:                                   if i['num_gpus'] == num_gpus and i['cudnn_version'] == cudnn_version and
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py:                            all_results[tr][p][c][d][r['gpu_name']] = r
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py:                            unique_gpus.add(r['gpu_name'])
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py:    # haha. Fuck this. Collect GPUs in the code above.
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py:    # unique_gpus = np.unique([i["gpu_name"] for tr in trainers for p in plans for c in configs for d in datasets for i in all_results[tr][p][c][d]])
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py:    unique_gpus = list(unique_gpus)
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py:    unique_gpus.sort()
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py:        for g in unique_gpus:
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py:                        gpu_results = []
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py:                        for g in unique_gpus:
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py:                                gpu_results.append(round(all_results[tr][p][c][d][g]["fastest_epoch"], ndigits=2))
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py:                                gpu_results.append("MISSING")
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py:                        if all([i == 'MISSING' for i in gpu_results]):
dissection_photo/nnUNet/package/nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py:                        for g in gpu_results:
dissection_photo/nnUNet/package/nnunetv2/batch_running/release_trainings/nnunetv2_v1/generate_lsf_commands.py:    num_gpus = 1
dissection_photo/nnUNet/package/nnunetv2/batch_running/release_trainings/nnunetv2_v1/generate_lsf_commands.py:    gpu_requirements = f"-gpu num={num_gpus}:j_exclusive=yes:gmem=1G"
dissection_photo/nnUNet/package/nnunetv2/batch_running/release_trainings/nnunetv2_v1/generate_lsf_commands.py:    queue = "-q gpu-lowprio"
dissection_photo/nnUNet/package/nnunetv2/batch_running/release_trainings/nnunetv2_v1/generate_lsf_commands.py:    train_command = 'nnUNet_keep_files_open=True nnUNet_results=/dkfz/cluster/gpu/data/OE0441/isensee/nnUNet_results_remake_release_normfix nnUNetv2_train'
dissection_photo/nnUNet/package/nnunetv2/batch_running/release_trainings/nnunetv2_v1/generate_lsf_commands.py:    additional_arguments = f'--disable_checkpointing -num_gpus {num_gpus}'  # ''
dissection_photo/nnUNet/package/nnunetv2/batch_running/release_trainings/nnunetv2_v1/generate_lsf_commands.py:                            command = f'bsub {exclude_hosts} {resources} {queue} {gpu_requirements} {preamble} {train_command} {dataset} {config} {fl} -tr {tr} -p {p}'
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/benchmarking/nnUNetTrainerBenchmark_5epochs.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/benchmarking/nnUNetTrainerBenchmark_5epochs.py:        assert torch.cuda.is_available(), "This only works on GPU"
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/benchmarking/nnUNetTrainerBenchmark_5epochs.py:            gpu_name = torch.cuda.get_device_name()
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/benchmarking/nnUNetTrainerBenchmark_5epochs.py:                num_gpus = dist.get_world_size()
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/benchmarking/nnUNetTrainerBenchmark_5epochs.py:                num_gpus = 1
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/benchmarking/nnUNetTrainerBenchmark_5epochs.py:            my_key = f"{cudnn_version}__{torch_version.replace(' ', '')}__{gpu_name.replace(' ', '')}__gpus_{num_gpus}"
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/benchmarking/nnUNetTrainerBenchmark_5epochs.py:                'gpu_name': gpu_name,
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/benchmarking/nnUNetTrainerBenchmark_5epochs.py:                'num_gpus': num_gpus,
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/benchmarking/nnUNetTrainerBenchmark_5epochs_noDataLoading.py:        device: torch.device = torch.device("cuda"),
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/data_augmentation/nnUNetTrainerDAOrd0.py:                                                        pin_memory=self.device.type == 'cuda', wait_time=0.02)
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/data_augmentation/nnUNetTrainerDAOrd0.py:                                                      num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/data_augmentation/nnUNetTrainerDA5.py:                                                        pin_memory=self.device.type == 'cuda', wait_time=0.02)
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/data_augmentation/nnUNetTrainerDA5.py:                                                      num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/data_augmentation/nnUNetTrainerDA5.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/optimizer/nnUNetTrainerAdan.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/optimizer/nnUNetTrainerAdan.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/optimizer/nnUNetTrainerAdan.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/optimizer/nnUNetTrainerAdan.py:    #              device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/optimizer/nnUNetTrainerAdam.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/optimizer/nnUNetTrainerAdam.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/optimizer/nnUNetTrainerAdam.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/optimizer/nnUNetTrainerAdam.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerCELoss.py:        device: torch.device = torch.device("cuda"),
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerNoDeepSupervision.py:        device: torch.device = torch.device("cuda"),
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/sampling/nnUNetTrainer_probabilisticOversampling.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/sampling/nnUNetTrainer_probabilisticOversampling.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/sampling/nnUNetTrainer_probabilisticOversampling.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs_NoMirroring.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs_NoMirroring.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs_NoMirroring.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs_NoMirroring.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:from torch.cuda import device_count
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:from torch.cuda.amp import GradScaler
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:        if self.is_ddp:  # implicitly it's clear that we use cuda in this case
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:            self.device = torch.device(type='cuda', index=self.local_rank)
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:            if self.device.type == 'cuda':
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:                # we might want to let the user pick this but for now please pick the correct GPU with CUDA_VISIBLE_DEVICES=X
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:                self.device = torch.device(type='cuda', index=0)
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:            if self.device.type == 'cuda':
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:                gpu_name = torch.cuda.get_device_name()
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:                dct['gpu_name'] = gpu_name
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:                                                    'GPUs... Duh.'
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:            batch_size_per_GPU = [global_batch_size // world_size] * world_size
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:            batch_size_per_GPU = [batch_size_per_GPU[i] + 1
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:                                  if (batch_size_per_GPU[i] * world_size + i) < global_batch_size
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:                                  else batch_size_per_GPU[i]
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:                                  for i in range(len(batch_size_per_GPU))]
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:            assert sum(batch_size_per_GPU) == global_batch_size
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:            sample_id_low = 0 if my_rank == 0 else np.sum(batch_size_per_GPU[:my_rank])
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:            sample_id_high = np.sum(batch_size_per_GPU[:my_rank + 1])
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:                oversample_percent = sum(oversample[sample_id_low:sample_id_high]) / batch_size_per_GPU[my_rank]
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:            print("worker", my_rank, "batch_size", batch_size_per_GPU[my_rank])
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:            self.batch_size = batch_size_per_GPU[my_rank]
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:        # care about distributing training cases across GPUs.
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:                                                      pin_memory=self.device.type == 'cuda',
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:        # So autocast will only be active if we have a cuda device.
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:        # So autocast will only be active if we have a cuda device.
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:                # we cannot just have barriers all over the place because the number of keys each GPU receives can be
dissection_photo/nnUNet/package/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:                # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
dissection_photo/nnUNet/package/nnunetv2/utilities/helpers.py:    if device.type == 'cuda':
dissection_photo/nnUNet/package/nnunetv2/utilities/helpers.py:        torch.cuda.empty_cache()
dissection_photo/nnUNet/package/nnunetv2/utilities/default_n_proc_DA.py:    Interpret the output as the number of processes used for data augmentation PER GPU.
dissection_photo/nnUNet/package/nnunetv2/utilities/default_n_proc_DA.py:    The way it is implemented here is simply a look up table. We know the hostnames, CPU and GPU configurations of our
dissection_photo/nnUNet/package/nnunetv2/utilities/default_n_proc_DA.py:    systems and set the numbers accordingly. For example, a system with 4 GPUs and 48 threads can use 12 threads per
dissection_photo/nnUNet/package/nnunetv2/utilities/default_n_proc_DA.py:    GPU without overloading the CPU (technically 11 because we have a main process as well), so that's what we use.
dissection_photo/nnUNet/package/nnunetv2/utilities/default_n_proc_DA.py:        elif hostname in ['hdf19-gpu16', 'hdf19-gpu17', 'hdf19-gpu18', 'hdf19-gpu19', 'e230-AMDworkstation']:
dissection_photo/nnUNet/package/nnunetv2/utilities/default_n_proc_DA.py:        elif hostname.startswith('hdf18-gpu') or hostname.startswith('e132-comp'):
dissection_photo/nnUNet/package/nnunetv2/utilities/default_n_proc_DA.py:        elif hostname.startswith('lsf22-gpu'):
dissection_photo/nnUNet/package/nnunetv2/utilities/default_n_proc_DA.py:        elif hostname.startswith('hdf19-gpu') or hostname.startswith('e071-gpu'):
dissection_photo/nnUNet/package/nnunetv2/preprocessing/preprocessors/default_preprocessor.py:    plans_file = '/home/isensee/drives/gpu_data/nnUNet_preprocessed/Dataset219_AMOS2022_postChallenge_task2/nnUNetPlans.json'
dissection_photo/nnUNet/package/nnunetv2/preprocessing/preprocessors/default_preprocessor.py:    dataset_json_file = '/home/isensee/drives/gpu_data/nnUNet_preprocessed/Dataset219_AMOS2022_postChallenge_task2/dataset.json'
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py:    parser.add_argument('-gpu_memory_target', default=None, type=float, required=False,
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py:                        help='[OPTIONAL] DANGER ZONE! Sets a custom GPU memory target (in GB). Default: None (=Planner '
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py:                        help='[OPTIONAL] DANGER ZONE! If you used -gpu_memory_target, -preprocessor_name or '
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py:    plan_experiments(args.d, args.pl, args.gpu_memory_target, args.preprocessor_name, args.overwrite_target_spacing,
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py:    parser.add_argument('-gpu_memory_target', default=None, type=float, required=False,
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py:                        help='[OPTIONAL] DANGER ZONE! Sets a custom GPU memory target (in GB). Default: None (=Planner '
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py:                        help='[OPTIONAL] uSE A CUSTOM PLANS IDENTIFIER. If you used -gpu_memory_target, '
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py:    plans_identifier = plan_experiments(args.d, args.pl, args.gpu_memory_target, args.preprocessor_name,
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/resencUNet_planner.py:                 gpu_memory_target_in_gb: float = 8,
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/resencUNet_planner.py:        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/resencUNet_planner.py:        # this may need to be adapted when using absurdly large GPU memory targets. Increasing this now would not be
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py:                 gpu_memory_target_in_gb: float = 8,
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py:        self.UNet_vram_target_GB = gpu_memory_target_in_gb
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py:        # this may need to be adapted when using absurdly large GPU memory targets. Increasing this now would not be
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py:            spacing_increase_factor = 1.03  # used to be 1.01 but that is slow with new GPU memory estimation!
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:                 gpu_memory_target_in_gb: float = 8,
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:        # this may need to be adapted when using absurdly large GPU memory targets. Increasing this now would not be
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:                 gpu_memory_target_in_gb: float = 8,
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:        if gpu_memory_target_in_gb != 8:
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:            warnings.warn("WARNING: You are running nnUNetPlannerM with a non-standard gpu_memory_target_in_gb. "
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:                          f"Expected 8, got {gpu_memory_target_in_gb}."
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:        self.UNet_vram_target_GB = gpu_memory_target_in_gb
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:        # this is supposed to give the same GPU memory requirement as the default nnU-Net
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:                 gpu_memory_target_in_gb: float = 24,
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:        if gpu_memory_target_in_gb != 24:
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:            warnings.warn("WARNING: You are running nnUNetPlannerL with a non-standard gpu_memory_target_in_gb. "
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:                          f"Expected 24, got {gpu_memory_target_in_gb}."
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:        self.UNet_vram_target_GB = gpu_memory_target_in_gb
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:                 gpu_memory_target_in_gb: float = 40,
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:        if gpu_memory_target_in_gb != 40:
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:            warnings.warn("WARNING: You are running nnUNetPlannerXL with a non-standard gpu_memory_target_in_gb. "
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:                          f"Expected 40, got {gpu_memory_target_in_gb}."
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/experiment_planners/residual_unets/residual_encoder_unet_planners.py:        self.UNet_vram_target_GB = gpu_memory_target_in_gb
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/plan_and_preprocess_api.py:                            gpu_memory_target_in_gb: float = None, preprocess_class_name: str = 'DefaultPreprocessor',
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/plan_and_preprocess_api.py:    if gpu_memory_target_in_gb is not None:
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/plan_and_preprocess_api.py:        kwargs['gpu_memory_target_in_gb'] = gpu_memory_target_in_gb
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/plan_and_preprocess_api.py:                     gpu_memory_target_in_gb: float = None, preprocess_class_name: str = 'DefaultPreprocessor',
dissection_photo/nnUNet/package/nnunetv2/experiment_planning/plan_and_preprocess_api.py:        _, plans_identifier = plan_experiment_dataset(d, experiment_planner, gpu_memory_target_in_gb,
dissection_photo/nnUNet/package/nnunetv2/run/load_pretrained_weights.py:        saved_model = torch.load(fname, map_location=torch.device('cuda', dist.get_rank()))
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:import torch.cuda
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:                          device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:    dist.init_process_group("nccl", rank=rank, world_size=world_size)
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:    torch.cuda.set_device(torch.device('cuda', dist.get_rank()))
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:    if torch.cuda.is_available():
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:                 num_gpus: int = 1,
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:                 device: torch.device = torch.device('cuda')):
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:    if num_gpus > 1:
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:        assert device.type == 'cuda', f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices. Your device: {device}"
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:                     num_gpus),
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:                 nprocs=num_gpus,
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:        if torch.cuda.is_available():
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:    parser.add_argument('-num_gpus', type=int, default=1, required=False,
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:                        help='Specify the number of GPUs to use for training')
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:    parser.add_argument('-device', type=str, default='cuda', required=False,
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:                    help="Use this to set the device the training should run with. Available options are 'cuda' "
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:                         "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:                         "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!")
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:    assert args.device in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:    elif args.device == 'cuda':
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:        # multithreading in torch doesn't help nnU-Net if run on GPU
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:        device = torch.device('cuda')
dissection_photo/nnUNet/package/nnunetv2/run/run_training.py:                 args.num_gpus, args.use_compressed, args.npz, args.c, args.val, args.disable_checkpointing, args.val_best,
dissection_photo/nnUNet/package/readme.md:They ship for a variety of GPU memory targets. It's all awesome stuff, promised! 
dissection_photo/nnUNet/package/readme.md:are adapted to the patch size; the patch size, network topology and batch size are optimized jointly given some GPU 
dissection_photo/nnUNet/process_directory_no_upsample.py:    device = 'cuda' if torch.cuda.is_available() else 'cpu'
dissection_photo/nnUNet/process_directory_no_upsample.py:    perform_everything_on_device = True if device=='cuda' else False
dissection_photo/nnUNet/process_directory.py:    device = 'cuda' if torch.cuda.is_available() else 'cpu'
dissection_photo/nnUNet/process_directory.py:    perform_everything_on_device = True if device=='cuda' else False
dissection_photo/py_scripts/func_mask_extraction.py:parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
diabetes/insulinExhaustiveSearch.m:gpuIndex = 1; % gpuIndex = 1 (or index of the GPU) will use the GPU. gpuIndex = 0 will use the CPU.
diabetes/insulinExhaustiveSearch.m:if gpuIndex
diabetes/insulinExhaustiveSearch.m:    gpu = gpuDevice(gpuIndex);
diabetes/insulinExhaustiveSearch.m:    disp(['Using the ' gpu.Name ' GPU...'])
diabetes/insulinExhaustiveSearch.m:            param(v).(paramFieldNames{n}) = gpuArray(param(v).(paramFieldNames{n}));
diabetes/insulinExhaustiveSearch.m:    if gpuIndex
diabetes/insulinExhaustiveSearch.m:        varRange{n} = gpuArray(varRange{n});
diabetes/insulinExhaustiveSearch.m:if gpuIndex
recon_any/recon-any/unet3d/trainer.py:    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
recon_any/recon-any/unet3d/trainer.py:        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')
recon_any/recon-any/unet3d/trainer.py:        model = model.cuda()
recon_any/recon-any/unet3d/trainer.py:    if torch.cuda.is_available() and not config['device'] == 'cpu':
recon_any/recon-any/unet3d/trainer.py:        model = model.cuda()
recon_any/recon-any/unet3d/trainer.py:        def _move_to_gpu(input):
recon_any/recon-any/unet3d/trainer.py:                return tuple([_move_to_gpu(x) for x in input])
recon_any/recon-any/unet3d/trainer.py:                if torch.cuda.is_available():
recon_any/recon-any/unet3d/trainer.py:                    input = input.cuda(non_blocking=True)
recon_any/recon-any/unet3d/trainer.py:        t = _move_to_gpu(t)
recon_any/recon-any/unet3d/config.py:    if torch.cuda.is_available():
recon_any/recon-any/unet3d/config.py:        config['device'] = 'cuda'
recon_any/recon-any/unet3d/config.py:        logger.warning('CUDA not available, using CPU')
recon_any/recon-any/unet3d/predictor.py:                # send batch to gpu
recon_any/recon-any/unet3d/predictor.py:                if torch.cuda.is_available():
recon_any/recon-any/unet3d/predictor.py:                    input = input.cuda(non_blocking=True)
recon_any/recon-any/unet3d/predictor.py:                # send batch to gpu
recon_any/recon-any/unet3d/predictor.py:                if torch.cuda.is_available():
recon_any/recon-any/unet3d/predictor.py:                    img = img.cuda(non_blocking=True)
recon_any/recon-any/unet3d/losses.py:            class_weights = torch.ones(input.size()[1]).float().cuda()
recon_any/recon-any/unet3d/losses.py:    if torch.cuda.is_available():
recon_any/recon-any/unet3d/losses.py:        loss = loss.cuda()
recon_any/recon-any/interpol/autograd.py:    from torch.cuda.amp import custom_fwd, custom_bwd
recon_any/recon-any/interpol/tests/test_gradcheck_pushpull.py:if torch.cuda.is_available():
recon_any/recon-any/interpol/tests/test_gradcheck_pushpull.py:    print('cuda backend available')
recon_any/recon-any/interpol/tests/test_gradcheck_pushpull.py:    devices.append('cuda')
recon_any/recon-any/interpol/tests/test_gradcheck_pushpull.py:    if device == 'cuda':
recon_any/recon-any/interpol/tests/test_gradcheck_pushpull.py:        torch.cuda.set_device(param)
recon_any/recon-any/interpol/tests/test_gradcheck_pushpull.py:        torch.cuda.init()
recon_any/recon-any/interpol/tests/test_gradcheck_pushpull.py:            torch.cuda.empty_cache()
recon_any/recon-any/interpol/utils.py:    current_version, *cuda_variant = torch.__version__.split('+')
recon_any/recon-any/inference.py:    parser.add_argument("--cpu", action="store_true", help="enforce running with CPU rather than GPU.")
recon_any/recon-any/inference.py:        print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
recon_any/recon-any/inference.py:        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
recon_any/recon-any/inference.py:        device = 'cuda'
recon_any/recon-any/inference.py:        # Detach from GPU
recon_any/recon-any/inference.py:        torch.cuda.empty_cache()
recon_any/recon-any/utils.py:        torch.cuda.empty_cache()
exvivo/mri_exvivo_strip:parser.add_argument('--gpu', help='GPU number - if not supplied, CPU is used')
exvivo/mri_exvivo_strip:if args.gpu:
exvivo/mri_exvivo_strip:    device, ngpus = ne.tf.utils.setup_device(args.gpu)
exvivo/train_unorm.py:gpu_id = 0
exvivo/train_unorm.py:os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
exvivo/train_unorm.py:#fsd.configure(gpu=gpu_id)
exvivo/train_unorm.py:ne.utils.setup_device(gpuid=gpu_id)
exvivo/train_unorm.py:#config.gpu_options.allow_growth = True
exvivo/mri_exvivo_norm:parser.add_argument('--gpu', help='GPU number - if not supplied, CPU is used')
exvivo/mri_exvivo_norm:if args.gpu:
exvivo/mri_exvivo_norm:    device, ngpus = ne.tf.utils.setup_device(args.gpu)
infant/infant_recon_all:# don't use GPU by default
infant/infant_recon_all:set gpuid = -1
infant/infant_recon_all:set usegpu = 0
infant/infant_recon_all:    case "--usegpu":
infant/infant_recon_all:      set usegpu = 1;
infant/infant_recon_all:    case "--gpuid":
infant/infant_recon_all:      set gpuid = $argv[1]; shift;
infant/infant_recon_all:      set usegpu = 1;
infant/infant_recon_all:#GPU ID
infant/infant_recon_all:if !($usegpu) then 
infant/infant_recon_all:  set gpuid = -1
infant/infant_recon_all:    # -use-gpu : use GPU versions of the skullstripper: sscnn_skullstrip.py
infant/infant_recon_all:    # $gpuID = 0 (otherwise user can specify it)
infant/infant_recon_all:    set cmd = (sscnn_skullstrip -i ${infile} -o $WORK_DIR/sscnn -c t1w --gpu $gpuid) # this will produce the mask file $WORK_DIR/sscnn/sscnn_skullstrip.nii.gz
infant/infant_recon_py:parser.add_argument('--usegpu', action='store_true', help='Use the powers of GPU for the skullstripping computations.')
infant/infant_recon_py:parser.add_argument('--gpuid', default='0', help='This option specifies a GPU to be used for the computations.')
infant/infant_recon_py:gpu = args.gpuid if args.usegpu else '-1'
infant/infant_recon_py:    commands.append(f'sscnn_skullstrip -i {conf} -o work/sscnn -c t1w --gpu {gpu}')
infant/infant_recon_all.help.xml:      <argument>--usegpu</argument>
infant/infant_recon_all.help.xml:      <explanation>Use the powers of GPU for the skullstripping computations.</explanation>
infant/infant_recon_all.help.xml:      <argument>--gpuid</argument>
infant/infant_recon_all.help.xml:      <explanation>This option specifies the specific GPU id to be used for the computations. By default it iset to be 1.</explanation>
sscnn_skullstripping/sscnn_skullstrip:parser.add_argument('--gpu', type=int, default=-1, help='GPU number - if not supplied, CPU is used')
sscnn_skullstripping/sscnn_skullstrip:os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
sscnn_skullstripping/sscnn_skullstrip:os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:from keras.utils import  multi_gpu_model
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                  loss='mean_absolute_error', initial_learning_rate=0.00001, deconvolution=False, use_patches=True, num_gpus=1):
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:        if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:            if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:            if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                  num_gpus=1, num_outputs=1):
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:        if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:            if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:            if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                  num_gpus=1, num_outputs=1):
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:        if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:            if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:            if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                  loss='mean_absolute_error', initial_learning_rate=0.00001, deconvolution=False, use_patches=True, num_gpus=1):
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:        if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:            if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:            if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:#         if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:#                 parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:#             if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:#                     parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:#             if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:#                     parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                         kernel_size=(3,3), pool_size=(2,2), n_labels=0, num_outputs=1, num_gpus=1):
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:def build_compile_model(input_layer, input_shape, conv, n_labels,  loss, num_gpus, initial_learning_rate):
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:        if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:            if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:            if num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                    parallel_model = multi_gpu_model(model, gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                  loss='mean_squared_error', initial_learning_rate=0.00001, deconvolution=False, num_gpus=1, num_outputs=1, add_modality_channel=False):
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:                                                       loss, num_gpus, initial_learning_rate)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:    os.environ['LD_LIBRARY_PATH'] = '/usr/pubsw/packages/CUDA/lib64 '
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/unet_model.py:    #                    use_patches=True, num_gpus=1)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:class MultiGPUCheckpointCallback(Callback):
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:        super(MultiGPUCheckpointCallback, self).__init__()
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                 fcn=True, num_gpus=1, preprocessing=False, augment=False, num_outputs=1,
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:        self.num_gpus = num_gpus
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                                       deconvolution=False, use_patches=use_patches, num_gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                                                            num_gpus=num_gpus)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                                       deconvolution=False, use_patches=use_patches, num_gpus=num_gpus, num_outputs=num_outputs)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                                                         deconvolution=False, num_gpus=num_gpus, num_outputs=num_outputs,
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                  wmp_standardize=True, rob_standardize=True,fcn=True, num_gpus=1,
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                           wmp_standardize=wmp_standardize, rob_standardize=rob_standardize, fcn=fcn, num_gpus=1,
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:            if self.num_gpus == 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                            generator=self.feature_generator.training_generator_t1beta(batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                                batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                                batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                                batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                                batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                                batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                        self.model.fit_generator(generator=self.feature_generator.training_generator(batch_size=batch_size*self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                                         validation_data=self.feature_generator.validation_generator(batch_size=batch_size*self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                callback = MultiGPUCheckpointCallback(output_prefix, save_per_epoch, save_weights, initial_epoch)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                    generator=self.feature_generator.training_generator(batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                    validation_data=self.feature_generator.validation_generator(batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:            if self.num_gpus == 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                    # self.model.fit_generator(generator=self.feature_generator.seg_training_generator_multichannel_nmr_t2(batch_size=batch_size*self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                    #                  validation_data=self.feature_generator.seg_validation_generator_multichannel_nmr(batch_size=batch_size*self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                        generator=self.feature_generator.dynamic_seg_training_generator_multichannel_nmr_t2(batch_size=batch_size * self.num_gpus,
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                            batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                        batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                        batch_size=batch_size * self.num_gpus, focus=focus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                                                 batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                        batch_size=batch_size * self.num_gpus, focus=focus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                                                 batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                        batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                                                 batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                        generator=self.feature_generator.seg_training_generator_multichannel(batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                            batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                callback = MultiGPUCheckpointCallback(output_prefix, save_per_epoch, save_weights, initial_epoch)
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                    generator=self.feature_generator.training_generator(batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                    validation_data=self.feature_generator.validation_generator(batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:            self.model.fit_generator(generator=self.feature_generator.training_label_generator(batch_size=batch_size*self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                                         batch_size=batch_size * self.num_gpus),
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:        # if self.num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:        if self.num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:        if self.num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:    #     if self.num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:        if self.num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:        if self.num_gpus > 1:
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sscnn_skullstripping/sscnn_skullstripping/deeplearn_utils/DeepImageSynth.py:                                              num_gpus=1,
sscnn_skullstripping/README.md:$>./sscnn_skullstrip.py  -i t1.nii.gz -o /loc/output/dir  -c t1w -gpu 0
sscnn_skullstripping/README.md:-gpu: for GPU to run on. Set to -1 if not running on a GPU.
mri_synthstrip/mri_synthstrip:p.add_argument('-g', '--gpu', action='store_true', help='use the GPU')
mri_synthstrip/mri_synthstrip:gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
mri_synthstrip/mri_synthstrip:if args.gpu:
mri_synthstrip/mri_synthstrip:    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
mri_synthstrip/mri_synthstrip:    device = torch.device('cuda')
mri_synthstrip/mri_synthstrip:    device_name = 'GPU'
mri_synthstrip/mri_synthstrip:    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
mri_synthstrip/CMakeLists.txt:# cmake ... -DTEST_WITH_CUDA=ON ...
mri_synthstrip/CMakeLists.txt:if(TEST_WITH_CUDA)
mri_synthstrip/test.sh:# GPU flags
mri_synthstrip/test.sh:test_command mri_synthstrip --gpu -g -i in.mgz --out out.mgz
mri_WMHsynthseg/WMHSynthSeg/unet3d/trainer.py:    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
mri_WMHsynthseg/WMHSynthSeg/unet3d/trainer.py:        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')
mri_WMHsynthseg/WMHSynthSeg/unet3d/trainer.py:        model = model.cuda()
mri_WMHsynthseg/WMHSynthSeg/unet3d/trainer.py:    if torch.cuda.is_available() and not config['device'] == 'cpu':
mri_WMHsynthseg/WMHSynthSeg/unet3d/trainer.py:        model = model.cuda()
mri_WMHsynthseg/WMHSynthSeg/unet3d/trainer.py:        def _move_to_gpu(input):
mri_WMHsynthseg/WMHSynthSeg/unet3d/trainer.py:                return tuple([_move_to_gpu(x) for x in input])
mri_WMHsynthseg/WMHSynthSeg/unet3d/trainer.py:                if torch.cuda.is_available():
mri_WMHsynthseg/WMHSynthSeg/unet3d/trainer.py:                    input = input.cuda(non_blocking=True)
mri_WMHsynthseg/WMHSynthSeg/unet3d/trainer.py:        t = _move_to_gpu(t)
mri_WMHsynthseg/WMHSynthSeg/unet3d/config.py:    if torch.cuda.is_available():
mri_WMHsynthseg/WMHSynthSeg/unet3d/config.py:        config['device'] = 'cuda'
mri_WMHsynthseg/WMHSynthSeg/unet3d/config.py:        logger.warning('CUDA not available, using CPU')
mri_WMHsynthseg/WMHSynthSeg/unet3d/predictor.py:                # send batch to gpu
mri_WMHsynthseg/WMHSynthSeg/unet3d/predictor.py:                if torch.cuda.is_available():
mri_WMHsynthseg/WMHSynthSeg/unet3d/predictor.py:                    input = input.cuda(non_blocking=True)
mri_WMHsynthseg/WMHSynthSeg/unet3d/predictor.py:                # send batch to gpu
mri_WMHsynthseg/WMHSynthSeg/unet3d/predictor.py:                if torch.cuda.is_available():
mri_WMHsynthseg/WMHSynthSeg/unet3d/predictor.py:                    img = img.cuda(non_blocking=True)
mri_WMHsynthseg/WMHSynthSeg/unet3d/losses.py:            class_weights = torch.ones(input.size()[1]).float().cuda()
mri_WMHsynthseg/WMHSynthSeg/unet3d/losses.py:    if torch.cuda.is_available():
mri_WMHsynthseg/WMHSynthSeg/unet3d/losses.py:        loss = loss.cuda()
mri_WMHsynthseg/WMHSynthSeg/inference.py:    parser.add_argument("--device", default='cpu', help="device (cpu or cuda; optional)")
mri_WMHsynthseg/WMHSynthSeg/inference.py:    parser.add_argument("--crop", action="store_true", help="(optional) Does two passes, to limit size to 192x224x192 cuboid (needed for GPU processing)")
mri_WMHsynthseg/WMHSynthSeg/utils.py:        torch.cuda.empty_cache()
mri_segment_thalamic_nuclei_dti_cnn/mri_segment_thalamic_nuclei_dti_cnn:    parser.add_argument("--cpu", action="store_true", help="(optional) Enforce running with CPU rather than GPU.")
mri_segment_thalamic_nuclei_dti_cnn/mri_segment_thalamic_nuclei_dti_cnn:        print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
mri_segment_thalamic_nuclei_dti_cnn/mri_segment_thalamic_nuclei_dti_cnn:        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
mri_synthmorph/Dockerfile:FROM tensorflow/tensorflow:2.17.0-gpu AS base
mri_synthmorph/fs-synthmorph-reg:set gpuopt = ""
mri_synthmorph/fs-synthmorph-reg:  set cmd = (mri_synthmorph -m affine -t $affinelta $involcrop $targvolcrop  -j $threads $gpuopt)
mri_synthmorph/fs-synthmorph-reg:  set cmd = (mri_synthmorph -m deform -t $deform -i $affinelta $involcrop $targvolcrop -j $threads $gpuopt)
mri_synthmorph/mri_synthmorph:                Use the GPU in environment variable CUDA_VISIBLE_DEVICES or GPU
mri_synthmorph/mri_synthmorph:        CUDA_VISIBLE_DEVICES
mri_synthmorph/mri_synthmorph:                Use a specific GPU. If unset or empty, passing {b}-g{n} will
mri_synthmorph/mri_synthmorph:                select GPU 0. Ignored without {b}-g{n}.
mri_synthmorph/mri_synthmorph:r.add_argument('-g', dest='gpu', action='store_true')
mri_synthmorph/mri_synthmorph:    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
mri_synthmorph/mri_synthmorph:    os.environ['CUDA_VISIBLE_DEVICES'] = gpu if arg.gpu else ''
include/gcalinearnode.hpp:namespace GPU
include/gcalinearnode.hpp:class GCAnodeGPU;
include/gcalinearnode.hpp:  friend class GPU::Classes::GCAnodeGPU;
include/gcamorph.h:  // Pull routines out of gcamorph.c for GPU acceleration
include/gcalinearprior.hpp:namespace GPU
include/gcalinearprior.hpp:class GCApriorGPU;
include/gcalinearprior.hpp:  friend class GPU::Classes::GCApriorGPU;
recon_all_clinical/python/mri_synth_surf.py:    parser.add_argument("--cpu", action="store_true", help="enforce running with CPU rather than GPU.")
recon_all_clinical/python/mri_synth_surf.py:        print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
recon_all_clinical/python/mri_synth_surf.py:        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
CMakeLists.txt:option(TEST_WITH_CUDA "Allow tests to run that use cuda libs and require install of cuda drivers" OFF)
CMakeLists.txt:option(FSPYTHON_INSTALL_CUDA "install torch and tensorflow packages that support cuda into fspython" OFF)
CMakeLists.txt:  if(EXISTS /usr/lib64/nvidia/libGL.so)
CMakeLists.txt:    # in the nvidia drivers, so as a temporary fix, link directly to the nvidia GL library
CMakeLists.txt:    list(APPEND CMAKE_PREFIX_PATH "/usr/lib64/nvidia")
mris_pmake/C_mpmProg.cpp:/// a representation that is formatted for consumption by the OpenCL
mris_pmake/C_mpmProg.cpp:void C_mpmProg_autodijk_fast::genOpenCLGraphRepresentation(GraphData *graph)
mris_pmake/C_mpmProg.cpp:    genOpenCLGraphRepresentation(&graph);
mris_pmake/C_mpmProg.cpp:#ifdef FS_OPENCL
mris_pmake/C_mpmProg.cpp:    // OpenCL using either CPU-only, GPU+CPU, Multi GPU, or Multi GPU + CPU.  Based
mris_pmake/C_mpmProg.cpp:    // If compiled with OpenCL support, run the OpenCL version of the algorithm
mris_pmake/C_mpmProg.cpp:    runDijkstraOpenCL(&graph, &sourceVertices, results, 1);
mris_pmake/C_mpmProg.cpp:    // If not compiled with OpenCL, run the reference version of the algorithm
mris_pmake/oclCommon.h://      with OpenCL such as error detection and kernel loading.  Much of this
mris_pmake/oclCommon.h://      code was adapted from the NVIDIA GPU Computing SDK.
mris_pmake/oclCommon.h:    #include <OpenCL/cl.h>
mris_pmake/oclCommon.h:/// Print info about the device to stdout (modified from NVIDIA SDK)
mris_pmake/oclCommon.h:/// @param device         OpenCL id of the device
mris_pmake/oclCommon.h:/// Loads a Program file and prepends the cPreamble to the code. (from the NVIDIA SDK)
mris_pmake/oclCommon.h:/// Gets the id of the nth device from the context (from the NVIDIA SDK)
mris_pmake/oclCommon.h:/// @param cxGPUContext         OpenCL context
mris_pmake/oclCommon.h:cl_device_id oclGetDev(cl_context cxGPUContext, unsigned int nr);
mris_pmake/oclCommon.h:/// Gets the id of the first device from the context (from the NVIDIA SDK)
mris_pmake/oclCommon.h:/// @param cxGPUContext         OpenCL context
mris_pmake/oclCommon.h:cl_device_id oclGetFirstDev(cl_context cxGPUContext);
mris_pmake/oclCommon.h:/// Gets the id of device with maximal FLOPS from the context (from NVIDIA SDK)
mris_pmake/oclCommon.h:/// @param cxGPUContext         OpenCL context
mris_pmake/oclCommon.h:cl_device_id oclGetMaxFlopsDev(cl_context cxGPUContext);
mris_pmake/C_mpmProg.h:///        of the Single Source Shortest Path algorithm.  If compiled with OpenCL, it
mris_pmake/C_mpmProg.h:    /// a representation that is formatted for consumption by the OpenCL
mris_pmake/C_mpmProg.h:    void genOpenCLGraphRepresentation(GraphData *graph);
mris_pmake/dijkstra.cl.h://      Implementation of Dijkstra's Single-Source Shortest Path (SSSP) algorithm on the GPU. \n\
mris_pmake/dijkstra.cl.h://          \"Accelerating large graph algorithms on the GPU using CUDA\" by \n\
mris_pmake/oclDijkstraKernel.cpp://      Implementation of Dijkstra's Single-Source Shortest Path (SSSP) algorithm on the GPU.
mris_pmake/oclDijkstraKernel.cpp://          "Accelerating large graph algorithms on the GPU using CUDA" by
mris_pmake/oclDijkstraKernel.cpp:#ifdef FS_OPENCL
mris_pmake/oclDijkstraKernel.cpp:// This structure is used in the multi-GPU implementation of the algorithm.
mris_pmake/oclDijkstraKernel.cpp:// This structure defines the workload for each GPU.  The code chunks up
mris_pmake/oclDijkstraKernel.cpp:// the work on a per-GPU basis.
mris_pmake/oclDijkstraKernel.cpp:/// Load and build an OpenCL program from source file
mris_pmake/oclDijkstraKernel.cpp:/// \param gpuContext GPU context on which to load and build the program
mris_pmake/oclDijkstraKernel.cpp:cl_program loadAndBuildProgram( cl_context gpuContext, const char *fileName )
mris_pmake/oclDijkstraKernel.cpp:    // Load the OpenCL source code from the .cl file
mris_pmake/oclDijkstraKernel.cpp:    // Create the program for all GPUs in the context
mris_pmake/oclDijkstraKernel.cpp:    program = clCreateProgramWithSource(gpuContext, 1, (const char **)&source, &programLength, &errNum);
mris_pmake/oclDijkstraKernel.cpp:        clGetProgramBuildInfo(program, oclGetFirstDev(gpuContext), CL_PROGRAM_BUILD_LOG,
mris_pmake/oclDijkstraKernel.cpp:///  Allocate memory for input CUDA buffers and copy the data into device memory
mris_pmake/oclDijkstraKernel.cpp:void allocateOCLBuffers(cl_context gpuContext, cl_command_queue commandQueue, GraphData *graph,
mris_pmake/oclDijkstraKernel.cpp:    // First, need to create OpenCL Host buffers that can be copied to device buffers
mris_pmake/oclDijkstraKernel.cpp:    hostVertexArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
mris_pmake/oclDijkstraKernel.cpp:    hostEdgeArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
mris_pmake/oclDijkstraKernel.cpp:    hostWeightArrayBuffer = clCreateBuffer(gpuContext, CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
mris_pmake/oclDijkstraKernel.cpp:    // Now create all of the GPU buffers
mris_pmake/oclDijkstraKernel.cpp:    *vertexArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * globalWorkSize, NULL, &errNum);
mris_pmake/oclDijkstraKernel.cpp:    *edgeArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * graph->edgeCount, NULL, &errNum);
mris_pmake/oclDijkstraKernel.cpp:    *weightArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY, sizeof(float) * graph->edgeCount, NULL, &errNum);
mris_pmake/oclDijkstraKernel.cpp:    *maskArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(int) * globalWorkSize, NULL, &errNum);
mris_pmake/oclDijkstraKernel.cpp:    *costArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(float) * globalWorkSize, NULL, &errNum);
mris_pmake/oclDijkstraKernel.cpp:    *updatingCostArrayDevice = clCreateBuffer(gpuContext, CL_MEM_READ_WRITE, sizeof(float) * globalWorkSize, NULL, &errNum);
mris_pmake/oclDijkstraKernel.cpp:/// Initialize OpenCL buffers for single run of Dijkstra
mris_pmake/oclDijkstraKernel.cpp:/// CPU + GPU, GPU, or Multi GPU depending on what compute resources are available
mris_pmake/oclDijkstraKernel.cpp:void runDijkstraOpenCL( GraphData* graph, int *sourceVertices,
mris_pmake/oclDijkstraKernel.cpp:    cl_context gpuContext;
mris_pmake/oclDijkstraKernel.cpp:    // create the OpenCL context on available GPU devices
mris_pmake/oclDijkstraKernel.cpp:    gpuContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
mris_pmake/oclDijkstraKernel.cpp:    // Create an OpenCL context on available CPU devices
mris_pmake/oclDijkstraKernel.cpp:    if (cpuContext == 0 && gpuContext == 0)
mris_pmake/oclDijkstraKernel.cpp:        cerr << "ERROR: could not create any OpenCL context on CPU or GPU" << endl;
mris_pmake/oclDijkstraKernel.cpp:    // For just a single result, just use multi-threaded CPU or single GPU
mris_pmake/oclDijkstraKernel.cpp:        if (gpuContext != 0)
mris_pmake/oclDijkstraKernel.cpp:            cout << "Dijkstra OpenCL: Running single GPU version." << endl;
mris_pmake/oclDijkstraKernel.cpp:            runDijkstra(gpuContext, oclGetMaxFlopsDev(gpuContext), graph, sourceVertices,
mris_pmake/oclDijkstraKernel.cpp:            cout << "Dijkstra OpenCL: Running multithreaded CPU version." << endl;
mris_pmake/oclDijkstraKernel.cpp:    // For multiple results, prefer multi-GPU and fallback to CPU
mris_pmake/oclDijkstraKernel.cpp:        // Prefer Multi-GPU if multiple GPUs are available
mris_pmake/oclDijkstraKernel.cpp:        if (gpuContext != 0)
mris_pmake/oclDijkstraKernel.cpp:            cout << "Dijkstra OpenCL: Running multi-GPU version." << endl;
mris_pmake/oclDijkstraKernel.cpp:            runDijkstraMultiGPU( gpuContext, graph, sourceVertices,
mris_pmake/oclDijkstraKernel.cpp:        // For now, fallback to CPU in this case.  I have a multi GPU+CPU path
mris_pmake/oclDijkstraKernel.cpp:        // running the GPU version slows down the CPU version.
mris_pmake/oclDijkstraKernel.cpp:            cout << "Dijkstra OpenCL: Running multithreaded CPU version." << endl;
mris_pmake/oclDijkstraKernel.cpp:    clReleaseContext(gpuContext);
mris_pmake/oclDijkstraKernel.cpp:/// This function will run the algorithm on a single GPU.
mris_pmake/oclDijkstraKernel.cpp:/// \param gpuContext Current GPU context, must be created by caller
mris_pmake/oclDijkstraKernel.cpp:///                 GPU version will automatically split the work across
mris_pmake/oclDijkstraKernel.cpp:void runDijkstra( cl_context gpuContext, cl_device_id deviceId, GraphData* graph,
mris_pmake/oclDijkstraKernel.cpp:    commandQueue = clCreateCommandQueue( gpuContext, deviceId, 0, &errNum );
mris_pmake/oclDijkstraKernel.cpp:    cl_program program = loadAndBuildProgram( gpuContext, "dijkstra.cl" );
mris_pmake/oclDijkstraKernel.cpp:    allocateOCLBuffers( gpuContext, commandQueue, graph, &vertexArrayDevice, &edgeArrayDevice, &weightArrayDevice,
mris_pmake/oclDijkstraKernel.cpp:/// This function will run the algorithm on as many GPUs as is available.  It will
mris_pmake/oclDijkstraKernel.cpp:/// create N threads, one for each GPU, and chunk the workload up to perform
mris_pmake/oclDijkstraKernel.cpp:/// (numResults / N) searches per GPU.
mris_pmake/oclDijkstraKernel.cpp:/// \param gpuContext Current GPU context, must be created by caller
mris_pmake/oclDijkstraKernel.cpp:void runDijkstraMultiGPU( cl_context gpuContext, GraphData* graph, int *sourceVertices,
mris_pmake/oclDijkstraKernel.cpp:    // Find out how many GPU's to compute on all available GPUs
mris_pmake/oclDijkstraKernel.cpp:    errNum = clGetContextInfo(gpuContext, CL_CONTEXT_DEVICES, 0, NULL, &deviceBytes);
mris_pmake/oclDijkstraKernel.cpp:        cerr << "ERROR: no GPUs present!" << endl;
mris_pmake/oclDijkstraKernel.cpp:        devicePlans[i].context = gpuContext;
mris_pmake/oclDijkstraKernel.cpp:        devicePlans[i].deviceId = oclGetDev(gpuContext, i);;
mris_pmake/oclDijkstraKernel.cpp:    // Add any remaining work to the last GPU
mris_pmake/oclDijkstraKernel.cpp:/// This function will run the algorithm on as many GPUs as is available along with
mris_pmake/oclDijkstraKernel.cpp:/// \param gpuContext Current GPU context, must be created by caller
mris_pmake/oclDijkstraKernel.cpp:void runDijkstraMultiGPUandCPU( cl_context gpuContext, cl_context cpuContext, GraphData* graph,
mris_pmake/oclDijkstraKernel.cpp:    float ratioCPUtoGPU = 0.65; // CPU seems to run it at 2.26X on GT120 GPU
mris_pmake/oclDijkstraKernel.cpp:    // Find out how many GPU's to compute on all available GPUs
mris_pmake/oclDijkstraKernel.cpp:    cl_uint gpuDeviceCount;
mris_pmake/oclDijkstraKernel.cpp:    errNum = clGetContextInfo(gpuContext, CL_CONTEXT_DEVICES, 0, NULL, &deviceBytes);
mris_pmake/oclDijkstraKernel.cpp:    gpuDeviceCount = (cl_uint)deviceBytes/sizeof(cl_device_id);
mris_pmake/oclDijkstraKernel.cpp:    if (gpuDeviceCount == 0)
mris_pmake/oclDijkstraKernel.cpp:        cerr << "ERROR: no GPUs present!" << endl;
mris_pmake/oclDijkstraKernel.cpp:    cl_uint totalDeviceCount = gpuDeviceCount + cpuDeviceCount;
mris_pmake/oclDijkstraKernel.cpp:    int gpuResults = numResults / (ratioCPUtoGPU);
mris_pmake/oclDijkstraKernel.cpp:    int cpuResults = numResults - gpuResults;
mris_pmake/oclDijkstraKernel.cpp:    int resultsPerGPU = gpuResults / totalDeviceCount;
mris_pmake/oclDijkstraKernel.cpp:    for (unsigned int i = 0; i < gpuDeviceCount; i++)
mris_pmake/oclDijkstraKernel.cpp:        devicePlans[curDevice].context = gpuContext;
mris_pmake/oclDijkstraKernel.cpp:        devicePlans[curDevice].deviceId = oclGetDev(gpuContext, i);;
mris_pmake/oclDijkstraKernel.cpp:        devicePlans[curDevice].numResults = resultsPerGPU;
mris_pmake/oclDijkstraKernel.cpp:        offset += resultsPerGPU;
mris_pmake/oclDijkstraKernel.cpp:    // Add any remaining work to the last GPU
mris_pmake/oclDijkstraKernel.cpp:#endif // FS_OPENCL
mris_pmake/dijkstra.cl://      Implementation of Dijkstra's Single-Source Shortest Path (SSSP) algorithm on the GPU.
mris_pmake/dijkstra.cl://          \"Accelerating large graph algorithms on the GPU using CUDA\" by
mris_pmake/oclDijkstraKernel.h://      Implementation of Dijkstra's Single-Source Shortest Path (SSSP) algorithm on the GPU.
mris_pmake/oclDijkstraKernel.h://          "Accelerating large graph algorithms on the GPU using CUDA" by
mris_pmake/oclDijkstraKernel.h:#ifdef FS_OPENCL
mris_pmake/oclDijkstraKernel.h:        #include <OpenCL/cl.h>      
mris_pmake/oclDijkstraKernel.h://  Accelerating large graph algorithms on the GPU using CUDA by
mris_pmake/oclDijkstraKernel.h:#ifdef FS_OPENCL
mris_pmake/oclDijkstraKernel.h:/// CPU + GPU, GPU, or Multi GPU depending on what compute resources are available
mris_pmake/oclDijkstraKernel.h:void runDijkstraOpenCL( GraphData* graph, int *sourceVertices, 
mris_pmake/oclDijkstraKernel.h:/// This function will run the algorithm on a single GPU.
mris_pmake/oclDijkstraKernel.h:/// \param gpuContext Current GPU context, must be created by caller
mris_pmake/oclDijkstraKernel.h:///                 GPU version will automatically split the work across
mris_pmake/oclDijkstraKernel.h:void runDijkstra( cl_context gpuContext, cl_device_id deviceId, GraphData* graph,
mris_pmake/oclDijkstraKernel.h:/// This function will run the algorithm on as many GPUs as is available.  It will
mris_pmake/oclDijkstraKernel.h:/// create N threads, one for each GPU, and chunk the workload up to perform
mris_pmake/oclDijkstraKernel.h:/// (numResults / N) searches per GPU.
mris_pmake/oclDijkstraKernel.h:/// \param gpuContext Current GPU context, must be created by caller
mris_pmake/oclDijkstraKernel.h:void runDijkstraMultiGPU( cl_context gpuContext, GraphData* graph, int *sourceVertices,
mris_pmake/oclDijkstraKernel.h:/// This function will run the algorithm on as many GPUs as is available along with
mris_pmake/oclDijkstraKernel.h:/// \param gpuContext Current GPU context, must be created by caller
mris_pmake/oclDijkstraKernel.h:void runDijkstraMultiGPUandCPU( cl_context gpuContext, cl_context cpuContext, GraphData* graph,
mris_pmake/oclDijkstraKernel.h:#endif // FS_OPENCL
mris_pmake/CMakeLists.txt:if(OpenCL_FOUND)
mris_pmake/CMakeLists.txt:  include_directories(${OpenCL_INCLUDE_DIR})
mris_pmake/CMakeLists.txt:if(OpenCL_FOUND)
mris_pmake/CMakeLists.txt:if(OpenCL_FOUND)
mris_pmake/CMakeLists.txt:  target_link_libraries(mris_pmake ${OpenCL_LIBRARY})
mris_pmake/oclCommon.cpp://      with OpenCL such as error detection and kernel loading.
mris_pmake/oclCommon.cpp:/// Print info about the device to stdout (modified from NVIDIA SDK)
mris_pmake/oclCommon.cpp:/// @param device         OpenCL id of the device
mris_pmake/oclCommon.cpp:    if( type & CL_DEVICE_TYPE_GPU )
mris_pmake/oclCommon.cpp:        fprintf(stdout, "  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_GPU");
mris_pmake/oclCommon.cpp:/// Loads a Program file and prepends the cPreamble to the code. (from the NVIDIA SDK)
mris_pmake/oclCommon.cpp:    // open the OpenCL source code file
mris_pmake/oclCommon.cpp:/// Gets the id of the nth device from the context (from the NVIDIA SDK)
mris_pmake/oclCommon.cpp:/// @param cxGPUContext         OpenCL context
mris_pmake/oclCommon.cpp:cl_device_id oclGetDev(cl_context cxGPUContext, unsigned int nr)
mris_pmake/oclCommon.cpp:    // get the list of GPU devices associated with context
mris_pmake/oclCommon.cpp:    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
mris_pmake/oclCommon.cpp:    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);
mris_pmake/oclCommon.cpp:/// Gets the id of the first device from the context (from the NVIDIA SDK)
mris_pmake/oclCommon.cpp:/// @param cxGPUContext         OpenCL context
mris_pmake/oclCommon.cpp:cl_device_id oclGetFirstDev(cl_context cxGPUContext)
mris_pmake/oclCommon.cpp:    // get the list of GPU devices associated with context
mris_pmake/oclCommon.cpp:    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
mris_pmake/oclCommon.cpp:    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);
mris_pmake/oclCommon.cpp:/// Gets the id of device with maximal FLOPS from the context (from NVIDIA SDK)
mris_pmake/oclCommon.cpp:/// @param cxGPUContext         OpenCL context
mris_pmake/oclCommon.cpp:cl_device_id oclGetMaxFlopsDev(cl_context cxGPUContext)
mris_pmake/oclCommon.cpp:    // get the list of GPU devices associated with context
mris_pmake/oclCommon.cpp:    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
mris_pmake/oclCommon.cpp:    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);
scripts/install_nnUNet_v1.7_instructions.md:    -c                                  # if passed, install cuda
scripts/install_nnUNet_v1.7_instructions.md:NOTE: Do not pass the -c flag on a Mac, cuda is not available for MacOS
scripts/bedpostx_mgh:    #echo "-c do not use CUDA capable hardware/queue (if found)"
scripts/bedpostx_mgh:nocuda=0
scripts/bedpostx_mgh:        -c) nocuda=1;;
scripts/bedpostx_mgh:	if [ -n "$FSLGECUDAQ" -a "$nocuda" -eq 0 ]; then
scripts/bedpostx_mgh:		qconf -sq $FSLGECUDAQ 2>&1 >/dev/null
scripts/bedpostx_mgh:			# CUDA queue exists
scripts/bedpostx_mgh:			if [ -x $0_gpu ]; then
scripts/bedpostx_mgh:				exec $0_gpu $myopts
scripts/xhemireg:    set regpure = $outdir/lrrev.pure.register.dat
scripts/xhemireg:    rm -f $regpure
scripts/xhemireg:    echo "subject-unknown" >> $regpure
scripts/xhemireg:    echo "1" >> $regpure
scripts/xhemireg:    echo "1" >> $regpure
scripts/xhemireg:    echo "1" >> $regpure
scripts/xhemireg:    echo "$xsign 0 0 0" >> $regpure
scripts/xhemireg:    echo " 0 $ysign 0 0" >> $regpure
scripts/xhemireg:    echo " 0 0 $zsign 0" >> $regpure
scripts/xhemireg:    echo " 0 0 0 1" >> $regpure
scripts/xhemireg:    echo "round" >> $regpure
scripts/xhemireg:      set cmd = ($cmd --reg $regpure)
scripts/create_nnUNet_v1.7_env.sh:    echo "  -c # boolean flag, pass to install cuda, Linux only" 
scripts/create_nnUNet_v1.7_env.sh:INSTALL_CUDA=0
scripts/create_nnUNet_v1.7_env.sh:        -c|--cuda)
scripts/create_nnUNet_v1.7_env.sh:            INSTALL_CUDA=1
scripts/create_nnUNet_v1.7_env.sh:# append the cuda dependencies if --cuda passed
scripts/create_nnUNet_v1.7_env.sh:if [[ $INSTALL_CUDA -eq 1 ]]; then
scripts/create_nnUNet_v1.7_env.sh:    CONDA_CREATE_CMD="$CONDA_CREATE_CMD pytorch-cuda=11.8 -c nvidia"
scripts/CMakeLists.txt:  fs_install_cuda
scripts/topofit:set UseGPU = 0
scripts/topofit:  if(! $UseGPU) set cmd = ($cmd --cpu)
scripts/topofit:    case "--gpu":
scripts/topofit:      set UseGPU = 1
scripts/topofit:    case "--no-gpu":
scripts/topofit:      set UseGPU = 0
scripts/topofit:  echo " --gpu : use GPU (may not work; does not apply to sphere)"
scripts/recon-all.v6.hires:  echo "  -use-gpu : use GPU versions of mri_em_register, mri_ca_register,"
scripts/recon-all.v6.hires:-use-gpu
scripts/recon-all.v6.hires:Use the GPU versions of the binaries mri_em_register, mri_ca_register,
scripts/hgoutgoing.bash:/usr/bin/curl http://avebury-vm.nmr.mgh.harvard.edu/redmine/projects/fsgpu/repository -o /dev/null >& /dev/null
scripts/recon-all:    if(! $UseGPU)  set cmd = ($cmd --cpu)
scripts/recon-all:  if($UseGPU)            set cmd = ($cmd --gpu)
scripts/fs_install_cuda:## $ sudo FREESURFER_HOME=/usr/local/freesurfer/7.5.0 ./fs_install_cuda
scripts/fs_install_cuda:## $ FREESURFER_HOME=/usr/local/freesurfer/7.5.0 ./fs_install_cuda
scripts/fs_install_cuda:## installation to install the cuda version of torch along with other cuda python
scripts/fs_install_cuda:    echo "INFO: sudo FREESURFER_HOME=\$FREESURFER_HOME fs_install_cuda $@"
scripts/fs_install_cuda:   ## remove cpu version and install non-cpu version which will also add other nvidia packages
scripts/fs_install_cuda:   ## check there is now a libtorch_cuda.so
scripts/fs_install_cuda:   path_torch_libcuda="$path_dev_base/python/lib/python3.8/site-packages/torch/lib/libtorch_cuda.so"
scripts/fs_install_cuda:   if [ ! -e $path_torch_libcuda ]; then
scripts/fs_install_cuda:      echo "Cannot stat a libtorch_cuda.so = $path_torch_libcuda in fspython torch module"
scripts/fs_install_cuda:      echo "cuda install success"
scripts/rca-fix-ento:learning model that may require as much as 20GB (GPU not needed).
mris_register_josa/README.md:- Current tensorflow is a cpu version in the conda env (better to have the gpu version, but will not break things)
mri_sclimbic_seg/mri_sclimbic_seg:    parser.add_argument('--cuda-device', help='Cuda device for GPU support.')
mri_sclimbic_seg/mri_sclimbic_seg:    # configure cuda device
mri_sclimbic_seg/mri_sclimbic_seg:    if args.cuda_device is not None:
mri_sclimbic_seg/mri_sclimbic_seg:        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
mri_sclimbic_seg/mri_sclimbic_seg:    cuda_device = os.getenv('CUDA_VISIBLE_DEVICES')
mri_sclimbic_seg/mri_sclimbic_seg:    if cuda_device is None or cuda_device == '-1':
mri_sclimbic_seg/mri_sclimbic_seg:        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
mri_sclimbic_seg/mri_sclimbic_seg:        print('Using GPU device', cuda_device)
mri_easyreg/mri_easywarp:    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
mri_easyreg/mri_easyreg:os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
mri_synthsr/mri_synthsr:    parser.add_argument("--cpu", action="store_true", help="(optional) Enforce running with CPU rather than GPU.")
mri_synthsr/mri_synthsr:        print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
mri_synthsr/mri_synthsr:        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
mri_synthsr/mri_synthsr_hyperfine:    parser.add_argument("--cpu", action="store_true", help="(optional) Enforce running with CPU rather than GPU.")
mri_synthsr/mri_synthsr_hyperfine:        print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
mri_synthsr/mri_synthsr_hyperfine:        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
cnn_sphere_register/ext/neuron/neuron/models.py:        # Omittting BatchNorm for now, it seems to have a cpu vs gpu problem
cnn_sphere_register/ext/neuron/neuron/utils.py:            #   It works on GPU because we do not perform index validation checking on GPU -- it's too 
cnn_sphere_register/ext/neuron/neuron/utils.py:def robust_multi_gpu_model(model, gpus, verbose=True):
cnn_sphere_register/ext/neuron/neuron/utils.py:    re-work keras model for multi-gpus if number of gpus is > 1
cnn_sphere_register/ext/neuron/neuron/utils.py:        gpus: list of gpus to split to (e.g. [1, 4, 6]), or count of gpus available (e.g. 3)
cnn_sphere_register/ext/neuron/neuron/utils.py:            Note: if given int, assume that is the count of gpus, 
cnn_sphere_register/ext/neuron/neuron/utils.py:            so if you want a single specific gpu, this function will not do that.
cnn_sphere_register/ext/neuron/neuron/utils.py:    islist = isinstance(gpus, (list, tuple))
cnn_sphere_register/ext/neuron/neuron/utils.py:    if (islist and len(gpus) > 1) or (not islist and gpus > 1):
cnn_sphere_register/ext/neuron/neuron/utils.py:        count = gpus if not islist else len(gpus)
cnn_sphere_register/ext/neuron/neuron/utils.py:        print("Returning multi-gpu (%d) model" % count)
cnn_sphere_register/ext/neuron/neuron/utils.py:        return keras.utils.multi_gpu_model(model, count)
cnn_sphere_register/ext/neuron/neuron/utils.py:        print("Returning keras model back (single gpu found)")
cnn_sphere_register/src/test_jc.py:def test(gpu_id, model_dir, iter_num, data_dir, file_name,
cnn_sphere_register/src/test_jc.py:         compute_type = 'GPU',  # GPU or CPU
cnn_sphere_register/src/test_jc.py:    # GPU handling
cnn_sphere_register/src/test_jc.py:    gpu = '/gpu:' + str(gpu_id)
cnn_sphere_register/src/test_jc.py:    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
cnn_sphere_register/src/test_jc.py:    config.gpu_options.allow_growth = True
cnn_sphere_register/src/test_jc.py:    with tf.device(gpu):
cnn_sphere_register/src/test_jc.py:    with tf.device(gpu):
cnn_sphere_register/src/test_jc.py:    else:  # GPU
cnn_sphere_register/src/test_jc.py:    python test_jc.py gpu_id model_dir data_dir iter_num
cnn_sphere_register/src/train_pad.py:def train(data_dir, model_dir, gpu_id, lr, n_iterations, alpha, model_save_iter, gamma=10000, batch_size=1):
cnn_sphere_register/src/train_pad.py:    :param gpu_id: integer specifying the gpu to use
cnn_sphere_register/src/train_pad.py:    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
cnn_sphere_register/src/train_pad.py:    # gpu handling
cnn_sphere_register/src/train_pad.py:    gpu = '/gpu:' + str(gpu_id)
cnn_sphere_register/src/train_pad.py:    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
cnn_sphere_register/src/train_pad.py:    config.gpu_options.allow_growth = True
cnn_sphere_register/src/train_pad.py:    with tf.device(gpu):
cnn_sphere_register/src/train_pad.py:        with tf.device(gpu):
cnn_sphere_register/src/train_pad.py:        with tf.device(gpu):
cnn_sphere_register/src/train_pad.py:    parser.add_argument("--gpu", type=int, default=0,
cnn_sphere_register/src/train_pad.py:                        dest="gpu_id", help="gpu id number")
CNN/CXR/train_serial_cxr.py:gpu_number = 0
CNN/CXR/train_serial_cxr.py:    gpu_number = 1
CNN/CXR/train_serial_cxr.py:    gpu_number = 0
CNN/CXR/train_serial_cxr.py:    gpu_number = 1
CNN/CXR/train_serial_cxr.py:elif (host == 'mlscgpu1'):
CNN/CXR/train_serial_cxr.py:    gpu_number = 5
CNN/CXR/train_serial_cxr.py:elif (host == 'mlscgpu2.nmr.mgh.harvard.edu'):
CNN/CXR/train_serial_cxr.py:    gpu_number = 5
CNN/CXR/train_cxr.py:gpu_number = 0
CNN/CXR/train_cxr.py:    gpu_number = 0
CNN/CXR/train_cxr.py:    gpu_number = 0
CNN/CXR/train_cxr.py:    gpu_number = 1
CNN/CXR/train_cxr.py:elif (host == 'mlscgpu1'):
CNN/CXR/train_cxr.py:    gpu_number = 5
CNN/CXR/train_cxr.py:elif (host == 'mlscgpu2.nmr.mgh.harvard.edu'):
CNN/CXR/train_cxr.py:    gpu_number = 5
CNN/CXR/train_cxr.py:print('running on host %s, GPU %d, train_affine %s' % (host, gpu_number, str(train_affine)))
CNN/CXR/train_cxr.py:fsd.configure(gpu=gpu_number)
utils/gcamorph.cpp:      Write out initial values to enable CUDA conversion
mri_segment_hypothalamic_subunits/mri_segment_hypothalamic_subunits:                                                           "than GPU.")
mri_segment_hypothalamic_subunits/mri_segment_hypothalamic_subunits:        print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
mri_segment_hypothalamic_subunits/mri_segment_hypothalamic_subunits:        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
freeview/LayerROI.cpp:#include "vtkImageOpenClose3D.h"
freeview/LayerROI.cpp:      vtkSmartPointer<vtkImageOpenClose3D> filter = vtkSmartPointer<vtkImageOpenClose3D>::New();
freeview/LayerROI.cpp:      vtkSmartPointer<vtkImageOpenClose3D> filter = vtkSmartPointer<vtkImageOpenClose3D>::New();

```
