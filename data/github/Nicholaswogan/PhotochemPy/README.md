# https://github.com/Nicholaswogan/PhotochemPy

```console
src/cvode-5.7.0/include/sunmatrix/sunmatrix_cusparse.h:#include <cuda_runtime.h>
src/cvode-5.7.0/include/sunmatrix/sunmatrix_cusparse.h:#include <sundials/sundials_cuda_policies.hpp>
src/cvode-5.7.0/include/sunmatrix/sunmatrix_cusparse.h:#if CUDART_VERSION >= 11000
src/cvode-5.7.0/include/sunmatrix/sunmatrix_cusparse.h:  SUNCudaExecPolicy* exec_policy;
src/cvode-5.7.0/include/sunmatrix/sunmatrix_cusparse.h:SUNDIALS_EXPORT int SUNMatrix_cuSparse_SetKernelExecPolicy(SUNMatrix A, SUNCudaExecPolicy* exec_policy);
src/cvode-5.7.0/include/sunmatrix/sunmatrix_magmadense.h:#if defined(SUNDIALS_MAGMA_BACKENDS_CUDA)
src/cvode-5.7.0/include/sunmemory/sunmemory_cuda.h: * SUNDIALS CUDA memory helper header file.
src/cvode-5.7.0/include/sunmemory/sunmemory_cuda.h:#ifndef _SUNDIALS_CUDAMEMORY_H
src/cvode-5.7.0/include/sunmemory/sunmemory_cuda.h:#define _SUNDIALS_CUDAMEMORY_H
src/cvode-5.7.0/include/sunmemory/sunmemory_cuda.h:#include <cuda_runtime.h>
src/cvode-5.7.0/include/sunmemory/sunmemory_cuda.h:SUNMemoryHelper SUNMemoryHelper_Cuda();
src/cvode-5.7.0/include/sunmemory/sunmemory_cuda.h:SUNDIALS_EXPORT int SUNMemoryHelper_Alloc_Cuda(SUNMemoryHelper helper, SUNMemory* memptr,
src/cvode-5.7.0/include/sunmemory/sunmemory_cuda.h:SUNDIALS_EXPORT int SUNMemoryHelper_Dealloc_Cuda(SUNMemoryHelper helper, SUNMemory mem);
src/cvode-5.7.0/include/sunmemory/sunmemory_cuda.h:SUNDIALS_EXPORT int SUNMemoryHelper_Copy_Cuda(SUNMemoryHelper helper, SUNMemory dst,
src/cvode-5.7.0/include/sunmemory/sunmemory_cuda.h:SUNDIALS_EXPORT int SUNMemoryHelper_CopyAsync_Cuda(SUNMemoryHelper helper, SUNMemory dst,
src/cvode-5.7.0/include/sunlinsol/sunlinsol_cusolversp_batchqr.h:#include <cuda_runtime.h>
src/cvode-5.7.0/include/sunlinsol/sunlinsol_magmadense.h:#if defined(SUNDIALS_MAGMA_BACKENDS_CUDA)
src/cvode-5.7.0/include/sundials/sundials_config.in:  * the CUDA NVector.
src/cvode-5.7.0/include/sundials/sundials_config.in:  * the CUDA NVector.
src/cvode-5.7.0/include/sundials/sundials_config.in:#cmakedefine SUNDIALS_MAGMA_BACKENDS_CUDA
src/cvode-5.7.0/include/sundials/sundials_config.in:#cmakedefine SUNDIALS_RAJA_BACKENDS_CUDA
src/cvode-5.7.0/include/sundials/sundials_nvector.h:  SUNDIALS_NVEC_CUDA,
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp: * This header files defines the CudaExecPolicy classes which
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp: * are utilized to determine CUDA kernel launch paramaters.
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:#ifndef _SUNDIALS_CUDAEXECPOLICIES_HPP
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:#define _SUNDIALS_CUDAEXECPOLICIES_HPP
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:#include <cuda_runtime.h>
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:class CudaExecPolicy
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  virtual const cudaStream_t* stream() const = 0;
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  virtual CudaExecPolicy* clone() const = 0;
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  virtual ~CudaExecPolicy() {}
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:class CudaThreadDirectExecPolicy : public CudaExecPolicy
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  CudaThreadDirectExecPolicy(const size_t blockDim, const cudaStream_t stream = 0)
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  CudaThreadDirectExecPolicy(const CudaThreadDirectExecPolicy& ex)
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  virtual const cudaStream_t* stream() const
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  virtual CudaExecPolicy* clone() const
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:    return static_cast<CudaExecPolicy*>(new CudaThreadDirectExecPolicy(*this));
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  const cudaStream_t stream_;
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:class CudaGridStrideExecPolicy : public CudaExecPolicy
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  CudaGridStrideExecPolicy(const size_t blockDim, const size_t gridDim, const cudaStream_t stream = 0)
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  CudaGridStrideExecPolicy(const CudaGridStrideExecPolicy& ex)
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  virtual const cudaStream_t* stream() const
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  virtual CudaExecPolicy* clone() const
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:    return static_cast<CudaExecPolicy*>(new CudaGridStrideExecPolicy(*this));
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  const cudaStream_t stream_;
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp: * multiple of the CUDA warp size. The number of blocks (gridSize) can be set to
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:class CudaBlockReduceExecPolicy : public CudaExecPolicy
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  CudaBlockReduceExecPolicy(const size_t blockDim, const size_t gridDim = 0, const cudaStream_t stream = 0)
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:      throw std::invalid_argument("the block size must be a multiple of the CUDA warp size");
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  CudaBlockReduceExecPolicy(const CudaBlockReduceExecPolicy& ex)
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  virtual const cudaStream_t* stream() const
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  virtual CudaExecPolicy* clone() const
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:    return static_cast<CudaExecPolicy*>(new CudaBlockReduceExecPolicy(*this));
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:  const cudaStream_t stream_;
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:typedef sundials::CudaExecPolicy SUNCudaExecPolicy;
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:typedef sundials::CudaThreadDirectExecPolicy SUNCudaThreadDirectExecPolicy;
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:typedef sundials::CudaGridStrideExecPolicy SUNCudaGridStrideExecPolicy;
src/cvode-5.7.0/include/sundials/sundials_cuda_policies.hpp:typedef sundials::CudaBlockReduceExecPolicy SUNCudaBlockReduceExecPolicy;
src/cvode-5.7.0/include/nvector/nvector_cuda.h: * This is the header file for the CUDA implementation of the
src/cvode-5.7.0/include/nvector/nvector_cuda.h:#ifndef _NVECTOR_CUDA_H
src/cvode-5.7.0/include/nvector/nvector_cuda.h:#define _NVECTOR_CUDA_H
src/cvode-5.7.0/include/nvector/nvector_cuda.h:#include <cuda_runtime.h>
src/cvode-5.7.0/include/nvector/nvector_cuda.h:#include <sundials/sundials_cuda_policies.hpp>
src/cvode-5.7.0/include/nvector/nvector_cuda.h:#include <sunmemory/sunmemory_cuda.h>
src/cvode-5.7.0/include/nvector/nvector_cuda.h: * CUDA implementation of N_Vector
src/cvode-5.7.0/include/nvector/nvector_cuda.h:struct _N_VectorContent_Cuda
src/cvode-5.7.0/include/nvector/nvector_cuda.h:  SUNCudaExecPolicy* stream_exec_policy;
src/cvode-5.7.0/include/nvector/nvector_cuda.h:  SUNCudaExecPolicy* reduce_exec_policy;
src/cvode-5.7.0/include/nvector/nvector_cuda.h:typedef struct _N_VectorContent_Cuda *N_VectorContent_Cuda;
src/cvode-5.7.0/include/nvector/nvector_cuda.h: * NVECTOR_CUDA implementation specific functions
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT N_Vector N_VNewEmpty_Cuda();
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT N_Vector N_VNew_Cuda(sunindextype length);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT N_Vector N_VNewManaged_Cuda(sunindextype length);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT N_Vector N_VNewWithMemHelp_Cuda(sunindextype length,
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT N_Vector N_VMake_Cuda(sunindextype length,
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT N_Vector N_VMakeManaged_Cuda(sunindextype length,
src/cvode-5.7.0/include/nvector/nvector_cuda.h:   Use N_VNewWithMemHelp_Cuda instead.
src/cvode-5.7.0/include/nvector/nvector_cuda.h:N_Vector N_VMakeWithManagedAllocator_Cuda(sunindextype length,
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT void N_VSetHostArrayPointer_Cuda(realtype* h_vdata, N_Vector v);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT void N_VSetDeviceArrayPointer_Cuda(realtype* d_vdata, N_Vector v);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT booleantype N_VIsManagedMemory_Cuda(N_Vector x);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VSetKernelExecPolicy_Cuda(N_Vector x,
src/cvode-5.7.0/include/nvector/nvector_cuda.h:                                                SUNCudaExecPolicy* stream_exec_policy,
src/cvode-5.7.0/include/nvector/nvector_cuda.h:                                                SUNCudaExecPolicy* reduce_exec_policy);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT void N_VCopyToDevice_Cuda(N_Vector v);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT void N_VCopyFromDevice_Cuda(N_Vector v);
src/cvode-5.7.0/include/nvector/nvector_cuda.h: /* DEPRECATED (to be removed in SUNDIALS v6): use N_VSetKerrnelExecPolicy_Cuda instead */
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_DEPRECATED_EXPORT void N_VSetCudaStream_Cuda(N_Vector x, cudaStream_t *stream);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:sunindextype N_VGetLength_Cuda(N_Vector x)
src/cvode-5.7.0/include/nvector/nvector_cuda.h:  N_VectorContent_Cuda content = (N_VectorContent_Cuda)x->content;
src/cvode-5.7.0/include/nvector/nvector_cuda.h:realtype *N_VGetHostArrayPointer_Cuda(N_Vector x)
src/cvode-5.7.0/include/nvector/nvector_cuda.h:  N_VectorContent_Cuda content = (N_VectorContent_Cuda)x->content;
src/cvode-5.7.0/include/nvector/nvector_cuda.h:realtype *N_VGetDeviceArrayPointer_Cuda(N_Vector x)
src/cvode-5.7.0/include/nvector/nvector_cuda.h:  N_VectorContent_Cuda content = (N_VectorContent_Cuda)x->content;
src/cvode-5.7.0/include/nvector/nvector_cuda.h:N_Vector_ID N_VGetVectorID_Cuda(N_Vector v)
src/cvode-5.7.0/include/nvector/nvector_cuda.h:  return SUNDIALS_NVEC_CUDA;
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT N_Vector N_VCloneEmpty_Cuda(N_Vector w);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT N_Vector N_VClone_Cuda(N_Vector w);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT void N_VDestroy_Cuda(N_Vector v);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT void N_VSpace_Cuda(N_Vector v, sunindextype *lrw, sunindextype *liw);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT void N_VLinearSum_Cuda(realtype a, N_Vector x, realtype b, N_Vector y, N_Vector z);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT void N_VConst_Cuda(realtype c, N_Vector z);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT void N_VProd_Cuda(N_Vector x, N_Vector y, N_Vector z);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT void N_VDiv_Cuda(N_Vector x, N_Vector y, N_Vector z);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT void N_VScale_Cuda(realtype c, N_Vector x, N_Vector z);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT void N_VAbs_Cuda(N_Vector x, N_Vector z);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT void N_VInv_Cuda(N_Vector x, N_Vector z);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT void N_VAddConst_Cuda(N_Vector x, realtype b, N_Vector z);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT realtype N_VDotProd_Cuda(N_Vector x, N_Vector y);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT realtype N_VMaxNorm_Cuda(N_Vector x);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT realtype N_VWrmsNorm_Cuda(N_Vector x, N_Vector w);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT realtype N_VWrmsNormMask_Cuda(N_Vector x, N_Vector w, N_Vector id);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT realtype N_VMin_Cuda(N_Vector x);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT realtype N_VWL2Norm_Cuda(N_Vector x, N_Vector w);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT realtype N_VL1Norm_Cuda(N_Vector x);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT void N_VCompare_Cuda(realtype c, N_Vector x, N_Vector z);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT booleantype N_VInvTest_Cuda(N_Vector x, N_Vector z);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT booleantype N_VConstrMask_Cuda(N_Vector c, N_Vector x, N_Vector m);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT realtype N_VMinQuotient_Cuda(N_Vector num, N_Vector denom);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VLinearCombination_Cuda(int nvec, realtype* c, N_Vector* X,
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VScaleAddMulti_Cuda(int nvec, realtype* c, N_Vector X,
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VDotProdMulti_Cuda(int nvec, N_Vector x, N_Vector* Y,
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VLinearSumVectorArray_Cuda(int nvec,
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VScaleVectorArray_Cuda(int nvec, realtype* c, N_Vector* X,
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VConstVectorArray_Cuda(int nvec, realtype c, N_Vector* Z);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VScaleAddMultiVectorArray_Cuda(int nvec, int nsum,
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VLinearCombinationVectorArray_Cuda(int nvec, int nsum,
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VWrmsNormVectorArray_Cuda(int nvec, N_Vector* X,
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VWrmsNormMaskVectorArray_Cuda(int nvec, N_Vector* X,
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT realtype N_VWSqrSumLocal_Cuda(N_Vector x, N_Vector w);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT realtype N_VWSqrSumMaskLocal_Cuda(N_Vector x, N_Vector w, N_Vector id);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VBufSize_Cuda(N_Vector x, sunindextype *size);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VBufPack_Cuda(N_Vector x, void *buf);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VBufUnpack_Cuda(N_Vector x, void *buf);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT void N_VPrint_Cuda(N_Vector v);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT void N_VPrintFile_Cuda(N_Vector v, FILE *outfile);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VEnableFusedOps_Cuda(N_Vector v, booleantype tf);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VEnableLinearCombination_Cuda(N_Vector v, booleantype tf);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VEnableScaleAddMulti_Cuda(N_Vector v, booleantype tf);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VEnableDotProdMulti_Cuda(N_Vector v, booleantype tf);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VEnableLinearSumVectorArray_Cuda(N_Vector v, booleantype tf);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VEnableScaleVectorArray_Cuda(N_Vector v, booleantype tf);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VEnableConstVectorArray_Cuda(N_Vector v, booleantype tf);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VEnableWrmsNormVectorArray_Cuda(N_Vector v, booleantype tf);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VEnableWrmsNormMaskVectorArray_Cuda(N_Vector v, booleantype tf);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VEnableScaleAddMultiVectorArray_Cuda(N_Vector v, booleantype tf);
src/cvode-5.7.0/include/nvector/nvector_cuda.h:SUNDIALS_EXPORT int N_VEnableLinearCombinationVectorArray_Cuda(N_Vector v, booleantype tf);
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:# Setup the CUDA languge and CUDA libraries.
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:# Configure options needed prior to enabling the CUDA language
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:if(NOT CMAKE_CUDA_HOST_COMPILER)
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "NVCC host compiler")
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:# Configure the CUDA flags
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:  if(CMAKE_CUDA_ARCHITECTURES)
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:    foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${_nvcc_arch_flags}")
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-mno-float128")
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:# Need c++11 for the CUDA compiler check.
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:# Enable CUDA lang and find the CUDA libraries.
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:enable_language(CUDA)
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:set(CUDA_FOUND TRUE)
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:# Need this as long as CUDA libraries like cuSOLVER are not available
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:find_package(CUDA REQUIRED)
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:# Hide legacy FindCUDA variables
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:  if("${_var}" MATCHES "^CUDA_[A-z]+_LIBRARY")
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:  elseif("${_var}" MATCHES "^CUDA_.*")
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:# Make the CUDA_rt_LIBRARY advanced like the other CUDA_*_LIBRARY variables
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:mark_as_advanced(FORCE CUDA_rt_LIBRARY)
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:# Show CUDA flags
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:mark_as_advanced(CLEAR CMAKE_CUDA_FLAGS)
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:# We need c++11 for the CUDA compiler check, but if we don't remove it,
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:# then we will get a redefinition error. CMAKE_CUDA_STANDARD ends up
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:if(CMAKE_CUDA_FLAGS)
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:  STRING(REPLACE "-std=c++11" " " CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:# Print out information about CUDA.
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:message(STATUS "CUDA Version:               ${CUDA_VERSION_STRING}")
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:message(STATUS "CUDA Architectures:         ${CMAKE_CUDA_ARCHITECTURES}")
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:message(STATUS "CUDA Compiler:              ${CMAKE_CUDA_COMPILER}")
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:message(STATUS "CUDA Host Compiler:         ${CMAKE_CUDA_HOST_COMPILER}")
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:message(STATUS "CUDA Include Path:          ${CUDA_INCLUDE_DIRS}")
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:message(STATUS "CUDA Libraries:             ${CUDA_LIBRARIES}")
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:message(STATUS "CUDA Compile Flags:         ${CMAKE_CUDA_FLAGS}")
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:message(STATUS "CUDA Link Flags:            ${CMAKE_CUDA_LINK_FLAGS}")
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:message(STATUS "CUDA Link Executable:       ${CMAKE_CUDA_LINK_EXECUTABLE}")
src/cvode-5.7.0/cmake/SundialsSetupCuda.cmake:message(STATUS "CUDA Separable Compilation: ${CMAKE_CUDA_SEPARABLE_COMPILATION}")
src/cvode-5.7.0/cmake/SundialsSetupCompilers.cmake:# (b) CUDA is enabled
src/cvode-5.7.0/cmake/SundialsSetupCompilers.cmake:    ENABLE_CUDA OR
src/cvode-5.7.0/cmake/SundialsSetupCompilers.cmake:# CUDA settings
src/cvode-5.7.0/cmake/SundialsSetupCompilers.cmake:if(ENABLE_CUDA)
src/cvode-5.7.0/cmake/SundialsSetupCompilers.cmake:  include(SundialsSetupCuda)
src/cvode-5.7.0/cmake/SundialsSetupCompilers.cmake:  # we treat CUDA as both a TPL and a language
src/cvode-5.7.0/cmake/SundialsSetupCompilers.cmake:  list(APPEND SUNDIALS_TPL_LIST "CUDA")
src/cvode-5.7.0/cmake/SundialsSetupCompilers.cmake:if(CUDA_FOUND)
src/cvode-5.7.0/cmake/SundialsSetupCompilers.cmake:  list(APPEND _SUNDIALS_ENABLED_LANGS "CUDA")
src/cvode-5.7.0/cmake/SundialsDeprecated.cmake:if(DEFINED CUDA_ENABLE)
src/cvode-5.7.0/cmake/SundialsDeprecated.cmake:  print_warning("The CMake option CUDA_ENABLE is deprecated" "Use ENABLE_CUDA instead"
src/cvode-5.7.0/cmake/SundialsDeprecated.cmake:  set(ENABLE_CUDA ${CUDA_ENABLE} CACHE BOOL "Enable CUDA support" FORCE)
src/cvode-5.7.0/cmake/SundialsDeprecated.cmake:  unset(CUDA_ENABLE CACHE)
src/cvode-5.7.0/cmake/SundialsDeprecated.cmake:# Deprecated CUDA_ARCH option
src/cvode-5.7.0/cmake/SundialsDeprecated.cmake:if(DEFINED CUDA_ARCH)
src/cvode-5.7.0/cmake/SundialsDeprecated.cmake:  print_warning("The CMake option CUDA_ARCH is deprecated" "Use CMAKE_CUDA_ARCHITECTURES instead"
src/cvode-5.7.0/cmake/SundialsDeprecated.cmake:  string(REGEX MATCH "[0-9]+" arch_name "${CUDA_ARCH}")
src/cvode-5.7.0/cmake/SundialsDeprecated.cmake:  set(CMAKE_CUDA_ARCHITECTURES ${arch_name} CACHE STRING "CUDA Architectures" FORCE)
src/cvode-5.7.0/cmake/SundialsDeprecated.cmake:  unset(CUDA_ARCH)
src/cvode-5.7.0/cmake/SundialsTPLOptions.cmake:# Enable CUDA support?
src/cvode-5.7.0/cmake/SundialsTPLOptions.cmake:sundials_option(ENABLE_CUDA BOOL "Enable CUDA support" OFF)
src/cvode-5.7.0/cmake/SundialsTPLOptions.cmake:sundials_option(CMAKE_CUDA_ARCHITECTURES STRING "Target CUDA architecture" "70"
src/cvode-5.7.0/cmake/SundialsTPLOptions.cmake:                SHOW_IF ENABLE_CUDA)
src/cvode-5.7.0/cmake/SundialsTPLOptions.cmake:sundials_option(SUNDIALS_MAGMA_BACKENDS STRING "Which MAGMA backend under the SUNDIALS MAGMA interfaces (CUDA, HIP)" "CUDA"
src/cvode-5.7.0/cmake/SundialsTPLOptions.cmake:                OPTIONS "CUDA;HIP"
src/cvode-5.7.0/cmake/SundialsTPLOptions.cmake:sundials_option(SUNDIALS_RAJA_BACKENDS STRING "Which RAJA backend under the SUNDIALS RAJA interfaces (CUDA, HIP)" "CUDA"
src/cvode-5.7.0/cmake/SundialsTPLOptions.cmake:                OPTIONS "CUDA;HIP"
src/cvode-5.7.0/cmake/SundialsBuildOptionsPost.cmake:sundials_option(SUNDIALS_BUILD_PACKAGE_FUSED_KERNELS BOOL "Build specialized fused CUDA kernels" OFF
src/cvode-5.7.0/cmake/SundialsBuildOptionsPost.cmake:                DEPENDS_ON ENABLE_CUDA CMAKE_CUDA_COMPILER BUILD_CVODE
src/cvode-5.7.0/cmake/SundialsBuildOptionsPost.cmake:                SHOW_IF ENABLE_CUDA CMAKE_CUDA_COMPILER BUILD_CVODE)
src/cvode-5.7.0/cmake/SundialsBuildOptionsPost.cmake:sundials_option(BUILD_NVECTOR_CUDA BOOL "Build the NVECTOR_CUDA module (requires CUDA)" ON
src/cvode-5.7.0/cmake/SundialsBuildOptionsPost.cmake:                DEPENDS_ON ENABLE_CUDA CMAKE_CUDA_COMPILER
src/cvode-5.7.0/cmake/SundialsBuildOptionsPost.cmake:list(APPEND SUNDIALS_BUILD_LIST "BUILD_NVECTOR_CUDA")
src/cvode-5.7.0/cmake/SundialsBuildOptionsPost.cmake:sundials_option(BUILD_SUNMATRIX_CUSPARSE BOOL "Build the SUNMATRIX_CUSPARSE module (requires CUDA and 32-bit indexing)" ON
src/cvode-5.7.0/cmake/SundialsBuildOptionsPost.cmake:                DEPENDS_ON ENABLE_CUDA CMAKE_CUDA_COMPILER _COMPATIBLE_INDEX_SIZE BUILD_NVECTOR_CUDA
src/cvode-5.7.0/cmake/SundialsBuildOptionsPost.cmake:sundials_option(BUILD_SUNLINSOL_CUSOLVERSP BOOL "Build the SUNLINSOL_CUSOLVERSP module (requires CUDA and 32-bit indexing)" ON
src/cvode-5.7.0/cmake/SundialsBuildOptionsPost.cmake:                DEPENDS_ON ENABLE_CUDA CMAKE_CUDA_COMPILER BUILD_NVECTOR_CUDA BUILD_SUNMATRIX_CUSPARSE
src/cvode-5.7.0/cmake/SundialsSetupHIP.cmake:if(NOT DEFINED ROCM_PATH)
src/cvode-5.7.0/cmake/SundialsSetupHIP.cmake:  if(NOT DEFINED ENV{ROCM_PATH})
src/cvode-5.7.0/cmake/SundialsSetupHIP.cmake:    set(ROCM_PATH "/opt/rocm/" CACHE PATH "Path to which ROCm has been installed")
src/cvode-5.7.0/cmake/SundialsSetupHIP.cmake:    set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "Path to which ROCm has been installed")
src/cvode-5.7.0/cmake/SundialsSetupHIP.cmake:    set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
src/cvode-5.7.0/cmake/SundialsSetupHIP.cmake:set(CMAKE_PREFIX_PATH "${ROCM_PATH};${HIP_PATH}")
src/cvode-5.7.0/cmake/SundialsSetupHIP.cmake:  print_error("Deprecated HCC compiler is not supported" "Please update ROCm")
src/cvode-5.7.0/cmake/SundialsSetupHIP.cmake:message(STATUS "AMD targets:      ${AMDGPU_TARGETS}")
src/cvode-5.7.0/cmake/SundialsExampleOptions.cmake:# Options for CUDA Examples
src/cvode-5.7.0/cmake/SundialsExampleOptions.cmake:sundials_option(EXAMPLES_ENABLE_CUDA BOOL "Build SUNDIALS CUDA examples" ON
src/cvode-5.7.0/cmake/SundialsExampleOptions.cmake:                DEPENDS_ON ENABLE_CUDA
src/cvode-5.7.0/cmake/SundialsExampleOptions.cmake:                SHOW_IF ENABLE_CUDA)
src/cvode-5.7.0/cmake/SundialsExampleOptions.cmake:   EXAMPLES_ENABLE_CUDA OR
src/cvode-5.7.0/cmake/tpl/SundialsRAJA.cmake:foreach(_backend CUDA HIP OPENMP TARGET_OPENMP)
src/cvode-5.7.0/cmake/tpl/SundialsRAJA.cmake:if((SUNDIALS_RAJA_BACKENDS MATCHES "CUDA") AND
src/cvode-5.7.0/cmake/tpl/SundialsRAJA.cmake:   (NOT RAJA_BACKENDS MATCHES "CUDA"))
src/cvode-5.7.0/cmake/tpl/SundialsRAJA.cmake:  print_error("Requested that SUNDIALS uses the CUDA RAJA backend, but RAJA was not built with the CUDA backend.")
src/cvode-5.7.0/cmake/tpl/SundialsMAGMA.cmake:if(SUNDIALS_MAGMA_BACKENDS MATCHES "CUDA" AND NOT ENABLE_CUDA)
src/cvode-5.7.0/cmake/tpl/SundialsMAGMA.cmake:  print_error("SUNDIALS_MAGMA_BACKENDS includes CUDA but CUDA is not enabled. Set ENABLE_CUDA=ON or change the backend.")
src/cvode-5.7.0/cmake/tpl/SundialsMAGMA.cmake:  elseif(SUNDIALS_MAGMA_BACKENDS MATCHES "CUDA")
src/cvode-5.7.0/cmake/tpl/SundialsMAGMA.cmake:    set(lang CUDA)
src/cvode-5.7.0/examples/cvode/hip/cvAdvDiff_kry_hip.cpp: * Acknowledgements: This example is based on cvAdvDiff_kry_cuda.cu.
src/cvode-5.7.0/examples/cvode/CMakeLists.txt:# CUDA examples
src/cvode-5.7.0/examples/cvode/CMakeLists.txt:if(EXAMPLES_ENABLE_CUDA)
src/cvode-5.7.0/examples/cvode/CMakeLists.txt:  if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
src/cvode-5.7.0/examples/cvode/CMakeLists.txt:    add_subdirectory(cuda)
src/cvode-5.7.0/examples/cvode/raja/CMakeLists.txt:  set_source_files_properties(${example}.cpp PROPERTIES LANGUAGE CUDA)
src/cvode-5.7.0/examples/cvode/raja/cvAdvDiff_kry_raja.cpp: * an executable that runs on a GPU device.
src/cvode-5.7.0/examples/cvode/raja/cvAdvDiff_kry_raja.cpp:  RAJA::forall<RAJA::cuda_exec<256> >(RAJA::RangeSegment(zero, NEQ),
src/cvode-5.7.0/examples/cvode/raja/cvAdvDiff_kry_raja.cpp:  RAJA::forall<RAJA::cuda_exec<256> >(RAJA::RangeSegment(zero, NEQ),
src/cvode-5.7.0/examples/cvode/raja/README:  cvAdvDiff_kry_cuda       : 2-D advection-diffusion (nonstiff)
src/cvode-5.7.0/examples/cvode/C_openmpdev/cvAdvDiff_kry_ompdev.c:  if(check_flag((void*)u, "N_VNew_Cuda", 0)) return(1);
src/cvode-5.7.0/examples/cvode/C_openmpdev/README:  LLNL LC's wrapped xlc compiler for gpu program compilation: xlc-gpu
src/cvode-5.7.0/examples/cvode/C_openmpdev/README:-DCMAKE_C_COMPILER=xlc-gpu \
src/cvode-5.7.0/examples/cvode/C_openmpdev/README:  C Compiler: xlc-gpu
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:#include <nvector/nvector_cuda.h>                     /* access to cuda N_Vector                       */
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  /* Create CUDA vector of length neq for I.C. and abstol */
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  y = N_VNew_Cuda(neq);
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  if (check_retval((void *)y, "N_VNew_Cuda", 0)) return(1);
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  abstol = N_VNew_Cuda(neq);
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  if (check_retval((void *)abstol, "N_VNew_Cuda", 0)) return(1);
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  ydata = N_VGetHostArrayPointer_Cuda(y);
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  abstol_data = N_VGetHostArrayPointer_Cuda(abstol);
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  N_VCopyToDevice_Cuda(y);
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  N_VCopyToDevice_Cuda(abstol);
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:    N_VCopyFromDevice_Cuda(y);
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:/* Right hand side function. This just launches the CUDA kernel
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  ydata = N_VGetDeviceArrayPointer_Cuda(y);
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  ydotdata = N_VGetDeviceArrayPointer_Cuda(ydot);
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  cudaError_t cuerr = cudaGetLastError();
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  if (cuerr != cudaSuccess) {
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:            ">>> ERROR in f: cudaGetLastError returned %s\n",
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:            cudaGetErrorName(cuerr));
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu: * This is done on the GPU.
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  ydata   = N_VGetDeviceArrayPointer_Cuda(y);
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  cudaError_t cuerr = cudaGetLastError();
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:  if (cuerr != cudaSuccess) {
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:            ">>> ERROR in Jac: cudaGetLastError returned %s\n",
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:            cudaGetErrorName(cuerr));
src/cvode-5.7.0/examples/cvode/cuda/cvRoberts_block_cusolversp_batchqr.cu:/* Jacobian evaluation GPU kernel */
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:#include <cuda_runtime.h>
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:#include <nvector/nvector_cuda.h>
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu: * CUDA kernels
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  cudaStream_t stream;
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  cudaError_t cuerr;
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  /* optional: create a cudaStream to use with the CUDA NVector
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  cuerr = cudaStreamCreate(&stream);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  if (cuerr != cudaSuccess) {
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:    printf("Error in cudaStreamCreate(): %s\n", cudaGetErrorString(cuerr));
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  SUNCudaThreadDirectExecPolicy stream_exec_policy(256, stream);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  SUNCudaBlockReduceExecPolicy reduce_exec_policy(256, 0, stream);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  /* Create a CUDA nvector with initial values using managed
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  u = N_VNewManaged_Cuda(data->NEQ);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  if(check_retval((void*)u, "N_VNewManaged_Cuda", 0)) return(1);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  /* Use a non-default cuda stream for kernel execution */
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  retval = N_VSetKernelExecPolicy_Cuda(u, &stream_exec_policy, &reduce_exec_policy);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  if(check_retval(&retval, "N_VSetKernelExecPolicy_Cuda", 0)) return(1);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  cuerr = cudaStreamDestroy(stream); /* Free and cleanup the CUDA stream */
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  if(cuerr != cudaSuccess) { printf("Error: cudaStreamDestroy() failed\n"); return(1); }
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  realtype *udata = N_VGetHostArrayPointer_Cuda(u);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  const realtype *udata = N_VGetDeviceArrayPointer_Cuda(u);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  realtype *dudata      = N_VGetDeviceArrayPointer_Cuda(udot);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  const realtype *vdata = N_VGetDeviceArrayPointer_Cuda(v);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda_managed.cu:  realtype *Jvdata      = N_VGetDeviceArrayPointer_Cuda(Jv);
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:# CMakeLists.txt file for CVODE CUDA examples
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:  "cvAdvDiff_kry_cuda\;\;develop"
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:  "cvAdvDiff_kry_cuda_managed\;\;develop"
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:  "cvAdvDiff_diag_cuda\;0 0\;develop"
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:  "cvAdvDiff_diag_cuda\;0 1\;develop"
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:  "cvAdvDiff_diag_cuda\;1 1\;develop"
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:                  sundials_nveccuda)
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:  set(SUNDIALS_LIBS ${SUNDIALS_LIBS} sundials_cvode_fused_cuda)
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:  set_source_files_properties(${example}.cu PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:      DESTINATION ${EXAMPLES_INSTALL_PATH}/cvode/cuda)
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:  install(FILES README DESTINATION ${EXAMPLES_INSTALL_PATH}/cvode/cuda)
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:  set(NVECTOR_LIB "sundials_nveccuda")
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:    set(LIBS "-lsundials_cvode_fused_cuda ${LIBS}")
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:    ${PROJECT_SOURCE_DIR}/examples/templates/cmakelists_serial_CUDA_ex.in
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:    ${PROJECT_BINARY_DIR}/examples/cvode/cuda/CMakeLists.txt
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:    FILES ${PROJECT_BINARY_DIR}/examples/cvode/cuda/CMakeLists.txt
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:    DESTINATION ${EXAMPLES_INSTALL_PATH}/cvode/cuda
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:      ${PROJECT_SOURCE_DIR}/examples/templates/makefile_serial_CUDA_ex.in
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:      ${PROJECT_BINARY_DIR}/examples/cvode/cuda/Makefile_ex
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:      FILES ${PROJECT_BINARY_DIR}/examples/cvode/cuda/Makefile_ex
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:      DESTINATION ${EXAMPLES_INSTALL_PATH}/cvode/cuda
src/cvode-5.7.0/examples/cvode/cuda/CMakeLists.txt:  sundials_add_test_install(cvode cuda)
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu: * ./cvAdvDiff_diag_cuda [0 (scalar atol) | 1 (vector atol)]
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu:#include <cuda_runtime.h>
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu:#include <nvector/nvector_cuda.h>         /* access to cuda N_Vector              */
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu:  u = N_VNew_Cuda(NEQ);  /* Allocate u vector */
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu:    N_Vector vabstol = N_VClone_Cuda(u);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu:    if (check_retval(&vabstol, "N_VClone_Cuda", 0)) return(1);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu:  udata = N_VGetHostArrayPointer_Cuda(u);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu:  N_VCopyToDevice_Cuda(u);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu:  cudaError_t cuerr;
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu:  udata = N_VGetDeviceArrayPointer_Cuda(u);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu:  dudata = N_VGetDeviceArrayPointer_Cuda(udot);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu:  cuerr = cudaGetLastError();
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu:  if (cuerr != cudaSuccess) {
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_diag_cuda.cu:    fprintf(stderr, "ERROR in f: f_kernel --> %s\n", cudaGetErrorString(cuerr));
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:#include <cuda_runtime.h>
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:#include <nvector/nvector_cuda.h>
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu: * CUDA kernels
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  cudaError_t cuerr;
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  /* optional: create a cudaStream to use with the CUDA NVector
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  cuerr = cudaStreamCreate(&stream);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  if (cuerr != cudaSuccess) {
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:    printf("Error in cudaStreamCreate(): %s\n", cudaGetErrorString(cuerr));
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  SUNCudaThreadDirectExecPolicy stream_exec_policy(256, stream);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  SUNCudaBlockReduceExecPolicy reduce_exec_policy(256, 0, stream);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  /* Create a CUDA vector with initial values */
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  u = N_VNew_Cuda(data->NEQ);  /* Allocate u vector */
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  if(check_retval((void*)u, "N_VNew_Cuda", 0)) return(1);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  /* Use a non-default cuda stream for kernel execution */
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  retval = N_VSetKernelExecPolicy_Cuda(u, &stream_exec_policy, &reduce_exec_policy);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  if(check_retval(&retval, "N_VSetKernelExecPolicy_Cuda", 0)) return(1);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  cuerr = cudaStreamDestroy(stream); /* Free and cleanup the CUDA stream */
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  if(cuerr != cudaSuccess) { printf("Error: cudaStreamDestroy() failed\n"); return(1); }
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  realtype *udata = N_VGetHostArrayPointer_Cuda(u);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  N_VCopyToDevice_Cuda(u);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  const realtype *udata = N_VGetDeviceArrayPointer_Cuda(u);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  realtype *dudata      = N_VGetDeviceArrayPointer_Cuda(udot);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  const realtype *vdata = N_VGetDeviceArrayPointer_Cuda(v);
src/cvode-5.7.0/examples/cvode/cuda/cvAdvDiff_kry_cuda.cu:  realtype *Jvdata      = N_VGetDeviceArrayPointer_Cuda(Jv);
src/cvode-5.7.0/examples/cvode/cuda/README:List of CUDA CVODE examples
src/cvode-5.7.0/examples/cvode/cuda/README:  cvAdvDiff_kry_cuda                 : 2-D advection-diffusion (nonstiff)
src/cvode-5.7.0/examples/cvode/magma/CMakeLists.txt:if(SUNDIALS_MAGMA_BACKENDS MATCHES "CUDA")
src/cvode-5.7.0/examples/cvode/magma/CMakeLists.txt:  set_source_files_properties(cvRoberts_blockdiag_magma.cpp PROPERTIES LANGUAGE CUDA)
src/cvode-5.7.0/examples/cvode/magma/CMakeLists.txt:  set(vector nveccuda)
src/cvode-5.7.0/examples/cvode/magma/CMakeLists.txt:  set(cuda_or_hip CUDA)
src/cvode-5.7.0/examples/cvode/magma/CMakeLists.txt:  set(cuda_or_hip HIP)
src/cvode-5.7.0/examples/cvode/magma/CMakeLists.txt:      cmakelists_${cuda_or_hip}_ex.in
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:#define HIP_OR_CUDA(a,b) a
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:#elif defined(SUNDIALS_MAGMA_BACKENDS_CUDA)
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:#define HIP_OR_CUDA(a,b) b
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:#define HIP_OR_CUDA(a,b) ((void)0);
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:#if defined(SUNDIALS_MAGMA_BACKENDS_CUDA)
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:#include <nvector/nvector_cuda.h>
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:#include <sunmemory/sunmemory_cuda.h>
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:  SUNMemoryHelper memhelper = HIP_OR_CUDA( SUNMemoryHelper_Hip();,
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:                                           SUNMemoryHelper_Cuda(); )
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:  /* Create CUDA or HIP vector of length neq for I.C. and abstol */
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:  y = HIP_OR_CUDA( N_VNew_Hip(neq);,
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:                   N_VNew_Cuda(neq); )
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:  HIP_OR_CUDA( N_VCopyToDevice_Hip(y);,
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:               N_VCopyToDevice_Cuda(y); )
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:  HIP_OR_CUDA( N_VCopyToDevice_Hip(abstol);,
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:               N_VCopyToDevice_Cuda(abstol); )
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:    HIP_OR_CUDA( N_VCopyFromDevice_Hip(y);,
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:                 N_VCopyFromDevice_Cuda(y); )
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:/* Right hand side function. This just launches the CUDA or HIP kernel
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:  unsigned block_size = HIP_OR_CUDA( 64, 32 );
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:  HIP_OR_CUDA( hipDeviceSynchronize();, cudaDeviceSynchronize(); )
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:  HIP_OR_CUDA( hipError_t cuerr = hipGetLastError();,
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:               cudaError_t cuerr = cudaGetLastError(); )
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:  if (cuerr != HIP_OR_CUDA( hipSuccess, cudaSuccess )) {
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:            ">>> ERROR in f: cudaGetLastError returned %s\n",
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:            HIP_OR_CUDA( hipGetErrorName(cuerr), cudaGetErrorName(cuerr) ));
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp: * This is done on the GPU.
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:  block_size = HIP_OR_CUDA( 64, 32 );
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:  HIP_OR_CUDA( hipDeviceSynchronize();, cudaDeviceSynchronize(); )
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:  HIP_OR_CUDA( hipError_t cuerr = hipGetLastError();,
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:               cudaError_t cuerr = cudaGetLastError(); )
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:  if (cuerr != HIP_OR_CUDA( hipSuccess, cudaSuccess )) {
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:            ">>> ERROR in Jac: cudaGetLastError returned %s\n",
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:            HIP_OR_CUDA( hipGetErrorName(cuerr), cudaGetErrorName(cuerr) ));
src/cvode-5.7.0/examples/cvode/magma/cvRoberts_blockdiag_magma.cpp:/* Jacobian evaluation GPU kernel */
src/cvode-5.7.0/examples/cvode/serial/cvRoberts_block_klu.c: * The problem is comparable to the CUDA version -
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:#include <cuda_runtime.h>
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:#include <nvector/nvector_cuda.h>
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:class ATestExecPolicy : public SUNCudaExecPolicy
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:  virtual const cudaStream_t* stream() const
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:  virtual SUNCudaExecPolicy* clone() const
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:    return static_cast<SUNCudaExecPolicy*>(new ATestExecPolicy());
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:  const cudaStream_t stream_;
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:   d_x = N_VNew_Cuda(N);
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:   d_y = N_VNew_Cuda(M);
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:      printf("ERROR: N_VNew_Cuda returned NULL\n");
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:   x = N_VMake_Serial(N, N_VGetHostArrayPointer_Cuda(d_x));
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:   y = N_VMake_Serial(M, N_VGetHostArrayPointer_Cuda(d_y));
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:   N_VCopyToDevice_Cuda(d_x);
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:   N_VCopyToDevice_Cuda(d_y);
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:     N_VCopyFromDevice_Cuda(d_x);
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:     N_VCopyFromDevice_Cuda(d_y);
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:     N_VPrint_Cuda(d_x);
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:     N_VPrint_Cuda(d_y);
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:   cudaDeviceSynchronize();
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:   cudaDeviceSynchronize();
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:   xdata = N_VGetHostArrayPointer_Cuda(expected);
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:   ydata = N_VGetHostArrayPointer_Cuda(computed);
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:   N_VCopyFromDevice_Cuda(expected);
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:   N_VCopyFromDevice_Cuda(computed);
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:   cudaDeviceSynchronize();
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:   xldata = N_VGetLength_Cuda(expected);
src/cvode-5.7.0/examples/sunmatrix/cusparse/test_sunmatrix_cusparse.cu:   yldata = N_VGetLength_Cuda(computed);
src/cvode-5.7.0/examples/sunmatrix/cusparse/CMakeLists.txt:# Examples using SUNDIALS cuda nvector
src/cvode-5.7.0/examples/sunmatrix/cusparse/CMakeLists.txt:                  sundials_nveccuda
src/cvode-5.7.0/examples/sunmatrix/cusparse/CMakeLists.txt:  set(NVECTOR_LIB " sundials_nvecserial sundials_nveccuda")
src/cvode-5.7.0/examples/sunmatrix/cusparse/CMakeLists.txt:    ${PROJECT_SOURCE_DIR}/examples/templates/cmakelists_serial_CUDA_ex.in
src/cvode-5.7.0/examples/sunmatrix/cusparse/CMakeLists.txt:      ${PROJECT_SOURCE_DIR}/examples/templates/makefile_serial_CUDA_ex.in
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:#define HIP_OR_CUDA(a,b) a
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:#elif defined(SUNDIALS_MAGMA_BACKENDS_CUDA)
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:#define HIP_OR_CUDA(a,b) b
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:#define HIP_OR_CUDA(a,b) ((void)0);
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:#if defined(SUNDIALS_MAGMA_BACKENDS_CUDA)
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:#include <nvector/nvector_cuda.h>
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:#include <sunmemory/sunmemory_cuda.h>
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:  SUNMemoryHelper memhelper = HIP_OR_CUDA( SUNMemoryHelper_Hip();,
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:                                           SUNMemoryHelper_Cuda(); )
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:  x = HIP_OR_CUDA( N_VNew_Hip(matcols*nblocks);,
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:                   N_VNew_Cuda(matcols*nblocks); )
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:  y = HIP_OR_CUDA( N_VNew_Hip(matrows*nblocks);,
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:                   N_VNew_Cuda(matrows*nblocks); )
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:  HIP_OR_CUDA(  N_VCopyToDevice_Hip(x);,
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:                N_VCopyToDevice_Cuda(x); )
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:  HIP_OR_CUDA(  N_VCopyToDevice_Hip(y);,
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:                N_VCopyToDevice_Cuda(y); )
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:  HIP_OR_CUDA( N_VCopyFromDevice_Hip(actual);,
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:               N_VCopyFromDevice_Cuda(actual); )
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:  HIP_OR_CUDA( N_VCopyFromDevice_Hip(expected);,
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:               N_VCopyFromDevice_Cuda(expected); )
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:  HIP_OR_CUDA( hipDeviceSynchronize();,
src/cvode-5.7.0/examples/sunmatrix/magmadense/test_sunmatrix_magmadense.cpp:               cudaDeviceSynchronize(); )
src/cvode-5.7.0/examples/sunmatrix/magmadense/CMakeLists.txt:if(SUNDIALS_MAGMA_BACKENDS MATCHES "CUDA")
src/cvode-5.7.0/examples/sunmatrix/magmadense/CMakeLists.txt:  set_source_files_properties(test_sunmatrix_magmadense.cpp PROPERTIES LANGUAGE CUDA)
src/cvode-5.7.0/examples/sunmatrix/magmadense/CMakeLists.txt:  set(vector nveccuda)
src/cvode-5.7.0/examples/sunmatrix/magmadense/CMakeLists.txt:  set(cuda_or_hip CUDA)
src/cvode-5.7.0/examples/sunmatrix/magmadense/CMakeLists.txt:  set(cuda_or_hip HIP)
src/cvode-5.7.0/examples/sunmatrix/magmadense/CMakeLists.txt:      cmakelists_${cuda_or_hip}_ex.in
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:#include <nvector/nvector_cuda.h>
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:  d_x = N_VNew_Cuda(N);
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:  d_xref = N_VNew_Cuda(N);
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:  d_b = N_VNew_Cuda(N);
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:  x = N_VMake_Serial(N, N_VGetHostArrayPointer_Cuda(d_x));
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:  b = N_VMake_Serial(N, N_VGetHostArrayPointer_Cuda(d_b));
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:  xdata = N_VGetHostArrayPointer_Cuda(d_x);
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:  xrefdata = N_VGetHostArrayPointer_Cuda(d_xref);
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:  N_VCopyToDevice_Cuda(d_x);
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:  N_VCopyToDevice_Cuda(d_xref);
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:  N_VCopyToDevice_Cuda(d_b);
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:    N_VCopyFromDevice_Cuda(d_xref);
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:    N_VPrint_Cuda(d_xref);
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:    N_VCopyFromDevice_Cuda(d_x); /* copy solution from device */
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:    N_VPrint_Cuda(d_x);
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:    N_VCopyFromDevice_Cuda(d_b);
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:    N_VPrint_Cuda(d_b);
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:  N_VCopyFromDevice_Cuda(X);
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:  N_VCopyFromDevice_Cuda(Y);
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:  Xdata = N_VGetHostArrayPointer_Cuda(X);
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:  Ydata = N_VGetHostArrayPointer_Cuda(Y);
src/cvode-5.7.0/examples/sunlinsol/cusolversp/test_sunlinsol_cusolversp_batchqr.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/examples/sunlinsol/cusolversp/CMakeLists.txt:# Examples using SUNDIALS cuda nvector
src/cvode-5.7.0/examples/sunlinsol/cusolversp/CMakeLists.txt:set(SUNDIALS_LIBS sundials_nveccuda
src/cvode-5.7.0/examples/sunlinsol/cusolversp/CMakeLists.txt:  set(NVECTOR_LIB "sundials_nvecserial sundials_nveccuda")
src/cvode-5.7.0/examples/sunlinsol/cusolversp/CMakeLists.txt:    ${PROJECT_SOURCE_DIR}/examples/templates/cmakelists_serial_CUDA_ex.in
src/cvode-5.7.0/examples/sunlinsol/cusolversp/CMakeLists.txt:      ${PROJECT_SOURCE_DIR}/examples/templates/makefile_serial_CUDA_ex.in
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:#define HIP_OR_CUDA(a,b) a
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:#elif defined(SUNDIALS_MAGMA_BACKENDS_CUDA)
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:#define HIP_OR_CUDA(a,b) b
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:#define HIP_OR_CUDA(a,b) ((void)0);
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:#if defined(SUNDIALS_MAGMA_BACKENDS_CUDA)
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:#include <nvector/nvector_cuda.h>
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:#include <sunmemory/sunmemory_cuda.h>
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:  SUNMemoryHelper memhelper = HIP_OR_CUDA( SUNMemoryHelper_Hip();,
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:                                           SUNMemoryHelper_Cuda(); )
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:  x = HIP_OR_CUDA( N_VNew_Hip(cols*nblocks);,
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:                   N_VNew_Cuda(cols*nblocks); )
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:  HIP_OR_CUDA( N_VCopyToDevice_Hip(x);,
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:               N_VCopyToDevice_Cuda(x); )
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:  HIP_OR_CUDA( N_VCopyFromDevice_Hip(X);,
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:               N_VCopyFromDevice_Cuda(X); )
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:  HIP_OR_CUDA( N_VCopyFromDevice_Hip(Y);,
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:               N_VCopyFromDevice_Cuda(Y); )
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:  HIP_OR_CUDA( hipDeviceSynchronize();,
src/cvode-5.7.0/examples/sunlinsol/magmadense/test_sunlinsol_magmadense.cpp:               cudaDeviceSynchronize(); )
src/cvode-5.7.0/examples/sunlinsol/magmadense/CMakeLists.txt:if(SUNDIALS_MAGMA_BACKENDS MATCHES "CUDA")
src/cvode-5.7.0/examples/sunlinsol/magmadense/CMakeLists.txt:  set_source_files_properties(test_sunlinsol_magmadense.cpp PROPERTIES LANGUAGE CUDA)
src/cvode-5.7.0/examples/sunlinsol/magmadense/CMakeLists.txt:  set(vector nveccuda)
src/cvode-5.7.0/examples/sunlinsol/magmadense/CMakeLists.txt:  set(cuda_or_hip CUDA)
src/cvode-5.7.0/examples/sunlinsol/magmadense/CMakeLists.txt:  set(cuda_or_hip HIP)
src/cvode-5.7.0/examples/sunlinsol/magmadense/CMakeLists.txt:      cmakelists_${cuda_or_hip}_ex.in
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h: * Example of a custom SUNMemoryHelper that only supports CUDA/HIP
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h:#include <cuda_runtime.h>
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h:#define EX_USES_CUDA
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h:#if defined(EX_USES_CUDA)
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h:#define MY_GPU(a) cuda ## a
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h:#define MY_GPU(a) hip ## a
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h:#define MY_GPUCHK(ans) { gpuVerify((ans), __FILE__, __LINE__, 1); }
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h:static void gpuVerify(MY_GPU(Error_t) code, const char *file, int line, int abort)
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h:   if (code != MY_GPU(Success))
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h:      fprintf(stderr, "GPU ERROR: %s %s %d\n", MY_GPU(GetErrorString)(code), file, line);
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h:    MY_GPUCHK( MY_GPU(Malloc)(&(mem->ptr), memsize) );
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h:        MY_GPUCHK( MY_GPU(Free)(mem->ptr) );
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h:        MY_GPUCHK( MY_GPU(Memcpy)(dst->ptr, src->ptr,
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h:                                  MY_GPU(MemcpyHostToDevice)) );
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h:        MY_GPUCHK( MY_GPU(Memcpy)(dst->ptr, src->ptr,
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h:                                  MY_GPU(MemcpyDeviceToHost)) );
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h:        MY_GPUCHK( MY_GPU(Memcpy)(dst->ptr, src->ptr,
src/cvode-5.7.0/examples/utilities/custom_memory_helper.h:                                  MY_GPU(MemcpyDeviceToDevice)) );
src/cvode-5.7.0/examples/templates/cmakelists_CUDA_ex.in:set(CMAKE_CUDA_COMPILER
src/cvode-5.7.0/examples/templates/cmakelists_CUDA_ex.in:  @CMAKE_CUDA_COMPILER@
src/cvode-5.7.0/examples/templates/cmakelists_CUDA_ex.in:  CACHE FILEPATH "CUDA compiler")
src/cvode-5.7.0/examples/templates/cmakelists_CUDA_ex.in:set(CMAKE_CUDA_FLAGS
src/cvode-5.7.0/examples/templates/cmakelists_CUDA_ex.in:  "@CMAKE_CUDA_FLAGS@"
src/cvode-5.7.0/examples/templates/cmakelists_CUDA_ex.in:  CACHE STRING "CUDA compiler flags")
src/cvode-5.7.0/examples/templates/cmakelists_CUDA_ex.in:set(CMAKE_CUDA_STANDARD 11)
src/cvode-5.7.0/examples/templates/cmakelists_CUDA_ex.in:set(CMAKE_CUDA_HOST_COMPILER
src/cvode-5.7.0/examples/templates/cmakelists_CUDA_ex.in:  CACHE FILEPATH "CUDA host compiler")
src/cvode-5.7.0/examples/templates/cmakelists_CUDA_ex.in:  set_source_files_properties(${example} PROPERTIES LANGUAGE CUDA)
src/cvode-5.7.0/examples/templates/makefile_parallel_RAJA_ex.in:NVCC        = @CMAKE_CUDA_COMPILER@
src/cvode-5.7.0/examples/templates/makefile_parallel_RAJA_ex.in:NVCCFLAGS   = $(subst ;, ,@CMAKE_CUDA_FLAGS@)
src/cvode-5.7.0/examples/templates/makefile_parallel_RAJA_ex.in:TMP_SUNDIALSLIBS = @SOLVER_LIB@ sundials_nvecparallel sundials_nveccudaraja \
src/cvode-5.7.0/examples/templates/makefile_parallel_RAJA_ex.in:LIBRARIES = ${SUNDIALSLIBS} ${RAJALIBS} ${CUDALIBS} ${LIBS}
src/cvode-5.7.0/examples/templates/makefile_serial_RAJA_ex.in:NVCC        = @CMAKE_CUDA_COMPILER@
src/cvode-5.7.0/examples/templates/makefile_serial_RAJA_ex.in:NVCCFLAGS   = $(subst ;, ,@CMAKE_CUDA_FLAGS@)
src/cvode-5.7.0/examples/templates/cmakelists_serial_RAJA_ex.in:set(CMAKE_CUDA_COMPILER
src/cvode-5.7.0/examples/templates/cmakelists_serial_RAJA_ex.in:  @CMAKE_CUDA_COMPILER@
src/cvode-5.7.0/examples/templates/cmakelists_serial_RAJA_ex.in:  CACHE FILEPATH "CUDA compiler")
src/cvode-5.7.0/examples/templates/cmakelists_serial_RAJA_ex.in:set(CMAKE_CUDA_HOST_COMPILER
src/cvode-5.7.0/examples/templates/cmakelists_serial_RAJA_ex.in:  CACHE FILEPATH "CUDA host compiler")
src/cvode-5.7.0/examples/templates/cmakelists_serial_RAJA_ex.in:project(@SOLVER@_raja_examples C CXX CUDA)
src/cvode-5.7.0/examples/templates/cmakelists_serial_RAJA_ex.in:  sundials_nveccudaraja ${SUNDIALS_LIBRARY_DIR}
src/cvode-5.7.0/examples/templates/cmakelists_serial_RAJA_ex.in:  set_source_files_properties(${example}.cpp PROPERTIES LANGUAGE CUDA)
src/cvode-5.7.0/examples/templates/cmakelists_serial_RAJA_ex.in:    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
src/cvode-5.7.0/examples/templates/cmakelists_serial_CUDA_ex.in:# CMakeLists.txt for @SOLVER@ CUDA examples.
src/cvode-5.7.0/examples/templates/cmakelists_serial_CUDA_ex.in:set(CMAKE_CUDA_COMPILER
src/cvode-5.7.0/examples/templates/cmakelists_serial_CUDA_ex.in:  @CMAKE_CUDA_COMPILER@
src/cvode-5.7.0/examples/templates/cmakelists_serial_CUDA_ex.in:  CACHE FILEPATH "CUDA compiler")
src/cvode-5.7.0/examples/templates/cmakelists_serial_CUDA_ex.in:set(CMAKE_CUDA_HOST_COMPILER
src/cvode-5.7.0/examples/templates/cmakelists_serial_CUDA_ex.in:  CACHE FILEPATH "CUDA host compiler")
src/cvode-5.7.0/examples/templates/cmakelists_serial_CUDA_ex.in:project(@SOLVER@_cuda_examples C CXX CUDA)
src/cvode-5.7.0/examples/templates/cmakelists_serial_CUDA_ex.in:# Need this as long as CUDA libraries like cuSOLVER are not
src/cvode-5.7.0/examples/templates/cmakelists_serial_CUDA_ex.in:find_package(CUDA REQUIRED)
src/cvode-5.7.0/examples/templates/cmakelists_serial_CUDA_ex.in:  sundials_nveccuda ${SUNDIALS_LIBRARY_DIR}
src/cvode-5.7.0/examples/templates/cmakelists_serial_CUDA_ex.in:  DOC "NVECTOR_CUDA library")
src/cvode-5.7.0/examples/templates/cmakelists_serial_CUDA_ex.in:  target_link_libraries(${example} ${CUDA_cusolver_LIBRARY})
src/cvode-5.7.0/examples/templates/cmakelists_serial_CUDA_ex.in:  target_link_libraries(${example} ${CUDA_cusparse_LIBRARY})
src/cvode-5.7.0/examples/templates/cmakelists_parallel_RAJA_ex.in:set(CMAKE_CUDA_COMPILER
src/cvode-5.7.0/examples/templates/cmakelists_parallel_RAJA_ex.in:  @CMAKE_CUDA_COMPILER@
src/cvode-5.7.0/examples/templates/cmakelists_parallel_RAJA_ex.in:  CACHE FILEPATH "CUDA compiler")
src/cvode-5.7.0/examples/templates/cmakelists_parallel_RAJA_ex.in:set(CMAKE_CUDA_HOST_COMPILER
src/cvode-5.7.0/examples/templates/cmakelists_parallel_RAJA_ex.in:  CACHE FILEPATH "CUDA host compiler")
src/cvode-5.7.0/examples/templates/cmakelists_parallel_RAJA_ex.in:project(@SOLVER@_mpi_raja_examples C CXX CUDA)
src/cvode-5.7.0/examples/templates/cmakelists_parallel_RAJA_ex.in:  sundials_nveccudaraja ${SUNDIALS_LIBRARY_DIR}
src/cvode-5.7.0/examples/templates/cmakelists_parallel_RAJA_ex.in:  set_source_files_properties(${example}.cpp PROPERTIES LANGUAGE CUDA)
src/cvode-5.7.0/examples/templates/cmakelists_parallel_RAJA_ex.in:    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
src/cvode-5.7.0/examples/templates/makefile_serial_CUDA_ex.in:# Makefile for @SOLVER@ CUDA examples
src/cvode-5.7.0/examples/templates/makefile_serial_CUDA_ex.in:NVCC        = @CMAKE_CUDA_COMPILER@
src/cvode-5.7.0/examples/templates/makefile_serial_CUDA_ex.in:NVCCFLAGS   = -ccbin=${CXX} -std=c++11 @CMAKE_CUDA_FLAGS@
src/cvode-5.7.0/examples/templates/cmakelists_MPI_ex.in:set(CMAKE_CUDA_COMPILER
src/cvode-5.7.0/examples/templates/cmakelists_MPI_ex.in:  @CMAKE_CUDA_COMPILER@
src/cvode-5.7.0/examples/templates/cmakelists_MPI_ex.in:  CACHE FILEPATH "CUDA compiler")
src/cvode-5.7.0/examples/templates/cmakelists_MPI_ex.in:set(CMAKE_CUDA_FLAGS
src/cvode-5.7.0/examples/templates/cmakelists_MPI_ex.in:  "@CMAKE_CUDA_FLAGS@"
src/cvode-5.7.0/examples/templates/cmakelists_MPI_ex.in:  CACHE STRING "CUDA compiler flags")
src/cvode-5.7.0/examples/templates/cmakelists_MPI_ex.in:set(CMAKE_CUDA_STANDARD 11)
src/cvode-5.7.0/examples/templates/cmakelists_MPI_ex.in:set(CMAKE_CUDA_HOST_COMPILER
src/cvode-5.7.0/examples/templates/cmakelists_MPI_ex.in:  CACHE FILEPATH "CUDA host compiler")
src/cvode-5.7.0/examples/templates/cmakelists_parallel_CUDA_ex.in:# CMakeLists.txt for @SOLVER@ MPI+CUDA examples.
src/cvode-5.7.0/examples/templates/cmakelists_parallel_CUDA_ex.in:set(CMAKE_CUDA_COMPILER
src/cvode-5.7.0/examples/templates/cmakelists_parallel_CUDA_ex.in:  @CMAKE_CUDA_COMPILER@
src/cvode-5.7.0/examples/templates/cmakelists_parallel_CUDA_ex.in:  CACHE FILEPATH "CUDA compiler")
src/cvode-5.7.0/examples/templates/cmakelists_parallel_CUDA_ex.in:set(CMAKE_CUDA_HOST_COMPILER
src/cvode-5.7.0/examples/templates/cmakelists_parallel_CUDA_ex.in:  CACHE FILEPATH "CUDA host compiler")
src/cvode-5.7.0/examples/templates/cmakelists_parallel_CUDA_ex.in:project(@SOLVER@_mpi_cuda_examples C CXX CUDA)
src/cvode-5.7.0/examples/templates/cmakelists_parallel_CUDA_ex.in:  sundials_nveccuda ${SUNDIALS_LIBRARY_DIR}
src/cvode-5.7.0/examples/templates/cmakelists_parallel_CUDA_ex.in:  DOC "NVECTOR_CUDA library")
src/cvode-5.7.0/examples/templates/makefile_parallel_CUDA_ex.in:# Makefile for @SOLVER@ CUDA examples
src/cvode-5.7.0/examples/templates/makefile_parallel_CUDA_ex.in:NVCC        = @CMAKE_CUDA_COMPILER@
src/cvode-5.7.0/examples/templates/makefile_parallel_CUDA_ex.in:NVCCFLAGS   = $(subst ;, ,@CMAKE_CUDA_FLAGS@)
src/cvode-5.7.0/examples/templates/makefile_parallel_CUDA_ex.in:TMP_SUNDIALSLIBS = @SOLVER_LIB@ sundials_nvecparallel sundials_nveccuda sundials_nvecserial \
src/cvode-5.7.0/examples/nvector/sycl/test_nvector_sycl.cpp:  /* Create an in-order GPU queue */
src/cvode-5.7.0/examples/nvector/sycl/test_nvector_sycl.cpp:  sycl::gpu_selector selector;
src/cvode-5.7.0/examples/nvector/sycl/test_nvector_sycl.cpp:  std::cout << " is gpu? "
src/cvode-5.7.0/examples/nvector/sycl/test_nvector_sycl.cpp:            << (dev.is_gpu() ? "Yes" : "No")
src/cvode-5.7.0/examples/nvector/mpiplusx/test_nvector_mpiplusx.c:  /* not running on GPU, just return */
src/cvode-5.7.0/examples/nvector/mpiraja/test_nvector_mpiraja.cpp:  /* sync with GPU */
src/cvode-5.7.0/examples/nvector/mpiraja/test_nvector_mpiraja.cpp:  cudaDeviceSynchronize();
src/cvode-5.7.0/examples/nvector/mpiraja/CMakeLists.txt:set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER})
src/cvode-5.7.0/examples/nvector/mpiraja/CMakeLists.txt:    set_source_files_properties(${example}.cpp PROPERTIES LANGUAGE CUDA)
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:# CMakeLists.txt file for MPIPlusX, X = CUDA NVECTOR examples.
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:# Examples using SUNDIALS MPI+cuda nvector
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:set(nvector_cuda_examples
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:  "test_nvector_mpicuda\;1000 0\;\;\;"    # run sequentially
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:  "test_nvector_mpicuda\;1000 0\;1\;4\;"  # run parallel on 4
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:set(CMAKE_CUDA_HOST_COMPILER ${MPI_CXX_COMPILER})
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:set(NVECS_LIB sundials_nvecmpiplusx sundials_nveccuda)
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:foreach(example_tuple ${nvector_cuda_examples})
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:      DESTINATION ${EXAMPLES_INSTALL_PATH}/nvector/mpicuda)
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:endforeach(example_tuple ${nvector_cuda_examples})
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:  install(FILES DESTINATION ${EXAMPLES_INSTALL_PATH}/nvector/mpicuda)
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:  examples2string(nvector_cuda_examples EXAMPLES)
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:    ${PROJECT_SOURCE_DIR}/examples/templates/cmakelists_parallel_CUDA_ex.in
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:    ${PROJECT_BINARY_DIR}/examples/nvector/mpicuda/CMakeLists.txt
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:    FILES ${PROJECT_BINARY_DIR}/examples/nvector/mpicuda/CMakeLists.txt
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:    DESTINATION ${EXAMPLES_INSTALL_PATH}/nvector/mpicuda
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:      ${PROJECT_SOURCE_DIR}/examples/templates/makefile_parallel_CUDA_ex.in
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:      ${PROJECT_BINARY_DIR}/examples/nvector/mpicuda/Makefile_ex
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:      FILES ${PROJECT_BINARY_DIR}/examples/nvector/mpicuda/Makefile_ex
src/cvode-5.7.0/examples/nvector/mpicuda/CMakeLists.txt:      DESTINATION ${EXAMPLES_INSTALL_PATH}/nvector/mpicuda
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu: * the X is the CUDA NVECTOR.
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:#include <nvector/nvector_cuda.h>
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:/* CUDA vector can use unmanaged or managed memory */
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:        printf("Testing CUDA N_Vector \n");
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:        printf("\nTesting CUDA N_Vector with managed memory \n");
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:    X = (i==UNMANAGED) ? N_VNew_Cuda(local_length) : N_VNewManaged_Cuda(local_length);
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:      if (myid == 0) printf("FAIL: Unable to create a new CUDA vector \n\n");
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:    U = (i==UNMANAGED) ? N_VNew_Cuda(local_length) : N_VNewManaged_Cuda(local_length);
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:    retval = N_VEnableFusedOps_Cuda(U, SUNFALSE);
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:      if (myid == 0) printf("FAIL: Unable to create a new CUDA vector \n\n");
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:    V = (i==UNMANAGED) ? N_VNew_Cuda(local_length) : N_VNewManaged_Cuda(local_length);
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:    retval = N_VEnableFusedOps_Cuda(V, SUNTRUE);
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:      if (myid == 0) printf("FAIL: Unable to create a new CUDA vector \n\n");
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:  N_VCopyFromDevice_Cuda(X);
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:  Xdata = N_VGetHostArrayPointer_Cuda(X);
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:  if ((N_VGetHostArrayPointer_Cuda(X) == NULL) &&
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:      (N_VGetDeviceArrayPointer_Cuda(X) == NULL))
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:  N_VCopyFromDevice_Cuda(X);
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:  xd = N_VGetHostArrayPointer_Cuda(X);
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:  N_VCopyToDevice_Cuda(X);
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:  N_VCopyFromDevice_Cuda(X);
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:  return (N_VGetHostArrayPointer_Cuda(X))[i];
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:  /* sync with GPU */
src/cvode-5.7.0/examples/nvector/mpicuda/test_nvector_mpicuda.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/examples/nvector/hip/test_nvector_hip.hip.cpp:  /* sync with GPU */
src/cvode-5.7.0/examples/nvector/parhyp/test_nvector_parhyp.c:  /* not running on GPU, just return */
src/cvode-5.7.0/examples/nvector/CMakeLists.txt:if(BUILD_NVECTOR_CUDA)
src/cvode-5.7.0/examples/nvector/CMakeLists.txt:  add_subdirectory(cuda)
src/cvode-5.7.0/examples/nvector/CMakeLists.txt:    add_subdirectory(mpicuda)
src/cvode-5.7.0/examples/nvector/petsc/test_nvector_petsc.c:  /* not running on GPU, just return */
src/cvode-5.7.0/examples/nvector/parallel/test_nvector_mpi.c:  /* not running on GPU, just return */
src/cvode-5.7.0/examples/nvector/C_openmp/test_nvector_openmp.c:  /* not running on GPU, just return */
src/cvode-5.7.0/examples/nvector/openmpdev/test_nvector_openmpdev.c: * Test for the CUDA N_Vector N_VMake_OpenMPDEV function. Requires N_VConst
src/cvode-5.7.0/examples/nvector/manyvector/test_nvector_manyvector.c:  /* not running on GPU, just return */
src/cvode-5.7.0/examples/nvector/pthreads/test_nvector_pthreads.c:  /* not running on GPU, just return */
src/cvode-5.7.0/examples/nvector/raja/test_nvector_raja.cpp:#if defined(SUNDIALS_RAJA_BACKENDS_CUDA)
src/cvode-5.7.0/examples/nvector/raja/test_nvector_raja.cpp:  cudaDeviceReset();
src/cvode-5.7.0/examples/nvector/raja/test_nvector_raja.cpp:  /* sync with GPU */
src/cvode-5.7.0/examples/nvector/raja/test_nvector_raja.cpp:  #if defined(SUNDIALS_RAJA_BACKENDS_CUDA)
src/cvode-5.7.0/examples/nvector/raja/test_nvector_raja.cpp:    cudaDeviceSynchronize();
src/cvode-5.7.0/examples/nvector/raja/CMakeLists.txt:if(SUNDIALS_RAJA_BACKENDS MATCHES "CUDA")
src/cvode-5.7.0/examples/nvector/raja/CMakeLists.txt:  set(_lang CUDA)
src/cvode-5.7.0/examples/nvector/raja/CMakeLists.txt:  set(NVECTOR_LIB "sundials_nveccudaraja")
src/cvode-5.7.0/examples/nvector/mpimanyvector/test_nvector_mpimanyvector_parallel1.c:  /* not running on GPU, just return */
src/cvode-5.7.0/examples/nvector/mpimanyvector/test_nvector_mpimanyvector_serial.c:  /* not running on GPU, just return */
src/cvode-5.7.0/examples/nvector/mpimanyvector/test_nvector_mpimanyvector_parallel2.c:  /* not running on GPU, just return */
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu: * This is the testing routine to check the NVECTOR CUDA module
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:#include <nvector/nvector_cuda.h>
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:/* CUDA vector variants */
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  int             threadsPerBlock;   /* cuda block size            */
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  cudaStream_t    stream;            /* cuda stream                */
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:    printf("ERROR: THREE (3) Inputs required: vector length, CUDA threads per block (0 for default), print timing \n");
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:    printf("ERROR: CUDA threads per block must be 0 to use the default or a multiple of 32\n");
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:    SUNCudaExecPolicy* stream_exec_policy = NULL;
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:    SUNCudaExecPolicy* reduce_exec_policy = NULL;
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:    cudaStreamCreate(&stream);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:      stream_exec_policy = new SUNCudaThreadDirectExecPolicy(actualThreadsPerBlock, stream);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:      reduce_exec_policy = new SUNCudaBlockReduceExecPolicy(actualThreadsPerBlock, 0, stream);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:      stream_exec_policy = new SUNCudaGridStrideExecPolicy(actualThreadsPerBlock, 1);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:      reduce_exec_policy = new SUNCudaBlockReduceExecPolicy(actualThreadsPerBlock, 1);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:        printf("Testing CUDA N_Vector, policy %d\n", policy);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:        printf("Testing CUDA N_Vector with managed memory, policy %d\n", policy);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:        printf("Testing CUDA N_Vector with user allocator, policy %d\n", policy);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:        printf("Testing CUDA N_Vector with SUNMemoryHelper, policy %d\n", policy);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:        X = N_VNew_Cuda(length);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:        X = N_VNewManaged_Cuda(length);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:        X = N_VMakeWithManagedAllocator_Cuda(length, sunalloc, sunfree);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:        X = N_VNewWithMemHelp_Cuda(length, SUNFALSE, mem_helper);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:        if (N_VSetKernelExecPolicy_Cuda(X, stream_exec_policy, reduce_exec_policy)) {
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:      realtype* xdata = N_VGetHostArrayPointer_Cuda(X);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:      N_VCopyToDevice_Cuda(X);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:      realtype* ydata = N_VGetHostArrayPointer_Cuda(Y);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:      realtype* zdata = N_VGetHostArrayPointer_Cuda(Z);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:      N_VCopyToDevice_Cuda(Y);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:      N_VCopyToDevice_Cuda(Z);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:      fails += Test_N_VGetVectorID(X, SUNDIALS_NVEC_CUDA, 0);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:      retval = N_VEnableFusedOps_Cuda(U, SUNFALSE);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:      retval = N_VEnableFusedOps_Cuda(V, SUNTRUE);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:      cudaDeviceSynchronize();
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:    cudaStreamDestroy(stream);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  cudaDeviceReset();
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  N_VCopyFromDevice_Cuda(X);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  Xdata = N_VGetHostArrayPointer_Cuda(X);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  if ((N_VGetHostArrayPointer_Cuda(X) == NULL) &&
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:      (N_VGetDeviceArrayPointer_Cuda(X) == NULL))
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  N_VCopyFromDevice_Cuda(X);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  xd = N_VGetHostArrayPointer_Cuda(X);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  N_VCopyToDevice_Cuda(X);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  N_VCopyFromDevice_Cuda(X);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  return (N_VGetHostArrayPointer_Cuda(X))[i];
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  /* sync with GPU */
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  cudaError_t err;
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  err = cudaMallocManaged(&ptr, mem_size);
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  if (err != cudaSuccess) {
src/cvode-5.7.0/examples/nvector/cuda/test_nvector_cuda.cu:  cudaFree(ptr);
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:# CMakeLists.txt file for CUDA nvector examples
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:# Examples using SUNDIALS cuda nvector
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:set(nvector_cuda_examples
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:  "test_nvector_cuda\;3 32 0\;\;\;"
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:  "test_nvector_cuda\;500 128 0\;\;\;"
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:  "test_nvector_cuda\;1000 0 0\;\;\;"
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:set(NVECS_LIB sundials_nveccuda sundials_nvecserial)
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:foreach(example_tuple ${nvector_cuda_examples})
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:      DESTINATION ${EXAMPLES_INSTALL_PATH}/nvector/cuda)
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:endforeach(example_tuple ${nvector_cuda_examples})
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:  install(FILES DESTINATION ${EXAMPLES_INSTALL_PATH}/nvector/cuda)
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:  set(NVECTOR_LIB "sundials_nveccuda")
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:  examples2string(nvector_cuda_examples EXAMPLES)
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:    ${PROJECT_SOURCE_DIR}/examples/templates/cmakelists_serial_CUDA_ex.in
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:    ${PROJECT_BINARY_DIR}/examples/nvector/cuda/CMakeLists.txt
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:    FILES ${PROJECT_BINARY_DIR}/examples/nvector/cuda/CMakeLists.txt
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:    DESTINATION ${EXAMPLES_INSTALL_PATH}/nvector/cuda
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:      ${PROJECT_SOURCE_DIR}/examples/templates/makefile_serial_CUDA_ex.in
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:      ${PROJECT_BINARY_DIR}/examples/nvector/cuda/Makefile_ex
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:      FILES ${PROJECT_BINARY_DIR}/examples/nvector/cuda/Makefile_ex
src/cvode-5.7.0/examples/nvector/cuda/CMakeLists.txt:      DESTINATION ${EXAMPLES_INSTALL_PATH}/nvector/cuda
src/cvode-5.7.0/examples/nvector/serial/test_nvector_serial.c:  /* not running on GPU, just return */
src/cvode-5.7.0/src/cvode/README.md:accelerator-based (e.g., GPU) systems. This flexibility is obtained from a
src/cvode-5.7.0/src/cvode/CMakeLists.txt:  sundials_add_library(sundials_cvode_fused_cuda
src/cvode-5.7.0/src/cvode/CMakeLists.txt:      cvode_fused_cuda.cu
src/cvode-5.7.0/src/cvode/CMakeLists.txt:      PRIVATE sundials_nveccuda
src/cvode-5.7.0/src/cvode/CMakeLists.txt:      sundials_cvode_fused_cuda
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu: * This file implements fused CUDA kernels for CVODE.
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu: #include <cuda_runtime.h>
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu: #include <nvector/nvector_cuda.h>
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu: #include "sundials_cuda_kernels.cuh"
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  const SUNCudaExecPolicy* exec_policy = ((N_VectorContent_Cuda)weight->content)->stream_exec_policy;
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(ycur),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(tempv),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(weight)
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:#ifdef SUNDIALS_DEBUG_CUDA_LASTERROR
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(cudaGetLastError())) return -1;
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  const SUNCudaExecPolicy* exec_policy = ((N_VectorContent_Cuda)weight->content)->stream_exec_policy;
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(Vabstol),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(ycur),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(tempv),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(weight)
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:#ifdef SUNDIALS_DEBUG_CUDA_LASTERROR
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(cudaGetLastError())) return -1;
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  const SUNCudaExecPolicy* exec_policy = ((N_VectorContent_Cuda)c->content)->stream_exec_policy;
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(c),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(ewt),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(y),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(mm),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(tempv)
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:#ifdef SUNDIALS_DEBUG_CUDA_LASTERROR
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(cudaGetLastError())) return -1;
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  const SUNCudaExecPolicy* exec_policy = ((N_VectorContent_Cuda)res->content)->stream_exec_policy;
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(zn1),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(ycor),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(ftemp),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(res)
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:#ifdef SUNDIALS_DEBUG_CUDA_LASTERROR
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(cudaGetLastError())) return -1;
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  const SUNCudaExecPolicy* exec_policy = ((N_VectorContent_Cuda)y->content)->stream_exec_policy;
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(fpred),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(zn1),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(ypred),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(ftemp),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(y)
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:#ifdef SUNDIALS_DEBUG_CUDA_LASTERROR
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(cudaGetLastError())) return -1;
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  const SUNCudaExecPolicy* exec_policy = ((N_VectorContent_Cuda)M->content)->stream_exec_policy;
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(ftemp),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(fpred),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(ewt),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(bit),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(bitcomp),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(y),
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(M)
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:#ifdef SUNDIALS_DEBUG_CUDA_LASTERROR
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(cudaGetLastError())) return -1;
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  const SUNCudaExecPolicy* exec_policy = ((N_VectorContent_Cuda)M->content)->stream_exec_policy;
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:    N_VGetDeviceArrayPointer_Cuda(M)
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:#ifdef SUNDIALS_DEBUG_CUDA_LASTERROR
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/src/cvode/cvode_fused_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(cudaGetLastError())) return -1;
src/cvode-5.7.0/src/cvode/cvode_io.c:      N_VGetVectorID(cv_mem->cv_ewt) != SUNDIALS_NVEC_CUDA) {
src/cvode-5.7.0/src/sunmatrix/cusparse/CMakeLists.txt:# CMakeLists.txt file for the CUDA cuSPARSE SUNMatrix
src/cvode-5.7.0/src/sunmatrix/cusparse/CMakeLists.txt:    sundials_sunmemcuda_obj
src/cvode-5.7.0/src/sunmatrix/cusparse/CMakeLists.txt:    PUBLIC ${CUDA_cusparse_LIBRARY}
src/cvode-5.7.0/src/sunmatrix/cusparse/CMakeLists.txt:    PRIVATE ${CUDA_cusolver_LIBRARY}
src/cvode-5.7.0/src/sunmatrix/cusparse/cusparse_kernels.cuh:#include <cuda_runtime.h>
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:#include <sunmemory/sunmemory_cuda.h>
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:#include "sundials_cuda.h"
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:#if CUDART_VERSION >= 11000
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:#define CUDA_R_XF CUDA_R_64F
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:#define CUDA_R_XF CUDA_R_32F
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:class SUNCuSparseMatrixExecPolicy : public SUNCudaExecPolicy
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  SUNCuSparseMatrixExecPolicy(const cudaStream_t stream = 0)
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:    return(max_block_size(CUDA_WARP_SIZE*(numWorkElements + CUDA_WARP_SIZE - 1)/CUDA_WARP_SIZE));
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  virtual const cudaStream_t* stream() const
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  virtual CudaExecPolicy* clone() const
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:    return(static_cast<CudaExecPolicy*>(new SUNCuSparseMatrixExecPolicy(*this)));
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:    return((val > MAX_CUDA_BLOCKSIZE) ? MAX_CUDA_BLOCKSIZE : val );
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  const cudaStream_t stream_;
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  SMCU_MEMHELP(A) = SUNMemoryHelper_Cuda();
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:    SUNDIALS_DEBUG_PRINT("ERROR in SUNMatrix_NewCSR_cuSparse: SUNMemoryHelper_Cuda returned NULL\n");
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:#if CUDART_VERSION >= 11000
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  SMCU_MEMHELP(A) = SUNMemoryHelper_Cuda();
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:    SUNDIALS_DEBUG_PRINT("ERROR in SUNMatrix_NewCSR_cuSparse: SUNMemoryHelper_Cuda returned NULL\n");
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:#if CUDART_VERSION >= 11000
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  SMCU_MEMHELP(A) = SUNMemoryHelper_Cuda();
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:    SUNDIALS_DEBUG_PRINT("ERROR in SUNMatrix_NewCSR_cuSparse: SUNMemoryHelper_Cuda returned NULL\n");
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:#if CUDART_VERSION >= 11000
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:int SUNMatrix_cuSparse_SetKernelExecPolicy(SUNMatrix A, SUNCudaExecPolicy* exec_policy)
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  const cudaStream_t* stream;
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  const cudaStream_t* stream;
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:#if CUDART_VERSION >= 11000
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  cudaError_t cuerr;
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  cuerr = cudaMemsetAsync(SMCU_DATAp(A), 0, SMCU_NNZ(A)*sizeof(realtype), stream);
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  if (!SUNDIALS_CUDA_VERIFY(cuerr)) return(SUNMAT_OPERATION_FAIL);
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:    cuerr = cudaMemsetAsync(SMCU_INDEXPTRSp(A), 0,
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:    if (!SUNDIALS_CUDA_VERIFY(cuerr)) return(SUNMAT_OPERATION_FAIL);
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:    cuerr = cudaMemsetAsync(SMCU_INDEXVALSp(A), 0,
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:    if (!SUNDIALS_CUDA_VERIFY(cuerr)) return(SUNMAT_OPERATION_FAIL);
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  const cudaStream_t* stream;
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  cudaStream_t stream = *SMCU_EXECPOLICY(A)->stream();
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:#ifdef SUNDIALS_DEBUG_CUDA_LASTERROR
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  if (!SUNDIALS_CUDA_VERIFY(cudaGetLastError())) return(SUNMAT_OPERATION_FAIL);
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:#ifdef SUNDIALS_DEBUG_CUDA_LASTERROR
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  if (!SUNDIALS_CUDA_VERIFY(cudaGetLastError())) return(SUNMAT_OPERATION_FAIL);
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:#if CUDART_VERSION >= 11000
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:                                                  placeholder, CUDA_R_XF) );
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:                                                  placeholder, CUDA_R_XF) );
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:                              CUDA_R_XF, CUSPARSE_MV_ALG_DEFAULT,
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:#if CUDART_VERSION >= 11000
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:                                             SMCU_CONTENT(A)->vecY, CUDA_R_XF,
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:    cudaStream_t stream;
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:#ifdef SUNDIALS_DEBUG_CUDA_LASTERROR
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:    cudaDeviceSynchronize();
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:    if (!SUNDIALS_CUDA_VERIFY(cudaGetLastError())) return(SUNMAT_OPERATION_FAIL);
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:#if CUDART_VERSION >= 11000
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:  /* CUDA 11 introduced the "Generic API" and removed the cusparseXcsrmv that
src/cvode-5.7.0/src/sunmatrix/cusparse/sunmatrix_cusparse.cu:                           CUSPARSE_INDEX_BASE_ZERO, CUDA_R_XF));
src/cvode-5.7.0/src/sunmatrix/magmadense/CMakeLists.txt:if(SUNDIALS_MAGMA_BACKENDS MATCHES "CUDA")
src/cvode-5.7.0/src/sunmatrix/magmadense/CMakeLists.txt:  set_source_files_properties(sunmatrix_magmadense.cpp PROPERTIES LANGUAGE CUDA)
src/cvode-5.7.0/src/sunmatrix/magmadense/CMakeLists.txt:  set(_libs_needed sundials_nveccuda ${CUDA_CUBLAS_LIBRARIES})
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:#if defined(SUNDIALS_MAGMA_BACKENDS_CUDA)
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:#include "sundials_cuda.h"
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:#include "dense_cuda_kernels.cuh"
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:using namespace sundials::sunmatrix_gpudense::cuda;
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:#define SUNDIALS_HIP_OR_CUDA(a,b) b
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:using namespace sundials::sunmatrix_gpudense::hip;
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:#define SUNDIALS_HIP_OR_CUDA(a,b) a
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:#define xprint(q,...) magma_dprint_gpu(__VA_ARGS__, q)
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:#define xprint(q,...) magma_sprint_gpu(__VA_ARGS__, q)
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:  SUNDIALS_HIP_OR_CUDA(
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:    magma_queue_create_from_cuda(A->device_id, (cudaStream_t) queue, NULL, NULL, &A->q); )
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:  SUNDIALS_HIP_OR_CUDA( hipStream_t stream = magma_queue_get_hip_stream(A->q);,
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:                        cudaStream_t stream = magma_queue_get_cuda_stream(A->q); )
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:  SUNDIALS_HIP_OR_CUDA( hipStream_t stream = magma_queue_get_hip_stream(A->q);,
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:                        cudaStream_t stream = magma_queue_get_cuda_stream(A->q); )
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:  SUNDIALS_HIP_OR_CUDA( hipStream_t stream = magma_queue_get_hip_stream(A->q);,
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:                        cudaStream_t stream = magma_queue_get_cuda_stream(A->q); )
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:    SUNDIALS_HIP_OR_CUDA( dim3(1,16,16), dim3(1,16,32) ), /* We choose slightly larger thread blocks when using HIP since the warps are larger */
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:    SUNDIALS_HIP_OR_CUDA( magma_queue_get_hip_stream(A->q), magma_queue_get_cuda_stream(A->q) ),
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:    SUNDIALS_HIP_OR_CUDA( dim3(1,16,16), dim3(1,16,32) ),
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:    SUNDIALS_HIP_OR_CUDA( magma_queue_get_hip_stream(A->q), magma_queue_get_cuda_stream(A->q) ),
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:    SUNDIALS_HIP_OR_CUDA( dim3(1,16,16), dim3(1,16,32) ),
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:    SUNDIALS_HIP_OR_CUDA( magma_queue_get_hip_stream(A->q), magma_queue_get_cuda_stream(A->q) ),
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:    SUNDIALS_HIP_OR_CUDA( dim3(1,16,16), dim3(1,16,32) ),
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:    SUNDIALS_HIP_OR_CUDA( magma_queue_get_hip_stream(A->q), magma_queue_get_cuda_stream(A->q) ),
src/cvode-5.7.0/src/sunmatrix/magmadense/sunmatrix_magmadense.cpp:      SUNDIALS_HIP_OR_CUDA( magma_queue_get_hip_stream(A->q), magma_queue_get_cuda_stream(A->q) ),
src/cvode-5.7.0/src/sunmatrix/magmadense/dense_hip_kernels.hip.hpp:#ifndef _SUNGPUDENSE_MATRIX_KERNELS_HIP
src/cvode-5.7.0/src/sunmatrix/magmadense/dense_hip_kernels.hip.hpp:#define _SUNGPUDENSE_MATRIX_KERNELS_HIP
src/cvode-5.7.0/src/sunmatrix/magmadense/dense_hip_kernels.hip.hpp:namespace sunmatrix_gpudense
src/cvode-5.7.0/src/sunmatrix/magmadense/dense_hip_kernels.hip.hpp:} // namespace cuda
src/cvode-5.7.0/src/sunmatrix/magmadense/dense_hip_kernels.hip.hpp:} // namespace sunmatrix_gpudense
src/cvode-5.7.0/src/sunmatrix/magmadense/dense_cuda_kernels.cuh: * This is the implementation file for the dense matrix CUDA kernels
src/cvode-5.7.0/src/sunmatrix/magmadense/dense_cuda_kernels.cuh:#ifndef _SUNGPUDENSE_MATRIX_KERNELS_CUH_
src/cvode-5.7.0/src/sunmatrix/magmadense/dense_cuda_kernels.cuh:#define _SUNGPUDENSE_MATRIX_KERNELS_CUH_
src/cvode-5.7.0/src/sunmatrix/magmadense/dense_cuda_kernels.cuh:#include <cuda_runtime.h>
src/cvode-5.7.0/src/sunmatrix/magmadense/dense_cuda_kernels.cuh:namespace sunmatrix_gpudense
src/cvode-5.7.0/src/sunmatrix/magmadense/dense_cuda_kernels.cuh:namespace cuda
src/cvode-5.7.0/src/sunmatrix/magmadense/dense_cuda_kernels.cuh:} // namespace cuda
src/cvode-5.7.0/src/sunmatrix/magmadense/dense_cuda_kernels.cuh:} // namespace sunmatrix_gpudense
src/cvode-5.7.0/src/sunmemory/CMakeLists.txt:if(ENABLE_CUDA)
src/cvode-5.7.0/src/sunmemory/CMakeLists.txt:  add_subdirectory(cuda)
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu: * SUNDIALS CUDA memory helper implementation.
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:#include <sunmemory/sunmemory_cuda.h>
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:#include "sundials_cuda.h"
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:SUNMemoryHelper SUNMemoryHelper_Cuda()
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:  helper->ops->alloc     = SUNMemoryHelper_Alloc_Cuda;
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:  helper->ops->dealloc   = SUNMemoryHelper_Dealloc_Cuda;
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:  helper->ops->copy      = SUNMemoryHelper_Copy_Cuda;
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:  helper->ops->copyasync = SUNMemoryHelper_CopyAsync_Cuda;
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:int SUNMemoryHelper_Alloc_Cuda(SUNMemoryHelper helper, SUNMemory* memptr,
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:      SUNDIALS_DEBUG_PRINT("ERROR in SUNMemoryHelper_Alloc_Cuda: malloc returned NULL\n");
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:    if (!SUNDIALS_CUDA_VERIFY(cudaMallocHost(&(mem->ptr), mem_size)))
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:      SUNDIALS_DEBUG_PRINT("ERROR in SUNMemoryHelper_Alloc_Cuda: cudaMallocHost failed\n");
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:    if (!SUNDIALS_CUDA_VERIFY(cudaMalloc(&(mem->ptr), mem_size)))
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:      SUNDIALS_DEBUG_PRINT("ERROR in SUNMemoryHelper_Alloc_Cuda: cudaMalloc failed\n");
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:    if (!SUNDIALS_CUDA_VERIFY(cudaMallocManaged(&(mem->ptr), mem_size)))
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:      SUNDIALS_DEBUG_PRINT("ERROR in SUNMemoryHelper_Alloc_Cuda: cudaMallocManaged failed\n");
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:    SUNDIALS_DEBUG_PRINT("ERROR in SUNMemoryHelper_Alloc_Cuda: unknown memory type\n");
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:int SUNMemoryHelper_Dealloc_Cuda(SUNMemoryHelper helper, SUNMemory mem)
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:      if (!SUNDIALS_CUDA_VERIFY(cudaFreeHost(mem->ptr)))
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:        SUNDIALS_DEBUG_PRINT("ERROR in SUNMemoryHelper_Dealloc_Cuda: cudaFreeHost failed\n");
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:      if (!SUNDIALS_CUDA_VERIFY(cudaFree(mem->ptr)))
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:        SUNDIALS_DEBUG_PRINT("ERROR in SUNMemoryHelper_Dealloc_Cuda: cudaFree failed\n");
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:      SUNDIALS_DEBUG_PRINT("ERROR in SUNMemoryHelper_Dealloc_Cuda: unknown memory type\n");
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:int SUNMemoryHelper_Copy_Cuda(SUNMemoryHelper helper, SUNMemory dst,
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:  cudaError_t cuerr = cudaSuccess;
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:        cuerr = cudaMemcpy(dst->ptr, src->ptr,
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:                           cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:      if (!SUNDIALS_CUDA_VERIFY(cuerr)) retval = -1;
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:        cuerr = cudaMemcpy(dst->ptr, src->ptr,
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:                           cudaMemcpyDeviceToHost);
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:        cuerr = cudaMemcpy(dst->ptr, src->ptr,
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:                           cudaMemcpyDeviceToDevice);
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:      if (!SUNDIALS_CUDA_VERIFY(cuerr)) retval = -1;
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:      SUNDIALS_DEBUG_PRINT("ERROR in SUNMemoryHelper_CopyAsync_Cuda: unknown memory type\n");
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:int SUNMemoryHelper_CopyAsync_Cuda(SUNMemoryHelper helper, SUNMemory dst,
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:  cudaError_t cuerr = cudaSuccess;
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:  cudaStream_t stream = 0;
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:    stream = *((cudaStream_t*) ctx);
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:        cuerr = cudaMemcpyAsync(dst->ptr, src->ptr,
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:                                cudaMemcpyHostToDevice,
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:      if (!SUNDIALS_CUDA_VERIFY(cuerr)) retval = -1;
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:        cuerr = cudaMemcpyAsync(dst->ptr, src->ptr,
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:                                cudaMemcpyDeviceToHost,
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:        cuerr = cudaMemcpyAsync(dst->ptr, src->ptr,
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:                                cudaMemcpyDeviceToDevice,
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:      if (!SUNDIALS_CUDA_VERIFY(cuerr)) retval = -1;
src/cvode-5.7.0/src/sunmemory/cuda/sundials_cuda_memory.cu:      SUNDIALS_DEBUG_PRINT("ERROR in SUNMemoryHelper_CopyAsync_Cuda: unknown memory type\n");
src/cvode-5.7.0/src/sunmemory/cuda/CMakeLists.txt:sundials_add_library(sundials_sunmemcuda
src/cvode-5.7.0/src/sunmemory/cuda/CMakeLists.txt:    sundials_cuda_memory.cu
src/cvode-5.7.0/src/sunmemory/cuda/CMakeLists.txt:    ${SUNDIALS_SOURCE_DIR}/include/sunmemory/sunmemory_cuda.h
src/cvode-5.7.0/src/sunlinsol/cusolversp/sunlinsol_cusolversp_batchqr.cu:#include "sundials_cuda.h"
src/cvode-5.7.0/src/sunlinsol/cusolversp/sunlinsol_cusolversp_batchqr.cu:  cudaError_t cuerr;
src/cvode-5.7.0/src/sunlinsol/cusolversp/sunlinsol_cusolversp_batchqr.cu:      cudaFree(SUN_CUSP_QRWORKSPACE(S));
src/cvode-5.7.0/src/sunlinsol/cusolversp/sunlinsol_cusolversp_batchqr.cu:    cuerr = cudaMalloc((void**) &SUN_CUSP_QRWORKSPACE(S), SUN_CUSP_WORK_SIZE(S));
src/cvode-5.7.0/src/sunlinsol/cusolversp/sunlinsol_cusolversp_batchqr.cu:    if (!SUNDIALS_CUDA_VERIFY(cuerr))
src/cvode-5.7.0/src/sunlinsol/cusolversp/sunlinsol_cusolversp_batchqr.cu:  cudaFree(SUN_CUSP_QRWORKSPACE(S));
src/cvode-5.7.0/src/sunlinsol/cusolversp/CMakeLists.txt:# CMakeLists.txt file for the cuda cuSolverSp SUNLinearSolver
src/cvode-5.7.0/src/sunlinsol/cusolversp/CMakeLists.txt:    PUBLIC sundials_sunmatrixcusparse ${CUDA_cusolver_LIBRARY}
src/cvode-5.7.0/src/sunlinsol/cusolversp/CMakeLists.txt:    PRIVATE ${CUDA_cusparse_LIBRARY}
src/cvode-5.7.0/src/sunlinsol/magmadense/sunlinsol_magmadense.cpp:#define xgetrf magma_dgetrf_gpu
src/cvode-5.7.0/src/sunlinsol/magmadense/sunlinsol_magmadense.cpp:#define xgetrs magma_dgetrs_gpu
src/cvode-5.7.0/src/sunlinsol/magmadense/sunlinsol_magmadense.cpp:#define xgetrf magma_sgetrf_gpu
src/cvode-5.7.0/src/sunlinsol/magmadense/sunlinsol_magmadense.cpp:#define xgetrs magma_sgetrs_gpu
src/cvode-5.7.0/src/sunlinsol/magmadense/CMakeLists.txt:if(SUNDIALS_MAGMA_BACKENDS MATCHES "CUDA")
src/cvode-5.7.0/src/sunlinsol/magmadense/CMakeLists.txt:  set_source_files_properties(sunlinsol_magmadense.cpp PROPERTIES LANGUAGE CUDA)
src/cvode-5.7.0/src/sunlinsol/magmadense/CMakeLists.txt:  set(_libs_needed sundials_sunmatrixmagmadense sundials_nveccuda)
src/cvode-5.7.0/src/sundials/fmod/fsundials_nvector_mod.f90:  enumerator :: SUNDIALS_NVEC_CUDA
src/cvode-5.7.0/src/sundials/fmod/fsundials_nvector_mod.f90:    SUNDIALS_NVEC_PETSC, SUNDIALS_NVEC_CUDA, SUNDIALS_NVEC_HIP, SUNDIALS_NVEC_SYCL, SUNDIALS_NVEC_RAJA, SUNDIALS_NVEC_OPENMPDEV, &
src/cvode-5.7.0/src/sundials/sundials_cuda.h: * for working with CUDA.
src/cvode-5.7.0/src/sundials/sundials_cuda.h:#include <cuda_runtime.h>
src/cvode-5.7.0/src/sundials/sundials_cuda.h:#ifndef _SUNDIALS_CUDA_H
src/cvode-5.7.0/src/sundials/sundials_cuda.h:#define _SUNDIALS_CUDA_H
src/cvode-5.7.0/src/sundials/sundials_cuda.h:#define CUDA_WARP_SIZE 32
src/cvode-5.7.0/src/sundials/sundials_cuda.h:#define MAX_CUDA_BLOCKSIZE 1024
src/cvode-5.7.0/src/sundials/sundials_cuda.h:#define SUNDIALS_CUDA_VERIFY(cuerr) SUNDIALS_CUDA_Assert(cuerr, __FILE__, __LINE__)
src/cvode-5.7.0/src/sundials/sundials_cuda.h:#ifndef SUNDIALS_DEBUG_CUDA_LASTERROR
src/cvode-5.7.0/src/sundials/sundials_cuda.h:  cudaDeviceSynchronize(); \
src/cvode-5.7.0/src/sundials/sundials_cuda.h:  SUNDIALS_CUDA_VERIFY(cudaGetLastError()); \
src/cvode-5.7.0/src/sundials/sundials_cuda.h:inline booleantype SUNDIALS_CUDA_Assert(cudaError_t cuerr, const char *file, int line)
src/cvode-5.7.0/src/sundials/sundials_cuda.h:  if (cuerr != cudaSuccess)
src/cvode-5.7.0/src/sundials/sundials_cuda.h:            "ERROR in CUDA runtime operation: %s %s:%d\n",
src/cvode-5.7.0/src/sundials/sundials_cuda.h:            cudaGetErrorString(cuerr), file, line);
src/cvode-5.7.0/src/sundials/sundials_cuda.h:#endif /* _SUNDIALS_CUDA_H */
src/cvode-5.7.0/src/sundials/sundials_hip_kernels.hip.hpp:#if defined(__CUDA_ARCH__)
src/cvode-5.7.0/src/sundials/sundials_hip_kernels.hip.hpp:#if defined(__CUDA_ARCH__) and __CUDA_ARCH__ < 600
src/cvode-5.7.0/src/sundials/sundials_cuda_kernels.cuh:#ifndef _SUNDIALS_CUDA_KERNELS_CUH
src/cvode-5.7.0/src/sundials/sundials_cuda_kernels.cuh:#define _SUNDIALS_CUDA_KERNELS_CUH
src/cvode-5.7.0/src/sundials/sundials_cuda_kernels.cuh:#include "sundials_cuda.h"
src/cvode-5.7.0/src/sundials/sundials_cuda_kernels.cuh:namespace cuda
src/cvode-5.7.0/src/sundials/sundials_cuda_kernels.cuh:#if __CUDA_ARCH__ < 600
src/cvode-5.7.0/src/sundials/sundials_cuda_kernels.cuh:  static __shared__ T shared[CUDA_WARP_SIZE]; 
src/cvode-5.7.0/src/sundials/sundials_cuda_kernels.cuh:} // namespace cuda
src/cvode-5.7.0/src/sundials/sundials_cuda_kernels.cuh:#endif // _SUNDIALS_CUDA_KERNELS_CUH
src/cvode-5.7.0/src/sundials/CMakeLists.txt:if(ENABLE_CUDA)
src/cvode-5.7.0/src/sundials/CMakeLists.txt:  list(APPEND sundials_HEADERS sundials_cuda_policies.hpp)
src/cvode-5.7.0/src/nvector/hip/VectorArrayKernels.hip.hpp: * example in NVIDIA Corporation CUDA Samples, and parallel reduction
src/cvode-5.7.0/src/nvector/hip/VectorArrayKernels.hip.hpp: * examples in textbook by J. Cheng at al. "CUDA C Programming".
src/cvode-5.7.0/src/nvector/hip/VectorKernels.hip.hpp: * example in NVIDIA Corporation CUDA Samples, and parallel reduction
src/cvode-5.7.0/src/nvector/hip/VectorKernels.hip.hpp: * examples in textbook by J. Cheng at al. "CUDA C Programming".
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  // Get result from the GPU
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  realtype gpu_result = NVEC_HIP_HBUFFERp(X)[0];
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  return gpu_result;
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  realtype gpu_result = NVEC_HIP_HBUFFERp(X)[0];
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  return gpu_result;
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  // Get result from the GPU
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  realtype gpu_result = NVEC_HIP_HBUFFERp(X)[0];
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  return gpu_result;
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  // Get result from the GPU
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  realtype gpu_result = NVEC_HIP_HBUFFERp(X)[0];
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  return gpu_result;
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  // Get result from the GPU
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  realtype gpu_result = NVEC_HIP_HBUFFERp(X)[0];
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  return gpu_result;
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  // Get result from the GPU
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  realtype gpu_result = NVEC_HIP_HBUFFERp(X)[0];
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  return gpu_result;
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  // Get result from the GPU
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  realtype gpu_result = NVEC_HIP_HBUFFERp(X)[0];
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  return (gpu_result < HALF);
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  // Get result from the GPU
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  realtype gpu_result = NVEC_HIP_HBUFFERp(X)[0];
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  return (gpu_result < HALF);
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  // Get result from the GPU
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  realtype gpu_result = NVEC_HIP_HBUFFERp(num)[0];
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  return gpu_result;
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  // Copy GPU result to the cpu.
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  // Copy GPU result to the cpu.
src/cvode-5.7.0/src/nvector/hip/nvector_hip.hip.cpp:  // Copy GPU result to the cpu.
src/cvode-5.7.0/src/nvector/CMakeLists.txt:if(BUILD_NVECTOR_CUDA)
src/cvode-5.7.0/src/nvector/CMakeLists.txt:  add_subdirectory(cuda)
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp: * of the NVECTOR package. This will support CUDA and HIP
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:#if defined(SUNDIALS_RAJA_BACKENDS_CUDA)
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:#include <sunmemory/sunmemory_cuda.h>
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:#include "sundials_cuda.h"
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:#define RAJA_NODE_TYPE RAJA::cuda_exec< 256 >
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:#define RAJA_REDUCE_TYPE RAJA::cuda_reduce
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:#define SUNDIALS_GPU_PREFIX(val) cuda ## val
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:#define SUNDIALS_GPU_VERIFY SUNDIALS_CUDA_VERIFY
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:#define SUNDIALS_GPU_PREFIX(val) hip ## val
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:#define SUNDIALS_GPU_VERIFY SUNDIALS_HIP_VERIFY
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:#if defined(SUNDIALS_RAJA_BACKENDS_CUDA)
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  NVEC_RAJA_CONTENT(v)->mem_helper      = SUNMemoryHelper_Cuda();
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:#if defined(SUNDIALS_RAJA_BACKENDS_CUDA)
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  NVEC_RAJA_CONTENT(v)->mem_helper      = SUNMemoryHelper_Cuda();
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:#if defined(SUNDIALS_RAJA_BACKENDS_CUDA)
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  NVEC_RAJA_CONTENT(v)->mem_helper      = SUNMemoryHelper_Cuda();
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:#if defined(SUNDIALS_RAJA_BACKENDS_CUDA)
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  NVEC_RAJA_CONTENT(v)->mem_helper      = SUNMemoryHelper_Cuda();
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  SUNDIALS_GPU_VERIFY(SUNDIALS_GPU_PREFIX(StreamSynchronize)(0));
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  SUNDIALS_GPU_VERIFY(SUNDIALS_GPU_PREFIX(StreamSynchronize)(0));
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  RAJA::ReduceSum< RAJA_REDUCE_TYPE, realtype> gpu_result(0.0);
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:      gpu_result += xdata[i] * ydata[i] ;
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  return (static_cast<realtype>(gpu_result));
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  RAJA::ReduceMax< RAJA_REDUCE_TYPE, realtype> gpu_result(0.0);
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:      gpu_result.max(abs(xdata[i]));
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  return (static_cast<realtype>(gpu_result));
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  RAJA::ReduceSum< RAJA_REDUCE_TYPE, realtype> gpu_result(0.0);
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:      gpu_result += (xdata[i] * wdata[i] * xdata[i] * wdata[i]);
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  return (static_cast<realtype>(gpu_result));
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  RAJA::ReduceSum< RAJA_REDUCE_TYPE, realtype> gpu_result(0.0);
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:        gpu_result += (xdata[i] * wdata[i] * xdata[i] * wdata[i]);
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  return (static_cast<realtype>(gpu_result));
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  RAJA::ReduceMin< RAJA_REDUCE_TYPE, realtype> gpu_result(std::numeric_limits<realtype>::max());
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:      gpu_result.min(xdata[i]);
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  return (static_cast<realtype>(gpu_result));
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  RAJA::ReduceSum< RAJA_REDUCE_TYPE, realtype> gpu_result(0.0);
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:      gpu_result += (abs(xdata[i]));
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  return (static_cast<realtype>(gpu_result));
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  RAJA::ReduceSum< RAJA_REDUCE_TYPE, realtype> gpu_result(ZERO);
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:        gpu_result += ONE;
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  realtype minimum = static_cast<realtype>(gpu_result);
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  RAJA::ReduceSum< RAJA_REDUCE_TYPE, realtype> gpu_result(ZERO);
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:      gpu_result += mdata[i];
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  realtype sum = static_cast<realtype>(gpu_result);
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  RAJA::ReduceMin< RAJA_REDUCE_TYPE, realtype> gpu_result(std::numeric_limits<realtype>::max());
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:        gpu_result.min(ndata[i]/ddata[i]);
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  return (static_cast<realtype>(gpu_result));
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  SUNDIALS_GPU_PREFIX(Error_t) cuerr;
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  cuerr = SUNDIALS_GPU_PREFIX(StreamSynchronize)(0);
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  return (!SUNDIALS_GPU_VERIFY(cuerr) || copy_fail ? -1 : 0);
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  SUNDIALS_GPU_PREFIX(Error_t) cuerr;
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  cuerr = SUNDIALS_GPU_PREFIX(StreamSynchronize)(0);
src/cvode-5.7.0/src/nvector/raja/nvector_raja.cpp:  return (!SUNDIALS_GPU_VERIFY(cuerr) || copy_fail ? -1 : 0);
src/cvode-5.7.0/src/nvector/raja/CMakeLists.txt:if(SUNDIALS_RAJA_BACKENDS MATCHES "CUDA")
src/cvode-5.7.0/src/nvector/raja/CMakeLists.txt:  set(_sunmemlib sundials_sunmemcuda_obj)
src/cvode-5.7.0/src/nvector/raja/CMakeLists.txt:  # set(_compile_defs SUNDIALS_RAJA_BACKENDS_CUDA)
src/cvode-5.7.0/src/nvector/raja/CMakeLists.txt:  set(_lib_output_name sundials_nveccudaraja)
src/cvode-5.7.0/src/nvector/raja/CMakeLists.txt:  set_source_files_properties(nvector_raja.cpp PROPERTIES LANGUAGE CUDA)
src/cvode-5.7.0/src/nvector/raja/CMakeLists.txt:    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
src/cvode-5.7.0/src/nvector/cuda/CMakeLists.txt:# CMakeLists.txt file for the cuda NVECTOR library
src/cvode-5.7.0/src/nvector/cuda/CMakeLists.txt:install(CODE "MESSAGE(\"\nInstall NVECTOR_CUDA\n\")")
src/cvode-5.7.0/src/nvector/cuda/CMakeLists.txt:sundials_add_library(sundials_nveccuda
src/cvode-5.7.0/src/nvector/cuda/CMakeLists.txt:    nvector_cuda.cu
src/cvode-5.7.0/src/nvector/cuda/CMakeLists.txt:    ${SUNDIALS_SOURCE_DIR}/include/nvector/nvector_cuda.h
src/cvode-5.7.0/src/nvector/cuda/CMakeLists.txt:    sundials_sunmemcuda_obj
src/cvode-5.7.0/src/nvector/cuda/CMakeLists.txt:    sundials_nveccuda
src/cvode-5.7.0/src/nvector/cuda/CMakeLists.txt:message(STATUS "Added NVECTOR_CUDA module")
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu: * This is the implementation file for a CUDA implementation
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:#include <nvector/nvector_cuda.h>
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:#include "sundials_cuda.h"
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:using namespace sundials::nvector_cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:#define NVEC_CUDA_CONTENT(x)  ((N_VectorContent_Cuda)(x->content))
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:#define NVEC_CUDA_PRIVATE(x)  ((N_PrivateVectorContent_Cuda)(NVEC_CUDA_CONTENT(x)->priv))
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:#define NVEC_CUDA_MEMSIZE(x)  (NVEC_CUDA_CONTENT(x)->length * sizeof(realtype))
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:#define NVEC_CUDA_MEMHELP(x)  (NVEC_CUDA_CONTENT(x)->mem_helper)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:#define NVEC_CUDA_HDATAp(x)   ((realtype*) NVEC_CUDA_CONTENT(x)->host_data->ptr)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:#define NVEC_CUDA_DDATAp(x)   ((realtype*) NVEC_CUDA_CONTENT(x)->device_data->ptr)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:#define NVEC_CUDA_HBUFFERp(x) ((realtype*) NVEC_CUDA_PRIVATE(x)->reduce_buffer_host->ptr)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:#define NVEC_CUDA_DBUFFERp(x) ((realtype*) NVEC_CUDA_PRIVATE(x)->reduce_buffer_dev->ptr)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:#define NVEC_CUDA_STREAM(x)   (NVEC_CUDA_CONTENT(x)->stream_exec_policy->stream())
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:struct _N_PrivateVectorContent_Cuda
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:typedef struct _N_PrivateVectorContent_Cuda *N_PrivateVectorContent_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                               size_t& shMemSize, cudaStream_t& stream, size_t n = 0);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu: * Private functions needed for N_VMakeWithManagedAllocator_Cuda
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:   N_VMakeWithManagedAllocator_Cuda (deprecated) is removed in the
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:N_Vector N_VNewEmpty_Cuda()
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvgetvectorid           = N_VGetVectorID_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvclone                 = N_VClone_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvcloneempty            = N_VCloneEmpty_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvdestroy               = N_VDestroy_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvspace                 = N_VSpace_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvgetlength             = N_VGetLength_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvgetarraypointer       = N_VGetHostArrayPointer_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvgetdevicearraypointer = N_VGetDeviceArrayPointer_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvsetarraypointer       = N_VSetHostArrayPointer_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvlinearsum    = N_VLinearSum_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvconst        = N_VConst_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvprod         = N_VProd_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvdiv          = N_VDiv_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvscale        = N_VScale_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvabs          = N_VAbs_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvinv          = N_VInv_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvaddconst     = N_VAddConst_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvdotprod      = N_VDotProd_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvmaxnorm      = N_VMaxNorm_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvmin          = N_VMin_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvl1norm       = N_VL1Norm_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvinvtest      = N_VInvTest_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvconstrmask   = N_VConstrMask_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvminquotient  = N_VMinQuotient_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvwrmsnormmask = N_VWrmsNormMask_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvwrmsnorm     = N_VWrmsNorm_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvwl2norm      = N_VWL2Norm_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvcompare      = N_VCompare_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvdotprodlocal     = N_VDotProd_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvmaxnormlocal     = N_VMaxNorm_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvminlocal         = N_VMin_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvl1normlocal      = N_VL1Norm_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvinvtestlocal     = N_VInvTest_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvconstrmasklocal  = N_VConstrMask_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvminquotientlocal = N_VMinQuotient_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvwsqrsumlocal     = N_VWSqrSumLocal_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvwsqrsummasklocal = N_VWSqrSumMaskLocal_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvbufsize   = N_VBufSize_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvbufpack   = N_VBufPack_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvbufunpack = N_VBufUnpack_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvprint     = N_VPrint_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->ops->nvprintfile = N_VPrintFile_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v->content = (N_VectorContent_Cuda) malloc(sizeof(_N_VectorContent_Cuda));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->priv = malloc(sizeof(_N_PrivateVectorContent_Cuda));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (NVEC_CUDA_CONTENT(v)->priv == NULL)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->length                        = 0;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->host_data                     = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->device_data                   = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->stream_exec_policy            = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->reduce_exec_policy            = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->mem_helper                    = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->own_helper                    = SUNFALSE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->own_exec                      = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->use_managed_mem               = SUNFALSE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_dev             = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_host            = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_allocated_bytes = 0;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:N_Vector N_VNew_Cuda(sunindextype length)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v = N_VNewEmpty_Cuda();
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->length                        = length;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->host_data                     = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->device_data                   = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->mem_helper                    = SUNMemoryHelper_Cuda();
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->stream_exec_policy            = new CudaThreadDirectExecPolicy(256);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->reduce_exec_policy            = new CudaBlockReduceExecPolicy(256);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->own_helper                    = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->own_exec                      = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->use_managed_mem               = SUNFALSE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_dev             = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_host            = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_allocated_bytes = 0;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (NVEC_CUDA_MEMHELP(v) == NULL)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VNew_Cuda: memory helper is NULL\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VNew_Cuda: AllocateData returned nonzero\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:N_Vector N_VNewWithMemHelp_Cuda(sunindextype length, booleantype use_managed_mem, SUNMemoryHelper helper)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VNewWithMemHelp_Cuda: helper is NULL\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VNewWithMemHelp_Cuda: helper doesn't implement all required ops\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v = N_VNewEmpty_Cuda();
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->length                        = length;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->host_data                     = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->device_data                   = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->mem_helper                    = helper;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->stream_exec_policy            = new CudaThreadDirectExecPolicy(256);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->reduce_exec_policy            = new CudaBlockReduceExecPolicy(256);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->own_helper                    = SUNFALSE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->own_exec                      = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->use_managed_mem               = use_managed_mem;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_dev             = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_host            = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_allocated_bytes = 0;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VNewWithMemHelp_Cuda: AllocateData returned nonzero\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:N_Vector N_VNewManaged_Cuda(sunindextype length)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v = N_VNewEmpty_Cuda();
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->length                        = length;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->host_data                     = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->device_data                   = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->stream_exec_policy            = new CudaThreadDirectExecPolicy(256);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->reduce_exec_policy            = new CudaBlockReduceExecPolicy(256);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->mem_helper                    = SUNMemoryHelper_Cuda();
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->own_helper                    = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->own_exec                      = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->use_managed_mem               = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_dev             = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_host            = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_allocated_bytes = 0;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (NVEC_CUDA_MEMHELP(v) == NULL)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VNewManaged_Cuda: memory helper is NULL\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VNewManaged_Cuda: AllocateData returned nonzero\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:N_Vector N_VMake_Cuda(sunindextype length, realtype *h_vdata, realtype *d_vdata)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v = N_VNewEmpty_Cuda();
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->length                        = length;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->host_data                     = SUNMemoryHelper_Wrap(h_vdata, SUNMEMTYPE_HOST);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->device_data                   = SUNMemoryHelper_Wrap(d_vdata, SUNMEMTYPE_DEVICE);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->stream_exec_policy            = new CudaThreadDirectExecPolicy(256);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->reduce_exec_policy            = new CudaBlockReduceExecPolicy(256);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->mem_helper                    = SUNMemoryHelper_Cuda();
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->own_helper                    = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->own_exec                      = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->use_managed_mem               = SUNFALSE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_dev             = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_host            = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_allocated_bytes = 0;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (NVEC_CUDA_MEMHELP(v) == NULL)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VMake_Cuda: memory helper is NULL\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (NVEC_CUDA_CONTENT(v)->device_data == NULL ||
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      NVEC_CUDA_CONTENT(v)->host_data == NULL)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VMake_Cuda: SUNMemoryHelper_Wrap returned NULL\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:N_Vector N_VMakeManaged_Cuda(sunindextype length, realtype *vdata)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v = N_VNewEmpty_Cuda();
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->length                        = length;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->host_data                     = SUNMemoryHelper_Wrap(vdata, SUNMEMTYPE_UVM);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->device_data                   = SUNMemoryHelper_Alias(NVEC_CUDA_CONTENT(v)->host_data);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->stream_exec_policy            = new CudaThreadDirectExecPolicy(256);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->reduce_exec_policy            = new CudaBlockReduceExecPolicy(256);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->mem_helper                    = SUNMemoryHelper_Cuda();
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->own_helper                    = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->own_exec                      = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->use_managed_mem               = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_dev             = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_host            = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_allocated_bytes = 0;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (NVEC_CUDA_MEMHELP(v) == NULL)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VMakeManaged_Cuda: memory helper is NULL\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (NVEC_CUDA_CONTENT(v)->device_data == NULL ||
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      NVEC_CUDA_CONTENT(v)->host_data == NULL)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VMakeManaged_Cuda: SUNMemoryHelper_Wrap returned NULL\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:N_Vector N_VMakeWithManagedAllocator_Cuda(sunindextype length,
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v = N_VNewEmpty_Cuda();
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->length                        = length;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->host_data                     = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->device_data                   = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->stream_exec_policy            = new CudaThreadDirectExecPolicy(256);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->reduce_exec_policy            = new CudaBlockReduceExecPolicy(256);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->mem_helper                    = SUNMemoryHelper_Cuda();
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->own_helper                    = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->own_exec                      = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->use_managed_mem               = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_dev             = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_host            = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_allocated_bytes = 0;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (NVEC_CUDA_MEMHELP(v) == NULL)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VMakeWithManagedAllocator_Cuda: memory helper is NULL\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_MEMHELP(v)->content      = (void*) ua;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_MEMHELP(v)->ops->alloc   = UserAlloc;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_MEMHELP(v)->ops->dealloc = UserDealloc;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_MEMHELP(v)->ops->clone   = HelperClone;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_MEMHELP(v)->ops->destroy = HelperDestroy;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VMakeWithManagedAllocator_Cuda: AllocateData returned nonzero\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VSetHostArrayPointer_Cuda(realtype* h_vdata, N_Vector v)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (N_VIsManagedMemory_Cuda(v))
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    if (NVEC_CUDA_CONTENT(v)->host_data)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      NVEC_CUDA_CONTENT(v)->host_data->ptr = (void*) h_vdata;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      NVEC_CUDA_CONTENT(v)->device_data->ptr = (void*) h_vdata;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      NVEC_CUDA_CONTENT(v)->host_data = SUNMemoryHelper_Wrap((void*) h_vdata, SUNMEMTYPE_UVM);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      NVEC_CUDA_CONTENT(v)->device_data = SUNMemoryHelper_Alias(NVEC_CUDA_CONTENT(v)->host_data);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    if (NVEC_CUDA_CONTENT(v)->host_data)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      NVEC_CUDA_CONTENT(v)->host_data->ptr = (void*) h_vdata;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      NVEC_CUDA_CONTENT(v)->host_data = SUNMemoryHelper_Wrap((void*) h_vdata, SUNMEMTYPE_HOST);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VSetDeviceArrayPointer_Cuda(realtype* d_vdata, N_Vector v)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (N_VIsManagedMemory_Cuda(v))
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    if (NVEC_CUDA_CONTENT(v)->device_data)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      NVEC_CUDA_CONTENT(v)->device_data->ptr = (void*) d_vdata;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      NVEC_CUDA_CONTENT(v)->host_data->ptr = (void*) d_vdata;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      NVEC_CUDA_CONTENT(v)->device_data = SUNMemoryHelper_Wrap((void*) d_vdata, SUNMEMTYPE_UVM);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      NVEC_CUDA_CONTENT(v)->host_data = SUNMemoryHelper_Alias(NVEC_CUDA_CONTENT(v)->device_data);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    if (NVEC_CUDA_CONTENT(v)->device_data)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      NVEC_CUDA_CONTENT(v)->device_data->ptr = (void*) d_vdata;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      NVEC_CUDA_CONTENT(v)->device_data = SUNMemoryHelper_Wrap((void*) d_vdata, SUNMEMTYPE_DEVICE);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:booleantype N_VIsManagedMemory_Cuda(N_Vector x)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  return NVEC_CUDA_PRIVATE(x)->use_managed_mem;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VSetKernelExecPolicy_Cuda(N_Vector x,
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                SUNCudaExecPolicy* stream_exec_policy,
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                SUNCudaExecPolicy* reduce_exec_policy)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (NVEC_CUDA_CONTENT(x)->own_exec)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    delete NVEC_CUDA_CONTENT(x)->stream_exec_policy;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    delete NVEC_CUDA_CONTENT(x)->reduce_exec_policy;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(x)->stream_exec_policy = stream_exec_policy;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(x)->reduce_exec_policy = reduce_exec_policy;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(x)->own_exec = SUNFALSE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu: * Sets the cudaStream_t to use for execution of the CUDA kernels.
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VSetCudaStream_Cuda(N_Vector x, cudaStream_t *stream)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  const CudaExecPolicy* xs = NVEC_CUDA_CONTENT(x)->stream_exec_policy;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  const CudaExecPolicy* xr = NVEC_CUDA_CONTENT(x)->reduce_exec_policy;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  CudaThreadDirectExecPolicy* s =
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    new CudaThreadDirectExecPolicy(xs->blockSize(), *stream);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  CudaBlockReduceExecPolicy* r =
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    new CudaBlockReduceExecPolicy(xr->blockSize(), xr->gridSize(), *stream);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  N_VSetKernelExecPolicy_Cuda(x, s, r);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(x)->own_exec = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VCopyToDevice_Cuda(N_Vector x)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  copy_fail = SUNMemoryHelper_CopyAsync(NVEC_CUDA_MEMHELP(x),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                        NVEC_CUDA_CONTENT(x)->device_data,
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                        NVEC_CUDA_CONTENT(x)->host_data,
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                        NVEC_CUDA_MEMSIZE(x),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                        (void*) NVEC_CUDA_STREAM(x));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VCopyToDevice_Cuda: SUNMemoryHelper_CopyAsync returned nonzero\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  SUNDIALS_CUDA_VERIFY(cudaStreamSynchronize(*NVEC_CUDA_STREAM(x)));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VCopyFromDevice_Cuda(N_Vector x)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  copy_fail = SUNMemoryHelper_CopyAsync(NVEC_CUDA_MEMHELP(x),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                        NVEC_CUDA_CONTENT(x)->host_data,
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                        NVEC_CUDA_CONTENT(x)->device_data,
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                        NVEC_CUDA_MEMSIZE(x),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                        (void*) NVEC_CUDA_STREAM(x));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VCopyFromDevice_Cuda: SUNMemoryHelper_CopyAsync returned nonzero\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  SUNDIALS_CUDA_VERIFY(cudaStreamSynchronize(*NVEC_CUDA_STREAM(x)));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu: * Function to print the a CUDA-based vector to stdout
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VPrint_Cuda(N_Vector x)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  N_VPrintFile_Cuda(x, stdout);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu: * Function to print the a CUDA-based vector to outfile
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VPrintFile_Cuda(N_Vector x, FILE *outfile)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  for (i = 0; i < NVEC_CUDA_CONTENT(x)->length; i++) {
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    fprintf(outfile, "%35.32Lg\n", NVEC_CUDA_HDATAp(x)[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    fprintf(outfile, "%19.16g\n", NVEC_CUDA_HDATAp(x)[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    fprintf(outfile, "%11.8g\n", NVEC_CUDA_HDATAp(x)[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:N_Vector N_VCloneEmpty_Cuda(N_Vector w)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v = N_VNewEmpty_Cuda();
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->length                        = NVEC_CUDA_CONTENT(w)->length;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->host_data                     = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->device_data                   = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->mem_helper                    = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->own_exec                      = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->use_managed_mem               = NVEC_CUDA_PRIVATE(w)->use_managed_mem;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_dev             = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_host            = NULL;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_PRIVATE(v)->reduce_buffer_allocated_bytes = 0;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:N_Vector N_VClone_Cuda(N_Vector w)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  v = N_VCloneEmpty_Cuda(w);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_MEMHELP(v) = SUNMemoryHelper_Clone(NVEC_CUDA_MEMHELP(w));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->own_helper = SUNTRUE;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->stream_exec_policy = NVEC_CUDA_CONTENT(w)->stream_exec_policy->clone();
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  NVEC_CUDA_CONTENT(v)->reduce_exec_policy = NVEC_CUDA_CONTENT(w)->reduce_exec_policy->clone();
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (NVEC_CUDA_MEMHELP(v) == NULL)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VClone_Cuda: SUNMemoryHelper_Clone returned NULL\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VClone_Cuda: AllocateData returned nonzero\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VDestroy_Cuda(N_Vector v)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  N_VectorContent_Cuda vc;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  N_PrivateVectorContent_Cuda vcp;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  vc = NVEC_CUDA_CONTENT(v);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  vcp = (N_PrivateVectorContent_Cuda) vc->priv;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (NVEC_CUDA_MEMHELP(v))
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNMemoryHelper_Dealloc(NVEC_CUDA_MEMHELP(v), vc->host_data);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNMemoryHelper_Dealloc(NVEC_CUDA_MEMHELP(v), vc->device_data);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VSpace_Cuda(N_Vector X, sunindextype *lrw, sunindextype *liw)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  *lrw = NVEC_CUDA_CONTENT(X)->length;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VConst_Cuda(realtype a, N_Vector X)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VLinearSum_Cuda(realtype a, N_Vector X, realtype b, N_Vector Y, N_Vector Z)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(Y),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(Z),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VProd_Cuda(N_Vector X, N_Vector Y, N_Vector Z)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(Y),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(Z),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VDiv_Cuda(N_Vector X, N_Vector Y, N_Vector Z)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(Y),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(Z),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VScale_Cuda(realtype a, N_Vector X, N_Vector Z)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(Z),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VAbs_Cuda(N_Vector X, N_Vector Z)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(Z),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VInv_Cuda(N_Vector X, N_Vector Z)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(Z),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VAddConst_Cuda(N_Vector X, realtype b, N_Vector Z)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(Z),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:realtype N_VDotProd_Cuda(N_Vector X, N_Vector Y)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VDotProd_Cuda: InitializeReductionBuffer returned nonzero\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(Y),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DBUFFERp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  // Get result from the GPU
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  realtype gpu_result = NVEC_CUDA_HBUFFERp(X)[0];
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  return gpu_result;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:realtype N_VMaxNorm_Cuda(N_Vector X)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VMaxNorm_Cuda: InitializeReductionBuffer returned nonzero\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DBUFFERp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  realtype gpu_result = NVEC_CUDA_HBUFFERp(X)[0];
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  return gpu_result;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:realtype N_VWSqrSumLocal_Cuda(N_Vector X, N_Vector W)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VWSqrSumLocal_Cuda: InitializeReductionBuffer returned nonzero\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(W),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DBUFFERp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  // Get result from the GPU
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  realtype gpu_result = NVEC_CUDA_HBUFFERp(X)[0];
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  return gpu_result;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:realtype N_VWrmsNorm_Cuda(N_Vector X, N_Vector W)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  const realtype sum = N_VWSqrSumLocal_Cuda(X, W);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  return std::sqrt(sum/NVEC_CUDA_CONTENT(X)->length);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:realtype N_VWSqrSumMaskLocal_Cuda(N_Vector X, N_Vector W, N_Vector Id)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VWSqrSumMaskLocal_Cuda: InitializeReductionBuffer returned nonzero\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(W),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(Id),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DBUFFERp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  // Get result from the GPU
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  realtype gpu_result = NVEC_CUDA_HBUFFERp(X)[0];
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  return gpu_result;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:realtype N_VWrmsNormMask_Cuda(N_Vector X, N_Vector W, N_Vector Id)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  const realtype sum = N_VWSqrSumMaskLocal_Cuda(X, W, Id);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  return std::sqrt(sum/NVEC_CUDA_CONTENT(X)->length);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:realtype N_VMin_Cuda(N_Vector X)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VMin_Cuda: InitializeReductionBuffer returned nonzero\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DBUFFERp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  // Get result from the GPU
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  realtype gpu_result = NVEC_CUDA_HBUFFERp(X)[0];
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  return gpu_result;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:realtype N_VWL2Norm_Cuda(N_Vector X, N_Vector W)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  const realtype sum = N_VWSqrSumLocal_Cuda(X, W);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:realtype N_VL1Norm_Cuda(N_Vector X)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VL1Norm_Cuda: InitializeReductionBuffer returned nonzero\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DBUFFERp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  // Get result from the GPU
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  realtype gpu_result = NVEC_CUDA_HBUFFERp(X)[0];
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  return gpu_result;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:void N_VCompare_Cuda(realtype c, N_Vector X, N_Vector Z)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(Z),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:booleantype N_VInvTest_Cuda(N_Vector X, N_Vector Z)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VInvTest_Cuda: InitializeReductionBuffer returned nonzero\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(Z),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DBUFFERp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  // Get result from the GPU
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  realtype gpu_result = NVEC_CUDA_HBUFFERp(X)[0];
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  return (gpu_result < HALF);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:booleantype N_VConstrMask_Cuda(N_Vector C, N_Vector X, N_Vector M)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VConstrMask_Cuda: InitializeReductionBuffer returned nonzero\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(C),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(M),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DBUFFERp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  // Get result from the GPU
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  realtype gpu_result = NVEC_CUDA_HBUFFERp(X)[0];
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  return (gpu_result < HALF);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:realtype N_VMinQuotient_Cuda(N_Vector num, N_Vector denom)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNDIALS_DEBUG_PRINT("ERROR in N_VMinQuotient_Cuda: InitializeReductionBuffer returned nonzero\n");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(num),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(denom),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DBUFFERp(num),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(num)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  // Get result from the GPU
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  realtype gpu_result = NVEC_CUDA_HBUFFERp(num)[0];
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  return gpu_result;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VLinearCombination_Cuda(int nvec, realtype* c, N_Vector* X, N_Vector Z)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaError_t err;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_c, nvec*sizeof(realtype));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_c, c, nvec*sizeof(realtype), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    h_Xd[i] = NVEC_CUDA_DDATAp(X[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Xd, nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Xd, h_Xd, nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(Z),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(Z)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_c);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Xd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VScaleAddMulti_Cuda(int nvec, realtype* c, N_Vector X, N_Vector* Y,
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaError_t err;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_c, nvec*sizeof(realtype));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_c, c, nvec*sizeof(realtype), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    h_Yd[i] = NVEC_CUDA_DDATAp(Y[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    h_Zd[i] = NVEC_CUDA_DDATAp(Z[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Yd, nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Yd, h_Yd, nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Zd, nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Zd, h_Zd, nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_c);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Yd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Zd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VDotProdMulti_Cuda(int nvec, N_Vector X, N_Vector* Y, realtype* dots)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaError_t err;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    h_Yd[i] = NVEC_CUDA_DDATAp(Y[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Yd, nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Yd, h_Yd, nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_buff, grid*sizeof(realtype));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemsetAsync(d_buff, 0, grid*sizeof(realtype));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(X),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X)->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  // Copy GPU result to the cpu.
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(dots, d_buff, grid*sizeof(realtype), cudaMemcpyDeviceToHost);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Yd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_buff);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VLinearSumVectorArray_Cuda(int nvec, realtype a, N_Vector* X, realtype b,
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaError_t err;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    h_Xd[i] = NVEC_CUDA_DDATAp(X[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    h_Yd[i] = NVEC_CUDA_DDATAp(Y[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    h_Zd[i] = NVEC_CUDA_DDATAp(Z[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Xd, nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Xd, h_Xd, nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Yd, nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Yd, h_Yd, nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Zd, nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Zd, h_Zd, nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(Z[0])->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Xd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Yd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Zd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VScaleVectorArray_Cuda(int nvec, realtype* c, N_Vector* X, N_Vector* Z)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaError_t err;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_c, nvec*sizeof(realtype));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_c, c, nvec*sizeof(realtype), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    h_Xd[i] = NVEC_CUDA_DDATAp(X[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    h_Zd[i] = NVEC_CUDA_DDATAp(Z[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Xd, nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Xd, h_Xd, nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Zd, nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Zd, h_Zd, nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(Z[0])->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_c);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Xd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Zd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VConstVectorArray_Cuda(int nvec, realtype c, N_Vector* Z)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaError_t err;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    h_Zd[i] = NVEC_CUDA_DDATAp(Z[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Zd, nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Zd, h_Zd, nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(Z[0])->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Zd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VWrmsNormVectorArray_Cuda(int nvec, N_Vector* X, N_Vector* W,
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaError_t err;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    h_Xd[i] = NVEC_CUDA_DDATAp(X[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    h_Wd[i] = NVEC_CUDA_DDATAp(W[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Xd, nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Xd, h_Xd, nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Wd, nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Wd, h_Wd, nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_buff, grid*sizeof(realtype));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemsetAsync(d_buff, 0, grid*sizeof(realtype));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X[0])->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  // Copy GPU result to the cpu.
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(norms, d_buff, grid*sizeof(realtype), cudaMemcpyDeviceToHost);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    norms[k] = std::sqrt(norms[k]/NVEC_CUDA_CONTENT(X[0])->length);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Xd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Wd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_buff);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VWrmsNormMaskVectorArray_Cuda(int nvec, N_Vector* X, N_Vector* W,
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaError_t err;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    h_Xd[i] = NVEC_CUDA_DDATAp(X[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    h_Wd[i] = NVEC_CUDA_DDATAp(W[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Xd, nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Xd, h_Xd, nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Wd, nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Wd, h_Wd, nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_buff, grid*sizeof(realtype));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemsetAsync(d_buff, 0, grid*sizeof(realtype));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_DDATAp(id),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(X[0])->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  // Copy GPU result to the cpu.
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(norms, d_buff, grid*sizeof(realtype), cudaMemcpyDeviceToHost);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    norms[k] = std::sqrt(norms[k]/NVEC_CUDA_CONTENT(X[0])->length);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Xd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Wd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_buff);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VScaleAddMultiVectorArray_Cuda(int nvec, int nsum, realtype* c,
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaError_t err;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_c, nsum*sizeof(realtype));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_c, c, nsum*sizeof(realtype), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    h_Xd[i] = NVEC_CUDA_DDATAp(X[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      h_Yd[j*nsum+i] = NVEC_CUDA_DDATAp(Y[i][j]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      h_Zd[j*nsum+i] = NVEC_CUDA_DDATAp(Z[i][j]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Xd, nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Xd, h_Xd, nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Yd, nsum*nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Yd, h_Yd, nsum*nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Zd, nsum*nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Zd, h_Zd, nsum*nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(Z[0][0])->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_c);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Xd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Yd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Zd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VLinearCombinationVectorArray_Cuda(int nvec, int nsum, realtype* c,
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaError_t err;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_c, nsum*sizeof(realtype));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_c, c, nsum*sizeof(realtype), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      h_Xd[j*nsum+i] = NVEC_CUDA_DDATAp(X[i][j]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    h_Zd[i] = NVEC_CUDA_DDATAp(Z[i]);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Xd, nsum*nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Xd, h_Xd, nsum*nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMalloc((void**) &d_Zd, nvec*sizeof(realtype*));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaMemcpy(d_Zd, h_Zd, nvec*sizeof(realtype*), cudaMemcpyHostToDevice);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaStream_t stream;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    NVEC_CUDA_CONTENT(Z[0])->length
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_c);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Xd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  err = cudaFree(d_Zd);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (!SUNDIALS_CUDA_VERIFY(err)) return(-1);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  return cudaGetLastError();
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VBufSize_Cuda(N_Vector x, sunindextype *size)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  *size = (sunindextype)NVEC_CUDA_MEMSIZE(x);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VBufPack_Cuda(N_Vector x, void *buf)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaError_t cuerr;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  copy_fail = SUNMemoryHelper_CopyAsync(NVEC_CUDA_MEMHELP(x),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                        NVEC_CUDA_CONTENT(x)->device_data,
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                        NVEC_CUDA_MEMSIZE(x),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                        (void*) NVEC_CUDA_STREAM(x));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cuerr = cudaStreamSynchronize(*NVEC_CUDA_STREAM(x));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  SUNMemoryHelper_Dealloc(NVEC_CUDA_MEMHELP(x), buf_mem);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  return (!SUNDIALS_CUDA_VERIFY(cuerr) || copy_fail ? -1 : 0);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VBufUnpack_Cuda(N_Vector x, void *buf)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaError_t cuerr;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  copy_fail = SUNMemoryHelper_CopyAsync(NVEC_CUDA_MEMHELP(x),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                        NVEC_CUDA_CONTENT(x)->device_data,
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                        NVEC_CUDA_MEMSIZE(x),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                        (void*) NVEC_CUDA_STREAM(x));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cuerr = cudaStreamSynchronize(*NVEC_CUDA_STREAM(x));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  SUNMemoryHelper_Dealloc(NVEC_CUDA_MEMHELP(x), buf_mem);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  return (!SUNDIALS_CUDA_VERIFY(cuerr) || copy_fail ? -1 : 0);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VEnableFusedOps_Cuda(N_Vector v, booleantype tf)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvlinearcombination = N_VLinearCombination_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvscaleaddmulti     = N_VScaleAddMulti_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvdotprodmulti      = N_VDotProdMulti_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvlinearsumvectorarray         = N_VLinearSumVectorArray_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvscalevectorarray             = N_VScaleVectorArray_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvconstvectorarray             = N_VConstVectorArray_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvwrmsnormvectorarray          = N_VWrmsNormVectorArray_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvwrmsnormmaskvectorarray      = N_VWrmsNormMaskVectorArray_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvscaleaddmultivectorarray     = N_VScaleAddMultiVectorArray_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvlinearcombinationvectorarray = N_VLinearCombinationVectorArray_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VEnableLinearCombination_Cuda(N_Vector v, booleantype tf)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvlinearcombination = N_VLinearCombination_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VEnableScaleAddMulti_Cuda(N_Vector v, booleantype tf)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvscaleaddmulti = N_VScaleAddMulti_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VEnableDotProdMulti_Cuda(N_Vector v, booleantype tf)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvdotprodmulti = N_VDotProdMulti_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VEnableLinearSumVectorArray_Cuda(N_Vector v, booleantype tf)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvlinearsumvectorarray = N_VLinearSumVectorArray_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VEnableScaleVectorArray_Cuda(N_Vector v, booleantype tf)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvscalevectorarray = N_VScaleVectorArray_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VEnableConstVectorArray_Cuda(N_Vector v, booleantype tf)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvconstvectorarray = N_VConstVectorArray_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VEnableWrmsNormVectorArray_Cuda(N_Vector v, booleantype tf)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvwrmsnormvectorarray = N_VWrmsNormVectorArray_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VEnableWrmsNormMaskVectorArray_Cuda(N_Vector v, booleantype tf)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvwrmsnormmaskvectorarray = N_VWrmsNormMaskVectorArray_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VEnableScaleAddMultiVectorArray_Cuda(N_Vector v, booleantype tf)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvscaleaddmultivectorarray = N_VScaleAddMultiVectorArray_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:int N_VEnableLinearCombinationVectorArray_Cuda(N_Vector v, booleantype tf)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    v->ops->nvlinearcombinationvectorarray = N_VLinearCombinationVectorArray_Cuda;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  N_VectorContent_Cuda vc = NVEC_CUDA_CONTENT(v);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  N_PrivateVectorContent_Cuda vcp = NVEC_CUDA_PRIVATE(v);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  if (N_VGetLength_Cuda(v) == 0) return(0);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    alloc_fail = SUNMemoryHelper_Alloc(NVEC_CUDA_MEMHELP(v), &(vc->device_data),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                       NVEC_CUDA_MEMSIZE(v), SUNMEMTYPE_UVM);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    alloc_fail = SUNMemoryHelper_Alloc(NVEC_CUDA_MEMHELP(v), &(vc->host_data),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                       NVEC_CUDA_MEMSIZE(v), SUNMEMTYPE_HOST);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    alloc_fail = SUNMemoryHelper_Alloc(NVEC_CUDA_MEMHELP(v), &(vc->device_data),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                       NVEC_CUDA_MEMSIZE(v), SUNMEMTYPE_DEVICE);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  N_PrivateVectorContent_Cuda vcp = NVEC_CUDA_PRIVATE(v);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    alloc_fail = SUNMemoryHelper_Alloc(NVEC_CUDA_MEMHELP(v),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      alloc_fail = SUNMemoryHelper_Alloc(NVEC_CUDA_MEMHELP(v),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    alloc_fail = SUNMemoryHelper_Alloc(NVEC_CUDA_MEMHELP(v),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    copy_fail = SUNMemoryHelper_CopyAsync(NVEC_CUDA_MEMHELP(v),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                          bytes, (void*) NVEC_CUDA_STREAM(v));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  SUNMemoryHelper_Dealloc(NVEC_CUDA_MEMHELP(v), value_mem);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  N_PrivateVectorContent_Cuda vcp = NVEC_CUDA_PRIVATE(v);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNMemoryHelper_Dealloc(NVEC_CUDA_MEMHELP(v), vcp->reduce_buffer_dev);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNMemoryHelper_Dealloc(NVEC_CUDA_MEMHELP(v), vcp->reduce_buffer_host);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaError_t cuerr;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  copy_fail = SUNMemoryHelper_CopyAsync(NVEC_CUDA_MEMHELP(v),
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                        NVEC_CUDA_PRIVATE(v)->reduce_buffer_host,
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                        NVEC_CUDA_PRIVATE(v)->reduce_buffer_dev,
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                                        (void*) NVEC_CUDA_STREAM(v));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cuerr = cudaStreamSynchronize(*NVEC_CUDA_STREAM(v));
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  return (!SUNDIALS_CUDA_VERIFY(cuerr) || copy_fail ? -1 : 0);
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:                               cudaStream_t& stream, size_t n)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  n = (n == 0) ? NVEC_CUDA_CONTENT(v)->length : n;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNCudaExecPolicy* reduce_exec_policy = NVEC_CUDA_CONTENT(v)->reduce_exec_policy;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    if (block % CUDA_WARP_SIZE)
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:      throw std::runtime_error("the block size must be a multiple must be of CUDA warp size");
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:    SUNCudaExecPolicy* stream_exec_policy = NVEC_CUDA_CONTENT(v)->stream_exec_policy;
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu: * If SUNDIALS_DEBUG_CUDA_LASTERROR is not defined, then the function does nothing.
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu: * If it is defined, the function will synchronize and check the last CUDA error.
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:#ifdef SUNDIALS_DEBUG_CUDA_LASTERROR
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  cudaDeviceSynchronize();
src/cvode-5.7.0/src/nvector/cuda/nvector_cuda.cu:  SUNDIALS_CUDA_VERIFY(cudaGetLastError());
src/cvode-5.7.0/src/nvector/cuda/VectorArrayKernels.cuh:#ifndef _NVECTOR_CUDA_ARRAY_KERNELS_CUH_
src/cvode-5.7.0/src/nvector/cuda/VectorArrayKernels.cuh:#define _NVECTOR_CUDA_ARRAY_KERNELS_CUH_
src/cvode-5.7.0/src/nvector/cuda/VectorArrayKernels.cuh:#include <cuda_runtime.h>
src/cvode-5.7.0/src/nvector/cuda/VectorArrayKernels.cuh:#include "sundials_cuda_kernels.cuh"
src/cvode-5.7.0/src/nvector/cuda/VectorArrayKernels.cuh:using namespace sundials::cuda;
src/cvode-5.7.0/src/nvector/cuda/VectorArrayKernels.cuh:namespace nvector_cuda
src/cvode-5.7.0/src/nvector/cuda/VectorArrayKernels.cuh: * The namespace for CUDA kernels
src/cvode-5.7.0/src/nvector/cuda/VectorArrayKernels.cuh: * Reduction CUDA kernels in nvector are based in part on "reduction"
src/cvode-5.7.0/src/nvector/cuda/VectorArrayKernels.cuh: * example in NVIDIA Corporation CUDA Samples, and parallel reduction
src/cvode-5.7.0/src/nvector/cuda/VectorArrayKernels.cuh: * examples in textbook by J. Cheng at al. "CUDA C Programming".
src/cvode-5.7.0/src/nvector/cuda/VectorArrayKernels.cuh:} // namespace nvector_cuda
src/cvode-5.7.0/src/nvector/cuda/VectorArrayKernels.cuh:#endif // _NVECTOR_CUDA_ARRAY_KERNELS_CUH_
src/cvode-5.7.0/src/nvector/cuda/VectorKernels.cuh:#ifndef _NVECTOR_CUDA_KERNELS_CUH_
src/cvode-5.7.0/src/nvector/cuda/VectorKernels.cuh:#define _NVECTOR_CUDA_KERNELS_CUH_
src/cvode-5.7.0/src/nvector/cuda/VectorKernels.cuh:#include <cuda_runtime.h>
src/cvode-5.7.0/src/nvector/cuda/VectorKernels.cuh:#include "sundials_cuda_kernels.cuh"
src/cvode-5.7.0/src/nvector/cuda/VectorKernels.cuh:using namespace sundials::cuda;
src/cvode-5.7.0/src/nvector/cuda/VectorKernels.cuh:namespace nvector_cuda
src/cvode-5.7.0/src/nvector/cuda/VectorKernels.cuh: * The namespace for CUDA kernels
src/cvode-5.7.0/src/nvector/cuda/VectorKernels.cuh: * Reduction CUDA kernels in nvector are based in part on "reduction"
src/cvode-5.7.0/src/nvector/cuda/VectorKernels.cuh: * example in NVIDIA Corporation CUDA Samples, and parallel reduction
src/cvode-5.7.0/src/nvector/cuda/VectorKernels.cuh: * examples in textbook by J. Cheng at al. "CUDA C Programming".
src/cvode-5.7.0/src/nvector/cuda/VectorKernels.cuh:} // namespace nvector_cuda
src/cvode-5.7.0/src/nvector/cuda/VectorKernels.cuh:#endif // _NVECTOR_CUDA_KERNELS_CUH_

```
