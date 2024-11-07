# https://github.com/CEED/libCEED

```console
setup.py:sparse matrices, and can achieve very high performance on modern CPU and GPU
setup.py:          'cuda': ['numba']
python/tests/test-1-vector.py:    # Skip test for non-GPU backend
python/tests/test-1-vector.py:    if 'gpu' in ceed_resource:
python/ceed_qfunctioncontext.py:                data.__cuda_array_interface__['data'][0])
python/ceed_qfunctioncontext.py:            # CUDA array interface
python/ceed_qfunctioncontext.py:            # https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html
python/ceed_qfunctioncontext.py:            import numba.cuda as nbcuda
python/ceed_qfunctioncontext.py:            return nbcuda.from_cuda_array_interface(desc)
python/ceed_vector.py:                array.__cuda_array_interface__['data'][0])
python/ceed_vector.py:            # CUDA array interface
python/ceed_vector.py:            # https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html
python/ceed_vector.py:            import numba.cuda as nbcuda
python/ceed_vector.py:            return nbcuda.from_cuda_array_interface(desc)
python/ceed_vector.py:            # CUDA array interface
python/ceed_vector.py:            # https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html
python/ceed_vector.py:            import numba.cuda as nbcuda
python/ceed_vector.py:            return nbcuda.from_cuda_array_interface(desc)
python/ceed_vector.py:            # CUDA array interface
python/ceed_vector.py:            # https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html
python/ceed_vector.py:            import numba.cuda as nbcuda
python/ceed_vector.py:            return nbcuda.from_cuda_array_interface(desc)
rust/libceed-sys/c-src/backends/sycl/online_compiler.sycl.cpp:  if (DeviceType == sycl::info::device_type::gpu) {
rust/libceed-sys/c-src/backends/sycl/online_compiler.sycl.cpp:      case device_arch::gpu_gen9:
rust/libceed-sys/c-src/backends/sycl/online_compiler.sycl.cpp:      case device_arch::gpu_gen9_5:
rust/libceed-sys/c-src/backends/sycl/online_compiler.sycl.cpp:      case device_arch::gpu_gen11:
rust/libceed-sys/c-src/backends/sycl/online_compiler.sycl.cpp:      case device_arch::gpu_gen12:
rust/libceed-sys/c-src/backends/sycl/online_compiler.sycl.cpp:    // For now "tgllp" is used as the option supported on all known GPU RT.
rust/libceed-sys/c-src/backends/sycl/online_compiler.sycl.cpp:/// @param Source - Either OpenCL or CM source code.
rust/libceed-sys/c-src/backends/sycl/online_compiler.sycl.cpp:/// @param DeviceType - SYCL device type, e.g. cpu, gpu, accelerator, etc.
rust/libceed-sys/c-src/backends/sycl/online_compiler.sycl.cpp:std::vector<byte> online_compiler<source_language::opencl_c>::compile(const std::string &Source, const std::vector<std::string> &UserArgs) {
rust/libceed-sys/c-src/backends/sycl/ceed-sycl-compile.sycl.cpp:// Compile an OpenCL source to SPIR-V using Intel's online compiler extension
rust/libceed-sys/c-src/backends/sycl/ceed-sycl-compile.sycl.cpp:static inline int CeedJitCompileSource_Sycl(Ceed ceed, const sycl::device &sycl_device, const std::string &opencl_source, ByteVector_t &il_binary,
rust/libceed-sys/c-src/backends/sycl/ceed-sycl-compile.sycl.cpp:  sycl::ext::libceed::online_compiler<sycl::ext::libceed::source_language::opencl_c> compiler(sycl_device);
rust/libceed-sys/c-src/backends/sycl/ceed-sycl-compile.sycl.cpp:    il_binary = compiler.compile(opencl_source, flags);
rust/libceed-sys/c-src/backends/sycl/ceed-sycl-common.sycl.cpp:  if (std::strstr(resource, "/gpu/sycl")) {
rust/libceed-sys/c-src/backends/sycl/ceed-sycl-common.sycl.cpp:    device_type = sycl::info::device_type::gpu;
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:  // TODO1: the list must be extended with a bunch of new GPUs available.
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:  // TODO2: the list of supported GPUs grows rapidly.
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:  // The API must allow user to define the target GPU option even if it is
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:  enum gpu {
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:    gpu_any    = 1,
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:    gpu_gen9   = 2,
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:    gpu_skl    = gpu_gen9,
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:    gpu_gen9_5 = 3,
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:    gpu_kbl    = gpu_gen9_5,
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:    gpu_cfl    = gpu_gen9_5,
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:    gpu_gen11  = 4,
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:    gpu_icl    = gpu_gen11,
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:    gpu_gen12  = 5,
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:    gpu_tgl    = gpu_gen12,
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:    gpu_tgllp  = gpu_gen12
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:enum class source_language { opencl_c = 0, cm = 1 };
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:/// Compiles the given OpenCL source. May throw \c online_compile_error.
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:///   OpenCL JIT compiler options must be supported.
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:std::vector<byte> online_compiler<source_language::opencl_c>::compile(const std::string &src, const std::vector<std::string> &options);
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:// /// Compiles the given OpenCL source. May throw \c online_compile_error.
rust/libceed-sys/c-src/backends/sycl/online_compiler.hpp:// online_compiler<source_language::opencl_c>::compile(const std::string &src) {
rust/libceed-sys/c-src/backends/occa/ceed-occa-context.cpp:  _usingGpuDevice        = (mode == "CUDA" || mode == "HIP" || mode == "OpenCL");
rust/libceed-sys/c-src/backends/occa/ceed-occa-context.cpp:bool Context::usingGpuDevice() const { return _usingGpuDevice; }
rust/libceed-sys/c-src/backends/occa/ceed-occa-gpu-operator.hpp:#ifndef CEED_OCCA_GPU_OPERATOR_HEADER
rust/libceed-sys/c-src/backends/occa/ceed-occa-gpu-operator.hpp:#define CEED_OCCA_GPU_OPERATOR_HEADER
rust/libceed-sys/c-src/backends/occa/ceed-occa-gpu-operator.hpp:class GpuOperator : public Operator {
rust/libceed-sys/c-src/backends/occa/ceed-occa-gpu-operator.hpp:  GpuOperator();
rust/libceed-sys/c-src/backends/occa/ceed-occa-gpu-operator.hpp:  ~GpuOperator();
rust/libceed-sys/c-src/backends/occa/ceed-occa-tensor-basis.cpp:  if (usingGpuDevice()) {
rust/libceed-sys/c-src/backends/occa/ceed-occa-tensor-basis.cpp:  // TODO: Add gpu function sources when split
rust/libceed-sys/c-src/backends/occa/ceed-occa-tensor-basis.cpp:  const char *gpuKernelSources[3]   = {occa_tensor_basis_1d_gpu_source, occa_tensor_basis_2d_gpu_source, occa_tensor_basis_3d_gpu_source};
rust/libceed-sys/c-src/backends/occa/ceed-occa-tensor-basis.cpp:  if (usingGpuDevice()) {
rust/libceed-sys/c-src/backends/occa/ceed-occa-tensor-basis.cpp:    kernelSource = gpuKernelSources[dim - 1];
rust/libceed-sys/c-src/backends/occa/ceed-occa-tensor-basis.cpp:  if (Q1D < P1D && Context::from(ceed)->usingGpuDevice()) {
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:static std::string getDefaultDeviceMode(const bool cpuMode, const bool gpuMode) {
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  // In case both cpuMode and gpuMode are set, prioritize the GPU if available
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  if (gpuMode) {
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:    if (::occa::modeIsEnabled("CUDA")) {
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:      return "CUDA";
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:    if (::occa::modeIsEnabled("OpenCL")) {
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:      return "OpenCL";
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  if (match == "cuda") {
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:    mode = "CUDA";
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  if (match == "opencl") {
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:    mode = "OpenCL";
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  const bool gpuMode  = match == "gpu";
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  mode = getDefaultDeviceMode(cpuMode || autoMode, gpuMode || autoMode);
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:   *    "/gpu/occa?mode='CUDA':device_id=0"
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:   *    ["gpu", "occa"]
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:   *    "gpu"
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:   *      "mode": "'CUDA'",
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  // Check for /gpu/cuda/occa, /gpu/hip/occa, /cpu/self/occa, /cpu/openmp/occa
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  if (resource == "/gpu/cuda/occa") {
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:    match = "cuda";
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  if (resource == "/gpu/hip/occa") {
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  if (resource == "/gpu/dpcpp/occa") {
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  if (resource == "/gpu/opencl/occa") {
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:    match = "opencl";
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  if ((mode == "CUDA") || (mode == "HIP") || (mode == "dpcpp") || (mode == "OpenCL")) {
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  if ((mode == "dpcpp") || (mode == "OpenCL")) {
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  // GPU Modes
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  CeedCallBackend(CeedRegister("/gpu/dpcpp/occa", ceed::occa::registerBackend, 240));
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  CeedCallBackend(CeedRegister("/gpu/opencl/occa", ceed::occa::registerBackend, 230));
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  CeedCallBackend(CeedRegister("/gpu/hip/occa", ceed::occa::registerBackend, 220));
rust/libceed-sys/c-src/backends/occa/ceed-occa.cpp:  CeedCallBackend(CeedRegister("/gpu/cuda/occa", ceed::occa::registerBackend, 210));
rust/libceed-sys/c-src/backends/occa/ceed-occa-ceed-object.cpp:bool CeedObject::usingGpuDevice() const { return Context::from(ceed)->usingGpuDevice(); }
rust/libceed-sys/c-src/backends/occa/ceed-occa-ceed-object.hpp:  bool usingGpuDevice() const;
rust/libceed-sys/c-src/backends/occa/ceed-occa-context.hpp:  bool _usingGpuDevice;
rust/libceed-sys/c-src/backends/occa/ceed-occa-context.hpp:  bool usingGpuDevice() const;
rust/libceed-sys/c-src/backends/occa/kernels/tensor-basis.hpp:// Kernels are based on the cuda backend from LLNL and VT groups
rust/libceed-sys/c-src/backends/occa/kernels/tensor-basis.hpp:extern const char *occa_tensor_basis_1d_gpu_source;
rust/libceed-sys/c-src/backends/occa/kernels/tensor-basis.hpp:extern const char *occa_tensor_basis_2d_gpu_source;
rust/libceed-sys/c-src/backends/occa/kernels/tensor-basis.hpp:extern const char *occa_tensor_basis_3d_gpu_source;
rust/libceed-sys/c-src/backends/occa/kernels/elem-restriction.hpp:// Kernels are based on the cuda backend from LLNL and VT groups
rust/libceed-sys/c-src/backends/occa/kernels/simplex-basis/gpu-simplex-basis.cpp:const char *occa_simplex_basis_gpu_source = STRINGIFY_SOURCE(
rust/libceed-sys/c-src/backends/occa/kernels/simplex-basis.hpp:// Kernels are based on the cuda backend from LLNL and VT groups
rust/libceed-sys/c-src/backends/occa/kernels/simplex-basis.hpp:extern const char *occa_simplex_basis_gpu_source;
rust/libceed-sys/c-src/backends/occa/kernels/tensor-basis/gpu/tensor-basis-2d.cpp:const char *occa_tensor_basis_2d_gpu_source = STRINGIFY_SOURCE(
rust/libceed-sys/c-src/backends/occa/kernels/tensor-basis/gpu/tensor-basis-1d.cpp:const char *occa_tensor_basis_1d_gpu_source = STRINGIFY_SOURCE(
rust/libceed-sys/c-src/backends/occa/kernels/tensor-basis/gpu/tensor-basis-3d.cpp:const char *occa_tensor_basis_3d_gpu_source = STRINGIFY_SOURCE(
rust/libceed-sys/c-src/backends/occa/kernels/elem-restriction.cpp:// Kernels are based on the cuda backend from LLNL and VT groups
rust/libceed-sys/c-src/backends/occa/ceed-occa-simplex-basis.cpp:  // TODO: Add gpu function sources when split
rust/libceed-sys/c-src/backends/occa/ceed-occa-simplex-basis.cpp:  if (usingGpuDevice()) {
rust/libceed-sys/c-src/backends/occa/ceed-occa-simplex-basis.cpp:  if (usingGpuDevice()) {
rust/libceed-sys/c-src/backends/occa/ceed-occa-simplex-basis.cpp:    kernelSource = occa_simplex_basis_gpu_source;
rust/libceed-sys/c-src/backends/occa/ceed-occa-gpu-operator.cpp:#include "ceed-occa-gpu-operator.hpp"
rust/libceed-sys/c-src/backends/occa/ceed-occa-gpu-operator.cpp:GpuOperator::GpuOperator() {}
rust/libceed-sys/c-src/backends/occa/ceed-occa-gpu-operator.cpp:GpuOperator::~GpuOperator() {}
rust/libceed-sys/c-src/backends/occa/ceed-occa-gpu-operator.cpp:::occa::kernel GpuOperator::buildApplyAddKernel() { return ::occa::kernel(); }
rust/libceed-sys/c-src/backends/occa/ceed-occa-gpu-operator.cpp:void GpuOperator::applyAdd(Vector *in, Vector *out) {
rust/libceed-sys/c-src/backends/occa/ceed-occa-operator.cpp:#include "ceed-occa-gpu-operator.hpp"
rust/libceed-sys/c-src/backends/occa/ceed-occa-operator.cpp:  // TODO: Add GPU specific operator
rust/libceed-sys/c-src/backends/occa/ceed-occa-operator.cpp:  Operator *operator_ = (Context::from(ceed)->usingCpuDevice() ? ((Operator *)new CpuOperator()) : ((Operator *)new GpuOperator()));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared.c:#include "ceed-cuda-shared.h"
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared.c:#include "../cuda/ceed-cuda-common.h"
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared.c:static int CeedInit_Cuda_shared(const char *resource, Ceed ceed) {
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared.c:  Ceed_Cuda *data;
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared.c:  CeedCheck(!strcmp(resource_root, "/gpu/cuda/shared"), ceed, CEED_ERROR_BACKEND, "Cuda backend cannot use resource: %s", resource);
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared.c:  CeedCallBackend(CeedInit_Cuda(ceed, resource));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared.c:  CeedCallBackend(CeedInit("/gpu/cuda/ref", &ceed_ref));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1", CeedBasisCreateTensorH1_Cuda_shared));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Cuda));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared.c:CEED_INTERN int CeedRegister_Cuda_Shared(void) { return CeedRegister("/gpu/cuda/shared", CeedInit_Cuda_shared, 25); }
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:#include <cuda.h>
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:#include <cuda_runtime.h>
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:#include "../cuda/ceed-cuda-common.h"
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:#include "../cuda/ceed-cuda-compile.h"
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:#include "ceed-cuda-shared.h"
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:int CeedInit_CudaInterp(CeedScalar *d_B, CeedInt P_1d, CeedInt Q_1d, CeedScalar **c_B);
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:int CeedInit_CudaGrad(CeedScalar *d_B, CeedScalar *d_G, CeedInt P_1d, CeedInt Q_1d, CeedScalar **c_B_ptr, CeedScalar **c_G_ptr);
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:int CeedInit_CudaCollocatedGrad(CeedScalar *d_B, CeedScalar *d_G, CeedInt P_1d, CeedInt Q_1d, CeedScalar **c_B_ptr, CeedScalar **c_G_ptr);
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:static int CeedBasisApplyTensorCore_Cuda_shared(CeedBasis basis, bool apply_add, const CeedInt num_elem, CeedTransposeMode t_mode,
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  Ceed_Cuda             *ceed_Cuda;
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedBasis_Cuda_shared *data;
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedGetData(ceed, &ceed_Cuda));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:      CeedCallBackend(CeedInit_CudaInterp(data->d_interp_1d, P_1d, Q_1d, &data->c_B));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:        CeedInt elems_per_block = CeedIntMin(ceed_Cuda->device_prop.maxThreadsDim[2], CeedIntMax(512 / thread_1d,
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->InterpTransposeAdd : data->InterpTranspose, grid, thread_1d, 1,
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Interp, grid, thread_1d, 1, elems_per_block, shared_mem, interp_args));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->InterpTransposeAdd : data->InterpTranspose, grid, thread_1d, thread_1d,
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Interp, grid, thread_1d, thread_1d, elems_per_block, shared_mem, interp_args));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->InterpTransposeAdd : data->InterpTranspose, grid, thread_1d, thread_1d,
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Interp, grid, thread_1d, thread_1d, elems_per_block, shared_mem, interp_args));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:        CeedCallBackend(CeedInit_CudaCollocatedGrad(data->d_interp_1d, data->d_collo_grad_1d, P_1d, Q_1d, &data->c_B, &data->c_G));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:        CeedCallBackend(CeedInit_CudaGrad(data->d_interp_1d, data->d_grad_1d, P_1d, Q_1d, &data->c_B, &data->c_G));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:        CeedInt elems_per_block = CeedIntMin(ceed_Cuda->device_prop.maxThreadsDim[2], CeedIntMax(512 / thread_1d,
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->GradTransposeAdd : data->GradTranspose, grid, thread_1d, 1,
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Grad, grid, thread_1d, 1, elems_per_block, shared_mem, grad_args));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->GradTransposeAdd : data->GradTranspose, grid, thread_1d, thread_1d,
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Grad, grid, thread_1d, thread_1d, elems_per_block, shared_mem, grad_args));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->GradTransposeAdd : data->GradTranspose, grid, thread_1d, thread_1d,
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Grad, grid, thread_1d, thread_1d, elems_per_block, shared_mem, grad_args));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, grid_size, Q_1d, elems_per_block, 1, weight_args));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, grid_size, Q_1d, Q_1d, elems_per_block, weight_args));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, grid_size, Q_1d, Q_1d, elems_per_block, weight_args));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:static int CeedBasisApplyTensor_Cuda_shared(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedBasisApplyTensorCore_Cuda_shared(basis, false, num_elem, t_mode, eval_mode, u, v));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:static int CeedBasisApplyAddTensor_Cuda_shared(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode,
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedBasisApplyTensorCore_Cuda_shared(basis, true, num_elem, t_mode, eval_mode, u, v));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:static int CeedBasisApplyAtPointsCore_Cuda_shared(CeedBasis basis, bool apply_add, const CeedInt num_elem, const CeedInt *num_points,
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedBasis_Cuda_shared *data;
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:              "Vector at points must be padded to the same number of points in each element for BasisApplyAtPoints on GPU backends."
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:      if (data->d_points_per_elem) CeedCallCuda(ceed, cudaFree(data->d_points_per_elem));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:      CeedCallCuda(ceed, cudaMalloc((void **)&data->d_points_per_elem, num_bytes));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:      CeedCallCuda(ceed, cudaMemcpy(data->d_points_per_elem, num_points, num_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:      CeedCallCuda(ceed, cudaMalloc((void **)&data->d_chebyshev_interp_1d, interp_bytes));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:      CeedCallCuda(ceed, cudaMemcpy(data->d_chebyshev_interp_1d, chebyshev_interp_1d, interp_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:    const char basis_kernel_source[] = "// AtPoints basis source\n#include <ceed/jit-source/cuda/cuda-ref-basis-tensor-at-points.h>\n";
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:    if (data->moduleAtPoints) CeedCallCuda(ceed, cuModuleUnload(data->moduleAtPoints));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:    CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->moduleAtPoints, 9, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "BASIS_BUF_LEN",
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:    CeedCallBackend(CeedGetKernel_Cuda(ceed, data->moduleAtPoints, "InterpAtPoints", &data->InterpAtPoints));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:    CeedCallBackend(CeedGetKernel_Cuda(ceed, data->moduleAtPoints, "GradAtPoints", &data->GradAtPoints));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:    CeedCallCuda(ceed, cudaMemset(d_v, 0, length * sizeof(CeedScalar)));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:      CeedCallBackend(CeedRunKernel_Cuda(ceed, data->InterpAtPoints, num_elem, block_size, interp_args));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:      CeedCallBackend(CeedRunKernel_Cuda(ceed, data->GradAtPoints, num_elem, block_size, grad_args));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:static int CeedBasisApplyAtPoints_Cuda_shared(CeedBasis basis, const CeedInt num_elem, const CeedInt *num_points, CeedTransposeMode t_mode,
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedBasisApplyAtPointsCore_Cuda_shared(basis, false, num_elem, num_points, t_mode, eval_mode, x_ref, u, v));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:static int CeedBasisApplyAddAtPoints_Cuda_shared(CeedBasis basis, const CeedInt num_elem, const CeedInt *num_points, CeedTransposeMode t_mode,
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedBasisApplyAtPointsCore_Cuda_shared(basis, true, num_elem, num_points, t_mode, eval_mode, x_ref, u, v));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:static int CeedBasisDestroy_Cuda_shared(CeedBasis basis) {
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedBasis_Cuda_shared *data;
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cuModuleUnload(data->module));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  if (data->moduleAtPoints) CeedCallCuda(ceed, cuModuleUnload(data->moduleAtPoints));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  if (data->d_q_weight_1d) CeedCallCuda(ceed, cudaFree(data->d_q_weight_1d));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  if (data->d_points_per_elem) CeedCallCuda(ceed, cudaFree(data->d_points_per_elem));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_interp_1d));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_grad_1d));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_collo_grad_1d));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_chebyshev_interp_1d));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:int CeedBasisCreateTensorH1_Cuda_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedBasis_Cuda_shared *data;
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  // Copy basis data to GPU
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight_1d, q_bytes));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight_1d, q_weight_1d, q_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp_1d, interp_bytes));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cudaMemcpy(data->d_interp_1d, interp_1d, interp_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_grad_1d, interp_bytes));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cudaMemcpy(data->d_grad_1d, grad_1d, interp_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  // Compute collocated gradient and copy to GPU
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_collo_grad_1d, q_bytes * Q_1d));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_collo_grad_1d, collo_grad_1d, q_bytes * Q_1d, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  const char basis_kernel_source[] = "// Tensor basis source\n#include <ceed/jit-source/cuda/cuda-shared-basis-tensor.h>\n";
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->module, 8, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "T_1D",
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Interp", &data->Interp));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTransposeAdd", &data->InterpTransposeAdd));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Grad", &data->Grad));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "GradTranspose", &data->GradTranspose));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "GradTransposeAdd", &data->GradTransposeAdd));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyTensor_Cuda_shared));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddTensor_Cuda_shared));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAtPoints", CeedBasisApplyAtPoints_Cuda_shared));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAddAtPoints", CeedBasisApplyAddAtPoints_Cuda_shared));
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Cuda_shared));
rust/libceed-sys/c-src/backends/cuda-shared/kernels/cuda-shared-basis.cu:#include <cuda.h>
rust/libceed-sys/c-src/backends/cuda-shared/kernels/cuda-shared-basis.cu:extern "C" int CeedInit_CudaInterp(CeedScalar *d_B, CeedInt P_1d, CeedInt Q_1d, CeedScalar **c_B_ptr) {
rust/libceed-sys/c-src/backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaMemcpyToSymbol(c_B, d_B, bytes, 0, cudaMemcpyDeviceToDevice);
rust/libceed-sys/c-src/backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaGetSymbolAddress((void **)c_B_ptr, c_B);
rust/libceed-sys/c-src/backends/cuda-shared/kernels/cuda-shared-basis.cu:extern "C" int CeedInit_CudaGrad(CeedScalar *d_B, CeedScalar *d_G, CeedInt P_1d, CeedInt Q_1d, CeedScalar **c_B_ptr, CeedScalar **c_G_ptr) {
rust/libceed-sys/c-src/backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaMemcpyToSymbol(c_B, d_B, bytes, 0, cudaMemcpyDeviceToDevice);
rust/libceed-sys/c-src/backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaGetSymbolAddress((void **)c_B_ptr, c_B);
rust/libceed-sys/c-src/backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaMemcpyToSymbol(c_G, d_G, bytes, 0, cudaMemcpyDeviceToDevice);
rust/libceed-sys/c-src/backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaGetSymbolAddress((void **)c_G_ptr, c_G);
rust/libceed-sys/c-src/backends/cuda-shared/kernels/cuda-shared-basis.cu:extern "C" int CeedInit_CudaCollocatedGrad(CeedScalar *d_B, CeedScalar *d_G, CeedInt P_1d, CeedInt Q_1d, CeedScalar **c_B_ptr, CeedScalar **c_G_ptr) {
rust/libceed-sys/c-src/backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaMemcpyToSymbol(c_B, d_B, bytes_interp, 0, cudaMemcpyDeviceToDevice);
rust/libceed-sys/c-src/backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaGetSymbolAddress((void **)c_B_ptr, c_B);
rust/libceed-sys/c-src/backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaMemcpyToSymbol(c_G, d_G, bytes_grad, 0, cudaMemcpyDeviceToDevice);
rust/libceed-sys/c-src/backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaGetSymbolAddress((void **)c_G_ptr, c_G);
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared.h:#include <cuda.h>
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared.h:} CeedBasis_Cuda_shared;
rust/libceed-sys/c-src/backends/cuda-shared/ceed-cuda-shared.h:CEED_INTERN int CeedBasisCreateTensorH1_Cuda_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
rust/libceed-sys/c-src/backends/sycl-gen/ceed-sycl-gen-qfunction.sycl.cpp:  CeedCheck(impl->qfunction_source, ceed, CEED_ERROR_UNSUPPORTED, "/gpu/sycl/gen backend requires QFunction source code file");
rust/libceed-sys/c-src/backends/sycl-gen/ceed-sycl-gen.sycl.cpp:  const char fallback_resource[] = "/gpu/sycl/ref";
rust/libceed-sys/c-src/backends/sycl-gen/ceed-sycl-gen.sycl.cpp:  CeedCheck(!strcmp(resource_root, "/gpu/sycl") || !strcmp(resource_root, "/gpu/sycl/gen"), ceed, CEED_ERROR_BACKEND,
rust/libceed-sys/c-src/backends/sycl-gen/ceed-sycl-gen.sycl.cpp:  CeedCallBackend(CeedInit("/gpu/sycl/shared", &ceed_shared));
rust/libceed-sys/c-src/backends/sycl-gen/ceed-sycl-gen.sycl.cpp:CEED_INTERN int CeedRegister_Sycl_Gen(void) { return CeedRegister("/gpu/sycl/gen", CeedInit_Sycl_gen, 20); }
rust/libceed-sys/c-src/backends/hip/ceed-hip-compile.cpp:  // Add hip runtime include statement for generation if runtime < 40400000 (implies ROCm < 4.5)
rust/libceed-sys/c-src/backends/hip/ceed-hip-compile.cpp:  // With ROCm 4.5, need to include these definitions specifically for hiprtc (but cannot include the runtime header)
rust/libceed-sys/c-src/backends/hip/ceed-hip-compile.cpp:  std::string arch_arg = "--gpu-architecture=" + std::string(prop.gcnArchName);
rust/libceed-sys/c-src/backends/sycl-shared/ceed-sycl-shared-basis.sycl.cpp:  // Copy basis data to GPU
rust/libceed-sys/c-src/backends/sycl-shared/ceed-sycl-shared-basis.sycl.cpp:  // Compute collocated gradient and copy to GPU
rust/libceed-sys/c-src/backends/sycl-shared/ceed-sycl-shared.sycl.cpp:  CeedCheck(!std::strcmp(resource_root, "/gpu/sycl/shared") || !std::strcmp(resource_root, "/cpu/sycl/shared"), ceed, CEED_ERROR_BACKEND,
rust/libceed-sys/c-src/backends/sycl-shared/ceed-sycl-shared.sycl.cpp:  CeedCallBackend(CeedRegister("/gpu/sycl/shared", CeedInit_Sycl_shared, 25));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:#include <cuda_runtime.h>
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:#include "../cuda-ref/ceed-cuda-ref.h"
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:#include "../cuda-shared/ceed-cuda-shared.h"
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:#include "../cuda/ceed-cuda-common.h"
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:#include "../cuda/ceed-cuda-compile.h"
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:#include "ceed-cuda-gen.h"
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:static int CeedOperatorBuildKernelData_Cuda_gen(Ceed ceed, CeedInt num_input_fields, CeedOperatorField *op_input_fields,
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:        CeedBasis_Cuda_shared *basis_data;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:        CeedBasis_Cuda_shared *basis_data;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:static int CeedOperatorBuildKernelFieldData_Cuda_gen(std::ostringstream &code, CeedOperator_Cuda_gen *data, CeedInt i, CeedOperatorField op_field,
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  CeedBasis_Cuda_shared *basis_data;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:static int CeedOperatorBuildKernelRestriction_Cuda_gen(std::ostringstream &code, CeedOperator_Cuda_gen *data, CeedInt i, CeedInt dim,
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  CeedElemRestriction_Cuda *rstr_data;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:static int CeedOperatorBuildKernelBasis_Cuda_gen(std::ostringstream &code, CeedOperator_Cuda_gen *data, CeedInt i, CeedInt dim,
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:        CeedBasis_Cuda_shared *basis_data;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:static int CeedOperatorBuildKernelQFunction_Cuda_gen(std::ostringstream &code, CeedOperator_Cuda_gen *data, CeedInt dim, CeedInt num_input_fields,
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:            CeedElemRestriction_Cuda *rstr_data;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:extern "C" int CeedOperatorBuildKernel_Cuda_gen(CeedOperator op) {
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  CeedQFunction_Cuda_gen *qf_data;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  CeedOperator_Cuda_gen  *data;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  CeedCallBackend(CeedOperatorBuildKernelData_Cuda_gen(ceed, num_input_fields, op_input_fields, qf_input_fields, num_output_fields, op_output_fields,
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  // Add atomicAdd function for old NVidia architectures
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    Ceed_Cuda            *ceed_data;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    struct cudaDeviceProp prop;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    CeedCallBackend(cudaGetDeviceProperties(&prop, ceed_data->device_id));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:      code << "#include <ceed/jit-source/cuda/cuda-atomic-add-fallback.h>\n\n";
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  code << "#include <ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h>\n\n";
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  code << "#include <ceed/jit-source/cuda/cuda-gen-templates.h>\n\n";
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  operator_name = "CeedKernelCudaGenOperator_" + qfunction_name;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    CeedCheck(source_path, ceed, CEED_ERROR_UNSUPPORTED, "/gpu/cuda/gen backend requires QFunction source code file");
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:       << "(CeedInt num_elem, void* ctx, FieldsInt_Cuda indices, Fields_Cuda fields, Fields_Cuda B, Fields_Cuda G, CeedScalar *W) {\n";
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  code << "  SharedData_Cuda data;\n";
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    CeedCallBackend(CeedOperatorBuildKernelFieldData_Cuda_gen(code, data, i, op_input_fields[i], qf_input_fields[i], Q_1d, true, use_3d_slices));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    CeedCallBackend(CeedOperatorBuildKernelFieldData_Cuda_gen(code, data, i, op_output_fields[i], qf_output_fields[i], Q_1d, false, use_3d_slices));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    CeedCallBackend(CeedOperatorBuildKernelRestriction_Cuda_gen(code, data, f, dim, field_rstr_in_buffer, op_input_fields[f], qf_input_fields[f],
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    CeedCallBackend(CeedOperatorBuildKernelBasis_Cuda_gen(code, data, f, dim, op_input_fields[f], qf_input_fields[f], Q_1d, true, use_3d_slices));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  CeedCallBackend(CeedOperatorBuildKernelQFunction_Cuda_gen(code, data, dim, num_input_fields, op_input_fields, qf_input_fields, num_output_fields,
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    CeedCallBackend(CeedOperatorBuildKernelBasis_Cuda_gen(code, data, i, dim, op_output_fields[i], qf_output_fields[i], Q_1d, false, use_3d_slices));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:        CeedOperatorBuildKernelRestriction_Cuda_gen(code, data, i, dim, NULL, op_output_fields[i], qf_output_fields[i], Q_1d, false, use_3d_slices));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  CeedCallBackend(CeedCompile_Cuda(ceed, code.str().c_str(), &data->module, 1, "T_1D", CeedIntMax(Q_1d, data->max_P_1d)));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, operator_name.c_str(), &data->op));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator-build.h:CEED_INTERN int CeedOperatorBuildKernel_Cuda_gen(CeedOperator op);
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:#include <ceed/jit-source/cuda/cuda-types.h>
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:#include "../cuda/ceed-cuda-common.h"
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:#include "../cuda/ceed-cuda-compile.h"
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:#include "ceed-cuda-gen-operator-build.h"
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:#include "ceed-cuda-gen.h"
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:static int CeedOperatorDestroy_Cuda_gen(CeedOperator op) {
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedOperator_Cuda_gen *impl;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:// Problem setting: we'd like to make occupancy high with relatively few inactive threads. CUDA (cuOccupancyMaxPotentialBlockSize) can tell us how
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:// block from running. The cuda-gen kernels are pretty heavy with lots of instruction-level parallelism (ILP) so we'll generally be okay with
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:// cuda-gen can't choose block sizes arbitrarily; they need to be a multiple of the number of quadrature points (or number of basis functions).
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:// CUDA schedules in units of full warps (32 threads), so 128 CUDA hardware threads are effectively committed to that block.
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:static int CeedOperatorApplyAdd_Cuda_gen(CeedOperator op, CeedVector input_vec, CeedVector output_vec, CeedRequest *request) {
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:  Ceed_Cuda              *cuda_data;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedQFunction_Cuda_gen *qf_data;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedOperator_Cuda_gen  *data;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:      CeedDebug256(CeedOperatorReturnCeed(op), CEED_DEBUG_COLOR_SUCCESS, "Falling back to /gpu/cuda/ref CeedOperator due to non-tensor bases");
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedCallBackend(CeedGetData(ceed, &cuda_data));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedCallBackend(CeedOperatorBuildKernel_Cuda_gen(op));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedCallCuda(ceed, cuOccupancyMaxPotentialBlockSize(&min_grid_size, &max_threads_per_block, data->op, dynamicSMemSize, 0, 0x10000));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedCallBackend(BlockGridCalculate(num_elem, min_grid_size / cuda_data->device_prop.multiProcessorCount, max_threads_per_block,
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:                                     cuda_data->device_prop.maxThreadsDim[2], cuda_data->device_prop.warpSize, block, &grid));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->op, grid, block[0], block[1], block[2], shared_mem, opargs));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:int CeedOperatorCreate_Cuda_gen(CeedOperator op) {
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedOperator_Cuda_gen *impl;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Cuda_gen));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Cuda_gen));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.h:#include <ceed/jit-source/cuda/cuda-types.h>
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.h:#include <cuda.h>
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.h:  FieldsInt_Cuda indices;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.h:  Fields_Cuda    fields;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.h:  Fields_Cuda    B;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.h:  Fields_Cuda    G;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.h:} CeedOperator_Cuda_gen;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.h:} CeedQFunction_Cuda_gen;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.h:CEED_INTERN int CeedQFunctionCreate_Cuda_gen(CeedQFunction qf);
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.h:CEED_INTERN int CeedOperatorCreate_Cuda_gen(CeedOperator op);
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.c:#include "ceed-cuda-gen.h"
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.c:#include "../cuda/ceed-cuda-common.h"
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.c:static int CeedInit_Cuda_gen(const char *resource, Ceed ceed) {
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.c:  const char fallback_resource[] = "/gpu/cuda/ref";
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.c:  Ceed_Cuda *data;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.c:  CeedCheck(!strcmp(resource_root, "/gpu/cuda") || !strcmp(resource_root, "/gpu/cuda/gen"), ceed, CEED_ERROR_BACKEND,
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.c:            "Cuda backend cannot use resource: %s", resource);
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.c:  CeedCallBackend(CeedInit_Cuda(ceed, resource));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.c:  CeedCallBackend(CeedInit("/gpu/cuda/shared", &ceed_shared));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate", CeedQFunctionCreate_Cuda_gen));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate", CeedOperatorCreate_Cuda_gen));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Cuda));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen.c:CEED_INTERN int CeedRegister_Cuda_Gen(void) { return CeedRegister("/gpu/cuda/gen", CeedInit_Cuda_gen, 20); }
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-qfunction.c:#include <cuda_runtime.h>
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-qfunction.c:#include "../cuda/ceed-cuda-common.h"
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-qfunction.c:#include "ceed-cuda-gen.h"
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-qfunction.c:static int CeedQFunctionApply_Cuda_gen(CeedQFunction qf, CeedInt Q, CeedVector *U, CeedVector *V) {
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-qfunction.c:static int CeedQFunctionDestroy_Cuda_gen(CeedQFunction qf) {
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-qfunction.c:  CeedQFunction_Cuda_gen *data;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-qfunction.c:  CeedCallCuda(CeedQFunctionReturnCeed(qf), cudaFree(data->d_c));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-qfunction.c:int CeedQFunctionCreate_Cuda_gen(CeedQFunction qf) {
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-qfunction.c:  CeedQFunction_Cuda_gen *data;
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-qfunction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Apply", CeedQFunctionApply_Cuda_gen));
rust/libceed-sys/c-src/backends/cuda-gen/ceed-cuda-gen-qfunction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy", CeedQFunctionDestroy_Cuda_gen));
rust/libceed-sys/c-src/backends/sycl-ref/ceed-sycl-ref-qfunction.sycl.cpp:    // Equivalent of CUDA Occupancy Calculator
rust/libceed-sys/c-src/backends/sycl-ref/ceed-sycl-ref-qfunction-load.sycl.cpp:  // OpenCL doesn't allow for structs with pointers.
rust/libceed-sys/c-src/backends/sycl-ref/ceed-sycl-ref.sycl.cpp:  CeedCheck(!std::strcmp(resource_root, "/gpu/sycl/ref") || !std::strcmp(resource_root, "/cpu/sycl/ref"), ceed, CEED_ERROR_BACKEND,
rust/libceed-sys/c-src/backends/sycl-ref/ceed-sycl-ref.sycl.cpp:  CeedCallBackend(CeedRegister("/gpu/sycl/ref", CeedInit_Sycl_ref, 40));
rust/libceed-sys/c-src/backends/hip-ref/ceed-hip-ref-basis.c:              "Vector at points must be padded to the same number of points in each element for BasisApplyAtPoints on GPU backends."
rust/libceed-sys/c-src/backends/hip-ref/ceed-hip-ref-basis.c:  // Copy data to GPU
rust/libceed-sys/c-src/backends/hip-ref/ceed-hip-ref-basis.c:  // Copy basis data to GPU
rust/libceed-sys/c-src/backends/hip-ref/ceed-hip-ref-basis.c:  // Copy basis data to GPU
rust/libceed-sys/c-src/backends/hip-ref/ceed-hip-ref-basis.c:  // Copy basis data to GPU
rust/libceed-sys/c-src/backends/hip-ref/ceed-hip-ref.c:  CeedCheck(!strcmp(resource_root, "/gpu/hip/ref"), ceed, CEED_ERROR_BACKEND, "Hip backend cannot use resource: %s", resource);
rust/libceed-sys/c-src/backends/hip-ref/ceed-hip-ref.c:CEED_INTERN int CeedRegister_Hip(void) { return CeedRegister("/gpu/hip/ref", CeedInit_Hip_ref, 40); }
rust/libceed-sys/c-src/backends/memcheck/ceed-memcheck-restriction.c:      // GPU default, contiguous by node, then element
rust/libceed-sys/c-src/backends/hip-shared/ceed-hip-shared-basis.c:              "Vector at points must be padded to the same number of points in each element for BasisApplyAtPoints on GPU backends."
rust/libceed-sys/c-src/backends/hip-shared/ceed-hip-shared-basis.c:  // Copy basis data to GPU
rust/libceed-sys/c-src/backends/hip-shared/ceed-hip-shared-basis.c:  // Compute collocated gradient and copy to GPU
rust/libceed-sys/c-src/backends/hip-shared/ceed-hip-shared.c:  CeedCheck(!strcmp(resource_root, "/gpu/hip/shared"), ceed, CEED_ERROR_BACKEND, "Hip backend cannot use resource: %s", resource);
rust/libceed-sys/c-src/backends/hip-shared/ceed-hip-shared.c:  CeedCallBackend(CeedInit("/gpu/hip/ref", &ceed_ref));
rust/libceed-sys/c-src/backends/hip-shared/ceed-hip-shared.c:CEED_INTERN int CeedRegister_Hip_Shared(void) { return CeedRegister("/gpu/hip/shared", CeedInit_Hip_shared, 25); }
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:#include "ceed-cuda-compile.h"
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:#include <cuda_runtime.h>
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:#include "ceed-cuda-common.h"
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:// Compile CUDA kernel
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:int CeedCompile_Cuda(Ceed ceed, const char *source, CUmodule *module, const CeedInt num_defines, ...) {
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:  struct cudaDeviceProp prop;
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:  Ceed_Cuda            *ceed_data;
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:  cudaFree(0);  // Make sure a Context exists for nvrtc
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:  // Standard libCEED definitions for CUDA backends
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:  code << "#include <ceed/jit-source/cuda/cuda-jit.h>\n\n";
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:  CeedCallCuda(ceed, cudaGetDeviceProperties(&prop, ceed_data->device_id));
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:#if CUDA_VERSION >= 11010
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:      // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#dynamic-code-generation
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:#if CUDA_VERSION >= 11010
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:  CeedCallCuda(ceed, cuModuleLoadData(module, ptx));
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:// Get CUDA kernel
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:int CeedGetKernel_Cuda(Ceed ceed, CUmodule module, const char *name, CUfunction *kernel) {
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:  CeedCallCuda(ceed, cuModuleGetFunction(kernel, module, name));
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:// Run CUDA kernel with block size selected automatically based on the kernel
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:int CeedRunKernelAutoblockCuda(Ceed ceed, CUfunction kernel, size_t points, void **args) {
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:  CeedCallCuda(ceed, cuOccupancyMaxPotentialBlockSize(&min_grid_size, &max_block_size, kernel, NULL, 0, 0x10000));
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:  CeedCallBackend(CeedRunKernel_Cuda(ceed, kernel, CeedDivUpInt(points, max_block_size), max_block_size, args));
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:// Run CUDA kernel
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:int CeedRunKernel_Cuda(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size, void **args) {
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:  CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, kernel, grid_size, block_size, 1, 1, 0, args));
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:// Run CUDA kernel for spatial dimension
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:int CeedRunKernelDim_Cuda(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size_x, const int block_size_y, const int block_size_z,
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:  CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, kernel, grid_size, block_size_x, block_size_y, block_size_z, 0, args));
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:// Run CUDA kernel for spatial dimension with shared memory
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:int CeedRunKernelDimShared_Cuda(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size_x, const int block_size_y,
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:#if CUDA_VERSION >= 9000
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:  if (result == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES) {
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.cpp:                     "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: max_threads_per_block %d on block size (%d,%d,%d), shared_size %d, num_regs %d",
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.h:#include <cuda.h>
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.h:CEED_INTERN int CeedCompile_Cuda(Ceed ceed, const char *source, CUmodule *module, const CeedInt num_defines, ...);
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.h:CEED_INTERN int CeedGetKernel_Cuda(Ceed ceed, CUmodule module, const char *name, CUfunction *kernel);
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.h:CEED_INTERN int CeedRunKernel_Cuda(Ceed ceed, CUfunction kernel, int grid_size, int block_size, void **args);
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.h:CEED_INTERN int CeedRunKernelAutoblockCuda(Ceed ceed, CUfunction kernel, size_t points, void **args);
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.h:CEED_INTERN int CeedRunKernelDim_Cuda(Ceed ceed, CUfunction kernel, int grid_size, int block_size_x, int block_size_y, int block_size_z, void **args);
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-compile.h:CEED_INTERN int CeedRunKernelDimShared_Cuda(Ceed ceed, CUfunction kernel, int grid_size, int block_size_x, int block_size_y, int block_size_z,
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.h:#include <cuda.h>
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.h:    CUresult cuda_result = (CUresult)x;                  \
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.h:    if (cuda_result != CUDA_SUCCESS) {                   \
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.h:      cuGetErrorName(cuda_result, &msg);                 \
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.h:#define CeedCallCuda(ceed, ...) \
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.h:  struct cudaDeviceProp device_prop;
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.h:} Ceed_Cuda;
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.h:CEED_INTERN int CeedInit_Cuda(Ceed ceed, const char *resource);
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.h:CEED_INTERN int CeedDestroy_Cuda(Ceed ceed);
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.h:CEED_INTERN int CeedSetDeviceBoolArray_Cuda(Ceed ceed, const bool *source_array, CeedCopyMode copy_mode, CeedSize num_values,
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.h:CEED_INTERN int CeedSetDeviceCeedInt8Array_Cuda(Ceed ceed, const CeedInt8 *source_array, CeedCopyMode copy_mode, CeedSize num_values,
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.h:CEED_INTERN int CeedSetDeviceCeedIntArray_Cuda(Ceed ceed, const CeedInt *source_array, CeedCopyMode copy_mode, CeedSize num_values,
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.h:CEED_INTERN int CeedSetDeviceCeedScalarArray_Cuda(Ceed ceed, const CeedScalar *source_array, CeedCopyMode copy_mode, CeedSize num_values,
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:#include "ceed-cuda-common.h"
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:#include <cuda_runtime.h>
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:int CeedInit_Cuda(Ceed ceed, const char *resource) {
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:  Ceed_Cuda  *data;
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:  CeedCallCuda(ceed, cudaGetDevice(&current_device_id));
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:    CeedCallCuda(ceed, cudaSetDevice(device_id));
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:  CeedCallCuda(ceed, cudaGetDeviceProperties(&data->device_prop, current_device_id));
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:int CeedDestroy_Cuda(Ceed ceed) {
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:  Ceed_Cuda *data;
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:static inline int CeedSetDeviceGenericArray_Cuda(Ceed ceed, const void *source_array, CeedCopyMode copy_mode, size_t size_unit, CeedSize num_values,
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:      if (!*(void **)target_array_owned) CeedCallCuda(ceed, cudaMalloc(target_array_owned, size_unit * num_values));
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:      if (source_array) CeedCallCuda(ceed, cudaMemcpy(*(void **)target_array_owned, source_array, size_unit * num_values, cudaMemcpyDeviceToDevice));
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:      CeedCallCuda(ceed, cudaFree(*(void **)target_array_owned));
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:      CeedCallCuda(ceed, cudaFree(*(void **)target_array_owned));
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:int CeedSetDeviceBoolArray_Cuda(Ceed ceed, const bool *source_array, CeedCopyMode copy_mode, CeedSize num_values, const bool **target_array_owned,
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:  CeedCallBackend(CeedSetDeviceGenericArray_Cuda(ceed, source_array, copy_mode, sizeof(bool), num_values, target_array_owned, target_array_borrowed,
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:int CeedSetDeviceCeedInt8Array_Cuda(Ceed ceed, const CeedInt8 *source_array, CeedCopyMode copy_mode, CeedSize num_values,
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:  CeedCallBackend(CeedSetDeviceGenericArray_Cuda(ceed, source_array, copy_mode, sizeof(CeedInt8), num_values, target_array_owned,
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:int CeedSetDeviceCeedIntArray_Cuda(Ceed ceed, const CeedInt *source_array, CeedCopyMode copy_mode, CeedSize num_values,
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:  CeedCallBackend(CeedSetDeviceGenericArray_Cuda(ceed, source_array, copy_mode, sizeof(CeedInt), num_values, target_array_owned,
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:int CeedSetDeviceCeedScalarArray_Cuda(Ceed ceed, const CeedScalar *source_array, CeedCopyMode copy_mode, CeedSize num_values,
rust/libceed-sys/c-src/backends/cuda/ceed-cuda-common.c:  CeedCallBackend(CeedSetDeviceGenericArray_Cuda(ceed, source_array, copy_mode, sizeof(CeedScalar), num_values, target_array_owned,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:#include <cuda.h>
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:#include <cuda_runtime.h>
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:#include "../cuda/ceed-cuda-common.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:#include "../cuda/ceed-cuda-compile.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:#include "ceed-cuda-ref.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApplyCore_Cuda(CeedBasis basis, bool apply_add, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasis_Cuda   *data;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemset(d_v, 0, length * sizeof(CeedScalar)));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallBackend(CeedRunKernel_Cuda(ceed, data->Interp, num_elem, block_size, interp_args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallBackend(CeedRunKernel_Cuda(ceed, data->Grad, num_elem, block_size, grad_args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, num_elem, block_size_x, block_size_y, 1, weight_args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApply_Cuda(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedBasisApplyCore_Cuda(basis, false, num_elem, t_mode, eval_mode, u, v));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApplyAdd_Cuda(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedBasisApplyCore_Cuda(basis, true, num_elem, t_mode, eval_mode, u, v));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApplyAtPointsCore_Cuda(CeedBasis basis, bool apply_add, const CeedInt num_elem, const CeedInt *num_points,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasis_Cuda   *data;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:              "Vector at points must be padded to the same number of points in each element for BasisApplyAtPoints on GPU backends."
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:      if (data->d_points_per_elem) CeedCallCuda(ceed, cudaFree(data->d_points_per_elem));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallCuda(ceed, cudaMalloc((void **)&data->d_points_per_elem, num_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallCuda(ceed, cudaMemcpy(data->d_points_per_elem, num_points, num_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallCuda(ceed, cudaMalloc((void **)&data->d_chebyshev_interp_1d, interp_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallCuda(ceed, cudaMemcpy(data->d_chebyshev_interp_1d, chebyshev_interp_1d, interp_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    const char basis_kernel_source[] = "// AtPoints basis source\n#include <ceed/jit-source/cuda/cuda-ref-basis-tensor-at-points.h>\n";
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    if (data->moduleAtPoints) CeedCallCuda(ceed, cuModuleUnload(data->moduleAtPoints));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->moduleAtPoints, 9, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "BASIS_BUF_LEN",
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallBackend(CeedGetKernel_Cuda(ceed, data->moduleAtPoints, "InterpAtPoints", &data->InterpAtPoints));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallBackend(CeedGetKernel_Cuda(ceed, data->moduleAtPoints, "GradAtPoints", &data->GradAtPoints));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemset(d_v, 0, length * sizeof(CeedScalar)));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallBackend(CeedRunKernel_Cuda(ceed, data->InterpAtPoints, num_elem, block_size, interp_args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallBackend(CeedRunKernel_Cuda(ceed, data->GradAtPoints, num_elem, block_size, grad_args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApplyAtPoints_Cuda(CeedBasis basis, const CeedInt num_elem, const CeedInt *num_points, CeedTransposeMode t_mode,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedBasisApplyAtPointsCore_Cuda(basis, false, num_elem, num_points, t_mode, eval_mode, x_ref, u, v));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApplyAddAtPoints_Cuda(CeedBasis basis, const CeedInt num_elem, const CeedInt *num_points, CeedTransposeMode t_mode,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedBasisApplyAtPointsCore_Cuda(basis, true, num_elem, num_points, t_mode, eval_mode, x_ref, u, v));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApplyNonTensorCore_Cuda(CeedBasis basis, bool apply_add, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasisNonTensor_Cuda *data;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemset(d_v, 0, length * sizeof(CeedScalar)));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->InterpTranspose, grid, block_size_x, 1, elems_per_block, interp_args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Interp, grid, block_size_x, 1, elems_per_block, interp_args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->DerivTranspose, grid, block_size_x, 1, elems_per_block, grad_args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Deriv, grid, block_size_x, 1, elems_per_block, grad_args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->DerivTranspose, grid, block_size_x, 1, elems_per_block, div_args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Deriv, grid, block_size_x, 1, elems_per_block, div_args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->DerivTranspose, grid, block_size_x, 1, elems_per_block, curl_args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Deriv, grid, block_size_x, 1, elems_per_block, curl_args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, grid, num_qpts, 1, elems_per_block, weight_args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApplyNonTensor_Cuda(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedBasisApplyNonTensorCore_Cuda(basis, false, num_elem, t_mode, eval_mode, u, v));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApplyAddNonTensor_Cuda(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedBasisApplyNonTensorCore_Cuda(basis, true, num_elem, t_mode, eval_mode, u, v));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisDestroy_Cuda(CeedBasis basis) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasis_Cuda *data;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cuModuleUnload(data->module));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  if (data->moduleAtPoints) CeedCallCuda(ceed, cuModuleUnload(data->moduleAtPoints));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  if (data->d_q_weight_1d) CeedCallCuda(ceed, cudaFree(data->d_q_weight_1d));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  if (data->d_points_per_elem) CeedCallCuda(ceed, cudaFree(data->d_points_per_elem));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_interp_1d));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_grad_1d));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_chebyshev_interp_1d));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisDestroyNonTensor_Cuda(CeedBasis basis) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasisNonTensor_Cuda *data;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cuModuleUnload(data->module));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  if (data->d_q_weight) CeedCallCuda(ceed, cudaFree(data->d_q_weight));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_interp));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_grad));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_div));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_curl));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:int CeedBasisCreateTensorH1_Cuda(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasis_Cuda *data;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  // Copy data to GPU
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight_1d, q_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight_1d, q_weight_1d, q_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp_1d, interp_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaMemcpy(data->d_interp_1d, interp_1d, interp_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_grad_1d, interp_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaMemcpy(data->d_grad_1d, grad_1d, interp_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  const char basis_kernel_source[] = "// Tensor basis source\n#include <ceed/jit-source/cuda/cuda-ref-basis-tensor.h>\n";
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->module, 7, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "BASIS_BUF_LEN",
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Interp", &data->Interp));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Grad", &data->Grad));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApply_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAdd_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAtPoints", CeedBasisApplyAtPoints_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAddAtPoints", CeedBasisApplyAddAtPoints_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:int CeedBasisCreateH1_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *grad,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasisNonTensor_Cuda *data;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  // Copy basis data to GPU
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight, q_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight, q_weight, q_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp, interp_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_interp, interp, interp_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_grad, grad_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_grad, grad, grad_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  const char basis_kernel_source[] = "// Nontensor basis source\n#include <ceed/jit-source/cuda/cuda-ref-basis-nontensor.h>\n";
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->module, 5, "BASIS_Q", num_qpts, "BASIS_P", num_nodes, "BASIS_Q_COMP_INTERP",
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Interp", &data->Interp));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Deriv", &data->Deriv));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "DerivTranspose", &data->DerivTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:int CeedBasisCreateHdiv_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *div,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasisNonTensor_Cuda *data;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  // Copy basis data to GPU
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight, q_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight, q_weight, q_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp, interp_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_interp, interp, interp_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_div, div_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_div, div, div_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  const char basis_kernel_source[] = "// Nontensor basis source\n#include <ceed/jit-source/cuda/cuda-ref-basis-nontensor.h>\n";
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->module, 5, "BASIS_Q", num_qpts, "BASIS_P", num_nodes, "BASIS_Q_COMP_INTERP",
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Interp", &data->Interp));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Deriv", &data->Deriv));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "DerivTranspose", &data->DerivTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:int CeedBasisCreateHcurl_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasisNonTensor_Cuda *data;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  // Copy basis data to GPU
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight, q_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight, q_weight, q_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp, interp_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_interp, interp, interp_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_curl, curl_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_curl, curl, curl_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  const char basis_kernel_source[] = "// Nontensor basis source\n#include <ceed/jit-source/cuda/cuda-ref-basis-nontensor.h>\n";
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->module, 5, "BASIS_Q", num_qpts, "BASIS_P", num_nodes, "BASIS_Q_COMP_INTERP",
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Interp", &data->Interp));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Deriv", &data->Deriv));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "DerivTranspose", &data->DerivTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:#include "../cuda/ceed-cuda-compile.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:#include "ceed-cuda-ref.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:extern "C" int CeedQFunctionBuildKernel_Cuda_ref(CeedQFunction qf) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:  CeedQFunction_Cuda *data;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:  string        kernel_name = "CeedKernelCudaRefQFunction_" + qfunction_name;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:  code << "#include <ceed/jit-source/cuda/cuda-ref-qfunction.h>\n\n";
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:  code << "extern \"C\" __global__ void " << kernel_name << "(void *ctx, CeedInt Q, Fields_Cuda fields) {\n";
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:  CeedCallBackend(CeedCompile_Cuda(ceed, code.str().c_str(), &data->module, 0));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, kernel_name.c_str(), &data->QFunction));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:#include "ceed-cuda-ref.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:#include "../cuda/ceed-cuda-common.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:// CUDA preferred MemType
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:static int CeedGetPreferredMemType_Cuda(CeedMemType *mem_type) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:int CeedGetCublasHandle_Cuda(Ceed ceed, cublasHandle_t *handle) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:  Ceed_Cuda *data;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:static int CeedInit_Cuda_ref(const char *resource, Ceed ceed) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:  Ceed_Cuda *data;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:  CeedCheck(!strcmp(resource_root, "/gpu/cuda/ref"), ceed, CEED_ERROR_BACKEND, "Cuda backend cannot use resource: %s", resource);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedInit_Cuda(ceed, resource));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "GetPreferredMemType", CeedGetPreferredMemType_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "VectorCreate", CeedVectorCreate_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1", CeedBasisCreateTensorH1_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1", CeedBasisCreateH1_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateHdiv", CeedBasisCreateHdiv_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateHcurl", CeedBasisCreateHcurl_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate", CeedElemRestrictionCreate_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreateAtPoints", CeedElemRestrictionCreate_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate", CeedQFunctionCreate_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionContextCreate", CeedQFunctionContextCreate_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate", CeedOperatorCreate_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreateAtPoints", CeedOperatorCreateAtPoints_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.c:CEED_INTERN int CeedRegister_Cuda(void) { return CeedRegister("/gpu/cuda/ref", CeedInit_Cuda_ref, 40); }
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction-load.h:CEED_INTERN int CeedQFunctionBuildKernel_Cuda_ref(CeedQFunction qf);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:#include <ceed/jit-source/cuda/cuda-types.h>
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:#include <cuda.h>
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:} CeedVector_Cuda;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:} CeedElemRestriction_Cuda;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:} CeedBasis_Cuda;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:} CeedBasisNonTensor_Cuda;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:  Fields_Cuda fields;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:} CeedQFunction_Cuda;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:} CeedQFunctionContext_Cuda;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:} CeedOperatorDiag_Cuda;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:} CeedOperatorAssemble_Cuda;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:  CeedOperatorDiag_Cuda     *diag;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:  CeedOperatorAssemble_Cuda *asmb;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:} CeedOperator_Cuda;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedGetCublasHandle_Cuda(Ceed ceed, cublasHandle_t *handle);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedVectorCreate_Cuda(CeedSize n, CeedVector vec);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedElemRestrictionCreate_Cuda(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, const bool *orients,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedBasisCreateTensorH1_Cuda(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedBasisCreateH1_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedBasisCreateHdiv_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedBasisCreateHcurl_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedQFunctionCreate_Cuda(CeedQFunction qf);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedQFunctionContextCreate_Cuda(CeedQFunctionContext ctx);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedOperatorCreate_Cuda(CeedOperator op);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedOperatorCreateAtPoints_Cuda(CeedOperator op);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:#include <cuda_runtime.h>
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:#include "../cuda/ceed-cuda-common.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:#include "ceed-cuda-ref.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static inline int CeedVectorNeedSync_Cuda(const CeedVector vec, CeedMemType mem_type, bool *need_sync) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static inline int CeedVectorSyncH2D_Cuda(const CeedVector vec) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallCuda(CeedVectorReturnCeed(vec), cudaMalloc((void **)&impl->d_array_owned, bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallCuda(CeedVectorReturnCeed(vec), cudaMemcpy(impl->d_array, impl->h_array, bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static inline int CeedVectorSyncD2H_Cuda(const CeedVector vec) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallCuda(CeedVectorReturnCeed(vec), cudaMemcpy(impl->h_array, impl->d_array, bytes, cudaMemcpyDeviceToHost));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorSyncArray_Cuda(const CeedVector vec, CeedMemType mem_type) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedVectorNeedSync_Cuda(vec, mem_type, &need_sync));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:      return CeedVectorSyncD2H_Cuda(vec);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:      return CeedVectorSyncH2D_Cuda(vec);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static inline int CeedVectorSetAllInvalid_Cuda(const CeedVector vec) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static inline int CeedVectorHasValidArray_Cuda(const CeedVector vec, bool *has_valid_array) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static inline int CeedVectorHasArrayOfType_Cuda(const CeedVector vec, CeedMemType mem_type, bool *has_array_of_type) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static inline int CeedVectorHasBorrowedArrayOfType_Cuda(const CeedVector vec, CeedMemType mem_type, bool *has_borrowed_array_of_type) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorSetArrayHost_Cuda(const CeedVector vec, const CeedCopyMode copy_mode, CeedScalar *array) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorSetArrayDevice_Cuda(const CeedVector vec, const CeedCopyMode copy_mode, CeedScalar *array) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetDeviceCeedScalarArray_Cuda(ceed, array, copy_mode, length, (const CeedScalar **)&impl->d_array_owned,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorSetArray_Cuda(const CeedVector vec, const CeedMemType mem_type, const CeedCopyMode copy_mode, CeedScalar *array) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedVectorSetAllInvalid_Cuda(vec));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:      return CeedVectorSetArrayHost_Cuda(vec, copy_mode, array);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:      return CeedVectorSetArrayDevice_Cuda(vec, copy_mode, array);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedHostCopyStrided_Cuda(CeedScalar *h_array, CeedSize start, CeedSize step, CeedSize length, CeedScalar *h_copy_array) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedDeviceCopyStrided_Cuda(CeedScalar *d_array, CeedSize start, CeedSize step, CeedSize length, CeedScalar *d_copy_array);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorCopyStrided_Cuda(CeedVector vec, CeedSize start, CeedSize step, CeedVector vec_copy) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedDeviceCopyStrided_Cuda(impl->d_array, start, step, length, copy_array));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedHostCopyStrided_Cuda(impl->h_array, start, step, length, copy_array));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedHostSetValue_Cuda(CeedScalar *h_array, CeedSize length, CeedScalar val) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedDeviceSetValue_Cuda(CeedScalar *d_array, CeedSize length, CeedScalar val);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorSetValue_Cuda(CeedVector vec, CeedScalar val) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedDeviceSetValue_Cuda(impl->d_array, length, val));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedHostSetValue_Cuda(impl->h_array, length, val));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedHostSetValueStrided_Cuda(CeedScalar *h_array, CeedSize start, CeedSize step, CeedSize length, CeedScalar val) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedDeviceSetValueStrided_Cuda(CeedScalar *d_array, CeedSize start, CeedSize step, CeedSize length, CeedScalar val);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorSetValueStrided_Cuda(CeedVector vec, CeedSize start, CeedSize step, CeedScalar val) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedDeviceSetValueStrided_Cuda(impl->d_array, start, step, length, val));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedHostSetValueStrided_Cuda(impl->h_array, start, step, length, val));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorTakeArray_Cuda(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorGetArrayCore_Cuda(const CeedVector vec, const CeedMemType mem_type, CeedScalar **array) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorGetArrayRead_Cuda(const CeedVector vec, const CeedMemType mem_type, const CeedScalar **array) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  return CeedVectorGetArrayCore_Cuda(vec, mem_type, (CeedScalar **)array);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorGetArray_Cuda(const CeedVector vec, const CeedMemType mem_type, CeedScalar **array) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedVectorGetArrayCore_Cuda(vec, mem_type, array));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedVectorSetAllInvalid_Cuda(vec));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorGetArrayWrite_Cuda(const CeedVector vec, const CeedMemType mem_type, CeedScalar **array) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedVectorHasArrayOfType_Cuda(vec, mem_type, &has_array_of_type));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  return CeedVectorGetArray_Cuda(vec, mem_type, array);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorNorm_Cuda(CeedVector vec, CeedNormType type, CeedScalar *norm) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:#if CUDA_VERSION < 12000
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda  *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedGetCublasHandle_Cuda(ceed, &handle));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:#if CUDA_VERSION < 12000
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  // With CUDA 12, we can use the 64-bit integer interface. Prior to that,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:#if CUDA_VERSION >= 12000  // We have CUDA 12, and can use 64-bit integers
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:#if CUDA_VERSION >= 12000
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:#if CUDA_VERSION >= 12000
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:#if CUDA_VERSION >= 12000
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:#if CUDA_VERSION >= 12000
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:        CeedCallCuda(ceed, cudaMemcpy(&norm_no_abs, impl->d_array + index - 1, sizeof(CeedScalar), cudaMemcpyDeviceToHost));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:          CeedCallCuda(ceed, cudaMemcpy(&sub_max, d_array_start + index - 1, sizeof(CeedScalar), cudaMemcpyDeviceToHost));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:#if CUDA_VERSION >= 12000
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:        CeedCallCuda(ceed, cudaMemcpy(&norm_no_abs, impl->d_array + index - 1, sizeof(CeedScalar), cudaMemcpyDeviceToHost));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:          CeedCallCuda(ceed, cudaMemcpy(&sub_max, d_array_start + index - 1, sizeof(CeedScalar), cudaMemcpyDeviceToHost));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedHostReciprocal_Cuda(CeedScalar *h_array, CeedSize length) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedDeviceReciprocal_Cuda(CeedScalar *d_array, CeedSize length);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorReciprocal_Cuda(CeedVector vec) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  if (impl->d_array) CeedCallBackend(CeedDeviceReciprocal_Cuda(impl->d_array, length));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  if (impl->h_array) CeedCallBackend(CeedHostReciprocal_Cuda(impl->h_array, length));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedHostScale_Cuda(CeedScalar *x_array, CeedScalar alpha, CeedSize length) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedDeviceScale_Cuda(CeedScalar *x_array, CeedScalar alpha, CeedSize length);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorScale_Cuda(CeedVector x, CeedScalar alpha) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *x_impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  if (x_impl->d_array) CeedCallBackend(CeedDeviceScale_Cuda(x_impl->d_array, alpha, length));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  if (x_impl->h_array) CeedCallBackend(CeedHostScale_Cuda(x_impl->h_array, alpha, length));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedHostAXPY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedSize length) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedDeviceAXPY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedSize length);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorAXPY_Cuda(CeedVector y, CeedScalar alpha, CeedVector x) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *y_impl, *x_impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedDeviceAXPY_Cuda(y_impl->d_array, alpha, x_impl->d_array, length));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedHostAXPY_Cuda(y_impl->h_array, alpha, x_impl->h_array, length));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedHostAXPBY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar beta, CeedScalar *x_array, CeedSize length) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedDeviceAXPBY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar beta, CeedScalar *x_array, CeedSize length);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorAXPBY_Cuda(CeedVector y, CeedScalar alpha, CeedScalar beta, CeedVector x) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *y_impl, *x_impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedDeviceAXPBY_Cuda(y_impl->d_array, alpha, beta, x_impl->d_array, length));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedHostAXPBY_Cuda(y_impl->h_array, alpha, beta, x_impl->h_array, length));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedHostPointwiseMult_Cuda(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedSize length) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedDevicePointwiseMult_Cuda(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedSize length);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorPointwiseMult_Cuda(CeedVector w, CeedVector x, CeedVector y) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *w_impl, *x_impl, *y_impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedDevicePointwiseMult_Cuda(w_impl->d_array, x_impl->d_array, y_impl->d_array, length));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedHostPointwiseMult_Cuda(w_impl->h_array, x_impl->h_array, y_impl->h_array, length));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorDestroy_Cuda(const CeedVector vec) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallCuda(CeedVectorReturnCeed(vec), cudaFree(impl->d_array_owned));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedVectorCreate_Cuda(CeedSize n, CeedVector vec) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "HasValidArray", CeedVectorHasValidArray_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "HasBorrowedArrayOfType", CeedVectorHasBorrowedArrayOfType_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetArray", CeedVectorSetArray_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "TakeArray", CeedVectorTakeArray_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "CopyStrided", CeedVectorCopyStrided_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetValue", CeedVectorSetValue_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetValueStrided", CeedVectorSetValueStrided_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SyncArray", CeedVectorSyncArray_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArray", CeedVectorGetArray_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead", CeedVectorGetArrayRead_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayWrite", CeedVectorGetArrayWrite_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Norm", CeedVectorNorm_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Reciprocal", CeedVectorReciprocal_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Scale", CeedVectorScale_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "AXPY", CeedVectorAXPY_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "AXPBY", CeedVectorAXPBY_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "PointwiseMult", CeedVectorPointwiseMult_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Destroy", CeedVectorDestroy_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/kernels/cuda-ref-vector.cu:#include <cuda.h>
rust/libceed-sys/c-src/backends/cuda-ref/kernels/cuda-ref-vector.cu:extern "C" int CeedDeviceCopyStrided_Cuda(CeedScalar *d_array, CeedSize start, CeedSize step, CeedSize length, CeedScalar *d_copy_array) {
rust/libceed-sys/c-src/backends/cuda-ref/kernels/cuda-ref-vector.cu:extern "C" int CeedDeviceSetValue_Cuda(CeedScalar *d_array, CeedSize length, CeedScalar val) {
rust/libceed-sys/c-src/backends/cuda-ref/kernels/cuda-ref-vector.cu:extern "C" int CeedDeviceSetValueStrided_Cuda(CeedScalar *d_array, CeedSize start, CeedSize step, CeedSize length, CeedScalar val) {
rust/libceed-sys/c-src/backends/cuda-ref/kernels/cuda-ref-vector.cu:extern "C" int CeedDeviceReciprocal_Cuda(CeedScalar *d_array, CeedSize length) {
rust/libceed-sys/c-src/backends/cuda-ref/kernels/cuda-ref-vector.cu:extern "C" int CeedDeviceScale_Cuda(CeedScalar *x_array, CeedScalar alpha, CeedSize length) {
rust/libceed-sys/c-src/backends/cuda-ref/kernels/cuda-ref-vector.cu:extern "C" int CeedDeviceAXPY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedSize length) {
rust/libceed-sys/c-src/backends/cuda-ref/kernels/cuda-ref-vector.cu:extern "C" int CeedDeviceAXPBY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar beta, CeedScalar *x_array, CeedSize length) {
rust/libceed-sys/c-src/backends/cuda-ref/kernels/cuda-ref-vector.cu:extern "C" int CeedDevicePointwiseMult_Cuda(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedSize length) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:#include <cuda_runtime.h>
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:#include "../cuda/ceed-cuda-common.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:#include "ceed-cuda-ref.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static inline int CeedQFunctionContextSyncH2D_Cuda(const CeedQFunctionContext ctx) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:    CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_data_owned, ctx_size));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallCuda(ceed, cudaMemcpy(impl->d_data, impl->h_data, ctx_size, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static inline int CeedQFunctionContextSyncD2H_Cuda(const CeedQFunctionContext ctx) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallCuda(ceed, cudaMemcpy(impl->h_data, impl->d_data, ctx_size, cudaMemcpyDeviceToHost));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static inline int CeedQFunctionContextSync_Cuda(const CeedQFunctionContext ctx, CeedMemType mem_type) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:      return CeedQFunctionContextSyncD2H_Cuda(ctx);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:      return CeedQFunctionContextSyncH2D_Cuda(ctx);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static inline int CeedQFunctionContextSetAllInvalid_Cuda(const CeedQFunctionContext ctx) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static inline int CeedQFunctionContextHasValidData_Cuda(const CeedQFunctionContext ctx, bool *has_valid_data) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static inline int CeedQFunctionContextHasBorrowedDataOfType_Cuda(const CeedQFunctionContext ctx, CeedMemType mem_type,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static inline int CeedQFunctionContextNeedSync_Cuda(const CeedQFunctionContext ctx, CeedMemType mem_type, bool *need_sync) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static int CeedQFunctionContextSetDataHost_Cuda(const CeedQFunctionContext ctx, const CeedCopyMode copy_mode, void *data) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static int CeedQFunctionContextSetDataDevice_Cuda(const CeedQFunctionContext ctx, const CeedCopyMode copy_mode, void *data) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallCuda(ceed, cudaFree(impl->d_data_owned));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:      CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_data_owned, ctx_size));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:      CeedCallCuda(ceed, cudaMemcpy(impl->d_data, data, ctx_size, cudaMemcpyDeviceToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static int CeedQFunctionContextSetData_Cuda(const CeedQFunctionContext ctx, const CeedMemType mem_type, const CeedCopyMode copy_mode, void *data) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedQFunctionContextSetAllInvalid_Cuda(ctx));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:      return CeedQFunctionContextSetDataHost_Cuda(ctx, copy_mode, data);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:      return CeedQFunctionContextSetDataDevice_Cuda(ctx, copy_mode, data);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static int CeedQFunctionContextTakeData_Cuda(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedQFunctionContextNeedSync_Cuda(ctx, mem_type, &need_sync));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  if (need_sync) CeedCallBackend(CeedQFunctionContextSync_Cuda(ctx, mem_type));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static int CeedQFunctionContextGetDataCore_Cuda(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedQFunctionContextNeedSync_Cuda(ctx, mem_type, &need_sync));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  if (need_sync) CeedCallBackend(CeedQFunctionContextSync_Cuda(ctx, mem_type));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static int CeedQFunctionContextGetDataRead_Cuda(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  return CeedQFunctionContextGetDataCore_Cuda(ctx, mem_type, data);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static int CeedQFunctionContextGetData_Cuda(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedQFunctionContextGetDataCore_Cuda(ctx, mem_type, data));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedQFunctionContextSetAllInvalid_Cuda(ctx));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static int CeedQFunctionContextDestroy_Cuda(const CeedQFunctionContext ctx) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallCuda(CeedQFunctionContextReturnCeed(ctx), cudaFree(impl->d_data_owned));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:int CeedQFunctionContextCreate_Cuda(CeedQFunctionContext ctx) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "HasValidData", CeedQFunctionContextHasValidData_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "HasBorrowedDataOfType", CeedQFunctionContextHasBorrowedDataOfType_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "SetData", CeedQFunctionContextSetData_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "TakeData", CeedQFunctionContextTakeData_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "GetData", CeedQFunctionContextGetData_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "GetDataRead", CeedQFunctionContextGetDataRead_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "Destroy", CeedQFunctionContextDestroy_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:#include <ceed/jit-source/cuda/cuda-types.h>
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:#include <cuda.h>
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:#include "../cuda/ceed-cuda-common.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:#include "../cuda/ceed-cuda-compile.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:#include "ceed-cuda-ref-qfunction-load.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:#include "ceed-cuda-ref.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:static int CeedQFunctionApply_Cuda(CeedQFunction qf, CeedInt Q, CeedVector *U, CeedVector *V) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:  Ceed_Cuda          *ceed_Cuda;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedQFunction_Cuda *data;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedCallBackend(CeedQFunctionBuildKernel_Cuda_ref(qf));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedCallBackend(CeedGetData(ceed, &ceed_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedCallBackend(CeedRunKernelAutoblockCuda(ceed, data->QFunction, Q, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:static int CeedQFunctionDestroy_Cuda(CeedQFunction qf) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedQFunction_Cuda *data;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:  if (data->module) CeedCallCuda(CeedQFunctionReturnCeed(qf), cuModuleUnload(data->module));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:static int CeedQFunctionSetCUDAUserFunction_Cuda(CeedQFunction qf, CUfunction f) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedQFunction_Cuda *data;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:int CeedQFunctionCreate_Cuda(CeedQFunction qf) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedQFunction_Cuda *data;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Apply", CeedQFunctionApply_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy", CeedQFunctionDestroy_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "SetCUDAUserFunction", CeedQFunctionSetCUDAUserFunction_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:#include <cuda.h>
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:#include <cuda_runtime.h>
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:#include "../cuda/ceed-cuda-common.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:#include "../cuda/ceed-cuda-compile.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:#include "ceed-cuda-ref.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:static inline int CeedElemRestrictionSetupCompile_Cuda(CeedElemRestriction rstr) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedElemRestriction_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  // Compile CUDA kernels
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      const char restriction_kernel_source[] = "// Strided restriction source\n#include <ceed/jit-source/cuda/cuda-ref-restriction-strided.h>\n";
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedCompile_Cuda(ceed, restriction_kernel_source, &impl->module, 6, "RSTR_ELEM_SIZE", elem_size, "RSTR_NUM_ELEM", num_elem,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "StridedNoTranspose", &impl->ApplyNoTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "StridedTranspose", &impl->ApplyTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      const char restriction_kernel_source[] = "// Standard restriction source\n#include <ceed/jit-source/cuda/cuda-ref-restriction-offset.h>\n";
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedCompile_Cuda(ceed, restriction_kernel_source, &impl->module, 6, "RSTR_ELEM_SIZE", elem_size, "RSTR_NUM_ELEM", num_elem,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetNoTranspose", &impl->ApplyNoTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetTranspose", &impl->ApplyTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          "// AtPoints restriction source\n#include <ceed/jit-source/cuda/cuda-ref-restriction-at-points.h>\n\n"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          "// Standard restriction source\n#include <ceed/jit-source/cuda/cuda-ref-restriction-offset.h>\n";
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedCompile_Cuda(ceed, restriction_kernel_source, &impl->module, 6, "RSTR_ELEM_SIZE", elem_size, "RSTR_NUM_ELEM", num_elem,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetNoTranspose", &impl->ApplyNoTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "AtPointsTranspose", &impl->ApplyTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          "// Oriented restriction source\n#include <ceed/jit-source/cuda/cuda-ref-restriction-oriented.h>\n\n"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          "// Standard restriction source\n#include <ceed/jit-source/cuda/cuda-ref-restriction-offset.h>\n";
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedCompile_Cuda(ceed, restriction_kernel_source, &impl->module, 6, "RSTR_ELEM_SIZE", elem_size, "RSTR_NUM_ELEM", num_elem,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OrientedNoTranspose", &impl->ApplyNoTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetNoTranspose", &impl->ApplyUnsignedNoTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OrientedTranspose", &impl->ApplyTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetTranspose", &impl->ApplyUnsignedTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          "// Curl oriented restriction source\n#include <ceed/jit-source/cuda/cuda-ref-restriction-curl-oriented.h>\n\n"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          "// Standard restriction source\n#include <ceed/jit-source/cuda/cuda-ref-restriction-offset.h>\n";
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedCompile_Cuda(ceed, restriction_kernel_source, &impl->module, 6, "RSTR_ELEM_SIZE", elem_size, "RSTR_NUM_ELEM", num_elem,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "CurlOrientedNoTranspose", &impl->ApplyNoTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "CurlOrientedUnsignedNoTranspose", &impl->ApplyUnsignedNoTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetNoTranspose", &impl->ApplyUnorientedNoTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "CurlOrientedTranspose", &impl->ApplyTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "CurlOrientedUnsignedTranspose", &impl->ApplyUnsignedTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetTranspose", &impl->ApplyUnorientedTranspose));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:static inline int CeedElemRestrictionApply_Cuda_Core(CeedElemRestriction rstr, CeedTransposeMode t_mode, bool use_signs, bool use_orients,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedElemRestriction_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:    CeedCallBackend(CeedElemRestrictionSetupCompile_Cuda(rstr));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:        CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyNoTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:        CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyNoTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyNoTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedNoTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyNoTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedNoTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnorientedNoTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:        CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnorientedTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnorientedTranspose, grid, block_size, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionApply_Cuda(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector v, CeedRequest *request) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  return CeedElemRestrictionApply_Cuda_Core(rstr, t_mode, true, true, u, v, request);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionApplyUnsigned_Cuda(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  return CeedElemRestrictionApply_Cuda_Core(rstr, t_mode, false, true, u, v, request);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionApplyUnoriented_Cuda(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  return CeedElemRestrictionApply_Cuda_Core(rstr, t_mode, false, false, u, v, request);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionGetOffsets_Cuda(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt **offsets) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedElemRestriction_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionGetOrientations_Cuda(CeedElemRestriction rstr, CeedMemType mem_type, const bool **orients) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedElemRestriction_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionGetCurlOrientations_Cuda(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt8 **curl_orients) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedElemRestriction_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionGetAtPointsElementOffset_Cuda(CeedElemRestriction rstr, CeedInt elem, CeedSize *elem_offset) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionDestroy_Cuda(CeedElemRestriction rstr) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedElemRestriction_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:    CeedCallCuda(ceed, cuModuleUnload(impl->module));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_offsets_owned));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_t_offsets));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_t_indices));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_l_vec_indices));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaFree((bool *)impl->d_orients_owned));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaFree((CeedInt8 *)impl->d_curl_orients_owned));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_offsets_at_points_owned));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_points_per_elem_owned));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionOffset_Cuda(const CeedElemRestriction rstr, const CeedInt elem_size, const CeedInt *indices) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedElemRestriction_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_l_vec_indices, num_nodes * sizeof(CeedInt)));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaMemcpy((CeedInt *)impl->d_l_vec_indices, l_vec_indices, num_nodes * sizeof(CeedInt), cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_t_offsets, size_offsets * sizeof(CeedInt)));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaMemcpy((CeedInt *)impl->d_t_offsets, t_offsets, size_offsets * sizeof(CeedInt), cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_t_indices, size_indices * sizeof(CeedInt)));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaMemcpy((CeedInt *)impl->d_t_indices, t_indices, size_indices * sizeof(CeedInt), cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:int CeedElemRestrictionCreate_Cuda(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, const bool *orients,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedElemRestriction_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:    CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_offsets_at_points_owned, at_points_size * sizeof(CeedInt)));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:    CeedCallCuda(ceed, cudaMemcpy((CeedInt **)impl->d_offsets_at_points_owned, impl->h_offsets_at_points, at_points_size * sizeof(CeedInt),
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:                                  cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:    CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_points_per_elem_owned, num_elem * sizeof(CeedInt)));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:    CeedCallCuda(ceed,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:                 cudaMemcpy((CeedInt **)impl->d_points_per_elem_owned, impl->h_points_per_elem, num_elem * sizeof(CeedInt), cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:        CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_offsets_owned, size * sizeof(CeedInt)));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:        CeedCallCuda(ceed, cudaMemcpy((CeedInt *)impl->d_offsets_owned, impl->h_offsets, size * sizeof(CeedInt), cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:        if (is_deterministic) CeedCallBackend(CeedElemRestrictionOffset_Cuda(rstr, elem_size, offsets));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:        CeedCallBackend(CeedSetDeviceCeedIntArray_Cuda(ceed, offsets, copy_mode, size, &impl->d_offsets_owned, &impl->d_offsets_borrowed,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:        CeedCallCuda(ceed, cudaMemcpy((CeedInt *)impl->h_offsets_owned, impl->d_offsets, size * sizeof(CeedInt), cudaMemcpyDeviceToHost));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:        if (is_deterministic) CeedCallBackend(CeedElemRestrictionOffset_Cuda(rstr, elem_size, offsets));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_orients_owned, size * sizeof(bool)));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallCuda(ceed, cudaMemcpy((bool *)impl->d_orients_owned, impl->h_orients, size * sizeof(bool), cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedSetDeviceBoolArray_Cuda(ceed, orients, copy_mode, size, &impl->d_orients_owned, &impl->d_orients_borrowed,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallCuda(ceed, cudaMemcpy((bool *)impl->h_orients_owned, impl->d_orients, size * sizeof(bool), cudaMemcpyDeviceToHost));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_curl_orients_owned, 3 * size * sizeof(CeedInt8)));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallCuda(ceed,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:                       cudaMemcpy((CeedInt8 *)impl->d_curl_orients_owned, impl->h_curl_orients, 3 * size * sizeof(CeedInt8), cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedSetDeviceCeedInt8Array_Cuda(ceed, curl_orients, copy_mode, 3 * size, &impl->d_curl_orients_owned,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallCuda(ceed,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:                       cudaMemcpy((CeedInt8 *)impl->h_curl_orients_owned, impl->d_curl_orients, 3 * size * sizeof(CeedInt8), cudaMemcpyDeviceToHost));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "Apply", CeedElemRestrictionApply_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "ApplyUnsigned", CeedElemRestrictionApplyUnsigned_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "ApplyUnoriented", CeedElemRestrictionApplyUnoriented_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetOffsets", CeedElemRestrictionGetOffsets_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetOrientations", CeedElemRestrictionGetOrientations_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetCurlOrientations", CeedElemRestrictionGetCurlOrientations_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:        CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetAtPointsElementOffset", CeedElemRestrictionGetAtPointsElementOffset_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "Destroy", CeedElemRestrictionDestroy_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:#include <cuda.h>
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:#include <cuda_runtime.h>
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:#include "../cuda/ceed-cuda-common.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:#include "../cuda/ceed-cuda-compile.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:#include "ceed-cuda-ref.h"
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorDestroy_Cuda(CeedOperator op) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedCallCuda(ceed, cuModuleUnload(impl->diag->module));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedCallCuda(ceed, cuModuleUnload(impl->diag->module_point_block));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_eval_modes_in));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_eval_modes_out));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_identity));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_interp_in));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_interp_out));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_grad_in));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_grad_out));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_div_in));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_div_out));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_curl_in));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_curl_out));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cuModuleUnload(impl->asmb->module));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->asmb->d_B_in));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->asmb->d_B_out));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorSetupFields_Cuda(CeedQFunction qf, CeedOperator op, bool is_input, bool is_at_points, bool *skip_rstr, bool *apply_add_basis,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorSetup_Cuda(CeedOperator op) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedOperatorSetupFields_Cuda(qf, op, true, false, impl->skip_rstr_in, NULL, impl->e_vecs_in, impl->q_vecs_in, num_input_fields, Q, num_elem));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorSetupFields_Cuda(qf, op, false, false, impl->skip_rstr_out, impl->apply_add_basis_out, impl->e_vecs_out,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static inline int CeedOperatorInputRestrict_Cuda(CeedOperatorField op_input_field, CeedQFunctionField qf_input_field, CeedInt input_field,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:                                                 CeedVector in_vec, CeedVector active_e_vec, const bool skip_active, CeedOperator_Cuda *impl,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static inline int CeedOperatorInputBasis_Cuda(CeedOperatorField op_input_field, CeedQFunctionField qf_input_field, CeedInt input_field,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:                                              CeedOperator_Cuda *impl) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static inline int CeedOperatorInputRestore_Cuda(CeedOperatorField op_input_field, CeedQFunctionField qf_input_field, CeedInt input_field,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:                                                CeedVector in_vec, CeedVector active_e_vec, const bool skip_active, CeedOperator_Cuda *impl) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorApplyAdd_Cuda(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorSetup_Cuda(op));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedOperatorInputRestrict_Cuda(op_input_fields[field], qf_input_fields[field], field, in_vec, active_e_vec, false, impl, request));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputBasis_Cuda(op_input_fields[field], qf_input_fields[field], field, in_vec, active_e_vec, num_elem, false, impl));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputRestore_Cuda(op_input_fields[i], qf_input_fields[i], i, in_vec, active_e_vec, false, impl));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorSetupAtPoints_Cuda(CeedOperator op) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorSetupFields_Cuda(qf, op, true, true, impl->skip_rstr_in, NULL, impl->e_vecs_in, impl->q_vecs_in, num_input_fields,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorSetupFields_Cuda(qf, op, false, true, impl->skip_rstr_out, impl->apply_add_basis_out, impl->e_vecs_out,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static inline int CeedOperatorInputBasisAtPoints_Cuda(CeedOperatorField op_input_field, CeedQFunctionField qf_input_field, CeedInt input_field,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:                                                      const bool skip_active, CeedOperator_Cuda *impl) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorApplyAddAtPoints_Cuda(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorSetupAtPoints_Cuda(op));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedOperatorInputRestrict_Cuda(op_input_fields[field], qf_input_fields[field], field, in_vec, active_e_vec, false, impl, request));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputBasisAtPoints_Cuda(op_input_fields[field], qf_input_fields[field], field, in_vec, active_e_vec, num_elem,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputRestore_Cuda(op_input_fields[i], qf_input_fields[i], i, in_vec, active_e_vec, false, impl));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static inline int CeedOperatorLinearAssembleQFunctionCore_Cuda(CeedOperator op, bool build_objects, CeedVector *assembled, CeedElemRestriction *rstr,
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorSetup_Cuda(op));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputRestrict_Cuda(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, true, impl, request));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputBasis_Cuda(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, num_elem, true, impl));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputRestore_Cuda(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, true, impl));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorLinearAssembleQFunction_Cuda(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  return CeedOperatorLinearAssembleQFunctionCore_Cuda(op, true, assembled, rstr, request);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorLinearAssembleQFunctionUpdate_Cuda(CeedOperator op, CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  return CeedOperatorLinearAssembleQFunctionCore_Cuda(op, false, &assembled, &rstr, request);
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static inline int CeedOperatorAssembleDiagonalSetup_Cuda(CeedOperator op) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperatorDiag_Cuda *diag = impl->diag;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaMalloc((void **)&diag->d_identity, interp_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaMemcpy(diag->d_identity, identity, interp_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMalloc((void **)&d_interp, interp_bytes * q_comp_interp));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMemcpy(d_interp, interp, interp_bytes * q_comp_interp, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMalloc((void **)&d_grad, interp_bytes * q_comp_grad));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMemcpy(d_grad, grad, interp_bytes * q_comp_grad, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMalloc((void **)&d_interp, interp_bytes * q_comp_interp));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMemcpy(d_interp, interp, interp_bytes * q_comp_interp, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMalloc((void **)&d_div, interp_bytes * q_comp_div));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMemcpy(d_div, div, interp_bytes * q_comp_div, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMalloc((void **)&d_interp, interp_bytes * q_comp_interp));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMemcpy(d_interp, interp, interp_bytes * q_comp_interp, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMalloc((void **)&d_curl, interp_bytes * q_comp_curl));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMemcpy(d_curl, curl, interp_bytes * q_comp_curl, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallCuda(ceed, cudaMalloc((void **)&diag->d_eval_modes_in, num_eval_modes_in * eval_modes_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallCuda(ceed, cudaMemcpy(diag->d_eval_modes_in, eval_modes_in, num_eval_modes_in * eval_modes_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallCuda(ceed, cudaMalloc((void **)&diag->d_eval_modes_out, num_eval_modes_out * eval_modes_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallCuda(ceed, cudaMemcpy(diag->d_eval_modes_out, eval_modes_out, num_eval_modes_out * eval_modes_bytes, cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static inline int CeedOperatorAssembleDiagonalSetupCompile_Cuda(CeedOperator op, CeedInt use_ceedsize_idx, const bool is_point_block) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperatorDiag_Cuda *diag = impl->diag;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  const char diagonal_kernel_source[] = "// Diagonal assembly source\n#include <ceed/jit-source/cuda/cuda-ref-operator-assemble-diagonal.h>\n";
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallCuda(ceed, CeedCompile_Cuda(ceed, diagonal_kernel_source, module, 8, "NUM_EVAL_MODES_IN", num_eval_modes_in, "NUM_EVAL_MODES_OUT",
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallCuda(ceed, CeedGetKernel_Cuda(ceed, *module, "LinearDiagonal", is_point_block ? &diag->LinearPointBlock : &diag->LinearDiagonal));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static inline int CeedOperatorAssembleDiagonalCore_Cuda(CeedOperator op, CeedVector assembled, CeedRequest *request, const bool is_point_block) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  if (!impl->diag) CeedCallBackend(CeedOperatorAssembleDiagonalSetup_Cuda(op));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperatorDiag_Cuda *diag = impl->diag;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorAssembleDiagonalSetupCompile_Cuda(op, use_ceedsize_idx, is_point_block));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedCallBackend(CeedRunKernelDim_Cuda(ceed, diag->LinearPointBlock, grid, num_nodes, 1, elems_per_block, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedCallBackend(CeedRunKernelDim_Cuda(ceed, diag->LinearDiagonal, grid, num_nodes, 1, elems_per_block, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorLinearAssembleAddDiagonal_Cuda(CeedOperator op, CeedVector assembled, CeedRequest *request) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorAssembleDiagonalCore_Cuda(op, assembled, request, false));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorLinearAssembleAddPointBlockDiagonal_Cuda(CeedOperator op, CeedVector assembled, CeedRequest *request) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorAssembleDiagonalCore_Cuda(op, assembled, request, true));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedSingleOperatorAssembleSetup_Cuda(CeedOperator op, CeedInt use_ceedsize_idx) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  Ceed_Cuda          *cuda_data;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperatorAssemble_Cuda *asmb = impl->asmb;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedGetData(ceed, &cuda_data));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  bool fallback = asmb->block_size_x * asmb->block_size_y * asmb->elems_per_block > cuda_data->device_prop.maxThreadsPerBlock;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  const char assembly_kernel_source[] = "// Full assembly source\n#include <ceed/jit-source/cuda/cuda-ref-operator-assemble.h>\n";
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedCompile_Cuda(ceed, assembly_kernel_source, &asmb->module, 10, "NUM_EVAL_MODES_IN", num_eval_modes_in, "NUM_EVAL_MODES_OUT",
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, asmb->module, "LinearAssemble", &asmb->LinearAssemble));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaMalloc((void **)&asmb->d_B_in, in_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedCallCuda(ceed, cudaMemcpy(&asmb->d_B_in[i * elem_size_in * num_qpts_in], h_B_in, elem_size_in * num_qpts_in * sizeof(CeedScalar),
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:                                    cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaMalloc((void **)&asmb->d_B_out, out_bytes));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedCallCuda(ceed, cudaMemcpy(&asmb->d_B_out[i * elem_size_out * num_qpts_out], h_B_out, elem_size_out * num_qpts_out * sizeof(CeedScalar),
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:                                    cudaMemcpyHostToDevice));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedSingleOperatorAssemble_Cuda(CeedOperator op, CeedInt offset, CeedVector values) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  if (!impl->asmb) CeedCallBackend(CeedSingleOperatorAssembleSetup_Cuda(op, use_ceedsize_idx));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperatorAssemble_Cuda *asmb = impl->asmb;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedRunKernelDimShared_Cuda(ceed, asmb->LinearAssemble, grid, asmb->block_size_x, asmb->block_size_y, asmb->elems_per_block, shared_mem, args));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorLinearAssembleQFunctionAtPoints_Cuda(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorLinearAssembleAddDiagonalAtPoints_Cuda(CeedOperator op, CeedVector assembled, CeedRequest *request) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorSetupAtPoints_Cuda(op));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputRestrict_Cuda(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, true, impl, request));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputBasisAtPoints_Cuda(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, num_elem, num_points, true, impl));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputRestore_Cuda(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, true, impl));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:int CeedOperatorCreate_Cuda(CeedOperator op) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunction_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionUpdate_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal", CeedOperatorLinearAssembleAddDiagonal_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddPointBlockDiagonal", CeedOperatorLinearAssembleAddPointBlockDiagonal_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleSingle", CeedSingleOperatorAssemble_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:int CeedOperatorCreateAtPoints_Cuda(CeedOperator op) {
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda *impl;
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunctionAtPoints_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal", CeedOperatorLinearAssembleAddDiagonalAtPoints_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAddAtPoints_Cuda));
rust/libceed-sys/c-src/backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Cuda));
rust/libceed-sys/c-src/backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Cuda, 1, "/gpu/cuda/ref")
rust/libceed-sys/c-src/backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Cuda_Gen, 1, "/gpu/cuda/gen")
rust/libceed-sys/c-src/backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Cuda_Shared, 1, "/gpu/cuda/shared")
rust/libceed-sys/c-src/backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Hip, 1, "/gpu/hip/ref")
rust/libceed-sys/c-src/backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Hip_Gen, 1, "/gpu/hip/gen")
rust/libceed-sys/c-src/backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Hip_Shared, 1, "/gpu/hip/shared")
rust/libceed-sys/c-src/backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Sycl, 1, "/gpu/sycl/ref")
rust/libceed-sys/c-src/backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Sycl_Shared, 1, "/gpu/sycl/shared")
rust/libceed-sys/c-src/backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Sycl_Gen, 1, "/gpu/sycl/gen")
rust/libceed-sys/c-src/backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Magma, 2, "/gpu/cuda/magma", "/gpu/hip/magma")
rust/libceed-sys/c-src/backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Magma_Det, 2, "/gpu/cuda/magma/det", "/gpu/hip/magma/det")
rust/libceed-sys/c-src/backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Occa, 6, "/cpu/self/occa", "/cpu/openmp/occa", "/gpu/dpcpp/occa", "/gpu/opencl/occa", "/gpu/hip/occa", "/gpu/cuda/occa")
rust/libceed-sys/c-src/backends/magma/ceed-magma-common.c:  magma_queue_create_from_cuda(data->device_id, NULL, NULL, NULL, &(data->queue));
rust/libceed-sys/c-src/backends/magma/ceed-magma-gemm-selector.cpp:static inline auto gemm_selector_get_data(int gpu_arch, char precision, char trans_A) -> decltype(dgemm_nn_mi100) {
rust/libceed-sys/c-src/backends/magma/ceed-magma-gemm-selector.cpp:  if (gpu_arch >= 910) {
rust/libceed-sys/c-src/backends/magma/ceed-magma-gemm-selector.cpp:static inline auto gemm_selector_get_data(int gpu_arch, char precision, char trans_A) -> decltype(dgemm_nn_v100) {
rust/libceed-sys/c-src/backends/magma/ceed-magma-gemm-selector.cpp:  if (gpu_arch >= 800) {
rust/libceed-sys/c-src/backends/magma/ceed-magma-gemm-selector.cpp:void gemm_selector(int gpu_arch, char precision, char trans_A, int m, int n, int k, int *n_batch, int *use_magma) {
rust/libceed-sys/c-src/backends/magma/ceed-magma-gemm-selector.cpp:  const auto &data = gemm_selector_get_data(gpu_arch, precision, trans_A);
rust/libceed-sys/c-src/backends/magma/ceed-magma-gemm-selector.cpp:static inline auto nontensor_rtc_get_data(int gpu_arch, char trans_A) -> decltype(drtc_n_mi100) {
rust/libceed-sys/c-src/backends/magma/ceed-magma-gemm-selector.cpp:  if (gpu_arch >= 910) {
rust/libceed-sys/c-src/backends/magma/ceed-magma-gemm-selector.cpp:static inline auto nontensor_rtc_get_data(int gpu_arch, char trans_A) -> decltype(drtc_n_v100) {
rust/libceed-sys/c-src/backends/magma/ceed-magma-gemm-selector.cpp:  if (gpu_arch >= 900) {
rust/libceed-sys/c-src/backends/magma/ceed-magma-gemm-selector.cpp:  } else if (gpu_arch >= 800) {
rust/libceed-sys/c-src/backends/magma/ceed-magma-gemm-selector.cpp:CeedInt nontensor_rtc_get_nb(int gpu_arch, char trans_A, int q_comp, int P, int Q, int N) {
rust/libceed-sys/c-src/backends/magma/ceed-magma-gemm-selector.cpp:  const auto &data = nontensor_rtc_get_data(gpu_arch, trans_A);
rust/libceed-sys/c-src/backends/magma/ceed-magma-gemm-selector.h:CEED_INTERN void gemm_selector(int gpu_arch, char precision, char trans_A, int m, int n, int k, int *n_batch, int *use_magma);
rust/libceed-sys/c-src/backends/magma/ceed-magma-gemm-selector.h:CEED_INTERN CeedInt nontensor_rtc_get_nb(int gpu_arch, char trans_A, int q_comp, int P, int Q, int N);
rust/libceed-sys/c-src/backends/magma/ceed-magma.h:#define CeedCompileMagma CeedCompile_Cuda
rust/libceed-sys/c-src/backends/magma/ceed-magma.h:#define CeedGetKernelMagma CeedGetKernel_Cuda
rust/libceed-sys/c-src/backends/magma/ceed-magma.h:#define CeedRunKernelMagma CeedRunKernel_Cuda
rust/libceed-sys/c-src/backends/magma/ceed-magma.h:#define CeedRunKernelDimMagma CeedRunKernelDim_Cuda
rust/libceed-sys/c-src/backends/magma/ceed-magma.h:#define CeedRunKernelDimSharedMagma CeedRunKernelDimShared_Cuda
rust/libceed-sys/c-src/backends/magma/ceed-magma.h:// If magma and cuda/ref are using the null stream, then ceed_magma_queue_sync should do nothing
rust/libceed-sys/c-src/backends/magma/tuning/mi100.h:// auto-generated from data on mi100-rocm5.0.2
rust/libceed-sys/c-src/backends/magma/tuning/generate_tuning.py:            ("hipDeviceSynchronize()" if "hip" in backend else "cudaDeviceSynchronize()"),
rust/libceed-sys/c-src/backends/magma/tuning/README.md:The `magma` backend uses specialized GPU kernels for a non-tensor basis with
rust/libceed-sys/c-src/backends/magma/tuning/README.md:header files called `<ARCH>_rtc.h`, where `<ARCH>` is the GPU name, as well as a
rust/libceed-sys/c-src/backends/magma/tuning/README.md:A sample run to generate the tuning data for an A100 GPU, considering values of
rust/libceed-sys/c-src/backends/magma/tuning/README.md:python generate_tuning.py -arch a100 -max-nb 32 -build-cmd "make" -ceed "/gpu/cuda/magma"
rust/libceed-sys/c-src/backends/magma/tuning/README.md:specifies the backend to use, typically one of `/gpu/cuda/magma` or
rust/libceed-sys/c-src/backends/magma/tuning/README.md:`/gpu/hip/magma`.
rust/libceed-sys/c-src/backends/magma/tuning/README.md:./tuning "/gpu/cuda/magma"
rust/libceed-sys/c-src/backends/magma/tuning/README.md:`cudaDeviceSynchronize()` or `hipDeviceSynchronize()`.
rust/libceed-sys/c-src/backends/magma/tuning/mi250x.h:// auto-generated from data on mi250x-rocm5.1.0
rust/libceed-sys/c-src/backends/magma/tuning/v100.h:// auto-generated from data on v100-cuda11.2
rust/libceed-sys/c-src/backends/magma/tuning/a100.h:// auto-generated from data on a100-cuda11.2
rust/libceed-sys/c-src/backends/magma/ceed-magma.c:  CeedCheck(!strncmp(resource, "/gpu/cuda/magma", nrc) || !strncmp(resource, "/gpu/hip/magma", nrc), ceed, CEED_ERROR_BACKEND,
rust/libceed-sys/c-src/backends/magma/ceed-magma.c:  CeedCallBackend(CeedInit("/gpu/hip/ref", &ceed_ref));
rust/libceed-sys/c-src/backends/magma/ceed-magma.c:  CeedCallBackend(CeedInit("/gpu/cuda/ref", &ceed_ref));
rust/libceed-sys/c-src/backends/magma/ceed-magma.c:  return CeedRegister("/gpu/hip/magma", CeedInit_Magma, 120);
rust/libceed-sys/c-src/backends/magma/ceed-magma.c:  return CeedRegister("/gpu/cuda/magma", CeedInit_Magma, 120);
rust/libceed-sys/c-src/backends/magma/ceed-magma-det.c:  CeedCheck(!strncmp(resource, "/gpu/cuda/magma/det", nrc) || !strncmp(resource, "/gpu/hip/magma/det", nrc), ceed, CEED_ERROR_BACKEND,
rust/libceed-sys/c-src/backends/magma/ceed-magma-det.c:  CeedCallBackend(CeedInit("/gpu/hip/magma", &ceed_ref));
rust/libceed-sys/c-src/backends/magma/ceed-magma-det.c:  CeedCallBackend(CeedInit("/gpu/cuda/magma", &ceed_ref));
rust/libceed-sys/c-src/backends/magma/ceed-magma-det.c:  return CeedRegister("/gpu/hip/magma/det", CeedInit_Magma_Det, 125);
rust/libceed-sys/c-src/backends/magma/ceed-magma-det.c:  return CeedRegister("/gpu/cuda/magma/det", CeedInit_Magma_Det, 125);
rust/libceed-sys/c-src/backends/magma/ceed-magma-basis.c:#include "../cuda/ceed-cuda-common.h"
rust/libceed-sys/c-src/backends/magma/ceed-magma-basis.c:#include "../cuda/ceed-cuda-compile.h"
rust/libceed-sys/c-src/backends/magma/ceed-magma-basis.c:      // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
rust/libceed-sys/c-src/backends/magma/ceed-magma-basis.c:  CeedCallCuda(ceed, cuModuleUnload(impl->module));
rust/libceed-sys/c-src/backends/magma/ceed-magma-basis.c:      CeedCallCuda(ceed, cuModuleUnload(impl->module[in]));
rust/libceed-sys/c-src/backends/magma/ceed-magma-basis.c:  // Copy basis data to GPU
rust/libceed-sys/c-src/backends/magma/ceed-magma-basis.c:  // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
rust/libceed-sys/c-src/backends/magma/ceed-magma-basis.c:  // Copy basis data to GPU
rust/libceed-sys/c-src/backends/magma/ceed-magma-basis.c:    // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
rust/libceed-sys/c-src/backends/magma/ceed-magma-basis.c:  // Copy basis data to GPU
rust/libceed-sys/c-src/backends/magma/ceed-magma-basis.c:    // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
rust/libceed-sys/c-src/backends/magma/ceed-magma-basis.c:  // Copy basis data to GPU
rust/libceed-sys/c-src/backends/magma/ceed-magma-basis.c:    // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
rust/libceed-sys/c-src/backends/hip-gen/ceed-hip-gen.c:  const char fallback_resource[] = "/gpu/hip/ref";
rust/libceed-sys/c-src/backends/hip-gen/ceed-hip-gen.c:  CeedCheck(!strcmp(resource_root, "/gpu/hip") || !strcmp(resource_root, "/gpu/hip/gen"), ceed, CEED_ERROR_BACKEND,
rust/libceed-sys/c-src/backends/hip-gen/ceed-hip-gen.c:  CeedCallBackend(CeedInit("/gpu/hip/shared", &ceed_shared));
rust/libceed-sys/c-src/backends/hip-gen/ceed-hip-gen.c:CEED_INTERN int CeedRegister_Hip_Gen(void) { return CeedRegister("/gpu/hip/gen", CeedInit_Hip_gen, 20); }
rust/libceed-sys/c-src/backends/hip-gen/ceed-hip-gen-operator.c:      CeedDebug256(CeedOperatorReturnCeed(op), CEED_DEBUG_COLOR_SUCCESS, "Falling back to /gpu/hip/ref CeedOperator due to non-tensor bases");
rust/libceed-sys/c-src/backends/hip-gen/ceed-hip-gen-operator-build.cpp:    CeedCheck(source_path, ceed, CEED_ERROR_UNSUPPORTED, "/gpu/hip/gen backend requires QFunction source code file");
rust/libceed-sys/c-src/Makefile:NVCC ?= $(CUDA_DIR)/bin/nvcc
rust/libceed-sys/c-src/Makefile:HIPCC ?= $(ROCM_DIR)/bin/hipcc
rust/libceed-sys/c-src/Makefile:# Often /opt/cuda or /usr/local/cuda, but sometimes present on machines that don't support CUDA
rust/libceed-sys/c-src/Makefile:CUDA_DIR  ?=
rust/libceed-sys/c-src/Makefile:CUDA_ARCH ?=
rust/libceed-sys/c-src/Makefile:# Often /opt/rocm, but sometimes present on machines that don't support HIP
rust/libceed-sys/c-src/Makefile:ROCM_DIR ?=
rust/libceed-sys/c-src/Makefile:# Warning: SANTIZ options still don't run with /gpu/occa
rust/libceed-sys/c-src/Makefile:ifneq ($(CUDA_ARCH),)
rust/libceed-sys/c-src/Makefile:  NVCCFLAGS += -arch=$(CUDA_ARCH)
rust/libceed-sys/c-src/Makefile:  HIPCCFLAGS += --amdgpu-target=$(HIP_ARCH)
rust/libceed-sys/c-src/Makefile:libceed.c := $(filter-out interface/ceed-cuda.c interface/ceed-hip.c interface/ceed-jit-source-root-$(if $(for_install),default,install).c, $(wildcard interface/ceed*.c backends/*.c gallery/*.c))
rust/libceed-sys/c-src/Makefile:cuda.c         := $(sort $(wildcard backends/cuda/*.c))
rust/libceed-sys/c-src/Makefile:cuda.cpp       := $(sort $(wildcard backends/cuda/*.cpp))
rust/libceed-sys/c-src/Makefile:cuda-ref.c     := $(sort $(wildcard backends/cuda-ref/*.c))
rust/libceed-sys/c-src/Makefile:cuda-ref.cpp   := $(sort $(wildcard backends/cuda-ref/*.cpp))
rust/libceed-sys/c-src/Makefile:cuda-ref.cu    := $(sort $(wildcard backends/cuda-ref/kernels/*.cu))
rust/libceed-sys/c-src/Makefile:cuda-shared.c  := $(sort $(wildcard backends/cuda-shared/*.c))
rust/libceed-sys/c-src/Makefile:cuda-shared.cu := $(sort $(wildcard backends/cuda-shared/kernels/*.cu))
rust/libceed-sys/c-src/Makefile:cuda-gen.c     := $(sort $(wildcard backends/cuda-gen/*.c))
rust/libceed-sys/c-src/Makefile:cuda-gen.cpp   := $(sort $(wildcard backends/cuda-gen/*.cpp))
rust/libceed-sys/c-src/Makefile:cuda-gen.cu    := $(sort $(wildcard backends/cuda-gen/kernels/*.cu))
rust/libceed-sys/c-src/Makefile:	$(info CUDA_DIR      = $(CUDA_DIR)$(call backend_status,$(CUDA_BACKENDS)))
rust/libceed-sys/c-src/Makefile:	$(info ROCM_DIR      = $(ROCM_DIR)$(call backend_status,$(HIP_BACKENDS)))
rust/libceed-sys/c-src/Makefile:  OCCA_BACKENDS += $(if $(filter dpcpp,$(OCCA_MODES)),/gpu/dpcpp/occa)
rust/libceed-sys/c-src/Makefile:  OCCA_BACKENDS += $(if $(filter OpenCL,$(OCCA_MODES)),/gpu/opencl/occa)
rust/libceed-sys/c-src/Makefile:  OCCA_BACKENDS += $(if $(filter HIP,$(OCCA_MODES)),/gpu/hip/occa)
rust/libceed-sys/c-src/Makefile:  OCCA_BACKENDS += $(if $(filter CUDA,$(OCCA_MODES)),/gpu/cuda/occa)
rust/libceed-sys/c-src/Makefile:# CUDA Backends
rust/libceed-sys/c-src/Makefile:ifneq ($(CUDA_DIR),)
rust/libceed-sys/c-src/Makefile:  CUDA_LIB_DIR := $(wildcard $(foreach d,lib lib64 lib/x86_64-linux-gnu,$(CUDA_DIR)/$d/libcudart.${SO_EXT}))
rust/libceed-sys/c-src/Makefile:  CUDA_LIB_DIR := $(patsubst %/,%,$(dir $(firstword $(CUDA_LIB_DIR))))
rust/libceed-sys/c-src/Makefile:CUDA_LIB_DIR_STUBS := $(CUDA_LIB_DIR)/stubs
rust/libceed-sys/c-src/Makefile:CUDA_BACKENDS = /gpu/cuda/ref /gpu/cuda/shared /gpu/cuda/gen
rust/libceed-sys/c-src/Makefile:ifneq ($(CUDA_LIB_DIR),)
rust/libceed-sys/c-src/Makefile:  $(libceeds) : CPPFLAGS += -I$(CUDA_DIR)/include
rust/libceed-sys/c-src/Makefile:  PKG_LIBS += -L$(abspath $(CUDA_LIB_DIR)) -lcudart -lnvrtc -lcuda -lcublas
rust/libceed-sys/c-src/Makefile:  PKG_STUBS_LIBS += -L$(CUDA_LIB_DIR_STUBS)
rust/libceed-sys/c-src/Makefile:  libceed.c     += interface/ceed-cuda.c
rust/libceed-sys/c-src/Makefile:  libceed.c     += $(cuda.c) $(cuda-ref.c) $(cuda-shared.c) $(cuda-gen.c)
rust/libceed-sys/c-src/Makefile:  libceed.cpp   += $(cuda.cpp) $(cuda-ref.cpp) $(cuda-gen.cpp)
rust/libceed-sys/c-src/Makefile:  libceed.cu    += $(cuda-ref.cu) $(cuda-shared.cu) $(cuda-gen.cu)
rust/libceed-sys/c-src/Makefile:  BACKENDS_MAKE += $(CUDA_BACKENDS)
rust/libceed-sys/c-src/Makefile:HIP_LIB_DIR := $(wildcard $(foreach d,lib lib64,$(ROCM_DIR)/$d/libamdhip64.${SO_EXT}))
rust/libceed-sys/c-src/Makefile:HIP_BACKENDS = /gpu/hip/ref /gpu/hip/shared /gpu/hip/gen
rust/libceed-sys/c-src/Makefile:  HIPCONFIG_CPPFLAGS := $(subst =,,$(shell $(ROCM_DIR)/bin/hipconfig -C))
rust/libceed-sys/c-src/Makefile:SYCL_BACKENDS = /gpu/sycl/ref /gpu/sycl/shared /gpu/sycl/gen
rust/libceed-sys/c-src/Makefile:  ifeq ($(MAGMA_ARCH), 0)  # CUDA MAGMA
rust/libceed-sys/c-src/Makefile:    ifneq ($(CUDA_LIB_DIR),)
rust/libceed-sys/c-src/Makefile:      cuda_link = $(if $(STATIC),,-Wl,-rpath,$(CUDA_LIB_DIR)) -L$(CUDA_LIB_DIR) -lcublas -lcusparse -lcudart
rust/libceed-sys/c-src/Makefile:      magma_link_static = -L$(MAGMA_DIR)/lib -lmagma $(cuda_link) $(omp_link)
rust/libceed-sys/c-src/Makefile:      $(magma.c:%.c=$(OBJDIR)/%.o) $(magma.c:%=%.tidy) : CPPFLAGS += -DADD_ -I$(MAGMA_DIR)/include -I$(CUDA_DIR)/include
rust/libceed-sys/c-src/Makefile:      $(magma.cpp:%.cpp=$(OBJDIR)/%.o) $(magma.cpp:%=%.tidy) : CPPFLAGS += -DADD_ -I$(MAGMA_DIR)/include -I$(CUDA_DIR)/include
rust/libceed-sys/c-src/Makefile:      MAGMA_BACKENDS = /gpu/cuda/magma /gpu/cuda/magma/det
rust/libceed-sys/c-src/Makefile:      $(magma.c:%.c=$(OBJDIR)/%.o) $(magma.c:%=%.tidy) : CPPFLAGS += $(HIPCONFIG_CPPFLAGS) -I$(MAGMA_DIR)/include -I$(ROCM_DIR)/include -DCEED_MAGMA_USE_HIP -DADD_
rust/libceed-sys/c-src/Makefile:      $(magma.cpp:%.cpp=$(OBJDIR)/%.o) $(magma.cpp:%=%.tidy) : CPPFLAGS += $(HIPCONFIG_CPPFLAGS) -I$(MAGMA_DIR)/include -I$(ROCM_DIR)/include -DCEED_MAGMA_USE_HIP -DADD_
rust/libceed-sys/c-src/Makefile:      MAGMA_BACKENDS = /gpu/hip/magma /gpu/hip/magma/det
rust/libceed-sys/c-src/Makefile:	  "$(includedir)/ceed/jit-source/cuda/" "$(includedir)/ceed/jit-source/hip/"\
rust/libceed-sys/c-src/Makefile:	$(INSTALL_DATA) include/ceed/cuda.h "$(DESTDIR)$(includedir)/ceed/"
rust/libceed-sys/c-src/Makefile:	$(INSTALL_DATA) $(wildcard include/ceed/jit-source/cuda/*.h) "$(DESTDIR)$(includedir)/ceed/jit-source/cuda/"
rust/libceed-sys/c-src/Makefile:	$(CLANG_TIDY) $(TIDY_OPTS) $^ -- $(CPPFLAGS) --std=c99 -I$(CUDA_DIR)/include -I$(ROCM_DIR)/include -DCEED_JIT_SOURCE_ROOT_DEFAULT="\"$(abspath ./include)/\""
rust/libceed-sys/c-src/Makefile:	$(CLANG_TIDY) $(TIDY_OPTS) $^ -- $(CPPFLAGS) --std=c++11 -I$(CUDA_DIR)/include -I$(OCCA_DIR)/include -I$(ROCM_DIR)/include
rust/libceed-sys/c-src/Makefile:#   make configure CC=/path/to/my/cc CUDA_DIR=/opt/cuda
rust/libceed-sys/c-src/Makefile:  MAGMA_DIR OCCA_DIR XSMM_DIR CUDA_DIR CUDA_ARCH MFEM_DIR PETSC_DIR NEK5K_DIR ROCM_DIR HIP_ARCH SYCL_DIR
rust/libceed-sys/c-src/interface/ceed-qfunction.c:                           The entire source file must only contain constructs supported by all targeted backends (i.e. CUDA for `/gpu/cuda`, OpenCL/SYCL for `/gpu/sycl`, etc.).
rust/libceed-sys/c-src/interface/ceed-qfunction.c:                           The entire contents of this file and all locally included files are used during JiT compilation for GPU backends.
rust/libceed-sys/c-src/interface/ceed-qfunction.c:  Setting `is_writable == false` may offer a performance improvement on GPU backends.
rust/libceed-sys/c-src/interface/ceed-basis.c:  // Create TensorContract object if needed, such as a basis from the GPU backends
rust/libceed-sys/c-src/interface/ceed-vector.c:  // Backend impl for GPU, if added
rust/libceed-sys/c-src/interface/ceed-vector.c:  // Backend impl for GPU, if added
rust/libceed-sys/c-src/interface/ceed.c:      CEED_FTABLE_ENTRY(CeedQFunction, SetCUDAUserFunction),
rust/libceed-sys/c-src/interface/ceed.c:  @brief Set the GPU stream for a `Ceed` context
rust/libceed-sys/c-src/interface/ceed.c:  @param[in]     handle Handle to GPU stream
rust/libceed-sys/c-src/interface/ceed-cuda.c:#include <ceed/cuda.h>
rust/libceed-sys/c-src/interface/ceed-cuda.c:#include <cuda.h>
rust/libceed-sys/c-src/interface/ceed-cuda.c:  @brief Set CUDA function pointer to evaluate action at quadrature points
rust/libceed-sys/c-src/interface/ceed-cuda.c:int CeedQFunctionSetCUDAUserFunction(CeedQFunction qf, CUfunction f) {
rust/libceed-sys/c-src/interface/ceed-cuda.c:  if (!qf->SetCUDAUserFunction) {
rust/libceed-sys/c-src/interface/ceed-cuda.c:    CeedCall(qf->SetCUDAUserFunction(qf, f));
rust/libceed-sys/c-src/include/ceed-impl.h:  int (*SetCUDAUserFunction)(CeedQFunction, void *);
rust/libceed-sys/c-src/include/ceed/jit-source/sycl/sycl-types.h:#ifdef __OPENCL_C_VERSION__
rust/libceed-sys/c-src/include/ceed/jit-source/sycl/sycl-gen-templates.h:#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
rust/libceed-sys/c-src/include/ceed/jit-source/sycl/sycl-gen-templates.h:#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
rust/libceed-sys/c-src/include/ceed/jit-source/hip/hip-ref-basis-tensor-at-points.h:/// Internal header for CUDA tensor product basis with AtPoints evaluation
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:/// Internal header for CUDA shared memory basis read/write templates
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void ReadElementStrided1d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void WriteElementStrided1d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void SumElementStrided1d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void ReadElementStrided2d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void WriteElementStrided2d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void SumElementStrided2d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void ReadElementStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void WriteElementStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void SumElementStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-ref-basis-tensor.h:/// Internal header for CUDA tensor product basis
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-ref-basis-tensor-at-points.h:/// Internal header for CUDA tensor product basis with AtPoints evaluation
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-ref-operator-assemble.h:/// Internal header for CUDA operator full assembly
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-ref-basis-nontensor.h:/// Internal header for CUDA non-tensor product basis
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-ref-basis-nontensor.h:#include "cuda-ref-basis-nontensor-templates.h"
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:/// Internal header for CUDA backend macro and type definitions for JiT source
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void loadMatrix(SharedData_Cuda &data, const CeedScalar *__restrict__ d_B, CeedScalar *B) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void readDofsOffset1d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void readDofsStrided1d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *__restrict__ d_u,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void writeDofsOffset1d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void writeDofsStrided1d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *__restrict__ r_v,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void readDofsOffset2d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void readDofsStrided2d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *__restrict__ d_u,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void writeDofsOffset2d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void writeDofsStrided2d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *__restrict__ r_v,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void readDofsOffset3d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void readDofsStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *__restrict__ d_u,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void readSliceQuadsOffset3d(SharedData_Cuda &data, const CeedInt nquads, const CeedInt elem, const CeedInt q,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void readSliceQuadsStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedInt q, const CeedScalar *__restrict__ d_u,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void writeDofsOffset3d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void writeDofsStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *__restrict__ r_v,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void gradCollo3d(SharedData_Cuda &data, const CeedInt q, const CeedScalar *__restrict__ r_U, const CeedScalar *c_G,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void gradColloTranspose3d(SharedData_Cuda &data, const CeedInt q, const CeedScalar *__restrict__ r_U, const CeedScalar *c_G,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:/// Internal header for CUDA shared memory tensor product basis
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:#include "cuda-shared-basis-read-write-templates.h"
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:#include "cuda-shared-basis-tensor-templates.h"
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:  SharedData_Cuda data;
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:  SharedData_Cuda data;
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:  SharedData_Cuda data;
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:  SharedData_Cuda data;
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:  SharedData_Cuda data;
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:  SharedData_Cuda data;
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:  SharedData_Cuda data;
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-ref-operator-assemble-diagonal.h:/// Internal header for CUDA operator diagonal assembly
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-ref-restriction-offset.h:/// Internal header for CUDA offset element restriction kernels
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-ref-qfunction.h:/// Internal header for CUDA backend QFunction read/write kernels
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-types.h:/// Internal header for CUDA type definitions
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-types.h:#define CEED_CUDA_NUMBER_FIELDS 16
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-types.h:  const CeedScalar *inputs[CEED_CUDA_NUMBER_FIELDS];
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-types.h:  CeedScalar       *outputs[CEED_CUDA_NUMBER_FIELDS];
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-types.h:} Fields_Cuda;
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-types.h:  CeedInt *inputs[CEED_CUDA_NUMBER_FIELDS];
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-types.h:  CeedInt *outputs[CEED_CUDA_NUMBER_FIELDS];
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-types.h:} FieldsInt_Cuda;
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-types.h:} SharedData_Cuda;
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-ref-restriction-strided.h:/// Internal header for CUDA strided element restriction kernels
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-ref-restriction-curl-oriented.h:/// Internal header for CUDA curl-oriented element restriction kernels
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:/// Internal header for CUDA shared memory tensor product basis templates
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractX1d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeX1d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void Interp1d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, CeedScalar *__restrict__ r_V) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void InterpTranspose1d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void Grad1d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void GradTranspose1d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void Weight1d(SharedData_Cuda &data, const CeedScalar *__restrict__ q_weight_1d, CeedScalar *w) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractX2d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractY2d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeY2d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeX2d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeAddX2d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void InterpTensor2d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void InterpTransposeTensor2d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void GradTensor2d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void GradTransposeTensor2d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void WeightTensor2d(SharedData_Cuda &data, const CeedScalar *__restrict__ q_weight_1d, CeedScalar *w) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractX3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractY3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractZ3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeZ3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeY3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeAddY3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeX3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeAddX3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void InterpTensor3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void InterpTransposeTensor3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void GradTensor3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void GradTransposeTensor3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void GradTensorCollocated3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void GradTransposeTensorCollocated3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void WeightTensor3d(SharedData_Cuda &data, const CeedScalar *__restrict__ q_weight_1d, CeedScalar *w) {
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-ref-restriction-oriented.h:/// Internal header for CUDA oriented element restriction kernels
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-ref-restriction-at-points.h:/// Internal header for CUDA offset element restriction kernels
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-atomic-add-fallback.h:/// Internal header for CUDA atomic add fallback definition
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-atomic-add-fallback.h:// Atomic add, for older CUDA
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-ref-basis-nontensor-templates.h:/// Internal header for CUDA non-tensor product basis templates
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-jit.h:/// Internal header for CUDA backend macro and type definitions for JiT source
rust/libceed-sys/c-src/include/ceed/jit-source/cuda/cuda-jit.h:#include "cuda-types.h"
rust/libceed-sys/c-src/include/ceed/types.h:    VLA is a C99 feature that is not supported by the C++ dialect used by CUDA.
rust/libceed-sys/c-src/include/ceed/types.h:    This macro allows users to use the VLA syntax with the CUDA backends.
rust/libceed-sys/c-src/include/ceed/cuda.h:/// Public header for CUDA utility components of libCEED
rust/libceed-sys/c-src/include/ceed/cuda.h:#include <cuda.h>
rust/libceed-sys/c-src/include/ceed/cuda.h:CEED_EXTERN int CeedQFunctionSetCUDAUserFunction(CeedQFunction qf, CUfunction f);
rust/libceed/README.md:The resource string passed to `Ceed::init` is used to identify the "backend", which includes algorithmic strategies and hardware such as NVIDIA and AMD GPUs.
backends/sycl/online_compiler.sycl.cpp:  if (DeviceType == sycl::info::device_type::gpu) {
backends/sycl/online_compiler.sycl.cpp:      case device_arch::gpu_gen9:
backends/sycl/online_compiler.sycl.cpp:      case device_arch::gpu_gen9_5:
backends/sycl/online_compiler.sycl.cpp:      case device_arch::gpu_gen11:
backends/sycl/online_compiler.sycl.cpp:      case device_arch::gpu_gen12:
backends/sycl/online_compiler.sycl.cpp:    // For now "tgllp" is used as the option supported on all known GPU RT.
backends/sycl/online_compiler.sycl.cpp:/// @param Source - Either OpenCL or CM source code.
backends/sycl/online_compiler.sycl.cpp:/// @param DeviceType - SYCL device type, e.g. cpu, gpu, accelerator, etc.
backends/sycl/online_compiler.sycl.cpp:std::vector<byte> online_compiler<source_language::opencl_c>::compile(const std::string &Source, const std::vector<std::string> &UserArgs) {
backends/sycl/ceed-sycl-compile.sycl.cpp:// Compile an OpenCL source to SPIR-V using Intel's online compiler extension
backends/sycl/ceed-sycl-compile.sycl.cpp:static inline int CeedJitCompileSource_Sycl(Ceed ceed, const sycl::device &sycl_device, const std::string &opencl_source, ByteVector_t &il_binary,
backends/sycl/ceed-sycl-compile.sycl.cpp:  sycl::ext::libceed::online_compiler<sycl::ext::libceed::source_language::opencl_c> compiler(sycl_device);
backends/sycl/ceed-sycl-compile.sycl.cpp:    il_binary = compiler.compile(opencl_source, flags);
backends/sycl/ceed-sycl-common.sycl.cpp:  if (std::strstr(resource, "/gpu/sycl")) {
backends/sycl/ceed-sycl-common.sycl.cpp:    device_type = sycl::info::device_type::gpu;
backends/sycl/online_compiler.hpp:  // TODO1: the list must be extended with a bunch of new GPUs available.
backends/sycl/online_compiler.hpp:  // TODO2: the list of supported GPUs grows rapidly.
backends/sycl/online_compiler.hpp:  // The API must allow user to define the target GPU option even if it is
backends/sycl/online_compiler.hpp:  enum gpu {
backends/sycl/online_compiler.hpp:    gpu_any    = 1,
backends/sycl/online_compiler.hpp:    gpu_gen9   = 2,
backends/sycl/online_compiler.hpp:    gpu_skl    = gpu_gen9,
backends/sycl/online_compiler.hpp:    gpu_gen9_5 = 3,
backends/sycl/online_compiler.hpp:    gpu_kbl    = gpu_gen9_5,
backends/sycl/online_compiler.hpp:    gpu_cfl    = gpu_gen9_5,
backends/sycl/online_compiler.hpp:    gpu_gen11  = 4,
backends/sycl/online_compiler.hpp:    gpu_icl    = gpu_gen11,
backends/sycl/online_compiler.hpp:    gpu_gen12  = 5,
backends/sycl/online_compiler.hpp:    gpu_tgl    = gpu_gen12,
backends/sycl/online_compiler.hpp:    gpu_tgllp  = gpu_gen12
backends/sycl/online_compiler.hpp:enum class source_language { opencl_c = 0, cm = 1 };
backends/sycl/online_compiler.hpp:/// Compiles the given OpenCL source. May throw \c online_compile_error.
backends/sycl/online_compiler.hpp:///   OpenCL JIT compiler options must be supported.
backends/sycl/online_compiler.hpp:std::vector<byte> online_compiler<source_language::opencl_c>::compile(const std::string &src, const std::vector<std::string> &options);
backends/sycl/online_compiler.hpp:// /// Compiles the given OpenCL source. May throw \c online_compile_error.
backends/sycl/online_compiler.hpp:// online_compiler<source_language::opencl_c>::compile(const std::string &src) {
backends/occa/ceed-occa-context.cpp:  _usingGpuDevice        = (mode == "CUDA" || mode == "HIP" || mode == "OpenCL");
backends/occa/ceed-occa-context.cpp:bool Context::usingGpuDevice() const { return _usingGpuDevice; }
backends/occa/ceed-occa-gpu-operator.hpp:#ifndef CEED_OCCA_GPU_OPERATOR_HEADER
backends/occa/ceed-occa-gpu-operator.hpp:#define CEED_OCCA_GPU_OPERATOR_HEADER
backends/occa/ceed-occa-gpu-operator.hpp:class GpuOperator : public Operator {
backends/occa/ceed-occa-gpu-operator.hpp:  GpuOperator();
backends/occa/ceed-occa-gpu-operator.hpp:  ~GpuOperator();
backends/occa/ceed-occa-tensor-basis.cpp:  if (usingGpuDevice()) {
backends/occa/ceed-occa-tensor-basis.cpp:  // TODO: Add gpu function sources when split
backends/occa/ceed-occa-tensor-basis.cpp:  const char *gpuKernelSources[3]   = {occa_tensor_basis_1d_gpu_source, occa_tensor_basis_2d_gpu_source, occa_tensor_basis_3d_gpu_source};
backends/occa/ceed-occa-tensor-basis.cpp:  if (usingGpuDevice()) {
backends/occa/ceed-occa-tensor-basis.cpp:    kernelSource = gpuKernelSources[dim - 1];
backends/occa/ceed-occa-tensor-basis.cpp:  if (Q1D < P1D && Context::from(ceed)->usingGpuDevice()) {
backends/occa/ceed-occa.cpp:static std::string getDefaultDeviceMode(const bool cpuMode, const bool gpuMode) {
backends/occa/ceed-occa.cpp:  // In case both cpuMode and gpuMode are set, prioritize the GPU if available
backends/occa/ceed-occa.cpp:  if (gpuMode) {
backends/occa/ceed-occa.cpp:    if (::occa::modeIsEnabled("CUDA")) {
backends/occa/ceed-occa.cpp:      return "CUDA";
backends/occa/ceed-occa.cpp:    if (::occa::modeIsEnabled("OpenCL")) {
backends/occa/ceed-occa.cpp:      return "OpenCL";
backends/occa/ceed-occa.cpp:  if (match == "cuda") {
backends/occa/ceed-occa.cpp:    mode = "CUDA";
backends/occa/ceed-occa.cpp:  if (match == "opencl") {
backends/occa/ceed-occa.cpp:    mode = "OpenCL";
backends/occa/ceed-occa.cpp:  const bool gpuMode  = match == "gpu";
backends/occa/ceed-occa.cpp:  mode = getDefaultDeviceMode(cpuMode || autoMode, gpuMode || autoMode);
backends/occa/ceed-occa.cpp:   *    "/gpu/occa?mode='CUDA':device_id=0"
backends/occa/ceed-occa.cpp:   *    ["gpu", "occa"]
backends/occa/ceed-occa.cpp:   *    "gpu"
backends/occa/ceed-occa.cpp:   *      "mode": "'CUDA'",
backends/occa/ceed-occa.cpp:  // Check for /gpu/cuda/occa, /gpu/hip/occa, /cpu/self/occa, /cpu/openmp/occa
backends/occa/ceed-occa.cpp:  if (resource == "/gpu/cuda/occa") {
backends/occa/ceed-occa.cpp:    match = "cuda";
backends/occa/ceed-occa.cpp:  if (resource == "/gpu/hip/occa") {
backends/occa/ceed-occa.cpp:  if (resource == "/gpu/dpcpp/occa") {
backends/occa/ceed-occa.cpp:  if (resource == "/gpu/opencl/occa") {
backends/occa/ceed-occa.cpp:    match = "opencl";
backends/occa/ceed-occa.cpp:  if ((mode == "CUDA") || (mode == "HIP") || (mode == "dpcpp") || (mode == "OpenCL")) {
backends/occa/ceed-occa.cpp:  if ((mode == "dpcpp") || (mode == "OpenCL")) {
backends/occa/ceed-occa.cpp:  // GPU Modes
backends/occa/ceed-occa.cpp:  CeedCallBackend(CeedRegister("/gpu/dpcpp/occa", ceed::occa::registerBackend, 240));
backends/occa/ceed-occa.cpp:  CeedCallBackend(CeedRegister("/gpu/opencl/occa", ceed::occa::registerBackend, 230));
backends/occa/ceed-occa.cpp:  CeedCallBackend(CeedRegister("/gpu/hip/occa", ceed::occa::registerBackend, 220));
backends/occa/ceed-occa.cpp:  CeedCallBackend(CeedRegister("/gpu/cuda/occa", ceed::occa::registerBackend, 210));
backends/occa/ceed-occa-ceed-object.cpp:bool CeedObject::usingGpuDevice() const { return Context::from(ceed)->usingGpuDevice(); }
backends/occa/ceed-occa-ceed-object.hpp:  bool usingGpuDevice() const;
backends/occa/ceed-occa-context.hpp:  bool _usingGpuDevice;
backends/occa/ceed-occa-context.hpp:  bool usingGpuDevice() const;
backends/occa/kernels/tensor-basis.hpp:// Kernels are based on the cuda backend from LLNL and VT groups
backends/occa/kernels/tensor-basis.hpp:extern const char *occa_tensor_basis_1d_gpu_source;
backends/occa/kernels/tensor-basis.hpp:extern const char *occa_tensor_basis_2d_gpu_source;
backends/occa/kernels/tensor-basis.hpp:extern const char *occa_tensor_basis_3d_gpu_source;
backends/occa/kernels/elem-restriction.hpp:// Kernels are based on the cuda backend from LLNL and VT groups
backends/occa/kernels/simplex-basis/gpu-simplex-basis.cpp:const char *occa_simplex_basis_gpu_source = STRINGIFY_SOURCE(
backends/occa/kernels/simplex-basis.hpp:// Kernels are based on the cuda backend from LLNL and VT groups
backends/occa/kernels/simplex-basis.hpp:extern const char *occa_simplex_basis_gpu_source;
backends/occa/kernels/tensor-basis/gpu/tensor-basis-2d.cpp:const char *occa_tensor_basis_2d_gpu_source = STRINGIFY_SOURCE(
backends/occa/kernels/tensor-basis/gpu/tensor-basis-1d.cpp:const char *occa_tensor_basis_1d_gpu_source = STRINGIFY_SOURCE(
backends/occa/kernels/tensor-basis/gpu/tensor-basis-3d.cpp:const char *occa_tensor_basis_3d_gpu_source = STRINGIFY_SOURCE(
backends/occa/kernels/elem-restriction.cpp:// Kernels are based on the cuda backend from LLNL and VT groups
backends/occa/ceed-occa-simplex-basis.cpp:  // TODO: Add gpu function sources when split
backends/occa/ceed-occa-simplex-basis.cpp:  if (usingGpuDevice()) {
backends/occa/ceed-occa-simplex-basis.cpp:  if (usingGpuDevice()) {
backends/occa/ceed-occa-simplex-basis.cpp:    kernelSource = occa_simplex_basis_gpu_source;
backends/occa/ceed-occa-gpu-operator.cpp:#include "ceed-occa-gpu-operator.hpp"
backends/occa/ceed-occa-gpu-operator.cpp:GpuOperator::GpuOperator() {}
backends/occa/ceed-occa-gpu-operator.cpp:GpuOperator::~GpuOperator() {}
backends/occa/ceed-occa-gpu-operator.cpp:::occa::kernel GpuOperator::buildApplyAddKernel() { return ::occa::kernel(); }
backends/occa/ceed-occa-gpu-operator.cpp:void GpuOperator::applyAdd(Vector *in, Vector *out) {
backends/occa/ceed-occa-operator.cpp:#include "ceed-occa-gpu-operator.hpp"
backends/occa/ceed-occa-operator.cpp:  // TODO: Add GPU specific operator
backends/occa/ceed-occa-operator.cpp:  Operator *operator_ = (Context::from(ceed)->usingCpuDevice() ? ((Operator *)new CpuOperator()) : ((Operator *)new GpuOperator()));
backends/cuda-shared/ceed-cuda-shared.c:#include "ceed-cuda-shared.h"
backends/cuda-shared/ceed-cuda-shared.c:#include "../cuda/ceed-cuda-common.h"
backends/cuda-shared/ceed-cuda-shared.c:static int CeedInit_Cuda_shared(const char *resource, Ceed ceed) {
backends/cuda-shared/ceed-cuda-shared.c:  Ceed_Cuda *data;
backends/cuda-shared/ceed-cuda-shared.c:  CeedCheck(!strcmp(resource_root, "/gpu/cuda/shared"), ceed, CEED_ERROR_BACKEND, "Cuda backend cannot use resource: %s", resource);
backends/cuda-shared/ceed-cuda-shared.c:  CeedCallBackend(CeedInit_Cuda(ceed, resource));
backends/cuda-shared/ceed-cuda-shared.c:  CeedCallBackend(CeedInit("/gpu/cuda/ref", &ceed_ref));
backends/cuda-shared/ceed-cuda-shared.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1", CeedBasisCreateTensorH1_Cuda_shared));
backends/cuda-shared/ceed-cuda-shared.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Cuda));
backends/cuda-shared/ceed-cuda-shared.c:CEED_INTERN int CeedRegister_Cuda_Shared(void) { return CeedRegister("/gpu/cuda/shared", CeedInit_Cuda_shared, 25); }
backends/cuda-shared/ceed-cuda-shared-basis.c:#include <cuda.h>
backends/cuda-shared/ceed-cuda-shared-basis.c:#include <cuda_runtime.h>
backends/cuda-shared/ceed-cuda-shared-basis.c:#include "../cuda/ceed-cuda-common.h"
backends/cuda-shared/ceed-cuda-shared-basis.c:#include "../cuda/ceed-cuda-compile.h"
backends/cuda-shared/ceed-cuda-shared-basis.c:#include "ceed-cuda-shared.h"
backends/cuda-shared/ceed-cuda-shared-basis.c:int CeedInit_CudaInterp(CeedScalar *d_B, CeedInt P_1d, CeedInt Q_1d, CeedScalar **c_B);
backends/cuda-shared/ceed-cuda-shared-basis.c:int CeedInit_CudaGrad(CeedScalar *d_B, CeedScalar *d_G, CeedInt P_1d, CeedInt Q_1d, CeedScalar **c_B_ptr, CeedScalar **c_G_ptr);
backends/cuda-shared/ceed-cuda-shared-basis.c:int CeedInit_CudaCollocatedGrad(CeedScalar *d_B, CeedScalar *d_G, CeedInt P_1d, CeedInt Q_1d, CeedScalar **c_B_ptr, CeedScalar **c_G_ptr);
backends/cuda-shared/ceed-cuda-shared-basis.c:static int CeedBasisApplyTensorCore_Cuda_shared(CeedBasis basis, bool apply_add, const CeedInt num_elem, CeedTransposeMode t_mode,
backends/cuda-shared/ceed-cuda-shared-basis.c:  Ceed_Cuda             *ceed_Cuda;
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedBasis_Cuda_shared *data;
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedGetData(ceed, &ceed_Cuda));
backends/cuda-shared/ceed-cuda-shared-basis.c:      CeedCallBackend(CeedInit_CudaInterp(data->d_interp_1d, P_1d, Q_1d, &data->c_B));
backends/cuda-shared/ceed-cuda-shared-basis.c:        CeedInt elems_per_block = CeedIntMin(ceed_Cuda->device_prop.maxThreadsDim[2], CeedIntMax(512 / thread_1d,
backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->InterpTransposeAdd : data->InterpTranspose, grid, thread_1d, 1,
backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Interp, grid, thread_1d, 1, elems_per_block, shared_mem, interp_args));
backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->InterpTransposeAdd : data->InterpTranspose, grid, thread_1d, thread_1d,
backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Interp, grid, thread_1d, thread_1d, elems_per_block, shared_mem, interp_args));
backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->InterpTransposeAdd : data->InterpTranspose, grid, thread_1d, thread_1d,
backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Interp, grid, thread_1d, thread_1d, elems_per_block, shared_mem, interp_args));
backends/cuda-shared/ceed-cuda-shared-basis.c:        CeedCallBackend(CeedInit_CudaCollocatedGrad(data->d_interp_1d, data->d_collo_grad_1d, P_1d, Q_1d, &data->c_B, &data->c_G));
backends/cuda-shared/ceed-cuda-shared-basis.c:        CeedCallBackend(CeedInit_CudaGrad(data->d_interp_1d, data->d_grad_1d, P_1d, Q_1d, &data->c_B, &data->c_G));
backends/cuda-shared/ceed-cuda-shared-basis.c:        CeedInt elems_per_block = CeedIntMin(ceed_Cuda->device_prop.maxThreadsDim[2], CeedIntMax(512 / thread_1d,
backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->GradTransposeAdd : data->GradTranspose, grid, thread_1d, 1,
backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Grad, grid, thread_1d, 1, elems_per_block, shared_mem, grad_args));
backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->GradTransposeAdd : data->GradTranspose, grid, thread_1d, thread_1d,
backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Grad, grid, thread_1d, thread_1d, elems_per_block, shared_mem, grad_args));
backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, apply_add ? data->GradTransposeAdd : data->GradTranspose, grid, thread_1d, thread_1d,
backends/cuda-shared/ceed-cuda-shared-basis.c:          CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->Grad, grid, thread_1d, thread_1d, elems_per_block, shared_mem, grad_args));
backends/cuda-shared/ceed-cuda-shared-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, grid_size, Q_1d, elems_per_block, 1, weight_args));
backends/cuda-shared/ceed-cuda-shared-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, grid_size, Q_1d, Q_1d, elems_per_block, weight_args));
backends/cuda-shared/ceed-cuda-shared-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, grid_size, Q_1d, Q_1d, elems_per_block, weight_args));
backends/cuda-shared/ceed-cuda-shared-basis.c:static int CeedBasisApplyTensor_Cuda_shared(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedBasisApplyTensorCore_Cuda_shared(basis, false, num_elem, t_mode, eval_mode, u, v));
backends/cuda-shared/ceed-cuda-shared-basis.c:static int CeedBasisApplyAddTensor_Cuda_shared(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode,
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedBasisApplyTensorCore_Cuda_shared(basis, true, num_elem, t_mode, eval_mode, u, v));
backends/cuda-shared/ceed-cuda-shared-basis.c:static int CeedBasisApplyAtPointsCore_Cuda_shared(CeedBasis basis, bool apply_add, const CeedInt num_elem, const CeedInt *num_points,
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedBasis_Cuda_shared *data;
backends/cuda-shared/ceed-cuda-shared-basis.c:              "Vector at points must be padded to the same number of points in each element for BasisApplyAtPoints on GPU backends."
backends/cuda-shared/ceed-cuda-shared-basis.c:      if (data->d_points_per_elem) CeedCallCuda(ceed, cudaFree(data->d_points_per_elem));
backends/cuda-shared/ceed-cuda-shared-basis.c:      CeedCallCuda(ceed, cudaMalloc((void **)&data->d_points_per_elem, num_bytes));
backends/cuda-shared/ceed-cuda-shared-basis.c:      CeedCallCuda(ceed, cudaMemcpy(data->d_points_per_elem, num_points, num_bytes, cudaMemcpyHostToDevice));
backends/cuda-shared/ceed-cuda-shared-basis.c:      CeedCallCuda(ceed, cudaMalloc((void **)&data->d_chebyshev_interp_1d, interp_bytes));
backends/cuda-shared/ceed-cuda-shared-basis.c:      CeedCallCuda(ceed, cudaMemcpy(data->d_chebyshev_interp_1d, chebyshev_interp_1d, interp_bytes, cudaMemcpyHostToDevice));
backends/cuda-shared/ceed-cuda-shared-basis.c:    const char basis_kernel_source[] = "// AtPoints basis source\n#include <ceed/jit-source/cuda/cuda-ref-basis-tensor-at-points.h>\n";
backends/cuda-shared/ceed-cuda-shared-basis.c:    if (data->moduleAtPoints) CeedCallCuda(ceed, cuModuleUnload(data->moduleAtPoints));
backends/cuda-shared/ceed-cuda-shared-basis.c:    CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->moduleAtPoints, 9, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "BASIS_BUF_LEN",
backends/cuda-shared/ceed-cuda-shared-basis.c:    CeedCallBackend(CeedGetKernel_Cuda(ceed, data->moduleAtPoints, "InterpAtPoints", &data->InterpAtPoints));
backends/cuda-shared/ceed-cuda-shared-basis.c:    CeedCallBackend(CeedGetKernel_Cuda(ceed, data->moduleAtPoints, "GradAtPoints", &data->GradAtPoints));
backends/cuda-shared/ceed-cuda-shared-basis.c:    CeedCallCuda(ceed, cudaMemset(d_v, 0, length * sizeof(CeedScalar)));
backends/cuda-shared/ceed-cuda-shared-basis.c:      CeedCallBackend(CeedRunKernel_Cuda(ceed, data->InterpAtPoints, num_elem, block_size, interp_args));
backends/cuda-shared/ceed-cuda-shared-basis.c:      CeedCallBackend(CeedRunKernel_Cuda(ceed, data->GradAtPoints, num_elem, block_size, grad_args));
backends/cuda-shared/ceed-cuda-shared-basis.c:static int CeedBasisApplyAtPoints_Cuda_shared(CeedBasis basis, const CeedInt num_elem, const CeedInt *num_points, CeedTransposeMode t_mode,
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedBasisApplyAtPointsCore_Cuda_shared(basis, false, num_elem, num_points, t_mode, eval_mode, x_ref, u, v));
backends/cuda-shared/ceed-cuda-shared-basis.c:static int CeedBasisApplyAddAtPoints_Cuda_shared(CeedBasis basis, const CeedInt num_elem, const CeedInt *num_points, CeedTransposeMode t_mode,
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedBasisApplyAtPointsCore_Cuda_shared(basis, true, num_elem, num_points, t_mode, eval_mode, x_ref, u, v));
backends/cuda-shared/ceed-cuda-shared-basis.c:static int CeedBasisDestroy_Cuda_shared(CeedBasis basis) {
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedBasis_Cuda_shared *data;
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cuModuleUnload(data->module));
backends/cuda-shared/ceed-cuda-shared-basis.c:  if (data->moduleAtPoints) CeedCallCuda(ceed, cuModuleUnload(data->moduleAtPoints));
backends/cuda-shared/ceed-cuda-shared-basis.c:  if (data->d_q_weight_1d) CeedCallCuda(ceed, cudaFree(data->d_q_weight_1d));
backends/cuda-shared/ceed-cuda-shared-basis.c:  if (data->d_points_per_elem) CeedCallCuda(ceed, cudaFree(data->d_points_per_elem));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_interp_1d));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_grad_1d));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_collo_grad_1d));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_chebyshev_interp_1d));
backends/cuda-shared/ceed-cuda-shared-basis.c:int CeedBasisCreateTensorH1_Cuda_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedBasis_Cuda_shared *data;
backends/cuda-shared/ceed-cuda-shared-basis.c:  // Copy basis data to GPU
backends/cuda-shared/ceed-cuda-shared-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight_1d, q_bytes));
backends/cuda-shared/ceed-cuda-shared-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight_1d, q_weight_1d, q_bytes, cudaMemcpyHostToDevice));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp_1d, interp_bytes));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cudaMemcpy(data->d_interp_1d, interp_1d, interp_bytes, cudaMemcpyHostToDevice));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_grad_1d, interp_bytes));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallCuda(ceed, cudaMemcpy(data->d_grad_1d, grad_1d, interp_bytes, cudaMemcpyHostToDevice));
backends/cuda-shared/ceed-cuda-shared-basis.c:  // Compute collocated gradient and copy to GPU
backends/cuda-shared/ceed-cuda-shared-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_collo_grad_1d, q_bytes * Q_1d));
backends/cuda-shared/ceed-cuda-shared-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_collo_grad_1d, collo_grad_1d, q_bytes * Q_1d, cudaMemcpyHostToDevice));
backends/cuda-shared/ceed-cuda-shared-basis.c:  const char basis_kernel_source[] = "// Tensor basis source\n#include <ceed/jit-source/cuda/cuda-shared-basis-tensor.h>\n";
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->module, 8, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "T_1D",
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Interp", &data->Interp));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTransposeAdd", &data->InterpTransposeAdd));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Grad", &data->Grad));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "GradTranspose", &data->GradTranspose));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "GradTransposeAdd", &data->GradTransposeAdd));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyTensor_Cuda_shared));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddTensor_Cuda_shared));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAtPoints", CeedBasisApplyAtPoints_Cuda_shared));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAddAtPoints", CeedBasisApplyAddAtPoints_Cuda_shared));
backends/cuda-shared/ceed-cuda-shared-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Cuda_shared));
backends/cuda-shared/kernels/cuda-shared-basis.cu:#include <cuda.h>
backends/cuda-shared/kernels/cuda-shared-basis.cu:extern "C" int CeedInit_CudaInterp(CeedScalar *d_B, CeedInt P_1d, CeedInt Q_1d, CeedScalar **c_B_ptr) {
backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaMemcpyToSymbol(c_B, d_B, bytes, 0, cudaMemcpyDeviceToDevice);
backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaGetSymbolAddress((void **)c_B_ptr, c_B);
backends/cuda-shared/kernels/cuda-shared-basis.cu:extern "C" int CeedInit_CudaGrad(CeedScalar *d_B, CeedScalar *d_G, CeedInt P_1d, CeedInt Q_1d, CeedScalar **c_B_ptr, CeedScalar **c_G_ptr) {
backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaMemcpyToSymbol(c_B, d_B, bytes, 0, cudaMemcpyDeviceToDevice);
backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaGetSymbolAddress((void **)c_B_ptr, c_B);
backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaMemcpyToSymbol(c_G, d_G, bytes, 0, cudaMemcpyDeviceToDevice);
backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaGetSymbolAddress((void **)c_G_ptr, c_G);
backends/cuda-shared/kernels/cuda-shared-basis.cu:extern "C" int CeedInit_CudaCollocatedGrad(CeedScalar *d_B, CeedScalar *d_G, CeedInt P_1d, CeedInt Q_1d, CeedScalar **c_B_ptr, CeedScalar **c_G_ptr) {
backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaMemcpyToSymbol(c_B, d_B, bytes_interp, 0, cudaMemcpyDeviceToDevice);
backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaGetSymbolAddress((void **)c_B_ptr, c_B);
backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaMemcpyToSymbol(c_G, d_G, bytes_grad, 0, cudaMemcpyDeviceToDevice);
backends/cuda-shared/kernels/cuda-shared-basis.cu:  cudaGetSymbolAddress((void **)c_G_ptr, c_G);
backends/cuda-shared/ceed-cuda-shared.h:#include <cuda.h>
backends/cuda-shared/ceed-cuda-shared.h:} CeedBasis_Cuda_shared;
backends/cuda-shared/ceed-cuda-shared.h:CEED_INTERN int CeedBasisCreateTensorH1_Cuda_shared(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
backends/sycl-gen/ceed-sycl-gen-qfunction.sycl.cpp:  CeedCheck(impl->qfunction_source, ceed, CEED_ERROR_UNSUPPORTED, "/gpu/sycl/gen backend requires QFunction source code file");
backends/sycl-gen/ceed-sycl-gen.sycl.cpp:  const char fallback_resource[] = "/gpu/sycl/ref";
backends/sycl-gen/ceed-sycl-gen.sycl.cpp:  CeedCheck(!strcmp(resource_root, "/gpu/sycl") || !strcmp(resource_root, "/gpu/sycl/gen"), ceed, CEED_ERROR_BACKEND,
backends/sycl-gen/ceed-sycl-gen.sycl.cpp:  CeedCallBackend(CeedInit("/gpu/sycl/shared", &ceed_shared));
backends/sycl-gen/ceed-sycl-gen.sycl.cpp:CEED_INTERN int CeedRegister_Sycl_Gen(void) { return CeedRegister("/gpu/sycl/gen", CeedInit_Sycl_gen, 20); }
backends/hip/ceed-hip-compile.cpp:  // Add hip runtime include statement for generation if runtime < 40400000 (implies ROCm < 4.5)
backends/hip/ceed-hip-compile.cpp:  // With ROCm 4.5, need to include these definitions specifically for hiprtc (but cannot include the runtime header)
backends/hip/ceed-hip-compile.cpp:  std::string arch_arg = "--gpu-architecture=" + std::string(prop.gcnArchName);
backends/sycl-shared/ceed-sycl-shared-basis.sycl.cpp:  // Copy basis data to GPU
backends/sycl-shared/ceed-sycl-shared-basis.sycl.cpp:  // Compute collocated gradient and copy to GPU
backends/sycl-shared/ceed-sycl-shared.sycl.cpp:  CeedCheck(!std::strcmp(resource_root, "/gpu/sycl/shared") || !std::strcmp(resource_root, "/cpu/sycl/shared"), ceed, CEED_ERROR_BACKEND,
backends/sycl-shared/ceed-sycl-shared.sycl.cpp:  CeedCallBackend(CeedRegister("/gpu/sycl/shared", CeedInit_Sycl_shared, 25));
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:#include <cuda_runtime.h>
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:#include "../cuda-ref/ceed-cuda-ref.h"
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:#include "../cuda-shared/ceed-cuda-shared.h"
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:#include "../cuda/ceed-cuda-common.h"
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:#include "../cuda/ceed-cuda-compile.h"
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:#include "ceed-cuda-gen.h"
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:static int CeedOperatorBuildKernelData_Cuda_gen(Ceed ceed, CeedInt num_input_fields, CeedOperatorField *op_input_fields,
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:        CeedBasis_Cuda_shared *basis_data;
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:        CeedBasis_Cuda_shared *basis_data;
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:static int CeedOperatorBuildKernelFieldData_Cuda_gen(std::ostringstream &code, CeedOperator_Cuda_gen *data, CeedInt i, CeedOperatorField op_field,
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  CeedBasis_Cuda_shared *basis_data;
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:static int CeedOperatorBuildKernelRestriction_Cuda_gen(std::ostringstream &code, CeedOperator_Cuda_gen *data, CeedInt i, CeedInt dim,
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  CeedElemRestriction_Cuda *rstr_data;
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:static int CeedOperatorBuildKernelBasis_Cuda_gen(std::ostringstream &code, CeedOperator_Cuda_gen *data, CeedInt i, CeedInt dim,
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:        CeedBasis_Cuda_shared *basis_data;
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:static int CeedOperatorBuildKernelQFunction_Cuda_gen(std::ostringstream &code, CeedOperator_Cuda_gen *data, CeedInt dim, CeedInt num_input_fields,
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:            CeedElemRestriction_Cuda *rstr_data;
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:extern "C" int CeedOperatorBuildKernel_Cuda_gen(CeedOperator op) {
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  CeedQFunction_Cuda_gen *qf_data;
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  CeedOperator_Cuda_gen  *data;
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  CeedCallBackend(CeedOperatorBuildKernelData_Cuda_gen(ceed, num_input_fields, op_input_fields, qf_input_fields, num_output_fields, op_output_fields,
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  // Add atomicAdd function for old NVidia architectures
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    Ceed_Cuda            *ceed_data;
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    struct cudaDeviceProp prop;
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    CeedCallBackend(cudaGetDeviceProperties(&prop, ceed_data->device_id));
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:      code << "#include <ceed/jit-source/cuda/cuda-atomic-add-fallback.h>\n\n";
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  code << "#include <ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h>\n\n";
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  code << "#include <ceed/jit-source/cuda/cuda-gen-templates.h>\n\n";
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  operator_name = "CeedKernelCudaGenOperator_" + qfunction_name;
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    CeedCheck(source_path, ceed, CEED_ERROR_UNSUPPORTED, "/gpu/cuda/gen backend requires QFunction source code file");
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:       << "(CeedInt num_elem, void* ctx, FieldsInt_Cuda indices, Fields_Cuda fields, Fields_Cuda B, Fields_Cuda G, CeedScalar *W) {\n";
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  code << "  SharedData_Cuda data;\n";
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    CeedCallBackend(CeedOperatorBuildKernelFieldData_Cuda_gen(code, data, i, op_input_fields[i], qf_input_fields[i], Q_1d, true, use_3d_slices));
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    CeedCallBackend(CeedOperatorBuildKernelFieldData_Cuda_gen(code, data, i, op_output_fields[i], qf_output_fields[i], Q_1d, false, use_3d_slices));
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    CeedCallBackend(CeedOperatorBuildKernelRestriction_Cuda_gen(code, data, f, dim, field_rstr_in_buffer, op_input_fields[f], qf_input_fields[f],
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    CeedCallBackend(CeedOperatorBuildKernelBasis_Cuda_gen(code, data, f, dim, op_input_fields[f], qf_input_fields[f], Q_1d, true, use_3d_slices));
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  CeedCallBackend(CeedOperatorBuildKernelQFunction_Cuda_gen(code, data, dim, num_input_fields, op_input_fields, qf_input_fields, num_output_fields,
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:    CeedCallBackend(CeedOperatorBuildKernelBasis_Cuda_gen(code, data, i, dim, op_output_fields[i], qf_output_fields[i], Q_1d, false, use_3d_slices));
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:        CeedOperatorBuildKernelRestriction_Cuda_gen(code, data, i, dim, NULL, op_output_fields[i], qf_output_fields[i], Q_1d, false, use_3d_slices));
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  CeedCallBackend(CeedCompile_Cuda(ceed, code.str().c_str(), &data->module, 1, "T_1D", CeedIntMax(Q_1d, data->max_P_1d)));
backends/cuda-gen/ceed-cuda-gen-operator-build.cpp:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, operator_name.c_str(), &data->op));
backends/cuda-gen/ceed-cuda-gen-operator-build.h:CEED_INTERN int CeedOperatorBuildKernel_Cuda_gen(CeedOperator op);
backends/cuda-gen/ceed-cuda-gen-operator.c:#include <ceed/jit-source/cuda/cuda-types.h>
backends/cuda-gen/ceed-cuda-gen-operator.c:#include "../cuda/ceed-cuda-common.h"
backends/cuda-gen/ceed-cuda-gen-operator.c:#include "../cuda/ceed-cuda-compile.h"
backends/cuda-gen/ceed-cuda-gen-operator.c:#include "ceed-cuda-gen-operator-build.h"
backends/cuda-gen/ceed-cuda-gen-operator.c:#include "ceed-cuda-gen.h"
backends/cuda-gen/ceed-cuda-gen-operator.c:static int CeedOperatorDestroy_Cuda_gen(CeedOperator op) {
backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedOperator_Cuda_gen *impl;
backends/cuda-gen/ceed-cuda-gen-operator.c:// Problem setting: we'd like to make occupancy high with relatively few inactive threads. CUDA (cuOccupancyMaxPotentialBlockSize) can tell us how
backends/cuda-gen/ceed-cuda-gen-operator.c:// block from running. The cuda-gen kernels are pretty heavy with lots of instruction-level parallelism (ILP) so we'll generally be okay with
backends/cuda-gen/ceed-cuda-gen-operator.c:// cuda-gen can't choose block sizes arbitrarily; they need to be a multiple of the number of quadrature points (or number of basis functions).
backends/cuda-gen/ceed-cuda-gen-operator.c:// CUDA schedules in units of full warps (32 threads), so 128 CUDA hardware threads are effectively committed to that block.
backends/cuda-gen/ceed-cuda-gen-operator.c:static int CeedOperatorApplyAdd_Cuda_gen(CeedOperator op, CeedVector input_vec, CeedVector output_vec, CeedRequest *request) {
backends/cuda-gen/ceed-cuda-gen-operator.c:  Ceed_Cuda              *cuda_data;
backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedQFunction_Cuda_gen *qf_data;
backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedOperator_Cuda_gen  *data;
backends/cuda-gen/ceed-cuda-gen-operator.c:      CeedDebug256(CeedOperatorReturnCeed(op), CEED_DEBUG_COLOR_SUCCESS, "Falling back to /gpu/cuda/ref CeedOperator due to non-tensor bases");
backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedCallBackend(CeedGetData(ceed, &cuda_data));
backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedCallBackend(CeedOperatorBuildKernel_Cuda_gen(op));
backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedCallCuda(ceed, cuOccupancyMaxPotentialBlockSize(&min_grid_size, &max_threads_per_block, data->op, dynamicSMemSize, 0, 0x10000));
backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedCallBackend(BlockGridCalculate(num_elem, min_grid_size / cuda_data->device_prop.multiProcessorCount, max_threads_per_block,
backends/cuda-gen/ceed-cuda-gen-operator.c:                                     cuda_data->device_prop.maxThreadsDim[2], cuda_data->device_prop.warpSize, block, &grid));
backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, data->op, grid, block[0], block[1], block[2], shared_mem, opargs));
backends/cuda-gen/ceed-cuda-gen-operator.c:int CeedOperatorCreate_Cuda_gen(CeedOperator op) {
backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedOperator_Cuda_gen *impl;
backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Cuda_gen));
backends/cuda-gen/ceed-cuda-gen-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Cuda_gen));
backends/cuda-gen/ceed-cuda-gen.h:#include <ceed/jit-source/cuda/cuda-types.h>
backends/cuda-gen/ceed-cuda-gen.h:#include <cuda.h>
backends/cuda-gen/ceed-cuda-gen.h:  FieldsInt_Cuda indices;
backends/cuda-gen/ceed-cuda-gen.h:  Fields_Cuda    fields;
backends/cuda-gen/ceed-cuda-gen.h:  Fields_Cuda    B;
backends/cuda-gen/ceed-cuda-gen.h:  Fields_Cuda    G;
backends/cuda-gen/ceed-cuda-gen.h:} CeedOperator_Cuda_gen;
backends/cuda-gen/ceed-cuda-gen.h:} CeedQFunction_Cuda_gen;
backends/cuda-gen/ceed-cuda-gen.h:CEED_INTERN int CeedQFunctionCreate_Cuda_gen(CeedQFunction qf);
backends/cuda-gen/ceed-cuda-gen.h:CEED_INTERN int CeedOperatorCreate_Cuda_gen(CeedOperator op);
backends/cuda-gen/ceed-cuda-gen.c:#include "ceed-cuda-gen.h"
backends/cuda-gen/ceed-cuda-gen.c:#include "../cuda/ceed-cuda-common.h"
backends/cuda-gen/ceed-cuda-gen.c:static int CeedInit_Cuda_gen(const char *resource, Ceed ceed) {
backends/cuda-gen/ceed-cuda-gen.c:  const char fallback_resource[] = "/gpu/cuda/ref";
backends/cuda-gen/ceed-cuda-gen.c:  Ceed_Cuda *data;
backends/cuda-gen/ceed-cuda-gen.c:  CeedCheck(!strcmp(resource_root, "/gpu/cuda") || !strcmp(resource_root, "/gpu/cuda/gen"), ceed, CEED_ERROR_BACKEND,
backends/cuda-gen/ceed-cuda-gen.c:            "Cuda backend cannot use resource: %s", resource);
backends/cuda-gen/ceed-cuda-gen.c:  CeedCallBackend(CeedInit_Cuda(ceed, resource));
backends/cuda-gen/ceed-cuda-gen.c:  CeedCallBackend(CeedInit("/gpu/cuda/shared", &ceed_shared));
backends/cuda-gen/ceed-cuda-gen.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate", CeedQFunctionCreate_Cuda_gen));
backends/cuda-gen/ceed-cuda-gen.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate", CeedOperatorCreate_Cuda_gen));
backends/cuda-gen/ceed-cuda-gen.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Cuda));
backends/cuda-gen/ceed-cuda-gen.c:CEED_INTERN int CeedRegister_Cuda_Gen(void) { return CeedRegister("/gpu/cuda/gen", CeedInit_Cuda_gen, 20); }
backends/cuda-gen/ceed-cuda-gen-qfunction.c:#include <cuda_runtime.h>
backends/cuda-gen/ceed-cuda-gen-qfunction.c:#include "../cuda/ceed-cuda-common.h"
backends/cuda-gen/ceed-cuda-gen-qfunction.c:#include "ceed-cuda-gen.h"
backends/cuda-gen/ceed-cuda-gen-qfunction.c:static int CeedQFunctionApply_Cuda_gen(CeedQFunction qf, CeedInt Q, CeedVector *U, CeedVector *V) {
backends/cuda-gen/ceed-cuda-gen-qfunction.c:static int CeedQFunctionDestroy_Cuda_gen(CeedQFunction qf) {
backends/cuda-gen/ceed-cuda-gen-qfunction.c:  CeedQFunction_Cuda_gen *data;
backends/cuda-gen/ceed-cuda-gen-qfunction.c:  CeedCallCuda(CeedQFunctionReturnCeed(qf), cudaFree(data->d_c));
backends/cuda-gen/ceed-cuda-gen-qfunction.c:int CeedQFunctionCreate_Cuda_gen(CeedQFunction qf) {
backends/cuda-gen/ceed-cuda-gen-qfunction.c:  CeedQFunction_Cuda_gen *data;
backends/cuda-gen/ceed-cuda-gen-qfunction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Apply", CeedQFunctionApply_Cuda_gen));
backends/cuda-gen/ceed-cuda-gen-qfunction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy", CeedQFunctionDestroy_Cuda_gen));
backends/sycl-ref/ceed-sycl-ref-qfunction.sycl.cpp:    // Equivalent of CUDA Occupancy Calculator
backends/sycl-ref/ceed-sycl-ref-qfunction-load.sycl.cpp:  // OpenCL doesn't allow for structs with pointers.
backends/sycl-ref/ceed-sycl-ref.sycl.cpp:  CeedCheck(!std::strcmp(resource_root, "/gpu/sycl/ref") || !std::strcmp(resource_root, "/cpu/sycl/ref"), ceed, CEED_ERROR_BACKEND,
backends/sycl-ref/ceed-sycl-ref.sycl.cpp:  CeedCallBackend(CeedRegister("/gpu/sycl/ref", CeedInit_Sycl_ref, 40));
backends/hip-ref/ceed-hip-ref-basis.c:              "Vector at points must be padded to the same number of points in each element for BasisApplyAtPoints on GPU backends."
backends/hip-ref/ceed-hip-ref-basis.c:  // Copy data to GPU
backends/hip-ref/ceed-hip-ref-basis.c:  // Copy basis data to GPU
backends/hip-ref/ceed-hip-ref-basis.c:  // Copy basis data to GPU
backends/hip-ref/ceed-hip-ref-basis.c:  // Copy basis data to GPU
backends/hip-ref/ceed-hip-ref.c:  CeedCheck(!strcmp(resource_root, "/gpu/hip/ref"), ceed, CEED_ERROR_BACKEND, "Hip backend cannot use resource: %s", resource);
backends/hip-ref/ceed-hip-ref.c:CEED_INTERN int CeedRegister_Hip(void) { return CeedRegister("/gpu/hip/ref", CeedInit_Hip_ref, 40); }
backends/memcheck/ceed-memcheck-restriction.c:      // GPU default, contiguous by node, then element
backends/hip-shared/ceed-hip-shared-basis.c:              "Vector at points must be padded to the same number of points in each element for BasisApplyAtPoints on GPU backends."
backends/hip-shared/ceed-hip-shared-basis.c:  // Copy basis data to GPU
backends/hip-shared/ceed-hip-shared-basis.c:  // Compute collocated gradient and copy to GPU
backends/hip-shared/ceed-hip-shared.c:  CeedCheck(!strcmp(resource_root, "/gpu/hip/shared"), ceed, CEED_ERROR_BACKEND, "Hip backend cannot use resource: %s", resource);
backends/hip-shared/ceed-hip-shared.c:  CeedCallBackend(CeedInit("/gpu/hip/ref", &ceed_ref));
backends/hip-shared/ceed-hip-shared.c:CEED_INTERN int CeedRegister_Hip_Shared(void) { return CeedRegister("/gpu/hip/shared", CeedInit_Hip_shared, 25); }
backends/cuda/ceed-cuda-compile.cpp:#include "ceed-cuda-compile.h"
backends/cuda/ceed-cuda-compile.cpp:#include <cuda_runtime.h>
backends/cuda/ceed-cuda-compile.cpp:#include "ceed-cuda-common.h"
backends/cuda/ceed-cuda-compile.cpp:// Compile CUDA kernel
backends/cuda/ceed-cuda-compile.cpp:int CeedCompile_Cuda(Ceed ceed, const char *source, CUmodule *module, const CeedInt num_defines, ...) {
backends/cuda/ceed-cuda-compile.cpp:  struct cudaDeviceProp prop;
backends/cuda/ceed-cuda-compile.cpp:  Ceed_Cuda            *ceed_data;
backends/cuda/ceed-cuda-compile.cpp:  cudaFree(0);  // Make sure a Context exists for nvrtc
backends/cuda/ceed-cuda-compile.cpp:  // Standard libCEED definitions for CUDA backends
backends/cuda/ceed-cuda-compile.cpp:  code << "#include <ceed/jit-source/cuda/cuda-jit.h>\n\n";
backends/cuda/ceed-cuda-compile.cpp:  CeedCallCuda(ceed, cudaGetDeviceProperties(&prop, ceed_data->device_id));
backends/cuda/ceed-cuda-compile.cpp:#if CUDA_VERSION >= 11010
backends/cuda/ceed-cuda-compile.cpp:      // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#dynamic-code-generation
backends/cuda/ceed-cuda-compile.cpp:#if CUDA_VERSION >= 11010
backends/cuda/ceed-cuda-compile.cpp:  CeedCallCuda(ceed, cuModuleLoadData(module, ptx));
backends/cuda/ceed-cuda-compile.cpp:// Get CUDA kernel
backends/cuda/ceed-cuda-compile.cpp:int CeedGetKernel_Cuda(Ceed ceed, CUmodule module, const char *name, CUfunction *kernel) {
backends/cuda/ceed-cuda-compile.cpp:  CeedCallCuda(ceed, cuModuleGetFunction(kernel, module, name));
backends/cuda/ceed-cuda-compile.cpp:// Run CUDA kernel with block size selected automatically based on the kernel
backends/cuda/ceed-cuda-compile.cpp:int CeedRunKernelAutoblockCuda(Ceed ceed, CUfunction kernel, size_t points, void **args) {
backends/cuda/ceed-cuda-compile.cpp:  CeedCallCuda(ceed, cuOccupancyMaxPotentialBlockSize(&min_grid_size, &max_block_size, kernel, NULL, 0, 0x10000));
backends/cuda/ceed-cuda-compile.cpp:  CeedCallBackend(CeedRunKernel_Cuda(ceed, kernel, CeedDivUpInt(points, max_block_size), max_block_size, args));
backends/cuda/ceed-cuda-compile.cpp:// Run CUDA kernel
backends/cuda/ceed-cuda-compile.cpp:int CeedRunKernel_Cuda(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size, void **args) {
backends/cuda/ceed-cuda-compile.cpp:  CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, kernel, grid_size, block_size, 1, 1, 0, args));
backends/cuda/ceed-cuda-compile.cpp:// Run CUDA kernel for spatial dimension
backends/cuda/ceed-cuda-compile.cpp:int CeedRunKernelDim_Cuda(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size_x, const int block_size_y, const int block_size_z,
backends/cuda/ceed-cuda-compile.cpp:  CeedCallBackend(CeedRunKernelDimShared_Cuda(ceed, kernel, grid_size, block_size_x, block_size_y, block_size_z, 0, args));
backends/cuda/ceed-cuda-compile.cpp:// Run CUDA kernel for spatial dimension with shared memory
backends/cuda/ceed-cuda-compile.cpp:int CeedRunKernelDimShared_Cuda(Ceed ceed, CUfunction kernel, const int grid_size, const int block_size_x, const int block_size_y,
backends/cuda/ceed-cuda-compile.cpp:#if CUDA_VERSION >= 9000
backends/cuda/ceed-cuda-compile.cpp:  if (result == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES) {
backends/cuda/ceed-cuda-compile.cpp:                     "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: max_threads_per_block %d on block size (%d,%d,%d), shared_size %d, num_regs %d",
backends/cuda/ceed-cuda-compile.h:#include <cuda.h>
backends/cuda/ceed-cuda-compile.h:CEED_INTERN int CeedCompile_Cuda(Ceed ceed, const char *source, CUmodule *module, const CeedInt num_defines, ...);
backends/cuda/ceed-cuda-compile.h:CEED_INTERN int CeedGetKernel_Cuda(Ceed ceed, CUmodule module, const char *name, CUfunction *kernel);
backends/cuda/ceed-cuda-compile.h:CEED_INTERN int CeedRunKernel_Cuda(Ceed ceed, CUfunction kernel, int grid_size, int block_size, void **args);
backends/cuda/ceed-cuda-compile.h:CEED_INTERN int CeedRunKernelAutoblockCuda(Ceed ceed, CUfunction kernel, size_t points, void **args);
backends/cuda/ceed-cuda-compile.h:CEED_INTERN int CeedRunKernelDim_Cuda(Ceed ceed, CUfunction kernel, int grid_size, int block_size_x, int block_size_y, int block_size_z, void **args);
backends/cuda/ceed-cuda-compile.h:CEED_INTERN int CeedRunKernelDimShared_Cuda(Ceed ceed, CUfunction kernel, int grid_size, int block_size_x, int block_size_y, int block_size_z,
backends/cuda/ceed-cuda-common.h:#include <cuda.h>
backends/cuda/ceed-cuda-common.h:    CUresult cuda_result = (CUresult)x;                  \
backends/cuda/ceed-cuda-common.h:    if (cuda_result != CUDA_SUCCESS) {                   \
backends/cuda/ceed-cuda-common.h:      cuGetErrorName(cuda_result, &msg);                 \
backends/cuda/ceed-cuda-common.h:#define CeedCallCuda(ceed, ...) \
backends/cuda/ceed-cuda-common.h:  struct cudaDeviceProp device_prop;
backends/cuda/ceed-cuda-common.h:} Ceed_Cuda;
backends/cuda/ceed-cuda-common.h:CEED_INTERN int CeedInit_Cuda(Ceed ceed, const char *resource);
backends/cuda/ceed-cuda-common.h:CEED_INTERN int CeedDestroy_Cuda(Ceed ceed);
backends/cuda/ceed-cuda-common.h:CEED_INTERN int CeedSetDeviceBoolArray_Cuda(Ceed ceed, const bool *source_array, CeedCopyMode copy_mode, CeedSize num_values,
backends/cuda/ceed-cuda-common.h:CEED_INTERN int CeedSetDeviceCeedInt8Array_Cuda(Ceed ceed, const CeedInt8 *source_array, CeedCopyMode copy_mode, CeedSize num_values,
backends/cuda/ceed-cuda-common.h:CEED_INTERN int CeedSetDeviceCeedIntArray_Cuda(Ceed ceed, const CeedInt *source_array, CeedCopyMode copy_mode, CeedSize num_values,
backends/cuda/ceed-cuda-common.h:CEED_INTERN int CeedSetDeviceCeedScalarArray_Cuda(Ceed ceed, const CeedScalar *source_array, CeedCopyMode copy_mode, CeedSize num_values,
backends/cuda/ceed-cuda-common.c:#include "ceed-cuda-common.h"
backends/cuda/ceed-cuda-common.c:#include <cuda_runtime.h>
backends/cuda/ceed-cuda-common.c:int CeedInit_Cuda(Ceed ceed, const char *resource) {
backends/cuda/ceed-cuda-common.c:  Ceed_Cuda  *data;
backends/cuda/ceed-cuda-common.c:  CeedCallCuda(ceed, cudaGetDevice(&current_device_id));
backends/cuda/ceed-cuda-common.c:    CeedCallCuda(ceed, cudaSetDevice(device_id));
backends/cuda/ceed-cuda-common.c:  CeedCallCuda(ceed, cudaGetDeviceProperties(&data->device_prop, current_device_id));
backends/cuda/ceed-cuda-common.c:int CeedDestroy_Cuda(Ceed ceed) {
backends/cuda/ceed-cuda-common.c:  Ceed_Cuda *data;
backends/cuda/ceed-cuda-common.c:static inline int CeedSetDeviceGenericArray_Cuda(Ceed ceed, const void *source_array, CeedCopyMode copy_mode, size_t size_unit, CeedSize num_values,
backends/cuda/ceed-cuda-common.c:      if (!*(void **)target_array_owned) CeedCallCuda(ceed, cudaMalloc(target_array_owned, size_unit * num_values));
backends/cuda/ceed-cuda-common.c:      if (source_array) CeedCallCuda(ceed, cudaMemcpy(*(void **)target_array_owned, source_array, size_unit * num_values, cudaMemcpyDeviceToDevice));
backends/cuda/ceed-cuda-common.c:      CeedCallCuda(ceed, cudaFree(*(void **)target_array_owned));
backends/cuda/ceed-cuda-common.c:      CeedCallCuda(ceed, cudaFree(*(void **)target_array_owned));
backends/cuda/ceed-cuda-common.c:int CeedSetDeviceBoolArray_Cuda(Ceed ceed, const bool *source_array, CeedCopyMode copy_mode, CeedSize num_values, const bool **target_array_owned,
backends/cuda/ceed-cuda-common.c:  CeedCallBackend(CeedSetDeviceGenericArray_Cuda(ceed, source_array, copy_mode, sizeof(bool), num_values, target_array_owned, target_array_borrowed,
backends/cuda/ceed-cuda-common.c:int CeedSetDeviceCeedInt8Array_Cuda(Ceed ceed, const CeedInt8 *source_array, CeedCopyMode copy_mode, CeedSize num_values,
backends/cuda/ceed-cuda-common.c:  CeedCallBackend(CeedSetDeviceGenericArray_Cuda(ceed, source_array, copy_mode, sizeof(CeedInt8), num_values, target_array_owned,
backends/cuda/ceed-cuda-common.c:int CeedSetDeviceCeedIntArray_Cuda(Ceed ceed, const CeedInt *source_array, CeedCopyMode copy_mode, CeedSize num_values,
backends/cuda/ceed-cuda-common.c:  CeedCallBackend(CeedSetDeviceGenericArray_Cuda(ceed, source_array, copy_mode, sizeof(CeedInt), num_values, target_array_owned,
backends/cuda/ceed-cuda-common.c:int CeedSetDeviceCeedScalarArray_Cuda(Ceed ceed, const CeedScalar *source_array, CeedCopyMode copy_mode, CeedSize num_values,
backends/cuda/ceed-cuda-common.c:  CeedCallBackend(CeedSetDeviceGenericArray_Cuda(ceed, source_array, copy_mode, sizeof(CeedScalar), num_values, target_array_owned,
backends/cuda-ref/ceed-cuda-ref-basis.c:#include <cuda.h>
backends/cuda-ref/ceed-cuda-ref-basis.c:#include <cuda_runtime.h>
backends/cuda-ref/ceed-cuda-ref-basis.c:#include "../cuda/ceed-cuda-common.h"
backends/cuda-ref/ceed-cuda-ref-basis.c:#include "../cuda/ceed-cuda-compile.h"
backends/cuda-ref/ceed-cuda-ref-basis.c:#include "ceed-cuda-ref.h"
backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApplyCore_Cuda(CeedBasis basis, bool apply_add, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode,
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasis_Cuda   *data;
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemset(d_v, 0, length * sizeof(CeedScalar)));
backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallBackend(CeedRunKernel_Cuda(ceed, data->Interp, num_elem, block_size, interp_args));
backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallBackend(CeedRunKernel_Cuda(ceed, data->Grad, num_elem, block_size, grad_args));
backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, num_elem, block_size_x, block_size_y, 1, weight_args));
backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApply_Cuda(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedBasisApplyCore_Cuda(basis, false, num_elem, t_mode, eval_mode, u, v));
backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApplyAdd_Cuda(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedBasisApplyCore_Cuda(basis, true, num_elem, t_mode, eval_mode, u, v));
backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApplyAtPointsCore_Cuda(CeedBasis basis, bool apply_add, const CeedInt num_elem, const CeedInt *num_points,
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasis_Cuda   *data;
backends/cuda-ref/ceed-cuda-ref-basis.c:              "Vector at points must be padded to the same number of points in each element for BasisApplyAtPoints on GPU backends."
backends/cuda-ref/ceed-cuda-ref-basis.c:      if (data->d_points_per_elem) CeedCallCuda(ceed, cudaFree(data->d_points_per_elem));
backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallCuda(ceed, cudaMalloc((void **)&data->d_points_per_elem, num_bytes));
backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallCuda(ceed, cudaMemcpy(data->d_points_per_elem, num_points, num_bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallCuda(ceed, cudaMalloc((void **)&data->d_chebyshev_interp_1d, interp_bytes));
backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallCuda(ceed, cudaMemcpy(data->d_chebyshev_interp_1d, chebyshev_interp_1d, interp_bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-basis.c:    const char basis_kernel_source[] = "// AtPoints basis source\n#include <ceed/jit-source/cuda/cuda-ref-basis-tensor-at-points.h>\n";
backends/cuda-ref/ceed-cuda-ref-basis.c:    if (data->moduleAtPoints) CeedCallCuda(ceed, cuModuleUnload(data->moduleAtPoints));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->moduleAtPoints, 9, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "BASIS_BUF_LEN",
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallBackend(CeedGetKernel_Cuda(ceed, data->moduleAtPoints, "InterpAtPoints", &data->InterpAtPoints));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallBackend(CeedGetKernel_Cuda(ceed, data->moduleAtPoints, "GradAtPoints", &data->GradAtPoints));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemset(d_v, 0, length * sizeof(CeedScalar)));
backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallBackend(CeedRunKernel_Cuda(ceed, data->InterpAtPoints, num_elem, block_size, interp_args));
backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallBackend(CeedRunKernel_Cuda(ceed, data->GradAtPoints, num_elem, block_size, grad_args));
backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApplyAtPoints_Cuda(CeedBasis basis, const CeedInt num_elem, const CeedInt *num_points, CeedTransposeMode t_mode,
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedBasisApplyAtPointsCore_Cuda(basis, false, num_elem, num_points, t_mode, eval_mode, x_ref, u, v));
backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApplyAddAtPoints_Cuda(CeedBasis basis, const CeedInt num_elem, const CeedInt *num_points, CeedTransposeMode t_mode,
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedBasisApplyAtPointsCore_Cuda(basis, true, num_elem, num_points, t_mode, eval_mode, x_ref, u, v));
backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApplyNonTensorCore_Cuda(CeedBasis basis, bool apply_add, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode,
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasisNonTensor_Cuda *data;
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemset(d_v, 0, length * sizeof(CeedScalar)));
backends/cuda-ref/ceed-cuda-ref-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->InterpTranspose, grid, block_size_x, 1, elems_per_block, interp_args));
backends/cuda-ref/ceed-cuda-ref-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Interp, grid, block_size_x, 1, elems_per_block, interp_args));
backends/cuda-ref/ceed-cuda-ref-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->DerivTranspose, grid, block_size_x, 1, elems_per_block, grad_args));
backends/cuda-ref/ceed-cuda-ref-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Deriv, grid, block_size_x, 1, elems_per_block, grad_args));
backends/cuda-ref/ceed-cuda-ref-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->DerivTranspose, grid, block_size_x, 1, elems_per_block, div_args));
backends/cuda-ref/ceed-cuda-ref-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Deriv, grid, block_size_x, 1, elems_per_block, div_args));
backends/cuda-ref/ceed-cuda-ref-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->DerivTranspose, grid, block_size_x, 1, elems_per_block, curl_args));
backends/cuda-ref/ceed-cuda-ref-basis.c:        CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Deriv, grid, block_size_x, 1, elems_per_block, curl_args));
backends/cuda-ref/ceed-cuda-ref-basis.c:      CeedCallBackend(CeedRunKernelDim_Cuda(ceed, data->Weight, grid, num_qpts, 1, elems_per_block, weight_args));
backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApplyNonTensor_Cuda(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedBasisApplyNonTensorCore_Cuda(basis, false, num_elem, t_mode, eval_mode, u, v));
backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisApplyAddNonTensor_Cuda(CeedBasis basis, const CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u,
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedBasisApplyNonTensorCore_Cuda(basis, true, num_elem, t_mode, eval_mode, u, v));
backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisDestroy_Cuda(CeedBasis basis) {
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasis_Cuda *data;
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cuModuleUnload(data->module));
backends/cuda-ref/ceed-cuda-ref-basis.c:  if (data->moduleAtPoints) CeedCallCuda(ceed, cuModuleUnload(data->moduleAtPoints));
backends/cuda-ref/ceed-cuda-ref-basis.c:  if (data->d_q_weight_1d) CeedCallCuda(ceed, cudaFree(data->d_q_weight_1d));
backends/cuda-ref/ceed-cuda-ref-basis.c:  if (data->d_points_per_elem) CeedCallCuda(ceed, cudaFree(data->d_points_per_elem));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_interp_1d));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_grad_1d));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_chebyshev_interp_1d));
backends/cuda-ref/ceed-cuda-ref-basis.c:static int CeedBasisDestroyNonTensor_Cuda(CeedBasis basis) {
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasisNonTensor_Cuda *data;
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cuModuleUnload(data->module));
backends/cuda-ref/ceed-cuda-ref-basis.c:  if (data->d_q_weight) CeedCallCuda(ceed, cudaFree(data->d_q_weight));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_interp));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_grad));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_div));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaFree(data->d_curl));
backends/cuda-ref/ceed-cuda-ref-basis.c:int CeedBasisCreateTensorH1_Cuda(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasis_Cuda *data;
backends/cuda-ref/ceed-cuda-ref-basis.c:  // Copy data to GPU
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight_1d, q_bytes));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight_1d, q_weight_1d, q_bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp_1d, interp_bytes));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaMemcpy(data->d_interp_1d, interp_1d, interp_bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaMalloc((void **)&data->d_grad_1d, interp_bytes));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallCuda(ceed, cudaMemcpy(data->d_grad_1d, grad_1d, interp_bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-basis.c:  const char basis_kernel_source[] = "// Tensor basis source\n#include <ceed/jit-source/cuda/cuda-ref-basis-tensor.h>\n";
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->module, 7, "BASIS_Q_1D", Q_1d, "BASIS_P_1D", P_1d, "BASIS_BUF_LEN",
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Interp", &data->Interp));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Grad", &data->Grad));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApply_Cuda));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAdd_Cuda));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAtPoints", CeedBasisApplyAtPoints_Cuda));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAddAtPoints", CeedBasisApplyAddAtPoints_Cuda));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroy_Cuda));
backends/cuda-ref/ceed-cuda-ref-basis.c:int CeedBasisCreateH1_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *grad,
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasisNonTensor_Cuda *data;
backends/cuda-ref/ceed-cuda-ref-basis.c:  // Copy basis data to GPU
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight, q_bytes));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight, q_weight, q_bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp, interp_bytes));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_interp, interp, interp_bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_grad, grad_bytes));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_grad, grad, grad_bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-basis.c:  const char basis_kernel_source[] = "// Nontensor basis source\n#include <ceed/jit-source/cuda/cuda-ref-basis-nontensor.h>\n";
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->module, 5, "BASIS_Q", num_qpts, "BASIS_P", num_nodes, "BASIS_Q_COMP_INTERP",
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Interp", &data->Interp));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Deriv", &data->Deriv));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "DerivTranspose", &data->DerivTranspose));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Cuda));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Cuda));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Cuda));
backends/cuda-ref/ceed-cuda-ref-basis.c:int CeedBasisCreateHdiv_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp, const CeedScalar *div,
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasisNonTensor_Cuda *data;
backends/cuda-ref/ceed-cuda-ref-basis.c:  // Copy basis data to GPU
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight, q_bytes));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight, q_weight, q_bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp, interp_bytes));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_interp, interp, interp_bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_div, div_bytes));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_div, div, div_bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-basis.c:  const char basis_kernel_source[] = "// Nontensor basis source\n#include <ceed/jit-source/cuda/cuda-ref-basis-nontensor.h>\n";
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->module, 5, "BASIS_Q", num_qpts, "BASIS_P", num_nodes, "BASIS_Q_COMP_INTERP",
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Interp", &data->Interp));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Deriv", &data->Deriv));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "DerivTranspose", &data->DerivTranspose));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Cuda));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Cuda));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Cuda));
backends/cuda-ref/ceed-cuda-ref-basis.c:int CeedBasisCreateHcurl_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedBasisNonTensor_Cuda *data;
backends/cuda-ref/ceed-cuda-ref-basis.c:  // Copy basis data to GPU
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_q_weight, q_bytes));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_q_weight, q_weight, q_bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_interp, interp_bytes));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_interp, interp, interp_bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMalloc((void **)&data->d_curl, curl_bytes));
backends/cuda-ref/ceed-cuda-ref-basis.c:    CeedCallCuda(ceed, cudaMemcpy(data->d_curl, curl, curl_bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-basis.c:  const char basis_kernel_source[] = "// Nontensor basis source\n#include <ceed/jit-source/cuda/cuda-ref-basis-nontensor.h>\n";
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedCompile_Cuda(ceed, basis_kernel_source, &data->module, 5, "BASIS_Q", num_qpts, "BASIS_P", num_nodes, "BASIS_Q_COMP_INTERP",
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Interp", &data->Interp));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "InterpTranspose", &data->InterpTranspose));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Deriv", &data->Deriv));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "DerivTranspose", &data->DerivTranspose));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, "Weight", &data->Weight));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Apply", CeedBasisApplyNonTensor_Cuda));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "ApplyAdd", CeedBasisApplyAddNonTensor_Cuda));
backends/cuda-ref/ceed-cuda-ref-basis.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Basis", basis, "Destroy", CeedBasisDestroyNonTensor_Cuda));
backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:#include "../cuda/ceed-cuda-compile.h"
backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:#include "ceed-cuda-ref.h"
backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:extern "C" int CeedQFunctionBuildKernel_Cuda_ref(CeedQFunction qf) {
backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:  CeedQFunction_Cuda *data;
backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:  string        kernel_name = "CeedKernelCudaRefQFunction_" + qfunction_name;
backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:  code << "#include <ceed/jit-source/cuda/cuda-ref-qfunction.h>\n\n";
backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:  code << "extern \"C\" __global__ void " << kernel_name << "(void *ctx, CeedInt Q, Fields_Cuda fields) {\n";
backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:  CeedCallBackend(CeedCompile_Cuda(ceed, code.str().c_str(), &data->module, 0));
backends/cuda-ref/ceed-cuda-ref-qfunction-load.cpp:  CeedCallBackend(CeedGetKernel_Cuda(ceed, data->module, kernel_name.c_str(), &data->QFunction));
backends/cuda-ref/ceed-cuda-ref.c:#include "ceed-cuda-ref.h"
backends/cuda-ref/ceed-cuda-ref.c:#include "../cuda/ceed-cuda-common.h"
backends/cuda-ref/ceed-cuda-ref.c:// CUDA preferred MemType
backends/cuda-ref/ceed-cuda-ref.c:static int CeedGetPreferredMemType_Cuda(CeedMemType *mem_type) {
backends/cuda-ref/ceed-cuda-ref.c:int CeedGetCublasHandle_Cuda(Ceed ceed, cublasHandle_t *handle) {
backends/cuda-ref/ceed-cuda-ref.c:  Ceed_Cuda *data;
backends/cuda-ref/ceed-cuda-ref.c:static int CeedInit_Cuda_ref(const char *resource, Ceed ceed) {
backends/cuda-ref/ceed-cuda-ref.c:  Ceed_Cuda *data;
backends/cuda-ref/ceed-cuda-ref.c:  CeedCheck(!strcmp(resource_root, "/gpu/cuda/ref"), ceed, CEED_ERROR_BACKEND, "Cuda backend cannot use resource: %s", resource);
backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedInit_Cuda(ceed, resource));
backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "GetPreferredMemType", CeedGetPreferredMemType_Cuda));
backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "VectorCreate", CeedVectorCreate_Cuda));
backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1", CeedBasisCreateTensorH1_Cuda));
backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1", CeedBasisCreateH1_Cuda));
backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateHdiv", CeedBasisCreateHdiv_Cuda));
backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateHcurl", CeedBasisCreateHcurl_Cuda));
backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate", CeedElemRestrictionCreate_Cuda));
backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreateAtPoints", CeedElemRestrictionCreate_Cuda));
backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate", CeedQFunctionCreate_Cuda));
backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionContextCreate", CeedQFunctionContextCreate_Cuda));
backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate", CeedOperatorCreate_Cuda));
backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreateAtPoints", CeedOperatorCreateAtPoints_Cuda));
backends/cuda-ref/ceed-cuda-ref.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy", CeedDestroy_Cuda));
backends/cuda-ref/ceed-cuda-ref.c:CEED_INTERN int CeedRegister_Cuda(void) { return CeedRegister("/gpu/cuda/ref", CeedInit_Cuda_ref, 40); }
backends/cuda-ref/ceed-cuda-ref-qfunction-load.h:CEED_INTERN int CeedQFunctionBuildKernel_Cuda_ref(CeedQFunction qf);
backends/cuda-ref/ceed-cuda-ref.h:#include <ceed/jit-source/cuda/cuda-types.h>
backends/cuda-ref/ceed-cuda-ref.h:#include <cuda.h>
backends/cuda-ref/ceed-cuda-ref.h:} CeedVector_Cuda;
backends/cuda-ref/ceed-cuda-ref.h:} CeedElemRestriction_Cuda;
backends/cuda-ref/ceed-cuda-ref.h:} CeedBasis_Cuda;
backends/cuda-ref/ceed-cuda-ref.h:} CeedBasisNonTensor_Cuda;
backends/cuda-ref/ceed-cuda-ref.h:  Fields_Cuda fields;
backends/cuda-ref/ceed-cuda-ref.h:} CeedQFunction_Cuda;
backends/cuda-ref/ceed-cuda-ref.h:} CeedQFunctionContext_Cuda;
backends/cuda-ref/ceed-cuda-ref.h:} CeedOperatorDiag_Cuda;
backends/cuda-ref/ceed-cuda-ref.h:} CeedOperatorAssemble_Cuda;
backends/cuda-ref/ceed-cuda-ref.h:  CeedOperatorDiag_Cuda     *diag;
backends/cuda-ref/ceed-cuda-ref.h:  CeedOperatorAssemble_Cuda *asmb;
backends/cuda-ref/ceed-cuda-ref.h:} CeedOperator_Cuda;
backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedGetCublasHandle_Cuda(Ceed ceed, cublasHandle_t *handle);
backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedVectorCreate_Cuda(CeedSize n, CeedVector vec);
backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedElemRestrictionCreate_Cuda(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, const bool *orients,
backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedBasisCreateTensorH1_Cuda(CeedInt dim, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d, const CeedScalar *grad_1d,
backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedBasisCreateH1_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedBasisCreateHdiv_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedBasisCreateHcurl_Cuda(CeedElemTopology topo, CeedInt dim, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedQFunctionCreate_Cuda(CeedQFunction qf);
backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedQFunctionContextCreate_Cuda(CeedQFunctionContext ctx);
backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedOperatorCreate_Cuda(CeedOperator op);
backends/cuda-ref/ceed-cuda-ref.h:CEED_INTERN int CeedOperatorCreateAtPoints_Cuda(CeedOperator op);
backends/cuda-ref/ceed-cuda-ref-vector.c:#include <cuda_runtime.h>
backends/cuda-ref/ceed-cuda-ref-vector.c:#include "../cuda/ceed-cuda-common.h"
backends/cuda-ref/ceed-cuda-ref-vector.c:#include "ceed-cuda-ref.h"
backends/cuda-ref/ceed-cuda-ref-vector.c:static inline int CeedVectorNeedSync_Cuda(const CeedVector vec, CeedMemType mem_type, bool *need_sync) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:static inline int CeedVectorSyncH2D_Cuda(const CeedVector vec) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallCuda(CeedVectorReturnCeed(vec), cudaMalloc((void **)&impl->d_array_owned, bytes));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallCuda(CeedVectorReturnCeed(vec), cudaMemcpy(impl->d_array, impl->h_array, bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-vector.c:static inline int CeedVectorSyncD2H_Cuda(const CeedVector vec) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallCuda(CeedVectorReturnCeed(vec), cudaMemcpy(impl->h_array, impl->d_array, bytes, cudaMemcpyDeviceToHost));
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorSyncArray_Cuda(const CeedVector vec, CeedMemType mem_type) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedVectorNeedSync_Cuda(vec, mem_type, &need_sync));
backends/cuda-ref/ceed-cuda-ref-vector.c:      return CeedVectorSyncD2H_Cuda(vec);
backends/cuda-ref/ceed-cuda-ref-vector.c:      return CeedVectorSyncH2D_Cuda(vec);
backends/cuda-ref/ceed-cuda-ref-vector.c:static inline int CeedVectorSetAllInvalid_Cuda(const CeedVector vec) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:static inline int CeedVectorHasValidArray_Cuda(const CeedVector vec, bool *has_valid_array) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:static inline int CeedVectorHasArrayOfType_Cuda(const CeedVector vec, CeedMemType mem_type, bool *has_array_of_type) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:static inline int CeedVectorHasBorrowedArrayOfType_Cuda(const CeedVector vec, CeedMemType mem_type, bool *has_borrowed_array_of_type) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorSetArrayHost_Cuda(const CeedVector vec, const CeedCopyMode copy_mode, CeedScalar *array) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorSetArrayDevice_Cuda(const CeedVector vec, const CeedCopyMode copy_mode, CeedScalar *array) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetDeviceCeedScalarArray_Cuda(ceed, array, copy_mode, length, (const CeedScalar **)&impl->d_array_owned,
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorSetArray_Cuda(const CeedVector vec, const CeedMemType mem_type, const CeedCopyMode copy_mode, CeedScalar *array) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedVectorSetAllInvalid_Cuda(vec));
backends/cuda-ref/ceed-cuda-ref-vector.c:      return CeedVectorSetArrayHost_Cuda(vec, copy_mode, array);
backends/cuda-ref/ceed-cuda-ref-vector.c:      return CeedVectorSetArrayDevice_Cuda(vec, copy_mode, array);
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedHostCopyStrided_Cuda(CeedScalar *h_array, CeedSize start, CeedSize step, CeedSize length, CeedScalar *h_copy_array) {
backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedDeviceCopyStrided_Cuda(CeedScalar *d_array, CeedSize start, CeedSize step, CeedSize length, CeedScalar *d_copy_array);
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorCopyStrided_Cuda(CeedVector vec, CeedSize start, CeedSize step, CeedVector vec_copy) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedDeviceCopyStrided_Cuda(impl->d_array, start, step, length, copy_array));
backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedHostCopyStrided_Cuda(impl->h_array, start, step, length, copy_array));
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedHostSetValue_Cuda(CeedScalar *h_array, CeedSize length, CeedScalar val) {
backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedDeviceSetValue_Cuda(CeedScalar *d_array, CeedSize length, CeedScalar val);
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorSetValue_Cuda(CeedVector vec, CeedScalar val) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedDeviceSetValue_Cuda(impl->d_array, length, val));
backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedHostSetValue_Cuda(impl->h_array, length, val));
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedHostSetValueStrided_Cuda(CeedScalar *h_array, CeedSize start, CeedSize step, CeedSize length, CeedScalar val) {
backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedDeviceSetValueStrided_Cuda(CeedScalar *d_array, CeedSize start, CeedSize step, CeedSize length, CeedScalar val);
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorSetValueStrided_Cuda(CeedVector vec, CeedSize start, CeedSize step, CeedScalar val) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedDeviceSetValueStrided_Cuda(impl->d_array, start, step, length, val));
backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedHostSetValueStrided_Cuda(impl->h_array, start, step, length, val));
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorTakeArray_Cuda(CeedVector vec, CeedMemType mem_type, CeedScalar **array) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorGetArrayCore_Cuda(const CeedVector vec, const CeedMemType mem_type, CeedScalar **array) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorGetArrayRead_Cuda(const CeedVector vec, const CeedMemType mem_type, const CeedScalar **array) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  return CeedVectorGetArrayCore_Cuda(vec, mem_type, (CeedScalar **)array);
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorGetArray_Cuda(const CeedVector vec, const CeedMemType mem_type, CeedScalar **array) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedVectorGetArrayCore_Cuda(vec, mem_type, array));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedVectorSetAllInvalid_Cuda(vec));
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorGetArrayWrite_Cuda(const CeedVector vec, const CeedMemType mem_type, CeedScalar **array) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedVectorHasArrayOfType_Cuda(vec, mem_type, &has_array_of_type));
backends/cuda-ref/ceed-cuda-ref-vector.c:  return CeedVectorGetArray_Cuda(vec, mem_type, array);
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorNorm_Cuda(CeedVector vec, CeedNormType type, CeedScalar *norm) {
backends/cuda-ref/ceed-cuda-ref-vector.c:#if CUDA_VERSION < 12000
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda  *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedGetCublasHandle_Cuda(ceed, &handle));
backends/cuda-ref/ceed-cuda-ref-vector.c:#if CUDA_VERSION < 12000
backends/cuda-ref/ceed-cuda-ref-vector.c:  // With CUDA 12, we can use the 64-bit integer interface. Prior to that,
backends/cuda-ref/ceed-cuda-ref-vector.c:#if CUDA_VERSION >= 12000  // We have CUDA 12, and can use 64-bit integers
backends/cuda-ref/ceed-cuda-ref-vector.c:#if CUDA_VERSION >= 12000
backends/cuda-ref/ceed-cuda-ref-vector.c:#if CUDA_VERSION >= 12000
backends/cuda-ref/ceed-cuda-ref-vector.c:#if CUDA_VERSION >= 12000
backends/cuda-ref/ceed-cuda-ref-vector.c:#if CUDA_VERSION >= 12000
backends/cuda-ref/ceed-cuda-ref-vector.c:        CeedCallCuda(ceed, cudaMemcpy(&norm_no_abs, impl->d_array + index - 1, sizeof(CeedScalar), cudaMemcpyDeviceToHost));
backends/cuda-ref/ceed-cuda-ref-vector.c:          CeedCallCuda(ceed, cudaMemcpy(&sub_max, d_array_start + index - 1, sizeof(CeedScalar), cudaMemcpyDeviceToHost));
backends/cuda-ref/ceed-cuda-ref-vector.c:#if CUDA_VERSION >= 12000
backends/cuda-ref/ceed-cuda-ref-vector.c:        CeedCallCuda(ceed, cudaMemcpy(&norm_no_abs, impl->d_array + index - 1, sizeof(CeedScalar), cudaMemcpyDeviceToHost));
backends/cuda-ref/ceed-cuda-ref-vector.c:          CeedCallCuda(ceed, cudaMemcpy(&sub_max, d_array_start + index - 1, sizeof(CeedScalar), cudaMemcpyDeviceToHost));
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedHostReciprocal_Cuda(CeedScalar *h_array, CeedSize length) {
backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedDeviceReciprocal_Cuda(CeedScalar *d_array, CeedSize length);
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorReciprocal_Cuda(CeedVector vec) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:  if (impl->d_array) CeedCallBackend(CeedDeviceReciprocal_Cuda(impl->d_array, length));
backends/cuda-ref/ceed-cuda-ref-vector.c:  if (impl->h_array) CeedCallBackend(CeedHostReciprocal_Cuda(impl->h_array, length));
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedHostScale_Cuda(CeedScalar *x_array, CeedScalar alpha, CeedSize length) {
backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedDeviceScale_Cuda(CeedScalar *x_array, CeedScalar alpha, CeedSize length);
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorScale_Cuda(CeedVector x, CeedScalar alpha) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *x_impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:  if (x_impl->d_array) CeedCallBackend(CeedDeviceScale_Cuda(x_impl->d_array, alpha, length));
backends/cuda-ref/ceed-cuda-ref-vector.c:  if (x_impl->h_array) CeedCallBackend(CeedHostScale_Cuda(x_impl->h_array, alpha, length));
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedHostAXPY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedSize length) {
backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedDeviceAXPY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedSize length);
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorAXPY_Cuda(CeedVector y, CeedScalar alpha, CeedVector x) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *y_impl, *x_impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedDeviceAXPY_Cuda(y_impl->d_array, alpha, x_impl->d_array, length));
backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedHostAXPY_Cuda(y_impl->h_array, alpha, x_impl->h_array, length));
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedHostAXPBY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar beta, CeedScalar *x_array, CeedSize length) {
backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedDeviceAXPBY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar beta, CeedScalar *x_array, CeedSize length);
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorAXPBY_Cuda(CeedVector y, CeedScalar alpha, CeedScalar beta, CeedVector x) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *y_impl, *x_impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedDeviceAXPBY_Cuda(y_impl->d_array, alpha, beta, x_impl->d_array, length));
backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedHostAXPBY_Cuda(y_impl->h_array, alpha, beta, x_impl->h_array, length));
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedHostPointwiseMult_Cuda(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedSize length) {
backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedDevicePointwiseMult_Cuda(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedSize length);
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorPointwiseMult_Cuda(CeedVector w, CeedVector x, CeedVector y) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *w_impl, *x_impl, *y_impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedDevicePointwiseMult_Cuda(w_impl->d_array, x_impl->d_array, y_impl->d_array, length));
backends/cuda-ref/ceed-cuda-ref-vector.c:    CeedCallBackend(CeedHostPointwiseMult_Cuda(w_impl->h_array, x_impl->h_array, y_impl->h_array, length));
backends/cuda-ref/ceed-cuda-ref-vector.c:static int CeedVectorDestroy_Cuda(const CeedVector vec) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallCuda(CeedVectorReturnCeed(vec), cudaFree(impl->d_array_owned));
backends/cuda-ref/ceed-cuda-ref-vector.c:int CeedVectorCreate_Cuda(CeedSize n, CeedVector vec) {
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedVector_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "HasValidArray", CeedVectorHasValidArray_Cuda));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "HasBorrowedArrayOfType", CeedVectorHasBorrowedArrayOfType_Cuda));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetArray", CeedVectorSetArray_Cuda));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "TakeArray", CeedVectorTakeArray_Cuda));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "CopyStrided", CeedVectorCopyStrided_Cuda));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetValue", CeedVectorSetValue_Cuda));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SetValueStrided", CeedVectorSetValueStrided_Cuda));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "SyncArray", CeedVectorSyncArray_Cuda));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArray", CeedVectorGetArray_Cuda));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead", CeedVectorGetArrayRead_Cuda));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayWrite", CeedVectorGetArrayWrite_Cuda));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Norm", CeedVectorNorm_Cuda));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Reciprocal", CeedVectorReciprocal_Cuda));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Scale", CeedVectorScale_Cuda));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "AXPY", CeedVectorAXPY_Cuda));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "AXPBY", CeedVectorAXPBY_Cuda));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "PointwiseMult", CeedVectorPointwiseMult_Cuda));
backends/cuda-ref/ceed-cuda-ref-vector.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Vector", vec, "Destroy", CeedVectorDestroy_Cuda));
backends/cuda-ref/kernels/cuda-ref-vector.cu:#include <cuda.h>
backends/cuda-ref/kernels/cuda-ref-vector.cu:extern "C" int CeedDeviceCopyStrided_Cuda(CeedScalar *d_array, CeedSize start, CeedSize step, CeedSize length, CeedScalar *d_copy_array) {
backends/cuda-ref/kernels/cuda-ref-vector.cu:extern "C" int CeedDeviceSetValue_Cuda(CeedScalar *d_array, CeedSize length, CeedScalar val) {
backends/cuda-ref/kernels/cuda-ref-vector.cu:extern "C" int CeedDeviceSetValueStrided_Cuda(CeedScalar *d_array, CeedSize start, CeedSize step, CeedSize length, CeedScalar val) {
backends/cuda-ref/kernels/cuda-ref-vector.cu:extern "C" int CeedDeviceReciprocal_Cuda(CeedScalar *d_array, CeedSize length) {
backends/cuda-ref/kernels/cuda-ref-vector.cu:extern "C" int CeedDeviceScale_Cuda(CeedScalar *x_array, CeedScalar alpha, CeedSize length) {
backends/cuda-ref/kernels/cuda-ref-vector.cu:extern "C" int CeedDeviceAXPY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar *x_array, CeedSize length) {
backends/cuda-ref/kernels/cuda-ref-vector.cu:extern "C" int CeedDeviceAXPBY_Cuda(CeedScalar *y_array, CeedScalar alpha, CeedScalar beta, CeedScalar *x_array, CeedSize length) {
backends/cuda-ref/kernels/cuda-ref-vector.cu:extern "C" int CeedDevicePointwiseMult_Cuda(CeedScalar *w_array, CeedScalar *x_array, CeedScalar *y_array, CeedSize length) {
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:#include <cuda_runtime.h>
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:#include "../cuda/ceed-cuda-common.h"
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:#include "ceed-cuda-ref.h"
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static inline int CeedQFunctionContextSyncH2D_Cuda(const CeedQFunctionContext ctx) {
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:    CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_data_owned, ctx_size));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallCuda(ceed, cudaMemcpy(impl->d_data, impl->h_data, ctx_size, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static inline int CeedQFunctionContextSyncD2H_Cuda(const CeedQFunctionContext ctx) {
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallCuda(ceed, cudaMemcpy(impl->h_data, impl->d_data, ctx_size, cudaMemcpyDeviceToHost));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static inline int CeedQFunctionContextSync_Cuda(const CeedQFunctionContext ctx, CeedMemType mem_type) {
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:      return CeedQFunctionContextSyncD2H_Cuda(ctx);
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:      return CeedQFunctionContextSyncH2D_Cuda(ctx);
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static inline int CeedQFunctionContextSetAllInvalid_Cuda(const CeedQFunctionContext ctx) {
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static inline int CeedQFunctionContextHasValidData_Cuda(const CeedQFunctionContext ctx, bool *has_valid_data) {
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static inline int CeedQFunctionContextHasBorrowedDataOfType_Cuda(const CeedQFunctionContext ctx, CeedMemType mem_type,
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static inline int CeedQFunctionContextNeedSync_Cuda(const CeedQFunctionContext ctx, CeedMemType mem_type, bool *need_sync) {
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static int CeedQFunctionContextSetDataHost_Cuda(const CeedQFunctionContext ctx, const CeedCopyMode copy_mode, void *data) {
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static int CeedQFunctionContextSetDataDevice_Cuda(const CeedQFunctionContext ctx, const CeedCopyMode copy_mode, void *data) {
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallCuda(ceed, cudaFree(impl->d_data_owned));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:      CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_data_owned, ctx_size));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:      CeedCallCuda(ceed, cudaMemcpy(impl->d_data, data, ctx_size, cudaMemcpyDeviceToDevice));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static int CeedQFunctionContextSetData_Cuda(const CeedQFunctionContext ctx, const CeedMemType mem_type, const CeedCopyMode copy_mode, void *data) {
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedQFunctionContextSetAllInvalid_Cuda(ctx));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:      return CeedQFunctionContextSetDataHost_Cuda(ctx, copy_mode, data);
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:      return CeedQFunctionContextSetDataDevice_Cuda(ctx, copy_mode, data);
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static int CeedQFunctionContextTakeData_Cuda(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedQFunctionContextNeedSync_Cuda(ctx, mem_type, &need_sync));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  if (need_sync) CeedCallBackend(CeedQFunctionContextSync_Cuda(ctx, mem_type));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static int CeedQFunctionContextGetDataCore_Cuda(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedQFunctionContextNeedSync_Cuda(ctx, mem_type, &need_sync));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  if (need_sync) CeedCallBackend(CeedQFunctionContextSync_Cuda(ctx, mem_type));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static int CeedQFunctionContextGetDataRead_Cuda(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  return CeedQFunctionContextGetDataCore_Cuda(ctx, mem_type, data);
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static int CeedQFunctionContextGetData_Cuda(const CeedQFunctionContext ctx, const CeedMemType mem_type, void *data) {
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedQFunctionContextGetDataCore_Cuda(ctx, mem_type, data));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedQFunctionContextSetAllInvalid_Cuda(ctx));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:static int CeedQFunctionContextDestroy_Cuda(const CeedQFunctionContext ctx) {
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallCuda(CeedQFunctionContextReturnCeed(ctx), cudaFree(impl->d_data_owned));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:int CeedQFunctionContextCreate_Cuda(CeedQFunctionContext ctx) {
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedQFunctionContext_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "HasValidData", CeedQFunctionContextHasValidData_Cuda));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "HasBorrowedDataOfType", CeedQFunctionContextHasBorrowedDataOfType_Cuda));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "SetData", CeedQFunctionContextSetData_Cuda));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "TakeData", CeedQFunctionContextTakeData_Cuda));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "GetData", CeedQFunctionContextGetData_Cuda));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "GetDataRead", CeedQFunctionContextGetDataRead_Cuda));
backends/cuda-ref/ceed-cuda-ref-qfunctioncontext.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunctionContext", ctx, "Destroy", CeedQFunctionContextDestroy_Cuda));
backends/cuda-ref/ceed-cuda-ref-qfunction.c:#include <ceed/jit-source/cuda/cuda-types.h>
backends/cuda-ref/ceed-cuda-ref-qfunction.c:#include <cuda.h>
backends/cuda-ref/ceed-cuda-ref-qfunction.c:#include "../cuda/ceed-cuda-common.h"
backends/cuda-ref/ceed-cuda-ref-qfunction.c:#include "../cuda/ceed-cuda-compile.h"
backends/cuda-ref/ceed-cuda-ref-qfunction.c:#include "ceed-cuda-ref-qfunction-load.h"
backends/cuda-ref/ceed-cuda-ref-qfunction.c:#include "ceed-cuda-ref.h"
backends/cuda-ref/ceed-cuda-ref-qfunction.c:static int CeedQFunctionApply_Cuda(CeedQFunction qf, CeedInt Q, CeedVector *U, CeedVector *V) {
backends/cuda-ref/ceed-cuda-ref-qfunction.c:  Ceed_Cuda          *ceed_Cuda;
backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedQFunction_Cuda *data;
backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedCallBackend(CeedQFunctionBuildKernel_Cuda_ref(qf));
backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedCallBackend(CeedGetData(ceed, &ceed_Cuda));
backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedCallBackend(CeedRunKernelAutoblockCuda(ceed, data->QFunction, Q, args));
backends/cuda-ref/ceed-cuda-ref-qfunction.c:static int CeedQFunctionDestroy_Cuda(CeedQFunction qf) {
backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedQFunction_Cuda *data;
backends/cuda-ref/ceed-cuda-ref-qfunction.c:  if (data->module) CeedCallCuda(CeedQFunctionReturnCeed(qf), cuModuleUnload(data->module));
backends/cuda-ref/ceed-cuda-ref-qfunction.c:static int CeedQFunctionSetCUDAUserFunction_Cuda(CeedQFunction qf, CUfunction f) {
backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedQFunction_Cuda *data;
backends/cuda-ref/ceed-cuda-ref-qfunction.c:int CeedQFunctionCreate_Cuda(CeedQFunction qf) {
backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedQFunction_Cuda *data;
backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Apply", CeedQFunctionApply_Cuda));
backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy", CeedQFunctionDestroy_Cuda));
backends/cuda-ref/ceed-cuda-ref-qfunction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "QFunction", qf, "SetCUDAUserFunction", CeedQFunctionSetCUDAUserFunction_Cuda));
backends/cuda-ref/ceed-cuda-ref-restriction.c:#include <cuda.h>
backends/cuda-ref/ceed-cuda-ref-restriction.c:#include <cuda_runtime.h>
backends/cuda-ref/ceed-cuda-ref-restriction.c:#include "../cuda/ceed-cuda-common.h"
backends/cuda-ref/ceed-cuda-ref-restriction.c:#include "../cuda/ceed-cuda-compile.h"
backends/cuda-ref/ceed-cuda-ref-restriction.c:#include "ceed-cuda-ref.h"
backends/cuda-ref/ceed-cuda-ref-restriction.c:static inline int CeedElemRestrictionSetupCompile_Cuda(CeedElemRestriction rstr) {
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedElemRestriction_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-restriction.c:  // Compile CUDA kernels
backends/cuda-ref/ceed-cuda-ref-restriction.c:      const char restriction_kernel_source[] = "// Strided restriction source\n#include <ceed/jit-source/cuda/cuda-ref-restriction-strided.h>\n";
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedCompile_Cuda(ceed, restriction_kernel_source, &impl->module, 6, "RSTR_ELEM_SIZE", elem_size, "RSTR_NUM_ELEM", num_elem,
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "StridedNoTranspose", &impl->ApplyNoTranspose));
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "StridedTranspose", &impl->ApplyTranspose));
backends/cuda-ref/ceed-cuda-ref-restriction.c:      const char restriction_kernel_source[] = "// Standard restriction source\n#include <ceed/jit-source/cuda/cuda-ref-restriction-offset.h>\n";
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedCompile_Cuda(ceed, restriction_kernel_source, &impl->module, 6, "RSTR_ELEM_SIZE", elem_size, "RSTR_NUM_ELEM", num_elem,
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetNoTranspose", &impl->ApplyNoTranspose));
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetTranspose", &impl->ApplyTranspose));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          "// AtPoints restriction source\n#include <ceed/jit-source/cuda/cuda-ref-restriction-at-points.h>\n\n"
backends/cuda-ref/ceed-cuda-ref-restriction.c:          "// Standard restriction source\n#include <ceed/jit-source/cuda/cuda-ref-restriction-offset.h>\n";
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedCompile_Cuda(ceed, restriction_kernel_source, &impl->module, 6, "RSTR_ELEM_SIZE", elem_size, "RSTR_NUM_ELEM", num_elem,
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetNoTranspose", &impl->ApplyNoTranspose));
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "AtPointsTranspose", &impl->ApplyTranspose));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          "// Oriented restriction source\n#include <ceed/jit-source/cuda/cuda-ref-restriction-oriented.h>\n\n"
backends/cuda-ref/ceed-cuda-ref-restriction.c:          "// Standard restriction source\n#include <ceed/jit-source/cuda/cuda-ref-restriction-offset.h>\n";
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedCompile_Cuda(ceed, restriction_kernel_source, &impl->module, 6, "RSTR_ELEM_SIZE", elem_size, "RSTR_NUM_ELEM", num_elem,
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OrientedNoTranspose", &impl->ApplyNoTranspose));
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetNoTranspose", &impl->ApplyUnsignedNoTranspose));
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OrientedTranspose", &impl->ApplyTranspose));
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetTranspose", &impl->ApplyUnsignedTranspose));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          "// Curl oriented restriction source\n#include <ceed/jit-source/cuda/cuda-ref-restriction-curl-oriented.h>\n\n"
backends/cuda-ref/ceed-cuda-ref-restriction.c:          "// Standard restriction source\n#include <ceed/jit-source/cuda/cuda-ref-restriction-offset.h>\n";
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedCompile_Cuda(ceed, restriction_kernel_source, &impl->module, 6, "RSTR_ELEM_SIZE", elem_size, "RSTR_NUM_ELEM", num_elem,
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "CurlOrientedNoTranspose", &impl->ApplyNoTranspose));
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "CurlOrientedUnsignedNoTranspose", &impl->ApplyUnsignedNoTranspose));
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetNoTranspose", &impl->ApplyUnorientedNoTranspose));
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "CurlOrientedTranspose", &impl->ApplyTranspose));
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "CurlOrientedUnsignedTranspose", &impl->ApplyUnsignedTranspose));
backends/cuda-ref/ceed-cuda-ref-restriction.c:      CeedCallBackend(CeedGetKernel_Cuda(ceed, impl->module, "OffsetTranspose", &impl->ApplyUnorientedTranspose));
backends/cuda-ref/ceed-cuda-ref-restriction.c:static inline int CeedElemRestrictionApply_Cuda_Core(CeedElemRestriction rstr, CeedTransposeMode t_mode, bool use_signs, bool use_orients,
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedElemRestriction_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-restriction.c:    CeedCallBackend(CeedElemRestrictionSetupCompile_Cuda(rstr));
backends/cuda-ref/ceed-cuda-ref-restriction.c:        CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyNoTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:        CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyNoTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyNoTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedNoTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyNoTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedNoTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnorientedNoTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:        CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnsignedTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnorientedTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:            CeedCallBackend(CeedRunKernel_Cuda(ceed, impl->ApplyUnorientedTranspose, grid, block_size, args));
backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionApply_Cuda(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector v, CeedRequest *request) {
backends/cuda-ref/ceed-cuda-ref-restriction.c:  return CeedElemRestrictionApply_Cuda_Core(rstr, t_mode, true, true, u, v, request);
backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionApplyUnsigned_Cuda(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
backends/cuda-ref/ceed-cuda-ref-restriction.c:  return CeedElemRestrictionApply_Cuda_Core(rstr, t_mode, false, true, u, v, request);
backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionApplyUnoriented_Cuda(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
backends/cuda-ref/ceed-cuda-ref-restriction.c:  return CeedElemRestrictionApply_Cuda_Core(rstr, t_mode, false, false, u, v, request);
backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionGetOffsets_Cuda(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt **offsets) {
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedElemRestriction_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionGetOrientations_Cuda(CeedElemRestriction rstr, CeedMemType mem_type, const bool **orients) {
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedElemRestriction_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionGetCurlOrientations_Cuda(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt8 **curl_orients) {
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedElemRestriction_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionGetAtPointsElementOffset_Cuda(CeedElemRestriction rstr, CeedInt elem, CeedSize *elem_offset) {
backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionDestroy_Cuda(CeedElemRestriction rstr) {
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedElemRestriction_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-restriction.c:    CeedCallCuda(ceed, cuModuleUnload(impl->module));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_offsets_owned));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_t_offsets));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_t_indices));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_l_vec_indices));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaFree((bool *)impl->d_orients_owned));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaFree((CeedInt8 *)impl->d_curl_orients_owned));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_offsets_at_points_owned));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaFree((CeedInt *)impl->d_points_per_elem_owned));
backends/cuda-ref/ceed-cuda-ref-restriction.c:static int CeedElemRestrictionOffset_Cuda(const CeedElemRestriction rstr, const CeedInt elem_size, const CeedInt *indices) {
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedElemRestriction_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_l_vec_indices, num_nodes * sizeof(CeedInt)));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaMemcpy((CeedInt *)impl->d_l_vec_indices, l_vec_indices, num_nodes * sizeof(CeedInt), cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_t_offsets, size_offsets * sizeof(CeedInt)));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaMemcpy((CeedInt *)impl->d_t_offsets, t_offsets, size_offsets * sizeof(CeedInt), cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_t_indices, size_indices * sizeof(CeedInt)));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallCuda(ceed, cudaMemcpy((CeedInt *)impl->d_t_indices, t_indices, size_indices * sizeof(CeedInt), cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-restriction.c:int CeedElemRestrictionCreate_Cuda(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, const bool *orients,
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedElemRestriction_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-restriction.c:    CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_offsets_at_points_owned, at_points_size * sizeof(CeedInt)));
backends/cuda-ref/ceed-cuda-ref-restriction.c:    CeedCallCuda(ceed, cudaMemcpy((CeedInt **)impl->d_offsets_at_points_owned, impl->h_offsets_at_points, at_points_size * sizeof(CeedInt),
backends/cuda-ref/ceed-cuda-ref-restriction.c:                                  cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-restriction.c:    CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_points_per_elem_owned, num_elem * sizeof(CeedInt)));
backends/cuda-ref/ceed-cuda-ref-restriction.c:    CeedCallCuda(ceed,
backends/cuda-ref/ceed-cuda-ref-restriction.c:                 cudaMemcpy((CeedInt **)impl->d_points_per_elem_owned, impl->h_points_per_elem, num_elem * sizeof(CeedInt), cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-restriction.c:        CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_offsets_owned, size * sizeof(CeedInt)));
backends/cuda-ref/ceed-cuda-ref-restriction.c:        CeedCallCuda(ceed, cudaMemcpy((CeedInt *)impl->d_offsets_owned, impl->h_offsets, size * sizeof(CeedInt), cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-restriction.c:        if (is_deterministic) CeedCallBackend(CeedElemRestrictionOffset_Cuda(rstr, elem_size, offsets));
backends/cuda-ref/ceed-cuda-ref-restriction.c:        CeedCallBackend(CeedSetDeviceCeedIntArray_Cuda(ceed, offsets, copy_mode, size, &impl->d_offsets_owned, &impl->d_offsets_borrowed,
backends/cuda-ref/ceed-cuda-ref-restriction.c:        CeedCallCuda(ceed, cudaMemcpy((CeedInt *)impl->h_offsets_owned, impl->d_offsets, size * sizeof(CeedInt), cudaMemcpyDeviceToHost));
backends/cuda-ref/ceed-cuda-ref-restriction.c:        if (is_deterministic) CeedCallBackend(CeedElemRestrictionOffset_Cuda(rstr, elem_size, offsets));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_orients_owned, size * sizeof(bool)));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallCuda(ceed, cudaMemcpy((bool *)impl->d_orients_owned, impl->h_orients, size * sizeof(bool), cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedSetDeviceBoolArray_Cuda(ceed, orients, copy_mode, size, &impl->d_orients_owned, &impl->d_orients_borrowed,
backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallCuda(ceed, cudaMemcpy((bool *)impl->h_orients_owned, impl->d_orients, size * sizeof(bool), cudaMemcpyDeviceToHost));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallCuda(ceed, cudaMalloc((void **)&impl->d_curl_orients_owned, 3 * size * sizeof(CeedInt8)));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallCuda(ceed,
backends/cuda-ref/ceed-cuda-ref-restriction.c:                       cudaMemcpy((CeedInt8 *)impl->d_curl_orients_owned, impl->h_curl_orients, 3 * size * sizeof(CeedInt8), cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallBackend(CeedSetDeviceCeedInt8Array_Cuda(ceed, curl_orients, copy_mode, 3 * size, &impl->d_curl_orients_owned,
backends/cuda-ref/ceed-cuda-ref-restriction.c:          CeedCallCuda(ceed,
backends/cuda-ref/ceed-cuda-ref-restriction.c:                       cudaMemcpy((CeedInt8 *)impl->h_curl_orients_owned, impl->d_curl_orients, 3 * size * sizeof(CeedInt8), cudaMemcpyDeviceToHost));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "Apply", CeedElemRestrictionApply_Cuda));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "ApplyUnsigned", CeedElemRestrictionApplyUnsigned_Cuda));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "ApplyUnoriented", CeedElemRestrictionApplyUnoriented_Cuda));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetOffsets", CeedElemRestrictionGetOffsets_Cuda));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetOrientations", CeedElemRestrictionGetOrientations_Cuda));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetCurlOrientations", CeedElemRestrictionGetCurlOrientations_Cuda));
backends/cuda-ref/ceed-cuda-ref-restriction.c:        CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetAtPointsElementOffset", CeedElemRestrictionGetAtPointsElementOffset_Cuda));
backends/cuda-ref/ceed-cuda-ref-restriction.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "Destroy", CeedElemRestrictionDestroy_Cuda));
backends/cuda-ref/ceed-cuda-ref-operator.c:#include <cuda.h>
backends/cuda-ref/ceed-cuda-ref-operator.c:#include <cuda_runtime.h>
backends/cuda-ref/ceed-cuda-ref-operator.c:#include "../cuda/ceed-cuda-common.h"
backends/cuda-ref/ceed-cuda-ref-operator.c:#include "../cuda/ceed-cuda-compile.h"
backends/cuda-ref/ceed-cuda-ref-operator.c:#include "ceed-cuda-ref.h"
backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorDestroy_Cuda(CeedOperator op) {
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedCallCuda(ceed, cuModuleUnload(impl->diag->module));
backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedCallCuda(ceed, cuModuleUnload(impl->diag->module_point_block));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_eval_modes_in));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_eval_modes_out));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_identity));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_interp_in));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_interp_out));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_grad_in));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_grad_out));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_div_in));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_div_out));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_curl_in));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->diag->d_curl_out));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cuModuleUnload(impl->asmb->module));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->asmb->d_B_in));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaFree(impl->asmb->d_B_out));
backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorSetupFields_Cuda(CeedQFunction qf, CeedOperator op, bool is_input, bool is_at_points, bool *skip_rstr, bool *apply_add_basis,
backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorSetup_Cuda(CeedOperator op) {
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedOperatorSetupFields_Cuda(qf, op, true, false, impl->skip_rstr_in, NULL, impl->e_vecs_in, impl->q_vecs_in, num_input_fields, Q, num_elem));
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorSetupFields_Cuda(qf, op, false, false, impl->skip_rstr_out, impl->apply_add_basis_out, impl->e_vecs_out,
backends/cuda-ref/ceed-cuda-ref-operator.c:static inline int CeedOperatorInputRestrict_Cuda(CeedOperatorField op_input_field, CeedQFunctionField qf_input_field, CeedInt input_field,
backends/cuda-ref/ceed-cuda-ref-operator.c:                                                 CeedVector in_vec, CeedVector active_e_vec, const bool skip_active, CeedOperator_Cuda *impl,
backends/cuda-ref/ceed-cuda-ref-operator.c:static inline int CeedOperatorInputBasis_Cuda(CeedOperatorField op_input_field, CeedQFunctionField qf_input_field, CeedInt input_field,
backends/cuda-ref/ceed-cuda-ref-operator.c:                                              CeedOperator_Cuda *impl) {
backends/cuda-ref/ceed-cuda-ref-operator.c:static inline int CeedOperatorInputRestore_Cuda(CeedOperatorField op_input_field, CeedQFunctionField qf_input_field, CeedInt input_field,
backends/cuda-ref/ceed-cuda-ref-operator.c:                                                CeedVector in_vec, CeedVector active_e_vec, const bool skip_active, CeedOperator_Cuda *impl) {
backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorApplyAdd_Cuda(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorSetup_Cuda(op));
backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedOperatorInputRestrict_Cuda(op_input_fields[field], qf_input_fields[field], field, in_vec, active_e_vec, false, impl, request));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputBasis_Cuda(op_input_fields[field], qf_input_fields[field], field, in_vec, active_e_vec, num_elem, false, impl));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputRestore_Cuda(op_input_fields[i], qf_input_fields[i], i, in_vec, active_e_vec, false, impl));
backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorSetupAtPoints_Cuda(CeedOperator op) {
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorSetupFields_Cuda(qf, op, true, true, impl->skip_rstr_in, NULL, impl->e_vecs_in, impl->q_vecs_in, num_input_fields,
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorSetupFields_Cuda(qf, op, false, true, impl->skip_rstr_out, impl->apply_add_basis_out, impl->e_vecs_out,
backends/cuda-ref/ceed-cuda-ref-operator.c:static inline int CeedOperatorInputBasisAtPoints_Cuda(CeedOperatorField op_input_field, CeedQFunctionField qf_input_field, CeedInt input_field,
backends/cuda-ref/ceed-cuda-ref-operator.c:                                                      const bool skip_active, CeedOperator_Cuda *impl) {
backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorApplyAddAtPoints_Cuda(CeedOperator op, CeedVector in_vec, CeedVector out_vec, CeedRequest *request) {
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorSetupAtPoints_Cuda(op));
backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedOperatorInputRestrict_Cuda(op_input_fields[field], qf_input_fields[field], field, in_vec, active_e_vec, false, impl, request));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputBasisAtPoints_Cuda(op_input_fields[field], qf_input_fields[field], field, in_vec, active_e_vec, num_elem,
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputRestore_Cuda(op_input_fields[i], qf_input_fields[i], i, in_vec, active_e_vec, false, impl));
backends/cuda-ref/ceed-cuda-ref-operator.c:static inline int CeedOperatorLinearAssembleQFunctionCore_Cuda(CeedOperator op, bool build_objects, CeedVector *assembled, CeedElemRestriction *rstr,
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorSetup_Cuda(op));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputRestrict_Cuda(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, true, impl, request));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputBasis_Cuda(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, num_elem, true, impl));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputRestore_Cuda(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, true, impl));
backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorLinearAssembleQFunction_Cuda(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
backends/cuda-ref/ceed-cuda-ref-operator.c:  return CeedOperatorLinearAssembleQFunctionCore_Cuda(op, true, assembled, rstr, request);
backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorLinearAssembleQFunctionUpdate_Cuda(CeedOperator op, CeedVector assembled, CeedElemRestriction rstr, CeedRequest *request) {
backends/cuda-ref/ceed-cuda-ref-operator.c:  return CeedOperatorLinearAssembleQFunctionCore_Cuda(op, false, &assembled, &rstr, request);
backends/cuda-ref/ceed-cuda-ref-operator.c:static inline int CeedOperatorAssembleDiagonalSetup_Cuda(CeedOperator op) {
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperatorDiag_Cuda *diag = impl->diag;
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaMalloc((void **)&diag->d_identity, interp_bytes));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaMemcpy(diag->d_identity, identity, interp_bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMalloc((void **)&d_interp, interp_bytes * q_comp_interp));
backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMemcpy(d_interp, interp, interp_bytes * q_comp_interp, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMalloc((void **)&d_grad, interp_bytes * q_comp_grad));
backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMemcpy(d_grad, grad, interp_bytes * q_comp_grad, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMalloc((void **)&d_interp, interp_bytes * q_comp_interp));
backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMemcpy(d_interp, interp, interp_bytes * q_comp_interp, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMalloc((void **)&d_div, interp_bytes * q_comp_div));
backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMemcpy(d_div, div, interp_bytes * q_comp_div, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMalloc((void **)&d_interp, interp_bytes * q_comp_interp));
backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMemcpy(d_interp, interp, interp_bytes * q_comp_interp, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMalloc((void **)&d_curl, interp_bytes * q_comp_curl));
backends/cuda-ref/ceed-cuda-ref-operator.c:        CeedCallCuda(ceed, cudaMemcpy(d_curl, curl, interp_bytes * q_comp_curl, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallCuda(ceed, cudaMalloc((void **)&diag->d_eval_modes_in, num_eval_modes_in * eval_modes_bytes));
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallCuda(ceed, cudaMemcpy(diag->d_eval_modes_in, eval_modes_in, num_eval_modes_in * eval_modes_bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallCuda(ceed, cudaMalloc((void **)&diag->d_eval_modes_out, num_eval_modes_out * eval_modes_bytes));
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallCuda(ceed, cudaMemcpy(diag->d_eval_modes_out, eval_modes_out, num_eval_modes_out * eval_modes_bytes, cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-operator.c:static inline int CeedOperatorAssembleDiagonalSetupCompile_Cuda(CeedOperator op, CeedInt use_ceedsize_idx, const bool is_point_block) {
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperatorDiag_Cuda *diag = impl->diag;
backends/cuda-ref/ceed-cuda-ref-operator.c:  const char diagonal_kernel_source[] = "// Diagonal assembly source\n#include <ceed/jit-source/cuda/cuda-ref-operator-assemble-diagonal.h>\n";
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallCuda(ceed, CeedCompile_Cuda(ceed, diagonal_kernel_source, module, 8, "NUM_EVAL_MODES_IN", num_eval_modes_in, "NUM_EVAL_MODES_OUT",
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallCuda(ceed, CeedGetKernel_Cuda(ceed, *module, "LinearDiagonal", is_point_block ? &diag->LinearPointBlock : &diag->LinearDiagonal));
backends/cuda-ref/ceed-cuda-ref-operator.c:static inline int CeedOperatorAssembleDiagonalCore_Cuda(CeedOperator op, CeedVector assembled, CeedRequest *request, const bool is_point_block) {
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
backends/cuda-ref/ceed-cuda-ref-operator.c:  if (!impl->diag) CeedCallBackend(CeedOperatorAssembleDiagonalSetup_Cuda(op));
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperatorDiag_Cuda *diag = impl->diag;
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorAssembleDiagonalSetupCompile_Cuda(op, use_ceedsize_idx, is_point_block));
backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedCallBackend(CeedRunKernelDim_Cuda(ceed, diag->LinearPointBlock, grid, num_nodes, 1, elems_per_block, args));
backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedCallBackend(CeedRunKernelDim_Cuda(ceed, diag->LinearDiagonal, grid, num_nodes, 1, elems_per_block, args));
backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorLinearAssembleAddDiagonal_Cuda(CeedOperator op, CeedVector assembled, CeedRequest *request) {
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorAssembleDiagonalCore_Cuda(op, assembled, request, false));
backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorLinearAssembleAddPointBlockDiagonal_Cuda(CeedOperator op, CeedVector assembled, CeedRequest *request) {
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorAssembleDiagonalCore_Cuda(op, assembled, request, true));
backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedSingleOperatorAssembleSetup_Cuda(CeedOperator op, CeedInt use_ceedsize_idx) {
backends/cuda-ref/ceed-cuda-ref-operator.c:  Ceed_Cuda          *cuda_data;
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperatorAssemble_Cuda *asmb = impl->asmb;
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedGetData(ceed, &cuda_data));
backends/cuda-ref/ceed-cuda-ref-operator.c:  bool fallback = asmb->block_size_x * asmb->block_size_y * asmb->elems_per_block > cuda_data->device_prop.maxThreadsPerBlock;
backends/cuda-ref/ceed-cuda-ref-operator.c:  const char assembly_kernel_source[] = "// Full assembly source\n#include <ceed/jit-source/cuda/cuda-ref-operator-assemble.h>\n";
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedCompile_Cuda(ceed, assembly_kernel_source, &asmb->module, 10, "NUM_EVAL_MODES_IN", num_eval_modes_in, "NUM_EVAL_MODES_OUT",
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedGetKernel_Cuda(ceed, asmb->module, "LinearAssemble", &asmb->LinearAssemble));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaMalloc((void **)&asmb->d_B_in, in_bytes));
backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedCallCuda(ceed, cudaMemcpy(&asmb->d_B_in[i * elem_size_in * num_qpts_in], h_B_in, elem_size_in * num_qpts_in * sizeof(CeedScalar),
backends/cuda-ref/ceed-cuda-ref-operator.c:                                    cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallCuda(ceed, cudaMalloc((void **)&asmb->d_B_out, out_bytes));
backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedCallCuda(ceed, cudaMemcpy(&asmb->d_B_out[i * elem_size_out * num_qpts_out], h_B_out, elem_size_out * num_qpts_out * sizeof(CeedScalar),
backends/cuda-ref/ceed-cuda-ref-operator.c:                                    cudaMemcpyHostToDevice));
backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedSingleOperatorAssemble_Cuda(CeedOperator op, CeedInt offset, CeedVector values) {
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
backends/cuda-ref/ceed-cuda-ref-operator.c:  if (!impl->asmb) CeedCallBackend(CeedSingleOperatorAssembleSetup_Cuda(op, use_ceedsize_idx));
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperatorAssemble_Cuda *asmb = impl->asmb;
backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedRunKernelDimShared_Cuda(ceed, asmb->LinearAssemble, grid, asmb->block_size_x, asmb->block_size_y, asmb->elems_per_block, shared_mem, args));
backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorLinearAssembleQFunctionAtPoints_Cuda(CeedOperator op, CeedVector *assembled, CeedElemRestriction *rstr, CeedRequest *request) {
backends/cuda-ref/ceed-cuda-ref-operator.c:static int CeedOperatorLinearAssembleAddDiagonalAtPoints_Cuda(CeedOperator op, CeedVector assembled, CeedRequest *request) {
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda  *impl;
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedOperatorSetupAtPoints_Cuda(op));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputRestrict_Cuda(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, true, impl, request));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputBasisAtPoints_Cuda(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, num_elem, num_points, true, impl));
backends/cuda-ref/ceed-cuda-ref-operator.c:    CeedCallBackend(CeedOperatorInputRestore_Cuda(op_input_fields[i], qf_input_fields[i], i, NULL, NULL, true, impl));
backends/cuda-ref/ceed-cuda-ref-operator.c:int CeedOperatorCreate_Cuda(CeedOperator op) {
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunction_Cuda));
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunctionUpdate", CeedOperatorLinearAssembleQFunctionUpdate_Cuda));
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal", CeedOperatorLinearAssembleAddDiagonal_Cuda));
backends/cuda-ref/ceed-cuda-ref-operator.c:      CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddPointBlockDiagonal", CeedOperatorLinearAssembleAddPointBlockDiagonal_Cuda));
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleSingle", CeedSingleOperatorAssemble_Cuda));
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAdd_Cuda));
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Cuda));
backends/cuda-ref/ceed-cuda-ref-operator.c:int CeedOperatorCreateAtPoints_Cuda(CeedOperator op) {
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedOperator_Cuda *impl;
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleQFunction", CeedOperatorLinearAssembleQFunctionAtPoints_Cuda));
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "LinearAssembleAddDiagonal", CeedOperatorLinearAssembleAddDiagonalAtPoints_Cuda));
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "ApplyAdd", CeedOperatorApplyAddAtPoints_Cuda));
backends/cuda-ref/ceed-cuda-ref-operator.c:  CeedCallBackend(CeedSetBackendFunction(ceed, "Operator", op, "Destroy", CeedOperatorDestroy_Cuda));
backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Cuda, 1, "/gpu/cuda/ref")
backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Cuda_Gen, 1, "/gpu/cuda/gen")
backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Cuda_Shared, 1, "/gpu/cuda/shared")
backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Hip, 1, "/gpu/hip/ref")
backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Hip_Gen, 1, "/gpu/hip/gen")
backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Hip_Shared, 1, "/gpu/hip/shared")
backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Sycl, 1, "/gpu/sycl/ref")
backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Sycl_Shared, 1, "/gpu/sycl/shared")
backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Sycl_Gen, 1, "/gpu/sycl/gen")
backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Magma, 2, "/gpu/cuda/magma", "/gpu/hip/magma")
backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Magma_Det, 2, "/gpu/cuda/magma/det", "/gpu/hip/magma/det")
backends/ceed-backend-list.h:CEED_BACKEND(CeedRegister_Occa, 6, "/cpu/self/occa", "/cpu/openmp/occa", "/gpu/dpcpp/occa", "/gpu/opencl/occa", "/gpu/hip/occa", "/gpu/cuda/occa")
backends/magma/ceed-magma-common.c:  magma_queue_create_from_cuda(data->device_id, NULL, NULL, NULL, &(data->queue));
backends/magma/ceed-magma-gemm-selector.cpp:static inline auto gemm_selector_get_data(int gpu_arch, char precision, char trans_A) -> decltype(dgemm_nn_mi100) {
backends/magma/ceed-magma-gemm-selector.cpp:  if (gpu_arch >= 910) {
backends/magma/ceed-magma-gemm-selector.cpp:static inline auto gemm_selector_get_data(int gpu_arch, char precision, char trans_A) -> decltype(dgemm_nn_v100) {
backends/magma/ceed-magma-gemm-selector.cpp:  if (gpu_arch >= 800) {
backends/magma/ceed-magma-gemm-selector.cpp:void gemm_selector(int gpu_arch, char precision, char trans_A, int m, int n, int k, int *n_batch, int *use_magma) {
backends/magma/ceed-magma-gemm-selector.cpp:  const auto &data = gemm_selector_get_data(gpu_arch, precision, trans_A);
backends/magma/ceed-magma-gemm-selector.cpp:static inline auto nontensor_rtc_get_data(int gpu_arch, char trans_A) -> decltype(drtc_n_mi100) {
backends/magma/ceed-magma-gemm-selector.cpp:  if (gpu_arch >= 910) {
backends/magma/ceed-magma-gemm-selector.cpp:static inline auto nontensor_rtc_get_data(int gpu_arch, char trans_A) -> decltype(drtc_n_v100) {
backends/magma/ceed-magma-gemm-selector.cpp:  if (gpu_arch >= 900) {
backends/magma/ceed-magma-gemm-selector.cpp:  } else if (gpu_arch >= 800) {
backends/magma/ceed-magma-gemm-selector.cpp:CeedInt nontensor_rtc_get_nb(int gpu_arch, char trans_A, int q_comp, int P, int Q, int N) {
backends/magma/ceed-magma-gemm-selector.cpp:  const auto &data = nontensor_rtc_get_data(gpu_arch, trans_A);
backends/magma/ceed-magma-gemm-selector.h:CEED_INTERN void gemm_selector(int gpu_arch, char precision, char trans_A, int m, int n, int k, int *n_batch, int *use_magma);
backends/magma/ceed-magma-gemm-selector.h:CEED_INTERN CeedInt nontensor_rtc_get_nb(int gpu_arch, char trans_A, int q_comp, int P, int Q, int N);
backends/magma/ceed-magma.h:#define CeedCompileMagma CeedCompile_Cuda
backends/magma/ceed-magma.h:#define CeedGetKernelMagma CeedGetKernel_Cuda
backends/magma/ceed-magma.h:#define CeedRunKernelMagma CeedRunKernel_Cuda
backends/magma/ceed-magma.h:#define CeedRunKernelDimMagma CeedRunKernelDim_Cuda
backends/magma/ceed-magma.h:#define CeedRunKernelDimSharedMagma CeedRunKernelDimShared_Cuda
backends/magma/ceed-magma.h:// If magma and cuda/ref are using the null stream, then ceed_magma_queue_sync should do nothing
backends/magma/tuning/mi100.h:// auto-generated from data on mi100-rocm5.0.2
backends/magma/tuning/generate_tuning.py:            ("hipDeviceSynchronize()" if "hip" in backend else "cudaDeviceSynchronize()"),
backends/magma/tuning/README.md:The `magma` backend uses specialized GPU kernels for a non-tensor basis with
backends/magma/tuning/README.md:header files called `<ARCH>_rtc.h`, where `<ARCH>` is the GPU name, as well as a
backends/magma/tuning/README.md:A sample run to generate the tuning data for an A100 GPU, considering values of
backends/magma/tuning/README.md:python generate_tuning.py -arch a100 -max-nb 32 -build-cmd "make" -ceed "/gpu/cuda/magma"
backends/magma/tuning/README.md:specifies the backend to use, typically one of `/gpu/cuda/magma` or
backends/magma/tuning/README.md:`/gpu/hip/magma`.
backends/magma/tuning/README.md:./tuning "/gpu/cuda/magma"
backends/magma/tuning/README.md:`cudaDeviceSynchronize()` or `hipDeviceSynchronize()`.
backends/magma/tuning/mi250x.h:// auto-generated from data on mi250x-rocm5.1.0
backends/magma/tuning/v100.h:// auto-generated from data on v100-cuda11.2
backends/magma/tuning/a100.h:// auto-generated from data on a100-cuda11.2
backends/magma/ceed-magma.c:  CeedCheck(!strncmp(resource, "/gpu/cuda/magma", nrc) || !strncmp(resource, "/gpu/hip/magma", nrc), ceed, CEED_ERROR_BACKEND,
backends/magma/ceed-magma.c:  CeedCallBackend(CeedInit("/gpu/hip/ref", &ceed_ref));
backends/magma/ceed-magma.c:  CeedCallBackend(CeedInit("/gpu/cuda/ref", &ceed_ref));
backends/magma/ceed-magma.c:  return CeedRegister("/gpu/hip/magma", CeedInit_Magma, 120);
backends/magma/ceed-magma.c:  return CeedRegister("/gpu/cuda/magma", CeedInit_Magma, 120);
backends/magma/ceed-magma-det.c:  CeedCheck(!strncmp(resource, "/gpu/cuda/magma/det", nrc) || !strncmp(resource, "/gpu/hip/magma/det", nrc), ceed, CEED_ERROR_BACKEND,
backends/magma/ceed-magma-det.c:  CeedCallBackend(CeedInit("/gpu/hip/magma", &ceed_ref));
backends/magma/ceed-magma-det.c:  CeedCallBackend(CeedInit("/gpu/cuda/magma", &ceed_ref));
backends/magma/ceed-magma-det.c:  return CeedRegister("/gpu/hip/magma/det", CeedInit_Magma_Det, 125);
backends/magma/ceed-magma-det.c:  return CeedRegister("/gpu/cuda/magma/det", CeedInit_Magma_Det, 125);
backends/magma/ceed-magma-basis.c:#include "../cuda/ceed-cuda-common.h"
backends/magma/ceed-magma-basis.c:#include "../cuda/ceed-cuda-compile.h"
backends/magma/ceed-magma-basis.c:      // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
backends/magma/ceed-magma-basis.c:  CeedCallCuda(ceed, cuModuleUnload(impl->module));
backends/magma/ceed-magma-basis.c:      CeedCallCuda(ceed, cuModuleUnload(impl->module[in]));
backends/magma/ceed-magma-basis.c:  // Copy basis data to GPU
backends/magma/ceed-magma-basis.c:  // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
backends/magma/ceed-magma-basis.c:  // Copy basis data to GPU
backends/magma/ceed-magma-basis.c:    // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
backends/magma/ceed-magma-basis.c:  // Copy basis data to GPU
backends/magma/ceed-magma-basis.c:    // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
backends/magma/ceed-magma-basis.c:  // Copy basis data to GPU
backends/magma/ceed-magma-basis.c:    // The RTC compilation code expects a Ceed with the common Ceed_Cuda or Ceed_Hip data
backends/hip-gen/ceed-hip-gen.c:  const char fallback_resource[] = "/gpu/hip/ref";
backends/hip-gen/ceed-hip-gen.c:  CeedCheck(!strcmp(resource_root, "/gpu/hip") || !strcmp(resource_root, "/gpu/hip/gen"), ceed, CEED_ERROR_BACKEND,
backends/hip-gen/ceed-hip-gen.c:  CeedCallBackend(CeedInit("/gpu/hip/shared", &ceed_shared));
backends/hip-gen/ceed-hip-gen.c:CEED_INTERN int CeedRegister_Hip_Gen(void) { return CeedRegister("/gpu/hip/gen", CeedInit_Hip_gen, 20); }
backends/hip-gen/ceed-hip-gen-operator.c:      CeedDebug256(CeedOperatorReturnCeed(op), CEED_DEBUG_COLOR_SUCCESS, "Falling back to /gpu/hip/ref CeedOperator due to non-tensor bases");
backends/hip-gen/ceed-hip-gen-operator-build.cpp:    CeedCheck(source_path, ceed, CEED_ERROR_UNSUPPORTED, "/gpu/hip/gen backend requires QFunction source code file");
tests/t409-qfunction.c:  //   to inconsistent data on the GPU.
tests/t119-vector.c:    // Sync memtype to device for GPU backends
tests/t122-vector.c:    // Sync memtype to device for GPU backends
tests/t108-vector.c:    // Sync memtype to device for GPU backends
tests/t121-vector.c:    // Sync memtype to device for GPU backends
tests/t125-vector.c:    // Sync memtype to device for GPU backends
tests/test-include/fake-sys-include.h:// Note - files included this way cannot transitively include any files CUDA/ROCm won't compile
tests/junit.py:        if test.startswith('t318') and contains_any(resource, ['/gpu/cuda/ref']):
tests/junit.py:            return 'CUDA ref backend not supported'
tests/junit.py:        if test.startswith('t506') and contains_any(resource, ['/gpu/cuda/shared']):
tests/junit.py:            return 'CUDA shared backend not supported'
tests/junit.py:            if (condition == 'cpu') and ('gpu' in resource):
tests/junit.py:                return 'CPU only test with GPU backend'
tests/t101-vector.c:  // Sync memtype to device for GPU backends
tests/t123-vector.c:    // Sync memtype to device for GPU backends
Makefile:NVCC ?= $(CUDA_DIR)/bin/nvcc
Makefile:HIPCC ?= $(ROCM_DIR)/bin/hipcc
Makefile:# Often /opt/cuda or /usr/local/cuda, but sometimes present on machines that don't support CUDA
Makefile:CUDA_DIR  ?=
Makefile:CUDA_ARCH ?=
Makefile:# Often /opt/rocm, but sometimes present on machines that don't support HIP
Makefile:ROCM_DIR ?=
Makefile:# Warning: SANTIZ options still don't run with /gpu/occa
Makefile:ifneq ($(CUDA_ARCH),)
Makefile:  NVCCFLAGS += -arch=$(CUDA_ARCH)
Makefile:  HIPCCFLAGS += --amdgpu-target=$(HIP_ARCH)
Makefile:libceed.c := $(filter-out interface/ceed-cuda.c interface/ceed-hip.c interface/ceed-jit-source-root-$(if $(for_install),default,install).c, $(wildcard interface/ceed*.c backends/*.c gallery/*.c))
Makefile:cuda.c         := $(sort $(wildcard backends/cuda/*.c))
Makefile:cuda.cpp       := $(sort $(wildcard backends/cuda/*.cpp))
Makefile:cuda-ref.c     := $(sort $(wildcard backends/cuda-ref/*.c))
Makefile:cuda-ref.cpp   := $(sort $(wildcard backends/cuda-ref/*.cpp))
Makefile:cuda-ref.cu    := $(sort $(wildcard backends/cuda-ref/kernels/*.cu))
Makefile:cuda-shared.c  := $(sort $(wildcard backends/cuda-shared/*.c))
Makefile:cuda-shared.cu := $(sort $(wildcard backends/cuda-shared/kernels/*.cu))
Makefile:cuda-gen.c     := $(sort $(wildcard backends/cuda-gen/*.c))
Makefile:cuda-gen.cpp   := $(sort $(wildcard backends/cuda-gen/*.cpp))
Makefile:cuda-gen.cu    := $(sort $(wildcard backends/cuda-gen/kernels/*.cu))
Makefile:	$(info CUDA_DIR      = $(CUDA_DIR)$(call backend_status,$(CUDA_BACKENDS)))
Makefile:	$(info ROCM_DIR      = $(ROCM_DIR)$(call backend_status,$(HIP_BACKENDS)))
Makefile:  OCCA_BACKENDS += $(if $(filter dpcpp,$(OCCA_MODES)),/gpu/dpcpp/occa)
Makefile:  OCCA_BACKENDS += $(if $(filter OpenCL,$(OCCA_MODES)),/gpu/opencl/occa)
Makefile:  OCCA_BACKENDS += $(if $(filter HIP,$(OCCA_MODES)),/gpu/hip/occa)
Makefile:  OCCA_BACKENDS += $(if $(filter CUDA,$(OCCA_MODES)),/gpu/cuda/occa)
Makefile:# CUDA Backends
Makefile:ifneq ($(CUDA_DIR),)
Makefile:  CUDA_LIB_DIR := $(wildcard $(foreach d,lib lib64 lib/x86_64-linux-gnu,$(CUDA_DIR)/$d/libcudart.${SO_EXT}))
Makefile:  CUDA_LIB_DIR := $(patsubst %/,%,$(dir $(firstword $(CUDA_LIB_DIR))))
Makefile:CUDA_LIB_DIR_STUBS := $(CUDA_LIB_DIR)/stubs
Makefile:CUDA_BACKENDS = /gpu/cuda/ref /gpu/cuda/shared /gpu/cuda/gen
Makefile:ifneq ($(CUDA_LIB_DIR),)
Makefile:  $(libceeds) : CPPFLAGS += -I$(CUDA_DIR)/include
Makefile:  PKG_LIBS += -L$(abspath $(CUDA_LIB_DIR)) -lcudart -lnvrtc -lcuda -lcublas
Makefile:  PKG_STUBS_LIBS += -L$(CUDA_LIB_DIR_STUBS)
Makefile:  libceed.c     += interface/ceed-cuda.c
Makefile:  libceed.c     += $(cuda.c) $(cuda-ref.c) $(cuda-shared.c) $(cuda-gen.c)
Makefile:  libceed.cpp   += $(cuda.cpp) $(cuda-ref.cpp) $(cuda-gen.cpp)
Makefile:  libceed.cu    += $(cuda-ref.cu) $(cuda-shared.cu) $(cuda-gen.cu)
Makefile:  BACKENDS_MAKE += $(CUDA_BACKENDS)
Makefile:HIP_LIB_DIR := $(wildcard $(foreach d,lib lib64,$(ROCM_DIR)/$d/libamdhip64.${SO_EXT}))
Makefile:HIP_BACKENDS = /gpu/hip/ref /gpu/hip/shared /gpu/hip/gen
Makefile:  HIPCONFIG_CPPFLAGS := $(subst =,,$(shell $(ROCM_DIR)/bin/hipconfig -C))
Makefile:SYCL_BACKENDS = /gpu/sycl/ref /gpu/sycl/shared /gpu/sycl/gen
Makefile:  ifeq ($(MAGMA_ARCH), 0)  # CUDA MAGMA
Makefile:    ifneq ($(CUDA_LIB_DIR),)
Makefile:      cuda_link = $(if $(STATIC),,-Wl,-rpath,$(CUDA_LIB_DIR)) -L$(CUDA_LIB_DIR) -lcublas -lcusparse -lcudart
Makefile:      magma_link_static = -L$(MAGMA_DIR)/lib -lmagma $(cuda_link) $(omp_link)
Makefile:      $(magma.c:%.c=$(OBJDIR)/%.o) $(magma.c:%=%.tidy) : CPPFLAGS += -DADD_ -I$(MAGMA_DIR)/include -I$(CUDA_DIR)/include
Makefile:      $(magma.cpp:%.cpp=$(OBJDIR)/%.o) $(magma.cpp:%=%.tidy) : CPPFLAGS += -DADD_ -I$(MAGMA_DIR)/include -I$(CUDA_DIR)/include
Makefile:      MAGMA_BACKENDS = /gpu/cuda/magma /gpu/cuda/magma/det
Makefile:      $(magma.c:%.c=$(OBJDIR)/%.o) $(magma.c:%=%.tidy) : CPPFLAGS += $(HIPCONFIG_CPPFLAGS) -I$(MAGMA_DIR)/include -I$(ROCM_DIR)/include -DCEED_MAGMA_USE_HIP -DADD_
Makefile:      $(magma.cpp:%.cpp=$(OBJDIR)/%.o) $(magma.cpp:%=%.tidy) : CPPFLAGS += $(HIPCONFIG_CPPFLAGS) -I$(MAGMA_DIR)/include -I$(ROCM_DIR)/include -DCEED_MAGMA_USE_HIP -DADD_
Makefile:      MAGMA_BACKENDS = /gpu/hip/magma /gpu/hip/magma/det
Makefile:	  "$(includedir)/ceed/jit-source/cuda/" "$(includedir)/ceed/jit-source/hip/"\
Makefile:	$(INSTALL_DATA) include/ceed/cuda.h "$(DESTDIR)$(includedir)/ceed/"
Makefile:	$(INSTALL_DATA) $(wildcard include/ceed/jit-source/cuda/*.h) "$(DESTDIR)$(includedir)/ceed/jit-source/cuda/"
Makefile:	$(CLANG_TIDY) $(TIDY_OPTS) $^ -- $(CPPFLAGS) --std=c99 -I$(CUDA_DIR)/include -I$(ROCM_DIR)/include -DCEED_JIT_SOURCE_ROOT_DEFAULT="\"$(abspath ./include)/\""
Makefile:	$(CLANG_TIDY) $(TIDY_OPTS) $^ -- $(CPPFLAGS) --std=c++11 -I$(CUDA_DIR)/include -I$(OCCA_DIR)/include -I$(ROCM_DIR)/include
Makefile:#   make configure CC=/path/to/my/cc CUDA_DIR=/opt/cuda
Makefile:  MAGMA_DIR OCCA_DIR XSMM_DIR CUDA_DIR CUDA_ARCH MFEM_DIR PETSC_DIR NEK5K_DIR ROCM_DIR HIP_ARCH SYCL_DIR
doc/sphinx/source/intro.md:Furthermore, software packages that provide high-performance implementations have often been special-purpose and intrusive. libCEED {cite}`libceed-joss-paper` is a new library that offers a purely algebraic interface for matrix-free operator representation and supports run-time selection of implementations tuned for a variety of computational device types, including CPUs and GPUs.
doc/sphinx/source/intro.md:libCEED provides a low-level Application Programming Interface (API) for user codes so that applications with their own discretization infrastructure (e.g., those in [PETSc](https://www.mcs.anl.gov/petsc/), [MFEM](https://mfem.org/) and [Nek5000](https://nek5000.mcs.anl.gov/)) can evaluate and use the core operations provided by libCEED. GPU implementations are available via pure [CUDA](https://developer.nvidia.com/about-cuda) and pure [HIP](https://rocmdocs.amd.com) as well as the [OCCA](http://github.com/libocca/occa) and [MAGMA](https://bitbucket.org/icl/magma) libraries.
doc/sphinx/source/gpu.md:# GPU Development
doc/sphinx/source/gpu.md:Code that produces correct results with CPU backends will produce correct results on GPU backends, provided that JiT and memory access assumptions of the libCEED API are respected.
doc/sphinx/source/gpu.md:The entire contents of this file and all locally included files (`#include "foo.h"`) are used during JiT compilation for GPU backends.
doc/sphinx/source/gpu.md:These source file must only contain syntax constructs supported by C99 and all targeted backends (i.e. CUDA for `/gpu/cuda`, OpenCL/SYCL for `/gpu/sycl`, etc.).
doc/sphinx/source/gpu.md:GPU backends require stricter adherence to memory access assumptions, but CPU backends may occasionally report correct results despite violations of memory access assumptions.
doc/sphinx/source/gpu.md:Read-only access of `CeedVector` and `CeedQFunctionContext` memory spaces must be respected for proper GPU behavior.
doc/sphinx/source/releasenotes.md:- Allow user to set additional compiler options for CUDA and HIP JiT.
doc/sphinx/source/releasenotes.md:- Added Sycl backends `/gpu/sycl/ref`, `/gpu/sycl/shared`, and `/gpu/sycl/gen`.
doc/sphinx/source/releasenotes.md:- Fix bug in setting device id for GPU backends.
doc/sphinx/source/releasenotes.md:- Fix storing of indices for `CeedElemRestriction` on the host with GPU backends.
doc/sphinx/source/releasenotes.md:- Various performance enhancements, analytic matrix-free and assembled Jacobian, and PETSc solver configurations for GPUs.
doc/sphinx/source/releasenotes.md:- Refactored `/gpu/cuda/shared` and `/gpu/cuda/gen` as well as `/gpu/hip/shared` and `/gpu/hip/gen` backend to improve maintainablity and reduce duplicated code.
doc/sphinx/source/releasenotes.md:- Enabled support for `p > 8` for `/gpu/*/shared` backends.
doc/sphinx/source/releasenotes.md:- Switched MAGMA backends to use runtime compilation for tensor basis kernels (and element restriction kernels, in non-deterministic `/gpu/*/magma` backends).
doc/sphinx/source/releasenotes.md:- Install JiT source files in install directory to fix GPU functionality for installed libCEED.
doc/sphinx/source/releasenotes.md:- Added JiT utilities in `ceed/jit-tools.h` to reduce duplicated code in GPU backends.
doc/sphinx/source/releasenotes.md:- Added {c:func}`CeedQFunctionSetContextWritable` and read-only access to `CeedQFunctionContext` data as an optional feature to improve GPU performance. By default, calling the `CeedQFunctionUser` during {c:func}`CeedQFunctionApply` is assumed to write into the `CeedQFunctionContext` data, consistent with the previous behavior. Note that if a user asserts that their `CeedQFunctionUser` does not write into the `CeedQFunctionContext` data, they are responsible for the validity of this assertion.
doc/sphinx/source/releasenotes.md:- Added support for element matrix assembly in GPU backends.
doc/sphinx/source/releasenotes.md:- Refactored preconditioner support internally to facilitate future development and improve GPU completeness/test coverage.
doc/sphinx/source/releasenotes.md:- Put GPU JiTed kernel source code into separate files.
doc/sphinx/source/releasenotes.md:- Fluid mechanics example adds GPU support and improves modularity.
doc/sphinx/source/releasenotes.md:- New HIP MAGMA backends for hipMAGMA library users: `/gpu/hip/magma` and `/gpu/hip/magma/det`.
doc/sphinx/source/releasenotes.md:- New HIP backends for improved tensor basis performance: `/gpu/hip/shared` and `/gpu/hip/gen`.
doc/sphinx/source/releasenotes.md:- New HIP backend: `/gpu/hip/ref`.
doc/sphinx/source/releasenotes.md:- The `/gpu/cuda/reg` backend has been removed, with its core features moved into `/gpu/cuda/ref` and `/gpu/cuda/shared`.
doc/sphinx/source/releasenotes.md:- QFunctions using variable-length array (VLA) pointer constructs can be used with CUDA
doc/sphinx/source/releasenotes.md:- Fix some missing edge cases in CUDA backend.
doc/sphinx/source/releasenotes.md:  - {file}`examples/petsc/bpsraw.c` (formerly `bps.c`): transparent CUDA support.
doc/sphinx/source/releasenotes.md:    and transparent CUDA support.
doc/sphinx/source/releasenotes.md:For this release, several improvements were made. Two new CUDA backends were added to
doc/sphinx/source/releasenotes.md:the family of backends, of which, the new `cuda-gen` backend achieves state-of-the-art
doc/sphinx/source/releasenotes.md:| `/gpu/occa`              | CUDA OCCA kernels                                   |
doc/sphinx/source/releasenotes.md:| `/ocl/occa`              | OpenCL OCCA kernels                                 |
doc/sphinx/source/releasenotes.md:| `/gpu/cuda/ref`          | Reference pure CUDA kernels                         |
doc/sphinx/source/releasenotes.md:| `/gpu/cuda/reg`          | Pure CUDA kernels using one thread per element      |
doc/sphinx/source/releasenotes.md:| `/gpu/cuda/shared`       | Optimized pure CUDA kernels using shared memory     |
doc/sphinx/source/releasenotes.md:| `/gpu/cuda/gen`          | Optimized pure CUDA kernels using code generation   |
doc/sphinx/source/releasenotes.md:| `/gpu/magma`             | CUDA MAGMA kernels                                  |
doc/sphinx/source/releasenotes.md:four new CPU backends, two new GPU backends, CPU backend optimizations, initial
doc/sphinx/source/releasenotes.md:performance. The `/gpu/cuda/*` backends provide GPU performance strictly using CUDA.
doc/sphinx/source/releasenotes.md:The `/gpu/cuda/ref` backend is a reference CUDA backend, providing reasonable
doc/sphinx/source/releasenotes.md:performance for most problem configurations. The `/gpu/cuda/reg` backend uses a simple
doc/sphinx/source/releasenotes.md:compilation, provided by nvrtc (NVidia Runtime Compiler), and runtime parameters, this
doc/sphinx/source/releasenotes.md:backend unroll loops and map memory address to registers. The `/gpu/cuda/reg` backend
doc/sphinx/source/releasenotes.md:| `/gpu/occa`              | CUDA OCCA kernels                                   |
doc/sphinx/source/releasenotes.md:| `/ocl/occa`              | OpenCL OCCA kernels                                 |
doc/sphinx/source/releasenotes.md:| `/gpu/cuda/ref`          | Reference pure CUDA kernels                         |
doc/sphinx/source/releasenotes.md:| `/gpu/cuda/reg`          | Pure CUDA kernels using one thread per element      |
doc/sphinx/source/releasenotes.md:| `/gpu/magma`             | CUDA MAGMA kernels                                  |
doc/sphinx/source/releasenotes.md:| `/gpu/occa`             | CUDA OCCA kernels                                   |
doc/sphinx/source/releasenotes.md:| `/ocl/occa`             | OpenCL OCCA kernels                                 |
doc/sphinx/source/releasenotes.md:| `/gpu/magma`            | CUDA MAGMA kernels                                  |
doc/sphinx/source/releasenotes.md:but also add corresponding device (e.g., GPU) pointers to the data. Coherency is handled
doc/sphinx/source/releasenotes.md:| `/gpu/occa`             | CUDA OCCA kernels               |
doc/sphinx/source/releasenotes.md:| `/ocl/occa`             | OpenCL OCCA kernels             |
doc/sphinx/source/releasenotes.md:| `/gpu/magma`            | CUDA MAGMA kernels              |
doc/sphinx/source/releasenotes.md:| `/gpu/occa`             | CUDA OCCA kernels               |
doc/sphinx/source/releasenotes.md:| `/ocl/occa`             | OpenCL OCCA kernels             |
doc/sphinx/source/releasenotes.md:`/gpu/occa`, and `/omp/occa`.
doc/sphinx/source/releasenotes.md:| `/gpu/occa`             | CUDA OCCA kernels               |
doc/sphinx/source/index.md:gpu
doc/sphinx/source/libCEEDapi.md:Thus, a natural mapping of $\bm{A}$ on a parallel computer is to split the **T-vector** over MPI ranks (a non-overlapping decomposition, as is typically used for sparse matrices), and then split the rest of the vector types over computational devices (CPUs, GPUs, etc.) as indicated by the shaded regions in the diagram above.
doc/sphinx/source/libCEEDapi.md:For example, on a node with 2 CPU sockets and 4 GPUs, one may decide to use 6 MPI ranks (each using a single {ref}`Ceed` object): 2 ranks using 1 CPU socket each, and 4 using 1 GPU each.
doc/sphinx/source/libCEEDapi.md:Another choice could be to run 1 MPI rank on the whole node and use 5 {ref}`Ceed` objects: 1 managing all CPU cores on the 2 sockets and 4 managing 1 GPU each.
doc/sphinx/source/libCEEDapi.md:Our long-term vision is to include a variety of backend implementations in libCEED, ranging from reference kernels to highly optimized kernels targeting specific devices (e.g. GPUs) or specific polynomial orders.
doc/sphinx/source/libCEEDapi.md:creates a logical device `ceed` on the specified *resource*, which could also be a coprocessor such as `"/nvidia/0"`.
doc/sphinx/source/libCEEDapi.md:This is used by backends that support Just-In-Time (JIT) compilation (i.e., CUDA and OCCA) to compile for coprocessors.
doc/sphinx/source/libCEEDapi.md:For full support across all backends, these {ref}`CeedQFunction` source files must only contain constructs mutually supported by C99, C++11, and CUDA.
doc/img/tex/libCEEDBackends.tex:    node[pos=.5,align=center,color=black] {CUDA};
doc/img/tex/libCEEDBackends.tex:    % CUDA GPU
doc/img/tex/libCEEDBackends.tex:    node[pos=.5,align=center,color=black] {NVIDIA GPU};
doc/img/tex/libCEEDBackends.tex:    % ROCm GPU
doc/img/tex/libCEEDBackends.tex:    node[pos=.5,align=center,color=black] {AMD GPU};
doc/img/tex/libCEEDBackends.tex:    node[pos=.5,align=center,color=black] {Intel GPU};
doc/papers/joss/paper.bib:@misc{CUDAwebsite,
doc/papers/joss/paper.bib:  title = "CUDA",
doc/papers/joss/paper.bib:  url = "https://developer.nvidia.com/about-cuda",
doc/papers/joss/paper.bib:  url = "https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html",
doc/papers/joss/paper.bib:  url = "https://docs.nvidia.com/cuda/nvrtc/index.html",
doc/papers/joss/paper.bib:  title="{CPU-GPU-MIC} Comparision Charts",
doc/papers/joss/paper.bib:  url={https://github.com/karlrupp/cpu-gpu-mic-comparison},
doc/papers/joss/paper.md:Methods pioneered by the spectral element community [@Orszag:1980; @deville2002highorder] exploit problem structure to reduce costs to $O(1)$ storage and $O(p)$ compute per DoF, with very high utilization of modern CPUs and GPUs.
doc/papers/joss/paper.md:`libCEED` provides portable performance via run-time selection of implementations optimized for CPUs and GPUs, including support for just-in-time (JIT) compilation.
doc/papers/joss/paper.md:Additionally, a single source implementation (in vanilla C or C++) for the `CeedQFunction`s can be used on CPUs or GPUs (transparently using the @NVRTCwebsite, HIPRTC, or OCCA [@OCCAwebsite] run-time compilation features).
doc/papers/joss/paper.md:The Python interface uses CFFI, the C Foreign Function Interface [@python-cffi]. CFFI allows reuse of most C declarations and requires only a minimal adaptation of some of them. The C and Python APIs are mapped in a nearly 1:1 correspondence. For instance, a `CeedVector` object is exposed as `libceed.Vector` in Python, and supports no-copy host and GPU device interperability with Python arrays from the NumPy [@NumPy] or Numba [@Numba] packages. The interested reader can find more details on `libCEED`'s Python interface in @libceed-paper-proc-scipy-2020.
doc/papers/joss/paper.md:The Julia interface, referred to as `LibCEED.jl`, provides both a low-level interface, which is generated automatically from `libCEED`'s C header files, and a high-level interface. The high-level interface takes advantage of Julia's metaprogramming and just-in-time compilation capabilities to enable concise definition of Q-functions that work on both CPUs and GPUs, along with their composition into operators as in \autoref{fig:decomposition}.
doc/papers/joss/paper.md:GPU implementations are available via pure @CUDAwebsite and pure @HIPwebsite, as well as the OCCA [@OCCAwebsite] and MAGMA [@MAGMAwebsite] libraries. CPU implementations are available via pure C and AVX intrinsics as well as the LIBXSMM library [@LIBXSMM]. `libCEED` provides a dynamic interface such that users only need to write a single source (no need for templates/generics) and can select the desired specialized implementation at run time. Moreover, each process or thread can instantiate an arbitrary number of backends on an arbitrary number of devices.
doc/papers/joss/paper.md:The Exascale Computing Project (ECP) co-design Center for Efficient Exascale Discretization [@CEEDwebsite] has defined a suite of Benchmark Problems (BPs) to test and compare the performance of high-order finite element implementations [@Fischer2020scalability; @CEED-ECP-paper]. \autoref{fig:bp3} compares the performance of `libCEED` solving BP3 (CG iteration on a 3D Poisson problem) or CPU and GPU systems of similar (purchase/operating and energy) cost. These tests use PETSc [@PETScUserManual] for unstructured mesh management and parallel solvers with GPU-aware communication [@zhang2021petscsf]; a similar implementation with comparable performance is available through MFEM.
doc/papers/joss/paper.md:![Performance for BP3 using the \texttt{xsmm/blocked} backend on a 2-socket AMD EPYC 7452 (32-core, 2.35GHz) and the \texttt{cuda/gen} backend on LLNL's Lassen system with NVIDIA V100 GPUs. Each curve represents fixing the basis degree $p$ and varying the number of elements. The CPU enables faster solution of smaller problem sizes (as in strong scaling) while the GPU is more efficient for applications that can afford to wait for larger sizes. Note that the CPU exhibits a performance drop when the working set becomes too large for L3 cache (128 MB/socket) while no such drop exists for the GPU. (This experiment was run with release candidates of PETSc 3.14 and libCEED 0.7 using gcc-10 on EPYC and clang-10/CUDA-10 on Lassen.) \label{fig:bp3}](img/bp3-2020.pdf)
doc/papers/joss/paper.md:If MFEM is built with `libCEED` support, existing MFEM users can pass `-d ceed-cuda:/gpu/cuda/gen` to use a `libCEED` CUDA backend, and similarly for other backends.
doc/papers/joss/paper.md:The `libCEED` implementations, accessed in this way, currently provide MFEM users with the fastest operator action on CPUs and GPUs (CUDA and HIP/ROCm) without writing any `libCEED` Q-functions.
interface/ceed-qfunction.c:                           The entire source file must only contain constructs supported by all targeted backends (i.e. CUDA for `/gpu/cuda`, OpenCL/SYCL for `/gpu/sycl`, etc.).
interface/ceed-qfunction.c:                           The entire contents of this file and all locally included files are used during JiT compilation for GPU backends.
interface/ceed-qfunction.c:  Setting `is_writable == false` may offer a performance improvement on GPU backends.
interface/ceed-basis.c:  // Create TensorContract object if needed, such as a basis from the GPU backends
interface/ceed-vector.c:  // Backend impl for GPU, if added
interface/ceed-vector.c:  // Backend impl for GPU, if added
interface/ceed.c:      CEED_FTABLE_ENTRY(CeedQFunction, SetCUDAUserFunction),
interface/ceed.c:  @brief Set the GPU stream for a `Ceed` context
interface/ceed.c:  @param[in]     handle Handle to GPU stream
interface/ceed-cuda.c:#include <ceed/cuda.h>
interface/ceed-cuda.c:#include <cuda.h>
interface/ceed-cuda.c:  @brief Set CUDA function pointer to evaluate action at quadrature points
interface/ceed-cuda.c:int CeedQFunctionSetCUDAUserFunction(CeedQFunction qf, CUfunction f) {
interface/ceed-cuda.c:  if (!qf->SetCUDAUserFunction) {
interface/ceed-cuda.c:    CeedCall(qf->SetCUDAUserFunction(qf, f));
README.md:The goal of libCEED is to propose such a format, as well as supporting implementations and data structures, that enable efficient operator evaluation on a variety of computational device types (CPUs, GPUs, etc.).
README.md:To enable CUDA support, add `CUDA_DIR=/opt/cuda` or an appropriate directory to your `make` invocation.
README.md:To enable HIP support, add `ROCM_DIR=/opt/rocm` or an appropriate directory.
README.md:$ make configure CUDA_DIR=/usr/local/cuda ROCM_DIR=/opt/rocm OPT='-O3 -march=znver2'
README.md:| **CUDA Native**            |
README.md:| `/gpu/cuda/ref`            | Reference pure CUDA kernels                       | Yes                   |
README.md:| `/gpu/cuda/shared`         | Optimized pure CUDA kernels using shared memory   | Yes                   |
README.md:| `/gpu/cuda/gen`            | Optimized pure CUDA kernels using code generation | No                    |
README.md:| `/gpu/hip/ref`             | Reference pure HIP kernels                        | Yes                   |
README.md:| `/gpu/hip/shared`          | Optimized pure HIP kernels using shared memory    | Yes                   |
README.md:| `/gpu/hip/gen`             | Optimized pure HIP kernels using code generation  | No                    |
README.md:| `/gpu/sycl/ref`            | Reference pure SYCL kernels                       | Yes                   |
README.md:| `/gpu/sycl/shared`         | Optimized pure SYCL kernels using shared memory   | Yes                   |
README.md:| `/gpu/cuda/magma`          | CUDA MAGMA kernels                                | No                    |
README.md:| `/gpu/cuda/magma/det`      | CUDA MAGMA kernels                                | Yes                   |
README.md:| `/gpu/hip/magma`           | HIP MAGMA kernels                                 | No                    |
README.md:| `/gpu/hip/magma/det`       | HIP MAGMA kernels                                 | Yes                   |
README.md:| `/gpu/cuda/occa`           | OCCA backend with CUDA kernels                    | Yes                   |
README.md:| `/gpu/hip/occa`            | OCCA backend with HIP kernels                     | Yes                   |
README.md:The `/gpu/cuda/*` backends provide GPU performance strictly using CUDA.
README.md:The `/gpu/hip/*` backends provide GPU performance strictly using HIP.
README.md:They are based on the `/gpu/cuda/*` backends.
README.md:ROCm version 4.2 or newer is required.
README.md:The `/gpu/sycl/*` backends provide GPU performance strictly using SYCL.
README.md:They are based on the `/gpu/cuda/*` and `/gpu/hip/*` backends.
README.md:The `/gpu/*/magma/*` backends rely upon the [MAGMA](https://bitbucket.org/icl/magma) package.
README.md:Currently, each MAGMA library installation is only built for either CUDA or HIP.
README.md:The corresponding set of libCEED backends (`/gpu/cuda/magma/*` or `/gpu/hip/magma/*`) will automatically be built for the version of the MAGMA library found in `MAGMA_DIR`.
README.md:Users can specify a device for all CUDA, HIP, and MAGMA backends through adding `:device_id=#` after the resource name.
README.md:> - `/gpu/cuda/gen:device_id=1`
README.md:> - `"/*/occa:mode='CUDA',device_id=0"`
README.md:# libCEED examples on CPU and GPU
README.md:$ ./ex1-volume -ceed /gpu/cuda
README.md:$ ./ex2-surface -ceed /gpu/cuda
README.md:# MFEM+libCEED examples on CPU and GPU
README.md:$ ./bp3 -ceed /gpu/cuda -no-vis
README.md:# Nek5000+libCEED examples on CPU and GPU
README.md:$ ./nek-examples.sh -e bp3 -ceed /gpu/cuda -b 3
README.md:# PETSc+libCEED examples on CPU and GPU
README.md:$ ./bps -problem bp2 -ceed /gpu/cuda
README.md:$ ./bps -problem bp4 -ceed /gpu/cuda
README.md:$ ./bps -problem bp6 -ceed /gpu/cuda
README.md:$ ./bpsraw -problem bp2 -ceed /gpu/cuda
README.md:$ ./bpsraw -problem bp4 -ceed /gpu/cuda
README.md:$ ./bpsraw -problem bp6 -ceed /gpu/cuda
README.md:$ ./bpssphere -problem bp2 -ceed /gpu/cuda
README.md:$ ./bpssphere -problem bp4 -ceed /gpu/cuda
README.md:$ ./bpssphere -problem bp6 -ceed /gpu/cuda
README.md:$ ./area -problem cube -ceed /gpu/cuda -degree 3
README.md:$ ./area -problem sphere -ceed /gpu/cuda -degree 3 -dm_refine 2
README.md:$ ./navierstokes -ceed /gpu/cuda -degree 1
README.md:$ ./elasticity -ceed /gpu/cuda -mesh [.exo file] -degree 2 -E 1 -nu 0.3 -problem Linear -forcing mms
README.md:The above code assumes a GPU-capable machine with the CUDA backends enabled.
include/ceed-impl.h:  int (*SetCUDAUserFunction)(CeedQFunction, void *);
include/ceed/jit-source/sycl/sycl-types.h:#ifdef __OPENCL_C_VERSION__
include/ceed/jit-source/sycl/sycl-gen-templates.h:#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
include/ceed/jit-source/sycl/sycl-gen-templates.h:#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
include/ceed/jit-source/hip/hip-ref-basis-tensor-at-points.h:/// Internal header for CUDA tensor product basis with AtPoints evaluation
include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:/// Internal header for CUDA shared memory basis read/write templates
include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void ReadElementStrided1d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void WriteElementStrided1d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void SumElementStrided1d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void ReadElementStrided2d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void WriteElementStrided2d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void SumElementStrided2d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void ReadElementStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void WriteElementStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
include/ceed/jit-source/cuda/cuda-shared-basis-read-write-templates.h:inline __device__ void SumElementStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedInt strides_node, const CeedInt strides_comp,
include/ceed/jit-source/cuda/cuda-ref-basis-tensor.h:/// Internal header for CUDA tensor product basis
include/ceed/jit-source/cuda/cuda-ref-basis-tensor-at-points.h:/// Internal header for CUDA tensor product basis with AtPoints evaluation
include/ceed/jit-source/cuda/cuda-ref-operator-assemble.h:/// Internal header for CUDA operator full assembly
include/ceed/jit-source/cuda/cuda-ref-basis-nontensor.h:/// Internal header for CUDA non-tensor product basis
include/ceed/jit-source/cuda/cuda-ref-basis-nontensor.h:#include "cuda-ref-basis-nontensor-templates.h"
include/ceed/jit-source/cuda/cuda-gen-templates.h:/// Internal header for CUDA backend macro and type definitions for JiT source
include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void loadMatrix(SharedData_Cuda &data, const CeedScalar *__restrict__ d_B, CeedScalar *B) {
include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void readDofsOffset1d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void readDofsStrided1d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *__restrict__ d_u,
include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void writeDofsOffset1d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void writeDofsStrided1d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *__restrict__ r_v,
include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void readDofsOffset2d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void readDofsStrided2d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *__restrict__ d_u,
include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void writeDofsOffset2d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void writeDofsStrided2d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *__restrict__ r_v,
include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void readDofsOffset3d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void readDofsStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *__restrict__ d_u,
include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void readSliceQuadsOffset3d(SharedData_Cuda &data, const CeedInt nquads, const CeedInt elem, const CeedInt q,
include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void readSliceQuadsStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedInt q, const CeedScalar *__restrict__ d_u,
include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void writeDofsOffset3d(SharedData_Cuda &data, const CeedInt num_nodes, const CeedInt elem, const CeedInt *__restrict__ indices,
include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void writeDofsStrided3d(SharedData_Cuda &data, const CeedInt elem, const CeedScalar *__restrict__ r_v,
include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void gradCollo3d(SharedData_Cuda &data, const CeedInt q, const CeedScalar *__restrict__ r_U, const CeedScalar *c_G,
include/ceed/jit-source/cuda/cuda-gen-templates.h:inline __device__ void gradColloTranspose3d(SharedData_Cuda &data, const CeedInt q, const CeedScalar *__restrict__ r_U, const CeedScalar *c_G,
include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:/// Internal header for CUDA shared memory tensor product basis
include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:#include "cuda-shared-basis-read-write-templates.h"
include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:#include "cuda-shared-basis-tensor-templates.h"
include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:  SharedData_Cuda data;
include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:  SharedData_Cuda data;
include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:  SharedData_Cuda data;
include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:  SharedData_Cuda data;
include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:  SharedData_Cuda data;
include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:  SharedData_Cuda data;
include/ceed/jit-source/cuda/cuda-shared-basis-tensor.h:  SharedData_Cuda data;
include/ceed/jit-source/cuda/cuda-ref-operator-assemble-diagonal.h:/// Internal header for CUDA operator diagonal assembly
include/ceed/jit-source/cuda/cuda-ref-restriction-offset.h:/// Internal header for CUDA offset element restriction kernels
include/ceed/jit-source/cuda/cuda-ref-qfunction.h:/// Internal header for CUDA backend QFunction read/write kernels
include/ceed/jit-source/cuda/cuda-types.h:/// Internal header for CUDA type definitions
include/ceed/jit-source/cuda/cuda-types.h:#define CEED_CUDA_NUMBER_FIELDS 16
include/ceed/jit-source/cuda/cuda-types.h:  const CeedScalar *inputs[CEED_CUDA_NUMBER_FIELDS];
include/ceed/jit-source/cuda/cuda-types.h:  CeedScalar       *outputs[CEED_CUDA_NUMBER_FIELDS];
include/ceed/jit-source/cuda/cuda-types.h:} Fields_Cuda;
include/ceed/jit-source/cuda/cuda-types.h:  CeedInt *inputs[CEED_CUDA_NUMBER_FIELDS];
include/ceed/jit-source/cuda/cuda-types.h:  CeedInt *outputs[CEED_CUDA_NUMBER_FIELDS];
include/ceed/jit-source/cuda/cuda-types.h:} FieldsInt_Cuda;
include/ceed/jit-source/cuda/cuda-types.h:} SharedData_Cuda;
include/ceed/jit-source/cuda/cuda-ref-restriction-strided.h:/// Internal header for CUDA strided element restriction kernels
include/ceed/jit-source/cuda/cuda-ref-restriction-curl-oriented.h:/// Internal header for CUDA curl-oriented element restriction kernels
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:/// Internal header for CUDA shared memory tensor product basis templates
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractX1d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeX1d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void Interp1d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, CeedScalar *__restrict__ r_V) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void InterpTranspose1d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void Grad1d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void GradTranspose1d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void Weight1d(SharedData_Cuda &data, const CeedScalar *__restrict__ q_weight_1d, CeedScalar *w) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractX2d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractY2d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeY2d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeX2d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeAddX2d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void InterpTensor2d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void InterpTransposeTensor2d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void GradTensor2d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void GradTransposeTensor2d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void WeightTensor2d(SharedData_Cuda &data, const CeedScalar *__restrict__ q_weight_1d, CeedScalar *w) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractX3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractY3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractZ3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeZ3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeY3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeAddY3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeX3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void ContractTransposeAddX3d(SharedData_Cuda &data, const CeedScalar *U, const CeedScalar *B, CeedScalar *V) {
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void InterpTensor3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void InterpTransposeTensor3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void GradTensor3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void GradTransposeTensor3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void GradTensorCollocated3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B, const CeedScalar *c_G,
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void GradTransposeTensorCollocated3d(SharedData_Cuda &data, const CeedScalar *__restrict__ r_U, const CeedScalar *c_B,
include/ceed/jit-source/cuda/cuda-shared-basis-tensor-templates.h:inline __device__ void WeightTensor3d(SharedData_Cuda &data, const CeedScalar *__restrict__ q_weight_1d, CeedScalar *w) {
include/ceed/jit-source/cuda/cuda-ref-restriction-oriented.h:/// Internal header for CUDA oriented element restriction kernels
include/ceed/jit-source/cuda/cuda-ref-restriction-at-points.h:/// Internal header for CUDA offset element restriction kernels
include/ceed/jit-source/cuda/cuda-atomic-add-fallback.h:/// Internal header for CUDA atomic add fallback definition
include/ceed/jit-source/cuda/cuda-atomic-add-fallback.h:// Atomic add, for older CUDA
include/ceed/jit-source/cuda/cuda-ref-basis-nontensor-templates.h:/// Internal header for CUDA non-tensor product basis templates
include/ceed/jit-source/cuda/cuda-jit.h:/// Internal header for CUDA backend macro and type definitions for JiT source
include/ceed/jit-source/cuda/cuda-jit.h:#include "cuda-types.h"
include/ceed/types.h:    VLA is a C99 feature that is not supported by the C++ dialect used by CUDA.
include/ceed/types.h:    This macro allows users to use the VLA syntax with the CUDA backends.
include/ceed/cuda.h:/// Public header for CUDA utility components of libCEED
include/ceed/cuda.h:#include <cuda.h>
include/ceed/cuda.h:CEED_EXTERN int CeedQFunctionSetCUDAUserFunction(CeedQFunction qf, CUfunction f);
julia/LibCEED.jl/gen/generator.jl:    header_files = ["ceed.h", "ceed/cuda.h", "ceed/backend.h"]
julia/LibCEED.jl/docs/src/LibCEED.md:bundled as a pre-built binary. In order to access more advanced features (CUDA
julia/LibCEED.jl/docs/src/UserQFunctions.md:natively in Julia. These user Q-functions work with both the CPU and CUDA
julia/LibCEED.jl/docs/src/UserQFunctions.md:## GPU Kernels
julia/LibCEED.jl/docs/src/UserQFunctions.md:If the `Ceed` resource uses a CUDA backend, then the user Q-functions defined
julia/LibCEED.jl/docs/src/UserQFunctions.md:using [`@interior_qf`](@ref) are automatically compiled as CUDA kernels using
julia/LibCEED.jl/docs/src/UserQFunctions.md:[`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl). Some Julia features are not
julia/LibCEED.jl/docs/src/UserQFunctions.md:available in GPU code (for example, dynamic dispatch), so if the Q-function is
julia/LibCEED.jl/docs/src/UserQFunctions.md:intended to be run on the GPU, the user should take care when defining the body
julia/LibCEED.jl/docs/src/Ceed.md:iscuda
julia/LibCEED.jl/docs/src/index.md:CUDA/GPU support, specific compiler flags, etc.) then you should compile your
julia/LibCEED.jl/docs/src/index.md:!!! warning "The pre-built libCEED binaries do not support CUDA backends"
julia/LibCEED.jl/docs/src/index.md:    are not built with CUDA support. If you want to run libCEED on the GPU, you
julia/LibCEED.jl/docs/src/index.md:Q-functions that automatically work on the GPU. See the [related
julia/LibCEED.jl/test/runtests.jl:            @test !iscuda(c)
julia/LibCEED.jl/README.md:If you require features of a specific build of libCEED (e.g. CUDA/GPU support, specific compiler flags, etc.) then you should compile your own version of the libCEED library, and configure LibCEED.jl to use this binary as described in the [Configuring LibCEED.jl](#configuring-libceedjl) section.
julia/LibCEED.jl/README.md:**Warning:** the pre-built libCEED binaries do not support CUDA backends
julia/LibCEED.jl/README.md:The pre-built binaries automatically installed by LibCEED.jl (through the [libCEED_jll](https://juliahub.com/ui/Packages/libCEED_jll/LB2fn) package) are not built with CUDA support.
julia/LibCEED.jl/README.md:If you want to run libCEED on the GPU, you will have to build libCEED from source and configure LibCEED.jl as described in the [Configuring LibCEED.jl](#configuring-libceedjl) section.
julia/LibCEED.jl/src/LibCEED.jl:    iscuda,
julia/LibCEED.jl/src/LibCEED.jl:    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("Cuda.jl")
julia/LibCEED.jl/src/UserQFunction.jl:    if iscuda(ceed)
julia/LibCEED.jl/src/UserQFunction.jl:        getresource(ceed) == "/gpu/cuda/gen" && error(
julia/LibCEED.jl/src/UserQFunction.jl:                "/gpu/cuda/gen is not compatible with user Q-functions defined with ",
julia/LibCEED.jl/src/UserQFunction.jl:                "libCEED.jl.\nPlease use a different backend, for example: /gpu/cuda/shared ",
julia/LibCEED.jl/src/UserQFunction.jl:                "or /gpu/cuda/ref",
julia/LibCEED.jl/src/UserQFunction.jl:        if isdefined(@__MODULE__, :CUDA)
julia/LibCEED.jl/src/UserQFunction.jl:            !has_cuda() && error("No valid CUDA installation found")
julia/LibCEED.jl/src/UserQFunction.jl:                    "User Q-functions with CUDA backends require the CUDA.jl package to be ",
julia/LibCEED.jl/src/UserQFunction.jl:                    "Please ensure that the CUDA.jl package is installed and loaded.",
julia/LibCEED.jl/src/QFunction.jl:        C.CeedQFunctionSetCUDAUserFunction(ref[], f.cuf)
julia/LibCEED.jl/src/QFunction.jl:    elseif iscuda(c) && !isdefined(@__MODULE__, :CUDA)
julia/LibCEED.jl/src/QFunction.jl:                "In order to use user Q-functions with a CUDA backend, the CUDA.jl package ",
julia/LibCEED.jl/src/Cuda.jl:using .CUDA, Cassette
julia/LibCEED.jl/src/Cuda.jl:const cudafuns = (
julia/LibCEED.jl/src/Cuda.jl:Cassette.@context CeedCudaContext
julia/LibCEED.jl/src/Cuda.jl:@inline function Cassette.overdub(::CeedCudaContext, ::typeof(Core.kwfunc), f)
julia/LibCEED.jl/src/Cuda.jl:@inline function Cassette.overdub(::CeedCudaContext, ::typeof(Core.apply_type), args...)
julia/LibCEED.jl/src/Cuda.jl:    ::CeedCudaContext,
julia/LibCEED.jl/src/Cuda.jl:for f in cudafuns
julia/LibCEED.jl/src/Cuda.jl:        ::CeedCudaContext,
julia/LibCEED.jl/src/Cuda.jl:        return CUDA.$f(x)
julia/LibCEED.jl/src/Cuda.jl:struct FieldsCuda
julia/LibCEED.jl/src/Cuda.jl:            # Cassette context for replacing intrinsics with CUDA versions
julia/LibCEED.jl/src/Cuda.jl:            ctx = LibCEED.CeedCudaContext()
julia/LibCEED.jl/src/Cuda.jl:    tt = Tuple{Ptr{Nothing},Int32,FieldsCuda}
julia/LibCEED.jl/src/generated/libceed_bindings.jl:function CeedQFunctionSetCUDAUserFunction(qf, f)
julia/LibCEED.jl/src/generated/libceed_bindings.jl:    ccall((:CeedQFunctionSetCUDAUserFunction, libceed), Cint, (CeedQFunction, Cint), qf, f)
julia/LibCEED.jl/src/Ceed.jl:    iscuda(c::Ceed)
julia/LibCEED.jl/src/Ceed.jl:Returns true if the given [`Ceed`](@ref) object has resource `"/gpu/cuda/*"` and false
julia/LibCEED.jl/src/Ceed.jl:function iscuda(c::Ceed)
julia/LibCEED.jl/src/Ceed.jl:    length(res_split) >= 3 && res_split[3] == "cuda"
examples/petsc/bpsraw.c://     ./bpsraw -problem bp6 -ceed /gpu/cuda
examples/petsc/bpsraw.c:      if (strstr(resolved, "/gpu/cuda")) default_vec_type = VECCUDA;
examples/petsc/bpsraw.c:      else if (strstr(resolved, "/gpu/hip/occa")) default_vec_type = VECSTANDARD;  // https://github.com/CEED/libCEED/issues/678
examples/petsc/bpsraw.c:      else if (strstr(resolved, "/gpu/hip")) default_vec_type = VECHIP;
examples/petsc/bpssphere.c://     bpssphere -problem bp6 -degree 3 -ceed /gpu/cuda
examples/petsc/bpssphere.c:        if (strstr(resolved, "/gpu/cuda")) vec_type = VECCUDA;
examples/petsc/bpssphere.c:        else if (strstr(resolved, "/gpu/hip/occa")) vec_type = VECSTANDARD;  // https://github.com/CEED/libCEED/issues/678
examples/petsc/bpssphere.c:        else if (strstr(resolved, "/gpu/hip")) vec_type = VECHIP;
examples/petsc/bpsswarm.c:        if (strstr(resolved, "/gpu/cuda")) vec_type = VECCUDA;
examples/petsc/bpsswarm.c:        else if (strstr(resolved, "/gpu/hip/occa")) vec_type = VECSTANDARD;  // https://github.com/CEED/libCEED/issues/678
examples/petsc/bpsswarm.c:        else if (strstr(resolved, "/gpu/hip")) vec_type = VECHIP;
examples/petsc/area.c:        if (strstr(resolved, "/gpu/cuda")) vec_type = VECCUDA;
examples/petsc/area.c:        else if (strstr(resolved, "/gpu/hip/occa")) vec_type = VECSTANDARD;  // https://github.com/CEED/libCEED/issues/678
examples/petsc/area.c:        else if (strstr(resolved, "/gpu/hip")) vec_type = VECHIP;
examples/petsc/multigrid.c://     multigrid -problem bp6 -ceed /gpu/cuda
examples/petsc/multigrid.c:      if (strstr(resolved, "/gpu/cuda")) vec_type = VECCUDA;
examples/petsc/multigrid.c:      else if (strstr(resolved, "/gpu/hip/occa")) vec_type = VECSTANDARD;  // https://github.com/CEED/libCEED/issues/678
examples/petsc/multigrid.c:      else if (strstr(resolved, "/gpu/hip")) vec_type = VECHIP;
examples/petsc/bps.c://     ./bps -problem bp6 -degree 3 -ceed /gpu/cuda
examples/petsc/bps.c:        if (strstr(resolved, "/gpu/cuda")) vec_type = VECCUDA;
examples/petsc/bps.c:        else if (strstr(resolved, "/gpu/hip/occa")) vec_type = VECSTANDARD;  // https://github.com/CEED/libCEED/issues/678
examples/petsc/bps.c:        else if (strstr(resolved, "/gpu/hip")) vec_type = VECHIP;
examples/solids/elasticity.c://     -ceed /gpu/cuda
examples/solids/elasticity.c:      if (strstr(resolved, "/gpu/cuda")) vectype = VECCUDA;
examples/solids/elasticity.c:      else if (strstr(resolved, "/gpu/hip")) vectype = VECHIP;
examples/mfem/bp3.cpp://     ./bp3 -ceed /gpu/cuda
examples/mfem/bp1.cpp://     ./bp1 -ceed /gpu/cuda
examples/fluids/pytorch_pkgconfig.py:if torch.cuda.is_available():
examples/fluids/pytorch_pkgconfig.py:    keywords['Libs'] += '-lc10_cuda -ltorch_cuda '
examples/fluids/pytorch_pkgconfig.py:    # Need to force linking with libtorch_cuda.so, so find path and specify linking flag to force it
examples/fluids/pytorch_pkgconfig.py:        torch_cuda_path = Path(lib_path) / 'libtorch_cuda.so'
examples/fluids/pytorch_pkgconfig.py:        if torch_cuda_path.exists():
examples/fluids/pytorch_pkgconfig.py:            variables['torch_cuda_path'] = torch_cuda_path.as_posix()
examples/fluids/pytorch_pkgconfig.py:            keywords['Libs'] += f'-Wl,--no-as-needed,"{torch_cuda_path.as_posix()}" '
examples/fluids/navierstokes.c://     ./navierstokes -ceed /gpu/cuda -problem advection -degree 1
examples/fluids/navierstokes.c:    if (strstr(resource, "/gpu/sycl")) {
examples/fluids/navierstokes.c:      if (strstr(resolved, "/gpu/cuda")) vec_type = VECCUDA;
examples/fluids/navierstokes.c:      else if (strstr(resolved, "/gpu/hip")) vec_type = VECKOKKOS;
examples/fluids/navierstokes.c:      else if (strstr(resolved, "/gpu/sycl")) vec_type = VECKOKKOS;
examples/fluids/navierstokes.c:  if (strstr(vec_type, VECCUDA)) mat_type = MATAIJCUSPARSE;
examples/fluids/navierstokes.c:  //    We use this for the main simulation DM because the reference DMPlexInsertBoundaryValues() is very slow on the GPU due to extra device-to-host
examples/fluids/src/mat-ceed.c:    if (strstr(vec_type, VECCUDA)) coo_mat_type = MATAIJCUSPARSE;
examples/fluids/src/mat-ceed.c:    PetscCall(PetscLogGpuTimeBegin());
examples/fluids/src/mat-ceed.c:    PetscCall(PetscLogGpuTimeEnd());
examples/fluids/src/mat-ceed.c:  if (PetscMemTypeDevice(ctx->mem_type)) PetscCall(PetscLogGpuFlops(ctx->flops_mult));
examples/fluids/src/mat-ceed.c:    PetscCall(PetscLogGpuTimeBegin());
examples/fluids/src/mat-ceed.c:    PetscCall(PetscLogGpuTimeEnd());
examples/fluids/src/mat-ceed.c:  if (PetscMemTypeDevice(ctx->mem_type)) PetscCall(PetscLogGpuFlops(ctx->flops_mult_transpose));
examples/fluids/src/cloptions.c:  // If we request a GPU, make sure PETSc has initialized its device (which is
examples/fluids/src/cloptions.c:  if (strncmp(app_ctx->ceed_resource, "/gpu", 4) == 0) PetscCall(PetscDeviceInitialize(PETSC_DEVICE_DEFAULT()));
examples/fluids/src/setupts.c:  PetscCall(PetscLogGpuTimeBegin());
examples/fluids/src/setupts.c:  PetscCall(PetscLogGpuTimeEnd());
examples/fluids/src/petsc_ops.c:  PetscCall(PetscLogGpuTimeBegin());
examples/fluids/src/petsc_ops.c:  PetscCall(PetscLogGpuTimeEnd());
examples/ceed/ex1-volume.c://     ./ex1-volume -ceed /gpu/cuda
examples/ceed/ex2-surface.c://     ./ex2-surface -ceed /gpu/cuda
.gitlab-ci.yml:  - test:gpu-and-float
.gitlab-ci.yml:    - echo "-------------- nproc ---------------" && NPROC_CPU=$(nproc) && NPROC_GPU=$(($(nproc)<8?$(nproc):8)) && echo "NPROC_CPU" $NPROC_CPU && echo "NPROC_GPU" $NPROC_GPU
.gitlab-ci.yml:    - echo "-------------- nproc ---------------" && NPROC_CPU=$(nproc) && NPROC_GPU=$(($(nproc)<8?$(nproc):8)) && echo "NPROC_CPU" $NPROC_CPU && echo "NPROC_GPU" $NPROC_GPU
.gitlab-ci.yml:    - cd .. && export OCCA_VERSION=occa-1.6.0 && { [[ -d $OCCA_VERSION ]] || { git clone --depth 1 --branch v1.6.0 https://github.com/libocca/occa.git $OCCA_VERSION && cd $OCCA_VERSION && export ENABLE_OPENCL="OFF" ENABLE_DPCPP="OFF" ENABLE_HIP="OFF" ENABLE_CUDA="OFF" && ./configure-cmake.sh && cmake --build build --parallel $NPROC_CPU && cmake --install build && cd ..; }; } && export OCCA_DIR=$PWD/$OCCA_VERSION/install && cd libCEED
.gitlab-ci.yml:  stage: test:gpu-and-float
.gitlab-ci.yml:    - echo "-------------- nproc ---------------" && NPROC_CPU=$(nproc) && NPROC_GPU=$(($(nproc)<8?$(nproc):8)) && echo "NPROC_CPU" $NPROC_CPU && echo "NPROC_GPU" $NPROC_GPU
.gitlab-ci.yml:# CUDA backends
.gitlab-ci.yml:noether-cuda:
.gitlab-ci.yml:  stage: test:gpu-and-float
.gitlab-ci.yml:    - cuda
.gitlab-ci.yml:    - echo "-------------- nproc ---------------" && NPROC_CPU=$(nproc) && NPROC_GPU=$(($(nproc)<8?$(nproc):8)) && echo "NPROC_CPU" $NPROC_CPU && echo "NPROC_GPU" $NPROC_GPU
.gitlab-ci.yml:    - make configure OPT='-O -march=native -ffp-contract=fast' CUDA_DIR=/usr
.gitlab-ci.yml:    - BACKENDS_GPU=$(make info-backends | grep -o '/gpu[^ ]*' | tr '\n' ' ')
.gitlab-ci.yml:    - echo "-------------- BACKENDS_GPU --------" && echo $BACKENDS_GPU
.gitlab-ci.yml:    - make -k -j$((NPROC_GPU / NPROC_POOL)) BACKENDS="$BACKENDS_GPU" JUNIT_BATCH="cuda" junit realsearch=%
.gitlab-ci.yml:# -- PETSc with CUDA (minimal)
.gitlab-ci.yml:    - export PETSC_DIR=/projects/petsc PETSC_ARCH=mpich-cuda-O PETSC_OPTIONS='-use_gpu_aware_mpi 0' && git -C $PETSC_DIR -c safe.directory=$PETSC_DIR describe
.gitlab-ci.yml:    - make -k -j$((NPROC_GPU / NPROC_POOL)) BACKENDS="$BACKENDS_GPU" JUNIT_BATCH="cuda" junit search="petsc fluids solids"
.gitlab-ci.yml:    - make -k -j$((NPROC_GPU / NPROC_POOL)) BACKENDS="$BACKENDS_GPU" JUNIT_BATCH="cuda" junit search=mfem
.gitlab-ci.yml:    - make -k -j$NPROC_GPU BACKENDS="$BACKENDS_GPU" JUNIT_BATCH="cuda" junit search=nek NEK5K_DIR=$NEK5K_DIR
.gitlab-ci.yml:    - make -k -j$((NPROC_GPU / NPROC_POOL)) BACKENDS="$BACKENDS_GPU" JUNIT_BATCH="cuda" junit search=dealii DEAL_II_DIR=$DEAL_II_DIR
.gitlab-ci.yml:# ROCm backends
.gitlab-ci.yml:noether-rocm:
.gitlab-ci.yml:  stage: test:gpu-and-float
.gitlab-ci.yml:    - rocm
.gitlab-ci.yml:    - echo "-------------- nproc ---------------" && NPROC_CPU=$(nproc) && NPROC_GPU=$(($(nproc)<8?$(nproc):8)) && echo "NPROC_CPU" $NPROC_CPU && echo "NPROC_GPU" $NPROC_GPU
.gitlab-ci.yml:    - make configure ROCM_DIR=/opt/rocm-6.1.0 OPT='-O -march=native -ffp-contract=fast'
.gitlab-ci.yml:    - BACKENDS_CPU=$(make info-backends-all | grep -o '/cpu[^ ]*' | tr '\n' ' ') && BACKENDS_GPU=$(make info-backends | grep -o '/gpu[^ ]*' | tr '\n' ' ')
.gitlab-ci.yml:    - echo "-------------- BACKENDS_GPU --------" && echo $BACKENDS_GPU
.gitlab-ci.yml:    - make -k -j$((NPROC_GPU / NPROC_POOL)) BACKENDS="$BACKENDS_GPU" JUNIT_BATCH="hip" junit realsearch=%
.gitlab-ci.yml:    - make -k -j$((NPROC_GPU / NPROC_POOL)) BACKENDS="$BACKENDS_GPU" JUNIT_BATCH="hip" junit search="petsc fluids solids"
.gitlab-ci.yml:# CPU + ROCm backends with CeedScalar == float (32 bit)
.gitlab-ci.yml:  stage: test:gpu-and-float
.gitlab-ci.yml:    - rocm
.gitlab-ci.yml:    - echo "-------------- nproc ---------------" && NPROC_CPU=$(nproc) && NPROC_GPU=$(($(nproc)<8?$(nproc):8)) && echo "NPROC_CPU" $NPROC_CPU && echo "NPROC_GPU" $NPROC_GPU
.gitlab-ci.yml:    - make configure ROCM_DIR=/opt/rocm-6.1.0 OPT='-O -march=native -ffp-contract=fast'
.gitlab-ci.yml:    - BACKENDS_CPU=$(make info-backends-all | grep -o '/cpu[^ ]*' | tr '\n' ' ') && BACKENDS_GPU=$(make info-backends | grep -o '/gpu[^ ]*' | tr '\n' ' ')
.gitlab-ci.yml:    - echo "-------------- BACKENDS_GPU --------" && echo $BACKENDS_GPU
.gitlab-ci.yml:    - make -k -j$((NPROC_GPU / NPROC_POOL)) BACKENDS="$BACKENDS_GPU" JUNIT_BATCH="float-hip" junit realsearch=%

```
