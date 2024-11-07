# https://github.com/NLESC-JCER/EigenCuda

```console
.zenodo.json:        "GPU",
.zenodo.json:    "title": "Eigencuda"
CHANGELOG.md:  - Split the memory management (`CudaMatrix`) from the [CUBLAS](https://docs.nvidia.com/cuda/cublas/index.html) invocation (`CudaPipeline`)
CHANGELOG.md:  - Moved all the allocation to the smart pointers inside `CudaMatrix`
CHANGELOG.md: - Smart pointers to handle cuda resources
CHANGELOG.md: - New CudaMatrix class
CHANGELOG.md: - Check available memory in the GPU before computing
CHANGELOG.md: - Tensor matrix multiplacation using [gemmbatched](https://docs.nvidia.com/cuda/CUBLAS/index.html#CUBLAS-lt-t-gt-gemmbatched).
CHANGELOG.md: - [Async calls](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79) to memory copies.
CHANGELOG.md: - Use a template function to perform matrix matrix multiplacation using [CUBLAS](https://docs.nvidia.com/cuda/CUBLAS/index.html).
CHANGELOG.md: - Use either *pinned* (**default**) or *pageable* memory, see [cuda optimizations](https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/).
CITATION.cff:abstract: "Offload Eigen matrix-matrix multiplications to an Nvidia GPU"
CITATION.cff:  - GPU
CITATION.cff:title: "Eigencuda"
README.md:# EigenCuda
README.md:Offload the [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) matrix matrix multiplication to an Nvidia GPU
README.md:using [cublas](https://docs.nvidia.com/cuda/cublas/index.html).
README.md:  * [Cudatoolkit](https://anaconda.org/anaconda/cudatoolkit)
README.md:#include "eigencuda.hpp"
README.md:#include "cudapipeline.hpp"
README.md:using eigencuda::CudaPipeline;
README.md:using eigencuda::CudaMatrix;
README.md:  // Call the class to handle GPU resources
README.md:  CudaPipeline cuda_pip;
README.md:CudaMatrix cuma_A{A, cuda_pip.get_stream()};
README.md:CudaMatrix cuma_B{3, 2, cuda_pip.get_stream()};
README.md:CudaMatrix cuma_C{3, 2, cuda_pip.get_stream()};
README.md:  cuma_B.copy_to_gpu(tensor[i]);
README.md:  cuda_pip.gemm(cuma_B, cuma_A, cuma_C);
include/cudamatrix.hpp:#ifndef EIGENCUDA_H_
include/cudamatrix.hpp:#define EIGENCUDA_H_
include/cudamatrix.hpp: * \brief Perform Tensor-matrix multiplications in a GPU
include/cudamatrix.hpp: * The `CudaPipeline` class handles the allocation and deallocation of arrays on
include/cudamatrix.hpp: * the GPU.
include/cudamatrix.hpp:namespace eigencuda {
include/cudamatrix.hpp:cudaError_t checkCuda(cudaError_t result);
include/cudamatrix.hpp:Index count_available_gpus();
include/cudamatrix.hpp:class CudaMatrix {
include/cudamatrix.hpp:  CudaMatrix(const Eigen::MatrixXd &matrix, const cudaStream_t &stream);
include/cudamatrix.hpp:  // Allocate memory in the GPU for a matrix
include/cudamatrix.hpp:  CudaMatrix(Index nrows, Index ncols, const cudaStream_t &stream);
include/cudamatrix.hpp:  // Convert A Cudamatrix to an EigenMatrix
include/cudamatrix.hpp:  void copy_to_gpu(const Eigen::MatrixXd &A);
include/cudamatrix.hpp:  using Unique_ptr_to_GPU_data = std::unique_ptr<double, void (*)(double *)>;
include/cudamatrix.hpp:  Unique_ptr_to_GPU_data alloc_matrix_in_gpu(size_t size_arr) const;
include/cudamatrix.hpp:  void throw_if_not_enough_memory_in_gpu(size_t requested_memory) const;
include/cudamatrix.hpp:  Unique_ptr_to_GPU_data _data{nullptr,
include/cudamatrix.hpp:                               [](double *x) { checkCuda(cudaFree(x)); }};
include/cudamatrix.hpp:  cudaStream_t _stream = nullptr;
include/cudamatrix.hpp:}  // namespace eigencuda
include/cudamatrix.hpp:#endif  // EIGENCUDA_H_
include/cudapipeline.hpp:#ifndef CUDA_PIPELINE__H
include/cudapipeline.hpp:#define CUDA_PIPELINE__H
include/cudapipeline.hpp:#include "cudamatrix.hpp"
include/cudapipeline.hpp: * \brief Perform Tensor-matrix multiplications in a GPU
include/cudapipeline.hpp: * The `CudaPipeline` class handles the allocation and deallocation of arrays on
include/cudapipeline.hpp: * the GPU.
include/cudapipeline.hpp:namespace eigencuda {
include/cudapipeline.hpp:/* \brief The CudaPipeline class offload Eigen operations to an *Nvidia* GPU
include/cudapipeline.hpp: * using the CUDA language. The Cublas handle is the context manager for all the
include/cudapipeline.hpp: * operations executed in the Nvidia device.
include/cudapipeline.hpp:class CudaPipeline {
include/cudapipeline.hpp:  CudaPipeline() {
include/cudapipeline.hpp:    cudaStreamCreate(&_stream);
include/cudapipeline.hpp:  ~CudaPipeline();
include/cudapipeline.hpp:  CudaPipeline(const CudaPipeline &) = delete;
include/cudapipeline.hpp:  CudaPipeline &operator=(const CudaPipeline &) = delete;
include/cudapipeline.hpp:  void gemm(const CudaMatrix &A, const CudaMatrix &B, CudaMatrix &C) const;
include/cudapipeline.hpp:  const cudaStream_t &get_stream() const { return _stream; };
include/cudapipeline.hpp:  cudaStream_t _stream;
include/cudapipeline.hpp:}  // namespace eigencuda
CMakeLists.txt:project(EigenCuda LANGUAGES CXX)
CMakeLists.txt:# Search for Cuda
CMakeLists.txt:find_package(CUDA REQUIRED)
src/cudapipeline.cc:#include "cudapipeline.hpp"
src/cudapipeline.cc:namespace eigencuda {
src/cudapipeline.cc:  CudaPipeline::~CudaPipeline() {
src/cudapipeline.cc:  cudaStreamDestroy(_stream);
src/cudapipeline.cc:void CudaPipeline::gemm(const CudaMatrix &A, const CudaMatrix &B,
src/cudapipeline.cc:                        CudaMatrix &C) const {
src/cudapipeline.cc:}  // namespace eigencuda
src/tests/test_dot.cc:#define BOOST_TEST_MODULE eigen_cuda
src/tests/test_dot.cc:#include "cudamatrix.hpp"
src/tests/test_dot.cc:#include "cudapipeline.hpp"
src/tests/test_dot.cc:using eigencuda::CudaMatrix;
src/tests/test_dot.cc:using eigencuda::CudaPipeline;
src/tests/test_dot.cc:using eigencuda::Index;
src/tests/test_dot.cc:BOOST_AUTO_TEST_CASE(create_cudamatrix) {
src/tests/test_dot.cc:  // Call the class to handle GPU resources
src/tests/test_dot.cc:  CudaPipeline cp;
src/tests/test_dot.cc:  // Call matrix multiplication GPU
src/tests/test_dot.cc:  // Copy matrix back and for to the GPU
src/tests/test_dot.cc:  CudaMatrix cumatrix{B, cp.get_stream()};
src/tests/test_dot.cc:  CudaPipeline cuda_pip;
src/tests/test_dot.cc:  CudaMatrix cuma_A{A, cuda_pip.get_stream()};
src/tests/test_dot.cc:  CudaMatrix cuma_B{B, cuda_pip.get_stream()};
src/tests/test_dot.cc:  CudaMatrix cuma_C{dim, dim, cuda_pip.get_stream()};
src/tests/test_dot.cc:  cuda_pip.gemm(cuma_A, cuma_B, cuma_C);
src/tests/test_dot.cc:  // Call the class to handle GPU resources
src/tests/test_dot.cc:  CudaPipeline cuda_pip;
src/tests/test_dot.cc:  // Call matrix multiplication GPU
src/tests/test_dot.cc:  CudaMatrix cuma_A{A, cuda_pip.get_stream()};
src/tests/test_dot.cc:  CudaMatrix cuma_B{3, 2, cuda_pip.get_stream()};
src/tests/test_dot.cc:  CudaMatrix cuma_C{3, 2, cuda_pip.get_stream()};
src/tests/test_dot.cc:    cuma_B.copy_to_gpu(tensor[i]);
src/tests/test_dot.cc:    cuda_pip.gemm(cuma_B, cuma_A, cuma_C);
src/tests/test_dot.cc:  CudaPipeline cuda_pip;
src/tests/test_dot.cc:  CudaMatrix cuma_A{A, cuda_pip.get_stream()};
src/tests/test_dot.cc:  CudaMatrix cuma_B{B, cuda_pip.get_stream()};
src/tests/test_dot.cc:  CudaMatrix cuma_C{2, 5, cuda_pip.get_stream()};
src/tests/test_dot.cc:  BOOST_REQUIRE_THROW(cuda_pip.gemm(cuma_A, cuma_B, cuma_C),
src/tests/CMakeLists.txt:    eigencuda
src/CMakeLists.txt:add_library(eigencuda cudamatrix.cc cudapipeline.cc)
src/CMakeLists.txt:target_include_directories(eigencuda
src/CMakeLists.txt:  ${CUDA_INCLUDE_DIRS}
src/CMakeLists.txt:set_target_properties(eigencuda
src/CMakeLists.txt:target_compile_options(eigencuda
src/CMakeLists.txt:target_link_libraries(eigencuda
src/CMakeLists.txt:    ${CUDA_LIBRARIES}
src/CMakeLists.txt:    ${CUDA_CUBLAS_LIBRARIES}
src/cudamatrix.cc:#include "cudamatrix.hpp"
src/cudamatrix.cc:namespace eigencuda {
src/cudamatrix.cc:cudaError_t checkCuda(cudaError_t result) {
src/cudamatrix.cc:  if (result != cudaSuccess) {
src/cudamatrix.cc:    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << "\n";
src/cudamatrix.cc:Index count_available_gpus() {
src/cudamatrix.cc:  cudaError_t err = cudaGetDeviceCount(&count);
src/cudamatrix.cc:  return 0 ? (err != cudaSuccess) : Index(count);
src/cudamatrix.cc:CudaMatrix::CudaMatrix(const Eigen::MatrixXd &matrix,
src/cudamatrix.cc:                       const cudaStream_t &stream)
src/cudamatrix.cc:  _data = alloc_matrix_in_gpu(size_matrix());
src/cudamatrix.cc:  cudaError_t err = cudaMemcpyAsync(_data.get(), matrix.data(), size_matrix(),
src/cudamatrix.cc:                                    cudaMemcpyHostToDevice, stream);
src/cudamatrix.cc:CudaMatrix::CudaMatrix(Index nrows, Index ncols, const cudaStream_t &stream)
src/cudamatrix.cc:  _data = alloc_matrix_in_gpu(size_matrix());
src/cudamatrix.cc:CudaMatrix::operator Eigen::MatrixXd() const {
src/cudamatrix.cc:  checkCuda(cudaMemcpyAsync(result.data(), this->data(), this->size_matrix(),
src/cudamatrix.cc:                            cudaMemcpyDeviceToHost, this->_stream));
src/cudamatrix.cc:  checkCuda(cudaStreamSynchronize(this->_stream));
src/cudamatrix.cc:void CudaMatrix::copy_to_gpu(const Eigen::MatrixXd &A) {
src/cudamatrix.cc:  checkCuda(cudaMemcpyAsync(this->data(), A.data(), size_A,
src/cudamatrix.cc:                            cudaMemcpyHostToDevice, _stream));
src/cudamatrix.cc:CudaMatrix::Unique_ptr_to_GPU_data CudaMatrix::alloc_matrix_in_gpu(
src/cudamatrix.cc:  throw_if_not_enough_memory_in_gpu(size_arr);
src/cudamatrix.cc:  checkCuda(cudaMalloc(&dmatrix, size_arr));
src/cudamatrix.cc:  Unique_ptr_to_GPU_data dev_ptr(dmatrix,
src/cudamatrix.cc:                                 [](double *x) { checkCuda(cudaFree(x)); });
src/cudamatrix.cc:void CudaMatrix::throw_if_not_enough_memory_in_gpu(
src/cudamatrix.cc:  checkCuda(cudaMemGetInfo(&free, &total));
src/cudamatrix.cc:}  // namespace eigencuda

```
