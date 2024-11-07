# https://github.com/HITS-AIN/PINK

```console
python/tests/test_dynamic.py:    trainer = pink.trainer(som, GaussianFunctor(sigma=1.1, damping=1.0), number_of_rotations=1, use_flip=False, use_gpu=True)
benchmark.md:GPU: RTX 2080
benchmark.md:| binary with CUDA [^1]                               |    7 |
benchmark.md:| binary without CUDA [^2]                            |  135 |
benchmark.md:| colab demo without CUDA                             |  127 |
benchmark.md:| colab demo without CUDA @ colab.research.google.com |  170 |
benchmark.md:[^2] `--cuda-off`
benchmark.md:| CPU-1 +    NVIDIA Tesla P40       |   3069 |    909 |
benchmark.md:| CPU-1 + 2x NVIDIA Tesla P40       |   2069 |    636 |
benchmark.md:| CPU-1 + 4x NVIDIA Tesla P40       |   1891 |    858 |
benchmark.md:| CPU-2 +    NVIDIA RTX 2080        |        |    673 |
benchmark.md:| CPU-3 +    NVIDIA GTX 750 Ti      |        |   7185 |
benchmark.md:| CPU-4 + 2x NVIDIA RTX 2080 SUPER  |        |    477 |
test/CMakeLists.txt:if(CMAKE_CUDA_COMPILER)
test/CMakeLists.txt:    add_subdirectory(CudaTest)
test/PythonBindingTest/DynamicTrainer.cpp:	DynamicTrainerTestData(bool use_gpu)
test/PythonBindingTest/DynamicTrainer.cpp:     : m_use_gpu(use_gpu)
test/PythonBindingTest/DynamicTrainer.cpp:    bool m_use_gpu;
test/PythonBindingTest/DynamicTrainer.cpp:    DynamicTrainer trainer(som, f, 0, 16, true, -1.0, Interpolation::BILINEAR, GetParam().m_use_gpu,
test/PythonBindingTest/DynamicTrainer.cpp:    DynamicTrainer trainer(som, f, 0, 16, true, -1.0, Interpolation::BILINEAR, GetParam().m_use_gpu,
test/CudaTest/mixed_precision.cu: * @file   CudaTest/mixed_precision.cpp
test/CudaTest/mixed_precision.cu:#include "CudaLib/dot_dp4a.h"
test/CudaTest/mixed_precision.cu:    cudaDeviceProp devProp;
test/CudaTest/mixed_precision.cu:    cudaGetDeviceProperties(&devProp, 0);
test/CudaTest/compare_trainer_mixed.cu: * @file   CudaTest/compare_trainer_mixed.cpp
test/CudaTest/compare_trainer_mixed.cu: * @brief  Compare generic GPU trainer with mixed precision generic GPU trainer.
test/CudaTest/compare_trainer_mixed.cu:    typedef Trainer<CartesianLayout<2>, CartesianLayout<2>, float, true> MyTrainer_gpu;
test/CudaTest/compare_trainer_mixed.cu:    MyTrainer_gpu trainer1(som1, f, 0, GetParam().m_num_rot, GetParam().m_use_flip, 0.0,
test/CudaTest/compare_trainer_mixed.cu:    MyTrainer_gpu trainer2(som2, f, 0, GetParam().m_num_rot, GetParam().m_use_flip, 0.0,
test/CudaTest/main.cpp: * @file   CudaTest/main.cpp
test/CudaTest/update_neurons.cu: * @file   CudaTest/update_neurons.cu
test/CudaTest/update_neurons.cu:#include "CudaLib/update_neurons.h"
test/CudaTest/rotate_90_degrees_list.cu: * @file   CudaTest/rotate_90_degrees_list.cu
test/CudaTest/rotate_90_degrees_list.cu:#include "CudaLib/rotate_90_degrees_list.h"
test/CudaTest/CMakeLists.txt:    CudaTest
test/CudaTest/CMakeLists.txt:    CudaTest
test/CudaTest/CMakeLists.txt:    CudaLib
test/CudaTest/CMakeLists.txt:    NAME CudaTest
test/CudaTest/CMakeLists.txt:    COMMAND CudaTest --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/CudaTest.xml
test/CudaTest/compare_trainer_cpu.cu: * @file   CudaTest/compare_trainer_cpu.cpp
test/CudaTest/compare_trainer_cpu.cu: * @brief  Compare generic GPU trainer against generic CPU trainer.
test/CudaTest/compare_trainer_cpu.cu:    typedef Trainer<CartesianLayout<2>, CartesianLayout<2>, float, true> MyTrainer_gpu;
test/CudaTest/compare_trainer_cpu.cu:    MyTrainer_gpu trainer2(som2, f, 0, GetParam().m_num_rot, GetParam().m_use_flip, -1.0,
test/CudaTest/circular_ed.cu: * @file   CudaTest/circular_ed.cu
test/CudaTest/circular_ed.cu:TEST(CudaTest, circular_ed)
test/CudaTest/resize.cu: * @file   CudaTest/resize.cu
test/CudaTest/resize.cu:#include "CudaLib/resize_kernel.h"
test/CudaTest/compare_trainer_3d_cpu.cu: * @file   CudaTest/compare_trainer_3d_cpu.cpp
test/CudaTest/compare_trainer_3d_cpu.cu: * @brief  Compare generic GPU trainer against generic CPU trainer.
test/CudaTest/compare_trainer_3d_cpu.cu:    typedef Trainer<SOMLayout, DataLayout, float, true> MyTrainer_gpu;
test/CudaTest/compare_trainer_3d_cpu.cu:    MyTrainer_gpu trainer2(som2, f, 0, GetParam().m_num_rot, GetParam().m_use_flip, -1.0,
README.md:  - CUDA >= 9.1 (highly recommended)
README.md:[Bernd Doser](https://github.com/BerndDoser), and Nikos Gianniotis. Parallelized rotation and flipping INvariant Kohonen maps (PINK) on GPUs.
CMakeLists.txt:check_language(CUDA)
CMakeLists.txt:if(CMAKE_CUDA_COMPILER)
CMakeLists.txt:    message(STATUS "CUDA support")
CMakeLists.txt:    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
CMakeLists.txt:        set(CMAKE_CUDA_ARCHITECTURES 75)
CMakeLists.txt:    enable_language(CUDA)
CMakeLists.txt:    message(STATUS "No CUDA support")
CMakeLists.txt:if(CMAKE_CUDA_COMPILER)
CMakeLists.txt:    set(PINK_USE_CUDA true)
CMakeLists.txt:    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPINK_USE_CUDA")
CMakeLists.txt:    message("CMAKE_CUDA_ARCHITECTURES : ${CMAKE_CUDA_ARCHITECTURES}")
CMakeLists.txt:    if(NOT DEFINED CMAKE_CUDA_STANDARD)
CMakeLists.txt:        set(CMAKE_CUDA_STANDARD 17)
CMakeLists.txt:        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
README-dev.md:docker run -it -v $PWD:/work -w /work bernddoser/manylinux2010-cuda /bin/bash
README-dev.md:mv libCudaLib.so libPythonBindingLib.so astro_pink.libs
Jenkinsfile:      label 'docker-gpu-host'
Jenkinsfile:      args '--gpus=all'
src/Pink/main.cpp:#ifdef PINK_USE_CUDA
src/Pink/main.cpp:    #include "CudaLib/main_gpu.h"
src/Pink/main.cpp:        if (input_data.m_use_gpu)
src/Pink/main.cpp:#ifdef PINK_USE_CUDA
src/Pink/main.cpp:            main_gpu(input_data);
src/Pink/main.cpp:            throw pink::exception("PINK was not compiled with CUDA support");
src/Pink/main_generic.h:template <typename SOMLayout, typename T, bool UseGPU>
src/Pink/main_generic.h:            main_generic<SOMLayout, CartesianLayout<1U>, T, UseGPU>(input_data);
src/Pink/main_generic.h:            main_generic<SOMLayout, CartesianLayout<2U>, T, UseGPU>(input_data);
src/Pink/main_generic.h:            main_generic<SOMLayout, CartesianLayout<3U>, T, UseGPU>(input_data);
src/Pink/main_generic.h:template <typename SOMLayout, typename DataLayout, typename T, bool UseGPU>
src/Pink/main_generic.h:        Trainer<SOMLayout, DataLayout, T, UseGPU> trainer(
src/Pink/main_generic.h:#ifdef __CUDACC__
src/Pink/main_generic.h:                    #ifdef __CUDACC__
src/Pink/main_generic.h:#ifdef __CUDACC__
src/Pink/main_generic.h:        Mapper<SOMLayout, DataLayout, T, UseGPU> mapper(
src/Pink/main_generic.h:#ifdef __CUDACC__
src/Pink/CMakeLists.txt:if(PINK_USE_CUDA)
src/Pink/CMakeLists.txt:        CudaLib
src/SelfOrganizingMapLib/Mapper.h:#ifdef __CUDACC__
src/SelfOrganizingMapLib/Mapper.h:    #include "CudaLib/CudaLib.h"
src/SelfOrganizingMapLib/Mapper.h:    #include "CudaLib/generate_euclidean_distance_matrix.h"
src/SelfOrganizingMapLib/Mapper.h:    #include "CudaLib/generate_rotated_images.h"
src/SelfOrganizingMapLib/Mapper.h:    #include "CudaLib/update_neurons.h"
src/SelfOrganizingMapLib/Mapper.h:template <typename SOMLayout, typename DataLayout, typename T, bool UseGPU>
src/SelfOrganizingMapLib/Mapper.h:#ifdef __CUDACC__
src/SelfOrganizingMapLib/Mapper.h:/// GPU version of training
src/SelfOrganizingMapLib/Mapper.h:        SpatialTransformerGPU<DataLayout>()(
src/SelfOrganizingMapLib/Trainer.h:#ifdef __CUDACC__
src/SelfOrganizingMapLib/Trainer.h:    #include "CudaLib/CudaLib.h"
src/SelfOrganizingMapLib/Trainer.h:    #include "CudaLib/generate_euclidean_distance_matrix.h"
src/SelfOrganizingMapLib/Trainer.h:    #include "CudaLib/generate_rotated_images.h"
src/SelfOrganizingMapLib/Trainer.h:    #include "CudaLib/update_neurons.h"
src/SelfOrganizingMapLib/Trainer.h:template <typename SOMLayout, typename DataLayout, typename T, bool UseGPU>
src/SelfOrganizingMapLib/Trainer.h:#ifdef __CUDACC__
src/SelfOrganizingMapLib/Trainer.h:/// GPU version of training
src/SelfOrganizingMapLib/Trainer.h:        SpatialTransformerGPU<DataLayout>()(
src/UtilitiesLib/expect_floats_nearly_eq.h: * @file   CudaTest/expect_floats_nearly_eq.h
src/UtilitiesLib/InputData.h:    bool m_use_gpu;
src/UtilitiesLib/InputData.cpp:   m_use_gpu(true),
src/UtilitiesLib/InputData.cpp:        {"cuda-off",                     0, nullptr, 3},
src/UtilitiesLib/InputData.cpp:                m_use_gpu = false;
src/UtilitiesLib/InputData.cpp:              << "  Use CUDA = " << m_use_gpu << "\n";
src/UtilitiesLib/InputData.cpp:                 "    --cuda-off                                    "
src/UtilitiesLib/InputData.cpp:                 "Switch off CUDA acceleration.\n"
src/CMakeLists.txt:if(CMAKE_CUDA_COMPILER)
src/CMakeLists.txt:    add_subdirectory(CudaLib)
src/CudaLib/rotate_bilinear_kernel.h: * @file   CudaLib/rotate_bilinear_kernel.h
src/CudaLib/rotate_bilinear_kernel.h:#include <cuda_runtime.h>
src/CudaLib/rotate_bilinear_kernel.h:/// CUDA device kernel for rotating a list of quadratic images
src/CudaLib/copy_and_transform_kernel.h: * @file   CudaLib/copy_and_transform_kernel.h
src/CudaLib/copy_and_transform_kernel.h:#include <cuda_runtime.h>
src/CudaLib/copy_and_transform_kernel.h:/// CUDA device kernel to copy a submatrix and transform elements
src/CudaLib/copy_and_transform_kernel.h:/// CUDA device kernel to copy a submatrix and transform elements
src/CudaLib/rotate_90_degrees_list.h: * @file   CudaLib/rotate_90_degrees_list.h
src/CudaLib/rotate_90_degrees_list.h:#include <cuda_runtime.h>
src/CudaLib/rotate_90_degrees_list.h: * CUDA Kernel Device code for special clockwise rotation of 90 degrees of a list of quadratic images.
src/CudaLib/circular_mapping.cu: * @file   CudaLib/circular_mapping.cu
src/CudaLib/main_gpu.cu: * @file   Pink/main_gpu.cpp
src/CudaLib/main_gpu.cu:#include "main_gpu.h"
src/CudaLib/main_gpu.cu:void main_gpu(InputData const & input_data)
src/CudaLib/main_gpu.cu:    cuda_print_properties();
src/CudaLib/CudaLib.h: * @file   CudaLib/CudaLib.h
src/CudaLib/CudaLib.h: * @brief  Basic CUDA functions
src/CudaLib/CudaLib.h:/// Print CUDA device properties
src/CudaLib/CudaLib.h:void cuda_print_properties();
src/CudaLib/CudaLib.h:/// Return IDs of available GPU devices in CUDA_VISIBLE_DEVICES
src/CudaLib/CudaLib.h:std::vector<int> cuda_get_gpu_ids();
src/CudaLib/CudaLib.h:#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
src/CudaLib/CudaLib.h:inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
src/CudaLib/CudaLib.h:   if (code != cudaSuccess)
src/CudaLib/CudaLib.h:      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
src/CudaLib/generate_euclidean_distance_matrix_first_step.h: * @file   CudaLib/generate_euclidean_distance_matrix_first_step.h
src/CudaLib/generate_euclidean_distance_matrix_first_step.h:/// Host function that prepares data array and passes it to the CUDA kernel
src/CudaLib/generate_euclidean_distance_matrix_first_step.h:    gpuErrchk(cudaDeviceSynchronize());
src/CudaLib/dot_dp4a.cu: * @file   CudaLib/dot_dp4a.cu
src/CudaLib/dot_dp4a.cu:#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
src/CudaLib/dot_dp4a.cu:    cudaDeviceSynchronize();
src/CudaLib/dot_dp4a.cu:#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
src/CudaLib/dot_dp4a.cu:    cudaDeviceSynchronize();
src/CudaLib/rotate90degrees_kernel.h: * @file   CudaLib/rotateAndCrop_kernel.h
src/CudaLib/rotate90degrees_kernel.h:#include <cuda_runtime.h>
src/CudaLib/rotate90degrees_kernel.h: * CUDA Kernel Device code for special clockwise rotation of 90 degrees of a quadratic image.
src/CudaLib/dot_dp4a.h: * @file   CudaLib/dot_dp4a.h
src/CudaLib/CudaLib.cu: * @file   CudaLib/CudaLib.cu
src/CudaLib/CudaLib.cu: * @brief  Basic CUDA functions
src/CudaLib/CudaLib.cu:#include <cuda_runtime.h>
src/CudaLib/CudaLib.cu:#include "CudaLib.h"
src/CudaLib/CudaLib.cu:void cuda_print_properties()
src/CudaLib/CudaLib.cu:    cudaGetDeviceCount(&devCount);
src/CudaLib/CudaLib.cu:    printf("  CUDA Device Query...\n");
src/CudaLib/CudaLib.cu:    printf("  There are %d CUDA devices.\n", devCount);
src/CudaLib/CudaLib.cu:        printf("\n  CUDA Device #%d\n", i);
src/CudaLib/CudaLib.cu:        cudaDeviceProp devProp;
src/CudaLib/CudaLib.cu:        cudaGetDeviceProperties(&devProp, i);
src/CudaLib/CudaLib.cu:std::vector<int> cuda_get_gpu_ids()
src/CudaLib/CudaLib.cu:    std::vector<int> gpu_ids;
src/CudaLib/CudaLib.cu:    int number_of_gpu_devices;
src/CudaLib/CudaLib.cu:    cudaGetDeviceCount(&number_of_gpu_devices);
src/CudaLib/CudaLib.cu:    char const* cuda_visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
src/CudaLib/CudaLib.cu:    if (cuda_visible_devices == nullptr) {
src/CudaLib/CudaLib.cu:        gpu_ids.resize(static_cast<size_t>(number_of_gpu_devices));
src/CudaLib/CudaLib.cu:        std::iota(std::begin(gpu_ids), std::end(gpu_ids), 0);
src/CudaLib/CudaLib.cu:        for(std::stringstream ss(cuda_visible_devices); std::getline(ss, token, ',');)
src/CudaLib/CudaLib.cu:            gpu_ids.push_back(std::stoi(token));
src/CudaLib/CudaLib.cu:    return gpu_ids;
src/CudaLib/CMakeLists.txt:    CudaLib
src/CudaLib/CMakeLists.txt:    CudaLib.cu
src/CudaLib/CMakeLists.txt:    main_gpu.cu
src/CudaLib/CMakeLists.txt:    CudaLib
src/CudaLib/CMakeLists.txt:install(TARGETS CudaLib DESTINATION pink)
src/CudaLib/rotateAndCropTexture_kernel.h:texture<float, 2, cudaReadModeElementType> image_texture;
src/CudaLib/rotateAndCropTexture_kernel.h: * CUDA Kernel Device code for combined rotation and cropping of a list of images.
src/CudaLib/generate_rotated_images.h: * @file   CudaLib/generate_rotated_images.h
src/CudaLib/generate_rotated_images.h:/// Primary template for SpatialTransformer (GPU)
src/CudaLib/generate_rotated_images.h:struct SpatialTransformerGPU
src/CudaLib/generate_rotated_images.h:/// SpatialTransformer (GPU): Specialization for CartesianLayout<1>
src/CudaLib/generate_rotated_images.h:struct SpatialTransformerGPU<CartesianLayout<1>>
src/CudaLib/generate_rotated_images.h:/// SpatialTransformer (GPU): Specialization for CartesianLayout<2>
src/CudaLib/generate_rotated_images.h:struct SpatialTransformerGPU<CartesianLayout<2>>
src/CudaLib/generate_rotated_images.h:            gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_rotated_images.h:                    gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_rotated_images.h:				gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_rotated_images.h:				gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_rotated_images.h:				gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_rotated_images.h:            gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_rotated_images.h:/// SpatialTransformer (GPU): Specialization for CartesianLayout<3>
src/CudaLib/generate_rotated_images.h:struct SpatialTransformerGPU<CartesianLayout<3>>
src/CudaLib/generate_rotated_images.h:                gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_rotated_images.h:                        gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_rotated_images.h:                    gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_rotated_images.h:            gpuErrchk(cudaPeekAtLastError());
src/CudaLib/update_neurons.h: * @file   CudaLib/update_neurons.h
src/CudaLib/update_neurons.h:#include "CudaLib.h"
src/CudaLib/update_neurons.h:/// CUDA Kernel Device code updating quadratic self organizing map using gaussian function.
src/CudaLib/update_neurons.h: * Host function that prepares data array and passes it to the CUDA kernel.
src/CudaLib/update_neurons.h:        gpuErrchk(cudaPeekAtLastError());
src/CudaLib/update_neurons.h:        gpuErrchk(cudaDeviceSynchronize());
src/CudaLib/update_neurons.h:        gpuErrchk(cudaPeekAtLastError());
src/CudaLib/update_neurons.h:        gpuErrchk(cudaDeviceSynchronize());
src/CudaLib/euclidean_distance_kernel.h: * @file   CudaLib/euclidean_distance_kernel.h
src/CudaLib/euclidean_distance_kernel.h:/// CUDA device kernel for reducing a data array with the length of 64 by
src/CudaLib/euclidean_distance_kernel.h:/// CUDA device kernel to computes the euclidean distance of two arrays
src/CudaLib/generate_euclidean_distance_matrix_second_step.h: * @file   CudaLib/generate_euclidean_distance_matrix_second_step.h
src/CudaLib/generate_euclidean_distance_matrix_second_step.h: * CUDA Kernel Device code
src/CudaLib/generate_euclidean_distance_matrix_second_step.h: * Host function that prepares data array and passes it to the CUDA kernel.
src/CudaLib/generate_euclidean_distance_matrix_second_step.h:    gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_euclidean_distance_matrix_second_step.h:    gpuErrchk(cudaDeviceSynchronize());
src/CudaLib/flip_kernel.h: * @file   CudaLib/flip_kernel.h
src/CudaLib/flip_kernel.h:#include <cuda_runtime.h>
src/CudaLib/flip_kernel.h: * CUDA Kernel Device code for flipping an image.
src/CudaLib/main_gpu.h: * @file   Pink/main_gpu.h
src/CudaLib/main_gpu.h:void main_gpu(InputData const & input_data);
src/CudaLib/crop_kernel.h: * @file   CudaLib/crop_kernel.h
src/CudaLib/crop_kernel.h:#include <cuda_runtime.h>
src/CudaLib/crop_kernel.h: * CUDA Kernel Device code for cropping an image.
src/CudaLib/resize_kernel.h: * @file   CudaLib/resize_kernel.h
src/CudaLib/resize_kernel.h:#include <cuda_runtime.h>
src/CudaLib/resize_kernel.h:/// CUDA device kernel to resize an quadratic, row-major image
src/CudaLib/rotate_and_crop_nearest_neighbor_kernel.h: * @file   CudaLib/rotate_and_crop_nearest_neighbor_kernel.h
src/CudaLib/rotate_and_crop_nearest_neighbor_kernel.h:#include <cuda_runtime.h>
src/CudaLib/rotate_and_crop_nearest_neighbor_kernel.h: * CUDA Kernel Device code for combined rotation and cropping of a list of quadratic images.
src/CudaLib/rotate_and_crop_bilinear_kernel.h: * @file   CudaLib/rotate_and_crop_bilinear_kernel.h
src/CudaLib/rotate_and_crop_bilinear_kernel.h:#include <cuda_runtime.h>
src/CudaLib/rotate_and_crop_bilinear_kernel.h: * CUDA Kernel Device code for combined rotation and cropping of a list of quadratic images.
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h: * @file   CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    auto&& gpu_ids = cuda_get_gpu_ids();
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:        cudaSetDevice(gpu_ids[i + 1]);
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    cudaSetDevice(gpu_ids[0]);
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:/// Calculate euclidean distance on multiple GPU devices
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:void generate_euclidean_distance_matrix_first_step_multi_gpu(thrust::device_vector<EuclideanType> const& d_som,
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    auto&& gpu_ids = cuda_get_gpu_ids();
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    auto number_of_gpus = gpu_ids.size();
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    if (number_of_threads < number_of_gpus) {
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:        std::cout << "Number of GPUs = " << number_of_gpus << std::endl;
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:        std::cout << "GPU IDs = ";
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:        for (auto id : gpu_ids) std::cout << id << " ";
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:        throw pink::exception("Number of CPU threads must not be smaller than the number of GPU devices");
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    std::vector<uint32_t> size(number_of_gpus);
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    uint32_t rest = som_size % number_of_gpus;
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    for (uint32_t i = 0; i < number_of_gpus; ++i) {
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:        size[i] = som_size / number_of_gpus;
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    std::vector<uint32_t> offset(number_of_gpus);
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    for (uint32_t i = 1; i < number_of_gpus; ++i) {
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:        std::vector<uint32_t>(number_of_gpus - 1, number_of_spatial_transformations * neuron_size));
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    assert(number_of_gpus < std::numeric_limits<int>::max());
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    for (int i = 1; i < static_cast<int>(number_of_gpus); ++i)
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:        // Set GPU device
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:        cudaSetDevice(gpu_ids[si]);
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:        gpuErrchk(cudaMemcpyPeer(thrust::raw_pointer_cast(d_som_local[si - 1].data()), i,
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:        gpuErrchk(cudaMemcpyPeer(thrust::raw_pointer_cast(d_rotated_images_local[si - 1].data()), i,
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:        gpuErrchk(cudaDeviceSynchronize());
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    for (int i = 1; i < static_cast<int>(number_of_gpus); ++i)
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:            // Set GPU device
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:            cudaSetDevice(gpu_ids[si]);
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:            gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:            gpuErrchk(cudaDeviceSynchronize());
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    // Set GPU device to master
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    cudaSetDevice(gpu_ids[0]);
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    gpuErrchk(cudaDeviceSynchronize());
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    for (int i = 1; i < static_cast<int>(number_of_gpus); ++i)
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:        gpuErrchk(cudaMemcpyPeer(
src/CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h:    gpuErrchk(cudaDeviceSynchronize());
src/CudaLib/generate_euclidean_distance_matrix.h: * @file   CudaLib/generate_euclidean_distance_matrix.h
src/CudaLib/generate_euclidean_distance_matrix.h:#include "CudaLib.h"
src/CudaLib/generate_euclidean_distance_matrix.h:#include "generate_euclidean_distance_matrix_first_step_multi_gpu.h"
src/CudaLib/generate_euclidean_distance_matrix.h: * Host function that prepares data array and passes it to the CUDA kernel.
src/CudaLib/generate_euclidean_distance_matrix.h:        gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_euclidean_distance_matrix.h:        gpuErrchk(cudaDeviceSynchronize());
src/CudaLib/generate_euclidean_distance_matrix.h:        gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_euclidean_distance_matrix.h:        gpuErrchk(cudaDeviceSynchronize());
src/CudaLib/generate_euclidean_distance_matrix.h:        if (cuda_get_gpu_ids().size() > 1) {
src/CudaLib/generate_euclidean_distance_matrix.h:            generate_euclidean_distance_matrix_first_step_multi_gpu(d_som_uint8, d_spatial_transformed_images_uint8,
src/CudaLib/generate_euclidean_distance_matrix.h:        gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_euclidean_distance_matrix.h:        gpuErrchk(cudaDeviceSynchronize());
src/CudaLib/generate_euclidean_distance_matrix.h:        gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_euclidean_distance_matrix.h:        gpuErrchk(cudaDeviceSynchronize());
src/CudaLib/generate_euclidean_distance_matrix.h:        if (cuda_get_gpu_ids().size() > 1) {
src/CudaLib/generate_euclidean_distance_matrix.h:            generate_euclidean_distance_matrix_first_step_multi_gpu(d_som_uint16, d_spatial_transformed_images_uint16,
src/CudaLib/generate_euclidean_distance_matrix.h:        gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_euclidean_distance_matrix.h:        gpuErrchk(cudaDeviceSynchronize());
src/CudaLib/generate_euclidean_distance_matrix.h:        gpuErrchk(cudaPeekAtLastError());
src/CudaLib/generate_euclidean_distance_matrix.h:        gpuErrchk(cudaDeviceSynchronize());
src/CudaLib/generate_euclidean_distance_matrix.h:        if (cuda_get_gpu_ids().size() > 1) {
src/CudaLib/generate_euclidean_distance_matrix.h:            generate_euclidean_distance_matrix_first_step_multi_gpu(d_som_float, d_spatial_transformed_images_float,
src/pink/DynamicTrainer.cpp:    Interpolation interpolation, bool use_gpu, uint32_t euclidean_distance_dim,
src/pink/DynamicTrainer.cpp:   m_use_gpu(use_gpu)
src/pink/DynamicMapper.h:        Interpolation interpolation, bool use_gpu, uint32_t euclidean_distance_dim,
src/pink/DynamicMapper.h:        if (m_use_gpu == false) {
src/pink/DynamicMapper.h:#ifdef __CUDACC__
src/pink/DynamicMapper.h:            throw pink::exception("GPU support is not supported");
src/pink/DynamicMapper.h:        if (m_use_gpu == false) {
src/pink/DynamicMapper.h:#ifdef __CUDACC__
src/pink/DynamicMapper.h:            throw pink::exception("GPU support is not supported");
src/pink/DynamicMapper.h:    bool m_use_gpu;
src/pink/pink.cpp:            py::arg("use_gpu") = true,
src/pink/pink.cpp:            py::arg("use_gpu") = true,
src/pink/CMakeLists.txt:if(CMAKE_CUDA_COMPILER)
src/pink/CMakeLists.txt:    set_source_files_properties(DynamicMapper.cpp DynamicTrainer.cpp PROPERTIES LANGUAGE CUDA)
src/pink/CMakeLists.txt:        CudaLib
src/pink/DynamicTrainer.h:        Interpolation interpolation, bool use_gpu, uint32_t euclidean_distance_dim,
src/pink/DynamicTrainer.h:        if (m_use_gpu == false) {
src/pink/DynamicTrainer.h:#ifdef __CUDACC__
src/pink/DynamicTrainer.h:            throw pink::exception("GPU is not supported");
src/pink/DynamicTrainer.h:        if (m_use_gpu == false) {
src/pink/DynamicTrainer.h:#ifdef __CUDACC__
src/pink/DynamicTrainer.h:            throw pink::exception("GPU is not supported");
src/pink/DynamicTrainer.h:    bool m_use_gpu;
src/pink/DynamicMapper.cpp:    bool use_flip, Interpolation interpolation, bool use_gpu, uint32_t euclidean_distance_dim,
src/pink/DynamicMapper.cpp:   m_use_gpu(use_gpu)

```
