# https://github.com/adamantine-sim/adamantine

```console
ci/Dockerfile:ARG BASE=nvidia/cuda:12.3.0-devel-ubuntu22.04
ci/Dockerfile:ARG CUDA=ON
ci/Dockerfile:    if [ $CUDA = ON ]; \
ci/Dockerfile:    ${OPENMPI_SOURCE_DIR}/configure --with-cuda=/usr/local/cuda --prefix=${OPENMPI_INSTALL_DIR};  \
ci/Dockerfile:        -DKokkos_ENABLE_CUDA=$CUDA \
ci/jenkins_config:              image "rombur/adamantine-stack:no_gpu-latest"
ci/jenkins_config:                label 'nvidia-docker || rocm-docker'
ci/jenkins_config:        stage('CUDA') {
ci/jenkins_config:                label 'nvidia-docker && ampere'
source/ThermalOperatorDevice.templates.hh:#include <deal.II/matrix_free/cuda_fe_evaluation.h>
source/ThermalOperatorDevice.templates.hh:#include <deal.II/matrix_free/cuda_matrix_free.h>
source/ThermalOperatorDevice.templates.hh:  operator()(dealii::CUDAWrappers::FEEvaluation<dim, fe_degree> *fe_eval,
source/ThermalOperatorDevice.templates.hh:             typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data const
source/ThermalOperatorDevice.templates.hh:                 *gpu_data,
source/ThermalOperatorDevice.templates.hh:             dealii::CUDAWrappers::SharedData<dim, double> *shared_data,
source/ThermalOperatorDevice.templates.hh:    typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data const
source/ThermalOperatorDevice.templates.hh:        *gpu_data,
source/ThermalOperatorDevice.templates.hh:    dealii::CUDAWrappers::SharedData<dim, double> *shared_data,
source/ThermalOperatorDevice.templates.hh:  dealii::CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double>
source/ThermalOperatorDevice.templates.hh:      fe_eval(gpu_data, shared_data);
source/ThermalOperatorDevice.templates.hh:      typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data const
source/ThermalOperatorDevice.templates.hh:          *gpu_data,
source/ThermalOperatorDevice.templates.hh:      : _cell(cell), _gpu_data(gpu_data), _cos(cos), _sin(sin),
source/ThermalOperatorDevice.templates.hh:  operator()(dealii::CUDAWrappers::FEEvaluation<dim, fe_degree> *fe_eval,
source/ThermalOperatorDevice.templates.hh:  typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data const *_gpu_data;
source/ThermalOperatorDevice.templates.hh:operator()(dealii::CUDAWrappers::FEEvaluation<dim, fe_degree> *fe_eval,
source/ThermalOperatorDevice.templates.hh:      _gpu_data->local_q_point_id(_cell, _n_q_points, q_point);
source/ThermalOperatorDevice.templates.hh:             typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data const
source/ThermalOperatorDevice.templates.hh:                 *gpu_data,
source/ThermalOperatorDevice.templates.hh:             dealii::CUDAWrappers::SharedData<dim, double> *shared_data,
source/ThermalOperatorDevice.templates.hh:           typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data const
source/ThermalOperatorDevice.templates.hh:               *gpu_data,
source/ThermalOperatorDevice.templates.hh:           dealii::CUDAWrappers::SharedData<dim, double> *shared_data,
source/ThermalOperatorDevice.templates.hh:  dealii::CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double>
source/ThermalOperatorDevice.templates.hh:      fe_eval(gpu_data, shared_data);
source/ThermalOperatorDevice.templates.hh:          cell, gpu_data, _cos, _sin, _powder_ratio, _liquid_ratio,
source/ThermalOperatorDevice.templates.hh:  // deal.II does not support QCollection on GPU
source/ThermalOperatorDevice.templates.hh:    typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data gpu_data =
source/ThermalOperatorDevice.templates.hh:    unsigned int const n_cells = gpu_data.n_cells;
source/ThermalOperatorDevice.templates.hh:    auto gpu_data_host =
source/ThermalOperatorDevice.templates.hh:        dealii::CUDAWrappers::copy_mf_data_to_host<dim, double>(
source/ThermalOperatorDevice.templates.hh:            gpu_data, _matrix_free_data.mapping_update_flags);
source/ThermalOperatorDevice.templates.hh:            gpu_data_host.local_q_point_id(cell_id, n_q_points_per_cell, i);
source/ThermalOperatorDevice.templates.hh:  dealii::CUDAWrappers::MatrixFree<dim, double> mass_matrix_free;
source/ThermalOperatorDevice.templates.hh:  typename dealii::CUDAWrappers::MatrixFree<dim, double>::AdditionalData
source/ThermalOperatorDevice.templates.hh:    typename dealii::CUDAWrappers::MatrixFree<dim, double>::Data gpu_data =
source/ThermalOperatorDevice.templates.hh:    unsigned int const n_cells = gpu_data.n_cells;
source/ThermalOperatorDevice.templates.hh:    auto gpu_data_host =
source/ThermalOperatorDevice.templates.hh:        dealii::CUDAWrappers::copy_mf_data_to_host<dim, double>(
source/ThermalOperatorDevice.templates.hh:            gpu_data, _matrix_free_data.mapping_update_flags);
source/MaterialProperty.hh:  // This cannot be private due to limitation of lambda function with CUDA
source/MaterialProperty.templates.hh:  // FIXME this is extremely slow on CUDA but this function should not exist in
source/MaterialProperty.templates.hh:  // FIXME this is extremely slow on CUDA but this function should not exist in
source/MaterialProperty.templates.hh:  // FIXME this is extremely slow on CUDA but this function should not exist in
source/MaterialProperty.templates.hh:        // Work-around CUDA compiler complaining that the first call to a
source/types.hh:  // GPU.
source/ThermalOperatorDevice.hh:#include <deal.II/matrix_free/cuda_matrix_free.h>
source/ThermalOperatorDevice.hh:  dealii::CUDAWrappers::MatrixFree<dim, double> const &get_matrix_free() const;
source/ThermalOperatorDevice.hh:  typename dealii::CUDAWrappers::MatrixFree<dim, double>::AdditionalData
source/ThermalOperatorDevice.hh:  dealii::CUDAWrappers::MatrixFree<dim, double> _matrix_free;
source/ThermalOperatorDevice.hh:inline dealii::CUDAWrappers::MatrixFree<dim, double> const &
source/ThermalPhysics.templates.hh:  // TODO do this on the GPU
source/ThermalPhysics.templates.hh:  // this if everything was done on the GPU.
tests/test_thermal_operator_device.cc:  dealii::CUDAWrappers::MatrixFree<2, double> const &matrix_free =
tests/test_thermal_operator_device.cc:  dealii::CUDAWrappers::MatrixFree<2, double> const &matrix_free =
README.md:* memory\_space (optional): device (use GPU if Kokkos compiled with GPU support) or host (use CPU) (default value: host)
application/input.info:memory_space device ; If Kokkos was compiled with GPU support, run on the device

```
