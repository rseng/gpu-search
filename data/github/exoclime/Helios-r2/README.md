# https://github.com/exoclime/Helios-r2

```console
SecondaryEclipseExample/retrieval.config:#Use GPU
EmissionExample/retrieval.config:#Use GPU
SecondaryEclipseExampleBB/retrieval.config:#Use GPU
FlatLineExample/retrieval.config:#Use GPU
README.md:BeAR uses a Bayesian statistics approach by employing a nested sampling method to generate posterior distributions and calculate the Bayesian evidence. The nested sampling itself is done by the Multinest library (https://github.com/farhanferoz/MultiNest). The computationally most demanding parts of the model have been written in NVIDIA's CUDA language for an increase in computational speed. BeAR can work on both, pure CPU as well as hybrid CPU/GPU setups. Running it purely on a CPU is not recommended, though, as the runtimes can be  by a factor of 10 or 100 longer compared to running it on a GPU.
CMakeLists.txt:# the script should autodetect the CUDA architecture, when run alone
CMakeLists.txt:set(SM "0" CACHE STRING "GPU SM value")
CMakeLists.txt:project(bear C CXX CUDA Fortran)
CMakeLists.txt:  set(CUDA_ARCH_FLAGS "-arch=sm_61")
CMakeLists.txt:  message(STATUS "CUDA Architecture manually set to: -arch=sm_${SM}")
CMakeLists.txt:  set(CUDA_ARCH_FLAGS "-arch=sm_${SM}")
CMakeLists.txt:#set CUDA flags
CMakeLists.txt:string(APPEND CMAKE_CUDA_FLAGS ${CUDA_ARCH_FLAGS})
CMakeLists.txt:string(APPEND CMAKE_CUDA_FLAGS " -std=c++11 -lineinfo -Xptxas -v")
CMakeLists.txt:#string (APPEND CMAKE_CUDA_FLAGS " -cudart shared" )
CMakeLists.txt:set(SRC_CUDA
CMakeLists.txt:  src/CUDA_kernels/band_integration_kernels.cu
CMakeLists.txt:  src/CUDA_kernels/convolution_kernels.cu
CMakeLists.txt:  src/CUDA_kernels/filter_response_kernels.cu
CMakeLists.txt:  src/CUDA_kernels/cross_section_kernels.cu
CMakeLists.txt:  src/CUDA_kernels/data_management_kernels.cu
CMakeLists.txt:  src/CUDA_kernels/log_like_kernels.cu
CMakeLists.txt:  src/CUDA_kernels/secondary_eclipse_kernels.cu
CMakeLists.txt:  src/CUDA_kernels/secondary_eclipse_bb.cu
CMakeLists.txt:  src/CUDA_kernels/transmission_spectrum_kernels.cu
CMakeLists.txt:  src/CUDA_kernels/short_characteristics_kernels.cu
CMakeLists.txt:  src/CUDA_kernels/rayleigh_scattering_kernels.cu
CMakeLists.txt:  src/CUDA_kernels/cloud_model_kernels.cu
CMakeLists.txt:  src/CUDA_kernels/contribution_function_kernels.cu
CMakeLists.txt:  src/CUDA_kernels/star_blackbody.cu
CMakeLists.txt:  src/CUDA_kernels/star_file_spectrum.cu
CMakeLists.txt:  src/CUDA_kernels/stellar_contamination.cu
CMakeLists.txt:  src/CUDA_kernels/stellar_spectrum_grid.cu
CMakeLists.txt:  src/CUDA_kernels/observation_kernels.cu)
CMakeLists.txt:#compilation target for the CUDA files
CMakeLists.txt:#CUDA requires different compiler options, so it's compiled separately
CMakeLists.txt:add_library(bear_cuda ${SRC_CUDA})
CMakeLists.txt:target_include_directories(bear_cuda PUBLIC ${boost_math_SOURCE_DIR}/include/)
CMakeLists.txt:target_link_libraries(bear PUBLIC fastchem_lib bear_cuda multinest_shared OpenMP::OpenMP_CXX ${LAPACK_LIBRARIES})
TransmissionExample/retrieval.config:#Use GPU
src/config/global_config.cpp:  if (input == "Y" || input == "1") use_gpu = true;
src/config/global_config.cpp:  std::cout << "- Use GPU: " << use_gpu << "\n";
src/config/global_config.h:  bool use_gpu = false;
src/retrieval/post_process_spectra.cpp:#include "../CUDA_kernels/data_management_kernels.h"
src/retrieval/retrieval.cpp:#include "../CUDA_kernels/data_management_kernels.h"
src/retrieval/retrieval.cpp:  if (config->use_gpu == false)
src/retrieval/retrieval.cpp:      Retrieval::multinestLogLikeGPU,
src/retrieval/retrieval.cpp:  if (config->use_gpu)
src/retrieval/retrieval.cpp:    deleteFromDevice(model_spectrum_gpu);
src/retrieval/retrieval.cpp:    deleteFromDevice(observation_data_gpu);
src/retrieval/retrieval.cpp:    deleteFromDevice(observation_error_gpu);
src/retrieval/retrieval.cpp:    deleteFromDevice(observation_likelihood_weight_gpu);
src/retrieval/retrieval_load_observations.cpp:#include "../CUDA_kernels/data_management_kernels.h"
src/retrieval/retrieval_load_observations.cpp:  //move the lists to the GPU, if necessary
src/retrieval/retrieval_load_observations.cpp:  if (config->use_gpu)
src/retrieval/retrieval_load_observations.cpp:    moveToDevice(observation_data_gpu, observation_data);
src/retrieval/retrieval_load_observations.cpp:    moveToDevice(observation_error_gpu, observation_error);
src/retrieval/retrieval_load_observations.cpp:    moveToDevice(observation_likelihood_weight_gpu, observation_likelihood_weight);
src/retrieval/retrieval_load_observations.cpp:  //create the vector for the high-res spectrum on the GPU
src/retrieval/retrieval_load_observations.cpp:  if (config->use_gpu)
src/retrieval/retrieval_load_observations.cpp:    allocateOnDevice(model_spectrum_gpu, spectral_grid.nbSpectralPoints());
src/retrieval/multinest_loglike.cpp:#include "../CUDA_kernels/data_management_kernels.h"
src/retrieval/multinest_loglike.cpp:void Retrieval::multinestLogLikeGPU(double *cube, int &nb_dim, int &nb_param, double &lnew, void *context)
src/retrieval/multinest_loglike.cpp:  //pointer to the spectrum on the GPU
src/retrieval/multinest_loglike.cpp:  intializeOnDevice(retrieval_ptr->model_spectrum_gpu, nb_points);
src/retrieval/multinest_loglike.cpp:  bool neglect = retrieval_ptr->forward_model->calcModelGPU(physical_parameter, retrieval_ptr->model_spectrum_gpu, model_spectrum_bands);
src/retrieval/retrieval.h:    double* observation_data_gpu = nullptr;             //pointer to the corresponding data on the GPU
src/retrieval/retrieval.h:    double* observation_error_gpu = nullptr;
src/retrieval/retrieval.h:    double* observation_likelihood_weight_gpu = nullptr;
src/retrieval/retrieval.h:    double* model_spectrum_gpu = nullptr;            //pointer to the high-res spectrum on the GPU
src/retrieval/retrieval.h:    static void multinestLogLikeGPU(
src/retrieval/post_process.cpp:#include "../CUDA_kernels/data_management_kernels.h"
src/retrieval/post_process.cpp:  if (config->use_gpu)
src/retrieval/post_process.cpp:    deleteFromDevice(observation_data_gpu);
src/retrieval/post_process.cpp:    deleteFromDevice(observation_error_gpu);
src/observations/observations_filter_response.cpp:#include "../CUDA_kernels/data_management_kernels.h"
src/observations/observations_filter_response.cpp:  if (config->use_gpu)
src/observations/observations_filter_response.cpp:    moveToDevice(filter_response_gpu, filter_response);
src/observations/observations_filter_response.cpp:    moveToDevice(filter_response_weight_gpu, filter_response_weight);
src/observations/observations_load_data.cpp:#include "../CUDA_kernels/data_management_kernels.h"
src/observations/observations.h:    double* filter_response_gpu = nullptr;
src/observations/observations.h:    double* filter_response_weight_gpu = nullptr;
src/observations/observations.h:    void applyFilterResponseGPU(double* spectrum);
src/observations/observations.h:    void processModelSpectrumGPU(
src/observations/observations.h:    void addShiftToSpectrumGPU(
src/observations/observations.cpp:#include "../CUDA_kernels/data_management_kernels.h"
src/observations/observations.cpp:  if (config->use_gpu)
src/observations/observations.cpp:    deleteFromDevice(filter_response_gpu);
src/observations/observations.cpp:    deleteFromDevice(filter_response_weight_gpu);
src/observations/observations.cpp:  if (config->use_gpu)
src/observations/observations.cpp:void Observation::processModelSpectrumGPU(
src/observations/observations.cpp:    applyFilterResponseGPU(spectrum);
src/observations/observations.cpp:      spectral_bands.convolveSpectrumGPU(
src/observations/observations.cpp:      spectral_bands.bandIntegrateSpectrumGPU(
src/observations/observations.cpp:      spectral_bands.bandIntegrateSpectrumGPU(
src/observations/observations.cpp:    spectral_bands.convolveSpectrumGPU(
src/observations/observations.cpp:    spectral_bands.bandIntegrateSpectrumGPU(
src/observations/observations.cpp:    spectral_bands.bandIntegrateSpectrumGPU(
src/transport_coeff/opacity_species.h:    virtual void calcTransportCoefficientsGPU(
src/transport_coeff/opacity_species.h:    virtual void calcContinuumAbsorptionGPU(
src/transport_coeff/opacity_species.h:    virtual void calcRayleighCrossSectionsGPU(
src/transport_coeff/opacity_species.h:    void calcAbsorptionCoefficientsGPU(
src/transport_coeff/species_definition.h:#include "../CUDA_kernels/cross_section_kernels.h"
src/transport_coeff/species_definition.h:    virtual void calcContinuumAbsorptionGPU(
src/transport_coeff/species_definition.h:          spectral_grid->wavelength_list_gpu,
src/transport_coeff/species_definition.h:    virtual void calcRayleighCrossSectionsGPU(
src/transport_coeff/species_definition.h:    virtual void calcRayleighCrossSectionsGPU(
src/transport_coeff/species_definition.h:    virtual void calcRayleighCrossSectionsGPU(
src/transport_coeff/species_definition.h:    virtual void calcRayleighCrossSectionsGPU(
src/transport_coeff/species_definition.h:    virtual void calcRayleighCrossSectionsGPU(
src/transport_coeff/species_definition.h:    virtual void calcRayleighCrossSectionsGPU(
src/transport_coeff/species_definition.h:    virtual void calcRayleighCrossSectionsGPU(
src/transport_coeff/sampled_data.cpp:#include "../CUDA_kernels/cross_section_kernels.h"
src/transport_coeff/sampled_data.cpp:#include "../CUDA_kernels/data_management_kernels.h"
src/transport_coeff/sampled_data.cpp:  if (use_gpu)
src/transport_coeff/transport_coeff.h:    void calculateGPU(
src/transport_coeff/opacity_species.cpp:#include "../CUDA_kernels/cross_section_kernels.h"
src/transport_coeff/opacity_species.cpp:#include "../CUDA_kernels/data_management_kernels.h"
src/transport_coeff/opacity_species.cpp:        temperature, pressure, file_name, log_storage, config->use_gpu) );
src/transport_coeff/opacity_species.cpp://this is version for doing the calculation on the GPU
src/transport_coeff/opacity_species.cpp://the four data points for the interpolation are still obtained on the CPU and then passed to the GPU
src/transport_coeff/opacity_species.cpp:void OpacitySpecies::calcAbsorptionCoefficientsGPU(
src/transport_coeff/opacity_species.cpp:void OpacitySpecies::calcTransportCoefficientsGPU(
src/transport_coeff/opacity_species.cpp:  if (cross_section_available == true) calcAbsorptionCoefficientsGPU(
src/transport_coeff/opacity_species.cpp:  calcRayleighCrossSectionsGPU(
src/transport_coeff/opacity_species.cpp:  calcContinuumAbsorptionGPU(
src/transport_coeff/sampled_data.h:      const bool gpu_avail)
src/transport_coeff/sampled_data.h:        , use_gpu(gpu_avail) 
src/transport_coeff/sampled_data.h:    double* cross_sections_device = nullptr; //pointer to the cross section data on the GPU
src/transport_coeff/sampled_data.h:    const bool use_gpu = false;
src/transport_coeff/opacity_calc.h:#include "../CUDA_kernels/data_management_kernels.h"
src/transport_coeff/opacity_calc.h:#include "../CUDA_kernels/cross_section_kernels.h"
src/transport_coeff/opacity_calc.h:      const bool use_gpu_,
src/transport_coeff/opacity_calc.h:      , use_gpu(use_gpu_)
src/transport_coeff/opacity_calc.h:    void calculateGPU(
src/transport_coeff/opacity_calc.h:    //pointer to the array that holds the pointers to the coefficients on the GPU
src/transport_coeff/opacity_calc.h:    double* absorption_coeff_gpu = nullptr;
src/transport_coeff/opacity_calc.h:    bool use_gpu = false;
src/transport_coeff/opacity_calc.h:inline void OpacityCalculation::calculateGPU(
src/transport_coeff/opacity_calc.h:    absorption_coeff_gpu);
src/transport_coeff/opacity_calc.h:    transport_coeff.calculateGPU(
src/transport_coeff/opacity_calc.h:      absorption_coeff_gpu, 
src/transport_coeff/opacity_calc.h:      cm->opticalPropertiesGPU(
src/transport_coeff/opacity_calc.h:  allocateOnDevice(absorption_coeff_gpu, nb_grid_points*nb_spectral_points);
src/transport_coeff/opacity_calc.h:  if (use_gpu)
src/transport_coeff/opacity_calc.h:    deleteFromDevice(absorption_coeff_gpu);
src/transport_coeff/transport_coeff.cpp://calculates the transport coefficients on the GPU
src/transport_coeff/transport_coeff.cpp://calculations are stored on the GPU, nothing is returned
src/transport_coeff/transport_coeff.cpp:void TransportCoefficients::calculateGPU(
src/transport_coeff/transport_coeff.cpp:    gas_species[i]->calcTransportCoefficientsGPU(
src/CUDA_kernels/data_management_kernels.cu://allocates a one-dimensional array of size nb_double_values on the GPU
src/CUDA_kernels/data_management_kernels.cu:  cudaMalloc((void**)&device_data, bytes);
src/CUDA_kernels/data_management_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/data_management_kernels.cu://deletes a 1D array of type double on the GPU and sets the pointer back to a null pointer
src/CUDA_kernels/data_management_kernels.cu:    cudaFree(device_data);
src/CUDA_kernels/data_management_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/data_management_kernels.cu://deletes a 1D array of type int on the GPU and sets the pointer back to a null pointer
src/CUDA_kernels/data_management_kernels.cu:    cudaFree(device_data);
src/CUDA_kernels/data_management_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/data_management_kernels.cu://deletes an array of pointers on the GPU and sets the pointer back to a null pointer
src/CUDA_kernels/data_management_kernels.cu:    cudaFree(device_data);
src/CUDA_kernels/data_management_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/data_management_kernels.cu://moves data array host_data of type double to the GPU
src/CUDA_kernels/data_management_kernels.cu://returns the pointer *device_data to the data on the GPU
src/CUDA_kernels/data_management_kernels.cu://if device_data already exists on the GPU it will be freed first
src/CUDA_kernels/data_management_kernels.cu:  //delete the array if it has been previously allocated on the GPU
src/CUDA_kernels/data_management_kernels.cu:    //cudaFree(*device_data);
src/CUDA_kernels/data_management_kernels.cu:  cudaMalloc((void**)&device_data, bytes);
src/CUDA_kernels/data_management_kernels.cu:  cudaMemcpy(device_data, &host_data[0], bytes, cudaMemcpyHostToDevice);
src/CUDA_kernels/data_management_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/data_management_kernels.cu://moves data array host_data of type double to the GPU
src/CUDA_kernels/data_management_kernels.cu://returns the pointer *device_data to the data on the GPU
src/CUDA_kernels/data_management_kernels.cu://if device_data already exists on the GPU it will be freed first
src/CUDA_kernels/data_management_kernels.cu:  //delete the array if it has been previously allocated on the GPU
src/CUDA_kernels/data_management_kernels.cu:    //cudaFree(*device_data);
src/CUDA_kernels/data_management_kernels.cu:  if (alloc_memory) cudaMalloc((void**)&device_data, bytes);
src/CUDA_kernels/data_management_kernels.cu:  cudaMemcpy(device_data, &host_data[0], bytes, cudaMemcpyHostToDevice);
src/CUDA_kernels/data_management_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/data_management_kernels.cu://moves data array host_data of pointers to double arrays to the GPU
src/CUDA_kernels/data_management_kernels.cu://returns the pointer **device_data to the data on the GPU
src/CUDA_kernels/data_management_kernels.cu://if device_data already exists on the GPU it will be freed first
src/CUDA_kernels/data_management_kernels.cu:  //delete the array if it has been previously allocated on the GPU
src/CUDA_kernels/data_management_kernels.cu:    //cudaFree(*device_data);
src/CUDA_kernels/data_management_kernels.cu:  cudaMalloc((void***)&device_data, bytes);
src/CUDA_kernels/data_management_kernels.cu:  cudaMemcpy(device_data, &host_data[0], bytes, cudaMemcpyHostToDevice);
src/CUDA_kernels/data_management_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/data_management_kernels.cu:  cudaMemcpy(host_data.data(), device_data, bytes, cudaMemcpyDeviceToHost);
src/CUDA_kernels/data_management_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() ); 
src/CUDA_kernels/data_management_kernels.cu://moves data array host_data of type int to the GPU
src/CUDA_kernels/data_management_kernels.cu://returns the pointer *device_data to the data on the GPU
src/CUDA_kernels/data_management_kernels.cu://if device_data already exists on the GPU it will be freed first
src/CUDA_kernels/data_management_kernels.cu:  //delete the array if it has been previously allocated on the GPU
src/CUDA_kernels/data_management_kernels.cu:    //cudaFree(*device_data);
src/CUDA_kernels/data_management_kernels.cu:  cudaMalloc((void**)&device_data, bytes);
src/CUDA_kernels/data_management_kernels.cu:  cudaMemcpy(device_data, &host_data[0], bytes, cudaMemcpyHostToDevice);
src/CUDA_kernels/data_management_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/data_management_kernels.cu://sets all entries of a 2D double array on the GPU to 0
src/CUDA_kernels/data_management_kernels.cu:      cudaMemset(device_data[i], 0, nb_rows*sizeof(double));
src/CUDA_kernels/data_management_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/data_management_kernels.cu://sets all entries of a 1D double array on the GPU to 0
src/CUDA_kernels/data_management_kernels.cu://device_data holds the pointer to the GPU array
src/CUDA_kernels/data_management_kernels.cu:    cudaMemset(device_data, 0, nb_points*sizeof(double));
src/CUDA_kernels/data_management_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/data_management_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/convolution_kernels.cu:      printf("Not enough memory on GPU! %d %d %d %lu\n", 
src/CUDA_kernels/convolution_kernels.cu:__host__ void SpectralBands::convolveSpectrumGPU(double* spectrum, double* spectrum_processed_dev)
src/CUDA_kernels/convolution_kernels.cu:    spectral_grid->wavelength_list_gpu, 
src/CUDA_kernels/convolution_kernels.cu:  cudaDeviceSynchronize(); 
src/CUDA_kernels/convolution_kernels.cu:  gpuErrchk( cudaPeekAtLastError() ); 
src/CUDA_kernels/short_characteristics_kernels.cu:  double* model_spectrum_gpu,
src/CUDA_kernels/short_characteristics_kernels.cu:    model_spectrum_gpu[tid] = 
src/CUDA_kernels/short_characteristics_kernels.cu:void ShortCharacteristics::calcSpectrumGPU(
src/CUDA_kernels/short_characteristics_kernels.cu:    spectral_grid->wavenumber_list_gpu,
src/CUDA_kernels/short_characteristics_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/short_characteristics_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/stellar_contamination.cu:__host__ void StellarContamination::modifySpectrumGPU(
src/CUDA_kernels/stellar_contamination.cu:  double* spectrum_gpu)
src/CUDA_kernels/stellar_contamination.cu:  if (spectrum_phot_gpu == nullptr)
src/CUDA_kernels/stellar_contamination.cu:    allocateOnDevice(spectrum_phot_gpu, nb_spectral_points);
src/CUDA_kernels/stellar_contamination.cu:  stellar_model->calcFluxGPU(stellar_param, spectrum_phot_gpu);
src/CUDA_kernels/stellar_contamination.cu:    if (spectrum_fac_gpu == nullptr)
src/CUDA_kernels/stellar_contamination.cu:      allocateOnDevice(spectrum_fac_gpu, nb_spectral_points);
src/CUDA_kernels/stellar_contamination.cu:    stellar_model->calcFluxGPU(stellar_param, spectrum_fac_gpu);
src/CUDA_kernels/stellar_contamination.cu:    if (spectrum_spot_gpu == nullptr)
src/CUDA_kernels/stellar_contamination.cu:      allocateOnDevice(spectrum_spot_gpu, nb_spectral_points);
src/CUDA_kernels/stellar_contamination.cu:    stellar_model->calcFluxGPU(stellar_param, spectrum_spot_gpu);
src/CUDA_kernels/stellar_contamination.cu:    spectrum_gpu,
src/CUDA_kernels/stellar_contamination.cu:    spectrum_phot_gpu,
src/CUDA_kernels/stellar_contamination.cu:    spectrum_fac_gpu,
src/CUDA_kernels/stellar_contamination.cu:    spectrum_spot_gpu,
src/CUDA_kernels/stellar_contamination.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/stellar_contamination.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/stellar_contamination.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/transmission_spectrum_kernels.cu:__host__ void  TransmissionModel::calcTransitDepthGPU(
src/CUDA_kernels/transmission_spectrum_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/transmission_spectrum_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/transmission_spectrum_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/rayleigh_scattering_kernels.cu:__host__ void GasCORayleigh::calcRayleighCrossSectionsGPU(
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/rayleigh_scattering_kernels.cu:    spectral_grid->wavelength_list_gpu,
src/CUDA_kernels/rayleigh_scattering_kernels.cu:    spectral_grid->wavenumber_list_gpu,
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/rayleigh_scattering_kernels.cu:__host__ void GasCO2Rayleigh::calcRayleighCrossSectionsGPU(
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/rayleigh_scattering_kernels.cu:    spectral_grid->wavelength_list_gpu,
src/CUDA_kernels/rayleigh_scattering_kernels.cu:    spectral_grid->wavenumber_list_gpu,
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/rayleigh_scattering_kernels.cu:__host__ void GasHRayleigh::calcRayleighCrossSectionsGPU(
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/rayleigh_scattering_kernels.cu:    spectral_grid->wavelength_list_gpu,
src/CUDA_kernels/rayleigh_scattering_kernels.cu:    spectral_grid->wavenumber_list_gpu,
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/rayleigh_scattering_kernels.cu:__host__ void GasH2Rayleigh::calcRayleighCrossSectionsGPU(
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/rayleigh_scattering_kernels.cu:    spectral_grid->wavelength_list_gpu,
src/CUDA_kernels/rayleigh_scattering_kernels.cu:    spectral_grid->wavenumber_list_gpu,
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/rayleigh_scattering_kernels.cu:__host__ void GasCH4Rayleigh::calcRayleighCrossSectionsGPU(
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/rayleigh_scattering_kernels.cu:    spectral_grid->wavelength_list_gpu,
src/CUDA_kernels/rayleigh_scattering_kernels.cu:    spectral_grid->wavenumber_list_gpu,
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/rayleigh_scattering_kernels.cu:__host__ void GasH2ORayleigh::calcRayleighCrossSectionsGPU(
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/rayleigh_scattering_kernels.cu:    spectral_grid->wavelength_list_gpu,
src/CUDA_kernels/rayleigh_scattering_kernels.cu:    spectral_grid->wavenumber_list_gpu,
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/rayleigh_scattering_kernels.cu:__host__ void GasHeRayleigh::calcRayleighCrossSectionsGPU(
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/rayleigh_scattering_kernels.cu:    spectral_grid->wavelength_list_gpu,
src/CUDA_kernels/rayleigh_scattering_kernels.cu:    spectral_grid->wavenumber_list_gpu,
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/rayleigh_scattering_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/secondary_eclipse_bb.cu:__host__ void SecondaryEclipseBlackBodyModel::calcPlanetSpectrumGPU(
src/CUDA_kernels/secondary_eclipse_bb.cu:    spectral_grid->wavenumber_list_gpu,
src/CUDA_kernels/secondary_eclipse_bb.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/secondary_eclipse_bb.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/secondary_eclipse_bb.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/secondary_eclipse_bb.cu:__host__ void SecondaryEclipseBlackBodyModel::calcSecondaryEclipseGPU(
src/CUDA_kernels/secondary_eclipse_bb.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/secondary_eclipse_bb.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/secondary_eclipse_bb.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/stellar_spectrum_grid.cu:__host__ void StellarSpectrumGrid::calcFluxGPU(
src/CUDA_kernels/stellar_spectrum_grid.cu:  double* spectrum_gpu)
src/CUDA_kernels/stellar_spectrum_grid.cu:    grid[t_i.first][g_i.first][m_i.first]->spectrum_gpu,    //c000
src/CUDA_kernels/stellar_spectrum_grid.cu:    grid[t_i.second][g_i.first][m_i.first]->spectrum_gpu,   //c100
src/CUDA_kernels/stellar_spectrum_grid.cu:    grid[t_i.first][g_i.second][m_i.first]->spectrum_gpu,   //c010
src/CUDA_kernels/stellar_spectrum_grid.cu:    grid[t_i.second][g_i.second][m_i.first]->spectrum_gpu,  //c110
src/CUDA_kernels/stellar_spectrum_grid.cu:    grid[t_i.first][g_i.first][m_i.second]->spectrum_gpu,   //c001
src/CUDA_kernels/stellar_spectrum_grid.cu:    grid[t_i.second][g_i.first][m_i.second]->spectrum_gpu,  //c101
src/CUDA_kernels/stellar_spectrum_grid.cu:    grid[t_i.first][g_i.second][m_i.second]->spectrum_gpu,  //c011
src/CUDA_kernels/stellar_spectrum_grid.cu:    grid[t_i.second][g_i.second][m_i.second]->spectrum_gpu, //c111
src/CUDA_kernels/stellar_spectrum_grid.cu:    spectrum_gpu);
src/CUDA_kernels/stellar_spectrum_grid.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/stellar_spectrum_grid.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/stellar_spectrum_grid.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/filter_response_kernels.cu:__host__ void Observation::applyFilterResponseGPU(double* spectrum)
src/CUDA_kernels/filter_response_kernels.cu:    spectral_grid->wavenumber_list_gpu,
src/CUDA_kernels/filter_response_kernels.cu:    filter_response_gpu,
src/CUDA_kernels/filter_response_kernels.cu:    filter_response_weight_gpu,
src/CUDA_kernels/filter_response_kernels.cu:  cudaDeviceSynchronize(); 
src/CUDA_kernels/filter_response_kernels.cu:  gpuErrchk( cudaPeekAtLastError() ); 
src/CUDA_kernels/cross_section_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/cross_section_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/cross_section_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/cross_section_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/cross_section_kernels.cu:  double log_pressure1_gpu = log_pressure1;
src/CUDA_kernels/cross_section_kernels.cu:  double log_pressure2_gpu = log_pressure2;
src/CUDA_kernels/cross_section_kernels.cu:  double temperature1_gpu = temperature1;
src/CUDA_kernels/cross_section_kernels.cu:  double temperature2_gpu = temperature2;
src/CUDA_kernels/cross_section_kernels.cu:  //the linear interpolation in the CUDA kernel will fail if the two temperatures or pressures are equal
src/CUDA_kernels/cross_section_kernels.cu:  //to avoid a bunch of if-statements in the CUDA kernel, we here simply offset one of the temperatures or pressures by a bit
src/CUDA_kernels/cross_section_kernels.cu:  if (temperature1_gpu == temperature2_gpu) temperature2_gpu += 1;
src/CUDA_kernels/cross_section_kernels.cu:  if (log_pressure1_gpu == log_pressure2_gpu) log_pressure2_gpu += 0.001;
src/CUDA_kernels/cross_section_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/cross_section_kernels.cu:                                     temperature1_gpu, temperature2_gpu,
src/CUDA_kernels/cross_section_kernels.cu:                                     log_pressure1_gpu, log_pressure2_gpu,
src/CUDA_kernels/cross_section_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/cross_section_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/cross_section_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/cross_section_kernels.cu:  double temperature1_gpu = temperature1;
src/CUDA_kernels/cross_section_kernels.cu:  double temperature2_gpu = temperature2;
src/CUDA_kernels/cross_section_kernels.cu:  //the linear interpolation in the CUDA kernel will fail if the two temperatures equal
src/CUDA_kernels/cross_section_kernels.cu:  //to avoid a bunch of if-statements in the CUDA kernel, we here simply offset one of the temperatures a bit
src/CUDA_kernels/cross_section_kernels.cu:  if (temperature2 == temperature1) temperature2_gpu += 1.0;
src/CUDA_kernels/cross_section_kernels.cu:                                                temperature1_gpu, temperature2_gpu,
src/CUDA_kernels/cross_section_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/cross_section_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/cross_section_kernels.cu:cudaDeviceSynchronize();
src/CUDA_kernels/cross_section_kernels.cu:cudaDeviceSynchronize();
src/CUDA_kernels/cross_section_kernels.cu:gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/log_like_kernels.cu:  cudaMalloc(&d_log_like, sizeof(double));
src/CUDA_kernels/log_like_kernels.cu:  cudaMemset(d_log_like, 0, sizeof(double));
src/CUDA_kernels/log_like_kernels.cu:    observation_data_gpu,
src/CUDA_kernels/log_like_kernels.cu:    observation_error_gpu,
src/CUDA_kernels/log_like_kernels.cu:    observation_likelihood_weight_gpu,
src/CUDA_kernels/log_like_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/log_like_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/log_like_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/log_like_kernels.cu:  cudaMemcpy(&h_log_like, d_log_like, sizeof(double), cudaMemcpyDeviceToHost);
src/CUDA_kernels/log_like_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/log_like_kernels.cu:  cudaFree(d_log_like);
src/CUDA_kernels/log_like_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/star_file_spectrum.cu:__host__ void StarSpectrumFile::calcFluxGPU(
src/CUDA_kernels/star_file_spectrum.cu:  double* spectrum_gpu)
src/CUDA_kernels/star_file_spectrum.cu:    spectrum_gpu);
src/CUDA_kernels/star_file_spectrum.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/star_file_spectrum.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/star_file_spectrum.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/band_integration_kernels.cu://spectrum_high_res is the pointer to the array on the GPU
src/CUDA_kernels/band_integration_kernels.cu:__host__ void SpectralBands::bandIntegrateSpectrumGPU(
src/CUDA_kernels/band_integration_kernels.cu:    spectral_grid->wavenumber_list_gpu,
src/CUDA_kernels/band_integration_kernels.cu:    spectral_grid->wavelength_list_gpu,
src/CUDA_kernels/band_integration_kernels.cu:  cudaDeviceSynchronize(); 
src/CUDA_kernels/band_integration_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/error_check.h:#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
src/CUDA_kernels/error_check.h:inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
src/CUDA_kernels/error_check.h:   if (code != cudaSuccess)
src/CUDA_kernels/error_check.h:      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
src/CUDA_kernels/secondary_eclipse_kernels.cu:__host__ void SecondaryEclipseModel::calcSecondaryEclipseGPU(
src/CUDA_kernels/secondary_eclipse_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/secondary_eclipse_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/secondary_eclipse_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/contribution_function_kernels.cu:__global__ void contributionFunctionDevice(double* contribution_function_gpu,
src/CUDA_kernels/contribution_function_kernels.cu:      contribution_function_gpu[i*nb_spectral_points + tid] = 2 * constants::pi * planckFunction(temperature_dev[i], wavenumber_list_dev[tid]) * (1.0 - layer_transmission) * cumulative_transmission;
src/CUDA_kernels/contribution_function_kernels.cu:      //printf("%d  %d  %f  %f  %f  %f\n", tid, i, optical_depth_layer, layer_transmission, cumulative_transmission, contribution_function_gpu[i*nb_spectral_points + tid]);
src/CUDA_kernels/contribution_function_kernels.cu:      contribution_function_gpu[i*nb_spectral_points + tid] = planckFunction(temperature_dev[i], wavenumber_list_dev[tid]) * weighting_function;*/
src/CUDA_kernels/contribution_function_kernels.cu:__host__ void contributionFunctionGPU(double* contribution_function_dev,
src/CUDA_kernels/contribution_function_kernels.cu:  cudaMalloc(&temperature_dev, bytes);
src/CUDA_kernels/contribution_function_kernels.cu:  cudaMemcpy(temperature_dev, &temperature[0], bytes, cudaMemcpyHostToDevice);
src/CUDA_kernels/contribution_function_kernels.cu:  cudaMalloc(&vertical_grid_dev, bytes);
src/CUDA_kernels/contribution_function_kernels.cu:  cudaMemcpy(vertical_grid_dev, &vertical_grid[0], bytes, cudaMemcpyHostToDevice);
src/CUDA_kernels/contribution_function_kernels.cu:  cudaThreadSynchronize();
src/CUDA_kernels/contribution_function_kernels.cu:  cudaThreadSynchronize();
src/CUDA_kernels/contribution_function_kernels.cu:  cudaFree(temperature_dev);
src/CUDA_kernels/contribution_function_kernels.cu:  cudaFree(vertical_grid_dev);
src/CUDA_kernels/contribution_function_kernels.cu:  gpuErrchk( cudaPeekAtLastError() ); 
src/CUDA_kernels/contribution_function_kernels.h:void contributionFunctionGPU(double* contribution_function_dev,
src/CUDA_kernels/cloud_model_kernels.cu:__host__ void CloudModel::convertOpticalDepthGPU(
src/CUDA_kernels/cloud_model_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/cloud_model_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/cloud_model_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/cloud_model_kernels.cu:__host__ void GreyCloudModel::opticalPropertiesGPU(const std::vector<double>& parameters, const Atmosphere& atmosphere,
src/CUDA_kernels/cloud_model_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/cloud_model_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/cloud_model_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() ); 
src/CUDA_kernels/cloud_model_kernels.cu:__host__ void KHCloudModel::opticalPropertiesGPU(
src/CUDA_kernels/cloud_model_kernels.cu:    spectral_grid->wavelength_list_gpu,
src/CUDA_kernels/cloud_model_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/cloud_model_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/cloud_model_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/cloud_model_kernels.cu:__host__ void PowerLawCloudModel::opticalPropertiesGPU(
src/CUDA_kernels/cloud_model_kernels.cu:    spectral_grid->wavelength_list_gpu,
src/CUDA_kernels/cloud_model_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/cloud_model_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/cloud_model_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/observation_kernels.cu:__host__ void Observation::addShiftToSpectrumGPU(
src/CUDA_kernels/observation_kernels.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/observation_kernels.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/observation_kernels.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/CUDA_kernels/star_blackbody.cu:__host__ void StarBlackBody::calcFluxGPU(
src/CUDA_kernels/star_blackbody.cu:  double* spectrum_gpu)
src/CUDA_kernels/star_blackbody.cu:    spectral_grid->wavenumber_list_gpu,
src/CUDA_kernels/star_blackbody.cu:    spectrum_gpu);
src/CUDA_kernels/star_blackbody.cu:  cudaDeviceSynchronize();
src/CUDA_kernels/star_blackbody.cu:  gpuErrchk( cudaPeekAtLastError() );
src/CUDA_kernels/star_blackbody.cu:  gpuErrchk( cudaDeviceSynchronize() );
src/cloud_model/power_law_cloud_model.cpp:#include "../CUDA_kernels/data_management_kernels.h"
src/cloud_model/kh_cloud_model.cpp:#include "../CUDA_kernels/data_management_kernels.h"
src/cloud_model/cloud_model.h:    virtual void opticalPropertiesGPU(
src/cloud_model/cloud_model.h:    void convertOpticalDepthGPU(
src/cloud_model/power_law_cloud_model.h:    virtual void opticalPropertiesGPU(
src/cloud_model/kh_cloud_model.h:    virtual void opticalPropertiesGPU(
src/cloud_model/grey_cloud_model.cpp:#include "../CUDA_kernels/data_management_kernels.h"
src/cloud_model/grey_cloud_model.h:    virtual void opticalPropertiesGPU(
src/spectral_grid/spectral_grid.h:    double * wavenumber_list_gpu = nullptr;   //corresponding pointer to data on the GPU 
src/spectral_grid/spectral_grid.h:    double * wavelength_list_gpu = nullptr;
src/spectral_grid/spectral_grid.cpp:#include "../CUDA_kernels/data_management_kernels.h"
src/spectral_grid/spectral_grid.cpp:  if (config->use_gpu)
src/spectral_grid/spectral_grid.cpp:  if (config->use_gpu == false)
src/spectral_grid/spectral_grid.cpp:  if (wavelength_list_gpu != nullptr)
src/spectral_grid/spectral_grid.cpp:    deleteFromDevice(wavelength_list_gpu);
src/spectral_grid/spectral_grid.cpp:    deleteFromDevice(wavenumber_list_gpu);
src/spectral_grid/spectral_grid.cpp:  moveToDevice(wavenumber_list_gpu, wavenumber_list);
src/spectral_grid/spectral_grid.cpp:  moveToDevice(wavelength_list_gpu, wavelength_list);
src/spectral_grid/spectral_grid.cpp:  if (config->use_gpu)
src/spectral_grid/spectral_grid.cpp:    deleteFromDevice(wavelength_list_gpu);
src/spectral_grid/spectral_grid.cpp:    deleteFromDevice(wavenumber_list_gpu);
src/spectral_grid/spectral_band.h:    void bandIntegrateSpectrumGPU(
src/spectral_grid/spectral_band.h:    void convolveSpectrumGPU(double* spectrum, double* spectrum_processed_dev);
src/spectral_grid/spectral_band.cpp:#include "../CUDA_kernels/data_management_kernels.h"
src/spectral_grid/spectral_band.cpp:  if (config->use_gpu == false) 
src/spectral_grid/spectral_band.cpp:  if (config->use_gpu)
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:        config->use_gpu)
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:        config->use_gpu,
src/forward_model/secondary_eclipse/secondary_eclipse.cpp://run the forward model with the help of the GPU
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:bool SecondaryEclipseModel::calcModelGPU(
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:  double* model_spectrum_gpu, 
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:  opacity_calc.calculateGPU(cloud_models, cloud_parameters);
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:  radiative_transfer->calcSpectrumGPU(
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:    opacity_calc.absorption_coeff_gpu, 
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:    model_spectrum_gpu);
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:  postProcessSpectrumGPU(
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:    model_spectrum_gpu, 
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:  stellar_model->calcFluxGPU(stellar_parameters, stellar_spectrum);
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:  postProcessSpectrumGPU(
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:  double* albedo_contribution_gpu = nullptr;
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:  double* albedo_contribution_bands_gpu = nullptr;
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:  //moveToDevice(albedo_contribution_gpu, albedo_contribution);
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:  calcSecondaryEclipseGPU(
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:    albedo_contribution_bands_gpu);
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:  deleteFromDevice(albedo_contribution_bands_gpu);
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:        observations[i].addShiftToSpectrumGPU(
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:  calcSecondaryEclipseGPU(
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:    model_spectrum_gpu, 
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:    model_spectrum_gpu, 
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:    albedo_contribution_gpu);
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:  deleteFromDevice(albedo_contribution_gpu);
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:void SecondaryEclipseModel::postProcessSpectrumGPU(
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:  double* model_spectrum_gpu, 
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:    observations[i].processModelSpectrumGPU(
src/forward_model/secondary_eclipse/secondary_eclipse.cpp:      model_spectrum_gpu, 
src/forward_model/secondary_eclipse/secondary_eclipse_init.cpp:#include "../../CUDA_kernels/data_management_kernels.h"
src/forward_model/secondary_eclipse/secondary_eclipse_test.cpp:#include "../../CUDA_kernels/data_management_kernels.h"
src/forward_model/secondary_eclipse/secondary_eclipse_test.cpp:bool SecondaryEclipseModel::testModel(const std::vector<double>& parameter, double* model_spectrum_gpu)
src/forward_model/secondary_eclipse/secondary_eclipse_test.cpp:  test_ok = testCPUvsGPU(parameter, model_spectrum_gpu);
src/forward_model/secondary_eclipse/secondary_eclipse_test.cpp:bool SecondaryEclipseModel::testCPUvsGPU(const std::vector<double>& parameter, double* model_spectrum_gpu)
src/forward_model/secondary_eclipse/secondary_eclipse_test.cpp:  //first we calculate the model on the GPU
src/forward_model/secondary_eclipse/secondary_eclipse_test.cpp:  std::cout << "Start test on GPU\n";
src/forward_model/secondary_eclipse/secondary_eclipse_test.cpp:  //pointer to the spectrum on the GPU
src/forward_model/secondary_eclipse/secondary_eclipse_test.cpp:  //intialise the high-res spectrum on the GPU (set it to 0)
src/forward_model/secondary_eclipse/secondary_eclipse_test.cpp:  intializeOnDevice(model_spectrum_gpu, spectral_grid->nbSpectralPoints());
src/forward_model/secondary_eclipse/secondary_eclipse_test.cpp:  calcModelGPU(parameter, model_spectrum_gpu, spectrum_bands_dev);
src/forward_model/secondary_eclipse/secondary_eclipse_test.cpp:  std::vector<double> spectrum_bands_gpu(nb_observation_points, 0);
src/forward_model/secondary_eclipse/secondary_eclipse_test.cpp:  moveToHost(spectrum_bands_dev, spectrum_bands_gpu);
src/forward_model/secondary_eclipse/secondary_eclipse_test.cpp:    difference[i] = std::abs(spectrum_bands_cpu[i] - spectrum_bands_gpu[i])/spectrum_bands_cpu[i];
src/forward_model/secondary_eclipse/secondary_eclipse_test.cpp:  std::cout << "Maximum difference of CPU vs GPU: " << max_diff << " at index " << max_diff_index << "\n";
src/forward_model/secondary_eclipse/secondary_eclipse_test.cpp:    //std::cout << i << "\t" << spectrum_bands_cpu[i] << "\t" << spectrum_bands_gpu[i] << "\n";
src/forward_model/secondary_eclipse/secondary_eclipse_post_process.cpp:#include "../../CUDA_kernels/data_management_kernels.h"
src/forward_model/secondary_eclipse/secondary_eclipse_post_process.cpp:#include "../../CUDA_kernels/contribution_function_kernels.h"
src/forward_model/secondary_eclipse/secondary_eclipse_post_process.cpp:  opacity_calc.calculateGPU(cloud_models, cloud_parameters);
src/forward_model/secondary_eclipse/secondary_eclipse_post_process.cpp:  //intialise the high-res spectrum on the GPU (set it to 0) 
src/forward_model/secondary_eclipse/secondary_eclipse_post_process.cpp:  contributionFunctionGPU(
src/forward_model/secondary_eclipse/secondary_eclipse_post_process.cpp:    opacity_calc.absorption_coeff_gpu,
src/forward_model/secondary_eclipse/secondary_eclipse_post_process.cpp:    spectral_grid->wavenumber_list_gpu,
src/forward_model/secondary_eclipse/secondary_eclipse.h:    virtual bool calcModelGPU(
src/forward_model/secondary_eclipse/secondary_eclipse.h:      double* model_spectrum_gpu);
src/forward_model/secondary_eclipse/secondary_eclipse.h:    void calcSecondaryEclipseGPU(
src/forward_model/secondary_eclipse/secondary_eclipse.h:    void postProcessSpectrumGPU(
src/forward_model/secondary_eclipse/secondary_eclipse.h:    bool testCPUvsGPU(const std::vector<double>& parameter, double* model_spectrum_gpu);
src/forward_model/atmosphere/atmosphere.cpp:#include "../../CUDA_kernels/data_management_kernels.h"
src/forward_model/atmosphere/atmosphere.cpp:  const bool use_gpu) : nb_grid_points(nb_grid_points_)
src/forward_model/atmosphere/atmosphere.cpp:  if (use_gpu)
src/forward_model/atmosphere/atmosphere.h:      const bool use_gpu);
src/forward_model/stellar_spectrum/sampled_stellar_spectrum.cpp:  const bool use_gpu)
src/forward_model/stellar_spectrum/sampled_stellar_spectrum.cpp:  if (use_gpu)
src/forward_model/stellar_spectrum/sampled_stellar_spectrum.cpp:    moveToDevice(spectrum_gpu, spectrum, true);
src/forward_model/stellar_spectrum/sampled_stellar_spectrum.cpp:  deleteFromDevice(spectrum_gpu);
src/forward_model/stellar_spectrum/stellar_spectrum.h:#include "../../CUDA_kernels/data_management_kernels.h"
src/forward_model/stellar_spectrum/stellar_spectrum.h:    virtual void calcFluxGPU(
src/forward_model/stellar_spectrum/stellar_spectrum.h:      double* spectrum_gpu) = 0;
src/forward_model/stellar_spectrum/stellar_spectrum_grid.h:      const bool use_gpu);
src/forward_model/stellar_spectrum/stellar_spectrum_grid.h:    double* spectrum_gpu = nullptr;
src/forward_model/stellar_spectrum/stellar_spectrum_grid.h:    virtual void calcFluxGPU(
src/forward_model/stellar_spectrum/stellar_spectrum_grid.h:      double* spectrum_gpu);
src/forward_model/stellar_spectrum/star_blackbody.h:    virtual void calcFluxGPU(
src/forward_model/stellar_spectrum/star_blackbody.h:      double* spectrum_gpu);
src/forward_model/stellar_spectrum/star_file_spectrum.h:#include "../../CUDA_kernels/data_management_kernels.h"
src/forward_model/stellar_spectrum/star_file_spectrum.h:    virtual void calcFluxGPU(
src/forward_model/stellar_spectrum/star_file_spectrum.h:      double* spectrum_gpu);
src/forward_model/modules/stellar_contamination.h:#include "../../CUDA_kernels/data_management_kernels.h"
src/forward_model/modules/stellar_contamination.h:      deleteFromDevice(spectrum_phot_gpu);
src/forward_model/modules/stellar_contamination.h:      deleteFromDevice(spectrum_fac_gpu);
src/forward_model/modules/stellar_contamination.h:      deleteFromDevice(spectrum_spot_gpu);
src/forward_model/modules/stellar_contamination.h:    virtual void modifySpectrumGPU(
src/forward_model/modules/stellar_contamination.h:      double* spectrum_gpu);
src/forward_model/modules/stellar_contamination.h:    double* spectrum_phot_gpu = nullptr;
src/forward_model/modules/stellar_contamination.h:    double* spectrum_fac_gpu = nullptr;
src/forward_model/modules/stellar_contamination.h:    double* spectrum_spot_gpu = nullptr;
src/forward_model/modules/module.h:#include "../../CUDA_kernels/data_management_kernels.h"
src/forward_model/modules/module.h:    virtual void modifySpectrumGPU(
src/forward_model/modules/module.h:      double* spectrum_gpu) = 0;
src/forward_model/forward_model.cpp:  if (config->use_gpu)
src/forward_model/forward_model.cpp:    calcModelGPU(model_parameter, spectrum_dev, spectrum_bands_dev);
src/forward_model/forward_model.h:#include "../CUDA_kernels/data_management_kernels.h"
src/forward_model/forward_model.h:    //calculate a model on the GPU
src/forward_model/forward_model.h:    virtual bool calcModelGPU(
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.h:    virtual bool calcModelGPU(
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.h:      double* model_spectrum_gpu);
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.h:    void calcSecondaryEclipseGPU(
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.h:    void calcPlanetSpectrumGPU(
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.h:    void postProcessSpectrumGPU(
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.h:    bool testCPUvsGPU(const std::vector<double>& parameter, double* model_spectrum_gpu);
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_init.cpp:#include "../../CUDA_kernels/data_management_kernels.h"
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_test.cpp:#include "../../CUDA_kernels/data_management_kernels.h"
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_test.cpp:bool SecondaryEclipseBlackBodyModel::testModel(const std::vector<double>& parameter, double* model_spectrum_gpu)
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_test.cpp:  test_ok = testCPUvsGPU(parameter, model_spectrum_gpu);
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_test.cpp:bool SecondaryEclipseBlackBodyModel::testCPUvsGPU(const std::vector<double>& parameter, double* model_spectrum_gpu)
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_test.cpp:  //first we calculate the model on the GPU
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_test.cpp:  std::cout << "Start test on GPU\n";
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_test.cpp:  //pointer to the spectrum on the GPU
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_test.cpp:  //intialise the high-res spectrum on the GPU (set it to 0)
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_test.cpp:  intializeOnDevice(model_spectrum_gpu, spectral_grid->nbSpectralPoints());
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_test.cpp:  calcModelGPU(parameter, model_spectrum_gpu, spectrum_bands_dev);
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_test.cpp:  std::vector<double> spectrum_bands_gpu(nb_observation_points, 0);
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_test.cpp:  moveToHost(spectrum_bands_dev, spectrum_bands_gpu);
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_test.cpp:    difference[i] = std::abs(spectrum_bands_cpu[i] - spectrum_bands_gpu[i])/spectrum_bands_cpu[i];
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_test.cpp:  std::cout << "Maximum difference of CPU vs GPU: " << max_diff << " at index " << max_diff_index << "\n";
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_test.cpp:    //std::cout << i << "\t" << spectrum_bands_cpu[i] << "\t" << spectrum_bands_gpu[i] << "\n";
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_post_process.cpp:#include "../../CUDA_kernels/data_management_kernels.h"
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb_post_process.cpp:#include "../../CUDA_kernels/contribution_function_kernels.h"
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp://run the forward model with the help of the GPU
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:bool SecondaryEclipseBlackBodyModel::calcModelGPU(
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:  double* model_spectrum_gpu, 
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:  calcPlanetSpectrumGPU(planet_temperature, model_spectrum_gpu);
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:  postProcessSpectrumGPU(
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:    model_spectrum_gpu, 
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:  stellar_model->calcFluxGPU(stellar_parameters, stellar_spectrum);
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:  postProcessSpectrumGPU(
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:  double* albedo_contribution_gpu = nullptr;
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:  double* albedo_contribution_bands_gpu = nullptr;
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:  //moveToDevice(albedo_contribution_gpu, albedo_contribution);
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:  calcSecondaryEclipseGPU(
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:    albedo_contribution_bands_gpu);
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:  deleteFromDevice(albedo_contribution_bands_gpu);
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:        observations[i].addShiftToSpectrumGPU(
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:  calcSecondaryEclipseGPU(
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:    model_spectrum_gpu, 
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:    model_spectrum_gpu, 
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:    albedo_contribution_gpu);
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:  deleteFromDevice(albedo_contribution_gpu);
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:void SecondaryEclipseBlackBodyModel::postProcessSpectrumGPU(
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:  double* model_spectrum_gpu, 
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:    observations[i].processModelSpectrumGPU(
src/forward_model/secondary_eclipse_bb/secondary_eclipse_bb.cpp:      model_spectrum_gpu, 
src/forward_model/transmission/transmission.cpp:        config->use_gpu)
src/forward_model/transmission/transmission.cpp:        config->use_gpu,
src/forward_model/transmission/transmission.cpp://run the forward model with the help of the GPU
src/forward_model/transmission/transmission.cpp:bool TransmissionModel::calcModelGPU(
src/forward_model/transmission/transmission.cpp:  double* model_spectrum_gpu, 
src/forward_model/transmission/transmission.cpp:  opacity_calc.calculateGPU(cloud_models, cloud_parameters);
src/forward_model/transmission/transmission.cpp:  if (cloud_extinction_gpu == nullptr)
src/forward_model/transmission/transmission.cpp:    allocateOnDevice(cloud_extinction_gpu, nb_grid_points*spectral_grid->nbSpectralPoints());
src/forward_model/transmission/transmission.cpp:    intializeOnDevice(cloud_extinction_gpu, nb_grid_points*spectral_grid->nbSpectralPoints());
src/forward_model/transmission/transmission.cpp:    cloud_models[0]->convertOpticalDepthGPU(
src/forward_model/transmission/transmission.cpp:      cloud_extinction_gpu);
src/forward_model/transmission/transmission.cpp:  calcTransitDepthGPU(
src/forward_model/transmission/transmission.cpp:    model_spectrum_gpu, 
src/forward_model/transmission/transmission.cpp:    opacity_calc.absorption_coeff_gpu, 
src/forward_model/transmission/transmission.cpp:    cloud_extinction_gpu,
src/forward_model/transmission/transmission.cpp:    m->modifySpectrumGPU(module_parameters, &atmosphere, model_spectrum_gpu);
src/forward_model/transmission/transmission.cpp:  postProcessSpectrumGPU(model_spectrum_gpu, model_spectrum_bands);
src/forward_model/transmission/transmission.cpp:        observations[i].addShiftToSpectrumGPU(
src/forward_model/transmission/transmission.cpp:void TransmissionModel::postProcessSpectrumGPU(
src/forward_model/transmission/transmission.cpp:  double* model_spectrum_gpu, 
src/forward_model/transmission/transmission.cpp:    observations[i].processModelSpectrumGPU(
src/forward_model/transmission/transmission.cpp:      model_spectrum_gpu, 
src/forward_model/transmission/transmission.cpp:  if (cloud_extinction_gpu != nullptr)
src/forward_model/transmission/transmission.cpp:    deleteFromDevice(cloud_extinction_gpu);
src/forward_model/transmission/transmission.h:    virtual bool calcModelGPU(
src/forward_model/transmission/transmission.h:      double* model_spectrum_gpu);
src/forward_model/transmission/transmission.h:    double* cloud_extinction_gpu = nullptr;
src/forward_model/transmission/transmission.h:    void postProcessSpectrumGPU(
src/forward_model/transmission/transmission.h:    void calcTransitDepthGPU(
src/forward_model/transmission/transmission.h:    bool testCPUvsGPU(
src/forward_model/transmission/transmission.h:      double* model_spectrum_gpu);
src/forward_model/transmission/transmission_test.cpp:#include "../../CUDA_kernels/data_management_kernels.h"
src/forward_model/transmission/transmission_test.cpp:bool TransmissionModel::testModel(const std::vector<double>& parameter, double* model_spectrum_gpu)
src/forward_model/transmission/transmission_test.cpp:  test_ok = testCPUvsGPU(parameter, model_spectrum_gpu);
src/forward_model/transmission/transmission_test.cpp:bool TransmissionModel::testCPUvsGPU(const std::vector<double>& parameter, double* model_spectrum_gpu)
src/forward_model/transmission/transmission_test.cpp:  //first we calculate the model on the GPU
src/forward_model/transmission/transmission_test.cpp:  std::cout << "Start test on GPU\n";
src/forward_model/transmission/transmission_test.cpp:  //pointer to the spectrum on the GPU
src/forward_model/transmission/transmission_test.cpp:  //intialise the high-res spectrum on the GPU (set it to 0)
src/forward_model/transmission/transmission_test.cpp:  intializeOnDevice(model_spectrum_gpu, spectral_grid->nbSpectralPoints());
src/forward_model/transmission/transmission_test.cpp:  calcModelGPU(parameter, model_spectrum_gpu, spectrum_bands_dev);
src/forward_model/transmission/transmission_test.cpp:  std::vector<double> spectrum_bands_gpu(nb_observation_points, 0);
src/forward_model/transmission/transmission_test.cpp:  moveToHost(spectrum_bands_dev, spectrum_bands_gpu);
src/forward_model/transmission/transmission_test.cpp:    difference[i] = std::abs(spectrum_bands_cpu[i] - spectrum_bands_gpu[i])/spectrum_bands_cpu[i];
src/forward_model/transmission/transmission_test.cpp:  std::cout << "Maximum difference of CPU vs GPU: " << max_diff << " at index " << max_diff_index << "\n";
src/forward_model/transmission/transmission_test.cpp:    //std::cout << i << "\t" << spectrum_bands_cpu[i] << "\t" << spectrum_bands_gpu[i] << "\n";
src/forward_model/emission/emission_test.cpp:#include "../../CUDA_kernels/data_management_kernels.h"
src/forward_model/emission/emission_test.cpp:bool EmissionModel::testModel(const std::vector<double>& parameter, double* model_spectrum_gpu)
src/forward_model/emission/emission_test.cpp:  test_ok = testCPUvsGPU(parameter, model_spectrum_gpu);
src/forward_model/emission/emission_test.cpp:bool EmissionModel::testCPUvsGPU(const std::vector<double>& parameter, double* model_spectrum_gpu)
src/forward_model/emission/emission_test.cpp:  //first we calculate the model on the GPU
src/forward_model/emission/emission_test.cpp:  std::cout << "Start test on GPU\n";
src/forward_model/emission/emission_test.cpp:  //pointer to the spectrum on the GPU
src/forward_model/emission/emission_test.cpp:  //intialise the high-res spectrum on the GPU (set it to 0)
src/forward_model/emission/emission_test.cpp:  intializeOnDevice(model_spectrum_gpu, spectral_grid->nbSpectralPoints());
src/forward_model/emission/emission_test.cpp:  calcModelGPU(parameter, model_spectrum_gpu, spectrum_bands_dev);
src/forward_model/emission/emission_test.cpp:  std::vector<double> spectrum_bands_gpu(nb_observation_points, 0);
src/forward_model/emission/emission_test.cpp:  moveToHost(spectrum_bands_dev, spectrum_bands_gpu);
src/forward_model/emission/emission_test.cpp:    difference[i] = std::abs(spectrum_bands_cpu[i] - spectrum_bands_gpu[i])/spectrum_bands_cpu[i];
src/forward_model/emission/emission_test.cpp:  std::cout << "Maximum difference of CPU vs GPU: " << max_diff << " at index " << max_diff_index << "\n";
src/forward_model/emission/emission_test.cpp:    //std::cout << i << "\t" << spectrum_bands_cpu[i] << "\t" << spectrum_bands_gpu[i] << "\n";
src/forward_model/emission/emission_post_process.cpp:#include "../../CUDA_kernels/data_management_kernels.h"
src/forward_model/emission/emission_post_process.cpp:#include "../../CUDA_kernels/contribution_function_kernels.h"
src/forward_model/emission/emission_post_process.cpp:  opacity_calc.calculateGPU(cloud_models, cloud_parameters);
src/forward_model/emission/emission_post_process.cpp:  //intialise the high-res spectrum on the GPU (set it to 0) 
src/forward_model/emission/emission_post_process.cpp:  contributionFunctionGPU(
src/forward_model/emission/emission_post_process.cpp:    opacity_calc.absorption_coeff_gpu,
src/forward_model/emission/emission_post_process.cpp:    spectral_grid->wavenumber_list_gpu,
src/forward_model/emission/emission.cpp:        config->use_gpu)
src/forward_model/emission/emission.cpp:        config->use_gpu,
src/forward_model/emission/emission.cpp://run the forward model with the help of the GPU
src/forward_model/emission/emission.cpp:bool EmissionModel::calcModelGPU(
src/forward_model/emission/emission.cpp:  double* model_spectrum_gpu, 
src/forward_model/emission/emission.cpp:  opacity_calc.calculateGPU(cloud_models, cloud_parameters);
src/forward_model/emission/emission.cpp:  radiative_transfer->calcSpectrumGPU(
src/forward_model/emission/emission.cpp:    opacity_calc.absorption_coeff_gpu, 
src/forward_model/emission/emission.cpp:    model_spectrum_gpu);
src/forward_model/emission/emission.cpp:  postProcessSpectrumGPU(model_spectrum_gpu, model_spectrum_bands);
src/forward_model/emission/emission.cpp:        observations[i].addShiftToSpectrumGPU(
src/forward_model/emission/emission.cpp:void EmissionModel::postProcessSpectrumGPU(
src/forward_model/emission/emission.cpp:  double* model_spectrum_gpu, 
src/forward_model/emission/emission.cpp:    observations[i].processModelSpectrumGPU(
src/forward_model/emission/emission.cpp:      model_spectrum_gpu, 
src/forward_model/emission/emission.h:    virtual bool calcModelGPU(
src/forward_model/emission/emission.h:      const std::vector<double>& parameter, double* model_spectrum_gpu);
src/forward_model/emission/emission.h:    void postProcessSpectrumGPU(
src/forward_model/emission/emission.h:    bool testCPUvsGPU(
src/forward_model/emission/emission.h:      const std::vector<double>& parameter, double* model_spectrum_gpu);
src/forward_model/flat_line/flat_line_post_process.cpp:#include "../../CUDA_kernels/data_management_kernels.h"
src/forward_model/flat_line/flat_line.h:    virtual bool calcModelGPU(
src/forward_model/flat_line/flat_line.h:      const std::vector<double>& parameter, double* model_spectrum_gpu);
src/forward_model/flat_line/flat_line.h:    void postProcessSpectrumGPU(
src/forward_model/flat_line/flat_line.h:    bool testCPUvsGPU(
src/forward_model/flat_line/flat_line.h:      const std::vector<double>& parameter, double* model_spectrum_gpu);
src/forward_model/flat_line/flat_line.cpp:#include "../../CUDA_kernels/data_management_kernels.h"
src/forward_model/flat_line/flat_line.cpp://run the forward model with the help of the GPU
src/forward_model/flat_line/flat_line.cpp:bool FlatLine::calcModelGPU(
src/forward_model/flat_line/flat_line.cpp:  double* model_spectrum_gpu, 
src/forward_model/flat_line/flat_line.cpp:  moveToDevice(model_spectrum_gpu, spectrum, false);
src/forward_model/flat_line/flat_line.cpp:  postProcessSpectrumGPU(model_spectrum_gpu, model_spectrum_bands);
src/forward_model/flat_line/flat_line.cpp:void FlatLine::postProcessSpectrumGPU(
src/forward_model/flat_line/flat_line.cpp:  double* model_spectrum_gpu, 
src/forward_model/flat_line/flat_line.cpp:    observations[i].processModelSpectrumGPU(
src/forward_model/flat_line/flat_line.cpp:      model_spectrum_gpu, 
src/forward_model/flat_line/flat_line_test.cpp:#include "../../CUDA_kernels/data_management_kernels.h"
src/forward_model/flat_line/flat_line_test.cpp:bool FlatLine::testModel(const std::vector<double>& parameter, double* model_spectrum_gpu)
src/forward_model/flat_line/flat_line_test.cpp:  test_ok = testCPUvsGPU(parameter, model_spectrum_gpu);
src/forward_model/flat_line/flat_line_test.cpp:bool FlatLine::testCPUvsGPU(const std::vector<double>& parameter, double* model_spectrum_gpu)
src/forward_model/flat_line/flat_line_test.cpp:  //first we calculate the model on the GPU
src/forward_model/flat_line/flat_line_test.cpp:  std::cout << "Start test on GPU\n";
src/forward_model/flat_line/flat_line_test.cpp:  //pointer to the spectrum on the GPU
src/forward_model/flat_line/flat_line_test.cpp:  //intialise the high-res spectrum on the GPU (set it to 0)
src/forward_model/flat_line/flat_line_test.cpp:  intializeOnDevice(model_spectrum_gpu, spectral_grid->nbSpectralPoints());
src/forward_model/flat_line/flat_line_test.cpp:  calcModelGPU(parameter, model_spectrum_gpu, spectrum_bands_dev);
src/forward_model/flat_line/flat_line_test.cpp:  std::vector<double> spectrum_bands_gpu(nb_observation_points, 0);
src/forward_model/flat_line/flat_line_test.cpp:  moveToHost(spectrum_bands_dev, spectrum_bands_gpu);
src/forward_model/flat_line/flat_line_test.cpp:    difference[i] = std::abs(spectrum_bands_cpu[i] - spectrum_bands_gpu[i])/spectrum_bands_cpu[i];
src/forward_model/flat_line/flat_line_test.cpp:  std::cout << "Maximum difference of CPU vs GPU: " << max_diff << " at index " << max_diff_index << "\n";
src/forward_model/flat_line/flat_line_test.cpp:    //std::cout << i << "\t" << spectrum_bands_cpu[i] << "\t" << spectrum_bands_gpu[i] << "\n";
src/radiative_transfer/radiative_transfer.h:    virtual void calcSpectrumGPU(
src/radiative_transfer/short_characteristics.h:    virtual void calcSpectrumGPU(
src/radiative_transfer/discrete_ordinate.h:      const bool use_gpu);
src/radiative_transfer/discrete_ordinate.h:    virtual void calcSpectrumGPU(
src/radiative_transfer/discrete_ordinate.h:        std::cout << "Sorry, CDISORT has no GPU option :(\n";
src/radiative_transfer/discrete_ordinate.cpp:  const bool use_gpu)
src/radiative_transfer/discrete_ordinate.cpp:  if (use_gpu)
src/radiative_transfer/discrete_ordinate.cpp:    std::string error_message = "Radiative transfer model CDISORT cannot run on the GPU\n";
src/radiative_transfer/select_radiative_transfer.h:          config->use_gpu); 

```
