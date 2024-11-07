# https://github.com/YinanZhao21/SOAP_GPU

```console
SOAP_code/config_conti_phoenix.cfg:# this is the directory for data preprocess of PHOENIX spectral library. It's also the input directory of SOAP-GPU.
SOAP_code/config_conti_phoenix.cfg:# wavelength array for SOAP-GPU
SOAP_code/Plot_animation.py:frame_name = file_name_prefix+'_GPU_SOAP_RV_data.npz'
SOAP_code/Plot_animation.py:output_gif=file_name_prefix+'_GPU_SOAP_visual.gif'
SOAP_code/install_SOAP_GPU.sh:echo "Start to install SOAP_GPU"
SOAP_code/run_all_SOAP_evo.py:print("GPU Calculation time (s):", start2 - start)
SOAP_code/run_all_SOAP_evo.py:    out_name = output_prefix+'/'+file_name_prefix+'_'+'RV_'+types+'_GPU.npz'
SOAP_code/run_all_SOAP_evo.py:data1 = np.load(output_prefix+'/'+file_name_prefix+'_'+'RV_tot_GPU.npz')
SOAP_code/run_all_SOAP_evo.py:data2 = np.load(output_prefix+'/'+file_name_prefix+'_'+'RV_flux_GPU.npz')
SOAP_code/run_all_SOAP_evo.py:data3 = np.load(output_prefix+'/'+file_name_prefix+'_'+'RV_bconv_GPU.npz')
SOAP_code/run_all_SOAP_evo.py:out_name = output_prefix+'/'+file_name_prefix+'_GPU_SOAP_RV_data.npz'
SOAP_code/Rassine_file/Rassine_config.py:#spectrum_name = '/Users/yinanzhao/SOAP2_with_spectrum/StarSpot/GPU_SOAP/GPU_SOAP_V2/Sun_5678.p' # full path of your spectrum pickle/csv file
SOAP_code/run.sbatch:#SBATCH --partition=debug-gpu
SOAP_code/run.sbatch:#SBATCH --gpus=1
SOAP_code/run.sbatch:module load fosscuda/2020a
SOAP_code/run.sbatch:# if you need to know the allocated CUDA device, you can obtain it here:
SOAP_code/run.sbatch:echo "I: CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
SOAP_code/run_all_SOAP.py:print("GPU Calculation time (s):", start2 - start)
SOAP_code/run_all_SOAP.py:    out_name = output_prefix+'/'+file_name_prefix+'_'+'RV_'+types+'_GPU.npz'
SOAP_code/run_all_SOAP.py:data1 = np.load(output_prefix+'/'+file_name_prefix+'_'+'RV_tot_GPU.npz')
SOAP_code/run_all_SOAP.py:data2 = np.load(output_prefix+'/'+file_name_prefix+'_'+'RV_flux_GPU.npz')
SOAP_code/run_all_SOAP.py:data3 = np.load(output_prefix+'/'+file_name_prefix+'_'+'RV_bconv_GPU.npz')
SOAP_code/run_all_SOAP.py:out_name = output_prefix+'/'+file_name_prefix+'_GPU_SOAP_RV_data.npz'
SOAP_code/SOAP_resolution.cu:#include <cuda.h>
SOAP_code/SOAP_resolution.cu:__global__ void conv_inst(double *wavelength, double *spec, double *sigmas, double *spec_out1, int wid, int vector_size,double *quiet_sun_gpu){
SOAP_code/SOAP_resolution.cu:            spec_out1[index] += wavelength_step_median*(1.0 - (quiet_sun_gpu[m]-spec[m])) * 1./(sigmas[index]*sqrt(2.*pi))*exp(-(m-index)*(m-index)*0.005*0.005/(2*sigmas[index]*sigmas[index]));
SOAP_code/SOAP_resolution.cu:__global__ void spec_diff(double *quiet_sun_gpu, double *active_spec, double *spec_difference, int vector_size){
SOAP_code/SOAP_resolution.cu:    spec_difference[index] = quiet_sun_gpu[index] - active_spec[index];
SOAP_code/SOAP_resolution.cu:void lower_resolution_gpu(double *wavelength, double *spectrum, double *spectrum_low_reso, int vector_size, double sigma_resolution, string phi_name, double *quiet_sun_gpu, string out_type,string final_spec_dir, int no_block,
SOAP_code/SOAP_resolution.cu:    double *sigma_wav, *wav_gpu, *spec_gpu,  *spec_out1;
SOAP_code/SOAP_resolution.cu:    cudaMalloc( (void**)&sigma_wav, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_resolution.cu:    cudaMalloc( (void**)&wav_gpu, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_resolution.cu:    cudaMalloc( (void**)&spec_gpu, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_resolution.cu:    cudaMalloc( (void**)&spec_out1, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_resolution.cu:    cudaMalloc( (void**)&spec_difference, (vector_size) * sizeof(double) ) ;
SOAP_code/SOAP_resolution.cu:    cudaMalloc( (void**)&spec_final_output, (vector_size) * sizeof(double) ) ;
SOAP_code/SOAP_resolution.cu:    cudaMemcpy( wav_gpu, wavelength, vector_size * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_resolution.cu:    cudaMemcpy( spec_gpu, spectrum, vector_size * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_resolution.cu:    wav_sigma<<<no_block, no_thread>>>(wav_gpu, sigma_wav, sigma_resolution);
SOAP_code/SOAP_resolution.cu:    conv_inst<<<no_block, no_thread>>>(wav_gpu, spec_gpu, sigma_wav, spec_out1, nb_pixel, vector_size, quiet_sun_gpu);
SOAP_code/SOAP_resolution.cu:    spec_diff<<<no_block, no_thread>>>(quiet_sun_gpu, spec_gpu, spec_difference, vector_size);
SOAP_code/SOAP_resolution.cu:    cudaMalloc( (void**)&max_val_dev, 1 * sizeof(double) ) ;
SOAP_code/SOAP_resolution.cu:    cudaMalloc( (void**)&max_idx_dev, 1 * sizeof(int) ) ;
SOAP_code/SOAP_resolution.cu:    cudaMalloc( (void**)&min_val_dev, 1 * sizeof(double) ) ;
SOAP_code/SOAP_resolution.cu:    cudaMalloc( (void**)&min_idx_dev, 1 * sizeof(int) ) ;
SOAP_code/SOAP_resolution.cu:    cudaMemcpy(max_val, max_val_dev, 1 * sizeof(double), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_resolution.cu:    cudaMemcpy(max_idx, max_idx_dev, 1 * sizeof(int), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_resolution.cu:    cudaMemcpy(min_val, min_val_dev, 1 * sizeof(double), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_resolution.cu:    cudaMemcpy(min_idx, min_idx_dev, 1 * sizeof(int), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_resolution.cu:    cudaMemcpy(spectrum_low_reso, spec_final_output, vector_size * sizeof(double), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_resolution.cu:    cudaFree(sigma_wav ) ;
SOAP_code/SOAP_resolution.cu:    cudaFree(wav_gpu) ;
SOAP_code/SOAP_resolution.cu:    cudaFree(spec_gpu) ;
SOAP_code/SOAP_resolution.cu:    cudaFree(spec_out1) ;
SOAP_code/SOAP_resolution.cu:    cudaFree(spec_final_output) ;
SOAP_code/SOAP_resolution.cu:    cudaFree(spec_difference) ;
SOAP_code/SOAP_resolution.cu:    double *quiet_sun_ptr, *quiet_sun_gpu;
SOAP_code/SOAP_resolution.cu:    cudaMalloc( (void**)&quiet_sun_gpu, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_resolution.cu:    cudaMemcpy( quiet_sun_gpu, quiet_sun_ptr, vector_size * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_resolution.cu:    double *quiet_sun_ptr_SED, *quiet_sun_gpu_SED;
SOAP_code/SOAP_resolution.cu:    cudaMalloc( (void**)&quiet_sun_gpu_SED, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_resolution.cu:    cudaMemcpy( quiet_sun_gpu_SED, quiet_sun_ptr_SED, vector_size * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_resolution.cu:        lower_resolution_gpu(quiet_wave_ptr, f_spot_tot, spectrum_low_reso_tot, vector_size, sigma_resolution, phi_name, quiet_sun_gpu_SED, type_tot, final_spec_dir, no_block, no_thread, out_put_prefix);
SOAP_code/SOAP_resolution.cu:        lower_resolution_gpu(quiet_wave_ptr, f_spot_flux, spectrum_low_reso_flux, vector_size, sigma_resolution, phi_name, quiet_sun_gpu_SED, type_flux, final_spec_dir, no_block, no_thread, out_put_prefix);
SOAP_code/SOAP_resolution.cu:        lower_resolution_gpu(quiet_wave_ptr, f_spot_bconv, spectrum_low_reso_bconv, vector_size, sigma_resolution, phi_name, quiet_sun_gpu, type_bconv, final_spec_dir, no_block, no_thread, out_put_prefix);
SOAP_code/SOAP_initialize.cu:#include <cuda.h>
SOAP_code/SOAP_initialize.cu:    cudaMalloc( (void**)&dev_vrad, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_initialize.cu:    cudaMalloc( (void**)&dev_grad, vector_size * no_spec* sizeof(double) ) ;
SOAP_code/SOAP_initialize.cu:    cudaMalloc( (void**)&dev_spec_out, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_initialize.cu:    cudaFree(dev_spec_out);
SOAP_code/SOAP_initialize.cu:    cudaFree(dev_vrad);
SOAP_code/SOAP_initialize.cu:    cudaFree(dev_grad);
SOAP_code/SOAP_initialize.cu:void lower_resolution_gpu(double *wavelength, double *spectrum, double *spectrum_low_reso, int vector_size, double sigma_resolution, double wavelength_step_median, double window_width, int no_block, int no_thread, string out_put_prefix){
SOAP_code/SOAP_initialize.cu:    double *sigma_wav, *wav_gpu, *spec_gpu,  *spec_out1, *spec_final_output;
SOAP_code/SOAP_initialize.cu:    cudaMalloc( (void**)&sigma_wav, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_initialize.cu:    cudaMalloc( (void**)&wav_gpu, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_initialize.cu:    cudaMalloc( (void**)&spec_gpu, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_initialize.cu:    cudaMalloc( (void**)&spec_out1, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_initialize.cu:    cudaMalloc( (void**)&spec_final_output, (vector_size) * sizeof(double) ) ;
SOAP_code/SOAP_initialize.cu:    cudaMemcpy( wav_gpu, wavelength, vector_size * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_initialize.cu:    cudaMemcpy( spec_gpu, spectrum, vector_size * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_initialize.cu:    wav_sigma<<<no_block,no_thread>>>(wav_gpu, sigma_wav, sigma_resolution);
SOAP_code/SOAP_initialize.cu:    conv_inst<<<no_block,no_thread>>>(wav_gpu, spec_gpu, sigma_wav, spec_out1, nb_pixel, vector_size, wavelength_step_median);
SOAP_code/SOAP_initialize.cu:    cudaMalloc( (void**)&max_val_dev, 1 * sizeof(double) ) ;
SOAP_code/SOAP_initialize.cu:    cudaMalloc( (void**)&max_idx_dev, 1 * sizeof(int) ) ;
SOAP_code/SOAP_initialize.cu:    cudaMalloc( (void**)&min_val_dev, 1 * sizeof(double) ) ;
SOAP_code/SOAP_initialize.cu:    cudaMalloc( (void**)&min_idx_dev, 1 * sizeof(int) ) ;
SOAP_code/SOAP_initialize.cu:    reduceMinIdxOptimizedShared<<<no_block, no_thread>>>(spec_gpu, vector_size, min_val_dev, min_idx_dev);
SOAP_code/SOAP_initialize.cu:    cudaMemcpy(max_val, max_val_dev, 1 * sizeof(double), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_initialize.cu:    cudaMemcpy(max_idx, max_idx_dev, 1 * sizeof(int), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_initialize.cu:    cudaMemcpy(min_val, min_val_dev, 1 * sizeof(double), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_initialize.cu:    cudaMemcpy(min_idx, min_idx_dev, 1 * sizeof(int), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_initialize.cu:    cudaMemcpy(spectrum_low_reso, spec_final_output, vector_size * sizeof(double), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_initialize.cu:    cudaMalloc( (void**)&dev_wave, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_initialize.cu:    cudaMalloc( (void**)&dev_spec, vector_size*no_spec * sizeof(double) ) ;
SOAP_code/SOAP_initialize.cu:    cudaMemcpy( dev_wave, wave_array, vector_size * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_initialize.cu:    cudaMemcpy( dev_spec, spec_array, vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_initialize.cu:    cudaMalloc( (void**)&dev_sum_star, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_initialize.cu:    cudaMemcpy( out_array, dev_sum_star, vector_size  * sizeof(double), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_initialize.cu:    lower_resolution_gpu(wave_array, out_array, spectrum_low_reso, vector_size, sigma_resolution,wavelength_step_median, window_width, no_block, no_thread, out_put_prefix);
SOAP_code/example/run_all_SOAP_obs.py:print("GPU Calculation time (s):", start2 - start)
SOAP_code/example/run_all_SOAP_obs.py:    out_name = output_prefix+'/'+file_name_prefix+'_'+'RV_'+types+'_GPU.npz'
SOAP_code/example/run_all_SOAP_obs.py:data1 = np.load(output_prefix+'/'+file_name_prefix+'_'+'RV_tot_GPU.npz')
SOAP_code/example/run_all_SOAP_obs.py:data2 = np.load(output_prefix+'/'+file_name_prefix+'_'+'RV_flux_GPU.npz')
SOAP_code/example/run_all_SOAP_obs.py:data3 = np.load(output_prefix+'/'+file_name_prefix+'_'+'RV_bconv_GPU.npz')
SOAP_code/example/run_all_SOAP_obs.py:out_name = output_prefix+'/'+file_name_prefix+'_GPU_SOAP_RV_data.npz'
SOAP_code/SOAP_integration_solar.cu:#include <cuda.h>
SOAP_code/SOAP_integration_solar.cu:__global__ void spot_phase_gpu(double *xyz, double inclination, int nrho, double phase,double *xyz2)
SOAP_code/SOAP_integration_solar.cu:__global__ void spot_init_gpu(double s, double longitude, double latitude, double inclination,
SOAP_code/SOAP_integration_solar.cu:    cudaMalloc( (void**)&spectra_SED_dev,  vector_size*no_spec * sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:    cudaMalloc( (void**)&spot_SED_dev,  vector_size*no_spec * sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:    cudaMalloc( (void**)&faculae_SED_dev,  vector_size*no_spec * sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:    cudaMemcpy( spectra_SED_dev, spectra_SED_array, vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_integration_solar.cu:    cudaMemcpy( spot_SED_dev, spot_SED_array,  vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_integration_solar.cu:    cudaMemcpy( faculae_SED_dev, faculae_SED_array,  vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_integration_solar.cu:    cudaMalloc( (void**)&dev_wave, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:    cudaMalloc( (void**)&dev_spec,  vector_size*no_spec* sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:    cudaMemcpy( dev_wave, wave_array, vector_size * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_integration_solar.cu:    cudaMemcpy( dev_spec, spec_array,  vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_integration_solar.cu:    cudaMalloc( (void**)&dev_wave_spot, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:    cudaMalloc( (void**)&dev_spec_spot,  vector_size*no_spec * sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:    cudaMalloc( (void**)&dev_spec_faculae,  vector_size*no_spec * sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:    cudaMemcpy( dev_wave_spot, wave_array_spot, vector_size * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_integration_solar.cu:    cudaMemcpy( dev_spec_spot, spec_array_spot,  vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_integration_solar.cu:    cudaMemcpy( dev_spec_faculae, spec_array_faculae,  vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_integration_solar.cu:    cudaMalloc( (void**)&dev_grad, vector_size*no_spec  * sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:    cudaMalloc( (void**)&dev_grad_spot, vector_size*no_spec * sizeof(double) );
SOAP_code/SOAP_integration_solar.cu:    cudaMalloc( (void**)&dev_grad_faculae, vector_size*no_spec * sizeof(double) );
SOAP_code/SOAP_integration_solar.cu:    cudaMalloc( (void**)&dev_grad_SED, vector_size*no_spec  * sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:        cudaMalloc( (void**)&dev_vrad, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:        cudaMalloc( (void**)&dev_spec_out, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:        cudaMalloc( (void**)&dev_spec_out_spot, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:        cudaMalloc( (void**)&dev_spec_SED_out, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:        cudaMalloc( (void**)&dev_f_spot_flux, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:        cudaMalloc( (void**)&dev_f_spot_bconv, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:        cudaMalloc( (void**)&dev_f_spot_tot, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:        cudaMalloc( (void**)&dev_intensity, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration_solar.cu:            cudaMalloc( (void**)&xyz, 3*nrho* sizeof(double) );
SOAP_code/SOAP_integration_solar.cu:            cudaMalloc( (void**)&xyz2, 3*nrho* sizeof(double) );
SOAP_code/SOAP_integration_solar.cu:            cudaMalloc( (void**)&rho, nrho* sizeof(double) );
SOAP_code/SOAP_integration_solar.cu:            cudaMalloc( (void**)&xyz22, 3*nrho* sizeof(double) );
SOAP_code/SOAP_integration_solar.cu:            cudaMalloc( (void**)&counton, nrho* sizeof(int) );
SOAP_code/SOAP_integration_solar.cu:            cudaMalloc( (void**)&countoff, nrho* sizeof(int) );
SOAP_code/SOAP_integration_solar.cu:            cudaMalloc( (void**)&maxminy_array, nrho* sizeof(double) );
SOAP_code/SOAP_integration_solar.cu:            cudaMalloc( (void**)&maxminz_array, nrho* sizeof(double) );
SOAP_code/SOAP_integration_solar.cu:            cudaMalloc( (void**)&boundaries, 5*sizeof(int) );
SOAP_code/SOAP_integration_solar.cu:            spot_init_gpu<<<nrho, 3>>>(size_maps[s_num], long_maps[s_num], lat_maps[s_num], inclination, nrho, xyz, xyz2, rho);
SOAP_code/SOAP_integration_solar.cu:            spot_phase_gpu<<<nrho, 3>>>(xyz, inclination, nrho, psi[ipsi], xyz22);
SOAP_code/SOAP_integration_solar.cu:            cudaMemcpy(boundaries_cpu, boundaries, 5*sizeof(int), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_integration_solar.cu:            cudaFree(counton );
SOAP_code/SOAP_integration_solar.cu:            cudaFree(countoff );
SOAP_code/SOAP_integration_solar.cu:            cudaFree( maxminy_array );
SOAP_code/SOAP_integration_solar.cu:            cudaFree( maxminz_array );
SOAP_code/SOAP_integration_solar.cu:            cudaFree(boundaries);
SOAP_code/SOAP_integration_solar.cu:            cudaFree(rho);
SOAP_code/SOAP_integration_solar.cu:            cudaFree(xyz2);
SOAP_code/SOAP_integration_solar.cu:            cudaFree(xyz);
SOAP_code/SOAP_integration_solar.cu:            cudaFree(xyz22);
SOAP_code/SOAP_integration_solar.cu:        cudaMemcpy( f_spot_flux, dev_f_spot_flux, vector_size  * sizeof(double), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_integration_solar.cu:        cudaMemcpy( f_spot_bconv, dev_f_spot_bconv, vector_size  * sizeof(double), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_integration_solar.cu:        cudaMemcpy( f_spot_tot, dev_f_spot_tot, vector_size  * sizeof(double), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_integration_solar.cu:        cudaFree(dev_f_spot_flux);
SOAP_code/SOAP_integration_solar.cu:        cudaFree(dev_f_spot_bconv);
SOAP_code/SOAP_integration_solar.cu:        cudaFree(dev_f_spot_tot);
SOAP_code/SOAP_integration_solar.cu:        cudaFree(dev_vrad) ;
SOAP_code/SOAP_integration_solar.cu:        cudaFree(dev_spec_out) ;
SOAP_code/SOAP_integration_solar.cu:        cudaFree(dev_spec_out_spot) ;
SOAP_code/SOAP_integration_solar.cu:        cudaFree(dev_spec_SED_out) ;
SOAP_code/SOAP_integration_solar.cu:    cudaFree(dev_grad_spot) ;
SOAP_code/SOAP_integration_solar.cu:    cudaFree(dev_grad_faculae) ;
SOAP_code/SOAP_integration_solar.cu:    cudaFree(dev_grad_SED) ;
SOAP_code/SOAP_integration_solar.cu:    cudaFree(dev_grad) ;
SOAP_code/SOAP_integration.cu:#include <cuda.h>
SOAP_code/SOAP_integration.cu:__global__ void spot_phase_gpu(double *xyz, double inclination, int nrho, double phase,double *xyz2)
SOAP_code/SOAP_integration.cu:__global__ void spot_init_gpu(double s, double longitude, double latitude, double inclination,
SOAP_code/SOAP_integration.cu:    cudaMalloc( (void**)&spectra_SED_dev,  vector_size*no_spec * sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:    cudaMalloc( (void**)&spot_SED_dev,  vector_size*no_spec * sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:    cudaMalloc( (void**)&faculae_SED_dev,  vector_size*no_spec * sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:    cudaMemcpy( spectra_SED_dev, spectra_SED_array, vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_integration.cu:    cudaMemcpy( spot_SED_dev, spot_SED_array,  vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_integration.cu:    cudaMemcpy( faculae_SED_dev, faculae_SED_array,  vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_integration.cu:    cudaMalloc( (void**)&dev_wave, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:    cudaMalloc( (void**)&dev_spec,  vector_size*no_spec* sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:    cudaMemcpy( dev_wave, wave_array, vector_size * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_integration.cu:    cudaMemcpy( dev_spec, spec_array,  vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_integration.cu:    cudaMalloc( (void**)&dev_wave_spot, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:    cudaMalloc( (void**)&dev_spec_spot,  vector_size*no_spec * sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:    cudaMalloc( (void**)&dev_spec_faculae,  vector_size*no_spec * sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:    cudaMemcpy( dev_wave_spot, wave_array_spot, vector_size * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_integration.cu:    cudaMemcpy( dev_spec_spot, spec_array_spot,  vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_integration.cu:    cudaMemcpy( dev_spec_faculae, spec_array_faculae,  vector_size*no_spec * sizeof(double),  cudaMemcpyHostToDevice );
SOAP_code/SOAP_integration.cu:    cudaMalloc( (void**)&dev_grad, vector_size*no_spec  * sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:    cudaMalloc( (void**)&dev_grad_spot, vector_size*no_spec * sizeof(double) );
SOAP_code/SOAP_integration.cu:    cudaMalloc( (void**)&dev_grad_faculae, vector_size*no_spec * sizeof(double) );
SOAP_code/SOAP_integration.cu:    cudaMalloc( (void**)&dev_grad_SED, vector_size*no_spec  * sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:        cudaMalloc( (void**)&dev_vrad, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:        cudaMalloc( (void**)&dev_spec_out, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:        cudaMalloc( (void**)&dev_spec_out_spot, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:        cudaMalloc( (void**)&dev_spec_SED_out, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:        cudaMalloc( (void**)&dev_f_spot_flux, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:        cudaMalloc( (void**)&dev_f_spot_bconv, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:        cudaMalloc( (void**)&dev_f_spot_tot, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:        cudaMalloc( (void**)&dev_intensity, vector_size * sizeof(double) ) ;
SOAP_code/SOAP_integration.cu:            cudaMalloc( (void**)&xyz, 3*nrho* sizeof(double) );
SOAP_code/SOAP_integration.cu:            cudaMalloc( (void**)&xyz2, 3*nrho* sizeof(double) );
SOAP_code/SOAP_integration.cu:            cudaMalloc( (void**)&rho, nrho* sizeof(double) );
SOAP_code/SOAP_integration.cu:            cudaMalloc( (void**)&xyz22, 3*nrho* sizeof(double) );
SOAP_code/SOAP_integration.cu:            cudaMalloc( (void**)&counton, nrho* sizeof(int) );
SOAP_code/SOAP_integration.cu:            cudaMalloc( (void**)&countoff, nrho* sizeof(int) );
SOAP_code/SOAP_integration.cu:            cudaMalloc( (void**)&maxminy_array, nrho* sizeof(double) );
SOAP_code/SOAP_integration.cu:            cudaMalloc( (void**)&maxminz_array, nrho* sizeof(double) );
SOAP_code/SOAP_integration.cu:            cudaMalloc( (void**)&boundaries, 5*sizeof(int) );
SOAP_code/SOAP_integration.cu:            spot_init_gpu<<<nrho, 3>>>(size_maps[s_num], long_maps[s_num], lat_maps[s_num], inclination, nrho, xyz, xyz2, rho);
SOAP_code/SOAP_integration.cu:            spot_phase_gpu<<<nrho, 3>>>(xyz, inclination, nrho, psi[ipsi], xyz22);
SOAP_code/SOAP_integration.cu:            cudaMemcpy(boundaries_cpu, boundaries, 5*sizeof(int), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_integration.cu:            cudaFree(counton );
SOAP_code/SOAP_integration.cu:            cudaFree(countoff );
SOAP_code/SOAP_integration.cu:            cudaFree( maxminy_array );
SOAP_code/SOAP_integration.cu:            cudaFree( maxminz_array );
SOAP_code/SOAP_integration.cu:            cudaFree(boundaries);
SOAP_code/SOAP_integration.cu:            cudaFree(rho);
SOAP_code/SOAP_integration.cu:            cudaFree(xyz2);
SOAP_code/SOAP_integration.cu:            cudaFree(xyz);
SOAP_code/SOAP_integration.cu:            cudaFree(xyz22);
SOAP_code/SOAP_integration.cu:        cudaMemcpy( f_spot_flux, dev_f_spot_flux, vector_size  * sizeof(double), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_integration.cu:        cudaMemcpy( f_spot_bconv, dev_f_spot_bconv, vector_size  * sizeof(double), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_integration.cu:        cudaMemcpy( f_spot_tot, dev_f_spot_tot, vector_size  * sizeof(double), cudaMemcpyDeviceToHost );
SOAP_code/SOAP_integration.cu:        cudaFree(dev_f_spot_flux);
SOAP_code/SOAP_integration.cu:        cudaFree(dev_f_spot_bconv);
SOAP_code/SOAP_integration.cu:        cudaFree(dev_f_spot_tot);
SOAP_code/SOAP_integration.cu:        cudaFree(dev_vrad) ;
SOAP_code/SOAP_integration.cu:        cudaFree(dev_spec_out) ;
SOAP_code/SOAP_integration.cu:        cudaFree(dev_spec_out_spot) ;
SOAP_code/SOAP_integration.cu:        cudaFree(dev_spec_SED_out) ;
SOAP_code/SOAP_integration.cu:    cudaFree(dev_grad_spot) ;
SOAP_code/SOAP_integration.cu:    cudaFree(dev_grad_faculae) ;
SOAP_code/SOAP_integration.cu:    cudaFree(dev_grad_SED) ;
SOAP_code/SOAP_integration.cu:    cudaFree(dev_grad) ;
SOAP_code/SOAP_GPU_instruction.txt:SOAP-GPU is a revised SOAP 2.0 code (Dumusque et al. 2014, ApJ, 796, 132) that simulate spectral time series with the effect of active regions (spot, faculae or both).
SOAP_code/SOAP_GPU_instruction.txt:SOAP-GPU generates the integrated spectra at each phase for given input spectra and spectral resolution.
SOAP_code/SOAP_GPU_instruction.txt:1) Spectral simulation of stellar activity can be fast performed with GPU acceleration.
SOAP_code/SOAP_GPU_instruction.txt:2) SOAP-GPU can simulate more complicated active region structures, with superposition between active regions.
SOAP_code/SOAP_GPU_instruction.txt:3) SOAP-GPU implements more realistic line bisectors, based on solar observations, that varies as function of mu angle for both quiet and active regions.
SOAP_code/SOAP_GPU_instruction.txt:4) SOAP-GPU can accept any input high resolution observed spectra. The PHOENIX synthetic spectral library are already implemented at the code level which
SOAP_code/SOAP_GPU_instruction.txt:5) SOAP-GPU can simulate realistic spectral time series with either spot number/SDO image as additional inputs.
SOAP_code/SOAP_GPU_instruction.txt:INSTALLATION AND RUNNING SOAP-GPU
SOAP_code/SOAP_GPU_instruction.txt:The SOAP GPU code is written in C, with python scripts for input pre-processing and output post-processing.
SOAP_code/SOAP_GPU_instruction.txt:The code has been tested on python 3.7 and CUDA 10.1 and though it is expected to work properly on more recent versions, the user is advised to use those versions in case of troubles.
SOAP_code/SOAP_GPU_instruction.txt:- CUDA Toolkit 10.1 or above (You may also need to install the proper version NVIDIA driver, for more information on installation, please see: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
SOAP_code/SOAP_GPU_instruction.txt:To install SOAP-GPU, after unziping the zip file:
SOAP_code/SOAP_GPU_instruction.txt:$cd SOAP_GPU
SOAP_code/SOAP_GPU_instruction.txt:$bash install_SOAP_GPU.sh
SOAP_code/SOAP_GPU_instruction.txt:Copy the files into the folder SOAP_GPU/Rassine_file/ into the folder of the RASSINE code
SOAP_code/SOAP_GPU_instruction.txt:Before running SOAP-GPU, there are two config files that should be updated:
SOAP_code/SOAP_GPU_instruction.txt:Change the path of "input_prefix" to the one of your current installation of SOAP-GPU
SOAP_code/SOAP_GPU_instruction.txt:2) config_hpc.cfg (this is the main config file to run SOAP-GPU, Each input variables are described in the config file)
SOAP_code/SOAP_GPU_instruction.txt:Change the path of "input_prefix" to the one of your current installation of SOAP-GPU
SOAP_code/SOAP_GPU_instruction.txt:To run SOAP-GPU:
SOAP_code/SOAP_GPU_instruction.txt:3) Run run_all_SOAP.py to launch the SOAP-GPU simulation.
SOAP_code/SOAP_GPU_instruction.txt:If user plan to run SOAP-GPU on server, please modify run.sbatch instead. The output of sbatch will be written in the file "slurm_SLURMID.out"
README.md:SOAP-GPU is a revised SOAP 2.0 code (Dumusque et al. 2014, ApJ, 796, 132) that simulate spectral time series with the effect of active regions (spot, faculae or both).
README.md:SOAP-GPU generates the integrated spectra at each phase for given input spectra and spectral resolution.
README.md:1) Spectral simulation of stellar activity can be fast performed with GPU acceleration.
README.md:2) SOAP-GPU can simulate more complicated active region structures, with superposition between active regions.
README.md:3) SOAP-GPU implements more realistic line bisectors, based on solar observations, that varies as function of mu angle for both quiet and active regions.
README.md:4) SOAP-GPU can accept any input high resolution observed spectra. The PHOENIX synthetic spectral library are already implemented at the code level which
README.md:5) SOAP-GPU can simulate realistic spectral time series with either spot number/SDO image as additional inputs.
README.md:INSTALLATION AND RUNNING SOAP-GPU
README.md:The SOAP GPU code is written in C, with python scripts for input pre-processing and output post-processing.
README.md:The code has been tested on python 3.7 and CUDA 10.1 and though it is expected to work properly on more recent versions, the user is advised to use those versions in case of troubles.
README.md:- CUDA Toolkit 10.1 or above (You may also need to install the proper version NVIDIA driver, for more information on installation, please see: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
README.md:To install SOAP-GPU, after unziping the zip file:
README.md:$cd SOAP_GPU
README.md:$bash install_SOAP_GPU.sh
README.md:Copy the files into the folder SOAP_GPU/Rassine_file/ into the folder of the RASSINE code
README.md:Before running SOAP-GPU, there are two config files that should be updated:
README.md:Change the path of "input_prefix" to the one of your current installation of SOAP-GPU
README.md:2) config_hpc.cfg (this is the main config file to run SOAP-GPU, Each input variables are described in the config file)
README.md:Change the path of "input_prefix" to the one of your current installation of SOAP-GPU
README.md:To run SOAP-GPU:
README.md:3) Run run_all_SOAP.py to launch the SOAP-GPU simulation.
README.md:If user plan to run SOAP-GPU on server, please modify run.sbatch instead. The output of sbatch will be written in the file "slurm_SLURMID.out"

```
