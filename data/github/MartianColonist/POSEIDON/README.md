# https://github.com/MartianColonist/POSEIDON

```console
POSEIDON/emission.py:from numba import jit, cuda
POSEIDON/emission.py:from .utility import mock_missing, interp_GPU
POSEIDON/emission.py:@cuda.jit
POSEIDON/emission.py:def planck_lambda_arr_GPU(T, wl, B_lambda):
POSEIDON/emission.py:    GPU variant of the 'planck_lambda_arr' function.
POSEIDON/emission.py:    thread = cuda.grid(1)
POSEIDON/emission.py:    stride = cuda.gridsize(1)
POSEIDON/emission.py:def emission_single_stream_GPU(T, dz, wl, kappa, Gauss_quad = 2):
POSEIDON/emission.py:    GPU variant of the 'emission_rad_transfer' function.
POSEIDON/emission.py:    # Define temperature, wavelength, and Planck function arrays on GPU
POSEIDON/emission.py:    planck_lambda_arr_GPU[block, thread](T, wl_m, B)
POSEIDON/emission.py:    @cuda.jit
POSEIDON/emission.py:        thread = cuda.grid(1)
POSEIDON/emission.py:        stride = cuda.gridsize(1)
POSEIDON/emission.py:                cuda.atomic.add(F, k, 2.0 * math.pi * mu[j] * I[j,k] * W[j])
POSEIDON/emission.py:@cuda.jit
POSEIDON/emission.py:def determine_photosphere_radii_GPU(tau_lambda, r_low, wl, R_p_eff, photosphere_tau = 2/3):
POSEIDON/emission.py:    GPU variant of the 'determine_photosphere_radii' function.
POSEIDON/emission.py:    thread = cuda.grid(1)
POSEIDON/emission.py:    stride = cuda.gridsize(1)
POSEIDON/emission.py:        R_p_eff[k] = interp_GPU(photosphere_tau, tau_lambda[:,k], r_low)
POSEIDON/utility.py:from numba import jit, cuda
POSEIDON/utility.py:@cuda.jit(device=True)
POSEIDON/utility.py:def prior_index_GPU(value, grid):
POSEIDON/utility.py:    GPU variant of the 'prior_index' function.
POSEIDON/utility.py:@cuda.jit(device=True)
POSEIDON/utility.py:def interp_GPU(x_value, x, y):
POSEIDON/utility.py:    Linear interpolation using a GPU.
POSEIDON/utility.py:    prior_index = prior_index_GPU(x_value, x)
POSEIDON/utility.py:@cuda.jit(device=True)
POSEIDON/utility.py:def closest_index_GPU(value, grid_start, grid_end, N_grid):
POSEIDON/utility.py:    GPU variant of the 'closest_index' function.
POSEIDON/absorption.py:from numba import cuda, jit
POSEIDON/absorption.py:from .utility import prior_index, prior_index_V2, closest_index, closest_index_GPU, \
POSEIDON/absorption.py:@cuda.jit
POSEIDON/absorption.py:def extinction_GPU(kappa_gas, kappa_Ray, kappa_cloud, i_bot, N_species, N_species_active, 
POSEIDON/absorption.py:    thread = cuda.grid(1)
POSEIDON/absorption.py:    stride = cuda.gridsize(1)
POSEIDON/absorption.py:                idx_T_fine = closest_index_GPU(T[i,j,k], T_fine[0], T_fine[-1], N_T_fine)
POSEIDON/absorption.py:                idx_P_fine = closest_index_GPU(math.log10(P[i]), log_P_fine[0], log_P_fine[-1], N_P_fine)
POSEIDON/core.py:# These settings only used for GPU models (experimental)
POSEIDON/core.py:                        extinction_LBL, extinction_GPU
POSEIDON/core.py:                      emission_single_stream_GPU, determine_photosphere_radii_GPU, \
POSEIDON/core.py:    # Move cross sections to GPU memory to speed up later computations
POSEIDON/core.py:    if (device == 'gpu'):
POSEIDON/core.py:            Experimental: use CPU or GPU (only for emission spectra)
POSEIDON/core.py:            (Options: cpu / gpu)
POSEIDON/core.py:        # Running POSEIDON on the GPU
POSEIDON/core.py:        elif (device == 'gpu'):
POSEIDON/core.py:            # Define extinction coefficient arrays explicitly on GPU
POSEIDON/core.py:            extinction_GPU[block, thread](kappa_gas, kappa_Ray, kappa_cloud, i_bot, 
POSEIDON/core.py:        if (device == 'gpu'):
POSEIDON/core.py:            raise Exception("GPU transmission spectra not yet supported.")
POSEIDON/core.py:        if (device == 'gpu'):
POSEIDON/core.py:            raise Exception("GPU transmission spectra not yet supported.")
POSEIDON/core.py:            # Compute planet flux (on CPU or GPU)
POSEIDON/core.py:            elif (device == 'gpu'):
POSEIDON/core.py:                F_p, dtau = emission_single_stream_GPU(T, dz, wl, kappa_tot, Gauss_quad)
POSEIDON/core.py:            # Running POSEIDON on the GPU
POSEIDON/core.py:            elif (device == 'gpu'):
POSEIDON/core.py:                # Calculate photosphere radius using GPU
POSEIDON/core.py:                determine_photosphere_radii_GPU[block, thread](tau_lambda, r_low_flipped, wl, R_p_eff, 2/3)
POSEIDON/contributions.py:# These settings only used for GPU models (experimental)
POSEIDON/contributions.py:                      emission_single_stream_GPU, determine_photosphere_radii_GPU, \
POSEIDON/contributions.py:            Experimental: use CPU or GPU (only for emission spectra)
POSEIDON/contributions.py:            (Options: cpu / gpu)
POSEIDON/contributions.py:        # Running POSEIDON on the GPU
POSEIDON/contributions.py:        elif (device == 'gpu'):
POSEIDON/contributions.py:            raise Exception("GPU transmission spectra not yet supported.")
POSEIDON/contributions.py:        if (device == 'gpu'):
POSEIDON/contributions.py:            raise Exception("GPU transmission spectra not yet supported.")
POSEIDON/contributions.py:            # Compute planet flux (on CPU or GPU)
POSEIDON/contributions.py:            elif (device == 'gpu'):
POSEIDON/contributions.py:                F_p, dtau = emission_single_stream_GPU(T, dz, wl, kappa_tot, Gauss_quad)
POSEIDON/contributions.py:            # Running POSEIDON on the GPU
POSEIDON/contributions.py:            elif (device == 'gpu'):
POSEIDON/contributions.py:                # Calculate photosphere radius using GPU
POSEIDON/contributions.py:                determine_photosphere_radii_GPU[block, thread](tau_lambda, r_low_flipped, wl, R_p_eff, 2/3)
POSEIDON/contributions.py:                # Compute planet flux (on CPU or GPU)
POSEIDON/contributions.py:                elif (device == 'gpu'):
POSEIDON/contributions.py:                    F_p, dtau = emission_single_stream_GPU(T, dz, wl, kappa_tot, Gauss_quad)
POSEIDON/contributions.py:                # Running POSEIDON on the GPU
POSEIDON/contributions.py:                elif (device == 'gpu'):
POSEIDON/contributions.py:                    # Calculate photosphere radius using GPU
POSEIDON/contributions.py:                    determine_photosphere_radii_GPU[block, thread](tau_lambda, r_low_flipped, wl, R_p_eff, 2/3)
POSEIDON/contributions.py:            Experimental: use CPU or GPU (only for emission spectra)
POSEIDON/contributions.py:            (Options: cpu / gpu)
POSEIDON/contributions.py:        # Running POSEIDON on the GPU
POSEIDON/contributions.py:        elif (device == 'gpu'):
POSEIDON/contributions.py:            raise Exception("GPU transmission spectra not yet supported.")
POSEIDON/contributions.py:        if (device == 'gpu'):
POSEIDON/contributions.py:            raise Exception("GPU transmission spectra not yet supported.")
POSEIDON/contributions.py:            # Compute planet flux (on CPU or GPU)
POSEIDON/contributions.py:            elif (device == 'gpu'):
POSEIDON/contributions.py:                F_p, dtau = emission_single_stream_GPU(T, dz, wl, kappa_tot, Gauss_quad)
POSEIDON/contributions.py:            # Running POSEIDON on the GPU
POSEIDON/contributions.py:            elif (device == 'gpu'):
POSEIDON/contributions.py:                # Calculate photosphere radius using GPU
POSEIDON/contributions.py:                determine_photosphere_radii_GPU[block, thread](tau_lambda, r_low_flipped, wl, R_p_eff, 2/3)
POSEIDON/contributions.py:                # Compute planet flux (on CPU or GPU)
POSEIDON/contributions.py:                elif (device == 'gpu'):
POSEIDON/contributions.py:                    F_p, dtau = emission_single_stream_GPU(T, dz, wl, kappa_tot, Gauss_quad)
POSEIDON/contributions.py:                # Running POSEIDON on the GPU
POSEIDON/contributions.py:                elif (device == 'gpu'):
POSEIDON/contributions.py:                    # Calculate photosphere radius using GPU
POSEIDON/contributions.py:                    determine_photosphere_radii_GPU[block, thread](tau_lambda, r_low_flipped, wl, R_p_eff, 2/3)
POSEIDON/contributions.py:            Experimental: use CPU or GPU (only for emission spectra)
POSEIDON/contributions.py:            (Options: cpu / gpu)

```
