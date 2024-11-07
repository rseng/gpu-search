# https://github.com/ELeeAstro/gCMCRT

```console
src_gCMCRT/exp_1D_sph_atm.f90:  use cudafor
src_gCMCRT/exp_1D_sph_atm.f90:  use cudafor
src_gCMCRT/exp_1D_sph_atm.f90:  print*, 'GPU info: '
src_gCMCRT/exp_1D_sph_atm.f90:  open(newunit=u1,file='moment_gpu_sph.txt',action='readwrite')
src_gCMCRT/exp_1D_sph_atm.f90:  open(newunit=u2,file='inten_gpu_sph.txt',action='readwrite')
src_gCMCRT/mc_k_raytrace.f90:  use cudafor
src_gCMCRT/exp_3D_sph_atm_alb.f90:  use cudafor
src_gCMCRT/exp_3D_sph_atm_alb.f90:  use cudafor
src_gCMCRT/exp_3D_sph_atm_alb.f90:  ! Send data to GPU data containers
src_gCMCRT/exp_3D_sph_atm_em.f90:  use cudafor
src_gCMCRT/exp_3D_sph_atm_em.f90:  use cudafor
src_gCMCRT/exp_3D_sph_atm_em.f90:  ! Send data to GPU data containers
src_gCMCRT/exp_3D_sph_atm_em.f90:      !istat = cudaDeviceSynchronize()
src_gCMCRT/mc_class_grid.f90:    !! Allocate device arrays and send to gpu
src_gCMCRT/mc_k_emit_iso.f90:!! Module containing device level subroutines (GPU) that emit packets isotropically
src_gCMCRT/mc_k_emit_iso.f90:  use cudafor
src_gCMCRT/mc_k_RR.f90:  use cudafor
src_gCMCRT/mc_k_tau_samp.f90:  use cudafor
src_gCMCRT/mc_k_limb_dark.f90:  use cudafor
src_gCMCRT/gpuCMCRT.f90:program gpuCMCRT
src_gCMCRT/gpuCMCRT.f90:end program gpuCMCRT
src_gCMCRT/exp_3D_sph_atm_trans_hires.f90:  use cudafor
src_gCMCRT/exp_3D_sph_atm_trans_hires.f90:  use cudafor
src_gCMCRT/exp_3D_sph_atm_trans_hires.f90:  ! Send data to GPU data containers
src_gCMCRT/exp_3D_sph_atm_trans_hires.f90:  ! Grid for GPU threads/blocks
src_gCMCRT/exp_3D_sph_atm_trans_hires.f90:      !istat = cudaDeviceSynchronize()
src_gCMCRT/exp_3D_sph_atm_pol.f90:  use cudafor
src_gCMCRT/exp_3D_sph_atm_pol.f90:  use cudafor
src_gCMCRT/exp_3D_sph_atm_pol.f90:  ! Send data to GPU data containers
src_gCMCRT/exp_3D_sph_atm_pol.f90:  ! Grid for GPU threads/blocks
src_gCMCRT/exp_3D_sph_atm_pol.f90:    istat = cudaDeviceSynchronize()
src_gCMCRT/mc_k_gord_samp.f90:  use cudafor
src_gCMCRT/mc_k_tauint.f90:  use cudafor
src_gCMCRT/Makefile:FFLAGS = -fast -O3 -cuda -cudalib=curand
src_gCMCRT/Makefile:DEBUG_FLAGS = -O0 -g -C -traceback -cuda -cudalib=curand
src_gCMCRT/Makefile:    gpuCMCRT.o
src_gCMCRT/mc_k_findcell.f90:  use cudafor
src_gCMCRT/mc_Draine_G.f90:  use cudafor
src_gCMCRT/mc_k_source_pac_inc.f90:  use cudafor
src_gCMCRT/exp_3D_sph_atm_trans.f90:  use cudafor
src_gCMCRT/exp_3D_sph_atm_trans.f90:  use cudafor
src_gCMCRT/exp_3D_sph_atm_trans.f90:  ! Send data to GPU data containers
src_gCMCRT/exp_3D_sph_atm_trans.f90:  ! Grid for GPU threads/blocks
src_gCMCRT/exp_3D_sph_atm_trans.f90:    istat = cudaDeviceSynchronize()
src_gCMCRT/exp_3D_sph_atm_tests.f90:  use cudafor
src_gCMCRT/exp_3D_sph_atm_tests.f90:  use cudafor
src_gCMCRT/exp_3D_sph_atm_tests.f90:  print*, 'GPU info: '
src_gCMCRT/exp_1D_pp_atm.f90:  use cudafor
src_gCMCRT/exp_1D_pp_atm.f90:  use cudafor
src_gCMCRT/exp_1D_pp_atm.f90:  print*, 'GPU info: '
src_gCMCRT/exp_1D_pp_atm.f90:  open(newunit=u1,file='moment_gpu_pp.txt',action='readwrite')
src_gCMCRT/exp_1D_pp_atm.f90:  open(newunit=u2,file='inten_gpu_pp.txt',action='readwrite')
src_gCMCRT/mc_k_vol_samp.f90:  use cudafor
src_gCMCRT/mc_k_peeloff_emit.f90:  use cudafor
src_gCMCRT/mc_read_prf.f90:    !! Send wl array to gpu memory
src_gCMCRT/mc_read_prf.f90:    ! Send data to GPU
src_gCMCRT/mc_k_findwall_sph.f90:  use cudafor
src_gCMCRT/mc_k_scatt.f90:!! Module containing device level subroutines (GPU) that scatter packets isotropically
src_gCMCRT/mc_k_scatt.f90:  use cudafor
src_gCMCRT/mc_k_moments.f90:  use cudafor
src_gCMCRT/mc_k_scatt_mat.f90:  use cudafor
src_gCMCRT/exp_3D_cart_galaxy.f90:!!! Kernel routine for the 3D diffuse galaxy test - this is run on the device (GPU)
src_gCMCRT/exp_3D_cart_galaxy.f90:  use cudafor
src_gCMCRT/exp_3D_cart_galaxy.f90:  use cudafor
src_gCMCRT/mc_k_peeloff_scatt.f90:  use cudafor
src_gCMCRT/mc_set_em.f90:      ! Send relevent GPU data
src_gCMCRT/mc_class_imag.f90:  use cudafor
src_gCMCRT/mc_class_imag.f90:      !! Give to the GPU
src_gCMCRT/exp_3D_sph_atm_em_hires.f90:  use cudafor
src_gCMCRT/exp_3D_sph_atm_em_hires.f90:  use cudafor
src_gCMCRT/exp_3D_sph_atm_em_hires.f90:  ! Send data to GPU data containers
src_gCMCRT/exp_3D_sph_atm_em_hires.f90:      !istat = cudaDeviceSynchronize()
README.md:# gCMCRT - 3D Monte Carlo Radiative Transfer for exoplanet atmospheres with GPUs
README.md:You will need a Nvidia GPU card on your system.
README.md:You will need to install the latest drivers from Nvidia: https://www.nvidia.com/Download/index.aspx
README.md:(optional) install the CUDA toolkit: https://developer.nvidia.com/cuda-toolkit
README.md:You will need to install the CUDA hpc sdk: https://developer.nvidia.com/hpc-sdk

```
