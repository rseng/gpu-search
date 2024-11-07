# https://github.com/grimme-lab/xtb

```console
meson/meson.build:elif fc_id == 'pgi' or fc_id == 'nvidia_hpc'
meson/meson.build:  if get_option('gpu')
meson/meson.build:    add_project_arguments('-acc', '-Minfo=accel', '-DXTB_GPU', language: 'fortran')
meson/meson.build:    gpu_arch = get_option('gpu_arch') 
meson/meson.build:    add_project_arguments('-ta=tesla:cc@0@'.format(gpu_arch), language: 'fortran')
meson/meson.build:    add_project_link_arguments('-ta=tesla:cc@0@'.format(gpu_arch), language: 'fortran')
meson/meson.build:      add_project_arguments('-Mcudalib=cusolver,cublas', '-DUSE_CUSOLVER', '-DUSE_CUBLAS', language: 'fortran')
meson/meson.build:      add_project_link_arguments('-Mcudalib=cusolver,cublas', language: 'fortran')
meson/meson.build:  omp_dep = dependency('openmp', required: fc_id != 'intel' and fc_id != 'intel-llvm' and fc_id != 'nvidia_hpc')
meson_options.txt:# GPU specific options
meson_options.txt:option('gpu', type: 'boolean', value: false,
meson_options.txt:       description: 'use GPU acceleration')
meson_options.txt:option('gpu_arch', type: 'string', value: '70',
meson_options.txt:       description: 'GPU architecture version string')
src/xtb/hamiltonian_gpu.f90:! Copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
src/xtb/hamiltonian_gpu.f90:module xtb_xtb_hamiltonian_gpu
src/xtb/hamiltonian_gpu.f90:   public :: build_SDQH0_gpu, build_dSDQH0_gpu
src/xtb/hamiltonian_gpu.f90:pure subroutine build_dsdq_ints_gpu(a,b,c,alpi,alpj,t,e,la,lb,v,g)
src/xtb/hamiltonian_gpu.f90:end subroutine build_dsdq_ints_gpu
src/xtb/hamiltonian_gpu.f90:subroutine build_SDQH0_gpu(nShell, hData, nat, at, nbf, nao, xyz, trans, selfEnergy, &
src/xtb/hamiltonian_gpu.f90:                          call build_sdq_ints_gpu(ra,rb,point,alpi,alpj, &
src/xtb/hamiltonian_gpu.f90:end subroutine build_SDQH0_gpu
src/xtb/hamiltonian_gpu.f90:subroutine build_dSDQH0_gpu(nShell, hData, selfEnergy, dSEdcn, intcut, nat, nao, nbf, &
src/xtb/hamiltonian_gpu.f90:                             call build_dsdq_ints_gpu(ri,rj,rj,alpi,alpj,t,e,&
src/xtb/hamiltonian_gpu.f90:                        call shiftintg_gpu(dum,sawg,saw,rij)
src/xtb/hamiltonian_gpu.f90:                  call dtrf2_gpu(sdq(1:6,1:6,1),ishtyp,jshtyp)
src/xtb/hamiltonian_gpu.f90:                        call dtrf2_gpu(sdqg(1:6,1:6,ixyz,k),ishtyp,jshtyp)
src/xtb/hamiltonian_gpu.f90:                        call dtrf2_gpu(sdqg2(1:6,1:6,ixyz,k),ishtyp,jshtyp)
src/xtb/hamiltonian_gpu.f90:end subroutine build_dSDQH0_gpu
src/xtb/hamiltonian_gpu.f90:subroutine dtrf_gpu(s,li,lj)
src/xtb/hamiltonian_gpu.f90:end subroutine dtrf_gpu
src/xtb/hamiltonian_gpu.f90:subroutine dtrf2_gpu(s,li,lj)
src/xtb/hamiltonian_gpu.f90:end subroutine dtrf2_gpu
src/xtb/hamiltonian_gpu.f90:pure subroutine shiftintg_gpu(g2,g,s,r)
src/xtb/hamiltonian_gpu.f90:end subroutine shiftintg_gpu
src/xtb/hamiltonian_gpu.f90:end module xtb_xtb_hamiltonian_gpu
src/xtb/repulsion.F90:#ifdef XTB_GPU
src/xtb/repulsion.F90:#ifdef XTB_GPU
src/xtb/CMakeLists.txt:  "${dir}/hamiltonian_gpu.f90"
src/xtb/meson.build:  'hamiltonian_gpu.f90',
src/disp/dftd4.F90:! Copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
src/disp/dftd4.F90:!#ifdef XTB_GPU
src/disp/dftd4.F90:!#ifdef XTB_GPU
src/disp/dftd4.F90:#ifdef XTB_GPU
src/disp/dftd4.F90:#ifdef XTB_GPU
src/disp/dftd4.F90:#ifdef XTB_GPU
src/disp/dftd4.F90:#ifdef XTB_GPU
src/disp/dftd4.F90:!#ifdef XTB_GPU
src/disp/dftd4.F90:!#ifdef XTB_GPU
src/disp/dftd4.F90:#ifdef XTB_GPU
src/disp/dftd4.F90:#ifdef XTB_GPU
src/disp/dftd4.F90:#ifdef XTB_GPU
src/disp/dftd4.F90:      call atm_gradient_latp_gpu(mol, trans, cutoff3, par, sqrtZr4r2, c6, dc6dcn, &
src/disp/dftd4.F90:#ifdef XTB_GPU
src/disp/dftd4.F90:   call atm_gradient_latp_gpu &
src/disp/dftd4.F90:subroutine atm_gradient_latp_gpu &
src/disp/dftd4.F90:end subroutine atm_gradient_latp_gpu
src/intgrad.f90:pure subroutine build_sdq_ints_gpu(a,b,c,alpi,alpj,la,lb,kab,t,e,lx,ly,lz,v)
src/intgrad.f90:end subroutine build_sdq_ints_gpu
src/intgrad.f90:subroutine sdqint_gpu(nShell, angShell, nat, at, nbf, nao, xyz, trans, &
src/intgrad.f90:                          call build_sdq_ints_gpu(ra,rb,point,alpi,alpj, &
src/intgrad.f90:end subroutine sdqint_gpu
src/nvtx.f90:! Copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
src/mctc/lapack/eigensolve.F90:! Copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
src/main/property.F90:#ifdef XTB_GPU
src/main/property.F90:      call sdqint_gpu(xtbData%nShell, xtbData%hamiltonian%angShell, mol%n, mol%at, &
src/scf_module.F90:   use xtb_xtb_hamiltonian_gpu, only: build_SDQH0_gpu, build_dSDQH0_gpu
src/scf_module.F90:#ifdef XTB_GPU
src/scf_module.F90:   call build_SDQH0_gpu(xtbData%nShell, xtbData%hamiltonian, mol%n, mol%at, &
src/scf_module.F90:#ifdef XTB_GPU
src/scf_module.F90:   call build_dSDQH0_gpu(xtbData%nShell, xtbData%hamiltonian, selfEnergy, dSEdcn, &

```
