# https://github.com/openmopac/mopac

```console
tests/keywords/CMakeLists.txt:#add_mopac_test(key-DEBUGPULAY "DEBUGPULAY.mop")
CMakeLists.txt:# GPU functionality (not functional at the moment)
CMakeLists.txt:option(GPU "GPU build flag" OFF)
CMakeLists.txt:if(GPU)
CMakeLists.txt:  add_definitions(-DGPU)
AUTHORS.rst:   expanded BLAS/LAPACK support, Intel MKL for multi-threading, & cuBLAS/MAGMA for GPU acceleration
src/run_mopac.F90:#ifdef GPU
src/run_mopac.F90:      Use mod_vars_cuda, only: lgpu, ngpus, gpu_id
src/run_mopac.F90:      Use gpu_info
src/run_mopac.F90:      Use settingGPUcard
src/run_mopac.F90:#ifdef GPU
src/run_mopac.F90:      logical :: lgpu_ref
src/run_mopac.F90:      logical(c_bool)    :: hasGpu = .false.
src/run_mopac.F90:      character*256	     :: gpuName(6)
src/run_mopac.F90:      logical            :: gpu_ok(6)
src/run_mopac.F90:#ifdef GPU
src/run_mopac.F90:        gpuName(1:6) = '' ; name_size(1:6) = 0 ; totalMem(1:6) = 0 ; clockRate(1:6) = 0
src/run_mopac.F90:        hasDouble(1:6) = .false. ; gpu_ok(1:6) = .false.
src/run_mopac.F90:        call gpuInfo(hasGpu, hasDouble, nDevices, gpuName,name_size, totalMem, &
src/run_mopac.F90:        lgpu = .false.
src/run_mopac.F90:        lgpu_ref = hasGPU
src/run_mopac.F90:        if (lgpu_ref) lgpu_ref = (index(keywrd, " NOGPU") == 0)
src/run_mopac.F90:        if (lgpu_ref) then
src/run_mopac.F90:          lgpu_ref = .false.
src/run_mopac.F90:! Counting how many GPUs are suitable to perform the calculations or with compute capability 2 (Fermi or Kepler).
src/run_mopac.F90:              gpu_ok(i) = .true.
src/run_mopac.F90:          lgpu_ref = (j >= 1)
src/run_mopac.F90:        ngpus = 1  ! in this version only single-GPU calculation are performed. This variable control it.
src/run_mopac.F90:!       ngpus = j ! in future versions of MOPAC Multi-GPUs calculations should be allowed
src/run_mopac.F90:        if (lgpu_ref) then
src/run_mopac.F90:          l = index(keywrd,' SETGPU=')
src/run_mopac.F90:          if (l /= 0) then  ! The user has inserted SETGPU keyword to Select one specific GPU
src/run_mopac.F90:            gpu_id = nint(reada(keywrd,l))
src/run_mopac.F90:            if (gpu_id > nDevices .or. gpu_id < 1 .or. (.not. gpu_ok(gpu_id))) then
src/run_mopac.F90:              Write(iw,'(/,5x,a)') ' Problem with the definition of SETGPU keyword ! '
src/run_mopac.F90:              Write(iw,'(5x,a,/)') ' MOPAC will automatically set a valid GPU card for the calculation '
src/run_mopac.F90:              on_off(gpu_id) = 'ON '
src/run_mopac.F90:              call setGPU(gpu_id - 1, lstat)
src/run_mopac.F90:                write (6,*) 'Problem to set GPU card ID = ', gpu_id
src/run_mopac.F90:          if (l == 0) then   ! Select GPU automatically
src/run_mopac.F90:             if (gpu_ok(i)) then
src/run_mopac.F90:                gpu_id = i - 1
src/run_mopac.F90:                call setGPU(gpu_id, lstat)
src/run_mopac.F90:                  write (6,*) 'Problem to set GPU card ID = ', gpu_id
src/run_mopac.F90:          ngpus = 0
src/run_mopac.F90:!  For small systems, using a GPU takes longer than not using a GPU,
src/run_mopac.F90:!  so do not use a GPU for small systems.  The lower limit, 100, is just a guess.
src/run_mopac.F90:        lgpu = (lgpu_ref .and. natoms > 100) ! Warning - there are problems with UHF calculations on small systems
src/input/wrtkey.F90:  if (myword(allkey, " NOGPU"))  write (iw, '(" *  NOGPU      - DO NOT USE GPU ACCELERATION")')
src/input/wrtkey.F90:if (myword(allkey,' SETGPU=')) then
src/input/wrtkey.F90:        i = index(keywrd,' SETGPU=')
src/input/wrtkey.F90:        write (iw,'(" *  SETGPU=   - YOUR CALCULATION WILL RUN IN THE GPU NUM. = ",i2)') j
src/properties/mullik.F90:      call density_for_GPU (vecs, fract, nclose, nopen, 2.d0, nlower, norbs, 2, pb, 3)
src/deprecated/mod_vars_cuda.F90:module mod_vars_cuda
src/deprecated/mod_vars_cuda.F90:  integer, parameter :: nthreads_gpu = 256, nblocks_gpu = 256
src/deprecated/mod_vars_cuda.F90:  logical :: lgpu = .false.
src/deprecated/mod_vars_cuda.F90:  real, parameter :: real_cuda = selected_real_kind(8)
src/deprecated/mod_vars_cuda.F90:  integer :: ngpus,gpu_id
src/deprecated/mod_vars_cuda.F90:  logical, parameter :: exe_gpu_kepler = .true.
src/deprecated/mod_vars_cuda.F90:end module mod_vars_cuda
src/deprecated/pulay_for_gpu.F90:      subroutine pulay_for_gpu(f, p, n, fppf, fock, emat, &
src/deprecated/pulay_for_gpu.F90:#ifdef GPU
src/deprecated/pulay_for_gpu.F90:      Use mod_vars_cuda, only: real_cuda, prec, ngpus
src/deprecated/pulay_for_gpu.F90:      Use mod_vars_cuda, only: lgpu 
src/deprecated/pulay_for_gpu.F90:#ifdef GPU
src/deprecated/pulay_for_gpu.F90:        debug = index(keywrd,'DEBUGPULAY') /= 0 
src/deprecated/pulay_for_gpu.F90:      if (lgpu) then
src/deprecated/pulay_for_gpu.F90:      end subroutine pulay_for_gpu
src/deprecated/mod_calls_cublas.F90:! For GPU MOPAC 
src/deprecated/mod_calls_cublas.F90:    interface asum_cublas_gpu
src/deprecated/mod_calls_cublas.F90:    interface axpy_cublas_gpu
src/deprecated/mod_calls_cublas.F90:    interface copy_cublas_gpu
src/deprecated/mod_calls_cublas.F90:    interface dot_cublas_gpu
src/deprecated/mod_calls_cublas.F90:    interface gemm_cublas_gpu
src/deprecated/mod_calls_cublas.F90:    interface gemm_cublas_multigpu
src/deprecated/mod_calls_cublas.F90:        subroutine gemm_cublas_mgpu(tra, trb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc) bind(c, &
src/deprecated/mod_calls_cublas.F90:            & name='call_gemm_cublas_mgpu')
src/deprecated/mod_calls_cublas.F90:        end subroutine gemm_cublas_mgpu
src/deprecated/mod_calls_cublas.F90:    interface gemm_phigemm_gpu
src/deprecated/mod_calls_cublas.F90:    interface gemm_cublas_gpu_thrust
src/deprecated/mod_calls_cublas.F90:    interface rot_cublas_gpu
src/deprecated/mod_calls_cublas.F90:    interface gemv_cublas_gpu
src/deprecated/mod_calls_cublas.F90:    interface ger_cublas_gpu
src/deprecated/mod_calls_cublas.F90:    interface nrm2_cublas_gpu
src/deprecated/mod_calls_cublas.F90:    interface scal_cublas_gpu
src/deprecated/mod_calls_cublas.F90:    interface swap_cublas_gpu
src/deprecated/mod_calls_cublas.F90:    interface trmm_cublas_gpu
src/deprecated/mod_calls_cublas.F90:    interface trmv_cublas_gpu
src/deprecated/mod_calls_cublas.F90:    interface trsm_cublas_gpu
src/deprecated/mod_calls_cublas.F90:    interface iamax_cublas_gpu
src/deprecated/mod_calls_cublas.F90:    interface iamin_cublas_gpu
src/deprecated/mod_calls_cublas.F90:    interface syrk_cublas_gpu
src/deprecated/mod_calls_cublas.F90:    interface syrk_cublas_gpu_thrust
src/deprecated/mod_calls_cublas.F90:  module cuda_alloc
src/deprecated/mod_calls_cublas.F90:    interface cudaMallocHost
src/deprecated/mod_calls_cublas.F90:      integer (C_INT) function cudaMallocHost(buffer, size)  bind(C,name="cudaMallocHost")
src/deprecated/mod_calls_cublas.F90:      end function cudaMallocHost
src/deprecated/mod_calls_cublas.F90:    interface cudaFreeHost
src/deprecated/mod_calls_cublas.F90:      integer (C_INT) function cudaFreeHost(buffer)  bind(C,name="cudaFreeHost")
src/deprecated/mod_calls_cublas.F90:      end function cudaFreeHost
src/deprecated/mod_calls_cublas.F90:  end module cuda_alloc
src/deprecated/mod_calls_cublas.F90:		subroutine magma_dsyevd_Driver1(ngpus,opt1, opt2,n,eigenvecs,m,eigvals, &
src/deprecated/mod_calls_cublas.F90:			integer(c_int),value :: n,m,lwork,liwork,ngpus
src/deprecated/mod_calls_cublas.F90:		subroutine magma_dsyevd_Driver2(ngpus,opt1, opt2,n,eigenvecs,m,eigvals, &
src/deprecated/mod_calls_cublas.F90:            integer(c_int),value :: n,m,lwork,liwork,ngpus
src/deprecated/mod_gpu_info.F90:        module gpu_info
src/deprecated/mod_gpu_info.F90:                subroutine gpuInfo(hasGpu, hasDouble, nDevices, name,name_size, totalMem, clockRate, major, minor) bind(c, name="getGPUInfo")
src/deprecated/mod_gpu_info.F90:                    logical(c_bool)		   :: hasGpu
src/deprecated/mod_gpu_info.F90:        module settingGPUcard
src/deprecated/mod_gpu_info.F90:                subroutine setGPU(idevice, stat) bind(c, name='setDevice')
src/matrix/mult_symm_AB.F90:!        Use mod_vars_cuda, only: ngpus
src/matrix/mult_symm_AB.F90:#ifdef GPU
src/matrix/mult_symm_AB.F90:        Use mamult_cuda_i
src/matrix/mult_symm_AB.F90:#ifdef GPU
src/matrix/mult_symm_AB.F90:#ifdef GPU
src/matrix/mult_symm_AB.F90:          case (2) ! mamult_gpu
src/matrix/mult_symm_AB.F90:            call mamult_gpu(a, b, c, ndim, mdim, ifact, beta, igrid, iblock, tt, 0)
src/matrix/mult_symm_AB.F90:#ifdef GPU
src/matrix/mult_symm_AB.F90:          case (4) ! dgemm_gpu
src/matrix/mult_symm_AB.F90:!            if (ngpus > 1 .and. ndim > 100) then
src/matrix/mult_symm_AB.F90:!               call gemm_cublas_mgpu ("N", "N", ndim, ndim, ndim, alpha, xa, ndim, xb, ndim, beta, xc, &
src/matrix/diag_for_GPU.F90:subroutine diag_for_GPU (fao, vector, nocc, eig, norbs, mpack)
src/matrix/diag_for_GPU.F90:#ifdef GPU
src/matrix/diag_for_GPU.F90:    Use mod_vars_cuda, only: lgpu, prec, ngpus
src/matrix/diag_for_GPU.F90:    use call_rot_cuda
src/matrix/diag_for_GPU.F90:#ifdef GPU
src/matrix/diag_for_GPU.F90:#ifdef GPU
src/matrix/diag_for_GPU.F90:    if (lgpu) then
src/matrix/diag_for_GPU.F90:#ifdef GPU
src/matrix/diag_for_GPU.F90:#ifdef GPU
src/matrix/diag_for_GPU.F90:    if (lgpu) then
src/matrix/diag_for_GPU.F90:#ifdef GPU
src/matrix/diag_for_GPU.F90:         if (ngpus > 1) then
src/matrix/diag_for_GPU.F90:            call rot_cuda_2gpu(fmo,eig,vector,ci0,ca0,nocc,lumo,n,bigeps,tiny)
src/matrix/diag_for_GPU.F90:            call rot_cuda(fmo,eig,vector,ci0,ca0,nocc,lumo,n,bigeps,tiny)
src/matrix/diag_for_GPU.F90:end subroutine diag_for_GPU
src/matrix/README.md:the prior support for GPUs in MOPAC, which is not presently functioning, needs to be re-introduced
src/matrix/CMakeLists.txt:    schmib linpack densit density_for_GPU mtxmc
src/matrix/CMakeLists.txt:    mxmt mat33 interp mxv diag_for_GPU
src/matrix/eigenvectors_LAPACK.F90:#ifdef GPU
src/matrix/eigenvectors_LAPACK.F90:      Use mod_vars_cuda, only: lgpu, ngpus, prec
src/matrix/eigenvectors_LAPACK.F90:#ifdef GPU
src/matrix/eigenvectors_LAPACK.F90:if (lgpu .and. (ngpus > 1 .and. ndim > 100)) then
src/matrix/eigenvectors_LAPACK.F90:      if (lgpu .and. ndim > 100) then
src/matrix/eigenvectors_LAPACK.F90:         if (ngpus > 1) then
src/matrix/eigenvectors_LAPACK.F90:             call magma_dsyevd_Driver1(ngpus,'v','l',ndim,eigenvecs,ndim,eigvals,&
src/matrix/eigenvectors_LAPACK.F90:             call magma_dsyevd_Driver1(ngpus,'v','u',ndim,eigenvecs,ndim,eigvals,&
src/matrix/eigenvectors_LAPACK.F90:      if (lgpu .and. ndim > 100) then
src/matrix/eigenvectors_LAPACK.F90:         if (ngpus > 1) then
src/matrix/eigenvectors_LAPACK.F90:             call magma_dsyevd_Driver2(ngpus,'v','l',ndim,eigenvecs,ndim,eigvals,&
src/matrix/eigenvectors_LAPACK.F90:             call magma_dsyevd_Driver2(ngpus,'v','u',ndim,eigenvecs,ndim,eigvals,&
src/matrix/density_for_GPU.F90:subroutine density_for_GPU (c, fract, ndubl, nsingl, occ, mpack, norbs, mode, pp, iopc)
src/matrix/density_for_GPU.F90:#ifdef GPU
src/matrix/density_for_GPU.F90:      Use mod_vars_cuda, only: real_cuda, prec, nthreads_gpu, nblocks_gpu
src/matrix/density_for_GPU.F90:      Use density_cuda_i
src/matrix/density_for_GPU.F90:#ifdef GPU
src/matrix/density_for_GPU.F90:#ifdef GPU
src/matrix/density_for_GPU.F90:#ifdef GPU
src/matrix/density_for_GPU.F90:End subroutine density_for_GPU
src/potentials/new_esp.F90:! For GPU MOPAC
src/potentials/new_esp.F90:        call density_for_GPU (vecs,fract,nclose,nopen, &
src/SCF/iter.F90:#ifdef GPU
src/SCF/iter.F90:      Use mod_vars_cuda, only: lgpu, real_cuda, prec
src/SCF/iter.F90:      use density_cuda_i
src/SCF/iter.F90:#ifdef GPU
src/SCF/iter.F90:          if (lgpu) iopc_calcp = 2  ! DGEMM on GPU
src/SCF/iter.F90:#ifdef GPU
src/SCF/iter.F90:          if (lgpu) iopc_calcp = 4  ! DSYRK on GPU
src/SCF/iter.F90:#ifdef GPU
src/SCF/iter.F90:              if (lgpu) then
src/SCF/iter.F90:                 call pulay_for_gpu (f, pa, norbs, pold, pold2, pold3, &
src/SCF/iter.F90:#ifdef GPU
src/SCF/iter.F90:!            if (lgpu) then
src/SCF/iter.F90:!               if (timitr) call timer ('BEFORE GPU DIAG')
src/SCF/iter.F90:!               call diag_for_GPU (f, c, na1el, eigs, norbs, mpack)
src/SCF/iter.F90:!               if (timitr) call timer ('AFTER  GPU DIAG')
src/SCF/iter.F90:!               call diag_for_GPU (f, c, na1el, eigs, norbs, mpack)
src/SCF/iter.F90:          call density_for_GPU (c, fract, nalpha, nalpha_open, 1.d0, mpack,norbs, 1, pa, iopc_calcp)
src/SCF/iter.F90:            call density_for_GPU (c, fract, na2el, na1el, 2.d0, mpack, norbs, 1, p, iopc_calcp)
src/SCF/iter.F90:#ifdef GPU
src/SCF/iter.F90:              if (lgpu) then
src/SCF/iter.F90:                 call pulay_for_gpu (fb, pb, norbs, pbold, pbold2, pbold3, &
src/SCF/iter.F90:#ifdef GPU
src/SCF/iter.F90:!              if (lgpu) then
src/SCF/iter.F90:!                 if (timitr) call timer ('BEFORE GPU DIAG')
src/SCF/iter.F90:!                 call diag_for_GPU (fb, cb, nb1el, eigb, norbs, mpack)
src/SCF/iter.F90:!                 if (timitr) call timer ('AFTER  GPU DIAG')
src/SCF/iter.F90:!                call diag_for_GPU (fb, cb, nb1el, eigb, norbs, mpack)
src/SCF/iter.F90:        call density_for_GPU (cb, fract, nbeta, nbeta_open, 1.d0, mpack, norbs, 1, pb, iopc_calcp)
src/SCF/pulay.F90:        debug = index(keywrd,'DEBUGPULAY') /= 0
src/optimization/ef.F90:! For GPU MOPAC

```
