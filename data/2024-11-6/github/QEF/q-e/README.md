# https://github.com/QEF/q-e

```console
dft-d3/CMakeLists.txt:        qe_openacc_fortran
dft-d3/core.f90:! OpenACC acceleration added by Ivan Carnimeo,   June 2021
QEHeat/CMakeLists.txt:    qe_openacc_fortran
QEHeat/CMakeLists.txt:qe_enable_cuda_fortran("${src_all_currents_x}")
QEHeat/src/all_currents.f90:   USE control_flags, ONLY: ethr, use_gpu
QEHeat/src/all_currents.f90:   LOGICAL,EXTERNAL :: check_gpu_support
QEHeat/src/all_currents.f90:   use_gpu = check_gpu_support()
QEHeat/src/all_currents.f90:   if(use_gpu) Call errore('QEHeat', 'QEHeat with GPU NYI.', 1)
PHonon/PH/addcore.f90:#if defined(__CUDA)
PHonon/PH/addcore.f90:  CALL start_clock_gpu('addcore')
PHonon/PH/addcore.f90:  CALL stop_clock_gpu('addcore')
PHonon/PH/dvqpsi_us.f90:#if defined(__CUDA)
PHonon/PH/dvqpsi_us.f90:  call start_clock_gpu ('dvqpsi_us')
PHonon/PH/dvqpsi_us.f90:  call stop_clock_gpu ('dvqpsi_us')
PHonon/PH/dvqpsi_us.f90:#if !defined(__CUDA)
PHonon/PH/phq_init.f90:#if defined(__CUDA)
PHonon/PH/phq_init.f90:#if defined(__CUDA)
PHonon/PH/phq_init.f90:#if defined(__CUDA)
PHonon/PH/phq_init.f90:#if defined(__CUDA)
PHonon/PH/phq_init.f90:#if defined(__CUDA)
PHonon/PH/phq_init.f90:#if defined(__CUDA)
PHonon/PH/phq_init.f90:#if defined(__CUDA)
PHonon/PH/phq_init.f90:#if defined(__CUDA)
PHonon/PH/phq_init.f90:#if defined(__CUDA)
PHonon/PH/do_phonon.f90:  USE control_flags,   ONLY : use_gpu
PHonon/PH/do_phonon.f90:  USE control_flags,  ONLY : use_gpu
PHonon/PH/do_phonon.f90:  USE environment,   ONLY : print_cuda_info
PHonon/PH/do_phonon.f90:  LOGICAL,EXTERNAL :: check_gpu_support
PHonon/PH/do_phonon.f90:     use_gpu = check_gpu_support()
PHonon/PH/do_phonon.f90:        CALL print_cuda_info(check_use_gpu=.true.) 
PHonon/PH/compute_dvloc.f90:#if defined(__CUDA)
PHonon/PH/compute_dvloc.f90:  call start_clock_gpu ('com_dvloc')
PHonon/PH/compute_dvloc.f90:#if !defined(__CUDA)
PHonon/PH/compute_dvloc.f90:  call stop_clock_gpu ('com_dvloc')
PHonon/PH/drhodv.f90:  !civn: at the time I am doing this OpenACC hardly manages offloading arrays of data structures.
PHonon/PH/drhodv.f90:#if defined(__CUDA)
PHonon/PH/drhodv.f90:#if defined(__CUDA)
PHonon/PH/drhodv.f90:#if defined(__CUDA)
PHonon/PH/drhodv.f90:#if defined(__CUDA)
PHonon/PH/drhodv.f90:#if defined(__CUDA)
PHonon/PH/drhodv.f90:#if defined(__CUDA)
PHonon/CMakeLists.txt:qe_enable_cuda_fortran("${src_ph}")
PHonon/CMakeLists.txt:qe_enable_cuda_fortran("${src_gamma}")
PHonon/CMakeLists.txt:	qe_openacc_fortran
PHonon/CMakeLists.txt:if(QE_ENABLE_CUDA)
PHonon/CMakeLists.txt:            CUDA::cublas)
PHonon/CMakeLists.txt:qe_enable_cuda_fortran("${src_phonon_x}")
PHonon/CMakeLists.txt:        qe_openacc_fortran
PHonon/CMakeLists.txt:        qe_openacc_fortran
PHonon/CMakeLists.txt:qe_enable_cuda_fortran("${src_phonon_phcg_x}")
HP/CMakeLists.txt:qe_enable_cuda_fortran("${src_hp_x}")
HP/src/hp_main.f90:  USE control_flags,     ONLY : dfpt_hub, use_para_diag, use_gpu
HP/src/hp_main.f90:  LOGICAL,EXTERNAL :: check_gpu_support 
HP/src/hp_main.f90:  use_gpu = check_gpu_support()
KCW/examples/example05/reference/Si.kcw-screen.out:                                        0.00s GPU  (   23296 calls)
KCW/examples/example05/reference/Si.kcw-screen.out:                                        0.00s GPU  (   23296 calls)
KCW/examples/example01/reference/Si.kcw-screen.out:                                        0.00s GPU  (    6128 calls)
KCW/examples/example01/reference/Si.kcw-screen.out:                                        0.00s GPU  (    6128 calls)
KCW/examples/example02/reference/h2o.kcw-screen.out:                                        0.00s GPU  (     120 calls)
KCW/examples/example02/reference/h2o.kcw-screen.out:                                        0.00s GPU  (     120 calls)
KCW/PP/CMakeLists.txt:qe_enable_cuda_fortran("${src_kcwpp_sh_x}")
KCW/PP/CMakeLists.txt:qe_enable_cuda_fortran("${src_kcwpp_interp_x}")
KCW/src/kcw.f90:  USE control_flags,     ONLY : use_gpu
KCW/src/kcw.f90:  LOGICAL,EXTERNAL :: check_gpu_support 
KCW/src/kcw.f90:  use_gpu = check_gpu_support()
KCW/src/kcw.f90:  IF(use_gpu) Call errore('KCW', 'KCW with GPU NYI', 1)
EPW/src/epw.f90:  USE control_flags,    ONLY : gamma_only, use_gpu
EPW/src/epw.f90:  LOGICAL,EXTERNAL    :: check_gpu_support
EPW/src/epw.f90:  use_gpu = check_gpu_support()
EPW/src/epw.f90:  IF(use_gpu) Call errore('EPW', 'EPW with GPU NYI', 1)
install/aclocal.m4:m4_include([m4/x_ac_qe_cuda.m4])
install/extlibs_makefile:all: libcuda
install/extlibs_makefile:# CUDA
install/extlibs_makefile:libcuda : $(addprefix libcuda_,$(CUDA_EXTLIBS))
install/extlibs_makefile:CUDA_PATH := $(if $(GPU_ARCH),$(CUDA_PATH),no)
install/extlibs_makefile:libcuda_devxlib :
install/extlibs_makefile:	touch $(TOPDIR)/install/libcuda_devxlib # do not download and configure again
install/extlibs_makefile:libcuda_devxlib_clean:
install/extlibs_makefile:	if test -f libcuda_devxlib; then rm libcuda_devxlib; fi
install/extlibs_makefile:libcuda_devxlib_veryclean:
install/extlibs_makefile:clean: lapack_clean fox_clean libcuda_devxlib_clean libmbd_clean
install/extlibs_makefile:veryclean: fox_clean libcuda_devxlib_veryclean libmbd_distclean
install/configure.ac:# Checking CUDA...
install/configure.ac:X_AC_QE_CUDA()
install/makedeps.sh:    # list of all cuda-related modules
install/makedeps.sh:    cudadeps="cublas cudafor curand cufft flops_tracker cusolverdn \
install/makedeps.sh:              zhegvdx_gpu dsyevd_gpu dsygvdx_gpu eigsolve_vars     \
install/makedeps.sh:# "cudadeps" defined in "install/makedeps.sh".
install/makedeps.sh:	for no_dep in $sysdeps $libdeps $cudadeps; do
install/m4/x_ac_qe_blas.m4:		    # NB: with nvidia hpc sdk 2020, linking to threaded mkl
install/m4/x_ac_qe_blas.m4:		    # NB: with (at least) nvidia 22.7, threaded mkl are
install/m4/x_ac_qe_cuda.m4:# AX_CHECK_CUDA
install/m4/x_ac_qe_cuda.m4:# Simplified compilation for NVidia GPUs using nvhpc compiler
install/m4/x_ac_qe_cuda.m4:#    gpu_arch
install/m4/x_ac_qe_cuda.m4:#    cuda_runtime
install/m4/x_ac_qe_cuda.m4:#    cuda_cflags
install/m4/x_ac_qe_cuda.m4:#    cuda_fflags
install/m4/x_ac_qe_cuda.m4:#    cuda_libs
install/m4/x_ac_qe_cuda.m4:#    cuda_extlibs
install/m4/x_ac_qe_cuda.m4:AC_DEFUN([X_AC_QE_CUDA], [
install/m4/x_ac_qe_cuda.m4:gpu_arch=
install/m4/x_ac_qe_cuda.m4:cuda_runtime=
install/m4/x_ac_qe_cuda.m4:cuda_cflags=
install/m4/x_ac_qe_cuda.m4:cuda_fflags=
install/m4/x_ac_qe_cuda.m4:cuda_libs=
install/m4/x_ac_qe_cuda.m4:# FIXME: currently devxlib is needed also for non-CUDA compilation
install/m4/x_ac_qe_cuda.m4:cuda_extlibs=devxlib
install/m4/x_ac_qe_cuda.m4:# Provide your CUDA path with this
install/m4/x_ac_qe_cuda.m4:AC_ARG_WITH([cuda],
install/m4/x_ac_qe_cuda.m4:   [AS_HELP_STRING([--with-cuda=PATH],[prefix where CUDA is installed @<:@default=no@:>@])],
install/m4/x_ac_qe_cuda.m4:   [with_cuda=no])
install/m4/x_ac_qe_cuda.m4:AC_ARG_WITH([cuda-cc],
install/m4/x_ac_qe_cuda.m4:   [AS_HELP_STRING([--with-cuda-cc=VAL],[GPU architecture (Kepler: 35, Pascal: 60, Volta: 70) @<:@default=35@:>@])],
install/m4/x_ac_qe_cuda.m4:   [with_cuda_cc=35])
install/m4/x_ac_qe_cuda.m4:AC_ARG_WITH([cuda-runtime],
install/m4/x_ac_qe_cuda.m4:   [AS_HELP_STRING([--with-cuda-runtime=VAL],[CUDA runtime (Pascal: 8+, Volta: 9+) @<:@default=10.1@:>@])],
install/m4/x_ac_qe_cuda.m4:   [with_cuda_runtime=10.1])
install/m4/x_ac_qe_cuda.m4:AC_ARG_WITH([cuda-mpi],
install/m4/x_ac_qe_cuda.m4:   [AS_HELP_STRING([--with-cuda-mpi=VAL],[CUDA-aware MPI (yes|no) @<:@default=no@:>@])],
install/m4/x_ac_qe_cuda.m4:   [with_cuda_mpi=no])
install/m4/x_ac_qe_cuda.m4:if test "x$with_cuda" != "xno"
install/m4/x_ac_qe_cuda.m4:   AX_CHECK_COMPILE_FLAG([-cuda -gpu=cuda$with_cuda_runtime], [have_cudafor=yes], [have_cudafor=no], [], [MODULE test; use cudafor; END MODULE])
install/m4/x_ac_qe_cuda.m4:   if test "x$have_cudafor" != "xyes"
install/m4/x_ac_qe_cuda.m4:      AC_MSG_ERROR([You do not have the cudafor module. Are you using NVHPC compiler?])
install/m4/x_ac_qe_cuda.m4:   try_dflags="$try_dflags -D__CUDA"
install/m4/x_ac_qe_cuda.m4:   if test "$use_parallel" -eq 1 && test "$with_cuda_mpi" == "yes"; then 
install/m4/x_ac_qe_cuda.m4:      try_dflags="$try_dflags -D__GPU_MPI"
install/m4/x_ac_qe_cuda.m4:   cuda_extlibs="devxlib"
install/m4/x_ac_qe_cuda.m4:   cuda_libs="-cudalib=cufft,cublas,cusolver,curand \$(TOPDIR)/external/devxlib/src/libdevXlib.a"
install/m4/x_ac_qe_cuda.m4:   cuda_fflags="-cuda -gpu=cc$with_cuda_cc,cuda$with_cuda_runtime"
install/m4/x_ac_qe_cuda.m4:   cuda_fflags="$cuda_fflags \$(MOD_FLAG)\$(TOPDIR)/external/devxlib/src"
install/m4/x_ac_qe_cuda.m4:   cuda_fflags="$cuda_fflags \$(MOD_FLAG)\$(TOPDIR)/external/devxlib/include"
install/m4/x_ac_qe_cuda.m4:      cuda_fflags="$cuda_fflags -InvToolsExt.h -lnvToolsExt"
install/m4/x_ac_qe_cuda.m4:   runtime_major_version=`echo $with_cuda_runtime | cut -d. -f1`
install/m4/x_ac_qe_cuda.m4:   runtime_minor_version=`echo $with_cuda_runtime | cut -d. -f2`
install/m4/x_ac_qe_cuda.m4:       # CUDA toolkit v < 10.1: cusolver not available
install/m4/x_ac_qe_cuda.m4:       AC_MSG_ERROR([Unsupported CUDA Toolkit, too old])
install/m4/x_ac_qe_cuda.m4:   cuda_cflags=" -gpu=cc$with_cuda_cc,cuda$with_cuda_runtime"
install/m4/x_ac_qe_cuda.m4:   ldflags="$ldflags -cuda -gpu=cc$with_cuda_cc,cuda$with_cuda_runtime"
install/m4/x_ac_qe_cuda.m4:   gpu_arch="$with_cuda_cc"
install/m4/x_ac_qe_cuda.m4:   cuda_runtime="$with_cuda_runtime"
install/m4/x_ac_qe_cuda.m4:   cuda_fflags="$cuda_fflags -acc"
install/m4/x_ac_qe_cuda.m4:   cuda_cflags="$cuda_cflags -acc"
install/m4/x_ac_qe_cuda.m4:AC_SUBST(gpu_arch)
install/m4/x_ac_qe_cuda.m4:AC_SUBST(cuda_runtime)
install/m4/x_ac_qe_cuda.m4:AC_SUBST(cuda_cflags)
install/m4/x_ac_qe_cuda.m4:AC_SUBST(cuda_fflags)
install/m4/x_ac_qe_cuda.m4:AC_SUBST(cuda_libs)
install/m4/x_ac_qe_cuda.m4:AC_SUBST(cuda_extlibs)
install/configure:cuda_extlibs
install/configure:cuda_libs
install/configure:cuda_fflags
install/configure:cuda_cflags
install/configure:cuda_runtime
install/configure:gpu_arch
install/configure:with_cuda
install/configure:with_cuda_cc
install/configure:with_cuda_runtime
install/configure:with_cuda_mpi
install/configure:  --with-cuda=PATH        prefix where CUDA is installed [default=no]
install/configure:  --with-cuda-cc=VAL      GPU architecture (Kepler: 35, Pascal: 60, Volta: 70)
install/configure:  --with-cuda-runtime=VAL CUDA runtime (Pascal: 8+, Volta: 9+) [default=10.1]
install/configure:  --with-cuda-mpi=VAL     CUDA-aware MPI (yes|no) [default=no]
install/configure:# Checking CUDA...
install/configure:gpu_arch=
install/configure:cuda_runtime=
install/configure:cuda_cflags=
install/configure:cuda_fflags=
install/configure:cuda_libs=
install/configure:# FIXME: currently devxlib is needed also for non-CUDA compilation
install/configure:cuda_extlibs=devxlib
install/configure:# Provide your CUDA path with this
install/configure:# Check whether --with-cuda was given.
install/configure:if test "${with_cuda+set}" = set; then :
install/configure:  withval=$with_cuda;
install/configure:  with_cuda=no
install/configure:# Check whether --with-cuda-cc was given.
install/configure:if test "${with_cuda_cc+set}" = set; then :
install/configure:  withval=$with_cuda_cc;
install/configure:  with_cuda_cc=35
install/configure:# Check whether --with-cuda-runtime was given.
install/configure:if test "${with_cuda_runtime+set}" = set; then :
install/configure:  withval=$with_cuda_runtime;
install/configure:  with_cuda_runtime=10.1
install/configure:# Check whether --with-cuda-mpi was given.
install/configure:if test "${with_cuda_mpi+set}" = set; then :
install/configure:  withval=$with_cuda_mpi;
install/configure:  with_cuda_mpi=no
install/configure:if test "x$with_cuda" != "xno"
install/configure:   as_CACHEVAR=`$as_echo "ax_cv_check_fcflags__-cuda -gpu=cuda$with_cuda_runtime" | $as_tr_sh`
install/configure:{ $as_echo "$as_me:${as_lineno-$LINENO}: checking whether Fortran compiler accepts -cuda -gpu=cuda$with_cuda_runtime" >&5
install/configure:$as_echo_n "checking whether Fortran compiler accepts -cuda -gpu=cuda$with_cuda_runtime... " >&6; }
install/configure:  FCFLAGS="$FCFLAGS  -cuda -gpu=cuda$with_cuda_runtime"
install/configure:MODULE test; use cudafor; END MODULE
install/configure:  have_cudafor=yes
install/configure:  have_cudafor=no
install/configure:   if test "x$have_cudafor" != "xyes"
install/configure:      as_fn_error $? "You do not have the cudafor module. Are you using NVHPC compiler?" "$LINENO" 5
install/configure:   try_dflags="$try_dflags -D__CUDA"
install/configure:   if test "$use_parallel" -eq 1 && test "$with_cuda_mpi" == "yes"; then
install/configure:      try_dflags="$try_dflags -D__GPU_MPI"
install/configure:   cuda_extlibs="devxlib"
install/configure:   cuda_libs="-cudalib=cufft,cublas,cusolver,curand \$(TOPDIR)/external/devxlib/src/libdevXlib.a"
install/configure:   cuda_fflags="-cuda -gpu=cc$with_cuda_cc,cuda$with_cuda_runtime"
install/configure:   cuda_fflags="$cuda_fflags \$(MOD_FLAG)\$(TOPDIR)/external/devxlib/src"
install/configure:   cuda_fflags="$cuda_fflags \$(MOD_FLAG)\$(TOPDIR)/external/devxlib/include"
install/configure:      cuda_fflags="$cuda_fflags -InvToolsExt.h -lnvToolsExt"
install/configure:   runtime_major_version=`echo $with_cuda_runtime | cut -d. -f1`
install/configure:   runtime_minor_version=`echo $with_cuda_runtime | cut -d. -f2`
install/configure:       # CUDA toolkit v < 10.1: cusolver not available
install/configure:       as_fn_error $? "Unsupported CUDA Toolkit, too old" "$LINENO" 5
install/configure:   cuda_cflags=" -gpu=cc$with_cuda_cc,cuda$with_cuda_runtime"
install/configure:   ldflags="$ldflags -cuda -gpu=cc$with_cuda_cc,cuda$with_cuda_runtime"
install/configure:   gpu_arch="$with_cuda_cc"
install/configure:   cuda_runtime="$with_cuda_runtime"
install/configure:   cuda_fflags="$cuda_fflags -acc"
install/configure:   cuda_cflags="$cuda_cflags -acc"
install/configure:		    # NB: with nvidia hpc sdk 2020, linking to threaded mkl
install/configure:		    # NB: with (at least) nvidia 22.7, threaded mkl are
install/make.inc.in:# GPU architecture (Kepler: 35, Pascal: 60, Volta: 70, Turing: 75, Ampere: 80)
install/make.inc.in:GPU_ARCH=@gpu_arch@
install/make.inc.in:# CUDA runtime (should be compatible with the CUDA driver)
install/make.inc.in:CUDA_RUNTIME=@cuda_runtime@
install/make.inc.in:# CUDA F90 Flags
install/make.inc.in:CUDA_F90FLAGS=@cuda_fflags@ $(MOD_FLAG)$(TOPDIR)/external/devxlib/src
install/make.inc.in:# CUDA C Flags
install/make.inc.in:CUDA_CFLAGS=@cuda_cflags@
install/make.inc.in:CFLAGS         = @cflags@ $(DFLAGS) $(IFLAGS) $(CUDA_CFLAGS)
install/make.inc.in:F90FLAGS       = @f90flags@ @pre_fdflags@$(FDFLAGS) $(CUDA_F90FLAGS) $(IFLAGS) $(MODFLAGS)
install/make.inc.in:# CUDA libraries
install/make.inc.in:CUDA_LIBS=@cuda_libs@ -L$(TOPDIR)/external/devxlib/src -ldevXlib
install/make.inc.in:CUDA_EXTLIBS = @cuda_extlibs@
install/make.inc.in:                 $(CUDA_LIBS) $(SCALAPACK_LIBS) $(LAPACK_LIBS) $(FOX_LIB) \
external/devxlib.cmake:    qe_enable_cuda_fortran("${src_devxlib}")
external/.gitignore:src/FortCuda/*.x
external/.gitignore:src/FortCuda/*.o
external/.gitignore:src/FortCuda/*.a
external/.gitignore:src/FortCuda/*.mod
external/.gitignore:src/FortCuda/*.MOD
NEB/CMakeLists.txt:qe_enable_cuda_fortran("${src_neb_x}")
NEB/examples/ESM_example/reference/Al001+H_bc3_n215_2/Al001+H_bc3_n215.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_bc3_n215_1/Al001+H_bc3_n215.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_bc3_2/Al001+H_bc3.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_bc3_5/Al001+H_bc3.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_GCSCF_vm05_1/Al001+H_GCSCF_vm05.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_bc3_1/Al001+H_bc3.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_GCSCF_vm05_5/Al001+H_GCSCF_vm05.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_bc3_3/Al001+H_bc3.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_GCSCF_vm05_3/Al001+H_GCSCF_vm05.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_GCSCF_vm05_4/Al001+H_GCSCF_vm05.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_bc3_n215_5/Al001+H_bc3_n215.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_bc3_n215_3/Al001+H_bc3_n215.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_bc3_4/Al001+H_bc3.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_FCP_vm05_5/Al001+H_FCP_vm05.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_FCP_vm05_2/Al001+H_FCP_vm05.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_FCP_vm05_3/Al001+H_FCP_vm05.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_GCSCF_vm05_2/Al001+H_GCSCF_vm05.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_bc3_n215_4/Al001+H_bc3_n215.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_FCP_vm05_1/Al001+H_FCP_vm05.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
NEB/examples/ESM_example/reference/Al001+H_FCP_vm05_4/Al001+H_FCP_vm05.xml:    <creator NAME="PWSCF" VERSION="6.7GPU">XML file generated by PWSCF</creator>
PW/Doc/user_guide.tex:  \item Elena de Paoli (IOM-CNR) for porting to GPU of the RMM-DIIS
PW/CMakeLists.txt:   # GPU
PW/CMakeLists.txt:   src/g_psi_mod_gpu.f90
PW/CMakeLists.txt:   src/newd_gpu.f90
PW/CMakeLists.txt:   src/add_paw_to_deeq_gpu.f90
PW/CMakeLists.txt:   src/rotate_wfc_gpu.f90
PW/CMakeLists.txt:   src/usnldiag_gpu.f90
PW/CMakeLists.txt:   src/add_vuspsi_gpu.f90
PW/CMakeLists.txt:   src/hs_1psi_gpu.f90
PW/CMakeLists.txt:   src/g_psi_gpu.f90
PW/CMakeLists.txt:   src/add_vhub_to_deeq_gpu.f90
PW/CMakeLists.txt:   src/s_1psi_gpu.f90
PW/CMakeLists.txt:   src/h_psi_gpu.f90
PW/CMakeLists.txt:   src/utils_gpu.f90
PW/CMakeLists.txt:   src/vhpsi_gpu.f90
PW/CMakeLists.txt:   src/vloc_psi_gpu.f90
PW/CMakeLists.txt:   src/hs_psi_gpu.f90
PW/CMakeLists.txt:        # GPU
PW/CMakeLists.txt:        src/oscdft_functions_gpu.f90)
PW/CMakeLists.txt:qe_enable_cuda_fortran("${src_pw}")
PW/CMakeLists.txt:        qe_openacc_fortran)
PW/CMakeLists.txt:if(QE_ENABLE_CUDA)
PW/CMakeLists.txt:            CUDA::cublas)
PW/CMakeLists.txt:qe_enable_cuda_fortran("${src_pw_x}")
PW/src/clean_pw.f90:  USE control_flags,        ONLY : ts_vdw, mbd_vdw, use_gpu
PW/src/clean_pw.f90:#if defined (__CUDA)
PW/src/clean_pw.f90:  USE cudafor
PW/src/clean_pw.f90:#if defined(__CUDA)
PW/src/clean_pw.f90:    IF(use_gpu) istat = cudaHostUnregister(C_LOC(evc(1,1)))
PW/src/add_vuspsi_gpu.f90:SUBROUTINE add_vuspsi_gpu( lda, n, m, hpsi_d )
PW/src/add_vuspsi_gpu.f90:#if defined(__CUDA)
PW/src/add_vuspsi_gpu.f90:  CALL start_clock_gpu( 'add_vuspsi' )  
PW/src/add_vuspsi_gpu.f90:     CALL add_vuspsi_gamma_gpu()
PW/src/add_vuspsi_gpu.f90:     CALL add_vuspsi_nc_gpu ()
PW/src/add_vuspsi_gpu.f90:     CALL add_vuspsi_k_gpu()
PW/src/add_vuspsi_gpu.f90:  CALL stop_clock_gpu( 'add_vuspsi' )  
PW/src/add_vuspsi_gpu.f90:     SUBROUTINE add_vuspsi_gamma_gpu()
PW/src/add_vuspsi_gpu.f90:#if defined(__CUDA)
PW/src/add_vuspsi_gpu.f90:       USE cudafor
PW/src/add_vuspsi_gpu.f90:#if defined(__CUDA)
PW/src/add_vuspsi_gpu.f90:     END SUBROUTINE add_vuspsi_gamma_gpu
PW/src/add_vuspsi_gpu.f90:     SUBROUTINE add_vuspsi_k_gpu()
PW/src/add_vuspsi_gpu.f90:#if defined(__CUDA)
PW/src/add_vuspsi_gpu.f90:       USE cudafor
PW/src/add_vuspsi_gpu.f90:#if defined(__CUDA)
PW/src/add_vuspsi_gpu.f90:     END SUBROUTINE add_vuspsi_k_gpu
PW/src/add_vuspsi_gpu.f90:     SUBROUTINE add_vuspsi_nc_gpu()
PW/src/add_vuspsi_gpu.f90:#if defined(__CUDA)
PW/src/add_vuspsi_gpu.f90:       USE cudafor
PW/src/add_vuspsi_gpu.f90:#if defined(__CUDA)
PW/src/add_vuspsi_gpu.f90:     END SUBROUTINE add_vuspsi_nc_gpu
PW/src/add_vuspsi_gpu.f90:END SUBROUTINE add_vuspsi_gpu
PW/src/stres_ewa.f90:#if !defined(_OPENACC)
PW/src/stres_ewa.f90:#if !defined(_OPENACC)
PW/src/scf_mod.f90:#if defined(_OPENACC)
PW/src/scf_mod.f90:#if !defined(_OPENACC) 
PW/src/h_psi_meta.f90:  !FIXME! this variable should be mapped with openACC 
PW/src/h_psi_gpu.f90:SUBROUTINE h_psi_gpu( lda, n, m, psi, hpsi )
PW/src/h_psi_gpu.f90:  CALL start_clock_gpu( 'h_psi_bgrp' ); !write (*,*) 'start h_psi_bgrp'; FLUSH(6)
PW/src/h_psi_gpu.f90:        CALL h_psi__gpu( lda, n, m_end-m_start+1, psi(1,m_start), hpsi(1,m_start) )
PW/src/h_psi_gpu.f90:     CALL h_psi__gpu( lda, n, m, psi, hpsi )
PW/src/h_psi_gpu.f90:  CALL stop_clock_gpu( 'h_psi_bgrp' )
PW/src/h_psi_gpu.f90:END SUBROUTINE h_psi_gpu
PW/src/h_psi_gpu.f90:SUBROUTINE h_psi__gpu( lda, n, m, psi, hpsi )
PW/src/h_psi_gpu.f90:#if defined(__CUDA)
PW/src/h_psi_gpu.f90:  USE cudafor
PW/src/h_psi_gpu.f90:  USE exx,                     ONLY: use_ace, vexx, vexxace_gamma_gpu, vexxace_k_gpu
PW/src/h_psi_gpu.f90:  USE oscdft_functions_gpu,    ONLY : oscdft_h_psi_gpu
PW/src/h_psi_gpu.f90:#if defined(__CUDA)
PW/src/h_psi_gpu.f90:  CALL start_clock_gpu( 'h_psi' ); !write (*,*) 'start h_psi';FLUSH(6)
PW/src/h_psi_gpu.f90:  CALL start_clock_gpu( 'h_psi:pot' ); !write (*,*) 'start h_pot';FLUSH(6)
PW/src/h_psi_gpu.f90:           CALL start_clock_gpu( 'h_psi:calbec' )
PW/src/h_psi_gpu.f90:           CALL stop_clock_gpu( 'h_psi:calbec' )
PW/src/h_psi_gpu.f90:        CALL vloc_psi_gamma_gpu ( lda, n, m, psi, vrs(1,current_spin), hpsi )
PW/src/h_psi_gpu.f90:     CALL vloc_psi_nc_gpu ( lda, n, m, psi, vrs, hpsi )
PW/src/h_psi_gpu.f90:           CALL start_clock_gpu( 'h_psi:calbec' )
PW/src/h_psi_gpu.f90:           CALL stop_clock_gpu( 'h_psi:calbec' )
PW/src/h_psi_gpu.f90:        CALL vloc_psi_k_gpu ( lda, n, m, psi, vrs(1,current_spin), hpsi )
PW/src/h_psi_gpu.f90:     CALL start_clock_gpu( 'h_psi:calbec' )
PW/src/h_psi_gpu.f90:#if defined(__CUDA)
PW/src/h_psi_gpu.f90:     CALL stop_clock_gpu( 'h_psi:calbec' )
PW/src/h_psi_gpu.f90:     CALL add_vuspsi_gpu( lda, n, m, hpsi )
PW/src/h_psi_gpu.f90:  CALL stop_clock_gpu( 'h_psi:pot' )
PW/src/h_psi_gpu.f90:          CALL vhpsi_gpu( lda, n, m, psi, hpsi )  ! DFT+U
PW/src/h_psi_gpu.f90:           CALL vexxace_gamma_gpu(lda,m,psi,ee,hpsi)
PW/src/h_psi_gpu.f90:           CALL vexxace_k_gpu(lda,m,psi,ee,hpsi)
PW/src/h_psi_gpu.f90:     CALL oscdft_h_psi_gpu(oscdft_ctx, lda, n, m, psi, hpsi)
PW/src/h_psi_gpu.f90:  CALL stop_clock_gpu( 'h_psi' )
PW/src/h_psi_gpu.f90:END SUBROUTINE h_psi__gpu
PW/src/g_psi_gpu.f90:subroutine g_psi_gpu (lda, n, m, npol, psi_d, e_d)
PW/src/g_psi_gpu.f90:#if defined(__CUDA)
PW/src/g_psi_gpu.f90:  USE cudafor
PW/src/g_psi_gpu.f90:  USE g_psi_mod_gpum, ONLY : h_diag_d, s_diag_d, using_h_diag_d, using_s_diag_d
PW/src/g_psi_gpu.f90:#if defined(__CUDA)
PW/src/g_psi_gpu.f90:  call start_clock_gpu ('g_psi')
PW/src/g_psi_gpu.f90:  call stop_clock_gpu ('g_psi')
PW/src/g_psi_gpu.f90:end subroutine g_psi_gpu
PW/src/g_psi_gpu.f90:subroutine g_1psi_gpu (lda, n, psi_d, e_d)
PW/src/g_psi_gpu.f90:#if defined(__CUDA)
PW/src/g_psi_gpu.f90:  USE cudafor
PW/src/g_psi_gpu.f90:#if defined(__CUDA)
PW/src/g_psi_gpu.f90:  ! cast scalar to size 1 vector to exactly match g_psi_gpu argument type
PW/src/g_psi_gpu.f90:#if defined(__CUDA)
PW/src/g_psi_gpu.f90:  CALL g_psi_gpu (lda, n, 1, npol, psi_d, e_d_vec)
PW/src/g_psi_gpu.f90:end subroutine g_1psi_gpu
PW/src/force_lc.f90:#if !defined(_OPENACC)
PW/src/force_lc.f90:#if !defined(_OPENACC)
PW/src/stres_hub.f90:   CALL start_clock_gpu( 'stres_hub' )
PW/src/stres_hub.f90:   CALL stop_clock_gpu( 'stres_hub' )
PW/src/stres_hub.f90:   CALL start_clock_gpu('dprojdepsilon')
PW/src/stres_hub.f90:   CALL stop_clock_gpu('dprojdepsilon')
PW/src/stres_hub.f90:   CALL start_clock_gpu('dprojdepsilon')
PW/src/stres_hub.f90:   CALL stop_clock_gpu('dprojdepsilon')
PW/src/rotate_wfc_gpu.f90:SUBROUTINE rotate_wfc_gpu &
PW/src/rotate_wfc_gpu.f90:#if defined(__CUDA)
PW/src/rotate_wfc_gpu.f90:  EXTERNAL h_psi, s_psi, h_psi_gpu, s_psi_acc
PW/src/rotate_wfc_gpu.f90:  CALL start_clock_gpu( 'wfcrot' ); !write (*,*) 'start wfcrot' ; FLUSH(6)
PW/src/rotate_wfc_gpu.f90:        CALL rotate_wfc_gamma_gpu ( h_psi_gpu, s_psi_acc, overlap, &
PW/src/rotate_wfc_gpu.f90:        CALL rotate_wfc_k_gpu ( h_psi_gpu, s_psi_acc, overlap, &
PW/src/rotate_wfc_gpu.f90:  CALL stop_clock_gpu( 'wfcrot' )!; write (*,*) 'stop wfcrot' ; FLUSH(6)
PW/src/rotate_wfc_gpu.f90:END SUBROUTINE rotate_wfc_gpu
PW/src/read_file_new.f90:  USE control_flags,        ONLY : gamma_only, use_gpu
PW/src/read_file_new.f90:  call allocate_uspp(use_gpu,noncolin,lspinorb,tqr,nhm,nsp,nat,nspin)
PW/src/g_psi_mod_gpu.f90:   MODULE g_psi_mod_gpum
PW/src/g_psi_mod_gpu.f90:#if defined(__CUDA)
PW/src/g_psi_mod_gpu.f90:     USE cudafor
PW/src/g_psi_mod_gpu.f90:#if defined(__CUDA)
PW/src/g_psi_mod_gpu.f90:#if defined(__CUDA)  || defined(__CUDA_GNU)
PW/src/g_psi_mod_gpu.f90:#if defined(__CUDA) || defined(__CUDA_GNU)
PW/src/g_psi_mod_gpu.f90:#if defined(__CUDA)  || defined(__CUDA_GNU)
PW/src/g_psi_mod_gpu.f90:#if defined(__CUDA) || defined(__CUDA_GNU)
PW/src/g_psi_mod_gpu.f90:     SUBROUTINE deallocate_g_psi_mod_gpu
PW/src/g_psi_mod_gpu.f90:     END SUBROUTINE deallocate_g_psi_mod_gpu
PW/src/g_psi_mod_gpu.f90:   END MODULE g_psi_mod_gpum
PW/src/sum_band.f90:     ! Note: becsum and ebecsum are computed on GPU and copied to CPU
PW/src/sum_band.f90:  ! FIXME: next line  needed for old (21.7 or so) NVIDIA compilers
PW/src/sum_band.f90:  ! FIXME: rho should be on GPU always, not just in this routine
PW/src/sum_band.f90:     ! Task groups: no GPU, no special cases, etc.
PW/src/sum_band.f90:     ! ... and over k-points (unsymmetrized). Then the CPU and GPU copies are aligned.
PW/src/sum_band.f90:             ! BEWARE: untested on GPUs - should work if v_dmft is present on host  
PW/src/exx_band.f90:  USE control_flags,        ONLY : gamma_only, use_gpu
PW/src/exx_band.f90:#if defined(__CUDA)
PW/src/exx_band.f90:       IF(use_gpu .and. ( .not. allocated(igk_exx_d) ) ) THEN
PW/src/exx_band.f90:#if defined(__CUDA)
PW/src/exx_band.f90:#if defined(__CUDA)
PW/src/exx_band.f90:#if defined(__CUDA)
PW/src/exx_band.f90:#if defined(__CUDA)
PW/src/exx_band.f90:#if defined(__CUDA)
PW/src/exx_band.f90:#if defined(__CUDA)
PW/src/exx_band.f90:       IF(use_gpu) ALLOCATE(igk_exx_d, source=igk_exx)
PW/src/hs_psi_gpu.f90:SUBROUTINE hs_psi_gpu( lda, n, m, psi, hpsi, spsi )
PW/src/hs_psi_gpu.f90:  CALL h_psi__gpu ( lda, n, m, psi, hpsi ) ! apply H to m wfcs (no bgrp parallelization here)
PW/src/hs_psi_gpu.f90:END SUBROUTINE hs_psi_gpu
PW/src/oscdft_wfcO.f90:         USE control_flags,            ONLY : io_level, restart, gamma_only, use_gpu
PW/src/oscdft_wfcO.f90:         USE control_flags,            ONLY : offload_type, use_gpu
PW/src/oscdft_wfcO.f90:            IF (use_gpu) THEN
PW/src/g_psi_mod.f90:#if defined(__CUDA)
PW/src/atomic_wfc.f90:  !! "wfcatom", on GPU if input is an ACC variable, copied to CPU otherwise
PW/src/stres_gradcorr.f90:                         v1x, v2x, v3x, v1c, v2cm, v3c, gpu_args_=.TRUE. )
PW/src/stres_gradcorr.f90:        CALL xc_gcx( nrxx, nspin0, rhoaux, grho, sx, sc, v1x, v2x, v1c, v2c, gpu_args_=.TRUE. )
PW/src/stres_gradcorr.f90:                         v1x, v2x, v3x, v1c, v2cm, v3c, gpu_args_=.TRUE. )
PW/src/stres_gradcorr.f90:        CALL xc_gcx( nrxx, nspin0, rhoaux, grho, sx, sc, v1x, v2x, v1c, v2c, v2c_ud, gpu_args_=.TRUE. )
PW/src/init_us_2.f90:  SUBROUTINE init_us_2( npw_, igk_, q_, vkb_, run_on_gpu_ )
PW/src/init_us_2.f90:    LOGICAL, OPTIONAL, INTENT(IN) :: run_on_gpu_
PW/src/init_us_2.f90:    !! if false (default), copy output vkb back to CPU using OpenACC:
PW/src/init_us_2.f90:    LOGICAL :: run_on_gpu
PW/src/init_us_2.f90:    run_on_gpu = .FALSE.
PW/src/init_us_2.f90:    IF (PRESENT(run_on_gpu_)) run_on_gpu = run_on_gpu_
PW/src/init_us_2.f90:    IF (.not.run_on_gpu) THEN
PW/src/mix_rho.f90:#if defined(_OPENACC)
PW/src/mix_rho.f90:#if !defined(_OPENACC)
PW/src/mix_rho.f90:#if defined(_OPENACC)
PW/src/mix_rho.f90:#if !defined(_OPENACC)
PW/src/mix_rho.f90:#if defined (_OPENACC) 
PW/src/mix_rho.f90:#if !defined(_OPENACC) 
PW/src/mix_rho.f90:#if defined(_OPENACC) 
PW/src/mix_rho.f90:#if !defined(_OPENACC)
PW/src/mix_rho.f90:#if defined (_OPENACC) 
PW/src/mix_rho.f90:#if !defined(_OPENACC) 
PW/src/mix_rho.f90:#if defined(_OPENACC) 
PW/src/mix_rho.f90:#if !defined(_OPENACC)
PW/src/mix_rho.f90:#if defined (_OPENACC)
PW/src/mix_rho.f90:#if !defined(_OPENACC)
PW/src/mix_rho.f90:#if defined (_OPENACC)
PW/src/mix_rho.f90:#if !defined(_OPENACC)
PW/src/mix_rho.f90:#if defined(_OPENACC) 
PW/src/mix_rho.f90:#if defined (_OPENACC) 
PW/src/mix_rho.f90:#if !defined(_OPENACC) 
PW/src/mix_rho.f90:#if defined(_OPENACC) 
PW/src/mix_rho.f90:#if !defined(_OPENACC) 
PW/src/mix_rho.f90:#if defined(_OPENACC) 
PW/src/mix_rho.f90:#if !defined(_OPENACC) 
PW/src/mix_rho.f90:#if defined(_OPENACC)
PW/src/mix_rho.f90:#if !defined(_OPENACC) 
PW/src/sic.f90:   USE control_flags,    ONLY : use_gpu, lbfgs
PW/src/oscdft_functions_gpu.f90:MODULE oscdft_functions_gpu
PW/src/oscdft_functions_gpu.f90:                                        check_bec_type_unallocated_gpu
PW/src/oscdft_functions_gpu.f90:   PUBLIC oscdft_h_diag_gpu, oscdft_h_psi_gpu
PW/src/oscdft_functions_gpu.f90:      SUBROUTINE oscdft_h_diag_gpu(ctx)
PW/src/oscdft_functions_gpu.f90:         USE g_psi_mod_gpum,   ONLY : h_diag_d, using_h_diag_d
PW/src/oscdft_functions_gpu.f90:         CALL start_clock_gpu("oscdft_hdiag")
PW/src/oscdft_functions_gpu.f90:         CALL stop_clock_gpu("oscdft_hdiag")
PW/src/oscdft_functions_gpu.f90:      END SUBROUTINE oscdft_h_diag_gpu
PW/src/oscdft_functions_gpu.f90:      SUBROUTINE oscdft_h_psi_gpu(ctx, lda, n, m, psi, hpsi)
PW/src/oscdft_functions_gpu.f90:         CALL start_clock_gpu("oscdft_hpsi")
PW/src/oscdft_functions_gpu.f90:         CALL stop_clock_gpu("oscdft_hpsi")
PW/src/oscdft_functions_gpu.f90:      END SUBROUTINE oscdft_h_psi_gpu
PW/src/oscdft_functions_gpu.f90:END MODULE oscdft_functions_gpu
PW/src/two_chem.f90:  USE control_flags,        ONLY : use_gpu
PW/src/two_chem.f90:  IF (use_gpu) CALL errore('init_twochem', 'twochem with GPU not present in this version',1)  
PW/src/Makefile:# GPU versions of routines
PW/src/Makefile:  g_psi_mod_gpu.o \
PW/src/Makefile:  g_psi_gpu.o \
PW/src/Makefile:  h_psi_gpu.o \
PW/src/Makefile:  vhpsi_gpu.o \
PW/src/Makefile:  vloc_psi_gpu.o \
PW/src/Makefile:  usnldiag_gpu.o \
PW/src/Makefile:  add_vuspsi_gpu.o \
PW/src/Makefile:  newd_gpu.o \
PW/src/Makefile:  add_paw_to_deeq_gpu.o \
PW/src/Makefile:  add_vhub_to_deeq_gpu.o \
PW/src/Makefile:  rotate_wfc_gpu.o \
PW/src/Makefile:  hs_1psi_gpu.o \
PW/src/Makefile:  hs_psi_gpu.o \
PW/src/Makefile:  s_1psi_gpu.o \
PW/src/Makefile:  utils_gpu.o \
PW/src/Makefile:  oscdft_functions_gpu.o
PW/src/newd.f90:  ! sync with GPUs
PW/src/setup.f90:  USE control_flags, ONLY : use_para_diag, use_gpu
PW/src/setup.f90:  LOGICAL, EXTERNAL  :: check_gpu_support
PW/src/setup.f90:  ! GPUs (not sure it serves any purpose)
PW/src/setup.f90:  use_gpu = check_gpu_support( )
PW/src/setup.f90:#if defined (__CUDA_OPTIMIZED)
PW/src/setup.f90:  ! linear algebra - for GPUs, ndiag = 1; otherwise, a value ensuring that
PW/src/setup.f90:  if ( ndiag_ == 0 .AND. use_gpu ) ndiag_ = 1
PW/src/setup.f90:LOGICAL FUNCTION check_gpu_support( )
PW/src/setup.f90:  ! Minimal case: returns true if compiled for GPUs
PW/src/setup.f90:#if defined(__CUDA)
PW/src/setup.f90:  check_gpu_support = .TRUE.
PW/src/setup.f90:  check_gpu_support = .FALSE.
PW/src/setup.f90:END FUNCTION check_gpu_support
PW/src/compute_becsum.f90:     ! ... actual calculation is performed (on GPU) inside routine "sum_bec"
PW/src/compute_becsum.f90:  ! ... No need to symmetrize becsums or to align GPU and CPU copies: they are used only here.
PW/src/pwcom.f90:#if defined (__CUDA)
PW/src/print_clock_pw.f90:   CALL print_clock( 'init_us_2:gpu' )
PW/src/oscdft_functions.f90:         USE control_flags,                ONLY : use_gpu
PW/src/oscdft_functions.f90:         USE g_psi_mod_gpum,   ONLY : using_h_diag
PW/src/force_cc.f90:#if !defined(_OPENACC)
PW/src/force_cc.f90:#if !defined(_OPENACC)
PW/src/ldaU.f90:#if defined(__CUDA)
PW/src/addusstress.f90:#if defined(_OPENACC)
PW/src/addusstress.f90:#if defined(_OPENACC)
PW/src/addusstress.f90:#if !defined(_OPENACC)
PW/src/hinit1.f90:  USE control_flags,       ONLY : tqr, use_gpu
PW/src/hinit1.f90:  USE dfunct_gpum,         ONLY : newd_gpu
PW/src/hinit1.f90:  IF (.not. use_gpu) CALL newd()
PW/src/hinit1.f90:  IF (      use_gpu) CALL newd_gpu()
PW/src/summary.f90:  USE environment,     ONLY : print_cuda_info
PW/src/summary.f90:  ! ... CUDA
PW/src/summary.f90:  CALL print_cuda_info(check_use_gpu = .TRUE.)
PW/src/stres_knl.f90:           ! Stupid workaround for NVidia HPC-SDK v. < 22
PW/src/add_vhub_to_deeq_gpu.f90:SUBROUTINE add_vhub_to_deeq_gpu( deeq_d )
PW/src/add_vhub_to_deeq_gpu.f90:#if defined(__CUDA)
PW/src/add_vhub_to_deeq_gpu.f90:END SUBROUTINE add_vhub_to_deeq_gpu
PW/src/s_1psi_gpu.f90:SUBROUTINE s_1psi_gpu( npwx, n, psi, spsi )
PW/src/s_1psi_gpu.f90:#if defined(__CUDA)
PW/src/s_1psi_gpu.f90:  CALL start_clock_gpu( 's_1psi' )
PW/src/s_1psi_gpu.f90:#if defined(__CUDA)
PW/src/s_1psi_gpu.f90:  CALL stop_clock_gpu( 's_1psi' )
PW/src/s_1psi_gpu.f90:END SUBROUTINE s_1psi_gpu
PW/src/oscdft_wavefunction_subs.f90:         USE control_flags,    ONLY : gamma_only, use_gpu
PW/src/oscdft_wavefunction_subs.f90:         IF (use_gpu) THEN
PW/src/oscdft_wavefunction_subs.f90:            CALL laxlib_cdiaghg_gpu(m, m, overlap, s, m, e, work, me_bgrp,&
PW/src/s_psi_acc.f90:#if defined (__CUDA)
PW/src/s_psi_acc.f90:#if defined(__CUDA)
PW/src/s_psi_acc.f90:#if defined(__CUDA)  
PW/src/s_psi_acc.f90:#if defined(__CUDA)  
PW/src/s_psi_acc.f90:#if defined(__CUDA)
PW/src/s_psi_acc.f90:#if defined(__CUDA)
PW/src/s_psi_acc.f90:#if defined(__CUDA)
PW/src/s_psi_acc.f90:#if defined(__CUDA)
PW/src/newd_gpu.f90:MODULE dfunct_gpum
PW/src/newd_gpu.f90:SUBROUTINE newq_gpu(vr,deeq_d,skip_vltot)
PW/src/newd_gpu.f90:#if defined(__CUDA)
PW/src/newd_gpu.f90:  USE cudafor
PW/src/newd_gpu.f90:#if defined(__CUDA)
PW/src/newd_gpu.f90:#if defined(__CUDA)
PW/src/newd_gpu.f90:#if defined(__CUDA)
PW/src/newd_gpu.f90:END SUBROUTINE newq_gpu
PW/src/newd_gpu.f90:SUBROUTINE newd_gpu( ) 
PW/src/newd_gpu.f90:#if defined(__CUDA)
PW/src/newd_gpu.f90:  use cudafor
PW/src/newd_gpu.f90:#if defined(__CUDA)
PW/src/newd_gpu.f90:  CALL start_clock_gpu( 'newd' )
PW/src/newd_gpu.f90:  ! move atom type info to GPU
PW/src/newd_gpu.f90:     CALL newq_gpu(v%of_r,deeq,.false.)
PW/src/newd_gpu.f90:    call add_paw_to_deeq_gpu(deeq)
PW/src/newd_gpu.f90:           CALL newd_so_gpu(nt)
PW/src/newd_gpu.f90:           CALL newd_nc_gpu(nt)
PW/src/newd_gpu.f90:    CALL add_paw_to_deeq_gpu(deeq)
PW/src/newd_gpu.f90:    CALL add_vhub_to_deeq_gpu(deeq)
PW/src/newd_gpu.f90:  CALL stop_clock_gpu( 'newd' )
PW/src/newd_gpu.f90:    SUBROUTINE newd_so_gpu(nt)
PW/src/newd_gpu.f90:    END SUBROUTINE newd_so_gpu
PW/src/newd_gpu.f90:    SUBROUTINE newd_nc_gpu(nt)
PW/src/newd_gpu.f90:    END SUBROUTINE newd_nc_gpu
PW/src/newd_gpu.f90:END SUBROUTINE newd_gpu
PW/src/newd_gpu.f90:END MODULE dfunct_gpum
PW/src/force_corr.f90:#if defined(_OPENACC)
PW/src/paw_onecenter.f90:#if defined(_OPENACC)
PW/src/paw_onecenter.f90:                               gpu_args_=.TRUE. )
PW/src/paw_onecenter.f90:                               gpu_args_=.TRUE. )
PW/src/paw_onecenter.f90:      CALL xc( im_sum, 4, 2, arho, ex, ec, vx, vc, gpu_args_=.TRUE. )
PW/src/paw_onecenter.f90:#if defined(_OPENACC)
PW/src/paw_onecenter.f90:#if defined(_OPENACC)
PW/src/paw_onecenter.f90:#if defined(_OPENACC)
PW/src/paw_onecenter.f90:#if defined(_OPENACC)
PW/src/paw_onecenter.f90:      CALL xc_gcx( im_sum, 1, rho_full, gradx, sx, sc, v1x, v2x, v1c, v2c, gpu_args_=.TRUE. )
PW/src/paw_onecenter.f90:      CALL xc_gcx( im_sum, 2, rho_full, gradx, sx, sc, v1x, v2x, v1c, v2c, v2cud, gpu_args_=.TRUE. )
PW/src/paw_onecenter.f90:#if defined(_OPENACC)
PW/src/paw_onecenter.f90:#if defined(_OPENACC)
PW/src/paw_onecenter.f90:       CALL dgcxc( im_sum, nspin_mag, r, grad, dsvxc_rr, dsvxc_sr, dsvxc_ss, gpu_args_=.TRUE. )
PW/src/paw_onecenter.f90:       CALL xc_gcx( im_sum, nspin_mag, r, gradsw, sx, sc, v1x, v2x, v1c, v2c, gpu_args_=.TRUE. )
PW/src/paw_onecenter.f90:       CALL dgcxc( im_sum, nspin_gga, r, grad, dsvxc_rr, dsvxc_sr, dsvxc_ss, gpu_args_=.TRUE. )
PW/src/paw_onecenter.f90:       CALL xc_gcx( im_sum, nspin_gga, r, gradsw, sx, sc, v1x, v2x, v1c, v2c, v2c_ud, gpu_args_=.TRUE. )
PW/src/vhpsi_gpu.f90:#if !defined(__CUDA)
PW/src/vhpsi_gpu.f90:SUBROUTINE vhpsi_gpu( ldap, np, mps, psip, hpsi )
PW/src/vhpsi_gpu.f90:#if defined(__CUDA)
PW/src/vhpsi_gpu.f90:  USE cudafor
PW/src/vhpsi_gpu.f90:  CALL start_clock_gpu( 'vhpsi' )
PW/src/vhpsi_gpu.f90:     CALL vhpsi_U_gpu()  ! DFT+U
PW/src/vhpsi_gpu.f90:     CALL errore('vhpsi', 'DFT+U+V case not implemented for GPU', 1 )
PW/src/vhpsi_gpu.f90:  CALL stop_clock_gpu( 'vhpsi' )
PW/src/vhpsi_gpu.f90:SUBROUTINE vhpsi_U_gpu()
PW/src/vhpsi_gpu.f90:#if defined(__CUDA)
PW/src/vhpsi_gpu.f90:END SUBROUTINE vhpsi_U_gpu
PW/src/vhpsi_gpu.f90:END SUBROUTINE vhpsi_gpu
PW/src/run_pwscf.f90:     IF ( ierr .ne. 0 ) CALL infomsg( 'run_pwscf', 'Cannot reset GPU buffers! Some buffers still locked.' )
PW/src/vloc_psi_gpu.f90:SUBROUTINE vloc_psi_gamma_gpu( lda, n, m, psi_d, v, hpsi_d )
PW/src/vloc_psi_gpu.f90:  IF ( dffts%has_task_groups ) CALL errore('Vloc_psi_gpu','no task groups!',1)
PW/src/vloc_psi_gpu.f90:  CALL start_clock_gpu( 'vloc_psi' )
PW/src/vloc_psi_gpu.f90:  CALL stop_clock_gpu ('vloc_psi')
PW/src/vloc_psi_gpu.f90:END SUBROUTINE vloc_psi_gamma_gpu
PW/src/vloc_psi_gpu.f90:SUBROUTINE vloc_psi_k_gpu( lda, n, m, psi_d, v, hpsi_d )
PW/src/vloc_psi_gpu.f90:  !! Calculation of Vloc*psi using dual-space technique - k-points. GPU double.
PW/src/vloc_psi_gpu.f90:  IF ( dffts%has_task_groups ) CALL errore('Vloc_psi_gpu','no task groups!',2)
PW/src/vloc_psi_gpu.f90:  CALL start_clock_gpu ('vloc_psi')
PW/src/vloc_psi_gpu.f90:  CALL stop_clock_gpu( 'vloc_psi' )
PW/src/vloc_psi_gpu.f90:END SUBROUTINE vloc_psi_k_gpu
PW/src/vloc_psi_gpu.f90:SUBROUTINE vloc_psi_nc_gpu( lda, n, m, psi_d, v, hpsi_d )
PW/src/vloc_psi_gpu.f90:  IF ( dffts%has_task_groups ) CALL errore('Vloc_psi_gpu','no task groups!',3)
PW/src/vloc_psi_gpu.f90:  CALL start_clock_gpu ('vloc_psi')
PW/src/vloc_psi_gpu.f90:  CALL stop_clock_gpu ('vloc_psi')
PW/src/vloc_psi_gpu.f90:END SUBROUTINE vloc_psi_nc_gpu
PW/src/hs_1psi_gpu.f90:SUBROUTINE hs_1psi_gpu( lda, n, psi, hpsi, spsi )
PW/src/hs_1psi_gpu.f90:  CALL start_clock_gpu( 'hs_1psi' )
PW/src/hs_1psi_gpu.f90:           CALL h_psi_gpu( lda, n, 1, psi, hpsi )
PW/src/hs_1psi_gpu.f90:  CALL h_psi_gpu( lda, n, 1, psi, hpsi ) ! apply H to a single wfc (no bgrp parallelization here)
PW/src/hs_1psi_gpu.f90:  CALL stop_clock_gpu( 'hs_1psi' )
PW/src/hs_1psi_gpu.f90:END SUBROUTINE hs_1psi_gpu
PW/src/oscdft_occupations.f90:         USE control_flags, ONLY : use_gpu
PW/src/oscdft_occupations.f90:         IF (use_gpu) THEN
PW/src/oscdft_occupations.f90:            CALL new_ns_normal_gpu(inp, idx, wfcS, nst, wfc_evc)
PW/src/oscdft_occupations.f90:      SUBROUTINE new_ns_normal_gpu(inp, idx, wfcS, nst, wfc_evc)
PW/src/oscdft_occupations.f90:         CALL start_clock_gpu("oscdft_ns")
PW/src/oscdft_occupations.f90:         CALL stop_clock_gpu("oscdft_ns")
PW/src/oscdft_occupations.f90:      END SUBROUTINE new_ns_normal_gpu
PW/src/oscdft_wavefunction.f90:          check_bec_type_unallocated_gpu
PW/src/addusforce.f90:#if defined(_OPENACC)
PW/src/addusforce.f90:#if !defined(_OPENACC)
PW/src/oscdft_forces_subs.f90:         USE control_flags,      ONLY : gamma_only, use_gpu, offload_type
PW/src/oscdft_forces_subs.f90:            IF (use_gpu) THEN
PW/src/oscdft_forces_subs.f90:#if defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if !defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if !defined(__CUDA)
PW/src/oscdft_forces_subs.f90:         ! openacc does not like deriv%...
PW/src/oscdft_forces_subs.f90:#if defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if !defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if !defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if !defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if !defined(__CUDA)
PW/src/oscdft_forces_subs.f90:         ! openacc does not like deriv%...
PW/src/oscdft_forces_subs.f90:#if defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if !defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if !defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if !defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if !defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if !defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if defined(__CUDA)
PW/src/oscdft_forces_subs.f90:#if !defined(__CUDA)
PW/src/v_of_rho.f90:                     v1x, v2x, v3x, v1c, v2c, v3c, gpu_args_=.TRUE. )
PW/src/v_of_rho.f90:                     v1x, v2x, v3x, v1c, v2c, v3c, gpu_args_=.TRUE. )
PW/src/v_of_rho.f90:     CALL xc( dfftp_nnr, 1, 1, rho%of_r, ex, ec, vx, vc, gpu_args_=.TRUE. )
PW/src/v_of_rho.f90:     CALL xc( dfftp_nnr, 2, 2, rho%of_r, ex, ec, vx, vc, gpu_args_=.TRUE. )
PW/src/v_of_rho.f90:      CALL xc( dfftp_nnr, 4, 2, rho%of_r, ex, ec, vx, vc, gpu_args_=.TRUE. )
PW/src/force_hub.f90:   CALL start_clock_gpu( 'force_hub' )
PW/src/force_hub.f90:   CALL stop_clock_gpu( 'force_hub' )
PW/src/force_hub.f90:   CALL start_clock_gpu( 'dngdtau' )
PW/src/force_hub.f90:   CALL stop_clock_gpu( 'dngdtau' )
PW/src/force_hub.f90:   CALL start_clock_gpu( 'dprojdtau' )
PW/src/force_hub.f90:   CALL stop_clock_gpu( 'dprojdtau' )
PW/src/force_hub.f90:   CALL start_clock_gpu( 'dprojdtau' )
PW/src/force_hub.f90:   CALL stop_clock_gpu( 'dprojdtau' )
PW/src/exx.f90:#error USE_MANY_FFT not implemented in the GPU version.
PW/src/exx.f90:  USE control_flags,        ONLY : gamma_only, tqr, use_gpu, many_fft
PW/src/exx.f90:  !! GPU duplicated data
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:    IF( .NOT. ALLOCATED(x_occupation_d) .and. use_gpu) &
PW/src/exx.f90:    IF (use_gpu) x_occupation_d = x_occupation
PW/src/exx.f90:      IF (.not. allocated(exxbuff_d) .and. use_gpu) THEN
PW/src/exx.f90:       IF (use_gpu) THEN
PW/src/exx.f90:#if defined (__CUDA)
PW/src/exx.f90:#if defined (__CUDA)
PW/src/exx.f90:                   IF (use_gpu) CALL dev_buf%lock_buffer(psic_nc_d, (/nrxxs, npol/), ierr)
PW/src/exx.f90:                   IF (use_gpu) psic_nc_d = psic_nc
PW/src/exx.f90:                      IF (use_gpu) THEN
PW/src/exx.f90:                      IF (use_gpu) THEN
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:                IF (use_gpu) CALL dev_buf%release_buffer(psic_nc_d, ierr)
PW/src/exx.f90:                IF (use_gpu) exxbuff = exxbuff_d
PW/src/exx.f90:          IF (.not. use_gpu) CALL vexx_gamma( lda, n, m, psi, hpsi, becpsi )
PW/src/exx.f90:          IF (      use_gpu) CALL vexx_gamma_gpu( lda, n, m, psi, hpsi, becpsi )
PW/src/exx.f90:          IF (.not. use_gpu) CALL vexx_gamma( lda, n, m, psi_exx, hpsi_exx, becpsi )
PW/src/exx.f90:          IF (      use_gpu) CALL vexx_gamma_gpu( lda, n, m, psi_exx, hpsi_exx, becpsi )
PW/src/exx.f90:          IF (.not. use_gpu) CALL vexx_k( lda, n, m, psi, hpsi, becpsi )
PW/src/exx.f90:          IF (      use_gpu) CALL vexx_k_gpu( lda, n, m, psi, hpsi, becpsi )
PW/src/exx.f90:          IF (.not. use_gpu) CALL vexx_k( lda, n, m, psi_exx, hpsi_exx, becpsi )
PW/src/exx.f90:          IF (      use_gpu) CALL vexx_k_gpu( lda, n, m, psi_exx, hpsi_exx, becpsi )
PW/src/exx.f90:  SUBROUTINE vexx_gamma_gpu(lda, n, m, psi, hpsi, becpsi)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:    ! CUDA Sync
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:  END SUBROUTINE vexx_gamma_gpu
PW/src/exx.f90:  SUBROUTINE vexx_k_gpu(lda, n, m, psi, hpsi, becpsi)
PW/src/exx.f90:    !CUDA stuff
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:  END SUBROUTINE vexx_k_gpu
PW/src/exx.f90:#if defined (__CUDA)
PW/src/exx.f90:#if defined (__CUDA)
PW/src/exx.f90:  SUBROUTINE vexxace_gamma_gpu( nnpw, nbnd, phi_d, exxe, vphi_d )
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:    CALL start_clock_gpu( 'vexxace' )
PW/src/exx.f90:    CALL matcalc_gpu( '<xi|phi>', .FALSE. , 0, nnpw, nbndproj, nbnd, xi_d, phi_d, rmexx_d, exxe )
PW/src/exx.f90:          CALL matcalc_gpu( 'ACE', .TRUE., 0, nnpw, nbnd, nbnd, phi_d, vv_d, rmexx_d, exxe )
PW/src/exx.f90:          CALL matcalc_gpu( 'ACE', .TRUE., 0, nnpw, nbnd, nbnd, phi_d, vphi_d, rmexx_d, exxe )
PW/src/exx.f90:    CALL stop_clock_gpu( 'vexxace' )
PW/src/exx.f90:  END SUBROUTINE vexxace_gamma_gpu
PW/src/exx.f90:  SUBROUTINE vexxace_k_gpu( nnpw, nbnd, phi_d, exxe, vphi_d )
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:#if defined(__CUDA)
PW/src/exx.f90:    CALL start_clock_gpu( 'vexxace' )
PW/src/exx.f90:    CALL matcalc_k_gpu( '<xi|phi>', .FALSE., 0, current_k, npwx*npol, nbndproj, nbnd, &
PW/src/exx.f90:          CALL matcalc_k_gpu( 'ACE', .TRUE., 0, current_k, npwx*npol, nbnd, nbnd, phi_d, &
PW/src/exx.f90:          CALL matcalc_k_gpu( 'ACE', .TRUE., 0, current_k, npwx*npol, nbnd, nbnd, phi_d, &
PW/src/exx.f90:    CALL stop_clock_gpu( 'vexxace' )
PW/src/exx.f90:  END SUBROUTINE vexxace_k_gpu
PW/src/forces.f90:#if defined(__CUDA)
PW/src/forces.f90:#if defined(__CUDA)
PW/src/forces.f90:  IF (ierr .ne. 0) CALL infomsg('forces', 'Cannot reset GPU buffers! Some buffers still locked.')
PW/src/forces.f90:#if defined(__CUDA)
PW/src/forces.f90:  IF (ierr .ne. 0) CALL errore('forces', 'Cannot reset GPU buffers! Buffers still locked: ', abs(ierr))
PW/src/oscdft_base.f90:      SUBROUTINE print_oscdft_clocks ! TODO: GPU
PW/src/oscdft_base.f90:         USE control_flags, ONLY : use_gpu
PW/src/usnldiag_gpu.f90:SUBROUTINE usnldiag_gpu( npw, h_diag_d, s_diag_d )
PW/src/usnldiag_gpu.f90:#if defined(__CUDA)
PW/src/usnldiag_gpu.f90:END SUBROUTINE usnldiag_gpu
PW/src/utils_gpu.f90:SUBROUTINE matcalc_gpu( label, DoE, PrtMat, ninner, n, m, U, V, mat, ee )
PW/src/utils_gpu.f90:#if defined(__CUDA)
PW/src/utils_gpu.f90:  CALL start_clock_gpu('matcalc')
PW/src/utils_gpu.f90:  IF( PrtMat > 1 ) CALL errore('matcalc_gpu', 'cannot print matrix', 1)
PW/src/utils_gpu.f90:  CALL stop_clock_gpu('matcalc')
PW/src/utils_gpu.f90:END SUBROUTINE matcalc_gpu
PW/src/utils_gpu.f90:SUBROUTINE matcalc_k_gpu (label, DoE, PrtMat, ik, ninner, n, m, U, V, mat, ee)
PW/src/utils_gpu.f90:#if defined(__CUDA)
PW/src/utils_gpu.f90:  CALL start_clock_gpu('matcalc')
PW/src/utils_gpu.f90:  IF( PrtMat > 1 ) CALL errore('matcalc_k_gpu', 'cannot print matrix', 1)
PW/src/utils_gpu.f90:  CALL stop_clock_gpu('matcalc')
PW/src/utils_gpu.f90:END SUBROUTINE matcalc_k_gpu
PW/src/compute_deff.f90:    ! ... set up index arrays to fill 'deff' in on gpu
PW/src/g_psi.f90:  USE g_psi_mod_gpum, ONLY : using_h_diag, using_s_diag
PW/src/force_us.f90:#if defined(_OPENACC)
PW/src/allocate_wfc.f90:#if defined (__CUDA)
PW/src/allocate_wfc.f90:  use cudafor
PW/src/allocate_wfc.f90:  USE control_flags,       ONLY : use_gpu
PW/src/allocate_wfc.f90:#if defined(__CUDA)
PW/src/allocate_wfc.f90:  IF(use_gpu) istat = cudaHostRegister(C_LOC(evc(1,1)), sizeof(evc), cudaHostRegisterMapped)
PW/src/c_bands.f90:  USE control_flags,        ONLY : ethr, isolve, restart, use_gpu, iverbosity
PW/src/c_bands.f90:     IF ( ierr .ne. 0 ) CALL infomsg( 'c_bands', 'Cannot reset GPU buffers! Some buffers still locked.' )
PW/src/c_bands.f90:                                   gamma_only, use_para_diag, use_gpu
PW/src/c_bands.f90:  USE g_psi_mod_gpum,       ONLY : h_diag_d, s_diag_d, using_h_diag, using_s_diag, using_h_diag_d, using_s_diag_d
PW/src/c_bands.f90:  USE oscdft_functions_gpu, ONLY : oscdft_h_diag_gpu
PW/src/c_bands.f90:#if defined(__CUDA)
PW/src/c_bands.f90:  EXTERNAL h_psi_gpu, s_psi_acc, g_psi_gpu
PW/src/c_bands.f90:  EXTERNAL hs_psi_gpu
PW/src/c_bands.f90:  EXTERNAL hs_1psi_gpu, s_1psi_gpu
PW/src/c_bands.f90:  external g_1psi_gpu
PW/src/c_bands.f90:                IF (.not. use_gpu) THEN
PW/src/c_bands.f90:                   CALL rotate_wfc_gpu( npwx, npw, nbnd, gstart, nbnd, evc, npol, okvan, evc, et(1,ik) )
PW/src/c_bands.f90:             IF (.not. use_gpu) THEN
PW/src/c_bands.f90:                CALL rcgdiagg_gpu( hs_1psi_gpu, s_1psi_gpu, h_diag_d, &
PW/src/c_bands.f90:             IF (.not. use_gpu) THEN
PW/src/c_bands.f90:               CALL ppcg_gamma_gpu( h_psi_gpu, s_psi_acc, okvan, h_diag_d, &
PW/src/c_bands.f90:             IF (.not. use_gpu ) THEN
PW/src/c_bands.f90:               CALL paro_gamma_new( h_psi_gpu, s_psi_acc, hs_psi_gpu, g_1psi_gpu, okvan, &
PW/src/c_bands.f90:              IF (.not. use_gpu ) THEN
PW/src/c_bands.f90:                CALL paro_gamma_new( h_psi_gpu, s_psi_acc, hs_psi_gpu, g_1psi_gpu, okvan, &
PW/src/c_bands.f90:             IF (.not. use_gpu) THEN
PW/src/c_bands.f90:#if defined(__CUDA)
PW/src/c_bands.f90:                CALL rotate_xpsi( h_psi, s_psi, h_psi_gpu, s_psi_acc, npwx, npw, nbnd, nbnd, evc, npol, okvan, &
PW/src/c_bands.f90:          IF (.not. use_gpu) THEN
PW/src/c_bands.f90:             CALL rrmmdiagg_gpu( h_psi_gpu, s_psi_acc, npwx, npw, nbnd, evc, hevc, sevc, &
PW/src/c_bands.f90:       IF (.not. use_gpu) THEN
PW/src/c_bands.f90:          CALL gram_schmidt_gamma_gpu( npwx, npw, nbnd, evc, hevc, sevc, et(1,ik), &
PW/src/c_bands.f90:       IF ( .not. use_gpu ) THEN
PW/src/c_bands.f90:          IF (use_oscdft) CALL oscdft_h_diag_gpu(oscdft_ctx)
PW/src/c_bands.f90:          CALL usnldiag_gpu( npw, h_diag_d, s_diag_d )
PW/src/c_bands.f90:          IF (.not. use_gpu) THEN
PW/src/c_bands.f90:                CALL pregterg_gpu( h_psi_gpu, s_psi_acc, okvan, g_psi_gpu, &
PW/src/c_bands.f90:                CALL regterg (  h_psi_gpu, s_psi_acc, okvan, g_psi_gpu, &
PW/src/c_bands.f90:                IF ( .not. use_gpu ) THEN
PW/src/c_bands.f90:                   CALL rotate_wfc_gpu( npwx, npw, nbnd, gstart, nbnd, evc, npol, okvan, evc, et(1,ik) )
PW/src/c_bands.f90:             IF ( .not. use_gpu ) THEN
PW/src/c_bands.f90:                CALL ccgdiagg_gpu( hs_1psi_gpu, s_1psi_gpu, h_diag_d, &
PW/src/c_bands.f90:             IF ( .not. use_gpu ) THEN
PW/src/c_bands.f90:               CALL ppcg_k_gpu( h_psi_gpu, s_psi_acc, okvan, h_diag_d, &
PW/src/c_bands.f90:             IF ( .not. use_gpu ) THEN
PW/src/c_bands.f90:               CALL paro_k_new( h_psi_gpu, s_psi_acc, hs_psi_gpu, g_1psi_gpu, okvan, &
PW/src/c_bands.f90:              IF ( .not. use_gpu ) THEN
PW/src/c_bands.f90:                CALL paro_k_new( h_psi_gpu, s_psi_acc, hs_psi_gpu, g_1psi_gpu, okvan, &
PW/src/c_bands.f90:             IF ( .not. use_gpu ) THEN
PW/src/c_bands.f90:#if defined(__CUDA)
PW/src/c_bands.f90:                CALL rotate_xpsi( h_psi, s_psi, h_psi_gpu, s_psi_acc, npwx, npw, nbnd, nbnd, evc, npol, okvan, &
PW/src/c_bands.f90:          IF ( .not. use_gpu ) THEN
PW/src/c_bands.f90:             CALL crmmdiagg_gpu( h_psi_gpu, s_psi_acc, npwx, npw, nbnd, npol, evc, hevc, sevc, &
PW/src/c_bands.f90:       IF ( .not. use_gpu ) THEN
PW/src/c_bands.f90:          CALL gram_schmidt_k_gpu( npwx, npw, nbnd, npol, evc, hevc, sevc, et(1,ik), &
PW/src/c_bands.f90:       IF ( .not. use_gpu ) THEN
PW/src/c_bands.f90:          IF (use_oscdft) CALL oscdft_h_diag_gpu(oscdft_ctx)
PW/src/c_bands.f90:          CALL usnldiag_gpu( npw, h_diag_d, s_diag_d )
PW/src/c_bands.f90:          IF (.not. use_gpu ) THEN
PW/src/c_bands.f90:                CALL pcegterg_gpu( h_psi_gpu, s_psi_acc, okvan, g_psi_gpu, &
PW/src/c_bands.f90:                CALL cegterg ( h_psi_gpu, s_psi_acc, okvan, g_psi_gpu, &
PW/src/c_bands.f90:  USE control_flags,        ONLY : ethr, restart, isolve, io_level, iverbosity, use_gpu
PW/src/add_paw_to_deeq_gpu.f90:SUBROUTINE add_paw_to_deeq_gpu(deeq_d)
PW/src/add_paw_to_deeq_gpu.f90:#if defined(__CUDA)
PW/src/add_paw_to_deeq_gpu.f90:END SUBROUTINE add_paw_to_deeq_gpu
PW/src/compute_rho.f90:#if !defined(_OPENACC)
PW/src/compute_rho.f90:#if !defined(_OPENACC)
PW/src/compute_rho.f90:#if !defined(_OPENACC)
PW/src/compute_rho.f90:#if !defined(_OPENACC)
PW/src/orthoatwfc.f90:  USE control_flags,    ONLY : gamma_only, use_gpu, offload_type
PW/src/orthoatwfc.f90:     CALL init_us_2 (npw, igk_k(1,ik), xk (1, ik), vkb, use_gpu)
PW/src/orthoatwfc.f90:     IF (save_wfcatom.and..not.use_gpu) THEN
PW/src/orthoatwfc.f90:  USE control_flags,    ONLY : gamma_only, use_gpu, offload_type
PW/src/orthoatwfc.f90:     CALL init_us_2 (npw, igk_k(1,ik), xk (1, ik), vkb, use_gpu)
PW/src/orthoatwfc.f90:  USE control_flags,    ONLY : use_gpu
PW/src/orthoatwfc.f90:  IF(use_gpu) THEN
PW/src/orthoatwfc.f90:    CALL laxlib_cdiaghg_gpu( m, m, overlap, s, m, e, work, me_bgrp, &
PW/src/init_run.f90:                                 lforce => tprnfor, tstress, tqr, use_gpu
PW/src/init_run.f90:  USE dfunct_gpum,        ONLY : newd_gpu
PW/src/init_run.f90:  call allocate_uspp(use_gpu,noncolin,lspinorb,tqr,nhm,nsp,nat,nspin)
PW/src/init_run.f90:  IF ( use_gpu ) THEN
PW/src/init_run.f90:    CALL newd_gpu()
PW/src/realus.f90:      ! and sync on GPUs
PW/src/realus.f90:      !! Sync with GPU memory is performed outside
PW/src/realus.f90:      ! sync to GPUs is performed outside
PW/src/realus.f90:#if defined(_OPENACC)
PW/src/realus.f90:#if defined(_OPENACC)                
PW/src/realus.f90:#if defined(_OPENACC)                
PW/src/realus.f90:#if defined(_OPENACC)
PW/src/realus.f90:#if defined(_OPENACC)
PW/src/wfcinit.f90:  USE control_flags,        ONLY : use_gpu
PW/src/wfcinit.f90:     IF ( nkb > 0 ) CALL init_us_2( ngk(ik), igk_k(1,ik), xk(1,ik), vkb , use_gpu)
PW/src/wfcinit.f90:#if defined(__CUDA)
PW/src/wfcinit.f90:  USE random_numbers_gpum,  ONLY : randy_vect_gpu ! => randy_vect_debug_gpu
PW/src/wfcinit.f90:                                                  ! use '=>randy_vect_debug_gpu'
PW/src/wfcinit.f90:  USE control_flags,        ONLY : lscf, use_gpu
PW/src/wfcinit.f90:#if defined(__CUDA)
PW/src/wfcinit.f90:#if defined(__CUDA)
PW/src/wfcinit.f90:#if defined(__CUDA)
PW/src/wfcinit.f90:         IF(use_gpu) THEN
PW/src/wfcinit.f90:           CALL randy_vect_gpu( randy_vec, 2 * n_starting_atomic_wfc * npol * ngk(ik) )
PW/src/wfcinit.f90:#if defined(__CUDA)
PW/src/wfcinit.f90:#if defined(__CUDA)
PW/src/wfcinit.f90:     CALL randy_vect_gpu( randy_vec , 2 * (n_starting_wfc-n_starting_atomic_wfc) * npol * ngk(ik) )
PW/src/wfcinit.f90:#if defined(__CUDA)
PW/src/wfcinit.f90:#if defined(__CUDA)
PW/src/wfcinit.f90:  IF(use_gpu) DEALLOCATE( randy_vec )
PW/src/wfcinit.f90:  IF(use_gpu) THEN
PW/src/wfcinit.f90:    CALL rotate_wfc_gpu ( npwx, ngk_ik, n_starting_wfc, gstart, nbnd, wfcatom, npol, okvan, evc, etatom )
PW/src/wfcinit.f90:#if defined (__CUDA)
PW/src/wfcinit.f90:#if defined (__CUDA)
PW/src/addusdens.f90:  CALL start_clock_gpu( 'addusdens' )
PW/src/addusdens.f90:        CALL start_clock_gpu( 'addusd:skk' )
PW/src/addusdens.f90:        CALL stop_clock_gpu( 'addusd:skk' )
PW/src/addusdens.f90:  CALL stop_clock_gpu( 'addusdens' )
PW/src/non_scf.f90:  USE control_flags,        ONLY : io_level, conv_elec, lbands, ethr, use_gpu
PW/src/gradcorr.f90:                  gpu_args_=.TRUE. )
PW/src/gradcorr.f90:                  v2c_ud, gpu_args_=.TRUE. )
PW/src/electrons.f90:                                   mbd_vdw, use_gpu
PW/src/electrons.f90:  USE dfunct_gpum,          ONLY : newd_gpu
PW/src/electrons.f90:     IF (use_gpu .and. (iverbosity >= 1) ) CALL dev_buf%print_report(stdout)
PW/src/electrons.f90:     IF (use_gpu .and. (iverbosity >= 1) ) CALL pin_buf%print_report(stdout)
PW/src/electrons.f90:     IF (.not. use_gpu) CALL newd()
PW/src/electrons.f90:     IF (      use_gpu) CALL newd_gpu()
PW/src/electrons.f90:                                 vexxace_gamma_gpu, vexxace_k_gpu
PW/src/electrons.f90:  USE control_flags,      ONLY : gamma_only, use_gpu
PW/src/electrons.f90:        IF (use_gpu) THEN
PW/src/electrons.f90:          CALL vexxace_gamma_gpu( npw, nbnd, evc, ex )
PW/src/electrons.f90:        IF (use_gpu) THEN
PW/src/electrons.f90:          CALL vexxace_k_gpu( npw, nbnd, evc, ex )
README_GPU.md:Quantum ESPRESSO GPU
README_GPU.md:This repository also contains the GPU-accelerated version of Quantum ESPRESSO.
README_GPU.md:NVidia HPC SDK, v.21.7 or later (freely downloadable from NVidia). 
README_GPU.md:You are advised to use the most recent version of NVidia software you can find. 
README_GPU.md:of a few cuda libraries. For this reason the path pointing to the cuda toolkit
README_GPU.md:./configure --with-cuda=XX --with-cuda-runtime=YY --with-cuda-cc=ZZ --enable-openmp [ --with-scalapack=no ][ --with-cuda-mpi=yes ]
README_GPU.md:where `XX` is the location of the CUDA Toolkit (in HPC environments is 
README_GPU.md:typically `$NVHPC_CUDA_HOME` or `$CUDA_HOME`), `YY` is the version of 
README_GPU.md:the cuda toolkit and `ZZ` is the compute capability of the card. You can get 
README_GPU.md:CUDA Driver Version:           11000
README_GPU.md:The version is returned as (1000 major + 10 minor). For example, CUDA 11.0
README_GPU.md:./configure --with-cuda=$CUDA_HOME --with-cuda-cc=70 --with-cuda-runtime=11.0
README_GPU.md:One can also use command `nvidia-smi`: for two GPUs with cc70,
README_GPU.md:$ nvidia-smi --query-gpu=compute_cap --format=csv
README_GPU.md:Enabling faster communications between GPUs, via NVlink or Infiniband RDMA,
README_GPU.md:CUDA-aware, then enable `--with-cuda-mpi=yes` (default: no). 
README_GPU.md:Option --with-openacc is no longer honored: OpenACC is always needed.
README_GPU.md:cases since the serial GPU eigensolver outperforms the parallel CPU
README_GPU.md:From time to time PGI links to the wrong CUDA libraries and fails reporting a 
README_GPU.md:by removing the cuda toolkit from the `LD_LIBRARY_PATH` before compiling.
README_GPU.md:By default, GPU support is active. The following message will appear at
README_GPU.md:     GPU acceleration is ACTIVE.
README_GPU.md:The current GPU version passes all tests with both parallel and serial 
Makefile:libla : $(LAPACK) libutil libcuda
Makefile:libupf : libutil libcuda
Makefile:libcuda: 
Doc/plumed_quick_ref.tex:\texttt{PLUMED}\cite{Bonomi:2009ul} is a plugin for free energy calculation in molecular systems which works together with some of the most popular molecular dynamics engines, including classical (GROMACS, NAMD, DL\_POLY, AMBER and LAMMPS), GPU-accelerated (ACEMD) and ab-initio (\qe) codes.
Doc/release-notes:  * Too small parameter "maxl" in upflib/ylmr2.f90 and upflib/ylmr2_gpu.f90
Doc/release-notes:  * In CPV the mgga contribution to the force was missing on the GPU side.
Doc/release-notes:    Now it has been added (CPU and GPU runs match).
Doc/release-notes:    crashed when used together with NVidia GPUs (thanks to Laura Bellentani)
Doc/release-notes:  * GPU-accelerated phonon code (CINECA Team)
Doc/release-notes:  * First steps towards GPU acceleration for AMD and Intel via OpenMP
Doc/release-notes:  * More CUDA Fortran code removed and replaced by ACC (Ivan Carnimeo)
Doc/release-notes:  * v.7.1 of CP for GPU was not working for pseudopotentials with nonlinear 
Doc/release-notes:  * QEHeat and KCW were not working if compiled for GPU
Doc/release-notes:  * Improved, streamlined and extended porting to NVidia GPUs
Doc/release-notes:  * In CMake GPU builds, one routine fftx_threed2oned_gpu was not compiled with
Doc/release-notes:    a proper GPU compiler option and caused failure in GGA-noncolin calculations.
Doc/release-notes:  * GPU support for PWscf and CP significantly extended
Doc/release-notes:  * RMM-DIIS for CPU (S. Nisihara) and GPU (E. de Paoli, P. Delugas)
Doc/release-notes:  * DFT-D3: MPI parallelization and GPU acceleration with OpenACC
Doc/release-notes:  * Support for GPU via CUDA Fortran brought to the main repository
Doc/release-notes:  * QE-GPU plugin not compatible with 6.x (new version is WIP)
Doc/quote.tex:Users of the GPU-enabled version should also cite the following paper:
Doc/user_guide.tex:repository also works with NVidia GPU's.
Doc/user_guide.tex:at the Exascale EU Centre of Excellence. The GPU porting is mostly the
Doc/user_guide.tex:  GPU porting, CI (Continuous integration), and testing;
Doc/user_guide.tex:\item Filippo Spiga (University of Cambridge, now at NVidia)
Doc/user_guide.tex:  GPU-enabled version;
Doc/user_guide.tex:and libraries are also required. To compile for GPUs you need a
Doc/user_guide.tex:recent version of the NVidia HPC SDK (software development kit),
Doc/user_guide.tex:For GPU compilation, you need v.21.7 or later of the NVidia HPC SDK
Doc/user_guide.tex:In order to compile the code for NVidia GPU's you will need a recent version
Doc/user_guide.tex:(v.21.7 or later) of the NVidia HPC software development kit (SDK). Beware:
Doc/user_guide.tex:\texttt{module}. Specify \texttt{--with-cuda} and \texttt{--with-cuda-cc} as explained 
Doc/user_guide.tex:are optional. Enabling faster communications between GPUs, via NVlink or Infiniband RDMA, 
Doc/user_guide.tex:\texttt{--with-cuda=value}&         enable compilation of GPU-accelerated subroutines.\\
Doc/user_guide.tex:                          &         \texttt{value} should point the path where the CUDA toolkit \\
Doc/user_guide.tex:                          &         is installed, e.g. \texttt{\$NVHPC\_CUDA\_HOME}\\
Doc/user_guide.tex:\texttt{--with-cuda-cc=value}&      sets the compute capabilities for the compilation\\
Doc/user_guide.tex:                             &      NVidia driver installed on the workstation or on the\\
Doc/user_guide.tex:\texttt{--with-cuda-runtime=value}& (optional) sets the version of the CUDA toolkit used \\
Doc/user_guide.tex:                                  & CUDA Toolkit installed on the workstation \\
Doc/user_guide.tex:\texttt{--with-cuda-mpi=value}    & \texttt{yes} enables the usage of CUDA-aware MPI library.\\
Doc/user_guide.tex:                                  & Beware: if you have no fast inter-GPU communications, e.g.,\\
Doc/user_guide.tex:The accelerated version of the code uses standard CUDA libraries 
Doc/user_guide.tex:\texttt{cublas, cufft, curand, cusolver}, available from the NVidia HPC SDK.
Doc/user_guide.tex:Currently, \configure\ supports gfortran and the Intel (ifort), NVidia
README.md:Quick installation instructions for CPU-based machines. For GPU execution, see
README.md:file [README_GPU.md](README_GPU.md). Go to the directory where this file is. 
README.md:- devXlib: low-level utilities for GPU execution
TDDFPT/CMakeLists.txt:qe_enable_cuda_fortran("${src_tddfpt}")
TDDFPT/CMakeLists.txt:        qe_openacc_fortran
TDDFPT/CMakeLists.txt:qe_enable_cuda_fortran("${src_turbo_lanczos_x}")
TDDFPT/CMakeLists.txt:        qe_openacc_fortran
TDDFPT/CMakeLists.txt:qe_enable_cuda_fortran("${src_turbo_davidson_x}")
TDDFPT/CMakeLists.txt:        qe_openacc_fortran
TDDFPT/CMakeLists.txt:qe_enable_cuda_fortran("${src_turbo_eels_x}")
TDDFPT/CMakeLists.txt:        qe_openacc_fortran
TDDFPT/CMakeLists.txt:# Because this main file includes laxlib.fh which is preprocessed with __CUDA definition
TDDFPT/CMakeLists.txt:qe_enable_cuda_fortran(src/lr_magnons_main.f90)
TDDFPT/CMakeLists.txt:        qe_openacc_fortran
TDDFPT/src/lr_eels_main.f90:  USE control_flags,         ONLY : use_para_diag, use_gpu
TDDFPT/src/lr_eels_main.f90:  LOGICAL, EXTERNAL   :: check_gpu_support
TDDFPT/src/lr_eels_main.f90:  use_gpu = check_gpu_support()
TDDFPT/src/lr_main.f90:  USE control_flags,         ONLY : use_gpu
TDDFPT/src/lr_main.f90:  USE control_flags,         ONLY : use_gpu
TDDFPT/src/lr_main.f90:  LOGICAL, EXTERNAL  :: check_gpu_support
TDDFPT/src/lr_main.f90:  use_gpu = check_gpu_support()
TDDFPT/src/lr_magnons_main.f90:  USE control_flags,         ONLY : use_gpu
TDDFPT/src/lr_magnons_main.f90:  LOGICAL, EXTERNAL   :: check_gpu_support
TDDFPT/src/lr_magnons_main.f90:  use_gpu = check_gpu_support()
TDDFPT/src/lr_dav_main.f90:  USE control_flags,         ONLY : do_makov_payne, use_gpu
TDDFPT/src/lr_dav_main.f90:  LOGICAL, EXTERNAL  :: check_gpu_support
TDDFPT/src/lr_dav_main.f90:  use_gpu = check_gpu_support()
TDDFPT/src/lr_apply_liouvillian.f90:#if defined(__CUDA)
TDDFPT/src/lr_apply_liouvillian.f90:  CALL start_clock_gpu('lr_apply')
TDDFPT/src/lr_apply_liouvillian.f90:  IF (interaction)      CALL start_clock_gpu('lr_apply_int')
TDDFPT/src/lr_apply_liouvillian.f90:  IF (.not.interaction) CALL start_clock_gpu('lr_apply_no')
TDDFPT/src/lr_apply_liouvillian.f90:  IF (interaction)      CALL stop_clock_gpu('lr_apply_int')
TDDFPT/src/lr_apply_liouvillian.f90:  IF (.not.interaction) CALL stop_clock_gpu('lr_apply_no')
TDDFPT/src/lr_apply_liouvillian.f90:  CALL stop_clock_gpu('lr_apply')
TDDFPT/src/lr_apply_liouvillian.f90:#if defined(__CUDA)
TDDFPT/src/lr_apply_liouvillian.f90:       CALL start_clock_gpu('interaction')
TDDFPT/src/lr_apply_liouvillian.f90:       CALL stop_clock_gpu('interaction')
TDDFPT/src/lr_apply_liouvillian.f90:#if defined(__CUDA)
TDDFPT/src/lr_apply_liouvillian.f90:    CALL h_psi_gpu (npwx,ngk(1),nbnd,evc1(1,1,1),sevc1_new(1,1,1))
TDDFPT/src/lr_apply_liouvillian.f90:#if defined(__CUDA)
TDDFPT/src/lr_calc_dens_magnons.f90:#if defined(__CUDA)
TDDFPT/src/lr_calc_dens_magnons.f90:  CALL start_clock_gpu ('lr_calc_dens')
TDDFPT/src/lr_calc_dens_magnons.f90:  CALL stop_clock_gpu ('lr_calc_dens')
TDDFPT/src/lr_calc_dens.f90:  CALL start_clock_gpu('lr_calc_dens')
TDDFPT/src/lr_calc_dens.f90: CALL stop_clock_gpu('lr_calc_dens')
TDDFPT/src/lr_init_nfo.f90:  USE control_flags,        ONLY : io_level, use_gpu
TDDFPT/src/lr_init_nfo.f90:#if defined (__CUDA)
TDDFPT/src/lr_init_nfo.f90:  USE cudafor
TDDFPT/src/lr_init_nfo.f90:#if defined(__CUDA)
TDDFPT/src/lr_init_nfo.f90:        IF(use_gpu) istat = cudaHostUnregister(C_LOC(evc(1,1)))
TDDFPT/src/lr_init_nfo.f90:#if defined(__CUDA)
TDDFPT/src/lr_init_nfo.f90:        IF(use_gpu) istat = cudaHostRegister(C_LOC(evc(1,1)), sizeof(evc), cudaHostRegisterMapped)
TDDFPT/src/lr_alloc_init.f90:  USE control_flags,        ONLY : gamma_only, use_gpu
TDDFPT/src/lr_alloc_init.f90:#if defined (__CUDA)
TDDFPT/src/lr_alloc_init.f90:  USE cudafor
TDDFPT/src/lr_alloc_init.f90:#if defined(__CUDA)
TDDFPT/src/lr_alloc_init.f90:        IF(use_gpu) istat = cudaHostUnregister(C_LOC(evc(1,1)))
TDDFPT/src/lr_alloc_init.f90:#if defined(__CUDA)
TDDFPT/src/lr_alloc_init.f90:        IF(use_gpu) istat = cudaHostRegister(C_LOC(evc(1,1)), sizeof(evc), cudaHostRegisterMapped)
include/cpv_device_macros.h:#if defined (_OPENACC)
include/defs.h.README:   Macros allowing transparent usage of either openacc and openmp offloading
include/defs.h.README:*     __CUDA      Compilation for NVidia GPUs
include/defs.h.README:      __GPU_MPI       	MPI via NVlink between GPUs (if your hd supports it)
dev-tools/get_device_props.py:print("\n\n This is a helper tool to check the details of your GPU before configuring QE.\n\n")
dev-tools/get_device_props.py:print("""Remeber to load CUDA environemt and run this on the COMPUTE NODE
dev-tools/get_device_props.py:    print("Compilation with nvcc failed. Did you load CUDA?")
dev-tools/get_device_props.py:    print("./configure --with-cuda=yes --with-cuda-cc={} --with-cuda-runtime={}\n".format(conf_cc, conf_rt))
dev-tools/porting/derived_type_duplicated_module.jf90:   MODULE {{module_name}}_gpum
dev-tools/porting/derived_type_duplicated_module.jf90:#if defined(__CUDA)
dev-tools/porting/derived_type_duplicated_module.jf90:     USE cudafor
dev-tools/porting/derived_type_duplicated_module.jf90:#if defined(__CUDA)
dev-tools/porting/derived_type_duplicated_module.jf90:#if defined(__CUDA) || defined(__CUDA_GNU)
dev-tools/porting/derived_type_duplicated_module.jf90:#if defined(__CUDA) || defined(__CUDA_GNU)
dev-tools/porting/derived_type_duplicated_module.jf90:     SUBROUTINE deallocate_{{module_name}}_gpu
dev-tools/porting/derived_type_duplicated_module.jf90:     END SUBROUTINE deallocate_{{module_name}}_gpu
dev-tools/porting/derived_type_duplicated_module.jf90:   END MODULE {{module_name}}_gpum
dev-tools/porting/regenerate_duplicated_variables.sh:mv gvect_gpu.f90 ../../Modules/recvec_gpu.f90
dev-tools/porting/regenerate_duplicated_variables.sh:mv wavefunctions_gpu.f90 ../../Modules/wavefunctions_gpu.f90
dev-tools/porting/regenerate_duplicated_variables.sh:mv wvfct_gpu.f90 ../../PW/src/pwcom_gpu.f90
dev-tools/porting/regenerate_duplicated_variables.sh:cat us_gpu.f90 >> ../../PW/src/pwcom_gpu.f90
dev-tools/porting/regenerate_duplicated_variables.sh:rm us_gpu.f90
dev-tools/porting/regenerate_duplicated_variables.sh:cat spin_orb_gpu.f90 >> ../../PW/src/pwcom_gpu.f90
dev-tools/porting/regenerate_duplicated_variables.sh:rm spin_orb_gpu.f90
dev-tools/porting/regenerate_duplicated_variables.sh:mv g_psi_mod_gpu.f90 ../../PW/src/g_psi_mod_gpu.f90
dev-tools/porting/regenerate_duplicated_variables.sh:mv scf_gpu.f90 ../../PW/src/scf_mod_gpu.f90
dev-tools/porting/regenerate_duplicated_variables.sh:mv uspp_gpu.f90 ../../upflib/uspp_gpu.f90
dev-tools/porting/regenerate_duplicated_variables.sh:mv becmod_gpu.f90 ../../Modules/becmod_gpu.f90
dev-tools/porting/duplicated_module.jf90:   MODULE {{module_name}}_gpum
dev-tools/porting/duplicated_module.jf90:#if defined(__CUDA)
dev-tools/porting/duplicated_module.jf90:     USE cudafor
dev-tools/porting/duplicated_module.jf90:#if defined(__CUDA)
dev-tools/porting/duplicated_module.jf90:#if defined(__CUDA)  || defined(__CUDA_GNU)
dev-tools/porting/duplicated_module.jf90:#if defined(__CUDA) || defined(__CUDA_GNU)
dev-tools/porting/duplicated_module.jf90:     SUBROUTINE deallocate_{{module_name}}_gpu
dev-tools/porting/duplicated_module.jf90:     END SUBROUTINE deallocate_{{module_name}}_gpu
dev-tools/porting/duplicated_module.jf90:   END MODULE {{module_name}}_gpum
dev-tools/porting/gen_intrinsic.py:with open(sys.argv[1]+'_gpu.f90','w') as f:
dev-tools/porting/gen_derived.py:with open(sys.argv[1]+'_gpu.f90','w') as f:
dev-tools/README.md:- GPU utilities by Pietro Bonfà:
dev-tools/device_props.c: * Returns a sorted list of available CUDA devices with properties colon delimited.
dev-tools/device_props.c:#include <cuda_runtime_api.h>
dev-tools/device_props.c:#define CUDART_LIBRARY_NAME "libcudart.dylib"
dev-tools/device_props.c:#define CUDART_LIBRARY_NAME "libcudart.so"
dev-tools/device_props.c:#error Must define CUDART_LIBRARY_NAME (e.g. libcudart.so or libcudart.dylib)
dev-tools/device_props.c:  DeviceProps_NoCudaRuntime,
dev-tools/device_props.c:  struct cudaDeviceProp props;
dev-tools/device_props.c:}simCudaDevice;
dev-tools/device_props.c:// Function pointer types to dynamically loaded functions from libcudart
dev-tools/device_props.c:typedef cudaError_t (*cudaGetDeviceCount_f)(int *);
dev-tools/device_props.c:typedef cudaError_t (*cudaGetDeviceProperties_f)(struct cudaDeviceProp*, int);
dev-tools/device_props.c:  int cudaRuntimeVersion;
dev-tools/device_props.c:  int cudaDriverVersion;
dev-tools/device_props.c:  if (cudaSuccess != cudaRuntimeGetVersion( &cudaRuntimeVersion )) {
dev-tools/device_props.c:  if (cudaSuccess != cudaDriverGetVersion( &cudaDriverVersion )) {
dev-tools/device_props.c:                          cudaDriverVersion, cudaRuntimeVersion);
dev-tools/device_props.c:  // Cuda Runtime interface
dev-tools/device_props.c:  void *cudaRT = NULL;
dev-tools/device_props.c:  cudaGetDeviceCount_f cudaGetDeviceCount = NULL;
dev-tools/device_props.c:  cudaGetDeviceProperties_f cudaGetDeviceProperties = NULL;
dev-tools/device_props.c:  cudaError_t cuErr;
dev-tools/device_props.c:  int ndevices; // Number of devices reported by Cuda runtime
dev-tools/device_props.c:  simCudaDevice *devices;
dev-tools/device_props.c:  cudaRT = dlopen(CUDART_LIBRARY_NAME, RTLD_NOW);
dev-tools/device_props.c:  if(!cudaRT) {
dev-tools/device_props.c:    sprintf(full_library_name, "/usr/local/cuda/lib64/%s", CUDART_LIBRARY_NAME);
dev-tools/device_props.c:    cudaRT = dlopen(full_library_name, RTLD_NOW);
dev-tools/device_props.c:    if(!cudaRT) {
dev-tools/device_props.c:      sprintf(full_library_name, "/usr/local/cuda/lib/%s", CUDART_LIBRARY_NAME);
dev-tools/device_props.c:      cudaRT = dlopen(full_library_name, RTLD_NOW);
dev-tools/device_props.c:      if(!cudaRT) {
dev-tools/device_props.c:		 "Failed to load CUDA runtime environment from %s.\n"
dev-tools/device_props.c:		 "\tIs the CUDA runtime environment installed in the default location\n"
dev-tools/device_props.c:		 "\tOR is LD_LIBRARY_PATH environment variable set to include CUDA libraries?",
dev-tools/device_props.c:		 CUDART_LIBRARY_NAME);
dev-tools/device_props.c:	return DeviceProps_NoCudaRuntime;
dev-tools/device_props.c:  cudaGetDeviceCount = (cudaGetDeviceCount_f)dlsym(cudaRT, "cudaGetDeviceCount");
dev-tools/device_props.c:  cudaGetDeviceProperties = (cudaGetDeviceProperties_f)dlsym(cudaRT, "cudaGetDeviceProperties");
dev-tools/device_props.c:  if(!cudaGetDeviceCount || !cudaGetDeviceProperties) {
dev-tools/device_props.c:	     "Failed to load CUDA functions from %s.\n"
dev-tools/device_props.c:	     "\tThe CUDA library found is incompatible with simEngine.",
dev-tools/device_props.c:	     CUDART_LIBRARY_NAME);
dev-tools/device_props.c:    return DeviceProps_NoCudaRuntime;
dev-tools/device_props.c:  if (cudaSuccess != cudaGetDeviceCount(&ndevices)) {
dev-tools/device_props.c:	     "\tIs there a CUDA capable GPU available on this computer?");
dev-tools/device_props.c:	     "\tIs your CUDA driver installed, and have you rebooted since installation?");
dev-tools/device_props.c:  devices = (simCudaDevice *)malloc(sizeof(simCudaDevice) * ndevices);
dev-tools/device_props.c:  // Retrieve the properties for all Cuda devices
dev-tools/device_props.c:    if (cudaSuccess != cudaGetDeviceProperties(&devices[deviceid-undevices].props, deviceid)) {
dev-tools/device_props.c:	       "\tThe CUDA library found is incompatible with simEngine.", 
dev-tools/device_props.c:	     "\tDo you have a CUDA device?\n"
dev-tools/device_props.c:	     "\tIs the CUDA driver installed?\n"
dev-tools/device_props.c:	     "\tDo you have device permissions set to allow CUDA computation?");
LAXlib/cdiaghg.f90:SUBROUTINE laxlib_cdiaghg_gpu( n, m, h_d, s_d, ldh, e_d, v_d, me_bgrp, root_bgrp, intra_bgrp_comm)
LAXlib/cdiaghg.f90:  !! GPU VERSION.
LAXlib/cdiaghg.f90:#if defined(__CUDA)
LAXlib/cdiaghg.f90:  USE cudafor
LAXlib/cdiaghg.f90:#if defined(__USE_GLOBAL_BUFFER) && defined(__CUDA)
LAXlib/cdiaghg.f90:  !! matrix to be diagonalized, allocated on the GPU
LAXlib/cdiaghg.f90:  !! overlap matrix, allocated on the GPU
LAXlib/cdiaghg.f90:  !! eigenvalues, , allocated on the GPU
LAXlib/cdiaghg.f90:  !! eigenvectors (column-wise), , allocated on the GPU
LAXlib/cdiaghg.f90:#if defined(__CUDA)
LAXlib/cdiaghg.f90:#if (! defined(__USE_GLOBAL_BUFFER)) && defined(__CUDA)
LAXlib/cdiaghg.f90:#if defined(__CUDA)
LAXlib/cdiaghg.f90:  CALL start_clock_gpu( 'cdiaghg' )
LAXlib/cdiaghg.f90:#if defined(__CUDA)
LAXlib/cdiaghg.f90:      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate h_bkp_d or s_bkp_d ', ABS( info ) )
LAXlib/cdiaghg.f90:      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate h_bkp_d ', ABS( info ) )
LAXlib/cdiaghg.f90:      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate s_bkp_d ', ABS( info ) )
LAXlib/cdiaghg.f90:      IF (omp_get_num_threads() > 1) CALL lax_error__( ' cdiaghg_gpu ', 'cdiaghg_gpu is not thread-safe',  ABS( info ) )
LAXlib/cdiaghg.f90:         IF ( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', 'cusolverDnCreate',  ABS( info ) )
LAXlib/cdiaghg.f90:      IF( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', ' cusolverDnZhegvdx_bufferSize failed ', ABS( info ) )
LAXlib/cdiaghg.f90:      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate work_d ', ABS( info ) )
LAXlib/cdiaghg.f90:      IF( info /= 0 ) CALL lax_error__( ' cdiaghg_gpu ', ' cannot allocate work_d ', ABS( info ) )
LAXlib/cdiaghg.f90:      IF( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', ' cusolverDnZhegvdx failed ', ABS( info ) )
LAXlib/cdiaghg.f90:      !IF( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' cdiaghg_gpu ', ' cusolverDnDestroy failed ', ABS( info ) )
LAXlib/cdiaghg.f90:     CALL lax_error__( 'cdiaghg', 'Called GPU eigensolver without GPU support', 1 )
LAXlib/cdiaghg.f90:#if defined __GPU_MPI
LAXlib/cdiaghg.f90:  info = cudaDeviceSynchronize()
LAXlib/cdiaghg.f90:  info = cudaDeviceSynchronize() ! this is probably redundant...
LAXlib/cdiaghg.f90:  CALL stop_clock_gpu( 'cdiaghg' )
LAXlib/cdiaghg.f90:END SUBROUTINE laxlib_cdiaghg_gpu
LAXlib/tests/test_diaghg_gpu_2.f90:#if defined(__CUDA)
LAXlib/tests/test_diaghg_gpu_2.f90:program test_diaghg_gpu_2
LAXlib/tests/test_diaghg_gpu_2.f90:    USE cudafor
LAXlib/tests/test_diaghg_gpu_2.f90:    ! Solve-again, with the same algorithm used in the GPU version.
LAXlib/tests/test_diaghg_gpu_2.f90:end program test_diaghg_gpu_2
LAXlib/tests/test_diaghg_gpu_2.f90:program test_diaghg_gpu_2
LAXlib/tests/test_diaghg_gpu_2.f90:end program test_diaghg_gpu_2
LAXlib/tests/test_diaghg_gpu_4.f90:! (GPU interface) to solve the problems stored in binary files:
LAXlib/tests/test_diaghg_gpu_4.f90:#if ( ! defined(__SCALAPACK) ) && defined(__CUDA) 
LAXlib/tests/test_diaghg_gpu_4.f90:program test_diaghg_gpu_4
LAXlib/tests/test_diaghg_gpu_4.f90:    USE cudafor
LAXlib/tests/test_diaghg_gpu_4.f90:    USE cudafor
LAXlib/tests/test_diaghg_gpu_4.f90:end program test_diaghg_gpu_4
LAXlib/tests/test_diaghg_gpu_4.f90:program test_diaghg_gpu_4
LAXlib/tests/test_diaghg_gpu_4.f90:end program test_diaghg_gpu_4
LAXlib/tests/Makefile:       test_diaghg_gpu_1.f90 \
LAXlib/tests/Makefile:       test_diaghg_gpu_2.f90 \
LAXlib/tests/Makefile:       test_diaghg_gpu_3.f90 \
LAXlib/tests/Makefile:       test_diaghg_gpu_4.f90
LAXlib/tests/test_diaghg_gpu_3.f90:#ifdef __CUDA
LAXlib/tests/test_diaghg_gpu_3.f90:program test_diaghg_gpu_3
LAXlib/tests/test_diaghg_gpu_3.f90:        ! GPU data & subroutines
LAXlib/tests/test_diaghg_gpu_3.f90:        ! Start from data on the GPU and diagonalize on the CPU
LAXlib/tests/test_diaghg_gpu_3.f90:        ! N.B.: GPU eigensolver uses a different algorithm: zhegvd
LAXlib/tests/test_diaghg_gpu_3.f90:        ! GPU data & subroutines
LAXlib/tests/test_diaghg_gpu_3.f90:end program test_diaghg_gpu_3
LAXlib/tests/test_diaghg_gpu_3.f90:program test_diaghg_gpu_3
LAXlib/tests/test_diaghg_gpu_3.f90:end program test_diaghg_gpu_3
LAXlib/tests/test_diaghg_gpu_1.f90:#if defined(__CUDA)
LAXlib/tests/test_diaghg_gpu_1.f90:program test_diaghg_gpu
LAXlib/tests/test_diaghg_gpu_1.f90:    USE cudafor
LAXlib/tests/test_diaghg_gpu_1.f90:    USE cudafor
LAXlib/tests/test_diaghg_gpu_1.f90:end program test_diaghg_gpu
LAXlib/tests/test_diaghg_gpu_1.f90:program test_diaghg_gpu
LAXlib/tests/test_diaghg_gpu_1.f90:end program test_diaghg_gpu
LAXlib/Makefile:EXTLIBS=$(CUDA_LIBS) $(SCALAPACK_LIBS) $(LAPACK_LIBS) $(BLAS_LIBS) $(MPI_LIBS)
LAXlib/laxlib_hi.h:#ifdef __CUDA
LAXlib/laxlib_hi.h:SUBROUTINE laxlib_rdiaghg_gpu( n, m, h, s, ldh, e, v, me_bgrp, root_bgrp, intra_bgrp_comm )
LAXlib/laxlib_hi.h:#ifdef __CUDA
LAXlib/laxlib_hi.h:SUBROUTINE laxlib_cdiaghg_gpu( n, m, h, s, ldh, e, v, me_bgrp, root_bgrp, intra_bgrp_comm )
LAXlib/laxlib_hi.h:#ifdef __CUDA
LAXlib/laxlib_hi.h:      SUBROUTINE diagonalize_serial_gpu( m, rhos, rhod, s, info )
LAXlib/la_helper.f90:#if defined(__SCALAPACK) && !defined(__CUDA)
LAXlib/la_helper.f90:   SUBROUTINE diagonalize_serial_gpu( m, rhos, rhod, s, info )
LAXlib/la_helper.f90:#if defined(__CUDA)
LAXlib/la_helper.f90:      use cudafor
LAXlib/la_helper.f90:         CALL lax_error__( ' diagonalize_serial_gpu ', 'cusolverDnCreate',  ABS( info ) )
LAXlib/la_helper.f90:      IF( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' laxlib diagonalize_serial_gpu ', ' error in solver 1 ', ABS( info ) )
LAXlib/la_helper.f90:      IF( info /= 0 ) CALL lax_error__( ' laxlib diagonalize_serial_gpu ', ' allocate work_d ', ABS( info ) )
LAXlib/la_helper.f90:      IF( info /= 0 ) CALL lax_error__( ' laxlib diagonalize_serial_gpu ', ' error in solver 2 ', ABS( info ) )
LAXlib/la_helper.f90:      info = cudaDeviceSynchronize()
LAXlib/la_helper.f90:      IF( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' diagonalize_serial_gpu ', ' cusolverDnDestroy failed ', ABS( info ) )
LAXlib/la_helper.f90:      CALL lax_error__( ' laxlib diagonalize_serial_gpu ', ' not compiled in this version ', 0 )
LAXlib/CMakeLists.txt:qe_enable_cuda_fortran("${src_lax}")
LAXlib/CMakeLists.txt:if(QE_ENABLE_CUDA)
LAXlib/CMakeLists.txt:        set(CMAKE_REQUIRED_LINK_OPTIONS "${CUDA_FLAG}lib=cusolver;-fortranlibs")
LAXlib/CMakeLists.txt:            message(FATAL_ERROR "The version of CUDAToolkit chosen by the PGI/NVHPC compiler internally"
LAXlib/CMakeLists.txt:                                " only supported since CUDAToolkit 10.1 release. Use a newer compiler"
LAXlib/CMakeLists.txt:                                " or select a newer CUDAToolkit internal to the PGI/NVHPC compiler.")
LAXlib/CMakeLists.txt:        if(CUDAToolkit_VERSION VERSION_LESS 10.1)
LAXlib/CMakeLists.txt:            message(FATAL_ERROR "cuSOLVER for LAXLib is only supported from CUDA compiler 10.1")
LAXlib/CMakeLists.txt:            CUDA::cusolver
LAXlib/CMakeLists.txt:            CUDA::cublas)
LAXlib/CMakeLists.txt:    qe_enable_cuda_fortran("${src_lax_test}")
LAXlib/laxlib_low.h:#if defined (__CUDA)
LAXlib/laxlib_low.h:SUBROUTINE laxlib_dsqmsym_gpu_x( n, a, lda, idesc )
LAXlib/laxlib_low.h:#if defined (__CUDA)
LAXlib/laxlib_low.h:SUBROUTINE sqr_dmm_cannon_gpu_x( transa, transb, n, alpha, a, lda, b, ldb, beta, c, ldc, idesc )
LAXlib/laxlib_low.h:#if defined (__CUDA)
LAXlib/laxlib_low.h:SUBROUTINE sqr_tr_cannon_gpu_x( n, a, lda, b, ldb, idesc )
LAXlib/laxlib_low.h:#if defined (__CUDA)
LAXlib/laxlib_low.h:SUBROUTINE redist_row2col_gpu_x( n, a, b, ldx, nx, idesc )
LAXlib/ptoolkit.f90:#if defined (__CUDA)
LAXlib/ptoolkit.f90:SUBROUTINE laxlib_dsqmsym_gpu_x( n, a, lda, idesc )
LAXlib/ptoolkit.f90:   !! GPU version
LAXlib/ptoolkit.f90:   USE cudafor
LAXlib/ptoolkit.f90:#if ! defined(__GPU_MPI)
LAXlib/ptoolkit.f90:#if defined(__GPU_MPI)
LAXlib/ptoolkit.f90:      ierr = cudaDeviceSynchronize()
LAXlib/ptoolkit.f90:#if defined(__GPU_MPI)
LAXlib/ptoolkit.f90:      ierr = cudaDeviceSynchronize()
LAXlib/ptoolkit.f90:#if defined(__GPU_MPI)
LAXlib/ptoolkit.f90:END SUBROUTINE laxlib_dsqmsym_gpu_x
LAXlib/ptoolkit.f90:#if defined (__CUDA)
LAXlib/ptoolkit.f90:SUBROUTINE sqr_dmm_cannon_gpu_x( transa, transb, n, alpha, a, lda, b, ldb, beta, c, ldc, idesc )
LAXlib/ptoolkit.f90:   !!  GPU version
LAXlib/ptoolkit.f90:   USE cudafor
LAXlib/ptoolkit.f90:   ierr = cudaDeviceSynchronize()
LAXlib/ptoolkit.f90:   ierr = cudaDeviceSynchronize()
LAXlib/ptoolkit.f90:      ierr = cudaDeviceSynchronize()
LAXlib/ptoolkit.f90:   ierr = cudaDeviceSynchronize()
LAXlib/ptoolkit.f90:#if ! defined(__GPU_MPI)
LAXlib/ptoolkit.f90:      ierr = cudaDeviceSynchronize()
LAXlib/ptoolkit.f90:#if defined(__GPU_MPI)
LAXlib/ptoolkit.f90:      ierr = cudaDeviceSynchronize()
LAXlib/ptoolkit.f90:         CALL lax_error__( " sqr_mm_cannon_gpu ", " in MPI_SENDRECV_REPLACE ", ABS( ierr ) )
LAXlib/ptoolkit.f90:         CALL lax_error__( " sqr_mm_cannon_gpu ", " in MPI_SENDRECV_REPLACE ", ABS( ierr ) )
LAXlib/ptoolkit.f90:      ierr = cudaDeviceSynchronize()
LAXlib/ptoolkit.f90:#if ! defined(__GPU_MPI)
LAXlib/ptoolkit.f90:      ierr = cudaDeviceSynchronize()
LAXlib/ptoolkit.f90:         CALL lax_error__( ' sqr_mm_cannon_gpu ', ' unknown shift_exch direction ', 1 )
LAXlib/ptoolkit.f90:#if defined(__GPU_MPI)
LAXlib/ptoolkit.f90:      ierr = cudaDeviceSynchronize()
LAXlib/ptoolkit.f90:         CALL lax_error__( " sqr_mm_cannon_gpu ", " in MPI_SENDRECV_REPLACE 2 ", ABS( ierr ) )
LAXlib/ptoolkit.f90:         CALL lax_error__( " sqr_mm_cannon_gpu ", " in MPI_SENDRECV_REPLACE 2 ", ABS( ierr ) )
LAXlib/ptoolkit.f90:      ierr = cudaDeviceSynchronize()
LAXlib/ptoolkit.f90:END SUBROUTINE sqr_dmm_cannon_gpu_x
LAXlib/ptoolkit.f90:#if defined (__CUDA)
LAXlib/ptoolkit.f90:SUBROUTINE sqr_tr_cannon_gpu_x( n, a, lda, b, ldb, idesc )
LAXlib/ptoolkit.f90:   !!  GPU version
LAXlib/ptoolkit.f90:   USE cudafor
LAXlib/ptoolkit.f90:#if defined (__GPU_MPI)
LAXlib/ptoolkit.f90:      CALL lax_error__( ' sqr_tr_cannon_gpu ', ' works only with square processor mesh ', 1 )
LAXlib/ptoolkit.f90:      CALL lax_error__( ' sqr_tr_cannon_gpu ', ' inconsistent size n  ', 1 )
LAXlib/ptoolkit.f90:      CALL lax_error__( ' sqr_tr_cannon_gpu ', ' inconsistent size lda  ', 1 )
LAXlib/ptoolkit.f90:      CALL lax_error__( ' sqr_tr_cannon_gpu ', ' inconsistent size ldb  ', 1 )
LAXlib/ptoolkit.f90:      CALL lax_error__( " sqr_tr_cannon_gpu ", " in MPI_BARRIER ", ABS( ierr ) )
LAXlib/ptoolkit.f90:#if defined (__GPU_MPI)
LAXlib/ptoolkit.f90:   ierr = cudaDeviceSynchronize()
LAXlib/ptoolkit.f90:   !ierr = cudaMemcpy2D(b, SIZE(b,1), ablk, ldx, nc, nr, cudaMemcpyHostToDevice )
LAXlib/ptoolkit.f90:   ierr = cudaDeviceSynchronize()
LAXlib/ptoolkit.f90:#if defined (__GPU_MPI)
LAXlib/ptoolkit.f90:#if defined (__GPU_MPI)
LAXlib/ptoolkit.f90:      ierr = cudaDeviceSynchronize()
LAXlib/ptoolkit.f90:         CALL lax_error__( " sqr_tr_cannon_gpu ", " in MPI_SENDRECV_REPLACE ", ABS( ierr ) )
LAXlib/ptoolkit.f90:END SUBROUTINE sqr_tr_cannon_gpu_x
LAXlib/ptoolkit.f90:#if defined (__CUDA)
LAXlib/ptoolkit.f90:SUBROUTINE redist_row2col_gpu_x( n, a, b, ldx, nx, idesc )
LAXlib/ptoolkit.f90:   !!  GPU version
LAXlib/ptoolkit.f90:   USE cudafor
LAXlib/ptoolkit.f90:      CALL lax_error__( ' redist_row2col_gpu ', ' works only with square processor mesh ', 1 )
LAXlib/ptoolkit.f90:      CALL lax_error__( ' redist_row2col_gpu ', ' inconsistent size n  ', 1 )
LAXlib/ptoolkit.f90:      CALL lax_error__( ' redist_row2col_gpu ', ' inconsistent size lda  ', 1 )
LAXlib/ptoolkit.f90:      CALL lax_error__( " redist_row2col_gpu ", " in MPI_BARRIER ", ABS( ierr ) )
LAXlib/ptoolkit.f90:#if defined(__GPU_MPI)
LAXlib/ptoolkit.f90:      CALL lax_error__( " redist_row2col_gpu ", " allocating wrk ", ABS( ierr ) )
LAXlib/ptoolkit.f90:   ierr = cudaDeviceSynchronize()
LAXlib/ptoolkit.f90:      CALL lax_error__( " redist_row2col_gpu ", " in MPI_SENDRECV ", ABS( ierr ) )
LAXlib/ptoolkit.f90:      CALL lax_error__( " redist_row2col_gpu ", " allocating a_h ", ABS( ierr ) )
LAXlib/ptoolkit.f90:      CALL lax_error__( " redist_row2col_gpu ", " allocating b_h ", ABS( ierr ) )
LAXlib/ptoolkit.f90:      CALL lax_error__( " redist_row2col_gpu ", " in MPI_SENDRECV ", ABS( ierr ) )
LAXlib/ptoolkit.f90:END SUBROUTINE redist_row2col_gpu_x
LAXlib/la_module.f90:#ifdef __CUDA
LAXlib/la_module.f90:  USE cudafor
LAXlib/la_module.f90:#ifdef __CUDA
LAXlib/la_module.f90:     MODULE PROCEDURE cdiaghg_gpu_, rdiaghg_gpu_
LAXlib/la_module.f90:#ifdef __CUDA
LAXlib/la_module.f90:     MODULE PROCEDURE pcdiaghg__gpu, prdiaghg__gpu
LAXlib/la_module.f90:#if defined (__CUDA)
LAXlib/la_module.f90:    USE cudafor
LAXlib/la_module.f90:    !! optionally solve the eigenvalue problem on the GPU
LAXlib/la_module.f90:#if defined(__CUDA)
LAXlib/la_module.f90:    ! the following ifdef ensures no offload if not compiling from GPU 
LAXlib/la_module.f90:#if defined(__CUDA)
LAXlib/la_module.f90:    ! ... always false when compiling without CUDA support
LAXlib/la_module.f90:#if defined(__CUDA)
LAXlib/la_module.f90:      CALL laxlib_cdiaghg_gpu(n, m, h_d, s_d, ldh, e_d, v_d, me_bgrp, root_bgrp, intra_bgrp_comm)
LAXlib/la_module.f90:#if defined(__CUDA)
LAXlib/la_module.f90:  SUBROUTINE cdiaghg_gpu_( n, m, h_d, s_d, ldh, e_d, v_d, me_bgrp, root_bgrp, intra_bgrp_comm, onhost )
LAXlib/la_module.f90:    !! GPU version
LAXlib/la_module.f90:    USE cudafor
LAXlib/la_module.f90:      CALL laxlib_cdiaghg_gpu(n, m, h_d, s_d, ldh, e_d, v_d, me_bgrp, root_bgrp, intra_bgrp_comm)
LAXlib/la_module.f90:  END SUBROUTINE cdiaghg_gpu_
LAXlib/la_module.f90:#if defined(__CUDA)
LAXlib/la_module.f90:    USE cudafor
LAXlib/la_module.f90:    !! optionally solve the eigenvalue problem on the GPU   
LAXlib/la_module.f90:#if defined(__CUDA)
LAXlib/la_module.f90:    ! the following ifdef ensures no offload if not compiling from GPU 
LAXlib/la_module.f90:#if defined(__CUDA)
LAXlib/la_module.f90:    ! ... always false when compiling without CUDA support
LAXlib/la_module.f90:#if defined(__CUDA)
LAXlib/la_module.f90:      CALL laxlib_rdiaghg_gpu(n, m, h_d, s_d, ldh, e_d, v_d, me_bgrp, root_bgrp, intra_bgrp_comm)
LAXlib/la_module.f90:#if defined(__CUDA)
LAXlib/la_module.f90:  SUBROUTINE rdiaghg_gpu_( n, m, h_d, s_d, ldh, e_d, v_d, me_bgrp, root_bgrp, intra_bgrp_comm, onhost )
LAXlib/la_module.f90:    !! GPU version 
LAXlib/la_module.f90:    USE cudafor
LAXlib/la_module.f90:      CALL laxlib_rdiaghg_gpu(n, m, h_d, s_d, ldh, e_d, v_d, me_bgrp, root_bgrp, intra_bgrp_comm)
LAXlib/la_module.f90:  END SUBROUTINE rdiaghg_gpu_
LAXlib/la_module.f90:    !! place-holder, offloading on GPU not implemented yet
LAXlib/la_module.f90:    !! place-holder, offloading on GPU not implemented yet
LAXlib/la_module.f90:#if defined(__CUDA)
LAXlib/la_module.f90:  SUBROUTINE prdiaghg__gpu( n, h_d, s_d, ldh, e_d, v_d, idesc, onhost )
LAXlib/la_module.f90:    !! Parallel GPU version with full data distribution
LAXlib/la_module.f90:    !! place-holder, prdiaghg on GPU not implemented yet
LAXlib/la_module.f90:  SUBROUTINE pcdiaghg__gpu( n, h_d, s_d, ldh, e_d, v_d, idesc, onhost )
LAXlib/la_module.f90:    !! Parallel GPU version with full data distribution
LAXlib/la_module.f90:    !! place-holder, pcdiaghg on GPU not implemented yet
LAXlib/rdiaghg.f90:SUBROUTINE laxlib_rdiaghg_gpu( n, m, h_d, s_d, ldh, e_d, v_d, me_bgrp, root_bgrp, intra_bgrp_comm )
LAXlib/rdiaghg.f90:  !! GPU VERSION.
LAXlib/rdiaghg.f90:#if defined(__CUDA)
LAXlib/rdiaghg.f90:  USE cudafor
LAXlib/rdiaghg.f90:#if defined(__USE_GLOBAL_BUFFER) && defined(__CUDA)
LAXlib/rdiaghg.f90:#if defined(__CUDA)
LAXlib/rdiaghg.f90:#if defined(__CUDA)
LAXlib/rdiaghg.f90:#if defined(__CUDA)
LAXlib/rdiaghg.f90:#if defined(__CUDA)
LAXlib/rdiaghg.f90:#if defined(__CUDA)
LAXlib/rdiaghg.f90:  CALL start_clock_gpu( 'rdiaghg' )
LAXlib/rdiaghg.f90:#if defined(__CUDA)
LAXlib/rdiaghg.f90:      IF( info /= 0 ) CALL lax_error__( ' rdiaghg_gpu ', ' cannot allocate h_bkp_d or s_bkp_d ', ABS( info ) )
LAXlib/rdiaghg.f90:      IF( info /= 0 ) CALL lax_error__( ' rdiaghg_gpu ', ' cannot allocate h_bkp_d ', ABS( info ) )
LAXlib/rdiaghg.f90:      IF( info /= 0 ) CALL lax_error__( ' rdiaghg_gpu ', ' cannot allocate s_bkp_d ', ABS( info ) )
LAXlib/rdiaghg.f90:      IF (omp_get_num_threads() > 1) CALL lax_error__( ' rdiaghg_gpu ', 'rdiaghg_gpu is not thread-safe',  ABS( info ) )
LAXlib/rdiaghg.f90:         IF( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' rdiaghg_gpu ', ' cusolverDnCreate failed ', ABS( info ) )
LAXlib/rdiaghg.f90:      IF( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' rdiaghg_gpu ', ' cusolverDnDsygvdx_bufferSize failed ', ABS( info ) )
LAXlib/rdiaghg.f90:      IF( info /= 0 ) CALL lax_error__( ' rdiaghg_gpu ', ' cannot allocate work_d ', ABS( info ) )
LAXlib/rdiaghg.f90:      IF( info /= 0 ) CALL lax_error__( ' rdiaghg_gpu ', ' allocate work_d ', ABS( info ) )
LAXlib/rdiaghg.f90:    IF( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' rdiaghg_gpu ', ' cusolverDnDsygvdx failed ', ABS( info ) )
LAXlib/rdiaghg.f90:      ! IF( info /= CUSOLVER_STATUS_SUCCESS ) CALL lax_error__( ' rdiaghg_gpu ', ' cusolverDnDestroy failed ', ABS( info ) )
LAXlib/rdiaghg.f90:     CALL lax_error__( 'cdiaghg', 'Called GPU eigensolver without GPU support', 1 )
LAXlib/rdiaghg.f90:#if defined __GPU_MPI
LAXlib/rdiaghg.f90:  info = cudaDeviceSynchronize()
LAXlib/rdiaghg.f90:  info = cudaDeviceSynchronize() ! this is probably redundant...
LAXlib/rdiaghg.f90:  CALL stop_clock_gpu( 'rdiaghg' )
LAXlib/rdiaghg.f90:END SUBROUTINE laxlib_rdiaghg_gpu
CMakeLists.txt:option(QE_ENABLE_CUDA
CMakeLists.txt:    "enable CUDA acceleration on NVIDIA GPUs" OFF)
CMakeLists.txt:if(QE_ENABLE_CUDA)
CMakeLists.txt:    option(QE_ENABLE_OPENACC "enable OpenACC acceleration" ON)
CMakeLists.txt:    # OpenMP enabled by default if CUDA is enable
CMakeLists.txt:    option(QE_ENABLE_OPENACC "enable OpenACC acceleration" OFF)
CMakeLists.txt:option(QE_ENABLE_MPI_GPU_AWARE
CMakeLists.txt:    "enable GPU aware MPI operations" OFF)
CMakeLists.txt:        "enable execution of NVIDIA NVTX profiler plugin" OFF)
CMakeLists.txt:if(QE_ENABLE_CUDA)
CMakeLists.txt:    qe_add_global_compile_definitions(__CUDA)
CMakeLists.txt:    if(QE_ENABLE_MPI_GPU_AWARE)
CMakeLists.txt:        qe_add_global_compile_definitions(__GPU_MPI)
CMakeLists.txt:if(QE_ENABLE_CUDA AND NOT (CMAKE_Fortran_COMPILER_ID MATCHES "PGI" OR CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC"))
CMakeLists.txt:    message(FATAL_ERROR "NVHPC compiler is mandatory when CUDA is enabled due QE is based on CUDA Fortran language")
CMakeLists.txt:if(QE_ENABLE_OPENACC AND NOT (CMAKE_Fortran_COMPILER_ID MATCHES "PGI" OR CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC"))
CMakeLists.txt:    message(FATAL_ERROR "NVHPC compiler is mandatory when OpenACC is enabled")
CMakeLists.txt:if(QE_ENABLE_MPI_GPU_AWARE AND NOT (QE_ENABLE_CUDA AND QE_ENABLE_MPI))
CMakeLists.txt:    message(FATAL_ERROR "GPU aware MPI requires both MPI and CUDA features enabled")
CMakeLists.txt:    target_compile_definitions(qe_openmp_fortran INTERFACE "$<$<COMPILE_LANGUAGE:Fortran>:__OPENMP_GPU>")
CMakeLists.txt:# OpenACC
CMakeLists.txt:add_library(qe_openacc_fortran INTERFACE)
CMakeLists.txt:add_library(qe_openacc_c INTERFACE)
CMakeLists.txt:qe_install_targets(qe_openacc_fortran qe_openacc_c)
CMakeLists.txt:if(QE_ENABLE_OPENACC)
CMakeLists.txt:    find_package(OpenACC REQUIRED Fortran C)
CMakeLists.txt:    target_link_libraries(qe_openacc_fortran INTERFACE OpenACC::OpenACC_Fortran)
CMakeLists.txt:    target_link_libraries(qe_openacc_c INTERFACE OpenACC::OpenACC_C)
CMakeLists.txt:endif(QE_ENABLE_OPENACC)
CMakeLists.txt:# CUDA
CMakeLists.txt:if(QE_ENABLE_CUDA OR QE_ENABLE_PROFILE_NVTX)
CMakeLists.txt:        add_library(CUDA::cufft INTERFACE IMPORTED)
CMakeLists.txt:        set_target_properties(CUDA::cufft PROPERTIES INTERFACE_LINK_LIBRARIES "${CUDA_FLAG}lib=cufft")
CMakeLists.txt:        add_library(CUDA::cublas INTERFACE IMPORTED)
CMakeLists.txt:        set_target_properties(CUDA::cublas PROPERTIES INTERFACE_LINK_LIBRARIES "${CUDA_FLAG}lib=cublas")
CMakeLists.txt:        add_library(CUDA::cusolver INTERFACE IMPORTED)
CMakeLists.txt:        set_target_properties(CUDA::cusolver PROPERTIES INTERFACE_LINK_LIBRARIES "${CUDA_FLAG}lib=cusolver")
CMakeLists.txt:        add_library(CUDA::curand INTERFACE IMPORTED)
CMakeLists.txt:        set_target_properties(CUDA::curand PROPERTIES INTERFACE_LINK_LIBRARIES "${CUDA_FLAG}lib=curand")
CMakeLists.txt:            add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
CMakeLists.txt:            set_target_properties(CUDA::nvToolsExt PROPERTIES INTERFACE_LINK_LIBRARIES "-cuda;libnvToolsExt.so")
CMakeLists.txt:            set(CMAKE_REQUIRED_LIBRARIES "-cuda;libnvToolsExt.so")
CMakeLists.txt:        find_package(CUDAToolkit REQUIRED)
CMakeLists.txt:endif(QE_ENABLE_CUDA OR QE_ENABLE_PROFILE_NVTX)
CMakeLists.txt:        CUDA::nvToolsExt)
LR_Modules/ccgsolve_all.f90:#if defined(__CUDA)
LR_Modules/ccgsolve_all.f90:#if defined(__CUDA)
LR_Modules/setup_dmuxc.f90:     CALL dmxc( dfftp_nnr, 2, rho_aux, dmuxc, gpu_args_=.TRUE. )
LR_Modules/setup_dmuxc.f90:        CALL dmxc( dfftp_nnr, 4, rho_aux, dmuxc, gpu_args_=.TRUE. )
LR_Modules/setup_dmuxc.f90:        CALL dmxc( dfftp_nnr, 1, rho_aux, dmuxc, gpu_args_=.TRUE. )
LR_Modules/ch_psi_all_complex.f90:#if defined(__CUDA)
LR_Modules/ch_psi_all_complex.f90:  CALL h_psi_gpu (npwx, n, m, h, hpsi)
LR_Modules/ch_psi_all_complex.f90:#if defined(__CUDA)
LR_Modules/ch_psi_all_complex.f90:    CALL start_clock_gpu ('ch_psi_all_k_complex')
LR_Modules/ch_psi_all_complex.f90:    CALL start_clock_gpu ('ch_psi_calbec')
LR_Modules/ch_psi_all_complex.f90:    CALL stop_clock_gpu ('ch_psi_calbec')
LR_Modules/ch_psi_all_complex.f90:    CALL stop_clock_gpu ('ch_psi_all_k_complex')
LR_Modules/compute_vsgga.f90:  CALL xc_gcx( dfftp%nnr, 2, rhoout, grho, sx, sc, v1x, v2x, v1c, v2c, v2cud, gpu_args_=.TRUE. )
LR_Modules/localdos.f90:#if defined(__CUDA)
LR_Modules/lr_dot.f90:#if defined(__CUDA)
LR_Modules/lr_dot.f90:    REAL(DP), EXTERNAL :: MYDDOT_VECTOR_GPU
LR_Modules/lr_dot.f90:    !$acc routine(MYDDOT_VECTOR_GPU) vector
LR_Modules/lr_dot.f90:#if defined(__CUDA)
LR_Modules/lr_dot.f90:       temp_gamma = temp_gamma + 2.D0*wg(ibnd,1)*MYDDOT_VECTOR_GPU(2*ngk(1),x(:,ibnd,1),y(:,ibnd,1))
LR_Modules/cgsolve_all.f90:#if defined(__CUDA)
LR_Modules/cgsolve_all.f90:#if defined(__CUDA)
LR_Modules/lr_two_chem.f90:  CALL start_clock_gpu ('incdrhoscf_cond')
LR_Modules/lr_two_chem.f90:  CALL stop_clock_gpu ('incdrhoscf_cond')
LR_Modules/lr_two_chem.f90:  CALL start_clock_gpu ('incdrhoscf_cond')
LR_Modules/lr_two_chem.f90:#if defined(__CUDA)
LR_Modules/lr_two_chem.f90:        CALL errore( ' incdrhoscf_cond_nc ', ' taskgroup par not implement with GPU offload', 1 )
LR_Modules/lr_two_chem.f90:  CALL stop_clock_gpu ('incdrhoscf_cond')
LR_Modules/CMakeLists.txt:qe_enable_cuda_fortran("${sources}")
LR_Modules/CMakeLists.txt:        qe_openacc_fortran
LR_Modules/CMakeLists.txt:if(QE_ENABLE_CUDA)
LR_Modules/CMakeLists.txt:            CUDA::cublas)
LR_Modules/orthogonalize.f90:#if defined(__CUDA)
LR_Modules/incdrhoscf.f90:#if defined(__CUDA)
LR_Modules/incdrhoscf.f90:  CALL start_clock_gpu ('incdrhoscf')
LR_Modules/incdrhoscf.f90:  CALL stop_clock_gpu ('incdrhoscf')
LR_Modules/lr_sm1_psi.f90:#if defined(__CUDA)
LR_Modules/ch_psi_all.f90:#if defined(__CUDA)
LR_Modules/ch_psi_all.f90:  CALL h_psi_gpu (npwx, n, m, h, hpsi)
LR_Modules/ch_psi_all.f90:#if defined(__CUDA)
LR_Modules/ch_psi_all.f90:    CALL start_clock_gpu ('ch_psi_all_k')
LR_Modules/ch_psi_all.f90:       CALL start_clock_gpu ('ch_psi_calbec')
LR_Modules/ch_psi_all.f90:       CALL stop_clock_gpu ('ch_psi_calbec')
LR_Modules/ch_psi_all.f90:    CALL stop_clock_gpu ('ch_psi_all_k')
LR_Modules/ch_psi_all.f90:    CALL start_clock_gpu ('ch_psi_all_gamma')
LR_Modules/ch_psi_all.f90:          CALL start_clock_gpu ('ch_psi_calbec')
LR_Modules/ch_psi_all.f90:          CALL stop_clock_gpu ('ch_psi_calbec')
LR_Modules/ch_psi_all.f90:    CALL stop_clock_gpu ('ch_psi_all_gamma')
LR_Modules/cft_wave.f90:#if defined(__CUDA) && defined(_OPENACC)
LR_Modules/cft_wave.f90:#if !defined(__CUDA) || !defined(_OPENACC)
LR_Modules/cft_wave.f90:#if defined(__CUDA) && defined(_OPENACC)
LR_Modules/cft_wave.f90:#if !defined(__CUDA) || !defined(_OPENACC)
LR_Modules/setup_dgc.f90:                 gpu_args_=.TRUE. )
LR_Modules/setup_dgc.f90:                  gpu_args_=.TRUE. )
LR_Modules/setup_dgc.f90:                 gpu_args_=.TRUE. )
LR_Modules/setup_dgc.f90:                  v2c_ud, gpu_args_=.TRUE. )
LR_Modules/incdrhoscf_nc.f90:#if defined(__CUDA)
LR_Modules/incdrhoscf_nc.f90:  CALL start_clock_gpu ('incdrhoscf')
LR_Modules/incdrhoscf_nc.f90:#if defined(__CUDA)
LR_Modules/incdrhoscf_nc.f90:        CALL errore( ' incdrhoscf_nc ', ' taskgroup par not implement with GPU offload', 1 )
LR_Modules/incdrhoscf_nc.f90:  CALL stop_clock_gpu ('incdrhoscf')
COUPLE/CMakeLists.txt:qe_enable_cuda_fortran("${src_couple}")
upflib/ylmr2.f90:  !     Last modified Jan. 2024, by PG: calls CUF version if __CUDA
upflib/ylmr2.f90:#if defined(__CUDA)
upflib/ylmr2.f90:  call ylmr2_gpu(lmax2, ng, g, gg, ylm)
upflib/ylmr2_gpu.f90:module ylmr2_gpum
upflib/ylmr2_gpu.f90:#if defined(__CUDA)
upflib/ylmr2_gpu.f90:! CUDA Kernel version
upflib/ylmr2_gpu.f90:use cudafor
upflib/ylmr2_gpu.f90:attributes(global) subroutine ylmr2_gpu_kernel (lmax,lmax2, ng, g, gg, ylm)
upflib/ylmr2_gpu.f90:  end subroutine ylmr2_gpu_kernel
upflib/ylmr2_gpu.f90:end module ylmr2_gpum
upflib/ylmr2_gpu.f90:subroutine ylmr2_gpu(lmax2, ng, g, gg, ylm)
upflib/ylmr2_gpu.f90:  !     Real spherical harmonics ylm(G) up to l=lmax, GPU version
upflib/ylmr2_gpu.f90:#if defined(__CUDA)
upflib/ylmr2_gpu.f90:  USE cudafor
upflib/ylmr2_gpu.f90:  USE ylmr2_gpum, ONLY : ylmr2_gpu_kernel, maxl
upflib/ylmr2_gpu.f90:#if defined(__CUDA)
upflib/ylmr2_gpu.f90:  ! CUDA Fortran Kernel implementation. Optimizes the use of Q (see below)
upflib/ylmr2_gpu.f90:  call ylmr2_gpu_kernel<<<grid,tBlock>>>(lmax, lmax2, ng, g, gg, ylm)
upflib/ylmr2_gpu.f90:  call upf_error('ylmr2_gpu','you should not be here, go away!',1)
upflib/ylmr2_gpu.f90:end subroutine ylmr2_gpu
upflib/init_us_1.f90:  ! update GPU memory (taking care of zero-dim allocations)
upflib/Makefile:# OBJS_GPU   are GPU-specific
upflib/Makefile:# GPU versions of routines
upflib/Makefile:OBJS_GPU= \
upflib/Makefile:  ylmr2_gpu.o
upflib/Makefile:OBJS_NODEP+= $(OBJS_GPU) dom.o wxml.o
upflib/Makefile:libupf.a: $(OBJS_DEP) $(OBJS_NODEP) $(OBJS_GPU)
upflib/CMakeLists.txt:    # GPU
upflib/CMakeLists.txt:    ylmr2_gpu.f90)
upflib/CMakeLists.txt:qe_enable_cuda_fortran("${src_upflib}")
upflib/CMakeLists.txt:        qe_openacc_fortran
upflib/gth.f90:  !! GPU version of 'dvloc_gth' from 'Modules/gth.f90'
upflib/gen_us_dy.f90:  ! AF: more extensive use of GPU-resident vars possible
upflib/uspp.f90:#if defined(__CUDA)
upflib/uspp.f90:  subroutine allocate_uspp(use_gpu,noncolin,lspinorb,tqr,nhm,nsp,nat,nspin)
upflib/uspp.f90:    logical, intent(in) :: use_gpu
.gitlab-ci-main.yml:  image: nvcr.io/nvidia/nvhpc:21.2-devel-cuda11.2-ubuntu20.04
.gitlab-ci-main.yml:            -DQE_ENABLE_CUDA=ON -DQE_ENABLE_OPENACC=ON -DNVFORTRAN_CUDA_CC=70 .. && make -j 4
.gitlab-ci-main.yml:build:cmake-nvhpc-nocuda:
.gitlab-ci-main.yml:  image: nvcr.io/nvidia/nvhpc:21.2-devel-cuda11.2-ubuntu20.04
.gitlab-ci-main.yml:            -DQE_ENABLE_CUDA=OFF -DQE_ENABLE_OPENACC=OFF .. && make -j 2
CPV/Doc/user_guide.md:-   Sergio Orlandini (CINECA) for completing the CUDA Fortran acceleration
CPV/Doc/user_guide.md:-   Ivan Carnimeo and Pietro Delugas (SISSA) for further openACC acceleration
CPV/Doc/user_guide.md:Users of the GPU-enabled version should also cite the following paper:
CPV/Doc/user_guide.md:advantage of both MPI and OpenMP parallelization and on GPU acceleration.
CPV/Doc/user_guide.md:At the moment the CG routine works only with the plane wave parallelization scheme, both on CPU and GPU machines. It can run "on the fly" by using the autopilot module. 
CPV/CMakeLists.txt:qe_enable_cuda_fortran("${src_cpv}")
CPV/CMakeLists.txt:        qe_openacc_fortran)
CPV/CMakeLists.txt:qe_enable_cuda_fortran("${src_cp_x}")
CPV/CMakeLists.txt:qe_enable_cuda_fortran("${src_manycp_x}")
CPV/src/exx_cg.f90:#ifdef __CUDA
CPV/src/exx_cg.f90:#ifdef __CUDA
CPV/src/exx_cg.f90:#ifdef __CUDA
CPV/src/exx_cg.f90:#ifndef __CUDA
CPV/src/exx_cg.f90:#ifdef __CUDA
CPV/src/exx_cg.f90:#ifndef __CUDA
CPV/src/exx_cg.f90:#ifdef __CUDA
CPV/src/exx_cg.f90:#ifndef __CUDA
CPV/src/exx_cg.f90:#ifdef __CUDA
CPV/src/exx_cg.f90:#ifndef __CUDA
CPV/src/exx_cg.f90:#ifdef __CUDA
CPV/src/exx_cg.f90:#ifndef __CUDA
CPV/src/exx_cg.f90:    !CALL PADX_CUDA(nd_d,nb_d,coeke_d,x_d,d0_d) ! TODO
CPV/src/exx_cg.f90:#ifdef __CUDA
CPV/src/exx_cg.f90:#ifndef __CUDA
CPV/src/exx_cg.f90:#ifdef __CUDA
CPV/src/exx_cg.f90:#ifdef __CUDA
CPV/src/exx_cg.f90:#ifndef __CUDA
CPV/src/cpr.f90:#if defined (__CUDA)
CPV/src/cpr.f90:#if defined(__CUDA)
CPV/src/cpr.f90:#if defined (__CUDA)
CPV/src/cpr.f90:#if defined (__CUDA)
CPV/src/cpr.f90:#if defined (__CUDA)
CPV/src/cpr.f90:#if defined (__CUDA)
CPV/src/cpr.f90:#if defined (__CUDA)
CPV/src/move_electrons.f90:#if defined(__CUDA)
CPV/src/move_electrons.f90:#if defined (__CUDA)
CPV/src/move_electrons.f90:#if defined (__CUDA)
CPV/src/cp_interfaces.f90:#if defined(__CUDA)
CPV/src/cp_interfaces.f90:#if defined(__CUDA)
CPV/src/cp_interfaces.f90:#if defined (__CUDA)
CPV/src/cp_interfaces.f90:      SUBROUTINE dforce_gpu_x( i, bec, vkb, c, df, da, v, ldv, ispin, f, n, nspin )
CPV/src/cp_interfaces.f90:      END SUBROUTINE dforce_gpu_x
CPV/src/cp_interfaces.f90:#if defined (__CUDA)
CPV/src/cp_interfaces.f90:#if defined (__CUDA)
CPV/src/cp_interfaces.f90:    SUBROUTINE c_bgrp_expand_gpu_x( c_bgrp )
CPV/src/cp_interfaces.f90:    END SUBROUTINE c_bgrp_expand_gpu_x
CPV/src/cp_interfaces.f90:#if defined (__CUDA)
CPV/src/cp_interfaces.f90:    SUBROUTINE c_bgrp_pack_gpu_x( c_bgrp )
CPV/src/cp_interfaces.f90:    END SUBROUTINE c_bgrp_pack_gpu_x
CPV/src/cp_interfaces.f90:#if defined (__CUDA)
CPV/src/cp_interfaces.f90:      FUNCTION enkin_gpu_x( c, f, n )
CPV/src/cp_interfaces.f90:         REAL(DP) :: enkin_gpu_x
CPV/src/cp_interfaces.f90:      END FUNCTION enkin_gpu_x
CPV/src/cp_interfaces.f90:#if defined (__CUDA)
CPV/src/cp_interfaces.f90:      SUBROUTINE nlsm1_gpu_x ( n, betae, c, becp, pptype_ )
CPV/src/cp_interfaces.f90:      END SUBROUTINE nlsm1_gpu_x
CPV/src/cp_interfaces.f90:#if defined (__CUDA)
CPV/src/cp_interfaces.f90:      SUBROUTINE  nlsm2_bgrp_gpu_x( ngw, nkb, betae, c_bgrp, becdr_bgrp, nbspx_bgrp, nbsp_bgrp )
CPV/src/cp_interfaces.f90:      END SUBROUTINE nlsm2_bgrp_gpu_x
CPV/src/cp_interfaces.f90:#if defined (__CUDA)
CPV/src/cp_interfaces.f90:      SUBROUTINE calbec_gpu_x( n, betae, c, bec, pptype_ )
CPV/src/cp_interfaces.f90:      END SUBROUTINE calbec_gpu_x
CPV/src/cp_interfaces.f90:#if defined (__CUDA)
CPV/src/cp_interfaces.f90:      SUBROUTINE caldbec_bgrp_gpu_x( eigr, c_bgrp, dbec, idesc )
CPV/src/cp_interfaces.f90:      END SUBROUTINE caldbec_bgrp_gpu_x
CPV/src/cp_interfaces.f90:#if defined (__CUDA)
CPV/src/cp_interfaces.f90:      SUBROUTINE dbeta_eigr_gpu_x( dbeigr, eigr )
CPV/src/cp_interfaces.f90:      END SUBROUTINE dbeta_eigr_gpu_x
CPV/src/ortho.f90:#if defined(__CUDA)
CPV/src/ortho.f90:#if defined(__CUDA)
CPV/src/ortho.f90:#if defined(__CUDA)
CPV/src/ortho.f90:   USE cudafor
CPV/src/ortho.f90:#if defined(__CUDA)
CPV/src/ortho.f90:#if ! defined(__CUDA)
CPV/src/ortho.f90:#if defined(__GPU_MPI)
CPV/src/ortho.f90:      ! Workaround for a bug in the MPI for GPUs: "mp_root_sum", called by
CPV/src/ortho.f90:      ! The bug is present in v.22.7 and previous (?) of the NVIDIA HPC SDK 
CPV/src/ortho.f90:#if defined(__CUDA)
CPV/src/ortho.f90:#if defined(__CUDA)
CPV/src/ortho.f90:#if defined(__CUDA)
CPV/src/ortho.f90:#if defined (__CUDA)
CPV/src/ortho.f90:#if defined (__CUDA)
CPV/src/ortho.f90:#if defined (__CUDA)
CPV/src/gram.f90:#if defined (__CUDA) && defined (_OPENACC)
CPV/src/gram.f90:#if defined(__CUDA) && defined(_OPENACC)
CPV/src/gram.f90:#if defined(__CUDA) && defined (_OPENACC)
CPV/src/gram.f90:#if defined (__CUDA) && (_OPENACC)
CPV/src/cg_sub.f90:#if defined(__CUDA)
CPV/src/cg_sub.f90:#if defined(__CUDA)
CPV/src/cg_sub.f90:         call errore(' runcg_uspp ', ' Ultrasoft case not ported to GPU ', 1)
CPV/src/cg_sub.f90:         call errore(' runcg_uspp ', ' Ensemble DFT case not ported to GPU ', 1)
CPV/src/cg_sub.f90:#if defined (__CUDA)
CPV/src/cg_sub.f90:#if defined (__CUDA)
CPV/src/cg_sub.f90:#if defined (__CUDA)
CPV/src/cg_sub.f90:#if defined (__CUDA)
CPV/src/cg_sub.f90:#if defined (__CUDA)
CPV/src/cg_sub.f90:#if defined (__CUDA)
CPV/src/cg_sub.f90:#if defined (__CUDA)
CPV/src/cg_sub.f90:#if defined(__CUDA)
CPV/src/cg_sub.f90:#if defined(__CUDA)
CPV/src/cg_sub.f90:#if defined(__CUDA)
CPV/src/wave.f90:#if defined (__CUDA)
CPV/src/wave.f90:    SUBROUTINE c_bgrp_expand_gpu_x( c_bgrp )
CPV/src/wave.f90:      USE cudafor
CPV/src/wave.f90:    END SUBROUTINE c_bgrp_expand_gpu_x
CPV/src/wave.f90:    SUBROUTINE c_bgrp_pack_gpu_x( c_bgrp )
CPV/src/wave.f90:      USE cudafor
CPV/src/wave.f90:    END SUBROUTINE c_bgrp_pack_gpu_x
CPV/src/fromscra.f90:#if defined(__CUDA)
CPV/src/fromscra.f90:#if defined(__CUDA)
CPV/src/fromscra.f90:#if defined (__CUDA)
CPV/src/fromscra.f90:#if defined (__CUDA)
CPV/src/fromscra.f90:#if defined (__CUDA)
CPV/src/fromscra.f90:#if defined (__CUDA)
CPV/src/vofrho.f90:#if defined(__CUDA) && defined(_OPENACC)
CPV/src/ortho_base.f90:#if defined(__CUDA)
CPV/src/ortho_base.f90:#if defined(__CUDA)
CPV/src/ortho_base.f90:#if defined(__CUDA)
CPV/src/ortho_base.f90:      USE cudafor
CPV/src/ortho_base.f90:#if defined (__CUDA)
CPV/src/ortho_base.f90:#if defined (__CUDA)
CPV/src/ortho_base.f90:#if defined (__CUDA)
CPV/src/ortho_base.f90:#if defined (__CUDA)
CPV/src/ortho_base.f90:#if defined (__CUDA)
CPV/src/ortho_base.f90:#if defined (__CUDA)
CPV/src/ortho_base.f90:      USE cudafor
CPV/src/ortho_base.f90:#if defined (__CUDA)
CPV/src/ortho_base.f90:#if defined (__CUDA)
CPV/src/ortho_base.f90:#if defined (__CUDA)
CPV/src/ortho_base.f90:#if defined (__CUDA)
CPV/src/ortho_base.f90:#if defined (__CUDA)
CPV/src/exx_module.f90:#if defined __CUDA
CPV/src/exx_module.f90:#ifdef __CUDA
CPV/src/exx_module.f90:#ifdef __CUDA
CPV/src/exx_module.f90:#ifdef __CUDA
CPV/src/exx_module.f90:#ifdef __CUDA
CPV/src/exx_module.f90:#ifdef __CUDA
CPV/src/exx_module.f90:#ifdef __CUDA
CPV/src/exx_module.f90:#ifdef __CUDA
CPV/src/exx_module.f90:#ifdef __CUDA
CPV/src/exx_module.f90:#ifdef __CUDA
CPV/src/exx_module.f90:#ifdef __CUDA
CPV/src/exx_module.f90:#ifdef __CUDA
CPV/src/exx_module.f90:#ifdef __CUDA
CPV/src/exx_module.f90:#ifdef __CUDA
CPV/src/exx_psi.f90:#ifdef __CUDA
CPV/src/exx_psi.f90:#ifdef __CUDA
CPV/src/exx_psi.f90:#ifdef __CUDA
CPV/src/exx_psi.f90:          CALL fftx_c2psi_gamma_gpu( dffts, psis_d, c_d(:,i), ca_d)
CPV/src/exx_psi.f90:#ifdef __CUDA
CPV/src/exx_psi.f90:          CALL fftx_c2psi_gamma_gpu( dffts, psis_d, c_d(:,i), c_d(:, i+1))
CPV/src/exx_psi.f90:#ifdef __CUDA
CPV/src/exx_psi.f90:#ifdef __CUDA
CPV/src/exx_psi.f90:#ifdef __CUDA
CPV/src/exx_psi.f90:#ifdef __CUDA
CPV/src/exx_psi.f90:#ifdef __CUDA
CPV/src/exx_psi.f90:#ifdef __CUDA
CPV/src/chargedensity.f90:#if defined(__CUDA)
CPV/src/chargedensity.f90:#if defined (__CUDA)
CPV/src/chargedensity.f90:#if defined (__CUDA)
CPV/src/chargedensity.f90:#if defined (__CUDA)
CPV/src/chargedensity.f90:            CALL loop_over_states_gpu()
CPV/src/chargedensity.f90:#if defined (__CUDA)
CPV/src/chargedensity.f90:      SUBROUTINE loop_over_states_gpu
CPV/src/chargedensity.f90:         USE cudafor
CPV/src/chargedensity.f90:      END SUBROUTINE loop_over_states_gpu
CPV/src/chargedensity.f90:#if defined (__CUDA)
CPV/src/chargedensity.f90:         USE cudafor
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:!#ifdef __CUDA 
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifndef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifndef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifndef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifndef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifndef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifndef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifdef __CUDA
CPV/src/exx_gs.f90:#ifndef __CUDA
CPV/src/cplib.f90:#if defined (__CUDA)
CPV/src/cplib.f90:   FUNCTION enkin_gpu_x( c, f, n )
CPV/src/cplib.f90:      USE cudafor
CPV/src/cplib.f90:      REAL(DP)                :: enkin_gpu_x
CPV/src/cplib.f90:      enkin_gpu_x = tpiba2 * sk
CPV/src/cplib.f90:   END FUNCTION enkin_gpu_x
CPV/src/forces.f90:#if defined(__CUDA)
CPV/src/forces.f90:#if defined (__CUDA)
CPV/src/forces.f90:      SUBROUTINE dforce_gpu_x ( i, bec, vkb, c, df, da, v, ldv, ispin, f, n, nspin )
CPV/src/forces.f90:      !! GPU double of \(\texttt{dforce_x}\).
CPV/src/forces.f90:      USE cudafor
CPV/src/forces.f90:            CALL fftx_psi2c_gamma_gpu( dffts, psi( 1+ioff : ioff+dffts%nnr ), df_d(1+igno:igno+ngw), da_d(1+igno:igno+ngw))
CPV/src/forces.f90:   END SUBROUTINE dforce_gpu_x
CPV/src/runcp.f90:#if defined(__CUDA)
CPV/src/runcp.f90:#if defined(__CUDA)
CPV/src/runcp.f90:#if defined (__CUDA)
CPV/src/runcp.f90:#if defined (__CUDA)
CPV/src/runcp.f90:        CALL errore(' runcp_uspp ', ' task groups not implemented on GPU ',1)
CPV/src/runcp.f90:#if defined (__CUDA)
CPV/src/runcp.f90:#if defined (__CUDA)
CPV/src/runcp.f90:#if defined (__CUDA)
CPV/src/runcp.f90:#if defined (__CUDA)
CPV/src/nl_base.f90:#if defined(__CUDA)
CPV/src/nl_base.f90:#if defined(__CUDA)
CPV/src/nl_base.f90:#if defined(__CUDA)
CPV/src/nl_base.f90:      USE cudafor
CPV/src/nl_base.f90:#if defined (__CUDA)
CPV/src/nl_base.f90:   subroutine nlsm1_gpu_x ( n, betae, c, becp, pptype_ )
CPV/src/nl_base.f90:#if defined(__CUDA)
CPV/src/nl_base.f90:   end subroutine nlsm1_gpu_x
CPV/src/nl_base.f90:#if defined (__CUDA)
CPV/src/nl_base.f90:   subroutine nlsm2_bgrp_gpu_x( ngw, nkb, betae, c_bgrp, becdr_bgrp, nbspx_bgrp, nbsp_bgrp )
CPV/src/nl_base.f90:      USE cudafor
CPV/src/nl_base.f90:   end subroutine nlsm2_bgrp_gpu_x
CPV/src/nl_base.f90:#if defined (__CUDA)
CPV/src/nl_base.f90:   subroutine calbec_gpu_x ( n, betae, c, bec, pptype_ )
CPV/src/nl_base.f90:   end subroutine calbec_gpu_x
CPV/src/nl_base.f90:#if defined (__CUDA)
CPV/src/nl_base.f90:SUBROUTINE dbeta_eigr_gpu_x( dbeigr, eigr )
CPV/src/nl_base.f90:end subroutine dbeta_eigr_gpu_x
CPV/src/nl_base.f90:#if defined (__CUDA)
CPV/src/nl_base.f90:SUBROUTINE caldbec_bgrp_gpu_x( eigr, c_bgrp, dbec, idesc )
CPV/src/nl_base.f90:end subroutine caldbec_bgrp_gpu_x
CPV/src/nl_base.f90:#if defined (__CUDA)
CPV/src/exx_vofr.f90:#ifdef __CUDA
CPV/src/exx_vofr.f90:  use cudafor
CPV/src/exx_vofr.f90:#ifdef __CUDA
CPV/src/exx_vofr.f90:#ifdef __CUDA
CPV/src/exx_vofr.f90:#ifdef __CUDA
CPV/src/exx_vofr.f90:#ifdef __CUDA
CPV/src/exx_vofr.f90:#ifdef __CUDA
CPV/src/exx_vofr.f90:#ifdef __CUDA
CPV/src/exx_vofr.f90:#ifdef __CUDA
CPV/src/exx_vofr.f90:#ifndef __CUDA
CPV/src/exx_vofr.f90:#ifdef __CUDA
CPV/src/exx_vofr.f90:#ifdef __CUDA
CPV/src/exx_vofr.f90:#ifdef __CUDA
CPV/src/exx_vofr.f90:                  costheta =     z*me_ri(1,i,j,k) ! HK: TODO: GPU method should compute me_ri directly as 1/r
CPV/src/exx_vofr.f90:#ifndef __CUDA
CPV/src/exx_vofr.f90:#ifdef __CUDA
CPV/src/exx_vofr.f90:#ifdef __CUDA
CPV/src/exx_vofr.f90:#ifdef __CUDA
CPV/src/exx_vofr.f90:#ifndef __CUDA
CPV/src/exx_vofr.f90:#ifdef __CUDA
CPV/src/exx_vofr.f90:#ifdef __CUDA
CPV/src/exx_vofr.f90:#ifndef __CUDA
CPV/src/mainvar.f90:#if defined(__CUDA)
CPV/src/mainvar.f90:#if defined (__CUDA)
CPV/src/mainvar.f90:  !! GPU double of \(\text{eigr}\)
CPV/src/mainvar.f90:  !! GPU double of bec (band group)
CPV/src/mainvar.f90:#if defined (__CUDA)
CPV/src/mainvar.f90:  !! GPU double of \(\text{dbec}\) 
CPV/src/mainvar.f90:#if defined (__CUDA)
CPV/src/mainvar.f90:#if defined (__CUDA)
CPV/src/mainvar.f90:#if defined (__CUDA)
CPV/src/mainvar.f90:#if defined (__CUDA)
CPV/src/mainvar.f90:#if defined (__CUDA)
CPV/src/mainvar.f90:#if defined (__CUDA)
CPV/src/init_run.f90:#if defined (__CUDA)
CPV/src/init_run.f90:  USE cudafor
CPV/src/cprstart.f90:  USE environment,   ONLY : environment_start, print_cuda_info
CPV/src/cprstart.f90:  CALL print_cuda_info() 
Modules/kind.f90:  TYPE :: offload_kind_acc  ! CUF/OpenACC offload (NVIDIA GPU hardware and software stack)
Modules/kind.f90:  TYPE :: offload_kind_omp  ! OpenMP5 offload (Intel and AMD GPU hardware and software stack)
Modules/gradutils.f90:#if defined(__CUDA) && defined(_OPENACC)
Modules/gradutils.f90:#if !defined(__CUDA) || !defined(_OPENACC)
Modules/gradutils.f90:#if defined(__CUDA) && defined(_OPENACC)
Modules/gradutils.f90:#if !defined(__CUDA) || !defined(_OPENACC)
Modules/invmat.f90:  ! Comment taken from v6.1 (https://github.com/fspiga/qe-gpu)
Modules/invmat.f90:  ! Comment taken from v6.1 (https://github.com/fspiga/qe-gpu)
Modules/mp_exx.f90:#if defined(__CUDA)
Modules/mp_exx.f90:    USE control_flags, ONLY : use_gpu
Modules/mp_exx.f90:       IF(use_gpu) DEALLOCATE(iexx_istart_d)
Modules/mp_exx.f90:    IF(use_gpu) ALLOCATE(iexx_istart_d, source=iexx_istart)
Modules/environment.f90:  PUBLIC :: print_cuda_info
Modules/environment.f90:SUBROUTINE print_cuda_info(check_use_gpu) 
Modules/environment.f90:  USE control_flags,   ONLY : use_gpu_=> use_gpu, iverbosity
Modules/environment.f90:#if defined(__CUDA)
Modules/environment.f90:  USE cudafor
Modules/environment.f90:  LOGICAL, OPTIONAL,INTENT(IN)  :: check_use_gpu 
Modules/environment.f90:  !! if present and trues the internal variable use_gpu is checked
Modules/environment.f90:#if defined (__CUDA) 
Modules/environment.f90:  TYPE (cudaDeviceProp) :: prop
Modules/environment.f90:  LOGICAL               :: use_gpu = .TRUE. 
Modules/environment.f90:  IF ( PRESENT(check_use_gpu) ) THEN 
Modules/environment.f90:    IF (check_use_gpu) use_gpu = use_gpu_ 
Modules/environment.f90:  ierr = cudaGetDevice( idev )
Modules/environment.f90:  ierr = cudaGetDeviceCount( ndev )
Modules/environment.f90:  IF (use_gpu) THEN
Modules/environment.f90:     WRITE( stdout, '(/,5X,"GPU acceleration is ACTIVE. ",i2," visible GPUs per MPI rank")' ) ndev
Modules/environment.f90:#if defined(__GPU_MPI)
Modules/environment.f90:     WRITE( stdout, '(5x, "GPU-aware MPI enabled")')
Modules/environment.f90:     WRITE( stdout, '(/,5X,"GPU acceleration is NOT ACTIVE.",/)' )
Modules/environment.f90:     CALL infomsg('print_cuda_info', &
Modules/environment.f90:      'High GPU oversubscription detected. Are you sure this is what you want?')
Modules/environment.f90:     WRITE( stdout, '(/,5X,"GPU used by master process:",/)' )
Modules/environment.f90:     ! https://devblogs.nvidia.com/how-query-device-properties-and-handle-errors-cuda-fortran/
Modules/environment.f90:     ierr = cudaGetDeviceProperties(prop, idev)
Modules/environment.f90:END SUBROUTINE print_cuda_info
Modules/fft_wave.f90:    !! gpu-enabled only (many_fft>1 case):  
Modules/fft_wave.f90:    !! gpu-enabled only (many_fft>1 case):  
Modules/electrons_base.f90:#if defined(__CUDA)
Modules/electrons_base.f90:#if defined (__CUDA)
Modules/electrons_base.f90:      USE cudafor
Modules/electrons_base.f90:#if defined (__CUDA)
Modules/electrons_base.f90:#if defined (__CUDA)
Modules/random_numbers_gpu.f90:MODULE random_numbers_gpum
Modules/random_numbers_gpu.f90:  !! Module for random numbers generation - GPU double.
Modules/random_numbers_gpu.f90:#if defined(__CUDA)
Modules/random_numbers_gpu.f90:  USE cudafor
Modules/random_numbers_gpu.f90:    FUNCTION randy_gpu ( irand )
Modules/random_numbers_gpu.f90:      REAL(DP) :: randy_gpu
Modules/random_numbers_gpu.f90:#if defined(__CUDA)
Modules/random_numbers_gpu.f90:      attributes(DEVICE) :: randy_gpu
Modules/random_numbers_gpu.f90:      call errore('randy','use randy_vect_gpu on GPUs',1)
Modules/random_numbers_gpu.f90:    END FUNCTION randy_gpu
Modules/random_numbers_gpu.f90:    SUBROUTINE randy_vect_gpu ( r_d, n, irand )
Modules/random_numbers_gpu.f90:      ! randy_vect_gpu(r, n, irand): reseed with initial seed idum=irand ( 0 <= irand <= ic, see below)
Modules/random_numbers_gpu.f90:      ! randy_vect_gpu(r, n) : generate uniform real(DP) numbers x in [0,1]
Modules/random_numbers_gpu.f90:#if defined(__CUDA)
Modules/random_numbers_gpu.f90:#if defined(__CUDA)
Modules/random_numbers_gpu.f90:#if defined(__CUDA)
Modules/random_numbers_gpu.f90:      ! randy_vect_gpu is not a GPU array in this case
Modules/random_numbers_gpu.f90:    END SUBROUTINE randy_vect_gpu
Modules/random_numbers_gpu.f90:    SUBROUTINE randy_vect_debug_gpu (r_d, n, irand )
Modules/random_numbers_gpu.f90:      ! randy_vect_debug_gpu(r, n, irand): reseed with initial seed idum=irand ( 0 <= irand <= ic, see below)
Modules/random_numbers_gpu.f90:      ! randy_vect_debug_gpu(r, n) : generate uniform real(DP) numbers x in [0,1]
Modules/random_numbers_gpu.f90:#if defined(__CUDA)
Modules/random_numbers_gpu.f90:    END SUBROUTINE randy_vect_debug_gpu
Modules/random_numbers_gpu.f90:#if defined(__CUDA)
Modules/random_numbers_gpu.f90:      CALL randy_vect_gpu ( drand, 1, iseed )
Modules/random_numbers_gpu.f90:      CALL randy_vect_debug_gpu (drand, 1, iseed )
Modules/random_numbers_gpu.f90:END MODULE random_numbers_gpum
Modules/Makefile:# GPU versions of modules
Modules/Makefile:  random_numbers_gpu.o
Modules/mp_global.f90:#if defined (__CUDA)
Modules/mp_global.f90:    IF ( ntg_ > 1 ) CALL errore('mp_startup','No task groups for GPUs',ntg_)
Modules/mp_global.f90:#if defined (__CUDA_OPTIMIZED)
Modules/CMakeLists.txt:    # GPU
Modules/CMakeLists.txt:    random_numbers_gpu.f90)
Modules/CMakeLists.txt:qe_enable_cuda_fortran("${src_modules}")
Modules/CMakeLists.txt:        qe_openacc_fortran
Modules/CMakeLists.txt:        qe_openacc_fortran)
Modules/CMakeLists.txt:if(QE_ENABLE_CUDA)
Modules/CMakeLists.txt:            CUDA::curand)
Modules/fft_rho.f90:#if defined(_OPENACC)
Modules/fft_rho.f90:#if !defined(_OPENACC)
Modules/fft_rho.f90:#if defined(_OPENACC)
Modules/fft_rho.f90:#if !defined(_OPENACC)
Modules/fft_rho.f90:#if defined(_OPENACC)
Modules/fft_rho.f90:#if !defined(_OPENACC)
Modules/fft_rho.f90:#if defined(_OPENACC)
Modules/fft_rho.f90:#if !defined(_OPENACC)
Modules/fft_rho.f90:#if defined(_OPENACC)
Modules/fft_rho.f90:#if !defined(_OPENACC)
Modules/fft_rho.f90:#if defined(_OPENACC)
Modules/fft_rho.f90:#if !defined(_OPENACC)
Modules/fft_rho.f90:#if defined(_OPENACC)
Modules/fft_rho.f90:#if !defined(_OPENACC)
Modules/fft_rho.f90:#if defined(_OPENACC)
Modules/fft_rho.f90:#if !defined(_OPENACC)
Modules/atomic_wfc_mod.f90:  !! Computation is performed on GPU if available
Modules/mp_world.f90:#if defined(__CUDA)
Modules/mp_world.f90:  use cudafor, ONLY : cudaSetDevice, cudaGetDeviceCount, cudaDeviceSynchronize
Modules/mp_world.f90:#if defined(__MPI) || defined(__CUDA)
Modules/mp_world.f90:#if defined(__CUDA)
Modules/mp_world.f90:    ierr = cudaGetDeviceCount( ndev )
Modules/mp_world.f90:    ierr = cudaSetDevice(mod(key, ndev))
Modules/mp_world.f90:    ierr = cudaDeviceSynchronize()
Modules/mp_world.f90:    write(*,*) "MPI ", key, " on node ", color, " is using GPU: ", mod(key, ndev)
Modules/wave_gauge.f90:#if defined (__CUDA)  
Modules/wave_gauge.f90:#if defined (__CUDA) 
Modules/wave_gauge.f90:#if defined (__CUDA) 
Modules/wave_gauge.f90:#if defined (__CUDA) 
Modules/wavefunctions.f90:#if defined(__CUDA)
Modules/wavefunctions.f90:#if defined (__CUDA)
Modules/wavefunctions.f90:     USE cudafor
Modules/wavefunctions.f90:#if defined(__CUDA)
Modules/wavefunctions.f90:!#if defined(__CUDA)
Modules/wavefunctions.f90:#if defined (__CUDA)
Modules/wavefunctions.f90:       USE control_flags,       ONLY : use_gpu
Modules/wavefunctions.f90:#if defined(__CUDA)
Modules/wavefunctions.f90:       IF(use_gpu) istat = cudaHostUnregister(C_LOC(evc(1,1)))
Modules/wavefunctions.f90:#if defined (__CUDA)
Modules/wavefunctions.f90:#if defined (__CUDA)
Modules/wavefunctions.f90:         CALL errore( ' allocate_cp_wavefunctions ', ' allocating on GPU ', ABS( ierr ) )
Modules/wavefunctions.f90:         CALL errore( ' allocate_cp_wavefunctions ', ' allocating on GPU ', ABS( ierr ) )
Modules/control_flags.f90:    use_gpu = .FALSE.          ! if .TRUE. selects the accelerated version of the subroutines
Modules/control_flags.f90:  TYPE(offload_kind_acc), PUBLIC :: offload_acc  ! flag to select CUF/OpenACC offload type
Modules/control_flags.f90:#if defined(__CUDA)
Modules/control_flags.f90:#elif defined(__OPENMP_GPU)
Modules/control_flags.f90:#if defined(__CUDA)
Modules/becmod.f90:!                             - beta, psi, betapsi can be CPU, OpenACC (or OpenMP5), 
Modules/becmod.f90:!                             - CPU, OpenACC (and OpenMP5) cases are distinguished by type(offload_type)
Modules/becmod.f90:    ! beta, psi, betapsi, are assumed OpenACC data on GPU
Modules/becmod.f90:    ! beta, psi, betapsi, are assumed OpenACC data on GPU
XClib/xc_wrapper_gga.f90:                   gpu_args_ )
XClib/xc_wrapper_gga.f90:  !! Wrapper to gpu or non gpu version of \(\texttt{xc_gcx}\).
XClib/xc_wrapper_gga.f90:  LOGICAL, INTENT(IN), OPTIONAL :: gpu_args_
XClib/xc_wrapper_gga.f90:  !! whether you wish to run on gpu in case use_gpu is true
XClib/xc_wrapper_gga.f90:  LOGICAL :: gpu_args
XClib/xc_wrapper_gga.f90:  gpu_args = .FALSE.
XClib/xc_wrapper_gga.f90:  IF ( PRESENT(gpu_args_) ) gpu_args = gpu_args_
XClib/xc_wrapper_gga.f90:  IF ( gpu_args ) THEN
XClib/xc_wrapper_gga.f90:  !! GGA wrapper routine - gpu double.
XClib/qe_drivers_d_lda_lsda.f90:#if defined(_OPENACC)
XClib/qe_drivers_d_lda_lsda.f90:#if !defined(_OPENACC)
XClib/qe_drivers_d_lda_lsda.f90:#if defined(_OPENACC)
XClib/xc_wrapper_d_gga.f90:SUBROUTINE dgcxc( length, sp, r_in, g_in, dvxc_rr, dvxc_sr, dvxc_ss, gpu_args_ )
XClib/xc_wrapper_d_gga.f90:  LOGICAL, OPTIONAL, INTENT(IN) :: gpu_args_
XClib/xc_wrapper_d_gga.f90:  !! whether you wish to run on gpu in case use_gpu is true
XClib/xc_wrapper_d_gga.f90:  LOGICAL :: gpu_args
XClib/xc_wrapper_d_gga.f90:  gpu_args = .FALSE.
XClib/xc_wrapper_d_gga.f90:  IF ( PRESENT(gpu_args_) ) gpu_args = gpu_args_
XClib/xc_wrapper_d_gga.f90:  IF ( gpu_args ) THEN
XClib/xc_wrapper_d_lda_lsda.f90:SUBROUTINE dmxc( length, srd, rho_in, dmuxc, gpu_args_ )
XClib/xc_wrapper_d_lda_lsda.f90:  LOGICAL, OPTIONAL, INTENT(IN) :: gpu_args_
XClib/xc_wrapper_d_lda_lsda.f90:  !! whether you wish to run on gpu in case use_gpu is true
XClib/xc_wrapper_d_lda_lsda.f90:  LOGICAL :: gpu_args
XClib/xc_wrapper_d_lda_lsda.f90:  gpu_args = .FALSE.
XClib/xc_wrapper_d_lda_lsda.f90:  IF ( PRESENT(gpu_args_) ) gpu_args = gpu_args_
XClib/xc_wrapper_d_lda_lsda.f90:  IF ( gpu_args ) THEN
XClib/CMakeLists.txt:        qe_openacc_fortran
XClib/CMakeLists.txt:        qe_openacc_c
XClib/xc_wrapper_lda_lsda.f90:SUBROUTINE xc( length, srd, svd, rho_in, ex_out, ec_out, vx_out, vc_out, gpu_args_ )
XClib/xc_wrapper_lda_lsda.f90:  !! Wrapper routine to \(\texttt{xc_}\) or \(\texttt{xc_gpu}\).
XClib/xc_wrapper_lda_lsda.f90:  LOGICAL, OPTIONAL, INTENT(IN) :: gpu_args_
XClib/xc_wrapper_lda_lsda.f90:  !! whether you wish to run on gpu in case use_gpu is true
XClib/xc_wrapper_lda_lsda.f90:  LOGICAL :: gpu_args
XClib/xc_wrapper_lda_lsda.f90:  gpu_args = .FALSE.
XClib/xc_wrapper_lda_lsda.f90:  IF ( PRESENT(gpu_args_) ) gpu_args = gpu_args_
XClib/xc_wrapper_lda_lsda.f90:  IF ( gpu_args ) THEN
XClib/xc_wrapper_lda_lsda.f90:  !! Wrapper xc LDA - openACC version.  
XClib/qe_drivers_mgga.f90:#if defined(_OPENACC)
XClib/qe_drivers_mgga.f90:#if defined(_OPENACC)
XClib/qe_drivers_mgga.f90:#if defined(_OPENACC)
XClib/qe_drivers_mgga.f90:#if defined(_OPENACC)
XClib/qe_drivers_lda_lsda.f90:#if defined(_OPENACC)
XClib/qe_drivers_lda_lsda.f90:#if defined(_OPENACC)
XClib/qe_drivers_lda_lsda.f90:#if defined(_OPENACC)  
XClib/qe_drivers_lda_lsda.f90:#if defined(_OPENACC)
XClib/qe_drivers_gga.f90:#if defined(_OPENACC)
XClib/qe_drivers_gga.f90:#if defined(_OPENACC)
XClib/qe_drivers_gga.f90:#if defined(_OPENACC)
XClib/qe_drivers_gga.f90:#if defined(_OPENACC)
XClib/qe_drivers_gga.f90:#if defined(_OPENACC)
XClib/qe_drivers_gga.f90:#if defined(_OPENACC)
XClib/qe_drivers_gga.f90:#if defined(_OPENACC)
XClib/qe_drivers_gga.f90:#if defined(_OPENACC)
XClib/qe_drivers_gga.f90:#if defined(_OPENACC) 
XClib/qe_drivers_gga.f90:#if defined(_OPENACC)
XClib/xclib_test.f90:  ! ... openacc init (otherwise it offsets the wall time of the first test)
XClib/xclib_test.f90:#if defined(_OPENACC)
XClib/xc_lib.f90:     SUBROUTINE xc( length, srd, svd, rho_in, ex_out, ec_out, vx_out, vc_out, gpu_args_ )
XClib/xc_lib.f90:       LOGICAL,  OPTIONAL, INTENT(IN) :: gpu_args_
XClib/xc_lib.f90:                        gpu_args_ )
XClib/xc_lib.f90:       LOGICAL,  OPTIONAL, INTENT(IN)  :: gpu_args_
XClib/xc_lib.f90:                            v1c, v2c, v3c, gpu_args_ )
XClib/xc_lib.f90:       LOGICAL,  OPTIONAL, INTENT(IN) :: gpu_args_
XClib/xc_lib.f90:     SUBROUTINE dmxc( length, srd, rho_in, dmuxc, gpu_args_ )
XClib/xc_lib.f90:       LOGICAL,  OPTIONAL, INTENT(IN) :: gpu_args_
XClib/xc_lib.f90:     SUBROUTINE dgcxc( length, sp, r_in, g_in, dvxc_rr, dvxc_sr, dvxc_ss, gpu_args_ )
XClib/xc_lib.f90:       LOGICAL, OPTIONAL, INTENT(IN) :: gpu_args_
XClib/xc_wrapper_mgga.f90:                       v2c, v3c, gpu_args_ )
XClib/xc_wrapper_mgga.f90:  !! Wrapper to gpu or non gpu version of \(\texttt{xc_metagcx}\).
XClib/xc_wrapper_mgga.f90:  LOGICAL, INTENT(IN), OPTIONAL :: gpu_args_
XClib/xc_wrapper_mgga.f90:  !! whether you wish to run on gpu in case use_gpu is true
XClib/xc_wrapper_mgga.f90:  LOGICAL :: gpu_args
XClib/xc_wrapper_mgga.f90:  gpu_args = .FALSE.
XClib/xc_wrapper_mgga.f90:  IF ( PRESENT(gpu_args_) ) gpu_args = gpu_args_
XClib/xc_wrapper_mgga.f90:  IF ( gpu_args ) THEN
cmake/FindELPA.cmake:#   - no components are available for now: maybe OpenMP CUDA in the future?
cmake/CrayFortranCompiler.cmake:if(NOT QE_ENABLE_OPENACC)
cmake/CrayFortranCompiler.cmake:  target_compile_options(qe_openacc_fortran INTERFACE "$<$<COMPILE_LANGUAGE:Fortran>:-hnoacc>")
cmake/CrayFortranCompiler.cmake:    message(FATAL_ERROR "Cannot find -haccel=<gpu_arc> option being used by the ftn compiler wrapper. "
cmake/CrayFortranCompiler.cmake:                        "Make sure the GPU architecture module is loaded."
cmake/qeHelpers.cmake:function(qe_enable_cuda_fortran SRCS)
cmake/qeHelpers.cmake:    if(QE_ENABLE_CUDA)
cmake/qeHelpers.cmake:                    COMPILE_OPTIONS "${QE_CUDA_COMPILE_OPTIONS}")
cmake/qeHelpers.cmake:endfunction(qe_enable_cuda_fortran)
cmake/qeHelpers.cmake:function(_qe_add_cuda_link_flags TGT)
cmake/qeHelpers.cmake:                    ${QE_CUDA_COMPILE_OPTIONS})
cmake/qeHelpers.cmake:endfunction(_qe_add_cuda_link_flags)
cmake/qeHelpers.cmake:    if(QE_ENABLE_CUDA)
cmake/qeHelpers.cmake:        _qe_add_cuda_link_flags(${TGT})
cmake/NVFortranCompiler.cmake:    if(QE_ENABLE_OPENACC AND QE_ENABLE_OPENMP)
cmake/NVFortranCompiler.cmake:                            " when QE is compiled with both OpenMP and OpenACC. "
cmake/NVFortranCompiler.cmake:# set up GPU architecture options which can be applied to CUDA Fortran, OpenACC and OpenMP offload
cmake/NVFortranCompiler.cmake:set(GPU_TARGET_COMPILE_OPTIONS)
cmake/NVFortranCompiler.cmake:if(QE_ENABLE_CUDA OR QE_ENABLE_OPENACC OR QE_ENABLE_OFFLOAD)
cmake/NVFortranCompiler.cmake:                            "GPU acceleration requires PGI 19.10 or NVIDIA HPC SDK 20.7 or higher!")
cmake/NVFortranCompiler.cmake:    if(DEFINED NVFORTRAN_CUDA_VERSION)
cmake/NVFortranCompiler.cmake:        list(APPEND GPU_TARGET_COMPILE_OPTIONS "-gpu=cuda${NVFORTRAN_CUDA_VERSION}")
cmake/NVFortranCompiler.cmake:    if(DEFINED NVFORTRAN_CUDA_CC)
cmake/NVFortranCompiler.cmake:        list(APPEND GPU_TARGET_COMPILE_OPTIONS "-gpu=cc${NVFORTRAN_CUDA_CC}")
cmake/NVFortranCompiler.cmake:    elseif(DEFINED QE_GPU_ARCHS)
cmake/NVFortranCompiler.cmake:        string(REPLACE "sm_" "" CUDA_ARCH_NUMBERS "${QE_GPU_ARCHS}")
cmake/NVFortranCompiler.cmake:        string(REPLACE ";" ",cc" OFFLOAD_ARCH "${CUDA_ARCH_NUMBERS}")
cmake/NVFortranCompiler.cmake:        list(APPEND GPU_TARGET_COMPILE_OPTIONS "-gpu=cc${OFFLOAD_ARCH}")
cmake/NVFortranCompiler.cmake:if(QE_ENABLE_CUDA)
cmake/NVFortranCompiler.cmake:        set(CUDA_FLAG "-cuda")
cmake/NVFortranCompiler.cmake:        set(CUDA_FLAG "-Mcuda")
cmake/NVFortranCompiler.cmake:    set(QE_CUDA_COMPILE_OPTIONS ${CUDA_FLAG})
cmake/NVFortranCompiler.cmake:    if(GPU_TARGET_COMPILE_OPTIONS)
cmake/NVFortranCompiler.cmake:      list(APPEND QE_CUDA_COMPILE_OPTIONS ${GPU_TARGET_COMPILE_OPTIONS})
cmake/NVFortranCompiler.cmake:    message("   nvfortran CUDA related compile and link options : ${QE_CUDA_COMPILE_OPTIONS}")
cmake/NVFortranCompiler.cmake:    set(CMAKE_REQUIRED_LINK_OPTIONS ${QE_CUDA_COMPILE_OPTIONS})
cmake/NVFortranCompiler.cmake:    check_fortran_compiler_flag("${QE_CUDA_COMPILE_OPTIONS}" NVFORTRAN_CUDA_VALID)
cmake/NVFortranCompiler.cmake:    if(NOT NVFORTRAN_CUDA_VALID)
cmake/NVFortranCompiler.cmake:        unset(NVFORTRAN_CUDA_VALID CACHE)
cmake/NVFortranCompiler.cmake:        message(FATAL_ERROR "nvfortran CUDA related option check failed! "
cmake/NVFortranCompiler.cmake:    # -O3 makes the CUDA runs fail at stres_us_gpu.f90, thus override
cmake/NVFortranCompiler.cmake:if(QE_ENABLE_OPENACC)
cmake/NVFortranCompiler.cmake:    if(GPU_TARGET_COMPILE_OPTIONS)
cmake/NVFortranCompiler.cmake:        target_compile_options(qe_openacc_fortran INTERFACE "$<$<COMPILE_LANGUAGE:Fortran>:${GPU_TARGET_COMPILE_OPTIONS}>")
cmake/NVFortranCompiler.cmake:        target_compile_options(qe_openacc_c INTERFACE "$<$<COMPILE_LANGUAGE:C>:${GPU_TARGET_COMPILE_OPTIONS}>")
cmake/NVFortranCompiler.cmake:    target_compile_options(qe_openmp_fortran INTERFACE "$<$<COMPILE_LANGUAGE:Fortran>:-mp=gpu>")
cmake/NVFortranCompiler.cmake:    target_link_options(qe_openmp_fortran INTERFACE "$<$<LINK_LANGUAGE:Fortran>:-mp=gpu>")
cmake/NVFortranCompiler.cmake:    if(GPU_TARGET_COMPILE_OPTIONS)
cmake/NVFortranCompiler.cmake:        target_compile_options(qe_openmp_fortran INTERFACE "$<$<COMPILE_LANGUAGE:Fortran>:${GPU_TARGET_COMPILE_OPTIONS}>")
cmake/GNUFortranCompiler.cmake:  if(NOT DEFINED QE_GPU_ARCHS)
cmake/GNUFortranCompiler.cmake:    message(FATAL_ERROR "Requires QE_GPU_ARCHS option. For example, sm_80 for NVIDIA A100 or gfx90a for AMD MI250X.")
cmake/GNUFortranCompiler.cmake:  if(QE_GPU_ARCHS MATCHES "sm_")
cmake/GNUFortranCompiler.cmake:  elseif(QE_GPU_ARCHS MATCHES "gfx")
cmake/GNUFortranCompiler.cmake:    message(FATAL_ERROR "Cannot derive OFFLOAD_TARGET from QE_GPU_ARCHS.")
cmake/GNUFortranCompiler.cmake:    target_compile_options(qe_openmp_fortran INTERFACE "-foffload-options=${OFFLOAD_TARGET}=-march=${QE_GPU_ARCHS}")
cmake/GNUFortranCompiler.cmake:    target_compile_options(qe_openmp_fortran INTERFACE "-foffload-options=${OFFLOAD_TARGET}=-misa=${QE_GPU_ARCHS}")
cmake/FindSCALAPACK.cmake:          "Intel( older versions of mkl 32 and 64 bit), ACML, ACML_MP, ACML_GPU, Apple, NAS, Generic")
cmake/FindSCALAPACK.cmake:            "Intel( older versions of mkl 32 and 64 bit), ACML, ACML_MP, ACML_GPU, Apple, NAS, Generic")
GWW/CMakeLists.txt:qe_enable_cuda_fortran("${src_pw4gww}")
UtilXlib/Makefile.test:mp_base_gpu.o \
UtilXlib/data_buffer.f90:#ifdef __CUDA
UtilXlib/data_buffer.f90:    USE cudafor
UtilXlib/data_buffer.f90:#ifdef __CUDA
UtilXlib/tests/test_mp_bcast_lv_buffer_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_bcast_lv_buffer_gpu.f90:PROGRAM test_mp_bcast_lv_buffer_gpu
UtilXlib/tests/test_mp_bcast_lv_buffer_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_bcast_lv_buffer_gpu.f90:END PROGRAM test_mp_bcast_lv_buffer_gpu
UtilXlib/tests/test_mp_bcast_lv_buffer_gpu.f90:PROGRAM test_mp_bcast_lv_buffer_gpu
UtilXlib/tests/test_mp_bcast_lv_buffer_gpu.f90:END PROGRAM test_mp_bcast_lv_buffer_gpu
UtilXlib/tests/test_mp_min_iv_buffer_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_min_iv_buffer_gpu.f90:PROGRAM test_mp_min_iv_buffer_gpu
UtilXlib/tests/test_mp_min_iv_buffer_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_min_iv_buffer_gpu.f90:END PROGRAM test_mp_min_iv_buffer_gpu
UtilXlib/tests/test_mp_min_iv_buffer_gpu.f90:PROGRAM test_mp_min_iv_buffer_gpu
UtilXlib/tests/test_mp_min_iv_buffer_gpu.f90:END PROGRAM test_mp_min_iv_buffer_gpu
UtilXlib/tests/test_mp_bcast_gpu.tmpl:#if defined(__CUDA)
UtilXlib/tests/test_mp_bcast_gpu.tmpl:PROGRAM test_mp_bcast_{vname}_gpu
UtilXlib/tests/test_mp_bcast_gpu.tmpl:    USE cudafor
UtilXlib/tests/test_mp_bcast_gpu.tmpl:    CALL save_random_seed("test_mp_bcast_{vname}_gpu", mpime)
UtilXlib/tests/test_mp_bcast_gpu.tmpl:END PROGRAM test_mp_bcast_{vname}_gpu
UtilXlib/tests/test_mp_bcast_gpu.tmpl:PROGRAM test_mp_bcast_{vname}_gpu
UtilXlib/tests/test_mp_bcast_gpu.tmpl:END PROGRAM test_mp_bcast_{vname}_gpu
UtilXlib/tests/test_mp_get_gpu.tmpl:#if defined(__CUDA)
UtilXlib/tests/test_mp_get_gpu.tmpl:PROGRAM test_mp_get_{vname}_gpu
UtilXlib/tests/test_mp_get_gpu.tmpl:    USE cudafor
UtilXlib/tests/test_mp_get_gpu.tmpl:    CALL save_random_seed("test_mp_get_{vname}_gpu", mpime)
UtilXlib/tests/test_mp_get_gpu.tmpl:END PROGRAM test_mp_get_{vname}_gpu
UtilXlib/tests/test_mp_get_gpu.tmpl:PROGRAM test_mp_get_{vname}_gpu
UtilXlib/tests/test_mp_get_gpu.tmpl:END PROGRAM test_mp_get_{vname}_gpu
UtilXlib/tests/test_mp_gather_gpu.tmpl:#if defined(__CUDA)
UtilXlib/tests/test_mp_gather_gpu.tmpl:PROGRAM test_mp_gather_{vname}_gpu
UtilXlib/tests/test_mp_gather_gpu.tmpl:    USE cudafor
UtilXlib/tests/test_mp_gather_gpu.tmpl:    CALL save_random_seed("test_mp_gather_{vname}_gpu", mpime)
UtilXlib/tests/test_mp_gather_gpu.tmpl:END PROGRAM test_mp_gather_{vname}_gpu
UtilXlib/tests/test_mp_gather_gpu.tmpl:PROGRAM test_mp_gather_{vname}_gpu
UtilXlib/tests/test_mp_gather_gpu.tmpl:END PROGRAM test_mp_gather_{vname}_gpu
UtilXlib/tests/test_mp_max_gpu.tmpl:#if defined(__CUDA) 
UtilXlib/tests/test_mp_max_gpu.tmpl:PROGRAM test_mp_max_{vname}_gpu
UtilXlib/tests/test_mp_max_gpu.tmpl:    USE cudafor
UtilXlib/tests/test_mp_max_gpu.tmpl:    CALL save_random_seed("test_mp_max_{vname}_gpu", mpime)
UtilXlib/tests/test_mp_max_gpu.tmpl:END PROGRAM test_mp_max_{vname}_gpu
UtilXlib/tests/test_mp_max_gpu.tmpl:PROGRAM test_mp_max_{vname}_gpu
UtilXlib/tests/test_mp_max_gpu.tmpl:END PROGRAM test_mp_max_{vname}_gpu
UtilXlib/tests/test_mp_bcast_i1_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_bcast_i1_gpu.f90:PROGRAM test_mp_bcast_i1_gpu
UtilXlib/tests/test_mp_bcast_i1_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_bcast_i1_gpu.f90:END PROGRAM test_mp_bcast_i1_gpu
UtilXlib/tests/test_mp_bcast_i1_gpu.f90:PROGRAM test_mp_bcast_i1_gpu
UtilXlib/tests/test_mp_bcast_i1_gpu.f90:END PROGRAM test_mp_bcast_i1_gpu
UtilXlib/tests/test_mp_allgatherv_type_gpu.tmpl:#if defined(__CUDA)
UtilXlib/tests/test_mp_allgatherv_type_gpu.tmpl:PROGRAM test_mp_gatherv_{vname}_gpu
UtilXlib/tests/test_mp_allgatherv_type_gpu.tmpl:    USE cudafor
UtilXlib/tests/test_mp_allgatherv_type_gpu.tmpl:    CALL save_random_seed("test_mp_gatherv_{vname}_gpu", mpime)
UtilXlib/tests/test_mp_allgatherv_type_gpu.tmpl:END PROGRAM test_mp_gatherv_{vname}_gpu
UtilXlib/tests/test_mp_allgatherv_type_gpu.tmpl:PROGRAM test_mp_gatherv_{vname}_gpu
UtilXlib/tests/test_mp_allgatherv_type_gpu.tmpl:END PROGRAM test_mp_gatherv_{vname}_gpu
UtilXlib/tests/test_mp_bcast_im_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_bcast_im_gpu.f90:PROGRAM test_mp_bcast_im_gpu
UtilXlib/tests/test_mp_bcast_im_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_bcast_im_gpu.f90:END PROGRAM test_mp_bcast_im_gpu
UtilXlib/tests/test_mp_bcast_im_gpu.f90:PROGRAM test_mp_bcast_im_gpu
UtilXlib/tests/test_mp_bcast_im_gpu.f90:END PROGRAM test_mp_bcast_im_gpu
UtilXlib/tests/Makefile:       test_mp_bcast_i1_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_bcast_iv_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_bcast_im_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_bcast_it_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_bcast_i4d_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_bcast_r4d_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_bcast_c4d_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_bcast_c5d_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_bcast_r5d_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_bcast_c6d_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_bcast_iv_buffer_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_bcast_lv_buffer_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_bcast_rv_buffer_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_max_iv_buffer_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_max_rv_buffer_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_min_iv_buffer_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_min_rv_buffer_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_sum_iv_buffer_gpu.f90 \
UtilXlib/tests/Makefile:       test_mp_sum_rv_buffer_gpu.f90
UtilXlib/tests/README.md: * -c: CUDA-Fortran interface with data transfer done with memory on the *host*
UtilXlib/tests/README.md: * -n: CUDA-Fortran interface with data transfer done with memory on the *device*
UtilXlib/tests/compile_and_run_tests.sh:GPU_ARCH=35
UtilXlib/tests/compile_and_run_tests.sh:CUDA_RUNTIME=8.0
UtilXlib/tests/compile_and_run_tests.sh:add_cuda=""
UtilXlib/tests/compile_and_run_tests.sh:add_cudampi=""
UtilXlib/tests/compile_and_run_tests.sh:      echo "CUDA build scheduled!" >&2
UtilXlib/tests/compile_and_run_tests.sh:      add_cuda="yes"
UtilXlib/tests/compile_and_run_tests.sh:      echo "CUDA+MPI build scheduled!" >&2
UtilXlib/tests/compile_and_run_tests.sh:      add_cudampi="yes"
UtilXlib/tests/compile_and_run_tests.sh:      echo "-c : cuda build" >&2
UtilXlib/tests/compile_and_run_tests.sh:      echo "-n : gpu+mpi (nvlink) build" >&2
UtilXlib/tests/compile_and_run_tests.sh:if [ "$add_cuda" != "" ]; then
UtilXlib/tests/compile_and_run_tests.sh:    flags+=(" $flag -Mcuda=cc${GPU_ARCH},cuda${CUDA_RUNTIME} -D__CUDA")
UtilXlib/tests/compile_and_run_tests.sh:if [ "$add_cudampi" != "" ]; then
UtilXlib/tests/compile_and_run_tests.sh:  flags+=(" -Mcuda=cc${GPU_ARCH},cuda${CUDA_RUNTIME} -D__MPI -D__CUDA -D__GPU_MPI")
UtilXlib/tests/compile_and_run_tests.sh:#declare -a    flags=("" "-D__CUDA" "-D__MPI" "-D__MPI -D__CUDA" "-D__MPI -D__CUDA -D__GPU_MPI")
UtilXlib/tests/test_mp_put_gpu.tmpl:#if defined(__CUDA)
UtilXlib/tests/test_mp_put_gpu.tmpl:PROGRAM test_mp_put_{vname}_gpu
UtilXlib/tests/test_mp_put_gpu.tmpl:    USE cudafor
UtilXlib/tests/test_mp_put_gpu.tmpl:    CALL save_random_seed("test_mp_put_{vname}_gpu", mpime)
UtilXlib/tests/test_mp_put_gpu.tmpl:END PROGRAM test_mp_put_{vname}_gpu
UtilXlib/tests/test_mp_put_gpu.tmpl:PROGRAM test_mp_put_{vname}_gpu
UtilXlib/tests/test_mp_put_gpu.tmpl:END PROGRAM test_mp_put_{vname}_gpu
UtilXlib/tests/test_offload_macros.f90:program test_gpu_macros
UtilXlib/tests/test_offload_macros.f90:#if defined(_OPENACC)
UtilXlib/tests/test_offload_macros.f90:  write(*,*) "Using OpenACC"
UtilXlib/tests/test_offload_macros.f90:#elif defined(__OPENMP_GPU)
UtilXlib/tests/test_offload_macros.f90:  write(*,*) "Using OpenMP GPU offload"
UtilXlib/tests/test_offload_macros.f90:  write(*,*) "Neither OpenMP nor OpenACC"
UtilXlib/tests/test_mp_bcast_iv_buffer_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_bcast_iv_buffer_gpu.f90:PROGRAM test_mp_bcast_iv_buffer_gpu
UtilXlib/tests/test_mp_bcast_iv_buffer_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_bcast_iv_buffer_gpu.f90:END PROGRAM test_mp_bcast_iv_buffer_gpu
UtilXlib/tests/test_mp_bcast_iv_buffer_gpu.f90:PROGRAM test_mp_bcast_iv_buffer_gpu
UtilXlib/tests/test_mp_bcast_iv_buffer_gpu.f90:END PROGRAM test_mp_bcast_iv_buffer_gpu
UtilXlib/tests/test_mp_bcast_c6d_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_bcast_c6d_gpu.f90:PROGRAM test_mp_bcast_c6d_gpu
UtilXlib/tests/test_mp_bcast_c6d_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_bcast_c6d_gpu.f90:END PROGRAM test_mp_bcast_c6d_gpu
UtilXlib/tests/test_mp_bcast_c6d_gpu.f90:PROGRAM test_mp_bcast_c6d_gpu
UtilXlib/tests/test_mp_bcast_c6d_gpu.f90:END PROGRAM test_mp_bcast_c6d_gpu
UtilXlib/tests/test_mp_bcast_r5d_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_bcast_r5d_gpu.f90:PROGRAM test_mp_bcast_r5d_gpu
UtilXlib/tests/test_mp_bcast_r5d_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_bcast_r5d_gpu.f90:END PROGRAM test_mp_bcast_r5d_gpu
UtilXlib/tests/test_mp_bcast_r5d_gpu.f90:PROGRAM test_mp_bcast_r5d_gpu
UtilXlib/tests/test_mp_bcast_r5d_gpu.f90:END PROGRAM test_mp_bcast_r5d_gpu
UtilXlib/tests/test_mp_alltoall_gpu.tmpl:#if defined(__CUDA)
UtilXlib/tests/test_mp_alltoall_gpu.tmpl:PROGRAM test_mp_alltoall_{vname}_gpu
UtilXlib/tests/test_mp_alltoall_gpu.tmpl:    USE cudafor
UtilXlib/tests/test_mp_alltoall_gpu.tmpl:    CALL save_random_seed("test_mp_alltoall_{vname}_gpu", mpime)
UtilXlib/tests/test_mp_alltoall_gpu.tmpl:END PROGRAM test_mp_alltoall_{vname}_gpu
UtilXlib/tests/test_mp_alltoall_gpu.tmpl:PROGRAM test_mp_alltoall_{vname}_gpu
UtilXlib/tests/test_mp_alltoall_gpu.tmpl:END PROGRAM test_mp_alltoall_{vname}_gpu
UtilXlib/tests/test_mp_sum_iv_buffer_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_sum_iv_buffer_gpu.f90:PROGRAM test_mp_sum_iv_buffer_gpu
UtilXlib/tests/test_mp_sum_iv_buffer_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_sum_iv_buffer_gpu.f90:END PROGRAM test_mp_sum_iv_buffer_gpu
UtilXlib/tests/test_mp_sum_iv_buffer_gpu.f90:PROGRAM test_mp_sum_iv_buffer_gpu
UtilXlib/tests/test_mp_sum_iv_buffer_gpu.f90:END PROGRAM test_mp_sum_iv_buffer_gpu
UtilXlib/tests/test_mp_bcast_c5d_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_bcast_c5d_gpu.f90:PROGRAM test_mp_bcast_c5d_gpu
UtilXlib/tests/test_mp_bcast_c5d_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_bcast_c5d_gpu.f90:END PROGRAM test_mp_bcast_c5d_gpu
UtilXlib/tests/test_mp_bcast_c5d_gpu.f90:PROGRAM test_mp_bcast_c5d_gpu
UtilXlib/tests/test_mp_bcast_c5d_gpu.f90:END PROGRAM test_mp_bcast_c5d_gpu
UtilXlib/tests/test_mp_bcast_c4d_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_bcast_c4d_gpu.f90:PROGRAM test_mp_bcast_c4d_gpu
UtilXlib/tests/test_mp_bcast_c4d_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_bcast_c4d_gpu.f90:END PROGRAM test_mp_bcast_c4d_gpu
UtilXlib/tests/test_mp_bcast_c4d_gpu.f90:PROGRAM test_mp_bcast_c4d_gpu
UtilXlib/tests/test_mp_bcast_c4d_gpu.f90:END PROGRAM test_mp_bcast_c4d_gpu
UtilXlib/tests/test_mp_max_iv_buffer_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_max_iv_buffer_gpu.f90:PROGRAM test_mp_max_iv_buffer_gpu
UtilXlib/tests/test_mp_max_iv_buffer_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_max_iv_buffer_gpu.f90:END PROGRAM test_mp_max_iv_buffer_gpu
UtilXlib/tests/test_mp_max_iv_buffer_gpu.f90:PROGRAM test_mp_max_iv_buffer_gpu
UtilXlib/tests/test_mp_max_iv_buffer_gpu.f90:END PROGRAM test_mp_max_iv_buffer_gpu
UtilXlib/tests/test_mp_max_rv_buffer_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_max_rv_buffer_gpu.f90:PROGRAM test_mp_max_rv_buffer_gpu
UtilXlib/tests/test_mp_max_rv_buffer_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_max_rv_buffer_gpu.f90:END PROGRAM test_mp_max_rv_buffer_gpu
UtilXlib/tests/test_mp_max_rv_buffer_gpu.f90:PROGRAM test_mp_max_rv_buffer_gpu
UtilXlib/tests/test_mp_max_rv_buffer_gpu.f90:END PROGRAM test_mp_max_rv_buffer_gpu
UtilXlib/tests/test_mp_bcast_r4d_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_bcast_r4d_gpu.f90:PROGRAM test_mp_bcast_r4d_gpu
UtilXlib/tests/test_mp_bcast_r4d_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_bcast_r4d_gpu.f90:END PROGRAM test_mp_bcast_r4d_gpu
UtilXlib/tests/test_mp_bcast_r4d_gpu.f90:PROGRAM test_mp_bcast_r4d_gpu
UtilXlib/tests/test_mp_bcast_r4d_gpu.f90:END PROGRAM test_mp_bcast_r4d_gpu
UtilXlib/tests/test_mp_bcast_i4d_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_bcast_i4d_gpu.f90:PROGRAM test_mp_bcast_i4d_gpu
UtilXlib/tests/test_mp_bcast_i4d_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_bcast_i4d_gpu.f90:END PROGRAM test_mp_bcast_i4d_gpu
UtilXlib/tests/test_mp_bcast_i4d_gpu.f90:PROGRAM test_mp_bcast_i4d_gpu
UtilXlib/tests/test_mp_bcast_i4d_gpu.f90:END PROGRAM test_mp_bcast_i4d_gpu
UtilXlib/tests/test_mp_bcast_rv_buffer_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_bcast_rv_buffer_gpu.f90:PROGRAM test_mp_bcast_rv_buffer_gpu
UtilXlib/tests/test_mp_bcast_rv_buffer_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_bcast_rv_buffer_gpu.f90:END PROGRAM test_mp_bcast_rv_buffer_gpu
UtilXlib/tests/test_mp_bcast_rv_buffer_gpu.f90:PROGRAM test_mp_bcast_rv_buffer_gpu
UtilXlib/tests/test_mp_bcast_rv_buffer_gpu.f90:END PROGRAM test_mp_bcast_rv_buffer_gpu
UtilXlib/tests/test_mp_bcast_iv_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_bcast_iv_gpu.f90:PROGRAM test_mp_bcast_iv_gpu
UtilXlib/tests/test_mp_bcast_iv_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_bcast_iv_gpu.f90:END PROGRAM test_mp_bcast_iv_gpu
UtilXlib/tests/test_mp_bcast_iv_gpu.f90:PROGRAM test_mp_bcast_iv_gpu
UtilXlib/tests/test_mp_bcast_iv_gpu.f90:END PROGRAM test_mp_bcast_iv_gpu
UtilXlib/tests/test_mp_gatherv_type_gpu.tmpl:#if defined(__CUDA)
UtilXlib/tests/test_mp_gatherv_type_gpu.tmpl:PROGRAM test_mp_gatherv_{vname}_gpu
UtilXlib/tests/test_mp_gatherv_type_gpu.tmpl:    USE cudafor
UtilXlib/tests/test_mp_gatherv_type_gpu.tmpl:    CALL save_random_seed("test_mp_gatherv_{vname}_gpu", mpime)
UtilXlib/tests/test_mp_gatherv_type_gpu.tmpl:END PROGRAM test_mp_gatherv_{vname}_gpu
UtilXlib/tests/test_mp_gatherv_type_gpu.tmpl:PROGRAM test_mp_gatherv_{vname}_gpu
UtilXlib/tests/test_mp_gatherv_type_gpu.tmpl:END PROGRAM test_mp_gatherv_{vname}_gpu
UtilXlib/tests/test_mp_min_rv_buffer_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_min_rv_buffer_gpu.f90:PROGRAM test_mp_min_rv_buffer_gpu
UtilXlib/tests/test_mp_min_rv_buffer_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_min_rv_buffer_gpu.f90:END PROGRAM test_mp_min_rv_buffer_gpu
UtilXlib/tests/test_mp_min_rv_buffer_gpu.f90:PROGRAM test_mp_min_rv_buffer_gpu
UtilXlib/tests/test_mp_min_rv_buffer_gpu.f90:END PROGRAM test_mp_min_rv_buffer_gpu
UtilXlib/tests/test_mp_gatherv_gpu.tmpl:#if defined(__CUDA)
UtilXlib/tests/test_mp_gatherv_gpu.tmpl:PROGRAM test_mp_gatherv_{vname}_gpu
UtilXlib/tests/test_mp_gatherv_gpu.tmpl:    USE cudafor
UtilXlib/tests/test_mp_gatherv_gpu.tmpl:    CALL save_random_seed("test_mp_gatherv_{vname}_gpu", mpime)
UtilXlib/tests/test_mp_gatherv_gpu.tmpl:END PROGRAM test_mp_gatherv_{vname}_gpu
UtilXlib/tests/test_mp_gatherv_gpu.tmpl:PROGRAM test_mp_gatherv_{vname}_gpu
UtilXlib/tests/test_mp_gatherv_gpu.tmpl:END PROGRAM test_mp_gatherv_{vname}_gpu
UtilXlib/tests/test_mp_sum_gpu.tmpl:#if defined(__CUDA)
UtilXlib/tests/test_mp_sum_gpu.tmpl:PROGRAM test_mp_sum_{vname}_gpu
UtilXlib/tests/test_mp_sum_gpu.tmpl:    USE cudafor
UtilXlib/tests/test_mp_sum_gpu.tmpl:    CALL save_random_seed("test_mp_sum_{vname}_gpu", mpime)
UtilXlib/tests/test_mp_sum_gpu.tmpl:END PROGRAM test_mp_sum_{vname}_gpu
UtilXlib/tests/test_mp_sum_gpu.tmpl:PROGRAM test_mp_sum_{vname}_gpu
UtilXlib/tests/test_mp_sum_gpu.tmpl:END PROGRAM test_mp_sum_{vname}_gpu
UtilXlib/tests/test_mp_bcast_it_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_bcast_it_gpu.f90:PROGRAM test_mp_bcast_it_gpu
UtilXlib/tests/test_mp_bcast_it_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_bcast_it_gpu.f90:END PROGRAM test_mp_bcast_it_gpu
UtilXlib/tests/test_mp_bcast_it_gpu.f90:PROGRAM test_mp_bcast_it_gpu
UtilXlib/tests/test_mp_bcast_it_gpu.f90:END PROGRAM test_mp_bcast_it_gpu
UtilXlib/tests/test_mp_sum_rv_buffer_gpu.f90:#if defined(__CUDA)
UtilXlib/tests/test_mp_sum_rv_buffer_gpu.f90:PROGRAM test_mp_sum_rv_buffer_gpu
UtilXlib/tests/test_mp_sum_rv_buffer_gpu.f90:    USE cudafor
UtilXlib/tests/test_mp_sum_rv_buffer_gpu.f90:END PROGRAM test_mp_sum_rv_buffer_gpu
UtilXlib/tests/test_mp_sum_rv_buffer_gpu.f90:PROGRAM test_mp_sum_rv_buffer_gpu
UtilXlib/tests/test_mp_sum_rv_buffer_gpu.f90:END PROGRAM test_mp_sum_rv_buffer_gpu
UtilXlib/tests/test_mp_root_sum_gpu.tmpl:#if defined(__CUDA)
UtilXlib/tests/test_mp_root_sum_gpu.tmpl:PROGRAM test_mp_root_sum_{vname}_gpu
UtilXlib/tests/test_mp_root_sum_gpu.tmpl:    USE cudafor
UtilXlib/tests/test_mp_root_sum_gpu.tmpl:    CALL save_random_seed("test_mp_root_sum_{vname}_gpu", mpime)
UtilXlib/tests/test_mp_root_sum_gpu.tmpl:END PROGRAM test_mp_root_sum_{vname}_gpu
UtilXlib/tests/test_mp_root_sum_gpu.tmpl:PROGRAM test_mp_root_sum_{vname}_gpu
UtilXlib/tests/test_mp_root_sum_gpu.tmpl:END PROGRAM test_mp_root_sum_{vname}_gpu
UtilXlib/tests/test_mp_circular_shift_left_gpu.tmpl:#if defined(__CUDA)
UtilXlib/tests/test_mp_circular_shift_left_gpu.tmpl:PROGRAM test_mp_circular_shift_left_{vname}_gpu
UtilXlib/tests/test_mp_circular_shift_left_gpu.tmpl:    USE cudafor
UtilXlib/tests/test_mp_circular_shift_left_gpu.tmpl:    CALL save_random_seed("test_mp_circular_shift_left_{vname}_gpu", mpime)
UtilXlib/tests/test_mp_circular_shift_left_gpu.tmpl:END PROGRAM test_mp_circular_shift_left_{vname}_gpu
UtilXlib/tests/test_mp_circular_shift_left_gpu.tmpl:PROGRAM test_mp_circular_shift_left_{vname}_gpu
UtilXlib/tests/test_mp_circular_shift_left_gpu.tmpl:END PROGRAM test_mp_circular_shift_left_{vname}_gpu
UtilXlib/tests/test_mp_min_gpu.tmpl:#if defined(__CUDA)
UtilXlib/tests/test_mp_min_gpu.tmpl:PROGRAM test_mp_min_{vname}_gpu
UtilXlib/tests/test_mp_min_gpu.tmpl:    USE cudafor
UtilXlib/tests/test_mp_min_gpu.tmpl:    CALL save_random_seed("test_mp_min_{vname}_gpu", mpime)
UtilXlib/tests/test_mp_min_gpu.tmpl:END PROGRAM test_mp_min_{vname}_gpu
UtilXlib/tests/test_mp_min_gpu.tmpl:PROGRAM test_mp_min_{vname}_gpu
UtilXlib/tests/test_mp_min_gpu.tmpl:END PROGRAM test_mp_min_{vname}_gpu
UtilXlib/Makefile:mp_base_gpu.o \
UtilXlib/mp_base_gpu.f90:#define __BCAST_MSGSIZ_MAX_GPU huge(n)
UtilXlib/mp_base_gpu.f90:! These routines allocate buffer spaces used in reduce_base_real_gpu.
UtilXlib/mp_base_gpu.f90:#if defined (__CUDA)
UtilXlib/mp_base_gpu.f90:   SUBROUTINE allocate_buffers_gpu
UtilXlib/mp_base_gpu.f90:   END SUBROUTINE allocate_buffers_gpu
UtilXlib/mp_base_gpu.f90:   SUBROUTINE deallocate_buffers_gpu
UtilXlib/mp_base_gpu.f90:   END SUBROUTINE deallocate_buffers_gpu
UtilXlib/mp_base_gpu.f90:   SUBROUTINE bcast_real_gpu( array_d, n, root, gid )
UtilXlib/mp_base_gpu.f90:        USE cudafor
UtilXlib/mp_base_gpu.f90:        INTEGER :: msgsiz_max = __BCAST_MSGSIZ_MAX_GPU
UtilXlib/mp_base_gpu.f90:        write(*,*) 'BCAST_REAL_GPU IN'
UtilXlib/mp_base_gpu.f90:        ierr = cudaDeviceSynchronize()
UtilXlib/mp_base_gpu.f90:        write(*,*) 'BCAST_REAL_GPU OUT'
UtilXlib/mp_base_gpu.f90:   END SUBROUTINE bcast_real_gpu
UtilXlib/mp_base_gpu.f90:   SUBROUTINE bcast_integer_gpu( array_d, n, root, gid )
UtilXlib/mp_base_gpu.f90:        USE cudafor
UtilXlib/mp_base_gpu.f90:        write(*,*) 'BCAST_INTEGER_GPU IN'
UtilXlib/mp_base_gpu.f90:           IF( ierr /= 0 ) CALL errore( ' bcast_integer_gpu ', ' error in mpi_bcast 1 ', ierr )
UtilXlib/mp_base_gpu.f90:              IF( ierr /= 0 ) CALL errore( ' bcast_integer_gpu ', ' error in mpi_bcast 2 ', ierr )
UtilXlib/mp_base_gpu.f90:              IF( ierr /= 0 ) CALL errore( ' bcast_integer_gpu ', ' error in mpi_bcast 3 ', ierr )
UtilXlib/mp_base_gpu.f90:        ierr = cudaDeviceSynchronize()
UtilXlib/mp_base_gpu.f90:        write(*,*) 'BCAST_INTEGER_GPU OUT'
UtilXlib/mp_base_gpu.f90:   END SUBROUTINE bcast_integer_gpu
UtilXlib/mp_base_gpu.f90:   SUBROUTINE bcast_logical_gpu( array_d, n, root, gid )
UtilXlib/mp_base_gpu.f90:        USE cudafor
UtilXlib/mp_base_gpu.f90:        write(*,*) 'BCAST_LOGICAL_GPU IN'
UtilXlib/mp_base_gpu.f90:           IF( ierr /= 0 ) CALL errore( ' bcast_logical_gpu ', ' error in mpi_bcast 1 ', ierr )
UtilXlib/mp_base_gpu.f90:              IF( ierr /= 0 ) CALL errore( ' bcast_logical_gpu ', ' error in mpi_bcast 2 ', ierr )
UtilXlib/mp_base_gpu.f90:              IF( ierr /= 0 ) CALL errore( ' bcast_logical_gpu ', ' error in mpi_bcast 3 ', ierr )
UtilXlib/mp_base_gpu.f90:        ierr = cudaDeviceSynchronize()
UtilXlib/mp_base_gpu.f90:        write(*,*) 'BCAST_LOGICAL_GPU OUT'
UtilXlib/mp_base_gpu.f90:   END SUBROUTINE bcast_logical_gpu
UtilXlib/mp_base_gpu.f90:SUBROUTINE reduce_base_real_gpu( dim, ps_d, comm, root )
UtilXlib/mp_base_gpu.f90:  USE cudafor
UtilXlib/mp_base_gpu.f90:  write(*,*) 'reduce_base_real_gpu IN'
UtilXlib/mp_base_gpu.f90:     IF( info /= 0 ) CALL errore( 'reduce_base_real_gpu', 'error in mpi_reduce 1', info )
UtilXlib/mp_base_gpu.f90:     IF( info /= 0 ) CALL errore( 'reduce_base_real_gpu', 'error in mpi_allreduce 1', info )
UtilXlib/mp_base_gpu.f90:  info = cudaDeviceSynchronize()
UtilXlib/mp_base_gpu.f90:  write(*,*) 'reduce_base_real_gpu OUT'
UtilXlib/mp_base_gpu.f90:END SUBROUTINE reduce_base_real_gpu
UtilXlib/mp_base_gpu.f90:SUBROUTINE reduce_base_real_gpu( dim, ps_d, comm, root )
UtilXlib/mp_base_gpu.f90:  USE cudafor
UtilXlib/mp_base_gpu.f90:  write(*,*) 'reduce_base_real_gpu IN'
UtilXlib/mp_base_gpu.f90:  IF( info /= 0 ) CALL errore( 'reduce_base_real_gpu', 'error in mpi_comm_size', info )
UtilXlib/mp_base_gpu.f90:  IF( info /= 0 ) CALL errore( 'reduce_base_real_gpu', 'error in mpi_comm_rank', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_real_gpu', 'error in mpi_reduce 1', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_real_gpu', 'error in mpi_allreduce 1', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+(n-1)*maxb)), mp_buff_r_d(1), maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_real_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+(n-1)*maxb)), mp_buff_r_d(1), maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_real_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+nbuf*maxb)), mp_buff_r_d(1), dim-nbuf*maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_real_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+nbuf*maxb)), mp_buff_r_d(1), dim-nbuf*maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_real_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:  info = cudaDeviceSynchronize()
UtilXlib/mp_base_gpu.f90:  write(*,*) 'reduce_base_real_gpu OUT'
UtilXlib/mp_base_gpu.f90:END SUBROUTINE reduce_base_real_gpu
UtilXlib/mp_base_gpu.f90:SUBROUTINE reduce_base_integer_gpu( dim, ps_d, comm, root )
UtilXlib/mp_base_gpu.f90:  USE cudafor
UtilXlib/mp_base_gpu.f90:  write(*,*) 'reduce_base_integer_gpu IN'
UtilXlib/mp_base_gpu.f90:     IF( info /= 0 ) CALL errore( 'reduce_base_integer_gpu', 'error in mpi_reduce 1', info )
UtilXlib/mp_base_gpu.f90:     IF( info /= 0 ) CALL errore( 'reduce_base_integer_gpu', 'error in mpi_allreduce 1', info )
UtilXlib/mp_base_gpu.f90:  info = cudaDeviceSynchronize()
UtilXlib/mp_base_gpu.f90:  write(*,*) 'reduce_base_integer_gpu OUT'
UtilXlib/mp_base_gpu.f90:END SUBROUTINE reduce_base_integer_gpu
UtilXlib/mp_base_gpu.f90:SUBROUTINE reduce_base_integer_gpu( dim, ps_d, comm, root )
UtilXlib/mp_base_gpu.f90:  USE cudafor
UtilXlib/mp_base_gpu.f90:  write(*,*) 'reduce_base_integer_gpu IN'
UtilXlib/mp_base_gpu.f90:  IF( info /= 0 ) CALL errore( 'reduce_base_integer_gpu', 'error in mpi_comm_size', info )
UtilXlib/mp_base_gpu.f90:  IF( info /= 0 ) CALL errore( 'reduce_base_integer_gpu', 'error in mpi_comm_rank', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_integer_gpu', 'error in mpi_reduce 1', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_integer_gpu', 'error in mpi_allreduce 1', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+(n-1)*maxb)), mp_buff_i_d(1), maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_integer_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+(n-1)*maxb)), mp_buff_i_d(1), maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_integer_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_integer_gpu', 'error in mpi_reduce 2', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_integer_gpu', 'error in mpi_allreduce 2', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+nbuf*maxb)), mp_buff_i_d(1), dim-nbuf*maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_integer_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+nbuf*maxb)), mp_buff_i_d(1), dim-nbuf*maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_integer_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:  info = cudaDeviceSynchronize()
UtilXlib/mp_base_gpu.f90:  write(*,*) 'reduce_base_integer_gpu OUT'
UtilXlib/mp_base_gpu.f90:END SUBROUTINE reduce_base_integer_gpu
UtilXlib/mp_base_gpu.f90:SUBROUTINE reduce_base_real_to_gpu( dim, ps_d, psout_d, comm, root )
UtilXlib/mp_base_gpu.f90:  USE cudafor
UtilXlib/mp_base_gpu.f90:  write(*,*) 'reduce_base_real_to_gpu IN'
UtilXlib/mp_base_gpu.f90:  IF( info /= 0 ) CALL errore( 'reduce_base_real_to_gpu', 'error in mpi_comm_size', info )
UtilXlib/mp_base_gpu.f90:  IF( info /= 0 ) CALL errore( 'reduce_base_real_to_gpu', 'error in mpi_comm_rank', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_real_to_gpu', 'error in mpi_reduce 1', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_real_to_gpu', 'error in mpi_allreduce 1', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_real_to_gpu', 'error in mpi_reduce 2', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_real_to_gpu', 'error in mpi_allreduce 2', info )
UtilXlib/mp_base_gpu.f90:  info = cudaDeviceSynchronize()
UtilXlib/mp_base_gpu.f90:  write(*,*) 'reduce_base_real_to_gpu OUT'
UtilXlib/mp_base_gpu.f90:END SUBROUTINE reduce_base_real_to_gpu
UtilXlib/mp_base_gpu.f90:SUBROUTINE reduce_base_integer_to_gpu( dim, ps_d, psout_d, comm, root )
UtilXlib/mp_base_gpu.f90:  USE cudafor
UtilXlib/mp_base_gpu.f90:  write(*,*) 'reduce_base_integer_to_gpu IN'
UtilXlib/mp_base_gpu.f90:  IF( info /= 0 ) CALL errore( 'reduce_base_integer_to_gpu', 'error in mpi_comm_size', info )
UtilXlib/mp_base_gpu.f90:  IF( info /= 0 ) CALL errore( 'reduce_base_integer_to_gpu', 'error in mpi_comm_rank', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_integer_to_gpu', 'error in mpi_reduce 1', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_integer_to_gpu', 'error in mpi_allreduce 1', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_integer_to_gpu', 'error in mpi_reduce 2', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'reduce_base_integer_to_gpu', 'error in mpi_allreduce 2', info )
UtilXlib/mp_base_gpu.f90:  info = cudaDeviceSynchronize()
UtilXlib/mp_base_gpu.f90:  write(*,*) 'reduce_base_integer_to_gpu OUT'
UtilXlib/mp_base_gpu.f90:END SUBROUTINE reduce_base_integer_to_gpu
UtilXlib/mp_base_gpu.f90:SUBROUTINE parallel_min_integer_gpu( dim, ps_d, comm, root )
UtilXlib/mp_base_gpu.f90:  USE cudafor
UtilXlib/mp_base_gpu.f90:  write(*,*) 'parallel_min_integer_gpu IN'
UtilXlib/mp_base_gpu.f90:  IF( info /= 0 ) CALL errore( 'parallel_min_integer_gpu', 'error in mpi_comm_size', info )
UtilXlib/mp_base_gpu.f90:  IF( info /= 0 ) CALL errore( 'parallel_min_integer_gpu', 'error in mpi_comm_rank', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_min_integer_gpu', 'error in mpi_reduce 1', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_min_integer_gpu', 'error in mpi_allreduce 1', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d(1+(n-1)*maxb), buff(1), maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_min_integer_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+(n-1)*maxb)), buff(1), maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_min_integer_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_min_integer_gpu', 'error in mpi_reduce 2', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_min_integer_gpu', 'error in mpi_allreduce 2', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+nbuf*maxb)), buff(1), dim-nbuf*maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_min_integer_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+nbuf*maxb)), buff(1), dim-nbuf*maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_min_integer_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:  info = cudaDeviceSynchronize()
UtilXlib/mp_base_gpu.f90:  write(*,*) 'parallel_min_integer_gpu OUT'
UtilXlib/mp_base_gpu.f90:END SUBROUTINE parallel_min_integer_gpu
UtilXlib/mp_base_gpu.f90:SUBROUTINE parallel_max_integer_gpu( dim, ps_d, comm, root )
UtilXlib/mp_base_gpu.f90:  USE cudafor
UtilXlib/mp_base_gpu.f90:  write(*,*) 'parallel_max_integer_gpu IN'
UtilXlib/mp_base_gpu.f90:  IF( info /= 0 ) CALL errore( 'parallel_max_integer_gpu', 'error in mpi_comm_size', info )
UtilXlib/mp_base_gpu.f90:  IF( info /= 0 ) CALL errore( 'parallel_max_integer_gpu', 'error in mpi_comm_rank', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_integer_gpu', 'error in mpi_reduce 1', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_integer_gpu', 'error in mpi_allreduce 1', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d(1+(n-1)*maxb), buff(1), maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_integer_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+(n-1)*maxb)), buff(1), maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_integer_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_integer_gpu', 'error in mpi_reduce 2', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_integer_gpu', 'error in mpi_allreduce 2', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+nbuf*maxb)), buff(1), dim-nbuf*maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_integer_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+nbuf*maxb)), buff(1), dim-nbuf*maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_integer_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:  info = cudaDeviceSynchronize()
UtilXlib/mp_base_gpu.f90:  write(*,*) 'parallel_max_integer_gpu OUT'
UtilXlib/mp_base_gpu.f90:END SUBROUTINE parallel_max_integer_gpu
UtilXlib/mp_base_gpu.f90:SUBROUTINE parallel_min_real_gpu( dim, ps_d, comm, root )
UtilXlib/mp_base_gpu.f90:  USE cudafor
UtilXlib/mp_base_gpu.f90:  write(*,*) 'parallel_min_real_gpu IN'
UtilXlib/mp_base_gpu.f90:  IF( info /= 0 ) CALL errore( 'parallel_min_real_gpu', 'error in mpi_comm_size', info )
UtilXlib/mp_base_gpu.f90:  IF( info /= 0 ) CALL errore( 'parallel_min_real_gpu', 'error in mpi_comm_rank', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_min_real_gpu', 'error in mpi_reduce 1', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_min_real_gpu', 'error in mpi_allreduce 1', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d(1+(n-1)*maxb), buff(1), maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+(n-1)*maxb)), buff(1), maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_min_real_gpu', 'error in mpi_reduce 2', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_min_real_gpu', 'error in mpi_allreduce 2', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+nbuf*maxb)), buff(1), dim-nbuf*maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_integer_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+nbuf*maxb)), buff(1), dim-nbuf*maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_integer_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:  info = cudaDeviceSynchronize()
UtilXlib/mp_base_gpu.f90:  write(*,*) 'parallel_min_real_gpu OUT'
UtilXlib/mp_base_gpu.f90:END SUBROUTINE parallel_min_real_gpu
UtilXlib/mp_base_gpu.f90:SUBROUTINE parallel_max_real_gpu( dim, ps_d, comm, root )
UtilXlib/mp_base_gpu.f90:  USE cudafor
UtilXlib/mp_base_gpu.f90:  write(*,*) 'parallel_max_real_gpu IN'
UtilXlib/mp_base_gpu.f90:  IF( info /= 0 ) CALL errore( 'parallel_max_real_gpu', 'error in mpi_comm_size', info )
UtilXlib/mp_base_gpu.f90:  IF( info /= 0 ) CALL errore( 'parallel_max_real_gpu', 'error in mpi_comm_rank', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_real_gpu', 'error in mpi_reduce 1', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_real_gpu', 'error in mpi_allreduce 1', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d(1+(n-1)*maxb), buff(1), maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_real_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+(n-1)*maxb)), buff(1), maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_real_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_real_gpu', 'error in mpi_reduce 2', info )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_real_gpu', 'error in mpi_allreduce 2', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+nbuf*maxb)), buff(1), dim-nbuf*maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_real_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:        info = cudaMemcpy( ps_d((1+nbuf*maxb)), buff(1), dim-nbuf*maxb, cudaMemcpyDeviceToDevice )
UtilXlib/mp_base_gpu.f90:        IF( info /= 0 ) CALL errore( 'parallel_max_real_gpu', 'error in cudaMemcpy ', info )
UtilXlib/mp_base_gpu.f90:  info = cudaDeviceSynchronize()
UtilXlib/mp_base_gpu.f90:  write(*,*) 'parallel_max_real_gpu OUT'
UtilXlib/mp_base_gpu.f90:END SUBROUTINE parallel_max_real_gpu
UtilXlib/mp.f90:#if defined(__CUDA)
UtilXlib/mp.f90:  USE cudafor,        ONLY : cudamemcpy, cudamemcpy2d, &
UtilXlib/mp.f90:                            & cudaMemcpyDeviceToDevice, &
UtilXlib/mp.f90:                            & cudaDeviceSynchronize
UtilXlib/mp.f90:#if defined(__CUDA)
UtilXlib/mp.f90:    MODULE PROCEDURE mp_bcast_i1_gpu, mp_bcast_r1_gpu, mp_bcast_c1_gpu, &
UtilXlib/mp.f90:      !mp_bcast_z_gpu, mp_bcast_zv_gpu, &
UtilXlib/mp.f90:      mp_bcast_iv_gpu, mp_bcast_rv_gpu, mp_bcast_cv_gpu, mp_bcast_l_gpu, mp_bcast_rm_gpu, &
UtilXlib/mp.f90:      mp_bcast_cm_gpu, mp_bcast_im_gpu, mp_bcast_it_gpu, mp_bcast_i4d_gpu, mp_bcast_rt_gpu, mp_bcast_lv_gpu, &
UtilXlib/mp.f90:      mp_bcast_lm_gpu, mp_bcast_r4d_gpu, mp_bcast_r5d_gpu, mp_bcast_ct_gpu,  mp_bcast_c4d_gpu,&
UtilXlib/mp.f90:      mp_bcast_c5d_gpu, mp_bcast_c6d_gpu
UtilXlib/mp.f90:#if defined(__CUDA)
UtilXlib/mp.f90:    MODULE PROCEDURE  mp_sum_i1_gpu, mp_sum_iv_gpu, mp_sum_im_gpu, mp_sum_it_gpu, &
UtilXlib/mp.f90:      mp_sum_r1_gpu, mp_sum_rv_gpu, mp_sum_rm_gpu, mp_sum_rm_nc_gpu, mp_sum_rt_gpu, mp_sum_r4d_gpu, &
UtilXlib/mp.f90:      mp_sum_c1_gpu, mp_sum_cv_gpu, mp_sum_cm_gpu, mp_sum_cm_nc_gpu, mp_sum_ct_gpu, mp_sum_c4d_gpu, &
UtilXlib/mp.f90:      mp_sum_c5d_gpu, mp_sum_c6d_gpu, mp_sum_rmm_gpu, mp_sum_cmm_gpu, mp_sum_r5d_gpu, &
UtilXlib/mp.f90:      mp_sum_r6d_gpu
UtilXlib/mp.f90:#if defined(__CUDA)
UtilXlib/mp.f90:    MODULE PROCEDURE mp_root_sum_rm_gpu, mp_root_sum_cm_gpu
UtilXlib/mp.f90:#if defined(__CUDA)
UtilXlib/mp.f90:    MODULE PROCEDURE mp_get_r1_gpu, mp_get_rv_gpu, mp_get_cv_gpu, mp_get_i1_gpu, mp_get_iv_gpu, &
UtilXlib/mp.f90:      mp_get_rm_gpu, mp_get_cm_gpu
UtilXlib/mp.f90:#if defined(__CUDA)
UtilXlib/mp.f90:   MODULE PROCEDURE mp_put_rv_gpu, mp_put_cv_gpu, mp_put_i1_gpu, mp_put_iv_gpu, &
UtilXlib/mp.f90:     mp_put_rm_gpu
UtilXlib/mp.f90:#if defined(__CUDA)
UtilXlib/mp.f90:     MODULE PROCEDURE mp_max_i_gpu, mp_max_r_gpu, mp_max_rv_gpu, mp_max_iv_gpu
UtilXlib/mp.f90:#if defined(__CUDA)
UtilXlib/mp.f90:     MODULE PROCEDURE mp_min_i_gpu, mp_min_r_gpu, mp_min_rv_gpu, mp_min_iv_gpu
UtilXlib/mp.f90:#if defined(__CUDA)
UtilXlib/mp.f90:     MODULE PROCEDURE mp_gather_i1_gpu, mp_gather_iv_gpu, mp_gatherv_rv_gpu, mp_gatherv_iv_gpu, &
UtilXlib/mp.f90:       mp_gatherv_rm_gpu, mp_gatherv_im_gpu, mp_gatherv_cv_gpu, mp_gatherv_inplace_cplx_array_gpu
UtilXlib/mp.f90:#if defined(__CUDA)
UtilXlib/mp.f90:     MODULE PROCEDURE mp_allgatherv_inplace_cplx_array_gpu
UtilXlib/mp.f90:     MODULE PROCEDURE mp_allgatherv_inplace_real_array_gpu
UtilXlib/mp.f90:#if defined(__CUDA)
UtilXlib/mp.f90:     MODULE PROCEDURE mp_alltoall_c3d_gpu, mp_alltoall_i3d_gpu
UtilXlib/mp.f90:#if defined(__CUDA)
UtilXlib/mp.f90:     MODULE PROCEDURE mp_circular_shift_left_i0_gpu, &
UtilXlib/mp.f90:       mp_circular_shift_left_i1_gpu, &
UtilXlib/mp.f90:       mp_circular_shift_left_i2_gpu, &
UtilXlib/mp.f90:       mp_circular_shift_left_r2d_gpu, &
UtilXlib/mp.f90:       mp_circular_shift_left_c2d_gpu
UtilXlib/mp.f90:#if defined(__CUDA)
UtilXlib/mp.f90:     MODULE PROCEDURE mp_type_create_cplx_column_section_gpu
UtilXlib/mp.f90:     MODULE PROCEDURE mp_type_create_real_column_section_gpu
UtilXlib/mp.f90:#if defined(__CUDA)
UtilXlib/mp.f90:     MODULE PROCEDURE mp_type_create_cplx_row_section_gpu
UtilXlib/mp.f90:     MODULE PROCEDURE mp_type_create_real_row_section_gpu
UtilXlib/mp.f90:#if defined(__CUDA)
UtilXlib/mp.f90:        CALL allocate_buffers_gpu()
UtilXlib/mp.f90:#if defined(__CUDA)
UtilXlib/mp.f90:        CALL deallocate_buffers_gpu()
UtilXlib/mp.f90:#if defined(__CUDA)
UtilXlib/mp.f90:           CALL deallocate_buffers_gpu()
UtilXlib/mp.f90:!  GPU specific subroutines (Pietro Bonfa')
UtilXlib/mp.f90:! Before hacking on the CUDA part remember that:
UtilXlib/mp.f90:! 1. all mp_* interface should be blocking with respect to both MPI and CUDA.
UtilXlib/mp.f90:!    available on the GPU. However, the user is still free to change the buffer
UtilXlib/mp.f90:!    (https://devtalk.nvidia.com/default/topic/471866/cuda-programming-and-performance/host-device-memory-copies-up-to-64-kb-are-asynchronous/)
UtilXlib/mp.f90:! 4. GPU synchronization is always enforced even if no communication takes place.
UtilXlib/mp.f90:#ifdef __CUDA
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_i1_gpu(msg_d,source,gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_integer_gpu( msg_d, msglen, source, group )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_i1_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_iv_gpu(msg_d,source,gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_integer_gpu( msg_d, msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_iv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_im_gpu( msg_d, source, gid )
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_integer_gpu( msg_d, msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_im_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_it_gpu( msg_d, source, gid )
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_integer_gpu( msg_d, msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_it_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_i4d_gpu(msg_d, source, gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_integer_gpu( msg_d, msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_i4d_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_r1_gpu( msg_d, source, gid )
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_real_gpu( msg_d, msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_r1_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_rv_gpu(msg_d,source,gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_real_gpu( msg_d, msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_rv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_rm_gpu(msg_d,source,gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_real_gpu( msg_d, msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_rm_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_rt_gpu(msg_d,source,gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_real_gpu( msg_d, msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_rt_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_r4d_gpu(msg_d, source, gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_real_gpu( msg_d, msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_r4d_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_r5d_gpu(msg_d, source, gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_real_gpu( msg_d, msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_r5d_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_c1_gpu(msg_d,source,gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_real_gpu( msg_d, 2 * msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_c1_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_cv_gpu(msg_d,source,gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_real_gpu( msg_d, 2 * msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_cv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_cm_gpu(msg_d,source,gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_real_gpu( msg_d, 2 * msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_cm_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_ct_gpu(msg_d,source,gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_real_gpu( msg_d, 2 * msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_ct_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_c4d_gpu(msg_d,source,gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_real_gpu( msg_d, 2 * msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_c4d_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_c5d_gpu(msg_d,source,gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_real_gpu( msg_d, 2 * msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_c5d_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_c6d_gpu(msg_d,source,gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_real_gpu( msg_d, 2 * msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_c6d_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_l_gpu(msg_d,source,gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_logical_gpu( msg_d, msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_l_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_lv_gpu(msg_d,source,gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_logical_gpu( msg_d, msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_lv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_bcast_lm_gpu(msg_d,source,gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()      ! This syncs __GPU_MPI case
UtilXlib/mp.f90:        CALL bcast_logical_gpu( msg_d, msglen, source, gid )
UtilXlib/mp.f90:        RETURN ! Sync done by MPI call (or inside bcast_xxx_gpu)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_bcast_lm_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_get_i1_gpu(msg_dest_d, msg_sour_d, mpime, dest, sour, ip, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL and __GPU_MPI
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI and __GPU_MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_get_i1_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_get_iv_gpu(msg_dest_d, msg_sour_d, mpime, dest, sour, ip, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL and __GPU_MPI
UtilXlib/mp.f90:          ierr = cudaMemcpy(msg_dest_d(1) , msg_sour_d(1), SIZE(msg_sour_d), cudaMemcpyDeviceToDevice )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI and __GPU_MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_get_iv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_get_r1_gpu(msg_dest_d, msg_sour_d, mpime, dest, sour, ip, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL and __GPU_MPI
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI and __GPU_MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_get_r1_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_get_rv_gpu(msg_dest_d, msg_sour_d, mpime, dest, sour, ip, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL and __GPU_MPI
UtilXlib/mp.f90:          ierr = cudaMemcpy(msg_dest_d(1) , msg_sour_d(1), SIZE(msg_sour_d), cudaMemcpyDeviceToDevice )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI and __GPU_MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_get_rv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_get_rm_gpu(msg_dest_d, msg_sour_d, mpime, dest, sour, ip, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL and __GPU_MPI
UtilXlib/mp.f90:          ! function cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kdir)
UtilXlib/mp.f90:          ierr = cudaMemcpy2D(msg_dest_d, SIZE(msg_dest_d,1),&
UtilXlib/mp.f90:                              cudaMemcpyDeviceToDevice )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI and __GPU_MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_get_rm_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_get_cv_gpu(msg_dest_d, msg_sour_d, mpime, dest, sour, ip, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL and __GPU_MPI
UtilXlib/mp.f90:          ierr = cudaMemcpy(msg_dest_d(1) , msg_sour_d(1), SIZE(msg_sour_d), cudaMemcpyDeviceToDevice )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI and __GPU_MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_get_cv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_get_cm_gpu(msg_dest_d, msg_sour_d, mpime, dest, sour, ip, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL and __GPU_MPI
UtilXlib/mp.f90:          ierr = cudaMemcpy2D(msg_dest_d, SIZE(msg_dest_d,1),&
UtilXlib/mp.f90:                              cudaMemcpyDeviceToDevice )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI and __GPU_MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_get_cm_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_put_i1_gpu(msg_dest_d, msg_sour_d, mpime, sour, dest, ip, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL and __GPU_MPI
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI and __GPU_MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_put_i1_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_put_iv_gpu(msg_dest_d, msg_sour_d, mpime, sour, dest, ip, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL and __GPU_MPI
UtilXlib/mp.f90:          ierr = cudaMemcpy(msg_dest_d(1) , msg_sour_d(1), SIZE(msg_sour_d), cudaMemcpyDeviceToDevice )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI and __GPU_MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_put_iv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_put_rv_gpu(msg_dest_d, msg_sour_d, mpime, sour, dest, ip, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL and __GPU_MPI
UtilXlib/mp.f90:          ierr = cudaMemcpy(msg_dest_d(1) , msg_sour_d(1), SIZE(msg_sour_d), cudaMemcpyDeviceToDevice )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI and __GPU_MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_put_rv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_put_rm_gpu(msg_dest_d, msg_sour_d, mpime, sour, dest, ip, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL and __GPU_MPI
UtilXlib/mp.f90:          ierr = cudaMemcpy2D(msg_dest_d, SIZE(msg_dest_d,1),&
UtilXlib/mp.f90:                              cudaMemcpyDeviceToDevice )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI and __GPU_MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_put_rm_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_put_cv_gpu(msg_dest_d, msg_sour_d, mpime, sour, dest, ip, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL and __GPU_MPI
UtilXlib/mp.f90:          ierr = cudaMemcpy(msg_dest_d(1) , msg_sour_d(1), SIZE(msg_sour_d), cudaMemcpyDeviceToDevice )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI and __GPU_MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_put_cv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_i1_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_integer_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_i1_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_iv_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_integer_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_iv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_im_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_integer_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_im_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_it_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_integer_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_it_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_r1_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_r1_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_rv_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_rv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_rm_gpu(msg_d, gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_rm_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_rm_nc_gpu(msg_d, k1, k2, k3, k4, gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_gpu( msglen, msg_buff, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_rm_nc_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_root_sum_rm_gpu( msg_d, res_d, root, gid )
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()            ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_to_gpu( msglen, msg_d, res_d, gid, root )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_root_sum_rm_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_root_sum_cm_gpu( msg_d, res_d, root, gid )
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()            ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_to_gpu( 2 * msglen, msg_d, res_d, gid, root )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_root_sum_cm_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_rmm_gpu( msg_d, res_d, root, gid )
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()            ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_to_gpu( msglen, msg_d, res_d, group, root )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_rmm_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_rt_gpu( msg_d, gid )
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()            ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_rt_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_r4d_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()            ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_r4d_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_c1_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()            ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_gpu( 2 * msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_c1_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_cv_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()            ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_gpu( 2 * msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs the device after small message copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_cv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_cm_gpu(msg_d, gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()            ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_gpu( 2 * msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_cm_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_cm_nc_gpu(msg_d, k1, k2, k3, k4, gid)
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_gpu( 2 * msglen, msg_buff, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_cm_nc_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_cmm_gpu(msg_d, res_d, gid)
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()            ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_to_gpu( 2 * msglen, msg_d, res_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_cmm_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_ct_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()            ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_gpu( 2 * msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_ct_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_c4d_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()            ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_gpu( 2 * msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_c4d_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_c5d_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()            ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_gpu( 2 * msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_c5d_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_r5d_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()            ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_r5d_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_r6d_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()            ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_r6d_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_sum_c6d_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()            ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL reduce_base_real_gpu( 2 * msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_sum_c6d_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_max_i_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()            ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL parallel_max_integer_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_max_i_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_max_iv_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()            ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL parallel_max_integer_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_max_iv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_max_r_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL parallel_max_real_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_max_r_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_max_rv_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL parallel_max_real_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_max_rv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_min_i_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL parallel_min_integer_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_min_i_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_min_iv_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL parallel_min_integer_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_min_iv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_min_r_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL parallel_min_real_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_min_r_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_min_rv_gpu(msg_d,gid)
UtilXlib/mp.f90:          ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:#if  defined(__GPU_MPI)   
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:        CALL parallel_min_real_gpu( msglen, msg_d, gid, -1 )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()  ! This syncs __MPI for small copies
UtilXlib/mp.f90:      END SUBROUTINE mp_min_rv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_gather_i1_gpu(mydata_d, alldata_d, root, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:        ierr = cudaMemcpy( alldata_d(1), mydata_d, 1, &
UtilXlib/mp.f90:                                            & cudaMemcpyDeviceToDevice )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_gather_i1_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_gather_iv_gpu(mydata_d, alldata_d, root, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:        ierr = cudaMemcpy(alldata_d(:,1) , mydata_d(1), msglen, cudaMemcpyDeviceToDevice )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_gather_iv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_gatherv_rv_gpu( mydata_d, alldata_d, recvcount, displs, root, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:        ierr = cudaMemcpy(alldata_d(1) , mydata_d(1), recvcount( 1 ), cudaMemcpyDeviceToDevice )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_gatherv_rv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_gatherv_cv_gpu( mydata_d, alldata_d, recvcount, displs, root, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:        ierr = cudaMemcpy(alldata_d(1) , mydata_d(1), recvcount( 1 ), cudaMemcpyDeviceToDevice )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_gatherv_cv_gpu
UtilXlib/mp.f90:!..mp_gatherv_rv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_gatherv_iv_gpu( mydata_d, alldata_d, recvcount, displs, root, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:        ierr = cudaMemcpy(alldata_d(1) , mydata_d(1), recvcount( 1 ), cudaMemcpyDeviceToDevice )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_gatherv_iv_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_gatherv_rm_gpu( mydata_d, alldata_d, recvcount, displs, root, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:        ierr = cudaMemcpy2D(alldata_d, SIZE(alldata_d,1),&
UtilXlib/mp.f90:                              cudaMemcpyDeviceToDevice )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_gatherv_rm_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_gatherv_im_gpu( mydata_d, alldata_d, recvcount, displs, root, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:        ierr = cudaMemcpy2D(alldata_d, SIZE(alldata_d,1),&
UtilXlib/mp.f90:                              cudaMemcpyDeviceToDevice )
UtilXlib/mp.f90:        ierr = cudaDeviceSynchronize()   ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_gatherv_im_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_alltoall_c3d_gpu( sndbuf_d, rcvbuf_d, gid )
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_alltoall_c3d_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_alltoall_i3d_gpu( sndbuf_d, rcvbuf_d, gid )
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_alltoall_i3d_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_circular_shift_left_i0_gpu( buf_d, itag, gid )
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_circular_shift_left_i0_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_circular_shift_left_i1_gpu( buf_d, itag, gid )
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_circular_shift_left_i1_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_circular_shift_left_i2_gpu( buf_d, itag, gid )
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_circular_shift_left_i2_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_circular_shift_left_r2d_gpu( buf_d, itag, gid )
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_circular_shift_left_r2d_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_circular_shift_left_c2d_gpu( buf_d, itag, gid )
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_circular_shift_left_c2d_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_gatherv_inplace_cplx_array_gpu(alldata_d, my_column_type, recvcount, displs, root, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:           ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs in case of small data chunks
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs SERIAL
UtilXlib/mp.f90:      END SUBROUTINE mp_gatherv_inplace_cplx_array_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_allgatherv_inplace_cplx_array_gpu(alldata_d, my_element_type, recvcount, displs, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:           ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs in case of small data chunks
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_allgatherv_inplace_cplx_array_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_allgatherv_inplace_real_array_gpu(alldata_d, my_element_type, recvcount, displs, gid)
UtilXlib/mp.f90:#if ! defined(__GPU_MPI)
UtilXlib/mp.f90:           ierr = cudaDeviceSynchronize()
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs in case of small data chunks
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs __GPU_MPI
UtilXlib/mp.f90:         ierr = cudaDeviceSynchronize()   ! This syncs SERIAL, __MPI
UtilXlib/mp.f90:      END SUBROUTINE mp_allgatherv_inplace_real_array_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_type_create_cplx_column_section_gpu(dummy, start, length, stride, mytype)
UtilXlib/mp.f90:      END SUBROUTINE mp_type_create_cplx_column_section_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_type_create_real_column_section_gpu(dummy, start, length, stride, mytype)
UtilXlib/mp.f90:      END SUBROUTINE mp_type_create_real_column_section_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_type_create_cplx_row_section_gpu(dummy, column_start, column_stride, row_length, mytype)
UtilXlib/mp.f90:      END SUBROUTINE mp_type_create_cplx_row_section_gpu
UtilXlib/mp.f90:      SUBROUTINE mp_type_create_real_row_section_gpu(dummy, column_start, column_stride, row_length, mytype)
UtilXlib/mp.f90:      END SUBROUTINE mp_type_create_real_row_section_gpu
UtilXlib/README.md:* `__CUDA` : activates CUDA Fortran based interfaces.
UtilXlib/README.md:* `__GPU_MPI` : use CUDA aware MPI calls instead of standard sync-send-update method (experimental).
UtilXlib/README.md:If CUDA Fortran support is enabled, almost all interfaces accept input
UtilXlib/README.md:data declared with the `device` attribute. Note however that CUDA Fortran
UtilXlib/README.md:CUDA specific notes
UtilXlib/README.md:both MPI and CUDA streams. The code will synchronize the device before
UtilXlib/README.md:synchronization behaviour is overridden by the user (see `cudaStreamCreateWithFlags`).
UtilXlib/README.md:Be careful when using CUDA-aware MPI. Some implementations are not
UtilXlib/README.md:complete. The library will not check for the CUDA-aware MPI APIs during
UtilXlib/README.md:If you encounter problems when adding the flag `__GPU_MPI` it might
UtilXlib/README.md:be that the MPI library does not support some CUDA-aware APIs.
UtilXlib/CMakeLists.txt:    # GPU
UtilXlib/CMakeLists.txt:    mp_base_gpu.f90)
UtilXlib/CMakeLists.txt:qe_enable_cuda_fortran("${src_util}")
UtilXlib/CMakeLists.txt:qe_enable_cuda_fortran("${src_device_lapack}")
UtilXlib/CMakeLists.txt:        qe_openacc_fortran
UtilXlib/CMakeLists.txt:    qe_enable_cuda_fortran("${src_util_tests}")
UtilXlib/CMakeLists.txt:    if(QE_ENABLE_CUDA)
UtilXlib/CMakeLists.txt:            test_mp_bcast_i1_gpu
UtilXlib/CMakeLists.txt:            test_mp_bcast_iv_gpu
UtilXlib/CMakeLists.txt:            test_mp_bcast_im_gpu
UtilXlib/CMakeLists.txt:            test_mp_bcast_it_gpu
UtilXlib/CMakeLists.txt:            test_mp_bcast_i4d_gpu
UtilXlib/CMakeLists.txt:            test_mp_bcast_r4d_gpu
UtilXlib/CMakeLists.txt:            test_mp_bcast_c4d_gpu
UtilXlib/CMakeLists.txt:            test_mp_bcast_c5d_gpu
UtilXlib/CMakeLists.txt:            test_mp_bcast_r5d_gpu
UtilXlib/CMakeLists.txt:            test_mp_bcast_c6d_gpu
UtilXlib/CMakeLists.txt:            test_mp_bcast_iv_buffer_gpu
UtilXlib/CMakeLists.txt:            test_mp_bcast_lv_buffer_gpu
UtilXlib/CMakeLists.txt:            test_mp_bcast_rv_buffer_gpu
UtilXlib/CMakeLists.txt:            test_mp_max_iv_buffer_gpu
UtilXlib/CMakeLists.txt:            test_mp_max_rv_buffer_gpu
UtilXlib/CMakeLists.txt:            test_mp_min_iv_buffer_gpu
UtilXlib/CMakeLists.txt:            test_mp_min_rv_buffer_gpu
UtilXlib/CMakeLists.txt:            test_mp_sum_iv_buffer_gpu
UtilXlib/CMakeLists.txt:            test_mp_sum_rv_buffer_gpu)
UtilXlib/CMakeLists.txt:        qe_enable_cuda_fortran("${src_test}")
UtilXlib/CMakeLists.txt:            qe_openacc_fortran
UtilXlib/mp_base.f90:! These routines allocate buffer spaces used in reduce_base_real_gpu.
UtilXlib/clocks_handler.f90:#if defined(__CUDA)
UtilXlib/clocks_handler.f90:  USE cudafor
UtilXlib/clocks_handler.f90:  REAL(DP)          :: gputime(maxclock)
UtilXlib/clocks_handler.f90:  INTEGER           :: gpu_called(maxclock)
UtilXlib/clocks_handler.f90:#if defined(__CUDA)
UtilXlib/clocks_handler.f90:  type(cudaEvent) :: gpu_starts(maxclock), gpu_stops(maxclock)
UtilXlib/clocks_handler.f90:  INTEGER :: gpu_starts(maxclock), gpu_stops(maxclock)
UtilXlib/clocks_handler.f90:  ! ... GPU related timers
UtilXlib/clocks_handler.f90:  USE mytime, ONLY : gpu_starts, gpu_stops, gpu_called, gputime
UtilXlib/clocks_handler.f90:#if defined(__CUDA)
UtilXlib/clocks_handler.f90:  USE cudafor
UtilXlib/clocks_handler.f90:     gpu_called(n)  = 0
UtilXlib/clocks_handler.f90:     gputime(n)     = 0.0_DP
UtilXlib/clocks_handler.f90:#if defined(__CUDA)
UtilXlib/clocks_handler.f90:     ierr = cudaEventCreate(gpu_starts(n))
UtilXlib/clocks_handler.f90:     ierr = cudaEventCreate(gpu_stops(n))
UtilXlib/clocks_handler.f90:SUBROUTINE start_clock_gpu( label )
UtilXlib/clocks_handler.f90:                        gputime, gpu_starts, gpu_stops
UtilXlib/clocks_handler.f90:#if defined(__CUDA)
UtilXlib/clocks_handler.f90:  USE cudafor
UtilXlib/clocks_handler.f90:#if defined(__CUDA)
UtilXlib/clocks_handler.f90:           ierr = cudaEventRecord(gpu_starts(n),0)
UtilXlib/clocks_handler.f90:#if defined(__CUDA)
UtilXlib/clocks_handler.f90:     ierr = cudaEventRecord(gpu_starts(nclock),0)
UtilXlib/clocks_handler.f90:END SUBROUTINE start_clock_gpu
UtilXlib/clocks_handler.f90:SUBROUTINE stop_clock_gpu( label )
UtilXlib/clocks_handler.f90:                        gpu_called, gputime, gpu_starts, gpu_stops
UtilXlib/clocks_handler.f90:#if defined(__CUDA)
UtilXlib/clocks_handler.f90:  USE cudafor
UtilXlib/clocks_handler.f90:  ! ... initialize time used in CUDA APIs if __CUDA is present.
UtilXlib/clocks_handler.f90:#if defined(__CUDA)
UtilXlib/clocks_handler.f90:           ierr         = cudaEventRecord(gpu_stops(n),0)
UtilXlib/clocks_handler.f90:           ierr         = cudaEventSynchronize(gpu_stops(n))
UtilXlib/clocks_handler.f90:           ierr         = cudaEventElapsedTime(time, gpu_starts(n), gpu_stops(n))
UtilXlib/clocks_handler.f90:           gputime(n)   = gputime(n) + time
UtilXlib/clocks_handler.f90:           gpu_called(n)= gpu_called(n) + 1
UtilXlib/clocks_handler.f90:  WRITE( stdout, '("stop_clock_gpu: no clock for ",A12," found !")' ) label
UtilXlib/clocks_handler.f90:END SUBROUTINE stop_clock_gpu
UtilXlib/clocks_handler.f90:  USE mytime,     ONLY : nclock, clock_label, gpu_called
UtilXlib/clocks_handler.f90:  LOGICAL          :: print_gpu
UtilXlib/clocks_handler.f90:  print_gpu = ANY(gpu_called > 0)
UtilXlib/clocks_handler.f90:        IF(print_gpu) CALL print_this_clock_gpu( n )
UtilXlib/clocks_handler.f90:           IF(print_gpu) CALL print_this_clock_gpu( n )
UtilXlib/clocks_handler.f90:SUBROUTINE print_this_clock_gpu( n )
UtilXlib/clocks_handler.f90:  USE mytime,     ONLY : clock_label, gputime, called, gpu_called, mpi_per_thread                
UtilXlib/clocks_handler.f90:  REAL(DP) :: elapsed_gpu_time
UtilXlib/clocks_handler.f90:  nmax = gpu_called(n)
UtilXlib/clocks_handler.f90:  elapsed_gpu_time = gputime(n) / 1000.d0 ! GPU times are stored in ms
UtilXlib/clocks_handler.f90:        '(5X,A12," : ",F9.2,"s GPU "/)' ) &
UtilXlib/clocks_handler.f90:        clock_label(n), elapsed_gpu_time
UtilXlib/clocks_handler.f90:        '(35X,F9.2,"s GPU  (",I8," calls)")' ) &
UtilXlib/clocks_handler.f90:        elapsed_gpu_time, nmax
UtilXlib/clocks_handler.f90:END SUBROUTINE print_this_clock_gpu
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:    use cudafor
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:    use cudafor
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:    use cudafor
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:    use cudafor
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:    use cudafor
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:    use cudafor
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:    use cudafor
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:DOUBLE PRECISION FUNCTION MYDDOT_VECTOR_GPU(N,DX,DY)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:!$acc routine(MYDDOT_VECTOR_GPU) vector
UtilXlib/device_helper.f90:#if defined (__CUDA)
UtilXlib/device_helper.f90:    MYDDOT_VECTOR_GPU = 0.0d0 
UtilXlib/device_helper.f90:        MYDDOT_VECTOR_GPU = RES
UtilXlib/device_helper.f90:    MYDDOT_VECTOR_GPU = RES
UtilXlib/device_helper.f90:    MYDDOT_VECTOR_GPU = DDOT(N,DX,1,DY,1)
UtilXlib/device_helper.f90:END FUNCTION MYDDOT_VECTOR_GPU
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:USE cudafor
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:USE cudafor
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/device_helper.f90:#if defined(__CUDA)
UtilXlib/print_mem.f90:  call print_gpu_mem(out_unit)
UtilXlib/print_mem.f90:subroutine print_gpu_mem(out_unit)
UtilXlib/print_mem.f90:#if defined(__CUDA)
UtilXlib/print_mem.f90:  use cudafor
UtilXlib/print_mem.f90:  integer(kind=cuda_count_kind) :: freeMem, totalMem, usedMB, freeMB, totalMB
UtilXlib/print_mem.f90:  istat = CudaMemGetInfo(freeMem, totalMem)
UtilXlib/print_mem.f90:  write(out_unit,'(5X, "GPU memory used/free/total (MiB): ", I0, A3, I0,A3, I0)') &
UtilXlib/nvtx_wrapper.f90:!Copyright (c) 2019 maxcuda
UtilXlib/nvtx_wrapper.f90:!   https://github.com/maxcuda/NVTX_example     
UtilXlib/nvtx_wrapper.f90:#ifdef __CUDA
UtilXlib/nvtx_wrapper.f90:  use cudafor
UtilXlib/nvtx_wrapper.f90:#if defined(__CUDA) && defined(__SYNC_NVPROF)
UtilXlib/nvtx_wrapper.f90:    istat = cudaDeviceSynchronize()
UtilXlib/nvtx_wrapper.f90:#if defined(__CUDA) && defined(__SYNC_NVPROF)
UtilXlib/nvtx_wrapper.f90:    istat = cudaDeviceSynchronize()
test-suite/hp_soc_U_us_magn/benchmark.out.git.inp=bn.scf.in.args=2:                                        0.00s GPU  (       1 calls)
test-suite/hp_soc_U_us_magn/benchmark.out.git.inp=bn.scf.in.args=2:                                        0.00s GPU  (       1 calls)
test-suite/hp_soc_U_us_magn/benchmark.out.git.inp=bn.scf.in.args=1:                                        0.00s GPU  (       1 calls)
test-suite/hp_soc_U_us_magn/benchmark.out.git.inp=bn.scf.in.args=1:                                        0.00s GPU  (       1 calls)
test-suite/pw_vc-relax/benchmark.out.git.inp=vc-relax6.in:     Program PWSCF v.6.7GPU starts on  2Feb2021 at 13:38:49 
test-suite/pw_vc-relax/benchmark.out.git.inp=vc-relax5.in:     Program PWSCF v.6.7GPU starts on  2Feb2021 at 13:38:42 
test-suite/pw_vc-relax/benchmark.out.git.inp=vc-md2.in:     Program PWSCF v.6.7GPU starts on  2Feb2021 at 13:38: 0 
test-suite/pw_vc-relax/benchmark.out.git.inp=vc-relax2.in:     Program PWSCF v.6.7GPU starts on  2Feb2021 at 13:38:22 
test-suite/pw_vc-relax/benchmark.out.git.inp=vc-relax3.in:     Program PWSCF v.6.7GPU starts on  2Feb2021 at 13:38:27 
test-suite/pw_vc-relax/benchmark.out.git.inp=vc-relax4.in:     Program PWSCF v.6.7GPU starts on  2Feb2021 at 13:38:38 
test-suite/pw_vc-relax/benchmark.out.git.inp=vc-md1.in:     Program PWSCF v.6.7GPU starts on  4Feb2021 at 15:28: 7 
test-suite/QEHeat_h2o/benchmark.out.git.inp=all_currents.in:     Program PWSCF v.6.7GPU starts on 19May2021 at 22:20:51 
test-suite/QEHeat_h2o/benchmark.out.git.inp=all_currents_2pt.in:     Program PWSCF v.6.7GPU starts on 19May2021 at 22:21:35 
test-suite/kcw_wann-nc/benchmark.out.git.inp=Si_nc.kcw-screen.in.args=6:                                        0.00s GPU  (    1488 calls)
test-suite/kcw_wann-nc/benchmark.out.git.inp=Si_nc.kcw-screen.in.args=6:                                        0.00s GPU  (    1488 calls)
test-suite/pw_pawatom/benchmark.out.git.inp=paw-bfgs.in:     Program PWSCF v.6.7GPU starts on  4Feb2021 at 15:22:25 
test-suite/pw_pawatom/benchmark.out.git.inp=paw-vcbfgs.in:     Program PWSCF v.6.7GPU starts on  2Feb2021 at 13:34:51 
test-suite/ph_1d/benchmark.out.git.inp=ch4.ph.in.args=2:                                        0.00s GPU  (      51 calls)
test-suite/ph_1d/benchmark.out.git.inp=ch4.ph.in.args=2:                                        0.00s GPU  (      18 calls)
test-suite/ph_1d/benchmark.out.git.inp=ch4.ph.in.args=2:                                        0.00s GPU  (     601 calls)
test-suite/ph_1d/benchmark.out.git.inp=ch4.ph.in.args=2:                                        0.00s GPU  (     601 calls)
test-suite/pw_scf/benchmark.out.git.inp=scf-rmm-paro-k.in:     Program PWSCF v.6.7GPU starts on 15Jul2021 at 20:16:32 
test-suite/epw_qdpt/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (     117 calls)
test-suite/epw_qdpt/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (      24 calls)
test-suite/epw_qdpt/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (    1387 calls)
test-suite/epw_qdpt/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (    1387 calls)
test-suite/ctest_runner.sh:if [ ! -z "$CTEST_RESOURCE_GROUP_0_NVIDIA_GPUS" ] ; then
test-suite/ctest_runner.sh:  GPU_ID=`echo $CTEST_RESOURCE_GROUP_0_NVIDIA_GPUS | sed "s/id:\(.*\),.*$/\1/"`
test-suite/ctest_runner.sh:  echo "Assign GPU $GPU_ID to the run"
test-suite/ctest_runner.sh:  export CUDA_VISIBLE_DEVICES=$GPU_ID
test-suite/QEHeat_rotation/benchmark.out.git.inp=all_currents.in:     Program PWSCF v.6.7GPU starts on 19May2021 at 22:22: 8 
test-suite/QEHeat_rotation/benchmark.out.git.inp=all_currents_not_rotated.in:     Program PWSCF v.6.7GPU starts on 19May2021 at 22:22:15 
test-suite/QEHeat_rotation/benchmark.out.git.inp=all_currents_2pt.in:     Program PWSCF v.6.7GPU starts on 19May2021 at 22:22:12 
test-suite/QEHeat_rotation/benchmark.out.git.inp=all_currents_not_rotated_2pt.in:     Program PWSCF v.6.7GPU starts on 19May2021 at 22:22:20 
test-suite/epw_2D/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (     494 calls)
test-suite/epw_2D/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (      63 calls)
test-suite/epw_2D/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (    9642 calls)
test-suite/epw_2D/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (    9642 calls)
test-suite/QEHeat_translation/benchmark.out.git.inp=input_ar.in:     Program PWSCF v.6.7GPU starts on 19May2021 at 22:22:31 
test-suite/QEHeat_translation/benchmark.out.git.inp=input_2pt.in:     Program PWSCF v.6.7GPU starts on 19May2021 at 22:22:27 
test-suite/QEHeat_translation/benchmark.out.git.inp=input_ar_2pt.in:     Program PWSCF v.6.7GPU starts on 19May2021 at 22:22:37 
test-suite/QEHeat_translation/benchmark.out.git.inp=input.in:     Program PWSCF v.6.7GPU starts on 19May2021 at 22:22:24 
test-suite/kcw_ks/benchmark.out.git.inp=h2o.kcw-screen.in.args=16:                                        0.00s GPU  (      94 calls)
test-suite/kcw_ks/benchmark.out.git.inp=h2o.kcw-screen.in.args=16:                                        0.00s GPU  (      94 calls)
test-suite/hp_soc_U_nc_nonmagn/benchmark.out.git.inp=au.scf.in.args=2:                                        0.00s GPU  (       1 calls)
test-suite/hp_soc_U_nc_nonmagn/benchmark.out.git.inp=au.scf.in.args=1:                                        0.00s GPU  (       1 calls)
test-suite/pw_spinorbit/benchmark.out.git.inp=spinorbit-paw.in:     Program PWSCF v.6.7GPU starts on 28Jan2021 at 13:29:25 
test-suite/pw_workflow_vc-relax_scf/benchmark.out.git.inp=vc-relax-scf-2.in.args=2:     Program PWSCF v.6.7GPU starts on  6Feb2021 at  3: 6:26 
test-suite/ph_restart/benchmark.out.git.inp=SiC.phG.in.args=2:                                        0.00s GPU  (     276 calls)
test-suite/ph_restart/benchmark.out.git.inp=SiC.phG.in.args=2:                                        0.00s GPU  (      66 calls)
test-suite/ph_restart/benchmark.out.git.inp=SiC.phG.in.args=2:                                        0.00s GPU  (       6 calls)
test-suite/ph_restart/benchmark.out.git.inp=SiC.phG.in.args=2:                                        0.00s GPU  (      48 calls)
test-suite/ph_restart/benchmark.out.git.inp=SiC.phG.in.args=2:                                        0.00s GPU  (    2851 calls)
test-suite/ph_restart/benchmark.out.git.inp=SiC.phG.in.args=2:                                        0.00s GPU  (    2851 calls)
test-suite/ph_restart/benchmark.out.git.inp=SiC.phG.restart1.in.args=2:                                        0.00s GPU  (      60 calls)
test-suite/ph_restart/benchmark.out.git.inp=SiC.phG.restart1.in.args=2:                                        0.00s GPU  (       6 calls)
test-suite/ph_restart/benchmark.out.git.inp=SiC.phG.restart1.in.args=2:                                        0.00s GPU  (     832 calls)
test-suite/ph_restart/benchmark.out.git.inp=SiC.phG.restart1.in.args=2:                                        0.00s GPU  (     832 calls)
test-suite/cp_h2o_exx/benchmark.out.git.inp=h2o-mt-b3lyp-1.in:   /scratch/gpfs/mandrade/codes/devel/q-e-gpu/test-suite/cp_h2o_exx/pseudo/O.blyp-mt.UPF
test-suite/cp_h2o_exx/benchmark.out.git.inp=h2o-mt-b3lyp-1.in:   /scratch/gpfs/mandrade/codes/devel/q-e-gpu/test-suite/cp_h2o_exx/pseudo/H.blyp-vbc.UPF
test-suite/buildbot/Udine_farm/farmer_pgi1910_GPU.cfg:'LD_LIBRARY_PATH' : '/opt/pgi/linux86-64/2019/cuda/10.1/lib64:/opt/intel/composerxe/mkl/lib/intel64:/opt/intel/composerxe/lib/intel64:/opt/pgi/linux86-64/19.10/lib:/opt/pgi/linux86-64/19.10/mpi/openmpi-3.1.3/lib',
test-suite/buildbot/Udine_farm/farmer_pgi1910_GPU.cfg:'PATH' : '/opt/pgi/linux86-64/2019/cuda/10.1/bin:/opt/pgi/linux86-64/19.10/bin:/opt/pgi/linux86-64/19.10/mpi/openmpi-3.1.3/bin:/home/buildbot/bin:/usr/local/bin:/usr/bin:/bin',
test-suite/buildbot/Udine_farm/farmer_pgi1910_GPU.cfg:f=BuildFactory(Step.clean+Step.checkout_qe_GPU+Step.configure_qe_GPU+Step.make_pw_GPU+\
test-suite/buildbot/Udine_farm/worker.py:        'quantum_espresso_GPU': {
test-suite/buildbot/Udine_farm/worker.py:            'repository': 'https://gitlab.com/QEF/q-e-gpu.git',
test-suite/buildbot/Udine_farm/worker.py:            'branch': 'gpu-develop',
test-suite/buildbot/Udine_farm/worker.py:    self.checkout_qe_GPU = [steps.Git(
test-suite/buildbot/Udine_farm/worker.py:                 repourl=all_repos["quantum_espresso_GPU"]["repository"],
test-suite/buildbot/Udine_farm/worker.py:                 branch=all_repos["quantum_espresso_GPU"]["branch"],
test-suite/buildbot/Udine_farm/worker.py:    self.configure_qe_GPU = [ShellCommand(
test-suite/buildbot/Udine_farm/worker.py:                   name="configure_qe_GPU",
test-suite/buildbot/Udine_farm/worker.py:                   command=["./configure","--with-cuda=/opt/pgi/linux86-64/2019/cuda/10.1/","--with-cuda-runtime=10.1","--with-cuda-cc=60","--with-scalapack=no","--enable-openmp"],
test-suite/buildbot/Udine_farm/worker.py:                   haltOnFailure = True,descriptionDone=["configure_qe_GPU"]
test-suite/buildbot/Udine_farm/worker.py:    self.make_pw_GPU = [ShellCommand(
test-suite/buildbot/Udine_farm/worker.py:                   name="make_pw_GPU",
test-suite/buildbot/Udine_farm/master.cfg:            "farmer_gcc102_openmpi404_hdf5","farmer_intel20_impi","farmer_pgi1910_GPU","farmer_gcc102_openmpi404_libxc"]}
test-suite/epw_wfpt/benchmark.out.git.inp=ahc1.in.args=7:                                        0.00s GPU  (     144 calls)
test-suite/epw_wfpt/benchmark.out.git.inp=ahc1.in.args=7:                                        0.00s GPU  (    2559 calls)
test-suite/epw_wfpt/benchmark.out.git.inp=ahc1.in.args=7:                                        0.00s GPU  (    2559 calls)
test-suite/epw_wfpt/benchmark.out.git.inp=ahc2.in.args=7:                                        0.00s GPU  (     144 calls)
test-suite/epw_wfpt/benchmark.out.git.inp=ahc2.in.args=7:                                        0.00s GPU  (    3328 calls)
test-suite/epw_wfpt/benchmark.out.git.inp=ahc2.in.args=7:                                        0.00s GPU  (    3328 calls)
test-suite/epw_wfpt/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (     516 calls)
test-suite/epw_wfpt/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (     120 calls)
test-suite/epw_wfpt/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (    5951 calls)
test-suite/epw_wfpt/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (    5951 calls)
test-suite/epw_trev/benchmark.out.git.inp=scf.in.args=1:     Program PWSCF v.6.7GPU starts on  6Feb2021 at  4:44:35 
test-suite/epw_trev/benchmark.out.git.inp=nscf_epw.in.args=1:     Program PWSCF v.6.7GPU starts on  6Feb2021 at  4:45:29 
test-suite/epw_trev/benchmark.out.git.inp=scf_epw.in.args=1:     Program PWSCF v.6.7GPU starts on  6Feb2021 at  4:45:29 
test-suite/epw_trev/benchmark.out.git.inp=ph.in.args=2:     Program PHONON v.6.7GPU starts on  6Feb2021 at  4:44:36 
test-suite/CMakeLists.txt:if (QE_ENABLE_CUDA) 
test-suite/CMakeLists.txt:  list(APPEND CHECK_SKIP_MESSAGE "not ported to GPU")
test-suite/CMakeLists.txt:        # Each test occupies one GPU regardless of the number of MPI ranks.
test-suite/CMakeLists.txt:        set_tests_properties(${test_name} PROPERTIES RESOURCE_GROUPS "nvidia_gpus:1")
test-suite/pw_lda+U/benchmark.out.git.inp=lda+U+V-user_ns.in:     Program PWSCF v.6.7GPU starts on  4Feb2021 at 15:14:52 
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.pw.in.args=1:     GPU acceleration is ACTIVE.  4 visible GPUs per MPI rank
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.pw.in.args=1:                                        0.75s GPU  (      38 calls)
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.pw.in.args=1:                                        1.97s GPU  (      39 calls)
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.pw.in.args=1:                                        0.00s GPU  (      26 calls)
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.pw.in.args=1:                                        1.93s GPU  (      39 calls)
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.pw.in.args=1:                                        0.03s GPU  (      39 calls)
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.pw.in.args=1:                                        0.00s GPU  (      39 calls)
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.pw.in.args=1:                                        0.04s GPU  (      38 calls)
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.pw.in.args=1:                                        0.00s GPU  (      12 calls)
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.pw.in.args=1:                                        0.02s GPU  (     102 calls)
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.tddfpt.in.args=2:                                        6.04s GPU  (     200 calls)
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.tddfpt.in.args=2:                                        5.20s GPU  (     100 calls)
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.tddfpt.in.args=2:                                        0.84s GPU  (     100 calls)
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.tddfpt.in.args=2:                                        0.23s GPU  (     214 calls)
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.tddfpt.in.args=2:                                        0.14s GPU  (     100 calls)
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.tddfpt.in.args=2:                                        0.20s GPU  (     100 calls)
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.tddfpt.in.args=2:                                        0.01s GPU  (       3 calls)
test-suite/tddfpt_CH4/benchmark.out.git.inp=CH4.tddfpt.in.args=2:                                        0.10s GPU  (     830 calls)
test-suite/userconfig.tmp:skip_args = 'not present in this version\|libxc needed for this functional\|not ported to GPU'
test-suite/gpu-resource-example.json:  "_comment": "A node with 4 GPUs. Only digits are allowed in id which is passed as CUDA_VISIBLE_DEVICES",
test-suite/gpu-resource-example.json:      "nvidia_gpus": [
test-suite/pw_workflow_scf_dos/benchmark.out.git.inp=scf-dos-1.in.args=1:     Program PWSCF v.6.7GPU starts on  2Feb2021 at 13:39:55 
test-suite/pw_workflow_scf_dos/benchmark.out.git.inp=scf-dos-2.in.args=2:     Program PWSCF v.6.7GPU starts on  2Feb2021 at 13:39:57 
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.pw.in.args=1:     GPU acceleration is ACTIVE.
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.pw.in.args=1:     /m100_work/Sis22_baroni_0/obaseggi/q-e-openacc/test-suite/..//pseudo/Fe.pz-n-nc.UPF
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.pw.in.args=1:                                        1.21s GPU  (       1 calls)
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.pw.in.args=1:                                        1.39s GPU  (    1482 calls)
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.pw.in.args=1:                                       24.20s GPU  (    1509 calls)
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.pw.in.args=1:                                        0.01s GPU  (    1050 calls)
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.pw.in.args=1:                                        0.62s GPU  (    1509 calls)
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.pw.in.args=1:                                       23.33s GPU  (    1509 calls)
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.pw.in.args=1:                                        0.09s GPU  (    1509 calls)
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.pw.in.args=1:                                        0.25s GPU  (     147 calls)
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.pw.in.args=1:                                        0.01s GPU  (      64 calls)
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.pw.in.args=1:                                       18.17s GPU  (   83896 calls)
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.tddfpt-magnons.in.args=6:     GPU acceleration is ACTIVE.
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.tddfpt-magnons.in.args=6:     /m100_work/Sis22_baroni_0/obaseggi/q-e-openacc/test-suite/..//pseudo/Fe.pz-n-nc.UPF
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.tddfpt-magnons.in.args=6:     /m100_work/Sis22_baroni_0/obaseggi/q-e-openacc/test-suite/..//pseudo/Fe.pz-n-nc.UPF
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.tddfpt-magnons.in.args=6:                                      205.35s GPU  (   18064 calls)
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.tddfpt-magnons.in.args=6:                                        0.28s GPU  (      14 calls)
test-suite/tddfpt_magnons_fe/benchmark.out.git.inp=Fe.tddfpt-magnons.in.args=6:                                      236.36s GPU  ( 1195920 calls)
test-suite/pw_workflow_relax_relax/benchmark.out.git.inp=relax-2.in.args=2:     Program PWSCF v.6.7GPU starts on  2Feb2021 at 13:39:52 
test-suite/pw_vdw/benchmark.out.git.inp=beef.in:     Program PWSCF v.6.7GPU starts on 10Feb2021 at 10: 9:35 
test-suite/pw_vdw/benchmark.out.git.inp=vdw-d3.in:     Program PWSCF v.6.7GPU starts on  4Feb2021 at 15:36:30 
test-suite/pw_vdw/benchmark.out.git.inp=vdw-d2.in:     Program PWSCF v.6.7GPU starts on  4Feb2021 at 15:36:28 
test-suite/pw_vdw/benchmark.out.git.inp=vdw-mbd.in:     Program PWSCF v.6.7GPU starts on 26Mar2021 at 13:44:53 
test-suite/pw_vdw/benchmark.out.git.inp=rVV10.in:     Program PWSCF v.6.7GPU starts on  4Feb2021 at 15:36:20 
test-suite/pw_vdw/benchmark.out.git.inp=beef-spin.in:     Program PWSCF v.6.7GPU starts on 10Feb2021 at 10:25:13 
test-suite/pw_vdw/benchmark.out.git.inp=xdm.in:     Program PWSCF v.6.7GPU starts on  4Feb2021 at 15:36:44 
test-suite/epw_plrn/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (    5889 calls)
test-suite/epw_plrn/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (     972 calls)
test-suite/epw_plrn/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (   61451 calls)
test-suite/epw_plrn/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (   61451 calls)
test-suite/epw_super/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (   20382 calls)
test-suite/epw_super/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (    3420 calls)
test-suite/epw_super/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (  185481 calls)
test-suite/epw_super/benchmark.out.git.inp=ph.in.args=2:                                        0.00s GPU  (  185481 calls)
test-suite/hp_soc_UV_paw_magn/benchmark.out.git.inp=bn.scf.in.args=2:                                        0.00s GPU  (       1 calls)
test-suite/hp_soc_UV_paw_magn/benchmark.out.git.inp=bn.scf.in.args=2:                                        0.00s GPU  (       1 calls)
test-suite/hp_soc_UV_paw_magn/benchmark.out.git.inp=bn.scf.in.args=1:                                        0.00s GPU  (       1 calls)
test-suite/hp_soc_UV_paw_magn/benchmark.out.git.inp=bn.scf.in.args=1:                                        0.00s GPU  (       1 calls)
test-suite/epw_hall/benchmark.out.git.inp=scf.in.args=0:     Program PWSCF v.6.7GPU starts on 29Apr2021 at  9: 2:42 
test-suite/epw_hall/benchmark.out.git.inp=nscf.in.args=1:     Program PWSCF v.6.7GPU starts on 29Apr2021 at  9: 3:31 
test-suite/epw_hall/benchmark.out.git.inp=ph.in.args=2:     Program PHONON v.6.7GPU starts on 29Apr2021 at  9:59:13 
test-suite/pw_noncolin/benchmark.out.git.inp=noncolin-rmm.in:     Program PWSCF v.6.7GPU starts on 15Jul2021 at 20:32:55 
test-suite/cp_h2o_wf/benchmark.out.git.inp=h2o-nspin1.in:     Program CP v.6.7GPU starts on 29Jun2021 at 14:54:31 
test-suite/cp_h2o_wf/benchmark.out.git.inp=h2o-nspin2.in:     Program CP v.6.7GPU starts on 29Jun2021 at 14:54:35 
test-suite/ph_twochem/benchmark.out.git.inp=phG_twochem.in.args=11:                                        0.00s GPU  (     384 calls)
test-suite/ph_twochem/benchmark.out.git.inp=phG_twochem.in.args=11:                                        0.00s GPU  (      48 calls)
test-suite/ph_twochem/benchmark.out.git.inp=phG_twochem.in.args=11:                                        0.00s GPU  (    3068 calls)
test-suite/ph_twochem/benchmark.out.git.inp=phG_twochem.in.args=11:                                        0.00s GPU  (    3068 calls)
test-suite/cp_h2o_scan_libxc/benchmark.out.git.inp=nspin2.in:     Program CP v.6.7GPU starts on  8Jun2021 at 20:33: 5 
test-suite/cp_h2o_scan_libxc/benchmark.out.git.inp=nspin1.in:     Program CP v.6.7GPU starts on  8Jun2021 at 20:32:58 
PP/CMakeLists.txt:qe_enable_cuda_fortran("${src_pp}")
PP/CMakeLists.txt:qe_enable_cuda_fortran("${src_projwfc_x}")
PP/CMakeLists.txt:qe_enable_cuda_fortran("${src_pw2wannier90_x}")
.gitlab-ci.yml:  image: nvcr.io/nvidia/nvhpc:21.2-devel-cuda11.2-ubuntu20.04
.gitlab-ci.yml:    - ./configure FC=pgf90 F90=pgf90 F77=pgfortran MPIF90=mpif90 --enable-openmp --with-cuda=yes --enable-cuda-env-check=no --with-cuda-runtime=11.2  --with-cuda-cc=70
FFTXlib/tests/test_fwinv_gpu.f90:#if defined(__CUDA)
FFTXlib/tests/test_fwinv_gpu.f90:program test_fwinv_gpu
FFTXlib/tests/test_fwinv_gpu.f90:    CALL save_random_seed("test_fwinv_gpu", mp%me)
FFTXlib/tests/test_fwinv_gpu.f90:        CALL test_fwfft_gpu_1(mp, test, .true., i)
FFTXlib/tests/test_fwinv_gpu.f90:        CALL test_fwfft_gpu_1(mp, test, .false., i)
FFTXlib/tests/test_fwinv_gpu.f90:        CALL test_invfft_gpu_1(mp, test, .true., i)
FFTXlib/tests/test_fwinv_gpu.f90:        CALL test_invfft_gpu_1(mp, test, .false., i)
FFTXlib/tests/test_fwinv_gpu.f90:#if defined(__CUDA)
FFTXlib/tests/test_fwinv_gpu.f90:    ! the batched FFT is only implemented for GPU,
FFTXlib/tests/test_fwinv_gpu.f90:    CALL test_fwfft_many_gpu_1(mp, test, .true., 1)
FFTXlib/tests/test_fwinv_gpu.f90:    CALL test_fwfft_many_gpu_1(mp, test, .false., 1)
FFTXlib/tests/test_fwinv_gpu.f90:    CALL test_invfft_many_gpu_1(mp, test, .true., 1)
FFTXlib/tests/test_fwinv_gpu.f90:    CALL test_invfft_many_gpu_1(mp, test, .false., 1)
FFTXlib/tests/test_fwinv_gpu.f90:  SUBROUTINE test_fwfft_gpu_1(mp, test, gamma_only, ny)
FFTXlib/tests/test_fwinv_gpu.f90:    ! data from GPU is moved to an auxiliary array to compare the results of the GPU
FFTXlib/tests/test_fwinv_gpu.f90:  END SUBROUTINE test_fwfft_gpu_1
FFTXlib/tests/test_fwinv_gpu.f90:  SUBROUTINE test_invfft_gpu_1(mp, test, gamma_only, ny)
FFTXlib/tests/test_fwinv_gpu.f90:      ! copy to gpu and cleanup aux
FFTXlib/tests/test_fwinv_gpu.f90:    ! copy to gpu and cleanup aux
FFTXlib/tests/test_fwinv_gpu.f90:  END SUBROUTINE test_invfft_gpu_1
FFTXlib/tests/test_fwinv_gpu.f90:  SUBROUTINE test_fwfft_many_gpu_1(mp, test, gamma_only, ny)
FFTXlib/tests/test_fwinv_gpu.f90:  END SUBROUTINE test_fwfft_many_gpu_1
FFTXlib/tests/test_fwinv_gpu.f90:  SUBROUTINE test_invfft_many_gpu_1(mp, test, gamma_only, ny)
FFTXlib/tests/test_fwinv_gpu.f90:      ! copy to gpu and cleanup aux
FFTXlib/tests/test_fwinv_gpu.f90:    ! copy to gpu input data and cleanup aux
FFTXlib/tests/test_fwinv_gpu.f90:  END SUBROUTINE test_invfft_many_gpu_1
FFTXlib/tests/test_fwinv_gpu.f90:end program test_fwinv_gpu
FFTXlib/tests/test_fwinv_gpu.f90:SUBROUTINE stop_clock_gpu(label)
FFTXlib/tests/test_fwinv_gpu.f90:END SUBROUTINE stop_clock_gpu
FFTXlib/tests/test_fwinv_gpu.f90:SUBROUTINE start_clock_gpu(label)
FFTXlib/tests/test_fwinv_gpu.f90:END SUBROUTINE start_clock_gpu
FFTXlib/tests/Makefile:SRCS = test_fft_scalar_gpu.f90 \
FFTXlib/tests/Makefile:       test_fft_scatter_mod_gpu.f90 \
FFTXlib/tests/Makefile:       test_fwinv_gpu.f90
FFTXlib/tests/Makefile:	$(LD) $(LDFLAGS) -Mcudalib=cufft $< utils.o tester.o sort.o recips.o -o $@ ../src/libqefft.a $(FFT_LIBS) $(BLAS_LIBS) $(MPI_LIBS) $(LD_LIBS)
FFTXlib/tests/CMakeLists.txt:set(source_names fft_scalar_gpu fft_scatter_mod_gpu  fwinv_gpu)
FFTXlib/tests/CMakeLists.txt:    qe_enable_cuda_fortran("${TEST_SOURCE_FILE}")
FFTXlib/tests/test_fft_scalar_gpu.f90:#if defined(__CUDA)
FFTXlib/tests/test_fft_scalar_gpu.f90:program test_fft_scalar_gpu
FFTXlib/tests/test_fft_scalar_gpu.f90:    CALL save_random_seed("test_fft_scalar_gpu", 0)
FFTXlib/tests/test_fft_scalar_gpu.f90:    CALL test_cft_2xy_gpu(test)
FFTXlib/tests/test_fft_scalar_gpu.f90:    CALL test_cft_1z_gpu(test)
FFTXlib/tests/test_fft_scalar_gpu.f90:    CALL test_cfft3d_gpu(test)
FFTXlib/tests/test_fft_scalar_gpu.f90:    CALL test_cfft3ds_gpu(test)
FFTXlib/tests/test_fft_scalar_gpu.f90:    USE cudafor
FFTXlib/tests/test_fft_scalar_gpu.f90:  SUBROUTINE test_cft_1z_gpu(test)
FFTXlib/tests/test_fft_scalar_gpu.f90:    USE fft_scalar, ONLY : cft_1z_gpu, cft_2xy_gpu, cfft3d_gpu, cfft3ds_gpu
FFTXlib/tests/test_fft_scalar_gpu.f90:    USE cudafor
FFTXlib/tests/test_fft_scalar_gpu.f90:    integer(kind = cuda_stream_kind) :: stream = 0
FFTXlib/tests/test_fft_scalar_gpu.f90:    CALL cft_1z_gpu(c_d, nsl, nz, ldz, 1, cout_d, stream)
FFTXlib/tests/test_fft_scalar_gpu.f90:    ! Use c as auxiliary variable hosting GPU results
FFTXlib/tests/test_fft_scalar_gpu.f90:    CALL cft_1z_gpu(c_d, nsl, nz, ldz, -1, cout_d, stream)
FFTXlib/tests/test_fft_scalar_gpu.f90:    ! Use c as auxiliary variable hosting GPU results
FFTXlib/tests/test_fft_scalar_gpu.f90:    CALL cft_1z_gpu(c_d, nsl, nz, ldz, 1, cout_d, stream, in_place=.true.)
FFTXlib/tests/test_fft_scalar_gpu.f90:    ! Use c as auxiliary variable hosting GPU results
FFTXlib/tests/test_fft_scalar_gpu.f90:    CALL cft_1z_gpu(c_d, nsl, nz, ldz, -1, cout_d, stream, in_place=.true.)
FFTXlib/tests/test_fft_scalar_gpu.f90:    ! Use c as auxiliary variable hosting GPU results
FFTXlib/tests/test_fft_scalar_gpu.f90:  END SUBROUTINE test_cft_1z_gpu
FFTXlib/tests/test_fft_scalar_gpu.f90:  SUBROUTINE test_cft_2xy_gpu(test)
FFTXlib/tests/test_fft_scalar_gpu.f90:    USE fft_scalar, ONLY : cft_2xy_gpu
FFTXlib/tests/test_fft_scalar_gpu.f90:    USE cudafor
FFTXlib/tests/test_fft_scalar_gpu.f90:    integer(kind = cuda_stream_kind) :: stream = 0
FFTXlib/tests/test_fft_scalar_gpu.f90:    CALL cft_2xy_gpu(c_d, tmp_d, nzl, nx, ny, ldx, ldy, 1, stream)
FFTXlib/tests/test_fft_scalar_gpu.f90:    ! Use c as auxiliary variable hosting GPU results
FFTXlib/tests/test_fft_scalar_gpu.f90:    CALL cft_2xy_gpu(c_d, tmp_d, nzl, nx, ny, ldx, ldy, -1, stream)
FFTXlib/tests/test_fft_scalar_gpu.f90:    ! Use c as auxiliary variable hosting GPU results
FFTXlib/tests/test_fft_scalar_gpu.f90:  END SUBROUTINE test_cft_2xy_gpu
FFTXlib/tests/test_fft_scalar_gpu.f90:  SUBROUTINE test_cfft3d_gpu(test)
FFTXlib/tests/test_fft_scalar_gpu.f90:    USE fft_scalar, ONLY : cfft3d_gpu
FFTXlib/tests/test_fft_scalar_gpu.f90:    USE cudafor
FFTXlib/tests/test_fft_scalar_gpu.f90:    integer(kind = cuda_stream_kind) :: stream = 0
FFTXlib/tests/test_fft_scalar_gpu.f90:    CALL cfft3d_gpu( c_d, nx, ny, nz, ldx, ldy, ldz, howmany, 1, stream )
FFTXlib/tests/test_fft_scalar_gpu.f90:    ! Use c as auxiliary variable hosting GPU results
FFTXlib/tests/test_fft_scalar_gpu.f90:    CALL cfft3d_gpu( c_d, nx, ny, nz, ldx, ldy, ldz, howmany, -1, stream )
FFTXlib/tests/test_fft_scalar_gpu.f90:    ! Use c as auxiliary variable hosting GPU results
FFTXlib/tests/test_fft_scalar_gpu.f90:  END SUBROUTINE test_cfft3d_gpu
FFTXlib/tests/test_fft_scalar_gpu.f90:  SUBROUTINE test_cfft3ds_gpu(test)
FFTXlib/tests/test_fft_scalar_gpu.f90:    USE fft_scalar, ONLY : cfft3ds_gpu
FFTXlib/tests/test_fft_scalar_gpu.f90:    USE cudafor
FFTXlib/tests/test_fft_scalar_gpu.f90:    integer(kind = cuda_stream_kind) :: stream = 0
FFTXlib/tests/test_fft_scalar_gpu.f90:    CALL cfft3ds_gpu( c_d, nx, ny, nz, ldx, ldy, ldz, howmany, 1, do_fft_z, do_fft_y, stream)
FFTXlib/tests/test_fft_scalar_gpu.f90:    ! Use c as auxiliary variable hosting GPU results
FFTXlib/tests/test_fft_scalar_gpu.f90:    CALL cfft3ds_gpu( c_d, nx, ny, nz, ldx, ldy, ldz, howmany, -1, do_fft_z, do_fft_y, stream )
FFTXlib/tests/test_fft_scalar_gpu.f90:    ! Use c as auxiliary variable hosting GPU results
FFTXlib/tests/test_fft_scalar_gpu.f90:  END SUBROUTINE test_cfft3ds_gpu
FFTXlib/tests/test_fft_scalar_gpu.f90:end program test_fft_scalar_gpu
FFTXlib/tests/test_fft_scalar_gpu.f90:program test_fft_scalar_gpu
FFTXlib/tests/test_fft_scalar_gpu.f90:end program test_fft_scalar_gpu
FFTXlib/tests/test_fft_scalar_gpu.f90:SUBROUTINE stop_clock_gpu(label)
FFTXlib/tests/test_fft_scalar_gpu.f90:END SUBROUTINE stop_clock_gpu
FFTXlib/tests/test_fft_scalar_gpu.f90:SUBROUTINE start_clock_gpu(label)
FFTXlib/tests/test_fft_scalar_gpu.f90:END SUBROUTINE start_clock_gpu
FFTXlib/tests/fft_test.f90:subroutine start_clock_gpu(label)
FFTXlib/tests/fft_test.f90:subroutine stop_clock_gpu(label)
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:#if defined(__CUDA)
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:program test_fft_scatter_mod_gpu
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:      CALL save_random_seed("test_fft_scatter_mod_gpu", mp%me)
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:          CALL test_fft_scatter_xy_gpu_1(mp, test, .true., i)
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:          CALL test_fft_scatter_xy_gpu_1(mp, test, .false., i)
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:          CALL test_fft_scatter_yz_gpu_1(mp, test, .true., i)
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:          CALL test_fft_scatter_yz_gpu_1(mp, test, .false., i)
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:      CALL test_fft_scatter_many_yz_gpu_1(mp, test, .true., 1)
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:      CALL test_fft_scatter_many_yz_gpu_1(mp, test, .false., 1)
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:    USE cudafor
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:  SUBROUTINE test_fft_scatter_xy_gpu_1(mp, test, gamma_only, ny)
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:    USE cudafor
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:    USE fft_scatter_gpu, ONLY : fft_scatter_xy_gpu
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:    integer(kind = cuda_stream_kind) :: stream = 0
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:    CALL fft_scatter_xy_gpu( dfft, scatter_in_d, scatter_out_d, vsiz, fft_sign, stream )
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:    CALL fft_scatter_xy_gpu( dfft, scatter_out_d, scatter_in_d, vsiz, -1*fft_sign, stream )
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:  END SUBROUTINE test_fft_scatter_xy_gpu_1
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:  SUBROUTINE test_fft_scatter_yz_gpu_1(mp, test, gamma_only, ny)
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:    USE cudafor
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:    USE fft_scatter_gpu, ONLY : fft_scatter_yz_gpu
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:    integer(kind = cuda_stream_kind) :: stream = 0
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:    CALL fft_scatter_yz_gpu( dfft, scatter_in_d, scatter_out_d, vsiz, fft_sign )
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:    CALL fft_scatter_yz_gpu( dfft, scatter_out_d, scatter_in_d, vsiz, -1*fft_sign )
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:  END SUBROUTINE test_fft_scatter_yz_gpu_1
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:  SUBROUTINE test_fft_scatter_many_yz_gpu_1(mp, test, gamma_only, ny)
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:    USE cudafor
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:    USE fft_scatter_gpu, ONLY : fft_scatter_yz_gpu, fft_scatter_many_yz_gpu
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:    !integer(kind = cuda_stream_kind) :: streams(5)
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:       CALL fft_scatter_yz_gpu( dfft, scatter_in_d(start_in:end_in), scatter_out_d(start_in:end_out), dfft%nnr, 2 )
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:    CALL fft_scatter_many_yz_gpu ( dfft, scatter_in_d, scatter_out_d, vsiz, 2, howmany )
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:       CALL fft_scatter_yz_gpu( dfft, scatter_out_d(start_out:end_out), scatter_in_d(start_in:end_in), dfft%nnr, -2 )
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:    CALL fft_scatter_many_yz_gpu ( dfft, scatter_out_d, scatter_in_d, vsiz, -2, howmany )
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:       ! Extract data from GPU. Data are spaced by nstick_zx*n3x
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:  END SUBROUTINE test_fft_scatter_many_yz_gpu_1
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:end program test_fft_scatter_mod_gpu
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:program test_fft_scatter_mod_gpu
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:end program test_fft_scatter_mod_gpu
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:SUBROUTINE stop_clock_gpu(label)
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:END SUBROUTINE stop_clock_gpu
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:SUBROUTINE start_clock_gpu(label)
FFTXlib/tests/test_fft_scatter_mod_gpu.f90:END SUBROUTINE start_clock_gpu
FFTXlib/src/fft_scatter_gpu.f90:! Rewritten by Stefano de Gironcoli, ported to GPU by Pietro Bonfa'
FFTXlib/src/fft_scatter_gpu.f90:#if defined(__CUDA)
FFTXlib/src/fft_scatter_gpu.f90:   MODULE fft_scatter_gpu
FFTXlib/src/fft_scatter_gpu.f90:#if defined(__CUDA)
FFTXlib/src/fft_scatter_gpu.f90:        PUBLIC :: fft_scatter_xy_gpu, fft_scatter_yz_gpu, fft_scatter_tg_gpu, &
FFTXlib/src/fft_scatter_gpu.f90:                  fft_scatter_tg_opt_gpu, fft_scatter_many_yz_gpu
FFTXlib/src/fft_scatter_gpu.f90:SUBROUTINE fft_scatter_xy_gpu ( desc, f_in_d, f_aux_d, nxx_, isgn, stream )
FFTXlib/src/fft_scatter_gpu.f90:  USE cudafor
FFTXlib/src/fft_scatter_gpu.f90:  INTEGER(kind=cuda_stream_kind), INTENT(IN) :: stream ! cuda stream for the execution
FFTXlib/src/fft_scatter_gpu.f90:  !CALL nvtxStartRangeAsync("fft_scatter_xy_gpu", isgn + 5)
FFTXlib/src/fft_scatter_gpu.f90:  ! `desc` appear in the body of the do loops, the compiler generates incorrect GPU code.
FFTXlib/src/fft_scatter_gpu.f90:        ! The two loops below are performed by a single call to cudaMemcpy2DAsync
FFTXlib/src/fft_scatter_gpu.f90:        ! that also moves data from the GPU to the CPU.
FFTXlib/src/fft_scatter_gpu.f90:        ! Commented code shows how to implement it without CUDA specific APIs.
FFTXlib/src/fft_scatter_gpu.f90:        ierr = cudaMemcpy2DAsync( f_aux(kdest + 1), nr2px, f_in_d(kfrom + 1 ), desc%nr2x, desc%nr2p( iproc2 ), ncp_(me2), cudaMemcpyDeviceToHost, stream )
FFTXlib/src/fft_scatter_gpu.f90:     ierr = cudaStreamSynchronize(stream)
FFTXlib/src/fft_scatter_gpu.f90:     ierr = cudaStreamSynchronize(stream)
FFTXlib/src/fft_scatter_gpu.f90:     ierr = cudaMemcpyAsync( f_in_d, f_in, nxx_, cudaMemcpyHostToDevice, stream )
FFTXlib/src/fft_scatter_gpu.f90:           ierr = cudaMemcpyAsync( f_in_d(kdest + 1), f_in(kdest + 1 ), sendsize, cudaMemcpyHostToDevice, stream )
FFTXlib/src/fft_scatter_gpu.f90:          ierr = cudaMemcpyAsync( f_in(kdest + 1), f_in_d(kdest + 1 ), sendsize, cudaMemcpyDeviceToHost, stream )
FFTXlib/src/fft_scatter_gpu.f90:        ierr = cudaStreamSynchronize(stream)
FFTXlib/src/fft_scatter_gpu.f90:     ierr = cudaMemcpy( f_in, f_in_d, nxx_, cudaMemcpyDeviceToHost)
FFTXlib/src/fft_scatter_gpu.f90:        ierr = cudaMemcpy2DAsync( f_in_d(kfrom +1 ), desc%nr2x, f_aux(kdest + 1), nr2px, desc%nr2p( iproc2 ), ncp_(me2), cudaMemcpyHostToDevice, stream )
FFTXlib/src/fft_scatter_gpu.f90:END SUBROUTINE fft_scatter_xy_gpu
FFTXlib/src/fft_scatter_gpu.f90:SUBROUTINE fft_scatter_yz_gpu ( desc, f_in_d, f_aux_d, nxx_, isgn )
FFTXlib/src/fft_scatter_gpu.f90:  USE cudafor
FFTXlib/src/fft_scatter_gpu.f90:!  INTEGER(kind=cuda_stream_kind), INTENT(IN) :: stream ! cuda stream for the execution
FFTXlib/src/fft_scatter_gpu.f90:  TYPE(cudaEvent) :: zero_event
FFTXlib/src/fft_scatter_gpu.f90:  !CALL nvtxStartRangeAsync("fft_scatter_yz_gpu", isgn + 5)
FFTXlib/src/fft_scatter_gpu.f90:  ierr = cudaEventCreate( zero_event )
FFTXlib/src/fft_scatter_gpu.f90:  ! `desc` appear in the body of the do loops, the compiler generates incorrect GPU code.
FFTXlib/src/fft_scatter_gpu.f90:           ierr = cudaMemcpy2DAsync( f_aux(kdest + 1), nr3px, f_in_d(kfrom + 1 ), nr3x, desc%nr3p( iproc3 ), ncp_(ip), cudaMemcpyDeviceToHost, desc%stream_scatter_yz(iproc3) )
FFTXlib/src/fft_scatter_gpu.f90:        ierr = cudaStreamSynchronize(desc%stream_scatter_yz(iproc3))
FFTXlib/src/fft_scatter_gpu.f90:     ierr = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_gpu.f90:     ierr = cudaMemcpy( f_in_d, f_in, nxx_, cudaMemcpyHostToDevice )
FFTXlib/src/fft_scatter_gpu.f90:     ierr = cudaEventRecord ( zero_event, desc%stream_scatter_yz(1) )
FFTXlib/src/fft_scatter_gpu.f90:           ierr = cudaMemcpyAsync( f_in_d(it0+1), f_in(it0+1), sendsize, cudaMemcpyHostToDevice, desc%stream_scatter_yz(iproc3)  )
FFTXlib/src/fft_scatter_gpu.f90:        IF (iproc3 == 2) ierr = cudaEventSynchronize( zero_event )
FFTXlib/src/fft_scatter_gpu.f90:           ierr = cudaMemcpyAsync( f_in(kdest + 1), f_in_d(kdest + 1 ), sendsize, cudaMemcpyDeviceToHost, desc%stream_scatter_yz(iproc3) )
FFTXlib/src/fft_scatter_gpu.f90:           ierr = cudaStreamSynchronize(desc%stream_scatter_yz(iproc3))
FFTXlib/src/fft_scatter_gpu.f90:     !ierr = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_gpu.f90:     ierr = cudaMemcpy( f_in, f_in_d, nxx_, cudaMemcpyDeviceToHost )
FFTXlib/src/fft_scatter_gpu.f90:          ierr = cudaMemcpy2DAsync( f_in_d(kfrom +1 ), nr3x, f_aux(kdest + 1), nr3px, desc%nr3p( iproc3 ), ncp_ (ip), cudaMemcpyHostToDevice, desc%stream_scatter_yz(iproc3) )
FFTXlib/src/fft_scatter_gpu.f90:        ierr = cudaStreamSynchronize(desc%stream_scatter_yz(iproc3))
FFTXlib/src/fft_scatter_gpu.f90:END SUBROUTINE fft_scatter_yz_gpu
FFTXlib/src/fft_scatter_gpu.f90:SUBROUTINE fft_scatter_tg_gpu ( desc, f_in_d, f_aux_d, nxx_, isgn, stream )
FFTXlib/src/fft_scatter_gpu.f90:  USE cudafor
FFTXlib/src/fft_scatter_gpu.f90:  INTEGER(kind=cuda_stream_kind), INTENT(IN) :: stream ! cuda stream for the execution
FFTXlib/src/fft_scatter_gpu.f90:  ! or possibly use GPU MPI?
FFTXlib/src/fft_scatter_gpu.f90:  ierr = cudaMemcpyAsync( f_aux, f_in_d, nxx_, cudaMemcpyDeviceToHost, stream )
FFTXlib/src/fft_scatter_gpu.f90:  ierr = cudaStreamSynchronize(stream)
FFTXlib/src/fft_scatter_gpu.f90:  ierr = cudaMemcpyAsync( f_in_d, f_in, nxx_, cudaMemcpyHostToDevice, stream )
FFTXlib/src/fft_scatter_gpu.f90:  !ierr = cudaStreamSynchronize(stream)
FFTXlib/src/fft_scatter_gpu.f90:END SUBROUTINE fft_scatter_tg_gpu
FFTXlib/src/fft_scatter_gpu.f90:SUBROUTINE fft_scatter_tg_opt_gpu ( desc, f_in_d, f_out_d, nxx_, isgn, stream )
FFTXlib/src/fft_scatter_gpu.f90:  USE cudafor
FFTXlib/src/fft_scatter_gpu.f90:  INTEGER(kind=cuda_stream_kind), INTENT(IN) :: stream ! cuda stream for the execution
FFTXlib/src/fft_scatter_gpu.f90:  ierr = cudaMemcpyAsync( f_in, f_in_d, nxx_, cudaMemcpyDeviceToHost, stream )
FFTXlib/src/fft_scatter_gpu.f90:  ierr = cudaStreamSynchronize(stream)
FFTXlib/src/fft_scatter_gpu.f90:  ierr = cudaMemcpyAsync( f_out_d, f_out, nxx_, cudaMemcpyHostToDevice, stream )
FFTXlib/src/fft_scatter_gpu.f90:  !ierr = cudaStreamSynchronize(stream)
FFTXlib/src/fft_scatter_gpu.f90:END SUBROUTINE fft_scatter_tg_opt_gpu
FFTXlib/src/fft_scatter_gpu.f90:SUBROUTINE fft_scatter_many_yz_gpu ( desc, f_in_d, f_aux_d, nxx_, isgn, howmany )
FFTXlib/src/fft_scatter_gpu.f90:  USE cudafor
FFTXlib/src/fft_scatter_gpu.f90:  !CALL nvtxStartRangeAsync("fft_scatter_many_yz_gpu", isgn + 5)
FFTXlib/src/fft_scatter_gpu.f90:  ! `desc` appear in the body of the do loops, the compiler generates incorrect GPU code.
FFTXlib/src/fft_scatter_gpu.f90:        ierr = cudaMemcpy2D( f_aux(kdest + 1), nr3px, f_in_d(kfrom + 1 ), nr3x, desc%nr3p( iproc3 ), howmany*ncpx, cudaMemcpyDeviceToHost)
FFTXlib/src/fft_scatter_gpu.f90:     ierr = cudaMemcpy( f_in_d, f_in, nxx_, cudaMemcpyHostToDevice)
FFTXlib/src/fft_scatter_gpu.f90:     ierr = cudaMemcpy( f_in, f_in_d, nxx_, cudaMemcpyDeviceToHost )
FFTXlib/src/fft_scatter_gpu.f90:        ierr = cudaMemcpy2D( f_in_d(kfrom +1 ), nr3x, f_aux(kdest + 1), nr3px, desc%nr3p( iproc3 ), howmany * ncpx, cudaMemcpyHostToDevice)
FFTXlib/src/fft_scatter_gpu.f90:END SUBROUTINE fft_scatter_many_yz_gpu
FFTXlib/src/fft_scatter_gpu.f90:END MODULE fft_scatter_gpu
FFTXlib/src/fft_scatter_gpu.f90:! defined (__CUDA)
FFTXlib/src/fft_scalar.cuFFT.f90:#ifdef __CUDA
FFTXlib/src/fft_scalar.cuFFT.f90:       USE cudafor
FFTXlib/src/fft_scalar.cuFFT.f90:       PUBLIC :: cft_1z_gpu, cft_2xy_gpu, cfft3d_gpu, cfft3ds_gpu
FFTXlib/src/fft_scalar.cuFFT.f90:   SUBROUTINE cft_1z_gpu(c_d, nsl, nz, ldz, isign, cout_d, stream, in_place)
FFTXlib/src/fft_scalar.cuFFT.f90:!     ### GPU VERSION IN PLACE!!! #### output : cout_d(ldz*nsl) (complex - NOTA BENE: transform is not in-place!)
FFTXlib/src/fft_scalar.cuFFT.f90:     INTEGER(kind = cuda_stream_kind) :: stream
FFTXlib/src/fft_scalar.cuFFT.f90:     call fftx_error__(" fft_scalar_cuFFT: cft_1z_gpu ", " failed to set stream ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:     CALL start_clock( 'GPU_cft_1z' )
FFTXlib/src/fft_scalar.cuFFT.f90:        call fftx_error__(" fft_scalar_cuFFT: cft_1z_gpu ", " cufftExecZ2Z failed ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:           call fftx_error__(" fft_scalar_cuFFT: cft_1z_gpu ", " cufftExecZ2Z failed ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:           call fftx_error__(" fft_scalar_cuFFT: cft_1z_gpu ", " cufftExecZ2Z failed ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:     CALL stop_clock( 'GPU_cft_1z' )
FFTXlib/src/fft_scalar.cuFFT.f90:           call fftx_error__(" fft_scalar_cuFFT: cft_1z_gpu ", " cufftDestroy failed ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:       call fftx_error__(" fft_scalar_cuFFT: cft_1z_gpu ", " cufftPlanMany failed ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:   END SUBROUTINE cft_1z_gpu
FFTXlib/src/fft_scalar.cuFFT.f90:   SUBROUTINE cft_2xy_gpu(r_d, temp_d, nzl, nx, ny, ldx, ldy, isign, stream, pl2ix)
FFTXlib/src/fft_scalar.cuFFT.f90:     INTEGER(kind = cuda_stream_kind) :: stream
FFTXlib/src/fft_scalar.cuFFT.f90:     call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", " failed to set stream ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:     call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", " failed to set stream ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:     call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", " failed to set stream ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:     call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", " failed to set stream ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:     CALL start_clock( 'GPU_cft_2xy' )
FFTXlib/src/fft_scalar.cuFFT.f90:        call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", " cufftExecZ2Z failed ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:        call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", &
FFTXlib/src/fft_scalar.cuFFT.f90:           call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", &
FFTXlib/src/fft_scalar.cuFFT.f90:           call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", &
FFTXlib/src/fft_scalar.cuFFT.f90:        call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", " cufftExecZ2Z failed ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:           call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", " in fftxy ffty batch_1 istat = ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:           call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", " in fftxy ffty batch_2 istat = ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:        call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", " in fftxy fftx istat = ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:     CALL stop_clock( 'GPU_cft_2xy' )
FFTXlib/src/fft_scalar.cuFFT.f90:           call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", " cufftDestroy failed ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:       call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", " cufftPlanMany failed ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:           call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", " cufftDestroy failed x ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:           call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", " cufftDestroy failed y1 ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:           call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", " cufftDestroy failed y2 ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:       call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", " cufftPlanMany failed batch_x ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:       call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", " cufftPlanMany failed batch_y1 ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:       call fftx_error__(" fft_scalar_cuFFT: cft_2xy_gpu ", " cufftPlanMany failed batch_y2 ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:   END SUBROUTINE cft_2xy_gpu
FFTXlib/src/fft_scalar.cuFFT.f90:   SUBROUTINE cfft3d_gpu( f_d, nx, ny, nz, ldx, ldy, ldz, howmany, isign, stream )
FFTXlib/src/fft_scalar.cuFFT.f90:     INTEGER(kind = cuda_stream_kind) :: stream
FFTXlib/src/fft_scalar.cuFFT.f90:     call fftx_error__(" fft_scalar_cuFFT: cfft3d_gpu ", " failed to set stream ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:        call fftx_error__(" fft_scalar_cuFFT: cfft3d_gpu ", " cufftExecZ2Z failed ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:        call fftx_error__(" fft_scalar_cuFFT: cfft3d_gpu ", " cufftExecZ2Z failed ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:           call fftx_error__(" fft_scalar_cuFFT: cfft3d_gpu ", " cufftDestroy failed ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:       call fftx_error__(" fft_scalar_cuFFT: cfft3d_gpu ", " cufftPlanMany failed ", istat)
FFTXlib/src/fft_scalar.cuFFT.f90:   END SUBROUTINE cfft3d_gpu
FFTXlib/src/fft_scalar.cuFFT.f90:   SUBROUTINE cfft3ds_gpu (f_d, nx, ny, nz, ldx, ldy, ldz, howmany, isign, &
FFTXlib/src/fft_scalar.cuFFT.f90:     integer(kind = cuda_stream_kind) :: stream
FFTXlib/src/fft_scalar.cuFFT.f90:     ! cfft3d_gpu outperforms an explicit implementation of cfft3ds_gpu
FFTXlib/src/fft_scalar.cuFFT.f90:     CALL cfft3d_gpu (f_d, nx, ny, nz, ldx, ldy, ldz, howmany, isign, stream)
FFTXlib/src/fft_scalar.cuFFT.f90:   END SUBROUTINE cfft3ds_gpu
FFTXlib/src/fft_buffers.f90:#if defined(__CUDA)
FFTXlib/src/fft_helper_subroutines.f90:#if defined(__CUDA)
FFTXlib/src/fft_helper_subroutines.f90:  PUBLIC :: fftx_psi2c_gamma_gpu, fftx_c2psi_gamma_gpu
FFTXlib/src/fft_helper_subroutines.f90:#ifdef __CUDA
FFTXlib/src/fft_helper_subroutines.f90:  SUBROUTINE fftx_c2psi_gamma_gpu( desc, psi, c, ca )
FFTXlib/src/fft_helper_subroutines.f90:     !! Provisional gpu double of c2psi_gamma for CPV calls (CPV/src/exx_psi.f90).
FFTXlib/src/fft_helper_subroutines.f90:     !! To be removed after porting exx_psi to openacc.
FFTXlib/src/fft_helper_subroutines.f90:  END SUBROUTINE fftx_c2psi_gamma_gpu
FFTXlib/src/fft_helper_subroutines.f90:#if defined(_OPENACC)
FFTXlib/src/fft_helper_subroutines.f90:#if !defined(_OPENACC)
FFTXlib/src/fft_helper_subroutines.f90:     !! space. If gpu_args_ is true it assumes the dummy arrays are present
FFTXlib/src/fft_helper_subroutines.f90:     !! on device (openACC porting).
FFTXlib/src/fft_helper_subroutines.f90:  SUBROUTINE fftx_psi2c_gamma_gpu( desc, vin, vout1, vout2 )
FFTXlib/src/fft_helper_subroutines.f90:#if defined (__CUDA)
FFTXlib/src/fft_helper_subroutines.f90:  END SUBROUTINE fftx_psi2c_gamma_gpu
FFTXlib/src/fft_helper_subroutines.f90:#if defined(_OPENACC)
FFTXlib/src/fft_helper_subroutines.f90:#if !defined(_OPENACC)
FFTXlib/src/fft_helper_subroutines.f90:#if !defined(_OPENACC)
FFTXlib/src/fft_helper_subroutines.f90:#if defined(_OPENACC)
FFTXlib/src/fft_helper_subroutines.f90:#if defined(_OPENACC)
FFTXlib/src/fft_helper_subroutines.f90:#if !defined(_OPENACC)
FFTXlib/src/fft_helper_subroutines.f90:#if !defined(_OPENACC)
FFTXlib/src/fft_helper_subroutines.f90:    !! Workaround to use nl and nlm arrays with or without openacc+cuda
FFTXlib/src/fft_helper_subroutines.f90:#if defined(__CUDA) && defined(_OPENACC)
FFTXlib/src/fft_helper_subroutines.f90:    !! Workaround to use nl and nlm arrays with or without openacc+cuda
FFTXlib/src/fft_helper_subroutines.f90:#if !defined(__CUDA) || !defined(_OPENACC)
FFTXlib/src/fft_fwinv.f90:#if defined(__CUDA)
FFTXlib/src/fft_fwinv.f90:SUBROUTINE invfft_y_gpu( fft_kind, f_d, dfft, howmany, stream )
FFTXlib/src/fft_fwinv.f90:  USE cudafor
FFTXlib/src/fft_fwinv.f90:  USE fft_scalar,    ONLY: cfft3d_gpu, cfft3ds_gpu
FFTXlib/src/fft_fwinv.f90:  USE fft_parallel,  ONLY: tg_cft3s_gpu, many_cft3s_gpu
FFTXlib/src/fft_fwinv.f90:  USE fft_parallel_2d,  ONLY: tg_cft3s_2d_gpu => tg_cft3s_gpu, &
FFTXlib/src/fft_fwinv.f90:                            & many_cft3s_2d_gpu => many_cft3s_gpu
FFTXlib/src/fft_fwinv.f90:  INTEGER(kind = cuda_stream_kind), OPTIONAL, INTENT(IN) :: stream
FFTXlib/src/fft_fwinv.f90:  INTEGER(kind = cuda_stream_kind) :: stream_  = 0
FFTXlib/src/fft_fwinv.f90:  CALL start_clock_gpu(clock_label)
FFTXlib/src/fft_fwinv.f90:           CALL tg_cft3s_gpu( f_d, dfft, 1 )
FFTXlib/src/fft_fwinv.f90:            CALL many_cft3s_gpu( f_d, dfft, 1, howmany_ )
FFTXlib/src/fft_fwinv.f90:            CALL tg_cft3s_gpu( f_d, dfft, 2 )
FFTXlib/src/fft_fwinv.f90:            CALL many_cft3s_gpu( f_d, dfft, 2, howmany_ )
FFTXlib/src/fft_fwinv.f90:        CALL tg_cft3s_gpu( f_d, dfft, 3 )
FFTXlib/src/fft_fwinv.f90:           CALL many_cft3s_2d_gpu( f_d, dfft, 1,  howmany_)
FFTXlib/src/fft_fwinv.f90:           CALL many_cft3s_2d_gpu( f_d, dfft, 2, howmany_ )
FFTXlib/src/fft_fwinv.f90:           CALL tg_cft3s_2d_gpu( f_d, dfft, 1 )
FFTXlib/src/fft_fwinv.f90:           CALL tg_cft3s_2d_gpu( f_d, dfft, 2 )
FFTXlib/src/fft_fwinv.f90:        CALL cfft3d_gpu( f_d, dfft%nr1, dfft%nr2, dfft%nr3, &
FFTXlib/src/fft_fwinv.f90:        CALL cfft3ds_gpu( f_d, dfft%nr1, dfft%nr2, dfft%nr3, &
FFTXlib/src/fft_fwinv.f90:  CALL stop_clock_gpu( clock_label )
FFTXlib/src/fft_fwinv.f90:END SUBROUTINE invfft_y_gpu
FFTXlib/src/fft_fwinv.f90:SUBROUTINE fwfft_y_gpu( fft_kind, f_d, dfft, howmany, stream )
FFTXlib/src/fft_fwinv.f90:  USE cudafor
FFTXlib/src/fft_fwinv.f90:  USE fft_scalar,    ONLY: cfft3d_gpu, cfft3ds_gpu
FFTXlib/src/fft_fwinv.f90:  USE fft_parallel,  ONLY: tg_cft3s_gpu, many_cft3s_gpu
FFTXlib/src/fft_fwinv.f90:  USE fft_parallel_2d,  ONLY: tg_cft3s_2d_gpu => tg_cft3s_gpu, &
FFTXlib/src/fft_fwinv.f90:                              & many_cft3s_2d_gpu => many_cft3s_gpu
FFTXlib/src/fft_fwinv.f90:  INTEGER(kind = cuda_stream_kind), OPTIONAL, INTENT(IN) :: stream
FFTXlib/src/fft_fwinv.f90:  INTEGER(kind = cuda_stream_kind) :: stream_  = 0
FFTXlib/src/fft_fwinv.f90:  CALL start_clock_gpu(clock_label)
FFTXlib/src/fft_fwinv.f90:           CALL tg_cft3s_gpu(f_d,dfft,-1)
FFTXlib/src/fft_fwinv.f90:           CALL many_cft3s_gpu(f_d,dfft,-1, howmany_)
FFTXlib/src/fft_fwinv.f90:           CALL tg_cft3s_gpu(f_d,dfft,-2)
FFTXlib/src/fft_fwinv.f90:           CALL many_cft3s_gpu(f_d,dfft,-2, howmany_)
FFTXlib/src/fft_fwinv.f90:        CALL tg_cft3s_gpu( f_d, dfft, -3 )
FFTXlib/src/fft_fwinv.f90:           CALL many_cft3s_2d_gpu( f_d, dfft, -1, howmany_)
FFTXlib/src/fft_fwinv.f90:           CALL many_cft3s_2d_gpu( f_d, dfft, -2, howmany_ )
FFTXlib/src/fft_fwinv.f90:           CALL tg_cft3s_2d_gpu( f_d, dfft, -1 )
FFTXlib/src/fft_fwinv.f90:           CALL tg_cft3s_2d_gpu( f_d, dfft, -2 )
FFTXlib/src/fft_fwinv.f90:        CALL cfft3d_gpu( f_d, dfft%nr1, dfft%nr2, dfft%nr3, &
FFTXlib/src/fft_fwinv.f90:        CALL cfft3ds_gpu( f_d, dfft%nr1, dfft%nr2, dfft%nr3, &
FFTXlib/src/fft_fwinv.f90:  CALL stop_clock_gpu( clock_label )
FFTXlib/src/fft_fwinv.f90:END SUBROUTINE fwfft_y_gpu
FFTXlib/src/Makefile:fft_scatter_gpu.o  \
FFTXlib/src/Makefile:fft_scatter_2d_gpu.o  \
FFTXlib/src/fft_scalar.f90:! CUDA FFT for NVidiia GPUs
FFTXlib/src/fft_scalar.f90:#if defined(__CUDA)
FFTXlib/src/fft_scalar.f90:#if defined(__CUDA)
FFTXlib/src/fft_scalar.f90:     PUBLIC :: cft_1z_gpu, cft_2xy_gpu, cfft3d_gpu, cfft3ds_gpu
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __CUDA
FFTXlib/src/fft_scatter_2d_gpu.f90:   MODULE fft_scatter_2d_gpu
FFTXlib/src/fft_scatter_2d_gpu.f90:        USE cudafor
FFTXlib/src/fft_scatter_2d_gpu.f90:        PUBLIC :: fft_scatter_gpu, fft_scatter_gpu_batch
FFTXlib/src/fft_scatter_2d_gpu.f90:SUBROUTINE fft_scatter_gpu ( dfft, f_in_d, f_in, nr3x, nxx_, f_aux_d, f_aux, ncp_, npp_, isgn )
FFTXlib/src/fft_scatter_2d_gpu.f90:  USE cudafor
FFTXlib/src/fft_scatter_2d_gpu.f90:  istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:        istat = cudaMemcpy2D( f_aux(kdest + 1), nppx, f_in_d(kfrom + 1 ), nr3x, npp_(gproc), ncp_(me), cudaMemcpyDeviceToHost )
FFTXlib/src/fft_scatter_2d_gpu.f90:        if( istat ) CALL fftx_error__("fft_scatter", "ERROR cudaMemcpy2D failed : ", istat)
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:     istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:     istat = cudaMemcpyAsync( f_in_d( (me-1)*sendsiz + 1), f_aux_d((me-1)*sendsiz + 1), sendsiz, stream=dfft%a2a_comp )
FFTXlib/src/fft_scatter_2d_gpu.f90:     istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifndef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifndef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:     istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:     istat = cudaMemcpyAsync( f_aux_d( (me-1)*sendsiz + 1), f_in_d((me-1)*sendsiz + 1), sendsiz, stream=dfft%a2a_comp )
FFTXlib/src/fft_scatter_2d_gpu.f90:     istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:        istat = cudaMemcpy2D( f_in_d(kfrom +1 ), nr3x, f_aux(kdest + 1), nppx, npp_(gproc), ncp_(me), cudaMemcpyHostToDevice )
FFTXlib/src/fft_scatter_2d_gpu.f90:  istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:END SUBROUTINE fft_scatter_gpu
FFTXlib/src/fft_scatter_2d_gpu.f90:SUBROUTINE fft_scatter_gpu_batch ( dfft, f_in_d, f_in, nr3x, nxx_, f_aux_d, f_aux, ncp_, npp_, isgn, batchsize, srh )
FFTXlib/src/fft_scatter_2d_gpu.f90:  ! This subroutine performs the same task as fft_scatter_gpu, but for
FFTXlib/src/fft_scatter_2d_gpu.f90:  USE cudafor
FFTXlib/src/fft_scatter_2d_gpu.f90:  istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:        istat = cudaMemcpy2D( f_aux(kdest + 1), nppx, f_in_d(kfrom + 1 ), nr3x, npp_(gproc), batchsize * ncpx, cudaMemcpyDeviceToHost )
FFTXlib/src/fft_scatter_2d_gpu.f90:        IF( istat /= cudaSuccess ) CALL fftx_error__ ('fft_scatter', 'cudaMemcpy2D failed: ', istat)
FFTXlib/src/fft_scatter_2d_gpu.f90:     istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:     istat = cudaMemcpyAsync( f_in_d( (me-1)*sendsiz + 1), f_aux_d((me-1)*sendsiz + 1), sendsiz, stream=dfft%a2a_comp )
FFTXlib/src/fft_scatter_2d_gpu.f90:     IF( istat /= cudaSuccess ) CALL fftx_error__ ('fft_scatter', 'cudaMemcpyAsync failed: ', istat)
FFTXlib/src/fft_scatter_2d_gpu.f90:     istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifndef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifndef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:     istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:     istat = cudaMemcpyAsync( f_aux_d( (me-1)*sendsiz + 1), f_in_d((me-1)*sendsiz + 1), sendsiz, stream=dfft%a2a_comp )
FFTXlib/src/fft_scatter_2d_gpu.f90:     if( istat /= cudaSuccess ) CALL fftx_error__ ('fft_scatter', 'cudaMemcpyAsync failed: ', istat)
FFTXlib/src/fft_scatter_2d_gpu.f90:     istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:        istat = cudaMemcpy2D( f_in_d(kfrom +1 ), nr3x, f_aux(kdest + 1), nppx, npp_(gproc), batchsize * ncpx, cudaMemcpyHostToDevice )
FFTXlib/src/fft_scatter_2d_gpu.f90:        IF( istat /= cudaSuccess ) CALL fftx_error__ ('fft_scatter', 'cudaMemcpy2D failed: ', istat)
FFTXlib/src/fft_scatter_2d_gpu.f90:  istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:END SUBROUTINE fft_scatter_gpu_batch
FFTXlib/src/fft_scatter_2d_gpu.f90:   USE cudafor
FFTXlib/src/fft_scatter_2d_gpu.f90:   !istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifndef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:      istat = cudaMemcpy2DAsync( f_aux_d(kdest + 1), nppx, f_in_d(kfrom + 1 ), nr3x, npp_(proc), batchsize * ncpx,cudaMemcpyDeviceToDevice, dfft%bstreams(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:      IF( istat /= cudaSuccess ) CALL fftx_error__ ('fft_scatter_many_columns_to_planes_store', 'cudaMemcpy2DAsync failed : ', istat)
FFTXlib/src/fft_scatter_2d_gpu.f90:         istat = cudaMemcpy2DAsync( f_aux_d(kdest + 1), nppx, f_in_d(kfrom + 1 ), nr3x, npp_(proc), batchsize * ncpx,cudaMemcpyDeviceToDevice, dfft%bstreams(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:         IF( istat /= cudaSuccess ) CALL fftx_error__ ('fft_scatter_many_columns_to_planes_store', 'cudaMemcpy2DAsync failed : ', istat)
FFTXlib/src/fft_scatter_2d_gpu.f90:         istat = cudaMemcpy2DAsync( f_aux(kdest + 1), nppx, f_in_d(kfrom + 1 ), nr3x, npp_(proc), batchsize * ncpx,cudaMemcpyDeviceToHost, dfft%bstreams(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:         IF( istat /= cudaSuccess ) CALL fftx_error__ ('fft_scatter_many_columns_to_planes_store', 'cudaMemcpy2DAsync failed : ', istat)
FFTXlib/src/fft_scatter_2d_gpu.f90:      istat = cudaMemcpy2DAsync( f_aux(kdest + 1), nppx, f_in_d(kfrom + 1 ), nr3x, npp_(proc), batchsize * ncpx,cudaMemcpyDeviceToHost, dfft%bstreams(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:      IF( istat /= cudaSuccess ) CALL fftx_error__ ('fft_scatter_many_columns_to_planes_store', 'cudaMemcpy2DAsync failed : ', istat)
FFTXlib/src/fft_scatter_2d_gpu.f90:   istat = cudaEventRecord( dfft%bevents(batch_id), dfft%bstreams(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:   USE cudafor
FFTXlib/src/fft_scatter_2d_gpu.f90:   !istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:   istat = cudaEventSynchronize( dfft%bevents(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:   istat = cudaMemcpy2DAsync( f_aux2_d((me-1)*sendsiz + 1), nppx, f_in_d(offset + 1 ), nr3x, npp_(me), batchsize * ncpx,cudaMemcpyDeviceToDevice, dfft%bstreams(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:   IF( istat /= cudaSuccess ) CALL fftx_error__ ('fft_scatter_many_columns_to_planes_store', 'cudaMemcpy2DAsync failed : ', istat)
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifndef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:            istat = cudaMemcpyAsync( f_aux2_d(kdest+1), f_aux2(kdest+1), sendsiz, stream=dfft%bstreams(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:         istat = cudaMemcpyAsync( f_aux2_d(kdest+1), f_aux2(kdest+1), sendsiz, stream=dfft%bstreams(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:   i = cudaEventRecord(dfft%bevents(batch_id), dfft%bstreams(batch_id))
FFTXlib/src/fft_scatter_2d_gpu.f90:   i = cudaStreamWaitEvent(dfft%a2a_comp, dfft%bevents(batch_id), 0)
FFTXlib/src/fft_scatter_2d_gpu.f90:  !istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:   USE cudafor
FFTXlib/src/fft_scatter_2d_gpu.f90:   !istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifndef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifndef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:   i = cudaEventRecord(dfft%bevents(batch_id), dfft%a2a_comp)
FFTXlib/src/fft_scatter_2d_gpu.f90:   i = cudaStreamWaitEvent(dfft%bstreams(batch_id), dfft%bevents(batch_id), 0)
FFTXlib/src/fft_scatter_2d_gpu.f90:            istat = cudaMemcpyAsync( f_aux2(kdest+1), f_aux2_d(kdest+1), sendsiz, stream=dfft%bstreams(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:         istat = cudaMemcpyAsync( f_aux2(kdest+1), f_aux2_d(kdest+1), sendsiz, stream=dfft%bstreams(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:   istat = cudaEventRecord( dfft%bevents(batch_id), dfft%a2a_comp )
FFTXlib/src/fft_scatter_2d_gpu.f90:   istat = cudaEventRecord( dfft%bevents(batch_id), dfft%bstreams(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:  !istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:   USE cudafor
FFTXlib/src/fft_scatter_2d_gpu.f90:   !istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:   istat = cudaEventSynchronize( dfft%bevents(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:   istat = cudaMemcpy2DAsync( f_in_d(offset + 1), nr3x, f_aux2_d((me-1)*sendsiz + 1), nppx, npp_(me), batchsize * ncpx, &
FFTXlib/src/fft_scatter_2d_gpu.f90:                              cudaMemcpyDeviceToDevice, dfft%bstreams(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:#ifdef __GPU_MPI
FFTXlib/src/fft_scatter_2d_gpu.f90:!         This commented code is left here for helping understand the following calls to CUDA APIs
FFTXlib/src/fft_scatter_2d_gpu.f90:        istat = cudaMemcpy2DAsync( f_in_d(kfrom +1 ), nr3x, f_aux_d(kdest + 1), nppx, npp_(gproc), batchsize * ncpx, &
FFTXlib/src/fft_scatter_2d_gpu.f90:        cudaMemcpyDeviceToDevice, dfft%bstreams(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:        IF( istat /= cudaSuccess ) CALL fftx_error__ ('fft_scatter_many_planes_to_columns_send', 'cudaMemcpy2DAsync failed : ', istat)
FFTXlib/src/fft_scatter_2d_gpu.f90:             istat = cudaMemcpy2DAsync( f_in_d(kfrom +1 ), nr3x, f_aux_d(kdest + 1), nppx, npp_(gproc), batchsize * ncpx, &
FFTXlib/src/fft_scatter_2d_gpu.f90:                                        cudaMemcpyDeviceToDevice, dfft%bstreams(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:             istat = cudaMemcpy2DAsync( f_in_d(kfrom +1 ), nr3x, f_aux(kdest + 1), nppx, npp_(gproc), batchsize * ncpx, &
FFTXlib/src/fft_scatter_2d_gpu.f90:                                        cudaMemcpyHostToDevice, dfft%bstreams(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:        IF( istat /= cudaSuccess ) CALL fftx_error__ ('fft_scatter_many_planes_to_columns_send', 'cudaMemcpy2DAsync failed : ', istat)
FFTXlib/src/fft_scatter_2d_gpu.f90:        istat = cudaMemcpy2DAsync( f_in_d(kfrom +1 ), nr3x, f_aux(kdest + 1), nppx, npp_(gproc), batchsize * ncpx, &
FFTXlib/src/fft_scatter_2d_gpu.f90:                                   cudaMemcpyHostToDevice, dfft%bstreams(batch_id) )
FFTXlib/src/fft_scatter_2d_gpu.f90:        IF( istat /= cudaSuccess ) CALL fftx_error__ ('fft_scatter_many_planes_to_columns_send', 'cudaMemcpy2DAsync failed : ', istat)
FFTXlib/src/fft_scatter_2d_gpu.f90:   !istat = cudaDeviceSynchronize()
FFTXlib/src/fft_scatter_2d_gpu.f90:END MODULE fft_scatter_2d_gpu
FFTXlib/src/CMakeLists.txt:if(QE_ENABLE_CUDA)
FFTXlib/src/CMakeLists.txt:        fft_scatter_gpu.f90
FFTXlib/src/CMakeLists.txt:        fft_scatter_2d_gpu.f90)
FFTXlib/src/CMakeLists.txt:qe_enable_cuda_fortran("${f_src_fftx}")
FFTXlib/src/CMakeLists.txt:if(QE_ENABLE_CUDA)
FFTXlib/src/CMakeLists.txt:            qe_openacc_fortran
FFTXlib/src/CMakeLists.txt:            CUDA::cufft)
FFTXlib/src/fft_ggen.f90:#if defined(__CUDA)
FFTXlib/src/fft_interfaces.f90:#if defined(__CUDA)
FFTXlib/src/fft_interfaces.f90:     SUBROUTINE invfft_y_gpu( grid_type, f_d, dfft, howmany, stream )
FFTXlib/src/fft_interfaces.f90:       USE cudafor
FFTXlib/src/fft_interfaces.f90:       INTEGER(kind = cuda_stream_kind), OPTIONAL, INTENT(IN) :: stream
FFTXlib/src/fft_interfaces.f90:     END SUBROUTINE invfft_y_gpu
FFTXlib/src/fft_interfaces.f90:#if defined(__CUDA)
FFTXlib/src/fft_interfaces.f90:     SUBROUTINE fwfft_y_gpu( grid_type, f_d, dfft, howmany, stream )
FFTXlib/src/fft_interfaces.f90:       USE cudafor
FFTXlib/src/fft_interfaces.f90:       INTEGER(kind = cuda_stream_kind), OPTIONAL, INTENT(IN) :: stream
FFTXlib/src/fft_interfaces.f90:     END SUBROUTINE fwfft_y_gpu
FFTXlib/src/fft_parallel_2d.f90:#ifdef __CUDA
FFTXlib/src/fft_parallel_2d.f90:   USE cudafor
FFTXlib/src/fft_parallel_2d.f90:#if defined(__CUDA)
FFTXlib/src/fft_parallel_2d.f90:SUBROUTINE tg_cft3s_gpu( f_d, dfft, isgn )
FFTXlib/src/fft_parallel_2d.f90:  USE fft_scalar, ONLY : cft_1z_gpu, cft_2xy_gpu
FFTXlib/src/fft_parallel_2d.f90:  USE fft_scatter_2d_gpu,   ONLY : fft_scatter_gpu
FFTXlib/src/fft_parallel_2d.f90:  INTEGER(kind = cuda_stream_kind) :: stream  = 0
FFTXlib/src/fft_parallel_2d.f90:        CALL cft_1z_gpu( f_d, dfft%nsp( me_p ), n3, nx3, isgn, aux_d, stream )
FFTXlib/src/fft_parallel_2d.f90:        CALL cft_1z_gpu( f_d, dfft%nsw( me_p ), n3, nx3, isgn, aux_d, stream )
FFTXlib/src/fft_parallel_2d.f90:     CALL fw_scatter_gpu( isgn ) ! forward scatter from stick to planes
FFTXlib/src/fft_parallel_2d.f90:     CALL cft_2xy_gpu( f_d, aux_d, dfft%my_nr3p, n1, n2, nx1, nx2, isgn, stream, planes )
FFTXlib/src/fft_parallel_2d.f90:     CALL cft_2xy_gpu( f_d, aux_d, dfft%my_nr3p, n1, n2, nx1, nx2, isgn, stream, planes)
FFTXlib/src/fft_parallel_2d.f90:     CALL bw_scatter_gpu( isgn )
FFTXlib/src/fft_parallel_2d.f90:        CALL cft_1z_gpu( aux_d, dfft%nsp( me_p ), n3, nx3, isgn, f_d, stream )
FFTXlib/src/fft_parallel_2d.f90:        CALL cft_1z_gpu( aux_d, dfft%nsw( me_p ), n3, nx3, isgn, f_d, stream )
FFTXlib/src/fft_parallel_2d.f90:  SUBROUTINE fw_scatter_gpu( iopt )
FFTXlib/src/fft_parallel_2d.f90:     USE fft_scatter_2d_gpu, ONLY: fft_scatter_gpu
FFTXlib/src/fft_parallel_2d.f90:        CALL fft_scatter_gpu( dfft, aux_d, aux_h, nx3, dfft%nnr, f_d, f_h, dfft%nsw, dfft%nr3p, iopt )
FFTXlib/src/fft_parallel_2d.f90:        CALL fft_scatter_gpu( dfft, aux_d, aux_h, nx3, dfft%nnr, f_d, f_h, dfft%nsp, dfft%nr3p, iopt )
FFTXlib/src/fft_parallel_2d.f90:  END SUBROUTINE fw_scatter_gpu
FFTXlib/src/fft_parallel_2d.f90:  SUBROUTINE bw_scatter_gpu( iopt )
FFTXlib/src/fft_parallel_2d.f90:     USE fft_scatter_2d_gpu, ONLY: fft_scatter_gpu
FFTXlib/src/fft_parallel_2d.f90:        CALL fft_scatter_gpu( dfft, aux_d, aux_h, nx3, dfft%nnr, f_d, f_h, dfft%nsw, dfft%nr3p, iopt )
FFTXlib/src/fft_parallel_2d.f90:        CALL fft_scatter_gpu( dfft, aux_d, aux_h, nx3, dfft%nnr, f_d, f_h, dfft%nsp, dfft%nr3p, iopt )
FFTXlib/src/fft_parallel_2d.f90:  END SUBROUTINE bw_scatter_gpu
FFTXlib/src/fft_parallel_2d.f90:END SUBROUTINE tg_cft3s_gpu
FFTXlib/src/fft_parallel_2d.f90:SUBROUTINE many_cft3s_gpu( f_d, dfft, isgn, batchsize )
FFTXlib/src/fft_parallel_2d.f90:  ! The GPU version is based on code written by Josh Romero, Everett Phillips
FFTXlib/src/fft_parallel_2d.f90:  USE fft_scalar, ONLY : cft_1z_gpu, cft_2xy_gpu
FFTXlib/src/fft_parallel_2d.f90:  USE fft_scatter_2d_gpu,   ONLY : fft_scatter_many_columns_to_planes_send, &
FFTXlib/src/fft_parallel_2d.f90:  INTEGER(kind = cuda_stream_kind) :: stream  = 0
FFTXlib/src/fft_parallel_2d.f90:     CALL fftx_error__( ' many_cft3s_gpu ', ' abs(isgn) /= 1 or 2 not implemented ', isgn )
FFTXlib/src/fft_parallel_2d.f90:  IF (dfft%nproc <= 1) CALL fftx_error__( ' many_cft3s_gpu ', ' this subroutine should never be called with nproc= ', dfft%nproc )
FFTXlib/src/fft_parallel_2d.f90:         CALL cft_1z_gpu( f_d((j+i)*dfft%nnr + 1:), sticks(me_p), n3, nx3, isgn, aux_d(j*dfft%nnr + i*ncpx*nx3 +1:), dfft%a2a_comp )
FFTXlib/src/fft_parallel_2d.f90:       i = cudaEventRecord(dfft%bevents(j/dfft%subbatchsize + 1), dfft%a2a_comp)
FFTXlib/src/fft_parallel_2d.f90:       i = cudaStreamWaitEvent(dfft%bstreams(j/dfft%subbatchsize + 1), dfft%bevents(j/dfft%subbatchsize + 1), 0)
FFTXlib/src/fft_parallel_2d.f90:       IF (j > 0) i = cudaStreamWaitEvent(dfft%bstreams(j/dfft%subbatchsize + 1), dfft%bevents(j/dfft%subbatchsize), 0)
FFTXlib/src/fft_parallel_2d.f90:         CALL cft_2xy_gpu( f_d(j*dfft%nnr + 1:), aux_d(j*dfft%nnr + 1:), currsize * nppx, n1, n2, nx1, nx2, isgn, dfft%a2a_comp, planes )
FFTXlib/src/fft_parallel_2d.f90:           CALL cft_2xy_gpu( f_d((j+i)*dfft%nnr + 1:), aux_d((j+i)*dfft%nnr + 1:), dfft%nr3p( me_p ), n1, n2, nx1, nx2, isgn,  &
FFTXlib/src/fft_parallel_2d.f90:!     i = cudaDeviceSynchronize()
FFTXlib/src/fft_parallel_2d.f90:!     i = cudaDeviceSynchronize()
FFTXlib/src/fft_parallel_2d.f90:         CALL cft_2xy_gpu( f_d(j*dfft%nnr + 1:), aux_d(j*dfft%nnr + 1:), currsize * nppx, n1, n2, nx1, nx2, isgn, dfft%a2a_comp, planes )
FFTXlib/src/fft_parallel_2d.f90:           CALL cft_2xy_gpu( f_d((j+i)*dfft%nnr + 1:), aux_d((j+i)*dfft%nnr + 1:), dfft%nr3p( me_p ), n1, n2, nx1, nx2, isgn, dfft%a2a_comp, planes )
FFTXlib/src/fft_parallel_2d.f90:       IF (j > 0) i = cudaStreamWaitEvent(dfft%bstreams(j/dfft%subbatchsize + 1), dfft%bevents(j/dfft%subbatchsize), 0)
FFTXlib/src/fft_parallel_2d.f90:       i = cudaEventRecord(dfft%bevents(j/dfft%subbatchsize + 1), dfft%bstreams(j/dfft%subbatchsize + 1))
FFTXlib/src/fft_parallel_2d.f90:       i = cudaStreamWaitEvent(dfft%a2a_comp, dfft%bevents(j/dfft%subbatchsize + 1), 0)
FFTXlib/src/fft_parallel_2d.f90:         CALL cft_1z_gpu( aux_d(j*dfft%nnr + i*ncpx*nx3 + 1:), sticks( me_p ), n3, nx3, isgn, f_d((j+i)*dfft%nnr + 1:), dfft%a2a_comp )
FFTXlib/src/fft_parallel_2d.f90:!    i = cudaDeviceSynchronize()
FFTXlib/src/fft_parallel_2d.f90:END SUBROUTINE many_cft3s_gpu
FFTXlib/src/tg_gather.f90:#if defined(__CUDA)
FFTXlib/src/tg_gather.f90:! === GPU CODE ===
FFTXlib/src/tg_gather.f90:SUBROUTINE tg_gather_gpu( dffts, v_d, tg_v_d )
FFTXlib/src/tg_gather.f90:  ! `dffts` appear in the body of do loops, the compiler generates incorrect GPU code.
FFTXlib/src/tg_gather.f90:END SUBROUTINE tg_gather_gpu
FFTXlib/src/tg_gather.f90:SUBROUTINE tg_cgather_gpu( dffts, v_d, tg_v_d )
FFTXlib/src/tg_gather.f90:  ! `dffts` appear in the body of do loops, the compiler generates incorrect GPU code.
FFTXlib/src/tg_gather.f90:END SUBROUTINE tg_cgather_gpu
FFTXlib/src/fft_parallel.f90:#if defined(__CUDA)
FFTXlib/src/fft_parallel.f90:!  General purpose driver, GPU version
FFTXlib/src/fft_parallel.f90:SUBROUTINE tg_cft3s_gpu( f_d, dfft, isgn )
FFTXlib/src/fft_parallel.f90:  USE cudafor
FFTXlib/src/fft_parallel.f90:  USE fft_scalar,     ONLY : cft_1z_gpu
FFTXlib/src/fft_parallel.f90:  USE fft_scatter_gpu,ONLY : fft_scatter_xy_gpu, fft_scatter_yz_gpu, fft_scatter_tg_gpu
FFTXlib/src/fft_parallel.f90:  USE fft_scatter_gpu,ONLY : fft_scatter_tg_opt_gpu
FFTXlib/src/fft_parallel.f90:  INTEGER(kind = cuda_stream_kind) :: stream  = 0
FFTXlib/src/fft_parallel.f90:     !CALL nvtxStartRangeAsync("tg_cft3s_gpu G->R", 1)
FFTXlib/src/fft_parallel.f90:        call fft_scatter_tg_opt_gpu ( dfft, f_d, aux_d, nnr_, isgn, stream)
FFTXlib/src/fft_parallel.f90:        CALL cft_1z_gpu( aux_d, nsticks_z, n3, nx3, isgn, f_d, stream )
FFTXlib/src/fft_parallel.f90:        !ierr = cudaMemcpy( aux_d(1), f_d(1), nnr_, cudaMemcpyDeviceToDevice )
FFTXlib/src/fft_parallel.f90:        CALL cft_1z_gpu( f_d, nsticks_z, n3, nx3, isgn, aux_d, stream, in_place=.true. )
FFTXlib/src/fft_parallel.f90:     CALL fft_scatter_yz_gpu ( dfft, f_d, aux_d, nnr_, isgn )
FFTXlib/src/fft_parallel.f90:     CALL cft_1z_gpu( aux_d, nsticks_y, n2, nx2, isgn, f_d, stream )
FFTXlib/src/fft_parallel.f90:     CALL fft_scatter_xy_gpu ( dfft, f_d, aux_d, nnr_, isgn, stream )
FFTXlib/src/fft_parallel.f90:     CALL cft_1z_gpu( aux_d, nsticks_x, n1, nx1, isgn, f_d, stream )
FFTXlib/src/fft_parallel.f90:     !CALL nvtxStartRangeAsync("tg_cft3s_gpu R->G", 2)
FFTXlib/src/fft_parallel.f90:     CALL cft_1z_gpu( f_d, nsticks_x, n1, nx1, isgn, aux_d, stream )
FFTXlib/src/fft_parallel.f90:     CALL fft_scatter_xy_gpu ( dfft, f_d, aux_d, nnr_, isgn, stream )
FFTXlib/src/fft_parallel.f90:     CALL cft_1z_gpu( f_d, nsticks_y, n2, nx2, isgn, aux_d, stream )
FFTXlib/src/fft_parallel.f90:     CALL fft_scatter_yz_gpu ( dfft, f_d, aux_d, nnr_, isgn )
FFTXlib/src/fft_parallel.f90:        CALL cft_1z_gpu( f_d, nsticks_z, n3, nx3, isgn, aux_d, stream )
FFTXlib/src/fft_parallel.f90:        call fft_scatter_tg_opt_gpu ( dfft, aux_d, f_d, nnr_, isgn, stream)
FFTXlib/src/fft_parallel.f90:        !ierr = cudaMemcpy( f_d(1), aux_d(1), nnr_, cudaMemcpyDeviceToDevice )
FFTXlib/src/fft_parallel.f90:        CALL cft_1z_gpu( f_d, nsticks_z, n3, nx3, isgn, aux_d, stream, in_place=.true. )
FFTXlib/src/fft_parallel.f90:END SUBROUTINE tg_cft3s_gpu
FFTXlib/src/fft_parallel.f90:!  Specific driver for the new 'many' call, GPU version
FFTXlib/src/fft_parallel.f90:SUBROUTINE many_cft3s_gpu( f_d, dfft, isgn, howmany )
FFTXlib/src/fft_parallel.f90:  USE cudafor
FFTXlib/src/fft_parallel.f90:  USE fft_scalar,     ONLY : cft_1z_gpu
FFTXlib/src/fft_parallel.f90:  USE fft_scatter_gpu,ONLY : fft_scatter_xy_gpu, fft_scatter_yz_gpu, &
FFTXlib/src/fft_parallel.f90:                              & fft_scatter_tg_gpu, fft_scatter_many_yz_gpu, &
FFTXlib/src/fft_parallel.f90:                              & fft_scatter_tg_opt_gpu
FFTXlib/src/fft_parallel.f90:  INTEGER(kind = cuda_stream_kind) :: stream  = 0 ! cuda_default_stream
FFTXlib/src/fft_parallel.f90:  ierr = cudaDeviceSynchronize()
FFTXlib/src/fft_parallel.f90:     !CALL nvtxStartRangeAsync("many_cft3s_gpu G->R", 1)
FFTXlib/src/fft_parallel.f90:        CALL cft_1z_gpu( f_d(i*nnr_+1:), nsticks_z, n3, nx3, isgn, aux_d(nx3*nsticks_zx*i+1:), &
FFTXlib/src/fft_parallel.f90:     CALL fft_scatter_many_yz_gpu ( dfft, aux_d(1), f_d(1), howmany*nnr_, isgn, howmany )
FFTXlib/src/fft_parallel.f90:        CALL cft_1z_gpu( f_d(i*nnr_+1:), nsticks_y, n2, nx2, isgn, aux_d(i*nnr_+1:), &
FFTXlib/src/fft_parallel.f90:        CALL fft_scatter_xy_gpu ( dfft, f_d(i*nnr_+1:), aux_d(i*nnr_+1:), nnr_, isgn, &
FFTXlib/src/fft_parallel.f90:        CALL cft_1z_gpu( aux_d(i*nnr_+1:), nsticks_x, n1, nx1, isgn, f_d(i*nnr_+1:), &
FFTXlib/src/fft_parallel.f90:     !CALL nvtxStartRangeAsync("many_cft3s_gpu R->G", 2)
FFTXlib/src/fft_parallel.f90:        CALL cft_1z_gpu( f_d(i*nnr_+1:), nsticks_x, n1, nx1, isgn, aux_d(i*nnr_+1:), &
FFTXlib/src/fft_parallel.f90:        CALL fft_scatter_xy_gpu ( dfft, f_d(i*nnr_+1), aux_d(i*nnr_+1), nnr_, isgn, &
FFTXlib/src/fft_parallel.f90:        CALL cft_1z_gpu( f_d(i*nnr_+1:), nsticks_y, n2, nx2, isgn, aux_d(i*nnr_+1:), &
FFTXlib/src/fft_parallel.f90:     CALL fft_scatter_many_yz_gpu ( dfft, aux_d, f_d, howmany*nnr_, isgn, howmany )
FFTXlib/src/fft_parallel.f90:        CALL cft_1z_gpu( aux_d(nx3*nsticks_zx*i+1:), nsticks_z, n3, nx3, isgn, f_d(i*nnr_+1:), &
FFTXlib/src/fft_parallel.f90:  !    ierr = cudaStreamSynchronize( dfft%stream_many(i) )
FFTXlib/src/fft_parallel.f90:END SUBROUTINE many_cft3s_gpu
FFTXlib/src/fft_types.f90:#if defined(__CUDA)
FFTXlib/src/fft_types.f90:#if defined(__CUDA)
FFTXlib/src/fft_types.f90:  USE cudafor
FFTXlib/src/fft_types.f90:#if defined(__CUDA)
FFTXlib/src/fft_types.f90:    ! These CUDA streams are used in the 1D+1D+1D GPU implementation
FFTXlib/src/fft_types.f90:    INTEGER(kind=cuda_stream_kind), allocatable, dimension(:) :: stream_scatter_yz
FFTXlib/src/fft_types.f90:    INTEGER(kind=cuda_stream_kind), allocatable, dimension(:) :: stream_many
FFTXlib/src/fft_types.f90:    ! These CUDA streams (and events) are used in the 1D+2D FPU implementation
FFTXlib/src/fft_types.f90:    INTEGER(kind=cuda_stream_kind) :: a2a_comp
FFTXlib/src/fft_types.f90:    INTEGER(kind=cuda_stream_kind), allocatable, dimension(:) :: bstreams
FFTXlib/src/fft_types.f90:    TYPE(cudaEvent), allocatable, dimension(:) :: bevents
FFTXlib/src/fft_types.f90:    ! * the 1D+2D GPU implementation:
FFTXlib/src/fft_types.f90:#if defined(__CUDA)
FFTXlib/src/fft_types.f90:        ierr = cudaStreamCreate(desc%stream_scatter_yz(iproc))
FFTXlib/src/fft_types.f90:        ierr = cudaStreamCreate(desc%stream_many(i))
FFTXlib/src/fft_types.f90:    ierr = cudaStreamCreate( desc%a2a_comp )
FFTXlib/src/fft_types.f90:      ierr = cudaStreamCreate( desc%bstreams(i) )
FFTXlib/src/fft_types.f90:      ierr = cudaEventCreate( desc%bevents(i) )
FFTXlib/src/fft_types.f90:#if defined(__CUDA)
FFTXlib/src/fft_types.f90:            ierr = cudaStreamDestroy(desc%stream_scatter_yz(i))
FFTXlib/src/fft_types.f90:            ierr = cudaStreamDestroy(desc%stream_many(i))
FFTXlib/src/fft_types.f90:      ierr = cudaStreamDestroy( desc%a2a_comp )
FFTXlib/src/fft_types.f90:          ierr = cudaStreamDestroy( desc%bstreams(i) )
FFTXlib/src/fft_types.f90:          ierr = cudaEventDestroy( desc%bevents(i) )
FFTXlib/src/fft_types.f90:#if defined(__CUDA)
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:SUBROUTINE ppcg_gamma_gpu( h_psi_ptr, s_psi_ptr, overlap, precondition_d, &
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:  ! IC gpu version
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:#if defined(__CUDA)
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:  USE cudafor
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:  ! device arrays and variables for GPU computation
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:#if defined(__CUDA)
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:  call gpu_threaded_memset( G_d, ZERO, nbnd*nbnd ) ! G = ZERO
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:  call gpu_threaded_assign( w_d, hpsi_d, npwx, nact, .false., act_idx_d, .true. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:     call gpu_threaded_assign( buffer_d, w_d, npwx, nact, .true., act_idx_d, .false.)
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:     call gpu_threaded_memset( G_d, ZERO, nbnd*nact ) ! G(1:nbnd,1:nact) = ZERO
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:     call gpu_threaded_assign( buffer_d, w_d, npwx, nact, .true., act_idx_d, .true. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:     call gpu_threaded_backassign( w_d, act_idx_d, buffer_d, npwx, nact, .false., w_d  )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:     call gpu_threaded_assign( buffer1_d, w_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:     call gpu_threaded_backassign( hw_d, act_idx_d, buffer_d, npwx, nact, .false., hw_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:        call gpu_threaded_backassign( sw_d, act_idx_d, buffer_d, npwx, nact, .false., sw_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:        call gpu_threaded_memset( G_d, ZERO, nbnd*nact ) ! G(1:nact,1:nact) = ZERO
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:           call gpu_threaded_assign( buffer_d, spsi_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:           call gpu_threaded_assign( buffer_d,  psi_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:        call gpu_threaded_assign( buffer1_d,  p_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:        call gpu_threaded_assign( buffer_d,  p_d, npwx, nact, .true., act_idx_d, .true. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:        call gpu_threaded_assign( buffer1_d,  psi_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:        call gpu_threaded_backassign( p_d, act_idx_d, buffer_d, npwx, nact, .false., p_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:        call gpu_threaded_assign( buffer_d,  hp_d, npwx, nact, .true., act_idx_d, .true. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:        call gpu_threaded_assign( buffer1_d,  hpsi_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:        call gpu_threaded_backassign( hp_d, act_idx_d, buffer_d, npwx, nact, .false., hp_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:           call gpu_threaded_assign( buffer_d,  sp_d, npwx, nact, .true., act_idx_d, .true. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:           call gpu_threaded_assign( buffer1_d,  spsi_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:           call gpu_threaded_backassign( sp_d, act_idx_d, buffer_d, npwx, nact, .false., sp_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:        call gpu_threaded_assign( buffer_d,  psi_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:        call gpu_threaded_assign( buffer1_d,  hpsi_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:           call gpu_threaded_assign( buffer1_d,  spsi_d, npwx, l, .true., col_idx_d, .false.)
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:           call gpu_threaded_assign( buffer1_d,  buffer_d, npwx, l, .false., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:        call gpu_threaded_assign( buffer_d,  w_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:        call gpu_threaded_assign( buffer1_d,  hw_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:           call gpu_threaded_assign( buffer1_d,  sw_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:           call gpu_threaded_assign( buffer1_d,  buffer_d, npwx, l, .false., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:        call gpu_threaded_assign( buffer_d,  psi_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:        call gpu_threaded_assign( buffer1_d,  hw_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:           call gpu_threaded_assign( buffer1_d,  sw_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:           call gpu_threaded_assign( buffer1_d,  w_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_assign( buffer_d,  p_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_assign( buffer1_d,  hp_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:             call gpu_threaded_assign( buffer1_d,  sp_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:             call gpu_threaded_assign( buffer1_d,  buffer_d, npwx, l, .false., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_assign( buffer_d,  psi_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_assign( buffer1_d,  hp_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:             call gpu_threaded_assign( buffer1_d,  sp_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:             call gpu_threaded_assign( buffer1_d,  p_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_assign( buffer_d,  w_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_assign( buffer1_d,  hp_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:             call gpu_threaded_assign( buffer1_d,  sp_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:             call gpu_threaded_assign( buffer1_d,  p_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_assign( buffer1_d,  p_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_assign( buffer1_d,  w_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_backassign( p_d, col_idx_d, buffer_d, npwx, l, .false., p_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_assign( buffer1_d,  hp_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_assign( buffer1_d,  hw_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_backassign( hp_d, col_idx_d, buffer_d, npwx, l, .false., hp_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:             call gpu_threaded_assign( buffer1_d,  sp_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:             call gpu_threaded_assign( buffer1_d,  sw_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:             call gpu_threaded_backassign( sp_d, col_idx_d, buffer_d, npwx, l, .false., sp_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_assign( buffer1_d,  w_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_backassign( p_d, col_idx_d, buffer_d, npwx, l, .false., p_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_assign( buffer1_d,  hw_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_backassign( hp_d, col_idx_d, buffer_d, npwx, l, .false., hp_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:             call gpu_threaded_assign( buffer1_d,  sw_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:             call gpu_threaded_backassign( sp_d, col_idx_d, buffer_d, npwx, l, .false., sp_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:       call gpu_threaded_assign( buffer1_d,  psi_d, npwx, l, .true., col_idx_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:       call gpu_threaded_backassign( psi_d, col_idx_d, buffer_d, npwx, l, .true., p_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:       call gpu_threaded_assign( buffer1_d,  hpsi_d, npwx, l, .true., col_idx_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:       call gpu_threaded_backassign( hpsi_d, col_idx_d, buffer_d, npwx, l, .true., hp_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_assign( buffer1_d,  spsi_d, npwx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_backassign( spsi_d, col_idx_d, buffer_d, npwx, l, .true., sp_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:         call gpu_threaded_assign( buffer_d,  psi_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:            call gpu_threaded_assign( buffer1_d,  spsi_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:            call gpu_threaded_assign( buffer1_d,  buffer_d, npwx, nact, .false., act_idx_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:         call gpu_threaded_backassign( psi_d, act_idx_d, buffer_d, npwx, nact, .false., psi_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:         call gpu_threaded_assign( buffer1_d,  hpsi_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:         CALL gpu_dgemm_dmat( npw, nact, npwx, idesc, ONE, buffer1_d, Gl_d, ZERO, buffer_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:         call gpu_threaded_backassign( hpsi_d, act_idx_d, buffer_d, npwx, nact, .false., hpsi_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:            call gpu_threaded_assign( buffer1_d,  spsi_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:            CALL gpu_dgemm_dmat( npw, nact, npwx, idesc, ONE, buffer1_d, Gl_d, ZERO, buffer_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:            call gpu_threaded_backassign( spsi_d, act_idx_d, buffer_d, npwx, nact, .false., spsi_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:         call gpu_threaded_assign( buffer_d,  psi_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:            call gpu_threaded_assign( buffer1_d,  spsi_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:            call gpu_threaded_assign( buffer1_d,  buffer_d, npwx, nact, .false., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:         call gpu_threaded_backassign( psi_d, act_idx_d, buffer_d, npwx, nact, .false., psi_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:         call gpu_threaded_assign( buffer_d,  hpsi_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:         call gpu_threaded_backassign( hpsi_d, act_idx_d, buffer_d, npwx, nact, .false., hpsi_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:            call gpu_threaded_assign( buffer_d,  spsi_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:            call gpu_threaded_backassign( spsi_d, act_idx_d, buffer_d, npwx, nact, .false., spsi_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:       call gpu_threaded_assign( buffer_d,  psi_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:       call gpu_threaded_assign( buffer1_d,  hpsi_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:       call gpu_threaded_memset( G_d, ZERO, nbnd*nbnd ) ! G = ZERO
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:       call gpu_threaded_assign( buffer_d,  hpsi_d, npwx, nact, .true., act_idx_d, .true. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_assign( buffer1_d,  spsi_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:          call gpu_threaded_assign( buffer1_d,  psi_d, npwx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:       call gpu_threaded_backassign( w_d, act_idx_d, buffer_d, npwx, nact, .false., w_d )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:  SUBROUTINE gpu_dgemm_dmat( n, k, ld, idesc, alpha, X, Gl, beta, Y  )
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:#if defined(__CUDA)
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:#if defined(__CUDA)
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:  END SUBROUTINE gpu_dgemm_dmat
KS_Solvers/PPCG/ppcg_gamma_gpu.f90:END SUBROUTINE ppcg_gamma_gpu
KS_Solvers/PPCG/Makefile:ppcg_gamma_gpu.o \
KS_Solvers/PPCG/Makefile:ppcg_k_gpu.o \
KS_Solvers/PPCG/ppcg_k_gpu.f90:SUBROUTINE ppcg_k_gpu( h_psi_ptr, s_psi_ptr, overlap, precondition_d, &
KS_Solvers/PPCG/ppcg_k_gpu.f90:  ! IC gpu version
KS_Solvers/PPCG/ppcg_k_gpu.f90:#if defined(__CUDA)
KS_Solvers/PPCG/ppcg_k_gpu.f90:  USE cudafor
KS_Solvers/PPCG/ppcg_k_gpu.f90:  ! device arrays and variables for GPU computation
KS_Solvers/PPCG/ppcg_k_gpu.f90:#if defined(__CUDA)
KS_Solvers/PPCG/ppcg_k_gpu.f90:  CALL gpu_threaded_assign( w_d, hpsi_d, kdimx, nact, .false., act_idx_d, .true. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:     call gpu_threaded_assign( buffer_d, w_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:     call gpu_threaded_memset( G_d, C_ZERO, nbnd*nact ) ! G(1:nbnd,1:nact) = ZERO
KS_Solvers/PPCG/ppcg_k_gpu.f90:     call gpu_threaded_assign( buffer_d, w_d, kdimx, nact, .true., act_idx_d, .true. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:     call gpu_threaded_assign( buffer1_d, w_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:           call gpu_threaded_assign( buffer_d, spsi_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:           call gpu_threaded_assign( buffer_d,  psi_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:        call gpu_threaded_assign( buffer1_d, p_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:        call gpu_threaded_assign( buffer_d, p_d, kdimx, nact, .true., act_idx_d, .true. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:        call gpu_threaded_assign( buffer1_d, psi_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:        call gpu_threaded_assign( buffer_d, hp_d, kdimx, nact, .true., act_idx_d, .true. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:        call gpu_threaded_assign( buffer1_d, hpsi_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:           call gpu_threaded_assign( buffer_d, sp_d, kdimx, nact, .true., act_idx_d, .true. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:           call gpu_threaded_assign( buffer1_d, spsi_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:        call gpu_threaded_assign( buffer_d, psi_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:        call gpu_threaded_assign( buffer1_d, hpsi_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:           call gpu_threaded_assign( buffer1_d, spsi_d, kdimx, l, .true., col_idx_d, .false. ) 
KS_Solvers/PPCG/ppcg_k_gpu.f90:           call gpu_threaded_assign( buffer1_d, buffer_d, kdimx, l, .false., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:        call gpu_threaded_assign( buffer_d, w_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:        call gpu_threaded_assign( buffer1_d, hw_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:           call gpu_threaded_assign( buffer1_d, sw_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:           call gpu_threaded_assign( buffer1_d, buffer_d, kdimx, l, .false., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:        call gpu_threaded_assign( buffer_d, psi_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:        call gpu_threaded_assign( buffer1_d, hw_d, kdimx, l, .true., col_idx_d, .false. ) 
KS_Solvers/PPCG/ppcg_k_gpu.f90:           call gpu_threaded_assign( buffer1_d, sw_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:           call gpu_threaded_assign( buffer1_d,  w_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:          call gpu_threaded_assign( buffer_d,  p_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:          call gpu_threaded_assign( buffer1_d,  hp_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:             call gpu_threaded_assign( buffer1_d,  sp_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:             call gpu_threaded_assign( buffer1_d,  buffer_d, kdimx, l, .false., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:          call gpu_threaded_assign( buffer_d,  psi_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:          call gpu_threaded_assign( buffer1_d,  hp_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:             call gpu_threaded_assign( buffer1_d,  sp_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:             call gpu_threaded_assign( buffer1_d,  p_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:          call gpu_threaded_assign( buffer_d,  w_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:          call gpu_threaded_assign( buffer1_d,  hp_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:             call gpu_threaded_assign( buffer1_d,  sp_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:             call gpu_threaded_assign( buffer1_d,  p_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:          call gpu_threaded_assign( buffer1_d,  p_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:          call gpu_threaded_assign( buffer1_d,  w_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:          call gpu_threaded_assign( buffer1_d,  hp_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:          call gpu_threaded_assign( buffer1_d,  hw_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:             call gpu_threaded_assign( buffer1_d,  sp_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:             call gpu_threaded_assign( buffer1_d,  sw_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:          call gpu_threaded_assign( buffer1_d,  w_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:          call gpu_threaded_assign( buffer1_d,  hw_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:             call gpu_threaded_assign( buffer1_d,  sw_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:       call gpu_threaded_assign( buffer1_d,  psi_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:       call gpu_threaded_assign( buffer1_d,  hpsi_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:          call gpu_threaded_assign( buffer1_d,  spsi_d, kdimx, l, .true., col_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:         call gpu_threaded_assign( buffer_d,  psi_d, kdimx, nact, .true., act_idx_d, .false.)
KS_Solvers/PPCG/ppcg_k_gpu.f90:            call gpu_threaded_assign( buffer1_d,  spsi_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:            call gpu_threaded_assign( buffer1_d,  buffer_d, kdimx, nact, .false., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:         call gpu_threaded_assign( buffer1_d,  hpsi_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:         CALL gpu_zgemm_dmat( kdim, nact, kdimx, idesc, C_ONE, buffer1_d, Gl_d, C_ZERO, buffer_d )
KS_Solvers/PPCG/ppcg_k_gpu.f90:            call gpu_threaded_assign( buffer1_d,  spsi_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:            CALL gpu_zgemm_dmat( kdim, nact, kdimx, idesc, C_ONE, buffer1_d, Gl_d, C_ZERO, buffer_d )
KS_Solvers/PPCG/ppcg_k_gpu.f90:         call gpu_threaded_assign( buffer_d,  psi_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:            call gpu_threaded_assign( buffer1_d,  spsi_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:            call gpu_threaded_assign( buffer1_d,  buffer_d, kdimx, nact, .false., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:         call gpu_threaded_assign( buffer_d,  hpsi_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:            call gpu_threaded_assign( buffer_d,  spsi_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:       call gpu_threaded_assign( buffer_d,  psi_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:       call gpu_threaded_assign( buffer1_d,  hpsi_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:       call gpu_threaded_assign( buffer_d,  hpsi_d, kdimx, nact, .true., act_idx_d, .true. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:          call gpu_threaded_assign( buffer1_d,  spsi_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:          call gpu_threaded_assign( buffer1_d,   psi_d, kdimx, nact, .true., act_idx_d, .false. )
KS_Solvers/PPCG/ppcg_k_gpu.f90:  SUBROUTINE gpu_zgemm_dmat( n, k, ld, idesc, alpha, X, Gl, beta, Y  )
KS_Solvers/PPCG/ppcg_k_gpu.f90:#if defined(__CUDA)
KS_Solvers/PPCG/ppcg_k_gpu.f90:#if defined(__CUDA)
KS_Solvers/PPCG/ppcg_k_gpu.f90:  END SUBROUTINE gpu_zgemm_dmat
KS_Solvers/PPCG/ppcg_k_gpu.f90:! gpu dmat end
KS_Solvers/PPCG/ppcg_k_gpu.f90:END SUBROUTINE ppcg_k_gpu
KS_Solvers/PPCG/generic_cublas.f90:SUBROUTINE gpu_threaded_memset(array, val, length)
KS_Solvers/PPCG/generic_cublas.f90:#if defined(__CUDA)
KS_Solvers/PPCG/generic_cublas.f90:  USE cudafor
KS_Solvers/PPCG/generic_cublas.f90:#if defined(__CUDA)
KS_Solvers/PPCG/generic_cublas.f90:END SUBROUTINE gpu_threaded_memset
KS_Solvers/PPCG/generic_cublas.f90:SUBROUTINE gpu_threaded_assign(array_out, array_in, kdimx, nact, use_idx, idx, bgrp_root_only)
KS_Solvers/PPCG/generic_cublas.f90:#if defined(__CUDA)
KS_Solvers/PPCG/generic_cublas.f90:  USE cudafor
KS_Solvers/PPCG/generic_cublas.f90:#if defined(__CUDA)
KS_Solvers/PPCG/generic_cublas.f90:END SUBROUTINE gpu_threaded_assign
KS_Solvers/PPCG/generic_cublas.f90:SUBROUTINE gpu_threaded_backassign(array_out, idx, array_in, kdimx, nact, use_a2, a2_in )
KS_Solvers/PPCG/generic_cublas.f90:#if defined(__CUDA)
KS_Solvers/PPCG/generic_cublas.f90:  USE cudafor
KS_Solvers/PPCG/generic_cublas.f90:#if defined(__CUDA)
KS_Solvers/PPCG/generic_cublas.f90:END SUBROUTINE gpu_threaded_backassign
KS_Solvers/ks_solver_interfaces.h:#if defined (__CUDA) 
KS_Solvers/ks_solver_interfaces.h:  !! Interface for the CUDA-Fortran case. 
KS_Solvers/CG/Makefile:rcgdiagg_gpu.o \
KS_Solvers/CG/Makefile:ccgdiagg_gpu.o
KS_Solvers/CG/rcgdiagg_gpu.f90:SUBROUTINE rcgdiagg_gpu( hs_1psi_ptr, s_1psi_ptr, precondition, &
KS_Solvers/CG/rcgdiagg_gpu.f90:#if defined(__CUDA)
KS_Solvers/CG/rcgdiagg_gpu.f90:  USE cudafor
KS_Solvers/CG/rcgdiagg_gpu.f90:#if defined(__CUDA)
KS_Solvers/CG/rcgdiagg_gpu.f90:END SUBROUTINE rcgdiagg_gpu
KS_Solvers/CG/ccgdiagg_gpu.f90:SUBROUTINE ccgdiagg_gpu( hs_1psi_ptr, s_1psi_ptr, precondition, &
KS_Solvers/CG/ccgdiagg_gpu.f90:#if defined(__CUDA)
KS_Solvers/CG/ccgdiagg_gpu.f90:  USE cudafor
KS_Solvers/CG/ccgdiagg_gpu.f90:#if defined(__CUDA)
KS_Solvers/CG/ccgdiagg_gpu.f90:END SUBROUTINE ccgdiagg_gpu
KS_Solvers/DENSE/rotate_xpsi_gamma_gpu.f90:SUBROUTINE rotate_xpsi_gamma_gpu( h_psi_ptr, s_psi_ptr, overlap, &
KS_Solvers/DENSE/rotate_xpsi_gamma_gpu.f90:#if defined(__CUDA)
KS_Solvers/DENSE/rotate_xpsi_gamma_gpu.f90:  USE cudafor
KS_Solvers/DENSE/rotate_xpsi_gamma_gpu.f90:#if defined(__CUDA)
KS_Solvers/DENSE/rotate_xpsi_gamma_gpu.f90:#if defined(__CUDA)
KS_Solvers/DENSE/rotate_xpsi_gamma_gpu.f90:END SUBROUTINE rotate_xpsi_gamma_gpu
KS_Solvers/DENSE/gram_schmidt_k_gpu.f90:SUBROUTINE gram_schmidt_k_gpu( npwx, npw, nbnd, npol, psi_d, hpsi_d, spsi_d, e, &
KS_Solvers/DENSE/gram_schmidt_k_gpu.f90:#if defined (__CUDA)
KS_Solvers/DENSE/gram_schmidt_k_gpu.f90:#if defined (__CUDA)
KS_Solvers/DENSE/gram_schmidt_k_gpu.f90:     CALL project_offdiag_gpu( ibnd_start, ibnd_end, jbnd_start, jbnd_end )
KS_Solvers/DENSE/gram_schmidt_k_gpu.f90:  IF ( eigen_ ) CALL energyeigen_gpu( )
KS_Solvers/DENSE/gram_schmidt_k_gpu.f90:  IF ( reorder ) CALL sort_vectors_gpu( )
KS_Solvers/DENSE/gram_schmidt_k_gpu.f90:  SUBROUTINE project_offdiag_gpu( ibnd_start, ibnd_end, jbnd_start, jbnd_end )
KS_Solvers/DENSE/gram_schmidt_k_gpu.f90:  END SUBROUTINE project_offdiag_gpu
KS_Solvers/DENSE/gram_schmidt_k_gpu.f90:  SUBROUTINE energyeigen_gpu( )
KS_Solvers/DENSE/gram_schmidt_k_gpu.f90:  END SUBROUTINE energyeigen_gpu
KS_Solvers/DENSE/gram_schmidt_k_gpu.f90:  SUBROUTINE sort_vectors_gpu( )
KS_Solvers/DENSE/gram_schmidt_k_gpu.f90:  END SUBROUTINE sort_vectors_gpu
KS_Solvers/DENSE/gram_schmidt_k_gpu.f90:END SUBROUTINE gram_schmidt_k_gpu
KS_Solvers/DENSE/rotate_driver_cuf.f90:#if defined(__CUDA)
KS_Solvers/DENSE/rotate_driver_cuf.f90:  CALL start_clock_gpu( 'wfcrot' ); !write (*,*) 'start wfcrot' ; FLUSH(6)
KS_Solvers/DENSE/rotate_driver_cuf.f90:        CALL rotate_xpsi_gamma_gpu ( h_psi_dptr, s_psi_dptr, overlap, &
KS_Solvers/DENSE/rotate_driver_cuf.f90:        CALL rotate_xpsi_k_gpu ( h_psi_dptr, s_psi_dptr, overlap, &
KS_Solvers/DENSE/rotate_driver_cuf.f90:  CALL stop_clock_gpu( 'wfcrot' )
KS_Solvers/DENSE/rotate_xpsi_k_gpu.f90:SUBROUTINE rotate_xpsi_k_gpu( h_psi_ptr, s_psi_ptr, overlap, &
KS_Solvers/DENSE/rotate_xpsi_k_gpu.f90:#if defined(__CUDA)
KS_Solvers/DENSE/rotate_xpsi_k_gpu.f90:  USE cudafor
KS_Solvers/DENSE/rotate_xpsi_k_gpu.f90:#if defined(__CUDA)
KS_Solvers/DENSE/rotate_xpsi_k_gpu.f90:#if defined(__CUDA)
KS_Solvers/DENSE/rotate_xpsi_k_gpu.f90:END SUBROUTINE rotate_xpsi_k_gpu
KS_Solvers/DENSE/Makefile:rotate_wfc_gamma_gpu.o \
KS_Solvers/DENSE/Makefile:rotate_wfc_k_gpu.o \
KS_Solvers/DENSE/Makefile:rotate_xpsi_k_gpu.o \
KS_Solvers/DENSE/Makefile:rotate_xpsi_gamma_gpu.o \
KS_Solvers/DENSE/Makefile:gram_schmidt_gamma_gpu.o \
KS_Solvers/DENSE/Makefile:gram_schmidt_k_gpu.o  
KS_Solvers/DENSE/gram_schmidt_gamma_gpu.f90:SUBROUTINE gram_schmidt_gamma_gpu( npwx, npw, nbnd, psi_d, hpsi_d, spsi_d, e, &
KS_Solvers/DENSE/gram_schmidt_gamma_gpu.f90:#if defined (__CUDA)
KS_Solvers/DENSE/gram_schmidt_gamma_gpu.f90:#if defined (__CUDA)
KS_Solvers/DENSE/gram_schmidt_gamma_gpu.f90:     CALL gram_schmidt_diag_gpu( ibnd_start, ibnd_end )
KS_Solvers/DENSE/gram_schmidt_gamma_gpu.f90:     CALL project_offdiag_gpu( ibnd_start, ibnd_end, jbnd_start, jbnd_end )
KS_Solvers/DENSE/gram_schmidt_gamma_gpu.f90: IF ( eigen_ ) CALL energyeigen_gpu( )
KS_Solvers/DENSE/gram_schmidt_gamma_gpu.f90:  IF ( reorder ) CALL sort_vectors_gpu( )
KS_Solvers/DENSE/gram_schmidt_gamma_gpu.f90:  SUBROUTINE gram_schmidt_diag_gpu( ibnd_start, ibnd_end )
KS_Solvers/DENSE/gram_schmidt_gamma_gpu.f90:  END SUBROUTINE gram_schmidt_diag_gpu
KS_Solvers/DENSE/gram_schmidt_gamma_gpu.f90:  SUBROUTINE project_offdiag_gpu( ibnd_start, ibnd_end, jbnd_start, jbnd_end )
KS_Solvers/DENSE/gram_schmidt_gamma_gpu.f90:  END SUBROUTINE project_offdiag_gpu
KS_Solvers/DENSE/gram_schmidt_gamma_gpu.f90:  SUBROUTINE energyeigen_gpu( )
KS_Solvers/DENSE/gram_schmidt_gamma_gpu.f90:  END SUBROUTINE energyeigen_gpu
KS_Solvers/DENSE/gram_schmidt_gamma_gpu.f90:  SUBROUTINE sort_vectors_gpu( )
KS_Solvers/DENSE/gram_schmidt_gamma_gpu.f90:  END SUBROUTINE sort_vectors_gpu
KS_Solvers/DENSE/gram_schmidt_gamma_gpu.f90:END SUBROUTINE gram_schmidt_gamma_gpu
KS_Solvers/DENSE/rotate_wfc_k_gpu.f90:SUBROUTINE rotate_wfc_k_gpu( h_psi_ptr, s_psi_ptr, overlap, &
KS_Solvers/DENSE/rotate_wfc_k_gpu.f90:#if defined(__CUDA)
KS_Solvers/DENSE/rotate_wfc_k_gpu.f90:  USE cudafor
KS_Solvers/DENSE/rotate_wfc_k_gpu.f90:#if defined(__CUDA)
KS_Solvers/DENSE/rotate_wfc_k_gpu.f90:#if defined(__CUDA)
KS_Solvers/DENSE/rotate_wfc_k_gpu.f90:  !! cdiaghg on GPU. See interface from LAXlib module
KS_Solvers/DENSE/rotate_wfc_k_gpu.f90:END SUBROUTINE rotate_wfc_k_gpu
KS_Solvers/DENSE/rotate_wfc_gamma_gpu.f90:SUBROUTINE rotate_wfc_gamma_gpu( h_psi_ptr, s_psi_ptr, overlap, &
KS_Solvers/DENSE/rotate_wfc_gamma_gpu.f90:#if defined(__CUDA)
KS_Solvers/DENSE/rotate_wfc_gamma_gpu.f90:  USE cudafor
KS_Solvers/DENSE/rotate_wfc_gamma_gpu.f90:#if defined(__CUDA)
KS_Solvers/DENSE/rotate_wfc_gamma_gpu.f90:#if defined(__CUDA)
KS_Solvers/DENSE/rotate_wfc_gamma_gpu.f90:END SUBROUTINE rotate_wfc_gamma_gpu
KS_Solvers/Makefile:DENSE/gram_schmidt_k_gpu.o \
KS_Solvers/Makefile:DENSE/gram_schmidt_gamma_gpu.o \
KS_Solvers/Makefile:DENSE/rotate_xpsi_gamma_gpu.o \
KS_Solvers/Makefile:DENSE/rotate_xpsi_k_gpu.o \
KS_Solvers/Makefile:RMM/crmmdiagg_gpu.o \
KS_Solvers/Makefile:RMM/rrmmdiagg_gpu.o 
KS_Solvers/Makefile:# GPU-related objects
KS_Solvers/Makefile:Davidson/cegterg_gpu.o \
KS_Solvers/Makefile:Davidson/regterg_gpu.o \
KS_Solvers/Makefile:DENSE/rotate_wfc_k_gpu.o \
KS_Solvers/Makefile:DENSE/rotate_wfc_gamma_gpu.o \
KS_Solvers/Makefile:CG/rcgdiagg_gpu.o \
KS_Solvers/Makefile:CG/ccgdiagg_gpu.o \
KS_Solvers/Makefile:PPCG/ppcg_gamma_gpu.o \
KS_Solvers/Makefile:PPCG/ppcg_k_gpu.o 
KS_Solvers/CMakeLists.txt:    # GPU
KS_Solvers/CMakeLists.txt:    Davidson/regterg_gpu.f90
KS_Solvers/CMakeLists.txt:    Davidson/cegterg_gpu.f90)
KS_Solvers/CMakeLists.txt:qe_enable_cuda_fortran("${src_davidson}")
KS_Solvers/CMakeLists.txt:    # GPU
KS_Solvers/CMakeLists.txt:    CG/rcgdiagg_gpu.f90
KS_Solvers/CMakeLists.txt:    CG/ccgdiagg_gpu.f90)
KS_Solvers/CMakeLists.txt:qe_enable_cuda_fortran("${src_cg}")
KS_Solvers/CMakeLists.txt:    # GPU
KS_Solvers/CMakeLists.txt:    PPCG/ppcg_gamma_gpu.f90
KS_Solvers/CMakeLists.txt:    PPCG/ppcg_k_gpu.f90
KS_Solvers/CMakeLists.txt:qe_enable_cuda_fortran("${src_ppcg}")
KS_Solvers/CMakeLists.txt:    # GPU
KS_Solvers/CMakeLists.txt:    DENSE/rotate_wfc_gamma_gpu.f90
KS_Solvers/CMakeLists.txt:    DENSE/rotate_xpsi_k_gpu.f90
KS_Solvers/CMakeLists.txt:    DENSE/rotate_xpsi_gamma_gpu.f90
KS_Solvers/CMakeLists.txt:    DENSE/gram_schmidt_k_gpu.f90
KS_Solvers/CMakeLists.txt:    DENSE/gram_schmidt_gamma_gpu.f90
KS_Solvers/CMakeLists.txt:    DENSE/rotate_wfc_k_gpu.f90
KS_Solvers/CMakeLists.txt:qe_enable_cuda_fortran("${src_dense}")
KS_Solvers/CMakeLists.txt:qe_enable_cuda_fortran("${src_paro}")
KS_Solvers/CMakeLists.txt:    # GPU
KS_Solvers/CMakeLists.txt:    RMM/crmmdiagg_gpu.f90
KS_Solvers/CMakeLists.txt:    RMM/rrmmdiagg_gpu.f90)
KS_Solvers/CMakeLists.txt:    qe_enable_cuda_fortran("${src_rmmdiis}")
KS_Solvers/CMakeLists.txt:        qe_openacc_fortran
KS_Solvers/CMakeLists.txt:        qe_openacc_fortran
KS_Solvers/CMakeLists.txt:        qe_openacc_fortran
KS_Solvers/CMakeLists.txt:        qe_openacc_fortran
KS_Solvers/CMakeLists.txt:        qe_openacc_fortran
KS_Solvers/CMakeLists.txt:        qe_openacc_fortran
KS_Solvers/ParO/bpcg_gamma.f90:! * GPU porting Ivan Carnimeo
KS_Solvers/ParO/bpcg_gamma.f90:  INTEGER :: cg_iter_l ! cg_iter(l) (useful for some GPU optimization)
KS_Solvers/ParO/bpcg_gamma.f90:  REAL(DP), EXTERNAL :: MYDDOT_VECTOR_GPU
KS_Solvers/ParO/bpcg_gamma.f90:  !$acc routine(MYDDOT_VECTOR_GPU) vector
KS_Solvers/ParO/bpcg_gamma.f90:           g0(l) = 2.D0*MYDDOT_VECTOR_GPU(npw2,z(:,l),b(:,l))
KS_Solvers/ParO/bpcg_gamma.f90:        gamma(l) = 2.D0*MYDDOT_VECTOR_GPU(npw2,p(:,l),hp(:,l)) &
KS_Solvers/ParO/bpcg_gamma.f90:                       - e(i) * 2.D0*MYDDOT_VECTOR_GPU(npw2,p(:,l),sp(:,l)) 
KS_Solvers/ParO/bpcg_gamma.f90:        g2(l) = 2.D0 * ( MYDDOT_VECTOR_GPU(npw2,z(:,l),b(:,l)) &
KS_Solvers/ParO/bpcg_gamma.f90:                + e(i) * MYDDOT_VECTOR_GPU(npw2,z(:,l),spsi(:,i)) &
KS_Solvers/ParO/bpcg_gamma.f90:                - MYDDOT_VECTOR_GPU(npw2,z(:,l),hpsi(:,i)) )
KS_Solvers/ParO/bpcg_gamma.f90:        g1(l) = 2.D0 * ( MYDDOT_VECTOR_GPU(npw2,z(:,l),b(:,l)) &
KS_Solvers/ParO/bpcg_gamma.f90:                + e(i) * MYDDOT_VECTOR_GPU(npw2,z(:,l),spsi(:,i)) &
KS_Solvers/ParO/bpcg_gamma.f90:                       - MYDDOT_VECTOR_GPU(npw2,z(:,l),hpsi(:,i)) )
KS_Solvers/ParO/bpcg_gamma.f90:        ff(l) = - ( e(i)*MYDDOT_VECTOR_GPU(npw2,psi(:,i),spsi(:,i)) &
KS_Solvers/ParO/bpcg_gamma.f90:                        -MYDDOT_VECTOR_GPU(npw2,psi(:,i),hpsi(:,i)) ) &
KS_Solvers/ParO/bpcg_gamma.f90:                - 2.D0 * MYDDOT_VECTOR_GPU(npw2,psi(:,i),b(:,l))
KS_Solvers/ParO/paro_k_new.f90:! GPU porting by Ivan Carnimeo
KS_Solvers/ParO/paro_k_new.f90:!   paro_k_new and paro_gamma_new have been ported to GPU with OpenACC, 
KS_Solvers/ParO/paro_k_new.f90:!   the previous CUF versions (paro_k_new_gpu and paro_gamma_new_gpu) have been removed, 
KS_Solvers/ParO/paro_k_new.f90:!   and now paro_k_new and paro_gamma_new are used for both CPU and GPU execution.
KS_Solvers/ParO/paro_k_new.f90:#if defined(__CUDA)
KS_Solvers/ParO/paro_k_new.f90:     Call errore('paro_k_new','nproc_ortho /= 1 with gpu NYI', 1)
KS_Solvers/ParO/paro_k_new.f90:#if defined(__CUDA)
KS_Solvers/ParO/paro_k_new.f90:       Call errore('paro_k_new','nproc_ortho /= 1 with gpu NYI', 2)
KS_Solvers/ParO/bpcg_k.f90:! * GPU porting Ivan Carnimeo
KS_Solvers/ParO/bpcg_k.f90:  INTEGER :: cg_iter_l ! cg_iter(l) (useful for some GPU optimization)
KS_Solvers/ParO/bpcg_k.f90:  REAL(DP), EXTERNAL :: MYDDOT_VECTOR_GPU
KS_Solvers/ParO/bpcg_k.f90:  !$acc routine(MYDDOT_VECTOR_GPU) vector
KS_Solvers/ParO/bpcg_k.f90:           g0(l) = MYDDOT_VECTOR_GPU( 2*kdim, z(:,l), b(:,l) )
KS_Solvers/ParO/bpcg_k.f90:        gamma(l) = MYDDOT_VECTOR_GPU( 2*kdim, p(:,l), hp(:,l) ) &
KS_Solvers/ParO/bpcg_k.f90:                       - e(i) * MYDDOT_VECTOR_GPU( 2*kdim, p(:,l), sp(:,l) )
KS_Solvers/ParO/bpcg_k.f90:        g2(l) = MYDDOT_VECTOR_GPU(2*kdim,z(:,l),b(:,l)) &
KS_Solvers/ParO/bpcg_k.f90:                + e(i) * MYDDOT_VECTOR_GPU(2*kdim,z(:,l),spsi(:,i)) &
KS_Solvers/ParO/bpcg_k.f90:                - MYDDOT_VECTOR_GPU(2*kdim,z(:,l),hpsi(:,i))
KS_Solvers/ParO/bpcg_k.f90:        g1(l) = MYDDOT_VECTOR_GPU(2*kdim,z(:,l),b(:,l)) &
KS_Solvers/ParO/bpcg_k.f90:                + e(i) * MYDDOT_VECTOR_GPU(2*kdim,z(:,l),spsi(:,i)) &
KS_Solvers/ParO/bpcg_k.f90:                - MYDDOT_VECTOR_GPU(2*kdim,z(:,l),hpsi(:,i))
KS_Solvers/ParO/bpcg_k.f90:        ff(l) = -0.5_DP * ( e(i)*MYDDOT_VECTOR_GPU(2*kdim,psi(:,i),spsi(:,i))&
KS_Solvers/ParO/bpcg_k.f90:                                -MYDDOT_VECTOR_GPU(2*kdim,psi(:,i),hpsi(:,i)) ) &
KS_Solvers/ParO/bpcg_k.f90:                - MYDDOT_VECTOR_GPU(2*kdim,psi(:,i),b(:,l))
KS_Solvers/ParO/paro_gamma_new.f90:! GPU porting by Ivan Carnimeo
KS_Solvers/ParO/paro_gamma_new.f90:!   paro_k_new and paro_gamma_new have been ported to GPU with OpenACC, 
KS_Solvers/ParO/paro_gamma_new.f90:!   the previous CUF versions (paro_k_new_gpu and paro_gamma_new_gpu) have been removed, 
KS_Solvers/ParO/paro_gamma_new.f90:!   and now paro_k_new and paro_gamma_new are used for both CPU and GPU execution.
KS_Solvers/ParO/paro_gamma_new.f90:#if defined(__CUDA)
KS_Solvers/ParO/paro_gamma_new.f90:     Call errore('paro_gamma_new','nproc_ortho /= 1 with gpu NYI', 1)
KS_Solvers/ParO/paro_gamma_new.f90:#if defined(__CUDA)
KS_Solvers/ParO/paro_gamma_new.f90:       Call errore('paro_gamma_new','nproc_ortho /= 1 with gpu NYI', 2)
KS_Solvers/Davidson/regterg_gpu.f90:!   cegterg and regterg have been ported to GPU with OpenACC, 
KS_Solvers/Davidson/regterg_gpu.f90:!   the previous CUF versions (cegterg_gpu and regterg_gpu) have been removed, 
KS_Solvers/Davidson/regterg_gpu.f90:!   and now cegterg and regterg are used for both CPU and GPU execution.
KS_Solvers/Davidson/regterg_gpu.f90:#if !defined(__CUDA)
KS_Solvers/Davidson/regterg_gpu.f90:! workaround for some old compilers that don't like CUDA fortran code
KS_Solvers/Davidson/regterg_gpu.f90:SUBROUTINE pregterg_gpu( )
KS_Solvers/Davidson/regterg_gpu.f90:end SUBROUTINE pregterg_gpu
KS_Solvers/Davidson/regterg_gpu.f90:SUBROUTINE pregterg_gpu(h_psi_ptr, s_psi_ptr, uspp, g_psi_ptr, &  
KS_Solvers/Davidson/regterg_gpu.f90:#if defined(__CUDA)
KS_Solvers/Davidson/regterg_gpu.f90:#if defined(__CUDA)
KS_Solvers/Davidson/regterg_gpu.f90:#if defined(__CUDA)
KS_Solvers/Davidson/regterg_gpu.f90:#if defined(__CUDA)
KS_Solvers/Davidson/regterg_gpu.f90:END SUBROUTINE pregterg_gpu
KS_Solvers/Davidson/cegterg.f90:!   cegterg and regterg have been ported to GPU with OpenACC, 
KS_Solvers/Davidson/cegterg.f90:!   the previous CUF versions (cegterg_gpu and regterg_gpu) have been removed, 
KS_Solvers/Davidson/cegterg.f90:!   and now cegterg and regterg are used for both CPU and GPU execution.
KS_Solvers/Davidson/cegterg.f90:#if defined(__CUDA)
KS_Solvers/Davidson/cegterg.f90:  REAL(DP), EXTERNAL :: MYDDOT_VECTOR_GPU
KS_Solvers/Davidson/cegterg.f90:  !$acc routine(MYDDOT_VECTOR_GPU) vector
KS_Solvers/Davidson/cegterg.f90:#if ! defined(__CUDA)
KS_Solvers/Davidson/cegterg.f90:#if defined(__CUDA)
KS_Solvers/Davidson/cegterg.f90:#if defined(__CUDA)
KS_Solvers/Davidson/cegterg.f90:#if defined(__CUDA)
KS_Solvers/Davidson/cegterg.f90:        ew(n) = MYDDOT_VECTOR_GPU( 2*npw, psi(1,nbn), psi(1,nbn) )
KS_Solvers/Davidson/cegterg.f90:         ew(n) = ew(n)  + MYDDOT_VECTOR_GPU( 2*npw, psi(npwx+1,nbn), psi(npwx+1,nbn) ) 
KS_Solvers/Davidson/cegterg.f90:#if defined(__CUDA)
KS_Solvers/Davidson/cegterg.f90:#if defined(__CUDA)
KS_Solvers/Davidson/cegterg.f90:#if defined(__CUDA)
KS_Solvers/Davidson/Makefile:cegterg_gpu.o \
KS_Solvers/Davidson/Makefile:regterg_gpu.o
KS_Solvers/Davidson/cegterg_gpu.f90:!   cegterg and regterg have been ported to GPU with OpenACC, 
KS_Solvers/Davidson/cegterg_gpu.f90:!   the previous CUF versions (cegterg_gpu and regterg_gpu) have been removed, 
KS_Solvers/Davidson/cegterg_gpu.f90:!   and now cegterg and regterg are used for both CPU and GPU execution.
KS_Solvers/Davidson/cegterg_gpu.f90:#if ! defined(__CUDA)
KS_Solvers/Davidson/cegterg_gpu.f90:! workaround for some old compilers that don't like CUDA fortran code
KS_Solvers/Davidson/cegterg_gpu.f90:SUBROUTINE pcegterg_gpu( )
KS_Solvers/Davidson/cegterg_gpu.f90:end SUBROUTINE pcegterg_gpu
KS_Solvers/Davidson/cegterg_gpu.f90:SUBROUTINE pcegterg_gpu(h_psi_ptr, s_psi_ptr, uspp, g_psi_ptr, &  
KS_Solvers/Davidson/cegterg_gpu.f90:#if defined(__CUDA)
KS_Solvers/Davidson/cegterg_gpu.f90:#if defined(__CUDA)
KS_Solvers/Davidson/cegterg_gpu.f90:#if defined(__CUDA)
KS_Solvers/Davidson/cegterg_gpu.f90:#if defined(__CUDA)
KS_Solvers/Davidson/cegterg_gpu.f90:END SUBROUTINE pcegterg_gpu
KS_Solvers/Davidson/regterg.f90:!   cegterg and regterg have been ported to GPU with OpenACC, 
KS_Solvers/Davidson/regterg.f90:!   the previous CUF versions (cegterg_gpu and regterg_gpu) have been removed, 
KS_Solvers/Davidson/regterg.f90:!   and now cegterg and regterg are used for both CPU and GPU execution.
KS_Solvers/Davidson/regterg.f90:#if defined(__CUDA)
KS_Solvers/Davidson/regterg.f90:  REAL(DP), EXTERNAL :: MYDDOT_VECTOR_GPU 
KS_Solvers/Davidson/regterg.f90:  !$acc routine(MYDDOT_VECTOR_GPU) vector
KS_Solvers/Davidson/regterg.f90:        ew(n) = 2.D0 * MYDDOT_VECTOR_GPU( npw2, psi(1,nbn), psi(1,nbn) )
KS_Solvers/RMM/Makefile:crmmdiagg_gpu.o \
KS_Solvers/RMM/Makefile:rrmmdiagg_gpu.o
KS_Solvers/RMM/crmmdiagg_gpu.f90:SUBROUTINE crmmdiagg_gpu( h_psi_ptr, s_psi_ptr, npwx, npw, nbnd, npol, psi, hpsi, spsi, e, &
KS_Solvers/RMM/crmmdiagg_gpu.f90:#if defined(__CUDA)
KS_Solvers/RMM/crmmdiagg_gpu.f90:     CALL calc_hpsi_k_gpu( )
KS_Solvers/RMM/crmmdiagg_gpu.f90:     CALL do_diis_gpu( idiis )
KS_Solvers/RMM/crmmdiagg_gpu.f90:     CALL cr_line_search_gpu( )
KS_Solvers/RMM/crmmdiagg_gpu.f90:  SUBROUTINE calc_hpsi_k_gpu( )
KS_Solvers/RMM/crmmdiagg_gpu.f90:  END SUBROUTINE calc_hpsi_k_gpu
KS_Solvers/RMM/crmmdiagg_gpu.f90:  SUBROUTINE do_diis_gpu( idiis )
KS_Solvers/RMM/crmmdiagg_gpu.f90:#if defined (__CUDA)
KS_Solvers/RMM/crmmdiagg_gpu.f90:  END SUBROUTINE do_diis_gpu
KS_Solvers/RMM/crmmdiagg_gpu.f90:  SUBROUTINE cr_line_search_gpu( )
KS_Solvers/RMM/crmmdiagg_gpu.f90:  END SUBROUTINE cr_line_search_gpu
KS_Solvers/RMM/crmmdiagg_gpu.f90:END SUBROUTINE crmmdiagg_gpu
KS_Solvers/RMM/rrmmdiagg_gpu.f90:SUBROUTINE rrmmdiagg_gpu( h_psi_ptr, s_psi_ptr, npwx, npw, nbnd, psi, hpsi, spsi, e, &
KS_Solvers/RMM/rrmmdiagg_gpu.f90:#if defined(__CUDA)
KS_Solvers/RMM/rrmmdiagg_gpu.f90:     CALL calc_hpsi_gamma_gpu( )
KS_Solvers/RMM/rrmmdiagg_gpu.f90:     CALL do_diis_gpu( idiis )
KS_Solvers/RMM/rrmmdiagg_gpu.f90:     CALL rr_line_search_gpu( )
KS_Solvers/RMM/rrmmdiagg_gpu.f90:  SUBROUTINE calc_hpsi_gamma_gpu( )
KS_Solvers/RMM/rrmmdiagg_gpu.f90:  END SUBROUTINE calc_hpsi_gamma_gpu
KS_Solvers/RMM/rrmmdiagg_gpu.f90:  SUBROUTINE do_diis_gpu( idiis )
KS_Solvers/RMM/rrmmdiagg_gpu.f90:#if defined (__CUDA) 
KS_Solvers/RMM/rrmmdiagg_gpu.f90:  END SUBROUTINE do_diis_gpu
KS_Solvers/RMM/rrmmdiagg_gpu.f90:  SUBROUTINE rr_line_search_gpu( )
KS_Solvers/RMM/rrmmdiagg_gpu.f90:  END SUBROUTINE rr_line_search_gpu
KS_Solvers/RMM/rrmmdiagg_gpu.f90:END SUBROUTINE rrmmdiagg_gpu

```
