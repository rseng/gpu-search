# https://github.com/nwchemgit/nwchem

```console
travis/build_env.sh:		wget -nv https://github.com/ROCm-Developer-Tools/aomp/releases/download/rel_"$aomp_major"."$aomp_minor"/aomp_Ubuntu2004_"$aomp_major"."$aomp_minor"_amd64.deb
travis/build_env.sh:	    rocm_version=5.6.1
travis/build_env.sh:	    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key |  $MYSUDO apt-key add - \
travis/build_env.sh:	    echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/'$rocm_version'/ ubuntu main' | $MYSUDO tee /etc/apt/sources.list.d/rocm.list
travis/build_env.sh:	    $MYSUDO apt-get  update -y && $MYSUDO apt-get -y install rocm-llvm openmp-extras \
travis/build_env.sh:	    export PATH=/opt/rocm/bin:$PATH
travis/build_env.sh:	    export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/llvm/lib:$LD_LIBRARY_PATH
travis/build_env.sh:	    curl https://developer.download.nvidia.com/hpc-sdk/ubuntu/DEB-GPG-KEY-NVIDIA-HPC-SDK | sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg
travis/build_env.sh:            echo 'deb [signed-by=/usr/share/keyrings/nvidia-hpcsdk-archive-keyring.gpg] https://developer.download.nvidia.com/hpc-sdk/ubuntu/'$arch_dpkg' /' | sudo tee /etc/apt/sources.list.d/nvhpc.list
travis/build_env.sh:	    export PATH=/opt/nvidia/hpc_sdk/Linux_"$arch"/"$nverdot"/compilers/bin:$PATH
travis/build_env.sh:	    export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_"$arch"/"$nverdot"/compilers/lib:$LD_LIBRARY_PATH
travis/build_env.sh:	    $MYSUDO /opt/nvidia/hpc_sdk/Linux_"$arch"/"$nverdot"/compilers/bin/makelocalrc -x
travis/build_env.sh:	    $MYSUDO rm -rf /opt/nvidia/hpc_sdk/Linux_"$arch"/"$nverdot"/profilers
travis/build_env.sh:	    $MYSUDO rm -rf /opt/nvidia/hpc_sdk/Linux_"$arch"/"$nverdot"/comm_libs
travis/build_env.sh:	    $MYSUDO rm -rf /opt/nvidia/hpc_sdk/Linux_"$arch"/"$nverdot"/math_libs
travis/nwchem.bashrc:    export PATH=/opt/rocm/bin:$PATH
travis/nwchem.bashrc:    export LD_LIBRARY_PATH=/opt/rocm-"$rocm_version"/lib:/opt/rocm/llvm/lib:$LD_LIBRARY_PATH
travis/nwchem.bashrc:     export PATH=/opt/nvidia/hpc_sdk/Linux_"$arch"/"$nverdot"/compilers/bin:$PATH
travis/nwchem.bashrc:     export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_"$arch"/"$nverdot"/compilers/lib:$LD_LIBRARY_PATH
travis/nwchem.bashrc:     sudo /opt/nvidia/hpc_sdk/Linux_"$arch"/"$nverdot"/compilers/bin/makelocalrc -x
contrib/pbs/pbsnw:  set NPROCMEM = "${NPROC}:bigmem,mem=512mb"
contrib/pbs/pbsnw:  set NPROCMEM = "${NPROC},mem=128mb"
contrib/pbs/pbsnw:#PBS -l nodes=${NPROCMEM},walltime=$TIME
contrib/perf_tests/perf_exp.F:! GPU is unlikely going to have such big caches. In addition Knights
contrib/distro-tools/build_nwchem:# Check whether this machine supports GPUs
contrib/distro-tools/build_nwchem:if [ ${#TCE_CUDA} -ne 0 ] ; then
contrib/distro-tools/build_nwchem:  if [ ${#CUDA} -eq 0 ] ; then
contrib/distro-tools/build_nwchem:     CUDA=`which nvcc`
contrib/distro-tools/build_nwchem:     if [ ${#CUDA} -ne 0 ] ; then
contrib/distro-tools/build_nwchem:       export CUDA
contrib/distro-tools/build_nwchem:       unset CUDA
contrib/distro-tools/build_nwchem:  if [ ${#CUDA} -ne 0 ] ; then
contrib/distro-tools/build_nwchem:     CUDA_PATH="`dirname ${CUDA}`"
contrib/distro-tools/build_nwchem:     CUDA_PATH="`dirname ${CUDA_PATH}`"
contrib/distro-tools/build_nwchem:     if [ ${#CUDA_INCLUDE} -eq 0 ] ; then
contrib/distro-tools/build_nwchem:       export CUDA_INCLUDE="-I${CUDA_PATH}/include"
contrib/distro-tools/build_nwchem:     if [ ${#CUDA_LIBS} -eq 0 ] ; then
contrib/distro-tools/build_nwchem:         export CUDA_LIBS="-L${CUDA_PATH}/lib -lcudart"
contrib/distro-tools/build_nwchem:         export CUDA_LIBS="-L${CUDA_PATH}/lib64 -lcudart"
contrib/distro-tools/build_nwchem:         echo "NWCHEM_PTR_SIZE unknown; cannot set CUDA_LIBS; aborting..."
contrib/distro-tools/build_nwchem:     unset CUDA_PATH
contrib/distro-tools/build_nwchem:  # There is also a variable CUDA_FLAGS but I have no idea what that should be.
contrib/distro-tools/build_nwchem:  # There is also a variable CUDA_FLAGS but I have no idea what that should be.
contrib/distro-tools/build_nwchem:if [ ${#CUDA} -ne 0 ] ; then
contrib/distro-tools/build_nwchem:  echo "CUDA               =" ${CUDA}
contrib/distro-tools/build_nwchem:  echo "export CUDA=${CUDA}" >> ${NWCHEM_ENV_SH}
contrib/distro-tools/build_nwchem:  echo "setenv CUDA ${CUDA}" >> ${NWCHEM_ENV_CSH}
contrib/distro-tools/build_nwchem:  echo "unset CUDA"          >> ${NWCHEM_UNENV_SH}
contrib/distro-tools/build_nwchem:  echo "unsetenv CUDA"       >> ${NWCHEM_UNENV_CSH}
contrib/distro-tools/build_nwchem:if [ ${#CUDA_INCLUDE} -ne 0 ] ; then
contrib/distro-tools/build_nwchem:  echo "CUDA_INCLUDE       =" ${CUDA_INCLUDE}
contrib/distro-tools/build_nwchem:  echo "export CUDA_INCLUDE=\"${CUDA_INCLUDE}\"" >> ${NWCHEM_ENV_SH}
contrib/distro-tools/build_nwchem:  echo "setenv CUDA_INCLUDE \"${CUDA_INCLUDE}\"" >> ${NWCHEM_ENV_CSH}
contrib/distro-tools/build_nwchem:  echo "unset CUDA_INCLUDE"                      >> ${NWCHEM_UNENV_SH}
contrib/distro-tools/build_nwchem:  echo "unsetenv CUDA_INCLUDE"                   >> ${NWCHEM_UNENV_CSH}
contrib/distro-tools/build_nwchem:if [ ${#CUDA_LIBS} -ne 0 ] ; then
contrib/distro-tools/build_nwchem:  echo "CUDA_LIBS          =" ${CUDA_LIBS}
contrib/distro-tools/build_nwchem:  echo "export CUDA_LIBS=\"${CUDA_LIBS}\"" >> ${NWCHEM_ENV_SH}
contrib/distro-tools/build_nwchem:  echo "setenv CUDA_LIBS \"${CUDA_LIBS}\"" >> ${NWCHEM_ENV_CSH}
contrib/distro-tools/build_nwchem:  echo "unset CUDA_LIBS"                   >> ${NWCHEM_UNENV_SH}
contrib/distro-tools/build_nwchem:  echo "unsetenv CUDA_LIBS"                >> ${NWCHEM_UNENV_CSH}
QA/tests/tce_cuda/tce_cuda.nw:cuda 1
QA/tests/tce_cuda/tce_cuda.out: argument  1 = ./tce_cuda/tce_cuda.nw
QA/tests/tce_cuda/tce_cuda.out:cuda 1
QA/tests/tce_cuda/tce_cuda.out:    hostname        = gpu019.local
QA/tests/tce_cuda/tce_cuda.out:    input           = ./tce_cuda/tce_cuda.nw
QA/tests/tce_cuda/tce_cuda.out: Using CUDA CCSD(T) code
QA/tests/tce_mrcc_bwcc/tce_mrcc_bwcc.out:    hostname        = gpu031.local
QA/tests/h2o_opt_simint/h2o_opt_simint.out:       1 offload enabled, GPU:  0
QA/tests/h2o_opt_simint/h2o_opt_simint.out:       0 offload enabled, GPU:  0
QA/doNightlyTests.mpi:if [ ${#TCE_CUDA} -eq 1 ] ; then
QA/doNightlyTests.mpi:  if ["x$TCE_CUDA" == "xy"] ; then
QA/doNightlyTests.mpi:    ./runtests.mpi.unix procs $np tce_cuda
QA/doqmtests.mpi:if [[ ! -z "$TCE_CUDA" ]]; then
QA/doqmtests.mpi:#  if ("x$TCE_CUDA" == "xy") then
QA/doqmtests.mpi:    ./runtests.mpi.unix procs $np tce_cuda
release.notes.7.0.0:- OpenMP GPU offload (work in progress)
.gitignore:# Created by https://www.gitignore.io/api/c,cuda,linux,windows
.gitignore:### CUDA ###
.gitignore:*.gpu
.gitignore:# End of https://www.gitignore.io/api/c,cuda,linux,windows
src/NWints/simint/libsimint_source/build_simint.sh:elif  [[ ${FC_EXTRA} == nvfortran || ${FC} == pgf90 || (${FC} == ftn && ${PE_ENV} == NVIDIA) ]]; then
src/NWints/simint/libsimint_source/build_simint.sh:    if  [[ ${PE_ENV} == NVIDIA ]]; then
src/config/makefile.h:            ifeq ($(PE_ENV),NVIDIA)
src/config/makefile.h:                        LDOPTIONS += -qoffload -lcudart -L$(NWC_CUDAPATH)
src/config/makefile.h:		  FOPTIONS += -mp=gpu #-gpu=cc70
src/config/makefile.h:		  LDOPTIONS += -mp=gpu # -gpu=cc70
src/config/makefile.h:# CUDA
src/config/makefile.h:ifndef CUDA
src/config/makefile.h:    CUDA = nvcc
src/config/makefile.h:ifdef TCE_CUDA
src/config/makefile.h:    DEFINES += -DTCE_CUDA
src/config/makefile.h:    CORE_LIBS += $(CUDA_LIBS)
src/config/makefile.h:ifdef TCE_OPENACC
src/config/makefile.h:        $(error USE_OPENMP must be unset when TCE_OPENACC is set)
src/config/makefile.h:    DEFINES +=-DUSE_F90_ALLOCATABLE -DTCE_OPENACC
src/config/makefile.h:        FOPTIONS += -fopenacc
src/config/makefile.h:        LDOPTIONS += -fopenacc
src/config/makefile.h:    NWCHEM_LINK_CUDA=1
src/config/makefile.h:ifdef NWCHEM_LINK_CUDA
src/config/makefile.h:       CORE_LIBS += -acc -cuda -cudalib=cublas
src/config/makefile.h:       CORE_LIBS +=  -fopenacc -lcublas
src/config/makefile.h:ifeq ($(shell echo $(BLASOPT) |awk '/\/nvidia\/hpc_sdk\// {print "Y"; exit}'),Y)
src/config/makefile.h:    ifdef GPU_ARCH
src/config/makefile.h:        CUDA_ARCH =  -arch=$(GPU_ARCH) 
src/config/makefile.h:        CUDA_ARCH =  -arch=sm_35
src/config/makefile.h:    ifdef TCE_CUDA
src/config/makefile.h:	    CUDA_VERS_GE8 := $(shell nvcc --version|grep rel|  awk '/release 9/ {print "Y";exit}; /release 8/ {print "Y";exit};{print "N"}')
src/config/makefile.h:            ifeq ($(CUDA_VERS_GE8),N)
src/config/makefile.h:                CUDA_FLAGS = -O3 -Xcompiler -std=c++11 -DNOHTIME -Xptxas --warn-on-spills $(CUDA_ARCH) 
src/config/makefile.h:                CUDA_FLAGS = -O3  -std=c++11 -DNOHTIME -Xptxas --warn-on-spills $(CUDA_ARCH) 
src/config/makefile.h:	$(CUDA) -c -DTCE_CUDA $(CUDA_FLAGS) $(CUDA_INCLUDE) -I$(NWCHEM_TOP)/src/tce/ttlg/includes -o $% $<
src/config/makefile.h:	$(CUDA) -c -DTCE_CUDA $(CUDA_FLAGS) $(CUDA_INCLUDE) -o $% $<
src/config/makefile.h:	$(HIP) -c -DTCE_HIP -fno-gpu-rdc -o $% $<
src/config/makefile-legacy.h:# CUDA
src/config/makefile-legacy.h:#ckbn gpu
src/config/makefile-legacy.h:ifdef TCE_CUDA
src/config/makefile-legacy.h: CORE_LIBS += $(CUDA_LIBS)
src/config/makefile-legacy.h:	$(CUDA) -c $(CUDA_FLAGS) -c  $(CUDA_LIBS) -o $% $<
src/ccsd/ccsd_pstat.F:      if (.not. pstat_allocate('ccsd:gpumove', pstat_qstat, 0, junk,
src/ccsd/ccsd_pstat.F:     $     ps_gpumove)) call errquit('ccsd: ccsd_pstat_init', 0,0)
src/ccsd/ccsd_pstat.F:         if(.not.pstat_free(ps_gpumove))call errquit('ccsd_pstat',0,0)
src/ccsd/ccsd_trpdrv_openmp_imax.F:!      use cudafor
src/ccsd/ccsd_trpdrv_openmp_imax.F:!  99 format(2x,'Using Fortran OpenACC+CUBLAS in CCSD(T)')
src/ccsd/ccsd_trpdrv_openmp_imax.F:!      ! setup CUDA streams
src/ccsd/ccsd_trpdrv_openmp_imax.F:!        err = cudaStreamCreate(stream(shi))
src/ccsd/ccsd_trpdrv_openmp_imax.F:!        if (err.ne.0) call errquit('cudaStreamCreate',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                              call pstat_on(ps_gpumove)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                              call qenter('gpumove',0)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                           !err = cudaMemcpyAsync(xJia,Jia,size(Jia),stream(1))
src/ccsd/ccsd_trpdrv_openmp_imax.F:                           !  call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                           !err = cudaMemcpyAsync(xTia,Tia,size(Tia),stream(1))
src/ccsd/ccsd_trpdrv_openmp_imax.F:                           !  call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                           !err = cudaMemcpyAsync(xKia,Kia,size(Kia),stream(2))
src/ccsd/ccsd_trpdrv_openmp_imax.F:                           !  call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                           !err = cudaMemcpyAsync(xXia,Xia,size(Xia),stream(2))
src/ccsd/ccsd_trpdrv_openmp_imax.F:                           !  call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                              !err = cudaMemcpyAsync(xTkj,Tkj,size(Tkj),stream(1))
src/ccsd/ccsd_trpdrv_openmp_imax.F:                              !  call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                              !err = cudaMemcpyAsync(xKkj,Kkj,size(Kkj),stream(1))
src/ccsd/ccsd_trpdrv_openmp_imax.F:                              !  call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                              !err = cudaMemcpyAsync(xJkj,Jkj,size(Jkj),stream(3))
src/ccsd/ccsd_trpdrv_openmp_imax.F:                              !  call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                                 !err = cudaMemcpyAsync(xJka,Jka,size(Jka),stream(5))
src/ccsd/ccsd_trpdrv_openmp_imax.F:                                 !  call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                                 !err = cudaMemcpyAsync(xTka,Tka,size(Tka),stream(5))
src/ccsd/ccsd_trpdrv_openmp_imax.F:                                 !  call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                                 !err = cudaMemcpyAsync(xKka,Kka,size(Kka),stream(6))
src/ccsd/ccsd_trpdrv_openmp_imax.F:                                 !  call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                                 !err = cudaMemcpyAsync(xXka,Xka,size(Xka),stream(6))
src/ccsd/ccsd_trpdrv_openmp_imax.F:                                 !  call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                           !err = cudaMemcpyAsync(xTij,Tij,size(Tij),stream(5))
src/ccsd/ccsd_trpdrv_openmp_imax.F:                           !  call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                           !err = cudaMemcpyAsync(xKij,Kij,size(Kij),stream(5)) ! and 6
src/ccsd/ccsd_trpdrv_openmp_imax.F:                           !  call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                           !err = cudaMemcpyAsync(xJij,Jij,size(Jij),stream(7)) ! and 8
src/ccsd/ccsd_trpdrv_openmp_imax.F:                           !  call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                           ! err = cudaStreamSynchronize(stream(shi))
src/ccsd/ccsd_trpdrv_openmp_imax.F:                           !  call errquit('cudaStreamSync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                              call pstat_off(ps_gpumove)
src/ccsd/ccsd_trpdrv_openmp_imax.F:                              call qexit('gpumove',0)
src/ccsd/ccsd_trpdrv_openmp_imax.F:!      if (alloc_error.ne.0) call errquit('free TKJKD GPU',1,MA_ERR)
src/ccsd/ccsd_trpdrv_openmp_imax.F:! CUDA stuff
src/ccsd/ccsd_trpdrv_openmp_imax.F:      !  err = cudaStreamDestroy(stream(shi))
src/ccsd/ccsd_trpdrv_openmp_imax.F:      !  if (err.ne.0) call errquit('cudaStreamDestroy',err,UNKNOWN_ERR)
src/ccsd/ccsdps.fh:     ,  ps_trpmos,ps_gpumove,ps_accwait
src/ccsd/ccsdps.fh:     $     ps_trpmos,ps_gpumove,ps_accwait
src/ccsd/module/GNUmakefile:  FOPTIONS += -fiopenmp -fopenmp-targets=spir64="-mllvm -vpo-paropt-enable-64bit-opencl-atomics=true -mllvm -vpo-paropt-opt-data-sharing-for-reduction=false" -qmkl -DMKL_ILP64 -I"${MKLROOT}/include" -fpp -fixed -free
src/ccsd/ccsd_trpdrv_openacc.F:      subroutine ccsd_trpdrv_openacc(t1,xeorb,
src/ccsd/ccsd_trpdrv_openacc.F:      use cudafor
src/ccsd/ccsd_trpdrv_openacc.F:      integer(kind=cuda_stream_kind) :: stream(8)
src/ccsd/ccsd_trpdrv_openacc.F:   99 format(2x,'Using Fortran OpenACC+CUBLAS in CCSD(T)')
src/ccsd/ccsd_trpdrv_openacc.F:      ! setup CUDA streams
src/ccsd/ccsd_trpdrv_openacc.F:        err = cudaStreamCreate(stream(shi))
src/ccsd/ccsd_trpdrv_openacc.F:        if (err.ne.0) call errquit('cudaStreamCreate',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:      if (alloc_error.ne.0) call errquit('TKJKD GPU alloc',1,MA_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:                              call pstat_on(ps_gpumove)
src/ccsd/ccsd_trpdrv_openacc.F:                              call qenter('gpumove',0)
src/ccsd/ccsd_trpdrv_openacc.F:                           err = cudaMemcpyAsync(xJia,Jia,size(Jia),stream(1))
src/ccsd/ccsd_trpdrv_openacc.F:                             call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:                           err = cudaMemcpyAsync(xTia,Tia,size(Tia),stream(1))
src/ccsd/ccsd_trpdrv_openacc.F:                             call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:                           err = cudaMemcpyAsync(xKia,Kia,size(Kia),stream(2))
src/ccsd/ccsd_trpdrv_openacc.F:                             call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:                           err = cudaMemcpyAsync(xXia,Xia,size(Xia),stream(2))
src/ccsd/ccsd_trpdrv_openacc.F:                             call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:                              err = cudaMemcpyAsync(xTkj,Tkj,size(Tkj),stream(1))
src/ccsd/ccsd_trpdrv_openacc.F:                                call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:                              err = cudaMemcpyAsync(xKkj,Kkj,size(Kkj),stream(1))
src/ccsd/ccsd_trpdrv_openacc.F:                                call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:                              err = cudaMemcpyAsync(xJkj,Jkj,size(Jkj),stream(3))
src/ccsd/ccsd_trpdrv_openacc.F:                                call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:                                 err = cudaMemcpyAsync(xJka,Jka,size(Jka),stream(5))
src/ccsd/ccsd_trpdrv_openacc.F:                                   call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:                                 err = cudaMemcpyAsync(xTka,Tka,size(Tka),stream(5))
src/ccsd/ccsd_trpdrv_openacc.F:                                   call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:                                 err = cudaMemcpyAsync(xKka,Kka,size(Kka),stream(6))
src/ccsd/ccsd_trpdrv_openacc.F:                                   call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:                                 err = cudaMemcpyAsync(xXka,Xka,size(Xka),stream(6))
src/ccsd/ccsd_trpdrv_openacc.F:                                   call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:                           err = cudaMemcpyAsync(xTij,Tij,size(Tij),stream(5))
src/ccsd/ccsd_trpdrv_openacc.F:                             call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:                           err = cudaMemcpyAsync(xKij,Kij,size(Kij),stream(5)) ! and 6
src/ccsd/ccsd_trpdrv_openacc.F:                             call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:                           err = cudaMemcpyAsync(xJij,Jij,size(Jij),stream(7)) ! and 8
src/ccsd/ccsd_trpdrv_openacc.F:                             call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:                            err = cudaStreamSynchronize(stream(shi))
src/ccsd/ccsd_trpdrv_openacc.F:                             call errquit('cudaStreamSync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:                              call pstat_off(ps_gpumove)
src/ccsd/ccsd_trpdrv_openacc.F:                              call qexit('gpumove',0)
src/ccsd/ccsd_trpdrv_openacc.F:                            err = cudaStreamSynchronize(stream(shi))
src/ccsd/ccsd_trpdrv_openacc.F:                             call errquit('cudaStreamSync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:                        err = cudaDeviceSynchronize()
src/ccsd/ccsd_trpdrv_openacc.F:                          call errquit('cudaDeviceSync',err,UNKNOWN_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:      if (alloc_error.ne.0) call errquit('free TKJKD GPU',1,MA_ERR)
src/ccsd/ccsd_trpdrv_openacc.F:! CUDA stuff
src/ccsd/ccsd_trpdrv_openacc.F:        err = cudaStreamDestroy(stream(shi))
src/ccsd/ccsd_trpdrv_openacc.F:        if (err.ne.0) call errquit('cudaStreamDestroy',err,UNKNOWN_ERR)
src/ccsd/aoccsd2.F:      logical use_trpdrv_openacc
src/ccsd/aoccsd2.F:      if (.not. rtdb_get(rtdb, 'ccsd:use_trpdrv_openacc', mt_log, 1,
src/ccsd/aoccsd2.F:     1                   use_trpdrv_openacc))
src/ccsd/aoccsd2.F:     2    use_trpdrv_openacc=.false.
src/ccsd/aoccsd2.F:         else if (use_trpdrv_openacc) then
src/ccsd/aoccsd2.F:#if defined(USE_OPENACC_TRPDRV)
src/ccsd/aoccsd2.F: 1808    format(' commencing triples evaluation - OpenACC version',i8,
src/ccsd/aoccsd2.F:         call ccsd_trpdrv_openacc(dbl_mb(k_t1),eorb,
src/ccsd/aoccsd2.F:!!         call errquit('aoccsd: trpdrv_openacc disabled ',0,0)
src/ccsd/aoccsd2.F:     I        'MAX GPU version',i8,' at ',f20.2,' secs')
src/ccsd/aoccsd2.F:         call errquit('aoccsd: trpdrv_openmp_gpu disabled ',0,0)
src/ccsd/GNUmakefile:     ccsd_trpdrv_openacc.F \
src/ccsd/GNUmakefile:ifdef USE_OPENACC_TRPDRV
src/ccsd/GNUmakefile:  OBJ_OPTIMIZE += ccsd_trpdrv_openacc.o
src/ccsd/GNUmakefile:  FOPTIONS += -DUSE_OPENACC_TRPDRV
src/ccsd/GNUmakefile:      FOPTIONS += -Mextend -acc -cuda -cudalib=cublas
src/ccsd/GNUmakefile:      FOPTIONS += -ffree-form -fopenacc -lcublas
src/rtdb/db/hash_page.c:	(void)sigprocmask(SIG_BLOCK, &set, &oset);
src/rtdb/db/hash_page.c:	(void)sigprocmask(SIG_SETMASK, &oset, (sigset_t *)NULL);
src/rtdb/db/compat.h:static int __sigtemp;		/* For the use of sigprocmask */
src/rtdb/db/compat.h:#define sigprocmask(how,set,oset) \
src/tce/ccsd/ccsd_t2_8.F:      use cudafor
src/tce/ccsd/ccsd_t2_8.F:      integer(kind=cuda_stream_kind) :: stream(2)
src/tce/ccsd/ccsd_t2_8.F:        err = cudaStreamCreate(stream(shi))
src/tce/ccsd/ccsd_t2_8.F:        if (err.ne.0) call errquit('cudaStreamCreate',err,UNKNOWN_ERR)
src/tce/ccsd/ccsd_t2_8.F:                   err = cudaStreamSynchronize(stream(oldphase))
src/tce/ccsd/ccsd_t2_8.F:                     call errquit('cudaStreamSync',err,UNKNOWN_ERR)
src/tce/ccsd/ccsd_t2_8.F:                   !print*,'cudaMemcpyAsync A'
src/tce/ccsd/ccsd_t2_8.F:                   err = cudaMemcpyAsync(x_a(:,phase),f_a(:,phase),dima,stream(phase))
src/tce/ccsd/ccsd_t2_8.F:                     call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/tce/ccsd/ccsd_t2_8.F:                   !print*,'cudaMemcpyAsync B'
src/tce/ccsd/ccsd_t2_8.F:                   err = cudaMemcpyAsync(x_b(:,phase),f_b(:,phase),dimb,stream(phase))
src/tce/ccsd/ccsd_t2_8.F:                     call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/tce/ccsd/ccsd_t2_8.F:                     !err = cudaStreamSynchronize(stream(phase))
src/tce/ccsd/ccsd_t2_8.F:                     !  call errquit('cudaStreamSync',err,UNKNOWN_ERR)
src/tce/ccsd/ccsd_t2_8.F:              !print*,'cudaMemcpyAsync C (out)'
src/tce/ccsd/ccsd_t2_8.F:                err = cudaMemcpyAsync(f_c(:,shi),x_c(:,shi),dimc,stream(shi))
src/tce/ccsd/ccsd_t2_8.F:                  call errquit('cudaMemcpyAsync',err,UNKNOWN_ERR)
src/tce/ccsd/ccsd_t2_8.F:                err = cudaStreamSynchronize(stream(shi))
src/tce/ccsd/ccsd_t2_8.F:                  call errquit('cudaStreamSync',err,UNKNOWN_ERR)
src/tce/ccsd/ccsd_t2_8.F:        err = cudaStreamDestroy(stream(shi))
src/tce/ccsd/ccsd_t2_8.F:        if (err.ne.0) call errquit('cudaStreamDestroy',err,UNKNOWN_ERR)
src/tce/ccsd/GNUmakefile:ifdef USE_OPENACC_TRPDRV
src/tce/ccsd/GNUmakefile:      FOPTIONS += -Mextend -acc -cuda -cudalib=cublas
src/tce/ccsd/GNUmakefile:      FOPTIONS += -ffree-form -fopenacc -lcublas
src/tce/ttlg/includes/ourinclude.h:#define SAFECUDAMALLOC(callstring) cudamalloc(callstring) ;\
src/tce/ttlg/includes/ourinclude.h:	{cudaError_t err = cudaGetLastError();\
src/tce/ttlg/includes/ourinclude.h:                if(err != cudaSuccess){\
src/tce/ttlg/includes/ourinclude.h:                        printf("\nKernel ERROR in hostCall: %s (line: %d)\n", cudaGetErrorString(err), __LINE__);\
src/tce/ttlg/includes/ourinclude.h:#define SAFECUDAMEMCPY(callstring) cudamemcpy(callstring) ;\
src/tce/ttlg/includes/ourinclude.h:        {cudaError_t err = cudaGetLastError();\
src/tce/ttlg/includes/ourinclude.h:                if(err != cudaSuccess){\
src/tce/ttlg/includes/ourinclude.h:                        printf("\nKernel ERROR in hostCall: %s (line: %d)\n", cudaGetErrorString(err), __LINE__);\
src/tce/ttlg/includes/ourmacros.h:#ifdef TCE_CUDA
src/tce/ttlg/includes/ourmacros.h:#define SAFECUDAMALLOC(pointer, size) cudaMalloc(pointer, size) ;\
src/tce/ttlg/includes/ourmacros.h:        {cudaError_t err = cudaGetLastError();\
src/tce/ttlg/includes/ourmacros.h:                if(err != cudaSuccess){\
src/tce/ttlg/includes/ourmacros.h:                        printf("\nKernel ERROR in hostCall: %s (line: %d)\n", cudaGetErrorString(err), __LINE__);\
src/tce/ttlg/includes/ourmacros.h:#define SAFECUDAMEMCPY(dest, src, size, type) cudaMemcpy(dest, src, size, type) ;\
src/tce/ttlg/includes/ourmacros.h:        {cudaError_t err = cudaGetLastError();\
src/tce/ttlg/includes/ourmacros.h:                if(err != cudaSuccess){\
src/tce/ttlg/includes/ourmacros.h:                        printf("\nKernel ERROR in hostCall: %s (line: %d)\n", cudaGetErrorString(err), __LINE__);\
src/tce/ttlg/includes/ourmacros.h:#define SAFECUDAMALLOC(pointer, size) hipMalloc(pointer, size) ;\
src/tce/ttlg/includes/ourmacros.h:#define SAFECUDAMEMCPY(dest, src, size, type) hipMemcpy(dest, src, size, type) ;\
src/tce/ttlg/includes/nohtimestop.h:cudaDeviceSynchronize(); 
src/tce/ttlg/test.cpp:#include <cuda_runtime.h>
src/tce/ttlg/test.cpp:	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
src/tce/ttlg/test.cpp:		cudaMalloc(&d_A, totalsize);
src/tce/ttlg/test.cpp:		cudaMemcpy(d_A, A, totalsize, cudaMemcpyHostToDevice);
src/tce/ttlg/test.cpp:		cudaMalloc(&d_B, totalsize);
src/tce/ttlg/test.cpp:		//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
src/tce/ttlg/test.cpp:		cudaMemcpy(d_B, B, totalsize, cudaMemcpyHostToDevice);
src/tce/ttlg/test.cpp:		cudaDeviceSynchronize();
src/tce/ttlg/test.cpp:		cudaMemcpy(B, d_B,totalsize, cudaMemcpyDeviceToHost);
src/tce/ttlg/test.cpp:		cudaFree(d_A);
src/tce/ttlg/test.cpp:		cudaFree(d_B);
src/tce/ttlg/main.cpp:#include <cuda_runtime.h>
src/tce/ttlg/main.cpp:		cudaMemcpy(output, input, sizes[0] * sizeof(input[0]), cudaMemcpyDeviceToDevice);
src/tce/ttlg/fvimatchl32.cu:#include <cuda_runtime.h> 
src/tce/ttlg/fvimatchl32.cu:	SAFECUDAMALLOC(&d_offset,limit*sizeof(short)); 
src/tce/ttlg/fvimatchl32.cu:	SAFECUDAMEMCPY(d_offset, offset,limit*sizeof(short), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvimatchl32.cu:	SAFECUDAMALLOC(&d_lda_s,newndim*sizeof(int)); 
src/tce/ttlg/fvimatchl32.cu:	SAFECUDAMALLOC(&d_ldb_s,newndim*sizeof(int)); 
src/tce/ttlg/fvimatchl32.cu:	SAFECUDAMALLOC(&d_idx_s,newndim*sizeof(int)); 
src/tce/ttlg/fvimatchl32.cu:	SAFECUDAMEMCPY(d_idx_s, idx_s+1,newndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvimatchl32.cu:	SAFECUDAMEMCPY(d_lda_s, lda_s+1,newndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvimatchl32.cu:	SAFECUDAMEMCPY(d_ldb_s, temp+1,newndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvimatchl32.cu:	{cudaError_t err = cudaGetLastError();
src/tce/ttlg/fvimatchl32.cu:		if(err != cudaSuccess){
src/tce/ttlg/fvimatchl32.cu:			printf("\nKernel ERROR in fvimatchl32: %s (line: %d)\n", cudaGetErrorString(err), __LINE__);
src/tce/ttlg/fvimatchl32.cu:	cudaFree(d_lda_s);
src/tce/ttlg/fvimatchl32.cu:	cudaFree(d_ldb_s);
src/tce/ttlg/fvimatchl32.cu:	cudaFree(d_idx_s);
src/tce/ttlg/fvimatchl32.cu:	cudaFree(d_offset);
src/tce/ttlg/CMakeLists.txt:#TODO: Double check the GPU part
src/tce/ttlg/CMakeLists.txt:INCLUDE(FindCUDA)
src/tce/ttlg/CMakeLists.txt:if(CUDA_TOOLKIT_INCLUDE) 
src/tce/ttlg/CMakeLists.txt:    list(APPEND TCE_INCLUDES ${CUDA_TOOLKIT_INCLUDE})
src/tce/ttlg/CMakeLists.txt:    message(WARNING "CUDA_TOOLKIT_INCLUDE not set.")
src/tce/ttlg/CMakeLists.txt:set(TTLG_CUDA_SOURCE_FILES 
src/tce/ttlg/CMakeLists.txt:set_source_files_properties(${TTLG_CPP_SOURCE_FILES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
src/tce/ttlg/CMakeLists.txt:cuda_add_library(ttlg STATIC ${TTLG_CPP_SOURCE_FILES} ${TTLG_CUDA_SOURCE_FILES})
src/tce/ttlg/fvigeneralolap.cu:#include <cuda_runtime.h> 
src/tce/ttlg/fvigeneralolap.cu:	SAFECUDAMALLOC(&input_base,ilimit*olimit*sizeof(int)); 
src/tce/ttlg/fvigeneralolap.cu:	SAFECUDAMALLOC(&output_base,ilimit*olimit*sizeof(int)); 
src/tce/ttlg/fvigeneralolap.cu:	SAFECUDAMALLOC(&tile_base1, ilimit*olimit *sizeof(int)); 
src/tce/ttlg/fvigeneralolap.cu:	SAFECUDAMALLOC(&tile_base2, ilimit*olimit*sizeof(int)); 
src/tce/ttlg/fvigeneralolap.cu:	SAFECUDAMALLOC(&d_lda_s,newndim*sizeof(int)); 
src/tce/ttlg/fvigeneralolap.cu:	SAFECUDAMALLOC(&d_ldb_s,newndim*sizeof(int)); 
src/tce/ttlg/fvigeneralolap.cu:	SAFECUDAMALLOC(&d_idx_s,newndim*sizeof(int)); 
src/tce/ttlg/fvigeneralolap.cu:	SAFECUDAMEMCPY(d_idx_s, idx_s+c,newndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvigeneralolap.cu:	SAFECUDAMEMCPY(d_lda_s, lda_s+c,newndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvigeneralolap.cu:	SAFECUDAMEMCPY(d_ldb_s, temp+c,newndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvigeneralolap.cu:	SAFECUDAMEMCPY(input_base, aexpr, ilimit*olimit*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvigeneralolap.cu:	SAFECUDAMEMCPY(output_base, bexpr, ilimit*olimit*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvigeneralolap.cu:	SAFECUDAMEMCPY(tile_base1, texpr1, ilimit*olimit*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvigeneralolap.cu:	SAFECUDAMEMCPY(tile_base2, texpr2,ilimit* olimit*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvigeneralolap.cu:	{cudaError_t err = cudaGetLastError();
src/tce/ttlg/fvigeneralolap.cu:		if(err != cudaSuccess){
src/tce/ttlg/fvigeneralolap.cu:			printf("\nKernel ERROR in fvi_nomatch_generalolap: %s (line: %d)\n", cudaGetErrorString(err), __LINE__);
src/tce/ttlg/fvigeneralolap.cu:	cudaFree(d_lda_s);
src/tce/ttlg/fvigeneralolap.cu:	cudaFree(d_ldb_s);
src/tce/ttlg/fvigeneralolap.cu:	cudaFree(d_idx_s);
src/tce/ttlg/fvigeneralolap.cu:	cudaFree(input_base);
src/tce/ttlg/fvigeneralolap.cu:	cudaFree(output_base);
src/tce/ttlg/fvigeneralolap.cu:	cudaFree(tile_base1);
src/tce/ttlg/fvigeneralolap.cu:	cudaFree(tile_base2);
src/tce/ttlg/fvimatchg32.cu:#include <cuda_runtime.h> 
src/tce/ttlg/fvimatchg32.cu:	SAFECUDAMALLOC(&d_lda_s,ndim*sizeof(int)); 
src/tce/ttlg/fvimatchg32.cu:	SAFECUDAMALLOC(&d_ldb_s,ndim*sizeof(int)); 
src/tce/ttlg/fvimatchg32.cu:	SAFECUDAMALLOC(&d_idx_s,ndim*sizeof(int)); 
src/tce/ttlg/fvimatchg32.cu:	SAFECUDAMEMCPY(d_idx_s, idx_s,ndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvimatchg32.cu:	SAFECUDAMEMCPY(d_lda_s, temp,ndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvimatchg32.cu:	SAFECUDAMEMCPY(d_ldb_s, ldb_s,ndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvimatchg32.cu:	cudaDeviceSynchronize();
src/tce/ttlg/fvimatchg32.cu:	{cudaError_t err = cudaGetLastError();
src/tce/ttlg/fvimatchg32.cu:		if(err != cudaSuccess){
src/tce/ttlg/fvimatchg32.cu:			printf("\nKernel ERROR in dCuKernel %s (line: %d)\n", cudaGetErrorString(err), __LINE__);
src/tce/ttlg/fvimatchg32.cu:	cudaFree(d_lda_s);
src/tce/ttlg/fvimatchg32.cu:	cudaFree(d_ldb_s);
src/tce/ttlg/fvimatchg32.cu:	cudaFree(d_idx_s);
src/tce/ttlg/fvinomatchgeneral.cu:#include <cuda_runtime.h> 
src/tce/ttlg/fvinomatchgeneral.cu:	SAFECUDAMALLOC(&input_base, olimit*sizeof(int)); 
src/tce/ttlg/fvinomatchgeneral.cu:	SAFECUDAMALLOC(&output_base, ilimit*sizeof(int)); 
src/tce/ttlg/fvinomatchgeneral.cu:	SAFECUDAMALLOC(&d_lda_s,newndim*sizeof(int)); 
src/tce/ttlg/fvinomatchgeneral.cu:	SAFECUDAMALLOC(&d_ldb_s,newndim*sizeof(int)); 
src/tce/ttlg/fvinomatchgeneral.cu:	SAFECUDAMALLOC(&d_idx_s,newndim*sizeof(int)); 
src/tce/ttlg/fvinomatchgeneral.cu:	SAFECUDAMEMCPY(d_idx_s, idx_s+c,newndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvinomatchgeneral.cu:	SAFECUDAMEMCPY(d_lda_s, lda_s+c,newndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvinomatchgeneral.cu:	SAFECUDAMEMCPY(d_ldb_s, temp+c,newndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvinomatchgeneral.cu:	SAFECUDAMEMCPY(input_base, aexpr, olimit*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvinomatchgeneral.cu:	SAFECUDAMEMCPY(output_base, bexpr, ilimit*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvinomatchgeneral.cu:	{cudaError_t err = cudaGetLastError();
src/tce/ttlg/fvinomatchgeneral.cu:		if(err != cudaSuccess){
src/tce/ttlg/fvinomatchgeneral.cu:			printf("\nKernel ERROR in fvi_nomatch_general: %s (line: %d)\n", cudaGetErrorString(err), __LINE__);
src/tce/ttlg/fvinomatchgeneral.cu:	cudaFree(d_lda_s);
src/tce/ttlg/fvinomatchgeneral.cu:	cudaFree(d_ldb_s);
src/tce/ttlg/fvinomatchgeneral.cu:	cudaFree(d_idx_s);
src/tce/ttlg/fvinomatchgeneral.cu:	cudaFree(input_base);
src/tce/ttlg/fvinomatchgeneral.cu:	cudaFree(output_base);
src/tce/ttlg/fvinomatchgeneral.cu:	//cudaFree(d_ablock);
src/tce/ttlg/fvinomatchgeneral.cu:	//cudaFree(d_bblock);
src/tce/ttlg/fvimatchg32_blocking.cu:#include <cuda_runtime.h> 
src/tce/ttlg/fvimatchg32_blocking.cu:	SAFECUDAMALLOC(&d_lda_s,ndim*sizeof(int)); 
src/tce/ttlg/fvimatchg32_blocking.cu:	SAFECUDAMALLOC(&d_ldb_s,ndim*sizeof(int)); 
src/tce/ttlg/fvimatchg32_blocking.cu:	SAFECUDAMALLOC(&d_idx_s,ndim*sizeof(int)); 
src/tce/ttlg/fvimatchg32_blocking.cu:	SAFECUDAMEMCPY(d_idx_s, idx_s,ndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvimatchg32_blocking.cu:	SAFECUDAMEMCPY(d_lda_s, temp,ndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvimatchg32_blocking.cu:	SAFECUDAMEMCPY(d_ldb_s, ldb_s,ndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvimatchg32_blocking.cu:	{cudaError_t err = cudaGetLastError();
src/tce/ttlg/fvimatchg32_blocking.cu:		if(err != cudaSuccess){
src/tce/ttlg/fvimatchg32_blocking.cu:			printf("\nKernel ERROR in dCuKernel %s (line: %d)\n", cudaGetErrorString(err), __LINE__);
src/tce/ttlg/fvimatchg32_blocking.cu:	cudaFree(d_lda_s);
src/tce/ttlg/fvimatchg32_blocking.cu:	cudaFree(d_ldb_s);
src/tce/ttlg/fvimatchg32_blocking.cu:	cudaFree(d_idx_s);
src/tce/ttlg/fvinomatchg32.cu:#include <cuda_runtime.h> 
src/tce/ttlg/fvinomatchg32.cu:	SAFECUDAMALLOC(&d_lda_s,newndim*sizeof(int)); 
src/tce/ttlg/fvinomatchg32.cu:	SAFECUDAMALLOC(&d_ldb_s,newndim*sizeof(int)); 
src/tce/ttlg/fvinomatchg32.cu:	SAFECUDAMALLOC(&d_idx_s,newndim*sizeof(int)); 
src/tce/ttlg/fvinomatchg32.cu:	SAFECUDAMEMCPY(d_idx_s, idx_s,newndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvinomatchg32.cu:	SAFECUDAMEMCPY(d_lda_s, lda_s,newndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvinomatchg32.cu:	SAFECUDAMEMCPY(d_ldb_s, temp,newndim*sizeof(int), cudaMemcpyHostToDevice); 
src/tce/ttlg/fvinomatchg32.cu:	cudaDeviceSynchronize();
src/tce/ttlg/fvinomatchg32.cu:	{cudaError_t err = cudaGetLastError();
src/tce/ttlg/fvinomatchg32.cu:		if(err != cudaSuccess){
src/tce/ttlg/fvinomatchg32.cu:			printf("\nKernel ERROR in fvi_nomatch_g32: %s (line: %d)\n", cudaGetErrorString(err), __LINE__);
src/tce/ttlg/fvinomatchg32.cu:	cudaFree(d_lda_s);
src/tce/ttlg/fvinomatchg32.cu:	cudaFree(d_ldb_s);
src/tce/ttlg/fvinomatchg32.cu:	cudaFree(d_idx_s);
src/tce/ttlg/GNUmakefile:#ifdef TCE_CUDA # not really needed, but keep the guard for now.
src/tce/ttlg/GNUmakefile: #LIB_DEFINES = -I./includes $(CUDA_INCLUDE)
src/tce/ttlg/GNUmakefile:ifdef TCE_CUDA
src/tce/ttlg/GNUmakefile:LIB_INCLUDES += $(CUDA_INCLUDE)
src/tce/ttlg/GNUmakefile: #CUDA_FLAGS = -I./includes $(CUDA_INCLUDE)
src/tce/ttlg/GNUmakefile:#CUDA_FLAGS += -O3 -Xcompiler -fPIC -std=c++11 -DNOHTIME -Xptxas --warn-on-spills $(ARCH)
src/tce/tce_input.F:! --- TCE_CUDA
src/tce/tce_input.F:      integer icuda
src/tce/tce_input.F:!      TCE_CUDA Number of CUDA devices per node
src/tce/tce_input.F:      else if (inp_compare(.false.,test,'cuda')) then
src/tce/tce_input.F:#if defined(TCE_CUDA) || defined(TCE_HIP)
src/tce/tce_input.F:        if (.not.inp_i(icuda))
src/tce/tce_input.F:     1    call errquit('tce_input: no icuda',0,INPUT_ERR)
src/tce/tce_input.F:        if (.not.rtdb_put(rtdb,'tce:cuda',mt_int,1,icuda))
src/tce/tce_input.F:        call errquit('cuda option needs TCE_CUDA compiled code',
src/tce/ccsd_t/ccsd_t_singles_l.F:#if defined(TCE_OPENACC) && defined(USE_F90_ALLOCATABLE)
src/tce/ccsd_t/ccsd_t_singles_l.F:#if defined(TCE_OPENACC) && defined(USE_F90_ALLOCATABLE)
src/tce/ccsd_t/ccsd_t_singles_l.F:#if defined(TCE_OPENACC) && defined(USE_F90_ALLOCATABLE)
src/tce/ccsd_t/ccsd_t_singles_l.F:#if defined(TCE_OPENACC) && defined(USE_F90_ALLOCATABLE)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#define CUDA_IMPL
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      SUBROUTINE ccsd_t_doubles_gpu(a_i0,d_t2,d_v2,k_t2_offset,
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write(6,*)'I am in ccsd_t_doubles_gpu',ga_nodeid()
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c        call sd_init_cuda()
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      IF(toggle .eq. 2) CALL ccsd_t_doubles_gpu_1(d_t2,k_t2_offset,d_v2,
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      IF(toggle .eq. 2) CALL ccsd_t_doubles_gpu_2(d_t2,k_t2_offset,d_v2,
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      SUBROUTINE ccsd_t_doubles_gpu_1(d_a,k_a_offset,d_b,k_b_offset,a_c,
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c allocate device memory on GPU      
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:ccx     & ERRQUIT('ccsd_t_doubles_gpu_1',0,MA_ERR)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:     & ERRQUIT('ccsd_t_doubles_gpu_1',1,MA_ERR)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:     &ccsd_t_doubles_gpu_1',2,MA_ERR)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      IF (.not.MA_POP_STACK(l_a)) CALL ERRQUIT('ccsd_t_doubles_gpu_1',3,
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:     & ERRQUIT('ccsd_t_doubles_gpu_1',4,MA_ERR)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:ccx     &ccsd_t_doubles_gpu_1',5,MA_ERR)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:ccx      IF (.not.MA_POP_STACK(l_b)) CALL ERRQUIT('ccsd_t_doubles_gpu_1',6,MA_E
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write (*,*) usedevice, 'Run CUDA '
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d1_1_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d1_2_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write (*,*) usedevice, 'Run CUDA '
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d1_3_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write (*,*) usedevice, 'Run CUDA '
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d1_4_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write (*,*) usedevice, 'Run CUDA '
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d1_5_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write (*,*) usedevice, 'Run CUDA '
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d1_6_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write (*,*) usedevice, 'Run CUDA '
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d1_7_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write (*,*) usedevice, 'Run CUDA '
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d1_8_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write (*,*) usedevice, 'Run CUDA '
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d1_9_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:     + CALL ERRQUIT('ccsd_t_doubles_gpu_1',7,MA_ERR)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:     + CALL ERRQUIT('ccsd_t_doubles_gpu_1',8,MA_ERR)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      SUBROUTINE ccsd_t_doubles_gpu_2(d_a,k_a_offset,d_b,k_b_offset,a_c,
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:ccx     & ERRQUIT('ccsd_t_doubles_gpu_2',0,MA_ERR)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:     & ERRQUIT('ccsd_t_doubles_gpu_2',1,MA_ERR)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:     &ccsd_t_doubles_gpu_2',2,MA_ERR)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:     + CALL ERRQUIT('ccsd_t_doubles_gpu_2',3,MA_ERR)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:     & ERRQUIT('ccsd_t_doubles_gpu_2',4,MA_ERR)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:ccx     &ccsd_t_doubles_gpu_2',5,MA_ERR)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:ccx      IF (.not.MA_POP_STACK(l_b)) CALL ERRQUIT('ccsd_t_doubles_gpu_2',6,MA_E
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      used for CUDA
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write (*,*) usedevice, 'Run CUDA '
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d2_1_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write (*,*) usedevice, 'Run CUDA '
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d2_2_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write (*,*) usedevice, 'Run CUDA '
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d2_3_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write (*,*) usedevice, 'Run CUDA '
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d2_4_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write (*,*) usedevice, 'Run CUDA '
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d2_5_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write (*,*) usedevice, 'Run CUDA '
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d2_6_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write (*,*) usedevice, 'Run CUDA '
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d2_7_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write (*,*) usedevice, 'Run CUDA '
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d2_8_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:c      write (*,*) usedevice, 'Run CUDA '
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:      call sd_t_d2_9_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:     + CALL ERRQUIT('ccsd_t_doubles_gpu_2',7,MA_ERR)
src/tce/ccsd_t/ccsd_t_doubles_gpu.F:     + CALL ERRQUIT('ccsd_t_doubles_gpu_2',8,MA_ERR)
src/tce/ccsd_t/sd_t_total.cu:    t3_d = (double *) getGpuMem(size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu:    cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu:        freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.cu:        freeGpuMem(t3_s_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_1_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d1_1_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.cu:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_1_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d1_1_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_2_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d1_2_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.cu:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  //CUDA_SAFE(
src/tce/ccsd_t/sd_t_total.cu:    cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice); //);
src/tce/ccsd_t/sd_t_total.cu:  //CUDA_SAFE(  
src/tce/ccsd_t/sd_t_total.cu:    cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice); //);
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_2_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d1_2_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_3_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d1_3_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.cu:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_3_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d1_3_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_4_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d1_4_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.cu:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_4_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d1_4_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_5_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d1_5_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.cu:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_5_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d1_5_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_6_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d1_6_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.cu:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_6_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d1_6_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_7_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d1_7_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.cu:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_7_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d1_7_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_8_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d1_8_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.cu:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_8_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d1_8_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_9_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d1_9_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.cu:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2sub_d,t2sub,size_t2sub,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2sub_d,v2sub,size_v2sub,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d1_9_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d1_9_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_1_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d2_1_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu:  //freeGpuMem(t3d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_1_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d2_1_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_2_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d2_2_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu:  //freeGpuMem(t3d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_2_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d2_2_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_3_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d2_3_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu://  freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_3_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d2_3_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_4_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d2_4_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu:  //freeGpuMem(t3d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_4_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d2_4_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_5_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d2_5_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu:  //freeGpuMem(t3d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_5_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d2_5_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_6_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d2_6_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu:  //freeGpuMem(t3d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_6_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d2_6_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_7_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d2_7_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu:  //freeGpuMem(t3d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_7_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d2_7_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_8_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d2_8_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu:  //freeGpuMem(t3d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_8_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d2_8_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_9_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t *streams;
src/tce/ccsd_t/sd_t_total.cu:  cudaFuncSetCacheConfig(sd_t_d2_9_kernel, cudaFuncCachePreferShared);
src/tce/ccsd_t/sd_t_total.cu:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams=(cudaStream_t*) malloc(nstreams*sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d,t2,size_t2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d,v2,size_v2,cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);}
src/tce/ccsd_t/sd_t_total.cu:  //freeGpuMem(t3d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:extern "C" void sd_t_d2_9_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.cu:  sd_t_d2_9_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.cu:    energy_d = (double*)getGpuMem(size_energy*total_block*2);
src/tce/ccsd_t/sd_t_total.cu:    eval_d1 = (double*)getGpuMem(h1d*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu:    eval_d2 = (double*)getGpuMem(h2d*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu:    eval_d3 = (double*)getGpuMem(h3d*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu:    eval_d4 = (double*)getGpuMem(p4d*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu:    eval_d5 = (double*)getGpuMem(p5d*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu:    eval_d6 = (double*)getGpuMem(p6d*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaMemcpy(eval_d1, eval1, h1d*sizeof(double), cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaMemcpy(eval_d2, eval2, h2d*sizeof(double), cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaMemcpy(eval_d3, eval3, h3d*sizeof(double), cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaMemcpy(eval_d4, eval4, p4d*sizeof(double), cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaMemcpy(eval_d5, eval5, p5d*sizeof(double), cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaMemcpy(eval_d6, eval6, p6d*sizeof(double), cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu://    CUDA_SAFE(cudaMemcpy(t3_s_d, host2, total_elements*h3d*sizeof(double), cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaMemcpy(((char *) energy_h) , ((char *) energy_d) , 
src/tce/ccsd_t/sd_t_total.cu:    size_energy*total_block*2, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.cu://    printf("CUDA energy_h is %f %f %d %d %d %d %d %d\n", energy_h[0], energy_h[dimGrid.x]); //, total_size, h1d, h2d, p4d, p5d,p6d);
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_d) , sizeof(double)*h3d*total_elements, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaMemcpy(((char *) ts3) , ((char *) t3_s_d) , sizeof(double)*h3d*total_elements, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.cu:    freeGpuMem(energy_d);
src/tce/ccsd_t/sd_t_total.cu:    freeGpuMem(eval_d1);
src/tce/ccsd_t/sd_t_total.cu:    freeGpuMem(eval_d2);
src/tce/ccsd_t/sd_t_total.cu:    freeGpuMem(eval_d3);
src/tce/ccsd_t/sd_t_total.cu:    freeGpuMem(eval_d4);
src/tce/ccsd_t/sd_t_total.cu:    freeGpuMem(eval_d5);
src/tce/ccsd_t/sd_t_total.cu:    freeGpuMem(eval_d6);
src/tce/ccsd_t/sd_t_total.cu:    t3_s_d = (double *) getGpuMem(size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu:    cudaMemset(t3_s_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_1_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total.cu://CUDA_SAFE(cudaMalloc((void**) &t3_d, size_t3));
src/tce/ccsd_t/sd_t_total.cu://CUDA_SAFE(cudaMalloc((void**) &t2_d, size_t2));
src/tce/ccsd_t/sd_t_total.cu://CUDA_SAFE(cudaMalloc((void**) &v2_d, size_v2));
src/tce/ccsd_t/sd_t_total.cu://  t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:  t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total.cu:    while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu://  CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total.cu://  freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:   //  cudaFree(t2_d);
src/tce/ccsd_t/sd_t_total.cu:   //  cudaFree(v2_d);
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_1_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  sd_t_s1_1_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d,  t3, t2, v2);
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_2_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total.cu:    t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:        cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu://CUDA_SAFE(cudaMalloc((void**) &t2_d, size_t2));
src/tce/ccsd_t/sd_t_total.cu://CUDA_SAFE(cudaMalloc((void**) &v2_d, size_v2));
src/tce/ccsd_t/sd_t_total.cu:  t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total.cu:    while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu://  CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_2_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  sd_t_s1_2_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d,  t3, t2, v2);
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_3_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total.cu:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:        cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu:  t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total.cu:    while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total.cu:*/  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:  //CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total.cu://  freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_3_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  sd_t_s1_3_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d, t3, t2, v2);
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_4_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total.cu:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:        cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu://  t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:  t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:  /*  CUDA_SAFE(cudaMemcpy(((char *) t3_p) , ((char *) t3_d) , size_block_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.cu://    while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total.cu://  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total.cu://  freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_4_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  sd_t_s1_4_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d,  t3, t2, v2);
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_5_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total.cu:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:        cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu://  t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:  t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total.cu:    while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:  //CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total.cu://  freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_5_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  sd_t_s1_5_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d,  t3, t2, v2);
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_6_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total.cu:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:        cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu://  t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:  t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total.cu:    while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:  //CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total.cu://  freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_6_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  sd_t_s1_6_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d, t3, t2, v2);
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_7_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total.cu:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:        cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu://  t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:  t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total.cu:    while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:  //CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total.cu://  freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_7_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  sd_t_s1_7_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d, t3, t2, v2);
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_8_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total.cu:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:        cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu://  t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:  t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total.cu:    while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu://  CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total.cu://  freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_8_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  sd_t_s1_8_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d, t3, t2, v2);
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_9_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total.cu:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:        cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total.cu://  t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.cu:  t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.cu:  v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.cu:  streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:  CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.cu:    CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total.cu:    while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total.cu:  cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total.cu:  //CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.cu:    cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total.cu:  //freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.cu:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.cu:sd_t_s1_9_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.cu:  sd_t_s1_9_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d,  t3, t2, v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:    t3_d = (double *) getGpuMem(size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total.hip.cpp:        freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:        freeGpuMem(t3_s_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_1_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(t2sub_d,t2sub,size_t2sub,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(v2sub_d,v2sub,size_v2sub,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_1_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d1_1_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_2_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  //CUDA_SAFE(
src/tce/ccsd_t/sd_t_total.hip.cpp:  //CUDA_SAFE(  
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_2_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d1_2_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_3_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(t2sub_d,t2sub,size_t2sub,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(v2sub_d,v2sub,size_v2sub,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_3_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d1_3_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_4_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(t2sub_d,t2sub,size_t2sub,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(v2sub_d,v2sub,size_v2sub,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_4_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d1_4_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_5_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(t2sub_d,t2sub,size_t2sub,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(v2sub_d,v2sub,size_v2sub,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_5_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d1_5_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_6_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(t2sub_d,t2sub,size_t2sub,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(v2sub_d,v2sub,size_v2sub,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_6_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d1_6_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_7_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(t2sub_d,t2sub,size_t2sub,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(v2sub_d,v2sub,size_v2sub,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_7_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d1_7_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_8_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(t2sub_d,t2sub,size_t2sub,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(v2sub_d,v2sub,size_v2sub,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_8_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d1_8_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_9_cuda(size_t h1d, size_t h2d, size_t h3d, size_t h7d, size_t p4d, size_t p5d, size_t p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(t2sub_d,t2sub,size_t2sub,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(v2sub_d,v2sub,size_v2sub,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d1_9_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d1_9_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*h7d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_1_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(t2_d,t2,size_t2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(v2_d,v2,size_v2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  //freeGpuMem(t3d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_1_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d2_1_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_2_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(t2_d,t2,size_t2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(v2_d,v2,size_v2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  //freeGpuMem(t3d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_2_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d2_2_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_3_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(t2_d,t2,size_t2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(v2_d,v2,size_v2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp://  freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_3_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d2_3_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_4_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(t2_d,t2,size_t2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(v2_d,v2,size_v2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  //freeGpuMem(t3d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_4_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d2_4_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_5_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(t2_d,t2,size_t2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(v2_d,v2,size_v2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  //freeGpuMem(t3d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_5_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d2_5_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_6_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(t2_d,t2,size_t2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(v2_d,v2,size_v2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  //freeGpuMem(t3d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_6_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d2_6_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_7_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(t2_d,t2,size_t2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(v2_d,v2,size_v2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  //freeGpuMem(t3d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_7_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d2_7_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_8_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(t2_d,t2,size_t2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(v2_d,v2,size_v2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  //freeGpuMem(t3d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_8_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d2_8_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_9_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, size_t p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  //t3d=(double*)getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp:  t2_d=(double*)getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:  v2_d=(double*)getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipStreamCreate(&streams[i])) ;
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(t2_d,t2,size_t2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  CUDA_SAFE(hipMemcpy(v2_d,v2,size_v2,hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:  //freeGpuMem(t3d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:  freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:extern "C" void sd_t_d2_9_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total.hip.cpp:  sd_t_d2_9_cuda((size_t)*h1d,(size_t)*h2d,(size_t)*h3d,(size_t)*p4d,(size_t)*p5d,(size_t)*p6d,(size_t)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:    energy_d = (double*)getGpuMem(size_energy*total_block*2);
src/tce/ccsd_t/sd_t_total.hip.cpp:    eval_d1 = (double*)getGpuMem(h1d*sizeof(double));
src/tce/ccsd_t/sd_t_total.hip.cpp:    eval_d2 = (double*)getGpuMem(h2d*sizeof(double));
src/tce/ccsd_t/sd_t_total.hip.cpp:    eval_d3 = (double*)getGpuMem(h3d*sizeof(double));
src/tce/ccsd_t/sd_t_total.hip.cpp:    eval_d4 = (double*)getGpuMem(p4d*sizeof(double));
src/tce/ccsd_t/sd_t_total.hip.cpp:    eval_d5 = (double*)getGpuMem(p5d*sizeof(double));
src/tce/ccsd_t/sd_t_total.hip.cpp:    eval_d6 = (double*)getGpuMem(p6d*sizeof(double));
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipMemcpy(eval_d1, eval1, h1d*sizeof(double), hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipMemcpy(eval_d2, eval2, h2d*sizeof(double), hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipMemcpy(eval_d3, eval3, h3d*sizeof(double), hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipMemcpy(eval_d4, eval4, p4d*sizeof(double), hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipMemcpy(eval_d5, eval5, p5d*sizeof(double), hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipMemcpy(eval_d6, eval6, p6d*sizeof(double), hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp://    CUDA_SAFE(hipMemcpy(t3_s_d, host2, total_elements*h3d*sizeof(double), hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipMemcpy(((char *) energy_h) , ((char *) energy_d) , 
src/tce/ccsd_t/sd_t_total.hip.cpp://    printf("CUDA energy_h is %f %f %d %d %d %d %d %d\n", energy_h[0], energy_h[dimGrid.x]); //, total_size, h1d, h2d, p4d, p5d,p6d);
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipMemcpy(((char *) t3) , ((char *) t3_d) , sizeof(double)*h3d*total_elements, hipMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.hip.cpp:    CUDA_SAFE(hipMemcpy(((char *) ts3) , ((char *) t3_s_d) , sizeof(double)*h3d*total_elements, hipMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.hip.cpp:    freeGpuMem(energy_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:    freeGpuMem(eval_d1);
src/tce/ccsd_t/sd_t_total.hip.cpp:    freeGpuMem(eval_d2);
src/tce/ccsd_t/sd_t_total.hip.cpp:    freeGpuMem(eval_d3);
src/tce/ccsd_t/sd_t_total.hip.cpp:    freeGpuMem(eval_d4);
src/tce/ccsd_t/sd_t_total.hip.cpp:    freeGpuMem(eval_d5);
src/tce/ccsd_t/sd_t_total.hip.cpp:    freeGpuMem(eval_d6);
src/tce/ccsd_t/sd_t_total.hip.cpp:    t3_s_d = (double *) getGpuMem(size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_1_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp://CUDA_SAFE(hipMalloc((void**) &t3_d, size_t3));
src/tce/ccsd_t/sd_t_total.hip.cpp://CUDA_SAFE(hipMalloc((void**) &t2_d, size_t2));
src/tce/ccsd_t/sd_t_total.hip.cpp://CUDA_SAFE(hipMalloc((void**) &v2_d, size_v2));
src/tce/ccsd_t/sd_t_total.hip.cpp://	t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:		CUDA_SAFE(hipStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(t2_d, t2, size_t2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(v2_d, v2, size_v2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:		CUDA_SAFE(hipMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, hipMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total.hip.cpp:		while (cudaStreamQuery(streams[stream]) != hipSuccess);
src/tce/ccsd_t/sd_t_total.hip.cpp://	CUDA_SAFE(hipMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, hipMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.hip.cpp://	freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_1_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp:	sd_t_s1_1_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d,  t3, t2, v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_2_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp:		t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp://CUDA_SAFE(hipMalloc((void**) &t2_d, size_t2));
src/tce/ccsd_t/sd_t_total.hip.cpp://CUDA_SAFE(hipMalloc((void**) &v2_d, size_v2));
src/tce/ccsd_t/sd_t_total.hip.cpp:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:		CUDA_SAFE(hipStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(t2_d, t2, size_t2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(v2_d, v2, size_v2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:		CUDA_SAFE(hipMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, hipMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total.hip.cpp:		while (cudaStreamQuery(streams[stream]) != hipSuccess);
src/tce/ccsd_t/sd_t_total.hip.cpp://	CUDA_SAFE(hipMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, hipMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_2_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp:	sd_t_s1_2_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d,  t3, t2, v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_3_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:		CUDA_SAFE(hipStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(t2_d, t2, size_t2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(v2_d, v2, size_v2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:		CUDA_SAFE(hipMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, hipMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total.hip.cpp:		while (cudaStreamQuery(streams[stream]) != hipSuccess);
src/tce/ccsd_t/sd_t_total.hip.cpp:	//CUDA_SAFE(hipMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, hipMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.hip.cpp://	freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_3_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp:	sd_t_s1_3_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d, t3, t2, v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_4_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp://	t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:		CUDA_SAFE(hipStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(t2_d, t2, size_t2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(v2_d, v2, size_v2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:	/*	CUDA_SAFE(hipMemcpy(((char *) t3_p) , ((char *) t3_d) , size_block_t3, hipMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.hip.cpp://		while (cudaStreamQuery(streams[stream]) != hipSuccess);
src/tce/ccsd_t/sd_t_total.hip.cpp://	freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_4_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp:	sd_t_s1_4_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d,  t3, t2, v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_5_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp://	t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:		CUDA_SAFE(hipStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(t2_d, t2, size_t2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(v2_d, v2, size_v2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:		CUDA_SAFE(hipMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, hipMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total.hip.cpp:		while (cudaStreamQuery(streams[stream]) != hipSuccess);
src/tce/ccsd_t/sd_t_total.hip.cpp:	//CUDA_SAFE(hipMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, hipMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.hip.cpp://	freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_5_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp:	sd_t_s1_5_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d,  t3, t2, v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_6_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp://	t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:		CUDA_SAFE(hipStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(t2_d, t2, size_t2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(v2_d, v2, size_v2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:		CUDA_SAFE(hipMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, hipMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total.hip.cpp:		while (cudaStreamQuery(streams[stream]) != hipSuccess);
src/tce/ccsd_t/sd_t_total.hip.cpp:	//CUDA_SAFE(hipMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, hipMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.hip.cpp://	freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_6_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp:	sd_t_s1_6_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d, t3, t2, v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_7_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp://	t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:		CUDA_SAFE(hipStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(t2_d, t2, size_t2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(v2_d, v2, size_v2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:		CUDA_SAFE(hipMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, hipMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total.hip.cpp:		while (cudaStreamQuery(streams[stream]) != hipSuccess);
src/tce/ccsd_t/sd_t_total.hip.cpp:	//CUDA_SAFE(hipMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, hipMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.hip.cpp://	freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_7_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp:	sd_t_s1_7_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d, t3, t2, v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_8_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp://	t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:		CUDA_SAFE(hipStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(t2_d, t2, size_t2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(v2_d, v2, size_v2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:		CUDA_SAFE(hipMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, hipMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total.hip.cpp:		while (cudaStreamQuery(streams[stream]) != hipSuccess);
src/tce/ccsd_t/sd_t_total.hip.cpp://	CUDA_SAFE(hipMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, hipMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.hip.cpp://	freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_8_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp:	sd_t_s1_8_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d, t3, t2, v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_9_cuda(size_t h1d, size_t h2d, size_t h3d, size_t p4d, size_t p5d, size_t p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp://	t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total.hip.cpp:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total.hip.cpp:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total.hip.cpp:		CUDA_SAFE(hipStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(t2_d, t2, size_t2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:	CUDA_SAFE(hipMemcpy(v2_d, v2, size_v2, hipMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total.hip.cpp:		CUDA_SAFE(hipMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, hipMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total.hip.cpp:		while (cudaStreamQuery(streams[stream]) != hipSuccess);
src/tce/ccsd_t/sd_t_total.hip.cpp:	//CUDA_SAFE(hipMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, hipMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total.hip.cpp:	//freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total.hip.cpp:sd_t_s1_9_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total.hip.cpp:	sd_t_s1_9_cuda((size_t) *h1d, (size_t) *h2d, (size_t) *h3d, (size_t) *p4d, (size_t) *p5d, (size_t) *p6d,  t3, t2, v2);
src/tce/ccsd_t/ccsd_t.F:#ifdef TCE_OPENACC
src/tce/ccsd_t/ccsd_t.F:#ifdef TCE_OPENACC
src/tce/ccsd_t/ccsd_t.F:#ifdef TCE_OPENACC
src/tce/ccsd_t/hybrid.c:#define OLD_CUDA 1
src/tce/ccsd_t/hybrid.c:#ifdef TCE_CUDA
src/tce/ccsd_t/hybrid.c:#ifdef OLD_CUDA
src/tce/ccsd_t/hybrid.c:#include <cuda_runtime_api.h>
src/tce/ccsd_t/hybrid.c:#include <cuda.h>
src/tce/ccsd_t/hybrid.c:int check_device_(long *icuda) {
src/tce/ccsd_t/hybrid.c:  /* Check whether this process is associated with a GPU */
src/tce/ccsd_t/hybrid.c:  if((util_my_smp_index())<*icuda) return 1;
src/tce/ccsd_t/hybrid.c://void device_init_(int *icuda) {
src/tce/ccsd_t/hybrid.c:int device_init_(long *icuda,long *cuda_device_number ) {
src/tce/ccsd_t/hybrid.c:#ifdef TCE_CUDA
src/tce/ccsd_t/hybrid.c:  cudaGetDeviceCount(&dev_count_check);
src/tce/ccsd_t/hybrid.c:  if(dev_count_check < *icuda){
src/tce/ccsd_t/hybrid.c:    printf("Warning: Please check whether you have %ld cuda devices per node\n",*icuda);
src/tce/ccsd_t/hybrid.c:    *cuda_device_number = 30;
src/tce/ccsd_t/hybrid.c:#ifdef TCE_CUDA
src/tce/ccsd_t/hybrid.c:    cudaSetDevice(device_id);
src/tce/ccsd_t/offl_ccsd_t_singles_l.F:         CALL offl_gpu_ccsd_t_singles_l_1(
src/tce/ccsd_t/header.h:#ifdef TCE_CUDA
src/tce/ccsd_t/header.h:#ifdef OLD_CUDA
src/tce/ccsd_t/header.h:#include <cuda_runtime_api.h>
src/tce/ccsd_t/header.h:#include <cuda.h>
src/tce/ccsd_t/header.h:#ifdef TCE_CUDA
src/tce/ccsd_t/header.h:    cudaError_t err = cudaGetLastError();\
src/tce/ccsd_t/header.h:    if (cudaSuccess != err) { \
src/tce/ccsd_t/header.h:        printf("%s\n",cudaGetErrorString(err)); \
src/tce/ccsd_t/header.h:#define CUDA_SAFE(x) if ( cudaSuccess != (x) ) {\
src/tce/ccsd_t/header.h:    printf("CUDA CALL FAILED AT LINE %d OF FILE %s error %s\n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); exit(1);}
src/tce/ccsd_t/header.h:#define CUDA_SAFE(x) if ( hipSuccess != (x) ) {\
src/tce/ccsd_t/header.h:void *getGpuMem(size_t bytes);
src/tce/ccsd_t/header.h:void freeGpuMem(void *p);
src/tce/ccsd_t/ccsd_t_dot.F:#ifdef TCE_OPENACC
src/tce/ccsd_t/ccsd_t_dot.F:#ifdef TCE_OPENACC
src/tce/ccsd_t/ccsd_t_gpu.F:      SUBROUTINE ccsd_t_gpu(d_t1,k_t1_offset,d_t2,k_t2_offset,
src/tce/ccsd_t/ccsd_t_gpu.F:     1                  d_v2,k_v2_offset,energy1,energy2,size_t1,icuda)
src/tce/ccsd_t/ccsd_t_gpu.F:      integer has_GPU
src/tce/ccsd_t/ccsd_t_gpu.F:      integer icuda
src/tce/ccsd_t/ccsd_t_gpu.F:      integer cuda_device_number
src/tce/ccsd_t/ccsd_t_gpu.F:      cuda_device_number = 0
src/tce/ccsd_t/ccsd_t_gpu.F:      has_GPU = check_device(icuda)
src/tce/ccsd_t/ccsd_t_gpu.F:      if (has_GPU.eq.1) then
src/tce/ccsd_t/ccsd_t_gpu.F:       call device_init(icuda,cuda_device_number)
src/tce/ccsd_t/ccsd_t_gpu.F:       if (cuda_device_number .eq. 30) then
src/tce/ccsd_t/ccsd_t_gpu.F:         call errquit("cuda",30,INPUT_ERR)
src/tce/ccsd_t/ccsd_t_gpu.F:        write(*,'(A,I3,A)') "Using ",icuda," device per node"
src/tce/ccsd_t/ccsd_t_gpu.F:              call errquit('ccsd_t_gpu: MA error - singles',size,MA_ERR)
src/tce/ccsd_t/ccsd_t_gpu.F:              call errquit('ccsd_t_gpu: MA error - doubles',size,MA_ERR)
src/tce/ccsd_t/ccsd_t_gpu.F:            has_GPU = check_device(icuda)
src/tce/ccsd_t/ccsd_t_gpu.F:            if (has_GPU.eq.1) then
src/tce/ccsd_t/ccsd_t_gpu.F:            has_GPU = check_device(icuda)
src/tce/ccsd_t/ccsd_t_gpu.F:            call ccsd_t_singles_gpu(dbl_mb(k_singles),
src/tce/ccsd_t/ccsd_t_gpu.F:     3        has_GPU)
src/tce/ccsd_t/ccsd_t_gpu.F:            call ccsd_t_doubles_gpu(dbl_mb(k_doubles),d_t2,d_v2,
src/tce/ccsd_t/ccsd_t_gpu.F:     2        has_GPU)
src/tce/ccsd_t/ccsd_t_gpu.F:            has_GPU = check_device(icuda)
src/tce/ccsd_t/ccsd_t_gpu.F:            if (has_GPU.eq.0) then
src/tce/ccsd_t/ccsd_t_gpu.F:c     GPU process
src/tce/ccsd_t/ccsd_t_gpu.F:c    release GPU memory
src/tce/ccsd_t/sd_t_total_ttlg.cu:#include <cuda_runtime.h>
src/tce/ccsd_t/sd_t_total_ttlg.cu:#include <cuda.h>
src/tce/ccsd_t/sd_t_total_ttlg.cu:    t3_d = (double *) getGpuMem(size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu:    cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu:        freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        freeGpuMem(t3_s_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_cuda(int h1d, int h2d, int h3d, int h7d, int p4d, int p5d, int p6d, double *triplesx, double *t2sub, double *v2sub, int id) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:        t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        	output_d=(double*)getGpuMem(size_triplesx);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total_ttlg.cu:        freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:  	freeGpuMem(output_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_1_cuda(int h1d, int h2d, int h3d, int h7d, int p4d, int p5d, int p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_d1_cuda( h1d,  h2d,  h3d,  h7d,  p4d,  p5d,  p6d, triplesx, t2sub, v2sub, 1);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_1_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d1_1_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_2_cuda(int h1d, int h2d, int h3d, int h7d, int p4d, int p5d, int p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_d1_cuda( h1d,  h2d,  h3d,  h7d,  p4d,  p5d,  p6d, triplesx, t2sub, v2sub, 2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_2_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d1_2_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_3_cuda(int h1d, int h2d, int h3d, int h7d, int p4d, int p5d, int p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_d1_cuda( h1d,  h2d,  h3d,  h7d,  p4d,  p5d,  p6d, triplesx, t2sub, v2sub, 3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_3_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d1_3_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_4_cuda(int h1d, int h2d, int h3d, int h7d, int p4d, int p5d, int p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_d1_cuda( h1d,  h2d,  h3d,  h7d,  p4d,  p5d,  p6d, triplesx, t2sub, v2sub, 4);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_4_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d1_4_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_5_cuda(int h1d, int h2d, int h3d, int h7d, int p4d, int p5d, int p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_d1_cuda( h1d,  h2d,  h3d,  h7d,  p4d,  p5d,  p6d, triplesx, t2sub, v2sub, 5);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_5_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d1_5_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_6_cuda(int h1d, int h2d, int h3d, int h7d, int p4d, int p5d, int p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_d1_cuda( h1d,  h2d,  h3d,  h7d,  p4d,  p5d,  p6d, triplesx, t2sub, v2sub, 6);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_6_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d1_6_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_7_cuda(int h1d, int h2d, int h3d, int h7d, int p4d, int p5d, int p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_d1_cuda( h1d,  h2d,  h3d,  h7d,  p4d,  p5d,  p6d, triplesx, t2sub, v2sub, 7);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_7_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d1_7_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_8_cuda(int h1d, int h2d, int h3d, int h7d, int p4d, int p5d, int p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_d1_cuda( h1d,  h2d,  h3d,  h7d,  p4d,  p5d,  p6d, triplesx, t2sub, v2sub, 8);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_8_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d1_8_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_9_cuda(int h1d, int h2d, int h3d, int h7d, int p4d, int p5d, int p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_d1_cuda( h1d,  h2d,  h3d,  h7d,  p4d,  p5d,  p6d, triplesx, t2sub, v2sub, 9);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d1_9_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* h7d, Integer* p4d, Integer* p5d, Integer* p6d, double *triplesx, double *t2sub, double *v2sub) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d1_9_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*h7d,(int)*p4d,(int)*p5d,(int)*p6d,triplesx,t2sub,v2sub);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d, int p7d, double *triplesx, double *t2sub, double *v2sub, int id) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:        t2sub_d=(double*)getGpuMem(size_t2sub);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        v2sub_d=(double*)getGpuMem(size_v2sub);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        	output_d=(double*)getGpuMem(size_triplesx);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total_ttlg.cu:        freeGpuMem(t2sub_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        freeGpuMem(v2sub_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        	freeGpuMem(output_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_1_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d, int p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_d2_cuda(h1d, h2d, h3d, p4d,  p5d, p6d,  p7d, t3, t2, v2, 1);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_1_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d2_1_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_2_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d, int p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_d2_cuda(h1d, h2d, h3d, p4d,  p5d, p6d,  p7d, t3, t2, v2, 2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_2_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d2_2_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_3_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d, int p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_d2_cuda(h1d, h2d, h3d, p4d,  p5d, p6d,  p7d, t3, t2, v2, 3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_3_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d2_3_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_4_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d, int p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_d2_cuda(h1d, h2d, h3d, p4d,  p5d, p6d,  p7d, t3, t2, v2, 4);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_4_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d2_4_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_5_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d, int p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_d2_cuda(h1d, h2d, h3d, p4d,  p5d, p6d,  p7d, t3, t2, v2, 5);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_5_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d2_5_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_6_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d, int p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_d2_cuda(h1d, h2d, h3d, p4d,  p5d, p6d,  p7d, t3, t2, v2, 6);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_6_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d2_6_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_7_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d, int p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_d2_cuda(h1d, h2d, h3d, p4d,  p5d, p6d,  p7d, t3, t2, v2, 7);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_7_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d2_7_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_8_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d, int p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_d2_cuda(h1d, h2d, h3d, p4d,  p5d, p6d,  p7d, t3, t2, v2, 8);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_8_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d2_8_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_9_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d, int p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_d2_cuda(h1d, h2d, h3d, p4d,  p5d, p6d,  p7d, t3, t2, v2, 9);
src/tce/ccsd_t/sd_t_total_ttlg.cu:extern "C" void sd_t_d2_9_cuda_(Integer *h1d, Integer* h2d, Integer* h3d, Integer* p4d, Integer* p5d, Integer* p6d, Integer* p7d, double *t3, double *t2, double *v2) {
src/tce/ccsd_t/sd_t_total_ttlg.cu:  sd_t_d2_9_cuda((int)*h1d,(int)*h2d,(int)*h3d,(int)*p4d,(int)*p5d,(int)*p6d,(int)*p7d,t3,t2,v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:    energy_d = (double*)getGpuMem(size_energy*total_block*2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:    eval_d1 = (double*)getGpuMem(h1d*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu:    eval_d2 = (double*)getGpuMem(h2d*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu:    eval_d3 = (double*)getGpuMem(h3d*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu:    eval_d4 = (double*)getGpuMem(p4d*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu:    eval_d5 = (double*)getGpuMem(p5d*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu:    eval_d6 = (double*)getGpuMem(p6d*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu:    CUDA_SAFE(cudaMemcpy(eval_d1, eval1, h1d*sizeof(double), cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:    CUDA_SAFE(cudaMemcpy(eval_d2, eval2, h2d*sizeof(double), cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:    CUDA_SAFE(cudaMemcpy(eval_d3, eval3, h3d*sizeof(double), cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:    CUDA_SAFE(cudaMemcpy(eval_d4, eval4, p4d*sizeof(double), cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:    CUDA_SAFE(cudaMemcpy(eval_d5, eval5, p5d*sizeof(double), cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:    CUDA_SAFE(cudaMemcpy(eval_d6, eval6, p6d*sizeof(double), cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu://    CUDA_SAFE(cudaMemcpy(t3_s_d, host2, total_elements*h3d*sizeof(double), cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total_ttlg.cu:    CUDA_SAFE(cudaMemcpy(((char *) energy_h) , ((char *) energy_d) , 
src/tce/ccsd_t/sd_t_total_ttlg.cu:    size_energy*total_block*2, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total_ttlg.cu://    printf("CUDA energy_h is %f %f %d %d %d %d %d %d\n", energy_h[0], energy_h[dimGrid.x]); //, total_size, h1d, h2d, p4d, p5d,p6d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:    CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_d) , sizeof(double)*h3d*total_elements, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total_ttlg.cu:    CUDA_SAFE(cudaMemcpy(((char *) ts3) , ((char *) t3_s_d) , sizeof(double)*h3d*total_elements, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total_ttlg.cu:    freeGpuMem(energy_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:    freeGpuMem(eval_d1);
src/tce/ccsd_t/sd_t_total_ttlg.cu:    freeGpuMem(eval_d2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:    freeGpuMem(eval_d3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:    freeGpuMem(eval_d4);
src/tce/ccsd_t/sd_t_total_ttlg.cu:    freeGpuMem(eval_d5);
src/tce/ccsd_t/sd_t_total_ttlg.cu:    freeGpuMem(eval_d6);
src/tce/ccsd_t/sd_t_total_ttlg.cu:    t3_s_d = (double *) getGpuMem(size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu:    cudaMemset(t3_s_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_1_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total_ttlg.cu://CUDA_SAFE(cudaMalloc((void**) &t3_d, size_t3));
src/tce/ccsd_t/sd_t_total_ttlg.cu://CUDA_SAFE(cudaMalloc((void**) &t2_d, size_t2));
src/tce/ccsd_t/sd_t_total_ttlg.cu://CUDA_SAFE(cudaMalloc((void**) &v2_d, size_v2));
src/tce/ccsd_t/sd_t_total_ttlg.cu://	t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total_ttlg.cu://	CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total_ttlg.cu://	freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:   //  cudaFree(t2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:   //  cudaFree(v2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_1_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_s1_1_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d,  t3, t2, v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_2_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total_ttlg.cu:		t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu://CUDA_SAFE(cudaMalloc((void**) &t2_d, size_t2));
src/tce/ccsd_t/sd_t_total_ttlg.cu://CUDA_SAFE(cudaMalloc((void**) &v2_d, size_v2));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total_ttlg.cu://	CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_2_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_s1_2_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d,  t3, t2, v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_3_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total_ttlg.cu:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total_ttlg.cu:*/	cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total_ttlg.cu:	//CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total_ttlg.cu://	freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_3_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_s1_3_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d, t3, t2, v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_4_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total_ttlg.cu:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu://	t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total_ttlg.cu:	/*	CUDA_SAFE(cudaMemcpy(((char *) t3_p) , ((char *) t3_d) , size_block_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total_ttlg.cu://		while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total_ttlg.cu://	cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total_ttlg.cu:		cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total_ttlg.cu://	freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_4_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_s1_4_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d,  t3, t2, v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_5_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total_ttlg.cu:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu://	t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total_ttlg.cu:	//CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total_ttlg.cu://	freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_5_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_s1_5_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d,  t3, t2, v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_6_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total_ttlg.cu:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu://	t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total_ttlg.cu:	//CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total_ttlg.cu://	freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_6_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_s1_6_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d, t3, t2, v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_7_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total_ttlg.cu:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu://	t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total_ttlg.cu:	//CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total_ttlg.cu://	freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_7_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_s1_7_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d, t3, t2, v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_8_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total_ttlg.cu:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu://	t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total_ttlg.cu://	CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total_ttlg.cu://	freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_8_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d, double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_s1_8_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d, t3, t2, v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_9_cuda(int h1d, int h2d, int h3d, int p4d, int p5d, int p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaStream_t   *streams;
src/tce/ccsd_t/sd_t_total_ttlg.cu:        t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:        cudaMemset(t3_d,0,size_t3*sizeof(double));
src/tce/ccsd_t/sd_t_total_ttlg.cu://	t3_d = (double *) getGpuMem(size_t3);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	t2_d = (double *) getGpuMem(size_t2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	v2_d = (double *) getGpuMem(size_v2);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		CUDA_SAFE(cudaStreamCreate(&streams[i]));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(t2_d, t2, size_t2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:	CUDA_SAFE(cudaMemcpy(v2_d, v2, size_v2, cudaMemcpyHostToDevice));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		CUDA_SAFE(cudaMemcpyAsync(((char *) t3_p) + i * size_block_t3, ((char *) t3_s_d) + i * size_block_t3, size_block_t3, cudaMemcpyDeviceToHost, streams[i]));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		while (cudaStreamQuery(streams[stream]) != cudaSuccess);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	cudaDeviceSynchronize();
src/tce/ccsd_t/sd_t_total_ttlg.cu:	//CUDA_SAFE(cudaMemcpy(((char *) t3) , ((char *) t3_s_d) , size_t3, cudaMemcpyDeviceToHost));
src/tce/ccsd_t/sd_t_total_ttlg.cu:		cudaStreamDestroy(streams[i]);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	//freeGpuMem(t3_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(t2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:	freeGpuMem(v2_d);
src/tce/ccsd_t/sd_t_total_ttlg.cu:sd_t_s1_9_cuda_(Integer * h1d, Integer * h2d, Integer * h3d, Integer * p4d, Integer * p5d, Integer * p6d,  double *t3, double *t2, double *v2)
src/tce/ccsd_t/sd_t_total_ttlg.cu:	sd_t_s1_9_cuda((int) *h1d, (int) *h2d, (int) *h3d, (int) *p4d, (int) *p5d, (int) *p6d,  t3, t2, v2);
src/tce/ccsd_t/memory.hip.cpp:  static map<size_t,set<void*> > free_list_gpu, free_list_host;
src/tce/ccsd_t/memory.hip.cpp:  static map<void *,size_t> live_ptrs_gpu, live_ptrs_host;
src/tce/ccsd_t/memory.hip.cpp:  static void clearGpuFreeList() {
src/tce/ccsd_t/memory.hip.cpp:    for(map<size_t,set<void*> >::iterator it=free_list_gpu.begin(); 
src/tce/ccsd_t/memory.hip.cpp:	it!=free_list_gpu.end(); ++it) {
src/tce/ccsd_t/memory.hip.cpp:    free_list_gpu.clear();
src/tce/ccsd_t/memory.hip.cpp:    CUDA_SAFE(fn((void **)&ptr, bytes));
src/tce/ccsd_t/memory.hip.cpp:      clearGpuFreeList();
src/tce/ccsd_t/memory.hip.cpp:void *getGpuMem(size_t bytes) {
src/tce/ccsd_t/memory.hip.cpp:  CUDA_SAFE(hipMalloc((void **) &ptr, bytes));
src/tce/ccsd_t/memory.hip.cpp:  if(free_list_gpu.find(bytes)!=free_list_gpu.end()) {
src/tce/ccsd_t/memory.hip.cpp:    set<void*> &lst = free_list_gpu.find(bytes)->second;
src/tce/ccsd_t/memory.hip.cpp:      ptr = resurrect_from_free_list(free_list_gpu, bytes, live_ptrs_gpu);
src/tce/ccsd_t/memory.hip.cpp:    for(map<size_t,set<void *> >::iterator it=free_list_gpu.begin();
src/tce/ccsd_t/memory.hip.cpp:	it != free_list_gpu.end(); ++it) {
src/tce/ccsd_t/memory.hip.cpp:	ptr = resurrect_from_free_list(free_list_gpu, it->first, live_ptrs_gpu);
src/tce/ccsd_t/memory.hip.cpp:  live_ptrs_gpu[ptr] = bytes;
src/tce/ccsd_t/memory.hip.cpp:  CUDA_SAFE(hipHostMalloc((void **) &ptr, bytes));
src/tce/ccsd_t/memory.hip.cpp:/* 	live_ptrs_gpu[ptr] = bytes; */
src/tce/ccsd_t/memory.hip.cpp:void freeGpuMem(void *p) {
src/tce/ccsd_t/memory.hip.cpp:  assert(live_ptrs_gpu.find(p) != live_ptrs_gpu.end());
src/tce/ccsd_t/memory.hip.cpp:  bytes = live_ptrs_gpu[p];
src/tce/ccsd_t/memory.hip.cpp:  live_ptrs_gpu.erase(p);
src/tce/ccsd_t/memory.hip.cpp:  free_list_gpu[bytes].insert(p);
src/tce/ccsd_t/memory.hip.cpp:  assert(live_ptrs_gpu.size()==0);
src/tce/ccsd_t/memory.hip.cpp:  clearGpuFreeList();
src/tce/ccsd_t/ccsd_t_singles_gpu.F:#define CUDA_IMPL
src/tce/ccsd_t/ccsd_t_singles_gpu.F:      SUBROUTINE ccsd_t_singles_gpu(a_i0,d_t1,d_v2,k_t1_offset,
src/tce/ccsd_t/ccsd_t_singles_gpu.F:      IF(toggle .eq. 2) CALL ccsd_t_singles_gpu_1(d_t1,k_t1_offset,d_v2,
src/tce/ccsd_t/ccsd_t_singles_gpu.F:      SUBROUTINE ccsd_t_singles_gpu_1(d_a,k_a_offset,d_b,k_b_offset,a_c,
src/tce/ccsd_t/ccsd_t_singles_gpu.F:c allocate device memory on GPU
src/tce/ccsd_t/ccsd_t_singles_gpu.F:ccx     & ERRQUIT('ccsd_t_singles_gpu_1',0,MA_ERR)
src/tce/ccsd_t/ccsd_t_singles_gpu.F:     & ERRQUIT('ccsd_t_singles_gpu_1',1,MA_ERR)
src/tce/ccsd_t/ccsd_t_singles_gpu.F:     &ccsd_t_singles_gpu_1',2,MA_ERR)
src/tce/ccsd_t/ccsd_t_singles_gpu.F:      IF (.not.MA_POP_STACK(l_a)) CALL ERRQUIT('ccsd_t_singles_gpu_1',3,
src/tce/ccsd_t/ccsd_t_singles_gpu.F:     & ERRQUIT('ccsd_t_singles_gpu_1',4,MA_ERR)
src/tce/ccsd_t/ccsd_t_singles_gpu.F:ccx     &ccsd_t_singles_gpu_1',5,MA_ERR)
src/tce/ccsd_t/ccsd_t_singles_gpu.F:ccx      IF (.not.MA_POP_STACK(l_b)) CALL ERRQUIT('ccsd_t_singles_gpu_1',6,MA_E
src/tce/ccsd_t/ccsd_t_singles_gpu.F:ccx      IF (.not.MA_POP_STACK(l_b_sort)) CALL ERRQUIT('ccsd_t_singles_gpu_1',7
src/tce/ccsd_t/ccsd_t_singles_gpu.F:ccx      IF (.not.MA_POP_STACK(l_a_sort)) CALL ERRQUIT('ccsd_t_singles_gpu_1',8
src/tce/ccsd_t/ccsd_t_singles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_singles_gpu.F:      call sd_t_s1_1_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_singles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_singles_gpu.F:      call sd_t_s1_2_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_singles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_singles_gpu.F:      call sd_t_s1_3_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_singles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_singles_gpu.F:      call sd_t_s1_4_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_singles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_singles_gpu.F:      call sd_t_s1_5_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_singles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_singles_gpu.F:      call sd_t_s1_6_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_singles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_singles_gpu.F:      call sd_t_s1_7_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_singles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_singles_gpu.F:      call sd_t_s1_8_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_singles_gpu.F:#elif defined(CUDA_IMPL)
src/tce/ccsd_t/ccsd_t_singles_gpu.F:      call sd_t_s1_9_cuda(int_mb(k_range+h1b-1),int_mb(k_range+h2b-1),
src/tce/ccsd_t/ccsd_t_singles_gpu.F:ccx      IF (.not.MA_POP_STACK(l_c_sort)) CALL ERRQUIT('ccsd_t_singles_gpu_1',9
src/tce/ccsd_t/ccsd_t_singles_gpu.F:     + CALL ERRQUIT('ccsd_t_singles_gpu_1',7,MA_ERR)
src/tce/ccsd_t/ccsd_t_singles_gpu.F:     + CALL ERRQUIT('ccsd_t_singles_gpu_1',8,MA_ERR)
src/tce/ccsd_t/memory.cu:  static map<size_t,set<void*> > free_list_gpu, free_list_host;
src/tce/ccsd_t/memory.cu:  static map<void *,size_t> live_ptrs_gpu, live_ptrs_host;
src/tce/ccsd_t/memory.cu:  static void clearGpuFreeList() {
src/tce/ccsd_t/memory.cu:    for(map<size_t,set<void*> >::iterator it=free_list_gpu.begin(); 
src/tce/ccsd_t/memory.cu:	it!=free_list_gpu.end(); ++it) {
src/tce/ccsd_t/memory.cu:	cudaFree(*it2);
src/tce/ccsd_t/memory.cu:    free_list_gpu.clear();
src/tce/ccsd_t/memory.cu:	cudaFreeHost(*it2);
src/tce/ccsd_t/memory.cu:  typedef cudaError (*mallocfn_t)(void **ptr, size_t bytes);
src/tce/ccsd_t/memory.cu:    CUDA_SAFE(fn((void **)&ptr, bytes));
src/tce/ccsd_t/memory.cu:      clearGpuFreeList();
src/tce/ccsd_t/memory.cu:void *getGpuMem(size_t bytes) {
src/tce/ccsd_t/memory.cu:  CUDA_SAFE(cudaMalloc((void **) &ptr, bytes));
src/tce/ccsd_t/memory.cu:  if(free_list_gpu.find(bytes)!=free_list_gpu.end()) {
src/tce/ccsd_t/memory.cu:    set<void*> &lst = free_list_gpu.find(bytes)->second;
src/tce/ccsd_t/memory.cu:      ptr = resurrect_from_free_list(free_list_gpu, bytes, live_ptrs_gpu);
src/tce/ccsd_t/memory.cu:    for(map<size_t,set<void *> >::iterator it=free_list_gpu.begin();
src/tce/ccsd_t/memory.cu:	it != free_list_gpu.end(); ++it) {
src/tce/ccsd_t/memory.cu:	ptr = resurrect_from_free_list(free_list_gpu, it->first, live_ptrs_gpu);
src/tce/ccsd_t/memory.cu:  ptr = morecore(cudaMalloc, bytes);
src/tce/ccsd_t/memory.cu:/*   cutilSafeCall(cudaMalloc((void **) &ptr, bytes)); */
src/tce/ccsd_t/memory.cu:  live_ptrs_gpu[ptr] = bytes;
src/tce/ccsd_t/memory.cu:  CUDA_SAFE(cudaMallocHost((void **) &ptr, bytes));
src/tce/ccsd_t/memory.cu:/* 	live_ptrs_gpu[ptr] = bytes; */
src/tce/ccsd_t/memory.cu:/*   cutilSafeCall(cudaMallocHost((void **) &ptr, bytes)); */
src/tce/ccsd_t/memory.cu:  ptr = morecore(cudaMallocHost, bytes);
src/tce/ccsd_t/memory.cu:  cudaFreeHost(p);
src/tce/ccsd_t/memory.cu:void freeGpuMem(void *p) {
src/tce/ccsd_t/memory.cu:  cudaFree(p);
src/tce/ccsd_t/memory.cu:  assert(live_ptrs_gpu.find(p) != live_ptrs_gpu.end());
src/tce/ccsd_t/memory.cu:  bytes = live_ptrs_gpu[p];
src/tce/ccsd_t/memory.cu:  live_ptrs_gpu.erase(p);
src/tce/ccsd_t/memory.cu:  free_list_gpu[bytes].insert(p);
src/tce/ccsd_t/memory.cu:  assert(live_ptrs_gpu.size()==0);
src/tce/ccsd_t/memory.cu:  clearGpuFreeList();
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      SUBROUTINE offl_gpu_ccsd_t_singles_l_1(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      call offl_gpu_sd_t_s1_1(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      call offl_gpu_sd_t_s1_2(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      call offl_gpu_sd_t_s1_3(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      call offl_gpu_sd_t_s1_4(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      call offl_gpu_sd_t_s1_5(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      call offl_gpu_sd_t_s1_6(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      call offl_gpu_sd_t_s1_7(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      call offl_gpu_sd_t_s1_8(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      call offl_gpu_sd_t_s1_9(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      subroutine offl_gpu_sd_t_s1_1(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      subroutine offl_gpu_sd_t_s1_2(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      subroutine offl_gpu_sd_t_s1_3(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      subroutine offl_gpu_sd_t_s1_4(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      subroutine offl_gpu_sd_t_s1_5(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      subroutine offl_gpu_sd_t_s1_6(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      subroutine offl_gpu_sd_t_s1_7(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      subroutine offl_gpu_sd_t_s1_8(
src/tce/ccsd_t/offl_ccsd_t_singles_l_1.F:      subroutine offl_gpu_sd_t_s1_9(
src/tce/ccsd_t/offl_ccsd_t_doubles_l.F:#ifdef TCE_OPENACC
src/tce/ccsd_t/offl_ccsd_t_doubles_l.F:#ifdef TCE_OPENACC
src/tce/ccsd_t/offl_ccsd_t_doubles_l.F:c if not using TCE_OPENACC:      
src/tce/ccsd_t/offl_ccsd_t_doubles_l.F:c end if using using TCE_OPENACC kernels
src/tce/ccsd_t/offl_ccsd_t_doubles_l.F:#ifdef TCE_OPENACC
src/tce/ccsd_t/offl_ccsd_t_doubles_l.F:c else if not using OPENACC kernels
src/tce/ccsd_t/offl_ccsd_t_doubles_l.F:#ifdef TCE_OPENACC
src/tce/ccsd_t/GNUmakefile:ifdef TCE_CUDA
src/tce/ccsd_t/GNUmakefile: OBJ_OPTIMIZE += hybrid.o memory.o ccsd_t_gpu.o ccsd_t_singles_gpu.o ccsd_t_doubles_gpu.o
src/tce/ccsd_t/GNUmakefile: USES_BLAS += ccsd_t_singles_gpu.F ccsd_t_doubles_gpu.F
src/tce/ccsd_t/GNUmakefile: LIB_DEFINES += $(CUDA_INCLUDE)
src/tce/ccsd_t/GNUmakefile: OBJ_OPTIMIZE += hybrid.o memory.o ccsd_t_gpu.o ccsd_t_singles_gpu.o ccsd_t_doubles_gpu.o sd_t_total.o
src/tce/ccsd_t/GNUmakefile: USES_BLAS += ccsd_t_singles_gpu.F ccsd_t_doubles_gpu.F
src/tce/ccsd_t/GNUmakefile:ifdef TCE_OPENACC
src/tce/tce_energy.F:      integer icuda
src/tce/tce_energy.F:            if (.not.rtdb_get(rtdb,'tce:cuda',mt_int,1,icuda)) icuda = 0
src/tce/tce_energy.F:            if (icuda .ne. 0) then
src/tce/tce_energy.F:#if defined(TCE_CUDA) || defined(TCE_HIP)
src/tce/tce_energy.F:              if (nodezero) write(LuOut,*) 'Using CUDA CCSD(T) code'
src/tce/tce_energy.F:              call ccsd_t_gpu(d_t1,k_t1_offset,d_t2,k_t2_offset,
src/tce/tce_energy.F:     &                        icuda)
src/tce/tce_energy.F:              call errquit('tce_energy: ccsd_t_gpu requested'//
src/tce/GNUmakefile:ifdef TCE_CUDA
src/tce/GNUmakefile:      LIB_DEFINES += -DTCE_CUDA
src/util/util_gpu_affinity.F:      subroutine util_setup_gpu_affinity
src/util/util_gpu_affinity.F:#ifdef USE_CUDA_AFFINITY
src/util/util_gpu_affinity.F:      use cudafor
src/util/util_gpu_affinity.F:      use openacc
src/util/util_gpu_affinity.F:      integer(INT32) :: num_devices, use_ngpus, my_gpu
src/util/util_gpu_affinity.F:      character*255 :: char_use_ngpus
src/util/util_gpu_affinity.F:      devicetype = acc_device_nvidia
src/util/util_gpu_affinity.F:      ! CUDA stuff
src/util/util_gpu_affinity.F:      ! how many GPUs are detected
src/util/util_gpu_affinity.F:      err = cudaGetDeviceCount(num_devices)
src/util/util_gpu_affinity.F:      if (err.ne.0) call errquit('cudaGetDeviceCount',err,UNKNOWN_ERR)
src/util/util_gpu_affinity.F:      if (num_devices.lt.1) call errquit('No GPU found!',0,UNKNOWN_ERR)
src/util/util_gpu_affinity.F:      call util_getenv('NWCHEM_OPENACC_USE_NGPUS',char_use_ngpus)
src/util/util_gpu_affinity.F:        write(6,701) 'CU NWCHEM_OPENACC_USE_NGPUS=',trim(char_use_ngpus)
src/util/util_gpu_affinity.F:      if (len(trim(char_use_ngpus)).gt.0) then
src/util/util_gpu_affinity.F:        read(char_use_ngpus,'(i255)') use_ngpus
src/util/util_gpu_affinity.F:        if (use_ngpus.gt.num_devices) then
src/util/util_gpu_affinity.F:          write(6,600) use_ngpus,num_devices
src/util/util_gpu_affinity.F:          use_ngpus = num_devices
src/util/util_gpu_affinity.F:     &         i2,' GPUs but only ',
src/util/util_gpu_affinity.F:        use_ngpus=num_devices
src/util/util_gpu_affinity.F:      ! assign GPUs to GA process ranks within a node (round-robin)
src/util/util_gpu_affinity.F:      my_gpu = modulo(node_rank,use_ngpus)
src/util/util_gpu_affinity.F:        write(6,700) 'use_ngpus',use_ngpus
src/util/util_gpu_affinity.F:        write(6,700) 'my_gpu   ',my_gpu
src/util/util_gpu_affinity.F:      err = cudaSetDevice(my_gpu)
src/util/util_gpu_affinity.F:      if (err.ne.0) call errquit('cudaSetDevice',my_gpu,UNKNOWN_ERR)
src/util/util_gpu_affinity.F:      call acc_set_device_num(my_gpu,devicetype)
src/util/util_gpu_affinity.F:#ifdef USE_OPENACC_AFFINITY
src/util/util_gpu_affinity.F:      use openacc
src/util/util_gpu_affinity.F:      integer(INT32) :: num_devices, use_ngpus, my_gpu
src/util/util_gpu_affinity.F:      character*255 :: char_use_ngpus
src/util/util_gpu_affinity.F:      devicetype = acc_device_nvidia
src/util/util_gpu_affinity.F:      if (num_devices.lt.1) call errquit('No GPU found!',0,UNKNOWN_ERR)
src/util/util_gpu_affinity.F:      call util_getenv('NWCHEM_OPENACC_USE_NGPUS',char_use_ngpus)
src/util/util_gpu_affinity.F:        write(6,701) 'CU NWCHEM_OPENACC_USE_NGPUS=',trim(char_use_ngpus)
src/util/util_gpu_affinity.F:      if (len(trim(char_use_ngpus)).gt.0) then
src/util/util_gpu_affinity.F:        read(char_use_ngpus,'(i255)') use_ngpus
src/util/util_gpu_affinity.F:        if (use_ngpus.gt.num_devices) then
src/util/util_gpu_affinity.F:          write(6,600) use_ngpus,num_devices
src/util/util_gpu_affinity.F:          use_ngpus = num_devices
src/util/util_gpu_affinity.F:     &         i2,' GPUs but only ',
src/util/util_gpu_affinity.F:        use_ngpus=num_devices
src/util/util_gpu_affinity.F:      ! assign GPUs to GA process ranks within a node (round-robin)
src/util/util_gpu_affinity.F:      my_gpu = modulo(node_rank,use_ngpus)
src/util/util_gpu_affinity.F:        write(6,700) 'use_ngpus',use_ngpus
src/util/util_gpu_affinity.F:        write(6,700) 'my_gpu   ',my_gpu
src/util/util_gpu_affinity.F:      call acc_set_device_num(my_gpu,devicetype)
src/util/util_gpu_affinity.F:      end subroutine util_setup_gpu_affinity
src/util/util_cuda_support.c:#define OLD_CUDA 1
src/util/util_cuda_support.c:#ifdef TCE_CUDA
src/util/util_cuda_support.c:#ifdef OLD_CUDA
src/util/util_cuda_support.c:#include <cuda_runtime_api.h>
src/util/util_cuda_support.c:#include <cuda.h>
src/util/util_cuda_support.c:Integer FATR util_cuda_get_num_devices_(){
src/util/util_cuda_support.c:#ifdef TCE_CUDA
src/util/util_cuda_support.c:  cudaGetDeviceCount(&dev_count_check);
src/util/GNUmakefile:              util_blasthreads.o util_gpu_affinity.o util_scalapack.o \
src/util/GNUmakefile:ifdef TCE_CUDA
src/util/GNUmakefile:  LIB_DEFINES += $(CUDA_INCLUDE)
src/util/GNUmakefile:  DEFINES += -DTCE_CUDA
src/util/GNUmakefile:  OBJ += util_cuda_support.o
src/util/GNUmakefile:  OBJ += util_cuda_support.o
src/util/GNUmakefile:ifdef USE_OPENACC_TRPDRV
src/util/GNUmakefile:    LIB_DEFINES += -cuda -DUSE_CUDA_AFFINITY
src/util/GNUmakefile:    LIB_DEFINES += $(CUDA_INCLUDE) -DUSE_CUDA_AFFINITY
src/util/GNUmakefile:ifdef TCE_OPENACC
src/util/GNUmakefile:    FOPTIONS += -cuda -acc
src/util/GNUmakefile:    LIB_DEFINES += -DUSE_CUDA_AFFINITY
src/util/GNUmakefile:    FOPTIONS += -fopenacc
src/util/GNUmakefile:    LIB_DEFINES += -DUSE_OPENACC_AFFINITY
src/lucia/lucia_csf.F:C          OCCLS_IN_CI(NOCCLS,IOCCLS,ICISPC,NINCCLS,INCCLS)
src/lucia/gasdir2.F_vog0514:      SUBROUTINE OCCLS_IN_CI(NOCCLS,IOCCLS,ICISPC,NINCCLS,INCCLS)
src/lucia/gasdir2.F_vog0514:      INTEGER INCCLS(*)
src/lucia/gasdir2.F_vog0514:      NINCCLS = 0
src/lucia/gasdir2.F_vog0514:          NINCCLS = NINCCLS + 1
src/lucia/gasdir2.F_vog0514:          INCCLS(JOCCLS) = 1      
src/lucia/gasdir2.F_vog0514:          INCCLS(JOCCLS) = 0
src/lucia/gasdir2.F_vog0514:        WRITE(6,*) ' Number of occupation classes included ',NINCCLS
src/lucia/gasdir2.F_vog0514:        CALL IWRTMA(INCCLS,1,NOCCLS,1,NOCCLS)  
src/lucia/gasdir2.F_vog0514:*          OCCLS_IN_CI(NOCCLS_MAX,IOCCLS,ICISPC,NINCCLS,INCCLS)
src/lucia/gasdir2.F:      SUBROUTINE OCCLS_IN_CI(NOCCLS,IOCCLS,ICISPC,NINCCLS,INCCLS)
src/lucia/gasdir2.F:      INTEGER INCCLS(*)
src/lucia/gasdir2.F:      NINCCLS = 0
src/lucia/gasdir2.F:          NINCCLS = NINCCLS + 1
src/lucia/gasdir2.F:          INCCLS(JOCCLS) = 1      
src/lucia/gasdir2.F:          INCCLS(JOCCLS) = 0
src/lucia/gasdir2.F:        WRITE(6,*) ' Number of occupation classes included ',NINCCLS
src/lucia/gasdir2.F:        CALL IWRTMA(INCCLS,1,NOCCLS,1,NOCCLS)  
src/lucia/gasdir2.F:*          OCCLS_IN_CI(NOCCLS_MAX,IOCCLS,ICISPC,NINCCLS,INCCLS)
src/tools/GNUmakefile:# CUDA UM support - disabled now that trpdrv_openacc does not need it
src/tools/GNUmakefile:#ifdef NWCHEM_LINK_CUDA
src/tools/GNUmakefile:#    MAYBE_ARMCI +=  --enable-cuda-mem
src/java/nwchem_Timing.java:	    accuData1 = new double[numFrames1];
src/java/nwchem_Timing.java:  void addAccuData(){
src/java/nwchem_Times.java:	while(readData()){addAccuData();};
src/java/nwchem_Times.java:	  if(curData==numData) {readData(); addAccuData(); accPlot.fillPlot();} else {curData++; retrieveData();}; 
src/java/nwchem_Times.java:  void addAccuData(){
src/libext/mpich/build_mpich.sh:#./configure --prefix=`pwd`/../.. --enable-fortran=all $SHARED_FLAGS  --disable-cxx --enable-romio --with-pm=gforker --with-device=ch3:nemesis --disable-cuda --disable-opencl --enable-silent-rules  --enable-fortran=all
src/libext/tblite/build_tblite_cmake.sh:    if [[ ${PE_ENV} == NVIDIA ]]; then
src/libext/openblas/build_openblas.sh:    if [[ ${PE_ENV} == NVIDIA ]]; then
src/libext/scalapack/build_scalapa.sh:    if [[ ${PE_ENV} == NVIDIA ]]; then
src/libext/scalapack/build_scalapa.sh:if [[ ${PE_ENV} == NVIDIA ]] || [[ ${FC} == nvfortran ]] ; then
src/libext/elpa/build_elpa.sh:if [[ ${FC} == nvfortran ]]  || [[ ${PE_ENV} == NVIDIA ]] ; then
src/libext/elpa/build_elpa.sh:elif [[ ${CC} == nvc ]]  || [[ ${PE_ENV} == NVIDIA ]] ; then
src/libext/elpa/build_elpa.sh:elif [[ ${FC} == nvfortran ]]  || [[ ${PE_ENV} == NVIDIA ]] ; then
src/nwchem.F:      call util_setup_gpu_affinity()
src/nwchem.F:     w              ' offload enabled, GPU: ', offload_device()
src/peigs/h/peigs_types.h:void g_exit2_ (Integer *n, char *array, Integer *procmap, Integer *len, Integer *iwork);
src/peigs/h/peigs_types.h:void g_exit_ (Integer *n, char *array, Integer *procmap, Integer *len, Integer *iwork, DoublePrecision *work);
src/peigs/h/peigs_types.h:void pxerbla2_ (Integer *n, char *array, Integer *procmap, Integer *len, Integer *iwork, Integer *info);
src/peigs/src/c/pxerbla.c:void pxerbla2_( n, array, procmap, len, iwork, info )
src/peigs/src/c/pxerbla.c:     Integer *n, *procmap, *len, *iwork, *info;
src/peigs/src/c/pxerbla.c:                in procmap.  To do this you need to do a global operation.
src/peigs/src/c/pxerbla.c:    *info = -50: this processor is not in procmap
src/peigs/src/c/pxerbla.c:    *info = -1:  number of distinct processor ids in procmap > # of 
src/peigs/src/c/pxerbla.c:    let nproc = Number of unique processor ids in procmap, i.e.,
src/peigs/src/c/pxerbla.c:                nprocs = reduce_list( *len, procmap, proclist).
src/peigs/src/c/pxerbla.c:  nprocs = reduce_list2( *len, procmap, proclist);
src/peigs/src/c/exit.c:void g_exit2_( n, array, procmap, len, iwork )
src/peigs/src/c/exit.c:     Integer *n, *procmap, *len, *iwork;
src/peigs/src/c/exit.c:void g_exit_( n, array, procmap, len, iwork, work )
src/peigs/src/c/exit.c:     Integer *n, *procmap, *len, *iwork;
src/peigs/src/c/exit.c:    on any processor in procmap the routine exits with a message from array.
src/peigs/src/c/exit.c:    n should be less than or equal to 0 on all processors in procmap.
src/peigs/src/c/exit.c:    if n is not 0 after a global combine then all processors in procmap
src/peigs/src/c/exit.c:    procmap = array of processors on which to check n
src/peigs/src/c/exit.c:    len     = length of the array procmap
src/peigs/src/c/exit.c:    let nproc = Number of unique processor ids in procmap, i.e.,
src/peigs/src/c/exit.c:                nprocs = reduce_list( *len, procmap, proclist).
src/peigs/src/c/exit.c:  nprocs = reduce_list2( *len, procmap, proclist);
src/stepper/stpr_save.F:#if defined(OPENCLOSE)
src/stepper/stpr_save.F:#if defined(OPENCLOSE)
src/stepper/stpr_save.F:#if defined(OPENCLOSE)
src/stepper/stpr_save.F:#if defined(OPENCLOSE)
src/stepper/stpr_save.F:#if defined(OPENCLOSE)
src/stepper/stpr_save.F:#if defined(OPENCLOSE)
src/stepper/stpr_save.F:#if defined(OPENCLOSE)
src/stepper/stpr_save.F:#if defined(OPENCLOSE)
src/stepper/stpr_fndmde.F:#if defined(OPENCLOSE)
src/stepper/stpr_fndmde.F:#if defined(OPENCLOSE)
src/stepper/stpr_fndmde.F:#if defined(OPENCLOSE)
src/stepper/stpr_stepcor.F:#if defined(OPENCLOSE)
src/stepper/stpr_stepcor.F:#if defined(OPENCLOSE)
src/stepper/stpr_place.F:#if defined(OPENCLOSE)
src/stepper/stpr_place.F:#if defined(OPENCLOSE)
src/stepper/stpr_place.F:#if defined(OPENCLOSE)
src/stepper/stpr_place.F:#if defined(OPENCLOSE)

```
