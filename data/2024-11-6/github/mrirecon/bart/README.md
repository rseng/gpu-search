# https://github.com/mrirecon/bart

```console
tests/nufft.mk:tests/test-nudft-gpu-forward: traj nufft reshape nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_fft.ra
tests/nufft.mk:tests/test-nudft-gpu-adjoint: zeros noise reshape traj nufft fmac nrmse
tests/nufft.mk:tests/test-nufft-gpu-inverse: traj phantom nufft nrmse
tests/nufft.mk:tests/test-nufft-gpu-adjoint: traj phantom nufft nrmse
tests/nufft.mk:tests/test-nufft-gpu-forward: traj phantom nufft nrmse
tests/nufft.mk:tests/test-nufft-gpu-inverse-lowmem: traj phantom nufft nrmse
tests/nufft.mk:tests/test-nufft-gpu-adjoint-lowmem: traj phantom nufft nrmse
tests/nufft.mk:tests/test-nufft-gpu-forward-lowmem: traj phantom nufft nrmse
tests/nufft.mk:tests/test-nufft-gpu-inverse-precomp: traj phantom nufft nrmse
tests/nufft.mk:tests/test-nufft-gpu-adjoint-precomp: traj phantom nufft nrmse
tests/nufft.mk:tests/test-nufft-gpu-forward-precomp: traj phantom nufft nrmse
tests/nufft.mk:TESTS_GPU += tests/test-nufft-gpu-inverse tests/test-nufft-gpu-adjoint tests/test-nufft-gpu-forward
tests/nufft.mk:TESTS_GPU += tests/test-nufft-gpu-inverse-lowmem tests/test-nufft-gpu-adjoint-lowmem tests/test-nufft-gpu-forward-lowmem
tests/nufft.mk:TESTS_GPU += tests/test-nufft-gpu-inverse-precomp tests/test-nufft-gpu-adjoint-precomp tests/test-nufft-gpu-forward-precomp
tests/nufft.mk:TESTS_GPU += tests/test-nudft-gpu-forward tests/test-nudft-gpu-adjoint
tests/gpu.mk:tests/test-pics-gpu: pics nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
tests/gpu.mk:tests/test-pics-gpu-noncart: traj scale phantom ones pics nrmse
tests/gpu.mk:tests/test-pics-gpu-noncart-gridding: traj scale phantom ones pics nrmse
tests/gpu.mk:	$(TOOLDIR)/pics -g --gpu-gridding -S -r0.001 -t traj2.ra ksp.ra o.ra reco2.ra	;\
tests/gpu.mk:tests/test-pics-gpu-weights: pics scale ones nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
tests/gpu.mk:# similar to the non-gpu test this had to be relaxed to 0.01
tests/gpu.mk:tests/test-pics-gpu-noncart-weights: traj scale ones phantom pics nrmse $(TESTS_OUT)/coils.ra
tests/gpu.mk:tests/test-pics-gpu-llr: traj scale phantom ones pics nrmse $(TESTS_OUT)/shepplogan.ra
tests/gpu.mk:tests/test-pics-multigpu: bart copy pics repmat nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
tests/gpu.mk:tests/test-pics-gpu-omp: bart nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
tests/gpu.mk:tests/test-pics-gpu-noncart-weights-omp: bart traj scale ones phantom pics nrmse $(TESTS_OUT)/coils.ra
tests/gpu.mk:tests/test-pics-gpu-mpi: bart nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
tests/gpu.mk:tests/test-pics-gpu-noncart-weights-mpi: bart traj scale ones phantom pics nrmse $(TESTS_OUT)/coils.ra
tests/gpu.mk:tests/test-pics-cart-mpi-gpu: bart pics copy nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
tests/gpu.mk:TESTS_GPU += tests/test-pics-gpu tests/test-pics-gpu-noncart tests/test-pics-gpu-noncart-gridding
tests/gpu.mk:TESTS_GPU += tests/test-pics-gpu-weights tests/test-pics-gpu-noncart-weights tests/test-pics-gpu-llr
tests/gpu.mk:TESTS_GPU += tests/test-pics-multigpu
tests/gpu.mk:TESTS_GPU += tests/test-pics-gpu-mpi tests/test-pics-gpu-noncart-weights-mpi tests/test-pics-cart-mpi-gpu
tests/gpu.mk:TESTS_GPU += tests/test-pics-gpu-omp tests/test-pics-gpu-noncart-weights-omp
tests/network.mk:tests/test-reconet-nnvn-train-gpu: nrmse $(TESTS_OUT)/pattern.ra reconet \
tests/network.mk:tests/test-reconet-nnmodl-train-gpu: nrmse $(TESTS_OUT)/pattern.ra reconet \
tests/network.mk:	CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$(TOOLDIR)/python python $(TOOLDIR)/tests/network_tf2.py ;\
tests/network.mk:	CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$(TOOLDIR)/python python $(TOOLDIR)/tests/network_tf1.py ;\
tests/network.mk:	CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$(TOOLDIR)/python python $(TOOLDIR)/tests/network_tf2.py ;\
tests/network.mk:tests/test-reconet-nnmodl-tensorflow2-gpu: nrmse multicfl $(TESTS_OUT)/pattern.ra reconet \
tests/network.mk:	CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$(TOOLDIR)/python python $(TOOLDIR)/tests/network_tf2.py ;\
tests/network.mk:tests/test-reconet-nnmodl-tensorflow1-gpu: nrmse multicfl $(TESTS_OUT)/pattern.ra reconet \
tests/network.mk:	CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$(TOOLDIR)/python python $(TOOLDIR)/tests/network_tf1.py ;\
tests/network.mk:tests/test-reconet-nnmodl-train-tensorflow2-gpu: nrmse $(TESTS_OUT)/pattern.ra reconet \
tests/network.mk:	CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$(TOOLDIR)/python python $(TOOLDIR)/tests/network_tf2.py ;\
tests/network.mk:TESTS_GPU += tests/test-reconet-nnvn-train-gpu
tests/network.mk:TESTS_GPU += tests/test-reconet-nnmodl-train-gpu
tests/network.mk:TESTS_GPU += tests/test-reconet-nnmodl-tensorflow1-gpu
tests/network.mk:TESTS_GPU += tests/test-reconet-nnmodl-tensorflow2-gpu
tests/network.mk:TESTS_GPU += tests/test-reconet-nnmodl-train-tensorflow2-gpu
tests/mobafit.mk:tests/test-mobafit-gpu: phantom signal reshape fmac index mobafit slice nrmse index extract invert
tests/mobafit.mk:TESTS_GPU += tests/test-mobafit-gpu
tests/nlinv.mk:tests/test-nlinv-gpu: normalize nlinv pocsense nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
tests/nlinv.mk:tests/test-nlinv-sms-gpu: repmat fft nlinv nrmse scale $(TESTS_OUT)/shepplogan_coil_ksp.ra
tests/nlinv.mk:tests/test-nlinv-noncart-fast-gpu: traj scale phantom nufft resize nlinv fmac nrmse
tests/nlinv.mk:TESTS_GPU += tests/test-nlinv-gpu tests/test-nlinv-sms-gpu
tests/moba.mk:tests/test-moba-t1-gpu: phantom signal fmac fft ones index scale moba nrmse
tests/moba.mk:TESTS_GPU += tests/test-moba-t1-gpu
tests/ecalib.mk:tests/test-ecalib-gpu: ecalib pocsense nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
tests/ecalib.mk:TESTS_GPU += tests/test-ecalib-gpu
Makefile:CUDA?=0
Makefile:# cuda
Makefile:CUDA_BASE ?= /usr/
Makefile:CUDA_LIB ?= lib
Makefile:CUDNN_BASE ?= $(CUDA_BASE)
Makefile:# cuda
Makefile:NVCC?=$(CUDA_BASE)/bin/nvcc
Makefile:ifeq ($(CUDA),1)
Makefile:CUDA_H := -I$(CUDA_BASE)/include
Makefile:CPPFLAGS += -DUSE_CUDA $(CUDA_H)
Makefile:CUDA_L := -L$(CUDA_BASE)/$(CUDA_LIB) -lcufft -lcudart -lcublas -m64 -lstdc++
Makefile:CUDA_L := -L$(CUDA_BASE)/$(CUDA_LIB) -L$(CUDNN_BASE)/$(CUDNN_LIB) -lcudnn -lcufft -lcudart -lcublas -lstdc++ -Wl,-rpath $(CUDA_BASE)/$(CUDA_LIB)
Makefile:CUDA_L := -L$(CUDA_BASE)/$(CUDA_LIB) -lcufft -lcudart -lcublas -lstdc++ -Wl,-rpath $(CUDA_BASE)/$(CUDA_LIB)
Makefile:CUDA_H :=
Makefile:CUDA_L :=
Makefile:# sm_20 no longer supported in CUDA 9
Makefile:GPUARCH_FLAGS ?=
Makefile:CUDA_CC ?= $(CC)
Makefile:NVCCFLAGS += -DUSE_CUDA -Xcompiler -fPIC -O2 $(GPUARCH_FLAGS) -I$(srcdir)/ -m64 -ccbin $(CUDA_CC)
Makefile:$(1)cudasrcs := $(wildcard $(srcdir)/$(1)/*.cu)
Makefile:ifeq ($(CUDA),1)
Makefile:$(1)objs += $$($(1)cudasrcs:.cu=.o)
Makefile:UTARGETS_GPU += test_cudafft test_cuda_flpmath test_cuda_flpmath2 test_cuda_gpukrnls test_cuda_convcorr test_cuda_multind test_cuda_shuffle test_cuda_memcache_clear test_cuda_rand
Makefile:UTARGETS_GPU += test_cuda_affine
Makefile:MODULES_test_cuda_affine+= -lmotion -lnlops -llinops -liter
Makefile:	@echo $(patsubst %, /%, $(CTARGETS) $(UTARGETS) $(UTARGETS_GPU)) | tr ' ' '\n' >> .gitignore
Makefile:	$(CC) $(CFLAGS) $(MATLAB_H) -omat2cfl  $+ $(MATLAB_L) $(CUDA_L)
Makefile:	$(LINKER) $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) -Dmain_real=main_$(@F) $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(PNG_L) $(ISMRM_L) $(LIBS) -lm $(LIBRT) -o $@
Makefile:	$(LINKER) $(LDFLAGS) -shared $(CFLAGS) $(CPPFLAGS) -Dmain_real=main_$@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(PNG_L) $(ISMRM_L) $(LIBS) -lm $(LIBRT) -o bart.o
Makefile:	$(LINKER) $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) -Dmain_real=main_$(@F) $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(PNG_L) $(ISMRM_L) $(LIBS) -lm $(LIBRT) -o $@
Makefile:	$(CC) $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) -DUTESTS="$(UTESTS)" $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(LIBS) -lm $(LIBRT) -o $@
Makefile:UTESTS_GPU=$(shell $(root)/utests/utests_gpu-collect.sh ./utests/$@.c)
Makefile:$(UTARGETS_GPU): % : utests/utest.c utests/%.o $$(MODULES_%) $(MODULES)
Makefile:	$(CC) $(LDFLAGS) $(CFLAGS) $(CPPFLAGS) -DUTESTS="$(UTESTS_GPU)" -DUTEST_GPU $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) $(LIBS) -lm $(LIBRT) -o $@
Makefile:#	$(CC) $(LDFLAGS) -Wl,-Tutests/utests.ld $(CFLAGS) -o $@ $+ $(FFTW_L) $(CUDA_L) $(BLAS_L) -lm -rt
Makefile:gputest: ${TESTS_GPU}
Makefile:.PHONY: utests-all utest utests_gpu-all utest_gpu
Makefile:utests_gpu-all: $(UTARGETS_GPU)
Makefile:	./utests/utests_run.sh "GPU" "$(UTEST_RUN)" $(UTARGETS_GPU)
Makefile:utest_gpu: utests_gpu-all
Makefile:	@echo ALL GPU UNIT TESTS PASSED.
Makefile:	rm -f $(patsubst %, %, $(UTARGETS_GPU))
Makefile:	gcc -shared -fopenmp src/bart.o -Wl,-whole-archive lib/lib*.a -Wl,-no-whole-archive -Wl,-Bdynamic $(FFTW_L) $(CUDA_L) $(BLAS_L) $(PNG_L) $(ISMRM_L) $(LIBS) -lm -lrt -o libbart.so
doc/building.txt:multi-GPU system, a HPC cluster, or in the cloud. The build
doc/building.txt:3.3.1. CUDA
doc/building.txt:Support for CUDA can be turned on. It may be necessary to also
doc/building.txt:provide the base path for CUDA installation.
doc/building.txt:CUDA is supported starting with version 8, however, newer versions
doc/building.txt:	CUDA=1
doc/building.txt:	CUDA_BASE=/usr/
doc/building.txt:MPI implementation as this enables automatic detection of CUDA-aware 
doc/singularity/bart_ubuntu.def:	It deploys the code of a specified version and compiles it with GPU support using CUDA.
doc/singularity/bart_ubuntu.def:	Ensure to select the same CUDA version as installed on your host.
doc/singularity/bart_ubuntu.def:	To compile the specified BART version without CUDA support remove `CUDA=1` from the `printf` string below.
doc/singularity/bart_ubuntu.def:	# Install CUDA
doc/singularity/bart_ubuntu.def:	CUDA_VERSION=12-0
doc/singularity/bart_ubuntu.def:	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
doc/singularity/bart_ubuntu.def:	dpkg -i cuda-keyring_1.1-1_all.deb
doc/singularity/bart_ubuntu.def:	DEBIAN_FRONTEND=noninteractive apt-get -y install cuda-${CUDA_VERSION} # `DEBIAN_FRONTEND=noninteractive` avoids keyboard pop-up
doc/singularity/bart_ubuntu.def:	# Install BART and compile it with GPU support
doc/singularity/bart_ubuntu.def:	printf "PARALLEL=1\nCUDA=1\nCUDA_BASE=/usr/local/cuda\nCUDA_LIB=lib64\n" > Makefiles/Makefile.local
doc/singularity/README.md:Both containers download and compile BART with all libraries including GPU support using CUDA.
doc/singularity/README.md:Make sure to select the CUDA version your local HPC host provides.
doc/singularity/README.md:and the `--nv` adds access to the installed Nvidia drivers on the host system.
doc/singularity/bart_debian.def:	It deploys the code of a specified version and compiles it with GPU support using CUDA.
doc/singularity/bart_debian.def:	Ensure to select the same CUDA version as installed on your host.
doc/singularity/bart_debian.def:	To compile the specified BART version without CUDA support remove `CUDA=1` from the `printf` string below.
doc/singularity/bart_debian.def:	# Install CUDA
doc/singularity/bart_debian.def:	CUDA_VERSION=12-3
doc/singularity/bart_debian.def:	wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
doc/singularity/bart_debian.def:	dpkg -i cuda-keyring_1.1-1_all.deb
doc/singularity/bart_debian.def:	DEBIAN_FRONTEND=noninteractive apt-get -y install cuda-${CUDA_VERSION} # `DEBIAN_FRONTEND=noninteractive` avoids keyboard pop-up
doc/singularity/bart_debian.def:	# Install BART and compile it with GPU support
doc/singularity/bart_debian.def:	printf "PARALLEL=1\nCUDA=1\nCUDA_BASE=/usr/local/cuda\nCUDA_LIB=lib64\n" > Makefiles/Makefile.local
doc/webasm.txt:CUDA=0
Makefiles/README.md:# GPU
Makefiles/README.md:CUDA=0
Makefiles/README.md:CUDA_BASE = /usr/local/cuda/
scripts/affine_kspace.sh:GPU=""
scripts/rovir.sh:-g 	use GPU
scripts/rovir.sh:GPU=""
scripts/rovir.sh:		GPU=" -g"
scripts/kspace_precond.sh:-g 	use GPU
scripts/kspace_precond.sh:GPU=""
scripts/kspace_precond.sh:		GPU="-g"
scripts/kspace_precond.sh:bart -l$LOOPFLAGS -r ksp nufft $BASIS $PATTERN -P --lowmem --no-precomp -a $GPU -x$X:$Y:$Z traj2 ksp psf
scripts/kspace_precond.sh:bart -l$LOOPFLAGS -r psf_mul nufft $BASIS $PATTERN -P --lowmem --no-precomp $GPU traj2 psf_mul pre_inv
utests/test_cuda_affine.c:	nlop_interp = nlop_gpu_wrapper_F(nlop_interp);
utests/test_cuda_affine.c:UT_GPU_REGISTER_TEST(test_affine_rot_transpose);
utests/test_cuda_affine.c:	nlop_rot = nlop_gpu_wrapper_F(nlop_rot);
utests/test_cuda_affine.c:UT_GPU_REGISTER_TEST(test_affine_nlop_rot2D);
utests/test_cuda_affine.c:	nlop_interp = nlop_gpu_wrapper_F(nlop_interp);
utests/test_cuda_affine.c:UT_GPU_REGISTER_TEST(test_affine_nlop_interpolate);
utests/test_cuda_affine.c:	nlop_interp = nlop_gpu_wrapper_F(nlop_interp);
utests/test_cuda_affine.c:UT_GPU_REGISTER_TEST(test_affine_nlop_interpolate_coord);
utests/test_cuda_affine.c:	nlop_interp = nlop_gpu_wrapper_F(nlop_interp);
utests/test_cuda_affine.c:UT_GPU_REGISTER_TEST(test_affine_nlop_interpolate_cood_points);
utests/test_cuda_affine.c:	nlop_interp = nlop_gpu_wrapper_F(nlop_interp);
utests/test_cuda_affine.c:UT_GPU_REGISTER_TEST(test_affine_nlop_interpolate_cood_points_keys);
utests/test_cuda_convcorr.c:UT_GPU_REGISTER_TEST(test_convcorr_cf_1D);
utests/test_cuda_convcorr.c:UT_GPU_REGISTER_TEST(test_convcorr_cf_2D);
utests/test_cuda_convcorr.c:UT_GPU_REGISTER_TEST(test_convcorr_cf_3D);
utests/test_cuda_convcorr.c:UT_GPU_REGISTER_TEST(test_convcorr_rand_ord);
utests/test_cuda_convcorr.c:UT_GPU_REGISTER_TEST(test_convcorr_cf_one_channel);
utests/test_cuda_convcorr.c:UT_GPU_REGISTER_TEST(test_convcorr_cf_dil_strs);
utests/test_cuda_flpmath2.c:#ifndef USE_CUDA
utests/test_cuda_flpmath2.c:	complex float* optr1 = md_alloc_gpu(D, odims, size);
utests/test_cuda_flpmath2.c:	complex float* optr2 = md_alloc_gpu(D, odims, size);
utests/test_cuda_flpmath2.c:	complex float* iptr1 = md_alloc_gpu(D, idims1, size);
utests/test_cuda_flpmath2.c:	complex float* iptr2 = md_alloc_gpu(D, idims2, size);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_dot);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_dot2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_gemv);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_gemv2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_gemv3);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_gemm);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_gemm2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_ger);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_ger2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_axpy);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmac2_axpy2);
utests/test_cuda_flpmath2.c://UT _GPU_REGISTER_TEST(test_optimized_md_zfmac2_dot_transp); only activated for large arrays
utests/test_cuda_flpmath2.c:#ifndef USE_CUDA
utests/test_cuda_flpmath2.c:	complex float* optr1 = md_alloc_gpu(D, odims, size);
utests/test_cuda_flpmath2.c:	complex float* optr2 = md_alloc_gpu(D, odims, size);
utests/test_cuda_flpmath2.c:	complex float* iptr1 = md_alloc_gpu(D, idims1, size);
utests/test_cuda_flpmath2.c:	complex float* iptr2 = md_alloc_gpu(D, idims2, size);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_dot);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_dot2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_gemv);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_gemv2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_gemv3);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_gemm);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_gemm2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_ger);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_ger2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_axpy);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zfmacc2_axpy2);
utests/test_cuda_flpmath2.c://UT _GPU_REGISTER_TEST(test_optimized_md_zfmacc2_dot_transp); only activated for large arrays
utests/test_cuda_flpmath2.c:#ifndef USE_CUDA
utests/test_cuda_flpmath2.c:	float* optr1 = md_alloc_gpu(D, odims, CFL_SIZE);
utests/test_cuda_flpmath2.c:	float* optr2 = md_alloc_gpu(D, odims, CFL_SIZE);
utests/test_cuda_flpmath2.c:	float* iptr1 = md_alloc_gpu(D, idims1, CFL_SIZE);
utests/test_cuda_flpmath2.c:	float* iptr2 = md_alloc_gpu(D, idims2, CFL_SIZE);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_dot);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_dot2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_gemv);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_gemv2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_gemv3);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_gemm);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_gemm2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_ger);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_ger2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_axpy);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_fmac2_axpy2);
utests/test_cuda_flpmath2.c:#ifndef USE_CUDA
utests/test_cuda_flpmath2.c:	complex float* optr1 = md_alloc_gpu(D, odims, size);
utests/test_cuda_flpmath2.c:	complex float* optr2 = md_alloc_gpu(D, odims, size);
utests/test_cuda_flpmath2.c:	complex float* iptr1 = md_alloc_gpu(D, idims1, size);
utests/test_cuda_flpmath2.c:	complex float* iptr2 = md_alloc_gpu(D, idims2, size);
utests/test_cuda_flpmath2.c:static bool test_optimized_md_zmul2_smul2(void) { UT_RETURN_ASSERT(test_optimized_md_zmul2_flags(~0ul, 2ul, 3ul, true, 1.e-7)); } // also dgmm on gpu
utests/test_cuda_flpmath2.c:static bool test_optimized_md_zmul2_dgmm(void) { UT_RETURN_ASSERT(test_optimized_md_zmul2_flags(~0ul, 1ul, 3ul, true, 1.e-7)); } // only on gpu
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmul2_smul);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmul2_smul2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmul2_dgmm);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmul2_ger);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmul2_ger2);
utests/test_cuda_flpmath2.c:#ifndef USE_CUDA
utests/test_cuda_flpmath2.c:	complex float* optr1 = md_alloc_gpu(D, odims, size);
utests/test_cuda_flpmath2.c:	complex float* optr2 = md_alloc_gpu(D, odims, size);
utests/test_cuda_flpmath2.c:	complex float* iptr1 = md_alloc_gpu(D, idims1, size);
utests/test_cuda_flpmath2.c:	complex float* iptr2 = md_alloc_gpu(D, idims2, size);
utests/test_cuda_flpmath2.c:static bool test_optimized_md_zmulc2_smul2(void) { UT_RETURN_ASSERT(test_optimized_md_zmulc2_flags(~0ul, 2ul, 3ul, true, 1.e-7)); } // also dgmm on gpu
utests/test_cuda_flpmath2.c:static bool test_optimized_md_zmulc2_dgmm(void) { UT_RETURN_ASSERT(test_optimized_md_zmulc2_flags(~0ul, 1ul, 3ul, true, 1.e-7)); } // only on gpu
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmulc2_smul);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmulc2_smul2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmulc2_dgmm);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmulc2_ger);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmulc2_ger2);
utests/test_cuda_flpmath2.c:#ifndef USE_CUDA
utests/test_cuda_flpmath2.c:	float* optr1 = md_alloc_gpu(D, odims, CFL_SIZE);
utests/test_cuda_flpmath2.c:	float* optr2 = md_alloc_gpu(D, odims, CFL_SIZE);
utests/test_cuda_flpmath2.c:	float* iptr1 = md_alloc_gpu(D, idims1, CFL_SIZE);
utests/test_cuda_flpmath2.c:	float* iptr2 = md_alloc_gpu(D, idims2, CFL_SIZE);
utests/test_cuda_flpmath2.c:static bool test_optimized_md_mul2_smul2(void) { UT_RETURN_ASSERT(test_optimized_md_mul2_flags(~0ul, 2ul, 3ul, true, 1.e-8)); } // also dgmm on gpu
utests/test_cuda_flpmath2.c:static bool test_optimized_md_mul2_dgmm(void) { UT_RETURN_ASSERT(test_optimized_md_mul2_flags(~0ul, 1ul, 3ul, true, 1.e-8)); } // only on gpu
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_mul2_smul);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_mul2_smul2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_mul2_dgmm);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_mul2_ger);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_mul2_ger2);
utests/test_cuda_flpmath2.c:#ifndef USE_CUDA
utests/test_cuda_flpmath2.c:	complex float* optr1 = md_alloc_gpu(D, odims, size);
utests/test_cuda_flpmath2.c:	complex float* optr2 = md_alloc_gpu(D, odims, size);
utests/test_cuda_flpmath2.c:	complex float* iptr1 = md_alloc_gpu(D, idims1, size);
utests/test_cuda_flpmath2.c:	complex float* iptr2 = md_alloc_gpu(D, idims2, size);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zadd2_reduce_inner1);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zadd2_reduce_inner2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zadd2_reduce_inner3);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zadd2_reduce_inner4);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zadd2_reduce_inner5);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zadd2_reduce_outer1);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zadd2_reduce_outer2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zadd2_reduce_outer3);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zadd2_reduce_outer4);
utests/test_cuda_flpmath2.c:#ifndef USE_CUDA
utests/test_cuda_flpmath2.c:	float* optr1 = md_alloc_gpu(D, odims, 2 * size);
utests/test_cuda_flpmath2.c:	float* optr2 = md_alloc_gpu(D, odims, 2 * size);
utests/test_cuda_flpmath2.c:	float* iptr1 = md_alloc_gpu(D, idims1, 2 * size);
utests/test_cuda_flpmath2.c:	float* iptr2 = md_alloc_gpu(D, idims2, 2 * size);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_add2_reduce_inner1);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_add2_reduce_inner2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_add2_reduce_inner3);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_add2_reduce_inner4);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_add2_reduce_inner5);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_add2_reduce_outer1);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_add2_reduce_outer2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_add2_reduce_outer3);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_add2_reduce_outer4);
utests/test_cuda_flpmath2.c:#ifndef USE_CUDA
utests/test_cuda_flpmath2.c:	complex float* optr1 = md_alloc_gpu(D, odims, size);
utests/test_cuda_flpmath2.c:	complex float* optr2 = md_alloc_gpu(D, odims, size);
utests/test_cuda_flpmath2.c:	complex float* iptr1 = md_alloc_gpu(D, idims1, size);
utests/test_cuda_flpmath2.c:	complex float* iptr2 = md_alloc_gpu(D, idims2, size);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmax2_reduce_inner1);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmax2_reduce_inner2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmax2_reduce_inner3);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmax2_reduce_inner4);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmax2_reduce_inner5);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmax2_reduce_outer1);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmax2_reduce_outer2);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmax2_reduce_outer3);
utests/test_cuda_flpmath2.c:UT_GPU_REGISTER_TEST(test_optimized_md_zmax2_reduce_outer4);
utests/test_cuda_multind.c:static bool test_cuda_compress(void)
utests/test_cuda_multind.c:	complex float* _ptr1 = md_alloc_gpu(N, dims, CFL_SIZE);
utests/test_cuda_multind.c:	uint32_t* compress = md_alloc_gpu(1, (long[1]){ M }, sizeof *compress);
utests/test_cuda_multind.c:	float* ptr2 = md_alloc_gpu(N, dims, FL_SIZE);
utests/test_cuda_multind.c:UT_GPU_REGISTER_TEST(test_cuda_compress);
utests/utests_gpu-collect.sh:UTESTS_GPU=$(grep UT_GPU_REGISTER $1 | cut -f2 -d'(' | cut -f1 -d')')
utests/utests_gpu-collect.sh:for i in $UTESTS_GPU; do
utests/test_rand.c:	bool sync_gpu = false; // not need, as it is CPU code
utests/test_rand.c:	run_bench(rounds, print_bench, sync_gpu, f_st);
utests/test_rand.c:	run_bench(rounds, print_bench, sync_gpu, f_mt);
utests/test_rand.c:	run_bench(rounds, print_bench, sync_gpu, f_mt2);
utests/test_flpmath2.c:static bool test_optimized_md_zmul2_smul2(void) { UT_RETURN_ASSERT(test_optimized_md_zmul2_flags(~0ul, 2ul, 3ul, true, 1.e-6)); } // also dgmm on gpu
utests/test_flpmath2.c:static bool test_optimized_md_zmul2_dgmm(void) { UT_RETURN_ASSERT(test_optimized_md_zmul2_flags(~0ul, 1ul, 3ul, false, 1.e-6)); } // only on gpu
utests/test_flpmath2.c:static bool test_optimized_md_zmulc2_smul2(void) { UT_RETURN_ASSERT(test_optimized_md_zmulc2_flags(~0ul, 2ul, 3ul, true, 1.e-6)); } // also dgmm on gpu
utests/test_flpmath2.c:static bool test_optimized_md_zmulc2_dgmm(void) { UT_RETURN_ASSERT(test_optimized_md_zmulc2_flags(~0ul, 1ul, 3ul, false, 1.e-6)); } // only on gpu
utests/test_flpmath2.c:static bool test_optimized_md_mul2_smul2(void) { UT_RETURN_ASSERT(test_optimized_md_mul2_flags(~0ul, 2ul, 3ul, true, 1.e-8)); } // also dgmm on gpu
utests/test_flpmath2.c:static bool test_optimized_md_mul2_dgmm(void) { UT_RETURN_ASSERT(test_optimized_md_mul2_flags(~0ul, 1ul, 3ul, false, 1.e-8)); } // only on gpu
utests/test_cuda_flpmath.c:	complex float* optr_gpu_cpu = md_alloc(N, dims, CFL_SIZE);
utests/test_cuda_flpmath.c:	complex float* optr_gpu = md_alloc_gpu(N, dims, CFL_SIZE);
utests/test_cuda_flpmath.c:	complex float* iptr1_gpu = md_alloc_gpu(N, dims, CFL_SIZE);
utests/test_cuda_flpmath.c:	complex float* iptr2_gpu = md_alloc_gpu(N, dims, CFL_SIZE);
utests/test_cuda_flpmath.c:	md_copy(N, dims, optr_gpu, optr_cpu, CFL_SIZE);
utests/test_cuda_flpmath.c:	md_copy(N, dims, iptr1_gpu, iptr1_cpu, CFL_SIZE);
utests/test_cuda_flpmath.c:	md_copy(N, dims, iptr2_gpu, iptr2_cpu, CFL_SIZE);
utests/test_cuda_flpmath.c:	function(N, dims, strs, optr_gpu, strs, iptr1_gpu, strs, iptr2_gpu);
utests/test_cuda_flpmath.c:	md_copy(N, dims, optr_gpu_cpu, optr_gpu, CFL_SIZE);
utests/test_cuda_flpmath.c:	float err = md_znrmse(N, dims, optr_cpu, optr_gpu_cpu);
utests/test_cuda_flpmath.c:	md_free(optr_gpu_cpu);
utests/test_cuda_flpmath.c:	md_free(optr_gpu);
utests/test_cuda_flpmath.c:	md_free(iptr1_gpu);
utests/test_cuda_flpmath.c:	md_free(iptr2_gpu);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zrmul2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zmul2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zdiv2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zmulc2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zpow2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zfmac2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zfmacc2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_ztenmul2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_ztenmulc2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zadd2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zsub2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zmax2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zlessequal2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zgreatequal2);
utests/test_cuda_flpmath.c:	complex float* optr_gpu_cpu = md_alloc(N, dims, CFL_SIZE);
utests/test_cuda_flpmath.c:	complex float* optr_gpu = md_alloc_gpu(N, dims, CFL_SIZE);
utests/test_cuda_flpmath.c:	complex float* iptr1_gpu = md_alloc_gpu(N, dims, CFL_SIZE);
utests/test_cuda_flpmath.c:	md_copy(N, dims, optr_gpu, optr_cpu, CFL_SIZE);
utests/test_cuda_flpmath.c:	md_copy(N, dims, iptr1_gpu, iptr1_cpu, CFL_SIZE);
utests/test_cuda_flpmath.c:	function(N, dims, strs, optr_gpu, strs, iptr1_gpu);
utests/test_cuda_flpmath.c:	md_copy(N, dims, optr_gpu_cpu, optr_gpu, CFL_SIZE);
utests/test_cuda_flpmath.c:	float err = md_znrmse(N, dims, optr_cpu, optr_gpu_cpu);
utests/test_cuda_flpmath.c:	md_free(optr_gpu_cpu);
utests/test_cuda_flpmath.c:	md_free(optr_gpu);
utests/test_cuda_flpmath.c:	md_free(iptr1_gpu);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zsqrt2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zabs2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zconj2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zreal2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zimag2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zexpj2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zexp2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zlog2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zarg2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zsin2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zcos2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zsinh2);
utests/test_cuda_flpmath.c:UT_GPU_REGISTER_TEST(test_md_zcosh2);
utests/test_cuda_shuffle.c:struct ptr_cpugpu {
utests/test_cuda_shuffle.c:	complex float* gpu;
utests/test_cuda_shuffle.c:static struct ptr_cpugpu alloc_pair_rand(int N, const long dims[N])
utests/test_cuda_shuffle.c:	struct ptr_cpugpu ret = {
utests/test_cuda_shuffle.c:		.gpu = md_alloc_gpu(N, dims, CFL_SIZE)
utests/test_cuda_shuffle.c:	md_copy(N, dims, ret.gpu, ret.cpu, CFL_SIZE);
utests/test_cuda_shuffle.c:static struct ptr_cpugpu alloc_pair_zero(int N, const long dims[N])
utests/test_cuda_shuffle.c:	struct ptr_cpugpu ret = {
utests/test_cuda_shuffle.c:		.gpu = md_alloc_gpu(N, dims, CFL_SIZE)
utests/test_cuda_shuffle.c:	md_clear(N, dims, ret.gpu, CFL_SIZE);
utests/test_cuda_shuffle.c:static float cmp_pair_F(int N, const long dims[N], struct ptr_cpugpu x)
utests/test_cuda_shuffle.c:	md_copy(N, dims, tmp, x.gpu, CFL_SIZE);
utests/test_cuda_shuffle.c:	md_free(x.gpu);
utests/test_cuda_shuffle.c:static void free_pair(struct ptr_cpugpu x)
utests/test_cuda_shuffle.c:	md_free(x.gpu);
utests/test_cuda_shuffle.c:static bool test_cuda_decompose1(void)
utests/test_cuda_shuffle.c:	struct ptr_cpugpu in = alloc_pair_rand(N, dims);
utests/test_cuda_shuffle.c:	struct ptr_cpugpu out = alloc_pair_zero(N + 1, odims);
utests/test_cuda_shuffle.c:	md_decompose(N, factors, odims, out.gpu, dims, in.gpu, CFL_SIZE);
utests/test_cuda_shuffle.c:static bool test_cuda_decompose2(void)
utests/test_cuda_shuffle.c:	struct ptr_cpugpu in = alloc_pair_rand(N, dims);
utests/test_cuda_shuffle.c:	struct ptr_cpugpu out = alloc_pair_zero(N + 1, odims);
utests/test_cuda_shuffle.c:	md_decompose(N, factors, odims, out.gpu, dims, in.gpu, CFL_SIZE);
utests/test_cuda_shuffle.c:static bool test_cuda_recompose1(void)
utests/test_cuda_shuffle.c:	struct ptr_cpugpu in = alloc_pair_rand(N + 1, odims);
utests/test_cuda_shuffle.c:	struct ptr_cpugpu out = alloc_pair_zero(N, dims);
utests/test_cuda_shuffle.c:	md_recompose(N, factors, dims, out.gpu, odims, in.gpu, CFL_SIZE);
utests/test_cuda_shuffle.c:static bool test_cuda_recompose2(void)
utests/test_cuda_shuffle.c:	struct ptr_cpugpu in = alloc_pair_rand(N + 1, odims);
utests/test_cuda_shuffle.c:	struct ptr_cpugpu out = alloc_pair_zero(N, dims);
utests/test_cuda_shuffle.c:	md_recompose(N, factors, dims, out.gpu, odims, in.gpu, CFL_SIZE);
utests/test_cuda_shuffle.c:UT_GPU_REGISTER_TEST(test_cuda_decompose1);
utests/test_cuda_shuffle.c:UT_GPU_REGISTER_TEST(test_cuda_decompose2);
utests/test_cuda_shuffle.c:UT_GPU_REGISTER_TEST(test_cuda_recompose1);
utests/test_cuda_shuffle.c:UT_GPU_REGISTER_TEST(test_cuda_recompose2);
utests/test_cudafft.c:static bool run_cuda_fft_test(const int D, const long* dims, const unsigned long flags,
utests/test_cudafft.c:			       complex float* gpu_inout, complex float* gpu_result)
utests/test_cudafft.c:	md_copy(D, dims, gpu_inout, in, CFL_SIZE);
utests/test_cudafft.c:	fft_exec(fftplan, gpu_inout, gpu_inout);
utests/test_cudafft.c:	md_copy(D, dims, gpu_result, gpu_inout, CFL_SIZE);
utests/test_cudafft.c:	UT_RETURN_ASSERT(md_znrmse(D, dims, cpu_inout, gpu_result) < UT_TOL);
utests/test_cudafft.c:static bool test_cuda_fft(void)
utests/test_cudafft.c:#ifndef USE_CUDA
utests/test_cudafft.c:	// TODO: detect if GPU works
utests/test_cudafft.c:	enum { test_cuda_fft_dims = 7 };
utests/test_cudafft.c:	const long dims[test_cuda_fft_dims] = { 4, 4, 4, 4, 4, 4, 1 }; // in last dim != 1 works...
utests/test_cudafft.c:	const bool transform_dims[][test_cuda_fft_dims] = {
utests/test_cudafft.c:	const unsigned int D = test_cuda_fft_dims;
utests/test_cudafft.c:	complex float* gpu_inout = md_alloc_gpu(D, dims, CFL_SIZE);
utests/test_cudafft.c:	complex float* gpu_result = md_alloc(D, dims, CFL_SIZE);
utests/test_cudafft.c:		run_cuda_fft_test(D, dims, flags, in, cpu_inout, gpu_inout, gpu_result);
utests/test_cudafft.c:	md_free(gpu_result);
utests/test_cudafft.c:	md_free(gpu_inout);
utests/test_cudafft.c:UT_GPU_REGISTER_TEST(test_cuda_fft);
utests/test_cudafft.c:static bool test_cuda_fftmod(void)
utests/test_cudafft.c:#ifndef USE_CUDA
utests/test_cudafft.c:	// TODO: detect if GPU works
utests/test_cudafft.c:	complex float* gpu = md_gpu_move(DIMS, dims, cpu1, CFL_SIZE);
utests/test_cudafft.c:	fftmod(DIMS, dims, 15, gpu, gpu);
utests/test_cudafft.c:	md_copy(DIMS, dims, cpu2, gpu, CFL_SIZE);
utests/test_cudafft.c:UT_GPU_REGISTER_TEST(test_cuda_fftmod);
utests/test_cudafft.c:static bool test_cuda_fftmod2(void)
utests/test_cudafft.c:#ifndef USE_CUDA
utests/test_cudafft.c:	// TODO: detect if GPU works
utests/test_cudafft.c:	complex float* gpu = md_gpu_move(DIMS, dims, cpu1, CFL_SIZE);
utests/test_cudafft.c:	fftmod(DIMS, dims, 15, gpu, gpu);
utests/test_cudafft.c:	md_copy(DIMS, dims, cpu2, gpu, CFL_SIZE);
utests/test_cudafft.c:UT_GPU_REGISTER_TEST(test_cuda_fftmod2);
utests/utest.c:#ifdef UTEST_GPU
utests/utest.c:	bart_use_gpu = true;
utests/utest.c:	num_init_gpu_support();
utests/test_cuda_gpukrnls.c:#include "num/gpukrnls.h"
utests/test_cuda_gpukrnls.c:#include "num/gpu_conv.h"
utests/test_cuda_gpukrnls.c:#include "num/gpuops.h"
utests/test_cuda_gpukrnls.c:	complex float* in_gpu = md_alloc_gpu(N, idims, CFL_SIZE);
utests/test_cuda_gpukrnls.c:	md_copy(N, idims, in_gpu, in_cpu, CFL_SIZE);
utests/test_cuda_gpukrnls.c:	complex float* imat_gpu = md_alloc_gpu(N + 3, idims_mat, CFL_SIZE);
utests/test_cuda_gpukrnls.c:	cuda_im2col(imat_gpu, in_gpu, odims, idims, kdims, NULL, NULL);
utests/test_cuda_gpukrnls.c:	complex float* imat_gpu_cpu = md_alloc(N + 3, idims_mat, CFL_SIZE);
utests/test_cuda_gpukrnls.c:	md_copy(N + 3, idims_mat, imat_gpu_cpu, imat_gpu, CFL_SIZE);
utests/test_cuda_gpukrnls.c:	float err = md_zrmse(N + 3, idims_mat, imat_gpu_cpu, imat_cpu);
utests/test_cuda_gpukrnls.c:	debug_printf(DP_DEBUG1, "%f, %f, %f\n", err, md_zrms(N + 3, idims_mat, imat_cpu), md_zrms(N + 3, idims_mat, imat_gpu_cpu));
utests/test_cuda_gpukrnls.c:	md_free(in_gpu);
utests/test_cuda_gpukrnls.c:	md_free(imat_gpu);
utests/test_cuda_gpukrnls.c:	md_free(imat_gpu_cpu);
utests/test_cuda_gpukrnls.c:UT_GPU_REGISTER_TEST(test_im2col_loop_in);
utests/test_cuda_gpukrnls.c:	complex float* in_gpu = md_alloc_gpu(N, idims, CFL_SIZE);
utests/test_cuda_gpukrnls.c:	md_copy(N, idims, in_gpu, in_cpu, CFL_SIZE);
utests/test_cuda_gpukrnls.c:	complex float* imat_gpu = md_alloc_gpu(N + 3, idims_mat, CFL_SIZE);
utests/test_cuda_gpukrnls.c:	cuda_im2col(imat_gpu, in_gpu, odims, idims, kdims, NULL, NULL);
utests/test_cuda_gpukrnls.c:	complex float* imat_gpu_cpu = md_alloc(N + 3, idims_mat, CFL_SIZE);
utests/test_cuda_gpukrnls.c:	md_copy(N + 3, idims_mat, imat_gpu_cpu, imat_gpu, CFL_SIZE);
utests/test_cuda_gpukrnls.c:	float err = md_zrmse(N + 3, idims_mat, imat_gpu_cpu, imat_cpu);
utests/test_cuda_gpukrnls.c:	debug_printf(DP_DEBUG1, "%f, %f, %f\n", err, md_zrms(N + 3, idims_mat, imat_cpu), md_zrms(N + 3, idims_mat, imat_gpu_cpu));
utests/test_cuda_gpukrnls.c:	md_free(in_gpu);
utests/test_cuda_gpukrnls.c:	md_free(imat_gpu);
utests/test_cuda_gpukrnls.c:	md_free(imat_gpu_cpu);
utests/test_cuda_gpukrnls.c:UT_GPU_REGISTER_TEST(test_im2col_loop_out);
utests/test_cuda_gpukrnls.c:	complex float* in_gpu = md_alloc_gpu(N, idims, CFL_SIZE);
utests/test_cuda_gpukrnls.c:	md_copy(N, idims, in_gpu, in_cpu, CFL_SIZE);
utests/test_cuda_gpukrnls.c:	complex float* imat_gpu = md_alloc_gpu(N + 3, idims_mat, CFL_SIZE);
utests/test_cuda_gpukrnls.c:	md_copy(N + 3, idims_mat, imat_gpu, imat_cpu, CFL_SIZE);
utests/test_cuda_gpukrnls.c:	cuda_im2col_transp(in_gpu, imat_gpu, odims, idims, kdims, NULL, NULL);
utests/test_cuda_gpukrnls.c:	complex float* in_gpu_cpu = md_alloc(5, idims, CFL_SIZE);
utests/test_cuda_gpukrnls.c:	md_copy(5, idims, in_gpu_cpu, in_gpu, CFL_SIZE);
utests/test_cuda_gpukrnls.c:	float err = md_znrmse(5, idims, in_gpu_cpu, in_cpu);
utests/test_cuda_gpukrnls.c:	debug_printf(DP_DEBUG1, "%f, %f, %f\n", err, md_zrms(5, idims, in_cpu), md_zrms(5, idims, in_gpu_cpu));
utests/test_cuda_gpukrnls.c:	md_free(in_gpu);
utests/test_cuda_gpukrnls.c:	md_free(imat_gpu);
utests/test_cuda_gpukrnls.c:	md_free(in_gpu_cpu);
utests/test_cuda_gpukrnls.c:UT_GPU_REGISTER_TEST(test_im2col_adj);
utests/utest.h:#define UT_GPU_REGISTER_TEST(x) UT_REGISTER_TEST(x)
utests/test_cuda_rand.c:#include "num/gpuops.h"
utests/test_cuda_rand.c:static bool test_cuda_uniform_rand(void)
utests/test_cuda_rand.c:#ifndef USE_CUDA
utests/test_cuda_rand.c:	// TODO: detect if GPU works
utests/test_cuda_rand.c:	complex float* gpu = md_alloc_gpu(N, dims, CFL_SIZE);
utests/test_cuda_rand.c:	complex float* gpu_result = md_alloc(N, dims, CFL_SIZE);
utests/test_cuda_rand.c:	md_uniform_rand(N, dims, gpu);
utests/test_cuda_rand.c:	md_copy(N, dims, gpu_result, gpu, CFL_SIZE);
utests/test_cuda_rand.c:	float err = md_znrmse(N, dims, cpu, gpu_result);
utests/test_cuda_rand.c:	md_free(gpu);
utests/test_cuda_rand.c:	md_free(gpu_result);
utests/test_cuda_rand.c:UT_GPU_REGISTER_TEST(test_cuda_uniform_rand);
utests/test_cuda_rand.c:static bool test_cuda_gaussian_rand(void)
utests/test_cuda_rand.c:#ifndef USE_CUDA
utests/test_cuda_rand.c:	// TODO: detect if GPU works
utests/test_cuda_rand.c:	complex float* gpu = md_alloc_gpu(N, dims, CFL_SIZE);
utests/test_cuda_rand.c:	complex float* gpu_result = md_alloc(N, dims, CFL_SIZE);
utests/test_cuda_rand.c:	md_gaussian_rand(N, dims, gpu);
utests/test_cuda_rand.c:	md_copy(N, dims, gpu_result, gpu, CFL_SIZE);
utests/test_cuda_rand.c:	float err = md_znrmse(N, dims, cpu, gpu_result);
utests/test_cuda_rand.c:	debug_printf((err != 0) ? DP_INFO : DP_DEBUG1, "test_cuda_gaussian_rand: error: %.8e, tol %.1e\n", err, 0.0);
utests/test_cuda_rand.c:	md_free(gpu);
utests/test_cuda_rand.c:	md_free(gpu_result);
utests/test_cuda_rand.c:UT_GPU_REGISTER_TEST(test_cuda_gaussian_rand);
utests/test_cuda_rand.c:static bool test_cuda_rand_one(void)
utests/test_cuda_rand.c:#ifndef USE_CUDA
utests/test_cuda_rand.c:	// TODO: detect if GPU works
utests/test_cuda_rand.c:	complex float* gpu = md_alloc_gpu(N, dims, CFL_SIZE);
utests/test_cuda_rand.c:	complex float* gpu_result = md_alloc(N, dims, CFL_SIZE);
utests/test_cuda_rand.c:	md_rand_one(N, dims, gpu, p);
utests/test_cuda_rand.c:	md_copy(N, dims, gpu_result, gpu, CFL_SIZE);
utests/test_cuda_rand.c:	UT_RETURN_ON_FAILURE(0 == md_znrmse(N, dims, cpu, gpu_result));
utests/test_cuda_rand.c:	md_free(gpu);
utests/test_cuda_rand.c:	md_free(gpu_result);
utests/test_cuda_rand.c:UT_GPU_REGISTER_TEST(test_cuda_rand_one);
utests/test_cuda_rand.c:static bool test_cuda_rand(md_rand_t function, const char* name, double tol)
utests/test_cuda_rand.c:	complex float* mt_gpu = md_alloc_gpu(N2, dims2, CFL_SIZE);
utests/test_cuda_rand.c:	complex float* mt_gpu_cpu = md_calloc(N2, dims2, CFL_SIZE);
utests/test_cuda_rand.c:	bool sync_gpu = true; // not need, as it is CPU code
utests/test_cuda_rand.c:	run_bench(rounds, print_bench, sync_gpu, f_mt);
utests/test_cuda_rand.c:	NESTED(void, f_mt_gpu, (void))
utests/test_cuda_rand.c:		function(N2, dims2, mt_gpu);
utests/test_cuda_rand.c:		bart_printf("\t\t\t\t\t\t\t\t\t  GPU: ");
utests/test_cuda_rand.c:	run_bench(rounds, print_bench, sync_gpu, f_mt_gpu);
utests/test_cuda_rand.c:	md_copy(N2, dims2, mt_gpu_cpu, mt_gpu, CFL_SIZE);
utests/test_cuda_rand.c:	double err = md_znrmse(N2, dims2, mt, mt_gpu_cpu);
utests/test_cuda_rand.c:	md_free(mt_gpu);
utests/test_cuda_rand.c:	md_free(mt_gpu_cpu);
utests/test_cuda_rand.c:	debug_printf((err > tol) ? DP_INFO : DP_DEBUG1, "test_cuda: %s, error: %.8e, tol %.1e\n", name, err, tol);
utests/test_cuda_rand.c:static bool test_mt_cuda_uniform_rand(void) { return test_cuda_rand(md_uniform_rand, " uniform", 0.0);}
utests/test_cuda_rand.c:UT_GPU_REGISTER_TEST(test_mt_cuda_uniform_rand);
utests/test_cuda_rand.c:static bool test_mt_cuda_gaussian_rand(void) { return test_cuda_rand(md_gaussian_rand, "gaussian", 1e-11);}
utests/test_cuda_rand.c:UT_GPU_REGISTER_TEST(test_mt_cuda_gaussian_rand);
utests/test_cuda_memcache_clear.c:static bool test_cuda_memcache_clear(void)
utests/test_cuda_memcache_clear.c:#ifndef USE_CUDA
utests/test_cuda_memcache_clear.c:	// TODO: detect if GPU works
utests/test_cuda_memcache_clear.c:	complex float* ptr1 = md_alloc_gpu(D, dims, CFL_SIZE);
utests/test_cuda_memcache_clear.c:	complex float* ptr2 = md_alloc_gpu(D, dims, CFL_SIZE);
utests/test_cuda_memcache_clear.c:	num_deinit_gpu();
utests/test_cuda_memcache_clear.c:UT_GPU_REGISTER_TEST(test_cuda_memcache_clear);
.gitlab-ci.yml:image: registry.gitlab.tugraz.at/ibi/reproducibility/gitlab-ci-containers/ibi_cuda_bart
.gitlab-ci.yml:Build_Clang_GPU:
.gitlab-ci.yml:    - CC=clang-16 CUDA_CC=clang-14 CUDA=1 CUDA_LIB=lib64 GPUARCH_FLAGS="-arch sm_35 -Wno-deprecated-gpu-targets" make all
.gitlab-ci.yml:Build_Shared_GPU:
.gitlab-ci.yml:    - CUDA=1 CUDA_LIB=lib64 GPUARCH_FLAGS="-arch sm_35 -Wno-deprecated-gpu-targets" make libbart.so
.gitlab-ci.yml:Build_GPU:
.gitlab-ci.yml:    - CUDA=1 CUDA_LIB=lib64 GPUARCH_FLAGS="-arch sm_35 -Wno-deprecated-gpu-targets" WERROR=1 make all
.gitlab-ci.yml:Build_MPI_GPU:
.gitlab-ci.yml:    - MPI=1 CUDA=1 CUDA_LIB=lib64 GPUARCH_FLAGS="-arch sm_35 -Wno-deprecated-gpu-targets" make all
.gitlab-ci.yml:#    - wget --no-verbose https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.4.0.tar.gz
.gitlab-ci.yml:#    - mkdir tensorflow_dir && tar -C tensorflow_dir -xvzf libtensorflow-gpu-linux-x86_64-2.4.0.tar.gz
.gitlab-ci.yml:    - TENSORFLOW=1 TENSORFLOW_BASE=/tensorflow_dir/ CUDA=1 CUDA_LIB=lib64 GPUARCH_FLAGS="-arch sm_35 -Wno-deprecated-gpu-targets" make all
.gitlab-ci.yml:        CUDA=0\n
.gitlab-ci.yml:UTest_Clang_GPU:
.gitlab-ci.yml:    - if ! nvidia-smi ; then printf "No usable GPU found, skipping GPU tests!\n"; exit 0; fi
.gitlab-ci.yml:    - AUTOCLEAN=0 CC=clang-16 CUDA_CC=clang-14 CUDA=1 CUDA_LIB=lib64 GPUARCH_FLAGS="-arch sm_35 -Wno-deprecated-gpu-targets" make utest_gpu
.gitlab-ci.yml:  needs: [Build_Clang_GPU]
.gitlab-ci.yml:    - Build_Clang_GPU
.gitlab-ci.yml:UTest_GPU:
.gitlab-ci.yml:    - if ! nvidia-smi ; then printf "No usable GPU found, skipping GPU tests!\n"; exit 0; fi
.gitlab-ci.yml:    - AUTOCLEAN=0 CUDA=1 CUDA_LIB=lib64 GPUARCH_FLAGS="-arch sm_35 -Wno-deprecated-gpu-targets" WERROR=1 make utest_gpu
.gitlab-ci.yml:  needs: [Build_GPU]
.gitlab-ci.yml:    - Build_GPU
.gitlab-ci.yml:#    - wget --no-verbose https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.4.0.tar.gz
.gitlab-ci.yml:#    - mkdir tensorflow_dir && tar -C tensorflow_dir -xvzf libtensorflow-gpu-linux-x86_64-2.4.0.tar.gz
.gitlab-ci.yml:    - AUTOCLEAN=0 TENSORFLOW=1 TENSORFLOW_BASE=/tensorflow_dir/ CUDA=1 CUDA_LIB=lib64 GPUARCH_FLAGS="-arch sm_35 -Wno-deprecated-gpu-targets" make utest
.gitlab-ci.yml:IntTest_GPU:
.gitlab-ci.yml:    - if ! nvidia-smi ; then printf "No usable GPU found, skipping GPU tests!\n"; exit 0; fi
.gitlab-ci.yml:    - AUTOCLEAN=0 CUDA=1 CUDA_LIB=lib64 GPUARCH_FLAGS="-arch sm_35 -Wno-deprecated-gpu-targets" WERROR=1 make gputest
.gitlab-ci.yml:  needs: [Build_GPU]
.gitlab-ci.yml:    - Build_GPU
.gitlab-ci.yml:IntTest_Clang_GPU:
.gitlab-ci.yml:    - if ! nvidia-smi ; then printf "No usable GPU found, skipping GPU tests!\n"; exit 0; fi
.gitlab-ci.yml:    - AUTOCLEAN=0 CC=clang-16 CUDA_CC=clang-14 CUDA=1 CUDA_LIB=lib64 GPUARCH_FLAGS="-arch sm_35 -Wno-deprecated-gpu-targets" make gputest
.gitlab-ci.yml:  needs: [Build_Clang_GPU]
.gitlab-ci.yml:    - Build_Clang_GPU
.gitlab-ci.yml:IntTest_MPI_GPU:
.gitlab-ci.yml:    - if ! nvidia-smi ; then printf "No usable GPU found, skipping GPU tests!\n"; exit 0; fi
.gitlab-ci.yml:    - AUTOCLEAN=0 CUDA=1 CUDA_LIB=lib64 GPUARCH_FLAGS="-arch sm_35 -Wno-deprecated-gpu-targets" MPI=1 make gputest
.gitlab-ci.yml:  needs: [Build_MPI_GPU]
.gitlab-ci.yml:    - Build_MPI_GPU
README:You can also try the package built with CUDA support:
README:    $ sudo apt-get install bart-cuda bart-view
README:GCC compiler, the FFTW library, and optionally CUDA.
src/interpolate.c:	num_init_gpu_support();
src/num/ops.h:extern const struct operator_s* operator_gpu_wrapper2(const struct operator_s* op, unsigned long move_flags);
src/num/ops.h:extern const struct operator_s* operator_gpu_wrapper(const struct operator_s* op);
src/num/vptr.c:#ifdef USE_CUDA
src/num/vptr.c:#include "num/gpuops.h"
src/num/vptr.c:	bool gpu;
src/num/vptr.c:#ifdef USE_CUDA
src/num/vptr.c:	if (cuda_ondevice(si->si_addr))
src/num/vptr.c:		error("Tried to access CUDA pointer at %x from CPU!\n", si->si_addr);
src/num/vptr.c:	x->gpu = false;
src/num/vptr.c:#ifdef USE_CUDA
src/num/vptr.c:		if (mem->gpu)
src/num/vptr.c:			mem->mem[idx] = cuda_malloc(mem->block_size);
src/num/vptr.c:bool is_vptr_gpu(const void* ptr)
src/num/vptr.c:	return mem && mem->gpu;
src/num/vptr.c:	return mem && !mem->gpu;
src/num/vptr.c:	ret->gpu = mem->gpu;
src/num/vptr.c:void* vptr_move_gpu(const void* ptr)
src/num/vptr.c:	ret->gpu = true;
src/num/vptr.c:	ret->gpu = false;
src/num/vptr.c:#ifdef USE_CUDA
src/num/vptr.c:	mem->gpu = cuda_ondevice(ptr);
src/num/fft.c:#ifdef USE_CUDA
src/num/fft.c:#include "num/gpuops.h"
src/num/fft.c:#include "num/gpukrnls.h"
src/num/fft.c:#include "fft-cuda.h"
src/num/fft.c:#define LAZY_CUDA
src/num/fft.c:		#ifdef USE_CUDA
src/num/fft.c:			if (cuda_ondevice(dst))
src/num/fft.c:				cuda_zfftmod_3d(tptr, ptr[0], ptr[1], inv, phase);
src/num/fft.c:	/* this will also currently be slow on the GPU because we do not
src/num/fft.c:#ifdef  USE_CUDA
src/num/fft.c:	struct fft_cuda_plan_s* cuplan;
src/num/fft.c:#ifdef  USE_CUDA
src/num/fft.c:	if (cuda_ondevice(src)) {
src/num/fft.c:#ifdef	LAZY_CUDA
src/num/fft.c:			plan->cuplan = fft_cuda_plan(plan->D, plan->dims, plan->flags, plan->ostrs, plan->istrs, plan->backwards);
src/num/fft.c:			error("Failed to plan a GPU FFT (too large?)\n");
src/num/fft.c:		fft_cuda_exec(plan->cuplan, dst, src);
src/num/fft.c:#ifdef	USE_CUDA
src/num/fft.c:		fft_cuda_free_plan(plan->cuplan);
src/num/fft.c:#ifdef  USE_CUDA
src/num/fft.c:#ifndef LAZY_CUDA
src/num/fft.c:	if (cuda_ondevice(src) && (0u != flags)
src/num/fft.c:		plan->cuplan = fft_cuda_plan(D, dimensions, flags, strides, strides, backwards);
src/num/fft.c:#ifdef  USE_CUDA
src/num/fft.c:#ifndef LAZY_CUDA
src/num/fft.c:	if (cuda_ondevice(src) && (0u != flags)
src/num/fft.c:		plan->cuplan = fft_cuda_plan(D, dimensions, flags, ostrides, istrides, backwards);
src/num/mem.c:#ifdef USE_CUDA
src/num/mem.c:#include "num/gpuops.h"
src/num/mem.c:#ifndef USE_CUDA
src/num/mem.c:#define CUDA_MAX_STREAMS 0
src/num/mem.c:static int cuda_get_stream_id(void)
src/num/mem.c:static long unused_memory[CUDA_MAX_STREAMS + 1] = { 0 };
src/num/mem.c:static long used_memory[CUDA_MAX_STREAMS + 1] = { 0 };
src/num/mem.c:static tree_t mem_allocs[CUDA_MAX_STREAMS + 1] = { NULL };
src/num/mem.c:static tree_t mem_cache[CUDA_MAX_STREAMS + 1] = { NULL };
src/num/mem.c:static const void* min_ptr[CUDA_MAX_STREAMS + 1] = { NULL };
src/num/mem.c:static const void* max_ptr[CUDA_MAX_STREAMS + 1] = { NULL };
src/num/mem.c:	for (int i = 0; i < CUDA_MAX_STREAMS + 1; i++) {
src/num/mem.c:	for (int i = 0; i < CUDA_MAX_STREAMS + 1; i++) {
src/num/mem.c:	for (int i = 0; i < CUDA_MAX_STREAMS + 1; i++) {
src/num/mem.c:	for (int i = 0; i < CUDA_MAX_STREAMS + 1; i++) {
src/num/mem.c:	struct mem_s* nptr = find_free(0, cuda_get_stream_id());
src/num/mem.c:		nptr = find_free(0, cuda_get_stream_id());
src/num/mem.c:	int stream = cuda_get_stream_id();
src/num/mem.c:	if (stream != CUDA_MAX_STREAMS)
src/num/mem.c:		return (NULL != search(ptr, false, cuda_get_stream_id())) || (NULL != search(ptr, false, CUDA_MAX_STREAMS));
src/num/mem.c:	if (NULL != search(ptr, false, CUDA_MAX_STREAMS))
src/num/mem.c:	for (int i = 0; i < CUDA_MAX_STREAMS; i++)
src/num/mem.c:	for (int i = 0; i < CUDA_MAX_STREAMS + 1; i++) {
src/num/mem.c:		if (i != cuda_get_stream_id() && CUDA_MAX_STREAMS != cuda_get_stream_id())
src/num/mem.c:	int stream = cuda_get_stream_id();
src/num/optimize.h:extern int optimize_dims_gpu(int D, int N, long dims[N], long (*strs[D])[N]);
src/num/fft-cuda.h:struct fft_cuda_plan_s;
src/num/fft-cuda.h:extern struct fft_cuda_plan_s* fft_cuda_plan(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], const long istrides[D], _Bool dir);
src/num/fft-cuda.h:extern void fft_cuda_free_plan(struct fft_cuda_plan_s* cuplan);
src/num/fft-cuda.h:extern void fft_cuda_exec(struct fft_cuda_plan_s* cuplan, complex float* dst, const complex float* src);
src/num/md_wrapper.h:void zfmac_gpu_batched_loop(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
src/num/md_wrapper.h:void zfmacc_gpu_batched_loop(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
src/num/md_wrapper.h:void add_gpu_unfold(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
src/num/md_wrapper.h:void zadd_gpu_unfold(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
src/num/md_wrapper.h:void mul_gpu_unfold(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
src/num/md_wrapper.h:void zmul_gpu_unfold(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
src/num/md_wrapper.h:void zmulc_gpu_unfold(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
src/num/md_wrapper.h:void fmac_gpu_unfold(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
src/num/md_wrapper.h:void zfmac_gpu_unfold(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
src/num/md_wrapper.h:void zfmacc_gpu_unfold(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:#include "num/gpuops.h"
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c://These pointers can point to cpu or gpu memory.
src/num/blas.c:#define CUBLAS_CALL(x)		({ CUDA_ASYNC_ERROR_NOTE("before cuBLAS call"); cublas_set_gpulock(); cublasStatus_t errval = (x); cublas_unset_gpulock(); if (CUBLAS_STATUS_SUCCESS != errval) cublas_error(__FILE__, __LINE__, errval); CUDA_ASYNC_ERROR_NOTE("after cuBLAS call"); })
src/num/blas.c:static cublasHandle_t handle_host[CUDA_MAX_STREAMS + 1];
src/num/blas.c:static cublasHandle_t handle_device[CUDA_MAX_STREAMS + 1];
src/num/blas.c://	 However, tests/test-pics-multigpu fails with too many (16) threads
src/num/blas.c:static omp_lock_t gpulock[CUDA_MAX_STREAMS + 1];;
src/num/blas.c:static void cublas_set_gpulock(void)
src/num/blas.c:	omp_set_lock(&(gpulock[cuda_get_stream_id()]));
src/num/blas.c:static void cublas_unset_gpulock(void)
src/num/blas.c:	omp_unset_lock(&(gpulock[cuda_get_stream_id()]));
src/num/blas.c:static void cublas_set_gpulock(void)
src/num/blas.c:static void cublas_unset_gpulock(void)
src/num/blas.c:	for (int i = 0; i < CUDA_MAX_STREAMS + 1; i++) {
src/num/blas.c:		CUBLAS_ERROR(cublasSetStream(handle_host[i], cuda_get_stream_by_id(i)));
src/num/blas.c:		CUBLAS_ERROR(cublasSetStream(handle_device[i], cuda_get_stream_by_id(i)));
src/num/blas.c:		omp_init_lock(&(gpulock[i]));
src/num/blas.c:	for (int i = 0; i < CUDA_MAX_STREAMS + 1; i++) {
src/num/blas.c:		omp_destroy_lock(&(gpulock[i]));
src/num/blas.c:	return handle_device[cuda_get_stream_id()];
src/num/blas.c:	return handle_host[cuda_get_stream_id()];
src/num/blas.c:double cuda_asum(long size, const float* src)
src/num/blas.c:void cuda_saxpy(long size, float* y, float alpha, const float* src)
src/num/blas.c:void cuda_swap(long size, float* a, float* b)
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(x)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(x)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(x)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(x)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(x)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(x)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(x)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(x)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(x)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(x)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:		complex float* zero = cuda_malloc(8);
src/num/blas.c:		cuda_clear(8, zero);
src/num/blas.c:		cuda_free(zero);
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:#ifdef USE_CUDA
src/num/blas.c:	if (cuda_ondevice(A)) {
src/num/blas.c:		float* zero = cuda_malloc(4);
src/num/blas.c:		cuda_clear(4, zero);
src/num/blas.c:		cuda_free(zero);
src/num/loop.c: * No GPU support at the moment!
src/num/loop.c:#ifdef USE_CUDA
src/num/loop.c:#include "num/gpuops.h"
src/num/loop.c:#ifdef USE_CUDA
src/num/loop.c:	if (cuda_ondevice(out)) {
src/num/flpmath.c: * All functions should work on CPU and GPU.
src/num/flpmath.c:#ifdef USE_CUDA
src/num/flpmath.c:#include "num/gpuops.h"
src/num/flpmath.c: * including gpukrnls.h so that I can directly call cuda_zreal.
src/num/flpmath.c: * this can be removed after md_zreal is optimized for GPU.
src/num/flpmath.c:#include "num/gpukrnls.h"
src/num/flpmath.c:#ifdef USE_CUDA
src/num/flpmath.c:static void* gpu_constant(const void* vp, size_t size)
src/num/flpmath.c:	return md_gpu_move(1, (long[1]){ 1 }, vp, size);
src/num/flpmath.c:#ifdef USE_CUDA
src/num/flpmath.c:	// FIXME: something is broken with the cuda implementation of zpow -> comparision test on cpu and gpu does not fail
src/num/flpmath.c:	//assert(!(cuda_ondevice(optr) || cuda_ondevice(iptr1) || cuda_ondevice(iptr2)));
src/num/flpmath.c:#ifdef USE_CUDA
src/num/flpmath.c:	if (cuda_ondevice(iptr)) {
src/num/flpmath.c:		assert(cuda_ondevice(optr));
src/num/flpmath.c:		gpu_ops.axpy(md_calc_size(D, dims), optr, val, iptr);
src/num/flpmath.c:#ifdef USE_CUDA
src/num/flpmath.c:	if (cuda_ondevice(iptr)) {
src/num/flpmath.c:		assert(cuda_ondevice(optr));
src/num/flpmath.c:		cuda_zreal(md_calc_size(D, dims), optr, iptr);
src/num/flpmath.c:#ifdef USE_CUDA
src/num/flpmath.c:		if (cuda_ondevice(ptr[0]))
src/num/flpmath.c:			ret += gpu_ops.dot(S, ptr[0], ptr[1]);
src/num/flpmath.c:#ifdef USE_CUDA
src/num/flpmath.c:		if (cuda_ondevice(ptr[0]))
src/num/flpmath.c:			ret += gpu_ops.zdot(S, ptr[0], ptr[1]);
src/num/flpmath.c:#ifdef USE_CUDA
src/num/flpmath.c:		if (cuda_ondevice(ptr))
src/num/flpmath.c:			return gpu_ops.asum(md_calc_size(D, dims), ptr);
src/num/flpmath.c:#ifdef USE_CUDA
src/num/flpmath.c:	if (cuda_ondevice(ptr))
src/num/flpmath.c:		retp = gpu_constant(&ret, FL_SIZE);
src/num/flpmath.c:#ifdef USE_CUDA
src/num/flpmath.c:	if (cuda_ondevice(ptr)) {
src/num/flpmath.c:	// slow on GPU due to make_3op_scalar
src/num/flpmath.c:#ifdef USE_CUDA
src/num/flpmath.c:	if (cuda_ondevice(dst) != cuda_ondevice(src)) {
src/num/flpmath.c:#ifdef USE_CUDA
src/num/flpmath.c:	if (cuda_ondevice(dst) != cuda_ondevice(src)) {
src/num/flpmath.c:#ifdef USE_CUDA
src/num/flpmath.c:	if (cuda_ondevice(dst) != cuda_ondevice(src)) {
src/num/flpmath.c:#ifdef USE_CUDA
src/num/flpmath.c:	if (cuda_ondevice(dst) != cuda_ondevice(src)) {
src/num/flpmath.c:#ifdef USE_CUDA
src/num/flpmath.c:	if ((cuda_ondevice(dst) != cuda_ondevice(src_real)) || (cuda_ondevice(dst) != cuda_ondevice(src_imag))) {
src/num/gpukrnls.cu: * for operations on the GPU. See the CPU version (vecops.c) for more
src/num/gpukrnls.cu:#include <cuda_runtime_api.h>
src/num/gpukrnls.cu:#include <cuda.h>
src/num/gpukrnls.cu:#include "num/gpukrnls.h"
src/num/gpukrnls.cu:#include "num/gpuops.h"
src/num/gpukrnls.cu:#include "num/gpukrnls_misc.h"
src/num/gpukrnls.cu:// http://stackoverflow.com/questions/5810447/cuda-block-and-grid-size-efficiencies
src/num/gpukrnls.cu:extern "C" void cuda_float2double(long N, double* dst, const float* src)
src/num/gpukrnls.cu:	kern_float2double<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src);
src/num/gpukrnls.cu:extern "C" void cuda_double2float(long N, float* dst, const double* src)
src/num/gpukrnls.cu:	kern_double2float<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src);
src/num/gpukrnls.cu:extern "C" void cuda_xpay(long N, float beta, float* dst, const float* src)
src/num/gpukrnls.cu:	kern_xpay<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, beta, dst, src);
src/num/gpukrnls.cu:extern "C" void cuda_axpbz(long N, float* dst, const float a1, const float* src1, const float a2, const float* src2)
src/num/gpukrnls.cu:	kern_axpbz<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, a1, src1, a2, src2);
src/num/gpukrnls.cu:extern "C" void cuda_smul(long N, float alpha, float* dst, const float* src)
src/num/gpukrnls.cu:	kern_smul<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, alpha, dst, src);
src/num/gpukrnls.cu:typedef void (*cuda_3op_f)(long N, float* dst, const float* src1, const float* src2);
src/num/gpukrnls.cu:extern "C" void cuda_3op(cuda_3op_f krn, long N, float* dst, const float* src1, const float* src2)
src/num/gpukrnls.cu:	krn<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src1, src2);
src/num/gpukrnls.cu:extern "C" void cuda_add(long N, float* dst, const float* src1, const float* src2)
src/num/gpukrnls.cu:	cuda_3op(kern_add, N, dst, src1, src2);
src/num/gpukrnls.cu:extern "C" void cuda_sadd(long N, float val, float* dst, const float* src1)
src/num/gpukrnls.cu:	kern_sadd<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, val, dst, src1);
src/num/gpukrnls.cu:extern "C" void cuda_zsadd(long N, _Complex float val, _Complex float* dst, const _Complex float* src1)
src/num/gpukrnls.cu:	kern_zsadd<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, make_cuFloatComplex(__real(val), __imag(val)), (cuFloatComplex*)dst, (const cuFloatComplex*)src1);
src/num/gpukrnls.cu:extern "C" void cuda_sub(long N, float* dst, const float* src1, const float* src2)
src/num/gpukrnls.cu:	cuda_3op(kern_sub, N, dst, src1, src2);
src/num/gpukrnls.cu:extern "C" void cuda_mul(long N, float* dst, const float* src1, const float* src2)
src/num/gpukrnls.cu:	cuda_3op(kern_mul, N, dst, src1, src2);
src/num/gpukrnls.cu:extern "C" void cuda_div(long N, float* dst, const float* src1, const float* src2)
src/num/gpukrnls.cu:	cuda_3op(kern_div, N, dst, src1, src2);
src/num/gpukrnls.cu:extern "C" void cuda_fmac(long N, float* dst, const float* src1, const float* src2)
src/num/gpukrnls.cu:	cuda_3op(kern_fmac, N, dst, src1, src2);
src/num/gpukrnls.cu:extern "C" void cuda_fmacD(long N, double* dst, const float* src1, const float* src2)
src/num/gpukrnls.cu:	kern_fmacD<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src1, src2);
src/num/gpukrnls.cu:extern "C" void cuda_zsmul(long N, _Complex float alpha, _Complex float* dst, const _Complex float* src1)
src/num/gpukrnls.cu:	kern_zsmul<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, make_cuFloatComplex(__real(alpha), __imag(alpha)), (cuFloatComplex*)dst, (const cuFloatComplex*)src1);
src/num/gpukrnls.cu:extern "C" void cuda_zmul(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
src/num/gpukrnls.cu:	kern_zmul<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
src/num/gpukrnls.cu:extern "C" void cuda_zdiv(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
src/num/gpukrnls.cu:	kern_zdiv<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
src/num/gpukrnls.cu:extern "C" void cuda_zfmac(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
src/num/gpukrnls.cu:	kern_zfmac<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
src/num/gpukrnls.cu:extern "C" void cuda_zfmacD(long N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2)
src/num/gpukrnls.cu:	kern_zfmacD<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuDoubleComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
src/num/gpukrnls.cu:extern "C" void cuda_zmulc(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
src/num/gpukrnls.cu:	kern_zmulc<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
src/num/gpukrnls.cu:extern "C" void cuda_zfmacc(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
src/num/gpukrnls.cu:	kern_zfmacc<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
src/num/gpukrnls.cu:extern "C" void cuda_zfmaccD(long N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2)
src/num/gpukrnls.cu:	kern_zfmaccD<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuDoubleComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
src/num/gpukrnls.cu:extern "C" void cuda_zfsq2(long N, _Complex float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_zfsq2<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpukrnls.cu:extern "C" void cuda_zfmac_strided(long N, long dims[3], unsigned long oflags, unsigned long iflags1, unsigned long iflags2, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
src/num/gpukrnls.cu:	kern_zfmac_strides<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(s, N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
src/num/gpukrnls.cu:extern "C" void cuda_zfmacc_strided(long N, long dims[3], unsigned long oflags, unsigned long iflags1, unsigned long iflags2, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
src/num/gpukrnls.cu:	kern_zfmacc_strides<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(s, N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
src/num/gpukrnls.cu:extern "C" void cuda_pow(long N, float* dst, const float* src1, const float* src2)
src/num/gpukrnls.cu:	cuda_3op(kern_pow, N, dst, src1, src2);
src/num/gpukrnls.cu:extern "C" void cuda_zpow(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
src/num/gpukrnls.cu:	kern_zpow<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
src/num/gpukrnls.cu:extern "C" void cuda_sqrt(long N, float* dst, const float* src)
src/num/gpukrnls.cu:	kern_sqrt<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src);
src/num/gpukrnls.cu:extern "C" void cuda_round(long N, float* dst, const float* src)
src/num/gpukrnls.cu:	kern_round<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src);
src/num/gpukrnls.cu:extern "C" void cuda_zconj(long N, _Complex float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_zconj<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpukrnls.cu:extern "C" void cuda_zcmp(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
src/num/gpukrnls.cu:	kern_zcmp<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
src/num/gpukrnls.cu:extern "C" void cuda_zdiv_reg(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2, _Complex float lambda)
src/num/gpukrnls.cu:	kern_zdiv_reg<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2, make_cuFloatComplex(__real(lambda), __imag(lambda)));
src/num/gpukrnls.cu:extern "C" void cuda_zphsr(long N, _Complex float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_zphsr<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpukrnls.cu:extern "C" void cuda_zexp(long N, _Complex float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_zexp<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpukrnls.cu:extern "C" void cuda_zexpj(long N, _Complex float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_zexpj<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpukrnls.cu:extern "C" void cuda_zlog(long N, _Complex float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_zlog<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpukrnls.cu:extern "C" void cuda_zarg(long N, _Complex float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_zarg<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpukrnls.cu:extern "C" void cuda_zsin(long N, _Complex float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_zsin<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpukrnls.cu:extern "C" void cuda_zcos(long N, _Complex float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_zcos<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpukrnls.cu:extern "C" void cuda_zsinh(long N, _Complex float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_zsinh<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpukrnls.cu:extern "C" void cuda_zcosh(long N, _Complex float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_zcosh<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpukrnls.cu:extern "C" void cuda_zabs(long N, _Complex float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_zabs<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpukrnls.cu:extern "C" void cuda_exp(long N, float* dst, const float* src)
src/num/gpukrnls.cu:	kern_exp<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src);
src/num/gpukrnls.cu:extern "C" void cuda_log(long N, float* dst, const float* src)
src/num/gpukrnls.cu:	kern_log<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src);
src/num/gpukrnls.cu:extern "C" void cuda_zatanr(long N, _Complex float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_zatanr<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpukrnls.cu:extern "C" void cuda_zacosr(long N, _Complex float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_zacosr<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpukrnls.cu: * (GPU) Step (1) of soft thesholding, y = ST(x, lambda).
src/num/gpukrnls.cu:extern "C" void cuda_zsoftthresh_half(long N, float lambda, _Complex float* d, const _Complex float* x)
src/num/gpukrnls.cu:	kern_zsoftthresh_half<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, lambda, (cuFloatComplex*)d, (const cuFloatComplex*)x);
src/num/gpukrnls.cu:extern "C" void cuda_zsoftthresh(long N, float lambda, _Complex float* d, const _Complex float* x)
src/num/gpukrnls.cu:	kern_zsoftthresh<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, lambda, (cuFloatComplex*)d, (const cuFloatComplex*)x);
src/num/gpukrnls.cu:extern "C" void cuda_softthresh_half(long N, float lambda, float* d, const float* x)
src/num/gpukrnls.cu:	kern_softthresh_half<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, lambda, d, x);
src/num/gpukrnls.cu:extern "C" void cuda_softthresh(long N, float lambda, float* d, const float* x)
src/num/gpukrnls.cu:	kern_softthresh<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, lambda, d, x);
src/num/gpukrnls.cu:extern "C" void cuda_zreal(long N, _Complex float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_zreal<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpukrnls.cu:extern "C" void cuda_zle(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
src/num/gpukrnls.cu:	kern_zle<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
src/num/gpukrnls.cu:extern "C" void cuda_le(long N, float* dst, const float* src1, const float* src2)
src/num/gpukrnls.cu:	kern_le<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src1, src2);
src/num/gpukrnls.cu:extern "C" void cuda_zfftmod(long N, _Complex float* dst, const _Complex float* src, int n, _Bool inv, double phase)
src/num/gpukrnls.cu:	kern_zfftmod<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src, n, inv, phase);
src/num/gpukrnls.cu:extern "C" void cuda_zfftmod_3d(const long dims[3], _Complex float* dst, const _Complex float* src, _Bool inv, double phase)
src/num/gpukrnls.cu:			kern_fftmod_3d_4<<<getGridSize3(dims, (const void*)kern_fftmod_3d_4), getBlockSize3(dims, (const void*)kern_fftmod_3d), 0, cuda_get_stream()>>>(dims[0], dims[1], dims[2], (cuFloatComplex*)dst, (const cuFloatComplex*)src, inv, scale);
src/num/gpukrnls.cu:	kern_fftmod_3d<<<getGridSize3(dims, (const void*)kern_fftmod_3d), getBlockSize3(dims, (const void*)kern_fftmod_3d), 0, cuda_get_stream()>>>(dims[0], dims[1], dims[2], (cuFloatComplex*)dst, (const cuFloatComplex*)src, inv, phase);
src/num/gpukrnls.cu:extern "C" void cuda_zmax(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2)
src/num/gpukrnls.cu:	kern_zmax<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
src/num/gpukrnls.cu:extern "C" void cuda_smax(long N, float val, float* dst, const float* src1)
src/num/gpukrnls.cu:	kern_smax<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, val, dst, src1);
src/num/gpukrnls.cu:extern "C" void cuda_max(long N, float* dst, const float* src1, const float* src2)
src/num/gpukrnls.cu:	kern_max<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src1, src2);
src/num/gpukrnls.cu:extern "C" void cuda_min(long N, float* dst, const float* src1, const float* src2)
src/num/gpukrnls.cu:	kern_min<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, src1, src2);
src/num/gpukrnls.cu:extern "C" void cuda_zsmax(long N, float alpha, _Complex float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_zsmax<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, alpha, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpukrnls.cu:extern "C" void cuda_zsmin(long N, float alpha, _Complex float* dst, const _Complex float* src)
src/num/gpukrnls.cu:extern "C" void cuda_zsum(long N, _Complex float* dst)
src/num/gpukrnls.cu:		kern_reduce_zsum<<<1, B, 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst);
src/num/gpukrnls.cu:extern "C" void cuda_pdf_gauss(long N, float mu, float sig, float* dst, const float* src)
src/num/gpukrnls.cu:	kern_pdf_gauss<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, mu, sig, dst, src);
src/num/gpukrnls.cu:extern "C" void cuda_real(long N, float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_real<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, (cuFloatComplex*)src);
src/num/gpukrnls.cu:extern "C" void cuda_imag(long N, float* dst, const _Complex float* src)
src/num/gpukrnls.cu:	kern_imag<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, dst, (cuFloatComplex*)src);
src/num/gpukrnls.cu:extern "C" void cuda_zcmpl_real(long N, _Complex float* dst, const float* src)
src/num/gpukrnls.cu:	kern_zcmpl_real<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, src);
src/num/gpukrnls.cu:extern "C" void cuda_zcmpl_imag(long N, _Complex float* dst, const float* src)
src/num/gpukrnls.cu:	kern_zcmpl_imag<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, src);
src/num/gpukrnls.cu:extern "C" void cuda_zcmpl(long N, _Complex float* dst, const float* real_src, const float* imag_src)
src/num/gpukrnls.cu:	kern_zcmpl<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, real_src, imag_src);
src/num/gpukrnls.cu:extern "C" void cuda_zfill(long N, _Complex float val, _Complex float* dst)
src/num/gpukrnls.cu:	kern_zfill<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(N, make_cuFloatComplex(__real(val), __imag(val)), (cuFloatComplex*)dst);
src/num/gpukrnls.cu:extern "C" void cuda_mask_compress(long N, uint32_t* dst, const float* src)
src/num/gpukrnls.cu:	kern_mask_compress<<<gridsize(N), blocksize(N), blocksize(N) * sizeof(float), cuda_get_stream()>>>(N, dst, src);
src/num/gpukrnls.cu:extern "C" void cuda_mask_decompress(long N, float* dst, const uint32_t* src)
src/num/gpukrnls.cu:	kern_mask_decompress<<<gridsize(N), blocksize(N), blocksize(N), cuda_get_stream()>>>(N, dst, src);
src/num/gpukrnls.cu:static _Complex double cuda_reduce_zsumD(long N, const _Complex double* src)
src/num/gpukrnls.cu:	_Complex double* tmp1 = (_Complex double*)cuda_malloc(gridsize(N) * sizeof(_Complex double));
src/num/gpukrnls.cu:	_Complex double* tmp2 = (_Complex double*)cuda_malloc(gridsize(gridsize(N)) * sizeof(_Complex double));
src/num/gpukrnls.cu:	kern_reduce_zsumD<<<gridsize(N), blocksize(N), blocksize(N) * sizeof(_Complex double), cuda_get_stream()>>>(N, (cuDoubleComplex*)tmp1, (const cuDoubleComplex*)src);
src/num/gpukrnls.cu:		kern_reduce_zsumD<<<gridsize(N), blocksize(N), blocksize(N) * sizeof(_Complex double), cuda_get_stream()>>>(N, (cuDoubleComplex*)tmp2, (const cuDoubleComplex*)tmp1);
src/num/gpukrnls.cu:	cuda_memcpy(sizeof(_Complex double), &ret, tmp1);
src/num/gpukrnls.cu:	cuda_free(tmp1);
src/num/gpukrnls.cu:	cuda_free(tmp2);
src/num/gpukrnls.cu:static double cuda_reduce_sumD(long N, const double* src)
src/num/gpukrnls.cu:	double* tmp1 = (double*)cuda_malloc(gridsize(N) * sizeof(double));
src/num/gpukrnls.cu:	double* tmp2 = (double*)cuda_malloc(gridsize(gridsize(N)) * sizeof(double));
src/num/gpukrnls.cu:	kern_reduce_sumD<<<gridsize(N), blocksize(N), blocksize(N) * sizeof(double), cuda_get_stream()>>>(N, tmp1, src);
src/num/gpukrnls.cu:		kern_reduce_sumD<<<gridsize(N), blocksize(N), blocksize(N) * sizeof(double), cuda_get_stream()>>>(N, tmp2, tmp1);
src/num/gpukrnls.cu:	cuda_memcpy(sizeof(double), &ret, tmp1);
src/num/gpukrnls.cu:	cuda_free(tmp1);
src/num/gpukrnls.cu:	cuda_free(tmp2);
src/num/gpukrnls.cu:extern "C" _Complex double cuda_cdot(long N, const _Complex float* src1, const _Complex float* src2)
src/num/gpukrnls.cu:	_Complex double* tmp = (_Complex double*)cuda_malloc(gridsize(N) * sizeof(_Complex double));
src/num/gpukrnls.cu:	kern_cdot<<<gridsize(N), blocksize(N), blocksize(N) * sizeof(_Complex double), cuda_get_stream()>>>(N, (cuDoubleComplex*)tmp, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
src/num/gpukrnls.cu:	_Complex double ret = cuda_reduce_zsumD(gridsize(N), tmp);
src/num/gpukrnls.cu:	cuda_free(tmp);
src/num/gpukrnls.cu:extern "C" double cuda_dot(long N, const float* src1, const float* src2)
src/num/gpukrnls.cu:	double* tmp = (double*)cuda_malloc(gridsize(N) * sizeof(double));
src/num/gpukrnls.cu:	kern_dot<<<gridsize(N), blocksize(N), blocksize(N) * sizeof(double), cuda_get_stream()>>>(N, tmp, src1, src2);
src/num/gpukrnls.cu:	double ret = cuda_reduce_sumD(gridsize(N), tmp);
src/num/gpukrnls.cu:	cuda_free(tmp);
src/num/gpukrnls.cu:extern "C" double cuda_norm(long N, const float* src)
src/num/gpukrnls.cu:	return sqrt(cuda_dot(N, src, src));
src/num/gpurand.cu:#include <cuda_runtime_api.h>
src/num/gpurand.cu:#include <cuda.h>
src/num/gpurand.cu:#include "num/gpuops.h"
src/num/gpurand.cu:#include "num/gpukrnls_misc.h"
src/num/gpurand.cu:extern "C" void cuda_gaussian_rand(long N, _Complex float* dst,  uint64_t state, uint64_t ctr1, uint64_t offset)
src/num/gpurand.cu:	kern_gaussian_rand<<<getGridSize(N, (const void*) kern_gaussian_rand), getBlockSize(N, (const void*) kern_gaussian_rand), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, ph_state, offset);
src/num/gpurand.cu:	CUDA_KERNEL_ERROR;
src/num/gpurand.cu:extern "C" void cuda_uniform_rand(long N, _Complex float* dst,  uint64_t state, uint64_t ctr1, uint64_t offset)
src/num/gpurand.cu:	kern_uniform_rand<<<getGridSize(N, (const void*) kern_uniform_rand), getBlockSize(N, (const void*) kern_uniform_rand), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst, ph_state, offset);
src/num/gpurand.cu:	CUDA_KERNEL_ERROR;
src/num/gpurand.cu:extern "C" void cuda_rand_one(long N, _Complex float* dst, double p, uint64_t state, uint64_t ctr1, uint64_t offset)
src/num/gpurand.cu:	kern_rand_one<<<getGridSize(N, (const void*) kern_rand_one), getBlockSize(N, (const void*) kern_rand_one), 0, cuda_get_stream()>>>(N, (cuFloatComplex*)dst,  p, ph_state, offset);
src/num/gpurand.cu:	CUDA_KERNEL_ERROR;
src/num/multind.c: * All functions should work on CPU and GPU and md_copy can be used
src/num/multind.c: * to copy between CPU and GPU.
src/num/multind.c:#ifdef USE_CUDA
src/num/multind.c:#include "num/gpuops.h"
src/num/multind.c:#include "num/gpukrnls.h"
src/num/multind.c:#include "num/gpukrnls_copy.h"
src/num/multind.c:#ifdef	USE_CUDA
src/num/multind.c:	bool use_gpu = cuda_ondevice(ptr);
src/num/multind.c:#ifdef 	USE_CUDA
src/num/multind.c:		if (use_gpu) {
src/num/multind.c:			cuda_clear((long)size2, ptr[0]);
src/num/multind.c:#ifdef	USE_CUDA
src/num/multind.c:	bool use_gpu = cuda_ondevice(optr) || cuda_ondevice(iptr);
src/num/multind.c:	int ND = optimize_dims_gpu(2, D, tdims, nstr2);
src/num/multind.c:#ifdef USE_CUDA
src/num/multind.c:	if (use_gpu && (cuda_ondevice(optr) == cuda_ondevice(iptr)) && ND <= 7) {
src/num/multind.c:		cuda_copy_ND(ND, tdims, tostr, optr, tistr, iptr, size);
src/num/multind.c:	if (use_gpu && (0 != fill_flags)) {
src/num/multind.c:	if (   use_gpu
src/num/multind.c:			debug_printf(DP_DEBUG4, "CUDA 2D copy %ld %ld %ld %ld %ld %ld\n",
src/num/multind.c:			cuda_memcpy_strided(sizesp, ostr2, ptr[0], istr2, ptr[1]);
src/num/multind.c:#ifdef  USE_CUDA
src/num/multind.c:		if (use_gpu) {
src/num/multind.c:			cuda_memcpy((long)size2, ptr[0], ptr[1]);
src/num/multind.c:#ifdef USE_CUDA
src/num/multind.c:static void* gpu_constant(const void* vp, size_t size)
src/num/multind.c:        return md_gpu_move(1, (long[1]){ 1 }, vp, size);
src/num/multind.c:#ifdef USE_CUDA
src/num/multind.c:	if (cuda_ondevice(ptr) && (!cuda_ondevice(iptr))) {
src/num/multind.c:		void* giptr = gpu_constant(iptr, size);
src/num/multind.c:#ifdef  USE_CUDA
src/num/multind.c:		assert(!cuda_ondevice(ptr[0]));
src/num/multind.c:		assert(!cuda_ondevice(ptr[1]));
src/num/multind.c:#ifdef USE_CUDA
src/num/multind.c:	if (cuda_ondevice(src)) {
src/num/multind.c:		cuda_mask_compress(N, dst, src);
src/num/multind.c:#ifdef USE_CUDA
src/num/multind.c:	if (cuda_ondevice(src)) {
src/num/multind.c:		cuda_mask_decompress(N, dst, src);
src/num/multind.c:#ifdef USE_CUDA
src/num/multind.c: * Allocate GPU memory
src/num/multind.c: * return pointer to GPU memory
src/num/multind.c:void* md_alloc_gpu(int D, const long dimensions[D], size_t size)
src/num/multind.c:	return cuda_malloc(md_calc_size(D, dimensions) * (long)size);
src/num/multind.c: * Allocate GPU memory and copy from CPU pointer
src/num/multind.c: * return pointer to GPU memory
src/num/multind.c:void* md_gpu_move(int D, const long dims[D], const void* ptr, size_t size)
src/num/multind.c:	void* gpu_ptr = md_alloc_gpu(D, dims, size);
src/num/multind.c:	md_copy(D, dims, gpu_ptr, ptr, size);
src/num/multind.c:	return gpu_ptr;
src/num/multind.c: * Allocate memory on the same device (CPU/GPU) place as ptr
src/num/multind.c: * return pointer to CPU memory if ptr is in CPU or to GPU memory if ptr is in GPU
src/num/multind.c:#ifdef USE_CUDA
src/num/multind.c:	return (cuda_ondevice(ptr) ? md_alloc_gpu : md_alloc)(D, dimensions, size);
src/num/multind.c: * Free CPU/GPU memory
src/num/multind.c:#ifdef USE_CUDA
src/num/multind.c:	if (cuda_ondevice(ptr))
src/num/multind.c:		cuda_free((void*)ptr);
src/num/multiplace.c:#ifdef USE_CUDA
src/num/multiplace.c:#include "num/gpuops.h"
src/num/multiplace.c:	void* ptr_gpu;
src/num/multiplace.c:	void* mpi_gpu;
src/num/multiplace.c:	result->ptr_gpu = NULL;
src/num/multiplace.c:	result->mpi_gpu = NULL;
src/num/multiplace.c:	if (ptr->ptr_ref == ptr->ptr_gpu)
src/num/multiplace.c:		ptr->ptr_gpu = NULL;
src/num/multiplace.c:	if (ptr->ptr_ref == ptr->mpi_gpu)
src/num/multiplace.c:		ptr->mpi_gpu = NULL;
src/num/multiplace.c:	md_free(ptr->ptr_gpu);
src/num/multiplace.c:	md_free(ptr->mpi_gpu);
src/num/multiplace.c:		if (is_vptr_gpu(ref)) {
src/num/multiplace.c:			if (NULL == ptr->mpi_gpu) {
src/num/multiplace.c:				ptr->mpi_gpu = vptr_alloc_sameplace(ptr->N, ptr->dims, ptr->size, ref);
src/num/multiplace.c:				md_copy(ptr->N, ptr->dims, ptr->mpi_gpu, ptr->ptr_ref, ptr->size);
src/num/multiplace.c:			return ptr->mpi_gpu;
src/num/multiplace.c:#ifdef USE_CUDA
src/num/multiplace.c:		if (cuda_ondevice(ref)) {
src/num/multiplace.c:			if (NULL == ptr->ptr_gpu) {
src/num/multiplace.c:				ptr->ptr_gpu = md_alloc_gpu(ptr->N, ptr->dims, ptr->size);
src/num/multiplace.c:				md_copy(ptr->N, ptr->dims, ptr->ptr_gpu, ptr->ptr_ref, ptr->size);
src/num/multiplace.c:			return ptr->ptr_gpu;
src/num/multiplace.c:		if (is_vptr_gpu(tmp))
src/num/multiplace.c:			result->mpi_gpu = tmp;
src/num/multiplace.c:#ifdef USE_CUDA
src/num/multiplace.c:		if (cuda_ondevice(tmp))
src/num/multiplace.c:			result->ptr_gpu = tmp;
src/num/multiplace.c:		if (is_vptr_gpu(ptr))
src/num/multiplace.c:			result->mpi_gpu = (void*)ptr;
src/num/multiplace.c:#ifdef USE_CUDA
src/num/multiplace.c:		if (cuda_ondevice(ptr))
src/num/multiplace.c:			result->ptr_gpu = (void*)ptr;
src/num/blas.h:#ifdef USE_CUDA
src/num/blas.h:extern double cuda_asum(long size, const float* src);
src/num/blas.h:extern void cuda_saxpy(long size, float* y, float alpha, const float* src);
src/num/blas.h:extern void cuda_swap(long size, float* a, float* b);
src/num/ops.c:#ifdef USE_CUDA
src/num/ops.c:#include "num/gpuops.h"
src/num/ops.c:#ifdef USE_CUDA
src/num/ops.c:			if (allocate || cuda_ondevice(args[i]))
src/num/ops.c:			if (allocate || !cuda_ondevice(args[i]))
src/num/ops.c:				ptr[i] = md_alloc_gpu(io->N, io->dims, io->size);
src/num/ops.c:	// merge gpu wrapper with copy wrpper (stides)
src/num/ops.c:#ifdef USE_CUDA
src/num/ops.c:		if (cuda_ondevice(ref))
src/num/ops.c:const struct operator_s* operator_gpu_wrapper2(const struct operator_s* op, unsigned long move_flags)
src/num/ops.c:	return operator_copy_wrapper_generic(N, strs, loc, op, -1 /*select gpu by thread*/, false);
src/num/ops.c:const struct operator_s* operator_gpu_wrapper(const struct operator_s* op)
src/num/ops.c:	return operator_gpu_wrapper2(op, ~0UL);
src/num/ops.c:// FIXME: we should reimplement link in terms of dup and bind (caveat: gpu; io_flags)
src/num/ops.c:#ifdef USE_CUDA
src/num/ops.c:	// Allocate tmp on GPU when one argument is on the GPU.
src/num/ops.c:	bool gpu = false;
src/num/ops.c:		gpu = gpu || cuda_ondevice(args[i]);
src/num/ops.c:	void* tmp = (gpu ? md_alloc_gpu : md_alloc)(iov->N, iov->dims, iov->size);
src/num/gpukrnls_misc.cu:#include <cuda_runtime_api.h>
src/num/gpukrnls_misc.cu:#include "gpukrnls_misc.h"
src/num/gpukrnls_misc.cu:	return getBlockSize3(dims, cuda_get_max_threads(func));
src/num/gpukrnls_misc.cu:	return getGridSize3(dims, cuda_get_max_threads(func));
src/num/gpukrnls_misc.cu:	return getBlockSize3(dims, cuda_get_max_threads(func));
src/num/gpukrnls_misc.cu:	return getGridSize3(dims, cuda_get_max_threads(func));
src/num/gpukrnls_misc.cu:int cuda_get_max_threads(const void* func)
src/num/gpukrnls_misc.cu:	struct cudaFuncAttributes attr;
src/num/gpukrnls_misc.cu:	cudaFuncGetAttributes(&attr, func);
src/num/gpuops.h:#ifdef USE_CUDA
src/num/gpuops.h:#include <cuda_runtime_api.h>
src/num/gpuops.h:extern void cuda_error(const char* file, int line, cudaError_t code);
src/num/gpuops.h:extern void cuda_gpu_check(const char* file, int line, const char* note);
src/num/gpuops.h:extern void cuda_check_ptr(const char* file, int line, int N, const void* ptr[__VLA(N)]);
src/num/gpuops.h:#define CUDA_ASYNC_ERROR_NOTE(x)	({ cuda_gpu_check(__FILE__, __LINE__, (x)); })
src/num/gpuops.h:#define CUDA_ASYNC_ERROR		CUDA_ASYNC_ERROR_NOTE("")
src/num/gpuops.h:#define CUDA_ERROR(x)			({ cudaError_t errval = (x); if (cudaSuccess != errval) cuda_error(__FILE__, __LINE__, errval); })
src/num/gpuops.h:#define CUDA_KERNEL_ERROR 		({ cudaError_t errval = cudaGetLastError(); if (cudaSuccess != errval) cuda_error(__FILE__, __LINE__, errval); CUDA_ASYNC_ERROR; })
src/num/gpuops.h:#define CUDA_ERROR_PTR(...)		({ CUDA_ASYNC_ERROR; const void* _ptr[] = { __VA_ARGS__}; cuda_check_ptr(__FILE__, __LINE__, (sizeof(_ptr) / sizeof(_ptr[0])), _ptr); })
src/num/gpuops.h:#define CUDA_MAX_STREAMS 8
src/num/gpuops.h:extern int cuda_num_streams;
src/num/gpuops.h:extern const struct vec_ops gpu_ops;
src/num/gpuops.h:extern void cuda_init(void);
src/num/gpuops.h:extern void cuda_exit(void);
src/num/gpuops.h:extern int cuda_get_device_id(void);
src/num/gpuops.h:extern int cuda_get_stream_id(void);
src/num/gpuops.h:#ifdef USE_CUDA
src/num/gpuops.h:extern cudaStream_t cuda_get_stream_by_id(int id);
src/num/gpuops.h:extern cudaStream_t cuda_get_stream(void);
src/num/gpuops.h:extern int cuda_set_stream_level(void);
src/num/gpuops.h:extern void cuda_sync_device(void);
src/num/gpuops.h:extern void cuda_sync_stream(void);
src/num/gpuops.h:extern void* cuda_malloc(long N);
src/num/gpuops.h:extern void cuda_free(void*);
src/num/gpuops.h:extern _Bool cuda_ondevice(const void* ptr);
src/num/gpuops.h:extern void cuda_clear(long size, void* ptr);
src/num/gpuops.h:extern void cuda_memcpy(long size, void* dst, const void* src);
src/num/gpuops.h:extern void cuda_memcpy_strided(const long dims[2], long ostr, void* dst, long istr, const void* src);
src/num/gpuops.h:extern void cuda_memcache_off(void);
src/num/gpuops.h:extern void cuda_memcache_clear(void);
src/num/gpuops.h:extern void cuda_use_global_memory(void);
src/num/gpuops.h:extern void print_cuda_meminfo(void);
src/num/gpurand.h:#ifndef GPURAND_H
src/num/gpurand.h:#define GPURAND_H
src/num/gpurand.h:extern void cuda_gaussian_rand(long N, _Complex float* dst, uint64_t state, uint64_t ctr1, uint64_t offset);
src/num/gpurand.h:extern void cuda_uniform_rand(long N, _Complex float* dst, uint64_t state, uint64_t ctr1, uint64_t offset);
src/num/gpurand.h:extern void cuda_rand_one(long N, _Complex float* dst, double p, uint64_t state, uint64_t ctr1, uint64_t offset);
src/num/gpurand.h:#endif // GPURAND_H
src/num/gpukrnls_bat.cu:#include <cuda_runtime_api.h>
src/num/gpukrnls_bat.cu:#include <cuda.h>
src/num/gpukrnls_bat.cu:#include "num/gpuops.h"
src/num/gpukrnls_bat.cu:#include "num/gpukrnls.h"
src/num/gpukrnls_bat.cu:	cudaFuncAttributes attr;
src/num/gpukrnls_bat.cu:	cudaFuncGetAttributes(&attr, func);
src/num/gpukrnls_bat.cu:	cudaFuncAttributes attr;
src/num/gpukrnls_bat.cu:	cudaFuncGetAttributes(&attr, func);
src/num/gpukrnls_bat.cu:extern "C" void cuda_xpay_bat(long Bi, long N, long Bo, const float* beta, float* a, const float* x)
src/num/gpukrnls_bat.cu:extern "C" void cuda_axpy_bat(long Bi, long N, long Bo, float* a, const float* alpha, const float* x)
src/num/gpukrnls_bat.cu:extern "C" void cuda_dot_bat(long Bi, long N, long Bo, float* dst, const float* x, const float* y)
src/num/gpu_conv.h:extern void cuda_im2col(_Complex float* dst, const _Complex float* src, const long odims[5], const long idims[5], const long kdims[5], const long dilation[5], const long strides[5]);
src/num/gpu_conv.h:extern void cuda_im2col_transp(_Complex float* dst, const _Complex float* src, const long odims[5], const long idims[5], const long kdims[5], const long dilation[5], const long strides[5]);
src/num/multind.h:#ifdef USE_CUDA
src/num/multind.h:extern void* md_alloc_gpu(int D, const long dimensions[__VLA(D)], size_t size);
src/num/multind.h:extern void* md_gpu_move(int D, const long dims[__VLA(D)], const void* ptr, size_t size);
src/num/md_wrapper.c:#ifdef USE_CUDA
src/num/md_wrapper.c:#include "num/gpukrnls.h"
src/num/md_wrapper.c:#include "num/gpukrnls_unfold.h"
src/num/md_wrapper.c:#include "num/gpuops.h"
src/num/md_wrapper.c:void zfmac_gpu_batched_loop(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
src/num/md_wrapper.c:#ifdef USE_CUDA
src/num/md_wrapper.c:	assert(cuda_ondevice(optr));
src/num/md_wrapper.c:	assert(cuda_ondevice(iptr1));
src/num/md_wrapper.c:	assert(cuda_ondevice(iptr2));
src/num/md_wrapper.c:	cuda_zfmac_strided(dims[0], tdims,
src/num/md_wrapper.c:void zfmac_gpu_unfold(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
src/num/md_wrapper.c:#ifdef USE_CUDA
src/num/md_wrapper.c:	cuda_zfmac_unfold(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
src/num/md_wrapper.c:void zfmacc_gpu_batched_loop(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
src/num/md_wrapper.c:#ifdef USE_CUDA
src/num/md_wrapper.c:	assert(cuda_ondevice(optr));
src/num/md_wrapper.c:	assert(cuda_ondevice(iptr1));
src/num/md_wrapper.c:	assert(cuda_ondevice(iptr2));
src/num/md_wrapper.c:	cuda_zfmacc_strided(dims[0], tdims,
src/num/md_wrapper.c:void zfmacc_gpu_unfold(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
src/num/md_wrapper.c:#ifdef USE_CUDA
src/num/md_wrapper.c:	cuda_zfmacc_unfold(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
src/num/md_wrapper.c:void fmac_gpu_unfold(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
src/num/md_wrapper.c:#ifdef USE_CUDA
src/num/md_wrapper.c:	cuda_fmac_unfold(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
src/num/md_wrapper.c:void add_gpu_unfold(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
src/num/md_wrapper.c:#ifdef USE_CUDA
src/num/md_wrapper.c:	cuda_add_unfold(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
src/num/md_wrapper.c:void zadd_gpu_unfold(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
src/num/md_wrapper.c:#ifdef USE_CUDA
src/num/md_wrapper.c:	cuda_zadd_unfold(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
src/num/md_wrapper.c:void mul_gpu_unfold(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
src/num/md_wrapper.c:#ifdef USE_CUDA
src/num/md_wrapper.c:	cuda_mul_unfold(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
src/num/md_wrapper.c:void zmul_gpu_unfold(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
src/num/md_wrapper.c:#ifdef USE_CUDA
src/num/md_wrapper.c:	cuda_zmul_unfold(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
src/num/md_wrapper.c:void zmulc_gpu_unfold(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
src/num/md_wrapper.c:#ifdef USE_CUDA
src/num/md_wrapper.c:	cuda_zmulc_unfold(N, dims, ostr, optr, istr1, iptr1, istr2, iptr2);
src/num/gpukrnls_copy.cu:#include <cuda_runtime_api.h>
src/num/gpukrnls_copy.cu:#include <cuda.h>
src/num/gpukrnls_copy.cu:#include "num/gpuops.h"
src/num/gpukrnls_copy.cu:#include "num/gpukrnls_misc.h"
src/num/gpukrnls_copy.cu:#include "gpukrnls_copy.h"
src/num/gpukrnls_copy.cu:struct cuda_strides_ND {
src/num/gpukrnls_copy.cu:typedef void(kern_copy_t)(cuda_strides_ND strs, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2);
src/num/gpukrnls_copy.cu:__global__ static void kern_copy_strides(cuda_strides_ND strs, T* dst, const T* src)
src/num/gpukrnls_copy.cu:		mt[N - 1] = cuda_get_max_threads(get_kern_fop_unfold<T>(N));
src/num/gpukrnls_copy.cu:static void cuda_copy_template_unfold(int D, const long dims[], const long ostrs[], void* dst, const long istrs[], const void* src)
src/num/gpukrnls_copy.cu:	cuda_strides_ND strs;
src/num/gpukrnls_copy.cu:	CUDA_ERROR_PTR(dst, src);
src/num/gpukrnls_copy.cu:		kern_copy_strides<T, 1><<<ex1, ex2, 0, cuda_get_stream()>>>(strs, (T*)dst, (const T*)src);
src/num/gpukrnls_copy.cu:		kern_copy_strides<T, 2><<<ex1, ex2, 0, cuda_get_stream()>>>(strs, (T*)dst, (const T*)src);
src/num/gpukrnls_copy.cu:		kern_copy_strides<T, 3><<<ex1, ex2, 0, cuda_get_stream()>>>(strs, (T*)dst, (const T*)src);
src/num/gpukrnls_copy.cu:		kern_copy_strides<T, 4><<<ex1, ex2, 0, cuda_get_stream()>>>(strs, (T*)dst, (const T*)src);
src/num/gpukrnls_copy.cu:		kern_copy_strides<T, 5><<<ex1, ex2, 0, cuda_get_stream()>>>(strs, (T*)dst, (const T*)src);
src/num/gpukrnls_copy.cu:		kern_copy_strides<T, 6><<<ex1, ex2, 0, cuda_get_stream()>>>(strs, (T*)dst, (const T*)src);
src/num/gpukrnls_copy.cu:		kern_copy_strides<T, 7><<<ex1, ex2, 0, cuda_get_stream()>>>(strs, (T*)dst, (const T*)src);
src/num/gpukrnls_copy.cu:		kern_copy_strides<T, 8><<<ex1, ex2, 0, cuda_get_stream()>>>(strs, (T*)dst, (const T*)src);
src/num/gpukrnls_copy.cu:	CUDA_KERNEL_ERROR;
src/num/gpukrnls_copy.cu:void cuda_copy_ND(int D, const long dims[], const long ostrs[], void* dst, const long istrs[], const void* src, size_t size)
src/num/gpukrnls_copy.cu:		cuda_copy_template_unfold<uint8_t>(D + 1, ndims, nostrs, dst, nistrs, src);
src/num/gpukrnls_copy.cu:		cuda_copy_template_unfold<uint16_t>(D + 1, ndims, nostrs, dst, nistrs, src);
src/num/gpukrnls_copy.cu:		cuda_copy_template_unfold<uint32_t>(D + 1, ndims, nostrs, dst, nistrs, src);
src/num/gpukrnls_copy.cu:		cuda_copy_template_unfold<uint64_t>(D + 1, ndims, nostrs, dst, nistrs, src);
src/num/ops_p.h:extern const struct operator_p_s* operator_p_gpu_wrapper(const struct operator_p_s* op);
src/num/cudnn_wrapper.c:#ifdef USE_CUDA
src/num/cudnn_wrapper.c:#include "num/gpuops.h"
src/num/cudnn_wrapper.c:#define CUDNN_ERROR(x)	({ CUDA_ASYNC_ERROR_NOTE("before cuDNN call"); cudnnStatus_t errval = (x); if (CUDNN_STATUS_SUCCESS  != errval) cudnn_error(__FILE__, __LINE__, errval); CUDA_ASYNC_ERROR_NOTE("after cuDNN call"); })
src/num/cudnn_wrapper.c:#define CUDNN_CALL(x)	({ CUDA_ASYNC_ERROR_NOTE("before cuDNN call"); cudnn_set_gpulock(); cudnnStatus_t errval = (x); if (CUDNN_STATUS_SUCCESS  != errval) cudnn_error(__FILE__, __LINE__, errval); cudnn_unset_gpulock(); CUDA_ASYNC_ERROR_NOTE("after cuBLAS call"); })
src/num/cudnn_wrapper.c:static cudnnHandle_t handle[CUDA_MAX_STREAMS + 1];
src/num/cudnn_wrapper.c:static omp_lock_t cudnn_gpulock[CUDA_MAX_STREAMS + 1];
src/num/cudnn_wrapper.c:static void cudnn_set_gpulock(void)
src/num/cudnn_wrapper.c:	omp_set_lock(&(cudnn_gpulock[cuda_get_stream_id()]));
src/num/cudnn_wrapper.c:static void cudnn_unset_gpulock(void)
src/num/cudnn_wrapper.c:	omp_unset_lock(&(cudnn_gpulock[cuda_get_stream_id()]));
src/num/cudnn_wrapper.c:static void cudnn_set_gpulock(void)
src/num/cudnn_wrapper.c:static void cudnn_unset_gpulock(void)
src/num/cudnn_wrapper.c:	for (int i = 0; i < CUDA_MAX_STREAMS + 1; i++) {
src/num/cudnn_wrapper.c:		omp_init_lock(&(cudnn_gpulock[i]));
src/num/cudnn_wrapper.c:	for (int i = 0; i < CUDA_MAX_STREAMS + 1; i++) {
src/num/cudnn_wrapper.c:		omp_destroy_lock(&(cudnn_gpulock[i]));
src/num/cudnn_wrapper.c:	return handle[cuda_get_stream_id()];
src/num/cudnn_wrapper.c:	void* workspace = (0 < ws_size) ? md_alloc_gpu(1, MD_DIMS(1), ws_size) : NULL;
src/num/cudnn_wrapper.c:	cudnn_set_gpulock();
src/num/cudnn_wrapper.c:	cudnn_unset_gpulock();
src/num/cudnn_wrapper.c:	void* workspace = (0 < ws_size) ? md_alloc_gpu(1, MD_DIMS(1), ws_size) : NULL;
src/num/cudnn_wrapper.c:	cudnn_set_gpulock();
src/num/cudnn_wrapper.c:	cudnn_unset_gpulock();
src/num/cudnn_wrapper.c:	void* workspace = (0 < ws_size) ? md_alloc_gpu(1, MD_DIMS(1), ws_size) : NULL;
src/num/cudnn_wrapper.c:	cudnn_set_gpulock();
src/num/cudnn_wrapper.c:	cudnn_unset_gpulock();
src/num/cudnn_wrapper.c:	decomp_bart = decomp_bart && (1 == optimize_dims_gpu(1, nbDims, dims, tstrs));
src/num/cudnn_wrapper.c:		float* real_tmp = md_alloc_gpu(1, dims, FL_SIZE);
src/num/cudnn_wrapper.c:		float* imag_tmp = md_alloc_gpu(1, dims, FL_SIZE);
src/num/cudnn_wrapper.c:	decomp_bart = decomp_bart && (1 == optimize_dims_gpu(1, nbDims, dims, tstrs));
src/num/cudnn_wrapper.c:		float* real_tmp = md_alloc_gpu(1, dims, FL_SIZE);
src/num/cudnn_wrapper.c:		float* imag_tmp = md_alloc_gpu(1, dims, FL_SIZE);
src/num/cudnn_wrapper.c:	float* krn_real = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:	float* krn_imag = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:		float* in_real2 = md_alloc_gpu(bcd.N, bcd.idims, FL_SIZE);
src/num/cudnn_wrapper.c:		float* in_imag2 = md_alloc_gpu(bcd.N, bcd.idims, FL_SIZE);
src/num/cudnn_wrapper.c:		float* out_real2 = md_alloc_gpu(bcd.N, bcd.odims, FL_SIZE);
src/num/cudnn_wrapper.c:		float* out_imag2 = md_alloc_gpu(bcd.N, bcd.odims, FL_SIZE);
src/num/cudnn_wrapper.c:	float* krn_real = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:	float* krn_imag = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:		float* out_real2 = md_alloc_gpu(1, MD_DIMS(1), out_desc.size_transformed);
src/num/cudnn_wrapper.c:		float* out_imag2 = md_alloc_gpu(1, MD_DIMS(1), out_desc.size_transformed);
src/num/cudnn_wrapper.c:		float* in_real2 = md_alloc_gpu(1, MD_DIMS(1), in_desc.size_transformed);
src/num/cudnn_wrapper.c:		float* in_imag2 = md_alloc_gpu(1, MD_DIMS(1), in_desc.size_transformed);
src/num/cudnn_wrapper.c:	float* krn_real = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:	float* krn_imag = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:		float* out_real2 = md_alloc_gpu(1, MD_DIMS(1), out_desc.size_transformed);
src/num/cudnn_wrapper.c:		float* out_imag2 = md_alloc_gpu(1, MD_DIMS(1), out_desc.size_transformed);
src/num/cudnn_wrapper.c:		float* in_real2 = md_alloc_gpu(1, MD_DIMS(1), in_desc.size_transformed);
src/num/cudnn_wrapper.c:		float* in_imag2 = md_alloc_gpu(1, MD_DIMS(1), in_desc.size_transformed);
src/num/cudnn_wrapper.c:		float* tmp = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:		float* in_tmp = md_alloc_gpu(bcd.N, bcd.idims, FL_SIZE);
src/num/cudnn_wrapper.c:		float* out_tmp = md_alloc_gpu(bcd.N, bcd.odims, FL_SIZE);
src/num/cudnn_wrapper.c:	float* krn_tmp = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:		float* out_tmp = md_alloc_gpu(1, MD_DIMS(1), out_desc.size_transformed);
src/num/cudnn_wrapper.c:		float* in_tmp = md_alloc_gpu(1, MD_DIMS(1), in_desc.size_transformed);
src/num/cudnn_wrapper.c:	float* krn_tmp = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:		float* out_tmp = md_alloc_gpu(1, MD_DIMS(1), out_desc.size_transformed);
src/num/cudnn_wrapper.c:		float* in_tmp = md_alloc_gpu(1, MD_DIMS(1), in_desc.size_transformed);
src/num/cudnn_wrapper.c:	float* krn_real = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:	float* krn_imag = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:	float* nkrn = md_alloc_gpu(bcd.N, rbcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:	float* krn_real = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:	float* krn_imag = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:	float* nkrn = md_alloc_gpu(bcd.N, rbcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:	float* krn_real = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:	float* krn_imag = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:	float* nkrn = md_alloc_gpu(bcd.N, rbcd.kdims, FL_SIZE);
src/num/cudnn_wrapper.c:	float* krn_tmp = md_alloc_gpu(bcd.N, bcd.kdims, FL_SIZE);
src/num/wavelet.c:#ifdef USE_CUDA
src/num/wavelet.c:#include "num/gpuops.h"
src/num/wavelet.c:#include "num/wlcuda.h"
src/num/wavelet.c:#ifdef USE_CUDA
src/num/wavelet.c:	if (cuda_ondevice(ptr))
src/num/wavelet.c:		cuda_cdf97(n, str / 4, ptr);
src/num/wavelet.c:#ifdef USE_CUDA
src/num/wavelet.c:	if (cuda_ondevice(ptr))
src/num/wavelet.c:		cuda_icdf97(n, str / 4, ptr);	
src/num/gpu_reduce.cu:#include <cuda_runtime_api.h>
src/num/gpu_reduce.cu:#include <cuda.h>
src/num/gpu_reduce.cu:#include "num/gpu_reduce.h"
src/num/gpu_reduce.cu:#include "num/gpuops.h"
src/num/gpu_reduce.cu:extern "C" void cuda_reduce_zadd_outer(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src)
src/num/gpu_reduce.cu:	long maxBlockSizeX_gpu = 32;
src/num/gpu_reduce.cu:	unsigned int blockSizeX = MIN(maxBlockSizeX_gpu, maxBlockSizeX_dim);
src/num/gpu_reduce.cu:	kern_reduce_zadd_outer<<<gridDim, blockDim, blockSizeX * blockSizeY * CFL_SIZE, cuda_get_stream()>>>(dim_reduce, dim_batch, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpu_reduce.cu:extern "C" void cuda_reduce_zadd_inner(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src)
src/num/gpu_reduce.cu:	long maxBlockSizeX_gpu = 32;
src/num/gpu_reduce.cu:	unsigned int blockSizeX = MIN(maxBlockSizeX_gpu, maxBlockSizeX_dim);
src/num/gpu_reduce.cu:	kern_reduce_zadd_inner<<<gridDim, blockDim, blockSizeX * blockSizeY * CFL_SIZE, cuda_get_stream()>>>(dim_reduce, dim_batch, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpu_reduce.cu:extern "C" void cuda_reduce_add_outer(long dim_reduce, long dim_batch, float* dst, const float* src)
src/num/gpu_reduce.cu:	long maxBlockSizeX_gpu = 32;
src/num/gpu_reduce.cu:	unsigned int blockSizeX = MIN(maxBlockSizeX_gpu, maxBlockSizeX_dim);
src/num/gpu_reduce.cu:	kern_reduce_add_outer<<<gridDim, blockDim, blockSizeX * blockSizeY * FL_SIZE, cuda_get_stream()>>>(dim_reduce, dim_batch, dst, src);
src/num/gpu_reduce.cu:extern "C" void cuda_reduce_add_inner(long dim_reduce, long dim_batch, float* dst, const float* src)
src/num/gpu_reduce.cu:	long maxBlockSizeX_gpu = 32;
src/num/gpu_reduce.cu:	unsigned int blockSizeX = MIN(maxBlockSizeX_gpu, maxBlockSizeX_dim);
src/num/gpu_reduce.cu:	kern_reduce_add_inner<<<gridDim, blockDim, blockSizeX * blockSizeY * FL_SIZE, cuda_get_stream()>>>(dim_reduce, dim_batch, dst, src);
src/num/gpu_reduce.cu:extern "C" void cuda_reduce_zmax_outer(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src)
src/num/gpu_reduce.cu:	long maxBlockSizeX_gpu = 32;
src/num/gpu_reduce.cu:	unsigned int blockSizeX = MIN(maxBlockSizeX_gpu, maxBlockSizeX_dim);
src/num/gpu_reduce.cu:	kern_reduce_zmax_outer<<<gridDim, blockDim, blockSizeX * blockSizeY * CFL_SIZE, cuda_get_stream()>>>(dim_reduce, dim_batch, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/gpu_reduce.cu:extern "C" void cuda_reduce_zmax_inner(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src)
src/num/gpu_reduce.cu:	long maxBlockSizeX_gpu = 32;
src/num/gpu_reduce.cu:	unsigned int blockSizeX = MIN(maxBlockSizeX_gpu, maxBlockSizeX_dim);
src/num/gpu_reduce.cu:	kern_reduce_zmax_inner<<<gridDim, blockDim, blockSizeX * blockSizeY * CFL_SIZE, cuda_get_stream()>>>(dim_reduce, dim_batch, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/num/reduce_md_wrapper.h:extern void reduce_zadd_inner_gpu(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
src/num/reduce_md_wrapper.h:extern void reduce_zadd_outer_gpu(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
src/num/reduce_md_wrapper.h:extern void reduce_add_inner_gpu(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
src/num/reduce_md_wrapper.h:extern void reduce_add_outer_gpu(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], float* optr, const long istr1[__VLA(N)], const float* iptr1, const long istr2[__VLA(N)], const float* iptr2);
src/num/reduce_md_wrapper.h:extern void reduce_zmax_inner_gpu(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
src/num/reduce_md_wrapper.h:extern void reduce_zmax_outer_gpu(int N, const long dims[__VLA(N)], const long ostr[__VLA(N)], _Complex float* optr, const long istr1[__VLA(N)], const _Complex float* iptr1, const long istr2[__VLA(N)], const _Complex float* iptr2);
src/num/gpu_conv.cu:#include <cuda_runtime_api.h>
src/num/gpu_conv.cu:#include <cuda.h>
src/num/gpu_conv.cu:#include "num/gpuops.h"
src/num/gpu_conv.cu:#include "num/gpu_conv.h"
src/num/gpu_conv.cu:#include "num/gpukrnls_misc.h"
src/num/gpu_conv.cu:static void cuda_im2col_int(_Complex float* dst, const _Complex float* src, const long odims[5], const long idims[5], const long kdims[5], const long dilation[5], const long strides[5])
src/num/gpu_conv.cu:		kern_im2col_valid_no_dil_str<DIMS, T, transp><<<getGridSize(config.N_in_elements, func), getBlockSize(config.N_in_elements, func), 0, cuda_get_stream() >>>(config, (cuFloatComplex*) dst, (cuFloatComplex*) src);
src/num/gpu_conv.cu:		kern_im2col_valid<DIMS, T, transp><<<getGridSize(config.N_in_elements, func), getBlockSize(config.N_in_elements, func), 0, cuda_get_stream() >>>(config, (cuFloatComplex*) dst, (cuFloatComplex*) src);
src/num/gpu_conv.cu:static void cuda_im2col_int2(_Complex float* dst, const _Complex float* src, const long odims[5], const long idims[5], const long kdims[5], const long dilation[5], const long strides[5])
src/num/gpu_conv.cu:			cuda_im2col_int<1, uint32_t, transp>(dst, src, odims, idims, kdims, dilation, strides);
src/num/gpu_conv.cu:			cuda_im2col_int<1, uint64_t, transp>(dst, src, odims, idims, kdims, dilation, strides);
src/num/gpu_conv.cu:			cuda_im2col_int<2, uint32_t, transp>(dst, src, odims, idims, kdims, dilation, strides);
src/num/gpu_conv.cu:			cuda_im2col_int<2, uint64_t, transp>(dst, src, odims, idims, kdims, dilation, strides);
src/num/gpu_conv.cu:			cuda_im2col_int<3, uint32_t, transp>(dst, src, odims, idims, kdims, dilation, strides);
src/num/gpu_conv.cu:			cuda_im2col_int<3, uint64_t, transp>(dst, src, odims, idims, kdims, dilation, strides);
src/num/gpu_conv.cu:extern "C" void cuda_im2col(_Complex float* dst, const _Complex float* src, const long odims[5], const long idims[5], const long kdims[5], const long dilation[5], const long strides[5])
src/num/gpu_conv.cu:	cuda_im2col_int2<false>(dst, src, odims, idims, kdims, dilation, strides);
src/num/gpu_conv.cu: * Transposed/adjoint of cuda im2col
src/num/gpu_conv.cu:extern "C" void cuda_im2col_transp(_Complex float* dst, const _Complex float* src, const long odims[5], const long idims[5], const long kdims[5], const long dilation[5], const long strides[5])
src/num/gpu_conv.cu:	cuda_im2col_int2<true>(dst, src, odims, idims, kdims, dilation, strides);
src/num/gpukrnls_bat.h:extern void cuda_xpay_bat(long Bi, long N, long Bo, const float* beta, float* a, const float* x);
src/num/gpukrnls_bat.h:extern void cuda_axpy_bat(long Bi, long N, long Bo, float* a, const float* alpha, const float* x);
src/num/gpukrnls_bat.h:extern void cuda_dot_bat(long Bi, long N, long Bo, float* dst, const float* x, const float* y);
src/num/ops_p.c:	assert(3 == N);	// FIXME: gpu
src/num/ops_p.c:const struct operator_p_s* operator_p_gpu_wrapper(const struct operator_p_s* op)
src/num/ops_p.c:	return operator_p_downcast(operator_gpu_wrapper2(operator_p_upcast(op), MD_BIT(1) | MD_BIT(2)));
src/num/fft-cuda.c:#include "fft-cuda.h"
src/num/fft-cuda.c:#ifdef USE_CUDA
src/num/fft-cuda.c:#include "num/gpuops.h"
src/num/fft-cuda.c:struct fft_cuda_plan_s {
src/num/fft-cuda.c:	struct fft_cuda_plan_s* chain;
src/num/fft-cuda.c:#define CUFFT_ERROR(x)	({ CUDA_ASYNC_ERROR_NOTE("before cuFFT call"); enum cufftResult_t errval = (x); if (CUFFT_SUCCESS != errval) cufft_error(__FILE__, __LINE__, errval); CUDA_ASYNC_ERROR_NOTE("after cuFFT call");})
src/num/fft-cuda.c:static struct fft_cuda_plan_s* fft_cuda_plan0(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], const long istrides[D], bool backwards)
src/num/fft-cuda.c:	PTR_ALLOC(struct fft_cuda_plan_s, plan);
src/num/fft-cuda.c:	CUFFT_ERROR(cufftSetStream(plan->cufft, cuda_get_stream()));
src/num/fft-cuda.c:struct fft_cuda_plan_s* fft_cuda_plan(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], const long istrides[D], bool backwards)
src/num/fft-cuda.c:	struct fft_cuda_plan_s* plan = fft_cuda_plan0(D, dimensions, flags, ostrides, istrides, backwards);
src/num/fft-cuda.c:		struct fft_cuda_plan_s* plan = fft_cuda_plan0(D, dimensions, msb, ostrides, istrides, backwards);
src/num/fft-cuda.c:		plan->chain = fft_cuda_plan(D, dimensions, flags & ~msb, ostrides, ostrides, backwards);
src/num/fft-cuda.c:			fft_cuda_free_plan(plan);
src/num/fft-cuda.c:void fft_cuda_free_plan(struct fft_cuda_plan_s* cuplan)
src/num/fft-cuda.c:		fft_cuda_free_plan(cuplan->chain);
src/num/fft-cuda.c:void fft_cuda_exec(struct fft_cuda_plan_s* cuplan, complex float* dst, const complex float* src)
src/num/fft-cuda.c:	assert(cuda_ondevice(src));
src/num/fft-cuda.c:	assert(cuda_ondevice(dst));
src/num/fft-cuda.c:	void* workspace = md_alloc_gpu(1, MAKE_ARRAY(1l), workspace_size);
src/num/fft-cuda.c:	CUDA_ERROR_PTR(dst, src, workspace);
src/num/fft-cuda.c:		fft_cuda_exec(cuplan->chain, dst, dst);
src/num/fft-cuda.c:#endif // USE_CUDA
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:#include "num/gpuops.h"
src/num/convcorr.c:#include "num/gpukrnls.h"
src/num/convcorr.c:#include "num/gpu_conv.h"
src/num/convcorr.c://#define CONVCORR_OPTIMIZE_GPU_ONLY
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:zconvcorr_bwd_krn_algo_f* algos_bwd_krn_gpu[] = {
src/num/convcorr.c:							zconvcorr_bwd_krn_im2col_cf_gpu,
src/num/convcorr.c:zconvcorr_fwd_algo_f* algos_fwd_gpu[] = {
src/num/convcorr.c:						zconvcorr_fwd_im2col_cf_gpu,
src/num/convcorr.c:zconvcorr_bwd_in_algo_f* algos_bwd_in_gpu[] = {
src/num/convcorr.c:						zconvcorr_bwd_in_im2col_cf_gpu,
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:	if (cuda_ondevice(out))
src/num/convcorr.c:		for (int i = 0; (unsigned long)i < sizeof(algos_fwd_gpu) / sizeof(algos_fwd_gpu[0]); i++)
src/num/convcorr.c:			if (algos_fwd_gpu[i](	N,
src/num/convcorr.c:	if (!cuda_ondevice(out))
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:	if (cuda_ondevice(out))
src/num/convcorr.c:		for(int i = 0; (unsigned long)i < sizeof(algos_bwd_in_gpu) / sizeof(algos_bwd_in_gpu[0]); i++)
src/num/convcorr.c:			if (algos_bwd_in_gpu[i](	N,
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:	if (!cuda_ondevice(out))
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:	if (cuda_ondevice(out))
src/num/convcorr.c:		for(int i = 0; (unsigned long)i < sizeof(algos_bwd_krn_gpu) / sizeof(algos_bwd_krn_gpu[0]); i++)
src/num/convcorr.c:			if (algos_bwd_krn_gpu[i](	N,
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:	if (!cuda_ondevice(out))
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:	if (cuda_ondevice(out))
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:	if (cuda_ondevice(out))
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:	if (cuda_ondevice(out))
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:bool zconvcorr_fwd_im2col_cf_gpu(int N,
src/num/convcorr.c:	if (!cuda_ondevice(out))
src/num/convcorr.c:			complex float* imat_tmp = md_alloc_gpu(1, &imat_size, CFL_SIZE);
src/num/convcorr.c:			cuda_im2col(imat_tmp, (const complex float*)ptr[1] + i * isize, odimsp, idimsp, kdimsp, dilationp, stridesp);
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:bool zconvcorr_bwd_krn_im2col_cf_gpu(int N,
src/num/convcorr.c:	if (!cuda_ondevice(out))
src/num/convcorr.c:			complex float* imat_tmp = md_alloc_gpu(1, &imat_size, size);
src/num/convcorr.c:			cuda_im2col(imat_tmp, (const complex float*)ptr[1] + i * isize, odimsp, idimsp, kdimsp, dilationp, stridesp);
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:bool zconvcorr_bwd_in_im2col_cf_gpu(int N,
src/num/convcorr.c:	if (!cuda_ondevice(out))
src/num/convcorr.c:			complex float* imat_tmp = md_alloc_gpu(1, &imat_size, CFL_SIZE);
src/num/convcorr.c:			cuda_im2col_transp((complex float*)ptr[0] + i * isize, imat_tmp , odimsp, idimsp, kdimsp, dilationp, stridesp);
src/num/convcorr.c:				float max_nrmse, bool gpu, long min_no_algos)
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:	void* ref_ptr = gpu ? md_alloc_gpu(1, MD_DIMS(1), CFL_SIZE) : md_alloc(1, MD_DIMS(1), CFL_SIZE);
src/num/convcorr.c:	assert(!gpu);
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:	int nr_algos = gpu ? ARRAY_SIZE(algos_fwd_gpu) : ARRAY_SIZE(algos_fwd_cpu);
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:		zconvcorr_fwd_algo_f* algo = gpu ? algos_fwd_gpu[i] : algos_fwd_cpu[i];
src/num/convcorr.c:				float max_nrmse, bool gpu, long min_no_algos)
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:	void* ref_ptr = gpu ? md_alloc_gpu(1, MD_DIMS(1), CFL_SIZE) : md_alloc(1, MD_DIMS(1), CFL_SIZE);
src/num/convcorr.c:	assert(!gpu);
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:	int nr_algos = gpu ? ARRAY_SIZE(algos_bwd_in_gpu) : ARRAY_SIZE(algos_bwd_in_cpu);
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:		zconvcorr_bwd_in_algo_f* algo = gpu ? algos_bwd_in_gpu[i] : algos_bwd_in_cpu[i];
src/num/convcorr.c:				float max_nrmse, bool gpu, long min_no_algos)
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:	void* ref_ptr = gpu ? md_alloc_gpu(1, MD_DIMS(1), CFL_SIZE) : md_alloc(1, MD_DIMS(1), CFL_SIZE);
src/num/convcorr.c:	assert(!gpu);
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:	int nr_algos = gpu ? ARRAY_SIZE(algos_bwd_krn_gpu) : ARRAY_SIZE(algos_bwd_krn_cpu);
src/num/convcorr.c:#ifdef USE_CUDA
src/num/convcorr.c:		zconvcorr_bwd_krn_algo_f* algo = gpu ? algos_bwd_krn_gpu[i] : algos_bwd_krn_cpu[i];
src/num/init.c:#ifdef USE_CUDA
src/num/init.c:#include "num/gpuops.h"
src/num/init.c:static bool bart_gpu_support = false;
src/num/init.c:bool bart_use_gpu = false;
src/num/init.c:#ifdef USE_CUDA
src/num/init.c:	const char* gpu_str;
src/num/init.c:	if (NULL != (gpu_str = getenv("BART_GPU"))) {
src/num/init.c:		int bart_num_gpus = strtol(gpu_str, NULL, 10);
src/num/init.c:		if (0 < bart_num_gpus)
src/num/init.c:			bart_use_gpu = true;
src/num/init.c:	if (NULL != (gpu_str = getenv("BART_GPU_STREAMS"))) {
src/num/init.c:		int bart_num_streams = strtol(gpu_str, NULL, 10);
src/num/init.c:			cuda_num_streams = bart_num_streams;
src/num/init.c:	if (NULL != (mem_str = getenv("BART_GPU_GLOBAL_MEMORY"))) {
src/num/init.c:			error("BART_GPU_GLOBAL_MEMORY environment variable must be 0 or 1!\n");
src/num/init.c:			cuda_use_global_memory();
src/num/init.c:#ifdef USE_CUDA
src/num/init.c:	if (bart_gpu_support && bart_use_gpu)
src/num/init.c:			cuda_init();
src/num/init.c:	if (bart_use_gpu)
src/num/init.c:		error("BART compiled without GPU support.\n");
src/num/init.c:void num_init_gpu_support(void)
src/num/init.c:	bart_gpu_support = true;
src/num/init.c:void num_deinit_gpu(void)
src/num/init.c:#ifdef USE_CUDA
src/num/init.c:	cuda_exit();
src/num/init.c:	error("BART compiled without GPU support.\n");
src/num/philox.inc:// for CUDA
src/num/blas_md_wrapper.c:#ifdef USE_CUDA
src/num/blas_md_wrapper.c:#include "num/gpuops.h"
src/num/blas_md_wrapper.c:#ifdef USE_CUDA
src/num/blas_md_wrapper.c:	if (cuda_ondevice(optr)) {
src/num/blas_md_wrapper.c:#ifdef USE_CUDA
src/num/blas_md_wrapper.c:	if (cuda_ondevice(optr)) {
src/num/blas_md_wrapper.c:#ifdef USE_CUDA
src/num/blas_md_wrapper.c:	if (cuda_ondevice(optr)) {
src/num/blas_md_wrapper.c:#ifdef USE_CUDA
src/num/blas_md_wrapper.c:	if (cuda_ondevice(optr)) {
src/num/reduce_md_wrapper.c:#ifdef USE_CUDA
src/num/reduce_md_wrapper.c:#include "num/gpuops.h"
src/num/reduce_md_wrapper.c:#include "num/gpu_reduce.h"
src/num/reduce_md_wrapper.c:void reduce_zadd_inner_gpu(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
src/num/reduce_md_wrapper.c:#ifdef USE_CUDA
src/num/reduce_md_wrapper.c:	assert(cuda_ondevice(optr) && cuda_ondevice(iptr2));
src/num/reduce_md_wrapper.c:	cuda_reduce_zadd_inner(dims[0], (2 == N) ? dims[1] : 1, optr, iptr2);
src/num/reduce_md_wrapper.c:	error("Compiled without gpu support!\n");
src/num/reduce_md_wrapper.c:void reduce_zadd_outer_gpu(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
src/num/reduce_md_wrapper.c:#ifdef USE_CUDA
src/num/reduce_md_wrapper.c:	assert(cuda_ondevice(optr) && cuda_ondevice(iptr2));
src/num/reduce_md_wrapper.c:	cuda_reduce_zadd_outer(dims[1], dims[0], optr, iptr2);
src/num/reduce_md_wrapper.c:	error("Compiled without gpu support!\n");
src/num/reduce_md_wrapper.c:void reduce_add_inner_gpu(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
src/num/reduce_md_wrapper.c:#ifdef USE_CUDA
src/num/reduce_md_wrapper.c:	assert(cuda_ondevice(optr) && cuda_ondevice(iptr2));
src/num/reduce_md_wrapper.c:	cuda_reduce_add_inner(dims[0], (2 == N) ? dims[1] : 1, optr, iptr2);
src/num/reduce_md_wrapper.c:	error("Compiled without gpu support!\n");
src/num/reduce_md_wrapper.c:void reduce_add_outer_gpu(int N, const long dims[N], const long ostr[N], float* optr, const long istr1[N], const float* iptr1, const long istr2[N], const float* iptr2)
src/num/reduce_md_wrapper.c:#ifdef USE_CUDA
src/num/reduce_md_wrapper.c:	assert(cuda_ondevice(optr) && cuda_ondevice(iptr2));
src/num/reduce_md_wrapper.c:	cuda_reduce_add_outer(dims[1], dims[0], optr, iptr2);
src/num/reduce_md_wrapper.c:	error("Compiled without gpu support!\n");
src/num/reduce_md_wrapper.c:void reduce_zmax_inner_gpu(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
src/num/reduce_md_wrapper.c:#ifdef USE_CUDA
src/num/reduce_md_wrapper.c:	assert(cuda_ondevice(optr) && cuda_ondevice(iptr2));
src/num/reduce_md_wrapper.c:	cuda_reduce_zmax_inner(dims[0], (2 == N) ? dims[1] : 1, optr, iptr2);
src/num/reduce_md_wrapper.c:	error("Compiled without gpu support!\n");
src/num/reduce_md_wrapper.c:void reduce_zmax_outer_gpu(int N, const long dims[N], const long ostr[N], complex float* optr, const long istr1[N], const complex float* iptr1, const long istr2[N], const complex float* iptr2)
src/num/reduce_md_wrapper.c:#ifdef USE_CUDA
src/num/reduce_md_wrapper.c:	assert(cuda_ondevice(optr) && cuda_ondevice(iptr2));
src/num/reduce_md_wrapper.c:	cuda_reduce_zmax_outer(dims[1], dims[0], optr, iptr2);
src/num/reduce_md_wrapper.c:	error("Compiled without gpu support!\n");
src/num/optimize.c:#ifdef USE_CUDA
src/num/optimize.c:#include "num/gpuops.h"
src/num/optimize.c:int optimize_dims_gpu(int D, int N, long dims[N], long (*strs[D])[N])
src/num/optimize.c:#ifdef USE_CUDA
src/num/optimize.c:static bool use_gpu(int p, void* ptr[p])
src/num/optimize.c:	bool gpu = false;
src/num/optimize.c:		gpu = gpu || cuda_ondevice(ptr[i]);
src/num/optimize.c:		gpu = gpu && cuda_ondevice(ptr[i]);
src/num/optimize.c:	if (!gpu) {
src/num/optimize.c:			assert(!cuda_ondevice(ptr[i]));
src/num/optimize.c:	return gpu;
src/num/optimize.c:static bool one_on_gpu(int p, void* ptr[p])
src/num/optimize.c:	bool gpu = false;
src/num/optimize.c:	for (int i = 0; !gpu && (i < p); i++)
src/num/optimize.c:		gpu |= cuda_ondevice(ptr[i]);
src/num/optimize.c:	return gpu;
src/num/optimize.c:#ifdef USE_CUDA
src/num/optimize.c:	bool gpu = use_gpu(N, nptr1);
src/num/optimize.c:	int ND = (gpu ? optimize_dims_gpu : optimize_dims)(N, D, tdims, nstr1);
src/num/optimize.c:#ifdef USE_CUDA
src/num/optimize.c:	if (gpu)	// not implemented yet
src/num/optimize.c:#ifdef USE_CUDA
src/num/optimize.c:	if (num_auto_parallelize && !gpu && !one_on_gpu(N, nptr1)) {
src/num/optimize.c:#ifdef USE_CUDA
src/num/optimize.c:	debug_printf(DP_DEBUG4, "This is a %s call\n.", gpu ? "gpu" : "cpu");
src/num/optimize.c:	__block struct nary_opt_data_s data = { md_calc_size(skip, tdims), gpu ? &gpu_ops : &cpu_ops };
src/num/gpukrnls_unfold.h:extern void cuda_add_unfold(int D, const long dims[], const long ostrs[], float* dst, const long istrs1[], const float* src1, const long istrs2[], const float* src2);
src/num/gpukrnls_unfold.h:extern void cuda_zadd_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2);
src/num/gpukrnls_unfold.h:extern void cuda_mul_unfold(int D, const long dims[], const long ostrs[], float* dst, const long istrs1[], const float* src1, const long istrs2[], const float* src2);
src/num/gpukrnls_unfold.h:extern void cuda_zmul_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2);
src/num/gpukrnls_unfold.h:extern void cuda_zmulc_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2);
src/num/gpukrnls_unfold.h:extern void cuda_fmac_unfold(int D, const long dims[], const long ostrs[], float* dst, const long istrs1[], const float* src1, const long istrs2[], const float* src2);
src/num/gpukrnls_unfold.h:extern void cuda_zfmac_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2);
src/num/gpukrnls_unfold.h:extern void cuda_zfmacc_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2);
src/num/vptr.h:extern _Bool is_vptr_gpu(const void* ptr);
src/num/vptr.h:extern void* vptr_move_gpu(const void* ptr);
src/num/init.h:extern _Bool bart_use_gpu;
src/num/init.h:extern void num_init_gpu_support(void);
src/num/init.h:extern void num_deinit_gpu(void);
src/num/gpukrnls_unfold.cu:#include <cuda_runtime_api.h>
src/num/gpukrnls_unfold.cu:#include <cuda.h>
src/num/gpukrnls_unfold.cu:#include "num/gpuops.h"
src/num/gpukrnls_unfold.cu:#include "num/gpukrnls.h"
src/num/gpukrnls_unfold.cu:struct cuda_strides_3D {
src/num/gpukrnls_unfold.cu:static struct cuda_strides_3D strs_ini = {
src/num/gpukrnls_unfold.cu:typedef void(kern_fOp_unfold)(cuda_strides_3D strs, float* dst, const float* src1, const float* src2);
src/num/gpukrnls_unfold.cu:typedef void(kern_zOp_unfold)(cuda_strides_3D strs, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2);
src/num/gpukrnls_unfold.cu:__global__ static void kern_fop_unfold_generic(cuda_strides_3D strs, float* dst, const float* src1, const float* src2)
src/num/gpukrnls_unfold.cu:__global__ static void kern_zop_unfold_generic(cuda_strides_3D strs, cuFloatComplex* dst, const cuFloatComplex* src1, const cuFloatComplex* src2)
src/num/gpukrnls_unfold.cu:static void cuda_fop_unfold(int D, const long dims[], const long ostrs[], float* dst, const long istrs1[], const float* src1, const long istrs2[], const float* src2)
src/num/gpukrnls_unfold.cu:	cuda_strides_3D strs = strs_ini;
src/num/gpukrnls_unfold.cu:	CUDA_ERROR_PTR(dst, src1, src2);
src/num/gpukrnls_unfold.cu:		cudaFuncAttributes attr;
src/num/gpukrnls_unfold.cu:		cudaFuncGetAttributes(&attr, func);
src/num/gpukrnls_unfold.cu:	func<<<gridDim, blockDim, sizeof(float) * (size1 + size2), cuda_get_stream()>>>(strs, dst, src1, src2);
src/num/gpukrnls_unfold.cu:	CUDA_KERNEL_ERROR;
src/num/gpukrnls_unfold.cu:static void cuda_zop_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2)
src/num/gpukrnls_unfold.cu:	cuda_strides_3D strs = strs_ini;
src/num/gpukrnls_unfold.cu:	CUDA_ERROR_PTR(dst, src1, src2);
src/num/gpukrnls_unfold.cu:		cudaFuncAttributes attr;
src/num/gpukrnls_unfold.cu:		cudaFuncGetAttributes(&attr, func);
src/num/gpukrnls_unfold.cu:	func<<<gridDim, blockDim, sizeof(cuFloatComplex) * (size1 + size2), cuda_get_stream()>>>(strs, (cuFloatComplex*)dst, (const cuFloatComplex*)src1, (const cuFloatComplex*)src2);
src/num/gpukrnls_unfold.cu:	CUDA_KERNEL_ERROR;
src/num/gpukrnls_unfold.cu:__device__ __forceinline__ static void cuda_device_add(float* dst, float x, float y)
src/num/gpukrnls_unfold.cu:extern "C" void cuda_add_unfold(int D, const long dims[], const long ostrs[], float* dst, const long istrs1[], const float* src1, const long istrs2[], const float* src2)
src/num/gpukrnls_unfold.cu:	cuda_fop_unfold<cuda_device_add>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
src/num/gpukrnls_unfold.cu:__device__ __forceinline__ static void cuda_device_zadd(cuFloatComplex* dst, cuFloatComplex x, cuFloatComplex y)
src/num/gpukrnls_unfold.cu:extern "C" void cuda_zadd_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2)
src/num/gpukrnls_unfold.cu:	cuda_zop_unfold<cuda_device_zadd>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
src/num/gpukrnls_unfold.cu:__device__ __forceinline__ static void cuda_device_mul(float* dst, float x, float y)
src/num/gpukrnls_unfold.cu:extern "C" void cuda_mul_unfold(int D, const long dims[], const long ostrs[], float* dst, const long istrs1[], const float* src1, const long istrs2[], const float* src2)
src/num/gpukrnls_unfold.cu:	cuda_fop_unfold<cuda_device_mul>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
src/num/gpukrnls_unfold.cu:__device__ __forceinline__ static void cuda_device_zmul(cuFloatComplex* dst, cuFloatComplex x, cuFloatComplex y)
src/num/gpukrnls_unfold.cu:extern "C" void cuda_zmul_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2)
src/num/gpukrnls_unfold.cu:	cuda_zop_unfold<cuda_device_zmul>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
src/num/gpukrnls_unfold.cu:__device__ __forceinline__ static void cuda_device_zmulc(cuFloatComplex* dst, cuFloatComplex x, cuFloatComplex y)
src/num/gpukrnls_unfold.cu:extern "C" void cuda_zmulc_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2)
src/num/gpukrnls_unfold.cu:	cuda_zop_unfold<cuda_device_zmulc>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
src/num/gpukrnls_unfold.cu:__device__ __forceinline__ static void cuda_device_fmac(float* dst, float x, float y)
src/num/gpukrnls_unfold.cu:extern "C" void cuda_fmac_unfold(int D, const long dims[], const long ostrs[], float* dst, const long istrs1[], const float* src1, const long istrs2[], const float* src2)
src/num/gpukrnls_unfold.cu:	cuda_fop_unfold<cuda_device_fmac>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
src/num/gpukrnls_unfold.cu:__device__ __forceinline__ static void cuda_device_zfmac(cuFloatComplex* dst, cuFloatComplex x, cuFloatComplex y)
src/num/gpukrnls_unfold.cu:extern "C" void cuda_zfmac_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2)
src/num/gpukrnls_unfold.cu:	cuda_zop_unfold<cuda_device_zfmac>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
src/num/gpukrnls_unfold.cu:__device__ __forceinline__ static void cuda_device_zfmacc(cuFloatComplex* dst, cuFloatComplex x, cuFloatComplex y)
src/num/gpukrnls_unfold.cu:extern "C" void cuda_zfmacc_unfold(int D, const long dims[], const long ostrs[], _Complex float* dst, const long istrs1[], const _Complex float* src1, const long istrs2[], const _Complex float* src2)
src/num/gpukrnls_unfold.cu:	cuda_zop_unfold<cuda_device_zfmacc>(D, dims, ostrs, dst, istrs1, src1, istrs2, src2);
src/num/rand.c:#ifdef USE_CUDA
src/num/rand.c:#include "num/gpuops.h"
src/num/rand.c:#include "num/gpurand.h"
src/num/rand.c:#ifdef  USE_CUDA
src/num/rand.c:	if (cuda_ondevice(dst)) {
src/num/rand.c:#ifdef  USE_CUDA
src/num/rand.c:	if (cuda_ondevice(dst)) {
src/num/rand.c:		cuda_gaussian_rand(N, dst, state.state, state.ctr1, (uint64_t)offset);
src/num/rand.c:#ifdef  USE_CUDA
src/num/rand.c:	if (cuda_ondevice(dst)) {
src/num/rand.c:#ifdef  USE_CUDA
src/num/rand.c:	if (cuda_ondevice(dst)) {
src/num/rand.c:		cuda_uniform_rand(N, dst, state.state, state.ctr1, (uint64_t)offset);
src/num/rand.c:#ifdef  USE_CUDA
src/num/rand.c:	if (cuda_ondevice(dst)) {
src/num/rand.c:#ifdef  USE_CUDA
src/num/rand.c:	if (cuda_ondevice(dst)) {
src/num/rand.c:		cuda_rand_one(N, dst, p, state.state, state.ctr1, (uint64_t) offset);
src/num/mpi_ops.c:#ifdef USE_CUDA
src/num/mpi_ops.c:#include "num/gpuops.h"
src/num/mpi_ops.c:bool cuda_aware_mpi = false;
src/num/mpi_ops.c:#ifdef MPIX_CUDA_AWARE_SUPPORT
src/num/mpi_ops.c:		if (1 == MPIX_Query_cuda_support())
src/num/mpi_ops.c:			cuda_aware_mpi = true;
src/num/mpi_ops.c:#ifdef USE_CUDA
src/num/mpi_ops.c:static void print_cuda_aware_warning(void)
src/num/mpi_ops.c:	if (!printed && !cuda_aware_mpi)
src/num/mpi_ops.c:		debug_printf(DP_WARN, "CUDA aware MPI is not activated. This may decrease performance for multi-GPU operations significantly!.\n");
src/num/mpi_ops.c:#ifdef USE_CUDA
src/num/mpi_ops.c:static void mpi_bcast_selected_gpu(bool tag, void* ptr, long size, int root)
src/num/mpi_ops.c:	print_cuda_aware_warning();
src/num/mpi_ops.c:		cuda_memcpy(size, tmp, ptr);
src/num/mpi_ops.c:		cuda_memcpy(size, ptr, tmp);
src/num/mpi_ops.c:#ifdef USE_CUDA
src/num/mpi_ops.c:	if (!cuda_aware_mpi && cuda_ondevice(ptr)) {
src/num/mpi_ops.c:		mpi_bcast_selected_gpu(tag, ptr, size, root);
src/num/mpi_ops.c:	if (cuda_ondevice(ptr))
src/num/mpi_ops.c:		cuda_sync_stream();
src/num/mpi_ops.c:#ifdef USE_CUDA
src/num/mpi_ops.c:			if (cuda_ondevice(dst) || cuda_ondevice(src))
src/num/mpi_ops.c:				cuda_memcpy(size, dst, src);
src/num/mpi_ops.c:#ifdef USE_CUDA
src/num/mpi_ops.c:		if (cuda_ondevice(src) && !cuda_aware_mpi) {
src/num/mpi_ops.c:			print_cuda_aware_warning();
src/num/mpi_ops.c:			cuda_memcpy(size, src2, src);
src/num/mpi_ops.c:		if (cuda_ondevice(src2))
src/num/mpi_ops.c:			cuda_sync_stream();
src/num/mpi_ops.c:#ifdef USE_CUDA
src/num/mpi_ops.c:		if (cuda_ondevice(src) && !cuda_aware_mpi)
src/num/mpi_ops.c:#ifdef USE_CUDA
src/num/mpi_ops.c:		if (cuda_ondevice(dst) && !cuda_aware_mpi) {
src/num/mpi_ops.c:			print_cuda_aware_warning();
src/num/mpi_ops.c:#ifdef USE_CUDA
src/num/mpi_ops.c:		if (cuda_ondevice(dst) && !cuda_aware_mpi) {
src/num/mpi_ops.c:			cuda_memcpy(size, dst, dst2);
src/num/mpi_ops.c:#ifdef USE_CUDA
src/num/mpi_ops.c:static void mpi_reduce_land_gpu(long N, bool vec[N])
src/num/mpi_ops.c:	print_cuda_aware_warning();
src/num/mpi_ops.c:	cuda_memcpy(size, tmp, vec);
src/num/mpi_ops.c:	cuda_memcpy(size, vec, tmp);
src/num/mpi_ops.c:#ifdef USE_CUDA
src/num/mpi_ops.c:	if (!cuda_aware_mpi && cuda_ondevice(vec)) {
src/num/mpi_ops.c:		mpi_reduce_land_gpu(N, vec);
src/num/mpi_ops.c:	if (cuda_ondevice(vec))
src/num/mpi_ops.c:		cuda_sync_stream();
src/num/mpi_ops.c:static void mpi_allreduce_sum_gpu(int N, float vec[N], MPI_Comm comm)
src/num/mpi_ops.c:#ifdef USE_CUDA
src/num/mpi_ops.c:	if (!cuda_aware_mpi && cuda_ondevice(vec)) {
src/num/mpi_ops.c:		print_cuda_aware_warning();
src/num/mpi_ops.c:		cuda_memcpy(size, tmp, vec);
src/num/mpi_ops.c:		cuda_memcpy(size, vec, tmp);
src/num/mpi_ops.c:	if (cuda_ondevice(vec))
src/num/mpi_ops.c:		cuda_sync_stream();
src/num/mpi_ops.c:			mpi_allreduce_sum_gpu(MIN(N - n, INT_MAX / 2), vec + n, comm_sub);
src/num/mpi_ops.c:		mpi_allreduce_sum_gpu(MIN(N - n, INT_MAX / 2), vec + n, mpi_get_comm());
src/num/mpi_ops.c:static void mpi_allreduce_sumD_gpu(int N, double vec[N], MPI_Comm comm)
src/num/mpi_ops.c:#ifdef USE_CUDA
src/num/mpi_ops.c:	if (!cuda_aware_mpi && cuda_ondevice(vec)) {
src/num/mpi_ops.c:		print_cuda_aware_warning();
src/num/mpi_ops.c:		cuda_memcpy(size, tmp, vec);
src/num/mpi_ops.c:		cuda_memcpy(size, vec, tmp);
src/num/mpi_ops.c:		cuda_sync_stream();
src/num/mpi_ops.c:	if (cuda_ondevice(vec))
src/num/mpi_ops.c:		cuda_sync_stream();
src/num/mpi_ops.c:			mpi_allreduce_sumD_gpu(MIN(N - n, INT_MAX / 2), vec2 + n, comm_sub);
src/num/gpukrnls.h:extern double cuda_dot(long N, const float* src1, const float* src2);
src/num/gpukrnls.h:extern double cuda_norm(long N, const float* src);
src/num/gpukrnls.h:extern _Complex double cuda_cdot(long N, const _Complex float* src1, const _Complex float* src2);
src/num/gpukrnls.h:extern void cuda_float2double(long size, double* dst, const float* src);
src/num/gpukrnls.h:extern void cuda_double2float(long size, float* dst, const double* src);
src/num/gpukrnls.h:extern void cuda_sxpay(long size, float* y, float alpha, const float* src);
src/num/gpukrnls.h:extern void cuda_xpay(long N, float beta, float* dst, const float* src);
src/num/gpukrnls.h:extern void cuda_axpbz(long N, float* dst, const float a, const float* x, const float b, const float* z);
src/num/gpukrnls.h:extern void cuda_smul(long N, float alpha, float* dst, const float* src);
src/num/gpukrnls.h:extern void cuda_mul(long N, float* dst, const float* src1, const float* src2);
src/num/gpukrnls.h:extern void cuda_div(long N, float* dst, const float* src1, const float* src2);
src/num/gpukrnls.h:extern void cuda_add(long N, float* dst, const float* src1, const float* src2);
src/num/gpukrnls.h:extern void cuda_sadd(long N, float val, float* dst, const float* src1);
src/num/gpukrnls.h:extern void cuda_zsadd(long N, _Complex float val, _Complex float* dst, const _Complex float* src1);
src/num/gpukrnls.h:extern void cuda_sub(long N, float* dst, const float* src1, const float* src2);
src/num/gpukrnls.h:extern void cuda_fmac(long N, float* dst, const float* src1, const float* src2);
src/num/gpukrnls.h:extern void cuda_fmacD(long N, double* dst, const float* src1, const float* src2);
src/num/gpukrnls.h:extern void cuda_zsmul(long N, _Complex float alpha, _Complex float* dst, const _Complex float* src1);
src/num/gpukrnls.h:extern void cuda_zmul(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
src/num/gpukrnls.h:extern void cuda_zdiv(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
src/num/gpukrnls.h:extern void cuda_zfmac(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
src/num/gpukrnls.h:extern void cuda_zfmacD(long N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2);
src/num/gpukrnls.h:extern void cuda_zmulc(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
src/num/gpukrnls.h:extern void cuda_zfmacc(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
src/num/gpukrnls.h:extern void cuda_zfmaccD(long N, _Complex double* dst, const _Complex float* src1, const _Complex float* src2);
src/num/gpukrnls.h:extern void cuda_zfsq2(long N, _Complex float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_pow(long N, float* dst, const float* src1, const float* src2);
src/num/gpukrnls.h:extern void cuda_zpow(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
src/num/gpukrnls.h:extern void cuda_sqrt(long N, float* dst, const float* src);
src/num/gpukrnls.h:extern void cuda_round(long N, float* dst, const float* src);
src/num/gpukrnls.h:extern void cuda_zconj(long N, _Complex float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_zphsr(long N, _Complex float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_zexpj(long N, _Complex float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_zexp(long N, _Complex float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_zsin(long N, _Complex float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_zcos(long N, _Complex float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_zsinh(long N, _Complex float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_zcosh(long N, _Complex float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_zlog(long N, _Complex float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_zarg(long N, _Complex float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_zabs(long N, _Complex float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_zatanr(long N, _Complex float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_zacosr(long N, _Complex float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_exp(long N, float* dst, const float* src);
src/num/gpukrnls.h:extern void cuda_log(long N, float* dst, const float* src);
src/num/gpukrnls.h:extern void cuda_zsoftthresh_half(long N, float lambda, _Complex float* d, const _Complex float* x);
src/num/gpukrnls.h:extern void cuda_zsoftthresh(long N, float lambda, _Complex float* d, const _Complex float* x);
src/num/gpukrnls.h:extern void cuda_softthresh_half(long N, float lambda, float* d, const float* x);
src/num/gpukrnls.h:extern void cuda_softthresh(long N, float lambda, float* d, const float* x);
src/num/gpukrnls.h:extern void cuda_zreal(long N, _Complex float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_zcmp(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
src/num/gpukrnls.h:extern void cuda_zdiv_reg(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2, _Complex float lambda);
src/num/gpukrnls.h:extern void cuda_le(long N, float* dst, const float* src1, const float* src2);
src/num/gpukrnls.h:extern void cuda_zfftmod(long N, _Complex float* dst, const _Complex float* src, int n, _Bool inv, double phase);
src/num/gpukrnls.h:extern void cuda_zfftmod_3d(const long dims[3], _Complex float* dst, const _Complex float* src, _Bool inv, double phase);
src/num/gpukrnls.h:extern void cuda_zmax(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
src/num/gpukrnls.h:extern void cuda_zle(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
src/num/gpukrnls.h:extern void cuda_smax(long N, float val, float* dst, const float* src1);
src/num/gpukrnls.h:extern void cuda_max(long N, float* dst, const float* src1, const float* src2);
src/num/gpukrnls.h:extern void cuda_min(long N, float* dst, const float* src1, const float* src2);
src/num/gpukrnls.h:extern void cuda_zsum(long N, _Complex float* dst);
src/num/gpukrnls.h:extern void cuda_zsmax(long N, float alpha, _Complex float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_zsmin(long N, float alpha, _Complex float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_pdf_gauss(long N, float mu, float sig, float* dst, const float* src);
src/num/gpukrnls.h:extern void cuda_real(long N, float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_imag(long N, float* dst, const _Complex float* src);
src/num/gpukrnls.h:extern void cuda_zcmpl_real(long N, _Complex float* dst, const float* src);
src/num/gpukrnls.h:extern void cuda_zcmpl_imag(long N, _Complex float* dst, const float* src);
src/num/gpukrnls.h:extern void cuda_zcmpl(long N, _Complex float* dst, const float* real_src, const float* imag_src);
src/num/gpukrnls.h:extern void cuda_zfill(long N, _Complex float val, _Complex float* dst);
src/num/gpukrnls.h:extern void cuda_mask_compress(long N, uint32_t* dst, const float* src);
src/num/gpukrnls.h:extern void cuda_mask_decompress(long N, float* dst, const uint32_t* src);
src/num/gpukrnls.h:extern void cuda_zfmac_strided(long N, long dims[3], unsigned long oflags, unsigned long iflags1, unsigned long iflags2, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
src/num/gpukrnls.h:extern void cuda_zfmacc_strided(long N, long dims[3], unsigned long oflags, unsigned long iflags1, unsigned long iflags2, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
src/num/gpu_reduce.h:extern void cuda_reduce_zadd_inner(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src);
src/num/gpu_reduce.h:extern void cuda_reduce_zadd_outer(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src);
src/num/gpu_reduce.h:extern void cuda_reduce_zmax_inner(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src);
src/num/gpu_reduce.h:extern void cuda_reduce_zmax_outer(long dim_reduce, long dim_batch, _Complex float* dst, const _Complex float* src);
src/num/gpu_reduce.h:extern void cuda_reduce_add_inner(long dim_reduce, long dim_batch, float* dst, const float* src);
src/num/gpu_reduce.h:extern void cuda_reduce_add_outer(long dim_reduce, long dim_batch, float* dst, const float* src);
src/num/shuffle.c:#ifdef USE_CUDA
src/num/shuffle.c:#include "num/gpuops.h"
src/num/shuffle.c:#ifdef USE_CUDA
src/num/shuffle.c:	if (cuda_ondevice(out) && (1 < n_factors)) {
src/num/shuffle.c:		//=> much less calls to cuda copy strided
src/num/shuffle.c:		//FIXME: we should have generic strided copy kernel on GPU! 
src/num/shuffle.c:#ifdef USE_CUDA
src/num/shuffle.c:	if (cuda_ondevice(out) && (1 < n_factors)) {
src/num/shuffle.c:		//=> much less calls to cuda copy strided
src/num/shuffle.c:		//FIXME: we should have generic strided copy kernel on GPU!
src/num/vecops.c: * are implemented for the GPU in gpukrnls.cu.
src/num/vecops.c: * If you add functions here, please also add to gpuops.c/gpukrnls.cu
src/num/gpuops.c: * CUDA support functions. The file exports gpu_ops of type struct vec_ops
src/num/gpuops.c: * in gpukrnls.cu. See vecops.c for the CPU version.
src/num/gpuops.c:#ifdef USE_CUDA
src/num/gpuops.c:#include <cuda_runtime_api.h>
src/num/gpuops.c:#include <cuda.h>
src/num/gpuops.c:#include "num/gpukrnls.h"
src/num/gpuops.c:#include "num/gpukrnls_bat.h"
src/num/gpuops.c:#include "gpuops.h"
src/num/gpuops.c:static int cuda_stream_level = -1;
src/num/gpuops.c:cudaStream_t cuda_streams[CUDA_MAX_STREAMS + 1];
src/num/gpuops.c:static int cuda_device_id = -1;
src/num/gpuops.c:static _Thread_local int cuda_device_id_thread = -1;
src/num/gpuops.c:bool cuda_memcache = true;
src/num/gpuops.c:bool cuda_global_memory = false;
src/num/gpuops.c:int cuda_num_streams = 1;
src/num/gpuops.c:void cuda_error(const char* file, int line, cudaError_t code)
src/num/gpuops.c:	const char *err_str = cudaGetErrorString(code);
src/num/gpuops.c:	error("CUDA Error: %s in %s:%d\n", err_str, file, line);
src/num/gpuops.c:void cuda_gpu_check(const char* file, int line, const char* note)
src/num/gpuops.c:#ifdef GPU_ASSERTS
src/num/gpuops.c:	cudaError_t code = cudaStreamSynchronize(cuda_get_stream());
src/num/gpuops.c:	if (cudaSuccess != code) {
src/num/gpuops.c:		const char *err_str = cudaGetErrorString(code);
src/num/gpuops.c:			error("CUDA Error: %s in %s:%d\n", err_str, file, line);
src/num/gpuops.c:			error("CUDA Error: %s in %s:%d (%s)\n", err_str, file, line, note);
src/num/gpuops.c:void cuda_check_ptr(const char* file, int line, int N, const void* ptr[N])
src/num/gpuops.c:#ifdef GPU_ASSERTS
src/num/gpuops.c:		if (!cuda_ondevice(ptr[i]))
src/num/gpuops.c:			error("CUDA Error: Pointer not on device in %s:%d", file, line);
src/num/gpuops.c:// Print free and used memory on GPU.
src/num/gpuops.c:void print_cuda_meminfo(void)
src/num/gpuops.c:	CUDA_ERROR(cudaMemGetInfo(&byte_free, &byte_tot));
src/num/gpuops.c:	debug_printf(DP_INFO , "GPU memory usage: used = %.4f MiB, free = %.4f MiB, total = %.4f MiB\n", dbyte_used/MiBYTE, dbyte_free/MiBYTE, dbyte_tot/MiBYTE);
src/num/gpuops.c://*************************************** CUDA Device Initialization *********************************************
src/num/gpuops.c:static bool cuda_try_init(int device)
src/num/gpuops.c:	if (-1 != cuda_device_id)
src/num/gpuops.c:	cudaError_t errval = cudaSetDevice(device);
src/num/gpuops.c:	if (cudaSuccess == errval)
src/num/gpuops.c:		errval = cudaDeviceSynchronize();
src/num/gpuops.c:	if (cudaSuccess == errval) {
src/num/gpuops.c:		cuda_device_id = device;
src/num/gpuops.c:		cuda_streams[CUDA_MAX_STREAMS] = cudaStreamLegacy;
src/num/gpuops.c:		for (int i = 0; i < CUDA_MAX_STREAMS; i++)
src/num/gpuops.c:			CUDA_ERROR(cudaStreamCreate(&(cuda_streams[i])));
src/num/gpuops.c:		if (cudaErrorDevicesUnavailable != errval) {
src/num/gpuops.c:			const char *err_str = cudaGetErrorString(errval);
src/num/gpuops.c:		cudaGetLastError();
src/num/gpuops.c:void cuda_init(void)
src/num/gpuops.c:	CUDA_ERROR(cudaGetDeviceCount(&count));
src/num/gpuops.c:		if (cuda_try_init(device % count))
src/num/gpuops.c:	if (-1 == cuda_device_id)
src/num/gpuops.c:		error("Could not allocate any GPU device!\n");
src/num/gpuops.c:	cuda_device_id_thread = cuda_device_id;
src/num/gpuops.c:int cuda_get_device_id(void)
src/num/gpuops.c:	return cuda_device_id;
src/num/gpuops.c:void cuda_exit(void)
src/num/gpuops.c:	cuda_memcache_clear();
src/num/gpuops.c:	for (int i = 0; i < CUDA_MAX_STREAMS; i++)
src/num/gpuops.c:		CUDA_ERROR(cudaStreamDestroy(cuda_streams[i]));
src/num/gpuops.c:	cuda_device_id = -1;
src/num/gpuops.c:	cuda_device_id_thread = -1;
src/num/gpuops.c:	CUDA_ERROR(cudaDeviceReset());
src/num/gpuops.c:int cuda_get_stream_id(void)
src/num/gpuops.c:	if (-1 == cuda_device_id)
src/num/gpuops.c:		error("CUDA not initialized!\n");
src/num/gpuops.c:	if (cuda_device_id != cuda_device_id_thread) {
src/num/gpuops.c:		CUDA_ERROR(cudaSetDevice(cuda_device_id));
src/num/gpuops.c:		cuda_device_id_thread = cuda_device_id;
src/num/gpuops.c:	if (omp_get_level() < cuda_stream_level)
src/num/gpuops.c:		cuda_stream_level = -1;
src/num/gpuops.c:	if (-1 == cuda_stream_level)
src/num/gpuops.c:		return CUDA_MAX_STREAMS;
src/num/gpuops.c:	return (0 < CUDA_MAX_STREAMS) ? omp_get_ancestor_thread_num(cuda_stream_level) % CUDA_MAX_STREAMS : 0;
src/num/gpuops.c:int cuda_set_stream_level(void)
src/num/gpuops.c:	if (-1 == cuda_stream_level)
src/num/gpuops.c:		cuda_stream_level = omp_get_level() + 1;
src/num/gpuops.c:	return MIN(cuda_num_streams, CUDA_MAX_STREAMS);
src/num/gpuops.c:cudaStream_t cuda_get_stream_by_id(int id)
src/num/gpuops.c:	return cuda_streams[id];
src/num/gpuops.c:cudaStream_t cuda_get_stream(void)
src/num/gpuops.c:	return cuda_streams[cuda_get_stream_id()];
src/num/gpuops.c:void cuda_sync_device(void)
src/num/gpuops.c:	// do not initialize gpu just for syncing
src/num/gpuops.c:	if (-1 == cuda_device_id)
src/num/gpuops.c:	CUDA_ERROR(cudaDeviceSynchronize());
src/num/gpuops.c:void cuda_sync_stream(void)
src/num/gpuops.c:	// do not initialize gpu just for syncing
src/num/gpuops.c:	if (-1 == cuda_device_id)
src/num/gpuops.c:	CUDA_ERROR(cudaStreamSynchronize(cuda_get_stream()));
src/num/gpuops.c:static void* cuda_malloc_wrapper(size_t size)
src/num/gpuops.c:	if (-1 == cuda_device_id)
src/num/gpuops.c:		error("CUDA_ERROR: No gpu initialized, run \"num_init_gpu\"!\n");
src/num/gpuops.c:	if (cuda_device_id == cuda_device_id_thread) {
src/num/gpuops.c:		CUDA_ERROR(cudaSetDevice(cuda_device_id));
src/num/gpuops.c:		cuda_device_id_thread = cuda_device_id;
src/num/gpuops.c:	if (cuda_global_memory) {
src/num/gpuops.c:		CUDA_ERROR(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
src/num/gpuops.c:		CUDA_ERROR(cudaDeviceGetAttribute(&access, cudaDevAttrConcurrentManagedAccess, cuda_device_id));
src/num/gpuops.c:			CUDA_ERROR(cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, cuda_device_id));
src/num/gpuops.c:			CUDA_ERROR(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cuda_device_id));
src/num/gpuops.c:			CUDA_ERROR(cudaMemPrefetchAsync(ptr, size, cuda_device_id, cuda_get_stream()));
src/num/gpuops.c:		CUDA_ERROR(cudaMalloc(&ptr, size));
src/num/gpuops.c:static void cuda_free_wrapper(const void* ptr)
src/num/gpuops.c:	CUDA_ERROR(cudaFree((void*)ptr));
src/num/gpuops.c:void cuda_free(void* ptr)
src/num/gpuops.c:	mem_device_free(ptr, cuda_free_wrapper);
src/num/gpuops.c:void* cuda_malloc(long size)
src/num/gpuops.c:	return mem_device_malloc((size_t)size, cuda_malloc_wrapper);
src/num/gpuops.c:void cuda_use_global_memory(void)
src/num/gpuops.c:	cuda_global_memory = true;
src/num/gpuops.c:void cuda_memcache_off(void)
src/num/gpuops.c:	cuda_memcache = false;
src/num/gpuops.c:void cuda_memcache_clear(void)
src/num/gpuops.c:	memcache_clear(cuda_free_wrapper);
src/num/gpuops.c://#if CUDART_VERSION >= 10000
src/num/gpuops.c://#define CUDA_GET_CUDA_DEVICE_NUM
src/num/gpuops.c:static bool cuda_ondevice_int(const void* ptr)
src/num/gpuops.c:#ifdef CUDA_GET_CUDA_DEVICE_NUM
src/num/gpuops.c:// Starting with CUDA 10 it has similar speed to the memcache but is 
src/num/gpuops.c:	if (-1 == cuda_device_id)
src/num/gpuops.c:	struct cudaPointerAttributes attr;
src/num/gpuops.c:	if (cudaSuccess != (cudaPointerGetAttributes(&attr, ptr)))
src/num/gpuops.c:	   is to clear the error using cudaGetLastError. See end of:
src/num/gpuops.c:	   http://www.alexstjohn.com/WP/2014/04/28/cuda-6-0-first-look/
src/num/gpuops.c:		cudaGetLastError();
src/num/gpuops.c:	if ((cudaMemoryTypeUnregistered == attr.type) || (cudaMemoryTypeHost == attr.type))
src/num/gpuops.c:bool cuda_ondevice(const void* ptr)
src/num/gpuops.c:	return cuda_ondevice_int(ptr) || is_vptr_gpu(ptr);
src/num/gpuops.c:void cuda_clear(long size, void* dst)
src/num/gpuops.c:	CUDA_ERROR_PTR(dst);
src/num/gpuops.c:	CUDA_ERROR(cudaMemsetAsync(dst, 0, (size_t)size, cuda_get_stream()));
src/num/gpuops.c:static void cuda_float_clear(long size, float* dst)
src/num/gpuops.c:	cuda_clear(size * (long)sizeof(float), (void*)dst);
src/num/gpuops.c:void cuda_memcpy(long size, void* dst, const void* src)
src/num/gpuops.c:	CUDA_ERROR(cudaMemcpyAsync(dst, src, (size_t)size, cudaMemcpyDefault, cuda_get_stream()));
src/num/gpuops.c:void cuda_memcpy_strided(const long dims[2], long ostr, void* dst, long istr, const void* src)
src/num/gpuops.c:	CUDA_ERROR(cudaMemcpy2DAsync(dst, (size_t)ostr, src, (size_t)istr, (size_t)dims[0], (size_t)dims[1], cudaMemcpyDefault, cuda_get_stream()));
src/num/gpuops.c:static void cuda_float_copy(long size, float* dst, const float* src)
src/num/gpuops.c:	cuda_memcpy(size * (long)sizeof(float), (void*)dst, (const void*)src);
src/num/gpuops.c:static float* cuda_float_malloc(long size)
src/num/gpuops.c:	return (float*)cuda_malloc(size * (long)sizeof(float));
src/num/gpuops.c:static void cuda_float_free(float* x)
src/num/gpuops.c:	cuda_free((void*)x);
src/num/gpuops.c:const struct vec_ops gpu_ops = {
src/num/gpuops.c:	.float2double = cuda_float2double,
src/num/gpuops.c:	.double2float = cuda_double2float,
src/num/gpuops.c:	.dot = cuda_dot,
src/num/gpuops.c:	.asum = cuda_asum,
src/num/gpuops.c:	.zsum = cuda_zsum,
src/num/gpuops.c:	.zdot = cuda_cdot,
src/num/gpuops.c:	.add = cuda_add,
src/num/gpuops.c:	.sub = cuda_sub,
src/num/gpuops.c:	.mul = cuda_mul,
src/num/gpuops.c:	.div = cuda_div,
src/num/gpuops.c:	.fmac = cuda_fmac,
src/num/gpuops.c:	.fmacD = cuda_fmacD,
src/num/gpuops.c:	.smul = cuda_smul,
src/num/gpuops.c:	.sadd = cuda_sadd,
src/num/gpuops.c:	.axpy = cuda_saxpy,
src/num/gpuops.c:	.pow = cuda_pow,
src/num/gpuops.c:	.sqrt = cuda_sqrt,
src/num/gpuops.c:	.round = cuda_round,
src/num/gpuops.c:	.le = cuda_le,
src/num/gpuops.c:	.zsmul = cuda_zsmul,
src/num/gpuops.c:	.zsadd = cuda_zsadd,
src/num/gpuops.c:	.zsmax = cuda_zsmax,
src/num/gpuops.c:	.zsmin = cuda_zsmin,
src/num/gpuops.c:	.zmul = cuda_zmul,
src/num/gpuops.c:	.zdiv = cuda_zdiv,
src/num/gpuops.c:	.zfmac = cuda_zfmac,
src/num/gpuops.c:	.zfmacD = cuda_zfmacD,
src/num/gpuops.c:	.zmulc = cuda_zmulc,
src/num/gpuops.c:	.zfmacc = cuda_zfmacc,
src/num/gpuops.c:	.zfmaccD = cuda_zfmaccD,
src/num/gpuops.c:	.zfsq2 = cuda_zfsq2,
src/num/gpuops.c:	.zpow = cuda_zpow,
src/num/gpuops.c:	.zphsr = cuda_zphsr,
src/num/gpuops.c:	.zconj = cuda_zconj,
src/num/gpuops.c:	.zexpj = cuda_zexpj,
src/num/gpuops.c:	.zexp = cuda_zexp,
src/num/gpuops.c:	.zsin = cuda_zsin,
src/num/gpuops.c:	.zcos = cuda_zcos,
src/num/gpuops.c:	.zsinh = cuda_zsinh,
src/num/gpuops.c:	.zcosh = cuda_zcosh,
src/num/gpuops.c:	.zlog = cuda_zlog,
src/num/gpuops.c:	.zarg = cuda_zarg,
src/num/gpuops.c:	.zabs = cuda_zabs,
src/num/gpuops.c:	.zatanr = cuda_zatanr,
src/num/gpuops.c:	.zacosr = cuda_zacosr,
src/num/gpuops.c:	.exp = cuda_exp,
src/num/gpuops.c:	.log = cuda_log,
src/num/gpuops.c:	.zcmp = cuda_zcmp,
src/num/gpuops.c:	.zdiv_reg = cuda_zdiv_reg,
src/num/gpuops.c:	.zfftmod = cuda_zfftmod,
src/num/gpuops.c:	.zmax = cuda_zmax,
src/num/gpuops.c:	.zle = cuda_zle,
src/num/gpuops.c:	.smax = cuda_smax,
src/num/gpuops.c:	.max = cuda_max,
src/num/gpuops.c:	.min = cuda_min,
src/num/gpuops.c:	.zsoftthresh = cuda_zsoftthresh,
src/num/gpuops.c:	.zsoftthresh_half = cuda_zsoftthresh_half,
src/num/gpuops.c:	.softthresh = cuda_softthresh,
src/num/gpuops.c:	.softthresh_half = cuda_softthresh_half,
src/num/gpuops.c:	.pdf_gauss = cuda_pdf_gauss,
src/num/gpuops.c:	.real = cuda_real,
src/num/gpuops.c:	.imag = cuda_imag,
src/num/gpuops.c:	.zcmpl_real = cuda_zcmpl_real,
src/num/gpuops.c:	.zcmpl_imag = cuda_zcmpl_imag,
src/num/gpuops.c:	.zcmpl = cuda_zcmpl,
src/num/gpuops.c:	.zfill = cuda_zfill,
src/num/gpuops.c:extern const struct vec_iter_s gpu_iter_ops;
src/num/gpuops.c:const struct vec_iter_s gpu_iter_ops = {
src/num/gpuops.c:	.allocate = cuda_float_malloc,
src/num/gpuops.c:	.del = cuda_float_free,
src/num/gpuops.c:	.clear = cuda_float_clear,
src/num/gpuops.c:	.copy = cuda_float_copy,
src/num/gpuops.c:	.dot = cuda_dot,
src/num/gpuops.c:	.norm = cuda_norm,
src/num/gpuops.c:	.axpy = cuda_saxpy,
src/num/gpuops.c:	.xpay = cuda_xpay,
src/num/gpuops.c:	.axpbz = cuda_axpbz,
src/num/gpuops.c:	.smul = cuda_smul,
src/num/gpuops.c:	.add = cuda_add,
src/num/gpuops.c:	.sub = cuda_sub,
src/num/gpuops.c:	.swap = cuda_swap,
src/num/gpuops.c:	.zmul = cuda_zmul,
src/num/gpuops.c:	.mul = cuda_mul,
src/num/gpuops.c:	.fmac = cuda_fmac,
src/num/gpuops.c:	.div = cuda_div,
src/num/gpuops.c:	.sqrt = cuda_sqrt,
src/num/gpuops.c:	.smax = cuda_smax,
src/num/gpuops.c:	.le = cuda_le,
src/num/gpuops.c:	.zsmax = cuda_zsmax,
src/num/gpuops.c:	.xpay_bat = cuda_xpay_bat,
src/num/gpuops.c:	.dot_bat = cuda_dot_bat,
src/num/gpuops.c:	.axpy_bat = cuda_axpy_bat,
src/num/gpukrnls_misc.h:#ifdef USE_CUDA
src/num/gpukrnls_misc.h:#include <cuda_runtime_api.h>
src/num/gpukrnls_misc.h:extern int cuda_get_max_threads(const void* func);
src/num/gpukrnls_misc.h:		_threads = cuda_get_max_threads((_func));	\
src/num/gpukrnls_misc.h:		_threads = cuda_get_max_threads((_func));	\
src/num/vecops_strided.c: * and a flag if the optimization should be applied on cpu/gpu
src/num/vecops_strided.c:#ifdef USE_CUDA
src/num/vecops_strided.c:#include "num/gpuops.h"
src/num/vecops_strided.c:	bool on_gpu;
src/num/vecops_strided.c:#define OPT_Z3OP(check_fun, strided_kernel, on_cpu, on_gpu, in_place, reduction, long_dims) \
src/num/vecops_strided.c:	(struct simple_z3op_check){ #strided_kernel, check_fun, strided_kernel, on_cpu, on_gpu, in_place, reduction, long_dims }
src/num/vecops_strided.c:	bool on_gpu;
src/num/vecops_strided.c:#define OPT_3OP(check_fun, strided_kernel, on_cpu, on_gpu, in_place, reduction, long_dims) \
src/num/vecops_strided.c:	(struct simple_3op_check){ #strided_kernel, check_fun, strided_kernel, on_cpu, on_gpu, in_place, reduction, long_dims }
src/num/vecops_strided.c:	bool on_gpu;
src/num/vecops_strided.c:	bool on_gpu = false;
src/num/vecops_strided.c:#ifdef USE_CUDA
src/num/vecops_strided.c:	on_gpu = cuda_ondevice(out);
src/num/vecops_strided.c:	if (on_gpu) {
src/num/vecops_strided.c:		assert(cuda_ondevice(in1));
src/num/vecops_strided.c:		assert(cuda_ondevice(in2));
src/num/vecops_strided.c:		bool applicable = on_gpu ? strided_call.on_gpu : strided_call.on_cpu;
src/num/vecops_strided.c:	bool on_gpu = false;
src/num/vecops_strided.c:#ifdef USE_CUDA
src/num/vecops_strided.c:	on_gpu = cuda_ondevice(out);
src/num/vecops_strided.c:	if (on_gpu) {
src/num/vecops_strided.c:		assert(cuda_ondevice(in1));
src/num/vecops_strided.c:		assert(cuda_ondevice(in2));
src/num/vecops_strided.c:		bool applicable = on_gpu ? strided_call.on_gpu : strided_call.on_cpu;
src/num/vecops_strided.c:	#ifdef USE_CUDA
src/num/vecops_strided.c:		if (cuda_ondevice(out))
src/num/vecops_strided.c:			applicable &= strided_calls[i].on_gpu;
src/num/vecops_strided.c:		OPT_Z3OP(check_batched_select,	zfmac_gpu_batched_loop, true, false, false, false, true),
src/num/vecops_strided.c:		OPT_Z3OP(check_unfold, zfmac_gpu_unfold, true, false, true, false, true),
src/num/vecops_strided.c:		OPT_Z3OP(check_batched_select,	zfmacc_gpu_batched_loop, true, false, false, false, true),
src/num/vecops_strided.c:		OPT_Z3OP(check_unfold,	zfmacc_gpu_unfold, true, false, true, false, true),
src/num/vecops_strided.c:		OPT_3OP(check_unfold, fmac_gpu_unfold, true, false, true, false, true),
src/num/vecops_strided.c:		OPT_Z3OP(check_unfold, zmul_gpu_unfold, true, false, true, false, true),
src/num/vecops_strided.c:		OPT_Z3OP(check_unfold,	zmulc_gpu_unfold, true, false, true, false, true),
src/num/vecops_strided.c:		OPT_3OP(check_unfold,	mul_gpu_unfold, true, false, true, false, true),
src/num/vecops_strided.c:		OPT_Z3OP(check_unfold,		zadd_gpu_unfold, true, false, false, false, true),
src/num/vecops_strided.c:		OPT_Z3OP(check_reduce_outer,	reduce_zadd_outer_gpu, true, false, false, true, false),
src/num/vecops_strided.c:		OPT_Z3OP(check_reduce_inner,	reduce_zadd_inner_gpu, true, false, false, true, false),
src/num/vecops_strided.c:		OPT_3OP(check_unfold,	add_gpu_unfold, true, false, false, false, true),
src/num/vecops_strided.c:		OPT_3OP(check_reduce_outer,	reduce_add_outer_gpu, true, false, false, true, false),
src/num/vecops_strided.c:		OPT_3OP(check_reduce_inner,	reduce_add_inner_gpu, true, false, false, true, false),
src/num/vecops_strided.c:		OPT_Z3OP(check_reduce_outer,	reduce_zmax_outer_gpu, true, false, false, true, false),
src/num/vecops_strided.c:		OPT_Z3OP(check_reduce_inner,	reduce_zmax_inner_gpu, true, false, false, true, false),
src/num/convcorr.h:zconvcorr_fwd_algo_f zconvcorr_fwd_im2col_cf_gpu;
src/num/convcorr.h:zconvcorr_bwd_in_algo_f zconvcorr_bwd_in_im2col_cf_gpu;
src/num/convcorr.h:zconvcorr_bwd_krn_algo_f zconvcorr_bwd_krn_im2col_cf_gpu;
src/num/convcorr.h:				float max_rmse, _Bool gpu, long min_no_algos);
src/num/convcorr.h:				float max_rmse, _Bool gpu, long min_no_algos);
src/num/convcorr.h:				float max_rmse, _Bool gpu, long min_no_algos);
src/num/gpukrnls_copy.h:extern void cuda_copy_ND(int D, const long dims[], const long ostrs[], void* dst, const long istrs[], const void* src, unsigned long size);
src/moba.c:#ifdef USE_CUDA
src/moba.c:#include "num/gpuops.h"
src/moba.c:		OPT_SET('g', &bart_use_gpu, "use gpu"),
src/moba.c:		OPTL_INT(0, "multi-gpu", &(conf.num_gpu), "num", "(number of gpus to use)"),
src/moba.c:	if (0 != conf.num_gpu)
src/moba.c:		error("Multi-GPU only supported by MPI!\n");
src/moba.c:	num_init_gpu_support();
src/moba.c:#ifdef  USE_CUDA
src/moba.c:	if (bart_use_gpu) {
src/moba.c:		complex float* kspace_gpu = md_alloc_gpu(DIMS, grid_dims, CFL_SIZE);
src/moba.c:		md_copy(DIMS, grid_dims, kspace_gpu, k_grid_data, CFL_SIZE);
src/moba.c:		moba_recon(&conf, &data, dims, img, sens, pattern, mask, TI, TE_IR_MGRE, b1, b0, kspace_gpu, init);
src/moba.c:		md_free(kspace_gpu);
src/pocsense.c:		OPT_SET('g', &bart_use_gpu, "()"),
src/pocsense.c:	num_init_gpu_support();
src/pocsense.c:	if (bart_use_gpu)
src/pocsense.c:#ifdef USE_CUDA
src/pocsense.c:		pocs_recon_gpu2(italgo, iconf, (const struct linop_s**)ops2, dims, thresh_op, alpha, lambda, result, sens_maps, pattern, kspace_data);
src/version.c:		bart_printf("CUDA=");
src/version.c:#ifdef USE_CUDA
src/mobafit.c:		OPT_SET('g', &bart_use_gpu, "use gpu"),
src/mobafit.c:	num_init_gpu_support();
src/mobafit.c:	lsqr_conf.it_gpu = false;
src/mobafit.c:	if (bart_use_gpu) {
src/mobafit.c:#ifdef USE_CUDA
src/mobafit.c:		y_patch = md_alloc_gpu(DIMS, y_patch_dims, CFL_SIZE);
src/mobafit.c:		x_patch = md_alloc_gpu(DIMS, x_patch_dims, CFL_SIZE);
src/mobafit.c:		error("Compiled without GPU support!\n");
src/bench.c:	complex float* x = md_alloc_gpu(DIMS, dims, CFL_SIZE);
src/bench.c:	complex float* y = md_alloc_gpu(DIMS, dims, CFL_SIZE);
src/networks/reconet.c:#ifdef USE_CUDA
src/networks/reconet.c:#include "num/gpuops.h"
src/networks/reconet.c:	.gpu = false,
src/networks/reconet.c:		ret = nn_stack_multigpu_F(mpi_get_num_procs(), tmp, BATCH_DIM);
src/networks/reconet.c:		mri_ops_activate_multigpu();
src/networks/reconet.c:	if (config->gpu)
src/networks/reconet.c:		move_gpu_nn_weights(config->weights);
src/networks/reconet.c:	if (config->gpu)
src/networks/reconet.c:		move_gpu_nn_weights(config->weights);
src/networks/reconet.h:	_Bool gpu;
src/networks/nlinvnet.c:#ifdef USE_CUDA
src/networks/nlinvnet.c:#include "num/gpuops.h"
src/networks/nlinvnet.c:	auto nn_train = (1 == M) ? train_ops[0] : nn_stack_multigpu_F(M, train_ops, -1);
src/networks/nlinvnet.c:	if (bart_use_gpu)
src/networks/nlinvnet.c:		move_gpu_nn_weights(nlinvnet->weights);
src/networks/nlinvnet.c:	if (bart_use_gpu)
src/networks/nlinvnet.c:		move_gpu_nn_weights(nlinvnet->weights);
src/networks/misc.h:	_Bool gpu;
src/networks/nnet.c:	.gpu = false,
src/networks/nnet.c:	if (config->gpu)
src/networks/nnet.c:		move_gpu_nn_weights(config->weights);
src/networks/nnet.c:	if (config->gpu)
src/networks/nnet.c:		move_gpu_nn_weights(config->weights);
src/networks/misc.c:	.gpu = false,
src/networks/misc.c:#ifdef USE_CUDA
src/networks/misc.c:	if (nd->gpu && !use_compat_to_version("v0.8.00"))
src/networks/misc.c:		ref = md_alloc_gpu(1, MD_DIMS(1), CFL_SIZE);
src/networks/misc.c:#ifdef USE_CUDA
src/networks/misc.c:	if (nd->gpu && !use_compat_to_version("v0.8.00"))
src/networks/misc.c:		ref = md_alloc_gpu(1, MD_DIMS(1), CFL_SIZE);
src/networks/nnet.h:	_Bool gpu;
src/mnist.c:#ifdef USE_CUDA
src/mnist.c:#include "num/gpuops.h"
src/mnist.c:		OPTL_SET('g', "gpu", &(bart_use_gpu), "run on gpu"),
src/mnist.c:	num_init_gpu_support();
src/mnist.c:		if (bart_use_gpu)
src/mnist.c:			move_gpu_nn_weights(weights);
src/mnist.c:		if (bart_use_gpu)
src/mnist.c:			move_gpu_nn_weights(weights);
src/bart.c:#ifdef USE_CUDA
src/bart.c:#include "num/gpuops.h"
src/bart.c:#ifdef USE_CUDA
src/bart.c:	cuda_memcache_clear();
src/bart.c:#ifdef USE_CUDA
src/bart.c:			cuda_set_stream_level();
src/ecalib.c:		OPT_SET('g', &conf.usegpu, "()"),
src/ecalib.c:	bart_use_gpu = conf.usegpu;
src/ecalib.c:	num_init_gpu_support();
src/motion/affine.c:	      const struct nlop_s* trafo, bool gpu, bool cubic)
src/motion/affine.c:	if (gpu)
src/motion/affine.c:		nlop_it = nlop_gpu_wrapper_F(nlop_it);
src/motion/affine.c:		if (gpu)
src/motion/affine.c:			nlop_itmm = nlop_gpu_wrapper_F(nlop_itmm);
src/motion/affine.c:		if (gpu)
src/motion/affine.c:			nlop_itms = nlop_gpu_wrapper_F(nlop_itms);
src/motion/affine.c:void affine_reg(bool gpu, bool cubic, complex float* affine, const struct nlop_s* _trafo, long sdims[3], const complex float* img_static, const complex float* msk_static, long mdims[3], const complex float* img_moving, const complex float* msk_moving, int N, float sigma[N], float factor[N])
src/motion/affine.c:		const struct nlop_s* nlop = affine_reg_nlop_create(cdims, wimg_static, wmsk_static, mdims, wimg_moving, msk_moving, trafo, gpu, cubic);
src/motion/interpolate.c:#ifdef USE_CUDA
src/motion/interpolate.c:#include "num/gpuops.h"
src/motion/interpolate.c:#ifdef USE_CUDA
src/motion/interpolate.c:#include "motion/gpu_interpolate.h"
src/motion/interpolate.c:#ifdef USE_CUDA
src/motion/interpolate.c:	if (cuda_ondevice(pos)) {
src/motion/interpolate.c:		cuda_positions(N, d, flags, sdims, pdims, pos);
src/motion/interpolate.c:#ifdef USE_CUDA
src/motion/interpolate.c:		if (cuda_ondevice(coor))
src/motion/interpolate.c:			cuda_interpolate2(ord, M, idims_red, istrs_red, intp + ioffset, cstrs_red, cstrs[d], coor + coffset, gdims_red, gstrs_red, grid + goffset);
src/motion/interpolate.c:#ifdef USE_CUDA
src/motion/interpolate.c:		if (cuda_ondevice(coor))
src/motion/interpolate.c:			cuda_interpolateH2(ord, M, gdims_red, gstrs_red, grid + goffset, idims_red, istrs_red, intp + ioffset, cstrs_red, cstrs[d], coor + coffset);
src/motion/interpolate.c:#ifdef USE_CUDA
src/motion/interpolate.c:		if (cuda_ondevice(dcoor))
src/motion/interpolate.c:			cuda_interpolate_adj_coor2(ord, M, idims_red, istrs_red, dintp + ioffset, cstrs_red, cstrs[d], coor + coffset, dcoor + coffset, gdims_red, gstrs_red, grid + goffset);
src/motion/interpolate.c:#ifdef USE_CUDA
src/motion/interpolate.c:		if (cuda_ondevice(dcoor))
src/motion/interpolate.c:			cuda_interpolate_der_coor2(ord, M, idims_red, istrs_red, dintp + ioffset, cstrs_red, cstrs[d], coor + coffset, dcoor + coffset, gdims_red, gstrs_red, grid + goffset);
src/motion/affine.h:extern void affine_reg(_Bool gpu, _Bool cubic, _Complex float* affine, const struct nlop_s* trafo, long sdims[3], const _Complex float* img_static, const _Complex float* msk_static, long mdims[3], const _Complex float* img_moving, const _Complex float* msk_moving,
src/motion/gpu_interpolate.h:void cuda_positions(int N, int d, unsigned long flags, const long sdims[__VLA(N)], const long pdims[__VLA(N)], _Complex float* pos);
src/motion/gpu_interpolate.h:void cuda_interpolate2(int ord, int M, 
src/motion/gpu_interpolate.h:void cuda_interpolateH2(int ord, int M, 
src/motion/gpu_interpolate.h:void cuda_interpolate_adj_coor2(int ord, int M, 
src/motion/gpu_interpolate.h:void cuda_interpolate_der_coor2(int ord, int M, 
src/motion/gpu_interpolate.cu:#include <cuda_runtime_api.h>
src/motion/gpu_interpolate.cu:#include <cuda.h>
src/motion/gpu_interpolate.cu:#include "num/gpukrnls_misc.h"
src/motion/gpu_interpolate.cu:#include "num/gpuops.h"
src/motion/gpu_interpolate.cu:#include "gpu_interpolate.h"
src/motion/gpu_interpolate.cu:void cuda_positions(int N, int d, unsigned long flags, const long sdims[__VLA(N)], const long pdims[__VLA(N)], _Complex float* pos)
src/motion/gpu_interpolate.cu:	kern_positions<<<cu_grid, cu_block, 0, cuda_get_stream() >>>(pd, tot, (cuFloatComplex*)pos);
src/motion/gpu_interpolate.cu:	CUDA_KERNEL_ERROR;
src/motion/gpu_interpolate.cu:static struct intp_data cuda_intp_get_data(int M, 
src/motion/gpu_interpolate.cu:static void cuda_intp_temp(int M, 
src/motion/gpu_interpolate.cu:	struct intp_data id = cuda_intp_get_data(M, grid_dims, grid_strs, intp_dims, intp_strs, coor_strs, coor_dir_dim_str, ord, width);
src/motion/gpu_interpolate.cu:		kern_intp<true><<<cu_grid, cu_block, 0, cuda_get_stream()>>>(id, (const cuFloatComplex*)coor, (cuFloatComplex*)grid, (const cuFloatComplex*)intp);
src/motion/gpu_interpolate.cu:		kern_intp<false><<<cu_grid, cu_block, 0, cuda_get_stream()>>>(id, (const cuFloatComplex*)coor, (cuFloatComplex*)intp, (const cuFloatComplex*)grid);
src/motion/gpu_interpolate.cu:	CUDA_KERNEL_ERROR;
src/motion/gpu_interpolate.cu:void cuda_interpolate2(int ord, int M, 
src/motion/gpu_interpolate.cu:	cuda_intp_temp<false>(M, grid_dims, grid_strs, (_Complex float*)grid, intp_dims, intp_strs, intp, coor_strs, coor_dir_dim_str, coor, ord, ord + 1);
src/motion/gpu_interpolate.cu:void cuda_interpolateH2(int ord, int M, 
src/motion/gpu_interpolate.cu:	cuda_intp_temp<true>(M, grid_dims, grid_strs, grid, intp_dims, intp_strs, (_Complex float*)intp, coor_strs, coor_dir_dim_str, coor, ord, ord + 1);
src/motion/gpu_interpolate.cu:void cuda_interpolate_adj_coor2(int ord, int M, 
src/motion/gpu_interpolate.cu:	struct intp_data id = cuda_intp_get_data(M, grid_dims, grid_strs, intp_dims, intp_strs, coor_strs, coor_dir_dim_str, ord, ord + 1);
src/motion/gpu_interpolate.cu:	kern_intp_point_adj_coor<<<cu_grid, cu_block, 0, cuda_get_stream() >>>(id, (const cuFloatComplex*)coor, (cuFloatComplex*)dcoor, (const cuFloatComplex*)grid, (const cuFloatComplex*)dintp);
src/motion/gpu_interpolate.cu:	CUDA_KERNEL_ERROR;
src/motion/gpu_interpolate.cu:void cuda_interpolate_der_coor2(int ord, int M, 
src/motion/gpu_interpolate.cu:	struct intp_data id = cuda_intp_get_data(M, grid_dims, grid_strs, intp_dims, intp_strs, coor_strs, coor_dir_dim_str, ord, ord + 1);
src/motion/gpu_interpolate.cu:	kern_intp_point_der_coor<<<cu_grid, cu_block, 0, cuda_get_stream() >>>(id, (const cuFloatComplex*)coor, (const cuFloatComplex*)dcoor, (const cuFloatComplex*)grid, (cuFloatComplex*)dintp);
src/motion/gpu_interpolate.cu:	CUDA_KERNEL_ERROR;
src/noncart/gpu_grid.cu:#include <cuda_runtime_api.h>
src/noncart/gpu_grid.cu:#include <cuda.h>
src/noncart/gpu_grid.cu:#include "num/gpukrnls_misc.h"
src/noncart/gpu_grid.cu:#include "num/gpuops.h"
src/noncart/gpu_grid.cu:#include "gpu_grid.h"
src/noncart/gpu_grid.cu:extern "C" void cuda_apply_linphases_3D(int N, const long img_dims[], const float shifts[3], _Complex float* dst, const _Complex float* src, _Bool conj, _Bool fmac, _Bool fftm, float scale)
src/noncart/gpu_grid.cu:		kern_apply_linphases_3D<true><<<getGridSize3(c.dims, func), getBlockSize3(c.dims, (const void*)func), 0, cuda_get_stream()>>>(c, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/noncart/gpu_grid.cu:		kern_apply_linphases_3D<false><<<getGridSize3(c.dims, func), getBlockSize3(c.dims, (const void*)func), 0, cuda_get_stream()>>>(c, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/noncart/gpu_grid.cu:extern "C" void cuda_apply_rolloff_correction2(float os, float width, float beta, int N, const long dims[4], const long ostrs[4], _Complex float* dst, const long istrs[4], const _Complex float* src)
src/noncart/gpu_grid.cu:	kern_apply_rolloff_correction<<<getGridSize3(c.dims, func), getBlockSize3(c.dims, (const void*)func), 0, cuda_get_stream()>>>(c, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
src/noncart/gpu_grid.cu:static void kb_precompute_gpu(double beta)
src/noncart/gpu_grid.cu:	#pragma omp critical(kb_tbale_gpu)
src/noncart/gpu_grid.cu:void cuda_grid(const struct grid_conf_s* conf, const long ksp_dims[4], const long trj_strs[4], const _Complex float* traj, const long grid_dims[4], const long grid_strs[4], _Complex float* grid, const long ksp_strs[4], const _Complex float* src)
src/noncart/gpu_grid.cu:	kb_precompute_gpu(conf->beta);
src/noncart/gpu_grid.cu:	kern_grid<false><<<cu_grid, cu_block, 0, cuda_get_stream() >>>(gd, (const cuFloatComplex*)traj, (cuFloatComplex*)grid, (const cuFloatComplex*)src);
src/noncart/gpu_grid.cu:	CUDA_KERNEL_ERROR;
src/noncart/gpu_grid.cu:void cuda_gridH(const struct grid_conf_s* conf, const long ksp_dims[4], const long trj_strs[4], const _Complex float* traj, const long ksp_strs[4], _Complex float* dst, const long grid_dims[4], const long grid_strs[4], const _Complex float* grid)
src/noncart/gpu_grid.cu:	kb_precompute_gpu(conf->beta);
src/noncart/gpu_grid.cu:	kern_grid<true><<<cu_grid, cu_block, 0, cuda_get_stream() >>>(gd, (const cuFloatComplex*)traj, (cuFloatComplex*)dst, (const cuFloatComplex*)grid);
src/noncart/gpu_grid.cu:	CUDA_KERNEL_ERROR;
src/noncart/grid.c:#ifdef USE_CUDA
src/noncart/grid.c:#include "num/gpuops.h"
src/noncart/grid.c:#include "noncart/gpu_grid.h"
src/noncart/grid.c:#ifdef USE_CUDA
src/noncart/grid.c:	if (cuda_ondevice(traj))
src/noncart/grid.c:		return cuda_gridH(conf, ksp_dims, trj_strs, traj, ksp_strs, dst, grid_dims, grid_strs, grid);
src/noncart/grid.c:#ifdef USE_CUDA
src/noncart/grid.c:	if (cuda_ondevice(traj))
src/noncart/grid.c:		return cuda_grid(conf, ksp_dims, trj_strs, traj, grid_dims, grid_strs, grid, ksp_strs, src);
src/noncart/grid.c:#ifdef USE_CUDA
src/noncart/grid.c:	if (cuda_ondevice(traj))
src/noncart/grid.c:#ifdef USE_CUDA
src/noncart/grid.c:	if (cuda_ondevice(traj))
src/noncart/grid.c:#ifdef USE_CUDA
src/noncart/grid.c:	assert(cuda_ondevice(dst) == cuda_ondevice(src));
src/noncart/grid.c:	if (cuda_ondevice(dst)) {
src/noncart/grid.c:		long dims_cuda[4] = { dims[0], dims[1], dims[2], md_calc_size(N - 3, dims + 3) };
src/noncart/grid.c:		long ostrs_cuda[4] = { ostrs[0] / (long)CFL_SIZE, ostrs[1] / (long)CFL_SIZE, ostrs[2] / (long)CFL_SIZE, obstr };
src/noncart/grid.c:		long istrs_cuda[4] = { istrs[0] / (long)CFL_SIZE, istrs[1] / (long)CFL_SIZE, istrs[2] / (long)CFL_SIZE, ibstr };
src/noncart/grid.c:		cuda_apply_rolloff_correction2(os, width, beta, N, dims_cuda, ostrs_cuda, dst, istrs_cuda, src);
src/noncart/gpu_grid.h:extern void cuda_apply_linphases_3D(int N, const long img_dims[__VLA(N)], const float shifts[3], _Complex float* dst, const _Complex float* src, _Bool conj, _Bool fmac, _Bool fftm, float scale);
src/noncart/gpu_grid.h:extern void cuda_apply_rolloff_correction2(float os, float width, float beta, int N, const long dims[4], const long ostrs[4], _Complex float* dst, const long istrs[4], const _Complex float* src);
src/noncart/gpu_grid.h:extern void cuda_grid(const struct grid_conf_s* conf, const long ksp_dims[4], const long trj_strs[4], const _Complex float* traj, const long grid_dims[4], const long grid_strs[4], _Complex float* grid, const long ksp_strs[4], const _Complex float* src);
src/noncart/gpu_grid.h:extern void cuda_gridH(const struct grid_conf_s* conf, const long ksp_dims[4], const long trj_strs[4], const _Complex float* traj, const long ksp_strs[4], _Complex float* dst, const long grid_dims[4], const long grid_strs[4], const _Complex float* grid);
src/noncart/nufft.c:#ifdef USE_CUDA
src/noncart/nufft.c:#include "num/gpuops.h"
src/noncart/nufft.c:#include "noncart/gpu_grid.h"
src/noncart/nufft.c:#ifdef USE_CUDA
src/noncart/nufft.c:	assert(cuda_ondevice(dst) == cuda_ondevice(src));
src/noncart/nufft.c:	if (cuda_ondevice(dst)) {
src/noncart/nufft.c:		cuda_apply_linphases_3D(N, img_dims, shifts, dst, src, conj, fmac, fftm, scale);
src/noncart/nufft.c:#ifdef USE_CUDA
src/noncart/nufft.c:	bool gpu = cuda_ondevice(traj);
src/noncart/nufft.c:	bool gpu = false;
src/noncart/nufft.c:	if (lowmem || gpu)
src/noncart/nufft.c:#ifdef USE_CUDA
src/noncart/nufft.c:	//assert(!cuda_ondevice(src));
src/grecon/optreg.h:extern void opt_reg_configure(int N, const long img_dims[__VLA(N)], struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], int llr_blk, int shift_mode, const char* wtype_str, _Bool use_gpu);
src/grecon/optreg.c:void opt_reg_configure(int N, const long img_dims[N], struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], int llr_blk, int shift_mode, const char* wtype_str, bool use_gpu)
src/grecon/optreg.c:			if (use_gpu)
src/grecon/optreg.c:				error("GPU operation is not currently implemented for NIHT.\n");
src/grecon/optreg.c:			if (use_gpu)
src/grecon/optreg.c:				error("GPU operation is not currently implemented for NIHT.\n");
src/grecon/optreg.c:			if (use_gpu) {
src/grecon/optreg.c:				debug_printf(DP_WARN, "Lowrank regularization is not GPU accelerated.\n");
src/grecon/optreg.c:			prox_ops[nr2] = lrthresh_create(img_dims, randshift, regs[nr].xflags, (const long (*)[DIMS])blkdims, regs[nr].lambda, false, 0, use_gpu);
src/grecon/optreg.c:			const struct linop_s* decom_op = sum_create( img_dims, use_gpu );
src/moba/recon_meco.c:			lsqr_conf.it_gpu = false;
src/moba/moba.h:	int num_gpu;
src/moba/iter_l1.c:#ifdef USE_CUDA
src/moba/iter_l1.c:#include "num/gpuops.h"
src/moba/iter_l1.c:	if (0 < cuda_num_devices()) {	// FIXME: not a correct check for GPU mode
src/moba/iter_l1.c:		tmp = operator_p_gpu_wrapper(tmp2);
src/moba/iter_l1.c:		lsqr_conf.it_gpu = false;
src/moba/iter_l1.c:		if (0 < cuda_num_devices())	// FIXME: not correct check for GPU mode
src/moba/iter_l1.c:			lsqr_conf.it_gpu = true;
src/moba/blochfun.c:	// Copy necessary files from GPU to CPU
src/moba/blochfun.c:	// Collect data of signal (potentially on GPU)
src/moba/meco.c:#include "num/gpuops.h"
src/wavelet/wl3-cuda.cu:#include <cuda.h>
src/wavelet/wl3-cuda.cu:#include "num/gpuops.h"
src/wavelet/wl3-cuda.cu:#include "wl3-cuda.h"
src/wavelet/wl3-cuda.cu:// extern "C" size_t cuda_shared_mem;
src/wavelet/wl3-cuda.cu:extern "C" void wl3_cuda_down3(const long dims[3], const long out_str[3], _Complex float* out, const long in_str[3], const _Complex float* in, unsigned int flen, const float filter[__VLA(flen)])
src/wavelet/wl3-cuda.cu:	kern_down3<<< bl, th, 0, cuda_get_stream() >>>(dims3, ostrs, (cuFloatComplex*)out, istrs, (const cuFloatComplex*)in, flen, filter);
src/wavelet/wl3-cuda.cu:	CUDA_KERNEL_ERROR;
src/wavelet/wl3-cuda.cu:extern "C" void wl3_cuda_up3(const long dims[3], const long out_str[3], _Complex float* out, const long in_str[3],  const _Complex float* in, unsigned int flen, const float filter[__VLA(flen)])
src/wavelet/wl3-cuda.cu:	kern_up3<<< bl, th, 0, cuda_get_stream() >>>(dims3, ostrs, (cuFloatComplex*)out, istrs, (const cuFloatComplex*)in, flen, filter);
src/wavelet/wl3-cuda.cu:	CUDA_KERNEL_ERROR;
src/wavelet/wl3-cuda.h:extern void wl3_cuda_down3(const long dims[3], const long out_str[3], _Complex float* out, const long in_str[3], const _Complex float* in, unsigned int flen, const float filter[__VLA(flen)]);
src/wavelet/wl3-cuda.h:extern void wl3_cuda_up3(const long dims[3], const long out_str[3], _Complex float* out, const long in_str[3],  const _Complex float* in, unsigned int flen, const float filter[__VLA(flen)]);
src/wavelet/wavelet.c: * - all higher-level code should work for GPU as well
src/wavelet/wavelet.c: * - GPU version is not optimized
src/wavelet/wavelet.c:#ifdef USE_CUDA
src/wavelet/wavelet.c:#include "num/gpuops.h"
src/wavelet/wavelet.c:#include "wavelet/wl3-cuda.h"
src/wavelet/wavelet.c:#ifdef  USE_CUDA
src/wavelet/wavelet.c:	if (cuda_ondevice(in)) {
src/wavelet/wavelet.c:		assert(cuda_ondevice(low));
src/wavelet/wavelet.c:		assert(cuda_ondevice(hgh));
src/wavelet/wavelet.c:		float* flow = md_gpu_move(1, MD_DIMS(flen), filter[0][0], FL_SIZE);
src/wavelet/wavelet.c:		float* fhgh = md_gpu_move(1, MD_DIMS(flen), filter[0][1], FL_SIZE);
src/wavelet/wavelet.c:		wl3_cuda_down3(wdims, wostr, low, wistr, in, flen, flow);
src/wavelet/wavelet.c:		wl3_cuda_down3(wdims, wostr, hgh, wistr, in, flen, fhgh);
src/wavelet/wavelet.c:#ifdef  USE_CUDA
src/wavelet/wavelet.c:	if (cuda_ondevice(out)) {
src/wavelet/wavelet.c:		assert(cuda_ondevice(low));
src/wavelet/wavelet.c:		assert(cuda_ondevice(hgh));
src/wavelet/wavelet.c:		float* flow = md_gpu_move(1, MD_DIMS(flen), filter[1][0], FL_SIZE);
src/wavelet/wavelet.c:		float* fhgh = md_gpu_move(1, MD_DIMS(flen), filter[1][1], FL_SIZE);
src/wavelet/wavelet.c:		wl3_cuda_up3(wdims, wostr, out, wistr, low, flen, flow);
src/wavelet/wavelet.c:		wl3_cuda_up3(wdims, wostr, out, wistr, hgh, flen, fhgh);
src/reconet.c:		OPTL_SET('g', "gpu", &(bart_use_gpu), "run on gpu"),
src/reconet.c:	config.gpu = bart_use_gpu;
src/reconet.c:	num_init_gpu_support();
src/reconet.c:	data.gpu = config.gpu;
src/reconet.c:		valid_data.gpu = config.gpu;
src/iter/prox.c:#ifdef USE_CUDA
src/iter/prox.c:#include "num/gpuops.h"
src/iter/iter6.c:	//gpu ref (dst[i] can be null if batch_gen)
src/iter/iter6.c:	float* gpu_ref = NULL;
src/iter/iter6.c:			gpu_ref = dst[i];
src/iter/iter6.c:	assert(NULL != gpu_ref);
src/iter/iter6.c:		select_vecops(gpu_ref),
src/iter/iter6.c:	//gpu ref (dst[i] can be null if batch_gen)
src/iter/iter6.c:	float* gpu_ref = NULL;
src/iter/iter6.c:			gpu_ref = dst[i];
src/iter/iter6.c:	assert(NULL != gpu_ref);
src/iter/iter6.c:			x_old[i] = md_alloc_sameplace(1, isize + i, FL_SIZE, gpu_ref);
src/iter/iter6.c:		select_vecops(gpu_ref),
src/iter/lsqr.h:	_Bool it_gpu;
src/iter/lsqr.c:const struct lsqr_conf lsqr_defaults = { .lambda = 0., .it_gpu = false, .warmstart = false, .icont = NULL };
src/iter/lsqr.c:	if (conf->it_gpu) {
src/iter/lsqr.c:		debug_printf(DP_DEBUG1, "lsqr: add GPU wrapper\n");
src/iter/lsqr.c:		auto tmp = operator_p_gpu_wrapper(itop_op);
src/iter/vec.h:#ifdef USE_CUDA
src/iter/vec.h:extern const struct vec_iter_s gpu_iter_ops;
src/iter/vec.c:#include "num/gpuops.h"
src/iter/vec.c:// defined in vecops.c and gpuops.c
src/iter/vec.c:#ifdef USE_CUDA
src/iter/vec.c:	return cuda_ondevice(x) ? &gpu_iter_ops : &cpu_iter_ops;
src/iter/misc.h:extern double estimate_maxeigenval_gpu(const struct operator_s* op);
src/iter/misc.c:#ifdef USE_CUDA
src/iter/misc.c:double estimate_maxeigenval_gpu(const struct operator_s* op)
src/iter/misc.c:	void* x = md_alloc_gpu(io->N, io->dims, io->size);
src/iter/proj.c:#ifdef USE_CUDA
src/iter/proj.c:#include "num/gpuops.h"
src/iter/proj.c:	md_real(data->N, bdims, (float*)dst, tmp); // I don't trust zmulc to have vanishing imag on gpu
src/sense/recon.c:	.gpu = false,
src/sense/recon.c:	lsqr_conf.it_gpu = conf->gpu;
src/sense/pocs.c:#ifdef USE_CUDA
src/sense/pocs.c:void pocs_recon_gpu(const long dims[DIMS], const struct operator_p_s* thresh, int maxiter, float alpha, float lambda, complex float* result, const complex float* maps, const complex float* pattern, const complex float* kspace)
src/sense/pocs.c:	pocs_recon_gpu2(iter2_pocs, &pconf, NULL, dims, thresh, alpha, lambda, result, maps, pattern, kspace);
src/sense/pocs.c:void pocs_recon_gpu2(italgo_fun2_t italgo, void* iconf, const struct linop_s** ops, const long dims[DIMS], const struct operator_p_s* thresh, float alpha, float lambda, complex float* result, const complex float* maps, const complex float* pattern, const complex float* kspace)
src/sense/pocs.c:	complex float* gpu_maps = md_gpu_move(DIMS, dims, maps, CFL_SIZE);
src/sense/pocs.c:	complex float* gpu_pat = md_gpu_move(DIMS, dims_pat, pattern, CFL_SIZE);
src/sense/pocs.c:	complex float* gpu_ksp = md_gpu_move(DIMS, dims_ksp, kspace, CFL_SIZE);
src/sense/pocs.c:	complex float* gpu_result = md_gpu_move(DIMS, dims_ksp, result, CFL_SIZE);
src/sense/pocs.c:	pocs_recon2(italgo, iconf, ops, dims, thresh, alpha, lambda, gpu_result, gpu_maps, gpu_pat, gpu_ksp);
src/sense/pocs.c:	md_copy(DIMS, dims_ksp, result, gpu_result, CFL_SIZE);
src/sense/pocs.c:	md_free(gpu_result);
src/sense/pocs.c:	md_free(gpu_pat);
src/sense/pocs.c:	md_free(gpu_ksp);
src/sense/pocs.c:	md_free(gpu_maps);
src/sense/recon.h:	_Bool gpu;
src/sense/pocs.h:#ifdef USE_CUDA
src/sense/pocs.h:extern void pocs_recon_gpu(const long dims[DIMS], const struct operator_p_s* thresh, int maxiter, float alpha, float lambda, _Complex float* result, const _Complex float* maps, const _Complex float* pattern, const _Complex float* kspace);
src/sense/pocs.h:extern void pocs_recon_gpu2(italgo_fun2_t italgo, void* iconf, const struct linop_s** ops, const long dims[DIMS], const struct operator_p_s* thresh, float alpha, float lambda, _Complex float* result, const _Complex float* maps, const _Complex float* pattern, const _Complex float* kspace);
src/wshfl.c:#ifdef USE_CUDA
src/wshfl.c:#include "num/gpuops.h"
src/wshfl.c:	complex float* gpu_kernel;
src/wshfl.c:	long gpu_kernel_dims[DIMS] = { [0 ... DIMS - 1] = 1};
src/wshfl.c:	md_copy_dims(DIMS, gpu_kernel_dims, data->kernel_dims);
src/wshfl.c:	gpu_kernel_dims[0] = wx;
src/wshfl.c:	gpu_kernel_dims[3] = nc;
src/wshfl.c:	long gpu_kernel_str[DIMS];
src/wshfl.c:	md_calc_strides(DIMS, gpu_kernel_str, gpu_kernel_dims, CFL_SIZE);
src/wshfl.c:#ifdef USE_CUDA
src/wshfl.c:	if (cuda_ondevice(src))
src/wshfl.c:		md_zfmac2(DIMS, fmac_dims, output_str, dst, input_str, src, gpu_kernel_str, data->gpu_kernel);
src/wshfl.c:#ifdef USE_CUDA
src/wshfl.c:	if (data->gpu_kernel != NULL)
src/wshfl.c:		md_free(data->gpu_kernel);
src/wshfl.c:static const struct linop_s* linop_kern_create(bool gpu_flag, 
src/wshfl.c:	data->gpu_kernel = NULL;
src/wshfl.c:#ifdef USE_CUDA
src/wshfl.c:	if (gpu_flag) {
src/wshfl.c:		data->gpu_kernel = md_gpu_move(DIMS, repmat_kernel_dims, repmat_kernel, CFL_SIZE);
src/wshfl.c:	(void)gpu_flag;
src/wshfl.c:		OPT_SET(    'g', &bart_use_gpu,         "Use GPU."),
src/wshfl.c:	num_init_gpu_support();
src/wshfl.c:		const struct linop_s* Knc = linop_kern_create(bart_use_gpu, reorder_dims, reorder, phi_dims, phi, kernel_dims, kernel, table_dims);
src/wshfl.c:	const struct linop_s* K = linop_kern_create(bart_use_gpu, reorder_dims, reorder, phi_dims, phi, kernel_dims, kernel, single_channel_table_dims);
src/wshfl.c:		const struct linop_s* K      = linop_kern_create(bart_use_gpu, reorder_dims, reorder, phi_dims, phi, kernel_dims, kernel, single_channel_table_dims);
src/wshfl.c:	opt_reg_configure(DIMS, coeff_dims, &ropts, thresh_ops, trafos, blksize, 1, "dau2", bart_use_gpu);
src/wshfl.c:#ifdef USE_CUDA
src/wshfl.c:			eval = bart_use_gpu ? estimate_maxeigenval_gpu(A_sc->normal) : estimate_maxeigenval(A_sc->normal);
src/wshfl.c:	lsqr_conf.it_gpu = bart_use_gpu;
src/calib/calib.c:#ifdef USE_CUDA
src/calib/calib.c:void eigenmaps(const long out_dims[DIMS], complex float* optr, complex float* eptr, const complex float* imgcov2, const long msk_dims[3], const bool* msk, bool orthiter, int num_orthiter, bool ecal_usegpu)
src/calib/calib.c:#ifdef USE_CUDA
src/calib/calib.c:	if (ecal_usegpu) {
src/calib/calib.c:		//FIXME cuda version should be able to return sensitivities for a subset of image-space points
src/calib/calib.c:	assert(!ecal_usegpu);
src/calib/calib.c:	eigenmaps(out_dims, out_data, emaps, imgcov2, msk_dims, msk, conf->orthiter, conf->num_orthiter, conf->usegpu);
src/calib/calib.c:	.usegpu = false,
src/calib/calib.c:#ifdef USE_CUDA
src/calib/calib.c:	if (conf->usegpu)
src/calib/calib.c:		data_tmp = md_gpu_move(DIMS, calreg_dims, data, CFL_SIZE);
src/calib/calib.h:	_Bool usegpu;
src/calib/calib.h:extern void eigenmaps(const long out_dims[DIMS], _Complex float* out_data, _Complex float* eptr, const _Complex float* imgcov, const long msk_dims[3], const _Bool* msk, _Bool orthiter, int num_orthiter, _Bool usegpu);
src/calib/calibcu.cu:#include <cuda.h>
src/calib/calibcu.cu:#include <cuda_runtime.h>
src/calib/calibcu.cu:#include "num/gpuops.h"
src/calib/calibcu.cu:	printf("CUDA Pointwise Eigendecomposition...\n");
src/calib/calibcu.cu:	cuFloatComplex* optr_device = (cuFloatComplex*)md_alloc_gpu(5, dims, sizeof(cuFloatComplex));
src/calib/calibcu.cu:	cuFloatComplex* imgcov2_device = (cuFloatComplex*)md_alloc_gpu(5, imgcov2_dims, sizeof(cuFloatComplex));
src/calib/calibcu.cu:	cuFloatComplex* imgcov2_device_filled = (cuFloatComplex*)md_alloc_gpu(5, imgcov2_df_dims, sizeof(cuFloatComplex));
src/calib/calibcu.cu:	cuFloatComplex* eptr_device = (cuFloatComplex*)md_alloc_gpu(5, eptr_dims, sizeof(cuFloatComplex));
src/calib/calibcu.cu:	struct cudaDeviceProp mycudaDeviceProperties;
src/calib/calibcu.cu:	cudaGetDeviceProperties(&mycudaDeviceProperties, 0);
src/calib/calibcu.cu:	const int maxSharedMemPerBlock = mycudaDeviceProperties.sharedMemPerBlock;
src/calib/calibcu.cu:	const int maxThreadsPerBlock = mycudaDeviceProperties.maxThreadsPerBlock;
src/calib/calibcu.cu:	const int maxRegsPerBlock = mycudaDeviceProperties.regsPerBlock;
src/calib/calibcu.cu:	const int maxCmemPerBlock = mycudaDeviceProperties.totalConstMem;  
src/calib/calibcu.cu:	eigenmapscu_kern<<<blocks, threads, sharedMem, cuda_get_stream()>>>(imgcov2_device_filled, imgcov2_device, optr_device, eptr_device, num_orthiter, x, y, z, N, M);
src/calib/calibcu.cu:	cudaDeviceSynchronize();
src/calib/calibcu.cu:	cudaError_t cu_error = cudaGetLastError();
src/calib/calibcu.cu:	if (cu_error != cudaSuccess) {
src/calib/calibcu.cu:		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cu_error));
src/nn/nn_ops.c:#ifdef USE_CUDA
src/nn/nn_ops.c:#include "num/gpuops.h"
src/nn/nn_ops.c:#ifdef USE_CUDA
src/nn/nn_ops.c:	assert((cuda_ondevice(dst) == cuda_ondevice(src)));
src/nn/nn_ops.c:#ifdef USE_CUDA
src/nn/nn_ops.c:	assert((cuda_ondevice(dst) == cuda_ondevice(src)));
src/nn/losses.c:#ifdef USE_CUDA
src/nn/losses.c:#include "num/gpuops.h"
src/nn/losses.c:#ifdef USE_CUDA
src/nn/losses.c:	assert((cuda_ondevice(dst) == cuda_ondevice(src)));
src/nn/losses.c:#ifdef USE_CUDA
src/nn/losses.c:	assert((cuda_ondevice(dst) == cuda_ondevice(src1)) && (cuda_ondevice(src1) == cuda_ondevice(src2)));
src/nn/losses.c:#ifdef USE_CUDA
src/nn/losses.c:	assert((cuda_ondevice(dst) == cuda_ondevice(src_pred)) && (cuda_ondevice(src_pred) == cuda_ondevice(src_true)));
src/nn/losses.c:	// will be initialized later, to transparently support GPU
src/nn/losses.c:#ifdef USE_CUDA
src/nn/losses.c:	assert(   (cuda_ondevice(dst) == cuda_ondevice(src_pred))
src/nn/losses.c:	       && (cuda_ondevice(src_pred) == cuda_ondevice(src_true)));
src/nn/losses.c:#ifdef USE_CUDA
src/nn/losses.c:	assert(cuda_ondevice(dst) == cuda_ondevice(src));
src/nn/losses.c:#ifdef USE_CUDA
src/nn/losses.c:	assert((cuda_ondevice(dst) == cuda_ondevice(src_pred)) && (cuda_ondevice(src_pred) == cuda_ondevice(src_true)));
src/nn/weights.h:void move_gpu_nn_weights(nn_weights_t weights);
src/nn/weights.h:_Bool nn_weights_on_gpu(nn_weights_t weights);
src/nn/batchnorm.c:#ifdef USE_CUDA
src/nn/batchnorm.c:#include "num/gpuops.h"
src/nn/batchnorm.c:#ifdef USE_CUDA
src/nn/batchnorm.c:	assert((cuda_ondevice(mean) == cuda_ondevice(src)) && (cuda_ondevice(var) == cuda_ondevice(src)));
src/nn/batchnorm.c:	// will be initialized later, to transparently support GPU
src/nn/batchnorm.c:#ifdef USE_CUDA //FIXME: Optimize zsub2 for these strides
src/nn/batchnorm.c:	if (cuda_ondevice(src)) {
src/nn/batchnorm.c:	// will be initialized later, to transparently support GPU
src/nn/tf_wrapper.c:#ifdef USE_CUDA
src/nn/tf_wrapper.c:#include "num/gpuops.h"
src/nn/tf_wrapper.c:Python code to generate session config for selecting GPUs (https://github.com/tensorflow/tensorflow/issues/13853):
src/nn/tf_wrapper.c:config = tf.compat.v1.ConfigProto(allow_soft_placement=True, device_count = {'GPU': 0})
src/nn/tf_wrapper.c:config.gpu_options.allow_growth=True
src/nn/tf_wrapper.c:print("uint8_t no_gpu[] = { "+ str(len(result))+", "+ ", ".join(result)+" };")
src/nn/tf_wrapper.c:    config.gpu_options.allow_growth=True
src/nn/tf_wrapper.c:    config.gpu_options.visible_device_list=str(i)
src/nn/tf_wrapper.c:    print('uint8_t gpu_{}[] = {{ '.format(i)+ str(len(result))+", "+ ", ".join(result)+" };")
src/nn/tf_wrapper.c:	uint8_t no_gpu[] = { 19, 0xa, 0x7, 0xa, 0x3, 0x47, 0x50, 0x55, 0x10, 0x0, 0x10, threads, 0x28, threads, 0x32, 0x2, 0x20, 0x1, 0x38, 0x1 };
src/nn/tf_wrapper.c:	uint8_t gpu_0[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x30, 0x38, 0x1 };
src/nn/tf_wrapper.c:	uint8_t gpu_1[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x31, 0x38, 0x1 };
src/nn/tf_wrapper.c:	uint8_t gpu_2[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x32, 0x38, 0x1 };
src/nn/tf_wrapper.c:	uint8_t gpu_3[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x33, 0x38, 0x1 };
src/nn/tf_wrapper.c:	uint8_t gpu_4[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x34, 0x38, 0x1 };
src/nn/tf_wrapper.c:	uint8_t gpu_5[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x35, 0x38, 0x1 };
src/nn/tf_wrapper.c:	uint8_t gpu_6[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x36, 0x38, 0x1 };
src/nn/tf_wrapper.c:	uint8_t gpu_7[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x37, 0x38, 0x1 };
src/nn/tf_wrapper.c:	uint8_t gpu_8[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x38, 0x38, 0x1 };
src/nn/tf_wrapper.c:	uint8_t gpu_9[] = { 13, 0x10, threads, 0x28, threads, 0x32, 0x5, 0x20, 0x1, 0x2a, 0x1, 0x39, 0x38, 0x1 };
src/nn/tf_wrapper.c:	uint8_t gpu_10[] = { 14, 0x10, threads, 0x28, threads, 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x30, 0x38, 0x1 };
src/nn/tf_wrapper.c:	uint8_t gpu_11[] = { 14, 0x10, threads, 0x28, threads, 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x31, 0x38, 0x1 };
src/nn/tf_wrapper.c:	uint8_t gpu_12[] = { 14, 0x10, threads, 0x28, threads, 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x32, 0x38, 0x1 };
src/nn/tf_wrapper.c:	uint8_t gpu_13[] = { 14, 0x10, threads, 0x28, threads, 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x33, 0x38, 0x1 };
src/nn/tf_wrapper.c:	uint8_t gpu_14[] = { 14, 0x10, threads, 0x28, threads, 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x34, 0x38, 0x1 };
src/nn/tf_wrapper.c:	uint8_t gpu_15[] = { 14, 0x10, threads, 0x28, threads, 0x32, 0x6, 0x20, 0x1, 0x2a, 0x2, 0x31, 0x35, 0x38, 0x1 };
src/nn/tf_wrapper.c:	uint8_t* gpu[] = { gpu_0, gpu_1, gpu_2, gpu_3, gpu_4, gpu_5, gpu_6, gpu_7, gpu_8, gpu_9, gpu_10, gpu_11, gpu_12, gpu_13, gpu_14, gpu_15 };
src/nn/tf_wrapper.c:	uint8_t* config = no_gpu;
src/nn/tf_wrapper.c:#ifdef USE_CUDA
src/nn/tf_wrapper.c:	if (-1 != cuda_get_device_id())
src/nn/tf_wrapper.c:		config = gpu[cuda_get_device_id()];
src/nn/tf_wrapper.c:	(void)gpu;
src/nn/chain.c:nn_t nn_stack_multigpu_F(int N , nn_t x[N], int stack_dim)
src/nn/rbf.c:#ifdef USE_CUDA
src/nn/rbf.c:#include "num/gpuops.h"
src/nn/chain.h:extern nn_t nn_stack_multigpu_F(int N , nn_t x[N], int stack_dim);
src/nn/weights.c:#ifdef USE_CUDA
src/nn/weights.c:#include "num/gpuops.h"
src/nn/weights.c: * Move weights to gpu
src/nn/weights.c:void move_gpu_nn_weights(nn_weights_t weights){
src/nn/weights.c:#ifdef USE_CUDA
src/nn/weights.c:		complex float* tmp = md_alloc_gpu(iov->N, iov->dims, iov->size);
src/nn/weights.c:	error("Compiled without gpu support!\n");
src/nn/weights.c: * Check if weights are copied to gpu
src/nn/weights.c: * @returns boolean if weights are copied to gpu
src/nn/weights.c:bool nn_weights_on_gpu(nn_weights_t weights)
src/nn/weights.c:#ifdef USE_CUDA
src/nn/weights.c:	return cuda_ondevice(weights->tensors[0]);
src/nn/activation.c:#ifdef USE_CUDA
src/nn/activation.c:#include "num/gpuops.h"
src/nn/activation.c:	complex float* tmp_gpu = md_alloc_sameplace(d->N, d->dom->dims, d->dom->size, dst);
src/nn/activation.c:	md_copy2(d->N, d->dom->dims, d->dom->strs, tmp_gpu, d->batchdom->strs, max, CFL_SIZE);
src/nn/activation.c:	md_zsub(d->N, d->dom->dims, tmp_real, tmp_real, tmp_gpu);
src/nn/activation.c:	md_copy2(d->N, d->dom->dims, d->dom->strs, tmp_gpu, d->batchdom->strs, scale, CFL_SIZE);
src/nn/activation.c:	md_zdiv(d->N, d->dom->dims, d->tmp, tmp_exp, tmp_gpu);
src/nn/activation.c:	md_free(tmp_gpu);
src/nn/activation.c:	complex float* tmp_gpu = md_alloc_sameplace(d->N, d->dom->dims, d->dom->size, dst);
src/nn/activation.c:	md_copy2(d->N, d->dom->dims, d->dom->strs, tmp_gpu, d->batchdom->strs, tmp1, CFL_SIZE);
src/nn/activation.c:	md_ztenmul(d->N, d->dom->dims, tmp2, d->dom->dims, d->tmp, d->dom->dims, tmp_gpu);
src/nn/activation.c:	md_free(tmp_gpu);
src/pics.c:	bool gpu_gridding = false;
src/pics.c:		OPT_SET('g', &bart_use_gpu, "use GPU"),
src/pics.c:		OPTL_SET(0, "gpu-gridding", &gpu_gridding, "use GPU for gridding"),
src/pics.c:	num_init_gpu_support();
src/pics.c:	conf.gpu = bart_use_gpu;
src/pics.c:	if (conf.gpu)
src/pics.c:		debug_printf(DP_INFO, "GPU reconstruction\n");
src/pics.c:		//for computation of psf on GPU
src/pics.c:#ifdef USE_CUDA
src/pics.c:		if (gpu_gridding) {
src/pics.c:			assert(conf.gpu);
src/pics.c:			traj_tmp = md_gpu_move(DIMS, traj_dims, traj, CFL_SIZE);
src/pics.c:#ifdef USE_CUDA
src/pics.c:		if (gpu_gridding)
src/pics.c:#ifdef USE_CUDA
src/pics.c:	if (conf.gpu && (gpu_gridding || NULL == traj)) {
src/pics.c:		auto tmp = linop_gpu_wrapper((struct linop_s*)forward_op);
src/pics.c:#ifdef USE_CUDA
src/pics.c:		if (conf.gpu) {
src/pics.c:			complex float* gpu_image_truth = md_gpu_move(DIMS, img_dims, image_truth, CFL_SIZE);
src/pics.c:			image_truth = gpu_image_truth;
src/pics.c:	opt_reg_configure(DIMS, img_dims, &ropts, thresh_ops, trafos, llr_blk, shift_mode, wtype_str, conf.gpu);
src/pics.c:#ifdef USE_CUDA
src/pics.c:		if (conf.gpu)
src/estmotion.c:		OPT_SET('g', &bart_use_gpu, "use gpu (if available)"),
src/estmotion.c:	num_init_gpu_support();
src/estmotion.c:#ifdef USE_CUDA
src/estmotion.c:	md_alloc_fun_t my_alloc = bart_use_gpu ? md_alloc_gpu : md_alloc;
src/estmotion.c:	assert(!bart_use_gpu);
src/ecaltwo.c:		OPT_SET('g', &conf.usegpu, "()"),
src/ecaltwo.c:	bart_use_gpu = conf.usegpu;
src/ecaltwo.c:	num_init_gpu_support();
src/nlinvnet.c:		OPTL_SET('g', "gpu", &bart_use_gpu, "run on gpu"),
src/nlinvnet.c:	num_init_gpu_support();
src/nnet.c:#ifdef USE_CUDA
src/nnet.c:#include "num/gpuops.h"
src/nnet.c:		OPTL_SET('g', "gpu", &(bart_use_gpu), "run on gpu"),
src/nnet.c:	config.gpu = bart_use_gpu;
src/nnet.c:	num_init_gpu_support();
src/nlinv.c:		OPT_SET('g', &bart_use_gpu, "use gpu"),
src/nlinv.c:	num_init_gpu_support();
src/nlinv.c:	conf.gpu = bart_use_gpu;
src/misc/bench.c:#include "num/gpuops.h"
src/misc/bench.c:static void bench_sync(bool sync_gpu)
src/misc/bench.c:#ifdef USE_CUDA
src/misc/bench.c:	if (sync_gpu)
src/misc/bench.c:		cuda_sync_stream();
src/misc/bench.c:	(void) sync_gpu;
src/misc/bench.c:void run_bench(long rounds, bool print, bool sync_gpu, bench_f fun)
src/misc/bench.c:	bench_sync(sync_gpu);
src/misc/bench.c:		bench_sync(sync_gpu);
src/misc/nested.h:#if defined(__clang__) && !defined(__CUDACC__)
src/misc/bench.h:void run_bench(long rounds, bool print, bool sync_gpu, bench_f fun);
src/ncalib.c:		OPT_SET('g', &bart_use_gpu, "use gpu"),
src/ncalib.c:	conf.gpu = bart_use_gpu;
src/ncalib.c:	num_init_gpu_support();
src/linops/someops.c:#ifdef USE_CUDA
src/linops/someops.c:#include "num/gpuops.h"
src/linops/someops.c:	// strided copies are more efficient than strided sum (gpu)
src/linops/someops.c:#ifdef USE_CUDA
src/linops/someops.c:	const complex float* mat_gpu;
src/linops/someops.c:	const complex float* mat_gram_gpu;
src/linops/someops.c:#ifdef USE_CUDA
src/linops/someops.c:	if (cuda_ondevice(src)) {
src/linops/someops.c:		if (NULL == data->mat_gpu)
src/linops/someops.c:			data->mat_gpu = md_gpu_move(data->N, data->mat_dims, data->mat, CFL_SIZE);
src/linops/someops.c:		mat = data->mat_gpu;
src/linops/someops.c:#ifdef USE_CUDA
src/linops/someops.c:	if (cuda_ondevice(src)) {
src/linops/someops.c:		if (NULL == data->mat_gpu)
src/linops/someops.c:			data->mat_gpu = md_gpu_move(data->N, data->mat_dims, data->mat, CFL_SIZE);
src/linops/someops.c:		mat = data->mat_gpu;
src/linops/someops.c:#ifdef USE_CUDA
src/linops/someops.c:		if (cuda_ondevice(src)) {
src/linops/someops.c:			if (NULL == data->mat_gram_gpu)
src/linops/someops.c:				data->mat_gram_gpu = md_gpu_move(2 * data->N, data->grm_dims, data->mat_gram, CFL_SIZE);
src/linops/someops.c:			mat_gram = data->mat_gram_gpu;
src/linops/someops.c:#ifdef USE_CUDA
src/linops/someops.c:	md_free(data->mat_gpu);
src/linops/someops.c:	md_free(data->mat_gram_gpu);
src/linops/someops.c:#ifdef USE_CUDA
src/linops/someops.c:	data->mat_gpu = NULL;
src/linops/someops.c:	data->mat_gram_gpu = NULL;
src/linops/someops.c:#ifdef USE_CUDA
src/linops/someops.c:	if (cuda_ondevice(out)) {
src/linops/someops.c:#ifdef USE_CUDA
src/linops/someops.c:	if (cuda_ondevice(out)) {
src/linops/linop.c:struct linop_s* linop_gpu_wrapper(struct linop_s* op)
src/linops/linop.c:	op2->forward = operator_gpu_wrapper(op->forward);
src/linops/linop.c:	op2->adjoint = operator_gpu_wrapper(op->adjoint);
src/linops/linop.c:	op2->normal = (NULL == op->normal) ? NULL : operator_gpu_wrapper(op->normal);
src/linops/linop.h:extern struct linop_s* linop_gpu_wrapper(struct linop_s* op);
src/tensorflow.c:		OPT_SET('g', &bart_use_gpu, "Use gpu"),
src/tensorflow.c:	num_init_gpu_support();
src/sqpics.c:		OPT_SET('g', &bart_use_gpu, "use GPU"),
src/sqpics.c:	num_init_gpu_support();
src/sqpics.c:	if (bart_use_gpu)
src/sqpics.c:		debug_printf(DP_INFO, "GPU reconstruction\n");
src/sqpics.c:	if (use_gpu)
src/sqpics.c:#ifdef USE_CUDA
src/sqpics.c:		sqpics_recon2_gpu(&conf, max_dims, image, forward_op, pat_dims, pattern,
src/noir/recon2.c:#ifdef USE_CUDA
src/noir/recon2.c:#include "num/gpuops.h"
src/noir/recon2.c:	.gpu = false,
src/noir/recon2.c:#ifdef USE_CUDA
src/noir/recon2.c:	if((conf->gpu) && !cuda_ondevice(data)) {
src/noir/recon2.c:		complex float* tmp_data = md_alloc_gpu(N, dat_dims, CFL_SIZE);
src/noir/recon2.c:	if(conf->gpu)
src/noir/recon2.c:		error("Compiled without GPU support!");
src/noir/recon2.c:		md_copy(DIMS, col_dims, sens, tmp, CFL_SIZE);	// needed for GPU
src/noir/recon2.c:#ifdef USE_CUDA
src/noir/recon2.c:	md_alloc_fun_t my_alloc = conf->gpu ? md_alloc_gpu : md_alloc;
src/noir/recon2.c:	assert(!conf->gpu);
src/noir/recon.c:	md_copy(DIMS, coil_dims, sens, x + skip, CFL_SIZE);	// needed for GPU
src/noir/model_net.c:static bool multigpu = false;
src/noir/model_net.c:void model_net_activate_multigpu(void)
src/noir/model_net.c:	multigpu = true;
src/noir/model_net.c:void model_net_deactivate_multigpu(void)
src/noir/model_net.c:	multigpu = false;
src/noir/model_net.c:	const struct nlop_s* ret = nlop_stack_multiple_F(model->Nb, nlops, 4, istack_dims, 1, ostack_dims, true, multigpu);
src/noir/model_net.c:	const struct nlop_s* ret = nlop_stack_multiple_F(model->Nb, nlops, 4, istack_dims, 1, ostack_dims, true, multigpu);
src/noir/model_net.c:	const struct nlop_s* ret = nlop_stack_multiple_F(B, nlops, 4, istack_dims, 1, ostack_dims, true, multigpu);
src/noir/model_net.c:	const struct nlop_s* ret = nlop_stack_multiple_F(model->Nb, nlops, 2, istack_dims, 1, ostack_dims, true, multigpu);
src/noir/model_net.c:	const struct nlop_s* ret = nlop_stack_multiple_F(model->Nb, nlops, 3, istack_dims, 1, ostack_dims, true, multigpu);
src/noir/model_net.c:	const struct nlop_s* ret = nlop_stack_multiple_F(model->Nb, nlops, 1, istack_dims, 1, ostack_dims, true, multigpu);
src/noir/model_net.c:	const struct nlop_s* ret = nlop_stack_multiple_F(model->Nb, nlops, 2, istack_dims, 1, ostack_dims, true, multigpu);
src/noir/model_net.c:	const struct nlop_s* ret = nlop_stack_multiple_F(model->Nb, nlops, 4, istack_dims, 1, ostack_dims, true, multigpu);
src/noir/recon2.h:	_Bool gpu;
src/noir/utils.c:#ifdef USE_CUDA
src/noir/model_net.h:extern void model_net_activate_multigpu(void);
src/noir/model_net.h:extern void model_net_deactivate_multigpu(void);
src/nufft.c:		OPT_SET('g', &bart_use_gpu, "GPU"),
src/nufft.c:	num_init_gpu_support();
src/nufft.c:#ifdef USE_CUDA
src/nufft.c:			if (bart_use_gpu && !precond && !dft) {
src/nufft.c:				complex float* traj_gpu = md_gpu_move(DIMS, traj_dims, traj, CFL_SIZE);
src/nufft.c:				auto tmp = nufft_create2(DIMS, ksp_dims, coilim_dims, traj_dims, traj_gpu, pattern_dims, pattern, basis_dims, basis, conf);
src/nufft.c:				nufft_op = linop_gpu_wrapper((struct linop_s*)tmp);
src/nufft.c:				md_free(traj_gpu);
src/nufft.c:			lsqr_conf.it_gpu = bart_use_gpu;
src/nufft.c:		if (bart_use_gpu) {
src/nufft.c:			nufft_op = linop_gpu_wrapper((struct linop_s*)tmp);
src/rtnlinv.c:		OPT_SET('g', &bart_use_gpu, "use gpu"),
src/rtnlinv.c:	num_init_gpu_support();
src/rtnlinv.c:#ifdef USE_CUDA
src/rtnlinv.c:		if (bart_use_gpu) {
src/rtnlinv.c:			complex float* kgrid1_gpu = md_alloc_gpu(DIMS, kgrid1_dims, CFL_SIZE);
src/rtnlinv.c:			md_copy(DIMS, kgrid1_dims, kgrid1_gpu, kgrid1, CFL_SIZE);
src/rtnlinv.c:			noir_recon(&conf, sens1_dims, img1, sens1, ksens1, ref, pattern1, mask, kgrid1_gpu);
src/rtnlinv.c:			md_free(kgrid1_gpu);
src/wave.c:#ifdef USE_CUDA
src/wave.c:#include "num/gpuops.h"
src/wave.c:		OPT_SET(   'g', &bart_use_gpu,      "use GPU"),
src/wave.c:	num_init_gpu_support();
src/wave.c:#ifdef USE_CUDA
src/wave.c:		eval = bart_use_gpu ? estimate_maxeigenval_gpu(A->normal) : estimate_maxeigenval(A->normal);
src/wave.c:	lsqr_conf.it_gpu = bart_use_gpu;
src/nlops/tenmul.c:#ifdef USE_CUDA
src/nlops/tenmul.c:#include "num/gpuops.h"
src/nlops/tenmul.c:#ifdef USE_CUDA
src/nlops/tenmul.c:	assert((cuda_ondevice(dst) == cuda_ondevice(src1)) && (cuda_ondevice(src1) == cuda_ondevice(src2)));
src/nlops/someops.c:#ifdef USE_CUDA
src/nlops/someops.c:#include "num/gpuops.h"
src/nlops/someops.c:	#ifdef USE_CUDA
src/nlops/someops.c:	assert((cuda_ondevice(dst) == cuda_ondevice(src1)) && (cuda_ondevice(src1) == cuda_ondevice(src2)));
src/nlops/someops.c:#ifdef USE_CUDA
src/nlops/someops.c:	if (cuda_ondevice(dst)) {
src/nlops/stack.c:#ifdef USE_CUDA
src/nlops/stack.c:#include "num/gpuops.h"
src/nlops/stack.c:#ifndef USE_CUDA
src/nlops/stack.c:	if (-1 < cuda_get_device_id())
src/nlops/stack.c:		return MIN(streams, cuda_set_stream_level());
src/nlops/stack.c:	if (-1 < cuda_get_device_id())
src/nlops/stack.c:		return cuda_get_stream_id();
src/nlops/stack.c:const struct nlop_s* nlop_stack_multigpu_create_F(int N, const struct nlop_s* nlops[N], int II, int in_stack_dim[II], int OO, int out_stack_dim[OO])
src/nlops/mri_ops.h:void mri_ops_activate_multigpu(void);
src/nlops/mri_ops.h:void mri_ops_deactivate_multigpu(void);
src/nlops/conv.c:#ifdef USE_CUDA
src/nlops/conv.c:#include "num/gpuops.h"
src/nlops/conv.c:#ifdef USE_CUDA
src/nlops/conv.c:	assert((cuda_ondevice(dst) == cuda_ondevice(src1)) && (cuda_ondevice(src1) == cuda_ondevice(src2)));
src/nlops/stack.h:extern const struct nlop_s* nlop_stack_multigpu_create_F(int N, const struct nlop_s* nlops[__VLA(N)], int II, int in_stack_dim[__VLA(II)], int OO, int out_stack_dim[__VLA(OO)]);
src/nlops/nlop.h:extern const struct nlop_s* nlop_gpu_wrapper(const struct nlop_s* op);
src/nlops/nlop.h:extern const struct nlop_s* nlop_gpu_wrapper_F(const struct nlop_s* op);
src/nlops/chain.c:struct nlop_s* nlop_stack_multiple_F(int N, const struct nlop_s* nlops[N], int II, int in_stack_dim[II], int OO, int out_stack_dim[OO], bool container, bool multigpu)
src/nlops/chain.c:	if (multigpu)
src/nlops/chain.c:		return (struct nlop_s*)nlop_stack_multigpu_create_F(N, nlops, II, in_stack_dim, OO, out_stack_dim);
src/nlops/chain.c:	assert(!multigpu);
src/nlops/chain.h:extern struct nlop_s* nlop_stack_multiple_F(int N, const struct nlop_s* nlops[N], int II, int in_stack_dim[II], int OO, int out_stack_dim[OO], _Bool container, _Bool multigpu);
src/nlops/mri_ops.c:static bool multigpu = false;
src/nlops/mri_ops.c:void mri_ops_activate_multigpu(void)
src/nlops/mri_ops.c:	multigpu = true;
src/nlops/mri_ops.c:void mri_ops_deactivate_multigpu(void)
src/nlops/mri_ops.c:	multigpu = false;
src/nlops/mri_ops.c:	return nlop_stack_multiple_F(Nb, nlops, 3, istack_dim, 1, ostack_dim, true , multigpu);
src/nlops/mri_ops.c:	return nlop_stack_multiple_F(Nb, nlops, (NULL == models[0]->nufft) ? 2 : 3, istack_dim, output_psf ? 2 : 1, ostack_dim, true , multigpu);
src/nlops/mri_ops.c:	return nlop_stack_multiple_F(Nb, nlops, 1, istack_dim, 1, ostack_dim, true , multigpu);
src/nlops/mri_ops.c:	return nlop_stack_multiple_F(Nb, nlops, 2, istack_dim, 1, ostack_dim, true , multigpu);
src/nlops/mri_ops.c:		result = nlop_stack_multiple_F(Nb, nlops, 1, istack_dim, 1, ostack_dim, true , multigpu);
src/nlops/mri_ops.c:	return nlop_stack_multiple_F(Nb, nlops, 3, istack_dim, 1, ostack_dim, true , multigpu);
src/nlops/mri_ops.c:	return nlop_stack_multiple_F(Nb, nlops, 4, istack_dim, 1, ostack_dim, true , multigpu);
src/nlops/mri_ops.c:	return nlop_stack_multiple_F(Nb, nlops, 2, istack_dim, 1, ostack_dim, true , multigpu);
src/nlops/checkpointing.c:#ifdef USE_CUDA
src/nlops/checkpointing.c:#include "num/gpuops.h"
src/nlops/checkpointing.c: * Checkpointing can reduce memory consumption drastically and the overhead of recomputing the forward operator is compensated by reduced swapping from gpu to cpu memory.
src/nlops/checkpointing.c: * Checkpointing can reduce memory consumption drastically and the overhead of recomputing the forward operator is compensated by reduced swapping from gpu to cpu memory.
src/nlops/nlop.c:const struct nlop_s* nlop_gpu_wrapper(const struct nlop_s* op) 
src/nlops/nlop.c:#ifdef USE_CUDA
src/nlops/nlop.c:	n->op = operator_gpu_wrapper(op->op);
src/nlops/nlop.c:			(*nder)[ii][oo] = linop_gpu_wrapper((struct linop_s*)(*der)[ii][oo]);
src/nlops/nlop.c:const struct nlop_s* nlop_gpu_wrapper_F(const struct nlop_s* op) 
src/nlops/nlop.c:	auto result = nlop_gpu_wrapper(op);
src/nlops/nlop_jacobian.c:#ifdef USE_CUDA
src/nlops/nlop_jacobian.c:#include "num/gpuops.h"
src/affinereg.c:		OPT_SET('g', &bart_use_gpu, "use gpu (if available)"),
src/affinereg.c:	num_init_gpu_support();
src/affinereg.c:	affine_reg(bart_use_gpu, false, affine, trafo, rdims, ref_ptr, msk_ref_ptr, mdims, mov_ptr, msk_mov_ptr, 3, sigmas, factors);

```
