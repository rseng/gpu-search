# https://github.com/andreasmang/claire

```console
deps/3rdparty/math_func.cu:This file is part of CUDA Cubic B-Spline Interpolation (CI).
deps/3rdparty/math_func.cu:   GPU Prefilter for Accurate Cubic B-Spline Interpolation, 
deps/3rdparty/math_func.cu:   http://dannyruijters.nl/docs/cudaPrefilter3.pdf
deps/3rdparty/math_func.cu:   Efficient GPU-Based Texture Interpolation using Uniform B-Splines,
deps/3rdparty/math_func.cu:#ifndef _MATH_FUNC_CUDA_H_
deps/3rdparty/math_func.cu:#define _MATH_FUNC_CUDA_H_
deps/3rdparty/math_func.cu:#endif  //_MATH_FUNC_CUDA_H_
deps/3rdparty/version.cu:This file is part of CUDA Cubic B-Spline Interpolation (CI).
deps/3rdparty/version.cu:   GPU Prefilter for Accurate Cubic B-Spline Interpolation, 
deps/3rdparty/version.cu:   http://dannyruijters.nl/docs/cudaPrefilter3.pdf
deps/3rdparty/version.cu:   Efficient GPU-Based Texture Interpolation using Uniform B-Splines,
deps/3rdparty/version.cu:const char* ciVersion = "CUDA Cubic B-Spline Interpolation (CI) Version 1.2";
deps/3rdparty/lagrange_kernel.cu:This file is part of CUDA Cubic B-Spline Interpolation (CI).
deps/3rdparty/lagrange_kernel.cu:   GPU Prefilter for Accurate Cubic B-Spline Interpolation, 
deps/3rdparty/lagrange_kernel.cu:   http://dannyruijters.nl/docs/cudaPrefilter3.pdf
deps/3rdparty/lagrange_kernel.cu:   Efficient GPU-Based Texture Interpolation using Uniform B-Splines,
deps/3rdparty/lagrange_kernel.cu:#ifndef _CUDA_LAGRANGE_H_
deps/3rdparty/lagrange_kernel.cu:#define _CUDA_LAGRANGE_H_
deps/3rdparty/lagrange_kernel.cu:#include "cuda_helper_math.h"
deps/3rdparty/lagrange_kernel.cu:#endif // _CUDA_LAGRANGE_H_
deps/3rdparty/bspline_kernel.cu:This file is part of CUDA Cubic B-Spline Interpolation (CI).
deps/3rdparty/bspline_kernel.cu:   GPU Prefilter for Accurate Cubic B-Spline Interpolation, 
deps/3rdparty/bspline_kernel.cu:   http://dannyruijters.nl/docs/cudaPrefilter3.pdf
deps/3rdparty/bspline_kernel.cu:   Efficient GPU-Based Texture Interpolation using Uniform B-Splines,
deps/3rdparty/bspline_kernel.cu:#ifndef _CUDA_BSPLINE_H_
deps/3rdparty/bspline_kernel.cu:#define _CUDA_BSPLINE_H_
deps/3rdparty/bspline_kernel.cu:#include "cuda_helper_math.h"
deps/3rdparty/bspline_kernel.cu:#endif // _CUDA_BSPLINE_H_
deps/3rdparty/cuda_helper_math.h: * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
deps/3rdparty/cuda_helper_math.h: * Please refer to the NVIDIA end user license agreement (EULA) associated
deps/3rdparty/cuda_helper_math.h: *  (float3, float4 etc.) since these are not provided as standard by CUDA.
deps/3rdparty/cuda_helper_math.h:#include "cuda_runtime.h"
deps/3rdparty/cuda_helper_math.h:#ifndef __CUDACC__
deps/3rdparty/cuda_helper_math.h:// host implementations of CUDA functions
deps/makefile:BUILD_GPU     = yes
deps/makefile:WITH_CUDA_MPI = yes
deps/makefile:GPU_VERSION =
deps/makefile:ifeq ($(WITH_CUDA_MPI), yes)
deps/makefile:	PETSC_OPTIONS += -use-gpu-aware-mpi=1
deps/makefile:	PETSC_OPTIONS += -use-gpu-aware-mpi=0
deps/makefile:ifeq ($(BUILD_GPU), yes)
deps/makefile:	PETSC_ARCH = gpu
deps/makefile:	PETSC_OPTIONS += --with-cuda=1 
deps/makefile:	PETSC_OPTIONS += --CUDAOPTFLAGS='-O3'
deps/makefile:	PETSC_OPTIONS += --with-cudac='$(NVCC) -ccbin=$(CXX)'
deps/makefile:ifdef GPU_VERSION
deps/makefile:	PETSC_OPTIONS += --with-cuda-gencodearch=$(GPU_VERSION)
deps/makefile:	PETSC_OPTIONS += --with-cuda=0
deps/makefile:	@echo "build with CUDA-aware MPI: $(WITH_CUDA_MPI)"
deps/makefile:	@echo "build for GPU:             $(BUILD_GPU)"
doc/README-REFERENCES.md:## Parallel GPU Implementation
doc/README-REFERENCES.md:* M. Brunn, N. Himthani, G. Biros, M. Mehl & A. Mang. *Multi-node multi-GPU diffeomorphic image registration for large-scale imaging problems*. Proc ACM/IEEE Conference on Supercomputing 2020. [[arxiv](https://arxiv.org/abs/2008.12820), [ieee](https://doi.ieeecomputersociety.org/10.1109/SC41405.2020.00042)].
doc/README-REFERENCES.md:* M. Brunn, N. Himthani, G. Biros, M. Mehl & A. Mang. *Fast GPU 3D diffeomorphic image registration*. Journal of Parallel and Distributed Computing, 149:149-162, 2021 [[arxiv](https://arxiv.org/abs/2004.08893), [jpdc](https://doi.org/10.1016/j.jpdc.2020.11.006)].
doc/paper/paper.bib:	title = {Multi-node multi-{GPU} diffeomorphic image registration for large-scale imaging problems},
doc/paper/paper.bib:	title = {Fast {GPU} {3D} diffeomorphic image registration},
doc/paper/paper.md:  - GPUs
doc/paper/paper.md:[`CLAIRE`](https://andreasmang.github.io/claire) [@claire-web] is a computational framework for **C**onstrained **LA**rge deformation diffeomorphic **I**mage **RE**gistration [@Mang:2019a]. It supports highly-optimized, parallel computational kernels for (multi-node) CPU [@Mang:2016a; @Gholami:2017a; @Mang:2019a] and (multi-node multi-)GPU architectures [@Brunn:2020a; @Brunn:2021a]. `CLAIRE` uses MPI for distributed-memory parallelism and can be scaled up to thousands of cores [@Mang:2019a; @Mang:2016a] and GPU devices [@Brunn:2020a]. The multi-GPU implementation uses device direct communication. The computational kernels are interpolation for semi-Lagrangian time integration, and a mixture of high-order finite difference operators and Fast-Fourier-Transforms (FFTs) for differentiation. `CLAIRE` uses a Newton--Krylov solver for numerical optimization [@Mang:2015a; @Mang:2017a]. It features various schemes for regularization of the control problem [@Mang:2016a] and different similarity measures. `CLAIRE` implements different preconditioners for the reduced space Hessian [@Brunn:2020a; @Mang:2019a] to optimize computational throughput and enable fast convergence. It uses `PETSc` [@petsc-web] for scalable and efficient linear algebra operations and solvers and `TAO` [@petsc-web; @Munson:2015a] for numerical optimization. `CLAIRE` can be downloaded at <https://github.com/andreasmang/claire>.
doc/paper/paper.md:Diffeomorphic image registration is an indispensable tool in medical image analysis [@Sotiras:2013a]. Computing diffeomorphisms that map one image to another is expensive. Deformable image registration is an infinite-dimensional problem that upon discretization leads to nonlinear optimality systems with millions or even billions of unknowns. For example, registering two typical medical imaging datasets of size $256^3$ necessitates solving for about 50 million unknowns (in our formulation). Additional complications are the ill-posedness and non-linearty of this inverse problem [@Fischer:2008a]. Consequently, image registration can take several minutes on multi-core high-end CPUs. Many of the available methods reduce the number of unknowns by using coarser resolutions either through parameterization or by solving the problem on coarser grids; they use simplified algorithms and deliver subpar registration quality. In the age of big data, clinical population studies that require thousands of registrations are incresingly common, and execution times of individual registrations become more critical. We provide technology that allows solving registration problems for clinical datasets in seconds. In addition, we have made available to the public a software that works on multi-node, multi-GPU architectures [@Brunn:2020a; @Brunn:2021a] that allows the registration of large-scale microscopic imaging data such as CLARITY imaging [@Tomer:2014a; @Kutten:2017a].
doc/paper/paper.md:`CLAIRE` can be used to register images of $2048^3$ (25 B unknowns) on 64 nodes with 256 GPUs on TACC’s Longhorn system [@Brunn:2020a]. `CLAIRE` has been used for the registration of high resolution CLARITY imaging data [@Brunn:2020a]. The GPU version of `CLAIRE` can solve clinically relevant problems (50 M unknowns) in approximately 5 seconds on a single NVIDIA Tesla V100 [@Brunn:2020a]. `CLAIRE` has also been applied to hundreds of images in brain tumor imaging studies [@Bakas:2018a; @Mang:2017c; @Scheufele:2021a], and has been integrated with models for biophysics inversion [@Mang:2018a; @Mang:2020a; @Scheufele:2020a; @Scheufele:2019a; @Scheufele:2021a; @Subramanian:2020b] and Alzheimer's disease progression [@Scheufele:2020c]. `CLAIRE` uses highly optimized computational kernels and effective, state-of-the-art algorithms for time integration and numerical optimization. Our most recent version of `CLAIRE` features a Python interface to assist users in their applications.
doc/paper/paper.md:This work was partly supported by the National Science Foundation (DMS-1854853, DMS-2009923, DMS-2012825, CCF-1817048, CCF-1725743), the NVIDIA Corporation (NVIDIA GPU Grant Program), the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy-EXC 2075-390740016, by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, Applied Mathematics program under Award Number DE-SC0019393; by the U.S. Air Force Office of Scientific Research award FA9550-17-1-0190; by the Portugal Foundation for Science and Technology and the UT Austin-Portugal program, and by NIH award 5R01NS042645-11A1. Any opinions, findings, and conclusions or recommendations expressed herein are those of the authors and do not necessarily reflect the views of the DFG, AFOSR, DOE, NIH, and NSF. Computing time on the Texas Advanced Computing Centers’ (TACC) systems was provided by an allocation from TACC and the NSF. This work was completed in part with resources provided by the Research Computing Data Core at the University of Houston.
doc/README-RUNME.md:We provide **several examples** for executing these binaries in the [doc/examples](https://github.com/andreasmang/claire/tree/gpu/examples) subfolder. We briefly explain these examples below.
doc/README-RUNME.md:If you are trying to execute CLAIRE on a dedicated HPC system (multi-GPU / multi-CPU envoriment) take a look at the examples provided in the [Job Submission on Dedicated Systems](#hpc) section.
doc/README-RUNME.md:In [runclaire01.sh](https://github.com/andreasmang/claire/tree/gpu/doc/examples/runclaire01.sh) we execute CLAIRE for a synthetic test problem of size 32x32x32. We use default settings for our solver:
doc/README-RUNME.md:In [runclaire03.sh](https://github.com/andreasmang/claire/tree/gpu/doc/examples/runclaire03.sh) we execute CLAIRE for real medical images (in NIfTI format) of size 128x150x128. We use 20 MPI tasks. The data can be found in the [docs/data](data) subdirectory. We use default settings for our solver:
doc/README-RUNME.md:In [runclaire04.sh](https://github.com/andreasmang/claire/tree/gpu/doc/examples/runclaire04.sh) we execute CLAIRE to automatically identify an adequate regularization parameter for a given set of images. We use default settings for our solver:
doc/README-RUNME.md:In [runclaire05.sh](https://github.com/andreasmang/claire/tree/gpu/doc/examples/runclaire05.sh) we show how to execute CLAIRE using a parameter continuation scheme with a target regularization parameter for the velocity. We use default settings for our solver:
doc/README-RUNME.md:In [runclaire06.sh](https://github.com/andreasmang/claire/tree/gpu/doc/examples/runclaire06.sh) we show how to store the computed velocity field on file. We use default settings for our solver:
doc/README-RUNME.md:In [runtools01.sh](https://github.com/andreasmang/claire/tree/gpu/doc/examples/runtools01.sh) we show how to transport an image (i.e., e.g., compute the deformed template image after a velocity has been computed using `claire`.)
doc/README-RUNME.md:The input are the three components of the computed velocity (`-v$i$ velocity-field-x$i$.nii.gz `) and the image to be transported (`-ifile $datdir/brain01.nii.gz`; `$datdir` points to the folder the data is located in, i.e., [doc/data](https://github.com/andreasmang/claire/tree/gpu/doc/data)). The output is the transported brain image (`-xfile brain01-transported.nii.gz`). The user can add a path as prefix if desired. The command to tell `clairetools` that we are interested in solving the forward problem (i.e., transporting/deforming an image) is `-deformimage`. The line breaks (backslashes `\`) are only added for readability.
doc/README-RUNME.md:In [runtools02.sh](https://github.com/andreasmang/claire/tree/gpu/doc/examples/runtools02.sh) we show how to compute the determinant of the deformation gradient (alas Jacobian) from a velocity field that has been computed using `claire`.
doc/README-RUNME.md:In [runtools03.sh](https://github.com/andreasmang/claire/tree/gpu/doc/examples/runtools03.sh) we show how to convert data from `*.nii.gz` to `*.nc`.
doc/README-RUNME.md:To execute CLAIRE on a single GPU node, interactively, do the following:
doc/README-RUNME.md:* multi-GPU (2 GPUs one node): [doc/examples/longhorn_mgpu.slurm](examples/longhorn_mgpu.slurm)
doc/README-RUNME.md:These tests are implemented in the `*.cpp` files available in the [UnitTests](https://github.com/andreasmang/claire/tree/gpu/src/UnitTests) subfolder.
doc/README-INSTALL.md:|Test   | Compiler  | MPI            | CUDA | PETSc  | CPU    | GPU   | System       |
doc/README-INSTALL.md:* CUDA-API
doc/README-INSTALL.md:The compiler needs `C++11` support. The GPU version of CLAIRE requires the following libraries to be installed on your system:
doc/README-INSTALL.md:* MPI (with GPU support (CUDA-aware MPI) for multi-GPU multi-node)
doc/README-INSTALL.md:* PETSc with CUDA support (see [https://www.mcs.anl.gov/petsc](https://www.mcs.anl.gov/petsc))
doc/README-INSTALL.md:The *compressed* tarball files (i.e, `LIBRARY-NAME.tar.gz`) should remain located in or be added to the [deps](../deps) folder. Make sure that all libraries are downloaded (the progress bar of `wget` should be full). To view the urls for the libraries you can take a look at the [deps/makefile](../deps/makefile). We provide additional information about these libraries [below](#depsinf). This also includes links to versions for these libraries that we have used to compile the GPU version of CLAIRE before.
doc/README-INSTALL.md:| WITH_CUDA_MPI   | MPI is CUDA-aware                                     | yes     | yes, no       |
doc/README-INSTALL.md:| NVCC            | Path to CUDA compiler                                 | nvcc    | file path     |
doc/README-INSTALL.md:| WITH_CUDA_MPI  | MPI is CUDA-aware                                     | yes     | yes, no       |
doc/README-INSTALL.md:| GPU_VERSION    | GPU CUDA version to compile, e.g. 35, 60, 70, 75      |         | Compute Capability |
doc/README-INSTALL.md:| NVCC_FLAGS     | additional flags for the CUDA compiler                |         |               |
doc/README-INSTALL.md:| CUDA_DIR       | main path to the CUDA include and lib directory       |         |               |
doc/README-INSTALL.md:2) spectrum_mpi/10.3.0   5) cmake/3.16.1    8) cuda/10.2 (g)
doc/README-INSTALL.md:make BUILD_TARGET=POWER9 GPU_VERSION=70
doc/README-INSTALL.md:A job submission file for TACC's Longhorn system (for multi-GPU exection) can be found in [doc/examples/longhorn_mgpu.slurm](examples/longhorn_mgpu.slurm).
doc/README-INSTALL.md:* if MPI is not compiled with CUDA-aware options, add the file `.petscrc` to the working directory and add the option `-use_gpu_aware_mpi 0`
doc/README-INSTALL.md:* CUDA >= 11.0 is only supported with PETSc >= 3.14.
doc/README-INSTALL.md:* Kepler GPUs work with PETSc 3.12.4  (others not tested)
doc/README-INSTALL.md:* Compiling PETSc with CUDA support on cluster login nodes without GPUs might fail
doc/README-INSTALL.md:* PNETCDF is currently not tested for GPUs
doc/README-INSTALL.md:* The GPU version of CLAIRE can currently only be compiled in single precision. This limits the selection of regularization operators to H1-type regularization only. There are issues with the numerical accuracy of H2- and H3-type regularization operators for single precision. Applying these operators requires a compilation in double precision (available on the GPU branch) 
doc/README-NEWS.md:* 28/2020 We have released a GPU version of CLAIRE. If you are interested in using our new (multi-node multi-)GPU version, switch to the **GPU branch**. If you are interested in learning more about the GPU version of CLAIRE, check out our [publications](README-REFERENCES.md).
doc/examples/longhorn_mgpu.slurm:#SBATCH --gpus-per-task=1
doc/examples/longhorn_mgpu.slurm:#SBATCH -n 2 # number of GPUs per node
doc/examples/longhorn_mgpu.slurm:export MY_SPECTRUM_OPTIONS="--gpu --aff on"
doc/examples/longhorn_mgpu.slurm:mpirun --gpu --aff on --host $HOSTS -np 2 $BINDIR/claire $CLAIREOPT
doc/CONTRIBUTING.md:We have implemented several tests to check the accuracy of our numerical implementation. These are described in more detail in [doc/README-RUNME.md](https://github.com/andreasmang/claire/blob/gpu/doc/README-RUNME.md#testing-and-benchmarks-).
README.md:**CLAIRE** stands for *Constrained Large Deformation Diffeomorphic Image Registration*. It is a C/C++ software package for velocity-based diffeomorphic image registration in three dimensions. Its performance is optimized for multi-core CPU systems (`cpu` branch) and multi-node, multi-GPU architectures (`gpu` branch; default). The CPU version uses MPI for data parallelism, and has been demonstrated to scale on several supercomputing platforms. CLAIRE can be executed on large-scale state-of-the-art computing systems as well as on local compute systems with limited resources.
README.md:Notice that the CPU version is accurate and running but new features are currently only being added to the GPU version. The GPU code is a major revision and therefore considered the default and recommended for use. 
include/DifferentiationSM.hpp://#ifdef REG_HAS_CUDA
include/DifferentiationSM.hpp://#ifdef REG_HAS_CUDA
include/interp3_gpu_mpi.hpp:#ifndef _INTERP3_GPU_MPI_HPP_
include/interp3_gpu_mpi.hpp:#define _INTERP3_GPU_MPI_HPP_
include/interp3_gpu_mpi.hpp:#include "interp3_gpu_new.hpp"
include/interp3_gpu_mpi.hpp://#define INTERP_PINNED // if defined will use pinned memory for GPU
include/interp3_gpu_mpi.hpp:struct Interp3_Plan_GPU{
include/interp3_gpu_mpi.hpp:  Interp3_Plan_GPU(size_t g_alloc_max, bool cuda_aware);
include/interp3_gpu_mpi.hpp:                    cudaTextureObject_t yi_tex, 
include/interp3_gpu_mpi.hpp:  bool cuda_aware;
include/interp3_gpu_mpi.hpp:  // GPU memory pointers for state and adjoint fields
include/interp3_gpu_mpi.hpp:  ~Interp3_Plan_GPU();
include/UnitTestOpt.hpp:  PetscErrorCode TestInterpolationMultiGPU(RegOpt *m_Opt);
include/UnitTestOpt.hpp:  PetscErrorCode TestVectorFieldInterpolationMultiGPU(RegOpt *m_Opt);
include/UnitTestOpt.hpp:  PetscErrorCode TestTrajectoryMultiGPU(RegOpt *m_Opt);
include/DifferentiationFD.hpp:#ifdef REG_HAS_CUDA
include/DifferentiationFD.hpp:  cudaTextureObject_t mtex;
include/MemoryUtils.hpp:#ifdef REG_HAS_CUDA
include/MemoryUtils.hpp:    ierr = cudaMalloc(reinterpret_cast<void**>(&ptr), size); CHKERRCUDA(ierr);
include/MemoryUtils.hpp:#ifdef REG_HAS_CUDA
include/MemoryUtils.hpp:    ierr = cudaFree(ptr); CHKERRCUDA(ierr);
include/MemoryUtils.hpp:#ifdef REG_HAS_CUDA
include/MemoryUtils.hpp:      ierr = cudaMemcpy(static_cast<void*>(this->m_DevicePtr),
include/MemoryUtils.hpp:        cudaMemcpyHostToDevice); CHKERRCUDA(ierr);
include/MemoryUtils.hpp:      ierr = cudaMemcpy(static_cast<void*>(this->m_DevicePtr),
include/MemoryUtils.hpp:        cudaMemcpyHostToDevice); CHKERRCUDA(ierr);
include/MemoryUtils.hpp:#ifdef REG_HAS_CUDA
include/MemoryUtils.hpp:      ierr = cudaMemcpy(static_cast<void*>(this->m_HostPtr),
include/MemoryUtils.hpp:        cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
include/MemoryUtils.hpp:      ierr = cudaMemcpy(static_cast<void*>(this->m_HostPtr),
include/MemoryUtils.hpp:        cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
include/MemoryUtils.hpp:#ifdef REG_HAS_CUDA
include/MemoryUtils.hpp:    ierr = cudaMalloc(reinterpret_cast<void**>(&this->m_DevicePtr), this->m_N*sizeof(T)); CHKERRCUDA(ierr);
include/MemoryUtils.hpp:#ifdef REG_HAS_CUDA
include/MemoryUtils.hpp:    ierr = cudaFree(this->m_DevicePtr); CHKERRCUDA(ierr);
include/mpicufft.hpp:#include <cuda.h>
include/mpicufft.hpp:  MPIcuFFT (MPI_Comm comm=MPI_COMM_WORLD, bool mpi_cuda_aware=false);
include/mpicufft.hpp:  bool cuda_aware;
include/SemiLagrangian.hpp:#ifdef REG_HAS_CUDA
include/SemiLagrangian.hpp:#include "SemiLagrangianGPUNew.hpp"
include/SemiLagrangian.hpp:#endif // REG_HAS_CUDA
include/cuda_helper.hpp:#ifndef __CUDA_HELPER_HPP__
include/cuda_helper.hpp:#define __CUDA_HELPER_HPP__
include/cuda_helper.hpp:#include <cuda_runtime.h>
include/cuda_helper.hpp:#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__,false)
include/cuda_helper.hpp:#define cudaCheckKernelError() cudaCheckError(cudaPeekAtLastError())
include/cuda_helper.hpp:#define cudaCheckLastError() cudaCheckError(cudaGetLastError())
include/cuda_helper.hpp:inline int cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
include/cuda_helper.hpp:  if (code != cudaSuccess) {
include/cuda_helper.hpp:    fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
include/cuda_helper.hpp:inline void cudaPrintDeviceMemory(const char *msg = "") {
include/cuda_helper.hpp:  cudaGetDevice(&dev);
include/cuda_helper.hpp:  cudaMemGetInfo(&free_mem, &total_mem);
include/cuda_helper.hpp:  printf("%s GPU %i memory usage: used = %lf MiB, free = %lf MiB, total = %lf MiB\n",
include/interp3_gpu_new.hpp:#ifndef _INTERP3_GPU_HPP_
include/interp3_gpu_new.hpp:#define _INTERP3_GPU_HPP_
include/interp3_gpu_new.hpp:#include <cuda.h>
include/interp3_gpu_new.hpp:#include <cuda_runtime.h>
include/interp3_gpu_new.hpp:#include <cuda_helper.hpp>
include/interp3_gpu_new.hpp://void gpuInterp3D(PetscScalar* yi, 
include/interp3_gpu_new.hpp://  int* nx, long int nq, cudaTextureObject_t yi_tex, int iporder, PetscScalar* interp_time);
include/interp3_gpu_new.hpp:void gpuInterp3D(PetscScalar* yi, 
include/interp3_gpu_new.hpp:  IntType* nx, long int nq, cudaTextureObject_t yi_tex, int iporder, PetscScalar* interp_time);
include/interp3_gpu_new.hpp://void gpuInterpVec3D(PetscScalar* yi1, PetscScalar* yi2, PetscScalar* yi3, 
include/interp3_gpu_new.hpp://    int* nx, long int nq, cudaTextureObject_t yi_tex, int iporder, PetscScalar* interp_time);
include/interp3_gpu_new.hpp:void gpuInterpVec3D(PetscScalar* yi1, PetscScalar* yi2, PetscScalar* yi3, 
include/interp3_gpu_new.hpp:    IntType* nx, long int nq, cudaTextureObject_t yi_tex, int iporder, PetscScalar* interp_time);
include/interp3_gpu_new.hpp:extern "C" cudaTextureObject_t gpuInitEmptyTexture(IntType* nx);
include/DeformationFields.hpp://#ifdef REG_HAS_CUDA
include/DeformationFields.hpp://#include "SemiLagrangianGPUNew.hpp"
include/DeformationFields.hpp://#ifdef REG_HAS_CUDA
include/DeformationFields.hpp://    typedef SemiLagrangianGPUNew SemiLagrangianType;
include/interp3.hpp:void gpu_interp3_p(Real* reg_grid_vals, int data_dof, int* N_reg,
include/interp3.hpp:void gpu_interp3_ghost_xyz_p(Real* reg_grid_vals_d, int data_dof, int* N_reg,
include/interp3.hpp:void gpu_par_interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
include/Spectral.hpp:#ifdef REG_HAS_CUDA
include/Spectral.hpp:#include "cuda_helper.hpp"
include/Spectral.hpp:#ifdef REG_HAS_CUDA
include/GhostPlan.hpp:#include <interp3_gpu_new.hpp>
include/GhostPlan.hpp:    cudaStream_t* stream;
include/RegOpt.hpp:    GPUCOMP,      ///< time spent on gpu
include/RegOpt.hpp:    PetscScalar m_GPUtime = 0;
include/RegOpt.hpp:#ifdef REG_HAS_CUDA
include/RegOpt.hpp:    int m_gpu_id;                      ///< id of used GPU
include/TextureDifferentiationKernel.hpp:    cudaTextureObject_t gpuInitEmptyGradientTexture(IntType *);
include/TextureDifferentiationKernel.hpp:    PetscErrorCode computeGradient(ScalarType* , ScalarType* , ScalarType* , const ScalarType*, cudaTextureObject_t, IntType*, IntType*, IntType*, ScalarType*, bool mgpu=false);
include/TextureDifferentiationKernel.hpp:    PetscErrorCode computeDivergence(ScalarType* , const ScalarType* , const ScalarType* , const ScalarType*, cudaTextureObject_t, IntType*, IntType*, IntType*,  ScalarType*, bool mgpu=false);
include/TextureDifferentiationKernel.hpp:    PetscErrorCode computeLaplacian(ScalarType* , const ScalarType*, cudaTextureObject_t, IntType*, IntType*, IntType*, ScalarType*,  ScalarType, bool mgpu=false);
include/TextureDifferentiationKernel.hpp:    PetscErrorCode computeDivergenceX(ScalarType* , ScalarType*, IntType*, IntType*, IntType*, ScalarType*, bool mgpu=false);
include/TextureDifferentiationKernel.hpp:    PetscErrorCode computeDivergenceY(ScalarType* , ScalarType*, IntType*, IntType*, IntType*,  ScalarType*, bool mgpu=false);
include/TextureDifferentiationKernel.hpp:    PetscErrorCode computeDivergenceZ(ScalarType* , ScalarType*, IntType*, IntType*, IntType*,  ScalarType*, bool mgpu=false);
include/KernelUtils.hpp:#if defined(__CUDACC__) && defined(REG_HAS_CUDA) // for CUDA compiler
include/KernelUtils.hpp:  #include "cuda_helper.hpp"
include/KernelUtils.hpp:#if defined(__CUDACC__) && defined(REG_HAS_CUDA) // compiled by CUDA compiler
include/KernelUtils.hpp: * @brief GPU kernel function wrapper for reduction kernels
include/KernelUtils.hpp:__global__ void ReductionKernelGPU(ScalarType *res, int nl, Args ... args) {
include/KernelUtils.hpp: * @brief GPU kernel function wrapper for reduction kernels
include/KernelUtils.hpp:__global__ void SpectralReductionKernelGPU(ScalarType *res, real3 wave, real3 nx, int3 nl, Args ... args) {
include/KernelUtils.hpp: * @brief GPU kernel function wrapper for spectral operators
include/KernelUtils.hpp:__global__ void SpectralKernelGPU(real3 wave, real3 nx, int3 nl, Args ... args) {
include/KernelUtils.hpp: * @brief GPU kernel function wrapper for spacial operators
include/KernelUtils.hpp:__global__ void SpacialKernelGPU(int3 p, int3 nl, Args ... args) {
include/KernelUtils.hpp: * @brief GPU kernel function wrapper
include/KernelUtils.hpp:__global__ void KernelGPU(int nl, Args ... args) {
include/KernelUtils.hpp: * @brief Starts a GPU kernel for spectral operators
include/KernelUtils.hpp:PetscErrorCode SpectralKernelCallGPU(IntType nstart[3], IntType nx[3], IntType nl[3],
include/KernelUtils.hpp:    SpectralKernelGPU<KernelFn><<<grid, block>>>(wave, nx3, nl3, args...);
include/KernelUtils.hpp:    ierr = cudaDeviceSynchronize(); CHKERRCUDA(ierr);
include/KernelUtils.hpp:    ierr = cudaCheckKernelError(); CHKERRCUDA(ierr);
include/KernelUtils.hpp: * @brief Starts a GPU kernel for spectral reduction operators
include/KernelUtils.hpp:PetscErrorCode SpectralReductionKernelCallGPU(ScalarType &value, ScalarType *workspace,
include/KernelUtils.hpp:    SpectralReductionKernelGPU<256, KernelFn><<<grid, block>>>(workspace, wave, nx3, nl3, args...);
include/KernelUtils.hpp:    ierr = cudaDeviceSynchronize(); CHKERRCUDA(ierr);
include/KernelUtils.hpp:    ierr = cudaCheckKernelError(); CHKERRCUDA(ierr);
include/KernelUtils.hpp:    ierr = cudaDeviceSynchronize(); CHKERRCUDA(ierr);
include/KernelUtils.hpp:    ierr = cudaCheckKernelError(); CHKERRCUDA(ierr);
include/KernelUtils.hpp:    ierr = cudaMemcpy(reinterpret_cast<void*>(&value), reinterpret_cast<void*>(workspace),
include/KernelUtils.hpp:                      sizeof(ScalarType), cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
include/KernelUtils.hpp: * @brief Starts a GPU kernel for spacial operators
include/KernelUtils.hpp:PetscErrorCode SpacialKernelCallGPU(IntType nstart[3], IntType nl[3], Args ... args) {
include/KernelUtils.hpp:    SpacialKernelGPU<KernelFn><<<grid, block>>>(p, nl3, args...);
include/KernelUtils.hpp:    ierr = cudaDeviceSynchronize(); CHKERRCUDA(ierr);
include/KernelUtils.hpp:    ierr = cudaCheckKernelError(); CHKERRCUDA(ierr);
include/KernelUtils.hpp: * @brief Starts a GPU kernel
include/KernelUtils.hpp:PetscErrorCode KernelCallGPU(IntType nl, Args ... args) {
include/KernelUtils.hpp:    KernelGPU<KernelFn><<<grid, block>>>(nl, args...);
include/KernelUtils.hpp:    ierr = cudaDeviceSynchronize(); CHKERRCUDA(ierr);
include/KernelUtils.hpp:    ierr = cudaCheckKernelError(); CHKERRCUDA(ierr);
include/KernelUtils.hpp: * @brief Starts a GPU kernel with value reduction
include/KernelUtils.hpp:PetscErrorCode ReductionKernelCallGPU(ScalarType &value, IntType nl, Args ... args) {
include/KernelUtils.hpp:  //res = GPUKernelWorkspace.ptr;
include/KernelUtils.hpp:    ReductionKernelGPU<256, KernelFn><<<grid, block>>>(res, nl, args...);
include/KernelUtils.hpp:    ierr = cudaDeviceSynchronize(); CHKERRCUDA(ierr);
include/KernelUtils.hpp:    ierr = cudaCheckKernelError(); CHKERRCUDA(ierr);
include/KernelUtils.hpp:    ierr = cudaDeviceSynchronize(); CHKERRCUDA(ierr);
include/KernelUtils.hpp:    ierr = cudaCheckKernelError(); CHKERRCUDA(ierr);
include/KernelUtils.hpp:    ierr = cudaMemcpy(reinterpret_cast<void*>(&value), reinterpret_cast<void*>(res),
include/KernelUtils.hpp:                      sizeof(ScalarType), cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
include/KernelUtils.hpp:PetscErrorCode ReductionKernelCallGPU(ScalarType &value, ScalarType *workspace, IntType nl, Args ... args) {
include/KernelUtils.hpp:    ReductionKernelGPU<256, KernelFn><<<grid, block>>>(workspace, nl, args...);
include/KernelUtils.hpp:    ierr = cudaDeviceSynchronize(); CHKERRCUDA(ierr);
include/KernelUtils.hpp:    ierr = cudaCheckKernelError(); CHKERRCUDA(ierr);
include/KernelUtils.hpp:    ierr = cudaDeviceSynchronize(); CHKERRCUDA(ierr);
include/KernelUtils.hpp:    ierr = cudaCheckKernelError(); CHKERRCUDA(ierr);
include/KernelUtils.hpp:    ierr = cudaMemcpy(reinterpret_cast<void*>(&value), reinterpret_cast<void*>(workspace),
include/KernelUtils.hpp:                      sizeof(ScalarType), cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
include/CLAIREUtils.hpp:PetscErrorCode VecFieldPointWiseNormGPU(ScalarType*, const ScalarType*, const ScalarType*, const ScalarType*, IntType);
include/CLAIREUtils.hpp:/*! generic copy function based CPU or GPU implementation */
include/SemiLagrangianGPUNew.hpp:#ifndef _SEMILAGRANGIANGPUNEW_HPP_
include/SemiLagrangianGPUNew.hpp:#define _SEMILAGRANGIANGPUNEW_HPP_
include/SemiLagrangianGPUNew.hpp:#include "interp3_gpu_mpi.hpp"
include/SemiLagrangianGPUNew.hpp:class SemiLagrangianGPUNew {
include/SemiLagrangianGPUNew.hpp:    SemiLagrangianGPUNew();
include/SemiLagrangianGPUNew.hpp:    SemiLagrangianGPUNew(RegOpt*);
include/SemiLagrangianGPUNew.hpp:    virtual ~SemiLagrangianGPUNew();
include/SemiLagrangianGPUNew.hpp:    cudaTextureObject_t m_texture;
include/SemiLagrangianGPUNew.hpp:    Interp3_Plan_GPU* m_StatePlan;
include/SemiLagrangianGPUNew.hpp:    Interp3_Plan_GPU* m_AdjointPlan;
include/SemiLagrangianGPUNew.hpp:    bool cuda_aware = true;
include/SemiLagrangianGPUNew.hpp:typedef SemiLagrangianGPUNew SemiLagrangian;
include/TypeDef.hpp:#if defined(REG_HAS_CUDA) || defined(REG_FFT_CUDA)
include/TypeDef.hpp:  #include "petsccuda.h"
include/TypeDef.hpp:#ifndef CHKERRCUDA
include/TypeDef.hpp:  #define CHKERRCUDA CHKERRQ
include/TypeDef.hpp:  #include <cuda.h>
include/TypeDef.hpp:  #include <cuda_runtime_api.h>
include/TypeDef.hpp:#ifdef REG_FFT_CUDA
include/TypeDef.hpp:  //#include "accfft_gpu.h"
include/TypeDef.hpp:  //#include "accfft_gpuf.h"
include/TypeDef.hpp:  //#include "accfft_operators_gpu.h"
include/TypeDef.hpp:#if defined(REG_DBG_CUDA) && defined(REG_HAS_CUDA)
include/TypeDef.hpp:  #define DebugGPUStartEvent(str) nvtxRangePushA(str)
include/TypeDef.hpp:  #define DebugGPUStopEvent() nvtxRangePop()
include/TypeDef.hpp:  #define DebugGPUNotImplemented() WrngMsg("Not implemented for GPU")
include/TypeDef.hpp:  #define DebugGPUStartEvent(str)
include/TypeDef.hpp:  #define DebugGPUStopEvent()
include/TypeDef.hpp:  #define DebugGPUNotImplemented() 0
include/TypeDef.hpp:#define VecCUDAGetArray VecCUDAGetArrayReadWrite
include/TypeDef.hpp:#define VecCUDARestoreArray VecCUDARestoreArrayReadWrite
include/TypeDef.hpp:#ifdef REG_FFT_CUDA // GPU FFT
include/TypeDef.hpp:  inline void* claire_alloc(size_t size) { void* ptr; cudaMallocHost(&ptr, size); return ptr; }
include/TypeDef.hpp:  inline void claire_free(void* ptr) { cudaFreeHost(ptr); }
include/TypeDef.hpp:    using FFTPlanType = accfft_plan_gpuf;
include/TypeDef.hpp:    const auto accfft_plan_dft_3d_r2c = accfft_plan_dft_3d_r2c_gpuf;
include/TypeDef.hpp:    const auto accfft_cleanup = accfft_cleanup_gpuf;
include/TypeDef.hpp:    using FFTPlanType = accfft_plan_gpu;
include/TypeDef.hpp:    const auto accfft_plan_dft_3d_r2c = accfft_plan_dft_3d_r2c_gpu;
include/TypeDef.hpp:    const auto accfft_cleanup = accfft_cleanup_gpu;
include/TypeDef.hpp:  // dummy function replacing fftw init (not needed by GPU fft)
include/TypeDef.hpp:  // AccFFT wrapper for GPU kernels
include/TypeDef.hpp:  #define accfft_execute_r2c_t accfft_execute_r2c_gpu_t
include/TypeDef.hpp:  #define accfft_execute_c2r_t accfft_execute_c2r_gpu_t
include/TypeDef.hpp:  #define accfft_grad_t accfft_grad_gpu_t
include/TypeDef.hpp:  #define accfft_laplace_t accfft_laplace_gpu_t
include/TypeDef.hpp:  #define accfft_divergence_t accfft_divergence_gpu_t
include/TypeDef.hpp:  #define accfft_biharmonic_t accfft_biharmonic_gpu_t
include/TypeDef.hpp:#include <cuda_runtime.h>
include/TypeDef.hpp:  cudaMemGetInfo(&free_mem, &total_mem); \
makefile:#build code for GPUs (yes, no)
makefile:BUILD_GPU = yes
makefile:WITH_CUDA_MPI = yes
makefile:GPU_VERSION = 
makefile:$(LIB_DIR)/libclaire.so: $(CPU_OBJS) $(GPU_OBJS)
makefile:	@echo "BUILD_GPU:    $(BUILD_GPU); [yes, no]"
makefile:	@echo "WITH_CUDA_MPI: $(WITH_CUDA_MPI); [yes, no]"
makefile:	@echo "CUDA_DIR:      $(CUDA_DIR)"
makefile:	@echo "GPU_VERSION:   $(GPU_VERSION)"
filelist.mk:GPU_FILES=
filelist.mk:ifneq ($(BUILD_GPU),yes)
filelist.mk:	# GPU build specific C++ files
filelist.mk:	CPU_FILES += $(SRC_DIR)/SemiLagrangian/SemiLagrangianGPUNew.cpp
filelist.mk:	# GPU build specific CUDA files
filelist.mk:	GPU_FILES += $(SRC_DIR)/Interpolation/interp3_gpu_new.cu
filelist.mk:	GPU_FILES += $(SRC_DIR)/Solver/TransportKernel.cu
filelist.mk:	GPU_FILES += $(SRC_DIR)/CLAIREUtilsKernel.cu
filelist.mk:	GPU_FILES += $(SRC_DIR)/DeformationFields/DeformationKernel.cu
filelist.mk:	GPU_FILES += $(SRC_DIR)/DistanceMeasure/DistanceMeasureKernel.cu
filelist.mk:	GPU_FILES += $(SRC_DIR)/Differentiation/DifferentiationKernel.cu
filelist.mk:	GPU_FILES += $(SRC_DIR)/Differentiation/TextureDifferentiationKernel.cu
filelist.mk:	GPU_FILES += $(SRC_DIR)/Spectral/SpectralKernel.cu
filelist.mk:	GPU_FILES += $(SRC_DIR)/PreconditionerKernel.cu
filelist.mk:	GPU_FILES += $(SRC_DIR)/Regularization/RegularizationKernel.cu
filelist.mk:	GPU_FILES += $(SRC_DIR)/Interpolation/Interp3_Plan_GPU.cu
filelist.mk:	GPU_FILES += $(SRC_DIR)/Interpolation/Interp3_Plan_GPU_kernel.cu
filelist.mk:	GPU_FILES += $(SRC_DIR)/SemiLagrangian/SemiLagrangianKernel.cu
filelist.mk:	GPU_FILES += $(SRC_DIR)/TwoLevel/TwoLevelKernel.cu
filelist.mk:GPU_OBJS = $(patsubst %.cu,$(OBJ_DIR)/%.co,$(GPU_FILES))
filelist.mk:	OBJS = $(CPU_OBJS) $(GPU_OBJS)
apps/benchmark.cpp:    ss << "GPU compute time:"<< std::scientific << opt->m_GPUtime;
config.mk:ifeq ($(WITH_CUDA_MPI), yes)
config.mk:	CXXFLAGS += -DREG_HAS_MPICUDA
config.mk:	CXXFLAGS += -DREG_HAS_MPICUDA
config.mk:ifeq ($(BUILD_GPU), yes)
config.mk:$(error GPU build only supports single precision)
config.mk:	CUDA_DIR = $(abspath $(subst bin,,$(dir $(shell which $(NVCC)))))
config.mk:	INCLUDES += $(CUDA_DIR)/include
config.mk:	LIBRARIES += $(CUDA_DIR)/lib
config.mk:	LIBRARIES += $(CUDA_DIR)/lib64
config.mk:	LDFLAGS += -lcuda -lcudart -lcufft -lcublas -lcusparse -lcusolver
config.mk:	CXXFLAGS += -DREG_HAS_CUDA
config.mk:	CXXFLAGS += -DREG_FFT_CUDA
config.mk:		CXXFLAGS += -DREG_DBG_CUDA
config.mk:	ifdef GPU_VERSION
config.mk:		NVCCFLAGS += -gencode arch=compute_$(GPU_VERSION),code=sm_$(GPU_VERSION)
config.mk:$(error This branch only supports GPU build)
config.mk:#link ACCFFT and FFTW if not building for GPUs
config.mk:ifneq ($(BUILD_GPU), yes)
config.mk:CACHE = "$(BUILD_GPU) $(BUILD_TEST) $(BUILD_PYTHON) $(WITH_NIFTI) $(WITH_PNETCDF) $(WITH_DOUBLE) $(WITH_DEBUG) $(BUILD_SHARED) $(WITH_DEBUG) $(BUILD_TARGET) $(GPU_VERSION) $(shell uname -a)"
legacy-build/config/setup.mk.power9v100:ifeq ($(USECUDA),yes)
legacy-build/config/setup.mk.power9v100:	CXXFLAGS += -DREG_HAS_CUDA
legacy-build/config/setup.mk.power9v100:	CXXFLAGS += -DREG_FFT_CUDA
legacy-build/config/setup.mk.power9v100:	ifeq ($(USECUDADBG),yes)
legacy-build/config/setup.mk.power9v100:		CXXFLAGS += -DREG_DBG_CUDA
legacy-build/config/setup.mk.power9v100:  ifeq ($(USEMPICUDA), yes)
legacy-build/config/setup.mk.power9v100:    CXXFLAGS += -DREG_HAS_MPICUDA
legacy-build/config/setup.mk.power9v100:ifeq ($(USECUDA),yes)
legacy-build/config/setup.mk.power9v100:	OBJDIR = ./objgpu
legacy-build/config/setup.mk.power9v100:ifeq ($(USECUDA),yes)
legacy-build/config/setup.mk.power9v100:    # CUDA includes
legacy-build/config/setup.mk.power9v100:    CUDA_INC = -I$(CUDA_DIR)/include -I$(INCDIR) -I$(EXSRCDIR) -I./deps/$(EXSRCDIR)
legacy-build/config/setup.mk.power9v100:ifeq ($(USECUDA),yes)
legacy-build/config/setup.mk.power9v100:				CLAIRE_INC += -isystem$(PETSC_DIR)/include -isystem$(PETSC_DIR)/$(PETSC_ARCH_CUDA_SINGLE_DBG)/include
legacy-build/config/setup.mk.power9v100:				CUDA_INC += -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH_CUDA_SINGLE_DBG)/include
legacy-build/config/setup.mk.power9v100:				CLAIRE_INC += -isystem$(PETSC_DIR)/include -isystem$(PETSC_DIR)/$(PETSC_ARCH_CUDA_SINGLE)/include
legacy-build/config/setup.mk.power9v100:				CUDA_INC += -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH_CUDA_SINGLE)/include
legacy-build/config/setup.mk.power9v100:				CLAIRE_INC += -isystem$(PETSC_DIR)/include -isystem$(PETSC_DIR)/$(PETSC_ARCH_CUDA_DOUBLE_DBG)/include
legacy-build/config/setup.mk.power9v100:				CUDA_INC += -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH_CUDA_DOUBLE_DBG)/include
legacy-build/config/setup.mk.power9v100:				CLAIRE_INC += -isystem$(PETSC_DIR)/include -isystem$(PETSC_DIR)/$(PETSC_ARCH_CUDA_DOUBLE)/include
legacy-build/config/setup.mk.power9v100:				CUDA_INC += -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH_CUDA_DOUBLE)/include
legacy-build/config/setup.mk.power9v100:CUDA_INC += -I$(MPI_INC)
legacy-build/config/setup.mk.power9v100:# CUDA INCLUDE in CLAIRE
legacy-build/config/setup.mk.power9v100:ifeq ($(USECUDA),yes)
legacy-build/config/setup.mk.power9v100:#		CUDA_INC += -I$(ACCFFT_DIR)/include
legacy-build/config/setup.mk.power9v100:#		CUDA_INC += -I$(FFTW_DIR)/include
legacy-build/config/setup.mk.power9v100:    CLAIRE_INC += -I$(CUDA_DIR)/include
legacy-build/config/setup.mk.power9v100:# CUDA flags
legacy-build/config/setup.mk.power9v100:CUDA_FLAGS=-c -Xcompiler "$(CXXFLAGS)" -std=c++11 -O3 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -m64
legacy-build/config/setup.mk.power9v100:#CUDA_FLAGS+=-gencode arch=compute_60,code=sm_60
legacy-build/config/setup.mk.power9v100:CUDA_FLAGS+=-gencode arch=compute_70,code=sm_70
legacy-build/config/setup.mk.power9v100:#CUDA_FLAGS+=-gencode arch=compute_50,code=sm_50
legacy-build/config/setup.mk.power9v100:  CUDA_FLAGS += -ccbin xlC
legacy-build/config/setup.mk.power9v100:	CUDA_INC += -I$(NIFTI_DIR)/include/nifti
legacy-build/config/setup.mk.power9v100:	CUDA_INC += -I$(PNETCDF_DIR)/include
legacy-build/config/setup.mk.power9v100:ifeq ($(USECUDA),yes)
legacy-build/config/setup.mk.power9v100:				LDFLAGS += -L$(PETSC_DIR)/lib -L$(PETSC_DIR)/$(PETSC_ARCH_CUDA_SINGLE_DBG)/lib
legacy-build/config/setup.mk.power9v100:				LDFLAGS += -L$(PETSC_DIR)/lib -L$(PETSC_DIR)/$(PETSC_ARCH_CUDA_SINGLE)/lib
legacy-build/config/setup.mk.power9v100:				LDFLAGS += -L$(PETSC_DIR)/lib -L$(PETSC_DIR)/$(PETSC_ARCH_CUDA_DOUBLE_DBG)/lib
legacy-build/config/setup.mk.power9v100:				LDFLAGS += -L$(PETSC_DIR)/lib -L$(PETSC_DIR)/$(PETSC_ARCH_CUDA_DOUBLE)/lib
legacy-build/config/setup.mk.power9v100:#CUDA LINKERS
legacy-build/config/setup.mk.power9v100:ifeq ($(USECUDA),yes)
legacy-build/config/setup.mk.power9v100:    LDFLAGS += -L$(CUDA_DIR)/lib64 -lcusparse -lcufft -lcublas -lcudart  -lcusolver
legacy-build/config/setup.mk.power9v100:    ifeq ($(USECUDADBG),yes)
legacy-build/config/setup.mk.power9v100:#ifeq ($(USECUDA),yes)
legacy-build/config/setup.mk.power9v100:#    LDFLAGS += -laccfft_gpu -laccfft_utils_gpu -lcudart -lcufft
legacy-build/config/files.mk:EXCUFILES=$(EXSRCDIR)/interp3_gpu_new.cu
legacy-build/config/files.mk:		$(SRCDIR)/Interpolation/Interp3_Plan_GPU.cu \
legacy-build/config/files.mk:		$(SRCDIR)/Interpolation/Interp3_Plan_GPU_kernel.cu \
legacy-build/config/files.mk:CPPFILESCUDA=$(SRCDIR)/RegOpt.cpp \
legacy-build/config/files.mk:		$(SRCDIR)/SemiLagrangian/SemiLagrangianGPUNew.cpp \
legacy-build/config/setup.mk:ifeq ($(USECUDA),yes)
legacy-build/config/setup.mk:	CXXFLAGS += -DREG_HAS_CUDA
legacy-build/config/setup.mk:	CXXFLAGS += -DREG_FFT_CUDA
legacy-build/config/setup.mk:	ifeq ($(USECUDADBG),yes)
legacy-build/config/setup.mk:		CXXFLAGS += -DREG_DBG_CUDA
legacy-build/config/setup.mk:ifeq ($(USECUDA),yes)
legacy-build/config/setup.mk:	BINDIR = ./bingpu
legacy-build/config/setup.mk:	OBJDIR = ./objgpu
legacy-build/config/setup.mk:ifeq ($(USECUDA),yes)
legacy-build/config/setup.mk:    # CUDA includes
legacy-build/config/setup.mk:    CUDA_INC = -I$(CUDA_DIR)/include -I$(INCDIR) -I$(EXSRCDIR) -I./deps/$(EXSRCDIR)
legacy-build/config/setup.mk:ifeq ($(USECUDA),yes)
legacy-build/config/setup.mk:				CLAIRE_INC += -isystem$(PETSC_DIR)/include -isystem$(PETSC_DIR)/$(PETSC_ARCH_CUDA_SINGLE_DBG)/include -I$(MPI_INC)
legacy-build/config/setup.mk:				CUDA_INC += -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH_CUDA_SINGLE_DBG)/include -I$(MPI_INC)
legacy-build/config/setup.mk:				CLAIRE_INC += -isystem$(PETSC_DIR)/include -isystem$(PETSC_DIR)/$(PETSC_ARCH_CUDA_SINGLE)/include -I$(MPI_INC)
legacy-build/config/setup.mk:				CUDA_INC += -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH_CUDA_SINGLE)/include -I$(MPI_INC)
legacy-build/config/setup.mk:				CLAIRE_INC += -isystem$(PETSC_DIR)/include -isystem$(PETSC_DIR)/$(PETSC_ARCH_CUDA_DOUBLE_DBG)/include -I$(MPI_INC)
legacy-build/config/setup.mk:				CUDA_INC += -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH_CUDA_DOUBLE_DBG)/include -I$(MPI_INC)
legacy-build/config/setup.mk:				CLAIRE_INC += -isystem$(PETSC_DIR)/include -isystem$(PETSC_DIR)/$(PETSC_ARCH_CUDA_DOUBLE)/include -I$(MPI_INC)
legacy-build/config/setup.mk:				CUDA_INC += -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH_CUDA_DOUBLE)/include -I$(MPI_INC)
legacy-build/config/setup.mk:# CUDA INCLUDE in CLAIRE
legacy-build/config/setup.mk:ifeq ($(USECUDA),yes)
legacy-build/config/setup.mk:	#CUDA_INC += -I$(ACCFFT_DIR)/include
legacy-build/config/setup.mk:	#CUDA_INC += -I$(FFTW_DIR)/include
legacy-build/config/setup.mk:  CLAIRE_INC += -I$(CUDA_DIR)/include
legacy-build/config/setup.mk:# CUDA flags
legacy-build/config/setup.mk:CUDA_FLAGS=-c -Xcompiler "$(CXXFLAGS)" --std=c++11 -O3 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -g
legacy-build/config/setup.mk:CUDA_FLAGS+=-gencode arch=compute_35,code=sm_35
legacy-build/config/setup.mk:#CUDA_FLAGS+=-gencode arch=compute_60,code=sm_60
legacy-build/config/setup.mk:#CUDA_FLAGS+=-gencode arch=compute_70,code=sm_70
legacy-build/config/setup.mk:CUDA_FLAGS+=-gencode arch=compute_75,code=sm_75
legacy-build/config/setup.mk:	CUDA_INC += -I$(NIFTI_DIR)/include/nifti
legacy-build/config/setup.mk:	CUDA_INC += -I$(PNETCDF_DIR)/include
legacy-build/config/setup.mk:ifeq ($(USECUDA),yes)
legacy-build/config/setup.mk:				LDFLAGS += -L$(PETSC_DIR)/lib -L$(PETSC_DIR)/$(PETSC_ARCH_CUDA_SINGLE_DBG)/lib
legacy-build/config/setup.mk:				LDFLAGS += -L$(PETSC_DIR)/lib -L$(PETSC_DIR)/$(PETSC_ARCH_CUDA_SINGLE)/lib
legacy-build/config/setup.mk:				LDFLAGS += -L$(PETSC_DIR)/lib -L$(PETSC_DIR)/$(PETSC_ARCH_CUDA_DOUBLE_DBG)/lib
legacy-build/config/setup.mk:				LDFLAGS += -L$(PETSC_DIR)/lib -L$(PETSC_DIR)/$(PETSC_ARCH_CUDA_DOUBLE)/lib
legacy-build/config/setup.mk:#CUDA LINKERS
legacy-build/config/setup.mk:ifeq ($(USECUDA),yes)
legacy-build/config/setup.mk:    LDFLAGS += -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lcublas -lcusparse -lcufft -lcusolver
legacy-build/config/setup.mk:    ifeq ($(USECUDADBG),yes)
legacy-build/config/setup.mk:ifeq ($(USECUDA),no)
legacy-build/config/setup.mk:ifeq ($(USECUDA),yes)
legacy-build/config/setup.mk:  #LDFLAGS += -laccfft_gpu -laccfft_utils_gpu -laccfft -laccfft_utils -lcudart -lcufft -lcublas -lcusolver
legacy-build/config/setup.mk:  LDFLAGS += -lcudart -lcufft -lcublas -lcusolver
legacy-build/deps/build_libs_default_gpu.sh:gpu=$2
legacy-build/deps/build_libs_default_gpu.sh:./build_libs.sh --gpu=$gpu
legacy-build/deps/build_libs_default_gpu.sh:    ./build_libs.sh --bpetsccudasgl --enableCUDA --gpu=$gpu
legacy-build/deps/build_libs_default_gpu.sh:    ./build_libs.sh --bfftw --enableCUDA --POWER9
legacy-build/deps/build_libs_default_gpu.sh:    ./build_libs.sh --bfftw --enableCUDA
legacy-build/deps/build_libs_default_gpu.sh:./build_libs.sh --baccfft --enableCUDA --gpu=$gpu
legacy-build/deps/build_libs_default_gpu.sh:./build_libs.sh --bzlib --enableCUDA
legacy-build/deps/build_libs_default_gpu.sh:./build_libs.sh --bnifti --enableCUDA
legacy-build/deps/build_libs_default_gpu.sh:./build_libs.sh --bmorton --enableCUDA
legacy-build/deps/build_libs_default_gpu_longhorn.sh:gpu="V100" # TACC Longhorn
legacy-build/deps/build_libs_default_gpu_longhorn.sh:#gpu="RTX" # TACC Frontera
legacy-build/deps/build_libs_default_gpu_longhorn.sh:#gpu="P100" # TACC Maverick2
legacy-build/deps/build_libs_default_gpu_longhorn.sh:./build_libs.sh --gpu=$gpu
legacy-build/deps/build_libs_default_gpu_longhorn.sh:    ./build_libs.sh --bpetsccudasgl --enableCUDA --gpu=$gpu
legacy-build/deps/build_libs_default_gpu_longhorn.sh:    ./build_libs.sh --bfftw --enableCUDA --POWER9
legacy-build/deps/build_libs_default_gpu_longhorn.sh:    ./build_libs.sh --bfftw --enableCUDA
legacy-build/deps/build_libs_default_gpu_longhorn.sh:./build_libs.sh --baccfft --enableCUDA --gpu=$gpu
legacy-build/deps/build_libs_default_gpu_longhorn.sh:./build_libs.sh --bzlib --enableCUDA
legacy-build/deps/build_libs_default_gpu_longhorn.sh:./build_libs.sh --bnifti --enableCUDA
legacy-build/deps/build_libs_here_shared.sh:enableCUDA=0
legacy-build/deps/build_libs_here_shared.sh:buildpetsccudasgl=0
legacy-build/deps/build_libs_here_shared.sh:buildpetsccudadbl=0
legacy-build/deps/build_libs_here_shared.sh:buildpetsccudasgldbg=0
legacy-build/deps/build_libs_here_shared.sh:buildpetsccudadbldbg=0
legacy-build/deps/build_libs_here_shared.sh:    --enableCUDA)
legacy-build/deps/build_libs_here_shared.sh:    enableCUDA=1
legacy-build/deps/build_libs_here_shared.sh:    --bpetsccudasgl)
legacy-build/deps/build_libs_here_shared.sh:    buildpetsccudasgl=1
legacy-build/deps/build_libs_here_shared.sh:    --bpetsccudasgldbg)
legacy-build/deps/build_libs_here_shared.sh:    buildpetsccudasgldbg=1
legacy-build/deps/build_libs_here_shared.sh:    --bpetsccudadbl)
legacy-build/deps/build_libs_here_shared.sh:    buildpetsccudadbl=1
legacy-build/deps/build_libs_here_shared.sh:    --bpetsccudadbldbg)
legacy-build/deps/build_libs_here_shared.sh:    buildpetsccudadbldbg=1
legacy-build/deps/build_libs_here_shared.sh:    echo "     --enableCUDA    flag: use CUDA for AccFFT"
legacy-build/deps/build_libs_here_shared.sh:    echo "     --bpetsccudasgl build PETSc library with nVidia-CUDA library (single precision)"
legacy-build/deps/build_libs_here_shared.sh:    echo "     --bpetsccudasgldbg build PETSc library with nVidia-CUDA library (single precision; debug mode)"
legacy-build/deps/build_libs_here_shared.sh:PETSC_CUDA_OPTIONS="
legacy-build/deps/build_libs_here_shared.sh:--with-cuda=1
legacy-build/deps/build_libs_here_shared.sh:--CUDAFLAGS='-arch=sm_35'
legacy-build/deps/build_libs_here_shared.sh:--CUDAOPTFLAGS='-O3'"
legacy-build/deps/build_libs_here_shared.sh:#-use-gpu-aware-mpi"
legacy-build/deps/build_libs_here_shared.sh:-DBUILD_GPU=false
legacy-build/deps/build_libs_here_shared.sh:if [ ${enableCUDA} -eq 1 ]; then
legacy-build/deps/build_libs_here_shared.sh:	ACCFFT_OPTIONS="${ACCFFT_OPTIONS} -DBUILD_GPU=true"
legacy-build/deps/build_libs_here_shared.sh:# PETSC-CUDA
legacy-build/deps/build_libs_here_shared.sh:PETSC_ARCH=cuda_opt_sgl
legacy-build/deps/build_libs_here_shared.sh:if [ ${builddep} -eq 1 -o ${buildpetsccudasgl} -eq 1 ]; then 
legacy-build/deps/build_libs_here_shared.sh:	echo "configuring PETSC-CUDA (single precision)"
legacy-build/deps/build_libs_here_shared.sh:	echo ./configure PETSC_DIR=${SRC_DIR} PETSC_ARCH=${PETSC_ARCH} --prefix=${BLD_DIR}/${PETSC_ARCH} ${PETSC_OPTIONS} ${PETSC_CUDA_OPTIONS} --with-precision=single
legacy-build/deps/build_libs_here_shared.sh:	./configure PETSC_DIR=${SRC_DIR} PETSC_ARCH=${PETSC_ARCH} --prefix=${BLD_DIR}/${PETSC_ARCH} ${PETSC_OPTIONS} ${PETSC_CUDA_OPTIONS} --with-precision=single
legacy-build/deps/build_libs_here_shared.sh:echo "export PETSC_ARCH_CUDA_SINGLE=${PETSC_ARCH}" >> ${BUILD_DIR}/environment_vars.sh
legacy-build/deps/build_libs_here_shared.sh:# PETSC-CUDA
legacy-build/deps/build_libs_here_shared.sh:PETSC_ARCH=cuda_opt_dbl
legacy-build/deps/build_libs_here_shared.sh:if [ ${builddep} -eq 1 -o ${buildpetsccudadbl} -eq 1 ]; then 
legacy-build/deps/build_libs_here_shared.sh:	echo "configuring PETSC-CUDA (double precision)"
legacy-build/deps/build_libs_here_shared.sh:	echo ./configure PETSC_DIR=${SRC_DIR} PETSC_ARCH=${PETSC_ARCH} --prefix=${BLD_DIR}/${PETSC_ARCH} ${PETSC_OPTIONS} ${PETSC_CUDA_OPTIONS}
legacy-build/deps/build_libs_here_shared.sh:	./configure PETSC_DIR=${SRC_DIR} PETSC_ARCH=${PETSC_ARCH} --prefix=${BLD_DIR}/${PETSC_ARCH} ${PETSC_OPTIONS} ${PETSC_CUDA_OPTIONS}
legacy-build/deps/build_libs_here_shared.sh:echo "export PETSC_ARCH_CUDA_DOUBLE=${PETSC_ARCH}" >> ${BUILD_DIR}/environment_vars.sh
legacy-build/deps/build_libs_here_shared.sh:# PETSC-CUDA DBG SINGLE
legacy-build/deps/build_libs_here_shared.sh:PETSC_ARCH=cuda_opt_dbg_sgl
legacy-build/deps/build_libs_here_shared.sh:if [ ${builddep} -eq 1 -o ${buildpetsccudasgldbg} -eq 1 ]; then 
legacy-build/deps/build_libs_here_shared.sh:	echo "configuring PETSC-CUDA (single precision; debug mode)"
legacy-build/deps/build_libs_here_shared.sh:	echo ./configure PETSC_DIR=${SRC_DIR} PETSC_ARCH=${PETSC_ARCH} --prefix=${BLD_DIR}/${PETSC_ARCH} ${PETSC_DBG_OPTIONS} ${PETSC_CUDA_OPTIONS} --with-precision=single
legacy-build/deps/build_libs_here_shared.sh:	./configure PETSC_DIR=${SRC_DIR} PETSC_ARCH=${PETSC_ARCH} --prefix=${BLD_DIR}/${PETSC_ARCH} ${PETSC_DBG_OPTIONS} ${PETSC_CUDA_OPTIONS} --with-precision=single
legacy-build/deps/build_libs_here_shared.sh:echo "export PETSC_ARCH_CUDA_SINGLE_DBG=${PETSC_ARCH}" >> ${BUILD_DIR}/environment_vars.sh
legacy-build/deps/build_libs_here_shared.sh:# PETSC-CUDA DBG DOUBLE
legacy-build/deps/build_libs_here_shared.sh:PETSC_ARCH=cuda_opt_dbg_dbl
legacy-build/deps/build_libs_here_shared.sh:if [ ${builddep} -eq 1 -o ${buildpetsccudadbldbg} -eq 1 ]; then 
legacy-build/deps/build_libs_here_shared.sh:	echo "configuring PETSC-CUDA (double precision; debug mode)"
legacy-build/deps/build_libs_here_shared.sh:	echo ./configure PETSC_DIR=${SRC_DIR} PETSC_ARCH=${PETSC_ARCH} --prefix=${BLD_DIR}/${PETSC_ARCH} ${PETSC_DBG_OPTIONS} ${PETSC_CUDA_OPTIONS}
legacy-build/deps/build_libs_here_shared.sh:	./configure PETSC_DIR=${SRC_DIR} PETSC_ARCH=${PETSC_ARCH} --prefix=${BLD_DIR}/${PETSC_ARCH} ${PETSC_DBG_OPTIONS} ${PETSC_CUDA_OPTIONS}
legacy-build/deps/build_libs_here_shared.sh:echo "export PETSC_ARCH_CUDA_DOUBLE_DBG=${PETSC_ARCH}" >> ${BUILD_DIR}/environment_vars.sh
legacy-build/deps/build_libs_here_shared.sh:echo "export CUDA_DIR=/usr/local/cuda-10.1" >> ${BUILD_DIR}/environment_vars.sh
legacy-build/deps/build_libs.sh:CUDA_C=nvcc
legacy-build/deps/build_libs.sh:enableCUDA=0
legacy-build/deps/build_libs.sh:GPU="V100" # TACC Longhorn
legacy-build/deps/build_libs.sh:#GPU="RTX" # TACC Frontera
legacy-build/deps/build_libs.sh:#GPU="P100" # TACC Maverick2
legacy-build/deps/build_libs.sh:buildpetsccudasgl=0
legacy-build/deps/build_libs.sh:buildpetsccudadbl=0
legacy-build/deps/build_libs.sh:buildpetsccudasgldbg=0
legacy-build/deps/build_libs.sh:buildpetsccudadbldbg=0
legacy-build/deps/build_libs.sh:    --gpu=*)
legacy-build/deps/build_libs.sh:    GPU="${i#*=}"
legacy-build/deps/build_libs.sh:    --enableCUDA)
legacy-build/deps/build_libs.sh:    enableCUDA=1
legacy-build/deps/build_libs.sh:    --bpetsccudasgl)
legacy-build/deps/build_libs.sh:    buildpetsccudasgl=1
legacy-build/deps/build_libs.sh:    --bpetsccudasgldbg)
legacy-build/deps/build_libs.sh:    buildpetsccudasgldbg=1
legacy-build/deps/build_libs.sh:    --bpetsccudadbl)
legacy-build/deps/build_libs.sh:    buildpetsccudadbl=1
legacy-build/deps/build_libs.sh:    --bpetsccudadbldbg)
legacy-build/deps/build_libs.sh:    buildpetsccudadbldbg=1
legacy-build/deps/build_libs.sh:    echo "     --enableCUDA    flag: use CUDA for AccFFT"
legacy-build/deps/build_libs.sh:    echo "     --bpetsccudasgl build PETSc library with nVidia-CUDA library (single precision)"
legacy-build/deps/build_libs.sh:    echo "     --bpetsccudasgldbg build PETSc library with nVidia-CUDA library (single precision; debug mode)"
legacy-build/deps/build_libs.sh:-DBUILD_GPU=false
legacy-build/deps/build_libs.sh:if [ ${enableCUDA} -eq 1 ]; then
legacy-build/deps/build_libs.sh:  if [ "$GPU" == "V100" ]; then
legacy-build/deps/build_libs.sh:    ACCFFT_OPTIONS="${ACCFFT_OPTIONS} -DBUILD_GPU=true -DCUDA_NVCC_FLAGS=-gencode;arch=compute_70,code=sm_70"
legacy-build/deps/build_libs.sh:  if [ "$GPU" == "P100" ]; then
legacy-build/deps/build_libs.sh:    ACCFFT_OPTIONS="${ACCFFT_OPTIONS} -DBUILD_GPU=true -DCUDA_NVCC_FLAGS=-gencode;arch=compute_60,code=sm_60"
legacy-build/deps/build_libs.sh:  if [ "$GPU" == "RTX" ]; then
legacy-build/deps/build_libs.sh:    ACCFFT_OPTIONS="${ACCFFT_OPTIONS} -DBUILD_GPU=true -DCUDA_NVCC_FLAGS=-gencode;arch=compute_75,code=sm_75"
legacy-build/deps/build_libs.sh:if [ ${enableCUDA} -eq 1 ]; then
legacy-build/deps/build_libs.sh:  CUDA_DIR=$(which ${CUDA_C})
legacy-build/deps/build_libs.sh:  CUDA_DIR=$(dirname "${CUDA_DIR}")
legacy-build/deps/build_libs.sh:  CUDA_DIR=$(dirname "${CUDA_DIR}")
legacy-build/deps/build_libs.sh:  echo " detected CUDA-toolkit directory: ${CUDA_DIR}"
legacy-build/deps/build_libs.sh:  echo "export CUDA_DIR=${CUDA_DIR}" >> ${BUILD_DIR}/environment_vars.sh
legacy-build/deps/build_libs.sh:  echo "export LD_LIBRARY_PATH=${CUDA_DIR}/lib64:\${LD_LIBRARY_PATH}" >> ${BUILD_DIR}/environment_vars.sh
legacy-build/deps/build_libs.sh:	python2 config/examples/configure_petsc.py $PETSC_ARCH $GPU
legacy-build/deps/build_libs.sh:# PETSC-CUDA-SINGLE-PRECISION
legacy-build/deps/build_libs.sh:PETSC_ARCH=cuda-opt-sgl-${GPU}
legacy-build/deps/build_libs.sh:if [ ${builddep} -eq 1 -o ${buildpetsccudasgl} -eq 1 ]; then 
legacy-build/deps/build_libs.sh:	build_petsc $PETSC_ARCH "configuring PETSC-CUDA (single precision)"
legacy-build/deps/build_libs.sh:echo "export PETSC_ARCH_CUDA_SINGLE=${PETSC_ARCH}" >> ${BUILD_DIR}/environment_vars.sh
legacy-build/deps/build_libs.sh:# PETSC-CUDA DBG SINGLE
legacy-build/deps/build_libs.sh:PETSC_ARCH=cuda-opt-sgl-dbg-${GPU}
legacy-build/deps/build_libs.sh:if [ ${builddep} -eq 1 -o ${buildpetsccudasgldbg} -eq 1 ]; then 
legacy-build/deps/build_libs.sh:	build_petsc $PETSC_ARCH "configuring PETSC-cuda (single precision) in debug mode"
legacy-build/deps/build_libs.sh:echo "export PETSC_ARCH_CUDA_SINGLE_DBG=${PETSC_ARCH}" >> ${BUILD_DIR}/environment_vars.sh
legacy-build/deps/build_libs_here.sh:enableCUDA=0
legacy-build/deps/build_libs_here.sh:buildpetsccudasgl=0
legacy-build/deps/build_libs_here.sh:buildpetsccudadbl=0
legacy-build/deps/build_libs_here.sh:buildpetsccudasgldbg=0
legacy-build/deps/build_libs_here.sh:buildpetsccudadbldbg=0
legacy-build/deps/build_libs_here.sh:    --enableCUDA)
legacy-build/deps/build_libs_here.sh:    enableCUDA=1
legacy-build/deps/build_libs_here.sh:    --bpetsccudasgl)
legacy-build/deps/build_libs_here.sh:    buildpetsccudasgl=1
legacy-build/deps/build_libs_here.sh:    --bpetsccudasgldbg)
legacy-build/deps/build_libs_here.sh:    buildpetsccudasgldbg=1
legacy-build/deps/build_libs_here.sh:    --bpetsccudadbl)
legacy-build/deps/build_libs_here.sh:    buildpetsccudadbl=1
legacy-build/deps/build_libs_here.sh:    --bpetsccudadbldbg)
legacy-build/deps/build_libs_here.sh:    buildpetsccudadbldbg=1
legacy-build/deps/build_libs_here.sh:    echo "     --enableCUDA    flag: use CUDA for AccFFT"
legacy-build/deps/build_libs_here.sh:    echo "     --bpetsccudasgl build PETSc library with nVidia-CUDA library (single precision)"
legacy-build/deps/build_libs_here.sh:    echo "     --bpetsccudasgldbg build PETSc library with nVidia-CUDA library (single precision; debug mode)"
legacy-build/deps/build_libs_here.sh:PETSC_CUDA_OPTIONS="
legacy-build/deps/build_libs_here.sh:--with-cuda=1
legacy-build/deps/build_libs_here.sh:--CUDAFLAGS='-arch=sm_35'
legacy-build/deps/build_libs_here.sh:--CUDAOPTFLAGS='-O3'"
legacy-build/deps/build_libs_here.sh:#-use-gpu-aware-mpi"
legacy-build/deps/build_libs_here.sh:-DBUILD_GPU=false
legacy-build/deps/build_libs_here.sh:if [ ${enableCUDA} -eq 1 ]; then
legacy-build/deps/build_libs_here.sh:	ACCFFT_OPTIONS="${ACCFFT_OPTIONS} -DBUILD_GPU=true"
legacy-build/deps/build_libs_here.sh:# PETSC-CUDA
legacy-build/deps/build_libs_here.sh:PETSC_ARCH=cuda_opt_sgl
legacy-build/deps/build_libs_here.sh:if [ ${builddep} -eq 1 -o ${buildpetsccudasgl} -eq 1 ]; then 
legacy-build/deps/build_libs_here.sh:	echo "configuring PETSC-CUDA (single precision)"
legacy-build/deps/build_libs_here.sh:	echo ./configure PETSC_DIR=${SRC_DIR} PETSC_ARCH=${PETSC_ARCH} --prefix=${BLD_DIR}/${PETSC_ARCH} ${PETSC_OPTIONS} ${PETSC_CUDA_OPTIONS} --with-precision=single
legacy-build/deps/build_libs_here.sh:	./configure PETSC_DIR=${SRC_DIR} PETSC_ARCH=${PETSC_ARCH} --prefix=${BLD_DIR}/${PETSC_ARCH} ${PETSC_OPTIONS} ${PETSC_CUDA_OPTIONS} --with-precision=single
legacy-build/deps/build_libs_here.sh:echo "export PETSC_ARCH_CUDA_SINGLE=${PETSC_ARCH}" >> ${BUILD_DIR}/environment_vars.sh
legacy-build/deps/build_libs_here.sh:# PETSC-CUDA
legacy-build/deps/build_libs_here.sh:PETSC_ARCH=cuda_opt_dbl
legacy-build/deps/build_libs_here.sh:if [ ${builddep} -eq 1 -o ${buildpetsccudadbl} -eq 1 ]; then 
legacy-build/deps/build_libs_here.sh:	echo "configuring PETSC-CUDA (double precision)"
legacy-build/deps/build_libs_here.sh:	echo ./configure PETSC_DIR=${SRC_DIR} PETSC_ARCH=${PETSC_ARCH} --prefix=${BLD_DIR}/${PETSC_ARCH} ${PETSC_OPTIONS} ${PETSC_CUDA_OPTIONS}
legacy-build/deps/build_libs_here.sh:	./configure PETSC_DIR=${SRC_DIR} PETSC_ARCH=${PETSC_ARCH} --prefix=${BLD_DIR}/${PETSC_ARCH} ${PETSC_OPTIONS} ${PETSC_CUDA_OPTIONS}
legacy-build/deps/build_libs_here.sh:echo "export PETSC_ARCH_CUDA_DOUBLE=${PETSC_ARCH}" >> ${BUILD_DIR}/environment_vars.sh
legacy-build/deps/build_libs_here.sh:# PETSC-CUDA DBG SINGLE
legacy-build/deps/build_libs_here.sh:PETSC_ARCH=cuda_opt_dbg_sgl
legacy-build/deps/build_libs_here.sh:if [ ${builddep} -eq 1 -o ${buildpetsccudasgldbg} -eq 1 ]; then 
legacy-build/deps/build_libs_here.sh:	echo "configuring PETSC-CUDA (single precision; debug mode)"
legacy-build/deps/build_libs_here.sh:	echo ./configure PETSC_DIR=${SRC_DIR} PETSC_ARCH=${PETSC_ARCH} --prefix=${BLD_DIR}/${PETSC_ARCH} ${PETSC_DBG_OPTIONS} ${PETSC_CUDA_OPTIONS} --with-precision=single
legacy-build/deps/build_libs_here.sh:	./configure PETSC_DIR=${SRC_DIR} PETSC_ARCH=${PETSC_ARCH} --prefix=${BLD_DIR}/${PETSC_ARCH} ${PETSC_DBG_OPTIONS} ${PETSC_CUDA_OPTIONS} --with-precision=single
legacy-build/deps/build_libs_here.sh:echo "export PETSC_ARCH_CUDA_SINGLE_DBG=${PETSC_ARCH}" >> ${BUILD_DIR}/environment_vars.sh
legacy-build/deps/build_libs_here.sh:# PETSC-CUDA DBG DOUBLE
legacy-build/deps/build_libs_here.sh:PETSC_ARCH=cuda_opt_dbg_dbl
legacy-build/deps/build_libs_here.sh:if [ ${builddep} -eq 1 -o ${buildpetsccudadbldbg} -eq 1 ]; then 
legacy-build/deps/build_libs_here.sh:	echo "configuring PETSC-CUDA (double precision; debug mode)"
legacy-build/deps/build_libs_here.sh:	echo ./configure PETSC_DIR=${SRC_DIR} PETSC_ARCH=${PETSC_ARCH} --prefix=${BLD_DIR}/${PETSC_ARCH} ${PETSC_DBG_OPTIONS} ${PETSC_CUDA_OPTIONS}
legacy-build/deps/build_libs_here.sh:	./configure PETSC_DIR=${SRC_DIR} PETSC_ARCH=${PETSC_ARCH} --prefix=${BLD_DIR}/${PETSC_ARCH} ${PETSC_DBG_OPTIONS} ${PETSC_CUDA_OPTIONS}
legacy-build/deps/build_libs_here.sh:echo "export PETSC_ARCH_CUDA_DOUBLE_DBG=${PETSC_ARCH}" >> ${BUILD_DIR}/environment_vars.sh
legacy-build/deps/configure_petsc.py:  gpu=sys.argv[2]
legacy-build/deps/configure_petsc.py:  #if "cuda" in mode:
legacy-build/deps/configure_petsc.py:  #  mode = mode + "-" + gpu
legacy-build/deps/configure_petsc.py:  if "cuda" in mode:
legacy-build/deps/configure_petsc.py:    #configure_options.append('--with-cuda-dir='+os.environ['TACC_CUDA_DIR'])
legacy-build/deps/configure_petsc.py:    configure_options.append('--with-cuda=1')
legacy-build/deps/configure_petsc.py:    configure_options.append('--with-cudac=nvcc')
legacy-build/deps/configure_petsc.py:    if gpu == "V100":
legacy-build/deps/configure_petsc.py:      configure_options.append('CUDAFLAGS="-arch=sm_70"')
legacy-build/deps/configure_petsc.py:    elif gpu == "P100":
legacy-build/deps/configure_petsc.py:      configure_options.append('CUDAFLAGS="-arch=sm_60"')
legacy-build/deps/configure_petsc.py:    elif gpu == "RTX":
legacy-build/deps/configure_petsc.py:      configure_options.append('CUDAFLAGS="-arch=sm_75"')
legacy-build/deps/configure_petsc.py:      configure_options.append('CUDAFLAGS="-arch=sm_70"')
legacy-build/deps/configure_petsc.py:    configure_options.append('CUDAOPTFLAGS="-O3"')
legacy-build/makefile:USECUDA=no
legacy-build/makefile:USECUDADBG=no
legacy-build/makefile:ifeq ($(USECUDA),yes)
legacy-build/makefile:CPPFILESCUDA += $(TESTFILES)
legacy-build/makefile:CUDAC=nvcc
legacy-build/makefile:CUDA_OBJS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/cuda/%.o,$(CUFILES))
legacy-build/makefile:CUDA_OBJS += $(patsubst $(EXSRCDIR)/%.cu,$(OBJDIR)/cuda/%.o,$(EXCUFILES))
legacy-build/makefile:OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(CPPFILESCUDA))
legacy-build/makefile:ifeq ($(USECUDA),yes)
legacy-build/makefile:#$(BINDIR)/%: $(OBJDIR)/%.o $(CUDA_OBJS) $(OBJS)
legacy-build/makefile:$(BINDIR)/%: $(OBJDIR)/bin/%.o $(CUDA_OBJS) $(OBJS)
legacy-build/makefile:$(OBJDIR)/cuda/%.o: $(SRCDIR)/%.cu
legacy-build/makefile:	$(CUDAC) $(CUDA_FLAGS) $(CUDA_INC) -c $^ -o $@
legacy-build/makefile:$(OBJDIR)/cuda/%.o: $(EXSRCDIR)/%.cu
legacy-build/makefile:	$(CUDAC) $(CUDA_FLAGS) $(CUDA_INC) -c $^ -o $@
legacy-build/makefile_dbl:USECUDA=no
legacy-build/makefile_dbl:USECUDADBG=no
legacy-build/makefile_dbl:ifeq ($(USECUDA),yes)
legacy-build/makefile_dbl:CPPFILESCUDA += $(TESTFILES)
legacy-build/makefile_dbl:CUDAC=nvcc
legacy-build/makefile_dbl:CUDA_OBJS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/cuda/%.o,$(CUFILES))
legacy-build/makefile_dbl:CUDA_OBJS += $(patsubst $(EXSRCDIR)/%.cu,$(OBJDIR)/cuda/%.o,$(EXCUFILES))
legacy-build/makefile_dbl:OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(CPPFILESCUDA))
legacy-build/makefile_dbl:ifeq ($(USECUDA),yes)
legacy-build/makefile_dbl:#$(BINDIR)/%: $(OBJDIR)/%.o $(CUDA_OBJS) $(OBJS)
legacy-build/makefile_dbl:$(BINDIR)/%: $(OBJDIR)/bin/%.o $(CUDA_OBJS) $(OBJS)
legacy-build/makefile_dbl:$(OBJDIR)/cuda/%.o: $(SRCDIR)/%.cu
legacy-build/makefile_dbl:	$(CUDAC) $(CUDA_FLAGS) $(CUDA_INC) -c $^ -o $@
legacy-build/makefile_dbl:$(OBJDIR)/cuda/%.o: $(EXSRCDIR)/%.cu
legacy-build/makefile_dbl:	$(CUDAC) $(CUDA_FLAGS) $(CUDA_INC) -c $^ -o $@
legacy-build/makefile_gpu:USECUDA=yes
legacy-build/makefile_gpu:USECUDADBG=no
legacy-build/makefile_gpu:ifeq ($(USECUDA),yes)
legacy-build/makefile_gpu:CPPFILESCUDA += $(TESTFILES)
legacy-build/makefile_gpu:CUDAC=nvcc
legacy-build/makefile_gpu:CUDA_OBJS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/cuda/%.o,$(CUFILES))
legacy-build/makefile_gpu:CUDA_OBJS += $(patsubst $(EXSRCDIR)/%.cu,$(OBJDIR)/cuda/%.o,$(EXCUFILES))
legacy-build/makefile_gpu:OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(CPPFILESCUDA))
legacy-build/makefile_gpu:ifeq ($(USECUDA),yes)
legacy-build/makefile_gpu:lib/libclaire.so: $(CUDA_OBJS) $(OBJS)
legacy-build/makefile_gpu:#$(BINDIR)/%: $(OBJDIR)/%.o $(CUDA_OBJS) $(OBJS)
legacy-build/makefile_gpu:#$(BINDIR)/%: $(OBJDIR)/bin/%.o $(CUDA_OBJS) $(OBJS)
legacy-build/makefile_gpu:$(BINDIR)/%: $(OBJDIR)/bin/%.o $(CUDA_OBJS) $(OBJS)
legacy-build/makefile_gpu:$(OBJDIR)/cuda/%.o: $(SRCDIR)/%.cu
legacy-build/makefile_gpu:	$(CUDAC) $(CUDA_FLAGS) $(CUDA_INC) -c $^ -o $@
legacy-build/makefile_gpu:$(OBJDIR)/cuda/%.o: $(EXSRCDIR)/%.cu
legacy-build/makefile_gpu:	$(CUDAC) $(CUDA_FLAGS) $(CUDA_INC) -c $^ -o $@
legacy-build/makefile_p9:USECUDA=yes
legacy-build/makefile_p9:USEMPICUDA=yes
legacy-build/makefile_p9:USECUDADBG=no
legacy-build/makefile_p9:ifeq ($(USECUDA),yes)
legacy-build/makefile_p9:CPPFILESCUDA += $(TESTFILES)
legacy-build/makefile_p9:CUDAC=nvcc
legacy-build/makefile_p9:CUDA_OBJS = $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/cuda/%.o,$(CUFILES))
legacy-build/makefile_p9:CUDA_OBJS += $(patsubst $(EXSRCDIR)/%.cu,$(OBJDIR)/cuda/%.o,$(EXCUFILES))
legacy-build/makefile_p9:OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(CPPFILESCUDA))
legacy-build/makefile_p9:ifeq ($(USECUDA),yes)
legacy-build/makefile_p9:#$(BINDIR)/%: $(OBJDIR)/%.o $(CUDA_OBJS) $(OBJS)
legacy-build/makefile_p9:$(BINDIR)/%: $(OBJDIR)/bin/%.o $(CUDA_OBJS) $(OBJS)
legacy-build/makefile_p9:$(OBJDIR)/cuda/%.o: $(SRCDIR)/%.cu
legacy-build/makefile_p9:	$(CUDAC) $(CUDA_FLAGS) $(CUDA_INC) -c $^ -o $@
legacy-build/makefile_p9:$(OBJDIR)/cuda/%.o: $(EXSRCDIR)/%.cu
legacy-build/makefile_p9:	$(CUDAC) $(CUDA_FLAGS) $(CUDA_INC) -c $^ -o $@
.gitignore:bingpu/
.gitignore:bingpu/
src/UnitTestOpt.cpp:    ierr = UnitTest::TestInterpolationMultiGPU(this); CHKERRQ(ierr);
src/UnitTestOpt.cpp:      ierr = UnitTest::TestInterpolationMultiGPU(this); CHKERRQ(ierr);
src/UnitTestOpt.cpp:    ierr = UnitTest::TestTrajectoryMultiGPU(this); CHKERRQ(ierr);
src/UnitTestOpt.cpp:    //ierr = UnitTest::TestDifferentiationMultiGPU(this); CHKERRQ(ierr);
src/VecField.cpp:    #ifdef REG_HAS_CUDA
src/VecField.cpp:        VecCreateMPICUDAWithArray(PETSC_COMM_WORLD, 1, nl, ng, &this->m_RawPtr[0*nl], &this->m_X1);
src/VecField.cpp:        VecCreateMPICUDAWithArray(PETSC_COMM_WORLD, 1, nl, ng, &this->m_RawPtr[1*nl], &this->m_X2);
src/VecField.cpp:        VecCreateMPICUDAWithArray(PETSC_COMM_WORLD, 1, nl, ng, &this->m_RawPtr[2*nl], &this->m_X3);
src/VecField.cpp:        VecCreateSeqCUDAWithArray(PETSC_COMM_WORLD, 1, nl, &this->m_RawPtr[0*nl], &this->m_X1);
src/VecField.cpp:        VecCreateSeqCUDAWithArray(PETSC_COMM_WORLD, 1, nl, &this->m_RawPtr[1*nl], &this->m_X2);
src/VecField.cpp:        VecCreateSeqCUDAWithArray(PETSC_COMM_WORLD, 1, nl, &this->m_RawPtr[2*nl], &this->m_X3);
src/VecField.cpp:      //ierr = VecSetType(this->m_X1, VECCUDA); CHKERRQ(ierr);
src/VecField.cpp:      //ierr = VecSetType(this->m_X2, VECCUDA); CHKERRQ(ierr);
src/VecField.cpp:      //ierr = VecSetType(this->m_X3, VECCUDA); CHKERRQ(ierr);
src/VecField.cpp:    #ifdef REG_HAS_CUDA
src/VecField.cpp:        ierr = VecCreateMPICUDAWithArray(PETSC_COMM_WORLD, 1, nl, ng, &this->m_RawPtr[0*nl], &this->m_X1);
src/VecField.cpp:        ierr = VecCreateMPICUDAWithArray(PETSC_COMM_WORLD, 1, nl, ng, &this->m_RawPtr[1*nl], &this->m_X2);
src/VecField.cpp:        ierr = VecCreateMPICUDAWithArray(PETSC_COMM_WORLD, 1, nl, ng, &this->m_RawPtr[2*nl], &this->m_X3);
src/VecField.cpp:        ierr = VecCreateSeqCUDAWithArray(PETSC_COMM_WORLD, 1, nl, &this->m_RawPtr[0*nl], &this->m_X1);
src/VecField.cpp:        ierr = VecCreateSeqCUDAWithArray(PETSC_COMM_WORLD, 1, nl, &this->m_RawPtr[1*nl], &this->m_X2);
src/VecField.cpp:        ierr = VecCreateSeqCUDAWithArray(PETSC_COMM_WORLD, 1, nl, &this->m_RawPtr[2*nl], &this->m_X3);
src/VecField.cpp:      //ierr = VecSetType(this->m_X1, VECCUDA); CHKERRQ(ierr);
src/VecField.cpp:      //ierr = VecSetType(this->m_X2, VECCUDA); CHKERRQ(ierr);
src/VecField.cpp:      //ierr = VecSetType(this->m_X3, VECCUDA); CHKERRQ(ierr);
src/VecField.cpp:    #ifdef REG_HAS_CUDA
src/VecField.cpp:        ierr = VecSetType(this->m_X, VECCUDA); CHKERRQ(ierr);
src/VecField.cpp:    #ifdef REG_HAS_CUDA
src/VecField.cpp:        ierr = VecCreateMPICUDAWithArray(PETSC_COMM_WORLD, 1, nl, ng, &this->m_RawPtr[0*nl], &this->m_X1);
src/VecField.cpp:        ierr = VecCreateMPICUDAWithArray(PETSC_COMM_WORLD, 1, nl, ng, &this->m_RawPtr[1*nl], &this->m_X2);
src/VecField.cpp:        ierr = VecCreateMPICUDAWithArray(PETSC_COMM_WORLD, 1, nl, ng, &this->m_RawPtr[2*nl], &this->m_X3);
src/VecField.cpp:        ierr = VecCreateSeqCUDAWithArray(PETSC_COMM_WORLD, 1, nl, &this->m_RawPtr[0*nl], &this->m_X1);
src/VecField.cpp:        ierr = VecCreateSeqCUDAWithArray(PETSC_COMM_WORLD, 1, nl, &this->m_RawPtr[1*nl], &this->m_X2);
src/VecField.cpp:        ierr = VecCreateSeqCUDAWithArray(PETSC_COMM_WORLD, 1, nl, &this->m_RawPtr[2*nl], &this->m_X3);
src/VecField.cpp:      //ierr = VecSetType(this->m_X1, VECCUDA); CHKERRQ(ierr);
src/VecField.cpp:      //ierr = VecSetType(this->m_X2, VECCUDA); CHKERRQ(ierr);
src/VecField.cpp:      //ierr = VecSetType(this->m_X3, VECCUDA); CHKERRQ(ierr);
src/VecField.cpp:    #ifdef REG_HAS_CUDA
src/VecField.cpp:        ierr = VecSetType(this->m_X1, VECCUDA); CHKERRQ(ierr);
src/VecField.cpp:    #ifdef REG_HAS_CUDA
src/VecField.cpp:        ierr = VecSetType(this->m_X2, VECCUDA); CHKERRQ(ierr);
src/VecField.cpp:    #ifdef REG_HAS_CUDA
src/VecField.cpp:        ierr = VecSetType(this->m_X3, VECCUDA); CHKERRQ(ierr);
src/VecField.cpp:#ifdef REG_HAS_CUDA
src/VecField.cpp:#if defined(REG_HAS_CUDA) || defined(REG_HAS_MPICUDA)
src/VecField.cpp:      ierr = cudaMemcpy(static_cast<void*>(p_x1), static_cast<const void*>(pX), 
src/VecField.cpp:                sizeof(ScalarType)*nl, cudaMemcpyDeviceToDevice); CHKERRCUDA(ierr);
src/VecField.cpp:      ierr = cudaMemcpy(static_cast<void*>(p_x2), static_cast<const void*>(&pX[nl]),
src/VecField.cpp:                sizeof(ScalarType)*nl, cudaMemcpyDeviceToDevice); CHKERRCUDA(ierr);
src/VecField.cpp:      ierr = cudaMemcpy(static_cast<void*>(p_x3), static_cast<const void*>(&pX[2*nl]),
src/VecField.cpp:                sizeof(ScalarType)*nl, cudaMemcpyDeviceToDevice); CHKERRCUDA(ierr);
src/VecField.cpp:      ierr = cudaDeviceSynchronize(); CHKERRCUDA(ierr);
src/VecField.cpp:#if defined(REG_HAS_CUDA) || defined(REG_HAS_MPICUDA)
src/VecField.cpp:      ierr = cudaMemcpy(static_cast<void*>(p_x1), static_cast<const void*>(pX1), 
src/VecField.cpp:                sizeof(ScalarType)*nl, cudaMemcpyDeviceToDevice); CHKERRCUDA(ierr);
src/VecField.cpp:      ierr = cudaMemcpy(static_cast<void*>(p_x2), static_cast<const void*>(pX2),
src/VecField.cpp:                sizeof(ScalarType)*nl, cudaMemcpyDeviceToDevice); CHKERRCUDA(ierr);
src/VecField.cpp:      ierr = cudaMemcpy(static_cast<void*>(p_x3), static_cast<const void*>(pX3),
src/VecField.cpp:                sizeof(ScalarType)*nl, cudaMemcpyDeviceToDevice); CHKERRCUDA(ierr);
src/VecField.cpp:#if defined(REG_HAS_CUDA) || defined(REG_HAS_MPICUDA)
src/VecField.cpp:      ierr = cudaMemcpy(static_cast<void*>(pX), static_cast<const void*>(p_x1), 
src/VecField.cpp:                sizeof(ScalarType)*nl, cudaMemcpyDeviceToDevice); CHKERRCUDA(ierr);
src/VecField.cpp:      ierr = cudaMemcpy(static_cast<void*>(&pX[nl]), static_cast<const void*>(p_x2), 
src/VecField.cpp:                sizeof(ScalarType)*nl, cudaMemcpyDeviceToDevice); CHKERRCUDA(ierr);
src/VecField.cpp:      ierr = cudaMemcpy(static_cast<void*>(&pX[2*nl]), static_cast<const void*>(p_x3), 
src/VecField.cpp:                sizeof(ScalarType)*nl, cudaMemcpyDeviceToDevice); CHKERRCUDA(ierr);
src/VecField.cpp:      ierr = CopyStridedToFlatVec(pX, p_x1, p_x2, p_x3, nl); CHKERRCUDA(ierr);
src/VecField.cpp:#if defined(REG_HAS_CUDA) || defined(REG_HAS_MPICUDA)
src/VecField.cpp:    ierr = cudaMemcpy(static_cast<void*>(pX1), static_cast<const void*>(p_x1), 
src/VecField.cpp:                sizeof(ScalarType)*nl, cudaMemcpyDeviceToDevice); CHKERRCUDA(ierr);
src/VecField.cpp:    ierr = cudaMemcpy(static_cast<void*>(pX2), static_cast<const void*>(p_x2), 
src/VecField.cpp:                sizeof(ScalarType)*nl, cudaMemcpyDeviceToDevice); CHKERRCUDA(ierr);
src/VecField.cpp:    ierr = cudaMemcpy(static_cast<void*>(pX3), static_cast<const void*>(p_x3), 
src/VecField.cpp:                sizeof(ScalarType)*nl, cudaMemcpyDeviceToDevice); CHKERRCUDA(ierr);
src/VecField.cpp:#ifdef REG_HAS_CUDA
src/VecField.cpp:    ierr = WrngMsg("Not implemented for CUDA"); CHKERRQ(ierr);
src/VecField.cpp:#ifdef REG_HAS_CUDA
src/VecField.cpp:    ierr = WrngMsg("Not implemented for CUDA"); CHKERRQ(ierr);
src/VecField.cpp:#ifdef REG_HAS_CUDA
src/VecField.cpp:    ierr = WrngMsg("Not implemented for CUDA"); CHKERRQ(ierr);
src/ScaField.cpp:    #ifdef REG_HAS_CUDA
src/ScaField.cpp:        ierr = VecSetType(this->m_X, VECCUDA); CHKERRQ(ierr);
src/ScaField.cpp:#ifndef REG_HAS_CUDA
src/ScaField.cpp:  ierr = cudaMemcpy((void*)ptr,(void*)orig,sizeof(ScalarType)*this->m_Size[1],cudaMemcpyDeviceToDevice); CHKERRCUDA(ierr);
src/ScaField.cpp:#ifdef REG_HAS_CUDA
src/ScaField.cpp:  cudaDeviceSynchronize();
src/ScaField.cpp:#ifndef REG_HAS_CUDA
src/ScaField.cpp:    ierr = cudaMemcpy((void*)ptr,(void*)orig,sizeof(ScalarType)*this->m_Size[1],cudaMemcpyDeviceToDevice); CHKERRCUDA(ierr);
src/ScaField.cpp:#ifdef REG_HAS_CUDA
src/ScaField.cpp:  cudaDeviceSynchronize();
src/ScaField.cpp:#ifndef REG_HAS_CUDA
src/ScaField.cpp:    cudaMemcpy((void*)dest,(void*)orig,sizeof(ScalarType)*nl,cudaMemcpyDeviceToDevice);
src/ScaField.cpp:#ifndef REG_HAS_CUDA
src/ScaField.cpp:    cudaMemcpy((void*)dest,(void*)orig,sizeof(ScalarType)*this->m_Size[1],cudaMemcpyDeviceToDevice);
src/ScaField.cpp:#ifdef REG_HAS_CUDA
src/ScaField.cpp:  cudaDeviceSynchronize();
src/Preprocessing.cpp:#ifndef REG_HAS_CUDA
src/Preprocessing.cpp:#ifndef REG_HAS_CUDA
src/Preprocessing.cpp:#ifndef REG_HAS_CUDA
src/CLAIREUtils.cpp://#include "cuda_helper.hpp"
src/CLAIREUtils.cpp:#if defined(REG_HAS_CUDA) || defined(REG_HAS_MPICUDA)
src/CLAIREUtils.cpp:    ierr = reg::VecFieldPointWiseNormGPU(p_m, p_X1, p_X2, p_X3, nl); CHKERRQ(ierr);    
src/CLAIREUtils.cpp:    //cudaPrintDeviceMemory();
src/CLAIREUtils.cpp:#if defined(REG_HAS_MPICUDA)
src/CLAIREUtils.cpp:    cudaMemcpy((void*)dst , (const void*)src, size, cudaMemcpyDeviceToDevice);
src/CLAIREUtils.cpp:#if defined(REG_HAS_CUDA)
src/CLAIREUtils.cpp:#if !defined(REG_HAS_CUDA)
src/CLAIREUtils.cpp:#if defined(REG_HAS_CUDA) || defined(REG_HAS_MPICUDA)
src/CLAIREUtils.cpp:    ierr = VecSetType(x, VECCUDA); CHKERRQ(ierr);
src/CLAIREUtils.cpp:    #if defined(REG_HAS_CUDA)
src/CLAIREUtils.cpp:        ierr = VecCUDAGetArray(v, a); CHKERRQ(ierr);
src/CLAIREUtils.cpp:    #if defined(REG_HAS_CUDA)
src/CLAIREUtils.cpp:        ierr = VecCUDARestoreArray(v, a); CHKERRQ(ierr);
src/CLAIREUtils.cpp:    #if defined(REG_HAS_CUDA)
src/CLAIREUtils.cpp:        ierr = VecCUDAGetArrayRead(v, a); CHKERRQ(ierr);
src/CLAIREUtils.cpp:    #if defined(REG_HAS_CUDA)
src/CLAIREUtils.cpp:        ierr = VecCUDARestoreArrayRead(v, a); CHKERRQ(ierr);
src/CLAIREUtils.cpp:    #ifdef REG_HAS_CUDA
src/CLAIREUtils.cpp:        ierr = VecCUDAGetArrayWrite(v, a); CHKERRQ(ierr);
src/CLAIREUtils.cpp:    #ifdef REG_HAS_CUDA
src/CLAIREUtils.cpp:        ierr = VecCUDARestoreArrayWrite(v, a); CHKERRQ(ierr);
src/Optimizer.cpp:#ifdef REG_HAS_CUDA
src/Optimizer.cpp:#include "cuda_helper.hpp"
src/Optimizer.cpp:        ierr = VecSetType(this->m_Solution, VECCUDA); CHKERRQ(ierr);
src/Optimizer.cpp:#ifdef REG_HAS_CUDA
src/Optimizer.cpp:          ierr = VecCreateMPICUDAWithArray(PETSC_COMM_WORLD, 1, nlu, ngu, ptr, &this->m_Solution);
src/Optimizer.cpp:          ierr = VecCreateSeqCUDAWithArray(PETSC_COMM_WORLD, 1, nlu, ptr, &this->m_Solution);
src/CLAIREUtilsKernel.cu:#include "cuda_helper.hpp"
src/CLAIREUtilsKernel.cu:// CUDA kernel to evaluate point-wise norm of a vector field
src/CLAIREUtilsKernel.cu:  cudaDeviceSynchronize();
src/CLAIREUtilsKernel.cu:  cudaCheckKernelError();
src/CLAIREUtilsKernel.cu:PetscErrorCode VecFieldPointWiseNormGPU(ScalarType* p_m, const ScalarType* p_X1, const ScalarType* p_X2, const ScalarType* p_X3, IntType nl) {
src/CLAIREUtilsKernel.cu:    cudaDeviceSynchronize();
src/CLAIREUtilsKernel.cu:    cudaCheckKernelError();
src/CLAIREUtilsKernel.cu:    cudaDeviceSynchronize();
src/CLAIREUtilsKernel.cu:    cudaCheckKernelError();
src/CLAIREUtilsKernel.cu:    cudaDeviceSynchronize();
src/CLAIREUtilsKernel.cu:    cudaCheckKernelError();
src/SemiLagrangian/SemiLagrangian.cpp:#ifdef REG_HAS_CUDA
src/SemiLagrangian/SemiLagrangian.cpp:    cudaMemcpy(xi, xi_d, nl*sizeof(ScalarType), cudaMemcpyDeviceToHost);
src/SemiLagrangian/SemiLagrangian.cpp:#ifdef REG_HAS_CUDA
src/SemiLagrangian/SemiLagrangian.cpp:    cudaMemcpy(xo_d, xo, nl*sizeof(ScalarType), cudaMemcpyHostToDevice);
src/SemiLagrangian/SemiLagrangian.cpp:#ifdef REG_HAS_CUDA
src/SemiLagrangian/SemiLagrangian.cpp:    cudaMemcpy(&this->m_X[0*nl], vx1, nl*sizeof(ScalarType), cudaMemcpyDeviceToHost);
src/SemiLagrangian/SemiLagrangian.cpp:    cudaMemcpy(&this->m_X[1*nl], vx2, nl*sizeof(ScalarType), cudaMemcpyDeviceToHost);
src/SemiLagrangian/SemiLagrangian.cpp:    cudaMemcpy(&this->m_X[2*nl], vx3, nl*sizeof(ScalarType), cudaMemcpyDeviceToHost);
src/SemiLagrangian/SemiLagrangian.cpp:    cudaCheckLastError();
src/SemiLagrangian/SemiLagrangian.cpp:#ifdef REG_HAS_CUDA
src/SemiLagrangian/SemiLagrangian.cpp:    cudaMemcpy(wx1, &this->m_X[0*nl], nl*sizeof(ScalarType), cudaMemcpyHostToDevice);
src/SemiLagrangian/SemiLagrangian.cpp:    cudaMemcpy(wx2, &this->m_X[1*nl], nl*sizeof(ScalarType), cudaMemcpyHostToDevice);
src/SemiLagrangian/SemiLagrangian.cpp:    cudaMemcpy(wx3, &this->m_X[2*nl], nl*sizeof(ScalarType), cudaMemcpyHostToDevice);
src/SemiLagrangian/SemiLagrangianKernel.cu:#include "cuda_helper.hpp"
src/SemiLagrangian/SemiLagrangianKernel.cu:using KernelUtils::SpacialKernelCallGPU;
src/SemiLagrangian/SemiLagrangianKernel.cu:  ierr = SpacialKernelCallGPU<RK2Kernel>(istart, isize,
src/SemiLagrangian/SemiLagrangianKernel.cu:  ierr = SpacialKernelCallGPU<RK2Kernel>(istart, isize,
src/SemiLagrangian/SemiLagrangianKernel.cu:  ierr = SpacialKernelCallGPU<RK2Kernel>(istart, isize,
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:#ifndef _SEMILAGRANGIANGPUNEW_CPP_
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:#define _SEMILAGRANGIANGPUNEW_CPP_
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:#include "SemiLagrangianGPUNew.hpp"
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:SemiLagrangianGPUNew::SemiLagrangianGPUNew() {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:SemiLagrangianGPUNew::SemiLagrangianGPUNew(RegOpt* opt) {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:      DbgMsg("SemiLagrangianGPUNew created");
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:SemiLagrangianGPUNew::~SemiLagrangianGPUNew() {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::Initialize() { 
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:#ifdef REG_HAS_MPICUDA
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:    this->cuda_aware = true;
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:    this->cuda_aware = false;
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:      cudaMalloc((void**)&this->m_VecFieldGhost, this->g_alloc_max); 
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:      ierr = AllocateOnce(this->m_StatePlan, this->g_alloc_max, this->cuda_aware);
src/SemiLagrangian/SemiLagrangianGPUNew.cpp: * perform everything on the GPU
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::ClearMemory() {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:      cudaFree(this->m_WorkScaField1);
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:      cudaFree(this->m_WorkScaField2);
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:      cudaFree(this->m_ScaFieldGhost);
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:      cudaFree(this->m_VecFieldGhost); 
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:      cudaDestroyTextureObject(this->m_texture);
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:      cudaFree(this->m_tmpInterpol1);
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:      cudaFree(this->m_tmpInterpol2);
src/SemiLagrangian/SemiLagrangianGPUNew.cpp: * @brief init empty texture for interpolation on GPU 
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::InitializeInterpolationTexture() {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:      this->m_texture = gpuInitEmptyTexture(this->m_Opt->m_Domain.isize);
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:        cudaMalloc((void**) &this->m_tmpInterpol1, sizeof(float)*this->m_Opt->m_Domain.nl);
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:        cudaMalloc((void**) &this->m_tmpInterpol2, sizeof(float)*this->m_Opt->m_Domain.nl);
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:      this->m_texture = gpuInitEmptyTexture(this->isize_g);
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::SetWorkVecField(VecField* x) {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::ComputeTrajectory(VecField* v, std::string flag) {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:    // compute trajectory by calling a CUDA kernel
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::ComputeInitialTrajectory() {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::SetInitialTrajectory(const ScalarType* pX) {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::ComputeTrajectoryRK2(VecField* v, std::string flag) {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::ComputeTrajectoryRK4(VecField* v, std::string flag) {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::Interpolate(Vec* xo, Vec xi, std::string flag) {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::Interpolate(ScalarType* xo, ScalarType* xi, std::string flag) {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:    Interp3_Plan_GPU* interp_plan = nullptr;
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:                                &(this->m_Opt->m_GPUtime), 0, flag);
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:          gpuInterp3D(xi, xq, xo, this->m_tmpInterpol1, this->m_tmpInterpol2, this->m_Opt->m_Domain.nx, static_cast<long int>(nl), this->m_texture, this->m_Opt->m_PDESolver.iporder, &(this->m_Opt->m_GPUtime));
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:          gpuInterp3D(xi, xq, xo, this->m_tmpInterpol1, this->m_tmpInterpol2, this->m_Opt->m_Domain.nx, static_cast<long int>(nl), this->m_texture, this->m_Opt->m_PDESolver.iporder, &(this->m_Opt->m_GPUtime));
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::Interpolate(VecField* vo, VecField* vi, std::string flag) {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp: * @brief interpolate vector field - single GPU optimised version
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::Interpolate(ScalarType* wx1, ScalarType* wx2, ScalarType* wx3,
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:    Interp3_Plan_GPU* interp_plan = nullptr;
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:                                  &(this->m_Opt->m_GPUtime), 0, flag);
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:          gpuInterpVec3D(vx1, vx2, vx3, xq, wx1, wx2, wx3, this->m_tmpInterpol1, this->m_tmpInterpol2, this->m_Opt->m_Domain.nx, static_cast<long int>(nl), this->m_texture, this->m_Opt->m_PDESolver.iporder, &(this->m_Opt->m_GPUtime));
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:          gpuInterpVec3D(vx1, vx2, vx3, xq, wx1, wx2, wx3, this->m_tmpInterpol1, this->m_tmpInterpol2, this->m_Opt->m_Domain.nx, static_cast<long int>(nl), this->m_texture, this->m_Opt->m_PDESolver.iporder, &(this->m_Opt->m_GPUtime));
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::SetQueryPoints(ScalarType* y1, ScalarType* y2, ScalarType* y3, std::string flag) {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::GetQueryPoints(ScalarType* y1, ScalarType* y2, ScalarType* y3) {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:#if defined(REG_HAS_CUDA) && !defined(REG_HAS_MPICUDA) 
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::GetQueryPoints(ScalarType* y) {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:#if defined(REG_HAS_CUDA) && !defined(REG_HAS_MPICUDA)
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::CommunicateCoord(std::string flag) {
src/SemiLagrangian/SemiLagrangianGPUNew.cpp:PetscErrorCode SemiLagrangianGPUNew::MapCoordinateVector(std::string flag) {
src/RegOpt.cpp:#include "cuda_helper.hpp"
src/RegOpt.cpp:    this->m_GPUtime = 0;
src/RegOpt.cpp:#ifdef REG_HAS_CUDA
src/RegOpt.cpp:    cudaGetDevice(&this->m_gpu_id);
src/RegOpt.cpp:#if defined(REG_HAS_CUDA)
src/RegOpt.cpp:    ierr = DbgMsg("Grid continuation not implemented for GPU"); CHKERRQ(ierr);
src/RegOpt.cpp:#ifdef REG_HAS_CUDA
src/RegOpt.cpp:        std::cout << " compiled for GPU" << std::endl;
src/RegOpt.cpp:#ifdef REG_HAS_CUDA
src/RegOpt.cpp:#ifdef REG_HAS_CUDA
src/RegOpt.cpp:#ifdef REG_HAS_CUDA
src/RegOpt.cpp:#ifdef REG_HAS_CUDA
src/Regularization/RegularizationKernel.cu:#include "cuda_helper.hpp"
src/Regularization/RegularizationKernel.cu:using KernelUtils::ReductionKernelCallGPU;
src/Regularization/RegularizationKernel.cu:  ierr = ReductionKernelCallGPU<NormKernel>(lnorm, workspace, nl, pX[0], pX[1], pX[2]); CHKERRQ(ierr);
src/DeformationFields/DeformationKernel.cu:#include "cuda_helper.hpp"
src/DeformationFields/DeformationKernel.cu:  ierr = KernelUtils::KernelCallGPU<DetDefGradSLKernel>(nl, pJ, pJx, pDivV, pDivVx, alpha, ht); CHKERRQ(ierr);
src/DeformationFields/DeformationKernel.cu:  ierr = KernelUtils::KernelCallGPU<DetDefGradSLKernel>(nl, pJ, val); CHKERRQ(ierr);
src/Interpolation/Interp3_Plan_GPU.cpp:#include <interp3_gpu_mpi.hpp>
src/Interpolation/Interp3_Plan_GPU.cpp:#include <cuda.h>
src/Interpolation/Interp3_Plan_GPU.cpp:#include <cuda_runtime_api.h>
src/Interpolation/Interp3_Plan_GPU.cpp:#ifndef ACCFFT_CHECKCUDA_H
src/Interpolation/Interp3_Plan_GPU.cpp:#define ACCFFT_CHECKCUDA_H
src/Interpolation/Interp3_Plan_GPU.cpp:inline cudaError_t checkCuda_accfft(cudaError_t result)
src/Interpolation/Interp3_Plan_GPU.cpp:  if (result != cudaSuccess) {
src/Interpolation/Interp3_Plan_GPU.cpp:    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
src/Interpolation/Interp3_Plan_GPU.cpp:    assert(result == cudaSuccess);
src/Interpolation/Interp3_Plan_GPU.cpp:inline cufftResult checkCuda_accfft(cufftResult result)
src/Interpolation/Interp3_Plan_GPU.cpp:    fprintf(stderr, "CUDA Runtime Error: %s\n", result);
src/Interpolation/Interp3_Plan_GPU.cpp:class Trip_GPU{
src/Interpolation/Interp3_Plan_GPU.cpp:    Trip_GPU() {};
src/Interpolation/Interp3_Plan_GPU.cpp:static bool ValueCmp(Trip_GPU const & a, Trip_GPU const & b)
src/Interpolation/Interp3_Plan_GPU.cpp:    Trip_GPU* trip=new Trip_GPU[qsize];
src/Interpolation/Interp3_Plan_GPU.cpp:Interp3_Plan_GPU::Interp3_Plan_GPU (size_t g_alloc_max) {
src/Interpolation/Interp3_Plan_GPU.cpp:void Interp3_Plan_GPU::allocate (int N_pts, int* data_dof, int nplans)
src/Interpolation/Interp3_Plan_GPU.cpp:  cudaMalloc((void**)&this->ghost_reg_grid_vals_d, g_alloc_max*data_dof_max);
src/Interpolation/Interp3_Plan_GPU.cpp:  cudaMalloc((void**)&this->ghost_reg_grid_vals_d, g_alloc_max*data_dof_max);
src/Interpolation/Interp3_Plan_GPU.cpp:Interp3_Plan_GPU::~Interp3_Plan_GPU ()
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFreeHost(ghost_reg_grid_vals_d);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFreeHost(all_f_cubic_d);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFreeHost(xq1);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFreeHost(xq2);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFreeHost(xq3);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFreeHost(all_query_points_d);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFree(ghost_reg_grid_vals_d);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFree(all_f_cubic_d);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFree(xq1);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFree(xq2);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFree(xq3);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFree(all_query_points_d);
src/Interpolation/Interp3_Plan_GPU.cpp:void Interp3_Plan_GPU::scatter( int* N_reg,  // global grid dimensions
src/Interpolation/Interp3_Plan_GPU.cpp:    std::cout<<"ERROR Interp3_Plan_GPU Scatter called before calling allocate.\n";
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFreeHost(this->all_query_points_d);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFreeHost(this->xq1);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFreeHost(this->xq2);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFreeHost(this->xq3);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFreeHost(this->all_f_cubic_d);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMallocHost((void**)&all_query_points_d,all_query_points_allocation*sizeof(Real) );
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMallocHost((void**)&xq1, total_query_points*sizeof(Real));
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMallocHost((void**)&xq2, total_query_points*sizeof(Real));
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMallocHost((void**)&xq3, total_query_points*sizeof(Real));
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMallocHost((void**)&all_f_cubic_d, total_query_points*sizeof(Real)*data_dof_max);
src/Interpolation/Interp3_Plan_GPU.cpp:    // freeing the cuda memory is required everytime scatter is called because the distribution of query points might not be uniform across all GPUs
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFree(this->all_query_points_d);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFree(this->xq1);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFree(this->xq2);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFree(this->xq3);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaFree(this->all_f_cubic_d);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMalloc((void**)&all_f_cubic_d, total_query_points*sizeof(Real)*data_dof_max);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMalloc((void**)&xq1, total_query_points*sizeof(Real));
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMalloc((void**)&xq2, total_query_points*sizeof(Real));
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMalloc((void**)&xq3, total_query_points*sizeof(Real));
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMalloc((void**)&all_query_points_d,all_query_points_allocation*sizeof(Real) );
src/Interpolation/Interp3_Plan_GPU.cpp:    // if this the first time scatter is being called (scatter_baked = False) then, only allocate the memory on GPU
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMallocHost((void**)&all_query_points_d,all_query_points_allocation*sizeof(Real) );
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMallocHost((void**)&xq1, total_query_points*sizeof(Real));
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMallocHost((void**)&xq2, total_query_points*sizeof(Real));
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMallocHost((void**)&xq3, total_query_points*sizeof(Real));
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMallocHost((void**)&all_f_cubic_d, total_query_points*sizeof(Real)*data_dof_max);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMalloc((void**)&all_f_cubic_d, total_query_points*sizeof(Real)*data_dof_max);
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMalloc((void**)&xq1, total_query_points*sizeof(Real));
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMalloc((void**)&xq2, total_query_points*sizeof(Real));
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMalloc((void**)&xq3, total_query_points*sizeof(Real));
src/Interpolation/Interp3_Plan_GPU.cpp:    cudaMalloc((void**)&all_f_cubic_d, total_query_points*sizeof(Real)*data_dof);
src/Interpolation/Interp3_Plan_GPU.cpp:  cudaMemcpy(all_query_points_d, all_query_points, all_query_points_allocation*sizeof(Real), cudaMemcpyHostToDevice);
src/Interpolation/Interp3_Plan_GPU.cpp:void Interp3_Plan_GPU::interpolate( Real* ghost_reg_grid_vals, // ghost padded regular grid values on CPU
src/Interpolation/Interp3_Plan_GPU.cpp:                                    Real* query_values_d,      // interpolation result on GPU
src/Interpolation/Interp3_Plan_GPU.cpp:                                    cudaTextureObject_t yi_tex,// texture object for interpolation
src/Interpolation/Interp3_Plan_GPU.cpp:    std::cout<<"ERROR Interp3_Plan_GPU interpolate called before calling allocate.\n";
src/Interpolation/Interp3_Plan_GPU.cpp:    std::cout<<"ERROR Interp3_Plan_GPU interpolate called before calling scatter.\n";
src/Interpolation/Interp3_Plan_GPU.cpp:  cudaMemcpy((void*)ghost_reg_grid_vals_d, (const void*)ghost_reg_grid_vals, nlghost*data_dof*sizeof(Real), cudaMemcpyHostToDevice);
src/Interpolation/Interp3_Plan_GPU.cpp:    gpuInterpVec3D(&ghost_reg_grid_vals_d[0*nlghost], 
src/Interpolation/Interp3_Plan_GPU.cpp:    gpuInterp3D(ghost_reg_grid_vals_d, 
src/Interpolation/Interp3_Plan_GPU.cpp:  cudaMemcpy(all_f_cubic, all_f_cubic_d, data_dof*total_query_points*sizeof(Real) ,cudaMemcpyDeviceToHost);
src/Interpolation/interp3_gpu_new.cu:#include <cuda.h>
src/Interpolation/interp3_gpu_new.cu:#include <cuda_runtime.h>
src/Interpolation/interp3_gpu_new.cu:#include "cuda_helper_math.h"
src/Interpolation/interp3_gpu_new.cu:#include "interp3_gpu_new.hpp"
src/Interpolation/interp3_gpu_new.cu:#include "cuda_helper.hpp"
src/Interpolation/interp3_gpu_new.cu:#include "cuda_profiler_api.h"
src/Interpolation/interp3_gpu_new.cu:#define TWO_PI 2.0f*CUDART_PI
src/Interpolation/interp3_gpu_new.cu:__global__ void interp0gpu(float* m, float* q1, float* q2, float *q3, float *q, dim3 nx) {
src/Interpolation/interp3_gpu_new.cu:  interp0gpu<<<nl/256,256>>>(m,q1,q2,q3,q,n);
src/Interpolation/interp3_gpu_new.cu:__global__ void mgpu_cubicTex3DFastLagrange(cudaTextureObject_t tex,
src/Interpolation/interp3_gpu_new.cu:__global__ void cubicTex3DFastLagrange(cudaTextureObject_t tex,
src/Interpolation/interp3_gpu_new.cu:__global__ void mgpu_cubicTex3DSlowLagrange(cudaTextureObject_t tex,
src/Interpolation/interp3_gpu_new.cu:__global__ void cubicTex3DSlowLagrange(cudaTextureObject_t tex,
src/Interpolation/interp3_gpu_new.cu:__global__ void cubicTex3DSlowSpline(cudaTextureObject_t tex,
src/Interpolation/interp3_gpu_new.cu:__global__ void cubicTex3DFastSpline(cudaTextureObject_t tex,
src/Interpolation/interp3_gpu_new.cu:__device__ float linTex3D(cudaTextureObject_t tex, const float3 coord_grid, const float3 inv_reg_extent)
src/Interpolation/interp3_gpu_new.cu:    cudaEvent_t startEvent, stopEvent;
src/Interpolation/interp3_gpu_new.cu:    cudaEventCreate(&startEvent);
src/Interpolation/interp3_gpu_new.cu:    cudaEventCreate(&stopEvent);
src/Interpolation/interp3_gpu_new.cu:    cudaMemcpyToSymbol(d_c, h_c, sizeof(float)*(HALO+1), 0, cudaMemcpyHostToDevice);*/
src/Interpolation/interp3_gpu_new.cu:    if ( cudaSuccess != cudaGetLastError())
src/Interpolation/interp3_gpu_new.cu:    cudaCheckKernelError();
src/Interpolation/interp3_gpu_new.cu:    if ( cudaSuccess != cudaGetLastError())
src/Interpolation/interp3_gpu_new.cu:    cudaCheckKernelError();
src/Interpolation/interp3_gpu_new.cu:    if ( cudaSuccess != cudaGetLastError())
src/Interpolation/interp3_gpu_new.cu:    cudaCheckKernelError();
src/Interpolation/interp3_gpu_new.cu: * @brief create texture object with empty data (cudaArray)
src/Interpolation/interp3_gpu_new.cu:extern "C" cudaTextureObject_t gpuInitEmptyTexture(IntType* nx) {
src/Interpolation/interp3_gpu_new.cu:   cudaError_t err = cudaSuccess;
src/Interpolation/interp3_gpu_new.cu:   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
src/Interpolation/interp3_gpu_new.cu:   cudaExtent extent = make_cudaExtent(nx[2], nx[1], nx[0]);
src/Interpolation/interp3_gpu_new.cu:   cudaArray* cuArray;
src/Interpolation/interp3_gpu_new.cu:   err = cudaMalloc3DArray(&cuArray, &channelDesc, extent, 0);
src/Interpolation/interp3_gpu_new.cu:   if (err != cudaSuccess){
src/Interpolation/interp3_gpu_new.cu:        fprintf(stderr, "Failed to allocate 3D cudaArray (error code %s)!\n", cudaGetErrorString(err));
src/Interpolation/interp3_gpu_new.cu:    /* create cuda resource description */
src/Interpolation/interp3_gpu_new.cu:    struct cudaResourceDesc resDesc;
src/Interpolation/interp3_gpu_new.cu:    resDesc.resType = cudaResourceTypeArray;
src/Interpolation/interp3_gpu_new.cu:    struct cudaTextureDesc texDesc;
src/Interpolation/interp3_gpu_new.cu:    texDesc.addressMode[0] = cudaAddressModeWrap;
src/Interpolation/interp3_gpu_new.cu:    texDesc.addressMode[1] = cudaAddressModeWrap;
src/Interpolation/interp3_gpu_new.cu:    texDesc.addressMode[2] = cudaAddressModeWrap;
src/Interpolation/interp3_gpu_new.cu:    texDesc.readMode = cudaReadModeElementType;
src/Interpolation/interp3_gpu_new.cu:    texDesc.filterMode = cudaFilterModeLinear;
src/Interpolation/interp3_gpu_new.cu:    cudaTextureObject_t texObj = 0;
src/Interpolation/interp3_gpu_new.cu:    err = cudaCreateTextureObject( &texObj, &resDesc, &texDesc, NULL);
src/Interpolation/interp3_gpu_new.cu:    if (err != cudaSuccess){
src/Interpolation/interp3_gpu_new.cu:        fprintf(stderr, "Failed to create texture (error code %s)!\n", cudaGetErrorString(err));
src/Interpolation/interp3_gpu_new.cu: * @brief update texture object by copying volume data to 3D cudaArray container
src/Interpolation/interp3_gpu_new.cu:void updateTextureFromVolume(cudaPitchedPtr volume, cudaExtent extent, cudaTextureObject_t texObj) {
src/Interpolation/interp3_gpu_new.cu:    cudaError_t err = cudaSuccess;
src/Interpolation/interp3_gpu_new.cu:    /* create cuda resource description */
src/Interpolation/interp3_gpu_new.cu:    struct cudaResourceDesc resDesc;
src/Interpolation/interp3_gpu_new.cu:    cudaGetTextureObjectResourceDesc( &resDesc, texObj);
src/Interpolation/interp3_gpu_new.cu:    cudaMemcpy3DParms p = {0};
src/Interpolation/interp3_gpu_new.cu:    p.kind = cudaMemcpyDeviceToDevice;
src/Interpolation/interp3_gpu_new.cu:    err = cudaMemcpy3D(&p);
src/Interpolation/interp3_gpu_new.cu:    if (err != cudaSuccess) {
src/Interpolation/interp3_gpu_new.cu:        fprintf(stderr, "Failed to copy 3D memory to cudaArray (error name %s = %s)!\n", cudaGetErrorName(err), cudaGetErrorString(err));
src/Interpolation/interp3_gpu_new.cu:__global__ void mgpu_interp3D_kernel_linear(
src/Interpolation/interp3_gpu_new.cu:        cudaTextureObject_t  yi_tex,
src/Interpolation/interp3_gpu_new.cu:        cudaTextureObject_t  yi_tex,
src/Interpolation/interp3_gpu_new.cu:void gpuInterp3Dkernel(
src/Interpolation/interp3_gpu_new.cu:           cudaTextureObject_t yi_tex,
src/Interpolation/interp3_gpu_new.cu:           cudaExtent yi_extent,
src/Interpolation/interp3_gpu_new.cu:           cudaStream_t* stream)
src/Interpolation/interp3_gpu_new.cu:    cudaPitchedPtr yi_cudaPitchedPtr;
src/Interpolation/interp3_gpu_new.cu:      //cudaMemcpyToSymbol(d_nx, &nx[0], sizeof(int), 0, cudaMemcpyHostToDevice);
src/Interpolation/interp3_gpu_new.cu:      //cudaMemcpyToSymbol(d_ny, &nx[1], sizeof(int), 0, cudaMemcpyHostToDevice);
src/Interpolation/interp3_gpu_new.cu:      //cudaMemcpyToSymbol(d_nz, &nx[2], sizeof(int), 0, cudaMemcpyHostToDevice);
src/Interpolation/interp3_gpu_new.cu:            yi_cudaPitchedPtr = make_cudaPitchedPtr(static_cast<void*>(tmp1), nx[2]*sizeof(float), nx[2], nx[1]);
src/Interpolation/interp3_gpu_new.cu:            yi_cudaPitchedPtr = make_cudaPitchedPtr(static_cast<void*>(tmp1), nx[2]*sizeof(float), nx[2], nx[1]);
src/Interpolation/interp3_gpu_new.cu:            yi_cudaPitchedPtr = make_cudaPitchedPtr(static_cast<void*>(yi), nx[2]*sizeof(float), nx[2], nx[1]);
src/Interpolation/interp3_gpu_new.cu:            yi_cudaPitchedPtr = make_cudaPitchedPtr(static_cast<void*>(yi), nx[2]*sizeof(float), nx[2], nx[1]);
src/Interpolation/interp3_gpu_new.cu:      // make input image a cudaPitchedPtr for fi
src/Interpolation/interp3_gpu_new.cu:      yi_cudaPitchedPtr = make_cudaPitchedPtr(static_cast<void*>(yi), nx[2]*sizeof(float), nx[2], nx[1]);
src/Interpolation/interp3_gpu_new.cu:    updateTextureFromVolume(yi_cudaPitchedPtr, yi_extent, yi_tex);
src/Interpolation/interp3_gpu_new.cu:        mgpu_interp3D_kernel_linear<<<blocks,threads, 0, *stream>>>(yi_tex, xq[0], xq[1], xq[2], yo, inv_nx, nq);
src/Interpolation/interp3_gpu_new.cu:              mgpu_cubicTex3DFastLagrange<<<blocks,threads, 0, *stream>>>(yi_tex, xq[0], xq[1], xq[2], yo, inv_nx, nq);
src/Interpolation/interp3_gpu_new.cu:              mgpu_cubicTex3DSlowLagrange<<<blocks,threads, 0, *stream>>>(yi_tex, xq[0], xq[1], xq[2], yo, inv_nx, nq);
src/Interpolation/interp3_gpu_new.cu:void gpuInterp3D(
src/Interpolation/interp3_gpu_new.cu:           cudaTextureObject_t yi_tex,
src/Interpolation/interp3_gpu_new.cu:    // create a cudaExtent for input resolution
src/Interpolation/interp3_gpu_new.cu:    cudaExtent yi_extent = make_cudaExtent(nx[2], nx[1], nx[0]);
src/Interpolation/interp3_gpu_new.cu:    cudaStream_t stream;
src/Interpolation/interp3_gpu_new.cu:    cudaStreamCreate(&stream);
src/Interpolation/interp3_gpu_new.cu:    gpuInterp3Dkernel(yi,xq,yo,tmp1,tmp2,nx,yi_tex,iporder,yi_extent,inv_nx,nq, &stream);
src/Interpolation/interp3_gpu_new.cu:    cudaStreamSynchronize(stream);
src/Interpolation/interp3_gpu_new.cu:    cudaStreamDestroy(stream);
src/Interpolation/interp3_gpu_new.cu:    cudaDeviceSynchronize();
src/Interpolation/interp3_gpu_new.cu: * @parm[in] nx array denoting number of query coordinates in each dimension  (this will be isize when using multi-GPU)
src/Interpolation/interp3_gpu_new.cu:void gpuInterpVec3D(
src/Interpolation/interp3_gpu_new.cu:           IntType*  inx, long int nq, cudaTextureObject_t yi_tex, int iporder, float* interp_time)
src/Interpolation/interp3_gpu_new.cu:    // create a cudaExtent for input resolution
src/Interpolation/interp3_gpu_new.cu:    cudaExtent yi_extent = make_cudaExtent(nx[2], nx[1], nx[0]);
src/Interpolation/interp3_gpu_new.cu:    // in case there is not enough work for the GPU
src/Interpolation/interp3_gpu_new.cu:    cudaStream_t stream[3];
src/Interpolation/interp3_gpu_new.cu:      cudaStreamCreate(&stream[i]);
src/Interpolation/interp3_gpu_new.cu:    gpuInterp3Dkernel(yi1,xq,yo1,tmp1,tmp2,nx,yi_tex,iporder,yi_extent,inv_nx,nq, &stream[0]);
src/Interpolation/interp3_gpu_new.cu:    gpuInterp3Dkernel(yi2,xq,yo2,tmp1,tmp2,nx,yi_tex,iporder,yi_extent,inv_nx,nq, &stream[1]);
src/Interpolation/interp3_gpu_new.cu:    gpuInterp3Dkernel(yi3,xq,yo3,tmp1,tmp2,nx,yi_tex,iporder,yi_extent,inv_nx,nq, &stream[2]);
src/Interpolation/interp3_gpu_new.cu:      cudaStreamSynchronize(stream[i]);
src/Interpolation/interp3_gpu_new.cu:      cudaStreamDestroy(stream[i]);
src/Interpolation/interp3_gpu_new.cu:    cudaDeviceSynchronize();
src/Interpolation/interp3_gpu_new.cu:    cudaDeviceSynchronize();
src/Interpolation/interp3_gpu_new.cu:    cudaDeviceSynchronize();
src/Interpolation/interp3_gpu_new.cu:    //cudaMemcpyToSymbol(d_h, h, sizeof(float)*(3), 0, cudaMemcpyHostToDevice);
src/Interpolation/interp3_gpu_new.cu:    cudaDeviceSynchronize();
src/Interpolation/interp3_gpu_new.cu:    //cudaMemcpyToSymbol(d_h, h, 3*sizeof(float), 0, cudaMemcpyHostToDevice);
src/Interpolation/interp3_gpu_new.cu:    //cudaMemcpyToSymbol(d_iX0, iX0, 3*sizeof(float), 0, cudaMemcpyHostToDevice);
src/Interpolation/interp3_gpu_new.cu:    //cudaMemcpyToSymbol(d_iX1, iX1, 3*sizeof(float), 0, cudaMemcpyHostToDevice);
src/Interpolation/interp3_gpu_new.cu:    cudaDeviceSynchronize();
src/Interpolation/interp3_gpu_new.cu:    cudaDeviceSynchronize();
src/Interpolation/interp3_gpu_new.cu:  cudaDeviceSynchronize();
src/Interpolation/Interp3_Plan_GPU.cu:#include <interp3_gpu_mpi.hpp>
src/Interpolation/Interp3_Plan_GPU.cu:#include <cuda.h>
src/Interpolation/Interp3_Plan_GPU.cu:#include <cuda_runtime_api.h>
src/Interpolation/Interp3_Plan_GPU.cu:#include <cuda_helper.hpp>
src/Interpolation/Interp3_Plan_GPU.cu://static void printGPUMemory(int rank) {
src/Interpolation/Interp3_Plan_GPU.cu://      cudaMemGetInfo(&free, &used);
src/Interpolation/Interp3_Plan_GPU.cu:#ifndef ACCFFT_CHECKCUDA_H
src/Interpolation/Interp3_Plan_GPU.cu:#define ACCFFT_CHECKCUDA_H
src/Interpolation/Interp3_Plan_GPU.cu:inline cudaError_t checkCuda_accfft(cudaError_t result)
src/Interpolation/Interp3_Plan_GPU.cu:  if (result != cudaSuccess) {
src/Interpolation/Interp3_Plan_GPU.cu:    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
src/Interpolation/Interp3_Plan_GPU.cu:    assert(result == cudaSuccess);
src/Interpolation/Interp3_Plan_GPU.cu:inline cufftResult checkCuda_accfft(cufftResult result)
src/Interpolation/Interp3_Plan_GPU.cu:    fprintf(stderr, "CUDA Runtime Error: %s\n", result);
src/Interpolation/Interp3_Plan_GPU.cu:class Trip_GPU{
src/Interpolation/Interp3_Plan_GPU.cu:    Trip_GPU(){};
src/Interpolation/Interp3_Plan_GPU.cu:static bool ValueCmp(Trip_GPU const & a, Trip_GPU const & b)
src/Interpolation/Interp3_Plan_GPU.cu:    Trip_GPU* trip=new Trip_GPU[qsize];
src/Interpolation/Interp3_Plan_GPU.cu:Interp3_Plan_GPU::Interp3_Plan_GPU (size_t g_alloc_max, bool cuda_aware) {
src/Interpolation/Interp3_Plan_GPU.cu:  this->cuda_aware = cuda_aware;
src/Interpolation/Interp3_Plan_GPU.cu:void Interp3_Plan_GPU::allocate (int N_pts, int* data_dofs, int nplans, int gsize, IntType *isize_g)
src/Interpolation/Interp3_Plan_GPU.cu:Interp3_Plan_GPU::~Interp3_Plan_GPU ()
src/Interpolation/Interp3_Plan_GPU.cu:void Interp3_Plan_GPU::scatter( IntType* N_reg,  // global grid dimensions
src/Interpolation/Interp3_Plan_GPU.cu:    std::cout<<"ERROR Interp3_Plan_GPU Scatter called before calling allocate.\n";
src/Interpolation/Interp3_Plan_GPU.cu:void Interp3_Plan_GPU::test_kernel(ScalarType* f, int nq) {
src/Interpolation/Interp3_Plan_GPU.cu:void Interp3_Plan_GPU::interpolate( ScalarType* f_ghost, // ghost padded regular grid values on GPU
src/Interpolation/Interp3_Plan_GPU.cu:                                    ScalarType** query_values,      // interpolation result on GPU
src/Interpolation/Interp3_Plan_GPU.cu:                                    cudaTextureObject_t yi_tex,// texture object for interpolation
src/Interpolation/Interp3_Plan_GPU.cu:    std::cout<<"ERROR Interp3_Plan_GPU interpolate called before calling allocate.\n";
src/Interpolation/Interp3_Plan_GPU.cu:    std::cout<<"ERROR Interp3_Plan_GPU interpolate called before calling scatter.\n";
src/Interpolation/Interp3_Plan_GPU.cu:  // compute the interpolation on the GPU
src/Interpolation/Interp3_Plan_GPU.cu:    gpuInterpVec3D(&f_ghost[0*nlghost], 
src/Interpolation/Interp3_Plan_GPU.cu:    gpuInterp3D(f_ghost, 
src/Interpolation/Interp3_Plan_GPU.cu:        cudaMemcpy(&f_unordered_ptr[roffset+dof*N_pts], &all_f[soffset+dof*Eq[id].total_query_points], sizeof(ScalarType)*Eq[id].f_index_procs_self_sizes[i], cudaMemcpyDeviceToDevice);
src/Interpolation/Interp3_Plan_GPU_kernel.cu:#include <interp3_gpu_mpi.hpp>
src/Interpolation/Interp3_Plan_GPU_kernel.cu:  cudaGetDevice(&dev_id);
src/Spectral/mpicufft.cpp:#include <cuda_runtime_api.h>
src/Spectral/mpicufft.cpp:#if (cudaError == 0) && (cufftError == 0)
src/Spectral/mpicufft.cpp:#define cudaCheck(e) {                                           \
src/Spectral/mpicufft.cpp:    printf("CUDA error code %s:%d: %i\n",__FILE__,__LINE__,err); \
src/Spectral/mpicufft.cpp:#define cudaCheck(e) {e}
src/Spectral/mpicufft.cpp:template<typename T> MPIcuFFT<T>::MPIcuFFT(MPI_Comm comm, bool cuda_aware)
src/Spectral/mpicufft.cpp:    : comm(comm), cuda_aware(cuda_aware) {
src/Spectral/mpicufft.cpp:  if (allocated_d && workarea_d) cudaFree(workarea_d);
src/Spectral/mpicufft.cpp:  if (allocated_h && workarea_h) cudaCheck(cudaFreeHost(workarea_h));
src/Spectral/mpicufft.cpp:  if (planR2C) cudaCheck(cufftDestroy(planR2C));
src/Spectral/mpicufft.cpp:  if (planC2R) cudaCheck(cufftDestroy(planC2R));
src/Spectral/mpicufft.cpp:  if (planC2C) cudaCheck(cufftDestroy(planC2C));
src/Spectral/mpicufft.cpp:  if (pcnt > 4 && cuda_aware && local_volume <= 524288) comm_mode = All2All;
src/Spectral/mpicufft.cpp:  cudaCheck(cufftCreate(&planC2R));
src/Spectral/mpicufft.cpp:  cudaCheck(cufftCreate(&planR2C));
src/Spectral/mpicufft.cpp:  cudaCheck(cufftSetAutoAllocation(planR2C, 0));
src/Spectral/mpicufft.cpp:  cudaCheck(cufftSetAutoAllocation(planC2R, 0));
src/Spectral/mpicufft.cpp:    cudaCheck(cufftMakePlan3d(planR2C, isizex[pidx], isizey, isizez, cuFFT<T>::R2Ctype, &ws_r2c));
src/Spectral/mpicufft.cpp:    cudaCheck(cufftMakePlan3d(planC2R, isizex[pidx], isizey, isizez, cuFFT<T>::C2Rtype, &ws_c2r));
src/Spectral/mpicufft.cpp:    cudaCheck(cufftCreate(&planC2C));
src/Spectral/mpicufft.cpp:    cudaCheck(cufftSetAutoAllocation(planC2C, 0));
src/Spectral/mpicufft.cpp:    cudaCheck(cufftMakePlanMany64(planR2C, 2, &n[1], 0, 0, 0, 0, 0, 0, cuFFT<T>::R2Ctype, batch, &ws_r2c));
src/Spectral/mpicufft.cpp:    cudaCheck(cufftMakePlanMany64(planC2R, 2, &n[1], 0, 0, 0, 0, 0, 0, cuFFT<T>::C2Rtype, batch, &ws_c2r));
src/Spectral/mpicufft.cpp:    cudaCheck(cufftMakePlanMany64(planC2C, 1, n, nembed, osizey[pidx]*osizez, 1, nembed, osizey[pidx]*osizez, 1, cuFFT<T>::C2Ctype, osizey[pidx]*osizez, &ws_c2c));
src/Spectral/mpicufft.cpp:  //worksize_h = (cuda_aware || fft3d ? 0 : 2*domainsize);
src/Spectral/mpicufft.cpp:  worksize_h = (cuda_aware || fft3d ? 0 : 2*domainsize);
src/Spectral/mpicufft.cpp:  cudaCheck(cudaDeviceSynchronize());
src/Spectral/mpicufft.cpp:    cudaCheck(cudaFree(workarea_d));
src/Spectral/mpicufft.cpp:    cudaCheck(cudaMalloc(&workarea_d, worksize_d));
src/Spectral/mpicufft.cpp:    cudaCheck(cufftSetWorkArea(planR2C, mem_d[0]));
src/Spectral/mpicufft.cpp:    cudaCheck(cufftSetWorkArea(planC2R, mem_d[0]));
src/Spectral/mpicufft.cpp:    cudaCheck(cufftSetWorkArea(planR2C, mem_d[2]));
src/Spectral/mpicufft.cpp:    cudaCheck(cufftSetWorkArea(planC2R, mem_d[2]));
src/Spectral/mpicufft.cpp:    cudaCheck(cufftSetWorkArea(planC2C, mem_d[2]));
src/Spectral/mpicufft.cpp:    cudaCheck(cudaFreeHost(workarea_h));
src/Spectral/mpicufft.cpp:    cudaCheck(cudaMallocHost(&workarea_h, worksize_h));
src/Spectral/mpicufft.cpp:    cudaCheck(cudaDeviceSynchronize());
src/Spectral/mpicufft.cpp:    if (cuda_aware) {
src/Spectral/mpicufft.cpp:    cudaCheck(cuFFT<T>::execR2C(planR2C, real, complex));
src/Spectral/mpicufft.cpp:    cudaCheck(cudaDeviceSynchronize());
src/Spectral/mpicufft.cpp:        cudaCheck(cudaMemcpy2DAsync(&send_ptr[oslice], sizeof(C_t)*osizey[p]*osizez,
src/Spectral/mpicufft.cpp:                                    cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));
src/Spectral/mpicufft.cpp:      cudaCheck(cuFFT<T>::execR2C(planR2C, &real[isizez*isizey*batch], &complex[osizez*isizey*batch]));
src/Spectral/mpicufft.cpp:      cudaCheck(cudaDeviceSynchronize());
src/Spectral/mpicufft.cpp:        cudaCheck(cudaMemcpy2DAsync(&send_ptr[oslice + batch*osizez*osizey[p]], sizeof(C_t)*osizey[p]*osizez,
src/Spectral/mpicufft.cpp:                                    cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));
src/Spectral/mpicufft.cpp:        cudaCheck(cudaDeviceSynchronize());
src/Spectral/mpicufft.cpp:        cudaCheck(cudaMemcpy2DAsync(&temp_ptr[oslice], sizeof(C_t)*osizey[pidx]*osizez,
src/Spectral/mpicufft.cpp:                                    cudaMemcpyDeviceToDevice));
src/Spectral/mpicufft.cpp:      if (!cuda_aware) { // copy received blocks to device
src/Spectral/mpicufft.cpp:          cudaCheck(cudaMemcpyAsync(&temp_ptr[istartx[p]*osizez*osizey[pidx]],
src/Spectral/mpicufft.cpp:                                    isizex[p]*osizez*osizey[pidx]*sizeof(C_t), cudaMemcpyHostToDevice));
src/Spectral/mpicufft.cpp:      cudaCheck(cudaDeviceSynchronize());
src/Spectral/mpicufft.cpp:          cudaCheck(cudaMemcpy2DAsync(&send_ptr[oslice], sizeof(C_t)*osizey[p]*osizez,
src/Spectral/mpicufft.cpp:                                      cudaMemcpyDeviceToDevice));
src/Spectral/mpicufft.cpp:          cudaCheck(cudaMemcpy2DAsync(&send_ptr[oslice + batch*osizez*osizey[p]], sizeof(C_t)*osizey[p]*osizez,
src/Spectral/mpicufft.cpp:                                      cudaMemcpyDeviceToDevice));
src/Spectral/mpicufft.cpp:      cudaCheck(cudaDeviceSynchronize());
src/Spectral/mpicufft.cpp:    // compute remaining 1d FFT, for cuda-aware recv and temp buffer are identical
src/Spectral/mpicufft.cpp:    cudaCheck(cuFFT<T>::execC2C(planC2C, temp_ptr, complex, CUFFT_FORWARD));
src/Spectral/mpicufft.cpp:    cudaCheck(cudaDeviceSynchronize());
src/Spectral/mpicufft.cpp:    cudaCheck(cudaDeviceSynchronize());
src/Spectral/mpicufft.cpp:    if (cuda_aware) {
src/Spectral/mpicufft.cpp:    cudaCheck(cuFFT<T>::execC2C(planC2C, complex, temp_ptr, CUFFT_INVERSE));
src/Spectral/mpicufft.cpp:    cudaCheck(cudaDeviceSynchronize());
src/Spectral/mpicufft.cpp:        if (!cuda_aware) {
src/Spectral/mpicufft.cpp:          cudaCheck(cudaMemcpy(&send_ptr[istartx[p]*osizez*osizey[pidx]],
src/Spectral/mpicufft.cpp:                               isizex[p]*osizez*osizey[pidx]*sizeof(C_t), cudaMemcpyDeviceToHost));
src/Spectral/mpicufft.cpp:          cudaCheck(cudaDeviceSynchronize());
src/Spectral/mpicufft.cpp:        cudaCheck(cudaMemcpy2DAsync(&copy_ptr[ostarty[pidx]*osizez], sizeof(C_t)*osizez*isizey,
src/Spectral/mpicufft.cpp:                                    cudaMemcpyDeviceToDevice));
src/Spectral/mpicufft.cpp:          cudaCheck(cudaMemcpy2DAsync(&copy_ptr[ostarty[p]*osizez], sizeof(C_t)*osizez*isizey,
src/Spectral/mpicufft.cpp:                                      cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice));
src/Spectral/mpicufft.cpp:      cudaCheck(cudaDeviceSynchronize());
src/Spectral/mpicufft.cpp:        cudaCheck(cudaMemcpy2DAsync(&copy_ptr[ostarty[p]*osizez], sizeof(C_t)*osizez*isizey,
src/Spectral/mpicufft.cpp:                                    cudaMemcpyDeviceToDevice));
src/Spectral/mpicufft.cpp:      cudaCheck(cudaDeviceSynchronize());
src/Spectral/mpicufft.cpp:    cudaCheck(cuFFT<T>::execC2R(planC2R, copy_ptr, real));
src/Spectral/mpicufft.cpp:    cudaCheck(cudaDeviceSynchronize());
src/Spectral/mpicufft.cpp:      cudaCheck(cuFFT<T>::execC2R(planC2R, &copy_ptr[osizez*isizey*batch], &real[isizez*isizey*batch]));
src/Spectral/mpicufft.cpp:      cudaCheck(cudaDeviceSynchronize());
src/Spectral/mpicufft.cpp:    cudaCheck(cudaMemset(out, 0, fft_o->domainsize));
src/Spectral/mpicufft.cpp:        cudaCheck(cudaMemcpy2DAsync(&data_o[offset_o], pitch_o,
src/Spectral/mpicufft.cpp:                                    cudaMemcpyDeviceToDevice));
src/Spectral/mpicufft.cpp:        cudaCheck(cudaMemcpy2DAsync(&data_o[offset_o], pitch_o,
src/Spectral/mpicufft.cpp:                                    cudaMemcpyDeviceToDevice));
src/Spectral/mpicufft.cpp:        cudaCheck(cudaMemcpy2DAsync(&data_o[offset_o], pitch_o,
src/Spectral/mpicufft.cpp:                                    cudaMemcpyDeviceToDevice));
src/Spectral/mpicufft.cpp:        cudaCheck(cudaMemcpy2DAsync(&data_o[offset_o], pitch_o,
src/Spectral/mpicufft.cpp:                                    cudaMemcpyDeviceToDevice));
src/Spectral/mpicufft.cpp:    if (cuda_aware) {
src/Spectral/mpicufft.cpp:          cudaCheck(cudaMemcpy2DAsync(&data_o[offset_o], pitch_o,
src/Spectral/mpicufft.cpp:                                      cudaMemcpyDeviceToDevice));
src/Spectral/mpicufft.cpp:          cudaCheck(cudaMemcpy2DAsync(&data_o[offset_o], pitch_o,
src/Spectral/mpicufft.cpp:                                      cudaMemcpyDeviceToDevice));
src/Spectral/mpicufft.cpp:          cudaCheck(cudaMemcpy2DAsync(&send_ptr[pitch*isend[i] + slice], width,
src/Spectral/mpicufft.cpp:                                      cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));
src/Spectral/mpicufft.cpp:          cudaCheck(cudaMemcpy2DAsync(&send_ptr[pitch*isend[i] + offset + slice], width,
src/Spectral/mpicufft.cpp:                                      cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));
src/Spectral/mpicufft.cpp:        cudaDeviceSynchronize();
src/Spectral/mpicufft.cpp:          cudaCheck(cudaMemcpy2DAsync(&data_o[offset_o], pitch_o,
src/Spectral/mpicufft.cpp:                                      cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice));
src/Spectral/mpicufft.cpp:          cudaCheck(cudaMemcpy2DAsync(&data_o[offset_o], pitch_o,
src/Spectral/mpicufft.cpp:                                      cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice));
src/Spectral/mpicufft.cpp:    cudaCheck(cudaDeviceSynchronize());
src/Spectral/mpicufft.cpp:          cudaCheck(cudaMemcpy2DAsync(&data_o[offset_o], pitch_o,
src/Spectral/mpicufft.cpp:                                      cudaMemcpyDeviceToDevice));
src/Spectral/mpicufft.cpp:          cudaCheck(cudaMemcpy2DAsync(&data_o[offset_o], pitch_o,
src/Spectral/mpicufft.cpp:                                      cudaMemcpyDeviceToDevice));
src/Spectral/mpicufft.cpp:          cudaCheck(cudaMemcpy2DAsync(&send_ptr[pitch*isend[i] + slice], width,
src/Spectral/mpicufft.cpp:                                      cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));
src/Spectral/mpicufft.cpp:          cudaCheck(cudaMemcpy2DAsync(&send_ptr[pitch*isend[i] + offset + slice], width,
src/Spectral/mpicufft.cpp:                                      cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));
src/Spectral/mpicufft.cpp:        cudaDeviceSynchronize();
src/Spectral/mpicufft.cpp:          cudaCheck(cudaMemcpy2DAsync(&data_o[offset_o], pitch_o,
src/Spectral/mpicufft.cpp:                                      cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice));
src/Spectral/mpicufft.cpp:          cudaCheck(cudaMemcpy2DAsync(&data_o[offset_o], pitch_o,
src/Spectral/mpicufft.cpp:                                      cuda_aware?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice));
src/Spectral/mpicufft.cpp:    cudaCheck(cudaDeviceSynchronize());
src/Spectral/Spectral.cpp:#ifdef REG_HAS_CUDA
src/Spectral/Spectral.cpp:#ifdef REG_HAS_CUDA
src/Spectral/Spectral.cpp:#ifdef REG_HAS_MPICUDA
src/Spectral/Spectral.cpp:#ifdef REG_HAS_CUDA
src/Spectral/Spectral.cpp:#ifdef REG_HAS_CUDA
src/Spectral/Spectral.cpp:#ifdef REG_HAS_CUDA
src/Spectral/Spectral.cpp:#if REG_HAS_CUDA
src/Spectral/Spectral.cpp:#if REG_HAS_CUDA
src/Spectral/Spectral.cpp:#if REG_HAS_CUDA
src/Spectral/SpectralKernel.cu:#include "cuda_helper.hpp"
src/Spectral/SpectralKernel.cu:#include <cuda.h>
src/Spectral/SpectralKernel.cu:#include <cuda_runtime.h>
src/Spectral/SpectralKernel.cu:using KernelUtils::SpectralKernelCallGPU;
src/Spectral/SpectralKernel.cu:using KernelUtils::SpectralReductionKernelCallGPU;
src/Spectral/SpectralKernel.cu:  ierr = SpectralKernelCallGPU<LowPassFilterKernel>(nstart, nx, nl, pXHat, l1, l2, l3, scale); CHKERRQ(ierr);
src/Spectral/SpectralKernel.cu:  ierr = SpectralKernelCallGPU<HighPassFilterKernel>(nstart, nx, nl, pXHat, l1, l2, l3, scale); CHKERRQ(ierr);
src/Spectral/SpectralKernel.cu:  ierr = SpectralKernelCallGPU<ScaleKernel>(nstart, nx, nl, pX, val); CHKERRQ(ierr);
src/Spectral/SpectralKernel.cu:  ierr = SpectralReductionKernelCallGPU<NormKernel>(norm, pWS, nstart, nx, nl, pXHat, l1, l2, l3); CHKERRQ(ierr);
src/Spectral/SpectralKernel.cu:  cudaMemset(pXc, 0, sizeof(ComplexType)*osize_c[0]*osize_c[1]*osize_c[2]);
src/Spectral/SpectralKernel.cu:    cudaMemcpy2DAsync(&pXc[offset_c], pitch_c, const_cast<ComplexType*>(&pXf[offset_f]), pitch_f, width, height_l, cudaMemcpyDeviceToDevice);
src/Spectral/SpectralKernel.cu:    cudaMemcpy2DAsync(&pXc[offset_c], pitch_c, const_cast<ComplexType*>(&pXf[offset_f]), pitch_f, width, height_h, cudaMemcpyDeviceToDevice);
src/Spectral/SpectralKernel.cu:    cudaMemcpy2DAsync(&pXc[offset_c], pitch_c, const_cast<ComplexType*>(&pXf[offset_f]), pitch_f, width, height_l, cudaMemcpyDeviceToDevice);
src/Spectral/SpectralKernel.cu:    cudaMemcpy2DAsync(&pXc[offset_c], pitch_c, const_cast<ComplexType*>(&pXf[offset_f]), pitch_f, width, height_h, cudaMemcpyDeviceToDevice);
src/Spectral/SpectralKernel.cu:  cudaDeviceSynchronize();
src/Spectral/SpectralKernel.cu:  cudaMemset(pXf, 0, sizeof(ComplexType)*nl[0]*nl[1]*nl[2]);
src/Spectral/SpectralKernel.cu:    cudaMemcpy2DAsync(&pXf[offset_f], pitch_f, const_cast<ComplexType*>(&pXc[offset_c]), pitch_c, width, height_l, cudaMemcpyDeviceToDevice);
src/Spectral/SpectralKernel.cu:    cudaMemcpy2DAsync(&pXf[offset_f], pitch_f, const_cast<ComplexType*>(&pXc[offset_c]), pitch_c, width, height_h, cudaMemcpyDeviceToDevice);
src/Spectral/SpectralKernel.cu:    cudaMemcpy2DAsync(&pXf[offset_f], pitch_f, const_cast<ComplexType*>(&pXc[offset_c]), pitch_c, width, height_l, cudaMemcpyDeviceToDevice);
src/Spectral/SpectralKernel.cu:    cudaMemcpy2DAsync(&pXf[offset_f], pitch_f, const_cast<ComplexType*>(&pXc[offset_c]), pitch_c, width, height_h, cudaMemcpyDeviceToDevice);
src/Spectral/SpectralKernel.cu:  cudaDeviceSynchronize();
src/Spectral/SpectralKernel.cpp:#include "cuda_helper.hpp"
src/CLAIREBase.cpp:#ifndef REG_HAS_CUDA
src/CLAIREBase.cpp:        cudaFree(this->m_x1hat);
src/CLAIREBase.cpp:#ifndef REG_HAS_CUDA
src/CLAIREBase.cpp:        cudaFree(this->m_x2hat);
src/CLAIREBase.cpp:#ifndef REG_HAS_CUDA
src/CLAIREBase.cpp:        cudaFree(this->m_x3hat);
src/CLAIREBase.cpp:    ierr = DebugGPUNotImplemented(); CHKERRQ(ierr);
src/TwoLevel/TwoLevelKernel.cu:__global__ void Restrict0stCentralGPU(T* __restrict__ dst,
src/TwoLevel/TwoLevelKernel.cu:__global__ void Restrict1stIncludeGPU(T* __restrict__ dst,
src/TwoLevel/TwoLevelKernel.cu:__global__ void Restrict1stCentralGPU(T* __restrict__ dst,
src/TwoLevel/TwoLevelKernel.cu:__global__ void Restrict2ndCentralGPU(T* __restrict__ dst,
src/TwoLevel/TwoLevelKernel.cu:__global__ void Restrict3rdCentralGPU(T* __restrict__ dst,
src/TwoLevel/TwoLevelKernel.cu:__global__ void Prolong0stCentralGPU(T* __restrict__ dst,
src/TwoLevel/TwoLevelKernel.cu:__global__ void Prolong1stIncludeGPU(T* __restrict__ dst,
src/TwoLevel/TwoLevelKernel.cu:__global__ void Prolong1stCentralGPU(T* __restrict__ dst,
src/TwoLevel/TwoLevelKernel.cu:__global__ void Prolong2ndCentralGPU(T* __restrict__ dst,
src/TwoLevel/TwoLevelKernel.cu:__global__ void Prolong3rdCentralGPU(T* __restrict__ dst,
src/TwoLevel/TwoLevelKernel.cu:__global__ void Prolong5thCentralGPU(T* __restrict__ dst,
src/TwoLevel/TwoLevelKernel.cu:  Restrict3rdCentralGPU<<<grid, block>>>(dst, src, nl3);
src/TwoLevel/TwoLevelKernel.cu:  ierr = cudaDeviceSynchronize(); CHKERRCUDA(ierr);
src/TwoLevel/TwoLevelKernel.cu:  ierr = cudaCheckKernelError(); CHKERRCUDA(ierr);  
src/TwoLevel/TwoLevelKernel.cu:  Prolong3rdCentralGPU<<<grid, block>>>(dst, src, nl3);
src/TwoLevel/TwoLevelKernel.cu:  ierr = cudaDeviceSynchronize(); CHKERRCUDA(ierr);
src/TwoLevel/TwoLevelKernel.cu:  ierr = cudaCheckKernelError(); CHKERRCUDA(ierr);  
src/TwoLevel/TwoLevelFinite.cpp:#if defined(REG_HAS_CUDA) || defined(REG_HAS_MPICUDA)
src/TwoLevel/TwoLevelFinite.cpp:      cudaMalloc((void**)&this->m_Ghost, this->g_alloc_max);
src/TwoLevel/TwoLevelFinite.cpp:      cudaFree(this->m_Ghost);
src/TwoLevel/TwoLevelFinite.cpp:  DebugGPUStartEvent("Finite Restrict");
src/TwoLevel/TwoLevelFinite.cpp:#if defined(REG_HAS_MPICUDA) || defined(REG_HAS_CUDA)
src/TwoLevel/TwoLevelFinite.cpp:  DebugGPUStopEvent();
src/TwoLevel/TwoLevelFinite.cpp:  DebugGPUStartEvent("Finite Restrict");
src/TwoLevel/TwoLevelFinite.cpp:#if defined(REG_HAS_MPICUDA) || defined(REG_HAS_CUDA)
src/TwoLevel/TwoLevelFinite.cpp:  DebugGPUStopEvent();
src/TwoLevel/TwoLevelFinite.cpp:  DebugGPUStartEvent("Finite Prolong");
src/TwoLevel/TwoLevelFinite.cpp:#if defined(REG_HAS_MPICUDA) || defined(REG_HAS_CUDA)
src/TwoLevel/TwoLevelFinite.cpp:  DebugGPUStopEvent();
src/TwoLevel/TwoLevelFinite.cpp:  DebugGPUStartEvent("Finite Prolong");
src/TwoLevel/TwoLevelFinite.cpp:#if defined(REG_HAS_MPICUDA) || defined(REG_HAS_CUDA)
src/TwoLevel/TwoLevelFinite.cpp:  DebugGPUStopEvent();
src/CLAIRE.cpp:#ifdef REG_HAS_CUDA
src/CLAIRE.cpp:#include "cuda_helper.hpp"
src/CLAIRE.cpp:#ifdef REG_HAS_CUDA
src/CLAIRE.cpp:    cudaPrintDeviceMemory();
src/CLAIRE.cpp:    DebugGPUStartEvent(__FUNCTION__);
src/CLAIRE.cpp:    DebugGPUStopEvent();
src/CLAIRE.cpp:    DebugGPUStartEvent(__FUNCTION__);
src/CLAIRE.cpp:    DebugGPUStopEvent();
src/CLAIRE.cpp:#ifdef REG_HAS_CUDA
src/CLAIRE.cpp:        ver = "gpu-";
src/CLAIRE.cpp:#ifdef REG_HAS_CUDA
src/CLAIRE.cpp:        ver = "gpu-";
src/CLAIRE.cpp:    ierr = DebugGPUNotImplemented(); CHKERRQ(ierr);
src/CLAIRE.cpp:    DebugGPUStartEvent(__FUNCTION__);
src/CLAIRE.cpp:    DebugGPUStopEvent();
src/CLAIRE.cpp:    DebugGPUStartEvent(__FUNCTION__);
src/CLAIRE.cpp:    DebugGPUStopEvent();
src/CLAIRE.cpp:    ierr = DebugGPUNotImplemented(); CHKERRQ(ierr);
src/CLAIRE.cpp:    DebugGPUStartEvent(__FUNCTION__);
src/CLAIRE.cpp:    DebugGPUStopEvent();
src/CLAIRE.cpp:    DebugGPUStartEvent(__FUNCTION__);
src/CLAIRE.cpp:    DebugGPUStopEvent();
src/CLAIRE.cpp:#ifdef REG_HAS_CUDA
src/CLAIRE.cpp:        ver = "gpu-";
src/Solver/TransportKernel.cu:#include "cuda_helper.hpp"
src/Solver/TransportKernel.cu:__global__ void TransformKernelAdjointSLGPU(ScalarType *pL, ScalarType* pLnext,
src/Solver/TransportKernel.cu:__global__ void TransformKernelAdjointSLDivGPU(ScalarType *pDivV, ScalarType *pDivVx, ScalarType ht, IntType nl) {
src/Solver/TransportKernel.cu:__global__ void TransformKernelAdjointSLGPU(ScalarType* pLnext, ScalarType *pDivVx, 
src/Solver/TransportKernel.cu:__global__ void TransformKernelAdjointSLGPU(const ScalarType *pL,
src/Solver/TransportKernel.cu:__global__ void TransformKernelAdjoint0SLGPU(const ScalarType *pL,
src/Solver/TransportKernel.cu:__global__ void TransformKernelIncStateSLGPU(ScalarType *pM,
src/Solver/TransportKernel.cu:__global__ void TransformKernelIncStateSLGPU(ScalarType *pM,
src/Solver/TransportKernel.cu:#define TransformKernelIncAdjointGPU TransformKernelAdjointSLGPU
src/Solver/TransportKernel.cu:__global__ void TransformKernelEulerGPU(const ScalarType *pM,
src/Solver/TransportKernel.cu:__global__ void TransformKernelEulerGPU(const ScalarType *pM,
src/Solver/TransportKernel.cu:__global__ void TransformKernelRK2GPU(const ScalarType *pM, const ScalarType *pRHS,
src/Solver/TransportKernel.cu:__global__ void TransformKernelEulerGPU(const ScalarType *pM,
src/Solver/TransportKernel.cu:__global__ void TransformKernelEulerGPU(const ScalarType *pM, const ScalarType *pRHS,
src/Solver/TransportKernel.cu:__global__ void TransformKernelRK2GPU(const ScalarType *pM, const ScalarType *pRHS, 
src/Solver/TransportKernel.cu:__global__ void TransformKernelRK2GPU(const ScalarType *pM, const ScalarType *pRHS1, const ScalarType *pRHS2,
src/Solver/TransportKernel.cu:__global__ void TransformKernelScaleGPU(const ScalarType *pL,
src/Solver/TransportKernel.cu:__global__ void TransformKernelScaleGPU(const ScalarType *pL, const ScalarType *pLt,
src/Solver/TransportKernel.cu:__global__ void TransformKernelScaleGPU(const ScalarType *pL, const ScalarType *pLt, const ScalarType* pRHS,
src/Solver/TransportKernel.cu:__global__ void TransformKernelScaleEulerGPU(const ScalarType *pL, const ScalarType *pRHS,
src/Solver/TransportKernel.cu:__global__ void TransformKernelScaleRK2GPU(const ScalarType *pL, 
src/Solver/TransportKernel.cu:#define TransformKernelScaleAddGPU TransformKernelAdjointSLGPU
src/Solver/TransportKernel.cu:__global__ void TransformKernelContinuityGPU(const ScalarType *pMx,
src/Solver/TransportKernel.cu:  TransformKernelAdjointSLGPU<<<grid, block>>>(pL, pLnext, pLx, pDivV, pDivVx, 
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelAdjointSLGPU<<<grid, block>>>(pLnext, pDivVx, 
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelAdjointSLDivGPU<<<grid, block>>>(pDivV, pDivVx, ht, nl);
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelAdjointSLGPU<<<grid, block>>>(pL, 
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelAdjoint0SLGPU<<<grid, block>>>(pL, 
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelIncStateSLGPU<<<grid, block>>>(pMtilde, 
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelIncStateSLGPU<<<grid, block>>>(pMtilde, 
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelIncStateSLGPU<<<grid, block>>>(pMtilde, 
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelIncAdjointGPU<<<grid, block>>>(pL, 
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelEulerGPU<<<grid, block>>>(pM,
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelRK2GPU<<<grid, block>>>(pM, pRHS,
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelScaleGPU<<<grid, block>>>(pL,
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelScaleEulerGPU<<<grid, block>>>(pL, pRHS[0],
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelScaleRK2GPU<<<grid, block>>>(pL, pRHS[0], pRHS[1],
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelScaleAddGPU<<<grid, block>>>(pL,
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelScaleGPU<<<grid, block>>>(pL,
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelScaleGPU<<<grid, block>>>(pL, pLt,
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelEulerGPU<<<grid, block>>>(pLt, pRHS[0], pLtnext, ht, nl);
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelScaleGPU<<<grid, block>>>(pL, pLt, pRHS[0],
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelRK2GPU<<<grid, block>>>(pLt, pRHS[0], pRHS[1], pLtnext, 0.5*ht, nl);
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelEulerGPU<<<grid, block>>>(pMt,
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelEulerGPU<<<grid, block>>>(pMt,
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelRK2GPU<<<grid, block>>>(pMt, pRHS,
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  TransformKernelContinuityGPU<<<grid, block>>>(pMx, pDivV, pDivVx, pMnext, ht, nl);
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/Solver/TransportKernel.cu:  cudaMemcpy((void*)dest,(void*)org,sizeof(T)*ne,cudaMemcpyDeviceToDevice);
src/Solver/TransportKernel.cu:  //cudaDeviceSynchronize();
src/Solver/TransportKernel.cu:  cudaCheckKernelError();
src/UnitTests/TestInterpolation.cpp:#if defined(REG_HAS_MPICUDA) || defined(REG_HAS_CUDA)
src/UnitTests/TestInterpolation.cpp:#include "interp3_gpu_mpi.hpp"
src/UnitTests/TestInterpolation.cpp:static void printGPUMemory(int rank) {
src/UnitTests/TestInterpolation.cpp:      cudaMemGetInfo(&free, &used);
src/UnitTests/TestInterpolation.cpp:void TestErrorMultiGPU(ScalarType *ref, ScalarType *eval, IntType nl, double *err, double *maxval) {
src/UnitTests/TestInterpolation.cpp:#ifdef REG_HAS_CUDA
src/UnitTests/TestInterpolation.cpp:#ifdef REG_HAS_CUDA
src/UnitTests/TestInterpolation.cpp:  cudaMalloc((void**)(&pg), sizeof(ScalarType)*nl);
src/UnitTests/TestInterpolation.cpp:  cudaMalloc((void**)(&pq1), sizeof(ScalarType)*nl);
src/UnitTests/TestInterpolation.cpp:  cudaMalloc((void**)(&pq2), sizeof(ScalarType)*nl);
src/UnitTests/TestInterpolation.cpp:  cudaMalloc((void**)(&pq3), sizeof(ScalarType)*nl);
src/UnitTests/TestInterpolation.cpp:  cudaMalloc((void**)(&pe), sizeof(ScalarType)*nl);
src/UnitTests/TestInterpolation.cpp:  cudaMalloc((void**)(&tmp1), sizeof(ScalarType)*nl);
src/UnitTests/TestInterpolation.cpp:  cudaMalloc((void**)(&tmp2), sizeof(ScalarType)*nl);
src/UnitTests/TestInterpolation.cpp:  cudaMemcpy(pg, grid, sizeof(ScalarType)*nl, cudaMemcpyHostToDevice);
src/UnitTests/TestInterpolation.cpp:  cudaMemcpy(pq1, q1, sizeof(ScalarType)*nl, cudaMemcpyHostToDevice);
src/UnitTests/TestInterpolation.cpp:  cudaMemcpy(pq2, q2, sizeof(ScalarType)*nl, cudaMemcpyHostToDevice);
src/UnitTests/TestInterpolation.cpp:  cudaMemcpy(pq3, q3, sizeof(ScalarType)*nl, cudaMemcpyHostToDevice);
src/UnitTests/TestInterpolation.cpp:  cudaTextureObject_t tex = gpuInitEmptyTexture(nx);
src/UnitTests/TestInterpolation.cpp:    gpuInterp3D(pg, pq, pe, tmp1, tmp2, nx, nl, tex, m_Opt->m_PDESolver.iporder, &timer);
src/UnitTests/TestInterpolation.cpp:    cudaDeviceSynchronize();
src/UnitTests/TestInterpolation.cpp:    gpuInterp3D(pg, pq, pe, tmp1, tmp2, nx, nl, tex, m_Opt->m_PDESolver.iporder, &timer);
src/UnitTests/TestInterpolation.cpp:    cudaDeviceSynchronize();
src/UnitTests/TestInterpolation.cpp:    cudaMemcpy(eval, pe, sizeof(ScalarType)*nl, cudaMemcpyDeviceToHost);
src/UnitTests/TestInterpolation.cpp:    cudaDeviceSynchronize();
src/UnitTests/TestInterpolation.cpp:  cudaMemcpy(pq1, q1, sizeof(ScalarType)*nl, cudaMemcpyHostToDevice);
src/UnitTests/TestInterpolation.cpp:  cudaMemcpy(pq2, q2, sizeof(ScalarType)*nl, cudaMemcpyHostToDevice);
src/UnitTests/TestInterpolation.cpp:  cudaMemcpy(pq3, q3, sizeof(ScalarType)*nl, cudaMemcpyHostToDevice);
src/UnitTests/TestInterpolation.cpp:    gpuInterp3D(pg, pq, pe, tmp1, tmp2, nx, nl, tex, m_Opt->m_PDESolver.iporder, &timer);
src/UnitTests/TestInterpolation.cpp:    cudaDeviceSynchronize();
src/UnitTests/TestInterpolation.cpp:    gpuInterp3D(pg, pq, pe, tmp1, tmp2, nx, nl, tex, m_Opt->m_PDESolver.iporder, &timer);
src/UnitTests/TestInterpolation.cpp:    cudaDeviceSynchronize();
src/UnitTests/TestInterpolation.cpp:    cudaMemcpy(eval, pe, sizeof(ScalarType)*nl, cudaMemcpyDeviceToHost);
src/UnitTests/TestInterpolation.cpp:    cudaDeviceSynchronize();
src/UnitTests/TestInterpolation.cpp:  cudaFree(pg);
src/UnitTests/TestInterpolation.cpp:  cudaFree(q1);
src/UnitTests/TestInterpolation.cpp:  cudaFree(q2);
src/UnitTests/TestInterpolation.cpp:  cudaFree(q3);
src/UnitTests/TestInterpolation.cpp:  cudaFree(pe);
src/UnitTests/TestInterpolation.cpp:  cudaFree(tmp1);
src/UnitTests/TestInterpolation.cpp:  cudaFree(tmp2);
src/UnitTests/TestInterpolation.cpp:  cudaDestroyTextureObject(tex);
src/UnitTests/TestInterpolation.cpp:PetscErrorCode TestInterpolationMultiGPU(RegOpt *m_Opt) {
src/UnitTests/TestInterpolation.cpp:  ScalarType m_GPUtime;
src/UnitTests/TestInterpolation.cpp:  Interp3_Plan_GPU* interp_plan = nullptr;
src/UnitTests/TestInterpolation.cpp:  cudaGetDevice ( &device );
src/UnitTests/TestInterpolation.cpp:  cudaDeviceGetPCIBusId ( pciBusId, 512, device );
src/UnitTests/TestInterpolation.cpp:  cudaMalloc((void**)&p_fghost, g_alloc_max);
src/UnitTests/TestInterpolation.cpp:  cudaTextureObject_t tex = gpuInitEmptyTexture(isize_g);   
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDAGetArray(xq, &p_xq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDAGetArray(yq, &p_yq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDAGetArray(zq, &p_zq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDAGetArray(f, &p_f); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDAGetArray(ref, &p_ref); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDARestoreArray(ref, &p_ref); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDARestoreArray(f,  &p_f); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDARestoreArray(zq, &p_zq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDARestoreArray(yq, &p_yq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDARestoreArray(xq, &p_xq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDAGetArray(xq, &p_xq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDAGetArray(yq, &p_yq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDAGetArray(zq, &p_zq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDARestoreArray(zq, &p_zq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDARestoreArray(yq, &p_yq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDARestoreArray(xq, &p_xq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:    ierr = VecCUDAGetArray(f, &p_f);  CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:    ierr = VecCUDARestoreArray(f, &p_f); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:    ierr = VecCUDAGetArray(fout, &p_fout); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:                              &m_GPUtime, 0, flag);
src/UnitTests/TestInterpolation.cpp:    ierr = VecCUDARestoreArray(fout, &p_fout); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:      TestErrorMultiGPU(p_ref, p_fout, nl, &error, &max);
src/UnitTests/TestInterpolation.cpp:    ierr = VecCUDAGetArray(f, &p_f);  CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:    ierr = VecCUDARestoreArray(f, &p_f); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:    ierr = VecCUDAGetArray(fout, &p_fout); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:                              &m_GPUtime, 0, flag);
src/UnitTests/TestInterpolation.cpp:    ierr = VecCUDARestoreArray(fout, &p_fout); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  cudaDestroyTextureObject(tex);
src/UnitTests/TestInterpolation.cpp:    cudaFree(p_fghost);
src/UnitTests/TestInterpolation.cpp:PetscErrorCode TestVectorFieldInterpolationMultiGPU(RegOpt *m_Opt) {
src/UnitTests/TestInterpolation.cpp:  ScalarType m_GPUtime;
src/UnitTests/TestInterpolation.cpp:  Interp3_Plan_GPU* interp_plan = nullptr;
src/UnitTests/TestInterpolation.cpp:  cudaGetDevice ( &device );
src/UnitTests/TestInterpolation.cpp:  cudaDeviceGetPCIBusId ( pciBusId, 512, device );
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDAGetArray(xq, &p_xq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDAGetArray(yq, &p_yq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDAGetArray(zq, &p_zq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDARestoreArray(zq, &p_zq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDARestoreArray(yq, &p_yq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDARestoreArray(xq, &p_xq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  cudaMalloc((void**)&p_fghost, 3*g_alloc_max);
src/UnitTests/TestInterpolation.cpp:  cudaTextureObject_t tex = gpuInitEmptyTexture(isize_g);   
src/UnitTests/TestInterpolation.cpp:      interp_plan = new Interp3_Plan_GPU(g_alloc_max, true); 
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDAGetArray(xq, &p_xq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDAGetArray(yq, &p_yq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDAGetArray(zq, &p_zq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDARestoreArray(zq, &p_zq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDARestoreArray(yq, &p_yq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDARestoreArray(xq, &p_xq); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDAGetArray(fout, &p_fout); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:                            &m_GPUtime, 1, "state");
src/UnitTests/TestInterpolation.cpp:  ierr = VecCUDARestoreArray(fout, &p_fout); CHKERRQ(ierr);
src/UnitTests/TestInterpolation.cpp:  cudaDestroyTextureObject(tex);
src/UnitTests/TestInterpolation.cpp:#if defined(REG_HAS_MPICUDA)
src/UnitTests/TestInterpolation.cpp:    cudaFree(p_fghost);
src/UnitTests/TestDifferentiation.cpp:#ifdef REG_HAS_CUDA
src/UnitTests/TestDifferentiation.cpp:#ifdef REG_HAS_CUDA
src/UnitTests/TestDifferentiation.cpp:  printf("rank %2i uses GPU %2i\n", rank, m_Opt->m_gpu_id);
src/UnitTests/TestDifferentiation.cpp:    cudaMalloc(&spectral, fft.nalloc);
src/UnitTests/TestDifferentiation.cpp:    cudaMalloc(&real, fft.nalloc);
src/UnitTests/TestClaire.cpp:PetscErrorCode TestTrajectoryMultiGPU(reg::RegOpt *m_Opt) {
src/UnitTests/TestClaire.cpp://#ifdef REG_HAS_CUDA
src/UnitTests/TestClaire.cpp://  reg::SemiLagrangianGPUNew *sl = nullptr;
src/UnitTests/TestClaire.cpp:#if defined(REG_HAS_CUDA) || defined(REG_HAS_MPICUDA)
src/UnitTests/TestClaire.cpp:  ierr = VecCUDAGetArray(X, &pX); CHKERRQ(ierr);
src/UnitTests/TestClaire.cpp:  ierr = VecCUDARestoreArray(X, &pX); CHKERRQ(ierr);
src/UnitTests/TestClaire.cpp:  ierr = VecCUDAGetArray(X, &pX); CHKERRQ(ierr);
src/UnitTests/TestClaire.cpp:  ierr = VecCUDARestoreArray(X, &pX); CHKERRQ(ierr);
src/UnitTests/TestClaire.cpp:  ierr = VecCUDAGetArray(X, &pX); CHKERRQ(ierr);
src/UnitTests/TestClaire.cpp:  ierr = VecCUDARestoreArray(X, &pX); CHKERRQ(ierr);
src/UnitTests/TestClaire.cpp:  ierr = VecCUDAGetArray(X, &pX); CHKERRQ(ierr);
src/UnitTests/TestClaire.cpp:  ierr = VecCUDARestoreArray(X, &pX); CHKERRQ(ierr);
src/DistanceMeasure/DistanceMeasureSL2.cpp:/*#ifdef REG_HAS_CUDA
src/DistanceMeasure/DistanceMeasureSL2.cpp:        DistanceMeasureSetFinalMaskGPU(&p_l[ll],&p_m[l],p_mr,p_w,nl,nc);
src/DistanceMeasure/DistanceMeasureSL2.cpp:/*#ifdef REG_HAS_CUDA
src/DistanceMeasure/DistanceMeasureSL2.cpp:        DistanceMeasureSetFinalGPU(&p_l[ll],&p_m[l],p_mr,nc*nl);
src/DistanceMeasure/DistanceMeasure.cpp:#if defined(REG_HAS_CUDA) || defined(REG_HAS_MPICUDA)
src/DistanceMeasure/DistanceMeasure.cpp:        ierr = VecSetType(this->m_ObjWts, VECSEQCUDA); CHKERRQ(ierr);
src/DistanceMeasure/DistanceMeasureKernel.cu:#include "cuda_helper.hpp"
src/DistanceMeasure/DistanceMeasureKernel.cu:__global__ void VecSubMulGPU(ScalarType *pL, const ScalarType *pW, const ScalarType *pWts,
src/DistanceMeasure/DistanceMeasureKernel.cu:__global__ void VecSubGPU(ScalarType *pL, const ScalarType *pWts, const ScalarType *pMr, 
src/DistanceMeasure/DistanceMeasureKernel.cu:__global__ void VecMulGPU(ScalarType *pL, const ScalarType *pW, const ScalarType *pWts,
src/DistanceMeasure/DistanceMeasureKernel.cu:__global__ void VecNegGPU(ScalarType *pL, const ScalarType *pWts, const ScalarType *pM, int nl) {
src/DistanceMeasure/DistanceMeasureKernel.cu:__global__ void DistanceMeasureFunctionalGPU(ScalarType *res, 
src/DistanceMeasure/DistanceMeasureKernel.cu:__global__ void DistanceMeasureFunctionalGPU(ScalarType *res, 
src/DistanceMeasure/DistanceMeasureKernel.cu:  DistanceMeasureFunctionalGPU<256><<<grid, block>>>(res, pW, pWts, pMr, pM, nl);
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaDeviceSynchronize();
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaCheckKernelError();
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaDeviceSynchronize();
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaCheckKernelError();
src/DistanceMeasure/DistanceMeasureKernel.cu:  ierr = cudaMemcpy(reinterpret_cast<void*>(&value), reinterpret_cast<void*>(res), sizeof(ScalarType), cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
src/DistanceMeasure/DistanceMeasureKernel.cu:  DistanceMeasureFunctionalGPU<256><<<grid, block>>>(res, pWts, pMr, pM, nl);
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaDeviceSynchronize();
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaCheckKernelError();
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaDeviceSynchronize();
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaCheckKernelError();
src/DistanceMeasure/DistanceMeasureKernel.cu:  ierr = cudaMemcpy(reinterpret_cast<void*>(&value), reinterpret_cast<void*>(res), sizeof(ScalarType), cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
src/DistanceMeasure/DistanceMeasureKernel.cu:  VecSubGPU<<<grid, block>>>(pL, pWts, pMr, pM, nl);
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaDeviceSynchronize();
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaCheckKernelError();
src/DistanceMeasure/DistanceMeasureKernel.cu:  VecSubMulGPU<<<grid, block>>>(pL, pW, pWts, pMr, pM, nl);
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaDeviceSynchronize();
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaCheckKernelError();
src/DistanceMeasure/DistanceMeasureKernel.cu:  VecNegGPU<<<grid, block>>>(pL, pWts, pM, nl);
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaDeviceSynchronize();
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaCheckKernelError();
src/DistanceMeasure/DistanceMeasureKernel.cu:  VecMulGPU<<<grid, block>>>(pL, pW, pWts, pM, nl);
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaDeviceSynchronize();
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaCheckKernelError();
src/DistanceMeasure/DistanceMeasureKernel.cu:  DistanceMeasureFunctionalGPU<256><<<grid, block>>>(res, pWts, pMr, pMt, nl);
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaDeviceSynchronize();
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaCheckKernelError();
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaDeviceSynchronize();
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaCheckKernelError();
src/DistanceMeasure/DistanceMeasureKernel.cu:  ierr = cudaMemcpy(reinterpret_cast<void*>(&norm_l2_loc), reinterpret_cast<void*>(res), sizeof(ScalarType), cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaCheckKernelError();
src/DistanceMeasure/DistanceMeasureKernel.cu:  cudaCheckKernelError();
src/Differentiation/DifferentiationFD.cpp:#ifdef REG_HAS_CUDA
src/Differentiation/DifferentiationFD.cpp:#ifdef REG_HAS_CUDA
src/Differentiation/DifferentiationFD.cpp:#if defined(REG_HAS_CUDA) || defined(REG_HAS_MPICUDA)
src/Differentiation/DifferentiationFD.cpp:      this->mtex = gpuInitEmptyGradientTexture(this->m_Opt->m_Domain.nx);
src/Differentiation/DifferentiationFD.cpp:      cudaMalloc((void**)&this->m_Ghost, g_alloc_max);
src/Differentiation/DifferentiationFD.cpp:      // ghost data mem alloc on GPU
src/Differentiation/DifferentiationFD.cpp:      //cudaMalloc((void**)&this->d_Ghost, this->nlghost*sizeof(ScalarType));
src/Differentiation/DifferentiationFD.cpp:      this->mtex = gpuInitEmptyGradientTexture(isizeg); CHKERRQ(ierr);
src/Differentiation/DifferentiationFD.cpp:#if defined(REG_HAS_CUDA)
src/Differentiation/DifferentiationFD.cpp:        cudaDestroyTextureObject(this->mtex);
src/Differentiation/DifferentiationFD.cpp:      cudaFree(this->m_Ghost);
src/Differentiation/DifferentiationFD.cpp:      cudaFree(this->d_Ghost);
src/Differentiation/DifferentiationFD.cpp:    DebugGPUStartEvent("FD Grad");
src/Differentiation/DifferentiationFD.cpp:#if defined(REG_HAS_MPICUDA) || defined(REG_HAS_CUDA)
src/Differentiation/DifferentiationFD.cpp:      //ierr = cudaMemcpy((void*)this->m_Work, (const void*)m, sizeof(ScalarType)*this->m_Opt->m_Domain.nl, cudaMemcpyDeviceToHost);  CHKERRCUDA(ierr);
src/Differentiation/DifferentiationFD.cpp:      //ierr = cudaMemcpy((void*)this->d_Ghost, (const void*)this->m_Ghost, this->nlghost*sizeof(ScalarType), cudaMemcpyHostToDevice); CHKERRCUDA(ierr);
src/Differentiation/DifferentiationFD.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationFD.cpp:    DebugGPUStartEvent("FD Laplacian");
src/Differentiation/DifferentiationFD.cpp:#if defined(REG_HAS_MPICUDA) || defined(REG_HAS_CUDA)
src/Differentiation/DifferentiationFD.cpp:      //ierr = cudaMemcpy((void*)this->m_Work, (const void*)m, sizeof(ScalarType)*this->m_Opt->m_Domain.nl, cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
src/Differentiation/DifferentiationFD.cpp:      //ierr = cudaMemcpy((void*)this->d_Ghost, (const void*)this->m_Ghost, this->nlghost*sizeof(ScalarType), cudaMemcpyHostToDevice); CHKERRCUDA(ierr);
src/Differentiation/DifferentiationFD.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationFD.cpp:    DebugGPUStartEvent("FD Laplacian");
src/Differentiation/DifferentiationFD.cpp:#if defined(REG_HAS_MPICUDA) || defined(REG_HAS_CUDA)
src/Differentiation/DifferentiationFD.cpp:        //ierr = cudaMemcpy((void*)this->m_Work, (const void*)pv[i], sizeof(ScalarType)*this->m_Opt->m_Domain.nl, cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
src/Differentiation/DifferentiationFD.cpp:        //ierr = cudaMemcpy((void*)this->d_Ghost, (const void*)this->m_Ghost, this->nlghost*sizeof(ScalarType), cudaMemcpyHostToDevice); CHKERRCUDA(ierr);
src/Differentiation/DifferentiationFD.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationFD.cpp:    DebugGPUStartEvent("FD Divergence");
src/Differentiation/DifferentiationFD.cpp:#if defined(REG_HAS_MPICUDA) || defined(REG_HAS_CUDA)
src/Differentiation/DifferentiationFD.cpp:      cudaMemset((void*)l, 0, this->m_Opt->m_Domain.nl*sizeof(ScalarType));
src/Differentiation/DifferentiationFD.cpp:      //ierr = cudaMemcpy((void*)this->m_Work, (const void*)v3, sizeof(ScalarType)*this->m_Opt->m_Domain.nl, cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
src/Differentiation/DifferentiationFD.cpp:      //ierr = cudaMemcpy((void*)this->d_Ghost, (const void*)this->m_Ghost, this->nlghost*sizeof(ScalarType), cudaMemcpyHostToDevice); CHKERRCUDA(ierr);
src/Differentiation/DifferentiationFD.cpp:      //ierr = cudaMemcpy((void*)this->m_Work, (const void*)v2, sizeof(ScalarType)*this->m_Opt->m_Domain.nl, cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
src/Differentiation/DifferentiationFD.cpp:      //ierr = cudaMemcpy((void*)this->d_Ghost, (const void*)this->m_Ghost, this->nlghost*sizeof(ScalarType), cudaMemcpyHostToDevice); CHKERRCUDA(ierr);
src/Differentiation/DifferentiationFD.cpp:      //ierr = cudaMemcpy((void*)this->m_Work, (const void*)v1, sizeof(ScalarType)*this->m_Opt->m_Domain.nl, cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
src/Differentiation/DifferentiationFD.cpp:      //ierr = cudaMemcpy((void*)this->d_Ghost, (const void*)this->m_Ghost, this->nlghost*sizeof(ScalarType), cudaMemcpyHostToDevice); CHKERRCUDA(ierr);
src/Differentiation/DifferentiationFD.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationFD.cpp:  DebugGPUStartEvent("FD Regularization");
src/Differentiation/DifferentiationFD.cpp:#if defined(REG_HAS_MPICUDA) || defined(REG_HAS_CUDA)
src/Differentiation/DifferentiationFD.cpp:        //ierr = cudaMemcpy((void*)this->m_Work, (const void*)pV[i], sizeof(ScalarType)*this->m_Opt->m_Domain.nl, cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
src/Differentiation/DifferentiationFD.cpp:        //ierr = cudaMemcpy((void*)this->d_Ghost, (const void*)this->m_Ghost, this->nlghost*sizeof(ScalarType), cudaMemcpyHostToDevice); CHKERRCUDA(ierr);
src/Differentiation/DifferentiationFD.cpp:  DebugGPUStopEvent();
src/Differentiation/DifferentiationFD.cpp:#if defined(REG_HAS_MPICUDA) || defined(REG_HAS_CUDA)
src/Differentiation/DifferentiationFD.cpp:        //ierr = cudaMemcpy((void*)this->m_Work, (const void*)pV[i], sizeof(ScalarType)*this->m_Opt->m_Domain.nl, cudaMemcpyDeviceToHost); CHKERRCUDA(ierr);
src/Differentiation/DifferentiationFD.cpp:        //ierr = cudaMemcpy((void*)this->d_Ghost, (const void*)this->m_Ghost, this->nlghost*sizeof(ScalarType), cudaMemcpyHostToDevice); CHKERRCUDA(ierr);
src/Differentiation/DifferentiationKernel.cu:#include "cuda_helper.hpp"
src/Differentiation/DifferentiationKernel.cu:using KernelUtils::SpectralKernelCallGPU;
src/Differentiation/DifferentiationKernel.cu:  ierr = SpectralKernelCallGPU<NLaplacianKernel<1> >(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:  ierr = SpectralKernelCallGPU<NLaplacianModKernel<1> >(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:    ierr = SpectralKernelCallGPU<NLaplacianKernel<1> >(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:    ierr = SpectralKernelCallGPU<RelaxedNLaplacianKernel<1> >(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:    ierr = SpectralKernelCallGPU<NLaplacianFilterKernel<1> >(nstart, nx, nl, v, 
src/Differentiation/DifferentiationKernel.cu:    ierr = SpectralKernelCallGPU<RelaxedNLaplacianKernel<1> >(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:    ierr = SpectralKernelCallGPU<NLaplacianKernel<2> >(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:    ierr = SpectralKernelCallGPU<RelaxedNLaplacianKernel<2> >(nstart, nx, nl,
src/Differentiation/DifferentiationKernel.cu:    ierr = SpectralKernelCallGPU<NLaplacianKernel<3> >(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:    ierr = SpectralKernelCallGPU<RelaxedNLaplacianKernel<3> >(nstart, nx, nl,
src/Differentiation/DifferentiationKernel.cu:      ierr = SpectralKernelCallGPU<InverseNLaplacianSqrtKernel<1> >(nstart, nx, nl,
src/Differentiation/DifferentiationKernel.cu:      ierr = SpectralKernelCallGPU<RelaxedInverseNLaplacianSqrtKernel<1> >(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:      ierr = SpectralKernelCallGPU<InverseNLaplacianKernel<1> >(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:      ierr = SpectralKernelCallGPU<RelaxedInverseNLaplacianKernel<1> >(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:      ierr = SpectralKernelCallGPU<InverseNLaplacianKernel<1> >(nstart, nx, nl,
src/Differentiation/DifferentiationKernel.cu:      ierr = SpectralKernelCallGPU<RelaxedInverseNLaplacianSqrtKernel<2> >(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:      ierr = SpectralKernelCallGPU<InverseNLaplacianKernel<2> >(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:      ierr = SpectralKernelCallGPU<RelaxedInverseNLaplacianKernel<2> >(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:      ierr = SpectralKernelCallGPU<InverseNLaplacianSqrtKernel<3> >(nstart, nx, nl,
src/Differentiation/DifferentiationKernel.cu:      ierr = SpectralKernelCallGPU<RelaxedInverseNLaplacianSqrtKernel<3> >(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:      ierr = SpectralKernelCallGPU<InverseNLaplacianKernel<3> >(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:      ierr = SpectralKernelCallGPU<RelaxedInverseNLaplacianKernel<3> >(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:  ierr = SpectralKernelCallGPU<LerayKernel>(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:  ierr = SpectralKernelCallGPU<LerayKernel>(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:  ierr = SpectralKernelCallGPU<InverseNLaplacianKernel<1> >(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:  ierr = SpectralKernelCallGPU<GaussianFilterKernel>(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:  ierr = SpectralKernelCallGPU<GradientKernel>(nstart, nx, nl, 
src/Differentiation/DifferentiationKernel.cu:  ierr = SpectralKernelCallGPU<DivergenceKernel>(nstart, nx, nl, 
src/Differentiation/DifferentiationSM.cpp://#ifdef REG_HAS_CUDA
src/Differentiation/DifferentiationSM.cpp:/*#ifdef REG_HAS_CUDA
src/Differentiation/DifferentiationSM.cpp:/*#ifdef REG_HAS_CUDA
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStartEvent("FFT Grad");
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStartEvent("FFT Laplacian");
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStartEvent("FFT Laplacian Field");
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStartEvent("FFT Divergence");
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStartEvent("FFT modified laplacian regularization operator");
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStartEvent("FFT laplacian regularization operator");
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStartEvent("FFT inverse bilaplacian regularization operator");
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStartEvent("FFT trilaplacian regularization operator");
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStartEvent("FFT trilaplacian regularization functional");
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStartEvent("FFT inverse laplacian regularization operator");
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStartEvent("FFT inverse bilaplacian regularization operator");
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStartEvent("FFT inverse trilaplacian regularization operator");
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStartEvent("FFT leray operator");
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStartEvent("FFT leray operator");
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStartEvent("FFT Gaussian filter");
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/DifferentiationSM.cpp:    DebugGPUStopEvent();
src/Differentiation/TextureDifferentiationKernel.cu:#include "cuda_helper.hpp"
src/Differentiation/TextureDifferentiationKernel.cu:#include "cuda_profiler_api.h"
src/Differentiation/TextureDifferentiationKernel.cu:__global__ void mgpu_gradient_z(ScalarType* dfz, const ScalarType* f, int3 nl, int3 ng, int3 halo, const float ih) {
src/Differentiation/TextureDifferentiationKernel.cu:__global__ void mgpu_gradient_y(ScalarType* dfy, const ScalarType* f, int3 nl, int3 ng, int3 halo, const float ih) {
src/Differentiation/TextureDifferentiationKernel.cu:__global__ void mgpu_gradient_x(ScalarType* dfx, const ScalarType* f, int3 nl, int3 ng, int3 halo, const float ih) {
src/Differentiation/TextureDifferentiationKernel.cu:__global__ void mgpu_d_zz(ScalarType* ddf, const ScalarType* f, const ScalarType beta, int3 nl, int3 ng, int3 halo, const float ih2) {
src/Differentiation/TextureDifferentiationKernel.cu:__global__ void mgpu_d_yy(ScalarType* ddf, const ScalarType* f, const ScalarType beta, int3 nl, int3 ng, int3 halo, const float ih2) {
src/Differentiation/TextureDifferentiationKernel.cu:__global__ void mgpu_d_xx(ScalarType* ddf, const ScalarType* f, const ScalarType beta, int3 nl, int3 ng, int3 halo, const float ih2) {
src/Differentiation/TextureDifferentiationKernel.cu: * @brief compute z-component of gradient using 8th order finite differencing (single GPU version)
src/Differentiation/TextureDifferentiationKernel.cu: * @brief compute y-component of gradient using 8th order finite differencing (single GPU version)
src/Differentiation/TextureDifferentiationKernel.cu: * @brief compute x-component of gradient using 8th order finite differencing (single GPU version)
src/Differentiation/TextureDifferentiationKernel.cu: * @brief compute laplacian using 8th order finite differencing (single GPU code)
src/Differentiation/TextureDifferentiationKernel.cu: * @brief compute laplacian using 8th order finite differencing (single GPU code)
src/Differentiation/TextureDifferentiationKernel.cu: * @brief compute laplacian using 8th order finite differencing (single GPU code)
src/Differentiation/TextureDifferentiationKernel.cu:__global__ void TextureDivXComputeKernel(cudaTextureObject_t tex, ScalarType* div, int3 nl, const float3 inx, const float ih) {
src/Differentiation/TextureDifferentiationKernel.cu:__global__ void TextureDivYComputeKernel(cudaTextureObject_t tex, ScalarType* div, int3 nl, const float3 inx, const float ih) {
src/Differentiation/TextureDifferentiationKernel.cu:__global__ void TextureDivZComputeKernel(cudaTextureObject_t tex, ScalarType* div, int3 nl, const float3 inx, const float ih) {
src/Differentiation/TextureDifferentiationKernel.cu:__global__ void TextureGradientComputeKernel(cudaTextureObject_t tex, ScalarType* dmx, ScalarType* dmy, ScalarType* dmz, int3 nl, const float3 inx, const float3 ih) {
src/Differentiation/TextureDifferentiationKernel.cu:cudaTextureObject_t gpuInitEmptyGradientTexture(IntType *nx) {
src/Differentiation/TextureDifferentiationKernel.cu:   cudaTextureObject_t texObj = 0;
src/Differentiation/TextureDifferentiationKernel.cu:   cudaError_t err = cudaSuccess;
src/Differentiation/TextureDifferentiationKernel.cu:   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
src/Differentiation/TextureDifferentiationKernel.cu:   cudaExtent extent = make_cudaExtent(nx[2], nx[1], nx[0]);
src/Differentiation/TextureDifferentiationKernel.cu:   cudaArray* cuArray;
src/Differentiation/TextureDifferentiationKernel.cu:   err = cudaMalloc3DArray(&cuArray, &channelDesc, extent, 0);
src/Differentiation/TextureDifferentiationKernel.cu:   if (err != cudaSuccess){
src/Differentiation/TextureDifferentiationKernel.cu:        fprintf(stderr, "Failed to allocate 3D cudaArray (error code %s)!\n", cudaGetErrorString(err));
src/Differentiation/TextureDifferentiationKernel.cu:    /* create cuda resource description */
src/Differentiation/TextureDifferentiationKernel.cu:    struct cudaResourceDesc resDesc;
src/Differentiation/TextureDifferentiationKernel.cu:    resDesc.resType = cudaResourceTypeArray;
src/Differentiation/TextureDifferentiationKernel.cu:    struct cudaTextureDesc texDesc;
src/Differentiation/TextureDifferentiationKernel.cu:    texDesc.addressMode[0] = cudaAddressModeWrap;
src/Differentiation/TextureDifferentiationKernel.cu:    texDesc.addressMode[1] = cudaAddressModeWrap;
src/Differentiation/TextureDifferentiationKernel.cu:    texDesc.addressMode[2] = cudaAddressModeWrap;
src/Differentiation/TextureDifferentiationKernel.cu:    texDesc.readMode = cudaReadModeElementType;
src/Differentiation/TextureDifferentiationKernel.cu:    texDesc.filterMode = cudaFilterModePoint;
src/Differentiation/TextureDifferentiationKernel.cu:    err = cudaCreateTextureObject( &texObj, &resDesc, &texDesc, NULL);
src/Differentiation/TextureDifferentiationKernel.cu:    if (err != cudaSuccess){
src/Differentiation/TextureDifferentiationKernel.cu:        fprintf(stderr, "Failed to create texture (error code %s)!\n", cudaGetErrorString(err));
src/Differentiation/TextureDifferentiationKernel.cu: * @brief update texture object by copying volume data to 3D cudaArray container
src/Differentiation/TextureDifferentiationKernel.cu:void updateTextureFromVolume(cudaPitchedPtr volume, cudaExtent extent, cudaTextureObject_t texObj) {
src/Differentiation/TextureDifferentiationKernel.cu:    cudaError_t err = cudaSuccess;
src/Differentiation/TextureDifferentiationKernel.cu:    /* create cuda resource description */
src/Differentiation/TextureDifferentiationKernel.cu:    struct cudaResourceDesc resDesc;
src/Differentiation/TextureDifferentiationKernel.cu:    cudaGetTextureObjectResourceDesc( &resDesc, texObj);
src/Differentiation/TextureDifferentiationKernel.cu:    cudaMemcpy3DParms p = {0};
src/Differentiation/TextureDifferentiationKernel.cu:    p.kind = cudaMemcpyDeviceToDevice;
src/Differentiation/TextureDifferentiationKernel.cu:    err = cudaMemcpy3D(&p);
src/Differentiation/TextureDifferentiationKernel.cu:    if (err != cudaSuccess){
src/Differentiation/TextureDifferentiationKernel.cu:        fprintf(stderr, "Failed to copy 3D memory to cudaArray (error code %s)!\n", cudaGetErrorString(err));
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(halo.x, &halo[0], sizeof(int), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(halo.y, &halo[1], sizeof(int), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(halo.z, &halo[2], sizeof(int), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(nl.x, &isize[0], sizeof(int), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(nl.y, &isize[1], sizeof(int), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(nl.z, &isize[2], sizeof(int), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(nl.x, &isize[0], sizeof(int), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(nl.y, &isize[1], sizeof(int), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(nl.z, &isize[2], sizeof(int), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(ng.x, &isize_g[0], sizeof(int), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(ng.y, &isize_g[1], sizeof(int), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(ng.z, &isize_g[2], sizeof(int), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(inx.x, &inv_nx.x, sizeof(float), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(inx.y, &inv_nx.y, sizeof(float), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(inx.z, &inv_nx.z, sizeof(float), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(d_invhx, &inv_hx.x, sizeof(float), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(d_invhy, &inv_hx.y, sizeof(float), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(d_invhz, &inv_hx.z, sizeof(float), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(d_cx, h_ct, sizeof(float)*HALO, 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(d_cy, h_ct, sizeof(float)*HALO, 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(d_cz, h_ct, sizeof(float)*HALO, 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(d_cxx, h_ct, sizeof(float)*(HALO+1), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(d_cyy, h_ct, sizeof(float)*(HALO+1), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:  //cudaMemcpyToSymbol(d_czz, h_ct, sizeof(float)*(HALO+1), 0, cudaMemcpyHostToDevice);
src/Differentiation/TextureDifferentiationKernel.cu:PetscErrorCode computeDivergence(ScalarType* l, const ScalarType* g1, const ScalarType* g2, const ScalarType* g3, cudaTextureObject_t mtex, IntType* nx, IntType* nghost, IntType* nhalo, ScalarType* hx, bool mgpu) {
src/Differentiation/TextureDifferentiationKernel.cu:  if (mgpu) {
src/Differentiation/TextureDifferentiationKernel.cu:    ierr = cudaMemset((void*)l, 0, count); CHKERRCUDA(ierr);
src/Differentiation/TextureDifferentiationKernel.cu:  // create a cudaExtent for input resolution
src/Differentiation/TextureDifferentiationKernel.cu:  cudaExtent extent = make_cudaExtent(nx[2], nx[1], nx[0]);
src/Differentiation/TextureDifferentiationKernel.cu:  cudaPitchedPtr m_cudaPitchedPtr;
src/Differentiation/TextureDifferentiationKernel.cu:  // make input image a cudaPitchedPtr for m
src/Differentiation/TextureDifferentiationKernel.cu:  m_cudaPitchedPtr = make_cudaPitchedPtr((void*)(g1), nx[2]*sizeof(ScalarType), nx[2], nx[1]);
src/Differentiation/TextureDifferentiationKernel.cu:  updateTextureFromVolume(m_cudaPitchedPtr, extent, mtex);
src/Differentiation/TextureDifferentiationKernel.cu:  cudaCheckKernelError();
src/Differentiation/TextureDifferentiationKernel.cu:  cudaDeviceSynchronize();
src/Differentiation/TextureDifferentiationKernel.cu:  // make input image a cudaPitchedPtr for m
src/Differentiation/TextureDifferentiationKernel.cu:  m_cudaPitchedPtr = make_cudaPitchedPtr((void*)(g2), nx[2]*sizeof(ScalarType), nx[2], nx[1]);
src/Differentiation/TextureDifferentiationKernel.cu:  updateTextureFromVolume(m_cudaPitchedPtr, extent, mtex);
src/Differentiation/TextureDifferentiationKernel.cu:  cudaCheckKernelError();
src/Differentiation/TextureDifferentiationKernel.cu:  cudaDeviceSynchronize();
src/Differentiation/TextureDifferentiationKernel.cu:  // make input image a cudaPitchedPtr for m
src/Differentiation/TextureDifferentiationKernel.cu:  m_cudaPitchedPtr = make_cudaPitchedPtr((void*)(g3), nx[2]*sizeof(ScalarType), nx[2], nx[1]);
src/Differentiation/TextureDifferentiationKernel.cu:  updateTextureFromVolume(m_cudaPitchedPtr, extent, mtex);
src/Differentiation/TextureDifferentiationKernel.cu:  cudaCheckKernelError();
src/Differentiation/TextureDifferentiationKernel.cu:  cudaDeviceSynchronize();
src/Differentiation/TextureDifferentiationKernel.cu:  if (mgpu)
src/Differentiation/TextureDifferentiationKernel.cu:    mgpu_gradient_z<<<numBlocks_z, threadsPerBlock_z>>>(l,g3,nl,ng,halo, 1./hx[2]);
src/Differentiation/TextureDifferentiationKernel.cu:  cudaCheckKernelError();
src/Differentiation/TextureDifferentiationKernel.cu:  if (mgpu)
src/Differentiation/TextureDifferentiationKernel.cu:    mgpu_gradient_y<<<numBlocks_y, threadsPerBlock_y>>>(l, g2,nl,ng,halo, 1./hx[1]);
src/Differentiation/TextureDifferentiationKernel.cu:  cudaCheckKernelError();
src/Differentiation/TextureDifferentiationKernel.cu:  if (mgpu)
src/Differentiation/TextureDifferentiationKernel.cu:    mgpu_gradient_x<<<numBlocks_x, threadsPerBlock_x>>>(l, g1,nl,ng,halo, 1./hx[0]);
src/Differentiation/TextureDifferentiationKernel.cu:  cudaCheckKernelError();
src/Differentiation/TextureDifferentiationKernel.cu:  cudaDeviceSynchronize();
src/Differentiation/TextureDifferentiationKernel.cu:PetscErrorCode computeDivergenceZ(ScalarType* l, ScalarType* gz, IntType* nx, IntType* nghost, IntType* nhalo, ScalarType* hx, bool mgpu) {
src/Differentiation/TextureDifferentiationKernel.cu:  if (mgpu)
src/Differentiation/TextureDifferentiationKernel.cu:    mgpu_gradient_z<<<numBlocks_z, threadsPerBlock_z>>>(l,gz,nl,ng,halo, 1./hx[2]);
src/Differentiation/TextureDifferentiationKernel.cu:  cudaCheckKernelError();
src/Differentiation/TextureDifferentiationKernel.cu:  cudaDeviceSynchronize();
src/Differentiation/TextureDifferentiationKernel.cu:PetscErrorCode computeDivergenceY(ScalarType* l, ScalarType* gy, IntType* nx, IntType* nghost, IntType* nhalo, ScalarType* hx, bool mgpu) {
src/Differentiation/TextureDifferentiationKernel.cu:  if (mgpu)
src/Differentiation/TextureDifferentiationKernel.cu:    mgpu_gradient_y<<<numBlocks_y, threadsPerBlock_y>>>(l, gy,nl,ng,halo, 1./hx[1]);
src/Differentiation/TextureDifferentiationKernel.cu:  cudaCheckKernelError();
src/Differentiation/TextureDifferentiationKernel.cu:  cudaDeviceSynchronize();
src/Differentiation/TextureDifferentiationKernel.cu:PetscErrorCode computeDivergenceX(ScalarType* l, ScalarType* gx, IntType* nx, IntType* nghost, IntType* nhalo, ScalarType* hx, bool mgpu) {
src/Differentiation/TextureDifferentiationKernel.cu:  if (mgpu)
src/Differentiation/TextureDifferentiationKernel.cu:    mgpu_gradient_x<<<numBlocks_x, threadsPerBlock_x>>>(l, gx,nl,ng,halo, 1./hx[0]);
src/Differentiation/TextureDifferentiationKernel.cu:  cudaCheckKernelError();
src/Differentiation/TextureDifferentiationKernel.cu:  cudaDeviceSynchronize();
src/Differentiation/TextureDifferentiationKernel.cu:PetscErrorCode computeGradient(ScalarType* gx, ScalarType* gy, ScalarType* gz, const ScalarType* m, cudaTextureObject_t mtex, IntType* nx, IntType* nghost, IntType* nhalo, ScalarType* hx, bool mgpu) {
src/Differentiation/TextureDifferentiationKernel.cu:    if (mgpu) {
src/Differentiation/TextureDifferentiationKernel.cu:      ierr = cudaMemset((void*)gz, 0, count); CHKERRCUDA(ierr);
src/Differentiation/TextureDifferentiationKernel.cu:      ierr = cudaMemset((void*)gy, 0, count); CHKERRCUDA(ierr);
src/Differentiation/TextureDifferentiationKernel.cu:      ierr = cudaMemset((void*)gx, 0, count); CHKERRCUDA(ierr);
src/Differentiation/TextureDifferentiationKernel.cu:    // make input image a cudaPitchedPtr for m
src/Differentiation/TextureDifferentiationKernel.cu:    cudaPitchedPtr m_cudaPitchedPtr = make_cudaPitchedPtr((void*)(m), nx[2]*sizeof(ScalarType), nx[2], nx[1]);
src/Differentiation/TextureDifferentiationKernel.cu:    // create a cudaExtent for input resolution
src/Differentiation/TextureDifferentiationKernel.cu:    cudaExtent extent = make_cudaExtent(nx[2], nx[1], nx[0]);
src/Differentiation/TextureDifferentiationKernel.cu:    updateTextureFromVolume(m_cudaPitchedPtr, extent, mtex);
src/Differentiation/TextureDifferentiationKernel.cu:    cudaCheckKernelError();
src/Differentiation/TextureDifferentiationKernel.cu:    cudaDeviceSynchronize();
src/Differentiation/TextureDifferentiationKernel.cu:    if (mgpu)
src/Differentiation/TextureDifferentiationKernel.cu:      mgpu_gradient_z<<<numBlocks_z, threadsPerBlock_z>>>(gz,m,nl,ng,halo, 1./hx[2]);
src/Differentiation/TextureDifferentiationKernel.cu:    cudaCheckKernelError();
src/Differentiation/TextureDifferentiationKernel.cu:    if (mgpu)
src/Differentiation/TextureDifferentiationKernel.cu:      mgpu_gradient_y<<<numBlocks_y, threadsPerBlock_y>>>(gy, m,nl,ng,halo, 1./hx[1]);
src/Differentiation/TextureDifferentiationKernel.cu:    cudaCheckKernelError();
src/Differentiation/TextureDifferentiationKernel.cu:    if (mgpu)
src/Differentiation/TextureDifferentiationKernel.cu:      mgpu_gradient_x<<<numBlocks_x, threadsPerBlock_x>>>(gx, m,nl,ng,halo, 1./hx[0]);
src/Differentiation/TextureDifferentiationKernel.cu:    cudaCheckKernelError();
src/Differentiation/TextureDifferentiationKernel.cu:    cudaDeviceSynchronize();
src/Differentiation/TextureDifferentiationKernel.cu:PetscErrorCode computeLaplacian(ScalarType* ddm, const ScalarType* m, cudaTextureObject_t mtex, IntType* nx, IntType* nghost, IntType* nhalo, ScalarType* hx, ScalarType beta, bool mgpu) {
src/Differentiation/TextureDifferentiationKernel.cu:    if (mgpu)
src/Differentiation/TextureDifferentiationKernel.cu:      mgpu_d_zz<<<numBlocks_z, threadsPerBlock_z>>>(ddm, m, beta,nl,ng,halo, 1./(hx[2]*hx[2]));
src/Differentiation/TextureDifferentiationKernel.cu:    cudaCheckKernelError();
src/Differentiation/TextureDifferentiationKernel.cu:    if (mgpu)
src/Differentiation/TextureDifferentiationKernel.cu:      mgpu_d_yy<<<numBlocks_y, threadsPerBlock_y>>>(ddm, m, beta,nl,ng,halo, 1./(hx[1]*hx[1]));
src/Differentiation/TextureDifferentiationKernel.cu:    cudaCheckKernelError();
src/Differentiation/TextureDifferentiationKernel.cu:    if (mgpu)
src/Differentiation/TextureDifferentiationKernel.cu:      mgpu_d_xx<<<numBlocks_x, threadsPerBlock_x>>>(ddm, m, beta,nl,ng,halo, 1./(hx[0]*hx[0]));
src/Differentiation/TextureDifferentiationKernel.cu:    cudaCheckKernelError();
src/Differentiation/TextureDifferentiationKernel.cu:    cudaDeviceSynchronize();
src/GhostPlan.cpp:  cudaMalloc((void**)&padded_data, sizeof(ScalarType)*( nl + 2*this->g_size*isize[2]*isize[0] ));
src/GhostPlan.cpp:  cudaMalloc((void**)&RS, rs_buf_size * sizeof(ScalarType));
src/GhostPlan.cpp:  cudaMalloc((void**)&GL, rs_buf_size * sizeof(ScalarType));
src/GhostPlan.cpp:  cudaMalloc((void**)&LS, ls_buf_size * sizeof(ScalarType));
src/GhostPlan.cpp:  cudaMalloc((void**)&GR, ls_buf_size * sizeof(ScalarType));
src/GhostPlan.cpp:  cudaMalloc((void**)&GB, sizeof(ScalarType)*ts_buf_size);
src/GhostPlan.cpp:  cudaMalloc((void**)&GT, sizeof(ScalarType)*bs_buf_size);
src/GhostPlan.cpp:  stream = new cudaStream_t[num_streams];
src/GhostPlan.cpp:      cudaStreamCreate(&stream[i]);
src/GhostPlan.cpp:    cudaFree(padded_data);
src/GhostPlan.cpp:    cudaFree(RS);
src/GhostPlan.cpp:    cudaFree(GL);
src/GhostPlan.cpp:    cudaFree(LS);
src/GhostPlan.cpp:    cudaFree(GR);
src/GhostPlan.cpp:    cudaFree(GB);
src/GhostPlan.cpp:    cudaFree(GT);
src/GhostPlan.cpp:    cudaStreamDestroy(stream[i]);
src/GhostPlan.cpp:    cudaMemcpyAsync((void*)&RS[x * g_size * isize[2]], 
src/GhostPlan.cpp:                    g_size * isize[2] * sizeof(ScalarType), cudaMemcpyDeviceToDevice, stream[x%num_streams]);
src/GhostPlan.cpp:    cudaStreamSynchronize(stream[i]);
src/GhostPlan.cpp:		cudaMemcpyAsync(&LS[x * g_size * isize[2]], 
src/GhostPlan.cpp:		                g_size * isize[2] * sizeof(ScalarType), cudaMemcpyDeviceToDevice, stream[x%num_streams]);
src/GhostPlan.cpp:    cudaStreamSynchronize (stream[i]);
src/GhostPlan.cpp:		cudaMemcpyAsync(&padded_data[i * isize[2] * (isize[1] + 2 * g_size)],
src/GhostPlan.cpp:				    g_size * isize[2] * sizeof(ScalarType), cudaMemcpyDeviceToDevice, stream[k%num_streams]);
src/GhostPlan.cpp:		cudaMemcpyAsync(&padded_data[i * isize[2] * (isize[1] + 2 * g_size) + g_size * isize[2]], 
src/GhostPlan.cpp:				    isize[1] * isize[2] * sizeof(ScalarType), cudaMemcpyDeviceToDevice, stream[k%num_streams]);
src/GhostPlan.cpp:		cudaMemcpyAsync(&padded_data[i * isize[2] * (isize[1] + 2 * g_size) + g_size * isize[2] + isize[2] * isize[1]],
src/GhostPlan.cpp:				    g_size * isize[2] * sizeof(ScalarType), cudaMemcpyDeviceToDevice, stream[k%num_streams]);
src/GhostPlan.cpp:    cudaStreamSynchronize (stream[i]);
src/GhostPlan.cpp:	cudaMemcpyAsync(&ghost_data[0], 
src/GhostPlan.cpp:	            g_size * isize[2] * (isize[1] + 2 * g_size) * sizeof(ScalarType), cudaMemcpyDeviceToDevice);
src/GhostPlan.cpp:	cudaMemcpyAsync(&ghost_data[g_size * isize[2] * (isize[1] + 2 * g_size)], 
src/GhostPlan.cpp:	            isize[0] * isize[2] * (isize[1] + 2 * g_size) * sizeof(ScalarType), cudaMemcpyDeviceToDevice);
src/GhostPlan.cpp:	cudaMemcpyAsync(&ghost_data[g_size * isize[2] * (isize[1] + 2 * g_size)+ isize[0] * isize[2] * (isize[1] + 2 * g_size)], 
src/GhostPlan.cpp:	            g_size * isize[2] * (isize[1] + 2 * g_size) * sizeof(ScalarType), cudaMemcpyDeviceToDevice);
src/GhostPlan.cpp:		cudaMemcpy(ghost_data, data, sizeof(ScalarType)*this->m_Opt->m_Domain.nl, cudaMemcpyDeviceToDevice);
src/GhostPlan.cpp:  // Do cudamemcpys while waiting for MPI to finish
src/GhostPlan.cpp:	cudaMemcpy(&ghost_data[g_size * isize[2] * isize[1]], 
src/GhostPlan.cpp:	            isize[0] * isize[2] * isize[1] * sizeof(ScalarType), cudaMemcpyDeviceToDevice);
src/GhostPlan.cpp:	cudaMemcpy(&ghost_data[0], 
src/GhostPlan.cpp:	            g_size * isize[2] * isize[1] * sizeof(ScalarType), cudaMemcpyDeviceToDevice);
src/GhostPlan.cpp:	cudaMemcpy(&ghost_data[(isize[0] + g_size) * isize[2] * isize[1]], 
src/GhostPlan.cpp:	            g_size * isize[2] * isize[1] * sizeof(ScalarType), cudaMemcpyDeviceToDevice);
src/GhostPlan.cpp:		cudaMemcpy(ghost_data, data, sizeof(ScalarType)*this->m_Opt->m_Domain.nl, cudaMemcpyDeviceToDevice);
src/PreconditionerKernel.cu:#include "cuda_helper.hpp"
src/PreconditionerKernel.cu:using KernelUtils::KernelCallGPU;
src/PreconditionerKernel.cu:using KernelUtils::ReductionKernelCallGPU;
src/PreconditionerKernel.cu:  ierr = KernelCallGPU<H0Kernel2>(nl, 
src/PreconditionerKernel.cu:  ierr = ReductionKernelCallGPU<H0Kernel2>(res, pWS, nl, 
src/PreconditionerKernel.cu:  ierr = ReductionKernelCallGPU<H0Kernel2>(res, pWS, nl, 
src/PreconditionerKernel.cu:  ierr = KernelCallGPU<H0Kernel>(nl, 
src/PreconditionerKernel.cu:  ierr = ReductionKernelCallGPU<H0Kernel>(res, pWS, nl, 
src/PreconditionerKernel.cu:  ierr = ReductionKernelCallGPU<H0Kernel>(res, pWS, nl, 
src/PreconditionerKernel.cu:  ierr = ReductionKernelCallGPU<H0KernelCG>(res, pWS, nl, 
src/PreconditionerKernel.cu:  ierr = KernelCallGPU<H0KernelCG>(nl, 
src/PreconditionerKernel.cu:  ierr = ReductionKernelCallGPU<CFLKernel>(res, nl, pV[0], h, dt); CHKERRQ(ierr);
src/PreconditionerKernel.cu:  ierr = ReductionKernelCallGPU<NormKernel>(res, nl, pGmt[0]); CHKERRQ(ierr);

```
