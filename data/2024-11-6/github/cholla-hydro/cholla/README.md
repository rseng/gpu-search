# https://github.com/cholla-hydro/cholla

```console
.clang-tidy:        -clang-diagnostic-unknown-cuda-version,
.clang-tidy:  bugprone-reserved-identifier.AllowedIdentifiers: '__cudaSafeCall;__cudaCheckError;__shfl_down;__CHOLLA_PRETTY_FUNC__'
builds/setup.birch.cce.sh:module load rocm
builds/setup.birch.cce.sh:#-- GPU-aware MPI
builds/setup.birch.cce.sh:export MPICH_GPU_SUPPORT_ENABLED=1
builds/setup.birch.cce.sh:export MPI_GPU="-DMPI_GPU"
builds/make.host.github:GPUFLAGS_DEBUG    = -g -G -cudart shared -O0 -std=c++17
builds/make.host.github:GPUFLAGS_OPTIMIZE = -g -O3 -std=c++17
builds/make.host.github:	GPUFLAGS_DEBUG = -g -O0 -std=c++17
builds/make.host.github:CUDA_ROOT       := $(CUDA_ROOT)
builds/make.host.github:#-- MPI calls accept GPU buffers (requires GPU-aware MPI)
builds/make.host.github:# MPI_GPU = -DMPI_GPU
builds/make.host.github:	GPUFLAGS_DEBUG    += -fPIE
builds/make.host.github:	GPUFLAGS_OPTIMIZE += -fPIE
builds/make.type.cooling:#-- Default hydro + cooling_gpu
builds/make.type.cooling:MPI_GPU   ?=
builds/make.type.cooling:# Apply the cooling in the GPU from precomputed tables
builds/make.type.cooling:DFLAGS    += -DCOOLING_GPU
builds/make.type.cooling:#Select if the Hydro Conserved data will reside in the GPU
builds/make.type.cooling:#and the MPI transfers are done from the GPU
builds/make.type.cooling:#If not specified, MPI_GPU is off by default
builds/make.type.cooling:DFLAGS    += $(MPI_GPU)
builds/make.type.disk:MPI_GPU = -DMPI_GPU
builds/make.type.disk:DFLAGS += -DPARTICLES_GPU
builds/make.type.disk:DFLAGS += -DGRAVITY_GPU
builds/make.type.disk:DFLAGS    += -DCOOLING_GPU
builds/make.type.disk:DFLAGS    += -DHYDRO_GPU
builds/make.type.disk:DFLAGS    += $(MPI_GPU)
builds/setup.spock.cce.sh:module load rocm/4.5.0
builds/setup.spock.cce.sh:#-- GPU-aware MPI
builds/setup.spock.cce.sh:export MPICH_GPU_SUPPORT_ENABLED=1
builds/setup.spock.cce.sh:export MPI_GPU="-DMPI_GPU"
builds/make.host.c3po:GPUFLAGS_DEBUG    = -g -G -cudart shared -O0 -std=c++17 -ccbin=mpicxx -Xcompiler -rdynamic
builds/make.host.c3po:GPUFLAGS_OPTIMIZE = -g -O3 -std=c++17 -ccbin=mpicxx
builds/make.host.c3po:CUDA_ROOT       := /usr/local/cuda-11.4
builds/make.host.c3po:#-- MPI calls accept GPU buffers (requires GPU-aware MPI)
builds/make.host.c3po:# MPI_GPU = -DMPI_GPU
builds/make.host.poplar.aomp:GPUFLAGS          = --offload-arch=gfx906,gfx908
builds/make.host.spock:GPUFLAGS          = --offload-arch=gfx908
builds/make.host.spock:#-- MPI calls accept GPU buffers (requires GPU-aware MPI)
builds/make.host.spock:MPI_GPU = -DMPI_GPU
builds/setup.poplar.cce+hip.sh:module load rocm/4.0.0
builds/setup.poplar.cce+hip.sh:module load cray-mpich/rocm4.0
builds/setup.poplar.cce+hip.sh:#-- Enable  GPU-MPI, requires OpenMP offload
builds/setup.poplar.cce+hip.sh:#export MPI_GPU="-DMPI_GPU"
builds/make.type.mhd:MPI_GPU   ?=
builds/make.type.mhd:# Apply the cooling in the GPU from precomputed tables
builds/make.type.mhd:# DFLAGS    += -DCOOLING_GPU
builds/make.type.mhd:#Select if the Hydro Conserved data will reside in the GPU
builds/make.type.mhd:#and the MPI transfers are done from the GPU
builds/make.type.mhd:#If not specified, MPI_GPU is off by default
builds/make.type.mhd:DFLAGS    += $(MPI_GPU)
builds/make.type.mhd:# Disable CUDA error checking
builds/make.type.mhd:# DFLAGS += -DDISABLE_GPU_ERROR_CHECKING
builds/setup.summit.xl.sh:module load xl cuda fftw hdf5 python googletest/1.11.0
builds/setup.lux.sh:module load hdf5/1.10.6 cuda10.2/10.2 openmpi/4.0.1
builds/make.host.poplar.cce+hip:GPUFLAGS          = --offload-arch=gfx906,gfx908
builds/make.host.poplar:GPUFLAGS          = --offload-arch=gfx906,gfx908
builds/make.type.gravity:#Keep the Gravity arrays resident in the GPU for high efficiency
builds/make.type.gravity:#NOTE: If using PARTICLES and GRAVITY_GPU is turned on, then PARTICLES_GPU has to be turned on 
builds/make.type.gravity:#NOTE: If using GRAVITY and MPI_GPU is turned on, then GRAVITY_GPU has to be turned on 
builds/make.type.gravity:DFLAGS += -DGRAVITY_GPU
builds/make.type.gravity:#Select if Paris will do GPU MPI transfers 
builds/make.type.gravity:#If not specified, Paris will do GPU MPI transfers by default
builds/make.type.gravity:DFLAGS += $(PARIS_MPI_GPU)
builds/make.inc.template:# Allocate GPU memory every timestep
builds/make.inc.template:#DFLAGS += -DDYNAMIC_GPU_ALLOC
builds/make.inc.template:#DFLAGS += -DCOOLING_GPU
builds/make.inc.template:#DFLAGS += -DPARTICLES_GPU
builds/make.host.frontier:#GPUCXX           ?= CC -x hip
builds/make.host.frontier:GPUCXX           ?= hipcc
builds/make.host.frontier:GPUFLAGS_OPTIMIZE = -std=c++17 --offload-arch=gfx90a -Wall -Wno-unused-result
builds/make.host.frontier:GPUFLAGS_DEBUG    = -g -O0 -std=c++17 --offload-arch=gfx90a -Wall -Wno-unused-result
builds/make.host.frontier:HIPCONFIG	  = -I$(ROCM_PATH)/include $(shell hipconfig -C) # workaround for Rocm 5.2 warnings
builds/make.host.frontier:#-- Use GPU-aware MPI
builds/make.host.frontier:MPI_GPU           = -DMPI_GPU
builds/make.type.rot_proj:# Apply cooling on the GPU from precomputed tables
builds/make.type.rot_proj:#DFLAGS    += -DCOOLING_GPU
builds/setup.github.gcc.sh:# export MPI_GPU="-DMPI_GPU"
builds/setup.poplar.aomp.sh:module load ompi/4.0.4-rocm-3.9 hdf5
builds/make.host.shamrock:CUDA_ROOT    = /usr/local/cuda-10.1/targets/x86_64-linux
builds/make.host.shamrock:CUDA_LIB     = -L$(CUDA_ROOT)/lib -lcudart -lcufft
builds/make.host.shamrock:#Paris does not do GPU_MPI transfers
builds/make.host.shamrock:PARIS_MPI_GPU = -DPARIS_NO_GPU_MPI
builds/setup.summit.gcc.sh:#module load gcc/10.2.0 cuda/11.4.0 fftw hdf5 python
builds/setup.summit.gcc.sh:module load gcc cuda fftw hdf5 python googletest/1.11.0
builds/make.host.lux:GPUFLAGS         = -std=c++17
builds/make.host.lux:CUDA_ROOT    = /cm/shared/apps/cuda10.2/toolkit/current
builds/make.host.lux:#Paris does not do GPU_MPI transfers
builds/make.host.lux:PARIS_MPI_GPU = -DPARIS_NO_GPU_MPI
builds/make.host.summit:GPUFLAGS_DEBUG    = -g -O0 -std=c++17 -ccbin=mpicxx -G -cudart shared
builds/make.host.summit:GPUFLAGS_OPTIMIZE = -g -O3 -std=c++17 -ccbin=mpicxx
builds/make.host.summit:CUDA_ROOT       = ${OLCF_CUDA_ROOT}
builds/make.host.summit:#-- MPI calls accept GPU buffers (requires GPU-aware MPI)
builds/make.host.summit:MPI_GPU = -DMPI_GPU
builds/prereq.sh:      if ! module is-loaded gcc hdf5 cuda fftw; then
builds/prereq.sh:        echo "modulefile required: gcc, hdf5, fftw, and cuda"
builds/prereq.sh:        echo "do: 'module load gcc hdf5 cuda fftw'"
builds/prereq.sh:          && ( module list 2>&1 | grep -q rocm \
builds/prereq.sh:          || module list 2>&1 | grep -q cuda )
builds/prereq.sh:       if ! module is-loaded gcc hdf5 cuda openmpi ; then
builds/prereq.sh:         echo "echo: requires loading modules: cuda, gcc, openmpi and hdf5"
builds/make.type.hydro:# Apply cooling on the GPU from precomputed tables
builds/make.type.hydro:#DFLAGS    += -DCOOLING_GPU
builds/make.type.cosmology:# Solve the Primordial Chemical Network (H+He) on the GPU (Includes Radiative Cooling, Photoheating and Photoionization)
builds/make.type.cosmology:#DFLAGS += -DCHEMISTRY_GPU -DOUTPUT_TEMPERATURE -DOUTPUT_CHEMISTRY 
builds/setup.crc.gcc.sh:module load python/anaconda3-2020.11 gcc/10.1.0 cuda/11.1.0 openmpi/4.0.5 hdf5/1.12.0 googletest/1.11.0
builds/setup.crc.gcc.sh:# export MPI_GPU="-DMPI_GPU"
builds/make.type.dust:MPI_GPU   ?=
builds/make.type.dust:# Apply the cooling in the GPU from precomputed tables
builds/make.type.dust:DFLAGS    += -DCOOLING_GPU
builds/make.type.dust:#Select if the Hydro Conserved data will reside in the GPU
builds/make.type.dust:#and the MPI transfers are done from the GPU
builds/make.type.dust:#If not specified, MPI_GPU is off by default
builds/make.type.dust:DFLAGS    += $(MPI_GPU)
builds/make.type.basic_scalar:# Apply cooling on the GPU from precomputed tables
builds/make.type.basic_scalar:#DFLAGS    += -DCOOLING_GPU
builds/make.host.tornado:CUDA_ROOT    = /usr/local/cuda-10.0
builds/setup.c3po.gcc.sh:# export MPI_GPU="-DMPI_GPU"
builds/make.type.FOM:DFLAGS += -DPARTICLES_GPU
builds/run_tests.sh:      export CHOLLA_LAUNCH_COMMAND=("jsrun --smpiargs=\"-gpu\" --cpu_per_rs 1 --tasks_per_rs 1 --gpu_per_rs 1 --nrs")
builds/make.type.cloudy:#-- Default hydro + cooling_gpu
builds/make.type.cloudy:MPI_GPU   ?=
builds/make.type.cloudy:# Apply the cooling in the GPU from precomputed tables
builds/make.type.cloudy:DFLAGS    += -DCOOLING_GPU
builds/make.type.cloudy:#Select if the Hydro Conserved data will reside in the GPU
builds/make.type.cloudy:#and the MPI transfers are done from the GPU
builds/make.type.cloudy:#If not specified, MPI_GPU is off by default
builds/make.type.cloudy:DFLAGS    += $(MPI_GPU)
builds/make.host.crc:GPUFLAGS_OPTIMIZE = -g -O3 -std=c++17
builds/make.host.crc:CUDA_ARCH       = sm_70
builds/make.host.crc:# CUDA_ROOT       = /ihome/crc/install/power9/cuda/11.1.0
builds/make.host.crc:#-- MPI calls accept GPU buffers (requires GPU-aware MPI)
builds/make.host.crc:MPI_GPU = -DMPI_GPU
builds/make.type.static_grav:# Apply cooling on the GPU from precomputed tables
builds/make.type.static_grav:#DFLAGS    += -DCOOLING_GPU
builds/setup.frontier.cce.sh:module load rocm
builds/setup.frontier.cce.sh:#-- GPU-aware MPI
builds/setup.frontier.cce.sh:export MPICH_GPU_SUPPORT_ENABLED=1
builds/make.type.particles:#Solve the particles in the GPU or CPU
builds/make.type.particles:#NOTE: If using PARTICLES and MPI_GPU is turned on, then PARTICLES_GPU has to be turned on
builds/make.type.particles:DFLAGS += -DPARTICLES_GPU
docs/doxygen/Doxyfile:PROJECT_BRIEF          = "Cholla - Massively parallel hydro on GPUs"
docs/sphinx/source/gettingstarted.rst:- An NVIDIA graphics card
docs/sphinx/source/gettingstarted.rst:- The NVIDIA cuda compiler, nvcc (the CUDA toolkit is available `here <https://developer.nvidia.com/accelerated-computing-toolkit>`_.
docs/sphinx/source/gettingstarted.rst:Note: It is important that the code be compiled for the correct GPU architecture. This is specified in the makefile via the -arch flag. The GPU architecture can be found by running the "Device_Query" sample program from the NVIDIA Cuda toolkit (located in the "Samples/Utilities" folder wherever Cuda was installed). Several common architectures are -arch=sm_35 for Tesla K20's, -arch=sm_60 for Tesla P100's, or -arch=sm_70 for Tesla V100's.
docs/sphinx/source/gettingstarted.rst:To run cholla on a single GPU, you execute the binary and provide it with an input parameter file. For example, to run a 1D Sod Shock tube test, within the top-level directory you would type:
docs/sphinx/source/gettingstarted.rst:Cholla can also be run using multiple GPUs when it is compiled using the Message Passing Interface (MPI) protocol. To run in parallel mode requires an mpi compiler, such as openmpi. Once the mpi compiler is installed or loaded, uncomment the relevant line in the makefile:
docs/sphinx/source/gettingstarted.rst:and compile the code. (If you have already compiled the code in serial mode, be sure to clean up first: ``make clean``.) Once the code is compiled with mpi, you can run it using as many processes as you have available GPUs - Cholla assumes there is one GPU per MPI process. For example, if you have 4 GPUs, you could run a 3D sound wave test via:
docs/sphinx/source/gettingstarted.rst:The code will automatically divide the simulation domain amongst the GPUs. If you are running on a cluster, you may have to specify additional information about the number of GPUs per node in the batch submission script (e.g. PBS, slurm, LSF).
docs/sphinx/source/index.rst:Cholla is a static-mesh, GPU-native hydrodynamics simulation code that efficiently runs high-resolution simulations on massively-parallel computers. The code is written in a combination of C++ and Cuda C and requires at least one NVIDIA GPU to run. Cholla was designed for astrophysics simulations, and the current release includes the following physics:
Makefile:# CUDA_ARCH defaults to sm_70 if not set in make.host
Makefile:CUDA_ARCH ?= sm_70
Makefile:DIRS     := src src/analysis src/chemistry_gpu src/cooling src/cooling_grackle src/cosmology \
Makefile:GPUFILES := $(foreach DIR,$(DIRS),$(wildcard $(DIR)/*.cu))
Makefile:              $(subst .cu,.o,$(GPUFILES))
Makefile:  GPUFLAGS += $(TEST_FLAGS)
Makefile:  GPUFILES := $(filter-out src/system_tests/% %_tests.cu,$(GPUFILES))
Makefile:            $(subst .cu,.o,$(GPUFILES))
Makefile:GPUFLAGS_OPTIMIZE ?= -g -O3 -std=c++17
Makefile:  GPUFLAGS_DEBUG    ?= -g -O0 -std=c++17
Makefile:  GPUFLAGS_DEBUG    ?= -g -G -cudart shared -O0 -std=c++17 -ccbin=mpicxx
Makefile:GPUFLAGS          += $(GPUFLAGS_$(BUILD))
Makefile:GPUFLAGS += $(DFLAGS) -Isrc
Makefile:    CXXFLAGS += -I$(ROCM_PATH)/include/hipfft -I$(ROCM_PATH)/hipfft/include
Makefile:    GPUFLAGS += -I$(ROCM_PATH)/include/hipfft -I$(ROCM_PATH)/hipfft/include
Makefile:    LIBS += -L$(ROCM_PATH)/hipfft/lib -lhipfft
Makefile:	CXXFLAGS += -I$(ROCM_PATH)/include/hiprand -I$(ROCM_PATH)/hiprand/include
Makefile:	GPUFLAGS += -I$(ROCM_PATH)/include/hiprand -I$(ROCM_PATH)/hiprand/include
Makefile:  GPUFLAGS += -I$(HDF5_ROOT)/include
Makefile:  GPUFLAGS += -I$(MPI_ROOT)/include
Makefile:  GPUFLAGS += -I$(FFTW_ROOT)/include
Makefile:  GPUCXX    ?= hipcc
Makefile:  #GPUFLAGS  += -Wall
Makefile:  LDFLAGS   := $(CXXFLAGS) -L$(ROCM_PATH)/lib
Makefile:  CUDA_INC  ?= -I$(CUDA_ROOT)/include
Makefile:  CUDA_LIB  ?= -L$(CUDA_ROOT)/lib64 -lcudart
Makefile:  CXXFLAGS  += $(CUDA_INC)
Makefile:  GPUCXX    ?= nvcc
Makefile:  GPUFLAGS  += --expt-extended-lambda -arch $(CUDA_ARCH) -fmad=false
Makefile:  GPUFLAGS  += $(CUDA_INC)
Makefile:  LIBS      += $(CUDA_LIB)
Makefile:  GPUFLAGS += -I$(GRACKLE_ROOT)/include
Makefile:ifeq ($(findstring -DCHEMISTRY_GPU,$(DFLAGS)),-DCHEMISTRY_GPU)
Makefile:GPUFLAGS_CLANG_TIDY := $(subst -I/, -isystem /,$(GPUFLAGS))
Makefile:GPUFLAGS_CLANG_TIDY := $(filter-out -ccbin=mpicxx -fmad=false --expt-extended-lambda,$(GPUFLAGS_CLANG_TIDY))
Makefile:GPUFLAGS_CLANG_TIDY += --cuda-host-only --cuda-path=$(CUDA_ROOT) -isystem /clang/includes
Makefile:GPUFILES_TIDY := $(GPUFILES)
Makefile:  GPUFILES_TIDY := $(filter $(TIDY_FILES), $(GPUFILES_TIDY))
Makefile:	$(GPUCXX) $(GPUFLAGS) -c $< -o $@
Makefile:	(time clang-tidy $(CLANG_TIDY_ARGS) $(GPUFILES_TIDY) -- $(DFLAGS) $(GPUFLAGS_CLANG_TIDY) $(LIBS_CLANG_TIDY)) > tidy_results_gpu_$(TYPE).log 2>&1 & \
Makefile:	@echo -e "\nResults from clang-tidy are available in the 'tidy_results_cpp_$(TYPE).log' and 'tidy_results_gpu_$(TYPE).log' files."
README.md:A 3D GPU-based hydrodynamics code (Schneider & Robertson, ApJS, 2015).
README.md:*Cholla* is designed to be run using (AMD or NVIDIA) GPUs, and can be run in serial mode using one GPU
README.md:or with MPI for multiple GPUs.
README.md:double precision, output format, the reconstruction method, Riemann solver, integrator, and cooling. Examples of configurations that require edits to a make.host file include library paths, compiler options, and gpu-enabled MPI. The entire code must be recompiled any time you change the configuration. For more information on the various options, see the "[Makefile](https://github.com/cholla-hydro/cholla/wiki/Makefile-Parameters)" page of the wiki.
README.md:Each process will be assigned a GPU. *Cholla* cannot be run with more processes than available GPUs,
README.md:so MPI mode is most useful on a cluster (or for testing parallel behavior with a single process). Note that more recent AMD devices have 2 GPUs (or GCDs) per accelerator device, so you can run with 2x the number of MPI tasks.
tools/cholla-nv-compute-sanitizer.sh:# Utility script for running the NVIDIA Compute Sanitizer.
tools/cholla-nv-compute-sanitizer.sh:# See the NVIDIA docs for more detail:
tools/cholla-nv-compute-sanitizer.sh:# https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
tools/analyze_tidy_checks.py:    with open(chollaPath / "tidy_results_gpu.log", "r") as file:
tools/analyze_tidy_checks.py:        gpuData = file.read()
tools/analyze_tidy_checks.py:    return cppData + gpuData
tools/clang-tidy_runner.sh:# and a thread for the GPU code
docker/rocm/Dockerfile:FROM rocm/dev-ubuntu-20.04:5.2.3
docker/rocm/Dockerfile:# Needed to trick ROCm into thinking there's a GPU
docker/rocm/Dockerfile:ENV HIPCONFIG=/opt/rocm-5.2.3
docker/rocm/Dockerfile:ENV ROCM_PATH=/opt/rocm-5.2.3
docker/cuda/Dockerfile:FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
docker/cuda/Dockerfile:ENV CUDA_ROOT=/usr/local/cuda-11/
Jenkinsfile:                                cat tidy_results_gpu_${CHOLLA_MAKE_TYPE}.log
examples/scripts/srun-paris-cufft.sh:module load gcc hdf5 cuda
examples/scripts/srun-paris-cufft.sh:jsrun --smpiargs="-gpu" -n1 -a1 -c16 -g1 ../../bin/cholla.paris.cufft ../../examples/scripts/parameter_file.txt |& tee tee
examples/scripts/srun-paris-sor.sh:module load gcc hdf5 cuda fftw
examples/scripts/srun-paris-sor.sh:jsrun --smpiargs="-gpu" -n4 -a1 -c4 -g1 ../../bin/cholla.paris.sor ../../examples/scripts/sphere.txt |& tee tee
examples/scripts/nrun-paris-pfft.sh:OUTDIR="run/out.paris.pfft-cuda.$(date +%m%d.%H%M%S)"
examples/scripts/nrun-paris-pfft.sh:export MV2_USE_CUDA=1
examples/scripts/nrun-paris-pfft.sh:srun -n4 -c$OMP_NUM_THREADS -N1 --exclusive -p v100 ../../bin/cholla.paris.pfft-cuda ../../examples/scripts/parameter_file.txt |& tee tee
examples/scripts/arun-paris-sor.sh:module load hdf5 gcc/8.1.0 rocm
examples/scripts/arun-paris-sor.sh:export MV2_USE_CUDA=0
examples/scripts/arun-paris-sor.sh:export MV2_SUPPRESS_CUDA_USAGE_WARNING=1
examples/scripts/arun-paris-sphere.sh:module load hdf5 gcc/8.1.0 rocm
examples/scripts/arun-paris-sphere.sh:export MV2_USE_CUDA=0
examples/scripts/arun-paris-sphere.sh:export MV2_SUPPRESS_CUDA_USAGE_WARNING=1
examples/scripts/nrun-paris-cufft.sh:export MV2_USE_CUDA=1
examples/scripts/nrun-paris.sh:OUTDIR="run/out.paris-cuda.$(date +%m%d.%H%M%S)"
examples/scripts/nrun-paris.sh:export MV2_USE_CUDA=1
examples/scripts/nrun-paris.sh:srun -n4 -c$OMP_NUM_THREADS -N1 --exclusive -p v100 ../../bin/cholla.paris-cuda ../../examples/scripts/parameter_file.txt |& tee tee
examples/scripts/srun-paris-pfft.sh:module load gcc hdf5 cuda fftw
examples/scripts/srun-paris-pfft.sh:jsrun --smpiargs="-gpu" -n4 -a1 -c4 -g1 ../../bin/cholla.paris.pfft ../../examples/scripts/parameter_file.txt |& tee tee
examples/scripts/arun-paris-hipfft.sh:module load rocm
examples/scripts/arun-paris-hipfft.sh:export MV2_USE_CUDA=0
examples/scripts/arun-paris-hipfft.sh:export MV2_SUPPRESS_CUDA_USAGE_WARNING=1
examples/scripts/arun-paris-pfft.sh:module load rocm
examples/scripts/arun-paris-pfft.sh:export MV2_USE_CUDA=0
examples/scripts/arun-paris-pfft.sh:export MV2_SUPPRESS_CUDA_USAGE_WARNING=1
examples/scripts/arun-sor.sh:module load hdf5 gcc/8.1.0 rocm
examples/scripts/arun-sor.sh:export MV2_USE_CUDA=0
examples/scripts/arun-sor.sh:export MV2_SUPPRESS_CUDA_USAGE_WARNING=1
examples/scripts/arun-paris.sh:module load rocm
examples/scripts/arun-paris.sh:export MV2_USE_CUDA=0
examples/scripts/arun-paris.sh:export MV2_SUPPRESS_CUDA_USAGE_WARNING=1
examples/scripts/nrun-paris-sor.sh:SUFFIX=paris.sor.cuda
examples/scripts/nrun-paris-sor.sh:export MV2_USE_CUDA=1
examples/scripts/arun-hydro.sh:module load rocm
examples/scripts/arun-hydro.sh:export MV2_USE_CUDA=0
examples/scripts/arun-hydro.sh:export MV2_SUPPRESS_CUDA_USAGE_WARNING=1
examples/scripts/srun-paris.sh:module load gcc hdf5 cuda fftw
examples/scripts/srun-paris.sh:jsrun --smpiargs="-gpu" -n4 -a1 -c4 -g1 ../../bin/cholla.paris ../../examples/scripts/parameter_file.txt |& tee tee
examples/scripts/nrun-sor.sh:OUTDIR="run/out.sor-cuda.$(date +%m%d.%H%M%S)"
examples/scripts/nrun-sor.sh:export MV2_USE_CUDA=1
examples/scripts/nrun-sor.sh:srun -n1 -c$OMP_NUM_THREADS -N1 --exclusive -p v100 ../../bin/cholla.sor-cuda ../../examples/scripts/sphere.txt |& tee tee
src/cosmology/cosmology_functions_gpu.cu:  #include "../cosmology/cosmology_functions_gpu.h"
src/cosmology/cosmology_functions_gpu.cu:  // NOTE If CHEMISTRY_GPU I need to add the conversion for the chemical species
src/cosmology/cosmology_functions_gpu.cu:void Grid3D::Change_GAS_Frame_System_GPU(bool forward)
src/cosmology/cosmology_functions_gpu.cu:  // set values for GPU kernels
src/cosmology/cosmology_functions.cpp:  Change_GAS_Frame_System_GPU(forward);
src/cosmology/cosmology_functions.cpp:  // NOTE:Not implemented for PARTICLES_GPU, doesn't matter as long as
src/cosmology/cosmology_functions.cpp:  #ifdef CHEMISTRY_GPU
src/cosmology/cosmology_functions_gpu.h:  #include "../utils/gpu.hpp"
src/chemistry_gpu/chemistry_functions_gpu.cu:#ifdef CHEMISTRY_GPU
src/chemistry_gpu/chemistry_functions_gpu.cu:  #include "../global/global_cuda.h"
src/chemistry_gpu/chemistry_functions_gpu.cu:  #include "../hydro/hydro_cuda.h"
src/chemistry_gpu/chemistry_functions_gpu.cu:  #include "chemistry_gpu.h"
src/chemistry_gpu/chemistry_functions_gpu.cu:void Chem_GPU::Allocate_Array_GPU_float(float **array_dev, int size)
src/chemistry_gpu/chemistry_functions_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)array_dev, size * sizeof(float)));
src/chemistry_gpu/chemistry_functions_gpu.cu:void Chem_GPU::Copy_Float_Array_to_Device(int size, float *array_h, float *array_d)
src/chemistry_gpu/chemistry_functions_gpu.cu:  GPU_Error_Check(cudaMemcpy(array_d, array_h, size * sizeof(float), cudaMemcpyHostToDevice));
src/chemistry_gpu/chemistry_functions_gpu.cu:  cudaDeviceSynchronize();
src/chemistry_gpu/chemistry_functions_gpu.cu:void Chem_GPU::Free_Array_GPU_float(float *array_dev) { GPU_Error_Check(cudaFree(array_dev)); }
src/chemistry_gpu/chemistry_functions_gpu.cu:void Chem_GPU::Allocate_Array_GPU_Real(Real **array_dev, int size)
src/chemistry_gpu/chemistry_functions_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)array_dev, size * sizeof(Real)));
src/chemistry_gpu/chemistry_functions_gpu.cu:  GPU_Error_Check();
src/chemistry_gpu/chemistry_functions_gpu.cu:void Chem_GPU::Copy_Real_Array_to_Device(int size, Real *array_h, Real *array_d)
src/chemistry_gpu/chemistry_functions_gpu.cu:  GPU_Error_Check(cudaMemcpy(array_d, array_h, size * sizeof(Real), cudaMemcpyHostToDevice));
src/chemistry_gpu/chemistry_functions_gpu.cu:  cudaDeviceSynchronize();
src/chemistry_gpu/chemistry_functions_gpu.cu:void Chem_GPU::Free_Array_GPU_Real(Real *array_dev)
src/chemistry_gpu/chemistry_functions_gpu.cu:  GPU_Error_Check(cudaFree(array_dev));
src/chemistry_gpu/chemistry_functions_gpu.cu:  GPU_Error_Check();
src/chemistry_gpu/chemistry_functions_gpu.cu:        "##### Chem_GPU: dt_hydro: %e   t_chem: %e   dens: %e   temp: %e  GE: "
src/chemistry_gpu/chemistry_functions_gpu.cu:    if (print) printf("Chem_GPU: N Iter:  %d\n", n_iter);
src/chemistry_gpu/chemistry_functions_gpu.cu:  cudaEvent_t start, stop;
src/chemistry_gpu/chemistry_functions_gpu.cu:  cudaEventCreate(&start);
src/chemistry_gpu/chemistry_functions_gpu.cu:  cudaEventCreate(&stop);
src/chemistry_gpu/chemistry_functions_gpu.cu:  cudaEventRecord(start, 0);
src/chemistry_gpu/chemistry_functions_gpu.cu:  GPU_Error_Check();
src/chemistry_gpu/chemistry_functions_gpu.cu:  cudaEventRecord(stop, 0);
src/chemistry_gpu/chemistry_functions_gpu.cu:  cudaEventSynchronize(stop);
src/chemistry_gpu/chemistry_functions_gpu.cu:  cudaEventElapsedTime(&time, start, stop);
src/chemistry_gpu/chemistry_gpu.h:#ifndef CHEMISTRY_GPU_H
src/chemistry_gpu/chemistry_gpu.h:#define CHEMISTRY_GPU_H
src/chemistry_gpu/chemistry_gpu.h:#ifdef CHEMISTRY_GPU
src/chemistry_gpu/chemistry_gpu.h:class Chem_GPU
src/chemistry_gpu/chemistry_gpu.h:  void Allocate_Array_GPU_Real(Real **array_dev, int size);
src/chemistry_gpu/chemistry_gpu.h:  void Free_Array_GPU_Real(Real *array_dev);
src/chemistry_gpu/chemistry_gpu.h:  void Allocate_Array_GPU_float(float **array_dev, int size);
src/chemistry_gpu/chemistry_gpu.h:  void Free_Array_GPU_float(float *array_dev);
src/chemistry_gpu/chemistry_gpu.h:  void Copy_UVB_Rates_to_GPU();
src/chemistry_gpu/chemistry_gpu.h:  void Bind_GPU_Textures(int size, float *H_HI_h, float *H_HeI_h, float *H_HeII_h, float *I_HI_h, float *I_HeI_h,
src/chemistry_gpu/rates_Katz95.cuh:#ifdef CHEMISTRY_GPU
src/chemistry_gpu/rates_Katz95.cuh:  #include "../global/global_cuda.h"
src/chemistry_gpu/rates_Katz95.cuh:  #include "chemistry_gpu.h"
src/chemistry_gpu/chemistry_functions.cpp:#ifdef CHEMISTRY_GPU
src/chemistry_gpu/chemistry_functions.cpp:  #include "chemistry_gpu.h"
src/chemistry_gpu/chemistry_functions.cpp:    #include "../hydro/hydro_cuda.h"
src/chemistry_gpu/chemistry_functions.cpp:  chprintf("Initializing the GPU Chemistry Solver... \n");
src/chemistry_gpu/chemistry_functions.cpp:void Chem_GPU::Generate_Reaction_Rate_Table(Real **rate_table_array_d, Rate_Function_T rate_function, Real units)
src/chemistry_gpu/chemistry_functions.cpp:  Allocate_Array_GPU_Real(rate_table_array_d, H.N_Temp_bins);
src/chemistry_gpu/chemistry_functions.cpp:void Chem_GPU::Initialize(struct Parameters *P)
src/chemistry_gpu/chemistry_functions.cpp:void Chem_GPU::Initialize_Cooling_Rates()
src/chemistry_gpu/chemistry_functions.cpp:void Chem_GPU::Initialize_Reaction_Rates()
src/chemistry_gpu/chemistry_functions.cpp:void Chem_GPU::Initialize_UVB_Ionization_and_Heating_Rates(struct Parameters *P)
src/chemistry_gpu/chemistry_functions.cpp:  Copy_UVB_Rates_to_GPU();
src/chemistry_gpu/chemistry_functions.cpp:  Bind_GPU_Textures(n_uvb_rates_samples, Heat_rates_HI_h, Heat_rates_HeI_h, Heat_rates_HeII_h, Ion_rates_HI_h,
src/chemistry_gpu/chemistry_functions.cpp:void Chem_GPU::Copy_UVB_Rates_to_GPU()
src/chemistry_gpu/chemistry_functions.cpp:  Allocate_Array_GPU_float(&rates_z_d, n_uvb_rates_samples);
src/chemistry_gpu/chemistry_functions.cpp:  Allocate_Array_GPU_float(&Heat_rates_HI_d, n_uvb_rates_samples);
src/chemistry_gpu/chemistry_functions.cpp:  Allocate_Array_GPU_float(&Heat_rates_HeI_d, n_uvb_rates_samples);
src/chemistry_gpu/chemistry_functions.cpp:  Allocate_Array_GPU_float(&Heat_rates_HeII_d, n_uvb_rates_samples);
src/chemistry_gpu/chemistry_functions.cpp:  Allocate_Array_GPU_float(&Ion_rates_HI_d, n_uvb_rates_samples);
src/chemistry_gpu/chemistry_functions.cpp:  Allocate_Array_GPU_float(&Ion_rates_HeI_d, n_uvb_rates_samples);
src/chemistry_gpu/chemistry_functions.cpp:  Allocate_Array_GPU_float(&Ion_rates_HeII_d, n_uvb_rates_samples);
src/chemistry_gpu/chemistry_functions.cpp:void Chem_GPU::Reset()
src/chemistry_gpu/chemistry_functions.cpp:  Free_Array_GPU_float(rates_z_d);
src/chemistry_gpu/chemistry_functions.cpp:  Free_Array_GPU_float(Heat_rates_HI_d);
src/chemistry_gpu/chemistry_functions.cpp:  Free_Array_GPU_float(Heat_rates_HeI_d);
src/chemistry_gpu/chemistry_functions.cpp:  Free_Array_GPU_float(Heat_rates_HeII_d);
src/chemistry_gpu/chemistry_functions.cpp:  Free_Array_GPU_float(Ion_rates_HI_d);
src/chemistry_gpu/chemistry_functions.cpp:  Free_Array_GPU_float(Ion_rates_HeI_d);
src/chemistry_gpu/chemistry_functions.cpp:  Free_Array_GPU_float(Ion_rates_HeII_d);
src/chemistry_gpu/chemistry_io.cpp:#ifdef CHEMISTRY_GPU
src/chemistry_gpu/chemistry_io.cpp:  #include "chemistry_gpu.h"
src/chemistry_gpu/chemistry_io.cpp:void Chem_GPU::Load_UVB_Ionization_and_Heating_Rates(struct Parameters *P)
src/chemistry_gpu/rates.cuh:#ifdef CHEMISTRY_GPU
src/chemistry_gpu/rates.cuh:  #include "../global/global_cuda.h"
src/chemistry_gpu/rates.cuh:  #include "chemistry_gpu.h"
src/gravity/potential_SOR_3D.cpp:  AllocateMemory_GPU();
src/gravity/potential_SOR_3D.cpp:void Potential_SOR_3D::AllocateMemory_GPU(void)
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.input_d, n_cells_local);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.density_d, n_cells_local);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.potential_d, n_cells_potential);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_bool(&F.converged_d, 1);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.boundaries_buffer_x0_d, size_buffer_x);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.boundaries_buffer_x1_d, size_buffer_x);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.boundaries_buffer_y0_d, size_buffer_y);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.boundaries_buffer_y1_d, size_buffer_y);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.boundaries_buffer_z0_d, size_buffer_z);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.boundaries_buffer_z1_d, size_buffer_z);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.recv_boundaries_buffer_x0_d, size_buffer_x);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.recv_boundaries_buffer_x1_d, size_buffer_x);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.recv_boundaries_buffer_y0_d, size_buffer_y);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.recv_boundaries_buffer_y1_d, size_buffer_y);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.recv_boundaries_buffer_z0_d, size_buffer_z);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.recv_boundaries_buffer_z1_d, size_buffer_z);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.boundary_isolated_x0_d, n_ghost * ny_local * nz_local);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.boundary_isolated_x1_d, n_ghost * ny_local * nz_local);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.boundary_isolated_y0_d, n_ghost * nx_local * nz_local);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.boundary_isolated_y1_d, n_ghost * nx_local * nz_local);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.boundary_isolated_z0_d, n_ghost * nx_local * ny_local);
src/gravity/potential_SOR_3D.cpp:  Allocate_Array_GPU_Real(&F.boundary_isolated_z1_d, n_ghost * nx_local * ny_local);
src/gravity/potential_SOR_3D.cpp:    GPU_Error_Check(
src/gravity/potential_SOR_3D.cpp:        cudaMemcpy(F.potential_d, input_potential, n_cells_potential * sizeof(Real), cudaMemcpyHostToDevice));
src/gravity/potential_SOR_3D.cpp:  Grav.Copy_Isolated_Boundaries_To_GPU(P);
src/gravity/potential_SOR_3D.cpp:void Grav3D::Copy_Isolated_Boundaries_To_GPU(struct Parameters *P)
src/gravity/potential_SOR_3D.cpp:    Copy_Isolated_Boundary_To_GPU_buffer(F.pot_boundary_x0, Poisson_solver.F.boundary_isolated_x0_d,
src/gravity/potential_SOR_3D.cpp:    Copy_Isolated_Boundary_To_GPU_buffer(F.pot_boundary_x1, Poisson_solver.F.boundary_isolated_x1_d,
src/gravity/potential_SOR_3D.cpp:    Copy_Isolated_Boundary_To_GPU_buffer(F.pot_boundary_y0, Poisson_solver.F.boundary_isolated_y0_d,
src/gravity/potential_SOR_3D.cpp:    Copy_Isolated_Boundary_To_GPU_buffer(F.pot_boundary_y1, Poisson_solver.F.boundary_isolated_y1_d,
src/gravity/potential_SOR_3D.cpp:    Copy_Isolated_Boundary_To_GPU_buffer(F.pot_boundary_z0, Poisson_solver.F.boundary_isolated_z0_d,
src/gravity/potential_SOR_3D.cpp:    Copy_Isolated_Boundary_To_GPU_buffer(F.pot_boundary_z1, Poisson_solver.F.boundary_isolated_z1_d,
src/gravity/potential_SOR_3D.cpp:  if (boundary_flags[0] == 3) Set_Isolated_Boundary_GPU(0, 0, F.boundary_isolated_x0_d);
src/gravity/potential_SOR_3D.cpp:  if (boundary_flags[1] == 3) Set_Isolated_Boundary_GPU(0, 1, F.boundary_isolated_x1_d);
src/gravity/potential_SOR_3D.cpp:  if (boundary_flags[2] == 3) Set_Isolated_Boundary_GPU(1, 0, F.boundary_isolated_y0_d);
src/gravity/potential_SOR_3D.cpp:  if (boundary_flags[3] == 3) Set_Isolated_Boundary_GPU(1, 1, F.boundary_isolated_y1_d);
src/gravity/potential_SOR_3D.cpp:  if (boundary_flags[4] == 3) Set_Isolated_Boundary_GPU(2, 0, F.boundary_isolated_z0_d);
src/gravity/potential_SOR_3D.cpp:  if (boundary_flags[5] == 3) Set_Isolated_Boundary_GPU(2, 1, F.boundary_isolated_z1_d);
src/gravity/potential_SOR_3D.cpp:  Load_Transfer_Buffer_GPU(direction, side_load, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:  Unload_Transfer_Buffer_GPU(direction, side_unload, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost,
src/gravity/potential_SOR_3D.cpp:void Potential_SOR_3D::FreeMemory_GPU(void)
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.input_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.density_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.potential_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.boundaries_buffer_x0_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.boundaries_buffer_x1_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.boundaries_buffer_y0_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.boundaries_buffer_y1_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.boundaries_buffer_z0_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.boundaries_buffer_z1_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.recv_boundaries_buffer_x0_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.recv_boundaries_buffer_x1_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.recv_boundaries_buffer_y0_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.recv_boundaries_buffer_y1_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.recv_boundaries_buffer_z0_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.recv_boundaries_buffer_z1_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.boundary_isolated_x0_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.boundary_isolated_x1_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.boundary_isolated_y0_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.boundary_isolated_y1_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.boundary_isolated_z0_d);
src/gravity/potential_SOR_3D.cpp:  Free_Array_GPU_Real(F.boundary_isolated_z1_d);
src/gravity/potential_SOR_3D.cpp:  FreeMemory_GPU();
src/gravity/potential_SOR_3D.cpp:  // Load the transfer buffer in the GPU
src/gravity/potential_SOR_3D.cpp:    if (side == 0) Grav.Poisson_solver.Load_Transfer_Buffer_GPU_x0();
src/gravity/potential_SOR_3D.cpp:    if (side == 1) Grav.Poisson_solver.Load_Transfer_Buffer_GPU_x1();
src/gravity/potential_SOR_3D.cpp:    if (side == 0) Grav.Poisson_solver.Load_Transfer_Buffer_GPU_y0();
src/gravity/potential_SOR_3D.cpp:    if (side == 1) Grav.Poisson_solver.Load_Transfer_Buffer_GPU_y1();
src/gravity/potential_SOR_3D.cpp:    if (side == 0) Grav.Poisson_solver.Load_Transfer_Buffer_GPU_z0();
src/gravity/potential_SOR_3D.cpp:    if (side == 1) Grav.Poisson_solver.Load_Transfer_Buffer_GPU_z1();
src/gravity/potential_SOR_3D.cpp:  // Unload the transfer buffer in the GPU
src/gravity/potential_SOR_3D.cpp:    if (side == 0) Grav.Poisson_solver.Unload_Transfer_Buffer_GPU_x0();
src/gravity/potential_SOR_3D.cpp:    if (side == 1) Grav.Poisson_solver.Unload_Transfer_Buffer_GPU_x1();
src/gravity/potential_SOR_3D.cpp:    if (side == 0) Grav.Poisson_solver.Unload_Transfer_Buffer_GPU_y0();
src/gravity/potential_SOR_3D.cpp:    if (side == 1) Grav.Poisson_solver.Unload_Transfer_Buffer_GPU_y1();
src/gravity/potential_SOR_3D.cpp:    if (side == 0) Grav.Poisson_solver.Unload_Transfer_Buffer_GPU_z0();
src/gravity/potential_SOR_3D.cpp:    if (side == 1) Grav.Poisson_solver.Unload_Transfer_Buffer_GPU_z1();
src/gravity/potential_SOR_3D.cpp:void Potential_SOR_3D::Load_Transfer_Buffer_GPU_x0()
src/gravity/potential_SOR_3D.cpp:  Load_Transfer_Buffer_Half_GPU(0, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:  Load_Transfer_Buffer_GPU(0, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:void Potential_SOR_3D::Load_Transfer_Buffer_GPU_x1()
src/gravity/potential_SOR_3D.cpp:  Load_Transfer_Buffer_Half_GPU(0, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:  Load_Transfer_Buffer_GPU(0, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:void Potential_SOR_3D::Load_Transfer_Buffer_GPU_y0()
src/gravity/potential_SOR_3D.cpp:  Load_Transfer_Buffer_Half_GPU(1, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:  Load_Transfer_Buffer_GPU(1, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:void Potential_SOR_3D::Load_Transfer_Buffer_GPU_y1()
src/gravity/potential_SOR_3D.cpp:  Load_Transfer_Buffer_Half_GPU(1, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:  Load_Transfer_Buffer_GPU(1, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:void Potential_SOR_3D::Load_Transfer_Buffer_GPU_z0()
src/gravity/potential_SOR_3D.cpp:  Load_Transfer_Buffer_Half_GPU(2, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:  Load_Transfer_Buffer_GPU(2, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:void Potential_SOR_3D::Load_Transfer_Buffer_GPU_z1()
src/gravity/potential_SOR_3D.cpp:  Load_Transfer_Buffer_Half_GPU(2, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:  Load_Transfer_Buffer_GPU(2, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:void Potential_SOR_3D::Unload_Transfer_Buffer_GPU_x0()
src/gravity/potential_SOR_3D.cpp:  Unload_Transfer_Buffer_Half_GPU(0, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:  Unload_Transfer_Buffer_GPU(0, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:void Potential_SOR_3D::Unload_Transfer_Buffer_GPU_x1()
src/gravity/potential_SOR_3D.cpp:  Unload_Transfer_Buffer_Half_GPU(0, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:  Unload_Transfer_Buffer_GPU(0, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:void Potential_SOR_3D::Unload_Transfer_Buffer_GPU_y0()
src/gravity/potential_SOR_3D.cpp:  Unload_Transfer_Buffer_Half_GPU(1, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:  Unload_Transfer_Buffer_GPU(1, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:void Potential_SOR_3D::Unload_Transfer_Buffer_GPU_y1()
src/gravity/potential_SOR_3D.cpp:  Unload_Transfer_Buffer_Half_GPU(1, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:  Unload_Transfer_Buffer_GPU(1, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:void Potential_SOR_3D::Unload_Transfer_Buffer_GPU_z0()
src/gravity/potential_SOR_3D.cpp:  Unload_Transfer_Buffer_Half_GPU(2, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:  Unload_Transfer_Buffer_GPU(2, 0, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:void Potential_SOR_3D::Unload_Transfer_Buffer_GPU_z1()
src/gravity/potential_SOR_3D.cpp:  Unload_Transfer_Buffer_Half_GPU(2, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/potential_SOR_3D.cpp:  Unload_Transfer_Buffer_GPU(2, 1, nx_local, ny_local, nz_local, n_ghost_transfer, n_ghost, F.potential_d,
src/gravity/paris/PoissonZero3DBlockedGPU.hpp:#include "../../utils/gpu.hpp"
src/gravity/paris/PoissonZero3DBlockedGPU.hpp:class PoissonZero3DBlockedGPU
src/gravity/paris/PoissonZero3DBlockedGPU.hpp:  PoissonZero3DBlockedGPU(const int n[3], const double lo[3], const double hi[3], const int m[3], const int id[3]);
src/gravity/paris/PoissonZero3DBlockedGPU.hpp:  ~PoissonZero3DBlockedGPU();
src/gravity/paris/PoissonZero3DBlockedGPU.hpp:#ifndef MPI_GPU
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  #include "PoissonZero3DBlockedGPU.hpp"
src/gravity/paris/PoissonZero3DBlockedGPU.cu:PoissonZero3DBlockedGPU::PoissonZero3DBlockedGPU(const int n[3], const double lo[3], const double hi[3], const int m[3],
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cufftPlanMany(&d2zk_, 1, &nk_, &nk_, 1, nk_, &nkh, 1, nkh, CUFFT_D2Z, dip_ * djq_));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cufftPlanMany(&d2zj_, 1, &nj_, &nj_, 1, nj_, &njh, 1, njh, CUFFT_D2Z, dip_ * dkq_));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cufftPlanMany(&d2zi_, 1, &ni_, &ni_, 1, ni_, &nih, 1, nih, CUFFT_D2Z, dkq_ * djp_));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  #ifndef MPI_GPU
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaHostAlloc(&ha_, bytes_ + bytes_, cudaHostAllocDefault));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:PoissonZero3DBlockedGPU::~PoissonZero3DBlockedGPU()
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  #ifndef MPI_GPU
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaFreeHost(ha_));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cufftDestroy(d2zi_));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cufftDestroy(d2zj_));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cufftDestroy(d2zk_));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:void PoissonZero3DBlockedGPU::solve(const long bytes, double *const density, double *const potential) const
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  gpuFor(
src/gravity/paris/PoissonZero3DBlockedGPU.cu:      mp, mq, dip, djq, dk, GPU_LAMBDA(const int p, const int q, const int i, const int j, const int k) {
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  #ifndef MPI_GPU
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaMemcpy(ha_, ua, bytes_, cudaMemcpyDeviceToHost));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaMemcpyAsync(ub, hb_, bytes_, cudaMemcpyHostToDevice, 0));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaDeviceSynchronize());
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  gpuFor(
src/gravity/paris/PoissonZero3DBlockedGPU.cu:      dip, djq, nk / 2 + 1, GPU_LAMBDA(const int i, const int j, const int k) {
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cufftExecD2Z(d2zk_, ua, uc));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  gpuFor(
src/gravity/paris/PoissonZero3DBlockedGPU.cu:      dip, nk / 2 + 1, djq, GPU_LAMBDA(const int i, const int k, const int j) {
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  #ifndef MPI_GPU
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaMemcpy(ha_, ua, bytes_, cudaMemcpyDeviceToHost));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaMemcpyAsync(ub, hb_, bytes_, cudaMemcpyHostToDevice, 0));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaDeviceSynchronize());
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  gpuFor(
src/gravity/paris/PoissonZero3DBlockedGPU.cu:      dip, dkq, nj / 2 + 1, GPU_LAMBDA(const int i, const int k, const int j) {
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cufftExecD2Z(d2zj_, ua, uc));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  gpuFor(
src/gravity/paris/PoissonZero3DBlockedGPU.cu:      dkq, nj / 2 + 1, dip, GPU_LAMBDA(const int k, const int j, const int i) {
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  #ifndef MPI_GPU
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaMemcpy(ha_, ua, bytes_, cudaMemcpyDeviceToHost));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaMemcpyAsync(ub, hb_, bytes_, cudaMemcpyHostToDevice, 0));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaDeviceSynchronize());
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  gpuFor(
src/gravity/paris/PoissonZero3DBlockedGPU.cu:      dkq, djp, ni / 2 + 1, GPU_LAMBDA(const int k, const int j, const int i) {
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cufftExecD2Z(d2zi_, ua, uc));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:    gpuFor(
src/gravity/paris/PoissonZero3DBlockedGPU.cu:        dkq, djp, ni / 2 + 1, GPU_LAMBDA(const int k, const int j, const int i) {
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cufftExecD2Z(d2zi_, ua, uc));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  gpuFor(
src/gravity/paris/PoissonZero3DBlockedGPU.cu:      dkq, ni / 2 + 1, djp, GPU_LAMBDA(const int k, const int i, const int j) {
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  #ifndef MPI_GPU
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaMemcpy(ha_, ua, bytes_, cudaMemcpyDeviceToHost));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaMemcpyAsync(ub, hb_, bytes_, cudaMemcpyHostToDevice, 0));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaDeviceSynchronize());
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  gpuFor(
src/gravity/paris/PoissonZero3DBlockedGPU.cu:      dkq, dip, nj / 2 + 1, GPU_LAMBDA(const int k, const int i, const int j) {
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cufftExecD2Z(d2zj_, ua, uc));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  gpuFor(
src/gravity/paris/PoissonZero3DBlockedGPU.cu:      dip, nj / 2 + 1, dkq, GPU_LAMBDA(const int i, const int j, const int k) {
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  #ifndef MPI_GPU
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaMemcpy(ha_, ua, bytes_, cudaMemcpyDeviceToHost));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaMemcpyAsync(ub, hb_, bytes_, cudaMemcpyHostToDevice, 0));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaDeviceSynchronize());
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  gpuFor(
src/gravity/paris/PoissonZero3DBlockedGPU.cu:      dip, djq, nk / 2 + 1, GPU_LAMBDA(const int i, const int j, const int k) {
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cufftExecD2Z(d2zk_, ua, uc));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  gpuFor(
src/gravity/paris/PoissonZero3DBlockedGPU.cu:      dip, djq, nk / 2 + 1, GPU_LAMBDA(const int i, const int j, const int k) {
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  #ifndef MPI_GPU
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaMemcpy(ha_, ua, bytes_, cudaMemcpyDeviceToHost));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaMemcpyAsync(ub, hb_, bytes_, cudaMemcpyHostToDevice, 0));
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  GPU_Error_Check(cudaDeviceSynchronize());
src/gravity/paris/PoissonZero3DBlockedGPU.cu:  gpuFor(
src/gravity/paris/PoissonZero3DBlockedGPU.cu:      mp, dip, mq, djq, dk, GPU_LAMBDA(const int p, const int i, const int q, const int j, const int k) {
src/gravity/paris/HenryPeriodic.hpp:#include "../../utils/gpu.hpp"
src/gravity/paris/HenryPeriodic.hpp:#ifndef MPI_GPU
src/gravity/paris/HenryPeriodic.hpp:#if defined(__HIP__) || defined(__CUDACC__)
src/gravity/paris/HenryPeriodic.hpp:  gpuFor(
src/gravity/paris/HenryPeriodic.hpp:      mp, mq, dip, djq, dk, GPU_LAMBDA(const int p, const int q, const int i, const int j, const int k) {
src/gravity/paris/HenryPeriodic.hpp:  #ifndef MPI_GPU
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaMemcpy(ha_, a, bytes, cudaMemcpyDeviceToHost));
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaMemcpy(b, hb_, bytes, cudaMemcpyHostToDevice));
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaDeviceSynchronize());
src/gravity/paris/HenryPeriodic.hpp:    gpuFor(
src/gravity/paris/HenryPeriodic.hpp:        iHi - iLo, jHi - jLo, mk, dk, GPU_LAMBDA(const int i, const int j, const int pq, const int k) {
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cufftExecD2Z(r2ck_, a, bc));
src/gravity/paris/HenryPeriodic.hpp:    gpuFor(
src/gravity/paris/HenryPeriodic.hpp:        mjq, iHi - iLo, jHi - jLo, dhq, GPU_LAMBDA(const int q, const int i, const int j, const int k) {
src/gravity/paris/HenryPeriodic.hpp:  #ifndef MPI_GPU
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaMemcpy(ha_, a, bytes, cudaMemcpyDeviceToHost));
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaMemcpy(b, hb_, bytes, cudaMemcpyHostToDevice));
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaDeviceSynchronize());
src/gravity/paris/HenryPeriodic.hpp:    gpuFor(
src/gravity/paris/HenryPeriodic.hpp:        kHi - kLo, iHi - iLo, mj, mq, djq, GPU_LAMBDA(const int k, const int i, const int r, const int q, const int j) {
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cufftExecZ2Z(c2cj_, ac, bc, CUFFT_FORWARD));
src/gravity/paris/HenryPeriodic.hpp:    gpuFor(
src/gravity/paris/HenryPeriodic.hpp:        mip, kHi - kLo, iHi - iLo, djp, GPU_LAMBDA(const int p, const int k, const int i, const int j) {
src/gravity/paris/HenryPeriodic.hpp:  #ifndef MPI_GPU
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaMemcpy(ha_, a, bytes, cudaMemcpyDeviceToHost));
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaMemcpy(b, hb_, bytes, cudaMemcpyHostToDevice));
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaDeviceSynchronize());
src/gravity/paris/HenryPeriodic.hpp:    gpuFor(
src/gravity/paris/HenryPeriodic.hpp:        jHi - jLo, kHi - kLo, mi, mp, dip, GPU_LAMBDA(const int j, const int k, const int r, const int p, const int i) {
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cufftExecZ2Z(c2ci_, ac, bc, CUFFT_FORWARD));
src/gravity/paris/HenryPeriodic.hpp:  gpuFor(
src/gravity/paris/HenryPeriodic.hpp:      jHi - jLo, kHi - kLo, ni, GPU_LAMBDA(const int j0, const int k0, const int i) {
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cufftExecZ2Z(c2ci_, ac, bc, CUFFT_INVERSE));
src/gravity/paris/HenryPeriodic.hpp:    gpuFor(
src/gravity/paris/HenryPeriodic.hpp:        mi, mp, jHi - jLo, kHi - kLo, dip, GPU_LAMBDA(const int r, const int p, const int j, const int k, const int i) {
src/gravity/paris/HenryPeriodic.hpp:  #ifndef MPI_GPU
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaMemcpy(ha_, a, bytes, cudaMemcpyDeviceToHost));
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaMemcpy(b, hb_, bytes, cudaMemcpyHostToDevice));
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaDeviceSynchronize());
src/gravity/paris/HenryPeriodic.hpp:    gpuFor(
src/gravity/paris/HenryPeriodic.hpp:        kHi - kLo, iHi - iLo, mip, djp, GPU_LAMBDA(const int k, const int i, const int p, const int j) {
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cufftExecZ2Z(c2cj_, ac, bc, CUFFT_INVERSE));
src/gravity/paris/HenryPeriodic.hpp:    gpuFor(
src/gravity/paris/HenryPeriodic.hpp:        mj, mq, kHi - kLo, iHi - iLo, djq, GPU_LAMBDA(const int r, const int q, const int k, const int i, const int j) {
src/gravity/paris/HenryPeriodic.hpp:  #ifndef MPI_GPU
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaMemcpy(ha_, a, bytes, cudaMemcpyDeviceToHost));
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaMemcpy(b, hb_, bytes, cudaMemcpyHostToDevice));
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaDeviceSynchronize());
src/gravity/paris/HenryPeriodic.hpp:    gpuFor(
src/gravity/paris/HenryPeriodic.hpp:        iHi - iLo, jHi - jLo, mjq, dhq, GPU_LAMBDA(const int i, const int j, const int q, const int k) {
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cufftExecZ2D(c2rk_, ac, b));
src/gravity/paris/HenryPeriodic.hpp:    gpuFor(
src/gravity/paris/HenryPeriodic.hpp:        mk, iHi - iLo, jHi - jLo, dk, GPU_LAMBDA(const int pq, const int i, const int j, const int k) {
src/gravity/paris/HenryPeriodic.hpp:  #ifndef MPI_GPU
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaMemcpy(ha_, a, bytes, cudaMemcpyDeviceToHost));
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaMemcpy(b, hb_, bytes, cudaMemcpyHostToDevice));
src/gravity/paris/HenryPeriodic.hpp:  GPU_Error_Check(cudaDeviceSynchronize());
src/gravity/paris/HenryPeriodic.hpp:    gpuFor(
src/gravity/paris/HenryPeriodic.hpp:        mp, dip, mq, djq, kHi - kLo, GPU_LAMBDA(const int p, const int i, const int q, const int j, const int k) {
src/gravity/paris/README.md:*PoissonZero3DBlockedGPU*
src/gravity/paris/README.md:*PoissonZero3DBlockedGPU* uses discrete sine transforms (DSTs) instead of Fourier transforms to enforce zero-valued, non-periodic boundary conditions.
src/gravity/paris/README.md:*PotentialParisGalactic::Get_Potential()* uses *PoissonZero3DBlockedGPU::solve()* as follows.
src/gravity/paris/README.md:- Call *PoissonZero3DBlockedGPU::solve()* with this density with zero-valued boundaries.
src/gravity/paris/HenryPeriodic.cu:  GPU_Error_Check(cufftPlanMany(&c2ci_, 1, &ni_, &ni_, 1, ni_, &ni_, 1, ni_, CUFFT_Z2Z, djp_ * dhq_));
src/gravity/paris/HenryPeriodic.cu:  GPU_Error_Check(cufftPlanMany(&c2cj_, 1, &nj_, &nj_, 1, nj_, &nj_, 1, nj_, CUFFT_Z2Z, dip_ * dhq_));
src/gravity/paris/HenryPeriodic.cu:  GPU_Error_Check(cufftPlanMany(&c2rk_, 1, &nk_, &nh_, 1, nh_, &nk_, 1, nk_, CUFFT_Z2D, dip_ * djq_));
src/gravity/paris/HenryPeriodic.cu:  GPU_Error_Check(cufftPlanMany(&r2ck_, 1, &nk_, &nk_, 1, nk_, &nh_, 1, nh_, CUFFT_D2Z, dip_ * djq_));
src/gravity/paris/HenryPeriodic.cu:  #ifndef MPI_GPU
src/gravity/paris/HenryPeriodic.cu:  GPU_Error_Check(cudaHostAlloc(&ha_, bytes_ + bytes_, cudaHostAllocDefault));
src/gravity/paris/HenryPeriodic.cu:  #ifndef MPI_GPU
src/gravity/paris/HenryPeriodic.cu:  GPU_Error_Check(cudaFreeHost(ha_));
src/gravity/paris/HenryPeriodic.cu:  GPU_Error_Check(cufftDestroy(r2ck_));
src/gravity/paris/HenryPeriodic.cu:  GPU_Error_Check(cufftDestroy(c2rk_));
src/gravity/paris/HenryPeriodic.cu:  GPU_Error_Check(cufftDestroy(c2cj_));
src/gravity/paris/HenryPeriodic.cu:  GPU_Error_Check(cufftDestroy(c2ci_));
src/gravity/potential_SOR_3D_gpu.cu:#if defined(CUDA) && defined(GRAVITY) && defined(SOR)
src/gravity/potential_SOR_3D_gpu.cu:  #include "../global/global_cuda.h"
src/gravity/potential_SOR_3D_gpu.cu:void Potential_SOR_3D::Allocate_Array_GPU_Real(Real **array_dev, grav_int_t size)
src/gravity/potential_SOR_3D_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)array_dev, size * sizeof(Real)));
src/gravity/potential_SOR_3D_gpu.cu:void Potential_SOR_3D::Allocate_Array_GPU_bool(bool **array_dev, grav_int_t size)
src/gravity/potential_SOR_3D_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)array_dev, size * sizeof(bool)));
src/gravity/potential_SOR_3D_gpu.cu:void Potential_SOR_3D::Free_Array_GPU_Real(Real *array_dev) { GPU_Error_Check(cudaFree(array_dev)); }
src/gravity/potential_SOR_3D_gpu.cu:void Potential_SOR_3D::Free_Array_GPU_bool(bool *array_dev) { GPU_Error_Check(cudaFree(array_dev)); }
src/gravity/potential_SOR_3D_gpu.cu:  cudaMemcpy(input_d, input_density_h, n_cells * sizeof(Real), cudaMemcpyHostToDevice);
src/gravity/potential_SOR_3D_gpu.cu:  // set values for GPU kernels
src/gravity/potential_SOR_3D_gpu.cu:void Grav3D::Copy_Isolated_Boundary_To_GPU_buffer(Real *isolated_boundary_h, Real *isolated_boundary_d,
src/gravity/potential_SOR_3D_gpu.cu:  cudaMemcpy(isolated_boundary_d, isolated_boundary_h, boundary_size * sizeof(Real), cudaMemcpyHostToDevice);
src/gravity/potential_SOR_3D_gpu.cu:  // set values for GPU kernels
src/gravity/potential_SOR_3D_gpu.cu:  // set values for GPU kernels
src/gravity/potential_SOR_3D_gpu.cu:  cudaMemset(converged_d, 1, sizeof(bool));
src/gravity/potential_SOR_3D_gpu.cu:  cudaMemcpy(converged_h, converged_d, sizeof(bool), cudaMemcpyDeviceToHost);
src/gravity/potential_SOR_3D_gpu.cu:  // set values for GPU kernels
src/gravity/potential_SOR_3D_gpu.cu:  cudaMemset(converged_d, 1, sizeof(bool));
src/gravity/potential_SOR_3D_gpu.cu:  // set values for GPU kernels
src/gravity/potential_SOR_3D_gpu.cu:  cudaMemcpy(converged_h, converged_d, sizeof(bool), cudaMemcpyDeviceToHost);
src/gravity/potential_SOR_3D_gpu.cu:__global__ void Set_Isolated_Boundary_GPU_kernel(int direction, int side, int size_buffer, int n_i, int n_j,
src/gravity/potential_SOR_3D_gpu.cu:void Potential_SOR_3D::Set_Isolated_Boundary_GPU(int direction, int side, Real *boundary_d)
src/gravity/potential_SOR_3D_gpu.cu:  // set values for GPU kernels
src/gravity/potential_SOR_3D_gpu.cu:  // Set_Isolated_Boundary_GPU_kernel<<<dim1dGrid,dim1dBlock>>>( direction,
src/gravity/potential_SOR_3D_gpu.cu:  hipLaunchKernelGGL(Set_Isolated_Boundary_GPU_kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, size_buffer, n_i,
src/gravity/potential_SOR_3D_gpu.cu:  cudaMemcpy(output_potential, F.potential_d, n_cells_potential * sizeof(Real), cudaMemcpyDeviceToHost);
src/gravity/potential_SOR_3D_gpu.cu:  cudaMemcpy(F.potential_d, output_potential, n_cells_potential * sizeof(Real), cudaMemcpyHostToDevice);
src/gravity/potential_SOR_3D_gpu.cu:__global__ void Load_Transfer_Buffer_GPU_kernel_SOR(int direction, int side, int size_buffer, int n_i, int n_j, int nx,
src/gravity/potential_SOR_3D_gpu.cu:__global__ void Load_Transfer_Buffer_GPU_Half_kernel(int direction, int side, int size_buffer, int n_i, int n_j, int nx,
src/gravity/potential_SOR_3D_gpu.cu:__global__ void Unload_Transfer_Buffer_GPU_kernel_SOR(int direction, int side, int size_buffer, int n_i, int n_j,
src/gravity/potential_SOR_3D_gpu.cu:__global__ void Unload_Transfer_Buffer_GPU_Half_kernel(int direction, int side, int size_buffer, int n_i, int n_j,
src/gravity/potential_SOR_3D_gpu.cu:void Potential_SOR_3D::Load_Transfer_Buffer_GPU(int direction, int side, int nx, int ny, int nz, int n_ghost_transfer,
src/gravity/potential_SOR_3D_gpu.cu:  // set values for GPU kernels
src/gravity/potential_SOR_3D_gpu.cu:  // Load_Transfer_Buffer_GPU_kernel<<<dim1dGrid,dim1dBlock>>>( direction, side,
src/gravity/potential_SOR_3D_gpu.cu:  hipLaunchKernelGGL(Load_Transfer_Buffer_GPU_kernel_SOR, dim1dGrid, dim1dBlock, 0, 0, direction, side, size_buffer,
src/gravity/potential_SOR_3D_gpu.cu:void Potential_SOR_3D::Load_Transfer_Buffer_Half_GPU(int direction, int side, int nx, int ny, int nz,
src/gravity/potential_SOR_3D_gpu.cu:  // set values for GPU kernels
src/gravity/potential_SOR_3D_gpu.cu:  // Load_Transfer_Buffer_GPU_Half_kernel<<<dim1dGrid,dim1dBlock>>>( direction,
src/gravity/potential_SOR_3D_gpu.cu:  hipLaunchKernelGGL(Load_Transfer_Buffer_GPU_Half_kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, size_buffer,
src/gravity/potential_SOR_3D_gpu.cu:void Potential_SOR_3D::Unload_Transfer_Buffer_GPU(int direction, int side, int nx, int ny, int nz, int n_ghost_transfer,
src/gravity/potential_SOR_3D_gpu.cu:  // set values for GPU kernels
src/gravity/potential_SOR_3D_gpu.cu:  // Unload_Transfer_Buffer_GPU_kernel<<<dim1dGrid,dim1dBlock>>>( direction,
src/gravity/potential_SOR_3D_gpu.cu:  hipLaunchKernelGGL(Unload_Transfer_Buffer_GPU_kernel_SOR, dim1dGrid, dim1dBlock, 0, 0, direction, side, size_buffer,
src/gravity/potential_SOR_3D_gpu.cu:void Potential_SOR_3D::Unload_Transfer_Buffer_Half_GPU(int direction, int side, int nx, int ny, int nz,
src/gravity/potential_SOR_3D_gpu.cu:  // set values for GPU kernels
src/gravity/potential_SOR_3D_gpu.cu:  // Unload_Transfer_Buffer_GPU_Half_kernel<<<dim1dGrid,dim1dBlock>>>(
src/gravity/potential_SOR_3D_gpu.cu:  hipLaunchKernelGGL(Unload_Transfer_Buffer_GPU_Half_kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, size_buffer,
src/gravity/potential_SOR_3D_gpu.cu:  GPU_Error_Check(cudaMemcpy(transfer_buffer_h, transfer_buffer_d, size_buffer * sizeof(Real), cudaMemcpyDeviceToHost));
src/gravity/potential_SOR_3D_gpu.cu:  GPU_Error_Check(cudaMemcpy(transfer_buffer_d, transfer_buffer_h, size_buffer * sizeof(Real), cudaMemcpyHostToDevice));
src/gravity/grav3D.cpp:  #ifdef GRAVITY_GPU
src/gravity/grav3D.cpp:  AllocateMemory_GPU();
src/gravity/gravity_boundaries_gpu.cu:#if defined(GRAVITY) && defined(GRAVITY_GPU)
src/gravity/gravity_boundaries_gpu.cu:void Grid3D::Set_Potential_Boundaries_Isolated_GPU(int direction, int side, int *flags)
src/gravity/gravity_boundaries_gpu.cu:  // set values for GPU kernels
src/gravity/gravity_boundaries_gpu.cu:  cudaMemcpy(pot_boundary_d, pot_boundary_h, size_buffer * sizeof(Real), cudaMemcpyHostToDevice);
src/gravity/gravity_boundaries_gpu.cu:  cudaDeviceSynchronize();
src/gravity/gravity_boundaries_gpu.cu:void Grid3D::Set_Potential_Boundaries_Periodic_GPU(int direction, int side, int *flags)
src/gravity/gravity_boundaries_gpu.cu:  // set values for GPU kernels
src/gravity/gravity_boundaries_gpu.cu:__global__ void Load_Transfer_Buffer_GPU_kernel(int direction, int side, int size_buffer, int n_i, int n_j, int nx,
src/gravity/gravity_boundaries_gpu.cu:int Grid3D::Load_Gravity_Potential_To_Buffer_GPU(int direction, int side, Real *buffer, int buffer_start)
src/gravity/gravity_boundaries_gpu.cu:  // set values for GPU kernels
src/gravity/gravity_boundaries_gpu.cu:  hipLaunchKernelGGL(Load_Transfer_Buffer_GPU_kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, size_buffer, n_i,
src/gravity/gravity_boundaries_gpu.cu:  GPU_Error_Check(cudaDeviceSynchronize());
src/gravity/gravity_boundaries_gpu.cu:__global__ void Unload_Transfer_Buffer_GPU_kernel(int direction, int side, int size_buffer, int n_i, int n_j, int nx,
src/gravity/gravity_boundaries_gpu.cu:void Grid3D::Unload_Gravity_Potential_from_Buffer_GPU(int direction, int side, Real *buffer, int buffer_start)
src/gravity/gravity_boundaries_gpu.cu:  // set values for GPU kernels
src/gravity/gravity_boundaries_gpu.cu:  hipLaunchKernelGGL(Unload_Transfer_Buffer_GPU_kernel, dim1dGrid, dim1dBlock, 0, 0, direction, side, size_buffer, n_i,
src/gravity/gravity_functions.cpp:  #include "../mpi/cuda_mpi_routines.h"
src/gravity/gravity_functions.cpp:      #ifdef CHEMISTRY_GPU
src/gravity/gravity_functions.cpp:  #if defined(PARTICLES_GPU) && defined(PRINT_MAX_MEMORY_USAGE)
src/gravity/gravity_functions.cpp:  #ifdef GRAVITY_GPU
src/gravity/gravity_functions.cpp:      #ifdef GRAVITY_GPU
src/gravity/gravity_functions.cpp:        #error "GRAVITY_GPU not yet supported with PARIS_GALACTIC_TEST"
src/gravity/gravity_functions.cpp:    #ifdef GRAVITY_GPU
src/gravity/gravity_functions.cpp:  GPU_Error_Check(cudaMemcpy(Grav.F.analytic_potential_d, Grav.F.analytic_potential_h,
src/gravity/gravity_functions.cpp:                             Grav.n_cells_potential * sizeof(Real), cudaMemcpyHostToDevice));
src/gravity/gravity_functions.cpp:    #ifdef GRAVITY_GPU
src/gravity/gravity_functions.cpp:  Add_Analytic_Potential_GPU();
src/gravity/gravity_functions.cpp:    #endif    // GRAVITY_GPU else
src/gravity/gravity_functions.cpp:  #ifdef GRAVITY_GPU
src/gravity/gravity_functions.cpp:  Copy_Hydro_Density_to_Gravity_GPU();
src/gravity/gravity_functions.cpp:  #endif  // GRAVITY_GPU
src/gravity/gravity_functions.cpp:  #ifdef GRAVITY_GPU
src/gravity/gravity_functions.cpp:  Extrapolate_Grav_Potential_GPU();
src/gravity/gravity_functions.cpp:  #endif  // GRAVITY_GPU
src/gravity/potential_paris_galactic.cu:  #include "../utils/gpu.hpp"
src/gravity/potential_paris_galactic.cu:  #ifndef GRAVITY_GPU
src/gravity/potential_paris_galactic.cu:  #ifdef GRAVITY_GPU
src/gravity/potential_paris_galactic.cu:  GPU_Error_Check(cudaMemcpyAsync(da, density, densityBytes_, cudaMemcpyHostToDevice, 0));
src/gravity/potential_paris_galactic.cu:  GPU_Error_Check(cudaMemcpyAsync(dc_, potential, potentialBytes_, cudaMemcpyHostToDevice, 0));
src/gravity/potential_paris_galactic.cu:  gpuFor(
src/gravity/potential_paris_galactic.cu:      nk, nj, ni, GPU_LAMBDA(const int k, const int j, const int i) {
src/gravity/potential_paris_galactic.cu:  gpuFor(
src/gravity/potential_paris_galactic.cu:      nk, nj, ni, GPU_LAMBDA(const int k, const int j, const int i) {
src/gravity/potential_paris_galactic.cu:  #ifndef GRAVITY_GPU
src/gravity/potential_paris_galactic.cu:  GPU_Error_Check(cudaMemcpy(potential, dc_, potentialBytes_, cudaMemcpyDeviceToHost));
src/gravity/potential_paris_galactic.cu:  pp_ = new PoissonZero3DBlockedGPU(n, lo_, hi, m, id);
src/gravity/potential_paris_galactic.cu:  GPU_Error_Check(cudaMalloc(reinterpret_cast<void **>(&da_), std::max(minBytes_, densityBytes_)));
src/gravity/potential_paris_galactic.cu:  GPU_Error_Check(cudaMalloc(reinterpret_cast<void **>(&db_), std::max(minBytes_, densityBytes_)));
src/gravity/potential_paris_galactic.cu:  #ifndef GRAVITY_GPU
src/gravity/potential_paris_galactic.cu:  GPU_Error_Check(cudaMalloc(reinterpret_cast<void **>(&dc_), potentialBytes_));
src/gravity/potential_paris_galactic.cu:  #ifndef GRAVITY_GPU
src/gravity/potential_paris_galactic.cu:    GPU_Error_Check(cudaFree(dc_));
src/gravity/potential_paris_galactic.cu:    GPU_Error_Check(cudaFree(db_));
src/gravity/potential_paris_galactic.cu:    GPU_Error_Check(cudaFree(da_));
src/gravity/static_grav.h:/*! \file gravity_cuda.cu
src/gravity/static_grav.h:           functions in hydro_cuda.cu. */
src/gravity/static_grav.h:// Work around lack of pow(Real,int) in Hip Clang for Rocm 3.5
src/gravity/gravity_functions_gpu.cu:#if defined(GRAVITY) && defined(GRAVITY_GPU)
src/gravity/gravity_functions_gpu.cu:void Grav3D::AllocateMemory_GPU()
src/gravity/gravity_functions_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)&F.density_d, n_cells * sizeof(Real)));
src/gravity/gravity_functions_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)&F.potential_d, n_cells_potential * sizeof(Real)));
src/gravity/gravity_functions_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)&F.potential_1_d, n_cells_potential * sizeof(Real)));
src/gravity/gravity_functions_gpu.cu:  #ifdef GRAVITY_GPU
src/gravity/gravity_functions_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)&F.analytic_potential_d, n_cells_potential * sizeof(Real)));
src/gravity/gravity_functions_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)&F.pot_boundary_x0_d, N_GHOST_POTENTIAL * ny_local * nz_local * sizeof(Real)));
src/gravity/gravity_functions_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)&F.pot_boundary_x1_d, N_GHOST_POTENTIAL * ny_local * nz_local * sizeof(Real)));
src/gravity/gravity_functions_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)&F.pot_boundary_y0_d, N_GHOST_POTENTIAL * nx_local * nz_local * sizeof(Real)));
src/gravity/gravity_functions_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)&F.pot_boundary_y1_d, N_GHOST_POTENTIAL * nx_local * nz_local * sizeof(Real)));
src/gravity/gravity_functions_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)&F.pot_boundary_z0_d, N_GHOST_POTENTIAL * nx_local * ny_local * sizeof(Real)));
src/gravity/gravity_functions_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)&F.pot_boundary_z1_d, N_GHOST_POTENTIAL * nx_local * ny_local * sizeof(Real)));
src/gravity/gravity_functions_gpu.cu:  #endif  // GRAVITY_GPU
src/gravity/gravity_functions_gpu.cu:  chprintf("Allocated Gravity GPU memory \n");
src/gravity/gravity_functions_gpu.cu:void Grav3D::FreeMemory_GPU(void)
src/gravity/gravity_functions_gpu.cu:  cudaFree(F.density_d);
src/gravity/gravity_functions_gpu.cu:  cudaFree(F.potential_d);
src/gravity/gravity_functions_gpu.cu:  cudaFree(F.potential_1_d);
src/gravity/gravity_functions_gpu.cu:  #ifdef GRAVITY_GPU
src/gravity/gravity_functions_gpu.cu:  cudaFree(F.analytic_potential_d);
src/gravity/gravity_functions_gpu.cu:  cudaFree(F.pot_boundary_x0_d);
src/gravity/gravity_functions_gpu.cu:  cudaFree(F.pot_boundary_x1_d);
src/gravity/gravity_functions_gpu.cu:  cudaFree(F.pot_boundary_y0_d);
src/gravity/gravity_functions_gpu.cu:  cudaFree(F.pot_boundary_y1_d);
src/gravity/gravity_functions_gpu.cu:  cudaFree(F.pot_boundary_z0_d);
src/gravity/gravity_functions_gpu.cu:  cudaFree(F.pot_boundary_z1_d);
src/gravity/gravity_functions_gpu.cu:  #endif  // GRAVITY_GPU
src/gravity/gravity_functions_gpu.cu:void Grid3D::Copy_Hydro_Density_to_Gravity_GPU()
src/gravity/gravity_functions_gpu.cu:  // set values for GPU kernels
src/gravity/gravity_functions_gpu.cu:void Grid3D::Add_Analytic_Potential_GPU()
src/gravity/gravity_functions_gpu.cu:  // set values for GPU kernels
src/gravity/gravity_functions_gpu.cu:  cudaDeviceSynchronize();
src/gravity/gravity_functions_gpu.cu:  /*gpuFor(10,
src/gravity/gravity_functions_gpu.cu:    GPU_LAMBDA(const int i) {
src/gravity/gravity_functions_gpu.cu:void Grid3D::Extrapolate_Grav_Potential_GPU()
src/gravity/gravity_functions_gpu.cu:  // set values for GPU kernels
src/gravity/gravity_functions_gpu.cu:void Grid3D::Copy_Potential_From_GPU()
src/gravity/gravity_functions_gpu.cu:  GPU_Error_Check(cudaMemcpy(Grav.F.potential_h, Grav.F.potential_d, Grav.n_cells_potential * sizeof(Real),
src/gravity/gravity_functions_gpu.cu:                             cudaMemcpyDeviceToHost));
src/gravity/gravity_functions_gpu.cu:  cudaDeviceSynchronize();
src/gravity/potential_paris_3D.cu:  #include "../utils/gpu.hpp"
src/gravity/potential_paris_3D.cu:  #ifdef GRAVITY_GPU
src/gravity/potential_paris_3D.cu:  GPU_Error_Check(cudaMemcpy(db, density, densityBytes_, cudaMemcpyDeviceToDevice));
src/gravity/potential_paris_3D.cu:  GPU_Error_Check(cudaMemcpy(db, density, densityBytes_, cudaMemcpyHostToDevice));
src/gravity/potential_paris_3D.cu:  gpuFor(
src/gravity/potential_paris_3D.cu:      n, GPU_LAMBDA(const int i) { db[i] = scale * (db[i] - offset); });
src/gravity/potential_paris_3D.cu:  gpuFor(
src/gravity/potential_paris_3D.cu:      nk, nj, ni, GPU_LAMBDA(const int k, const int j, const int i) {
src/gravity/potential_paris_3D.cu:  #ifdef GRAVITY_GPU
src/gravity/potential_paris_3D.cu:  GPU_Error_Check(cudaMemcpy(potential, db, potentialBytes_, cudaMemcpyDeviceToDevice));
src/gravity/potential_paris_3D.cu:  GPU_Error_Check(cudaMemcpy(potential, db, potentialBytes_, cudaMemcpyDeviceToHost));
src/gravity/potential_paris_3D.cu:  GPU_Error_Check(cudaMalloc(reinterpret_cast<void **>(&da_), std::max(minBytes_, densityBytes_)));
src/gravity/potential_paris_3D.cu:  GPU_Error_Check(cudaMalloc(reinterpret_cast<void **>(&db_), std::max(minBytes_, potentialBytes_)));
src/gravity/potential_paris_3D.cu:    GPU_Error_Check(cudaFree(db_));
src/gravity/potential_paris_3D.cu:    GPU_Error_Check(cudaFree(da_));
src/gravity/grav3D.h:#ifdef GRAVITY_GPU
src/gravity/grav3D.h:#endif  // GRAVITY_GPU
src/gravity/grav3D.h:#ifdef GRAVITY_GPU
src/gravity/grav3D.h:#endif  // GRAVITY_GPU
src/gravity/grav3D.h:  void Copy_Isolated_Boundary_To_GPU_buffer(Real *isolated_boundary_h, Real *isolated_boundary_d, int boundary_size);
src/gravity/grav3D.h:  void Copy_Isolated_Boundaries_To_GPU(struct Parameters *P);
src/gravity/grav3D.h:#ifdef GRAVITY_GPU
src/gravity/grav3D.h:  void AllocateMemory_GPU(void);
src/gravity/grav3D.h:  void FreeMemory_GPU(void);
src/gravity/potential_SOR_3D.h:  void AllocateMemory_GPU(void);
src/gravity/potential_SOR_3D.h:  void FreeMemory_GPU(void);
src/gravity/potential_SOR_3D.h:  void Allocate_Array_GPU_Real(Real **array_dev, grav_int_t size);
src/gravity/potential_SOR_3D.h:  void Allocate_Array_GPU_bool(bool **array_dev, grav_int_t size);
src/gravity/potential_SOR_3D.h:  void Free_Array_GPU_Real(Real *array_dev);
src/gravity/potential_SOR_3D.h:  void Free_Array_GPU_bool(bool *array_dev);
src/gravity/potential_SOR_3D.h:  void Load_Transfer_Buffer_GPU(int direction, int side, int nx, int ny, int nz, int n_ghost_transfer,
src/gravity/potential_SOR_3D.h:  void Load_Transfer_Buffer_Half_GPU(int direction, int side, int nx, int ny, int nz, int n_ghost_transfer,
src/gravity/potential_SOR_3D.h:  void Load_Transfer_Buffer_GPU_x0();
src/gravity/potential_SOR_3D.h:  void Load_Transfer_Buffer_GPU_x1();
src/gravity/potential_SOR_3D.h:  void Load_Transfer_Buffer_GPU_y0();
src/gravity/potential_SOR_3D.h:  void Load_Transfer_Buffer_GPU_y1();
src/gravity/potential_SOR_3D.h:  void Load_Transfer_Buffer_GPU_z0();
src/gravity/potential_SOR_3D.h:  void Load_Transfer_Buffer_GPU_z1();
src/gravity/potential_SOR_3D.h:  void Unload_Transfer_Buffer_GPU(int direction, int side, int nx, int ny, int nz, int n_ghost_transfer,
src/gravity/potential_SOR_3D.h:  void Unload_Transfer_Buffer_Half_GPU(int direction, int side, int nx, int ny, int nz, int n_ghost_transfer,
src/gravity/potential_SOR_3D.h:  void Unload_Transfer_Buffer_GPU_x0();
src/gravity/potential_SOR_3D.h:  void Unload_Transfer_Buffer_GPU_x1();
src/gravity/potential_SOR_3D.h:  void Unload_Transfer_Buffer_GPU_y0();
src/gravity/potential_SOR_3D.h:  void Unload_Transfer_Buffer_GPU_y1();
src/gravity/potential_SOR_3D.h:  void Unload_Transfer_Buffer_GPU_z0();
src/gravity/potential_SOR_3D.h:  void Unload_Transfer_Buffer_GPU_z1();
src/gravity/potential_SOR_3D.h:  // void Load_Transfer_Buffer_GPU_All();
src/gravity/potential_SOR_3D.h:  // void Unload_Transfer_Buffer_GPU_All();
src/gravity/potential_SOR_3D.h:  void Set_Isolated_Boundary_GPU(int direction, int side, Real *boundary_d);
src/gravity/potential_paris_galactic.h:  #include "paris/PoissonZero3DBlockedGPU.hpp"
src/gravity/potential_paris_galactic.h:  PoissonZero3DBlockedGPU *pp_;
src/gravity/potential_paris_galactic.h:  #ifndef GRAVITY_GPU
src/gravity/gravity_restart.cpp:  #ifdef GRAVITY_GPU
src/gravity/gravity_restart.cpp:  GPU_Error_Check(
src/gravity/gravity_restart.cpp:      cudaMemcpy(F.potential_1_d, F.potential_1_h, n_cells_potential * sizeof(Real), cudaMemcpyHostToDevice));
src/gravity/gravity_restart.cpp:  #ifdef GRAVITY_GPU
src/gravity/gravity_restart.cpp:  GPU_Error_Check(
src/gravity/gravity_restart.cpp:      cudaMemcpy(F.potential_1_h, F.potential_1_d, n_cells_potential * sizeof(Real), cudaMemcpyDeviceToHost));
src/mhd/magnetic_divergence.h:#include "../global/global_cuda.h"
src/mhd/magnetic_divergence.h:#include "../utils/gpu.hpp"
src/mhd/ct_electric_fields.cu:  cuda_utilities::compute3DIndices(threadId, nx, ny, xid, yid, zid);
src/mhd/ct_electric_fields.cu:    signUpwind = fluxZ[cuda_utilities::compute1DIndex(xid, yid, zid - 1, nx, ny) + grid_enum::density * n_cells];
src/mhd/ct_electric_fields.cu:    signUpwind = fluxZ[cuda_utilities::compute1DIndex(xid, yid - 1, zid - 1, nx, ny) + grid_enum::density * n_cells];
src/mhd/ct_electric_fields.cu:    signUpwind = fluxY[cuda_utilities::compute1DIndex(xid, yid - 1, zid, nx, ny) + grid_enum::density * n_cells];
src/mhd/ct_electric_fields.cu:    signUpwind = fluxY[cuda_utilities::compute1DIndex(xid, yid - 1, zid - 1, nx, ny) + grid_enum::density * n_cells];
src/mhd/ct_electric_fields.cu:        +fluxZ[cuda_utilities::compute1DIndex(xid, yid, zid - 1, nx, ny) + (grid_enum::fluxZ_magnetic_x)*n_cells];
src/mhd/ct_electric_fields.cu:        +fluxZ[cuda_utilities::compute1DIndex(xid, yid - 1, zid - 1, nx, ny) + (grid_enum::fluxZ_magnetic_x)*n_cells];
src/mhd/ct_electric_fields.cu:        -fluxY[cuda_utilities::compute1DIndex(xid, yid - 1, zid, nx, ny) + (grid_enum::fluxY_magnetic_x)*n_cells];
src/mhd/ct_electric_fields.cu:        -fluxY[cuda_utilities::compute1DIndex(xid, yid - 1, zid - 1, nx, ny) + (grid_enum::fluxY_magnetic_x)*n_cells];
src/mhd/ct_electric_fields.cu:    signUpwind = fluxZ[cuda_utilities::compute1DIndex(xid, yid, zid - 1, nx, ny) + grid_enum::density * n_cells];
src/mhd/ct_electric_fields.cu:    signUpwind = fluxZ[cuda_utilities::compute1DIndex(xid - 1, yid, zid - 1, nx, ny) + grid_enum::density * n_cells];
src/mhd/ct_electric_fields.cu:    signUpwind = fluxX[cuda_utilities::compute1DIndex(xid - 1, yid, zid, nx, ny) + grid_enum::density * n_cells];
src/mhd/ct_electric_fields.cu:    signUpwind = fluxX[cuda_utilities::compute1DIndex(xid - 1, yid, zid - 1, nx, ny) + grid_enum::density * n_cells];
src/mhd/ct_electric_fields.cu:        -fluxZ[cuda_utilities::compute1DIndex(xid, yid, zid - 1, nx, ny) + (grid_enum::fluxZ_magnetic_y)*n_cells];
src/mhd/ct_electric_fields.cu:        -fluxZ[cuda_utilities::compute1DIndex(xid - 1, yid, zid - 1, nx, ny) + (grid_enum::fluxZ_magnetic_y)*n_cells];
src/mhd/ct_electric_fields.cu:        +fluxX[cuda_utilities::compute1DIndex(xid - 1, yid, zid, nx, ny) + (grid_enum::fluxX_magnetic_y)*n_cells];
src/mhd/ct_electric_fields.cu:        +fluxX[cuda_utilities::compute1DIndex(xid - 1, yid, zid - 1, nx, ny) + (grid_enum::fluxX_magnetic_y)*n_cells];
src/mhd/ct_electric_fields.cu:    signUpwind = fluxX[cuda_utilities::compute1DIndex(xid - 1, yid, zid, nx, ny) + grid_enum::density * n_cells];
src/mhd/ct_electric_fields.cu:    signUpwind = fluxX[cuda_utilities::compute1DIndex(xid - 1, yid - 1, zid, nx, ny) + grid_enum::density * n_cells];
src/mhd/ct_electric_fields.cu:    signUpwind = fluxY[cuda_utilities::compute1DIndex(xid, yid - 1, zid, nx, ny) + grid_enum::density * n_cells];
src/mhd/ct_electric_fields.cu:    signUpwind = fluxY[cuda_utilities::compute1DIndex(xid - 1, yid - 1, zid, nx, ny) + grid_enum::density * n_cells];
src/mhd/ct_electric_fields.cu:        +fluxY[cuda_utilities::compute1DIndex(xid, yid - 1, zid, nx, ny) + (grid_enum::fluxY_magnetic_z)*n_cells];
src/mhd/ct_electric_fields.cu:        +fluxY[cuda_utilities::compute1DIndex(xid - 1, yid - 1, zid, nx, ny) + (grid_enum::fluxY_magnetic_z)*n_cells];
src/mhd/ct_electric_fields.cu:        -fluxX[cuda_utilities::compute1DIndex(xid - 1, yid, zid, nx, ny) + (grid_enum::fluxX_magnetic_z)*n_cells];
src/mhd/ct_electric_fields.cu:        -fluxX[cuda_utilities::compute1DIndex(xid - 1, yid - 1, zid, nx, ny) + (grid_enum::fluxX_magnetic_z)*n_cells];
src/mhd/ct_electric_fields_tests.cu:    GPU_Error_Check(cudaMalloc(&dev_fluxX, fluxX.size() * sizeof(double)));
src/mhd/ct_electric_fields_tests.cu:    GPU_Error_Check(cudaMalloc(&dev_fluxY, fluxY.size() * sizeof(double)));
src/mhd/ct_electric_fields_tests.cu:    GPU_Error_Check(cudaMalloc(&dev_fluxZ, fluxZ.size() * sizeof(double)));
src/mhd/ct_electric_fields_tests.cu:    GPU_Error_Check(cudaMalloc(&dev_grid, grid.size() * sizeof(double)));
src/mhd/ct_electric_fields_tests.cu:    GPU_Error_Check(cudaMalloc(&dev_testCTElectricFields, testCTElectricFields.size() * sizeof(double)));
src/mhd/ct_electric_fields_tests.cu:    // Copy values to GPU
src/mhd/ct_electric_fields_tests.cu:    GPU_Error_Check(cudaMemcpy(dev_fluxX, fluxX.data(), fluxX.size() * sizeof(Real), cudaMemcpyHostToDevice));
src/mhd/ct_electric_fields_tests.cu:    GPU_Error_Check(cudaMemcpy(dev_fluxY, fluxY.data(), fluxY.size() * sizeof(Real), cudaMemcpyHostToDevice));
src/mhd/ct_electric_fields_tests.cu:    GPU_Error_Check(cudaMemcpy(dev_fluxZ, fluxZ.data(), fluxZ.size() * sizeof(Real), cudaMemcpyHostToDevice));
src/mhd/ct_electric_fields_tests.cu:    GPU_Error_Check(cudaMemcpy(dev_grid, grid.data(), grid.size() * sizeof(Real), cudaMemcpyHostToDevice));
src/mhd/ct_electric_fields_tests.cu:    GPU_Error_Check(cudaMemcpy(dev_testCTElectricFields, testCTElectricFields.data(),
src/mhd/ct_electric_fields_tests.cu:                               testCTElectricFields.size() * sizeof(Real), cudaMemcpyHostToDevice));
src/mhd/ct_electric_fields_tests.cu:    GPU_Error_Check();
src/mhd/ct_electric_fields_tests.cu:    GPU_Error_Check(cudaMemcpy(testCTElectricFields.data(), dev_testCTElectricFields,
src/mhd/ct_electric_fields_tests.cu:                               testCTElectricFields.size() * sizeof(Real), cudaMemcpyDeviceToHost));
src/mhd/ct_electric_fields_tests.cu:    cudaDeviceSynchronize();
src/mhd/magnetic_update_tests.cu:#include "../utils/cuda_utilities.h"
src/mhd/magnetic_update_tests.cu:    GPU_Error_Check(cudaMalloc(&dev_sourceGrid, sourceGrid.size() * sizeof(double)));
src/mhd/magnetic_update_tests.cu:    GPU_Error_Check(cudaMalloc(&dev_destinationGrid, destinationGrid.size() * sizeof(double)));
src/mhd/magnetic_update_tests.cu:    GPU_Error_Check(cudaMalloc(&dev_ctElectricFields, ctElectricFields.size() * sizeof(double)));
src/mhd/magnetic_update_tests.cu:    // Copy values to GPU
src/mhd/magnetic_update_tests.cu:    GPU_Error_Check(
src/mhd/magnetic_update_tests.cu:        cudaMemcpy(dev_sourceGrid, sourceGrid.data(), sourceGrid.size() * sizeof(Real), cudaMemcpyHostToDevice));
src/mhd/magnetic_update_tests.cu:    GPU_Error_Check(cudaMemcpy(dev_destinationGrid, destinationGrid.data(), destinationGrid.size() * sizeof(Real),
src/mhd/magnetic_update_tests.cu:                               cudaMemcpyHostToDevice));
src/mhd/magnetic_update_tests.cu:    GPU_Error_Check(cudaMemcpy(dev_ctElectricFields, ctElectricFields.data(), ctElectricFields.size() * sizeof(Real),
src/mhd/magnetic_update_tests.cu:                               cudaMemcpyHostToDevice));
src/mhd/magnetic_update_tests.cu:    GPU_Error_Check();
src/mhd/magnetic_update_tests.cu:    GPU_Error_Check(cudaMemcpy(destinationGrid.data(), dev_destinationGrid, destinationGrid.size() * sizeof(Real),
src/mhd/magnetic_update_tests.cu:                               cudaMemcpyDeviceToHost));
src/mhd/magnetic_update_tests.cu:    cudaDeviceSynchronize();
src/mhd/magnetic_update_tests.cu:      cuda_utilities::compute3DIndices(i, nx, ny, xid, yid, zid);
src/mhd/magnetic_divergence.cu: * integrator. Due to the CUDA/HIP compiler requiring that device functions be
src/mhd/magnetic_divergence.cu:#include "../utils/cuda_utilities.h"
src/mhd/magnetic_divergence.cu:    cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);
src/mhd/magnetic_divergence.cu:      id_xMin1 = cuda_utilities::compute1DIndex(xid - 1, yid, zid, nx, ny);
src/mhd/magnetic_divergence.cu:      id_yMin1 = cuda_utilities::compute1DIndex(xid, yid - 1, zid, nx, ny);
src/mhd/magnetic_divergence.cu:      id_zMin1 = cuda_utilities::compute1DIndex(xid, yid, zid - 1, nx, ny);
src/mhd/magnetic_divergence.cu:  cuda_utilities::AutomaticLaunchParams static const launchParams(mhd::calculateMagneticDivergence);
src/mhd/magnetic_divergence.cu:  cuda_utilities::DeviceVector<Real> static dev_maxDivergence(1);
src/mhd/magnetic_divergence.cu:  GPU_Error_Check();
src/mhd/magnetic_update.h:#include "../global/global_cuda.h"
src/mhd/magnetic_update.h:#include "../utils/gpu.hpp"
src/mhd/magnetic_update.cu:#include "../utils/cuda_utilities.h"
src/mhd/magnetic_update.cu:  cuda_utilities::compute3DIndices(threadId, nx, ny, xid, yid, zid);
src/mhd/magnetic_update.cu:        ctElectricFields[(cuda_utilities::compute1DIndex(xid, yid + 1, zid, nx, ny)) + grid_enum::ct_elec_x * n_cells];
src/mhd/magnetic_update.cu:        ctElectricFields[(cuda_utilities::compute1DIndex(xid, yid, zid + 1, nx, ny)) + grid_enum::ct_elec_x * n_cells];
src/mhd/magnetic_update.cu:    Real electric_x_3 = ctElectricFields[(cuda_utilities::compute1DIndex(xid, yid + 1, zid + 1, nx, ny)) +
src/mhd/magnetic_update.cu:        ctElectricFields[(cuda_utilities::compute1DIndex(xid + 1, yid, zid, nx, ny)) + grid_enum::ct_elec_y * n_cells];
src/mhd/magnetic_update.cu:        ctElectricFields[(cuda_utilities::compute1DIndex(xid, yid, zid + 1, nx, ny)) + grid_enum::ct_elec_y * n_cells];
src/mhd/magnetic_update.cu:    Real electric_y_3 = ctElectricFields[(cuda_utilities::compute1DIndex(xid + 1, yid, zid + 1, nx, ny)) +
src/mhd/magnetic_update.cu:        ctElectricFields[(cuda_utilities::compute1DIndex(xid + 1, yid, zid, nx, ny)) + grid_enum::ct_elec_z * n_cells];
src/mhd/magnetic_update.cu:        ctElectricFields[(cuda_utilities::compute1DIndex(xid, yid + 1, zid, nx, ny)) + grid_enum::ct_elec_z * n_cells];
src/mhd/magnetic_update.cu:    Real electric_z_3 = ctElectricFields[(cuda_utilities::compute1DIndex(xid + 1, yid + 1, zid, nx, ny)) +
src/mhd/ct_electric_fields.h:#include "../global/global_cuda.h"
src/mhd/ct_electric_fields.h:#include "../utils/cuda_utilities.h"
src/mhd/ct_electric_fields.h:#include "../utils/gpu.hpp"
src/mhd/ct_electric_fields.h:  int const idxCentered = cuda_utilities::compute1DIndex(xidCentered, yidCentered, zidCentered, nx, ny);
src/mhd/ct_electric_fields.h:  int const idxFlux = cuda_utilities::compute1DIndex(xid - int(fluxQuadrent1 == 0) - int(fluxQuadrent2 == 0),
src/mhd/ct_electric_fields.h:  int const idxB2Shift = cuda_utilities::compute1DIndex(
src/mhd/ct_electric_fields.h:  int const idxB3Shift = cuda_utilities::compute1DIndex(
src/mhd/magnetic_divergence_tests.cu:  cuda_utilities::DeviceVector<double> dev_grid(host_grid.size());
src/cooling/texture_utilities.h:#include "../utils/gpu.hpp"
src/cooling/texture_utilities.h:/* \fn float Bilinear_Texture(cudaTextureObject_t tex, float x, float y)
src/cooling/texture_utilities.h:inline __device__ float Bilinear_Texture(cudaTextureObject_t tex, float x, float y)
src/cooling/load_cloudy_texture.cu: *  \brief Wrapper file to load cloudy cooling table as CUDA texture. */
src/cooling/load_cloudy_texture.cu:  #include "../cooling/cooling_cuda.h"
src/cooling/load_cloudy_texture.cu:  #include "../global/global_cuda.h"
src/cooling/load_cloudy_texture.cu:cudaArray *cuCoolArray;
src/cooling/load_cloudy_texture.cu:cudaArray *cuHeatArray;
src/cooling/load_cloudy_texture.cu:/* \fn void Load_Cuda_Textures()
src/cooling/load_cloudy_texture.cu: * \brief Load the Cloudy cooling tables into texture memory on the GPU. */
src/cooling/load_cloudy_texture.cu:void Load_Cuda_Textures()
src/cooling/load_cloudy_texture.cu:  GPU_Error_Check(cudaHostAlloc(&cooling_table, nx * ny * sizeof(float), cudaHostAllocDefault));
src/cooling/load_cloudy_texture.cu:  GPU_Error_Check(cudaHostAlloc(&heating_table, nx * ny * sizeof(float), cudaHostAllocDefault));
src/cooling/load_cloudy_texture.cu:  // Allocate CUDA arrays in device memory
src/cooling/load_cloudy_texture.cu:  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
src/cooling/load_cloudy_texture.cu:  GPU_Error_Check(cudaMallocArray(&cuCoolArray, &channelDesc, nx, ny));
src/cooling/load_cloudy_texture.cu:  GPU_Error_Check(cudaMallocArray(&cuHeatArray, &channelDesc, nx, ny));
src/cooling/load_cloudy_texture.cu:  // cudaMemcpyToArray is being deprecated
src/cooling/load_cloudy_texture.cu:  // cudaMemcpyToArray(cuCoolArray, 0, 0, cooling_table, nx*ny*sizeof(float),
src/cooling/load_cloudy_texture.cu:  // cudaMemcpyHostToDevice); cudaMemcpyToArray(cuHeatArray, 0, 0,
src/cooling/load_cloudy_texture.cu:  // heating_table, nx*ny*sizeof(float), cudaMemcpyHostToDevice);
src/cooling/load_cloudy_texture.cu:  cudaMemcpy2DToArray(cuCoolArray, 0, 0, cooling_table, nx * sizeof(float), nx * sizeof(float), ny,
src/cooling/load_cloudy_texture.cu:                      cudaMemcpyHostToDevice);
src/cooling/load_cloudy_texture.cu:  cudaMemcpy2DToArray(cuHeatArray, 0, 0, heating_table, nx * sizeof(float), nx * sizeof(float), ny,
src/cooling/load_cloudy_texture.cu:                      cudaMemcpyHostToDevice);
src/cooling/load_cloudy_texture.cu:  struct cudaResourceDesc coolResDesc;
src/cooling/load_cloudy_texture.cu:  coolResDesc.resType         = cudaResourceTypeArray;
src/cooling/load_cloudy_texture.cu:  struct cudaResourceDesc heatResDesc;
src/cooling/load_cloudy_texture.cu:  heatResDesc.resType         = cudaResourceTypeArray;
src/cooling/load_cloudy_texture.cu:  struct cudaTextureDesc texDesc;
src/cooling/load_cloudy_texture.cu:  texDesc.addressMode[0] = cudaAddressModeClamp;  // out-of-bounds fetches return border values
src/cooling/load_cloudy_texture.cu:  texDesc.addressMode[1] = cudaAddressModeClamp;  // out-of-bounds fetches return border values
src/cooling/load_cloudy_texture.cu:  texDesc.filterMode = cudaFilterModePoint;
src/cooling/load_cloudy_texture.cu:  // cudaFilterModeLinear;
src/cooling/load_cloudy_texture.cu:  texDesc.readMode = cudaReadModeElementType;
src/cooling/load_cloudy_texture.cu:  cudaCreateTextureObject(&coolTexObj, &coolResDesc, &texDesc, NULL);
src/cooling/load_cloudy_texture.cu:  cudaCreateTextureObject(&heatTexObj, &heatResDesc, &texDesc, NULL);
src/cooling/load_cloudy_texture.cu:  GPU_Error_Check(cudaFreeHost(cooling_table));
src/cooling/load_cloudy_texture.cu:  GPU_Error_Check(cudaFreeHost(heating_table));
src/cooling/load_cloudy_texture.cu:void Free_Cuda_Textures()
src/cooling/load_cloudy_texture.cu:  // unbind the cuda textures
src/cooling/load_cloudy_texture.cu:  cudaDestroyTextureObject(coolTexObj);
src/cooling/load_cloudy_texture.cu:  cudaDestroyTextureObject(heatTexObj);
src/cooling/load_cloudy_texture.cu:  // Free the device memory associated with the cuda arrays
src/cooling/load_cloudy_texture.cu:  cudaFreeArray(cuCoolArray);
src/cooling/load_cloudy_texture.cu:  cudaFreeArray(cuHeatArray);
src/cooling/load_cloudy_texture.cu:/* Consider this function only to be used at the end of Load_Cuda_Textures when
src/cooling/load_cloudy_texture.cu:__global__ void Test_Cloudy_Textures_Kernel(int num_n, int num_T, cudaTextureObject_t coolTexObj,
src/cooling/load_cloudy_texture.cu:                                            cudaTextureObject_t heatTexObj)
src/cooling/load_cloudy_texture.cu:/* Consider this function only to be used at the end of Load_Cuda_Textures when
src/cooling/load_cloudy_texture.cu:__global__ void Test_Cloudy_Speed_Kernel(int num_n, int num_T, cudaTextureObject_t coolTexObj,
src/cooling/load_cloudy_texture.cu:                                         cudaTextureObject_t heatTexObj)
src/cooling/load_cloudy_texture.cu:/* Consider this function only to be used at the end of Load_Cuda_Textures when
src/cooling/load_cloudy_texture.cu:  GPU_Error_Check(cudaDeviceSynchronize());
src/cooling/load_cloudy_texture.cu:  GPU_Error_Check(cudaDeviceSynchronize());
src/cooling/load_cloudy_texture.cu:  GPU_Error_Check(cudaDeviceSynchronize());
src/cooling/load_cloudy_texture.h: *  \brief Wrapper file to load cloudy cooling table as CUDA texture. */
src/cooling/load_cloudy_texture.h:/* \fn void Load_Cuda_Textures()
src/cooling/load_cloudy_texture.h: * \brief Load the Cloudy cooling tables into texture memory on the GPU. */
src/cooling/load_cloudy_texture.h:void Load_Cuda_Textures();
src/cooling/load_cloudy_texture.h:/* \fn void Free_Cuda_Textures()
src/cooling/load_cloudy_texture.h: * \brief Unbind the texture memory on the GPU, and free the associated Cuda
src/cooling/load_cloudy_texture.h:void Free_Cuda_Textures();
src/cooling/cooling_cuda.cu:/*! \file cooling_cuda.cu
src/cooling/cooling_cuda.cu:#ifdef COOLING_GPU
src/cooling/cooling_cuda.cu:  #include "../cooling/cooling_cuda.h"
src/cooling/cooling_cuda.cu:  #include "../global/global_cuda.h"
src/cooling/cooling_cuda.cu:  #include "../utils/gpu.hpp"
src/cooling/cooling_cuda.cu:cudaTextureObject_t coolTexObj = 0;
src/cooling/cooling_cuda.cu:cudaTextureObject_t heatTexObj = 0;
src/cooling/cooling_cuda.cu:  GPU_Error_Check();
src/cooling/cooling_cuda.cu: n_ghost, int n_fields, Real dt, Real gamma, cudaTextureObject_t coolTexObj,
src/cooling/cooling_cuda.cu: cudaTextureObject_t heatTexObj)
src/cooling/cooling_cuda.cu:                               Real gamma, cudaTextureObject_t coolTexObj, cudaTextureObject_t heatTexObj)
src/cooling/cooling_cuda.cu:/* \fn __device__ Real Cloudy_cool(Real n, Real T, cudaTextureObject_t
src/cooling/cooling_cuda.cu: coolTexObj, cudaTextureObject_t heatTexObj)
src/cooling/cooling_cuda.cu:__device__ Real Cloudy_cool(Real n, Real T, cudaTextureObject_t coolTexObj, cudaTextureObject_t heatTexObj)
src/cooling/cooling_cuda.cu:#endif  // COOLING_GPU
src/cooling/cooling_cuda.h:/*! \file cooling_cuda.h
src/cooling/cooling_cuda.h:#ifdef COOLING_GPU
src/cooling/cooling_cuda.h:  #include "../utils/gpu.hpp"
src/cooling/cooling_cuda.h:extern cudaTextureObject_t coolTexObj;
src/cooling/cooling_cuda.h:extern cudaTextureObject_t heatTexObj;
src/cooling/cooling_cuda.h:                               Real gamma, cudaTextureObject_t coolTexObj, cudaTextureObject_t heatTexObj);
src/cooling/cooling_cuda.h:/* \fn __device__ Real Cloudy_cool(Real n, Real T, cudaTextureObject_t
src/cooling/cooling_cuda.h: coolTexObj, cudaTextureObject_t heatTexObj)
src/cooling/cooling_cuda.h:__device__ Real Cloudy_cool(Real n, Real T, cudaTextureObject_t coolTexObj, cudaTextureObject_t heatTexObj);
src/cooling/cooling_cuda.h:#endif  // COOLING_GPU
src/main.cpp:#include "utils/cuda_utilities.h"
src/main.cpp:#ifdef CHEMISTRY_GPU
src/main.cpp:    cuda_utilities::Print_GPU_Memory_Usage();
src/dust/dust_cuda.cu: * \file dust_cuda.cu
src/dust/dust_cuda.cu:  #include "../dust/dust_cuda.h"
src/dust/dust_cuda.cu:  #include "../global/global_cuda.h"
src/dust/dust_cuda.cu:  #include "../utils/cuda_utilities.h"
src/dust/dust_cuda.cu:  #include "../utils/gpu.hpp"
src/dust/dust_cuda.cu:  GPU_Error_Check();
src/dust/dust_cuda.cu:  cuda_utilities::Get_Real_Indices(n_ghost, nx, ny, nz, is, ie, js, je, ks, ke);
src/dust/dust_cuda.h: * \file dust_cuda.h
src/dust/dust_cuda.h:  #ifndef DUST_CUDA_H
src/dust/dust_cuda.h:    #define DUST_CUDA_H
src/dust/dust_cuda.h:    #include "../utils/gpu.hpp"
src/dust/dust_cuda.h:  #endif  // DUST_CUDA_H
src/dust/dust_cuda_tests.cpp: * \file dust_cuda_tests.cpp
src/dust/dust_cuda_tests.cpp:#include "../dust/dust_cuda.h"
src/dust/dust_cuda_tests.cpp:#include "../global/global_cuda.h"
src/dust/dust_cuda_tests.cpp:#include "../utils/gpu.hpp"
src/integrators/VL_1D_cuda.cu:/*! \file VL_1D_cuda.cu
src/integrators/VL_1D_cuda.cu: *  \brief Definitions of the cuda VL algorithm functions. */
src/integrators/VL_1D_cuda.cu:  #include "../global/global_cuda.h"
src/integrators/VL_1D_cuda.cu:  #include "../hydro/hydro_cuda.h"
src/integrators/VL_1D_cuda.cu:  #include "../integrators/VL_1D_cuda.h"
src/integrators/VL_1D_cuda.cu:  #include "../reconstruction/pcm_cuda.h"
src/integrators/VL_1D_cuda.cu:  #include "../reconstruction/plmc_cuda.h"
src/integrators/VL_1D_cuda.cu:  #include "../reconstruction/plmp_cuda.h"
src/integrators/VL_1D_cuda.cu:  #include "../reconstruction/ppmc_cuda.h"
src/integrators/VL_1D_cuda.cu:  #include "../reconstruction/ppmp_cuda.h"
src/integrators/VL_1D_cuda.cu:  #include "../riemann_solvers/exact_cuda.h"
src/integrators/VL_1D_cuda.cu:  #include "../riemann_solvers/hllc_cuda.h"
src/integrators/VL_1D_cuda.cu:  #include "../riemann_solvers/roe_cuda.h"
src/integrators/VL_1D_cuda.cu:  #include "../utils/gpu.hpp"
src/integrators/VL_1D_cuda.cu:void VL_Algorithm_1D_CUDA(Real *d_conserved, int nx, int x_off, int n_ghost, Real dx, Real xbound, Real dt,
src/integrators/VL_1D_cuda.cu:  // set the dimensions of the cuda grid
src/integrators/VL_1D_cuda.cu:    // allocate memory on the GPU
src/integrators/VL_1D_cuda.cu:    // GPU_Error_Check( cudaMalloc((void**)&dev_conserved,
src/integrators/VL_1D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&dev_conserved_half, n_fields * n_cells * sizeof(Real)));
src/integrators/VL_1D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Lx, n_fields * n_cells * sizeof(Real)));
src/integrators/VL_1D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Rx, n_fields * n_cells * sizeof(Real)));
src/integrators/VL_1D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&F_x, n_fields * n_cells * sizeof(Real)));
src/integrators/VL_1D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_1D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dimGrid, dimBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama,
src/integrators/VL_1D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dimGrid, dimBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0,
src/integrators/VL_1D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dimGrid, dimBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0,
src/integrators/VL_1D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_1D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_1D_cuda.cu:  hipLaunchKernelGGL(PLMC_cuda, dimGrid, dimBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, dx, dt, gama, 0,
src/integrators/VL_1D_cuda.cu:  hipLaunchKernelGGL(PLMP_cuda, dimGrid, dimBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt,
src/integrators/VL_1D_cuda.cu:  hipLaunchKernelGGL(PPMP_cuda, dimGrid, dimBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt,
src/integrators/VL_1D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_1D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dimGrid, dimBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama,
src/integrators/VL_1D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dimGrid, dimBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0,
src/integrators/VL_1D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dimGrid, dimBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0,
src/integrators/VL_1D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_1D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_1D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_1D_cuda.cu:  // free the GPU memory
src/integrators/VL_1D_cuda.cu:  cudaFree(dev_conserved);
src/integrators/VL_1D_cuda.cu:  cudaFree(dev_conserved_half);
src/integrators/VL_1D_cuda.cu:  cudaFree(Q_Lx);
src/integrators/VL_1D_cuda.cu:  cudaFree(Q_Rx);
src/integrators/VL_1D_cuda.cu:  cudaFree(F_x);
src/integrators/VL_3D_cuda.h:/*! \file VL_3D_cuda.h
src/integrators/VL_3D_cuda.h: *  \brief Declarations for the cuda version of the 3D VL algorithm. */
src/integrators/VL_3D_cuda.h:#ifndef VL_3D_CUDA_H
src/integrators/VL_3D_cuda.h:#define VL_3D_CUDA_H
src/integrators/VL_3D_cuda.h:void VL_Algorithm_3D_CUDA(Real *d_conserved, Real *d_grav_potential, int nx, int ny, int nz, int x_off, int y_off,
src/integrators/VL_3D_cuda.h:#endif  // VL_3D_CUDA_H
src/integrators/VL_2D_cuda.cu:/*! \file VL_2D_cuda.cu
src/integrators/VL_2D_cuda.cu: *  \brief Definitions of the cuda 2D VL algorithm functions. */
src/integrators/VL_2D_cuda.cu:  #include "../global/global_cuda.h"
src/integrators/VL_2D_cuda.cu:  #include "../hydro/hydro_cuda.h"
src/integrators/VL_2D_cuda.cu:  #include "../integrators/VL_2D_cuda.h"
src/integrators/VL_2D_cuda.cu:  #include "../reconstruction/pcm_cuda.h"
src/integrators/VL_2D_cuda.cu:  #include "../reconstruction/plmc_cuda.h"
src/integrators/VL_2D_cuda.cu:  #include "../reconstruction/plmp_cuda.h"
src/integrators/VL_2D_cuda.cu:  #include "../reconstruction/ppmc_cuda.h"
src/integrators/VL_2D_cuda.cu:  #include "../reconstruction/ppmp_cuda.h"
src/integrators/VL_2D_cuda.cu:  #include "../riemann_solvers/exact_cuda.h"
src/integrators/VL_2D_cuda.cu:  #include "../riemann_solvers/hllc_cuda.h"
src/integrators/VL_2D_cuda.cu:  #include "../riemann_solvers/roe_cuda.h"
src/integrators/VL_2D_cuda.cu:  #include "../utils/gpu.hpp"
src/integrators/VL_2D_cuda.cu:void VL_Algorithm_2D_CUDA(Real *d_conserved, int nx, int ny, int x_off, int y_off, int n_ghost, Real dx, Real dy,
src/integrators/VL_2D_cuda.cu:  // set values for GPU kernels
src/integrators/VL_2D_cuda.cu:    // allocate GPU arrays
src/integrators/VL_2D_cuda.cu:    // GPU_Error_Check( cudaMalloc((void**)&dev_conserved,
src/integrators/VL_2D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&dev_conserved_half, n_fields * n_cells * sizeof(Real)));
src/integrators/VL_2D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Lx, n_fields * n_cells * sizeof(Real)));
src/integrators/VL_2D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Rx, n_fields * n_cells * sizeof(Real)));
src/integrators/VL_2D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Ly, n_fields * n_cells * sizeof(Real)));
src/integrators/VL_2D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Ry, n_fields * n_cells * sizeof(Real)));
src/integrators/VL_2D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&F_x, n_fields * n_cells * sizeof(Real)));
src/integrators/VL_2D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&F_y, n_fields * n_cells * sizeof(Real)));
src/integrators/VL_2D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost,
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost,
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama,
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama,
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost,
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost,
src/integrators/VL_2D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_2D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(PLMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx,
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(PLMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy,
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(PLMC_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, dx, dt, gama,
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(PLMC_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx, ny, nz, dy, dt, gama,
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(PPMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx,
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(PPMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy,
src/integrators/VL_2D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost,
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost,
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama,
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama,
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost,
src/integrators/VL_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost,
src/integrators/VL_2D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_2D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_2D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_2D_cuda.cu:  // free the GPU memory
src/integrators/VL_2D_cuda.cu:  cudaFree(dev_conserved);
src/integrators/VL_2D_cuda.cu:  cudaFree(dev_conserved_half);
src/integrators/VL_2D_cuda.cu:  cudaFree(Q_Lx);
src/integrators/VL_2D_cuda.cu:  cudaFree(Q_Rx);
src/integrators/VL_2D_cuda.cu:  cudaFree(Q_Ly);
src/integrators/VL_2D_cuda.cu:  cudaFree(Q_Ry);
src/integrators/VL_2D_cuda.cu:  cudaFree(F_x);
src/integrators/VL_2D_cuda.cu:  cudaFree(F_y);
src/integrators/VL_3D_cuda.cu:/*! \file VL_3D_cuda.cu
src/integrators/VL_3D_cuda.cu: *  \brief Definitions of the cuda 3 D VL algorithm functions. MHD algorithm
src/integrators/VL_3D_cuda.cu:  #include "../global/global_cuda.h"
src/integrators/VL_3D_cuda.cu:  #include "../hydro/hydro_cuda.h"
src/integrators/VL_3D_cuda.cu:  #include "../integrators/VL_3D_cuda.h"
src/integrators/VL_3D_cuda.cu:  #include "../reconstruction/pcm_cuda.h"
src/integrators/VL_3D_cuda.cu:  #include "../reconstruction/plmc_cuda.h"
src/integrators/VL_3D_cuda.cu:  #include "../reconstruction/plmp_cuda.h"
src/integrators/VL_3D_cuda.cu:  #include "../reconstruction/ppmc_cuda.h"
src/integrators/VL_3D_cuda.cu:  #include "../reconstruction/ppmp_cuda.h"
src/integrators/VL_3D_cuda.cu:  #include "../riemann_solvers/exact_cuda.h"
src/integrators/VL_3D_cuda.cu:  #include "../riemann_solvers/hll_cuda.h"
src/integrators/VL_3D_cuda.cu:  #include "../riemann_solvers/hllc_cuda.h"
src/integrators/VL_3D_cuda.cu:  #include "../riemann_solvers/hlld_cuda.h"
src/integrators/VL_3D_cuda.cu:  #include "../riemann_solvers/roe_cuda.h"
src/integrators/VL_3D_cuda.cu:  #include "../utils/gpu.hpp"
src/integrators/VL_3D_cuda.cu:void VL_Algorithm_3D_CUDA(Real *d_conserved, Real *d_grav_potential, int nx, int ny, int nz, int x_off, int y_off,
src/integrators/VL_3D_cuda.cu:  // set values for GPU kernels
src/integrators/VL_3D_cuda.cu:    // allocate memory on the GPU
src/integrators/VL_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&dev_conserved_half, n_fields * n_cells * sizeof(Real)));
src/integrators/VL_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Lx, arraySize));
src/integrators/VL_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Rx, arraySize));
src/integrators/VL_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Ly, arraySize));
src/integrators/VL_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Ry, arraySize));
src/integrators/VL_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Lz, arraySize));
src/integrators/VL_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Rz, arraySize));
src/integrators/VL_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&F_x, arraySize));
src/integrators/VL_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&F_y, arraySize));
src/integrators/VL_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&F_z, arraySize));
src/integrators/VL_3D_cuda.cu:    cuda_utilities::initGpuMemory(dev_conserved_half, n_fields * n_cells * sizeof(Real));
src/integrators/VL_3D_cuda.cu:    cuda_utilities::initGpuMemory(Q_Lx, arraySize);
src/integrators/VL_3D_cuda.cu:    cuda_utilities::initGpuMemory(Q_Rx, arraySize);
src/integrators/VL_3D_cuda.cu:    cuda_utilities::initGpuMemory(Q_Ly, arraySize);
src/integrators/VL_3D_cuda.cu:    cuda_utilities::initGpuMemory(Q_Ry, arraySize);
src/integrators/VL_3D_cuda.cu:    cuda_utilities::initGpuMemory(Q_Lz, arraySize);
src/integrators/VL_3D_cuda.cu:    cuda_utilities::initGpuMemory(Q_Rz, arraySize);
src/integrators/VL_3D_cuda.cu:    cuda_utilities::initGpuMemory(F_x, arraySize);
src/integrators/VL_3D_cuda.cu:    cuda_utilities::initGpuMemory(F_y, arraySize);
src/integrators/VL_3D_cuda.cu:    cuda_utilities::initGpuMemory(F_z, arraySize);
src/integrators/VL_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&ctElectricFields, ctArraySize));
src/integrators/VL_3D_cuda.cu:  #if defined(GRAVITY) && !defined(GRAVITY_GPU)
src/integrators/VL_3D_cuda.cu:  GPU_Error_Check(cudaMemcpy(dev_grav_potential, temp_potential, n_cells * sizeof(Real), cudaMemcpyHostToDevice));
src/integrators/VL_3D_cuda.cu:  #endif  // GRAVITY and GRAVITY_GPU
src/integrators/VL_3D_cuda.cu:  cuda_utilities::AutomaticLaunchParams static const pcm_launch_params(PCM_Reconstruction_3D, n_cells);
src/integrators/VL_3D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_3D_cuda.cu:  cuda_utilities::AutomaticLaunchParams static const exact_launch_params(Calculate_Exact_Fluxes_CUDA,
src/integrators/VL_3D_cuda.cu:                                                                         n_cellsCalculate_Exact_Fluxes_CUDA);
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, exact_launch_params.numBlocks, exact_launch_params.threadsPerBlock, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, exact_launch_params.numBlocks, exact_launch_params.threadsPerBlock, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, exact_launch_params.numBlocks, exact_launch_params.threadsPerBlock, 0,
src/integrators/VL_3D_cuda.cu:  cuda_utilities::AutomaticLaunchParams static const roe_launch_params(Calculate_Roe_Fluxes_CUDA, n_cells);
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, roe_launch_params.numBlocks, roe_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, roe_launch_params.numBlocks, roe_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, roe_launch_params.numBlocks, roe_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  cuda_utilities::AutomaticLaunchParams static const hllc_launch_params(Calculate_HLLC_Fluxes_CUDA, n_cells);
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, hllc_launch_params.numBlocks, hllc_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, hllc_launch_params.numBlocks, hllc_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, hllc_launch_params.numBlocks, hllc_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  cuda_utilities::AutomaticLaunchParams static const hll_launch_params(Calculate_HLL_Fluxes_CUDA, n_cells);
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, hll_launch_params.numBlocks, hll_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, hll_launch_params.numBlocks, hll_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, hll_launch_params.numBlocks, hll_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  cuda_utilities::AutomaticLaunchParams static const hlld_launch_params(mhd::Calculate_HLLD_Fluxes_CUDA, n_cells);
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(mhd::Calculate_HLLD_Fluxes_CUDA, hlld_launch_params.numBlocks, hlld_launch_params.threadsPerBlock,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(mhd::Calculate_HLLD_Fluxes_CUDA, hlld_launch_params.numBlocks, hlld_launch_params.threadsPerBlock,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(mhd::Calculate_HLLD_Fluxes_CUDA, hlld_launch_params.numBlocks, hlld_launch_params.threadsPerBlock,
src/integrators/VL_3D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_3D_cuda.cu:  cuda_utilities::AutomaticLaunchParams static const ct_launch_params(mhd::Calculate_CT_Electric_Fields, n_cells);
src/integrators/VL_3D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_3D_cuda.cu:  cuda_utilities::AutomaticLaunchParams static const update_half_launch_params(Update_Conserved_Variables_3D_half,
src/integrators/VL_3D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_3D_cuda.cu:  cuda_utilities::AutomaticLaunchParams static const update_magnetic_launch_params(mhd::Update_Magnetic_Field_3D,
src/integrators/VL_3D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_3D_cuda.cu:  cuda_utilities::AutomaticLaunchParams static const plmp_launch_params(PLMP_cuda, n_cells);
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(PLMP_cuda, plmp_launch_params.numBlocks, plmp_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(PLMP_cuda, plmp_launch_params.numBlocks, plmp_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(PLMP_cuda, plmp_launch_params.numBlocks, plmp_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  cuda_utilities::AutomaticLaunchParams static const plmc_vl_launch_params(PLMC_cuda, n_cells);
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(PLMC_cuda, plmc_vl_launch_params.numBlocks, plmc_vl_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(PLMC_cuda, plmc_vl_launch_params.numBlocks, plmc_vl_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(PLMC_cuda, plmc_vl_launch_params.numBlocks, plmc_vl_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  cuda_utilities::AutomaticLaunchParams static const ppmp_launch_params(PPMP_cuda, n_cells);
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(PPMP_cuda, ppmp_launch_params.numBlocks, ppmp_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(PPMP_cuda, ppmp_launch_params.numBlocks, ppmp_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(PPMP_cuda, ppmp_launch_params.numBlocks, ppmp_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  cuda_utilities::AutomaticLaunchParams static const ppmc_vl_launch_params(PPMC_VL, n_cells);
src/integrators/VL_3D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, exact_launch_params.numBlocks, exact_launch_params.threadsPerBlock, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, exact_launch_params.numBlocks, exact_launch_params.threadsPerBlock, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, exact_launch_params.numBlocks, exact_launch_params.threadsPerBlock, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, roe_launch_params.numBlocks, roe_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, roe_launch_params.numBlocks, roe_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, roe_launch_params.numBlocks, roe_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, hllc_launch_params.numBlocks, hllc_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, hllc_launch_params.numBlocks, hllc_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, hllc_launch_params.numBlocks, hllc_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, hll_launch_params.numBlocks, hll_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, hll_launch_params.numBlocks, hll_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, hll_launch_params.numBlocks, hll_launch_params.threadsPerBlock, 0, 0,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(mhd::Calculate_HLLD_Fluxes_CUDA, hlld_launch_params.numBlocks, hlld_launch_params.threadsPerBlock,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(mhd::Calculate_HLLD_Fluxes_CUDA, hlld_launch_params.numBlocks, hlld_launch_params.threadsPerBlock,
src/integrators/VL_3D_cuda.cu:  hipLaunchKernelGGL(mhd::Calculate_HLLD_Fluxes_CUDA, hlld_launch_params.numBlocks, hlld_launch_params.threadsPerBlock,
src/integrators/VL_3D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_3D_cuda.cu:  cuda_utilities::AutomaticLaunchParams static const de_advect_launch_params(Partial_Update_Advected_Internal_Energy_3D,
src/integrators/VL_3D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_3D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_3D_cuda.cu:  cuda_utilities::AutomaticLaunchParams static const update_full_launch_params(Update_Conserved_Variables_3D, n_cells);
src/integrators/VL_3D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_3D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_3D_cuda.cu:  cuda_utilities::AutomaticLaunchParams static const de_select_launch_params(Select_Internal_Energy_3D, n_cells);
src/integrators/VL_3D_cuda.cu:  cuda_utilities::AutomaticLaunchParams static const de_sync_launch_params(Sync_Energies_3D, n_cells);
src/integrators/VL_3D_cuda.cu:  GPU_Error_Check();
src/integrators/VL_3D_cuda.cu:  // free the GPU memory
src/integrators/VL_3D_cuda.cu:  cudaFree(dev_conserved);
src/integrators/VL_3D_cuda.cu:  cudaFree(dev_conserved_half);
src/integrators/VL_3D_cuda.cu:  cudaFree(Q_Lx);
src/integrators/VL_3D_cuda.cu:  cudaFree(Q_Rx);
src/integrators/VL_3D_cuda.cu:  cudaFree(Q_Ly);
src/integrators/VL_3D_cuda.cu:  cudaFree(Q_Ry);
src/integrators/VL_3D_cuda.cu:  cudaFree(Q_Lz);
src/integrators/VL_3D_cuda.cu:  cudaFree(Q_Rz);
src/integrators/VL_3D_cuda.cu:  cudaFree(F_x);
src/integrators/VL_3D_cuda.cu:  cudaFree(F_y);
src/integrators/VL_3D_cuda.cu:  cudaFree(F_z);
src/integrators/VL_3D_cuda.cu:  cudaFree(ctElectricFields);
src/integrators/VL_2D_cuda.h:/*! \file VL_2D_cuda.h
src/integrators/VL_2D_cuda.h: *  \brief Declarations for the cuda version of the 2D VL algorithm. */
src/integrators/VL_2D_cuda.h:#ifndef VL_2D_CUDA_H
src/integrators/VL_2D_cuda.h:#define VL_2D_CUDA_H
src/integrators/VL_2D_cuda.h:void VL_Algorithm_2D_CUDA(Real *d_conserved, int nx, int ny, int x_off, int y_off, int n_ghost, Real dx, Real dy,
src/integrators/VL_2D_cuda.h:#endif  // VL_2D_CUDA_H
src/integrators/simple_1D_cuda.h:/*! \file simple_1D_cuda.h
src/integrators/simple_1D_cuda.h:#ifndef SIMPLE_1D_CUDA_H
src/integrators/simple_1D_cuda.h:#define SIMPLE_1D_CUDA_H
src/integrators/simple_1D_cuda.h:void Simple_Algorithm_1D_CUDA(Real *d_conserved, int nx, int x_off, int n_ghost, Real dx, Real xbound, Real dt,
src/integrators/simple_1D_cuda.h:#endif  // Simple_1D_CUDA_H
src/integrators/simple_3D_cuda.h:/*! \file simple_3D_cuda.h
src/integrators/simple_3D_cuda.h: *  \brief Declarations for the cuda version of the 3D simple algorithm. */
src/integrators/simple_3D_cuda.h:#ifndef SIMPLE_3D_CUDA_H
src/integrators/simple_3D_cuda.h:#define SIMPLE_3D_CUDA_H
src/integrators/simple_3D_cuda.h:#include "../chemistry_gpu/chemistry_gpu.h"
src/integrators/simple_3D_cuda.h:void Simple_Algorithm_3D_CUDA(Real *d_conserved, Real *d_grav_potential, int nx, int ny, int nz, int x_off, int y_off,
src/integrators/simple_3D_cuda.h:#endif  // SIMPLE_3D_CUDA_H
src/integrators/simple_3D_cuda.cu:/*! \file simple_3D_cuda.cu
src/integrators/simple_3D_cuda.cu: *  \brief Definitions of the cuda 3D simple algorithm functions. */
src/integrators/simple_3D_cuda.cu:  #include "../global/global_cuda.h"
src/integrators/simple_3D_cuda.cu:  #include "../hydro/hydro_cuda.h"
src/integrators/simple_3D_cuda.cu:  #include "../integrators/simple_3D_cuda.h"
src/integrators/simple_3D_cuda.cu:  #include "../reconstruction/pcm_cuda.h"
src/integrators/simple_3D_cuda.cu:  #include "../reconstruction/plmc_cuda.h"
src/integrators/simple_3D_cuda.cu:  #include "../reconstruction/plmp_cuda.h"
src/integrators/simple_3D_cuda.cu:  #include "../reconstruction/ppmc_cuda.h"
src/integrators/simple_3D_cuda.cu:  #include "../reconstruction/ppmp_cuda.h"
src/integrators/simple_3D_cuda.cu:  #include "../riemann_solvers/exact_cuda.h"
src/integrators/simple_3D_cuda.cu:  #include "../riemann_solvers/hll_cuda.h"
src/integrators/simple_3D_cuda.cu:  #include "../riemann_solvers/hllc_cuda.h"
src/integrators/simple_3D_cuda.cu:  #include "../riemann_solvers/roe_cuda.h"
src/integrators/simple_3D_cuda.cu:  #include "../utils/gpu.hpp"
src/integrators/simple_3D_cuda.cu:void Simple_Algorithm_3D_CUDA(Real *d_conserved, Real *d_grav_potential, int nx, int ny, int nz, int x_off, int y_off,
src/integrators/simple_3D_cuda.cu:  // set values for GPU kernels
src/integrators/simple_3D_cuda.cu:    GPU_Error_Check(cudaMemGetInfo(&global_free, &global_total));
src/integrators/simple_3D_cuda.cu:    // allocate memory on the GPU
src/integrators/simple_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Lx, n_fields * n_cells * sizeof(Real)));
src/integrators/simple_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Rx, n_fields * n_cells * sizeof(Real)));
src/integrators/simple_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Ly, n_fields * n_cells * sizeof(Real)));
src/integrators/simple_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Ry, n_fields * n_cells * sizeof(Real)));
src/integrators/simple_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Lz, n_fields * n_cells * sizeof(Real)));
src/integrators/simple_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Rz, n_fields * n_cells * sizeof(Real)));
src/integrators/simple_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&F_x, n_fields * n_cells * sizeof(Real)));
src/integrators/simple_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&F_y, n_fields * n_cells * sizeof(Real)));
src/integrators/simple_3D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&F_z, n_fields * n_cells * sizeof(Real)));
src/integrators/simple_3D_cuda.cu:    // GPU_Error_Check( cudaMalloc((void**)&dev_grav_potential,
src/integrators/simple_3D_cuda.cu:  #if defined(GRAVITY) && !defined(GRAVITY_GPU)
src/integrators/simple_3D_cuda.cu:  GPU_Error_Check(cudaMemcpy(dev_grav_potential, temp_potential, n_cells * sizeof(Real), cudaMemcpyHostToDevice));
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(PLMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(PLMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy, dt,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(PLMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lz, Q_Rz, nx, ny, nz, n_ghost, dz, dt,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(PLMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, dx, dt, gama, 0,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(PLMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, dy, dt, gama, 1,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(PLMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lz, Q_Rz, nx, ny, nz, dz, dt, gama, 2,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(PPMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(PPMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy, dt,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(PPMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lz, Q_Rz, nx, ny, nz, n_ghost, dz, dt,
src/integrators/simple_3D_cuda.cu:  GPU_Error_Check();
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost, gama,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama,
src/integrators/simple_3D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost, gama,
src/integrators/simple_3D_cuda.cu:  GPU_Error_Check();
src/integrators/simple_3D_cuda.cu:  GPU_Error_Check();
src/integrators/simple_3D_cuda.cu:  GPU_Error_Check();
src/integrators/simple_3D_cuda.cu:  GPU_Error_Check();
src/integrators/simple_3D_cuda.cu:  // free the GPU memory
src/integrators/simple_3D_cuda.cu:  cudaFree(dev_conserved);
src/integrators/simple_3D_cuda.cu:  cudaFree(Q_Lx);
src/integrators/simple_3D_cuda.cu:  cudaFree(Q_Rx);
src/integrators/simple_3D_cuda.cu:  cudaFree(Q_Ly);
src/integrators/simple_3D_cuda.cu:  cudaFree(Q_Ry);
src/integrators/simple_3D_cuda.cu:  cudaFree(Q_Lz);
src/integrators/simple_3D_cuda.cu:  cudaFree(Q_Rz);
src/integrators/simple_3D_cuda.cu:  cudaFree(F_x);
src/integrators/simple_3D_cuda.cu:  cudaFree(F_y);
src/integrators/simple_3D_cuda.cu:  cudaFree(F_z);
src/integrators/simple_2D_cuda.cu:/*! \file simple_2D_cuda.cu
src/integrators/simple_2D_cuda.cu: *  \brief Definitions of the cuda 2D simple algorithm functions. */
src/integrators/simple_2D_cuda.cu:#include "../global/global_cuda.h"
src/integrators/simple_2D_cuda.cu:#include "../hydro/hydro_cuda.h"
src/integrators/simple_2D_cuda.cu:#include "../integrators/simple_2D_cuda.h"
src/integrators/simple_2D_cuda.cu:#include "../reconstruction/pcm_cuda.h"
src/integrators/simple_2D_cuda.cu:#include "../reconstruction/plmc_cuda.h"
src/integrators/simple_2D_cuda.cu:#include "../reconstruction/plmp_cuda.h"
src/integrators/simple_2D_cuda.cu:#include "../reconstruction/ppmc_cuda.h"
src/integrators/simple_2D_cuda.cu:#include "../reconstruction/ppmp_cuda.h"
src/integrators/simple_2D_cuda.cu:#include "../riemann_solvers/exact_cuda.h"
src/integrators/simple_2D_cuda.cu:#include "../riemann_solvers/hllc_cuda.h"
src/integrators/simple_2D_cuda.cu:#include "../riemann_solvers/roe_cuda.h"
src/integrators/simple_2D_cuda.cu:#include "../utils/gpu.hpp"
src/integrators/simple_2D_cuda.cu:void Simple_Algorithm_2D_CUDA(Real *d_conserved, int nx, int ny, int x_off, int y_off, int n_ghost, Real dx, Real dy,
src/integrators/simple_2D_cuda.cu:  // set values for GPU kernels
src/integrators/simple_2D_cuda.cu:    // allocate memory on the GPU
src/integrators/simple_2D_cuda.cu:    // GPU_Error_Check( cudaMalloc((void**)&dev_conserved,
src/integrators/simple_2D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Lx, n_fields * n_cells * sizeof(Real)));
src/integrators/simple_2D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Rx, n_fields * n_cells * sizeof(Real)));
src/integrators/simple_2D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Ly, n_fields * n_cells * sizeof(Real)));
src/integrators/simple_2D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Ry, n_fields * n_cells * sizeof(Real)));
src/integrators/simple_2D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&F_x, n_fields * n_cells * sizeof(Real)));
src/integrators/simple_2D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&F_y, n_fields * n_cells * sizeof(Real)));
src/integrators/simple_2D_cuda.cu:  hipLaunchKernelGGL(PLMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt,
src/integrators/simple_2D_cuda.cu:  hipLaunchKernelGGL(PLMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy, dt,
src/integrators/simple_2D_cuda.cu:  hipLaunchKernelGGL(PLMC_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, dx, dt, gama, 0,
src/integrators/simple_2D_cuda.cu:  hipLaunchKernelGGL(PLMC_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, dy, dt, gama, 1,
src/integrators/simple_2D_cuda.cu:  hipLaunchKernelGGL(PPMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt,
src/integrators/simple_2D_cuda.cu:  hipLaunchKernelGGL(PPMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy, dt,
src/integrators/simple_2D_cuda.cu:  GPU_Error_Check();
src/integrators/simple_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost,
src/integrators/simple_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost,
src/integrators/simple_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama,
src/integrators/simple_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama,
src/integrators/simple_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost,
src/integrators/simple_2D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost,
src/integrators/simple_2D_cuda.cu:  GPU_Error_Check();
src/integrators/simple_2D_cuda.cu:  GPU_Error_Check();
src/integrators/simple_2D_cuda.cu:  GPU_Error_Check();
src/integrators/simple_2D_cuda.cu:  // free the GPU memory
src/integrators/simple_2D_cuda.cu:  cudaFree(dev_conserved);
src/integrators/simple_2D_cuda.cu:  cudaFree(Q_Lx);
src/integrators/simple_2D_cuda.cu:  cudaFree(Q_Rx);
src/integrators/simple_2D_cuda.cu:  cudaFree(Q_Ly);
src/integrators/simple_2D_cuda.cu:  cudaFree(Q_Ry);
src/integrators/simple_2D_cuda.cu:  cudaFree(F_x);
src/integrators/simple_2D_cuda.cu:  cudaFree(F_y);
src/integrators/simple_2D_cuda.h:/*! \file simple_2D_cuda.h
src/integrators/simple_2D_cuda.h: *  \brief Declarations for the cuda version of the 2D simple algorithm. */
src/integrators/simple_2D_cuda.h:#ifndef SIMPLE_2D_CUDA_H
src/integrators/simple_2D_cuda.h:#define SIMPLE_2D_CUDA_H
src/integrators/simple_2D_cuda.h:void Simple_Algorithm_2D_CUDA(Real *d_conserved, int nx, int ny, int x_off, int y_off, int n_ghost, Real dx, Real dy,
src/integrators/simple_2D_cuda.h:#endif  // SIMPLE_2D_CUDA_H
src/integrators/simple_1D_cuda.cu:/*! \file simple_1D_cuda.cu
src/integrators/simple_1D_cuda.cu:#include "../global/global_cuda.h"
src/integrators/simple_1D_cuda.cu:#include "../hydro/hydro_cuda.h"
src/integrators/simple_1D_cuda.cu:#include "../integrators/simple_1D_cuda.h"
src/integrators/simple_1D_cuda.cu:#include "../reconstruction/pcm_cuda.h"
src/integrators/simple_1D_cuda.cu:#include "../reconstruction/plmc_cuda.h"
src/integrators/simple_1D_cuda.cu:#include "../reconstruction/plmp_cuda.h"
src/integrators/simple_1D_cuda.cu:#include "../reconstruction/ppmc_cuda.h"
src/integrators/simple_1D_cuda.cu:#include "../reconstruction/ppmp_cuda.h"
src/integrators/simple_1D_cuda.cu:#include "../riemann_solvers/exact_cuda.h"
src/integrators/simple_1D_cuda.cu:#include "../riemann_solvers/hllc_cuda.h"
src/integrators/simple_1D_cuda.cu:#include "../riemann_solvers/roe_cuda.h"
src/integrators/simple_1D_cuda.cu:#include "../utils/gpu.hpp"
src/integrators/simple_1D_cuda.cu:void Simple_Algorithm_1D_CUDA(Real *d_conserved, int nx, int x_off, int n_ghost, Real dx, Real xbound, Real dt,
src/integrators/simple_1D_cuda.cu:  // set the dimensions of the cuda grid
src/integrators/simple_1D_cuda.cu:    // allocate memory on the GPU
src/integrators/simple_1D_cuda.cu:    // GPU_Error_Check( cudaMalloc((void**)&dev_conserved,
src/integrators/simple_1D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Lx, n_fields * n_cells * sizeof(Real)));
src/integrators/simple_1D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&Q_Rx, n_fields * n_cells * sizeof(Real)));
src/integrators/simple_1D_cuda.cu:    GPU_Error_Check(cudaMalloc((void **)&F_x, (n_fields)*n_cells * sizeof(Real)));
src/integrators/simple_1D_cuda.cu:  GPU_Error_Check();
src/integrators/simple_1D_cuda.cu:  hipLaunchKernelGGL(PLMP_cuda, dimGrid, dimBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama,
src/integrators/simple_1D_cuda.cu:  GPU_Error_Check();
src/integrators/simple_1D_cuda.cu:  hipLaunchKernelGGL(PLMC_cuda, dimGrid, dimBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, dx, dt, gama, 0,
src/integrators/simple_1D_cuda.cu:  GPU_Error_Check();
src/integrators/simple_1D_cuda.cu:  hipLaunchKernelGGL(PPMP_cuda, dimGrid, dimBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama,
src/integrators/simple_1D_cuda.cu:  GPU_Error_Check();
src/integrators/simple_1D_cuda.cu:  GPU_Error_Check();
src/integrators/simple_1D_cuda.cu:  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dimGrid, dimBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama,
src/integrators/simple_1D_cuda.cu:  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dimGrid, dimBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0,
src/integrators/simple_1D_cuda.cu:  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dimGrid, dimBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0,
src/integrators/simple_1D_cuda.cu:  GPU_Error_Check();
src/integrators/simple_1D_cuda.cu:  GPU_Error_Check();
src/integrators/simple_1D_cuda.cu:  GPU_Error_Check();
src/integrators/simple_1D_cuda.cu:  // free the GPU memory
src/integrators/simple_1D_cuda.cu:  cudaFree(dev_conserved);
src/integrators/simple_1D_cuda.cu:  cudaFree(Q_Lx);
src/integrators/simple_1D_cuda.cu:  cudaFree(Q_Rx);
src/integrators/simple_1D_cuda.cu:  cudaFree(F_x);
src/integrators/VL_1D_cuda.h:/*! \file VL_1D_cuda.h
src/integrators/VL_1D_cuda.h: *  \brief Declarations for the cuda version of the 1D VL algorithm. */
src/integrators/VL_1D_cuda.h:#ifndef VL_1D_CUDA_H
src/integrators/VL_1D_cuda.h:#define VL_1D_CUDA_H
src/integrators/VL_1D_cuda.h:void VL_Algorithm_1D_CUDA(Real *d_conserved, int nx, int x_off, int n_ghost, Real dx, Real xbound, Real dt,
src/integrators/VL_1D_cuda.h:#endif  // VL_1D_CUDA_H
src/riemann_solvers/hllc_cuda.cu:/*! \file hllc_cuda.cu
src/riemann_solvers/hllc_cuda.cu: *  \brief Function definitions for the cuda HLLC Riemann solver.*/
src/riemann_solvers/hllc_cuda.cu:#include "../global/global_cuda.h"
src/riemann_solvers/hllc_cuda.cu:#include "../riemann_solvers/hllc_cuda.h"
src/riemann_solvers/hllc_cuda.cu:#include "../utils/gpu.hpp"
src/riemann_solvers/hllc_cuda.cu:/*! \fn Calculate_HLLC_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real
src/riemann_solvers/hllc_cuda.cu:__global__ void Calculate_HLLC_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny,
src/riemann_solvers/hllc_cuda.cu:    Sl = sgn_CUDA(Sl) * fmax(fabs(Sl), etah);
src/riemann_solvers/hllc_cuda.cu:    Sr = sgn_CUDA(Sr) * fmax(fabs(Sr), etah);
src/riemann_solvers/hllc_cuda.h:/*! \file hllc_cuda.h
src/riemann_solvers/hllc_cuda.h: *  \brief Declarations of functions for the cuda hllc riemann solver kernel. */
src/riemann_solvers/hllc_cuda.h:#ifndef HLLC_CUDA_H
src/riemann_solvers/hllc_cuda.h:#define HLLC_CUDA_H
src/riemann_solvers/hllc_cuda.h:/*! \fn Calculate_HLLC_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real
src/riemann_solvers/hllc_cuda.h:__global__ void Calculate_HLLC_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny,
src/riemann_solvers/hllc_cuda.h:#endif  // HLLC_CUDA_H
src/riemann_solvers/hll_cuda.h:/*! \file hllc_cuda.h
src/riemann_solvers/hll_cuda.h: *  \brief Declarations of functions for the cuda hllc riemann solver kernel. */
src/riemann_solvers/hll_cuda.h:#ifndef HLL_CUDA_H
src/riemann_solvers/hll_cuda.h:#define HLL_CUDA_H
src/riemann_solvers/hll_cuda.h:/*! \fn Calculate_HLLC_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real
src/riemann_solvers/hll_cuda.h:__global__ void Calculate_HLL_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny,
src/riemann_solvers/hll_cuda.h:#endif  // HLLC_CUDA_H
src/riemann_solvers/exact_cuda.h:/*! \file exact_cuda.h
src/riemann_solvers/exact_cuda.h: *  \brief Declarations of functions for the cuda exact riemann solver kernel.
src/riemann_solvers/exact_cuda.h:#ifndef EXACT_CUDA_H
src/riemann_solvers/exact_cuda.h:#define EXACT_CUDA_H
src/riemann_solvers/exact_cuda.h:/*! \fn Calculate_Exact_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real
src/riemann_solvers/exact_cuda.h:__global__ void Calculate_Exact_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny,
src/riemann_solvers/exact_cuda.h:__device__ Real guessp_CUDA(Real dl, Real vxl, Real pl, Real cl, Real dr, Real vxr, Real pr, Real cr, Real gamma);
src/riemann_solvers/exact_cuda.h:__device__ void prefun_CUDA(Real *f, Real *fd, Real p, Real dk, Real pk, Real ck, Real gamma);
src/riemann_solvers/exact_cuda.h:__device__ void starpv_CUDA(Real *p, Real *v, Real dl, Real vxl, Real pl, Real cl, Real dr, Real vxr, Real pr, Real cr,
src/riemann_solvers/exact_cuda.h:__device__ void sample_CUDA(const Real pm, const Real vm, Real *d, Real *v, Real *p, Real dl, Real vxl, Real pl,
src/riemann_solvers/exact_cuda.h:#endif  // EXACT_CUDA_H
src/riemann_solvers/exact_cuda.cu:/*! \file exact_cuda.cu
src/riemann_solvers/exact_cuda.cu: *  \brief Function definitions for the cuda exact Riemann solver.*/
src/riemann_solvers/exact_cuda.cu:#include "../global/global_cuda.h"
src/riemann_solvers/exact_cuda.cu:#include "../riemann_solvers/exact_cuda.h"
src/riemann_solvers/exact_cuda.cu:#include "../utils/gpu.hpp"
src/riemann_solvers/exact_cuda.cu:/*! \fn Calculate_Exact_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real
src/riemann_solvers/exact_cuda.cu:__global__ void Calculate_Exact_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny,
src/riemann_solvers/exact_cuda.cu:  Real ds, vs, ps, Es;  // sample_CUDAd density, velocity, pressure, total
src/riemann_solvers/exact_cuda.cu:    starpv_CUDA(&pm, &vm, dl, vxl, pl, cl, dr, vxr, pr, cr, gamma);
src/riemann_solvers/exact_cuda.cu:    // sample_CUDA the solution at the cell interface
src/riemann_solvers/exact_cuda.cu:    sample_CUDA(pm, vm, &ds, &vs, &ps, dl, vxl, pl, cl, dr, vxr, pr, cr, gamma);
src/riemann_solvers/exact_cuda.cu:__device__ Real guessp_CUDA(Real dl, Real vxl, Real pl, Real cl, Real dr, Real vxr, Real pr, Real cr, Real gamma)
src/riemann_solvers/exact_cuda.cu:__device__ void prefun_CUDA(Real *f, Real *fd, Real p, Real dk, Real pk, Real ck, Real gamma)
src/riemann_solvers/exact_cuda.cu:__device__ void starpv_CUDA(Real *p, Real *v, Real dl, Real vxl, Real pl, Real cl, Real dr, Real vxr, Real pr, Real cr,
src/riemann_solvers/exact_cuda.cu:  pstart = guessp_CUDA(dl, vxl, pl, cl, dr, vxr, pr, cr, gamma);
src/riemann_solvers/exact_cuda.cu:    prefun_CUDA(&fl, &fld, pold, dl, pl, cl, gamma);
src/riemann_solvers/exact_cuda.cu:    prefun_CUDA(&fr, &frd, pold, dr, pr, cr, gamma);
src/riemann_solvers/exact_cuda.cu:__device__ void sample_CUDA(const Real pm, const Real vm, Real *d, Real *v, Real *p, Real dl, Real vxl, Real pl,
src/riemann_solvers/hll_cuda.cu:/*! \file hllc_cuda.cu
src/riemann_solvers/hll_cuda.cu: *  \brief Function definitions for the cuda HLLC Riemann solver.*/
src/riemann_solvers/hll_cuda.cu:#include "../global/global_cuda.h"
src/riemann_solvers/hll_cuda.cu:#include "../riemann_solvers/hll_cuda.h"
src/riemann_solvers/hll_cuda.cu:#include "../utils/gpu.hpp"
src/riemann_solvers/hll_cuda.cu:/*! \fn Calculate_HLLC_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real
src/riemann_solvers/hll_cuda.cu:__global__ void Calculate_HLL_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny,
src/riemann_solvers/hll_cuda.cu:    // Sl = sgn_CUDA(Sl)*fmax(fabs(Sl), etah);
src/riemann_solvers/hll_cuda.cu:    // Sr = sgn_CUDA(Sr)*fmax(fabs(Sr), etah);
src/riemann_solvers/roe_cuda.h:/*! \file roe_cuda.h
src/riemann_solvers/roe_cuda.h: *  \brief Declarations of functions for the cuda roe riemann solver kernel. */
src/riemann_solvers/roe_cuda.h:#ifndef ROE_CUDA_H
src/riemann_solvers/roe_cuda.h:#define ROE_CUDA_H
src/riemann_solvers/roe_cuda.h:/*! \fn Calculate_Roe_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real
src/riemann_solvers/roe_cuda.h:__global__ void Calculate_Roe_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny,
src/riemann_solvers/roe_cuda.h:#endif  // ROE_CUDA_H
src/riemann_solvers/hlld_cuda.cu: * \file hlld_cuda.cu
src/riemann_solvers/hlld_cuda.cu:#include "../global/global_cuda.h"
src/riemann_solvers/hlld_cuda.cu:#include "../riemann_solvers/hlld_cuda.h"
src/riemann_solvers/hlld_cuda.cu:#include "../utils/cuda_utilities.h"
src/riemann_solvers/hlld_cuda.cu:#include "../utils/gpu.hpp"
src/riemann_solvers/hlld_cuda.cu:__global__ void Calculate_HLLD_Fluxes_CUDA(Real const *dev_bounds_L, Real const *dev_bounds_R,
src/riemann_solvers/roe_cuda.cu:/*! \file roe_cuda.cu
src/riemann_solvers/roe_cuda.cu: *  \brief Function definitions for the cuda Roe Riemann solver.*/
src/riemann_solvers/roe_cuda.cu:#include "../global/global_cuda.h"
src/riemann_solvers/roe_cuda.cu:#include "../riemann_solvers/roe_cuda.h"
src/riemann_solvers/roe_cuda.cu:#include "../utils/gpu.hpp"
src/riemann_solvers/roe_cuda.cu:/*! \fn Calculate_Roe_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real
src/riemann_solvers/roe_cuda.cu:__global__ void Calculate_Roe_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny,
src/riemann_solvers/hlld_cuda.h: * \file hlld_cuda.cu
src/riemann_solvers/hlld_cuda.h:__global__ void Calculate_HLLD_Fluxes_CUDA(Real const *dev_bounds_L, Real const *dev_bounds_R,
src/riemann_solvers/hlld_cuda_tests.cu: * \file hlld_cuda_tests.cpp
src/riemann_solvers/hlld_cuda_tests.cu: * \brief Test the code units within hlld_cuda.cu
src/riemann_solvers/hlld_cuda_tests.cu:#include "../global/global_cuda.h"
src/riemann_solvers/hlld_cuda_tests.cu:#include "../riemann_solvers/hlld_cuda.h"  // Include code to test
src/riemann_solvers/hlld_cuda_tests.cu:#include "../utils/gpu.hpp"
src/riemann_solvers/hlld_cuda_tests.cu:class tMHDCalculateHLLDFluxesCUDA : public ::testing::Test
src/riemann_solvers/hlld_cuda_tests.cu:    GPU_Error_Check(cudaMalloc(&devConservedLeft, stateLeft.size() * sizeof(Real)));
src/riemann_solvers/hlld_cuda_tests.cu:    GPU_Error_Check(cudaMalloc(&devConservedRight, stateRight.size() * sizeof(Real)));
src/riemann_solvers/hlld_cuda_tests.cu:    GPU_Error_Check(cudaMalloc(&devConservedMagXFace, magneticX.size() * sizeof(Real)));
src/riemann_solvers/hlld_cuda_tests.cu:    GPU_Error_Check(cudaMalloc(&devTestFlux, testFlux.size() * sizeof(Real)));
src/riemann_solvers/hlld_cuda_tests.cu:    GPU_Error_Check(
src/riemann_solvers/hlld_cuda_tests.cu:        cudaMemcpy(devConservedLeft, stateLeft.data(), stateLeft.size() * sizeof(Real), cudaMemcpyHostToDevice));
src/riemann_solvers/hlld_cuda_tests.cu:    GPU_Error_Check(
src/riemann_solvers/hlld_cuda_tests.cu:        cudaMemcpy(devConservedRight, stateRight.data(), stateRight.size() * sizeof(Real), cudaMemcpyHostToDevice));
src/riemann_solvers/hlld_cuda_tests.cu:    GPU_Error_Check(
src/riemann_solvers/hlld_cuda_tests.cu:        cudaMemcpy(devConservedMagXFace, magneticX.data(), magneticX.size() * sizeof(Real), cudaMemcpyHostToDevice));
src/riemann_solvers/hlld_cuda_tests.cu:    hipLaunchKernelGGL(mhd::Calculate_HLLD_Fluxes_CUDA, dimGrid, dimBlock, 0, 0,
src/riemann_solvers/hlld_cuda_tests.cu:    GPU_Error_Check();
src/riemann_solvers/hlld_cuda_tests.cu:    GPU_Error_Check(cudaMemcpy(testFlux.data(), devTestFlux, testFlux.size() * sizeof(Real), cudaMemcpyDeviceToHost));
src/riemann_solvers/hlld_cuda_tests.cu:    cudaDeviceSynchronize();
src/riemann_solvers/hlld_cuda_tests.cu:    GPU_Error_Check();
src/riemann_solvers/hlld_cuda_tests.cu:    cudaFree(devConservedLeft);
src/riemann_solvers/hlld_cuda_tests.cu:    cudaFree(devConservedRight);
src/riemann_solvers/hlld_cuda_tests.cu:    cudaFree(devConservedMagXFace);
src/riemann_solvers/hlld_cuda_tests.cu:    cudaFree(devTestFlux);
src/riemann_solvers/hlld_cuda_tests.cu:TEST_F(tMHDCalculateHLLDFluxesCUDA, BrioAndWuShockTubeCorrectInputExpectCorrectOutput)
src/riemann_solvers/hlld_cuda_tests.cu:TEST_F(tMHDCalculateHLLDFluxesCUDA, DaiAndWoodwardShockTubeCorrectInputExpectCorrectOutput)
src/riemann_solvers/hlld_cuda_tests.cu:TEST_F(tMHDCalculateHLLDFluxesCUDA, RyuAndJones4dShockTubeCorrectInputExpectCorrectOutput)
src/riemann_solvers/hlld_cuda_tests.cu:TEST_F(tMHDCalculateHLLDFluxesCUDA, EinfeldtStrongRarefactionCorrectInputExpectCorrectOutput)
src/riemann_solvers/hlld_cuda_tests.cu:TEST_F(tMHDCalculateHLLDFluxesCUDA, ConstantStatesExpectCorrectFlux)
src/riemann_solvers/hlld_cuda_tests.cu:TEST_F(tMHDCalculateHLLDFluxesCUDA, DegenerateStateCorrectInputExpectCorrectOutput)
src/riemann_solvers/hlld_cuda_tests.cu:TEST_F(tMHDCalculateHLLDFluxesCUDA, AllZeroesExpectAllZeroes)
src/riemann_solvers/hlld_cuda_tests.cu:TEST_F(tMHDCalculateHLLDFluxesCUDA, UnphysicalValuesExpectAutomaticFix)
src/riemann_solvers/hllc_cuda_tests.cu: * \file hllc_cuda_tests.cpp
src/riemann_solvers/hllc_cuda_tests.cu: * \brief Test the code units within hllc_cuda.cu
src/riemann_solvers/hllc_cuda_tests.cu:#include "../global/global_cuda.h"
src/riemann_solvers/hllc_cuda_tests.cu:#include "../riemann_solvers/hllc_cuda.h"  // Include code to test
src/riemann_solvers/hllc_cuda_tests.cu:#include "../utils/gpu.hpp"
src/riemann_solvers/hllc_cuda_tests.cu:#if defined(CUDA) && defined(HLLC)
src/riemann_solvers/hllc_cuda_tests.cu:class tHYDROCalculateHLLCFluxesCUDA : public ::testing::Test
src/riemann_solvers/hllc_cuda_tests.cu:    GPU_Error_Check(cudaMalloc(&devConservedLeft, nFields * sizeof(Real)));
src/riemann_solvers/hllc_cuda_tests.cu:    GPU_Error_Check(cudaMalloc(&devConservedRight, nFields * sizeof(Real)));
src/riemann_solvers/hllc_cuda_tests.cu:    GPU_Error_Check(cudaMalloc(&devTestFlux, nFields * sizeof(Real)));
src/riemann_solvers/hllc_cuda_tests.cu:    GPU_Error_Check(cudaMemcpy(devConservedLeft, stateLeft.data(), nFields * sizeof(Real), cudaMemcpyHostToDevice));
src/riemann_solvers/hllc_cuda_tests.cu:    GPU_Error_Check(cudaMemcpy(devConservedRight, stateRight.data(), nFields * sizeof(Real), cudaMemcpyHostToDevice));
src/riemann_solvers/hllc_cuda_tests.cu:    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dimGrid, dimBlock, 0, 0,
src/riemann_solvers/hllc_cuda_tests.cu:    GPU_Error_Check();
src/riemann_solvers/hllc_cuda_tests.cu:    GPU_Error_Check(cudaMemcpy(testFlux.data(), devTestFlux, nFields * sizeof(Real), cudaMemcpyDeviceToHost));
src/riemann_solvers/hllc_cuda_tests.cu:    cudaDeviceSynchronize();
src/riemann_solvers/hllc_cuda_tests.cu:    GPU_Error_Check();
src/riemann_solvers/hllc_cuda_tests.cu:// Testing Calculate_HLLC_Fluxes_CUDA
src/riemann_solvers/hllc_cuda_tests.cu:TEST_F(tHYDROCalculateHLLCFluxesCUDA,        // Test suite name
src/global/global_cuda.cu:/*! \file global_cuda.cu
src/global/global_cuda.cu: *  \brief Declarations of the cuda global variables. */
src/global/global_cuda.cu:// Arrays for potential in GPU: Will be set to NULL if not using GRAVITY
src/global/global.cpp:#ifdef CHEMISTRY_GPU
src/global/global.h:#ifdef COOLING_GPU
src/global/global.h:#if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
src/global/global_cuda.h:/*! /file global_cuda.h
src/global/global_cuda.h: *  /brief Declarations of global variables and functions for the cuda kernels.
src/global/global_cuda.h:#ifndef GLOBAL_CUDA_H
src/global/global_cuda.h:#define GLOBAL_CUDA_H
src/global/global_cuda.h:#include "../utils/gpu.hpp"
src/global/global_cuda.h:// GPU arrays
src/global/global_cuda.h:// Arrays for potential in GPU: Will be set to NULL if not using GRAVITY
src/global/global_cuda.h:/*! \fn int sgn_CUDA
src/global/global_cuda.h:__device__ inline int sgn_CUDA(Real x)
src/global/global_cuda.h:#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
src/global/global_cuda.h:#endif  // GLOBAL_CUDA_H
src/io/io.cpp:#include "../utils/cuda_utilities.h"
src/io/io.cpp:  cudaMemcpy(G.C.density, G.C.device, G.H.n_fields * G.H.n_cells * sizeof(Real), cudaMemcpyDeviceToHost);
src/io/io.cpp:    cuda_utilities::DeviceVector<float> static device_dataset_vector{buffer_size};
src/io/io.cpp:void Fill_HDF5_Buffer_From_Grid_GPU(int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, int n_ghost,
src/io/io.cpp:// From src/io/io_gpu
src/io/io.cpp:void Write_Grid_HDF5_Field_GPU(Header H, hid_t file_id, Real *dataset_buffer, Real *device_hdf5_buffer,
src/io/io.cpp:  Fill_HDF5_Buffer_From_Grid_GPU(H.nx, H.ny, H.nz, H.nx_real, H.ny_real, H.nz_real, H.n_ghost, dataset_buffer,
src/io/io.cpp:void Write_Generic_HDF5_Field_GPU(int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, int n_ghost,
src/io/io.cpp:  Fill_HDF5_Buffer_From_Grid_GPU(nx, ny, nz, nx_real, ny_real, nz_real, n_ghost, dataset_buffer, device_hdf5_buffer,
src/io/io.cpp:  #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
src/io/io.cpp:  #endif  // COOLING_GRACKLE or CHEMISTRY_GPU
src/io/io.cpp:  cuda_utilities::DeviceVector<Real> static device_dataset_vector{buffer_size};
src/io/io.cpp:  Write_Grid_HDF5_Field_GPU(H, file_id, dataset_buffer, device_dataset_vector.data(), C.d_density, "/density");
src/io/io.cpp:    Write_Grid_HDF5_Field_GPU(H, file_id, dataset_buffer, device_dataset_vector.data(), C.d_momentum_x, "/momentum_x");
src/io/io.cpp:    Write_Grid_HDF5_Field_GPU(H, file_id, dataset_buffer, device_dataset_vector.data(), C.d_momentum_y, "/momentum_y");
src/io/io.cpp:    Write_Grid_HDF5_Field_GPU(H, file_id, dataset_buffer, device_dataset_vector.data(), C.d_momentum_z, "/momentum_z");
src/io/io.cpp:    Write_Grid_HDF5_Field_GPU(H, file_id, dataset_buffer, device_dataset_vector.data(), C.d_Energy, "/Energy");
src/io/io.cpp:    Write_Grid_HDF5_Field_GPU(H, file_id, dataset_buffer, device_dataset_vector.data(), C.d_GasEnergy, "/GasEnergy");
src/io/io.cpp:  Write_Grid_HDF5_Field_GPU(H, file_id, dataset_buffer, device_dataset_vector.data(), C.d_basic_scalar, "/scalar0");
src/io/io.cpp:  Write_Grid_HDF5_Field_GPU(H, file_id, dataset_buffer, device_dataset_vector.data(), C.d_dust_density,
src/io/io.cpp:      #ifdef CHEMISTRY_GPU
src/io/io.cpp:    #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
src/io/io.cpp:        #ifdef CHEMISTRY_GPU
src/io/io.cpp:    #endif  // COOLING_GRACKLE || CHEMISTRY_GPU
src/io/io.cpp:    Write_Generic_HDF5_Field_GPU(Grav.nx_local + 2 * N_GHOST_POTENTIAL, Grav.ny_local + 2 * N_GHOST_POTENTIAL,
src/io/io.cpp:          int const id  = cuda_utilities::compute1DIndex(xid, yid, zid, H.nx, H.ny);
src/io/io.cpp:          int const id  = cuda_utilities::compute1DIndex(xid, yid, zid, H.nx, H.ny);
src/io/io.cpp:          int const id  = cuda_utilities::compute1DIndex(xid, yid, zid, H.nx, H.ny);
src/io/io.cpp:        id     = cuda_utilities::compute1DIndex(i + H.n_ghost, j + H.n_ghost, zslice, H.nx, H.ny);
src/io/io.cpp:        int id_xm1 = cuda_utilities::compute1DIndex(i + H.n_ghost - 1, j + H.n_ghost, zslice, H.nx, H.ny);
src/io/io.cpp:        int id_ym1 = cuda_utilities::compute1DIndex(i + H.n_ghost, j + H.n_ghost - 1, zslice, H.nx, H.ny);
src/io/io.cpp:        int id_zm1 = cuda_utilities::compute1DIndex(i + H.n_ghost, j + H.n_ghost, zslice - 1, H.nx, H.ny);
src/io/io.cpp:          id = cuda_utilities::compute1DIndex(i + H.n_ghost, j + H.n_ghost, zslice - nz_local_start + H.n_ghost, H.nx,
src/io/io.cpp:          int id_xm1 = cuda_utilities::compute1DIndex(i + H.n_ghost - 1, j + H.n_ghost,
src/io/io.cpp:          int id_ym1 = cuda_utilities::compute1DIndex(i + H.n_ghost, j + H.n_ghost - 1,
src/io/io.cpp:          int id_zm1 = cuda_utilities::compute1DIndex(i + H.n_ghost, j + H.n_ghost,
src/io/io.cpp:        id     = cuda_utilities::compute1DIndex(i + H.n_ghost, yslice, k + H.n_ghost, H.nx, H.ny);
src/io/io.cpp:        int id_xm1 = cuda_utilities::compute1DIndex(i + H.n_ghost - 1, yslice, k + H.n_ghost, H.nx, H.ny);
src/io/io.cpp:        int id_ym1 = cuda_utilities::compute1DIndex(i + H.n_ghost, yslice - 1, k + H.n_ghost, H.nx, H.ny);
src/io/io.cpp:        int id_zm1 = cuda_utilities::compute1DIndex(i + H.n_ghost, yslice, k + H.n_ghost - 1, H.nx, H.ny);
src/io/io.cpp:          id = cuda_utilities::compute1DIndex(i + H.n_ghost, yslice - ny_local_start + H.n_ghost, k + H.n_ghost, H.nx,
src/io/io.cpp:          int id_xm1 = cuda_utilities::compute1DIndex(i + H.n_ghost - 1, yslice - ny_local_start + H.n_ghost,
src/io/io.cpp:          int id_ym1 = cuda_utilities::compute1DIndex(i + H.n_ghost, yslice - ny_local_start + H.n_ghost - 1,
src/io/io.cpp:          int id_zm1 = cuda_utilities::compute1DIndex(i + H.n_ghost, yslice - ny_local_start + H.n_ghost,
src/io/io.cpp:        id     = cuda_utilities::compute1DIndex(xslice, j + H.n_ghost, k + H.n_ghost, H.nx, H.ny);
src/io/io.cpp:        int id_xm1 = cuda_utilities::compute1DIndex(xslice - 1, j + H.n_ghost, k + H.n_ghost, H.nx, H.ny);
src/io/io.cpp:        int id_ym1 = cuda_utilities::compute1DIndex(xslice, j + H.n_ghost - 1, k + H.n_ghost, H.nx, H.ny);
src/io/io.cpp:        int id_zm1 = cuda_utilities::compute1DIndex(xslice, j + H.n_ghost, k + H.n_ghost - 1, H.nx, H.ny);
src/io/io.cpp:          id = cuda_utilities::compute1DIndex(xslice - nx_local_start, j + H.n_ghost, k + H.n_ghost, H.nx, H.ny);
src/io/io.cpp:              cuda_utilities::compute1DIndex(xslice - nx_local_start - 1, j + H.n_ghost, k + H.n_ghost, H.nx, H.ny);
src/io/io.cpp:              cuda_utilities::compute1DIndex(xslice - nx_local_start, j + H.n_ghost - 1, k + H.n_ghost, H.nx, H.ny);
src/io/io.cpp:              cuda_utilities::compute1DIndex(xslice - nx_local_start, j + H.n_ghost, k + H.n_ghost - 1, H.nx, H.ny);
src/io/io.cpp:    #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
src/io/io.cpp:    #endif    // COOLING_GRACKLE , CHEMISTRY_GPU
src/io/io.h:// From io/io_gpu.cu
src/io/io.h:// Use GPU to pack source -> device_buffer, then copy device_buffer -> buffer,
src/io/io_gpu.cu:  #include "../utils/cuda_utilities.h"
src/io/io_gpu.cu:// 2D version of CopyReal3D_GPU_Kernel. Note that magnetic fields and float32 output are not enabled in 2-D so this is a
src/io/io_gpu.cu:__global__ void CopyReal2D_GPU_Kernel(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost,
src/io/io_gpu.cu:  cuda_utilities::compute3DIndices(id, nx_real, ny_real, i, j, k);
src/io/io_gpu.cu:__global__ void CopyReal3D_GPU_Kernel(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost,
src/io/io_gpu.cu:  cuda_utilities::compute3DIndices(id, nx_real, ny_real, i, j, k);
src/io/io_gpu.cu:__global__ void CopyReal3D_GPU_Kernel(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost,
src/io/io_gpu.cu:  cuda_utilities::compute3DIndices(id, nx_real, ny_real, i, j, k);
src/io/io_gpu.cu:  hipLaunchKernelGGL(CopyReal3D_GPU_Kernel, dim1dGrid, dim1dBlock, 0, 0, nx, ny, nx_real, ny_real, nz_real, n_ghost,
src/io/io_gpu.cu:  GPU_Error_Check(
src/io/io_gpu.cu:      cudaMemcpy(buffer, device_buffer, nx_real * ny_real * nz_real * sizeof(double), cudaMemcpyDeviceToHost));
src/io/io_gpu.cu:  hipLaunchKernelGGL(CopyReal3D_GPU_Kernel, dim1dGrid, dim1dBlock, 0, 0, nx, ny, nx_real, ny_real, nz_real, n_ghost,
src/io/io_gpu.cu:  GPU_Error_Check(
src/io/io_gpu.cu:      cudaMemcpy(buffer, device_buffer, nx_real * ny_real * nz_real * sizeof(float), cudaMemcpyDeviceToHost));
src/io/io_gpu.cu:void Fill_HDF5_Buffer_From_Grid_GPU(int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, int n_ghost,
src/io/io_gpu.cu:    hipLaunchKernelGGL(CopyReal3D_GPU_Kernel, dim1dGrid, dim1dBlock, 0, 0, nx, ny, nx_real, ny_real, nz_real, n_ghost,
src/io/io_gpu.cu:    GPU_Error_Check(cudaMemcpy(hdf5_buffer, device_hdf5_buffer, nx_real * ny_real * nz_real * sizeof(Real),
src/io/io_gpu.cu:                               cudaMemcpyDeviceToHost));
src/io/io_gpu.cu:    hipLaunchKernelGGL(CopyReal2D_GPU_Kernel, dim1dGrid, dim1dBlock, 0, 0, nx, ny, nx_real, ny_real, nz_real, n_ghost,
src/io/io_gpu.cu:    GPU_Error_Check(
src/io/io_gpu.cu:        cudaMemcpy(hdf5_buffer, device_hdf5_buffer, nx_real * ny_real * sizeof(Real), cudaMemcpyDeviceToHost));
src/io/io_gpu.cu:    GPU_Error_Check(
src/io/io_gpu.cu:        cudaMemcpy(hdf5_buffer, device_grid_buffer + n_ghost, nx_real * sizeof(Real), cudaMemcpyDeviceToHost));
src/analysis/feedback_analysis.h:#ifdef PARTICLES_GPU
src/analysis/feedback_analysis.h:  void Compute_Gas_Velocity_Dispersion_GPU(Grid3D& G);
src/analysis/phase_diagram.cpp:  #elif defined CHEMISTRY_GPU
src/analysis/phase_diagram.cpp:            "CHEMISTRY_GPU\n");
src/analysis/feedback_analysis_gpu.cu:#ifdef PARTICLES_GPU
src/analysis/feedback_analysis_gpu.cu:void FeedbackAnalysis::Compute_Gas_Velocity_Dispersion_GPU(Grid3D &G)
src/analysis/feedback_analysis_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)&d_partial_mass, ngrid * sizeof(Real)));
src/analysis/feedback_analysis_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)&d_partial_vel, ngrid * sizeof(Real)));
src/analysis/feedback_analysis_gpu.cu:  // cudaDeviceSynchronize();
src/analysis/feedback_analysis_gpu.cu:  GPU_Error_Check(cudaMemcpy(h_partial_mass, d_partial_mass, ngrid * sizeof(Real), cudaMemcpyDeviceToHost));
src/analysis/feedback_analysis_gpu.cu:  GPU_Error_Check(cudaMemcpy(h_partial_vel, d_partial_vel, ngrid * sizeof(Real), cudaMemcpyDeviceToHost));
src/analysis/feedback_analysis_gpu.cu:  GPU_Error_Check(cudaFree(d_partial_vel));
src/analysis/feedback_analysis_gpu.cu:  GPU_Error_Check(cudaFree(d_partial_mass));
src/analysis/feedback_analysis_gpu.cu:#endif  // PARTICLES_GPU
src/analysis/analysis.cpp:  cudaMemcpy(C.density, C.device, H.n_fields * H.n_cells * sizeof(Real), cudaMemcpyDeviceToHost);
src/analysis/analysis.cpp:    #ifdef CHEMISTRY_GPU
src/analysis/lya_statistics.cpp:    #elif defined CHEMISTRY_GPU
src/analysis/lya_statistics.cpp:            "CHEMISTRY_GPU\n");
src/analysis/feedback_analysis.cpp:#ifdef PARTICLES_GPU
src/analysis/feedback_analysis.cpp:  GPU_Error_Check(cudaMalloc((void**)&d_circ_vel_x, G.H.n_cells * sizeof(Real)));
src/analysis/feedback_analysis.cpp:  GPU_Error_Check(cudaMalloc((void**)&d_circ_vel_y, G.H.n_cells * sizeof(Real)));
src/analysis/feedback_analysis.cpp:#ifdef PARTICLES_GPU
src/analysis/feedback_analysis.cpp:  GPU_Error_Check(cudaMemcpy(d_circ_vel_x, h_circ_vel_x, G.H.n_cells * sizeof(Real), cudaMemcpyHostToDevice));
src/analysis/feedback_analysis.cpp:  GPU_Error_Check(cudaMemcpy(d_circ_vel_y, h_circ_vel_y, G.H.n_cells * sizeof(Real), cudaMemcpyHostToDevice));
src/analysis/feedback_analysis.cpp:#ifdef PARTICLES_GPU
src/analysis/feedback_analysis.cpp:  GPU_Error_Check(cudaFree(d_circ_vel_x));
src/analysis/feedback_analysis.cpp:  GPU_Error_Check(cudaFree(d_circ_vel_y));
src/analysis/feedback_analysis.cpp:#elif defined(PARTICLES_GPU)
src/analysis/feedback_analysis.cpp:  Compute_Gas_Velocity_Dispersion_GPU(G);
src/hydro/hydro_cuda.cu:/*! \file hydro_cuda.cu
src/hydro/hydro_cuda.cu: *  \brief Definitions of functions used in all cuda integration algorithms. */
src/hydro/hydro_cuda.cu:#include "../global/global_cuda.h"
src/hydro/hydro_cuda.cu:#include "../hydro/hydro_cuda.h"
src/hydro/hydro_cuda.cu:#include "../utils/cuda_utilities.h"
src/hydro/hydro_cuda.cu:#include "../utils/gpu.hpp"
src/hydro/hydro_cuda.cu:    cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);
src/hydro/hydro_cuda.cu:Real Calc_dt_GPU(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dx, Real dy, Real dz,
src/hydro/hydro_cuda.cu:  cuda_utilities::DeviceVector<Real> static dev_dti(1);
src/hydro/hydro_cuda.cu:    // set launch parameters for GPU kernels.
src/hydro/hydro_cuda.cu:    cuda_utilities::AutomaticLaunchParams static const launchParams(Calc_dt_1D);
src/hydro/hydro_cuda.cu:    // set launch parameters for GPU kernels.
src/hydro/hydro_cuda.cu:    cuda_utilities::AutomaticLaunchParams static const launchParams(Calc_dt_2D);
src/hydro/hydro_cuda.cu:    // set launch parameters for GPU kernels.
src/hydro/hydro_cuda.cu:    cuda_utilities::AutomaticLaunchParams static const launchParams(Calc_dt_3D);
src/hydro/hydro_cuda.cu:  GPU_Error_Check();
src/hydro/hydro_cuda.cu:  // cudaMemcpy
src/hydro/hydro_cuda.cu:  // set values for GPU kernels
src/hydro/hydro_cuda.cu:  cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);
src/hydro/hydro_cuda.cu:  // set values for GPU kernels
src/hydro/hydro_cuda.cu:  // set values for GPU kernels
src/hydro/hydro_cuda.h:/*! \file hydro_cuda.h
src/hydro/hydro_cuda.h: *  \brief Declarations of functions used in all cuda integration algorithms. */
src/hydro/hydro_cuda.h:#ifndef HYDRO_CUDA_H
src/hydro/hydro_cuda.h:#define HYDRO_CUDA_H
src/hydro/hydro_cuda.h:Real Calc_dt_GPU(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dx, Real dy, Real dz,
src/hydro/hydro_cuda.h:#endif  // HYDRO_CUDA_H
src/hydro/hydro_cuda_tests.cu: * \file hydro_cuda_tests.cu
src/hydro/hydro_cuda_tests.cu: * \brief Test the code units within hydro_cuda.cu
src/hydro/hydro_cuda_tests.cu:#include "../global/global_cuda.h"
src/hydro/hydro_cuda_tests.cu:#include "../hydro/hydro_cuda.h"  // Include code to test
src/hydro/hydro_cuda_tests.cu:#include "../utils/gpu.hpp"
src/hydro/hydro_cuda_tests.cu:// Tests for the Calc_dt_GPU function
src/hydro/hydro_cuda_tests.cu:  cuda_utilities::DeviceVector<Real> dev_conserved(n_fields);
src/hydro/hydro_cuda_tests.cu:  cuda_utilities::DeviceVector<Real> dev_dti(1);
src/hydro/hydro_cuda_tests.cu:  GPU_Error_Check();
src/hydro/hydro_cuda_tests.cu:// End of tests for the Calc_dt_GPU function
src/hydro/hydro_cuda_tests.cu:  cuda_utilities::DeviceVector<Real> dev_conserved(n_fields);
src/particles/gravity_CIC_gpu.cu:  #include "../global/global_cuda.h"
src/particles/gravity_CIC_gpu.cu:  #include "../utils/gpu.hpp"
src/particles/gravity_CIC_gpu.cu:  #ifdef GRAVITY_GPU
src/particles/gravity_CIC_gpu.cu:  #ifdef PARTICLES_GPU
src/particles/gravity_CIC_gpu.cu:void Particles3D::Copy_Potential_To_GPU(Real *potential_host, Real *potential_dev, int n_cells_potential)
src/particles/gravity_CIC_gpu.cu:  GPU_Error_Check(cudaMemcpy(potential_dev, potential_host, n_cells_potential * sizeof(Real), cudaMemcpyHostToDevice));
src/particles/gravity_CIC_gpu.cu:void Particles3D::Get_Gravity_Field_Particles_GPU_function(int nx_local, int ny_local, int nz_local,
src/particles/gravity_CIC_gpu.cu:    #ifndef GRAVITY_GPU
src/particles/gravity_CIC_gpu.cu:  Copy_Potential_To_GPU(potential_host, potential_dev, n_cells_potential);
src/particles/gravity_CIC_gpu.cu:  // set values for GPU kernels
src/particles/gravity_CIC_gpu.cu:  GPU_Error_Check();
src/particles/gravity_CIC_gpu.cu:void Particles3D::Get_Gravity_CIC_GPU_function(part_int_t n_local, int nx_local, int ny_local, int nz_local,
src/particles/gravity_CIC_gpu.cu:  // set values for GPU kernels
src/particles/gravity_CIC_gpu.cu:    GPU_Error_Check();
src/particles/gravity_CIC_gpu.cu:  #endif  // PARTICLES_GPU
src/particles/gravity_CIC_gpu.cu:  #ifdef GRAVITY_GPU
src/particles/gravity_CIC_gpu.cu:void Grid3D::Copy_Particles_Density_GPU()
src/particles/gravity_CIC_gpu.cu:  // set values for GPU kernels
src/particles/gravity_CIC_gpu.cu:  #endif  // GRAVITY_GPU
src/particles/gravity_CIC.cpp:    #ifdef GRAVITY_GPU
src/particles/gravity_CIC.cpp:  Copy_Potential_From_GPU();
src/particles/gravity_CIC.cpp:  #ifdef PARTICLES_GPU
src/particles/gravity_CIC.cpp:  Particles.Get_Gravity_Field_Particles_GPU(Grav.F.potential_h);
src/particles/gravity_CIC.cpp:  #ifdef PARTICLES_GPU
src/particles/gravity_CIC.cpp:  Particles.Get_Gravity_CIC_GPU();
src/particles/gravity_CIC.cpp:  #ifdef PARTICLES_GPU
src/particles/gravity_CIC.cpp:void Particles3D::Get_Gravity_Field_Particles_GPU(Real *potential_host)
src/particles/gravity_CIC.cpp:  Get_Gravity_Field_Particles_GPU_function(G.nx_local, G.ny_local, G.nz_local, G.n_ghost_particles_grid,
src/particles/gravity_CIC.cpp:void Particles3D::Get_Gravity_CIC_GPU()
src/particles/gravity_CIC.cpp:  Get_Gravity_CIC_GPU_function(n_local, G.nx_local, G.ny_local, G.nz_local, G.n_ghost_particles_grid, G.xMin, G.xMax,
src/particles/gravity_CIC.cpp:  #endif  // PARTICLES_GPU
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:      #include "../utils/gpu_arrays_functions.h"
src/particles/particles_boundaries.cpp:      #include "particles_boundaries_gpu.h"
src/particles/particles_boundaries.cpp:    #endif  // PARTICLES_GPU
src/particles/particles_boundaries.cpp:  GPU_Error_Check();
src/particles/particles_boundaries.cpp:  GPU_Error_Check();
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:      #ifdef MPI_GPU
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:      #ifdef MPI_GPU
src/particles/particles_boundaries.cpp:      Extend_GPU_Array(&recv_buffer_x0_particles, Particles.G.recv_buffer_size_x0,
src/particles/particles_boundaries.cpp:                       Particles.G.gpu_allocation_factor * buffer_length, true);
src/particles/particles_boundaries.cpp:      Particles.G.recv_buffer_size_x0 = (part_int_t)Particles.G.gpu_allocation_factor * buffer_length;
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:      #ifdef MPI_GPU
src/particles/particles_boundaries.cpp:      Extend_GPU_Array(&recv_buffer_x1_particles, Particles.G.recv_buffer_size_x1,
src/particles/particles_boundaries.cpp:                       Particles.G.gpu_allocation_factor * buffer_length, true);
src/particles/particles_boundaries.cpp:      Particles.G.recv_buffer_size_x1 = (part_int_t)Particles.G.gpu_allocation_factor * buffer_length;
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:      #ifdef MPI_GPU
src/particles/particles_boundaries.cpp:      Extend_GPU_Array(&recv_buffer_y0_particles, Particles.G.recv_buffer_size_y0,
src/particles/particles_boundaries.cpp:                       Particles.G.gpu_allocation_factor * buffer_length, true);
src/particles/particles_boundaries.cpp:      Particles.G.recv_buffer_size_y0 = (part_int_t)Particles.G.gpu_allocation_factor * buffer_length;
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:      #ifdef MPI_GPU
src/particles/particles_boundaries.cpp:      Extend_GPU_Array(&recv_buffer_y1_particles, Particles.G.recv_buffer_size_y1,
src/particles/particles_boundaries.cpp:                       Particles.G.gpu_allocation_factor * buffer_length, true);
src/particles/particles_boundaries.cpp:      Particles.G.recv_buffer_size_y1 = (part_int_t)Particles.G.gpu_allocation_factor * buffer_length;
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:      #ifdef MPI_GPU
src/particles/particles_boundaries.cpp:      Extend_GPU_Array(&recv_buffer_z0_particles, Particles.G.recv_buffer_size_z0,
src/particles/particles_boundaries.cpp:                       Particles.G.gpu_allocation_factor * buffer_length, true);
src/particles/particles_boundaries.cpp:      Particles.G.recv_buffer_size_z0 = (part_int_t)Particles.G.gpu_allocation_factor * buffer_length;
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:      #ifdef MPI_GPU
src/particles/particles_boundaries.cpp:      Extend_GPU_Array(&recv_buffer_z1_particles, Particles.G.recv_buffer_size_z1,
src/particles/particles_boundaries.cpp:                       Particles.G.gpu_allocation_factor * buffer_length, true);
src/particles/particles_boundaries.cpp:      Particles.G.recv_buffer_size_z1 = (part_int_t)Particles.G.gpu_allocation_factor * buffer_length;
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:  Particles.Load_Particles_to_Buffer_GPU(0, 0, send_buffer_x0_particles, buffer_length_particles_x0_send);
src/particles/particles_boundaries.cpp:    #endif  // PARTICLES_GPU
src/particles/particles_boundaries.cpp:    #if defined(PARTICLES_GPU) && !defined(MPI_GPU)
src/particles/particles_boundaries.cpp:  cudaMemcpy(h_send_buffer_x0_particles, d_send_buffer_x0_particles, buffer_length * sizeof(Real),
src/particles/particles_boundaries.cpp:             cudaMemcpyDeviceToHost);
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:  Particles.Load_Particles_to_Buffer_GPU(0, 1, send_buffer_x1_particles, buffer_length_particles_x1_send);
src/particles/particles_boundaries.cpp:    #endif  // PARTICLES_GPU
src/particles/particles_boundaries.cpp:    #if defined(PARTICLES_GPU) && !defined(MPI_GPU)
src/particles/particles_boundaries.cpp:  cudaMemcpy(h_send_buffer_x1_particles, d_send_buffer_x1_particles, buffer_length * sizeof(Real),
src/particles/particles_boundaries.cpp:             cudaMemcpyDeviceToHost);
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:  Particles.Load_Particles_to_Buffer_GPU(1, 0, send_buffer_y0_particles, buffer_length_particles_y0_send);
src/particles/particles_boundaries.cpp:    #endif  // PARTICLES_GPU
src/particles/particles_boundaries.cpp:    #if defined(PARTICLES_GPU) && !defined(MPI_GPU)
src/particles/particles_boundaries.cpp:  cudaMemcpy(h_send_buffer_y0_particles, d_send_buffer_y0_particles, buffer_length * sizeof(Real),
src/particles/particles_boundaries.cpp:             cudaMemcpyDeviceToHost);
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:  Particles.Load_Particles_to_Buffer_GPU(1, 1, send_buffer_y1_particles, buffer_length_particles_y1_send);
src/particles/particles_boundaries.cpp:    #endif  // PARTICLES_GPU
src/particles/particles_boundaries.cpp:    #if defined(PARTICLES_GPU) && !defined(MPI_GPU)
src/particles/particles_boundaries.cpp:  cudaMemcpy(h_send_buffer_y1_particles, d_send_buffer_y1_particles, buffer_length * sizeof(Real),
src/particles/particles_boundaries.cpp:             cudaMemcpyDeviceToHost);
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:  Particles.Load_Particles_to_Buffer_GPU(2, 0, send_buffer_z0_particles, buffer_length_particles_z0_send);
src/particles/particles_boundaries.cpp:    #endif  // PARTICLES_GPU
src/particles/particles_boundaries.cpp:    #if defined(PARTICLES_GPU) && !defined(MPI_GPU)
src/particles/particles_boundaries.cpp:  cudaMemcpy(h_send_buffer_z0_particles, d_send_buffer_z0_particles, buffer_length * sizeof(Real),
src/particles/particles_boundaries.cpp:             cudaMemcpyDeviceToHost);
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:  Particles.Load_Particles_to_Buffer_GPU(2, 1, send_buffer_z1_particles, buffer_length_particles_z1_send);
src/particles/particles_boundaries.cpp:    #endif  // PARTICLES_GPU
src/particles/particles_boundaries.cpp:    #if defined(PARTICLES_GPU) && !defined(MPI_GPU)
src/particles/particles_boundaries.cpp:  cudaMemcpy(h_send_buffer_z1_particles, d_send_buffer_z1_particles, buffer_length * sizeof(Real),
src/particles/particles_boundaries.cpp:             cudaMemcpyDeviceToHost);
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:      #ifndef MPI_GPU
src/particles/particles_boundaries.cpp:  cudaMemcpy(d_recv_buffer_x0_particles, h_recv_buffer_x0_particles, buffer_length_particles_x0_recv * sizeof(Real),
src/particles/particles_boundaries.cpp:             cudaMemcpyHostToDevice);
src/particles/particles_boundaries.cpp:  Particles.Unload_Particles_from_Buffer_GPU(0, 0, d_recv_buffer_x0_particles, Particles.n_recv_x0);
src/particles/particles_boundaries.cpp:    #endif  // PARTICLES_GPU
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:      #ifndef MPI_GPU
src/particles/particles_boundaries.cpp:  cudaMemcpy(d_recv_buffer_x1_particles, h_recv_buffer_x1_particles, buffer_length_particles_x1_recv * sizeof(Real),
src/particles/particles_boundaries.cpp:             cudaMemcpyHostToDevice);
src/particles/particles_boundaries.cpp:  Particles.Unload_Particles_from_Buffer_GPU(0, 1, d_recv_buffer_x1_particles, Particles.n_recv_x1);
src/particles/particles_boundaries.cpp:    #endif  // PARTICLES_GPU
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:      #ifndef MPI_GPU
src/particles/particles_boundaries.cpp:  cudaMemcpy(d_recv_buffer_y0_particles, h_recv_buffer_y0_particles, buffer_length_particles_y0_recv * sizeof(Real),
src/particles/particles_boundaries.cpp:             cudaMemcpyHostToDevice);
src/particles/particles_boundaries.cpp:  Particles.Unload_Particles_from_Buffer_GPU(1, 0, d_recv_buffer_y0_particles, Particles.n_recv_y0);
src/particles/particles_boundaries.cpp:    #endif  // PARTICLES_GPU
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:      #ifndef MPI_GPU
src/particles/particles_boundaries.cpp:  cudaMemcpy(d_recv_buffer_y1_particles, h_recv_buffer_y1_particles, buffer_length_particles_y1_recv * sizeof(Real),
src/particles/particles_boundaries.cpp:             cudaMemcpyHostToDevice);
src/particles/particles_boundaries.cpp:  Particles.Unload_Particles_from_Buffer_GPU(1, 1, d_recv_buffer_y1_particles, Particles.n_recv_y1);
src/particles/particles_boundaries.cpp:    #endif  // PARTICLES_GPU
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:      #ifndef MPI_GPU
src/particles/particles_boundaries.cpp:  cudaMemcpy(d_recv_buffer_z0_particles, h_recv_buffer_z0_particles, buffer_length_particles_z0_recv * sizeof(Real),
src/particles/particles_boundaries.cpp:             cudaMemcpyHostToDevice);
src/particles/particles_boundaries.cpp:  Particles.Unload_Particles_from_Buffer_GPU(2, 0, d_recv_buffer_z0_particles, Particles.n_recv_z0);
src/particles/particles_boundaries.cpp:    #endif  // PARTICLES_GPU
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:      #ifndef MPI_GPU
src/particles/particles_boundaries.cpp:  cudaMemcpy(d_recv_buffer_z1_particles, h_recv_buffer_z1_particles, buffer_length_particles_z1_recv * sizeof(Real),
src/particles/particles_boundaries.cpp:             cudaMemcpyHostToDevice);
src/particles/particles_boundaries.cpp:  Particles.Unload_Particles_from_Buffer_GPU(2, 1, d_recv_buffer_z1_particles, Particles.n_recv_z1);
src/particles/particles_boundaries.cpp:    #endif  // PARTICLES_GPU
src/particles/particles_boundaries.cpp:  // When using PARTICLES_GPU the particles that need to be Transferred
src/particles/particles_boundaries.cpp:  // are selected on the Load_Buffer_GPU functions
src/particles/particles_boundaries.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_boundaries.cpp:int Particles3D::Select_Particles_to_Transfer_GPU(int direction, int side)
src/particles/particles_boundaries.cpp:  n_transfer = Select_Particles_to_Transfer_GPU_function(
src/particles/particles_boundaries.cpp:  GPU_Error_Check(cudaDeviceSynchronize());
src/particles/particles_boundaries.cpp:void Particles3D::Copy_Transfer_Particles_to_Buffer_GPU(int n_transfer, int direction, int side, Real *send_buffer_h,
src/particles/particles_boundaries.cpp:    Extend_GPU_Array(&send_buffer_d, *buffer_size,
src/particles/particles_boundaries.cpp:                     G.gpu_allocation_factor * (*n_send + n_transfer) * N_DATA_PER_PARTICLE_TRANSFER, true);
src/particles/particles_boundaries.cpp:    *buffer_size = (part_int_t)G.gpu_allocation_factor * (*n_send + n_transfer) * N_DATA_PER_PARTICLE_TRANSFER;
src/particles/particles_boundaries.cpp:  Load_Particles_to_Transfer_GPU_function(n_transfer, ++field_id, n_fields_to_transfer, pos_x_dev,
src/particles/particles_boundaries.cpp:  Load_Particles_to_Transfer_GPU_function(n_transfer, ++field_id, n_fields_to_transfer, pos_y_dev,
src/particles/particles_boundaries.cpp:  Load_Particles_to_Transfer_GPU_function(n_transfer, ++field_id, n_fields_to_transfer, pos_z_dev,
src/particles/particles_boundaries.cpp:  Load_Particles_to_Transfer_GPU_function(n_transfer, ++field_id, n_fields_to_transfer, vel_x_dev,
src/particles/particles_boundaries.cpp:  Load_Particles_to_Transfer_GPU_function(n_transfer, ++field_id, n_fields_to_transfer, vel_y_dev,
src/particles/particles_boundaries.cpp:  Load_Particles_to_Transfer_GPU_function(n_transfer, ++field_id, n_fields_to_transfer, vel_z_dev,
src/particles/particles_boundaries.cpp:  Load_Particles_to_Transfer_GPU_function(n_transfer, ++field_id, n_fields_to_transfer, mass_dev,
src/particles/particles_boundaries.cpp:  Load_Particles_to_Transfer_Int_GPU_function(n_transfer, ++field_id, n_fields_to_transfer, partIDs_dev,
src/particles/particles_boundaries.cpp:  Load_Particles_to_Transfer_GPU_function(n_transfer, ++field_id, n_fields_to_transfer, age_dev,
src/particles/particles_boundaries.cpp:  GPU_Error_Check(cudaDeviceSynchronize());
src/particles/particles_boundaries.cpp:void Particles3D::Replace_Tranfered_Particles_GPU(int n_transfer)
src/particles/particles_boundaries.cpp:  Replace_Transfered_Particles_GPU_function(n_transfer, pos_x_dev, G.transfer_particles_indices_d,
src/particles/particles_boundaries.cpp:  Replace_Transfered_Particles_GPU_function(n_transfer, pos_y_dev, G.transfer_particles_indices_d,
src/particles/particles_boundaries.cpp:  Replace_Transfered_Particles_GPU_function(n_transfer, pos_z_dev, G.transfer_particles_indices_d,
src/particles/particles_boundaries.cpp:  Replace_Transfered_Particles_GPU_function(n_transfer, vel_x_dev, G.transfer_particles_indices_d,
src/particles/particles_boundaries.cpp:  Replace_Transfered_Particles_GPU_function(n_transfer, vel_y_dev, G.transfer_particles_indices_d,
src/particles/particles_boundaries.cpp:  Replace_Transfered_Particles_GPU_function(n_transfer, vel_z_dev, G.transfer_particles_indices_d,
src/particles/particles_boundaries.cpp:  Replace_Transfered_Particles_GPU_function(n_transfer, mass_dev, G.transfer_particles_indices_d,
src/particles/particles_boundaries.cpp:  Replace_Transfered_Particles_Int_GPU_function(n_transfer, partIDs_dev, G.transfer_particles_indices_d,
src/particles/particles_boundaries.cpp:  Replace_Transfered_Particles_GPU_function(n_transfer, age_dev, G.transfer_particles_indices_d,
src/particles/particles_boundaries.cpp:  GPU_Error_Check(cudaDeviceSynchronize());
src/particles/particles_boundaries.cpp:void Particles3D::Load_Particles_to_Buffer_GPU(int direction, int side, Real *send_buffer_h, int buffer_length)
src/particles/particles_boundaries.cpp:  n_transfer = Select_Particles_to_Transfer_GPU(direction, side);
src/particles/particles_boundaries.cpp:  Copy_Transfer_Particles_to_Buffer_GPU(n_transfer, direction, side, send_buffer_h, buffer_length);
src/particles/particles_boundaries.cpp:  Replace_Tranfered_Particles_GPU(n_transfer);
src/particles/particles_boundaries.cpp: * Load_Particles_to_Buffer_GPU, except that the particles that are selected for
src/particles/particles_boundaries.cpp: * transfer are not moved into any buffer (Copy_Transfer_Particles_to_Buffer_GPU
src/particles/particles_boundaries.cpp:void Particles3D::Set_Particles_Open_Boundary_GPU(int dir, int side)
src/particles/particles_boundaries.cpp:  n_transfer = Select_Particles_to_Transfer_GPU(dir, side);
src/particles/particles_boundaries.cpp:  // n_transfer = Select_Particles_to_Transfer_GPU_function(  n_local, side,
src/particles/particles_boundaries.cpp:  // GPU_Error_Check(cudaDeviceSynchronize());
src/particles/particles_boundaries.cpp:  Replace_Tranfered_Particles_GPU(n_transfer);
src/particles/particles_boundaries.cpp:void Particles3D::Copy_Transfer_Particles_from_Buffer_GPU(int n_recv, Real *recv_buffer_d)
src/particles/particles_boundaries.cpp:    printf(" Reallocating GPU particles arrays. N local particles: %ld \n", n_local_after);
src/particles/particles_boundaries.cpp:    int new_size = G.gpu_allocation_factor * n_local_after;
src/particles/particles_boundaries.cpp:    Extend_GPU_Array(&pos_x_dev, (int)particles_array_size, new_size, true);
src/particles/particles_boundaries.cpp:    Extend_GPU_Array(&pos_y_dev, (int)particles_array_size, new_size, false);
src/particles/particles_boundaries.cpp:    Extend_GPU_Array(&pos_z_dev, (int)particles_array_size, new_size, false);
src/particles/particles_boundaries.cpp:    Extend_GPU_Array(&vel_x_dev, (int)particles_array_size, new_size, false);
src/particles/particles_boundaries.cpp:    Extend_GPU_Array(&vel_y_dev, (int)particles_array_size, new_size, false);
src/particles/particles_boundaries.cpp:    Extend_GPU_Array(&vel_z_dev, (int)particles_array_size, new_size, false);
src/particles/particles_boundaries.cpp:    Extend_GPU_Array(&grav_x_dev, (int)particles_array_size, new_size, false);
src/particles/particles_boundaries.cpp:    Extend_GPU_Array(&grav_y_dev, (int)particles_array_size, new_size, false);
src/particles/particles_boundaries.cpp:    Extend_GPU_Array(&grav_z_dev, (int)particles_array_size, new_size, false);
src/particles/particles_boundaries.cpp:    Extend_GPU_Array(&mass_dev, (int)particles_array_size, new_size, false);
src/particles/particles_boundaries.cpp:    Extend_GPU_Array(&partIDs_dev, (int)particles_array_size, new_size, false);
src/particles/particles_boundaries.cpp:    Extend_GPU_Array(&age_dev, (int)particles_array_size, new_size, false);
src/particles/particles_boundaries.cpp:    ReAllocate_Memory_GPU_MPI();
src/particles/particles_boundaries.cpp:  Unload_Particles_to_Transfer_GPU_function(n_local, n_recv, ++field_id, n_fields_to_transfer, pos_x_dev,
src/particles/particles_boundaries.cpp:  Unload_Particles_to_Transfer_GPU_function(n_local, n_recv, ++field_id, n_fields_to_transfer, pos_y_dev,
src/particles/particles_boundaries.cpp:  Unload_Particles_to_Transfer_GPU_function(n_local, n_recv, ++field_id, n_fields_to_transfer, pos_z_dev,
src/particles/particles_boundaries.cpp:  Unload_Particles_to_Transfer_GPU_function(n_local, n_recv, ++field_id, n_fields_to_transfer, vel_x_dev,
src/particles/particles_boundaries.cpp:  Unload_Particles_to_Transfer_GPU_function(n_local, n_recv, ++field_id, n_fields_to_transfer, vel_y_dev,
src/particles/particles_boundaries.cpp:  Unload_Particles_to_Transfer_GPU_function(n_local, n_recv, ++field_id, n_fields_to_transfer, vel_z_dev,
src/particles/particles_boundaries.cpp:  Unload_Particles_to_Transfer_GPU_function(n_local, n_recv, ++field_id, n_fields_to_transfer, mass_dev, recv_buffer_d);
src/particles/particles_boundaries.cpp:  Unload_Particles_Int_to_Transfer_GPU_function(n_local, n_recv, ++field_id, n_fields_to_transfer, partIDs_dev,
src/particles/particles_boundaries.cpp:  Unload_Particles_to_Transfer_GPU_function(n_local, n_recv, ++field_id, n_fields_to_transfer, age_dev, recv_buffer_d);
src/particles/particles_boundaries.cpp:void Particles3D::Unload_Particles_from_Buffer_GPU(int direction, int side, Real *recv_buffer_h, int n_recv)
src/particles/particles_boundaries.cpp:  GPU_Error_Check();
src/particles/particles_boundaries.cpp:  Copy_Transfer_Particles_from_Buffer_GPU(n_recv, recv_buffer_d);
src/particles/particles_boundaries.cpp:    #endif  // PARTICLES_GPU
src/particles/feedback_CIC_gpu.cu:#if defined(SUPERNOVA) && defined(PARTICLES_GPU) && defined(PARTICLE_AGE) && defined(PARTICLE_IDS)
src/particles/feedback_CIC_gpu.cu:  #include "../global/global_cuda.h"
src/particles/feedback_CIC_gpu.cu: * the cuRAND context for each thread on the GPU. Initialize more than the
src/particles/feedback_CIC_gpu.cu: * @param n_local  number of local particles on the GPU
src/particles/feedback_CIC_gpu.cu:    GPU_Error_Check(cudaMalloc((void**)&dev_snr, snr.size() * sizeof(Real)));
src/particles/feedback_CIC_gpu.cu:    GPU_Error_Check(cudaMemcpy(dev_snr, snr.data(), snr.size() * sizeof(Real), cudaMemcpyHostToDevice));
src/particles/feedback_CIC_gpu.cu:  GPU_Error_Check(cudaMalloc((void**)&randStates, n_states * sizeof(FeedbackPrng)));
src/particles/feedback_CIC_gpu.cu:  GPU_Error_Check(cudaDeviceSynchronize());
src/particles/feedback_CIC_gpu.cu:          " Feedback GPU: Particle outside local domain [%f  %f  %f]  [%f %f] "
src/particles/feedback_CIC_gpu.cu:          " Feedback GPU: Particle CIC index err [%f  %f  %f]  [%d %d %d] [%d "
src/particles/feedback_CIC_gpu.cu:          // cudaGetDeviceCount(&devcount);
src/particles/feedback_CIC_gpu.cu:          // cudaGetDevice(&devId);
src/particles/feedback_CIC_gpu.cu:    GPU_Error_Check(cudaMalloc(&d_dti, sizeof(Real)));
src/particles/feedback_CIC_gpu.cu:    GPU_Error_Check(cudaMemcpy(d_dti, &h_dti, sizeof(Real), cudaMemcpyHostToDevice));
src/particles/feedback_CIC_gpu.cu:    GPU_Error_Check(cudaMalloc(&d_prev_dens, G.Particles.n_local * sizeof(Real)));
src/particles/feedback_CIC_gpu.cu:    GPU_Error_Check(cudaMalloc(&d_prev_N, G.Particles.n_local * sizeof(int)));
src/particles/feedback_CIC_gpu.cu:    GPU_Error_Check(cudaMemset(d_prev_dens, 0, G.Particles.n_local * sizeof(Real)));
src/particles/feedback_CIC_gpu.cu:    GPU_Error_Check(cudaMemset(d_prev_N, 0, G.Particles.n_local * sizeof(int)));
src/particles/feedback_CIC_gpu.cu:    GPU_Error_Check(cudaMalloc((void**)&d_info, FEED_INFO_N * ngrid * sizeof(Real)));
src/particles/feedback_CIC_gpu.cu:      GPU_Error_Check(cudaMemcpy(&h_dti, d_dti, sizeof(Real), cudaMemcpyDeviceToHost));
src/particles/feedback_CIC_gpu.cu:        GPU_Error_Check(cudaDeviceSynchronize());
src/particles/feedback_CIC_gpu.cu:    GPU_Error_Check(cudaMemcpy(&h_info, d_info, FEED_INFO_N * sizeof(Real), cudaMemcpyDeviceToHost));
src/particles/feedback_CIC_gpu.cu:    GPU_Error_Check(cudaFree(d_dti));
src/particles/feedback_CIC_gpu.cu:    GPU_Error_Check(cudaFree(d_info));
src/particles/feedback_CIC_gpu.cu:    GPU_Error_Check(cudaFree(d_prev_dens));
src/particles/feedback_CIC_gpu.cu:    GPU_Error_Check(cudaFree(d_prev_N));
src/particles/feedback_CIC_gpu.cu:#endif  // SUPERNOVA & PARTICLES_GPU & PARTICLE_IDS & PARTICLE_AGE
src/particles/particles_boundaries_gpu.h:#if defined(PARTICLES) && defined(PARTICLES_GPU)
src/particles/particles_boundaries_gpu.h:part_int_t Select_Particles_to_Transfer_GPU_function(part_int_t n_local, int side, Real domainMin, Real domainMax,
src/particles/particles_boundaries_gpu.h:void Load_Particles_to_Transfer_GPU_function(int n_transfer, int field_id, int n_fields_to_transfer, Real *field_d,
src/particles/particles_boundaries_gpu.h:void Load_Particles_to_Transfer_Int_GPU_function(int n_transfer, int field_id, int n_fields_to_transfer,
src/particles/particles_boundaries_gpu.h:void Replace_Transfered_Particles_GPU_function(int n_transfer, Real *field_d, int *transfer_indices_d,
src/particles/particles_boundaries_gpu.h:void Replace_Transfered_Particles_Int_GPU_function(int n_transfer, part_int_t *field_d, int *transfer_indices_d,
src/particles/particles_boundaries_gpu.h:void Copy_Particles_GPU_Buffer_to_Host_Buffer(int n_transfer, Real *buffer_h, Real *buffer_d);
src/particles/particles_boundaries_gpu.h:void Copy_Particles_Host_Buffer_to_GPU_Buffer(int n_transfer, Real *buffer_h, Real *buffer_d);
src/particles/particles_boundaries_gpu.h:void Unload_Particles_to_Transfer_GPU_function(int n_local, int n_transfer, int field_id, int n_fields_to_transfer,
src/particles/particles_boundaries_gpu.h:void Unload_Particles_Int_to_Transfer_GPU_function(int n_local, int n_transfer, int field_id, int n_fields_to_transfer,
src/particles/particles_3D.cpp:  #if defined(PARTICLES_GPU) && defined(GRAVITY_GPU)
src/particles/particles_3D.cpp:  // Set the GPU array for the particles potential equal to the Gravity GPU
src/particles/particles_3D.cpp:  #ifdef PARTICLES_GPU
src/particles/particles_3D.cpp:    // Factor to allocate the particles data arrays on the GPU.
src/particles/particles_3D.cpp:    // When using MPI particles will be transferred to other GPU, for that
src/particles/particles_3D.cpp:  G.gpu_allocation_factor = 1.25;
src/particles/particles_3D.cpp:  G.gpu_allocation_factor = 1.0;
src/particles/particles_3D.cpp:  #endif  // PARTICLES_GPU
src/particles/particles_3D.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_3D.cpp:  Allocate_Memory_GPU_MPI();
src/particles/particles_3D.cpp:    #endif  // PARTICLES_GPU
src/particles/particles_3D.cpp:    #ifdef GRAVITY_GPU
src/particles/particles_3D.cpp:  #ifdef PARTICLES_GPU
src/particles/particles_3D.cpp:  Allocate_Memory_GPU();
src/particles/particles_3D.cpp:  #ifdef PARTICLES_GPU
src/particles/particles_3D.cpp:void Particles3D::Allocate_Memory_GPU()
src/particles/particles_3D.cpp:  // Allocate arrays for density and gravitational field on the GPU
src/particles/particles_3D.cpp:    #ifndef GRAVITY_GPU
src/particles/particles_3D.cpp:  chprintf(" Allocated GPU memory.\n");
src/particles/particles_3D.cpp:part_int_t Particles3D::Compute_Particles_GPU_Array_Size(part_int_t n)
src/particles/particles_3D.cpp:  part_int_t buffer_size = n * G.gpu_allocation_factor;
src/particles/particles_3D.cpp:void Particles3D::ReAllocate_Memory_GPU_MPI()
src/particles/particles_3D.cpp:  Free_GPU_Array_bool(G.transfer_particles_flags_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_int(G.transfer_particles_indices_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_int(G.replace_particles_indices_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_int(G.transfer_particles_prefix_sum_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_int(G.transfer_particles_prefix_sum_blocks_d);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_bool(&G.transfer_particles_flags_d, buffer_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_int(&G.transfer_particles_indices_d, buffer_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_int(&G.replace_particles_indices_d, buffer_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_int(&G.transfer_particles_prefix_sum_d, buffer_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_int(&G.transfer_particles_prefix_sum_blocks_d, half_blocks_size);
src/particles/particles_3D.cpp:void Particles3D::Allocate_Memory_GPU_MPI()
src/particles/particles_3D.cpp:  buffer_size      = Compute_Particles_GPU_Array_Size(n_local);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_bool(&G.transfer_particles_flags_d, buffer_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_int(&G.transfer_particles_indices_d, buffer_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_int(&G.replace_particles_indices_d, buffer_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_int(&G.transfer_particles_prefix_sum_d, buffer_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_int(&G.transfer_particles_prefix_sum_blocks_d, half_blocks_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_int(&G.n_transfer_d, 1);
src/particles/particles_3D.cpp:void Particles3D::Free_Memory_GPU()
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.density_dev);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.gravity_x_dev);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.gravity_y_dev);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.gravity_z_dev);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.dti_array_dev);
src/particles/particles_3D.cpp:    #ifndef GRAVITY_GPU
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.potential_dev);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(pos_x_dev);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(pos_y_dev);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(pos_z_dev);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(vel_x_dev);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(vel_y_dev);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(vel_z_dev);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(grav_x_dev);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(grav_y_dev);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(grav_z_dev);
src/particles/particles_3D.cpp:  Free_GPU_Array(partIDs_dev);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(age_dev);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(mass_dev);
src/particles/particles_3D.cpp:  Free_GPU_Array_bool(G.transfer_particles_flags_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_int(G.transfer_particles_prefix_sum_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_int(G.transfer_particles_prefix_sum_blocks_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_int(G.transfer_particles_indices_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_int(G.replace_particles_indices_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_int(G.n_transfer_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.send_buffer_x0_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.send_buffer_x1_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.send_buffer_y0_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.send_buffer_y1_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.send_buffer_z0_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.send_buffer_z1_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.recv_buffer_x0_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.recv_buffer_x1_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.recv_buffer_y0_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.recv_buffer_y1_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.recv_buffer_z0_d);
src/particles/particles_3D.cpp:  Free_GPU_Array_Real(G.recv_buffer_z1_d);
src/particles/particles_3D.cpp:  #endif  // PARTICLES_GPU
src/particles/particles_3D.cpp:  #ifdef PARTICLES_GPU
src/particles/particles_3D.cpp:  // Alocate memory in GPU for particle data
src/particles/particles_3D.cpp:  particles_array_size = Compute_Particles_GPU_Array_Size(n_particles_local);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&pos_x_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&pos_y_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&pos_z_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&vel_x_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&vel_y_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&vel_z_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&grav_x_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&grav_y_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&grav_z_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&mass_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Part_Int(&partIDs_dev, particles_array_size);
src/particles/particles_3D.cpp:  chprintf(" Allocated GPU memory for particle data\n");
src/particles/particles_3D.cpp:  #endif  // PARTICLES_GPU
src/particles/particles_3D.cpp:  #ifdef PARTICLES_GPU
src/particles/particles_3D.cpp:  #endif  // PARTICLES_GPU
src/particles/particles_3D.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_3D.cpp:    #endif  // PARTICLES_GPU
src/particles/particles_3D.cpp:  #ifdef PARTICLES_GPU
src/particles/particles_3D.cpp:  // Copyt the particle data from tepmpotal Host buffer to GPU memory
src/particles/particles_3D.cpp:  #endif  // PARTICLES_GPU
src/particles/particles_3D.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_3D.cpp:  particles_array_size = Compute_Particles_GPU_Array_Size(n_local);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&pos_x_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&pos_y_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&pos_z_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&vel_x_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&vel_y_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&vel_z_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&grav_x_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&grav_y_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&grav_z_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&mass_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Part_Int(&partIDs_dev, particles_array_size);
src/particles/particles_3D.cpp:  Allocate_Particles_GPU_Array_Real(&age_dev, particles_array_size);
src/particles/particles_3D.cpp:    #endif  // PARTICLES_GPU
src/particles/particles_3D.cpp:  #ifdef PARTICLES_GPU
src/particles/particles_3D.cpp:  Free_Memory_GPU();
src/particles/particles_boundaries_gpu.cu:#if defined(PARTICLES) && defined(PARTICLES_GPU)
src/particles/particles_boundaries_gpu.cu:  #include "../global/global_cuda.h"
src/particles/particles_boundaries_gpu.cu:  #include "../utils/gpu.hpp"
src/particles/particles_boundaries_gpu.cu:  #include "particles_boundaries_gpu.h"
src/particles/particles_boundaries_gpu.cu:void Grid3D::Set_Particles_Boundary_GPU(int dir, int side)
src/particles/particles_boundaries_gpu.cu:  // set values for GPU kernels
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check();
src/particles/particles_boundaries_gpu.cu:void Replace_Transfered_Particles_GPU_function(int n_transfer, Real *field_d, int *transfer_indices_d,
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check();
src/particles/particles_boundaries_gpu.cu:void Replace_Transfered_Particles_Int_GPU_function(int n_transfer, part_int_t *field_d, int *transfer_indices_d,
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check();
src/particles/particles_boundaries_gpu.cu:part_int_t Select_Particles_to_Transfer_GPU_function(part_int_t n_local, int side, Real domainMin, Real domainMax,
src/particles/particles_boundaries_gpu.cu:  // set values for GPU kernels
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check();
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check();
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check();
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check();
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check();
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check(cudaMemcpy(n_transfer_h, n_transfer_d, sizeof(int), cudaMemcpyDeviceToHost));
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check();
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check();
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check();
src/particles/particles_boundaries_gpu.cu:void Load_Particles_to_Transfer_GPU_function(int n_transfer, int field_id, int n_fields_to_transfer, Real *field_d,
src/particles/particles_boundaries_gpu.cu:  // set values for GPU kernels
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check();
src/particles/particles_boundaries_gpu.cu:void Load_Particles_to_Transfer_Int_GPU_function(int n_transfer, int field_id, int n_fields_to_transfer,
src/particles/particles_boundaries_gpu.cu:  // set values for GPU kernels
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check();
src/particles/particles_boundaries_gpu.cu:void Copy_Particles_GPU_Buffer_to_Host_Buffer(int n_transfer, Real *buffer_h, Real *buffer_d)
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check(cudaMemcpy(buffer_h, buffer_d, transfer_size * sizeof(Real), cudaMemcpyDeviceToHost));
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check();
src/particles/particles_boundaries_gpu.cu:void Copy_Particles_Host_Buffer_to_GPU_Buffer(int n_transfer, Real *buffer_h, Real *buffer_d)
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check(cudaMemcpy(buffer_d, buffer_h, transfer_size * sizeof(Real), cudaMemcpyHostToDevice));
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check();
src/particles/particles_boundaries_gpu.cu:void Unload_Particles_to_Transfer_GPU_function(int n_local, int n_transfer, int field_id, int n_fields_to_transfer,
src/particles/particles_boundaries_gpu.cu:  // set values for GPU kernels
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check();
src/particles/particles_boundaries_gpu.cu:void Unload_Particles_Int_to_Transfer_GPU_function(int n_local, int n_transfer, int field_id, int n_fields_to_transfer,
src/particles/particles_boundaries_gpu.cu:  // set values for GPU kernels
src/particles/particles_boundaries_gpu.cu:  GPU_Error_Check();
src/particles/density_CIC_gpu.cu:  #include "../global/global_cuda.h"
src/particles/density_CIC_gpu.cu:  #include "../utils/gpu.hpp"
src/particles/density_CIC_gpu.cu:  #ifdef GRAVITY_GPU
src/particles/density_CIC_gpu.cu:void Grid3D::Copy_Particles_Density_to_GPU()
src/particles/density_CIC_gpu.cu:  GPU_Error_Check(cudaMemcpy(Particles.G.density_dev, Particles.G.density, Particles.G.n_cells * sizeof(Real),
src/particles/density_CIC_gpu.cu:                             cudaMemcpyHostToDevice));
src/particles/density_CIC_gpu.cu:  #ifdef PARTICLES_GPU
src/particles/density_CIC_gpu.cu:    #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
src/particles/density_CIC_gpu.cu:// CUDA Kernel to compute the CIC density from the particles positions
src/particles/density_CIC_gpu.cu:void Particles3D::Clear_Density_GPU_function(Real *density_dev, int n_cells)
src/particles/density_CIC_gpu.cu:void Particles3D::Get_Density_CIC_GPU_function(part_int_t n_local, Real particle_mass, Real xMin, Real xMax, Real yMin,
src/particles/density_CIC_gpu.cu:  // set values for GPU kernels
src/particles/density_CIC_gpu.cu:    GPU_Error_Check();
src/particles/density_CIC_gpu.cu:    cudaDeviceSynchronize();
src/particles/density_CIC_gpu.cu:    #if !defined(GRAVITY_GPU)
src/particles/density_CIC_gpu.cu:  GPU_Error_Check(cudaMemcpy(density_h, density_dev, n_cells * sizeof(Real), cudaMemcpyDeviceToHost));
src/particles/density_CIC_gpu.cu:  #endif  // PARTICLES_GPU
src/particles/density_CIC.cpp:  #ifdef PARTICLES_GPU
src/particles/density_CIC.cpp:  Get_Density_CIC_GPU();
src/particles/density_CIC.cpp:  #ifdef GRAVITY_GPU
src/particles/density_CIC.cpp:  Copy_Particles_Density_to_GPU();
src/particles/density_CIC.cpp:  Copy_Particles_Density_GPU();
src/particles/density_CIC.cpp:  #endif  // GRAVITY_GPU
src/particles/density_CIC.cpp:  #ifdef PARTICLES_GPU
src/particles/density_CIC.cpp:  Clear_Density_GPU();
src/particles/density_CIC.cpp:  #ifdef PARTICLES_GPU
src/particles/density_CIC.cpp:void Particles3D::Clear_Density_GPU() { Clear_Density_GPU_function(G.density_dev, G.n_cells); }
src/particles/density_CIC.cpp:void Particles3D::Get_Density_CIC_GPU()
src/particles/density_CIC.cpp:  Get_Density_CIC_GPU_function(n_local, particle_mass, G.xMin, G.xMax, G.yMin, G.yMax, G.zMin, G.zMax, G.dx, G.dy, G.dz,
src/particles/density_CIC.cpp:  #endif  // PARTICLES_GPU
src/particles/particles_dynamics.cpp:  #ifdef PARTICLES_GPU
src/particles/particles_dynamics.cpp:  dt_particles = Calc_Particles_dt_GPU();
src/particles/particles_dynamics.cpp:  #endif  // PARTICLES_GPU
src/particles/particles_dynamics.cpp:  #ifdef PARTICLES_GPU
src/particles/particles_dynamics.cpp:// Go over all the particles and find dt_min in the GPU
src/particles/particles_dynamics.cpp:Real Grid3D::Calc_Particles_dt_GPU()
src/particles/particles_dynamics.cpp:  // set values for GPU kernels
src/particles/particles_dynamics.cpp:  max_dti = Particles.Calc_Particles_dt_GPU_function(
src/particles/particles_dynamics.cpp:// Update positions and velocities (step 1 of KDK scheme ) in the GPU
src/particles/particles_dynamics.cpp:void Grid3D::Advance_Particles_KDK_Step1_GPU()
src/particles/particles_dynamics.cpp:  Particles.Advance_Particles_KDK_Step1_Cosmo_GPU_function(
src/particles/particles_dynamics.cpp:  Particles.Advance_Particles_KDK_Step1_GPU_function(Particles.n_local, Particles.dt, Particles.pos_x_dev,
src/particles/particles_dynamics.cpp:// Update velocities (step 2 of KDK scheme ) in the GPU
src/particles/particles_dynamics.cpp:void Grid3D::Advance_Particles_KDK_Step2_GPU()
src/particles/particles_dynamics.cpp:  Particles.Advance_Particles_KDK_Step2_Cosmo_GPU_function(
src/particles/particles_dynamics.cpp:  Particles.Advance_Particles_KDK_Step2_GPU_function(Particles.n_local, Particles.dt, Particles.vel_x_dev,
src/particles/particles_dynamics.cpp:  #endif  // PARTICLES_GPU
src/particles/particles_dynamics.cpp:  GPU_Error_Check();
src/particles/particles_dynamics.cpp:  GPU_Error_Check();
src/particles/particles_dynamics.cpp:  #ifdef PARTICLES_GPU
src/particles/particles_dynamics.cpp:  Advance_Particles_KDK_Step1_GPU();
src/particles/particles_dynamics.cpp:  #endif  // PARTICLES_GPU
src/particles/particles_dynamics.cpp:  #ifdef PARTICLES_GPU
src/particles/particles_dynamics.cpp:  Advance_Particles_KDK_Step2_GPU();
src/particles/particles_dynamics.cpp:  #endif  // PARTICLES_GPU
src/particles/particles_dynamics.cpp:    #ifdef PARTICLES_GPU
src/particles/particles_dynamics.cpp:  dt_particles = Calc_Particles_dt_GPU();
src/particles/particles_dynamics.cpp:    #endif  // PARTICLES_GPU
src/particles/particles_3D.h:    #ifdef PARTICLES_GPU
src/particles/particles_3D.h:      // #define PRINT_GPU_MEMORY
src/particles/particles_3D.h:    #ifdef PARTICLES_GPU
src/particles/particles_3D.h:    #endif  // PARTICLES_GPU
src/particles/particles_3D.h:    #ifdef PARTICLES_GPU
src/particles/particles_3D.h:    Real gpu_allocation_factor;
src/particles/particles_3D.h:      #ifdef GRAVITY_GPU
src/particles/particles_3D.h:    #ifdef PARTICLES_GPU
src/particles/particles_3D.h:    #endif  // PARTICLES_GPU
src/particles/particles_3D.h:  void Free_GPU_Array_Real(Real *array);
src/particles/particles_3D.h:    #ifdef PARTICLES_GPU
src/particles/particles_3D.h:  void Free_GPU_Array_int(int *array);
src/particles/particles_3D.h:  void Free_GPU_Array_bool(bool *array);
src/particles/particles_3D.h:  void Free_GPU_Array(T *array)
src/particles/particles_3D.h:    cudaFree(array);
src/particles/particles_3D.h:  }  // TODO remove the Free_GPU_Array_<type> functions
src/particles/particles_3D.h:  void Allocate_Memory_GPU();
src/particles/particles_3D.h:  void Allocate_Particles_GPU_Array_Real(Real **array_dev, part_int_t size);
src/particles/particles_3D.h:  void Allocate_Particles_GPU_Array_bool(bool **array_dev, part_int_t size);
src/particles/particles_3D.h:  void Allocate_Particles_GPU_Array_int(int **array_dev, part_int_t size);
src/particles/particles_3D.h:  void Allocate_Particles_GPU_Array_Part_Int(part_int_t **array_dev, part_int_t size);
src/particles/particles_3D.h:  void Free_Memory_GPU();
src/particles/particles_3D.h:  void Initialize_Grid_Values_GPU();
src/particles/particles_3D.h:  void Get_Density_CIC_GPU();
src/particles/particles_3D.h:  void Get_Density_CIC_GPU_function(part_int_t n_local, Real particle_mass, Real xMin, Real xMax, Real yMin, Real yMax,
src/particles/particles_3D.h:  void Clear_Density_GPU();
src/particles/particles_3D.h:  void Clear_Density_GPU_function(Real *density_dev, int n_cells);
src/particles/particles_3D.h:  void Copy_Potential_To_GPU(Real *potential_host, Real *potential_dev, int n_cells_potential);
src/particles/particles_3D.h:  void Get_Gravity_Field_Particles_GPU(Real *potential_host);
src/particles/particles_3D.h:  void Get_Gravity_Field_Particles_GPU_function(int nx_local, int ny_local, int nz_local, int n_ghost_particles_grid,
src/particles/particles_3D.h:  void Get_Gravity_CIC_GPU();
src/particles/particles_3D.h:  void Get_Gravity_CIC_GPU_function(part_int_t n_local, int nx_local, int ny_local, int nz_local,
src/particles/particles_3D.h:  Real Calc_Particles_dt_GPU_function(int ngrid, part_int_t n_local, Real dx, Real dy, Real dz, Real *vel_x_dev,
src/particles/particles_3D.h:  void Advance_Particles_KDK_Step1_GPU_function(part_int_t n_local, Real dt, Real *pos_x_dev, Real *pos_y_dev,
src/particles/particles_3D.h:  void Advance_Particles_KDK_Step1_Cosmo_GPU_function(part_int_t n_local, Real delta_a, Real *pos_x_dev,
src/particles/particles_3D.h:  void Advance_Particles_KDK_Step2_GPU_function(part_int_t n_local, Real dt, Real *vel_x_dev, Real *vel_y_dev,
src/particles/particles_3D.h:  void Advance_Particles_KDK_Step2_Cosmo_GPU_function(part_int_t n_local, Real delta_a, Real *vel_x_dev,
src/particles/particles_3D.h:  part_int_t Compute_Particles_GPU_Array_Size(part_int_t n);
src/particles/particles_3D.h:  int Select_Particles_to_Transfer_GPU(int direction, int side);
src/particles/particles_3D.h:  void Copy_Transfer_Particles_to_Buffer_GPU(int n_transfer, int direction, int side, Real *send_buffer,
src/particles/particles_3D.h:  void Replace_Tranfered_Particles_GPU(int n_transfer);
src/particles/particles_3D.h:  void Unload_Particles_from_Buffer_GPU(int direction, int side, Real *recv_buffer_h, int n_recv);
src/particles/particles_3D.h:  void Copy_Transfer_Particles_from_Buffer_GPU(int n_recv, Real *recv_buffer_d);
src/particles/particles_3D.h:  void Set_Particles_Open_Boundary_GPU(int dir, int side);
src/particles/particles_3D.h:    #endif  // PARTICLES_GPU
src/particles/particles_3D.h:      #ifdef PARTICLES_GPU
src/particles/particles_3D.h:  void Allocate_Memory_GPU_MPI();
src/particles/particles_3D.h:  void ReAllocate_Memory_GPU_MPI();
src/particles/particles_3D.h:  void Load_Particles_to_Buffer_GPU(int direction, int side, Real *send_buffer, int buffer_length);
src/particles/particles_3D.h:      #endif  // PARTICLES_GPU
src/particles/density_boundaries.cpp:  cudaMemcpy(buffer_h, buffer_d, buffer_length * sizeof(Real), cudaMemcpyDeviceToHost);
src/particles/supernova.h:#if defined(PARTICLES_GPU) && defined(SUPERNOVA)
src/particles/supernova.h:#endif  // PARTICLES_GPU && SUPERNOVA
src/particles/particles_dynamics_gpu.cu:#if defined(PARTICLES) && defined(PARTICLES_GPU)
src/particles/particles_dynamics_gpu.cu:  #include "../global/global_cuda.h"
src/particles/particles_dynamics_gpu.cu:  #include "../utils/gpu.hpp"
src/particles/particles_dynamics_gpu.cu:// #include "../cosmology/cosmology_functions_gpu.h"
src/particles/particles_dynamics_gpu.cu:Real Particles3D::Calc_Particles_dt_GPU_function(int ngrid, part_int_t n_particles_local, Real dx, Real dy, Real dz,
src/particles/particles_dynamics_gpu.cu:  // // set values for GPU kernels
src/particles/particles_dynamics_gpu.cu:  GPU_Error_Check();
src/particles/particles_dynamics_gpu.cu:  GPU_Error_Check(cudaMemcpy(dti_array_host, dti_array_dev, ngrid * sizeof(Real), cudaMemcpyDeviceToHost));
src/particles/particles_dynamics_gpu.cu:void Particles3D::Advance_Particles_KDK_Step1_GPU_function(part_int_t n_local, Real dt, Real *pos_x_dev,
src/particles/particles_dynamics_gpu.cu:  // set values for GPU kernels
src/particles/particles_dynamics_gpu.cu:    GPU_Error_Check();
src/particles/particles_dynamics_gpu.cu:void Particles3D::Advance_Particles_KDK_Step2_GPU_function(part_int_t n_local, Real dt, Real *vel_x_dev,
src/particles/particles_dynamics_gpu.cu:  // set values for GPU kernels
src/particles/particles_dynamics_gpu.cu:    GPU_Error_Check();
src/particles/particles_dynamics_gpu.cu:void Particles3D::Advance_Particles_KDK_Step1_Cosmo_GPU_function(part_int_t n_local, Real delta_a, Real *pos_x_dev,
src/particles/particles_dynamics_gpu.cu:  // set values for GPU kernels
src/particles/particles_dynamics_gpu.cu:    GPU_Error_Check(cudaDeviceSynchronize());
src/particles/particles_dynamics_gpu.cu:    // GPU_Error_Check();
src/particles/particles_dynamics_gpu.cu:void Particles3D::Advance_Particles_KDK_Step2_Cosmo_GPU_function(part_int_t n_local, Real delta_a, Real *vel_x_dev,
src/particles/particles_dynamics_gpu.cu:  // set values for GPU kernels
src/particles/particles_dynamics_gpu.cu:    GPU_Error_Check(cudaDeviceSynchronize());
src/particles/particles_dynamics_gpu.cu:    // GPU_Error_Check();
src/particles/io_particles.cpp:      #ifdef PARTICLES_GPU
src/particles/io_particles.cpp:    // If PARTICLES_GPU: The positions are copied directly from the buffers so
src/particles/io_particles.cpp:      #endif  // PARTICLES_GPU
src/particles/io_particles.cpp:    #ifdef PARTICLES_GPU
src/particles/io_particles.cpp:  // Alocate memory in GPU for particle data
src/particles/io_particles.cpp:  particles_array_size = Compute_Particles_GPU_Array_Size(n_to_load);
src/particles/io_particles.cpp:  chprintf(" Allocating GPU buffer size: %ld * %f = %ld \n", n_to_load, G.gpu_allocation_factor, particles_array_size);
src/particles/io_particles.cpp:  Allocate_Particles_GPU_Array_Real(&pos_x_dev, particles_array_size);
src/particles/io_particles.cpp:  Allocate_Particles_GPU_Array_Real(&pos_y_dev, particles_array_size);
src/particles/io_particles.cpp:  Allocate_Particles_GPU_Array_Real(&pos_z_dev, particles_array_size);
src/particles/io_particles.cpp:  Allocate_Particles_GPU_Array_Real(&vel_x_dev, particles_array_size);
src/particles/io_particles.cpp:  Allocate_Particles_GPU_Array_Real(&vel_y_dev, particles_array_size);
src/particles/io_particles.cpp:  Allocate_Particles_GPU_Array_Real(&vel_z_dev, particles_array_size);
src/particles/io_particles.cpp:  Allocate_Particles_GPU_Array_Real(&grav_x_dev, particles_array_size);
src/particles/io_particles.cpp:  Allocate_Particles_GPU_Array_Real(&grav_y_dev, particles_array_size);
src/particles/io_particles.cpp:  Allocate_Particles_GPU_Array_Real(&grav_z_dev, particles_array_size);
src/particles/io_particles.cpp:  Allocate_Particles_GPU_Array_Real(&mass_dev, particles_array_size);
src/particles/io_particles.cpp:  Allocate_Particles_GPU_Array_Part_Int(&partIDs_dev, particles_array_size);
src/particles/io_particles.cpp:  Allocate_Particles_GPU_Array_Real(&age_dev, particles_array_size);
src/particles/io_particles.cpp:  chprintf(" Allocated GPU memory for particle data\n");
src/particles/io_particles.cpp:  // Copy the particle data to GPU memory
src/particles/io_particles.cpp:    #endif  // PARTICLES_GPU
src/particles/io_particles.cpp:    #ifdef PARTICLES_GPU
src/particles/io_particles.cpp:  GPU_Error_Check(cudaMemcpy(Particles.G.density, Particles.G.density_dev, Particles.G.n_cells * sizeof(Real),
src/particles/io_particles.cpp:                             cudaMemcpyDeviceToHost));
src/particles/io_particles.cpp:    #endif  // PARTICLES_GPU
src/particles/io_particles.cpp:    #if defined(OUTPUT_POTENTIAL) && defined(ONLY_PARTICLES) && defined(GRAVITY_GPU)
src/particles/io_particles.cpp:  GPU_Error_Check(cudaMemcpy(Grav.F.potential_h, Grav.F.potential_d, Grav.n_cells_potential * sizeof(Real),
src/particles/io_particles.cpp:                             cudaMemcpyDeviceToHost));
src/particles/io_particles.cpp:    #ifdef PARTICLES_GPU
src/particles/io_particles.cpp:    #endif  // PARTICLES_GPU
src/particles/io_particles.cpp:    #ifdef PARTICLES_GPU
src/particles/io_particles.cpp:    #endif  // PARTICLES_GPU
src/particles/io_particles.cpp:    #ifdef PARTICLES_GPU
src/particles/io_particles.cpp:    #endif  // PARTICLES_GPU
src/particles/io_particles.cpp:    #ifdef PARTICLES_GPU
src/particles/io_particles.cpp:    #endif  // PARTICLES_GPU
src/particles/io_particles.cpp:    #ifdef PARTICLES_GPU
src/particles/io_particles.cpp:    #endif  // PARTICLES_GPU
src/particles/io_particles.cpp:    #ifdef PARTICLES_GPU
src/particles/io_particles.cpp:    #endif  // PARTICLES_GPU
src/particles/io_particles.cpp:      #ifdef PARTICLES_GPU
src/particles/io_particles.cpp:      #endif  // PARTICLES_GPU
src/particles/io_particles.cpp:      #ifdef PARTICLES_GPU
src/particles/io_particles.cpp:      #endif  // PARTICLES_GPU
src/particles/io_particles.cpp:      #ifdef PARTICLES_GPU
src/particles/io_particles.cpp:      #endif  // PARTICLES_GPU
src/particles/density_boundaries_gpu.cu:#if defined(PARTICLES_GPU) && defined(GRAVITY_GPU)
src/particles/density_boundaries_gpu.cu:void Grid3D::Set_Particles_Density_Boundaries_Periodic_GPU(int direction, int side)
src/particles/density_boundaries_gpu.cu:  // set values for GPU kernels
src/particles/density_boundaries_gpu.cu:int Grid3D::Load_Particles_Density_Boundary_to_Buffer_GPU(int direction, int side, Real *buffer)
src/particles/density_boundaries_gpu.cu:  // set values for GPU kernels
src/particles/density_boundaries_gpu.cu:  cudaDeviceSynchronize();
src/particles/density_boundaries_gpu.cu:void Grid3D::Unload_Particles_Density_Boundary_From_Buffer_GPU(int direction, int side, Real *buffer)
src/particles/density_boundaries_gpu.cu:  // set values for GPU kernels
src/particles/density_boundaries_gpu.cu:#endif  // PARTICLES_GPU & GRAVITY_GPU
src/particles/particles_3D_gpu.cu:  #include "../global/global_cuda.h"
src/particles/particles_3D_gpu.cu:  #include "../utils/gpu.hpp"
src/particles/particles_3D_gpu.cu:void Particles3D::Free_GPU_Array_Real(Real *array) { cudaFree(array); }
src/particles/particles_3D_gpu.cu:  GPU_Error_Check(cudaMemGetInfo(&global_free, &global_total));
src/particles/particles_3D_gpu.cu:  #ifdef PRINT_GPU_MEMORY
src/particles/particles_3D_gpu.cu:  chprintf("Allocating GPU Memory:  %ld  MB free \n", global_free / 1000000);
src/particles/particles_3D_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)array_dev, size * sizeof(Real)));
src/particles/particles_3D_gpu.cu:  cudaDeviceSynchronize();
src/particles/particles_3D_gpu.cu:  #ifdef PARTICLES_GPU
src/particles/particles_3D_gpu.cu:  GPU_Error_Check(cudaMemGetInfo(&global_free, &global_total));
src/particles/particles_3D_gpu.cu:  cudaDeviceSynchronize();
src/particles/particles_3D_gpu.cu:      " Particles GPU Memory: N_local_max: %ld  (%.1f %)  mem_usage: %ld MB    "
src/particles/particles_3D_gpu.cu:void Particles3D::Free_GPU_Array_int(int *array) { cudaFree(array); }
src/particles/particles_3D_gpu.cu:void Particles3D::Free_GPU_Array_bool(bool *array) { cudaFree(array); }
src/particles/particles_3D_gpu.cu:  GPU_Error_Check();
src/particles/particles_3D_gpu.cu:void Particles3D::Allocate_Particles_GPU_Array_Real(Real **array_dev, part_int_t size)
src/particles/particles_3D_gpu.cu:  GPU_Error_Check(cudaMemGetInfo(&global_free, &global_total));
src/particles/particles_3D_gpu.cu:    #ifdef PRINT_GPU_MEMORY
src/particles/particles_3D_gpu.cu:  chprintf("Allocating GPU Memory:  %ld  MB free \n", global_free / 1000000);
src/particles/particles_3D_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)array_dev, size * sizeof(Real)));
src/particles/particles_3D_gpu.cu:  cudaDeviceSynchronize();
src/particles/particles_3D_gpu.cu:void Particles3D::Allocate_Particles_GPU_Array_int(int **array_dev, part_int_t size)
src/particles/particles_3D_gpu.cu:  GPU_Error_Check(cudaMemGetInfo(&global_free, &global_total));
src/particles/particles_3D_gpu.cu:    #ifdef PRINT_GPU_MEMORY
src/particles/particles_3D_gpu.cu:  chprintf("Allocating GPU Memory:  %ld  MB free \n", global_free / 1000000);
src/particles/particles_3D_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)array_dev, size * sizeof(int)));
src/particles/particles_3D_gpu.cu:  cudaDeviceSynchronize();
src/particles/particles_3D_gpu.cu:void Particles3D::Allocate_Particles_GPU_Array_Part_Int(part_int_t **array_dev, part_int_t size)
src/particles/particles_3D_gpu.cu:  GPU_Error_Check(cudaMemGetInfo(&global_free, &global_total));
src/particles/particles_3D_gpu.cu:    #ifdef PRINT_GPU_MEMORY
src/particles/particles_3D_gpu.cu:  chprintf("Allocating GPU Memory:  %ld  MB free \n", global_free / 1000000);
src/particles/particles_3D_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)array_dev, size * sizeof(part_int_t)));
src/particles/particles_3D_gpu.cu:  cudaDeviceSynchronize();
src/particles/particles_3D_gpu.cu:void Particles3D::Allocate_Particles_GPU_Array_bool(bool **array_dev, part_int_t size)
src/particles/particles_3D_gpu.cu:  GPU_Error_Check(cudaMemGetInfo(&global_free, &global_total));
src/particles/particles_3D_gpu.cu:    #ifdef PRINT_GPU_MEMORY
src/particles/particles_3D_gpu.cu:  chprintf("Allocating GPU Memory:  %ld  MB free \n", global_free / 1000000);
src/particles/particles_3D_gpu.cu:  GPU_Error_Check(cudaMalloc((void **)array_dev, size * sizeof(bool)));
src/particles/particles_3D_gpu.cu:  cudaDeviceSynchronize();
src/particles/particles_3D_gpu.cu:  GPU_Error_Check(cudaMemcpy(array_dev, array_host, size * sizeof(Real), cudaMemcpyHostToDevice));
src/particles/particles_3D_gpu.cu:  cudaDeviceSynchronize();
src/particles/particles_3D_gpu.cu:  GPU_Error_Check(cudaMemcpy(array_host, array_dev, size * sizeof(Real), cudaMemcpyDeviceToHost));
src/particles/particles_3D_gpu.cu:  cudaDeviceSynchronize();
src/particles/particles_3D_gpu.cu:  GPU_Error_Check(cudaMemcpy(array_dev, array_host, size * sizeof(part_int_t), cudaMemcpyHostToDevice));
src/particles/particles_3D_gpu.cu:  cudaDeviceSynchronize();
src/particles/particles_3D_gpu.cu:  GPU_Error_Check(cudaMemcpy(array_host, array_dev, size * sizeof(part_int_t), cudaMemcpyDeviceToHost));
src/particles/particles_3D_gpu.cu:  cudaDeviceSynchronize();
src/particles/particles_3D_gpu.cu:  // set values for GPU kernels
src/particles/particles_3D_gpu.cu:  GPU_Error_Check();
src/particles/particles_3D_gpu.cu:  #endif  // PARTICLES_GPU
src/reconstruction/ppmc_cuda.cu:/*! \file ppmc_cuda.cu
src/reconstruction/ppmc_cuda.cu:#include "../global/global_cuda.h"
src/reconstruction/ppmc_cuda.cu:#include "../reconstruction/ppmc_cuda.h"
src/reconstruction/ppmc_cuda.cu:#include "../utils/gpu.hpp"
src/reconstruction/ppmc_cuda.cu:  cuda_utilities::compute3DIndices(thread_id, nx, ny, xid, yid, zid);
src/reconstruction/ppmc_cuda.cu:  size_t id = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);
src/reconstruction/ppmc_cuda.cu:  id = cuda_utilities::compute1DIndex(xid - int(dir == 0), yid - int(dir == 1), zid - int(dir == 2), nx, ny);
src/reconstruction/ppmc_cuda.cu:  cuda_utilities::compute3DIndices(thread_id, nx, ny, xid, yid, zid);
src/reconstruction/ppmc_cuda.cu:  size_t id = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);
src/reconstruction/ppmc_cuda.cu:  id = cuda_utilities::compute1DIndex(xid - int(dir == 0), yid - int(dir == 1), zid - int(dir == 2), nx, ny);
src/reconstruction/plmc_cuda.h:/*! \file plmc_cuda.h
src/reconstruction/plmc_cuda.h: *  \brief Declarations of the cuda plm kernels, characteristic reconstruction
src/reconstruction/plmc_cuda.h:#ifndef PLMC_CUDA_H
src/reconstruction/plmc_cuda.h:#define PLMC_CUDA_H
src/reconstruction/plmc_cuda.h:/*! \fn __global__ void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real
src/reconstruction/plmc_cuda.h:__global__ __launch_bounds__(TPB) void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx,
src/reconstruction/plmc_cuda.h:#endif  // PLMC_CUDA_H
src/reconstruction/ppmc_cuda_tests.cu: * \file ppmc_cuda_tests.cu
src/reconstruction/ppmc_cuda_tests.cu: * \brief Tests for the contents of ppmc_cuda.h and ppmc_cuda.cu
src/reconstruction/ppmc_cuda_tests.cu:#include "../reconstruction/ppmc_cuda.h"
src/reconstruction/ppmc_cuda_tests.cu:  cuda_utilities::DeviceVector<double> dev_grid(host_grid.size());
src/reconstruction/ppmc_cuda_tests.cu:    cuda_utilities::DeviceVector<double> dev_interface_left(host_grid.size(), true);
src/reconstruction/ppmc_cuda_tests.cu:    cuda_utilities::DeviceVector<double> dev_interface_right(host_grid.size(), true);
src/reconstruction/ppmc_cuda_tests.cu:    GPU_Error_Check();
src/reconstruction/ppmc_cuda_tests.cu:    GPU_Error_Check(cudaDeviceSynchronize());
src/reconstruction/ppmc_cuda_tests.cu:  cuda_utilities::DeviceVector<double> dev_grid(host_grid.size());
src/reconstruction/ppmc_cuda_tests.cu:    cuda_utilities::DeviceVector<double> dev_interface_left(nx * ny * nz * (n_fields - 1), true);
src/reconstruction/ppmc_cuda_tests.cu:    cuda_utilities::DeviceVector<double> dev_interface_right(nx * ny * nz * (n_fields - 1), true);
src/reconstruction/ppmc_cuda_tests.cu:    GPU_Error_Check();
src/reconstruction/ppmc_cuda_tests.cu:    GPU_Error_Check(cudaDeviceSynchronize());
src/reconstruction/plmp_cuda.cu:/*! \file plmp_cuda.cu
src/reconstruction/plmp_cuda.cu:#include "../global/global_cuda.h"
src/reconstruction/plmp_cuda.cu:#include "../reconstruction/plmp_cuda.h"
src/reconstruction/plmp_cuda.cu:#include "../utils/gpu.hpp"
src/reconstruction/plmp_cuda.cu:/*! \fn __global__ void PLMP_cuda(Real *dev_conserved, Real *dev_bounds_L, Real
src/reconstruction/plmp_cuda.cu:__global__ void PLMP_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz,
src/reconstruction/plmp_cuda.cu:  // del_q_m = sgn_CUDA(del_q_C)*fmin(2.0*lim_slope_a, fabs(del_q_C));
src/reconstruction/plmp_cuda.cu:  del_q_m = sgn_CUDA(del_q_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
src/reconstruction/reconstruction.h:#include "../global/global_cuda.h"
src/reconstruction/reconstruction.h:#include "../utils/cuda_utilities.h"
src/reconstruction/reconstruction.h:#include "../utils/gpu.hpp"
src/reconstruction/reconstruction.h:  size_t const id = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);
src/reconstruction/plmc_cuda.cu:/*! \file plmc_cuda.cu
src/reconstruction/plmc_cuda.cu:#include "../global/global_cuda.h"
src/reconstruction/plmc_cuda.cu:#include "../reconstruction/plmc_cuda.h"
src/reconstruction/plmc_cuda.cu:#include "../utils/cuda_utilities.h"
src/reconstruction/plmc_cuda.cu:#include "../utils/gpu.hpp"
src/reconstruction/plmc_cuda.cu:/*! \fn __global__ void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real
src/reconstruction/plmc_cuda.cu:__global__ __launch_bounds__(TPB) void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx,
src/reconstruction/plmc_cuda.cu:  cuda_utilities::compute3DIndices(thread_id, nx, ny, xid, yid, zid);
src/reconstruction/plmc_cuda.cu:  size_t id = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);
src/reconstruction/plmc_cuda.cu:  id = cuda_utilities::compute1DIndex(xid - int(dir == 0), yid - int(dir == 1), zid - int(dir == 2), nx, ny);
src/reconstruction/ppmp_cuda.cu:/*! \file ppmp_cuda.cu
src/reconstruction/ppmp_cuda.cu:  #include "../global/global_cuda.h"
src/reconstruction/ppmp_cuda.cu:  #include "../reconstruction/ppmp_cuda.h"
src/reconstruction/ppmp_cuda.cu:  #include "../utils/gpu.hpp"
src/reconstruction/ppmp_cuda.cu:/*! \fn __global__ void PPMP_cuda(Real *dev_conserved, Real *dev_bounds_L, Real
src/reconstruction/ppmp_cuda.cu:__global__ void PPMP_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz,
src/reconstruction/ppmp_cuda.cu:  // del_q_m = sgn_CUDA(del_q_C)*fmin(2.0*lim_slope_a, fabs(del_q_C));
src/reconstruction/ppmp_cuda.cu:  del_q_m = sgn_CUDA(del_q_C) * fmin((Real)2.0 * lim_slope_a, lim_slope_b);
src/reconstruction/ppmp_cuda.h:/*! \file ppmp_cuda.h
src/reconstruction/ppmp_cuda.h: *  \brief Declarations of the cuda ppmp kernels. */
src/reconstruction/ppmp_cuda.h:#ifndef PPMP_CUDA_H
src/reconstruction/ppmp_cuda.h:#define PPMP_CUDA_H
src/reconstruction/ppmp_cuda.h:/*! \fn __global__ void PPMP_cuda(Real *dev_conserved, Real *dev_bounds_L, Real
src/reconstruction/ppmp_cuda.h:__global__ void PPMP_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz,
src/reconstruction/ppmp_cuda.h:#endif  // PPMP_CUDA_H
src/reconstruction/plmc_cuda_tests.cu: * \file plmc_cuda_tests.cu
src/reconstruction/plmc_cuda_tests.cu: * \brief Tests for the contents of plmc_cuda.h and plmc_cuda.cu
src/reconstruction/plmc_cuda_tests.cu:#include "../reconstruction/plmc_cuda.h"
src/reconstruction/plmc_cuda_tests.cu:  cuda_utilities::DeviceVector<double> dev_grid(host_grid.size());
src/reconstruction/plmc_cuda_tests.cu:    cuda_utilities::DeviceVector<double> dev_interface_left(host_grid.size(), true);
src/reconstruction/plmc_cuda_tests.cu:    cuda_utilities::DeviceVector<double> dev_interface_right(host_grid.size(), true);
src/reconstruction/plmc_cuda_tests.cu:    hipLaunchKernelGGL(PLMC_cuda, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
src/reconstruction/plmc_cuda_tests.cu:    GPU_Error_Check();
src/reconstruction/plmc_cuda_tests.cu:    GPU_Error_Check(cudaDeviceSynchronize());
src/reconstruction/plmc_cuda_tests.cu:  cuda_utilities::DeviceVector<double> dev_grid(host_grid.size());
src/reconstruction/plmc_cuda_tests.cu:    cuda_utilities::DeviceVector<double> dev_interface_left(n_cells_interface, true);
src/reconstruction/plmc_cuda_tests.cu:    cuda_utilities::DeviceVector<double> dev_interface_right(n_cells_interface, true);
src/reconstruction/plmc_cuda_tests.cu:    hipLaunchKernelGGL(PLMC_cuda, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
src/reconstruction/plmc_cuda_tests.cu:    GPU_Error_Check();
src/reconstruction/plmc_cuda_tests.cu:    GPU_Error_Check(cudaDeviceSynchronize());
src/reconstruction/ppmc_cuda.h:/*! \file ppmc_cuda.h
src/reconstruction/ppmc_cuda.h: *  \brief Declarations of the cuda ppm kernels, characteristic reconstruction
src/reconstruction/ppmc_cuda.h:#ifndef PPMC_CUDA_H
src/reconstruction/ppmc_cuda.h:#define PPMC_CUDA_H
src/reconstruction/ppmc_cuda.h:#endif  // PPMC_CUDA_H
src/reconstruction/plmp_cuda.h:/*! \file plmp_cuda.h
src/reconstruction/plmp_cuda.h: *  \brief Declarations of the cuda plmp kernels. */
src/reconstruction/plmp_cuda.h:#ifndef PLMP_CUDA_H
src/reconstruction/plmp_cuda.h:#define PLMP_CUDA_H
src/reconstruction/plmp_cuda.h:/*! \fn __global__ void PLMP_cuda(Real *dev_conserved, Real *dev_bounds_L, Real
src/reconstruction/plmp_cuda.h:__global__ void PLMP_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz,
src/reconstruction/plmp_cuda.h:#endif  // PLMP_CUDA_H
src/reconstruction/pcm_cuda.cu:/*! \file pcm_cuda.cu
src/reconstruction/pcm_cuda.cu:#include "../global/global_cuda.h"
src/reconstruction/pcm_cuda.cu:#include "../reconstruction/pcm_cuda.h"
src/reconstruction/pcm_cuda.cu:#include "../utils/cuda_utilities.h"
src/reconstruction/pcm_cuda.cu:#include "../utils/gpu.hpp"
src/reconstruction/pcm_cuda.cu:  cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);
src/reconstruction/pcm_cuda.cu:      id                              = cuda_utilities::compute1DIndex(xid - 1, yid, zid, nx, ny);
src/reconstruction/pcm_cuda.cu:      id                              = cuda_utilities::compute1DIndex(xid, yid - 1, zid, nx, ny);
src/reconstruction/pcm_cuda.cu:      id                              = cuda_utilities::compute1DIndex(xid, yid, zid - 1, nx, ny);
src/reconstruction/pcm_cuda.h:/*! \file pcm_cuda.h
src/reconstruction/pcm_cuda.h: *  \brief Declarations of the cuda pcm kernels */
src/reconstruction/pcm_cuda.h:#ifndef PCM_CUDA_H
src/reconstruction/pcm_cuda.h:#define PCM_CUDA_H
src/reconstruction/pcm_cuda.h:#endif  // PCM_CUDA_H
src/reconstruction/reconstruction_tests.cu:#include "../global/global_cuda.h"
src/reconstruction/reconstruction_tests.cu:#include "../utils/cuda_utilities.h"
src/reconstruction/reconstruction_tests.cu:#include "../utils/gpu.hpp"
src/reconstruction/reconstruction_tests.cu:  cuda_utilities::DeviceVector<reconstruction::Characteristic> dev_results(1);
src/reconstruction/reconstruction_tests.cu:  GPU_Error_Check();
src/reconstruction/reconstruction_tests.cu:  cudaDeviceSynchronize();
src/reconstruction/reconstruction_tests.cu:  cuda_utilities::DeviceVector<reconstruction::Primitive> dev_results(1);
src/reconstruction/reconstruction_tests.cu:  GPU_Error_Check();
src/reconstruction/reconstruction_tests.cu:  cudaDeviceSynchronize();
src/reconstruction/reconstruction_tests.cu:  cuda_utilities::DeviceVector<reconstruction::EigenVecs> dev_results(1);
src/reconstruction/reconstruction_tests.cu:  GPU_Error_Check();
src/reconstruction/reconstruction_tests.cu:  cudaDeviceSynchronize();
src/reconstruction/reconstruction_tests.cu:        int id = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);
src/reconstruction/reconstruction_tests.cu:  cuda_utilities::DeviceVector<reconstruction::Primitive> dev_results(1);
src/reconstruction/reconstruction_tests.cu:  GPU_Error_Check();
src/reconstruction/reconstruction_tests.cu:  cudaDeviceSynchronize();
src/reconstruction/reconstruction_tests.cu:  size_t const id = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);
src/system_tests/system_tester.h:  /// appear to differ from NVIDIA/GCC/XL by roughly 1E-12
src/h_correction/h_correction_3D_cuda.h:/*! \file h_correction_3D_cuda.h
src/h_correction/h_correction_3D_cuda.h:#include "../utils/gpu.hpp"
src/h_correction/h_correction_2D_cuda.h:/*! \file h_correction_2D_cuda.h
src/h_correction/h_correction_2D_cuda.h:    #include "../global/global_cuda.h"
src/h_correction/h_correction_2D_cuda.h:    #include "../utils/gpu.hpp"
src/h_correction/h_correction_2D_cuda.cu:/*! \file h_correction_2D_cuda.cu
src/h_correction/h_correction_2D_cuda.cu:  #include "../global/global_cuda.h"
src/h_correction/h_correction_2D_cuda.cu:  #include "../h_correction/h_correction_2D_cuda.h"
src/h_correction/h_correction_2D_cuda.cu:  #include "../utils/gpu.hpp"
src/h_correction/h_correction_3D_cuda.cu:/*! \file h_correction_3D_cuda.cu
src/h_correction/h_correction_3D_cuda.cu:#include "../global/global_cuda.h"
src/h_correction/h_correction_3D_cuda.cu:#include "../h_correction/h_correction_3D_cuda.h"
src/h_correction/h_correction_3D_cuda.cu:#include "../utils/gpu.hpp"
src/utils/hydro_utilities.h:#include "../global/global_cuda.h"
src/utils/hydro_utilities.h:#include "../utils/gpu.hpp"
src/utils/timing_functions.h:  OneTime Cooling_GPU;
src/utils/mhd_utilities.cu: * integrator. Due to the CUDA/HIP compiler requiring that device functions be
src/utils/mhd_utilities.cu:        size_t const id    = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);
src/utils/mhd_utilities.cu:        size_t const idxmo = cuda_utilities::compute1DIndex(i - 1, j, k, H.nx, H.ny);
src/utils/mhd_utilities.cu:        size_t const idymo = cuda_utilities::compute1DIndex(i, j - 1, k, H.nx, H.ny);
src/utils/mhd_utilities.cu:        size_t const idzmo = cuda_utilities::compute1DIndex(i, j, k - 1, H.nx, H.ny);
src/utils/timing_functions.cpp:  #include "../global/global_cuda.h"
src/utils/timing_functions.cpp:  cudaDeviceSynchronize();
src/utils/timing_functions.cpp:  cudaDeviceSynchronize();
src/utils/timing_functions.cpp:    // Get GPU ID
src/utils/timing_functions.cpp:    std::string gpu_id(MPI_MAX_PROCESSOR_NAME, ' ');
src/utils/timing_functions.cpp:    GPU_Error_Check(cudaGetDevice(&device));
src/utils/timing_functions.cpp:    GPU_Error_Check(cudaDeviceGetPCIBusId(gpu_id.data(), gpu_id.size(), device));
src/utils/timing_functions.cpp:    gpu_id.erase(
src/utils/timing_functions.cpp:        std::find_if(gpu_id.rbegin(), gpu_id.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
src/utils/timing_functions.cpp:        gpu_id.end());
src/utils/timing_functions.cpp:              << "         GPU PCI Bus ID: " << gpu_id << std::endl;
src/utils/timing_functions.cpp:  #ifdef COOLING_GPU
src/utils/timing_functions.cpp:      &(Cooling_GPU = OneTime("Cooling_GPU")),
src/utils/timing_functions.cpp:  #ifdef CHEMISTRY_GPU
src/utils/cuda_utilities.cpp: * \file cuda_utilities.cpp
src/utils/cuda_utilities.cpp: * \brief Implementation file for cuda_utilities.h
src/utils/cuda_utilities.cpp:#include "../utils/cuda_utilities.h"
src/utils/cuda_utilities.cpp:namespace cuda_utilities
src/utils/cuda_utilities.cpp:void Print_GPU_Memory_Usage(std::string const &additional_text)
src/utils/cuda_utilities.cpp:  size_t gpu_free_memory, gpu_total_memory;
src/utils/cuda_utilities.cpp:  GPU_Error_Check(cudaMemGetInfo(&gpu_free_memory, &gpu_total_memory));
src/utils/cuda_utilities.cpp:  // Assuming that all GPUs in the system have the same amount of memory
src/utils/cuda_utilities.cpp:  size_t const gpu_used_memory = Reduce_size_t_Max(gpu_total_memory - gpu_free_memory);
src/utils/cuda_utilities.cpp:  Real const percent_used = 100.0 * (static_cast<Real>(gpu_used_memory) / static_cast<Real>(gpu_total_memory));
src/utils/cuda_utilities.cpp:  output_message_stream << "Percentage of GPU memory used: " << percent_used << "%. GPU memory used "
src/utils/cuda_utilities.cpp:                        << std::to_string(gpu_used_memory) << ", GPU total memory " << std::to_string(gpu_total_memory)
src/utils/cuda_utilities.cpp:}  // end namespace cuda_utilities
src/utils/testing_utilities.h: * considered compatible with CUDA/HIP.
src/utils/reduction_utilities_tests.cu:#include "../utils/cuda_utilities.h"
src/utils/reduction_utilities_tests.cu:  cuda_utilities::AutomaticLaunchParams static const launchParams(reduction_utilities::kernelReduceMax);
src/utils/reduction_utilities_tests.cu:  cuda_utilities::DeviceVector<Real> dev_grid(host_grid.size());
src/utils/reduction_utilities_tests.cu:  cuda_utilities::DeviceVector<Real> static dev_max(1);
src/utils/reduction_utilities_tests.cu:  GPU_Error_Check();
src/utils/gpu.hpp:  #define cudaDeviceSynchronize              hipDeviceSynchronize
src/utils/gpu.hpp:  #define cudaError                          hipError_t
src/utils/gpu.hpp:  #define cudaError_t                        hipError_t
src/utils/gpu.hpp:  #define cudaErrorInsufficientDriver        hipErrorInsufficientDriver
src/utils/gpu.hpp:  #define cudaErrorNoDevice                  hipErrorNoDevice
src/utils/gpu.hpp:  #define cudaEvent_t                        hipEvent_t
src/utils/gpu.hpp:  #define cudaEventCreate                    hipEventCreate
src/utils/gpu.hpp:  #define cudaEventElapsedTime               hipEventElapsedTime
src/utils/gpu.hpp:  #define cudaEventRecord                    hipEventRecord
src/utils/gpu.hpp:  #define cudaEventSynchronize               hipEventSynchronize
src/utils/gpu.hpp:  #define cudaFree                           hipFree
src/utils/gpu.hpp:  #define cudaFreeHost                       hipHostFree
src/utils/gpu.hpp:  #define cudaGetDevice                      hipGetDevice
src/utils/gpu.hpp:  #define cudaGetDeviceCount                 hipGetDeviceCount
src/utils/gpu.hpp:  #define cudaGetErrorString                 hipGetErrorString
src/utils/gpu.hpp:  #define cudaGetLastError                   hipGetLastError
src/utils/gpu.hpp:  #define cudaHostAlloc                      hipHostMalloc
src/utils/gpu.hpp:  #define cudaHostAllocDefault               hipHostMallocDefault
src/utils/gpu.hpp:  #define cudaMalloc                         hipMalloc
src/utils/gpu.hpp:  #define cudaMemcpy                         hipMemcpy
src/utils/gpu.hpp:  #define cudaMemcpyAsync                    hipMemcpyAsync
src/utils/gpu.hpp:  #define cudaMemcpyPeer                     hipMemcpyPeer
src/utils/gpu.hpp:  #define cudaMemcpyDeviceToHost             hipMemcpyDeviceToHost
src/utils/gpu.hpp:  #define cudaMemcpyDeviceToDevice           hipMemcpyDeviceToDevice
src/utils/gpu.hpp:  #define cudaMemcpyHostToDevice             hipMemcpyHostToDevice
src/utils/gpu.hpp:  #define cudaMemGetInfo                     hipMemGetInfo
src/utils/gpu.hpp:  #define cudaMemset                         hipMemset
src/utils/gpu.hpp:  #define cudaReadModeElementType            hipReadModeElementType
src/utils/gpu.hpp:  #define cudaSetDevice                      hipSetDevice
src/utils/gpu.hpp:  #define cudaSuccess                        hipSuccess
src/utils/gpu.hpp:  #define cudaDeviceProp                     hipDeviceProp_t
src/utils/gpu.hpp:  #define cudaGetDeviceProperties            hipGetDeviceProperties
src/utils/gpu.hpp:  #define cudaPointerAttributes              hipPointerAttribute_t
src/utils/gpu.hpp:  #define cudaPointerGetAttributes           hipPointerGetAttributes
src/utils/gpu.hpp:  #define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
src/utils/gpu.hpp:  #define cudaMemGetInfo                     hipMemGetInfo
src/utils/gpu.hpp:  #define cudaDeviceGetPCIBusId              hipDeviceGetPCIBusId
src/utils/gpu.hpp:  #define cudaPeekAtLastError                hipPeekAtLastError
src/utils/gpu.hpp:  #define cudaArray           hipArray
src/utils/gpu.hpp:  #define cudaMallocArray     hipMallocArray
src/utils/gpu.hpp:  #define cudaFreeArray       hipFreeArray
src/utils/gpu.hpp:  #define cudaMemcpyToArray   hipMemcpyToArray
src/utils/gpu.hpp:  #define cudaMemcpy2DToArray hipMemcpy2DToArray
src/utils/gpu.hpp:  #define cudaTextureObject_t      hipTextureObject_t
src/utils/gpu.hpp:  #define cudaCreateTextureObject  hipCreateTextureObject
src/utils/gpu.hpp:  #define cudaDestroyTextureObject hipDestroyTextureObject
src/utils/gpu.hpp:  #define cudaChannelFormatDesc      hipChannelFormatDesc
src/utils/gpu.hpp:  #define cudaCreateChannelDesc      hipCreateChannelDesc
src/utils/gpu.hpp:  #define cudaChannelFormatKindFloat hipChannelFormatKindFloat
src/utils/gpu.hpp:  #define cudaResourceDesc      hipResourceDesc
src/utils/gpu.hpp:  #define cudaResourceTypeArray hipResourceTypeArray
src/utils/gpu.hpp:  #define cudaTextureDesc       hipTextureDesc
src/utils/gpu.hpp:  #define cudaAddressModeClamp  hipAddressModeClamp
src/utils/gpu.hpp:  #define cudaFilterModeLinear  hipFilterModeLinear
src/utils/gpu.hpp:  #define cudaFilterModePoint   hipFilterModePoint
src/utils/gpu.hpp:  #define cudaPointerAttributes    hipPointerAttribute_t
src/utils/gpu.hpp:  #define cudaPointerGetAttributes hipPointerGetAttributes
src/utils/gpu.hpp:  #include <cuda_runtime.h>
src/utils/gpu.hpp:#define GPU_MAX_THREADS 256
src/utils/gpu.hpp: * \brief Check for CUDA/HIP error codes. Can be called wrapping a GPU function that returns a value or with no
src/utils/gpu.hpp:inline void GPU_Error_Check(cudaError_t code = cudaPeekAtLastError(), bool abort = true,
src/utils/gpu.hpp:#ifndef DISABLE_GPU_ERROR_CHECKING
src/utils/gpu.hpp:  code = cudaDeviceSynchronize();
src/utils/gpu.hpp:  if (code != cudaSuccess) {
src/utils/gpu.hpp:    std::cout << "GPU_Error_Check: Failed at "
src/utils/gpu.hpp:              << ", Function: " << location.function_name() << ", with code: " << cudaGetErrorString(code) << std::endl;
src/utils/gpu.hpp:#endif  // DISABLE_GPU_ERROR_CHECKING
src/utils/gpu.hpp:inline void GPU_Error_Check(cufftResult_t code, bool abort = true,
src/utils/gpu.hpp:  #ifndef DISABLE_GPU_ERROR_CHECKING
src/utils/gpu.hpp:    std::cout << "GPU_Error_Check: Failed at "
src/utils/gpu.hpp:  #endif  // DISABLE_GPU_ERROR_CHECKING
src/utils/gpu.hpp:#if defined(__CUDACC__) || defined(__HIPCC__)
src/utils/gpu.hpp:__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun0(const int n0, const F f)
src/utils/gpu.hpp:void gpuFor(const int n0, const F f)
src/utils/gpu.hpp:  const int b0 = (n0 + GPU_MAX_THREADS - 1) / GPU_MAX_THREADS;
src/utils/gpu.hpp:  gpuRun0<<<b0, t0>>>(n0, f);
src/utils/gpu.hpp:  GPU_Error_Check();
src/utils/gpu.hpp:__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun0x2(const F f)
src/utils/gpu.hpp:__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun1x1(const F f)
src/utils/gpu.hpp:__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun2x0(const int n1, const F f)
src/utils/gpu.hpp:void gpuFor(const int n0, const int n1, const F f)
src/utils/gpu.hpp:  if (n1 > GPU_MAX_THREADS) {
src/utils/gpu.hpp:    const int b1 = (n1 + GPU_MAX_THREADS - 1) / GPU_MAX_THREADS;
src/utils/gpu.hpp:    gpuRun2x0<<<dim3(b1, n0), dim3(t1)>>>(n1, f);
src/utils/gpu.hpp:    GPU_Error_Check();
src/utils/gpu.hpp:  } else if (nl01 > GPU_MAX_THREADS) {
src/utils/gpu.hpp:    gpuRun1x1<<<n0, n1>>>(f);
src/utils/gpu.hpp:    GPU_Error_Check();
src/utils/gpu.hpp:    gpuRun0x2<<<1, dim3(n1, n0)>>>(f);
src/utils/gpu.hpp:    GPU_Error_Check();
src/utils/gpu.hpp:__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun0x3(const F f)
src/utils/gpu.hpp:__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun1x2(const F f)
src/utils/gpu.hpp:__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun2x1(const F f)
src/utils/gpu.hpp:__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun3x0(const int n2, const F f)
src/utils/gpu.hpp:void gpuFor(const int n0, const int n1, const int n2, const F f)
src/utils/gpu.hpp:  if (n2 > GPU_MAX_THREADS) {
src/utils/gpu.hpp:    const int b2 = (n2 + GPU_MAX_THREADS - 1) / GPU_MAX_THREADS;
src/utils/gpu.hpp:    gpuRun3x0<<<dim3(b2, n1, n0), t2>>>(n2, f);
src/utils/gpu.hpp:    GPU_Error_Check();
src/utils/gpu.hpp:  } else if (nl12 > GPU_MAX_THREADS) {
src/utils/gpu.hpp:    gpuRun2x1<<<dim3(n1, n0), n2>>>(f);
src/utils/gpu.hpp:    GPU_Error_Check();
src/utils/gpu.hpp:  } else if (nl012 > GPU_MAX_THREADS) {
src/utils/gpu.hpp:    gpuRun1x2<<<n0, dim3(n2, n1)>>>(f);
src/utils/gpu.hpp:    GPU_Error_Check();
src/utils/gpu.hpp:    gpuRun0x3<<<1, dim3(n2, n1, n0)>>>(f);
src/utils/gpu.hpp:    GPU_Error_Check();
src/utils/gpu.hpp:__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun1x3(const F f)
src/utils/gpu.hpp:__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun2x2(const F f)
src/utils/gpu.hpp:__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun3x1(const F f)
src/utils/gpu.hpp:__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun4x0(const int n23, const int n3, const F f)
src/utils/gpu.hpp:void gpuFor(const int n0, const int n1, const int n2, const int n3, const F f)
src/utils/gpu.hpp:  if (n3 > GPU_MAX_THREADS) {
src/utils/gpu.hpp:    const int b23 = (n23 + GPU_MAX_THREADS - 1) / GPU_MAX_THREADS;
src/utils/gpu.hpp:    gpuRun4x0<<<dim3(b23, n1, n0), t23>>>(n23, n3, f);
src/utils/gpu.hpp:    GPU_Error_Check();
src/utils/gpu.hpp:  } else if (n23 > GPU_MAX_THREADS) {
src/utils/gpu.hpp:    gpuRun3x1<<<dim3(n2, n1, n0), n3>>>(f);
src/utils/gpu.hpp:    GPU_Error_Check();
src/utils/gpu.hpp:  } else if (n123 > GPU_MAX_THREADS) {
src/utils/gpu.hpp:    gpuRun2x2<<<dim3(n1, n0), dim3(n3, n2)>>>(f);
src/utils/gpu.hpp:    GPU_Error_Check();
src/utils/gpu.hpp:    gpuRun1x3<<<n0, dim3(n3, n2, n1)>>>(f);
src/utils/gpu.hpp:    GPU_Error_Check();
src/utils/gpu.hpp:__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun2x3(const F f)
src/utils/gpu.hpp:__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun3x2(const F f)
src/utils/gpu.hpp:__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun4x1(const int n1, const F f)
src/utils/gpu.hpp:__global__ __launch_bounds__(GPU_MAX_THREADS) void gpuRun5x0(const int n1, const int n34, const int n4, const F f)
src/utils/gpu.hpp:void gpuFor(const int n0, const int n1, const int n2, const int n3, const int n4, const F f)
src/utils/gpu.hpp:  if (n4 > GPU_MAX_THREADS) {
src/utils/gpu.hpp:    const int b34 = (n34 + GPU_MAX_THREADS - 1) / GPU_MAX_THREADS;
src/utils/gpu.hpp:    gpuRun5x0<<<dim3(b34, n2, n01), t34>>>(n1, n34, n4, f);
src/utils/gpu.hpp:    GPU_Error_Check();
src/utils/gpu.hpp:  } else if (n34 > GPU_MAX_THREADS) {
src/utils/gpu.hpp:    gpuRun4x1<<<dim3(n3, n2, n01), n4>>>(n1, f);
src/utils/gpu.hpp:    GPU_Error_Check();
src/utils/gpu.hpp:  } else if (n2 * n34 > GPU_MAX_THREADS) {
src/utils/gpu.hpp:    gpuRun3x2<<<dim3(n2, n1, n0), dim3(n4, n3)>>>(f);
src/utils/gpu.hpp:    GPU_Error_Check();
src/utils/gpu.hpp:    gpuRun2x3<<<dim3(n1, n0), dim3(n4, n3, n2)>>>(f);
src/utils/gpu.hpp:    GPU_Error_Check();
src/utils/gpu.hpp:  #define GPU_LAMBDA [=] __device__
src/utils/gpu_arrays_functions.h:#ifndef GPU_ARRAY_FUNCTIONS_H
src/utils/gpu_arrays_functions.h:#define GPU_ARRAY_FUNCTIONS_H
src/utils/gpu_arrays_functions.h:#include "../global/global_cuda.h"
src/utils/gpu_arrays_functions.h:#include "../utils/gpu.hpp"
src/utils/gpu_arrays_functions.h:#include "../utils/gpu_arrays_functions.h"
src/utils/gpu_arrays_functions.h:void Extend_GPU_Array(T **current_array_d, int current_size, int new_size, bool print_out)
src/utils/gpu_arrays_functions.h:    std::cout << " Extending GPU Array, size: " << current_size << "  new_size: " << new_size << std::endl;
src/utils/gpu_arrays_functions.h:  GPU_Error_Check(cudaMemGetInfo(&global_free, &global_total));
src/utils/gpu_arrays_functions.h:  cudaDeviceSynchronize();
src/utils/gpu_arrays_functions.h:#ifdef PRINT_GPU_MEMORY
src/utils/gpu_arrays_functions.h:  printf("ReAllocating GPU Memory:  %ld  MB free \n", global_free / 1000000);
src/utils/gpu_arrays_functions.h:  GPU_Error_Check(cudaMalloc((void **)&new_array_d, new_size * sizeof(T)));
src/utils/gpu_arrays_functions.h:  cudaDeviceSynchronize();
src/utils/gpu_arrays_functions.h:  GPU_Error_Check();
src/utils/gpu_arrays_functions.h:    std::cout << " Error When Allocating New GPU Array" << std::endl;
src/utils/gpu_arrays_functions.h:  GPU_Error_Check(cudaMemcpy(new_array_d, *current_array_d, current_size * sizeof(T), cudaMemcpyDeviceToDevice));
src/utils/gpu_arrays_functions.h:  cudaDeviceSynchronize();
src/utils/gpu_arrays_functions.h:  GPU_Error_Check();
src/utils/gpu_arrays_functions.h:  cudaFree(*current_array_d);
src/utils/gpu_arrays_functions.h:  cudaDeviceSynchronize();
src/utils/gpu_arrays_functions.h:  GPU_Error_Check();
src/utils/reduction_utilities.cu: * \brief Contains the implementation of the GPU resident reduction utilities
src/utils/DeviceVector_tests.cu:void Check_Pointer_Attributes(cuda_utilities::DeviceVector<T> &devVector)
src/utils/DeviceVector_tests.cu:  cudaPointerAttributes ptrAttributes;
src/utils/DeviceVector_tests.cu:  GPU_Error_Check(cudaPointerGetAttributes(&ptrAttributes, devVector.data()));
src/utils/DeviceVector_tests.cu:      "that indicates type cudaMemoryTypeDevice. "
src/utils/DeviceVector_tests.cu:      "0 is cudaMemoryTypeUnregistered, "
src/utils/DeviceVector_tests.cu:      "1 is cudaMemoryTypeHost, and "
src/utils/DeviceVector_tests.cu:      "3 is cudaMemoryTypeManaged";
src/utils/DeviceVector_tests.cu:#else   // O_HIP is not defined i.e. we're using CUDA
src/utils/DeviceVector_tests.cu:  cuda_utilities::DeviceVector<double> devVector{vectorSize};
src/utils/DeviceVector_tests.cu:  cuda_utilities::DeviceVector<double> devVector{vectorSize};
src/utils/DeviceVector_tests.cu:  cudaPointerAttributes ptrAttributes;
src/utils/DeviceVector_tests.cu:  cudaPointerGetAttributes(&ptrAttributes, devVector.data());
src/utils/DeviceVector_tests.cu:      "that indicates type cudaMemoryTypeUnregistered"
src/utils/DeviceVector_tests.cu:      "0 is cudaMemoryTypeUnregistered, "
src/utils/DeviceVector_tests.cu:      "1 is cudaMemoryTypeHost, "
src/utils/DeviceVector_tests.cu:      "2 is cudaMemoryTypeDevice, and"
src/utils/DeviceVector_tests.cu:      "3 is cudaMemoryTypeManaged";
src/utils/DeviceVector_tests.cu:#else   // O_HIP is not defined i.e. we're using CUDA
src/utils/DeviceVector_tests.cu:  new (&devVector) cuda_utilities::DeviceVector<double>{vectorSize};
src/utils/DeviceVector_tests.cu:  cuda_utilities::DeviceVector<double> devVector{vectorSize};
src/utils/DeviceVector_tests.cu:  cuda_utilities::DeviceVector<double> devVector{vectorSize};
src/utils/DeviceVector_tests.cu:  cuda_utilities::DeviceVector<double> devVector{vectorSize};
src/utils/DeviceVector_tests.cu:  cuda_utilities::DeviceVector<double> devVector{vectorSize};
src/utils/DeviceVector_tests.cu:  cuda_utilities::DeviceVector<double> devVector{vectorSize};
src/utils/DeviceVector_tests.cu:  cuda_utilities::DeviceVector<double> devVector{vectorSize};
src/utils/DeviceVector_tests.cu:  cuda_utilities::DeviceVector<double> devVector{originalSize};
src/utils/DeviceVector_tests.cu:  cuda_utilities::DeviceVector<double> devVector{vectorSize};
src/utils/DeviceVector_tests.cu:  cuda_utilities::DeviceVector<double> devVector{vectorSize};
src/utils/DeviceVector_tests.cu:  cuda_utilities::DeviceVector<double> devVector{vectorSize};
src/utils/DeviceVector_tests.cu:  cuda_utilities::DeviceVector<double> devVector{vectorSize};
src/utils/reduction_utilities.h: * \brief Contains the declaration of the GPU resident reduction utilities
src/utils/reduction_utilities.h:#include "../global/global_cuda.h"
src/utils/reduction_utilities.h:#include "../utils/gpu.hpp"
src/utils/reduction_utilities.h:// This section handles the atomics. It is complicated because CUDA
src/utils/reduction_utilities.h: * potential race condition; the `cuda_utilities::setScalarDeviceMemory`
src/utils/reduction_utilities.h: * potential race condition; the `cuda_utilities::setScalarDeviceMemory`
src/utils/reduction_utilities.h: * `cuda_utilities::setScalarDeviceMemory` function exists for this
src/utils/reduction_utilities.h: * `cuda_utilities::setScalarDeviceMemory` function exists for this
src/utils/error_handling.cpp:#ifndef DISABLE_GPU_ERROR_CHECKING
src/utils/error_handling.cpp:  #warning "CUDA error checking is disabled. Enable it by compiling without the DISABLE_GPU_ERROR_CHECKING macro."
src/utils/error_handling.cpp:#endif  //! DISABLE_GPU_ERROR_CHECKING
src/utils/error_check_cuda.cu:/*! \file error_check_cuda.cu
src/utils/error_check_cuda.cu: *  \brief Error Check Cuda */
src/utils/error_check_cuda.cu:#include "../global/global_cuda.h"
src/utils/error_check_cuda.cu:#include "../utils/error_check_cuda.h"
src/utils/error_check_cuda.cu:#include "../utils/gpu.hpp"
src/utils/error_check_cuda.cu:  GPU_Error_Check(cudaMalloc((void **)&error_value_dev, sizeof(int)));
src/utils/error_check_cuda.cu:  GPU_Error_Check(cudaMemcpy(&error_value_host, error_value_dev, sizeof(int), cudaMemcpyDeviceToHost));
src/utils/cuda_utilities_tests.cpp: * \file cuda_utilities_tests.cpp
src/utils/cuda_utilities_tests.cpp: * (helenarichie@pitt.edu) \brief Tests for the contents of cuda_utilities.h and
src/utils/cuda_utilities_tests.cpp: * cuda_utilities.cpp
src/utils/cuda_utilities_tests.cpp:#include "../utils/cuda_utilities.h"
src/utils/cuda_utilities_tests.cpp:TEST(tHYDROCudaUtilsGetRealIndices, CorrectInputExpectCorrectOutput)
src/utils/cuda_utilities_tests.cpp:    cuda_utilities::Get_Real_Indices(parameters.n_ghost.at(i), parameters.nx.at(i), parameters.ny.at(i),
src/utils/cuda_utilities_tests.cpp:  cuda_utilities::compute3DIndices(id, nx, ny, testXid, testYid, testZid);
src/utils/cuda_utilities_tests.cpp:  testId = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);
src/utils/mhd_utilities.h:#include "../global/global_cuda.h"
src/utils/mhd_utilities.h:#include "../utils/cuda_utilities.h"
src/utils/mhd_utilities.h:#include "../utils/gpu.hpp"
src/utils/mhd_utilities.h:                                                          cuda_utilities::compute1DIndex(xid - 1, yid, zid, nx, ny)])
src/utils/mhd_utilities.h:                                                          cuda_utilities::compute1DIndex(xid, yid - 1, zid, nx, ny)])
src/utils/mhd_utilities.h:                                                          cuda_utilities::compute1DIndex(xid, yid, zid - 1, nx, ny)])
src/utils/gpu_arrays_functions.cu:#include "../global/global_cuda.h"
src/utils/gpu_arrays_functions.cu:#include "../utils/gpu.hpp"
src/utils/gpu_arrays_functions.cu:#include "../utils/gpu_arrays_functions.h"
src/utils/gpu_arrays_functions.cu:void Extend_GPU_Array_Real(Real **current_array_d, int current_size, int new_size, bool print_out)
src/utils/gpu_arrays_functions.cu:    std::cout << " Extending GPU Array, size: " << current_size << "  new_size: " << new_size << std::endl;
src/utils/gpu_arrays_functions.cu:  GPU_Error_Check(cudaMemGetInfo(&global_free, &global_total));
src/utils/gpu_arrays_functions.cu:  cudaDeviceSynchronize();
src/utils/gpu_arrays_functions.cu:#ifdef PRINT_GPU_MEMORY
src/utils/gpu_arrays_functions.cu:  printf("ReAllocating GPU Memory:  %d  MB free \n", (int)global_free / 1000000);
src/utils/gpu_arrays_functions.cu:  GPU_Error_Check(cudaMalloc((void **)&new_array_d, new_size * sizeof(Real)));
src/utils/gpu_arrays_functions.cu:  cudaDeviceSynchronize();
src/utils/gpu_arrays_functions.cu:  GPU_Error_Check();
src/utils/gpu_arrays_functions.cu:    std::cout << " Error When Allocating New GPU Array" << std::endl;
src/utils/gpu_arrays_functions.cu:  GPU_Error_Check(cudaMemcpy(new_array_d, *current_array_d, current_size * sizeof(Real), cudaMemcpyDeviceToDevice));
src/utils/gpu_arrays_functions.cu:  cudaDeviceSynchronize();
src/utils/gpu_arrays_functions.cu:  GPU_Error_Check();
src/utils/gpu_arrays_functions.cu:  // GPU_Error_Check( cudaMemGetInfo( &global_free_before, &global_total ) );
src/utils/gpu_arrays_functions.cu:  // cudaDeviceSynchronize();
src/utils/gpu_arrays_functions.cu:  cudaFree(*current_array_d);
src/utils/gpu_arrays_functions.cu:  cudaDeviceSynchronize();
src/utils/gpu_arrays_functions.cu:  GPU_Error_Check();
src/utils/gpu_arrays_functions.cu:  // GPU_Error_Check( cudaMemGetInfo( &global_free_after, &global_total ) );
src/utils/gpu_arrays_functions.cu:  // cudaDeviceSynchronize();
src/utils/error_check_cuda.h:/*! \file error_check_cuda.h
src/utils/error_check_cuda.h: *  \brief error_check_cuda.h */
src/utils/error_check_cuda.h:#ifndef ERROR_CHECK_CUDA_H
src/utils/error_check_cuda.h:#define ERROR_CHECK_CUDA_H
src/utils/error_check_cuda.h:#endif  // ERROR_CHECK_CUDA_H
src/utils/cuda_utilities.h: * \brief Contains the declaration of various utility functions for CUDA
src/utils/cuda_utilities.h:#include "../global/global_cuda.h"
src/utils/cuda_utilities.h:#include "../utils/gpu.hpp"
src/utils/cuda_utilities.h:namespace cuda_utilities
src/utils/cuda_utilities.h: * \brief Initialize GPU memory
src/utils/cuda_utilities.h: * \param[in] ptr The pointer to GPU memory
src/utils/cuda_utilities.h:inline void initGpuMemory(Real *ptr, size_t N) { GPU_Error_Check(cudaMemset(ptr, 0, N)); }
src/utils/cuda_utilities.h:    cudaOccupancyMaxPotentialBlockSize(&numBlocks, &threadsPerBlock, kernel, 0, 0);
src/utils/cuda_utilities.h:      // cudaOccupancyMaxPotentialBlockSize threadsPerBlock can be zero according to clang-tidy so this line sets it to
src/utils/cuda_utilities.h: * \brief Print the current GPU memory usage to standard out
src/utils/cuda_utilities.h:void Print_GPU_Memory_Usage(std::string const &additional_text = "");
src/utils/cuda_utilities.h:}  // end namespace cuda_utilities
src/utils/debug_utilities.cu:#include "../global/global_cuda.h"
src/utils/debug_utilities.cu:  GPU_Error_Check(cudaMalloc((void**)&out_bool, sizeof(bool)));
src/utils/debug_utilities.cu:  cudaMemcpy(out_bool, host_out_bool, sizeof(bool), cudaMemcpyHostToDevice);
src/utils/debug_utilities.cu:  cudaMemcpy(host_out_bool, out_bool, sizeof(bool), cudaMemcpyDeviceToHost);
src/utils/debug_utilities.cu:  cudaFree(out_bool);
src/utils/math_utilities.h:#include "../global/global_cuda.h"
src/utils/math_utilities.h:#include "../utils/gpu.hpp"
src/utils/DeviceVector.h:#include "../global/global_cuda.h"
src/utils/DeviceVector.h:#include "../utils/gpu.hpp"
src/utils/DeviceVector.h:namespace cuda_utilities
src/utils/DeviceVector.h:                "usage of functions like cudaMemcpy, cudaMemcpyPeer, cudaMemset");
src/utils/DeviceVector.h:   * \param[in] initialize (optional) If true then initialize the GPU
src/utils/DeviceVector.h:   * This method performs a cudaMemcpy to copy the desired element to the
src/utils/DeviceVector.h:   * cudaMemcpy to copy the desired element to the host then returns it.
src/utils/DeviceVector.h:    GPU_Error_Check(cudaMalloc(&_ptr, _size * sizeof(T)));
src/utils/DeviceVector.h:  void _deAllocate() { GPU_Error_Check(cudaFree(_ptr)); }
src/utils/DeviceVector.h:}  // namespace cuda_utilities
src/utils/DeviceVector.h:namespace cuda_utilities
src/utils/DeviceVector.h:    GPU_Error_Check(cudaMemset(_ptr, 0, _size * sizeof(T)));
src/utils/DeviceVector.h:  GPU_Error_Check(cudaMemcpyPeer(_ptr, 0, oldDevPtr, 0, count));
src/utils/DeviceVector.h:  GPU_Error_Check(cudaFree(oldDevPtr));
src/utils/DeviceVector.h:  GPU_Error_Check(cudaMemcpy(&hostValue, &(_ptr[index]), sizeof(T), cudaMemcpyDeviceToHost));
src/utils/DeviceVector.h:    // Use the overloaded [] operator to grab the value from GPU memory
src/utils/DeviceVector.h:  GPU_Error_Check(cudaMemcpy(&(_ptr[index]),  // destination
src/utils/DeviceVector.h:                             sizeof(T), cudaMemcpyHostToDevice));
src/utils/DeviceVector.h:    GPU_Error_Check(cudaMemcpy(_ptr, arrIn, arrSize * sizeof(T), cudaMemcpyHostToDevice));
src/utils/DeviceVector.h:    GPU_Error_Check(cudaMemcpy(arrOut, _ptr, _size * sizeof(T), cudaMemcpyDeviceToHost));
src/utils/DeviceVector.h:}  // end namespace cuda_utilities
src/mpi/mpi_routines.cpp:  #include "../mpi/cuda_mpi_routines.h"
src/mpi/mpi_routines.cpp:  // chprintf("ONLY_PARTICLES: Initializing without CUDA support.\n");
src/mpi/mpi_routines.cpp:  // // Needed to initialize cuda after gravity in order to work on Summit
src/mpi/mpi_routines.cpp:  // //initialize cuda for use with mpi
src/mpi/mpi_routines.cpp:  if (initialize_cuda_mpi(procID_node, nproc_node)) {
src/mpi/mpi_routines.cpp:    chprintf("Error initializing cuda with mpi.\n");
src/mpi/mpi_routines.cpp:  chprintf("Allocating MPI communication buffers on GPU ");
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_send_buffer_x0, xbsize * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_send_buffer_x1, xbsize * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_recv_buffer_x0, xbsize * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_recv_buffer_x1, xbsize * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_send_buffer_y0, ybsize * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_send_buffer_y1, ybsize * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_recv_buffer_y0, ybsize * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_recv_buffer_y1, ybsize * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_send_buffer_z0, zbsize * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_send_buffer_z1, zbsize * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_recv_buffer_z0, zbsize * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_recv_buffer_z1, zbsize * sizeof(Real)));
src/mpi/mpi_routines.cpp:  #if !defined(MPI_GPU)
src/mpi/mpi_routines.cpp:  // Whether or not MPI_GPU is on, the device has transfer buffers for
src/mpi/mpi_routines.cpp:  // PARTICLES_GPU
src/mpi/mpi_routines.cpp:  #if defined(PARTICLES) && defined(PARTICLES_GPU)
src/mpi/mpi_routines.cpp:      "Allocating MPI communication buffers on GPU for particle transfers ( "
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_send_buffer_x0_particles, buffer_length_particles_x0_send * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_send_buffer_x1_particles, buffer_length_particles_x1_send * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_send_buffer_y0_particles, buffer_length_particles_y0_send * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_send_buffer_y1_particles, buffer_length_particles_y1_send * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_send_buffer_z0_particles, buffer_length_particles_z0_send * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_send_buffer_z1_particles, buffer_length_particles_z1_send * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_recv_buffer_x0_particles, buffer_length_particles_x0_recv * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_recv_buffer_x1_particles, buffer_length_particles_x1_recv * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_recv_buffer_y0_particles, buffer_length_particles_y0_recv * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_recv_buffer_y1_particles, buffer_length_particles_y1_recv * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_recv_buffer_z0_particles, buffer_length_particles_z0_recv * sizeof(Real)));
src/mpi/mpi_routines.cpp:  GPU_Error_Check(cudaMalloc(&d_recv_buffer_z1_particles, buffer_length_particles_z1_recv * sizeof(Real)));
src/mpi/mpi_routines.cpp:  #endif  // PARTICLES && PARTICLES_GPU
src/mpi/mpi_routines.cpp:  // CPU relies on host buffers, GPU without MPI_GPU relies on host buffers
src/mpi/mpi_routines.cpp:    #if (defined(PARTICLES_GPU) && !defined(MPI_GPU)) || defined(PARTICLES_CPU)
src/mpi/mpi_routines.cpp:    #endif  // (defined(PARTICLES_GPU) && !defined(MPI_GPU)) ||
src/mpi/mpi_routines.cpp:      cudaMemcpy(d_recv_buffer_x0, h_recv_buffer_x0, xbsize * sizeof(Real), cudaMemcpyHostToDevice);
src/mpi/mpi_routines.cpp:      cudaMemcpy(d_recv_buffer_x1, h_recv_buffer_x1, xbsize * sizeof(Real), cudaMemcpyHostToDevice);
src/mpi/mpi_routines.cpp:      cudaMemcpy(d_recv_buffer_y0, h_recv_buffer_y0, ybsize * sizeof(Real), cudaMemcpyHostToDevice);
src/mpi/mpi_routines.cpp:      cudaMemcpy(d_recv_buffer_y1, h_recv_buffer_y1, ybsize * sizeof(Real), cudaMemcpyHostToDevice);
src/mpi/mpi_routines.cpp:      cudaMemcpy(d_recv_buffer_z0, h_recv_buffer_z0, zbsize * sizeof(Real), cudaMemcpyHostToDevice);
src/mpi/mpi_routines.cpp:      cudaMemcpy(d_recv_buffer_z1, h_recv_buffer_z1, zbsize * sizeof(Real), cudaMemcpyHostToDevice);
src/mpi/cuda_mpi_routines.cu:  #include "../mpi/cuda_mpi_routines.h"
src/mpi/cuda_mpi_routines.cu:  #include "../utils/gpu.hpp"
src/mpi/cuda_mpi_routines.cu:/*! \fn int initialize_cuda_mpi(int myid, int nprocs);
src/mpi/cuda_mpi_routines.cu: *  \brief CUDA initialization within MPI. */
src/mpi/cuda_mpi_routines.cu:int initialize_cuda_mpi(int myid, int nprocs)
src/mpi/cuda_mpi_routines.cu:  int i_device = 0;  // GPU device for this process
src/mpi/cuda_mpi_routines.cu:  int n_device;      // number of GPU devices available
src/mpi/cuda_mpi_routines.cu:  cudaError_t flag_error;
src/mpi/cuda_mpi_routines.cu:  // get the number of cuda devices
src/mpi/cuda_mpi_routines.cu:  flag_error = cudaGetDeviceCount(&n_device);
src/mpi/cuda_mpi_routines.cu:  if (flag_error != cudaSuccess) {
src/mpi/cuda_mpi_routines.cu:    if (flag_error == cudaErrorNoDevice) {
src/mpi/cuda_mpi_routines.cu:              "cudaGetDeviceCount: Error! for myid = %d and n_device = %d; "
src/mpi/cuda_mpi_routines.cu:              "cudaErrorNoDevice\n",
src/mpi/cuda_mpi_routines.cu:    if (flag_error == cudaErrorInsufficientDriver) {
src/mpi/cuda_mpi_routines.cu:              "cudaGetDeviceCount: Error! for myid = %d and n_device = %d; "
src/mpi/cuda_mpi_routines.cu:              "cudaErrorInsufficientDriver\n",
src/mpi/cuda_mpi_routines.cu:  // set a cuda device for each process
src/mpi/cuda_mpi_routines.cu:  cudaSetDevice(myid % n_device);
src/mpi/cuda_mpi_routines.cu:  cudaGetDevice(&i_device);
src/mpi/cuda_mpi_routines.cu:      "In initialize_cuda_mpi: name:%s myid = %d, i_device = %d, n_device = "
src/mpi/mpi_routines.h:/* Allocate MPI communication GPU buffers for a BLOCK decomposition */
src/mpi/cuda_mpi_routines.h:#ifndef CUDA_MPI_ROUTINES
src/mpi/cuda_mpi_routines.h:#define CUDA_MPI_ROUTINES
src/mpi/cuda_mpi_routines.h:/*! \fn int initialize_cuda_mpi(int myid, int nprocs);
src/mpi/cuda_mpi_routines.h: *  \brief CUDA initialization within MPI. */
src/mpi/cuda_mpi_routines.h:int initialize_cuda_mpi(int myid, int nprocs);
src/mpi/cuda_mpi_routines.h:#endif  // CUDA_MPI_ROUTINES
src/grid/mpi_boundaries.cpp:#include "../global/global_cuda.h"    //provides TPB
src/grid/mpi_boundaries.cpp:#include "../grid/cuda_boundaries.h"  // provides PackBuffers3D and UnpackBuffers3D
src/grid/mpi_boundaries.cpp:#include "../utils/gpu.hpp"
src/grid/mpi_boundaries.cpp:  #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_x0, d_send_buffer_x0, xbsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:    #ifdef GRAVITY_GPU
src/grid/mpi_boundaries.cpp:        buffer_length = Load_Gravity_Potential_To_Buffer_GPU(0, 0, d_send_buffer_x0, 0);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_x0, d_send_buffer_x0, xbsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:    #ifdef PARTICLES_GPU
src/grid/mpi_boundaries.cpp:        buffer_length = Load_Particles_Density_Boundary_to_Buffer_GPU(0, 0, d_send_buffer_x0);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_x0, d_send_buffer_x0, xbsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(d_send_buffer_x0, h_send_buffer_x0_particles, buffer_length * sizeof(Real), cudaMemcpyHostToDevice);
src/grid/mpi_boundaries.cpp:  #if defined(MPI_GPU)
src/grid/mpi_boundaries.cpp:  #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_x1, d_send_buffer_x1, xbsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:    #ifdef GRAVITY_GPU
src/grid/mpi_boundaries.cpp:        buffer_length = Load_Gravity_Potential_To_Buffer_GPU(0, 1, d_send_buffer_x1, 0);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_x1, d_send_buffer_x1, xbsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:    #ifdef PARTICLES_GPU
src/grid/mpi_boundaries.cpp:        buffer_length = Load_Particles_Density_Boundary_to_Buffer_GPU(0, 1, d_send_buffer_x1);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_x1, d_send_buffer_x1, xbsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(d_send_buffer_x1, h_send_buffer_x1_particles, buffer_length * sizeof(Real), cudaMemcpyHostToDevice);
src/grid/mpi_boundaries.cpp:  #if defined(MPI_GPU)
src/grid/mpi_boundaries.cpp:  #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_y0, d_send_buffer_y0, ybsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:    #ifdef GRAVITY_GPU
src/grid/mpi_boundaries.cpp:        buffer_length = Load_Gravity_Potential_To_Buffer_GPU(1, 0, d_send_buffer_y0, 0);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_y0, d_send_buffer_y0, ybsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:    #ifdef PARTICLES_GPU
src/grid/mpi_boundaries.cpp:        buffer_length = Load_Particles_Density_Boundary_to_Buffer_GPU(1, 0, d_send_buffer_y0);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_y0, d_send_buffer_y0, ybsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(d_send_buffer_y0, h_send_buffer_y0_particles, buffer_length * sizeof(Real), cudaMemcpyHostToDevice);
src/grid/mpi_boundaries.cpp:  #if defined(MPI_GPU)
src/grid/mpi_boundaries.cpp:  #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_y1, d_send_buffer_y1, ybsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:    #ifdef GRAVITY_GPU
src/grid/mpi_boundaries.cpp:        buffer_length = Load_Gravity_Potential_To_Buffer_GPU(1, 1, d_send_buffer_y1, 0);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_y1, d_send_buffer_y1, ybsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:    #ifdef PARTICLES_GPU
src/grid/mpi_boundaries.cpp:        buffer_length = Load_Particles_Density_Boundary_to_Buffer_GPU(1, 1, d_send_buffer_y1);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_y1, d_send_buffer_y1, ybsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(d_send_buffer_y1, h_send_buffer_y1_particles, buffer_length * sizeof(Real), cudaMemcpyHostToDevice);
src/grid/mpi_boundaries.cpp:  #if defined(MPI_GPU)
src/grid/mpi_boundaries.cpp:  #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_z0, d_send_buffer_z0, zbsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:    #ifdef GRAVITY_GPU
src/grid/mpi_boundaries.cpp:        buffer_length = Load_Gravity_Potential_To_Buffer_GPU(2, 0, d_send_buffer_z0, 0);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_z0, d_send_buffer_z0, zbsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:    #ifdef PARTICLES_GPU
src/grid/mpi_boundaries.cpp:        buffer_length = Load_Particles_Density_Boundary_to_Buffer_GPU(2, 0, d_send_buffer_z0);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_z0, d_send_buffer_z0, zbsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(d_send_buffer_z0, h_send_buffer_z0_particles, buffer_length * sizeof(Real), cudaMemcpyHostToDevice);
src/grid/mpi_boundaries.cpp:  #if defined(MPI_GPU)
src/grid/mpi_boundaries.cpp:  #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_z1, d_send_buffer_z1, zbsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:    #ifdef GRAVITY_GPU
src/grid/mpi_boundaries.cpp:        buffer_length = Load_Gravity_Potential_To_Buffer_GPU(2, 1, d_send_buffer_z1, 0);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_z1, d_send_buffer_z1, zbsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:    #ifdef PARTICLES_GPU
src/grid/mpi_boundaries.cpp:        buffer_length = Load_Particles_Density_Boundary_to_Buffer_GPU(2, 1, d_send_buffer_z1);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(h_send_buffer_z1, d_send_buffer_z1, zbsize * sizeof(Real), cudaMemcpyDeviceToHost);
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:        cudaMemcpy(d_send_buffer_z1, h_send_buffer_z1_particles, buffer_length * sizeof(Real), cudaMemcpyHostToDevice);
src/grid/mpi_boundaries.cpp:  #if defined(MPI_GPU)
src/grid/mpi_boundaries.cpp:  #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:    #ifdef GRAVITY_GPU
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:      #endif  // MPI_GPU
src/grid/mpi_boundaries.cpp:    Fptr_Unload_Gravity_Potential = &Grid3D::Unload_Gravity_Potential_from_Buffer_GPU;
src/grid/mpi_boundaries.cpp:    #endif  // GRAVITY_GPU
src/grid/mpi_boundaries.cpp:    #ifdef PARTICLES_GPU
src/grid/mpi_boundaries.cpp:      #ifndef MPI_GPU
src/grid/mpi_boundaries.cpp:    Fptr_Unload_Particle_Density = &Grid3D::Unload_Particles_Density_Boundary_From_Buffer_GPU;
src/grid/mpi_boundaries.cpp:      #ifdef MPI_GPU
src/grid/mpi_boundaries.cpp:      #endif  // MPI_GPU
src/grid/mpi_boundaries.cpp:    #endif  // PARTICLES_GPU
src/grid/cuda_boundaries.h:#include "../global/global_cuda.h"
src/grid/cuda_boundaries.h:#include "../utils/gpu.hpp"
src/grid/cuda_boundaries.h:void Wind_Boundary_CUDA(Real* c_device, int nx, int ny, int nz, int n_cells, int n_ghost, int x_off, int y_off,
src/grid/cuda_boundaries.h:void Noh_Boundary_CUDA(Real* c_device, int nx, int ny, int nz, int n_cells, int n_ghost, int x_off, int y_off,
src/grid/grid3D.h:#include "../global/global_cuda.h"
src/grid/grid3D.h:#ifdef CHEMISTRY_GPU
src/grid/grid3D.h:  #include "chemistry_gpu/chemistry_gpu.h"
src/grid/grid3D.h:#ifdef CHEMISTRY_GPU
src/grid/grid3D.h:  // Object that contains data for the GPU chemistry solver
src/grid/grid3D.h:  Chem_GPU Chem;
src/grid/grid3D.h:#ifdef CHEMISTRY_GPU
src/grid/grid3D.h:  #ifdef GRAVITY_GPU
src/grid/grid3D.h:  void Copy_Hydro_Density_to_Gravity_GPU();
src/grid/grid3D.h:  void Extrapolate_Grav_Potential_GPU();
src/grid/grid3D.h:  int Load_Gravity_Potential_To_Buffer_GPU(int direction, int side, Real *buffer, int buffer_start);
src/grid/grid3D.h:  void Unload_Gravity_Potential_from_Buffer_GPU(int direction, int side, Real *buffer, int buffer_start);
src/grid/grid3D.h:  void Set_Potential_Boundaries_Isolated_GPU(int direction, int side, int *flags);
src/grid/grid3D.h:  void Set_Potential_Boundaries_Periodic_GPU(int direction, int side, int *flags);
src/grid/grid3D.h:  #ifdef GRAVITY_GPU
src/grid/grid3D.h:  void Add_Analytic_Potential_GPU();
src/grid/grid3D.h:  #ifdef PARTICLES_GPU
src/grid/grid3D.h:  Real Calc_Particles_dt_GPU();
src/grid/grid3D.h:  void Advance_Particles_KDK_Step1_GPU();
src/grid/grid3D.h:  void Advance_Particles_KDK_Step2_GPU();
src/grid/grid3D.h:  void Set_Particles_Boundary_GPU(int dir, int side);
src/grid/grid3D.h:  void Set_Particles_Density_Boundaries_Periodic_GPU(int direction, int side);
src/grid/grid3D.h:  #endif  // PARTICLES_GPU
src/grid/grid3D.h:  #ifdef GRAVITY_GPU
src/grid/grid3D.h:  void Copy_Potential_From_GPU();
src/grid/grid3D.h:  void Copy_Particles_Density_to_GPU();
src/grid/grid3D.h:  void Copy_Particles_Density_GPU();
src/grid/grid3D.h:  int Load_Particles_Density_Boundary_to_Buffer_GPU(int direction, int side, Real *buffer);
src/grid/grid3D.h:  void Unload_Particles_Density_Boundary_From_Buffer_GPU(int direction, int side, Real *buffer);
src/grid/grid3D.h:  #endif  // GRAVITY_GPU
src/grid/grid3D.h:  void Change_GAS_Frame_System_GPU(bool forward);
src/grid/grid3D.h:  #ifdef PARTICLES_GPU
src/grid/grid3D.h:  void Advance_Particles_KDK_Cosmo_Step1_GPU();
src/grid/grid3D.h:  void Advance_Particles_KDK_Cosmo_Step2_GPU();
src/grid/grid3D.h:  #endif  // PARTICLES_GPU
src/grid/grid3D.h:#ifdef CHEMISTRY_GPU
src/grid/boundary_conditions.cpp:#include "../grid/cuda_boundaries.h"  // provides SetGhostCells
src/grid/boundary_conditions.cpp:  #ifdef GRAVITY_GPU
src/grid/boundary_conditions.cpp:        Set_Potential_Boundaries_Periodic_GPU(0, 0, flags);
src/grid/boundary_conditions.cpp:        Set_Potential_Boundaries_Periodic_GPU(0, 1, flags);
src/grid/boundary_conditions.cpp:        Set_Potential_Boundaries_Periodic_GPU(1, 0, flags);
src/grid/boundary_conditions.cpp:        Set_Potential_Boundaries_Periodic_GPU(1, 1, flags);
src/grid/boundary_conditions.cpp:        Set_Potential_Boundaries_Periodic_GPU(2, 0, flags);
src/grid/boundary_conditions.cpp:        Set_Potential_Boundaries_Periodic_GPU(2, 1, flags);
src/grid/boundary_conditions.cpp:  #ifdef GRAVITY_GPU
src/grid/boundary_conditions.cpp:        Set_Potential_Boundaries_Isolated_GPU(0, 0, flags);
src/grid/boundary_conditions.cpp:        Set_Potential_Boundaries_Isolated_GPU(0, 1, flags);
src/grid/boundary_conditions.cpp:        Set_Potential_Boundaries_Isolated_GPU(1, 0, flags);
src/grid/boundary_conditions.cpp:        Set_Potential_Boundaries_Isolated_GPU(1, 1, flags);
src/grid/boundary_conditions.cpp:        Set_Potential_Boundaries_Isolated_GPU(2, 0, flags);
src/grid/boundary_conditions.cpp:        Set_Potential_Boundaries_Isolated_GPU(2, 1, flags);
src/grid/boundary_conditions.cpp:  #endif  // GRAVITY_GPU
src/grid/boundary_conditions.cpp:  #ifdef PARTICLES_GPU
src/grid/boundary_conditions.cpp:        Set_Particles_Density_Boundaries_Periodic_GPU(0, 0);
src/grid/boundary_conditions.cpp:        Set_Particles_Density_Boundaries_Periodic_GPU(0, 1);
src/grid/boundary_conditions.cpp:        Set_Particles_Density_Boundaries_Periodic_GPU(1, 0);
src/grid/boundary_conditions.cpp:        Set_Particles_Density_Boundaries_Periodic_GPU(1, 1);
src/grid/boundary_conditions.cpp:        Set_Particles_Density_Boundaries_Periodic_GPU(2, 0);
src/grid/boundary_conditions.cpp:        Set_Particles_Density_Boundaries_Periodic_GPU(2, 1);
src/grid/boundary_conditions.cpp:  #ifdef PARTICLES_GPU
src/grid/boundary_conditions.cpp:        Set_Particles_Boundary_GPU(0, 0);
src/grid/boundary_conditions.cpp:        Set_Particles_Boundary_GPU(0, 1);
src/grid/boundary_conditions.cpp:        Set_Particles_Boundary_GPU(1, 0);
src/grid/boundary_conditions.cpp:        Set_Particles_Boundary_GPU(1, 1);
src/grid/boundary_conditions.cpp:        Set_Particles_Boundary_GPU(2, 0);
src/grid/boundary_conditions.cpp:        Set_Particles_Boundary_GPU(2, 1);
src/grid/boundary_conditions.cpp:  #endif  // PARTICLES_GPU
src/grid/boundary_conditions.cpp:  #ifdef PARTICLES_GPU
src/grid/boundary_conditions.cpp:      Particles.Set_Particles_Open_Boundary_GPU(dir / 2, dir % 2);
src/grid/boundary_conditions.cpp:  // from grid/cuda_boundaries.cu
src/grid/boundary_conditions.cpp:    // from grid/cuda_boundaries.cu
src/grid/boundary_conditions.cpp:    // from grid/cuda_boundaries.cu
src/grid/boundary_conditions.cpp:  // set x, y, & z offsets of local CPU volume to pass to GPU
src/grid/boundary_conditions.cpp:  Wind_Boundary_CUDA(C.device, H.nx, H.ny, H.nz, H.n_cells, H.n_ghost, x_off, y_off, z_off, H.dx, H.dy, H.dz, H.xbound,
src/grid/boundary_conditions.cpp:  // functions are in grid/cuda_boundaries.cu
src/grid/boundary_conditions.cpp:  // set x, y, & z offsets of local CPU volume to pass to GPU
src/grid/boundary_conditions.cpp:  Noh_Boundary_CUDA(C.device, H.nx, H.ny, H.nz, H.n_cells, H.n_ghost, x_off, y_off, z_off, H.dx, H.dy, H.dz, H.xbound,
src/grid/initial_conditions.cpp:    GPU_Error_Check(cudaMemcpy(C.device, C.density, H.n_fields * H.n_cells * sizeof(Real), cudaMemcpyHostToDevice));
src/grid/initial_conditions.cpp:  //       size_t const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);
src/grid/initial_conditions.cpp:        size_t const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);
src/grid/initial_conditions.cpp:        size_t const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);
src/grid/initial_conditions.cpp:  #ifdef CHEMISTRY_GPU
src/grid/initial_conditions.cpp:        int const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);
src/grid/initial_conditions.cpp:        int const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);
src/grid/initial_conditions.cpp:        int const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);
src/grid/initial_conditions.cpp:        int const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);
src/grid/initial_conditions.cpp:        int const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);
src/grid/initial_conditions.cpp:        int const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);
src/grid/initial_conditions.cpp:        int const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);
src/grid/initial_conditions.cpp:        int const id = cuda_utilities::compute1DIndex(i, j, k, H.nx, H.ny);
src/grid/grid3D.cpp:#include "../hydro/hydro_cuda.h"  // provides Calc_dt_GPU
src/grid/grid3D.cpp:#include "../integrators/VL_1D_cuda.h"
src/grid/grid3D.cpp:#include "../integrators/VL_2D_cuda.h"
src/grid/grid3D.cpp:#include "../integrators/VL_3D_cuda.h"
src/grid/grid3D.cpp:#include "../integrators/simple_1D_cuda.h"
src/grid/grid3D.cpp:#include "../integrators/simple_2D_cuda.h"
src/grid/grid3D.cpp:#include "../integrators/simple_3D_cuda.h"
src/grid/grid3D.cpp:  #include "../cooling/load_cloudy_texture.h"  // provides Load_Cuda_Textures and Free_Cuda_Textures
src/grid/grid3D.cpp:#ifdef COOLING_GPU
src/grid/grid3D.cpp:  #include "../cooling/cooling_cuda.h"  // provides Cooling_Update
src/grid/grid3D.cpp:  #include "../dust/dust_cuda.h"  // provides Dust_Update
src/grid/grid3D.cpp:  // ==Calculate the next inverse time step using Calc_dt_GPU from
src/grid/grid3D.cpp:  // hydro/hydro_cuda.h==
src/grid/grid3D.cpp:  return Calc_dt_GPU(C.device, H.nx, H.ny, H.nz, H.n_ghost, H.n_cells, H.dx, H.dy, H.dz, gama);
src/grid/grid3D.cpp:  GPU_Error_Check(cudaHostAlloc((void **)&C.host, H.n_fields * H.n_cells * sizeof(Real), cudaHostAllocDefault));
src/grid/grid3D.cpp:  GPU_Error_Check(cudaMalloc((void **)&C.device, H.n_fields * H.n_cells * sizeof(Real)));
src/grid/grid3D.cpp:  cuda_utilities::initGpuMemory(C.device, H.n_fields * H.n_cells * sizeof(Real));
src/grid/grid3D.cpp:  GPU_Error_Check(cudaHostAlloc(&C.Grav_potential, H.n_cells * sizeof(Real), cudaHostAllocDefault));
src/grid/grid3D.cpp:  GPU_Error_Check(cudaMalloc((void **)&C.d_Grav_potential, H.n_cells * sizeof(Real)));
src/grid/grid3D.cpp:#ifdef CHEMISTRY_GPU
src/grid/grid3D.cpp:  Load_Cuda_Textures();
src/grid/grid3D.cpp:  // set x, y, & z offsets of local CPU volume to pass to GPU
src/grid/grid3D.cpp:    VL_Algorithm_1D_CUDA(C.device, H.nx, x_off, H.n_ghost, H.dx, H.xbound, H.dt, H.n_fields, H.custom_grav);
src/grid/grid3D.cpp:    Simple_Algorithm_1D_CUDA(C.device, H.nx, x_off, H.n_ghost, H.dx, H.xbound, H.dt, H.n_fields, H.custom_grav);
src/grid/grid3D.cpp:    VL_Algorithm_2D_CUDA(C.device, H.nx, H.ny, x_off, y_off, H.n_ghost, H.dx, H.dy, H.xbound, H.ybound, H.dt,
src/grid/grid3D.cpp:    Simple_Algorithm_2D_CUDA(C.device, H.nx, H.ny, x_off, y_off, H.n_ghost, H.dx, H.dy, H.xbound, H.ybound, H.dt,
src/grid/grid3D.cpp:    VL_Algorithm_3D_CUDA(C.device, C.d_Grav_potential, H.nx, H.ny, H.nz, x_off, y_off, z_off, H.n_ghost, H.dx, H.dy,
src/grid/grid3D.cpp:    Simple_Algorithm_3D_CUDA(C.device, C.d_Grav_potential, H.nx, H.ny, H.nz, x_off, y_off, z_off, H.n_ghost, H.dx, H.dy,
src/grid/grid3D.cpp:#ifdef COOLING_GPU
src/grid/grid3D.cpp:  Timer.Cooling_GPU.Start();
src/grid/grid3D.cpp:  // ==Apply Cooling from cooling/cooling_cuda.h==
src/grid/grid3D.cpp:  Timer.Cooling_GPU.End();
src/grid/grid3D.cpp:#endif  // COOLING_GPU
src/grid/grid3D.cpp:  // ==Apply dust from dust/dust_cuda.h==
src/grid/grid3D.cpp:#ifdef CHEMISTRY_GPU
src/grid/grid3D.cpp:  // ==Calculate the next time step using Calc_dt_GPU from hydro/hydro_cuda.h==
src/grid/grid3D.cpp:  GPU_Error_Check(cudaFreeHost(C.host));
src/grid/grid3D.cpp:  GPU_Error_Check(cudaFreeHost(C.Grav_potential));
src/grid/grid3D.cpp:  GPU_Error_Check(cudaFree(C.d_Grav_potential));
src/grid/grid3D.cpp:  #ifdef GRAVITY_GPU
src/grid/grid3D.cpp:  Grav.FreeMemory_GPU();
src/grid/grid3D.cpp:#ifdef COOLING_GPU
src/grid/grid3D.cpp:  Free_Cuda_Textures();
src/grid/grid3D.cpp:#ifdef CHEMISTRY_GPU
src/grid/cuda_boundaries.cu:#include "../global/global_cuda.h"
src/grid/cuda_boundaries.cu:#include "../utils/cuda_utilities.h"
src/grid/cuda_boundaries.cu:#include "../utils/gpu.hpp"
src/grid/cuda_boundaries.cu:#include "cuda_boundaries.h"
src/grid/cuda_boundaries.cu:  GPU_Error_Check(cudaDeviceSynchronize());
src/grid/cuda_boundaries.cu:  // calculate ghost cell ID and i,j,k in GPU grid
src/grid/cuda_boundaries.cu:  // not true i,j,k but relative i,j,k in the GPU grid
src/grid/cuda_boundaries.cu:  cuda_utilities::compute3DIndices(id, n_ghost, ny, xid, yid, zid);
src/grid/cuda_boundaries.cu:  // calculate ghost cell ID and i,j,k in GPU grid
src/grid/cuda_boundaries.cu:  // not true i,j,k but relative i,j,k in the GPU grid
src/grid/cuda_boundaries.cu:void Wind_Boundary_CUDA(Real *c_device, int nx, int ny, int nz, int n_cells, int n_ghost, int x_off, int y_off,
src/grid/cuda_boundaries.cu:void Noh_Boundary_CUDA(Real *c_device, int nx, int ny, int nz, int n_cells, int n_ghost, int x_off, int y_off,
src/grid/grid_enum.h:  #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)

```
