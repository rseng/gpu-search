# https://github.com/artis-mcrt/artis

```console
Makefile:ifeq ($(GPU),ON)
Makefile:	CXXFLAGS += -DGPU_ON=true -DUSE_SIMPSON_INTEGRATOR=true
Makefile:	BUILD_DIR := $(BUILD_DIR)_gpu
Makefile:else ifeq ($(GPU),OFF)
Makefile:else ifeq ($(GPU),)
Makefile:    $(error bad value for GPU option. Should be ON or OFF)
Makefile:		ifeq ($(GPU),ON)
Makefile:			CXXFLAGS += -mp=gpu -gpu=mem:unified
Makefile:		ifeq ($(GPU),ON)
Makefile:			CXXFLAGS += -stdpar=gpu -gpu=mem:unified
Makefile:			# CXXFLAGS += -gpu=cc80
input.cc:#ifndef GPU_ON
input.cc:#ifndef GPU_ON
input.cc:#if defined(_OPENMP) && !defined(GPU_ON)
input.cc:#ifndef GPU_ON
update_packets.cc:#ifdef GPU_ON
exspec.cc:#ifndef GPU_ON
exspec.cc:#ifndef GPU_ON
sn3d.h:#ifndef GPU_ON
sn3d.h:#ifdef __NVCOMPILER_CUDA_ARCH__
sn3d.h:#ifndef GPU_ON
sn3d.h:// if not set, force Simpson integrator on GPU mode (since gsl doesn't work there!)
sn3d.h:#ifndef GPU_ON
sn3d.h:#ifdef __NVCOMPILER_CUDA_ARCH__
sn3d.h:#ifndef GPU_ON
macroatom.cc:#if (defined(STDPAR_ON) || defined(_OPENMP_ON)) && !defined(GPU_ON)
sn3d.cc:#ifndef GPU_ON
sn3d.cc:#ifndef GPU_ON
sn3d.cc:#if defined(_OPENMP) && !defined(GPU_ON)
sn3d.cc:#ifdef GPU_ON
sn3d.cc:  printout("GPU_ON is enabled\n");

```
