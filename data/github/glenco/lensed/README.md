# https://github.com/glenco/lensed

```console
docs/dependencies.md:-   [OpenCL](#opencl) .
docs/dependencies.md:OpenCL
docs/dependencies.md:using [OpenCL](https://www.khronos.org/opencl/) to communicate with both CPU
docs/dependencies.md:and GPU devices through a unified programming platform.
docs/dependencies.md:A OpenCL runtime library for the compute devices must present in the system.
docs/dependencies.md:Such a library usually comes with the driver of an OpenCL-enabled device. When
docs/dependencies.md:building Lensed from source, it is further necessary to have the OpenCL headers
docs/dependencies.md:For the OpenCL runtime library, it is necessary to install the device drivers
docs/dependencies.md:for the CPUs/GPUs present in the system. Please refer to the Ubuntu manual for
docs/dependencies.md:The OpenCL headers necessary for compiling Lensed can be found in package
docs/dependencies.md:`opencl-headers`.
docs/dependencies.md:$ sudo apt-get install opencl-headers
docs/dependencies.md:-   [Intel OpenCL Technology](https://software.intel.com/en-us/intel-opencl) ,
docs/dependencies.md:-   [AMD APP SDK](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/) ,
docs/dependencies.md:-   [Nvidia OpenCL](https://developer.nvidia.com/opencl) .
docs/dependencies.md:Mac OS X ships with the libraries and headers required to build OpenCL programs
docs/configuration.md:GPU device, in case one exists. Alternatively, it will select the first device
docs/troubleshooting.md:	libOpenCL.so.1 => /usr/lib/fglrx/libOpenCL.so.1
docs/troubleshooting.md:	/System/Library/Frameworks/OpenCL.framework/Versions/A/OpenCL
docs/building.md:-   OpenCL headers and runtime library .
docs/building.md:| `OPENCL_DIR`            | path to the OpenCL implementation                 |
docs/building.md:| `OPENCL_INCLUDE_DIR`    | path to the `CL/cl.h` header                      |
docs/building.md:| `OPENCL_LIB_DIR`        | path to the OpenCL library                        |
docs/building.md:| `OPENCL_LIB`            | OpenCL runtime library (e.g. `-lOpenCL`)          |
docs/building.md:OPENCL_INCLUDE_DIR = 
docs/building.md:OPENCL_LIB_DIR = 
docs/building.md:OPENCL_LIB = -framework OpenCL
docs/building.md:### OpenCL
docs/building.md:OpenCL headers and library. It is not necessary to change the OpenCL settings.
docs/building.md:In order to build Lensed, the compiler needs the OpenCL header `CL/cl.h` and
docs/building.md:the OpenCL runtime library.
docs/building.md:If these are provided by a vendor SDK, it can be enough to set the `OPENCL_DIR`
docs/building.md:$ make OPENCL_DIR="/usr/local/cuda"
docs/building.md:This sets the variables `OPENCL_INCLUDE_DIR` and `OPENCL_LIB_DIR` to default
docs/building.md:values of `$OPENCL_DIR/include` and `$OPENCL_DIR/lib`, respectively.
docs/building.md:Alternatively, it is possible to set `OPENCL_INCLUDE_DIR` and `OPENCL_LIB_DIR`
docs/building.md:manually to the paths containing `CL/cl.h` (note the subfolder) and the OpenCL
docs/building.md:library (usually `libOpenCL`), respectively.
docs/building.md:$ make OPENCL_INCLUDE_DIR="$HOME/headers" OPENCL_LIB_DIR="$HOME/libraries"
docs/building.md:The name of the OpenCL runtime library can be set by the `OPENCL_LIB` variable.
docs/building.md:This can be a linker flag such as `-lOpenCL` or an explicit path to a library.
docs/building.md:$ make OPENCL_LIB="$HOME/experimental-driver/libOpenCL.so"
docs/releases.md:-   [OpenCL drivers](dependencies.md#opencl)
docs/releases.md:-   OpenCL header files
CHANGELOG.md:  * fix OpenCL compile errors due to INFINITY macro
CHANGELOG.md:  * fix OpenCL compatibility issues when building
CHANGELOG.md:  * simple profiling for OpenCL functions
CHANGELOG.md:  * added OpenCL to list of requirements
CHANGELOG.md:  * OpenCL device selection
CHANGELOG.md:  * Limit block size by OpenCL device capabilities.
CHANGELOG.md:  * GPU option added.
CHANGELOG.md:  * Optimised GPU performance.
CHANGELOG.md:  * Notifications from OpenCL already include build log.
CHANGELOG.md:  * Generic OpenCL header inclusion.
CHANGELOG.md:  * Add notification callback to OpenCL context.
CHANGELOG.md:  * Working OpenCL version.
Makefile:#   OPENCL_DIR                                                       #
Makefile:#     path to the OpenCL implementation                              #
Makefile:#   OPENCL_INCLUDE_DIR                                               #
Makefile:#   OPENCL_LIB_DIR                                                   #
Makefile:#     path to the OpenCL library                                     #
Makefile:#   OPENCL_LIB                                                       #
Makefile:#     OpenCL runtime library (e.g. `-lOpenCL`)                       #
Makefile:          opencl.h \
Makefile:          opencl.c \
Makefile:# system-dependent OpenCL library
Makefile:OPENCL_LIB_Linux = -lOpenCL
Makefile:OPENCL_LIB_Darwin = -framework OpenCL
Makefile:ifndef OPENCL_LIB
Makefile:OPENCL_LIB = $(OPENCL_LIB_$(OS))
Makefile:ifdef OPENCL_DIR
Makefile:OPENCL_INCLUDE_DIR = $(OPENCL_DIR)/include
Makefile:OPENCL_LIB_DIR = $(OPENCL_DIR)/lib
Makefile:ifdef OPENCL_INCLUDE_DIR
Makefile:CFLAGS += -I$(OPENCL_INCLUDE_DIR)
Makefile:ifdef OPENCL_LIB_DIR
Makefile:LDFLAGS += -L$(OPENCL_LIB_DIR) -Wl,-rpath,$(OPENCL_LIB_DIR)
Makefile:LDLIBS += $(OPENCL_LIB)
Makefile:	@$(ECHO) "OPENCL_INCLUDE_DIR = $(OPENCL_INCLUDE_DIR)" >> $(CACHE)
Makefile:	@$(ECHO) "OPENCL_LIB_DIR = $(OPENCL_LIB_DIR)" >> $(CACHE)
Makefile:	@$(ECHO) "OPENCL_LIB = $(OPENCL_LIB)" >> $(CACHE)
.travis.yml:    - opencl-headers
.travis.yml:    curl http://registrationcenter-download.intel.com/akdlm/irc_nas/9019/opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz | tar xz
.travis.yml:    cd opencl_runtime_*
src/input.c:        printf("  %-20s  %s\n", "--profile", "Enable OpenCL profiling.");
src/opencl.h:// use OpenCL 1.2 API
src/opencl.h:#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
src/opencl.h:// OpenCL headers
src/opencl.h:#include <OpenCL/opencl.h>
src/opencl.h:// OpenCL device
src/opencl.h:// OpenCL environment
src/opencl.h:// list available OpenCL devices
src/opencl.h:// get Lensed OpenCL environment
src/opencl.h:// free Lensed OpenCL environment
src/input/objects.c:#include "../opencl.h"
src/input/objects.c:    // OpenCL
src/lensed.c:#include "opencl.h"
src/lensed.c:    // OpenCL error code
src/lensed.c:    // OpenCL structures
src/lensed.c:    // OpenCL device info
src/lensed.c:            err = clGetDeviceInfo(d->device_id, CL_DEVICE_OPENCL_C_VERSION, sizeof(device_compiler), device_compiler, NULL);
src/lensed.c:        // get the OpenCL environment
src/lensed.c:            verbose("    type: %s", device_type == CL_DEVICE_TYPE_CPU ? "CPU" : (device_type == CL_DEVICE_TYPE_GPU ? "GPU" : "(unknown)"));
src/lensed.c:            err = clGetDeviceInfo(lcl->device_id, CL_DEVICE_OPENCL_C_VERSION, sizeof(device_compiler), device_compiler, NULL);
src/lensed.c:        // get work group size multiple for kernel if OpenCL version > 1.0
src/lensed.c:            // fixed work group size multiple of 16 for OpenCL 1.0
src/lensed.c:        // get work group size multiple for kernel if OpenCL version > 1.0
src/lensed.c:            // fixed work group size multiple of 16 for OpenCL 1.0
src/nested.c:#include "opencl.h"
src/profile.c:#include "opencl.h"
src/kernel.c:static const char OPENCL_EXT[] = ".cl";
src/kernel.c:    filename = malloc(strlen(LENSED_PATH) + strlen(KERNEL_DIR) + strlen(name) + strlen(OPENCL_EXT) + 1);
src/kernel.c:    sprintf(filename, "%s%s%s%s", LENSED_PATH, KERNEL_DIR, name, OPENCL_EXT);
src/kernel.c:    filename = malloc(strlen(LENSED_PATH) + strlen(OBJECT_DIR) + strlen(name) + strlen(OPENCL_EXT) + 1);
src/kernel.c:    sprintf(filename, "%s%s%s%s", LENSED_PATH, OBJECT_DIR, name, OPENCL_EXT);
src/quadrature.c:#include "opencl.h"
src/data.c:#include "opencl.h"
src/opencl.c:#include "opencl.h"
src/opencl.c:    // OpenCL error code
src/opencl.c:    // counter for GPUs and CPUs
src/opencl.c:    unsigned ngpu = 0;
src/opencl.c:        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, 0, NULL, &ndevices);
src/opencl.c:        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, 0, NULL, &ndevices);
src/opencl.c:        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, ndevices, devices, NULL);
src/opencl.c:                case CL_DEVICE_TYPE_GPU: sprintf(item->name, "gpu%u", ngpu++); break;
src/opencl.c:    // OpenCL error code
src/opencl.c:            // try to find GPU device
src/opencl.c:                if(device->device_type == CL_DEVICE_TYPE_GPU)
src/opencl.c:            // if no GPU was found, use first device
src/opencl.c:    // return the OpenCL environment

```
