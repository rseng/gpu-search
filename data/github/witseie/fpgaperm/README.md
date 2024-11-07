# https://github.com/witseie/fpgaperm

```console
Makefile:LDFLAGS = -lOpenCL -lpthread -lrt -lstdc++ -lgomp -L$(XILINX_XRT)/lib/
README.md:In order to compile the host application, the OpenCL, OpenMP and Xilinx XRT libraries must be installed.
README.md:### OpenCL installation
README.md:sudo apt-get install ocl-icd-libopencl1
README.md:sudo apt-get install opencl-headers
README.md:sudo apt-get install ocl-icd-opencl-dev
README.md:sudo yum install opencl-headers
host_app/include/OclApi.h:#define CL_HPP_TARGET_OPENCL_VERSION 120
host_app/include/OclApi.h:#define CL_HPP_MINIMUM_OPENCL_VERSION 120
host_app/include/OclApi.h:#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
host_app/include/OclApi.h:#include "CL/opencl.h"
host_app/include/Types.h:#include "CL/opencl.h"

```
