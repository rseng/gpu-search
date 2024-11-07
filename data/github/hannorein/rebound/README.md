# https://github.com/hannorein/rebound

```console
rebound/widget.py:            #TODO: Implement better GPU size change
legacy/gravity_opencl.c:#include <OpenCL/cl.h>
legacy/gravity_opencl.c:#warning GRAVITY_OPENCL might not work with MPI for your problem. 
legacy/gravity_opencl.c:#pragma OPENCL EXTENSION cl_khr_fp64 : enable
legacy/gravity_opencl.c:#pragma OPENCL EXTENSION cl_amd_fp64 : enable
legacy/gravity_opencl.c:"void gravity_opencl_kernel(				\n"
legacy/gravity_opencl.c:		OCL_CREATE_DEVICE(platforms[0],CL_DEVICE_TYPE_GPU|CL_DEVICE_TYPE_ACCELERATOR,device_list);
legacy/gravity_opencl.c:		printf("\n\nOpenCL Setup.\n----------\n");
legacy/gravity_opencl.c:		kernel = clCreateKernel(program,"gravity_opencl_kernel",&clStatus);
legacy/opencl/Makefile:export LIB=-framework OpenCL
legacy/opencl/Makefile:	ln -fs gravity_opencl.c ../../src/gravity.c
legacy/opencl/problem.c: * @brief 	Example problem: opencl.
legacy/opencl/problem.c: * the OpenCL direct gravity summation module.
legacy/opencl/problem.c: * This is a very simple implementation (see `gravity_opencl.c`). 
legacy/opencl/problem.c: * transfers the data back and forth from the GPU every timestep.
legacy/opencl/problem.c: * `make && ./rebound`, which will run on the GPU.
src/glad.h:#define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117
src/display.c:    // Update data on GPU 

```
