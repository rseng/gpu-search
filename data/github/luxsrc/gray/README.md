# https://github.com/luxsrc/gray

```console
sim/initcond.h:#include <lux/opencl.h>
sim/gray.h:#include <lux/opencl.h>
sim/gray.h:	Lux_opencl *ocl;
sim/gray.c:	lux_print("GRay2:init: setup opencl module\n");
sim/gray.c:		struct LuxOopencl opts = OPENCL_NULL;
sim/gray.c:		EGO->ocl = lux_load("opencl", &opts);
sim/gray/Kerr/ocl.c:#include <lux/opencl.h>
sim/gray/Kerr/ocl.c:	Lux_opencl       *ocl  = NULL;
sim/gray/Kerr/ocl.c:	struct LuxOopencl opts = OPENCL_NULL;
sim/gray/Kerr/ocl.c:	ocl = lux_load("opencl", &opts);
README.md:wide range of modern hardware/accelerators such as GPUs and Intel&reg;
sim-org/RK4.cl: ** GRay2 uses OpenCL's just-in-time compilation feature to implement
sim-org/RK4.cl: ** OpenCL implementation of the classical 4th-order Runge-Kutta integrator
sim-org/preamble.cl: ** Preamble: useful OpenCL macros and functions
sim-org/preamble.cl: ** GRay2 uses OpenCL's just-in-time compilation feature to implement
sim-org/preamble.cl: ** OpenCL macros and functions that help implementing the other parts
sim-org/evolve.c:	Lux_opencl        *ocl    = EGO->ocl;
sim-org/evolve.c:	Lux_opencl_kernel *evolve = EGO->evolve;
sim-org/Makefile:	lux-build gray.h *.c -f opencl hdf5 -o gray.la
sim-org/driver.cl: ** GRay2 uses OpenCL's just-in-time compilation feature to implement
sim-org/driver.cl:/** OpenCL driver kernel for initializing states */
sim-org/driver.cl:/** OpenCL driver kernel for integrating the geodesic equations */
sim-org/icond.c:	Lux_opencl *ocl = EGO->ocl;
sim-org/icond.c:	Lux_opencl_kernel *icond;
sim-org/gray.h:#include <lux/opencl.h>
sim-org/gray.h:	Lux_opencl *ocl;
sim-org/gray.h:	Lux_opencl_kernel *evolve;
sim-org/gray.h:	/* We need these quantities to convert from unnormalized OpenCL coordiantes
sim-org/gray.h:/** Build the OpenCL module for GRay2 */
sim-org/gray.h:extern Lux_opencl *build(Lux_job *);
sim-org/build.c:Lux_opencl *
sim-org/build.c:	/** \page newkern New OpenCL Kernels
sim-org/build.c:	 ** Extend GRay2 by adding new OpenCL kernels
sim-org/build.c:	 ** GRay2 uses the just-in-time compilation feature of OpenCL
sim-org/build.c:	 ** level OpenCL codes are actually in a lux module called
sim-org/build.c:	 ** "opencl".  GRay2 developers simply need to load this
sim-org/build.c:	 ** module with a list of OpenCL source codes, e.g.,
sim-org/build.c:	 **   struct LuxOopencl otps = {..., flags, src};
sim-org/build.c:	 **   Lux_opencl *ocl = lux_load("opencl", &opts);
sim-org/build.c:	 ** and then an OpenCL kernel can be obtained and run by
sim-org/build.c:	 **   Lux_opencl_kernel *icond = ocl->mkkern(ocl, "icond_drv");
sim-org/build.c:	 ** straightforward to add a new OpenCL kernels to GRay2:
sim-org/build.c:	 ** -# Name the OpenCL source code with an extension ".cl" and
sim-org/build.c:	 ** sure that the new OpenCL source code is compatible with
sim-org/build.c:	 ** other OpenCL codes.  This is because GRay2 place all the
sim-org/build.c:	 ** OpenCL codes together and build them as a single program.
sim-org/build.c:	struct LuxOopencl opts = OPENCL_NULL;
sim-org/build.c:	return lux_load("opencl", &opts);
sim-org/SoA.cl: ** GRay2 uses OpenCL's just-in-time compilation feature to implement
sim-org/gray.c:	Lux_opencl *ocl; /* to be loaded */
sim-org/gray.c:	Lux_opencl *ocl = EGO->ocl;
sim-org/io.c:	Lux_opencl *ocl = EGO->ocl;
sim-org/io.c:	/* OpenCL Image properties */
sim-org/io.c:	/* OpenCL Image properties */
sim-org/io.c:				/* https://streamhpc.com/blog/2013-04-28/opencl-error-codes/ */
sim-org/io.c:			/* https://streamhpc.com/blog/2013-04-28/opencl-error-codes/ */
sim-org/io.c:		/* https://streamhpc.com/blog/2013-04-28/opencl-error-codes/ */
sim-org/AoS.cl: ** GRay2 uses OpenCL's just-in-time compilation feature to implement
sim-org/interp.cl: ** When fisheye coordinates are used, it is critical to correct how OpenCL
sim-org/interp.cl: * unnormalized, OpenCL ones.  The values are always stored in the .s123 slots
sim-org/interp.cl:  /* Return the OpenCL unnormalized coordinates uvw that, if plugged in the
sim-org/interp.cl:  /* Finally, we have to offset by 0.5.  This 0.5 is very important because OpenCL

```
