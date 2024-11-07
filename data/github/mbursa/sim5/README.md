# https://github.com/mbursa/sim5

```console
Makefile:cuda: lib-clean nvcc-check
Makefile:#    $(error "FAILED PREREQUISITY: NVCC has not been found on the system PATH, install the NVIDIA CUDA SDK")
doc/doxygen.xsl:        <xsl:text>SIM5 is a collection of C routines for relativistic raytracing and radiation transfer. It has a special focus on raytracing from accretion disks, tori, hot spots or any custom 3D configuration of matter in Kerr geometry, but it can be used with any other metrics as well. It can handle both optically thick and thin sources as well as transport of polarization properties and helps to calculate the propagation of light rays from the source to an observer through a curved spacetimes. The library is threas-safe (with a few documented exceptions) compiles with Nvidia CUDA compiler which opens the door to massive parallelization using GPUs.</xsl:text>
doc/sim5lib-doc.md:SIM5 is a collection of C routines for relativistic raytracing and radiation transfer. It has a special focus on raytracing from accretion disks, tori, hot spots or any custom 3D configuration of matter in Kerr geometry, but it can be used with any other metrics as well. It can handle both optically thick and thin sources as well as transport of polarization properties and helps to calculate the propagation of light rays from the source to an observer through a curved spacetimes. The library is threas-safe (with a few documented exceptions) compiles with Nvidia CUDA compiler which opens the door to massive parallelization using GPUs.
doc/sim5lib-doc.md:NOTE: This unit uses some static variables to store some persistent information and due to that it is NOT thread-safe. For the same reasons, the routines declared here are not available to CUDA. 
doc/sim5lib-doc.md:NOTE: This unit uses static variables to store persistent information about the linked library. As a result, routines in this module are NOT thread-safe in a sense different threads cannot each link a different library. They can, however, all make calls to the already linked library. For the same reasons, the routines declared here are not available to CUDA. 
README.md:SIM5 is a C library with a Python interface that contains a collection of routines for relativistic raytracing and radiation transfer in GR. It has a special focus on raytracing from accretion disks, tori, hot spots or any other 3D configuration of matter in Kerr geometry, but it can be used with any other metric as well. It can handle both optically thick and thin sources as well as transport of polarization of the radiation and helps to calculate the propagation of light rays from the source to an observer through a curved spacetime. It supports parallelization and runs on GPUs.
README.md:The library is thread-safe and supports parallelization of the calculations at both CPUs (OpenMP/MPI) and GPUs (CUDA/OpenCL). 
README.md:As the library is a collection of routines, it does not have any parallelization built in itself, but it is very easy to write CPU/GPU parallel code with it. SIM5 uses strict encapsulation and all its functions use only the data that is passed in as parameters; there are no global variables in the library. This allows to write thread-safe codes that can be straightforwardly parallelized. Examples for OpenMP, MPI and CUDA parallelization are provided in the `examples` folder.
src/sim5config.h.default:#ifdef __CUDACC__
src/sim5config.h.default:    #define CUDA
src/sim5lib.h:#ifndef CUDA
src/sim5lib.h:#ifndef CUDA
src/sim5lib.h:#ifndef CUDA
src/sim5math.h:#ifndef CUDA
src/sim5math.h:#ifdef CUDA
src/sim5disk-nt.h:#ifndef CUDA
src/sim5disk-nt.h:#endif //CUDA
src/sim5lib.c:#ifndef CUDA
src/sim5lib.c:#ifndef CUDA
src/sim5lib.c:#ifndef CUDA
src/sim5interpolation.c:#ifndef CUDA
src/sim5interpolation.c:    //#ifndef CUDA
src/sim5interpolation.c:        //#ifndef CUDA
src/sim5interpolation.c:                //#ifndef CUDA
src/sim5interpolation.c:                //#ifndef CUDA
src/sim5interpolation.c:            //#ifndef CUDA
src/sim5interpolation.c:        //#ifndef CUDA
src/sim5interpolation.c:        //#ifndef CUDA
src/sim5interpolation.c:        //#ifndef CUDA
src/sim5interpolation.c:            //#ifndef CUDA
src/sim5interpolation.c:#endif //CUDA
src/sim5math.c:// see http://stackoverflow.com/questions/11832202/cuda-random-number-generating
src/sim5math.c:// and http://cs.umw.edu/~finlayson/class/fall16/cpsc425/notes/cuda-random.html
src/sim5math.c:    #ifndef CUDA
src/sim5math.c:    #ifndef CUDA
src/sim5math.c:    #ifndef CUDA
src/sim5math.c:#ifdef CUDA
src/sim5math.c:#ifdef CUDA
src/sim5math.c:#ifdef CUDA
src/sim5math.c:#ifdef CUDA
src/sim5distributions.h:#ifndef CUDA
src/sim5distributions.h:#endif //CUDA
src/sim5raytrace.c:#ifndef CUDA
src/sim5raytrace.c:    #ifndef CUDA
src/sim5raytrace.c:#ifdef CUDA
src/sim5raytrace.c:    #ifndef CUDA    
src/sim5raytrace.c:    // - for CUDA it cannot be nested, so it is taked out of raytrace()
src/sim5raytrace.c:            #ifdef CUDA
src/sim5raytrace.c:        #ifdef CUDA
src/mt19937/mt19937.c:#ifndef __CUDACC__
src/mt19937/mt19937.c:#endif //CUDA
src/mt19937/mt19937.h:#ifndef __CUDACC__
src/mt19937/mt19937.h:#endif //CUDA
src/sim5utils.h:#ifndef CUDA
src/sim5utils.c:    #ifndef CUDA
src/sim5utils.c:    #ifndef CUDA
src/sim5utils.c:    #ifndef CUDA
src/sim5utils.c:#ifndef CUDA
src/sim5lib.cu:#ifndef CUDA
src/sim5lib.cu:#ifndef CUDA
src/sim5lib.cu:#ifndef CUDA
src/sim5include.h:#ifndef CUDA
src/sim5kerr.c:    #ifndef CUDA
src/sim5kerr.c:    #ifndef CUDA
src/sim5interpolation.h:#ifndef CUDA
src/sim5interpolation.h:#endif // CUDA
src/sim5disk-nt.c://! it is NOT thread-safe. For the same reasons, the routines declared here are not available to CUDA.
src/sim5disk-nt.c:#ifndef CUDA
src/sim5integration.c:    #ifndef CUDA
src/sim5integration.c:    #ifndef CUDA
src/sim5disk.c://! For the same reasons, the routines declared here are not available to CUDA.
src/sim5kerr-geod.c:#ifdef CUDA
src/sim5kerr-geod.c:    #ifdef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:        #ifndef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:            #ifndef CUDA
src/sim5kerr-geod.c:        #ifndef CUDA
src/sim5kerr-geod.c:    #ifndef CUDA
src/sim5elliptic.c:#ifndef CUDA
src/sim5elliptic.c:        #ifndef CUDA
src/sim5elliptic.c:        #ifndef CUDA
src/sim5elliptic.c:        #ifndef CUDA
src/sim5elliptic.c:        #ifndef CUDA
src/sim5elliptic.c:  #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:	#ifdef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5elliptic.c:    #ifndef CUDA
src/sim5distributions.c:#ifndef CUDA

```
