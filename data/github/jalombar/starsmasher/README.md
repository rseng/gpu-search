# https://github.com/jalombar/starsmasher

```console
documentation/installation.md:## [0.1] A Linux computer with an NVIDIA graphics card
documentation/installation.md:First, you should have a computer running Linux with at least one NVIDIA graphics card.  The instructions in this section [0] below assume that you'll be setting up your computer *without* using modules.  If you are working on a research cluster of a university, for example, then you may be able to set up the necessary software with ``module load`` commands.  For example ``module load cuda/cuda-10.1.2`` would load version 10.1.2 of cuda, while ``module load mpi/openmpi-1.10.5-gcc-6.4.0`` would load version 1.10.5 of openmpi (compiled with version 6.4.0 of gcc), if available.  To see what modules are available on your system, type ``module avail``.  **If you are using modules, then load up cuda and openmpi, and move on to section [1] of this installation guide!**
documentation/installation.md:## [0.2] nvcc (the NVIDIA cuda compiler)
documentation/installation.md:The gravity library in StarSmasher uses the NVIDIA graphics card and is compiled using the NVIDIA cuda compiler nvcc.  To check if nvcc is installed, type
documentation/installation.md:**If the version information is returned, then nvcc is properly installed and you can skip to section [0.3]!**  If instead you know that your system will need cuda installed, then follow NVIDIA's installation guide to do so:
documentation/installation.md:https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
documentation/installation.md:**Troubleshooting cuda compiler usage**
documentation/installation.md:If ``nvcc --version`` returns a command not found error message, it's possible that ``cuda`` is installed but your $PATH environment variable is improperly set.  If you think that may be the case, try to find where nvcc is located, for example by using
documentation/installation.md:Let's say for the sake of argument that you find nvcc exists in ``/usr/local/cuda-11.4/bin`` but that this directory is not in $PATH.  You should then update your $PATH and corresponding $LD_LIBRARY_PATH variables.  In bash, this can be done with
documentation/installation.md:export PATH=/usr/local/cuda-11.4/bin:$PATH
documentation/installation.md:export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
documentation/installation.md:## [0.3] NVIDIA driver communication
documentation/installation.md:The cuda installation should enable communications with the graphics card via the NVIDIA driver.  To test this, type
documentation/installation.md:nvidia-smi
documentation/installation.md:**Troubleshooting NVIDIA driver communications**
documentation/installation.md:If cuda has been installed but ``nvidia-smi`` doesn't work, then double check that you have completed the post installation steps from the NVIDIA installation guide.  Unfortunately, a reboot after a fresh cuda installation may be the easiest way to cure driver communication issues.  If the machine stalls on reboot, make sure that in the BIOS the EFI Secure Boot is disabled.
documentation/installation.md:* misc: miscellaneous files such as assorted makefiles, the StarCrash manual, and a tree gravity GPU library
documentation/installation.md:To compile the gravity library, cuda will need to be installed.  You can find the gravity library in both parallel_bleeding_edge/src/SPHgrav_lib2 and Blackollider/src/SPHgrav_lib2/.  The SPHgrav_lib2 subdirectory contains code written by Evghenii Gaburov (and somewhat modified by Jamie Lombardi and Sam Knarr) for calculating softened gravitational forces and potentials on NVIDIA GPUs. From within the main starsmasher folder,
documentation/installation.md:As written, this string is for an NVIDIA graphics card with compute capability (version) 6.1, such as an NVIDIA GTX 1070. Figure out what GPU you have (for example, by using ``nvidia-smi``) and then look up its compute capability here:
documentation/installation.md:https://en.wikipedia.org/wiki/CUDA
documentation/installation.md:If your graphics card has a different compute capability, then you would want to change the "61" portion of this line accordingly.  For example, if you have an NVIDIA TITAN RTX, which has a compute capability of 7.5, then write
documentation/installation.md:If compilation of the gravity library fails, the most likely explanation is that the Makefile is not identifying the correct location of the nvcc executable and/or cuda libraries.  Within the Makefile, look for the line
documentation/installation.md:CUDAPATH       := $(shell dirname $(shell dirname $(shell which nvcc)))
documentation/installation.md:and change it so that it identifies the main cuda directory.  For example, let's say that your system contains the files /usr/local/cuda-11.2/bin/nvcc and /usr/local/cuda-11.2/lib64/libcudart.so.  Then your main cuda directory is /usr/local/cuda-11.2, and you can change the above line in the makefile to
documentation/installation.md:CUDAPATH       := /usr/local/cuda-11.2
documentation/installation.md:***MADE VERSION THAT USES GPUS***
documentation/installation.md:When you complete the compilation, it will be create an executable file ending with "\_gpu\_sph." This executable is automatically moved to the directory *above* src. To run StarSmasher, you will have to move to that folder. Then just open your terminal where the .exe file is present or, after the compilation, type
documentation/installation.md:* If you had to hard-code in a CUDAPATH in SPHgrav_lib2/Makefile to get the gravity library to compile (see the end of section [2.1] above), then you may need to make the same change in the main makefile in the src directory.
documentation/walkthroughs/star_star_flyby.md:Note that ngravprocs=-6 and ppn=16... this means there are abs(ngravprocs)=6 GPU units per 16 CPU cores, as on one node of the supercomputer forge.
documentation/walkthroughs/star_star_flyby.md:For example, on Keeneland KFS, there are 3 gpus and 12 cores per node.
Blackollider/Relaxation_files/sph.input: qthreads=0 ! number of gpu threads per particle. typically set to 1, 2, 4, or 8.  set to a negative value to optimize the number of threads by timing.  set to 0 to guess the best number of threads without timing.
Blackollider/Relaxation_files/sph.input: computeexclusivemode=0, ! set this to 1 if on machine like grapefree with gpus in compute exclusive mode; set this to 0 on supercomputers like forge
Blackollider/Readme.md:mpirun -np N test_gpu_sph
Blackollider/Readme.md:When your star touches the black hole, the simulation will instantly become slower, as simulating the close interaction between a star and a pointmass black hole / neutron star is computationally expensive.  You'll also want to use the GPU version of the code to simulate this kind of collision.
Blackollider/collision_files/sph.input: computeexclusivemode=0, ! set this to 1 if on machine like grapefree with gpus in compute exclusive mode; set this to 0 on supercomputers like lincoln
Blackollider/src/makefile.grapefree:CUDAPATH = $(shell dirname $(shell dirname $(shell which nvcc))) 
Blackollider/src/makefile.grapefree:LIBS =  -lm #-L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
Blackollider/src/makefile.grapefree:GPUOBJS = $(FOBJS) gpu_grav.o
Blackollider/src/makefile.grapefree:GPUEXEC = $(shell basename $(shell dirname $(shell pwd)))_gpu_sph
Blackollider/src/makefile.grapefree:gpu: $(GPUOBJS)
Blackollider/src/makefile.grapefree:	$(LD) -o $(GPUEXEC) $(LDFLAGS) $(GPUOBJS) $(LIBS) -L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
Blackollider/src/makefile.grapefree:	mv $(GPUEXEC) ..
Blackollider/src/makefile.grapefree:	echo ***MADE VERSION THAT USES GPUS***
Blackollider/src/makefile.grapefree:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
Blackollider/src/makefile.gonzales_ifort:CUDAPATH = $(shell dirname $(shell dirname $(shell which nvcc)))
Blackollider/src/makefile.gonzales_ifort:LIBS =  -lm #-L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
Blackollider/src/makefile.gonzales_ifort:GPUOBJS = $(FOBJS) gpu_grav.o
Blackollider/src/makefile.gonzales_ifort:GPUEXEC = $(shell basename $(shell dirname $(shell pwd)))_gpu_sph
Blackollider/src/makefile.gonzales_ifort:gpu: $(GPUOBJS)
Blackollider/src/makefile.gonzales_ifort:	$(LD) -o $(GPUEXEC) $(GPUOBJS) -L$(GRAVLIB) $(LDFLAGS) -L$(CUDAPATH)/lib64 -lcudart  -lSPHgrav $(LIBS) -L/cm/shared/apps/openmpi/gcc/64/1.8.1/lib64 -lmpi_mpifh
Blackollider/src/makefile.gonzales_ifort:	mv $(GPUEXEC) ..
Blackollider/src/makefile.gonzales_ifort:	echo ***MADE VERSION THAT USES GPUS***
Blackollider/src/makefile.gonzales_ifort:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
Blackollider/src/gpu_grav.f:      subroutine set_nusegpus
Blackollider/src/gpu_grav.f:      integer nintvar,neos,nusegpus,nselfgravity
Blackollider/src/gpu_grav.f:      common/integration/nintvar,neos,nusegpus,nselfgravity
Blackollider/src/gpu_grav.f:      nusegpus=1
Blackollider/src/makefileGPUubuntu20.04:# makefile is set for a system76 laptop with Ubuntu 20.04 with Nvidia GTX 1070, cuda 11.1, grortran 9.3 and openMPI 4.03
Blackollider/src/makefileGPUubuntu20.04:CUDAPATH = $(shell dirname $(shell dirname $(shell which nvcc)))
Blackollider/src/makefileGPUubuntu20.04:#LDFLAGS = -lpthread -lifcore -lsvml -lifport -limf -lintlc -lrt -lstdc++ -lcudart
Blackollider/src/makefileGPUubuntu20.04:GPUOBJS = $(FOBJS) gpu_grav.o
Blackollider/src/makefileGPUubuntu20.04:GPUEXEC = $(shell basename $(shell dirname $(shell pwd)))_gpu_sph
Blackollider/src/makefileGPUubuntu20.04:gpu: $(GPUOBJS)
Blackollider/src/makefileGPUubuntu20.04:	$(LD) -o $(GPUEXEC) $(LDFLAGS) $(LIBS) $(GPUOBJS) -L $(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart  -lstdc++ 
Blackollider/src/makefileGPUubuntu20.04:	mv $(GPUEXEC) ..
Blackollider/src/makefileGPUubuntu20.04:	echo ***MADE VERSION THAT USES GPUS***
Blackollider/src/makefileGPUubuntu20.04:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
Blackollider/src/cpu_grav.f:!     Relate to the CUDA code grav_force_direct.cu...
Blackollider/src/cpu_grav.f:!     Relate to the CUDA code grav_force_direct.cu...
Blackollider/src/cpu_grav.f:      subroutine set_nusegpus
Blackollider/src/cpu_grav.f:      integer nintvar,neos,nusegpus,nselfgravity,ncooling
Blackollider/src/cpu_grav.f:      common/integration/nintvar,neos,nusegpus,nselfgravity,ncooling
Blackollider/src/cpu_grav.f:      nusegpus=0
Blackollider/src/cpu_grav.f:      subroutine gpu_init_dev(i,theta_angle)
Blackollider/src/SPHgrav_lib2/cutil.h:#include <cuda_runtime.h>
Blackollider/src/SPHgrav_lib2/cutil.h:#if CUDART_VERSION >= 4000
Blackollider/src/SPHgrav_lib2/cutil.h:#define CUT_DEVICE_SYNCHRONIZE( )   cudaDeviceSynchronize();
Blackollider/src/SPHgrav_lib2/cutil.h:#define CUT_DEVICE_SYNCHRONIZE( )   cudaThreadSynchronize();
Blackollider/src/SPHgrav_lib2/cutil.h:#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
Blackollider/src/SPHgrav_lib2/cutil.h:    cudaError err = call;                                                    \
Blackollider/src/SPHgrav_lib2/cutil.h:    if( cudaSuccess != err) {                                                \
Blackollider/src/SPHgrav_lib2/cutil.h:        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
Blackollider/src/SPHgrav_lib2/cutil.h:                __FILE__, __LINE__, cudaGetErrorString( err) );              \
Blackollider/src/SPHgrav_lib2/cutil.h:#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);
Blackollider/src/SPHgrav_lib2/cutil.h:    //! Check for CUDA error
Blackollider/src/SPHgrav_lib2/cutil.h:    cudaError_t err = cudaGetLastError();                                    \
Blackollider/src/SPHgrav_lib2/cutil.h:    if( cudaSuccess != err) {                                                \
Blackollider/src/SPHgrav_lib2/cutil.h:        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
Blackollider/src/SPHgrav_lib2/cutil.h:                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
Blackollider/src/SPHgrav_lib2/cutil.h:    if( cudaSuccess != err) {                                                \
Blackollider/src/SPHgrav_lib2/cutil.h:        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
Blackollider/src/SPHgrav_lib2/cutil.h:                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
Blackollider/src/SPHgrav_lib2/cutil.h:    cudaError_t err = cudaGetLastError();                                    \
Blackollider/src/SPHgrav_lib2/cutil.h:    if( cudaSuccess != err) {                                                \
Blackollider/src/SPHgrav_lib2/cutil.h:        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
Blackollider/src/SPHgrav_lib2/cutil.h:                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
Blackollider/src/SPHgrav_lib2/cutil.h:    cudaError_t err = cudaGetLastError();
Blackollider/src/SPHgrav_lib2/cutil.h:    if( cudaSuccess != err) {
Blackollider/src/SPHgrav_lib2/cutil.h:        FPRINTF((stderr, "%s(%i) : cutilCheckMsg() CUTIL CUDA error : %s : (%d) %s.\n",
Blackollider/src/SPHgrav_lib2/cutil.h:                file, line, errorMessage, (int)err, cudaGetErrorString( err ) ));
Blackollider/src/SPHgrav_lib2/Makefile:# path to your CUDA installation: CUDAPATH should contain the directory lib64 that contains libcudart.so as well as the bin directory that contains nvcc
Blackollider/src/SPHgrav_lib2/Makefile:CUDAPATH       := $(shell dirname $(shell dirname $(shell which nvcc)))
Blackollider/src/SPHgrav_lib2/Makefile:CUDAINCLUDE    := -I$(CUDAPATH)/include
Blackollider/src/SPHgrav_lib2/Makefile:NVCC           := $(CUDAPATH)/bin/nvcc
Blackollider/src/SPHgrav_lib2/Makefile:# type of GPU to be used and likely will need to be changed.  Look at https://en.wikipedia.org/wiki/CUDA to find the
Blackollider/src/SPHgrav_lib2/Makefile:# compute capability of your GPU.
Blackollider/src/SPHgrav_lib2/Makefile:NVCCFLAGS += -O4 -g  $(CUDAINCLUDE)  -I./ -Xptxas -v ##########,-abi=no 
Blackollider/src/SPHgrav_lib2/Makefile:CUDA_LIBS = -L$(CUDAPATH)/lib64 -lcudart
Blackollider/src/SPHgrav_lib2/Makefile:LDGPUGLAGS := $(LDFLAGS) $(CUDA_LIBS)
Blackollider/src/SPHgrav_lib2/cuVector.h:    CUDA_SAFE_CALL(cudaMemcpy(host_pointer, data, size*sizeof(T), cudaMemcpyHostToHost));
Blackollider/src/SPHgrav_lib2/cuVector.h:    CUDA_SAFE_CALL(cudaMemcpy(host_pointer, data, _size*sizeof(T), cudaMemcpyHostToHost));
Blackollider/src/SPHgrav_lib2/cuVector.h:    CUDA_SAFE_CALL(cudaMalloc(&p, size * sizeof(T)));
Blackollider/src/SPHgrav_lib2/cuVector.h:      CUDA_SAFE_CALL(cudaMallocHost(&p, size * sizeof(T)));
Blackollider/src/SPHgrav_lib2/cuVector.h:      CUDA_SAFE_CALL(cudaFree(dev_pointer));
Blackollider/src/SPHgrav_lib2/cuVector.h:        CUDA_SAFE_CALL(cudaFreeHost(host_pointer));
Blackollider/src/SPHgrav_lib2/cuVector.h:      CUDA_SAFE_CALL(cudaMemcpy(dev_pointer, host_pointer, count * sizeof(T), cudaMemcpyHostToDevice));
Blackollider/src/SPHgrav_lib2/cuVector.h:      CUDA_SAFE_CALL(cudaMemcpy(host_pointer, dev_pointer, count * sizeof(T), cudaMemcpyDeviceToHost));
Blackollider/src/SPHgrav_lib2/grav_force_direct.cu:  cudaEvent_t start, stop;
Blackollider/src/SPHgrav_lib2/grav_force_direct.cu:    assert(cudaGetDeviceCount(&ndevice) == 0);
Blackollider/src/SPHgrav_lib2/grav_force_direct.cu:      fprintf(stderr, " SPHgrav found %d CUDA devices \n", ndevice);
Blackollider/src/SPHgrav_lib2/grav_force_direct.cu:      cudaDeviceProp p;
Blackollider/src/SPHgrav_lib2/grav_force_direct.cu:      assert(cudaGetDeviceProperties(&p, dev) == cudaSuccess);
Blackollider/src/SPHgrav_lib2/grav_force_direct.cu:    assert(cudaSetDevice(device) == cudaSuccess);
Blackollider/src/SPHgrav_lib2/grav_force_direct.cu:    cudaEventCreate( &start );
Blackollider/src/SPHgrav_lib2/grav_force_direct.cu:    cudaEventCreate( &stop  );
Blackollider/src/SPHgrav_lib2/grav_force_direct.cu:    cudaEventRecord( start, 0 );
Blackollider/src/SPHgrav_lib2/grav_force_direct.cu:    cudaEventRecord( stop, 0 );
Blackollider/src/SPHgrav_lib2/grav_force_direct.cu:    cudaDeviceSynchronize();
Blackollider/src/SPHgrav_lib2/grav_force_direct.cu:    cudaEventElapsedTime( &elapsed_time_ms, start, stop );
Blackollider/src/SPHgrav_lib2/grav_force_direct.cu:  void gpu_init_dev_(int *myrank)
Blackollider/src/SPHgrav_lib2/grav_force_direct.cu:  void gpu_init_dev_(int *myrank, double *theta)
Blackollider/src/grav.f:      integer maxGPUtemperature
Blackollider/src/grav.f:      if(nusegpus.eq.0)return
Blackollider/src/grav.f:            inquire(exist=ex, file='maxGPUtemperature')
Blackollider/src/grav.f:               call system ("mkfifo maxGPUtemperature")
Blackollider/src/grav.f:     $           "nvidia-smi -q -d TEMPERATURE|grep -A2 'GPU 0'
Blackollider/src/grav.f:     $|tail -n1>maxGPUtemperature&") ! Find temperatures in second to last column, sort them, and take the largest
Blackollider/src/grav.f:            open(68, file='maxGPUtemperature', action='read')
Blackollider/src/grav.f:            read(68, *) maxGPUtemperature
Blackollider/src/grav.f:            write(6, *) 'max GPU temp=', maxGPUtemperature
Blackollider/src/grav.f:            if(maxGPUtemperature.gt.91) then
Blackollider/src/grav.f:               call system ("rm maxGPUtemperature")
Blackollider/src/init.f:         if(nusegpus.eq.0)then
Blackollider/src/init.f:            write(69,*)'gpus will be used for gravity'
Blackollider/src/init.f:      qthreads=0               ! number of gpu threads per particle. typically set to 1, 2, 4, or 8.  set to a negative value to optimize the number of threads by timing.  set to 0 to guess the best number of threads without timing.
Blackollider/src/init.f:      computeexclusivemode=0   ! set this to 1 if on machine like grapefree with gpus in compute exclusive mode; set this to 0 on supercomputers like lincoln
Blackollider/src/init.f:      call set_nusegpus         ! if using gpus, this sets nusegpus=1 *and* nselfgravity=1
Blackollider/src/init.f:      if(ngr.ne.0 .and. nusegpus.eq.0) then
Blackollider/src/prereqs.sh:module load cuda/cuda_5.0.35
Blackollider/src/makefile.quest3:CUDAPATH = $(shell dirname $(shell dirname $(shell which nvcc)))
Blackollider/src/makefile.quest3:LIBS =  -lm #-L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
Blackollider/src/makefile.quest3:GPUOBJS = $(FOBJS) gpu_grav.o
Blackollider/src/makefile.quest3:GPUEXEC = $(shell basename $(shell dirname $(shell pwd)))_gpu_sph
Blackollider/src/makefile.quest3:gpu: $(GPUOBJS)
Blackollider/src/makefile.quest3:	$(LD) -o $(GPUEXEC) $(LDFLAGS) $(GPUOBJS) $(LIBS) -L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart -lmpi_mpifh #-L/cm/shared/apps/openmpi/gcc/64/1.8.1/lib64
Blackollider/src/makefile.quest3:	mv $(GPUEXEC) ..
Blackollider/src/makefile.quest3:	echo ***MADE VERSION THAT USES GPUS***
Blackollider/src/makefile.quest3:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
Blackollider/src/makefile:CUDAPATH = $(shell dirname $(shell dirname $(shell which nvcc)))
Blackollider/src/makefile:LIBS =  -lm -lstdc++ #-L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
Blackollider/src/makefile:GPUOBJS = $(FOBJS) gpu_grav.o
Blackollider/src/makefile:GPUEXEC = $(shell basename $(shell dirname $(shell pwd)))_gpu_sph
Blackollider/src/makefile:gpu: $(GPUOBJS)
Blackollider/src/makefile:	$(LD) -o $(GPUEXEC) $(LDFLAGS) $(GPUOBJS) $(LIBS) -L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart -L/cm/shared/apps/openmpi/open64/64/1.10.1/lib64/ -lmpi_mpifh
Blackollider/src/makefile:	mv $(GPUEXEC) ..
Blackollider/src/makefile:	echo ***MADE VERSION THAT USES GPUS***
Blackollider/src/makefile:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
Blackollider/src/changetf.f:c     one outfile to the next.... (this happens in some of evghenii's gpu runs)
Blackollider/src/initialize_polyes.f:               if(nusegpus.eq.1)then
Blackollider/src/skipahead.f:         if(nusegpus.eq.1)then
Blackollider/src/skipahead.f:         if(nusegpus.eq.1)then
Blackollider/src/skipahead.f:            if(nusegpus.eq.1)then
Blackollider/src/starsmasher.h:      integer neos,nusegpus,nselfgravity,ncooling,nkernel
Blackollider/src/starsmasher.h:      common/integration/nintvar,neos,nusegpus,nselfgravity,ncooling,nkernel,usegravitycorrections
Blackollider/src/main.f:!     the following line assumes ppn cpu threads and abs(ngravprocs) gpu threads per node
Blackollider/src/main.f:!         call gpu_init_dev(myrank/((nprocs+ppn-1)/ppn)) ! if the gpus are set up in device exclusive mode,
Blackollider/src/main.f:      if(myrank.lt.ngravprocs .and. .not. alreadyinitialized .and. computeexclusivemode.ne.1 .and. nusegpus.eq.1) then
Blackollider/src/main.f:!          gpus must always be initialized, even if we use just 1 mpi process
Blackollider/src/main.f:         call gpu_init_dev(myrank/((nprocs+ppn-1)/ppn), theta_angle) ! if the gpus are set up in device exclusive mode,
Blackollider/src/main.f:         write(6,"('myrank=',I3,' is running on ',A,' with gpu',i3)")
Blackollider/src/main.f:         if(nusegpus.eq.1) then
Blackollider/src/main.f:     $              'gpurank,gravdispl,gravrecvcount=',
Blackollider/src/main.f:         if(myrank.lt.ngravprocs .and. nusegpus.eq.1) then
Blackollider/src/main.f:c     the first call to a gpu is usually slow, so let's not time this one:
Blackollider/src/initialize_parent.f:               if(nusegpus.eq.1)then
Blackollider/src/balAV3.f:         if(nusegpus.eq.1)then
Blackollider/src/balAV3.f:            if(nusegpus.eq.1)then
Blackollider/src/balAV3.f:         if(nusegpus.eq.1)then
example_input/corotating_binary/sph.input: computeexclusivemode=0, ! set this to 1 if on machine like grapefree with gpus in compute exclusive mode; set this to 0 on supercomputers like lincoln
example_input/Creating_a_MESA_star/Readme.md:To make the star, after that in you have compiled the code, after that in your folder there are all the files needed (like in this example folder) and everything is been setted in sph.input (like number of cores and GPUs of your machine [per node]), type in your terminal:
example_input/Creating_a_MESA_star/Readme.md:mpirun -np N test_gpu_sph
example_input/Creating_a_MESA_star/Readme.md:mpirun -np N test_gpu_sph
example_input/Creating_a_MESA_star/sph.input: qthreads=0 ! number of gpu threads per particle. typically set to 1, 2, 4, or 8.  set to a negative value to optimize the number of threads by timing.  set to 0 to guess the best number of threads without timing.
example_input/Creating_a_MESA_star/sph.input: computeexclusivemode=0, ! set this to 1 if on machine like grapefree with gpus in compute exclusive mode; set this to 0 on supercomputers like forge
example_input/Tips_and_tricks.md:mpirun -np N test_gpu_sph
example_input/Tips_and_tricks.md:SPHgrav_lib is the library used to calculate gravity with CUDA, and then is used to have a faster simulation. This is the standard gravity library. However there is a second that is been optimized for StarSmasher, and that one is SPHgrav_lib2. This is more accurate and equally faster, so it's reccomendend to use this instead of the first. To do so, just delete SPHgravlib from your folder and rename SPHgrav_lib2 in SPHgravlib. The code will recompile equally, and don't forget to use the correct Makefile for you!
example_input/Tips_and_tricks.md:(b) the GPUs have been configured by the system administrator to be in exclusive mode.  
example_input/Tips_and_tricks.md:For example, if a compute node has 4 GPUs and the user wants one job to use two of the GPUs and another job to use the other two GPUs, then, depending on how 
example_input/Tips_and_tricks.md:is telling StarSmasher that the GPUs have been configure such that only one COMPUTE thread is allowed to run on each GPU.  
example_input/Tips_and_tricks.md:Setting the value to 1 does *not* actually change the configuration of the GPUs (that would be done by the system administrator with commands 
example_input/Tips_and_tricks.md:such as "nvidia-smi --id=0 --compute-mode=EXCLUSIVE_PROCESS".  If the user sets computeexclusivemode equal to 1 when the GPUs are in their default configuration, then 
example_input/Tips_and_tricks.md:the code will run slower because a single GPU will do all of the calculations which could have been spread out over multiple GPUs
example_input/3_stars/sph.input: computeexclusivemode=0, ! set this to 1 if on machine like grapefree with gpus in compute exclusive mode; set this to 0 on supercomputers like lincoln
example_input/relaxation_rotating_giant/sph.input: qthreads=0 ! number of gpu threads per particle. typically set to 1, 2, 4, or 8.  set to a negative value to optimize the number of threads by timing.  set to 0 to guess the best number of threads without timing.
example_input/relaxation_rotating_giant/sph.input: computeexclusivemode=0, ! set this to 1 if on machine like grapefree with gpus in compute exclusive mode; set this to 0 on supercomputers like lincoln
example_input/collision_elliptical/sph.input: qthreads=0 ! number of gpu threads per particle. typically set to 1, 2, 4, or 8.  set to a negative value to optimize the number of threads by timing.  set to 0 to guess the best number of threads without timing.
example_input/collision_elliptical/sph.input: computeexclusivemode=0, ! set this to 1 if on machine like grapefree with gpus in compute exclusive mode; set this to 0 on supercomputers like lincoln
example_input/relaxation_preMS/sph.input: computeexclusivemode=0, ! set this to 1 if on machine like grapefree with gpus in compute exclusive mode; set this to 0 on supercomputers like forge
example_input/oscillating_preMS/sph.input: qthreads=0 ! number of gpu threads per particle. typically set to 1, 2, 4, or 8.  set to a negative value to optimize the number of threads by timing.  set to 0 to guess the best number of threads without timing.
example_input/oscillating_preMS/sph.input: computeexclusivemode=0, ! set this to 1 if on machine like grapefree with gpus in compute exclusive mode; set this to 0 on supercomputers like lincoln
example_input/collision/sph.input: computeexclusivemode=0, ! set this to 1 if on machine like grapefree with gpus in compute exclusive mode; set this to 0 on supercomputers like lincoln
README.md:The code now implements variational equations of motion and libraries to calculate the gravitational forces between particles using direct summation on NVIDIA graphics cards as described in Gaburov et al. (2010b). 
misc/SPHgravtree_lib/main.cpp:#include "../lib/include/my_cuda.h"
misc/SPHgravtree_lib/main.cpp:  //Copy data in the GPU Host and devices buffers
misc/SPHgravtree_lib/main.cpp:        /*    fprintf(stderr, "TESTSTAT %d \t acc: %f %f %f %f \t\tds2: %f \tgpu-nngb: %d cpu-nngb: %d  \n", i*jump,
misc/SPHgravtree_lib/grav_force_tree.cu:  cudaEvent_t start, stop;
misc/SPHgravtree_lib/grav_force_tree.cu:    assert(cudaGetDeviceCount(&ndevice) == 0);
misc/SPHgravtree_lib/grav_force_tree.cu:      fprintf(stderr, " SPHgrav found %d CUDA devices \n", ndevice);
misc/SPHgravtree_lib/grav_force_tree.cu:      cudaDeviceProp p;
misc/SPHgravtree_lib/grav_force_tree.cu:      assert(cudaGetDeviceProperties(&p, dev) == cudaSuccess);
misc/SPHgravtree_lib/grav_force_tree.cu:    cudaEventCreate( &start );
misc/SPHgravtree_lib/grav_force_tree.cu:    cudaEventCreate( &stop  );
misc/SPHgravtree_lib/grav_force_tree.cu:    cudaEventRecord( start, 0 );
misc/SPHgravtree_lib/grav_force_tree.cu:    cudaEventRecord( stop, 0 );
misc/SPHgravtree_lib/grav_force_tree.cu:    cudaThreadSynchronize();
misc/SPHgravtree_lib/grav_force_tree.cu:    cudaEventElapsedTime( &elapsed_time_ms, start, stop );
misc/SPHgravtree_lib/grav_force_tree.cu:  void gpu_init_dev_(int *myrank, double *theta)
misc/SPHgravtree_lib/Makefile:CUDAPATH       := /usr/local/cuda
misc/SPHgravtree_lib/Makefile:CUDAINCLUDE    := -I$(CUDAPATH)/include
misc/SPHgravtree_lib/Makefile:NVCC           := $(CUDAPATH)/bin/nvcc
misc/SPHgravtree_lib/Makefile:NVCCFLAGS += -O4 -g  $(CUDAINCLUDE)  -I./ -Xptxas -v,-abi=no 
misc/SPHgravtree_lib/Makefile:CUDA_LIBS = -L$(CUDAPATH)/lib64 -lcudart
misc/SPHgravtree_lib/Makefile:LDGPUGLAGS := $(LDFLAGS) $(CUDA_LIBS)
misc/SPHgravtree_lib/libSequoia/Makefile:# path to your CUDA installation
misc/SPHgravtree_lib/libSequoia/Makefile:CUDA_TK  = /usr/local/cuda
misc/SPHgravtree_lib/libSequoia/Makefile:CXXFLAGS =  -fPIC $(OFLAGS) -I$(CUDA_TK)/include 
misc/SPHgravtree_lib/libSequoia/Makefile:# NVCC      = $(CUDA_TK)/bin/nvcc  --device-emulation
misc/SPHgravtree_lib/libSequoia/Makefile:# NVCCFLAGS = -D_DEBUG -O0 -g -I$(CUDA_SDK)/common/inc -arch=sm_12 --maxrregcount=64  --opencc-options -OPT:Olimit=0 -I$(CUDPP)/cudpp/include
misc/SPHgravtree_lib/libSequoia/Makefile:NVCC      = $(CUDA_TK)/bin/nvcc  
misc/SPHgravtree_lib/libSequoia/Makefile:LDFLAGS = -lcuda -lOpenCL  
misc/SPHgravtree_lib/libSequoia/Makefile:CUDAKERNELSPATH = CUDAkernels
misc/SPHgravtree_lib/libSequoia/Makefile:CUDAKERNELS = build_tree.cu \
misc/SPHgravtree_lib/libSequoia/Makefile:CUDAPTX = $(CUDAKERNELS:%.cu=%.ptx)
misc/SPHgravtree_lib/libSequoia/Makefile:#SRC = main.cpp octree.cpp load_kernels.cpp scanFunctions.cpp build.cpp compute_properties.cpp sort_bodies_gpu.cpp gpu_iterate.cpp parallel.cpp libraryInterface.cpp
misc/SPHgravtree_lib/libSequoia/Makefile:SRC = octree.cpp load_kernels.cpp scanFunctions.cpp sort_bodies_gpu.cpp sequoiaInterface.cpp build.cpp compute_properties.cpp useTreeFunctions.cpp
misc/SPHgravtree_lib/libSequoia/Makefile:#all:	  $(OBJ) $(CUDAPTX) $(PROG) $(CODELIB)
misc/SPHgravtree_lib/libSequoia/Makefile:all:	  $(OBJ) $(CUDAPTX) $(CODELIB)
misc/SPHgravtree_lib/libSequoia/Makefile:kernels:  $(CUDAPTX)
misc/SPHgravtree_lib/libSequoia/Makefile:$(CODELIB): $(OBJ) $(CUDAPTX)
misc/SPHgravtree_lib/libSequoia/Makefile:%.ptx: $(CUDAKERNELSPATH)/%.cu
misc/SPHgravtree_lib/libSequoia/Makefile:build_tree.ptx: $(CUDAKERNELSPATH)/support_kernels.cu $(INCLUDEPATH)/node_specs.h
misc/SPHgravtree_lib/libSequoia/Makefile:compute_properties.ptx: $(CUDAKERNELSPATH)/support_kernels.cu $(INCLUDEPATH)/node_specs.h
misc/SPHgravtree_lib/libSequoia/Makefile:compute_propertiesD.ptx: $(CUDAKERNELSPATH)/support_kernels.cu $(INCLUDEPATH)/node_specs.h
misc/SPHgravtree_lib/libSequoia/Makefile:dev_approximate_gravity.ptx: $(CUDAKERNELSPATH)/support_kernels.cu $(CUDAKERNELSPATH)/dev_shared_traverse_functions.cu $(INCLUDEPATH)/node_specs.h
misc/SPHgravtree_lib/libSequoia/Makefile:dev_approximate_gravity_let.ptx: $(CUDAKERNELSPATH)/support_kernels.cu $(INCLUDEPATH)/node_specs.h
misc/SPHgravtree_lib/libSequoia/Makefile:timestep.ptx: $(CUDAKERNELSPATH)/support_kernels.cu $(INCLUDEPATH)/node_specs.h
misc/SPHgravtree_lib/libSequoia/Makefile:sortKernels.ptx: $(CUDAKERNELSPATH)/scanKernels.cu  $(CUDAKERNELSPATH)/support_kernels.cu $(INCLUDEPATH)/node_specs.h
misc/SPHgravtree_lib/libSequoia/Makefile:parallel.ptx: $(CUDAKERNELSPATH)/support_kernels.cu $(INCLUDEPATH)/node_specs.h
misc/SPHgravtree_lib/libSequoia/include/octree.h:#define USE_CUDA
misc/SPHgravtree_lib/libSequoia/include/octree.h:#ifdef USE_CUDA
misc/SPHgravtree_lib/libSequoia/include/octree.h:  #include "my_cuda.h"
misc/SPHgravtree_lib/libSequoia/include/octree.h:  //GPU kernels and functions
misc/SPHgravtree_lib/libSequoia/include/octree.h:  void gpuCompact(my_dev::context&, my_dev::dev_mem<uint> &srcValues,
misc/SPHgravtree_lib/libSequoia/include/octree.h:  void gpuSplit(my_dev::context&, my_dev::dev_mem<uint> &srcValues,
misc/SPHgravtree_lib/libSequoia/include/octree.h:  void gpuSort(my_dev::context &devContext,
misc/SPHgravtree_lib/libSequoia/include/octree.h:  void gpuSort_32b(my_dev::context &devContext,
misc/SPHgravtree_lib/libSequoia/include/octree.h://    devID = procId % getNumberOfCUDADevices();
misc/SPHgravtree_lib/libSequoia/include/octree.h:    getNumberOfCUDADevices();  //Stop compiler warnings
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:#ifndef _MY_CUDA_H_
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:#define _MY_CUDA_H_
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h://   __cuda_assign_operators(float4)
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:#include <cuda.h>
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h://Function made by NVIDIA
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:  // open the OpenCL source code file
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_SUCCESS : return "CUDA_SUCCESS";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_INVALID_VALUE : return "CUDA_ERROR_INVALID_VALUE";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_OUT_OF_MEMORY : return "CUDA_ERROR_OUT_OF_MEMORY";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_NOT_INITIALIZED : return "CUDA_ERROR_NOT_INITIALIZED";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_DEINITIALIZED : return "CUDA_ERROR_DEINITIALIZED";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_NO_DEVICE : return "CUDA_ERROR_NO_DEVICE";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_INVALID_DEVICE : return "CUDA_ERROR_INVALID_DEVICE";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_INVALID_IMAGE : return "CUDA_ERROR_INVALID_IMAGE";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_INVALID_CONTEXT : return "CUDA_ERROR_INVALID_CONTEXT";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT : return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_MAP_FAILED : return "CUDA_ERROR_MAP_FAILED";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_UNMAP_FAILED : return "CUDA_ERROR_UNMAP_FAILED";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_ARRAY_IS_MAPPED : return "CUDA_ERROR_ARRAY_IS_MAPPED";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_ALREADY_MAPPED : return "CUDA_ERROR_ALREADY_MAPPED";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_NO_BINARY_FOR_GPU : return "CUDA_ERROR_NO_BINARY_FOR_GPU";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_ALREADY_ACQUIRED : return "CUDA_ERROR_ALREADY_ACQUIRED";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_NOT_MAPPED : return "CUDA_ERROR_NOT_MAPPED";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_INVALID_SOURCE : return "CUDA_ERROR_INVALID SOURCE";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_FILE_NOT_FOUND : return "CUDA_ERROR_FILE_NOT_FOUND";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_INVALID_HANDLE : return "CASE_ERROR_INVALID_HANDLE";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_NOT_FOUND : return "CUDA_ERROR_NOT_FOUND";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_NOT_READY : return "CUDA_ERROR_NOT_READY";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_LAUNCH_FAILED : return "CUDA_ERROR_LAUNCH_FAILED";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES : return "CUDA_ERROR_LAUNCH_OUT_OF_RESOUCES";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_LAUNCH_TIMEOUT : return "CUDA_ERROR_LAUNCH_TIMEOUT";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING : return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    case CUDA_ERROR_UNKNOWN : return "CUDA_ERROR_UNKNOWN";
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:  if( CUDA_SUCCESS != err) {                                               \
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    fprintf(stderr, "Cuda driver error <%s> in file '%s' in line %i.\n",   \
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:  if( CUDA_SUCCESS != err) {                                               \
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:    fprintf(stderr, "Cuda driver error <%s> in file '%s' in line %i. (Kernel: %s)\n",   \
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h://OpenCL to CUDA macro / functions
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:static int getNumberOfCUDADevices()
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:  // Get number of devices supporting CUDA
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:        printf("Creating CUDA context \n");
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:        // Get number of devices supporting CUDA
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:        if(res != CUDA_SUCCESS)
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:            if(cuCtxCreate(&hContext, ctxCreateFlags, hDevice) != CUDA_SUCCESS)
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:#ifdef _DEBUG_PRINT_CUDA_
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:        if(res == CUDA_SUCCESS) return true;
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:        if(res == CUDA_ERROR_NOT_READY) return false;
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:#ifdef _DEBUG_PRINT_CUDA_
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:        void cuda_free() 
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:        } //cuda_free
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:        //CUDA has no memory flags like opencl
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:          cuda_free();
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:          cuda_free();
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:        void cmalloc_copy(bool pinned, bool flags, CUdeviceptr cudaMem, 
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:          hDeviceMem   = cudaMem + offset*sizeof(uint) + allignOffset*sizeof(uint);
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:          if (size > 0) cuda_free();
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:          if (size > 0) cuda_free();
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:        //Not implemented in CUDA
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:        if (size > 0) cuda_free();
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:        printf("TODO: This function does NOT copy memory in CUDA!! \n");      
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:            cuda_free();
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:        //In cuda version we assume that the code is already compiled into ptx
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:      //NVIDIA macro
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:      //Cuda does not have a function like clSetKernelArg
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:        //Since the values between CUDA and OpenCL differ:
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:        //Cuda is specific size of each block, while OpenCL
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:#ifdef _DEBUG_PRINT_CUDA_
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:  }     // end of namespace my_cuda
misc/SPHgravtree_lib/libSequoia/include/my_cuda.h:#endif // _MY_CUDA_H_
misc/SPHgravtree_lib/libSequoia/include/node_specs.h:#define TREE_WALK_BLOCKS_PER_SM    8           //Number of GPU thread-blocks used for tree-walk
misc/SPHgravtree_lib/libSequoia/include/sequoiaInterface.h:#define USE_CUDA
misc/SPHgravtree_lib/libSequoia/include/sequoiaInterface.h:#ifdef USE_CUDA
misc/SPHgravtree_lib/libSequoia/include/sequoiaInterface.h:  #include "my_cuda.h"
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_get_grp_ngb.cu:texture<float4, 1, cudaReadModeElementType> texNodeSize;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_get_grp_ngb.cu:texture<float4, 1, cudaReadModeElementType> texNodeCenter;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_get_grp_ngb.cu:texture<float4, 1, cudaReadModeElementType> texMultipole;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_get_grp_ngb.cu:texture<float4, 1, cudaReadModeElementType> texBody;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity_let.cu:texture<float4, 1, cudaReadModeElementType> texNodeSize;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity_let.cu:texture<float4, 1, cudaReadModeElementType> texNodeCenter;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity_let.cu:texture<float4, 1, cudaReadModeElementType> texMultipole;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity_let.cu:texture<float4, 1, cudaReadModeElementType> texBody;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity_let.cu:        if(n_nodes0 > 0){//Work around pre CUDA 4.1 compiler bug
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity_let.cu:// 	n_total = calc_prefix<DIM2>(prefix, tid,  !split && use_node);         // for some unkown reason this does not work right on the GPU
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_get_ngb.cu:texture<float4, 1, cudaReadModeElementType> texNodeSize;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_get_ngb.cu:texture<float4, 1, cudaReadModeElementType> texNodeCenter;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_get_ngb.cu:texture<float4, 1, cudaReadModeElementType> texMultipole;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_get_ngb.cu:texture<float4, 1, cudaReadModeElementType> texBody;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/build_tree.cu:// //#include "/home/jbedorf/papers/GBPZ2010/codes/jb/build_tree/CUDA/support_kernels.cu"
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity.cu_ids:texture<float4, 1, cudaReadModeElementType> texNodeSize;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity.cu_ids:texture<float4, 1, cudaReadModeElementType> texNodeCenter;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity.cu_ids:texture<float4, 1, cudaReadModeElementType> texMultipole;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity.cu_ids:texture<float4, 1, cudaReadModeElementType> texBody;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity.cu_ids:  //float ids  = (1.0f / sqrtf(ds2)) * selfGrav; Slower in Pre CUDA4.1
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity.cu_ids:// 	n_total = calc_prefix<DIM2>(prefix, tid,  !split && use_node);         // for some unkown reason this does not work right on the GPU
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity.cu_ori:texture<float4, 1, cudaReadModeElementType> texNodeSize;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity.cu_ori:texture<float4, 1, cudaReadModeElementType> texNodeCenter;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity.cu_ori:texture<float4, 1, cudaReadModeElementType> texMultipole;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity.cu_ori:texture<float4, 1, cudaReadModeElementType> texBody;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity.cu_ori:  //float ids  = (1.0f / sqrtf(ds2)) * selfGrav; Slower in Pre CUDA4.1
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity.cu_ori:// 	n_total = calc_prefix<DIM2>(prefix, tid,  !split && use_node);         // for some unkown reason this does not work right on the GPU
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity.cu:texture<float4, 1, cudaReadModeElementType> texNodeSize;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity.cu:texture<float4, 1, cudaReadModeElementType> texNodeCenter;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity.cu:texture<float4, 1, cudaReadModeElementType> texMultipole;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity.cu:texture<float4, 1, cudaReadModeElementType> texBody;
misc/SPHgravtree_lib/libSequoia/CUDAkernels/dev_approximate_gravity.cu:        // 	n_total = calc_prefix<DIM2>(prefix, tid,  !split && use_node);         // for some unkown reason this does not work right on the GPU
misc/SPHgravtree_lib/libSequoia/src/sequoiaInterface.cpp:  char *gpu_prof_log;
misc/SPHgravtree_lib/libSequoia/src/sequoiaInterface.cpp:  gpu_prof_log=getenv("CUDA_PROFILE_LOG");
misc/SPHgravtree_lib/libSequoia/src/sequoiaInterface.cpp:  if(gpu_prof_log){
misc/SPHgravtree_lib/libSequoia/src/sequoiaInterface.cpp:    sprintf(tmp,"process%d_%s",0,gpu_prof_log);
misc/SPHgravtree_lib/libSequoia/src/sequoiaInterface.cpp:    setenv("CUDA_PROFILE_LOG",tmp,1);
misc/SPHgravtree_lib/libSequoia/src/sequoiaInterface.cpp:  string logFileName    = "gpuLog.log";
misc/SPHgravtree_lib/libSequoia/src/load_kernels.cpp:#ifdef USE_CUDA
misc/SPHgravtree_lib/libSequoia/src/load_kernels.cpp:  compactCount.load_source("scanKernels.cl", "OpenCLKernels");
misc/SPHgravtree_lib/libSequoia/src/load_kernels.cpp:  exScanBlock.load_source("scanKernels.cl", "OpenCLKernels");
misc/SPHgravtree_lib/libSequoia/src/load_kernels.cpp:  compactMove.load_source("scanKernels.cl", "OpenCLKernels");
misc/SPHgravtree_lib/libSequoia/src/load_kernels.cpp:  splitMove.load_source("scanKernels.cl", "OpenCLKernels");
misc/SPHgravtree_lib/libSequoia/src/load_kernels.cpp:#ifdef USE_CUDA
misc/SPHgravtree_lib/libSequoia/src/load_kernels.cpp:#ifdef USE_CUDA
misc/SPHgravtree_lib/libSequoia/src/load_kernels.cpp:#ifdef USE_CUDA
misc/SPHgravtree_lib/libSequoia/src/load_kernels.cpp:#ifdef USE_CUDA
misc/SPHgravtree_lib/libSequoia/src/scanFunctions.cpp:void octree::gpuCompact(my_dev::context &devContext, 
misc/SPHgravtree_lib/libSequoia/src/scanFunctions.cpp:  // In the next step we associate the GPU memory with the Kernel arguments
misc/SPHgravtree_lib/libSequoia/src/scanFunctions.cpp:  #ifdef USE_CUDA
misc/SPHgravtree_lib/libSequoia/src/scanFunctions.cpp:void octree::gpuSplit(my_dev::context &devContext, 
misc/SPHgravtree_lib/libSequoia/src/scanFunctions.cpp:  // In the next step we associate the GPU memory with the Kernel arguments
misc/SPHgravtree_lib/libSequoia/src/scanFunctions.cpp:  #ifdef USE_CUDA
misc/SPHgravtree_lib/libSequoia/src/scanFunctions.cpp:void  octree::gpuSort(my_dev::context &devContext,
misc/SPHgravtree_lib/libSequoia/src/scanFunctions.cpp:  gpuSort_32b(devContext, 
misc/SPHgravtree_lib/libSequoia/src/scanFunctions.cpp:  gpuSort_32b(devContext, 
misc/SPHgravtree_lib/libSequoia/src/scanFunctions.cpp:  gpuSort_32b(devContext, 
misc/SPHgravtree_lib/libSequoia/src/scanFunctions.cpp:void octree::gpuSort_32b(my_dev::context &devContext, 
misc/SPHgravtree_lib/libSequoia/src/scanFunctions.cpp:  #ifdef USE_CUDA
misc/SPHgravtree_lib/libSequoia/src/sort_bodies_gpu.cpp:  //We assume the bodies are already onthe GPU
misc/SPHgravtree_lib/libSequoia/src/sort_bodies_gpu.cpp:  //Call the GPUSort function, since we made it general 
misc/SPHgravtree_lib/libSequoia/src/sort_bodies_gpu.cpp:  gpuSort(devContext, srcValues, output, srcValues, n_bodies, 32, 3, tree);
misc/SPHgravtree_lib/libSequoia/src/build.cpp:    //gpuCompact to get number of created nodes    
misc/SPHgravtree_lib/libSequoia/src/build.cpp:    gpuCompact(devContext, validList, compactList, n_bodies*2, &validCount);
misc/SPHgravtree_lib/libSequoia/src/build.cpp:  gpuSplit(devContext, validList, tree.leafNodeIdx, tree.n_nodes, &tree.n_leafs);     
misc/SPHgravtree_lib/libSequoia/src/build.cpp:  gpuCompact(devContext, validList, tree.node_level_list, 
misc/SPHgravtree_lib/libSequoia/src/build.cpp:    //gpuCompact to get number of created nodes    
misc/SPHgravtree_lib/libSequoia/src/build.cpp:    gpuCompact(devContext, validList, compactList, n_bodies*2, &validCount);
misc/SPHgravtree_lib/libSequoia/src/build.cpp:  gpuCompact(devContext, validList, compactList, n_bodies*2, &validCount);
misc/SPHgravtree_lib/libSequoia/src/build.cpp:  //gpuCompact    
misc/SPHgravtree_lib/libSequoia/src/build.cpp:  gpuCompact(devContext, validList, compactList, tree.n*2, &validCount);
misc/sph.pbs.kfs:module unload cuda
misc/sph.pbs.kfs:module load cuda/5.0
misc/makefiles/makefile.grapefree:CUDAPATH = /usr/local/cuda-5.0
misc/makefiles/makefile.grapefree:LIBS = -L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart -limf -lm #-lcuda
misc/makefiles/makefile.checkers_ifort:CUDAPATH = /global/scratch/software/cuda/cuda-3.2
misc/makefiles/makefile.checkers_ifort:LIBS = $(GRAVLIB) -L. -lGPUsph_gatherscatter -L$(CUDAPATH)/lib -L/usr/mpi/gcc/openmpi-1.4/lib64 -L /home/jalombar/lib64 -lmpi -lmpi_f77 -lcudart
misc/makefiles/makefile.gonzales_ifort:CUDAPATH = /cm/shared/apps/cuda60/toolkit/6.0.37/
misc/makefiles/makefile.gonzales_ifort:LIBS =  -lm #-L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
misc/makefiles/makefile.gonzales_ifort:GPUOBJS = $(FOBJS) gpu_grav.o
misc/makefiles/makefile.gonzales_ifort:GPUEXEC = $(shell basename $(shell dirname $(shell pwd)))_gpu_sph
misc/makefiles/makefile.gonzales_ifort:gpu: $(GPUOBJS)
misc/makefiles/makefile.gonzales_ifort:	$(LD) -o $(GPUEXEC) $(GPUOBJS) -L$(GRAVLIB) $(LDFLAGS) -L$(CUDAPATH)/lib64 -lcudart  -lSPHgrav $(LIBS) -L/cm/shared/apps/openmpi/gcc/64/1.8.1/lib64 -lmpi_mpifh
misc/makefiles/makefile.gonzales_ifort:	mv $(GPUEXEC) ..
misc/makefiles/makefile.gonzales_ifort:	echo ***MADE VERSION THAT USES GPUS***
misc/makefiles/makefile.gonzales_ifort:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
misc/makefiles/makefile.gonzales:#CUDAPATH = $(shell dirname $(shell dirname $(shell which nvcc)))
misc/makefiles/makefile.gonzales:CUDAPATH = /cm/shared/apps/cuda60/toolkit/current
misc/makefiles/makefile.gonzales:LIBS =  -lm #-L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
misc/makefiles/makefile.gonzales:GPUOBJS = $(FOBJS) $(COBJS) gpu_grav.o
misc/makefiles/makefile.gonzales:GPUEXEC = $(shell basename $(shell dirname $(shell pwd)))_gpu_sph
misc/makefiles/makefile.gonzales:gpu: $(GPUOBJS)
misc/makefiles/makefile.gonzales:	$(LD) -o $(GPUEXEC) $(LDFLAGS) $(GPUOBJS) $(LIBS) -L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
misc/makefiles/makefile.gonzales:	mv $(GPUEXEC) ..
misc/makefiles/makefile.gonzales:	echo ***MADE VERSION THAT USES GPUS***
misc/makefiles/makefile.gonzales:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
misc/makefiles/makefile:CUDAPATH = $(shell dirname $(shell dirname $(shell which nvcc)))
misc/makefiles/makefile:#CUDAPATH = /sw/kfs/cuda/5.0/linux_binary
misc/makefiles/makefile:LIBS =  -lm #-L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
misc/makefiles/makefile:GPUOBJS = $(FOBJS) $(COBJS) gpu_grav.o
misc/makefiles/makefile:GPUEXEC = $(shell basename $(shell dirname $(shell pwd)))_gpu_sph
misc/makefiles/makefile:gpu: $(GPUOBJS)
misc/makefiles/makefile:	$(LD) -o $(GPUEXEC) $(LDFLAGS) $(GPUOBJS) $(LIBS) -L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
misc/makefiles/makefile:	mv $(GPUEXEC) ..
misc/makefiles/makefile:	echo ***MADE VERSION THAT USES GPUS***
misc/makefiles/makefile:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
misc/makefiles/makefile.checkers:CUDAPATH = /global/scratch/software/cuda/cuda-3.2
misc/makefiles/makefile.checkers:LIBS = $(GRAVLIB) -LGPUsph_lib -lGPUsph_gatherscatter -L$(CUDAPATH)/lib -L/usr/mpi/gcc/openmpi-1.4/lib64 -L /home/jalombar/lib64 -lmpi -lmpi_f77 -lcudart
misc/makefiles/makefile.forge:CUDAPATH = #/global/scratch/software/cuda/cuda-3.2/lib
misc/makefiles/makefile.forge:LIBS = $(GRAVLIB) -LGPUsph_lib -lGPUsph_gatherscatter -L/usr/local/cuda/lib64 -L/usr/mpi/gcc/openmpi-1.4.2/lib64 -lmpi -lmpi_f77 -lcudart
misc/makefiles/Makefile-SPHgrav:CUDAPATH = /usr/local/cuda
misc/makefiles/Makefile-SPHgrav:LIBS = -L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart -lcuda
misc/makefiles/makefile.keeneland:#CUDAPATH = /usr/local/cuda
misc/makefiles/makefile.keeneland:#CUDAPATH = /sw/keeneland/cuda/4.2/linux_binary
misc/makefiles/makefile.keeneland:CUDAPATH = /sw/kfs/cuda/5.0/linux_binary
misc/makefiles/makefile.keeneland:LIBS = -L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart -lcuda -limf -lm
misc/makefiles/makefile.keeneland:GPUOBJS = $(FOBJS) $(COBJS) gpu_grav.o
misc/makefiles/makefile.keeneland:GPUEXEC = $(shell basename $(shell dirname $(shell pwd)))_gpu_sph
misc/makefiles/makefile.keeneland:gpu: $(GPUOBJS)
misc/makefiles/makefile.keeneland:	$(LD) -o $(GPUEXEC) $(LDFLAGS) $(GPUOBJS) $(LIBS) 
misc/makefiles/makefile.keeneland:	mv $(GPUEXEC) ..
misc/makefiles/makefile.keeneland:	echo ***MADE VERSION THAT USES GPUS***
misc/makefiles/makefile.keeneland:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
misc/makefiles/makefile.windows:#CUDAPATH = $(shell dirname $(shell dirname $(shell which nvcc)))
misc/makefiles/makefile.windows:CUDAPATH = /cygdrive/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v6.5
misc/makefiles/makefile.windows:LIBS =  -lm #-L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib/x64 -lcudart
misc/makefiles/makefile.windows:GPUOBJS = $(FOBJS) $(COBJS) gpu_grav.o
misc/makefiles/makefile.windows:GPUEXEC = $(shell basename $(shell dirname $(shell pwd)))_gpu_sph
misc/makefiles/makefile.windows:gpu: $(GPUOBJS)
misc/makefiles/makefile.windows:	$(LD) -o $(GPUEXEC) $(LDFLAGS) $(GPUOBJS) $(LIBS) -L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib/x64 -lcudart
misc/makefiles/makefile.windows:	mv $(GPUEXEC) ..
misc/makefiles/makefile.windows:	echo ***MADE VERSION THAT USES GPUS***
misc/makefiles/makefile.windows:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
misc/makefiles/makefile.lincoln:LIBS = $(GRAVLIB) -L. -lGPUsph_gatherscatter -LGPUsph_lib -L/usr/local/cuda-3.2/lib64 -lcudart -L/usr/local/mvapich2-1.2-intel-ofed-1.2.5.5/lib -lmpich -lmpichf90
misc/makefiles/makefile.grapefree2:CUDAPATH = /usr/local/cuda-5.0
misc/makefiles/makefile.grapefree2:LIBS =  -lm #-L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
misc/makefiles/makefile.grapefree2:GPUOBJS = $(FOBJS) $(COBJS) gpu_grav.o
misc/makefiles/makefile.grapefree2:GPUEXEC = $(shell basename $(shell dirname $(shell pwd)))_gpu_sph
misc/makefiles/makefile.grapefree2:gpu: $(GPUOBJS)
misc/makefiles/makefile.grapefree2:	$(LD) -o $(GPUEXEC) $(LDFLAGS) $(GPUOBJS) $(LIBS) -L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
misc/makefiles/makefile.grapefree2:	mv $(GPUEXEC) ..
misc/makefiles/makefile.grapefree2:	echo ***MADE VERSION THAT USES GPUS***
misc/makefiles/makefile.grapefree2:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
misc/makefiles/Makefile-SPHgrav.forge:CUDAPATH = /usr/local/cuda
misc/makefiles/Makefile-SPHgrav.forge:LIBS = -L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart -lcuda
parallel_bleeding_edge/README.md:Please make sure that you do not have more than one `cuda` library in your system. If you do, then make sure that the makefile in the `SPHgrav_lib2` and in the main `src` directories are (1) using the same one and (2) that you choose the more recent release.
parallel_bleeding_edge/README.md:First find out what CUDA card you have. The best is to run `$ lspci | grep VGA`
parallel_bleeding_edge/README.md:45:00.0 VGA compatible controller: NVIDIA Corporation TU102GL [Quadro
parallel_bleeding_edge/README.md:61:00.0 VGA compatible controller: NVIDIA Corporation TU102GL [Quadro
parallel_bleeding_edge/README.md:<a href="https://en.wikipedia.org/wiki/CUDA">https://en.wikipedia.org/wiki/CUDA</a>
parallel_bleeding_edge/README.md:(look for "Compute Capability, GPU semiconductors and Nvidia GPU board products"),
parallel_bleeding_edge/tools/sph.input_collision: ngravprocs=2, ! Number of GPUS (must be <= min(nprocs,ngravprocsmax)).
parallel_bleeding_edge/tools/sph.input_MESA: qthreads=0 ! number of gpu threads per particle. typically set to 1, 2, 4, or 8.  set to a negative value to optimize the number of threads by timing.  set to 0 to guess the best number of threads without timing.
parallel_bleeding_edge/tools/sph.input_MESA: computeexclusivemode=0, ! set this to 1 if on machine like grapefree with gpus in compute exclusive mode; set this to 0 on supercomputers like forge
parallel_bleeding_edge/MESA/MESA_initial_3D_models/sph.input: qthreads=0 ! number of gpu threads per particle. typically set to 1, 2, 4, or 8.  set to a negative value to optimize the number of threads by timing.  set to 0 to guess the best number of threads without timing.
parallel_bleeding_edge/MESA/MESA_initial_3D_models/sph.input: computeexclusivemode=0, ! set this to 1 if on machine like grapefree with gpus in compute exclusive mode; set this to 0 on supercomputers like forge
parallel_bleeding_edge/bin/StSm_run.sh:BIN=/projects/StarSmasher/starsmasher/parallel_bleeding_edge/bin/test_gpu_sph
parallel_bleeding_edge/src/makefile_templates/makefile.grapefree:CUDAPATH = /usr/local/cuda-5.0
parallel_bleeding_edge/src/makefile_templates/makefile.grapefree:LIBS =  -lm #-L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
parallel_bleeding_edge/src/makefile_templates/makefile.grapefree:GPUOBJS = $(FOBJS) gpu_grav.o
parallel_bleeding_edge/src/makefile_templates/makefile.grapefree:GPUEXEC = $(shell basename $(shell dirname $(shell pwd)))_gpu_sph
parallel_bleeding_edge/src/makefile_templates/makefile.grapefree:gpu: $(GPUOBJS)
parallel_bleeding_edge/src/makefile_templates/makefile.grapefree:	$(LD) -o $(GPUEXEC) $(LDFLAGS) $(GPUOBJS) $(LIBS) -L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
parallel_bleeding_edge/src/makefile_templates/makefile.grapefree:	mv $(GPUEXEC) ..
parallel_bleeding_edge/src/makefile_templates/makefile.grapefree:	echo ***MADE VERSION THAT USES GPUS***
parallel_bleeding_edge/src/makefile_templates/makefile.grapefree:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
parallel_bleeding_edge/src/makefile_templates/makefile.gonzales_ifort:CUDAPATH = /cm/shared/apps/cuda60/toolkit/6.0.37/
parallel_bleeding_edge/src/makefile_templates/makefile.gonzales_ifort:LIBS =  -lm #-L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
parallel_bleeding_edge/src/makefile_templates/makefile.gonzales_ifort:GPUOBJS = $(FOBJS) gpu_grav.o
parallel_bleeding_edge/src/makefile_templates/makefile.gonzales_ifort:GPUEXEC = $(shell basename $(shell dirname $(shell pwd)))_gpu_sph
parallel_bleeding_edge/src/makefile_templates/makefile.gonzales_ifort:gpu: $(GPUOBJS)
parallel_bleeding_edge/src/makefile_templates/makefile.gonzales_ifort:	$(LD) -o $(GPUEXEC) $(GPUOBJS) -L$(GRAVLIB) $(LDFLAGS) -L$(CUDAPATH)/lib64 -lcudart  -lSPHgrav $(LIBS) -L/cm/shared/apps/openmpi/gcc/64/1.8.1/lib64 -lmpi_mpifh
parallel_bleeding_edge/src/makefile_templates/makefile.gonzales_ifort:	mv $(GPUEXEC) ..
parallel_bleeding_edge/src/makefile_templates/makefile.gonzales_ifort:	echo ***MADE VERSION THAT USES GPUS***
parallel_bleeding_edge/src/makefile_templates/makefile.gonzales_ifort:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu20.04:# makefile is set for a laptop with Ubuntu 20.04 with a general NVIDIA graphic card, the standard cuda-toolkit, grortran 9.3 and openMPI 4.03
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu20.04:CUDAPATH =  /usr/lib/nvidia-cuda-toolkit/
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu20.04:#LDFLAGS = -lpthread -lifcore -lsvml -lifport -limf -lintlc -lrt -lstdc++ -lcudart
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu20.04:GPUOBJS = $(FOBJS) gpu_grav.o
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu20.04:GPUEXEC = test_gpu_sph
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu20.04:gpu: $(GPUOBJS)
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu20.04:	$(LD) -o $(GPUEXEC) $(LDFLAGS) $(LIBS) $(GPUOBJS) -L $(GRAVLIB) -lSPHgrav -L/usr/local/cuda/lib64 -lcudart  -lstdc++ 
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu20.04:	mv $(GPUEXEC) ..
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu20.04:	echo ***MADE VERSION THAT USES GPUS***
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu20.04:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
parallel_bleeding_edge/src/makefile_templates/makefile.applesilicon:# CUDAPATH = $(shell dirname $(shell dirname $(shell which nvcc)))
parallel_bleeding_edge/src/makefile_templates/makefile.applesilicon:LIBS =  -lm -lstdc++ #-L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
parallel_bleeding_edge/src/makefile_templates/makefile.applesilicon:# GPUOBJS = $(FOBJS) gpu_grav.o
parallel_bleeding_edge/src/makefile_templates/makefile.applesilicon:# GPUEXEC = $(shell basename $(shell dirname $(shell pwd)))_gpu_sph
parallel_bleeding_edge/src/makefile_templates/makefile.applesilicon:# gpu: $(GPUOBJS)
parallel_bleeding_edge/src/makefile_templates/makefile.applesilicon:	# $(LD) -o $(GPUEXEC) $(LDFLAGS) $(GPUOBJS) $(LIBS) -L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart -L/cm/shared/apps/openmpi/open64/64/1.10.1/lib64/ -lmpi_mpifh
parallel_bleeding_edge/src/makefile_templates/makefile.applesilicon:	# mv $(GPUEXEC) ..
parallel_bleeding_edge/src/makefile_templates/makefile.applesilicon:	# echo ***MADE VERSION THAT USES GPUS***
parallel_bleeding_edge/src/makefile_templates/makefile.applesilicon:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
parallel_bleeding_edge/src/makefile_templates/makefile.quest3:CUDAPATH = /software/cuda/cuda_5.0.35#/usr/local/cuda-5.0
parallel_bleeding_edge/src/makefile_templates/makefile.quest3:LIBS =  -lm #-L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
parallel_bleeding_edge/src/makefile_templates/makefile.quest3:GPUOBJS = $(FOBJS) gpu_grav.o
parallel_bleeding_edge/src/makefile_templates/makefile.quest3:GPUEXEC = $(shell basename $(shell dirname $(shell pwd)))_gpu_sph
parallel_bleeding_edge/src/makefile_templates/makefile.quest3:gpu: $(GPUOBJS)
parallel_bleeding_edge/src/makefile_templates/makefile.quest3:	$(LD) -o $(GPUEXEC) $(LDFLAGS) $(GPUOBJS) $(LIBS) -L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart -lmpi_mpifh #-L/cm/shared/apps/openmpi/gcc/64/1.8.1/lib64
parallel_bleeding_edge/src/makefile_templates/makefile.quest3:	mv $(GPUEXEC) ..
parallel_bleeding_edge/src/makefile_templates/makefile.quest3:	echo ***MADE VERSION THAT USES GPUS***
parallel_bleeding_edge/src/makefile_templates/makefile.quest3:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
parallel_bleeding_edge/src/makefile_templates/makefile.ubuntu:# makefile is set for laptop Ubuntu 16.04 with Nvidia GTX 1070, cuda 9.0, ifort version 18.0.01 and openmpi 2.1.1
parallel_bleeding_edge/src/makefile_templates/makefile.ubuntu:CUDAPATH = /usr/local/cuda-9.0
parallel_bleeding_edge/src/makefile_templates/makefile.ubuntu:LIBS =   -L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
parallel_bleeding_edge/src/makefile_templates/makefile.ubuntu:GPUOBJS = $(FOBJS) gpu_grav.o
parallel_bleeding_edge/src/makefile_templates/makefile.ubuntu:GPUEXEC = $(shell basename $(shell dirname $(shell pwd)))_gpu_sph
parallel_bleeding_edge/src/makefile_templates/makefile.ubuntu:gpu: $(GPUOBJS)
parallel_bleeding_edge/src/makefile_templates/makefile.ubuntu:	$(LD) -o $(GPUEXEC) $(LDFLAGS) $(LIBS) $(GPUOBJS) -lSPHgrav
parallel_bleeding_edge/src/makefile_templates/makefile.ubuntu:	mv $(GPUEXEC) ..
parallel_bleeding_edge/src/makefile_templates/makefile.ubuntu:	echo ***MADE VERSION THAT USES GPUS***
parallel_bleeding_edge/src/makefile_templates/makefile.ubuntu:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu:CUDAPATH = /usr/lib/nvidia-cuda-toolkit/
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu:CUDAPATH = $(shell dirname $(shell dirname $(shell which nvcc)))
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu:LIBS =  -lm -lstdc++ #-L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu:GPUOBJS = $(FOBJS) gpu_grav.o
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu:GPUEXEC = test_gpu_sph
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu:gpu: $(GPUOBJS)
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu:	$(LD) -o $(GPUEXEC) $(LDFLAGS) $(GPUOBJS) $(LIBS) -L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart -L/cm/shared/apps/openmpi/open64/64/1.10.1/lib64/ -lmpi_mpifh
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu:	mv $(GPUEXEC) ..
parallel_bleeding_edge/src/makefile_templates/makefileGPUubuntu:	echo ***MADE VERSION THAT USES GPUS***
parallel_bleeding_edge/src/makefile_templates/makefileCPUubuntu:CUDAPATH = /cm/shared/apps/cuda80/toolkit/8.0.44/
parallel_bleeding_edge/src/makefile_templates/makefileCPUubuntu:CUDAPATH = $(shell dirname $(shell dirname $(shell which nvcc)))
parallel_bleeding_edge/src/makefile_templates/makefileCPUubuntu:LIBS =  -lm -lstdc++ #-L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
parallel_bleeding_edge/src/makefile_templates/makefileCPUubuntu:GPUOBJS = $(FOBJS) gpu_grav.o
parallel_bleeding_edge/src/makefile_templates/makefileCPUubuntu:GPUEXEC = test_gpu_sph
parallel_bleeding_edge/src/makefile_templates/makefileCPUubuntu:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
parallel_bleeding_edge/src/gpu_grav.f:      subroutine set_nusegpus
parallel_bleeding_edge/src/gpu_grav.f:      integer nintvar,neos,nusegpus,nselfgravity
parallel_bleeding_edge/src/gpu_grav.f:      common/integration/nintvar,neos,nusegpus,nselfgravity
parallel_bleeding_edge/src/gpu_grav.f:      nusegpus=1
parallel_bleeding_edge/src/initialize_multiequalmass.f:               if(nusegpus.eq.1)then
parallel_bleeding_edge/src/cpu_grav.f:!     Relate to the CUDA code grav_force_direct.cu...
parallel_bleeding_edge/src/cpu_grav.f:!     Relate to the CUDA code grav_force_direct.cu...
parallel_bleeding_edge/src/cpu_grav.f:      subroutine set_nusegpus
parallel_bleeding_edge/src/cpu_grav.f:      integer nintvar,neos,nusegpus,nselfgravity,ncooling
parallel_bleeding_edge/src/cpu_grav.f:      common/integration/nintvar,neos,nusegpus,nselfgravity,ncooling
parallel_bleeding_edge/src/cpu_grav.f:      nusegpus=0
parallel_bleeding_edge/src/cpu_grav.f:      subroutine gpu_init_dev(i,theta_angle)
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:#include <cuda_runtime.h>
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:#if CUDART_VERSION >= 4000
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:#define CUT_DEVICE_SYNCHRONIZE( )   cudaDeviceSynchronize();
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:#define CUT_DEVICE_SYNCHRONIZE( )   cudaThreadSynchronize();
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                    \
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:    cudaError err = call;                                                    \
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:    if( cudaSuccess != err) {                                                \
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:                __FILE__, __LINE__, cudaGetErrorString( err) );              \
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:    //! Check for CUDA error
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:    cudaError_t err = cudaGetLastError();                                    \
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:    if( cudaSuccess != err) {                                                \
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:    if( cudaSuccess != err) {                                                \
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:    cudaError_t err = cudaGetLastError();                                    \
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:    if( cudaSuccess != err) {                                                \
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:    cudaError_t err = cudaGetLastError();
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:    if( cudaSuccess != err) {
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:        FPRINTF((stderr, "%s(%i) : cutilCheckMsg() CUTIL CUDA error : %s : (%d) %s.\n",
parallel_bleeding_edge/src/SPHgrav_lib2/cutil.h:                file, line, errorMessage, (int)err, cudaGetErrorString( err ) ));
parallel_bleeding_edge/src/SPHgrav_lib2/Makefile:# Take into account the localtion of CUDAPATH and the
parallel_bleeding_edge/src/SPHgrav_lib2/Makefile:# Path to your CUDA installation
parallel_bleeding_edge/src/SPHgrav_lib2/Makefile:# If that's the case, use the same CUDAPATH here and in the
parallel_bleeding_edge/src/SPHgrav_lib2/Makefile:CUDAPATH       := /usr/local/cuda-12.6
parallel_bleeding_edge/src/SPHgrav_lib2/Makefile:CUDAINCLUDE    := -I$(CUDAPATH)/include
parallel_bleeding_edge/src/SPHgrav_lib2/Makefile:NVCC           := $(CUDAPATH)/bin/nvcc
parallel_bleeding_edge/src/SPHgrav_lib2/Makefile:#NVCCFLAGS += -O4 -g  $(CUDAINCLUDE)  -I./ -Xptxas -v,-abi=no 
parallel_bleeding_edge/src/SPHgrav_lib2/Makefile:NVCCFLAGS += -O4 -g  $(CUDAINCLUDE)  -I./ -Xptxas -v
parallel_bleeding_edge/src/SPHgrav_lib2/Makefile:CUDA_LIBS = -L$(CUDAPATH)/lib64 -lcudart
parallel_bleeding_edge/src/SPHgrav_lib2/Makefile:LDGPUGLAGS := $(LDFLAGS) $(CUDA_LIBS)
parallel_bleeding_edge/src/SPHgrav_lib2/cuVector.h:    CUDA_SAFE_CALL(cudaMemcpy(host_pointer, data, size*sizeof(T), cudaMemcpyHostToHost));
parallel_bleeding_edge/src/SPHgrav_lib2/cuVector.h:    CUDA_SAFE_CALL(cudaMemcpy(host_pointer, data, _size*sizeof(T), cudaMemcpyHostToHost));
parallel_bleeding_edge/src/SPHgrav_lib2/cuVector.h:    CUDA_SAFE_CALL(cudaMalloc(&p, size * sizeof(T)));
parallel_bleeding_edge/src/SPHgrav_lib2/cuVector.h:      CUDA_SAFE_CALL(cudaMallocHost(&p, size * sizeof(T)));
parallel_bleeding_edge/src/SPHgrav_lib2/cuVector.h:      CUDA_SAFE_CALL(cudaFree(dev_pointer));
parallel_bleeding_edge/src/SPHgrav_lib2/cuVector.h:        CUDA_SAFE_CALL(cudaFreeHost(host_pointer));
parallel_bleeding_edge/src/SPHgrav_lib2/cuVector.h:      CUDA_SAFE_CALL(cudaMemcpy(dev_pointer, host_pointer, count * sizeof(T), cudaMemcpyHostToDevice));
parallel_bleeding_edge/src/SPHgrav_lib2/cuVector.h:      CUDA_SAFE_CALL(cudaMemcpy(host_pointer, dev_pointer, count * sizeof(T), cudaMemcpyDeviceToHost));
parallel_bleeding_edge/src/SPHgrav_lib2/grav_force_direct.cu:  cudaEvent_t start, stop;
parallel_bleeding_edge/src/SPHgrav_lib2/grav_force_direct.cu:    assert(cudaGetDeviceCount(&ndevice) == 0);
parallel_bleeding_edge/src/SPHgrav_lib2/grav_force_direct.cu:      fprintf(stderr, " SPHgrav found %d CUDA devices \n", ndevice);
parallel_bleeding_edge/src/SPHgrav_lib2/grav_force_direct.cu:      cudaDeviceProp p;
parallel_bleeding_edge/src/SPHgrav_lib2/grav_force_direct.cu:      assert(cudaGetDeviceProperties(&p, dev) == cudaSuccess);
parallel_bleeding_edge/src/SPHgrav_lib2/grav_force_direct.cu:    assert(cudaSetDevice(device) == cudaSuccess);
parallel_bleeding_edge/src/SPHgrav_lib2/grav_force_direct.cu:    cudaEventCreate( &start );
parallel_bleeding_edge/src/SPHgrav_lib2/grav_force_direct.cu:    cudaEventCreate( &stop  );
parallel_bleeding_edge/src/SPHgrav_lib2/grav_force_direct.cu:    cudaEventRecord( start, 0 );
parallel_bleeding_edge/src/SPHgrav_lib2/grav_force_direct.cu:    cudaEventRecord( stop, 0 );
parallel_bleeding_edge/src/SPHgrav_lib2/grav_force_direct.cu:    cudaDeviceSynchronize();
parallel_bleeding_edge/src/SPHgrav_lib2/grav_force_direct.cu:    cudaEventElapsedTime( &elapsed_time_ms, start, stop );
parallel_bleeding_edge/src/SPHgrav_lib2/grav_force_direct.cu:  void gpu_init_dev_(int *myrank)
parallel_bleeding_edge/src/SPHgrav_lib2/grav_force_direct.cu:  void gpu_init_dev_(int *myrank, double *theta)
parallel_bleeding_edge/src/grav.f:      if(nusegpus.eq.0)return
parallel_bleeding_edge/src/Makefile:CUDAPATH = /usr/local/cuda-12.6/
parallel_bleeding_edge/src/Makefile:LIBS =  -lm -lstdc++   #-L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart
parallel_bleeding_edge/src/Makefile:GPUOBJS = $(FOBJS) gpu_grav.o
parallel_bleeding_edge/src/Makefile:GPUEXEC = test_gpu_sph
parallel_bleeding_edge/src/Makefile:gpu: $(GPUOBJS)
parallel_bleeding_edge/src/Makefile:	$(LD) -o $(GPUEXEC) $(LDFLAGS) $(GPUOBJS) $(LIBS)           \
parallel_bleeding_edge/src/Makefile:	   	-L$(GRAVLIB) -lSPHgrav -L$(CUDAPATH)/lib64 -lcudart \
parallel_bleeding_edge/src/Makefile:	mv $(GPUEXEC) ../bin/
parallel_bleeding_edge/src/Makefile:	echo ***MADE VERSION THAT USES GPUS***
parallel_bleeding_edge/src/Makefile:	echo ***MADE VERSION THAT DOES NOT NEED GPUS***
parallel_bleeding_edge/src/init.f:         if(nusegpus.eq.0)then
parallel_bleeding_edge/src/init.f:            write(69,*)'gpus will be used for gravity'
parallel_bleeding_edge/src/init.f:      qthreads=0               ! number of gpu threads per particle. typically set to 1, 2, 4, or 8.  set to a negative value to optimize the number of threads by timing.  set to 0 to guess the best number of threads without timing.
parallel_bleeding_edge/src/init.f:      computeexclusivemode=0   ! set this to 1 if on machine like grapefree with gpus in compute exclusive mode; set this to 0 on supercomputers like lincoln
parallel_bleeding_edge/src/init.f:      call set_nusegpus         ! if using gpus, this sets nusegpus=1 *and* nselfgravity=1
parallel_bleeding_edge/src/init.f:      if(ngr.ne.0 .and. nusegpus.eq.0) then
parallel_bleeding_edge/src/eatem.f:               if(nusegpus.eq.1)then
parallel_bleeding_edge/src/changetf.f:c     one outfile to the next.... (this happens in some of evghenii's gpu runs)
parallel_bleeding_edge/src/initialize_polyes.f:               if(nusegpus.eq.1)then
parallel_bleeding_edge/src/skipahead.f:         if(nusegpus.eq.1)then
parallel_bleeding_edge/src/skipahead.f:         if(nusegpus.eq.1)then
parallel_bleeding_edge/src/skipahead.f:            if(nusegpus.eq.1)then
parallel_bleeding_edge/src/starsmasher.h:      integer neos,nusegpus,nselfgravity,ncooling,nkernel
parallel_bleeding_edge/src/starsmasher.h:      common/integration/nintvar,neos,nusegpus,nselfgravity,ncooling,nkernel
parallel_bleeding_edge/src/main.f:!     the following line assumes ppn cpu threads and abs(ngravprocs) gpu threads per node
parallel_bleeding_edge/src/main.f:!         call gpu_init_dev(myrank/((nprocs+ppn-1)/ppn)) ! if the gpus are set up in device exclusive mode,
parallel_bleeding_edge/src/main.f:      if(myrank.lt.ngravprocs .and. .not. alreadyinitialized .and. computeexclusivemode.ne.1 .and. nusegpus.eq.1) then
parallel_bleeding_edge/src/main.f:!          gpus must always be initialized, even if we use just 1 mpi process
parallel_bleeding_edge/src/main.f:         call gpu_init_dev(myrank/((nprocs+ppn-1)/ppn), theta_angle) ! if the gpus are set up in device exclusive mode,
parallel_bleeding_edge/src/main.f:         write(6,"('myrank=',I3,' is running on ',A,' with gpu',i3)")
parallel_bleeding_edge/src/main.f:         if(nusegpus.eq.1) then
parallel_bleeding_edge/src/main.f:     $              'gpurank,gravdispl,gravrecvcount=',
parallel_bleeding_edge/src/main.f:         if(myrank.lt.ngravprocs .and. nusegpus.eq.1) then
parallel_bleeding_edge/src/main.f:c     the first call to a gpu is usually slow, so let's not time this one:
parallel_bleeding_edge/src/initialize_parent.f:               if(nusegpus.eq.1)then
parallel_bleeding_edge/src/initialize_parent.f.orig:               if(nusegpus.eq.1)then
parallel_bleeding_edge/src/balAV3.f:         if(nusegpus.eq.1)then
parallel_bleeding_edge/src/balAV3.f:            if(nusegpus.eq.1)then
parallel_bleeding_edge/src/balAV3.f:         if(nusegpus.eq.1)then
splot_routines/Mass_orbits_spin/pplot.f:     $      ppn,neos,nselfgravity,nusegpus,ncooling
splot_routines/Mass_orbits_spin/pplot.f:     $      ppn,omega_spin,neos,nselfgravity,gam,reat,nusegpus,
splot_routines/Mass_orbits_spin/pplot.f:      common/integration/nintvar,neos,nusegpus,nselfgravity,ncooling
splot_routines/Mass_orbits_spin/makefile:LIBS      = #-L../. -lGPUsph_gs -L/usr/local/cuda/lib64 -lcudart #-lcuda #-L/opt/MDGRAPE3/lib -lmdgrape3
splot_routines/Mass_orbits_spin/robusttrajectories.f:c     one outfile to the next.... (this happens in some of evghenii's gpu runs)

```
