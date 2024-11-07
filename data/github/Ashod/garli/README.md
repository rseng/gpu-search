# https://github.com/Ashod/garli

```console
config/libtool.m4:    nvcc*) # Cuda Compiler Driver 2.2
config/libtool.m4:	nvcc*)	# Cuda Compiler Driver 2.2
src/funcs.cpp:#ifdef BROOK_GPU
src/garlimain.cpp:#ifdef CUDA_GPU
src/garlimain.cpp:#include "cudaman.h"
src/garlimain.cpp:CudaManager *cudaman;
src/garlimain.cpp:int cuda_device_number=0;
src/garlimain.cpp:#ifdef CUDA_GPU
src/garlimain.cpp:	outman.UserMessage    ("  --device d_number	use specified CUDA device");
src/garlimain.cpp:#ifdef CUDA_GPU
src/garlimain.cpp:					else if(!_stricmp(argv[curarg], "--device")) cuda_device_number = atoi(argv[++curarg]);
src/garlimain.cpp:#ifdef CUDA_GPU
src/garlimain.cpp:			outman.UserMessage("->CUDA GPU version<-\n");

```
