# https://github.com/tbronzwaer/raptor

```console
core.c:                clock_t startgpu=clock();
core.c:                        diff=clock()-startgpu;
core.c:                        print_time(startgpu);
makefile:	GPU=0
makefile:	OPENACC=0
makefile:ifeq ($(OPENACC),1)
makefile:	CFLAGS = -acc -fast ‑Mipa=inline,reshape  -Minfo=accel -ta=tesla:cuda8.0,fastmath,maxregcount:255
parameters.h:// OpenACC or OMP

```
