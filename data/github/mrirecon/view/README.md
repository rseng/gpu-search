# https://github.com/mrirecon/view

```console
Makefile:CUDA?=0
Makefile:CUDA_BASE ?= /usr/local/cuda
Makefile:CUDA_LIB ?= lib
Makefile:ifeq ($(CUDA),1)
Makefile:    CUDA_L := -L$(CUDA_BASE)/$(CUDA_LIB) -lcufft -lcudart -lcublas -lblas
Makefile:    CUDA_L :=
Makefile:	$(CC) $(CFLAGS) $(CPPFLAGS) $(EXPDYN) -o view -I$(TOOLBOX_INC) `$(PKG_CONFIG) --cflags gtk+-3.0` src/main.c src/view.c src/gtk_ui.c src/draw.c `$(PKG_CONFIG) --libs gtk+-3.0` $(TOOLBOX_LIB)/libmisc.a $(TOOLBOX_LIB)/libgeom.a $(TOOLBOX_LIB)/libnum.a $(TOOLBOX_LIB)/libmisc.a $(CUDA_L) $(LDFLAGS)
Makefile:	$(CC) $(CFLAGS) $(CPPFLAGS) $(EXPDYN) -o cfl2png -I$(TOOLBOX_INC) src/cfl2png.c src/draw.c $(TOOLBOX_LIB)/libmisc.a  $(TOOLBOX_LIB)/libgeom.a $(TOOLBOX_LIB)/libnum.a $(TOOLBOX_LIB)/libmisc.a $(CUDA_L) $(LDFLAGS)

```
