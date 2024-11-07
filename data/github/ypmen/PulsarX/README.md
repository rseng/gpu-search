# https://github.com/ypmen/PulsarX

```console
config/libtool.m4:    nvcc*) # Cuda Compiler Driver 2.2
config/libtool.m4:	nvcc*)	# Cuda Compiler Driver 2.2
config/cuda.m4:# AX_CHECK_CUDA
config/cuda.m4:# Figures out if CUDA Driver API/nvcc is available, i.e. existence of:
config/cuda.m4:# 	cuda.h
config/cuda.m4:#   libcuda.so
config/cuda.m4:#   CUDA_CFLAGS and 
config/cuda.m4:#   CUDA_LDFLAGS.
config/cuda.m4:# The author is personally using CUDA such that the .cu code is generated
config/cuda.m4:AC_DEFUN([AX_CHECK_CUDA], [
config/cuda.m4:# Provide your CUDA path with this		
config/cuda.m4:AC_ARG_WITH(cuda, [  --with-cuda=PREFIX      Prefix of your CUDA installation], [cuda_prefix=$withval], [cuda_prefix="/usr/local/cuda"])
config/cuda.m4:# Setting the prefix to the default if only --with-cuda was given
config/cuda.m4:if test "$cuda_prefix" == "yes"; then
config/cuda.m4:		cuda_prefix="/usr/local/cuda"
config/cuda.m4:AC_MSG_CHECKING([nvcc in $cuda_prefix/bin])
config/cuda.m4:if test -x "$cuda_prefix/bin/nvcc"; then
config/cuda.m4:	AC_DEFINE_UNQUOTED([NVCC_PATH], ["$cuda_prefix/bin/nvcc"], [Path to nvcc binary])
config/cuda.m4:	# We need to add the CUDA search directories for header and lib searches
config/cuda.m4:	CUDA_CFLAGS=""
config/cuda.m4:	 AC_SUBST([CUDA_PATH],[$cuda_prefix])
config/cuda.m4:	AC_SUBST([CUDA_CFLAGS])
config/cuda.m4:	AC_SUBST([CUDA_LDFLAGS])
config/cuda.m4:	AC_SUBST([NVCC],[$cuda_prefix/bin/nvcc])
config/cuda.m4:	AC_CHECK_FILE([$cuda_prefix/lib64],[lib64_found=yes],[lib64_found=no])
config/cuda.m4:		AC_CHECK_FILE([$cuda_prefix/lib],[lib32_found=yes],[lib32_found=no])
config/cuda.m4:			AC_SUBST([CUDA_LIBDIR],[$cuda_prefix/lib])
config/cuda.m4:			AC_MSG_WARN([Couldn't find cuda lib directory])
config/cuda.m4:			VALID_CUDA=no
config/cuda.m4:			AC_SUBST([CUDA_LIBDIR],[$cuda_prefix/lib64])
config/cuda.m4:			CUDA_CFLAGS+=" -m64"
config/cuda.m4:			AC_CHECK_FILE([$cuda_prefix/lib32],[lib32_found=yes],[lib32_found=no])
config/cuda.m4:				AC_SUBST([CUDA_LIBDIR],[$cuda_prefix/lib])
config/cuda.m4:				CUDA_CFLAGS+=" -m32"
config/cuda.m4:				AC_MSG_WARN([Couldn't find cuda lib directory])
config/cuda.m4:				VALID_CUDA=no
config/cuda.m4:	if test "x$VALID_CUDA" != xno ; then
config/cuda.m4:		CUDA_CFLAGS+=" -I$cuda_prefix/include"
config/cuda.m4:		CFLAGS="$CUDA_CFLAGS $CFLAGS"
config/cuda.m4:		CUDA_LDFLAGS="-L$CUDA_LIBDIR"
config/cuda.m4:		LDFLAGS="$CUDA_LDFLAGS $LDFLAGS"
config/cuda.m4:		AC_CHECK_HEADER([cuda.h], [],
config/cuda.m4:			AC_MSG_WARN([Couldn't find cuda.h])
config/cuda.m4:			VALID_CUDA=no
config/cuda.m4:			,[#include <cuda.h>])
config/cuda.m4:		if test "x$VALID_CUDA" != "xno" ; then
config/cuda.m4:			AC_CHECK_LIB([cuda], [cuInit], [VALID_CUDA=yes], AC_MSG_WARN([Couldn't find libcuda]
config/cuda.m4:			VALID_CUDA=no))
config/cuda.m4:	AC_MSG_WARN([nvcc was not found in $cuda_prefix/bin])
config/cuda.m4:	VALID_CUDA=no
config/cuda.m4:AC_ARG_ENABLE(cuda, [  --enable-cuda  enable cuda [default=no]], [cuda=true], [cuda=false])
config/cuda.m4:if test "x$enable_cuda" = xyes && test x$VALID_CUDA = xyes ; then 
config/cuda.m4:	AC_MSG_NOTICE([Building with CUDA bindings])
config/cuda.m4:elif test "x$enable_cuda" = xyes && test x$VALID_CUDA = xno ; then 
config/cuda.m4:	AC_MSG_ERROR([Cannot build CUDA bindings. Check errors])
config/cuda.m4:AM_CONDITIONAL(ENABLE_CUDA,[test "$enable_cuda" = "yes"])
configure.ac:AX_CHECK_CUDA

```
