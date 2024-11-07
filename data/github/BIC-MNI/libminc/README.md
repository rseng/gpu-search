# https://github.com/BIC-MNI/libminc

```console
libsrc2/minc_compat2.h:#define ncclose miclose
testdir/test_mconv.c:   (void) ncclose(cdfid);
libsrc/netcdf_convenience.c:#undef ncclose
libsrc/netcdf_convenience.c:      (void) ncclose(status);
libsrc/netcdf_convenience.c:@DESCRIPTION: A wrapper for routine ncclose, allowing future enhancements.
libsrc/netcdf_convenience.c:       status = ncclose(cdfid);
libsrc/netcdf_convenience.c:   status = ncclose(cdfid);
libsrc/minc_compat.h:#define ncclose miclose

```
