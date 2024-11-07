# https://github.com/HDFGroup/hdf5

```console
config/nvidia-flags:# NVIDIA nvc compiler or a derivative.  It is careful not to do anything
config/nvidia-flags:        echo "compiler '$CC' is NVIDIA $cc_vendor-$cc_version"
config/nvidia-flags:        # NVIDIA version numbers are of the form: "major.minor-patch"
config/nvidia-fflags:# NVIDIA nvfortran compiler or a derivative.  It is careful not to do anything
config/nvidia-fflags:# if the compiler is not NVIDIA; otherwise `f9x_flags_set' is set to `yes'
config/nvidia-fflags:# Get the compiler version in a way that works for NVIDIA nvfortran
config/nvidia-fflags:        echo "compiler '$FC' is NVIDIA $f9x_vendor-$f9x_version"
config/nvidia-fflags:        # NVIDIA version numbers are of the form: "major.minor-patch"
config/nvidia-fflags:# Common NVIDIA flags for various situations
config/nvidia-fflags:    Fortran_COMPILER_ID=NVIDIA
config/linux-gnu:. $srcdir/config/nvidia-flags
config/linux-gnu:. $srcdir/config/nvidia-fflags
config/linux-gnu:. $srcdir/config/nvidia-cxxflags
config/nvidia-cxxflags:# NVIDIA nvc++ compiler or a derivative.  It is careful not to do anything
config/nvidia-cxxflags:# if the compiler is not NVIDIA; otherwise `cxx_flags_set' is set to `yes'
config/nvidia-cxxflags:# Get the compiler version in a way that works for NVIDIA nvc++
config/nvidia-cxxflags:        echo "compiler '$CXX' is NVIDIA $cxx_vendor-$cxx_version"
config/nvidia-cxxflags:        # NVIDIA version numbers are of the form: "major.minor-patch"
fortran/test/tH5F.F90:!  file_close, file_space, h5openclose, test_get_file_image
fortran/test/tH5F.F90:  SUBROUTINE h5openclose(total_error)
fortran/test/tH5F.F90:  END SUBROUTINE h5openclose
fortran/test/fortranlib_test.F90:  CALL h5openclose(ret_total_error)
ACKNOWLEDGMENTS:NVIDIA, for contributing multithreaded concurrency support.
tools/lib/io_timer.h:    HDF5_FILE_OPENCLOSE,
release_docs/RELEASE.txt:                                     NVIDIA nvc, nvfortran and nvc++ version 22.5-0
release_docs/RELEASE.txt:    23.11. If you are using an affected version of the NVidia compiler, the
release_docs/RELEASE.txt:    https://forums.developer.nvidia.com/t/hdf5-no-longer-compiles-with-nv-23-9/269045
release_docs/HISTORY-1_14_0-2_0_0.txt:                                     NVIDIA nvc, nvfortran and nvc++ version 22.5-0
release_docs/HISTORY-1_14_0-2_0_0.txt:    23.11. If you are using an affected version of the NVidia compiler, the
release_docs/HISTORY-1_14_0-2_0_0.txt:    https://forums.developer.nvidia.com/t/hdf5-no-longer-compiles-with-nv-23-9/269045
release_docs/HISTORY-1_14_0-2_0_0.txt:                                     NVIDIA nvc, nvfortran and nvc++ version 22.5-0
release_docs/HISTORY-1_14_0-2_0_0.txt:    23.11. If you are using an affected version of the NVidia compiler, the
release_docs/HISTORY-1_14_0-2_0_0.txt:    https://forums.developer.nvidia.com/t/hdf5-no-longer-compiles-with-nv-23-9/269045
release_docs/HISTORY-1_14_0-2_0_0.txt:                                     NVIDIA nvc, nvfortran and nvc++ version 22.5-0
release_docs/HISTORY-1_14_0-2_0_0.txt:                                     NVIDIA nvc, nvfortran and nvc++ version 22.5-0
src/H5Rpublic.h:#include "H5Gpublic.h" /* Groups                                   */
src/H5Rpublic.h: *          following valid object type values (defined in H5Gpublic.h):
src/H5Rpublic.h: *          \snippet H5Gpublic.h H5G_obj_t_snip
src/H5Rdeprec.c: * Return:      Success:    Object type (as defined in H5Gpublic.h)
src/H5Gprivate.h:#include "H5Gpublic.h"
src/hdf5.h:#include "H5Gpublic.h"  /* Groups                                   */
src/CMakeLists.txt:    ${HDF5_SRC_DIR}/H5Gpublic.h
src/Makefile.am:        H5Gpublic.h  H5Ipublic.h H5Lpublic.h \
src/H5Gpublic.h: * Created:             H5Gpublic.h
src/H5Gpublic.h:#ifndef H5Gpublic_H
src/H5Gpublic.h:#define H5Gpublic_H
src/H5Gpublic.h: *          \p ginfo is an H5G_info_t struct and is defined (in H5Gpublic.h)
src/H5Gpublic.h: *          \p ginfo is an H5G_info_t struct and is defined (in H5Gpublic.h)
src/H5Gpublic.h: *          H5Gpublic.h):
src/H5Gpublic.h:#endif /* H5Gpublic_H */

```
