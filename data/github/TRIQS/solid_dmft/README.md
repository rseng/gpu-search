# https://github.com/TRIQS/solid_dmft

```console
Docker/openmpi_qe.make.inc:# GPU architecture (Kepler: 35, Pascal: 60, Volta: 70 )
Docker/openmpi_qe.make.inc:GPU_ARCH=
Docker/openmpi_qe.make.inc:# CUDA runtime (Pascal: 8.0, Volta: 9.0)
Docker/openmpi_qe.make.inc:CUDA_RUNTIME=
Docker/openmpi_qe.make.inc:# CUDA F90 Flags
Docker/openmpi_qe.make.inc:CUDA_F90FLAGS=
Docker/openmpi_qe.make.inc:F90FLAGS       = $(FFLAGS) -cpp $(FDFLAGS) $(CUDA_F90FLAGS) $(IFLAGS) $(MODFLAGS)
Docker/openmpi_qe.make.inc:# CUDA libraries
Docker/openmpi_qe.make.inc:CUDA_LIBS=
Docker/openmpi_qe.make.inc:CUDA_EXTLIBS = 
Docker/openmpi_qe.make.inc:QELIBS         = $(CUDA_LIBS) $(SCALAPACK_LIBS) $(LAPACK_LIBS) $(FOX_LIB) $(FFT_LIBS) $(BLAS_LIBS) $(MPI_LIBS) $(BEEF_LIBS) $(MASS_LIBS) $(HDF5_LIBS) $(LIBXC_LIBS) $(LD_LIBS)

```
