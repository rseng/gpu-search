# https://github.com/zenitani/OpenMHD

```console
2D_reconnection_gpu/parallel.cuf:  use cudafor
2D_reconnection_gpu/parallel.cuf:  integer(kind=cuda_stream_kind), private :: comm_stream
2D_reconnection_gpu/parallel.cuf:    use cudafor
2D_reconnection_gpu/parallel.cuf:    stat = cudaStreamCreate(comm_stream)
2D_reconnection_gpu/parallel.cuf:    use cudafor
2D_reconnection_gpu/parallel.cuf:    stat = cudaStreamDestroy(comm_stream)
2D_reconnection_gpu/parallel.cuf:    use cudafor
2D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:    use cudafor
2D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
2D_reconnection_gpu/flux_resistive.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
2D_reconnection_gpu/mainp.cuf:!     2020/08/04  S. Zenitani  CUDA + MPI version
2D_reconnection_gpu/mainp.cuf:  use cudafor
2D_reconnection_gpu/mpibc.cuf:!     Boundary conditions (parallel CUDA version)
2D_reconnection_gpu/Makefile:### NVIDIA HPC SDK
2D_reconnection_gpu/Makefile:# F90 = nvfortran -cuda -O2
2D_reconnection_gpu/Makefile:# F90 = mpif90 -cuda -O2
2D_reconnection_gpu/Makefile:F90 = mpif90 -cuda -O2 -mcmodel=medium
2D_reconnection_gpu/Makefile:# F90 = mpif90 -cuda -O2 -mcmodel=medium -gpu=cuda11.6,cc80 -tp=zen2
2D_reconnection_gpu/main.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
2D_reconnection_gpu/main.cuf:  use cudafor
2D_reconnection_gpu/limiter.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
2D_reconnection_gpu/flux_solver.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
2D_KH_gpu/Makefile:### NVIDIA HPC SDK
2D_KH_gpu/Makefile:F90 = nvfortran -cuda -O2
2D_KH_gpu/Makefile:# F90 = mpif90 -cuda -O2
2D_KH_gpu/Makefile:# F90 = mpif90 -cuda -O2 -mcmodel=medium
2D_KH_gpu/Makefile:# F90 = mpif90 -cuda -O2 -mcmodel=medium -gpu=cuda11.6,cc80 -tp=zen2
2D_KH_gpu/main.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
2D_KH_gpu/main.cuf:  use cudafor
2D_KH_gpu/limiter.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
2D_KH_gpu/flux_solver.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
3D_basic_gpu/parallel.cuf:  use cudafor
3D_basic_gpu/parallel.cuf:  integer(kind=cuda_stream_kind), private :: comm_stream
3D_basic_gpu/parallel.cuf:    use cudafor
3D_basic_gpu/parallel.cuf:    stat = cudaStreamCreate(comm_stream)
3D_basic_gpu/parallel.cuf:    use cudafor
3D_basic_gpu/parallel.cuf:    stat = cudaStreamDestroy(comm_stream)
3D_basic_gpu/parallel.cuf:    use cudafor
3D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:    use cudafor
3D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
3D_basic_gpu/mainp.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
3D_basic_gpu/mainp.cuf:  use cudafor
3D_basic_gpu/Makefile:### NVIDIA HPC SDK
3D_basic_gpu/Makefile:# F90 = nvfortran -cuda -O2
3D_basic_gpu/Makefile:# F90 = mpif90 -cuda -O2
3D_basic_gpu/Makefile:F90 = mpif90 -cuda -O2 -mcmodel=medium
3D_basic_gpu/Makefile:# F90 = mpif90 -cuda -O2 -mcmodel=medium -gpu=cuda11.6,cc80 -tp=zen2
3D_basic_gpu/main.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
3D_basic_gpu/main.cuf:  use cudafor
3D_basic_gpu/limiter.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
3D_basic_gpu/flux_solver.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
2D_basic_gpu/parallel.cuf:  use cudafor
2D_basic_gpu/parallel.cuf:  integer(kind=cuda_stream_kind), private :: comm_stream
2D_basic_gpu/parallel.cuf:    use cudafor
2D_basic_gpu/parallel.cuf:    stat = cudaStreamCreate(comm_stream)
2D_basic_gpu/parallel.cuf:    use cudafor
2D_basic_gpu/parallel.cuf:    stat = cudaStreamDestroy(comm_stream)
2D_basic_gpu/parallel.cuf:    use cudafor
2D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:    use cudafor
2D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
2D_basic_gpu/mainp.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
2D_basic_gpu/mainp.cuf:  use cudafor
2D_basic_gpu/Makefile:### NVIDIA HPC SDK
2D_basic_gpu/Makefile:# F90 = nvfortran -cuda -O2
2D_basic_gpu/Makefile:# F90 = mpif90 -cuda -O2
2D_basic_gpu/Makefile:F90 = mpif90 -cuda -O2 -mcmodel=medium
2D_basic_gpu/Makefile:# F90 = mpif90 -cuda -O2 -mcmodel=medium -gpu=cuda11.6,cc80 -tp=zen2
2D_basic_gpu/main.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
2D_basic_gpu/main.cuf:  use cudafor
2D_basic_gpu/limiter.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
2D_basic_gpu/flux_solver.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
3D_reconnection_gpu/parallel.cuf:  use cudafor
3D_reconnection_gpu/parallel.cuf:  integer(kind=cuda_stream_kind), private :: comm_stream
3D_reconnection_gpu/parallel.cuf:    use cudafor
3D_reconnection_gpu/parallel.cuf:    stat = cudaStreamCreate(comm_stream)
3D_reconnection_gpu/parallel.cuf:    use cudafor
3D_reconnection_gpu/parallel.cuf:    stat = cudaStreamDestroy(comm_stream)
3D_reconnection_gpu/parallel.cuf:    use cudafor
3D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:    use cudafor
3D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:          stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:             stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/parallel.cuf:       stat = cudaStreamSynchronize(comm_stream)
3D_reconnection_gpu/flux_resistive.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
3D_reconnection_gpu/mainp.cuf:!     2020/08/04  S. Zenitani  CUDA + MPI version
3D_reconnection_gpu/mainp.cuf:  use cudafor
3D_reconnection_gpu/mpibc.cuf:!     Boundary conditions (parallel CUDA version)
3D_reconnection_gpu/Makefile:### NVIDIA HPC SDK
3D_reconnection_gpu/Makefile:# F90 = nvfortran -cuda -O2
3D_reconnection_gpu/Makefile:# F90 = mpif90 -cuda -O2
3D_reconnection_gpu/Makefile:F90 = mpif90 -cuda -O2 -mcmodel=medium
3D_reconnection_gpu/Makefile:# F90 = mpif90 -cuda -O2 -mcmodel=medium -gpu=cuda11.6,cc80 -tp=zen2
3D_reconnection_gpu/main.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
3D_reconnection_gpu/main.cuf:  use cudafor
3D_reconnection_gpu/limiter.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version
3D_reconnection_gpu/flux_solver.cuf:!     2020/07/04  S. Zenitani  CUDA fortran version

```
