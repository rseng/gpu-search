# https://github.com/alphaparrot/ExoPlaSim

```console
exoplasim/sam/src/sam.f90:      real, target,  allocatable  :: gpu(:)    ! Term u * p
exoplasim/sam/src/sam.f90:allocate(gpu(nhor))   ; gpu(:)  = 0.0  ! Term u * p
exoplasim/sam/src/sam.f90:      gpu =   -gu * gp
exoplasim/sam/src/sam.f90:      call gp2fc(gpu ,NLON,NLPP)
exoplasim/sam/src/sam.f90:         call filter_zonal_waves(gpu)
exoplasim/sam/src/sam.f90:      call mktend(sdf,spf,szf,gtn,gfu,gfv,gke,gpu,gpv)
exoplasim/pyburn.py:                gpuvar = np.asfortranarray(
exoplasim/pyburn.py:                gpuvar = np.asfortranarray(
exoplasim/pyburn.py:            gpuvar, gpvvar = pyfft.spvgp(spuvar,spvvar,rdcostheta,nlon,ntru,int(physfilter))
exoplasim/pyburn.py:            fcuvar = pyfft.gp3fc(gpuvar)
exoplasim/pyburn.py:                gpuvar = np.asfortranarray(
exoplasim/pyburn.py:                gpuvar = np.asfortranarray(
exoplasim/pyburn.py:            fcuvar = pyfft.gp3fc(gpuvar)
exoplasim/pyburn.py:            gpuvar = np.asfortranarray(
exoplasim/pyburn.py:            gpuvar = np.asfortranarray(
exoplasim/pyburn.py:        fcuvar = pyfft.gp3fc(gpuvar)

```
