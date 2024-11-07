# https://github.com/teuben/nemo

```console
docs/source/codes.rst:      N-body GPU tree-code, in :term:`AMUSE`.
docs/source/codes.rst:      Gravitational ENcounters with Gpu Acceleration
docs/source/codes.rst:  HiGPUs
docs/source/codes.rst:      A GPU-accelerated hybrid symplectic integrator
data/testsuite.log:### Warning [snapstack]: zerocm=false is now the default!!
text/bugs/problems:1	./josh/nbody/zerocms.o
text/bugs/Illinois:			 Piet should add a zerocm=f/t to his mkplummer
text/bugs/Illinois:			  'zerocm' to allow yes/no C.O.M. shift, as
text/bugs/Illinois:			  now has extra keyword 'zerocm=t'
text/manybody/manybody.tex:mkplummer out=??? nbody=??? mfrac=0.999 rfrac=22.8042468 seed=0 time=0.0 zerocm=t 
scripts/csh/mktt72.sh:snapstack $run.1r $run.2 $run.3 5,2,0 -1,0,0  zerocm=t
scripts/csh/twoplummers:snapstack p1 p1 p3 deltar=$dr,$dy,0 deltav=-$dv,0,0 zerocm=t
scripts/csh/plummer_collision.sh:snapstack $run.1 $run.2 $run.3 deltar=$r0,$rp,0 deltav=-$v0,0,0  zerocm=t
scripts/csh/bench1: mkplummer - $nbody zerocm=f |\
scripts/csh/bench1:    mkplummer $tmp.1 $nbody zerocm=f 
scripts/csh/centering.sh:	mkplummer - nbody=$nbody seed=$seed zerocm=f | snapsort - - rank=r | snapshift - - $shift,0,0 > p.dat
scripts/csh/centering.sh:	mkommod $NEMODAT/plum.dat          - nbody=$nbody seed=$seed zerocm=f | snapsort - - rank=r | snapshift - - $shift,0,0 > p.dat
scripts/csh/centering.sh:	mkommod $NEMODAT/devauc.dat        - nbody=$nbody seed=$seed zerocm=f | snapsort - - rank=r | snapshift - - $shift,0,0 > p.dat	
scripts/csh/centering.sh:	mkommod $NEMODAT/k${model}isot.dat - nbody=$nbody seed=$seed zerocm=f | snapsort - - rank=r | snapshift - - $shift,0,0 > p.dat
scripts/csh/centering.sh:	mkommod $NEMODAT/k${model}isot.dat - nbody=$nbody seed=$seed zerocm=f | snapsort - - rank=r | snapshift - - $shift,0,0 > p.dat
scripts/csh/centering.sh:	mkommod $NEMODAT/k${model}isot.dat - nbody=$nbody seed=$seed zerocm=f | snapsort - - rank=r | snapshift - - $shift,0,0 > p.dat
scripts/csh/mkmh97.md:4. **zerocm=**:  if set to true, apply center of mass to the input model *not implemented yet*
scripts/csh/mkmh97.md:   compile. code=2 is reserved for GPU enabled machines where **bonsai2**
scripts/csh/mkmh97.md:   in the GPU memory.
scripts/csh/mk_galaxy_clumpy:    Qtoomre=$Qtoomre z0=$z0 zerocm=f seed=$seed
scripts/csh/mkmh97.sh:#           6-dec-2023   added zerocm=t|f and zm=0|1
scripts/csh/mkmh97.sh:zerocm=t        # apply a zerocm to models (@todo needs more careful check where to use)
scripts/csh/mkmh97.sh:code=1          # 0=hackcode1  0q=hackcode1_qp  1=gyrfalcON   2=bonsai2 (GPU)  3=rungravidy  4=directcodeb
scripts/csh/mkmh97.sh:save_vars="run nbody m em zm fixed potname potpars zerocm step v0 rp r0 eps kmax eta code seed trim tstop box r16 vbox npixel power bsigma tplot yapp"
scripts/csh/mkmh97.sh:    mkplummer $run.1 $nbody      seed=$seed zerocm=$zerocm
scripts/csh/mkmh97.sh:	mkplummer -      $nbody      seed=$seed zerocm=$zerocm | snapscale - $run.2 mscale="$m" rscale="$m**0.5" vscale="$m**0.25"
scripts/csh/mkmh97.sh:	mkplummer -      "$nbody*$m" seed=$seed zerocm=$zerocm | snapscale - $run.2 mscale="$m" rscale="$m**0.5" vscale="$m**0.25"
scripts/csh/mkmh97.sh:	    # @todo   zerocm?
scripts/csh/mkmh97.sh:	    snapstack $run.1 $run.2 $run.3 deltar=$r0,$rp,0 deltav=-$v0,0,0  zerocm=$zerocm
scripts/csh/mkmh97.sh:	    # snapscale $run.1 - mscale=0  | snapstack - $run.2 $run.3 deltar=$r0,$rp,0 deltav=-$v0,0,0  zerocm=t
scripts/csh/mkmh97.sh:	snapstack $run.1 $run.2 $run.3 deltar=$r0,$rp,0  deltav=0,$v0,0   zerocm=$zerocm
scripts/csh/mkmh97.sh:	    bonsai2 -i $run.3t --snapname $run. --logfile $run.gpu --dev 0 --snapiter $step -T $tstop -t $(nemoinp 1/2**$kmax)  1>$run.4.log 2>$run.4.err
inc/cfitsio/longnam.h:#define fits_read_grppar_usht  ffggpui
inc/cfitsio/longnam.h:#define fits_read_grppar_ulng  ffggpuj
inc/cfitsio/longnam.h:#define fits_read_grppar_ulnglng  ffggpujj
inc/cfitsio/longnam.h:#define fits_read_grppar_uint  ffggpuk
inc/cfitsio/longnam.h:#define fits_write_grppar_usht ffpgpui
inc/cfitsio/longnam.h:#define fits_write_grppar_ulng ffpgpuj
inc/cfitsio/longnam.h:#define fits_write_grppar_ulnglng ffpgpujj
inc/cfitsio/longnam.h:#define fits_write_grppar_uint ffpgpuk
inc/cfitsio/fitsio.h:int CFITS_API ffggpui(fitsfile *fptr, long group, long firstelem, long nelem,
inc/cfitsio/fitsio.h:int CFITS_API ffggpuj(fitsfile *fptr, long group, long firstelem, long nelem,
inc/cfitsio/fitsio.h:int CFITS_API ffggpujj(fitsfile *fptr, long group, long firstelem, long nelem,
inc/cfitsio/fitsio.h:int CFITS_API ffggpuk(fitsfile *fptr, long group, long firstelem, long nelem,
inc/cfitsio/fitsio.h:int CFITS_API ffpgpui(fitsfile *fptr, long group, long firstelem,
inc/cfitsio/fitsio.h:int CFITS_API ffpgpuj(fitsfile *fptr, long group, long firstelem,
inc/cfitsio/fitsio.h:int CFITS_API ffpgpuk(fitsfile *fptr, long group, long firstelem,
inc/cfitsio/fitsio.h:int CFITS_API ffpgpujj(fitsfile *fptr, long group, long firstelem,
inc/snipshot.h:void zerocm4snip(/* snipptr */);
usr/vogl/hershey/data/hersh.or: 3785 81D_KGLFJFJU RKGKU RPGQFOFOU RPGPU RGISIRHQI RKMOM RKQOQ RGUSURTQU
usr/bonsai/README.md:Bonsai needs a GPU, see Makefile for example how to compile.
usr/bonsai/README.md:Examples taken from a 2022 desktop: Xeon(R) CPU E5-2687W 0 @ 3.10GHz;  Nvidia RTX 3070 w/ 8GB memory
usr/wolfire/Testfile:	time runchemh2 run151 IBRLO=0 FGPUMP=1
usr/wolfire/runchemh2.c:  "FGPUMP=\n             5th line...",
usr/wolfire/runchemh2.c:  /* FGPUMP,IBRLO,ISO,ITURB,ITHP */
usr/wolfire/runchemh2.c:  l5[0] = patch_f("FGPUMP",  line,0,0, "%g");    
usr/aarseth/Makefile:URL2 = https://github.com/nbodyx/Nbody6ppGPU
usr/aarseth/Makefile:nbody6++: Nbody6ppGPU
usr/aarseth/Makefile:Nbody6ppGPU:
usr/aarseth/Makefile:	(cd nbody6/GPU2 ;  make clean; make sse ; cp run/nbody7.sse $(NEMOBIN))
usr/aarseth/Makefile:install_nbody6++.avx:  Nbody6ppGPU
usr/aarseth/Makefile:	(cd Nbody6ppGPU ; make clean ;\
usr/aarseth/Makefile:	./configure --disable-gpu --disable-mpi ;\
usr/aarseth/Makefile:install_nbody6++:  Nbody6ppGPU
usr/aarseth/Makefile:	(cd Nbody6ppGPU ; make clean ;\
usr/aarseth/Makefile:	./configure --disable-gpu --disable-mpi --enable-simd=sse ;\
usr/gravidy/rungravidy.c:    "gpu=0\n          GPUs to use, by default is the maximum available devices (use even numbers) (-g --gpu)",
usr/gravidy/rungravidy.c:    int ngpu = getiparam("gpu");
usr/gravidy/rungravidy.c:    if (ngpu > 0) warning("gpu=%d not used yet", ngpu);
usr/gravidy/Makefile:NEMO_NVCC   = /test/usr/local/cuda/bin/nvcc
usr/gravidy/Makefile:## install_gpu:   example gravidy-gpu (uses NEMO_BFLAGS and NEMO_NVCC)
usr/gravidy/Makefile:install_gpu: gravidy
usr/gravidy/Makefile:	make clean gpu NVCC=$(NEMO_NVCC) BOOSTFLAGS = "$(NEMO_BFLAGS)"; \
usr/dehnen/falcON/src/public/exe/mkking.cc:// v 1.0.2 19/09/2001  WD bug fixed; centrate centre of mass (option zerocm)   |
usr/dehnen/falcON/src/public/exe/mkking.cc:// v 1.4.1 30/04/2004  WD abandoned zerocm; new body.h; happy icc 8.0          |
usr/dehnen/falcON/src/public/exe/snapstac.cc:  "zerocm=f\n         zero center of mass (after all shifting etc.)      ",
usr/dehnen/falcON/src/public/exe/snapstac.cc:  if(getbparam("zerocm")) {
man/man3/index.3:zerocms  	libNbody.a
man/man5/bench.5:gyrfalcON(1NEMO), data(5NEMO), tabgen(1NEMO), mkspiral(1NEMO), mkplummer(1NEMO), hackcode1(1NEMO), nbody1(1NEMO), scfm(1NEMO), CGS(1NEMO), triple(1NEMO), accudate(lNEMO), bsf(1NEMO), nemobench(8NEMO)
man/doc/mkommod.doc:%A zerocm
man/doc/mksphere.doc:%A zerocm
man/doc/snapstack.doc:%A zerocm
man/doc/mkhomsph.doc:%A zerocm
man/doc/snapadd.doc:%A zerocm
man/doc/mkcube.doc:%A zerocm
man/doc/mkconfig.doc:%A zerocm
man/doc/mkplummer.doc:%A zerocm
man/doc/mkop.doc:%A zerocm
man/doc/mkop73.doc:%A zerocm
man/manl/accudate.l:\fBaccudate\fP \- 
man/manl/accudate.l:src/tools/misc/accudate.c
man/man1/gravidy.1:gravidy, rungravidy \- direct summation CPU/MPI/GPU hermite N-body integrator with variable block timesteps
man/man1/gravidy.1:an MPI, and a GPU (using CUDA) version. Depending on your needs, a single core version using GNU parallel can be a faster
man/man1/gravidy.1:\fBgpu=\fP
man/man1/gravidy.1:GPUs to use, by default is the maximum available devices (use even numbers) (-g --gpu) [0]
man/man1/gravidy.1:  -g [ --gpu ] <value> (=0)       GPUs to use, by default is the maximum available
man/man1/gravidy.1:this will install the vanilla "cpu" version.   The "gpu" and "mpi" version will need more guidance.
man/man1/gravidy.1:Maureira-Fredes, C and Amaro-Seoane, P. "GraviDy, a GPU modular, parallel direct-summation N-body integrator: Dynamics with softening",
man/man1/bsf.1:because mkplummer by default value centers the snapshot (\fBzerocm=t\fP).
man/man1/snapcenter.1:   % mkplummer - nbody=100 zerocm=f seed=123 | snapcenter - . report=t
man/man1/snapcenter.1:   mkplummer - nbody=100 zerocm=f seed=123 | snapcenterp - . report=t
man/man1/snapcenter.1:    % mkplummer - nbody=100 zerocm=f nmodel=100 seed=123 | snapcenter - . report=t | tabstat  - 1:6
man/man1/mkkd95.1:\fBzerocm=\fP
man/man1/snapstack.1:\fBzerocm=\fP\fIzero_cm_flag\fP
man/man1/snapstack.1:5-aug-06	V2.0 added shift= and made zerocm=f default	PJT
man/man1/runchemh2.1:\fBFGPUMP=\fP
man/man1/uns_stack.1:\fBzerocm=\fP\fIzero_cm_flag\fP
man/man1/mkconfig.1:\fBzerocm=\fP\fIzero_cm_flag\fP
man/man1/glnemo2.1:rendering. Millions of particles can be rendered, in real time, with a fast GPU. With this
man/man1/glnemo2.1:Glnemo2 uses hardware accelerated feature of video card and especially GLSL capabilities. Not all the video cards have this requirement under Linux. The very well supported card are Nvidia card with proprietary driver. If glnemo2 crashs at starting, try to launch the program again by adding to the command line "glsl=f". It will deactivate the hardware accelerated engine, therefore the rendering will be slow but the program might work at least.
man/man1/mksphere.1:\fBzerocm=\fP
man/man1/mknsh96.1:\fBzerocm=\fP
man/man1/mkhernquist.1:\fBzerocm=\fP
man/man1/mkhernquist.1:setting zerocm=t will cause the center to be significantly 
man/man1/snapadd.1:\fBzerocm=t|f\fP
man/man1/mkgrid.1:\fBzerocm=t|f\fP
man/man1/mcluster.1:GPU usage; 0= no GPU, 1= use GPU                   
man/man1/mkommod.1:\fBzerocm=\fP\fIzero_flag\fP
man/man1/mkexpdisk.1:\fBzerocm=\fP\fIt|f\fP
man/man1/mkcube.1:As with most NEMO programs, the center of mass is (roughly, see \fBzerocm=\fP)
man/man1/mkcube.1:\fBzerocm=t|f\fP
man/man1/mkhomsph.1:\fBzerocm=\fBt|f\fP
man/man1/mkhomsph.1:6-apr-89	V1.3 keyword zerocm added	PJT
man/man1/mkop73.1:\fBzerocm=t|f\fP
man/man1/mk2body.1:\fBzerocm=\fP
man/man1/mkvh60.1:\fBzerocm=\fP
man/man1/mkplummer.1:\fBzerocm=t|f\fP
man/man1/mkplummer.1: mkplummer - 10 quiet=0 zerocm=f | snapsort - - r | snapprint - r | tabplot - 0 1 line=1,1 point=2,0.1
man/man1/mkplummer.1: mkplummer - 10 quiet=1 zerocm=f | snapsort - - r | snapprint - r | tabplot - 0 1 line=1,1 point=2,0.1
man/man1/mkplummer.1: mkplummer - 10 quiet=2 zerocm=f | snapsort - - r | snapprint - r | tabplot - 0 1 line=1,1 point=2,0.1
man/man1/mkplummer.1:Important to note that with \fBzerocm=f\fP the center of mass will drift, while the "mathematical center"
man/man1/mkplummer.1: gsprealize plum.gsp plum.snap zerocm=true
man/man1/mkplummer.1:xx-xxx-88	V1.2: zerocm keyword added	PJT
man/man1/hackdens.1:   mkplummer - 100 zerocm=f | snapsort - - rank=r > p.dat
man/man1/mkisosph.1:\fBzerocm=\fBt|f\fP
man/man1/mkisosph.1:6-apr-89	V1.1 zerocm keyword added	PJT
man/man1/runbody6.1:nbody6++:  https://github.com/nbodyx/Nbody6ppGPU
src/tools/misc/Makefile:#UTILITIES = crc hd changed redir age  accudate
src/tools/misc/Makefile:UTILITIES = hd redir accudate
src/tools/misc/Makefile:accudate:	accudate.c
src/tools/misc/Makefile:	$(CC) $(CFLAGS) -o accudate accudate.c
src/tools/misc/accudate.c:Program:   accudate.c
src/tools/misc/accudate.c:    gcc -Wall -ansi -pedantic -o accudate accudate.c
src/tools/misc/accudate.c:    IIi machine a series of accudate commands emit times .01
src/tools/misc/accudate.c:    time should drop below 1 millisecond, at which point accudate
src/tools/misc/accudate.c:(void) fprintf(stderr,"accudate command summary:\n\n");
src/nbody/init/mk2body.c:    "zerocm=true\n	if true, zero the center of mass",
src/nbody/init/mk2body.c:  if (getbparam("zerocm"))
src/nbody/init/mk2body.c:    zerocms(phase[0][0], 2 * NDIM, m, nobj, nobj);
src/nbody/init/mkop73.c: *	 3-nov-93	added zerocm=; implemented options=acc,phi also
src/nbody/init/mkop73.c:    "zerocm=t\n		  Re-center at center of mass?",
src/nbody/init/mkop73.c:local bool zerocm;
src/nbody/init/mkop73.c:    zerocm = getbparam("zerocm");
src/nbody/init/mkop73.c:    if (zerocm)
src/nbody/init/mksphere.c:    "zerocm=t\n               Centrate snapshot (t/f)?",
src/nbody/init/mksphere.c:    bool    zerocm;
src/nbody/init/mksphere.c:    zerocm = getbparam("zerocm");
src/nbody/init/mksphere.c:      btab = mksphere(nbody, seed, zerocm);
src/nbody/init/mksphere.c:      btab = mkplummer(nbody, seed, zerocm);
src/nbody/init/mksphere.c:local Body *mkplummer(int nbody, int seed, bool zerocm)
src/nbody/init/mksphere.c:    if (zerocm) {       /* False for Masspectrum */
src/nbody/init/mksphere.c:local Body *mksphere(int nbody, int seed, bool zerocm)
src/nbody/init/mksphere.c:    if (zerocm) {       /* False for Masspectrum */
src/nbody/init/mkkd95.c:  "zerocm=t\n        in.- Center the snapshot?",
src/nbody/init/mkkd95.c:    int seed, zerocm, nbulge, ndisk, nhalo;
src/nbody/init/mkkd95.c:    zerocm = getbparam("zerocm") ? 1 : 0;
src/nbody/init/mkkd95.c:	      zerocm);
src/nbody/init/mkkd95.c:	      zerocm);
src/nbody/init/mkkd95.c:	      zerocm);
src/nbody/init/mkommod.c:    "zerocm=true\n		  if true, zero center of mass",
src/nbody/init/mkommod.c:    if (getbparam("zerocm"))			/* zero center of mass?     */
src/nbody/init/mkgalaxy2.c:  "zerocm=t\n             Centrate snapshot (t/f)?",
src/nbody/init/mkhernquist.c:    "zerocm=t\n               Centrate snapshot (t/f)?",
src/nbody/init/mkhernquist.c:        bool Qcenter = getbparam("zerocm");
src/nbody/init/mkexpdisk.c:    "zerocm=t\n           center the snapshot?",
src/nbody/init/mkexpdisk.c:    if (getbparam("zerocm"))
src/nbody/init/mkgrid.c:    "zerocm=f\n     Center c.o.m. ?",
src/nbody/init/mkgrid.c:local bool zerocm;
src/nbody/init/mkgrid.c:    zerocm = getbparam("zerocm");
src/nbody/init/mkgrid.c:    if (zerocm)
src/nbody/init/mkconfig.c:    "zerocm=false\n	if true, zero center of mass",
src/nbody/init/mkconfig.c:extern void  zerocms(double *, int, double *, int, int);
src/nbody/init/mkconfig.c:    if (total_mass > 0 && getbparam("zerocm"))
src/nbody/init/mkconfig.c:	zerocms(phase[0][0], 2 * NDIM, mass, nobj, nobj);
src/nbody/init/mkvh60.c:    "zerocm=t\n     Center c.o.m. ?",
src/nbody/init/mkvh60.c:local bool zerocm;
src/nbody/init/mkvh60.c:    zerocm = getbparam("zerocm");
src/nbody/init/mkvh60.c:    if (zerocm)
src/nbody/init/mkplummer.c:    "zerocm=t\n               Centrate snapshot (t/f)?",
src/nbody/init/mkplummer.c:    bool    zerocm;
src/nbody/init/mkplummer.c:    zerocm = getbparam("zerocm");
src/nbody/init/mkplummer.c:      btab[i] = mkplummer(nbody, mlow, mfrac, rfrac, seed, snap_time, zerocm, scale,
src/nbody/init/mkplummer.c: *                          zerocm: logical determining if to center snapshot
src/nbody/init/mkplummer.c:Body *mkplummer(nbody, mlow, mfrac, rfrac, seed, snap_time,zerocm,scale,quiet,mr,mf)
src/nbody/init/mkplummer.c:bool  zerocm;
src/nbody/init/mkplummer.c:    if (zerocm) {       /* False for Masspectrum */
src/nbody/init/mknsh96.c:    "zerocm=t\n           center the snapshot?",
src/nbody/init/mknsh96.c:  if (getbparam("zerocm"))
src/nbody/init/mkcube.c:    "zerocm=t\n     Center c.o.m. ?",
src/nbody/init/mkcube.c:local bool zerocm;
src/nbody/init/mkcube.c:    zerocm = getbparam("zerocm");
src/nbody/init/mkcube.c:    if (zerocm)
src/nbody/init/mkhomsph.c:    "zerocm=t\n     Center c.o.m. ?",
src/nbody/init/mkhomsph.c:local bool zerocm;
src/nbody/init/mkhomsph.c:    zerocm = getbparam("zerocm");
src/nbody/init/mkhomsph.c:    if (zerocm)
src/nbody/init/mkpolytrope.c:extern void  zerocms(double *, int, double *, int, int);
src/nbody/init/mkpolytrope.c:    zerocms(phase[0], 6, mass, nobj, nobj);
src/nbody/glnemo/README.bench:  It's possible to benchmark glnemo to test your CPU and especially your GPU (Graphic Processor Unit) aka Video Card.
src/nbody/glnemo/ChangeLog:	card with an accelerated driver (NVIDIA recommended) and not too
src/nbody/cores/zerocms.c: * ZEROCMS.C: routines to find and zero the center of mass.
src/nbody/cores/zerocms.c:void zerocms(real *space, int ndim, real *mass, int npnt, int nzer)
src/nbody/cores/zerocms.c:        error("zerocms: too many dimensions; ndim=%d MDIM=%d, ndim, MDIM");
src/nbody/cores/Makefile:OBJFILES = pickpnt.o units.o zerocms.o bodytrans.o
src/nbody/cores/Makefile:LOBJFILES = $L(pickpnt.o) $L(units.o) $L(zerocms.o) $L(bodytrans.o)
src/nbody/trans/snapadd.c: *	13-may-92  2.2 Allow concurrent adding; default zerocm=false now
src/nbody/trans/snapadd.c:    "zerocm=false\n	zero new center of mass",
src/nbody/trans/snapadd.c:bool Qmass, Qzerocm, Qsync, needphi, needacc, needkey;
src/nbody/trans/snapadd.c:extern void  zerocms(double *, int, double *, int, int);
src/nbody/trans/snapadd.c:    Qzerocm = getbparam("zerocm");
src/nbody/trans/snapadd.c:    if (Qzerocm) {
src/nbody/trans/snapadd.c:	zerocms(phasetot, 2*NDIM, masstot, nbodytot, nbodytot);
src/nbody/trans/snapstack.c: *       5-aug-06  2.0  change default of zerocm=, added shift= (after strong urge from WD) PJT
src/nbody/trans/snapstack.c:    "zerocm=false\n		  zero the center of mass after stacking?",
src/nbody/trans/snapstack.c:    if (getbparam("zerocm")) snapcenter();
src/nbody/trans/snapstack.c:    else warning("zerocm=false is now the default!!");

```
