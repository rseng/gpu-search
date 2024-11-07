# https://github.com/astroumd/miriad

```console
TODO:  - fftw (CUDA)
borrow/readline/readline.c:	sigprocmask (SIG_SETMASK, &set, (sigset_t *)NULL);
borrow/readline/readline.c:  sigprocmask (SIG_BLOCK, &set, &oset);
borrow/readline/readline.c:  sigprocmask (SIG_SETMASK, &oset, (sigset_t *)NULL);
borrow/readline/readline.c:  sigprocmask (SIG_BLOCK, &set, &oset);
borrow/readline/readline.c:  sigprocmask (SIG_SETMASK, &oset, (sigset_t *)NULL);
src/prog/map/invert.for:          call ProcMFS (tno,uvw,Wt,rms2,Wtf,rms2f,systemp(2),data,flags,
src/prog/map/invert.for:      subroutine ProcMFS(tno,uvw,Wt,rms2,Wtf,rms2f,systempf,data,flags,
src/prog/map/invert.for:      if(ncorr.ne.npol)call bug('f','Something is screwy in ProcMFS')
src/prog/map/invert.for:     *                'Buffer overflow, in ProcMFS')
src/prog/deconv/clean.for:c    zerocmp    True if the iterating was stopped by a zero component.
src/prog/deconv/clean.for:      logical more,ZeroCmp
src/prog/deconv/clean.for:      ZeroCmp = Wts.eq.0
src/prog/deconv/clean.for:     *        .not.(negStop .and. negFound) .and. .not.ZeroCmp
src/prog/deconv/clean.for:        zeroCmp = Wts.eq.0
src/prog/deconv/clean.for:     *        .not.(negStop .and. negFound) .and. .not.zeroCmp
src/prog/wsrt/invertf.for:	subroutine ProcMFS(tno,uvw,Wt,rms2,data,flags,
src/prog/wsrt/invertf.for:!     *			'Buffer overflow, in ProcMFS')
src/prog/wsrt/invertf.for:	  call ProcMFS (tno,uvw,Wt,rms2,data,flags,

```
