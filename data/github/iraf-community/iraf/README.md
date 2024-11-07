# https://github.com/iraf-community/iraf

```console
pkg/ecl/modes.c:		c_stgputline ((XINT)STDOUT, buf);
pkg/cl/modes.c:		c_stgputline ((XINT)STDOUT, buf);
unix/f2c/src/defines.h:#define PROCMAIN 1
unix/hlib/libc/iraf_libc.h:extern int	c_stgputline (XINT fd, char *buf);
unix/hlib/libc/iraf_xnames.h:#define	STG_PUTLINE	stgpue_
unix/os/zfioks.c:	if (sigprocmask (0, NULL, &sigset) < 0)
unix/os/zfioks.c:	    dbgmsg ("sigprocmask error");
unix/os/zfiopr.c:	sigprocmask (SIG_BLOCK, &set, &sigmask_save);
unix/os/zfiopr.c:	sigprocmask (SIG_SETMASK, &sigmask_save, NULL);
unix/os/zfiopr.c:	sigprocmask (SIG_BLOCK, &set, &sigmask_save);
unix/os/zfiopr.c:	sigprocmask (SIG_SETMASK, &sigmask_save, NULL);
unix/boot/spp/rpp/ratlibr/patsiz.r:   else if (pat (n) == CCL | pat (n) == NCCL)
unix/boot/spp/rpp/ratlibr/rdefs:define (NCCL,110)
unix/boot/spp/rpp/ratlibr/getccl.r:      junk = addset (NCCL, pat, j, MAXPAT)
unix/boot/spp/rpp/ratlibr/omatch.r:   else if (pat (j) == NCCL) {
sys/tty/ttyputl.x:	call ttygputline (fd, tty, text, map_cc)
sys/tty/ttyputl.x:# TTYGPUTLINE -- This is the original ttypuline.  The code is not very
sys/tty/ttyputl.x:procedure ttygputline (fd, tty, text, map_cc)
sys/fmtio/patmatch.x:define	NCCL		-10		# [^...
sys/fmtio/patmatch.x:	case CCL, NCCL:
sys/fmtio/patmatch.x:	case NCCL:					# not in char class
sys/fmtio/patmatch.x:	    cval = NCCL
sys/libc/libc_proto.h:extern int c_stgputline(int fd, char *buf);
sys/libc/stgio.c:**		     c_stgputline (STDOUT, buf)
sys/libc/stgio.c:/* C_STGPUTLINE -- Put a line of text to the graphics terminal.
sys/libc/stgio.c:c_stgputline (
sys/NAMES:c_stgputline    c_stgputline
sys/NAMES:pgpusd          pg_pushcmd
sys/NAMES:stgpue          stg_putline
sys/NAMES:stgpuy          stg_putcellarray
sys/NAMES:ttygpe          ttygputline
sys/INDEX:c_stgputline        1 ./libc/stgio.c     c_stgputline (fd, buf)
sys/INDEX:ttygputline       134 ./tty/ttyputl.x    procedure ttygputline (fd, tty, text, map_cc)

```
