# https://github.com/ericmandel/funtools

```console
util/xlaunch.c:  if (sigprocmask(SIG_BLOCK, &chldmask, &savemask) < 0)
util/xlaunch.c:      sigprocmask(SIG_SETMASK, &savemask, NULL);
util/xlaunch.c:  if( sigprocmask(SIG_SETMASK, &savemask, NULL) < 0 ) return -1;

```
