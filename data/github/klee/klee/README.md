# https://github.com/klee/klee

```console
runtime/POSIX/FreeBSD.h:#define	__NR_sigprocmask	SYS_sigprocmask
runtime/POSIX/stubs.c:int sigprocmask(int how, const sigset_t *set, sigset_t *oldset)
runtime/POSIX/stubs.c:int sigprocmask(int how, const sigset_t *set, sigset_t *oldset) {

```
