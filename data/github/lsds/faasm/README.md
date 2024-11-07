# https://github.com/faasm/faasm

```console
src/wavm/syscalls.cpp:            return s__rt_sigprocmask(a, b, c, d);
src/wavm/syscalls.h:int32_t s__rt_sigprocmask(int32_t how,
src/wavm/signals.cpp:I32 s__rt_sigprocmask(I32 how, I32 sigSetPtr, I32 oldSetPtr, I32 sigsetsize)
src/wavm/signals.cpp:    SPDLOG_DEBUG("S - rt_sigprocmask - {} {} {} {}",

```
