# https://github.com/linksmart/thing-directory

```console
vendor/github.com/eclipse/paho.mqtt.golang/client.go:	incomingPubChan chan *packets.PublishPacket
vendor/github.com/eclipse/paho.mqtt.golang/client.go:		c.incomingPubChan = make(chan *packets.PublishPacket, c.options.MessageChannelDepth)
vendor/github.com/eclipse/paho.mqtt.golang/client.go:		c.msgRouter.matchAndDispatch(c.incomingPubChan, c.options.Order, c)
vendor/github.com/eclipse/paho.mqtt.golang/net.go:					c.incomingPubChan <- m
vendor/github.com/eclipse/paho.mqtt.golang/net.go:					DEBUG.Println(NET, "done putting msg on incomingPubChan")
vendor/github.com/eclipse/paho.mqtt.golang/net.go:					c.incomingPubChan <- m
vendor/github.com/eclipse/paho.mqtt.golang/net.go:					DEBUG.Println(NET, "done putting msg on incomingPubChan")
vendor/github.com/eclipse/paho.mqtt.golang/net.go:					case c.incomingPubChan <- m:
vendor/github.com/eclipse/paho.mqtt.golang/net.go:					DEBUG.Println(NET, "done putting msg on incomingPubChan")
vendor/github.com/syndtr/goleveldb/leveldb/storage/file_storage_windows.go:	procMoveFileExW = modkernel32.NewProc("MoveFileExW")
vendor/github.com/syndtr/goleveldb/leveldb/storage/file_storage_windows.go:	r1, _, e1 := syscall.Syscall(procMoveFileExW.Addr(), 3, uintptr(unsafe.Pointer(from)), uintptr(unsafe.Pointer(to)), uintptr(flags))
vendor/golang.org/x/sys/windows/zerrors_windows.go:	ERROR_GRAPHICS_GPU_EXCEPTION_ON_DEVICE                                    Handle        = 0xC0262200
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMoveFileW                                            = modkernel32.NewProc("MoveFileW")
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMoveFileExW                                          = modkernel32.NewProc("MoveFileExW")
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMapViewOfFile                                        = modkernel32.NewProc("MapViewOfFile")
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMessageBoxW                                          = moduser32.NewProc("MessageBoxW")
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMultiByteToWideChar                                  = modkernel32.NewProc("MultiByteToWideChar")
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMakeAbsoluteSD                                       = modadvapi32.NewProc("MakeAbsoluteSD")
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	procMakeSelfRelativeSD                                   = modadvapi32.NewProc("MakeSelfRelativeSD")
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r1, _, e1 := syscall.Syscall(procMoveFileW.Addr(), 2, uintptr(unsafe.Pointer(from)), uintptr(unsafe.Pointer(to)), 0)
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r1, _, e1 := syscall.Syscall(procMoveFileExW.Addr(), 3, uintptr(unsafe.Pointer(from)), uintptr(unsafe.Pointer(to)), uintptr(flags))
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r0, _, e1 := syscall.Syscall6(procMapViewOfFile.Addr(), 5, uintptr(handle), uintptr(access), uintptr(offsetHigh), uintptr(offsetLow), uintptr(length), 0)
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r0, _, e1 := syscall.Syscall6(procMessageBoxW.Addr(), 4, uintptr(hwnd), uintptr(unsafe.Pointer(text)), uintptr(unsafe.Pointer(caption)), uintptr(boxtype), 0, 0)
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r0, _, e1 := syscall.Syscall6(procMultiByteToWideChar.Addr(), 6, uintptr(codePage), uintptr(dwFlags), uintptr(unsafe.Pointer(str)), uintptr(nstr), uintptr(unsafe.Pointer(wchar)), uintptr(nwchar))
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r1, _, e1 := syscall.Syscall12(procMakeAbsoluteSD.Addr(), 11, uintptr(unsafe.Pointer(selfRelativeSD)), uintptr(unsafe.Pointer(absoluteSD)), uintptr(unsafe.Pointer(absoluteSDSize)), uintptr(unsafe.Pointer(dacl)), uintptr(unsafe.Pointer(daclSize)), uintptr(unsafe.Pointer(sacl)), uintptr(unsafe.Pointer(saclSize)), uintptr(unsafe.Pointer(owner)), uintptr(unsafe.Pointer(ownerSize)), uintptr(unsafe.Pointer(group)), uintptr(unsafe.Pointer(groupSize)), 0)
vendor/golang.org/x/sys/windows/zsyscall_windows.go:	r1, _, e1 := syscall.Syscall(procMakeSelfRelativeSD.Addr(), 3, uintptr(unsafe.Pointer(absoluteSD)), uintptr(unsafe.Pointer(selfRelativeSD)), uintptr(unsafe.Pointer(selfRelativeSDSize)))
vendor/golang.org/x/sys/unix/zsysnum_freebsd_386.go:	SYS_SIGPROCMASK              = 340 // { int sigprocmask(int how, const sigset_t *set, sigset_t *oset); }
vendor/golang.org/x/sys/unix/zsysnum_darwin_arm.go:	SYS_SIGPROCMASK                    = 48
vendor/golang.org/x/sys/unix/zsysnum_freebsd_arm64.go:	SYS_SIGPROCMASK              = 340 // { int sigprocmask(int how, const sigset_t *set, sigset_t *oset); }
vendor/golang.org/x/sys/unix/zsysnum_linux_ppc64le.go:	SYS_SIGPROCMASK            = 126
vendor/golang.org/x/sys/unix/zsysnum_linux_ppc64le.go:	SYS_RT_SIGPROCMASK         = 174
vendor/golang.org/x/sys/unix/zsysnum_linux_mips64.go:	SYS_RT_SIGPROCMASK         = 5014
vendor/golang.org/x/sys/unix/syscall_darwin.go:// Sigprocmask
vendor/golang.org/x/sys/unix/zsysnum_linux_mips.go:	SYS_SIGPROCMASK                  = 4126
vendor/golang.org/x/sys/unix/zsysnum_linux_mips.go:	SYS_RT_SIGPROCMASK               = 4195
vendor/golang.org/x/sys/unix/zsysnum_openbsd_amd64.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
vendor/golang.org/x/sys/unix/zsysnum_linux_riscv64.go:	SYS_RT_SIGPROCMASK         = 135
vendor/golang.org/x/sys/unix/zsysnum_openbsd_arm.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
vendor/golang.org/x/sys/unix/zsysnum_linux_mipsle.go:	SYS_SIGPROCMASK                  = 4126
vendor/golang.org/x/sys/unix/zsysnum_linux_mipsle.go:	SYS_RT_SIGPROCMASK               = 4195
vendor/golang.org/x/sys/unix/zsysnum_freebsd_amd64.go:	SYS_SIGPROCMASK              = 340 // { int sigprocmask(int how, const sigset_t *set, sigset_t *oset); }
vendor/golang.org/x/sys/unix/zsysnum_openbsd_386.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
vendor/golang.org/x/sys/unix/zsysnum_darwin_386.go:	SYS_SIGPROCMASK                    = 48
vendor/golang.org/x/sys/unix/zsysnum_darwin_amd64.go:	SYS_SIGPROCMASK                    = 48
vendor/golang.org/x/sys/unix/zsysnum_linux_amd64.go:	SYS_RT_SIGPROCMASK         = 14
vendor/golang.org/x/sys/unix/zsysnum_linux_386.go:	SYS_SIGPROCMASK                  = 126
vendor/golang.org/x/sys/unix/zsysnum_linux_386.go:	SYS_RT_SIGPROCMASK               = 175
vendor/golang.org/x/sys/unix/zsysnum_freebsd_arm.go:	SYS_SIGPROCMASK              = 340 // { int sigprocmask(int how, const sigset_t *set, sigset_t *oset); }
vendor/golang.org/x/sys/unix/syscall_openbsd.go:// sigprocmask
vendor/golang.org/x/sys/unix/zsysnum_linux_ppc64.go:	SYS_SIGPROCMASK            = 126
vendor/golang.org/x/sys/unix/zsysnum_linux_ppc64.go:	SYS_RT_SIGPROCMASK         = 174
vendor/golang.org/x/sys/unix/zsysnum_linux_arm64.go:	SYS_RT_SIGPROCMASK         = 135
vendor/golang.org/x/sys/unix/zsysnum_linux_mips64le.go:	SYS_RT_SIGPROCMASK         = 5014
vendor/golang.org/x/sys/unix/zsysnum_openbsd_arm64.go:	SYS_SIGPROCMASK    = 48  // { int sys_sigprocmask(int how, sigset_t mask); }
vendor/golang.org/x/sys/unix/syscall_linux.go:// RtSigprocmask
vendor/golang.org/x/sys/unix/zsysnum_darwin_arm64.go:	SYS_SIGPROCMASK                    = 48
vendor/golang.org/x/sys/unix/syscall_dragonfly.go:// Sigprocmask
vendor/golang.org/x/sys/unix/zsysnum_linux_sparc64.go:	SYS_RT_SIGPROCMASK         = 103
vendor/golang.org/x/sys/unix/zsysnum_linux_sparc64.go:	SYS_SIGPROCMASK            = 220
vendor/golang.org/x/sys/unix/syscall_freebsd.go:// Sigprocmask
vendor/golang.org/x/sys/unix/syscall_netbsd.go:// __sigprocmask14
vendor/golang.org/x/sys/unix/syscall_netbsd.go:// compat_13_sigprocmask13
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMadvise libc_madvise
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMkdir libc_mkdir
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMkdirat libc_mkdirat
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMkfifo libc_mkfifo
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMkfifoat libc_mkfifoat
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMknod libc_mknod
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMknodat libc_mknodat
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMlock libc_mlock
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMlockall libc_mlockall
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMprotect libc_mprotect
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMsync libc_msync
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMunlock libc_munlock
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procMunlockall libc_munlockall
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procmmap libc_mmap
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go://go:linkname procmunmap libc_munmap
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMadvise,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMkdir,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMkdirat,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMkfifo,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMkfifoat,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMknod,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMknodat,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMlock,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMlockall,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMprotect,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMsync,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMunlock,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procMunlockall,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procmmap,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	procmunmap,
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMadvise)), 3, uintptr(unsafe.Pointer(_p0)), uintptr(len(b)), uintptr(advice), 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMkdir)), 2, uintptr(unsafe.Pointer(_p0)), uintptr(mode), 0, 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMkdirat)), 3, uintptr(dirfd), uintptr(unsafe.Pointer(_p0)), uintptr(mode), 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMkfifo)), 2, uintptr(unsafe.Pointer(_p0)), uintptr(mode), 0, 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMkfifoat)), 3, uintptr(dirfd), uintptr(unsafe.Pointer(_p0)), uintptr(mode), 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMknod)), 3, uintptr(unsafe.Pointer(_p0)), uintptr(mode), uintptr(dev), 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMknodat)), 4, uintptr(dirfd), uintptr(unsafe.Pointer(_p0)), uintptr(mode), uintptr(dev), 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMlock)), 2, uintptr(unsafe.Pointer(_p0)), uintptr(len(b)), 0, 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMlockall)), 1, uintptr(flags), 0, 0, 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMprotect)), 3, uintptr(unsafe.Pointer(_p0)), uintptr(len(b)), uintptr(prot), 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMsync)), 3, uintptr(unsafe.Pointer(_p0)), uintptr(len(b)), uintptr(flags), 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMunlock)), 2, uintptr(unsafe.Pointer(_p0)), uintptr(len(b)), 0, 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procMunlockall)), 0, 0, 0, 0, 0, 0, 0)
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	r0, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procmmap)), 6, uintptr(addr), uintptr(length), uintptr(prot), uintptr(flag), uintptr(fd), uintptr(pos))
vendor/golang.org/x/sys/unix/zsyscall_solaris_amd64.go:	_, _, e1 := sysvicall6(uintptr(unsafe.Pointer(&procmunmap)), 2, uintptr(addr), uintptr(length), 0, 0, 0, 0)
vendor/golang.org/x/sys/unix/zsysnum_linux_s390x.go:	SYS_SIGPROCMASK            = 126
vendor/golang.org/x/sys/unix/zsysnum_linux_s390x.go:	SYS_RT_SIGPROCMASK         = 175
vendor/golang.org/x/sys/unix/zsysnum_dragonfly_amd64.go:	SYS_SIGPROCMASK            = 340 // { int sigprocmask(int how, const sigset_t *set, sigset_t *oset); }
vendor/golang.org/x/sys/unix/zsysnum_linux_arm.go:	SYS_SIGPROCMASK                  = 126
vendor/golang.org/x/sys/unix/zsysnum_linux_arm.go:	SYS_RT_SIGPROCMASK               = 175

```
