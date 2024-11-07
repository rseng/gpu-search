# https://github.com/ncbi/ncbi-vdb

```console
libs/vxf/fzip.c:    VBlobHeaderArgPushTail(hdr, self->mantissa);
libs/vxf/zstd.c:        VBlobHeaderArgPushTail ( hdr, ( int64_t ) ( sbits & 7 ) );
libs/vxf/bzip.c:        VBlobHeaderArgPushTail ( hdr, ( int64_t ) ( sbits & 7 ) );
libs/vxf/irzip.c:		rc = VBlobHeaderArgPushTail(hdr, min[0]);
libs/vxf/irzip.c:			rc = VBlobHeaderArgPushTail(hdr, slope[0]);
libs/vxf/irzip.c:					rc = VBlobHeaderArgPushTail(hdr, min[1]);
libs/vxf/irzip.c:					if(rc == 0) rc = VBlobHeaderArgPushTail(hdr, slope[1]);
libs/vxf/zip.c:        VBlobHeaderArgPushTail ( hdr, ( int64_t ) ( sbits & 7 ) );
libs/align/zz_samextract-grammar.output:   53   | RGPU VALUE
libs/align/zz_samextract-grammar.output:    RGPU <strval> (305) 53
libs/align/zz_samextract-grammar.output:    RGPU   shift, and go to state 45
libs/align/zz_samextract-grammar.output:   53 rg: RGPU . VALUE
libs/align/zz_samextract-grammar.output:    RGPU   shift, and go to state 45
libs/align/zz_samextract-grammar.output:   53 rg: RGPU VALUE .
libs/align/samextract-grammar.y:%token <strval> RGPU
libs/align/samextract-grammar.y:   | RGPU VALUE {
libs/align/samextract-lex.l:<INRG>"\tPU:" { SAMlval.strval="PU"; return RGPU; }
libs/align/zz_samextract-grammar.h:    RGPU = 305,
libs/align/zz_samextract-grammar.c:    RGPU = 305,
libs/align/zz_samextract-grammar.c:  "RGPI", "RGPL", "RGPM", "RGPU", "RGSM", "PGID", "PGPN", "PGCL", "PGPP",
libs/align/zz_samextract-lex.c:{ SAMlval.strval="PU"; return RGPU; }
libs/kproc/procmgr.c:#include <kproc/procmgr.h>
libs/kproc/procmgr.c:#include "procmgr-whack.h" /* WHACK_PROC_MGR */
libs/kproc/procmgr.c: * KProcMgr
libs/kproc/procmgr.c:struct KProcMgr
libs/kproc/procmgr.c:LIB_EXPORT rc_t CC KProcMgrWhack ( void )
libs/kproc/procmgr.c:    KProcMgr * test, * self = s_proc_mgr . ptr;
libs/kproc/procmgr.c:    /* check to see if this thread will be cleaning up on procmgr */
libs/kproc/procmgr.c:LIB_EXPORT rc_t CC KProcMgrInit ( void )
libs/kproc/procmgr.c:        KProcMgr * mgr = calloc ( 1, sizeof * mgr );
libs/kproc/procmgr.c:            KProcMgr * rslt;
libs/kproc/procmgr.c:            KRefcountInit ( & mgr -> refcount, 0, "KProcMgr", "init", "process mgr" );
libs/kproc/procmgr.c:LIB_EXPORT rc_t CC KProcMgrMakeSingleton ( KProcMgr ** mgrp )
libs/kproc/procmgr.c:            rc = KProcMgrAddRef ( * mgrp );
libs/kproc/procmgr.c:LIB_EXPORT rc_t CC KProcMgrAddRef ( const KProcMgr *self )
libs/kproc/procmgr.c:        switch ( KRefcountAdd ( & self -> refcount, "KProcMgr" ) )
libs/kproc/procmgr.c:LIB_EXPORT rc_t CC KProcMgrRelease ( const KProcMgr *self )
libs/kproc/procmgr.c:                rc_t rc = KRefcountDrop ( & self -> refcount, "KProcMgr" );
libs/kproc/procmgr.c:                    KProcMgrWhack(); KLockRelease(cleanup_lock);
libs/kproc/procmgr.c:LIB_EXPORT rc_t CC KProcMgrAddCleanupTask ( KProcMgr *self, KTaskTicket *ticket, KTask *task )
libs/kproc/procmgr.c:LIB_EXPORT rc_t CC KProcMgrRemoveCleanupTask ( KProcMgr *self, const KTaskTicket *ticket )
libs/kproc/procmgr.c:LIB_EXPORT rc_t CC KProcMgrGetPID ( const KProcMgr * self, uint32_t * pid )
libs/kproc/procmgr.c:LIB_EXPORT rc_t CC KProcMgrGetHostName ( const KProcMgr * self, char * buffer, size_t buffer_size )
libs/kproc/win/sysmgr.c:#include <kproc/procmgr.h>
libs/kproc/win/sysmgr.c:LIB_EXPORT bool CC KProcMgrOnMainThread ( void )
libs/kproc/linux/sysmgr.c:#include <kproc/procmgr.h>
libs/kproc/linux/sysmgr.c:LIB_EXPORT bool CC KProcMgrOnMainThread ( void )
libs/kproc/bsd/sysmgr.c:#include <kproc/procmgr.h>
libs/kproc/bsd/sysmgr.c:LIB_EXPORT bool CC KProcMgrOnMainThread ( void )
libs/kproc/CMakeLists.txt:    procmgr
libs/cloud/gcp.c:#include <kproc/procmgr.h>
libs/cloud/gcp.c:    KProcMgr * procMgr;
libs/cloud/gcp.c:    rc_t rc = KProcMgrMakeSingleton(&procMgr);
libs/cloud/gcp.c:        rc = KProcMgrGetPID(procMgr, pid);
libs/cloud/gcp.c:        rc2 = KProcMgrRelease(procMgr);
libs/ext/mbedtls/sha512.c:    if( sigprocmask( 0, NULL, &old_mask ) )
libs/ext/mbedtls/sha512.c:    sigprocmask( SIG_SETMASK, &old_mask, NULL );
libs/ext/mbedtls/sha256.c:    if( sigprocmask( 0, NULL, &old_mask ) )
libs/ext/mbedtls/sha256.c:    sigprocmask( SIG_SETMASK, &old_mask, NULL );
libs/kfc/rsrc.c:#include <kproc/procmgr.h>
libs/kfc/rsrc.c:        rc = KProcMgrAddRef ( rsrc -> proc = src -> proc );
libs/kfc/rsrc.c:            rsrc -> thread = KProcMgrMakeThreadState ( rsrc -> proc );
libs/kfc/rsrc.c:        KProcMgrRelease ( self -> proc );
libs/kfc/rsrc.c:        rc = KProcMgrInit ();
libs/kfc/rsrc.c:            rc = KProcMgrMakeSingleton ( & rsrc -> proc );
libs/kfc/rsrc.c:                rsrc -> thread = KProcMgrMakeThreadState ( rsrc -> proc );
libs/kfc/win/sysctx.c:#include <kproc/procmgr.h>
libs/kfc/win/sysctx.c:    rc_t rc = KProcMgrInit ();
libs/kfc/win/sysctx.c:        KProcMgr * mgr;
libs/kfc/win/sysctx.c:        rc = KProcMgrMakeSingleton ( & mgr );
libs/kfc/win/sysctx.c:            /* create task to install into procmgr */
libs/kfc/win/sysctx.c:                    rc = KProcMgrAddCleanupTask ( mgr, & ticket, task );
libs/kfc/win/sysctx.c:            KProcMgrRelease ( mgr );
libs/kfc/win/sysrsrc.c:#include <kproc/procmgr.h>
libs/kfc/win/sysrsrc.c:            /* early whack of KProcMgr */
libs/kfc/win/sysrsrc.c:            KProcMgrRelease ( s_rsrc . proc );
libs/kfc/win/sysrsrc.c:            KProcMgrWhack ();
libs/kfc/unix/sysctx.c:#include <kproc/procmgr.h>
libs/kfc/unix/sysctx.c:    if ( KProcMgrOnMainThread () )
libs/kfc/unix/sysrsrc.c:#include <kproc/procmgr.h>
libs/kfc/unix/sysrsrc.c:            /* early whack of KProcMgr */
libs/kfc/unix/sysrsrc.c:            KProcMgrRelease ( s_rsrc . proc );
libs/kfc/unix/sysrsrc.c:            KProcMgrWhack ();
libs/kfc/tstate.c:KThreadState * KProcMgrMakeThreadState ( struct KProcMgr const * self )
libs/kfs/kfsmagic:>1	belong&0xffffff00	0x00007a00	Nvidia*
libs/kfs/kfsmagic:>1	belong&0xffffff00	0x00007a00	Nvidia*
libs/kfs/lockfile.c:#include <kproc/procmgr.h>
libs/kfs/lockfile.c:    KProcMgr *pmgr;
libs/kfs/lockfile.c:        rc = KProcMgrRemoveCleanupTask ( self -> pmgr, & self -> ticket );
libs/kfs/lockfile.c:        KProcMgrRelease ( self -> pmgr );
libs/kfs/lockfile.c:            rc = KProcMgrMakeSingleton ( & f -> pmgr );
libs/kfs/lockfile.c:                rc = KProcMgrAddCleanupTask ( f -> pmgr, & f -> ticket, f -> cleanup );
libs/kfs/lockfile.c:                KProcMgrRelease ( f -> pmgr );
libs/kfs/cacheteefile3.c:#include <kproc/procmgr.h>
libs/kfs/cacheteefile3.c:    KProcMgr * procmgr;             /* fg thread use */
libs/kfs/cacheteefile3.c:    if ( self -> procmgr != NULL )
libs/kfs/cacheteefile3.c:        KProcMgrRemoveCleanupTask ( self -> procmgr, & self -> rm_file_tkt );
libs/kfs/cacheteefile3.c:        KProcMgrRelease ( self -> procmgr );
libs/kfs/cacheteefile3.c:        KProcMgr * procmgr;
libs/kfs/cacheteefile3.c:        rc = KProcMgrMakeSingleton ( & procmgr );
libs/kfs/cacheteefile3.c:                rc = KProcMgrAddCleanupTask ( procmgr, & self -> rm_file_tkt, t );
libs/kfs/cacheteefile3.c:                    self -> procmgr = procmgr;
libs/kfs/cacheteefile3.c:                    procmgr = NULL;
libs/kfs/cacheteefile3.c:            KProcMgrRelease ( procmgr );
libs/vdb/delta_average.c:					rc = VBlobHeaderArgPushTail(hdr, max_rl_bytes);
libs/vdb/delta_average.c:					rc = VBlobHeaderArgPushTail(hdr, elem_bytes);
libs/vdb/merge.c:    VBlobHeaderArgPushTail(hdr, num_inputs);
libs/vdb/merge.c:        VBlobHeaderArgPushTail(hdr, n);
libs/vdb/merge.c:            VBlobHeaderArgPushTail(hdr, v[i]);
libs/vdb/merge.c:            VBlobHeaderArgPushTail(hdr, sz);
libs/vdb/merge.c:            VBlobHeaderArgPushTail(hdr, sz);
libs/vdb/merge.c:            VBlobHeaderArgPushTail(hdr, 0);
libs/vdb/merge.c:        rc = VBlobHeaderArgPushTail( hdr, sz );
libs/vdb/blob-headers.c:rc_t VBlobHeaderArgPushHead ( VBlobHeader *self, int64_t arg )
libs/vdb/blob-headers.c:LIB_EXPORT rc_t CC VBlobHeaderArgPushTail ( VBlobHeader *self, int64_t arg )
libs/kapp/main.c:#include <kproc/procmgr.h>
libs/kapp/main.c:    KProcMgrWhack ();
libs/kapp/main.c:    rc = KProcMgrInit ();
test/ext/mbedtls/ca-certificates.crt:4CgPukLjbo73FCeTae6RDqNfDrHrZqJyTxIThmV6PttPB/SnCWDaOkKZx7J/sxaV
test/ext/mbedtls/ca-certificates.crt:V2ccHsBqBt5ZtJot39wZhi4wNgYDVR0fBC8wLTAroCmgJ4YlaHR0cDovL2NybC5n
test/ext/mbedtls/ca-certificates.crt:GPUqUfA5hJeVbG4bwyvEdGB5JbAKJ9/fXtI5z0V9QkvfsywexcZdylU6oJxpmo/a
test/ext/mbedtls/ca-certificates.crt:Sewn6EAes6aJInKc9Q0ztFijMDvd1GpUk74aTfOTlPf8hAs/hCBcNANExdqtvArB
test/ext/mbedtls/ca-certificates.crt:IR9NmXmd4c8nnxCbHIgNsIpkQTG4DmyQJKSbXHGPurt+HBvbaoAPIbzp26a3QPSy
test/kfc/test-except.c:    rsrc -> thread = KProcMgrMakeThreadState ( ( void * ) 1 );
interfaces/kproc/procmgr.h:#ifndef _h_kproc_procmgr_
interfaces/kproc/procmgr.h:#define _h_kproc_procmgr_
interfaces/kproc/procmgr.h: * KProcMgr
interfaces/kproc/procmgr.h:typedef struct KProcMgr KProcMgr;
interfaces/kproc/procmgr.h:KPROC_EXTERN rc_t CC KProcMgrInit ( void );
interfaces/kproc/procmgr.h:KPROC_EXTERN rc_t CC KProcMgrWhack ( void );
interfaces/kproc/procmgr.h:KPROC_EXTERN rc_t CC KProcMgrMakeSingleton ( KProcMgr ** mgr );
interfaces/kproc/procmgr.h:KPROC_EXTERN rc_t CC KProcMgrAddRef ( const KProcMgr *self );
interfaces/kproc/procmgr.h:KPROC_EXTERN rc_t CC KProcMgrRelease ( const KProcMgr *self );
interfaces/kproc/procmgr.h:KPROC_EXTERN rc_t CC KProcMgrAddCleanupTask ( KProcMgr *self,
interfaces/kproc/procmgr.h:KPROC_EXTERN rc_t CC KProcMgrRemoveCleanupTask ( KProcMgr *self,
interfaces/kproc/procmgr.h:KPROC_EXTERN bool CC KProcMgrOnMainThread ( void );
interfaces/kproc/procmgr.h:KPROC_EXTERN rc_t CC KProcMgrGetPID ( const KProcMgr * self, uint32_t * pid );
interfaces/kproc/procmgr.h:KPROC_EXTERN rc_t CC KProcMgrGetHostName ( const KProcMgr * self, char * buffer, size_t buffer_size );
interfaces/kproc/procmgr.h:#endif /* _h_kproc_procmgr_ */
interfaces/kfc/tstate.h:struct KProcMgr;
interfaces/kfc/tstate.h:KThreadState * KProcMgrMakeThreadState ( struct KProcMgr const * self );
interfaces/kfc/rsrc.h:struct KProcMgr;
interfaces/kfc/rsrc.h:    struct KProcMgr     * proc;
interfaces/vdb/xform.h:/* ArgPushTail
interfaces/vdb/xform.h: *       2a. ArgPushTail ( MIN ( x ) )
interfaces/vdb/xform.h: *       4a. ArgPushTail ( bits-required )
interfaces/vdb/xform.h:VDB_EXTERN rc_t CC VBlobHeaderArgPushTail ( VBlobHeader *self, int64_t arg );

```
