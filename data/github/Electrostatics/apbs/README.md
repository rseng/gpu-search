# https://github.com/Electrostatics/apbs

```console
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.8.patch:   echo '  --without-cuda (default)  --with-cuda'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.8.patch:   echo '      (do NOT use CUDA-enabled Charm++, NAMD does not need it)'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.8.patch:   echo '  --cuda-prefix <directory containing CUDA bin, lib, and include>'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.8.patch:   set use_cuda = 0
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.8.patch: LIBS = $(CUDAOBJS) $(PLUGINLIB) $(DPMTALIBS) $(DPMELIBS) $(TCLDLL)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.8.patch:-CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF) $(EXTRAINCS)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.8.patch:+CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(APBS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF) $(EXTRAINCS)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20092904.patch: LIBS = $(CUDAOBJS) $(PLUGINLIB) $(DPMTALIBS) $(DPMELIBS) $(TCLDLL)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20092904.patch:-CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20092904.patch:+CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(APBS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20092904.patch:   echo '  --without-cuda (default)  --with-cuda'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20092904.patch:   echo '  --cuda-prefix <directory containing CUDA bin, lib, and include>'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20092904.patch:   set use_cuda = 0
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.7b2.patch:   echo '  --without-cuda (default)  --with-cuda'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.7b2.patch:   echo '      (do NOT use CUDA-enabled Charm++, NAMD does not need it)'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.7b2.patch:   echo '  --cuda-prefix <directory containing CUDA bin, lib, and include>'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.7b2.patch:   set use_cuda = 0
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.7b2.patch: LIBS = $(CUDAOBJS) $(PLUGINLIB) $(DPMTALIBS) $(DPMELIBS) $(TCLDLL)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.7b2.patch:-CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.7b2.patch:+CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(APBS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20091117.patch: LIBS = $(CUDAOBJS) $(PLUGINLIB) $(DPMTALIBS) $(DPMELIBS) $(TCLDLL)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20091117.patch:-CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20091117.patch:+CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(APBS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20091117.patch:   echo '  --without-cuda (default)  --with-cuda'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20091117.patch:   echo '      (do NOT use CUDA-enabled Charm++, NAMD does not need it)'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20091117.patch:   echo '  --cuda-prefix <directory containing CUDA bin, lib, and include>'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20091117.patch:   set use_cuda = 0
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20090727.patch: LIBS = $(CUDAOBJS) $(PLUGINLIB) $(DPMTALIBS) $(DPMELIBS) $(TCLDLL)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20090727.patch:-CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20090727.patch:+CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(APBS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20090727.patch:   echo '  --without-cuda (default)  --with-cuda'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20090727.patch:   echo '      (do NOT use CUDA-enabled Charm++, NAMD does not need it)'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20090727.patch:   echo '  --cuda-prefix <directory containing CUDA bin, lib, and include>'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20090727.patch:   set use_cuda = 0
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.10b2.patch:   echo '  --without-cuda (default)  --with-cuda'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.10b2.patch:   echo '      (do NOT use CUDA-enabled Charm++, NAMD does not need it)'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.10b2.patch: LIBS = $(CUDAOBJS) $(PLUGINLIB) $(DPMTALIBS) $(DPMELIBS) $(FMMLIBS) $(TCLDLL)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.10b2.patch:-CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(FMM) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MIC) $(MEMOPT) $(CCS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF) $(EXTRAINCS) $(MSA)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.10b2.patch:+CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(FMM) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MIC) $(MEMOPT) $(CCS) $(APBS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF) $(EXTRAINCS) $(MSA)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20120215.patch: LIBS = $(CUDAOBJS) $(PLUGINLIB) $(DPMTALIBS) $(DPMELIBS) $(TCLDLL)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20120215.patch:-CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF) $(EXTRAINCS) $(MSA)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20120215.patch:+CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(APBS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF) $(EXTRAINCS) $(MSA)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20120215.patch:   echo '  --without-cuda (default)  --with-cuda'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20120215.patch:   echo '      (do NOT use CUDA-enabled Charm++, NAMD does not need it)'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20120215.patch:   echo '  --cuda-prefix <directory containing CUDA bin, lib, and include>'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20120215.patch:   set use_cuda = 0
contrib/iapbs/modules/NAMD/patches/namd2-apbs-cvs-2013.3.18.patch:   echo '  --without-cuda (default)  --with-cuda'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-cvs-2013.3.18.patch:   echo '      (do NOT use CUDA-enabled Charm++, NAMD does not need it)'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-cvs-2013.3.18.patch:   echo '  --cuda-prefix <directory containing CUDA bin, lib, and include>'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-cvs-2013.3.18.patch:   set use_cuda = 0
contrib/iapbs/modules/NAMD/patches/namd2-apbs-cvs-2013.3.18.patch: LIBS = $(CUDAOBJS) $(PLUGINLIB) $(DPMTALIBS) $(DPMELIBS) $(TCLDLL)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-cvs-2013.3.18.patch:-CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF) $(EXTRAINCS) $(MSA)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-cvs-2013.3.18.patch:+CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(APBS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF) $(EXTRAINCS) $(MSA)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.9.patch:   echo '  --without-cuda (default)  --with-cuda'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.9.patch:   echo '      (do NOT use CUDA-enabled Charm++, NAMD does not need it)'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.9.patch:   echo '  --cuda-prefix <directory containing CUDA bin, lib, and include>'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.9.patch:   set use_cuda = 0
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.9.patch: LIBS = $(CUDAOBJS) $(PLUGINLIB) $(DPMTALIBS) $(DPMELIBS) $(TCLDLL)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.9.patch:-CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF) $(EXTRAINCS) $(MSA)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.9.patch:+CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(APBS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF) $(EXTRAINCS) $(MSA)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20110315.patch: LIBS = $(CUDAOBJS) $(PLUGINLIB) $(DPMTALIBS) $(DPMELIBS) $(TCLDLL)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20110315.patch:-CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20110315.patch:+CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MEMOPT) $(CCS) $(APBS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20110315.patch:   echo '  --without-cuda (default)  --with-cuda'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20110315.patch:   echo '      (do NOT use CUDA-enabled Charm++, NAMD does not need it)'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20110315.patch:   echo '  --cuda-prefix <directory containing CUDA bin, lib, and include>'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-20110315.patch:   set use_cuda = 0
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.11.patch:   echo '  --without-cuda (default)  --with-cuda'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.11.patch:   echo '      (do NOT use CUDA-enabled Charm++, NAMD does not need it)'
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.11.patch: LIBS = $(CUDAOBJS) $(PLUGINLIB) $(SBLIB) $(DPMTALIBS) $(DPMELIBS) $(FMMLIBS) $(TCLDLL)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.11.patch:-CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(FMM) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MIC) $(MEMOPT) $(CCS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF) $(EXTRAINCS) $(MSA) $(CKLOOP)
contrib/iapbs/modules/NAMD/patches/namd2-apbs-2.11.patch:+CXXBASEFLAGS = $(COPTI)$(CHARMINC) $(COPTI)$(SRCDIR) $(COPTI)$(INCDIR) $(DPMTA) $(DPME) $(FMM) $(COPTI)$(PLUGININCDIR) $(COPTD)STATIC_PLUGIN $(TCL) $(FFT) $(CUDA) $(MIC) $(MEMOPT) $(CCS) $(APBS) $(RELEASE) $(EXTRADEFINES) $(TRACEOBJDEF) $(EXTRAINCS) $(MSA) $(CKLOOP)
examples/pygbe/lys/lys.param:GPU         1
src/mg/vpmg.c:VPRIVATE void packAtomsOpenCL(float *ax, float *ay, float *az,
src/mg/vpmg.c:VPRIVATE void packUnpackOpenCL(int nx, int ny, int nz, int ngrid,
src/mg/vpmg.c:VPRIVATE void bcflnewOpenCL(Vpmg *thee){
src/mg/vpmg.c:    packAtomsOpenCL(ax,ay,az,charge,size,thee);
src/mg/vpmg.c:    packUnpackOpenCL(nx,ny,nz,ngrid,gx,gy,gz,val,thee,1);
src/mg/vpmg.c:    packUnpackOpenCL(nx,ny,nz,ngrid,gx,gy,gz,val,thee,0);
src/mg/vpmg.c:             * If OpenCL is available we use it, otherwise fall back to
src/mg/vpmg.c:            if (kOpenCLAvailable == 1) bcflnewOpenCL(thee);
src/mg/vpmgp.c:    if(kOpenCLAvailable)
src/main.c:        int ret = initOpenCL();
src/main.c:        printf("OpenCL runtime present - initialized = %i\n",ret);
src/main.c:        setkOpenCLAvailable_(0);
src/main.c:        printf("OpenCL is not present!\n");

```
