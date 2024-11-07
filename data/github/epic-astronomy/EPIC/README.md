# https://github.com/epic-astronomy/EPIC

```console
epic/examples/install_notes/ASU_bifrost_install:export PATH=/usr/local/cuda-9.1/bin:$PATH
epic/examples/install_notes/ASU_bifrost_install:export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:$LD_LIBRARY_PATH
epic/examples/install_notes/ASU_bifrost_install:GPU_ARCHS     ?= 50 # Nap time!
epic/examples/install_notes/ASU_bifrost_install:#GPU_ARCHS     ?= 35 52
epic/examples/install_notes/ASU_bifrost_install:#GPU_ARCHS     ?= 52
epic/examples/install_notes/ASU_bifrost_install:CUDA_HOME     ?= /usr/local/cuda-9.1
epic/examples/install_notes/ASU_bifrost_install:CUDA_LIBDIR   ?= $(CUDA_HOME)/lib
epic/examples/install_notes/ASU_bifrost_install:CUDA_LIBDIR64 ?= $(CUDA_HOME)/lib64
epic/examples/install_notes/ASU_bifrost_install:CUDA_INCDIR   ?= $(CUDA_HOME)/include
epic/examples/install_notes/ASU_bifrost_install:#NOCUDA     = 1 # Disable CUDA support
epic/examples/install_notes/ASU_bifrost_install:#CUDA_DEBUG = 1 # Enable CUDA debugging (nvcc -G)
LWA/LWA_bifrost_alt_ordering.py:from bifrost.device import set_device as BFSetGPU, get_device as BFGetGPU, set_devices_no_spin_cpu as BFNoSpinZone
LWA/LWA_bifrost_alt_ordering.py:    def __init__(self, log, iring, oring, ntime_gulp=2500, nchan_out=1, profile=False, core=-1, gpu=-1):
LWA/LWA_bifrost_alt_ordering.py:        self.gpu = gpu
LWA/LWA_bifrost_alt_ordering.py:        if self.gpu != -1:
LWA/LWA_bifrost_alt_ordering.py:            BFSetGPU(self.gpu)
LWA/LWA_bifrost_alt_ordering.py:                                  'ngpu': 1,
LWA/LWA_bifrost_alt_ordering.py:                                  'gpu0': BFGetGPU(),})
LWA/LWA_bifrost_alt_ordering.py:                                odata[...] = qdata.copy(space='cuda_host').view(numpy.int8).reshape(oshape)
LWA/LWA_bifrost_alt_ordering.py:                        odata[...] = qdata.copy(space='cuda_host').view(numpy.int8).reshape(oshape)
LWA/LWA_bifrost_alt_ordering.py:                 ntime_gulp=2500, accumulation_time=10000, core=-1, gpu=-1, 
LWA/LWA_bifrost_alt_ordering.py:        self.gpu = gpu
LWA/LWA_bifrost_alt_ordering.py:        if self.gpu != -1:
LWA/LWA_bifrost_alt_ordering.py:            BFSetGPU(self.gpu)
LWA/LWA_bifrost_alt_ordering.py:                                  'ngpu': 1,
LWA/LWA_bifrost_alt_ordering.py:                                  'gpu0': BFGetGPU(),})
LWA/LWA_bifrost_alt_ordering.py:                    self.locs = bifrost.ndarray(locs.astype(numpy.int32), space='cuda')
LWA/LWA_bifrost_alt_ordering.py:                    gphases = phases.copy(space='cuda')
LWA/LWA_bifrost_alt_ordering.py:                            tdata = tdata.copy(space='cuda')
LWA/LWA_bifrost_alt_ordering.py:                            #    udata = bifrost.ndarray(shape=tdata.shape, dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost_alt_ordering.py:                                gdata = bifrost.zeros(shape=(self.ntime_gulp,nchan,npol,self.grid_size,self.grid_size),dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost_alt_ordering.py:                                                             dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost_alt_ordering.py:                                                                      dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost_alt_ordering.py:                                    autocorrs = bifrost.ndarray(shape=(self.ntime_gulp,nchan,nstand,npol**2),dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost_alt_ordering.py:                                    autocorrs_av = bifrost.zeros(shape=(1,nchan,nstand,npol**2), dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost_alt_ordering.py:                                    autocorr_g = bifrost.zeros(shape=(1,nchan,npol**2,self.grid_size,self.grid_size), dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost_alt_ordering.py:                                    autocorr_lo = bifrost.ndarray(numpy.ones(shape=(3,1,nchan,nstand,npol**2),dtype=numpy.int32)*self.grid_size/2,space='cuda')
LWA/LWA_bifrost_alt_ordering.py:                                    autocorr_il = bifrost.ndarray(numpy.ones(shape=(1,nchan,nstand,npol**2,self.ant_extent,self.ant_extent),dtype=numpy.complex64),space='cuda')
LWA/LWA_bifrost_alt_ordering.py:                                print("-> GPU Time Taken: %f"%(time2-time1))
LWA/LWA_bifrost_alt_ordering.py:                                print("-> Average GPU Time Taken: %f (%i samples)" % (1.0*sum(runtime_history)/len(runtime_history), len(runtime_history)))
LWA/LWA_bifrost_alt_ordering.py:                 core=-1, gpu=-1, *args, **kwargs):
LWA/LWA_bifrost_alt_ordering.py:        self.gpu = gpu
LWA/LWA_bifrost_alt_ordering.py:        if self.gpu != -1:
LWA/LWA_bifrost_alt_ordering.py:            BFSetGPU(self.gpu)
LWA/LWA_bifrost_alt_ordering.py:                                  'ngpu': 1,
LWA/LWA_bifrost_alt_ordering.py:                                  'gpu0': BFGetGPU(),})
LWA/LWA_bifrost_alt_ordering.py:                itemp = idata.copy(space='cuda_host')
LWA/LWA_bifrost_alt_ordering.py:    def __init__(self, log, iring, filename, core=-1, gpu=-1, cpu=False,
LWA/LWA_bifrost_alt_ordering.py:        self.gpu = gpu
LWA/LWA_bifrost_alt_ordering.py:        if self.gpu != -1:
LWA/LWA_bifrost_alt_ordering.py:            BFSetGPU(self.gpu)
LWA/LWA_bifrost_alt_ordering.py:                                  'ngpu': 1,
LWA/LWA_bifrost_alt_ordering.py:                                  'gpu0': BFGetGPU(),})
LWA/LWA_bifrost_alt_ordering.py:                itemp = idata.copy(space='cuda_host')
LWA/LWA_bifrost_alt_ordering.py:    # Setup the cores and GPUs to use
LWA/LWA_bifrost_alt_ordering.py:    gpus  = [0, 0, 0, 0, 0, 0, 0]
LWA/LWA_bifrost_alt_ordering.py:    fcapture_ring = Ring(name="capture",space="cuda_host")
LWA/LWA_bifrost_alt_ordering.py:    fdomain_ring = Ring(name="fengine", space="cuda_host")
LWA/LWA_bifrost_alt_ordering.py:    transpose_ring = Ring(name="transpose", space="cuda_host")
LWA/LWA_bifrost_alt_ordering.py:    gridandfft_ring = Ring(name="gridandfft", space="cuda")
LWA/LWA_bifrost_alt_ordering.py:                                 nchan_out=args.channels, core=cores.pop(0), gpu=gpus.pop(0),
LWA/LWA_bifrost_alt_ordering.py:                                core=cores.pop(0), gpu=gpus.pop(0),benchmark=args.benchmark,
LWA/LWA_bifrost_alt_ordering.py:        ops.append(TriggerOp(log, gridandfft_ring, core=cores.pop(0), gpu=gpus.pop(0), 
LWA/LWA_bifrost_alt_ordering.py:                         core=cores.pop(0), gpu=gpus.pop(0), cpu=False,
LWA/LWA_bifrost_DFT.py:from bifrost.device import set_device as BFSetGPU, get_device as BFGetGPU, set_devices_no_spin_cpu as BFNoSpinZone
LWA/LWA_bifrost_DFT.py:    def __init__(self, log, iring, oring, ntime_gulp=2500, nchan_out=1, profile=False, core=-1, gpu=-1):
LWA/LWA_bifrost_DFT.py:        self.gpu = gpu
LWA/LWA_bifrost_DFT.py:        if self.gpu != -1:
LWA/LWA_bifrost_DFT.py:            BFSetGPU(self.gpu)
LWA/LWA_bifrost_DFT.py:                                  'ngpu': 1,
LWA/LWA_bifrost_DFT.py:                                  'gpu0': BFGetGPU(),})
LWA/LWA_bifrost_DFT.py:                                odata[...] = qdata.copy(space='cuda_host').view(numpy.int8).reshape(oshape)
LWA/LWA_bifrost_DFT.py:                        odata[...] = qdata.copy(space='cuda_host').view(numpy.int8).reshape(oshape)
LWA/LWA_bifrost_DFT.py:                 ntime_gulp=2500, accumulation_time=10000, core=-1, gpu=-1, 
LWA/LWA_bifrost_DFT.py:        self.gpu = gpu
LWA/LWA_bifrost_DFT.py:        if self.gpu != -1:
LWA/LWA_bifrost_DFT.py:            BFSetGPU(self.gpu)
LWA/LWA_bifrost_DFT.py:                                  'ngpu': 1,
LWA/LWA_bifrost_DFT.py:                                  'gpu0': BFGetGPU(),})
LWA/LWA_bifrost_DFT.py:                    self.locs = bifrost.ndarray(locs.astype(numpy.int32), space='cuda')
LWA/LWA_bifrost_DFT.py:                    gphases = phases.copy(space='cuda')
LWA/LWA_bifrost_DFT.py:                            tdata = tdata.copy(space='cuda')
LWA/LWA_bifrost_DFT.py:                                udata = bifrost.ndarray(shape=tdata.shape, dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost_DFT.py:                                gdata = bifrost.zeros(shape=(self.ntime_gulp,nchan,npol,self.grid_size,self.grid_size),dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost_DFT.py:                                                             dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost_DFT.py:                                                                      dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost_DFT.py:                                    autocorrs = bifrost.ndarray(shape=(self.ntime_gulp,nchan,npol**2,nstand),dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost_DFT.py:                                    autocorrs_av = bifrost.zeros(shape=(1,nchan,npol**2,nstand), dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost_DFT.py:                                    autocorr_g = bifrost.zeros(shape=(1,nchan,npol**2,self.grid_size,self.grid_size), dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost_DFT.py:                                    autocorr_lo = bifrost.ndarray(numpy.ones(shape=(3,1,nchan,npol**2,nstand),dtype=numpy.int32)*self.grid_size/2,space='cuda')
LWA/LWA_bifrost_DFT.py:                                    autocorr_il = bifrost.ndarray(numpy.ones(shape=(1,nchan,npol**2,nstand,self.ant_extent,self.ant_extent),dtype=numpy.complex64),space='cuda')
LWA/LWA_bifrost_DFT.py:                                print("-> GPU Time Taken: %f"%(time2-time1))
LWA/LWA_bifrost_DFT.py:                                print("-> Average GPU Time Taken: %f (%i samples)" % (1.0*sum(runtime_history)/len(runtime_history), len(runtime_history)))
LWA/LWA_bifrost_DFT.py:                 ntime_gulp=2500, accumulation_time=10000, core=-1, gpu=-1, 
LWA/LWA_bifrost_DFT.py:        self.gpu = gpu
LWA/LWA_bifrost_DFT.py:        if self.gpu != -1:
LWA/LWA_bifrost_DFT.py:            BFSetGPU(self.gpu)
LWA/LWA_bifrost_DFT.py:                                  'ngpu': 1,
LWA/LWA_bifrost_DFT.py:                                  'gpu0': BFGetGPU(),})
LWA/LWA_bifrost_DFT.py:                    self.locs = bifrost.ndarray(locs.astype(numpy.int32), space='cuda')
LWA/LWA_bifrost_DFT.py:                dftm_cu = self.dftm.copy(space='cuda')
LWA/LWA_bifrost_DFT.py:                            tdata = tdata.copy(space='cuda')
LWA/LWA_bifrost_DFT.py:                                udata = bifrost.ndarray(shape=tdata.shape, dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost_DFT.py:                                gdata = bifrost.zeros(shape=(nchan*npol,self.skymodes,self.ntime_gulp),dtype=numpy.complex64,space='cuda')
LWA/LWA_bifrost_DFT.py:                                    gdatas = bifrost.zeros(shape=(nchan,npol**2,self.skymodes,self.ntime_gulp),dtype=numpy.complex64,space='cuda')
LWA/LWA_bifrost_DFT.py:                                    accumulated_image = bifrost.zeros(shape=(nchan,npol**2,self.skymodes,1),dtype=numpy.complex64,space='cuda')
LWA/LWA_bifrost_DFT.py:                                print("-> GPU Time Taken: %f"%(time2-time1))
LWA/LWA_bifrost_DFT.py:                                print("-> Average GPU Time Taken: %f (%i samples)" % (1.0*sum(runtime_history)/len(runtime_history), len(runtime_history)))
LWA/LWA_bifrost_DFT.py:                 core=-1, gpu=-1, *args, **kwargs):
LWA/LWA_bifrost_DFT.py:        self.gpu = gpu
LWA/LWA_bifrost_DFT.py:        if self.gpu != -1:
LWA/LWA_bifrost_DFT.py:            BFSetGPU(self.gpu)
LWA/LWA_bifrost_DFT.py:                                  'ngpu': 1,
LWA/LWA_bifrost_DFT.py:                                  'gpu0': BFGetGPU(),})
LWA/LWA_bifrost_DFT.py:                itemp = idata.copy(space='cuda_host')
LWA/LWA_bifrost_DFT.py:    def __init__(self, log, iring, filename, core=-1, gpu=-1, cpu=False,
LWA/LWA_bifrost_DFT.py:        self.gpu = gpu
LWA/LWA_bifrost_DFT.py:        if self.gpu != -1:
LWA/LWA_bifrost_DFT.py:            BFSetGPU(self.gpu)
LWA/LWA_bifrost_DFT.py:                                  'ngpu': 1,
LWA/LWA_bifrost_DFT.py:                                  'gpu0': BFGetGPU(),})
LWA/LWA_bifrost_DFT.py:                itemp = idata.copy(space='cuda_host')
LWA/LWA_bifrost_DFT.py:    def __init__(self, log, iring, filename, core=-1, gpu=-1, cpu=False,
LWA/LWA_bifrost_DFT.py:        self.gpu = gpu
LWA/LWA_bifrost_DFT.py:        if self.gpu != -1:
LWA/LWA_bifrost_DFT.py:            BFSetGPU(self.gpu)
LWA/LWA_bifrost_DFT.py:                                  'ngpu': 1,
LWA/LWA_bifrost_DFT.py:                                  'gpu0': BFGetGPU(),})
LWA/LWA_bifrost_DFT.py:                itemp = idata.copy(space='cuda_host')
LWA/LWA_bifrost_DFT.py:    # Setup the cores and GPUs to use
LWA/LWA_bifrost_DFT.py:    gpus  = [0, 0, 0, 0, 0, 0, 0]
LWA/LWA_bifrost_DFT.py:    fcapture_ring = Ring(name="capture",space="cuda_host")
LWA/LWA_bifrost_DFT.py:    fdomain_ring = Ring(name="fengine", space="cuda_host")
LWA/LWA_bifrost_DFT.py:    transpose_ring = Ring(name="transpose", space="cuda_host")
LWA/LWA_bifrost_DFT.py:    gridandfft_ring = Ring(name="gridandfft", space="cuda")
LWA/LWA_bifrost_DFT.py:                                 nchan_out=args.channels, core=cores.pop(0), gpu=gpus.pop(0),
LWA/LWA_bifrost_DFT.py:                                         core=cores.pop(0), gpu=gpus.pop(0), benchmark=args.benchmark,
LWA/LWA_bifrost_DFT.py:                                    core=cores.pop(0), gpu=gpus.pop(0),benchmark=args.benchmark,
LWA/LWA_bifrost_DFT.py:        ops.append(TriggerOp(log, gridandfft_ring, core=cores.pop(0), gpu=gpus.pop(0), 
LWA/LWA_bifrost_DFT.py:                               core=cores.pop(0), gpu=gpus.pop(0), cpu=False,
LWA/LWA_bifrost_DFT.py:                          core=cores.pop(0), gpu=gpus.pop(0), cpu=False,
LWA/LWA_bifrost.py:from bifrost.device import set_device as BFSetGPU, get_device as BFGetGPU, set_devices_no_spin_cpu as BFNoSpinZone
LWA/LWA_bifrost.py:    def __init__(self, log, iring, oring, ntime_gulp=2500, nchan_out=1, profile=False, core=-1, gpu=-1):
LWA/LWA_bifrost.py:        self.gpu = gpu
LWA/LWA_bifrost.py:        if self.gpu != -1:
LWA/LWA_bifrost.py:            BFSetGPU(self.gpu)
LWA/LWA_bifrost.py:                                  'ngpu': 1,
LWA/LWA_bifrost.py:                                  'gpu0': BFGetGPU(),})
LWA/LWA_bifrost.py:                                odata[...] = qdata.copy(space='cuda_host').view(numpy.int8).reshape(oshape)
LWA/LWA_bifrost.py:                        odata[...] = qdata.copy(space='cuda_host').view(numpy.int8).reshape(oshape)
LWA/LWA_bifrost.py:                 ntime_gulp=2500, accumulation_time=10000, core=-1, gpu=-1, 
LWA/LWA_bifrost.py:        self.gpu = gpu
LWA/LWA_bifrost.py:        if self.gpu != -1:
LWA/LWA_bifrost.py:            BFSetGPU(self.gpu)
LWA/LWA_bifrost.py:                                  'ngpu': 1,
LWA/LWA_bifrost.py:                                  'gpu0': BFGetGPU(),})
LWA/LWA_bifrost.py:                    self.locs = bifrost.ndarray(locs.astype(numpy.int32), space='cuda')
LWA/LWA_bifrost.py:                    gphases = phases.copy(space='cuda')
LWA/LWA_bifrost.py:                            tdata = tdata.copy(space='cuda')
LWA/LWA_bifrost.py:                                udata = bifrost.ndarray(shape=tdata.shape, dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost.py:                                gdata = bifrost.zeros(shape=(self.ntime_gulp,nchan,npol,self.grid_size,self.grid_size),dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost.py:                                                             dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost.py:                                                                      dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost.py:                                    autocorrs = bifrost.ndarray(shape=(self.ntime_gulp,nchan,npol**2,nstand),dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost.py:                                    autocorrs_av = bifrost.zeros(shape=(1,nchan,npol**2,nstand), dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost.py:                                    autocorr_g = bifrost.zeros(shape=(1,nchan,npol**2,self.grid_size,self.grid_size), dtype=numpy.complex64, space='cuda')
LWA/LWA_bifrost.py:                                    autocorr_lo = bifrost.ndarray(numpy.ones(shape=(3,1,nchan,npol**2,nstand),dtype=numpy.int32)*self.grid_size/2,space='cuda')
LWA/LWA_bifrost.py:                                    autocorr_il = bifrost.ndarray(numpy.ones(shape=(1,nchan,npol**2,nstand,self.ant_extent,self.ant_extent),dtype=numpy.complex64),space='cuda')
LWA/LWA_bifrost.py:                                print("-> GPU Time Taken: %f"%(time2-time1))
LWA/LWA_bifrost.py:                                print("-> Average GPU Time Taken: %f (%i samples)" % (1.0*sum(runtime_history)/len(runtime_history), len(runtime_history)))
LWA/LWA_bifrost.py:                 core=-1, gpu=-1, *args, **kwargs):
LWA/LWA_bifrost.py:        self.gpu = gpu
LWA/LWA_bifrost.py:        if self.gpu != -1:
LWA/LWA_bifrost.py:            BFSetGPU(self.gpu)
LWA/LWA_bifrost.py:                                  'ngpu': 1,
LWA/LWA_bifrost.py:                                  'gpu0': BFGetGPU(),})
LWA/LWA_bifrost.py:                itemp = idata.copy(space='cuda_host')
LWA/LWA_bifrost.py:    def __init__(self, log, iring, filename, core=-1, gpu=-1, cpu=False,
LWA/LWA_bifrost.py:        self.gpu = gpu
LWA/LWA_bifrost.py:        if self.gpu != -1:
LWA/LWA_bifrost.py:            BFSetGPU(self.gpu)
LWA/LWA_bifrost.py:                                  'ngpu': 1,
LWA/LWA_bifrost.py:                                  'gpu0': BFGetGPU(),})
LWA/LWA_bifrost.py:                itemp = idata.copy(space='cuda_host')
LWA/LWA_bifrost.py:    # Setup the cores and GPUs to use
LWA/LWA_bifrost.py:    gpus  = [0, 0, 0, 0, 0, 0, 0]
LWA/LWA_bifrost.py:    fcapture_ring = Ring(name="capture",space="cuda_host")
LWA/LWA_bifrost.py:    fdomain_ring = Ring(name="fengine", space="cuda_host")
LWA/LWA_bifrost.py:    transpose_ring = Ring(name="transpose", space="cuda_host")
LWA/LWA_bifrost.py:    gridandfft_ring = Ring(name="gridandfft", space="cuda")
LWA/LWA_bifrost.py:                                 nchan_out=args.channels, core=cores.pop(0), gpu=gpus.pop(0),
LWA/LWA_bifrost.py:                                core=cores.pop(0), gpu=gpus.pop(0),benchmark=args.benchmark,
LWA/LWA_bifrost.py:        ops.append(TriggerOp(log, gridandfft_ring, core=cores.pop(0), gpu=gpus.pop(0), 
LWA/LWA_bifrost.py:                         core=cores.pop(0), gpu=gpus.pop(0), cpu=False,

```
