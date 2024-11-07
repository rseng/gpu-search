# https://github.com/Acellera/htmd

```console
htmd/membranebuilder/build_membrane.py:    equilplatform="CUDA",
htmd/membranebuilder/build_membrane.py:        The platform on which to run the minimization ('CUDA' or 'CPU')
htmd/membranebuilder/build_membrane.py:        The platform on which to run the equilibration ('CUDA' or 'CPU')
htmd/membranebuilder/simulate_openmm.py:    if plat == "CUDA":
htmd/membranebuilder/simulate_openmm.py:        prop = {"CudaPrecision": "single", "CudaDeviceIndex": str(device)}
htmd/membranebuilder/simulate_openmm.py:    equilplatform="CUDA",
htmd/share/builder/charmmfiles/str/lipid/toppar_all36_lipid_cardiolipin.str:RESI LNCCL1     -1.00 ! 3 Linoleic acid + 1 linolenic acid cardiolipin with head group charge = -1
htmd/share/builder/charmmfiles/str/lipid/toppar_all36_lipid_cardiolipin.str:RESI LNCCL2     -2.00 ! 3 linoleic acid + 1 linolenic acid cardiolipin with head group charge = -2
htmd/ui.py:from jobqueues.localqueue import LocalGPUQueue, LocalCPUQueue
htmd/util.py:    from jobqueues.localqueue import LocalGPUQueue
htmd/util.py:    md = LocalGPUQueue()
htmd/adaptive/adaptivegoal.py:    >>> ag.app = LocalGPUQueue()
htmd/adaptive/adaptivegoal.py:    >>> ag.app = LocalGPUQueue()
htmd/adaptive/adaptivegoal.py:    from jobqueues.localqueue import LocalGPUQueue
htmd/adaptive/adaptivegoal.py:    # md.app = LocalGPUQueue()
htmd/adaptive/adaptivegoal.py:    ad.app = LocalGPUQueue()
htmd/adaptive/adaptivegoal.py:    ad.app = LocalGPUQueue()
htmd/adaptive/adaptive.py:            self.app.submit(batch, nvidia_mps=len(batch) > 1)
htmd/adaptive/adaptivegoaleg.py:    >>> ag.app = LocalGPUQueue()
htmd/adaptive/adaptivegoaleg.py:    >>> ag.app = LocalGPUQueue()
htmd/adaptive/adaptivegoaleg.py:    from jobqueues.localqueue import LocalGPUQueue
htmd/adaptive/adaptivegoaleg.py:    # md.app = LocalGPUQueue()
htmd/adaptive/adaptivegoaleg.py:    ad.app = LocalGPUQueue()
htmd/adaptive/adaptiverun.py:    >>> adapt.app = LocalGPUQueue()
htmd/adaptive/adaptiverun.py:                def ngpu(self):
doc/source/userguide/adaptive-sampling-explained.rst:    app = LocalGPUQueue()
doc/source/simulation.rst:ACEMD, a powerful and simple MD engine which has pioneered GPU computing since 2009, is distributed together with HTMD

```
