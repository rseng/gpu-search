# https://github.com/imcgreer/simqso

```console
simqso/sqrun.py:def buildSpectraBySightLine(wave,qsoGrid,procMap=map,
simqso/sqrun.py:    specOut = list(procMap(build_grp_spec,list(qsoGroups)))
simqso/sqrun.py:def buildSpectraBulk(wave,qsoGrid,procMap=map,
simqso/sqrun.py:        specOut = list(procMap(build_one_spec,qsoGrid))
simqso/sqrun.py:        procMap = pool.map
simqso/sqrun.py:        procMap = map
simqso/sqrun.py:    _,spectra = buildSpec(wave,qsoGrid,procMap,
sdss/ebosscore.py:           maxIter=3,procMap=map,wave=None,
sdss/ebosscore.py:                                             procMap=procMap,
sdss/ebosscore.py:def apply_selection_fun(fileName,verbose=0,redo=False,procMap=None,nsplit=1):
sdss/ebosscore.py:            procMap(run_xdqso,procArgs)
sdss/ebosscore.py:        procMap = map
sdss/ebosscore.py:        procMap = pool.map
sdss/ebosscore.py:                      procMap=procMap,outputDir=args.outputdir)
sdss/ebosscore.py:               procMap=procMap,outputDir=args.outputdir)
sdss/ebosscore.py:            apply_selection_fun(fn,verbose=1,procMap=procMap,redo=True,

```
