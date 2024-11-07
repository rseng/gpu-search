# https://github.com/hpparvi/MuSCAT2_transit_pipeline

```console
doc/analysis_manual.md:### OpenCL
doc/analysis_manual.md:The pipeline can use PyTransit's OpenCL transit model for transit modelling, which can significantly accelerate the 
doc/analysis_manual.md:analysis. This can be done by initialising `TransitAnalysis` with the `with_opencl` argument set to `True`.
doc/analysis_manual.md:    ta = TransitAnalysis(target, night, tid=TID, cids=CIDS, with_opencl=True)
muscat2ta/m2lpf.py:                 filters: tuple, aperture_lims: tuple = (0, inf), use_opencl: bool = False,
muscat2ta/m2lpf.py:        self.use_opencl = use_opencl
muscat2ta/m2lpf.py:        if use_opencl:
muscat2ta/m2lpf.py:            import pyopencl as cl
muscat2ta/transitanalysis.py:                 use_opencl: bool = False, with_transit: bool = True, with_contamination: bool = False,
muscat2ta/transitanalysis.py:        self.use_opencl = use_opencl
muscat2ta/transitanalysis.py:        self.lpf = M2LPF(target, self.phs, tid, cids, pbs, aperture_lims=aperture_lims, use_opencl=use_opencl,
muscat2ta/tfopanalysis.py:                 use_opencl: bool = False, with_transit: bool = True, with_contamination: bool = False,
muscat2ta/tfopanalysis.py:                 use_opencl=use_opencl, with_transit=with_transit, with_contamination=with_contamination,

```
