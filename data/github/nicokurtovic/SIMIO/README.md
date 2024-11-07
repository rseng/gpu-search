# https://github.com/nicokurtovic/SIMIO

```console
paper/paper.bib:        title = "{GALARIO: a GPU accelerated library for analysing radio interferometer observations}",
codes/analysis_scripts/analysisUtils.py:        self.IFProcMin = {}
codes/analysis_scripts/analysisUtils.py:        self.IFProcMax = {}
codes/analysis_scripts/analysisUtils.py:            self.IFProcMin[a] = {}
codes/analysis_scripts/analysisUtils.py:            self.IFProcMax[a] = {}
codes/analysis_scripts/analysisUtils.py:        print("Range of attenuator settings for pol %d: IFproc=%4.1f-%4.1f, IFswitch:sb1=%4.1f-%4.1f, sb2=%4.1f-%4.1f" % (pol, self.IFProcMin[antenna][pol],
codes/analysis_scripts/analysisUtils.py:                                                                              self.IFProcMax[antenna][pol],
codes/analysis_scripts/analysisUtils.py:            self.IFProcMin[antenna][pol] = -1
codes/analysis_scripts/analysisUtils.py:            self.IFProcMax[antenna][pol] = -1
codes/analysis_scripts/analysisUtils.py:            self.IFProcMin[antenna][pol] = np.min(alldB)
codes/analysis_scripts/analysisUtils.py:            self.IFProcMax[antenna][pol] = np.max(alldB)
codes/analysis_scripts/analysisUtils.py:                        startdB = self.IFProcMin[antenna][pol]-1
codes/analysis_scripts/analysisUtils.py:                        stopdB =  self.IFProcMax[antenna][pol]+1
codes/analysis_scripts/analysisUtils.py:                        xlevel = np.arange(self.IFProcMin[antenna][pol], self.IFProcMax[antenna][pol]+0.6, 0.5)

```
