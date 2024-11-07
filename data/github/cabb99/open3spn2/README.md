# https://github.com/cabb99/open3spn2

```console
docs/source/tutorial.rst:    s.initializeMD(temperature=300 * simtk.unit.kelvin,platform_name='OpenCL')
docs/source/tutorial.rst:Please make sure that the energies obtained coincide with the energies shown here. Also you can check the energy obtained using other platforms_ by changing ``OpenCL`` to ``Reference``, ``CUDA`` or ``CPU``. 
docs/source/tutorial.rst:    platform_name='OpenCL' #'Reference','CPU','CUDA', 'OpenCL'
tests/test_cases.csv:#OpenCL
tests/test_cases.csv:Bond,B_curved,tests/bdna_curv/,traj.xyz,sim.log,E_bond,150.54,OpenCL
tests/test_cases.csv:Angle,B_curved,tests/bdna_curv/,traj.xyz,sim_harmonic.log,E_angle,150.54,OpenCL
tests/test_cases.csv:Stacking,B_curved,tests/bdna_curv/,traj.xyz,sim_stacking.log,E_angle,150.54,OpenCL
tests/test_cases.csv:Dihedral,B_curved,tests/bdna_curv/,traj.xyz,sim.log,E_dihed,150.54,OpenCL
tests/test_cases.csv:BasePair,B_curved,tests/bdna_curv/,traj.xyz,sim_stacking.log,ebp,150.54,OpenCL
tests/test_cases.csv:CrossStacking,B_curved,tests/bdna_curv/,traj.xyz,sim_stacking.log,ecstk,150.54,OpenCL
tests/test_cases.csv:Exclusion,B_curved,tests/bdna_curv/,traj.xyz,sim.log,eexcl,150.54,OpenCL
tests/test_cases.csv:Electrostatics,B_curved,tests/bdna_curv/,traj.xyz,sim.log,dna_ecou,150.54,OpenCL
open3SPN2/force/dna.py:    def __init__(self, dna, force_group=6, OpenCLPatch=True):
open3SPN2/force/dna.py:        super().__init__(dna, OpenCLPatch=OpenCLPatch)
open3SPN2/force/dna.py:    def __init__(self, dna, force_group=7, OpenCLPatch=True):
open3SPN2/force/dna.py:        super().__init__(dna, OpenCLPatch=OpenCLPatch)
open3SPN2/force/dna.py:    def __init__(self, dna, force_group=8, OpenCLPatch=True):
open3SPN2/force/dna.py:        super().__init__(dna, OpenCLPatch=OpenCLPatch)
open3SPN2/force/dna.py:    def __init__(self, dna, force_group=9, OpenCLPatch=True):
open3SPN2/force/dna.py:        super().__init__(dna, OpenCLPatch=OpenCLPatch)
open3SPN2/force/dna.py:    def __init__(self, dna, force_group=10, OpenCLPatch=True):
open3SPN2/force/dna.py:        super().__init__(dna, OpenCLPatch=OpenCLPatch)
open3SPN2/force/dna.py:                # since the maximum number of exclusions in OpenCL is 4.
open3SPN2/force/dna.py:    def __init__(self, dna, force_group=11, OpenCLPatch=True):
open3SPN2/force/dna.py:        super().__init__(dna, OpenCLPatch=OpenCLPatch)
open3SPN2/force/dna.py:                    # since the maximum number of exclusions in OpenCL is 4.
open3SPN2/force/dna.py:                    maxn = 6 if self.OpenCLPatch else 9
open3SPN2/force/dna.py:                            (not self.OpenCLPatch and i > j):
open3SPN2/force/dna.py:def addNonBondedExclusions(dna, force, OpenCLPatch=True):
open3SPN2/force/dna.py:    if OpenCLPatch:
open3SPN2/force/dna.py:    def __init__(self, dna, force_group = 12, OpenCLPatch=True):
open3SPN2/force/dna.py:        super().__init__(dna, OpenCLPatch=OpenCLPatch)
open3SPN2/force/dna.py:    def __init__(self, dna, force_group=13, temperature=300*unit.kelvin, salt_concentration=100*unit.millimolar, OpenCLPatch=True):
open3SPN2/force/dna.py:        super().__init__(dna, OpenCLPatch=OpenCLPatch)
open3SPN2/force/dna.py:        addNonBondedExclusions(self.dna, self.force, self.OpenCLPatch)
open3SPN2/force/template.py:    def __init__(self, dna, OpenCLPatch=True):
open3SPN2/force/template.py:        # The patch allows the crosstacking force to run in OpenCL
open3SPN2/force/template.py:        self.OpenCLPatch = OpenCLPatch
examples/Protein_DNA/Protein_DNA_example.py:platform_name='OpenCL' #'Reference','CPU','CUDA', 'OpenCL'
examples/DNA_analysis.py:platform_name='OpenCL' #'Reference','CPU','CUDA', 'OpenCL'

```
