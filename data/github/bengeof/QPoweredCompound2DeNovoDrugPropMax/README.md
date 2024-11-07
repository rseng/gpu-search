# https://github.com/bengeof/QPoweredCompound2DeNovoDrugPropMax

```console
Compound2DeNovoDrugPropMax-main/conda-environment.yml:  - _tflow_select=2.1.0=gpu
Compound2DeNovoDrugPropMax-main/conda-environment.yml:  - cudatoolkit=10.0.130=0
Compound2DeNovoDrugPropMax-main/conda-environment.yml:  - cudnn=7.6.0=cuda10.0_0
Compound2DeNovoDrugPropMax-main/conda-environment.yml:  - tensorflow=1.13.1=gpu_py37h83e5d6a_0
Compound2DeNovoDrugPropMax-main/conda-environment.yml:  - tensorflow-base=1.13.1=gpu_py37h871c8ca_0
Compound2DeNovoDrugPropMax-main/conda-environment.yml:  - tensorflow-gpu=1.13.1=h0d30ee6_0
Compound2DeNovoDrugPropMax-main/results/mdp/md_prod.mdp:nstlist           = 25               ; with Verlet lists the optimal nstlist is >= 10, with GPUs >= 20.
Compound2DeNovoDrugPropMax-main/results/umbrella.sh:$GMX mdrun -v -gpu_id 0 -deffnm em
Compound2DeNovoDrugPropMax-main/results/umbrella.sh:$GMX mdrun -gpu_id 0  -deffnm npt
Compound2DeNovoDrugPropMax-main/results/umbrella.sh:$GMX mdrun -gpu_id 0 -deffnm md_out 
Compound2DeNovoDrugPropMax-main/results/umbrella.sh:$GMX mdrun -deffnm pull -pf pullf.xvg -px pullx.xvg -gpu_id 0 
Compound2DeNovoDrugPropMax-main/results/umbrella.sh:$GMX mdrun -deffnm npt${ii} -gpu_id 0 
Compound2DeNovoDrugPropMax-main/results/umbrella.sh:$GMX mdrun -deffnm umbrella${ii} -gpu_id 0 
Compound2DeNovoDrugPropMax-main/results/md_prod.mdp:nstlist           = 25               ; with Verlet lists the optimal nstlist is >= 10, with GPUs >= 20.

```
