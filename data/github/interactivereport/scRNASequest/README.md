# https://github.com/interactivereport/scRNAsequest

```console
scRMambient:  echo "-t: (option) the number of CPU threats cores [default GPU]"
src/cmdUtility.py:    jID = sbatch(cmds,config['output'],core,memG=memG,jID=config.get("jobID"),gpu=config.get('gpu'))
src/cmdUtility.py:      squeue(jID,config['output'],cmds,cmdN,core,memG,failedJobs,gpu=config.get('gpu'))
src/cmdUtility.py:def sbatch(cmds,strPath,core,memG=0,jID=None,gpu=False):
src/cmdUtility.py:  gpu=False if gpu is None else gpu
src/cmdUtility.py:  if gpu:
src/cmdUtility.py:    sbatchScript=sbatchScriptGPU
src/cmdUtility.py:def squeue(jID,strPath,cmds,cmdN,core,memG,failedJobs,gpu=False):
src/cmdUtility.py:  gpu=False if gpu is None else gpu
src/cmdUtility.py:    re1=sbatch(resub,strPath,core,memG,jID,gpu=gpu)
src/cmdUtility.py:sbatchScriptGPU='''#!/bin/bash
src/cmdUtility.py:#SBATCH --gres=gpu:1
src/cmdUtility.py:#SBATCH -p gpu
src/scRMambient.py:    useGPU=True
src/scRMambient.py:    useCuda="--cuda "
src/scRMambient.py:    useGPU=False
src/scRMambient.py:    useCuda=""
src/scRMambient.py:        meta[UMIcol][i],oneH5,useCuda,meta[CB_expCellNcol][i],meta[CB_dropletNcol][i],meta[CB_count][i],meta[CB_learningR][i],cpuCMD)
src/scRMambient.py:    cU.submit_cmd(cmds,{'parallel':'slurm' if useParallel else False,'output':strOut,'gpu':useGPU},core=nCore,memG=mem)

```
