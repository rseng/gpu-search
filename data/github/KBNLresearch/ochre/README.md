# https://github.com/KBNLresearch/ochre

```console
job.sh:#SBATCH --gres=gpu:1
job.sh:module load cuda80/toolkit/8.0.61
job.sh:module load tensorflow/python3.x/gpu/r1.4.0-py3
job.sh:module load cuDNN/cuda80/5.1.5
nematus.sh:#SBATCH --gres=gpu:1
transformer.sh:#SBATCH --gres=gpu:1
nematus_translate.sh:#SBATCH --gres=gpu:1
nematusA8P3.sh:#SBATCH --gres=gpu:1
nematusA8P2.sh:#SBATCH --gres=gpu:1

```
