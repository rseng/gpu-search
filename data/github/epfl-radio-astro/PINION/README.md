# https://github.com/epfl-radio-astro/PINION

```console
prediction/full_volume_prediction.sh:#SBATCH --constraint=gpu
prediction/full_volume_prediction.sh:# module load daint-gpu
prediction/full_volume_prediction.sh:# export CRAY_CUDA_MPS=1
prediction/subvolume_prediction.py:# prepare for cuda
prediction/subvolume_prediction.py:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
README.md:This project uses Python3 and the packages are listed in `requirements.txt`. This project was tested a GPU-equipped Linux device running Python 3.9.10.

```
