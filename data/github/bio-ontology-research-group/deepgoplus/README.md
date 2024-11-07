# https://github.com/bio-ontology-research-group/deepgoplus

```console
deepgoplus.py:# config.gpu_options.allow_growth = True
deepgoplus.py:    '--device', '-d', default='gpu:0',
update.sh:export LD_LIBRARY_PATH=/usr/lib/cuda/include:/usr/lib/cuda/lib64:$LD_LIBRARY_PATH
README.md:Current version of Tensorflow will require Cuda 10.1 and Cudnn 7.6.5
requirements.txt:tensorflow-gpu==2.3.1   # Released: Sep 24, 2020
jobTrain:#SBATCH --gres=gpu:4
jobTrain:#SBATCH --constraint=[gpu]
run_deepgoplus.sh:	python deepgoplus.py -ld -bs 32 -d gpu:1 -pi $i >> job_${i}.out 2>&1

```
